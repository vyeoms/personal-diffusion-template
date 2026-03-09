# Minimal DDPM implementation (x0-prediction). Created with Claude Opus 4.6

# The backbone is injected as a parameter — it must satisfy:
#     backbone(x_t, t) -> predicted_x0
# where x_t is (B, C, H, W) and t is (B,) float timesteps in [0, T-1].


import torch
import torch.nn.functional as F

from utils.misc_utils import append_dims # Utility function for appending dimensions to tensors.

class DDPM(torch.nn.Module):

    def __init__(
        self,
        backbone: torch.nn.Module,
        num_steps: int = 1000,
        schedule: str = "linear",       # "linear", "cosine", "sqrt"
        beta_start: float = 1e-5,
        beta_end: float = 1e-1,
        cosine_s: float = 8e-3,         # offset for cosine schedule
    ):
        super().__init__()
        self.backbone = backbone
        self.T = num_steps

        # ---- Build beta schedule ----
        if schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, num_steps, dtype=torch.float64)
        elif schedule == "cosine":
            # Nichol & Dhariwal (2021) — "Improved Denoising Diffusion Probabilistic Models"
            steps = torch.arange(num_steps + 1, dtype=torch.float64)
            f = torch.cos((steps / num_steps + cosine_s) / (1 + cosine_s) * torch.pi / 2) ** 2
            alpha_bar = f / f[0]
            betas = (1 - alpha_bar[1:] / alpha_bar[:-1]).clamp(max=beta_end)

        elif schedule == "sqrt":
            # Linear in sqrt(beta) — sometimes called "quadratic"
            betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_steps, dtype=torch.float64) ** 2

        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        # ---- Precompute everything we need as buffers ----
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        alpha_bar_prev = F.pad(alpha_bar[:-1], (1, 0), value=1.0)

        # For q(x_t | x_0)
        self.register_buffer("sqrt_alpha_bar", alpha_bar.sqrt().float())
        self.register_buffer("sqrt_one_minus_alpha_bar", (1.0 - alpha_bar).sqrt().float())

        # For DDPM posterior q(x_{t-1} | x_t, x_0)
        self.register_buffer("betas", betas.float())
        self.register_buffer("alphas", alphas.float())
        self.register_buffer("alpha_bar", alpha_bar.float())
        self.register_buffer("alpha_bar_prev", alpha_bar_prev.float())
        self.register_buffer("sqrt_recip_alphas", (1.0 / alphas).sqrt().float())
        posterior_variance = betas * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)
        self.register_buffer("posterior_variance", posterior_variance.float())

    # ------------------------------------------------------------------
    # Forward diffusion: q(x_t | x_0) = N(sqrt(a_bar) * x_0, (1 - a_bar) * I)
    # ------------------------------------------------------------------
    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None):
        if noise is None:
            noise = torch.randn_like(x_0)
        return (
            append_dims(self.sqrt_alpha_bar[t], x_0.ndim) * x_0
            + append_dims(self.sqrt_one_minus_alpha_bar[t], x_0.ndim) * noise,
            noise,
        )
    
    def forward(self, x_t: torch.Tensor, t: torch.Tensor, labels=None):
        return self.backbone(x_t, t.float(), labels)

    # ------------------------------------------------------------------
    # Derive eps from (x_t, x0_hat)
    # ------------------------------------------------------------------
    def _predict_eps(self, x_t, t, x0_hat):
        return (
            x_t - append_dims(self.sqrt_alpha_bar[t], x_t.ndim) * x0_hat
        ) / append_dims(self.sqrt_one_minus_alpha_bar[t], x_t.ndim)

# ------------------------------------------------------------------
# Training loss: min-SNR-γ weighted x0-prediction MSE.
#
# Plain uniform-t MSE lets high-noise steps dominate training: at those
# steps x0-prediction reduces to "guess the dataset mean", contributing
# large but uninformative gradients that swamp the low-noise regime where
# perceptual quality is determined.  Min-SNR-γ (Hang et al. 2023) clips
# each timestep's effective weight at γ so all noise levels contribute
# comparably.  γ=5 is the recommended default.
# ------------------------------------------------------------------
class DDPMLoss:
    def __init__(self, gamma: float = 5.0):
        self.gamma = gamma

    def __call__(self, diffusion_model, x_0: torch.Tensor) -> torch.Tensor:
        t = torch.randint(0, diffusion_model.T, (x_0.shape[0],), device=x_0.device)
        x_t, _ = diffusion_model.q_sample(x_0, t)
        x0_hat = diffusion_model(x_t, t.float())

        # SNR(t) = ᾱ_t / (1 − ᾱ_t)
        snr = diffusion_model.alpha_bar[t] / (1.0 - diffusion_model.alpha_bar[t])
        weight = append_dims(snr.clamp(max=self.gamma) / self.gamma, x_0.ndim)

        return (weight * (x0_hat - x_0) ** 2).mean()
