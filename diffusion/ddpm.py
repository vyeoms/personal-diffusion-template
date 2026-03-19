# Minimal DDPM implementation (x0-prediction). Created with Claude Opus 4.6

# The backbone is injected as a parameter — it must satisfy:
#     backbone(x_t, t) -> predicted_x0
# where x_t is (B, C, H, W) and t is (B,) float timesteps in [0, T-1].

import numpy as np
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

def evaluate_log_likelihood(
        model: torch.nn.Module,
        samples: torch.Tensor,
        **net_fwd_kwargs
    ):
    """
    Implements the ODE from DDIM Eq (14): d(x_bar) = epsilon * d(sigma).
    Evaluates log-likelihood by integrating from Data (t=0) -> Noise (t=T).
    
    x_bar = x / sqrt(alpha)
    sigma = sqrt(1-alpha) / sqrt(alpha)
    """
    device = samples.device
    batch_size = samples.shape[0]
    data_dim = np.prod(samples.shape[1:])
    
    # 1. Prepare Schedule: alpha_bar
    # alpha_bar usually goes from ~1.0 (data) to ~0.0 (noise)
    alphas = model.alpha_bar.to(device)
    
    # Determine integration steps (Forward: Data -> Noise)
    # We iterate 0 -> T-1
    timesteps = range(0, alphas.shape[0] - 1)

    nll_over_steps = []
    x0_std_steps = []
    eps_std_steps = []
    x0_mean_steps = []
    eps_mean_steps = []
    dt_steps = []

    # 2. Initialize x_bar (Equation 13/14 context)
    # At t=0, alpha=1.0 (approx), so sigma=0. 
    # Therefore x_bar_0 = x_data / 1.0 = x_data
    with torch.enable_grad():
        x_bar = samples.clone().detach().requires_grad_(True)
        
        # Accumulator for Log Likelihood (in Bits Per Dimension)
        # We initialize to 0 and add the divergence terms + final prior term
        ll_accumulator_bpd = torch.zeros(batch_size, device=device)
        sigma_over_steps = []
        sigma_next_steps = []
        
        for i in timesteps:
            # Get current and next sigma
            # We assume alphas[i] corresponds to step i
            alpha_t = alphas[i]
            alpha_next = alphas[i+1]
            
            sigma_t = torch.sqrt(1 - alpha_t) / torch.sqrt(alpha_t)
            sigma_next = torch.sqrt(1 - alpha_next) / torch.sqrt(alpha_next)
            sigma_over_steps.append(sigma_t.cpu())
            sigma_next_steps.append(sigma_next.cpu())
            
            # d_sigma (Positive because we are encoding: sigma increases)
            dt = sigma_next - sigma_t
            dt_steps.append(dt.cpu())
            
            # Prepare Model Input
            # From Image Eq (14): input is x_bar / sqrt(sigma^2 + 1)
            # This mathematically simplifies to x_original (the pixel space)
            scale_factor = torch.sqrt(sigma_t**2 + 1)
            model_input = x_bar / scale_factor
            
            t_tensor = torch.full((batch_size,), i, device=device, dtype=torch.long)
            
            # Forward Pass
            # We assume the model outputs 'x_0_pred'. 
            # If your model outputs 'epsilon' directly, remove the conversion block.
            model_output = model(model_input, t_tensor, **net_fwd_kwargs)
            
            # --- CONVERSION BLOCK (Adjust based on model type) ---
            # We need epsilon for the ODE drift.
            # If model predicts x_0: eps = (x_t - sqrt(alpha)*x_0) / sqrt(1-alpha)
            # Note: model_input is x_t (pixel space)
            
            # Using the standard formula for epsilon derived from x_0 prediction:
            eps = (model_input - torch.sqrt(alpha_t) * model_output) / torch.sqrt(1 - alpha_t)
            
            # If your model predicts epsilon directly:
            # eps = model_output
            # -----------------------------------------------------

            # Drift term for the ODE: f(x_bar, sigma) = epsilon
            drift = eps
            
            # --- Divergence Estimation (Hutchinson) ---
            noise = torch.randn_like(x_bar)
            
            # Compute vjp = noise^T * J
            # We need grad of drift w.r.t x_bar (the loop variable)
            vjp = torch.autograd.grad(drift, x_bar, noise, create_graph=False)[0]
            
            # Trace estimation: sum(vjp * noise)
            div_est = (vjp * noise).sum(dim=tuple(range(1, len(x_bar.shape))))
            
            # Update Likelihood
            # CNF Formula: log p_0 = log p_T + Integral(div) d_sigma
            # We add the divergence term immediately, scaled by dimensions
            ll_accumulator_bpd += (div_est * dt) / (data_dim * np.log(2))
            nll_over_steps.append(ll_accumulator_bpd.detach().cpu().mean())
            
            # --- Euler Step ---
            # dx_bar = epsilon * d_sigma
            with torch.no_grad():
                x_bar = x_bar + drift * dt

            x0_std_steps.append(model_output.detach().cpu().std())
            eps_std_steps.append(eps.detach().cpu().std())
            x0_mean_steps.append(model_output.detach().cpu().mean())
            eps_mean_steps.append(eps.detach().cpu().mean())
            
            # Re-attach gradient for next step
            x_bar = x_bar.detach().requires_grad_(True)
            
        # 3. Final Prior Evaluation (at T)
        # At the end of encoding, x_bar_T should be Gaussian.
        # But what variance?
        # x_T ~ N(0, I). 
        # x_bar_T = x_T / sqrt(alpha_T).
        # Var(x_bar_T) = 1/alpha_T = sigma_T^2 + 1.
        final_var = ( sigma_next**2 + 1 ).cpu()
        
        # Log Prob of Gaussian with covariance 'final_var * I'
        # -d/2 * log(2*pi*var) - sum(x^2)/(2*var)
        log_prob_prior = (
            -0.5 * data_dim * np.log(2 * np.pi * final_var) 
            - 0.5 * (x_bar**2).sum(dim=tuple(range(1, len(x_bar.shape)))) / final_var
        )
        
        # Add prior term (converted to bits per dim)
        ll_accumulator_bpd += log_prob_prior / (data_dim * np.log(2))
        
    return ll_accumulator_bpd