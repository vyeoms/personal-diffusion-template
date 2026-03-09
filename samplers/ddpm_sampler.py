# DDPM sampler implementation (x0-prediction). Created with Claude Opus 4.6

import torch

# ------------------------------------------------------------------
# DDPM ancestral sampling: x_{t-1} ~ p(x_{t-1} | x_t)
# ------------------------------------------------------------------
@torch.no_grad()
def ddpm_sampler(
    ddpm_model: torch.nn.Module, 
    noise: torch.Tensor = None, # Optional initial noise. If None, will sample fresh noise.
    num_steps: int = 1000,
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ) -> torch.Tensor:
    x = noise
    for i in reversed(range(num_steps)):
        t = torch.full((x.shape[0],), i, device=device, dtype=torch.long)
        x0_hat = ddpm_model(x, t)
        eps_hat = ddpm_model._predict_eps(x, t, x0_hat)

        mean = ddpm_model.sqrt_recip_alphas[i] * (
            x - ddpm_model.betas[i] / ddpm_model.sqrt_one_minus_alpha_bar[i] * eps_hat
        )

        if i > 0:
            x = mean + ddpm_model.posterior_variance[i].sqrt() * torch.randn_like(x)
        else:
            x = mean
    return x
