# DDIM sampler implementation (x0-prediction). Created with Claude Opus 4.6

import torch
from utils.misc_utils import append_dims

# ------------------------------------------------------------------
# DDIM sampling (Song et al., 2021)
#   - num_steps << T  (sub-sequence of timesteps)
#   - eta = 0 → deterministic;  eta = 1 → equivalent to DDPM
# ------------------------------------------------------------------
@torch.no_grad()
def ddim_sampler(
    ddpm_model: torch.nn.Module,
    noise: torch.Tensor = None, # Optional initial noise. If None, will sample fresh noise.
    num_steps: int = 50,
    eta: float = 0.0,
    ddpm_T: int = 100,
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
) -> torch.Tensor:
    if noise is None:
        noise = torch.randn((num_steps, 2), device=device)
    x = noise

    alpha_bar = ddpm_model.alpha_bar

    # Evenly-spaced sub-sequence of timesteps
    times = torch.linspace(ddpm_T - 1, 0, num_steps + 1, device=device).long()

    for i in range(num_steps):
        t = times[i].expand(x.shape[0])
        t_next = times[i + 1].expand(x.shape[0])

        x0_pred = ddpm_model(x, t)
        eps_hat = ddpm_model._predict_eps(x, t, x0_pred)

        ab_t = append_dims(alpha_bar[t], x.ndim)
        ab_next = append_dims(alpha_bar[t_next], x.ndim)
        # DDIM variance
        sigma = eta * ((1 - ab_next) / (1 - ab_t) * (1 - ab_t / ab_next)).sqrt()

        # Predicted direction
        dir_xt = (1 - ab_next - sigma ** 2).clamp(min=0).sqrt() * eps_hat

        z = torch.randn_like(x) if (eta > 0 and i < num_steps - 1) else 0.0
        x = ab_next.sqrt() * x0_pred + dir_xt + sigma * z

    return x
