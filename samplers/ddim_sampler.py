import torch
from utils.misc_utils import append_dims

@torch.no_grad()
def ddim_sampler(ddpm_model, noise, num_steps=50, eta=0.0, labels=None, **kwargs):
    x = noise
    device = x.device
    ddpm_T = ddpm_model.T
    alpha_bar = ddpm_model.alpha_bar

    times = torch.linspace(ddpm_T - 1, 0, num_steps + 1, device=device).long()

    for i in range(num_steps):
        t = times[i].expand(x.shape[0])
        t_next = times[i + 1].expand(x.shape[0])

        x0_pred = ddpm_model(x, t, labels)
        eps_hat = ddpm_model._predict_eps(x, t, x0_pred)

        ab_t = append_dims(alpha_bar[t], x.ndim)
        ab_next = append_dims(alpha_bar[t_next], x.ndim)
        sigma = eta * ((1 - ab_next) / (1 - ab_t) * (1 - ab_t / ab_next)).sqrt()

        dir_xt = (1 - ab_next - sigma ** 2).clamp(min=0).sqrt() * eps_hat

        z = torch.randn_like(x) if (eta > 0 and i < num_steps - 1) else 0.0
        x = ab_next.sqrt() * x0_pred + dir_xt + sigma * z

    return x
