import torch

@torch.no_grad()
def ddpm_sampler(ddpm_model, noise, labels=None, **kwargs):
    x = noise
    device = x.device
    for i in reversed(range(ddpm_model.T)):
        t = torch.full((x.shape[0],), i, device=device, dtype=torch.long)
        x0_hat = ddpm_model(x, t, labels)
        eps_hat = ddpm_model._predict_eps(x, t, x0_hat)

        mean = ddpm_model.sqrt_recip_alphas[i] * (
            x - ddpm_model.betas[i] / ddpm_model.sqrt_one_minus_alpha_bar[i] * eps_hat
        )

        if i > 0:
            x = mean + ddpm_model.posterior_variance[i].sqrt() * torch.randn_like(x)
        else:
            x = mean
    return x
