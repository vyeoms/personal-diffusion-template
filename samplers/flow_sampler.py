import torch
from utils.misc_utils import append_dims

@torch.no_grad()
def flow_sampler(model, noise, num_steps=100, labels=None, **kwargs):
    x = noise
    device = x.device
    dt = 1.0 / num_steps
    for i in range(num_steps):
        t = i / num_steps
        t_batch = torch.full((x.shape[0],), t, device=device)
        x1_hat = model(x, t_batch, labels)
        t_expanded = append_dims(t_batch, x.ndim)
        v = (x1_hat - x) / (1 - t_expanded).clamp(min=1e-5)
        x = x + v * dt
    return x
