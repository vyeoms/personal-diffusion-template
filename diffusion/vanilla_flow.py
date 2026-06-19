# Minimal Flow Matching implementation.

import torch

from utils.misc_utils import append_dims # Utility function for appending dimensions to tensors.

class Flow(torch.nn.Module):
    def __init__(self,
        backbone_net: torch.nn.Module, # Backbone network class. Trying to standardize the interface for UNet and DiT. Assuming it comes initialized.
        label_dim: int                # Class label dimensionality. 0 = unconditional.
    ):
        super().__init__()
        self.label_dim = label_dim
        self.backbone = backbone_net

    def forward(self, x, t, labels=None, **backbone_kwargs):
        x = x.to(torch.float32)
        t = t.to(torch.float32).flatten()
        labels = None if self.label_dim == 0 else torch.zeros([1, self.label_dim], device=x.device) if labels is None else labels.to(torch.float32).reshape(-1, self.label_dim)

        c_noise = t.logit()/4

        return self.backbone(x, c_noise, labels, **backbone_kwargs)

def vanilla_fm_loss(model, target: torch.Tensor, labels=None) -> torch.Tensor:
    B = target.shape[0]
    device = target.device

    x0 = torch.randn_like(target)
    t = torch.rand(B, device=device)
    t = append_dims(t, target.ndim)

    x_t = (1 - t) * x0 + t * target
    u_t = target - x0

    v_t = model(x_t, t, labels)
    return torch.nn.functional.mse_loss(v_t, u_t)

def logit_fm_loss(model, target: torch.Tensor, labels=None,
                 logit_mean = 0.0, logit_std=1.0) -> torch.Tensor:
    B = target.shape[0]
    device = target.device

    x0 = torch.randn_like(target)
    rnd_normal = torch.randn([target.shape[0]], device=target.device)
    t = torch.nn.functional.sigmoid(rnd_normal * logit_std + logit_mean)
    t = append_dims(t, target.ndim)

    x_t = (1 - t) * x0 + t * target
    u_t = target - x0

    v_t = model(x_t, t, labels)
    return torch.nn.functional.mse_loss(v_t, u_t)
