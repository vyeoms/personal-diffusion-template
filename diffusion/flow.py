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

    def forward(self, x, sigma, class_labels=None, **backbone_kwargs):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).flatten()
        class_labels = None if self.label_dim == 0 else torch.zeros([1, self.label_dim], device=x.device) if class_labels is None else class_labels.to(torch.float32).reshape(-1, self.label_dim)

        return self.backbone(x, sigma, class_labels, **backbone_kwargs)

class FMLoss:
    def __init__(self, P_mean=0.0, P_std=1.0):
        self.P_mean = P_mean
        self.P_std = P_std

    def __call__(self, net, target, labels=None, visualize=False, min_weight=0.05):
        rnd_normal = torch.randn([target.shape[0]], device=target.device)
        t = torch.nn.functional.sigmoid(rnd_normal * self.P_std + self.P_mean)

        # Reshape sigma and weight to be broadcastable to target's shape
        t = append_dims(t, target.ndim)
        c_noise = t.flatten().logit()

        noise = torch.randn_like(target)
        z_t = t*target + (1-t)*noise
        vel_target = target - noise
        flow_velocity = (net(z_t, c_noise, labels) - z_t) / (1 - t).clamp(min=min_weight) # Clipping as "Back to Basics"
        loss = (flow_velocity - vel_target).pow(2)
        if visualize:
            return loss, t.flatten()
        return loss
