# Classifier-free guidance (Ho & Salimans, 2022 — arxiv.org/abs/2207.12598).

import torch

from guidance.base import GuidedDenoiser


class CFGDenoiser(GuidedDenoiser):
    """D(·|∅) + w·(D(·|c) − D(·|∅)), with the null label ∅ = the all-zeros vector."""

    def __init__(self, net, guidance_scale=1.0):
        super().__init__(net)
        self.w = float(guidance_scale)

    def __call__(self, x, t, labels=None, **kwargs):
        if labels is None or self.w == 1.0:
            return self.net(x, t, labels, **kwargs)
        x_in = torch.cat([x, x])
        c_in = torch.cat([labels, torch.zeros_like(labels)])
        # Duplicate t only when it is per-sample; a scalar σ (EDM) broadcasts.
        t_in = torch.cat([t, t]) if torch.is_tensor(t) and t.ndim and t.shape[0] == x.shape[0] else t
        d_cond, d_uncond = self.net(x_in, t_in, c_in, **kwargs).chunk(2)
        return d_uncond.lerp(d_cond, self.w)
