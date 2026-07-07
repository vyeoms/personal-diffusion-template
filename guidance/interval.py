# Limited-interval guidance (Kynkäänniemi et al., 2024 — arxiv.org/abs/2404.07724).

import torch

from guidance.base import GuidedDenoiser


class LimitedIntervalGuidance(GuidedDenoiser):
    """Use `guided` only for t in [t_lo, t_hi], else the plain conditional `base`.
    Composes on top of CFG / autoguidance. Units of t follow the sampler: σ for
    edm_sampler, t∈[0,1] for flow_sampler, step index for DDPM/DDIM."""

    def __init__(self, base, guided, t_lo=0.0, t_hi=float("inf")):
        super().__init__(base)
        self.guided = guided
        self.t_lo, self.t_hi = float(t_lo), float(t_hi)

    def __call__(self, x, t, labels=None, **kwargs):
        t_val = float(t.mean()) if torch.is_tensor(t) else float(t)
        inner = self.guided if self.t_lo <= t_val <= self.t_hi else self.net
        return inner(x, t, labels, **kwargs)
