# Autoguidance (Karras et al., 2024 — arxiv.org/abs/2406.02507).

from guidance.base import GuidedDenoiser


class AutoGuidanceDenoiser(GuidedDenoiser):
    """D_bad + w·(D − D_bad), where D_bad is a weaker copy of the model. Needs no
    null label, so it guides unconditional models too."""

    def __init__(self, net, ref_net, guidance_scale=1.0):
        super().__init__(net)
        self.ref_net = ref_net
        self.w = float(guidance_scale)

    def __call__(self, x, t, labels=None, **kwargs):
        d = self.net(x, t, labels, **kwargs)
        if self.w == 1.0:
            return d
        return self.ref_net(x, t, labels, **kwargs).lerp(d, self.w)
