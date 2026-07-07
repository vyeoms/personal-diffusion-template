class GuidedDenoiser:
    """Wraps a denoiser as another denoiser: subclasses override __call__ to
    reshape D(x, t, labels); every other attribute delegates to the wrapped net
    (so guided denoisers stay drop-in for samplers that touch _predict_eps etc.)."""

    def __init__(self, net):
        self.net = net

    def __call__(self, x, t, labels=None, **kwargs):
        raise NotImplementedError

    def __getattr__(self, name):
        return getattr(self.__dict__["net"], name)
