"""Guidance (④): wrap a trained denoiser in another with the same
D(x, σ, labels) interface, so samplers are untouched. `build_guidance` turns a
`guidance:` config block into a guided denoiser (see config/base_config_edm_cfg.yaml)."""

from omegaconf import OmegaConf


def _kwargs(node):
    if node is None:
        return {}
    return OmegaConf.to_container(node, resolve=True) if OmegaConf.is_config(node) else dict(node)


def build_guidance(guidance_cfg, model, ref_model=None):
    if guidance_cfg is None:
        return model
    gtype = guidance_cfg.get("type", "none")
    kw = _kwargs(guidance_cfg.get("kwargs", None))

    if gtype == "none":
        return model
    if gtype == "cfg":
        from guidance.cfg import CFGDenoiser
        return CFGDenoiser(model, **kw)
    if gtype == "autoguidance":
        from guidance.autoguidance import AutoGuidanceDenoiser
        if ref_model is None:
            raise ValueError("autoguidance needs a reference model (set guidance.ref_checkpoint)")
        return AutoGuidanceDenoiser(model, ref_model, **kw)
    if gtype == "interval":
        # `base` = the method that runs inside the window (default CFG).
        from guidance.interval import LimitedIntervalGuidance
        inner = build_guidance(guidance_cfg.get("base", None) or {"type": "cfg"}, model, ref_model)
        return LimitedIntervalGuidance(model, inner, **kw)
    raise ValueError(f"Unsupported guidance type: {gtype}")
