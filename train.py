from collections import deque
from importlib import import_module
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import wandb

from dataset.custom_data import CustomImageDataset, ConditionalToyDataset
from guidance import build_guidance
from utils.debug_viz_utils import plot_bucketed_data
from utils.misc_utils import noop
from utils.train_utils import cycle, EMA, load, save, learning_rate_schedule

# ---------------------------------------------------------------------------
# Registries — add a single line to support a new component.
# Each entry is (module_path, class_or_function_name).
# For diffusion: (module, model_cls, loss_cls, ll_fn_name_or_None).
# ---------------------------------------------------------------------------
BACKBONE_REGISTRY = {
    "unet":       ("backbones.karras_unet", "UNet"),
    "mlp":        ("backbones.mlp", "MLP"),
    "res_mlp":    ("backbones.res_mlp", "ResMLP"),
    "mp_res_mlp": ("backbones.mp_res_mlp", "MPResMLP"),
    "mp_mlp":     ("backbones.mp_mlp", "MPMLP"),
    "dit":        ("backbones.dit", "DiT"),
}

DIFFUSION_REGISTRY = {
    "ddpm": ("diffusion.ddpm",      "DDPM",    "DDPMLoss",    "evaluate_log_likelihood"),
    "edm":  ("diffusion.edm",       "Precond", "EDM2Loss",    "evaluate_log_likelihood"),
    "flow": ("diffusion.b2b_flow",  "Flow",    "LogitFMLoss", None),
}

SAMPLER_REGISTRY = {
    "ddpm": ("samplers.ddpm_sampler", "ddpm_sampler"),
    "ddim": ("samplers.ddim_sampler", "ddim_sampler"),
    "edm":  ("samplers.edm_sampler",  "edm_sampler"),
    "flow": ("samplers.flow_sampler", "flow_sampler"),
}

OPTIMIZER_REGISTRY = {
    "adam": torch.optim.Adam,
}

def _import(module_path, attr_name):
    return getattr(import_module(module_path), attr_name)

def _build_model(cfg, device):
    """①+② — backbone wrapped in its diffusion formulation (the denoiser D)."""
    backbone = _import(*BACKBONE_REGISTRY[cfg.backbone.architecture])(**cfg.backbone.init_kwargs).to(device)
    diff_mod, model_cls, *_ = DIFFUSION_REGISTRY[cfg.diffusion.type]
    model = _import(diff_mod, model_cls)(backbone, **cfg.diffusion.init_kwargs).to(device)
    return backbone, model

def parse_config_init(cfg, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    if cfg.backbone.architecture not in BACKBONE_REGISTRY:
        raise ValueError(f"Unknown backbone: {cfg.backbone.architecture}. Available: {list(BACKBONE_REGISTRY)}")
    if cfg.diffusion.type not in DIFFUSION_REGISTRY:
        raise ValueError(f"Unknown diffusion type: {cfg.diffusion.type}. Available: {list(DIFFUSION_REGISTRY)}")
    if cfg.sampler.type not in SAMPLER_REGISTRY:
        raise ValueError(f"Unknown sampler: {cfg.sampler.type}. Available: {list(SAMPLER_REGISTRY)}")
    if cfg.training.optimizer not in OPTIMIZER_REGISTRY:
        raise ValueError(f"Unknown optimizer: {cfg.training.optimizer}. Available: {list(OPTIMIZER_REGISTRY)}")

    backbone, model = _build_model(cfg, device)

    diff_mod, _model_cls, loss_cls, ll_name = DIFFUSION_REGISTRY[cfg.diffusion.type]
    loss_fn = _import(diff_mod, loss_cls)(**cfg.diffusion.loss_kwargs)
    ll_fn = _import(diff_mod, ll_name) if ll_name else None

    sampler = _import(*SAMPLER_REGISTRY[cfg.sampler.type])

    optimizer = OPTIMIZER_REGISTRY[cfg.training.optimizer](
        model.parameters(), lr=cfg.training.lr,
        **OmegaConf.to_container(cfg.training.optimizer_kwargs, resolve=True),
    )

    return backbone, model, sampler, loss_fn, ll_fn, optimizer

def build_reference_model(cfg, device):
    """Autoguidance reference: same architecture, EMA weights of an earlier checkpoint."""
    _, ref = _build_model(cfg, device)
    ref.load_state_dict(torch.load(cfg.guidance.ref_checkpoint, map_location=device)["ema"])
    ref.eval()
    return ref

def _prep_batch(batch, conditional, num_classes, nf, device):
    """Dataloader item -> (normalized data, one-hot labels or None)."""
    if conditional:
        x, y = batch
        return x.to(device) * nf, F.one_hot(y.to(device), num_classes).float()
    return batch.to(device) * nf, None

def _set_lr(optimizer, step, cfg, grad_accum):
    """Per-step LR: the EDM2 effective-LR schedule if training.lr_schedule is set
    (rampup + inverse-sqrt decay, for serious runs), else linear warmup_steps."""
    sched = cfg.training.get("lr_schedule", None)
    if sched is not None:
        lr = learning_rate_schedule(step * cfg.training.batch_size * grad_accum,
                                    cfg.training.batch_size, **sched)
    elif cfg.training.get("warmup_steps", 0) > 0:
        lr = cfg.training.lr * min((step + 1) / cfg.training.warmup_steps, 1.0)
    else:
        return
    for group in optimizer.param_groups:
        group["lr"] = lr

@hydra.main(version_base=None, config_path="./config", config_name="base_config_edm")
def train(cfg: DictConfig):

    name = f"{cfg.diffusion.type}_{cfg.backbone.architecture}_lr{cfg.training.lr}_bs{cfg.training.batch_size}"

    if cfg.logging.wandb.track:
        wandb.init(project=cfg.logging.wandb.project, name=name)
        log_fn = wandb.log
    else:
        log_fn = noop

    print(OmegaConf.to_yaml(cfg))

    ############################### Data block ###############################

    # Preprocess so per-dim std matches sigma_data (preconditioning assumes it).
    # Config-overridable; defaults to the 2D-toy value.
    # CHANGE WITH YOUR DATA STANDARD DEVIATION
    nf = cfg.data.get("normalization_factor", None)
    normalization_factor = torch.tensor(float(nf)) if nf is not None else 0.5 / torch.sqrt(torch.tensor(2.0))

    # num_classes > 0 flips on the label plumbing (dataset -> loss -> guidance).
    num_classes = cfg.data.get("num_classes", 0)
    conditional = num_classes > 0

    # CHANGE WITH YOUR DATASET
    if cfg.data.get("dataset", "custom") == "conditional_toy":
        train_data = ConditionalToyDataset(split="train", n_samples=1024 - 256, num_classes=num_classes)
        val_data = ConditionalToyDataset(split="val", n_samples=256, num_classes=num_classes)
    else:
        train_data = CustomImageDataset(split="train", n_samples=1024 - 256)
        val_data = CustomImageDataset(split="val", n_samples=256)

    train_dl = cycle( DataLoader(train_data, batch_size=cfg.training.batch_size, shuffle=True) )
    val_dl = cycle( DataLoader(val_data, batch_size=cfg.training.batch_size, shuffle=False) )
    ############################### Data block ###############################

    ############################### Setup block ###############################
    # Probably could use some refactoring a-la Karras for iteration, but it works ¯\_(ツ)_/¯
    step = 0
    torch.random.manual_seed(cfg.training.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = Path(cfg.logging.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    _, model, sampler, loss_fn, ll_fn, optimizer = parse_config_init(cfg, device=device)

    # Initialize EMA
    ema = EMA(model, decay=cfg.training.ema_decay)

    # Using a checkpoint?
    if cfg.training.continue_from_checkpoint:
        assert Path(cfg.training.checkpoint_path).exists(), "Specified checkpoint file does not exist."
        step = load(cfg.training.checkpoint_path, model, optimizer, ema, device)
        print(f"Resuming training from step {step}...")
    else:
        print("Starting training from scratch...")

    # ④ Guidance wraps the EMA denoiser for sampling (type: none -> returned as-is).
    guidance_cfg = cfg.get("guidance", None)
    ref_model = build_reference_model(cfg, device) if (
        guidance_cfg is not None and guidance_cfg.get("type", "none") == "autoguidance") else None
    guided_model = build_guidance(guidance_cfg, ema.get_model(), ref_model)

    # Fixed round-robin labels for validation sampling (one per class, repeating).
    sample_labels = None
    if conditional:
        idx = torch.arange(cfg.validation.n_samples, device=device) % num_classes
        sample_labels = F.one_hot(idx, num_classes).float()

    # For EDM loss visualization and tracking
    if cfg.diffusion.type == "edm" and cfg.logging.wandb.track:
        not_agg_list = deque(maxlen=cfg.logging.edm_loss_viz_start)
        sigmas_list = deque(maxlen=cfg.logging.edm_loss_viz_start)
        P_mean = cfg.diffusion.loss_kwargs.P_mean
        P_std = cfg.diffusion.loss_kwargs.P_std
    ############################### Setup block ###############################

    # Training loop
    grad_accum = max(1, cfg.training.get("gradient_accumulation_steps", 1))
    max_grad_norm = cfg.training.get("max_grad_norm", None)
    for step in range(step, cfg.training.iters):
        _set_lr(optimizer, step, cfg, grad_accum)

        optimizer.zero_grad()
        losses = []
        for _ in range(grad_accum):  # accumulate grads over micro-batches
            train_batch, labels = _prep_batch(next(train_dl), conditional, num_classes, normalization_factor, device)
            if cfg.diffusion.type == "edm" and cfg.logging.wandb.track:
                loss_elem, sigma = loss_fn(model, train_batch, labels, visualize=True)
                not_agg_list.append(loss_elem)
                sigmas_list.append(sigma)
            else:
                loss_elem = loss_fn(model, train_batch, labels)
            micro_loss = loss_elem.mean()
            (micro_loss / grad_accum).backward()
            losses.append(micro_loss.detach())

        if max_grad_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        ema.update()

        loss = torch.stack(losses).mean()

        # Logging
        if step % cfg.logging.train_log_freq == 0:
            log_fn({"training_loss": loss.item()}, step=step)

        # Validation
        if step % cfg.logging.val_log_freq == 0:
            # This visualization is heavier so we only do it every val_log_freq steps.
            if cfg.diffusion.type == "edm" and cfg.logging.wandb.track and step >= cfg.logging.edm_loss_viz_start:
                y = torch.cat([a.squeeze().detach().cpu() for a in not_agg_list])
                x = torch.cat([a.squeeze().detach().cpu() for a in sigmas_list])
                plot_bucketed_data(x, y, P_mean, P_std, log_fn,
                                   num_buckets=100, method='equal_count', step=step)

            val_batch, val_labels = _prep_batch(next(val_dl), conditional, num_classes, normalization_factor, device)
            cpu_state, gpu_state = torch.random.get_rng_state(), torch.cuda.get_rng_state()
            validate(ema.get_model(),
                        guided_model,
                        val_batch,
                        step,
                        n_samples=cfg.validation.n_samples,
                        sampler=sampler,
                        loss_fn=loss_fn,
                        ll_fn=ll_fn,
                        log_fn=log_fn,
                        normalization_factor=normalization_factor,
                        loss_labels=val_labels,
                        sample_labels=sample_labels,
                        **cfg.sampler.sampler_kwargs)
            torch.random.set_rng_state(cpu_state)
            torch.cuda.set_rng_state(gpu_state)

        # Checkpoint
        if step % cfg.logging.save_freq == 0:
            save(model, optimizer, ema, checkpoint_dir / f'model-{step}.pt', step=step)

def validate(model, guided_model, val_batch, step, n_samples, sampler=None, loss_fn=None,
             ll_fn=None, log_fn=None, normalization_factor=1.0,
             loss_labels=None, sample_labels=None, **sampler_kwargs):
    """Val loss + NLL under `model`; sampling goes through `guided_model`."""

    model.eval()
    with torch.no_grad():
        noise = torch.randn([n_samples] + list(val_batch.shape[1:])).to(val_batch.device)
        loss = loss_fn(model, val_batch, loss_labels).mean()
        norm_samples = sampler(guided_model, noise, labels=sample_labels, **sampler_kwargs)
        samples = norm_samples / normalization_factor  # back to original variance
        log_fn({'val_loss': loss.item()}, step=step)

        # Colour by class when conditional, so guidance's effect is visible.
        scatter_kwargs = dict(x=samples.cpu().numpy()[:, 0], y=samples.cpu().numpy()[:, 1])
        if sample_labels is not None:
            scatter_kwargs.update(hue=sample_labels.argmax(dim=-1).cpu().numpy(), palette="tab10", legend=False)
        ax = sns.scatterplot(**scatter_kwargs)
        plt.xticks(ticks=[], labels=[])
        plt.yticks(ticks=[], labels=[])
        fig = ax.get_figure()
        log_fn({ f'Generated samples' : wandb.Image(fig) }, step=step)
        plt.close(fig)
        if ll_fn is not None:
            nll_kwargs = {} if sample_labels is None else {"labels": sample_labels}
            nll = -ll_fn(model, norm_samples, **nll_kwargs)
            log_fn({'nll': nll.mean().item()}, step=step)

    print(f"Validation loss at step {step}: {loss.item():.4f}")

    model.train()


if __name__ == "__main__":
    train()
