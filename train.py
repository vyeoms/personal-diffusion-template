from collections import deque
from importlib import import_module
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader
import wandb

from dataset.custom_data import CustomImageDataset
from utils.debug_viz_utils import plot_bucketed_data
from utils.misc_utils import noop
from utils.train_utils import cycle, EMA, load, save

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

def parse_config_init(cfg, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    # Backbone
    arch = cfg.backbone.architecture
    if arch not in BACKBONE_REGISTRY:
        raise ValueError(f"Unknown backbone: {arch}. Available: {list(BACKBONE_REGISTRY)}")
    backbone = _import(*BACKBONE_REGISTRY[arch])(**cfg.backbone.init_kwargs).to(device)

    # Diffusion
    diff_type = cfg.diffusion.type
    if diff_type not in DIFFUSION_REGISTRY:
        raise ValueError(f"Unknown diffusion type: {diff_type}. Available: {list(DIFFUSION_REGISTRY)}")
    diff_mod, model_cls, loss_cls, ll_name = DIFFUSION_REGISTRY[diff_type]
    model = _import(diff_mod, model_cls)(backbone, **cfg.diffusion.init_kwargs).to(device)
    loss_fn = _import(diff_mod, loss_cls)(**cfg.diffusion.loss_kwargs)
    ll_fn = _import(diff_mod, ll_name) if ll_name else None

    # Sampler
    samp_type = cfg.sampler.type
    if samp_type not in SAMPLER_REGISTRY:
        raise ValueError(f"Unknown sampler: {samp_type}. Available: {list(SAMPLER_REGISTRY)}")
    sampler = _import(*SAMPLER_REGISTRY[samp_type])

    # Optimizer
    opt_type = cfg.training.optimizer
    if opt_type not in OPTIMIZER_REGISTRY:
        raise ValueError(f"Unknown optimizer: {opt_type}. Available: {list(OPTIMIZER_REGISTRY)}")
    optimizer = OPTIMIZER_REGISTRY[opt_type](
        model.parameters(), lr=cfg.training.lr,
        **OmegaConf.to_container(cfg.training.optimizer_kwargs, resolve=True),
    )

    return backbone, model, sampler, loss_fn, ll_fn, optimizer

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
    
    # Preprocess to variance 0.5, which is the expected standard deviation of the training data.
    # CHANGE WITH YOUR DATA STANDARD DEVIATION
    normalization_factor = 0.5/torch.sqrt(torch.tensor(2))

    # CHANGE WITH YOUR DATASET
    train_data = CustomImageDataset(split="train", n_samples=1024 - 256 )
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
    
    # For EDM loss visualization and tracking
    if cfg.diffusion.type == "edm" and cfg.logging.wandb.track:
        not_agg_list = deque(maxlen=cfg.logging.edm_loss_viz_start)
        sigmas_list = deque(maxlen=cfg.logging.edm_loss_viz_start)
        P_mean = cfg.diffusion.loss_kwargs.P_mean
        P_std = cfg.diffusion.loss_kwargs.P_std
    ############################### Setup block ############################### 
    
    # Training loop
    for step in range(step, cfg.training.iters):
        train_batch = next(train_dl).to(device) * normalization_factor
        optimizer.zero_grad()
        if cfg.diffusion.type == "edm" and cfg.logging.wandb.track:
            loss, sigma = loss_fn(model, train_batch, visualize=True)
            not_agg_list.append(loss)
            sigmas_list.append(sigma)
        else:
            loss = loss_fn(model, train_batch)
        loss = loss.mean()
        loss.backward()
        optimizer.step()
        
        ema.update()

        # Logging
        if step % cfg.logging.train_log_freq == 0:
            wandb.log({"training_loss": loss.item()}, step=step)

        # Validation
        if step % cfg.logging.val_log_freq == 0:
            # This visualization is heavier so we only do it every val_log_freq steps.
            if cfg.diffusion.type == "edm" and step >= cfg.logging.edm_loss_viz_start:
                y = torch.cat([a.squeeze().detach().cpu() for a in not_agg_list])
                x = torch.cat([a.squeeze().detach().cpu() for a in sigmas_list])
                plot_bucketed_data(x, y, P_mean, P_std, log_fn, 
                                   num_buckets=100, method='equal_count', step=step)

            val_batch = next(val_dl).to(device) * normalization_factor
            cpu_state, gpu_state = torch.random.get_rng_state(), torch.cuda.get_rng_state()
            validate(ema.get_model(), 
                        val_batch, 
                        step, 
                        n_samples=cfg.validation.n_samples, 
                        sampler=sampler, 
                        loss_fn=loss_fn, 
                        ll_fn=ll_fn, 
                        log_fn=log_fn, 
                        normalization_factor=normalization_factor, 
                        **cfg.sampler.sampler_kwargs)
            torch.random.set_rng_state(cpu_state)
            torch.cuda.set_rng_state(gpu_state)
        
        # Checkpoint
        if step % cfg.logging.save_freq == 0:
            save(model, optimizer, ema, checkpoint_dir / f'model-{step}.pt', step=step)

def validate(model, val_batch, step, n_samples, sampler=None, loss_fn=None, 
             ll_fn=None, log_fn=None, normalization_factor=1.0, **sampler_kwargs):
    """Validation using EMA model."""

    # Just evaluation
    model.eval()
    with torch.no_grad():
        noise = torch.randn([n_samples] + list(val_batch.shape[1:])).to(val_batch.device)
        loss = loss_fn(model, val_batch).mean()  # Diffusion loss
        norm_samples = sampler(model, noise, **sampler_kwargs)  # Sample from the model
        samples = norm_samples / normalization_factor  # Scale back to original variance
        log_fn({'val_loss': loss.item()}, step=step)
        ax = sns.scatterplot(x=samples.cpu().numpy()[:, 0], y=samples.cpu().numpy()[:, 1])
        plt.xticks(ticks=[], labels=[])
        plt.yticks(ticks=[], labels=[])
        fig = ax.get_figure()
        log_fn({ f'Generated samples' : wandb.Image(fig) }, step=step)
        plt.close(fig)
        if ll_fn is not None:
            nll = -ll_fn(model, norm_samples)
            log_fn({'nll': nll.mean().item()}, step=step)
    
    print(f"Validation loss at step {step}: {loss.item():.4f}")

    model.train()


if __name__ == "__main__":
    train()
