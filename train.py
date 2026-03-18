from collections import deque
from copy import deepcopy
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader
import wandb

from dataset.custom_data import CustomImageDataset
from utils.debug_viz_utils import plot_bucketed_data
from utils.misc_utils import noop, EMA, cycle

def parse_config_init(cfg, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    # Backbone
    if cfg.backbone.architecture == "unet":
        from backbones.karras_unet import UNet
        backbone = UNet(**cfg.backbone.init_kwargs).to(device)
    elif cfg.backbone.architecture == "mlp":
        from backbones.mlp import MLP
        backbone = MLP(**cfg.backbone.init_kwargs).to(device)
    elif cfg.backbone.architecture == "simple_mlp":
        from backbones.simple_mlp import SimpleMLP
        backbone = SimpleMLP(**cfg.backbone.init_kwargs).to(device)
    elif cfg.backbone.architecture == "mp_simple_mlp":
        from backbones.mp_simple_mlp import MPSimpleMLP
        backbone = MPSimpleMLP(**cfg.backbone.init_kwargs).to(device)
    elif cfg.backbone.architecture == "mp_mlp":
        from backbones.mp_mlp import MPMLP
        backbone = MPMLP(**cfg.backbone.init_kwargs).to(device)
    elif cfg.backbone.architecture == "dit":
        from backbones.dit import DiT
        backbone = DiT(**cfg.backbone.init_kwargs).to(device)
    else:
        raise ValueError(f"Unsupported architecture: {cfg.backbone.architecture}")
    
    # Diffusion type
    if cfg.diffusion.type == "ddpm":
        from diffusion.ddpm import DDPM, DDPMLoss
        model = DDPM(backbone, 
                     **cfg.diffusion.init_kwargs).to(device)
        loss_fn = DDPMLoss(**cfg.diffusion.loss_kwargs)
    elif cfg.diffusion.type == "edm":
        from diffusion.edm import Precond, EDM2Loss
        model = Precond(backbone, 
                        **cfg.diffusion.init_kwargs).to(device)
        loss_fn = EDM2Loss(**cfg.diffusion.loss_kwargs)
    else:
        raise ValueError(f"Unsupported diffusion type: {cfg.diffusion.type}")
    
    if cfg.sampler.type == "ddpm":
        from samplers.ddpm_sampler import ddpm_sampler
        sampler = ddpm_sampler
    elif cfg.sampler.type == "ddim":
        from samplers.ddim_sampler import ddim_sampler
        sampler = ddim_sampler
    elif cfg.sampler.type == "edm":
        from samplers.edm_sampler import edm_sampler
        sampler = edm_sampler
    else:
        raise ValueError(f"Unsupported sampler type: {cfg.sampler.type}")
    
    if cfg.training.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), 
                                     lr=cfg.training.lr, 
                                     **cfg.training.optimizer_kwargs)
    else:
        raise ValueError(f"Unsupported optimizer type: {cfg.training.optimizer}")
    return backbone, model, sampler, loss_fn, optimizer

@hydra.main(version_base=None, config_path="./config", config_name="base_config_edm")
def train(cfg: DictConfig):

    if cfg.logging.wandb.track:
        wandb.init(project=cfg.logging.wandb.project, name="test_run")
        log_fn = wandb.log
    else:
        log_fn = noop

    torch.random.manual_seed(cfg.training.random_seed)
    
    # Preprocess to variance 0.5, which is the expected standard deviation of the training data.
    # CHANGE WITH YOUR DATA STANDARD DEVIATION
    normalization_factor = 0.5/torch.sqrt(torch.tensor(2))

    # CHANGE WITH YOUR DATASET
    train_data = CustomImageDataset(split="train", n_samples=1024 - 256 )
    val_data = CustomImageDataset(split="val", n_samples=256)

    train_dl = cycle( DataLoader(train_data, batch_size=cfg.training.batch_size, shuffle=True) )
    val_dl = cycle( DataLoader(val_data, batch_size=cfg.training.batch_size, shuffle=False) )

    print(OmegaConf.to_yaml(cfg))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Setup block. Probably could use some refactoring a-la Karras for iteration, but it works ¯\_(ツ)_/¯
    backbone, model, sampler, loss_fn, optimizer = parse_config_init(cfg, device=device)
    
    # Initialize EMA
    ema = EMA(model, decay=cfg.training.ema_decay)
    
    checkpoint_dir = Path(cfg.logging.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    batch_size = cfg.training.batch_size
    
    # For EDM loss visualization and tracking
    if cfg.diffusion.type == "edm" and cfg.logging.wandb.track:
        not_agg_list = deque(maxlen=cfg.logging.edm_loss_viz_start)
        sigmas_list = deque(maxlen=cfg.logging.edm_loss_viz_start)
        P_mean = cfg.diffusion.loss_kwargs.P_mean
        P_std = cfg.diffusion.loss_kwargs.P_std
    
    # Training loop
    for step in range(cfg.training.iters):
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
            if step >= cfg.logging.edm_loss_viz_start:
                y = torch.cat([a.squeeze().detach().cpu() for a in not_agg_list])
                x = torch.cat([a.squeeze().detach().cpu() for a in sigmas_list])
                plot_bucketed_data(x, y, P_mean, P_std, log_fn, 
                                   num_buckets=100, method='equal_count', step=step)

            val_batch = next(val_dl).to(device) * normalization_factor
            validate(ema.get_model(), 
                        val_batch, 
                        step, 
                        n_samples=cfg.validation.n_samples, 
                        sampler=sampler, 
                        loss_fn=loss_fn, 
                        log_fn=log_fn, 
                        normalization_factor=normalization_factor, 
                        **cfg.sampler.sampler_kwargs)
        
        # Checkpoint
        if step % cfg.logging.save_freq == 0:
            torch.save(model.state_dict(), checkpoint_dir / f"step_{step}.pt")

def validate(model, val_batch, step, n_samples, sampler=None, loss_fn=None, 
             log_fn=None, normalization_factor=1.0, **sampler_kwargs):
    """Validation using EMA model."""

    # Just evaluation
    ema_model = model.eval()
    with torch.no_grad():
        noise = torch.randn([n_samples, 2]).to(val_batch.device)  # Sample noise for generation
        loss = loss_fn(ema_model, val_batch).mean()  # Diffusion loss
        samples = sampler(ema_model, noise, **sampler_kwargs)  # Sample from the model
        samples = samples / normalization_factor  # Scale back to original variance
        log_fn({'val_loss': loss.item()}, step=step)
        ax = sns.scatterplot(x=samples.cpu().numpy()[:, 0], y=samples.cpu().numpy()[:, 1])
        plt.xticks(ticks=[], labels=[])
        plt.yticks(ticks=[], labels=[])
        fig = ax.get_figure()
        log_fn({ f'Generated samples' : wandb.Image(fig) }, step=step)
        plt.close(fig)
    
    print(f"Validation loss at step {step}: {loss.item():.4f}")

    model.train()


if __name__ == "__main__":
    train()
