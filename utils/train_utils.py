# Utilities for training

from copy import deepcopy

import numpy as np
import torch

class EMA:
    """Exponential Moving Average for model parameters."""
    def __init__(self, model: torch.nn.Module, decay=0.99):
        self.model = model
        self.decay = decay
        self.ema_model = deepcopy(model)
        self.ema_model.eval()
    
    def update(self):
        for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
            ema_param.data = self.decay * ema_param.data + (1 - self.decay) * param.data.detach()
    
    def get_model(self):
        return self.ema_model

#----------------------------------------------------------------------------
# Learning rate decay schedule used in the paper "Analyzing and Improving
# the Training Dynamics of Diffusion Models".

def learning_rate_schedule(cur_nimg, batch_size, ref_lr=100e-4, ref_batches=70e3, rampup_Mimg=10):
    lr = ref_lr
    if ref_batches > 0:
        lr /= np.sqrt(max(cur_nimg / (ref_batches * batch_size), 1))
    if rampup_Mimg > 0:
        lr *= min(cur_nimg / (rampup_Mimg * 1e6), 1)
    return lr

def cycle(dl):
    while True:
        for data in dl:
            yield data

def save(model: torch.nn.Module, opt: torch.optim.Optimizer, ema: EMA, checkpoint_path, step=None):

    data = {
        'step': step,
        'model': model.state_dict(),
        'opt': opt.state_dict(),
        'ema': ema.get_model().state_dict(),
        'rng_state': torch.random.get_rng_state(),
        'cuda_rng_state': torch.cuda.get_rng_state()
    }

    torch.save(data, checkpoint_path)

def load(checkpoint_path, model: torch.nn.Module, opt: torch.optim.Optimizer, ema: EMA, device):

    data = torch.load(checkpoint_path)

    model.load_state_dict(data['model'])
    opt.load_state_dict(data['opt'])
    ema.model = model
    ema.ema_model.load_state_dict(data["ema"])
    torch.random.set_rng_state(data['rng_state'])
    torch.cuda.set_rng_state(data['cuda_rng_state'])
    step = data['step']

    return step
