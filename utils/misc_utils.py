# Miscellaneous utilities for general use

from copy import deepcopy

import torch

# Appends dimensions to the end of a tensor until it has target_dims dimensions
# Taken from k-diffusion: https://github.com/crowsonkb/k-diffusion/blob/master/k_diffusion/utils.py#L43C1-L48C48
def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f'input has {x.ndim} dims but target_dims is {target_dims}, which is less')
    return x[(...,) + (None,) * dims_to_append]

def noop(*args, **kwargs):
    pass

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
