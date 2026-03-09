# Miscellaneous utilities for general use

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
