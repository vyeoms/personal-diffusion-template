# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

# Implementation by Karras et al., taken from https://github.com/NVlabs/edm2/tree/main

"""Improved diffusion model architecture proposed in the paper
"Analyzing and Improving the Training Dynamics of Diffusion Models"."""

import numpy as np
import torch

from utils.mp_utils import MPConv, MPFourier # Modules from Karras et al's codebase.
from utils.misc_utils import append_dims # Utility function for appending dimensions to tensors.

class Precond(torch.nn.Module):
    def __init__(self,
        backbone_net: torch.nn.Module, # Backbone network class. Trying to standardize the interface for UNet and DiT. Assuming it comes initialized.
        label_dim: int,                # Class label dimensionality. 0 = unconditional.
        use_fp16        = True,        # Run the model at FP16 precision?
        sigma_data      = 0.5,         # Expected standard deviation of the training data.
        logvar_channels = 128          # Intermediate dimensionality for uncertainty estimation.
    ):
        super().__init__()
        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.sigma_data = sigma_data
        self.logvar_fourier = MPFourier(logvar_channels)
        self.logvar_linear = MPConv(logvar_channels, 1, kernel=[])
        self.backbone = backbone_net

    def forward(self, x, sigma, class_labels=None, force_fp32=False, return_logvar=False, **backbone_kwargs):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).flatten()
        class_labels = None if self.label_dim == 0 else torch.zeros([1, self.label_dim], device=x.device) if class_labels is None else class_labels.to(torch.float32).reshape(-1, self.label_dim)
        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32

        # Preconditioning weights.
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.flatten().log() / 4

        # Broadcast to match x's shape
        c_skip = append_dims(c_skip, x.ndim)
        c_out = append_dims(c_out, x.ndim)
        c_in = append_dims(c_in, x.ndim)

        # Run the model.
        x_in = (c_in * x).to(dtype)
        F_x = self.backbone(x_in, c_noise, class_labels, **backbone_kwargs)
        D_x = c_skip * x + c_out * F_x.to(torch.float32)

        # Estimate uncertainty if requested.
        if return_logvar:
            logvar = append_dims( self.logvar_linear(self.logvar_fourier(c_noise)), x.ndim )
            return D_x, logvar # u(sigma) in Equation 21
        return D_x

class EDM2Loss:
    def __init__(self, P_mean=-0.4, P_std=1.0, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, target, labels=None):
        rnd_normal = torch.randn([target.shape[0]], device=target.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2

        # Reshape sigma and weight to be broadcastable to target's shape
        sigma  = append_dims(sigma,  target.ndim)
        weight = append_dims(weight, target.ndim)

        noise = torch.randn_like(target) * sigma
        denoised, logvar = net(target + noise, sigma, labels, return_logvar=True)
        logvar = append_dims(logvar.flatten(), target.ndim) # ensure logvar is broadcastable to target's shape
        loss = (weight / logvar.exp()) * ((denoised - target) ** 2) + logvar
        return loss

def evaluate_log_likelihood(
        model: torch.nn.Module, 
        samples: torch.Tensor,
        tmax: float = 80.,
        tmin: float = 0.002,
        rho: float = 7,
        num_steps: int = 100,
        num_pixels = 1,
        **net_fwd_kwargs
    ):
    """
    Evaluate the log likelihood of given samples under the diffusion model.
    
    Args:
        target: Target distribution
        model: Score-based model that approximates the score function
        samples: Input samples to evaluate [batch_size, ...]
        tmax: Maximum diffusion time
        tmin: Minimum diffusion time
        rho: Parameter controlling the time schedule
        num_steps: Number of integration steps
        
    Returns:
        log_likelihood: Log likelihood of the input samples
    """
    # Generate time steps from tmin to tmax
    ts = tmin ** (1/rho) + np.arange(num_steps)/(num_steps-1) * (tmax ** (1/rho) - tmin ** (1/rho))
    ts = ts ** rho
    
    # dimensionality
    d = np.prod(samples.shape[1:])
    
    with torch.enable_grad():
        current_samples = samples.clone().detach().requires_grad_(True)
        
        # Start with the log probability under the data distribution
        # Initialize this to zero as we'll track the change in log probability
        log_likelihood = torch.zeros(samples.shape[0], device=samples.device)
        
        # Forward diffusion process (ODE from data to noise)
        for i in range(1, ts.shape[0]):
            t_prev = torch.ones(current_samples.shape[0], 1, 1, 1).to(current_samples.device) * ts[i-1]
            t = torch.ones(current_samples.shape[0], 1, 1, 1).to(current_samples.device) * ts[i]
            
            # Get score estimate from the model
            x_hat,_ = model(current_samples, t_prev, **net_fwd_kwargs)
            f = -(x_hat - current_samples)/t_prev
            
            # Estimate trace of Jacobian using Hutchinson's estimator
            epsilon = torch.randint(0, 2, current_samples.shape, device=current_samples.device).float() * 2 - 1
            
            vjp = torch.autograd.grad(f, current_samples, epsilon, create_graph=False, retain_graph=False)[0]
            trace_est = (vjp * epsilon).sum(dim=tuple(range(1, len(current_samples.shape))))
            
            # Update log likelihood (negative trace because we're going forward)
            dt = t - t_prev
            log_likelihood = log_likelihood + ( trace_est * dt / (num_pixels*torch.log(torch.tensor(2))) )
            
            # Forward ODE step (deterministic, no noise)
            with torch.no_grad():
                current_samples = current_samples + f * dt
                
            current_samples = current_samples.detach().requires_grad_(True)
                    
        # Add log probability of the final noise distribution (Gaussian with variance tmax^2)
        log_likelihood = log_likelihood - ( 0.5 * d * np.log(2 * np.pi * tmax**2) + \
                        torch.sum(current_samples**2, dim=tuple(range(1, len(current_samples.shape)))) / (2 * tmax**2) )/(num_pixels*torch.log(torch.tensor(2)))
                
        return log_likelihood
