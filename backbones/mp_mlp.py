import torch

from utils.mp_utils import MPSiLU, MPLinear

#----------------------------------------------------------------------------
# Denoiser model for learning 2D toy distributions. Inputs a set of sample
# positions and a single scalar for each, representing the logarithm of the
# corresponding unnormalized probability density. The score vector can then
# be obtained through automatic differentiation.

class MPMLP(torch.nn.Module):
    def __init__(self,
        in_dim      = 2,    # Input dimensionality.
        num_layers  = 4,    # Number of hidden layers.
        hidden_dim  = 64,   # Number of hidden features.
        sigma_data  = 0.5,  # Expected standard deviation of the training data.
    ):
        super().__init__()
        self.sigma_data = sigma_data
        self.layers = torch.nn.Sequential()
        self.layers.append(MPLinear(in_dim + 2, hidden_dim))
        for _layer_idx in range(num_layers):
            self.layers.append(MPSiLU())
            self.layers.append(MPLinear(hidden_dim, hidden_dim))
        self.layers.append(MPLinear(hidden_dim, in_dim))

    def forward(self, x, t=0, class_labels=None):
        sigma = torch.as_tensor(t, dtype=torch.float32, device=x.device).broadcast_to(x.shape[:-1]).unsqueeze(-1)
        x = x / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        y = self.layers(torch.cat([x, sigma, torch.ones_like(sigma)], dim=-1))
        return y
