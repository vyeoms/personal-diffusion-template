import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    """Standard sinusoidal positional embedding for discrete timesteps."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) integer or float timesteps
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, dtype=torch.float32, device=t.device) / (half - 1)
        )  # (half,)
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)  # (B, half)
        return torch.cat([args.sin(), args.cos()], dim=-1)   # (B, dim)


class ResBlock(nn.Module):
    """Linear residual block with affine time conditioning (scale + shift)."""

    def __init__(self, dim: int, time_emb_dim: int):
        super().__init__()
        self.lin = nn.Linear(dim, dim)
        # Projects time embedding to (scale, shift) for affine conditioning
        self.time_proj = nn.Linear(time_emb_dim, 2 * dim)
        nn.init.zeros_(self.time_proj.weight)
        nn.init.zeros_(self.time_proj.bias)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        scale, shift = self.time_proj(t_emb).chunk(2, dim=-1)  # (B, dim) each
        h = self.lin(x)
        h = h * (1.0 + scale) + shift      # affine conditioning
        return x + F.silu(h)               # residual connection


class SimpleMLP(nn.Module):
    """
    Lightweight MLP backbone for x0-prediction DDPM on low-dimensional data.

    Interface:
        forward(x_t, t, labels=None) -> predicted_x0
        x_t : (B, in_dim)  — noisy input
        t   : (B,)  float  — timestep indices in [0, T-1]
    """

    def __init__(
        self,
        in_dim: int = 2,
        hidden_dim: int = 128,
        num_layers: int = 3,
        time_emb_dim: int = 64,
    ):
        super().__init__()

        self.time_emb = SinusoidalTimeEmbedding(time_emb_dim)
        # Two-layer MLP to embed raw sinusoidal features into richer time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim),
        )

        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [ResBlock(hidden_dim, time_emb_dim) for _ in range(num_layers)]
        )
        self.out = nn.Linear(hidden_dim, in_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor, labels=None) -> torch.Tensor:
        t_emb = self.time_mlp(self.time_emb(t))   # (B, time_emb_dim)
        h = F.silu(self.input_proj(x))             # (B, hidden_dim)
        for block in self.blocks:
            h = block(h, t_emb)
        return self.out(h)
