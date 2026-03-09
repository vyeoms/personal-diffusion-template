import torch
import torch.nn as nn

from utils.mp_utils import MPFourier, MPLinear, mp_sum, mp_silu

class MPResBlock(nn.Module):
    """
    Magnitude-preserving residual block with additive time conditioning.

    Both the main path (MPLinear → MPSiLU) and the time path (MPLinear) have
    unit-magnitude outputs, so mp_sum combines them at equal weight while
    preserving magnitude.  The outer mp_sum does the same for the skip connection.
    """

    def __init__(self, dim: int, time_emb_dim: int):
        super().__init__()
        self.lin    = MPLinear(dim, dim)
        self.t_proj = MPLinear(time_emb_dim, dim)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = mp_sum(self.lin(x), self.t_proj(t_emb))   # fuse x-path and t-path
        return mp_sum(x, mp_silu(h))                   # residual


class MPSimpleMLP(nn.Module):
    """
    Magnitude-preserving MLP backbone for x0-prediction DDPM (Karras et al. EDM2).

    A direct MP counterpart of SimpleMLP — same architecture, same interface,
    with every non-MP primitive swapped out:

      SinusoidalTimeEmbedding  →  MPFourier   (unit-norm cosine features, Eq. 75)
      nn.Linear                →  MPLinear    (weight-normalised matmul,   Eq. 47/66)
      F.silu / nn.SiLU         →  mp_silu     (rescaled activation,        Eq. 81)
      x + h (residual)         →  mp_sum      (magnitude-preserving lerp,  Eq. 88)

    Interface:
        forward(x_t, t, labels=None) -> predicted_x0
        x_t : (B, in_dim)   noisy input
        t   : (B,)  float   timestep indices in [0, T-1]
    """

    def __init__(
        self,
        in_dim:       int = 2,
        hidden_dim:   int = 128,
        num_layers:   int = 3,
        time_emb_dim: int = 64,
    ):
        super().__init__()

        # MPFourier produces unit-magnitude embeddings: output var ≈ 1 by construction
        # (cos features scaled by sqrt(2); Eq. 75).  Pass t.float() directly.
        self.time_emb   = MPFourier(time_emb_dim)
        self.input_proj = MPLinear(in_dim, hidden_dim)
        self.blocks     = nn.ModuleList(
            [MPResBlock(hidden_dim, time_emb_dim) for _ in range(num_layers)]
        )
        self.out = MPLinear(hidden_dim, in_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor, labels=None) -> torch.Tensor:
        t_emb = self.time_emb(t)           # (B, time_emb_dim), magnitude ≈ 1
        h = mp_silu(self.input_proj(x.float()))    # (B, hidden_dim),   magnitude ≈ 1
        for block in self.blocks:
            h = block(h, t_emb)
        return self.out(h)                 # (B, in_dim)
