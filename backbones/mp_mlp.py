import torch

from utils.mp_utils import MPFourier, MPLinear, MPSiLU

class MPMLP(torch.nn.Module):
    def __init__(self,
        in_dim       = 2,
        num_layers   = 4,
        hidden_dim   = 64,
        time_emb_dim = 32,
    ):
        super().__init__()
        self.time_emb = MPFourier(time_emb_dim)
        self.layers = torch.nn.Sequential()
        self.layers.append(MPLinear(in_dim + time_emb_dim, hidden_dim))
        for _ in range(num_layers):
            self.layers.append(MPSiLU())
            self.layers.append(MPLinear(hidden_dim, hidden_dim))
        self.layers.append(MPLinear(hidden_dim, in_dim))

    def forward(self, x, t, labels=None):
        t_emb = self.time_emb(t)
        y = self.layers(torch.cat([x.float(), t_emb], dim=-1))
        return y
