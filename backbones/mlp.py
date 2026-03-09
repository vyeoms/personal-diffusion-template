import torch

from utils.mp_utils import MPFourier

class ConditionalLinear(torch.nn.Module):
    def __init__(self, num_in, num_out, bandwidth=1, num_channels=50, embnetworklayers=1):
        super(ConditionalLinear, self).__init__()
        self.num_out = num_out
        self.lin = torch.nn.Linear(num_in, num_out)
        self.embed = MPFourier(num_channels, bandwidth)
        if embnetworklayers == 1:
            self.embnetwork = torch.nn.Linear(num_channels,num_out)
        else:
            self.embnetwork = torch.nn.Sequential(*[(torch.nn.Linear(num_channels, num_channels), torch.nn.Softplus()) for _ in range(embnetworklayers-1)] + 
                                           torch.nn.Linear(num_channels,num_out))

    def forward(self, x, sigma):
        out = self.lin(x)
        gamma = self.embnetwork(self.embed(sigma))
        out = gamma * out
        return out

class MLP(torch.nn.Module):
    def __init__(
            self,
            in_dim=2,
            hidden_dim=128,
            num_layers=2,
            bandwidth=1,
            num_channels=100,
            embnetworklayers=1,
        ):
        super(MLP, self).__init__()
        self.hidden_layers = torch.nn.ModuleList(
            [ConditionalLinear(in_dim, hidden_dim, bandwidth, num_channels, embnetworklayers)] +
            [ConditionalLinear(hidden_dim, hidden_dim, bandwidth, num_channels, embnetworklayers) for _ in range(num_layers)]
        )
        self.out = torch.nn.Linear(hidden_dim, in_dim)

    def forward(self, x, sigma, labels=None):
        for layer in self.hidden_layers:
            x = torch.nn.functional.softplus(layer(x, sigma))
        x = self.out(x)
        return x
