import torch
from torch.utils.data import Dataset

# Custom dataset of points on a unit circle in 2D.
class CustomImageDataset(Dataset):
    def __init__(
        self,
        split="train",
        n_samples=1_000_000
    ):
        super().__init__()
        self.n_samples = n_samples
        thetas = torch.linspace(-torch.pi, torch.pi, n_samples)
        radii = torch.ones(n_samples)
        x = radii * torch.cos(thetas)
        y = radii * torch.sin(thetas)
        self.data = torch.stack([x, y], dim=-1)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        sample = self.data[idx]
        return sample
