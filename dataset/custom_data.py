import torch
from torch.utils.data import Dataset

# Custom dataset of points on a unit circle in 2D.
class CustomImageDataset(Dataset):  # noqa: N801 (name kept for back-compat)
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


# K Gaussian blobs on a ring, label = blob index. The minimal conditional toy
# where CFG visibly tightens the selected blob. Returns (point, label).
class ConditionalToyDataset(Dataset):
    def __init__(self, split="train", n_samples=768, num_classes=6, radius=1.0, blob_std=0.1, seed=0):
        super().__init__()
        self.n_samples = n_samples
        self.num_classes = num_classes

        g = torch.Generator().manual_seed(seed + (0 if split == "train" else 1))  # train/val disjoint
        labels = (torch.arange(n_samples) % num_classes)[torch.randperm(n_samples, generator=g)]
        angles = 2 * torch.pi * labels.float() / num_classes
        centers = torch.stack([radius * torch.cos(angles), radius * torch.sin(angles)], dim=-1)
        self.data = centers + blob_std * torch.randn(n_samples, 2, generator=g)
        self.labels = labels.long()

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
