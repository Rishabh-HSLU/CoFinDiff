from abc import ABC, abstractmethod
import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path
from torch.utils.data import DataLoader

class Dataset(ABC):
    def __init__(self, dataset_path, transform=None):
        self.dataset_path = dataset_path
        self.transform = transform

    @abstractmethod
    def __len__(self):
        ...

    @abstractmethod
    def __getitem__(self, idx: int):
        ...

    @abstractmethod
    def transform(self, image: Image.Image) -> torch.Tensor:
        ...

class AFHQImageDataset(Dataset):
    def __init__(self, dataset_path):
        super().__init__(dataset_path)
        self.image_paths = list(Path(dataset_path).glob("*.png"))

    def __getitem__(self, idx: int) -> torch.Tensor:
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")

        image = self.transform(image)
        return image

    def __len__(self):
        return len(self.image_paths)

    def transform(self, image: Image.Image) -> torch.Tensor:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])(image)


def get_dataloader(dataset: Dataset, batch_size: int = 1, shuffle: bool = True, num_workers: int = 0) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

class NoiseTimeSampler:
    def __init__(self, num_samples: int, noise_dim: int) -> None:
        self.num_samples = num_samples
        self.noise_dim = noise_dim

    def sample(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Samples noise and time uniformly.
        Returns:
            - noise: shape (num_samples, noise_dim)
            - time: shape (num_samples, 1)
        """
        noise = torch.randn(self.num_samples, self.noise_dim)
        time = torch.rand(self.num_samples, 1)
        return noise, time