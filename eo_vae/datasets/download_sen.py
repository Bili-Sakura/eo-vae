import json
import os
from glob import glob
from pathlib import Path
from typing import Any, Callable, Sequence
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor
from torchgeo.datasets import NonGeoDataset, NonGeoDataModule


class Sen2NaipCrossSensorLatent(NonGeoDataset):
    """Sen2Naip latent encodings dataset."""

    valid_splits = ['train', 'val', 'test']

    def __init__(
        self,
        root: Path = 'data',
        split: str = 'train',
        latent_scale_factor: float = 1.0,
        transforms: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        """Initialize a new Sen2Naip dataset instance.

        Args:
            root: Root directory where the dataset should be stored.
            split: Dataset split to load. Must be one of 'train' or 'val'.
            latent_scale_factor: Scale factor to apply to the latent encodings.
            transforms: A function/transform that takes input sample and its target as entry
                and returns a transformed version.
            download: Whether to download the dataset if it is not found on disk.
            checksum: Whether to verify the integrity of the dataset after download.
        """

        assert split in self.valid_splits, f'Split must be one of {self.valid_splits}'

        self.root = root
        self.transforms = transforms
        self.download = download
        self.checksum = checksum
        self.latent_scale_factor = latent_scale_factor

        self.aois = glob(os.path.join(self.root, split, '*.npz'))

        self.metadata_df = pd.DataFrame(self.aois, columns=['path'])

        # --- Load Normalization Stats from JSON ---
        stats_path = os.path.join(self.root, 'latent_stats.json')
        if not os.path.exists(stats_path):
            raise FileNotFoundError(f"Latent stats file not found at {stats_path}")

        with open(stats_path, 'r') as f:
            stats_data = json.load(f)

        # Pre-load statistics as tensors to avoid overhead in __getitem__
        # We assume the file contains 'lr_latent' and 'hr_latent' keys
        self.lr_mean = torch.tensor(stats_data['lr_latent']['mean'], dtype=torch.float32).view(-1, 1, 1)
        self.lr_std = torch.tensor(stats_data['lr_latent']['std'], dtype=torch.float32).view(-1, 1, 1)
        
        self.hr_mean = torch.tensor(stats_data['hr_latent']['mean'], dtype=torch.float32).view(-1, 1, 1)
        self.hr_std = torch.tensor(stats_data['hr_latent']['std'], dtype=torch.float32).view(-1, 1, 1)

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.metadata_df)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        """Return dataset sample at this index.

        Args:
            idx: Index of the dataset sample.

        Returns:
            A dataset sample containing the lr and hr image
        """
        row = self.metadata_df.iloc[idx]
        path = row['path']

        with np.load(path) as data:
            hr_latent = torch.from_numpy(data['hr_latent'])
            lr_latent = torch.from_numpy(data['lr_latent'])
            orig_image_hr = torch.from_numpy(data['hr_image'])
            orig_image_lr = torch.from_numpy(data['lr_image'])

        # Normalize latents using the loaded statistics
        # Note: We apply specific LR stats to LR latents and HR stats to HR latents
        hr_latent = (hr_latent - self.hr_mean) / self.hr_std
        lr_latent = (lr_latent - self.lr_mean) / self.lr_std

        sample = {
            'image_hr': hr_latent,
            'image_lr': lr_latent,
            'orig_image_hr': orig_image_hr,
            'orig_image_lr': orig_image_lr,
        }
        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample
    
def sen2naip_cross_sensor_collate_fn(
    batch: Sequence[dict[str, Tensor]],
) -> dict[str, Tensor]:
    """Collate function for the Sen2NaipCrossSensor dataset."""


    # LR Stats
    lr_mean = torch.tensor([1302.9685, 1085.2820, 764.7739, 2769.4824]).view(1, 4, 1, 1)
    lr_std = torch.tensor([780.8768, 513.2825, 414.3385, 793.6396]).view(1, 4, 1, 1)

    # HR Stats
    hr_mean = torch.tensor([125.1176, 121.9117, 100.0240, 143.8500]).view(1, 4, 1, 1)
    hr_std = torch.tensor([39.8066, 30.3501, 28.9109, 28.8952]).view(1, 4, 1, 1)

    # import pdb
    # pdb.set_trace()

    images_hr = torch.stack([sample['image_hr'] for sample in batch])

    # Z-score normalization for HR
    new_images_hr = (images_hr - hr_mean) / hr_std

    images_lr = torch.stack([sample['image_lr'] for sample in batch])

    # Z-score normalization for LR
    images_lr = (images_lr - lr_mean) / lr_std

    # Interpolate low res image to high res image size
    new_images_lr = F.interpolate(
        images_lr, size=images_hr.shape[-2:], mode='bicubic', align_corners=False
    )

    return {
        'image_lr': new_images_lr,
        'image_hr': new_images_hr,
        'aoi': [sample['aoi'] for sample in batch],
    }


class Sen2NaipCrossSensorDataModule(NonGeoDataModule):
    std = torch.tensor([1])

    def __init__(
        self, batch_size: int = 64, num_workers: int = 0, **kwargs: Any
    ) -> None:
        """Initialize a new Original WorldStrat DataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for parallel data loading
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.Sen2NaipCrossSensor`.
        """
        super().__init__(Sen2NaipCrossSensor, batch_size, num_workers, **kwargs)

        self.collate_fn = sen2naip_cross_sensor_collate_fn

    def setup(self, stage: str) -> None:
        """Set up datasets."""
        self.train_dataset = self.dataset_class(**self.kwargs, split='train')
        self.val_dataset = self.dataset_class(**self.kwargs, split='val')
        self.test_dataset = self.dataset_class(**self.kwargs, split='test')

    def on_after_batch_transfer(
        self, batch: dict[str, Tensor], dataloader_idx: int
    ) -> dict[str, Tensor]:
        """Hook to modify batch after transfer to device."""
        # IMPORTANT to not use torchgeo default for now
        return batch


class Sen2NaipLatentCrossSensorDataModule(NonGeoDataModule):
    def __init__(
        self, batch_size: int = 64, num_workers: int = 0, **kwargs: Any
    ) -> None:
        """Initialize a new Original WorldStrat DataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for parallel data loading
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.Sen2NaipCrossSensor`.
        """
        super().__init__(Sen2NaipCrossSensorLatent, batch_size, num_workers, **kwargs)

    def setup(self, stage: str) -> None:
        """Set up datasets."""
        self.train_dataset = self.dataset_class(split='train', **self.kwargs)
        self.val_dataset = self.dataset_class(split='val', **self.kwargs)
        self.test_dataset = self.dataset_class(split='test', **self.kwargs)

    def on_after_batch_transfer(
        self, batch: dict[str, Tensor], dataloader_idx: int
    ) -> dict[str, Tensor]:
        """Hook to modify batch after transfer to device."""
        # IMPORTANT to not use torchgeo default for now
        return batch