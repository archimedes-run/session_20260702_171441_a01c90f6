"""
Data loader for ImageNet-C dataset.
ImageNet-C consists of 15 corruption types with 5 severity levels each.
"""
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import Tuple, List, Optional
import requests
import tarfile
from pathlib import Path

# ImageNet-C corruption types
CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise',
    'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
    'snow', 'frost', 'fog', 'brightness',
    'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'
]

# ImageNet normalization statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ImageNetCDataset(Dataset):
    """
    ImageNet-C dataset loader.

    Args:
        root: Root directory containing ImageNet-C data
        corruption: Corruption type (e.g., 'gaussian_noise')
        severity: Severity level (1-5)
        transform: Optional transform to apply
    """
    def __init__(
        self,
        root: str,
        corruption: str,
        severity: int,
        transform: Optional[transforms.Compose] = None
    ):
        self.root = Path(root)
        self.corruption = corruption
        self.severity = severity
        self.transform = transform

        # Path to corruption data: root/corruption/severity/
        self.data_path = self.root / corruption / str(severity)

        if not self.data_path.exists():
            raise FileNotFoundError(
                f"ImageNet-C data not found at {self.data_path}. "
                f"Please download ImageNet-C dataset first."
            )

        # Load image paths and labels
        self.samples = []
        self.labels = []
        self._load_samples()

    def _load_samples(self):
        """Load all image paths and corresponding labels."""
        # ImageNet-C is organized as: corruption/severity/class_idx/image.JPEG
        class_folders = sorted(self.data_path.glob('*'))

        for class_folder in class_folders:
            if not class_folder.is_dir():
                continue

            try:
                class_idx = int(class_folder.name)
            except ValueError:
                continue

            # Get all images in this class folder
            images = list(class_folder.glob('*.JPEG')) + list(class_folder.glob('*.png'))

            for img_path in images:
                self.samples.append(img_path)
                self.labels.append(class_idx)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.samples[idx]
        label = self.labels[idx]

        # Load image
        img = Image.open(img_path).convert('RGB')

        # Apply transforms
        if self.transform:
            img = self.transform(img)

        return img, label


def get_imagenet_c_transforms(input_size: int = 224) -> transforms.Compose:
    """
    Get standard ImageNet-C transforms.

    Args:
        input_size: Input image size (default: 224 for ViT-Base)

    Returns:
        Transform pipeline
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def create_imagenet_c_loader(
    root: str,
    corruption: str,
    severity: int,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = False
) -> DataLoader:
    """
    Create a DataLoader for ImageNet-C.

    Args:
        root: Root directory containing ImageNet-C data
        corruption: Corruption type
        severity: Severity level (1-5)
        batch_size: Batch size
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle data

    Returns:
        DataLoader instance
    """
    transform = get_imagenet_c_transforms()
    dataset = ImageNetCDataset(root, corruption, severity, transform)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )

    return loader


def check_imagenet_c_availability(root: str) -> bool:
    """
    Check if ImageNet-C dataset is available.

    Args:
        root: Root directory to check

    Returns:
        True if dataset is found, False otherwise
    """
    root_path = Path(root)
    if not root_path.exists():
        return False

    # Check for at least one corruption type
    for corruption in CORRUPTIONS:
        corruption_path = root_path / corruption / "1"
        if corruption_path.exists():
            return True

    return False


def print_imagenet_c_instructions():
    """Print instructions for downloading ImageNet-C."""
    instructions = """
    ImageNet-C Dataset Instructions:
    ================================

    ImageNet-C is not automatically downloadable due to licensing.
    You need to manually download it:

    1. Visit: https://zenodo.org/record/2235448
    2. Download the ImageNet-C dataset
    3. Extract it to a directory (e.g., ./data/imagenet-c/)
    4. The structure should be: data/imagenet-c/corruption_type/severity/class/images

    Alternatively, for research purposes, you can use a smaller validation set
    or work with the authors to obtain access.

    For this replication, we need the full ImageNet-C dataset with all:
    - 15 corruption types
    - 5 severity levels
    - 50,000 validation images per corruption
    """
    print(instructions)


if __name__ == "__main__":
    # Test the data loader
    print("Testing ImageNet-C data loader...")

    # Check if data is available
    data_root = "./data/imagenet-c"

    if not check_imagenet_c_availability(data_root):
        print("ImageNet-C data not found.")
        print_imagenet_c_instructions()
    else:
        print(f"ImageNet-C data found at {data_root}")

        # Test loading a corruption
        loader = create_imagenet_c_loader(
            data_root,
            corruption='gaussian_noise',
            severity=1,
            batch_size=32
        )

        print(f"Dataset size: {len(loader.dataset)}")
        print(f"Number of batches: {len(loader)}")

        # Test loading one batch
        images, labels = next(iter(loader))
        print(f"Batch shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        print("Data loader test successful!")
