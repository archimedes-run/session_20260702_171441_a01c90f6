"""
Pre-compute Source Domain Statistics for FOA

This script computes the mean and standard deviation of [CLS] token activations
from clean ImageNet samples. These statistics are used for the Back-to-Source
Activation Shifting mechanism in FOA.

As per the paper, we use 32 unlabeled in-distribution samples.
"""

import torch
import torch.nn as nn
import timm
import argparse
import os
import sys
from tqdm import tqdm
import numpy as np
import random

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from foa_method import compute_source_statistics, save_source_statistics, ActivationHook


def set_random_seeds(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def get_imagenet_val_loader(data_root: str, batch_size: int = 8, num_workers: int = 2):
    """
    Create a DataLoader for clean ImageNet validation set.

    Args:
        data_root: Path to ImageNet validation set
        batch_size: Batch size
        num_workers: Number of workers for data loading

    Returns:
        DataLoader
    """
    from torchvision import datasets, transforms

    # Standard ImageNet preprocessing
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(data_root, transform=transform)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return loader


def create_synthetic_source_loader(num_samples: int = 32, batch_size: int = 8):
    """
    Create synthetic source data for testing when ImageNet is not available.

    Args:
        num_samples: Number of synthetic samples
        batch_size: Batch size

    Returns:
        DataLoader
    """
    print(f"[WARNING] Creating {num_samples} synthetic source samples for testing.")
    print("[WARNING] For actual experiments, use real ImageNet validation data!")

    # Generate random normalized images
    data = torch.randn(num_samples, 3, 224, 224)
    # Apply ImageNet normalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    data = (data - mean) / std

    labels = torch.zeros(num_samples, dtype=torch.long)

    dataset = torch.utils.data.TensorDataset(data, labels)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return loader


def get_vit_layer_names(model: nn.Module) -> list:
    """
    Get layer names for ViT blocks to track.

    Args:
        model: ViT model

    Returns:
        List of layer names
    """
    # For ViT-Base, we have 12 transformer blocks
    # We'll track every block's output
    layer_names = []

    for i in range(12):
        layer_names.append(f'blocks.{i}')

    return layer_names


def main():
    parser = argparse.ArgumentParser(description='Compute source domain statistics for FOA')
    parser.add_argument('--data_root', type=str, default=None,
                        help='Path to ImageNet validation set (if available)')
    parser.add_argument('--num_samples', type=int, default=32,
                        help='Number of source samples to use (default: 32)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for processing')
    parser.add_argument('--output', type=str, default='../results/source_statistics.pth',
                        help='Output path for statistics')
    parser.add_argument('--model', type=str, default='vit_base_patch16_224',
                        help='Model architecture')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto/cuda/mps/cpu)')
    parser.add_argument('--use_synthetic', action='store_true',
                        help='Use synthetic data (for testing only)')
    args = parser.parse_args()

    # Set random seeds
    set_random_seeds(42)

    # Determine device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    else:
        device = args.device

    print("=" * 80)
    print("Computing Source Domain Statistics for FOA")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Number of samples: {args.num_samples}")
    print(f"Batch size: {args.batch_size}")
    print()

    # Load model
    print("Loading pre-trained ViT model...")
    model = timm.create_model(args.model, pretrained=True)
    model = model.to(device)
    model.eval()
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model loaded: {num_params:.2f}M parameters")
    print()

    # Get layer names
    layer_names = get_vit_layer_names(model)
    print(f"Tracking {len(layer_names)} layers:")
    for name in layer_names:
        print(f"  - {name}")
    print()

    # Create data loader
    if args.use_synthetic or args.data_root is None:
        print("[INFO] Using synthetic source data for testing")
        print("[WARNING] For real experiments, provide --data_root with ImageNet validation set")
        loader = create_synthetic_source_loader(args.num_samples, args.batch_size)
    else:
        print(f"[INFO] Loading ImageNet validation data from: {args.data_root}")
        if not os.path.exists(args.data_root):
            print(f"[ERROR] Data root not found: {args.data_root}")
            print("[HALT_ROUTINE] ImageNet validation set required but not found.")
            print("Please provide path to ImageNet validation set with --data_root")
            sys.exit(1)
        loader = get_imagenet_val_loader(args.data_root, args.batch_size)

    # Compute statistics
    print("Computing source domain statistics...")
    print(f"This will process up to {args.num_samples} samples")
    print()

    source_stats = compute_source_statistics(
        model=model,
        dataloader=loader,
        layer_names=layer_names,
        device=device,
        num_samples=args.num_samples
    )

    # Print statistics summary
    print("\nSource Statistics Summary:")
    print("-" * 80)
    for layer_name, stats in source_stats.items():
        mean = stats['mean']
        std = stats['std']
        print(f"{layer_name}:")
        print(f"  Mean: shape={mean.shape}, min={mean.min():.4f}, max={mean.max():.4f}, avg={mean.mean():.4f}")
        print(f"  Std:  shape={std.shape}, min={std.min():.4f}, max={std.max():.4f}, avg={std.mean():.4f}")

    # Save statistics
    print("\nSaving statistics...")
    save_source_statistics(source_stats, args.output)

    print("\n" + "=" * 80)
    print(f"Source statistics computation complete!")
    print(f"Statistics saved to: {args.output}")
    print("=" * 80)


if __name__ == "__main__":
    main()
