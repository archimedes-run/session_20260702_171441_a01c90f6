"""
Evaluate FOA on ImageNet-C

This script evaluates the Forward-Optimization Adaptation (FOA) method
on all ImageNet-C corruptions and severity levels.
"""

import torch
import timm
import argparse
import os
import sys
import json
import pandas as pd
import numpy as np
import random
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from foa_method import FOAAdapter, load_source_statistics
from data_loader import get_imagenet_c_loader


def set_random_seeds(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def evaluate_foa_on_corruption(
    foa_adapter: FOAAdapter,
    corruption: str,
    severity: int,
    data_root: str,
    batch_size: int = 64,
    max_batches: int = None
) -> dict:
    """
    Evaluate FOA on a single corruption and severity.

    Args:
        foa_adapter: FOA adapter instance
        corruption: Corruption type
        severity: Severity level (1-5)
        data_root: Root directory of ImageNet-C
        batch_size: Batch size
        max_batches: Maximum number of batches to process (for testing)

    Returns:
        Dictionary with results
    """
    # Reset adapter
    foa_adapter.reset()

    # Get data loader
    loader = get_imagenet_c_loader(
        data_root, corruption, severity, batch_size, shuffle=False
    )

    # Metrics
    total_correct = 0
    total_samples = 0
    total_entropy = 0.0
    total_activation_disc = 0.0
    total_fitness = 0.0

    # Evaluate
    pbar = tqdm(loader, desc=f"{corruption} severity {severity}")
    for batch_idx, (images, labels) in enumerate(pbar):
        if max_batches and batch_idx >= max_batches:
            break

        # Adapt and predict
        logits, info = foa_adapter.adapt_batch(images, labels)

        # Compute accuracy
        preds = logits.argmax(dim=1)
        labels = labels.to(logits.device)
        correct = (preds == labels).sum().item()

        total_correct += correct
        total_samples += images.size(0)
        total_entropy += info['entropy'] * images.size(0)
        total_activation_disc += info['activation_discrepancy'] * images.size(0)
        total_fitness += info['fitness'] * images.size(0)

        # Update progress bar
        current_acc = total_correct / total_samples
        pbar.set_postfix({
            'acc': f'{current_acc:.4f}',
            'entropy': f'{info["entropy"]:.4f}',
            'act_disc': f'{info["activation_discrepancy"]:.4f}'
        })

    # Compute averages
    accuracy = total_correct / total_samples
    avg_entropy = total_entropy / total_samples
    avg_activation_disc = total_activation_disc / total_samples
    avg_fitness = total_fitness / total_samples
    error_rate = 1.0 - accuracy

    return {
        'corruption': corruption,
        'severity': severity,
        'accuracy': accuracy,
        'error_rate': error_rate,
        'entropy': avg_entropy,
        'activation_discrepancy': avg_activation_disc,
        'fitness': avg_fitness,
        'total_samples': total_samples
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate FOA on ImageNet-C')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Path to ImageNet-C dataset')
    parser.add_argument('--source_stats', type=str, default='../results/source_statistics.pth',
                        help='Path to pre-computed source statistics')
    parser.add_argument('--corruption', type=str, default=None,
                        help='Single corruption to evaluate (default: all)')
    parser.add_argument('--severity', type=int, default=None,
                        help='Single severity to evaluate (default: all)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--num_prompts', type=int, default=10,
                        help='Number of prompt embeddings')
    parser.add_argument('--lambda_activation', type=float, default=0.1,
                        help='Weight for activation discrepancy term')
    parser.add_argument('--cma_population', type=int, default=10,
                        help='CMA-ES population size')
    parser.add_argument('--cma_iterations', type=int, default=20,
                        help='CMA-ES max iterations')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto/cuda/mps/cpu)')
    parser.add_argument('--output', type=str, default='../results',
                        help='Output directory')
    parser.add_argument('--max_batches', type=int, default=None,
                        help='Maximum batches per corruption (for testing)')
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
    print("Evaluating FOA on ImageNet-C")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Data root: {args.data_root}")
    print(f"Source statistics: {args.source_stats}")
    print(f"Batch size: {args.batch_size}")
    print(f"Number of prompts: {args.num_prompts}")
    print(f"Lambda (activation): {args.lambda_activation}")
    print(f"CMA-ES population: {args.cma_population}")
    print(f"CMA-ES iterations: {args.cma_iterations}")
    print()

    # Load model
    print("Loading pre-trained ViT model...")
    model = timm.create_model('vit_base_patch16_224', pretrained=True)
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model loaded: {num_params:.2f}M parameters")
    print()

    # Load source statistics
    print("Loading source statistics...")
    if not os.path.exists(args.source_stats):
        print(f"[ERROR] Source statistics not found: {args.source_stats}")
        print("[HALT_ROUTINE] Source statistics required but not found.")
        print("Please run compute_source_stats.py first to generate source statistics.")
        sys.exit(1)

    source_stats = load_source_statistics(args.source_stats)
    print(f"Loaded statistics for {len(source_stats)} layers")
    print()

    # Initialize FOA adapter
    print("Initializing FOA adapter...")
    foa_adapter = FOAAdapter(
        model=model,
        source_stats=source_stats,
        num_prompts=args.num_prompts,
        lambda_activation=args.lambda_activation,
        cma_population_size=args.cma_population,
        cma_max_iterations=args.cma_iterations,
        device=device
    )
    print("FOA adapter initialized")
    print()

    # Define corruptions and severities
    all_corruptions = [
        'gaussian_noise', 'shot_noise', 'impulse_noise',
        'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
        'snow', 'frost', 'fog', 'brightness',
        'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'
    ]

    corruptions = [args.corruption] if args.corruption else all_corruptions
    severities = [args.severity] if args.severity else [1, 2, 3, 4, 5]

    # Evaluate
    results = []
    total_combinations = len(corruptions) * len(severities)

    print(f"Evaluating {len(corruptions)} corruptions × {len(severities)} severities = {total_combinations} combinations")
    print("=" * 80)
    print()

    for corruption in corruptions:
        for severity in severities:
            print(f"\nEvaluating: {corruption} (severity {severity})")
            print("-" * 80)

            result = evaluate_foa_on_corruption(
                foa_adapter, corruption, severity, args.data_root,
                args.batch_size, args.max_batches
            )

            results.append(result)

            print(f"Results: Accuracy={result['accuracy']:.4f}, "
                  f"Error={result['error_rate']:.4f}, "
                  f"Entropy={result['entropy']:.4f}, "
                  f"ActDisc={result['activation_discrepancy']:.4f}")

    # Save results
    print("\n" + "=" * 80)
    print("Saving results...")

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Save as CSV
    df = pd.DataFrame(results)
    csv_path = os.path.join(args.output, 'foa_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"CSV saved to: {csv_path}")

    # Save as JSON
    json_path = os.path.join(args.output, 'foa_results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"JSON saved to: {json_path}")

    # Compute summary statistics
    avg_accuracy = df['accuracy'].mean()
    avg_error = df['error_rate'].mean()
    std_accuracy = df['accuracy'].std()

    summary = {
        'average_accuracy': avg_accuracy,
        'average_error_rate': avg_error,
        'std_accuracy': std_accuracy,
        'num_corruptions': len(corruptions),
        'num_severities': len(severities),
        'total_evaluations': len(results),
        'hyperparameters': {
            'num_prompts': args.num_prompts,
            'lambda_activation': args.lambda_activation,
            'cma_population': args.cma_population,
            'cma_iterations': args.cma_iterations
        }
    }

    summary_path = os.path.join(args.output, 'foa_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_path}")

    print("\n" + "=" * 80)
    print("FOA Evaluation Summary")
    print("=" * 80)
    print(f"Average Accuracy: {avg_accuracy:.4f} ± {std_accuracy:.4f}")
    print(f"Average Error Rate: {avg_error:.4f}")
    print(f"Total Evaluations: {len(results)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
