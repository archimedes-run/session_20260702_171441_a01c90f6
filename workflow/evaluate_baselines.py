"""
Comprehensive evaluation script for Source and TENT baselines.

This script evaluates both baselines across all ImageNet-C corruptions
and severity levels, saving results and generating comparison plots.
"""
import torch
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List
import sys

from data_loader import CORRUPTIONS, check_imagenet_c_availability, print_imagenet_c_instructions
from source_baseline import SourceModel
from tent_baseline import TENT


def evaluate_all_corruptions(
    data_root: str,
    method: str,
    batch_size: int = 64,
    device: str = 'cuda',
    results_dir: str = '../results'
) -> pd.DataFrame:
    """
    Evaluate a method on all ImageNet-C corruptions and severities.

    Args:
        data_root: Root directory for ImageNet-C
        method: 'source' or 'tent'
        batch_size: Batch size for evaluation
        device: Device to run on
        results_dir: Directory to save results

    Returns:
        DataFrame with results
    """
    from data_loader import create_imagenet_c_loader

    results_dir = Path(results_dir)
    results_dir.mkdir(exist_ok=True, parents=True)

    # Initialize model based on method
    if method.lower() == 'source':
        print("Initializing Source baseline...")
        model = SourceModel(device=device)
    elif method.lower() == 'tent':
        print("Initializing TENT baseline...")
        model = TENT(device=device, learning_rate=1e-3)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Store all results
    all_results = []

    # Evaluate on each corruption and severity
    for corruption in CORRUPTIONS:
        print(f"\n{'='*80}")
        print(f"Evaluating {method.upper()} on: {corruption}")
        print(f"{'='*80}")

        for severity in range(1, 6):
            print(f"\nSeverity {severity}/5...")

            try:
                # Create data loader
                data_loader = create_imagenet_c_loader(
                    data_root,
                    corruption,
                    severity,
                    batch_size=batch_size,
                    num_workers=4,
                    shuffle=False
                )

                # Evaluate
                results = model.evaluate(data_loader)

                # Add metadata
                results['corruption'] = corruption
                results['severity'] = severity
                results['method'] = method

                all_results.append(results)

                print(f"  Accuracy: {results['accuracy']:.2f}%")

                # Reset model for next evaluation (important for TENT)
                model.reset()

            except Exception as e:
                print(f"Error evaluating {corruption} severity {severity}: {e}")
                continue

    # Convert to DataFrame
    df = pd.DataFrame(all_results)

    # Save results
    results_file = results_dir / f'{method}_results.csv'
    df.to_csv(results_file, index=False)
    print(f"\nResults saved to: {results_file}")

    # Save as JSON as well
    json_file = results_dir / f'{method}_results.json'
    with open(json_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    return df


def compare_methods(
    source_df: pd.DataFrame,
    tent_df: pd.DataFrame,
    results_dir: str = '../results'
) -> None:
    """
    Generate comparison plots and statistics.

    Args:
        source_df: DataFrame with Source results
        tent_df: DataFrame with TENT results
        results_dir: Directory to save plots
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(exist_ok=True, parents=True)

    # Set matplotlib to non-interactive backend
    plt.switch_backend('Agg')

    # 1. Accuracy vs Severity for each corruption
    fig, axes = plt.subplots(5, 3, figsize=(15, 20))
    axes = axes.flatten()

    for i, corruption in enumerate(CORRUPTIONS):
        ax = axes[i]

        source_data = source_df[source_df['corruption'] == corruption]
        tent_data = tent_df[tent_df['corruption'] == corruption]

        ax.plot(source_data['severity'], source_data['accuracy'],
                marker='o', label='Source', linewidth=2)
        ax.plot(tent_data['severity'], tent_data['accuracy'],
                marker='s', label='TENT', linewidth=2)

        ax.set_title(corruption.replace('_', ' ').title())
        ax.set_xlabel('Severity')
        ax.set_ylabel('Accuracy (%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks([1, 2, 3, 4, 5])

    plt.tight_layout()
    plt.savefig(results_dir / 'accuracy_vs_severity.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {results_dir / 'accuracy_vs_severity.png'}")

    # 2. Average accuracy across all corruptions by severity
    fig, ax = plt.subplots(figsize=(10, 6))

    source_by_severity = source_df.groupby('severity')['accuracy'].mean()
    tent_by_severity = tent_df.groupby('severity')['accuracy'].mean()

    ax.plot(source_by_severity.index, source_by_severity.values,
            marker='o', label='Source', linewidth=3, markersize=10)
    ax.plot(tent_by_severity.index, tent_by_severity.values,
            marker='s', label='TENT', linewidth=3, markersize=10)

    ax.set_title('Average Accuracy vs Severity (All Corruptions)', fontsize=14)
    ax.set_xlabel('Severity', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xticks([1, 2, 3, 4, 5])

    plt.tight_layout()
    plt.savefig(results_dir / 'average_accuracy_vs_severity.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {results_dir / 'average_accuracy_vs_severity.png'}")

    # 3. Heatmap of accuracy improvement (TENT - Source)
    improvement_matrix = np.zeros((len(CORRUPTIONS), 5))

    for i, corruption in enumerate(CORRUPTIONS):
        for severity in range(1, 6):
            source_acc = source_df[
                (source_df['corruption'] == corruption) &
                (source_df['severity'] == severity)
            ]['accuracy'].values

            tent_acc = tent_df[
                (tent_df['corruption'] == corruption) &
                (tent_df['severity'] == severity)
            ]['accuracy'].values

            if len(source_acc) > 0 and len(tent_acc) > 0:
                improvement_matrix[i, severity-1] = tent_acc[0] - source_acc[0]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        improvement_matrix,
        annot=True,
        fmt='.2f',
        cmap='RdYlGn',
        center=0,
        xticklabels=[1, 2, 3, 4, 5],
        yticklabels=[c.replace('_', ' ').title() for c in CORRUPTIONS],
        cbar_kws={'label': 'Accuracy Improvement (%)'},
        ax=ax
    )
    ax.set_xlabel('Severity', fontsize=12)
    ax.set_ylabel('Corruption', fontsize=12)
    ax.set_title('TENT Accuracy Improvement over Source (%)', fontsize=14)

    plt.tight_layout()
    plt.savefig(results_dir / 'improvement_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {results_dir / 'improvement_heatmap.png'}")

    # 4. Summary statistics
    summary = {
        'source': {
            'mean_accuracy': source_df['accuracy'].mean(),
            'std_accuracy': source_df['accuracy'].std(),
            'mean_error': source_df['error_rate'].mean(),
            'mean_entropy': source_df['avg_entropy'].mean(),
        },
        'tent': {
            'mean_accuracy': tent_df['accuracy'].mean(),
            'std_accuracy': tent_df['accuracy'].std(),
            'mean_error': tent_df['error_rate'].mean(),
            'mean_entropy': tent_df['avg_entropy'].mean(),
        },
        'improvement': {
            'mean_accuracy_gain': tent_df['accuracy'].mean() - source_df['accuracy'].mean(),
            'mean_error_reduction': source_df['error_rate'].mean() - tent_df['error_rate'].mean(),
        }
    }

    summary_file = results_dir / 'summary_statistics.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved: {summary_file}")
    print("\nSummary Statistics:")
    print(f"  Source: {summary['source']['mean_accuracy']:.2f}% ± {summary['source']['std_accuracy']:.2f}%")
    print(f"  TENT:   {summary['tent']['mean_accuracy']:.2f}% ± {summary['tent']['std_accuracy']:.2f}%")
    print(f"  Improvement: +{summary['improvement']['mean_accuracy_gain']:.2f}%")


def main():
    parser = argparse.ArgumentParser(description='Evaluate baselines on ImageNet-C')
    parser.add_argument('--data_root', type=str, default='./data/imagenet-c',
                        help='Root directory for ImageNet-C')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda, mps, or cpu)')
    parser.add_argument('--results_dir', type=str, default='../results',
                        help='Directory to save results')
    parser.add_argument('--method', type=str, default='both',
                        choices=['source', 'tent', 'both'],
                        help='Which method to evaluate')

    args = parser.parse_args()

    # Auto-detect device
    if args.device == 'cuda' and not torch.cuda.is_available():
        if torch.backends.mps.is_available():
            args.device = 'mps'
        else:
            args.device = 'cpu'

    print(f"Using device: {args.device}")

    # Check if ImageNet-C is available
    if not check_imagenet_c_availability(args.data_root):
        print("\nERROR: ImageNet-C dataset not found!")
        print_imagenet_c_instructions()
        print("\n[HALT_ROUTINE]")
        print("FAILURE TRACE: ImageNet-C dataset is not available at the specified path.")
        print(f"Expected path: {args.data_root}")
        print("REASON: ImageNet-C requires manual download due to licensing restrictions.")
        print("SCIENTIFIC VALIDITY: Cannot use synthetic or proxy data for this benchmark.")
        print("REQUIRED ACTION: Download ImageNet-C from https://zenodo.org/record/2235448")
        sys.exit(1)

    # Evaluate methods
    if args.method in ['source', 'both']:
        print("\n" + "="*80)
        print("EVALUATING SOURCE BASELINE")
        print("="*80)
        source_df = evaluate_all_corruptions(
            args.data_root,
            'source',
            args.batch_size,
            args.device,
            args.results_dir
        )
    else:
        # Load existing results
        source_df = pd.read_csv(Path(args.results_dir) / 'source_results.csv')

    if args.method in ['tent', 'both']:
        print("\n" + "="*80)
        print("EVALUATING TENT BASELINE")
        print("="*80)
        tent_df = evaluate_all_corruptions(
            args.data_root,
            'tent',
            args.batch_size,
            args.device,
            args.results_dir
        )
    else:
        # Load existing results
        tent_df = pd.read_csv(Path(args.results_dir) / 'tent_results.csv')

    # Generate comparison plots
    if args.method == 'both' or (
        (Path(args.results_dir) / 'source_results.csv').exists() and
        (Path(args.results_dir) / 'tent_results.csv').exists()
    ):
        print("\n" + "="*80)
        print("GENERATING COMPARISON PLOTS")
        print("="*80)
        compare_methods(source_df, tent_df, args.results_dir)

    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
