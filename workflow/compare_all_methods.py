"""
Comprehensive Comparison: Source, TENT, and FOA

This script evaluates all three methods on ImageNet-C and generates
comparative visualizations and ablation studies.
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
import matplotlib.pyplot as plt
import seaborn as sns

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


def evaluate_source(model, corruption, severity, data_root, batch_size, device, max_batches=None):
    """Evaluate source model (no adaptation)."""
    model.eval()
    loader = get_imagenet_c_loader(data_root, corruption, severity, batch_size, shuffle=False)

    total_correct = 0
    total_samples = 0
    total_entropy = 0.0

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(loader, desc=f"Source: {corruption}-{severity}")):
            if max_batches and batch_idx >= max_batches:
                break

            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            preds = logits.argmax(dim=1)

            total_correct += (preds == labels).sum().item()
            total_samples += images.size(0)

            # Compute entropy
            probs = torch.softmax(logits, dim=1)
            log_probs = torch.log(probs + 1e-10)
            entropy = -(probs * log_probs).sum(dim=1).mean()
            total_entropy += entropy.item() * images.size(0)

    accuracy = total_correct / total_samples
    avg_entropy = total_entropy / total_samples

    return {
        'method': 'Source',
        'corruption': corruption,
        'severity': severity,
        'accuracy': accuracy,
        'error_rate': 1.0 - accuracy,
        'entropy': avg_entropy
    }


def evaluate_tent(model, corruption, severity, data_root, batch_size, device, lr=1e-3, max_batches=None):
    """Evaluate TENT (entropy minimization)."""
    from tent_baseline import configure_tent_model

    # Reset model
    model.load_state_dict(timm.create_model('vit_base_patch16_224', pretrained=True).state_dict())
    model = configure_tent_model(model, lr=lr)
    model.train()  # TENT requires train mode

    loader = get_imagenet_c_loader(data_root, corruption, severity, batch_size, shuffle=False)

    total_correct = 0
    total_samples = 0
    total_entropy = 0.0

    for batch_idx, (images, labels) in enumerate(tqdm(loader, desc=f"TENT: {corruption}-{severity}")):
        if max_batches and batch_idx >= max_batches:
            break

        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        logits = model(images)

        # TENT: minimize entropy
        probs = torch.softmax(logits, dim=1)
        log_probs = torch.log(probs + 1e-10)
        entropy_loss = -(probs * log_probs).sum(dim=1).mean()

        # Backward and update
        model.optimizer.zero_grad()
        entropy_loss.backward()
        model.optimizer.step()

        # Evaluation
        with torch.no_grad():
            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += images.size(0)
            total_entropy += entropy_loss.item() * images.size(0)

    accuracy = total_correct / total_samples
    avg_entropy = total_entropy / total_samples

    return {
        'method': 'TENT',
        'corruption': corruption,
        'severity': severity,
        'accuracy': accuracy,
        'error_rate': 1.0 - accuracy,
        'entropy': avg_entropy
    }


def evaluate_foa(foa_adapter, corruption, severity, data_root, batch_size, max_batches=None):
    """Evaluate FOA."""
    foa_adapter.reset()
    loader = get_imagenet_c_loader(data_root, corruption, severity, batch_size, shuffle=False)

    total_correct = 0
    total_samples = 0
    total_entropy = 0.0
    total_activation_disc = 0.0

    for batch_idx, (images, labels) in enumerate(tqdm(loader, desc=f"FOA: {corruption}-{severity}")):
        if max_batches and batch_idx >= max_batches:
            break

        logits, info = foa_adapter.adapt_batch(images, labels)

        preds = logits.argmax(dim=1)
        labels = labels.to(logits.device)

        total_correct += (preds == labels).sum().item()
        total_samples += images.size(0)
        total_entropy += info['entropy'] * images.size(0)
        total_activation_disc += info['activation_discrepancy'] * images.size(0)

    accuracy = total_correct / total_samples
    avg_entropy = total_entropy / total_samples
    avg_activation_disc = total_activation_disc / total_samples

    return {
        'method': 'FOA',
        'corruption': corruption,
        'severity': severity,
        'accuracy': accuracy,
        'error_rate': 1.0 - accuracy,
        'entropy': avg_entropy,
        'activation_discrepancy': avg_activation_disc
    }


def plot_comparison(df, output_dir):
    """Generate comparison visualizations."""
    os.makedirs(output_dir, exist_ok=True)

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 300

    # 1. Accuracy vs Severity by Method
    fig, axes = plt.subplots(5, 3, figsize=(15, 20))
    axes = axes.flatten()

    corruptions = df['corruption'].unique()
    for idx, corruption in enumerate(corruptions):
        if idx >= len(axes):
            break

        ax = axes[idx]
        corruption_data = df[df['corruption'] == corruption]

        for method in ['Source', 'TENT', 'FOA']:
            method_data = corruption_data[corruption_data['method'] == method]
            if not method_data.empty:
                ax.plot(method_data['severity'], method_data['accuracy'],
                       marker='o', label=method, linewidth=2)

        ax.set_title(corruption.replace('_', ' ').title(), fontsize=10)
        ax.set_xlabel('Severity')
        ax.set_ylabel('Accuracy')
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_vs_severity_all.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: accuracy_vs_severity_all.png")

    # 2. Average Accuracy by Method
    fig, ax = plt.subplots(figsize=(10, 6))

    avg_by_method_severity = df.groupby(['method', 'severity'])['accuracy'].mean().reset_index()

    for method in ['Source', 'TENT', 'FOA']:
        method_data = avg_by_method_severity[avg_by_method_severity['method'] == method]
        ax.plot(method_data['severity'], method_data['accuracy'],
               marker='o', label=method, linewidth=3, markersize=8)

    ax.set_title('Average Accuracy Across All Corruptions', fontsize=14, fontweight='bold')
    ax.set_xlabel('Severity', fontsize=12)
    ax.set_ylabel('Average Accuracy', fontsize=12)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'average_accuracy_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: average_accuracy_comparison.png")

    # 3. Method Improvement Heatmap
    # Calculate improvement of TENT and FOA over Source
    source_data = df[df['method'] == 'Source'].set_index(['corruption', 'severity'])['accuracy']

    improvements = {}
    for method in ['TENT', 'FOA']:
        method_data = df[df['method'] == method].set_index(['corruption', 'severity'])['accuracy']
        improvement = method_data - source_data
        improvements[method] = improvement

    # Create heatmap for each method
    for method, improvement in improvements.items():
        improvement_df = improvement.unstack()

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(improvement_df, annot=True, fmt='.3f', cmap='RdYlGn',
                   center=0, vmin=-0.1, vmax=0.1, ax=ax, cbar_kws={'label': 'Accuracy Improvement'})
        ax.set_title(f'{method} Improvement over Source', fontsize=14, fontweight='bold')
        ax.set_xlabel('Severity', fontsize=12)
        ax.set_ylabel('Corruption', fontsize=12)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{method.lower()}_improvement_heatmap.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {method.lower()}_improvement_heatmap.png")

    # 4. Summary Bar Chart
    avg_by_method = df.groupby('method')['accuracy'].agg(['mean', 'std']).reset_index()

    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(len(avg_by_method))
    bars = ax.bar(x, avg_by_method['mean'], yerr=avg_by_method['std'],
                 capsize=5, alpha=0.7, color=['#1f77b4', '#ff7f0e', '#2ca02c'])

    ax.set_ylabel('Average Accuracy', fontsize=12)
    ax.set_title('Overall Method Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(avg_by_method['method'], fontsize=12)
    ax.set_ylim(0, 1)
    ax.grid(True, axis='y', alpha=0.3)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.4f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'method_comparison_bar.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: method_comparison_bar.png")


def run_ablation_studies(model, source_stats, data_root, batch_size, device, output_dir):
    """
    Run ablation studies on FOA components.

    Ablations:
    1. FOA without prompts (only activation shifting)
    2. FOA without activation shifting (only prompts)
    3. FOA with different lambda values
    4. FOA with different prompt lengths
    """
    print("\n" + "=" * 80)
    print("Running Ablation Studies")
    print("=" * 80)

    # Test on a subset of corruptions
    test_corruptions = ['gaussian_noise', 'defocus_blur', 'brightness']
    test_severity = 3

    results = []

    # Ablation 1: Lambda values
    print("\nAblation 1: Effect of Lambda (activation weight)")
    lambda_values = [0.0, 0.01, 0.05, 0.1, 0.5, 1.0]

    for lambda_val in lambda_values:
        print(f"\nTesting lambda = {lambda_val}")

        foa = FOAAdapter(
            model=model,
            source_stats=source_stats,
            num_prompts=10,
            lambda_activation=lambda_val,
            cma_population_size=8,
            cma_max_iterations=15,
            device=device
        )

        for corruption in test_corruptions:
            result = evaluate_foa(foa, corruption, test_severity, data_root, batch_size, max_batches=5)
            result['ablation'] = f'lambda={lambda_val}'
            result['component'] = 'lambda'
            results.append(result)

    # Ablation 2: Prompt lengths
    print("\nAblation 2: Effect of Prompt Length")
    prompt_lengths = [0, 5, 10, 20, 50]

    for num_prompts in prompt_lengths:
        print(f"\nTesting num_prompts = {num_prompts}")

        if num_prompts == 0:
            # Skip CMA-ES, only use activation shifting
            continue

        foa = FOAAdapter(
            model=model,
            source_stats=source_stats,
            num_prompts=num_prompts,
            lambda_activation=0.1,
            cma_population_size=8,
            cma_max_iterations=15,
            device=device
        )

        for corruption in test_corruptions:
            result = evaluate_foa(foa, corruption, test_severity, data_root, batch_size, max_batches=5)
            result['ablation'] = f'prompts={num_prompts}'
            result['component'] = 'prompts'
            results.append(result)

    # Save ablation results
    os.makedirs(output_dir, exist_ok=True)

    df_ablation = pd.DataFrame(results)
    ablation_csv = os.path.join(output_dir, 'ablation_results.csv')
    df_ablation.to_csv(ablation_csv, index=False)
    print(f"\nAblation results saved to: {ablation_csv}")

    # Plot ablation results
    # Lambda ablation
    lambda_data = df_ablation[df_ablation['component'] == 'lambda']
    lambda_avg = lambda_data.groupby('ablation')['accuracy'].mean().reset_index()

    fig, ax = plt.subplots(figsize=(10, 6))
    lambda_vals = [float(x.split('=')[1]) for x in lambda_avg['ablation']]
    ax.plot(lambda_vals, lambda_avg['accuracy'], marker='o', linewidth=2, markersize=8)
    ax.set_xlabel('Lambda (Activation Weight)', fontsize=12)
    ax.set_ylabel('Average Accuracy', fontsize=12)
    ax.set_title('Ablation: Effect of Lambda Parameter', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ablation_lambda.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: ablation_lambda.png")

    # Prompt length ablation
    prompt_data = df_ablation[df_ablation['component'] == 'prompts']
    prompt_avg = prompt_data.groupby('ablation')['accuracy'].mean().reset_index()

    fig, ax = plt.subplots(figsize=(10, 6))
    prompt_lens = [int(x.split('=')[1]) for x in prompt_avg['ablation']]
    ax.plot(prompt_lens, prompt_avg['accuracy'], marker='o', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Prompts', fontsize=12)
    ax.set_ylabel('Average Accuracy', fontsize=12)
    ax.set_title('Ablation: Effect of Prompt Length', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ablation_prompts.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: ablation_prompts.png")

    return df_ablation


def main():
    parser = argparse.ArgumentParser(description='Compare all methods and run ablations')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Path to ImageNet-C dataset')
    parser.add_argument('--source_stats', type=str, default='../results/source_statistics.pth',
                        help='Path to source statistics')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (auto/cuda/mps/cpu)')
    parser.add_argument('--output', type=str, default='../results',
                        help='Output directory')
    parser.add_argument('--max_batches', type=int, default=None,
                        help='Max batches per corruption (testing)')
    parser.add_argument('--skip_ablations', action='store_true',
                        help='Skip ablation studies')
    parser.add_argument('--test_only', action='store_true',
                        help='Test on subset of corruptions')
    args = parser.parse_args()

    set_random_seeds(42)

    # Device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    else:
        device = args.device

    print("=" * 80)
    print("Comprehensive Method Comparison")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Data root: {args.data_root}")
    print()

    # Load model
    print("Loading model...")
    model = timm.create_model('vit_base_patch16_224', pretrained=True)
    model = model.to(device)
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")

    # Load source stats
    print("Loading source statistics...")
    source_stats = load_source_statistics(args.source_stats)
    print(f"Loaded statistics for {len(source_stats)} layers")

    # Initialize FOA
    foa_adapter = FOAAdapter(
        model=model,
        source_stats=source_stats,
        num_prompts=10,
        lambda_activation=0.1,
        cma_population_size=10,
        cma_max_iterations=20,
        device=device
    )

    # Define corruptions
    if args.test_only:
        corruptions = ['gaussian_noise', 'defocus_blur', 'brightness']
        severities = [1, 3, 5]
    else:
        corruptions = [
            'gaussian_noise', 'shot_noise', 'impulse_noise',
            'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
            'snow', 'frost', 'fog', 'brightness',
            'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'
        ]
        severities = [1, 2, 3, 4, 5]

    # Evaluate all methods
    print("\n" + "=" * 80)
    print("Evaluating all methods...")
    print("=" * 80)

    results = []

    for corruption in corruptions:
        for severity in severities:
            print(f"\n{corruption} (severity {severity})")
            print("-" * 80)

            # Source
            result_source = evaluate_source(model, corruption, severity, args.data_root,
                                           args.batch_size, device, args.max_batches)
            results.append(result_source)
            print(f"Source: {result_source['accuracy']:.4f}")

            # TENT
            result_tent = evaluate_tent(model, corruption, severity, args.data_root,
                                       args.batch_size, device, max_batches=args.max_batches)
            results.append(result_tent)
            print(f"TENT: {result_tent['accuracy']:.4f}")

            # FOA
            result_foa = evaluate_foa(foa_adapter, corruption, severity, args.data_root,
                                     args.batch_size, args.max_batches)
            results.append(result_foa)
            print(f"FOA: {result_foa['accuracy']:.4f}")

    # Save results
    df = pd.DataFrame(results)
    os.makedirs(args.output, exist_ok=True)

    csv_path = os.path.join(args.output, 'comprehensive_comparison.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")

    json_path = os.path.join(args.output, 'comprehensive_comparison.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {json_path}")

    # Generate plots
    print("\nGenerating visualizations...")
    plot_comparison(df, args.output)

    # Print summary
    print("\n" + "=" * 80)
    print("Summary Statistics")
    print("=" * 80)

    summary = df.groupby('method')['accuracy'].agg(['mean', 'std'])
    print(summary)

    # Run ablations
    if not args.skip_ablations:
        run_ablation_studies(model, source_stats, args.data_root, args.batch_size, device, args.output)

    print("\n" + "=" * 80)
    print("Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
