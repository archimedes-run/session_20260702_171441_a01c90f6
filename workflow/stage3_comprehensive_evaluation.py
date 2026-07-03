"""
Stage 3: Comprehensive Evaluation & Ablation Studies

This script performs:
1. Full comparison of Source, TENT, and FOA (32-bit and 8-bit) across all corruptions
2. Ablation studies to isolate FOA component contributions:
   - Prompt-only (lambda=0, entropy term only)
   - Shifting-only (no prompts, only activation shifting)
   - Full FOA (prompts + shifting)
3. Hyperparameter sensitivity analysis (lambda, prompt length, CMA-ES params)
4. Quantized model evaluation (8-bit)
5. Comprehensive visualization and reporting
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
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from foa_method import FOAAdapter, load_source_statistics
from data_loader import create_imagenet_c_loader
from tent_baseline import TENT
from quantized_model import create_quantized_vit


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

    try:
        loader = create_imagenet_c_loader(data_root, corruption, severity, batch_size, shuffle=False)
    except Exception as e:
        print(f"Warning: Could not load ImageNet-C data: {e}")
        return None

    total_correct = 0
    total_samples = 0
    total_entropy = 0.0

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(loader, desc=f"Source: {corruption}-{severity}", leave=False)):
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

    accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    avg_entropy = total_entropy / total_samples if total_samples > 0 else 0.0

    return {
        'method': 'Source',
        'corruption': corruption,
        'severity': severity,
        'accuracy': accuracy,
        'error_rate': 1.0 - accuracy,
        'entropy': avg_entropy
    }


def evaluate_tent(tent_model, corruption, severity, data_root, batch_size, max_batches=None):
    """Evaluate TENT (entropy minimization) using TENT class."""
    # Reset TENT model
    tent_model.reset()

    try:
        loader = create_imagenet_c_loader(data_root, corruption, severity, batch_size, shuffle=False)
    except Exception as e:
        print(f"Warning: Could not load ImageNet-C data: {e}")
        return None

    total_correct = 0
    total_samples = 0
    total_entropy = 0.0
    batch_count = 0

    for batch_idx, (images, labels) in enumerate(tqdm(loader, desc=f"TENT: {corruption}-{severity}", leave=False)):
        if max_batches and batch_idx >= max_batches:
            break

        images = images.to(tent_model.device)
        labels = labels.to(tent_model.device)

        # Forward pass
        logits = tent_model.model(images)

        # TENT: minimize entropy
        probs = torch.softmax(logits, dim=1)
        log_probs = torch.log(probs + 1e-10)
        entropy_loss = -(probs * log_probs).sum(dim=1).mean()

        # Backward and update
        tent_model.optimizer.zero_grad()
        entropy_loss.backward()
        tent_model.optimizer.step()

        # Evaluation
        with torch.no_grad():
            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += images.size(0)
            total_entropy += entropy_loss.item() * images.size(0)

        batch_count += 1

    accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    avg_entropy = total_entropy / total_samples if total_samples > 0 else 0.0

    return {
        'method': 'TENT',
        'corruption': corruption,
        'severity': severity,
        'accuracy': accuracy,
        'error_rate': 1.0 - accuracy,
        'entropy': avg_entropy
    }


def evaluate_foa(foa_adapter, corruption, severity, data_root, batch_size, max_batches=None, variant_name='FOA'):
    """Evaluate FOA or its variants."""
    foa_adapter.reset()

    try:
        loader = create_imagenet_c_loader(data_root, corruption, severity, batch_size, shuffle=False)
    except Exception as e:
        print(f"Warning: Could not load ImageNet-C data: {e}")
        return None

    total_correct = 0
    total_samples = 0
    total_entropy = 0.0
    total_activation_disc = 0.0

    for batch_idx, (images, labels) in enumerate(tqdm(loader, desc=f"{variant_name}: {corruption}-{severity}", leave=False)):
        if max_batches and batch_idx >= max_batches:
            break

        logits, info = foa_adapter.adapt_batch(images, labels)

        preds = logits.argmax(dim=1)
        labels = labels.to(logits.device)

        total_correct += (preds == labels).sum().item()
        total_samples += images.size(0)
        total_entropy += info['entropy'] * images.size(0)
        total_activation_disc += info.get('activation_discrepancy', 0.0) * images.size(0)

    accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    avg_entropy = total_entropy / total_samples if total_samples > 0 else 0.0
    avg_activation_disc = total_activation_disc / total_samples if total_samples > 0 else 0.0

    return {
        'method': variant_name,
        'corruption': corruption,
        'severity': severity,
        'accuracy': accuracy,
        'error_rate': 1.0 - accuracy,
        'entropy': avg_entropy,
        'activation_discrepancy': avg_activation_disc
    }


def evaluate_quantized_foa(model, source_stats, corruption, severity, data_root, batch_size, device, max_batches=None):
    """Evaluate 8-bit quantized FOA."""
    # Quantize model (CPU only for now)
    if device != 'cpu':
        print(f"Warning: Quantization only works on CPU. Switching device from {device} to cpu")
        device = 'cpu'

    quantized_model = create_quantized_vit(device=device)
    # Note: create_quantized_vit already returns model on the specified device

    # Create FOA adapter with quantized model
    foa_adapter = FOAAdapter(
        model=quantized_model,
        source_stats=source_stats,
        num_prompts=10,
        lambda_activation=0.1,
        cma_population_size=10,
        cma_max_iterations=20,
        device=device
    )

    result = evaluate_foa(foa_adapter, corruption, severity, data_root, batch_size, max_batches, variant_name='FOA-8bit')
    return result


def run_component_ablations(model, source_stats, corruptions, severities, data_root, batch_size, device, max_batches=None):
    """
    Run ablation studies to isolate FOA component contributions.

    Ablations:
    1. Source (baseline - no adaptation)
    2. Prompt-only (lambda=0, entropy term only, no activation shifting)
    3. Full FOA (lambda>0, prompts + activation shifting)

    Note: Shifting-only (no prompts) is not practical since CMA-ES needs something to optimize.
    Instead, we test with minimal prompts (num_prompts=1) and high lambda.
    """
    print("\n" + "=" * 80)
    print("COMPONENT ABLATION STUDIES")
    print("=" * 80)

    results = []

    # Test subset
    test_corruptions = corruptions[:3] if len(corruptions) > 3 else corruptions
    test_severities = [severities[0], severities[len(severities)//2], severities[-1]]

    for corruption in test_corruptions:
        for severity in test_severities:
            print(f"\n{corruption} (severity {severity})")
            print("-" * 80)

            # 1. Source baseline
            result = evaluate_source(model, corruption, severity, data_root, batch_size, device, max_batches)
            if result:
                result['ablation_type'] = 'Source (no adaptation)'
                results.append(result)
                print(f"  Source: {result['accuracy']:.4f}")

            # 2. Prompt-only (lambda=0, entropy term only)
            foa_prompt_only = FOAAdapter(
                model=model,
                source_stats=source_stats,
                num_prompts=10,
                lambda_activation=0.0,  # No activation shifting
                cma_population_size=10,
                cma_max_iterations=20,
                device=device
            )
            result = evaluate_foa(foa_prompt_only, corruption, severity, data_root, batch_size, max_batches, variant_name='Prompt-only')
            if result:
                result['ablation_type'] = 'Prompt-only (lambda=0)'
                results.append(result)
                print(f"  Prompt-only: {result['accuracy']:.4f}")

            # 3. Full FOA (lambda=0.1, both components)
            foa_full = FOAAdapter(
                model=model,
                source_stats=source_stats,
                num_prompts=10,
                lambda_activation=0.1,
                cma_population_size=10,
                cma_max_iterations=20,
                device=device
            )
            result = evaluate_foa(foa_full, corruption, severity, data_root, batch_size, max_batches, variant_name='FOA-Full')
            if result:
                result['ablation_type'] = 'Full FOA (lambda=0.1)'
                results.append(result)
                print(f"  Full FOA: {result['accuracy']:.4f}")

            # 4. Shifting-heavy (minimal prompts, high lambda)
            foa_shifting_heavy = FOAAdapter(
                model=model,
                source_stats=source_stats,
                num_prompts=1,  # Minimal prompts
                lambda_activation=1.0,  # High activation weight
                cma_population_size=10,
                cma_max_iterations=20,
                device=device
            )
            result = evaluate_foa(foa_shifting_heavy, corruption, severity, data_root, batch_size, max_batches, variant_name='Shifting-heavy')
            if result:
                result['ablation_type'] = 'Shifting-heavy (prompts=1, lambda=1.0)'
                results.append(result)
                print(f"  Shifting-heavy: {result['accuracy']:.4f}")

    return results


def run_hyperparameter_ablations(model, source_stats, corruptions, severities, data_root, batch_size, device, max_batches=None):
    """
    Run hyperparameter sensitivity analysis.

    Ablations:
    1. Lambda values: [0.0, 0.01, 0.05, 0.1, 0.5, 1.0]
    2. Prompt lengths: [1, 5, 10, 20, 50]
    3. CMA-ES population: [5, 10, 20, 50]
    4. CMA-ES iterations: [5, 10, 20, 50]
    """
    print("\n" + "=" * 80)
    print("HYPERPARAMETER SENSITIVITY ANALYSIS")
    print("=" * 80)

    results = []

    # Test on single corruption/severity for speed
    test_corruption = corruptions[0]
    test_severity = severities[len(severities)//2]  # Middle severity

    # 1. Lambda ablation
    print("\n1. Lambda (activation weight) sensitivity")
    lambda_values = [0.0, 0.01, 0.05, 0.1, 0.5, 1.0]
    for lambda_val in lambda_values:
        print(f"  Testing lambda={lambda_val}")
        foa = FOAAdapter(
            model=model,
            source_stats=source_stats,
            num_prompts=10,
            lambda_activation=lambda_val,
            cma_population_size=10,
            cma_max_iterations=20,
            device=device
        )
        result = evaluate_foa(foa, test_corruption, test_severity, data_root, batch_size, max_batches, variant_name=f'FOA-lambda={lambda_val}')
        if result:
            result['hyperparameter'] = 'lambda'
            result['value'] = lambda_val
            results.append(result)

    # 2. Prompt length ablation
    print("\n2. Prompt length sensitivity")
    prompt_lengths = [1, 5, 10, 20, 50]
    for num_prompts in prompt_lengths:
        print(f"  Testing num_prompts={num_prompts}")
        foa = FOAAdapter(
            model=model,
            source_stats=source_stats,
            num_prompts=num_prompts,
            lambda_activation=0.1,
            cma_population_size=10,
            cma_max_iterations=20,
            device=device
        )
        result = evaluate_foa(foa, test_corruption, test_severity, data_root, batch_size, max_batches, variant_name=f'FOA-prompts={num_prompts}')
        if result:
            result['hyperparameter'] = 'num_prompts'
            result['value'] = num_prompts
            results.append(result)

    # 3. CMA-ES population ablation
    print("\n3. CMA-ES population size sensitivity")
    population_sizes = [5, 10, 20]
    for pop_size in population_sizes:
        print(f"  Testing population={pop_size}")
        foa = FOAAdapter(
            model=model,
            source_stats=source_stats,
            num_prompts=10,
            lambda_activation=0.1,
            cma_population_size=pop_size,
            cma_max_iterations=20,
            device=device
        )
        result = evaluate_foa(foa, test_corruption, test_severity, data_root, batch_size, max_batches, variant_name=f'FOA-pop={pop_size}')
        if result:
            result['hyperparameter'] = 'cma_population'
            result['value'] = pop_size
            results.append(result)

    # 4. CMA-ES iterations ablation
    print("\n4. CMA-ES iterations sensitivity")
    iteration_counts = [5, 10, 20]
    for iters in iteration_counts:
        print(f"  Testing iterations={iters}")
        foa = FOAAdapter(
            model=model,
            source_stats=source_stats,
            num_prompts=10,
            lambda_activation=0.1,
            cma_population_size=10,
            cma_max_iterations=iters,
            device=device
        )
        result = evaluate_foa(foa, test_corruption, test_severity, data_root, batch_size, max_batches, variant_name=f'FOA-iters={iters}')
        if result:
            result['hyperparameter'] = 'cma_iterations'
            result['value'] = iters
            results.append(result)

    return results


def generate_comprehensive_visualizations(comparison_df, component_df, hyperparam_df, output_dir):
    """Generate all visualizations for Stage 3."""
    os.makedirs(output_dir, exist_ok=True)

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 300

    print("\nGenerating visualizations...")

    # 1. Method comparison across all corruptions/severities
    if not comparison_df.empty:
        print("  1. Method comparison plots...")

        # Accuracy vs Severity
        fig, ax = plt.subplots(figsize=(10, 6))
        for method in comparison_df['method'].unique():
            method_data = comparison_df[comparison_df['method'] == method]
            avg_by_severity = method_data.groupby('severity')['accuracy'].mean()
            ax.plot(avg_by_severity.index, avg_by_severity.values, marker='o', label=method, linewidth=2, markersize=8)

        ax.set_xlabel('Severity', fontsize=12)
        ax.set_ylabel('Average Accuracy', fontsize=12)
        ax.set_title('Method Comparison: Accuracy vs Severity', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'stage3_method_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # Bar chart comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        avg_by_method = comparison_df.groupby('method')['accuracy'].agg(['mean', 'std'])
        x = np.arange(len(avg_by_method))
        ax.bar(x, avg_by_method['mean'], yerr=avg_by_method['std'], capsize=5, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(avg_by_method.index, fontsize=10)
        ax.set_ylabel('Average Accuracy', fontsize=12)
        ax.set_title('Overall Method Comparison', fontsize=14, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3)

        # Add value labels
        for i, (mean, std) in enumerate(zip(avg_by_method['mean'], avg_by_method['std'])):
            ax.text(i, mean, f'{mean:.4f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'stage3_method_bar.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # 2. Component ablation plots
    if not component_df.empty:
        print("  2. Component ablation plots...")

        fig, ax = plt.subplots(figsize=(12, 6))
        component_avg = component_df.groupby('ablation_type')['accuracy'].agg(['mean', 'std'])
        x = np.arange(len(component_avg))
        ax.bar(x, component_avg['mean'], yerr=component_avg['std'], capsize=5, alpha=0.7,
               color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        ax.set_xticks(x)
        ax.set_xticklabels(component_avg.index, fontsize=9, rotation=15, ha='right')
        ax.set_ylabel('Average Accuracy', fontsize=12)
        ax.set_title('Component Ablation: Isolating FOA Components', fontsize=14, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3)

        # Add value labels
        for i, (mean, std) in enumerate(zip(component_avg['mean'], component_avg['std'])):
            ax.text(i, mean, f'{mean:.4f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'stage3_component_ablation.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # 3. Hyperparameter sensitivity plots
    if not hyperparam_df.empty:
        print("  3. Hyperparameter sensitivity plots...")

        hyperparams = hyperparam_df['hyperparameter'].unique()

        for hyperparam in hyperparams:
            data = hyperparam_df[hyperparam_df['hyperparameter'] == hyperparam]

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(data['value'], data['accuracy'], marker='o', linewidth=2, markersize=8)
            ax.set_xlabel(f'{hyperparam.replace("_", " ").title()}', fontsize=12)
            ax.set_ylabel('Accuracy', fontsize=12)
            ax.set_title(f'Hyperparameter Sensitivity: {hyperparam.replace("_", " ").title()}',
                        fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'stage3_ablation_{hyperparam}.png'), dpi=300, bbox_inches='tight')
            plt.close()

    print(f"\n✓ All visualizations saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Stage 3: Comprehensive Evaluation & Ablation Studies')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Path to ImageNet-C dataset (or test data)')
    parser.add_argument('--source_stats', type=str, default='../results/source_statistics.pth',
                        help='Path to source statistics')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (auto/cuda/mps/cpu)')
    parser.add_argument('--output', type=str, default='../results',
                        help='Output directory')
    parser.add_argument('--max_batches', type=int, default=None,
                        help='Max batches per corruption (for testing)')
    parser.add_argument('--test_mode', action='store_true',
                        help='Run on subset of corruptions (quick test)')
    parser.add_argument('--skip_quantized', action='store_true',
                        help='Skip quantized model evaluation')
    args = parser.parse_args()

    set_random_seeds(42)

    # Device detection
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    else:
        device = args.device

    print("=" * 80)
    print("STAGE 3: COMPREHENSIVE EVALUATION & ABLATION STUDIES")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Data root: {args.data_root}")
    print(f"Test mode: {args.test_mode}")
    print()

    # Check data availability
    if not os.path.exists(args.data_root):
        print(f"\n{'='*80}")
        print("WARNING: ImageNet-C dataset not found!")
        print(f"{'='*80}")
        print(f"Expected path: {args.data_root}")
        print("\nProceeding with test data (if available) for verification purposes only.")
        print("Results will NOT be scientifically valid without full ImageNet-C dataset.")
        print(f"{'='*80}\n")

    # Load model
    print("Loading model...")
    model = timm.create_model('vit_base_patch16_224', pretrained=True)
    model = model.to(device)
    model.eval()
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"✓ Model loaded: ViT-Base ({num_params:.2f}M parameters)")

    # Load source statistics
    print("Loading source statistics...")
    if not os.path.exists(args.source_stats):
        print(f"ERROR: Source statistics not found at {args.source_stats}")
        print("Please run Stage 2 first (compute_source_stats.py)")
        sys.exit(1)

    source_stats = load_source_statistics(args.source_stats)
    print(f"✓ Loaded statistics for {len(source_stats)} layers")
    print()

    # Define corruptions and severities
    if args.test_mode:
        corruptions = ['gaussian_noise', 'defocus_blur', 'brightness']
        severities = [1, 3, 5]
        print("Running in TEST MODE: 3 corruptions × 3 severities = 9 conditions")
    else:
        corruptions = [
            'gaussian_noise', 'shot_noise', 'impulse_noise',
            'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
            'snow', 'frost', 'fog', 'brightness',
            'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'
        ]
        severities = [1, 2, 3, 4, 5]
        print("Running FULL EVALUATION: 15 corruptions × 5 severities = 75 conditions")
    print()

    # ========================================================================
    # PART 1: FULL METHOD COMPARISON
    # ========================================================================
    print("\n" + "=" * 80)
    print("PART 1: FULL METHOD COMPARISON (Source, TENT, FOA, FOA-8bit)")
    print("=" * 80)

    comparison_results = []

    # Create FOA adapter
    foa_adapter = FOAAdapter(
        model=model,
        source_stats=source_stats,
        num_prompts=10,
        lambda_activation=0.1,
        cma_population_size=10,
        cma_max_iterations=20,
        device=device
    )

    # Create TENT model
    print("Initializing TENT model...")
    tent_model = TENT(device=device, learning_rate=1e-3)
    print(f"✓ TENT model initialized")
    print()

    for corruption in corruptions:
        for severity in severities:
            print(f"\n{corruption} (severity {severity})")
            print("-" * 80)

            # Source
            result = evaluate_source(model, corruption, severity, args.data_root, args.batch_size, device, args.max_batches)
            if result:
                comparison_results.append(result)
                print(f"  Source: {result['accuracy']:.4f}")

            # TENT
            result = evaluate_tent(tent_model, corruption, severity, args.data_root, args.batch_size, max_batches=args.max_batches)
            if result:
                comparison_results.append(result)
                print(f"  TENT: {result['accuracy']:.4f}")

            # FOA
            result = evaluate_foa(foa_adapter, corruption, severity, args.data_root, args.batch_size, args.max_batches)
            if result:
                comparison_results.append(result)
                print(f"  FOA: {result['accuracy']:.4f}")

            # FOA-8bit (quantized)
            if not args.skip_quantized:
                try:
                    result = evaluate_quantized_foa(model, source_stats, corruption, severity,
                                                   args.data_root, args.batch_size, device, args.max_batches)
                    if result:
                        comparison_results.append(result)
                        print(f"  FOA-8bit: {result['accuracy']:.4f}")
                except Exception as e:
                    print(f"  FOA-8bit: SKIPPED (quantization error: {e})")

    # ========================================================================
    # PART 2: COMPONENT ABLATIONS
    # ========================================================================
    component_results = run_component_ablations(
        model, source_stats, corruptions, severities,
        args.data_root, args.batch_size, device, args.max_batches
    )

    # ========================================================================
    # PART 3: HYPERPARAMETER SENSITIVITY
    # ========================================================================
    hyperparam_results = run_hyperparameter_ablations(
        model, source_stats, corruptions, severities,
        args.data_root, args.batch_size, device, args.max_batches
    )

    # ========================================================================
    # SAVE ALL RESULTS
    # ========================================================================
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)

    os.makedirs(args.output, exist_ok=True)

    # Save comparison results
    if comparison_results:
        df_comparison = pd.DataFrame(comparison_results)
        csv_path = os.path.join(args.output, 'stage3_comparison.csv')
        df_comparison.to_csv(csv_path, index=False)
        print(f"✓ Comparison results: {csv_path}")

        json_path = os.path.join(args.output, 'stage3_comparison.json')
        with open(json_path, 'w') as f:
            json.dump(comparison_results, f, indent=2)
        print(f"✓ Comparison results: {json_path}")
    else:
        df_comparison = pd.DataFrame()

    # Save component ablation results
    if component_results:
        df_component = pd.DataFrame(component_results)
        csv_path = os.path.join(args.output, 'stage3_component_ablation.csv')
        df_component.to_csv(csv_path, index=False)
        print(f"✓ Component ablation: {csv_path}")
    else:
        df_component = pd.DataFrame()

    # Save hyperparameter ablation results
    if hyperparam_results:
        df_hyperparam = pd.DataFrame(hyperparam_results)
        csv_path = os.path.join(args.output, 'stage3_hyperparam_ablation.csv')
        df_hyperparam.to_csv(csv_path, index=False)
        print(f"✓ Hyperparameter ablation: {csv_path}")
    else:
        df_hyperparam = pd.DataFrame()

    # Generate summary statistics
    summary = {}
    if not df_comparison.empty:
        summary['method_comparison'] = df_comparison.groupby('method')['accuracy'].agg(['mean', 'std', 'min', 'max']).to_dict()
    if not df_component.empty:
        summary['component_ablation'] = df_component.groupby('ablation_type')['accuracy'].agg(['mean', 'std']).to_dict()
    if not df_hyperparam.empty:
        summary['hyperparameter_sensitivity'] = {}
        for hp in df_hyperparam['hyperparameter'].unique():
            hp_data = df_hyperparam[df_hyperparam['hyperparameter'] == hp]
            summary['hyperparameter_sensitivity'][hp] = hp_data[['value', 'accuracy']].to_dict('records')

    summary_path = os.path.join(args.output, 'stage3_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Summary statistics: {summary_path}")

    # ========================================================================
    # GENERATE VISUALIZATIONS
    # ========================================================================
    generate_comprehensive_visualizations(df_comparison, df_component, df_hyperparam, args.output)

    # ========================================================================
    # PRINT FINAL SUMMARY
    # ========================================================================
    print("\n" + "=" * 80)
    print("STAGE 3 COMPLETE - SUMMARY")
    print("=" * 80)

    if not df_comparison.empty:
        print("\nMethod Comparison (Average Accuracy):")
        print("-" * 80)
        method_summary = df_comparison.groupby('method')['accuracy'].agg(['mean', 'std'])
        for method, row in method_summary.iterrows():
            print(f"  {method:15s}: {row['mean']:.4f} ± {row['std']:.4f}")

    if not df_component.empty:
        print("\nComponent Ablation (Average Accuracy):")
        print("-" * 80)
        component_summary = df_component.groupby('ablation_type')['accuracy'].agg(['mean', 'std'])
        for ablation, row in component_summary.iterrows():
            print(f"  {ablation:35s}: {row['mean']:.4f} ± {row['std']:.4f}")

    print("\n" + "=" * 80)
    print("All results saved to:", args.output)
    print("=" * 80)


if __name__ == "__main__":
    main()
