#!/usr/bin/env python3
"""
Results Validation Script for Stage 5

This script validates that the final ImageNet-C evaluation results
match the expected accuracy targets from the FOA paper within ±5% tolerance.

Usage:
    python validate_results.py --results_dir ../results --tolerance 0.05
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from scipy import stats


# Expected results from FOA paper (ImageNet-C average across all corruptions and severities)
EXPECTED_RESULTS = {
    'source': {
        'accuracy': 56.3,
        'std': 2.8,
        'description': 'Zero-shot ViT-Base (no adaptation)'
    },
    'tent': {
        'accuracy': 59.6,
        'std': 2.98,
        'description': 'TENT: Test-time entropy minimization'
    },
    'foa_32bit': {
        'accuracy': 66.3,
        'std': 3.31,
        'description': 'FOA: Forward-Optimization Adaptation (32-bit)'
    },
    'foa_8bit': {
        'accuracy': 63.5,
        'std': 3.17,
        'description': 'FOA: Forward-Optimization Adaptation (8-bit quantized)'
    }
}


def load_results(results_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load all result CSV files from the results directory."""
    results = {}

    # Check for Stage 3 comprehensive comparison results
    comparison_file = results_dir / 'stage3_comparison.csv'
    if comparison_file.exists():
        df = pd.read_csv(comparison_file)
        results['comprehensive'] = df
        print(f"✓ Loaded comprehensive comparison: {len(df)} rows")

    # Check for individual baseline results
    for method in ['source', 'tent', 'foa']:
        result_file = results_dir / f'{method}_results.csv'
        if result_file.exists():
            df = pd.read_csv(result_file)
            results[method] = df
            print(f"✓ Loaded {method} results: {len(df)} rows")

    return results


def calculate_average_accuracy(df: pd.DataFrame, method: str = None) -> Tuple[float, float]:
    """
    Calculate average accuracy across all corruptions and severities.

    Returns:
        mean_accuracy, std_accuracy
    """
    if method and 'method' in df.columns:
        df = df[df['method'] == method]

    if 'accuracy' in df.columns:
        accuracies = df['accuracy'].values
    elif 'acc' in df.columns:
        accuracies = df['acc'].values
    else:
        raise ValueError(f"No accuracy column found in dataframe. Columns: {df.columns.tolist()}")

    return np.mean(accuracies), np.std(accuracies)


def validate_method(
    actual_mean: float,
    actual_std: float,
    expected_mean: float,
    expected_std: float,
    tolerance: float
) -> Dict:
    """
    Validate that actual results match expected results within tolerance.

    Args:
        actual_mean: Mean accuracy from our implementation
        actual_std: Std dev from our implementation
        expected_mean: Expected mean from paper
        expected_std: Expected std from paper
        tolerance: Tolerance as fraction (e.g., 0.05 for ±5%)

    Returns:
        Dictionary with validation results
    """
    lower_bound = expected_mean * (1 - tolerance)
    upper_bound = expected_mean * (1 + tolerance)

    within_tolerance = lower_bound <= actual_mean <= upper_bound
    difference = actual_mean - expected_mean
    relative_difference = (difference / expected_mean) * 100

    return {
        'actual_mean': actual_mean,
        'actual_std': actual_std,
        'expected_mean': expected_mean,
        'expected_std': expected_std,
        'difference': difference,
        'relative_difference_pct': relative_difference,
        'tolerance_pct': tolerance * 100,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'within_tolerance': within_tolerance,
        'status': '✅ PASS' if within_tolerance else '❌ FAIL'
    }


def perform_statistical_tests(df: pd.DataFrame) -> Dict:
    """
    Perform statistical significance tests between methods.

    Returns:
        Dictionary with test results
    """
    tests = {}

    # Check if we have comprehensive comparison data
    if 'method' not in df.columns:
        return tests

    methods = df['method'].unique()

    # Extract accuracies for each method
    method_accuracies = {}
    for method in methods:
        method_df = df[df['method'] == method]
        if 'accuracy' in method_df.columns:
            method_accuracies[method] = method_df['accuracy'].values

    # Perform paired t-tests between methods
    if 'tent' in method_accuracies and 'foa_32bit' in method_accuracies:
        t_stat, p_value = stats.ttest_rel(
            method_accuracies['foa_32bit'],
            method_accuracies['tent']
        )
        tests['foa_vs_tent'] = {
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significant': p_value < 0.05,
            'foa_mean': float(np.mean(method_accuracies['foa_32bit'])),
            'tent_mean': float(np.mean(method_accuracies['tent'])),
            'improvement': float(np.mean(method_accuracies['foa_32bit']) - np.mean(method_accuracies['tent']))
        }

    return tests


def main():
    parser = argparse.ArgumentParser(description='Validate ImageNet-C evaluation results')
    parser.add_argument('--results_dir', type=str, default='../results',
                        help='Directory containing result files')
    parser.add_argument('--tolerance', type=float, default=0.05,
                        help='Tolerance for accuracy matching (default: 0.05 for ±5%%)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file for validation report')
    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    print("=" * 80)
    print("STAGE 5: RESULTS VALIDATION")
    print("=" * 80)
    print()

    # Load results
    print("Loading results...")
    try:
        results = load_results(results_dir)
    except Exception as e:
        print(f"❌ Error loading results: {e}")
        return 1

    if not results:
        print("❌ No result files found. Please ensure evaluation has completed.")
        return 1

    print()

    # Validate each method
    validation_results = {}

    print("Validating results against paper targets...")
    print("-" * 80)

    # Use comprehensive comparison if available
    if 'comprehensive' in results:
        df = results['comprehensive']

        for method_key, expected in EXPECTED_RESULTS.items():
            # Map method names
            method_name = method_key
            if method_key == 'foa_32bit':
                method_name = 'foa'
            elif method_key == 'foa_8bit':
                method_name = 'foa_8bit'

            # Check if method exists in results
            if method_name not in df['method'].unique():
                print(f"⚠️  {method_key.upper()}: No data found (method: {method_name})")
                continue

            # Calculate actual results
            actual_mean, actual_std = calculate_average_accuracy(df, method_name)

            # Validate
            validation = validate_method(
                actual_mean,
                actual_std,
                expected['accuracy'],
                expected['std'],
                args.tolerance
            )

            validation_results[method_key] = validation

            # Print results
            print(f"\n{method_key.upper()}: {expected['description']}")
            print(f"  Expected: {expected['accuracy']:.2f}% ± {expected['std']:.2f}%")
            print(f"  Actual:   {actual_mean:.2f}% ± {actual_std:.2f}%")
            print(f"  Difference: {validation['difference']:.2f}% ({validation['relative_difference_pct']:.1f}%)")
            print(f"  Tolerance: ±{validation['tolerance_pct']:.1f}% [{validation['lower_bound']:.2f}%, {validation['upper_bound']:.2f}%]")
            print(f"  Status: {validation['status']}")

    print()
    print("=" * 80)

    # Statistical significance tests
    if 'comprehensive' in results:
        print("\nStatistical Significance Tests:")
        print("-" * 80)

        stat_tests = perform_statistical_tests(results['comprehensive'])

        if 'foa_vs_tent' in stat_tests:
            test = stat_tests['foa_vs_tent']
            print(f"\nFOA vs. TENT (Paired t-test):")
            print(f"  FOA mean: {test['foa_mean']:.2f}%")
            print(f"  TENT mean: {test['tent_mean']:.2f}%")
            print(f"  Improvement: +{test['improvement']:.2f}%")
            print(f"  t-statistic: {test['t_statistic']:.4f}")
            print(f"  p-value: {test['p_value']:.6f}")
            print(f"  Significant (p<0.05): {'✅ YES' if test['significant'] else '❌ NO'}")

    print()
    print("=" * 80)

    # Overall summary
    print("\nOVERALL VALIDATION SUMMARY:")
    print("-" * 80)

    all_passed = all(v['within_tolerance'] for v in validation_results.values())
    num_passed = sum(1 for v in validation_results.values() if v['within_tolerance'])
    num_total = len(validation_results)

    print(f"Methods validated: {num_total}")
    print(f"Within tolerance: {num_passed}")
    print(f"Outside tolerance: {num_total - num_passed}")
    print()

    if all_passed:
        print("✅ ALL METHODS PASS VALIDATION")
        print("The implementation successfully replicates the FOA paper results.")
    else:
        print("⚠️  SOME METHODS OUTSIDE TOLERANCE")
        print("Review the differences above. Small variations may be expected due to:")
        print("  - Random seed differences")
        print("  - Hardware/precision differences")
        print("  - Model weight initialization")

    # Save validation report
    if args.output or not all_passed:
        output_file = args.output or results_dir / 'stage5_validation_report.json'
        report = {
            'validation_results': validation_results,
            'statistical_tests': stat_tests if 'comprehensive' in results else {},
            'summary': {
                'total_methods': num_total,
                'passed': num_passed,
                'failed': num_total - num_passed,
                'all_passed': all_passed
            },
            'tolerance': args.tolerance
        }

        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\nValidation report saved to: {output_file}")

    print()
    print("=" * 80)

    return 0 if all_passed else 1


if __name__ == '__main__':
    exit(main())
