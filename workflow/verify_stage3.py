"""
Stage 3 Verification Script

Verifies that all Stage 3 outputs have been generated correctly.
"""

import os
import json
import pandas as pd
import sys

def verify_stage3(results_dir='../results'):
    """Verify Stage 3 completion."""
    print("=" * 80)
    print("STAGE 3 VERIFICATION")
    print("=" * 80)

    required_files = [
        'stage3_comparison.csv',
        'stage3_comparison.json',
        'stage3_component_ablation.csv',
        'stage3_hyperparam_ablation.csv',
        'stage3_summary.json',
        'stage3_method_comparison.png',
        'stage3_method_bar.png',
        'stage3_component_ablation.png',
    ]

    hyperparam_plots = [
        'stage3_ablation_lambda.png',
        'stage3_ablation_num_prompts.png',
        'stage3_ablation_cma_population.png',
        'stage3_ablation_cma_iterations.png',
    ]

    all_passed = True

    print("\nChecking required files:")
    print("-" * 80)

    for file in required_files:
        file_path = os.path.join(results_dir, file)
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"  ✓ {file:45s} ({size:,} bytes)")
        else:
            print(f"  ✗ {file:45s} MISSING")
            all_passed = False

    print("\nChecking hyperparameter ablation plots (optional):")
    print("-" * 80)

    for file in hyperparam_plots:
        file_path = os.path.join(results_dir, file)
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"  ✓ {file:45s} ({size:,} bytes)")
        else:
            print(f"  - {file:45s} (optional)")

    # Verify data integrity
    print("\nVerifying data integrity:")
    print("-" * 80)

    # Check comparison results
    comparison_csv = os.path.join(results_dir, 'stage3_comparison.csv')
    if os.path.exists(comparison_csv):
        try:
            df = pd.read_csv(comparison_csv)
            print(f"  ✓ Comparison CSV: {len(df)} rows, {len(df.columns)} columns")
            print(f"    Methods: {df['method'].unique().tolist()}")
            print(f"    Corruptions: {df['corruption'].nunique()}")
            print(f"    Severities: {df['severity'].unique().tolist()}")
        except Exception as e:
            print(f"  ✗ Error reading comparison CSV: {e}")
            all_passed = False
    else:
        print(f"  ✗ Comparison CSV missing")
        all_passed = False

    # Check component ablation
    component_csv = os.path.join(results_dir, 'stage3_component_ablation.csv')
    if os.path.exists(component_csv):
        try:
            df = pd.read_csv(component_csv)
            print(f"  ✓ Component ablation CSV: {len(df)} rows")
            print(f"    Ablation types: {df['ablation_type'].unique().tolist()}")
        except Exception as e:
            print(f"  ✗ Error reading component ablation CSV: {e}")
            all_passed = False
    else:
        print(f"  ✗ Component ablation CSV missing")
        all_passed = False

    # Check hyperparameter ablation
    hyperparam_csv = os.path.join(results_dir, 'stage3_hyperparam_ablation.csv')
    if os.path.exists(hyperparam_csv):
        try:
            df = pd.read_csv(hyperparam_csv)
            print(f"  ✓ Hyperparameter ablation CSV: {len(df)} rows")
            print(f"    Hyperparameters tested: {df['hyperparameter'].unique().tolist()}")
        except Exception as e:
            print(f"  ✗ Error reading hyperparameter ablation CSV: {e}")
            all_passed = False
    else:
        print(f"  ✗ Hyperparameter ablation CSV missing")
        all_passed = False

    # Check summary
    summary_json = os.path.join(results_dir, 'stage3_summary.json')
    if os.path.exists(summary_json):
        try:
            with open(summary_json, 'r') as f:
                summary = json.load(f)
            print(f"  ✓ Summary JSON loaded successfully")
            print(f"    Summary sections: {list(summary.keys())}")
        except Exception as e:
            print(f"  ✗ Error reading summary JSON: {e}")
            all_passed = False
    else:
        print(f"  ✗ Summary JSON missing")
        all_passed = False

    print("\n" + "=" * 80)
    if all_passed:
        print("✓ STAGE 3 VERIFICATION PASSED")
        print("=" * 80)
        return 0
    else:
        print("✗ STAGE 3 VERIFICATION FAILED")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    results_dir = sys.argv[1] if len(sys.argv) > 1 else '../results'
    exit_code = verify_stage3(results_dir)
    sys.exit(exit_code)
