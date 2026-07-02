"""
Test script to verify the baseline implementations work correctly.

This script creates a small synthetic test dataset to verify that:
1. Models load correctly
2. Data loading pipeline works
3. Source baseline runs
4. TENT baseline runs
5. All components integrate properly

Note: This is NOT for scientific results, only for code verification.
"""
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from PIL import Image
import json

from source_baseline import SourceModel
from tent_baseline import TENT


def create_synthetic_test_data(output_dir: str, num_samples: int = 100):
    """
    Create a small synthetic test dataset for code verification.

    Args:
        output_dir: Directory to save synthetic data
        num_samples: Number of samples to generate
    """
    output_path = Path(output_dir)

    # Create directory structure: corruption/severity/class/images
    corruption = 'test_corruption'
    severity = 1
    num_classes = 10

    print(f"Creating synthetic test data at: {output_path}")

    for class_idx in range(num_classes):
        class_dir = output_path / corruption / str(severity) / str(class_idx)
        class_dir.mkdir(parents=True, exist_ok=True)

        # Generate random images for this class
        samples_per_class = num_samples // num_classes

        for i in range(samples_per_class):
            # Create random RGB image (224x224)
            img_array = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)

            # Save image
            img_path = class_dir / f'image_{i:04d}.JPEG'
            img.save(img_path)

    print(f"Created {num_samples} synthetic images across {num_classes} classes")
    return str(output_path)


def test_model_loading(device: str = 'cpu'):
    """Test that models load correctly."""
    print("\n" + "="*80)
    print("TEST 1: Model Loading")
    print("="*80)

    try:
        # Test Source model
        print("\nLoading Source model...")
        source_model = SourceModel(device=device)
        print("✓ Source model loaded successfully")

        # Test TENT model
        print("\nLoading TENT model...")
        tent_model = TENT(device=device, learning_rate=1e-3)
        print("✓ TENT model loaded successfully")

        # Test forward pass
        print("\nTesting forward pass...")
        dummy_input = torch.randn(2, 3, 224, 224).to(device)

        with torch.no_grad():
            source_output = source_model.predict(dummy_input)
            print(f"✓ Source output shape: {source_output.shape}")

        tent_output = tent_model.adapt_batch(dummy_input)
        print(f"✓ TENT output shape: {tent_output.shape}")

        print("\n✓ TEST 1 PASSED: Models load and run correctly")
        return True

    except Exception as e:
        print(f"\n✗ TEST 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_loading(data_root: str):
    """Test data loading pipeline."""
    print("\n" + "="*80)
    print("TEST 2: Data Loading")
    print("="*80)

    try:
        from data_loader import create_imagenet_c_loader

        print(f"\nCreating data loader for: {data_root}")
        loader = create_imagenet_c_loader(
            data_root,
            corruption='test_corruption',
            severity=1,
            batch_size=8,
            num_workers=0,
            shuffle=False
        )

        print(f"✓ Data loader created")
        print(f"  Dataset size: {len(loader.dataset)}")
        print(f"  Number of batches: {len(loader)}")

        # Test loading one batch
        images, labels = next(iter(loader))
        print(f"✓ Batch loaded successfully")
        print(f"  Images shape: {images.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Image range: [{images.min():.3f}, {images.max():.3f}]")
        print(f"  Unique labels: {labels.unique().tolist()}")

        print("\n✓ TEST 2 PASSED: Data loading works correctly")
        return True

    except Exception as e:
        print(f"\n✗ TEST 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_source_baseline(data_root: str, device: str = 'cpu'):
    """Test Source baseline evaluation."""
    print("\n" + "="*80)
    print("TEST 3: Source Baseline Evaluation")
    print("="*80)

    try:
        from data_loader import create_imagenet_c_loader

        # Create data loader
        loader = create_imagenet_c_loader(
            data_root,
            corruption='test_corruption',
            severity=1,
            batch_size=8,
            num_workers=0,
            shuffle=False
        )

        # Create Source model
        print("\nInitializing Source model...")
        model = SourceModel(device=device)

        # Evaluate
        print("\nRunning evaluation...")
        results = model.evaluate(loader)

        print("\n✓ Evaluation complete:")
        print(f"  Accuracy: {results['accuracy']:.2f}%")
        print(f"  Error Rate: {results['error_rate']:.2f}%")
        print(f"  Avg Entropy: {results['avg_entropy']:.4f}")
        print(f"  Avg Confidence: {results['avg_confidence']:.4f}")
        print(f"  Total Samples: {results['total_samples']}")

        print("\n✓ TEST 3 PASSED: Source baseline works correctly")
        return True, results

    except Exception as e:
        print(f"\n✗ TEST 3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_tent_baseline(data_root: str, device: str = 'cpu'):
    """Test TENT baseline evaluation."""
    print("\n" + "="*80)
    print("TEST 4: TENT Baseline Evaluation")
    print("="*80)

    try:
        from data_loader import create_imagenet_c_loader

        # Create data loader
        loader = create_imagenet_c_loader(
            data_root,
            corruption='test_corruption',
            severity=1,
            batch_size=8,
            num_workers=0,
            shuffle=False
        )

        # Create TENT model
        print("\nInitializing TENT model...")
        model = TENT(device=device, learning_rate=1e-3)

        # Evaluate
        print("\nRunning evaluation...")
        results = model.evaluate(loader)

        print("\n✓ Evaluation complete:")
        print(f"  Accuracy: {results['accuracy']:.2f}%")
        print(f"  Error Rate: {results['error_rate']:.2f}%")
        print(f"  Avg Entropy: {results['avg_entropy']:.4f}")
        print(f"  Avg Confidence: {results['avg_confidence']:.4f}")
        print(f"  Avg Adaptation Loss: {results['avg_adaptation_loss']:.4f}")
        print(f"  Total Samples: {results['total_samples']}")

        print("\n✓ TEST 4 PASSED: TENT baseline works correctly")
        return True, results

    except Exception as e:
        print(f"\n✗ TEST 4 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def run_all_tests():
    """Run all verification tests."""
    print("\n" + "="*80)
    print("RUNNING IMPLEMENTATION VERIFICATION TESTS")
    print("="*80)

    # Auto-detect device
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    print(f"\nUsing device: {device}")

    # Create synthetic test data
    test_data_dir = './test_data'
    data_root = create_synthetic_test_data(test_data_dir, num_samples=100)

    # Run tests
    results = {
        'device': device,
        'tests': {}
    }

    # Test 1: Model loading
    test1_passed = test_model_loading(device)
    results['tests']['model_loading'] = test1_passed

    if not test1_passed:
        print("\n✗ CRITICAL: Model loading failed. Cannot continue.")
        return results

    # Test 2: Data loading
    test2_passed = test_data_loading(data_root)
    results['tests']['data_loading'] = test2_passed

    if not test2_passed:
        print("\n✗ CRITICAL: Data loading failed. Cannot continue.")
        return results

    # Test 3: Source baseline
    test3_passed, source_results = test_source_baseline(data_root, device)
    results['tests']['source_baseline'] = test3_passed
    if source_results:
        results['source_results'] = source_results

    # Test 4: TENT baseline
    test4_passed, tent_results = test_tent_baseline(data_root, device)
    results['tests']['tent_baseline'] = test4_passed
    if tent_results:
        results['tent_results'] = tent_results

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    all_passed = all(results['tests'].values())

    for test_name, passed in results['tests'].items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:30s}: {status}")

    if all_passed:
        print("\n" + "="*80)
        print("✓ ALL TESTS PASSED")
        print("="*80)
        print("\nThe implementation is verified and ready for evaluation on real ImageNet-C data.")
    else:
        print("\n" + "="*80)
        print("✗ SOME TESTS FAILED")
        print("="*80)
        print("\nPlease review the errors above.")

    # Save test results (convert numpy types to Python types)
    def convert_to_json_serializable(obj):
        """Convert numpy types to Python native types."""
        if isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    results_file = '../results/test_results.json'
    Path(results_file).parent.mkdir(exist_ok=True, parents=True)
    with open(results_file, 'w') as f:
        json.dump(convert_to_json_serializable(results), f, indent=2)

    print(f"\nTest results saved to: {results_file}")

    return results


if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Run all tests
    results = run_all_tests()
