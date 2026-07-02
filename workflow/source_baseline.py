"""
Source Baseline: No adaptation.
Simply evaluates the pre-trained ViT-Base model on ImageNet-C without any updates.
"""
import torch
import torch.nn as nn
import timm
from typing import Dict, Tuple
from tqdm import tqdm
import numpy as np


class SourceModel:
    """
    Source baseline model (no adaptation).

    This baseline simply evaluates the pre-trained ViT-Base model
    on the target test data without any modifications or updates.

    Args:
        model_name: Name of the pre-trained model (default: vit_base_patch16_224)
        device: Device to run on (cuda, mps, or cpu)
    """
    def __init__(self, model_name: str = 'vit_base_patch16_224', device: str = 'cuda'):
        self.device = device
        self.model_name = model_name

        # Load pre-trained ViT-Base model
        print(f"Loading pre-trained {model_name}...")
        self.model = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=1000
        )
        self.model = self.model.to(device)
        self.model.eval()

        print(f"Model loaded successfully on {device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()) / 1e6:.2f}M")

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x: Input tensor [batch_size, 3, 224, 224]

        Returns:
            Logits [batch_size, 1000]
        """
        return self.model(x)

    @torch.no_grad()
    def evaluate(self, data_loader) -> Dict[str, float]:
        """
        Evaluate the model on a dataset.

        Args:
            data_loader: DataLoader for the test set

        Returns:
            Dictionary containing evaluation metrics
        """
        self.model.eval()

        correct = 0
        total = 0
        all_probs = []
        all_labels = []

        pbar = tqdm(data_loader, desc="Evaluating Source")

        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            logits = self.predict(images)
            probs = torch.softmax(logits, dim=1)

            # Compute accuracy
            _, predicted = torch.max(logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            # Store for metrics
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

            # Update progress bar
            current_acc = 100 * correct / total
            pbar.set_postfix({'Accuracy': f'{current_acc:.2f}%'})

        # Compute final metrics
        accuracy = 100 * correct / total

        all_probs = np.concatenate(all_probs, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        # Compute average entropy
        entropy = -np.sum(all_probs * np.log(all_probs + 1e-10), axis=1)
        avg_entropy = np.mean(entropy)

        # Compute average confidence
        avg_confidence = np.mean(np.max(all_probs, axis=1))

        results = {
            'accuracy': accuracy,
            'error_rate': 100 - accuracy,
            'avg_entropy': avg_entropy,
            'avg_confidence': avg_confidence,
            'total_samples': total
        }

        return results

    def reset(self):
        """
        Reset model to initial state.
        For Source baseline, this is a no-op since we don't modify the model.
        """
        pass


def evaluate_source_baseline(
    data_root: str,
    corruption: str,
    severity: int,
    batch_size: int = 64,
    device: str = 'cuda'
) -> Dict[str, float]:
    """
    Evaluate Source baseline on a specific corruption and severity.

    Args:
        data_root: Root directory for ImageNet-C
        corruption: Corruption type
        severity: Severity level (1-5)
        batch_size: Batch size for evaluation
        device: Device to run on

    Returns:
        Dictionary of evaluation results
    """
    from data_loader import create_imagenet_c_loader

    # Create data loader
    data_loader = create_imagenet_c_loader(
        data_root,
        corruption,
        severity,
        batch_size=batch_size,
        num_workers=4,
        shuffle=False
    )

    # Create model
    model = SourceModel(device=device)

    # Evaluate
    print(f"\nEvaluating Source baseline on {corruption} (severity {severity})")
    results = model.evaluate(data_loader)

    print(f"\nResults:")
    print(f"  Accuracy: {results['accuracy']:.2f}%")
    print(f"  Error Rate: {results['error_rate']:.2f}%")
    print(f"  Avg Entropy: {results['avg_entropy']:.4f}")
    print(f"  Avg Confidence: {results['avg_confidence']:.4f}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Source Baseline Evaluation')
    parser.add_argument('--data_root', type=str, default='./data/imagenet-c',
                        help='Root directory for ImageNet-C')
    parser.add_argument('--corruption', type=str, default='gaussian_noise',
                        help='Corruption type')
    parser.add_argument('--severity', type=int, default=1,
                        help='Severity level (1-5)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda, mps, or cpu)')

    args = parser.parse_args()

    # Auto-detect device
    if args.device == 'cuda' and not torch.cuda.is_available():
        if torch.backends.mps.is_available():
            args.device = 'mps'
        else:
            args.device = 'cpu'

    print(f"Using device: {args.device}")

    # Run evaluation
    results = evaluate_source_baseline(
        args.data_root,
        args.corruption,
        args.severity,
        args.batch_size,
        args.device
    )
