"""
TENT Baseline: Test-time Entropy Minimization.

TENT adapts the model by minimizing the entropy of predictions at test time.
It updates only the affine parameters (scale and bias) of normalization layers
via backpropagation.

Reference:
Wang, D., et al. (2021). Tent: Fully Test-time Adaptation by Entropy Minimization. ICLR.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import timm
from typing import Dict, List, Tuple
from tqdm import tqdm
import numpy as np
import copy


class TENT:
    """
    TENT: Test-time Entropy Minimization.

    Adapts the model by minimizing prediction entropy, updating only
    the affine parameters of normalization layers.

    Args:
        model_name: Name of pre-trained model (default: vit_base_patch16_224)
        learning_rate: Learning rate for adaptation (default: 1e-3)
        device: Device to run on (cuda, mps, or cpu)
    """
    def __init__(
        self,
        model_name: str = 'vit_base_patch16_224',
        learning_rate: float = 1e-3,
        device: str = 'cuda'
    ):
        self.device = device
        self.model_name = model_name
        self.learning_rate = learning_rate

        # Load pre-trained model
        print(f"Loading pre-trained {model_name}...")
        self.model = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=1000
        )
        self.model = self.model.to(device)

        # Store original model state for reset
        self.original_state = copy.deepcopy(self.model.state_dict())

        # Configure model for TENT
        self._configure_model()

        # Setup optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        print(f"Model loaded and configured for TENT on {device}")
        print(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")

    def _configure_model(self):
        """
        Configure model for TENT adaptation:
        1. Set model to train mode (for normalization statistics)
        2. Freeze all parameters
        3. Enable gradients only for affine parameters of normalization layers
        """
        self.model.train()

        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # Enable gradients for normalization layer affine parameters
        # For ViT, we need to identify LayerNorm layers
        norm_layers = []

        for name, module in self.model.named_modules():
            # Check for LayerNorm, BatchNorm2d, and GroupNorm
            if isinstance(module, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
                # Enable gradients for weight (scale) and bias (shift)
                if hasattr(module, 'weight') and module.weight is not None:
                    module.weight.requires_grad = True
                    norm_layers.append(f"{name}.weight")

                if hasattr(module, 'bias') and module.bias is not None:
                    module.bias.requires_grad = True
                    norm_layers.append(f"{name}.bias")

        print(f"Found {len(norm_layers)} normalization affine parameters:")
        print(f"  Sample: {norm_layers[:5]}")

    def entropy_loss(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy loss for TENT.

        Args:
            logits: Model output logits [batch_size, num_classes]

        Returns:
            Entropy loss (scalar)
        """
        # Compute softmax probabilities
        probs = torch.softmax(logits, dim=1)

        # Compute entropy: -sum(p * log(p))
        # Use clamp to avoid log(0)
        log_probs = torch.log(probs + 1e-10)
        entropy = -torch.sum(probs * log_probs, dim=1)

        # Return mean entropy across batch
        return entropy.mean()

    def adapt_batch(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adapt the model on a single batch and return predictions.

        Args:
            x: Input batch [batch_size, 3, 224, 224]

        Returns:
            Logits [batch_size, 1000]
        """
        # Forward pass
        logits = self.model(x)

        # Compute entropy loss
        loss = self.entropy_loss(logits)

        # Backward pass and update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return logits.detach()

    def evaluate(self, data_loader) -> Dict[str, float]:
        """
        Evaluate TENT on a dataset.
        Adapts on each batch during evaluation.

        Args:
            data_loader: DataLoader for the test set

        Returns:
            Dictionary containing evaluation metrics
        """
        correct = 0
        total = 0
        all_probs = []
        all_labels = []
        all_losses = []

        pbar = tqdm(data_loader, desc="Evaluating TENT")

        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Adapt and get predictions
            logits = self.adapt_batch(images)
            probs = torch.softmax(logits, dim=1)

            # Compute accuracy
            _, predicted = torch.max(logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            # Compute entropy for logging
            with torch.no_grad():
                entropy = self.entropy_loss(logits).item()
                all_losses.append(entropy)

            # Store for metrics
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

            # Update progress bar
            current_acc = 100 * correct / total
            pbar.set_postfix({
                'Accuracy': f'{current_acc:.2f}%',
                'Entropy': f'{entropy:.4f}'
            })

        # Compute final metrics
        accuracy = 100 * correct / total

        all_probs = np.concatenate(all_probs, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        # Compute average entropy
        entropy = -np.sum(all_probs * np.log(all_probs + 1e-10), axis=1)
        avg_entropy = np.mean(entropy)

        # Compute average confidence
        avg_confidence = np.mean(np.max(all_probs, axis=1))

        # Compute average adaptation loss
        avg_loss = np.mean(all_losses)

        results = {
            'accuracy': accuracy,
            'error_rate': 100 - accuracy,
            'avg_entropy': avg_entropy,
            'avg_confidence': avg_confidence,
            'avg_adaptation_loss': avg_loss,
            'total_samples': total
        }

        return results

    def reset(self):
        """
        Reset model to original pre-trained state.
        """
        self.model.load_state_dict(self.original_state)
        self._configure_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)


def evaluate_tent_baseline(
    data_root: str,
    corruption: str,
    severity: int,
    learning_rate: float = 1e-3,
    batch_size: int = 64,
    device: str = 'cuda'
) -> Dict[str, float]:
    """
    Evaluate TENT baseline on a specific corruption and severity.

    Args:
        data_root: Root directory for ImageNet-C
        corruption: Corruption type
        severity: Severity level (1-5)
        learning_rate: Learning rate for adaptation
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

    # Create TENT model
    model = TENT(learning_rate=learning_rate, device=device)

    # Evaluate
    print(f"\nEvaluating TENT baseline on {corruption} (severity {severity})")
    results = model.evaluate(data_loader)

    print(f"\nResults:")
    print(f"  Accuracy: {results['accuracy']:.2f}%")
    print(f"  Error Rate: {results['error_rate']:.2f}%")
    print(f"  Avg Entropy: {results['avg_entropy']:.4f}")
    print(f"  Avg Confidence: {results['avg_confidence']:.4f}")
    print(f"  Avg Adaptation Loss: {results['avg_adaptation_loss']:.4f}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='TENT Baseline Evaluation')
    parser.add_argument('--data_root', type=str, default='./data/imagenet-c',
                        help='Root directory for ImageNet-C')
    parser.add_argument('--corruption', type=str, default='gaussian_noise',
                        help='Corruption type')
    parser.add_argument('--severity', type=int, default=1,
                        help='Severity level (1-5)')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate for adaptation')
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
    results = evaluate_tent_baseline(
        args.data_root,
        args.corruption,
        args.severity,
        args.learning_rate,
        args.batch_size,
        args.device
    )
