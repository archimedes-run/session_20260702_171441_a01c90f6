"""
Forward-Optimization Adaptation (FOA) Implementation

This module implements the FOA method for test-time adaptation using only forward passes.
Key components:
1. CMA-ES-based prompt adaptation (derivative-free optimization)
2. Back-to-Source Activation Shifting
3. Composite fitness function (entropy + activation discrepancy)

Mathematical Formulation:
L(f_Θ(p; X_t)) = Σ -ŷ_c * log(ŷ_c) + λ * Σ (||μ_i(X_t) - μ_i^S||_2 + ||σ_i(X_t) - σ_i^S||_2)

References:
- Test-Time Model Adaptation with Only Forward Passes (FOA 2024)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cma
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import timm
import json
import os


class ActivationHook:
    """Hook to capture intermediate activations from the model."""

    def __init__(self):
        self.activations: Dict[str, torch.Tensor] = {}
        self.hooks = []

    def get_activation(self, name: str):
        """Create a hook function that captures activation."""
        def hook(module, input, output):
            # For ViT, we need the [CLS] token (first token)
            if isinstance(output, torch.Tensor):
                # Extract [CLS] token: shape [B, N, D] -> [B, D]
                if len(output.shape) == 3:
                    self.activations[name] = output[:, 0, :].detach()
                else:
                    self.activations[name] = output.detach()
        return hook

    def register_hooks(self, model: nn.Module, layer_names: List[str]):
        """Register hooks to specified layers."""
        for name, module in model.named_modules():
            if name in layer_names:
                handle = module.register_forward_hook(self.get_activation(name))
                self.hooks.append(handle)

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def clear_activations(self):
        """Clear stored activations."""
        self.activations = {}


class PromptGenerator(nn.Module):
    """
    Generates learnable prompt embeddings for ViT.
    Prompts are prepended to the input sequence.
    """

    def __init__(self, num_prompts: int, embed_dim: int, device: str = 'cuda'):
        super().__init__()
        self.num_prompts = num_prompts
        self.embed_dim = embed_dim
        self.device = device

        # Initialize prompt embeddings (will be optimized by CMA-ES)
        self.prompt_embeddings = None

    def set_prompts(self, prompt_vector: np.ndarray):
        """
        Set prompt embeddings from a flat vector (from CMA-ES).

        Args:
            prompt_vector: Flat numpy array of shape (num_prompts * embed_dim,)
        """
        # Reshape to (num_prompts, embed_dim)
        prompts = prompt_vector.reshape(self.num_prompts, self.embed_dim)
        self.prompt_embeddings = torch.FloatTensor(prompts).to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Prepend prompts to input sequence.

        Args:
            x: Input tensor of shape [B, N, D] where N is sequence length

        Returns:
            Tensor of shape [B, N + num_prompts, D]
        """
        batch_size = x.size(0)

        # Expand prompts for batch
        prompts = self.prompt_embeddings.unsqueeze(0).expand(batch_size, -1, -1)

        # Concatenate prompts with input
        return torch.cat([prompts, x], dim=1)


class FOAAdapter:
    """
    Forward-Optimization Adaptation using CMA-ES and activation shifting.
    """

    def __init__(
        self,
        model: nn.Module,
        source_stats: Dict[str, Dict[str, torch.Tensor]],
        num_prompts: int = 10,
        lambda_activation: float = 0.1,
        cma_population_size: int = 10,
        cma_max_iterations: int = 20,
        device: str = 'cuda'
    ):
        """
        Initialize FOA adapter.

        Args:
            model: Pre-trained ViT model (frozen)
            source_stats: Pre-computed source domain statistics
                         {layer_name: {'mean': tensor, 'std': tensor}}
            num_prompts: Number of prompt embeddings (N_p)
            lambda_activation: Weight for activation discrepancy term (λ)
            cma_population_size: CMA-ES population size (K)
            cma_max_iterations: Maximum CMA-ES iterations
            device: Device to run on
        """
        self.model = model.to(device)
        self.model.eval()  # Always in eval mode (frozen)
        self.source_stats = source_stats
        self.num_prompts = num_prompts
        self.lambda_activation = lambda_activation
        self.cma_population_size = cma_population_size
        self.cma_max_iterations = cma_max_iterations
        self.device = device

        # Get embedding dimension from model
        self.embed_dim = self.model.patch_embed.proj.out_channels

        # Initialize prompt generator
        self.prompt_gen = PromptGenerator(num_prompts, self.embed_dim, device)

        # Setup activation hooks
        self.hook_manager = ActivationHook()
        self.layer_names = list(source_stats.keys())

        # Track statistics
        self.adaptation_history = []

    def compute_entropy(self, logits: torch.Tensor) -> float:
        """
        Compute prediction entropy: Σ -ŷ_c * log(ŷ_c)

        Args:
            logits: Model logits [B, C]

        Returns:
            Mean entropy across batch
        """
        probs = F.softmax(logits, dim=1)
        # Add small epsilon to avoid log(0)
        log_probs = torch.log(probs + 1e-10)
        entropy = -(probs * log_probs).sum(dim=1).mean()
        return entropy.item()

    def compute_activation_discrepancy(self, activations: Dict[str, torch.Tensor]) -> float:
        """
        Compute activation statistics discrepancy:
        Σ_i (||μ_i(X_t) - μ_i^S||_2 + ||σ_i(X_t) - σ_i^S||_2)

        Args:
            activations: Current batch activations {layer_name: tensor [B, D]}

        Returns:
            Total discrepancy across all layers
        """
        total_discrepancy = 0.0

        for layer_name in self.layer_names:
            if layer_name not in activations:
                continue

            # Current batch statistics
            act = activations[layer_name]  # [B, D]
            mu_current = act.mean(dim=0)  # [D]
            sigma_current = act.std(dim=0)  # [D]

            # Source statistics
            mu_source = self.source_stats[layer_name]['mean'].to(self.device)
            sigma_source = self.source_stats[layer_name]['std'].to(self.device)

            # L2 norm of differences
            mean_diff = torch.norm(mu_current - mu_source, p=2).item()
            std_diff = torch.norm(sigma_current - sigma_source, p=2).item()

            total_discrepancy += (mean_diff + std_diff)

        return total_discrepancy

    def fitness_function(self, prompt_vector: np.ndarray, batch: torch.Tensor) -> float:
        """
        CMA-ES fitness function to minimize:
        L = entropy + λ * activation_discrepancy

        Args:
            prompt_vector: Flat prompt parameters from CMA-ES
            batch: Input batch [B, C, H, W]

        Returns:
            Fitness value (to be minimized)
        """
        with torch.no_grad():
            # Set prompts
            self.prompt_gen.set_prompts(prompt_vector)

            # Register hooks to capture activations
            self.hook_manager.register_hooks(self.model, self.layer_names)
            self.hook_manager.clear_activations()

            # Forward pass with prompts
            # Note: We need to modify the forward pass to include prompts
            # For ViT, we'll prepend prompts after patch embedding
            logits = self._forward_with_prompts(batch)

            # Get activations
            activations = self.hook_manager.activations

            # Remove hooks
            self.hook_manager.remove_hooks()

            # Compute fitness components
            entropy = self.compute_entropy(logits)
            activation_disc = self.compute_activation_discrepancy(activations)

            # Combined fitness
            fitness = entropy + self.lambda_activation * activation_disc

        return fitness

    def _forward_with_prompts(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ViT with prompt injection.

        Args:
            x: Input images [B, C, H, W]

        Returns:
            Logits [B, num_classes]
        """
        # This is a simplified implementation
        # In practice, you'd need to hook into the ViT's embedding layer
        # and inject prompts before the first transformer block

        # For timm ViT models, we can use forward_features to get embeddings
        # then inject prompts

        # Get patch embeddings
        x = self.model.patch_embed(x)

        # Add class token
        cls_token = self.model.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        # Add positional embeddings
        x = x + self.model.pos_embed

        # Inject prompts AFTER cls token and pos embed
        if self.prompt_gen.prompt_embeddings is not None:
            # Insert prompts after CLS token
            cls = x[:, :1, :]  # [B, 1, D]
            patches = x[:, 1:, :]  # [B, N, D]

            # Expand prompts
            batch_size = x.size(0)
            prompts = self.prompt_gen.prompt_embeddings.unsqueeze(0).expand(batch_size, -1, -1)

            # Concatenate: [CLS] + [Prompts] + [Patches]
            x = torch.cat([cls, prompts, patches], dim=1)

        # Apply dropout
        x = self.model.pos_drop(x)

        # Pass through transformer blocks
        x = self.model.blocks(x)
        x = self.model.norm(x)

        # Extract CLS token and classify
        x = x[:, 0]  # [CLS] token
        x = self.model.head(x)

        return x

    def apply_activation_shifting(self, activations: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Apply back-to-source activation shifting.
        Shift mean and std of activations to match source statistics.

        Args:
            activations: Current activations {layer_name: tensor}

        Returns:
            Shifted activations
        """
        shifted = {}

        for layer_name, act in activations.items():
            if layer_name in self.source_stats:
                # Current statistics
                mu_current = act.mean(dim=0, keepdim=True)
                sigma_current = act.std(dim=0, keepdim=True) + 1e-6

                # Source statistics
                mu_source = self.source_stats[layer_name]['mean'].to(self.device).unsqueeze(0)
                sigma_source = self.source_stats[layer_name]['std'].to(self.device).unsqueeze(0) + 1e-6

                # Normalize and shift
                act_normalized = (act - mu_current) / sigma_current
                act_shifted = act_normalized * sigma_source + mu_source

                shifted[layer_name] = act_shifted
            else:
                shifted[layer_name] = act

        return shifted

    def adapt_batch(self, batch: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        """
        Adapt model to a test batch using CMA-ES optimization.

        Args:
            batch: Input images [B, C, H, W]
            labels: Optional ground truth labels for evaluation

        Returns:
            logits: Adapted predictions [B, num_classes]
            info: Dictionary with adaptation statistics
        """
        batch = batch.to(self.device)

        # Initialize CMA-ES
        initial_prompt = np.random.randn(self.num_prompts * self.embed_dim) * 0.01
        sigma0 = 0.1  # Initial step size

        options = {
            'popsize': self.cma_population_size,
            'maxiter': self.cma_max_iterations,
            'verbose': -1,  # Suppress output
            'seed': 42
        }

        # Run CMA-ES optimization
        es = cma.CMAEvolutionStrategy(initial_prompt, sigma0, options)

        iteration_losses = []
        for iteration in range(self.cma_max_iterations):
            # Ask for new candidate solutions
            solutions = es.ask()

            # Evaluate fitness for each solution
            fitness_values = [self.fitness_function(sol, batch) for sol in solutions]

            # Tell CMA-ES the fitness values
            es.tell(solutions, fitness_values)

            # Track best loss
            iteration_losses.append(min(fitness_values))

            # Early stopping if converged
            if es.stop():
                break

        # Get best solution
        best_prompt = es.result.xbest

        # Final forward pass with best prompts
        with torch.no_grad():
            self.prompt_gen.set_prompts(best_prompt)

            # Register hooks
            self.hook_manager.register_hooks(self.model, self.layer_names)
            self.hook_manager.clear_activations()

            # Forward pass
            logits = self._forward_with_prompts(batch)

            # Get activations and apply shifting
            activations = self.hook_manager.activations
            shifted_activations = self.apply_activation_shifting(activations)

            # Remove hooks
            self.hook_manager.remove_hooks()

            # Compute final metrics
            entropy = self.compute_entropy(logits)
            activation_disc = self.compute_activation_discrepancy(activations)

            # Compute accuracy if labels provided
            accuracy = None
            if labels is not None:
                labels = labels.to(self.device)
                preds = logits.argmax(dim=1)
                accuracy = (preds == labels).float().mean().item()

        # Package info
        info = {
            'entropy': entropy,
            'activation_discrepancy': activation_disc,
            'fitness': entropy + self.lambda_activation * activation_disc,
            'cma_iterations': iteration + 1,
            'iteration_losses': iteration_losses,
            'accuracy': accuracy
        }

        self.adaptation_history.append(info)

        return logits, info

    def reset(self):
        """Reset adaptation history."""
        self.adaptation_history = []
        self.hook_manager.clear_activations()


def compute_source_statistics(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    layer_names: List[str],
    device: str = 'cuda',
    num_samples: int = 32
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Pre-compute source domain statistics from clean ImageNet samples.

    Args:
        model: Pre-trained ViT model
        dataloader: DataLoader for clean ImageNet samples
        layer_names: Names of layers to track
        device: Device to run on
        num_samples: Number of samples to use (default: 32 as per paper)

    Returns:
        Dictionary: {layer_name: {'mean': tensor, 'std': tensor}}
    """
    model = model.to(device)
    model.eval()

    # Activation hook
    hook_manager = ActivationHook()
    hook_manager.register_hooks(model, layer_names)

    # Collect activations
    layer_activations = {name: [] for name in layer_names}

    samples_collected = 0
    with torch.no_grad():
        for batch, _ in tqdm(dataloader, desc="Computing source statistics"):
            if samples_collected >= num_samples:
                break

            batch = batch.to(device)
            hook_manager.clear_activations()

            # Forward pass
            _ = model(batch)

            # Collect activations
            for layer_name in layer_names:
                if layer_name in hook_manager.activations:
                    layer_activations[layer_name].append(
                        hook_manager.activations[layer_name].cpu()
                    )

            samples_collected += batch.size(0)

    hook_manager.remove_hooks()

    # Compute statistics
    source_stats = {}
    for layer_name in layer_names:
        if layer_activations[layer_name]:
            # Concatenate all activations
            all_acts = torch.cat(layer_activations[layer_name], dim=0)  # [N, D]

            # Compute mean and std
            mean = all_acts.mean(dim=0)  # [D]
            std = all_acts.std(dim=0)    # [D]

            source_stats[layer_name] = {
                'mean': mean,
                'std': std
            }

    return source_stats


def save_source_statistics(stats: Dict, filepath: str):
    """Save source statistics to disk."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(stats, filepath)
    print(f"Source statistics saved to {filepath}")


def load_source_statistics(filepath: str) -> Dict:
    """Load source statistics from disk."""
    stats = torch.load(filepath)
    print(f"Source statistics loaded from {filepath}")
    return stats


if __name__ == "__main__":
    """Test FOA implementation with dummy data."""

    print("Testing FOA Implementation...")
    print("=" * 80)

    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Device: {device}")

    # Load model
    print("\nLoading ViT-Base model...")
    model = timm.create_model('vit_base_patch16_224', pretrained=True)
    model.eval()
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")

    # Define layers to track (using ViT block outputs)
    layer_names = [f'blocks.{i}' for i in range(0, 12, 3)]  # Track every 3rd block
    print(f"\nTracking layers: {layer_names}")

    # Create dummy source data
    print("\nCreating dummy source data...")
    num_source_samples = 32
    source_data = torch.randn(num_source_samples, 3, 224, 224)
    source_dataset = torch.utils.data.TensorDataset(source_data, torch.zeros(num_source_samples))
    source_loader = torch.utils.data.DataLoader(source_dataset, batch_size=8, shuffle=False)

    # Compute source statistics
    print("\nComputing source statistics...")
    source_stats = compute_source_statistics(
        model, source_loader, layer_names, device, num_samples=num_source_samples
    )
    print(f"Source statistics computed for {len(source_stats)} layers")

    # Initialize FOA adapter
    print("\nInitializing FOA adapter...")
    foa = FOAAdapter(
        model=model,
        source_stats=source_stats,
        num_prompts=10,
        lambda_activation=0.1,
        cma_population_size=8,
        cma_max_iterations=10,
        device=device
    )
    print("FOA adapter initialized")

    # Test adaptation on dummy batch
    print("\nTesting adaptation on dummy batch...")
    test_batch = torch.randn(4, 3, 224, 224)
    test_labels = torch.randint(0, 1000, (4,))

    logits, info = foa.adapt_batch(test_batch, test_labels)

    print("\nAdaptation Results:")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Entropy: {info['entropy']:.4f}")
    print(f"  Activation Discrepancy: {info['activation_discrepancy']:.4f}")
    print(f"  Total Fitness: {info['fitness']:.4f}")
    print(f"  CMA-ES Iterations: {info['cma_iterations']}")
    print(f"  Accuracy: {info['accuracy']:.4f}" if info['accuracy'] else "  Accuracy: N/A")

    print("\n" + "=" * 80)
    print("FOA implementation test complete!")
