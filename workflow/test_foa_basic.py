"""
Basic test of FOA components without CMA-ES.
This verifies the core mechanisms work correctly.
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Add parent to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("Testing FOA basic components...")
print("=" * 80)

# Test 1: Activation Hook
print("\n1. Testing ActivationHook...")
from foa_method import ActivationHook

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 20)
        self.layer2 = nn.Linear(20, 30)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

model = DummyModel()
hook = ActivationHook()
hook.register_hooks(model, ['layer1', 'layer2'])

x = torch.randn(4, 10)
y = model(x)

print(f"  Captured activations: {list(hook.activations.keys())}")
print(f"  layer1 shape: {hook.activations['layer1'].shape}")
print(f"  layer2 shape: {hook.activations['layer2'].shape}")
hook.remove_hooks()
print("  ✓ ActivationHook working correctly")

# Test 2: Prompt Generator
print("\n2. Testing PromptGenerator...")
from foa_method import PromptGenerator

num_prompts = 5
embed_dim = 768
prompt_gen = PromptGenerator(num_prompts, embed_dim, device='cpu')

# Create random prompts
prompt_vector = np.random.randn(num_prompts * embed_dim) * 0.01
prompt_gen.set_prompts(prompt_vector)

# Test prepending
batch_size = 4
seq_length = 196
x = torch.randn(batch_size, seq_length, embed_dim)

output = prompt_gen.forward(x)
expected_length = seq_length + num_prompts

print(f"  Input shape: {x.shape}")
print(f"  Output shape: {output.shape}")
print(f"  Expected length: {expected_length}, Got: {output.shape[1]}")
assert output.shape == (batch_size, expected_length, embed_dim), "Shape mismatch!"
print("  ✓ PromptGenerator working correctly")

# Test 3: Source Statistics
print("\n3. Testing source statistics computation...")
from foa_method import compute_source_statistics

# Create dummy model and data
class SimpleViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.Linear(768, 768) for _ in range(3)
        ])

    def forward(self, x):
        # Simulate patch embedding
        b, c, h, w = x.shape
        x = x.view(b, c, -1).permute(0, 2, 1)  # [B, N, C]
        x = nn.functional.adaptive_avg_pool1d(x.permute(0, 2, 1), 768).permute(0, 2, 1)

        for block in self.blocks:
            x = block(x)
        return x

dummy_model = SimpleViT()
dummy_data = torch.randn(16, 3, 224, 224)
dummy_labels = torch.zeros(16, dtype=torch.long)
dataset = torch.utils.data.TensorDataset(dummy_data, dummy_labels)
loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False)

layer_names = ['blocks.0', 'blocks.1', 'blocks.2']
stats = compute_source_statistics(dummy_model, loader, layer_names, 'cpu', num_samples=16)

print(f"  Computed statistics for {len(stats)} layers")
for name, stat in stats.items():
    print(f"    {name}: mean shape={stat['mean'].shape}, std shape={stat['std'].shape}")
print("  ✓ Source statistics computation working correctly")

# Test 4: Save/Load statistics
print("\n4. Testing save/load statistics...")
from foa_method import save_source_statistics, load_source_statistics
import tempfile

with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp:
    temp_path = tmp.name

save_source_statistics(stats, temp_path)
loaded_stats = load_source_statistics(temp_path)

assert len(loaded_stats) == len(stats), "Stats count mismatch!"
print("  ✓ Save/load working correctly")

# Clean up
os.remove(temp_path)

print("\n" + "=" * 80)
print("All basic FOA component tests passed! ✓")
print("=" * 80)
