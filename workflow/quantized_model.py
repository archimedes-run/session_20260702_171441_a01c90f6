"""
8-bit Quantized ViT-Base Model

This module implements quantization for ViT-Base to reduce memory footprint
and potentially speed up inference, as mentioned in the FOA paper.
"""

import torch
import torch.nn as nn
import timm
from torch.quantization import quantize_dynamic


def create_quantized_vit(model_name: str = 'vit_base_patch16_224',
                        pretrained: bool = True,
                        quantization_type: str = 'dynamic') -> nn.Module:
    """
    Create a quantized version of ViT-Base.

    Args:
        model_name: Model architecture name
        pretrained: Load pretrained weights
        quantization_type: Type of quantization ('dynamic' or 'static')

    Returns:
        Quantized model
    """
    # Load pre-trained model
    model = timm.create_model(model_name, pretrained=pretrained)
    model.eval()

    print(f"Original model size: {get_model_size(model):.2f} MB")

    # Apply dynamic quantization
    # This quantizes Linear and LayerNorm layers to int8
    if quantization_type == 'dynamic':
        quantized_model = quantize_dynamic(
            model,
            {nn.Linear},  # Quantize only Linear layers
            dtype=torch.qint8
        )
    else:
        # For static quantization, we would need calibration data
        raise NotImplementedError("Static quantization requires calibration data")

    print(f"Quantized model size: {get_model_size(quantized_model):.2f} MB")

    return quantized_model


def get_model_size(model: nn.Module) -> float:
    """
    Calculate model size in MB.

    Args:
        model: PyTorch model

    Returns:
        Size in MB
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb


def test_quantized_model():
    """Test quantized model accuracy."""
    print("=" * 80)
    print("Testing Quantized ViT-Base Model")
    print("=" * 80)

    # Load original model
    print("\nLoading original model...")
    model_fp32 = timm.create_model('vit_base_patch16_224', pretrained=True)
    model_fp32.eval()

    # Create quantized model
    print("\nCreating quantized model...")
    model_int8 = create_quantized_vit()

    # Create dummy input
    dummy_input = torch.randn(4, 3, 224, 224)

    # Compare outputs
    print("\nComparing outputs...")
    with torch.no_grad():
        output_fp32 = model_fp32(dummy_input)
        output_int8 = model_int8(dummy_input)

    # Compute difference
    max_diff = (output_fp32 - output_int8).abs().max().item()
    mean_diff = (output_fp32 - output_int8).abs().mean().item()

    print(f"Max output difference: {max_diff:.6f}")
    print(f"Mean output difference: {mean_diff:.6f}")

    # Check predictions match
    pred_fp32 = output_fp32.argmax(dim=1)
    pred_int8 = output_int8.argmax(dim=1)
    match_rate = (pred_fp32 == pred_int8).float().mean().item()

    print(f"Prediction match rate: {match_rate:.4f}")

    # Measure inference time
    import time

    num_runs = 100
    print(f"\nMeasuring inference time ({num_runs} runs)...")

    # FP32
    start = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model_fp32(dummy_input)
    fp32_time = (time.time() - start) / num_runs

    # INT8
    start = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model_int8(dummy_input)
    int8_time = (time.time() - start) / num_runs

    print(f"FP32 inference time: {fp32_time*1000:.2f} ms")
    print(f"INT8 inference time: {int8_time*1000:.2f} ms")
    print(f"Speedup: {fp32_time/int8_time:.2f}x")

    print("\n" + "=" * 80)
    print("Quantization test complete!")
    print("=" * 80)


if __name__ == "__main__":
    test_quantized_model()
