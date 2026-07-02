# Literature Review: Test-Time Adaptation (TTA)

## 1. Research Context

The target paper, "Test-Time Model Adaptation with Only Forward Passes" (FOA), addresses a critical limitation in the field of Test-Time Adaptation (TTA). While existing TTA methods can improve model performance on distributionally shifted data, they typically rely on gradient-based backpropagation to update model weights. This requirement makes them unsuitable for deployment on resource-constrained edge devices, quantized models, or specialized hardware where gradients are unavailable or computationally prohibitive.

FOA introduces a novel paradigm that performs adaptation using only forward passes. This is achieved by optimizing a small input prompt with a derivative-free evolution strategy (CMA-ES) and directly shifting the model's internal activations to match source statistics. This approach maintains the benefits of adaptation while being compatible with a much wider range of real-world deployment scenarios.

## 2. Required Baselines for Replication

A faithful replication of the FOA paper requires comparison against the specific state-of-the-art TTA methods evaluated in the original work. The downstream computational agents **must** implement and evaluate the following baselines:

*   **Source (No Adaptation)**:
    *   **Description**: This is the fundamental baseline representing the performance of the original, un-adapted pre-trained model on the target test data. It serves to quantify the initial performance degradation caused by the distribution shift.
    *   **Implementation**: The pre-trained ViT-Base model is used for inference without any modifications or updates.

*   **TENT (Tent)**:
    *   **Description**: A widely-cited, gradient-based TTA method that adapts a model by minimizing the entropy of its predictions. It updates only the affine parameters of the model's Batch Normalization layers.
    *   **Reference**: Wang, D., et al. (2021). *Tent: Fully Test-time Adaptation by Entropy Minimization*. ICLR.

*   **SAR (Stochastic Answer-Regularization)**:
    *   **Description**: A gradient-based method that adapts the model by generating pseudo-labels. It sharpens the predictions on augmented views of a test sample and uses the resulting confident prediction as a target to update the model via backpropagation.
    *   **Reference**: Niu, S., et al. (2023). *Uncertainty-Calibrated Test-Time Model Adaptation Without Forgetting*. 

*   **MEMO (Memory-based Model)**:
    *   **Description**: A gradient-based method that leverages a memory bank of feature representations from recent, high-confidence test samples to guide the adaptation of the current sample.
    *   **Reference**: Zhang, L., et al. (2022). *MEMO: Test-Time Adaptation with a Memory Bank*. CVPR.

*   **BN Adapt (Batch Norm Adaptation)**:
    *   **Description**: A simple, gradient-free baseline that adapts the statistics (mean and variance) of the Batch Normalization layers using the statistics of the incoming test batch. This is a common baseline to show the benefits of more sophisticated adaptation methods.
