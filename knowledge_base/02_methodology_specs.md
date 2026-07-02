# Replication Methodology: Test-Time Model Adaptation with Only Forward Passes (FOA)

This document specifies the exact methodology for replicating the FOA paper. No deviations from this blueprint are permitted.

## 1. Core Methodology & Architecture

The implementation must replicate the Forward-Optimization Adaptation (FOA) method applied to a pre-trained Vision Transformer (ViT-Base) model. The model's original weights must remain frozen throughout the adaptation process. The method consists of two and only two components, which operate using only forward passes:

1.  **Forward-Only Prompt Adaptation**: A set of `N_p` learnable prompt embeddings, `p`, must be prepended to the input sequence of the ViT. These prompts are to be optimized on-the-fly for each batch of test data using the **Covariance Matrix Adaptation Evolution Strategy (CMA-ES)**, a derivative-free black-box optimization algorithm.
2.  **Back-to-Source Activation Shifting**: Following the prompt application and forward pass, the activation of the final `[CLS]` token (before the final prediction head) must be directly modified. This is achieved by shifting the activation's mean and standard deviation to align with pre-computed statistics from the source training data.

## 2. Mathematical Formulation & Objective Function

The optimization of the input prompt `p` must be guided by the following fitness function, `L`, which is minimized by the CMA-ES algorithm. This exact function must be implemented for the unsupervised setting.

For a given test batch `X_t`, the fitness function is:

`L(f_Θ(p; X_t)) = Σ_{x ∈ X_t} Σ_{c ∈ C} -ŷ_c * log(ŷ_c) + λ * Σ_{i=1 to N} (||μ_i(X_t) - μ_i^S||_2 + ||σ_i(X_t) - σ_i^S||_2)`

Where:
- `f_Θ` is the frozen pre-trained ViT model.
- `p` is the input prompt being optimized.
- `ŷ_c` is the predicted probability for class `c`.
- `μ_i(X_t)` and `σ_i(X_t)` are the channel-wise mean and standard deviation of the `[CLS]` token activation at layer `i` for the current test batch.
- `μ_i^S` and `σ_i^S` are the pre-computed channel-wise mean and standard deviation of the `[CLS]` token activation at layer `i` for the source data.
- `λ` is a hyperparameter balancing the two terms.
- `N` is the number of layers in the ViT.

## 3. Hyperparameters & Constraints

The implementation must adhere to the following hyperparameters as specified in the original paper:

-   **Model**: Vision Transformer (ViT-Base). Experiments must be run on both full-precision (32-bit) and quantized (8-bit) versions.
-   **Prompt Length (`N_p`)**: The number of prompt embeddings. This should be treated as a key hyperparameter to be set according to the paper's experiments.
-   **Fitness Function Trade-off (`λ`)**: The weight for the activation discrepancy term. This should be set according to the paper's experiments.
-   **CMA-ES Population Size (`K`)**: The number of candidate prompt solutions sampled in each CMA-ES iteration. This should be set according to the paper's experiments.
-   **Source Statistics**: The source domain statistics (`μ_i^S`, `σ_i^S`) must be pre-computed using a small, unlabeled set of in-distribution samples (e.g., 32 samples for ImageNet, as specified in the paper).

## 4. Target Datasets & Environments

The replication must be evaluated on the following image classification benchmarks under distribution shift:

-   **Primary Benchmark**: **ImageNet-C**. The model must be evaluated against all 15 corruption types at all 5 severity levels.
-   **Secondary Benchmarks (as per Appendix)**: ImageNet-R, ImageNet-Sketch, and ImageNet-A.

The use of simplified or toy datasets is forbidden. The evaluation must be performed on the full, standard versions of these benchmarks.
