# LBEADS-NET v1: Learnable BEADS Network

An unrolled neural network implementation of the BEADS (Baseline Estimation And Denoising with Sparsity) algorithm for chromatogram baseline correction.

## Overview

LBEADS-NET transforms the iterative BEADS algorithm into a deep neural network where each iteration becomes a trainable layer. This enables:

1. **End-to-end learning**: Optimize regularization parameters directly from data
2. **Layer-wise parameters**: Each unrolled iteration can have its own learnable parameters
3. **Faster inference**: GPU acceleration and potential for reduced iterations
4. **Differentiability**: Enable supervised learning with task-specific losses

## Architecture

```
Input y (noisy signal)
    │
    ▼
┌──────────────────────────────────────┐
│  Unrolled Layer 1 (λ₀¹, λ₁¹, λ₂¹, r¹) │
│  x¹ = BEADS_iteration(x⁰, y)          │
└──────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────┐
│  Unrolled Layer 2 (λ₀², λ₁², λ₂², r²) │
│  x² = BEADS_iteration(x¹, y)          │
└──────────────────────────────────────┘
    │
    ⋮
    │
    ▼
┌──────────────────────────────────────┐
│  Unrolled Layer K (λ₀ᴷ, λ₁ᴷ, λ₂ᴷ, rᴷ) │
│  xᴷ = BEADS_iteration(xᴷ⁻¹, y)        │
└──────────────────────────────────────┘
    │
    ▼
Output: x (sparse signal), f (baseline)
```

## Files

- `lbeads_net.py`: Core model implementations
  - `LBEADS_NET`: Exact unrolled version (matches original BEADS)
  - `LBEADS_NET_Fast`: Faster gradient-descent version (fully differentiable)
  
- `train.py`: Training script with example usage
- `demo.py`: Demonstration comparing LBEADS-NET with original BEADS

## Usage

### Basic Usage

```python
from lbeads_net import LBEADS_NET, LBEADS_NET_Fast

# Create model
model = LBEADS_NET_Fast(
    N=3800,          # Signal length
    d=1,             # Filter order
    fc=0.006,        # Cut-off frequency
    num_layers=10,   # Number of unrolled iterations
    init_lam0=0.4,   # Initial regularization parameters
    init_lam1=4.0,
    init_lam2=3.2,
    init_r=6.0       # Asymmetry ratio
)

# Forward pass
x_signal, f_baseline = model(y_noisy)
```

### Training

```python
# Define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training loop
for epoch in range(num_epochs):
    x_pred, f_pred = model(y_noisy)
    loss = criterion(x_pred, x_target)
    loss.backward()
    optimizer.step()
```

### Running Demo

```bash
cd Initial_implementations/LBEADS_NETv1
python demo.py
```

### Running Training

```bash
python train.py
```

## Learnable Parameters

Each unrolled layer can learn:

| Parameter | Description | Constraint |
|-----------|-------------|------------|
| λ₀ | Asymmetric penalty weight | > 0 (log parameterization) |
| λ₁ | First derivative penalty weight | > 0 (log parameterization) |
| λ₂ | Second derivative penalty weight | > 0 (log parameterization) |
| r | Asymmetry ratio | > 0 (optional) |
| step_size | Gradient step (fast version) | > 0 |

## Two Model Variants

### 1. LBEADS_NET (Exact)
- Solves the exact linear system at each layer
- Matches original BEADS algorithm exactly
- More accurate but slower
- Uses scipy sparse solvers (less GPU-friendly)

### 2. LBEADS_NET_Fast
- Uses gradient descent updates instead of exact solve
- Fully differentiable with PyTorch operations
- Much faster, especially on GPU
- Learnable step sizes for convergence control

## Requirements

- PyTorch >= 1.9
- NumPy
- SciPy
- Matplotlib (for visualization)

## References

Original BEADS Algorithm:
> Xiaoran Ning, Ivan W. Selesnick, Laurent Duval  
> "Chromatogram baseline estimation and denoising using sparsity (BEADS)"  
> Chemometrics and Intelligent Laboratory Systems (2014)  
> doi: 10.1016/j.chemolab.2014.09.014

Algorithm Unrolling:
> Monga, V., Li, Y., & Eldar, Y. C. (2021)  
> "Algorithm unrolling: Interpretable, efficient deep learning for signal and image processing"  
> IEEE Signal Processing Magazine, 38(2), 18-44.
