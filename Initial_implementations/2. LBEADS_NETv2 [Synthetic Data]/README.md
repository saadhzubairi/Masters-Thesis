# LBEADS-NET v2: Learnable BEADS Network (Synthetic Data)

An unrolled neural network implementation of the BEADS (Baseline Estimation And Denoising with Sparsity) algorithm for chromatogram baseline correction, **trained on synthetic data with ground truth**.

## What's New in v2

This version trains LBEADS-NET on **synthetic chromatogram data** instead of a single real chromatogram:

- **Synthetic Data Generation**: Creates signals with known ground truth
  - Sparse Gaussian peaks (varying number, width, amplitude)
  - Smooth baseline drift (polynomial + sinusoidal)
  - Additive Gaussian noise (varying levels)
  
- **Supervised Learning**: Train with ground truth peaks as target
  - Signal model: `y = x_true (peaks) + f_true (baseline) + noise`
  - Target: `x_true` (the clean peaks)

- **Train/Test Split**: 80/20 split for proper evaluation

## Overview

LBEADS-NET transforms the iterative BEADS algorithm into a deep neural network where each iteration becomes a trainable layer. This enables:

1. **End-to-end learning**: Optimize regularization parameters directly from data
2. **Layer-wise parameters**: Each unrolled iteration can have its own learnable parameters
3. **Faster inference**: GPU acceleration and potential for reduced iterations
4. **Differentiability**: Enable supervised learning with task-specific losses

## Architecture

```
Input y (observed signal = peaks + baseline + noise)
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
Output: x (estimated peaks), f (estimated baseline)
```

## Files

- `lbeads_net.py`: Core model implementations
  - `LBEADS_NET`: Exact unrolled version (matches original BEADS)
  - `LBEADS_NET_Fast`: Faster gradient-descent version (fully differentiable)
  
- `train.py`: Training script with synthetic data generation
  - `SyntheticDataGenerator`: Generates peaks, baseline, noise
  - 80/20 train/test split
  - Saves trained model and metrics
  
- `demo.py`: Demonstration script
  - Loads trained model
  - Tests on new synthetic data
  - Visualizes predictions vs ground truth

## Usage

### Training on Synthetic Data

```bash
python train.py
```

This will:
1. Generate 200 synthetic signals (80% train, 20% test)
2. Train LBEADS-NET for 100 epochs
3. Evaluate on test set
4. Save model, metrics, and visualizations

### Testing the Trained Model

```bash
python demo.py
```

This will:
1. Load the most recently trained model
2. Generate new synthetic test signals
3. Run inference and compare with ground truth
4. Save detailed visualizations

### Basic Usage in Code

```python
from lbeads_net import LBEADS_NET_Fast

# Create model
model = LBEADS_NET_Fast(
    N=1024,          # Signal length
    d=1,             # Filter order
    fc=0.006,        # Cut-off frequency
    num_layers=10,   # Number of unrolled iterations
    init_lam0=0.4,   # Initial regularization parameters
    init_lam1=4.0,
    init_lam2=3.2,
    init_r=6.0       # Asymmetry ratio
)

# Forward pass: y -> (estimated_peaks, estimated_baseline)
x_peaks, f_baseline = model(y_observed)
```

### Synthetic Data Generation

```python
from train import SyntheticDataGenerator

# Create generator
generator = SyntheticDataGenerator(N=1024, seed=42)

# Generate a single signal
signal = generator.generate_signal(noise_level=0.1)
# signal.y       = observed (peaks + baseline + noise)
# signal.x_true  = ground truth peaks (target)
# signal.f_true  = ground truth baseline
# signal.noise   = noise component

# Generate a dataset
dataset = generator.generate_dataset(
    n_samples=200,
    noise_level_range=(0.05, 0.15)
)
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
