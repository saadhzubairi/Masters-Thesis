# LBEADS-NET v3: Learnable BEADS Network (Sparsity-Based Loss Function)

An unrolled neural network implementation of the BEADS (Baseline Estimation And Denoising with Sparsity) algorithm for chromatogram baseline correction, **trained on synthetic data with a sparsity-promoting loss function**.

## What's New in v3

This version builds on v2 by adding **sparsity-based loss functions** to encourage sparse peak predictions:

### Sparsity-Based Loss Function

The total loss combines multiple terms:

```
L_total = α_mse·L_reconstruction + α_l1·L_sparsity + α_tv·L_tv + α_smooth·L_smooth + α_neg·L_neg
```

**Loss Components:**

1. **Reconstruction Loss** (MSE or Huber):
   - Measures how well predicted peaks match ground truth
   - Huber loss is more robust to outliers than MSE
   - `L_mse = ||x_pred - x_true||²`

2. **L1 Sparsity Loss**:
   - Penalizes non-zero values in peak predictions
   - Encourages sparse, localized peaks
   - `L_l1 = ||x_pred||₁`

3. **Total Variation (TV) Loss**:
   - Penalizes first-order differences in peaks
   - Encourages piecewise-constant solutions
   - `L_tv = ||Dx_pred||₁` where D is the first-difference operator

4. **Baseline Smoothness Loss**:
   - Penalizes sharp changes in baseline estimate
   - Encourages smooth, slowly-varying baselines
   - `L_smooth = ||D²f_pred||₂²` where D² is the second-difference operator

5. **Non-negativity Penalty**:
   - Penalizes negative values in peak predictions
   - Chromatogram peaks should be non-negative
   - `L_neg = ||max(0, -x_pred)||²`

### Default Loss Configuration

```python
loss_config = {
    'alpha_mse': 1.0,      # Reconstruction weight
    'alpha_l1': 0.01,      # Sparsity weight  
    'alpha_tv': 0.001,     # Total variation weight
    'alpha_smooth': 0.1,   # Baseline smoothness weight
    'alpha_neg': 1.0,      # Non-negativity penalty weight
    'use_huber': True,     # Use Huber loss instead of MSE
    'huber_delta': 1.0     # Huber loss delta parameter
}
```

### Synthetic Data (from v2)

- **Sharp peaks**: Width 1-4 samples (matching real chromatogram characteristics)
- **High amplitude**: 10-100 relative to baseline
- **Smooth baseline**: Polynomial + sinusoidal drift
- **Additive noise**: Gaussian noise σ=0.5-2.0

### Train/Test Split

- 80% training, 20% testing
- Random seed for reproducibility

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
