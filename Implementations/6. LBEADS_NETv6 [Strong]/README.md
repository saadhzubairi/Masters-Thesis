# LBEADS-NET v6: Adaptive Post-Processing + Hybrid BEADS Refine

An unrolled neural network implementation of the BEADS (Baseline Estimation And Denoising with Sparsity) algorithm for chromatogram baseline correction, **with fix for baseline leakage**.

## What's New in v6

This version adds a **hybrid inference mode** designed for robustness on real chromatograms:

1. Run `LBEADS_NET` once to get an initial decomposition.
2. Apply adaptive denoising/post-lowpass cleanup on the peak estimate.
3. Run a short classical BEADS refinement warm-started from learned peaks.
4. Optionally run a full classical fallback when no-reference quality metrics are poor.

The hybrid path is implemented in `lbeads_net.py` as:
- `HybridConfig`
- `hybrid_infer_1d(...)`
- `beads_classic_with_init(...)`

`demo_chromatogram.py` now visualizes raw LBEADS output and final hybrid-selected output.



This version **fixes the baseline leakage problem** from v3 where peaks were incorrectly appearing in the baseline estimate.

### The Problem (v3)

In v3, the predicted baseline showed "bumps" at peak locations - the model was incorrectly putting part of the peak signal into the baseline. This happened because:

1. The loss function only supervised peak reconstruction (`x_pred` vs `x_true`)
2. There was no supervision on baseline estimation (`f_pred` vs `f_true`)
3. The baseline smoothness regularization alone wasn't enough to prevent leakage

### The Solution (v4)

**Add baseline supervision!** Now we train the model to match BOTH:
- `x_pred → x_true` (peaks)
- `f_pred → f_true` (baseline)

### Updated Loss Function

```
L_total = α_mse·L_peak_recon + α_baseline·L_baseline_recon + α_l1·L_sparsity + α_tv·L_tv + α_smooth·L_smooth + α_neg·L_neg
```

**NEW Loss Component:**

6. **Baseline Reconstruction Loss**:
   - Directly compares predicted baseline to ground truth baseline
   - `L_baseline = ||f_pred - f_true||²` (or Huber loss)
   - **This is the KEY fix for baseline leakage!**

### Updated Default Loss Configuration

```python
loss_config = {
    'alpha_mse': 1.0,         # Peak reconstruction weight
    'alpha_baseline': 1.0,    # **NEW** Baseline reconstruction weight
    'alpha_l1': 0.001,        # Sparsity weight  
    'alpha_tv': 0.001,        # Total variation weight
    'alpha_smooth': 0.1,      # Baseline smoothness (INCREASED)
    'alpha_neg': 0.1,         # Non-negativity penalty weight
    'use_huber': True,        # Use Huber loss instead of MSE
    'huber_delta': 1.0        # Huber loss delta parameter
}
```

### Code Changes

1. **`create_train_test_split()`** now returns baseline ground truth (`f_true`)
2. **`SparsityLoss`** class has new `alpha_baseline` parameter and `f_target` input
3. **`train_lbeads_net()`** accepts `train_f_true` for baseline supervision
4. Plotting now shows both predicted and ground truth baselines

### Synthetic Data

Same as v3:
- **Sharp peaks**: Width 1-4 samples (matching real chromatogram characteristics)
- **High amplitude**: 10-100 relative to baseline
- **Smooth baseline**: Polynomial + sinusoidal drift  
- **Additive noise**: Gaussian noise σ=0.5-2.0
- **Signal model**: `y = x_true (peaks) + f_true (baseline) + noise`

### Train/Test Split

- 80% training, 20% testing
- Now includes baseline ground truth for supervision

## Expected Results

With baseline supervision:
- Predicted baseline should closely match ground truth baseline
- No more "bumps" in baseline at peak locations
- Better peak recovery (peaks won't be partially absorbed by baseline)

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
from lbeads_net import LBEADS_NET

# Create model
model = LBEADS_NET(
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
