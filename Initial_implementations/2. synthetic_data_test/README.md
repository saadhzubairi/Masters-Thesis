# Synthetic Data Test: BEADS vs LBEADS-NET

This folder contains the experimental framework for comparing classical BEADS with LBEADS-NET on synthetic signals with known ground truth.

## Purpose

This experiment produces **Table 1** and **Figure 1** for the thesis, demonstrating that LBEADS-NET (the unrolled neural network version) can match or exceed the performance of classical BEADS on baseline estimation and sparse signal recovery.

## Files

| File | Description |
|------|-------------|
| `synthetic_data_generator.py` | Generates synthetic chromatogram-like signals with polynomial+sinusoidal baselines, Gaussian peaks, and Gaussian/Laplacian noise |
| `metrics.py` | Evaluation metrics: MSE (signal & baseline), SNR, ΔSNR |
| `visualization.py` | Plotting functions for Figure 1 and metric summaries |
| `run_experiments.py` | Main experiment script that orchestrates everything |

## Quick Start

```bash
# Run the full experiment
python run_experiments.py

# Run with custom parameters
python run_experiments.py --n-samples 50 --seed 123

# Run without generating plots
python run_experiments.py --no-plots

# Run with a trained LBEADS-NET model
python run_experiments.py --model-path ../LBEADS_NETv1/lbeads_net_trained.pth
```

## Synthetic Data Generation

### Signal Model

The observed signal is:
$$y = x_{\text{true}} + f_{\text{true}} + \text{noise}$$

Where:

1. **Baseline $f_{\text{true}}$**: Polynomial + sinusoidal
   - Polynomial: degree 2 or 3, coefficients ~ Uniform([-0.5, 0.5])
   - Sinusoid: frequency ∈ [0.5, 2] cycles, amplitude ∈ [0.1, 0.3]

2. **Sparse Signal $x_{\text{true}}$**: Sum of Gaussian peaks
   - Number of peaks: 3-6
   - Peak centers: [0.1N, 0.9N]
   - Peak widths (σ): 5-20 samples
   - Peak amplitudes: 0.5-1.5

3. **Noise**: Gaussian or Laplacian
   - Gaussian: $\mathcal{N}(0, \sigma^2)$, σ ∈ [0.05, 0.15]
   - Laplacian: $\text{Laplace}(0, b)$, b ∈ [0.05, 0.15]

### Dataset Structure

The experiment generates 30 signals by default:
- **10 Easy**: Low Gaussian noise (σ ∈ [0.03, 0.07])
- **10 Medium**: Moderate Gaussian noise (σ ∈ [0.08, 0.12])
- **10 Hard**: High noise, mixed Gaussian/Laplacian (σ ∈ [0.12, 0.18])

## Evaluation Metrics

### MSE (Mean Squared Error)

Signal MSE:
$$\text{MSE}_x = \frac{1}{N} \|x_{\text{true}} - x_{\text{est}}\|^2$$

Baseline MSE:
$$\text{MSE}_f = \frac{1}{N} \|f_{\text{true}} - f_{\text{est}}\|^2$$

### SNR (Signal-to-Noise Ratio)

Input SNR:
$$\text{SNR}_{\text{in}} = 10 \log_{10} \frac{\|x_{\text{true}}\|^2}{\|y - (x_{\text{true}} + f_{\text{true}})\|^2}$$

Output SNR:
$$\text{SNR}_{\text{out}} = 10 \log_{10} \frac{\|x_{\text{true}}\|^2}{\|x_{\text{true}} - x_{\text{est}}\|^2}$$

SNR Improvement:
$$\Delta\text{SNR} = \text{SNR}_{\text{out}} - \text{SNR}_{\text{in}}$$

## Expected Output

### Table 1

```
===========================================================================
TABLE 1: Comparison on Synthetic Data
===========================================================================
Method          | MSE (signal) ↓       | MSE (baseline) ↓     | ΔSNR (dB) ↑
---------------------------------------------------------------------------
BEADS           | 0.XXXX ± 0.XXXX      | 0.XXXX ± 0.XXXX      | XX.XX ± X.XX
LBEADS-NET      | 0.XXXX ± 0.XXXX      | 0.XXXX ± 0.XXXX      | XX.XX ± X.XX
===========================================================================
```

### Figure 1

A multi-panel figure showing:
- Column 1: Baseline estimation (y, f_true, f_beads, f_lbeads)
- Column 2: Signal recovery (x_true, x_beads, x_lbeads)
- Column 3: Estimation error

Rows show different difficulty levels (easy, medium, hard, Laplacian).

## Output Files

After running, the `results/` folder will contain:

```
results/
├── figure1_comparison.png      # Main thesis figure
├── metrics_summary.png         # Bar chart comparison
├── boxplot_mse_signal.png      # Distribution of MSE
├── boxplot_mse_baseline.png    # Distribution of baseline MSE
├── boxplot_delta_snr.png       # Distribution of ΔSNR
├── test_dataset.npz            # Saved synthetic dataset
├── raw_results.npz             # Raw metric values
├── results_summary.txt         # Text summary
└── table1.tex                  # LaTeX code for Table 1
```

## BEADS Parameters

Fixed parameters used for all signals (not tuned per-signal):

| Parameter | Value | Description |
|-----------|-------|-------------|
| d | 1 | Filter order |
| fc | 0.006 | Cutoff frequency |
| r | 6.0 | Asymmetry ratio |
| λ₀ | 0.5 | Asymmetric penalty |
| λ₁ | 4.0 | First derivative penalty |
| λ₂ | 4.0 | Second derivative penalty |
| Nit | 30 | Number of iterations |

## Notes

1. **Reproducibility**: Use the same random seed for comparable results.

2. **LBEADS-NET Mode**: By default, LBEADS-NET uses the same initialization as BEADS (shared parameters). This tests whether the unrolled formulation alone provides any benefit.

3. **Trained Models**: For best results, use a trained LBEADS-NET model via `--model-path`. The training learns optimal per-layer parameters.

4. **Signal Length**: Fixed at N=1024 for consistency. All methods use the same length.

## Citation

If using this code, please cite the original BEADS paper:

> Ning, X., Selesnick, I. W., & Duval, L. (2014). Chromatogram baseline estimation and denoising using sparsity (BEADS). *Chemometrics and Intelligent Laboratory Systems*, 139, 156-167.
