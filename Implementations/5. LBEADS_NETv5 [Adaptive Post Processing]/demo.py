"""
Demo script for LBEADS-NET with Synthetic Data

This script demonstrates the LBEADS-NET model on synthetic chromatogram data,
comparing its performance in separating peaks from baseline drift.

Signal model:
    y = x_true (peaks) + f_true (baseline) + noise

- the high-pass and low-pass filters became small learnable Conv1D kernels, 
- the asymmetric regularization became a learnable shrinkage operator, and 
- the BEADS hyperparameters (λ₀, λ₁, λ₂, r) became trainable scalars that can vary across layers. 

By stacking K of these modified BEADS steps, the resulting model acts as a deep, interpretable network whose forward pass mimics the classical iterations but whose parameters are optimized end-to-end from data.

"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time
import glob
import scipy.io as sio

from lbeads_net import LBEADS_NET, LBEADS_NET_Fast

# Import synthetic data generator from train.py
from train import SyntheticDataGenerator, SyntheticSignal


def load_trained_model(script_dir: str, N: int = 4096):
    """
    Load the best trained model from the script directory.
    
    Args:
        script_dir: Directory containing .pth files
        N: Signal length (must match training)
    
    Returns:
        model: Loaded LBEADS_NET_Fast model
        checkpoint: Full checkpoint dictionary
    """
    print(f"Looking for models in: {script_dir}")
    
    # Escape square brackets for glob (they're special characters)
    escaped_dir = script_dir.replace('[', '[[]').replace(']', '[]]')
    
    # Find most recent model file (try baseline_fix models first, then sparsity, then synthetic)
    model_files = glob.glob(os.path.join(escaped_dir, 'lbeads_net_baseline_fix_*.pth'))
    print(f"  Found {len(model_files)} baseline_fix models")
    
    if not model_files:
        model_files = glob.glob(os.path.join(escaped_dir, 'lbeads_net_sparsity_*.pth'))
        print(f"  Found {len(model_files)} sparsity models")
    
    if not model_files:
        model_files = glob.glob(os.path.join(escaped_dir, 'lbeads_net_synthetic_*.pth'))
        print(f"  Found {len(model_files)} synthetic models")
    
    if not model_files:
        # Fallback: use os.listdir instead of glob to avoid pattern issues
        print("  Trying fallback with os.listdir...")
        try:
            all_files = os.listdir(script_dir)
            model_files = [os.path.join(script_dir, f) for f in all_files 
                          if f.startswith('lbeads_net_baseline_fix_') and f.endswith('.pth')]
            if not model_files:
                model_files = [os.path.join(script_dir, f) for f in all_files 
                              if f.startswith('lbeads_net_sparsity_') and f.endswith('.pth')]
            if not model_files:
                model_files = [os.path.join(script_dir, f) for f in all_files 
                              if f.startswith('lbeads_net_synthetic_') and f.endswith('.pth')]
            print(f"  Found {len(model_files)} models via listdir")
        except Exception as e:
            print(f"  Fallback failed: {e}")
    
    if not model_files:
        print("No trained model found. Please run train.py first.")
        return None, None
    
    # Select by saved test metrics when available; fallback to newest by mtime.
    ranked_candidates = []
    for candidate_path in model_files:
        try:
            candidate_ckpt = torch.load(candidate_path, map_location='cpu', weights_only=False)
        except Exception as e:
            print(f"  Skipping corrupt file {os.path.basename(candidate_path)}: {e}")
            continue
        config = candidate_ckpt.get('model_config', {})
        if config.get('N') != N:
            continue
        tm = candidate_ckpt.get('test_metrics', {})
        corr = float(tm.get('correlation', -1.0))
        mse = float(tm.get('mse', 1e12))
        mae = float(tm.get('mae', 1e12))
        mtime = os.path.getmtime(candidate_path)
        ranked_candidates.append((-mae, -mse, corr, mtime, candidate_path, candidate_ckpt))

    if not ranked_candidates:
        print(f"No trained model with N={N} found. Please retrain with train.py.")
        return None, None

    ranked_candidates.sort(reverse=True)
    neg_mae, neg_mse, corr, _, model_path, checkpoint = ranked_candidates[0]
    print(f"  Selected by metrics: mae={-neg_mae:.6f}, mse={-neg_mse:.6f}, corr={corr:.4f}")
    
    print(f"Loading model from: {model_path}")
    
    # Create model with saved config
    config = checkpoint['model_config']
    print(f"Model config: {config}")
    
    model = LBEADS_NET_Fast(
        N=config['N'],
        d=config['d'],
        fc=config['fc'],
        num_layers=config['num_layers'],
        init_lam0=0.4,
        init_lam1=4.0,
        init_lam2=3.2,
        init_r=6.0,
        init_step_size=0.001,
        lowpass_iterations=config.get('lowpass_iterations', 1),
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, checkpoint


def main():
    """Demo script showing LBEADS-NET on synthetic data with ground truth comparison."""
    print("=" * 60)
    print("LBEADS-NET Demo with Synthetic Data")
    print("=" * 60)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Configuration
    N = 4096  # Signal length (must match training)
    num_test_samples = 6  # Number of test samples to visualize
    
    # Generate synthetic test data
    print("\nGenerating synthetic test data...")
    generator = SyntheticDataGenerator(N=N, seed=123)  # Different seed from training
    
    test_signals = []
    for i in range(num_test_samples):
        noise_level = 0.15 + (i * 0.05)  # Slightly stronger high-frequency noise
        signal = generator.generate_signal(noise_level=noise_level)
        test_signals.append(signal)
    
    print(f"  Generated {num_test_samples} test signals")
    print(f"  Signal length: {N}")
    
    # Try to load trained model
    model, checkpoint = load_trained_model(script_dir, N)
    
    if model is None:
        print("\nNo trained model found. Using untrained model for demo...")
        model = LBEADS_NET_Fast(
            N=N,
            d=1,
            fc=0.01,
            num_layers=15,
            init_lam0=0.5,
            init_lam1=1.0,
            init_lam2=1.0,
            init_r=6.0,
            init_step_size=0.01
        )
        model.eval()
        trained = False
    else:
        trained = True
        print("\nLoaded trained model!")
        
        # Show loss config if available (sparsity-based model)
        if 'loss_config' in checkpoint:
            print("\nLoss configuration:")
            for k, v in checkpoint['loss_config'].items():
                print(f"  {k}: {v}")
        
        print("\nTraining metrics:")
        if 'test_metrics' in checkpoint:
            for k, v in checkpoint['test_metrics'].items():
                print(f"  {k}: {v:.4f}")
    
    # Print model parameters
    print("\nModel parameters:")
    params = model.get_learned_params()
    for k, v in list(params.items())[:8]:
        print(f"  {k}: {v:.4f}")
    
    # Run inference and compute metrics
    print("\n" + "=" * 60)
    print("Running Inference on Test Data")
    print("=" * 60)
    
    results = []
    total_time = 0
    
    for i, signal in enumerate(test_signals):
        # Match training-time per-sample normalization.
        y_scale = max(np.max(np.abs(signal.y)), 1e-8)
        y_normed = signal.y / y_scale
        y_tensor = torch.tensor(y_normed, dtype=torch.float64).unsqueeze(0)
        
        start_time = time.time()
        with torch.no_grad():
            x_pred, f_pred = model(y_tensor)
        inference_time = time.time() - start_time
        total_time += inference_time
        
        x_pred_np = x_pred[0].numpy() * y_scale
        f_pred_np = f_pred[0].numpy() * y_scale
        
        # Compute metrics against ground truth
        mse = np.mean((x_pred_np - signal.x_true) ** 2)
        mae = np.mean(np.abs(x_pred_np - signal.x_true))
        corr = np.corrcoef(x_pred_np, signal.x_true)[0, 1]
        
        # Baseline estimation error
        baseline_mse = np.mean((f_pred_np - signal.f_true) ** 2)
        
        results.append({
            'signal': signal,
            'x_pred': x_pred_np,
            'f_pred': f_pred_np,
            'mse': mse,
            'mae': mae,
            'correlation': corr,
            'baseline_mse': baseline_mse,
            'time': inference_time
        })
        
        print(f"\nTest Sample {i+1} (noise={signal.metadata['noise']['noise_level']:.3f}):")
        print(f"  Peak MSE: {mse:.6f}")
        print(f"  Peak Correlation: {corr:.4f}")
        print(f"  Baseline MSE: {baseline_mse:.6f}")
        print(f"  Inference time: {inference_time*1000:.2f} ms")
    
    # Average metrics
    avg_mse = np.mean([r['mse'] for r in results])
    avg_corr = np.mean([r['correlation'] for r in results])
    avg_baseline_mse = np.mean([r['baseline_mse'] for r in results])
    
    print("\n" + "=" * 40)
    print("Average Metrics")
    print("=" * 40)
    print(f"  Peak MSE: {avg_mse:.6f}")
    print(f"  Peak Correlation: {avg_corr:.4f}")
    print(f"  Baseline MSE: {avg_baseline_mse:.6f}")
    print(f"  Average inference time: {total_time/len(results)*1000:.2f} ms")
    
    # Visualization
    print("\nGenerating visualizations...")
    
    # Figure 1: Test samples with predictions
    fig1, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    for idx, (ax, result) in enumerate(zip(axes.flat, results)):
        signal = result['signal']
        
        ax.plot(signal.y, 'gray', alpha=0.5, linewidth=0.5, label='Observed (y)')
        ax.plot(result['x_pred'], 'b', linewidth=1, label='Predicted Peaks')
        ax.plot(signal.x_true, 'g--', linewidth=1, label='True Peaks')
        ax.plot(result['f_pred'], 'r', linewidth=0.8, alpha=0.7, label='Predicted Baseline')
        ax.plot(signal.f_true, 'm--', linewidth=0.8, alpha=0.7, label='True Baseline')
        
        ax.set_title(f'Sample {idx+1} | MSE={result["mse"]:.4f} | Corr={result["correlation"]:.3f}')
        ax.set_xlim([0, N])
        ax.legend(fontsize=7)
    
    model_status = "Trained" if trained else "Untrained"
    plt.suptitle(f'LBEADS-NET ({model_status}) - Synthetic Data Test Results', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, 'demo_synthetic_results.png'), dpi=150)
    print(f"  Saved to {os.path.join(script_dir, 'demo_synthetic_results.png')}")
    
    # Figure 2: Detailed view of one sample - IMPROVED VISUALIZATION
    fig2, axes2 = plt.subplots(4, 1, figsize=(14, 12))
    
    result = results[0]  # Use first sample
    signal = result['signal']
    
    # Panel 1: All signal components separately
    axes2[0].plot(signal.x_true, 'g', linewidth=1, alpha=0.8, label='Peaks (x_true)')
    axes2[0].plot(signal.f_true, 'm', linewidth=1, alpha=0.8, label='Baseline (f_true)')
    # Noise is much smaller than peaks, so scale it for visibility.
    noise_vis_scale = 20.0
    axes2[0].plot(signal.noise * noise_vis_scale, 'gray', linewidth=0.8, alpha=0.9,
                  label=f'Noise x{noise_vis_scale:.0f} (for visibility)')
    axes2[0].set_title('Signal Components: peaks + baseline + noise')
    axes2[0].set_xlim([0, N])
    axes2[0].set_ylabel('Amplitude')
    axes2[0].legend(loc='upper right')
    axes2[0].grid(True, alpha=0.3)
    
    # Panel 2: Observed signal only
    axes2[1].plot(signal.y, 'b', linewidth=0.5, alpha=0.7, label='Observed (y = x + f + noise)')
    axes2[1].set_title('Observed Signal')
    axes2[1].set_xlim([0, N])
    axes2[1].set_ylabel('Amplitude')
    axes2[1].legend(loc='upper right')
    axes2[1].grid(True, alpha=0.3)
    
    # Panel 3: Peak recovery
    axes2[2].plot(signal.x_true, 'g', linewidth=1.5, label='Ground Truth Peaks')
    axes2[2].plot(result['x_pred'], 'b--', linewidth=1.5, label='Predicted Peaks')
    axes2[2].set_title(f'Peak Recovery (MSE={result["mse"]:.6f}, Corr={result["correlation"]:.4f})')
    axes2[2].set_xlim([0, N])
    axes2[2].set_ylabel('Amplitude')
    axes2[2].legend(loc='upper right')
    axes2[2].grid(True, alpha=0.3)
    
    # Panel 4: Baseline estimation - ZOOMED to see detail
    axes2[3].plot(signal.f_true, 'm', linewidth=2, label='Ground Truth Baseline')
    axes2[3].plot(result['f_pred'], 'r--', linewidth=2, label='Predicted Baseline')
    # Calculate baseline error
    baseline_error = result['f_pred'] - signal.f_true
    axes2[3].fill_between(range(N), signal.f_true, result['f_pred'], 
                          alpha=0.3, color='orange', label='Error')
    axes2[3].set_title(f'Baseline Estimation (MSE={result["baseline_mse"]:.6f})')
    axes2[3].set_xlim([0, N])
    axes2[3].set_xlabel('Sample Index')
    axes2[3].set_ylabel('Amplitude')
    axes2[3].legend(loc='upper right')
    axes2[3].grid(True, alpha=0.3)
    
    plt.suptitle('LBEADS-NET Detailed Analysis (v4 - Baseline Supervision)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, 'demo_detailed_analysis.png'), dpi=150)
    print(f"  Saved to {os.path.join(script_dir, 'demo_detailed_analysis.png')}")
    
    print("\nSynthetic demo complete!")
    return model, trained


# =============================================================================
# Legacy Signal Data Demo
# =============================================================================

def demo_legacy_data(model=None, trained: bool = False):
    """
    Run inference on legacy chromatogram data from the BEADS paper.
    
    Data location: Implementations/0. BEADS/data/
    - chromatograms.mat: 8 chromatogram signals of length 4000 (key 'X', shape 4000x8)
    - noise.mat: noise vector of length 4000 (key 'noise', shape 4000x1)
    
    Since the model is trained on N=4096, we pad the 4000-length signals with
    reflection padding and crop back after inference.
    """
    print("\n" + "=" * 60)
    print("LBEADS-NET Demo on Legacy BEADS Signal Data")
    print("=" * 60)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    N_model = 4096  # Model's expected signal length
    
    # --- Load legacy data ---
    data_dir = os.path.join(script_dir, '..', '0. BEADS', 'data')
    data_dir = os.path.normpath(data_dir)
    
    if not os.path.isdir(data_dir):
        print(f"Legacy data directory not found: {data_dir}")
        print("Skipping legacy demo.")
        return
    
    print(f"Loading data from: {data_dir}")
    chromatograms_data = sio.loadmat(os.path.join(data_dir, 'chromatograms.mat'))
    noise_data = sio.loadmat(os.path.join(data_dir, 'noise.mat'))
    
    X = chromatograms_data['X']       # (4000, 8)
    noise = noise_data['noise'].flatten()  # (4000,)
    N_legacy = X.shape[0]
    n_chromatograms = X.shape[1]
    
    print(f"  Chromatograms: {X.shape} ({n_chromatograms} signals of length {N_legacy})")
    print(f"  Noise: {noise.shape}")
    
    # --- Load or use provided model ---
    if model is None:
        model, checkpoint = load_trained_model(script_dir, N_model)
        if model is None:
            print("No trained model available. Run train.py first.")
            return
        trained = True
    
    model.eval()
    
    # --- Pad helper functions ---
    pad_amount = N_model - N_legacy  # 96 samples to pad
    
    def pad_signal(sig):
        """Reflect-pad a (N_legacy,) signal to (N_model,)."""
        return np.pad(sig, (0, pad_amount), mode='reflect')
    
    def crop_signal(sig):
        """Crop back from (N_model,) to (N_legacy,)."""
        return sig[:N_legacy]

    def infer_signal(sig):
        """
        Normalize -> infer -> denormalize for one legacy signal.

        The model is trained on per-sample unit-scale inputs, so inference
        must apply the same scaling to avoid distribution mismatch.
        """
        y_padded = pad_signal(sig)
        y_scale = np.max(np.abs(y_padded))
        y_scale = max(y_scale, 1e-8)
        y_normed = y_padded / y_scale
        y_tensor = torch.tensor(y_normed, dtype=torch.float64).unsqueeze(0)
        with torch.no_grad():
            x_pred, f_pred = model(y_tensor)
        x_pred_np = crop_signal(x_pred[0].numpy()) * y_scale
        f_pred_np = crop_signal(f_pred[0].numpy()) * y_scale
        return x_pred_np, f_pred_np
    
    # --- Noise levels to test ---
    noise_levels = [0.0, 0.3, 0.5]
    
    # --- Pick a subset of chromatogram columns to display ---
    # Use columns 0-5 (6 chromatograms) to fill a 2x3 grid per noise level
    cols_to_show = list(range(min(6, n_chromatograms)))
    
    # --- Run inference on the classic BEADS test case first (col 2, noise*0.5) ---
    print("\n--- Classic BEADS test case (column 3, noise x0.5) ---")
    y_classic = X[:, 2] + noise * 0.5
    x_pred_np, f_pred_np = infer_signal(y_classic)
    
    # Classic figure (matches BEADS paper figure)
    fig_classic, axes_c = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    
    axes_c[0].plot(y_classic, 'b', linewidth=0.5)
    axes_c[0].set_title('Data (Chromatogram 3 + noise x0.5)')
    axes_c[0].set_ylabel('Amplitude')
    axes_c[0].set_xlim([0, N_legacy])
    axes_c[0].grid(True, alpha=0.3)
    
    axes_c[1].plot(y_classic, 'gray', alpha=0.5, linewidth=0.5, label='Observed')
    axes_c[1].plot(f_pred_np, 'r', linewidth=1.5, label='Estimated Baseline')
    axes_c[1].set_title('Baseline Estimation')
    axes_c[1].set_ylabel('Amplitude')
    axes_c[1].legend(loc='upper right')
    axes_c[1].grid(True, alpha=0.3)
    
    axes_c[2].plot(x_pred_np, 'b', linewidth=0.8)
    axes_c[2].set_title('Baseline-Corrected Signal (Estimated Peaks)')
    axes_c[2].set_ylabel('Amplitude')
    axes_c[2].grid(True, alpha=0.3)
    
    residual_classic = y_classic - x_pred_np - f_pred_np
    axes_c[3].plot(residual_classic, 'gray', linewidth=0.5)
    axes_c[3].set_title('Residual (y - x - f)')
    axes_c[3].set_xlabel('Sample Index')
    axes_c[3].set_ylabel('Amplitude')
    axes_c[3].grid(True, alpha=0.3)
    
    model_tag = "Trained" if trained else "Untrained"
    fig_classic.suptitle(f'LBEADS-NET ({model_tag}) — Classic BEADS Test Case', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, 'demo_legacy_classic.png'), dpi=150)
    print(f"  Saved to demo_legacy_classic.png")
    
    # --- Grid figure: all chromatograms at multiple noise levels ---
    print(f"\n--- All chromatograms at noise levels {noise_levels} ---")
    
    fig_grid, axes_g = plt.subplots(
        len(noise_levels), len(cols_to_show),
        figsize=(3.5 * len(cols_to_show), 3.5 * len(noise_levels)),
        sharex=True,
    )
    if len(noise_levels) == 1:
        axes_g = axes_g[np.newaxis, :]
    if len(cols_to_show) == 1:
        axes_g = axes_g[:, np.newaxis]
    
    for row, nl in enumerate(noise_levels):
        for col_idx, chromatogram_col in enumerate(cols_to_show):
            ax = axes_g[row, col_idx]
            
            # Build observed signal
            clean = X[:, chromatogram_col]
            y_obs = clean + noise * nl if nl > 0 else clean.copy()
            
            # Pad → infer → crop
            xp_np, fp_np = infer_signal(y_obs)
            
            # Plot
            ax.plot(y_obs, 'gray', alpha=0.5, linewidth=0.4, label='Observed')
            ax.plot(xp_np, 'b', linewidth=0.8, label='Peaks')
            ax.plot(fp_np, 'r', linewidth=1.0, alpha=0.8, label='Baseline')
            ax.set_xlim([0, N_legacy])
            
            if row == 0:
                ax.set_title(f'Chrom {chromatogram_col + 1}', fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(f'noise x{nl}', fontsize=10)
            if row == 0 and col_idx == len(cols_to_show) - 1:
                ax.legend(fontsize=7, loc='upper right')
    
    fig_grid.suptitle(
        f'LBEADS-NET ({model_tag}) — Legacy Chromatogram Data',
        fontsize=14,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, 'demo_legacy_grid.png'), dpi=150)
    print(f"  Saved to demo_legacy_grid.png")
    
    print("\nLegacy data demo complete!")


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    # Part 1 & 2: Synthetic data demo
    model, trained = main()
    
    # Part 3: Legacy BEADS signal data demo
    demo_legacy_data(model=model, trained=trained)
    
    # Show all plots at once
    plt.show()
