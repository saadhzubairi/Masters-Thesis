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

from lbeads_net import LBEADS_NET, HybridConfig, hybrid_infer_1d

# Import synthetic data generator from train.py
from train import SyntheticDataGenerator, SyntheticSignal


def _infer_model_variant(checkpoint: dict) -> str:
    """
    Infer checkpoint model variant from explicit metadata or state_dict keys.
    """
    config = checkpoint.get('model_config', {})
    declared = str(config.get('model_variant', '')).lower()
    if declared in ('lbeads_net', 'non_fast', 'classic'):
        return 'lbeads_net'
    if declared in ('lbeads_net_fast', 'fast'):
        return 'lbeads_net_fast'

    state_dict = checkpoint.get('model_state_dict', {})
    keys = list(state_dict.keys())
    if any(k.startswith('log_step_size') for k in keys):
        return 'lbeads_net_fast'
    if any(k.startswith('layers.') for k in keys):
        return 'lbeads_net'
    return 'unknown'


def load_trained_model(script_dir: str, N: int = 4096):
    """
    Load the best trained model from the script directory.
    
    Args:
        script_dir: Directory containing .pth files
        N: Signal length (must match training)
    
    Returns:
        model: Loaded LBEADS_NET model
        checkpoint: Full checkpoint dictionary
    """
    print(f"Looking for models in: {script_dir}")
    search_dirs = [
        script_dir,
        os.path.normpath(os.path.join(script_dir, "..", "5. LBEADS_NETv5 [Adaptive Post Processing]")),
        os.path.normpath(os.path.join(script_dir, "..", "5. LBEADS_NETv5")),
        os.path.normpath(os.path.join(script_dir, "..", "5. LBEADS_NETv5 VS BEADS")),
    ]

    def collect_by_pattern(pattern):
        out = []
        for d in search_dirs:
            if not os.path.isdir(d):
                continue
            escaped_d = glob.escape(d)
            out.extend(glob.glob(os.path.join(escaped_d, pattern)))
        return out

    # Find model file candidates by preference order.
    model_files = collect_by_pattern('lbeads_net_baseline_fix_*.pth')
    print(f"  Found {len(model_files)} baseline_fix models")

    if not model_files:
        model_files = collect_by_pattern('lbeads_net_sparsity_*.pth')
        print(f"  Found {len(model_files)} sparsity models")

    if not model_files:
        model_files = collect_by_pattern('lbeads_net_synthetic_*.pth')
        print(f"  Found {len(model_files)} synthetic models")
    
    if not model_files:
        print("No trained model found. Please run train.py first.")
        return None, None
    
    # Select by saved test metrics when available; fallback to newest by mtime.
    ranked_candidates = []
    skipped_fast = 0
    for candidate_path in model_files:
        try:
            candidate_ckpt = torch.load(candidate_path, map_location='cpu', weights_only=False)
        except Exception as e:
            print(f"  Skipping corrupt file {os.path.basename(candidate_path)}: {e}")
            continue
        config = candidate_ckpt.get('model_config', {})
        if config.get('N') != N:
            continue
        model_variant = _infer_model_variant(candidate_ckpt)
        if model_variant != 'lbeads_net':
            if model_variant == 'lbeads_net_fast':
                skipped_fast += 1
            continue
        tm = candidate_ckpt.get('test_metrics', {})
        corr = float(tm.get('correlation', -1.0))
        mse = float(tm.get('mse', 1e12))
        mae = float(tm.get('mae', 1e12))
        mtime = os.path.getmtime(candidate_path)
        ranked_candidates.append((-mae, -mse, corr, mtime, candidate_path, candidate_ckpt))

    if not ranked_candidates:
        if skipped_fast > 0:
            print(f"No compatible non-fast model with N={N} found (skipped {skipped_fast} fast checkpoint(s)).")
        else:
            print(f"No trained model with N={N} found.")
        print("Please retrain with train.py to generate an LBEADS_NET checkpoint.")
        return None, None

    ranked_candidates.sort(reverse=True)
    neg_mae, neg_mse, corr, _, model_path, checkpoint = ranked_candidates[0]
    print(f"  Selected by metrics: mae={-neg_mae:.6f}, mse={-neg_mse:.6f}, corr={corr:.4f}")
    
    print(f"Loading model from: {model_path}")
    
    # Create model with saved config
    config = checkpoint['model_config']
    print(f"Model config: {config}")
    
    loaded_layers = int(config.get('num_layers', 20))
    infer_layers = min(loaded_layers, 10)
    if infer_layers < loaded_layers:
        print(f"Using shallow inference depth: {infer_layers} layers (checkpoint has {loaded_layers}).")

    model = LBEADS_NET(
        N=config.get('N', N),
        d=config.get('d', 1),
        fc=config.get('fc', 0.006),
        num_layers=infer_layers,
        shared_params=config.get('shared_params', False),
        lowpass_iterations=config.get('lowpass_iterations', 1),
        solve_cg_iters=config.get('solve_cg_iters', 18),
        lowpass_cg_iters=config.get('lowpass_cg_iters', 12),
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    
    return model, checkpoint


def _save_synthetic_variant_plots(results, output_dir, N, model_status, variant_label):
    """Save per-variant synthetic plots including all candidate outputs."""
    n_samples = len(results)
    n_cols = min(3, max(1, n_samples))
    n_rows = int(np.ceil(n_samples / n_cols))

    # Plot selected output for the current variant.
    fig_selected, axes_selected = plt.subplots(n_rows, n_cols, figsize=(5.2 * n_cols, 3.8 * n_rows))
    axes_selected = np.atleast_1d(axes_selected).reshape(-1)
    for idx, result in enumerate(results):
        ax = axes_selected[idx]
        signal = result['signal']
        ax.plot(signal.y, 'gray', alpha=0.45, linewidth=0.5, label='Observed')
        ax.plot(result['x_pred'], 'b', linewidth=1.0, label='Pred Peaks')
        ax.plot(signal.x_true, 'g--', linewidth=1.0, label='True Peaks')
        ax.plot(result['f_pred'], 'r', linewidth=0.9, alpha=0.8, label='Pred Baseline')
        ax.plot(signal.f_true, 'm--', linewidth=0.9, alpha=0.8, label='True Baseline')
        ax.set_xlim([0, N])
        ax.set_title(
            f"S{idx + 1} | stage={result['selected_stage']}\n"
            f"MSE={result['mse']:.4f}, Corr={result['correlation']:.3f}"
        )
        if idx == 0:
            ax.legend(fontsize=7, loc='upper right')
    for ax in axes_selected[n_samples:]:
        ax.axis('off')
    fig_selected.suptitle(f'LBEADS-NET ({model_status}) - {variant_label} - Selected Output', fontsize=13)
    fig_selected.tight_layout()
    selected_path = os.path.join(output_dir, 'synthetic_selected_output.png')
    fig_selected.savefig(selected_path, dpi=150)
    print(f"  Saved to {selected_path}")

    # Plot all three peak candidates.
    fig_peaks, axes_peaks = plt.subplots(n_rows, n_cols, figsize=(5.2 * n_cols, 3.8 * n_rows))
    axes_peaks = np.atleast_1d(axes_peaks).reshape(-1)
    for idx, result in enumerate(results):
        ax = axes_peaks[idx]
        signal = result['signal']
        ax.plot(signal.x_true, 'k', linewidth=1.0, label='True Peaks')
        ax.plot(result['x_lbeads'], 'b', linewidth=0.9, label='x_lbeads')
        ax.plot(result['x_post'], 'c', linewidth=0.9, label='x_post')
        ax.plot(result['x_refine'], 'r', linewidth=0.9, label='x_refine')
        ax.set_xlim([0, N])
        ax.set_title(
            f"S{idx + 1} | raw_max={np.max(result['x_lbeads']):.2f}, "
            f"post_max={np.max(result['x_post']):.2f}"
        )
        if idx == 0:
            ax.legend(fontsize=7, loc='upper right')
    for ax in axes_peaks[n_samples:]:
        ax.axis('off')
    fig_peaks.suptitle(f'LBEADS-NET ({model_status}) - {variant_label} - Peak Candidates', fontsize=13)
    fig_peaks.tight_layout()
    peaks_path = os.path.join(output_dir, 'synthetic_candidates_peaks.png')
    fig_peaks.savefig(peaks_path, dpi=150)
    print(f"  Saved to {peaks_path}")

    # Plot all three baseline candidates.
    fig_base, axes_base = plt.subplots(n_rows, n_cols, figsize=(5.2 * n_cols, 3.8 * n_rows))
    axes_base = np.atleast_1d(axes_base).reshape(-1)
    for idx, result in enumerate(results):
        ax = axes_base[idx]
        signal = result['signal']
        ax.plot(signal.f_true, 'k', linewidth=1.0, label='True Baseline')
        ax.plot(result['f_lbeads'], 'b', linewidth=0.9, label='f_lbeads')
        ax.plot(result['f_post'], 'c', linewidth=0.9, label='f_post')
        ax.plot(result['f_refine'], 'r', linewidth=0.9, label='f_refine')
        ax.set_xlim([0, N])
        ax.set_title(f"S{idx + 1} | stage={result['selected_stage']}")
        if idx == 0:
            ax.legend(fontsize=7, loc='upper right')
    for ax in axes_base[n_samples:]:
        ax.axis('off')
    fig_base.suptitle(f'LBEADS-NET ({model_status}) - {variant_label} - Baseline Candidates', fontsize=13)
    fig_base.tight_layout()
    baseline_path = os.path.join(output_dir, 'synthetic_candidates_baselines.png')
    fig_base.savefig(baseline_path, dpi=150)
    print(f"  Saved to {baseline_path}")


def main():
    """Demo script showing LBEADS-NET on synthetic data with ground truth comparison."""
    print("=" * 60)
    print("LBEADS-NET Demo with Synthetic Data")
    print("=" * 60)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "demo")
    os.makedirs(output_dir, exist_ok=True)

    # Configuration
    N = 4096  # Signal length (must match training)
    num_test_samples = 6  # Number of test samples to visualize
    synthetic_noise_range = (0.01, 0.01)  # Match train.py synthetic regime
    synthetic_amplitude_range = (300.0, 1800.0)  # Target max-|y| scale (chromatogram-like)

    # Generate synthetic test data
    print("\nGenerating synthetic test data...")
    generator = SyntheticDataGenerator(N=N, seed=123)  # Different seed from training

    test_signals = []
    for _ in range(num_test_samples):
        noise_level = float(generator.rng.uniform(*synthetic_noise_range))
        signal = generator.generate_signal(noise_level=noise_level)
        target_amp = float(generator.rng.uniform(*synthetic_amplitude_range))
        signal.y = signal.y * target_amp
        signal.x_true = signal.x_true * target_amp
        signal.f_true = signal.f_true * target_amp
        signal.noise = signal.noise * target_amp
        signal.metadata['target_amplitude_scale'] = target_amp
        test_signals.append(signal)

    print(f"  Generated {num_test_samples} test signals")
    print(f"  Signal length: {N}")
    print(f"  Synthetic noise range: {synthetic_noise_range}")
    print(f"  Synthetic amplitude scale: {synthetic_amplitude_range[0]:.0f} to {synthetic_amplitude_range[1]:.0f}")

    # Try to load trained model
    model, checkpoint = load_trained_model(script_dir, N)

    if model is None:
        print("\nNo trained model found. Using untrained model for demo...")
        model = LBEADS_NET(
            N=N,
            d=1,
            fc=0.006,
            num_layers=10,
            init_lam0=0.001,
            init_lam1=0.2,
            init_lam2=0.2,
            init_r=6.0,
            shared_params=True,
            solve_cg_iters=96,
            lowpass_cg_iters=128,
        )
        model.eval()
        trained = False
    else:
        trained = True
        print("\nLoaded trained model!")
        if 'loss_config' in checkpoint:
            print("\nLoss configuration:")
            for k, v in checkpoint['loss_config'].items():
                print(f"  {k}: {v}")
        print("\nTraining metrics:")
        if 'test_metrics' in checkpoint:
            for k, v in checkpoint['test_metrics'].items():
                print(f"  {k}: {v:.4f}")

    print("\nModel parameters:")
    params = model.get_learned_params()
    for k, v in list(params.items())[:8]:
        print(f"  {k}: {v:.4f}")

    print("\n" + "=" * 60)
    print("Running Synthetic Inference Variants")
    print("=" * 60)

    variant_specs = [
        {
            "key": "raw",
            "label": "Raw net only (x_lbeads/f_lbeads)",
            "x_key": "x_lbeads",
            "f_key": "f_lbeads",
            "config": HybridConfig(
                noise_k=2.5,
                lowpass_iterations=1,
                short_refine_iterations=8,
                full_refine_iterations=24,
            ),
        },
        {
            "key": "hybrid",
            "label": "Hybrid no-threshold (noise_k=0.0)",
            "x_key": "x_hybrid",
            "f_key": "f_hybrid",
            "config": HybridConfig(
                noise_k=0.0,
                lowpass_iterations=1,
                short_refine_iterations=8,
                full_refine_iterations=24,
            ),
        },
        {
            "key": "hybrid-snr",
            "label": "Hybrid current settings (noise_k=2.5)",
            "x_key": "x_hybrid",
            "f_key": "f_hybrid",
            "config": HybridConfig(
                noise_k=2.5,
                lowpass_iterations=1,
                short_refine_iterations=8,
                full_refine_iterations=24,
            ),
        },
    ]

    model_status = "Trained" if trained else "Untrained"
    for spec in variant_specs:
        variant_output_dir = os.path.join(output_dir, spec["key"])
        os.makedirs(variant_output_dir, exist_ok=True)
        print("\n" + "-" * 60)
        print(f"Variant: {spec['label']}")
        print("-" * 60)

        results = []
        total_time = 0.0
        for i, signal in enumerate(test_signals):
            start_time = time.time()
            hybrid_result = hybrid_infer_1d(model, signal.y, config=spec["config"])
            inference_time = time.time() - start_time
            total_time += inference_time

            x_pred_np = np.asarray(hybrid_result[spec["x_key"]], dtype=np.float64)
            f_pred_np = np.asarray(hybrid_result[spec["f_key"]], dtype=np.float64)
            mse = float(np.mean((x_pred_np - signal.x_true) ** 2))
            mae = float(np.mean(np.abs(x_pred_np - signal.x_true)))
            corr = float(np.corrcoef(x_pred_np, signal.x_true)[0, 1])
            baseline_mse = float(np.mean((f_pred_np - signal.f_true) ** 2))

            results.append({
                'signal': signal,
                'x_pred': x_pred_np,
                'f_pred': f_pred_np,
                'x_lbeads': np.asarray(hybrid_result['x_lbeads'], dtype=np.float64),
                'x_post': np.asarray(hybrid_result['x_post'], dtype=np.float64),
                'x_refine': np.asarray(hybrid_result['x_refine'], dtype=np.float64),
                'f_lbeads': np.asarray(hybrid_result['f_lbeads'], dtype=np.float64),
                'f_post': np.asarray(hybrid_result['f_post'], dtype=np.float64),
                'f_refine': np.asarray(hybrid_result['f_refine'], dtype=np.float64),
                'mse': mse,
                'mae': mae,
                'correlation': corr,
                'baseline_mse': baseline_mse,
                'time': inference_time,
                'selected_stage': hybrid_result['selected_stage'],
                'noise_sigma_normalized': float(hybrid_result['noise_sigma_normalized']),
                'quality_post': hybrid_result['quality'].get('post'),
                'quality_short_refine': hybrid_result['quality'].get('short_refine'),
            })

            raw_peak_max = float(np.max(hybrid_result['x_lbeads']))
            post_peak_max = float(np.max(hybrid_result['x_post']))
            print(f"\nTest Sample {i + 1} (noise={signal.metadata['noise']['noise_level']:.3f}):")
            print(f"  Max |y|: {np.max(np.abs(signal.y)):.2f}")
            print(f"  selected_stage: {hybrid_result['selected_stage']}")
            print(f"  noise_sigma_normalized: {hybrid_result['noise_sigma_normalized']:.6f}")
            print(f"  quality['post']: {hybrid_result['quality'].get('post')}")
            print(f"  quality['short_refine']: {hybrid_result['quality'].get('short_refine')}")
            if raw_peak_max > 1e-8 and post_peak_max < 0.05 * raw_peak_max:
                print(
                    "  WARNING: x_post is near zero relative to x_lbeads "
                    f"(post/raw={post_peak_max / raw_peak_max:.4f})."
                )
            print(f"  Peak MSE: {mse:.6f}")
            print(f"  Peak Correlation: {corr:.4f}")
            print(f"  Baseline MSE: {baseline_mse:.6f}")
            print(f"  Inference time: {inference_time * 1000:.2f} ms")

        avg_mse = float(np.mean([r['mse'] for r in results]))
        avg_corr = float(np.mean([r['correlation'] for r in results]))
        avg_baseline_mse = float(np.mean([r['baseline_mse'] for r in results]))
        avg_infer_ms = float((total_time / max(len(results), 1)) * 1000.0)

        print("\nAverage Metrics")
        print(f"  Peak MSE: {avg_mse:.6f}")
        print(f"  Peak Correlation: {avg_corr:.4f}")
        print(f"  Baseline MSE: {avg_baseline_mse:.6f}")
        print(f"  Average inference time: {avg_infer_ms:.2f} ms")

        print("Generating visualizations...")
        _save_synthetic_variant_plots(results, variant_output_dir, N, model_status, spec["label"])

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
    output_dir = os.path.join(script_dir, "demo")
    os.makedirs(output_dir, exist_ok=True)
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
    
    def infer_signal(sig):
        """Hybrid inference (LBEADS + BEADS refine) for one legacy signal."""
        hybrid_cfg = HybridConfig(
            noise_k=2.5,
            lowpass_iterations=3,
            short_refine_iterations=8,
            full_refine_iterations=24,
        )
        result = hybrid_infer_1d(model, sig, config=hybrid_cfg)
        return result["x_hybrid"], result["f_hybrid"], result
    
    # --- Noise levels to test ---
    noise_levels = [0.0, 0.3, 0.5]
    
    # --- Pick a subset of chromatogram columns to display ---
    # Use columns 0-5 (6 chromatograms) to fill a 2x3 grid per noise level
    cols_to_show = list(range(min(6, n_chromatograms)))
    
    # --- Run inference on the classic BEADS test case first (col 2, noise*0.5) ---
    print("\n--- Classic BEADS test case (column 3, noise x0.5) ---")
    y_classic = X[:, 2] + noise * 0.5
    x_pred_np, f_pred_np, classic_result = infer_signal(y_classic)
    print(
        "  Hybrid stage:"
        f" {classic_result['selected_stage']}"
        f" | score={classic_result['quality_selected']['score']:.4f}"
    )
    
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
    legacy_classic_path = os.path.join(output_dir, 'demo_legacy_classic.png')
    plt.savefig(legacy_classic_path, dpi=150)
    print(f"  Saved to {legacy_classic_path}")
    
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
            xp_np, fp_np, _ = infer_signal(y_obs)
            
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
    legacy_grid_path = os.path.join(output_dir, 'demo_legacy_grid.png')
    plt.savefig(legacy_grid_path, dpi=150)
    print(f"  Saved to {legacy_grid_path}")
    
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
