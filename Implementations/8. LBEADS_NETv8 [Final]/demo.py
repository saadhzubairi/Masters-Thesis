"""
Demo script for LBEADS-NET v8 with Synthetic Data

Generates synthetic chromatogram signals, loads a trained v8 model,
runs inference, and plots: input signal, ground truth peaks/baseline,
model output peaks/baseline.

Usage:
    python demo.py
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import glob

from lbeads_net import LBEADS_NET
from train import SyntheticDataGenerator, SyntheticSignal


def load_trained_model(script_dir: str, N: int = 4096):
    """
    Load the best trained v8 model from the script directory.

    Selects by saved test metrics (lowest MAE, then lowest MSE, then highest
    correlation, then most recent).

    Args:
        script_dir: Directory containing .pth files
        N: Signal length (must match training)

    Returns:
        model: Loaded LBEADS_NET model (or None)
        checkpoint: Full checkpoint dictionary (or None)
    """
    print(f"Looking for models in: {script_dir}")

    # Find model files via os.listdir to avoid glob bracket issues.
    try:
        all_files = os.listdir(script_dir)
    except Exception as e:
        print(f"  Cannot list directory: {e}")
        return None, None

    model_files = [
        os.path.join(script_dir, f)
        for f in all_files
        if f.endswith('.pth') and f.startswith('lbeads_net_')
    ]
    print(f"  Found {len(model_files)} .pth files")

    if not model_files:
        print("No trained model found. Please run train.py first.")
        return None, None

    # Rank candidates by metrics.
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

    config = checkpoint['model_config']
    print(f"Model config: {config}")

    model = LBEADS_NET(
        N=config['N'],
        d=config['d'],
        fc=config['fc'],
        num_layers=config['num_layers'],
        lowpass_iterations=config.get('lowpass_iterations', 3),
        solve_cg_iters=config.get('solve_cg_iters', 12),
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, checkpoint


def main():
    """Demo script showing LBEADS-NET v8 on synthetic data with ground truth comparison."""
    print("=" * 60)
    print("LBEADS-NET v8 Demo with Synthetic Data")
    print("=" * 60)

    script_dir = os.path.dirname(os.path.abspath(__file__))

    N = 4096
    num_test_samples = 6

    # Generate synthetic test data (different seed from training).
    print("\nGenerating synthetic test data...")
    generator = SyntheticDataGenerator(N=N, seed=123)

    test_signals = []
    for i in range(num_test_samples):
        noise_level = 0.15 + (i * 0.05)
        signal = generator.generate_signal(noise_level=noise_level)
        test_signals.append(signal)

    print(f"  Generated {num_test_samples} test signals")
    print(f"  Signal length: {N}")

    # Load trained model.
    model, checkpoint = load_trained_model(script_dir, N)

    if model is None:
        print("\nNo trained model found. Using untrained model for demo...")
        model = LBEADS_NET(
            N=N, d=1, fc=0.002,
            num_layers=6,
            init_lam0=0.01, init_lam1=0.5, init_lam2=0.5,
            init_r=6.0, init_step_size=0.05,
            lowpass_iterations=3,
            solve_cg_iters=12,
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
                if isinstance(v, float):
                    print(f"  {k}: {v:.4f}")

    print("\nModel parameters:")
    params = model.get_learned_params()
    for k, v in list(params.items())[:8]:
        print(f"  {k}: {v:.4f}")

    # Run inference
    print("\n" + "=" * 60)
    print("Running Inference on Test Data")
    print("=" * 60)

    results = []
    total_time = 0

    for i, signal in enumerate(test_signals):
        y_scale = max(np.max(np.abs(signal.y)), 1e-8)
        y_normed = signal.y / y_scale
        y_tensor = torch.tensor(y_normed, dtype=torch.float64).unsqueeze(0)

        start_time = time.time()
        with torch.no_grad():
            x_pred, f_pred = model(y_tensor)
            x_pred = F.softplus(x_pred, beta=20.0)
        inference_time = time.time() - start_time
        total_time += inference_time

        x_pred_np = x_pred[0].numpy() * y_scale
        f_pred_np = f_pred[0].numpy() * y_scale

        mse = np.mean((x_pred_np - signal.x_true) ** 2)
        mae = np.mean(np.abs(x_pred_np - signal.x_true))
        corr = np.corrcoef(x_pred_np, signal.x_true)[0, 1]
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

    # Figure 1: Grid of test samples
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
    plt.suptitle(f'LBEADS-NET v8 ({model_status}) - Synthetic Data Test Results', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, 'demo_synthetic_results.png'), dpi=150)
    print(f"  Saved to {os.path.join(script_dir, 'demo_synthetic_results.png')}")

    # Figure 2: Detailed view of one sample
    fig2, axes2 = plt.subplots(4, 1, figsize=(14, 12))

    result = results[0]
    signal = result['signal']

    axes2[0].plot(signal.x_true, 'g', linewidth=1, alpha=0.8, label='Peaks (x_true)')
    axes2[0].plot(signal.f_true, 'm', linewidth=1, alpha=0.8, label='Baseline (f_true)')
    noise_vis_scale = 20.0
    axes2[0].plot(signal.noise * noise_vis_scale, 'gray', linewidth=0.8, alpha=0.9,
                  label=f'Noise x{noise_vis_scale:.0f}')
    axes2[0].set_title('Signal Components')
    axes2[0].set_xlim([0, N])
    axes2[0].set_ylabel('Amplitude')
    axes2[0].legend(loc='upper right')
    axes2[0].grid(True, alpha=0.3)

    axes2[1].plot(signal.y, 'b', linewidth=0.5, alpha=0.7, label='Observed (y)')
    axes2[1].set_title('Observed Signal')
    axes2[1].set_xlim([0, N])
    axes2[1].set_ylabel('Amplitude')
    axes2[1].legend(loc='upper right')
    axes2[1].grid(True, alpha=0.3)

    axes2[2].plot(signal.x_true, 'g', linewidth=1.5, label='Ground Truth Peaks')
    axes2[2].plot(result['x_pred'], 'b--', linewidth=1.5, label='Predicted Peaks')
    axes2[2].set_title(f'Peak Recovery (MSE={result["mse"]:.6f}, Corr={result["correlation"]:.4f})')
    axes2[2].set_xlim([0, N])
    axes2[2].set_ylabel('Amplitude')
    axes2[2].legend(loc='upper right')
    axes2[2].grid(True, alpha=0.3)

    axes2[3].plot(signal.f_true, 'm', linewidth=2, label='Ground Truth Baseline')
    axes2[3].plot(result['f_pred'], 'r--', linewidth=2, label='Predicted Baseline')
    axes2[3].fill_between(range(N), signal.f_true, result['f_pred'],
                          alpha=0.3, color='orange', label='Error')
    axes2[3].set_title(f'Baseline Estimation (MSE={result["baseline_mse"]:.6f})')
    axes2[3].set_xlim([0, N])
    axes2[3].set_xlabel('Sample Index')
    axes2[3].set_ylabel('Amplitude')
    axes2[3].legend(loc='upper right')
    axes2[3].grid(True, alpha=0.3)

    plt.suptitle('LBEADS-NET v8 Detailed Analysis (ISTA + Banded Ops)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, 'demo_detailed_analysis.png'), dpi=150)
    print(f"  Saved to {os.path.join(script_dir, 'demo_detailed_analysis.png')}")

    print("\nSynthetic demo complete!")
    return model, trained


if __name__ == "__main__":
    model, trained = main()
    plt.show()
