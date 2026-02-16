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

from lbeads_net import LBEADS_NET, LBEADS_NET_Fast

# Import synthetic data generator from train.py
from train import SyntheticDataGenerator, SyntheticSignal


def load_trained_model(script_dir: str, N: int = 1024):
    """
    Load the most recently trained model from the script directory.
    
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
    
    # Find most recent model file (try sparsity models first, then synthetic)
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
    
    # Sort by modification time and get most recent
    model_files.sort(key=os.path.getmtime, reverse=True)
    model_path = model_files[0]
    
    print(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
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
        init_step_size=0.001
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
    N = 1024  # Signal length (must match training)
    num_test_samples = 6  # Number of test samples to visualize
    
    # Generate synthetic test data
    print("\nGenerating synthetic test data...")
    generator = SyntheticDataGenerator(N=N, seed=123)  # Different seed from training
    
    test_signals = []
    for i in range(num_test_samples):
        noise_level = 0.05 + (i * 0.02)  # Varying noise levels
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
        y_tensor = torch.tensor(signal.y, dtype=torch.float64).unsqueeze(0)
        
        start_time = time.time()
        with torch.no_grad():
            x_pred, f_pred = model(y_tensor)
        inference_time = time.time() - start_time
        total_time += inference_time
        
        x_pred_np = x_pred[0].numpy()
        f_pred_np = f_pred[0].numpy()
        
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
    
    # Figure 2: Detailed view of one sample
    fig2, axes2 = plt.subplots(3, 1, figsize=(14, 10))
    
    result = results[0]  # Use first sample
    signal = result['signal']
    
    # Panel 1: Observed signal
    axes2[0].plot(signal.y, 'b', linewidth=0.5)
    axes2[0].set_title('Observed Signal: y = peaks + baseline + noise')
    axes2[0].set_xlim([0, N])
    axes2[0].set_ylabel('Amplitude')
    
    # Panel 2: Peak recovery
    axes2[1].plot(signal.x_true, 'g', linewidth=1, label='Ground Truth Peaks')
    axes2[1].plot(result['x_pred'], 'b--', linewidth=1, label='Predicted Peaks')
    axes2[1].set_title(f'Peak Recovery (MSE={result["mse"]:.6f}, Corr={result["correlation"]:.4f})')
    axes2[1].set_xlim([0, N])
    axes2[1].set_ylabel('Amplitude')
    axes2[1].legend()
    
    # Panel 3: Baseline estimation
    axes2[2].plot(signal.f_true, 'm', linewidth=1, label='Ground Truth Baseline')
    axes2[2].plot(result['f_pred'], 'r--', linewidth=1, label='Predicted Baseline')
    axes2[2].set_title(f'Baseline Estimation (MSE={result["baseline_mse"]:.6f})')
    axes2[2].set_xlim([0, N])
    axes2[2].set_xlabel('Sample Index')
    axes2[2].set_ylabel('Amplitude')
    axes2[2].legend()
    
    plt.suptitle('LBEADS-NET Detailed Analysis', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, 'demo_detailed_analysis.png'), dpi=150)
    print(f"  Saved to {os.path.join(script_dir, 'demo_detailed_analysis.png')}")
    
    plt.show()
    
    print("\nDemo complete!")


if __name__ == "__main__":
    main()
