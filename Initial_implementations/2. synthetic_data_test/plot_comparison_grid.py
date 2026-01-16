"""
Plot Comparison Grid: Before/After Visualization

Creates a multi-row, 3-column figure showing:
- Column 1: Original signal (y) with ground truth components
- Column 2: BEADS result (x_beads, f_beads)
- Column 3: LBEADS-NET result (x_lbeads, f_lbeads)

Author: Thesis Work
Date: January 2026
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Add parent directories to path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, 'LBEADS_NETv1'))
sys.path.insert(0, os.path.join(parent_dir, 'BEADS', 'Replicate'))

from synthetic_data_generator import SyntheticDataGenerator, load_dataset
from beads import beads as original_beads
from lbeads_net import LBEADS_NET_Fast
from metrics import compute_mse


def load_trained_model(model_path, N):
    """Load trained LBEADS-NET model."""
    checkpoint = torch.load(model_path, weights_only=False)
    saved_config = checkpoint.get('config', {})
    
    model = LBEADS_NET_Fast(
        N=N,
        d=saved_config.get('d', 1),
        fc=saved_config.get('fc', 0.006),
        num_layers=saved_config.get('num_layers', 10),
        init_lam0=0.5,
        init_lam1=4.0,
        init_lam2=4.0,
        init_r=6.0,
        init_step_size=0.1
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def run_beads(y, d=1, fc=0.01, r=6.0, lam0=0.3, lam1=3.0, lam2=3.0, Nit=50):
    """
    Run classical BEADS with parameters tuned for synthetic data.
    
    Key parameter insights:
    - fc: Higher value (0.01-0.02) to preserve wider peaks
    - lam0: Lower value to allow more signal through
    - lam1, lam2: Lower values for less aggressive smoothing
    - Nit: More iterations for better convergence
    """
    x, f, _ = original_beads(y, d, fc, r, lam0, lam1, lam2, Nit)
    if isinstance(x, torch.Tensor):
        x = x.numpy()
    if isinstance(f, torch.Tensor):
        f = f.numpy()
    return x, f


def run_lbeads(y, model):
    """Run LBEADS-NET."""
    y_tensor = torch.tensor(y, dtype=torch.float64)
    with torch.no_grad():
        x, f = model(y_tensor)
    return x.numpy(), f.numpy()


def plot_comparison_grid(signals, beads_results, lbeads_results, 
                         n_samples=4, figsize=(16, 12), save_path=None):
    """
    Plot a grid comparing original signals with BEADS and LBEADS-NET results.
    
    Args:
        signals: List of SyntheticSignal objects
        beads_results: List of (x_beads, f_beads) tuples
        lbeads_results: List of (x_lbeads, f_lbeads) tuples
        n_samples: Number of samples to show
        figsize: Figure size
        save_path: Path to save figure
    """
    n_samples = min(n_samples, len(signals))
    
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(n_samples, 3, figure=fig, hspace=0.35, wspace=0.2)
    
    for i in range(n_samples):
        signal = signals[i]
        x_beads, f_beads = beads_results[i]
        x_lbeads, f_lbeads = lbeads_results[i]
        
        N = len(signal.y)
        t = np.arange(N)
        
        # Compute MSE for display
        mse_beads_x = compute_mse(signal.x_true, x_beads)
        mse_beads_f = compute_mse(signal.f_true, f_beads)
        mse_lbeads_x = compute_mse(signal.x_true, x_lbeads)
        mse_lbeads_f = compute_mse(signal.f_true, f_lbeads)
        
        noise_info = f"{signal.noise_type}, σ={signal.noise_level:.2f}"
        
        # ===== Column 1: Original Signal =====
        ax1 = fig.add_subplot(gs[i, 0])
        ax1.plot(t, signal.y, 'gray', linewidth=0.5, alpha=0.8, label='y (observed)')
        ax1.plot(t, signal.x_true, 'b', linewidth=1.2, label='x_true (sparse)')
        ax1.plot(t, signal.f_true, 'g', linewidth=1.2, label='f_true (baseline)')
        
        if i == 0:
            ax1.set_title('Original Signal', fontsize=12, fontweight='bold')
            ax1.legend(loc='upper right', fontsize=7)
        ax1.set_ylabel(f'Sample {i+1}\n({noise_info})', fontsize=9)
        if i == n_samples - 1:
            ax1.set_xlabel('Sample Index')
        ax1.grid(True, alpha=0.3)
        
        # ===== Column 2: BEADS Result =====
        ax2 = fig.add_subplot(gs[i, 1])
        ax2.plot(t, signal.y, 'lightgray', linewidth=0.3, alpha=0.5)
        ax2.plot(t, signal.x_true, 'k--', linewidth=0.8, alpha=0.5, label='x_true')
        ax2.plot(t, x_beads, 'r', linewidth=1, label='x_beads')
        ax2.plot(t, f_beads, 'orange', linewidth=1, label='f_beads')
        
        if i == 0:
            ax2.set_title('BEADS Result', fontsize=12, fontweight='bold')
            ax2.legend(loc='upper right', fontsize=7)
        if i == n_samples - 1:
            ax2.set_xlabel('Sample Index')
        ax2.grid(True, alpha=0.3)
        
        # Add MSE annotation
        ax2.text(0.02, 0.98, f'MSE(x)={mse_beads_x:.3f}\nMSE(f)={mse_beads_f:.3f}',
                transform=ax2.transAxes, fontsize=7, va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # ===== Column 3: LBEADS-NET Result =====
        ax3 = fig.add_subplot(gs[i, 2])
        ax3.plot(t, signal.y, 'lightgray', linewidth=0.3, alpha=0.5)
        ax3.plot(t, signal.x_true, 'k--', linewidth=0.8, alpha=0.5, label='x_true')
        ax3.plot(t, x_lbeads, 'b', linewidth=1, label='x_lbeads')
        ax3.plot(t, f_lbeads, 'cyan', linewidth=1, label='f_lbeads')
        
        if i == 0:
            ax3.set_title('LBEADS-NET Result', fontsize=12, fontweight='bold')
            ax3.legend(loc='upper right', fontsize=7)
        if i == n_samples - 1:
            ax3.set_xlabel('Sample Index')
        ax3.grid(True, alpha=0.3)
        
        # Add MSE annotation
        ax3.text(0.02, 0.98, f'MSE(x)={mse_lbeads_x:.3f}\nMSE(f)={mse_lbeads_f:.3f}',
                transform=ax3.transAxes, fontsize=7, va='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.suptitle('Comparison: Original Signals vs BEADS vs LBEADS-NET', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison grid to {save_path}")
    
    return fig


def plot_detailed_comparison(signal, x_beads, f_beads, x_lbeads, f_lbeads,
                              figsize=(14, 10), save_path=None):
    """
    Plot detailed comparison for a single signal with error analysis.
    
    Shows:
    - Row 1: Full signal comparison
    - Row 2: Baseline comparison
    - Row 3: Sparse signal comparison
    - Row 4: Error comparison
    """
    fig, axes = plt.subplots(4, 2, figsize=figsize)
    
    N = len(signal.y)
    t = np.arange(N)
    
    noise_info = f"{signal.noise_type}, σ={signal.noise_level:.2f}"
    
    # ===== Row 1: Full observed signal =====
    axes[0, 0].plot(t, signal.y, 'gray', linewidth=0.5, label='y (observed)')
    axes[0, 0].plot(t, signal.x_true + signal.f_true, 'k--', linewidth=1, label='x_true + f_true')
    axes[0, 0].set_title(f'Observed Signal ({noise_info})')
    axes[0, 0].legend(loc='upper right', fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(t, signal.noise, 'gray', linewidth=0.5)
    axes[0, 1].set_title('Noise Component')
    axes[0, 1].grid(True, alpha=0.3)
    
    # ===== Row 2: Baseline comparison =====
    axes[1, 0].plot(t, signal.f_true, 'k', linewidth=1.5, label='f_true')
    axes[1, 0].plot(t, f_beads, 'r', linewidth=1, label='f_beads')
    axes[1, 0].plot(t, f_lbeads, 'b', linewidth=1, label='f_lbeads')
    axes[1, 0].set_title('Baseline Estimation')
    axes[1, 0].legend(loc='upper right', fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Baseline error
    axes[1, 1].plot(t, signal.f_true - f_beads, 'r', linewidth=0.8, label='BEADS error')
    axes[1, 1].plot(t, signal.f_true - f_lbeads, 'b', linewidth=0.8, label='LBEADS error')
    axes[1, 1].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    axes[1, 1].set_title('Baseline Error (f_true - f_est)')
    axes[1, 1].legend(loc='upper right', fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)
    
    # ===== Row 3: Sparse signal comparison =====
    axes[2, 0].plot(t, signal.x_true, 'k', linewidth=1.5, label='x_true')
    axes[2, 0].plot(t, x_beads, 'r', linewidth=1, alpha=0.8, label='x_beads')
    axes[2, 0].plot(t, x_lbeads, 'b', linewidth=1, alpha=0.8, label='x_lbeads')
    axes[2, 0].set_title('Sparse Signal Recovery')
    axes[2, 0].legend(loc='upper right', fontsize=8)
    axes[2, 0].grid(True, alpha=0.3)
    
    # Signal error
    axes[2, 1].plot(t, signal.x_true - x_beads, 'r', linewidth=0.8, label='BEADS error')
    axes[2, 1].plot(t, signal.x_true - x_lbeads, 'b', linewidth=0.8, label='LBEADS error')
    axes[2, 1].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    axes[2, 1].set_title('Signal Error (x_true - x_est)')
    axes[2, 1].legend(loc='upper right', fontsize=8)
    axes[2, 1].grid(True, alpha=0.3)
    
    # ===== Row 4: Reconstruction check =====
    recon_beads = x_beads + f_beads
    recon_lbeads = x_lbeads + f_lbeads
    
    axes[3, 0].plot(t, signal.y, 'gray', linewidth=0.5, alpha=0.7, label='y')
    axes[3, 0].plot(t, recon_beads, 'r', linewidth=0.8, label='x+f (BEADS)')
    axes[3, 0].plot(t, recon_lbeads, 'b', linewidth=0.8, label='x+f (LBEADS)')
    axes[3, 0].set_title('Reconstruction (x_est + f_est)')
    axes[3, 0].legend(loc='upper right', fontsize=8)
    axes[3, 0].set_xlabel('Sample Index')
    axes[3, 0].grid(True, alpha=0.3)
    
    # MSE bar comparison
    mse_beads_x = compute_mse(signal.x_true, x_beads)
    mse_beads_f = compute_mse(signal.f_true, f_beads)
    mse_lbeads_x = compute_mse(signal.x_true, x_lbeads)
    mse_lbeads_f = compute_mse(signal.f_true, f_lbeads)
    
    x_pos = np.arange(2)
    width = 0.35
    
    axes[3, 1].bar(x_pos - width/2, [mse_beads_x, mse_beads_f], width, 
                   label='BEADS', color='red', alpha=0.7)
    axes[3, 1].bar(x_pos + width/2, [mse_lbeads_x, mse_lbeads_f], width,
                   label='LBEADS-NET', color='blue', alpha=0.7)
    axes[3, 1].set_xticks(x_pos)
    axes[3, 1].set_xticklabels(['MSE (signal)', 'MSE (baseline)'])
    axes[3, 1].set_title('MSE Comparison')
    axes[3, 1].legend(fontsize=8)
    axes[3, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved detailed comparison to {save_path}")
    
    return fig


def main():
    """Generate comparison plots."""
    print("=" * 60)
    print("Generating Before/After Comparison Plots")
    print("=" * 60)
    
    # Configuration
    results_dir = os.path.join(script_dir, 'results')
    model_path = os.path.join(script_dir, 'trained_models', 'lbeads_net_trained.pth')
    
    # Generate fresh signals for visualization
    print("\nGenerating visualization signals...")
    generator = SyntheticDataGenerator(N=1024, seed=123)  # Different seed for variety
    
    # Generate diverse samples
    signals = []
    
    # Easy case (low Gaussian noise)
    signals.append(generator.generate_signal(noise_type='gaussian', noise_level=0.05))
    
    # Medium case (moderate Gaussian noise)
    signals.append(generator.generate_signal(noise_type='gaussian', noise_level=0.10))
    
    # Hard case (high Gaussian noise)
    signals.append(generator.generate_signal(noise_type='gaussian', noise_level=0.15))
    
    # Laplacian noise case
    signals.append(generator.generate_signal(noise_type='laplacian', noise_level=0.10))
    
    print(f"Generated {len(signals)} signals for visualization")
    
    # Load trained model
    print("\nLoading trained LBEADS-NET model...")
    model = load_trained_model(model_path, N=1024)
    
    # Process all signals
    print("Processing signals with BEADS and LBEADS-NET...")
    beads_results = []
    lbeads_results = []
    
    for i, signal in enumerate(signals):
        print(f"  Processing signal {i+1}/{len(signals)}...")
        x_beads, f_beads = run_beads(signal.y)
        x_lbeads, f_lbeads = run_lbeads(signal.y, model)
        beads_results.append((x_beads, f_beads))
        lbeads_results.append((x_lbeads, f_lbeads))
    
    # Generate grid comparison
    print("\nGenerating comparison grid...")
    grid_path = os.path.join(results_dir, 'comparison_grid.png')
    plot_comparison_grid(signals, beads_results, lbeads_results,
                         n_samples=4, save_path=grid_path)
    
    # Generate detailed comparison for first signal
    print("\nGenerating detailed comparison for sample 1...")
    detailed_path = os.path.join(results_dir, 'detailed_comparison_sample1.png')
    plot_detailed_comparison(signals[0], 
                             beads_results[0][0], beads_results[0][1],
                             lbeads_results[0][0], lbeads_results[0][1],
                             save_path=detailed_path)
    
    # Generate detailed comparison for hard case
    print("Generating detailed comparison for hard case (sample 3)...")
    detailed_path2 = os.path.join(results_dir, 'detailed_comparison_hard.png')
    plot_detailed_comparison(signals[2], 
                             beads_results[2][0], beads_results[2][1],
                             lbeads_results[2][0], lbeads_results[2][1],
                             save_path=detailed_path2)
    
    print("\n" + "=" * 60)
    print("PLOTS GENERATED SUCCESSFULLY")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  - {grid_path}")
    print(f"  - {detailed_path}")
    print(f"  - {detailed_path2}")
    
    plt.show()


if __name__ == "__main__":
    main()
