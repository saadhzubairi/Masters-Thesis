"""
Demo script for LBEADS-NET

This script demonstrates the LBEADS-NET model on a single example,
comparing it with the original BEADS algorithm.
"""

import torch
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import os
import sys
import time

# Add parent directory to path for importing original BEADS
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'BEADS', 'Replicate'))

from lbeads_net import LBEADS_NET, LBEADS_NET_Fast
# Ensure the parent BEADS directory is on sys.path, then import the original beads function
beads_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'BEADS', 'Replicate')
beads_dir = os.path.normpath(beads_dir)
if beads_dir not in sys.path:
    sys.path.insert(0, beads_dir)
from beads import beads as original_beads


def main():
    """Demo script comparing LBEADS-NET with original BEADS."""
    print("=" * 60)
    print("LBEADS-NET Demo")
    print("=" * 60)
    
    # Get data directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(script_dir), 'BEADS', 'data')
    
    # Load data
    print("\nLoading data...")
    noise_data = sio.loadmat(os.path.join(data_dir, 'noise.mat'))
    chromatograms_data = sio.loadmat(os.path.join(data_dir, 'chromatograms.mat'))
    
    noise = noise_data['noise'].flatten()
    X = chromatograms_data['X']
    
    # Create noisy signal (same as main.py in BEADS)
    y = X[:, 2] + noise * 0.5
    N = len(y)
    
    print(f"Signal length: {N}")
    
    # Parameters
    fc = 0.006
    d = 1
    r = 6
    amp = 0.8
    lam0 = 0.5 * amp
    lam1 = 5 * amp
    lam2 = 4 * amp
    
    # Run original BEADS
    print("\nRunning original BEADS...")
    start_time = time.time()
    x_beads, f_beads, cost_beads = original_beads(y, d, fc, r, lam0, lam1, lam2, Nit=30)
    beads_time = time.time() - start_time
    print(f"Original BEADS time: {beads_time:.4f} seconds")
    
    # Convert to numpy
    x_beads = x_beads.numpy()
    f_beads = f_beads.numpy()
    
    # Create LBEADS-NET with same initialization
    print("\nCreating LBEADS-NET (exact version)...")
    model_exact = LBEADS_NET(
        N=N,
        d=d,
        fc=fc,
        num_layers=30,  # Same as BEADS iterations
        shared_params=True,  # All layers share parameters (like original BEADS)
        init_lam0=lam0,
        init_lam1=lam1,
        init_lam2=lam2,
        init_r=r
    )
    
    # Run LBEADS-NET (exact)
    y_tensor = torch.tensor(y, dtype=torch.float64)
    
    print("Running LBEADS-NET (exact)...")
    start_time = time.time()
    with torch.no_grad():
        x_lbeads, f_lbeads = model_exact(y_tensor)
    lbeads_time = time.time() - start_time
    print(f"LBEADS-NET time: {lbeads_time:.4f} seconds")
    
    x_lbeads = x_lbeads.numpy()
    f_lbeads = f_lbeads.numpy()
    
    # Create fast version
    print("\nCreating LBEADS-NET (fast version)...")
    model_fast = LBEADS_NET_Fast(
        N=N,
        d=d,
        fc=fc,
        num_layers=30,
        init_lam0=lam0,
        init_lam1=lam1,
        init_lam2=lam2,
        init_r=r,
        init_step_size=0.1
    )
    
    print("Running LBEADS-NET (fast)...")
    start_time = time.time()
    with torch.no_grad():
        x_fast, f_fast = model_fast(y_tensor)
    fast_time = time.time() - start_time
    print(f"LBEADS-NET (fast) time: {fast_time:.4f} seconds")
    
    x_fast = x_fast.numpy()
    f_fast = f_fast.numpy()
    
    # Compare results
    print("\n" + "=" * 40)
    print("Comparison with Original BEADS")
    print("=" * 40)
    
    mse_exact = np.mean((x_lbeads - x_beads) ** 2)
    mse_fast = np.mean((x_fast - x_beads) ** 2)
    
    print(f"MSE (LBEADS-NET exact vs BEADS): {mse_exact:.2e}")
    print(f"MSE (LBEADS-NET fast vs BEADS): {mse_fast:.2e}")
    
    # Print learned parameters
    print("\nModel parameters (LBEADS-NET exact):")
    params = model_exact.get_learned_params()
    for k, v in params.items():
        print(f"  {k}: {v:.4f}")
    
    # Plot comparison
    fig, axes = plt.subplots(4, 1, figsize=(12, 12))
    
    xlim1 = [0, 3800]
    ylim1 = [-50, 200]
    
    # Original data
    axes[0].plot(y, 'b', linewidth=0.5)
    axes[0].set_title('Noisy Data')
    axes[0].set_xlim(xlim1)
    axes[0].set_ylim(ylim1)
    
    # BEADS result
    axes[1].plot(y, color=[0.7, 0.7, 0.7], linewidth=0.5, label='Data')
    axes[1].plot(f_beads, 'b', linewidth=1, label='Baseline (BEADS)')
    axes[1].legend()
    axes[1].set_title('Original BEADS Baseline')
    axes[1].set_xlim(xlim1)
    axes[1].set_ylim(ylim1)
    
    # LBEADS-NET exact result
    axes[2].plot(y, color=[0.7, 0.7, 0.7], linewidth=0.5, label='Data')
    axes[2].plot(f_lbeads, 'r', linewidth=1, label='Baseline (LBEADS-NET)')
    axes[2].legend()
    axes[2].set_title('LBEADS-NET (Exact) Baseline')
    axes[2].set_xlim(xlim1)
    axes[2].set_ylim(ylim1)
    
    # Compare signals
    axes[3].plot(x_beads, 'b', linewidth=0.5, label='BEADS Signal')
    axes[3].plot(x_lbeads, 'r--', linewidth=0.5, label='LBEADS-NET Signal')
    axes[3].legend()
    axes[3].set_title('Baseline-Corrected Signal Comparison')
    axes[3].set_xlim(xlim1)
    axes[3].set_ylim(ylim1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, 'demo_comparison.png'), dpi=150)
    print(f"\nSaved comparison to {os.path.join(script_dir, 'demo_comparison.png')}")
    
    # Plot intermediate results
    print("\nGenerating intermediate results visualization...")
    with torch.no_grad():
        x_lbeads, f_lbeads, intermediates = model_exact(y_tensor, return_intermediate=True)
    
    fig2, axes2 = plt.subplots(2, 3, figsize=(15, 8))
    
    layers_to_show = [0, 5, 10, 15, 20, 29]  # Iterations to visualize
    
    for idx, (ax, layer_idx) in enumerate(zip(axes2.flat, layers_to_show)):
        if layer_idx < len(intermediates):
            x_iter = intermediates[layer_idx].squeeze().numpy()
            ax.plot(y, color=[0.8, 0.8, 0.8], linewidth=0.5)
            ax.plot(x_iter, 'b', linewidth=0.5)
            ax.set_title(f'Iteration {layer_idx}')
            ax.set_xlim([0, 1000])
            ax.set_ylim(ylim1)
    
    plt.suptitle('LBEADS-NET Intermediate Results (Unrolled Iterations)')
    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, 'demo_iterations.png'), dpi=150)
    print(f"Saved iterations to {os.path.join(script_dir, 'demo_iterations.png')}")
    
    plt.show()
    
    print("\nDemo complete!")


if __name__ == "__main__":
    main()
