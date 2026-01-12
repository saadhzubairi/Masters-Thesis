"""
Visualization Module for LBEADS-NET Evaluation

This module provides plotting functions for generating thesis-quality figures
comparing baseline estimation methods.

Figures:
- Figure 1: Multi-panel comparison of methods on representative signals
- Intermediate iteration plots
- Metric summary plots

Author: Thesis Work
Date: January 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

# Import local modules
from synthetic_data_generator import SyntheticSignal
from metrics import EvaluationResult


# Set publication-quality style
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})


@dataclass
class PlotResult:
    """Container for method results for plotting."""
    method_name: str
    x_est: np.ndarray      # Estimated signal
    f_est: np.ndarray      # Estimated baseline
    color: str             # Plot color
    linestyle: str = '-'   # Line style


def plot_single_comparison(signal: SyntheticSignal,
                           results: List[PlotResult],
                           title: str = "",
                           figsize: Tuple[float, float] = (12, 10),
                           save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot comparison of methods on a single signal.
    
    Creates a 4-panel figure showing:
    1. Observed signal y
    2. Baseline comparison (f_true vs f_est)
    3. Signal comparison (x_true vs x_est)
    4. Residual/error analysis
    
    Args:
        signal: SyntheticSignal with ground truth
        results: List of PlotResult objects for each method
        title: Figure title
        figsize: Figure size
        save_path: Path to save figure (optional)
        
    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
    
    N = len(signal.y)
    t = np.arange(N)
    
    # Panel 1: Observed signal
    axes[0].plot(t, signal.y, 'gray', linewidth=0.5, alpha=0.8, label='y (observed)')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title('Observed Noisy Signal')
    axes[0].legend(loc='upper right')
    
    # Panel 2: Baseline comparison
    axes[1].plot(t, signal.y, 'lightgray', linewidth=0.3, alpha=0.5)
    axes[1].plot(t, signal.f_true, 'k--', linewidth=1.5, label='f_true (ground truth)')
    for result in results:
        axes[1].plot(t, result.f_est, color=result.color, 
                     linestyle=result.linestyle, linewidth=1,
                     label=f'f ({result.method_name})')
    axes[1].set_ylabel('Amplitude')
    axes[1].set_title('Baseline Estimation')
    axes[1].legend(loc='upper right')
    
    # Panel 3: Signal comparison
    axes[2].plot(t, signal.x_true, 'k--', linewidth=1.5, label='x_true (ground truth)')
    for result in results:
        axes[2].plot(t, result.x_est, color=result.color,
                     linestyle=result.linestyle, linewidth=1,
                     label=f'x ({result.method_name})')
    axes[2].set_ylabel('Amplitude')
    axes[2].set_title('Sparse Signal Recovery')
    axes[2].legend(loc='upper right')
    
    # Panel 4: Error comparison
    for result in results:
        error = signal.x_true - result.x_est
        axes[3].plot(t, error, color=result.color,
                     linestyle=result.linestyle, linewidth=0.7,
                     label=f'Error ({result.method_name})')
    axes[3].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    axes[3].set_xlabel('Sample Index')
    axes[3].set_ylabel('Error')
    axes[3].set_title('Estimation Error (x_true - x_est)')
    axes[3].legend(loc='upper right')
    
    # Add overall title
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    return fig


def plot_figure1_thesis(signals: List[SyntheticSignal],
                        beads_results: List[Tuple[np.ndarray, np.ndarray]],
                        lbeads_results: List[Tuple[np.ndarray, np.ndarray]],
                        case_labels: List[str],
                        figsize: Tuple[float, float] = (14, 12),
                        save_path: Optional[str] = None) -> plt.Figure:
    """
    Generate Figure 1 for thesis: Multi-case comparison.
    
    Shows 3-5 representative cases (easy, medium, hard, Laplacian).
    Each case gets one row with signal + baseline + error subplots.
    
    Args:
        signals: List of SyntheticSignal objects (representative cases)
        beads_results: List of (x_beads, f_beads) tuples
        lbeads_results: List of (x_lbeads, f_lbeads) tuples
        case_labels: Labels for each case (e.g., "Low Noise", "High Noise")
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        matplotlib Figure object
    """
    n_cases = len(signals)
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(n_cases, 3, figure=fig, hspace=0.3, wspace=0.25)
    
    for i, (signal, (x_beads, f_beads), (x_lbeads, f_lbeads), label) in enumerate(
            zip(signals, beads_results, lbeads_results, case_labels)):
        
        N = len(signal.y)
        t = np.arange(N)
        
        # Column 1: Observed signal with baseline estimates
        ax1 = fig.add_subplot(gs[i, 0])
        ax1.plot(t, signal.y, 'gray', linewidth=0.4, alpha=0.7, label='y')
        ax1.plot(t, signal.f_true, 'k--', linewidth=1.2, label='f_true')
        ax1.plot(t, f_beads, 'r', linewidth=0.8, label='BEADS')
        ax1.plot(t, f_lbeads, 'b', linewidth=0.8, label='LBEADS-NET')
        ax1.set_ylabel('Amplitude')
        if i == 0:
            ax1.set_title('Baseline Estimation')
            ax1.legend(loc='upper right', fontsize=7)
        if i == n_cases - 1:
            ax1.set_xlabel('Sample')
        ax1.text(0.02, 0.95, label, transform=ax1.transAxes, 
                fontsize=9, fontweight='bold', va='top')
        
        # Column 2: Signal recovery
        ax2 = fig.add_subplot(gs[i, 1])
        ax2.plot(t, signal.x_true, 'k--', linewidth=1.2, label='x_true')
        ax2.plot(t, x_beads, 'r', linewidth=0.8, alpha=0.8, label='BEADS')
        ax2.plot(t, x_lbeads, 'b', linewidth=0.8, alpha=0.8, label='LBEADS-NET')
        if i == 0:
            ax2.set_title('Signal Recovery')
            ax2.legend(loc='upper right', fontsize=7)
        if i == n_cases - 1:
            ax2.set_xlabel('Sample')
        
        # Column 3: Error comparison
        ax3 = fig.add_subplot(gs[i, 2])
        error_beads = signal.x_true - x_beads
        error_lbeads = signal.x_true - x_lbeads
        ax3.plot(t, error_beads, 'r', linewidth=0.6, alpha=0.7, label='BEADS')
        ax3.plot(t, error_lbeads, 'b', linewidth=0.6, alpha=0.7, label='LBEADS-NET')
        ax3.axhline(y=0, color='k', linestyle='-', linewidth=0.3)
        if i == 0:
            ax3.set_title('Estimation Error')
            ax3.legend(loc='upper right', fontsize=7)
        if i == n_cases - 1:
            ax3.set_xlabel('Sample')
        
        # Compute and show MSE in error panel
        mse_beads = np.mean(error_beads**2)
        mse_lbeads = np.mean(error_lbeads**2)
        ax3.text(0.98, 0.95, f'MSE: B={mse_beads:.4f}\n      L={mse_lbeads:.4f}',
                transform=ax3.transAxes, fontsize=7, va='top', ha='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add figure caption info
    fig.suptitle('Figure 1: Comparison of BEADS and LBEADS-NET on Synthetic Signals',
                 fontsize=13, fontweight='bold', y=1.01)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved Figure 1 to {save_path}")
    
    return fig


def plot_metrics_summary(beads_results: List[EvaluationResult],
                         lbeads_results: List[EvaluationResult],
                         figsize: Tuple[float, float] = (12, 5),
                         save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot summary of metrics as bar charts.
    
    Args:
        beads_results: Results from classical BEADS
        lbeads_results: Results from LBEADS-NET
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        matplotlib Figure object
    """
    from metrics import aggregate_results
    
    beads_agg = aggregate_results(beads_results)
    lbeads_agg = aggregate_results(lbeads_results)
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    methods = ['BEADS', 'LBEADS-NET']
    x = np.arange(len(methods))
    width = 0.6
    colors = ['red', 'blue']
    
    # MSE Signal
    mse_sig_means = [beads_agg['mse_signal'][0], lbeads_agg['mse_signal'][0]]
    mse_sig_stds = [beads_agg['mse_signal'][1], lbeads_agg['mse_signal'][1]]
    bars1 = axes[0].bar(x, mse_sig_means, width, yerr=mse_sig_stds, 
                        color=colors, capsize=5, alpha=0.8)
    axes[0].set_ylabel('MSE')
    axes[0].set_title('MSE (Signal) ↓')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(methods)
    
    # MSE Baseline
    mse_base_means = [beads_agg['mse_baseline'][0], lbeads_agg['mse_baseline'][0]]
    mse_base_stds = [beads_agg['mse_baseline'][1], lbeads_agg['mse_baseline'][1]]
    bars2 = axes[1].bar(x, mse_base_means, width, yerr=mse_base_stds,
                        color=colors, capsize=5, alpha=0.8)
    axes[1].set_ylabel('MSE')
    axes[1].set_title('MSE (Baseline) ↓')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(methods)
    
    # Delta SNR
    dsnr_means = [beads_agg['delta_snr'][0], lbeads_agg['delta_snr'][0]]
    dsnr_stds = [beads_agg['delta_snr'][1], lbeads_agg['delta_snr'][1]]
    bars3 = axes[2].bar(x, dsnr_means, width, yerr=dsnr_stds,
                        color=colors, capsize=5, alpha=0.8)
    axes[2].set_ylabel('dB')
    axes[2].set_title('ΔSNR ↑')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(methods)
    
    # Add value labels on bars
    for ax, means, stds in [(axes[0], mse_sig_means, mse_sig_stds),
                            (axes[1], mse_base_means, mse_base_stds),
                            (axes[2], dsnr_means, dsnr_stds)]:
        for i, (mean, std) in enumerate(zip(means, stds)):
            ax.text(i, mean + std + 0.02 * max(means), f'{mean:.3f}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.suptitle('Quantitative Comparison: BEADS vs LBEADS-NET', 
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved metrics summary to {save_path}")
    
    return fig


def plot_noise_level_analysis(noise_levels: List[float],
                              beads_mse: List[float],
                              lbeads_mse: List[float],
                              figsize: Tuple[float, float] = (8, 5),
                              save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot MSE vs noise level for both methods.
    
    Args:
        noise_levels: List of noise levels
        beads_mse: MSE values for BEADS at each noise level
        lbeads_mse: MSE values for LBEADS-NET at each noise level
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(noise_levels, beads_mse, 'ro-', linewidth=2, markersize=8, label='BEADS')
    ax.plot(noise_levels, lbeads_mse, 'bs-', linewidth=2, markersize=8, label='LBEADS-NET')
    
    ax.set_xlabel('Noise Level (σ)')
    ax.set_ylabel('MSE (Signal)')
    ax.set_title('Robustness to Noise: MSE vs Noise Level')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved noise analysis to {save_path}")
    
    return fig


def plot_boxplot_comparison(beads_results: List[EvaluationResult],
                            lbeads_results: List[EvaluationResult],
                            metric: str = 'mse_signal',
                            figsize: Tuple[float, float] = (6, 5),
                            save_path: Optional[str] = None) -> plt.Figure:
    """
    Create boxplot comparison for a specific metric.
    
    Args:
        beads_results: Results from classical BEADS
        lbeads_results: Results from LBEADS-NET
        metric: Which metric to plot
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    beads_values = [getattr(r, metric) for r in beads_results]
    lbeads_values = [getattr(r, metric) for r in lbeads_results]
    
    data = [beads_values, lbeads_values]
    bp = ax.boxplot(data, labels=['BEADS', 'LBEADS-NET'], patch_artist=True)
    
    colors = ['lightcoral', 'lightblue']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    metric_labels = {
        'mse_signal': 'MSE (Signal)',
        'mse_baseline': 'MSE (Baseline)',
        'delta_snr': 'ΔSNR (dB)'
    }
    
    ax.set_ylabel(metric_labels.get(metric, metric))
    ax.set_title(f'Distribution of {metric_labels.get(metric, metric)}')
    ax.grid(True, alpha=0.3, axis='y')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved boxplot to {save_path}")
    
    return fig


if __name__ == "__main__":
    # Test visualization with dummy data
    print("Testing Visualization Module")
    print("=" * 50)
    
    from synthetic_data_generator import SyntheticDataGenerator
    
    # Generate test data
    generator = SyntheticDataGenerator(N=1024, seed=42)
    signal = generator.generate_signal(noise_type='gaussian', noise_level=0.1)
    
    # Create dummy results
    x_beads = signal.x_true + np.random.normal(0, 0.05, len(signal.x_true))
    f_beads = signal.f_true + np.random.normal(0, 0.03, len(signal.f_true))
    
    x_lbeads = signal.x_true + np.random.normal(0, 0.03, len(signal.x_true))
    f_lbeads = signal.f_true + np.random.normal(0, 0.02, len(signal.f_true))
    
    # Test single comparison plot
    results = [
        PlotResult('BEADS', x_beads, f_beads, 'red'),
        PlotResult('LBEADS-NET', x_lbeads, f_lbeads, 'blue')
    ]
    
    fig = plot_single_comparison(signal, results, 
                                  title="Test Comparison Plot",
                                  save_path='test_comparison.png')
    
    print("\nTest complete! Generated test_comparison.png")
    plt.show()
