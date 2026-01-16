"""
Metrics for LBEADS-NET Evaluation

This module provides evaluation metrics for comparing baseline estimation
and denoising methods.

Metrics:
- MSE (Mean Squared Error) for signal and baseline
- SNR (Signal-to-Noise Ratio) and ΔSNR (improvement)
- Additional metrics for comprehensive evaluation

Author: Thesis Work
Date: January 2026
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class EvaluationResult:
    """Container for evaluation metrics of a single signal."""
    mse_signal: float      # MSE between estimated and true signal
    mse_baseline: float    # MSE between estimated and true baseline
    snr_in: float          # Input SNR (dB)
    snr_out: float         # Output SNR (dB)
    delta_snr: float       # SNR improvement (dB)
    
    # Optional additional metrics
    mae_signal: Optional[float] = None      # Mean Absolute Error for signal
    mae_baseline: Optional[float] = None    # Mean Absolute Error for baseline
    peak_error: Optional[float] = None      # Error at peak locations
    

def compute_mse(x_true: np.ndarray, x_est: np.ndarray) -> float:
    """
    Compute Mean Squared Error.
    
    MSE = (1/N) * ||x_true - x_est||^2
    
    Args:
        x_true: Ground truth signal
        x_est: Estimated signal
        
    Returns:
        MSE value
    """
    return np.mean((x_true - x_est) ** 2)


def compute_mae(x_true: np.ndarray, x_est: np.ndarray) -> float:
    """
    Compute Mean Absolute Error.
    
    MAE = (1/N) * ||x_true - x_est||_1
    
    Args:
        x_true: Ground truth signal
        x_est: Estimated signal
        
    Returns:
        MAE value
    """
    return np.mean(np.abs(x_true - x_est))


def compute_snr_input(x_true: np.ndarray, y: np.ndarray, f_true: np.ndarray) -> float:
    """
    Compute input SNR.
    
    The noise is defined as y - (x_true + f_true).
    
    SNR_in = 10 * log10( ||x_true||^2 / ||y - (x_true + f_true)||^2 )
    
    Args:
        x_true: Ground truth sparse signal
        y: Observed noisy signal
        f_true: Ground truth baseline
        
    Returns:
        Input SNR in dB
    """
    signal_power = np.sum(x_true ** 2)
    noise = y - (x_true + f_true)
    noise_power = np.sum(noise ** 2)
    
    if noise_power < 1e-12:
        return np.inf
    
    return 10 * np.log10(signal_power / noise_power)


def compute_snr_output(x_true: np.ndarray, x_est: np.ndarray) -> float:
    """
    Compute output SNR (after estimation).
    
    SNR_out = 10 * log10( ||x_true||^2 / ||x_true - x_est||^2 )
    
    Args:
        x_true: Ground truth sparse signal
        x_est: Estimated sparse signal
        
    Returns:
        Output SNR in dB
    """
    signal_power = np.sum(x_true ** 2)
    error_power = np.sum((x_true - x_est) ** 2)
    
    if error_power < 1e-12:
        return np.inf
    
    return 10 * np.log10(signal_power / error_power)


def compute_delta_snr(snr_in: float, snr_out: float) -> float:
    """
    Compute SNR improvement.
    
    ΔSNR = SNR_out - SNR_in
    
    Args:
        snr_in: Input SNR (dB)
        snr_out: Output SNR (dB)
        
    Returns:
        SNR improvement in dB
    """
    return snr_out - snr_in


def evaluate_single(x_true: np.ndarray, f_true: np.ndarray, y: np.ndarray,
                    x_est: np.ndarray, f_est: np.ndarray,
                    compute_additional: bool = True) -> EvaluationResult:
    """
    Evaluate estimation for a single signal.
    
    Args:
        x_true: Ground truth sparse signal
        f_true: Ground truth baseline
        y: Observed noisy signal
        x_est: Estimated sparse signal
        f_est: Estimated baseline
        compute_additional: Whether to compute additional metrics
        
    Returns:
        EvaluationResult containing all metrics
    """
    # Core metrics
    mse_signal = compute_mse(x_true, x_est)
    mse_baseline = compute_mse(f_true, f_est)
    snr_in = compute_snr_input(x_true, y, f_true)
    snr_out = compute_snr_output(x_true, x_est)
    delta_snr = compute_delta_snr(snr_in, snr_out)
    
    result = EvaluationResult(
        mse_signal=mse_signal,
        mse_baseline=mse_baseline,
        snr_in=snr_in,
        snr_out=snr_out,
        delta_snr=delta_snr
    )
    
    # Additional metrics
    if compute_additional:
        result.mae_signal = compute_mae(x_true, x_est)
        result.mae_baseline = compute_mae(f_true, f_est)
    
    return result


def aggregate_results(results: List[EvaluationResult]) -> Dict[str, Tuple[float, float]]:
    """
    Aggregate evaluation results across multiple signals.
    
    Computes mean ± std for each metric.
    
    Args:
        results: List of EvaluationResult objects
        
    Returns:
        Dictionary mapping metric name to (mean, std) tuple
    """
    metrics = {
        'mse_signal': [],
        'mse_baseline': [],
        'snr_in': [],
        'snr_out': [],
        'delta_snr': [],
        'mae_signal': [],
        'mae_baseline': []
    }
    
    for r in results:
        metrics['mse_signal'].append(r.mse_signal)
        metrics['mse_baseline'].append(r.mse_baseline)
        metrics['snr_in'].append(r.snr_in)
        metrics['snr_out'].append(r.snr_out)
        metrics['delta_snr'].append(r.delta_snr)
        if r.mae_signal is not None:
            metrics['mae_signal'].append(r.mae_signal)
        if r.mae_baseline is not None:
            metrics['mae_baseline'].append(r.mae_baseline)
    
    aggregated = {}
    for name, values in metrics.items():
        if len(values) > 0:
            arr = np.array(values)
            # Handle inf values
            finite_mask = np.isfinite(arr)
            if np.any(finite_mask):
                aggregated[name] = (np.mean(arr[finite_mask]), np.std(arr[finite_mask]))
            else:
                aggregated[name] = (np.nan, np.nan)
    
    return aggregated


def format_table_row(method_name: str, aggregated: Dict[str, Tuple[float, float]]) -> str:
    """
    Format a table row for display.
    
    Args:
        method_name: Name of the method
        aggregated: Aggregated results from aggregate_results()
        
    Returns:
        Formatted table row string
    """
    mse_sig_mean, mse_sig_std = aggregated['mse_signal']
    mse_base_mean, mse_base_std = aggregated['mse_baseline']
    dsnr_mean, dsnr_std = aggregated['delta_snr']
    
    return (f"{method_name:<15} | "
            f"{mse_sig_mean:.4f} ± {mse_sig_std:.4f} | "
            f"{mse_base_mean:.4f} ± {mse_base_std:.4f} | "
            f"{dsnr_mean:.2f} ± {dsnr_std:.2f}")


def print_table1(beads_results: List[EvaluationResult],
                 lbeads_results: List[EvaluationResult],
                 title: str = "Table 1: Comparison Results"):
    """
    Print Table 1 in thesis format.
    
    Args:
        beads_results: Results from classical BEADS
        lbeads_results: Results from LBEADS-NET
        title: Table title
    """
    beads_agg = aggregate_results(beads_results)
    lbeads_agg = aggregate_results(lbeads_results)
    
    print("\n" + "=" * 75)
    print(title)
    print("=" * 75)
    print(f"{'Method':<15} | {'MSE (signal) ↓':<20} | {'MSE (baseline) ↓':<20} | {'ΔSNR (dB) ↑':<15}")
    print("-" * 75)
    print(format_table_row("BEADS", beads_agg))
    print(format_table_row("LBEADS-NET", lbeads_agg))
    print("=" * 75)
    
    # Also show percentage improvement
    mse_sig_improvement = (beads_agg['mse_signal'][0] - lbeads_agg['mse_signal'][0]) / beads_agg['mse_signal'][0] * 100
    mse_base_improvement = (beads_agg['mse_baseline'][0] - lbeads_agg['mse_baseline'][0]) / beads_agg['mse_baseline'][0] * 100
    dsnr_improvement = lbeads_agg['delta_snr'][0] - beads_agg['delta_snr'][0]
    
    print("\nRelative Improvement (LBEADS-NET vs BEADS):")
    print(f"  MSE (signal):   {mse_sig_improvement:+.1f}% {'(better)' if mse_sig_improvement > 0 else '(worse)'}")
    print(f"  MSE (baseline): {mse_base_improvement:+.1f}% {'(better)' if mse_base_improvement > 0 else '(worse)'}")
    print(f"  ΔSNR:           {dsnr_improvement:+.2f} dB {'(better)' if dsnr_improvement > 0 else '(worse)'}")


def generate_latex_table(beads_results: List[EvaluationResult],
                         lbeads_results: List[EvaluationResult]) -> str:
    """
    Generate LaTeX code for Table 1.
    
    Args:
        beads_results: Results from classical BEADS
        lbeads_results: Results from LBEADS-NET
        
    Returns:
        LaTeX table code as string
    """
    beads_agg = aggregate_results(beads_results)
    lbeads_agg = aggregate_results(lbeads_results)
    
    latex = r"""
\begin{table}[htbp]
\centering
\caption{Comparison of classical BEADS and LBEADS-NET on synthetic signals with known ground truth.}
\label{tab:comparison}
\begin{tabular}{lccc}
\toprule
\textbf{Method} & \textbf{MSE (signal) $\downarrow$} & \textbf{MSE (baseline) $\downarrow$} & \textbf{$\Delta$SNR (dB) $\uparrow$} \\
\midrule
"""
    
    # BEADS row
    latex += f"BEADS & ${beads_agg['mse_signal'][0]:.4f} \\pm {beads_agg['mse_signal'][1]:.4f}$ & "
    latex += f"${beads_agg['mse_baseline'][0]:.4f} \\pm {beads_agg['mse_baseline'][1]:.4f}$ & "
    latex += f"${beads_agg['delta_snr'][0]:.2f} \\pm {beads_agg['delta_snr'][1]:.2f}$ \\\\\n"
    
    # LBEADS-NET row
    latex += f"LBEADS-NET & ${lbeads_agg['mse_signal'][0]:.4f} \\pm {lbeads_agg['mse_signal'][1]:.4f}$ & "
    latex += f"${lbeads_agg['mse_baseline'][0]:.4f} \\pm {lbeads_agg['mse_baseline'][1]:.4f}$ & "
    latex += f"${lbeads_agg['delta_snr'][0]:.2f} \\pm {lbeads_agg['delta_snr'][1]:.2f}$ \\\\\n"
    
    latex += r"""
\bottomrule
\end{tabular}
\end{table}
"""
    
    return latex


if __name__ == "__main__":
    # Test metrics with dummy data
    print("Testing Metrics Module")
    print("=" * 50)
    
    np.random.seed(42)
    N = 100
    
    # Create dummy ground truth
    x_true = np.zeros(N)
    x_true[30:40] = 1.0  # Peak
    x_true[60:70] = 0.5  # Smaller peak
    
    f_true = np.linspace(0, 0.2, N)  # Linear baseline
    
    noise = np.random.normal(0, 0.05, N)
    y = x_true + f_true + noise
    
    # Create "estimated" versions (with some error)
    x_est = x_true + np.random.normal(0, 0.02, N)
    f_est = f_true + np.random.normal(0, 0.01, N)
    
    # Evaluate
    result = evaluate_single(x_true, f_true, y, x_est, f_est)
    
    print(f"MSE (signal):   {result.mse_signal:.6f}")
    print(f"MSE (baseline): {result.mse_baseline:.6f}")
    print(f"SNR_in:         {result.snr_in:.2f} dB")
    print(f"SNR_out:        {result.snr_out:.2f} dB")
    print(f"ΔSNR:           {result.delta_snr:.2f} dB")
    
    # Test aggregation
    results = [result for _ in range(10)]
    aggregated = aggregate_results(results)
    
    print("\nAggregated Results:")
    for name, (mean, std) in aggregated.items():
        print(f"  {name}: {mean:.4f} ± {std:.4f}")
    
    print("\nTest complete!")
