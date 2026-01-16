"""
Run Experiments: BEADS vs LBEADS-NET Comparison

This is the main script that orchestrates the synthetic data experiments
for comparing classical BEADS with LBEADS-NET.

Generates:
- Table 1: Quantitative comparison (MSE, ΔSNR)
- Figure 1: Visual comparison on representative signals

Usage:
    python run_experiments.py

Author: Thesis Work
Date: January 2026
"""

import os
import sys
import time
import numpy as np
import torch
import argparse
from typing import List, Tuple, Dict
from pathlib import Path

# Add parent directories to path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, 'LBEADS_NETv1'))
sys.path.insert(0, os.path.join(parent_dir, 'BEADS', 'Replicate'))

# Import local modules
from synthetic_data_generator import SyntheticDataGenerator, SyntheticSignal, save_dataset, load_dataset
from metrics import (EvaluationResult, evaluate_single, aggregate_results, 
                     print_table1, generate_latex_table)
from visualization import (plot_figure1_thesis, plot_metrics_summary, 
                           plot_boxplot_comparison, plot_single_comparison, PlotResult)

# Import BEADS implementations
from beads import beads as original_beads
from lbeads_net import LBEADS_NET, LBEADS_NET_Fast


# ============================================================================
# CONFIGURATION
# ============================================================================

class ExperimentConfig:
    """Configuration for the experiments."""
    
    # Signal parameters
    N: int = 1024                    # Signal length
    SEED: int = 42                   # Random seed for reproducibility
    
    # Dataset parameters
    N_SAMPLES: int = 30              # Total number of test signals
    N_EASY: int = 10                 # Low noise signals
    N_MEDIUM: int = 10               # Medium noise signals
    N_HARD: int = 10                 # High noise signals
    
    # BEADS parameters (tuned for synthetic data with wide Gaussian peaks)
    BEADS_D: int = 1                 # Filter order
    BEADS_FC: float = 0.002          # Filter cutoff frequency (lower for wider peaks)
    BEADS_R: float = 6.0             # Asymmetry ratio
    BEADS_LAM0: float = 0.1          # Asymmetric penalty weight (lower = more signal)
    BEADS_LAM1: float = 0.5          # First derivative penalty (lower = less smoothing)
    BEADS_LAM2: float = 0.5          # Second derivative penalty
    BEADS_NIT: int = 50              # Number of iterations (more for convergence)
    
    # LBEADS-NET parameters
    LBEADS_NUM_LAYERS: int = 10      # Number of unrolled layers
    LBEADS_SHARED_PARAMS: bool = False  # Layer-wise params for learning
    # Default to trained model if it exists
    LBEADS_MODEL_PATH: str = os.path.join(script_dir, 'trained_models', 'lbeads_net_trained.pth')
    
    # Output settings
    OUTPUT_DIR: str = os.path.join(script_dir, 'results')
    SAVE_DATASET: bool = True
    
    @classmethod
    def print_config(cls):
        """Print current configuration."""
        print("\n" + "=" * 60)
        print("EXPERIMENT CONFIGURATION")
        print("=" * 60)
        print(f"Signal length N: {cls.N}")
        print(f"Random seed: {cls.SEED}")
        print(f"Number of samples: {cls.N_SAMPLES}")
        print(f"  - Easy: {cls.N_EASY}")
        print(f"  - Medium: {cls.N_MEDIUM}")
        print(f"  - Hard: {cls.N_HARD}")
        print(f"\nBEADS Parameters:")
        print(f"  d={cls.BEADS_D}, fc={cls.BEADS_FC}, r={cls.BEADS_R}")
        print(f"  λ₀={cls.BEADS_LAM0}, λ₁={cls.BEADS_LAM1}, λ₂={cls.BEADS_LAM2}")
        print(f"  Nit={cls.BEADS_NIT}")
        print(f"\nLBEADS-NET Parameters:")
        print(f"  num_layers={cls.LBEADS_NUM_LAYERS}")
        print(f"  shared_params={cls.LBEADS_SHARED_PARAMS}")
        print(f"  model_path={cls.LBEADS_MODEL_PATH or 'None (using init)'}")
        print("=" * 60)


# ============================================================================
# EXPERIMENT FUNCTIONS
# ============================================================================

def run_beads(y: np.ndarray, config: ExperimentConfig = ExperimentConfig) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run classical BEADS on a signal.
    
    Args:
        y: Input signal
        config: Experiment configuration
        
    Returns:
        x_est: Estimated sparse signal
        f_est: Estimated baseline
    """
    x, f, _ = original_beads(
        y=y,
        d=config.BEADS_D,
        fc=config.BEADS_FC,
        r=config.BEADS_R,
        lam0=config.BEADS_LAM0,
        lam1=config.BEADS_LAM1,
        lam2=config.BEADS_LAM2,
        Nit=config.BEADS_NIT
    )
    
    # Convert to numpy if needed
    if isinstance(x, torch.Tensor):
        x = x.numpy()
    if isinstance(f, torch.Tensor):
        f = f.numpy()
    
    return x, f


def create_lbeads_model(N: int, config: ExperimentConfig = ExperimentConfig):
    """
    Create LBEADS-NET model.
    
    Args:
        N: Signal length
        config: Experiment configuration
        
    Returns:
        LBEADS_NET_Fast model (trainable, uses ISTA-style updates)
    """
    # Check if we have a trained model
    if config.LBEADS_MODEL_PATH and os.path.exists(config.LBEADS_MODEL_PATH):
        checkpoint = torch.load(config.LBEADS_MODEL_PATH, weights_only=False)
        saved_config = checkpoint.get('config', {})
        
        model = LBEADS_NET_Fast(
            N=N,
            d=saved_config.get('d', config.BEADS_D),
            fc=saved_config.get('fc', config.BEADS_FC),
            num_layers=saved_config.get('num_layers', config.LBEADS_NUM_LAYERS),
            init_lam0=config.BEADS_LAM0,
            init_lam1=config.BEADS_LAM1,
            init_lam2=config.BEADS_LAM2,
            init_r=config.BEADS_R,
            init_step_size=0.1
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded trained LBEADS-NET from {config.LBEADS_MODEL_PATH}")
        print(f"  Layers: {saved_config.get('num_layers', 'N/A')}")
        print(f"  Val loss: {checkpoint.get('val_loss', 'N/A'):.4f}" if checkpoint.get('val_loss') else "")
    else:
        # Use untrained LBEADS_NET_Fast with good initial parameters
        model = LBEADS_NET_Fast(
            N=N,
            d=config.BEADS_D,
            fc=config.BEADS_FC,
            num_layers=config.LBEADS_NUM_LAYERS,
            init_lam0=config.BEADS_LAM0,
            init_lam1=config.BEADS_LAM1,
            init_lam2=config.BEADS_LAM2,
            init_r=config.BEADS_R,
            init_step_size=0.1
        )
        print("Using untrained LBEADS-NET (ISTA-style)")
        print(f"  Layers: {config.LBEADS_NUM_LAYERS}")
        print(f"  Init params: lam0={config.BEADS_LAM0}, lam1={config.BEADS_LAM1}, lam2={config.BEADS_LAM2}")
    
    model.eval()
    return model


def run_lbeads(y: np.ndarray, model: LBEADS_NET) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run LBEADS-NET on a signal.
    
    Args:
        y: Input signal
        model: LBEADS-NET model
        
    Returns:
        x_est: Estimated sparse signal
        f_est: Estimated baseline
    """
    y_tensor = torch.tensor(y, dtype=torch.float64)
    
    with torch.no_grad():
        x, f = model(y_tensor)
    
    return x.numpy(), f.numpy()


def generate_test_dataset(config: ExperimentConfig = ExperimentConfig) -> List[SyntheticSignal]:
    """
    Generate the test dataset with stratified difficulty levels.
    
    Args:
        config: Experiment configuration
        
    Returns:
        List of SyntheticSignal objects
    """
    print("\nGenerating synthetic test dataset...")
    
    generator = SyntheticDataGenerator(N=config.N, seed=config.SEED)
    stratified = generator.generate_stratified_dataset(
        n_easy=config.N_EASY,
        n_medium=config.N_MEDIUM,
        n_hard=config.N_HARD
    )
    
    # Combine all signals into one list
    dataset = []
    for category, signals in stratified.items():
        for signal in signals:
            # Add category to metadata
            signal.metadata['category'] = category
            dataset.append(signal)
    
    print(f"  Generated {len(dataset)} signals total")
    print(f"    Easy (low noise):     {config.N_EASY}")
    print(f"    Medium (moderate):    {config.N_MEDIUM}")
    print(f"    Hard (high noise):    {config.N_HARD}")
    
    # Count noise types
    gaussian_count = sum(1 for s in dataset if s.noise_type == 'gaussian')
    laplacian_count = sum(1 for s in dataset if s.noise_type == 'laplacian')
    print(f"    Gaussian noise:       {gaussian_count}")
    print(f"    Laplacian noise:      {laplacian_count}")
    
    return dataset


def run_all_experiments(dataset: List[SyntheticSignal], 
                        config: ExperimentConfig = ExperimentConfig
                        ) -> Tuple[List[EvaluationResult], List[EvaluationResult], 
                                   List[Tuple], List[Tuple]]:
    """
    Run both BEADS and LBEADS-NET on all signals.
    
    Args:
        dataset: List of SyntheticSignal objects
        config: Experiment configuration
        
    Returns:
        beads_results: List of EvaluationResult for BEADS
        lbeads_results: List of EvaluationResult for LBEADS-NET
        beads_outputs: List of (x_beads, f_beads) tuples
        lbeads_outputs: List of (x_lbeads, f_lbeads) tuples
    """
    print("\n" + "=" * 60)
    print("RUNNING EXPERIMENTS")
    print("=" * 60)
    
    # Create LBEADS-NET model
    N = config.N
    lbeads_model = create_lbeads_model(N, config)
    
    beads_results = []
    lbeads_results = []
    beads_outputs = []
    lbeads_outputs = []
    
    beads_total_time = 0
    lbeads_total_time = 0
    
    for i, signal in enumerate(dataset):
        print(f"\rProcessing signal {i+1}/{len(dataset)}...", end="", flush=True)
        
        y = signal.y
        x_true = signal.x_true
        f_true = signal.f_true
        
        # Run BEADS
        start = time.time()
        x_beads, f_beads = run_beads(y, config)
        beads_total_time += time.time() - start
        
        # Run LBEADS-NET
        start = time.time()
        x_lbeads, f_lbeads = run_lbeads(y, lbeads_model)
        lbeads_total_time += time.time() - start
        
        # Evaluate BEADS
        beads_eval = evaluate_single(x_true, f_true, y, x_beads, f_beads)
        beads_results.append(beads_eval)
        beads_outputs.append((x_beads, f_beads))
        
        # Evaluate LBEADS-NET
        lbeads_eval = evaluate_single(x_true, f_true, y, x_lbeads, f_lbeads)
        lbeads_results.append(lbeads_eval)
        lbeads_outputs.append((x_lbeads, f_lbeads))
    
    print("\n")
    print(f"BEADS total time:      {beads_total_time:.2f}s ({beads_total_time/len(dataset)*1000:.1f}ms/signal)")
    print(f"LBEADS-NET total time: {lbeads_total_time:.2f}s ({lbeads_total_time/len(dataset)*1000:.1f}ms/signal)")
    
    return beads_results, lbeads_results, beads_outputs, lbeads_outputs


def select_representative_signals(dataset: List[SyntheticSignal],
                                   beads_results: List[EvaluationResult],
                                   lbeads_results: List[EvaluationResult]
                                   ) -> List[int]:
    """
    Select representative signals for Figure 1.
    
    Selects:
    - 1 easy case (low noise, Gaussian)
    - 1 medium case (moderate noise, Gaussian)
    - 1 hard case (high noise, Gaussian)
    - 1 Laplacian noise case
    
    Args:
        dataset: All test signals
        beads_results: BEADS evaluation results
        lbeads_results: LBEADS-NET evaluation results
        
    Returns:
        List of indices for representative signals
    """
    selected = []
    
    # Find signals by category
    easy_indices = [i for i, s in enumerate(dataset) 
                    if s.metadata.get('category') == 'easy']
    medium_indices = [i for i, s in enumerate(dataset) 
                      if s.metadata.get('category') == 'medium']
    hard_indices = [i for i, s in enumerate(dataset) 
                    if s.metadata.get('category') == 'hard']
    laplacian_indices = [i for i, s in enumerate(dataset) 
                         if s.noise_type == 'laplacian']
    
    # Select one from each category (pick median performance)
    for indices, name in [(easy_indices, 'easy'), 
                          (medium_indices, 'medium'),
                          (hard_indices, 'hard')]:
        if indices:
            # Sort by LBEADS improvement over BEADS
            improvements = [(i, lbeads_results[i].delta_snr - beads_results[i].delta_snr) 
                           for i in indices]
            improvements.sort(key=lambda x: x[1])
            # Pick median
            selected.append(improvements[len(improvements)//2][0])
    
    # Add a Laplacian case if not already included
    laplacian_not_selected = [i for i in laplacian_indices if i not in selected]
    if laplacian_not_selected:
        selected.append(laplacian_not_selected[0])
    
    return selected


def save_results(beads_results: List[EvaluationResult],
                 lbeads_results: List[EvaluationResult],
                 output_dir: str):
    """Save results to files."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save aggregated results
    beads_agg = aggregate_results(beads_results)
    lbeads_agg = aggregate_results(lbeads_results)
    
    results_file = os.path.join(output_dir, 'results_summary.txt')
    with open(results_file, 'w') as f:
        f.write("LBEADS-NET vs BEADS Comparison Results\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("BEADS Results:\n")
        for name, (mean, std) in beads_agg.items():
            f.write(f"  {name}: {mean:.6f} ± {std:.6f}\n")
        
        f.write("\nLBEADS-NET Results:\n")
        for name, (mean, std) in lbeads_agg.items():
            f.write(f"  {name}: {mean:.6f} ± {std:.6f}\n")
    
    print(f"Saved results summary to {results_file}")
    
    # Save LaTeX table
    latex_file = os.path.join(output_dir, 'table1.tex')
    latex_code = generate_latex_table(beads_results, lbeads_results)
    with open(latex_file, 'w') as f:
        f.write(latex_code)
    print(f"Saved LaTeX table to {latex_file}")
    
    # Save raw results as numpy
    raw_file = os.path.join(output_dir, 'raw_results.npz')
    np.savez(
        raw_file,
        beads_mse_signal=[r.mse_signal for r in beads_results],
        beads_mse_baseline=[r.mse_baseline for r in beads_results],
        beads_delta_snr=[r.delta_snr for r in beads_results],
        lbeads_mse_signal=[r.mse_signal for r in lbeads_results],
        lbeads_mse_baseline=[r.mse_baseline for r in lbeads_results],
        lbeads_delta_snr=[r.delta_snr for r in lbeads_results]
    )
    print(f"Saved raw results to {raw_file}")


def main():
    """Main experiment runner."""
    
    parser = argparse.ArgumentParser(description='Run BEADS vs LBEADS-NET comparison')
    parser.add_argument('--n-samples', type=int, default=30, 
                        help='Number of test signals')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip generating plots')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to trained LBEADS-NET model (default: use trained_models/lbeads_net_trained.pth)')
    parser.add_argument('--no-trained-model', action='store_true',
                        help='Use untrained LBEADS-NET (for comparison/debugging)')
    args = parser.parse_args()
    
    # Update config
    ExperimentConfig.N_SAMPLES = args.n_samples
    ExperimentConfig.N_EASY = args.n_samples // 3
    ExperimentConfig.N_MEDIUM = args.n_samples // 3
    ExperimentConfig.N_HARD = args.n_samples - 2 * (args.n_samples // 3)
    ExperimentConfig.SEED = args.seed
    
    # Handle model path: use provided path, or default, or None if explicitly disabled
    if args.no_trained_model:
        ExperimentConfig.LBEADS_MODEL_PATH = None
    elif args.model_path is not None:
        ExperimentConfig.LBEADS_MODEL_PATH = args.model_path
    # else: keep the default from ExperimentConfig (trained_models/lbeads_net_trained.pth)
    
    # Print header
    print("\n" + "=" * 60)
    print("LBEADS-NET vs BEADS: Synthetic Data Comparison")
    print("DAY 1-2: Baseline Comparison Experiments")
    print("=" * 60)
    
    ExperimentConfig.print_config()
    
    # Create output directory
    os.makedirs(ExperimentConfig.OUTPUT_DIR, exist_ok=True)
    
    # Step 1: Generate dataset
    dataset = generate_test_dataset(ExperimentConfig)
    
    if ExperimentConfig.SAVE_DATASET:
        dataset_path = os.path.join(ExperimentConfig.OUTPUT_DIR, 'test_dataset.npz')
        save_dataset(dataset, dataset_path)
    
    # Step 2: Run experiments
    beads_results, lbeads_results, beads_outputs, lbeads_outputs = \
        run_all_experiments(dataset, ExperimentConfig)
    
    # Step 3: Print Table 1
    print_table1(beads_results, lbeads_results, 
                 title="TABLE 1: Comparison on Synthetic Data")
    
    # Step 4: Save results
    save_results(beads_results, lbeads_results, ExperimentConfig.OUTPUT_DIR)
    
    # Step 5: Generate plots (if not disabled)
    if not args.no_plots:
        print("\n" + "=" * 60)
        print("GENERATING FIGURES")
        print("=" * 60)
        
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
        
        # Select representative signals for Figure 1
        selected_indices = select_representative_signals(
            dataset, beads_results, lbeads_results
        )
        
        selected_signals = [dataset[i] for i in selected_indices]
        selected_beads = [beads_outputs[i] for i in selected_indices]
        selected_lbeads = [lbeads_outputs[i] for i in selected_indices]
        
        # Create case labels
        case_labels = []
        for i in selected_indices:
            signal = dataset[i]
            category = signal.metadata.get('category', 'unknown')
            noise = signal.noise_type
            level = signal.noise_level
            case_labels.append(f"{category.title()} ({noise}, σ={level:.2f})")
        
        # Generate Figure 1
        fig1_path = os.path.join(ExperimentConfig.OUTPUT_DIR, 'figure1_comparison.png')
        plot_figure1_thesis(
            selected_signals, selected_beads, selected_lbeads, case_labels,
            save_path=fig1_path
        )
        plt.close()
        
        # Generate metrics summary bar chart
        metrics_path = os.path.join(ExperimentConfig.OUTPUT_DIR, 'metrics_summary.png')
        plot_metrics_summary(beads_results, lbeads_results, save_path=metrics_path)
        plt.close()
        
        # Generate boxplots
        for metric in ['mse_signal', 'mse_baseline', 'delta_snr']:
            boxplot_path = os.path.join(ExperimentConfig.OUTPUT_DIR, f'boxplot_{metric}.png')
            plot_boxplot_comparison(beads_results, lbeads_results, 
                                    metric=metric, save_path=boxplot_path)
            plt.close()
        
        print(f"\nAll figures saved to {ExperimentConfig.OUTPUT_DIR}")
    
    # Print final summary
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    
    beads_agg = aggregate_results(beads_results)
    lbeads_agg = aggregate_results(lbeads_results)
    
    print("\nKey Findings:")
    mse_improvement = (beads_agg['mse_signal'][0] - lbeads_agg['mse_signal'][0]) / beads_agg['mse_signal'][0] * 100
    dsnr_improvement = lbeads_agg['delta_snr'][0] - beads_agg['delta_snr'][0]
    
    if mse_improvement > 0:
        print(f"  ✓ LBEADS-NET reduces signal MSE by {mse_improvement:.1f}%")
    else:
        print(f"  ✗ LBEADS-NET increases signal MSE by {-mse_improvement:.1f}%")
    
    if dsnr_improvement > 0:
        print(f"  ✓ LBEADS-NET improves ΔSNR by {dsnr_improvement:.2f} dB")
    else:
        print(f"  ✗ LBEADS-NET decreases ΔSNR by {-dsnr_improvement:.2f} dB")
    
    print(f"\nResults saved to: {ExperimentConfig.OUTPUT_DIR}")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
