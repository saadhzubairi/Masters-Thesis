"""
Run Experiments v2: BEADS vs LBEADS-NET v2 Comparison

This is the main script for comparing classical BEADS with LBEADS-NET v2.

Key differences from v1:
- Uses LBEADS-NET v2 with improved architecture
- SNR-based evaluation metrics
- Better visualization
- Support for warm-start initialization

Generates:
- Table 1: Quantitative comparison (MSE, SNR, Peak Error)
- Figure 1: Visual comparison on representative signals

Usage:
    python run_experiments_v2.py
    python run_experiments_v2.py --model-version v2
    python run_experiments_v2.py --train-first

Author: Thesis Work
Date: January 2026
"""

import os
import sys
import time
import numpy as np
import torch
import argparse
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import matplotlib.pyplot as plt

# Add parent directories to path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, '1. LBEADS_NETv1'))
sys.path.insert(0, os.path.join(parent_dir, '0. BEADS', 'Replicate'))

# Import local modules
from synthetic_data_generator import SyntheticDataGenerator, SyntheticSignal
from metrics import EvaluationResult, evaluate_single, aggregate_results, generate_latex_table

# Import BEADS implementations
from beads import beads as original_beads
from lbeads_net import LBEADS_NET_Fast
from lbeads_net_v2 import LBEADS_NET_v2, create_lbeads_net_v2, beads_warm_start
from losses_v2 import compute_snr, compute_rmse, compute_peak_error


# ============================================================================
# CONFIGURATION
# ============================================================================

class ExperimentConfigV2:
    """Configuration for the v2 experiments."""
    
    # Signal parameters
    N: int = 1024
    SEED: int = 42
    
    # Dataset parameters
    N_SAMPLES: int = 50              # More test samples
    N_EASY: int = 15
    N_MEDIUM: int = 20
    N_HARD: int = 15
    
    # BEADS parameters (tuned for synthetic data)
    BEADS_D: int = 1
    BEADS_FC: float = 0.006          # Standard cutoff
    BEADS_R: float = 6.0
    BEADS_LAM0: float = 0.5          # Standard BEADS params
    BEADS_LAM1: float = 4.0
    BEADS_LAM2: float = 4.0
    BEADS_NIT: int = 50
    
    # LBEADS-NET v1 model path
    LBEADS_V1_MODEL_PATH: str = os.path.join(script_dir, 'trained_models', 'lbeads_net_trained.pth')
    
    # LBEADS-NET v2 model path  
    LBEADS_V2_MODEL_PATH: str = os.path.join(script_dir, 'trained_models', 'lbeads_net_v2_trained.pth')
    
    # Which model version to use ('v1', 'v2', or 'both')
    MODEL_VERSION: str = 'v2'
    
    # Output settings
    OUTPUT_DIR: str = os.path.join(script_dir, 'results_v2')
    SAVE_DATASET: bool = True


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_lbeads_v1(N: int, config: ExperimentConfigV2):
    """Load LBEADS-NET v1 model."""
    from lbeads_net import LBEADS_NET_Fast
    
    if os.path.exists(config.LBEADS_V1_MODEL_PATH):
        checkpoint = torch.load(config.LBEADS_V1_MODEL_PATH, weights_only=False)
        saved_config = checkpoint.get('config', {})
        
        model = LBEADS_NET_Fast(
            N=N,
            d=saved_config.get('d', config.BEADS_D),
            fc=saved_config.get('fc', config.BEADS_FC),
            num_layers=saved_config.get('num_layers', 10),
            init_lam0=0.1,
            init_lam1=0.5,
            init_lam2=0.5,
            init_r=6.0,
            init_step_size=0.1
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded LBEADS-NET v1 from {config.LBEADS_V1_MODEL_PATH}")
        print(f"  Layers: {saved_config.get('num_layers', 'N/A')}")
    else:
        # Use untrained model
        model = LBEADS_NET_Fast(
            N=N, d=config.BEADS_D, fc=config.BEADS_FC,
            num_layers=10, init_step_size=0.1
        )
        print("Using untrained LBEADS-NET v1")
    
    model.eval()
    return model


def load_lbeads_v2(N: int, config: ExperimentConfigV2):
    """Load LBEADS-NET v2 model."""
    
    if os.path.exists(config.LBEADS_V2_MODEL_PATH):
        checkpoint = torch.load(config.LBEADS_V2_MODEL_PATH, weights_only=False)
        saved_config = checkpoint.get('config', {})
        
        model = LBEADS_NET_v2(
            N=N,
            d=saved_config.get('d', config.BEADS_D),
            fc=saved_config.get('fc', config.BEADS_FC),
            num_layers=saved_config.get('num_layers', 20),
            use_momentum=saved_config.get('use_momentum', True),
            use_skip_connection=saved_config.get('use_skip_connection', True),
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"Loaded LBEADS-NET v2 from {config.LBEADS_V2_MODEL_PATH}")
        print(f"  Layers: {saved_config.get('num_layers', 'N/A')}")
        print(f"  Val SNR: {checkpoint.get('val_snr', 'N/A'):.2f} dB" 
              if checkpoint.get('val_snr') else "")
    else:
        # Use untrained model with default preset
        model = create_lbeads_net_v2(N=N, preset='default', d=config.BEADS_D, fc=config.BEADS_FC)
        print("Using untrained LBEADS-NET v2 (default preset)")
        print(f"  Layers: {model.num_layers}")
    
    model.eval()
    return model


# ============================================================================
# EXPERIMENT FUNCTIONS
# ============================================================================

def run_beads(y: np.ndarray, config: ExperimentConfigV2) -> Tuple[np.ndarray, np.ndarray]:
    """Run classical BEADS on a signal."""
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
    
    if isinstance(x, torch.Tensor):
        x = x.numpy()
    if isinstance(f, torch.Tensor):
        f = f.numpy()
    
    return x, f


def run_lbeads(y: np.ndarray, model, use_warmstart: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Run LBEADS-NET on a signal."""
    y_tensor = torch.tensor(y, dtype=torch.float64)
    
    with torch.no_grad():
        if use_warmstart and hasattr(model, 'net'):
            x, f = model(y_tensor, use_warmstart=True)
        else:
            x, f = model(y_tensor)
    
    return x.numpy(), f.numpy()


def generate_test_dataset(config: ExperimentConfigV2) -> List[SyntheticSignal]:
    """Generate the test dataset."""
    print("\nGenerating synthetic test dataset...")
    
    generator = SyntheticDataGenerator(N=config.N, seed=config.SEED)
    
    dataset = []
    
    # Easy signals (low noise)
    for i in range(config.N_EASY):
        noise_level = np.random.uniform(0.02, 0.05)
        signal = generator.generate_signal(
            peak_params={'amplitude_range': (1.0, 2.0)},
            noise_type='gaussian',
            noise_level=noise_level
        )
        signal.metadata['category'] = 'easy'
        dataset.append(signal)
    
    # Medium signals
    for i in range(config.N_MEDIUM):
        noise_level = np.random.uniform(0.05, 0.12)
        noise_type = 'gaussian' if i % 2 == 0 else 'laplacian'
        signal = generator.generate_signal(
            peak_params={'amplitude_range': (0.8, 1.8)},
            noise_type=noise_type,
            noise_level=noise_level
        )
        signal.metadata['category'] = 'medium'
        dataset.append(signal)
    
    # Hard signals (high noise, challenging baselines)
    for i in range(config.N_HARD):
        noise_level = np.random.uniform(0.12, 0.20)
        noise_type = 'laplacian' if i % 2 == 0 else 'gaussian'
        signal = generator.generate_signal(
            peak_params={'amplitude_range': (0.5, 1.5)},
            noise_type=noise_type,
            noise_level=noise_level
        )
        signal.metadata['category'] = 'hard'
        dataset.append(signal)
    
    print(f"  Generated {len(dataset)} signals")
    print(f"    Easy: {config.N_EASY}, Medium: {config.N_MEDIUM}, Hard: {config.N_HARD}")
    
    return dataset


def evaluate_results(
    x_true: np.ndarray,
    f_true: np.ndarray,
    x_est: np.ndarray,
    f_est: np.ndarray
) -> Dict[str, float]:
    """Evaluate estimation results."""
    
    x_true_t = torch.tensor(x_true, dtype=torch.float64)
    x_est_t = torch.tensor(x_est, dtype=torch.float64)
    f_true_t = torch.tensor(f_true, dtype=torch.float64)
    f_est_t = torch.tensor(f_est, dtype=torch.float64)
    
    return {
        'mse_signal': float(np.mean((x_true - x_est) ** 2)),
        'mse_baseline': float(np.mean((f_true - f_est) ** 2)),
        'snr': compute_snr(x_true_t, x_est_t),
        'rmse_signal': compute_rmse(x_true_t, x_est_t),
        'rmse_baseline': compute_rmse(f_true_t, f_est_t),
        'peak_error': compute_peak_error(x_true_t, x_est_t)
    }


def run_experiments(
    dataset: List[SyntheticSignal],
    config: ExperimentConfigV2
) -> Dict:
    """Run all experiments."""
    
    print("\n" + "=" * 70)
    print("RUNNING EXPERIMENTS")
    print("=" * 70)
    
    N = config.N
    results = {
        'beads': {'metrics': [], 'outputs': [], 'times': []},
    }
    
    # Load models based on version
    if config.MODEL_VERSION in ['v1', 'both']:
        lbeads_v1 = load_lbeads_v1(N, config)
        results['lbeads_v1'] = {'metrics': [], 'outputs': [], 'times': []}
    
    if config.MODEL_VERSION in ['v2', 'both']:
        lbeads_v2 = load_lbeads_v2(N, config)
        results['lbeads_v2'] = {'metrics': [], 'outputs': [], 'times': []}
    
    print(f"\nProcessing {len(dataset)} signals...")
    
    for i, signal in enumerate(dataset):
        if (i + 1) % 10 == 0:
            print(f"  Processing signal {i+1}/{len(dataset)}...")
        
        y = signal.y
        x_true = signal.x_true
        f_true = signal.f_true
        
        # Run BEADS
        start = time.time()
        x_beads, f_beads = run_beads(y, config)
        beads_time = time.time() - start
        
        results['beads']['metrics'].append(evaluate_results(x_true, f_true, x_beads, f_beads))
        results['beads']['outputs'].append((x_beads, f_beads))
        results['beads']['times'].append(beads_time)
        
        # Run LBEADS v1
        if 'lbeads_v1' in results:
            start = time.time()
            x_v1, f_v1 = run_lbeads(y, lbeads_v1)
            v1_time = time.time() - start
            
            results['lbeads_v1']['metrics'].append(evaluate_results(x_true, f_true, x_v1, f_v1))
            results['lbeads_v1']['outputs'].append((x_v1, f_v1))
            results['lbeads_v1']['times'].append(v1_time)
        
        # Run LBEADS v2
        if 'lbeads_v2' in results:
            start = time.time()
            x_v2, f_v2 = run_lbeads(y, lbeads_v2)
            v2_time = time.time() - start
            
            results['lbeads_v2']['metrics'].append(evaluate_results(x_true, f_true, x_v2, f_v2))
            results['lbeads_v2']['outputs'].append((x_v2, f_v2))
            results['lbeads_v2']['times'].append(v2_time)
    
    return results


def print_results_table(results: Dict, dataset: List[SyntheticSignal]):
    """Print results summary table."""
    
    print("\n" + "=" * 90)
    print("RESULTS SUMMARY")
    print("=" * 90)
    
    # Get categories
    categories = ['all', 'easy', 'medium', 'hard']
    
    def get_category_indices(cat):
        if cat == 'all':
            return list(range(len(dataset)))
        return [i for i, s in enumerate(dataset) if s.metadata.get('category') == cat]
    
    # Print header
    print(f"\n{'Method':<15} {'Category':<10} {'SNR (dB)':<12} {'MSE (x)':<12} {'RMSE':<12} {'Peak Err':<12} {'Time (ms)':<12}")
    print("-" * 90)
    
    for method in results.keys():
        for cat in categories:
            indices = get_category_indices(cat)
            if not indices:
                continue
            
            metrics = [results[method]['metrics'][i] for i in indices]
            times = [results[method]['times'][i] for i in indices]
            
            snr = np.mean([m['snr'] for m in metrics])
            mse = np.mean([m['mse_signal'] for m in metrics])
            rmse = np.mean([m['rmse_signal'] for m in metrics])
            peak_err = np.mean([m['peak_error'] for m in metrics])
            avg_time = np.mean(times) * 1000  # Convert to ms
            
            method_name = method.upper().replace('_', ' ')
            print(f"{method_name:<15} {cat:<10} {snr:>10.2f}  {mse:>10.6f}  {rmse:>10.4f}  {peak_err:>10.4f}  {avg_time:>10.2f}")
        
        print("-" * 90)


def plot_comparison_figure(
    dataset: List[SyntheticSignal],
    results: Dict,
    output_dir: str,
    n_examples: int = 2
):
    """Plot comparison figure."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Select representative examples (one easy, one hard)
    easy_idx = next((i for i, s in enumerate(dataset) if s.metadata.get('category') == 'easy'), 0)
    hard_idx = next((i for i, s in enumerate(dataset) if s.metadata.get('category') == 'hard'), len(dataset)-1)
    
    examples = [easy_idx, hard_idx]
    example_names = ['Easy (gaussian, low noise)', 'Hard (laplacian, high noise)']
    
    n_methods = len(results)
    fig, axes = plt.subplots(len(examples), 3, figsize=(16, 4 * len(examples)))
    
    if len(examples) == 1:
        axes = axes.reshape(1, -1)
    
    for row, (idx, name) in enumerate(zip(examples, example_names)):
        signal = dataset[idx]
        y = signal.y
        x_true = signal.x_true
        f_true = signal.f_true
        noise_level = signal.noise_level
        
        # Column 1: Baseline estimation
        ax = axes[row, 0]
        ax.plot(y, 'gray', alpha=0.5, linewidth=0.5, label='y (observed)')
        ax.plot(f_true, 'k--', linewidth=2, label='f_true')
        
        colors = {'beads': 'red', 'lbeads_v1': 'blue', 'lbeads_v2': 'green'}
        
        for method, data in results.items():
            _, f_est = data['outputs'][idx]
            color = colors.get(method, 'purple')
            label = method.upper().replace('_', '-')
            ax.plot(f_est, color=color, linewidth=1.5, label=label)
        
        ax.set_title(f'Baseline Estimation\n{name}')
        ax.set_xlabel('Sample')
        ax.set_ylabel('Amplitude')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Column 2: Signal recovery
        ax = axes[row, 1]
        ax.plot(x_true, 'k--', linewidth=2, label='x_true')
        
        for method, data in results.items():
            x_est, _ = data['outputs'][idx]
            color = colors.get(method, 'purple')
            label = method.upper().replace('_', '-')
            ax.plot(x_est, color=color, linewidth=1.5, label=label)
        
        ax.set_title(f'Signal Recovery')
        ax.set_xlabel('Sample')
        ax.set_ylabel('Amplitude')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Column 3: Estimation error
        ax = axes[row, 2]
        
        for method, data in results.items():
            x_est, _ = data['outputs'][idx]
            error = x_true - x_est
            color = colors.get(method, 'purple')
            label = method.upper().replace('_', '-')
            mse = results[method]['metrics'][idx]['mse_signal']
            ax.plot(error, color=color, linewidth=1, alpha=0.8, label=f'{label} (MSE={mse:.4f})')
        
        ax.axhline(0, color='k', linestyle='--', linewidth=0.5)
        ax.set_title('Estimation Error (x_true - x_est)')
        ax.set_xlabel('Sample')
        ax.set_ylabel('Error')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'figure1_comparison_v2.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved comparison figure to {save_path}")
    plt.close()


def generate_latex_table_v2(results: Dict, dataset: List[SyntheticSignal], output_path: str):
    """Generate LaTeX table for the paper."""
    
    categories = ['easy', 'medium', 'hard', 'all']
    
    def get_category_indices(cat):
        if cat == 'all':
            return list(range(len(dataset)))
        return [i for i, s in enumerate(dataset) if s.metadata.get('category') == cat]
    
    latex = []
    latex.append("\\begin{table}[htbp]")
    latex.append("\\centering")
    latex.append("\\caption{Quantitative comparison of BEADS and LBEADS-NET methods}")
    latex.append("\\label{tab:results}")
    latex.append("\\begin{tabular}{llcccc}")
    latex.append("\\toprule")
    latex.append("Category & Method & SNR (dB) & MSE ($\\times 10^{-3}$) & RMSE & Peak Error \\\\")
    latex.append("\\midrule")
    
    for cat in categories:
        indices = get_category_indices(cat)
        if not indices:
            continue
        
        first_method = True
        for method in results.keys():
            metrics = [results[method]['metrics'][i] for i in indices]
            
            snr = np.mean([m['snr'] for m in metrics])
            mse = np.mean([m['mse_signal'] for m in metrics]) * 1000
            rmse = np.mean([m['rmse_signal'] for m in metrics])
            peak_err = np.mean([m['peak_error'] for m in metrics])
            
            cat_name = cat.capitalize() if first_method else ""
            method_name = method.upper().replace('_', '-')
            
            latex.append(f"{cat_name} & {method_name} & {snr:.2f} & {mse:.2f} & {rmse:.4f} & {peak_err:.4f} \\\\")
            first_method = False
        
        if cat != 'all':
            latex.append("\\midrule")
    
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(latex))
    
    print(f"Saved LaTeX table to {output_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Run BEADS vs LBEADS-NET v2 experiments')
    
    parser.add_argument('--model-version', type=str, default='v2',
                        choices=['v1', 'v2', 'both'],
                        help='Which LBEADS-NET version to use')
    parser.add_argument('--n-samples', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--train-first', action='store_true',
                        help='Train v2 model before experiments')
    
    args = parser.parse_args()
    
    # Update config
    config = ExperimentConfigV2()
    config.MODEL_VERSION = args.model_version
    config.SEED = args.seed
    
    if args.output_dir:
        config.OUTPUT_DIR = args.output_dir
    
    # Optionally train v2 model first
    if args.train_first:
        print("\n" + "=" * 70)
        print("TRAINING LBEADS-NET v2 FIRST")
        print("=" * 70)
        from train_lbeads_v2 import train_v2, TrainingConfigV2
        train_config = TrainingConfigV2()
        train_v2(train_config)
    
    # Generate dataset
    dataset = generate_test_dataset(config)
    
    # Run experiments
    results = run_experiments(dataset, config)
    
    # Print results
    print_results_table(results, dataset)
    
    # Create output directory
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # Generate plots
    plot_comparison_figure(dataset, results, config.OUTPUT_DIR)
    
    # Generate LaTeX table
    latex_path = os.path.join(config.OUTPUT_DIR, 'table1_v2.tex')
    generate_latex_table_v2(results, dataset, latex_path)
    
    print("\n" + "=" * 70)
    print("EXPERIMENTS COMPLETE")
    print("=" * 70)
    print(f"Results saved to: {config.OUTPUT_DIR}")


if __name__ == "__main__":
    main()
