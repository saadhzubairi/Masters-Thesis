"""
Experiment Worker for LBEADS-NET Hyperparameter Tuning
=======================================================

This worker:
1. Takes a hyperparameter grid configuration
2. Generates all experiment combinations
3. Trains a model for each combination
4. Saves results, plots, metrics, and configs to individual folders

Usage:
    python worker.py --grid gaussian --max-experiments 10
    python worker.py --grid quick
    python worker.py --grid full --start-from 5
"""

import os
import sys
import json
import time
import shutil
import argparse
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, StepLR

# Add parent paths
sys.path.insert(0, str(Path(__file__).parent.parent / "LBEADS_NETv1"))
sys.path.insert(0, str(Path(__file__).parent.parent / "synthetic_data_test"))

from config import (
    ExperimentConfig, 
    generate_experiment_configs, 
    count_experiments,
    AVAILABLE_GRIDS
)

# =============================================================================
# LBEADS-NET MODEL (Modified to accept config)
# =============================================================================

def gaussian_kernel_smooth(signal: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
    """Apply Gaussian smoothing for baseline estimation."""
    device = signal.device
    x = torch.arange(kernel_size, device=device, dtype=signal.dtype) - kernel_size // 2
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel = kernel / kernel.sum()
    
    if signal.dim() == 1:
        signal = signal.unsqueeze(0).unsqueeze(0)
        squeeze_back = True
    elif signal.dim() == 2:
        signal = signal.unsqueeze(1)
        squeeze_back = False
    else:
        squeeze_back = False
    
    pad_size = kernel_size // 2
    padded = torch.nn.functional.pad(signal, (pad_size, pad_size), mode='reflect')
    smoothed = torch.nn.functional.conv1d(padded, kernel.unsqueeze(0).unsqueeze(0))
    
    if squeeze_back:
        return smoothed.squeeze(0).squeeze(0)
    return smoothed.squeeze(1)


class ConfigurableLBEADS_NET(nn.Module):
    """LBEADS-NET with configurable hyperparameters."""
    
    def __init__(self, config: ExperimentConfig):
        super().__init__()
        self.config = config
        self.num_layers = config.num_layers
        self.kernel_size = config.kernel_size
        self.sigma = config.sigma
        self.baseline_method = getattr(config, 'baseline_method', 'gaussian')
        self.ema_alpha = getattr(config, 'ema_alpha', 0.02)
        
        # Learnable parameters per layer
        self.lam0 = nn.Parameter(torch.ones(self.num_layers) * config.lam0_init)
        self.lam1 = nn.Parameter(torch.ones(self.num_layers) * config.lam1_init)
        self.lam2 = nn.Parameter(torch.ones(self.num_layers) * config.lam2_init)
        self.r = nn.Parameter(torch.ones(self.num_layers) * config.r_init)
        self.step_size = nn.Parameter(torch.ones(self.num_layers) * config.step_size_init)
    
    def soft_threshold(self, x: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
        """Soft thresholding operator."""
        return torch.sign(x) * torch.clamp(torch.abs(x) - lam, min=0)
    
    def asymmetric_penalty_grad(self, r: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        """Gradient of asymmetric penalty (Huber-like for positives)."""
        grad = torch.zeros_like(residual)
        pos_mask = residual > 0
        neg_mask = residual <= 0
        grad[pos_mask] = r * torch.sign(residual[pos_mask])
        grad[neg_mask] = torch.sign(residual[neg_mask])
        return grad
    
    def exponential_smooth(self, signal: torch.Tensor) -> torch.Tensor:
        """Apply bidirectional exponential moving average for baseline estimation."""
        alpha = self.ema_alpha
        
        if signal.dim() == 1:
            signal = signal.unsqueeze(0)
            squeeze_back = True
        else:
            squeeze_back = False
        
        # Forward pass
        f = torch.zeros_like(signal)
        f[:, 0] = signal[:, 0]
        for i in range(1, signal.shape[1]):
            f[:, i] = alpha * signal[:, i] + (1 - alpha) * f[:, i-1]
        
        # Backward pass (zero-phase filtering)
        f_back = torch.zeros_like(f)
        f_back[:, -1] = f[:, -1]
        for i in range(signal.shape[1] - 2, -1, -1):
            f_back[:, i] = alpha * f[:, i] + (1 - alpha) * f_back[:, i+1]
        
        if squeeze_back:
            return f_back.squeeze(0)
        return f_back
    
    def smooth_baseline(self, signal: torch.Tensor) -> torch.Tensor:
        """Apply chosen baseline smoothing method."""
        if self.baseline_method == 'exponential':
            return self.exponential_smooth(signal)
        else:  # gaussian
            return gaussian_kernel_smooth(signal, self.kernel_size, self.sigma)
    
    def forward(self, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through unrolled BEADS.
        
        Args:
            y: Noisy signal [batch, N] or [N]
            
        Returns:
            x: Recovered sparse signal
            f: Estimated baseline
        """
        if y.dim() == 1:
            y = y.unsqueeze(0)
            single_sample = True
        else:
            single_sample = False
        
        batch_size, N = y.shape
        device = y.device
        
        # Initialize estimates
        x = torch.zeros_like(y)
        f = self.smooth_baseline(y)
        
        # Unrolled iterations
        for layer in range(self.num_layers):
            lam0 = torch.clamp(self.lam0[layer], min=0.01)
            r = torch.clamp(self.r[layer], min=1.01)
            step = torch.clamp(self.step_size[layer], min=0.001, max=0.5)
            
            # Compute residual
            residual = y - x - f
            
            # Gradient step for x
            grad_x = -residual + self.asymmetric_penalty_grad(r, residual)
            x_new = x - step * grad_x
            
            # Soft threshold for sparsity
            x = self.soft_threshold(x_new, lam0 * step)
            
            # Update baseline estimate
            residual = y - x
            f = self.smooth_baseline(residual)
        
        if single_sample:
            return x.squeeze(0), f.squeeze(0)
        return x, f
    
    def get_learned_params(self) -> Dict:
        """Return learned parameters."""
        return {
            'lam0': self.lam0.detach().cpu().numpy().tolist(),
            'lam1': self.lam1.detach().cpu().numpy().tolist(),
            'lam2': self.lam2.detach().cpu().numpy().tolist(),
            'r': self.r.detach().cpu().numpy().tolist(),
            'step_size': self.step_size.detach().cpu().numpy().tolist(),
        }


# =============================================================================
# DATA GENERATION
# =============================================================================

class SyntheticDataGenerator:
    """Generate synthetic chromatogram-like signals."""
    
    def __init__(self, config: ExperimentConfig, seed: int = 42):
        self.config = config
        self.N = config.signal_length
        self.rng = np.random.RandomState(seed)
    
    def generate_baseline(self) -> np.ndarray:
        """Generate smooth polynomial + sinusoidal baseline."""
        t = np.linspace(0, 1, self.N)
        
        # Polynomial component
        degree = self.rng.randint(2, 4)
        coeffs = self.rng.randn(degree + 1) * 0.5
        baseline = np.polyval(coeffs, t)
        
        # Sinusoidal component
        n_sins = self.rng.randint(1, 3)
        for _ in range(n_sins):
            freq = self.rng.uniform(0.5, 3)
            amp = self.rng.uniform(0.1, 0.5)
            phase = self.rng.uniform(0, 2*np.pi)
            baseline += amp * np.sin(2*np.pi*freq*t + phase)
        
        return baseline
    
    def generate_peaks(self) -> np.ndarray:
        """Generate sparse Gaussian peaks."""
        t = np.arange(self.N)
        peaks = np.zeros(self.N)
        
        n_peaks = self.rng.randint(3, 8)
        for _ in range(n_peaks):
            center = self.rng.randint(50, self.N - 50)
            sigma = self.rng.uniform(10, 30)
            amplitude = self.rng.uniform(0.5, 2.0)
            peaks += amplitude * np.exp(-0.5 * ((t - center) / sigma) ** 2)
        
        return peaks
    
    def generate_noise(self, noise_type: str = 'gaussian') -> np.ndarray:
        """Generate noise."""
        noise_std = self.rng.uniform(self.config.noise_std_min, self.config.noise_std_max)
        
        if noise_type == 'gaussian':
            return self.rng.randn(self.N) * noise_std
        elif noise_type == 'laplacian':
            return self.rng.laplace(0, noise_std / np.sqrt(2), self.N)
        else:
            return self.rng.randn(self.N) * noise_std
    
    def generate_sample(self, noise_type: str = 'gaussian'):
        """Generate one complete sample."""
        baseline = self.generate_baseline()
        peaks = self.generate_peaks()
        noise = self.generate_noise(noise_type)
        
        signal = baseline + peaks + noise
        
        return {
            'signal': signal.astype(np.float32),
            'baseline': baseline.astype(np.float32),
            'peaks': peaks.astype(np.float32),
            'noise': noise.astype(np.float32),
        }
    
    def generate_dataset(self, n_samples: int) -> Dict:
        """Generate multiple samples."""
        signals = []
        baselines = []
        peaks_list = []
        
        for i in range(n_samples):
            noise_type = 'gaussian' if i % 5 != 0 else 'laplacian'
            sample = self.generate_sample(noise_type)
            signals.append(sample['signal'])
            baselines.append(sample['baseline'])
            peaks_list.append(sample['peaks'])
        
        return {
            'signals': np.stack(signals),
            'baselines': np.stack(baselines),
            'peaks': np.stack(peaks_list),
        }


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def create_optimizer(model: nn.Module, config: ExperimentConfig):
    """Create optimizer based on config."""
    params = model.parameters()
    
    if config.optimizer == "adam":
        return optim.Adam(params, lr=config.learning_rate, weight_decay=config.weight_decay)
    elif config.optimizer == "adamw":
        return optim.AdamW(params, lr=config.learning_rate, weight_decay=config.weight_decay)
    elif config.optimizer == "sgd":
        return optim.SGD(params, lr=config.learning_rate, weight_decay=config.weight_decay, momentum=0.9)
    elif config.optimizer == "rmsprop":
        return optim.RMSprop(params, lr=config.learning_rate, weight_decay=config.weight_decay)
    else:
        return optim.Adam(params, lr=config.learning_rate, weight_decay=config.weight_decay)


def create_scheduler(optimizer, config: ExperimentConfig):
    """Create learning rate scheduler based on config."""
    if config.scheduler == "plateau":
        return ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    elif config.scheduler == "cosine":
        return CosineAnnealingLR(optimizer, T_max=config.epochs)
    elif config.scheduler == "step":
        return StepLR(optimizer, step_size=15, gamma=0.5)
    else:
        return None


def train_epoch(model, dataloader, optimizer, config, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    n_batches = 0
    
    for y, x_true, f_true in dataloader:
        y = y.to(device)
        x_true = x_true.to(device)
        f_true = f_true.to(device)
        
        optimizer.zero_grad()
        
        x_pred, f_pred = model(y)
        
        # Combined loss
        signal_loss = nn.functional.mse_loss(x_pred, x_true)
        baseline_loss = nn.functional.mse_loss(f_pred, f_true)
        loss = config.signal_weight * signal_loss + config.baseline_weight * baseline_loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / n_batches


def validate(model, dataloader, config, device):
    """Validate model."""
    model.eval()
    total_loss = 0
    n_batches = 0
    
    with torch.no_grad():
        for y, x_true, f_true in dataloader:
            y = y.to(device)
            x_true = x_true.to(device)
            f_true = f_true.to(device)
            
            x_pred, f_pred = model(y)
            
            signal_loss = nn.functional.mse_loss(x_pred, x_true)
            baseline_loss = nn.functional.mse_loss(f_pred, f_true)
            loss = config.signal_weight * signal_loss + config.baseline_weight * baseline_loss
            
            total_loss += loss.item()
            n_batches += 1
    
    return total_loss / n_batches


def train_model(config: ExperimentConfig, device: str = 'cuda') -> Tuple[nn.Module, Dict]:
    """
    Train a model with given configuration.
    
    Returns:
        model: Trained model
        history: Training history
    """
    # Generate data
    train_gen = SyntheticDataGenerator(config, seed=42)
    val_gen = SyntheticDataGenerator(config, seed=123)
    
    train_data = train_gen.generate_dataset(config.n_train)
    val_data = val_gen.generate_dataset(config.n_val)
    
    # Create dataloaders
    train_dataset = TensorDataset(
        torch.tensor(train_data['signals']),
        torch.tensor(train_data['peaks']),
        torch.tensor(train_data['baselines'])
    )
    val_dataset = TensorDataset(
        torch.tensor(val_data['signals']),
        torch.tensor(val_data['peaks']),
        torch.tensor(val_data['baselines'])
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    # Create model
    model = ConfigurableLBEADS_NET(config).to(device)
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)
    
    # Training loop
    history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rates': [],
    }
    
    best_val_loss = float('inf')
    best_state = None
    
    for epoch in range(config.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, config, device)
        val_loss = validate(model, val_loader, config, device)
        
        current_lr = optimizer.param_groups[0]['lr']
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['learning_rates'].append(current_lr)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()
        
        # Update scheduler
        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
    
    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)
    
    history['best_val_loss'] = best_val_loss
    history['final_params'] = model.get_learned_params()
    
    return model, history


# =============================================================================
# EVALUATION AND VISUALIZATION
# =============================================================================

def evaluate_model(model, config: ExperimentConfig, device: str = 'cuda', n_test: int = 30) -> Dict:
    """Evaluate model on test data."""
    test_gen = SyntheticDataGenerator(config, seed=999)
    test_data = test_gen.generate_dataset(n_test)
    
    model.eval()
    
    mse_signal_list = []
    mse_baseline_list = []
    delta_snr_list = []
    
    with torch.no_grad():
        for i in range(n_test):
            y = torch.tensor(test_data['signals'][i]).to(device)
            x_true = test_data['peaks'][i]
            f_true = test_data['baselines'][i]
            
            x_pred, f_pred = model(y)
            x_pred = x_pred.cpu().numpy()
            f_pred = f_pred.cpu().numpy()
            
            # Compute metrics
            mse_signal = np.mean((x_pred - x_true) ** 2)
            mse_baseline = np.mean((f_pred - f_true) ** 2)
            
            # SNR computation
            noise_in = test_data['signals'][i] - x_true - f_true
            noise_out = x_pred - x_true
            
            snr_in = 10 * np.log10(np.var(x_true) / (np.var(noise_in) + 1e-10))
            snr_out = 10 * np.log10(np.var(x_true) / (np.var(noise_out) + 1e-10))
            delta_snr = snr_out - snr_in
            
            mse_signal_list.append(mse_signal)
            mse_baseline_list.append(mse_baseline)
            delta_snr_list.append(delta_snr)
    
    return {
        'mse_signal_mean': float(np.mean(mse_signal_list)),
        'mse_signal_std': float(np.std(mse_signal_list)),
        'mse_baseline_mean': float(np.mean(mse_baseline_list)),
        'mse_baseline_std': float(np.std(mse_baseline_list)),
        'delta_snr_mean': float(np.mean(delta_snr_list)),
        'delta_snr_std': float(np.std(delta_snr_list)),
        'mse_signal_all': [float(x) for x in mse_signal_list],
        'mse_baseline_all': [float(x) for x in mse_baseline_list],
        'delta_snr_all': [float(x) for x in delta_snr_list],
    }


def generate_comparison_plots(model, config: ExperimentConfig, output_dir: Path, device: str = 'cuda'):
    """Generate comparison plots for this experiment."""
    import matplotlib.pyplot as plt
    
    # Generate a few test signals
    test_gen = SyntheticDataGenerator(config, seed=777)
    
    fig, axes = plt.subplots(4, 3, figsize=(15, 12))
    
    model.eval()
    with torch.no_grad():
        for row in range(4):
            sample = test_gen.generate_sample()
            y = torch.tensor(sample['signal']).to(device)
            
            x_pred, f_pred = model(y)
            x_pred = x_pred.cpu().numpy()
            f_pred = f_pred.cpu().numpy()
            
            # Original signal
            axes[row, 0].plot(sample['signal'], 'b-', alpha=0.7, label='Noisy')
            axes[row, 0].plot(sample['baseline'], 'g--', lw=2, label='True baseline')
            axes[row, 0].set_title(f'Original Signal {row+1}')
            if row == 0:
                axes[row, 0].legend(fontsize=8)
            
            # Recovered signal
            axes[row, 1].plot(sample['peaks'], 'g-', lw=2, label='True peaks')
            axes[row, 1].plot(x_pred, 'r--', lw=1.5, label='LBEADS-NET')
            axes[row, 1].set_title('Signal Recovery')
            if row == 0:
                axes[row, 1].legend(fontsize=8)
            
            # Baseline estimation
            axes[row, 2].plot(sample['baseline'], 'g-', lw=2, label='True baseline')
            axes[row, 2].plot(f_pred, 'r--', lw=1.5, label='LBEADS-NET')
            axes[row, 2].set_title('Baseline Estimation')
            if row == 0:
                axes[row, 2].legend(fontsize=8)
    
    plt.suptitle(f"Experiment: {config.get_short_name()}", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_grid.png', dpi=150, bbox_inches='tight')
    plt.close()


def generate_training_plot(history: Dict, output_dir: Path):
    """Generate training history plot."""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss plot
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train')
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Validation')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training History')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Learning rate plot
    axes[1].plot(epochs, history['learning_rates'], 'g-')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Learning Rate')
    axes[1].set_title('Learning Rate Schedule')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_history.png', dpi=150, bbox_inches='tight')
    plt.close()


def generate_metrics_plot(metrics: Dict, output_dir: Path):
    """Generate metrics box plots."""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    axes[0].boxplot(metrics['mse_signal_all'])
    axes[0].set_title(f"MSE (Signal)\n{metrics['mse_signal_mean']:.4f} ± {metrics['mse_signal_std']:.4f}")
    axes[0].set_ylabel('MSE')
    
    axes[1].boxplot(metrics['mse_baseline_all'])
    axes[1].set_title(f"MSE (Baseline)\n{metrics['mse_baseline_mean']:.4f} ± {metrics['mse_baseline_std']:.4f}")
    axes[1].set_ylabel('MSE')
    
    axes[2].boxplot(metrics['delta_snr_all'])
    axes[2].set_title(f"ΔSNR (dB)\n{metrics['delta_snr_mean']:.2f} ± {metrics['delta_snr_std']:.2f}")
    axes[2].set_ylabel('dB')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_boxplot.png', dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

def run_single_experiment(
    config: ExperimentConfig,
    output_base_dir: Path,
    device: str = 'cuda',
    verbose: bool = True
) -> Dict:
    """
    Run a single experiment with given configuration.
    
    Args:
        config: Experiment configuration
        output_base_dir: Base directory for all experiments
        device: Device to use for training
        verbose: Print progress
        
    Returns:
        Summary dictionary
    """
    start_time = time.time()
    
    # Create experiment folder
    folder_name = f"{config.experiment_id}_{config.get_short_name()}"
    exp_dir = output_base_dir / folder_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"EXPERIMENT: {config.experiment_id}")
        print(f"{'='*60}")
        print(f"Folder: {folder_name}")
        print(f"Config: layers={config.num_layers}, kernel={config.kernel_size}, sigma={config.sigma}")
        print(f"        lr={config.learning_rate}, batch={config.batch_size}, opt={config.optimizer}")
    
    # Save config
    config.to_json(str(exp_dir / 'config.json'))
    
    try:
        # Train model
        if verbose:
            print(f"\nTraining...")
        model, history = train_model(config, device)
        
        # Save model
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config.to_dict(),
            'history': history,
        }, exp_dir / 'model.pth')
        
        # Evaluate
        if verbose:
            print(f"Evaluating...")
        metrics = evaluate_model(model, config, device)
        
        # Generate plots
        if verbose:
            print(f"Generating plots...")
        generate_comparison_plots(model, config, exp_dir, device)
        generate_training_plot(history, exp_dir)
        generate_metrics_plot(metrics, exp_dir)
        
        # Save metrics
        with open(exp_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        elapsed = time.time() - start_time
        
        # Create summary
        summary = {
            'experiment_id': config.experiment_id,
            'folder': folder_name,
            'status': 'success',
            'elapsed_time': elapsed,
            'best_val_loss': history['best_val_loss'],
            'mse_signal': metrics['mse_signal_mean'],
            'mse_baseline': metrics['mse_baseline_mean'],
            'delta_snr': metrics['delta_snr_mean'],
            'config': config.to_dict(),
        }
        
        # Save summary
        with open(exp_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        if verbose:
            print(f"\n✓ Completed in {elapsed:.1f}s")
            print(f"  Val Loss: {history['best_val_loss']:.4f}")
            print(f"  MSE (signal): {metrics['mse_signal_mean']:.4f}")
            print(f"  MSE (baseline): {metrics['mse_baseline_mean']:.4f}")
            print(f"  ΔSNR: {metrics['delta_snr_mean']:.2f} dB")
        
        return summary
        
    except Exception as e:
        elapsed = time.time() - start_time
        error_summary = {
            'experiment_id': config.experiment_id,
            'folder': folder_name,
            'status': 'failed',
            'error': str(e),
            'elapsed_time': elapsed,
            'config': config.to_dict(),
        }
        
        with open(exp_dir / 'error.json', 'w') as f:
            json.dump(error_summary, f, indent=2)
        
        if verbose:
            print(f"\n✗ FAILED: {str(e)}")
        
        return error_summary


def run_all_experiments(
    grid_name: str = "quick",
    output_dir: str = None,
    max_experiments: int = None,
    start_from: int = 0,
    device: str = None,
):
    """
    Run all experiments for a given grid.
    
    Args:
        grid_name: Name of the hyperparameter grid to use
        output_dir: Output directory (default: experiments/{grid_name}_{timestamp})
        max_experiments: Maximum number of experiments to run
        start_from: Start from this experiment index (for resuming)
        device: Device to use (auto-detect if None)
    """
    # Get grid
    if grid_name not in AVAILABLE_GRIDS:
        print(f"Error: Unknown grid '{grid_name}'")
        print(f"Available grids: {list(AVAILABLE_GRIDS.keys())}")
        return
    
    grid = AVAILABLE_GRIDS[grid_name]
    
    # Generate configs
    configs = generate_experiment_configs(grid)
    total_experiments = len(configs)
    
    if max_experiments is not None:
        configs = configs[start_from:start_from + max_experiments]
    else:
        configs = configs[start_from:]
    
    # Setup output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(__file__).parent / "experiments" / f"{grid_name}_{timestamp}"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("=" * 70)
    print("LBEADS-NET HYPERPARAMETER EXPERIMENT WORKER")
    print("=" * 70)
    print(f"Grid: {grid_name}")
    print(f"Total experiments in grid: {total_experiments}")
    print(f"Running experiments: {start_from + 1} to {start_from + len(configs)}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {device}")
    print("=" * 70)
    
    # Save grid config
    with open(output_dir / 'grid_config.json', 'w') as f:
        json.dump({
            'grid_name': grid_name,
            'grid': {k: [float(v) if isinstance(v, (int, float)) else v for v in vals] 
                     for k, vals in grid.items()},
            'total_experiments': total_experiments,
            'start_from': start_from,
            'max_experiments': max_experiments,
            'device': device,
            'timestamp': datetime.now().isoformat(),
        }, f, indent=2)
    
    # Run experiments
    all_summaries = []
    
    for i, config in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] Running {config.experiment_id}...")
        
        summary = run_single_experiment(config, output_dir, device)
        all_summaries.append(summary)
        
        # Save running summary
        with open(output_dir / 'all_summaries.json', 'w') as f:
            json.dump(all_summaries, f, indent=2)
    
    # Generate final report
    generate_final_report(all_summaries, output_dir)
    
    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 70)
    print(f"Results saved to: {output_dir}")


def generate_final_report(summaries: List[Dict], output_dir: Path):
    """Generate final comparison report."""
    import matplotlib.pyplot as plt
    
    successful = [s for s in summaries if s['status'] == 'success']
    
    if not successful:
        print("No successful experiments to report.")
        return
    
    # Sort by MSE signal
    successful.sort(key=lambda x: x['mse_signal'])
    
    # Create report
    report_lines = [
        "=" * 70,
        "EXPERIMENT RESULTS SUMMARY",
        "=" * 70,
        f"Total experiments: {len(summaries)}",
        f"Successful: {len(successful)}",
        f"Failed: {len(summaries) - len(successful)}",
        "",
        "TOP 10 BY MSE (SIGNAL):",
        "-" * 70,
    ]
    
    for i, s in enumerate(successful[:10]):
        report_lines.append(
            f"{i+1}. {s['experiment_id']}: MSE={s['mse_signal']:.4f}, "
            f"dSNR={s['delta_snr']:.2f}dB, ValLoss={s['best_val_loss']:.4f}"
        )
    
    report_lines.extend([
        "",
        "BEST CONFIGURATION:",
        "-" * 70,
    ])
    
    best = successful[0]
    for key, val in best['config'].items():
        report_lines.append(f"  {key}: {val}")
    
    report = "\n".join(report_lines)
    
    with open(output_dir / 'final_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(report)
    
    # Create comparison plot
    if len(successful) > 1:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        exp_ids = [s['experiment_id'] for s in successful[:20]]
        mse_signals = [s['mse_signal'] for s in successful[:20]]
        mse_baselines = [s['mse_baseline'] for s in successful[:20]]
        delta_snrs = [s['delta_snr'] for s in successful[:20]]
        
        axes[0].barh(exp_ids, mse_signals, color='steelblue')
        axes[0].set_xlabel('MSE (Signal)')
        axes[0].set_title('MSE (Signal) - Lower is Better')
        axes[0].invert_yaxis()
        
        axes[1].barh(exp_ids, mse_baselines, color='seagreen')
        axes[1].set_xlabel('MSE (Baseline)')
        axes[1].set_title('MSE (Baseline) - Lower is Better')
        axes[1].invert_yaxis()
        
        axes[2].barh(exp_ids, delta_snrs, color='coral')
        axes[2].set_xlabel('ΔSNR (dB)')
        axes[2].set_title('ΔSNR - Higher is Better')
        axes[2].invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(output_dir / 'experiment_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='LBEADS-NET Hyperparameter Experiment Worker')
    parser.add_argument('--grid', type=str, default='quick',
                        choices=list(AVAILABLE_GRIDS.keys()),
                        help='Hyperparameter grid to use')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for experiments')
    parser.add_argument('--max-experiments', type=int, default=None,
                        help='Maximum number of experiments to run')
    parser.add_argument('--start-from', type=int, default=0,
                        help='Start from this experiment index')
    parser.add_argument('--device', type=str, default=None,
                        choices=['cuda', 'cpu'],
                        help='Device to use for training')
    parser.add_argument('--list-grids', action='store_true',
                        help='List available grids and exit')
    
    args = parser.parse_args()
    
    if args.list_grids:
        print("\nAvailable hyperparameter grids:")
        print("-" * 40)
        for name, grid in AVAILABLE_GRIDS.items():
            n_exp = count_experiments(grid)
            print(f"  {name}: {n_exp} experiments")
        return
    
    run_all_experiments(
        grid_name=args.grid,
        output_dir=args.output_dir,
        max_experiments=args.max_experiments,
        start_from=args.start_from,
        device=args.device,
    )


if __name__ == "__main__":
    main()
