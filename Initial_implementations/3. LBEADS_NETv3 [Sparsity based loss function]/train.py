"""
Training script for LBEADS-NET with Synthetic Data and Sparsity-Based Loss

This script trains the unrolled LBEADS-NET model on synthetic chromatogram data
with known ground truth peaks, baseline drift, and noise.

KEY CHANGE: Uses sparsity-promoting loss functions (L1, Total Variation)

Signal model:
    y = x_true (peaks) + f_true (baseline) + noise
    
Ground truth for training: x_true (the peaks)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass

from lbeads_net import LBEADS_NET, LBEADS_NET_Fast


# =============================================================================
# Synthetic Data Generation
# =============================================================================

@dataclass
class SyntheticSignal:
    """Container for a synthetic chromatogram signal."""
    y: np.ndarray           # Observed signal (peaks + baseline + noise)
    x_true: np.ndarray      # Ground truth peaks
    f_true: np.ndarray      # Ground truth baseline
    noise: np.ndarray       # Noise component
    metadata: Dict          # Generation parameters


class SyntheticDataGenerator:
    """
    Generate synthetic chromatogram-like data with:
    - Sparse Gaussian peaks
    - Smooth polynomial/sinusoidal baseline drift
    - Additive Gaussian noise
    """
    
    def __init__(self, N: int = 1024, seed: Optional[int] = None):
        """
        Args:
            N: Signal length
            seed: Random seed for reproducibility
        """
        self.N = N
        self.t = np.linspace(0.0, 1.0, N)
        self.rng = np.random.default_rng(seed)
    
    def generate_baseline(
        self,
        poly_degree_range: Tuple[int, int] = (2, 4),
        poly_coeff_range: Tuple[float, float] = (-1.0, 1.0),
        sine_freq_range: Tuple[float, float] = (0.3, 1.5),
        sine_amp_range: Tuple[float, float] = (0.5, 2.0),
    ) -> Tuple[np.ndarray, Dict]:
        """
        Generate smooth baseline drift using polynomial + sinusoidal components.
        Based on real chromatogram characteristics (baseline ~5-15 amplitude).
        """
        meta = {}
        
        # Polynomial component - slow drift
        degree = int(self.rng.integers(poly_degree_range[0], poly_degree_range[1] + 1))
        coeffs = self.rng.uniform(poly_coeff_range[0], poly_coeff_range[1], degree + 1)
        poly = np.zeros(self.N)
        for i, coeff in enumerate(coeffs):
            poly += coeff * (self.t ** i)
        meta["poly_degree"] = degree
        meta["poly_coeffs"] = coeffs.tolist()
        
        # Sinusoidal component - very slow oscillation
        freq = self.rng.uniform(sine_freq_range[0], sine_freq_range[1])
        amp = self.rng.uniform(sine_amp_range[0], sine_amp_range[1])
        phase = self.rng.uniform(0.0, 2.0 * np.pi)
        sine = amp * np.sin(2.0 * np.pi * freq * self.t + phase)
        meta["sine_freq"] = float(freq)
        meta["sine_amp"] = float(amp)
        meta["sine_phase"] = float(phase)
        
        baseline = poly + sine
        return baseline, meta
    
    def generate_peaks(
        self,
        num_peaks_range: Tuple[int, int] = (5, 15),
        center_margin: float = 0.05,
        width_range: Tuple[float, float] = (1.0, 4.0),  # SHARP peaks: 1-4 samples (real data: 1-7)
        amplitude_range: Tuple[float, float] = (10.0, 100.0),  # HIGH amplitude (real: 13x-215x baseline)
        negative_peak_prob: float = 0.0,  # Chromatogram peaks are always positive
    ) -> Tuple[np.ndarray, Dict]:
        """
        Generate sparse Gaussian peaks (chromatogram-like).
        
        Real chromatogram characteristics (from data analysis):
        - Peaks are VERY SHARP: width 1-7 samples for N=1024
        - Peaks are HIGH: 13x-215x the baseline level
        - Peaks are ALWAYS POSITIVE
        """
        num_peaks = int(self.rng.integers(num_peaks_range[0], num_peaks_range[1] + 1))
        
        x_true = np.zeros(self.N)
        peak_info: List[Dict] = []
        
        min_center = int(center_margin * self.N)
        max_center = int((1.0 - center_margin) * self.N)
        indices = np.arange(self.N)
        
        for _ in range(num_peaks):
            center = int(self.rng.integers(min_center, max_center))
            # Sharp peaks with sigma (width) of 1-4 samples
            width = float(self.rng.uniform(width_range[0], width_range[1]))
            # High amplitude - peaks should dominate over baseline
            amplitude = float(self.rng.uniform(amplitude_range[0], amplitude_range[1]))
            
            # Chromatogram peaks are always positive
            sign = 1.0
            
            peak = sign * amplitude * np.exp(-((indices - center) ** 2) / (2.0 * width ** 2))
            x_true += peak
            
            peak_info.append({
                "center": center,
                "width": width,
                "amplitude": sign * amplitude,
            })
        
        params = {
            "num_peaks": num_peaks,
            "peaks": peak_info,
        }
        
        return x_true, params
    
    def generate_noise(
        self, 
        noise_level: float = 1.0
    ) -> Tuple[np.ndarray, Dict]:
        """
        Generate additive Gaussian noise.
        Noise level is relative to baseline (typically 0.5-2.0).
        """
        noise = self.rng.normal(0.0, noise_level, self.N)
        params = {"noise_level": float(noise_level)}
        return noise, params
    
    def generate_signal(
        self,
        noise_level: float = 0.1,
        **kwargs
    ) -> SyntheticSignal:
        """
        Generate a complete synthetic signal.
        
        Args:
            noise_level: Standard deviation of Gaussian noise
            **kwargs: Additional parameters for baseline/peaks generation
        
        Returns:
            SyntheticSignal with y, x_true, f_true, noise, and metadata
        """
        f_true, baseline_meta = self.generate_baseline(**{k: v for k, v in kwargs.items() 
                                                          if k.startswith('poly') or k.startswith('sine')})
        x_true, peak_meta = self.generate_peaks(**{k: v for k, v in kwargs.items() 
                                                   if k.startswith('num_peaks') or k.startswith('center') 
                                                   or k.startswith('width') or k.startswith('amplitude')
                                                   or k.startswith('negative')})
        noise, noise_meta = self.generate_noise(noise_level)
        
        # Observed signal: peaks + baseline + noise
        y = x_true + f_true + noise
        
        metadata = {
            "N": self.N,
            "baseline": baseline_meta,
            "peaks": peak_meta,
            "noise": noise_meta,
        }
        
        return SyntheticSignal(
            y=y,
            x_true=x_true,
            f_true=f_true,
            noise=noise,
            metadata=metadata,
        )
    
    def generate_dataset(
        self,
        n_samples: int,
        noise_level_range: Tuple[float, float] = (0.5, 2.0),  # Realistic noise levels
        **kwargs
    ) -> List[SyntheticSignal]:
        """
        Generate a dataset of synthetic signals.
        
        Args:
            n_samples: Number of samples to generate
            noise_level_range: Range of noise levels to sample from
            **kwargs: Additional generation parameters
        
        Returns:
            List of SyntheticSignal objects
        """
        dataset = []
        for _ in range(n_samples):
            noise_level = float(self.rng.uniform(noise_level_range[0], noise_level_range[1]))
            signal = self.generate_signal(noise_level=noise_level, **kwargs)
            dataset.append(signal)
        return dataset


def create_train_test_split(
    dataset: List[SyntheticSignal],
    train_ratio: float = 0.8,
    seed: Optional[int] = None
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """
    Create 80/20 train/test split from synthetic dataset.
    
    Args:
        dataset: List of SyntheticSignal objects
        train_ratio: Fraction for training (default 0.8)
        seed: Random seed for shuffling
    
    Returns:
        (train_y, train_x_true), (test_y, test_x_true)
        where train_x_true is the ground truth peaks (target for learning)
    """
    rng = np.random.default_rng(seed)
    indices = np.arange(len(dataset))
    rng.shuffle(indices)
    
    split_idx = int(len(dataset) * train_ratio)
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    # Stack into arrays
    train_y = np.stack([dataset[i].y for i in train_indices])
    train_x_true = np.stack([dataset[i].x_true for i in train_indices])
    
    test_y = np.stack([dataset[i].y for i in test_indices])
    test_x_true = np.stack([dataset[i].x_true for i in test_indices])
    
    # Convert to tensors
    train_y_tensor = torch.tensor(train_y, dtype=torch.float64)
    train_x_tensor = torch.tensor(train_x_true, dtype=torch.float64)
    test_y_tensor = torch.tensor(test_y, dtype=torch.float64)
    test_x_tensor = torch.tensor(test_x_true, dtype=torch.float64)
    
    return (train_y_tensor, train_x_tensor), (test_y_tensor, test_x_tensor)


# =============================================================================
# Loss Function with Sparsity Penalties
# =============================================================================

class SparsityLoss(nn.Module):
    """
    Sparsity-promoting loss function for chromatogram peak recovery.
    
    Combines multiple loss terms:
    1. Reconstruction loss (MSE or Huber) - match ground truth peaks
    2. L1 sparsity on peaks - encourage most values to be zero
    3. Total Variation (TV) on peaks - encourage piecewise constant (sharp peaks)
    4. Baseline smoothness - penalize non-smooth baselines
    5. Non-negativity penalty - peaks should be positive
    
    The key insight: chromatogram peaks are:
    - SPARSE: Most of the signal is zero (only peaks have values)
    - POSITIVE: Peaks are always positive
    - SHARP: Peaks have steep edges (high gradient at boundaries, flat elsewhere)
    """
    
    def __init__(self, 
                 alpha_mse: float = 1.0,        # Reconstruction weight
                 alpha_l1: float = 0.01,        # L1 sparsity on peaks
                 alpha_tv: float = 0.01,        # Total Variation on peaks
                 alpha_smooth: float = 0.01,    # Baseline smoothness
                 alpha_neg: float = 0.1,        # Non-negativity penalty
                 use_huber: bool = True,        # Use Huber loss instead of MSE
                 huber_delta: float = 1.0):     # Huber delta parameter
        super(SparsityLoss, self).__init__()
        self.alpha_mse = alpha_mse
        self.alpha_l1 = alpha_l1
        self.alpha_tv = alpha_tv
        self.alpha_smooth = alpha_smooth
        self.alpha_neg = alpha_neg
        self.use_huber = use_huber
        self.huber_delta = huber_delta
        
        if use_huber:
            self.huber = nn.HuberLoss(reduction='mean', delta=huber_delta)
    
    def forward(self, x_pred: torch.Tensor, x_target: torch.Tensor,
                f_pred: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total loss with breakdown.
        
        Args:
            x_pred: Predicted peaks (batch, N) or (N,)
            x_target: Ground truth peaks (batch, N) or (N,)
            f_pred: Predicted baseline (optional)
            
        Returns:
            total_loss: Combined loss value
            loss_dict: Dictionary with individual loss components
        """
        # Ensure batch dimension
        if x_pred.dim() == 1:
            x_pred = x_pred.unsqueeze(0)
            x_target = x_target.unsqueeze(0)
        
        loss_dict = {}
        
        # 1. RECONSTRUCTION LOSS - match ground truth peaks
        if self.use_huber:
            recon_loss = self.huber(x_pred, x_target)
        else:
            recon_loss = torch.mean((x_pred - x_target) ** 2)
        loss_dict['reconstruction'] = recon_loss.item()
        
        total_loss = self.alpha_mse * recon_loss
        
        # 2. L1 SPARSITY LOSS - encourage peaks to be sparse
        # Most values should be zero; only peak locations should be non-zero
        if self.alpha_l1 > 0:
            l1_loss = torch.mean(torch.abs(x_pred))
            loss_dict['l1_sparsity'] = l1_loss.item()
            total_loss = total_loss + self.alpha_l1 * l1_loss
        
        # 3. TOTAL VARIATION (TV) LOSS - encourage piecewise constant
        # TV = sum of |x[i+1] - x[i]|
        # Low TV means signal is mostly flat with sharp transitions (peaks!)
        if self.alpha_tv > 0:
            diff1 = x_pred[:, 1:] - x_pred[:, :-1]
            tv_loss = torch.mean(torch.abs(diff1))
            loss_dict['total_variation'] = tv_loss.item()
            total_loss = total_loss + self.alpha_tv * tv_loss
        
        # 4. BASELINE SMOOTHNESS - penalize non-smooth baselines
        if self.alpha_smooth > 0 and f_pred is not None:
            if f_pred.dim() == 1:
                f_pred = f_pred.unsqueeze(0)
            # Second derivative penalty (curvature)
            diff2 = f_pred[:, 2:] - 2 * f_pred[:, 1:-1] + f_pred[:, :-2]
            smooth_loss = torch.mean(diff2 ** 2)
            loss_dict['baseline_smooth'] = smooth_loss.item()
            total_loss = total_loss + self.alpha_smooth * smooth_loss
        
        # 5. NON-NEGATIVITY PENALTY - peaks should be positive
        # Penalize negative values (chromatogram peaks are always positive)
        if self.alpha_neg > 0:
            neg_values = torch.clamp(-x_pred, min=0)  # Only negative parts
            neg_loss = torch.mean(neg_values ** 2)
            loss_dict['non_negativity'] = neg_loss.item()
            total_loss = total_loss + self.alpha_neg * neg_loss
        
        loss_dict['total'] = total_loss.item()
        
        return total_loss, loss_dict


# Legacy loss function for backward compatibility
class BEADSLoss(nn.Module):
    """
    Custom loss function for BEADS training (legacy).
    
    Combines:
    1. MSE between estimated signal (peaks) and target
    2. Optional: Smoothness regularization on baseline
    3. Optional: Sparsity regularization on signal
    """
    
    def __init__(self, alpha_mse: float = 1.0, 
                 alpha_smooth: float = 0.0,
                 alpha_sparse: float = 0.0):
        super(BEADSLoss, self).__init__()
        self.alpha_mse = alpha_mse
        self.alpha_smooth = alpha_smooth
        self.alpha_sparse = alpha_sparse
    
    def forward(self, x_pred: torch.Tensor, x_target: torch.Tensor,
                f_pred: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute loss.
        
        Args:
            x_pred: Predicted sparse signal (peaks)
            x_target: Target signal (ground truth peaks)
            f_pred: Predicted baseline (optional)
        """
        # MSE loss - main objective
        mse_loss = torch.mean((x_pred - x_target) ** 2)
        
        total_loss = self.alpha_mse * mse_loss
        
        # Baseline smoothness (penalize second derivative)
        if self.alpha_smooth > 0 and f_pred is not None:
            if f_pred.dim() == 1:
                f_pred = f_pred.unsqueeze(0)
            diff2 = f_pred[:, 2:] - 2 * f_pred[:, 1:-1] + f_pred[:, :-2]
            smooth_loss = torch.mean(diff2 ** 2)
            total_loss = total_loss + self.alpha_smooth * smooth_loss
        
        # Signal sparsity (L1 on first derivative)
        if self.alpha_sparse > 0:
            if x_pred.dim() == 1:
                x_pred_batch = x_pred.unsqueeze(0)
            else:
                x_pred_batch = x_pred
            diff1 = x_pred_batch[:, 1:] - x_pred_batch[:, :-1]
            sparse_loss = torch.mean(torch.abs(diff1))
            total_loss = total_loss + self.alpha_sparse * sparse_loss
        
        return total_loss


# =============================================================================
# Training Function
# =============================================================================


def train_lbeads_net(model: nn.Module,
                     train_y: torch.Tensor,
                     train_x_true: torch.Tensor,
                     num_epochs: int = 100,
                     learning_rate: float = 1e-3,
                     batch_size: int = 8,
                     device: str = 'cpu',
                     verbose: bool = True,
                     loss_config: Optional[Dict] = None) -> Tuple[List[float], List[Dict]]:
    """
    Train LBEADS-NET model on synthetic data with sparsity-based loss.
    
    Args:
        model: LBEADS-NET model
        train_y: Training observed signals (num_samples, N) - peaks + baseline + noise
        train_x_true: Training ground truth peaks (num_samples, N)
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        batch_size: Batch size
        device: Device to train on
        verbose: Print progress
        loss_config: Dictionary with loss weights (alpha_mse, alpha_l1, alpha_tv, etc.)
    
    Returns:
        loss_history: List of total training losses
        loss_details: List of loss component dictionaries
    """
    model = model.to(device)
    model.train()
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    
    # Configure loss function with sparsity penalties
    if loss_config is None:
        loss_config = {
            'alpha_mse': 1.0,       # Reconstruction (main objective)
            'alpha_l1': 0.001,      # L1 sparsity on peaks
            'alpha_tv': 0.001,      # Total Variation
            'alpha_smooth': 0.01,   # Baseline smoothness
            'alpha_neg': 0.1,       # Non-negativity
            'use_huber': True,
            'huber_delta': 1.0
        }
    
    criterion = SparsityLoss(**loss_config)
    
    num_samples = train_y.shape[0]
    loss_history = []
    loss_details = []
    last_epoch_time = 0
    
    for epoch in range(num_epochs):
        start_time = time.time()
        if verbose:
            print(f"Epoch {epoch + 1}/{num_epochs}, Last epoch time: {last_epoch_time:.2f}s")
        epoch_loss = 0.0
        epoch_loss_dict = {}
        num_batches = 0
        
        # Shuffle data
        perm = torch.randperm(num_samples)
        
        for i in range(0, num_samples, batch_size):
            batch_indices = perm[i:min(i + batch_size, num_samples)]
            
            y_batch = train_y[batch_indices].to(device)
            x_true_batch = train_x_true[batch_indices].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass: model predicts peaks (x) and baseline (f) from observed y
            x_pred, f_pred = model(y_batch)
            
            # Loss with sparsity penalties
            loss, loss_dict = criterion(x_pred, x_true_batch, f_pred)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            # Accumulate loss components
            for k, v in loss_dict.items():
                epoch_loss_dict[k] = epoch_loss_dict.get(k, 0.0) + v
            num_batches += 1
        
        scheduler.step()
        
        avg_loss = epoch_loss / num_batches
        loss_history.append(avg_loss)
        
        # Average loss components
        avg_loss_dict = {k: v / num_batches for k, v in epoch_loss_dict.items()}
        loss_details.append(avg_loss_dict)
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Total Loss: {avg_loss:.6f}")
            print(f"  Components: recon={avg_loss_dict.get('reconstruction', 0):.4f}, "
                  f"L1={avg_loss_dict.get('l1_sparsity', 0):.4f}, "
                  f"TV={avg_loss_dict.get('total_variation', 0):.4f}, "
                  f"neg={avg_loss_dict.get('non_negativity', 0):.6f}")
            # Print current learned parameters
            params = model.get_learned_params()
            param_str = ", ".join([f"{k}: {v:.4f}" for k, v in list(params.items())[:4]])
            print(f"  Model Params: {param_str}")
        last_epoch_time = time.time() - start_time
        
    return loss_history, loss_details


# =============================================================================
# Evaluation Function
# =============================================================================


def evaluate_model(model: nn.Module,
                   test_y: torch.Tensor,
                   test_x_true: torch.Tensor,
                   device: str = 'cpu') -> dict:
    """
    Evaluate trained model on test set.
    
    Args:
        model: Trained LBEADS-NET model
        test_y: Test observed signals (num_samples, N)
        test_x_true: Test ground truth peaks (num_samples, N)
        device: Device for computation
    
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    model.eval()
    model = model.to(device)
    
    with torch.no_grad():
        test_y = test_y.to(device)
        test_x_true = test_x_true.to(device)
        
        # Predict peaks and baseline
        x_pred, f_pred = model(test_y)
        
        # MSE between predicted peaks and ground truth peaks
        mse = torch.mean((x_pred - test_x_true) ** 2).item()
        
        # PSNR
        max_val = torch.max(torch.abs(test_x_true)).item()
        psnr = 10 * np.log10(max_val ** 2 / mse) if mse > 0 else float('inf')
        
        # MAE
        mae = torch.mean(torch.abs(x_pred - test_x_true)).item()
        
        # Correlation coefficient (average over samples)
        correlations = []
        for i in range(x_pred.shape[0]):
            pred_np = x_pred[i].cpu().numpy()
            true_np = test_x_true[i].cpu().numpy()
            corr = np.corrcoef(pred_np, true_np)[0, 1]
            correlations.append(corr)
        avg_corr = np.mean(correlations)
    
    return {
        'mse': mse,
        'psnr': psnr,
        'mae': mae,
        'correlation': avg_corr
    }


# =============================================================================
# Main Training Script
# =============================================================================


def main():
    """Main training script with synthetic data."""
    print("=" * 60)
    print("LBEADS-NET Training on Synthetic Data")
    print("=" * 60)
    
    # Configuration
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Synthetic data parameters
    N = 1024           # Signal length
    n_samples = 200    # Total number of samples
    train_ratio = 0.8  # 80% train, 20% test
    seed = 42          # For reproducibility
    
    # Generate synthetic dataset
    print("\nGenerating synthetic dataset...")
    print(f"  Signal length: {N}")
    print(f"  Total samples: {n_samples}")
    print(f"  Train/Test split: {int(train_ratio*100)}/{int((1-train_ratio)*100)}")
    
    generator = SyntheticDataGenerator(N=N, seed=seed)
    dataset = generator.generate_dataset(
        n_samples=n_samples,
        noise_level_range=(0.5, 2.0),  # Realistic noise levels
    )
    
    # Create train/test split (80/20)
    print("\nCreating train/test split...")
    (train_y, train_x_true), (test_y, test_x_true) = create_train_test_split(
        dataset, train_ratio=train_ratio, seed=seed
    )
    
    print(f"  Training samples: {train_y.shape[0]}")
    print(f"  Test samples: {test_y.shape[0]}")
    print(f"  Signal length: {train_y.shape[1]}")
    
    # Visualize a few examples
    print("\nVisualizing sample synthetic data...")
    fig_samples, axes = plt.subplots(2, 3, figsize=(14, 8))
    for i, ax in enumerate(axes.flat):
        if i < len(dataset):
            signal = dataset[i]
            ax.plot(signal.y, 'b', alpha=0.7, linewidth=0.5, label='Observed (y)')
            ax.plot(signal.x_true, 'g', linewidth=1, label='Peaks (x_true)')
            ax.plot(signal.f_true, 'r', linewidth=1, label='Baseline (f_true)')
            ax.set_title(f'Sample {i+1} (noise={signal.metadata["noise"]["noise_level"]:.3f})')
            ax.legend(fontsize=8)
            ax.set_xlim([0, N])
    plt.suptitle('Synthetic Training Data Examples', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, 'synthetic_data_samples.png'), dpi=150)
    print(f"  Saved to {os.path.join(script_dir, 'synthetic_data_samples.png')}")
    
    # Create model
    # Note: fc=0.006 means cutoff at 0.6% of Nyquist - VERY low frequency
    # This separates slow baseline from fast peaks
    print("\nCreating LBEADS-NET model...")
    model = LBEADS_NET_Fast(
        N=N,
        d=1,
        fc=0.01,  # Slightly higher cutoff for sharper separation
        num_layers=15,  # More iterations for better convergence
        init_lam0=0.5,  # Asymmetric sparsity penalty
        init_lam1=1.0,  # First derivative (smoothness)
        init_lam2=1.0,  # Second derivative (smoothness)
        init_r=6.0,     # Asymmetry ratio (penalize negative 6x more)
        init_step_size=0.01  # Larger step size for faster convergence
    )
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {num_params}")
    print(f"  Trainable parameters: {num_trainable}")
    
    # Print initial parameters
    print("\nInitial parameters:")
    init_params = model.get_learned_params()
    for k, v in list(init_params.items())[:8]:
        print(f"  {k}: {v:.4f}")
    
    # Configure sparsity-based loss function
    loss_config = {
        'alpha_mse': 1.0,       # Main reconstruction objective
        'alpha_l1': 0.001,      # L1 sparsity - encourage zeros in non-peak regions
        'alpha_tv': 0.001,      # Total Variation - sharp transitions
        'alpha_smooth': 0.01,   # Baseline smoothness
        'alpha_neg': 0.1,       # Non-negativity - peaks are positive
        'use_huber': True,      # Huber loss is more robust than MSE
        'huber_delta': 1.0
    }
    
    print("\nLoss function configuration:")
    for k, v in loss_config.items():
        print(f"  {k}: {v}")
    
    # Train model
    print("\n" + "=" * 60)
    print("Training with Sparsity-Based Loss...")
    print("=" * 60)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    start_time = time.time()
    loss_history, loss_details = train_lbeads_net(
        model,
        train_y,
        train_x_true,
        num_epochs=100,
        learning_rate=1e-2,
        batch_size=8,
        device=device,
        verbose=True,
        loss_config=loss_config
    )
    elapsed_time = time.time() - start_time
    print(f"\nTraining completed in {elapsed_time:.2f} seconds")
    
    # Evaluate on test set
    print("\n" + "=" * 60)
    print("Evaluating on Test Set...")
    print("=" * 60)
    test_metrics = evaluate_model(model, test_y, test_x_true, device)
    print(f"  MSE: {test_metrics['mse']:.6f}")
    print(f"  PSNR: {test_metrics['psnr']:.2f} dB")
    print(f"  MAE: {test_metrics['mae']:.6f}")
    print(f"  Correlation: {test_metrics['correlation']:.4f}")
    
    # Also evaluate on training set for comparison
    train_metrics = evaluate_model(model, train_y, train_x_true, device)
    print("\nTraining Set Metrics (for comparison):")
    print(f"  MSE: {train_metrics['mse']:.6f}")
    print(f"  PSNR: {train_metrics['psnr']:.2f} dB")
    print(f"  MAE: {train_metrics['mae']:.6f}")
    print(f"  Correlation: {train_metrics['correlation']:.4f}")
    
    # Print final parameters
    print("\nFinal learned parameters:")
    final_params = model.get_learned_params()
    for k, v in list(final_params.items())[:8]:
        print(f"  {k}: {v:.4f}")
    
    # Plot results
    fig_results = plt.figure(figsize=(16, 12))
    
    # Loss history
    plt.subplot(2, 3, 1)
    plt.plot(loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss')
    plt.title('Training Loss')
    plt.grid(True)
    
    # Loss components over time
    plt.subplot(2, 3, 2)
    epochs = range(1, len(loss_details) + 1)
    plt.plot(epochs, [d.get('reconstruction', 0) for d in loss_details], label='Reconstruction')
    plt.plot(epochs, [d.get('l1_sparsity', 0) for d in loss_details], label='L1 Sparsity')
    plt.plot(epochs, [d.get('total_variation', 0) for d in loss_details], label='Total Variation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Component')
    plt.title('Loss Components')
    plt.legend()
    plt.grid(True)
    
    # Test example 1
    plt.subplot(2, 3, 3)
    model.eval()
    with torch.no_grad():
        test_idx = 0
        y = test_y[test_idx:test_idx+1].to(device)
        x_pred, f_pred = model(y)
        
        y_np = y[0].cpu().numpy()
        x_np = x_pred[0].cpu().numpy()
        f_np = f_pred[0].cpu().numpy()
        x_true_np = test_x_true[test_idx].numpy()
    
    plt.plot(y_np, 'gray', alpha=0.5, linewidth=0.5, label='Observed')
    plt.plot(x_np, 'b', linewidth=1, label='Predicted Peaks')
    plt.plot(x_true_np, 'g--', linewidth=1, label='Ground Truth Peaks')
    plt.plot(f_np, 'r', linewidth=1, alpha=0.7, label='Predicted Baseline')
    plt.legend(fontsize=8)
    plt.title('Test Sample 1: Peak Recovery')
    plt.xlim([0, N])
    
    # Test example 2
    plt.subplot(2, 3, 4)
    with torch.no_grad():
        test_idx = min(5, test_y.shape[0] - 1)
        y = test_y[test_idx:test_idx+1].to(device)
        x_pred, f_pred = model(y)
        
        y_np = y[0].cpu().numpy()
        x_np = x_pred[0].cpu().numpy()
        f_np = f_pred[0].cpu().numpy()
        x_true_np = test_x_true[test_idx].numpy()
    
    plt.plot(y_np, 'gray', alpha=0.5, linewidth=0.5, label='Observed')
    plt.plot(x_np, 'b', linewidth=1, label='Predicted Peaks')
    plt.plot(x_true_np, 'g--', linewidth=1, label='Ground Truth Peaks')
    plt.plot(f_np, 'r', linewidth=1, alpha=0.7, label='Predicted Baseline')
    plt.legend(fontsize=8)
    plt.title('Test Sample 2: Peak Recovery')
    plt.xlim([0, N])
    
    # Error distribution
    plt.subplot(2, 3, 5)
    with torch.no_grad():
        all_pred, _ = model(test_y.to(device))
        errors = (all_pred.cpu() - test_x_true).numpy().flatten()
    plt.hist(errors, bins=50, density=True, alpha=0.7)
    plt.xlabel('Prediction Error')
    plt.ylabel('Density')
    plt.title(f'Test Error Distribution (MSE={test_metrics["mse"]:.4f})')
    plt.grid(True)
    
    # Sparsity visualization
    plt.subplot(2, 3, 6)
    with torch.no_grad():
        # Show sparsity pattern for one sample
        x_pred_flat = all_pred[0].cpu().numpy()
        x_true_flat = test_x_true[0].numpy()
    plt.plot(np.abs(x_true_flat), 'g-', alpha=0.7, label='|Ground Truth|')
    plt.plot(np.abs(x_pred_flat), 'b-', alpha=0.7, label='|Predicted|')
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Sparsity threshold')
    plt.xlabel('Sample Index')
    plt.ylabel('Absolute Value')
    plt.title('Sparsity Pattern')
    plt.legend(fontsize=8)
    plt.xlim([0, N])
    
    plt.suptitle('LBEADS-NET Training Results (Sparsity-Based Loss)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, 'training_results.png'), dpi=150)
    print(f"\nSaved results to {os.path.join(script_dir, 'training_results.png')}")
    
    # Save model
    model_path = os.path.join(script_dir, f'lbeads_net_sparsity_{int(time.time())}.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'N': N,
            'd': 1,
            'fc': 0.01,
            'num_layers': 15
        },
        'loss_config': loss_config,
        'final_params': final_params,
        'loss_history': loss_history,
        'loss_details': loss_details,
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'data_config': {
            'n_samples': n_samples,
            'train_ratio': train_ratio,
            'seed': seed
        }
    }, model_path)
    print(f"Saved model to {model_path}")
    
    plt.show()


if __name__ == "__main__":
    main()
