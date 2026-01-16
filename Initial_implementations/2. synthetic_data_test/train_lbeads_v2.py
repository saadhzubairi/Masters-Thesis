"""
Training Script for LBEADS-NET v2

This script trains the improved LBEADS-NET v2 model with:
1. SNR-focused loss functions
2. More layers (20+)
3. BEADS warm-start initialization
4. Improved architecture with momentum and skip connections

Author: Thesis Work
Date: January 2026
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import argparse
from typing import List, Tuple, Optional, Dict
import matplotlib.pyplot as plt
import json

# Add parent directories to path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, 'BEADS', 'Replicate'))

from synthetic_data_generator import SyntheticDataGenerator, SyntheticSignal
from lbeads_net_v2 import LBEADS_NET_v2, LBEADS_NET_v2_WarmStart, create_lbeads_net_v2, beads_warm_start
from losses_v2 import (
    LBEADSv2Loss, SNRLoss, NormalizedMSELoss, PeakAwareLoss,
    compute_snr, compute_rmse, compute_peak_error
)


# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

class TrainingConfigV2:
    """Configuration for LBEADS-NET v2 training."""
    
    # Signal parameters
    N: int = 1024
    SEED: int = 42
    
    # Training data
    N_TRAIN: int = 500           # More training data for better generalization
    N_VAL: int = 100             # More validation samples
    BATCH_SIZE: int = 16         # Larger batches for stability
    
    # Model architecture (KEY IMPROVEMENTS)
    NUM_LAYERS: int = 20         # More layers (BEADS uses 30-50 iterations)
    USE_MOMENTUM: bool = True    # FISTA-style acceleration
    USE_SKIP_CONNECTION: bool = True  # Residual learning
    USE_WARMSTART: bool = False  # Optional BEADS warm-start during training
    
    # Model preset ('default', 'fast', 'accurate', 'beads_match')
    MODEL_PRESET: str = 'default'
    
    # Initialization
    INIT_D: int = 1
    INIT_FC: float = 0.006       # Filter cutoff
    INIT_R: float = 6.0
    INIT_LAM0: float = 0.3       # Asymmetric penalty
    INIT_LAM1: float = 2.0       # First derivative penalty
    INIT_LAM2: float = 2.0       # Second derivative penalty
    INIT_STEP_SIZE: float = 0.05
    
    # Training hyperparameters
    LEARNING_RATE: float = 5e-4   # Slightly lower for stability
    WEIGHT_DECAY: float = 1e-5
    NUM_EPOCHS: int = 100         # More epochs for convergence
    SCHEDULER_PATIENCE: int = 10
    EARLY_STOP_PATIENCE: int = 20
    GRAD_CLIP: float = 1.0
    
    # Loss function weights (KEY: SNR-focused!)
    USE_SNR_LOSS: bool = True
    USE_PEAK_LOSS: bool = True
    USE_ASYMMETRIC: bool = True
    SIGNAL_WEIGHT: float = 1.0
    BASELINE_WEIGHT: float = 0.3
    SNR_WEIGHT: float = 0.5       # Important for noise reduction
    PEAK_WEIGHT: float = 0.3      # Important for peak preservation
    PEAK_HEIGHT_WEIGHT: float = 0.2
    ASYMMETRIC_WEIGHT: float = 0.1
    BASELINE_SMOOTH_WEIGHT: float = 0.05
    SIGNAL_SPARSE_WEIGHT: float = 0.02
    
    # Data augmentation
    USE_AUGMENTATION: bool = True
    NOISE_LEVEL_RANGE: Tuple[float, float] = (0.02, 0.20)  # Wider range
    AMPLITUDE_RANGE: Tuple[float, float] = (0.5, 2.5)
    
    # Output
    OUTPUT_DIR: str = os.path.join(script_dir, 'trained_models')
    MODEL_NAME: str = 'lbeads_net_v2_trained.pth'
    SAVE_HISTORY: bool = True


# ============================================================================
# DATASET
# ============================================================================

class SyntheticDatasetV2(Dataset):
    """PyTorch Dataset for synthetic signals with augmentation."""
    
    def __init__(
        self,
        signals: List[SyntheticSignal],
        augment: bool = False,
        noise_range: Tuple[float, float] = (0.02, 0.15)
    ):
        self.signals = signals
        self.augment = augment
        self.noise_range = noise_range
    
    def __len__(self):
        return len(self.signals)
    
    def __getitem__(self, idx):
        signal = self.signals[idx]
        
        y = signal.y.copy()
        x_true = signal.x_true.copy()
        f_true = signal.f_true.copy()
        
        # Data augmentation during training
        if self.augment:
            # Random amplitude scaling
            scale = np.random.uniform(0.8, 1.2)
            y = y * scale
            x_true = x_true * scale
            f_true = f_true * scale
            
            # Random noise level adjustment
            current_noise = np.std(y - x_true - f_true)
            if current_noise > 0:
                target_noise = np.random.uniform(*self.noise_range)
                noise = y - x_true - f_true
                noise = noise * (target_noise / current_noise)
                y = x_true + f_true + noise
        
        return {
            'y': torch.tensor(y, dtype=torch.float64),
            'x_true': torch.tensor(x_true, dtype=torch.float64),
            'f_true': torch.tensor(f_true, dtype=torch.float64)
        }


def generate_training_data_v2(config: TrainingConfigV2) -> Tuple[List[SyntheticSignal], List[SyntheticSignal]]:
    """Generate diverse training and validation datasets."""
    
    print("\nGenerating training data...")
    
    np.random.seed(config.SEED)
    
    train_generator = SyntheticDataGenerator(N=config.N, seed=config.SEED + 1000)
    val_generator = SyntheticDataGenerator(N=config.N, seed=config.SEED + 2000)
    
    # Training signals with diverse parameters
    train_signals = []
    for i in range(config.N_TRAIN):
        # Vary noise levels
        noise_level = np.random.uniform(*config.NOISE_LEVEL_RANGE)
        
        # Mix of noise types
        noise_type = np.random.choice(['gaussian', 'laplacian'], p=[0.7, 0.3])
        
        # Vary peak parameters
        peak_params = {
            'amplitude_range': config.AMPLITUDE_RANGE,
            'width_range': (20, 80),
            'num_peaks_range': (2, 5)
        }
        
        signal = train_generator.generate_signal(
            peak_params=peak_params,
            noise_type=noise_type,
            noise_level=noise_level
        )
        train_signals.append(signal)
    
    # Validation signals (fixed for consistent evaluation)
    val_signals = []
    for i in range(config.N_VAL):
        noise_level = np.random.uniform(0.03, 0.12)  # Slightly narrower range
        noise_type = 'gaussian' if i % 3 != 0 else 'laplacian'
        
        signal = val_generator.generate_signal(
            peak_params={'amplitude_range': (0.8, 2.0)},
            noise_type=noise_type,
            noise_level=noise_level
        )
        val_signals.append(signal)
    
    print(f"  Training signals: {len(train_signals)}")
    print(f"  Validation signals: {len(val_signals)}")
    
    return train_signals, val_signals


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def create_model_v2(config: TrainingConfigV2) -> nn.Module:
    """Create LBEADS-NET v2 model."""
    
    if config.MODEL_PRESET != 'custom':
        # Use preset configuration
        model = create_lbeads_net_v2(
            N=config.N,
            preset=config.MODEL_PRESET,
            d=config.INIT_D,
            fc=config.INIT_FC,
        )
    else:
        # Custom configuration
        model = LBEADS_NET_v2(
            N=config.N,
            d=config.INIT_D,
            fc=config.INIT_FC,
            num_layers=config.NUM_LAYERS,
            init_lam0=config.INIT_LAM0,
            init_lam1=config.INIT_LAM1,
            init_lam2=config.INIT_LAM2,
            init_r=config.INIT_R,
            init_step_size=config.INIT_STEP_SIZE,
            use_momentum=config.USE_MOMENTUM,
            use_skip_connection=config.USE_SKIP_CONNECTION,
        )
    
    return model


def create_loss_v2(config: TrainingConfigV2) -> nn.Module:
    """Create LBEADS v2 loss function."""
    
    return LBEADSv2Loss(
        signal_weight=config.SIGNAL_WEIGHT,
        baseline_weight=config.BASELINE_WEIGHT,
        use_snr_loss=config.USE_SNR_LOSS,
        use_peak_loss=config.USE_PEAK_LOSS,
        use_asymmetric=config.USE_ASYMMETRIC,
        snr_weight=config.SNR_WEIGHT,
        peak_weight=config.PEAK_WEIGHT,
        peak_height_weight=config.PEAK_HEIGHT_WEIGHT,
        asymmetric_weight=config.ASYMMETRIC_WEIGHT,
        baseline_smooth_weight=config.BASELINE_SMOOTH_WEIGHT,
        signal_sparse_weight=config.SIGNAL_SPARSE_WEIGHT,
    )


def train_epoch_v2(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    config: TrainingConfigV2,
    device: str = 'cpu'
) -> Dict[str, float]:
    """Train for one epoch."""
    
    model.train()
    metrics = {
        'loss': 0.0,
        'snr': 0.0,
        'rmse_signal': 0.0,
        'rmse_baseline': 0.0,
        'peak_error': 0.0
    }
    n_batches = 0
    
    for batch in dataloader:
        y = batch['y'].to(device)
        x_true = batch['x_true'].to(device)
        f_true = batch['f_true'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass (with optional warm-start)
        if config.USE_WARMSTART and hasattr(model, 'ws_lam0'):
            x_est, f_est = model(y, use_warmstart=True)
        else:
            x_est, f_est = model(y)
        
        # Compute loss
        loss = criterion(x_est, f_est, x_true, f_true)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if config.GRAD_CLIP > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
        
        optimizer.step()
        
        # Record metrics
        with torch.no_grad():
            metrics['loss'] += loss.item()
            metrics['snr'] += compute_snr(x_true, x_est)
            metrics['rmse_signal'] += compute_rmse(x_true, x_est)
            metrics['rmse_baseline'] += compute_rmse(f_true, f_est)
            metrics['peak_error'] += compute_peak_error(x_true, x_est)
        
        n_batches += 1
    
    # Average metrics
    for key in metrics:
        metrics[key] /= n_batches
    
    return metrics


def validate_v2(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    config: TrainingConfigV2,
    device: str = 'cpu'
) -> Dict[str, float]:
    """Validate the model."""
    
    model.eval()
    metrics = {
        'loss': 0.0,
        'snr': 0.0,
        'rmse_signal': 0.0,
        'rmse_baseline': 0.0,
        'peak_error': 0.0
    }
    n_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            y = batch['y'].to(device)
            x_true = batch['x_true'].to(device)
            f_true = batch['f_true'].to(device)
            
            # Forward pass
            if config.USE_WARMSTART and hasattr(model, 'ws_lam0'):
                x_est, f_est = model(y, use_warmstart=True)
            else:
                x_est, f_est = model(y)
            
            loss = criterion(x_est, f_est, x_true, f_true)
            
            metrics['loss'] += loss.item()
            metrics['snr'] += compute_snr(x_true, x_est)
            metrics['rmse_signal'] += compute_rmse(x_true, x_est)
            metrics['rmse_baseline'] += compute_rmse(f_true, f_est)
            metrics['peak_error'] += compute_peak_error(x_true, x_est)
            
            n_batches += 1
    
    for key in metrics:
        metrics[key] /= n_batches
    
    return metrics


def print_metrics(metrics: Dict[str, float], prefix: str = ""):
    """Print metrics in a formatted way."""
    parts = [f"{prefix}"]
    parts.append(f"Loss: {metrics['loss']:.4f}")
    parts.append(f"SNR: {metrics['snr']:.2f}dB")
    parts.append(f"RMSE: {metrics['rmse_signal']:.4f}")
    parts.append(f"Peak Err: {metrics['peak_error']:.4f}")
    print(" | ".join(parts))


def train_v2(config: TrainingConfigV2):
    """Main training function for LBEADS-NET v2."""
    
    print("\n" + "=" * 70)
    print("LBEADS-NET v2 TRAINING")
    print("=" * 70)
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    # Set seeds
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.SEED)
    
    # Generate data
    train_signals, val_signals = generate_training_data_v2(config)
    
    train_dataset = SyntheticDatasetV2(
        train_signals,
        augment=config.USE_AUGMENTATION,
        noise_range=config.NOISE_LEVEL_RANGE
    )
    val_dataset = SyntheticDatasetV2(val_signals, augment=False)
    
    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE,
        shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE,
        shuffle=False, num_workers=0
    )
    
    # Create model
    print("\nCreating LBEADS-NET v2 model...")
    model = create_model_v2(config)
    model = model.to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Preset: {config.MODEL_PRESET}")
    print(f"  Layers: {model.num_layers}")
    print(f"  Momentum: {model.use_momentum}")
    print(f"  Skip connections: {model.use_skip_connection}")
    print(f"  Total parameters: {n_params}")
    print(f"  Trainable parameters: {n_trainable}")
    
    # Create loss and optimizer
    criterion = create_loss_v2(config)
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5,
        patience=config.SCHEDULER_PATIENCE, min_lr=1e-6
    )
    
    # Training loop
    print("\n" + "-" * 70)
    print("Starting training...")
    print("-" * 70)
    
    best_val_loss = float('inf')
    best_val_snr = float('-inf')
    patience_counter = 0
    
    history = {
        'train_loss': [], 'val_loss': [],
        'train_snr': [], 'val_snr': [],
        'train_rmse': [], 'val_rmse': [],
        'lr': []
    }
    
    start_time = time.time()
    
    for epoch in range(config.NUM_EPOCHS):
        epoch_start = time.time()
        
        # Train
        train_metrics = train_epoch_v2(
            model, train_loader, criterion, optimizer, config, device
        )
        
        # Validate
        val_metrics = validate_v2(
            model, val_loader, criterion, config, device
        )
        
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']
        
        # Record history
        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['loss'])
        history['train_snr'].append(train_metrics['snr'])
        history['val_snr'].append(val_metrics['snr'])
        history['train_rmse'].append(train_metrics['rmse_signal'])
        history['val_rmse'].append(val_metrics['rmse_signal'])
        history['lr'].append(current_lr)
        
        # Print progress
        print(f"\nEpoch {epoch+1:3d}/{config.NUM_EPOCHS} ({epoch_time:.1f}s) | LR: {current_lr:.2e}")
        print_metrics(train_metrics, "  Train")
        print_metrics(val_metrics, "  Val  ")
        
        # Learning rate scheduling
        scheduler.step(val_metrics['loss'])
        
        # Save best model (by SNR - most important metric!)
        if val_metrics['snr'] > best_val_snr:
            best_val_snr = val_metrics['snr']
            best_val_loss = val_metrics['loss']
            patience_counter = 0
            
            os.makedirs(config.OUTPUT_DIR, exist_ok=True)
            model_path = os.path.join(config.OUTPUT_DIR, config.MODEL_NAME)
            
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': {
                    'N': config.N,
                    'd': config.INIT_D,
                    'fc': config.INIT_FC,
                    'num_layers': model.num_layers,
                    'use_momentum': model.use_momentum,
                    'use_skip_connection': model.use_skip_connection,
                    'model_type': 'LBEADS_NET_v2'
                },
                'epoch': epoch,
                'val_loss': val_metrics['loss'],
                'val_snr': val_metrics['snr'],
                'learned_params': model.get_learned_params()
            }, model_path)
            
            print(f"  ★ New best model! SNR={val_metrics['snr']:.2f}dB")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config.EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    total_time = time.time() - start_time
    
    # Load best model
    checkpoint = torch.load(os.path.join(config.OUTPUT_DIR, config.MODEL_NAME))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Print final summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Best validation SNR: {best_val_snr:.2f} dB")
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    # Print learned parameters
    model.print_params_summary()
    
    # Save training history
    if config.SAVE_HISTORY:
        history_path = os.path.join(config.OUTPUT_DIR, 'training_history_v2.json')
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        print(f"\nSaved training history to {history_path}")
        
        # Plot training curves
        plot_training_history(history, config.OUTPUT_DIR)
    
    return model, history


def plot_training_history(history: Dict, output_dir: str):
    """Plot and save training history."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss
    ax = axes[0, 0]
    ax.plot(history['train_loss'], 'b-', label='Train', linewidth=2)
    ax.plot(history['val_loss'], 'r-', label='Validation', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # SNR
    ax = axes[0, 1]
    ax.plot(history['train_snr'], 'b-', label='Train', linewidth=2)
    ax.plot(history['val_snr'], 'r-', label='Validation', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('SNR (dB)')
    ax.set_title('Signal-to-Noise Ratio')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # RMSE
    ax = axes[1, 0]
    ax.plot(history['train_rmse'], 'b-', label='Train', linewidth=2)
    ax.plot(history['val_rmse'], 'r-', label='Validation', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('RMSE')
    ax.set_title('Signal RMSE')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Learning rate
    ax = axes[1, 1]
    ax.plot(history['lr'], 'g-', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'training_history_v2.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved training plot to {save_path}")
    plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point."""
    
    parser = argparse.ArgumentParser(description='Train LBEADS-NET v2')
    
    # Data parameters
    parser.add_argument('--n-train', type=int, default=500)
    parser.add_argument('--n-val', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=16)
    
    # Model parameters
    parser.add_argument('--preset', type=str, default='default',
                        choices=['default', 'fast', 'accurate', 'beads_match', 'custom'])
    parser.add_argument('--layers', type=int, default=20)
    parser.add_argument('--no-momentum', action='store_true')
    parser.add_argument('--no-skip', action='store_true')
    parser.add_argument('--warmstart', action='store_true')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--seed', type=int, default=42)
    
    # Loss parameters
    parser.add_argument('--no-snr-loss', action='store_true')
    parser.add_argument('--no-peak-loss', action='store_true')
    parser.add_argument('--snr-weight', type=float, default=0.5)
    parser.add_argument('--peak-weight', type=float, default=0.3)
    
    # Output
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--model-name', type=str, default='lbeads_net_v2_trained.pth')
    
    args = parser.parse_args()
    
    # Update config
    config = TrainingConfigV2()
    config.N_TRAIN = args.n_train
    config.N_VAL = args.n_val
    config.BATCH_SIZE = args.batch_size
    config.MODEL_PRESET = args.preset
    config.NUM_LAYERS = args.layers
    config.USE_MOMENTUM = not args.no_momentum
    config.USE_SKIP_CONNECTION = not args.no_skip
    config.USE_WARMSTART = args.warmstart
    config.NUM_EPOCHS = args.epochs
    config.LEARNING_RATE = args.lr
    config.SEED = args.seed
    config.USE_SNR_LOSS = not args.no_snr_loss
    config.USE_PEAK_LOSS = not args.no_peak_loss
    config.SNR_WEIGHT = args.snr_weight
    config.PEAK_WEIGHT = args.peak_weight
    config.MODEL_NAME = args.model_name
    
    if args.output_dir:
        config.OUTPUT_DIR = args.output_dir
    
    # Train
    train_v2(config)


if __name__ == "__main__":
    main()
