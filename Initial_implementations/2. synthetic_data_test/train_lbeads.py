"""
Training Script for LBEADS-NET

This script trains LBEADS-NET on synthetic data to learn optimal
per-layer parameters for baseline estimation and denoising.

Key differences from untrained version:
- Uses fewer layers (5-10 instead of 30)
- Unshared parameters across layers
- End-to-end training with MSE loss

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
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt

# Add parent directories to path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, 'LBEADS_NETv1'))
sys.path.insert(0, os.path.join(parent_dir, 'BEADS', 'Replicate'))

from synthetic_data_generator import SyntheticDataGenerator, SyntheticSignal
from lbeads_net import LBEADS_NET, LBEADS_NET_Fast


# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

class TrainingConfig:
    """Configuration for training."""
    
    # Signal parameters
    N: int = 1024
    SEED: int = 42
    
    # Training data
    N_TRAIN: int = 200           # Number of training signals
    N_VAL: int = 50              # Number of validation signals
    BATCH_SIZE: int = 8          # Batch size
    
    # Model architecture
    NUM_LAYERS: int = 10         # Fewer layers for learning
    SHARED_PARAMS: bool = False  # Different params per layer (KEY!)
    
    # Initialization (tuned for synthetic data - IMPORTANT!)
    INIT_D: int = 1
    INIT_FC: float = 0.002           # Lower cutoff for wider peaks
    INIT_R: float = 6.0
    INIT_LAM0: float = 0.1           # Lower for more signal recovery
    INIT_LAM1: float = 0.5           # Lower for less aggressive smoothing
    INIT_LAM2: float = 0.5
    
    # Training hyperparameters
    LEARNING_RATE: float = 1e-3
    WEIGHT_DECAY: float = 1e-5
    NUM_EPOCHS: int = 50
    SCHEDULER_PATIENCE: int = 5
    EARLY_STOP_PATIENCE: int = 10
    
    # Loss weights
    SIGNAL_LOSS_WEIGHT: float = 1.0
    BASELINE_LOSS_WEIGHT: float = 0.5
    
    # Output
    OUTPUT_DIR: str = os.path.join(script_dir, 'trained_models')
    MODEL_NAME: str = 'lbeads_net_trained.pth'


# ============================================================================
# DATASET
# ============================================================================

class SyntheticDataset(Dataset):
    """PyTorch Dataset for synthetic signals."""
    
    def __init__(self, signals: List[SyntheticSignal]):
        self.signals = signals
    
    def __len__(self):
        return len(self.signals)
    
    def __getitem__(self, idx):
        signal = self.signals[idx]
        return {
            'y': torch.tensor(signal.y, dtype=torch.float64),
            'x_true': torch.tensor(signal.x_true, dtype=torch.float64),
            'f_true': torch.tensor(signal.f_true, dtype=torch.float64)
        }


def generate_training_data(config: TrainingConfig) -> Tuple[List[SyntheticSignal], List[SyntheticSignal]]:
    """Generate training and validation datasets."""
    
    print("\nGenerating training data...")
    
    # Use different seed for training data (but reproducible)
    train_generator = SyntheticDataGenerator(N=config.N, seed=config.SEED + 1000)
    val_generator = SyntheticDataGenerator(N=config.N, seed=config.SEED + 2000)
    
    # Generate with higher signal amplitudes for better SNR
    train_signals = []
    for i in range(config.N_TRAIN):
        # Mix of noise levels
        noise_level = np.random.uniform(0.03, 0.15)
        noise_type = 'gaussian' if i % 4 != 0 else 'laplacian'
        
        signal = train_generator.generate_signal(
            peak_params={'amplitude_range': (0.8, 2.0)},  # Higher amplitude peaks
            noise_type=noise_type,
            noise_level=noise_level
        )
        train_signals.append(signal)
    
    val_signals = []
    for i in range(config.N_VAL):
        noise_level = np.random.uniform(0.03, 0.15)
        noise_type = 'gaussian' if i % 4 != 0 else 'laplacian'
        
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
# LOSS FUNCTION
# ============================================================================

class CombinedLoss(nn.Module):
    """Combined loss for signal and baseline estimation."""
    
    def __init__(self, signal_weight: float = 1.0, baseline_weight: float = 0.5):
        super().__init__()
        self.signal_weight = signal_weight
        self.baseline_weight = baseline_weight
        self.mse = nn.MSELoss()
    
    def forward(self, x_est, f_est, x_true, f_true):
        """
        Compute combined loss.
        
        Args:
            x_est: Estimated signal (batch, N)
            f_est: Estimated baseline (batch, N)
            x_true: Ground truth signal (batch, N)
            f_true: Ground truth baseline (batch, N)
        """
        loss_signal = self.mse(x_est, x_true)
        loss_baseline = self.mse(f_est, f_true)
        
        total = self.signal_weight * loss_signal + self.baseline_weight * loss_baseline
        
        return total, loss_signal, loss_baseline


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def create_model(config: TrainingConfig):
    """Create LBEADS-NET model for training.
    
    Uses LBEADS_NET_Fast which implements an ISTA-style proximal gradient
    approach that's fast, differentiable, and can learn good parameters.
    """
    
    model = LBEADS_NET_Fast(
        N=config.N,
        d=config.INIT_D,
        fc=config.INIT_FC,
        num_layers=config.NUM_LAYERS,
        init_lam0=config.INIT_LAM0,
        init_lam1=config.INIT_LAM1,
        init_lam2=config.INIT_LAM2,
        init_r=config.INIT_R,
        init_step_size=0.1
    )
    
    return model


def train_epoch(model, dataloader: DataLoader, 
                criterion: CombinedLoss, optimizer: optim.Optimizer,
                device: str = 'cpu') -> Tuple[float, float, float]:
    """Train for one epoch."""
    
    model.train()
    total_loss = 0.0
    total_signal_loss = 0.0
    total_baseline_loss = 0.0
    
    for batch in dataloader:
        y = batch['y'].to(device)
        x_true = batch['x_true'].to(device)
        f_true = batch['f_true'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        x_est, f_est = model(y)
        
        # Compute loss
        loss, loss_sig, loss_base = criterion(x_est, f_est, x_true, f_true)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        total_signal_loss += loss_sig.item()
        total_baseline_loss += loss_base.item()
    
    n_batches = len(dataloader)
    return total_loss / n_batches, total_signal_loss / n_batches, total_baseline_loss / n_batches


def validate(model, dataloader: DataLoader,
             criterion: CombinedLoss, device: str = 'cpu') -> Tuple[float, float, float]:
    """Validate the model."""
    
    model.eval()
    total_loss = 0.0
    total_signal_loss = 0.0
    total_baseline_loss = 0.0
    
    with torch.no_grad():
        for batch in dataloader:
            y = batch['y'].to(device)
            x_true = batch['x_true'].to(device)
            f_true = batch['f_true'].to(device)
            
            x_est, f_est = model(y)
            loss, loss_sig, loss_base = criterion(x_est, f_est, x_true, f_true)
            
            total_loss += loss.item()
            total_signal_loss += loss_sig.item()
            total_baseline_loss += loss_base.item()
    
    n_batches = len(dataloader)
    return total_loss / n_batches, total_signal_loss / n_batches, total_baseline_loss / n_batches


def print_params(model, title: str = "Model Parameters"):
    """Print current model parameters."""
    print(f"\n{title}:")
    params = model.get_learned_params()
    for k, v in params.items():
        print(f"  {k}: {v:.4f}")


def train(config: TrainingConfig):
    """Main training function."""
    
    print("\n" + "=" * 60)
    print("LBEADS-NET TRAINING")
    print("=" * 60)
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    # Generate data
    train_signals, val_signals = generate_training_data(config)
    
    train_dataset = SyntheticDataset(train_signals)
    val_dataset = SyntheticDataset(val_signals)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    # Create model
    print("\nCreating model...")
    model = create_model(config)
    model = model.to(device)
    
    print(f"  Layers: {config.NUM_LAYERS}")
    print(f"  Shared params: {config.SHARED_PARAMS}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # Print initial parameters
    print_params(model, "Initial Parameters (before training)")
    
    # Loss and optimizer
    criterion = CombinedLoss(config.SIGNAL_LOSS_WEIGHT, config.BASELINE_LOSS_WEIGHT)
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, 
                            weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=config.SCHEDULER_PATIENCE
    )
    
    # Training loop
    print("\n" + "-" * 60)
    print("Starting training...")
    print("-" * 60)
    
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'train_sig': [], 'val_sig': []}
    
    for epoch in range(config.NUM_EPOCHS):
        start_time = time.time()
        
        # Train
        train_loss, train_sig, train_base = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_sig, val_base = validate(model, val_loader, criterion, device)
        
        epoch_time = time.time() - start_time
        
        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_sig'].append(train_sig)
        history['val_sig'].append(val_sig)
        
        # Print progress
        print(f"Epoch {epoch+1:3d}/{config.NUM_EPOCHS} | "
              f"Train: {train_loss:.4f} (sig={train_sig:.4f}) | "
              f"Val: {val_loss:.4f} (sig={val_sig:.4f}) | "
              f"Time: {epoch_time:.1f}s")
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            os.makedirs(config.OUTPUT_DIR, exist_ok=True)
            model_path = os.path.join(config.OUTPUT_DIR, config.MODEL_NAME)
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': {
                    'N': config.N,
                    'd': config.INIT_D,
                    'fc': config.INIT_FC,
                    'num_layers': config.NUM_LAYERS,
                    'model_type': 'LBEADS_NET_Fast'
                },
                'epoch': epoch,
                'val_loss': val_loss
            }, model_path)
            print(f"  ✓ Saved best model (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config.EARLY_STOP_PATIENCE:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    # Print final parameters
    print_params(model, "Final Parameters (after training)")
    
    # Load best model
    checkpoint = torch.load(os.path.join(config.OUTPUT_DIR, config.MODEL_NAME))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Plot training history
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(history['train_loss'], 'b-', label='Train')
    axes[0].plot(history['val_loss'], 'r-', label='Validation')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Total Loss')
    axes[0].set_title('Training History')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(history['train_sig'], 'b-', label='Train (signal)')
    axes[1].plot(history['val_sig'], 'r-', label='Val (signal)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MSE')
    axes[1].set_title('Signal MSE')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    history_path = os.path.join(config.OUTPUT_DIR, 'training_history.png')
    plt.savefig(history_path, dpi=150)
    print(f"\nSaved training history to {history_path}")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {os.path.join(config.OUTPUT_DIR, config.MODEL_NAME)}")
    
    return model


def main():
    """Main entry point."""
    
    parser = argparse.ArgumentParser(description='Train LBEADS-NET')
    parser.add_argument('--n-train', type=int, default=200, help='Training samples')
    parser.add_argument('--n-val', type=int, default=50, help='Validation samples')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--layers', type=int, default=10, help='Number of layers')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    # Update config
    TrainingConfig.N_TRAIN = args.n_train
    TrainingConfig.N_VAL = args.n_val
    TrainingConfig.NUM_EPOCHS = args.epochs
    TrainingConfig.NUM_LAYERS = args.layers
    TrainingConfig.LEARNING_RATE = args.lr
    TrainingConfig.BATCH_SIZE = args.batch_size
    TrainingConfig.SEED = args.seed
    
    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Train
    train(TrainingConfig)


if __name__ == "__main__":
    main()
