"""
Training script for LBEADS-NET

This script demonstrates how to train the unrolled LBEADS-NET model
to learn optimal regularization parameters for baseline estimation
and denoising.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import os
import time
from typing import Tuple, List, Optional

from lbeads_net import LBEADS_NET, LBEADS_NET_Fast


def load_data(data_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load chromatogram data and noise."""
    noise_data = sio.loadmat(os.path.join(data_dir, 'noise.mat'))
    chromatograms_data = sio.loadmat(os.path.join(data_dir, 'chromatograms.mat'))
    
    noise = noise_data['noise'].flatten()
    X = chromatograms_data['X']
    
    return X, noise


def create_training_data(X: np.ndarray, noise: np.ndarray, 
                         noise_levels: List[float] = [0.3, 0.5, 0.7, 1.0],
                         column_indices: List[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create training data by adding noise at various levels.
    
    For supervised learning, we need (noisy, clean) pairs.
    Here we assume the original chromatograms X have some baseline,
    and we want to learn to separate signal from baseline.
    
    Args:
        X: Chromatogram data (N, num_samples)
        noise: Noise vector
        noise_levels: List of noise scaling factors
        column_indices: Which columns to use (default: all)
    
    Returns:
        noisy_signals: Tensor of noisy observations
        clean_signals: Tensor of clean targets (for now, use X as approximation)
    """
    if column_indices is None:
        column_indices = list(range(X.shape[1]))
    
    noisy_list = []
    clean_list = []
    
    for col_idx in column_indices:
        clean = X[:, col_idx]
        for noise_level in noise_levels:
            noisy = clean + noise * noise_level
            noisy_list.append(noisy)
            clean_list.append(clean)
    
    noisy_signals = torch.tensor(np.array(noisy_list), dtype=torch.float64)
    clean_signals = torch.tensor(np.array(clean_list), dtype=torch.float64)
    
    return noisy_signals, clean_signals


class BEADSLoss(nn.Module):
    """
    Custom loss function for BEADS training.
    
    Combines:
    1. MSE between estimated signal and target
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
            x_pred: Predicted sparse signal
            x_target: Target signal
            f_pred: Predicted baseline (optional)
        """
        # MSE loss
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


def train_lbeads_net(model: nn.Module,
                     train_noisy: torch.Tensor,
                     train_clean: torch.Tensor,
                     num_epochs: int = 100,
                     learning_rate: float = 1e-3,
                     batch_size: int = 4,
                     device: str = 'cpu',
                     verbose: bool = True) -> List[float]:
    """
    Train LBEADS-NET model.
    
    Args:
        model: LBEADS-NET model
        train_noisy: Training noisy signals (num_samples, N)
        train_clean: Training clean signals (num_samples, N)
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        batch_size: Batch size
        device: Device to train on
        verbose: Print progress
    
    Returns:
        loss_history: List of training losses
    """
    model = model.to(device)
    model.train()
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    criterion = BEADSLoss(alpha_mse=1.0, alpha_smooth=0.01, alpha_sparse=0.01)
    
    num_samples = train_noisy.shape[0]
    loss_history = []
    last_epoch_time = 0
    for epoch in range(num_epochs):
        start_time = time.time()
        print(f"Epoch {epoch + 1}/{num_epochs}, Last epoch time: {last_epoch_time:.2f}s")
        epoch_loss = 0.0
        num_batches = 0
        
        # Shuffle data
        perm = torch.randperm(num_samples)
        
        for i in range(0, num_samples, batch_size):
            batch_indices = perm[i:min(i + batch_size, num_samples)]
            
            noisy_batch = train_noisy[batch_indices].to(device)
            clean_batch = train_clean[batch_indices].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            x_pred, f_pred = model(noisy_batch)
            
            # Compute loss
            loss = criterion(x_pred, clean_batch, f_pred)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        scheduler.step()
        
        avg_loss = epoch_loss / num_batches
        loss_history.append(avg_loss)
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.6f}")
            # Print current learned parameters
            params = model.get_learned_params()
            param_str = ", ".join([f"{k}: {v:.4f}" for k, v in list(params.items())[:4]])
            print(f"  Params: {param_str}")
        last_epoch_time = time.time() - start_time
        
    return loss_history


def evaluate_model(model: nn.Module,
                   test_noisy: torch.Tensor,
                   test_clean: torch.Tensor,
                   device: str = 'cpu') -> dict:
    """
    Evaluate trained model.
    
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    model.eval()
    model = model.to(device)
    
    with torch.no_grad():
        test_noisy = test_noisy.to(device)
        test_clean = test_clean.to(device)
        
        x_pred, f_pred = model(test_noisy)
        
        # MSE
        mse = torch.mean((x_pred - test_clean) ** 2).item()
        
        # PSNR
        max_val = torch.max(test_clean).item()
        psnr = 10 * np.log10(max_val ** 2 / mse) if mse > 0 else float('inf')
        
        # MAE
        mae = torch.mean(torch.abs(x_pred - test_clean)).item()
    
    return {
        'mse': mse,
        'psnr': psnr,
        'mae': mae
    }


def main():
    """Main training script."""
    print("=" * 60)
    print("LBEADS-NET Training")
    print("=" * 60)
    
    # Get data directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(script_dir), 'BEADS', 'data')
    
    # Check if data exists
    if not os.path.exists(os.path.join(data_dir, 'noise.mat')):
        print(f"Data not found in {data_dir}")
        print("Please ensure noise.mat and chromatograms.mat are present.")
        return
    
    # Load data
    print("\nLoading data...")
    X, noise = load_data(data_dir)
    print(f"Chromatogram data shape: {X.shape}")
    print(f"Noise shape: {noise.shape}")
    
    # Create training data
    print("\nCreating training data...")
    train_noisy, train_clean = create_training_data(
        X, noise, 
        noise_levels=[0.3, 0.5, 0.7],
        column_indices=[0, 1, 2, 3, 4]  # Use first 5 columns
    )
    print(f"Training samples: {train_noisy.shape[0]}")
    
    # Signal length
    N = train_noisy.shape[1]
    print(f"Signal length: {N}")
    
    # Create model
    print("\nCreating LBEADS-NET model...")
    model = LBEADS_NET_Fast(
        N=N,
        d=1,
        fc=0.006,
        num_layers=10,  # 10 unrolled iterations
        init_lam0=0.4,
        init_lam1=4.0,
        init_lam2=3.2,
        init_r=6.0,
        init_step_size=0.001  # Smaller step size for stability
    )
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {num_params}")
    print(f"Trainable parameters: {num_trainable}")
    
    # Print initial parameters
    print("\nInitial parameters:")
    init_params = model.get_learned_params()
    for k, v in list(init_params.items())[:8]:
        print(f"  {k}: {v:.4f}")
    
    # Train model
    print("\nTraining...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    start_time = time.time()
    loss_history = train_lbeads_net(
        model,
        train_noisy,
        train_clean,
        num_epochs=50,
        learning_rate=1e-2,
        batch_size=4,
        device=device,
        verbose=True
    )
    elapsed_time = time.time() - start_time
    print(f"\nTraining completed in {elapsed_time:.2f} seconds")
    
    # Evaluate
    print("\nEvaluating...")
    metrics = evaluate_model(model, train_noisy[:4], train_clean[:4], device)
    print(f"MSE: {metrics['mse']:.6f}")
    print(f"PSNR: {metrics['psnr']:.2f} dB")
    print(f"MAE: {metrics['mae']:.6f}")
    
    # Print final parameters
    print("\nFinal learned parameters:")
    final_params = model.get_learned_params()
    for k, v in list(final_params.items())[:8]:
        print(f"  {k}: {v:.4f}")
    
    # Plot loss history
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)
    
    # Plot example result
    plt.subplot(1, 2, 2)
    model.eval()
    with torch.no_grad():
        test_idx = 0
        y = train_noisy[test_idx:test_idx+1].to(device)
        x_pred, f_pred = model(y)
        
        y_np = y[0].cpu().numpy()
        x_np = x_pred[0].cpu().numpy()
        f_np = f_pred[0].cpu().numpy()
        clean_np = train_clean[test_idx].numpy()
    
    plt.plot(y_np, 'gray', alpha=0.5, label='Noisy')
    plt.plot(x_np, 'b', label='Predicted Signal')
    plt.plot(f_np, 'r', label='Predicted Baseline')
    plt.plot(clean_np, 'g--', alpha=0.7, label='Target')
    plt.legend()
    plt.title('LBEADS-NET Result')
    plt.xlim([0, 500])  # Show first 500 samples
    
    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, 'training_results.png'), dpi=150)
    print(f"\nSaved results to {os.path.join(script_dir, 'training_results.png')}")
    
    # Save model
    model_path = os.path.join(script_dir, f'lbeads_net_trained_{int(time.time())}.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'N': N,
            'd': 1,
            'fc': 0.006,
            'num_layers': 10
        },
        'final_params': final_params,
        'loss_history': loss_history
    }, model_path)
    print(f"Saved model to {model_path}")
    
    plt.show()


if __name__ == "__main__":
    main()
