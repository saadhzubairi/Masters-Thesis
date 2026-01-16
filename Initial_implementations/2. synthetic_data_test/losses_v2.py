"""
Improved Loss Functions for LBEADS-NET v2

This module provides SNR-focused and physics-informed loss functions
that are better suited for baseline estimation and denoising than simple MSE.

Key improvements:
1. SNR-based loss: Directly optimizes signal-to-noise ratio
2. Peak-aware loss: Gives higher weight to peak regions
3. Multi-scale loss: Combines losses at different frequency scales
4. Perceptual loss: Focuses on preserving peak shape and height
5. Physics-informed loss: Enforces baseline smoothness and signal sparsity

Author: Thesis Work
Date: January 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


# ===========================================================================
# SNR-BASED LOSSES
# ===========================================================================

class SNRLoss(nn.Module):
    """
    Signal-to-Noise Ratio based loss.
    
    Maximizes: SNR = 10 * log10(||x_true||^2 / ||x_true - x_pred||^2)
    
    Equivalently minimizes: -SNR or the normalized MSE ratio.
    
    This loss is scale-invariant and focuses on reconstruction quality
    relative to signal power, which is better for chromatography where
    peak heights vary significantly.
    """
    
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
    
    def forward(
        self,
        x_pred: torch.Tensor,
        x_true: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute negative SNR loss (minimizing this maximizes SNR).
        
        Args:
            x_pred: Predicted signal (batch, N)
            x_true: Ground truth signal (batch, N)
        """
        if x_pred.dim() == 1:
            x_pred = x_pred.unsqueeze(0)
            x_true = x_true.unsqueeze(0)
        
        # Signal power
        signal_power = torch.sum(x_true ** 2, dim=1) + self.eps
        
        # Noise power (error power)
        noise_power = torch.sum((x_true - x_pred) ** 2, dim=1) + self.eps
        
        # SNR in dB (we want to maximize this, so return negative)
        snr_db = 10 * torch.log10(signal_power / noise_power)
        
        # Return negative mean SNR (minimize this to maximize SNR)
        return -snr_db.mean()


class NormalizedMSELoss(nn.Module):
    """
    Normalized MSE loss (equivalent to minimizing 1/SNR in linear scale).
    
    NMSE = ||x_true - x_pred||^2 / ||x_true||^2
    
    This is better than MSE when signals have different amplitudes.
    """
    
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
    
    def forward(
        self,
        x_pred: torch.Tensor,
        x_true: torch.Tensor,
    ) -> torch.Tensor:
        if x_pred.dim() == 1:
            x_pred = x_pred.unsqueeze(0)
            x_true = x_true.unsqueeze(0)
        
        mse = torch.sum((x_true - x_pred) ** 2, dim=1)
        signal_power = torch.sum(x_true ** 2, dim=1) + self.eps
        
        nmse = mse / signal_power
        return nmse.mean()


# ===========================================================================
# PEAK-AWARE LOSSES
# ===========================================================================

class PeakAwareLoss(nn.Module):
    """
    Loss that gives higher weight to peak regions.
    
    Peaks are the most important part of chromatograms - we want to
    preserve their height and shape accurately. This loss weights
    errors by the local signal amplitude.
    """
    
    def __init__(
        self,
        peak_weight: float = 5.0,
        threshold_quantile: float = 0.7,
        eps: float = 1e-8
    ):
        """
        Args:
            peak_weight: Extra weight for peak regions
            threshold_quantile: Quantile above which to consider as "peak"
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.peak_weight = peak_weight
        self.threshold_quantile = threshold_quantile
        self.eps = eps
    
    def forward(
        self,
        x_pred: torch.Tensor,
        x_true: torch.Tensor,
    ) -> torch.Tensor:
        if x_pred.dim() == 1:
            x_pred = x_pred.unsqueeze(0)
            x_true = x_true.unsqueeze(0)
        
        # Compute per-sample threshold
        threshold = torch.quantile(x_true, self.threshold_quantile, dim=1, keepdim=True)
        
        # Create weight mask: higher weight where x_true is above threshold
        weights = torch.ones_like(x_true)
        peak_mask = x_true > threshold
        weights[peak_mask] = self.peak_weight
        
        # Weighted MSE
        squared_error = (x_true - x_pred) ** 2
        weighted_error = weights * squared_error
        
        return weighted_error.mean()


class PeakHeightLoss(nn.Module):
    """
    Loss that specifically penalizes errors in peak heights.
    
    Finds local maxima and compares their heights.
    This is crucial because peak attenuation is a common problem.
    """
    
    def __init__(self, height_weight: float = 2.0):
        super().__init__()
        self.height_weight = height_weight
    
    def find_peaks_soft(self, x: torch.Tensor, window: int = 5) -> torch.Tensor:
        """
        Soft peak detection using local maximum filter.
        Returns a soft mask indicating peak locations.
        """
        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(0)  # (1, 1, N)
        elif x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, N)
        
        # Max pooling to find local maxima
        padding = window // 2
        local_max = F.max_pool1d(x, window, stride=1, padding=padding)
        
        # Points where x equals local max are peaks
        peak_mask = (x >= local_max - 1e-6).float().squeeze(1)
        
        # Weight by amplitude (higher peaks get higher weight)
        peak_weight = peak_mask * torch.abs(x.squeeze(1))
        
        return peak_weight
    
    def forward(
        self,
        x_pred: torch.Tensor,
        x_true: torch.Tensor,
    ) -> torch.Tensor:
        if x_pred.dim() == 1:
            x_pred = x_pred.unsqueeze(0)
            x_true = x_true.unsqueeze(0)
        
        # Find peaks in true signal
        peak_weights = self.find_peaks_soft(x_true)
        
        # Weighted error at peak locations
        peak_error = peak_weights * (x_true - x_pred) ** 2
        
        # Normalize by sum of weights
        weight_sum = peak_weights.sum(dim=1, keepdim=True) + 1e-8
        normalized_error = peak_error.sum(dim=1) / weight_sum.squeeze()
        
        return self.height_weight * normalized_error.mean()


# ===========================================================================
# MULTI-SCALE LOSS
# ===========================================================================

class MultiScaleLoss(nn.Module):
    """
    Multi-scale loss that combines errors at different frequency bands.
    
    This helps capture both:
    - High-frequency noise (should be removed)
    - Low-frequency baseline (should be removed)
    - Mid-frequency peaks (should be preserved)
    """
    
    def __init__(
        self,
        scales: list = [1, 2, 4, 8],
        weights: Optional[list] = None
    ):
        super().__init__()
        self.scales = scales
        self.weights = weights or [1.0] * len(scales)
    
    def forward(
        self,
        x_pred: torch.Tensor,
        x_true: torch.Tensor,
    ) -> torch.Tensor:
        if x_pred.dim() == 1:
            x_pred = x_pred.unsqueeze(0)
            x_true = x_true.unsqueeze(0)
        
        total_loss = 0.0
        
        for scale, weight in zip(self.scales, self.weights):
            if scale == 1:
                # Original scale
                loss = F.mse_loss(x_pred, x_true)
            else:
                # Downsampled scale
                x_pred_ds = F.avg_pool1d(
                    x_pred.unsqueeze(1), scale, stride=scale
                ).squeeze(1)
                x_true_ds = F.avg_pool1d(
                    x_true.unsqueeze(1), scale, stride=scale
                ).squeeze(1)
                loss = F.mse_loss(x_pred_ds, x_true_ds)
            
            total_loss = total_loss + weight * loss
        
        return total_loss / sum(self.weights)


# ===========================================================================
# PHYSICS-INFORMED LOSSES
# ===========================================================================

class BaselineSmoothnessLoss(nn.Module):
    """
    Regularization loss that encourages smooth baseline estimates.
    
    A good baseline should be:
    1. Smooth (small second derivative)
    2. Slowly varying (small first derivative)
    3. Non-negative in many applications
    """
    
    def __init__(
        self,
        d1_weight: float = 0.1,
        d2_weight: float = 1.0,
        noneg_weight: float = 0.0  # Optional non-negativity
    ):
        super().__init__()
        self.d1_weight = d1_weight
        self.d2_weight = d2_weight
        self.noneg_weight = noneg_weight
    
    def forward(self, f_pred: torch.Tensor) -> torch.Tensor:
        if f_pred.dim() == 1:
            f_pred = f_pred.unsqueeze(0)
        
        total_loss = 0.0
        
        # First derivative penalty
        if self.d1_weight > 0:
            diff1 = f_pred[:, 1:] - f_pred[:, :-1]
            d1_loss = torch.mean(diff1 ** 2)
            total_loss = total_loss + self.d1_weight * d1_loss
        
        # Second derivative penalty (curvature)
        if self.d2_weight > 0:
            diff2 = f_pred[:, 2:] - 2 * f_pred[:, 1:-1] + f_pred[:, :-2]
            d2_loss = torch.mean(diff2 ** 2)
            total_loss = total_loss + self.d2_weight * d2_loss
        
        # Non-negativity penalty
        if self.noneg_weight > 0:
            neg_penalty = torch.mean(F.relu(-f_pred) ** 2)
            total_loss = total_loss + self.noneg_weight * neg_penalty
        
        return total_loss


class SignalSparsityLoss(nn.Module):
    """
    Regularization that encourages the signal to have sparse derivatives.
    
    Chromatogram peaks have:
    1. Sparse first derivative (mostly flat, occasional jumps)
    2. Sparse second derivative (smooth peaks)
    """
    
    def __init__(
        self,
        d1_weight: float = 0.1,
        d2_weight: float = 0.1,
        eps: float = 1e-6
    ):
        super().__init__()
        self.d1_weight = d1_weight
        self.d2_weight = d2_weight
        self.eps = eps
    
    def forward(self, x_pred: torch.Tensor) -> torch.Tensor:
        if x_pred.dim() == 1:
            x_pred = x_pred.unsqueeze(0)
        
        total_loss = 0.0
        
        # L1 on first derivative (promotes sparsity)
        if self.d1_weight > 0:
            diff1 = x_pred[:, 1:] - x_pred[:, :-1]
            d1_loss = torch.mean(torch.sqrt(diff1 ** 2 + self.eps))  # Smooth L1
            total_loss = total_loss + self.d1_weight * d1_loss
        
        # L1 on second derivative
        if self.d2_weight > 0:
            diff2 = x_pred[:, 2:] - 2 * x_pred[:, 1:-1] + x_pred[:, :-2]
            d2_loss = torch.mean(torch.sqrt(diff2 ** 2 + self.eps))
            total_loss = total_loss + self.d2_weight * d2_loss
        
        return total_loss


class AsymmetricLoss(nn.Module):
    """
    Asymmetric loss that penalizes under-estimation more than over-estimation.
    
    This is important for chromatography where:
    - Under-estimating peak height loses analyte quantification
    - Over-estimating is less harmful (conservative)
    """
    
    def __init__(self, under_weight: float = 2.0, over_weight: float = 1.0):
        super().__init__()
        self.under_weight = under_weight
        self.over_weight = over_weight
    
    def forward(
        self,
        x_pred: torch.Tensor,
        x_true: torch.Tensor,
    ) -> torch.Tensor:
        if x_pred.dim() == 1:
            x_pred = x_pred.unsqueeze(0)
            x_true = x_true.unsqueeze(0)
        
        error = x_true - x_pred
        
        # Under-estimation: error > 0 (pred < true)
        under_mask = error > 0
        under_loss = self.under_weight * (error[under_mask] ** 2).sum()
        
        # Over-estimation: error < 0 (pred > true)
        over_mask = error < 0
        over_loss = self.over_weight * (error[over_mask] ** 2).sum()
        
        # Normalize
        n_elements = error.numel()
        total_loss = (under_loss + over_loss) / n_elements
        
        return total_loss


# ===========================================================================
# COMBINED LOSS FOR LBEADS-NET v2
# ===========================================================================

class LBEADSv2Loss(nn.Module):
    """
    Combined loss function for LBEADS-NET v2 training.
    
    Combines multiple loss terms:
    1. SNR loss for signal quality
    2. Peak-aware loss for preserving peak heights
    3. Baseline smoothness regularization
    4. Signal sparsity regularization
    
    Default weights are tuned for chromatogram baseline estimation.
    """
    
    def __init__(
        self,
        # Main loss weights
        signal_weight: float = 1.0,
        baseline_weight: float = 0.5,
        # Loss type options
        use_snr_loss: bool = True,
        use_peak_loss: bool = True,
        use_asymmetric: bool = True,
        # SNR loss weight (if used)
        snr_weight: float = 1.0,
        # Peak loss weight
        peak_weight: float = 0.5,
        peak_height_weight: float = 0.3,
        # Asymmetric loss weight
        asymmetric_weight: float = 0.2,
        # Regularization weights
        baseline_smooth_weight: float = 0.1,
        signal_sparse_weight: float = 0.05,
        # Advanced options
        eps: float = 1e-8
    ):
        super().__init__()
        
        self.signal_weight = signal_weight
        self.baseline_weight = baseline_weight
        self.use_snr_loss = use_snr_loss
        self.use_peak_loss = use_peak_loss
        self.use_asymmetric = use_asymmetric
        
        # Initialize loss components
        self.mse = nn.MSELoss()
        
        if use_snr_loss:
            self.snr_loss = SNRLoss(eps)
            self.snr_weight = snr_weight
        
        if use_peak_loss:
            self.peak_loss = PeakAwareLoss()
            self.peak_weight = peak_weight
            self.peak_height_loss = PeakHeightLoss()
            self.peak_height_weight = peak_height_weight
        
        if use_asymmetric:
            self.asymmetric_loss = AsymmetricLoss(under_weight=2.0, over_weight=1.0)
            self.asymmetric_weight = asymmetric_weight
        
        self.baseline_smooth = BaselineSmoothnessLoss()
        self.baseline_smooth_weight = baseline_smooth_weight
        
        self.signal_sparse = SignalSparsityLoss()
        self.signal_sparse_weight = signal_sparse_weight
    
    def forward(
        self,
        x_pred: torch.Tensor,
        f_pred: torch.Tensor,
        x_true: torch.Tensor,
        f_true: torch.Tensor,
        return_components: bool = False
    ) -> torch.Tensor:
        """
        Compute combined loss.
        
        Args:
            x_pred: Predicted signal
            f_pred: Predicted baseline
            x_true: Ground truth signal
            f_true: Ground truth baseline
            return_components: If True, also return individual loss components
        """
        losses = {}
        total_loss = 0.0
        
        # =====================================================================
        # SIGNAL LOSSES
        # =====================================================================
        
        # Base MSE on signal
        mse_signal = self.mse(x_pred, x_true)
        losses['mse_signal'] = mse_signal.item()
        total_loss = total_loss + self.signal_weight * mse_signal
        
        # SNR loss
        if self.use_snr_loss:
            snr = self.snr_loss(x_pred, x_true)
            losses['snr'] = -snr.item()  # Positive SNR for logging
            total_loss = total_loss + self.snr_weight * snr
        
        # Peak-aware loss
        if self.use_peak_loss:
            peak = self.peak_loss(x_pred, x_true)
            losses['peak_aware'] = peak.item()
            total_loss = total_loss + self.peak_weight * peak
            
            peak_height = self.peak_height_loss(x_pred, x_true)
            losses['peak_height'] = peak_height.item()
            total_loss = total_loss + self.peak_height_weight * peak_height
        
        # Asymmetric loss
        if self.use_asymmetric:
            asym = self.asymmetric_loss(x_pred, x_true)
            losses['asymmetric'] = asym.item()
            total_loss = total_loss + self.asymmetric_weight * asym
        
        # =====================================================================
        # BASELINE LOSSES
        # =====================================================================
        
        # MSE on baseline
        mse_baseline = self.mse(f_pred, f_true)
        losses['mse_baseline'] = mse_baseline.item()
        total_loss = total_loss + self.baseline_weight * mse_baseline
        
        # =====================================================================
        # REGULARIZATION LOSSES
        # =====================================================================
        
        # Baseline smoothness
        if self.baseline_smooth_weight > 0:
            smooth = self.baseline_smooth(f_pred)
            losses['baseline_smooth'] = smooth.item()
            total_loss = total_loss + self.baseline_smooth_weight * smooth
        
        # Signal sparsity
        if self.signal_sparse_weight > 0:
            sparse = self.signal_sparse(x_pred)
            losses['signal_sparse'] = sparse.item()
            total_loss = total_loss + self.signal_sparse_weight * sparse
        
        losses['total'] = total_loss.item()
        
        if return_components:
            return total_loss, losses
        return total_loss


# ===========================================================================
# UTILITY FUNCTIONS
# ===========================================================================

def compute_snr(x_true: torch.Tensor, x_pred: torch.Tensor, eps: float = 1e-8) -> float:
    """Compute SNR in dB."""
    if x_true.dim() == 1:
        x_true = x_true.unsqueeze(0)
        x_pred = x_pred.unsqueeze(0)
    
    signal_power = torch.sum(x_true ** 2, dim=1)
    noise_power = torch.sum((x_true - x_pred) ** 2, dim=1) + eps
    snr_db = 10 * torch.log10(signal_power / noise_power + eps)
    
    return snr_db.mean().item()


def compute_rmse(x_true: torch.Tensor, x_pred: torch.Tensor) -> float:
    """Compute RMSE."""
    return torch.sqrt(F.mse_loss(x_pred, x_true)).item()


def compute_peak_error(x_true: torch.Tensor, x_pred: torch.Tensor) -> float:
    """Compute mean relative error at peak locations."""
    if x_true.dim() == 1:
        x_true = x_true.unsqueeze(0)
        x_pred = x_pred.unsqueeze(0)
    
    # Find peaks (simple local max detection)
    threshold = torch.quantile(x_true, 0.8, dim=1, keepdim=True)
    peak_mask = x_true > threshold
    
    if peak_mask.sum() == 0:
        return 0.0
    
    # Relative error at peaks
    rel_error = torch.abs(x_true - x_pred) / (torch.abs(x_true) + 1e-8)
    peak_error = (rel_error * peak_mask.float()).sum() / peak_mask.sum()
    
    return peak_error.item()
