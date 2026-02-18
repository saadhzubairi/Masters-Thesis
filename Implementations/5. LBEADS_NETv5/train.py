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
    Generate realistic synthetic chromatogram data matching real LC/GC data.
    
    Real chromatogram characteristics (measured from 8 real signals, N=4000):
    - Peak heights: 10 to 2600+ (huge dynamic range), mean ~174
    - Peak widths (FWHM): 3 to 370+ samples, median ~10, mean ~16
    - Peaks per signal: 29 to 58
    - Peaks are often asymmetric (tailing) and overlapping in clusters
    - Baseline: smooth, slowly varying, amplitude 0-35
    - Noise std: ~5
    
    Signal model: y = x_true (peaks) + f_true (baseline) + noise
    """
    
    def __init__(self, N: int = 4096, seed: Optional[int] = None):
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
        amplitude_range: Tuple[float, float] = (2.0, 35.0),
        num_knots_range: Tuple[int, int] = (4, 8),
        drift_type: Optional[str] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Generate a smooth, realistic chromatogram baseline.
        
        Real baselines are smooth, slowly varying, mostly positive,
        with amplitude 0-35 and gentle broad humps.
        
        Uses cubic spline interpolation of random knot points for
        realistic smooth baseline shapes.
        """
        from scipy.interpolate import CubicSpline
        
        meta = {}
        
        # Choose baseline type
        if drift_type is None:
            drift_type = self.rng.choice(['spline', 'exponential_decay', 'mixed'])
        meta['drift_type'] = drift_type
        
        indices = np.arange(self.N, dtype=float)
        
        if drift_type == 'spline':
            # Smooth spline through random knot points
            num_knots = int(self.rng.integers(num_knots_range[0], num_knots_range[1] + 1))
            knot_x = np.sort(self.rng.uniform(0, self.N, num_knots))
            knot_x = np.concatenate([[0], knot_x, [self.N - 1]])  # Include endpoints
            
            amp = float(self.rng.uniform(amplitude_range[0], amplitude_range[1]))
            knot_y = self.rng.uniform(0, amp, len(knot_x))
            # Ensure endpoints are low (common in chromatograms)
            knot_y[0] = float(self.rng.uniform(0, amp * 0.3))
            knot_y[-1] = float(self.rng.uniform(0, amp * 0.3))
            
            cs = CubicSpline(knot_x, knot_y, bc_type='natural')
            baseline = cs(indices)
            meta['num_knots'] = num_knots
            meta['amplitude'] = amp
            
        elif drift_type == 'exponential_decay':
            # Exponential decay + gentle hump (common in LC)
            amp = float(self.rng.uniform(amplitude_range[0], amplitude_range[1]))
            decay_rate = float(self.rng.uniform(0.5, 3.0))
            baseline = amp * np.exp(-decay_rate * self.t)
            
            # Add a broad Gaussian hump
            hump_center = float(self.rng.uniform(0.2, 0.8)) * self.N
            hump_sigma = float(self.rng.uniform(0.15, 0.4)) * self.N
            hump_amp = float(self.rng.uniform(0.5, max(1.0, amp * 0.5)))
            baseline += hump_amp * np.exp(-((indices - hump_center) ** 2) / (2.0 * hump_sigma ** 2))
            meta['decay_rate'] = decay_rate
            meta['amplitude'] = amp
            
        else:  # mixed: polynomial + sine
            # Low-order polynomial drift + very low-freq sine
            amp = float(self.rng.uniform(amplitude_range[0], amplitude_range[1]))
            degree = int(self.rng.integers(2, 4))
            coeffs = self.rng.uniform(-1, 1, degree + 1)
            centered_t = self.t - 0.5
            poly = np.zeros(self.N)
            for i, c in enumerate(coeffs):
                poly += c * (centered_t ** i)
            # Normalize to [0, 1] then scale
            poly = (poly - poly.min()) / (poly.max() - poly.min() + 1e-10)
            baseline = poly * amp
            
            # Add 1-2 very low-frequency sine components
            num_sines = int(self.rng.integers(1, 3))
            for _ in range(num_sines):
                freq = float(self.rng.uniform(0.3, 2.0))
                s_amp = float(self.rng.uniform(0.5, max(1.5, amp * 0.3)))
                phase = float(self.rng.uniform(0, 2 * np.pi))
                baseline += s_amp * np.sin(2 * np.pi * freq * self.t + phase)
            
            meta['amplitude'] = amp
            meta['poly_degree'] = degree
        
        # Ensure baseline is mostly non-negative (shift if needed)
        if baseline.min() < -2.0:
            baseline -= baseline.min() - float(self.rng.uniform(0, 2.0))
        
        meta['final_range'] = (float(baseline.min()), float(baseline.max()))
        return baseline, meta
    
    def _emg_peak(self, indices: np.ndarray, center: float, sigma: float,
                  amplitude: float, tau: float) -> np.ndarray:
        """
        Asymmetric chromatographic peak using bi-Gaussian model.
        
        This is a numerically stable alternative to the Exponentially Modified
        Gaussian (EMG). Uses different widths for left and right halves:
        - sigma_left = sigma (leading edge)
        - sigma_right = sigma + tau (trailing edge / tailing)
        
        Args:
            indices: Sample indices
            center: Peak center (mu)
            sigma: Base Gaussian width
            amplitude: Peak height
            tau: Extra width on right side (0 = symmetric Gaussian)
        """
        sigma_left = sigma
        sigma_right = sigma + tau  # tau adds width to the right (tailing) side
        
        peak = np.zeros_like(indices, dtype=float)
        left_mask = indices <= center
        right_mask = indices > center
        
        peak[left_mask] = amplitude * np.exp(
            -((indices[left_mask] - center) ** 2) / (2.0 * sigma_left ** 2)
        )
        peak[right_mask] = amplitude * np.exp(
            -((indices[right_mask] - center) ** 2) / (2.0 * sigma_right ** 2)
        )
        
        return peak
    
    def generate_peaks(
        self,
        num_peaks_range: Tuple[int, int] = (20, 55),
        sigma_range: Tuple[float, float] = (1.5, 25.0),
        amplitude_range: Tuple[float, float] = (15.0, 2500.0),
        cluster_prob: float = 0.4,
        tail_prob: float = 0.6,
        tail_tau_range: Tuple[float, float] = (1.0, 15.0),
    ) -> Tuple[np.ndarray, Dict]:
        """
        Generate realistic chromatographic peaks.
        
        Features matched to real data:
        - Wide range of widths: FWHM 3-60+ samples (sigma 1.5-25)
        - Huge dynamic range: amplitude 15-2500
        - Log-normal amplitude distribution (few tall peaks, many small ones)
        - Peak clusters (overlapping peaks)
        - Asymmetric tailing (EMG model)
        - Always positive
        
        Args:
            num_peaks_range: Min/max number of peaks
            sigma_range: Range of Gaussian sigma (width) in samples
            amplitude_range: Range of peak amplitudes
            cluster_prob: Probability of generating a peak cluster
            tail_prob: Probability of a peak having asymmetric tail
            tail_tau_range: Range of tailing parameter
        """
        num_peaks = int(self.rng.integers(num_peaks_range[0], num_peaks_range[1] + 1))
        
        x_true = np.zeros(self.N)
        peak_info: List[Dict] = []
        indices = np.arange(self.N, dtype=float)
        
        margin = int(0.02 * self.N)  # 2% margin from edges
        
        # Generate peak centers - mix of random placement and clusters
        centers = []
        i = 0
        while i < num_peaks:
            if self.rng.random() < cluster_prob and i > 0 and num_peaks - i >= 2:
                # Generate a cluster of 2-4 peaks near a random position
                cluster_size = min(int(self.rng.integers(2, 5)), num_peaks - i)
                cluster_center = int(self.rng.integers(margin, self.N - margin))
                cluster_spread = float(self.rng.uniform(15, 80))
                for j in range(cluster_size):
                    c = cluster_center + int(self.rng.normal(0, cluster_spread))
                    c = np.clip(c, margin, self.N - margin - 1)
                    centers.append(int(c))
                i += cluster_size
            else:
                # Random isolated peak
                centers.append(int(self.rng.integers(margin, self.N - margin)))
                i += 1
        
        centers = sorted(centers[:num_peaks])
        
        for center in centers:
            # Width: log-uniform distribution favoring narrower peaks (matching real data)
            # Real data: median FWHM ~10, so median sigma ~4.2
            log_sigma = float(self.rng.uniform(
                np.log(sigma_range[0]), np.log(sigma_range[1])
            ))
            sigma = np.exp(log_sigma)
            
            # Amplitude: log-normal distribution (few very tall peaks, many shorter ones)
            # Real data: heights 10-2600, mean ~174
            log_amp = float(self.rng.uniform(
                np.log(amplitude_range[0]), np.log(amplitude_range[1])
            ))
            amplitude = np.exp(log_amp)
            
            # Occasionally generate a very tall "dominant" peak
            if self.rng.random() < 0.08:
                amplitude = float(self.rng.uniform(1000.0, amplitude_range[1]))
            
            # Tailing (EMG)
            tau = 0.0
            if self.rng.random() < tail_prob:
                tau = float(self.rng.uniform(tail_tau_range[0], tail_tau_range[1]))
                # Scale tau with sigma for consistent shape
                tau = tau * sigma / 5.0
            
            peak = self._emg_peak(indices, float(center), sigma, amplitude, tau)
            x_true += peak
            
            peak_info.append({
                "center": center,
                "sigma": float(sigma),
                "fwhm": float(2.355 * sigma),
                "amplitude": float(amplitude),
                "tau": float(tau),
            })
        
        params = {
            "num_peaks": len(centers),
            "peaks": peak_info,
        }
        
        return x_true, params
    
    def generate_noise(
        self, 
        noise_level: float = 3.0
    ) -> Tuple[np.ndarray, Dict]:
        """
        Generate additive Gaussian noise.
        
        Real chromatogram noise has std ~5.
        We use 2-6 range to cover various instrument noise levels.
        """
        noise = self.rng.normal(0.0, noise_level, self.N)
        params = {"noise_level": float(noise_level)}
        return noise, params
    
    def generate_signal(
        self,
        noise_level: float = 3.0,
        baseline_peak_ratio_range: Tuple[float, float] = (0.01, 0.5),
    ) -> SyntheticSignal:
        """
        Generate a complete synthetic chromatogram signal.

        The baseline is rescaled per sample so max(|f_true|)/max(|x_true|)
        spans a broad range. This prevents training from collapsing to
        peak-dominant synthetic regimes where baseline is always tiny.
        
        Returns:
            SyntheticSignal with y, x_true, f_true, noise, and metadata
        """
        x_true, peak_meta = self.generate_peaks()
        f_true, baseline_meta = self.generate_baseline()

        # Match a target baseline/peak strength ratio in log-space.
        ratio_min, ratio_max = baseline_peak_ratio_range
        ratio_min = max(float(ratio_min), 1e-6)
        ratio_max = max(float(ratio_max), ratio_min + 1e-6)
        target_ratio = float(np.exp(self.rng.uniform(np.log(ratio_min), np.log(ratio_max))))

        peak_scale = float(np.max(np.abs(x_true)))
        baseline_scale = float(np.max(np.abs(f_true)))
        baseline_rescale = 1.0
        if peak_scale > 1e-8 and baseline_scale > 1e-8:
            baseline_rescale = (target_ratio * peak_scale) / baseline_scale
            f_true = f_true * baseline_rescale

        noise, noise_meta = self.generate_noise(noise_level)
        
        # Observed signal: peaks + baseline + noise
        y = x_true + f_true + noise

        achieved_ratio = float(np.max(np.abs(f_true)) / (peak_scale + 1e-8))
        baseline_meta["target_peak_ratio"] = target_ratio
        baseline_meta["achieved_peak_ratio"] = achieved_ratio
        baseline_meta["rescale_factor"] = float(baseline_rescale)
        
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
        noise_level_range: Tuple[float, float] = (2.0, 6.0),
    ) -> List[SyntheticSignal]:
        """
        Generate a dataset of synthetic chromatogram signals.
        
        Args:
            n_samples: Number of samples to generate
            noise_level_range: Range of noise standard deviations
        
        Returns:
            List of SyntheticSignal objects
        """
        dataset = []
        for _ in range(n_samples):
            noise_level = float(self.rng.uniform(noise_level_range[0], noise_level_range[1]))
            signal = self.generate_signal(noise_level=noise_level)
            dataset.append(signal)
        return dataset


def create_train_test_split(
    dataset: List[SyntheticSignal],
    train_ratio: float = 0.8,
    seed: Optional[int] = None,
    normalize: bool = True,
) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Create 80/20 train/test split from synthetic dataset.
    
    Per-sample normalization: each signal is divided by max(|y|) so the
    model always sees inputs in a consistent amplitude range. This is
    CRITICAL because the learnable thresholds work at a single scale.
    
    Args:
        dataset: List of SyntheticSignal objects
        train_ratio: Fraction for training (default 0.8)
        seed: Random seed for shuffling
        normalize: If True, normalize each sample by max(|y|)
    
    Returns:
        (train_y, train_x_true, train_f_true), (test_y, test_x_true, test_f_true)
        where train_x_true is ground truth peaks, train_f_true is ground truth baseline
    """
    rng = np.random.default_rng(seed)
    indices = np.arange(len(dataset))
    rng.shuffle(indices)
    
    split_idx = int(len(dataset) * train_ratio)
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    def stack_and_normalize(idxs):
        ys = np.stack([dataset[i].y for i in idxs])
        xs = np.stack([dataset[i].x_true for i in idxs])
        fs = np.stack([dataset[i].f_true for i in idxs])
        if normalize:
            # Per-sample normalization: divide by max(|y|) per row
            scales = np.max(np.abs(ys), axis=1, keepdims=True)
            scales = np.maximum(scales, 1e-8)  # avoid div by zero
            ys = ys / scales
            xs = xs / scales
            fs = fs / scales
        return ys, xs, fs
    
    train_y, train_x_true, train_f_true = stack_and_normalize(train_indices)
    test_y, test_x_true, test_f_true = stack_and_normalize(test_indices)
    
    # Convert to tensors
    train_y_tensor = torch.tensor(train_y, dtype=torch.float64)
    train_x_tensor = torch.tensor(train_x_true, dtype=torch.float64)
    train_f_tensor = torch.tensor(train_f_true, dtype=torch.float64)
    test_y_tensor = torch.tensor(test_y, dtype=torch.float64)
    test_x_tensor = torch.tensor(test_x_true, dtype=torch.float64)
    test_f_tensor = torch.tensor(test_f_true, dtype=torch.float64)
    
    return (train_y_tensor, train_x_tensor, train_f_tensor), (test_y_tensor, test_x_tensor, test_f_tensor)


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
    6. Baseline reconstruction - match ground truth baseline (FIXES LEAKAGE!)
    7. Peak-baseline orthogonality - penalize correlation between peaks and baseline
    8. Baseline TV (3rd derivative) - penalize high-freq content in baseline
    
    The key insight: chromatogram peaks are:
    - SPARSE: Most of the signal is zero (only peaks have values)
    - POSITIVE: Peaks are always positive
    - SHARP: Peaks have steep edges (high gradient at boundaries, flat elsewhere)
    
    The baseline should be:
    - SMOOTH: No high-frequency content
    - MATCH GROUND TRUTH: Supervised learning on baseline too!
    - ORTHOGONAL to peaks: No bumps where peaks exist!
    """
    
    def __init__(self, 
                 alpha_mse: float = 1.0,          # Peak reconstruction weight
                 alpha_l1: float = 0.01,          # L1 sparsity on peaks
                 alpha_tv: float = 0.01,          # Total Variation on peaks
                 alpha_smooth: float = 0.01,      # Baseline smoothness (2nd deriv)
                 alpha_neg: float = 0.1,          # Non-negativity penalty
                 alpha_baseline: float = 1.0,     # Baseline reconstruction weight
                 alpha_ortho: float = 0.5,        # Peak-baseline orthogonality weight
                 alpha_baseline_tv: float = 0.1,  # Baseline 3rd derivative penalty
                 use_huber: bool = True,          # Use Huber loss instead of MSE
                 huber_delta: float = 1.0):       # Huber delta parameter
        super(SparsityLoss, self).__init__()
        self.alpha_mse = alpha_mse
        self.alpha_l1 = alpha_l1
        self.alpha_tv = alpha_tv
        self.alpha_smooth = alpha_smooth
        self.alpha_neg = alpha_neg
        self.alpha_baseline = alpha_baseline
        self.alpha_ortho = alpha_ortho
        self.alpha_baseline_tv = alpha_baseline_tv
        self.use_huber = use_huber
        self.huber_delta = huber_delta
        
        if use_huber:
            self.huber = nn.HuberLoss(reduction='mean', delta=huber_delta)
    
    def forward(self, x_pred: torch.Tensor, x_target: torch.Tensor,
                f_pred: Optional[torch.Tensor] = None,
                f_target: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total loss with breakdown.
        
        Args:
            x_pred: Predicted peaks (batch, N) or (N,)
            x_target: Ground truth peaks (batch, N) or (N,)
            f_pred: Predicted baseline (optional)
            f_target: Ground truth baseline (optional) - for baseline supervision!
            
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
        
        # 4. BASELINE SMOOTHNESS - penalize non-smooth baselines (2nd derivative)
        if self.alpha_smooth > 0 and f_pred is not None:
            if f_pred.dim() == 1:
                f_pred_batch = f_pred.unsqueeze(0)
            else:
                f_pred_batch = f_pred
            # Second derivative penalty (curvature)
            diff2 = f_pred_batch[:, 2:] - 2 * f_pred_batch[:, 1:-1] + f_pred_batch[:, :-2]
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
        
        # 6. BASELINE RECONSTRUCTION LOSS - match ground truth baseline!
        # This is the KEY fix for baseline leakage!
        if self.alpha_baseline > 0 and f_pred is not None and f_target is not None:
            if f_pred.dim() == 1:
                f_pred_batch = f_pred.unsqueeze(0)
            else:
                f_pred_batch = f_pred
            if f_target.dim() == 1:
                f_target_batch = f_target.unsqueeze(0)
            else:
                f_target_batch = f_target
            
            if self.use_huber:
                baseline_loss = self.huber(f_pred_batch, f_target_batch)
            else:
                baseline_loss = torch.mean((f_pred_batch - f_target_batch) ** 2)
            loss_dict['baseline_recon'] = baseline_loss.item()
            total_loss = total_loss + self.alpha_baseline * baseline_loss
        
        # 7. PEAK-BASELINE ORTHOGONALITY - penalize baseline gradient at peak locations
        # Where peaks are large, baseline should NOT have slope (should be flat there)
        # This directly fights leakage: if x has a peak, f should be smooth there
        if self.alpha_ortho > 0 and f_pred is not None:
            if f_pred.dim() == 1:
                f_pred_batch = f_pred.unsqueeze(0)
            else:
                f_pred_batch = f_pred
            
            # Compute first derivative of baseline (should be ~0 at peak locations)
            f_diff1 = f_pred_batch[:, 1:] - f_pred_batch[:, :-1]
            
            # Weight by peak magnitude: penalize baseline changes MORE where peaks exist
            # Use absolute peak values as weights (detached so we don't affect peak gradients)
            peak_weights = torch.abs(x_pred[:, :-1]).detach()
            # Normalize weights to avoid scale issues
            peak_weights = peak_weights / (peak_weights.max() + 1e-8)
            
            # Penalize baseline gradient at peak locations
            ortho_loss = torch.mean(peak_weights * f_diff1 ** 2)
            loss_dict['peak_baseline_ortho'] = ortho_loss.item()
            total_loss = total_loss + self.alpha_ortho * ortho_loss
        
        # 8. BASELINE TOTAL VARIATION (3rd derivative) - penalize high-freq in baseline
        # The baseline should be ultra-smooth; any rapid changes are leakage
        # Third derivative catches localized bumps that 2nd derivative misses
        if self.alpha_baseline_tv > 0 and f_pred is not None:
            if f_pred.dim() == 1:
                f_pred_batch = f_pred.unsqueeze(0)
            else:
                f_pred_batch = f_pred
            
            diff3 = (f_pred_batch[:, 3:] - 3 * f_pred_batch[:, 2:-1] 
                     + 3 * f_pred_batch[:, 1:-2] - f_pred_batch[:, :-3])
            baseline_tv_loss = torch.mean(diff3 ** 2)
            loss_dict['baseline_tv'] = baseline_tv_loss.item()
            total_loss = total_loss + self.alpha_baseline_tv * baseline_tv_loss
        
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
                     train_f_true: Optional[torch.Tensor] = None,
                     num_epochs: int = 22,
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
        train_f_true: Training ground truth baselines (num_samples, N) - NEW for baseline supervision!
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        batch_size: Batch size
        device: Device to train on
        verbose: Print progress
        loss_config: Dictionary with loss weights (alpha_mse, alpha_l1, alpha_tv, alpha_baseline, etc.)
    
    Returns:
        loss_history: List of total training losses
        loss_details: List of loss component dictionaries
    """
    model = model.to(device)
    model.train()
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    
    # Configure loss function with sparsity penalties + baseline reconstruction
    if loss_config is None:
        loss_config = {
            'alpha_mse': 1.0,          # Peak reconstruction (main objective)
            'alpha_l1': 0.01,          # L1 sparsity on peaks
            'alpha_tv': 0.01,          # Total Variation
            'alpha_smooth': 2.0,       # Baseline smoothness 2nd deriv
            'alpha_neg': 0.5,          # Non-negativity
            'alpha_baseline': 10.0,    # Baseline reconstruction - VERY STRONG
            'alpha_ortho': 1.0,        # Peak-baseline orthogonality
            'alpha_baseline_tv': 0.5,  # Baseline 3rd derivative penalty
            'use_huber': True,
            'huber_delta': 0.1
        }
    
    criterion = SparsityLoss(**loss_config)
    
    num_samples = train_y.shape[0]
    loss_history = []
    loss_details = []
    last_epoch_time = 0
    
    # Check if we have baseline ground truth for supervision
    has_baseline_supervision = train_f_true is not None
    if has_baseline_supervision and verbose:
        print("  Using baseline supervision (f_true provided)")
    
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
            
            # Get baseline ground truth if available
            f_true_batch = None
            if has_baseline_supervision:
                f_true_batch = train_f_true[batch_indices].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass: model predicts peaks (x) and baseline (f) from observed y
            x_pred, f_pred = model(y_batch)
            
            # Loss with sparsity penalties AND baseline supervision
            loss, loss_dict = criterion(x_pred, x_true_batch, f_pred, f_true_batch)
            
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
                  f"baseline={avg_loss_dict.get('baseline_recon', 0):.4f}, "
                  f"ortho={avg_loss_dict.get('peak_baseline_ortho', 0):.4f}, "
                  f"bltv={avg_loss_dict.get('baseline_tv', 0):.6f}, "
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
    N = 4096           # Signal length
    n_samples = 500    # Total number of samples (more diversity)
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
        noise_level_range=(2.0, 6.0),  # Realistic noise (real data has std ~5)
    )

    peak_baseline_ratios = np.array([
        np.max(np.abs(s.f_true)) / (np.max(np.abs(s.x_true)) + 1e-8)
        for s in dataset
    ])
    print("  Baseline/peak max-ratio stats:")
    print(f"    mean={peak_baseline_ratios.mean():.4f}, median={np.median(peak_baseline_ratios):.4f}, "
          f"p10={np.percentile(peak_baseline_ratios, 10):.4f}, "
          f"p90={np.percentile(peak_baseline_ratios, 90):.4f}")
    
    # Create train/test split (80/20) - now includes baseline ground truth!
    print("\nCreating train/test split...")
    (train_y, train_x_true, train_f_true), (test_y, test_x_true, test_f_true) = create_train_test_split(
        dataset, train_ratio=train_ratio, seed=seed
    )
    
    print(f"  Training samples: {train_y.shape[0]}")
    print(f"  Test samples: {test_y.shape[0]}")
    print(f"  Signal length: {train_y.shape[1]}")
    print(f"  Per-sample normalization: enabled (all signals scaled to unit max)")
    print(f"  Baseline supervision enabled: True")
    
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
        fc=0.006,  # Match classical BEADS cutoff (0.6% of Nyquist)
        num_layers=20,  # Sufficient for refinement (x starts from highpass init)
        init_lam0=0.01, # Small threshold - x is already initialized well
        init_lam1=0.5,  # Moderate: don't kill wide peaks during refinement
        init_lam2=0.5,  # Moderate: same reasoning
        init_r=6.0,     # Asymmetry ratio (penalize negative 6x more)
        init_step_size=0.05,  # Moderate step size for stable refinement
        lowpass_iterations=1,  # Single-pass lowpass (same as classical BEADS)
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
    
    # Configure sparsity-based loss function with BASELINE SUPERVISION
    # With per-sample normalization all signals are in [0, 1] range
    loss_config = {
        'alpha_mse': 1.0,           # Main reconstruction objective (peaks)
        'alpha_l1': 0.01,           # L1 sparsity (for unit-scale signals)
        'alpha_tv': 0.01,           # Total Variation
        'alpha_smooth': 2.0,        # Baseline smoothness 2nd deriv
        'alpha_neg': 0.5,           # Non-negativity - peaks are positive
        'alpha_baseline': 10.0,     # Baseline reconstruction - VERY STRONG to prevent leakage!
        'alpha_ortho': 1.0,         # Peak-baseline orthogonality - penalize baseline slope at peaks
        'alpha_baseline_tv': 0.5,   # Baseline 3rd derivative smoothness - catch localized bumps
        'use_huber': True,          # Huber loss is more robust than MSE
        'huber_delta': 0.1          # Appropriate for unit-scale signals
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
        train_f_true,  # Baseline ground truth for supervision
        num_epochs=50,           # Sufficient for normalized data
        learning_rate=5e-3,      # Moderate LR
        batch_size=4,            # Slightly larger batch for normalized data
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
    plt.plot(epochs, [d.get('reconstruction', 0) for d in loss_details], label='Peak Recon')
    plt.plot(epochs, [d.get('baseline_recon', 0) for d in loss_details], label='Baseline Recon')
    plt.plot(epochs, [d.get('l1_sparsity', 0) for d in loss_details], label='L1 Sparsity')
    plt.plot(epochs, [d.get('total_variation', 0) for d in loss_details], label='Total Variation')
    plt.plot(epochs, [d.get('peak_baseline_ortho', 0) for d in loss_details], label='Ortho')
    plt.plot(epochs, [d.get('baseline_tv', 0) for d in loss_details], label='Baseline TV')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Component')
    plt.title('Loss Components')
    plt.legend(fontsize=8)
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
        f_true_np = test_f_true[test_idx].numpy()
    
    plt.plot(y_np, 'gray', alpha=0.5, linewidth=0.5, label='Observed')
    plt.plot(x_np, 'b', linewidth=1, label='Predicted Peaks')
    plt.plot(x_true_np, 'g--', linewidth=1, label='Ground Truth Peaks')
    plt.plot(f_np, 'r', linewidth=1, alpha=0.7, label='Predicted Baseline')
    plt.plot(f_true_np, 'm--', linewidth=1, alpha=0.7, label='True Baseline')
    plt.legend(fontsize=7)
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
        f_true_np = test_f_true[test_idx].numpy()
    
    plt.plot(y_np, 'gray', alpha=0.5, linewidth=0.5, label='Observed')
    plt.plot(x_np, 'b', linewidth=1, label='Predicted Peaks')
    plt.plot(x_true_np, 'g--', linewidth=1, label='Ground Truth Peaks')
    plt.plot(f_np, 'r', linewidth=1, alpha=0.7, label='Predicted Baseline')
    plt.plot(f_true_np, 'm--', linewidth=1, alpha=0.7, label='True Baseline')
    plt.legend(fontsize=7)
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
    
    plt.suptitle('LBEADS-NET Training Results (Baseline Supervision)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, 'training_results.png'), dpi=150)
    print(f"\nSaved results to {os.path.join(script_dir, 'training_results.png')}")
    
    # Save model
    model_path = os.path.join(script_dir, f'lbeads_net_baseline_fix_{int(time.time())}.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'N': N,
            'd': 1,
            'fc': 0.006,
            'num_layers': 20,
            'lowpass_iterations': 1
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
