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
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import random
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass

from lbeads_net import (
    LBEADS_NET,
    apply_highpass_filter,
    apply_highpass_filter_np,
    apply_lowpass_filter_np,
    beads_classic_with_init,
    compute_lowpass_matrix_np,
)


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
    Generate BEADS-aligned synthetic data with derivative-sparse peaks.

    Signal model:
        y = x_true + f_true + noise

    where:
        x_true: sparse, piecewise peaks (BEADS hypothesis aligned)
        f_true: ultra-smooth baseline
        noise: low-amplitude Gaussian noise
    """
    
    def __init__(self, N: int = 4096, seed: Optional[int] = None, peak_shape_mode: str = 'linear'):
        """
        Args:
            N: Signal length
            seed: Random seed for reproducibility
            peak_shape_mode: 'linear', 'exp', or 'mixed'
        """
        self.N = N
        self.t = np.linspace(0.0, 1.0, N)
        self.rng = np.random.default_rng(seed)
        if peak_shape_mode not in ('linear', 'exp', 'mixed'):
            raise ValueError("peak_shape_mode must be one of: 'linear', 'exp', 'mixed'")
        self.peak_shape_mode = peak_shape_mode

    @staticmethod
    def beads_peak(N: int, center: int, amplitude: float, rise_w: int, decay_w: int, plateau_w: int) -> np.ndarray:
        """Piecewise-linear derivative-sparse peak (spec-aligned construction)."""
        x = np.zeros(N, dtype=np.float64)

        # Rise segment
        start = center - rise_w
        for i in range(start, center):
            if 0 <= i < N:
                x[i] = amplitude * (i - start) / max(rise_w, 1)

        # Plateau (optional)
        for i in range(center, center + plateau_w):
            if 0 <= i < N:
                x[i] = amplitude

        # Decay segment
        end = center + plateau_w + decay_w
        for i in range(center + plateau_w, end):
            if 0 <= i < N:
                x[i] = amplitude * (1 - (i - (center + plateau_w)) / max(decay_w, 1))

        return x

    @staticmethod
    def beads_exp_peak(N: int, center: int, amplitude: float, rise_tau: float, decay_tau: float) -> np.ndarray:
        """Asymmetric exponential peak (Gaussian-free alternative)."""
        t = np.arange(N, dtype=np.float64)
        x = np.zeros(N, dtype=np.float64)

        left = t <= center
        right = t > center

        rt = max(float(rise_tau), 1e-6)
        dt = max(float(decay_tau), 1e-6)

        # Continuous at apex: x(center) ~= amplitude.
        x[left] = amplitude * np.exp(-(center - t[left]) / rt)
        x[right] = amplitude * np.exp(-(t[right] - center) / dt)
        return x
    
    def generate_baseline(
        self,
        smooth_sigma: float = 100.0,
        sine_amp: float = 0.1,
        sine_freq_range: Tuple[float, float] = (0.5, 2.0),
        baseline_amp_range: Tuple[float, float] = (0.08, 0.35),
    ) -> Tuple[np.ndarray, Dict]:
        """
        Generate ultra-smooth baseline using low-order polynomial + low-freq sine,
        then aggressively smooth with Gaussian filtering.
        """
        from scipy.ndimage import gaussian_filter1d

        coeffs = self.rng.uniform(-0.5, 0.5, size=3)
        baseline = coeffs[0] + coeffs[1] * self.t + coeffs[2] * (self.t ** 2)

        sine_freq = float(self.rng.uniform(sine_freq_range[0], sine_freq_range[1]))
        sine_phase = float(self.rng.uniform(0.0, 2.0 * np.pi))
        baseline = baseline + float(sine_amp) * np.sin(2.0 * np.pi * self.t * sine_freq + sine_phase)

        baseline = gaussian_filter1d(baseline, sigma=float(smooth_sigma), mode='nearest')

        # Normalize shape then apply controlled amplitude.
        bmax = float(np.max(np.abs(baseline)))
        if bmax > 1e-12:
            baseline = baseline / bmax
        baseline_amp = float(self.rng.uniform(baseline_amp_range[0], baseline_amp_range[1]))
        baseline = baseline * baseline_amp

        # Shift to mostly-positive drift as in chromatography.
        offset = float(self.rng.uniform(0.0, 0.12))
        baseline = baseline - float(np.min(baseline)) + offset

        diff3 = np.diff(baseline, n=3)
        meta = {
            "poly_coeffs": coeffs.tolist(),
            "sine_freq": sine_freq,
            "sine_phase": sine_phase,
            "smooth_sigma": float(smooth_sigma),
            "baseline_amp": baseline_amp,
            "tv3_energy": float(np.mean(diff3 ** 2)) if diff3.size > 0 else 0.0,
            "final_range": (float(baseline.min()), float(baseline.max())),
        }
        return baseline.astype(np.float64), meta
    
    def generate_peaks(
        self,
        num_peaks_range: Tuple[int, int] = (2, 6),
        amplitude_range: Tuple[float, float] = (0.2, 1.0),
        rise_width_range: Tuple[int, int] = (10, 80),
        decay_width_range: Tuple[int, int] = (20, 200),
        plateau_width_range: Tuple[int, int] = (0, 10),
        center_margin: int = 200,
        peak_shape_mode: Optional[str] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Generate derivative-sparse peaks (linear/exponential, no Gaussian).
        """
        mode = self.peak_shape_mode if peak_shape_mode is None else peak_shape_mode
        if mode not in ('linear', 'exp', 'mixed'):
            raise ValueError("peak_shape_mode must be one of: 'linear', 'exp', 'mixed'")

        num_peaks = int(self.rng.integers(int(num_peaks_range[0]), int(num_peaks_range[1]) + 1))
        x_true = np.zeros(self.N, dtype=np.float64)
        peak_info: List[Dict] = []

        low = int(center_margin)
        high = self.N - int(center_margin) + 1
        if high <= low:
            low, high = 0, self.N
        centers = np.sort(self.rng.integers(low, high, size=num_peaks))

        for center in centers:
            center = int(center)
            amplitude = float(self.rng.uniform(amplitude_range[0], amplitude_range[1]))
            peak_kind = mode
            if mode == 'mixed':
                peak_kind = 'linear' if self.rng.random() < 0.75 else 'exp'

            if peak_kind == 'linear':
                rise_w = int(self.rng.integers(rise_width_range[0], rise_width_range[1] + 1))
                decay_w = int(self.rng.integers(decay_width_range[0], decay_width_range[1] + 1))
                plateau_w = int(self.rng.integers(plateau_width_range[0], plateau_width_range[1] + 1))
                peak = self.beads_peak(self.N, center, amplitude, rise_w, decay_w, plateau_w)
                peak_info.append({
                    "type": "linear",
                    "center": center,
                    "amplitude": amplitude,
                    "rise_width": rise_w,
                    "decay_width": decay_w,
                    "plateau_width": plateau_w,
                })
            else:
                rise_tau = float(self.rng.uniform(rise_width_range[0], rise_width_range[1]))
                decay_tau = float(self.rng.uniform(decay_width_range[0], decay_width_range[1]))
                peak = self.beads_exp_peak(self.N, center, amplitude, rise_tau, decay_tau)
                peak_info.append({
                    "type": "exp",
                    "center": center,
                    "amplitude": amplitude,
                    "rise_tau": rise_tau,
                    "decay_tau": decay_tau,
                })

            x_true += peak

        x_true = np.clip(x_true, a_min=0.0, a_max=None)
        peak_max = float(np.max(x_true))
        if peak_max > 1e-8:
            x_true = x_true / peak_max

        params = {
            "num_peaks": int(num_peaks),
            "shape_mode": mode,
            "peaks": peak_info,
            "peak_max_after_norm": float(np.max(x_true)),
        }
        return x_true.astype(np.float64), params
    
    def generate_noise(
        self, 
        noise_level: float = 0.01
    ) -> Tuple[np.ndarray, Dict]:
        """
        Generate small additive Gaussian noise.
        """
        noise = self.rng.normal(0.0, float(noise_level), self.N)
        params = {"noise_level": float(noise_level)}
        return noise.astype(np.float64), params
    
    def generate_signal(
        self,
        noise_level: float = 0.01,
    ) -> SyntheticSignal:
        """
        Generate a complete synthetic chromatogram signal.
        """
        x_true, peak_meta = self.generate_peaks()
        f_true, baseline_meta = self.generate_baseline()
        noise, noise_meta = self.generate_noise(noise_level)

        # Signal composition.
        y = x_true + f_true + noise

        # Per-sample normalization (spec).
        scale = float(np.max(np.abs(y)))
        scale = max(scale, 1e-8)
        y = y / scale
        x_true = x_true / scale
        f_true = f_true / scale
        noise = noise / scale

        d1 = np.diff(x_true)
        d2 = np.diff(d1)
        d1_thr = 1e-3
        d2_thr = 1e-3

        d1_nz_count = int(np.sum(np.abs(d1) > d1_thr))
        d2_nz_count = int(np.sum(np.abs(d2) > d2_thr))

        metadata = {
            "N": self.N,
            "signal_scale": scale,
            "baseline": baseline_meta,
            "peaks": peak_meta,
            "noise": noise_meta,
            "derivative_stats": {
                "mean_abs_diff1": float(np.mean(np.abs(d1))) if d1.size > 0 else 0.0,
                "mean_abs_diff2": float(np.mean(np.abs(d2))) if d2.size > 0 else 0.0,
                "diff1_nonzero_count": d1_nz_count,
                "diff2_nonzero_count": d2_nz_count,
                "diff1_nonzero_fraction": float(d1_nz_count / max(len(d1), 1)),
                "diff2_nonzero_fraction": float(d2_nz_count / max(len(d2), 1)),
                "curvature_sparsity": float(1.0 - (d2_nz_count / max(len(d2), 1))),
            },
        }

        return SyntheticSignal(
            y=y.astype(np.float64),
            x_true=x_true.astype(np.float64),
            f_true=f_true.astype(np.float64),
            noise=noise.astype(np.float64),
            metadata=metadata,
        )
    
    def generate_dataset(
        self,
        n_samples: int,
        noise_level_range: Tuple[float, float] = (0.01, 0.01),
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


def summarize_derivative_sparsity(
    dataset: List[SyntheticSignal],
    diff1_threshold: float = 1e-3,
    diff2_threshold: float = 1e-3,
) -> Dict[str, float]:
    """Aggregate derivative-sparsity diagnostics over a dataset."""
    if len(dataset) == 0:
        return {
            "mean_abs_diff1": float("nan"),
            "mean_abs_diff2": float("nan"),
            "mean_diff1_nonzero_count": float("nan"),
            "mean_diff2_nonzero_count": float("nan"),
            "mean_diff1_nonzero_fraction": float("nan"),
            "mean_diff2_nonzero_fraction": float("nan"),
            "mean_curvature_sparsity": float("nan"),
        }

    mean_abs_diff1 = []
    mean_abs_diff2 = []
    diff1_nonzero_counts = []
    diff2_nonzero_counts = []
    diff1_nonzero_fracs = []
    diff2_nonzero_fracs = []
    curvature_sparsities = []

    for sig in dataset:
        x = np.asarray(sig.x_true, dtype=np.float64)
        d1 = np.diff(x)
        d2 = np.diff(d1)

        d1_nz = int(np.sum(np.abs(d1) > diff1_threshold))
        d2_nz = int(np.sum(np.abs(d2) > diff2_threshold))
        d1_len = max(len(d1), 1)
        d2_len = max(len(d2), 1)

        mean_abs_diff1.append(float(np.mean(np.abs(d1))) if d1.size > 0 else 0.0)
        mean_abs_diff2.append(float(np.mean(np.abs(d2))) if d2.size > 0 else 0.0)
        diff1_nonzero_counts.append(d1_nz)
        diff2_nonzero_counts.append(d2_nz)
        diff1_nonzero_fracs.append(float(d1_nz / d1_len))
        diff2_nonzero_fracs.append(float(d2_nz / d2_len))
        curvature_sparsities.append(float(1.0 - (d2_nz / d2_len)))

    return {
        "mean_abs_diff1": float(np.mean(mean_abs_diff1)),
        "mean_abs_diff2": float(np.mean(mean_abs_diff2)),
        "mean_diff1_nonzero_count": float(np.mean(diff1_nonzero_counts)),
        "mean_diff2_nonzero_count": float(np.mean(diff2_nonzero_counts)),
        "mean_diff1_nonzero_fraction": float(np.mean(diff1_nonzero_fracs)),
        "mean_diff2_nonzero_fraction": float(np.mean(diff2_nonzero_fracs)),
        "mean_curvature_sparsity": float(np.mean(curvature_sparsities)),
    }


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


def run_lowpass_operator_diagnostic(N: int, d: int = 1, fc: float = 0.006, iterations: int = 1) -> Dict[str, float]:
    """
    Numerical sanity check: low-pass a smooth baseline and measure remaining HF energy.
    """
    rng = np.random.default_rng(0)
    smooth = np.zeros(N, dtype=np.float64)
    for _ in range(4):
        center = float(rng.uniform(0.1, 0.9) * N)
        sigma = float(rng.uniform(0.06, 0.22) * N)
        amp = float(rng.uniform(0.2, 1.0))
        idx = np.arange(N, dtype=np.float64)
        smooth += amp * np.exp(-((idx - center) ** 2) / (2.0 * sigma ** 2))
    smooth = smooth / (np.max(np.abs(smooth)) + 1e-8)

    lowpass_matrix = compute_lowpass_matrix_np(N, d=d, fc=fc)
    smooth_lp = apply_lowpass_filter_np(smooth, lowpass_matrix, iterations=max(1, int(iterations)))
    smooth_hf = apply_highpass_filter_np(smooth_lp, lowpass_matrix)

    hf_rms = float(np.sqrt(np.mean(smooth_hf ** 2)))
    sig_rms = float(np.sqrt(np.mean(smooth_lp ** 2)) + 1e-12)
    hf_ratio = float(hf_rms / sig_rms)
    diff3 = np.diff(smooth_lp, n=3)
    baseline_tv3 = float(np.mean(diff3 ** 2)) if diff3.size > 0 else 0.0

    return {
        'hf_rms': hf_rms,
        'hf_ratio': hf_ratio,
        'baseline_tv3': baseline_tv3,
    }


def run_classic_beads_mse_floor_check(
    dataset: List[SyntheticSignal],
    n_samples: int = 20,
    n_trials: int = 24,
    d: int = 1,
    fc: float = 0.006,
    Nit: int = 24,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Estimate a BEADS-achievable MSE floor by random-searching classical parameters.

    Each sample is normalized by max(|y|), matching training normalization.
    """
    if len(dataset) == 0:
        return {'mean_best_mse': float('nan'), 'median_best_mse': float('nan'), 'num_samples': 0}

    rng = np.random.default_rng(seed)
    n_eval = min(int(n_samples), len(dataset))
    sample_indices = rng.choice(len(dataset), size=n_eval, replace=False)

    trials = [
        {'lam0': 0.002, 'lam1': 0.3, 'lam2': 0.3, 'r': 6.0},
    ]
    for _ in range(max(1, int(n_trials) - 1)):
        trials.append({
            'lam0': float(np.exp(rng.uniform(np.log(5e-4), np.log(5e-2)))),
            'lam1': float(np.exp(rng.uniform(np.log(1e-2), np.log(2.0)))),
            'lam2': float(np.exp(rng.uniform(np.log(1e-2), np.log(2.0)))),
            'r': float(rng.uniform(2.0, 12.0)),
        })

    best_mses: List[float] = []
    for idx in sample_indices:
        sig = dataset[int(idx)]
        scale = float(np.max(np.abs(sig.y)) + 1e-8)
        y = np.asarray(sig.y, dtype=np.float64) / scale
        x_true = np.asarray(sig.x_true, dtype=np.float64) / scale

        best = float('inf')
        for p in trials:
            x_hat, _ = beads_classic_with_init(
                y,
                d=d,
                fc=fc,
                r=p['r'],
                lam0=p['lam0'],
                lam1=p['lam1'],
                lam2=p['lam2'],
                Nit=Nit,
            )
            mse = float(np.mean((x_hat - x_true) ** 2))
            if mse < best:
                best = mse
        best_mses.append(best)

    arr = np.asarray(best_mses, dtype=np.float64)
    return {
        'mean_best_mse': float(np.mean(arr)),
        'median_best_mse': float(np.median(arr)),
        'p10_best_mse': float(np.percentile(arr, 10)),
        'p90_best_mse': float(np.percentile(arr, 90)),
        'num_samples': int(n_eval),
        'num_trials': int(len(trials)),
    }


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
                 alpha_l1: float = 0.005,         # L1 sparsity on peaks
                 alpha_tv: float = 0.005,         # Total Variation on peaks
                 alpha_smooth: float = 0.01,      # Baseline smoothness (2nd deriv)
                 alpha_neg: float = 0.1,          # Non-negativity penalty
                 alpha_baseline: float = 2.0,     # Baseline reconstruction weight (masked to non-peak areas)
                 alpha_leakage: float = 1.0,      # High-frequency baseline leakage penalty at peak locations
                 alpha_ortho: float = 0.5,        # Peak-baseline orthogonality weight
                 alpha_baseline_tv: float = 0.0,  # Baseline 3rd derivative penalty
                 peak_mask_rel_threshold: float = 0.02,  # Peak mask threshold as fraction of max x_true
                 peak_mask_abs_min: float = 1e-4,        # Absolute floor for peak mask threshold
                 use_huber: bool = True,          # Use Huber loss instead of MSE
                 huber_delta: float = 1.0):       # Huber delta parameter
        super(SparsityLoss, self).__init__()
        self.alpha_mse = alpha_mse
        self.alpha_l1 = alpha_l1
        self.alpha_tv = alpha_tv
        self.alpha_smooth = alpha_smooth
        self.alpha_neg = alpha_neg
        self.alpha_baseline = alpha_baseline
        self.alpha_leakage = alpha_leakage
        self.alpha_ortho = alpha_ortho
        self.alpha_baseline_tv = alpha_baseline_tv
        self.peak_mask_rel_threshold = peak_mask_rel_threshold
        self.peak_mask_abs_min = peak_mask_abs_min
        self.use_huber = use_huber
        self.huber_delta = huber_delta
        
        if use_huber:
            self.huber = nn.HuberLoss(reduction='mean', delta=huber_delta)
    
    def forward(self, x_pred: torch.Tensor, x_target: torch.Tensor,
                f_pred: Optional[torch.Tensor] = None,
                f_target: Optional[torch.Tensor] = None,
                f_pred_highpass: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
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
        if f_pred is not None and f_pred.dim() == 1:
            f_pred = f_pred.unsqueeze(0)
        if f_target is not None and f_target.dim() == 1:
            f_target = f_target.unsqueeze(0)
        if f_pred_highpass is not None and f_pred_highpass.dim() == 1:
            f_pred_highpass = f_pred_highpass.unsqueeze(0)
        
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
        
        # Build peak/background masks from x_true.
        peak_amp = torch.max(torch.abs(x_target), dim=1, keepdim=True).values
        peak_thr = torch.clamp(
            peak_amp * self.peak_mask_rel_threshold,
            min=self.peak_mask_abs_min
        )
        peak_mask = (torch.abs(x_target) >= peak_thr).to(x_pred.dtype)
        bg_mask = 1.0 - peak_mask

        # 6. MASKED BASELINE RECONSTRUCTION - supervise baseline mostly off-peak.
        if self.alpha_baseline > 0 and f_pred is not None and f_target is not None:
            f_pred_batch = f_pred
            f_target_batch = f_target

            if self.use_huber:
                baseline_err = F.huber_loss(
                    f_pred_batch, f_target_batch, delta=self.huber_delta, reduction='none'
                )
            else:
                baseline_err = (f_pred_batch - f_target_batch) ** 2

            bg_denom = torch.sum(bg_mask) + 1e-8
            baseline_loss = torch.sum(baseline_err * bg_mask) / bg_denom
            loss_dict['baseline_recon'] = baseline_loss.item()
            loss_dict['background_fraction'] = float(torch.mean(bg_mask).item())
            total_loss = total_loss + self.alpha_baseline * baseline_loss

        # 6b. High-frequency leakage penalty:
        # baseline may be non-zero under peaks, but should avoid HF content there.
        if self.alpha_leakage > 0 and f_pred is not None:
            if f_pred_highpass is not None:
                peak_denom = torch.sum(peak_mask) + 1e-8
                leakage_loss = torch.sum((f_pred_highpass ** 2) * peak_mask) / peak_denom
            else:
                # Fallback if no operator high-pass signal was provided.
                f_pred_batch = f_pred
                if f_pred_batch.shape[1] >= 3:
                    f_diff2 = f_pred_batch[:, 2:] - 2 * f_pred_batch[:, 1:-1] + f_pred_batch[:, :-2]
                    peak_mask_mid = peak_mask[:, 1:-1]
                    peak_denom = torch.sum(peak_mask_mid) + 1e-8
                    leakage_loss = torch.sum((f_diff2 ** 2) * peak_mask_mid) / peak_denom
                else:
                    leakage_loss = torch.mean(f_pred_batch * 0.0)
            loss_dict['baseline_leakage'] = leakage_loss.item()
            total_loss = total_loss + self.alpha_leakage * leakage_loss
        
        # 7. PEAK-BASELINE ORTHOGONALITY - penalize baseline gradient at peak locations
        # Where peaks are large, baseline should NOT have slope (should be flat there)
        # This directly fights leakage: if x has a peak, f should be smooth there
        if self.alpha_ortho > 0 and f_pred is not None:
            f_pred_batch = f_pred
            
            # Compute first derivative of baseline (should be ~0 at peak locations)
            f_diff1 = f_pred_batch[:, 1:] - f_pred_batch[:, :-1]
            
            # Weight by true peak support so this term remains active even when x_pred collapses.
            peak_weights = torch.abs(x_target[:, :-1]).detach()
            peak_weights = peak_weights / (torch.amax(peak_weights, dim=1, keepdim=True) + 1e-8)
            
            # Penalize baseline gradient at peak locations
            ortho_loss = torch.mean(peak_weights * f_diff1 ** 2)
            loss_dict['peak_baseline_ortho'] = ortho_loss.item()
            total_loss = total_loss + self.alpha_ortho * ortho_loss
        
        # 8. BASELINE TOTAL VARIATION (3rd derivative) - penalize high-freq in baseline
        # The baseline should be ultra-smooth; any rapid changes are leakage
        # Third derivative catches localized bumps that 2nd derivative misses
        if self.alpha_baseline_tv > 0 and f_pred is not None:
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
                     test_y: Optional[torch.Tensor] = None,
                     test_x_true: Optional[torch.Tensor] = None,
                     test_f_true: Optional[torch.Tensor] = None,
                     num_epochs: int = 22,
                     learning_rate: float = 1e-3,
                     batch_size: int = 8,
                     device: str = 'cpu',
                     verbose: bool = True,
                     loss_config: Optional[Dict] = None,
                     stage_configs: Optional[List[Dict]] = None) -> Tuple[List[float], List[Dict]]:
    """
    Train LBEADS-NET model on synthetic data with sparsity-based loss.
    
    Args:
        model: LBEADS-NET model
        train_y: Training observed signals (num_samples, N) - peaks + baseline + noise
        train_x_true: Training ground truth peaks (num_samples, N)
        train_f_true: Training ground truth baselines (num_samples, N) - NEW for baseline supervision!
        test_y: Optional test observed signals for per-epoch test-loss logging
        test_x_true: Optional test peaks for per-epoch test-loss logging
        test_f_true: Optional test baselines for per-epoch test-loss logging
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        batch_size: Batch size
        device: Device to train on
        verbose: Print progress
        loss_config: Dictionary with loss weights (used if stage_configs is None)
        stage_configs: Optional staged training configs:
            [
              {'name': 'A', 'epochs': int, 'loss_config': {...}},
              {'name': 'B', 'epochs': int, 'loss_config': {...}}
            ]
    
    Returns:
        loss_history: List of total training losses
        loss_details: List of loss component dictionaries
    """
    model = model.to(device)
    model.train()
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    
    # Non-fast model defaults tuned for stability with shallow unrolling.
    default_loss_config = {
        'alpha_mse': 1.0,
        'alpha_l1': 0.01,
        'alpha_tv': 0.01,
        'alpha_smooth': 0.2,
        'alpha_neg': 2.0,
        'alpha_baseline': 0.5,
        'alpha_leakage': 0.5,
        'alpha_ortho': 0.2,
        'alpha_baseline_tv': 0.0,
        'peak_mask_rel_threshold': 0.02,
        'peak_mask_abs_min': 1e-4,
        'use_huber': False,
        'huber_delta': 0.1
    }

    if loss_config is None:
        loss_config = dict(default_loss_config)
    else:
        merged = dict(default_loss_config)
        merged.update(loss_config)
        loss_config = merged

    if stage_configs is None:
        stage_configs = [
            {
                'name': 'single',
                'epochs': int(num_epochs),
                'loss_config': dict(loss_config),
            }
        ]
    
    num_samples = train_y.shape[0]
    loss_history = []
    loss_details = []
    last_epoch_time = 0
    
    # Check if we have baseline ground truth for supervision
    has_baseline_supervision = train_f_true is not None
    if has_baseline_supervision and verbose:
        print("  Using baseline supervision (f_true provided)")

    has_test_data = (test_y is not None) and (test_x_true is not None)
    if has_test_data and verbose:
        print("  Per-epoch test-loss logging: enabled")
    
    total_epochs = int(sum(int(stage.get('epochs', 0)) for stage in stage_configs))
    global_epoch = 0
    last_train_loss = None
    last_test_loss = None

    def compute_loss_highpass(f_batch: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if f_batch is None:
            return None
        if not (hasattr(model, 'a_coeff') and hasattr(model, 'b_coeff')):
            return None
        cg_iters = int(getattr(model, 'lowpass_cg_iters', 32))
        if not model.training:
            cg_iters = max(cg_iters, 128)
        return apply_highpass_filter(
            f_batch,
            model.a_coeff,
            model.b_coeff,
            solve_cg_iters=cg_iters,
        )

    for stage_idx, stage in enumerate(stage_configs):
        stage_name = str(stage.get('name', f'stage_{stage_idx + 1}'))
        stage_epochs = int(stage.get('epochs', 0))
        stage_loss_cfg = dict(loss_config)
        stage_loss_cfg.update(stage.get('loss_config', {}))
        criterion = SparsityLoss(**stage_loss_cfg)

        if verbose:
            print("\n" + "-" * 60)
            print(f"Training Stage {stage_idx + 1}/{len(stage_configs)}: {stage_name}")
            print(f"  Stage epochs: {stage_epochs}")
            print(f"  Stage loss weights: alpha_mse={stage_loss_cfg['alpha_mse']}, "
                  f"alpha_l1={stage_loss_cfg['alpha_l1']}, alpha_tv={stage_loss_cfg['alpha_tv']}, "
                  f"alpha_baseline={stage_loss_cfg['alpha_baseline']}, alpha_leakage={stage_loss_cfg['alpha_leakage']}")
            print("-" * 60)

        for _ in range(stage_epochs):
            start_time = time.time()
            global_epoch += 1
            if verbose:
                last_train_str = "n/a" if last_train_loss is None else f"{last_train_loss:.6f}"
                last_test_str = "n/a" if last_test_loss is None else f"{last_test_loss:.6f}"
                print(
                    f"Epoch {global_epoch}/{total_epochs} ({stage_name}), "
                    f"Last epoch time: {last_epoch_time:.2f}s, "
                    f"Last train loss: {last_train_str}, Last test loss: {last_test_str}"
                )
            epoch_loss = 0.0
            epoch_loss_dict = {}
            num_batches = 0
            epoch_abs_x_pred_sum = 0.0
            epoch_abs_x_true_sum = 0.0
            epoch_abs_count = 0

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

                f_pred_highpass = None
                if criterion.alpha_leakage > 0:
                    f_pred_highpass = compute_loss_highpass(f_pred)

                # Loss with sparsity penalties AND baseline supervision
                loss, loss_dict = criterion(
                    x_pred,
                    x_true_batch,
                    f_pred,
                    f_true_batch,
                    f_pred_highpass=f_pred_highpass,
                )

                # Track per-epoch peak scale diagnostic.
                epoch_abs_x_pred_sum += torch.sum(torch.abs(x_pred)).item()
                epoch_abs_x_true_sum += torch.sum(torch.abs(x_true_batch)).item()
                epoch_abs_count += int(x_pred.numel())

                # Backward pass
                loss.backward()

                # Gentle clipping prevents rare spikes without freezing updates.
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

                optimizer.step()

                epoch_loss += loss.item()
                # Accumulate loss components
                for k, v in loss_dict.items():
                    epoch_loss_dict[k] = epoch_loss_dict.get(k, 0.0) + v
                num_batches += 1

            scheduler.step()

            avg_loss = epoch_loss / num_batches
            loss_history.append(avg_loss)
            last_train_loss = avg_loss
            mean_abs_x_pred = epoch_abs_x_pred_sum / max(epoch_abs_count, 1)
            mean_abs_x_true = epoch_abs_x_true_sum / max(epoch_abs_count, 1)

            # Optional per-epoch test loss with the same criterion used this stage.
            avg_test_loss = None
            if has_test_data:
                model.eval()
                with torch.no_grad():
                    test_epoch_loss = 0.0
                    test_batches = 0
                    for j in range(0, test_y.shape[0], batch_size):
                        test_slice = slice(j, min(j + batch_size, test_y.shape[0]))
                        y_test_batch = test_y[test_slice].to(device)
                        x_test_batch = test_x_true[test_slice].to(device)

                        f_test_batch = None
                        if test_f_true is not None:
                            f_test_batch = test_f_true[test_slice].to(device)

                        x_test_pred, f_test_pred = model(y_test_batch)
                        f_test_pred_highpass = None
                        if criterion.alpha_leakage > 0:
                            f_test_pred_highpass = compute_loss_highpass(f_test_pred)
                        test_loss, _ = criterion(
                            x_test_pred,
                            x_test_batch,
                            f_test_pred,
                            f_test_batch,
                            f_pred_highpass=f_test_pred_highpass,
                        )
                        test_epoch_loss += test_loss.item()
                        test_batches += 1

                    if test_batches > 0:
                        avg_test_loss = test_epoch_loss / test_batches
                        last_test_loss = avg_test_loss
                model.train()

            # Average loss components
            avg_loss_dict = {k: v / num_batches for k, v in epoch_loss_dict.items()}
            if avg_test_loss is not None:
                avg_loss_dict['test_total'] = avg_test_loss
            avg_loss_dict['stage'] = stage_name
            loss_details.append(avg_loss_dict)

            if verbose:
                print(f"  Mean |x_pred|: {mean_abs_x_pred:.6f}, Mean |x_true|: {mean_abs_x_true:.6f}")
                with torch.no_grad():
                    model.eval()
                    diag_n = min(8, train_y.shape[0])
                    x_diag_pred, _ = model(train_y[:diag_n].to(device))
                    recon_mse_sample = torch.mean(
                        (x_diag_pred - train_x_true[:diag_n].to(device)) ** 2
                    ).item()
                    model.train()
                print(f"  recon_mse(sample): {recon_mse_sample:.6f}")
                params = model.get_learned_params()
                print(params)
                print(f"END Epoch {global_epoch}: Train Loss NOW = {avg_loss:.6f}")

            should_log = (global_epoch % 10 == 0) or (global_epoch == total_epochs)
            if verbose and should_log:
                test_loss_msg = "" if avg_test_loss is None else f", Test Loss: {avg_test_loss:.6f}"
                print(f"Epoch {global_epoch}/{total_epochs}, Train Loss: {avg_loss:.6f}{test_loss_msg}")
                print(f"  Components: recon={avg_loss_dict.get('reconstruction', 0):.4f}, "
                      f"L1={avg_loss_dict.get('l1_sparsity', 0):.4f}, "
                      f"TV={avg_loss_dict.get('total_variation', 0):.4f}, "
                      f"baseline={avg_loss_dict.get('baseline_recon', 0):.4f}, "
                      f"leak={avg_loss_dict.get('baseline_leakage', 0):.4f}, "
                      f"ortho={avg_loss_dict.get('peak_baseline_ortho', 0):.4f}, "
                      f"bltv={avg_loss_dict.get('baseline_tv', 0):.6f}, "
                      f"neg={avg_loss_dict.get('non_negativity', 0):.6f}")
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
    output_dir = os.path.join(script_dir, "train")
    os.makedirs(output_dir, exist_ok=True)
    
    # Synthetic data parameters
    N = 4096           # Signal length
    n_samples = 500    # Total number of samples (more diversity)
    train_ratio = 0.8  # 80% train, 20% test
    seed = 42          # For reproducibility
    synthetic_peak_mode = 'linear'  # 'linear', 'exp', or 'mixed'
    synthetic_noise_range = (0.01, 0.01)
    run_beads_floor_check = False  # Optional: classical BEADS mismatch diagnostic

    # Reproducibility: stabilize training variance across runs/checkpoints.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Generate synthetic dataset
    print("\nGenerating synthetic dataset...")
    print(f"  Signal length: {N}")
    print(f"  Total samples: {n_samples}")
    print(f"  Train/Test split: {int(train_ratio*100)}/{int((1-train_ratio)*100)}")
    print(f"  Synthetic regime: BEADS-aligned ({synthetic_peak_mode} peaks)")
    print(f"  Noise std range: {synthetic_noise_range}")
    
    generator = SyntheticDataGenerator(N=N, seed=seed, peak_shape_mode=synthetic_peak_mode)
    dataset = generator.generate_dataset(
        n_samples=n_samples,
        noise_level_range=synthetic_noise_range,
    )

    peak_baseline_ratios = np.array([
        np.max(np.abs(s.f_true)) / (np.max(np.abs(s.x_true)) + 1e-8)
        for s in dataset
    ])
    print("  Baseline/peak max-ratio stats:")
    print(f"    mean={peak_baseline_ratios.mean():.4f}, median={np.median(peak_baseline_ratios):.4f}, "
          f"p10={np.percentile(peak_baseline_ratios, 10):.4f}, "
          f"p90={np.percentile(peak_baseline_ratios, 90):.4f}")

    morph_stats = summarize_derivative_sparsity(dataset, diff1_threshold=1e-3, diff2_threshold=1e-3)
    print("  Derivative-sparsity diagnostics (x_true):")
    print(f"    mean(|diff1|)={morph_stats['mean_abs_diff1']:.6e}")
    print(f"    mean(|diff2|)={morph_stats['mean_abs_diff2']:.6e}")
    print(f"    mean nonzero diff1 locations={morph_stats['mean_diff1_nonzero_count']:.2f} "
          f"({100.0 * morph_stats['mean_diff1_nonzero_fraction']:.2f}%)")
    print(f"    mean nonzero diff2 locations={morph_stats['mean_diff2_nonzero_count']:.2f} "
          f"({100.0 * morph_stats['mean_diff2_nonzero_fraction']:.2f}%)")
    print(f"    mean curvature sparsity={morph_stats['mean_curvature_sparsity']:.4f}")

    if run_beads_floor_check:
        print("\nRunning classical BEADS MSE-floor diagnostic...")
        floor_stats = run_classic_beads_mse_floor_check(
            dataset,
            n_samples=20,
            n_trials=24,
            d=1,
            fc=0.006,
            Nit=24,
            seed=seed,
        )
        print(f"  Classical best-MSE (n={floor_stats['num_samples']}, trials={floor_stats['num_trials']}):")
        print(f"    mean={floor_stats['mean_best_mse']:.6f}, median={floor_stats['median_best_mse']:.6f}, "
              f"p10={floor_stats['p10_best_mse']:.6f}, p90={floor_stats['p90_best_mse']:.6f}")
    
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
    
    # Create model.
    # Use enough unrolled iterations to separate peaks/baseline under supervision.
    model_num_layers = 5
    model_shared_params = False
    model_solve_cg_iters = 5
    model_lowpass_cg_iters = 24
    model_learn_step = True
    model_learn_output_gain = True

    print("\nCreating LBEADS-NET model...")
    model = LBEADS_NET(
        N=N,
        d=1,
        fc=0.006,  # Match classical BEADS cutoff (0.6% of Nyquist)
        num_layers=model_num_layers,
        shared_params=model_shared_params,
        init_lam0=0.002,
        init_lam1=0.3,
        init_lam2=0.3,
        init_r=6.0,
        learn_r=True,
        init_step=1.0,
        learn_step=model_learn_step,
        init_output_gain=1.0,
        learn_output_gain=model_learn_output_gain,
        lowpass_iterations=1,  # Single-pass lowpass (same as classical BEADS)
        solve_cg_iters=model_solve_cg_iters,
        lowpass_cg_iters=model_lowpass_cg_iters,
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

    # Operator sanity check: low-pass on a smooth baseline should retain very low HF energy.
    diagnostic = run_lowpass_operator_diagnostic(
        N=N,
        d=1,
        fc=0.006,
        iterations=int(getattr(model, 'lowpass_iterations', 1)),
    )
    print("\nLow-pass operator diagnostic (smooth baseline input):")
    print(f"  hf_rms: {diagnostic['hf_rms']:.6e}")
    print(f"  hf_ratio: {diagnostic['hf_ratio']:.6e}")
    print(f"  baseline_tv3: {diagnostic['baseline_tv3']:.6e}")
    
    # Configure sparsity-based loss function with baseline supervision.
    loss_config = {
        'alpha_mse': 1.0,
        'alpha_l1': 0.01,
        'alpha_tv': 0.01,
        'alpha_smooth': 0.2,
        'alpha_neg': 2.0,
        'alpha_baseline': 0.5,
        'alpha_leakage': 0.5,
        'alpha_ortho': 0.2,
        'alpha_baseline_tv': 0.0,
        'peak_mask_rel_threshold': 0.02,
        'peak_mask_abs_min': 1e-4,
        'use_huber': False,
        'huber_delta': 0.1
    }

    stage_configs = [
        {
            'name': 'A_peak_recon',
            'epochs': 5,
            'loss_config': {
                'alpha_mse': 1.0,
                'alpha_l1': 0.0,
                'alpha_tv': 0.0,
                'alpha_neg': 0.0,
                'alpha_baseline': 0.0,
                'alpha_leakage': 0.0,
                'alpha_ortho': 0.0,
                'alpha_smooth': 0.0,
                'alpha_baseline_tv': 0.0,
            }
        },
        {
            'name': 'B_masked_baseline_leakage',
            'epochs': 20,
            'loss_config': dict(loss_config)
        }
    ]
    
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
        test_y=test_y,
        test_x_true=test_x_true,
        test_f_true=test_f_true,
        num_epochs=50,           # Used only when stage_configs is None
        learning_rate=1e-3,
        batch_size=4,            # Slightly larger batch for normalized data
        device=device,
        verbose=True,
        loss_config=loss_config,
        stage_configs=stage_configs
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
    training_plot_path = os.path.join(output_dir, 'training_results.png')
    plt.savefig(training_plot_path, dpi=150)
    print(f"\nSaved results to {training_plot_path}")
    
    # Save model
    model_path = os.path.join(script_dir, f'lbeads_net_baseline_fix_{int(time.time())}.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'N': N,
            'd': 1,
            'fc': 0.006,
            'num_layers': model_num_layers,
            'lowpass_iterations': 1,
            'shared_params': bool(getattr(model, 'shared_params', model_shared_params)),
            'solve_cg_iters': int(getattr(model, 'solve_cg_iters', model_solve_cg_iters)),
            'lowpass_cg_iters': int(getattr(model, 'lowpass_cg_iters', model_lowpass_cg_iters)),
            'learn_step': bool(model_learn_step),
            'learn_output_gain': bool(model_learn_output_gain),
            'model_variant': 'lbeads_net',
            'model_class': model.__class__.__name__,
        },
        'loss_config': loss_config,
        'stage_configs': stage_configs,
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


def run_training(config: dict, output_dir: str, callback=None):
    """
    Run full training pipeline with configurable params and event callbacks.

    config keys:
      model: {N, d, fc, num_layers, solve_cg_iters, lowpass_cg_iters, shared_params}
      training: {learning_rate, batch_size, num_samples, noise_level, train_ratio, seed}
      loss: {alpha_mse, alpha_l1, ...all alpha values}
      stages: [{name, epochs, loss_config}, ...]

    callback: function(event_dict) called per-epoch with structured data.
    Returns: dict with keys "metrics", "checkpoint_path", "model"
    """
    import time as _time

    mc = config.get("model", {})
    tc = config.get("training", {})
    lc = config.get("loss", {})
    stages_cfg = config.get("stages", [])

    N = mc.get("N", 4096)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1. Generate synthetic data
    gen = SyntheticDataGenerator(N=N, seed=tc.get("seed", 42))
    signals = gen.generate_dataset(
        n_samples=tc.get("num_samples", 500),
        noise_level_range=(tc.get("noise_level", 0.01), tc.get("noise_level", 0.01))
    )

    n_train = int(len(signals) * tc.get("train_ratio", 0.8))
    train_signals = signals[:n_train]
    test_signals = signals[n_train:]

    # Stack into tensors
    train_y = torch.stack([torch.tensor(s.y, dtype=torch.float32) for s in train_signals])
    train_x = torch.stack([torch.tensor(s.x_true, dtype=torch.float32) for s in train_signals])
    train_f = torch.stack([torch.tensor(s.f_true, dtype=torch.float32) for s in train_signals])
    test_y = torch.stack([torch.tensor(s.y, dtype=torch.float32) for s in test_signals])
    test_x = torch.stack([torch.tensor(s.x_true, dtype=torch.float32) for s in test_signals])
    test_f = torch.stack([torch.tensor(s.f_true, dtype=torch.float32) for s in test_signals])

    # 2. Build model
    model = LBEADS_NET(
        N=N,
        d=mc.get("d", 1),
        fc=mc.get("fc", 0.006),
        num_layers=mc.get("num_layers", 5),
        shared_params=mc.get("shared_params", False),
        lowpass_iterations=1,
        solve_cg_iters=mc.get("solve_cg_iters", 5),
        lowpass_cg_iters=mc.get("lowpass_cg_iters", 24),
    ).to(device)

    # 3. Build loss_config (global defaults + user overrides)
    loss_config = {
        'alpha_mse': 1.0, 'alpha_l1': 0.01, 'alpha_tv': 0.01,
        'alpha_smooth': 0.2, 'alpha_neg': 2.0, 'alpha_baseline': 0.5,
        'alpha_leakage': 0.5, 'alpha_ortho': 0.2, 'alpha_baseline_tv': 0.0,
        'peak_mask_rel_threshold': 0.02, 'peak_mask_abs_min': 1e-4,
        'use_huber': False, 'huber_delta': 0.1,
    }
    loss_config.update(lc)

    # 4. Build stage_configs
    stage_configs = []
    for sc in stages_cfg:
        stage_loss = dict(loss_config)  # inherit global
        stage_loss.update(sc.get("loss_config", {}))
        stage_configs.append({
            'name': sc.get('name', 'stage'),
            'epochs': sc.get('epochs', 10),
            'loss_config': stage_loss,
        })
    if not stage_configs:
        stage_configs = [{'name': 'default', 'epochs': 25, 'loss_config': loss_config}]

    # 5. Run training with callback integration
    global_epoch = [0]
    start_time = _time.time()
    current_stage_name = [stage_configs[0]['name'] if stage_configs else '']

    all_loss_history = []
    all_loss_details = []

    for stage in stage_configs:
        current_stage_name[0] = stage['name']
        loss_history, loss_details = train_lbeads_net(
            model=model,
            train_y=train_y, train_x_true=train_x, train_f_true=train_f,
            test_y=test_y, test_x_true=test_x, test_f_true=test_f,
            num_epochs=stage['epochs'],
            learning_rate=tc.get('learning_rate', 1e-3),
            batch_size=tc.get('batch_size', 4),
            device=device,
            verbose=True,
            loss_config=stage['loss_config'],
        )

        for i, (loss_val, details) in enumerate(zip(loss_history, loss_details)):
            global_epoch[0] += 1
            all_loss_history.append(loss_val)
            all_loss_details.append(details)

            if callback:
                epoch_event = {
                    "type": "epoch",
                    "epoch": global_epoch[0],
                    "stage": current_stage_name[0],
                    "train_loss": loss_val,
                    "test_loss": details.get("test_total"),
                    "components": {k: v for k, v in details.items()
                                   if k not in ("test_total", "stage", "total")},
                    "learned_params": model.get_learned_params() if hasattr(model, 'get_learned_params') else {},
                    "elapsed_s": _time.time() - start_time,
                }
                callback(epoch_event)

    # 6. Compute final metrics
    model.eval()
    with torch.no_grad():
        test_y_dev = test_y.to(device)
        x_pred, f_pred, _ = model(test_y_dev)
        x_pred_np = x_pred.cpu().numpy()
        x_true_np = test_x.numpy()

        test_mse = float(np.mean((x_pred_np - x_true_np) ** 2))
        test_mae = float(np.mean(np.abs(x_pred_np - x_true_np)))
        correlation = float(np.mean([
            np.corrcoef(x_pred_np[i].flatten(), x_true_np[i].flatten())[0, 1]
            for i in range(len(x_pred_np))
        ]))

    final_metrics = {
        "train_mse": float(all_loss_history[-1]) if all_loss_history else 0,
        "test_mse": test_mse,
        "test_mae": test_mae,
        "test_correlation": correlation,
    }

    # 7. Save checkpoint
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, "checkpoint.pth")
    final_params = model.get_learned_params() if hasattr(model, 'get_learned_params') else {}
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': mc,
        'loss_config': loss_config,
        'stage_configs': stage_configs,
        'final_params': final_params,
        'loss_history': all_loss_history,
        'loss_details': all_loss_details,
        'train_metrics': {"mse": float(all_loss_history[-1]) if all_loss_history else 0},
        'test_metrics': final_metrics,
        'data_config': tc,
    }, checkpoint_path)

    # 8. Save training plot
    try:
        plot_path = os.path.join(output_dir, "training_plot.png")
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(all_loss_history)
        axes[0].set_title('Training Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')

        # Component breakdown from last epoch
        if all_loss_details:
            last = all_loss_details[-1]
            comp_keys = [k for k in last.keys() if k not in ('test_total', 'stage', 'total') and isinstance(last[k], (int, float))]
            comp_vals = [last[k] for k in comp_keys]
            axes[1].barh(comp_keys, comp_vals)
            axes[1].set_title('Final Loss Components')

        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()
    except Exception:
        pass  # Plot generation is non-critical

    return {"metrics": final_metrics, "checkpoint_path": checkpoint_path, "model": model}


if __name__ == "__main__":
    main()
