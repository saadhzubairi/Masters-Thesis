#!/usr/bin/env python3
"""
Phase 1: Progressive Development Experiments for LBEADS-NET thesis.

This script demonstrates how each additional loss term improves performance
by training models with progressively more loss components.

Experiments:
    1.0  Classical BEADS (no training, baseline reference)
    1.1  MSE Only
    1.2  MSE + Sparsity (L1, TV, Non-negativity)
    1.3  + Baseline Supervision
    1.4  + Orthogonality (v5 config)
    1.5  Full Anti-Leakage (v7 config)

All experiments share the same synthetic dataset (seed=42) and are evaluated
with identical metrics on a held-out test set.

Results are saved as JSON to Experiments/results/phase1/.
"""

import json
import os
import time
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.interpolate import CubicSpline
from scipy.stats import pearsonr


# =============================================================================
# Device detection
# =============================================================================

def get_device() -> str:
    """Detect best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return 'cuda'
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # MPS only supports float32; we need float64 for numerical stability
        # in BEADS matrix operations, so fall back to CPU.
        return 'cpu'
    return 'cpu'


# =============================================================================
# Model architecture (from v5: LBEADS_NET_Fast)
# =============================================================================

def BAfilt(d, fc, N):
    """
    Banded matrices for zero-phase high-pass filter.

    INPUT
        d  : degree of filter is 2d (use d = 1 or 2)
        fc : cut-off frequency (normalized frequency, 0 < fc < 0.5)
        N  : length of signal

    OUTPUT
        A  : Symmetric banded matrix (scipy sparse)
        B  : Banded matrix (scipy sparse)
    """
    b1 = np.array([1, -1])
    for i in range(d - 1):
        b1 = np.convolve(b1, np.array([-1, 2, -1]))
    b = np.convolve(b1, np.array([-1, 1]))

    omc = 2 * np.pi * fc
    t = ((1 - np.cos(omc)) / (1 + np.cos(omc))) ** d

    a = np.array([1])
    for i in range(d):
        a = np.convolve(a, np.array([1, 2, 1]))
    a = b + t * a

    diagonals_A = []
    diagonals_B = []
    offsets = list(range(-d, d + 1))

    for k, offset in enumerate(offsets):
        if offset <= 0:
            diag_len = N + offset
        else:
            diag_len = N - offset
        diagonals_A.append(np.full(diag_len, float(a[k])))
        diagonals_B.append(np.full(diag_len, float(b[k])))

    A = sparse.diags(diagonals_A, offsets, shape=(N, N), format='csc', dtype=np.float64)
    B = sparse.diags(diagonals_B, offsets, shape=(N, N), format='csc', dtype=np.float64)

    return A, B


def compute_lowpass_matrix(A_dense, B_dense):
    """
    Compute low-pass matrix L where:
        baseline = L @ residual
        L = I - B @ inv(A)
    """
    n = A_dense.shape[0]
    eye = torch.eye(n, dtype=A_dense.dtype, device=A_dense.device)
    A_inv = torch.linalg.solve(A_dense, eye)
    return eye - (B_dense @ A_inv)


def compute_iterated_lowpass_matrix(A_dense, B_dense, iterations=3):
    """
    Compute iterated low-pass matrix for stronger baseline smoothing.
    L_iter = L^iterations where L = I - B @ inv(A)
    """
    L = compute_lowpass_matrix(A_dense, B_dense)
    L_iter = L.clone()
    for _ in range(iterations - 1):
        L_iter = L_iter @ L
    return L_iter


def apply_lowpass_filter(residual, lowpass_matrix):
    """
    Apply low-pass matrix to residual.

    Args:
        residual: (N,) or (batch, N)
        lowpass_matrix: (N, N), column-operator form
    """
    squeeze_output = False
    if residual.dim() == 1:
        residual = residual.unsqueeze(0)
        squeeze_output = True

    baseline = residual @ lowpass_matrix.T

    if squeeze_output:
        baseline = baseline.squeeze(0)
    return baseline


def apply_highpass_filter(signal, lowpass_matrix):
    """
    Apply complementary high-pass operator H = I - L using the low-pass matrix L.
    """
    low = apply_lowpass_filter(signal, lowpass_matrix)
    return signal - low


class LBEADS_NET_Fast(nn.Module):
    """
    A fast, trainable version of LBEADS-NET.

    Uses a proximal gradient approach that's faithful to BEADS while being
    fully differentiable and fast enough to train.
    """

    def __init__(self, N, d=1, fc=0.006, num_layers=10,
                 init_lam0=0.4, init_lam1=4.0, init_lam2=3.2,
                 init_r=6.0, init_step_size=0.1,
                 lowpass_iterations=1):
        super(LBEADS_NET_Fast, self).__init__()

        self.N = N
        self.d = d
        self.fc = fc
        self.num_layers = num_layers
        self.EPS0 = 1e-6
        self.EPS1 = 1e-6

        # Pre-compute filter matrices
        A, B = BAfilt(d, fc, N)
        self.A_sp = A
        self.B_sp = B

        self.register_buffer('A_dense', torch.tensor(A.toarray(), dtype=torch.float64))
        self.register_buffer('B_dense', torch.tensor(B.toarray(), dtype=torch.float64))
        BTB = B.T @ B
        self.register_buffer('BTB_dense', torch.tensor(BTB.toarray(), dtype=torch.float64))
        self.register_buffer(
            'lowpass_dense',
            compute_iterated_lowpass_matrix(
                torch.tensor(A.toarray(), dtype=torch.float64),
                torch.tensor(B.toarray(), dtype=torch.float64),
                iterations=lowpass_iterations,
            ),
            persistent=False,
        )

        # Layer-wise learnable parameters
        self.log_lam0 = nn.ParameterList([
            nn.Parameter(torch.tensor(np.log(init_lam0), dtype=torch.float64))
            for _ in range(num_layers)
        ])
        self.log_lam1 = nn.ParameterList([
            nn.Parameter(torch.tensor(np.log(init_lam1), dtype=torch.float64))
            for _ in range(num_layers)
        ])
        self.log_lam2 = nn.ParameterList([
            nn.Parameter(torch.tensor(np.log(init_lam2), dtype=torch.float64))
            for _ in range(num_layers)
        ])
        self.log_r = nn.ParameterList([
            nn.Parameter(torch.tensor(np.log(init_r), dtype=torch.float64))
            for _ in range(num_layers)
        ])
        self.log_step_size = nn.ParameterList([
            nn.Parameter(torch.tensor(np.log(init_step_size), dtype=torch.float64))
            for _ in range(num_layers)
        ])

    def diff1(self, x):
        """First difference: D1 @ x (vectorized)."""
        return x[:, 1:] - x[:, :-1]

    def diff2(self, x):
        """Second difference: D2 @ x (vectorized)."""
        return x[:, 2:] - 2 * x[:, 1:-1] + x[:, :-2]

    def diff1_T(self, v):
        """Transpose of first difference: D1.T @ v (vectorized)."""
        batch_size = v.shape[0]
        N = v.shape[1] + 1
        result = torch.zeros(batch_size, N, dtype=v.dtype, device=v.device)
        result[:, :-1] -= v
        result[:, 1:] += v
        return result

    def diff2_T(self, v):
        """Transpose of second difference: D2.T @ v (vectorized)."""
        batch_size = v.shape[0]
        N = v.shape[1] + 2
        result = torch.zeros(batch_size, N, dtype=v.dtype, device=v.device)
        result[:, :-2] += v
        result[:, 1:-1] -= 2 * v
        result[:, 2:] += v
        return result

    def asymmetric_soft_threshold(self, x, lam, r):
        """
        Asymmetric soft thresholding for BEADS.
        Penalizes negative values r times more than positive.
        """
        pos_thresh = lam
        neg_thresh = lam * r

        result = torch.zeros_like(x)
        pos_mask = x > pos_thresh
        neg_mask = x < -neg_thresh

        result[pos_mask] = x[pos_mask] - pos_thresh
        result[neg_mask] = x[neg_mask] + neg_thresh
        return result

    def forward(self, y, return_intermediate=False):
        """
        Forward pass using ISTA-style proximal gradient with asymmetric thresholding.
        """
        if y.dim() == 1:
            y = y.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        # Initialize x from high-pass filtered y
        f_init = apply_lowpass_filter(y, self.lowpass_dense)
        x = y - f_init
        intermediates = [x.clone()] if return_intermediate else None

        for k in range(self.num_layers):
            lam0 = torch.exp(self.log_lam0[k])
            lam1 = torch.exp(self.log_lam1[k])
            lam2 = torch.exp(self.log_lam2[k])
            r = torch.exp(self.log_r[k])
            step_size = torch.exp(self.log_step_size[k])

            # Data fidelity gradient (high-pass residual)
            residual = y - x
            data_grad = apply_highpass_filter(residual, self.lowpass_dense)

            # Smoothness penalty gradients
            Dx1 = self.diff1(x)
            w1 = Dx1 / (torch.abs(Dx1) + self.EPS1)
            grad_D1 = lam1 * self.diff1_T(w1)

            Dx2 = self.diff2(x)
            w2 = Dx2 / (torch.abs(Dx2) + self.EPS1)
            grad_D2 = lam2 * self.diff2_T(w2)

            # Gradient descent step
            x_update = x + step_size * data_grad - step_size * (grad_D1 + grad_D2)

            # Proximal step (asymmetric soft thresholding)
            x = self.asymmetric_soft_threshold(x_update, lam0 * step_size, r)

            if return_intermediate:
                intermediates.append(x.clone())

        # Compute baseline
        residual = y - x
        f = apply_lowpass_filter(residual, self.lowpass_dense)

        if squeeze_output:
            x = x.squeeze(0)
            f = f.squeeze(0)

        if return_intermediate:
            return x, f, intermediates
        return x, f

    def get_learned_params(self):
        """Return dictionary of current learned parameters."""
        params = {}
        for i in range(self.num_layers):
            params[f"layer_{i}_lam0"] = torch.exp(self.log_lam0[i]).item()
            params[f"layer_{i}_lam1"] = torch.exp(self.log_lam1[i]).item()
            params[f"layer_{i}_lam2"] = torch.exp(self.log_lam2[i]).item()
            params[f"layer_{i}_r"] = torch.exp(self.log_r[i]).item()
            params[f"layer_{i}_step_size"] = torch.exp(self.log_step_size[i]).item()
        return params


# =============================================================================
# Synthetic Data Generation (from v5)
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
    """

    def __init__(self, N: int = 4096, seed: Optional[int] = None):
        self.N = N
        self.t = np.linspace(0.0, 1.0, N)
        self.rng = np.random.default_rng(seed)

    def generate_baseline(
        self,
        amplitude_range: Tuple[float, float] = (2.0, 35.0),
        num_knots_range: Tuple[int, int] = (4, 8),
        drift_type: Optional[str] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """Generate a smooth, realistic chromatogram baseline."""
        meta = {}

        if drift_type is None:
            drift_type = self.rng.choice(['spline', 'exponential_decay', 'mixed'])
        meta['drift_type'] = drift_type

        indices = np.arange(self.N, dtype=float)

        if drift_type == 'spline':
            num_knots = int(self.rng.integers(num_knots_range[0], num_knots_range[1] + 1))
            knot_x = np.sort(self.rng.uniform(0, self.N, num_knots))
            knot_x = np.concatenate([[0], knot_x, [self.N - 1]])
            amp = float(self.rng.uniform(amplitude_range[0], amplitude_range[1]))
            knot_y = self.rng.uniform(0, amp, len(knot_x))
            knot_y[0] = float(self.rng.uniform(0, amp * 0.3))
            knot_y[-1] = float(self.rng.uniform(0, amp * 0.3))
            cs = CubicSpline(knot_x, knot_y, bc_type='natural')
            baseline = cs(indices)
            meta['num_knots'] = num_knots
            meta['amplitude'] = amp

        elif drift_type == 'exponential_decay':
            amp = float(self.rng.uniform(amplitude_range[0], amplitude_range[1]))
            decay_rate = float(self.rng.uniform(0.5, 3.0))
            baseline = amp * np.exp(-decay_rate * self.t)
            hump_center = float(self.rng.uniform(0.2, 0.8)) * self.N
            hump_sigma = float(self.rng.uniform(0.15, 0.4)) * self.N
            hump_amp = float(self.rng.uniform(0.5, max(1.0, amp * 0.5)))
            baseline += hump_amp * np.exp(-((indices - hump_center) ** 2) / (2.0 * hump_sigma ** 2))
            meta['decay_rate'] = decay_rate
            meta['amplitude'] = amp

        else:  # mixed
            amp = float(self.rng.uniform(amplitude_range[0], amplitude_range[1]))
            degree = int(self.rng.integers(2, 4))
            coeffs = self.rng.uniform(-1, 1, degree + 1)
            centered_t = self.t - 0.5
            poly = np.zeros(self.N)
            for i, c in enumerate(coeffs):
                poly += c * (centered_t ** i)
            poly = (poly - poly.min()) / (poly.max() - poly.min() + 1e-10)
            baseline = poly * amp
            num_sines = int(self.rng.integers(1, 3))
            for _ in range(num_sines):
                freq = float(self.rng.uniform(0.3, 2.0))
                s_amp = float(self.rng.uniform(0.5, max(1.5, amp * 0.3)))
                phase = float(self.rng.uniform(0, 2 * np.pi))
                baseline += s_amp * np.sin(2 * np.pi * freq * self.t + phase)
            meta['amplitude'] = amp
            meta['poly_degree'] = degree

        if baseline.min() < -2.0:
            baseline -= baseline.min() - float(self.rng.uniform(0, 2.0))

        meta['final_range'] = (float(baseline.min()), float(baseline.max()))
        return baseline, meta

    def _emg_peak(self, indices: np.ndarray, center: float, sigma: float,
                  amplitude: float, tau: float) -> np.ndarray:
        """Asymmetric chromatographic peak using bi-Gaussian model."""
        sigma_left = sigma
        sigma_right = sigma + tau
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
        """Generate realistic chromatographic peaks."""
        num_peaks = int(self.rng.integers(num_peaks_range[0], num_peaks_range[1] + 1))

        x_true = np.zeros(self.N)
        peak_info: List[Dict] = []
        indices = np.arange(self.N, dtype=float)
        margin = int(0.02 * self.N)

        centers = []
        i = 0
        while i < num_peaks:
            if self.rng.random() < cluster_prob and i > 0 and num_peaks - i >= 2:
                cluster_size = min(int(self.rng.integers(2, 5)), num_peaks - i)
                cluster_center = int(self.rng.integers(margin, self.N - margin))
                cluster_spread = float(self.rng.uniform(15, 80))
                for j in range(cluster_size):
                    c = cluster_center + int(self.rng.normal(0, cluster_spread))
                    c = np.clip(c, margin, self.N - margin - 1)
                    centers.append(int(c))
                i += cluster_size
            else:
                centers.append(int(self.rng.integers(margin, self.N - margin)))
                i += 1

        centers = sorted(centers[:num_peaks])

        for center in centers:
            log_sigma = float(self.rng.uniform(np.log(sigma_range[0]), np.log(sigma_range[1])))
            sigma = np.exp(log_sigma)
            log_amp = float(self.rng.uniform(np.log(amplitude_range[0]), np.log(amplitude_range[1])))
            amplitude = np.exp(log_amp)
            if self.rng.random() < 0.08:
                amplitude = float(self.rng.uniform(1000.0, amplitude_range[1]))
            tau = 0.0
            if self.rng.random() < tail_prob:
                tau = float(self.rng.uniform(tail_tau_range[0], tail_tau_range[1]))
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

        params = {"num_peaks": len(centers), "peaks": peak_info}
        return x_true, params

    def generate_noise(self, noise_level: float = 3.0) -> Tuple[np.ndarray, Dict]:
        """Generate additive Gaussian noise."""
        noise = self.rng.normal(0.0, noise_level, self.N)
        return noise, {"noise_level": float(noise_level)}

    def generate_signal(
        self,
        noise_level: float = 3.0,
        baseline_peak_ratio_range: Tuple[float, float] = (0.01, 0.5),
    ) -> SyntheticSignal:
        """Generate a complete synthetic chromatogram signal."""
        x_true, peak_meta = self.generate_peaks()
        f_true, baseline_meta = self.generate_baseline()

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

        return SyntheticSignal(y=y, x_true=x_true, f_true=f_true, noise=noise, metadata=metadata)

    def generate_dataset(
        self,
        n_samples: int,
        noise_level_range: Tuple[float, float] = (2.0, 6.0),
    ) -> List[SyntheticSignal]:
        """Generate a dataset of synthetic chromatogram signals."""
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
) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
           Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Create 80/20 train/test split from synthetic dataset.
    Per-sample normalization by max(|y|).
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
            scales = np.max(np.abs(ys), axis=1, keepdims=True)
            scales = np.maximum(scales, 1e-8)
            ys = ys / scales
            xs = xs / scales
            fs = fs / scales
        return ys, xs, fs

    train_y, train_x, train_f = stack_and_normalize(train_indices)
    test_y, test_x, test_f = stack_and_normalize(test_indices)

    return (
        (torch.tensor(train_y, dtype=torch.float64),
         torch.tensor(train_x, dtype=torch.float64),
         torch.tensor(train_f, dtype=torch.float64)),
        (torch.tensor(test_y, dtype=torch.float64),
         torch.tensor(test_x, dtype=torch.float64),
         torch.tensor(test_f, dtype=torch.float64)),
    )


# =============================================================================
# Loss Function (full v7 SparsityLoss with all 12 terms)
# =============================================================================

class SparsityLoss(nn.Module):
    """
    Sparsity-promoting loss function for chromatogram peak recovery.

    Combines up to 12 loss terms:
     1. Peak reconstruction (MSE/Huber)
     2. L1 sparsity on peaks
     3. Total Variation on peaks
     4. Baseline smoothness (2nd derivative)
     5. Non-negativity penalty
     6. Masked baseline reconstruction
     6b. High-frequency baseline leakage penalty
     7. Peak-baseline orthogonality (element-wise)
     8. Baseline TV (3rd derivative)
     9. Asymmetric baseline loss
    10. Envelope constraint
    11. Frequency separation
    """

    def __init__(self,
                 alpha_mse: float = 1.0,
                 alpha_l1: float = 0.0,
                 alpha_tv: float = 0.0,
                 alpha_smooth: float = 0.0,
                 alpha_neg: float = 0.0,
                 alpha_baseline: float = 0.0,
                 alpha_leakage: float = 0.0,
                 alpha_ortho: float = 0.0,
                 alpha_baseline_tv: float = 0.0,
                 alpha_asym_baseline: float = 0.0,
                 asym_alpha: float = 0.9,
                 alpha_envelope: float = 0.0,
                 alpha_freq: float = 0.0,
                 peak_mask_rel_threshold: float = 0.02,
                 peak_mask_abs_min: float = 1e-4,
                 use_huber: bool = True,
                 huber_delta: float = 1.0):
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
        self.alpha_asym_baseline = alpha_asym_baseline
        self.asym_alpha = asym_alpha
        self.alpha_envelope = alpha_envelope
        self.alpha_freq = alpha_freq
        self.peak_mask_rel_threshold = peak_mask_rel_threshold
        self.peak_mask_abs_min = peak_mask_abs_min
        self.use_huber = use_huber
        self.huber_delta = huber_delta

        if use_huber:
            self.huber = nn.HuberLoss(reduction='mean', delta=huber_delta)

    # -- helper methods from v7 --

    def _soft_local_min(self, y: torch.Tensor, window: int = 51, tau: float = 0.1) -> torch.Tensor:
        """Differentiable sliding-window local minimum via log-sum-exp."""
        pad = window // 2
        y_pad = F.pad(y.unsqueeze(1), (pad, pad), mode='reflect')
        y_unf = y_pad.unfold(-1, window, 1).squeeze(1)
        local_min = -tau * torch.logsumexp(-y_unf / tau, dim=-1)
        return local_min

    def _asymmetric_baseline_loss(self, f_pred: torch.Tensor, f_true: torch.Tensor,
                                  alpha: float = None) -> torch.Tensor:
        """Penalize over-estimation alpha times more than under-estimation."""
        if alpha is None:
            alpha = self.asym_alpha
        residual = f_pred - f_true
        loss = torch.where(
            residual > 0,
            alpha * residual ** 2,
            (1 - alpha) * residual ** 2,
        )
        return loss.mean()

    def _envelope_loss(self, f_pred: torch.Tensor, y: torch.Tensor,
                       window: int = 51) -> torch.Tensor:
        """Baseline should not exceed local signal minimum."""
        local_min = self._soft_local_min(y, window=window).detach()
        violation = F.relu(f_pred - local_min)
        return violation.pow(2).mean()

    def _freq_separation_loss(self, x_pred: torch.Tensor, f_pred: torch.Tensor,
                              fc: float = 0.005) -> torch.Tensor:
        """Penalize high-freq in baseline and low-freq in peaks."""
        X_peak = torch.fft.rfft(x_pred, dim=-1)
        X_base = torch.fft.rfft(f_pred, dim=-1)
        freqs = torch.fft.rfftfreq(x_pred.shape[-1], device=x_pred.device)
        hpf = (freqs > fc).float()
        lpf = (freqs <= fc).float()
        loss = (X_base.abs() * hpf).pow(2).mean() + (X_peak.abs() * lpf).pow(2).mean()
        return loss

    def forward(self, x_pred: torch.Tensor, x_target: torch.Tensor,
                f_pred: Optional[torch.Tensor] = None,
                f_target: Optional[torch.Tensor] = None,
                y: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute total loss with breakdown."""
        # Ensure batch dimension
        if x_pred.dim() == 1:
            x_pred = x_pred.unsqueeze(0)
            x_target = x_target.unsqueeze(0)
        if f_pred is not None and f_pred.dim() == 1:
            f_pred = f_pred.unsqueeze(0)
        if f_target is not None and f_target.dim() == 1:
            f_target = f_target.unsqueeze(0)
        if y is not None and y.dim() == 1:
            y = y.unsqueeze(0)

        loss_dict = {}

        # 1. RECONSTRUCTION LOSS
        if self.use_huber:
            recon_loss = self.huber(x_pred, x_target)
        else:
            recon_loss = torch.mean((x_pred - x_target) ** 2)
        loss_dict['reconstruction'] = recon_loss.item()
        total_loss = self.alpha_mse * recon_loss

        # 2. L1 SPARSITY
        if self.alpha_l1 > 0:
            l1_loss = torch.mean(torch.abs(x_pred))
            loss_dict['l1_sparsity'] = l1_loss.item()
            total_loss = total_loss + self.alpha_l1 * l1_loss

        # 3. TOTAL VARIATION
        if self.alpha_tv > 0:
            diff1 = x_pred[:, 1:] - x_pred[:, :-1]
            tv_loss = torch.mean(torch.abs(diff1))
            loss_dict['total_variation'] = tv_loss.item()
            total_loss = total_loss + self.alpha_tv * tv_loss

        # 4. BASELINE SMOOTHNESS (2nd derivative)
        if self.alpha_smooth > 0 and f_pred is not None:
            diff2 = f_pred[:, 2:] - 2 * f_pred[:, 1:-1] + f_pred[:, :-2]
            smooth_loss = torch.mean(diff2 ** 2)
            loss_dict['baseline_smooth'] = smooth_loss.item()
            total_loss = total_loss + self.alpha_smooth * smooth_loss

        # 5. NON-NEGATIVITY
        if self.alpha_neg > 0:
            neg_values = torch.clamp(-x_pred, min=0)
            neg_loss = torch.mean(neg_values ** 2)
            loss_dict['non_negativity'] = neg_loss.item()
            total_loss = total_loss + self.alpha_neg * neg_loss

        # Build peak/background masks from x_target
        peak_amp = torch.max(torch.abs(x_target), dim=1, keepdim=True).values
        peak_thr = torch.clamp(
            peak_amp * self.peak_mask_rel_threshold,
            min=self.peak_mask_abs_min,
        )
        peak_mask = (torch.abs(x_target) >= peak_thr).to(x_pred.dtype)
        bg_mask = 1.0 - peak_mask

        # 6. MASKED BASELINE RECONSTRUCTION
        if self.alpha_baseline > 0 and f_pred is not None and f_target is not None:
            if self.use_huber:
                baseline_err = F.huber_loss(
                    f_pred, f_target, delta=self.huber_delta, reduction='none',
                )
            else:
                baseline_err = (f_pred - f_target) ** 2
            bg_denom = torch.sum(bg_mask) + 1e-8
            baseline_loss = torch.sum(baseline_err * bg_mask) / bg_denom
            loss_dict['baseline_recon'] = baseline_loss.item()
            total_loss = total_loss + self.alpha_baseline * baseline_loss

        # 6b. HIGH-FREQUENCY LEAKAGE
        if self.alpha_leakage > 0 and f_pred is not None:
            if f_pred.shape[1] >= 3:
                f_diff2 = f_pred[:, 2:] - 2 * f_pred[:, 1:-1] + f_pred[:, :-2]
                peak_mask_mid = peak_mask[:, 1:-1]
                peak_denom = torch.sum(peak_mask_mid) + 1e-8
                leakage_loss = torch.sum((f_diff2 ** 2) * peak_mask_mid) / peak_denom
            else:
                leakage_loss = torch.tensor(0.0, device=x_pred.device, dtype=x_pred.dtype)
            loss_dict['baseline_leakage'] = leakage_loss.item()
            total_loss = total_loss + self.alpha_leakage * leakage_loss

        # 7. PEAK-BASELINE ORTHOGONALITY (element-wise product)
        if self.alpha_ortho > 0 and f_pred is not None:
            ortho_loss = (torch.abs(x_pred) * torch.abs(f_pred)).mean()
            loss_dict['peak_baseline_ortho'] = ortho_loss.item()
            total_loss = total_loss + self.alpha_ortho * ortho_loss

        # 8. BASELINE TV (3rd derivative)
        if self.alpha_baseline_tv > 0 and f_pred is not None:
            diff3 = (f_pred[:, 3:] - 3 * f_pred[:, 2:-1]
                     + 3 * f_pred[:, 1:-2] - f_pred[:, :-3])
            baseline_tv_loss = torch.mean(diff3 ** 2)
            loss_dict['baseline_tv'] = baseline_tv_loss.item()
            total_loss = total_loss + self.alpha_baseline_tv * baseline_tv_loss

        # 9. ASYMMETRIC BASELINE LOSS
        if self.alpha_asym_baseline > 0 and f_pred is not None and f_target is not None:
            asym_loss = self._asymmetric_baseline_loss(f_pred, f_target, alpha=self.asym_alpha)
            loss_dict['asym_baseline'] = asym_loss.item()
            total_loss = total_loss + self.alpha_asym_baseline * asym_loss

        # 10. ENVELOPE CONSTRAINT
        if self.alpha_envelope > 0 and f_pred is not None and y is not None:
            env_loss = self._envelope_loss(f_pred, y)
            loss_dict['envelope'] = env_loss.item()
            total_loss = total_loss + self.alpha_envelope * env_loss

        # 11. FREQUENCY SEPARATION
        if self.alpha_freq > 0 and f_pred is not None:
            freq_loss = self._freq_separation_loss(x_pred, f_pred)
            loss_dict['freq_separation'] = freq_loss.item()
            total_loss = total_loss + self.alpha_freq * freq_loss

        loss_dict['total'] = total_loss.item()
        return total_loss, loss_dict


# =============================================================================
# Classical BEADS (from v0, standalone numpy implementation)
# =============================================================================

def beads_classical(y, d, fc, r, lam0, lam1, lam2, Nit=30, EPS0=1e-6, EPS1=1e-6):
    """
    Classical BEADS algorithm (numpy-only, no torch dependency).

    Solves the chromatogram baseline estimation problem iteratively.

    Args:
        y: 1-D signal array
        d: filter order (1 or 2)
        fc: cut-off frequency (0 < fc < 0.5)
        r: asymmetry ratio
        lam0, lam1, lam2: regularization parameters
        Nit: number of iterations

    Returns:
        x: estimated peaks
        f: estimated baseline
    """
    y = np.asarray(y, dtype=np.float64).flatten()
    N = y.size
    x = y.copy()

    # Build filter matrices (dense, for numpy linalg.solve)
    b1 = np.array([1.0, -1.0])
    for _ in range(d - 1):
        b1 = np.convolve(b1, np.array([-1.0, 2.0, -1.0]))
    b_coeff = np.convolve(b1, np.array([-1.0, 1.0]))

    omc = 2.0 * np.pi * fc
    t = ((1.0 - np.cos(omc)) / (1.0 + np.cos(omc))) ** d

    a_coeff = np.array([1.0])
    for _ in range(d):
        a_coeff = np.convolve(a_coeff, np.array([1.0, 2.0, 1.0]))
    a_coeff = b_coeff + t * a_coeff

    # Build banded matrices
    A = np.zeros((N, N), dtype=float)
    B = np.zeros((N, N), dtype=float)
    offsets = np.arange(-d, d + 1)
    for idx, k in enumerate(offsets):
        if k < 0:
            i_start, j_start = -k, 0
            length = N + k
        else:
            i_start, j_start = 0, k
            length = N - k
        if length <= 0:
            continue
        A[i_start:i_start + length, j_start:j_start + length] += np.diag(
            np.full(length, a_coeff[idx])
        )
        B[i_start:i_start + length, j_start:j_start + length] += np.diag(
            np.full(length, b_coeff[idx])
        )

    def H(vec):
        return B @ np.linalg.solve(A, vec)

    # Difference operators
    e1 = np.ones(N - 1)
    D1 = np.zeros((N - 1, N))
    D1[np.arange(N - 1), np.arange(N - 1)] = -e1
    D1[np.arange(N - 1), np.arange(1, N)] = e1

    e2 = np.ones(N - 2)
    D2 = np.zeros((N - 2, N))
    D2[np.arange(N - 2), np.arange(N - 2)] = e2
    D2[np.arange(N - 2), np.arange(1, N - 1)] = -2.0 * e2
    D2[np.arange(N - 2), np.arange(2, N)] = e2

    D = np.vstack([D1, D2])
    BTB = B.T @ B
    w = np.concatenate([lam1 * np.ones(N - 1), lam2 * np.ones(N - 2)])
    b_vec = (1.0 - r) / 2.0 * np.ones(N)
    d_vec = BTB @ np.linalg.solve(A, y) - lam0 * (A.T @ b_vec)

    gamma = np.ones(N)

    def wfun(z):
        return 1.0 / (np.abs(z) + EPS1)

    for it in range(Nit):
        Dx = D @ x
        Lambda_diag = w * wfun(Dx)
        Lambda = np.diag(Lambda_diag)

        k = np.abs(x) > EPS0
        gamma[~k] = ((1.0 + r) / 4.0) / abs(EPS0)
        gamma[k] = ((1.0 + r) / 4.0) / np.abs(x[k])

        Gamma = np.diag(gamma)
        M = 2.0 * lam0 * Gamma + D.T @ Lambda @ D
        K = BTB + A.T @ M @ A
        z = np.linalg.solve(K, d_vec)
        x = A @ z

    f = y - x - H(y - x)
    return x, f


# =============================================================================
# Metrics
# =============================================================================

def compute_metrics(x_pred: np.ndarray, x_true: np.ndarray,
                    f_pred: np.ndarray, f_true: np.ndarray) -> Dict[str, float]:
    """
    Compute all evaluation metrics for a single signal.

    Args:
        x_pred: predicted peaks (N,)
        x_true: ground truth peaks (N,)
        f_pred: predicted baseline (N,)
        f_true: ground truth baseline (N,)

    Returns:
        Dictionary of metric values.
    """
    eps = 1e-12

    # Peak metrics
    peak_mse = float(np.mean((x_pred - x_true) ** 2))
    peak_mae = float(np.mean(np.abs(x_pred - x_true)))

    # Pearson correlation
    if np.std(x_pred) < eps or np.std(x_true) < eps:
        peak_corr = 0.0
    else:
        peak_corr = float(pearsonr(x_pred.flatten(), x_true.flatten())[0])

    # Baseline metrics
    baseline_mse = float(np.mean((f_pred - f_true) ** 2))
    baseline_mae = float(np.mean(np.abs(f_pred - f_true)))

    # Area error (relative)
    sum_true = float(np.sum(x_true))
    sum_pred = float(np.sum(x_pred))
    area_error = abs(sum_pred - sum_true) / (abs(sum_true) + eps) * 100.0

    # Build peak mask from x_true
    peak_amp = float(np.max(np.abs(x_true)))
    peak_thr = max(peak_amp * 0.02, 1e-4)
    peak_mask = np.abs(x_true) >= peak_thr
    bg_mask = ~peak_mask

    # BLI: Baseline Leakage Index
    residual_f = f_pred - f_true
    leakage = np.maximum(residual_f, 0.0) * peak_mask.astype(float)
    denom_bli = float(np.sum(x_true * peak_mask.astype(float))) + eps
    bli = float(np.sum(leakage)) / denom_bli

    # PBER: Peak-to-Baseline Error Ratio
    if np.sum(peak_mask) > 0:
        rmse_peak = float(np.sqrt(np.mean((f_pred[peak_mask] - f_true[peak_mask]) ** 2)))
    else:
        rmse_peak = 0.0
    if np.sum(bg_mask) > 0:
        rmse_bg = float(np.sqrt(np.mean((f_pred[bg_mask] - f_true[bg_mask]) ** 2)))
    else:
        rmse_bg = 0.0
    pber = rmse_peak / (rmse_bg + eps)

    # Negative fraction
    total_elements = float(x_pred.size)
    neg_fraction = float(np.sum(x_pred < 0)) / total_elements

    return {
        'peak_mse': peak_mse,
        'peak_mae': peak_mae,
        'peak_corr': peak_corr,
        'baseline_mse': baseline_mse,
        'baseline_mae': baseline_mae,
        'area_error': area_error,
        'bli': bli,
        'pber': pber,
        'neg_fraction': neg_fraction,
    }


def aggregate_metrics(per_signal_metrics: List[Dict[str, float]]) -> Dict[str, float]:
    """Aggregate per-signal metrics into mean values."""
    if not per_signal_metrics:
        return {}
    keys = per_signal_metrics[0].keys()
    agg = {}
    for k in keys:
        vals = [m[k] for m in per_signal_metrics]
        agg[f'{k}_mean'] = float(np.mean(vals))
        agg[f'{k}_std'] = float(np.std(vals))
        agg[f'{k}_median'] = float(np.median(vals))
    return agg


# =============================================================================
# Training function
# =============================================================================

def train_model(
    loss_config: Dict,
    train_y: torch.Tensor,
    train_x_true: torch.Tensor,
    train_f_true: torch.Tensor,
    device: str = 'cpu',
    num_epochs: int = 30,
    learning_rate: float = 1e-3,
    batch_size: int = 16,
    N: int = 4096,
    seed: int = 42,
) -> LBEADS_NET_Fast:
    """
    Train a fresh LBEADS_NET_Fast model with the given loss configuration.

    Returns the trained model.
    """
    # Fresh reproducible init
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = LBEADS_NET_Fast(
        N=N,
        d=1,
        fc=0.006,
        num_layers=20,
        init_lam0=0.01,
        init_lam1=0.5,
        init_lam2=0.5,
        init_r=6.0,
        init_step_size=0.05,
        lowpass_iterations=1,
    ).to(device)
    model.train()

    # Build loss criterion -- merge user config with safe defaults (all zeros)
    full_config = {
        'alpha_mse': 0.0,
        'alpha_l1': 0.0,
        'alpha_tv': 0.0,
        'alpha_smooth': 0.0,
        'alpha_neg': 0.0,
        'alpha_baseline': 0.0,
        'alpha_leakage': 0.0,
        'alpha_ortho': 0.0,
        'alpha_baseline_tv': 0.0,
        'alpha_asym_baseline': 0.0,
        'asym_alpha': 0.9,
        'alpha_envelope': 0.0,
        'alpha_freq': 0.0,
        'peak_mask_rel_threshold': 0.02,
        'peak_mask_abs_min': 1e-4,
        'use_huber': True,
        'huber_delta': 0.1,
    }
    full_config.update(loss_config)
    criterion = SparsityLoss(**full_config)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    num_samples = train_y.shape[0]

    for epoch in range(num_epochs):
        perm = torch.randperm(num_samples)
        epoch_loss = 0.0
        num_batches = 0

        for i in range(0, num_samples, batch_size):
            batch_indices = perm[i:min(i + batch_size, num_samples)]
            y_batch = train_y[batch_indices].to(device)
            x_true_batch = train_x_true[batch_indices].to(device)
            f_true_batch = train_f_true[batch_indices].to(device)

            optimizer.zero_grad()
            x_pred, f_pred = model(y_batch)
            loss, _ = criterion(x_pred, x_true_batch, f_pred, f_true_batch, y=y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / num_batches
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1:3d}/{num_epochs}  loss={avg_loss:.6f}")

    return model


# =============================================================================
# Evaluation function
# =============================================================================

def evaluate_model_on_test(
    model: LBEADS_NET_Fast,
    test_y: torch.Tensor,
    test_x_true: torch.Tensor,
    test_f_true: torch.Tensor,
    device: str = 'cpu',
) -> Tuple[List[Dict[str, float]], Dict[str, float]]:
    """
    Evaluate a trained model on every test signal.

    Returns (per_signal_metrics, aggregated_metrics).
    """
    model.eval()
    model = model.to(device)
    per_signal = []

    with torch.no_grad():
        # Process one at a time to avoid OOM on large batches
        for i in range(test_y.shape[0]):
            y_i = test_y[i:i+1].to(device)
            x_pred_i, f_pred_i = model(y_i)
            x_pred_np = x_pred_i[0].cpu().numpy()
            f_pred_np = f_pred_i[0].cpu().numpy()
            x_true_np = test_x_true[i].numpy()
            f_true_np = test_f_true[i].numpy()
            m = compute_metrics(x_pred_np, x_true_np, f_pred_np, f_true_np)
            per_signal.append(m)

    agg = aggregate_metrics(per_signal)
    return per_signal, agg


def evaluate_classical_beads(
    test_y: torch.Tensor,
    test_x_true: torch.Tensor,
    test_f_true: torch.Tensor,
    beads_params: Dict,
) -> Tuple[List[Dict[str, float]], Dict[str, float]]:
    """
    Evaluate classical BEADS on every test signal.

    Returns (per_signal_metrics, aggregated_metrics).
    """
    per_signal = []
    n_test = test_y.shape[0]

    for i in range(n_test):
        y_np = test_y[i].numpy()
        x_true_np = test_x_true[i].numpy()
        f_true_np = test_f_true[i].numpy()

        x_pred, f_pred = beads_classical(
            y_np,
            d=beads_params['d'],
            fc=beads_params['fc'],
            r=beads_params['r'],
            lam0=beads_params['lam0'],
            lam1=beads_params['lam1'],
            lam2=beads_params['lam2'],
            Nit=beads_params['Nit'],
        )
        m = compute_metrics(x_pred, x_true_np, f_pred, f_true_np)
        per_signal.append(m)

    agg = aggregate_metrics(per_signal)
    return per_signal, agg


# =============================================================================
# Results saving
# =============================================================================

def save_results(results: Dict, filepath: str):
    """Save results dict as JSON (handle numpy types)."""
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)


# =============================================================================
# Summary table
# =============================================================================

def print_summary_table(all_results: Dict[str, Dict]):
    """Print a nicely formatted comparison table to stdout."""

    metric_keys = [
        ('peak_mse_mean', 'Peak MSE'),
        ('peak_mae_mean', 'Peak MAE'),
        ('peak_corr_mean', 'Peak Corr'),
        ('baseline_mse_mean', 'BL MSE'),
        ('baseline_mae_mean', 'BL MAE'),
        ('area_error_mean', 'Area Err%'),
        ('bli_mean', 'BLI'),
        ('pber_mean', 'PBER'),
        ('neg_fraction_mean', 'Neg Frac'),
    ]

    exp_names = list(all_results.keys())
    col_width = 14
    header_width = 12

    # Header
    print("\n" + "=" * (header_width + col_width * len(exp_names) + 4))
    print("PHASE 1: PROGRESSIVE DEVELOPMENT -- SUMMARY TABLE")
    print("=" * (header_width + col_width * len(exp_names) + 4))

    # Experiment names row
    header = f"{'Metric':<{header_width}}"
    for name in exp_names:
        # Truncate long names
        short = name[:col_width - 2]
        header += f"  {short:>{col_width - 2}}"
    print(header)
    print("-" * len(header))

    # Metrics rows
    for key, label in metric_keys:
        row = f"{label:<{header_width}}"
        for name in exp_names:
            agg = all_results[name].get('aggregated', {})
            val = agg.get(key, float('nan'))
            if 'corr' in key.lower():
                row += f"  {val:>{col_width - 2}.4f}"
            elif 'neg' in key.lower():
                row += f"  {val:>{col_width - 2}.4f}"
            elif 'area' in key.lower() or 'bli' in key.lower() or 'pber' in key.lower():
                row += f"  {val:>{col_width - 2}.4f}"
            else:
                row += f"  {val:>{col_width - 2}.6f}"
        print(row)

    print("=" * len(header))
    print()


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("PHASE 1: Progressive Development Experiments for LBEADS-NET")
    print("=" * 70)

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results', 'phase1')
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ---- Shared configuration ----
    N = 4096
    n_samples = 200
    train_ratio = 0.8
    seed = 42
    num_epochs = 30
    learning_rate = 1e-3
    batch_size = 16
    device = get_device()

    print(f"\nDevice: {device}")
    print(f"Signal length: {N}")
    print(f"Dataset: {n_samples} signals, {int(train_ratio*100)}/{int((1-train_ratio)*100)} split, seed={seed}")
    print(f"Training: {num_epochs} epochs, lr={learning_rate}, batch_size={batch_size}")

    # ---- Generate dataset ----
    print("\n[1/8] Generating synthetic dataset...")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    generator = SyntheticDataGenerator(N=N, seed=seed)
    dataset = generator.generate_dataset(n_samples=n_samples, noise_level_range=(2.0, 6.0))
    (train_y, train_x, train_f), (test_y, test_x, test_f) = create_train_test_split(
        dataset, train_ratio=train_ratio, seed=seed, normalize=True,
    )
    print(f"  Train: {train_y.shape[0]} signals, Test: {test_y.shape[0]} signals")

    # ---- Experiment configurations ----
    experiments = {
        '1.0 Classical': None,  # No training, classical BEADS
        '1.1 MSE Only': {
            'alpha_mse': 1.0,
        },
        '1.2 MSE+Sparse': {
            'alpha_mse': 1.0,
            'alpha_l1': 0.01,
            'alpha_tv': 0.01,
            'alpha_neg': 0.1,
        },
        '1.3 +Baseline': {
            'alpha_mse': 1.0,
            'alpha_l1': 0.01,
            'alpha_tv': 0.01,
            'alpha_neg': 0.1,
            'alpha_baseline': 1.0,
            'alpha_smooth': 0.01,
        },
        '1.4 +Ortho': {
            'alpha_mse': 1.0,
            'alpha_l1': 0.01,
            'alpha_tv': 0.01,
            'alpha_neg': 0.1,
            'alpha_baseline': 1.0,
            'alpha_smooth': 0.01,
            'alpha_ortho': 0.5,
            'alpha_baseline_tv': 0.1,
        },
        '1.5 Full v7': {
            'alpha_mse': 1.0,
            'alpha_l1': 0.005,
            'alpha_tv': 0.005,
            'alpha_neg': 2.0,
            'alpha_baseline': 2.0,
            'alpha_smooth': 0.01,
            'alpha_ortho': 0.5,
            'alpha_baseline_tv': 0.0,
            'alpha_leakage': 1.0,
            'alpha_asym_baseline': 1.0,
            'asym_alpha': 0.9,
            'alpha_envelope': 0.5,
            'alpha_freq': 0.05,
        },
    }

    all_results = {}

    # ---- Experiment 1.0: Classical BEADS ----
    exp_name = '1.0 Classical'
    print(f"\n[2/8] Experiment {exp_name}: running classical BEADS on test set...")
    t0 = time.time()
    beads_params = {
        'd': 1, 'fc': 0.006, 'r': 6, 'lam0': 0.4,
        'lam1': 4.0, 'lam2': 3.2, 'Nit': 30,
    }
    per_signal, agg = evaluate_classical_beads(test_y, test_x, test_f, beads_params)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s.  Peak MSE={agg['peak_mse_mean']:.6f}, BLI={agg['bli_mean']:.4f}")

    all_results[exp_name] = {
        'config': beads_params,
        'per_signal': per_signal,
        'aggregated': agg,
        'elapsed_s': elapsed,
    }
    save_results(all_results[exp_name], os.path.join(RESULTS_DIR, 'exp_1_0_classical.json'))

    # ---- Experiments 1.1 -- 1.5: Progressive models ----
    step_labels = ['3/8', '4/8', '5/8', '6/8', '7/8']
    for idx, (exp_name, loss_config) in enumerate(list(experiments.items())[1:]):
        step = step_labels[idx]
        print(f"\n[{step}] Experiment {exp_name}: training...")

        t0 = time.time()
        model = train_model(
            loss_config=loss_config,
            train_y=train_y,
            train_x_true=train_x,
            train_f_true=train_f,
            device=device,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            N=N,
            seed=seed,
        )
        train_time = time.time() - t0

        print(f"  Training done in {train_time:.1f}s. Evaluating on test set...")
        per_signal, agg = evaluate_model_on_test(model, test_y, test_x, test_f, device)
        print(f"  Peak MSE={agg['peak_mse_mean']:.6f}, BLI={agg['bli_mean']:.4f}, "
              f"Neg={agg['neg_fraction_mean']:.4f}")

        all_results[exp_name] = {
            'config': loss_config,
            'per_signal': per_signal,
            'aggregated': agg,
            'train_time_s': train_time,
        }

        # Safe filename
        safe_name = exp_name.replace(' ', '_').replace('+', 'plus').lower()
        save_results(all_results[exp_name], os.path.join(RESULTS_DIR, f'exp_{safe_name}.json'))

        # Free GPU memory
        del model
        if device == 'cuda':
            torch.cuda.empty_cache()

    # ---- Save combined summary ----
    summary = {}
    for name, res in all_results.items():
        summary[name] = res['aggregated']
    save_results(summary, os.path.join(RESULTS_DIR, 'summary.json'))

    # ---- Print comparison table ----
    print_summary_table(all_results)

    print(f"Results saved to: {RESULTS_DIR}/")
    print("Done.")


if __name__ == '__main__':
    main()
