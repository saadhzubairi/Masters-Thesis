"""
Standalone synthetic data generator (numpy/scipy only).

Extracted from train.py to avoid torch dependency chain when used
by the preview endpoint.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class SyntheticSignal:
    """Container for a synthetic chromatogram signal."""
    y: np.ndarray
    x_true: np.ndarray
    f_true: np.ndarray
    noise: np.ndarray
    metadata: Dict


class SyntheticDataGenerator:
    """
    Generate BEADS-aligned synthetic data with derivative-sparse peaks.

    Signal model:  y = x_true + f_true + noise
    """

    def __init__(self, N: int = 4096, seed: Optional[int] = None, peak_shape_mode: str = "linear"):
        self.N = N
        self.t = np.linspace(0.0, 1.0, N)
        self.rng = np.random.default_rng(seed)
        if peak_shape_mode not in ("linear", "exp", "mixed"):
            raise ValueError("peak_shape_mode must be one of: 'linear', 'exp', 'mixed'")
        self.peak_shape_mode = peak_shape_mode

    @staticmethod
    def beads_peak(N: int, center: int, amplitude: float, rise_w: int, decay_w: int, plateau_w: int) -> np.ndarray:
        x = np.zeros(N, dtype=np.float64)
        start = center - rise_w
        for i in range(start, center):
            if 0 <= i < N:
                x[i] = amplitude * (i - start) / max(rise_w, 1)
        for i in range(center, center + plateau_w):
            if 0 <= i < N:
                x[i] = amplitude
        end = center + plateau_w + decay_w
        for i in range(center + plateau_w, end):
            if 0 <= i < N:
                x[i] = amplitude * (1 - (i - (center + plateau_w)) / max(decay_w, 1))
        return x

    @staticmethod
    def beads_exp_peak(N: int, center: int, amplitude: float, rise_tau: float, decay_tau: float) -> np.ndarray:
        t = np.arange(N, dtype=np.float64)
        x = np.zeros(N, dtype=np.float64)
        left = t <= center
        right = t > center
        rt = max(float(rise_tau), 1e-6)
        dt = max(float(decay_tau), 1e-6)
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
        from scipy.ndimage import gaussian_filter1d

        coeffs = self.rng.uniform(-0.5, 0.5, size=3)
        baseline = coeffs[0] + coeffs[1] * self.t + coeffs[2] * (self.t ** 2)

        sine_freq = float(self.rng.uniform(sine_freq_range[0], sine_freq_range[1]))
        sine_phase = float(self.rng.uniform(0.0, 2.0 * np.pi))
        baseline = baseline + float(sine_amp) * np.sin(2.0 * np.pi * self.t * sine_freq + sine_phase)

        baseline = gaussian_filter1d(baseline, sigma=float(smooth_sigma), mode="nearest")

        bmax = float(np.max(np.abs(baseline)))
        if bmax > 1e-12:
            baseline = baseline / bmax
        baseline_amp = float(self.rng.uniform(baseline_amp_range[0], baseline_amp_range[1]))
        baseline = baseline * baseline_amp

        offset = float(self.rng.uniform(0.0, 0.12))
        baseline = baseline - float(np.min(baseline)) + offset

        return baseline.astype(np.float64), {}

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
        mode = self.peak_shape_mode if peak_shape_mode is None else peak_shape_mode

        num_peaks = int(self.rng.integers(int(num_peaks_range[0]), int(num_peaks_range[1]) + 1))
        x_true = np.zeros(self.N, dtype=np.float64)

        low = int(center_margin)
        high = self.N - int(center_margin) + 1
        if high <= low:
            low, high = 0, self.N
        centers = np.sort(self.rng.integers(low, high, size=num_peaks))

        for center in centers:
            center = int(center)
            amplitude = float(self.rng.uniform(amplitude_range[0], amplitude_range[1]))
            peak_kind = mode
            if mode == "mixed":
                peak_kind = "linear" if self.rng.random() < 0.75 else "exp"

            if peak_kind == "linear":
                rise_w = int(self.rng.integers(rise_width_range[0], rise_width_range[1] + 1))
                decay_w = int(self.rng.integers(decay_width_range[0], decay_width_range[1] + 1))
                plateau_w = int(self.rng.integers(plateau_width_range[0], plateau_width_range[1] + 1))
                peak = self.beads_peak(self.N, center, amplitude, rise_w, decay_w, plateau_w)
            else:
                rise_tau = float(self.rng.uniform(rise_width_range[0], rise_width_range[1]))
                decay_tau = float(self.rng.uniform(decay_width_range[0], decay_width_range[1]))
                peak = self.beads_exp_peak(self.N, center, amplitude, rise_tau, decay_tau)

            x_true += peak

        x_true = np.clip(x_true, a_min=0.0, a_max=None)
        peak_max = float(np.max(x_true))
        if peak_max > 1e-8:
            x_true = x_true / peak_max

        return x_true.astype(np.float64), {}

    def generate_noise(self, noise_level: float = 0.01) -> Tuple[np.ndarray, Dict]:
        noise = self.rng.normal(0.0, float(noise_level), self.N)
        return noise.astype(np.float64), {}
