"""
Synthetic data generation for fast LBEADS validation.

Signal model:
    y = x_true + f_true + noise

- x_true: sparse peaks
- f_true: smooth baseline drift
- noise: additive noise (default Gaussian)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import json
import numpy as np


@dataclass
class SyntheticSignal:
    y: np.ndarray
    x_true: np.ndarray
    f_true: np.ndarray
    noise: np.ndarray
    noise_type: str
    noise_level: float
    metadata: Dict


class SyntheticDataGenerator:
    def __init__(self, N: int = 1024, seed: Optional[int] = None):
        self.N = N
        self.t = np.linspace(0.0, 1.0, N)
        self.rng = np.random.default_rng(seed)

    def _gaussian_kernel(self, sigma: float) -> np.ndarray:
        size = int(max(5, 6 * sigma + 1))
        if size % 2 == 0:
            size += 1
        x = np.arange(size) - size // 2
        kernel = np.exp(-0.5 * (x / sigma) ** 2)
        return kernel / np.sum(kernel)

    def generate_baseline(
        self,
        baseline_type: str = "mixed",
        poly_degree_range: Tuple[int, int] = (2, 3),
        poly_coeff_range: Tuple[float, float] = (-0.5, 0.5),
        sine_freq_range: Tuple[float, float] = (0.2, 1.5),
        sine_amp_range: Tuple[float, float] = (0.05, 0.25),
        lowpass_sigma_range: Tuple[float, float] = (20.0, 80.0),
        lowpass_scale_range: Tuple[float, float] = (0.05, 0.2),
    ) -> Tuple[np.ndarray, Dict]:
        parts = []
        meta: Dict = {"baseline_type": baseline_type}

        if baseline_type in ("poly", "poly_sine", "mixed"):
            degree = int(self.rng.integers(poly_degree_range[0], poly_degree_range[1] + 1))
            coeffs = self.rng.uniform(poly_coeff_range[0], poly_coeff_range[1], degree + 1)
            poly = np.zeros(self.N)
            for i, coeff in enumerate(coeffs):
                poly += coeff * (self.t ** i)
            parts.append(poly)
            meta["poly_degree"] = degree
            meta["poly_coeffs"] = coeffs.tolist()

        if baseline_type in ("poly_sine", "mixed"):
            freq = self.rng.uniform(sine_freq_range[0], sine_freq_range[1])
            amp = self.rng.uniform(sine_amp_range[0], sine_amp_range[1])
            phase = self.rng.uniform(0.0, 2.0 * np.pi)
            sine = amp * np.sin(2.0 * np.pi * freq * self.t + phase)
            parts.append(sine)
            meta["sine_freq"] = float(freq)
            meta["sine_amp"] = float(amp)
            meta["sine_phase"] = float(phase)

        if baseline_type in ("lowpass_noise", "mixed"):
            sigma = self.rng.uniform(lowpass_sigma_range[0], lowpass_sigma_range[1])
            scale = self.rng.uniform(lowpass_scale_range[0], lowpass_scale_range[1])
            noise = self.rng.normal(0.0, scale, self.N)
            kernel = self._gaussian_kernel(sigma)
            smooth = np.convolve(noise, kernel, mode="same")
            parts.append(smooth)
            meta["lowpass_sigma"] = float(sigma)
            meta["lowpass_scale"] = float(scale)

        if baseline_type == "spline":
            # Optional spline baseline. Uses scipy if available, falls back to linear interp.
            n_knots = int(self.rng.integers(4, 8))
            knot_x = np.linspace(0.0, 1.0, n_knots)
            knot_y = self.rng.uniform(-0.5, 0.5, n_knots)
            try:
                from scipy.interpolate import CubicSpline

                spline = CubicSpline(knot_x, knot_y)(self.t)
            except Exception:
                spline = np.interp(self.t, knot_x, knot_y)
            parts.append(spline)
            meta["spline_knots_x"] = knot_x.tolist()
            meta["spline_knots_y"] = knot_y.tolist()

        if not parts:
            baseline = np.zeros(self.N)
        else:
            baseline = np.sum(parts, axis=0)

        return baseline, meta

    def generate_peaks(
        self,
        num_peaks_range: Tuple[int, int] = (3, 6),
        center_margin: float = 0.1,
        width_range: Tuple[float, float] = (4.0, 20.0),
        amplitude_range: Tuple[float, float] = (0.5, 2.5),
        positive_dominant: bool = True,
        negative_peak_prob: float = 0.15,
        negative_peak_scale: float = 0.5,
    ) -> Tuple[np.ndarray, Dict]:
        num_peaks = int(self.rng.integers(num_peaks_range[0], num_peaks_range[1] + 1))

        x_true = np.zeros(self.N)
        peak_info: List[Dict] = []

        min_center = int(center_margin * self.N)
        max_center = int((1.0 - center_margin) * self.N)
        indices = np.arange(self.N)

        for _ in range(num_peaks):
            center = int(self.rng.integers(min_center, max_center))
            width = float(self.rng.uniform(width_range[0], width_range[1]))
            amplitude = float(self.rng.uniform(amplitude_range[0], amplitude_range[1]))

            sign = 1.0
            if positive_dominant:
                if self.rng.random() < negative_peak_prob:
                    sign = -1.0
                    amplitude *= negative_peak_scale
            else:
                if self.rng.random() < 0.5:
                    sign = -1.0

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
            "positive_dominant": positive_dominant,
            "negative_peak_prob": float(negative_peak_prob),
            "negative_peak_scale": float(negative_peak_scale),
        }

        return x_true, params

    def generate_noise(self, noise_type: str = "gaussian", noise_level: float = 0.1) -> Tuple[np.ndarray, Dict]:
        if noise_type == "gaussian":
            noise = self.rng.normal(0.0, noise_level, self.N)
        elif noise_type == "laplacian":
            noise = self.rng.laplace(0.0, noise_level, self.N)
        elif noise_type == "student_t":
            df = 3.0
            noise = self.rng.standard_t(df, self.N) * noise_level
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")

        params = {
            "noise_type": noise_type,
            "noise_level": float(noise_level),
        }
        return noise, params

    def generate_signal(
        self,
        baseline_type: str = "mixed",
        noise_type: str = "gaussian",
        noise_level: float = 0.1,
        poly_degree_range: Tuple[int, int] = (2, 3),
        poly_coeff_range: Tuple[float, float] = (-0.5, 0.5),
        sine_freq_range: Tuple[float, float] = (0.2, 1.5),
        sine_amp_range: Tuple[float, float] = (0.05, 0.25),
        lowpass_sigma_range: Tuple[float, float] = (20.0, 80.0),
        lowpass_scale_range: Tuple[float, float] = (0.05, 0.2),
        num_peaks_range: Tuple[int, int] = (3, 6),
        center_margin: float = 0.1,
        width_range: Tuple[float, float] = (4.0, 20.0),
        amplitude_range: Tuple[float, float] = (0.5, 2.5),
        positive_dominant: bool = True,
        negative_peak_prob: float = 0.15,
        negative_peak_scale: float = 0.5,
    ) -> SyntheticSignal:
        f_true, baseline_meta = self.generate_baseline(
            baseline_type=baseline_type,
            poly_degree_range=poly_degree_range,
            poly_coeff_range=poly_coeff_range,
            sine_freq_range=sine_freq_range,
            sine_amp_range=sine_amp_range,
            lowpass_sigma_range=lowpass_sigma_range,
            lowpass_scale_range=lowpass_scale_range,
        )

        x_true, peak_meta = self.generate_peaks(
            num_peaks_range=num_peaks_range,
            center_margin=center_margin,
            width_range=width_range,
            amplitude_range=amplitude_range,
            positive_dominant=positive_dominant,
            negative_peak_prob=negative_peak_prob,
            negative_peak_scale=negative_peak_scale,
        )

        noise, noise_meta = self.generate_noise(noise_type, noise_level)

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
            noise_type=noise_type,
            noise_level=float(noise_level),
            metadata=metadata,
        )

    def generate_dataset(
        self,
        n_samples: int,
        noise_types: List[str],
        noise_level_range: Tuple[float, float],
        noise_level_choices: Optional[List[float]] = None,
        baseline_type: str = "mixed",
        poly_degree_range: Tuple[int, int] = (2, 3),
        poly_coeff_range: Tuple[float, float] = (-0.5, 0.5),
        sine_freq_range: Tuple[float, float] = (0.2, 1.5),
        sine_amp_range: Tuple[float, float] = (0.05, 0.25),
        lowpass_sigma_range: Tuple[float, float] = (20.0, 80.0),
        lowpass_scale_range: Tuple[float, float] = (0.05, 0.2),
        num_peaks_range: Tuple[int, int] = (3, 6),
        center_margin: float = 0.1,
        width_range: Tuple[float, float] = (4.0, 20.0),
        amplitude_range: Tuple[float, float] = (0.5, 2.5),
        positive_dominant: bool = True,
        negative_peak_prob: float = 0.15,
        negative_peak_scale: float = 0.5,
    ) -> List[SyntheticSignal]:
        dataset: List[SyntheticSignal] = []

        for _ in range(n_samples):
            noise_type = str(self.rng.choice(noise_types))
            if noise_level_choices:
                noise_level = float(self.rng.choice(noise_level_choices))
            else:
                noise_level = float(self.rng.uniform(noise_level_range[0], noise_level_range[1]))

            signal = self.generate_signal(
                baseline_type=baseline_type,
                noise_type=noise_type,
                noise_level=noise_level,
                poly_degree_range=poly_degree_range,
                poly_coeff_range=poly_coeff_range,
                sine_freq_range=sine_freq_range,
                sine_amp_range=sine_amp_range,
                lowpass_sigma_range=lowpass_sigma_range,
                lowpass_scale_range=lowpass_scale_range,
                num_peaks_range=num_peaks_range,
                center_margin=center_margin,
                width_range=width_range,
                amplitude_range=amplitude_range,
                positive_dominant=positive_dominant,
                negative_peak_prob=negative_peak_prob,
                negative_peak_scale=negative_peak_scale,
            )
            dataset.append(signal)

        return dataset


def _json_convert(obj):
    if isinstance(obj, dict):
        return {k: _json_convert(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_convert(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def save_dataset(dataset: List[SyntheticSignal], filepath: str) -> None:
    n = len(dataset)
    y_arr = np.stack([s.y for s in dataset])
    x_true_arr = np.stack([s.x_true for s in dataset])
    f_true_arr = np.stack([s.f_true for s in dataset])
    noise_arr = np.stack([s.noise for s in dataset])

    metadata_list = [_json_convert(s.metadata) for s in dataset]
    noise_types = [s.noise_type for s in dataset]
    noise_levels = [float(s.noise_level) for s in dataset]

    np.savez(
        filepath,
        y=y_arr,
        x_true=x_true_arr,
        f_true=f_true_arr,
        noise=noise_arr,
        noise_types=noise_types,
        noise_levels=noise_levels,
        metadata=json.dumps(metadata_list),
    )

    print(f"Saved {n} signals to {filepath}")


def load_dataset(filepath: str) -> List[SyntheticSignal]:
    data = np.load(filepath, allow_pickle=True)

    y_arr = data["y"]
    x_true_arr = data["x_true"]
    f_true_arr = data["f_true"]
    noise_arr = data["noise"]
    noise_types = data["noise_types"]
    noise_levels = data["noise_levels"]
    metadata_list = json.loads(str(data["metadata"]))

    dataset: List[SyntheticSignal] = []
    for i in range(len(y_arr)):
        signal = SyntheticSignal(
            y=y_arr[i],
            x_true=x_true_arr[i],
            f_true=f_true_arr[i],
            noise=noise_arr[i],
            noise_type=str(noise_types[i]),
            noise_level=float(noise_levels[i]),
            metadata=metadata_list[i],
        )
        dataset.append(signal)

    return dataset
