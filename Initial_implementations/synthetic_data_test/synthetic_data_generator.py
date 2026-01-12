"""
Synthetic Data Generator for LBEADS-NET Evaluation

This module generates synthetic chromatogram-like signals with known ground truth
for evaluating baseline estimation and denoising methods.

Components:
- Smooth baseline: polynomial + low-frequency sinusoid
- Sparse peaks: Gaussian-shaped peaks
- Noise: Gaussian or Laplacian

Author: Thesis Work
Date: January 2026
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Literal
import json


@dataclass
class SyntheticSignal:
    """Container for a single synthetic signal with ground truth."""
    y: np.ndarray          # Observed noisy signal
    x_true: np.ndarray     # True sparse signal (peaks only)
    f_true: np.ndarray     # True baseline
    noise: np.ndarray      # Added noise
    noise_type: str        # 'gaussian' or 'laplacian'
    noise_level: float     # Noise parameter (std or scale)
    metadata: Dict         # Additional info about generation


class SyntheticDataGenerator:
    """
    Generator for synthetic chromatogram-like signals.
    
    Creates signals with:
    - Polynomial + sinusoidal smooth baseline (ground truth)
    - Sparse Gaussian peaks (ground truth)
    - Gaussian or Laplacian noise
    
    Parameters:
        N: Signal length (default 1024)
        seed: Random seed for reproducibility
    """
    
    def __init__(self, N: int = 1024, seed: Optional[int] = None):
        self.N = N
        self.t = np.linspace(0, 1, N)  # Normalized time [0, 1]
        self.rng = np.random.default_rng(seed)
    
    def generate_baseline(self, 
                          degree: int = 2,
                          coeff_range: Tuple[float, float] = (-0.5, 0.5),
                          sin_freq_range: Tuple[float, float] = (0.5, 2.0),
                          sin_amp_range: Tuple[float, float] = (0.1, 0.3)
                          ) -> Tuple[np.ndarray, Dict]:
        """
        Generate smooth baseline (polynomial + sinusoid).
        
        Args:
            degree: Polynomial degree (2 or 3)
            coeff_range: Range for random polynomial coefficients
            sin_freq_range: Range for sinusoid frequency (cycles over full signal)
            sin_amp_range: Range for sinusoid amplitude
            
        Returns:
            f_true: Ground truth baseline (N,)
            params: Dictionary of generation parameters
        """
        # Generate polynomial baseline
        coeffs = self.rng.uniform(coeff_range[0], coeff_range[1], degree + 1)
        f_poly = np.zeros(self.N)
        for i, coeff in enumerate(coeffs):
            f_poly += coeff * (self.t ** i)
        
        # Generate sinusoidal component
        sin_freq = self.rng.uniform(sin_freq_range[0], sin_freq_range[1])
        sin_amp = self.rng.uniform(sin_amp_range[0], sin_amp_range[1])
        sin_phase = self.rng.uniform(0, 2 * np.pi)
        f_sin = sin_amp * np.sin(2 * np.pi * sin_freq * self.t + sin_phase)
        
        # Combine
        f_true = f_poly + f_sin
        
        params = {
            'degree': degree,
            'poly_coeffs': coeffs.tolist(),
            'sin_freq': sin_freq,
            'sin_amp': sin_amp,
            'sin_phase': sin_phase
        }
        
        return f_true, params
    
    def generate_peaks(self,
                       num_peaks_range: Tuple[int, int] = (3, 6),
                       center_margin: float = 0.1,
                       width_range: Tuple[float, float] = (10, 30),
                       amplitude_range: Tuple[float, float] = (1.0, 3.0)
                       ) -> Tuple[np.ndarray, Dict]:
        """
        Generate sparse Gaussian peaks.
        
        Args:
            num_peaks_range: Range for number of peaks (min, max inclusive)
            center_margin: Fraction of signal to exclude from edges (e.g., 0.1 means [0.1N, 0.9N])
            width_range: Range for peak width (sigma in samples)
            amplitude_range: Range for peak amplitude
            
        Returns:
            x_true: Ground truth sparse signal (N,)
            params: Dictionary of generation parameters
        """
        num_peaks = self.rng.integers(num_peaks_range[0], num_peaks_range[1] + 1)
        
        x_true = np.zeros(self.N)
        peak_info = []
        
        # Define valid center range
        min_center = int(center_margin * self.N)
        max_center = int((1 - center_margin) * self.N)
        
        for _ in range(num_peaks):
            center = self.rng.integers(min_center, max_center)
            width = self.rng.uniform(width_range[0], width_range[1])
            amplitude = self.rng.uniform(amplitude_range[0], amplitude_range[1])
            
            # Generate Gaussian peak
            indices = np.arange(self.N)
            peak = amplitude * np.exp(-((indices - center) ** 2) / (2 * width ** 2))
            x_true += peak
            
            peak_info.append({
                'center': int(center),
                'width': float(width),
                'amplitude': float(amplitude)
            })
        
        params = {
            'num_peaks': num_peaks,
            'peaks': peak_info
        }
        
        return x_true, params
    
    def generate_noise(self,
                       noise_type: Literal['gaussian', 'laplacian'] = 'gaussian',
                       noise_level: float = 0.1
                       ) -> Tuple[np.ndarray, Dict]:
        """
        Generate noise (Gaussian or Laplacian).
        
        Args:
            noise_type: 'gaussian' or 'laplacian'
            noise_level: Standard deviation (Gaussian) or scale parameter (Laplacian)
            
        Returns:
            noise: Noise array (N,)
            params: Dictionary of noise parameters
        """
        if noise_type == 'gaussian':
            noise = self.rng.normal(0, noise_level, self.N)
        elif noise_type == 'laplacian':
            noise = self.rng.laplace(0, noise_level, self.N)
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")
        
        params = {
            'noise_type': noise_type,
            'noise_level': noise_level
        }
        
        return noise, params
    
    def generate_signal(self,
                        baseline_params: Optional[Dict] = None,
                        peak_params: Optional[Dict] = None,
                        noise_type: Literal['gaussian', 'laplacian'] = 'gaussian',
                        noise_level: float = 0.1
                        ) -> SyntheticSignal:
        """
        Generate a complete synthetic signal with ground truth.
        
        Args:
            baseline_params: Parameters for baseline generation (or None for defaults)
            peak_params: Parameters for peak generation (or None for defaults)
            noise_type: Type of noise ('gaussian' or 'laplacian')
            noise_level: Noise level (std or scale)
            
        Returns:
            SyntheticSignal object containing all components
        """
        # Generate baseline
        baseline_kwargs = baseline_params or {}
        f_true, baseline_meta = self.generate_baseline(**baseline_kwargs)
        
        # Generate peaks
        peak_kwargs = peak_params or {}
        x_true, peak_meta = self.generate_peaks(**peak_kwargs)
        
        # Generate noise
        noise, noise_meta = self.generate_noise(noise_type, noise_level)
        
        # Construct observed signal
        y = x_true + f_true + noise
        
        # Compile metadata
        metadata = {
            'N': self.N,
            'baseline': baseline_meta,
            'peaks': peak_meta,
            'noise': noise_meta
        }
        
        return SyntheticSignal(
            y=y,
            x_true=x_true,
            f_true=f_true,
            noise=noise,
            noise_type=noise_type,
            noise_level=noise_level,
            metadata=metadata
        )
    
    def generate_dataset(self,
                         n_samples: int = 30,
                         noise_types: List[str] = ['gaussian', 'laplacian'],
                         noise_levels: List[float] = [0.05, 0.10, 0.15],
                         vary_baseline_degree: bool = True
                         ) -> List[SyntheticSignal]:
        """
        Generate a complete dataset of synthetic signals.
        
        Args:
            n_samples: Total number of samples to generate
            noise_types: List of noise types to sample from
            noise_levels: List of noise levels to sample from
            vary_baseline_degree: If True, randomly choose degree 2 or 3
            
        Returns:
            List of SyntheticSignal objects
        """
        dataset = []
        
        for i in range(n_samples):
            # Randomly select noise parameters
            noise_type = noise_types[self.rng.integers(len(noise_types))]
            noise_level = noise_levels[self.rng.integers(len(noise_levels))]
            
            # Optionally vary baseline degree
            baseline_params = {}
            if vary_baseline_degree:
                baseline_params['degree'] = self.rng.choice([2, 3])
            
            signal = self.generate_signal(
                baseline_params=baseline_params,
                noise_type=noise_type,
                noise_level=noise_level
            )
            dataset.append(signal)
        
        return dataset
    
    def generate_stratified_dataset(self,
                                    n_easy: int = 10,
                                    n_medium: int = 10,
                                    n_hard: int = 10
                                    ) -> Dict[str, List[SyntheticSignal]]:
        """
        Generate a stratified dataset with easy, medium, and hard cases.
        
        Args:
            n_easy: Number of easy cases (low noise)
            n_medium: Number of medium cases (moderate noise)
            n_hard: Number of hard cases (high noise, Laplacian)
            
        Returns:
            Dictionary with 'easy', 'medium', 'hard' keys
        """
        dataset = {'easy': [], 'medium': [], 'hard': []}
        
        # Easy: Low Gaussian noise
        for _ in range(n_easy):
            signal = self.generate_signal(
                noise_type='gaussian',
                noise_level=self.rng.uniform(0.03, 0.07)
            )
            dataset['easy'].append(signal)
        
        # Medium: Moderate Gaussian noise
        for _ in range(n_medium):
            signal = self.generate_signal(
                noise_type='gaussian',
                noise_level=self.rng.uniform(0.08, 0.12)
            )
            dataset['medium'].append(signal)
        
        # Hard: High noise (mix of Gaussian and Laplacian)
        for i in range(n_hard):
            noise_type = 'laplacian' if i % 2 == 0 else 'gaussian'
            signal = self.generate_signal(
                noise_type=noise_type,
                noise_level=self.rng.uniform(0.12, 0.18)
            )
            dataset['hard'].append(signal)
        
        return dataset


def convert_to_json_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def save_dataset(dataset: List[SyntheticSignal], filepath: str):
    """Save dataset to .npz file."""
    n = len(dataset)
    y_arr = np.stack([s.y for s in dataset])
    x_true_arr = np.stack([s.x_true for s in dataset])
    f_true_arr = np.stack([s.f_true for s in dataset])
    noise_arr = np.stack([s.noise for s in dataset])
    
    # Save metadata as JSON string (convert numpy types first)
    metadata_list = [convert_to_json_serializable(s.metadata) for s in dataset]
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
        metadata=json.dumps(metadata_list)
    )
    print(f"Saved {n} signals to {filepath}")


def load_dataset(filepath: str) -> List[SyntheticSignal]:
    """Load dataset from .npz file."""
    data = np.load(filepath, allow_pickle=True)
    
    y_arr = data['y']
    x_true_arr = data['x_true']
    f_true_arr = data['f_true']
    noise_arr = data['noise']
    noise_types = data['noise_types']
    noise_levels = data['noise_levels']
    metadata_list = json.loads(str(data['metadata']))
    
    dataset = []
    for i in range(len(y_arr)):
        signal = SyntheticSignal(
            y=y_arr[i],
            x_true=x_true_arr[i],
            f_true=f_true_arr[i],
            noise=noise_arr[i],
            noise_type=str(noise_types[i]),
            noise_level=float(noise_levels[i]),
            metadata=metadata_list[i]
        )
        dataset.append(signal)
    
    print(f"Loaded {len(dataset)} signals from {filepath}")
    return dataset


if __name__ == "__main__":
    # Test the generator
    import matplotlib.pyplot as plt
    
    print("Testing Synthetic Data Generator")
    print("=" * 50)
    
    generator = SyntheticDataGenerator(N=1024, seed=42)
    
    # Generate a single signal
    signal = generator.generate_signal(
        noise_type='gaussian',
        noise_level=0.1
    )
    
    print(f"Signal length: {len(signal.y)}")
    print(f"Noise type: {signal.noise_type}")
    print(f"Noise level: {signal.noise_level}")
    print(f"Number of peaks: {signal.metadata['peaks']['num_peaks']}")
    
    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    
    axes[0].plot(signal.y, 'b', linewidth=0.5, label='y (observed)')
    axes[0].set_title('Observed Signal (y = x_true + f_true + noise)')
    axes[0].legend()
    
    axes[1].plot(signal.f_true, 'g', linewidth=1, label='f_true (baseline)')
    axes[1].plot(signal.x_true, 'r', linewidth=1, label='x_true (peaks)')
    axes[1].legend()
    axes[1].set_title('Ground Truth Components')
    
    axes[2].plot(signal.noise, 'gray', linewidth=0.5, label='noise')
    axes[2].legend()
    axes[2].set_title('Noise')
    
    plt.tight_layout()
    plt.savefig('test_synthetic_data.png', dpi=150)
    print("\nSaved test plot to test_synthetic_data.png")
    
    # Generate full dataset
    print("\nGenerating full dataset...")
    dataset = generator.generate_dataset(n_samples=30)
    print(f"Generated {len(dataset)} signals")
    
    # Count noise types
    gaussian_count = sum(1 for s in dataset if s.noise_type == 'gaussian')
    laplacian_count = sum(1 for s in dataset if s.noise_type == 'laplacian')
    print(f"  Gaussian noise: {gaussian_count}")
    print(f"  Laplacian noise: {laplacian_count}")
    
    # Save and reload test
    save_dataset(dataset, 'test_dataset.npz')
    loaded = load_dataset('test_dataset.npz')
    print(f"Reload successful: {len(loaded)} signals")
    
    plt.show()
