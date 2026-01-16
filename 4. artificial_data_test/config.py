"""
Experiment configuration for artificial data tests of LBEADS.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Optional
import os


@dataclass
class DataConfig:
    signal_length: int = 1024
    n_train: int = 200
    n_val: int = 50
    n_test: int = 50
    seed: int = 42

    noise_types: List[str] = field(default_factory=lambda: ["gaussian"])
    noise_level_range: Tuple[float, float] = (0.03, 0.15)
    noise_level_choices: Optional[List[float]] = None

    baseline_type: str = "mixed"  # poly, poly_sine, lowpass_noise, mixed, spline
    poly_degree_range: Tuple[int, int] = (2, 3)
    poly_coeff_range: Tuple[float, float] = (-0.5, 0.5)
    sine_freq_range: Tuple[float, float] = (0.2, 1.5)
    sine_amp_range: Tuple[float, float] = (0.05, 0.25)
    lowpass_sigma_range: Tuple[float, float] = (20.0, 80.0)
    lowpass_scale_range: Tuple[float, float] = (0.05, 0.2)

    peak_num_range: Tuple[int, int] = (3, 6)
    peak_width_range: Tuple[float, float] = (4.0, 20.0)
    peak_amplitude_range: Tuple[float, float] = (0.5, 2.5)
    peak_center_margin: float = 0.1

    positive_dominant: bool = True
    negative_peak_prob: float = 0.15
    negative_peak_scale: float = 0.5


@dataclass
class ModelConfig:
    model_type: str = "full"  # full (regular LBEADS) or fast
    d: int = 1
    fc: float = 0.006
    num_layers: int = 12
    init_lam0: float = 0.3
    init_lam1: float = 1.5
    init_lam2: float = 1.5
    init_r: float = 6.0
    learn_r: bool = True
    init_step_size: float = 0.1
    shared_params: bool = False


@dataclass
class TrainingConfig:
    batch_size: int = 8
    epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    signal_loss_weight: float = 1.0
    baseline_loss_weight: float = 0.5
    early_stop_patience: int = 10
    lr_scheduler_patience: int = 5
    grad_clip: float = 1.0


@dataclass
class EvalConfig:
    support_threshold_ratio: float = 0.1
    psnr_max_value: Optional[float] = None


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)
    run_root: str = field(default_factory=lambda: os.path.dirname(os.path.abspath(__file__)))
    device: Optional[str] = None
    plot_samples: List[int] = field(default_factory=lambda: [0, 1, 2])


def _convert_to_json_friendly(obj):
    if isinstance(obj, dict):
        return {k: _convert_to_json_friendly(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_convert_to_json_friendly(v) for v in obj]
    if isinstance(obj, tuple):
        return [_convert_to_json_friendly(v) for v in obj]
    return obj


def config_to_dict(config: ExperimentConfig) -> dict:
    return _convert_to_json_friendly(asdict(config))
