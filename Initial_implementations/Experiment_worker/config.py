"""
Hyperparameter Configuration for LBEADS-NET Experiments
========================================================

All tunable hyperparameters are defined here as lists.
The worker will train a model for each combination.

HYPERPARAMETER CATEGORIES:
--------------------------
1. NETWORK ARCHITECTURE
   - num_layers: Number of unrolled iterations (depth of network)
   - kernel_size: Gaussian smoothing kernel size for baseline estimation
   - sigma: Gaussian smoothing sigma (controls smoothness)

2. INITIAL BEADS PARAMETERS (learnable starting points)
   - lam0_init: Sparsity penalty on signal x
   - lam1_init: Penalty on first derivative of baseline
   - lam2_init: Penalty on second derivative of baseline  
   - r_init: Asymmetry parameter (r>1 penalizes positive residuals more)
   - step_size_init: Gradient descent step size

3. TRAINING HYPERPARAMETERS
   - learning_rate: Optimizer learning rate
   - batch_size: Training batch size
   - epochs: Number of training epochs
   - optimizer: Optimizer type (adam, adamw, sgd, rmsprop)
   - weight_decay: L2 regularization
   - scheduler: Learning rate scheduler (none, plateau, cosine, step)

4. LOSS FUNCTION
   - signal_weight: Weight for signal reconstruction loss
   - baseline_weight: Weight for baseline estimation loss

5. DATA GENERATION
   - noise_std_range: (min, max) noise standard deviation
   - noise_types: Types of noise to include
   - n_train: Number of training samples
   - n_val: Number of validation samples
   - signal_length: Length of synthetic signals (N)

"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from itertools import product
import json

# =============================================================================
# HYPERPARAMETER GRID DEFINITION
# =============================================================================

HYPERPARAMETER_GRID = {
    # -------------------------------------------------------------------------
    # NETWORK ARCHITECTURE
    # -------------------------------------------------------------------------
    "num_layers": [5, 10, 15],              # Unrolled iterations
    "kernel_size": [31, 51, 71],            # Gaussian kernel size (must be odd)
    "sigma": [10.0, 15.0, 20.0],            # Gaussian sigma for baseline smoothing
    "baseline_method": ["gaussian"],        # gaussian or exponential
    "ema_alpha": [0.02],                    # EMA alpha (for exponential method)
    
    # -------------------------------------------------------------------------
    # INITIAL BEADS PARAMETERS (learnable)
    # -------------------------------------------------------------------------
    "lam0_init": [0.3],                     # Initial sparsity penalty
    "lam1_init": [3.0],                     # Initial 1st derivative penalty
    "lam2_init": [3.0],                     # Initial 2nd derivative penalty
    "r_init": [6.0],                        # Initial asymmetry parameter
    "step_size_init": [0.1],                # Initial gradient step size
    
    # -------------------------------------------------------------------------
    # TRAINING HYPERPARAMETERS
    # -------------------------------------------------------------------------
    "learning_rate": [0.01, 0.005, 0.001],  # Optimizer learning rate
    "batch_size": [8, 16],                  # Training batch size
    "epochs": [50],                         # Training epochs
    "optimizer": ["adam", "adamw"],         # Optimizer type
    "weight_decay": [0.0, 1e-5],            # L2 regularization
    "scheduler": ["plateau", "cosine"],     # LR scheduler
    
    # -------------------------------------------------------------------------
    # LOSS FUNCTION WEIGHTS
    # -------------------------------------------------------------------------
    "signal_weight": [0.7],                 # Weight for signal loss
    "baseline_weight": [0.3],               # Weight for baseline loss
    
    # -------------------------------------------------------------------------
    # DATA GENERATION
    # -------------------------------------------------------------------------
    "noise_std_min": [0.05],                # Minimum noise std
    "noise_std_max": [0.15],                # Maximum noise std
    "n_train": [150],                       # Training samples
    "n_val": [30],                          # Validation samples
    "signal_length": [1024],                # Signal length N
}

# =============================================================================
# QUICK EXPERIMENT PRESETS
# =============================================================================

# For quick testing - reduced grid
QUICK_TEST_GRID = {
    "num_layers": [10],
    "kernel_size": [51],
    "sigma": [15.0],
    "baseline_method": ["gaussian"],
    "ema_alpha": [0.02],
    "lam0_init": [0.3],
    "lam1_init": [3.0],
    "lam2_init": [3.0],
    "r_init": [6.0],
    "step_size_init": [0.1],
    "learning_rate": [0.01],
    "batch_size": [8],
    "epochs": [20],
    "optimizer": ["adam"],
    "weight_decay": [0.0],
    "scheduler": ["plateau"],
    "signal_weight": [0.7],
    "baseline_weight": [0.3],
    "noise_std_min": [0.05],
    "noise_std_max": [0.15],
    "n_train": [100],
    "n_val": [20],
    "signal_length": [1024],
}

# Focused on Gaussian kernel parameters (your current issue)
GAUSSIAN_TUNING_GRID = {
    "num_layers": [10],
    "kernel_size": [21, 31, 41, 51, 61, 71, 81],  # Extensive kernel size search
    "sigma": [5.0, 10.0, 15.0, 20.0, 25.0, 30.0], # Extensive sigma search
    "baseline_method": ["gaussian"],
    "ema_alpha": [0.02],
    "lam0_init": [0.3],
    "lam1_init": [3.0],
    "lam2_init": [3.0],
    "r_init": [6.0],
    "step_size_init": [0.1],
    "learning_rate": [0.01],
    "batch_size": [8],
    "epochs": [30],
    "optimizer": ["adam"],
    "weight_decay": [0.0],
    "scheduler": ["plateau"],
    "signal_weight": [0.7],
    "baseline_weight": [0.3],
    "noise_std_min": [0.05],
    "noise_std_max": [0.15],
    "n_train": [150],
    "n_val": [30],
    "signal_length": [1024],
}

# Focused on network depth
DEPTH_TUNING_GRID = {
    "num_layers": [3, 5, 7, 10, 15, 20, 25, 30],
    "kernel_size": [51],
    "sigma": [15.0],
    "baseline_method": ["gaussian"],
    "ema_alpha": [0.02],
    "lam0_init": [0.3],
    "lam1_init": [3.0],
    "lam2_init": [3.0],
    "r_init": [6.0],
    "step_size_init": [0.1],
    "learning_rate": [0.01],
    "batch_size": [8],
    "epochs": [50],
    "optimizer": ["adam"],
    "weight_decay": [0.0],
    "scheduler": ["plateau"],
    "signal_weight": [0.7],
    "baseline_weight": [0.3],
    "noise_std_min": [0.05],
    "noise_std_max": [0.15],
    "n_train": [150],
    "n_val": [30],
    "signal_length": [1024],
}

# Focused on optimizer comparison
OPTIMIZER_TUNING_GRID = {
    "num_layers": [10],
    "kernel_size": [51],
    "sigma": [15.0],
    "baseline_method": ["gaussian"],
    "ema_alpha": [0.02],
    "lam0_init": [0.3],
    "lam1_init": [3.0],
    "lam2_init": [3.0],
    "r_init": [6.0],
    "step_size_init": [0.1],
    "learning_rate": [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005],
    "batch_size": [4, 8, 16, 32],
    "epochs": [50],
    "optimizer": ["adam", "adamw", "sgd", "rmsprop"],
    "weight_decay": [0.0, 1e-5, 1e-4],
    "scheduler": ["none", "plateau", "cosine", "step"],
    "signal_weight": [0.7],
    "baseline_weight": [0.3],
    "noise_std_min": [0.05],
    "noise_std_max": [0.15],
    "n_train": [150],
    "n_val": [30],
    "signal_length": [1024],
}

# Focused on loss weighting
LOSS_WEIGHT_GRID = {
    "num_layers": [10],
    "kernel_size": [51],
    "sigma": [15.0],
    "baseline_method": ["gaussian"],
    "ema_alpha": [0.02],
    "lam0_init": [0.3],
    "lam1_init": [3.0],
    "lam2_init": [3.0],
    "r_init": [6.0],
    "step_size_init": [0.1],
    "learning_rate": [0.01],
    "batch_size": [8],
    "epochs": [50],
    "optimizer": ["adam"],
    "weight_decay": [0.0],
    "scheduler": ["plateau"],
    "signal_weight": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "baseline_weight": [0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
    "noise_std_min": [0.05],
    "noise_std_max": [0.15],
    "n_train": [150],
    "n_val": [30],
    "signal_length": [1024],
}

# Noise regime experiments
NOISE_REGIME_GRID = {
    "num_layers": [10],
    "kernel_size": [51],
    "sigma": [15.0],
    "baseline_method": ["gaussian"],        # gaussian or exponential
    "ema_alpha": [0.02],                    # For exponential method
    "lam0_init": [0.3],
    "lam1_init": [3.0],
    "lam2_init": [3.0],
    "r_init": [6.0],
    "step_size_init": [0.1],
    "learning_rate": [0.01],
    "batch_size": [8],
    "epochs": [50],
    "optimizer": ["adam"],
    "weight_decay": [0.0],
    "scheduler": ["plateau"],
    "signal_weight": [0.7],
    "baseline_weight": [0.3],
    "noise_std_min": [0.01, 0.05, 0.10],   # Different noise regimes
    "noise_std_max": [0.05, 0.15, 0.25],   # Different noise regimes
    "n_train": [150],
    "n_val": [30],
    "signal_length": [1024],
}

# =============================================================================
# ORIGINAL TRAINING CONFIG (matches train_lbeads.py)
# =============================================================================
# This is the EXACT config used in the initial successful training run:
#   python train_lbeads.py --epochs 50 --layers 10 --n-train 150 --n-val 30
ORIGINAL_CONFIG_GRID = {
    "num_layers": [10],
    "kernel_size": [51],                    # Not used with exponential
    "sigma": [15.0],                        # Not used with exponential
    "baseline_method": ["exponential"],     # Original used exponential MA
    "ema_alpha": [0.02],                    # Original alpha value
    "lam0_init": [0.3],
    "lam1_init": [3.0],
    "lam2_init": [3.0],
    "r_init": [6.0],
    "step_size_init": [0.1],
    "learning_rate": [0.001],               # Original: 1e-3
    "batch_size": [8],
    "epochs": [50],
    "optimizer": ["adam"],
    "weight_decay": [1e-5],                 # Original used weight decay
    "scheduler": ["plateau"],
    "signal_weight": [0.67],                # Original: 1.0/(1.0+0.5) ≈ 0.67
    "baseline_weight": [0.33],              # Original: 0.5/(1.0+0.5) ≈ 0.33
    "noise_std_min": [0.05],
    "noise_std_max": [0.15],
    "n_train": [150],
    "n_val": [30],
    "signal_length": [1024],
}

# Baseline method comparison (Gaussian vs Exponential)
BASELINE_METHOD_GRID = {
    "num_layers": [10],
    "kernel_size": [31, 51, 71],            # For Gaussian
    "sigma": [10.0, 15.0, 20.0],            # For Gaussian
    "baseline_method": ["gaussian", "exponential"],
    "ema_alpha": [0.01, 0.02, 0.05],        # For Exponential
    "lam0_init": [0.3],
    "lam1_init": [3.0],
    "lam2_init": [3.0],
    "r_init": [6.0],
    "step_size_init": [0.1],
    "learning_rate": [0.001],
    "batch_size": [8],
    "epochs": [50],
    "optimizer": ["adam"],
    "weight_decay": [0.0],
    "scheduler": ["plateau"],
    "signal_weight": [0.7],
    "baseline_weight": [0.3],
    "noise_std_min": [0.05],
    "noise_std_max": [0.15],
    "n_train": [150],
    "n_val": [30],
    "signal_length": [1024],
}

# =============================================================================
# EXPERIMENT CONFIG CLASS
# =============================================================================

@dataclass
class ExperimentConfig:
    """Single experiment configuration."""
    # Network architecture
    num_layers: int = 10
    kernel_size: int = 51
    sigma: float = 15.0
    baseline_method: str = "gaussian"      # gaussian or exponential
    ema_alpha: float = 0.02                 # For exponential method
    
    # Initial BEADS parameters
    lam0_init: float = 0.3
    lam1_init: float = 3.0
    lam2_init: float = 3.0
    r_init: float = 6.0
    step_size_init: float = 0.1
    
    # Training hyperparameters
    learning_rate: float = 0.01
    batch_size: int = 8
    epochs: int = 50
    optimizer: str = "adam"
    weight_decay: float = 0.0
    scheduler: str = "plateau"
    
    # Loss weights
    signal_weight: float = 0.7
    baseline_weight: float = 0.3
    
    # Data generation
    noise_std_min: float = 0.05
    noise_std_max: float = 0.15
    n_train: int = 150
    n_val: int = 30
    signal_length: int = 1024
    
    # Experiment metadata
    experiment_id: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "num_layers": self.num_layers,
            "kernel_size": self.kernel_size,
            "sigma": self.sigma,
            "baseline_method": self.baseline_method,
            "ema_alpha": self.ema_alpha,
            "lam0_init": self.lam0_init,
            "lam1_init": self.lam1_init,
            "lam2_init": self.lam2_init,
            "r_init": self.r_init,
            "step_size_init": self.step_size_init,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "optimizer": self.optimizer,
            "weight_decay": self.weight_decay,
            "scheduler": self.scheduler,
            "signal_weight": self.signal_weight,
            "baseline_weight": self.baseline_weight,
            "noise_std_min": self.noise_std_min,
            "noise_std_max": self.noise_std_max,
            "n_train": self.n_train,
            "n_val": self.n_val,
            "signal_length": self.signal_length,
            "experiment_id": self.experiment_id,
        }
    
    def to_json(self, path: str):
        """Save config to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ExperimentConfig':
        """Create from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
    
    def get_short_name(self) -> str:
        """Generate short descriptive name for folder."""
        if self.baseline_method == 'exponential':
            return (f"L{self.num_layers}_EMA{self.ema_alpha}_"
                    f"LR{self.learning_rate}_BS{self.batch_size}_{self.optimizer}")
        else:
            return (f"L{self.num_layers}_K{self.kernel_size}_S{self.sigma:.0f}_"
                    f"LR{self.learning_rate}_BS{self.batch_size}_{self.optimizer}")


def generate_experiment_configs(grid: Dict[str, List]) -> List[ExperimentConfig]:
    """
    Generate all experiment configurations from a hyperparameter grid.
    
    Args:
        grid: Dictionary mapping hyperparameter names to lists of values
        
    Returns:
        List of ExperimentConfig objects for each combination
    """
    # Get all parameter names and their values
    param_names = list(grid.keys())
    param_values = [grid[name] for name in param_names]
    
    # Generate all combinations
    configs = []
    for i, combo in enumerate(product(*param_values)):
        # Create config dict
        config_dict = dict(zip(param_names, combo))
        config_dict["experiment_id"] = f"exp_{i+1:04d}"
        
        # Create ExperimentConfig
        config = ExperimentConfig.from_dict(config_dict)
        configs.append(config)
    
    return configs


def count_experiments(grid: Dict[str, List]) -> int:
    """Count total number of experiments in a grid."""
    total = 1
    for values in grid.values():
        total *= len(values)
    return total


# =============================================================================
# AVAILABLE GRIDS
# =============================================================================

AVAILABLE_GRIDS = {
    "full": HYPERPARAMETER_GRID,
    "quick": QUICK_TEST_GRID,
    "gaussian": GAUSSIAN_TUNING_GRID,
    "depth": DEPTH_TUNING_GRID,
    "optimizer": OPTIMIZER_TUNING_GRID,
    "loss": LOSS_WEIGHT_GRID,
    "noise": NOISE_REGIME_GRID,
    "original": ORIGINAL_CONFIG_GRID,
    "baseline": BASELINE_METHOD_GRID,
}


if __name__ == "__main__":
    # Print summary of all grids
    print("=" * 60)
    print("AVAILABLE HYPERPARAMETER GRIDS")
    print("=" * 60)
    
    for name, grid in AVAILABLE_GRIDS.items():
        n_exp = count_experiments(grid)
        print(f"\n{name.upper()} ({n_exp} experiments):")
        for param, values in grid.items():
            if len(values) > 1:
                print(f"  {param}: {values}")
    
    print("\n" + "=" * 60)
    print("HYPERPARAMETER DESCRIPTIONS")
    print("=" * 60)
    print("""
NETWORK ARCHITECTURE:
  num_layers     : Depth of unrolled network (more = finer optimization)
  kernel_size    : Gaussian kernel size for baseline (larger = smoother)
  sigma          : Gaussian sigma (larger = more smoothing)

INITIAL BEADS PARAMETERS:
  lam0_init      : Sparsity penalty on recovered signal
  lam1_init      : First derivative penalty on baseline
  lam2_init      : Second derivative penalty on baseline
  r_init         : Asymmetry (r>1 penalizes positive residuals more)
  step_size_init : Gradient descent step size per layer

TRAINING:
  learning_rate  : How fast parameters are updated
  batch_size     : Samples per gradient update
  epochs         : Training iterations over full dataset
  optimizer      : adam, adamw, sgd, rmsprop
  weight_decay   : L2 regularization strength
  scheduler      : Learning rate schedule (plateau, cosine, step, none)

LOSS FUNCTION:
  signal_weight  : Importance of signal reconstruction
  baseline_weight: Importance of baseline estimation

DATA GENERATION:
  noise_std_min/max : Range of noise levels in training data
  n_train/n_val     : Number of training/validation samples
  signal_length     : Length of synthetic signals (N)
""")
