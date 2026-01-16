# LBEADS-NET Experiment Worker
================================

Automated hyperparameter tuning system for LBEADS-NET.

## Quick Start

```bash
# List available experiment grids
python run_experiments.py --list-grids

# Run quick test (1 experiment, ~2 min)
python run_experiments.py --grid quick

# Run Gaussian kernel tuning (42 experiments)
python run_experiments.py --grid gaussian

# Run with GPU
python run_experiments.py --grid gaussian --device cuda
```

## Available Grids

| Grid | Experiments | Focus |
|------|-------------|-------|
| `quick` | 1 | Quick test with default params |
| `gaussian` | 42 | Kernel size (21-81) × Sigma (5-30) |
| `depth` | 8 | Network depth (3-30 layers) |
| `optimizer` | 384 | LR, batch size, optimizer, scheduler |
| `loss` | 6 | Signal vs baseline weight |
| `noise` | 9 | Different noise regimes |
| `full` | 648+ | Full grid search |

## Hyperparameters

### Network Architecture
- `num_layers`: Depth of unrolled network (more = finer optimization)
- `kernel_size`: Gaussian kernel size for baseline (larger = smoother)
- `sigma`: Gaussian sigma (larger = more smoothing)

### Initial BEADS Parameters (learnable)
- `lam0_init`: Sparsity penalty on recovered signal
- `lam1_init`: First derivative penalty on baseline
- `lam2_init`: Second derivative penalty on baseline
- `r_init`: Asymmetry (r>1 penalizes positive residuals more)
- `step_size_init`: Gradient descent step size per layer

### Training
- `learning_rate`: How fast parameters are updated
- `batch_size`: Samples per gradient update
- `epochs`: Training iterations over full dataset
- `optimizer`: adam, adamw, sgd, rmsprop
- `weight_decay`: L2 regularization strength
- `scheduler`: Learning rate schedule (plateau, cosine, step, none)

### Loss Function
- `signal_weight`: Importance of signal reconstruction
- `baseline_weight`: Importance of baseline estimation

### Data Generation
- `noise_std_min/max`: Range of noise levels in training data
- `n_train/n_val`: Number of training/validation samples
- `signal_length`: Length of synthetic signals (N)

## Output Structure

Each experiment creates a folder with:

```
experiments/gaussian_20260113_120000/
├── grid_config.json           # Grid configuration
├── all_summaries.json         # Running summary of all experiments
├── final_report.txt           # Final comparison report
├── experiment_comparison.png  # Bar chart comparing all experiments
│
├── exp_0001_L10_K21_S5_LR0.01_BS8_adam/
│   ├── config.json           # Experiment hyperparameters
│   ├── model.pth             # Trained model weights
│   ├── metrics.json          # Evaluation metrics
│   ├── summary.json          # Quick summary
│   ├── comparison_grid.png   # Signal comparison plots
│   ├── training_history.png  # Loss curves
│   └── metrics_boxplot.png   # Metrics distribution
│
├── exp_0002_L10_K21_S10_LR0.01_BS8_adam/
│   └── ...
```

## Customizing Grids

Edit `config.py` to create custom grids:

```python
MY_CUSTOM_GRID = {
    "num_layers": [5, 10, 15],
    "kernel_size": [51],
    "sigma": [10.0, 15.0, 20.0],
    # ... other params with single values
}

# Add to AVAILABLE_GRIDS
AVAILABLE_GRIDS["custom"] = MY_CUSTOM_GRID
```

Then run:
```bash
python run_experiments.py --grid custom
```

## Resuming Experiments

If interrupted, resume from a specific experiment:

```bash
# Start from experiment 10
python run_experiments.py --grid gaussian --start-from 10

# Run only 5 more experiments
python run_experiments.py --grid gaussian --start-from 10 --max-experiments 5
```

## Analyzing Results

After experiments complete, check:
1. `final_report.txt` - Top configurations ranked by MSE
2. `experiment_comparison.png` - Visual comparison
3. Individual experiment folders for detailed plots
