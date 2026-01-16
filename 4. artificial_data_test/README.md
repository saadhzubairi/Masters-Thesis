# Artificial Data Test for LBEADS (Full or Fast)

This folder contains a self-contained synthetic experiment pipeline to validate LBEADS under controlled conditions. It generates artificial signals with known ground truth, trains LBEADS end-to-end, evaluates reconstruction quality, and saves thesis-ready figures and metrics in a timestamped run folder.

Models are imported from:
`Initial_implementations/1. LBEADS_NETv1/lbeads_net.py` (classes `LBEADS_NET` and `LBEADS_NET_Fast`).

## Quick start

From this folder:

```bash
python run_experiment.py
```

Optional overrides:

```bash
python run_experiment.py --model-type full
python run_experiment.py --model-type fast
python run_experiment.py --model-type full --shared-params
python run_experiment.py --model-type full --pretrain-eval-samples 5
python run_experiment.py --model-type full --skip-pretrain-eval
python run_experiment.py --fast-train-full-eval
python run_experiment.py --n-train 300 --n-test 60 --epochs 75 --layers 16
python run_experiment.py --baseline-type mixed --noise-types gaussian,laplacian --noise-levels 0.03,0.07,0.12
python run_experiment.py --plot-samples 0,1,2 --device cuda
```

## Output structure

Each run creates a timestamped folder:

```
4. artificial_data_test/
  run_YYYY-MM-DD_HH-MM-SS/
    figures/
    metrics/
    models/
    logs/
```

Key outputs:
- `figures/comparison_grid.png` (mandatory 3-column plot)
- `figures/baseline_overlay.png`, `figures/sparse_overlay.png`
- `figures/training_curves.png`, `figures/param_evolution.png`
- `metrics/*_metrics.json`, `metrics/*_summary.json`, `metrics/posttrain_metrics.csv`
- `metrics/*_dataset.npz` (train/val/test ground truth)
- `models/lbeads_full.pth` or `models/lbeads_fast.pth` (trained model + learned parameters)
- `logs/run.log`, `logs/config.json`

## Synthetic signal model

Signals follow the BEADS model:

```
y = x_true + f_true + noise
```

- `x_true`: sparse Gaussian peaks (configurable count, width, amplitude), with optional positive-dominant asymmetry.
- `f_true`: smooth baseline drift (polynomial + low-pass noise; optional sine and spline modes).
- `noise`: additive Gaussian by default (Laplacian and Student-t supported).

## Notes

- The evaluation reports MSE/MAE for signal and baseline, SNR improvement, PSNR, and sparse-support F1.
- The comparison grid plot is created from the test set and is suitable for thesis inclusion.
- Set `--no-train` to evaluate without training, or `--model-path` to load a saved model.
- The full LBEADS model (matrix-solve) is slower; reduce `--layers` or dataset sizes if needed.
- `--fast-train-full-eval` trains the fast model but generates plots/metrics with the full model.
