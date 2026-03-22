# ML Experiment Hub — Design Spec

## Overview

A local web app for orchestrating LBEADS-NET training jobs. Configure hyperparameters and alpha values in a UI, run multiple training jobs in parallel, monitor progress in real-time, and view results (plots, demos, metrics) all in one place.

## Architecture

**Two-server setup:**

- **Next.js Frontend** (`:3000`) — Dashboard UI, config forms, live monitoring, results viewer
- **FastAPI Backend** (`:8000`) — REST API, SSE streaming, process management, experiment storage

**Data flow:** Frontend → HTTP/SSE → FastAPI → `subprocess.Popen` → Python training runner → stdout JSON lines → FastAPI forwards via SSE → Frontend updates live.

### Directory Structure

```
tooling/
├── app/                          # Next.js frontend (existing)
│   ├── layout.tsx
│   ├── page.tsx                  # Dashboard
│   ├── runs/
│   │   ├── new/page.tsx          # New run config form
│   │   └── [id]/page.tsx         # Run detail / monitoring
│   └── globals.css
├── backend/
│   ├── server.py                 # FastAPI app
│   ├── process_manager.py        # Spawn, track, kill training processes
│   ├── experiment_store.py       # Read/write experiment JSON + files
│   ├── train_runner.py           # Wrapper: accepts JSON config, runs train → demo pipeline
│   ├── models/
│   │   ├── lbeads_net.py         # Model architecture (copied from v6)
│   │   ├── train.py              # Training logic (adapted for JSON config + JSON line output)
│   │   ├── demo.py               # Synthetic inference demo
│   │   └── demo_chromatogram.py  # Real chromatogram demo
│   ├── experiments/              # Experiment output storage
│   │   └── {run_id}/
│   │       ├── config.json       # Full training config
│   │       ├── metrics.json      # Per-epoch losses + final metrics
│   │       ├── checkpoint.pth    # Trained model weights
│   │       ├── training_plot.png # 6-panel training results
│   │       ├── demo/             # demo.py outputs (raw/, hybrid/, hybrid-snr/)
│   │       └── demo_chrom/       # demo_chromatogram.py outputs
│   └── requirements.txt
├── package.json
└── next.config.ts
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/runs` | Start a new training run. Body: full config JSON. Returns run ID. |
| `GET` | `/runs` | List all experiments (summary: id, name, status, key metrics). |
| `GET` | `/runs/{id}` | Full experiment details (config, all metrics, file paths). |
| `GET` | `/runs/{id}/stream` | SSE endpoint. Streams JSON events for live epoch updates. |
| `POST` | `/runs/{id}/stop` | Kill a running training process. |
| `GET` | `/runs/{id}/files/{path}` | Serve experiment files (plots, checkpoints). Path is validated to stay within the run's directory (no traversal). |

## Frontend Pages

### 1. Dashboard (sidebar: "Dashboard")

- Sidebar navigation: Dashboard, New Run, Experiments
- Main area: list of all runs sorted by recency
- Each run card shows: name, status badge (RUNNING/COMPLETE/FAILED), progress bar, epoch count, train/test loss, elapsed time
- Running jobs update live via SSE
- "+ New Run" button in top right

### 2. New Run (sidebar: "New Run")

**Clone & modify workflow:**

- Dropdown to clone config from any previous run, or start with defaults
- Run name text field

**Config sections (collapsible):**

1. **Model Architecture** — N (signal length), num_layers, fc (filter cutoff), solve_cg_iters, lowpass_cg_iters, shared_params toggle
2. **Training Parameters** — learning_rate, batch_size, num_samples, noise_level, train_ratio, seed
3. **Loss Weights** — All alpha values with slider (range 0–10, step 0.01) + text input:
   - alpha_mse, alpha_l1, alpha_tv, alpha_smooth, alpha_neg, alpha_baseline, alpha_leakage, alpha_ortho, alpha_baseline_tv
   - Additional: peak_mask_rel_threshold, peak_mask_abs_min, use_huber toggle, huber_delta
4. **Training Stages** — Dynamic list of stages. Each stage has:
   - Name (text input)
   - Epoch count
   - Alpha overrides (toggle which losses are active, override values)
   - Add/remove stages, drag to reorder
   - Unset values inherit from global loss weights

**Actions:** Start Training (no draft system — just clone a previous run when you want to reuse a config)

### 3. Run Detail (click any run)

**Top stats bar:** Epoch progress, current train/test loss, active stage, elapsed time, stop button (if running).

**Progress bar:** Overall completion with stage boundary markers.

**Live loss chart:** Real-time loss curve (train + test), stage boundaries marked.

**Epoch table:** Every epoch as a row with columns: epoch, stage, train loss, test loss, key loss components. Click to expand full breakdown:
- All 9 loss components: recon_mse, l1_sparsity, total_variation, baseline_smooth, baseline_recon, baseline_leakage, peak_baseline_ortho, non_negativity, baseline_tv
- All learned parameters: lam0, lam1, lam2, r, step, output_gain

**Results gallery (after completion):**
- Training plot (6-panel)
- demo.py results (3 variants: raw, hybrid, hybrid-snr)
- demo_chromatogram.py comparison plot
- Click any image to view full-size

## Training Runner

`train_runner.py` is the bridge between FastAPI and the existing training code.

**Input:** JSON config file path via `--config` CLI argument.

**Output:** JSON lines to stdout (must flush after each line), one per event:

```json
{"type": "started", "run_id": "1719000000", "total_epochs": 25}
{"type": "epoch", "epoch": 1, "stage": "A", "train_loss": 0.045, "test_loss": 0.052, "components": {"recon_mse": 0.04, "l1_sparsity": 0.001, "total_variation": 0.0008, "baseline_smooth": 0.002, "baseline_recon": 0.001, "baseline_leakage": 0.0005, "peak_baseline_ortho": 0.0003, "non_negativity": 0.0001, "baseline_tv": 0.0}, "learned_params": {"lam0": 0.002, "lam1": 0.31, "lam2": 0.29, "r": 6.1, "step": 0.85, "output_gain": 1.02}, "elapsed_s": 12.3}
{"type": "stage_change", "from": "A", "to": "B", "epoch": 6}
{"type": "training_done", "checkpoint": "checkpoint.pth", "final_metrics": {"train_mse": 0.002, "test_mse": 0.003, "test_mae": 0.04, "test_psnr": 28.5, "test_correlation": 0.987}}
{"type": "demo_started", "demo": "demo.py"}
{"type": "demo_done", "demo": "demo.py", "outputs": ["demo/raw/plot.png", "demo/hybrid/plot.png", "demo/hybrid-snr/plot.png"]}
{"type": "demo_error", "demo": "demo.py", "error": "error message here"}
{"type": "demo_done", "demo": "demo_chromatogram.py", "outputs": ["demo_chrom/comparison.png"]}
{"type": "error", "message": "NaN loss detected at epoch 8", "fatal": true}
{"type": "complete"}
```

**Pipeline:** train → save checkpoint → run demo.py → run demo_chromatogram.py → emit complete. If a demo fails, emit `demo_error` and continue to the next demo. If training itself fails, emit `error` with `fatal: true`.

**Adaptation of train.py:**
- `main()` replaced with a function that accepts a config dict
- Per-epoch callback that prints JSON lines instead of console logging
- Output paths configurable (write to experiment directory, not cwd)
- Imports remain the same; model architecture untouched

## Process Management

`process_manager.py` handles:

- Spawning training processes via `subprocess.Popen`
- Tracking PIDs and status (running/complete/failed) in memory
- Reading stdout lines and buffering for SSE consumers
- Killing processes on stop request
- Multiple parallel processes supported (capped at 4 concurrent to prevent resource exhaustion)
- On FastAPI startup: scan experiments/ for incomplete runs and mark them as failed (crash recovery)

## Storage

Filesystem + JSON only. No database.

- Each run gets `experiments/{run_id}/` with a millisecond-timestamp ID (e.g., `1719000000123`)
- `config.json`: Full config used for the run (all values explicit, no inheritance — resolved at launch time)
- `metrics.json`: Structure: `{"epochs": [{epoch metrics}...], "summary": {final metrics}}` — appended per-epoch during training
- Plots and checkpoints as files in the run directory
- Listing runs = scanning experiment directories and reading their config/metrics JSON

## Visual Design

- **Light mode only** — white backgrounds, zinc/gray text hierarchy
- **No rounded borders** — sharp corners on all elements (border-radius: 0)
- **shadcn/ui** — Use shadcn components (Button, Card, Input, Slider, Badge, Table, Collapsible, Tabs)
- **Color palette:** `#09090b` (text), `#71717a` (muted), `#f4f4f5` (muted fill), `#e5e5e5` (borders), `#fafafa` (sidebar bg)
- **Status badges:** Amber bg for running, green for complete, red for failed
- **Progress bars:** Black fill on light gray track, no rounded ends

## Configurable Parameters (Complete List)

### Model Architecture
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| N | int | 4096 | Signal length |
| d | int | 1 | Filter order |
| fc | float | 0.006 | Filter cutoff frequency |
| num_layers | int | 5 | Number of unrolled BEADS layers |
| solve_cg_iters | int | 5 | CG iterations for solve step |
| lowpass_cg_iters | int | 24 | CG iterations for lowpass filter |
| shared_params | bool | false | Share parameters across layers |

### Training
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| learning_rate | float | 1e-3 | Adam optimizer LR |
| batch_size | int | 4 | Training batch size |
| num_samples | int | 500 | Synthetic training samples |
| train_ratio | float | 0.8 | Train/test split |
| noise_level | float | 0.01 | Gaussian noise std |
| seed | int | 42 | Random seed |

### Loss Weights
| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| alpha_mse | float | 1.0 | 0–10 | Peak reconstruction (MSE) |
| alpha_l1 | float | 0.01 | 0–10 | L1 sparsity |
| alpha_tv | float | 0.01 | 0–10 | Total variation |
| alpha_smooth | float | 0.2 | 0–10 | Baseline curvature penalty |
| alpha_neg | float | 2.0 | 0–10 | Non-negativity penalty |
| alpha_baseline | float | 0.5 | 0–10 | Baseline reconstruction |
| alpha_leakage | float | 0.5 | 0–10 | Baseline high-freq penalty |
| alpha_ortho | float | 0.2 | 0–10 | Peak-baseline orthogonality |
| alpha_baseline_tv | float | 0.0 | 0–10 | Baseline TV (3rd deriv) |
| peak_mask_rel_threshold | float | 0.02 | 0–1 | Peak mask relative threshold |
| peak_mask_abs_min | float | 1e-4 | 0–0.1 | Peak mask absolute floor |
| use_huber | bool | false | — | Use Huber loss instead of MSE |
| huber_delta | float | 0.1 | 0–1 | Huber loss delta |

### Training Stages
Each stage overrides a subset of the loss weights above, plus:
| Field | Type | Description |
|-------|------|-------------|
| name | string | Stage name (e.g., "MSE warmup") |
| epochs | int | Number of epochs for this stage |
| loss_config | object | Alpha overrides (unset = inherit global) |

## Tech Stack

- **Frontend:** Next.js 16, React 19, TypeScript, Tailwind CSS, shadcn/ui
- **Backend:** Python, FastAPI, uvicorn
- **Communication:** HTTP REST + SSE (Server-Sent Events)
- **Storage:** Filesystem + JSON
- **ML:** PyTorch (existing model code)
