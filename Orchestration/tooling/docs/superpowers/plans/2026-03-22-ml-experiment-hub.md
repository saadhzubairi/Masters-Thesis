# ML Experiment Hub Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local web app for orchestrating LBEADS-NET training jobs with real-time monitoring, configurable hyperparameters, and automatic demo execution.

**Architecture:** FastAPI backend (`:8000`) manages training processes and serves experiment data via REST + SSE. Next.js frontend (`:3000`) provides the UI. Training scripts are adapted from LBEADS_NETv6 to accept JSON configs and emit JSON-line progress to stdout.

**Tech Stack:** Next.js 16, React 19, TypeScript, Tailwind CSS v4, shadcn/ui, FastAPI, uvicorn, PyTorch

**Spec:** `docs/superpowers/specs/2026-03-22-ml-experiment-hub-design.md`

---

## File Structure

```
tooling/
├── app/
│   ├── layout.tsx                    # Root layout with sidebar
│   ├── page.tsx                      # Dashboard (list runs)
│   ├── globals.css                   # Tailwind + shadcn theme (light, no radius)
│   ├── runs/
│   │   ├── new/page.tsx              # New run config form
│   │   └── [id]/page.tsx             # Run detail / live monitoring
│   └── components/
│       ├── sidebar.tsx               # Sidebar navigation
│       ├── run-card.tsx              # Run summary card for dashboard
│       ├── config-form.tsx           # Training config form (all sections)
│       ├── stage-editor.tsx          # Training stages editor
│       ├── alpha-slider.tsx          # Slider + text input for alpha values
│       ├── epoch-table.tsx           # Epoch breakdown table with expand
│       ├── loss-chart.tsx            # Live loss curve chart
│       └── results-gallery.tsx       # Image gallery for plots
├── lib/
│   ├── api.ts                        # Backend API client (fetch wrappers)
│   ├── types.ts                      # Shared TypeScript types
│   └── utils.ts                      # shadcn cn() utility
├── components/
│   └── ui/                           # shadcn components (auto-generated)
├── backend/
│   ├── server.py                     # FastAPI app with all routes
│   ├── process_manager.py            # Spawn, track, kill training processes
│   ├── experiment_store.py           # Read/write experiment JSON + files
│   ├── train_runner.py               # CLI wrapper: JSON config → JSON line output
│   ├── models/
│   │   ├── lbeads_net.py             # Model architecture (copied from v6)
│   │   ├── train.py                  # Training logic (adapted)
│   │   ├── demo.py                   # Synthetic demo (adapted)
│   │   └── demo_chromatogram.py      # Chromatogram demo (adapted)
│   ├── tests/
│   │   ├── test_experiment_store.py
│   │   ├── test_process_manager.py
│   │   └── test_train_runner.py
│   └── requirements.txt
├── next.config.ts                    # Add rewrites to proxy /api → FastAPI
└── package.json
```

---

## Chunk 1: Backend Foundation

### Task 1: Set up Python backend project

**Files:**
- Create: `backend/requirements.txt`
- Create: `backend/server.py`
- Create: `backend/experiment_store.py`
- Create: `backend/tests/test_experiment_store.py`

- [ ] **Step 1: Create requirements.txt**

```
fastapi==0.115.0
uvicorn[standard]==0.30.0
pydantic==2.9.0
pytest==8.3.0
torch>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
matplotlib>=3.7.0
```

- [ ] **Step 2: Create virtual environment and install deps**

Run: `cd backend && python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt`

- [ ] **Step 3: Write test for ExperimentStore**

```python
# backend/tests/test_experiment_store.py
import json
import os
import pytest
from experiment_store import ExperimentStore

@pytest.fixture
def store(tmp_path):
    return ExperimentStore(str(tmp_path / "experiments"))

def test_create_run_returns_id(store):
    config = {"model": {"N": 4096}, "loss": {"alpha_mse": 1.0}}
    run_id = store.create_run("test run", config)
    assert isinstance(run_id, str)
    assert len(run_id) > 0

def test_create_run_writes_config(store):
    config = {"model": {"N": 4096}, "loss": {"alpha_mse": 1.0}}
    run_id = store.create_run("test run", config)
    run_dir = os.path.join(store.base_dir, run_id)
    assert os.path.exists(os.path.join(run_dir, "config.json"))
    with open(os.path.join(run_dir, "config.json")) as f:
        saved = json.load(f)
    assert saved["loss"]["alpha_mse"] == 1.0
    assert saved["name"] == "test run"

def test_list_runs_empty(store):
    assert store.list_runs() == []

def test_list_runs_returns_summaries(store):
    store.create_run("run A", {"loss": {}})
    store.create_run("run B", {"loss": {}})
    runs = store.list_runs()
    assert len(runs) == 2
    assert all("id" in r and "name" in r and "status" in r for r in runs)

def test_get_run_returns_full_details(store):
    config = {"model": {"N": 4096}, "loss": {"alpha_mse": 1.0}}
    run_id = store.create_run("test", config)
    details = store.get_run(run_id)
    assert details["id"] == run_id
    assert details["config"]["model"]["N"] == 4096
    assert details["status"] == "pending"

def test_update_status(store):
    run_id = store.create_run("test", {"loss": {}})
    store.update_status(run_id, "running")
    assert store.get_run(run_id)["status"] == "running"

def test_append_epoch(store):
    run_id = store.create_run("test", {"loss": {}})
    epoch_data = {"epoch": 1, "train_loss": 0.05, "test_loss": 0.06}
    store.append_epoch(run_id, epoch_data)
    details = store.get_run(run_id)
    assert len(details["metrics"]["epochs"]) == 1
    assert details["metrics"]["epochs"][0]["train_loss"] == 0.05

def test_set_summary(store):
    run_id = store.create_run("test", {"loss": {}})
    summary = {"test_mse": 0.003, "test_mae": 0.04}
    store.set_summary(run_id, summary)
    details = store.get_run(run_id)
    assert details["metrics"]["summary"]["test_mse"] == 0.003

def test_get_nonexistent_run_returns_none(store):
    assert store.get_run("fake_id") is None
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `cd backend && source venv/bin/activate && python -m pytest tests/test_experiment_store.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'experiment_store'`

- [ ] **Step 5: Implement ExperimentStore**

```python
# backend/experiment_store.py
import json
import os
import time
from typing import Optional

class ExperimentStore:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

    def create_run(self, name: str, config: dict) -> str:
        run_id = str(int(time.time() * 1000))
        run_dir = os.path.join(self.base_dir, run_id)
        os.makedirs(run_dir, exist_ok=True)
        config_with_meta = {**config, "name": name, "run_id": run_id}
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump(config_with_meta, f, indent=2)
        metrics = {"epochs": [], "summary": {}}
        with open(os.path.join(run_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        status = {"status": "pending", "created_at": time.time()}
        with open(os.path.join(run_dir, "status.json"), "w") as f:
            json.dump(status, f, indent=2)
        return run_id

    def list_runs(self) -> list:
        if not os.path.exists(self.base_dir):
            return []
        runs = []
        for entry in os.listdir(self.base_dir):
            run_dir = os.path.join(self.base_dir, entry)
            if not os.path.isdir(run_dir):
                continue
            config_path = os.path.join(run_dir, "config.json")
            status_path = os.path.join(run_dir, "status.json")
            metrics_path = os.path.join(run_dir, "metrics.json")
            if not os.path.exists(config_path):
                continue
            with open(config_path) as f:
                config = json.load(f)
            with open(status_path) as f:
                status_data = json.load(f)
            summary = {}
            epochs = []
            if os.path.exists(metrics_path):
                with open(metrics_path) as f:
                    metrics = json.load(f)
                    summary = metrics.get("summary", {})
                    epochs = metrics.get("epochs", [])
            runs.append({
                "id": entry,
                "name": config.get("name", ""),
                "status": status_data.get("status", "pending"),
                "created_at": status_data.get("created_at", 0),
                "epoch_count": len(epochs),
                "summary": summary,
            })
        runs.sort(key=lambda r: r["created_at"], reverse=True)
        return runs

    def get_run(self, run_id: str) -> Optional[dict]:
        run_dir = os.path.join(self.base_dir, run_id)
        if not os.path.isdir(run_dir):
            return None
        config_path = os.path.join(run_dir, "config.json")
        status_path = os.path.join(run_dir, "status.json")
        metrics_path = os.path.join(run_dir, "metrics.json")
        with open(config_path) as f:
            config = json.load(f)
        with open(status_path) as f:
            status_data = json.load(f)
        with open(metrics_path) as f:
            metrics = json.load(f)
        files = []
        for root, dirs, filenames in os.walk(run_dir):
            for fname in filenames:
                if fname.endswith(('.png', '.pth')):
                    rel = os.path.relpath(os.path.join(root, fname), run_dir)
                    files.append(rel)
        return {
            "id": run_id,
            "name": config.get("name", ""),
            "config": config,
            "status": status_data.get("status", "pending"),
            "created_at": status_data.get("created_at", 0),
            "metrics": metrics,
            "files": files,
        }

    def update_status(self, run_id: str, status: str):
        status_path = os.path.join(self.base_dir, run_id, "status.json")
        with open(status_path) as f:
            data = json.load(f)
        data["status"] = status
        data["updated_at"] = time.time()
        with open(status_path, "w") as f:
            json.dump(data, f, indent=2)

    def append_epoch(self, run_id: str, epoch_data: dict):
        metrics_path = os.path.join(self.base_dir, run_id, "metrics.json")
        with open(metrics_path) as f:
            metrics = json.load(f)
        metrics["epochs"].append(epoch_data)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

    def set_summary(self, run_id: str, summary: dict):
        metrics_path = os.path.join(self.base_dir, run_id, "metrics.json")
        with open(metrics_path) as f:
            metrics = json.load(f)
        metrics["summary"] = summary
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd backend && source venv/bin/activate && python -m pytest tests/test_experiment_store.py -v`
Expected: All 8 tests PASS

- [ ] **Step 7: Commit**

```bash
git add backend/requirements.txt backend/experiment_store.py backend/tests/test_experiment_store.py
git commit -m "feat: add ExperimentStore with filesystem-based experiment storage"
```

---

### Task 2: Implement ProcessManager

**Files:**
- Create: `backend/process_manager.py`
- Create: `backend/tests/test_process_manager.py`

- [ ] **Step 1: Write test for ProcessManager**

```python
# backend/tests/test_process_manager.py
import asyncio
import pytest
from process_manager import ProcessManager

@pytest.fixture
def manager(tmp_path):
    return ProcessManager(max_concurrent=2)

@pytest.mark.asyncio
async def test_start_process_returns_pid(manager):
    # Use a simple echo script as a stand-in
    pid = await manager.start("test_1", ["python3", "-c", "import time; print('{\"type\":\"complete\"}'); time.sleep(0.1)"])
    assert isinstance(pid, int)
    assert pid > 0

@pytest.mark.asyncio
async def test_get_status_running(manager):
    pid = await manager.start("test_1", ["python3", "-c", "import time; time.sleep(5)"])
    status = manager.get_status("test_1")
    assert status == "running"
    await manager.stop("test_1")

@pytest.mark.asyncio
async def test_get_status_unknown(manager):
    status = manager.get_status("nonexistent")
    assert status is None

@pytest.mark.asyncio
async def test_stop_kills_process(manager):
    pid = await manager.start("test_1", ["python3", "-c", "import time; time.sleep(60)"])
    await manager.stop("test_1")
    status = manager.get_status("test_1")
    assert status in ("stopped", "failed")

@pytest.mark.asyncio
async def test_max_concurrent_enforced(manager):
    await manager.start("a", ["python3", "-c", "import time; time.sleep(60)"])
    await manager.start("b", ["python3", "-c", "import time; time.sleep(60)"])
    with pytest.raises(RuntimeError, match="concurrent"):
        await manager.start("c", ["python3", "-c", "import time; time.sleep(60)"])
    await manager.stop("a")
    await manager.stop("b")

@pytest.mark.asyncio
async def test_read_output_lines(manager):
    script = "import sys, json; print(json.dumps({'type':'epoch','epoch':1})); sys.stdout.flush()"
    await manager.start("test_1", ["python3", "-c", script])
    await asyncio.sleep(0.5)
    lines = manager.get_output_lines("test_1")
    assert len(lines) >= 1
    assert lines[0]["type"] == "epoch"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && source venv/bin/activate && pip install pytest-asyncio && python -m pytest tests/test_process_manager.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement ProcessManager**

```python
# backend/process_manager.py
import asyncio
import json
import signal
import subprocess
from typing import Optional

class ProcessManager:
    def __init__(self, max_concurrent: int = 4):
        self.max_concurrent = max_concurrent
        self._processes: dict[str, dict] = {}

    async def start(self, run_id: str, cmd: list[str], cwd: str = None) -> int:
        running = sum(1 for p in self._processes.values() if p["status"] == "running")
        if running >= self.max_concurrent:
            raise RuntimeError(f"Max concurrent processes ({self.max_concurrent}) reached")

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=cwd,
                text=True,
                bufsize=1,
            )
        except OSError as e:
            raise RuntimeError(f"Failed to start process: {e}")
        self._processes[run_id] = {
            "process": proc,
            "pid": proc.pid,
            "status": "running",
            "output_lines": [],
        }
        asyncio.get_event_loop().run_in_executor(None, self._read_output, run_id)
        return proc.pid

    def _read_output(self, run_id: str):
        entry = self._processes.get(run_id)
        if not entry:
            return
        proc = entry["process"]
        try:
            for line in proc.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    parsed = json.loads(line)
                    entry["output_lines"].append(parsed)
                except json.JSONDecodeError:
                    entry["output_lines"].append({"type": "log", "message": line})
            proc.wait()
            if entry["status"] == "running":
                entry["status"] = "complete" if proc.returncode == 0 else "failed"
        except Exception:
            entry["status"] = "failed"

    def get_status(self, run_id: str) -> Optional[str]:
        entry = self._processes.get(run_id)
        if not entry:
            return None
        return entry["status"]

    async def stop(self, run_id: str):
        entry = self._processes.get(run_id)
        if not entry or entry["status"] != "running":
            return
        proc = entry["process"]
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        entry["status"] = "stopped"

    def get_output_lines(self, run_id: str, since: int = 0) -> list:
        entry = self._processes.get(run_id)
        if not entry:
            return []
        return entry["output_lines"][since:]

    def get_active_count(self) -> int:
        return sum(1 for p in self._processes.values() if p["status"] == "running")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && source venv/bin/activate && python -m pytest tests/test_process_manager.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add backend/process_manager.py backend/tests/test_process_manager.py
git commit -m "feat: add ProcessManager for parallel training process lifecycle"
```

---

### Task 3: Copy and adapt model files

**Files:**
- Create: `backend/models/lbeads_net.py` (copy from v6)
- Create: `backend/models/train.py` (adapted from v6)
- Create: `backend/models/demo.py` (adapted from v6)
- Create: `backend/models/demo_chromatogram.py` (adapted from v6)
- Create: `backend/models/__init__.py`

- [ ] **Step 1: Copy model architecture unchanged**

```bash
mkdir -p backend/models
cp "../../Implementations/6. LBEADS_NETv6 [Strong]/lbeads_net.py" backend/models/lbeads_net.py
touch backend/models/__init__.py
```

The model architecture file is used as-is — no modifications needed.

- [ ] **Step 2: Copy and adapt train.py**

Copy from v6 and make these specific changes:

1. Extract the hardcoded config from `main()` into a `DEFAULT_CONFIG` dict at module level
2. Add a new function `run_training(config: dict, output_dir: str, callback=None)`
3. Keep `SyntheticDataGenerator`, `SparsityLoss`, and `train_lbeads_net()` unchanged
4. Keep the original `main()` intact (for standalone use)

Add this function after the existing `train_lbeads_net()` function:

```python
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
    # We monkey-patch the verbose output to call our callback instead of printing.
    # train_lbeads_net uses verbose=True to print; we intercept by wrapping it.
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
        # Reuse existing plot_training_results if available, or create simple one
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
```

Key design decisions:
- Runs stages sequentially via separate `train_lbeads_net()` calls (matching v6's staged approach)
- Callback receives structured epoch events after each stage's training completes
- Stage loss configs inherit from global then override (matching spec's "unset = inherit" behavior)
- `get_learned_params()` is called per-epoch if available on the model
- Checkpoint format matches v6's format for compatibility with demo scripts

- [ ] **Step 3: Copy and adapt demo.py**

Copy from v6 and add:
```python
def run_demo(checkpoint_path: str, output_dir: str, N: int = 4096):
    """Load model from checkpoint_path, run demo, save plots to output_dir."""
    # Uses existing load_trained_model() logic but with explicit path
    # Saves to output_dir/raw/, output_dir/hybrid/, output_dir/hybrid-snr/
```

Keep the original `main()` intact.

- [ ] **Step 4: Copy and adapt demo_chromatogram.py**

Copy from v6 and add:
```python
def run_chromatogram_demo(checkpoint_path: str, output_dir: str, N: int = 4096):
    """Load model, run chromatogram comparison, save to output_dir."""
    # Needs access to BEADS data files — use a configurable data_dir
    # Falls back to relative path from original location
```

Keep the original `main()` intact.

- [ ] **Step 5: Commit**

```bash
git add backend/models/
git commit -m "feat: copy v6 model files and add configurable run_training/run_demo interfaces"
```

---

### Task 4: Implement train_runner.py CLI wrapper

**Files:**
- Create: `backend/train_runner.py`
- Create: `backend/tests/test_train_runner.py`

- [ ] **Step 1: Write test for train_runner output format**

```python
# backend/tests/test_train_runner.py
import json
import subprocess
import os
import pytest

RUNNER_PATH = os.path.join(os.path.dirname(__file__), "..", "train_runner.py")

def test_runner_emits_valid_json_lines(tmp_path):
    config = {
        "model": {"N": 256, "num_layers": 2, "d": 1, "fc": 0.006,
                   "solve_cg_iters": 2, "lowpass_cg_iters": 2, "shared_params": True},
        "training": {"learning_rate": 1e-3, "batch_size": 2, "num_samples": 10,
                      "noise_level": 0.01, "train_ratio": 0.8, "seed": 42},
        "loss": {"alpha_mse": 1.0, "alpha_l1": 0.0, "alpha_tv": 0.0,
                 "alpha_smooth": 0.0, "alpha_neg": 0.0, "alpha_baseline": 0.0,
                 "alpha_leakage": 0.0, "alpha_ortho": 0.0, "alpha_baseline_tv": 0.0},
        "stages": [{"name": "test", "epochs": 2, "loss_config": {"alpha_mse": 1.0}}],
        "skip_demos": True
    }
    config_path = str(tmp_path / "config.json")
    output_dir = str(tmp_path / "output")
    with open(config_path, "w") as f:
        json.dump(config, f)

    result = subprocess.run(
        ["python3", RUNNER_PATH, "--config", config_path, "--output-dir", output_dir],
        capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"

    lines = [json.loads(l) for l in result.stdout.strip().split("\n") if l.strip()]
    types = [l["type"] for l in lines]
    assert "started" in types
    assert "epoch" in types
    assert "complete" in types or "training_done" in types

def test_runner_epoch_has_required_fields(tmp_path):
    config = {
        "model": {"N": 256, "num_layers": 2, "d": 1, "fc": 0.006,
                   "solve_cg_iters": 2, "lowpass_cg_iters": 2, "shared_params": True},
        "training": {"learning_rate": 1e-3, "batch_size": 2, "num_samples": 10,
                      "noise_level": 0.01, "train_ratio": 0.8, "seed": 42},
        "loss": {"alpha_mse": 1.0},
        "stages": [{"name": "test", "epochs": 1, "loss_config": {"alpha_mse": 1.0}}],
        "skip_demos": True
    }
    config_path = str(tmp_path / "config.json")
    output_dir = str(tmp_path / "output")
    with open(config_path, "w") as f:
        json.dump(config, f)

    result = subprocess.run(
        ["python3", RUNNER_PATH, "--config", config_path, "--output-dir", output_dir],
        capture_output=True, text=True, timeout=120
    )
    lines = [json.loads(l) for l in result.stdout.strip().split("\n") if l.strip()]
    epochs = [l for l in lines if l["type"] == "epoch"]
    assert len(epochs) >= 1
    e = epochs[0]
    assert "epoch" in e
    assert "stage" in e
    assert "train_loss" in e
    assert "components" in e
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && source venv/bin/activate && python -m pytest tests/test_train_runner.py -v`
Expected: FAIL

- [ ] **Step 3: Implement train_runner.py**

```python
# backend/train_runner.py
"""CLI wrapper: accepts --config JSON, runs training pipeline, emits JSON lines to stdout."""
import argparse
import json
import os
import sys
import time

def emit(event: dict):
    print(json.dumps(event), flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config JSON file")
    parser.add_argument("--output-dir", required=True, help="Directory for outputs")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    # Validate required config sections exist
    for key in ("model", "training", "loss", "stages"):
        if key not in config:
            emit({"type": "error", "message": f"Missing config section: {key}", "fatal": True})
            sys.exit(1)
    if not isinstance(config["stages"], list) or len(config["stages"]) == 0:
        emit({"type": "error", "message": "stages must be a non-empty list", "fatal": True})
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # Import model training code
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "models"))
    from train import run_training

    total_epochs = sum(s["epochs"] for s in config.get("stages", [{"epochs": 25}]))
    emit({"type": "started", "run_id": os.path.basename(args.output_dir), "total_epochs": total_epochs})

    current_stage = [None]

    def on_event(event):
        if event.get("type") == "epoch":
            if event.get("stage") != current_stage[0]:
                if current_stage[0] is not None:
                    emit({"type": "stage_change", "from": current_stage[0], "to": event["stage"], "epoch": event["epoch"]})
                current_stage[0] = event["stage"]
        emit(event)

    try:
        result = run_training(config, args.output_dir, callback=on_event)
        emit({
            "type": "training_done",
            "checkpoint": "checkpoint.pth",
            "final_metrics": result.get("metrics", {})
        })
    except Exception as e:
        emit({"type": "error", "message": str(e), "fatal": True})
        sys.exit(1)

    # Run demos unless skipped
    if not config.get("skip_demos", False):
        checkpoint_path = os.path.join(args.output_dir, "checkpoint.pth")

        # demo.py
        emit({"type": "demo_started", "demo": "demo.py"})
        try:
            from demo import run_demo
            demo_dir = os.path.join(args.output_dir, "demo")
            outputs = run_demo(checkpoint_path, demo_dir, N=config["model"].get("N", 4096))
            emit({"type": "demo_done", "demo": "demo.py", "outputs": outputs})
        except Exception as e:
            emit({"type": "demo_error", "demo": "demo.py", "error": str(e)})

        # demo_chromatogram.py
        emit({"type": "demo_started", "demo": "demo_chromatogram.py"})
        try:
            from demo_chromatogram import run_chromatogram_demo
            chrom_dir = os.path.join(args.output_dir, "demo_chrom")
            outputs = run_chromatogram_demo(checkpoint_path, chrom_dir, N=config["model"].get("N", 4096))
            emit({"type": "demo_done", "demo": "demo_chromatogram.py", "outputs": outputs})
        except Exception as e:
            emit({"type": "demo_error", "demo": "demo_chromatogram.py", "error": str(e)})

    emit({"type": "complete"})

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && source venv/bin/activate && python -m pytest tests/test_train_runner.py -v --timeout=120`
Expected: Both tests PASS (note: may take ~30s due to actual training with tiny config)

- [ ] **Step 5: Commit**

```bash
git add backend/train_runner.py backend/tests/test_train_runner.py
git commit -m "feat: add train_runner CLI wrapper with JSON line output protocol"
```

---

### Task 5: Implement FastAPI server

**Files:**
- Create: `backend/server.py`

- [ ] **Step 1: Implement FastAPI app with all endpoints**

```python
# backend/server.py
import asyncio
import json
import os
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from experiment_store import ExperimentStore
from process_manager import ProcessManager

app = FastAPI(title="LBEADS Experiment Hub")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

EXPERIMENTS_DIR = os.path.join(os.path.dirname(__file__), "experiments")
store = ExperimentStore(EXPERIMENTS_DIR)
manager = ProcessManager(max_concurrent=4)

class RunConfig(BaseModel):
    name: str
    model: dict
    training: dict
    loss: dict
    stages: list

@app.post("/runs")
async def create_run(config: RunConfig):
    run_id = store.create_run(config.name, config.model_dump())
    run_dir = os.path.join(EXPERIMENTS_DIR, run_id)
    config_path = os.path.join(run_dir, "config.json")

    runner_path = os.path.join(os.path.dirname(__file__), "train_runner.py")
    cmd = ["python3", runner_path, "--config", config_path, "--output-dir", run_dir]

    try:
        await manager.start(run_id, cmd, cwd=os.path.dirname(__file__))
        store.update_status(run_id, "running")
    except RuntimeError as e:
        store.update_status(run_id, "failed")
        raise HTTPException(status_code=503, detail=str(e))

    return {"run_id": run_id}

@app.get("/runs")
async def list_runs():
    runs = store.list_runs()
    # Enrich with live status from process manager
    for run in runs:
        live_status = manager.get_status(run["id"])
        if live_status:
            run["status"] = live_status
    return runs

@app.get("/runs/{run_id}")
async def get_run(run_id: str):
    details = store.get_run(run_id)
    if not details:
        raise HTTPException(status_code=404, detail="Run not found")
    live_status = manager.get_status(run_id)
    if live_status:
        details["status"] = live_status
    return details

@app.get("/runs/{run_id}/stream")
async def stream_run(run_id: str):
    if not store.get_run(run_id):
        raise HTTPException(status_code=404, detail="Run not found")

    async def event_generator():
        last_index = 0
        while True:
            lines = manager.get_output_lines(run_id, since=last_index)
            for line in lines:
                yield {"data": json.dumps(line)}
                # Also persist epoch data to store
                if line.get("type") == "epoch":
                    store.append_epoch(run_id, line)
                elif line.get("type") == "training_done":
                    store.set_summary(run_id, line.get("final_metrics", {}))
                elif line.get("type") == "complete":
                    store.update_status(run_id, "complete")
                elif line.get("type") == "error" and line.get("fatal"):
                    store.update_status(run_id, "failed")
            last_index += len(lines)

            status = manager.get_status(run_id)
            if status and status != "running":
                break
            await asyncio.sleep(0.5)

    return EventSourceResponse(event_generator())

@app.post("/runs/{run_id}/stop")
async def stop_run(run_id: str):
    status = manager.get_status(run_id)
    if status != "running":
        raise HTTPException(status_code=400, detail=f"Run is not running (status: {status})")
    await manager.stop(run_id)
    store.update_status(run_id, "stopped")
    return {"status": "stopped"}

@app.get("/runs/{run_id}/files/{file_path:path}")
async def get_file(run_id: str, file_path: str):
    if not store.get_run(run_id):
        raise HTTPException(status_code=404, detail="Run not found")
    run_dir = os.path.realpath(os.path.join(EXPERIMENTS_DIR, run_id))
    full_path = os.path.realpath(os.path.join(run_dir, file_path))
    # Path traversal protection: resolved path must be under run_dir
    if not full_path.startswith(run_dir + os.sep) and full_path != run_dir:
        raise HTTPException(status_code=403, detail="Access denied")
    if not os.path.isfile(full_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(full_path)

@app.on_event("startup")
async def startup():
    # Mark any incomplete runs from previous crashes as failed
    for run in store.list_runs():
        if run["status"] == "running" and not manager.get_status(run["id"]):
            store.update_status(run["id"], "failed")
```

- [ ] **Step 2: Add sse-starlette to requirements.txt**

Append `sse-starlette==2.1.0` to `backend/requirements.txt` and install:
```bash
cd backend && source venv/bin/activate && pip install sse-starlette==2.1.0
```

- [ ] **Step 3: Test server starts without errors**

Run: `cd backend && source venv/bin/activate && timeout 5 python -m uvicorn server:app --port 8000 || true`
Expected: Server starts, then timeout kills it. No import errors.

- [ ] **Step 4: Commit**

```bash
git add backend/server.py backend/requirements.txt
git commit -m "feat: add FastAPI server with REST + SSE endpoints for experiment management"
```

---

## Chunk 2: Frontend Foundation

### Task 6: Set up shadcn/ui and theme

**Files:**
- Modify: `app/globals.css`
- Modify: `app/layout.tsx`
- Create: `lib/utils.ts`
- Create: `components.json` (via shadcn init)

- [ ] **Step 1: Initialize shadcn/ui**

Run:
```bash
cd /Users/saadhzubairi/Work/Masters-Thesis/Orchestration/tooling
npx shadcn@latest init
```

Select: TypeScript, default style, no dark mode, base color zinc.

**Important:** Check `node_modules/next/dist/docs/` for any Next.js 16 specific guidance on component libraries before proceeding.

- [ ] **Step 2: Install shadcn components we need**

```bash
npx shadcn@latest add button card input slider badge table collapsible tabs select label separator progress
```

- [ ] **Step 3: Override globals.css for light-only, no-radius theme**

Replace `app/globals.css` with:

```css
@import "tailwindcss";

@layer base {
  :root {
    --background: 0 0% 100%;
    --foreground: 240 10% 3.9%;
    --card: 0 0% 100%;
    --card-foreground: 240 10% 3.9%;
    --popover: 0 0% 100%;
    --popover-foreground: 240 10% 3.9%;
    --primary: 240 5.9% 10%;
    --primary-foreground: 0 0% 98%;
    --secondary: 240 4.8% 95.9%;
    --secondary-foreground: 240 5.9% 10%;
    --muted: 240 4.8% 95.9%;
    --muted-foreground: 240 3.8% 46.1%;
    --accent: 240 4.8% 95.9%;
    --accent-foreground: 240 5.9% 10%;
    --destructive: 0 84.2% 60.2%;
    --destructive-foreground: 0 0% 98%;
    --border: 240 5.9% 90%;
    --input: 240 5.9% 90%;
    --ring: 240 5.9% 10%;
    --radius: 0px;
  }
}

@layer base {
  * {
    @apply border-border;
  }
  body {
    @apply bg-background text-foreground;
    font-family: var(--font-geist-sans), system-ui, sans-serif;
  }
}
```

Key: `--radius: 0px` removes all rounded corners globally.

- [ ] **Step 4: Create lib/utils.ts if not already created by shadcn**

```typescript
// lib/utils.ts
import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}
```

- [ ] **Step 5: Verify the app builds**

Run: `npm run build`
Expected: Build succeeds

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat: initialize shadcn/ui with light-only, no-radius zinc theme"
```

---

### Task 7: Create shared types and API client

**Files:**
- Create: `lib/types.ts`
- Create: `lib/api.ts`
- Modify: `next.config.ts`

- [ ] **Step 1: Define TypeScript types**

```typescript
// lib/types.ts
export interface RunConfig {
  name: string
  model: ModelConfig
  training: TrainingConfig
  loss: LossConfig
  stages: StageConfig[]
}

export interface ModelConfig {
  N: number
  d: number
  fc: number
  num_layers: number
  solve_cg_iters: number
  lowpass_cg_iters: number
  shared_params: boolean
}

export interface TrainingConfig {
  learning_rate: number
  batch_size: number
  num_samples: number
  noise_level: number
  train_ratio: number
  seed: number
}

export interface LossConfig {
  alpha_mse: number
  alpha_l1: number
  alpha_tv: number
  alpha_smooth: number
  alpha_neg: number
  alpha_baseline: number
  alpha_leakage: number
  alpha_ortho: number
  alpha_baseline_tv: number
  peak_mask_rel_threshold: number
  peak_mask_abs_min: number
  use_huber: boolean
  huber_delta: number
}

export interface StageConfig {
  name: string
  epochs: number
  loss_config: Partial<LossConfig>
}

export interface RunSummary {
  id: string
  name: string
  status: "pending" | "running" | "complete" | "failed" | "stopped"
  created_at: number
  epoch_count: number
  summary: Record<string, number>
}

export interface RunDetail extends RunSummary {
  config: RunConfig
  metrics: {
    epochs: EpochData[]
    summary: Record<string, number>
  }
  files: string[]
}

export interface EpochData {
  epoch: number
  stage: string
  train_loss: number
  test_loss?: number
  components: Record<string, number>
  learned_params?: Record<string, number>
  elapsed_s?: number
}

export interface SSEEvent {
  type: "started" | "epoch" | "stage_change" | "training_done" | "demo_started" | "demo_done" | "demo_error" | "error" | "complete"
  [key: string]: unknown
}

export const DEFAULT_MODEL_CONFIG: ModelConfig = {
  N: 4096, d: 1, fc: 0.006, num_layers: 5,
  solve_cg_iters: 5, lowpass_cg_iters: 24, shared_params: false,
}

export const DEFAULT_TRAINING_CONFIG: TrainingConfig = {
  learning_rate: 1e-3, batch_size: 4, num_samples: 500,
  noise_level: 0.01, train_ratio: 0.8, seed: 42,
}

export const DEFAULT_LOSS_CONFIG: LossConfig = {
  alpha_mse: 1.0, alpha_l1: 0.01, alpha_tv: 0.01, alpha_smooth: 0.2,
  alpha_neg: 2.0, alpha_baseline: 0.5, alpha_leakage: 0.5, alpha_ortho: 0.2,
  alpha_baseline_tv: 0.0, peak_mask_rel_threshold: 0.02, peak_mask_abs_min: 1e-4,
  use_huber: false, huber_delta: 0.1,
}

export const DEFAULT_STAGES: StageConfig[] = [
  {
    name: "A_peak_recon",
    epochs: 5,
    loss_config: {
      alpha_mse: 1.0, alpha_l1: 0, alpha_tv: 0, alpha_smooth: 0,
      alpha_neg: 0, alpha_baseline: 0, alpha_leakage: 0, alpha_ortho: 0, alpha_baseline_tv: 0,
    },
  },
  {
    name: "B_full_loss",
    epochs: 20,
    loss_config: {},
  },
]
```

- [ ] **Step 2: Create API client**

```typescript
// lib/api.ts
import type { RunConfig, RunSummary, RunDetail, SSEEvent } from "./types"

const API_BASE = "/api"

export async function createRun(config: RunConfig): Promise<{ run_id: string }> {
  const res = await fetch(`${API_BASE}/runs`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(config),
  })
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export async function listRuns(): Promise<RunSummary[]> {
  const res = await fetch(`${API_BASE}/runs`)
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export async function getRun(id: string): Promise<RunDetail> {
  const res = await fetch(`${API_BASE}/runs/${id}`)
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export async function stopRun(id: string): Promise<void> {
  const res = await fetch(`${API_BASE}/runs/${id}/stop`, { method: "POST" })
  if (!res.ok) throw new Error(await res.text())
}

export function streamRun(id: string, onEvent: (event: SSEEvent) => void): () => void {
  const eventSource = new EventSource(`${API_BASE}/runs/${id}/stream`)
  eventSource.onmessage = (e) => {
    const event: SSEEvent = JSON.parse(e.data)
    onEvent(event)
  }
  eventSource.onerror = () => {
    eventSource.close()
  }
  return () => eventSource.close()
}

export function getFileUrl(runId: string, filePath: string): string {
  return `${API_BASE}/runs/${runId}/files/${filePath}`
}
```

- [ ] **Step 3: Add API proxy rewrite in next.config.ts**

Check `node_modules/next/dist/docs/` for the correct Next.js 16 rewrites syntax, then update:

```typescript
// next.config.ts
import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: "http://localhost:8000/:path*",
      },
    ]
  },
};

export default nextConfig;
```

- [ ] **Step 4: Verify build**

Run: `npm run build`
Expected: Build succeeds

- [ ] **Step 5: Commit**

```bash
git add lib/types.ts lib/api.ts next.config.ts
git commit -m "feat: add TypeScript types, API client, and proxy rewrite to FastAPI"
```

---

### Task 8: Create sidebar layout

**Files:**
- Create: `app/components/sidebar.tsx`
- Modify: `app/layout.tsx`
- Modify: `app/page.tsx`

- [ ] **Step 1: Create Sidebar component**

```tsx
// app/components/sidebar.tsx
"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import { cn } from "@/lib/utils"

const navItems = [
  { href: "/", label: "Dashboard", icon: "LayoutDashboard" },
  { href: "/runs/new", label: "New Run", icon: "Play" },
  { href: "/experiments", label: "Experiments", icon: "History" },
]

export function Sidebar() {
  const pathname = usePathname()

  return (
    <aside className="w-56 border-r bg-[#fafafa] flex flex-col">
      <div className="p-4 border-b">
        <h1 className="text-sm font-bold tracking-tight">LBEADS Hub</h1>
      </div>
      <nav className="flex-1 p-2">
        {navItems.map((item) => {
          const active = item.href === "/" ? pathname === "/" : pathname.startsWith(item.href)
          return (
            <Link
              key={item.href}
              href={item.href}
              className={cn(
                "block px-3 py-2 text-sm",
                active
                  ? "bg-white border font-medium text-foreground"
                  : "text-muted-foreground hover:text-foreground"
              )}
            >
              {item.label}
            </Link>
          )
        })}
      </nav>
    </aside>
  )
}
```

- [ ] **Step 2: Update layout.tsx**

```tsx
// app/layout.tsx
import type { Metadata } from "next"
import { Geist, Geist_Mono } from "next/font/google"
import { Sidebar } from "./components/sidebar"
import "./globals.css"

const geistSans = Geist({ variable: "--font-geist-sans", subsets: ["latin"] })
const geistMono = Geist_Mono({ variable: "--font-geist-mono", subsets: ["latin"] })

export const metadata: Metadata = {
  title: "LBEADS Hub",
  description: "ML Experiment Orchestration",
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${geistSans.variable} ${geistMono.variable}`}>
      <body className="h-screen flex">
        <Sidebar />
        <main className="flex-1 overflow-y-auto">{children}</main>
      </body>
    </html>
  )
}
```

- [ ] **Step 3: Replace page.tsx with dashboard placeholder**

```tsx
// app/page.tsx
export default function Dashboard() {
  return (
    <div className="p-6">
      <h2 className="text-lg font-semibold">Dashboard</h2>
      <p className="text-sm text-muted-foreground mt-1">Your experiments will appear here.</p>
    </div>
  )
}
```

- [ ] **Step 4: Verify app renders**

Run: `npm run dev` and check http://localhost:3000 — should show sidebar + empty dashboard.

- [ ] **Step 5: Commit**

```bash
git add app/components/sidebar.tsx app/layout.tsx app/page.tsx
git commit -m "feat: add sidebar layout with navigation"
```

---

## Chunk 3: Frontend Pages

### Task 9: Build Dashboard page with run cards

**Files:**
- Create: `app/components/run-card.tsx`
- Modify: `app/page.tsx`

- [ ] **Step 1: Create RunCard component**

```tsx
// app/components/run-card.tsx
"use client"

import Link from "next/link"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import type { RunSummary } from "@/lib/types"

const statusVariant: Record<string, string> = {
  running: "bg-amber-100 text-amber-800 border-amber-200",
  complete: "bg-green-100 text-green-800 border-green-200",
  failed: "bg-red-100 text-red-800 border-red-200",
  stopped: "bg-zinc-100 text-zinc-800 border-zinc-200",
  pending: "bg-zinc-100 text-zinc-600 border-zinc-200",
}

interface RunCardProps {
  run: RunSummary
  totalEpochs?: number
}

export function RunCard({ run, totalEpochs }: RunCardProps) {
  const progress = totalEpochs ? (run.epoch_count / totalEpochs) * 100 : 0

  return (
    <Link href={`/runs/${run.id}`} className="block border p-4 hover:bg-zinc-50 transition-colors">
      <div className="flex items-center justify-between mb-2">
        <div>
          <span className="font-semibold text-sm">Run #{run.id.slice(-6)}</span>
          <span className="text-muted-foreground text-sm ml-2">{run.name}</span>
        </div>
        <Badge variant="outline" className={statusVariant[run.status] || ""}>
          {run.status.toUpperCase()}
        </Badge>
      </div>
      {run.status === "running" && (
        <Progress value={progress} className="h-1.5 mb-2" />
      )}
      <div className="flex gap-6 text-xs text-muted-foreground">
        <span>Epochs <span className="text-foreground font-medium">{run.epoch_count}</span></span>
        {run.summary?.test_mse != null && (
          <span>Test MSE <span className="text-foreground font-medium">{run.summary.test_mse.toFixed(5)}</span></span>
        )}
        {run.summary?.test_mae != null && (
          <span>MAE <span className="text-foreground font-medium">{run.summary.test_mae.toFixed(4)}</span></span>
        )}
        <span className="ml-auto">
          {new Date(run.created_at * 1000).toLocaleString()}
        </span>
      </div>
    </Link>
  )
}
```

- [ ] **Step 2: Build Dashboard page**

```tsx
// app/page.tsx
"use client"

import { useEffect, useState } from "react"
import Link from "next/link"
import { Button } from "@/components/ui/button"
import { RunCard } from "./components/run-card"
import { listRuns } from "@/lib/api"
import type { RunSummary } from "@/lib/types"

export default function Dashboard() {
  const [runs, setRuns] = useState<RunSummary[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    loadRuns()
    const interval = setInterval(loadRuns, 3000) // Poll for status updates
    return () => clearInterval(interval)
  }, [])

  async function loadRuns() {
    try {
      const data = await listRuns()
      setRuns(data)
    } catch {
      // Backend not running yet — that's fine
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="p-6">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-lg font-semibold">Dashboard</h2>
          <p className="text-sm text-muted-foreground">
            {runs.length} experiment{runs.length !== 1 ? "s" : ""}
          </p>
        </div>
        <Button asChild>
          <Link href="/runs/new">+ New Run</Link>
        </Button>
      </div>
      {loading ? (
        <p className="text-sm text-muted-foreground">Loading...</p>
      ) : runs.length === 0 ? (
        <p className="text-sm text-muted-foreground">No experiments yet. Start a new run!</p>
      ) : (
        <div className="space-y-2">
          {runs.map((run) => (
            <RunCard key={run.id} run={run} />
          ))}
        </div>
      )}
    </div>
  )
}
```

- [ ] **Step 3: Commit**

```bash
git add app/components/run-card.tsx app/page.tsx
git commit -m "feat: add dashboard page with run cards and auto-refresh"
```

---

### Task 10: Build New Run config form

**Files:**
- Create: `app/components/alpha-slider.tsx`
- Create: `app/components/stage-editor.tsx`
- Create: `app/components/config-form.tsx`
- Create: `app/runs/new/page.tsx`

- [ ] **Step 1: Create AlphaSlider component**

```tsx
// app/components/alpha-slider.tsx
"use client"

import { Slider } from "@/components/ui/slider"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"

interface AlphaSliderProps {
  name: string
  value: number
  onChange: (value: number) => void
  min?: number
  max?: number
  step?: number
  description?: string
}

export function AlphaSlider({ name, value, onChange, min = 0, max = 10, step = 0.01, description }: AlphaSliderProps) {
  return (
    <div className="flex items-center gap-3">
      <Label className="w-40 text-xs font-mono shrink-0">{name}</Label>
      <Slider
        value={[value]}
        onValueChange={([v]) => onChange(v)}
        min={min}
        max={max}
        step={step}
        className="flex-1"
      />
      <Input
        type="number"
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value) || 0)}
        className="w-20 text-xs font-mono h-8"
        min={min}
        max={max}
        step={step}
      />
    </div>
  )
}
```

- [ ] **Step 2: Create StageEditor component**

```tsx
// app/components/stage-editor.tsx
"use client"

import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { AlphaSlider } from "./alpha-slider"
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible"
import type { StageConfig, LossConfig } from "@/lib/types"
import { DEFAULT_LOSS_CONFIG } from "@/lib/types"

interface StageEditorProps {
  stages: StageConfig[]
  onChange: (stages: StageConfig[]) => void
}

const ALPHA_KEYS = [
  "alpha_mse", "alpha_l1", "alpha_tv", "alpha_smooth", "alpha_neg",
  "alpha_baseline", "alpha_leakage", "alpha_ortho", "alpha_baseline_tv",
] as const

export function StageEditor({ stages, onChange }: StageEditorProps) {
  function updateStage(index: number, updates: Partial<StageConfig>) {
    const next = stages.map((s, i) => (i === index ? { ...s, ...updates } : s))
    onChange(next)
  }

  function removeStage(index: number) {
    onChange(stages.filter((_, i) => i !== index))
  }

  function addStage() {
    onChange([...stages, { name: `Stage ${String.fromCharCode(65 + stages.length)}`, epochs: 10, loss_config: {} }])
  }

  function toggleAlpha(stageIndex: number, key: string, enabled: boolean) {
    const stage = stages[stageIndex]
    const nextConfig = { ...stage.loss_config }
    if (enabled) {
      nextConfig[key as keyof LossConfig] = DEFAULT_LOSS_CONFIG[key as keyof LossConfig] as never
    } else {
      delete nextConfig[key as keyof LossConfig]
    }
    updateStage(stageIndex, { loss_config: nextConfig })
  }

  return (
    <div className="space-y-3">
      {stages.map((stage, i) => (
        <Collapsible key={i} defaultOpen={i === stages.length - 1}>
          <div className="border p-3">
            <CollapsibleTrigger className="flex items-center justify-between w-full">
              <div className="flex items-center gap-2">
                <span className="bg-foreground text-background px-2 py-0.5 text-xs font-bold">
                  {String.fromCharCode(65 + i)}
                </span>
                <Input
                  value={stage.name}
                  onChange={(e) => updateStage(i, { name: e.target.value })}
                  className="h-7 text-xs w-40"
                  onClick={(e) => e.stopPropagation()}
                />
              </div>
              <div className="flex items-center gap-2">
                <Label className="text-xs text-muted-foreground">Epochs:</Label>
                <Input
                  type="number"
                  value={stage.epochs}
                  onChange={(e) => updateStage(i, { epochs: parseInt(e.target.value) || 1 })}
                  className="h-7 text-xs w-16"
                  onClick={(e) => e.stopPropagation()}
                />
                {stages.length > 1 && (
                  <Button variant="ghost" size="sm" className="h-7 text-xs text-destructive" onClick={(e) => { e.stopPropagation(); removeStage(i) }}>
                    Remove
                  </Button>
                )}
              </div>
            </CollapsibleTrigger>
            <CollapsibleContent className="mt-3 space-y-2">
              {ALPHA_KEYS.map((key) => {
                const active = key in stage.loss_config
                const value = active ? (stage.loss_config[key] as number) : 0
                return (
                  <div key={key} className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      checked={active}
                      onChange={(e) => toggleAlpha(i, key, e.target.checked)}
                      className="accent-foreground"
                    />
                    <div className={active ? "flex-1" : "flex-1 opacity-40"}>
                      <AlphaSlider
                        name={key}
                        value={value}
                        onChange={(v) => {
                          const nextConfig = { ...stage.loss_config, [key]: v }
                          updateStage(i, { loss_config: nextConfig })
                        }}
                      />
                    </div>
                  </div>
                )
              })}
            </CollapsibleContent>
          </div>
        </Collapsible>
      ))}
      <Button variant="outline" size="sm" onClick={addStage} className="text-xs">
        + Add Stage
      </Button>
    </div>
  )
}
```

- [ ] **Step 3: Create ConfigForm component**

```tsx
// app/components/config-form.tsx
"use client"

import { useState } from "react"
import { useRouter } from "next/navigation"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible"
import { Separator } from "@/components/ui/separator"
import { AlphaSlider } from "./alpha-slider"
import { StageEditor } from "./stage-editor"
import { createRun, listRuns } from "@/lib/api"
import type { RunConfig, RunSummary, ModelConfig, TrainingConfig, LossConfig, StageConfig } from "@/lib/types"
import {
  DEFAULT_MODEL_CONFIG,
  DEFAULT_TRAINING_CONFIG,
  DEFAULT_LOSS_CONFIG,
  DEFAULT_STAGES,
} from "@/lib/types"

export function ConfigForm() {
  const router = useRouter()
  const [name, setName] = useState("")
  const [model, setModel] = useState<ModelConfig>({ ...DEFAULT_MODEL_CONFIG })
  const [training, setTraining] = useState<TrainingConfig>({ ...DEFAULT_TRAINING_CONFIG })
  const [loss, setLoss] = useState<LossConfig>({ ...DEFAULT_LOSS_CONFIG })
  const [stages, setStages] = useState<StageConfig[]>(structuredClone(DEFAULT_STAGES))
  const [submitting, setSubmitting] = useState(false)
  const [previousRuns, setPreviousRuns] = useState<RunSummary[]>([])

  // Load previous runs for cloning
  useState(() => {
    listRuns().then(setPreviousRuns).catch(() => {})
  })

  async function handleClone(runId: string) {
    try {
      const { getRun } = await import("@/lib/api")
      const run = await getRun(runId)
      if (run.config) {
        setModel(run.config.model || DEFAULT_MODEL_CONFIG)
        setTraining(run.config.training || DEFAULT_TRAINING_CONFIG)
        setLoss(run.config.loss || DEFAULT_LOSS_CONFIG)
        setStages(run.config.stages || DEFAULT_STAGES)
        setName(`${run.name} (clone)`)
      }
    } catch { /* ignore */ }
  }

  async function handleSubmit() {
    setSubmitting(true)
    try {
      const config: RunConfig = { name: name || "Untitled Run", model, training, loss, stages }
      const { run_id } = await createRun(config)
      router.push(`/runs/${run_id}`)
    } catch (err) {
      alert(`Failed to start run: ${err}`)
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="space-y-6">
      {/* Clone Source */}
      <div>
        <Label className="text-xs font-semibold">Clone From</Label>
        <div className="flex gap-2 mt-1">
          <Select onValueChange={handleClone}>
            <SelectTrigger className="h-9 text-xs">
              <SelectValue placeholder="Select a previous run..." />
            </SelectTrigger>
            <SelectContent>
              {previousRuns.map((r) => (
                <SelectItem key={r.id} value={r.id} className="text-xs">
                  Run #{r.id.slice(-6)} — {r.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>

      {/* Run Name */}
      <div>
        <Label className="text-xs font-semibold">Run Name</Label>
        <Input value={name} onChange={(e) => setName(e.target.value)} placeholder="e.g. high ortho experiment" className="mt-1 h-9 text-sm" />
      </div>

      <Separator />

      {/* Model Architecture */}
      <Collapsible defaultOpen>
        <CollapsibleTrigger className="flex items-center justify-between w-full">
          <span className="text-sm font-semibold">Model Architecture</span>
          <span className="text-xs text-muted-foreground">toggle</span>
        </CollapsibleTrigger>
        <CollapsibleContent className="mt-3 grid grid-cols-3 gap-3">
          {Object.entries(model).map(([key, val]) => (
            <div key={key}>
              <Label className="text-xs font-mono">{key}</Label>
              {typeof val === "boolean" ? (
                <div className="mt-1">
                  <input type="checkbox" checked={val} onChange={(e) => setModel({ ...model, [key]: e.target.checked })} className="accent-foreground" />
                </div>
              ) : (
                <Input
                  type="number"
                  value={val}
                  onChange={(e) => setModel({ ...model, [key]: parseFloat(e.target.value) || 0 })}
                  className="mt-1 h-8 text-xs font-mono"
                  step={key === "fc" ? 0.001 : 1}
                />
              )}
            </div>
          ))}
        </CollapsibleContent>
      </Collapsible>

      <Separator />

      {/* Training Parameters */}
      <Collapsible defaultOpen>
        <CollapsibleTrigger className="flex items-center justify-between w-full">
          <span className="text-sm font-semibold">Training Parameters</span>
          <span className="text-xs text-muted-foreground">toggle</span>
        </CollapsibleTrigger>
        <CollapsibleContent className="mt-3 grid grid-cols-3 gap-3">
          {Object.entries(training).map(([key, val]) => (
            <div key={key}>
              <Label className="text-xs font-mono">{key}</Label>
              <Input
                type="number"
                value={val}
                onChange={(e) => setTraining({ ...training, [key]: parseFloat(e.target.value) || 0 })}
                className="mt-1 h-8 text-xs font-mono"
                step={key === "learning_rate" ? 0.0001 : key === "noise_level" ? 0.001 : 1}
              />
            </div>
          ))}
        </CollapsibleContent>
      </Collapsible>

      <Separator />

      {/* Loss Weights */}
      <Collapsible defaultOpen>
        <CollapsibleTrigger className="flex items-center justify-between w-full">
          <span className="text-sm font-semibold">Loss Weights</span>
          <span className="text-xs text-muted-foreground">toggle</span>
        </CollapsibleTrigger>
        <CollapsibleContent className="mt-3 space-y-2">
          {Object.entries(loss).map(([key, val]) => {
            if (typeof val === "boolean") {
              return (
                <div key={key} className="flex items-center gap-3">
                  <Label className="w-40 text-xs font-mono">{key}</Label>
                  <input type="checkbox" checked={val} onChange={(e) => setLoss({ ...loss, [key]: e.target.checked })} className="accent-foreground" />
                </div>
              )
            }
            return (
              <AlphaSlider
                key={key}
                name={key}
                value={val as number}
                onChange={(v) => setLoss({ ...loss, [key]: v })}
                max={key.includes("threshold") ? 1 : key.includes("abs_min") ? 0.1 : key === "huber_delta" ? 1 : 10}
              />
            )
          })}
        </CollapsibleContent>
      </Collapsible>

      <Separator />

      {/* Training Stages */}
      <div>
        <span className="text-sm font-semibold">Training Stages</span>
        <div className="mt-3">
          <StageEditor stages={stages} onChange={setStages} />
        </div>
      </div>

      <Separator />

      {/* Submit */}
      <div className="flex justify-end">
        <Button onClick={handleSubmit} disabled={submitting} className="px-8">
          {submitting ? "Starting..." : "Start Training"}
        </Button>
      </div>
    </div>
  )
}
```

- [ ] **Step 4: Create the New Run page**

```tsx
// app/runs/new/page.tsx
import { ConfigForm } from "@/app/components/config-form"

export default function NewRunPage() {
  return (
    <div className="p-6 max-w-3xl">
      <h2 className="text-lg font-semibold mb-1">New Training Run</h2>
      <p className="text-sm text-muted-foreground mb-6">Configure hyperparameters and start training.</p>
      <ConfigForm />
    </div>
  )
}
```

- [ ] **Step 5: Verify the form renders**

Run: `npm run dev` and navigate to http://localhost:3000/runs/new — form should display with all sections, sliders, and stage editor.

- [ ] **Step 6: Commit**

```bash
git add app/components/alpha-slider.tsx app/components/stage-editor.tsx app/components/config-form.tsx app/runs/new/page.tsx
git commit -m "feat: add new run config form with alpha sliders, stages, and clone support"
```

---

### Task 11: Build Run Detail page with live monitoring

**Files:**
- Create: `app/components/epoch-table.tsx`
- Create: `app/components/loss-chart.tsx`
- Create: `app/components/results-gallery.tsx`
- Create: `app/runs/[id]/page.tsx`

- [ ] **Step 1: Create EpochTable component**

```tsx
// app/components/epoch-table.tsx
"use client"

import { useState } from "react"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import type { EpochData } from "@/lib/types"

interface EpochTableProps {
  epochs: EpochData[]
}

const COMPONENT_KEYS = [
  "reconstruction", "l1_sparsity", "total_variation", "baseline_smooth",
  "baseline_recon", "baseline_leakage", "peak_baseline_ortho", "non_negativity", "baseline_tv",
]

const PARAM_KEYS = ["lam0", "lam1", "lam2", "r", "step", "output_gain"]

export function EpochTable({ epochs }: EpochTableProps) {
  const [expanded, setExpanded] = useState<number | null>(null)
  const sorted = [...epochs].reverse()

  return (
    <div className="border">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead className="text-xs w-16">Epoch</TableHead>
            <TableHead className="text-xs w-16">Stage</TableHead>
            <TableHead className="text-xs">Train Loss</TableHead>
            <TableHead className="text-xs">Test Loss</TableHead>
            <TableHead className="text-xs">Recon</TableHead>
            <TableHead className="text-xs">Baseline</TableHead>
            <TableHead className="text-xs">Leakage</TableHead>
            <TableHead className="text-xs">Ortho</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {sorted.map((ep) => (
            <>
              <TableRow
                key={ep.epoch}
                className={`cursor-pointer hover:bg-zinc-50 ${ep.epoch === sorted[0]?.epoch ? "bg-green-50" : ""}`}
                onClick={() => setExpanded(expanded === ep.epoch ? null : ep.epoch)}
              >
                <TableCell className="text-xs font-medium">{ep.epoch}</TableCell>
                <TableCell className="text-xs">{ep.stage}</TableCell>
                <TableCell className="text-xs font-mono">{ep.train_loss?.toFixed(6)}</TableCell>
                <TableCell className="text-xs font-mono">{ep.test_loss?.toFixed(6) ?? "—"}</TableCell>
                <TableCell className="text-xs font-mono">{ep.components?.reconstruction?.toFixed(6) ?? "—"}</TableCell>
                <TableCell className="text-xs font-mono">{ep.components?.baseline_recon?.toFixed(6) ?? "—"}</TableCell>
                <TableCell className="text-xs font-mono">{ep.components?.baseline_leakage?.toFixed(6) ?? "—"}</TableCell>
                <TableCell className="text-xs font-mono">{ep.components?.peak_baseline_ortho?.toFixed(6) ?? "—"}</TableCell>
              </TableRow>
              {expanded === ep.epoch && (
                <TableRow key={`${ep.epoch}-detail`}>
                  <TableCell colSpan={8} className="bg-zinc-50 p-4">
                    <div className="grid grid-cols-3 gap-2 text-xs mb-3">
                      <div className="font-semibold col-span-3 text-muted-foreground">All Loss Components</div>
                      {COMPONENT_KEYS.map((k) => (
                        <div key={k}>
                          <span className="text-muted-foreground">{k}:</span>{" "}
                          <span className="font-mono">{ep.components?.[k]?.toFixed(6) ?? "—"}</span>
                        </div>
                      ))}
                    </div>
                    {ep.learned_params && (
                      <div className="grid grid-cols-3 gap-2 text-xs border-t pt-3">
                        <div className="font-semibold col-span-3 text-muted-foreground">Learned Parameters</div>
                        {PARAM_KEYS.map((k) => (
                          <div key={k}>
                            <span className="text-muted-foreground">{k}:</span>{" "}
                            <span className="font-mono">{ep.learned_params?.[k]?.toFixed(4) ?? "—"}</span>
                          </div>
                        ))}
                      </div>
                    )}
                  </TableCell>
                </TableRow>
              )}
            </>
          ))}
        </TableBody>
      </Table>
    </div>
  )
}
```

- [ ] **Step 2: Create LossChart component**

A simple canvas-based line chart (no heavy chart library needed for MVP):

```tsx
// app/components/loss-chart.tsx
"use client"

import { useRef, useEffect } from "react"
import type { EpochData } from "@/lib/types"

interface LossChartProps {
  epochs: EpochData[]
}

export function LossChart({ epochs }: LossChartProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || epochs.length === 0) return
    const ctx = canvas.getContext("2d")
    if (!ctx) return

    const dpr = window.devicePixelRatio || 1
    const w = canvas.clientWidth
    const h = canvas.clientHeight
    canvas.width = w * dpr
    canvas.height = h * dpr
    ctx.scale(dpr, dpr)

    ctx.clearRect(0, 0, w, h)

    const trainLosses = epochs.map((e) => e.train_loss)
    const testLosses = epochs.map((e) => e.test_loss).filter((v): v is number => v != null)
    const allValues = [...trainLosses, ...testLosses]
    const maxVal = Math.max(...allValues) * 1.1
    const minVal = 0

    const padL = 50, padR = 10, padT = 10, padB = 30
    const plotW = w - padL - padR
    const plotH = h - padT - padB

    function x(i: number) { return padL + (i / Math.max(epochs.length - 1, 1)) * plotW }
    function y(v: number) { return padT + (1 - (v - minVal) / (maxVal - minVal)) * plotH }

    // Grid
    ctx.strokeStyle = "#e5e5e5"
    ctx.lineWidth = 1
    for (let i = 0; i <= 4; i++) {
      const yy = padT + (i / 4) * plotH
      ctx.beginPath(); ctx.moveTo(padL, yy); ctx.lineTo(w - padR, yy); ctx.stroke()
      ctx.fillStyle = "#71717a"; ctx.font = "10px monospace"
      ctx.fillText(((maxVal - minVal) * (1 - i / 4) + minVal).toFixed(4), 2, yy + 3)
    }

    // Train loss
    ctx.strokeStyle = "#18181b"
    ctx.lineWidth = 2
    ctx.beginPath()
    trainLosses.forEach((v, i) => { i === 0 ? ctx.moveTo(x(i), y(v)) : ctx.lineTo(x(i), y(v)) })
    ctx.stroke()

    // Test loss
    if (testLosses.length > 0) {
      ctx.strokeStyle = "#dc2626"
      ctx.lineWidth = 2
      ctx.beginPath()
      epochs.forEach((e, i) => {
        if (e.test_loss != null) {
          i === 0 || epochs[i - 1].test_loss == null ? ctx.moveTo(x(i), y(e.test_loss)) : ctx.lineTo(x(i), y(e.test_loss))
        }
      })
      ctx.stroke()
    }

    // Stage boundary markers
    ctx.strokeStyle = "#a1a1aa"
    ctx.lineWidth = 1
    ctx.setLineDash([4, 4])
    for (let i = 1; i < epochs.length; i++) {
      if (epochs[i].stage !== epochs[i - 1].stage) {
        ctx.beginPath(); ctx.moveTo(x(i), padT); ctx.lineTo(x(i), padT + plotH); ctx.stroke()
        ctx.fillStyle = "#71717a"; ctx.font = "9px monospace"
        ctx.fillText(epochs[i].stage, x(i) + 2, padT + 10)
      }
    }
    ctx.setLineDash([])

    // X axis labels
    ctx.fillStyle = "#71717a"; ctx.font = "10px monospace"
    const step = Math.max(1, Math.floor(epochs.length / 10))
    for (let i = 0; i < epochs.length; i += step) {
      ctx.fillText(String(epochs[i].epoch), x(i) - 4, h - 5)
    }
  }, [epochs])

  return (
    <div className="border p-4">
      <div className="flex items-center gap-4 mb-2">
        <span className="text-sm font-semibold">Loss Curves</span>
        <span className="text-xs"><span className="inline-block w-3 h-0.5 bg-foreground mr-1 align-middle" /> Train</span>
        <span className="text-xs"><span className="inline-block w-3 h-0.5 bg-red-600 mr-1 align-middle" /> Test</span>
      </div>
      <canvas ref={canvasRef} className="w-full h-48" />
    </div>
  )
}
```

- [ ] **Step 3: Create ResultsGallery component**

```tsx
// app/components/results-gallery.tsx
"use client"

import { useState } from "react"
import { getFileUrl } from "@/lib/api"

interface ResultsGalleryProps {
  runId: string
  files: string[]
}

export function ResultsGallery({ runId, files }: ResultsGalleryProps) {
  const [expanded, setExpanded] = useState<string | null>(null)
  const images = files.filter((f) => f.endsWith(".png"))

  if (images.length === 0) return null

  return (
    <div>
      <h3 className="text-sm font-semibold mb-3">Results</h3>
      <div className="grid grid-cols-3 gap-3">
        {images.map((file) => (
          <div key={file} className="border p-2 cursor-pointer hover:bg-zinc-50" onClick={() => setExpanded(file)}>
            <img src={getFileUrl(runId, file)} alt={file} className="w-full" />
            <p className="text-xs text-muted-foreground mt-1 truncate">{file}</p>
          </div>
        ))}
      </div>
      {expanded && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50" onClick={() => setExpanded(null)}>
          <div className="bg-white p-4 max-w-4xl max-h-[90vh] overflow-auto" onClick={(e) => e.stopPropagation()}>
            <img src={getFileUrl(runId, expanded)} alt={expanded} className="w-full" />
            <p className="text-xs text-muted-foreground mt-2">{expanded}</p>
          </div>
        </div>
      )}
    </div>
  )
}
```

- [ ] **Step 4: Create Run Detail page**

```tsx
// app/runs/[id]/page.tsx
"use client"

import { useEffect, useState, useCallback, use } from "react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { getRun, stopRun, streamRun } from "@/lib/api"
import { EpochTable } from "@/app/components/epoch-table"
import { LossChart } from "@/app/components/loss-chart"
import { ResultsGallery } from "@/app/components/results-gallery"
import type { RunDetail, EpochData, SSEEvent } from "@/lib/types"

const statusVariant: Record<string, string> = {
  running: "bg-amber-100 text-amber-800 border-amber-200",
  complete: "bg-green-100 text-green-800 border-green-200",
  failed: "bg-red-100 text-red-800 border-red-200",
  stopped: "bg-zinc-100 text-zinc-800 border-zinc-200",
  pending: "bg-zinc-100 text-zinc-600 border-zinc-200",
}

export default function RunDetailPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params)
  const [run, setRun] = useState<RunDetail | null>(null)
  const [liveEpochs, setLiveEpochs] = useState<EpochData[]>([])
  const [currentStage, setCurrentStage] = useState<string>("")
  const [totalEpochs, setTotalEpochs] = useState(0)
  const [elapsed, setElapsed] = useState(0)

  useEffect(() => {
    getRun(id).then((data) => {
      setRun(data)
      if (data.metrics?.epochs) {
        setLiveEpochs(data.metrics.epochs)
      }
      if (data.config?.stages) {
        setTotalEpochs(data.config.stages.reduce((s: number, st: { epochs: number }) => s + st.epochs, 0))
      }
    })
  }, [id])

  useEffect(() => {
    if (!run || run.status !== "running") return

    const cleanup = streamRun(id, (event: SSEEvent) => {
      if (event.type === "epoch") {
        const ep = event as unknown as EpochData
        setLiveEpochs((prev) => [...prev, ep])
        setCurrentStage(ep.stage || "")
        if (ep.elapsed_s) setElapsed(ep.elapsed_s)
      } else if (event.type === "started") {
        setTotalEpochs((event.total_epochs as number) || 0)
      } else if (event.type === "complete") {
        // Refresh full run data
        getRun(id).then(setRun)
      } else if (event.type === "error") {
        getRun(id).then(setRun)
      }
    })

    return cleanup
  }, [id, run?.status])

  const handleStop = useCallback(async () => {
    await stopRun(id)
    const data = await getRun(id)
    setRun(data)
  }, [id])

  if (!run) return <div className="p-6 text-sm text-muted-foreground">Loading...</div>

  const progress = totalEpochs > 0 ? (liveEpochs.length / totalEpochs) * 100 : 0
  const lastEpoch = liveEpochs[liveEpochs.length - 1]

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold">
            Run #{id.slice(-6)} — {run.name}
          </h2>
        </div>
        <div className="flex items-center gap-3">
          <Badge variant="outline" className={statusVariant[run.status] || ""}>
            {run.status.toUpperCase()}
          </Badge>
          {run.status === "running" && (
            <Button variant="destructive" size="sm" onClick={handleStop}>
              Stop
            </Button>
          )}
        </div>
      </div>

      {/* Stats bar */}
      <div className="flex gap-4">
        {[
          { label: "PROGRESS", value: `${liveEpochs.length} / ${totalEpochs}` },
          { label: "TRAIN LOSS", value: lastEpoch?.train_loss?.toFixed(6) || "—" },
          { label: "TEST LOSS", value: lastEpoch?.test_loss?.toFixed(6) || "—" },
          { label: "STAGE", value: currentStage || "—" },
          { label: "ELAPSED", value: elapsed > 0 ? `${Math.floor(elapsed / 60)}m ${Math.floor(elapsed % 60)}s` : "—" },
        ].map(({ label, value }) => (
          <div key={label} className="border px-4 py-2">
            <div className="text-[10px] text-muted-foreground">{label}</div>
            <div className="text-sm font-semibold font-mono">{value}</div>
          </div>
        ))}
      </div>

      {/* Progress bar */}
      {run.status === "running" && <Progress value={progress} className="h-2" />}

      {/* Loss chart */}
      {liveEpochs.length > 0 && <LossChart epochs={liveEpochs} />}

      {/* Epoch table */}
      {liveEpochs.length > 0 && (
        <div>
          <h3 className="text-sm font-semibold mb-2">Epoch Breakdown</h3>
          <EpochTable epochs={liveEpochs} />
        </div>
      )}

      {/* Results gallery */}
      {run.files && run.files.length > 0 && (
        <ResultsGallery runId={id} files={run.files} />
      )}
    </div>
  )
}
```

- [ ] **Step 5: Verify page renders**

Run: `npm run dev` and navigate to http://localhost:3000/runs/test — should show "Loading..." (since no backend yet, that's expected).

- [ ] **Step 6: Commit**

```bash
git add app/components/epoch-table.tsx app/components/loss-chart.tsx app/components/results-gallery.tsx app/runs/
git commit -m "feat: add run detail page with live monitoring, epoch table, loss chart, and results gallery"
```

---

## Chunk 4: Integration & Polish

### Task 12: Wire up Next.js proxy and test end-to-end

**Files:**
- No new files, integration testing

- [ ] **Step 1: Start both servers**

Terminal 1:
```bash
cd backend && source venv/bin/activate && uvicorn server:app --port 8000 --reload
```

Terminal 2:
```bash
npm run dev
```

- [ ] **Step 2: Test API proxy works**

Open http://localhost:3000 — dashboard should load (empty list, no errors in console).
Open http://localhost:3000/api/runs — should return `[]` from FastAPI.

- [ ] **Step 3: Create a test run through the UI**

1. Navigate to http://localhost:3000/runs/new
2. Set run name to "test"
3. Under Training Stages, set Stage A epochs to 2, Stage B epochs to 3 (small for testing)
4. Under Training Parameters, set num_samples to 20, N to 256 (small for speed)
5. Click "Start Training"
6. Should redirect to run detail page
7. Watch epochs appear in real time via SSE

- [ ] **Step 4: Verify results after completion**

1. Run should show COMPLETE status
2. Training plot should appear in results gallery
3. Demo images should appear (if demo scripts are working)
4. All epoch data should be in the table with expandable rows

- [ ] **Step 5: Test concurrent runs**

1. Start a second run from the UI
2. Both should show as RUNNING on dashboard
3. Both should stream independently

- [ ] **Step 6: Commit any fixes from integration testing**

```bash
git add -A
git commit -m "fix: integration adjustments from end-to-end testing"
```

---

### Task 13: Add startup script for convenience

**Files:**
- Modify: `package.json`

- [ ] **Step 1: Add dev:backend and dev:all scripts**

Add to `package.json` scripts:

```json
{
  "scripts": {
    "dev": "next dev",
    "dev:backend": "cd backend && source venv/bin/activate && uvicorn server:app --port 8000 --reload",
    "build": "next build",
    "start": "next start",
    "lint": "eslint"
  }
}
```

Note: `dev:backend` is a convenience reference — in practice you'll run both in separate terminals.

- [ ] **Step 2: Commit**

```bash
git add package.json
git commit -m "feat: add backend dev script for convenience"
```
