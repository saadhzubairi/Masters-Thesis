import asyncio
import json
import os
import time
from pathlib import Path
from typing import Optional

import math
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from experiment_store import ExperimentStore
from models.synth_generator import SyntheticDataGenerator
from process_manager import ProcessManager

app = FastAPI(title="LBEADS Experiment Hub")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def _sanitize(obj):
    """Replace NaN/Inf floats with None so JSON serialization doesn't blow up."""
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj


EXPERIMENTS_DIR = os.path.join(os.path.dirname(__file__), "experiments")
store = ExperimentStore(EXPERIMENTS_DIR)


def _on_process_event(run_id: str, event: dict):
    """Called from ProcessManager background thread whenever a JSON line is read."""
    event_type = event.get("type")
    if event_type == "epoch":
        store.append_epoch(run_id, event)
    elif event_type == "training_done":
        store.set_summary(run_id, event.get("final_metrics", {}))
    elif event_type == "complete":
        store.update_status(run_id, "complete")
    elif event_type == "error" and event.get("fatal"):
        store.update_status(run_id, "failed")
    elif event_type == "demo_error":
        store.append_error(run_id, {
            "source": event.get("demo", "unknown"),
            "message": event.get("error", "Unknown error"),
        })
    elif event_type == "_process_ended":
        # Process exited — ensure status is synced
        status = event.get("status", "failed")
        current = store.get_run(run_id)
        if current and current["status"] == "running":
            store.update_status(run_id, status)


manager = ProcessManager(max_concurrent=4, on_event=_on_process_event)


DATA_CONFIG_PATH = os.path.join(EXPERIMENTS_DIR, "data_config.json")

DEFAULT_DATA_CONFIG = {
    "baseline": {
        "smooth_sigma": 100.0,
        "sine_amp": 0.1,
        "sine_freq_min": 0.5,
        "sine_freq_max": 2.0,
        "baseline_amp_min": 0.08,
        "baseline_amp_max": 0.35,
    },
    "peak_layers": [
        {
            "num_peaks_min": 2,
            "num_peaks_max": 6,
            "amplitude_min": 0.2,
            "amplitude_max": 1.0,
            "rise_width_min": 10,
            "rise_width_max": 80,
            "decay_width_min": 20,
            "decay_width_max": 200,
            "plateau_width_min": 0,
            "plateau_width_max": 10,
            "peak_shape_mode": "mixed",
        }
    ],
    "noise": {"noise_level": 0.01},
}


class RunConfig(BaseModel):
    name: str
    model_type: str = "lbeads"
    device: str = "cpu"
    model: dict
    model_fast: Optional[dict] = None
    training: dict
    loss: dict
    stages: list
    data: Optional[dict] = None


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
async def list_runs(deleted: bool = False):
    runs = store.list_runs(include_deleted=deleted)
    for run in runs:
        live_status = manager.get_status(run["id"])
        if live_status:
            run["status"] = live_status
        # Include live epoch count from process manager output
        live_lines = manager.get_output_lines(run["id"])
        live_epochs = [l for l in live_lines if l.get("type") == "epoch"]
        if live_epochs:
            run["epoch_count"] = len(live_epochs)
            last = live_epochs[-1]
            if last.get("train_loss") is not None:
                run["summary"] = {
                    **run.get("summary", {}),
                    "train_loss": last["train_loss"],
                    "test_loss": last.get("test_loss"),
                }
    return _sanitize(runs)


@app.get("/runs/{run_id}")
async def get_run(run_id: str):
    details = store.get_run(run_id)
    if not details:
        raise HTTPException(status_code=404, detail="Run not found")
    live_status = manager.get_status(run_id)
    if live_status:
        details["status"] = live_status
    # Include stderr logs for failed runs
    stderr = manager.get_stderr(run_id)
    if stderr:
        details["logs"] = stderr
    return _sanitize(details)


@app.get("/runs/{run_id}/logs")
async def get_run_logs(run_id: str):
    if not store.get_run(run_id):
        raise HTTPException(status_code=404, detail="Run not found")
    return {"logs": manager.get_stderr(run_id)}


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
        raise HTTPException(
            status_code=400, detail=f"Run is not running (status: {status})"
        )
    await manager.stop(run_id)
    store.update_status(run_id, "stopped")
    return {"status": "stopped"}


@app.post("/runs/{run_id}/retry")
async def retry_run(run_id: str):
    original = store.get_run(run_id)
    if not original:
        raise HTTPException(status_code=404, detail="Run not found")

    config = original["config"]
    new_name = config.get("name", "Untitled") + " (retry)"
    new_run_id = store.create_run(new_name, {**config, "name": new_name})
    run_dir = os.path.join(EXPERIMENTS_DIR, new_run_id)
    config_path = os.path.join(run_dir, "config.json")

    runner_path = os.path.join(os.path.dirname(__file__), "train_runner.py")
    cmd = ["python3", runner_path, "--config", config_path, "--output-dir", run_dir]

    try:
        await manager.start(new_run_id, cmd, cwd=os.path.dirname(__file__))
        store.update_status(new_run_id, "running")
    except RuntimeError as e:
        store.update_status(new_run_id, "failed")
        raise HTTPException(status_code=503, detail=str(e))

    return {"run_id": new_run_id}


@app.post("/runs/{run_id}/demos")
async def run_demos(run_id: str):
    details = store.get_run(run_id)
    if not details:
        raise HTTPException(status_code=404, detail="Run not found")

    run_dir = os.path.join(EXPERIMENTS_DIR, run_id)
    checkpoint_path = os.path.join(run_dir, "checkpoint.pth")
    if not os.path.exists(checkpoint_path):
        raise HTTPException(status_code=400, detail="No checkpoint found — training may not have completed")

    config = details["config"]

    # Clear old demo errors before re-running
    metrics_path = os.path.join(run_dir, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
        metrics.pop("errors", None)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f)

    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "models"))
    from demo import run_demo
    from demo_chromatogram import run_chromatogram_demo

    results = {"demo": None, "demo_chromatogram": None, "errors": []}

    # Determine N from the right config section
    model_type = config.get("model_type", "lbeads")
    if model_type in ("lbeads_fast", "lbeads_v5"):
        N = config.get("model_fast", config.get("model", {})).get("N", 4096)
    else:
        N = config.get("model", {}).get("N", 4096)

    try:
        demo_dir = os.path.join(run_dir, "demo")
        outputs = run_demo(checkpoint_path, demo_dir, N=N, data_config=config.get("data"))
        results["demo"] = outputs
    except Exception as e:
        results["errors"].append({"source": "demo.py", "message": str(e)})
        store.append_error(run_id, {"source": "demo.py", "message": str(e)})

    try:
        chrom_dir = os.path.join(run_dir, "demo_chrom")
        outputs = run_chromatogram_demo(checkpoint_path, chrom_dir, N=N, data_config=config.get("data"))
        results["demo_chromatogram"] = outputs
    except Exception as e:
        results["errors"].append({"source": "demo_chromatogram.py", "message": str(e)})
        store.append_error(run_id, {"source": "demo_chromatogram.py", "message": str(e)})

    # If training had completed but demos failed, update status to complete now
    if not results["errors"] and details.get("status") == "failed":
        store.update_status(run_id, "complete")

    return _sanitize(results)


@app.post("/runs/{run_id}/delete")
async def soft_delete_run(run_id: str):
    if manager.get_status(run_id) == "running":
        raise HTTPException(status_code=400, detail="Cannot delete a running run")
    if not store.soft_delete_run(run_id):
        raise HTTPException(status_code=404, detail="Run not found")
    return {"status": "deleted"}


@app.post("/runs/{run_id}/restore")
async def restore_run(run_id: str):
    if not store.restore_run(run_id):
        raise HTTPException(status_code=404, detail="Run not found or not deleted")
    return {"status": "restored"}


@app.delete("/runs/{run_id}")
async def permanently_delete_run(run_id: str):
    if manager.get_status(run_id) == "running":
        raise HTTPException(status_code=400, detail="Cannot delete a running run")
    if not store.permanently_delete_run(run_id):
        raise HTTPException(status_code=404, detail="Run not found")
    return {"status": "permanently_deleted"}


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


@app.get("/data-config")
async def get_data_config():
    if os.path.isfile(DATA_CONFIG_PATH):
        with open(DATA_CONFIG_PATH) as f:
            return json.load(f)
    return DEFAULT_DATA_CONFIG


@app.post("/data-config")
async def save_data_config(config: dict):
    os.makedirs(os.path.dirname(DATA_CONFIG_PATH), exist_ok=True)
    with open(DATA_CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)
    return {"status": "saved"}


class BaselineParams(BaseModel):
    smooth_sigma: float = 100.0
    sine_amp: float = 0.1
    sine_freq_min: float = 0.5
    sine_freq_max: float = 2.0
    baseline_amp_min: float = 0.08
    baseline_amp_max: float = 0.35


class PeaksParams(BaseModel):
    num_peaks_min: int = 2
    num_peaks_max: int = 6
    amplitude_min: float = 0.2
    amplitude_max: float = 1.0
    rise_width_min: int = 10
    rise_width_max: int = 80
    decay_width_min: int = 20
    decay_width_max: int = 200
    plateau_width_min: int = 0
    plateau_width_max: int = 10
    peak_shape_mode: str = "mixed"


class NoiseParams(BaseModel):
    noise_level: float = 0.01


class GeneratePreviewConfig(BaseModel):
    n_signals: int = 5
    N: int = 4096
    seed: Optional[int] = None
    baseline: BaselineParams = BaselineParams()
    peak_layers: list[PeaksParams] = [PeaksParams()]
    noise: NoiseParams = NoiseParams()


@app.post("/generate-preview")
async def generate_preview(config: GeneratePreviewConfig):
    gen = SyntheticDataGenerator(
        N=config.N,
        seed=config.seed,
    )

    signals = []
    for _ in range(config.n_signals):
        f_true, _ = gen.generate_baseline(
            smooth_sigma=config.baseline.smooth_sigma,
            sine_amp=config.baseline.sine_amp,
            sine_freq_range=(config.baseline.sine_freq_min, config.baseline.sine_freq_max),
            baseline_amp_range=(config.baseline.baseline_amp_min, config.baseline.baseline_amp_max),
        )
        x_true = np.zeros(config.N, dtype=np.float64)
        for layer in config.peak_layers:
            gen.peak_shape_mode = layer.peak_shape_mode
            peaks, _ = gen.generate_peaks(
                num_peaks_range=(layer.num_peaks_min, layer.num_peaks_max),
                amplitude_range=(layer.amplitude_min, layer.amplitude_max),
                rise_width_range=(layer.rise_width_min, layer.rise_width_max),
                decay_width_range=(layer.decay_width_min, layer.decay_width_max),
                plateau_width_range=(layer.plateau_width_min, layer.plateau_width_max),
            )
            x_true += peaks
        noise, _ = gen.generate_noise(noise_level=config.noise.noise_level)

        y = x_true + f_true + noise
        scale = max(float(np.max(np.abs(y))), 1e-8)
        y /= scale
        x_true /= scale
        f_true /= scale
        noise /= scale

        # Downsample to ~512 points for transfer
        step = max(1, config.N // 512)
        signals.append({
            "y": y[::step].tolist(),
            "x_true": x_true[::step].tolist(),
            "f_true": f_true[::step].tolist(),
            "noise": noise[::step].tolist(),
        })

    return {"signals": signals}


@app.on_event("startup")
async def startup():
    # Mark any incomplete runs from previous crashes as failed
    for run in store.list_runs(include_deleted=False):
        if run["status"] == "running" and not manager.get_status(run["id"]):
            store.update_status(run["id"], "failed")
