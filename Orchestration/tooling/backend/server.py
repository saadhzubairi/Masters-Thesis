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
    elif event_type == "_process_ended":
        # Process exited — ensure status is synced
        status = event.get("status", "failed")
        current = store.get_run(run_id)
        if current and current["status"] == "running":
            store.update_status(run_id, status)


manager = ProcessManager(max_concurrent=4, on_event=_on_process_event)


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
