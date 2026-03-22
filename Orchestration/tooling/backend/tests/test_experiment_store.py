import json
import os
import pytest
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
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
