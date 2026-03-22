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
