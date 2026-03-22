#!/usr/bin/env python3
"""CLI wrapper: accepts --config JSON, runs training pipeline, emits JSON lines to stdout."""
import argparse
import json
import os
import sys
import time


def emit(event: dict):
    """Write a JSON event to the real stdout (fd 3 or saved reference)."""
    _real_stdout.write(json.dumps(event) + "\n")
    _real_stdout.flush()


# Save real stdout before any redirection
_real_stdout = sys.stdout


def main():
    parser = argparse.ArgumentParser(description="Train runner with JSON line protocol")
    parser.add_argument("--config", required=True, help="Path to config JSON file")
    parser.add_argument("--output-dir", required=True, help="Directory for outputs")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    # Validate required config sections
    for key in ("model", "training", "loss", "stages"):
        if key not in config:
            emit({"type": "error", "message": f"Missing config section: {key}", "fatal": True})
            sys.exit(1)
    if not isinstance(config["stages"], list) or len(config["stages"]) == 0:
        emit({"type": "error", "message": "stages must be a non-empty list", "fatal": True})
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # Redirect stdout to stderr so train.py print() calls don't corrupt JSON protocol
    sys.stdout = sys.stderr

    # Import model training code
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"))
    from train import run_training

    total_epochs = sum(s["epochs"] for s in config["stages"])
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
