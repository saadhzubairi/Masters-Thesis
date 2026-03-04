import json
import os
import inspect
import sys
from datetime import datetime
from typing import Any, Dict, List

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from demo import load_trained_model
from lbeads_net import HybridConfig, hybrid_infer_1d
from train import SyntheticDataGenerator


def energy(x: np.ndarray) -> float:
    return float(np.sum(np.asarray(x, dtype=np.float64) ** 2))


def _to_python(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_python(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _median_param(params: Dict[str, Any], prefix: str) -> float:
    vals = []
    for k, v in params.items():
        if isinstance(k, str) and (k == prefix or k.startswith(prefix) or k.endswith(f"_{prefix}")):
            try:
                vals.append(float(v))
            except (TypeError, ValueError):
                pass
    if not vals:
        return float("nan")
    return float(np.median(np.asarray(vals, dtype=np.float64)))


def main() -> None:
    script_dir = SCRIPT_DIR
    project_dir = PROJECT_DIR
    data_dir = os.path.join(script_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    N = 4096
    num_samples = 6
    synthetic_noise_range = (0.01, 0.01)
    synthetic_amplitude_range = (300.0, 1800.0)

    model, checkpoint = load_trained_model(project_dir, N)
    if model is None:
        raise RuntimeError("No trained model found. Run train.py first.")

    model_params = model.get_learned_params()
    print("\n=== Learned Parameters (full) ===")
    print(model_params)
    print("\n=== Learned Parameters (requested summary) ===")
    print("lam0 (median):", _median_param(model_params, "lam0"))
    print("lam1 (median):", _median_param(model_params, "lam1"))
    print("lam2 (median):", _median_param(model_params, "lam2"))
    print("r (median):", _median_param(model_params, "r"))
    print("step (median):", _median_param(model_params, "step"))
    print("output_gain (median):", _median_param(model_params, "output_gain"))

    print("\n=== Training Regime ===")
    # Current train.py main() uses this value explicitly.
    print("Noise std range used during training:", synthetic_noise_range)
    peak_sig = inspect.signature(SyntheticDataGenerator.generate_peaks)
    print(
        "Peak width range in generator:",
        {
            "rise_width_range": peak_sig.parameters["rise_width_range"].default,
            "decay_width_range": peak_sig.parameters["decay_width_range"].default,
            "plateau_width_range": peak_sig.parameters["plateau_width_range"].default,
        },
    )
    stage_configs = checkpoint.get("stage_configs", [])
    total_epochs = int(sum(int(stage.get("epochs", 0)) for stage in stage_configs))
    print("Number of epochs:", total_epochs)
    print("Loss used (checkpoint loss_config):", checkpoint.get("loss_config", {}))

    generator = SyntheticDataGenerator(N=N, seed=123)
    hybrid_cfg = HybridConfig(
        noise_k=2.5,
        lowpass_iterations=1,
        short_refine_iterations=8,
        full_refine_iterations=24,
    )

    report: Dict[str, Any] = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "config": {
            "N": N,
            "num_samples": num_samples,
            "synthetic_noise_range": synthetic_noise_range,
            "synthetic_amplitude_range": synthetic_amplitude_range,
            "hybrid_config": _to_python(hybrid_cfg.__dict__),
        },
        "training_regime": {
            "noise_std_range": synthetic_noise_range,
            "peak_width_ranges": {
                "rise_width_range": peak_sig.parameters["rise_width_range"].default,
                "decay_width_range": peak_sig.parameters["decay_width_range"].default,
                "plateau_width_range": peak_sig.parameters["plateau_width_range"].default,
            },
            "epochs": total_epochs,
            "loss_config": checkpoint.get("loss_config", {}),
        },
        "learned_params_full": _to_python(model_params),
        "learned_params_summary": {
            "lam0_median": _median_param(model_params, "lam0"),
            "lam1_median": _median_param(model_params, "lam1"),
            "lam2_median": _median_param(model_params, "lam2"),
            "r_median": _median_param(model_params, "r"),
            "step_median": _median_param(model_params, "step"),
            "output_gain_median": _median_param(model_params, "output_gain"),
        },
        "samples": [],
    }

    print("\n=== Per-sample Diagnostics ===")
    for i in range(num_samples):
        noise_level = float(generator.rng.uniform(*synthetic_noise_range))
        signal = generator.generate_signal(noise_level=noise_level)
        amp = float(generator.rng.uniform(*synthetic_amplitude_range))
        signal.y = signal.y * amp
        signal.x_true = signal.x_true * amp
        signal.f_true = signal.f_true * amp
        signal.noise = signal.noise * amp

        hybrid_result = hybrid_infer_1d(model, signal.y, config=hybrid_cfg)

        print(f"\n--- Sample {i + 1} ---")
        print("selected_stage:", hybrid_result["selected_stage"])
        print("noise_sigma_normalized:", hybrid_result["noise_sigma_normalized"])
        print("scale:", hybrid_result["scale"])
        print("regularization:", hybrid_result["regularization"])
        print("quality_post:", hybrid_result["quality"]["post"])
        print("quality_short:", hybrid_result["quality"]["short_refine"])
        if "full_refine" in hybrid_result["quality"]:
            print("quality_full:", hybrid_result["quality"]["full_refine"])

        print("||x_true||^2:", energy(signal.x_true))
        print("||x_lbeads||^2:", energy(hybrid_result["x_lbeads"]))
        print("||x_post||^2:", energy(hybrid_result["x_post"]))
        print("||x_refine||^2:", energy(hybrid_result["x_refine"]))
        print("||x_hybrid||^2:", energy(hybrid_result["x_hybrid"]))

        corr_raw = float(np.corrcoef(hybrid_result["x_lbeads"], signal.x_true)[0, 1])
        print("Corr raw:", corr_raw)

        report["samples"].append(
            {
                "index": i + 1,
                "noise_level": noise_level,
                "amplitude_scale": amp,
                "selected_stage": hybrid_result["selected_stage"],
                "noise_sigma_normalized": float(hybrid_result["noise_sigma_normalized"]),
                "scale": float(hybrid_result["scale"]),
                "regularization": _to_python(hybrid_result["regularization"]),
                "quality_post": _to_python(hybrid_result["quality"]["post"]),
                "quality_short": _to_python(hybrid_result["quality"]["short_refine"]),
                "quality_full": _to_python(hybrid_result["quality"].get("full_refine"))
                if "full_refine" in hybrid_result["quality"]
                else None,
                "energies": {
                    "x_true": energy(signal.x_true),
                    "x_lbeads": energy(hybrid_result["x_lbeads"]),
                    "x_post": energy(hybrid_result["x_post"]),
                    "x_refine": energy(hybrid_result["x_refine"]),
                    "x_hybrid": energy(hybrid_result["x_hybrid"]),
                },
                "corr_raw": corr_raw,
            }
        )

    json_path = os.path.join(data_dir, "hybrid_diagnostics.json")
    txt_path = os.path.join(data_dir, "hybrid_diagnostics.txt")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(_to_python(report), f, indent=2)

    lines: List[str] = []
    lines.append("Hybrid Diagnostics Report")
    lines.append(f"Generated: {report['timestamp']}")
    lines.append("")
    lines.append("Training regime:")
    lines.append(f"  Noise std range: {report['training_regime']['noise_std_range']}")
    lines.append(f"  Peak width ranges: {report['training_regime']['peak_width_ranges']}")
    lines.append(f"  Epochs: {report['training_regime']['epochs']}")
    lines.append(f"  Loss config: {report['training_regime']['loss_config']}")
    lines.append("")
    lines.append("Learned params (full):")
    lines.append(str(report["learned_params_full"]))
    lines.append("")
    lines.append("Learned params (summary):")
    lines.append(str(report["learned_params_summary"]))
    lines.append("")
    for sample in report["samples"]:
        lines.append(f"Sample {sample['index']}:")
        lines.append(f"  selected_stage: {sample['selected_stage']}")
        lines.append(f"  noise_sigma_normalized: {sample['noise_sigma_normalized']}")
        lines.append(f"  scale: {sample['scale']}")
        lines.append(f"  regularization: {sample['regularization']}")
        lines.append(f"  quality_post: {sample['quality_post']}")
        lines.append(f"  quality_short: {sample['quality_short']}")
        if sample["quality_full"] is not None:
            lines.append(f"  quality_full: {sample['quality_full']}")
        lines.append(f"  energies: {sample['energies']}")
        lines.append(f"  Corr raw: {sample['corr_raw']}")
        lines.append("")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\nSaved:")
    print(f"  {json_path}")
    print(f"  {txt_path}")


if __name__ == "__main__":
    main()
