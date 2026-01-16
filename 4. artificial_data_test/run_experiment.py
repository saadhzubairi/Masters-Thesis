"""
Run synthetic-data experiments for LBEADS.
"""

import argparse
import csv
import json
import logging
import os
import sys
from datetime import datetime
from typing import List, Optional

import numpy as np
import torch

from config import ExperimentConfig, config_to_dict
from synthetic_data import SyntheticDataGenerator, save_dataset
from training import SyntheticDataset, train_model
from evaluation import evaluate_model
from visualization import (
    plot_three_column,
    plot_component_overlay,
    plot_training_curves,
    plot_param_evolution,
)


def _add_lbeads_path() -> None:
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    lbeads_dir = os.path.join(root_dir, "Initial_implementations", "1. LBEADS_NETv1")
    if lbeads_dir not in sys.path:
        sys.path.insert(0, lbeads_dir)


_add_lbeads_path()
from lbeads_net import LBEADS_NET, LBEADS_NET_Fast  # noqa: E402


def setup_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger("artificial_data_test")
    logger.setLevel(logging.INFO)
    logger.handlers = []

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger


def create_run_dirs(run_root: str) -> dict:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(run_root, f"run_{timestamp}")
    subdirs = {
        "run": run_dir,
        "figures": os.path.join(run_dir, "figures"),
        "metrics": os.path.join(run_dir, "metrics"),
        "models": os.path.join(run_dir, "models"),
        "logs": os.path.join(run_dir, "logs"),
    }
    for path in subdirs.values():
        os.makedirs(path, exist_ok=False)
    return subdirs


def save_metrics_json(results: List[dict], summary: dict, output_prefix: str) -> None:
    with open(f"{output_prefix}_per_sample.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    with open(f"{output_prefix}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def save_metrics_csv(results: List[dict], output_path: str) -> None:
    if not results:
        return
    fieldnames = sorted(results[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def parse_list(arg: Optional[str], cast_fn=float) -> Optional[List]:
    if arg is None:
        return None
    items = [x.strip() for x in arg.split(",") if x.strip()]
    return [cast_fn(x) for x in items]


def main() -> None:
    parser = argparse.ArgumentParser(description="Artificial data test for LBEADS")
    parser.add_argument("--run-root", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--length", type=int, default=None)
    parser.add_argument("--n-train", type=int, default=None)
    parser.add_argument("--n-val", type=int, default=None)
    parser.add_argument("--n-test", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--model-type", type=str, choices=["fast", "full"], default=None)
    parser.add_argument("--shared-params", action="store_true")
    parser.add_argument("--skip-pretrain-eval", action="store_true")
    parser.add_argument("--pretrain-eval-samples", type=int, default=None)
    parser.add_argument("--fast-train-full-eval", action="store_true")
    parser.add_argument("--baseline-type", type=str, default=None)
    parser.add_argument("--noise-types", type=str, default=None)
    parser.add_argument("--noise-levels", type=str, default=None)
    parser.add_argument("--plot-samples", type=str, default=None)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--no-train", action="store_true")

    args = parser.parse_args()

    config = ExperimentConfig()

    if args.run_root:
        config.run_root = args.run_root
    if args.seed is not None:
        config.data.seed = args.seed
    if args.length is not None:
        config.data.signal_length = args.length
    if args.n_train is not None:
        config.data.n_train = args.n_train
    if args.n_val is not None:
        config.data.n_val = args.n_val
    if args.n_test is not None:
        config.data.n_test = args.n_test
    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.layers is not None:
        config.model.num_layers = args.layers
    if args.lr is not None:
        config.training.learning_rate = args.lr
    if args.device is not None:
        config.device = args.device
    if args.model_type is not None:
        config.model.model_type = args.model_type
    if args.shared_params:
        config.model.shared_params = True
    if args.baseline_type is not None:
        config.data.baseline_type = args.baseline_type
    if args.noise_types is not None:
        config.data.noise_types = parse_list(args.noise_types, cast_fn=str)
    if args.noise_levels is not None:
        config.data.noise_level_choices = parse_list(args.noise_levels, cast_fn=float)
    if args.plot_samples is not None:
        config.plot_samples = parse_list(args.plot_samples, cast_fn=int)

    run_dirs = create_run_dirs(config.run_root)
    logger = setup_logger(os.path.join(run_dirs["logs"], "run.log"))

    logger.info("Run directory: %s", run_dirs["run"])
    logger.info("Device override: %s", config.device)

    with open(os.path.join(run_dirs["logs"], "config.json"), "w", encoding="utf-8") as f:
        json.dump(config_to_dict(config), f, indent=2)

    np.random.seed(config.data.seed)
    torch.manual_seed(config.data.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.data.seed)

    device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    logger.info("Generating synthetic datasets...")
    train_gen = SyntheticDataGenerator(config.data.signal_length, seed=config.data.seed + 1000)
    val_gen = SyntheticDataGenerator(config.data.signal_length, seed=config.data.seed + 2000)
    test_gen = SyntheticDataGenerator(config.data.signal_length, seed=config.data.seed + 3000)

    train_signals = train_gen.generate_dataset(
        n_samples=config.data.n_train,
        noise_types=config.data.noise_types,
        noise_level_range=config.data.noise_level_range,
        noise_level_choices=config.data.noise_level_choices,
        baseline_type=config.data.baseline_type,
        poly_degree_range=config.data.poly_degree_range,
        poly_coeff_range=config.data.poly_coeff_range,
        sine_freq_range=config.data.sine_freq_range,
        sine_amp_range=config.data.sine_amp_range,
        lowpass_sigma_range=config.data.lowpass_sigma_range,
        lowpass_scale_range=config.data.lowpass_scale_range,
        num_peaks_range=config.data.peak_num_range,
        center_margin=config.data.peak_center_margin,
        width_range=config.data.peak_width_range,
        amplitude_range=config.data.peak_amplitude_range,
        positive_dominant=config.data.positive_dominant,
        negative_peak_prob=config.data.negative_peak_prob,
        negative_peak_scale=config.data.negative_peak_scale,
    )

    val_signals = val_gen.generate_dataset(
        n_samples=config.data.n_val,
        noise_types=config.data.noise_types,
        noise_level_range=config.data.noise_level_range,
        noise_level_choices=config.data.noise_level_choices,
        baseline_type=config.data.baseline_type,
        poly_degree_range=config.data.poly_degree_range,
        poly_coeff_range=config.data.poly_coeff_range,
        sine_freq_range=config.data.sine_freq_range,
        sine_amp_range=config.data.sine_amp_range,
        lowpass_sigma_range=config.data.lowpass_sigma_range,
        lowpass_scale_range=config.data.lowpass_scale_range,
        num_peaks_range=config.data.peak_num_range,
        center_margin=config.data.peak_center_margin,
        width_range=config.data.peak_width_range,
        amplitude_range=config.data.peak_amplitude_range,
        positive_dominant=config.data.positive_dominant,
        negative_peak_prob=config.data.negative_peak_prob,
        negative_peak_scale=config.data.negative_peak_scale,
    )

    test_signals = test_gen.generate_dataset(
        n_samples=config.data.n_test,
        noise_types=config.data.noise_types,
        noise_level_range=config.data.noise_level_range,
        noise_level_choices=config.data.noise_level_choices,
        baseline_type=config.data.baseline_type,
        poly_degree_range=config.data.poly_degree_range,
        poly_coeff_range=config.data.poly_coeff_range,
        sine_freq_range=config.data.sine_freq_range,
        sine_amp_range=config.data.sine_amp_range,
        lowpass_sigma_range=config.data.lowpass_sigma_range,
        lowpass_scale_range=config.data.lowpass_scale_range,
        num_peaks_range=config.data.peak_num_range,
        center_margin=config.data.peak_center_margin,
        width_range=config.data.peak_width_range,
        amplitude_range=config.data.peak_amplitude_range,
        positive_dominant=config.data.positive_dominant,
        negative_peak_prob=config.data.negative_peak_prob,
        negative_peak_scale=config.data.negative_peak_scale,
    )

    save_dataset(train_signals, os.path.join(run_dirs["metrics"], "train_dataset.npz"))
    save_dataset(val_signals, os.path.join(run_dirs["metrics"], "val_dataset.npz"))
    save_dataset(test_signals, os.path.join(run_dirs["metrics"], "test_dataset.npz"))

    train_loader = torch.utils.data.DataLoader(
        SyntheticDataset(train_signals),
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = torch.utils.data.DataLoader(
        SyntheticDataset(val_signals),
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_loader = torch.utils.data.DataLoader(
        SyntheticDataset(test_signals),
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model_type = config.model.model_type.lower()
    if args.fast_train_full_eval:
        model_type = "fast"
    if model_type == "fast":
        model = LBEADS_NET_Fast(
            N=config.data.signal_length,
            d=config.model.d,
            fc=config.model.fc,
            num_layers=config.model.num_layers,
            init_lam0=config.model.init_lam0,
            init_lam1=config.model.init_lam1,
            init_lam2=config.model.init_lam2,
            init_r=config.model.init_r,
            init_step_size=config.model.init_step_size,
        ).to(device)
        model_label = "Fast LBEADS"
        model_filename = "lbeads_fast.pth"
    elif model_type == "full":
        model = LBEADS_NET(
            N=config.data.signal_length,
            d=config.model.d,
            fc=config.model.fc,
            num_layers=config.model.num_layers,
            shared_params=config.model.shared_params,
            init_lam0=config.model.init_lam0,
            init_lam1=config.model.init_lam1,
            init_lam2=config.model.init_lam2,
            init_r=config.model.init_r,
            learn_r=config.model.learn_r,
        ).to(device)
        model_label = "LBEADS (full)"
        model_filename = "lbeads_full.pth"
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    logger.info("Model type: %s", model_type)
    logger.info("Shared params: %s", config.model.shared_params)
    logger.info("Learn r: %s", config.model.learn_r)
    logger.info("Fast-train full-eval: %s", args.fast_train_full_eval)

    if args.model_path:
        logger.info("Loading model from %s", args.model_path)
        checkpoint = torch.load(args.model_path, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)

    sample_indices = [idx for idx in config.plot_samples if idx < config.data.n_test]
    if not sample_indices:
        sample_indices = list(range(min(config.data.n_test, 3)))

    pre_summary = None
    skip_pretrain_eval = args.skip_pretrain_eval
    if model_type == "full" and args.pretrain_eval_samples is None:
        skip_pretrain_eval = True

    if not args.model_path and not args.no_train and not skip_pretrain_eval:
        eval_loader = test_loader
        if args.pretrain_eval_samples is not None:
            n_pre = max(1, min(args.pretrain_eval_samples, len(test_signals)))
            eval_loader = torch.utils.data.DataLoader(
                SyntheticDataset(test_signals[:n_pre]),
                batch_size=config.training.batch_size,
                shuffle=False,
                num_workers=0,
            )
            logger.info("Pre-train eval on %d samples", n_pre)

        logger.info("Evaluating pre-train model...")
        pre_results, pre_summary, _ = evaluate_model(
            model,
            eval_loader,
            device,
            support_threshold_ratio=config.evaluation.support_threshold_ratio,
            psnr_max_value=config.evaluation.psnr_max_value,
            use_tqdm=True,
            desc="Pre-eval",
        )
        save_metrics_json(pre_results, pre_summary, os.path.join(run_dirs["metrics"], "pretrain_metrics"))

    history = {}
    if not args.no_train:
        logger.info("Training %s...", model_label)
        history = train_model(
            model,
            train_loader,
            val_loader,
            config.training,
            device,
            logger,
        )
        with open(os.path.join(run_dirs["metrics"], "training_history.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    logger.info("Evaluating model on test set...")
    post_results, post_summary, samples = evaluate_model(
        model,
        test_loader,
        device,
        support_threshold_ratio=config.evaluation.support_threshold_ratio,
        psnr_max_value=config.evaluation.psnr_max_value,
        sample_indices=sample_indices,
        use_tqdm=True,
        desc="Post-eval",
    )

    save_metrics_json(post_results, post_summary, os.path.join(run_dirs["metrics"], "posttrain_metrics"))
    save_metrics_csv(post_results, os.path.join(run_dirs["metrics"], "posttrain_metrics.csv"))

    full_eval_results = None
    full_eval_summary = None
    full_eval_samples = None
    if args.fast_train_full_eval:
        logger.info("Evaluating full LBEADS (no training) on test set...")
        full_model = LBEADS_NET(
            N=config.data.signal_length,
            d=config.model.d,
            fc=config.model.fc,
            num_layers=config.model.num_layers,
            shared_params=config.model.shared_params,
            init_lam0=config.model.init_lam0,
            init_lam1=config.model.init_lam1,
            init_lam2=config.model.init_lam2,
            init_r=config.model.init_r,
            learn_r=config.model.learn_r,
        ).to(device)
        full_eval_results, full_eval_summary, full_eval_samples = evaluate_model(
            full_model,
            test_loader,
            device,
            support_threshold_ratio=config.evaluation.support_threshold_ratio,
            psnr_max_value=config.evaluation.psnr_max_value,
            sample_indices=sample_indices,
            use_tqdm=True,
            desc="Full-eval",
        )
        save_metrics_json(
            full_eval_results,
            full_eval_summary,
            os.path.join(run_dirs["metrics"], "full_eval_metrics"),
        )
        save_metrics_csv(
            full_eval_results,
            os.path.join(run_dirs["metrics"], "full_eval_metrics.csv"),
        )

    learned_params = model.get_learned_params() if hasattr(model, "get_learned_params") else {}

    model_path = os.path.join(run_dirs["models"], model_filename)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config_to_dict(config),
            "learned_params": learned_params,
        },
        model_path,
    )

    if history:
        plot_training_curves(history, os.path.join(run_dirs["figures"], "training_curves.png"))

    if learned_params:
        plot_param_evolution(learned_params, os.path.join(run_dirs["figures"], "param_evolution.png"))

    if full_eval_samples is not None:
        samples_sorted = sorted(full_eval_samples, key=lambda s: s["index"])
        plot_label = "LBEADS (full): input, baseline, sparse estimates"
    else:
        samples_sorted = sorted(samples, key=lambda s: s["index"])
        plot_label = f"{model_label}: input, baseline, sparse estimates"
    plot_three_column(
        samples_sorted,
        os.path.join(run_dirs["figures"], "comparison_grid.png"),
        plot_label,
    )

    if samples_sorted:
        plot_component_overlay(
            samples_sorted[0],
            "baseline",
            os.path.join(run_dirs["figures"], "baseline_overlay.png"),
        )
        plot_component_overlay(
            samples_sorted[0],
            "sparse",
            os.path.join(run_dirs["figures"], "sparse_overlay.png"),
        )

    summary_path = os.path.join(run_dirs["metrics"], "run_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        summary = {
            "posttrain_summary": post_summary,
            "model_path": model_path,
            "sample_indices": sample_indices,
            "model_type": model_type,
            "shared_params": config.model.shared_params,
            "learn_r": config.model.learn_r,
            "fast_train_full_eval": args.fast_train_full_eval,
        }
        if pre_summary is not None:
            summary["pretrain_summary"] = pre_summary
        if full_eval_summary is not None:
            summary["full_eval_summary"] = full_eval_summary
        json.dump(summary, f, indent=2)

    logger.info("Run complete.")
    logger.info("Outputs saved to: %s", run_dirs["run"])


if __name__ == "__main__":
    main()
