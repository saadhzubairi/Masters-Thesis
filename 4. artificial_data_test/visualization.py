"""
Visualization utilities for artificial data experiments.
"""

from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt


def apply_plot_style() -> None:
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        plt.style.use("default")
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 200,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "lines.linewidth": 1.2,
    })


def plot_three_column(samples: List[Dict], output_path: str, title: str) -> None:
    if not samples:
        return

    apply_plot_style()
    n_rows = len(samples)
    fig, axes = plt.subplots(n_rows, 3, figsize=(12, 3.2 * n_rows))

    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, sample in enumerate(samples):
        axes[row, 0].plot(sample["y"], color="#1f77b4")
        axes[row, 0].set_ylabel("Amplitude")
        axes[row, 0].set_title("Input signal" if row == 0 else "")

        axes[row, 1].plot(sample["f_est"], color="#2ca02c")
        axes[row, 1].set_ylabel("Amplitude")
        axes[row, 1].set_title("Estimated baseline" if row == 0 else "")

        axes[row, 2].plot(sample["x_est"], color="#d62728")
        axes[row, 2].set_ylabel("Amplitude")
        axes[row, 2].set_title("Estimated sparse signal" if row == 0 else "")

        if row == n_rows - 1:
            for col in range(3):
                axes[row, col].set_xlabel("Sample index")

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_component_overlay(sample: Dict, component: str, output_path: str) -> None:
    apply_plot_style()
    fig, ax = plt.subplots(figsize=(10, 3.5))

    if component == "baseline":
        ax.plot(sample["f_true"], color="#1f77b4", label="True baseline")
        ax.plot(sample["f_est"], color="#ff7f0e", label="Estimated baseline", linestyle="--")
        ax.set_title("Baseline estimation")
    elif component == "sparse":
        ax.plot(sample["x_true"], color="#1f77b4", label="True sparse")
        ax.plot(sample["x_est"], color="#ff7f0e", label="Estimated sparse", linestyle="--")
        ax.set_title("Sparse signal recovery")
    else:
        raise ValueError("component must be 'baseline' or 'sparse'")

    ax.set_xlabel("Sample index")
    ax.set_ylabel("Amplitude")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(history: Dict[str, List[float]], output_path: str) -> None:
    if not history:
        return

    apply_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(history["train_loss"], label="Train")
    axes[0, 0].plot(history["val_loss"], label="Validation")
    axes[0, 0].set_title("Total loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()

    axes[0, 1].plot(history["train_signal"], label="Train signal")
    axes[0, 1].plot(history["val_signal"], label="Val signal")
    axes[0, 1].set_title("Signal loss")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].legend()

    axes[1, 0].plot(history["train_baseline"], label="Train baseline")
    axes[1, 0].plot(history["val_baseline"], label="Val baseline")
    axes[1, 0].set_title("Baseline loss")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Loss")
    axes[1, 0].legend()

    axes[1, 1].plot(history["lr"], label="Learning rate")
    axes[1, 1].set_title("Learning rate")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("LR")
    axes[1, 1].set_yscale("log")
    axes[1, 1].legend()

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _extract_layer_params(params: Dict[str, float]) -> Dict[str, List[float]]:
    layer_values: Dict[str, Dict[int, float]] = {}
    for key, value in params.items():
        if not key.startswith("layer_"):
            continue
        rest = key[len("layer_") :]
        parts = rest.split("_", 1)
        if len(parts) != 2:
            continue
        layer_idx, name = parts
        layer_idx = int(layer_idx)
        layer_values.setdefault(name, {})
        layer_values[name][layer_idx] = value

    ordered: Dict[str, List[float]] = {}
    for name, mapping in layer_values.items():
        max_idx = max(mapping.keys())
        ordered[name] = [mapping.get(i, float("nan")) for i in range(max_idx + 1)]

    return ordered


def plot_param_evolution(params: Dict[str, float], output_path: str) -> None:
    if not params:
        return

    apply_plot_style()
    layer_params = _extract_layer_params(params)
    if not layer_params:
        return

    fig, axes = plt.subplots(len(layer_params), 1, figsize=(8, 2.6 * len(layer_params)))
    if len(layer_params) == 1:
        axes = [axes]

    for ax, (name, values) in zip(axes, layer_params.items()):
        ax.plot(values, marker="o", markersize=3)
        ax.set_title(f"{name} across layers")
        ax.set_xlabel("Layer index")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
