"""
Evaluation metrics for fast LBEADS on synthetic data.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

def compute_mse(x_true: np.ndarray, x_est: np.ndarray) -> float:
    return float(np.mean((x_true - x_est) ** 2))


def compute_mae(x_true: np.ndarray, x_est: np.ndarray) -> float:
    return float(np.mean(np.abs(x_true - x_est)))


def compute_snr_input(x_true: np.ndarray, y: np.ndarray, f_true: np.ndarray) -> float:
    signal_power = np.sum(x_true ** 2)
    noise = y - (x_true + f_true)
    noise_power = np.sum(noise ** 2)
    if noise_power < 1e-12:
        return float("inf")
    return float(10.0 * np.log10(signal_power / noise_power))


def compute_snr_output(x_true: np.ndarray, x_est: np.ndarray) -> float:
    signal_power = np.sum(x_true ** 2)
    error_power = np.sum((x_true - x_est) ** 2)
    if error_power < 1e-12:
        return float("inf")
    return float(10.0 * np.log10(signal_power / error_power))


def compute_psnr(x_true: np.ndarray, x_est: np.ndarray, max_value: Optional[float] = None) -> float:
    mse = np.mean((x_true - x_est) ** 2)
    if mse < 1e-12:
        return float("inf")
    if max_value is None:
        max_value = float(np.max(np.abs(x_true)))
    if max_value <= 0:
        return float("nan")
    return float(20.0 * np.log10(max_value) - 10.0 * np.log10(mse))


def compute_support_metrics(
    x_true: np.ndarray,
    x_est: np.ndarray,
    threshold_ratio: float = 0.1,
) -> Tuple[float, float, float]:
    max_val = np.max(np.abs(x_true))
    if max_val <= 0:
        return float("nan"), float("nan"), float("nan")

    threshold = threshold_ratio * max_val
    true_support = np.abs(x_true) >= threshold
    est_support = np.abs(x_est) >= threshold

    tp = np.sum(true_support & est_support)
    fp = np.sum(~true_support & est_support)
    fn = np.sum(true_support & ~est_support)

    precision = float(tp / (tp + fp + 1e-12))
    recall = float(tp / (tp + fn + 1e-12))
    f1 = float(2.0 * precision * recall / (precision + recall + 1e-12))

    return precision, recall, f1


def compute_metrics(
    x_true: np.ndarray,
    f_true: np.ndarray,
    y: np.ndarray,
    x_est: np.ndarray,
    f_est: np.ndarray,
    support_threshold_ratio: float = 0.1,
    psnr_max_value: Optional[float] = None,
) -> Dict[str, float]:
    snr_in = compute_snr_input(x_true, y, f_true)
    snr_out = compute_snr_output(x_true, x_est)
    precision, recall, f1 = compute_support_metrics(x_true, x_est, support_threshold_ratio)

    return {
        "mse_signal": compute_mse(x_true, x_est),
        "mse_baseline": compute_mse(f_true, f_est),
        "mae_signal": compute_mae(x_true, x_est),
        "mae_baseline": compute_mae(f_true, f_est),
        "snr_in": snr_in,
        "snr_out": snr_out,
        "delta_snr": snr_out - snr_in,
        "psnr": compute_psnr(x_true, x_est, psnr_max_value),
        "support_precision": precision,
        "support_recall": recall,
        "support_f1": f1,
    }


def aggregate_metrics(results: List[Dict[str, float]]) -> Dict[str, Tuple[float, float]]:
    if not results:
        return {}

    keys = sorted(results[0].keys())
    aggregated: Dict[str, Tuple[float, float]] = {}

    for key in keys:
        values = np.array([r[key] for r in results], dtype=float)
        mask = np.isfinite(values)
        if np.any(mask):
            aggregated[key] = (float(np.mean(values[mask])), float(np.std(values[mask])))
        else:
            aggregated[key] = (float("nan"), float("nan"))

    return aggregated


def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str,
    support_threshold_ratio: float = 0.1,
    psnr_max_value: Optional[float] = None,
    sample_indices: Optional[List[int]] = None,
    use_tqdm: bool = False,
    desc: Optional[str] = None,
) -> Tuple[List[Dict[str, float]], Dict[str, Tuple[float, float]], List[Dict[str, np.ndarray]]]:
    model.eval()
    results: List[Dict[str, float]] = []
    samples: List[Dict[str, np.ndarray]] = []

    sample_indices = sample_indices or []
    sample_set = set(sample_indices)

    iterator = dataloader
    if use_tqdm and tqdm is not None:
        iterator = tqdm(dataloader, desc=desc or "Eval", leave=False)

    with torch.no_grad():
        offset = 0
        for batch in iterator:
            y = batch["y"].to(device)
            x_true = batch["x_true"].to(device)
            f_true = batch["f_true"].to(device)

            x_est, f_est = model(y)

            y_np = y.detach().cpu().numpy()
            x_true_np = x_true.detach().cpu().numpy()
            f_true_np = f_true.detach().cpu().numpy()
            x_est_np = x_est.detach().cpu().numpy()
            f_est_np = f_est.detach().cpu().numpy()

            batch_size = y_np.shape[0]
            for i in range(batch_size):
                metrics = compute_metrics(
                    x_true_np[i],
                    f_true_np[i],
                    y_np[i],
                    x_est_np[i],
                    f_est_np[i],
                    support_threshold_ratio,
                    psnr_max_value,
                )
                results.append(metrics)

                idx = offset + i
                if idx in sample_set:
                    samples.append({
                        "index": idx,
                        "y": y_np[i],
                        "x_true": x_true_np[i],
                        "f_true": f_true_np[i],
                        "x_est": x_est_np[i],
                        "f_est": f_est_np[i],
                    })

            offset += batch_size

    aggregated = aggregate_metrics(results)
    return results, aggregated, samples
