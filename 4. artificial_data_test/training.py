"""
Training utilities for fast LBEADS on synthetic data.
"""

from typing import Dict, List, Tuple
import copy
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from synthetic_data import SyntheticSignal


class SyntheticDataset(Dataset):
    def __init__(self, signals: List[SyntheticSignal]):
        self.signals = signals

    def __len__(self) -> int:
        return len(self.signals)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        signal = self.signals[idx]
        return {
            "y": torch.tensor(signal.y, dtype=torch.float64),
            "x_true": torch.tensor(signal.x_true, dtype=torch.float64),
            "f_true": torch.tensor(signal.f_true, dtype=torch.float64),
        }


class CombinedLoss(nn.Module):
    def __init__(self, signal_weight: float = 1.0, baseline_weight: float = 0.5):
        super().__init__()
        self.signal_weight = signal_weight
        self.baseline_weight = baseline_weight
        self.mse = nn.MSELoss()

    def forward(
        self,
        x_est: torch.Tensor,
        f_est: torch.Tensor,
        x_true: torch.Tensor,
        f_true: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        loss_signal = self.mse(x_est, x_true)
        loss_baseline = self.mse(f_est, f_true)
        total = self.signal_weight * loss_signal + self.baseline_weight * loss_baseline
        return total, loss_signal, loss_baseline


def _train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: CombinedLoss,
    optimizer: torch.optim.Optimizer,
    device: str,
    grad_clip: float,
) -> Dict[str, float]:
    model.train()
    total = 0.0
    total_sig = 0.0
    total_base = 0.0

    for batch in dataloader:
        y = batch["y"].to(device)
        x_true = batch["x_true"].to(device)
        f_true = batch["f_true"].to(device)

        optimizer.zero_grad()
        x_est, f_est = model(y)
        loss, loss_sig, loss_base = criterion(x_est, f_est, x_true, f_true)
        loss.backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        total += loss.item()
        total_sig += loss_sig.item()
        total_base += loss_base.item()

    n_batches = max(1, len(dataloader))
    return {
        "loss": total / n_batches,
        "signal_loss": total_sig / n_batches,
        "baseline_loss": total_base / n_batches,
    }


def _validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: CombinedLoss,
    device: str,
) -> Dict[str, float]:
    model.eval()
    total = 0.0
    total_sig = 0.0
    total_base = 0.0

    with torch.no_grad():
        for batch in dataloader:
            y = batch["y"].to(device)
            x_true = batch["x_true"].to(device)
            f_true = batch["f_true"].to(device)

            x_est, f_est = model(y)
            loss, loss_sig, loss_base = criterion(x_est, f_est, x_true, f_true)

            total += loss.item()
            total_sig += loss_sig.item()
            total_base += loss_base.item()

    n_batches = max(1, len(dataloader))
    return {
        "loss": total / n_batches,
        "signal_loss": total_sig / n_batches,
        "baseline_loss": total_base / n_batches,
    }


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config,
    device: str,
    logger,
) -> Dict[str, List[float]]:
    criterion = CombinedLoss(config.signal_loss_weight, config.baseline_loss_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=config.lr_scheduler_patience,
        min_lr=1e-6,
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_signal": [],
        "val_signal": [],
        "train_baseline": [],
        "val_baseline": [],
        "lr": [],
    }

    best_val = float("inf")
    best_state = None
    patience = 0

    for epoch in range(1, config.epochs + 1):
        train_metrics = _train_epoch(
            model, train_loader, criterion, optimizer, device, config.grad_clip
        )
        val_metrics = _validate_epoch(model, val_loader, criterion, device)

        scheduler.step(val_metrics["loss"])
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_signal"].append(train_metrics["signal_loss"])
        history["val_signal"].append(val_metrics["signal_loss"])
        history["train_baseline"].append(train_metrics["baseline_loss"])
        history["val_baseline"].append(val_metrics["baseline_loss"])
        history["lr"].append(current_lr)

        logger.info(
            "Epoch %d/%d | train=%.6f val=%.6f | sig=%.6f base=%.6f | lr=%.2e",
            epoch,
            config.epochs,
            train_metrics["loss"],
            val_metrics["loss"],
            val_metrics["signal_loss"],
            val_metrics["baseline_loss"],
            current_lr,
        )

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= config.early_stop_patience:
                logger.info("Early stopping at epoch %d", epoch)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return history
