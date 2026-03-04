"""
Simple chromatogram demo: before & after baseline correction.

Loads legacy chromatogram data (column 3 + noise*0.5, exactly as in BEADS paper)
and shows three panels:
  1. Raw observed signal
  2. Classical BEADS reconstruction (iterative, no training needed)
  3. LBEADS-NET reconstruction (learned, single forward pass)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import os
import glob
from scipy import sparse
from scipy.sparse.linalg import spsolve

from lbeads_net import LBEADS_NET, BAfilt, HybridConfig, hybrid_infer_1d


# ── Classical BEADS algorithm (no training needed) ──────────────────────────

def beads_classic(y, d=1, fc=0.006, r=6, lam0=0.4, lam1=4.0, lam2=3.2,
                  Nit=30, EPS0=1e-6, EPS1=1e-6):
    """Run the original iterative BEADS algorithm."""
    wfun = lambda x: 1.0 / (np.abs(x) + EPS1)
    y = np.asarray(y, dtype=np.float64).flatten()
    N = len(y)
    x = y.copy()

    A, B = BAfilt(d, fc, N)
    e = np.ones(N)
    D1 = sparse.spdiags([-e[:-1], e[:-1]], [0, 1], N - 1, N, format='csc')
    D2 = sparse.spdiags([e[:-2], -2 * e[:-2], e[:-2]], [0, 1, 2], N - 2, N, format='csc')
    D = sparse.vstack([D1, D2], format='csc')
    BTB = B.T @ B
    w = np.concatenate([lam1 * np.ones(N - 1), lam2 * np.ones(N - 2)])
    b_vec = (1 - r) / 2 * np.ones(N)
    d_vec = BTB @ spsolve(A, y) - lam0 * A.T @ b_vec
    gamma = np.ones(N)

    for _ in range(Nit):
        Dx = D @ x
        Lambda = sparse.diags(w * wfun(Dx), 0, format='csc')
        k = np.abs(x) > EPS0
        gamma[~k] = ((1 + r) / 4) / abs(EPS0)
        gamma[k] = ((1 + r) / 4) / np.abs(x[k])
        Gamma = sparse.diags(gamma, 0, format='csc')
        M = 2 * lam0 * Gamma + D.T @ Lambda @ D
        x = A @ spsolve(BTB + A.T @ M @ A, d_vec)

    residual = y - x
    f = y - x - B @ spsolve(A, residual)
    return x, f


def fit_endpoint_parabola(y):
    """Fit quadratic trend through start/middle/end anchors."""
    y = np.asarray(y, dtype=np.float64).flatten()
    n = y.size
    if n < 3:
        return np.zeros_like(y), {"coeffs": [0.0, 0.0, 0.0], "anchors": [0, 0, 0]}
    anchors_x = np.array([0.0, float(n // 2), float(n - 1)], dtype=np.float64)
    anchors_y = np.array([y[0], y[n // 2], y[-1]], dtype=np.float64)
    coeffs = np.polyfit(anchors_x, anchors_y, deg=2)
    trend = np.polyval(coeffs, np.arange(n, dtype=np.float64))
    return trend, {"coeffs": coeffs.tolist(), "anchors": [0, int(n // 2), int(n - 1)]}


def preprocess_signal_for_beads(y, apply_endpoint_parabola=True):
    """
    Preprocess signal for BEADS-style endpoint assumptions.

    Returns:
        y_proc: Preprocessed signal (input for BEADS/LBEADS)
        trend: Removed trend (add back to baseline output)
        meta:  Metadata about preprocessing
    """
    y = np.asarray(y, dtype=np.float64).flatten()
    if not apply_endpoint_parabola:
        return y.copy(), np.zeros_like(y), {"enabled": False}
    trend, trend_meta = fit_endpoint_parabola(y)
    y_proc = y - trend
    meta = {"enabled": True, **trend_meta}
    return y_proc, trend, meta


# ── Model loader ─────────────────────────────────────────────────────────────

def _infer_model_variant(checkpoint):
    """
    Infer checkpoint model variant from metadata or state_dict keys.
    """
    cfg = checkpoint.get('model_config', {})
    declared = str(cfg.get('model_variant', '')).lower()
    if declared in ('lbeads_net', 'non_fast', 'classic'):
        return 'lbeads_net'
    if declared in ('lbeads_net_fast', 'fast'):
        return 'lbeads_net_fast'

    state_dict = checkpoint.get('model_state_dict', {})
    keys = list(state_dict.keys())
    if any(k.startswith('log_step_size') for k in keys):
        return 'lbeads_net_fast'
    if any(k.startswith('layers.') for k in keys):
        return 'lbeads_net'
    return 'unknown'


def load_best_model(script_dir, N):
    """
    Load the best available .pth model for N based on saved test metrics.

    Falls back to newest timestamp if metrics are missing.
    """
    search_dirs = [script_dir]
    model_files = []
    for d in search_dirs:
        if os.path.isdir(d):
            escaped_d = glob.escape(d)
            model_files.extend(glob.glob(os.path.join(escaped_d, "*.pth")))
    candidates = []
    skipped_fast = 0

    for path in model_files:
        try:
            ckpt = torch.load(path, map_location='cpu', weights_only=False)
        except Exception:
            continue
        cfg = ckpt.get('model_config', {})
        if cfg.get('N') != N:
            continue
        model_variant = _infer_model_variant(ckpt)
        if model_variant != 'lbeads_net':
            if model_variant == 'lbeads_net_fast':
                skipped_fast += 1
            continue

        tm = ckpt.get('test_metrics', {})
        corr = float(tm.get('correlation', -1.0))
        mse = float(tm.get('mse', 1e12))
        mae = float(tm.get('mae', 1e12))
        mtime = os.path.getmtime(path)
        # Rank by low error first, then high correlation, then recency.
        candidates.append((-mae, -mse, corr, mtime, path, ckpt))

    if not candidates:
        if skipped_fast > 0:
            print(f"No compatible non-fast checkpoints found (skipped {skipped_fast} fast checkpoint(s)).")
        return None

    # Lowest mae/mse, then highest corr, then newest.
    candidates.sort(reverse=True)
    neg_mae, neg_mse, corr, _, path, ckpt = candidates[0]

    cfg = ckpt.get('model_config', {})
    print(f"Loaded model: {os.path.basename(path)}")
    print(f"  Source dir: {os.path.dirname(path)}")
    if corr >= 0:
        print(f"  Selected by metrics: mae={-neg_mae:.6f}, mse={-neg_mse:.6f}, corr={corr:.4f}")

    loaded_layers = int(cfg.get('num_layers', 20))
    infer_layers = min(loaded_layers, 10)
    if infer_layers < loaded_layers:
        print(f"  Using shallow inference depth: {infer_layers} layers (checkpoint has {loaded_layers}).")

    model = LBEADS_NET(
        N=cfg.get('N', N),
        d=cfg.get('d', 1),
        fc=cfg.get('fc', 0.006),
        num_layers=infer_layers,
        shared_params=cfg.get('shared_params', False),
        lowpass_iterations=cfg.get('lowpass_iterations', 1),
        solve_cg_iters=cfg.get('solve_cg_iters', 18),
        lowpass_cg_iters=cfg.get('lowpass_cg_iters', 12),
    )
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.eval()
    return model


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "demo-chrom")
    os.makedirs(output_dir, exist_ok=True)
    N_model = 4096
    classical_fc = 0.006
    apply_endpoint_parabola_preprocessing = True

    # --- Load data (same as BEADS paper / Replicate/main.py) ---
    data_dir = os.path.normpath(os.path.join(script_dir, '..', '..', '0. BEADS', 'data'))
    noise = sio.loadmat(os.path.join(data_dir, 'noise.mat'))['noise'].flatten()
    X = sio.loadmat(os.path.join(data_dir, 'chromatograms.mat'))['X']

    # y = X(:,3) + noise*0.5   (MATLAB col 3 → Python col 2)
    y = X[:, 2] + noise * 0.5
    N = len(y)
    print(f"Signal length N = {N}")
    y_proc, trend, preprocess_meta = preprocess_signal_for_beads(
        y,
        apply_endpoint_parabola=apply_endpoint_parabola_preprocessing,
    )
    if preprocess_meta.get("enabled", False):
        coeffs = preprocess_meta.get("coeffs", [0.0, 0.0, 0.0])
        print(
            "Endpoint preprocessing: parabola subtraction enabled "
            f"(a={coeffs[0]:.3e}, b={coeffs[1]:.3e}, c={coeffs[2]:.3e})"
        )
    else:
        print("Endpoint preprocessing: disabled")

    # ── 1) Classical BEADS ───────────────────────────────────────────────────
    print("Running classical BEADS (30 iterations)...")
    amp = 0.8
    x_beads, f_beads = beads_classic(y_proc, d=1, fc=classical_fc, r=6,
                                     lam0=0.5*amp, lam1=5*amp, lam2=4*amp)
    f_beads = f_beads + trend
    print("  Done.")

    # ── 2) LBEADS-NET ────────────────────────────────────────────────────────
    model = load_best_model(script_dir, N_model)
    has_model = model is not None

    if has_model:
        print("Running LBEADS-NET (single forward pass)...")
        hybrid_cfg = HybridConfig(
            noise_k=2.5,
            lowpass_iterations=3,
            short_refine_iterations=8,
            full_refine_iterations=24,
        )
        hybrid_result = hybrid_infer_1d(model, y_proc, config=hybrid_cfg)
        x_net_np = hybrid_result["x_lbeads"]
        f_net_np = hybrid_result["f_lbeads"] + trend
        x_hybrid = hybrid_result["x_hybrid"]
        f_hybrid = hybrid_result["f_hybrid"] + trend
        selected_stage = hybrid_result["selected_stage"]
        qsel = hybrid_result["quality_selected"]
        print(f"  Hybrid selected stage: {selected_stage}")
        print(
            "  Hybrid quality:"
            f" score={qsel['score']:.4f},"
            f" active_fraction={qsel['active_fraction']:.4f},"
            f" baseline_hf_ratio={qsel['baseline_hf_ratio']:.4f}"
        )
        print("  Done.")
    else:
        print("No trained model found — showing only classical BEADS.")

    # ── Plot ─────────────────────────────────────────────────────────────────
    ylim = [-50, 200]
    xlim = [0, N]
    n_panels = 4 if has_model else 2
    fig, axes = plt.subplots(n_panels, 1, figsize=(14, 4 * n_panels), sharex=True)

    # Panel 1: Raw signal
    axes[0].plot(y, 'b', linewidth=0.5)
    axes[0].set_title('Before: Raw Chromatogram (Column 3 + noise × 0.5)', fontsize=12)
    axes[0].set_ylabel('Amplitude')
    axes[0].set_xlim(xlim)
    axes[0].set_ylim(ylim)
    axes[0].grid(True, alpha=0.3)

    # Panel 2: Classical BEADS
    axes[1].plot(x_beads, 'b', linewidth=0.8, label='Peaks (BEADS)')
    axes[1].plot(f_beads, 'r', linewidth=1.2, alpha=0.8, label='Baseline (BEADS)')
    endpoint_label = "parabola-preprocessed" if apply_endpoint_parabola_preprocessing else "raw-input"
    axes[1].set_title(
        f'After: Classical BEADS (fc={classical_fc:.4f}, {endpoint_label}, r=6, 30 iterations)',
        fontsize=12,
    )
    axes[1].set_ylabel('Amplitude')
    axes[1].set_xlim(xlim)
    axes[1].set_ylim(ylim)
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)

    # Panel 3: LBEADS-NET
    if has_model:
        axes[2].plot(x_net_np, 'b', linewidth=0.8, label='Peaks (LBEADS-NET)')
        axes[2].plot(f_net_np, 'r', linewidth=1.2, alpha=0.8, label='Baseline (LBEADS-NET)')
        axes[2].set_title('After: LBEADS-NET Raw Output', fontsize=12)
        axes[2].set_ylabel('Amplitude')
        axes[2].set_xlim(xlim)
        axes[2].set_ylim(ylim)
        axes[2].legend(loc='upper right')
        axes[2].grid(True, alpha=0.3)

        # Panel 4: Hybrid-selected output
        axes[3].plot(x_hybrid, 'b', linewidth=0.8, label='Peaks (Hybrid)')
        axes[3].plot(f_hybrid, 'r', linewidth=1.2, alpha=0.8, label='Baseline (Hybrid)')
        axes[3].set_title(f'After: Hybrid (selected={selected_stage})', fontsize=12)
        axes[3].set_ylabel('Amplitude')
        axes[3].set_xlim(xlim)
        axes[3].set_ylim(ylim)
        axes[3].legend(loc='upper right')
        axes[3].grid(True, alpha=0.3)

    axes[-1].set_xlabel('Sample Index')
    plt.tight_layout()
    chromatogram_path = os.path.join(output_dir, 'demo_chromatogram.png')
    plt.savefig(chromatogram_path, dpi=150)
    print(f"Saved {chromatogram_path}")
    plt.show()


if __name__ == "__main__":
    main()
