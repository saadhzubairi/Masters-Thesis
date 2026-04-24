"""
Real chromatogram demo: before & after baseline correction.

Loads legacy chromatogram data (column 3 + noise*0.5, exactly as in BEADS paper)
and shows:
  1. Raw observed signal
  2. Classical BEADS reconstruction (iterative, no training needed)
  3. LBEADS-NET v8 reconstruction (learned, single forward pass)

Usage:
    python demo_chromatogram.py
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import os
import glob
from scipy import sparse
from scipy.sparse.linalg import spsolve

from lbeads_net import LBEADS_NET, BAfilt, apply_lowpass_filter_np, compute_lowpass_matrix_np


# -- Classical BEADS algorithm (no training needed) --

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


# -- Model loader --

def load_best_model(script_dir, N):
    """Load the best available .pth model for N based on saved test metrics."""
    try:
        all_files = os.listdir(script_dir)
    except Exception:
        return None

    model_files = [
        os.path.join(script_dir, f)
        for f in all_files
        if f.endswith('.pth') and f.startswith('lbeads_net_')
    ]

    candidates = []
    for path in model_files:
        try:
            ckpt = torch.load(path, map_location='cpu', weights_only=False)
        except Exception:
            continue
        cfg = ckpt.get('model_config', {})
        if cfg.get('N') != N:
            continue
        tm = ckpt.get('test_metrics', {})
        corr = float(tm.get('correlation', -1.0))
        mse = float(tm.get('mse', 1e12))
        mae = float(tm.get('mae', 1e12))
        mtime = os.path.getmtime(path)
        candidates.append((-mae, -mse, corr, mtime, path, ckpt))

    if not candidates:
        return None

    candidates.sort(reverse=True)
    neg_mae, neg_mse, corr, _, path, ckpt = candidates[0]

    cfg = ckpt.get('model_config', {})
    print(f"Loaded model: {os.path.basename(path)}")
    if corr >= 0:
        print(f"  Selected by metrics: mae={-neg_mae:.6f}, mse={-neg_mse:.6f}, corr={corr:.4f}")

    model = LBEADS_NET(
        N=cfg['N'], d=cfg['d'], fc=cfg['fc'],
        num_layers=cfg['num_layers'],
        lowpass_iterations=cfg.get('lowpass_iterations', 3),
        solve_cg_iters=cfg.get('solve_cg_iters', 12),
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model


def postprocess_prediction(y, x_pred, lowpass_matrix, noise_k=2.5, lowpass_iters=3):
    """
    Lightweight cleanup:
    1) Adaptive threshold on x using high-pass residual MAD
    2) Recompute baseline as iterated low-pass of residual
    """
    from lbeads_net import apply_highpass_filter_np

    residual = y - x_pred
    noise_hp = apply_highpass_filter_np(residual, lowpass_matrix)
    sigma = np.median(np.abs(noise_hp)) / 0.6745 + 1e-8

    x_clean = np.maximum(x_pred - noise_k * sigma, 0.0)
    baseline = apply_lowpass_filter_np(y - x_clean, lowpass_matrix, iterations=lowpass_iters)
    return x_clean, baseline, sigma


# -- Main --

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    N_model = 4096

    # Load data (same as BEADS paper / Replicate/main.py).
    data_dir = os.path.normpath(os.path.join(script_dir, '..', '0. BEADS', 'data'))
    if not os.path.isdir(data_dir):
        print(f"Data directory not found: {data_dir}")
        print("Please ensure the BEADS data is available at ../0. BEADS/data/")
        return

    noise = sio.loadmat(os.path.join(data_dir, 'noise.mat'))['noise'].flatten()
    X = sio.loadmat(os.path.join(data_dir, 'chromatograms.mat'))['X']

    # y = X(:,3) + noise*0.5   (MATLAB col 3 -> Python col 2)
    y = X[:, 2] + noise * 0.5
    N = len(y)
    print(f"Signal length N = {N}")

    # 1) Classical BEADS
    print("Running classical BEADS (30 iterations)...")
    amp = 0.8
    x_beads, f_beads = beads_classic(y, d=1, fc=0.006, r=6,
                                     lam0=0.5*amp, lam1=5*amp, lam2=4*amp)
    print("  Done.")

    # 2) LBEADS-NET v8
    model = load_best_model(script_dir, N_model)
    has_model = model is not None

    if has_model:
        print("Running LBEADS-NET v8 (single forward pass)...")
        pad = N_model - N
        y_padded = np.pad(y, (0, pad), mode='reflect')
        y_scale = np.max(np.abs(y_padded))
        y_normed = y_padded / y_scale
        y_tensor = torch.tensor(y_normed, dtype=torch.float64).unsqueeze(0)
        with torch.no_grad():
            x_net, f_net = model(y_tensor)
            x_net = F.softplus(x_net, beta=20.0)
        x_net_np = x_net[0].numpy()[:N] * y_scale
        f_net_np = f_net[0].numpy()[:N] * y_scale

        # Post-process
        lp = compute_lowpass_matrix_np(N, d=1, fc=0.006)
        x_net_pp, f_net_pp, sigma = postprocess_prediction(y, x_net_np, lp)
        print(f"  Post-process sigma={sigma:.4f}")
        print("  Done.")
    else:
        print("No trained model found -- showing only classical BEADS.")

    # Plot
    ylim = [-50, 200]
    xlim = [0, N]
    n_panels = 4 if has_model else 2
    fig, axes = plt.subplots(n_panels, 1, figsize=(14, 4 * n_panels), sharex=True)

    axes[0].plot(y, 'b', linewidth=0.5)
    axes[0].set_title('Before: Raw Chromatogram (Column 3 + noise x 0.5)', fontsize=12)
    axes[0].set_ylabel('Amplitude')
    axes[0].set_xlim(xlim)
    axes[0].set_ylim(ylim)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x_beads, 'b', linewidth=0.8, label='Peaks (BEADS)')
    axes[1].plot(f_beads, 'r', linewidth=1.2, alpha=0.8, label='Baseline (BEADS)')
    axes[1].set_title('After: Classical BEADS (fc=0.006, r=6, 30 iterations)', fontsize=12)
    axes[1].set_ylabel('Amplitude')
    axes[1].set_xlim(xlim)
    axes[1].set_ylim(ylim)
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)

    if has_model:
        axes[2].plot(x_net_np, 'b', linewidth=0.8, label='Peaks (LBEADS-NET v8)')
        axes[2].plot(f_net_np, 'r', linewidth=1.2, alpha=0.8, label='Baseline (LBEADS-NET v8)')
        axes[2].set_title('After: LBEADS-NET v8 Raw Output', fontsize=12)
        axes[2].set_ylabel('Amplitude')
        axes[2].set_xlim(xlim)
        axes[2].set_ylim(ylim)
        axes[2].legend(loc='upper right')
        axes[2].grid(True, alpha=0.3)

        axes[3].plot(x_net_pp, 'b', linewidth=0.8, label='Peaks (v8 + post)')
        axes[3].plot(f_net_pp, 'r', linewidth=1.2, alpha=0.8, label='Baseline (v8 + post)')
        axes[3].set_title('After: LBEADS-NET v8 + Adaptive Post-Processing', fontsize=12)
        axes[3].set_ylabel('Amplitude')
        axes[3].set_xlim(xlim)
        axes[3].set_ylim(ylim)
        axes[3].legend(loc='upper right')
        axes[3].grid(True, alpha=0.3)

    axes[-1].set_xlabel('Sample Index')
    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, 'demo_chromatogram.png'), dpi=150)
    print("Saved demo_chromatogram.png")
    plt.show()


if __name__ == "__main__":
    main()
