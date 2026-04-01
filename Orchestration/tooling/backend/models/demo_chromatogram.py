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
from scipy.signal import resample
from scipy.sparse.linalg import spsolve

from lbeads_net import LBEADS_NET, LBEADS_NET_Fast, BAfilt, HybridConfig, hybrid_infer_1d


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
    search_dirs = [
        script_dir,
        os.path.normpath(os.path.join(script_dir, "..", "5. LBEADS_NETv5 [Adaptive Post Processing]")),
        os.path.normpath(os.path.join(script_dir, "..", "5. LBEADS_NETv5")),
        os.path.normpath(os.path.join(script_dir, "..", "5. LBEADS_NETv5 VS BEADS")),
    ]
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

    # --- Load data (same as BEADS paper / Replicate/main.py) ---
    data_dir = os.path.normpath(os.path.join(script_dir, '..', '0. BEADS', 'data'))
    noise = sio.loadmat(os.path.join(data_dir, 'noise.mat'))['noise'].flatten()
    X = sio.loadmat(os.path.join(data_dir, 'chromatograms.mat'))['X']

    # y = X(:,3) + noise*0.5   (MATLAB col 3 → Python col 2)
    y = X[:, 2] + noise * 0.5
    N = len(y)
    print(f"Signal length N = {N}")

    # ── 1) Classical BEADS ───────────────────────────────────────────────────
    print("Running classical BEADS (30 iterations)...")
    amp = 0.8
    x_beads, f_beads = beads_classic(y, d=1, fc=0.006, r=6,
                                     lam0=0.5*amp, lam1=5*amp, lam2=4*amp)
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
        hybrid_result = hybrid_infer_1d(model, y, config=hybrid_cfg)
        x_net_np = hybrid_result["x_lbeads"]
        f_net_np = hybrid_result["f_lbeads"]
        x_hybrid = hybrid_result["x_hybrid"]
        f_hybrid = hybrid_result["f_hybrid"]
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
    axes[1].set_title('After: Classical BEADS (fc=0.006, r=6, 30 iterations)', fontsize=12)
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


def _hybrid_infer_resample(model, y, config):
    """Run hybrid_infer_1d, resampling the signal if it exceeds model N."""
    y = np.asarray(y, dtype=np.float64).flatten()
    N_orig = len(y)
    N_model = int(getattr(model, "N", N_orig))

    if N_orig <= N_model:
        return hybrid_infer_1d(model, y, config=config)

    # Resample down to model N, run inference, resample outputs back
    y_down = resample(y, N_model)
    hr = hybrid_infer_1d(model, y_down, config=config)

    # Resample all signal outputs back to original length
    for key in ("x_lbeads", "f_lbeads", "x_post", "f_post",
                "x_refine", "f_refine", "x_hybrid", "f_hybrid"):
        if key in hr:
            hr[key] = resample(np.asarray(hr[key], dtype=np.float64), N_orig)
    return hr


def _cosine_sim(a, b):
    dot = np.dot(a, b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(dot / max(na * nb, 1e-12))


def _corr(a, b):
    if np.std(a) < 1e-10 or np.std(b) < 1e-10:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _generate_synthetic_signals(data_config, N, n=3, seed=777):
    """Generate synthetic test signals using the data config (same as data generator page)."""
    from synth_generator import SyntheticDataGenerator, SyntheticSignal

    gen = SyntheticDataGenerator(N=N, seed=seed)
    bl = (data_config or {}).get("baseline", {})
    peak_layers = (data_config or {}).get("peak_layers", [{}])
    noise_cfg = (data_config or {}).get("noise", {})

    signals = []
    for _ in range(n):
        f_true, _ = gen.generate_baseline(
            smooth_sigma=bl.get("smooth_sigma", 100.0),
            sine_amp=bl.get("sine_amp", 0.1),
            sine_freq_range=(bl.get("sine_freq_min", 0.5), bl.get("sine_freq_max", 2.0)),
            baseline_amp_range=(bl.get("baseline_amp_min", 0.08), bl.get("baseline_amp_max", 0.35)),
        )
        x_true = np.zeros(N, dtype=np.float64)
        for layer in peak_layers:
            gen.peak_shape_mode = layer.get("peak_shape_mode", "mixed")
            peaks, _ = gen.generate_peaks(
                num_peaks_range=(layer.get("num_peaks_min", 2), layer.get("num_peaks_max", 6)),
                amplitude_range=(layer.get("amplitude_min", 0.2), layer.get("amplitude_max", 1.0)),
                rise_width_range=(layer.get("rise_width_min", 10), layer.get("rise_width_max", 80)),
                decay_width_range=(layer.get("decay_width_min", 20), layer.get("decay_width_max", 200)),
                plateau_width_range=(layer.get("plateau_width_min", 0), layer.get("plateau_width_max", 10)),
            )
            x_true += peaks
        noise, _ = gen.generate_noise(noise_level=noise_cfg.get("noise_level", 0.01))

        y = x_true + f_true + noise
        scale = max(float(np.max(np.abs(y))), 1e-8)
        target_amp = float(gen.rng.uniform(300.0, 1800.0))
        signals.append(SyntheticSignal(
            y=(y / scale) * target_amp,
            x_true=(x_true / scale) * target_amp,
            f_true=(f_true / scale) * target_amp,
            noise=(noise / scale) * target_amp,
            metadata={},
        ))
    return signals


def run_chromatogram_demo(checkpoint_path: str, output_dir: str, N: int = 4096,
                          data_config: dict = None):
    """
    Run chromatogram comparison demo from a saved checkpoint.

    Compares classical BEADS vs LBEADS-NET on:
      1. Legacy real chromatogram data
      2. 3 synthetic test signals (with ground truth)

    Produces 4 output images.
    """
    import matplotlib
    matplotlib.use('Agg')

    os.makedirs(output_dir, exist_ok=True)

    # Load model from checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    cfg = checkpoint.get('model_config', {})
    model_type = checkpoint.get('model_type', 'lbeads')

    loaded_layers = int(cfg.get('num_layers', 10))
    infer_layers = min(loaded_layers, 10)

    if model_type == 'lbeads_fast':
        model = LBEADS_NET_Fast(
            N=cfg.get('N', N),
            d=cfg.get('d', 1),
            fc=cfg.get('fc', 0.006),
            num_layers=infer_layers,
            lowpass_iterations=cfg.get('lowpass_iterations', 3),
            lowpass_cg_iters=cfg.get('lowpass_cg_iters', 12),
        )
    else:
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
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()

    hybrid_cfg = HybridConfig(
        noise_k=2.5, lowpass_iterations=3,
        short_refine_iterations=8, full_refine_iterations=24,
    )
    beads_fc = cfg.get('fc', 0.006)
    beads_d = cfg.get('d', 1)

    output_files = []

    # ── Image 1: Real chromatogram ──────────────────────────────────────────
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.normpath(os.path.join(script_dir, '..', '..', '..', '..',
                                              'Implementations', '0. BEADS', 'data'))
    if not os.path.isdir(data_dir):
        data_dir = os.path.normpath(os.path.join(script_dir, '..', '..', '..', '..',
                                                  '..', 'Implementations', '0. BEADS', 'data'))
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"Legacy chromatogram data directory not found. Tried: {data_dir}"
        )

    mat_noise = sio.loadmat(os.path.join(data_dir, 'noise.mat'))['noise'].flatten()
    X = sio.loadmat(os.path.join(data_dir, 'chromatograms.mat'))['X']
    y = X[:, 2] + mat_noise * 0.5
    N_sig = len(y)

    amp = 0.8
    x_beads, f_beads = beads_classic(y, d=beads_d, fc=beads_fc, r=6,
                                     lam0=0.5 * amp, lam1=5 * amp, lam2=4 * amp)
    hr = _hybrid_infer_resample(model, y, config=hybrid_cfg)

    m_raw = {
        "pk_corr": _corr(hr["x_lbeads"], x_beads),
        "pk_cos": _cosine_sim(hr["x_lbeads"], x_beads),
        "bl_corr": _corr(hr["f_lbeads"], f_beads),
    }
    m_hyb = {
        "pk_corr": _corr(hr["x_hybrid"], x_beads),
        "pk_cos": _cosine_sim(hr["x_hybrid"], x_beads),
        "bl_corr": _corr(hr["f_hybrid"], f_beads),
    }

    ylim = [-50, 200]
    fig, axes = plt.subplots(4, 1, figsize=(14, 18), sharex=True)
    axes[0].plot(y, 'b', linewidth=0.5)
    axes[0].set_title('Before: Raw Chromatogram (Column 3 + noise x 0.5)', fontsize=12)
    axes[0].set_ylabel('Amplitude'); axes[0].set_xlim([0, N_sig]); axes[0].set_ylim(ylim); axes[0].grid(True, alpha=0.3)

    axes[1].plot(x_beads, 'b', linewidth=0.8, label='Peaks (BEADS)')
    axes[1].plot(f_beads, 'r', linewidth=1.2, alpha=0.8, label='Baseline (BEADS)')
    axes[1].set_title('After: Classical BEADS (30 iterations)', fontsize=12)
    axes[1].set_ylabel('Amplitude'); axes[1].set_xlim([0, N_sig]); axes[1].set_ylim(ylim); axes[1].legend(loc='upper right'); axes[1].grid(True, alpha=0.3)

    axes[2].plot(hr["x_lbeads"], 'b', linewidth=0.8, label='Peaks (LBEADS-NET)')
    axes[2].plot(hr["f_lbeads"], 'r', linewidth=1.2, alpha=0.8, label='Baseline (LBEADS-NET)')
    axes[2].set_title(f'After: LBEADS-NET Raw — vs BEADS: pk_corr={m_raw["pk_corr"]:.3f}, cos={m_raw["pk_cos"]:.3f}, bl_corr={m_raw["bl_corr"]:.3f}', fontsize=10)
    axes[2].set_ylabel('Amplitude'); axes[2].set_xlim([0, N_sig]); axes[2].set_ylim(ylim); axes[2].legend(loc='upper right'); axes[2].grid(True, alpha=0.3)

    axes[3].plot(hr["x_hybrid"], 'b', linewidth=0.8, label='Peaks (Hybrid)')
    axes[3].plot(hr["f_hybrid"], 'r', linewidth=1.2, alpha=0.8, label='Baseline (Hybrid)')
    axes[3].set_title(f'After: Hybrid (stage={hr["selected_stage"]}) — vs BEADS: pk_corr={m_hyb["pk_corr"]:.3f}, cos={m_hyb["pk_cos"]:.3f}, bl_corr={m_hyb["bl_corr"]:.3f}', fontsize=10)
    axes[3].set_ylabel('Amplitude'); axes[3].set_xlim([0, N_sig]); axes[3].set_ylim(ylim); axes[3].legend(loc='upper right'); axes[3].grid(True, alpha=0.3)

    axes[-1].set_xlabel('Sample Index')
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'demo_chromatogram.png'), dpi=150)
    plt.close()
    output_files.append('demo_chromatogram.png')

    # ── Images 2-4: Synthetic BEADS vs LBEADS-NET (with ground truth) ───────
    synth_signals = _generate_synthetic_signals(data_config, N, n=3, seed=777)

    for idx, sig in enumerate(synth_signals):
        y_s = sig.y.astype(np.float64)
        N_s = len(y_s)

        # Classical BEADS on synthetic signal
        amp_s = 0.8
        x_beads_s, f_beads_s = beads_classic(y_s, d=beads_d, fc=beads_fc, r=6,
                                              lam0=0.5 * amp_s, lam1=5 * amp_s, lam2=4 * amp_s)
        # LBEADS-NET hybrid on synthetic signal
        hr_s = _hybrid_infer_resample(model, y_s, config=hybrid_cfg)

        # Metrics vs ground truth
        beads_pk_mse = float(np.mean((x_beads_s - sig.x_true) ** 2))
        beads_pk_corr = _corr(x_beads_s, sig.x_true)
        beads_bl_mse = float(np.mean((f_beads_s - sig.f_true) ** 2))
        beads_bl_corr = _corr(f_beads_s, sig.f_true)

        raw_pk_mse = float(np.mean((hr_s["x_lbeads"] - sig.x_true) ** 2))
        raw_pk_corr = _corr(hr_s["x_lbeads"], sig.x_true)
        raw_bl_mse = float(np.mean((hr_s["f_lbeads"] - sig.f_true) ** 2))
        raw_bl_corr = _corr(hr_s["f_lbeads"], sig.f_true)

        lbeads_pk_mse = float(np.mean((hr_s["x_hybrid"] - sig.x_true) ** 2))
        lbeads_pk_corr = _corr(hr_s["x_hybrid"], sig.x_true)
        lbeads_bl_mse = float(np.mean((hr_s["f_hybrid"] - sig.f_true) ** 2))
        lbeads_bl_corr = _corr(hr_s["f_hybrid"], sig.f_true)

        fig, axes = plt.subplots(5, 1, figsize=(14, 20), sharex=True)

        # Row 1: Observed
        axes[0].plot(y_s, 'k', linewidth=0.5, alpha=0.7)
        axes[0].set_title(f'Synthetic Signal {idx+1} — Observed', fontsize=12)
        axes[0].set_ylabel('Amplitude'); axes[0].set_xlim([0, N_s]); axes[0].grid(True, alpha=0.3)

        # Row 2: Ground truth
        axes[1].plot(sig.x_true, 'g', linewidth=0.8, label='True Peaks')
        axes[1].plot(sig.f_true, 'm', linewidth=0.8, alpha=0.8, label='True Baseline')
        axes[1].set_title('Ground Truth', fontsize=12)
        axes[1].set_ylabel('Amplitude'); axes[1].set_xlim([0, N_s]); axes[1].legend(loc='upper right'); axes[1].grid(True, alpha=0.3)

        # Row 3: Classical BEADS
        axes[2].plot(x_beads_s, 'b', linewidth=0.8, label='Peaks (BEADS)')
        axes[2].plot(f_beads_s, 'r', linewidth=0.8, alpha=0.8, label='Baseline (BEADS)')
        axes[2].plot(sig.x_true, 'g--', linewidth=0.6, alpha=0.5, label='True Peaks')
        axes[2].set_title(
            f'Classical BEADS — pk_MSE={beads_pk_mse:.2f}, pk_corr={beads_pk_corr:.3f}, '
            f'bl_MSE={beads_bl_mse:.2f}, bl_corr={beads_bl_corr:.3f}', fontsize=10)
        axes[2].set_ylabel('Amplitude'); axes[2].set_xlim([0, N_s]); axes[2].legend(loc='upper right', fontsize=7); axes[2].grid(True, alpha=0.3)

        # Row 4: LBEADS-NET Raw
        axes[3].plot(hr_s["x_lbeads"], 'b', linewidth=0.8, label='Peaks (Raw)')
        axes[3].plot(hr_s["f_lbeads"], 'r', linewidth=0.8, alpha=0.8, label='Baseline (Raw)')
        axes[3].plot(sig.x_true, 'g--', linewidth=0.6, alpha=0.5, label='True Peaks')
        axes[3].set_title(
            f'LBEADS-NET Raw — pk_MSE={raw_pk_mse:.2f}, pk_corr={raw_pk_corr:.3f}, '
            f'bl_MSE={raw_bl_mse:.2f}, bl_corr={raw_bl_corr:.3f}', fontsize=10)
        axes[3].set_ylabel('Amplitude'); axes[3].set_xlim([0, N_s]); axes[3].legend(loc='upper right', fontsize=7); axes[3].grid(True, alpha=0.3)

        # Row 5: LBEADS-NET Hybrid
        axes[4].plot(hr_s["x_hybrid"], 'b', linewidth=0.8, label='Peaks (Hybrid)')
        axes[4].plot(hr_s["f_hybrid"], 'r', linewidth=0.8, alpha=0.8, label='Baseline (Hybrid)')
        axes[4].plot(sig.x_true, 'g--', linewidth=0.6, alpha=0.5, label='True Peaks')
        axes[4].set_title(
            f'LBEADS-NET Hybrid (stage={hr_s["selected_stage"]}) — pk_MSE={lbeads_pk_mse:.2f}, '
            f'pk_corr={lbeads_pk_corr:.3f}, bl_MSE={lbeads_bl_mse:.2f}, bl_corr={lbeads_bl_corr:.3f}', fontsize=10)
        axes[4].set_ylabel('Amplitude'); axes[4].set_xlim([0, N_s]); axes[4].legend(loc='upper right', fontsize=7); axes[4].grid(True, alpha=0.3)

        axes[-1].set_xlabel('Sample Index')
        plt.tight_layout()
        fname = f'synthetic_beads_vs_lbeads_{idx+1}.png'
        fig.savefig(os.path.join(output_dir, fname), dpi=150)
        plt.close()
        output_files.append(fname)

    return output_files


if __name__ == "__main__":
    main()
