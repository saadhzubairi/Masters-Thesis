"""
LBEADS-NET v8: Learnable BEADS Network (Final)

Combines v5's ISTA proximal-gradient architecture with v7's O(N) banded
operators and training infrastructure.

Architecture:
  - Forward pass: highpass init -> loop(gradient step -> asymmetric soft threshold) -> lowpass baseline
  - 5 learnable params per layer via nn.ParameterList: log_lam0, log_lam1, log_lam2, log_r, log_step_size
  - Banded O(N) operators from v7 replace dense N x N matrices
  - Supports return_intermediate=True for intermediate supervision

The ISTA variant (LBEADS_NET) is the primary model.  A CG-based variant
(LBEADS_NET_CG) is included for reference and comparison.

Reference:
Original BEADS: Chromatogram baseline estimation and denoising using sparsity
Xiaoran Ning, Ivan W. Selesnick, Laurent Duval
Chemometrics and Intelligent Laboratory Systems (2014)
"""

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
from scipy import sparse
from scipy.sparse.linalg import spsolve, factorized


# =============================================================================
# Banded Operator Infrastructure (from v7)
# =============================================================================


def _compute_filter_coefficients(d: int, fc: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return symmetric FIR coefficients for BEADS banded filters A and B.

    Args:
        d: Filter degree (filter order is 2d; use d=1 or d=2)
        fc: Normalized cutoff frequency (0 < fc < 0.5)

    Returns:
        a: Coefficient vector of length 2d+1 for filter A
        b: Coefficient vector of length 2d+1 for filter B
    """
    b1 = np.array([1.0, -1.0], dtype=np.float64)
    for _ in range(d - 1):
        b1 = np.convolve(b1, np.array([-1.0, 2.0, -1.0], dtype=np.float64))
    b = np.convolve(b1, np.array([-1.0, 1.0], dtype=np.float64))

    omc = 2 * np.pi * fc
    t = ((1 - np.cos(omc)) / (1 + np.cos(omc))) ** d

    a = np.array([1.0], dtype=np.float64)
    for _ in range(d):
        a = np.convolve(a, np.array([1.0, 2.0, 1.0], dtype=np.float64))
    a = b + t * a
    return a.astype(np.float64), b.astype(np.float64)


def _banded_apply(x: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
    """
    Apply a constant-diagonal banded operator to batched row vectors.

    The operator has bandwidth 2d+1 with offsets [-d, ..., d].

    Args:
        x: (N,) or (batch, N)
        coeffs: (2*d+1,) with offsets [-d, ..., d]

    Returns:
        out: Same shape as x
    """
    squeeze_output = False
    if x.dim() == 1:
        x = x.unsqueeze(0)
        squeeze_output = True

    batch, n = x.shape
    d = (coeffs.numel() - 1) // 2
    out = torch.zeros(batch, n, dtype=x.dtype, device=x.device)

    for idx in range(coeffs.numel()):
        offset = idx - d
        c = coeffs[idx]
        if offset < 0:
            out[:, : n + offset] += c * x[:, -offset:]
        elif offset > 0:
            out[:, offset:] += c * x[:, : n - offset]
        else:
            out += c * x

    if squeeze_output:
        out = out.squeeze(0)
    return out


def _banded_apply_T(x: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
    """Apply transpose of a constant-diagonal banded operator."""
    return _banded_apply(x, torch.flip(coeffs, dims=(0,)))


def _cg_solve_fixed(matvec, b: torch.Tensor, max_iter: int = 16,
                    eps: float = 1e-12, x0: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Batched fixed-iteration conjugate gradient solver for SPD systems.

    Args:
        matvec: callable mapping (batch, N) -> (batch, N)
        b: RHS tensor (N,) or (batch, N)
        max_iter: fixed number of CG steps
        eps: numerical stabilizer
        x0: optional initial guess with same shape as b
    """
    squeeze_output = False
    if b.dim() == 1:
        b = b.unsqueeze(0)
        squeeze_output = True

    if x0 is None:
        x = torch.zeros_like(b)
    else:
        x = x0.unsqueeze(0) if x0.dim() == 1 else x0.clone()

    r = b - matvec(x)
    p = r.clone()
    rr = torch.sum(r * r, dim=1)

    for _ in range(max(1, int(max_iter))):
        Ap = matvec(p)
        denom = torch.sum(p * Ap, dim=1) + eps
        alpha = rr / denom
        x = x + alpha.unsqueeze(1) * p
        r = r - alpha.unsqueeze(1) * Ap
        rr_new = torch.sum(r * r, dim=1)
        beta = rr_new / (rr + eps)
        p = r + beta.unsqueeze(1) * p
        rr = rr_new

    if squeeze_output:
        x = x.squeeze(0)
    return x


def _solve_tridiagonal_constant(
    b: torch.Tensor,
    lower: torch.Tensor,
    diag: torch.Tensor,
    upper: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Solve a batched tridiagonal system with constant diagonals via Thomas algorithm.

    Matrix form:
      diag on main diagonal, lower on sub-diagonal, upper on super-diagonal.
    """
    squeeze_output = False
    if b.dim() == 1:
        b = b.unsqueeze(0)
        squeeze_output = True

    batch, n = b.shape
    lower = torch.as_tensor(lower, dtype=b.dtype, device=b.device)
    diag = torch.as_tensor(diag, dtype=b.dtype, device=b.device)
    upper = torch.as_tensor(upper, dtype=b.dtype, device=b.device)

    if n == 1:
        out = b / (diag + eps)
        return out.squeeze(0) if squeeze_output else out

    c_prime = torch.empty(n - 1, dtype=b.dtype, device=b.device)
    d_prime = torch.empty(batch, n, dtype=b.dtype, device=b.device)

    denom0 = diag + eps
    c_prime[0] = upper / denom0
    d_prime[:, 0] = b[:, 0] / denom0

    for i in range(1, n - 1):
        denom = (diag - lower * c_prime[i - 1]) + eps
        c_prime[i] = upper / denom
        d_prime[:, i] = (b[:, i] - lower * d_prime[:, i - 1]) / denom

    denom_last = (diag - lower * c_prime[n - 2]) + eps
    d_prime[:, n - 1] = (b[:, n - 1] - lower * d_prime[:, n - 2]) / denom_last

    x = torch.empty_like(b)
    x[:, n - 1] = d_prime[:, n - 1]
    for i in range(n - 2, -1, -1):
        x[:, i] = d_prime[:, i] - c_prime[i] * x[:, i + 1]

    if squeeze_output:
        x = x.squeeze(0)
    return x


def _solve_A_system(rhs: torch.Tensor, a_coeff: torch.Tensor,
                    solve_cg_iters: int = 16,
                    x0: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Solve A z = rhs.

    Uses exact tridiagonal solve for d=1 (len(a_coeff)==3), otherwise CG.
    """
    if int(a_coeff.numel()) == 3:
        upper = a_coeff[0]
        diag = a_coeff[1]
        lower = a_coeff[2]
        return _solve_tridiagonal_constant(rhs, lower=lower, diag=diag, upper=upper)

    solve = lambda v: _banded_apply(v, a_coeff)
    return _cg_solve_fixed(solve, rhs, max_iter=solve_cg_iters, x0=x0)


def _highpass_once_torch(signal: torch.Tensor, a_coeff: torch.Tensor,
                         b_coeff: torch.Tensor, solve_cg_iters: int) -> torch.Tensor:
    """High-pass operator H(v) = B @ (A^{-1} v) via exact/iterative A-solve."""
    z = _solve_A_system(signal, a_coeff, solve_cg_iters=solve_cg_iters, x0=signal)
    return _banded_apply(z, b_coeff)


# =============================================================================
# Legacy Scipy Sparse Helpers (for BAfilt, classical BEADS, etc.)
# =============================================================================


def BAfilt(d, fc, N):
    """
    Banded matrices for zero-phase high-pass filter (scipy sparse).

    INPUT
        d  : degree of filter is 2d (use d = 1 or 2)
        fc : cut-off frequency (normalized frequency, 0 < fc < 0.5)
        N  : length of signal

    OUTPUT
        A  : Symmetric banded matrix (scipy sparse)
        B  : Banded matrix (scipy sparse)
    """
    a, b = _compute_filter_coefficients(d, fc)

    diagonals_A = []
    diagonals_B = []
    offsets = list(range(-d, d + 1))

    for k, offset in enumerate(offsets):
        if offset <= 0:
            diag_len = N + offset
        else:
            diag_len = N - offset
        diagonals_A.append(np.full(diag_len, float(a[k])))
        diagonals_B.append(np.full(diag_len, float(b[k])))

    A = sparse.diags(diagonals_A, offsets, shape=(N, N), format='csc', dtype=np.float64)
    B = sparse.diags(diagonals_B, offsets, shape=(N, N), format='csc', dtype=np.float64)

    return A, B


def build_difference_matrices(N):
    """
    Build first and second order difference matrices.

    Returns:
        D1: First difference matrix (N-1) x N
        D2: Second difference matrix (N-2) x N
        D: Stacked difference matrix [D1; D2]
    """
    e = np.ones(N)
    D1 = sparse.spdiags([-e[:-1], e[:-1]], [0, 1], N - 1, N, format='csc')
    D2 = sparse.spdiags([e[:-2], -2 * e[:-2], e[:-2]], [0, 1, 2], N - 2, N, format='csc')
    D = sparse.vstack([D1, D2], format='csc')
    return D1, D2, D


def sparse_to_torch(sp_matrix, device='cpu'):
    """Convert scipy sparse matrix to PyTorch sparse tensor."""
    coo = sp_matrix.tocoo()
    indices = torch.LongTensor(np.vstack([coo.row, coo.col]))
    values = torch.DoubleTensor(coo.data)
    shape = torch.Size(coo.shape)
    return torch.sparse_coo_tensor(indices, values, shape, device=device)


# =============================================================================
# Lowpass / Highpass Filter Application (dual-mode: banded or legacy dense)
# =============================================================================


def apply_lowpass_filter(residual, lowpass_or_a_coeff, b_coeff: Optional[torch.Tensor] = None,
                         iterations: int = 1, solve_cg_iters: int = 16):
    """
    Apply low-pass filtering.

    Two modes:
    1) Legacy dense mode: apply_lowpass_filter(residual, lowpass_dense)
    2) Operator mode:     apply_lowpass_filter(residual, a_coeff, b_coeff, iterations, solve_cg_iters)
    """
    # Legacy dense-matrix mode.
    if b_coeff is None:
        lowpass_matrix = lowpass_or_a_coeff
        squeeze_output = False
        if residual.dim() == 1:
            residual = residual.unsqueeze(0)
            squeeze_output = True
        baseline = residual @ lowpass_matrix.T
        if squeeze_output:
            baseline = baseline.squeeze(0)
        return baseline

    out = residual
    for _ in range(max(1, int(iterations))):
        high = _highpass_once_torch(out, lowpass_or_a_coeff, b_coeff, solve_cg_iters)
        out = out - high
    return out


def apply_highpass_filter(signal, lowpass_or_a_coeff, b_coeff: Optional[torch.Tensor] = None,
                          solve_cg_iters: int = 16):
    """
    Apply high-pass filtering.

    Two modes:
    1) Legacy dense mode: apply_highpass_filter(signal, lowpass_dense)
    2) Operator mode:     apply_highpass_filter(signal, a_coeff, b_coeff, solve_cg_iters)
    """
    if b_coeff is None:
        low = apply_lowpass_filter(signal, lowpass_or_a_coeff)
        return signal - low
    return _highpass_once_torch(signal, lowpass_or_a_coeff, b_coeff, solve_cg_iters)


# =============================================================================
# NumPy-side lowpass / highpass (for post-processing and diagnostics)
# =============================================================================


def compute_lowpass_matrix_np(N, d=1, fc=0.006):
    """
    Build a sparse low-pass operator using factorized solves (no dense NxN matrix).
    """
    A_sp, B_sp = BAfilt(d, fc, N)
    solve_A = factorized(A_sp)
    return {
        "A_sp": A_sp,
        "B_sp": B_sp,
        "solve_A": solve_A,
    }


def apply_lowpass_filter_np(signal, lowpass_matrix, iterations=1):
    """Apply low-pass operator one or more times to a 1D NumPy signal."""
    out = np.asarray(signal, dtype=np.float64).copy()

    # Dense compatibility path.
    if isinstance(lowpass_matrix, np.ndarray):
        for _ in range(max(1, int(iterations))):
            out = lowpass_matrix @ out
        return out

    solve_A = lowpass_matrix["solve_A"]
    B_sp = lowpass_matrix["B_sp"]
    for _ in range(max(1, int(iterations))):
        z = solve_A(out)
        hp = B_sp @ z
        out = out - hp
    return out


def apply_highpass_filter_np(signal, lowpass_matrix):
    """Apply complementary high-pass filter H = B @ (A^{-1} v)."""
    sig = np.asarray(signal, dtype=np.float64)

    # Dense compatibility path.
    if isinstance(lowpass_matrix, np.ndarray):
        return sig - (lowpass_matrix @ sig)

    z = lowpass_matrix["solve_A"](sig)
    return np.asarray(lowpass_matrix["B_sp"] @ z, dtype=np.float64)


# =============================================================================
# Classical BEADS (warm-start capable, for hybrid inference)
# =============================================================================


def beads_classic_with_init(y, d=1, fc=0.006, r=6.0, lam0=0.4, lam1=4.0, lam2=3.2,
                            Nit=30, EPS0=1e-6, EPS1=1e-6, x_init=None):
    """
    Classical BEADS solver with optional warm start.

    Args:
        y: 1D observed signal
        x_init: optional initial x for iterative updates
    """
    wfun = lambda x: 1.0 / (np.abs(x) + EPS1)
    y = np.asarray(y, dtype=np.float64).flatten()
    N = len(y)
    if x_init is None:
        x = y.copy()
    else:
        x = np.asarray(x_init, dtype=np.float64).flatten().copy()
        if x.shape != y.shape:
            raise ValueError("x_init must have same shape as y")

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

    for _ in range(int(Nit)):
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


# =============================================================================
# LBEADS_NET: ISTA Proximal-Gradient Model (Primary Architecture)
# =============================================================================


class LBEADS_NET(nn.Module):
    """
    LBEADS-NET v8: ISTA proximal-gradient unrolled network with banded O(N) operators.

    This is the primary model architecture combining:
    - v5's ISTA forward pass (gradient step + asymmetric soft threshold per layer)
    - v7's banded operators (O(N) memory and compute instead of O(N^2))

    Each layer has 5 learnable parameters (in log space for positivity):
        log_lam0:      Asymmetric sparsity penalty weight
        log_lam1:      First-derivative regularization weight
        log_lam2:      Second-derivative regularization weight
        log_r:         Asymmetry ratio (penalize negatives r times more)
        log_step_size: Proximal gradient step size

    Args:
        N: Signal length
        d: Filter order (1 or 2)
        fc: Filter cutoff frequency (normalized, 0 < fc < 0.5)
        num_layers: Number of unrolled ISTA iterations
        init_lam0: Initial lam0 value
        init_lam1: Initial lam1 value
        init_lam2: Initial lam2 value
        init_r: Initial asymmetry ratio
        init_step_size: Initial gradient step size
        lowpass_iterations: Number of iterated lowpass applications for baseline
        solve_cg_iters: CG iterations for A-system solves (train; increased at inference)
    """

    def __init__(self, N, d=1, fc=0.006, num_layers=6,
                 init_lam0=0.01, init_lam1=0.5, init_lam2=0.5,
                 init_r=6.0, init_step_size=0.05,
                 lowpass_iterations=3,
                 solve_cg_iters=12):
        super(LBEADS_NET, self).__init__()

        self.N = N
        self.d = d
        self.fc = fc
        self.num_layers = num_layers
        self.lowpass_iterations = int(lowpass_iterations)
        self.solve_cg_iters = int(solve_cg_iters)
        self.EPS0 = 1e-6
        self.EPS1 = 1e-6

        # Banded filter coefficients (O(d) memory, not O(N^2)).
        a_np, b_np = _compute_filter_coefficients(d, fc)
        self.register_buffer('a_coeff', torch.tensor(a_np, dtype=torch.float64))
        self.register_buffer('b_coeff', torch.tensor(b_np, dtype=torch.float64))

        # Keep sparse matrices for NumPy-side utilities only.
        A, B = BAfilt(d, fc, N)
        self.A_sp = A
        self.B_sp = B

        # Layer-wise learnable parameters (all in log space for positivity).
        self.log_lam0 = nn.ParameterList([
            nn.Parameter(torch.tensor(np.log(init_lam0), dtype=torch.float64))
            for _ in range(num_layers)
        ])
        self.log_lam1 = nn.ParameterList([
            nn.Parameter(torch.tensor(np.log(init_lam1), dtype=torch.float64))
            for _ in range(num_layers)
        ])
        self.log_lam2 = nn.ParameterList([
            nn.Parameter(torch.tensor(np.log(init_lam2), dtype=torch.float64))
            for _ in range(num_layers)
        ])
        self.log_r = nn.ParameterList([
            nn.Parameter(torch.tensor(np.log(init_r), dtype=torch.float64))
            for _ in range(num_layers)
        ])
        self.log_step_size = nn.ParameterList([
            nn.Parameter(torch.tensor(np.log(init_step_size), dtype=torch.float64))
            for _ in range(num_layers)
        ])

    # ----- O(N) difference operators (from v5, already efficient) -----

    def diff1(self, x):
        """First difference: D1 @ x. Shape (batch, N) -> (batch, N-1)."""
        return x[:, 1:] - x[:, :-1]

    def diff2(self, x):
        """Second difference: D2 @ x. Shape (batch, N) -> (batch, N-2)."""
        return x[:, 2:] - 2 * x[:, 1:-1] + x[:, :-2]

    def diff1_T(self, v):
        """Transpose of first difference: D1.T @ v. Shape (batch, N-1) -> (batch, N)."""
        batch_size = v.shape[0]
        N = v.shape[1] + 1
        result = torch.zeros(batch_size, N, dtype=v.dtype, device=v.device)
        result[:, :-1] -= v
        result[:, 1:] += v
        return result

    def diff2_T(self, v):
        """Transpose of second difference: D2.T @ v. Shape (batch, N-2) -> (batch, N)."""
        batch_size = v.shape[0]
        N = v.shape[1] + 2
        result = torch.zeros(batch_size, N, dtype=v.dtype, device=v.device)
        result[:, :-2] += v
        result[:, 1:-1] -= 2 * v
        result[:, 2:] += v
        return result

    # ----- Proximal operator -----

    def asymmetric_soft_threshold(self, x, lam, r):
        """
        Asymmetric soft thresholding for BEADS.

        Penalizes negative values r times more than positive.
        For positive x: threshold at lam.
        For negative x: threshold at lam * r.
        Values between thresholds are zeroed (sparsity).
        """
        pos_thresh = lam
        neg_thresh = lam * r

        result = torch.zeros_like(x)
        pos_mask = x > pos_thresh
        neg_mask = x < -neg_thresh

        result[pos_mask] = x[pos_mask] - pos_thresh
        result[neg_mask] = x[neg_mask] + neg_thresh

        return result

    # ----- Forward pass (ISTA unrolling) -----

    def forward(self, y, return_intermediate=False):
        """
        Forward pass using ISTA-style proximal gradient with asymmetric thresholding.

        Each layer performs:
          1. Compute data fidelity gradient via banded highpass operator
          2. Compute smoothness penalty gradients (D1, D2)
          3. Gradient descent step
          4. Asymmetric soft thresholding (proximal step)

        After all layers, the baseline is extracted via banded lowpass filtering.

        Args:
            y: Input signal (N,) or (batch, N)
            return_intermediate: If True, also return per-layer peak estimates

        Returns:
            x: Estimated peaks (batch, N) or (N,)
            f: Estimated baseline (batch, N) or (N,)
            intermediates: (optional) List of per-layer x estimates
        """
        if y.dim() == 1:
            y = y.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        # Use tighter solves at inference for numerical quality.
        effective_cg_iters = self.solve_cg_iters if self.training else max(self.solve_cg_iters, 48)

        # Initialize x from highpass-filtered y: x0 = y - lowpass(y).
        # This is a much better starting point than zeros -- without it,
        # small peaks below the initial threshold are permanently lost.
        f_init = apply_lowpass_filter(
            y,
            self.a_coeff,
            self.b_coeff,
            iterations=self.lowpass_iterations,
            solve_cg_iters=effective_cg_iters,
        )
        x = y - f_init
        intermediates = [x.clone()] if return_intermediate else None

        for k in range(self.num_layers):
            lam0 = torch.exp(self.log_lam0[k])
            lam1 = torch.exp(self.log_lam1[k])
            lam2 = torch.exp(self.log_lam2[k])
            r = torch.exp(self.log_r[k])
            step_size = torch.exp(self.log_step_size[k])

            # === DATA FIDELITY GRADIENT ===
            # residual = y - x; highpass(residual) gives gradient toward data
            # in the peak (high-frequency) subspace, ignoring baseline.
            residual = y - x
            data_grad = apply_highpass_filter(
                residual,
                self.a_coeff,
                self.b_coeff,
                solve_cg_iters=effective_cg_iters,
            )

            # === SMOOTHNESS PENALTY GRADIENTS ===
            # D1 penalty gradient (first derivative)
            Dx1 = self.diff1(x)
            w1 = Dx1 / (torch.abs(Dx1) + self.EPS1)
            grad_D1 = lam1 * self.diff1_T(w1)

            # D2 penalty gradient (second derivative)
            Dx2 = self.diff2(x)
            w2 = Dx2 / (torch.abs(Dx2) + self.EPS1)
            grad_D2 = lam2 * self.diff2_T(w2)

            # === GRADIENT DESCENT STEP ===
            x_update = x + step_size * data_grad - step_size * (grad_D1 + grad_D2)

            # === PROXIMAL STEP (ASYMMETRIC SOFT THRESHOLDING) ===
            x = self.asymmetric_soft_threshold(x_update, lam0 * step_size, r)

            if return_intermediate:
                intermediates.append(x.clone())

        # === COMPUTE BASELINE ===
        # Lowpass filter the residual (y - x) to get the smooth baseline.
        residual = y - x
        f = apply_lowpass_filter(
            residual,
            self.a_coeff,
            self.b_coeff,
            iterations=self.lowpass_iterations,
            solve_cg_iters=effective_cg_iters,
        )

        if squeeze_output:
            x = x.squeeze(0)
            f = f.squeeze(0)

        if return_intermediate:
            return x, f, intermediates
        return x, f

    def get_learned_params(self):
        """Return dictionary of current learned parameters per layer."""
        params = {}
        for i in range(self.num_layers):
            params[f"layer_{i}_lam0"] = torch.exp(self.log_lam0[i]).item()
            params[f"layer_{i}_lam1"] = torch.exp(self.log_lam1[i]).item()
            params[f"layer_{i}_lam2"] = torch.exp(self.log_lam2[i]).item()
            params[f"layer_{i}_r"] = torch.exp(self.log_r[i]).item()
            params[f"layer_{i}_step_size"] = torch.exp(self.log_step_size[i]).item()
        return params

    def load_state_dict(self, state_dict, strict: bool = True):
        """Backward-compatible loading from legacy checkpoints with dense buffers."""
        remap = dict(state_dict)
        for key in ("A_dense", "B_dense", "BTB_dense", "lowpass_dense"):
            remap.pop(key, None)

        legacy_missing_coeffs = ("a_coeff" not in remap) or ("b_coeff" not in remap)
        return super().load_state_dict(remap, strict=False if legacy_missing_coeffs else strict)


# =============================================================================
# LBEADS_NET_CG: CG-Based Model (Reference/Comparison from v7)
# =============================================================================


class BEADSLayer(nn.Module):
    """
    A single unrolled BEADS iteration layer using CG-based linear system solve.

    Each layer solves the BEADS majorization-minimization linear system
    using matrix-free conjugate gradient with banded operators.

    Parameters:
        lam0: Regularization for asymmetric penalty (sparsity)
        lam1: Regularization for first derivative penalty
        lam2: Regularization for second derivative penalty
        r: Asymmetry ratio
        step: Layer relaxation step
    """

    def __init__(self, N, d, fc,
                 init_lam0=0.001, init_lam1=0.2, init_lam2=0.2,
                 init_r=6.0, learn_r=True,
                 init_step=1.0, learn_step=True,
                 EPS0=1e-6, EPS1=1e-6):
        super(BEADSLayer, self).__init__()

        self.N = N
        self.d = int(d)
        self.fc = float(fc)
        self.EPS0 = EPS0
        self.EPS1 = EPS1

        # Learnable parameters (stored in log space for positivity)
        self.log_lam0 = nn.Parameter(torch.tensor(np.log(init_lam0), dtype=torch.float64))
        self.log_lam1 = nn.Parameter(torch.tensor(np.log(init_lam1), dtype=torch.float64))
        self.log_lam2 = nn.Parameter(torch.tensor(np.log(init_lam2), dtype=torch.float64))

        if learn_r:
            self.log_r = nn.Parameter(torch.tensor(np.log(init_r), dtype=torch.float64))
        else:
            self.register_buffer('log_r', torch.tensor(np.log(init_r), dtype=torch.float64))

        if learn_step:
            self.log_step = nn.Parameter(torch.tensor(np.log(init_step), dtype=torch.float64))
        else:
            self.register_buffer('log_step', torch.tensor(np.log(init_step), dtype=torch.float64))

    @property
    def lam0(self):
        return torch.clamp(torch.exp(self.log_lam0), min=1e-4, max=5e-2)

    @property
    def lam1(self):
        return torch.clamp(torch.exp(self.log_lam1), min=1e-4, max=1.0)

    @property
    def lam2(self):
        return torch.clamp(torch.exp(self.log_lam2), min=1e-4, max=1.0)

    @property
    def r(self):
        return torch.clamp(torch.exp(self.log_r), min=2.0, max=12.0)

    @property
    def step(self):
        return torch.clamp(torch.exp(self.log_step), min=5e-2, max=2.0)

    def wfun(self, x):
        """Penalty weight function (L1_v2)."""
        return 1.0 / (torch.abs(x) + self.EPS1)

    @staticmethod
    def diff1(x: torch.Tensor) -> torch.Tensor:
        return x[:, 1:] - x[:, :-1]

    @staticmethod
    def diff2(x: torch.Tensor) -> torch.Tensor:
        return x[:, 2:] - 2 * x[:, 1:-1] + x[:, :-2]

    @staticmethod
    def diff1_T(v: torch.Tensor) -> torch.Tensor:
        batch = v.shape[0]
        n = v.shape[1] + 1
        out = torch.zeros(batch, n, dtype=v.dtype, device=v.device)
        out[:, :-1] -= v
        out[:, 1:] += v
        return out

    @staticmethod
    def diff2_T(v: torch.Tensor) -> torch.Tensor:
        batch = v.shape[0]
        n = v.shape[1] + 2
        out = torch.zeros(batch, n, dtype=v.dtype, device=v.device)
        out[:, :-2] += v
        out[:, 1:-1] -= 2 * v
        out[:, 2:] += v
        return out

    def forward(self, x, d_vec, a_coeff: torch.Tensor, b_coeff: torch.Tensor, cg_iters: int = 12):
        """
        Forward pass: one BEADS iteration via matrix-free CG.

        Args:
            x: Current estimate (N,) or (batch, N)
            d_vec: Pre-computed RHS vector
            a_coeff: Banded coefficients for A
            b_coeff: Banded coefficients for B
            cg_iters: CG iterations for linear solve
        """
        lam0, lam1, lam2, r = self.lam0, self.lam1, self.lam2, self.r

        if x.dim() == 1:
            x = x.unsqueeze(0)
            d_vec = d_vec.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        Dx1 = self.diff1(x)
        Dx2 = self.diff2(x)
        lambda1 = lam1 * self.wfun(Dx1)
        lambda2 = lam2 * self.wfun(Dx2)

        gamma = torch.ones_like(x) * (((1 + r) / 4.0) / abs(self.EPS0))
        mask = torch.abs(x) > self.EPS0
        gamma[mask] = ((1 + r) / 4.0) / torch.abs(x[mask])

        def lhs_matvec(v: torch.Tensor) -> torch.Tensor:
            Av = _banded_apply(v, a_coeff)
            D1Av = self.diff1(Av)
            D2Av = self.diff2(Av)
            reg = 2.0 * lam0 * gamma * Av
            reg = reg + self.diff1_T(lambda1 * D1Av) + self.diff2_T(lambda2 * D2Av)
            At_reg = _banded_apply_T(reg, a_coeff)
            Bv = _banded_apply(v, b_coeff)
            BTBv = _banded_apply_T(Bv, b_coeff)
            return BTBv + At_reg

        z_init = _solve_A_system(x.detach(), a_coeff, solve_cg_iters=cg_iters, x0=x.detach())
        z = _cg_solve_fixed(lhs_matvec, d_vec, max_iter=cg_iters, x0=z_init)
        x_new = _banded_apply(z, a_coeff)
        x_new = x + self.step * (x_new - x)

        if squeeze_output:
            x_new = x_new.squeeze(0)

        return x_new


class LBEADS_NET_CG(nn.Module):
    """
    LBEADS-NET CG variant: unrolled BEADS with conjugate gradient linear system solves.

    This is the v7 architecture preserved for reference and comparison.
    For new work, prefer LBEADS_NET (ISTA variant) which has better gradient flow.

    Args:
        N: Signal length
        d: Filter order (1 or 2)
        fc: Filter cutoff frequency
        num_layers: Number of unrolled iterations
        shared_params: If True, all layers share parameters
        init_lam0, init_lam1, init_lam2: Initial regularization parameters
        init_r: Initial asymmetry ratio
        learn_r: Whether to make r learnable
        init_step: Initial relaxation step per layer
        learn_step: Whether to learn relaxation steps
        lowpass_iterations: Number of iterated lowpass for baseline
        solve_cg_iters: CG iterations for layer solves (train)
        lowpass_cg_iters: CG iterations for lowpass (train)
    """

    def __init__(self, N, d=1, fc=0.006, num_layers=10,
                 shared_params=True,
                 init_lam0=0.001, init_lam1=0.2, init_lam2=0.2,
                 init_r=6.0, learn_r=True,
                 init_step=1.0, learn_step=True,
                 lowpass_iterations=1,
                 solve_cg_iters=16,
                 lowpass_cg_iters=32):
        super(LBEADS_NET_CG, self).__init__()

        self.N = N
        self.d = d
        self.fc = fc
        self.num_layers = num_layers
        self.shared_params = shared_params
        self.lowpass_iterations = int(lowpass_iterations)
        self.solve_cg_iters = int(solve_cg_iters)
        self.lowpass_cg_iters = int(lowpass_cg_iters)

        # Shared banded coefficients.
        a_np, b_np = _compute_filter_coefficients(d, fc)
        self.register_buffer('a_coeff', torch.tensor(a_np, dtype=torch.float64))
        self.register_buffer('b_coeff', torch.tensor(b_np, dtype=torch.float64))

        A, B = BAfilt(d, fc, N)
        self.A_sp = A
        self.B_sp = B

        ones = torch.ones(1, N, dtype=torch.float64)
        self.register_buffer('AT_ones', _banded_apply_T(ones, self.a_coeff).squeeze(0), persistent=False)

        # Create layers
        if shared_params:
            self.layers = nn.ModuleList([
                BEADSLayer(N, d, fc, init_lam0, init_lam1, init_lam2,
                           init_r, learn_r, init_step, learn_step)
            ])
        else:
            self.layers = nn.ModuleList([
                BEADSLayer(N, d, fc, init_lam0, init_lam1, init_lam2,
                           init_r, learn_r, init_step, learn_step)
                for _ in range(num_layers)
            ])

    def compute_d_vec(self, d_base, layer_idx=0):
        """Compute layer-specific RHS: d = BTB * (A^-1 * y) - lam0 * A^T * b."""
        layer = self.layers[layer_idx] if not self.shared_params else self.layers[0]
        r = layer.r
        lam0 = layer.lam0

        squeeze_output = False
        if d_base.dim() == 1:
            d_base = d_base.unsqueeze(0)
            squeeze_output = True

        b_scale = (1.0 - r) / 2.0
        AT_b = b_scale * self.AT_ones.unsqueeze(0)
        d_vec = d_base - lam0 * AT_b

        if squeeze_output:
            d_vec = d_vec.squeeze(0)
        return d_vec

    def forward(self, y, return_intermediate=False):
        """Forward pass through all unrolled CG layers."""
        if y.dim() == 1:
            y = y.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        effective_solve_cg_iters = self.solve_cg_iters if self.training else max(self.solve_cg_iters, 128)
        effective_lowpass_cg_iters = self.lowpass_cg_iters if self.training else max(self.lowpass_cg_iters, 128)

        # Initialize with highpass residual.
        f0 = apply_lowpass_filter(
            y, self.a_coeff, self.b_coeff,
            iterations=self.lowpass_iterations,
            solve_cg_iters=effective_lowpass_cg_iters,
        )
        x = y - f0

        intermediates = [x.clone()] if return_intermediate else None

        # Precompute BTB * (A^-1 * y) once.
        z0 = _solve_A_system(y, self.a_coeff, solve_cg_iters=effective_solve_cg_iters, x0=y)
        d_base = _banded_apply_T(_banded_apply(z0, self.b_coeff), self.b_coeff)

        for k in range(self.num_layers):
            layer_idx = 0 if self.shared_params else k
            layer = self.layers[layer_idx]
            d_vec = self.compute_d_vec(d_base, layer_idx)
            x = layer(x, d_vec, self.a_coeff, self.b_coeff, cg_iters=effective_solve_cg_iters)

            if return_intermediate:
                intermediates.append(x.clone())

        # Compute baseline.
        residual = y - x
        f = apply_lowpass_filter(
            residual, self.a_coeff, self.b_coeff,
            iterations=self.lowpass_iterations,
            solve_cg_iters=effective_lowpass_cg_iters,
        )

        if squeeze_output:
            x = x.squeeze(0)
            f = f.squeeze(0)

        if return_intermediate:
            return x, f, intermediates
        return x, f

    def get_learned_params(self):
        """Return dictionary of current learned parameters."""
        params = {}
        for i, layer in enumerate(self.layers):
            prefix = f"layer_{i}_" if not self.shared_params else ""
            params[f"{prefix}lam0"] = layer.lam0.item()
            params[f"{prefix}lam1"] = layer.lam1.item()
            params[f"{prefix}lam2"] = layer.lam2.item()
            params[f"{prefix}r"] = layer.r.item()
            params[f"{prefix}step"] = layer.step.item()
        return params
