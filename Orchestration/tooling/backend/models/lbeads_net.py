"""
LBEADS-NET: Learnable BEADS Network (Unrolled Model)

This module implements an unrolled version of the BEADS algorithm where
each iteration becomes a trainable layer with learnable parameters.

The original BEADS algorithm iteratively solves:
    x = A * inv(BTB + A'*M*A) * d
where M depends on regularization parameters (lam0, lam1, lam2) and 
penalty weights that are recomputed each iteration.

In LBEADS-NET, we unroll this into K layers where:
- Each layer can have its own learnable regularization parameters
- The penalty function parameters can also be learned
- Optionally, additional learnable transformations can be added

Reference:
Original BEADS: Chromatogram baseline estimation and denoising using sparsity
Xiaoran Ning, Ivan W. Selesnick, Laurent Duval
Chemometrics and Intelligent Laboratory Systems (2014)
"""

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from scipy import sparse
from scipy.sparse.linalg import spsolve, factorized


def _compute_filter_coefficients(d: int, fc: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return symmetric FIR coefficients for BEADS banded filters A and B."""
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


def BAfilt(d, fc, N):
    """
    Banded matrices for zero-phase high-pass filter.
    
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


def _banded_apply(x: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
    """
    Apply a constant-diagonal banded operator to batched row vectors.

    Args:
        x: (N,) or (batch, N)
        coeffs: (2*d+1,) with offsets [-d, ..., d]
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


def _cg_solve_fixed(matvec, b: torch.Tensor, max_iter: int = 16, eps: float = 1e-12, x0: Optional[torch.Tensor] = None) -> torch.Tensor:
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


def _solve_A_system(rhs: torch.Tensor, a_coeff: torch.Tensor, solve_cg_iters: int = 16,
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


def _highpass_once_torch(signal: torch.Tensor, a_coeff: torch.Tensor, b_coeff: torch.Tensor, solve_cg_iters: int) -> torch.Tensor:
    """High-pass operator H(v) = B @ (A^{-1} v) via exact/iterative A-solve."""
    z = _solve_A_system(signal, a_coeff, solve_cg_iters=solve_cg_iters, x0=signal)
    return _banded_apply(z, b_coeff)


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


def compute_lowpass_matrix(A_dense, B_dense):
    """
    Legacy API kept for compatibility.

    WARNING: This returns a dense operator and is memory-heavy for large N.
    Prefer `apply_lowpass_filter(..., a_coeff, b_coeff, ...)` operator mode.
    """
    n = A_dense.shape[0]
    eye = torch.eye(n, dtype=A_dense.dtype, device=A_dense.device)
    A_inv = torch.linalg.solve(A_dense, eye)
    return eye - (B_dense @ A_inv)


def compute_iterated_lowpass_matrix(A_dense, B_dense, iterations=3):
    """
    Legacy dense low-pass power operator (compatibility only).
    """
    L = compute_lowpass_matrix(A_dense, B_dense)
    L_iter = L.clone()
    for _ in range(iterations - 1):
        L_iter = L_iter @ L
    return L_iter


def apply_lowpass_filter(residual, lowpass_or_a_coeff, b_coeff: Optional[torch.Tensor] = None,
                         iterations: int = 1, solve_cg_iters: int = 16):
    """
    Apply low-pass filtering.

    Two modes:
    1) Legacy dense mode: `apply_lowpass_filter(residual, lowpass_dense)`
    2) Operator mode: `apply_lowpass_filter(residual, a_coeff, b_coeff, iterations, solve_cg_iters)`
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
    1) Legacy dense mode: `apply_highpass_filter(signal, lowpass_dense)`
    2) Operator mode: `apply_highpass_filter(signal, a_coeff, b_coeff, solve_cg_iters)`
    """
    if b_coeff is None:
        low = apply_lowpass_filter(signal, lowpass_or_a_coeff)
        return signal - low
    return _highpass_once_torch(signal, lowpass_or_a_coeff, b_coeff, solve_cg_iters)


class BEADSLayer(nn.Module):
    """
    A single unrolled BEADS iteration layer.
    
    This layer performs one iteration of the BEADS algorithm with
    learnable regularization parameters.
    
    Parameters:
        lam0: Regularization for asymmetric penalty (sparsity)
        lam1: Regularization for first derivative penalty
        lam2: Regularization for second derivative penalty
        r: Asymmetry ratio (can be learnable or fixed)
    """
    
    def __init__(self, N, d, fc,
                 init_lam0=0.001, init_lam1=0.2, init_lam2=0.2,
                 init_r=6.0, learn_r=True,
                 init_step=1.0, learn_step=True,
                 EPS0=1e-6, EPS1=1e-6):
        """
        Initialize a BEADS layer.
        
        Args:
            N: Signal length
            d: Filter order (1 or 2)
            fc: Filter cut-off frequency
            init_lam0, init_lam1, init_lam2: Initial regularization parameters
            init_r: Initial asymmetry ratio
            learn_r: Whether to make r learnable
            init_step: Initial layer relaxation step
            learn_step: Whether to learn the layer relaxation step
            EPS0, EPS1: Smoothing parameters
        """
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
        
        # Asymmetry ratio (optionally learnable)
        if learn_r:
            self.log_r = nn.Parameter(torch.tensor(np.log(init_r), dtype=torch.float64))
        else:
            self.register_buffer('log_r', torch.tensor(np.log(init_r), dtype=torch.float64))

        # Layer relaxation step for iterative dynamics.
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
        Forward pass: one memory-safe BEADS iteration (matrix-free CG).
        
        Args:
            x: Current estimate (N,) or (batch, N)
            d_vec: Pre-computed vector BTB @ (A^-1 @ y) - lam0 * A^T @ b
            a_coeff: Banded coefficients for A
            b_coeff: Banded coefficients for B
            cg_iters: CG iterations for linear solve
            
        Returns:
            x_new: Updated estimate
        """
        lam0, lam1, lam2, r = self.lam0, self.lam1, self.lam2, self.r
        
        # Handle batch dimension
        if x.dim() == 1:
            x = x.unsqueeze(0)
            d_vec = d_vec.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        # Weights are fixed for this iteration (majorization-minimization).
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

        # Warm-start in z-domain: x = A z => z0 = A^{-1} x.
        z_init = _solve_A_system(x.detach(), a_coeff, solve_cg_iters=cg_iters, x0=x.detach())
        z = _cg_solve_fixed(lhs_matvec, d_vec, max_iter=cg_iters, x0=z_init)
        x_new = _banded_apply(z, a_coeff)
        x_new = x + self.step * (x_new - x)
        
        if squeeze_output:
            x_new = x_new.squeeze(0)
        
        return x_new


class LBEADS_NET(nn.Module):
    """
    LBEADS-NET: Learnable BEADS Network
    
    An unrolled neural network version of the BEADS algorithm.
    Each iteration of BEADS becomes a layer with learnable parameters.
    
    Two modes of operation:
    1. Shared parameters: All layers share the same parameters (like original BEADS)
    2. Layer-wise parameters: Each layer has its own learnable parameters
    
    Args:
        N: Signal length
        d: Filter order (1 or 2)
        fc: Filter cut-off frequency
        num_layers: Number of unrolled iterations (K)
        shared_params: If True, all layers share parameters
        init_lam0, init_lam1, init_lam2: Initial regularization parameters
        init_r: Initial asymmetry ratio
        learn_r: Whether to make r learnable
        learn_fc: Whether to make fc learnable (advanced)
    """
    
    def __init__(self, N, d=1, fc=0.006, num_layers=10,
                 shared_params=True,
                 init_lam0=0.001, init_lam1=0.2, init_lam2=0.2,
                 init_r=6.0, learn_r=True,
                 init_step=1.0, learn_step=True,
                 init_output_gain=1.0, learn_output_gain=False,
                 lowpass_iterations=1,
                 solve_cg_iters=16,
                 lowpass_cg_iters=32):
        super(LBEADS_NET, self).__init__()
        
        self.N = N
        self.d = d
        self.fc = fc
        self.num_layers = num_layers
        self.shared_params = shared_params
        self.lowpass_iterations = int(lowpass_iterations)
        self.solve_cg_iters = int(solve_cg_iters)
        self.lowpass_cg_iters = int(lowpass_cg_iters)
        self.learn_output_gain = bool(learn_output_gain)

        # Shared banded coefficients (single copy regardless of layer count).
        a_np, b_np = _compute_filter_coefficients(d, fc)
        self.register_buffer('a_coeff', torch.tensor(a_np, dtype=torch.float64))
        self.register_buffer('b_coeff', torch.tensor(b_np, dtype=torch.float64))

        # Keep sparse matrices only for NumPy-side utilities.
        A, B = BAfilt(d, fc, N)
        self.A_sp = A
        self.B_sp = B

        ones = torch.ones(1, N, dtype=torch.float64)
        self.register_buffer('AT_ones', _banded_apply_T(ones, self.a_coeff).squeeze(0), persistent=False)
        
        # Create layers
        if shared_params:
            # Single layer with shared parameters
            self.layers = nn.ModuleList([
                BEADSLayer(
                    N, d, fc,
                    init_lam0, init_lam1, init_lam2,
                    init_r, learn_r,
                    init_step, learn_step,
                )
            ])
        else:
            # Layer-wise learnable parameters
            self.layers = nn.ModuleList([
                BEADSLayer(
                    N, d, fc,
                    init_lam0, init_lam1, init_lam2,
                    init_r, learn_r,
                    init_step, learn_step,
                )
                for _ in range(num_layers)
            ])

        if learn_output_gain:
            self.log_output_gain = nn.Parameter(torch.tensor(np.log(init_output_gain), dtype=torch.float64))
        else:
            self.register_buffer('log_output_gain', torch.tensor(np.log(init_output_gain), dtype=torch.float64))

    @property
    def output_gain(self):
        return torch.clamp(torch.exp(self.log_output_gain), min=1e-1, max=10.0)
    
    def compute_d_vec(self, d_base, layer_idx=0):
        """
        Compute layer-specific d for:
            d = BTB * (A^-1 * y) - lam0 * A^T * b
        
        Args:
            d_base: Precomputed BTB * (A^-1 * y), shape (N,) or (batch, N)
            layer_idx: Which layer's parameters to use
        """
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
        """
        Forward pass through all unrolled layers.
        
        Args:
            y: Noisy input signal (N,) or (batch, N)
            return_intermediate: If True, return estimates from all layers
            
        Returns:
            x: Estimated sparse-derivative signal
            f: Estimated baseline
            intermediates: (optional) List of intermediate estimates
        """
        # Handle batch dimension
        if y.dim() == 1:
            y = y.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        # Use tighter linear solves at inference for better numerical stability.
        effective_solve_cg_iters = self.solve_cg_iters if self.training else max(self.solve_cg_iters, 128)
        effective_lowpass_cg_iters = self.lowpass_cg_iters if self.training else max(self.lowpass_cg_iters, 128)
        
        # Initialize peaks with a high-pass residual estimate: x0 = y - lowpass(y).
        f0 = apply_lowpass_filter(
            y,
            self.a_coeff,
            self.b_coeff,
            iterations=self.lowpass_iterations,
            solve_cg_iters=effective_lowpass_cg_iters,
        )
        x = y - f0
        
        intermediates = [x.clone()] if return_intermediate else None

        # Precompute BTB * (A^-1 * y) once per sample; independent of layer params.
        z0 = _solve_A_system(y, self.a_coeff, solve_cg_iters=effective_solve_cg_iters, x0=y)
        d_base = _banded_apply_T(_banded_apply(z0, self.b_coeff), self.b_coeff)
        
        # Run through unrolled layers
        for k in range(self.num_layers):
            layer_idx = 0 if self.shared_params else k
            layer = self.layers[layer_idx]

            # Layer-specific correction term depends on (lam0, r) only.
            d_vec = self.compute_d_vec(d_base, layer_idx)

            # Matrix-free BEADS update.
            x = layer(x, d_vec, self.a_coeff, self.b_coeff, cg_iters=effective_solve_cg_iters)
            
            if return_intermediate:
                intermediates.append(x.clone())

        x = self.output_gain * x
        
        # Compute baseline in operator form (no dense NxN low-pass matrix).
        residual = y - x
        f = apply_lowpass_filter(
            residual,
            self.a_coeff,
            self.b_coeff,
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
        params["output_gain"] = self.output_gain.item()
        return params

    def load_state_dict(self, state_dict, strict: bool = True):
        """
        Backward-compatible loading from legacy checkpoints with dense buffers.
        """
        remap = dict(state_dict)
        for key in ("A_dense", "B_dense", "lowpass_dense"):
            remap.pop(key, None)

        legacy_missing_coeffs = ("a_coeff" not in remap) or ("b_coeff" not in remap)
        legacy_missing_step = not any(k.endswith("log_step") for k in remap.keys())
        legacy_missing_output_gain = ("log_output_gain" not in remap)
        needs_relaxed = legacy_missing_coeffs or legacy_missing_step or legacy_missing_output_gain
        return super().load_state_dict(remap, strict=False if needs_relaxed else strict)


class LBEADS_NET_Fast(nn.Module):
    """
    A fast, trainable version of LBEADS-NET.
    
    This version uses a proximal gradient approach that's faithful to BEADS
    while being fully differentiable and fast enough to train.
    
    Key insight: Instead of solving the full linear system each iteration,
    we use a learnable proximal operator that approximates the BEADS update.
    """
    
    def __init__(self, N, d=1, fc=0.006, num_layers=10,
                 init_lam0=0.4, init_lam1=4.0, init_lam2=3.2,
                 init_r=6.0, init_step_size=0.1,
                 lowpass_iterations=3,
                 lowpass_cg_iters=12):
        super(LBEADS_NET_Fast, self).__init__()
        
        self.N = N
        self.d = d
        self.fc = fc
        self.num_layers = num_layers
        self.lowpass_iterations = int(lowpass_iterations)
        self.lowpass_cg_iters = int(lowpass_cg_iters)
        self.EPS0 = 1e-6
        self.EPS1 = 1e-6
        
        # Sparse matrices are kept for non-differentiable NumPy-side utilities only.
        A, B = BAfilt(d, fc, N)
        self.A_sp = A
        self.B_sp = B

        # Store only compact banded coefficients (O(d) memory, not O(N^2)).
        a_np, b_np = _compute_filter_coefficients(d, fc)
        self.register_buffer('a_coeff', torch.tensor(a_np, dtype=torch.float64))
        self.register_buffer('b_coeff', torch.tensor(b_np, dtype=torch.float64))
        
        # Layer-wise learnable parameters
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
        # Learnable step sizes for proximal updates
        self.log_step_size = nn.ParameterList([
            nn.Parameter(torch.tensor(np.log(init_step_size), dtype=torch.float64))
            for _ in range(num_layers)
        ])
    
    def diff1(self, x):
        """First difference: D1 @ x (vectorized)."""
        # x: (batch, N) -> (batch, N-1)
        return x[:, 1:] - x[:, :-1]
    
    def diff2(self, x):
        """Second difference: D2 @ x (vectorized)."""
        # x: (batch, N) -> (batch, N-2)
        return x[:, 2:] - 2 * x[:, 1:-1] + x[:, :-2]
    
    def diff1_T(self, v):
        """Transpose of first difference: D1.T @ v (vectorized)."""
        # v: (batch, N-1) -> (batch, N)
        batch_size = v.shape[0]
        N = v.shape[1] + 1
        result = torch.zeros(batch_size, N, dtype=v.dtype, device=v.device)
        result[:, :-1] -= v
        result[:, 1:] += v
        return result
    
    def diff2_T(self, v):
        """Transpose of second difference: D2.T @ v (vectorized)."""
        # v: (batch, N-2) -> (batch, N)
        batch_size = v.shape[0]
        N = v.shape[1] + 2
        result = torch.zeros(batch_size, N, dtype=v.dtype, device=v.device)
        result[:, :-2] += v
        result[:, 1:-1] -= 2 * v
        result[:, 2:] += v
        return result
    
    def asymmetric_penalty_grad(self, x, r):
        """Gradient of the asymmetric penalty function theta(x) - vectorized."""
        EPS0 = self.EPS0
        grad = torch.zeros_like(x)
        
        pos_mask = x > EPS0
        neg_mask = x < -EPS0
        mid_mask = ~pos_mask & ~neg_mask
        
        grad[pos_mask] = 1.0
        grad[neg_mask] = -r
        grad[mid_mask] = (1 + r) / (2 * EPS0) * x[mid_mask] + (1 - r) / 2
        
        return grad
    
    def soft_threshold(self, x, lam):
        """Soft thresholding (proximal operator for L1)."""
        return torch.sign(x) * torch.clamp(torch.abs(x) - lam, min=0)
    
    def asymmetric_soft_threshold(self, x, lam, r):
        """
        Asymmetric soft thresholding for BEADS.
        Penalizes negative values r times more than positive.
        """
        # For positive: threshold at lam
        # For negative: threshold at lam * r
        pos_thresh = lam
        neg_thresh = lam * r
        
        result = torch.zeros_like(x)
        pos_mask = x > pos_thresh
        neg_mask = x < -neg_thresh
        
        result[pos_mask] = x[pos_mask] - pos_thresh
        result[neg_mask] = x[neg_mask] + neg_thresh
        # Values between thresholds stay at 0 (sparse!)
        
        return result
    
    def forward(self, y, return_intermediate=False):
        """
        Forward pass using ISTA-style proximal gradient with asymmetric thresholding.
        
        This is a proper unrolled optimization that learns:
        - Per-layer regularization weights (lam0, lam1, lam2)
        - Per-layer step sizes
        - Per-layer asymmetry ratios
        
        Key insight: x represents the PEAKS (sparse, positive), not the full signal.
        We want to recover peaks from y = x + f + noise where f is smooth baseline.
        """
        if y.dim() == 1:
            y = y.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        # Use a tighter solve at inference for stability/quality on legacy checkpoints.
        effective_lowpass_cg_iters = self.lowpass_cg_iters if self.training else max(self.lowpass_cg_iters, 48)
        
        # Initialize x from high-pass filtered y: x0 = y - lowpass(y)
        # This gives a MUCH better starting point than zeros!
        # Without this, small peaks (< threshold) are permanently zeroed.
        f_init = apply_lowpass_filter(
            y,
            self.a_coeff,
            self.b_coeff,
            iterations=self.lowpass_iterations,
            solve_cg_iters=effective_lowpass_cg_iters,
        )
        x = y - f_init
        intermediates = [x.clone()] if return_intermediate else None
        
        for k in range(self.num_layers):
            lam0 = torch.exp(self.log_lam0[k])
            lam1 = torch.exp(self.log_lam1[k])
            lam2 = torch.exp(self.log_lam2[k])
            r = torch.exp(self.log_r[k])
            step_size = torch.exp(self.log_step_size[k])
            
            # ===== DATA FIDELITY GRADIENT =====
            # We model y = x + f + noise where:
            #   - x is sparse peaks (what we want)
            #   - f is smooth baseline  
            # The high-pass filter H extracts high-frequency content
            # So x should capture high-freq peaks, f captures low-freq baseline
            
            # Residual: what's left after removing current peak estimate
            residual = y - x
            
            # Use the high-pass residual for data fidelity so baseline (low-freq)
            # is not pulled into x during optimization updates.
            data_grad = apply_highpass_filter(
                residual,
                self.a_coeff,
                self.b_coeff,
                solve_cg_iters=effective_lowpass_cg_iters,
            )
            
            # ===== SMOOTHNESS PENALTY GRADIENTS =====
            # These penalize non-smooth variations in x (but peaks ARE non-smooth!)
            # So we want small weights on these for peak signals
            
            # D1 penalty gradient - penalizes first derivative
            Dx1 = self.diff1(x)
            w1 = Dx1 / (torch.abs(Dx1) + self.EPS1)
            grad_D1 = lam1 * self.diff1_T(w1)
            
            # D2 penalty gradient - penalizes second derivative  
            Dx2 = self.diff2(x)
            w2 = Dx2 / (torch.abs(Dx2) + self.EPS1)
            grad_D2 = lam2 * self.diff2_T(w2)
            
            # ===== GRADIENT DESCENT STEP =====
            # Move toward data while smoothing
            x_update = x + step_size * data_grad - step_size * (grad_D1 + grad_D2)
            
            # ===== PROXIMAL STEP (ASYMMETRIC SOFT THRESHOLDING) =====
            # This is the key for BEADS: it enforces sparsity with asymmetry
            # - Positive values (peaks) are thresholded lightly
            # - Negative values are thresholded more heavily (by factor r)
            # This encourages positive sparse peaks!
            x = self.asymmetric_soft_threshold(x_update, lam0 * step_size, r)
            
            if return_intermediate:
                intermediates.append(x.clone())
        
        # ===== COMPUTE BASELINE =====
        # f = y - x - H(y - x) where H is high-pass filter
        # This extracts the low-frequency (baseline) component
        residual = y - x
        f = apply_lowpass_filter(
            residual,
            self.a_coeff,
            self.b_coeff,
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
        for i in range(self.num_layers):
            params[f"layer_{i}_lam0"] = torch.exp(self.log_lam0[i]).item()
            params[f"layer_{i}_lam1"] = torch.exp(self.log_lam1[i]).item()
            params[f"layer_{i}_lam2"] = torch.exp(self.log_lam2[i]).item()
            params[f"layer_{i}_r"] = torch.exp(self.log_r[i]).item()
            params[f"layer_{i}_step_size"] = torch.exp(self.log_step_size[i]).item()
        return params

    def load_state_dict(self, state_dict, strict: bool = True):
        """
        Backward-compatible loading from legacy checkpoints with dense buffers.
        """
        remap = dict(state_dict)
        for key in ("A_dense", "B_dense", "BTB_dense", "lowpass_dense"):
            remap.pop(key, None)

        legacy_missing_coeffs = ("a_coeff" not in remap) or ("b_coeff" not in remap)
        return super().load_state_dict(remap, strict=False if legacy_missing_coeffs else strict)


@dataclass
class HybridConfig:
    """Configuration for hybrid LBEADS + classical BEADS inference."""
    noise_k: float = 2.5
    lowpass_iterations: int = 3
    short_refine_iterations: int = 8
    full_refine_iterations: int = 24
    run_full_refine_score_threshold: float = 0.35
    max_active_fraction_for_short_only: float = 0.20
    min_peak_threshold: float = 0.005
    baseline_hf_refine_threshold: float = 0.10
    baseline_hf_full_refine_threshold: float = 0.14


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


def _median_learned_param(params: Dict[str, float], suffix: str, default: float) -> float:
    vals = [float(v) for k, v in params.items() if k.endswith(suffix)]
    if not vals and suffix in params:
        vals = [float(params[suffix])]
    if not vals:
        return float(default)
    return float(np.median(np.asarray(vals, dtype=np.float64)))


def _extract_regularization_from_model(model) -> Dict[str, float]:
    """Extract stable BEADS regularization values from learned layer parameters."""
    if hasattr(model, "get_learned_params"):
        params = model.get_learned_params()
    else:
        params = {}

    lam0 = _median_learned_param(params, "lam0", 0.002)
    lam1 = _median_learned_param(params, "lam1", 0.2)
    lam2 = _median_learned_param(params, "lam2", 0.2)
    r = _median_learned_param(params, "r", 6.0)

    # Conservative clipping for normalized signals.
    lam0 = float(np.clip(lam0, 1e-4, 5e-2))
    lam1 = float(np.clip(lam1, 1e-4, 1.0))
    lam2 = float(np.clip(lam2, 1e-4, 1.0))
    r = float(np.clip(r, 2.0, 12.0))

    d = int(getattr(model, "d", 1))
    fc = float(getattr(model, "fc", 0.006))

    return {"lam0": lam0, "lam1": lam1, "lam2": lam2, "r": r, "d": d, "fc": fc}


def _quality_metrics(y, x, f, lowpass_matrix, min_peak_threshold=0.005):
    """
    Compute no-reference quality metrics for chromatogram decomposition.

    Lower is better for the composite `score`.
    """
    y = np.asarray(y, dtype=np.float64)
    x = np.maximum(np.asarray(x, dtype=np.float64), 0.0)
    f = np.asarray(f, dtype=np.float64)

    residual = y - x - f
    residual_hf = apply_highpass_filter_np(residual, lowpass_matrix)
    residual_hf_rms = float(np.sqrt(np.mean(residual_hf ** 2)))

    baseline_hf = apply_highpass_filter_np(f, lowpass_matrix)
    baseline_rms = float(np.sqrt(np.mean(f ** 2)))
    baseline_hf_ratio = float(np.sqrt(np.mean(baseline_hf ** 2)) / (baseline_rms + 1e-8))

    noise_proxy = apply_highpass_filter_np(y - x, lowpass_matrix)
    sigma = float(np.median(np.abs(noise_proxy)) / 0.6745 + 1e-8)
    active_threshold = max(3.0 * sigma, float(min_peak_threshold))
    active_fraction = float(np.mean(x > active_threshold))

    # Composite heuristic score (lower is better).
    score = active_fraction + 2.0 * baseline_hf_ratio + 0.5 * residual_hf_rms
    return {
        "score": float(score),
        "active_fraction": active_fraction,
        "baseline_hf_ratio": baseline_hf_ratio,
        "residual_hf_rms": residual_hf_rms,
        "sigma": sigma,
        "active_threshold": float(active_threshold),
    }


def hybrid_infer_1d(model, y, config: Optional[HybridConfig] = None):
    """
    Hybrid inference:
      1) LBEADS forward pass (learned)
      2) Adaptive peak denoise + low-pass baseline post-processing
      3) Short classical BEADS refinement initialized from learned output
      4) Optional full classical fallback if quality remains poor

    Returns a dictionary containing raw LBEADS output, post-processed output,
    refined outputs, selected final output, and diagnostics.
    """
    if config is None:
        config = HybridConfig()

    y = np.asarray(y, dtype=np.float64).flatten()
    N_orig = len(y)
    N_model = int(getattr(model, "N", N_orig))
    if N_orig > N_model:
        raise ValueError(f"Signal length {N_orig} exceeds model length {N_model}")

    pad = N_model - N_orig
    if pad > 0:
        y_padded = np.pad(y, (0, pad), mode="reflect")
    else:
        y_padded = y.copy()

    scale = float(max(np.max(np.abs(y_padded)), 1e-8))
    y_norm = y_padded / scale

    try:
        model_param = next(model.parameters())
        device = model_param.device
        dtype = model_param.dtype
    except StopIteration:
        device = torch.device("cpu")
        dtype = torch.float64

    model.eval()
    with torch.no_grad():
        y_tensor = torch.tensor(y_norm, dtype=dtype, device=device).unsqueeze(0)
        x_pred, f_pred = model(y_tensor)
    x_raw = x_pred[0].detach().cpu().numpy().astype(np.float64)
    f_raw = f_pred[0].detach().cpu().numpy().astype(np.float64)

    reg = _extract_regularization_from_model(model)
    lowpass_matrix = compute_lowpass_matrix_np(N_model, d=reg["d"], fc=reg["fc"])

    # Candidate A: post-processed learned output.
    # Estimate sigma from quiet regions so peaks do not inflate the noise level.
    x_raw_pos = np.maximum(x_raw, 0.0)
    residual_raw = y_norm - x_raw_pos
    noise_hp = apply_highpass_filter_np(residual_raw, lowpass_matrix)
    x_median = float(np.median(x_raw_pos))
    x_mad = float(np.median(np.abs(x_raw_pos - x_median)) + 1e-12)
    quiet_threshold = float(config.min_peak_threshold + 0.5 * x_mad)
    quiet_mask = x_raw_pos < quiet_threshold
    quiet_noise_hp = noise_hp[quiet_mask]
    if quiet_noise_hp.size < 16:
        quiet_noise_hp = noise_hp
    sigma = float(np.median(np.abs(quiet_noise_hp)) / 0.6745 + 1e-8)
    x_post = np.maximum(x_raw - config.noise_k * sigma, 0.0)
    f_post = apply_lowpass_filter_np(y_norm - x_post, lowpass_matrix, iterations=config.lowpass_iterations)
    q_post = _quality_metrics(y_norm, x_post, f_post, lowpass_matrix, config.min_peak_threshold)

    # Candidate B: short BEADS refine from warm start.
    nit_short = int(config.short_refine_iterations)
    if q_post["active_fraction"] > config.max_active_fraction_for_short_only:
        nit_short += 4
    if q_post["baseline_hf_ratio"] > config.baseline_hf_refine_threshold:
        nit_short += 6
    x_ref, f_ref = beads_classic_with_init(
        y_norm,
        d=reg["d"],
        fc=reg["fc"],
        r=reg["r"],
        lam0=reg["lam0"],
        lam1=reg["lam1"],
        lam2=reg["lam2"],
        Nit=nit_short,
        x_init=x_post,
    )
    x_ref = np.maximum(x_ref, 0.0)
    f_ref = apply_lowpass_filter_np(y_norm - x_ref, lowpass_matrix, iterations=1)
    q_ref = _quality_metrics(y_norm, x_ref, f_ref, lowpass_matrix, config.min_peak_threshold)

    candidates = {
        "post": (x_post, f_post, q_post),
        "short_refine": (x_ref, f_ref, q_ref),
    }

    # Candidate C (optional): full fallback from scratch if quality is still poor.
    force_full_refine = (
        q_post["baseline_hf_ratio"] > config.baseline_hf_full_refine_threshold
        or q_ref["baseline_hf_ratio"] > config.baseline_hf_refine_threshold
    )
    if force_full_refine or (q_post["score"] > config.run_full_refine_score_threshold) or (q_ref["score"] > q_post["score"] * 1.10):
        full_lam0 = float(np.clip(reg["lam0"] * 1.15, 1e-4, 5e-2))
        full_lam1 = float(np.clip(reg["lam1"] * 1.10, 1e-4, 1.0))
        full_lam2 = float(np.clip(reg["lam2"] * 1.10, 1e-4, 1.0))
        x_full, f_full = beads_classic_with_init(
            y_norm,
            d=reg["d"],
            fc=reg["fc"],
            r=reg["r"],
            lam0=full_lam0,
            lam1=full_lam1,
            lam2=full_lam2,
            Nit=int(config.full_refine_iterations),
            x_init=None,
        )
        x_full = np.maximum(x_full, 0.0)
        f_full = apply_lowpass_filter_np(y_norm - x_full, lowpass_matrix, iterations=1)
        q_full = _quality_metrics(y_norm, x_full, f_full, lowpass_matrix, config.min_peak_threshold)
        candidates["full_refine"] = (x_full, f_full, q_full)

    # Select best candidate by quality score.
    best_stage = min(candidates.keys(), key=lambda k: candidates[k][2]["score"])
    x_best, f_best, q_best = candidates[best_stage]

    def crop_and_scale(arr):
        return arr[:N_orig] * scale

    return {
        "x_lbeads": crop_and_scale(np.maximum(x_raw, 0.0)),
        "f_lbeads": crop_and_scale(f_raw),
        "x_post": crop_and_scale(x_post),
        "f_post": crop_and_scale(f_post),
        "x_refine": crop_and_scale(x_ref),
        "f_refine": crop_and_scale(f_ref),
        "x_hybrid": crop_and_scale(x_best),
        "f_hybrid": crop_and_scale(f_best),
        "selected_stage": best_stage,
        "quality": {name: cand[2] for name, cand in candidates.items()},
        "noise_sigma_normalized": sigma,
        "scale": scale,
        "regularization": reg,
        "config": config,
        "quality_selected": q_best,
    }
