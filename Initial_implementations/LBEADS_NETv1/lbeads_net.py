"""
LBEADS-NET: Learnable BEADS Network (Unrolled Model)

This module implements an unrolled version of the BEADS algorithm where
each iteration becomes a trainable layer with learnable parameters.

The original BEADS algorithm iteratively solves:
    x = A * ((BTB + A'*M*A) \ d)
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
from scipy import sparse
from scipy.sparse.linalg import spsolve


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
    b1 = np.array([1, -1])
    for i in range(d - 1):
        b1 = np.convolve(b1, np.array([-1, 2, -1]))
    b = np.convolve(b1, np.array([-1, 1]))
    
    omc = 2 * np.pi * fc
    t = ((1 - np.cos(omc)) / (1 + np.cos(omc))) ** d
    
    a = np.array([1])
    for i in range(d):
        a = np.convolve(a, np.array([1, 2, 1]))
    a = b + t * a
    
    diagonals_A = []
    diagonals_B = []
    offsets = list(range(-d, d + 1))
    
    for k, offset in enumerate(offsets):
        if offset <= 0:
            diag_len = N + offset
        else:
            diag_len = N - offset
        diagonals_A.append(np.full(diag_len, a[k]))
        diagonals_B.append(np.full(diag_len, b[k]))
    
    A = sparse.diags(diagonals_A, offsets, shape=(N, N), format='csc')
    B = sparse.diags(diagonals_B, offsets, shape=(N, N), format='csc')
    
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
                 init_lam0=0.4, init_lam1=4.0, init_lam2=3.2,
                 init_r=6.0, learn_r=False,
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
            EPS0, EPS1: Smoothing parameters
        """
        super(BEADSLayer, self).__init__()
        
        self.N = N
        self.d = d
        self.fc = fc
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
        
        # Pre-compute fixed matrices (not learnable)
        self._precompute_matrices()
    
    def _precompute_matrices(self):
        """Pre-compute filter and difference matrices."""
        N = self.N
        
        # Filter matrices
        A, B = BAfilt(self.d, self.fc, N)
        
        # Difference matrices
        D1, D2, D = build_difference_matrices(N)
        
        # Store as scipy sparse (for efficient solving)
        self.A_sp = A
        self.B_sp = B
        self.D_sp = D
        self.BTB_sp = B.T @ B
        
        # Also store dense versions for gradient computation
        # (This is a trade-off between memory and differentiability)
        self.register_buffer('A_dense', torch.tensor(A.toarray(), dtype=torch.float64))
        self.register_buffer('B_dense', torch.tensor(B.toarray(), dtype=torch.float64))
        self.register_buffer('D_dense', torch.tensor(D.toarray(), dtype=torch.float64))
        self.register_buffer('BTB_dense', torch.tensor(self.BTB_sp.toarray(), dtype=torch.float64))
    
    @property
    def lam0(self):
        return torch.exp(self.log_lam0)
    
    @property
    def lam1(self):
        return torch.exp(self.log_lam1)
    
    @property
    def lam2(self):
        return torch.exp(self.log_lam2)
    
    @property
    def r(self):
        return torch.exp(self.log_r)
    
    def wfun(self, x):
        """Penalty weight function (L1_v2)."""
        return 1.0 / (torch.abs(x) + self.EPS1)
    
    def forward(self, x, y, d_vec):
        """
        Forward pass: one BEADS iteration.
        
        Args:
            x: Current estimate (N,) or (batch, N)
            y: Original noisy signal (N,) or (batch, N)
            d_vec: Pre-computed vector BTB @ (A^-1 @ y) - lam0 * A^T @ b
            
        Returns:
            x_new: Updated estimate
        """
        N = self.N
        lam0, lam1, lam2, r = self.lam0, self.lam1, self.lam2, self.r
        
        # Handle batch dimension
        if x.dim() == 1:
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)
            d_vec = d_vec.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size = x.shape[0]
        x_new_list = []
        
        for b in range(batch_size):
            x_b = x[b]
            d_vec_b = d_vec[b]
            
            # Compute D @ x
            Dx = self.D_dense @ x_b
            
            # Compute lambda weights
            w = torch.cat([lam1 * torch.ones(N - 1, dtype=torch.float64, device=x.device),
                           lam2 * torch.ones(N - 2, dtype=torch.float64, device=x.device)])
            lambda_diag = w * self.wfun(Dx)
            
            # Compute gamma weights
            gamma = torch.ones(N, dtype=torch.float64, device=x.device)
            k = torch.abs(x_b) > self.EPS0
            gamma[~k] = ((1 + r) / 4) / abs(self.EPS0)
            gamma[k] = ((1 + r) / 4) / torch.abs(x_b[k])
            
            # Build M matrix: M = 2 * lam0 * Gamma + D' * Lambda * D
            Gamma = torch.diag(gamma)
            Lambda = torch.diag(lambda_diag)
            M = 2 * lam0 * Gamma + self.D_dense.T @ Lambda @ self.D_dense
            
            # Compute A' * M * A
            AMA = self.A_dense.T @ M @ self.A_dense
            
            # Solve: (BTB + A'*M*A) * z = d_vec
            lhs = self.BTB_dense + AMA
            z = torch.linalg.solve(lhs, d_vec_b)
            
            # x = A * z
            x_new = self.A_dense @ z
            x_new_list.append(x_new)
        
        x_new = torch.stack(x_new_list, dim=0)
        
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
                 shared_params=False,
                 init_lam0=0.4, init_lam1=4.0, init_lam2=3.2,
                 init_r=6.0, learn_r=False):
        super(LBEADS_NET, self).__init__()
        
        self.N = N
        self.d = d
        self.fc = fc
        self.num_layers = num_layers
        self.shared_params = shared_params
        
        # Pre-compute filter matrices (shared across all layers)
        A, B = BAfilt(d, fc, N)
        self.A_sp = A
        self.B_sp = B
        self.BTB_sp = B.T @ B
        
        self.register_buffer('A_dense', torch.tensor(A.toarray(), dtype=torch.float64))
        self.register_buffer('B_dense', torch.tensor(B.toarray(), dtype=torch.float64))
        
        # Create layers
        if shared_params:
            # Single layer with shared parameters
            self.layers = nn.ModuleList([
                BEADSLayer(N, d, fc, init_lam0, init_lam1, init_lam2, init_r, learn_r)
            ])
        else:
            # Layer-wise learnable parameters
            self.layers = nn.ModuleList([
                BEADSLayer(N, d, fc, init_lam0, init_lam1, init_lam2, init_r, learn_r)
                for _ in range(num_layers)
            ])
    
    def compute_d_vec(self, y, layer_idx=0):
        """
        Compute the constant vector d for the linear system.
        
        d = BTB * (A^-1 * y) - lam0 * A^T * b
        
        Args:
            y: Input signal
            layer_idx: Which layer's parameters to use
        """
        layer = self.layers[layer_idx] if not self.shared_params else self.layers[0]
        r = layer.r
        lam0 = layer.lam0
        
        # Solve A * z = y using scipy sparse solver (not differentiable part)
        y_np = y.detach().cpu().numpy()
        z = spsolve(self.A_sp, y_np)
        z = torch.tensor(z, dtype=torch.float64, device=y.device)
        
        # b = (1 - r) / 2 * ones(N)
        b_vec = (1 - r) / 2 * torch.ones(self.N, dtype=torch.float64, device=y.device)
        
        # d = BTB @ z - lam0 * A^T @ b
        BTB_z = self.layers[0].BTB_dense @ z
        AT_b = self.A_dense.T @ b_vec
        d_vec = BTB_z - lam0 * AT_b
        
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
        
        batch_size = y.shape[0]
        
        # Initialize x = y
        x = y.clone()
        
        intermediates = [x.clone()] if return_intermediate else None
        
        # Run through unrolled layers
        for k in range(self.num_layers):
            layer_idx = 0 if self.shared_params else k
            layer = self.layers[layer_idx]
            
            # Compute d_vec for each sample in batch
            d_vec_list = []
            for b in range(batch_size):
                d_vec_b = self.compute_d_vec(y[b], layer_idx)
                d_vec_list.append(d_vec_b)
            d_vec = torch.stack(d_vec_list, dim=0)
            
            # Apply layer
            x = layer(x, y, d_vec)
            
            if return_intermediate:
                intermediates.append(x.clone())
        
        # Compute baseline: f = y - x - H(y - x)
        # H(z) = B * (A^-1 * z)
        f_list = []
        for b in range(batch_size):
            residual = y[b] - x[b]
            residual_np = residual.detach().cpu().numpy()
            z = spsolve(self.A_sp, residual_np)
            H_res = self.B_sp @ z
            f_b = residual - torch.tensor(H_res, dtype=torch.float64, device=y.device)
            f_list.append(f_b)
        f = torch.stack(f_list, dim=0)
        
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
        return params


class LBEADS_NET_Fast(nn.Module):
    """
    A fast, fully vectorized version of LBEADS-NET.
    
    Uses 1D convolutions instead of dense matrix multiplications for efficiency.
    All operations are batched and vectorized for speed.
    """
    
    def __init__(self, N, d=1, fc=0.006, num_layers=10,
                 init_lam0=0.4, init_lam1=4.0, init_lam2=3.2,
                 init_r=6.0, init_step_size=0.1):
        super(LBEADS_NET_Fast, self).__init__()
        
        self.N = N
        self.d = d
        self.fc = fc
        self.num_layers = num_layers
        self.EPS0 = 1e-6
        self.EPS1 = 1e-6
        
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
    
    def forward(self, y, return_intermediate=False):
        """
        Forward pass using proximal gradient descent style updates.
        Fully vectorized for speed.
        """
        if y.dim() == 1:
            y = y.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        x = y.clone()
        intermediates = [x.clone()] if return_intermediate else None
        
        for k in range(self.num_layers):
            lam0 = torch.exp(self.log_lam0[k])
            lam1 = torch.exp(self.log_lam1[k])
            lam2 = torch.exp(self.log_lam2[k])
            r = torch.exp(self.log_r[k])
            step_size = torch.exp(self.log_step_size[k])
            
            # Simple data fidelity gradient: just (x - y)
            # This is a simplification - the full BEADS uses filtered version
            grad_data = x - y
            
            # Asymmetric penalty gradient (vectorized over batch)
            grad_asym = lam0 * self.asymmetric_penalty_grad(x, r)
            
            # First derivative penalty gradient (L1): D1.T @ (D1 @ x / |D1 @ x|)
            Dx1 = self.diff1(x)  # (batch, N-1)
            w1 = Dx1 / (torch.abs(Dx1) + self.EPS1)
            grad_D1 = lam1 * self.diff1_T(w1)  # (batch, N)
            
            # Second derivative penalty gradient (L1): D2.T @ (D2 @ x / |D2 @ x|)
            Dx2 = self.diff2(x)  # (batch, N-2)
            w2 = Dx2 / (torch.abs(Dx2) + self.EPS1)
            grad_D2 = lam2 * self.diff2_T(w2)  # (batch, N)
            
            # Total gradient and update
            grad = grad_data + grad_asym + grad_D1 + grad_D2
            x = x - step_size * grad
            
            if return_intermediate:
                intermediates.append(x.clone())
        
        # Compute baseline using simple smoothing
        # f = smooth(y - x), here we use a simple approximation
        residual = y - x
        # Simple low-pass approximation for baseline
        f = residual - self.diff2_T(self.diff2(residual)) * 0.1
        
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
