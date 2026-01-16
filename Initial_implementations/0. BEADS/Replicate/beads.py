"""
BEADS: Baseline Estimation And Denoising with Sparsity

Reference:
Chromatogram baseline estimation and denoising using sparsity (BEADS)
Xiaoran Ning, Ivan W. Selesnick, Laurent Duval
Chemometrics and Intelligent Laboratory Systems (2014)
doi: 10.1016/j.chemolab.2014.09.014
"""

import torch
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
    # b1 = [1, -1]
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
    
    # Build sparse banded matrices A and B using scipy.sparse.spdiags
    # a and b have length 2*d + 1, diagonals from -d to d
    # spdiags in MATLAB: A = spdiags(a(ones(N,1), :), -d:d, N, N)
    # In scipy: sparse.diags expects diagonals in order
    
    # Create diagonal arrays for A and B
    # Each row of 'diagonals' corresponds to one diagonal
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


def beads(y, d, fc, r, lam0, lam1, lam2, Nit=30, pen='L1_v2', EPS0=1e-6, EPS1=1e-6):
    """
    Baseline estimation and denoising using sparsity (BEADS)
    
    INPUT
        y: Noisy observation (numpy array or torch tensor)
        d: Filter order (d = 1 or 2)
        fc: Filter cut-off frequency (cycles/sample) (0 < fc < 0.5)
        r: Asymmetry ratio
        lam0, lam1, lam2: Regularization parameters
        Nit: Number of iterations (default 30)
        pen: Penalty function ('L1_v1' or 'L1_v2')
        EPS0: Cost smoothing parameter for x
        EPS1: Cost smoothing parameter for derivatives
        
    OUTPUT
        x: Estimated sparse-derivative signal (torch tensor)
        f: Estimated baseline (torch tensor)
        cost: Cost function history (torch tensor)
    """
    # Define penalty functions (using numpy for efficiency)
    if pen == 'L1_v1':
        phi = lambda x: np.sqrt(np.abs(x)**2 + EPS1)
        wfun = lambda x: 1.0 / np.sqrt(np.abs(x)**2 + EPS1)
    elif pen == 'L1_v2':
        phi = lambda x: np.abs(x) - EPS1 * np.log(np.abs(x) + EPS1)
        wfun = lambda x: 1.0 / (np.abs(x) + EPS1)
    else:
        raise ValueError("penalty must be 'L1_v1' or 'L1_v2'")
    
    def theta(x):
        """Asymmetric penalty function"""
        pos_mask = x > EPS0
        neg_mask = x < -EPS0
        mid_mask = np.abs(x) <= EPS0
        
        result = np.sum(x[pos_mask]) - r * np.sum(x[neg_mask])
        result = result + np.sum(
            (1 + r) / (4 * EPS0) * x[mid_mask]**2 
            + (1 - r) / 2 * x[mid_mask] 
            + EPS0 * (1 + r) / 4
        )
        return result
    
    # Convert input to numpy
    if isinstance(y, torch.Tensor):
        y = y.numpy()
    y = np.asarray(y, dtype=np.float64).flatten()
    
    N = len(y)
    x = y.copy()
    cost = np.zeros(Nit)
    
    # Get filter matrices (scipy sparse)
    A, B = BAfilt(d, fc, N)
    
    # First difference matrix D1: (N-1) x N
    # D1[i, i] = -1, D1[i, i+1] = 1
    e = np.ones(N)
    D1 = sparse.spdiags([-e[:-1], e[:-1]], [0, 1], N - 1, N, format='csc')
    
    # Second difference matrix D2: (N-2) x N
    # D2[i, i] = 1, D2[i, i+1] = -2, D2[i, i+2] = 1
    D2 = sparse.spdiags([e[:-2], -2 * e[:-2], e[:-2]], [0, 1, 2], N - 2, N, format='csc')
    
    # D = [D1; D2]
    D = sparse.vstack([D1, D2], format='csc')
    
    BTB = B.T @ B
    
    # w = [lam1 * ones(N-1, 1); lam2 * ones(N-2, 1)]
    w = np.concatenate([lam1 * np.ones(N - 1), lam2 * np.ones(N - 2)])
    
    # b = (1-r)/2 * ones(N, 1)
    b_vec = (1 - r) / 2 * np.ones(N)
    
    # d_vec = BTB * (A\y) - lam0 * A' * b
    # Solve A * z = y
    z = spsolve(A, y)
    d_vec = BTB @ z - lam0 * A.T @ b_vec
    
    gamma = np.ones(N)
    
    for i in range(Nit):
        # Lambda = spdiags(w.*wfun(D*x), 0, 2*N-3, 2*N-3)
        Dx = D @ x
        lambda_diag = w * wfun(Dx)
        Lambda = sparse.diags(lambda_diag, 0, format='csc')
        
        # k = abs(x) > EPS0
        k = np.abs(x) > EPS0
        # gamma(~k) = ((1 + r)/4) / abs(EPS0)
        gamma[~k] = ((1 + r) / 4) / abs(EPS0)
        # gamma(k) = ((1 + r)/4) ./  abs(x(k))
        gamma[k] = ((1 + r) / 4) / np.abs(x[k])
        Gamma = sparse.diags(gamma, 0, format='csc')
        
        # M = 2 * lam0 * Gamma + D' * Lambda * D
        M = 2 * lam0 * Gamma + D.T @ Lambda @ D
        
        # x = A * ((BTB + A'*M*A)\d)
        AMA = A.T @ M @ A
        lhs = BTB + AMA
        x_new = spsolve(lhs, d_vec)
        x = A @ x_new
        
        # Compute cost function
        # cost(i) = 0.5 * sum(abs(H(y - x)).^2) + lam0 * theta(x) ...
        #     + lam1 * sum(phi(diff(x))) + lam2 * sum(phi(diff(x, 2)))
        
        # H(y - x) = B * (A \ (y - x))
        residual = y - x
        z_res = spsolve(A, residual)
        H_res = B @ z_res
        
        diff1 = np.diff(x)  # diff(x)
        diff2 = np.diff(x, n=2)  # diff(x, 2)
        
        cost[i] = (0.5 * np.sum(np.abs(H_res)**2) 
                   + lam0 * theta(x) 
                   + lam1 * np.sum(phi(diff1)) 
                   + lam2 * np.sum(phi(diff2)))
    
    # f = y - x - H(y-x)
    residual = y - x
    z_res = spsolve(A, residual)
    H_res = B @ z_res
    f = y - x - H_res
    
    # Convert back to torch tensors for output
    x = torch.tensor(x, dtype=torch.float64)
    f = torch.tensor(f, dtype=torch.float64)
    cost = torch.tensor(cost, dtype=torch.float64)
    
    return x, f, cost
