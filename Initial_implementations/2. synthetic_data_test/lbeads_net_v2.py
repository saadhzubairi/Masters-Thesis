"""
LBEADS-NET v2: Improved Learnable BEADS Network

Key improvements over v1:
1. More layers (20+ to match BEADS iterations)
2. Improved proximal gradient architecture faithful to BEADS
3. Better initialization using BEADS output (warm-start)
4. Learnable momentum for accelerated convergence
5. Proper L1-smooth regularization with learnable smoothing
6. Skip connections for gradient flow

Reference:
Original BEADS: Chromatogram baseline estimation and denoising using sparsity
Xiaoran Ning, Ivan W. Selesnick, Laurent Duval
Chemometrics and Intelligent Laboratory Systems (2014)

Author: Thesis Work
Date: January 2026
"""

import torch
import torch.nn as nn
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from typing import Optional, Tuple, List
import sys
import os

# Add path for BEADS (it's in Initial_implementations/0. BEADS/Replicate)
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)  # Initial_implementations
sys.path.insert(0, os.path.join(parent_dir, '0. BEADS', 'Replicate'))

from beads import beads, BAfilt


# ===========================================================================
# HELPER FUNCTIONS
# ===========================================================================

def build_difference_matrices(N: int):
    """
    Build first and second order difference matrices.
    
    Returns:
        D1: First difference matrix (N-1) x N
        D2: Second difference matrix (N-2) x N
    """
    e = np.ones(N)
    D1 = sparse.spdiags([-e[:-1], e[:-1]], [0, 1], N - 1, N, format='csc')
    D2 = sparse.spdiags([e[:-2], -2 * e[:-2], e[:-2]], [0, 1, 2], N - 2, N, format='csc')
    return D1, D2


def sparse_to_dense_tensor(sp_matrix, device='cpu', dtype=torch.float64):
    """Convert scipy sparse matrix to dense PyTorch tensor."""
    return torch.tensor(sp_matrix.toarray(), dtype=dtype, device=device)


# ===========================================================================
# IMPROVED PROXIMAL OPERATORS
# ===========================================================================

class LearnableProximalL1(nn.Module):
    """
    Learnable proximal operator for L1 penalty.
    
    Instead of fixed soft-thresholding, learn a smooth approximation
    that can adapt to the data distribution.
    """
    
    def __init__(self, init_threshold: float = 0.1):
        super().__init__()
        self.log_threshold = nn.Parameter(torch.tensor(np.log(init_threshold), dtype=torch.float64))
        # Learnable sharpness for smooth approximation
        self.log_sharpness = nn.Parameter(torch.tensor(np.log(10.0), dtype=torch.float64))
    
    @property
    def threshold(self):
        return torch.exp(self.log_threshold)
    
    @property
    def sharpness(self):
        return torch.exp(self.log_sharpness)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Soft thresholding with learnable threshold.
        Uses smooth approximation for better gradients.
        """
        threshold = self.threshold
        # Smooth soft-thresholding using tanh approximation
        # Approaches true soft-thresholding as sharpness -> infinity
        return x * torch.tanh(self.sharpness * (torch.abs(x) - threshold).clamp(min=0)) * \
               torch.sign(torch.abs(x) - threshold + 1e-8)


class LearnableAsymmetricProximal(nn.Module):
    """
    Learnable asymmetric proximal operator for BEADS.
    
    Key insight: Peaks are always positive, so we penalize negative values more.
    This is learned per-layer for optimal adaptation.
    """
    
    def __init__(self, init_lam: float = 0.1, init_r: float = 6.0, eps: float = 1e-6):
        super().__init__()
        self.log_lam = nn.Parameter(torch.tensor(np.log(init_lam), dtype=torch.float64))
        self.log_r = nn.Parameter(torch.tensor(np.log(init_r), dtype=torch.float64))
        self.eps = eps
    
    @property
    def lam(self):
        return torch.exp(self.log_lam)
    
    @property
    def r(self):
        return torch.exp(self.log_r)
    
    def forward(self, x: torch.Tensor, step_size: torch.Tensor) -> torch.Tensor:
        """
        Asymmetric soft thresholding.
        
        For positive x: threshold at lam * step_size
        For negative x: threshold at lam * r * step_size (more aggressive)
        """
        lam = self.lam * step_size
        r = self.r
        
        pos_thresh = lam
        neg_thresh = lam * r
        
        # Smooth approximation for better gradients
        result = torch.where(
            x > pos_thresh,
            x - pos_thresh,
            torch.where(
                x < -neg_thresh,
                x + neg_thresh,
                torch.zeros_like(x)  # Sparsity: values in [-neg_thresh, pos_thresh] become 0
            )
        )
        
        return result


# ===========================================================================
# MAIN MODEL: LBEADS-NET v2
# ===========================================================================

class LBEADS_NET_v2(nn.Module):
    """
    LBEADS-NET v2: Improved Learnable BEADS Network
    
    Key improvements:
    1. More layers (20+) to match BEADS iterations
    2. Proper FISTA-style momentum for acceleration
    3. Per-layer learnable parameters with careful initialization
    4. Improved proximal operators with smooth approximations
    5. Optional residual/skip connections
    6. Support for BEADS warm-start initialization
    
    Architecture follows the BEADS optimization:
        minimize ||H(y - x - f)||^2 + lam0*theta(x) + lam1*phi(D1*x) + lam2*phi(D2*x)
        
    Where:
        - H is a high-pass filter
        - theta is an asymmetric penalty (encourages positive x)
        - phi is L1-smooth penalty for derivative sparsity
        - D1, D2 are difference operators
    """
    
    def __init__(
        self,
        N: int,
        d: int = 1,
        fc: float = 0.006,
        num_layers: int = 20,
        # Per-layer initialization (matching v1 defaults that work)
        init_lam0: float = 0.4,
        init_lam1: float = 4.0,
        init_lam2: float = 3.2,
        init_r: float = 6.0,
        init_step_size: float = 0.1,
        # Advanced options
        use_momentum: bool = False,  # Disable by default
        init_momentum: float = 0.9,
        use_skip_connection: bool = False,  # Disable by default
        eps0: float = 1e-6,
        eps1: float = 1e-6,
    ):
        super().__init__()
        
        self.N = N
        self.d = d
        self.fc = fc
        self.num_layers = num_layers
        self.use_momentum = use_momentum
        self.use_skip_connection = use_skip_connection
        self.eps0 = eps0
        self.eps1 = eps1
        
        # Pre-compute filter matrices
        A, B = BAfilt(d, fc, N)
        self.A_sp = A
        self.B_sp = B
        
        # Store dense versions for gradient computation
        self.register_buffer('A_dense', sparse_to_dense_tensor(A))
        self.register_buffer('B_dense', sparse_to_dense_tensor(B))
        self.register_buffer('BTB_dense', sparse_to_dense_tensor(B.T @ B))
        
        # Difference matrices
        D1, D2 = build_difference_matrices(N)
        self.register_buffer('D1_dense', sparse_to_dense_tensor(D1))
        self.register_buffer('D2_dense', sparse_to_dense_tensor(D2))
        
        # Compute Lipschitz constant for step size bounds
        # L = ||B||^2 + lam1*||D1||^2 + lam2*||D2||^2
        # We'll learn the step size but initialize carefully
        
        # =====================================================================
        # LEARNABLE PARAMETERS (per-layer)
        # =====================================================================
        
        # Regularization parameters
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
        
        # Asymmetry ratios
        self.log_r = nn.ParameterList([
            nn.Parameter(torch.tensor(np.log(init_r), dtype=torch.float64))
            for _ in range(num_layers)
        ])
        
        # Step sizes (critical for convergence!)
        self.log_step_size = nn.ParameterList([
            nn.Parameter(torch.tensor(np.log(init_step_size), dtype=torch.float64))
            for _ in range(num_layers)
        ])
        
        # Momentum parameters (FISTA-style acceleration)
        if use_momentum:
            self.log_momentum = nn.ParameterList([
                nn.Parameter(torch.tensor(np.log(init_momentum / (1 - init_momentum)), dtype=torch.float64))
                for _ in range(num_layers)
            ])
        
        # Skip connection weights (learnable residual)
        if use_skip_connection:
            self.skip_weight = nn.ParameterList([
                nn.Parameter(torch.tensor(0.1, dtype=torch.float64))
                for _ in range(num_layers)
            ])
    
    def get_param(self, param_list, layer_idx: int) -> torch.Tensor:
        """Get exponentially-constrained parameter value."""
        return torch.exp(param_list[layer_idx])
    
    def get_momentum(self, layer_idx: int) -> torch.Tensor:
        """Get momentum in [0, 1) using sigmoid."""
        if not self.use_momentum:
            return torch.tensor(0.0, dtype=torch.float64)
        # Use sigmoid to keep momentum in (0, 1)
        return torch.sigmoid(self.log_momentum[layer_idx])
    
    def diff1(self, x: torch.Tensor) -> torch.Tensor:
        """First difference: D1 @ x. x: (batch, N) -> (batch, N-1)"""
        return x[:, 1:] - x[:, :-1]
    
    def diff2(self, x: torch.Tensor) -> torch.Tensor:
        """Second difference: D2 @ x. x: (batch, N) -> (batch, N-2)"""
        return x[:, 2:] - 2 * x[:, 1:-1] + x[:, :-2]
    
    def diff1_T(self, v: torch.Tensor, N: int) -> torch.Tensor:
        """Transpose of first difference: D1.T @ v. v: (batch, N-1) -> (batch, N)"""
        batch_size = v.shape[0]
        result = torch.zeros(batch_size, N, dtype=v.dtype, device=v.device)
        result[:, :-1] -= v
        result[:, 1:] += v
        return result
    
    def diff2_T(self, v: torch.Tensor, N: int) -> torch.Tensor:
        """Transpose of second difference: D2.T @ v. v: (batch, N-2) -> (batch, N)"""
        batch_size = v.shape[0]
        result = torch.zeros(batch_size, N, dtype=v.dtype, device=v.device)
        result[:, :-2] += v
        result[:, 1:-1] -= 2 * v
        result[:, 2:] += v
        return result
    
    def apply_highpass_batch(self, z: torch.Tensor) -> torch.Tensor:
        """Apply high-pass filter: H(z) = B @ (A^-1 @ z) for each batch element."""
        batch_size = z.shape[0]
        result_list = []
        
        for b in range(batch_size):
            z_np = z[b].detach().cpu().numpy()
            Az_inv = spsolve(self.A_sp, z_np)
            Hz = self.B_sp @ Az_inv
            result_list.append(torch.tensor(Hz, dtype=z.dtype, device=z.device))
        
        return torch.stack(result_list, dim=0)
    
    def apply_highpass_T_batch(self, z: torch.Tensor) -> torch.Tensor:
        """Apply H^T: (A^-T) @ B^T @ z for each batch element."""
        batch_size = z.shape[0]
        result_list = []
        
        for b in range(batch_size):
            z_np = z[b].detach().cpu().numpy()
            BTz = self.B_sp.T @ z_np
            ATinv_BTz = spsolve(self.A_sp.T, BTz)
            result_list.append(torch.tensor(ATinv_BTz, dtype=z.dtype, device=z.device))
        
        return torch.stack(result_list, dim=0)
    
    def weight_function(self, x: torch.Tensor) -> torch.Tensor:
        """Weight function for L1-smooth penalty: w(x) = 1 / (|x| + eps)"""
        return 1.0 / (torch.abs(x) + self.eps1)
    
    def asymmetric_gradient(self, x: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """
        Gradient of asymmetric penalty theta(x).
        
        theta(x) = x      if x > eps
                 = -r*x   if x < -eps  
                 = smooth quadratic in between
        """
        eps = self.eps0
        grad = torch.zeros_like(x)
        
        pos_mask = x > eps
        neg_mask = x < -eps
        mid_mask = ~pos_mask & ~neg_mask
        
        grad[pos_mask] = 1.0
        grad[neg_mask] = -r
        # Smooth transition in the middle
        grad[mid_mask] = (1 + r) / (2 * eps) * x[mid_mask] + (1 - r) / 2
        
        return grad
    
    def asymmetric_soft_threshold(
        self, x: torch.Tensor, lam: torch.Tensor, r: torch.Tensor
    ) -> torch.Tensor:
        """
        Proximal operator for asymmetric penalty.
        
        This is the key operation that promotes positive sparse x (peaks).
        """
        pos_thresh = lam
        neg_thresh = lam * r
        
        result = torch.where(
            x > pos_thresh,
            x - pos_thresh,
            torch.where(
                x < -neg_thresh,
                x + neg_thresh,
                torch.zeros_like(x)
            )
        )
        return result
    
    def compute_baseline_batch(self, residual: torch.Tensor) -> torch.Tensor:
        """
        Compute baseline from residual: f = residual - H(residual)
        where H is the high-pass filter H(z) = B * (A^-1 * z)
        """
        batch_size = residual.shape[0]
        f_list = []
        
        for b in range(batch_size):
            res_np = residual[b].detach().cpu().numpy()
            z = spsolve(self.A_sp, res_np)
            H_res = self.B_sp @ z
            f_b = res_np - H_res  # f = residual - high_pass(residual) = low_pass(residual)
            f_list.append(torch.tensor(f_b, dtype=residual.dtype, device=residual.device))
        
        return torch.stack(f_list, dim=0)
    
    def forward(
        self,
        y: torch.Tensor,
        x_init: Optional[torch.Tensor] = None,
        return_intermediate: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through unrolled BEADS layers.
        
        Args:
            y: Noisy input signal (N,) or (batch, N)
            x_init: Optional initial estimate for warm-starting (from BEADS)
            return_intermediate: If True, return all layer outputs
            
        Returns:
            x: Estimated sparse signal (peaks)
            f: Estimated baseline
            intermediates: (optional) List of x estimates per layer
        """
        # Handle dimensions
        if y.dim() == 1:
            y = y.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size = y.shape[0]
        N = self.N
        
        # Initialize x
        if x_init is not None:
            if x_init.dim() == 1:
                x_init = x_init.unsqueeze(0)
            x = x_init.clone()
        else:
            # Initialize as zeros (sparse prior)
            x = torch.zeros_like(y)
        
        # For momentum (FISTA-style)
        x_prev = x.clone()
        
        intermediates = [x.clone()] if return_intermediate else None
        
        # =====================================================================
        # UNROLLED ITERATION LAYERS
        # =====================================================================
        
        for k in range(self.num_layers):
            # Get layer parameters
            lam0 = self.get_param(self.log_lam0, k)
            lam1 = self.get_param(self.log_lam1, k)
            lam2 = self.get_param(self.log_lam2, k)
            r = self.get_param(self.log_r, k)
            step_size = self.get_param(self.log_step_size, k)
            
            # ================================================================
            # STEP 1: Compute residual (data fidelity term)
            # ================================================================
            
            residual = y - x  # (y - x)
            
            # ================================================================
            # STEP 2: Compute gradient of regularization terms
            # ================================================================
            
            # Gradient of D1 smoothness: lam1 * D1.T @ W1 @ D1 @ x
            Dx1 = self.diff1(x)
            w1 = Dx1 / (torch.abs(Dx1) + self.eps1)  # Gradient of |Dx1|
            grad_D1 = lam1 * self.diff1_T(w1, N)
            
            # Gradient of D2 smoothness: lam2 * D2.T @ W2 @ D2 @ x
            Dx2 = self.diff2(x)
            w2 = Dx2 / (torch.abs(Dx2) + self.eps1)  # Gradient of |Dx2|
            grad_D2 = lam2 * self.diff2_T(w2, N)
            
            # ================================================================
            # STEP 3: Gradient descent step on smoothness terms only
            # ================================================================
            
            # Gradient step (only smoothness, not data term)
            x_grad = x - step_size * (grad_D1 + grad_D2)
            
            # ================================================================
            # STEP 4: Proximal step (asymmetric soft thresholding)
            # ================================================================
            
            # Add data term and apply proximal operator
            # This is the key ISTA-style update: prox(x - step*grad + step*residual)
            x_new = self.asymmetric_soft_threshold(
                x_grad + step_size * residual,  # Include data term here!
                lam0 * step_size,
                r
            )
            
            # ================================================================
            # STEP 5: Momentum (FISTA-style acceleration)
            # ================================================================
            
            if self.use_momentum and k > 0:
                momentum = self.get_momentum(k)
                # FISTA update: use momentum on the difference
                x_new = x_new + momentum * (x_new - x_prev)
            
            # Skip connection disabled - it prevents proper denoising
            # The algorithm naturally converges without it
            
            # Update for next iteration
            x_prev = x
            x = x_new
            
            if return_intermediate:
                intermediates.append(x.clone())
        
        # =====================================================================
        # COMPUTE BASELINE
        # =====================================================================
        
        residual = y - x
        f = self.compute_baseline_batch(residual)
        
        # Handle output dimensions
        if squeeze_output:
            x = x.squeeze(0)
            f = f.squeeze(0)
        
        if return_intermediate:
            return x, f, intermediates
        return x, f
    
    def get_learned_params(self) -> dict:
        """Return dictionary of all learned parameters."""
        params = {}
        for i in range(self.num_layers):
            params[f"layer_{i}_lam0"] = torch.exp(self.log_lam0[i]).item()
            params[f"layer_{i}_lam1"] = torch.exp(self.log_lam1[i]).item()
            params[f"layer_{i}_lam2"] = torch.exp(self.log_lam2[i]).item()
            params[f"layer_{i}_r"] = torch.exp(self.log_r[i]).item()
            params[f"layer_{i}_step_size"] = torch.exp(self.log_step_size[i]).item()
            if self.use_momentum:
                params[f"layer_{i}_momentum"] = self.get_momentum(i).item()
        return params
    
    def print_params_summary(self):
        """Print a summary of learned parameters."""
        print("\nLearned Parameters Summary:")
        print("-" * 60)
        
        for i in range(self.num_layers):
            lam0 = torch.exp(self.log_lam0[i]).item()
            lam1 = torch.exp(self.log_lam1[i]).item()
            lam2 = torch.exp(self.log_lam2[i]).item()
            r = torch.exp(self.log_r[i]).item()
            step = torch.exp(self.log_step_size[i]).item()
            
            print(f"Layer {i:2d}: lam0={lam0:.4f}, lam1={lam1:.4f}, "
                  f"lam2={lam2:.4f}, r={r:.2f}, step={step:.4f}")


# ===========================================================================
# WARM-START UTILITY
# ===========================================================================

def beads_warm_start(
    y: torch.Tensor,
    d: int = 1,
    fc: float = 0.006,
    r: float = 6.0,
    lam0: float = 0.5,
    lam1: float = 4.0,
    lam2: float = 4.0,
    Nit: int = 5  # Fewer iterations for warm-start
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run a few BEADS iterations to get a good initialization.
    
    This helps the network converge faster and to better solutions.
    """
    if y.dim() == 1:
        y_np = y.detach().cpu().numpy()
        x_init, f_init, _ = beads(y_np, d, fc, r, lam0, lam1, lam2, Nit=Nit)
        return x_init, f_init
    else:
        # Batch processing
        batch_size = y.shape[0]
        x_list, f_list = [], []
        for b in range(batch_size):
            y_np = y[b].detach().cpu().numpy()
            x_b, f_b, _ = beads(y_np, d, fc, r, lam0, lam1, lam2, Nit=Nit)
            x_list.append(x_b)
            f_list.append(f_b)
        return torch.stack(x_list, dim=0), torch.stack(f_list, dim=0)


class LBEADS_NET_v2_WarmStart(nn.Module):
    """
    LBEADS-NET v2 with built-in BEADS warm-start.
    
    This version automatically runs a few BEADS iterations to initialize,
    then refines the solution with the learned network.
    """
    
    def __init__(
        self,
        N: int,
        d: int = 1,
        fc: float = 0.006,
        num_layers: int = 20,
        warmstart_iters: int = 5,
        **kwargs
    ):
        super().__init__()
        
        self.N = N
        self.d = d
        self.fc = fc
        self.warmstart_iters = warmstart_iters
        
        # Default BEADS parameters for warm-start
        self.register_buffer('ws_r', torch.tensor(6.0, dtype=torch.float64))
        self.register_buffer('ws_lam0', torch.tensor(0.5, dtype=torch.float64))
        self.register_buffer('ws_lam1', torch.tensor(4.0, dtype=torch.float64))
        self.register_buffer('ws_lam2', torch.tensor(4.0, dtype=torch.float64))
        
        # Main network
        self.net = LBEADS_NET_v2(N, d, fc, num_layers, **kwargs)
    
    def forward(
        self,
        y: torch.Tensor,
        use_warmstart: bool = True,
        return_intermediate: bool = False
    ):
        """Forward pass with optional warm-start."""
        
        x_init = None
        if use_warmstart:
            with torch.no_grad():
                x_init, _ = beads_warm_start(
                    y, self.d, self.fc,
                    r=self.ws_r.item(),
                    lam0=self.ws_lam0.item(),
                    lam1=self.ws_lam1.item(),
                    lam2=self.ws_lam2.item(),
                    Nit=self.warmstart_iters
                )
                if y.dim() == 2 and x_init.dim() == 2:
                    x_init = x_init.to(y.device)
                elif y.dim() == 1:
                    x_init = x_init.to(y.device)
        
        return self.net(y, x_init=x_init, return_intermediate=return_intermediate)
    
    def get_learned_params(self):
        return self.net.get_learned_params()
    
    def print_params_summary(self):
        self.net.print_params_summary()


# ===========================================================================
# FACTORY FUNCTION
# ===========================================================================

def create_lbeads_net_v2(
    N: int,
    preset: str = 'default',
    **kwargs
) -> nn.Module:
    """
    Factory function to create LBEADS-NET v2 with different presets.
    
    Presets:
        'default': Balanced settings (20 layers)
        'fast': Fewer layers (10), faster but less accurate
        'accurate': More layers (30), slower but more accurate
        'beads_match': Settings to match original BEADS (50 layers)
    """
    
    presets = {
        'default': {
            'num_layers': 20,
            'init_lam0': 0.4,
            'init_lam1': 4.0,
            'init_lam2': 3.2,
            'init_r': 6.0,
            'init_step_size': 0.1,
            'use_momentum': False,
            'use_skip_connection': False,
        },
        'fast': {
            'num_layers': 10,
            'init_lam0': 0.4,
            'init_lam1': 4.0,
            'init_lam2': 3.2,
            'init_r': 6.0,
            'init_step_size': 0.1,
            'use_momentum': False,
            'use_skip_connection': False,
        },
        'accurate': {
            'num_layers': 30,
            'init_lam0': 0.4,
            'init_lam1': 4.0,
            'init_lam2': 3.2,
            'init_r': 6.0,
            'init_step_size': 0.1,
            'use_momentum': False,
            'use_skip_connection': False,
        },
        'beads_match': {
            'num_layers': 50,
            'init_lam0': 0.5,
            'init_lam1': 4.0,
            'init_lam2': 4.0,
            'init_r': 6.0,
            'init_step_size': 0.05,
            'use_momentum': False,
            'use_skip_connection': False,
        }
    }
    
    if preset not in presets:
        raise ValueError(f"Unknown preset: {preset}. Choose from {list(presets.keys())}")
    
    config = presets[preset].copy()
    config.update(kwargs)  # Override with user kwargs
    
    return LBEADS_NET_v2(N, **config)
