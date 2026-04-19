LBEADS-NET: comprehensive engineering briefing for fixing baseline leakage
The baseline leakage problem in LBEADS-NET stems from a fundamental assumption violation: BEADS models peaks as sparse, but dense-peak regions violate sparsity, causing the low-pass filter to absorb peak energy into the baseline. The fix requires a multi-pronged approach — asymmetric loss functions that penalize baseline over-estimation 10–20× more than under-estimation, element-wise orthogonality penalties between peak and baseline components, learned proximal operators replacing fixed soft-thresholding, and signal-adaptive regularization that relaxes sparsity constraints in dense-peak regions. This document provides every technique, citation, hyperparameter, and implementation sketch needed to implement these improvements.

1. Why BEADS leaks baseline into peaks
The original BEADS cost function (Ning, Selesnick & Duval, Chemometrics and Intelligent Laboratory Systems, 2014) minimizes:

F(x) = (1/2)||y - x - Hx||² + Σᵢ λᵢ φ(Dⁱx) + r·θ(x)
where x = peaks (sparse), Hx = baseline (low-pass filtered), Dⁱx = i-th derivative, φ = smoothed ℓ₁ penalty, and θ = asymmetric positivity penalty. Three failure modes cause leakage in dense-peak regions:

Sparsity assumption violation. The ℓ₁ penalty on x and its derivatives assumes peaks are sparse. When peaks are densely packed (metabolomics, complex mixtures), the composite signal is neither sparse nor has sparse derivatives. The ℓ₁ penalty over-shrinks peaks, and the surplus energy is attributed to the baseline.

Low-pass filter bandwidth mismatch. The cutoff frequency fc determines what counts as "baseline." When peak tails overlap and their valleys don't reach the true baseline, the sustained elevation looks low-frequency to the filter, causing it to interpret overlapping peak tails as baseline content. Too-high fc directly causes peak leakage; too-low fc misses real baseline variation.

Uniform regularization. Classical BEADS applies the same λ₀, λ₁, λ₂ everywhere. Dense-peak regions need different regularization than sparse regions, but the algorithm has no mechanism to adapt spatially.

The Gharbi, Chouzenoux, Pesquet & Duval team (MLSP 2024, Signal Processing 2024) — the most directly relevant published work — compared unrolled primal-dual, unrolled ISTA, and unrolled Half-Quadratic algorithms for 1D chromatographic signal restoration 
Inria
 and found that unrolled HQ tends to underestimate peak intensities (measured by TSNR), 
Inria
 confirming that this leakage problem is fundamental to the class of unrolled sparse-recovery networks. The DIRAS+ paper (Analytical Chemistry, 2025) explicitly identified baseline leakage as a fundamental limitation of end-to-end deep learning approaches, noting that such architectures "may inadvertently encode chemically relevant features along with baseline trends." 
ACS Publications

2. Loss functions and regularizers that prevent leakage
The recommended composite loss
python
L_total = (L_recon 
         + λ_s * L_sparse 
         + λ_sm * L_smooth 
         + λ_orth * L_ortho 
         + λ_asym * L_asym 
         + λ_nn * L_nonneg 
         + λ_freq * L_freq 
         + λ_env * L_envelope)
Each term addresses a specific failure mode. Below are the mathematical formulations, recommended weight ranges, and implementation details for each.

Reconstruction fidelity
python
L_recon = ||y - x_peak - x_base||₂²
Standard data fidelity. No changes needed; this is the anchor term with weight 1.0.

Asymmetric baseline penalty (highest priority)
This is the single most impactful addition. Penalize baseline over-estimation (where leakage occurs) 10–20× more than under-estimation:

python
def asymmetric_loss(baseline_pred, baseline_true, alpha=0.9):
    """alpha controls asymmetry. 0.9 = 9:1 over:under penalty ratio."""
    residual = baseline_pred - baseline_true
    loss = torch.where(
        residual > 0,
        alpha * residual ** 2,         # over-estimation: heavy penalty
        (1 - alpha) * residual ** 2    # under-estimation: light penalty
    )
    return loss.mean()
When ground truth is unavailable (real data), use a proxy: residual = baseline_pred - soft_min(y, window) where soft_min is a differentiable local minimum approximation:

python
def soft_local_min(y, window=51, tau=0.1):
    """Differentiable approximation to sliding-window minimum."""
    y_unf = F.unfold(y.unsqueeze(1).unsqueeze(-1), (window, 1), padding=(window//2, 0))
    return -tau * torch.logsumexp(-y_unf / tau, dim=1)
This derives from the AsLS family (Eilers, 2005) and arPLS (Baek et al., Analyst, 2015). Recommended weight: λ_asym = 0.1–1.0, α = 0.85–0.95. Start with α=0.9.

Element-wise orthogonality penalty
Directly penalizes regions where both peak and baseline are simultaneously active:

python
L_ortho = ||x_peak ⊙ x_base||₁  # element-wise product, L1 norm
This is adapted from orthogonal NMF (Ding et al., KDD 2006; Pompili et al., IEEE TKDE 2014) and PE-NMF (Zhang et al., Computational Intelligence and Neuroscience, 2008). A stricter variant uses the squared L2: ||x_peak ⊙ x_base||₂². Recommended weight: λ_orth = 0.01–0.1. Start with 0.05.

Frequency-domain separation loss
Since peaks have broadband frequency content and baselines are low-frequency, directly penalize high-frequency content in the baseline and low-frequency content in peaks:

python
def freq_separation_loss(x_base, x_peak, fc=0.05):
    """Penalize HPF(baseline) and LPF(peaks)."""
    X_base = torch.fft.rfft(x_base, dim=-1)
    X_peak = torch.fft.rfft(x_peak, dim=-1)
    freqs = torch.fft.rfftfreq(x_base.shape[-1])
    hpf_mask = (freqs > fc).float()
    lpf_mask = (freqs <= fc).float()
    return (X_base.abs() * hpf_mask).pow(2).mean() + (X_peak.abs() * lpf_mask).pow(2).mean()
Recommended weight: λ_freq = 0.01–0.5. The cutoff fc should match the BEADS filter cutoff (typically 0.003–0.05 cycles/sample).

Envelope constraint (prevents baseline from exceeding local minima)
python
L_envelope = ||ReLU(x_base - soft_local_min(y, window))||₂²
This provides a hard upper bound: the baseline should never significantly exceed the local signal minimum. Recommended weight: λ_env = 0.5–5.0, window = 2–5× typical peak FWHM.

Gradient-overlap penalty (specifically targets dense-peak regions)
python
L_grad_overlap = ||∇x_peak ⊙ ∇x_base||₁
Penalizes regions where both components have significant gradients simultaneously — exactly the boundary regions where leakage occurs. Recommended weight: λ_grad = 0.01–0.1.

Smoothness on baseline
python
L_smooth = ||D² x_base||₂²  # second derivative penalty (Tikhonov)
Or OGS-TV (Selesnick & Chen, ICASSP 2013) for promoting extended smooth regions rather than pointwise smoothness. Recommended weight: λ_sm = 1.0–100.0.

Locally adaptive smoothness (key for dense peaks)
Increase the smoothness constraint where peaks are detected:

python
def adaptive_smooth_loss(x_base, x_peak_estimate):
    weights = 1.0 + 10.0 * torch.abs(x_peak_estimate).detach()  # detach to avoid collapse
    D2_base = x_base[:, :, 2:] - 2*x_base[:, :, 1:-1] + x_base[:, :, :-2]
    return (weights[:, :, 1:-1] * D2_base ** 2).mean()
The .detach() on peak estimates prevents a degenerate solution where the network zeroes out peaks to reduce the smoothness penalty. Weight: included in λ_sm.

Non-negativity of peaks
Enforce architecturally (preferred) or as a soft penalty:

python
# Architectural (hard constraint, recommended):
x_peak = F.softplus(z_peak, beta=5)  # or F.relu(z_peak)

# Soft penalty (alternative):
L_nonneg = ||ReLU(-x_peak)||₂²
Softplus with β=5 provides a good balance between hard non-negativity and gradient flow. ReLU works but has zero gradients for negative inputs, which can slow training.

3. Algorithm unrolling: architectural improvements for LBEADS-NET
Replace soft-thresholding with a learned proximal operator
This is the highest-impact architectural change. The fixed soft-thresholding in BEADS cannot adapt to varying peak densities. Replace it with a small 1D CNN:

python
class LearnedProximal(nn.Module):
    """Replaces soft-thresholding in each unrolled BEADS iteration."""
    def __init__(self, channels=1, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, hidden, 7, padding=3),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, 5, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden, channels, 3, padding=1),
        )
    def forward(self, x, threshold):
        # Residual structure: learned correction to soft-thresholding
        x_st = F.softshrink(x, lambd=threshold)
        return x_st + self.net(x.unsqueeze(1) if x.dim()==1 else x)
This follows ISTA-Net+ (Zhang & Ghanem, CVPR 2018), which replaces handcrafted sparsifying transforms with learned nonlinear transforms and achieves significant improvements. The ODP framework (Diamond & Sitzmann, 2017) showed that the choice of optimization algorithm matters less than the quality of the learned prior 
arXiv
 — directly motivating this change. The Hybrid ISTA paper (Zheng et al., IEEE TPAMI 2022; GitHub: ZhengZY-EE/Hybrid_ISTA) proved that incorporating arbitrary neural networks into unrolled ISTA maintains convergence if the learned component satisfies a contractivity condition.

Make parameters layer-specific and signal-adaptive
Each unrolled stage should have its own λ₀, λ₁, λ₂, and additionally condition these on the input signal:

python
class SignalAdaptiveParams(nn.Module):
    """Predicts per-signal parameter adjustments from input."""
    def __init__(self, signal_len, n_stages, n_params=5):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, 15, stride=4, padding=7),
            nn.GELU(),
            nn.Conv1d(16, 32, 7, stride=4, padding=3),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(16),
            nn.Flatten(),
            nn.Linear(32 * 16, n_stages * n_params),
            nn.Softplus(),  # parameters must be positive
        )
    def forward(self, y):
        # Returns (n_stages, n_params) parameter adjustments
        return self.encoder(y.unsqueeze(1)).reshape(-1, n_stages, n_params)
This follows ISTA-Net++ (You et al., IEEE TIP 2021), which uses dynamic parameters conditioned on input characteristics, and the Neurally Augmented ALISTA approach, which uses an LSTM to compute per-signal step sizes and thresholds. DIRAS+ (Analytical Chemistry, 2025) uses a CNN+XGBoost pipeline to predict optimal λ per-spectrum, achieving 0.036 s/spectrum inference. 
ACS Publications

Add intermediate supervision
Add loss at every unrolled stage, not just the final output:

python
total_loss = 0
for k, (peak_k, base_k) in enumerate(intermediate_outputs):
    stage_weight = 0.1 + 0.9 * (k / (n_stages - 1))  # linearly increasing weight
    total_loss += stage_weight * compute_loss(peak_k, base_k, y, target)
DeMUN (Entropy, MDPI, 2025) showed that intermediate supervision significantly improves training by avoiding poor local minima and providing gradient information at every depth. 
MDPI
 Use linearly increasing weights so later (more refined) stages contribute more.

ADMM-style variable splitting
Reformulate BEADS as an ADMM problem where peak and baseline are separate auxiliary variables. Each stage alternates:

python
# Stage k of unrolled ADMM-BEADS:
# Step 1: Update peak estimate (sparse denoising)
x_peak_k = proximal_peak(y - x_base_{k-1} - u_k, lambda_peak_k)

# Step 2: Update baseline estimate (low-pass filtering)  
x_base_k = lowpass_filter(y - x_peak_k + u_k, fc_k)

# Step 3: Dual variable update
u_k = u_{k-1} + rho_k * (y - x_peak_k - x_base_k)
ADMM-Net (Yang et al., NeurIPS 2016) showed that ADMM's variable splitting provides natural separation between data fidelity and regularization substeps 
ResearchGate
Coder-nova
 — directly analogous to peak/baseline separation. The piecewise-linear shrinkage functions learned by ADMM-Net are more flexible than soft-thresholding.

Recommended number of stages and weight sharing
Based on the literature: start with 8–12 stages. Beyond ~15 stages, returns diminish and training becomes harder (vanishing gradients). Use layer-specific parameters for λ₀, λ₁, λ₂ (they should adapt per stage — early stages do rough separation, later stages refine), but share structural parameters (difference operators D, filter structure H) unless training data is abundant. ALISTA (Liu et al., ICLR 2019) showed that sharing weight matrices while allowing layer-specific step sizes and thresholds gives the best parameter efficiency — only 2T learnable scalars for T layers.

Initialization
Initialize all learnable parameters at their classical BEADS values before end-to-end training. This is critical — ALISTA, ADMM-Net, and ISTA-Net all use this strategy. Specifically:

λ₀, λ₁, λ₂ → classical BEADS optimal values for representative chromatograms
Filter cutoff fc → 0.003–0.01 cycles/sample (from BEADS paper)
Asymmetry ratio r → classical BEADS value (typically 1–6)
output_gain → 1.0
Learned proximal CNN → pre-train as a denoiser on synthetic peak signals
Deep Equilibrium Model alternative
For maximum flexibility, convert LBEADS-NET to a DEQ formulation (Bai, Kolter & Koltun, NeurIPS 2019):

python
# Instead of fixed K stages, find fixed point:
# z* = f(z*, y) where f is one BEADS iteration
# Backward pass uses implicit differentiation (O(1) memory)
from torchdeq import get_deq

deq = get_deq(f_solver='anderson', b_solver='broyden')
z_star = deq(beads_iteration, z0, solver_kwargs={'threshold': 30, 'eps': 1e-5})
This allows the network to iterate until convergence for each signal — dense-peak signals that need more iterations get them automatically. MsDC-DEQ-Net (Yu & Dansereau, 2024) demonstrated this for compressive sensing. 
arXiv
 GUDL (2025) combined DEQ with GSURE for unsupervised sparse recovery without ground truth. 
arXiv
 Recommendation: implement as a follow-up experiment after the fixed-stage improvements are validated.

4. Classical baseline methods: what handles dense peaks best
Comparative ranking for dense-peak regions
Based on the comprehensive survey:

arPLS (Baek et al., Analyst, 2015) — Best among AsLS variants. Its logistic-function weighting adapts to noise level, degrading gracefully in dense regions. 
ScienceDirect
 The baseline rises toward inter-peak valleys (elevation, not leakage). Implementation: pybaselines.Baseline().arpls(y, lam=1e6).
SNIP (Ryan et al., 1988) — Locally adaptive, no parametric model assumed. The LLS compression prevents large peaks from dominating. Works well when at least some valleys reach near the true baseline. Single parameter (max_half_window). Implementation: pybaselines.Baseline().snip(y, max_half_window=40).
Rolling ball / morphological — If structuring element is well-chosen (radius > peak width), traces through dense valleys. Robust to peak density. Implementation: pybaselines.Baseline().mor(y, half_window=100).
airPLS (Zhang et al., Analyst, 2010) — Good in moderate density but has three known failure modes: non-smooth baselines, significant errors in broad peak regions, and difficulties with complex spectral regions. 
ACS Publications
 OP-airPLS (2025) added ML-predicted parameters for 2100× speedup.
BEADS — Worst in dense-peak regions due to sparsity assumption violation. The ℓ₁ penalty over-shrinks dense peaks and the low-pass filter absorbs the residual. This is exactly the problem LBEADS-NET must solve.
Key insight for LBEADS-NET
arPLS has no sparsity assumption — it simply fits a smooth curve below the data with adaptive weighting. In dense regions, arPLS's baseline rises toward inter-peak valleys (which may be elevated) but doesn't actively compress peak signal. The failure mode is "elevation" (baseline too high, but smooth) rather than "leakage" (baseline follows peak shapes). This suggests that relaxing the sparsity penalty in dense regions — which a learned network can do spatially — would eliminate the core failure mode. The LBEADS-NET architecture should learn to behave more like arPLS in dense regions and more like BEADS in sparse regions.

Neural approaches (2020–2025)
The most relevant published neural approaches:

CAE+ (Han et al., Sensors, 2024): Convolutional autoencoder with comparison function. 
MDPI
PubMed
 Peak preservation rates of 0.851–0.96. 
MDPI
PubMed Central
 airPLS showed 2–3× higher error in peak regions vs. CAE+.
ResNet+UNet (Chen et al., Analyst, 2022): Hybrid architecture for Raman baseline correction. Eliminates manual parameter adjustment. 
ResearchGate
RSC Publishing
1dTrans (Zhao et al., Spectrochimica Acta Part A, 2025): First Transformer for baseline estimation. Lower MAE and SAM than CNN, ResUNet, and classical methods. 
ResearchGate
ScienceDirect
Kensert autoencoder (Kensert et al., J. Chromatography A, 2021): 1D conv autoencoder trained on 190,000 simulated chromatograms. 
ResearchGate
ScienceDirect
 RMSE 1.094 mAU vs. 2.074 (SG). 
X-MOL
 GitHub: akensert/autoencoder-chromatogram-enhancement.
RSPSSL (Hu et al., Light: Science & Applications, 2024): Self-supervised approach using Raman Spectral GAN. 88% RMSE reduction and 60% L∞ reduction vs. established methods. Processes ~1,900 spectra/second. 
Nature
The pybaselines library
The pybaselines library (GitHub: derb12/pybaselines, BSD-3) implements 50+ algorithms with a unified API: Baseline(x_data).method(y, **params). It includes BEADS (Baseline().beads()), all AsLS variants, SNIP, morphological methods, polynomial methods, and more. This is the reference implementation for benchmarking LBEADS-NET against classical methods.

5. Alternative architectures worth considering
Conformer (convolution + Transformer): most promising alternative
The Conformer (Gulati et al., Interspeech 2020) combines local feature extraction (depthwise convolutions) with global context modeling (self-attention) in a sandwich structure. 
ScienceDirect
 Each block processes: FFN → Self-Attention → Conv Module → FFN → LayerNorm. The convolution module captures local peak shapes (sharp, narrow features) while self-attention captures global baseline trends (smooth, broad features).

For LBEADS-NET: Replace the proximal operator or the entire unrolled iteration with Conformer blocks. Use linear attention (DF-Conformer, Interspeech 2022) to avoid O(N²) complexity for long chromatograms. 
ResearchGate
 PyTorch implementation: github.com/sooftware/conformer.

python
class ConformerProximal(nn.Module):
    """Conformer-based proximal operator replacement."""
    def __init__(self, d_model=64, n_heads=4, conv_kernel=31, n_layers=2):
        super().__init__()
        self.input_proj = nn.Conv1d(1, d_model, 1)
        self.conformer_blocks = nn.ModuleList([
            ConformerBlock(d_model, n_heads, conv_kernel) for _ in range(n_layers)
        ])
        self.output_proj = nn.Conv1d(d_model, 1, 1)
    
    def forward(self, x):
        h = self.input_proj(x.unsqueeze(1))   # (B, d_model, T)
        h = h.permute(0, 2, 1)                 # (B, T, d_model)
        for block in self.conformer_blocks:
            h = block(h)
        h = h.permute(0, 2, 1)                 # (B, d_model, T)
        return self.output_proj(h).squeeze(1)   # (B, T)
WaveNet-style dilated convolutions: best multi-scale approach
WaveNet (van den Oord et al., 2016) uses exponentially increasing dilation factors (1, 2, 4, ..., 512) to achieve large receptive fields without downsampling. 
arXiv
 A modified WaveNet won first place in the MIT RF Challenge for signal separation (ICASSP 2024) with learnable dilation parameters that adaptively modulate receptive field size — achieving 58.82% SINR improvement. 
arXiv

For LBEADS-NET: A dilated convolution backbone preserves sample-level resolution (critical for narrow chromatographic peaks) while capturing global baseline context. The gated activation (tanh × sigmoid) acts as a learnable filter that can selectively pass baseline or peak information.

Wave-U-Net difference output layer: enforces energy conservation
Wave-U-Net (Stoller et al., ISMIR 2018) computes the last source as the residual: peaks = signal - baseline. This architectural constraint enforces perfect reconstruction and prevents the network from creating or destroying signal energy. This should be adopted in LBEADS-NET: always compute peaks as the residual after baseline estimation, never estimate peaks directly.

Mamba / State Space Models: unexplored opportunity
Mamba (Gu & Dao, 2023) achieves linear O(N) complexity with data-dependent selection that can distinguish relevant from irrelevant features. 
AI Intuition
 No paper has applied Mamba to spectroscopy or chromatography baseline correction — this represents a novel research direction. The selection mechanism could learn to identify peak regions (where baseline estimation should be cautious) vs. baseline regions.

Peak-aware masked autoencoder pre-training
A novel approach not yet published: train a 1D masked autoencoder where peak regions are preferentially masked, forcing the network to interpolate the smooth baseline under peaks:

python
# Pre-training phase:
# 1. Detect approximate peak regions (using simple threshold or derivative)
# 2. Mask 90% of peak regions + 30% of baseline regions
# 3. Train MAE to reconstruct masked regions
# The decoder learns to "fill in" baseline under peaks

class PeakAwareMAE(nn.Module):
    def __init__(self, ...):
        self.peak_detector = simple_derivative_detector()  # non-learned
        self.encoder = Encoder(...)
        self.decoder = Decoder(...)
    
    def create_mask(self, y):
        peak_mask = self.peak_detector(y)
        # Mask 90% of peak regions, 30% of baseline regions
        mask = torch.where(peak_mask, torch.rand_like(y) < 0.9, torch.rand_like(y) < 0.3)
        return mask
6. Training data generation pipeline
Synthetic chromatogram generator
python
import numpy as np
from scipy.special import erfc

def generate_chromatogram(length=4096, difficulty='hard'):
    """Generate one synthetic chromatogram with known ground truth."""
    t = np.linspace(0, 30, length)  # 30-minute run
    
    # 1. Peak generation (EMG model)
    if difficulty == 'easy':
        n_peaks = np.random.randint(3, 10)
    elif difficulty == 'medium':
        n_peaks = np.random.randint(10, 30)
    else:  # hard
        n_peaks = np.random.randint(30, 100)
    
    peaks = np.zeros(length)
    for _ in range(n_peaks):
        tR = np.random.uniform(1, 29)           # retention time
        sigma = np.random.uniform(0.02, 0.3)     # Gaussian width
        tau = np.random.uniform(0, 5*sigma)       # exponential tail
        amp = 10 ** np.random.uniform(0, 3)       # amplitude (3 orders of magnitude)
        
        if tau > 1e-6:  # EMG peak
            z = (sigma**2 - tau*(t - tR)) / (np.sqrt(2) * sigma * tau)
            peak = (amp / (2*tau)) * np.exp(sigma**2/(2*tau**2) - (t-tR)/tau) * erfc(z)
        else:  # Pure Gaussian
            peak = amp * np.exp(-(t - tR)**2 / (2*sigma**2))
        peaks += peak
    
    # 2. Baseline generation (GP or polynomial + artifacts)
    baseline_type = np.random.choice(['gp', 'polynomial', 'spline', 'sinusoidal'])
    if baseline_type == 'gp':
        from sklearn.gaussian_processes.kernels import RBF
        lengthscale = np.random.uniform(3, 15)  # 10-50x peak width
        K = RBF(length_scale=lengthscale)(t.reshape(-1,1))
        baseline = np.random.multivariate_normal(np.zeros(length), K)
        baseline = baseline * np.random.uniform(0.5, 5)  # scale
    elif baseline_type == 'polynomial':
        order = np.random.randint(2, 6)
        coeffs = np.random.randn(order + 1) * np.array([0.1**i for i in range(order+1)])
        baseline = np.polyval(coeffs, np.linspace(-1, 1, length))
    # ... (spline, sinusoidal similar)
    
    # 3. Artifacts (probabilistic)
    if np.random.random() < 0.3:  # column bleed
        baseline += np.random.uniform(0.5, 3) * np.exp(0.1 * np.linspace(0, 1, length))
    if np.random.random() < 0.2:  # solvent front
        peaks += np.random.uniform(5, 50) * np.exp(-np.linspace(0, 10, length))
    
    # 4. Noise
    snr_db = np.random.uniform(10, 60)
    noise_std = np.max(peaks) / (10 ** (snr_db / 20))
    noise = np.random.randn(length) * noise_std
    
    signal = peaks + baseline + noise
    return signal, peaks, baseline
The EMG peak model is the gold standard for chromatographic simulation, validated by Grushka (Anal. Chem., 1972) and Naish & Hartwell (Chromatographia, 1988). Kensert et al. (J. 
ScienceDirect
 Chromatography A, 2021) trained on 190,000 simulated chromatograms using a similar generator and achieved good transfer to real data. 
ScienceDirect
 For maximum realism, merge synthetic peaks with real blank chromatograms (Kanazawa et al., J. Biosci. Bioeng., 2020).

Domain randomization checklist
Randomize everything that varies between labs and instruments:

Peak density: Poisson(λ) with λ ∈ [3, 100]
Peak widths: σ ∈ [0.01, 0.5] min, with optional linear broadening σ(tR) = a + b·tR
Peak asymmetry: τ/σ ∈ [0, 5]
Amplitude range: 3–4 orders of magnitude within a single chromatogram
Baseline shape: polynomials, GPs (lengthscale 3–15 min), splines (5–15 knots), sinusoids
Baseline amplitude: 0.1–10× mean peak amplitude
SNR: 10–60 dB, heteroscedastic (σ_noise = α + β·√signal)
Artifacts: column bleed (30% probability), gradient steps (20%), solvent front (20%), injection spike (10%)
Signal length: 1024–8192 points
Curriculum learning schedule
python
curriculum = {
    'stage1': {  # Epochs 1-30: easy
        'n_peaks': (3, 10), 'snr': (40, 60), 'baseline': 'polynomial',
        'peak_overlap': 'none'
    },
    'stage2': {  # Epochs 30-60: medium  
        'n_peaks': (10, 30), 'snr': (20, 40), 'baseline': 'gp+spline',
        'peak_overlap': 'moderate'
    },
    'stage3': {  # Epochs 60-100+: hard
        'n_peaks': (30, 100), 'snr': (10, 20), 'baseline': 'gp+artifacts',
        'peak_overlap': 'heavy'
    },
}
Use a difficulty score d = α·(peak_density) + β·(1/SNR) + γ·(baseline_complexity) and pacing function that linearly increases the sampling probability of harder examples (Hacohen & Weinshall, ICML 2019).

Transfer to real data
Three-stage transfer (validated in spectroscopy literature):

Pre-train on 500K+ synthetic chromatograms
Fine-tune on semi-supervised real data: use classical methods (arPLS, SNIP) to generate pseudo-labels, keep only high-confidence estimates
Self-supervised refinement: enforce non-negativity of peaks, smoothness of baseline, reconstruction consistency on unlabeled real chromatograms
7. Evaluation metrics and baseline leakage detection
Supervised metrics (synthetic test set)
Metric	Formula	Purpose
RMSE_baseline	√(Σ(b̂ᵢ - bᵢ)²/N)	Overall baseline accuracy
MAE_baseline	Σ|b̂ᵢ - bᵢ|/N	Robust baseline accuracy
Peak-region RMSE	RMSE computed only within ±3σ of each peak	Directly measures leakage
Peak area preservation	|Â_peak - A_peak| / A_peak × 100%	Quantitative accuracy
Peak height preservation	|Ĥ - H| / H × 100%	Detection sensitivity
L∞ (max error)	max|b̂ᵢ - bᵢ|	Worst-case leakage
SSIM	Structural similarity of corrected signal	Used by DIRAS+ (2025)
Baseline Leakage Index (BLI) — proposed metric
python
def baseline_leakage_index(baseline_pred, baseline_true, peak_signal, peak_mask):
    """BLI > 0 indicates baseline absorbing peak energy.
    Compute separately for dense and sparse regions."""
    residual = baseline_pred - baseline_true  # positive = over-estimation
    peak_energy = (peak_signal * peak_mask).sum()
    leakage = (F.relu(residual) * peak_mask).sum()  # only count over-estimation in peak regions
    return leakage / (peak_energy + 1e-8)
Stratified evaluation: Classify regions as "sparse" (<1 peak per window) or "dense" (>3 overlapping peaks per window) and report all metrics separately. The dense-region penalty = metric_dense / metric_sparse − 1 quantifies how much worse the method performs in the problematic regions.

Unsupervised metrics (real data without ground truth)
Non-negativity fraction: Fraction of corrected signal that is negative (should be ~0 + noise floor)
Baseline smoothness: TV(b̂) and ||D²b̂||₂² — compare to classical method baselines
Residual autocorrelation: In peak-free regions, residuals should be white noise (Durbin-Watson test)
Peak detection F1: Apply peak detection to corrected signal, compare to reference peak list
SNR improvement: ΔSNR = SNR_after − SNR_before
Peak-to-baseline error ratio
python
def peak_baseline_error_ratio(baseline_pred, baseline_true, peak_mask):
    """Ratio > 1 indicates worse performance in peak regions = leakage."""
    rmse_peak = rmse(baseline_pred[peak_mask], baseline_true[peak_mask])
    rmse_base = rmse(baseline_pred[~peak_mask], baseline_true[~peak_mask])
    return rmse_peak / (rmse_base + 1e-8)
The CAE+ paper (Sensors, 2024) found that airPLS and polynomial methods show 2–3× higher error in peak regions vs. baseline regions, while their neural method maintained consistent performance. This ratio is the simplest and most diagnostic metric for baseline leakage.

8. Concrete implementation plan for Claude Code
Phase 1: Quick wins (1–2 days)
These changes require minimal architectural modification and are likely to produce the largest immediate improvement:

1a. Add asymmetric loss. Implement asymmetric_loss with α=0.9. Add to existing loss with weight λ_asym=0.5. This single change should significantly reduce over-estimation.

1b. Add orthogonality penalty. Implement L_ortho = ||x_peak ⊙ x_base||₁ with λ_orth=0.05. Forces the network to choose: is this peak or baseline?

1c. Enforce non-negativity architecturally. Replace the final peak output with x_peak = F.softplus(z_peak, beta=5). Peaks are always physically non-negative.

1d. Add envelope constraint. Implement soft_local_min and penalize baseline exceeding it. Window = 3× typical peak FWHM, λ_env=1.0.

1e. Compute peaks as residual. Always use x_peak = y - x_base rather than estimating peaks directly. This enforces perfect reconstruction.

Phase 2: Architectural improvements (3–5 days)
2a. Layer-specific parameters. Make λ₀, λ₁, λ₂ learnable per unrolling stage. Initialize at classical BEADS values.

2b. Signal-adaptive parameter prediction. Add SignalAdaptiveParams module that takes the input signal and outputs per-signal adjustments to regularization weights. This allows the network to reduce sparsity penalty in dense-peak regions.

2c. Replace soft-thresholding with learned proximal. Implement LearnedProximal (3-layer 1D CNN, 32 channels). Use residual structure: output = soft_threshold(x) + CNN(x). Initialize CNN at zero output.

2d. Add intermediate supervision. Apply loss at every unrolled stage with linearly increasing weights.

2e. Add frequency-domain separation loss. Implement freq_separation_loss matching the BEADS filter cutoff.

Phase 3: Data and training improvements (2–3 days)
3a. Build comprehensive data generator. Implement the EMG-based generator with full domain randomization. Target: 500K training chromatograms.

3b. Implement curriculum learning. Three-stage difficulty progression over 100 epochs.

3c. Implement stratified evaluation. BLI, peak-region RMSE, peak area preservation, dense-region penalty.

3d. Benchmark against pybaselines methods. Compare LBEADS-NET against arPLS, SNIP, airPLS, and vanilla BEADS on both synthetic and real data.

Phase 4: Advanced architecture (5–7 days, if Phase 1–3 insufficient)
4a. Conformer-based proximal operator. Replace the learned proximal CNN with a 2-layer Conformer block (d_model=64, n_heads=4, conv_kernel=31). This captures both local peak structure and global baseline context.

4b. Gated skip connections. In any encoder-decoder components, replace standard skip connections with learned gates: skip_out = sigmoid(gate_conv(skip)) * skip. Prevents peak-like features from leaking into the baseline path.

4c. ADMM-style reformulation. Reformulate the unrolled BEADS as ADMM with explicit variable splitting for peaks and baseline.

4d. Multi-scale processing. Add a WaveNet-style dilated convolution backbone (dilations 1,2,4,...,512) before the unrolled stages for multi-scale feature extraction.

Phase 5: Alternative architectures to evaluate (research track)
5a. Pure end-to-end 1D U-Net. Wave-U-Net style with difference output layer. Baseline: ~20 layers, 64 base channels, skip connections. May outperform unrolled approaches for maximum flexibility.

5b. DEQ formulation. Convert LBEADS-NET to deep equilibrium model using TorchDEQ. Iterate until convergence per-signal.

5c. Mamba-based backbone. Replace convolutions with Mamba blocks for linear-complexity long-range modeling.

Hyperparameter starting points
python
config = {
    # Loss weights
    'lambda_recon': 1.0,
    'lambda_sparse': 0.1,       # L1 on peaks
    'lambda_smooth': 10.0,      # L2 on D² baseline
    'lambda_ortho': 0.05,       # orthogonality penalty
    'lambda_asym': 0.5,         # asymmetric loss
    'alpha_asym': 0.9,          # asymmetry ratio (9:1)
    'lambda_nonneg': 1.0,       # non-negativity (or use architectural)
    'lambda_freq': 0.1,         # frequency separation
    'lambda_envelope': 1.0,     # envelope constraint
    
    # Architecture
    'n_stages': 10,             # unrolling depth
    'proximal_hidden': 32,      # learned proximal channels
    'proximal_layers': 3,       # learned proximal depth
    'adaptive_params': True,    # signal-adaptive regularization
    
    # Training
    'lr': 1e-3,                 # with cosine annealing
    'batch_size': 32,
    'epochs': 100,
    'curriculum': True,
    'intermediate_supervision': True,
    
    # Data
    'train_size': 500000,
    'signal_length': 4096,
    'peak_model': 'emg',
    'baseline_model': 'gp+polynomial+artifacts',
}
Key implementation pitfalls to avoid
Degenerate solutions. The orthogonality penalty can cause the network to set both components to zero. Counter with strong reconstruction loss weight (keep at 1.0) and non-zero minimum for peak sparsity.

Gradient collapse from detaching. The adaptive smoothness loss must .detach() the peak estimate to avoid the network zeroing out peaks to minimize smoothness cost. Similarly, the envelope constraint should detach the local minimum computation.

Over-regularization. If too many penalty terms have high weights, the network under-fits. Start with only reconstruction + asymmetric + orthogonality, then gradually add others.

Training instability from intermediate supervision. Start with low weights (0.1) on early-stage losses and increase gradually during training. The first few stages produce poor estimates that shouldn't dominate the gradient.

Conclusion: the minimal viable fix and the full fix
The minimal viable fix for baseline leakage requires three changes: (1) an asymmetric loss with α=0.9 that penalizes baseline over-estimation 9× more than under-estimation, (2) an element-wise orthogonality penalty ||peaks ⊙ baseline||₁ forcing mutual exclusivity, and (3) architectural non-negativity on peaks via softplus. These three changes directly address the three failure modes — the asymmetric loss prevents the network from preferring to put energy into the baseline, the orthogonality penalty prevents spatial overlap, and non-negativity prevents negative-peak artifacts that enable leakage.

The full fix additionally requires signal-adaptive regularization (a small encoder that predicts per-signal λ adjustments, allowing the network to relax sparsity in dense-peak regions), a learned proximal operator (replacing rigid soft-thresholding with a 3-layer CNN), and intermediate supervision (loss at every unrolled stage). The Conformer-based proximal operator represents the most promising advanced architectural change, as it captures both local peak structure and global baseline context within a single module. Training should use EMG-peak synthetic data with GP baselines, curriculum learning from sparse to dense peaks, and transfer to real data via pseudo-labeling. Evaluate using the Baseline Leakage Index and stratified peak-region RMSE as the primary diagnostic metrics — these directly measure the failure mode that motivated this work.

