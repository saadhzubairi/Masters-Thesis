"""
Example: Chromatograms BEADS (Baseline Estimation And Denoising with Sparsity)

This example illustrates the use of BEADS to estimate and remove the
baseline of chromatogram series.

Reference:
'BEADS: Joint baseline estimation and denoising of chromatograms using
sparse derivatives'

Xiaoran Ning, Ivan Selesnick,
Polytechnic School of Engineering, New York University, Brooklyn, NY, USA

Laurent Duval,
IFP Energies nouvelles, Technology Division, Rueil-Malmaison, France,
Universite Paris-Est, LIGM, ESIEE Paris, France

2014
"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import time
import os

from beads import beads

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(os.path.dirname(script_dir), 'data')

# Load data
noise_data = sio.loadmat(os.path.join(data_dir, 'noise.mat'))
chromatograms_data = sio.loadmat(os.path.join(data_dir, 'chromatograms.mat'))

# Print info about loaded data (equivalent to whos)
print("Loaded data:")
print(f"noise: shape = {noise_data['noise'].shape}")
print(f"X: shape = {chromatograms_data['X'].shape}")

noise = noise_data['noise'].flatten()
X = chromatograms_data['X']

# Print noise and X (equivalent to print(noise) and print(X) in MATLAB)
print("\nnoise:")
print(noise)
print("\nX:")
print(X)

# Load data and add noise
# y = X(:, 3) + noise * 0.5  (MATLAB uses 1-based indexing, so column 3 is index 2 in Python)
y = X[:, 2] + noise * 0.5

N = len(y)
print(f"\nSignal length N = {N}")

# Run the BEADS algorithm

# Filter parameters
fc = 0.006      # fc : cut-off frequency (cycles/sample)
d = 1           # d : filter order parameter (d = 1 or 2)

# Positivity bias (peaks are positive)
r = 6           # r : asymmetry parameter

# Regularization parameters
amp = 0.8
lam0 = 0.5 * amp
lam1 = 5 * amp
lam2 = 4 * amp

print("\nRunning BEADS algorithm...")
start_time = time.time()
x1, f1, cost = beads(y, d, fc, r, lam0, lam1, lam2)
elapsed_time = time.time() - start_time
print(f"Elapsed time: {elapsed_time:.4f} seconds")

# Convert tensors to numpy for plotting
x1 = x1.numpy()
f1 = f1.numpy()
cost = cost.numpy()

# Display the output of BEADS
ylim1 = [-50, 200]
xlim1 = [0, 3800]

fig1, axes = plt.subplots(4, 1, figsize=(10, 12))

# Subplot 1: Data
axes[0].plot(y)
axes[0].set_title('Data')
axes[0].set_xlim(xlim1)
axes[0].set_ylim(ylim1)
axes[0].set_yticks(ylim1)

# Subplot 2: Baseline
axes[1].plot(y, color=[0.7, 0.7, 0.7], label='Data')
axes[1].plot(np.arange(N), f1, linewidth=1, label='Baseline')
axes[1].legend(frameon=False)
axes[1].set_title(f'Baseline, as estimated by BEADS (r = {r}, fc = {fc}, d = {d})')
axes[1].set_xlim(xlim1)
axes[1].set_ylim(ylim1)
axes[1].set_yticks(ylim1)

# Subplot 3: Baseline-corrected data
axes[2].plot(x1)
axes[2].set_title('Baseline-corrected data')
axes[2].set_xlim(xlim1)
axes[2].set_ylim(ylim1)
axes[2].set_yticks(ylim1)

# Subplot 4: Residual
axes[3].plot(y - x1 - f1)
axes[3].set_title('Residual')
axes[3].set_xlim(xlim1)
axes[3].set_ylim(ylim1)
axes[3].set_yticks(ylim1)

plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'example.pdf'))
print(f"\nSaved figure to {os.path.join(script_dir, 'example.pdf')}")

# Display cost function history
fig2, ax = plt.subplots(figsize=(8, 6))
ax.plot(cost)
ax.set_xlabel('iteration number')
ax.set_ylabel('Cost function value')
ax.set_title('Cost function history')

plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'cost_history.pdf'))
print(f"Saved cost history to {os.path.join(script_dir, 'cost_history.pdf')}")

plt.show()
