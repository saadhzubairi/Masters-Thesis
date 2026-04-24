# LBEADS-NET v8 (Final)

v8 = v5's ISTA architecture + v7's banded O(N) operators + v7's training/loss suite.

## Why

- **ISTA (v5)** has better gradient flow than CG-based unrolling: each layer is a simple gradient step + proximal operator, making backpropagation stable and fast.
- **Banded operators (v7)** give O(N) memory and compute instead of O(N^2) dense matrices, enabling longer signals without memory blowup.
- **v7's training** provides a proven 3-stage curriculum and 11-term sparsity loss that eliminates baseline leakage.

## Architecture

Each of the K unrolled layers performs:
1. **Data fidelity gradient** via banded highpass operator (O(N))
2. **Smoothness penalty gradients** via O(N) difference operators (D1, D2)
3. **Gradient descent step** with learnable step size
4. **Asymmetric soft thresholding** (proximal operator for sparsity)

After all layers, the baseline is extracted via iterated banded lowpass filtering.

5 learnable parameters per layer: lam0, lam1, lam2, r, step_size (all in log space).

## Files

- `lbeads_net.py` -- Model architecture (LBEADS_NET = ISTA primary, LBEADS_NET_CG = CG reference)
- `train.py` -- Training with synthetic data, 3-stage curriculum, MLflow logging
- `demo.py` -- Synthetic data demo (generate signals, run model, plot results)
- `demo_chromatogram.py` -- Real chromatogram demo (BEADS paper data)

## Usage

```bash
# Train
python train.py

# Demo on synthetic data
python demo.py

# Demo on real chromatogram data
python demo_chromatogram.py
```
