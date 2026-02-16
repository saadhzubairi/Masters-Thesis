import numpy as np
import matplotlib.pyplot as plt
from beads import beads  # or paste the function above in the same file


def demo_beads():
    N = 400
    t = np.linspace(0.0, 1.0, N)

    # baseline
    baseline = 0.5 * np.sin(2.0 * np.pi * 0.5 * t)

    # peaks
    peak1 = np.exp(-((t - 0.3) ** 2) / (2.0 * 0.001))
    peak2 = 0.7 * np.exp(-((t - 0.7) ** 2) / (2.0 * 0.0005))
    peaks = peak1 + peak2

    # noise
    rng = np.random.default_rng(0)
    noise = 0.05 * rng.standard_normal(N)

    # observed signal
    y = baseline + peaks + noise

    # BEADS params (tweak as needed)
    d = 2        # filter order
    fc = 0.05    # cutoff frequency (cycles/sample)
    r = 6.0      # asymmetry ratio
    lam0 = 0.1
    lam1 = 0.01
    lam2 = 0.01

    x_hat, f_hat, cost = beads(y, d, fc, r, lam0, lam1, lam2,
                               Nit=50, pen='L1_v2')

    print("Final cost:", cost[-1])
    print("Cost monotone? ", np.all(np.diff(cost) <= 1e-8))

    # plotting
    plt.figure(figsize=(10, 6))
    plt.plot(t, y, label="y (noisy)", linewidth=1)
    plt.plot(t, f_hat, label="estimated baseline f", linewidth=2)
    plt.plot(t, x_hat, label="x (sparse-derivative signal)", linewidth=1)
    plt.legend()
    plt.xlabel("t")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    demo_beads()
