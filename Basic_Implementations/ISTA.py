import numpy as np
import matplotlib.pyplot as plt


def soft_threshold(v, tau):
    """
    Element-wise soft-thresholding:
        S_tau(v) = sign(v) * max(|v| - tau, 0).
    """
    return np.sign(v) * np.maximum(np.abs(v) - tau, 0.0)


def ista(A, y, lam, max_iter=300, step_size=None, x0=None, x_true=None, verbose=False):
    """
    ISTA for L1-regularized least squares (LASSO):

        minimize_x  0.5 * ||A x - y||_2^2 + lam * ||x||_1

    Parameters
    ----------
    A : ndarray, shape (m, n)
        Measurement / design matrix.
    y : ndarray, shape (m,)
        Observed data.
    lam : float
        L1 regularization parameter.
    max_iter : int
        Number of ISTA iterations.
    step_size : float or None
        Gradient step size. If None, use 1 / ||A^T A||_2.
    x0 : ndarray or None
        Initial guess for x. If None, start from zeros.
    x_true : ndarray or None
        Ground-truth x (for MSE tracking). If None, MSE history is empty.
    verbose : bool
        If True, prints occasional debug info.

    Returns
    -------
    x : ndarray, shape (n,)
        Final estimate.
    obj_history : list of float
        Objective value per iteration.
    mse_history : list of float
        MSE to x_true per iteration (empty if x_true is None).
    """
    m, n = A.shape

    if x0 is None:
        x = np.zeros(n)
    else:
        x = x0.copy()

    AT = A.T

    # If step size not provided, pick 1 / Lipschitz constant of grad f
    # L = ||A^T A||_2, we approximate via spectral norm of A
    if step_size is None:
        s_max = np.linalg.norm(A, 2)  # spectral norm of A
        L = s_max**2
        step_size = 1.0 / (L + 1e-12)

    if verbose:
        print(f"[ISTA] Using step_size = {step_size:.3e}")

    obj_history = []
    mse_history = []

    for k in range(max_iter):
        # Gradient of 0.5 * ||A x - y||^2 is A^T (A x - y)
        grad = AT @ (A @ x - y)

        # Gradient step
        z = x - step_size * grad

        # Proximal step: soft-thresholding
        x = soft_threshold(z, lam * step_size)

        # Objective
        residual = A @ x - y
        obj = 0.5 * np.dot(residual, residual) + lam * np.sum(np.abs(x))
        obj_history.append(obj)

        # MSE (if true x provided)
        if x_true is not None:
            mse = np.mean((x - x_true) ** 2)
            mse_history.append(mse)

        if verbose and (k % 50 == 0 or k == max_iter - 1):
            if x_true is not None:
                print(f"[ISTA] iter {k:4d}, obj = {obj:.4e}, MSE = {mse:.4e}")
            else:
                print(f"[ISTA] iter {k:4d}, obj = {obj:.4e}")

    return x, obj_history, mse_history


def generate_sparse_signal(n=200, sparsity=0.1, seed=0):
    """
    Generate a 1D sparse signal of length n with given sparsity fraction.
    """
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    k = int(sparsity * n)
    idx = rng.choice(n, size=k, replace=False)
    x[idx] = rng.normal(loc=0.0, scale=1.0, size=k)
    return x


def main():
    # --------------------------
    # 1. Problem setup
    # --------------------------
    rng = np.random.default_rng(42)

    n = 200          # number of unknowns
    m = 120          # number of measurements (m < n -> underdetermined)
    sparsity = 0.1   # fraction of nonzero entries in x_true

    # Ground-truth sparse signal
    x_true = generate_sparse_signal(n=n, sparsity=sparsity, seed=1)

    # Measurement matrix A: Gaussian, column-normalized
    A = rng.normal(size=(m, n))
    A /= np.linalg.norm(A, axis=0, keepdims=True) + 1e-12

    # Noise added to measurements
    noise_std = 0.05
    noise = rng.normal(scale=noise_std, size=m)

    # Observations
    y = A @ x_true + noise

    # L1 regularization parameter
    lam = 0.05

    # --------------------------
    # 2. Run ISTA
    # --------------------------
    max_iter = 300
    x0 = np.zeros(n)

    x_hat, obj_hist, mse_hist = ista(
        A,
        y,
        lam,
        max_iter=max_iter,
        step_size=None,
        x0=x0,
        x_true=x_true,
        verbose=False,
    )

    # --------------------------
    # 3. Print basic stats
    # --------------------------
    final_mse = np.mean((x_hat - x_true) ** 2)
    print(f"Final MSE: {final_mse:.4e}")
    print(f"True nonzeros: {(np.abs(x_true) > 1e-6).sum()}, "
          f"Estimated nonzeros: {(np.abs(x_hat) > 1e-3).sum()}")

    # --------------------------
    # 4. Plots
    # --------------------------
    # (a) True vs ISTA-recovered signal
    plt.figure(figsize=(12, 4))
    plt.stem(
        x_true,
        linefmt="C0-",
        markerfmt="C0o",
        basefmt="k-",
        label="True x",
    )
    plt.stem(
        x_hat,
        linefmt="C1--",
        markerfmt="C1x",
        basefmt="k-",
        label="ISTA estimate",
    )
    plt.title("True vs ISTA-Recovered Sparse Signal")
    plt.xlabel("Index")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True, linestyle=":")

    plt.tight_layout()
    plt.show()

    # (b) Objective + MSE vs iteration
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.semilogy(obj_hist)
    plt.title("Objective vs Iteration (ISTA)")
    plt.xlabel("Iteration")
    plt.ylabel("Objective (log scale)")
    plt.grid(True, linestyle=":")

    if len(mse_hist) > 0:
        plt.subplot(1, 2, 2)
        plt.semilogy(mse_hist)
        plt.title("MSE to Ground Truth vs Iteration")
        plt.xlabel("Iteration")
        plt.ylabel("MSE (log scale)")
        plt.grid(True, linestyle=":")
    else:
        plt.subplot(1, 2, 2)
        plt.text(0.5, 0.5, "No x_true provided,\nno MSE history.",
                 ha="center", va="center", fontsize=12)
        plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
