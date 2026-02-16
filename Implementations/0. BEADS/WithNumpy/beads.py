import numpy as np


def BAfilt(d_filt, fc, N):
    """
    Python translation of BAfilt(d, fc, N) from the MATLAB code.
    Builds dense (N x N) matrices A, B corresponding to the
    zero-phase high-pass filter.
    """
    # b1 = [1 -1]
    b1 = np.array([1.0, -1.0])
    for _ in range(d_filt - 1):
        b1 = np.convolve(b1, np.array([-1.0, 2.0, -1.0]))

    # b = conv(b1, [-1 1])
    b = np.convolve(b1, np.array([-1.0, 1.0]))

    omc = 2.0 * np.pi * fc
    t = ((1.0 - np.cos(omc)) / (1.0 + np.cos(omc))) ** d_filt

    a = np.array([1.0])
    for _ in range(d_filt):
        a = np.convolve(a, np.array([1.0, 2.0, 1.0]))

    a = b + t * a

    # A = spdiags(a(ones(N,1),:), -d:d, N, N)
    # B = spdiags(b(ones(N,1),:), -d:d, N, N)
    A = np.zeros((N, N), dtype=float)
    B = np.zeros((N, N), dtype=float)

    offsets = np.arange(-d_filt, d_filt + 1)  # [-d, ..., d]
    # a and b both have length 2*d_filt + 1
    for idx, k in enumerate(offsets):
        if k < 0:
            # diagonal below main: row starts at -k, col at 0
            i = -k
            j = 0
            length = N + k  # k is negative
        else:
            # diagonal above main: row starts at 0, col at k
            i = 0
            j = k
            length = N - k

        if length <= 0:
            continue

        A[i:i + length, j:j + length] += np.diag(np.full(length, a[idx]))
        B[i:i + length, j:j + length] += np.diag(np.full(length, b[idx]))

    return A, B


def beads(y, d, fc, r, lam0, lam1, lam2,
          Nit=30, pen='L1_v2', EPS0=1e-6, EPS1=1e-6):
    """
    Python/NumPy translation of:

        [x, f, cost] = beads(y, d, fc, r, lam0, lam1, lam2)

    from the MATLAB BEADS implementation.
    """

    # ensure 1D float64
    y = np.asarray(y, dtype=float).flatten()
    N = y.size

    x = y.copy()
    cost = np.zeros(Nit, dtype=float)

    # ----- penalty selection -----
    if pen == 'L1_v1':
        def phi(z):
            return np.sqrt(np.abs(z) ** 2 + EPS1)

        def wfun(z):
            return 1.0 / np.sqrt(np.abs(z) ** 2 + EPS1)

    elif pen == 'L1_v2':
        def phi(z):
            return np.abs(z) - EPS1 * np.log(np.abs(z) + EPS1)

        def wfun(z):
            return 1.0 / (np.abs(z) + EPS1)

    else:
        raise ValueError("penalty must be 'L1_v1' or 'L1_v2'")

    # ----- theta(x) -----
    def theta_func(x_vec):
        x_vec = np.asarray(x_vec)
        pos = x_vec > EPS0
        neg = x_vec < -EPS0
        mid = ~(pos | neg)  # |x| <= EPS0

        term_pos = x_vec[pos].sum()
        term_neg = x_vec[neg].sum()

        c = (1.0 + r) / (4.0 * EPS0)
        term_mid = (
            c * x_vec[mid] ** 2 +
            (1.0 - r) / 2.0 * x_vec[mid] +
            EPS0 * (1.0 + r) / 4.0
        ).sum()

        return term_pos - r * term_neg + term_mid

    # ----- operators A, B, H -----
    A, B = BAfilt(d, fc, N)

    def H(vec):
        # H(x) = B * (A \ x)
        return B @ np.linalg.solve(A, vec)

    # ----- first and second difference operators -----
    if N < 3:
        raise ValueError("N must be >= 3 for D2 (second difference) to be defined.")

    # D1: (N-1) x N, forward difference
    e1 = np.ones(N - 1)
    D1 = np.zeros((N - 1, N))
    D1[np.arange(N - 1), np.arange(N - 1)] = -e1
    D1[np.arange(N - 1), np.arange(1, N)] = e1

    # D2: (N-2) x N, second difference
    e2 = np.ones(N - 2)
    D2 = np.zeros((N - 2, N))
    D2[np.arange(N - 2), np.arange(N - 2)] = e2
    D2[np.arange(N - 2), np.arange(1, N - 1)] = -2.0 * e2
    D2[np.arange(N - 2), np.arange(2, N)] = e2

    # D = [D1; D2]
    D = np.vstack([D1, D2])  # (2N-3) x N

    # BTB = B' * B
    BTB = B.T @ B

    # w = [lam1 * ones(N-1); lam2 * ones(N-2)]
    w = np.concatenate([
        lam1 * np.ones(N - 1),
        lam2 * np.ones(N - 2)
    ])

    # b = (1-r)/2 * ones(N,1)
    b = (1.0 - r) / 2.0 * np.ones(N)

    # d_vec = BTB * (A \ y) - lam0 * A' * b
    d_vec = BTB @ np.linalg.solve(A, y) - lam0 * (A.T @ b)

    gamma = np.ones(N)

    # ===== main MM loop =====
    for it in range(Nit):
        print(f"Iteration {it + 1}/{Nit}")
        Dx = D @ x  # (2N-3)

        # Lambda = spdiags(w .* wfun(D*x), 0, 2N-3, 2N-3)
        Lambda_diag = w * wfun(Dx)
        Lambda = np.diag(Lambda_diag)

        # gamma update
        k = np.abs(x) > EPS0
        gamma[~k] = ((1.0 + r) / 4.0) / abs(EPS0)
        gamma[k] = ((1.0 + r) / 4.0) / np.abs(x[k])

        # Gamma = spdiags(gamma, 0, N, N)
        Gamma = np.diag(gamma)

        # M = 2 * lam0 * Gamma + D' * Lambda * D
        M = 2.0 * lam0 * Gamma + D.T @ Lambda @ D

        # x = A * ((BTB + A' * M * A) \ d_vec)
        K = BTB + A.T @ M @ A
        z = np.linalg.solve(K, d_vec)
        x = A @ z

        # cost(i)
        Hyx = H(y - x)
        cost[it] = (
            0.5 * np.sum(np.abs(Hyx) ** 2) +
            lam0 * theta_func(x) +
            lam1 * np.sum(phi(np.diff(x))) +
            lam2 * np.sum(phi(np.diff(x, n=2)))
        )

    # baseline
    f = y - x - H(y - x)

    return x, f, cost
