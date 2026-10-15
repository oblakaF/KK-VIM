"""QA-δ3 (R4_NUMERICAL_SPEC 5.2, 5.2.1, 5.4.1): rank diagnostics of N-term fields on synthetic factors of known rank.

No import of ``kkvim`` (D-028); independent implementation of the diagnostics of 5.2 (QA does not read the solver):

    W_w = U_w V_w^T, U_w = sqrt(h_x) U, V_w = sqrt(h_y) V;  rho_N = sigma_N / sigma_1 of W_w,
    sigma_k from thin QR U_w = Q_U R_U, V_w = Q_V R_V and the SVD of R_U R_V^T (5.2, item 4);
    kappa_rep = |U_w|_2 |V_w|_2 / |W_w|_2 (5.2, item 5);  set check sigma_N(R_V D_c), D_c = diag(1/|v_w,j|_2) (5.4.1).

Synthetic factors with an EXACTLY known rank of the product of the STORED double factors are built from small integers
(all products and sums exact in binary floating point), and the rank is confirmed by exact rational elimination
(``exact_rank``). This separates the error of the diagnostics itself (c_d u kappa_rep, 5.2 item 6(d)) from the forward
error of the factors (delta_f, 5.2 item 6(f)), which is zero for such factors.

Storage component of delta_f (own, strict in the IEEE model, round to nearest, no underflow): storing exact factors
U*, V* as doubles gives |dU_w|_2 <= |dU_w|_F <= u |U*_w|_F <= u sqrt(N) |U*_w|_2, the same for V, so by 5.2, item 6(f)
    delta_f(storage) <= kappa_rep (2 sqrt(N) u + N u^2).
The constant c_d of Householder QR and of the SVD is NOT derived here (no source in the project library): only its
empirical ratio rho_hat / (u kappa_rep) on the synthetic sets is measured (report, section 5).
"""

import math
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

__all__ = ["RankDiagnostics", "diagnostics", "exact_rank", "storage_delta_f", "cancellation_set", "tiny_rank_set",
           "dense_transform"]

U_ROUND = 2.0**-53


@dataclass(frozen=True)
class RankDiagnostics:
    sigma: np.ndarray  # singular values of W_w (QR path), descending
    rho_N: float
    kappa_rep: float
    set_sigma_min: float  # sigma_N(R_V D_c)
    sigma_full: np.ndarray | None  # singular values by the full SVD of W_w (small sizes only), for comparison


def diagnostics(U, V, hx=1.0, hy=1.0, full=True):
    Uw, Vw = math.sqrt(hx) * np.asarray(U, float), math.sqrt(hy) * np.asarray(V, float)
    _, RU = np.linalg.qr(Uw)
    _, RV = np.linalg.qr(Vw)
    sigma = np.linalg.svd(RU @ RV.T, compute_uv=False)
    kappa = float(np.linalg.norm(RU, 2) * np.linalg.norm(RV, 2) / sigma[0])
    Dc = 1.0 / np.linalg.norm(Vw, axis=0)
    set_min = float(np.linalg.svd(RV * Dc[None, :], compute_uv=False)[-1])
    sf = np.linalg.svd(Uw @ Vw.T, compute_uv=False)[: U.shape[1]] if full else None
    return RankDiagnostics(sigma=sigma, rho_N=float(sigma[-1] / sigma[0]), kappa_rep=kappa, set_sigma_min=set_min,
                           sigma_full=sf)


def exact_rank(U, V):
    """Exact rank of the product of the stored doubles U V^T by rational Gaussian elimination."""
    U = [[Fraction(float(x)) for x in row] for row in np.asarray(U, float)]
    V = [[Fraction(float(x)) for x in row] for row in np.asarray(V, float)]
    W = [[sum((ui * vj for ui, vj in zip(urow, vrow)), Fraction(0)) for vrow in V] for urow in U]
    rank, rows, cols = 0, len(W), len(W[0])
    for c in range(cols):
        pivot = next((r for r in range(rank, rows) if W[r][c] != 0), None)
        if pivot is None:
            continue
        W[rank], W[pivot] = W[pivot], W[rank]
        for r in range(rows):
            if r != rank and W[r][c] != 0:
                f = W[r][c] / W[rank][c]
                W[r] = [x - f * y for x, y in zip(W[r], W[rank])]
        rank += 1
    return rank


def storage_delta_f(kappa_rep, N):
    """Strict bound of delta_f caused only by storing exact factors in double (see the module docstring)."""
    return kappa_rep * (2.0 * math.sqrt(N) * U_ROUND + N * U_ROUND**2)


def _int_vectors(rng, n, k, scale=7):
    return rng.integers(-scale, scale + 1, size=(n, k)).astype(float)


def cancellation_set(n_x, n_y, K, seed=0, N=3):
    """Factors of an exact rank-(N-1) field written with N terms and strong cancellation, kappa_rep ~ K (integers).

    N = 3: U = [u1, u2, u1 + u2], V = [K g1, K g1 + g2, -K g1 + g3]  =>  U V^T = u1 g3^T + u2 (g2 + g3)^T exactly;
    N = 2: U = [u1, u1],          V = [K g1 + g2, -K g1 + g3]          =>  U V^T = u1 (g2 + g3)^T exactly.
    """
    rng = np.random.default_rng(seed)
    u = _int_vectors(rng, n_x, 2)
    g = _int_vectors(rng, n_y, 3)
    if N == 2:
        return (np.column_stack([u[:, 0], u[:, 0]]),
                np.column_stack([K * g[:, 0] + g[:, 1], -K * g[:, 0] + g[:, 2]]))
    if N != 3:
        raise ValueError("N must be 2 or 3")
    U = np.column_stack([u[:, 0], u[:, 1], u[:, 0] + u[:, 1]])
    V = np.column_stack([K * g[:, 0], K * g[:, 0] + g[:, 1], -K * g[:, 0] + g[:, 2]])
    return U, V


def tiny_rank_set(n_x, n_y, e, seed=0):
    """N = 3 integer factors of an exact FULL-rank field with a small third component 2^-e (exact power of two):
    U = [u1, u2, 2^-e u3], V = [g1, g2, g3]; the product of the stored doubles is exact. A large kappa_rep for the
    same field is obtained with ``dense_transform`` (then the stored factors are rounded)."""
    rng = np.random.default_rng(seed)
    u = _int_vectors(rng, n_x, 3)
    g = _int_vectors(rng, n_y, 3)
    return np.column_stack([u[:, 0], u[:, 1], 2.0**-e * u[:, 2]]), g


def dense_transform(N, cond, seed=0):
    """T in GL(N) with prescribed 2-norm condition number (random orthogonal factors, geometric singular values)."""
    rng = np.random.default_rng(seed)
    Q1, _ = np.linalg.qr(rng.standard_normal((N, N)))
    Q2, _ = np.linalg.qr(rng.standard_normal((N, N)))
    s = np.geomspace(1.0, 1.0 / cond, N)
    return Q1 @ np.diag(s) @ Q2
