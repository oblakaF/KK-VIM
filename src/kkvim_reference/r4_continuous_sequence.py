"""QA reference candidate for R4, s >= 1: the continuous N-term KK-VIM sequence by the matrix Parseval recursion.

R4_NUMERICAL_SPEC 9.2 (ЧВ4-18; principle accepted by D-046, РВ4-8; qualification NOT_QUALIFIED until QA). No grid
operators, no grid sums and no import of ``kkvim`` (D-028). Every set is a finite sine series

    X_j(x) = sum_{m odd <= M} x_mj sin(kappa_m x),  kappa_m = m pi / a;   Y_j(y) = sum_{n odd <= M} y_nj sin(nu_n y),

(odd modes only: q = const and the odd-sine start keep every function symmetric, MATH_SPEC ВБ-22, item 6).
Y-half-step with the given x-set (MATH_SPEC part B, (Ф-2) with Y~ = e_i sin(nu_n y); orthogonality of the sines):

    a^X = (a/2) x^T x,  b^X = (a/2) x^T diag(kappa^2) x,  g^X = (a/2) x^T diag(kappa^4) x,  F^X = q x^T (2a/(m pi)),
    y_n = (4 / (n pi D)) (nu_n^4 a^X + 2 nu_n^2 b^X + g^X)^(-1) F^X,

and symmetrically for the X-half-step. The method is the original procedure: every half-step determines all N
functions together from one coupled system; the given set may be replaced by an orthonormal basis of its span (the
field and the energy do not change, MATH_SPEC ВБ-16).

Independent cross-check of the same truncated problem (``ls_halfstep_*``): the half-step field is the projection of
omega_0 in the energy norm; with the sine coefficients w_mn of omega_0 (L-107 (131)) and the weights
lambda_mn = (m^2/a^2 + n^2/b^2)^2 the Y-half-step is, for every y-mode n, the weighted least-squares problem
min sum_{m <= M} lambda_mn (x_m . y_n - w_mn)^2, solved by Householder QR. Its normal equations are algebraically the
recursion above; the two computations share no code path (no Parseval sums, no N x N normal matrix).

Truncation: for exact input data, |y_jn| <= 4 |F|_2 / (n pi D nu_n^4 lambda_min(a^X)), hence
sup_y |Y(y) - Y_M(y)|_2 <= 4 |F|_2 b^4 / (pi^5 D lambda_min(a^X)) * 1/(8 M^4) (sum over odd n > M of n^-5 <= 1/(8 M^4)).
The transfer of errors through the half-steps is not bounded here (R4_NUMERICAL_SPEC 9.2: UNRESOLVED, Q-022);
practical control: doubling M (diagnostic, not a proof).
"""

import math
from dataclasses import dataclass

import numpy as np

__all__ = ["ContinuousStep", "continuous_sequence", "omega0_coefficients", "sine_values", "field_at_nodes"]


def _odd(M):
    if isinstance(M, bool) or not isinstance(M, int) or M < 1 or M % 2 == 0:
        raise ValueError(f"M must be a positive odd integer, got {M!r}")
    return np.arange(1, M + 1, 2, dtype=float)


def _gram(coef, length, k):
    """(length/2) coef^T diag(k^p) coef for p = 0, 2, 4 by exact Parseval sums; entries summed with math.fsum."""
    N = coef.shape[1]
    out = []
    for p in (0, 2, 4):
        w = k**p
        G = np.empty((N, N))
        for i in range(N):
            for j in range(i, N):
                G[i, j] = G[j, i] = 0.5 * length * math.fsum((w * coef[:, i] * coef[:, j]).tolist())
        out.append(G)
    return out


def _load(coef, length, modes, q):
    return np.array([q * math.fsum((coef[:, j] * 2.0 * length / (modes * math.pi)).tolist())
                     for j in range(coef.shape[1])])


def _halfstep(coef, length_given, modes_given, length_new, modes_new, q, D):
    k = modes_given * math.pi / length_given
    a_, b_, g_ = _gram(coef, length_given, k)
    F = _load(coef, length_given, modes_given, q)
    nu = modes_new * math.pi / length_new
    M = nu[:, None, None] ** 4 * a_ + 2.0 * nu[:, None, None] ** 2 * b_ + g_
    rhs = (4.0 / (modes_new * math.pi * D))[:, None] * F[None, :]
    new = np.linalg.solve(M, rhs[:, :, None])[:, :, 0]
    lam_min = float(np.linalg.eigvalsh(a_)[0])
    tail = 4.0 * float(np.linalg.norm(F)) * length_new**4 / (math.pi**5 * D * lam_min) / (8.0 * modes_new[-1] ** 4) \
        if lam_min > 0 else math.inf
    return new, (a_, b_, g_, F), tail


def omega0_coefficients(a, b, q, D, M):
    """w_mn = 16 q / (pi^6 D m n (m^2/a^2 + n^2/b^2)^2), odd m, n <= M (L-107 (131), Navier)."""
    m = _odd(M)
    return 16.0 * q / (math.pi**6 * D * np.outer(m, m) * (m[:, None] ** 2 / a**2 + m[None, :] ** 2 / b**2) ** 2)


def _ls(weights_sqrt, basis, target):
    A = weights_sqrt[:, :, None] * basis[None, :, :]
    Q, R = np.linalg.qr(A, mode="reduced")
    y = np.einsum("mkn,mk->mn", Q, weights_sqrt * target)
    return np.linalg.solve(R, y[:, :, None])[:, :, 0]


@dataclass(frozen=True)
class ContinuousStep:
    s: int
    k: int
    direction: str
    x: np.ndarray  # sine coefficients of the x-set (odd modes), representation dependent
    y: np.ndarray  # sine coefficients of the y-set
    energy: float  # Pi(w_s) = 1/2 a(w, w) - q int w, by Parseval
    integral: float  # int w over the plate
    tail_bound: float  # sup norm bound of the truncation of the computed set (one half-step, exact input)
    given_cond: float  # sigma_1/sigma_N of the given set (unit columns), before orthonormalisation


def _energy(x, y, a, b, modes, q, D):
    ax, bx, gx = _gram(x, a, modes * math.pi / a)
    ay, by, gy = _gram(y, b, modes * math.pi / b)
    norm2 = D * math.fsum((gx * ay + 2.0 * bx * by + ax * gy).ravel().tolist())
    Ix = np.array([math.fsum((x[:, j] * 2.0 * a / (modes * math.pi)).tolist()) for j in range(x.shape[1])])
    Iy = np.array([math.fsum((y[:, j] * 2.0 * b / (modes * math.pi)).tolist()) for j in range(y.shape[1])])
    integral = math.fsum((Ix * Iy).tolist())
    return 0.5 * norm2 - q * integral, integral


def _cond(X):
    s = np.linalg.svd(X / np.linalg.norm(X, axis=0), compute_uv=False)
    return float(s[0] / s[-1])


def continuous_sequence(a, b, N, s_max, M, *, q=1.0, D=1.0, start=None, orthonormalize=True, method="recursion"):
    """Continuous half-steps s = 1..s_max with odd modes <= M in both directions.

    start -- sine coefficients (odd modes) of the start x-set, default the odd sines (unit vectors);
    method -- "recursion" (R4_NUMERICAL_SPEC 9.2) or "ls" (weighted least squares on omega_0, independent check).
    """
    modes = _odd(M)
    K = len(modes)
    if N > K:
        raise ValueError("N exceeds the number of odd modes")
    x = np.eye(K, N) if start is None else np.asarray(start, dtype=float)
    if x.shape != (K, N):
        raise ValueError(f"start must have shape {(K, N)}")
    if method == "ls":
        lam_sqrt = (modes[:, None] ** 2 / a**2 + modes[None, :] ** 2 / b**2)  # sqrt of lambda_mn
        w0 = omega0_coefficients(a, b, q, D, M)
    elif method != "recursion":
        raise ValueError(f"unknown method {method!r}")
    y = None
    steps = []
    for s in range(1, s_max + 1):
        if s % 2:
            cond = _cond(x)
            if orthonormalize:
                x, _ = np.linalg.qr(x)
            if method == "recursion":
                y, _, tail = _halfstep(x, a, modes, b, modes, q, D)
            else:
                y, tail = _ls(lam_sqrt.T, x, w0.T), math.nan
        else:
            cond = _cond(y)
            if orthonormalize:
                y, _ = np.linalg.qr(y)
            if method == "recursion":
                x, _, tail = _halfstep(y, b, modes, a, modes, q, D)
            else:
                x, tail = _ls(lam_sqrt, y, w0), math.nan
        energy, integral = _energy(x, y, a, b, modes, q, D)
        steps.append(ContinuousStep(s=s, k=s // 2, direction="Y" if s % 2 else "X", x=x.copy(), y=y.copy(),
                                    energy=energy, integral=integral, tail_bound=tail, given_cond=cond))
    return steps


def sine_values(coef, n):
    """Values of sum_{m odd} coef_m sin(m pi i / n) at the nodes i = 0..n (integer reduction of the argument)."""
    K = coef.shape[0]
    modes = np.arange(1, 2 * K, 2)
    i = np.arange(n + 1)
    r = np.mod(np.outer(i, modes), 2 * n)
    sign = np.where(r < n, 1.0, -1.0)
    r1 = np.mod(r, n)
    S = sign * np.sin(np.pi * np.minimum(r1, n - r1) / n)
    return S @ coef


def field_at_nodes(step, nx, ny):
    """w_s at all nodes x_i = i a/nx, y_j = j b/ny (edges included, exact zeros)."""
    return sine_values(step.x, nx) @ sine_values(step.y, ny).T
