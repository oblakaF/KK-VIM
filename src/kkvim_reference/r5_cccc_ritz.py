"""QA reference B for the clamped rectangle (CCCC): Ritz method with the double cosine basis of Taylor & Govindjee (R5-V).

Independent of the main solver: no import of ``kkvim`` (D-028); nothing of Builder is used (D-049.8). Source:
L-120 (P7) = R. L. Taylor, S. Govindjee, Report UCB/SEMM-2002/09 (2002), eqs. (4)-(18), pp. 1-3 of the report (PDF pp. 4-6);
transcription docs/R5_CCCC_REFERENCE_LITERATURE.review.md 4.2; Reviewer 62db9f2 R-16 (system rederived), R-17 (tables);
QA check docs/R5_CCCC_REFERENCES.q0-readiness.md 1.2 (own derivation of (11)-(13), pages compared).

Plate 0 <= x <= a, 0 <= y <= b (FULL sides; a along x), D lap^2 w = q, w = dw/dn = 0. Basis (P7 (4), (8))

    w = sum_{m=1..M} sum_{n=1..N} (1 - cos 2 m pi xi)(1 - cos 2 n pi eta) w_mn,  xi = x/a, eta = y/b,

every term satisfies w = dw/dn = 0. Ritz with the functional (5)/(7) gives K w = b (9)-(13), unknown order
i = M (n - 1) + m (10):
    K_ii^(1) = [m^2 + (a/b)^2 n^2]^2,
    K^(2) = 2 m^4 between all (m, n), (m, p) with the same m (p = 1..N, including p = n),
    K^(3) = 2 (a n / b)^4 between all (m, n), (p, n) with the same n (p = 1..M, including p = m),
    b = q a^4 / (4 pi^4 D) e*.
Own derivation of the same system (``gram_matrix``): with phi_m = 1 - cos 2 m pi xi, int phi_m phi_p = 1 + delta/2,
int phi_m'' phi_p'' = 8 pi^4 m^4 delta, int phi_m'' phi_p = -2 pi^2 m^2 delta; the Ritz matrix of (7) divided by 4 pi^4 is K.

Solution: K = A + U D U^T (14)-(17) with A = diag K^(1), D = diag(2 m^4, 2 (a n/b)^4), U the (MN) x (M + N) incidence of
the m and n groups; w = A^-1 b - A^-1 U (D^-1 + U^T A^-1 U)^-1 U^T A^-1 b (exact algebra of the same system; the capacity
matrix is scaled to unit diagonal as on p. 3 and solved by Cholesky). A dense direct solve is kept for small M N.

Quantities: centre w(a/2, b/2) = 4 sum_{m, n odd} w_mn; load work int int w = a b sum w_mn (int_0^1 phi_m = 1).
Ritz property (standard; level B): with exact integrals Pi(w_MN) = -1/2 q int w_MN >= Pi(omega_0), so the work is a
LOWER bound of the exact work and does not decrease for nested bases. There is no bound for the centre deflection.

Independence (D-049.8, item 2): independent of KK-VIM by the procedure, not by the functional; with M = N in x the
first half-step of KK-VIM with the start T_N coincides with it by construction (comparisons only with M, N -> infinity).
"""

import math
from dataclasses import dataclass

import numpy as np
import scipy.linalg


@dataclass(frozen=True)
class RitzProblem:
    a: float  # full side along x
    b: float  # full side along y
    M: int
    N: int
    D: float = 1.0
    q: float = 1.0

    def __post_init__(self):
        for name in ("M", "N"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v < 1:
                raise ValueError(f"{name} must be an integer >= 1, got {v!r}")
        for name in ("a", "b", "D"):
            v = getattr(self, name)
            if not (math.isfinite(v) and v > 0):
                raise ValueError(f"{name} must be finite and > 0, got {v!r}")

    @property
    def rho(self) -> float:
        return self.a / self.b

    def rhs_value(self) -> float:
        return self.q * self.a**4 / (4.0 * math.pi**4 * self.D)


def _mn(problem: RitzProblem):
    """m and n of the unknowns in the order i = M (n - 1) + m (m fastest)."""
    m = np.tile(np.arange(1, problem.M + 1), problem.N)
    n = np.repeat(np.arange(1, problem.N + 1), problem.M)
    return m.astype(float), n.astype(float)


def p7_matrix(problem: RitzProblem) -> np.ndarray:
    """Dense K of P7 (11)-(12) (small M N only)."""
    m, n = _mn(problem)
    K = np.diag((m**2 + problem.rho**2 * n**2) ** 2)
    K += np.where(m[:, None] == m[None, :], 2.0 * m[:, None] ** 4, 0.0)
    K += np.where(n[:, None] == n[None, :], 2.0 * (problem.rho * n[:, None]) ** 4, 0.0)
    return K


def gram_matrix(problem: RitzProblem) -> np.ndarray:
    """Own derivation: the Ritz matrix of (7) from one-dimensional Gram matrices, divided by 4 pi^4 (small M N only)."""
    def grams(count):
        k = np.arange(1, count + 1, dtype=float)
        g0 = np.ones((count, count)) + 0.5 * np.eye(count)  # int phi phi
        g2 = np.diag(-2.0 * math.pi**2 * k**2)  # int phi'' phi
        g4 = np.diag(8.0 * math.pi**4 * k**4)  # int phi'' phi''
        return g0, g2, g4

    x0, x2, x4 = grams(problem.M)
    y0, y2, y4 = grams(problem.N)
    r2 = problem.rho**2
    # w_xx + r^2 w_yy with the x index fastest: kron(y-part, x-part)
    full = np.kron(y0, x4) + r2 * (np.kron(y2, x2) + np.kron(y2.T, x2.T)) + r2**2 * np.kron(y4, x0)
    return full / (4.0 * math.pi**4)


@dataclass
class RitzSolution:
    problem: RitzProblem
    w: np.ndarray  # coefficients w_mn in the order i = M (n - 1) + m
    capacity_cond: float | None = None  # condition number of the scaled capacity matrix (P7: kappa ~ 4)

    def coefficients(self) -> np.ndarray:
        """w_mn as an M x N array."""
        return self.w.reshape(self.problem.N, self.problem.M).T

    def centre(self) -> float:
        c = self.coefficients()
        return 4.0 * math.fsum(c[0::2, 0::2].ravel().tolist())

    def alpha(self) -> float:
        p = self.problem
        return self.centre() * p.D / (p.q * min(p.a, p.b) ** 4)

    def work(self) -> float:
        """int int w over the plate (load work / q)."""
        return self.problem.a * self.problem.b * math.fsum(self.w.tolist())

    def work_normalized(self) -> float:
        """int int w D / (q l^6), l = the shorter full side (the 'E Work' of P7, tables I-II, x 1e4)."""
        p = self.problem
        return self.work() * p.D / (p.q * min(p.a, p.b) ** 6)

    def value(self, x, y) -> np.ndarray:
        """w at points (x, y) (arrays of the same shape)."""
        p = self.problem
        x, y = np.asarray(x, float), np.asarray(y, float)
        phi_x = 1.0 - np.cos(2.0 * math.pi * np.outer(x.ravel() / p.a, np.arange(1, p.M + 1)))
        phi_y = 1.0 - np.cos(2.0 * math.pi * np.outer(y.ravel() / p.b, np.arange(1, p.N + 1)))
        return np.einsum("km,mn,kn->k", phi_x, self.coefficients(), phi_y).reshape(x.shape)


def solve_direct(problem: RitzProblem, matrix: str = "p7") -> RitzSolution:
    """Dense solve of K w = b (control; small M N)."""
    K = p7_matrix(problem) if matrix == "p7" else gram_matrix(problem)
    rhs = np.full(problem.M * problem.N, problem.rhs_value())
    return RitzSolution(problem, scipy.linalg.solve(K, rhs, assume_a="pos"))


def solve_smw(problem: RitzProblem) -> RitzSolution:
    """Sherman-Morrison-Woodbury form (14)-(17) with the capacity matrix scaled to unit diagonal (p. 3)."""
    M, N = problem.M, problem.N
    k_m = np.arange(1, M + 1, dtype=float)
    k_n = np.arange(1, N + 1, dtype=float)
    A = (k_m[:, None] ** 2 + problem.rho**2 * k_n[None, :] ** 2) ** 2  # M x N, A_(m,n)
    Ainv = 1.0 / A
    d = np.concatenate([2.0 * k_m**4, 2.0 * (problem.rho * k_n) ** 4])
    rhs = problem.rhs_value()
    w0 = rhs * Ainv  # A^-1 b, M x N
    v0 = np.concatenate([w0.sum(axis=1), w0.sum(axis=0)])  # U^T w0: sums over n (m groups), over m (n groups)
    C = np.zeros((M + N, M + N))
    C[:M, :M] = np.diag(Ainv.sum(axis=1))
    C[M:, M:] = np.diag(Ainv.sum(axis=0))
    C[:M, M:] = Ainv
    C[M:, :M] = Ainv.T
    C[np.diag_indices(M + N)] += 1.0 / d
    s = 1.0 / np.sqrt(np.diag(C))
    Cs = C * s[:, None] * s[None, :]
    cho = scipy.linalg.cho_factor(Cs)
    v1 = s * scipy.linalg.cho_solve(cho, s * v0)
    w1 = v1[:M, None] + v1[None, M:]  # U v1 on the M x N grid
    w = w0 - Ainv * w1
    ev = np.linalg.eigvalsh(Cs) if M + N <= 4000 else None
    cond = None if ev is None else float(ev[-1] / ev[0])
    return RitzSolution(problem, w.T.ravel(), cond)  # order i = M (n - 1) + m


def residual(solution: RitzSolution) -> float:
    """max |K w - b| / max |b| with K applied by its structure (no dense matrix)."""
    p = solution.problem
    c = solution.coefficients()
    k_m = np.arange(1, p.M + 1, dtype=float)
    k_n = np.arange(1, p.N + 1, dtype=float)
    A = (k_m[:, None] ** 2 + p.rho**2 * k_n[None, :] ** 2) ** 2
    Kw = A * c + (2.0 * k_m**4)[:, None] * c.sum(axis=1)[:, None] + (2.0 * (p.rho * k_n) ** 4)[None, :] * c.sum(axis=0)[None, :]
    return float(np.max(np.abs(Kw - p.rhs_value())) / p.rhs_value())
