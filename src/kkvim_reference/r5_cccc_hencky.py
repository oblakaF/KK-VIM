"""QA reference A for the clamped rectangle (CCCC): Hencky-Inglis superposition, finite sections (R5-V). OBSERVATION only.

Independent of the main solver: no import of ``kkvim`` (D-028); nothing of Builder is used (D-049.8). Sources: L-118 (P5,
Meleshko & Gomilko 2004) (3.1)-(3.3), (3.7), pp. 810-811; L-117 (P4, 1997) (2.1), (3.3), (3.6); L-058 (Batista 2010)
(24)-(28), p. 12 as the second transcription. Formulas used here are the project's derivation
docs/R5_CCCC_REFERENCE_MATH.pre-review.md (RM) 2.1-2.5, 4.2, 4.4 (Reviewer 62db9f2: R-1...R-9, R-13, R-14 PASS).

Half sides: |x| <= a, |y| <= b (origin at the centre), P = q/D, alpha_m = (2m - 1) pi/(2a), beta_l = (2l - 1) pi/(2b),
Delta(t) = tanh t + t/cosh^2 t,

    w = (P/8)(a^2 - x^2)(b^2 - y^2) - b sum_m (-1)^m (X_m/alpha_m) H(y, alpha_m, b) cos alpha_m x
                                    + a sum_l (-1)^l (Y_l/beta_l) H(x, beta_l, a) cos beta_l y,        (A-1)
    H(z, lambda, h) = h tanh(lambda h) cosh(lambda z)/cosh(lambda h) - z sinh(lambda z)/cosh(lambda h);
    X_m b Delta(alpha_m b) = sum_l Y_l 4 alpha_m^2 beta_l/(alpha_m^2 + beta_l^2)^2 - P b/(a alpha_m^2),   (A-2a)
    Y_l a Delta(beta_l a)  = sum_m X_m 4 alpha_m beta_l^2/(alpha_m^2 + beta_l^2)^2 + P a/(b beta_l^2).    (A-2b)
Centre (RM 4.2): w(0,0) = (P/8) a^2 b^2 - b^2 sum (-1)^m tanh(alpha_m b) X_m/(alpha_m cosh alpha_m b)
                                       + a^2 sum (-1)^l tanh(beta_l a) Y_l/(beta_l cosh beta_l a).
Work (RM 4.4): int int w = (2P/9) a^3 b^3 + 2b sum X_m I_H(alpha_m, b)/alpha_m^2 - 2a sum Y_l I_H(beta_l, a)/beta_l^2,
I_H(lambda, h) = 2 tanh(lambda h)/lambda^2 - 2h/(lambda cosh^2 lambda h).

Engineering (level C): the finite section (N_X, N_Y) is solved after dividing every row by its diagonal (z = K z + f~,
RM 3.1) with a dense LU; hyperbolic ratios are evaluated as exp(-lambda (h - |z|)) (1 +- exp(-2 lambda |z|))/(1 + exp(-2 lambda h))
(no overflow; P7 p. 1 remarks the overflow problem of Hencky's method).

Accuracy: rho = sup f < 1 is proved for t >= 20 only (numerically 0.708 for t < 20, OV-M1); there is no strict bound of the
truncation error (OV-M2, OV-M3). All results of this module are OBSERVATION (D-049.8, item 2); the diagnostics below are not
error estimates.
"""

import math
from dataclasses import dataclass

import numpy as np
import scipy.linalg


def _delta(t: np.ndarray) -> np.ndarray:
    """Delta(t) = tanh t + t sech^2 t."""
    e = np.exp(-2.0 * t)
    sech2 = 4.0 * e / (1.0 + e) ** 2
    return np.tanh(t) + t * sech2


@dataclass(frozen=True)
class HenckyProblem:
    lx: float  # full side along x
    ly: float  # full side along y
    NX: int  # number of X_m
    NY: int  # number of Y_l
    D: float = 1.0
    q: float = 1.0

    @property
    def a(self) -> float:
        return self.lx / 2.0

    @property
    def b(self) -> float:
        return self.ly / 2.0

    @property
    def P(self) -> float:
        return self.q / self.D

    def alphas(self) -> np.ndarray:
        return (2.0 * np.arange(1, self.NX + 1) - 1.0) * math.pi / (2.0 * self.a)

    def betas(self) -> np.ndarray:
        return (2.0 * np.arange(1, self.NY + 1) - 1.0) * math.pi / (2.0 * self.b)


def section(problem: HenckyProblem):
    """Scaled finite section: matrix K (rows divided by the diagonal), right side f~ and the diagonals."""
    a, b, P = problem.a, problem.b, problem.P
    al, be = problem.alphas(), problem.betas()
    d_x = b * _delta(al * b)
    d_y = a * _delta(be * a)
    den = (al[:, None] ** 2 + be[None, :] ** 2) ** 2
    Kxy = 4.0 * al[:, None] ** 2 * be[None, :] / den / d_x[:, None]  # NX x NY
    Kyx = (4.0 * al[:, None] * be[None, :] ** 2 / den).T / d_y[:, None]  # NY x NX
    f = np.concatenate([-P * b / (a * al**2) / d_x, P * a / (b * be**2) / d_y])
    nx, ny = problem.NX, problem.NY
    K = np.zeros((nx + ny, nx + ny))
    K[:nx, nx:] = Kxy
    K[nx:, :nx] = Kyx
    return K, f, d_x, d_y


@dataclass
class HenckySolution:
    problem: HenckyProblem
    X: np.ndarray
    Y: np.ndarray
    row_sum_max: float  # max row sum of the section K (<= rho if rho < 1)
    residual: float  # max |(I - K) z - f~| / max |f~|

    def centre(self) -> float:
        p = self.problem
        a, b, P = p.a, p.b, p.P
        al, be = p.alphas(), p.betas()
        sx = (-1.0) ** np.arange(1, p.NX + 1)
        sy = (-1.0) ** np.arange(1, p.NY + 1)
        tx = np.tanh(al * b) / np.cosh(np.minimum(al * b, 700.0))
        ty = np.tanh(be * a) / np.cosh(np.minimum(be * a, 700.0))
        terms = [P / 8.0 * a**2 * b**2]
        terms += (-(b**2) * sx * tx * self.X / al).tolist()
        terms += (a**2 * sy * ty * self.Y / be).tolist()
        return math.fsum(terms)

    def alpha(self) -> float:
        p = self.problem
        return self.centre() * p.D / (p.q * min(p.lx, p.ly) ** 4)

    def work(self) -> float:
        """int int w over the plate (RM 4.4)."""
        p = self.problem
        a, b, P = p.a, p.b, p.P
        al, be = p.alphas(), p.betas()

        def i_h(lam, h):
            e = np.exp(-2.0 * lam * h)
            sech2 = 4.0 * e / (1.0 + e) ** 2
            return 2.0 * np.tanh(lam * h) / lam**2 - 2.0 * h * sech2 / lam

        terms = [2.0 * P / 9.0 * a**3 * b**3]
        terms += (2.0 * b * self.X * i_h(al, b) / al**2).tolist()
        terms += (-2.0 * a * self.Y * i_h(be, a) / be**2).tolist()
        return math.fsum(terms)

    def work_normalized(self) -> float:
        p = self.problem
        return self.work() * p.D / (p.q * min(p.lx, p.ly) ** 6)

    def value(self, x, y) -> np.ndarray:
        """w at points with the origin at the CORNER (0 <= x <= l_x, 0 <= y <= l_y), by (A-1)."""
        p = self.problem
        a, b, P = p.a, p.b, p.P
        X = np.asarray(x, float).ravel() - a
        Yc = np.asarray(y, float).ravel() - b
        al, be = p.alphas(), p.betas()
        w = P / 8.0 * (a**2 - X**2) * (b**2 - Yc**2)
        sx = (-1.0) ** np.arange(1, p.NX + 1)
        sy = (-1.0) ** np.arange(1, p.NY + 1)
        hy = _H(Yc[:, None], al[None, :], b)  # points x NX
        w = w - b * np.sum(sx * self.X / al * hy * np.cos(al[None, :] * X[:, None]), axis=1)
        hx = _H(X[:, None], be[None, :], a)
        w = w + a * np.sum(sy * self.Y / be * hx * np.cos(be[None, :] * Yc[:, None]), axis=1)
        return w.reshape(np.shape(x))

    def boundary_slope(self, x) -> np.ndarray:
        """dw/dy at y = b (top edge) by the series (A-1) (RM 2.2); zero for the exact solution. Diagnostic only."""
        p = self.problem
        a, b, P = p.a, p.b, p.P
        X = np.asarray(x, float).ravel() - a
        al, be = p.alphas(), p.betas()
        sx = (-1.0) ** np.arange(1, p.NX + 1)
        s = -P / 4.0 * b * (a**2 - X**2)
        s = s + b * np.sum(sx * self.X / al * _delta(al * b) * np.cos(al[None, :] * X[:, None]), axis=1)
        s = s + a * np.sum(self.Y * _H(X[:, None], be[None, :], a), axis=1)
        return s.reshape(np.shape(x))


def _H(z, lam, h):
    """H(z, lambda, h) without overflow."""
    az = np.abs(z)
    e_h = np.exp(-2.0 * lam * h)
    ratio_c = np.exp(-lam * (h - az)) * (1.0 + np.exp(-2.0 * lam * az)) / (1.0 + e_h)  # cosh(lz)/cosh(lh)
    ratio_s = np.sign(z) * np.exp(-lam * (h - az)) * (1.0 - np.exp(-2.0 * lam * az)) / (1.0 + e_h)  # sinh(lz)/cosh(lh)
    return h * np.tanh(lam * h) * ratio_c - z * ratio_s


def solve(problem: HenckyProblem) -> HenckySolution:
    K, f, _, _ = section(problem)
    I = np.eye(K.shape[0])
    z = scipy.linalg.solve(I - K, f)
    res = float(np.max(np.abs((I - K) @ z - f)) / np.max(np.abs(f)))
    return HenckySolution(problem, z[:problem.NX], z[problem.NX:], float(np.max(K.sum(axis=1))), res)


def default_section(lx: float, ly: float, NY: int, D: float = 1.0, q: float = 1.0) -> HenckyProblem:
    """N_X proportional to the side ratio so that alpha_NX ~ beta_NY (RM 5, item 2): N_X = round(N_Y l_x/l_y)."""
    return HenckyProblem(lx, ly, max(1, round(NY * lx / ly)), NY, D, q)
