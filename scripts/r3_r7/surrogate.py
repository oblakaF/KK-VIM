"""Scratch surrogate for the discrete energy test A-7 (R3, R-7). Not project code.

Surrogate: continuous KK-VIM fields w_SA^(s) of kkvim_reference.parseval_recursion (rectangle, l_s = b, a),
sampled at the nodes of the tensor grid, inserted into the DISCRETE energy form a_h and the rounding
model of NUMERICAL_SPEC.md 4.1, 10.3, 10.4:

    Q_s     = 1/2 a_h(d, d),  d = w^(s-1) - w^(s)   (d = fixed factor (x) (new - old changed factor)),
    R(w)    = u (ceil(log2 N) + 16) S_hat(w),
    R_f     = 4u (max(nx, ny)/pi)^2 |w^(s)|_a |d|_a,
    tau     = |gamma| + 10 (R(w^(s-1)) + R(w^(s)) + R_f),
    margin  = Q / (10 tau).

gamma_s of the future R3 solver is unknown; margin_upper uses gamma = 0 and is an UPPER bound of the
margin for the same Q, R, R_f (|gamma| >= 0 only increases tau).
"""

import math
from fractions import Fraction

import numpy as np

from kkvim_reference.parseval_recursion import ParsevalRecursion

U = 2.0**-53


def sine_table(n, K):
    """S[i, k] = sin((2k+1) pi i / n), i = 0..n, by exact integer reduction (as kkvim_reference)."""
    r = np.arange(2 * n, dtype=np.int64)
    sign = np.where(r >= n, -1.0, 1.0)
    rr = np.where(r >= n, r - n, r)
    rr = np.minimum(rr, n - rr)
    table = sign * np.sin(math.pi * rr / n)
    table[rr == 0] = 0.0
    m = 2 * np.arange(K, dtype=np.int64) + 1
    i = np.arange(n + 1, dtype=np.int64)
    return table[np.outer(i, m) % (2 * n)]


class Surrogate:
    """Continuous sequence for the plate a x b (a = 1), q = D = 1, half-steps 1..s_max."""

    def __init__(self, a, b, s_max=4):
        self.a, self.b = float(a), float(b)
        self.rec = ParsevalRecursion(a=self.a, b=self.b, q=1.0, D=1.0, s_max=s_max)
        self.s_max = s_max
        self.coeffs = {0: np.array([1.0])}  # X0 = sin(pi x / a)
        for s in range(1, s_max + 1):
            self.coeffs[s] = np.array(self.rec.halfstep(s).step.coefficients())
        self.K = max(len(c) for c in self.coeffs.values())
        self._tables = {}

    def _table(self, n):
        if n not in self._tables:
            self._tables[n] = sine_table(n, self.K)
        return self._tables[n]

    def values(self, index, n, coeffs=None):
        c = self.coeffs[index] if coeffs is None else coeffs
        return self._table(n)[:, : len(c)] @ c

    @staticmethod
    def x_index(s):
        return s if s % 2 == 0 else s - 1

    @staticmethod
    def y_index(s):
        return s if s % 2 == 1 else s - 1

    def field(self, s, nx, ny):
        return np.outer(self.values(self.x_index(s), nx), self.values(self.y_index(s), ny))

    def change(self, s, nx, ny):
        """d = w^(s-1) - w^(s) = fixed (x) (old - new) of the factor changed on half-step s."""
        old, new = self.coeffs[s - 2], self.coeffs[s]
        size = max(len(old), len(new))
        diff = np.zeros(size)
        diff[: len(old)] += old
        diff[: len(new)] -= new
        fixed = self.coeffs[s - 1]
        if s % 2:  # Y-half-step: fixed X^(s-1), changed Y
            return np.outer(self.values(s - 1, nx, fixed), self.values(None, ny, diff))
        return np.outer(self.values(None, nx, diff), self.values(s - 1, ny, fixed))

    def continuous_Q(self, s):
        return 0.5 * self.rec.energy(s).change_norm2


def laplacian(w, hx, hy):
    c = w[1:-1, 1:-1]
    return (w[:-2, 1:-1] - 2 * c + w[2:, 1:-1]) / hx**2 + (w[1:-1, :-2] - 2 * c + w[1:-1, 2:]) / hy**2


def abs_laplacian(w, hx, hy):
    a = np.abs(w)
    c = a[1:-1, 1:-1]
    return (a[:-2, 1:-1] + 2 * c + a[2:, 1:-1]) / hx**2 + (a[1:-1, :-2] + 2 * c + a[1:-1, 2:]) / hy**2


def a_h(w, hx, hy, D=1.0):
    return D * hx * hy * float(np.sum(laplacian(w, hx, hy) ** 2))


def rounding_R(w, hx, hy, D=1.0, q=1.0):
    s_hat = D * hx * hy * float(np.sum(np.abs(laplacian(w, hx, hy)) * abs_laplacian(w, hx, hy)))
    s_hat += abs(q) * hx * hy * float(np.sum(np.abs(w[1:-1, 1:-1])))
    N = (w.shape[0] - 2) * (w.shape[1] - 2)
    return U * ((N - 1).bit_length() + 16) * s_hat


def test_quantities(sur, s, nx, ny):
    hx, hy = sur.a / nx, sur.b / ny
    w_prev, w_now = sur.field(s - 1, nx, ny), sur.field(s, nx, ny)
    d = sur.change(s, nx, ny)
    a_d = a_h(d, hx, hy)
    Q = 0.5 * a_d
    R_prev, R_now = rounding_R(w_prev, hx, hy), rounding_R(w_now, hx, hy)
    R_f = 4 * U * (max(nx, ny) / math.pi) ** 2 * math.sqrt(a_h(w_now, hx, hy)) * math.sqrt(a_d)
    R_part = 10 * (R_prev + R_now + R_f)
    return {
        "Q": Q, "R_prev": R_prev, "R_now": R_now, "R_f": R_f, "tau_R": R_part,
        "margin_upper": Q / (10 * R_part), "gamma_allowance": Q / 10 - R_part,
        "norm_w": math.sqrt(a_h(w_now, hx, hy)), "norm_d": math.sqrt(a_d),
    }


RATIOS = {  # exact rational representations of the control ratios r = b/a
    "0,2": Fraction(1, 5), "1/3": Fraction(1, 3), "1/2": Fraction(1, 2), "2/3": Fraction(2, 3), "1": Fraction(1),
    "1,4": Fraction(7, 5), "1,5": Fraction(3, 2), "2": Fraction(2), "3": Fraction(3), "5": Fraction(5),
}


def family_I():
    return [(n, n) for n in range(20, 641, 4)]


def family_II(r):
    pairs = []
    for nx in range(4, 641, 4):
        ny = r * nx
        if ny.denominator == 1 and ny.numerator % 4 == 0 and max(nx, ny.numerator) <= 640 and ny.numerator >= 4:
            pairs.append((nx, ny.numerator))
    return pairs
