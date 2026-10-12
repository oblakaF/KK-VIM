"""Sine-series solution of one 1D KK-VIM half-step (R2, semi-analytic reference).

Independent of the finite-difference solver: no grid operators, no matrices,
no imports of ``kkvim`` (D-028, D-033). Solves

    D (A f'''' - 2 B f'' + C f) = F,   0 < xi < ell,
    f(0) = f(ell) = 0,   f''(0) = f''(ell) = 0,

with constant A > 0, B >= 0, C >= 0, F (NUMERICAL_SPEC.md, section 15.1,
ЧВ-17):

    f(xi) = sum_{m odd} c_m sin(lambda_m xi),   lambda_m = m pi / ell,
    c_m = 4 F / [D m pi (A lambda_m^4 + 2 B lambda_m^2 + C)],   c_m = 0 for even m.

Truncation (NUMERICAL_SPEC.md, section 15.4, item 1 and table of item 3,
ЧВ-20): M is the largest kept odd index ((M + 1)/2 terms), and

    K_tail = 4 |F| ell^4 / (D A pi^5),   max |f - f_M| <= K_tail / (8 M^4).

The bound holds for the exact series; rounding is accounted separately
(section 15.5). Only one half-step is solved here: the coefficients of the
next half-step, the recursion G_s and normalisation (sections 15.2-15.3) are
not part of this module.
"""

import math
from dataclasses import dataclass

__all__ = ["SineHalfStep", "odd_index_for_tail"]


def _check_finite(name, value):
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")


def _check_odd_index(M):
    if isinstance(M, bool) or not isinstance(M, int) or M < 1 or M % 2 == 0:
        raise ValueError(f"M must be a positive odd integer, got {M!r}")


def _tail_constant(A, F, D, ell):
    return 4.0 * abs(F) * ell**4 / (D * A * math.pi**5)


def odd_index_for_tail(target, *, A, F, D, ell):
    """Smallest odd M with K_tail / (8 M^4) <= target (target > 0).

    The choice follows from the analytic bound of section 15.4 only, not from
    a comparison with any other solution.
    """
    _check_finite("target", target)
    if target <= 0:
        raise ValueError(f"target must be positive, got {target!r}")
    k_tail = _tail_constant(A, F, D, ell)
    M = max(1, math.ceil((k_tail / (8.0 * target)) ** 0.25))
    M += 1 - M % 2
    while k_tail / (8.0 * M**4) > target:
        M += 2
    while M > 1 and k_tail / (8.0 * (M - 2) ** 4) <= target:
        M -= 2
    return M


@dataclass(frozen=True)
class SineHalfStep:
    """One half-step D(A f'''' - 2B f'' + C f) = F on (0, ell), series up to odd M.

    A, B, C, F -- coefficients from the given multiplier (MATH_SPEC.md, 4.2);
    D -- flexural rigidity; ell -- length of the interval of the unknown
    multiplier; M -- largest kept odd index.
    """

    A: float
    B: float
    C: float
    F: float
    D: float
    ell: float
    M: int

    def __post_init__(self):
        for name in ("A", "B", "C", "F", "D", "ell"):
            _check_finite(name, getattr(self, name))
        if not self.A > 0:
            raise ValueError(f"A must be positive, got {self.A!r}")
        if self.B < 0 or self.C < 0:
            raise ValueError(f"B and C must be non-negative, got B={self.B!r}, C={self.C!r}")
        if not self.D > 0:
            raise ValueError(f"D must be positive, got {self.D!r}")
        if not self.ell > 0:
            raise ValueError(f"ell must be positive, got {self.ell!r}")
        _check_odd_index(self.M)

    @property
    def n_terms(self):
        """Number of kept (odd) terms, (M + 1) / 2."""
        return (self.M + 1) // 2

    def coefficient(self, m):
        """c_m by ЧВ-17 for an integer m >= 1; exactly 0.0 for even m."""
        if isinstance(m, bool) or not isinstance(m, int) or m < 1:
            raise ValueError(f"m must be a positive integer, got {m!r}")
        if m % 2 == 0:
            return 0.0
        lam = m * math.pi / self.ell
        lam2 = lam * lam
        p_m = self.A * lam2 * lam2 + 2.0 * self.B * lam2 + self.C
        return 4.0 * self.F / (self.D * m * math.pi * p_m)

    def coefficients(self):
        """List of c_m for the kept odd indices m = 1, 3, ..., M."""
        return [self.coefficient(m) for m in range(1, self.M + 1, 2)]

    @property
    def tail_constant(self):
        """K_tail = 4 |F| ell^4 / (D A pi^5): |c_m| <= K_tail / m^5 (section 15.4)."""
        return _tail_constant(self.A, self.F, self.D, self.ell)

    @property
    def tail_bound(self):
        """Uniform truncation bound max |f - f_M| <= K_tail / (8 M^4) (section 15.4)."""
        return self.tail_constant / (8.0 * self.M**4)

    def abs_coefficient_sum(self):
        """Sum of |c_m| over kept terms: scale for rounding estimates (section 15.5)."""
        return math.fsum(abs(c) for c in self.coefficients())

    def value(self, xi):
        """Truncated series f_M(xi) for 0 <= xi <= ell (math.fsum).

        The argument m pi xi / ell is reduced exactly modulo 2 pi through
        t = fmod(m xi / ell, 2), so no large arguments reach sin.
        """
        _check_finite("xi", xi)
        if not 0.0 <= xi <= self.ell:
            raise ValueError(f"xi must lie in [0, ell] = [0, {self.ell!r}], got {xi!r}")
        theta = xi / self.ell
        terms = [
            c * math.sin(math.pi * math.fmod(m * theta, 2.0))
            for m, c in zip(range(1, self.M + 1, 2), self.coefficients())
        ]
        return math.fsum(terms)

    def nodal_values(self, n):
        """Truncated series at the nodes xi_i = i ell / n, i = 0..n (И-11).

        sin(m pi i / n) = sin(pi r / n) with the integer reduction
        r = (m i) mod 2n; the values sin(pi r / n) are computed once, reduced to
        the first quarter period (exact zeros at r = 0, n).
        """
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            raise ValueError(f"n must be a positive integer, got {n!r}")
        table = [_sin_pi_ratio(r, n) for r in range(2 * n)]
        odd = list(range(1, self.M + 1, 2))
        coeffs = self.coefficients()
        two_n = 2 * n
        return [
            math.fsum(c * table[(m * i) % two_n] for m, c in zip(odd, coeffs))
            for i in range(n + 1)
        ]


def _sin_pi_ratio(r, n):
    """sin(pi r / n) for 0 <= r < 2n via reduction to [0, pi/2]."""
    sign = 1.0
    if r >= n:
        r -= n
        sign = -1.0
    if 2 * r > n:
        r = n - r
    if r == 0:
        return 0.0
    return sign * math.sin(math.pi * r / n)
