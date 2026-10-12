"""Half-step coefficients A_h, B_h, C_h, F_h of a given multiplier (R2).

Variant B of docs/NUMERICAL_SPEC.md, sections 5.2, 6.2-6.3 and 7.1 (D-032): the
coefficients are grid sums induced by the discrete functional Pi_h, computed the
same way on every half-step, including s = 1:

- A_h = h * sum f_i^2 over the interior nodes;
- B_h = h * sum (D+ f)^2 over all n intervals, including the two end intervals
  (sum of squares, so B_h >= 0 by construction; equals -<f, L f> by ChV-1);
- C_h = h * sum (L f)_i^2 over the interior nodes;
- F_h = q * h * sum f_i over the interior nodes.

Derivatives of f are always difference quotients; no analytic derivatives, no
continuous integrals and no other quadrature are used.

Checks, all done before any linear solve (NUMERICAL_SPEC.md 8.3, 9.2, 11.3):

- q not finite or q = 0 -> ERROR_INPUT (R2 requires q != 0);
- NaN or infinity in f or in the computed coefficients -> ERROR_NONFINITE;
- A_h <= 0 or F_h = 0 -> ERROR_DEGENERATE.

F_h = 0 is checked by exact equality, as specified; no closeness threshold is used.
"""

import math
from dataclasses import dataclass

import numpy as np

from kkvim.grid import Grid1D, forward_difference, second_difference
from kkvim.linear_solver import ERROR_NONFINITE

ERROR_INPUT = "ERROR_INPUT"
ERROR_DEGENERATE = "ERROR_DEGENERATE"


@dataclass(frozen=True)
class HalfStepCoefficients:
    """Coefficients of the one-dimensional half-step system D [A L^2 - 2B L + C I] Y = F 1."""

    A: float
    B: float
    C: float
    F: float


class CoefficientError(RuntimeError):
    """Rejected input or multiplier, with a specification error code."""

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code


def compute_coefficients(grid: Grid1D, f, q: float) -> HalfStepCoefficients:
    """Coefficients of the given multiplier f (interior values, shape (n - 1,)) and load q.

    Raises CoefficientError (ERROR_INPUT, ERROR_NONFINITE, ERROR_DEGENERATE),
    TypeError for a non-numeric q and ValueError for a wrong shape of f.
    """
    if isinstance(q, bool) or not isinstance(q, (int, float, np.integer, np.floating)):
        raise TypeError(f"q must be a real number, got {q!r}")
    q = float(q)
    if not math.isfinite(q) or q == 0.0:
        raise CoefficientError(ERROR_INPUT, f"load q must be finite and nonzero, got {q!r}")

    f = np.asarray(f, dtype=float)
    if f.shape != (grid.interior_size,):
        raise ValueError(f"expected f of shape ({grid.interior_size},), got {f.shape}")
    if not np.all(np.isfinite(f)):
        raise CoefficientError(ERROR_NONFINITE, "multiplier contains NaN or infinity")

    h = grid.h
    with np.errstate(over="ignore", invalid="ignore"):
        A = h * float(np.sum(f * f))
        B = h * float(np.sum(forward_difference(grid, f) ** 2))
        C = h * float(np.sum(second_difference(grid, f) ** 2))
        F = q * h * float(np.sum(f))
    coefficients = HalfStepCoefficients(A=A, B=B, C=C, F=F)

    if not all(math.isfinite(value) for value in (A, B, C, F)):
        raise CoefficientError(ERROR_NONFINITE, f"non-finite coefficients {coefficients}")
    if A <= 0.0:
        raise CoefficientError(ERROR_DEGENERATE, f"A_h = {A!r} <= 0 (zero multiplier)")
    if F == 0.0:
        raise CoefficientError(ERROR_DEGENERATE, "F_h = 0 (multiplier with zero grid sum)")
    return coefficients
