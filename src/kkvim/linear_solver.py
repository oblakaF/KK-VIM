"""One-dimensional linear solver K Y = g of the R2 half-steps.

K = D [A L^2 - 2B L + C I] on a grid G(length, n), with L from ``kkvim.grid``
(zero end values, natural condition for L^2). Algorithm and control follow
docs/NUMERICAL_SPEC.md, section 8 (D-032):

- 8.1: roots r1,2 = (B +- sqrt(B^2 - AC)) / A with a complex square root;
  K = D A (L - r1 I)(L - r2 I) is solved as two tridiagonal Dirichlet problems
  (L - r1 I) Z = g / (D A), (L - r2 I) Y = Z by banded LU (LAPACK gbsv via SciPy);
  the five-diagonal matrix K is never assembled;
- 8.2: residual r = g - K Y applied matrix-free and the normalized backward error
  eta = |r|_inf / (D (16A/h^4 + 8B/h^2 + C) |Y|_inf + |g|_inf);
- 8.3: accept if eta <= 1e-12 and |Im Y|_inf <= 1e-12 |Y|_inf; otherwise at most
  one step of iterative refinement; then ERROR_LINEAR_SOLVE. Non-finite values
  give ERROR_NONFINITE.

Failures raise ``LinearSolveError`` with ``code`` and the diagnostics gathered so far.
"""

import math
from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_banded

from kkvim.grid import Grid1D, fourth_difference, second_difference

ETA_MAX = 1e-12
"""Maximal normalized backward error (NUMERICAL_SPEC.md, 8.3, I-7)."""

IMAG_MAX = 1e-12
"""Maximal relative imaginary part |Im Y|_inf / |Y|_inf (NUMERICAL_SPEC.md, 8.3)."""

MAX_REFINEMENTS = 1
"""At most one step of iterative refinement (NUMERICAL_SPEC.md, 8.3)."""

ERROR_LINEAR_SOLVE = "ERROR_LINEAR_SOLVE"
ERROR_NONFINITE = "ERROR_NONFINITE"


@dataclass(frozen=True)
class SolveDiagnostics:
    """Control values of the last checked candidate solution."""

    eta: float
    residual_norm: float
    imag_ratio: float
    refinements: int

    @property
    def refined(self) -> bool:
        return self.refinements > 0


@dataclass(frozen=True)
class SolveResult:
    """Accepted solution: interior values y of shape (n - 1,) and diagnostics."""

    y: np.ndarray
    diagnostics: SolveDiagnostics


class LinearSolveError(RuntimeError):
    """Solver failure with a specification error code (ERROR_LINEAR_SOLVE, ERROR_NONFINITE)."""

    def __init__(self, code: str, message: str, diagnostics: SolveDiagnostics | None = None):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.diagnostics = diagnostics


def characteristic_roots(A: float, B: float, C: float) -> tuple[complex, complex]:
    """Roots r1, r2 = (B +- sqrt(B^2 - AC)) / A of A r^2 - 2B r + C, complex square root."""
    root = np.sqrt(complex(B * B - A * C))
    return complex((B + root) / A), complex((B - root) / A)


def apply_k(grid: Grid1D, A: float, B: float, C: float, D: float, y) -> np.ndarray:
    """K y = D [A L(L y) - 2B L y + C y], matrix-free; shape (n - 1,)."""
    y = np.asarray(y, dtype=float)
    return D * (A * fourth_difference(grid, y) - 2.0 * B * second_difference(grid, y) + C * y)


def backward_error(grid: Grid1D, A: float, B: float, C: float, D: float, y, g) -> tuple[float, float]:
    """Return (eta, |r|_inf) for r = g - K y (NUMERICAL_SPEC.md, 8.2)."""
    y = np.asarray(y, dtype=float)
    g = np.asarray(g, dtype=float)
    residual_norm = _max_abs(g - apply_k(grid, A, B, C, D, y))
    h = grid.h
    scale = D * (16.0 * A / h**4 + 8.0 * B / h**2 + C) * _max_abs(y) + _max_abs(g)
    eta = residual_norm / scale if scale > 0.0 else (0.0 if residual_norm == 0.0 else math.inf)
    return eta, residual_norm


def solve_k(grid: Grid1D, A: float, B: float, C: float, D: float, g) -> SolveResult:
    """Solve K Y = g for the interior values Y; g has shape (n - 1,).

    Requires A > 0, B >= 0, C >= 0, D > 0 (NUMERICAL_SPEC.md, 8.1).
    Raises ValueError/TypeError for invalid input and LinearSolveError on failure.
    """
    A, B, C, D = (_coefficient(name, value) for name, value in zip("ABCD", (A, B, C, D)))
    if A <= 0.0 or D <= 0.0 or B < 0.0 or C < 0.0:
        raise ValueError(f"require A > 0, B >= 0, C >= 0, D > 0; got A={A}, B={B}, C={C}, D={D}")
    g = np.asarray(g, dtype=float)
    if g.shape != (grid.interior_size,):
        raise ValueError(f"expected g of shape ({grid.interior_size},), got {g.shape}")
    if not np.all(np.isfinite(g)):
        raise LinearSolveError(ERROR_NONFINITE, "right-hand side contains NaN or infinity")

    r1, r2 = characteristic_roots(A, B, C)
    y_complex = _factored_solve(grid, r1, r2, D * A, g)
    refinements = 0
    while True:
        y, diagnostics = _check(grid, A, B, C, D, g, y_complex, refinements)
        if diagnostics.eta <= ETA_MAX and diagnostics.imag_ratio <= IMAG_MAX:
            return SolveResult(y=y, diagnostics=diagnostics)
        if refinements >= MAX_REFINEMENTS:
            raise LinearSolveError(
                ERROR_LINEAR_SOLVE,
                f"eta={diagnostics.eta:.3e}, imag_ratio={diagnostics.imag_ratio:.3e} "
                f"after {refinements} refinement(s)",
                diagnostics,
            )
        residual = g - apply_k(grid, A, B, C, D, y)
        y_complex = y_complex + _factored_solve(grid, r1, r2, D * A, residual)
        refinements += 1


def _factored_solve(grid: Grid1D, r1: complex, r2: complex, scale: float, rhs: np.ndarray) -> np.ndarray:
    """(L - r1 I) Z = rhs / scale, (L - r2 I) Y = Z; returns complex Y of shape (n - 1,)."""
    z = _shifted_dirichlet_solve(grid, r1, rhs / scale)
    return _shifted_dirichlet_solve(grid, r2, z)


def _shifted_dirichlet_solve(grid: Grid1D, shift: complex, rhs: np.ndarray) -> np.ndarray:
    """Solve the tridiagonal system (L - shift I) u = rhs by banded LU."""
    size = grid.interior_size
    off = 1.0 / grid.h**2
    bands = np.empty((3, size), dtype=complex)
    bands[0, :] = off
    bands[1, :] = -2.0 * off - shift
    bands[2, :] = off
    return solve_banded((1, 1), bands, np.asarray(rhs, dtype=complex), check_finite=False)


def _check(grid, A, B, C, D, g, y_complex, refinements):
    """Finite check, real part, imaginary-part ratio and backward error of a candidate."""
    if not np.all(np.isfinite(y_complex)):
        raise LinearSolveError(ERROR_NONFINITE, f"solution contains NaN or infinity after {refinements} refinement(s)")
    y = np.ascontiguousarray(y_complex.real)
    imag_norm = _max_abs(y_complex.imag)
    size = _max_abs(y_complex)
    imag_ratio = imag_norm / size if size > 0.0 else (0.0 if imag_norm == 0.0 else math.inf)
    eta, residual_norm = backward_error(grid, A, B, C, D, y, g)
    if not (math.isfinite(eta) and math.isfinite(residual_norm)):
        raise LinearSolveError(ERROR_NONFINITE, "residual is NaN or infinite")
    return y, SolveDiagnostics(eta=eta, residual_norm=residual_norm, imag_ratio=imag_ratio, refinements=refinements)


def _coefficient(name: str, value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"{name} must be a real number, got {value!r}")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return value


def _max_abs(values) -> float:
    values = np.asarray(values)
    return float(np.max(np.abs(values))) if values.size else 0.0
