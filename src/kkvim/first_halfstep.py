"""First KK-VIM half-step s = 1 (k = 0, D-030): given X0, compute Y1 and the field w1 (R2).

Square SSSS Kirchhoff-Love plate, N = 1, uniform load q (D-031, D-032, D-033).
docs/NUMERICAL_SPEC.md: grids 3.1-3.3, Y-half-step 5.2, variant B coefficients 6.2-7.1,
linear solve 8, normalization I-4 (9.1).

- ``solve_halfstep`` is the general building block of one half-step: coefficients of
  the fixed factor on its own grid (all coefficient errors are raised before any
  solve), right-hand side F_h * 1, solve of K Y = F_h * 1 on the other grid.
- ``first_halfstep`` builds X0_i = sin(pi i / n_x), calls ``solve_halfstep`` with
  grid_x as the fixed grid and grid_y as the solve grid, normalizes the pair by I-4
  and forms w1 = X1_scaled (x) Y1_normalized on the tensor nodes, axis 0 = x, axis 1 = y.
- ``node_value`` returns the field at a rationally given point only if it is a node.

D is the already computed bending stiffness; this is a low-level API, not the final
user interface. Errors are raised, never replaced by a zero solution.
"""

import math
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from kkvim.coefficients import ERROR_DEGENERATE, ERROR_INPUT, HalfStepCoefficients, compute_coefficients
from kkvim.grid import Grid1D
from kkvim.linear_solver import ERROR_NONFINITE, SolveDiagnostics, solve_k

NOT_A_NODE_MESSAGE = "точка не является узлом сетки"


class HalfStepError(RuntimeError):
    """Rejected input or degenerate normalization, with a specification error code."""

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code


class NotANodeError(ValueError):
    """The requested point is not a node of the tensor grid (NUMERICAL_SPEC.md, 3.2)."""

    def __init__(self):
        super().__init__(NOT_A_NODE_MESSAGE)


@dataclass(frozen=True)
class HalfStepSolution:
    """Unnormalized computed factor (interior values on the solve grid) and its data."""

    raw: np.ndarray
    coefficients: HalfStepCoefficients
    solver_diagnostics: SolveDiagnostics


@dataclass(frozen=True)
class FirstHalfStepResult:
    """Result of s = 1. Factors are interior values; the field holds all tensor nodes."""

    grid_x: Grid1D
    grid_y: Grid1D
    x0: np.ndarray  # given factor X0, interior values on grid_x
    y1_raw: np.ndarray  # computed Y1 before normalization, interior values on grid_y
    x1_scaled: np.ndarray  # X0 * c, interior values on grid_x
    y1_normalized: np.ndarray  # Y1_raw / c: max |.| = 1, positive grid sum; given factor of s = 2
    c: float  # normalization factor sigma * max |Y1_raw|
    w1: np.ndarray  # field, shape (n_x + 1, n_y + 1), axis 0 = x, axis 1 = y
    coefficients: HalfStepCoefficients  # A_h, B_h, C_h, F_h of X0 on grid_x
    solver_diagnostics: SolveDiagnostics
    s: int = 1
    k: int = 0


def solve_halfstep(fixed_grid: Grid1D, fixed_factor, solve_grid: Grid1D, q: float, D: float) -> HalfStepSolution:
    """One half-step: coefficients of the fixed factor on fixed_grid, then K Y = F_h * 1 on solve_grid."""
    coefficients = compute_coefficients(fixed_grid, fixed_factor, q)
    rhs = coefficients.F * np.ones(solve_grid.interior_size)
    result = solve_k(solve_grid, coefficients.A, coefficients.B, coefficients.C, D, rhs)
    return HalfStepSolution(raw=result.y, coefficients=coefficients, solver_diagnostics=result.diagnostics)


def first_halfstep(a: float, D: float, q: float, n_x: int, n_y: int) -> FirstHalfStepResult:
    """Half-step s = 1 for the square of side a with independent n_x and n_y."""
    a = _positive_input("a", a)
    D = _positive_input("D", D)
    grid_x = Grid1D(a, n_x)
    grid_y = Grid1D(a, n_y)

    x0 = np.sin(math.pi * np.arange(1, grid_x.n) / grid_x.n)
    step = solve_halfstep(grid_x, x0, grid_y, q, D)

    y1_raw = step.raw
    c = _normalization_factor(grid_y, y1_raw)
    y1_normalized = y1_raw / c
    x1_scaled = x0 * c
    w1 = np.outer(grid_x.with_boundary(x1_scaled), grid_y.with_boundary(y1_normalized))
    return FirstHalfStepResult(
        grid_x=grid_x,
        grid_y=grid_y,
        x0=x0,
        y1_raw=y1_raw,
        x1_scaled=x1_scaled,
        y1_normalized=y1_normalized,
        c=c,
        w1=w1,
        coefficients=step.coefficients,
        solver_diagnostics=step.solver_diagnostics,
    )


def node_value(result: FirstHalfStepResult, x_fraction, y_fraction) -> float:
    """Field value at (x_fraction * a, y_fraction * a) if that point is a grid node.

    Fractions are rational numbers: Fraction, int or a string such as "1/2". A point
    that is not a node raises NotANodeError; no nearest node and no interpolation.
    """
    i = _node_index(x_fraction, result.grid_x.n)
    j = _node_index(y_fraction, result.grid_y.n)
    return float(result.w1[i, j])


def _normalization_factor(grid: Grid1D, f: np.ndarray) -> float:
    """c = sigma * max |f|, sigma = sign <f, 1> (I-4); errors instead of division by zero."""
    total = float(np.sum(f))
    peak = float(np.max(np.abs(f)))
    if not (math.isfinite(total) and math.isfinite(peak)):
        raise HalfStepError(ERROR_NONFINITE, "normalization quantities are not finite")
    if total == 0.0 or peak == 0.0:
        raise HalfStepError(ERROR_DEGENERATE, f"cannot normalize: grid sum {total!r}, max |f| {peak!r}")
    c = math.copysign(peak, total)
    if not math.isfinite(c):
        raise HalfStepError(ERROR_NONFINITE, "normalization factor is not finite")
    return c


def _positive_input(name: str, value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"{name} must be a real number, got {value!r}")
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise HalfStepError(ERROR_INPUT, f"{name} must be finite and positive, got {value!r}")
    return value


def _node_index(fraction, n: int) -> int:
    if isinstance(fraction, (bool, float)) or not isinstance(fraction, (Fraction, int, str)):
        raise TypeError(f"point coordinate must be a rational number (Fraction, int or str), got {fraction!r}")
    value = Fraction(fraction)
    if not 0 <= value <= 1:
        raise ValueError(f"point coordinate {value} is outside [0, 1]")
    index = value * n
    if index.denominator != 1:
        raise NotANodeError()
    return int(index)
