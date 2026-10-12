"""Second KK-VIM half-step s = 2 (k = 1, D-030): given Y1, compute X2 and the field w2 (R2).

docs/NUMERICAL_SPEC.md 5.3: the X-half-step is the same one-dimensional procedure as
the Y-half-step with x and y exchanged, so ``solve_halfstep`` is reused:
coefficients A_h, B_h, C_h, F_h of the normalized Y1 on grid_y, then
K X = F_h * 1 on grid_x. The pair is normalized by I-4 (9.1):
c2 = sigma * max |X2_raw|, X2_normalized = X2_raw / c2, Y2_scaled = Y1_normalized * c2,
and w2 = X2_normalized (x) Y2_scaled on the tensor nodes (axis 0 = x, axis 1 = y).

D and q are taken from the first half-step unchanged. Errors are raised, never
replaced by a field.
"""

from dataclasses import dataclass

import numpy as np

from kkvim.coefficients import HalfStepCoefficients
from kkvim.first_halfstep import FirstHalfStepResult, normalization_factor, solve_halfstep
from kkvim.grid import Grid1D
from kkvim.linear_solver import SolveDiagnostics


@dataclass(frozen=True)
class SecondHalfStepResult:
    """Result of s = 2. Factors are interior values; fields hold all tensor nodes."""

    first: FirstHalfStepResult  # the half-step s = 1, including w1
    x2_raw: np.ndarray  # computed X2 before normalization, interior values on grid_x
    x2_normalized: np.ndarray  # X2_raw / c2: max |.| = 1, positive grid sum
    y2_scaled: np.ndarray  # Y1_normalized * c2, interior values on grid_y
    c2: float  # normalization factor sigma * max |X2_raw|
    w2: np.ndarray  # field, shape (n_x + 1, n_y + 1), axis 0 = x, axis 1 = y
    coefficients: HalfStepCoefficients  # A_h, B_h, C_h, F_h of Y1_normalized on grid_y
    solver_diagnostics: SolveDiagnostics
    s: int = 2
    k: int = 1

    @property
    def grid_x(self) -> Grid1D:
        return self.first.grid_x

    @property
    def grid_y(self) -> Grid1D:
        return self.first.grid_y

    @property
    def w1(self) -> np.ndarray:
        return self.first.w1

    @property
    def D(self) -> float:
        return self.first.D

    @property
    def q(self) -> float:
        return self.first.q


def second_halfstep(first: FirstHalfStepResult) -> SecondHalfStepResult:
    """Half-step s = 2 from the result of s = 1, with the same D and q."""
    if not isinstance(first, FirstHalfStepResult):
        raise TypeError(f"expected FirstHalfStepResult, got {type(first).__name__}")
    grid_x, grid_y = first.grid_x, first.grid_y

    step = solve_halfstep(grid_y, first.y1_normalized, grid_x, first.q, first.D)
    x2_raw = step.raw
    c2 = normalization_factor(x2_raw)
    x2_normalized = x2_raw / c2
    y2_scaled = first.y1_normalized * c2
    w2 = np.outer(grid_x.with_boundary(x2_normalized), grid_y.with_boundary(y2_scaled))
    return SecondHalfStepResult(
        first=first,
        x2_raw=x2_raw,
        x2_normalized=x2_normalized,
        y2_scaled=y2_scaled,
        c2=c2,
        w2=w2,
        coefficients=step.coefficients,
        solver_diagnostics=step.solver_diagnostics,
    )
