"""KK-VIM half-steps on the rectangle a x b (R3, stage A; D-040, D-041).

docs/R3_NUMERICAL_SPEC.md 1.2, 2, 3: the R2 scheme without change on two independent
grids G(a, n_x) and G(b, n_y). Odd s is a Y-half-step solved on G(b, n_y) with the
coefficients of X on G(a, n_x); even s is an X-half-step solved on G(a, n_x) with the
coefficients of Y on G(b, n_y) (Delta-2). The start is X0_i = sin(pi i / n_x) on
G(a, n_x) (D-030); the second start (ChV-R1) is the plate b x a, not an option here.

- ``rectangle_first_halfstep`` is ``first_halfstep`` of R2 with the y grid of length b;
  it returns the R2 ``FirstHalfStepResult``, so ``second_halfstep``, ``as_record``,
  ``next_halfstep``, ``node_value`` and ``kkvim.energy`` are reused unchanged.
- ``rectangle_halfsteps`` runs a fixed number of half-steps like ``run_halfsteps``.
- ``run_r3_halfsteps`` takes the physical input a, b, t, E, nu, q, n_x, n_y: the R2
  checks (``_validated``, ``_stiffness``, N_MIN) and the exact range check of the sides
  ``validate_r3_sides`` (D-038, RV-2); invalid input is ERROR_INPUT.

w_bar = w D / (q a^4) with the reference side a (1.2). No stopping controller and no
energy test are run here. Errors are raised, never replaced by a field.
"""

import math
from dataclasses import dataclass

import numpy as np

from kkvim.first_halfstep import FirstHalfStepResult, _positive_input, normalization_factor, solve_halfstep
from kkvim.grid import Grid1D
from kkvim.iteration import HalfStepHistory, as_record, next_halfstep
from kkvim.physical import _stiffness, _validated
from kkvim.r3_input import validate_r3_sides


@dataclass(frozen=True)
class RectangleInput:
    a: float
    b: float
    t: float
    E: float
    nu: float
    q: float
    n_x: int
    n_y: int


@dataclass(frozen=True)
class RectangleHalfSteps:
    """Validated input, D and the half-steps s = 1 .. count."""

    input: RectangleInput
    D: float
    history: HalfStepHistory

    def dimensionless_field(self, s: int | None = None) -> np.ndarray:
        """w_bar = w(s) D / (q a^4), by default of the last half-step."""
        s = len(self.history) if s is None else s
        return self.history.field(s) * self.D / (self.input.q * self.input.a**4)


def rectangle_first_halfstep(a: float, b: float, D: float, q: float, n_x: int, n_y: int) -> FirstHalfStepResult:
    """Half-step s = 1 on a x b: X0 on G(a, n_x), Y1 solved on G(b, n_y)."""
    a, b = validate_r3_sides(a, b)
    D = _positive_input("D", D)
    grid_x = Grid1D(a, n_x)
    grid_y = Grid1D(b, n_y)

    x0 = np.sin(math.pi * np.arange(1, grid_x.n) / grid_x.n)
    step = solve_halfstep(grid_x, x0, grid_y, q, D)

    y1_raw = step.raw
    c = normalization_factor(y1_raw)
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
        D=D,
        q=float(q),
    )


def rectangle_halfsteps(a: float, b: float, D: float, q: float, n_x: int, n_y: int, count: int) -> HalfStepHistory:
    """Exactly ``count`` half-steps s = 1 .. count on a x b; no automatic stopping."""
    if isinstance(count, bool) or not isinstance(count, (int, np.integer)):
        raise TypeError(f"count must be an integer, got {count!r}")
    if count < 1:
        raise ValueError(f"count must be >= 1, got {count}")
    records = [as_record(rectangle_first_halfstep(a, b, D, q, n_x, n_y))]
    while len(records) < count:
        records.append(next_halfstep(records[-1]))
    return HalfStepHistory(records=tuple(records))


def validated_rectangle_input(a, b, t, E, nu, q, n_x, n_y) -> tuple[RectangleInput, float]:
    """R2 input checks, the range of the sides (D-038) and D; PhysicalInputError (ERROR_INPUT) otherwise."""
    physical = _validated(a, t, E, nu, q, n_x, n_y)
    a, b = validate_r3_sides(a, b)
    D = _stiffness(physical)
    rectangle = RectangleInput(a=a, b=b, t=physical.t, E=physical.E, nu=physical.nu, q=physical.q,
                               n_x=physical.n_x, n_y=physical.n_y)
    return rectangle, D


def run_r3_halfsteps(a, b, t, E, nu, q, n_x, n_y, count: int) -> RectangleHalfSteps:
    """Physical input a, b, t, E, nu, q -> D and ``count`` half-steps on a x b."""
    rectangle, D = validated_rectangle_input(a, b, t, E, nu, q, n_x, n_y)
    history = rectangle_halfsteps(rectangle.a, rectangle.b, D, rectangle.q, rectangle.n_x, rectangle.n_y, count)
    return RectangleHalfSteps(input=rectangle, D=D, history=history)
