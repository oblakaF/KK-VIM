"""Physical input of R2: a, t, E, nu, q -> D and the stopping controller (D-009, D-032, D-033).

docs/NUMERICAL_SPEC.md 9.2 (I-5): a, t, E finite and > 0; q finite and != 0;
|nu| < 1 (D > 0, MATH_SPEC B-1); no other restriction (in particular none on t/a).
D = E t^3 / (12 (1 - nu^2)) must be finite and positive; otherwise ERROR_INPUT, so an
overflowed, zero or non-numeric D is never used. Units must be consistent; no unit
conversion is done here.

The run itself is ``run_until_stopped(a, D, q, n_x, n_y, ...)`` unchanged. The
dimensionless field w_bar = w D / (q a^4) (MATH_SPEC 2.5) is taken from the field of
the last completed half-step; without a computed field there is no w_bar.
"""

import math
from dataclasses import dataclass

import numpy as np

from kkvim.coefficients import ERROR_INPUT
from kkvim.controller import S_MAX, S_MIN, TOL_E, TOL_INF, StoppingResult, run_until_stopped
from kkvim.grid import N_MIN


@dataclass(frozen=True)
class PhysicalInput:
    a: float
    t: float
    E: float
    nu: float
    q: float
    n_x: int
    n_y: int


@dataclass(frozen=True)
class PhysicalResult:
    """Status, physical input, computed D and the controller result (None on input errors)."""

    status: str
    input: PhysicalInput | None
    D: float | None
    run: StoppingResult | None
    error: Exception | None = None

    def dimensionless_field(self) -> np.ndarray | None:
        """w_bar = w D / (q a^4) of the last completed half-step, or None if there is no field."""
        if self.run is None:
            return None
        w = self.run.last_field()
        if w is None:
            return None
        return w * self.D / (self.input.q * self.input.a**4)


class PhysicalInputError(ValueError):
    """Invalid physical input (ERROR_INPUT)."""

    code = ERROR_INPUT


def bending_stiffness(E: float, t: float, nu: float) -> float:
    """D = E t^3 / (12 (1 - nu^2)) without validation."""
    return E * t**3 / (12.0 * (1.0 - nu**2))


def run_physical(
    a: float, t: float, E: float, nu: float, q: float, n_x: int, n_y: int, *,
    tol_inf: float = TOL_INF, tol_E: float = TOL_E, s_min: int = S_MIN, s_max: int = S_MAX,
) -> PhysicalResult:
    """Validate the physical input, compute D and run the stopping controller."""
    try:
        physical = _validated(a, t, E, nu, q, n_x, n_y)
        D = _stiffness(physical)
    except PhysicalInputError as exc:
        return PhysicalResult(ERROR_INPUT, None, None, None, exc)
    run = run_until_stopped(physical.a, D, physical.q, physical.n_x, physical.n_y,
                            tol_inf=tol_inf, tol_E=tol_E, s_min=s_min, s_max=s_max)
    return PhysicalResult(run.status, physical, D, run, run.error)


def _validated(a, t, E, nu, q, n_x, n_y) -> PhysicalInput:
    values = {}
    for name, value in (("a", a), ("t", t), ("E", E), ("nu", nu), ("q", q)):
        if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise PhysicalInputError(f"{name} must be a real number, got {value!r}")
        value = float(value)
        if not math.isfinite(value):
            raise PhysicalInputError(f"{name} must be finite, got {value!r}")
        values[name] = value
    for name in ("a", "t", "E"):
        if values[name] <= 0.0:
            raise PhysicalInputError(f"{name} must be positive, got {values[name]!r}")
    if not abs(values["nu"]) < 1.0:
        raise PhysicalInputError(f"|nu| must be < 1, got {values['nu']!r}")
    if values["q"] == 0.0:
        raise PhysicalInputError("q must be nonzero")
    for name, value in (("n_x", n_x), ("n_y", n_y)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < N_MIN:
            raise PhysicalInputError(f"{name} must be an integer >= {N_MIN}, got {value!r}")
    return PhysicalInput(n_x=int(n_x), n_y=int(n_y), **values)


def _stiffness(physical: PhysicalInput) -> float:
    with np.errstate(over="ignore", under="ignore"):
        try:
            D = bending_stiffness(physical.E, physical.t, physical.nu)
        except OverflowError:
            D = math.inf
    if not (math.isfinite(D) and D > 0.0):
        raise PhysicalInputError(f"bending stiffness D = {D!r} is not finite and positive")
    return D
