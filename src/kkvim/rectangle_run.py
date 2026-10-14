"""KK-VIM on the rectangle a x b until the stopping rule (R3, stage B; D-040, D-041).

docs/R3_NUMERICAL_SPEC.md 4 (R2 stopping without change), 1.2 (input, w_bar) and
NUMERICAL_SPEC.md 11.1-11.5.

- ``run_rectangle_until_stopped`` is the loop of ``kkvim.controller.run_until_stopped``
  with only the half-step s = 1 taken from ``rectangle_first_halfstep``: the same
  ``next_halfstep``, energy test (``first_check``, ``check_pair``), field changes,
  pure ``stopping_decision``, parameters, statuses CONVERGED, STAGNATED,
  MAX_HALF_STEPS and error exits, which keep the completed half-steps. Invalid sides
  (D-038) are ERROR_INPUT like any other rejected input of s = 1.
- ``run_rectangle_physical`` takes a, b, t, E, nu, q with the checks of
  ``validated_rectangle_input`` and returns the run with D and w_bar = w D / (q a^4).
- ``halfstep_report`` lists the quantities of NUMERICAL_SPEC.md 11.5 for every
  completed half-step, from the stored results; it does not change the run.

The energy test status is not a stopping condition: UNRESOLVED stays UNRESOLVED, a
diagnostic E2 FAIL stays visible, a resolved E1 FAIL is ERROR_ENERGY_TEST.
"""

import math
import platform
import statistics
from dataclasses import dataclass
from importlib import metadata

import numpy as np

from kkvim.coefficients import ERROR_INPUT, HalfStepCoefficients, compute_coefficients
from kkvim.controller import (
    _ERRORS,
    S_MAX,
    S_MIN,
    TOL_E,
    TOL_INF,
    StepChanges,
    StoppingDiagnosticsError,
    StoppingResult,
    _ConfigurationError,
    _field_changes,
    _parameters,
    stopping_decision,
    thetas,
)
from kkvim.energy import EnergyCheck, EnergyError, check_pair, discrete_energy, first_check
from kkvim.first_halfstep import NOT_A_NODE_MESSAGE, NotANodeError, field_node_value
from kkvim.iteration import HalfStepHistory, HalfStepRecord, as_record, next_halfstep
from kkvim.linear_solver import ERROR_NONFINITE
from kkvim.physical import PhysicalInputError
from kkvim.rectangle import RectangleInput, rectangle_first_halfstep, validated_rectangle_input

_RECTANGLE_ERRORS = _ERRORS + (PhysicalInputError,)
CENTRE = ("1/2", "1/2")


def run_rectangle_until_stopped(
    a: float, b: float, D: float, q: float, n_x: int, n_y: int,
    tol_inf: float = TOL_INF, tol_E: float = TOL_E, s_min: int = S_MIN, s_max: int = S_MAX,
) -> StoppingResult:
    """Half-steps s = 1, 2, ... on a x b until CONVERGED, STAGNATED, MAX_HALF_STEPS or an error exit."""
    try:
        parameters = _parameters(n_x, n_y, tol_inf, tol_E, s_min, s_max)
    except _ConfigurationError as exc:
        return StoppingResult(ERROR_INPUT, HalfStepHistory(()), (), (), None, exc)

    records: list[HalfStepRecord] = []
    changes: list[StepChanges] = []
    checks: list[EnergyCheck] = []
    diagnostics: list[tuple[int, float, float]] = []

    def result(status, error=None):
        return StoppingResult(status, HalfStepHistory(tuple(records)), tuple(changes), tuple(checks), parameters, error)

    while True:
        s = len(records) + 1
        try:
            if s == 1:
                record = as_record(rectangle_first_halfstep(a, b, D, q, n_x, n_y))
            else:
                record = next_halfstep(records[-1])
        except _RECTANGLE_ERRORS as exc:
            return result(exc.code, exc)
        records.append(record)
        try:
            checks.append(first_check(record) if s == 1 else check_pair(records[-2], record))
        except EnergyError as exc:
            if exc.check is not None:
                checks.append(exc.check)
            return result(exc.code, exc)

        if s >= 2:
            eps_inf, eps_E, eps_L2 = _field_changes(records[-2], record)
            if not all(math.isfinite(v) for v in (eps_inf, eps_E, eps_L2)):
                return result(ERROR_NONFINITE, StoppingDiagnosticsError(
                    ERROR_NONFINITE, f"non-finite changes at s={s}: {eps_inf!r}, {eps_E!r}, {eps_L2!r}"))
            diagnostics.append((s, eps_inf, eps_E))
            theta_all = thetas(row[2] for row in diagnostics)
            theta = theta_all[-1]
            tail = None
            last_three = theta_all[-3:]
            if len(last_three) == 3 and all(t is not None for t in last_three):
                median = statistics.median(last_three)
                if median < 1.0:
                    tail = eps_E * median / (1.0 - median)
            satisfied = eps_inf <= parameters.tol_inf and eps_E <= parameters.tol_E
            changes.append(StepChanges(s, record.k, eps_inf, eps_E, eps_L2, theta, satisfied, tail))
        decision = stopping_decision(
            s, diagnostics, parameters.tol_inf, parameters.tol_E, parameters.s_min, parameters.s_max
        )
        if decision is not None:
            return result(decision)


@dataclass(frozen=True)
class RectangleResult:
    """Status, validated input, D and the controller result (None on input errors)."""

    status: str
    input: RectangleInput | None
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


def run_rectangle_physical(
    a: float, b: float, t: float, E: float, nu: float, q: float, n_x: int, n_y: int, *,
    tol_inf: float = TOL_INF, tol_E: float = TOL_E, s_min: int = S_MIN, s_max: int = S_MAX,
) -> RectangleResult:
    """Validate a, b, t, E, nu, q, n_x, n_y, compute D and run the stopping controller on a x b."""
    try:
        rectangle, D = validated_rectangle_input(a, b, t, E, nu, q, n_x, n_y)
    except PhysicalInputError as exc:
        return RectangleResult(ERROR_INPUT, None, None, None, exc)
    run = run_rectangle_until_stopped(rectangle.a, rectangle.b, D, rectangle.q, rectangle.n_x, rectangle.n_y,
                                      tol_inf=tol_inf, tol_E=tol_E, s_min=s_min, s_max=s_max)
    return RectangleResult(run.status, rectangle, D, run, run.error)


# report of NUMERICAL_SPEC.md 11.5 ------------------------------------------------------------

@dataclass(frozen=True)
class HalfStepReport:
    """Quantities of NUMERICAL_SPEC.md 11.5 for one completed half-step s."""

    s: int
    k: int
    direction: str  # computed factor: "Y" (odd s) or "X" (even s)
    solve_axis: str  # "y" on G(b, n_y) or "x" on G(a, n_x)
    solve_grid: tuple[float, int]  # (length, n) of the solved one-dimensional problem
    fixed_grid: tuple[float, int]  # (length, n) of the given factor
    coefficients: HalfStepCoefficients  # of the given factor after normalization of half-step s - 1
    coefficients_before_normalization: HalfStepCoefficients | None  # None for s = 1 (X0 is not normalized)
    beta: float  # B/A
    gamma: float  # C/A
    chi: float  # F^2/A
    c: float  # I-4 normalization factor of the computed factor
    x: np.ndarray  # normalized or scaled X factor of w(s), all nodes of G(a, n_x)
    y: np.ndarray  # Y factor of w(s), all nodes of G(b, n_y)
    residual_norm: float
    eta: float
    refined: bool
    imag_ratio: float
    P: float | None  # Pi_h(w(s)); None if the energy of s was not obtained
    norm_a: float | None  # |w(s)|_a
    changes: StepChanges | None  # eps_inf, eps_E, eps_L2, theta, tail estimate (s >= 2)
    energy: EnergyCheck | None  # Delta P_s, Q_s, gamma_s, tau_s and the test status
    control_values: tuple[tuple[tuple[str, str], float | str], ...]  # w_bar or NOT_A_NODE_MESSAGE


@dataclass(frozen=True)
class RunReport:
    """Report of a run: input, parameters, exit status, versions and the half-steps."""

    status: str
    input: RectangleInput | None
    D: float | None
    parameters: object
    error: str | None
    halfsteps: tuple[HalfStepReport, ...]
    versions: dict[str, str]


def halfstep_report(result: RectangleResult, control_points=(CENTRE,)) -> RunReport:
    """NUMERICAL_SPEC.md 11.5 for every completed half-step; control points are rational (x/a, y/b)."""
    if type(result) is not RectangleResult:
        raise TypeError(f"expected RectangleResult, got {type(result).__name__}")
    rows = ()
    if result.run is not None:
        run = result.run
        changes = {c.s: c for c in run.changes}
        checks = {c.s: c for c in run.energy_checks}
        records = run.history.records
        rows = tuple(
            _row(record, records[i - 1] if i else None, changes.get(record.s), checks.get(record.s), control_points)
            for i, record in enumerate(records)
        )
    return RunReport(
        status=result.status, input=result.input, D=result.D,
        parameters=None if result.run is None else result.run.parameters,
        error=None if result.error is None else str(result.error), halfsteps=rows, versions=_versions(),
    )


def _row(record: HalfStepRecord, previous: HalfStepRecord | None, changes, energy, control_points) -> HalfStepReport:
    gx, gy = record.grid_x, record.grid_y
    solve, fixed = (gy, gx) if record.direction == "Y" else (gx, gy)
    c = record.coefficients
    before = None if previous is None else compute_coefficients(fixed, previous.raw, record.q)
    try:
        value = discrete_energy(gx, gy, record.field(), record.D, record.q)
        P, norm_a = value.pi, value.norm_a
    except EnergyError:
        P = norm_a = None
    w_bar = record.field() * record.D / (record.q * gx.length**4)
    control = []
    for point in control_points:
        try:
            control.append((tuple(point), field_node_value(gx, gy, w_bar, *point)))
        except NotANodeError:
            control.append((tuple(point), NOT_A_NODE_MESSAGE))
    d = record.solver_diagnostics
    return HalfStepReport(
        s=record.s, k=record.k, direction=record.direction, solve_axis="y" if record.direction == "Y" else "x",
        solve_grid=(solve.length, solve.n), fixed_grid=(fixed.length, fixed.n), coefficients=c,
        coefficients_before_normalization=before, beta=c.B / c.A, gamma=c.C / c.A, chi=c.F**2 / c.A, c=record.c,
        x=gx.with_boundary(record.x), y=gy.with_boundary(record.y), residual_norm=d.residual_norm, eta=d.eta,
        refined=d.refined, imag_ratio=d.imag_ratio, P=P, norm_a=norm_a, changes=changes, energy=energy,
        control_values=tuple(control),
    )


def _versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for package in ("kkvim", "numpy", "scipy"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "unknown"
    return versions
