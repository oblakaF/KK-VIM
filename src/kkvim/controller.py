"""Automatic stopping of the KK-VIM half-steps (R2, step 8; D-032, D-033).

docs/NUMERICAL_SPEC.md, section 11 (Q-009, I-10):

- 11.1, for every s >= 2 on the full field:
  eps_inf = max|w(s) - w(s-1)| / max|w(s)|; eps_E = |w(s) - w(s-1)|_a / |w(s)|_a
  (from the difference field, not from energy values); eps_L2 = |.|_h / |.|_h
  (diagnostic); theta_s = eps_E(s) / eps_E(s-1) for s >= 3, undefined (None) if the
  denominator is zero; the energy test of section 10 (``kkvim.energy``);
- 11.2: defaults tol_inf = 1e-10, tol_E = 1e-8, s_min = 3, s_max = 100;
  eps_floor = 10 u (max(n_x, n_y) / pi)^2; tol_E < 10 eps_floor is ERROR_INPUT;
  strict mode tol_inf / 100 and max(tol_E / 100, 10 eps_floor) (``strict_parameters``);
- 11.3, checked in this order after each half-step:
  CONVERGED if s >= s_min and eps_inf <= tol_inf and eps_E <= tol_E hold for both
  s - 1 and s; STAGNATED after 10 consecutive half-steps with theta >= 0.9 and
  eps_E > tol_E; MAX_HALF_STEPS at s = s_max. Error exits ERROR_LINEAR_SOLVE,
  ERROR_NONFINITE, ERROR_DEGENERATE, ERROR_ENERGY_TEST, ERROR_INPUT keep the
  completed half-steps. The energy test status is not a stopping condition; a
  diagnostic E2 FAIL stays visible in the energy checks.
- The stopping rule is the pure function ``stopping_decision`` on the sequence
  (s, eps_inf, eps_E); ``run_until_stopped`` only computes the quantities and acts
  on its decision.
- Tail diagnostic T_s = eps_E theta_bar / (1 - theta_bar), theta_bar the median of
  the last three theta, only if all three are defined and theta_bar < 1. It is an
  estimate under the assumption of contraction, not a proven error bound.
"""

import math
import statistics
from dataclasses import dataclass

import numpy as np

from kkvim.coefficients import ERROR_INPUT, CoefficientError
from kkvim.energy import EnergyCheck, EnergyError, check_pair, discrete_energy, first_check
from kkvim.first_halfstep import HalfStepError, first_halfstep
from kkvim.grid import N_MIN
from kkvim.iteration import HalfStepHistory, HalfStepRecord, as_record, next_halfstep
from kkvim.linear_solver import LinearSolveError

UNIT_ROUNDOFF = 2.0**-53
TOL_INF = 1e-10
TOL_E = 1e-8
S_MIN = 3
S_MAX = 100
STAGNATION_THETA = 0.9
STAGNATION_COUNT = 10

CONVERGED = "CONVERGED"
STAGNATED = "STAGNATED"
MAX_HALF_STEPS = "MAX_HALF_STEPS"

TAIL_ESTIMATE_NOTE = "оценка при предположении сжатия; не доказанная граница ошибки"

_ERRORS = (CoefficientError, LinearSolveError, HalfStepError, EnergyError)


@dataclass(frozen=True)
class StoppingParameters:
    tol_inf: float
    tol_E: float
    s_min: int
    s_max: int
    eps_floor: float


@dataclass(frozen=True)
class StepChanges:
    """Quantities of section 11.1 for half-step s >= 2."""

    s: int
    k: int
    eps_inf: float
    eps_E: float
    eps_L2: float
    theta: float | None  # None for s = 2 or a zero denominator
    satisfied: bool  # eps_inf <= tol_inf and eps_E <= tol_E
    tail_estimate: float | None  # T_s; see TAIL_ESTIMATE_NOTE


@dataclass(frozen=True)
class StoppingResult:
    """Exit status, completed half-steps and their diagnostics."""

    status: str
    history: HalfStepHistory  # completed half-steps (empty if none)
    changes: tuple[StepChanges, ...]  # s = 2 .. last
    energy_checks: tuple[EnergyCheck, ...]  # s = 1 .. last (incl. a failing check)
    parameters: StoppingParameters | None
    error: Exception | None = None
    tail_estimate_note: str = TAIL_ESTIMATE_NOTE

    @property
    def s(self) -> int:
        return len(self.history)

    @property
    def k(self) -> int:
        return self.s // 2

    @property
    def converged(self) -> bool:
        return self.status == CONVERGED

    def last_field(self) -> np.ndarray | None:
        """Deflection field w(s) of the last completed half-step."""
        return self.history.field(self.s) if self.s else None


def thetas(eps_E_values) -> list[float | None]:
    """theta_s = eps_E(s) / eps_E(s-1) along eps_E of s = 2, 3, ...; None for s = 2 and zero denominators."""
    values = list(eps_E_values)
    return [None] + [None if prev == 0.0 else now / prev for prev, now in zip(values, values[1:])]


def stopping_decision(s: int, diagnostics, tol_inf: float, tol_E: float, s_min: int, s_max: int) -> str | None:
    """Pure stopping rule of NUMERICAL_SPEC.md 11.3 after half-step s.

    ``diagnostics`` holds (s', eps_inf, eps_E) for s' = 2 .. s in order (empty for s = 1).
    Returns CONVERGED, STAGNATED or MAX_HALF_STEPS in this priority, or None to continue.
    It does not run the solver, form fields or modify the diagnostics.
    """
    rows = tuple(diagnostics)
    if [row[0] for row in rows] != list(range(2, s + 1)):
        raise ValueError(f"diagnostics must cover s = 2 .. {s} in order")
    satisfied = [eps_inf <= tol_inf and eps_E <= tol_E for _, eps_inf, eps_E in rows]
    if s >= s_min and len(rows) >= 2 and satisfied[-1] and satisfied[-2]:
        return CONVERGED
    run = 0
    for theta, (_, _, eps_E) in zip(thetas(row[2] for row in rows), rows):
        run = run + 1 if (theta is not None and theta >= STAGNATION_THETA and eps_E > tol_E) else 0
    if run >= STAGNATION_COUNT:
        return STAGNATED
    if s >= s_max:
        return MAX_HALF_STEPS
    return None


def eps_floor(n_x: int, n_y: int) -> float:
    """eps_floor = 10 u (max(n_x, n_y) / pi)^2 (NUMERICAL_SPEC.md 11.2)."""
    return 10.0 * UNIT_ROUNDOFF * (max(n_x, n_y) / math.pi) ** 2


def strict_parameters(n_x: int, n_y: int, tol_inf: float = TOL_INF, tol_E: float = TOL_E) -> tuple[float, float]:
    """(tol_inf / 100, max(tol_E / 100, 10 eps_floor)) of the strict mode (11.2, 11.4)."""
    return tol_inf / 100.0, max(tol_E / 100.0, 10.0 * eps_floor(n_x, n_y))


def run_until_stopped(
    a: float, D: float, q: float, n_x: int, n_y: int,
    tol_inf: float = TOL_INF, tol_E: float = TOL_E, s_min: int = S_MIN, s_max: int = S_MAX,
) -> StoppingResult:
    """Half-steps s = 1, 2, ... until CONVERGED, STAGNATED, MAX_HALF_STEPS or an error exit."""
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
            record = as_record(first_halfstep(a, D, q, n_x, n_y)) if s == 1 else next_halfstep(records[-1])
        except _ERRORS as exc:
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


def _field_changes(previous: HalfStepRecord, current: HalfStepRecord) -> tuple[float, float, float]:
    """(eps_inf, eps_E, eps_L2) of section 11.1 from the difference field."""
    w_prev, w_now = previous.field(), current.field()
    d = w_now - w_prev
    gx, gy = current.grid_x, current.grid_y
    eps_inf = float(np.max(np.abs(d))) / float(np.max(np.abs(w_now)))
    norm_d = discrete_energy(gx, gy, d, current.D, current.q).norm_a
    norm_now = discrete_energy(gx, gy, w_now, current.D, current.q).norm_a
    cell = gx.h * gy.h
    eps_L2 = math.sqrt(cell * float(np.sum(d * d))) / math.sqrt(cell * float(np.sum(w_now * w_now)))
    return eps_inf, norm_d / norm_now, eps_L2


class _ConfigurationError(ValueError):
    pass


def _parameters(n_x, n_y, tol_inf, tol_E, s_min, s_max) -> StoppingParameters:
    for name, value in (("tol_inf", tol_inf), ("tol_E", tol_E)):
        if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise _ConfigurationError(f"{name} must be a real number, got {value!r}")
        if not math.isfinite(value) or value <= 0.0:
            raise _ConfigurationError(f"{name} must be finite and positive, got {value!r}")
    for name, value in (("s_min", s_min), ("s_max", s_max)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise _ConfigurationError(f"{name} must be a positive integer, got {value!r}")
    if s_min > s_max:
        raise _ConfigurationError(f"s_min={s_min} exceeds s_max={s_max}")
    for name, value in (("n_x", n_x), ("n_y", n_y)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < N_MIN:
            raise _ConfigurationError(f"{name} must be an integer >= {N_MIN}, got {value!r}")
    floor = eps_floor(n_x, n_y)
    if tol_E < 10.0 * floor:
        raise _ConfigurationError(f"tol_E={tol_E!r} < 10 eps_floor = {10.0 * floor!r} (unattainable)")
    return StoppingParameters(float(tol_inf), float(tol_E), int(s_min), int(s_max), floor)
