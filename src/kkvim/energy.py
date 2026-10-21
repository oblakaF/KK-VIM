"""Discrete energy Pi_h and the energy test of consecutive half-steps (R2, step 7).

docs/NUMERICAL_SPEC.md (D-032), Q-021:

- 4.1: Delta_h w = L_x w + L_y w at the interior nodes (zero on the boundary),
  a_h(w, w) = D hx hy sum (Delta_h w)^2, l_h(w) = q hx hy sum w,
  Pi_h(w) = a_h(w, w) / 2 - l_h(w); computed by two-dimensional assembly on the
  full field, independent of the one-dimensional coefficients A, B, C, F;
- 10.2: gamma_s = h rho^T d_f with rho = F 1 - K Y_raw and d_f = f_{s-2} - Y_raw
  on the recomputed axis, so that delta_P = Q - gamma;
- 10.3: rounding model R(w) = u (ceil(log2 N) + 16) S_hat(w) with the absolute
  stencil |Delta|_h w, and R_f = 4u (max(nx, ny)/pi)^2 |w(s)|_a |d|_a;
- 10.4: tau = |gamma| + 10 (R(w(s-1)) + R(w(s)) + R_f); E1: delta_P >= -tau;
  E2: |delta_P - Q| <= tau; resolved if Q >= 10 tau. Statuses NOT_APPLICABLE (s = 1),
  UNRESOLVED, PASS, FAIL; FAIL by E1 of a resolved test raises ERROR_ENERGY_TEST;
- 11.3: Pi_h(w(s)) >= 0 raises ERROR_DEGENERATE.

Fields are formed from the stored factors only for the computation; nothing here
stops the iterations.
"""

import math
from dataclasses import dataclass

import numpy as np

from kkvim.coefficients import ERROR_DEGENERATE
from kkvim.grid import Grid1D
from kkvim.iteration import HalfStepHistory, HalfStepRecord
from kkvim.linear_solver import ERROR_NONFINITE, apply_k

UNIT_ROUNDOFF = 2.0**-53
ERROR_ENERGY_TEST = "ERROR_ENERGY_TEST"

NOT_APPLICABLE = "NOT_APPLICABLE"
UNRESOLVED = "UNRESOLVED"
PASS = "PASS"
FAIL = "FAIL"


class EnergyError(RuntimeError):
    """Energy computation or test failure with a specification code; ``check`` holds the result if any."""

    def __init__(self, code: str, message: str, check=None):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.check = check


@dataclass(frozen=True)
class EnergyValue:
    """Pi_h(w) with its parts and the energy norm |w|_a = sqrt(a_h(w, w))."""

    pi: float
    a: float
    l: float
    norm_a: float


@dataclass(frozen=True)
class EnergyCheck:
    """Energy record of half-step s; test quantities are None for s = 1."""

    s: int
    k: int
    status: str
    P_now: float
    P_prev: float | None = None
    delta_P: float | None = None
    Q: float | None = None
    gamma: float | None = None
    R_prev: float | None = None
    R_now: float | None = None
    R_f: float | None = None
    tau: float | None = None
    resolution_ratio: float | None = None  # Q / (10 tau), None if tau = 0
    e1: bool | None = None
    e2: bool | None = None


def discrete_energy(grid_x: Grid1D, grid_y: Grid1D, w, D: float, q: float) -> EnergyValue:
    """Pi_h, a_h(w, w), l_h(w) and |w|_a of a full field w of shape (n_x + 1, n_y + 1)."""
    w = _checked_field(grid_x, grid_y, w)
    D, q = _finite("D", D), _finite("q", q)
    cell = grid_x.h * grid_y.h
    with np.errstate(over="ignore", invalid="ignore"):
        a = D * cell * float(np.sum(_laplacian(grid_x, grid_y, w) ** 2))
        l = q * cell * float(np.sum(w[1:-1, 1:-1]))
    value = EnergyValue(pi=0.5 * a - l, a=a, l=l, norm_a=math.sqrt(a))
    if not all(math.isfinite(v) for v in (value.pi, value.a, value.l, value.norm_a)):
        raise EnergyError(ERROR_NONFINITE, "energy is not finite")
    return value


def rounding_threshold(grid_x: Grid1D, grid_y: Grid1D, w, D: float, q: float) -> float:
    """R(w) = u (ceil(log2 N) + 16) S_hat(w), N = (n_x - 1)(n_y - 1) (NUMERICAL_SPEC.md 10.3)."""
    w = _checked_field(grid_x, grid_y, w)
    D, q = _finite("D", D), _finite("q", q)
    cell = grid_x.h * grid_y.h
    with np.errstate(over="ignore", invalid="ignore"):
        s_hat = D * cell * float(np.sum(np.abs(_laplacian(grid_x, grid_y, w)) * _abs_laplacian(grid_x, grid_y, w)))
        s_hat += abs(q) * cell * float(np.sum(np.abs(w[1:-1, 1:-1])))
    n_nodes = (grid_x.n - 1) * (grid_y.n - 1)
    ceil_log2 = (n_nodes - 1).bit_length()  # exact integer ceil(log2 N) for N >= 1
    value = UNIT_ROUNDOFF * (ceil_log2 + 16) * s_hat
    if not math.isfinite(value):
        raise EnergyError(ERROR_NONFINITE, "rounding threshold is not finite")
    return value


def check_pair(previous: HalfStepRecord, current: HalfStepRecord) -> EnergyCheck:
    """Energy test of consecutive half-steps s - 1, s >= 2 (NUMERICAL_SPEC.md 10.2-10.4)."""
    _check_consecutive(previous, current)
    gx, gy, D, q = current.grid_x, current.grid_y, current.D, current.q
    w_prev, w_now = previous.field(), current.field()
    e_prev = discrete_energy(gx, gy, w_prev, D, q)
    e_now = discrete_energy(gx, gy, w_now, D, q)
    _check_negative(previous.s, e_prev.pi)
    _check_negative(current.s, e_now.pi)

    d = w_prev - w_now
    e_d = discrete_energy(gx, gy, d, D, q)
    Q = 0.5 * e_d.a
    delta_P = e_prev.pi - e_now.pi
    gamma = _gamma(previous, current)
    R_prev = rounding_threshold(gx, gy, w_prev, D, q)
    R_now = rounding_threshold(gx, gy, w_now, D, q)
    R_f = 4.0 * UNIT_ROUNDOFF * (max(gx.n, gy.n) / math.pi) ** 2 * e_now.norm_a * e_d.norm_a
    tau = abs(gamma) + 10.0 * (R_prev + R_now + R_f)
    values = (Q, delta_P, gamma, R_prev, R_now, R_f, tau)
    if not all(math.isfinite(v) for v in values):
        raise EnergyError(ERROR_NONFINITE, f"non-finite energy test quantities at s={current.s}")

    e1 = delta_P >= -tau
    e2 = abs(delta_P - Q) <= tau
    resolved = Q >= 10.0 * tau
    if not resolved:
        status = UNRESOLVED
    else:
        status = PASS if (e1 and e2) else FAIL
    check = EnergyCheck(
        s=current.s, k=current.k, status=status, P_now=e_now.pi, P_prev=e_prev.pi, delta_P=delta_P, Q=Q,
        gamma=gamma, R_prev=R_prev, R_now=R_now, R_f=R_f, tau=tau,
        resolution_ratio=Q / (10.0 * tau) if tau > 0.0 else None, e1=e1, e2=e2,
    )
    if resolved and not e1:
        raise EnergyError(ERROR_ENERGY_TEST, f"resolved energy test fails E1 at s={current.s}", check)
    return check


def first_check(record: HalfStepRecord) -> EnergyCheck:
    """s = 1: energy P1 only; the test is NOT_APPLICABLE (no previous field)."""
    if type(record) is not HalfStepRecord or record.s != 1:
        raise ValueError("expected the HalfStepRecord of s = 1")
    e = discrete_energy(record.grid_x, record.grid_y, record.field(), record.D, record.q)
    _check_negative(1, e.pi)
    return EnergyCheck(s=1, k=0, status=NOT_APPLICABLE, P_now=e.pi)


def check_history(history: HalfStepHistory) -> tuple[EnergyCheck, ...]:
    """Energy records of all half-steps of a finished history, s = 1 .. count."""
    if type(history) is not HalfStepHistory:
        raise TypeError(f"expected HalfStepHistory, got {type(history).__name__}")
    records = history.records
    checks = [first_check(records[0])]
    for previous, current in zip(records, records[1:]):
        checks.append(check_pair(previous, current))
    return tuple(checks)


def _gamma(previous: HalfStepRecord, current: HalfStepRecord) -> float:
    """gamma_s = h rho^T d_f on the recomputed axis (NUMERICAL_SPEC.md 10.2)."""
    if current.direction == "Y":
        grid, previous_factor = current.grid_y, previous.y
    else:
        grid, previous_factor = current.grid_x, previous.x
    c = current.coefficients
    rho = c.F * np.ones(grid.interior_size) - apply_k(grid, c.A, c.B, c.C, current.D, current.raw)
    d_f = previous_factor - current.raw
    return grid.h * float(np.dot(rho, d_f))


def _check_consecutive(previous: HalfStepRecord, current: HalfStepRecord) -> None:
    if type(previous) is not HalfStepRecord or type(current) is not HalfStepRecord:
        raise TypeError("expected two HalfStepRecord objects")
    if current.s != previous.s + 1 or current.s < 2:
        raise ValueError(f"half-steps s={previous.s}, s={current.s} are not consecutive")
    if current.k != current.s // 2 or previous.k != previous.s // 2:
        raise ValueError("counters do not satisfy k = s // 2 (D-030)")
    expected = "Y" if current.s % 2 else "X"
    if current.direction != expected or previous.direction == current.direction:
        raise ValueError("directions do not alternate as Y (odd s), X (even s)")
    same = (
        current.grid_x == previous.grid_x and current.grid_y == previous.grid_y
        and current.D == previous.D and current.q == previous.q
    )
    if not same:
        raise ValueError("grids, D or q differ between the half-steps")
    # w(s-1) must lie in M_s: the fixed factor of s is the computed factor of s - 1 (ChV-11).
    if current.direction == "Y":
        consistent = np.array_equal(current.x, previous.computed * current.c)
    else:
        consistent = np.array_equal(current.y, previous.computed * current.c)
    if not consistent:
        raise ValueError(f"the fixed factor of s={current.s} is not the computed factor of s={previous.s}")


def _check_negative(s: int, pi: float) -> None:
    if not pi < 0.0:
        raise EnergyError(ERROR_DEGENERATE, f"Pi_h(w({s})) = {pi!r} >= 0")


def _laplacian(grid_x: Grid1D, grid_y: Grid1D, w: np.ndarray) -> np.ndarray:
    """(Delta_h w) at the interior nodes, shape (n_x - 1, n_y - 1)."""
    centre = w[1:-1, 1:-1]
    lx = (w[:-2, 1:-1] - 2.0 * centre + w[2:, 1:-1]) / grid_x.h**2
    ly = (w[1:-1, :-2] - 2.0 * centre + w[1:-1, 2:]) / grid_y.h**2
    return lx + ly


def _abs_laplacian(grid_x: Grid1D, grid_y: Grid1D, w: np.ndarray) -> np.ndarray:
    """(|Delta|_h w) of NUMERICAL_SPEC.md 10.3: moduli of the five values, positive weights."""
    a = np.abs(w)
    centre = a[1:-1, 1:-1]
    return (a[:-2, 1:-1] + 2.0 * centre + a[2:, 1:-1]) / grid_x.h**2 + (a[1:-1, :-2] + 2.0 * centre + a[1:-1, 2:]) / grid_y.h**2


def _checked_field(grid_x: Grid1D, grid_y: Grid1D, w) -> np.ndarray:
    w = np.asarray(w, dtype=float)
    if w.shape != (grid_x.n + 1, grid_y.n + 1):
        raise ValueError(f"field shape {w.shape} does not match the grids ({grid_x.n + 1}, {grid_y.n + 1})")
    if not np.all(np.isfinite(w)):
        raise EnergyError(ERROR_NONFINITE, "field contains NaN or infinity")
    if np.any(w[0, :] != 0.0) or np.any(w[-1, :] != 0.0) or np.any(w[:, 0] != 0.0) or np.any(w[:, -1] != 0.0):
        raise ValueError("field must be exactly zero on the boundary")
    return w


def _finite(name: str, value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"{name} must be a real number, got {value!r}")
    value = float(value)
    if not math.isfinite(value):
        raise EnergyError(ERROR_NONFINITE, f"{name} is not finite")
    return value
