"""CCCC, N = 1, exactly the given number of half-steps: the limited research mode of D-061. Never a converged or qualified solution.

    run_cccc_n1(a, b, D, q, n_x, n_y, half_steps)
    run_cccc_n1_physical(a, b, t, E, nu, q, n_x, n_y, half_steps)

"Policy" is docs/R5_CCCC_N1_LIMITED_POLICY.proposal.md, whose sections 2-6 and 11 are the specification by D-061.1.

| Component                                  | Specification        | Level        | Decision             | Check   |
| w = X(x) Y(y), half-steps Y, X, Y, ...     | MATH_SPEC 1.4, 10    | A (authors)  | D-024, D-049.3       | QA K-3  |
| energy, D4, reflection w_-1 = w_1, system  | Policy 2             | B (project)  | D-049.1              | B-1     |
| start sin^2, normalization I-4             | Policy 5             | B (project)  | D-049.1, D-049.3     | B-3, B-4|
| assembly of C (C-3+), premises P-1...P-5   | Policy 3             | V (numerics) | D-054.3, D-055.1     | B-2     |
| half-step solve B-C (DST-I + SMW)          | Policy 4             | V (numerics) | D-049.2              | QA K-1  |
| eta and its guard 100u; mode, statuses     | Policy 4, 6, 8       | V (numerics) | D-061.2, D-061.3     | B-5     |

Limits: P-4 is assumed (classical dot products in BLAS); B-C is not qualified on trajectories of working grids; nothing is claimed about
convergence or closeness to the solution of the continuous or the discrete problem. The guard has no fallback: no refinement, no other
assembly, no A-C, no repeated run. cond(S) is only written. TS-1 (c_f = 10) and |eta - eta_QA| <= 16u are criteria of the QA acceptance of
the first version on the cases of Policy 9; the solver neither computes nor applies them. No kkvim module of the SSSS scheme is used.

Statuses never raise: a refused input is a result with its status and reason, without a field.
"""

import math
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np
import scipy.fft

from kkvim.physical import PhysicalInputError
from kkvim.r3_input import validate_r3_sides
from kkvim.rectangle import validated_rectangle_input

MODE = "FIXED_HALF_STEPS_RESEARCH"
APPLICATION_STATUS = "RESEARCH_ONLY / PROVISIONAL / NOT_QUALIFIED"

COMPLETED_FIXED_HALF_STEPS = "COMPLETED_FIXED_HALF_STEPS"
NOT_EXECUTED_OUTSIDE_DOMAIN = "NOT_EXECUTED_OUTSIDE_DOMAIN"
ERROR_INPUT = "ERROR_INPUT"
ERROR_NONFINITE = "ERROR_NONFINITE"
UNRESOLVED_C3PLUS_PREMISE = "UNRESOLVED_C3PLUS_PREMISE"
PROTECTIVE_STOP_FACTORIZATION = "PROTECTIVE_STOP_FACTORIZATION"
STOPPED_ETA_GUARD = "STOPPED_ETA_GUARD"
STOP_ZERO_FIELD = "STOP_ZERO_FIELD"
HALF_STEP_DONE = "HALF_STEP_DONE"  # a row of history.csv before the last one; never the status of a run

PREMISES_ESTABLISHED = "ESTABLISHED_CONDITIONAL_ON_P4"
PREMISES_UNRESOLVED = "UNRESOLVED"
P4_ASSUMED = "ASSUMED: classical dot products in BLAS (D-055.1, D-057.3); cannot be checked"

U = 2.0**-53
ETA_MAX_C = 100.0 * U  # D-061.2, variant A: the guard of this mode only, not a qualified value
N_MIN, N_MAX = 7, 1280  # D-061.2: bounds of the domain, not tolerances
S_MIN, S_MAX = 1, 8  # D-061.2: bounds of the domain, not tolerances
K_EXTRA = 13  # k = m + 13 of the bound of C-3+ (Policy 3)
SMALLEST_NORMAL = 2.0**-1022
DBL_MAX = float(np.finfo(float).max)


@dataclass
class HalfStep:
    """One half-step; what it did not reach stays None. ``status`` of the last half-step is the status of the run."""

    s: int
    k: int
    direction: str  # the solved factor: Y for odd s, X for even s
    n_s: int
    h_s: float
    given: np.ndarray
    status: str = HALF_STEP_DONE
    reason: str | None = None
    A: float | None = None
    B: float | None = None
    C: float | None = None
    F: float | None = None
    C_L: float | None = None
    E: float | None = None
    premises_status: str | None = None
    premises_reasons: tuple = ()
    cond_S: float | None = None
    min_DM: float | None = None
    eta: float | None = None
    raw: np.ndarray | None = None  # the solution y before normalization
    c: float | None = None
    sigma: float | None = None
    next_given: np.ndarray | None = None
    next_conjugate: np.ndarray | None = None
    P_hat: float | None = None
    delta_P_hat: float | None = None
    field_change_inf: float | None = None
    alpha_centre: float | None = None

    @property
    def done(self) -> bool:
        return self.next_given is not None


@dataclass(frozen=True)
class CCCCN1Result:
    """The run; ``status`` of Policy 6. Every result is RESEARCH_ONLY / PROVISIONAL / NOT_QUALIFIED."""

    status: str
    reason: str | None
    a: object
    b: object
    D: float | None
    q: object
    n_x: object
    n_y: object
    half_steps_requested: object
    steps: tuple = ()
    physical: dict | None = None  # t, E, nu of run_cccc_n1_physical
    mode: str = field(default=MODE)
    application_status: str = field(default=APPLICATION_STATUS)

    @property
    def completed(self) -> tuple:
        return tuple(step for step in self.steps if step.done)

    @property
    def half_steps_completed(self) -> int:
        return len(self.completed)

    def factors(self):
        """(X, Y) on the inner nodes of the last completed half-step, or None; w = X ⊗ Y."""
        if not self.completed:
            return None
        return _field_factors(self.completed[-1])

    def field(self):
        """w with the boundary nodes (axis 0 = x, axis 1 = y) of the last completed half-step, or None."""
        factors = self.factors()
        return None if factors is None else _outer(*factors)

    def dimensionless_field(self):
        """w_bar = w D / (q a^4) (Policy 1), or None."""
        w = self.field()
        return None if w is None else w * self.D / (self.q * self.a**4)


class _Stop(Exception):
    """A row of the table of Policy 6 in the half-step: the status and the reason."""

    def __init__(self, status: str, reason: str):
        super().__init__(reason)
        self.status, self.reason = status, reason


# --- the difference operators of Policy 2 ----------------------------------------------------------------------------------------

def second_difference(v: np.ndarray, h: float) -> np.ndarray:
    """L v on the inner nodes, the boundary values being zero."""
    padded = np.zeros(v.size + 2)
    padded[1:-1] = v
    return (padded[:-2] - 2.0 * padded[1:-1] + padded[2:]) / h**2


def d4_apply(v: np.ndarray, h: float) -> np.ndarray:
    """D4 v: inner rows (1, -4, 6, -4, 1) / h^4, the first and the last row (7, -4, 1) / h^4 (the reflection w_-1 = w_1)."""
    padded = np.zeros(v.size + 4)
    padded[2:-2] = v
    out = padded[:-4] - 4.0 * padded[1:-3] + 6.0 * padded[2:-2] - 4.0 * padded[3:-1] + padded[4:]
    out[0] = 7.0 * v[0] - 4.0 * v[1] + v[2]
    out[-1] = 7.0 * v[-1] - 4.0 * v[-2] + v[-3]
    return out / h**4


def _third_order_sums(g: np.ndarray) -> np.ndarray:
    """g[i-1] - 2 g[i] + g[i+1], each with one correct rounding of the exact sum; the values outside the grid are 0 (Policy 3)."""
    values = [0.0, *g.tolist(), 0.0]
    return np.array([float(Fraction(values[i - 1]) - 2 * Fraction(values[i]) + Fraction(values[i + 1]))
                     for i in range(1, len(values) - 1)])


# --- assembly: the normative order of operations of Policy 3 -----------------------------------------------------------------------

def assemble(g: np.ndarray, l_f: float, n_f: int, q: float) -> dict:
    """A, B, C (by C-3+), F of the given factor g, and the intermediates of C for the premises."""
    with np.errstate(all="ignore"):
        G = g[:, None]
        h = l_f / n_f
        A = h * (G.T @ G)[0, 0]
        Lg = second_difference(g, h)
        B = -h * (G.T @ Lg[:, None])[0, 0]
        F = q * h * g.sum()
        Y3 = _third_order_sums(g)
        h2 = h * h
        h3 = h2 * h
        s = 2.0 / h3
        X = (Y3 / h2)[:, None]
        C_L = h * (X.T @ X)[0, 0]
        E = s * (g[0] * g[0] + g[-1] * g[-1])
        C = C_L + E
    return {"A": float(A), "B": float(B), "C": float(C), "F": float(F), "C_L": float(C_L), "E": float(E), "Y3": Y3, "X": X[:, 0],
            "g1": float(g[0]), "gm": float(g[-1]), "h": h, "h2": h2, "h3": h3, "s": s}


def rounding_probe_failures() -> list:
    """P-1 and P-3 at this moment: round to nearest in Python, NumPy and a matrix product; subnormal values are neither flushed nor zeroed."""
    half = 2.0 * U  # the spacing of the doubles above 1
    failed = []
    if not (1.0 + 0.75 * half == 1.0 + half and 1.0 + 0.25 * half == 1.0):
        failed.append("P-1: round to nearest, Python")
    ones = np.array([1.0, 1.0])
    if (ones + np.array([0.75, 0.25]) * half).tolist() != [1.0 + half, 1.0]:
        failed.append("P-1: round to nearest, NumPy")
    if float((np.array([[1.0, 0.75 * half]]) @ ones[:, None])[0, 0]) != 1.0 + half:
        failed.append("P-1: round to nearest, matrix product")
    subnormal = np.array([SMALLEST_NORMAL / 2.0, 2.0**-1074])
    if not (SMALLEST_NORMAL / 2.0 == 2.0**-1023 and float((np.array([SMALLEST_NORMAL]) / 2.0)[0]) == 2.0**-1023):
        failed.append("P-3: flush to zero")
    if not (float((subnormal + subnormal)[1]) == 2.0**-1073 and float((subnormal * 1.0)[1]) != 0.0):
        failed.append("P-3: subnormal input treated as zero")
    return failed


def premise_failures(parts: dict, n_f: int) -> list:
    """P-2, P-3 and P-5 of Policy 3 for the values of ``assemble``. P-4 cannot be checked; it is written as assumed."""
    failed = []
    values = {name: np.atleast_1d(parts[name]) for name in ("Y3", "X", "g1", "gm", "C", "C_L", "E", "h", "h2", "h3", "s")}
    for name, value in values.items():
        if not np.all(np.isfinite(value)):
            failed.append(f"P-2: {name} не конечно")
            continue
        nonzero = np.abs(value[value != 0.0])
        if nonzero.size and float(nonzero.min()) < SMALLEST_NORMAL:
            failed.append(f"P-3: {name} субнормально")
    if failed:
        return failed
    for name in ("h", "h2", "h3"):
        if parts[name] == 0.0:
            failed.append(f"P-3: {name} = 0")
    m = n_f - 1
    x = np.abs(parts["X"])
    with np.errstate(all="ignore"):
        corners = max(parts["g1"] * parts["g1"], parts["gm"] * parts["gm"])
        if not m * float(x.max()) ** 2 < DBL_MAX:
            failed.append("P-2: m max|X|^2 не меньше DBL_MAX")
        if not 2.0 * corners * parts["s"] < DBL_MAX:
            failed.append("P-2: 2 max(g1^2, gm^2) s не меньше DBL_MAX")
        smallest = {"X": x[x != 0.0].min() if np.any(x != 0.0) else None, "g1": abs(parts["g1"]) or None, "gm": abs(parts["gm"]) or None}
        for name, value in smallest.items():
            if value is None:
                continue
            square = float(value) * float(value)
            if square < SMALLEST_NORMAL:
                failed.append(f"P-3: квадрат наименьшего ненулевого |{name}| ниже 2^-1022")
            elif name != "X" and square * parts["s"] < SMALLEST_NORMAL:
                failed.append(f"P-3: произведение {name}^2 s ниже 2^-1022")
    if not (m + K_EXTRA) * U < 1.0:
        failed.append("P-5: k u не меньше 1")
    return failed


# --- the half-step: B-C of Policy 4 ------------------------------------------------------------------------------------------------

def _dst(v: np.ndarray) -> np.ndarray:
    return scipy.fft.dst(v, type=1, norm="ortho")


def solve_bc(step: HalfStep, A: float, B: float, C: float, F: float, D: float, l_s: float, n_s: int) -> np.ndarray:
    """y of D [A D4 - 2 B L + C I] y = F 1: DST-I for A L^2 - 2 B L + C I, Sherman-Morrison-Woodbury for the rank-2 correction of D4."""
    if A <= 0.0:
        raise _Stop(PROTECTIVE_STOP_FACTORIZATION, f"A = {A!r} ≤ 0")
    h = l_s / n_s
    m = n_s - 1
    k = np.arange(1, n_s)
    with np.errstate(all="ignore"):
        mu = 4.0 / h**2 * np.sin(k * math.pi / (2 * n_s)) ** 2
        DM = D * (mu**2 * A + 2.0 * mu * B + C)
        if not np.all(np.isfinite(DM)):
            raise _Stop(ERROR_NONFINITE, "DM не конечно")
        step.min_DM = float(DM.min())
        if step.min_DM <= 0.0:
            raise _Stop(PROTECTIVE_STOP_FACTORIZATION, f"min DM = {step.min_DM!r} ≤ 0")
        Minv = 1.0 / DM
        x0 = _dst(F * _dst(np.ones(m)) / DM)
        e_1, e_m = np.zeros(m), np.zeros(m)
        e_1[0], e_m[-1] = 1.0, 1.0
        Y0 = np.column_stack([_dst(Minv * _dst(e_1)), _dst(Minv * _dst(e_m))])
        psi = math.sqrt(2.0 / n_s) * np.sin(k * math.pi / n_s)
        z = np.stack([psi, np.where(k % 2 == 1, 1.0, -1.0) * psi], axis=1)
        S = h**4 / (2.0 * D) * (1.0 / A) * np.eye(2) + np.einsum("k,ks,kt->st", Minv, z, z)
        if not np.all(np.isfinite(S)):
            raise _Stop(ERROR_NONFINITE, "S не конечно")
        try:
            np.linalg.cholesky(S)
            corrections = np.linalg.solve(S, x0[[0, m - 1]])
        except np.linalg.LinAlgError as exc:
            raise _Stop(PROTECTIVE_STOP_FACTORIZATION, f"S: {exc}") from exc
        step.cond_S = float(np.linalg.cond(S))
        return x0 - Y0 @ corrections


def residual_eta(y: np.ndarray, A: float, B: float, C: float, F: float, D: float, h: float) -> float:
    """The relative residual of the original system, without the modal transformation (Policy 4)."""
    with np.errstate(all="ignore"):
        r = F - D * (A * d4_apply(y, h) - 2.0 * B * second_difference(y, h) + C * y)
        return float(np.max(np.abs(r)) / (D * (16 * abs(A) / h**4 + 8 * abs(B) / h**2 + abs(C)) * np.max(np.abs(y)) + abs(F)))


# --- the sequence: Policy 5 and 6 ------------------------------------------------------------------------------------------------

def start_factor(n_x: int) -> np.ndarray:
    return np.sin(math.pi * np.arange(1, n_x) / n_x) ** 2


def _with_boundary(v: np.ndarray) -> np.ndarray:
    return np.concatenate(([0.0], v, [0.0]))


def _outer(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return np.outer(_with_boundary(X), _with_boundary(Y))


def _field_factors(step: HalfStep):
    """(X, Y): at odd s the solved factor is Y, at even s it is X (Policy 5, item 5)."""
    return (step.next_conjugate, step.next_given) if step.s % 2 == 1 else (step.next_given, step.next_conjugate)


def discrete_energy(X: np.ndarray, Y: np.ndarray, a: float, b: float, D: float, q: float, n_x: int, n_y: int) -> float:
    """P_hat: the energy of Policy 2 at w = X ⊗ Y through the moments of the factors (Policy 6; derivation of the project, level B)."""
    x, y = assemble(X, a, n_x, 1.0), assemble(Y, b, n_y, 1.0)
    bending = 0.5 * D * (x["C"] * y["A"] + 2.0 * x["B"] * y["B"] + x["A"] * y["C"])
    return bending - q * (x["h"] * float(X.sum())) * (y["h"] * float(Y.sum()))


def _half_step(step: HalfStep, l_f: float, n_f: int, l_s: float, D: float, q: float) -> None:
    """Fills the step up to the normalization (Policy 5, item 3); a row of the table of Policy 6 raises ``_Stop``."""
    g = step.given
    if not np.all(np.isfinite(g)):
        raise _Stop(ERROR_NONFINITE, "заданный множитель не конечен")
    failures = rounding_probe_failures()
    parts = assemble(g, l_f, n_f, q)
    step.A, step.B, step.C, step.F, step.C_L, step.E = (parts[name] for name in ("A", "B", "C", "F", "C_L", "E"))
    failures += premise_failures(parts, n_f)
    step.premises_status = PREMISES_UNRESOLVED if failures else PREMISES_ESTABLISHED
    step.premises_reasons = tuple(failures)
    if not all(math.isfinite(value) for value in (step.A, step.B, step.F)):
        raise _Stop(ERROR_NONFINITE, "A, B или F не конечно")
    if failures:
        raise _Stop(UNRESOLVED_C3PLUS_PREMISE, "; ".join(failures))
    y = solve_bc(step, step.A, step.B, step.C, step.F, D, l_s, step.n_s)
    failures = rounding_probe_failures()
    if failures:
        step.premises_status, step.premises_reasons = PREMISES_UNRESOLVED, tuple(failures)
        raise _Stop(UNRESOLVED_C3PLUS_PREMISE, "; ".join(failures))
    if not np.all(np.isfinite(y)):
        raise _Stop(ERROR_NONFINITE, "y не конечно")
    step.raw = y
    if not np.any(y):  # eta is not defined at y = 0 and F = 0
        raise _Stop(STOP_ZERO_FIELD, "y = 0 точно")
    step.eta = residual_eta(y, step.A, step.B, step.C, step.F, D, step.h_s)
    if not step.eta <= ETA_MAX_C:  # NaN does not pass
        raise _Stop(STOPPED_ETA_GUARD, f"η = {step.eta!r} > η_max^C = 100u (D-061.2)")
    step.sigma = 1.0 if math.fsum(y.tolist()) >= 0 else -1.0
    step.c = step.sigma * float(np.max(np.abs(y)))
    with np.errstate(all="ignore"):
        next_given, next_conjugate = y / step.c, g * step.c
    if not (math.isfinite(step.c) and np.all(np.isfinite(next_given)) and np.all(np.isfinite(next_conjugate))):
        raise _Stop(ERROR_NONFINITE, "c или новые множители не конечны")
    step.next_given, step.next_conjugate = next_given, next_conjugate


def _execute(a: float, b: float, D: float, q: float, n_x: int, n_y: int, half_steps: int, physical=None) -> CCCCN1Result:
    steps = []
    given = start_factor(n_x)
    previous_energy, previous_field = 0.0, None  # w(0) = 0
    status, reason = COMPLETED_FIXED_HALF_STEPS, None
    centre = (n_x // 2, n_y // 2) if n_x % 2 == 0 and n_y % 2 == 0 else None
    for s in range(1, half_steps + 1):
        l_f, n_f, l_s, n_s = (a, n_x, b, n_y) if s % 2 == 1 else (b, n_y, a, n_x)
        step = HalfStep(s=s, k=s // 2, direction="Y" if s % 2 == 1 else "X", n_s=n_s, h_s=l_s / n_s, given=given)
        steps.append(step)
        try:
            _half_step(step, l_f, n_f, l_s, D, q)
        except _Stop as stop:
            status, reason = stop.status, stop.reason
            step.status, step.reason = status, reason
            break
        X, Y = _field_factors(step)
        w = _outer(X, Y)
        largest = float(np.max(np.abs(w)))
        step.field_change_inf = (float(np.max(np.abs(w - previous_field))) if previous_field is not None else largest) / largest
        step.P_hat = discrete_energy(X, Y, a, b, D, q, n_x, n_y)
        step.delta_P_hat = previous_energy - step.P_hat
        if centre is not None:
            step.alpha_centre = float(w[centre]) * D / (q * min(a, b) ** 4)
        previous_energy, previous_field, given = step.P_hat, w, step.next_given
    else:
        steps[-1].status = COMPLETED_FIXED_HALF_STEPS
    return CCCCN1Result(status, reason, a, b, D, q, n_x, n_y, half_steps, tuple(steps), physical)


# --- refusals and the two entries ------------------------------------------------------------------------------------------------

def _is_integer(value) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)


def _is_positive_finite(value) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def domain_refusal(a, b, n_x, n_y, half_steps) -> str | None:
    """Why the input is outside the domain of D-061.2, or None. Values that are not valid numbers are left to the input checks."""
    if not _is_integer(half_steps) or not S_MIN <= half_steps <= S_MAX:
        return f"число полушагов должно быть целым от {S_MIN} до {S_MAX}, получено {half_steps!r}"
    for name, value in (("n_x", n_x), ("n_y", n_y)):
        if _is_integer(value) and not N_MIN <= value <= N_MAX:
            return f"{name} должно быть от {N_MIN} до {N_MAX}, получено {value!r}"
    if _is_positive_finite(a) and _is_positive_finite(b):
        try:
            validate_r3_sides(a, b)
        except PhysicalInputError as exc:
            return f"вне диапазона 0,2 ≤ b/a ≤ 5: {exc}"
    return None


def _not_run(status, reason, a, b, D, q, n_x, n_y, half_steps, physical=None) -> CCCCN1Result:
    return CCCCN1Result(status, reason, a, b, D, q, n_x, n_y, half_steps, (), physical)


def run_cccc_n1(a, b, D, q, n_x, n_y, half_steps) -> CCCCN1Result:
    """Plate a x b, stiffness D, load q != 0, clamped on all sides, N = 1: exactly ``half_steps`` half-steps, starting with Y."""
    reason = domain_refusal(a, b, n_x, n_y, half_steps)
    if reason is not None:
        return _not_run(NOT_EXECUTED_OUTSIDE_DOMAIN, reason, a, b, D, q, n_x, n_y, half_steps)
    for name, value in (("a", a), ("b", b), ("D", D)):
        if not _is_positive_finite(value):
            return _not_run(ERROR_INPUT, f"{name} должно быть конечным и положительным, получено {value!r}", a, b, D, q, n_x, n_y, half_steps)
    if isinstance(q, bool) or not isinstance(q, (int, float, np.integer, np.floating)) or not math.isfinite(q) or q == 0:
        return _not_run(ERROR_INPUT, f"q должно быть конечным и ненулевым, получено {q!r}", a, b, D, q, n_x, n_y, half_steps)
    for name, value in (("n_x", n_x), ("n_y", n_y)):
        if not _is_integer(value):
            return _not_run(ERROR_INPUT, f"{name} должно быть целым, получено {value!r}", a, b, D, q, n_x, n_y, half_steps)
    return _execute(float(a), float(b), float(D), float(q), int(n_x), int(n_y), int(half_steps))


def run_cccc_n1_physical(a, b, t, E, nu, q, n_x, n_y, half_steps) -> CCCCN1Result:
    """The same with the physical input; D = E t^3 / (12 (1 - nu^2)) and the checks of ``validated_rectangle_input``."""
    physical = {"t": t, "E": E, "nu": nu}
    reason = domain_refusal(a, b, n_x, n_y, half_steps)
    if reason is not None:
        return _not_run(NOT_EXECUTED_OUTSIDE_DOMAIN, reason, a, b, None, q, n_x, n_y, half_steps, physical)
    try:
        plate, D = validated_rectangle_input(a, b, t, E, nu, q, n_x, n_y)
    except PhysicalInputError as exc:
        return _not_run(ERROR_INPUT, str(exc), a, b, None, q, n_x, n_y, half_steps, physical)
    return _execute(plate.a, plate.b, D, plate.q, plate.n_x, plate.n_y, int(half_steps), {"t": plate.t, "E": plate.E, "nu": plate.nu})
