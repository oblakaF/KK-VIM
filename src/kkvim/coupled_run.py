"""Research controller of coupled KK-VIM half-steps at fixed N = 2-3 (R4; D-046, D-047, D-048). Internal.

Levels (AGENTS.md, section 2):
- authors' method: w = sum_j X_j Y_j^T; s = 1 solves Y from the given X, s = 2 solves X from
  the given Y, and so on; all N functions jointly; N fixed (D-004, MATH_SPEC part B);
- project's discretization: ChF-1/ChF-2 by ChF-3 (``kkvim.coupled``), rho_N, kappa_rep and
  the QR with the companion transformation (``kkvim.coupled_rank``), the energy test model
  (``kkvim.coupled_energy``);
- engineering: this loop, the log, the guard arithmetic, the refinement step and the library calls.

Order of every half-step s (R4_NUMERICAL_SPEC 5.4.1): solve -> residual control -> field
diagnostics -> set diagnostics -> energy -> guards -> stop checks -> QR -> next half-step.
A stop is taken before the QR and keeps N, without regularization or restored directions
(5.5, variant A).

Modes:
- normative (default, ``research_mode=False``): unchanged (D-047; D-048.3 scope). c_d, c_rank,
  tau_set and delta_f are NOT_QUALIFIED, so the sequence does not continue through a QR; after
  the diagnostics of s = 1 it stops with NOT_QUALIFIED (HALF_STEP_LIMIT_REACHED if
  half_steps = 1). No D-048 stopping, residual control or configuration rule is applied.
- research (``research_mode=True``, D-047 addendum 1, D-048):
  * D-048.5: K_RESEARCH = 1e6, u = 2^-53; before the QR and before any stop status:
    K u kappa_rep < 1, rho_N > K u kappa_rep, sigma_min(R_V D_c) > K u; a failure is
    PROVISIONAL_RANK_GUARD_STOP or PROVISIONAL_SET_GUARD_STOP; S-2 is never assigned from K;
  * D-048.2: eta_N <= ETA_MAX = 1e-12; otherwise exactly one refinement step with the same mode
    systems (``kkvim.coupled.refine_once``), then ERROR_LINEAR_SOLVE if still above;
  * D-048.3: TOL_INF = 1e-10, TOL_E = 1e-8, S_MIN = 3, S_MAX = 100; stabilization when
    eps_inf <= TOL_INF and eps_E <= TOL_E hold at s - 1 and s with s >= S_MIN; the configuration
    rule TOL_E >= 10 eps_floor^(N), eps_floor^(N) = 10u (max(n_x, n_y)/pi)^2 sqrt(N)/(1 - gamma_N sqrt(N)),
    gamma_N = N u / (1 - N u) (first-order model, R4_NUMERICAL_SPEC 6.3), else ERROR_INPUT;
  * D-048.4: PROVISIONAL_FIELD_STABILIZED (never CONVERGED; not convergence to omega_0, not a rank
    certificate); S_MAX without stabilization is MAX_HALF_STEPS; an explicit half_steps < S_MAX
    reached without stabilization is RESEARCH_LIMIT_REACHED; no half-step beyond S_MAX;
  * D-048.1: the energy test of s = 2 is mandatory on the checked grids; an UNRESOLVED mandatory
    test does not stop the run and is reported by energy_s2_mandatory_unresolved = True with m_2
    and the reason; a resolved E1 violation stops with PROVISIONAL_ENERGY_E1_STOP.
  STAGNATED, the strict mode and an analogue of A-11 are not implemented (D-048.3).
- research with a fixed number of half-steps (``research_mode=True, stop_on_stabilization=False``; dated
  addendum to D-048 of 2026-10-18): the stabilization criterion is checked and recorded
  (``stabilization_ok``, ``first_stabilization_s``) but does not stop the run; the QR is done after all guards
  before every next half-step; ``half_steps < S_MAX`` ends with RESEARCH_LIMIT_REACHED, S_MAX with
  MAX_HALF_STEPS. Finiteness, the residual control, E1, the rank and set guards, S-1, ERROR_INPUT and S_MAX
  are not bypassed; PROVISIONAL_FIELD_STABILIZED is not used. ``stop_on_stabilization=False`` without the
  research mode is ERROR_INPUT.

S-1 (STOPPED_RANK_LOSS_PROVEN) is given only for a proven dependence: n_x < 2N or n_y < 2N
before the first half-step (ChV4-8, ChV4-9) or a set that is exactly dependent (zero or
duplicated column). Every log record and result carries PROVISIONAL / NOT_QUALIFIED.
"""

import math
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np

from kkvim.coefficients import ERROR_DEGENERATE, ERROR_INPUT
from kkvim.coupled import (
    N_SOLVE,
    TRANSFORMS,
    CoupledHalfStepError,
    coupled_residual,
    odd_sine_set,
    refine_once,
    solve_coupled_halfstep,
)
from kkvim.coupled_energy import (
    E1_VIOLATED,
    FACTOR_10,
    UNRESOLVED,
    CoupledEnergy,
    field_from_sets,
    first_energy,
    pair_energy,
)
from kkvim.coupled_rank import RankDiagnostics, mode_conditions, orthogonalize_received, rank_diagnostics
from kkvim.energy import EnergyError, discrete_energy
from kkvim.first_halfstep import HalfStepError, _positive_input
from kkvim.grid import Grid1D
from kkvim.linear_solver import ERROR_LINEAR_SOLVE, ERROR_NONFINITE
from kkvim.r3_input import validate_r3_sides

MARK = "PROVISIONAL / NOT_QUALIFIED"
K_RESEARCH = 1_000_000  # owner's temporary research guard (D-047 addendum 1, D-048.5); not a normative constant
UNIT_ROUNDOFF = 2.0**-53

# D-048.2, D-048.3 (research mode only)
ETA_MAX = 1e-12
TOL_INF = 1e-10
TOL_E = 1e-8
S_MIN = 3
S_MAX = 100

RESEARCH_LIMIT_REACHED = "RESEARCH_LIMIT_REACHED"
HALF_STEP_LIMIT_REACHED = "HALF_STEP_LIMIT_REACHED"  # normative mode, requested half-steps done without a QR
NOT_QUALIFIED = "NOT_QUALIFIED"
STOPPED_RANK_LOSS_PROVEN = "STOPPED_RANK_LOSS_PROVEN"  # S-1
STOPPED_RANK_NUMERICALLY_UNRESOLVED = "STOPPED_RANK_NUMERICALLY_UNRESOLVED"  # S-2: never assigned at this stage
PROVISIONAL_RANK_GUARD_STOP = "PROVISIONAL_RANK_GUARD_STOP"
PROVISIONAL_SET_GUARD_STOP = "PROVISIONAL_SET_GUARD_STOP"
PROVISIONAL_ENERGY_E1_STOP = "PROVISIONAL_ENERGY_E1_STOP"
PROVISIONAL_FIELD_STABILIZED = "PROVISIONAL_FIELD_STABILIZED"  # D-048.4
MAX_HALF_STEPS = "MAX_HALF_STEPS"  # D-048.3

# D-048.1: mandatory grids of the energy test at s = 2
MANDATORY_S2_FAMILY_I = (160, 640, 1280)  # n_x = n_y = n, the ten control r
MANDATORY_S2_FAMILY_II = ((1600, 320), (960, 320), (640, 320), (320, 640), (320, 960), (320, 1600))  # h_x = h_y
CONTROL_R = (Fraction(1, 5), Fraction(1, 3), Fraction(1, 2), Fraction(2, 3), Fraction(1), Fraction(7, 5),
             Fraction(3, 2), Fraction(2), Fraction(3), Fraction(5))


@dataclass(frozen=True)
class GuardResult:
    """Research guards before the QR (None if not evaluated)."""

    finite: bool
    applicable: bool | None  # K u kappa_rep < 1
    rank_ok: bool | None  # rho_N > K u kappa_rep
    set_ok: bool | None  # sigma_min(R_V D_c) > K u
    rank_threshold: float | None
    set_threshold: float | None

    @property
    def passed(self) -> bool:
        return bool(self.finite and self.applicable and self.rank_ok and self.set_ok)


@dataclass(frozen=True)
class StepLog:
    """Log of half-step s (every value PROVISIONAL / NOT_QUALIFIED)."""

    s: int
    k: int
    direction: str  # "Y" (solve Y on G(b, n_y)) or "X" (solve X on G(a, n_x))
    eta_N: float  # eta_N of the accepted solution (after a refinement if one was done)
    rank: RankDiagnostics
    cond_A_given: float
    cond_M_max: float
    energy: CoupledEnergy
    w_max: float | None  # max |w_s| (formed field before any QR)
    eps_inf: float | None  # max|w_s - w_{s-1}| / max|w_s|
    eps_E: float | None  # |w_s - w_{s-1}|_a / |w_s|_a
    guards: GuardResult | None
    qr_done: bool
    qr_field_error: float | None
    P_after_qr: float | None  # Pi_h of w_s from the transformed arrays (P_s^QR)
    stop_reason: str | None
    mark: str = MARK
    eta_initial: float | None = None  # eta_N before the refinement decision (research mode)
    refined: bool = False  # one refinement step done (D-048.2)
    m_s: float | None = None  # Q_s / (10 tau_s) for s >= 2
    stabilization_ok: bool | None = None  # eps_inf <= TOL_INF and eps_E <= TOL_E at this s (research mode, s >= 2)


@dataclass(frozen=True)
class CoupledRunParameters:
    a: float
    b: float
    D: float
    q: float
    n_x: int
    n_y: int
    N: int
    transform: str
    half_steps: int
    research_mode: bool
    K_research: int
    eps_floor: float | None = None  # eps_floor^(N) (D-048.3), research mode
    tol_inf: float | None = None
    tol_E: float | None = None
    s_min: int | None = None
    s_max: int | None = None
    eta_max: float | None = None
    stop_on_stabilization: bool | None = None  # research mode: True (D-048 stop) or False (fixed half-steps)


@dataclass(frozen=True)
class CoupledRun:
    """Status, parameters, step logs and the sets of the last completed half-step."""

    status: str
    parameters: CoupledRunParameters | None
    steps: tuple[StepLog, ...]
    grid_x: Grid1D | None
    grid_y: Grid1D | None
    x: np.ndarray | None  # X set of the last completed half-step (after a QR if one was done)
    y: np.ndarray | None
    reason: str = ""
    error: Exception | None = field(default=None, compare=False)
    mark: str = MARK
    energy_s2_checked: bool = False  # the energy test of s = 2 was evaluated
    energy_s2_mandatory: bool | None = None  # the grid is a mandatory s = 2 grid of D-048.1 (None: not reached)
    energy_s2_status: str | None = None
    energy_s2_m: float | None = None  # m_2 = Q_2 / (10 tau_2)
    energy_s2_mandatory_unresolved: bool | None = None  # True: mandatory and UNRESOLVED (None: not reached)
    energy_s2_reason: str | None = None
    first_stabilization_s: int | None = None  # first s with the two-step criterion of D-048.3 (research mode)

    @property
    def s(self) -> int:
        return len(self.steps)

    @property
    def converged(self) -> bool:
        return False  # never: PROVISIONAL_FIELD_STABILIZED is a stabilization of the finite problem, not convergence

    def field(self) -> np.ndarray | None:
        if self.x is None:
            return None
        return field_from_sets(self.grid_x, self.grid_y, self.x, self.y)


def eps_floor(n_x: int, n_y: int, N: int) -> float:
    """eps_floor^(N) = 10u (max(n_x, n_y)/pi)^2 sqrt(N) / (1 - gamma_N sqrt(N)) (D-048.3; first-order model)."""
    gamma_N = N * UNIT_ROUNDOFF / (1.0 - N * UNIT_ROUNDOFF)
    root = math.sqrt(N)
    return 10.0 * UNIT_ROUNDOFF * (max(n_x, n_y) / math.pi) ** 2 * root / (1.0 - gamma_N * root)


def is_mandatory_s2_grid(a: float, b: float, n_x: int, n_y: int) -> bool:
    """D-048.1: family I n in {160, 640, 1280} at the ten control r, or one of the six family II grids with h_x = h_y.

    r = b/a is compared with the exact rational value within the representation of the input floats (4u, as D-038).
    """
    fa, fb = Fraction(a), Fraction(b)
    tol = 4 * Fraction(UNIT_ROUNDOFF)

    def close(r: Fraction) -> bool:
        return abs(fb - r * fa) <= tol * max(fb, r * fa)

    if n_x == n_y and n_x in MANDATORY_S2_FAMILY_I:
        return any(close(r) for r in CONTROL_R)
    if (n_x, n_y) in MANDATORY_S2_FAMILY_II:
        return close(Fraction(n_y, n_x))  # h_x = h_y
    return False


def run_coupled(a: float, b: float, D: float, q: float, n_x: int, n_y: int, N: int, *, transform: str,
                half_steps: int, research_mode: bool = False, stop_on_stabilization: bool = True) -> CoupledRun:
    """Coupled half-steps s = 1, 2, ... at fixed N (research controller; D-047, D-048 and its addendum)."""
    if transform not in TRANSFORMS:
        raise ValueError(f"transform must be one of {TRANSFORMS}, got {transform!r}")
    if isinstance(half_steps, bool) or not isinstance(half_steps, (int, np.integer)) or half_steps < 1:
        raise ValueError(f"half_steps must be a positive integer, got {half_steps!r}")
    if not isinstance(research_mode, bool):
        raise TypeError("research_mode must be True or False")
    if not isinstance(stop_on_stabilization, bool):
        raise TypeError("stop_on_stabilization must be True or False")
    if not research_mode and not stop_on_stabilization:
        return CoupledRun(ERROR_INPUT, None, (), None, None, None, None,
                          reason="stop_on_stabilization=False requires research_mode=True (addendum to D-048)")
    try:
        a, b = validate_r3_sides(a, b)
        D = _positive_input("D", D)
        if isinstance(N, bool) or not isinstance(N, (int, np.integer)) or N not in N_SOLVE:
            raise CoupledHalfStepError(ERROR_INPUT, f"N must be one of {N_SOLVE} (D-047), got {N!r}")
        grid_x, grid_y = Grid1D(a, n_x), Grid1D(b, n_y)
    except (CoupledHalfStepError, HalfStepError, ValueError, TypeError) as exc:
        code = getattr(exc, "code", ERROR_INPUT)
        return CoupledRun(code, None, (), None, None, None, None, reason=str(exc), error=exc)
    floor = eps_floor(n_x, n_y, N) if research_mode else None
    params = CoupledRunParameters(
        a=a, b=b, D=D, q=float(q), n_x=int(n_x), n_y=int(n_y), N=int(N), transform=transform,
        half_steps=int(half_steps), research_mode=research_mode, K_research=K_RESEARCH, eps_floor=floor,
        tol_inf=TOL_INF if research_mode else None, tol_E=TOL_E if research_mode else None,
        s_min=S_MIN if research_mode else None, s_max=S_MAX if research_mode else None,
        eta_max=ETA_MAX if research_mode else None,
        stop_on_stabilization=stop_on_stabilization if research_mode else None)

    # S-1 before the first half-step: n_x, n_y >= 2N is necessary for the full rank (ChV4-8, ChV4-9)
    for axis, grid in (("x", grid_x), ("y", grid_y)):
        if grid.n < 2 * N:
            return CoupledRun(STOPPED_RANK_LOSS_PROVEN, params, (), grid_x, grid_y, None, None,
                              reason=f"n_{axis} = {grid.n} < 2N = {2 * N}: rank W^1 < N is proven (ChV4-8, ChV4-9)")
    # D-048.3 configuration rule (research mode)
    if research_mode and TOL_E < 10.0 * floor:
        return CoupledRun(ERROR_INPUT, params, (), grid_x, grid_y, None, None,
                          reason=f"tol_E = {TOL_E!r} < 10 eps_floor^(N) = {10.0 * floor!r} (D-048.3)")

    X = odd_sine_set(grid_x, N)
    Y = None  # no previous field before s = 1
    steps: list[StepLog] = []
    s2 = {}  # run-level record of the energy test at s = 2 (D-048.1)
    first_stabilization = []  # the first s with the two-step criterion (research mode)

    def stop(status, reason, error=None):
        return CoupledRun(status, params, tuple(steps), grid_x, grid_y, X if Y is not None else None, Y,
                          reason=reason, error=error, energy_s2_checked=bool(s2),
                          first_stabilization_s=first_stabilization[0] if first_stabilization else None, **s2)

    last = min(half_steps, S_MAX) if research_mode else half_steps
    previous_field = None
    previous_ok = False
    for s in range(1, last + 1):
        direction = "Y" if s % 2 else "X"
        given_grid, solve_grid = (grid_x, grid_y) if direction == "Y" else (grid_y, grid_x)
        given, companion = (X, Y) if direction == "Y" else (Y, X)
        try:
            step = solve_coupled_halfstep(given_grid, given, solve_grid, q, D, direction=direction, transform=transform)
        except CoupledHalfStepError as exc:
            status = STOPPED_RANK_LOSS_PROVEN if exc.code == ERROR_DEGENERATE else exc.code
            return stop(status, f"half-step s={s}: {exc}", exc)

        _, eta = coupled_residual(solve_grid, step.coefficients, D, step.solved)
        eta_initial, refined = (eta if research_mode else None), False
        if research_mode and not (eta <= ETA_MAX):  # D-048.2: exactly one refinement step
            try:
                step = refine_once(step)
            except CoupledHalfStepError as exc:
                return stop(exc.code, f"refinement of s={s}: {exc}", exc)
            _, eta = coupled_residual(solve_grid, step.coefficients, D, step.solved)
            refined = True
            if not (eta <= ETA_MAX):
                code = ERROR_LINEAR_SOLVE if math.isfinite(eta) else ERROR_NONFINITE
                return stop(code, f"s={s}: eta_N = {eta!r} > {ETA_MAX!r} after one refinement step "
                                  f"(before: {eta_initial!r})")
        received = step.solved
        new_X, new_Y = (X, received) if direction == "Y" else (received, Y)

        rank = rank_diagnostics(given_grid, given, solve_grid, received)
        cond_M, cond_A = mode_conditions(step.coefficients, solve_grid)
        try:
            if s == 1:
                energy = first_energy(grid_x, grid_y, new_X, new_Y, D, q)
            else:
                energy = pair_energy(grid_x, grid_y, X, Y, new_X, new_Y, D, q, solve_grid=solve_grid,
                                     coefficients=step.coefficients, received=received, companion_prev=companion)
        except EnergyError as exc:
            X, Y = new_X, new_Y
            steps.append(_log(s, direction, eta, rank, cond_A, cond_M, None, None, None, None, None, False, None, None,
                              exc.code, eta_initial, refined))
            return stop(exc.code, f"energy of s={s}: {exc}", exc)
        m_s = None if energy.tau is None else (energy.Q / (FACTOR_10 * energy.tau) if energy.tau > 0 else math.inf)
        if s == 2:
            mandatory = is_mandatory_s2_grid(a, b, n_x, n_y)
            unresolved = energy.status == UNRESOLVED
            s2 = dict(energy_s2_mandatory=mandatory, energy_s2_status=energy.status, energy_s2_m=m_s,
                      energy_s2_mandatory_unresolved=bool(mandatory and unresolved),
                      energy_s2_reason=(f"mandatory s = 2 test UNRESOLVED: Q_2 < 10 tau_2 (m_2 = {m_s!r}); not PASS"
                                        if mandatory and unresolved else None))
        w_now = field_from_sets(grid_x, grid_y, new_X, new_Y)
        eps_inf = eps_E = None
        if previous_field is not None:
            d = w_now - previous_field
            eps_inf = float(np.max(np.abs(d)) / np.max(np.abs(w_now)))
            eps_E = discrete_energy(grid_x, grid_y, d, D, q).norm_a / energy.norm_a
        stab_ok = None
        if research_mode and eps_inf is not None:
            stab_ok = bool(eps_inf <= TOL_INF and eps_E <= TOL_E)
        w_max = float(np.max(np.abs(w_now)))
        X, Y = new_X, new_Y
        previous_field = w_now

        def log(guards, reason, qr_done=False, qr_error=None, P_qr=None):
            steps.append(_log(s, direction, eta, rank, cond_A, cond_M, energy, w_max, eps_inf, eps_E, guards, qr_done,
                              qr_error, P_qr, reason, eta_initial, refined, m_s, stab_ok))

        if not (energy.P < 0.0):
            log(None, ERROR_DEGENERATE)
            return stop(ERROR_DEGENERATE, f"Pi_h(w_{s}) = {energy.P!r} >= 0")
        if energy.status == E1_VIOLATED:
            log(None, PROVISIONAL_ENERGY_E1_STOP)
            return stop(PROVISIONAL_ENERGY_E1_STOP, f"resolved energy test violates E1 at s={s} (research model)")

        guards = _guards(rank) if research_mode else None
        if not math.isfinite(eta) or not rank.finite:
            log(guards, ERROR_NONFINITE)
            return stop(ERROR_NONFINITE, f"non-finite diagnostics at s={s}")
        # research guards precede every stop status and the QR, also on the last half-step (QA C-1; 5.4.1; D-048.4)
        if research_mode and not (guards.applicable and guards.rank_ok):
            log(guards, PROVISIONAL_RANK_GUARD_STOP)
            why = "K u kappa_rep >= 1" if not guards.applicable else "rho_N <= K u kappa_rep"
            return stop(PROVISIONAL_RANK_GUARD_STOP, f"s={s}: {why}; no QR, N kept")
        if research_mode and not guards.set_ok:
            log(guards, PROVISIONAL_SET_GUARD_STOP)
            return stop(PROVISIONAL_SET_GUARD_STOP, f"s={s}: sigma_min(R_V D_c) <= K u; no QR, N kept")
        if research_mode:
            stabilized = bool(s >= S_MIN and stab_ok and previous_ok)
            if stabilized and not first_stabilization:
                first_stabilization.append(s)
            if stabilized and stop_on_stabilization:
                log(guards, PROVISIONAL_FIELD_STABILIZED)
                return stop(PROVISIONAL_FIELD_STABILIZED,
                            f"field stabilization of the finite problem at s={s} (D-048.3, D-048.4); not convergence "
                            "to omega_0 and not a rank certificate")
            previous_ok = bool(stab_ok)
            if s == S_MAX:
                log(guards, MAX_HALF_STEPS)
                if first_stabilization:  # fixed half-steps: stabilization recorded, run not stopped by it
                    why = f"s_max = {S_MAX} reached; stabilization first recorded at s={first_stabilization[0]} (fixed half-steps)"
                else:
                    why = f"s_max = {S_MAX} reached without field stabilization (D-048.3)"
                return stop(MAX_HALF_STEPS, why)
        if s == half_steps:
            status = RESEARCH_LIMIT_REACHED if research_mode else HALF_STEP_LIMIT_REACHED
            log(guards, status)
            return stop(status, f"{half_steps} half-steps done; this is not convergence")
        if not research_mode:
            log(None, NOT_QUALIFIED)
            return stop(NOT_QUALIFIED, "the QR criterion (c_d, c_rank, tau_set, delta_f) is NOT_QUALIFIED; "
                                       "continue only with research_mode=True")

        ortho = orthogonalize_received(solve_grid, received, given)
        if direction == "Y":
            X, Y = ortho.companion, ortho.received
        else:
            X, Y = ortho.received, ortho.companion
        P_qr = discrete_energy(grid_x, grid_y, field_from_sets(grid_x, grid_y, X, Y), D, q).pi
        log(guards, None, True, ortho.field_error, P_qr)
    raise AssertionError("unreachable")


def _guards(rank: RankDiagnostics) -> GuardResult:
    if not rank.finite:
        return GuardResult(False, None, None, None, None, None)
    base = K_RESEARCH * UNIT_ROUNDOFF
    rank_threshold = base * rank.kappa_rep
    return GuardResult(finite=True, applicable=rank_threshold < 1.0, rank_ok=rank.rho_N > rank_threshold,
                       set_ok=rank.set_sigma_min > base, rank_threshold=rank_threshold, set_threshold=base)


def _log(s, direction, eta, rank, cond_A, cond_M, energy, w_max, eps_inf, eps_E, guards, qr_done, qr_error, P_qr,
         reason, eta_initial=None, refined=False, m_s=None, stabilization_ok=None):
    return StepLog(s=s, k=s // 2, direction=direction, eta_N=eta, rank=rank, cond_A_given=cond_A, cond_M_max=cond_M,
                   energy=energy, w_max=w_max, eps_inf=eps_inf, eps_E=eps_E, guards=guards, qr_done=qr_done,
                   qr_field_error=qr_error, P_after_qr=P_qr, stop_reason=reason, eta_initial=eta_initial,
                   refined=refined, m_s=m_s, stabilization_ok=stabilization_ok)
