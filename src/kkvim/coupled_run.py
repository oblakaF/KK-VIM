"""Research controller of coupled KK-VIM half-steps at fixed N = 2-3 (R4, step 16; D-046, D-047). Internal.

Levels (AGENTS.md, section 2):
- authors' method: w = sum_j X_j Y_j^T; s = 1 solves Y from the given X, s = 2 solves X from
  the given Y, and so on; all N functions jointly; N fixed (D-004, MATH_SPEC part B);
- project's discretization: ChF-1/ChF-2 by ChF-3 (``kkvim.coupled``), rho_N, kappa_rep and
  the QR with the companion transformation (``kkvim.coupled_rank``), the energy test model
  (``kkvim.coupled_energy``);
- engineering: this loop, the log, the guard arithmetic and the library calls.

Order of every half-step s (R4_NUMERICAL_SPEC 5.4.1): solve -> field diagnostics -> set
diagnostics -> guards -> QR -> next half-step. A stop is taken before the QR and keeps N,
without regularization or restored directions (5.5, variant A).

Modes:
- normative (default, ``research_mode=False``): c_d, c_rank, tau_set and delta_f are
  NOT_QUALIFIED, so the sequence does not continue through a QR by an unconfirmed
  criterion; after the diagnostics of s = 1 it stops with NOT_QUALIFIED.
- research (``research_mode=True``, D-047 addendum 1): the owner's temporary guard
  K_RESEARCH = 1e6 with u = 2^-53; before the QR: K u kappa_rep < 1,
  rho_N > K u kappa_rep and sigma_min(R_V D_c) > K u. A failure is
  PROVISIONAL_RANK_GUARD_STOP or PROVISIONAL_SET_GUARD_STOP; S-2 is never assigned from K.
  The requested number of half-steps ends with RESEARCH_LIMIT_REACHED, never CONVERGED.

S-1 (STOPPED_RANK_LOSS_PROVEN) is given only for a proven dependence: n_x < 2N or n_y < 2N
before the first half-step (ChV4-8, ChV4-9) or a set that is exactly dependent (zero or
duplicated column). Every log record carries PROVISIONAL / NOT_QUALIFIED.
"""

import math
from dataclasses import dataclass, field

import numpy as np

from kkvim.coefficients import ERROR_DEGENERATE, ERROR_INPUT
from kkvim.coupled import (
    N_SOLVE,
    TRANSFORMS,
    CoupledHalfStepError,
    coupled_residual,
    odd_sine_set,
    solve_coupled_halfstep,
)
from kkvim.coupled_energy import E1_VIOLATED, CoupledEnergy, field_from_sets, first_energy, pair_energy
from kkvim.coupled_rank import RankDiagnostics, mode_conditions, orthogonalize_received, rank_diagnostics
from kkvim.energy import EnergyError, discrete_energy
from kkvim.first_halfstep import HalfStepError, _positive_input
from kkvim.grid import Grid1D
from kkvim.linear_solver import ERROR_NONFINITE
from kkvim.r3_input import validate_r3_sides

MARK = "PROVISIONAL / NOT_QUALIFIED"
K_RESEARCH = 1_000_000  # owner's temporary research guard (D-047 addendum 1); not a normative constant
UNIT_ROUNDOFF = 2.0**-53

RESEARCH_LIMIT_REACHED = "RESEARCH_LIMIT_REACHED"
HALF_STEP_LIMIT_REACHED = "HALF_STEP_LIMIT_REACHED"  # normative mode, requested half-steps done without a QR
NOT_QUALIFIED = "NOT_QUALIFIED"
STOPPED_RANK_LOSS_PROVEN = "STOPPED_RANK_LOSS_PROVEN"  # S-1
STOPPED_RANK_NUMERICALLY_UNRESOLVED = "STOPPED_RANK_NUMERICALLY_UNRESOLVED"  # S-2: never assigned at this stage
PROVISIONAL_RANK_GUARD_STOP = "PROVISIONAL_RANK_GUARD_STOP"
PROVISIONAL_SET_GUARD_STOP = "PROVISIONAL_SET_GUARD_STOP"
PROVISIONAL_ENERGY_E1_STOP = "PROVISIONAL_ENERGY_E1_STOP"


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
    eta_N: float
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

    @property
    def s(self) -> int:
        return len(self.steps)

    @property
    def converged(self) -> bool:
        return False  # this controller never declares convergence

    def field(self) -> np.ndarray | None:
        if self.x is None:
            return None
        return field_from_sets(self.grid_x, self.grid_y, self.x, self.y)


def run_coupled(a: float, b: float, D: float, q: float, n_x: int, n_y: int, N: int, *, transform: str,
                half_steps: int, research_mode: bool = False) -> CoupledRun:
    """Coupled half-steps s = 1 .. half_steps at fixed N (research controller, R4 step 16)."""
    if transform not in TRANSFORMS:
        raise ValueError(f"transform must be one of {TRANSFORMS}, got {transform!r}")
    if isinstance(half_steps, bool) or not isinstance(half_steps, (int, np.integer)) or half_steps < 1:
        raise ValueError(f"half_steps must be a positive integer, got {half_steps!r}")
    if not isinstance(research_mode, bool):
        raise TypeError("research_mode must be True or False")
    try:
        a, b = validate_r3_sides(a, b)
        D = _positive_input("D", D)
        if isinstance(N, bool) or not isinstance(N, (int, np.integer)) or N not in N_SOLVE:
            raise CoupledHalfStepError(ERROR_INPUT, f"N must be one of {N_SOLVE} (D-047), got {N!r}")
        grid_x, grid_y = Grid1D(a, n_x), Grid1D(b, n_y)
    except (CoupledHalfStepError, HalfStepError, ValueError, TypeError) as exc:
        code = getattr(exc, "code", ERROR_INPUT)
        return CoupledRun(code, None, (), None, None, None, None, reason=str(exc), error=exc)
    params = CoupledRunParameters(a=a, b=b, D=D, q=float(q), n_x=int(n_x), n_y=int(n_y), N=int(N),
                                  transform=transform, half_steps=int(half_steps), research_mode=research_mode,
                                  K_research=K_RESEARCH)

    # S-1 before the first half-step: n_x, n_y >= 2N is necessary for the full rank (ChV4-8, ChV4-9)
    for axis, grid in (("x", grid_x), ("y", grid_y)):
        if grid.n < 2 * N:
            return CoupledRun(STOPPED_RANK_LOSS_PROVEN, params, (), grid_x, grid_y, None, None,
                              reason=f"n_{axis} = {grid.n} < 2N = {2 * N}: rank W^1 < N is proven (ChV4-8, ChV4-9)")

    X = odd_sine_set(grid_x, N)
    Y = None  # no previous field before s = 1
    steps: list[StepLog] = []

    def stop(status, reason, error=None):
        return CoupledRun(status, params, tuple(steps), grid_x, grid_y, X if Y is not None else None, Y,
                          reason=reason, error=error)

    previous_field = None
    for s in range(1, half_steps + 1):
        direction = "Y" if s % 2 else "X"
        given_grid, solve_grid = (grid_x, grid_y) if direction == "Y" else (grid_y, grid_x)
        given, companion = (X, Y) if direction == "Y" else (Y, X)
        try:
            step = solve_coupled_halfstep(given_grid, given, solve_grid, q, D, direction=direction, transform=transform)
        except CoupledHalfStepError as exc:
            status = STOPPED_RANK_LOSS_PROVEN if exc.code == ERROR_DEGENERATE else exc.code
            return stop(status, f"half-step s={s}: {exc}", exc)
        received = step.solved
        new_X, new_Y = (X, received) if direction == "Y" else (received, Y)

        _, eta = coupled_residual(solve_grid, step.coefficients, D, received)
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
                              exc.code))
            return stop(exc.code, f"energy of s={s}: {exc}", exc)
        w_now = field_from_sets(grid_x, grid_y, new_X, new_Y)
        eps_inf = eps_E = None
        if previous_field is not None:
            d = w_now - previous_field
            eps_inf = float(np.max(np.abs(d)) / np.max(np.abs(w_now)))
            eps_E = discrete_energy(grid_x, grid_y, d, D, q).norm_a / energy.norm_a
        w_max = float(np.max(np.abs(w_now)))
        X, Y = new_X, new_Y
        previous_field = w_now

        if not (energy.P < 0.0):
            steps.append(_log(s, direction, eta, rank, cond_A, cond_M, energy, w_max, eps_inf, eps_E, None, False, None, None,
                              ERROR_DEGENERATE))
            return stop(ERROR_DEGENERATE, f"Pi_h(w_{s}) = {energy.P!r} >= 0")
        if energy.status == E1_VIOLATED:
            steps.append(_log(s, direction, eta, rank, cond_A, cond_M, energy, w_max, eps_inf, eps_E, None, False, None, None,
                              PROVISIONAL_ENERGY_E1_STOP))
            return stop(PROVISIONAL_ENERGY_E1_STOP, f"resolved energy test violates E1 at s={s} (research model)")

        guards = _guards(rank) if research_mode else None
        if not math.isfinite(eta) or not rank.finite:
            steps.append(_log(s, direction, eta, rank, cond_A, cond_M, energy, w_max, eps_inf, eps_E, guards, False, None, None,
                              ERROR_NONFINITE))
            return stop(ERROR_NONFINITE, f"non-finite diagnostics at s={s}")
        # research guards precede the limit check, also on the last half-step (QA C-1; R4_NUMERICAL_SPEC 5.4.1)
        if research_mode and not (guards.applicable and guards.rank_ok):
            steps.append(_log(s, direction, eta, rank, cond_A, cond_M, energy, w_max, eps_inf, eps_E, guards, False, None, None,
                              PROVISIONAL_RANK_GUARD_STOP))
            why = "K u kappa_rep >= 1" if not guards.applicable else "rho_N <= K u kappa_rep"
            return stop(PROVISIONAL_RANK_GUARD_STOP, f"s={s}: {why}; no QR, N kept")
        if research_mode and not guards.set_ok:
            steps.append(_log(s, direction, eta, rank, cond_A, cond_M, energy, w_max, eps_inf, eps_E, guards, False, None, None,
                              PROVISIONAL_SET_GUARD_STOP))
            return stop(PROVISIONAL_SET_GUARD_STOP, f"s={s}: sigma_min(R_V D_c) <= K u; no QR, N kept")
        if s == half_steps:
            status = RESEARCH_LIMIT_REACHED if research_mode else HALF_STEP_LIMIT_REACHED
            steps.append(_log(s, direction, eta, rank, cond_A, cond_M, energy, w_max, eps_inf, eps_E, guards, False, None, None,
                              status))
            return stop(status, f"{half_steps} half-steps done; this is not convergence")
        if not research_mode:
            steps.append(_log(s, direction, eta, rank, cond_A, cond_M, energy, w_max, eps_inf, eps_E, None, False, None, None,
                              NOT_QUALIFIED))
            return stop(NOT_QUALIFIED, "the QR criterion (c_d, c_rank, tau_set, delta_f) is NOT_QUALIFIED; "
                                       "continue only with research_mode=True")

        ortho = orthogonalize_received(solve_grid, received, given)
        if direction == "Y":
            X, Y = ortho.companion, ortho.received
        else:
            X, Y = ortho.received, ortho.companion
        P_qr = discrete_energy(grid_x, grid_y, field_from_sets(grid_x, grid_y, X, Y), D, q).pi
        steps.append(_log(s, direction, eta, rank, cond_A, cond_M, energy, w_max, eps_inf, eps_E, guards, True,
                          ortho.field_error, P_qr, None))
    raise AssertionError("unreachable")


def _guards(rank: RankDiagnostics) -> GuardResult:
    if not rank.finite:
        return GuardResult(False, None, None, None, None, None)
    base = K_RESEARCH * UNIT_ROUNDOFF
    rank_threshold = base * rank.kappa_rep
    return GuardResult(finite=True, applicable=rank_threshold < 1.0, rank_ok=rank.rho_N > rank_threshold,
                       set_ok=rank.set_sigma_min > base, rank_threshold=rank_threshold, set_threshold=base)


def _log(s, direction, eta, rank, cond_A, cond_M, energy, w_max, eps_inf, eps_E, guards, qr_done, qr_error, P_qr,
         reason):
    return StepLog(s=s, k=s // 2, direction=direction, eta_N=eta, rank=rank, cond_A_given=cond_A, cond_M_max=cond_M,
                   energy=energy, w_max=w_max, eps_inf=eps_inf, eps_E=eps_E, guards=guards, qr_done=qr_done, qr_field_error=qr_error,
                   P_after_qr=P_qr, stop_reason=reason)
