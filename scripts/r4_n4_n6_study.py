"""R4, N = 4-6: research study of the original coupled KK-VIM by the accepted modal scheme (Builder; G2, G3.3, G4).

Task "Builder - R4: исследование N = 4-6 по утверждённой схеме КК-МВИ" (ROADMAP R4; D-003; D-046, РВ4-7: N = 4-6 is a
research area). This is research, NOT a numerical acceptance of N = 4-6: no threshold is chosen from the results, every
computed value is PROVISIONAL / NOT_QUALIFIED, and the grids of this study are research grids, not normative ones
(D-046, D-048.9).

Levels (AGENTS.md, section 2, "Сохранение математической идентичности КК-МВИ"):
- A, authors' method (R4_SOURCE_NOTES, D-044; MATH_SPEC part B, (Ф-2)-(Ф-5), D-045; D-004): w = sum_j X_j Y_j^T, N fixed;
  the half-steps alternate Y -> X -> Y ...; in every half-step all N functions of the unknown set are found jointly from
  one coupled system, the other set being given.
- B, project's discretization (R4_NUMERICAL_SPEC 2.2-2.4, 3.2: ЧФ-1/ЧФ-2 and the modal splitting ЧФ-3, D-046): the grid
  sums A, B, C, F (``kkvim.coupled.coupled_coefficients``), the eigenvalues mu_k, the DST-I (``sine_transform``) and the
  mode systems D M_k v_k = F (psi_k^T 1) (``kkvim.coupled._solve_modes``); the QR of the received set with the companion
  transformation (5.4, ``kkvim.coupled_rank``); the energy identity and the test model (6.2, 6.3, ``kkvim.coupled_energy``).
- C, engineering: this orchestration, the logs and tables, the library calls inside the solver modules.

Isolation (task, item 2). The production solver is neither changed nor patched: ``kkvim.coupled.N_SOLVE`` stays (2, 3) and
``solve_coupled_halfstep`` / ``run_coupled`` keep rejecting N = 4-6 (D-047). ``research_halfstep`` performs the SAME
operations as ``solve_coupled_halfstep`` (coefficients -> mu, psi^T 1 -> _solve_modes -> inverse transform -> finiteness)
without its N_SOLVE guard, and ``run_cell`` repeats the order of ``run_coupled(research_mode=True,
stop_on_stabilization=False)`` (dated addendum to D-048, confirmed by D-049.9): solve -> residual control (eta_max = 1e-12,
one refinement step, D-048.2) -> field and set diagnostics -> energy -> guards K_RESEARCH = 1e6 (D-048.5) -> QR -> next
half-step. A stop is taken before the QR and keeps N; the stabilization criterion of D-048.3 is recorded and does not stop
the run (fixed number of half-steps). The constants, the guards and the configuration rule are imported from
``kkvim.coupled_run``; no new definition of the method, the half-steps, the rank or of a stopping rule is introduced.
D-048 covers N = 2-3 only: applying its research controller unchanged to N = 4-6 is the research choice of this study,
not an extension of D-048. For N = 2-3 the orchestration equals ``run_coupled`` bitwise; this is checked on the control
cells in every run and by tests/cross/test_r4_n4_n6_study.py.

PLAN - fixed in this file before the computations (task, item 3)
----------------------------------------------------------------
Inputs: a = 1, b = r, q = D = 1 (relative quantities do not depend on the scale); transform "fft" (D-048.6); odd sine start.
- N = 4, 5, 6 (study) on every grid below; N = 2, 3 (control) on the common subsample CONTROL_R x CONTROL_N.
- r: the ten control r of D-046 / D-048 (R_TEN).
- Family I (n_x = n_y = n): 40, 80, 160, 320, 640, 1280. Small and moderate grids first; the level 1280 > 640 (the R3 bound)
  is taken because the resolution estimate of R4_NUMERICAL_SPEC 4.4 asks for n_y >= 20 (2N - 1) r (N = 6, r = 5: 1100),
  not by carrying the R3 limit over. The D-048.3 configuration rule tol_E >= 10 eps_floor^(N) caps max(n_x, n_y) at 1905
  for N = 6 (2107, 2266, 2507 for N = 4, 3, 2): it is a computational limit of the imported controller and is recorded.
- Family II (h_x = h_y): the six grids of D-048.1 for r = 1/5, 1/3, 1/2, 2, 3, 5.
- s = 1 .. S_RUN = 40 half-steps in every cell (fixed number); tables at the fixed S_LIST = (1, 2, 3, 4, 10, 20, 40).
Sequence of comparisons (each one a table of tables.md; numbers in the order of execution):
  V1  by s at fixed N and grid: every cell, every half-step (steps.csv);
  V2  by N at fixed s and grid: N = 4, 5, 6 on every grid; N = 2, 3 added on the control subsample;
  V3  by grid at fixed N and s: family I chain and family II; observed order p of the discretization error on the 41 x 41
      points (i a/40, j b/40) common to all grids (an observation; no Richardson, no automatic UNRESOLVED - РВ4-7);
  V4  control N = 2, 3: bitwise equality with run_coupled; reproduction of the existing results of group B
      (docs/results/r4_group_b) and of the prequalification B5 (docs/results/r4_numerical_prequal);
  V5  implementation: solver field against the independent discrete control (kkvim_reference.r4_spectral_control,
      the same discrete problem minimised in the 2D sine representation; grids with max(n_x, n_y) <= CONTROL_MAX_N);
  V6  discretization: w_h^(N,s) - w_cont^(N,s), w_cont by the continuous matrix Parseval recursion
      (kkvim_reference.r4_continuous_sequence, M = 4001, doubling M = 8001 as diagnostic); s = 1 also against the
      m-truncated Navier sum with its strict tail (kkvim_reference.r4_levy_partial);
  V7  method and total error: w_cont - omega_0 and w_h - omega_0, omega_0 by Levy (kkvim_reference.levy); energy error
      e_rel of w_cont by the group-B function continuous_case (Pi(omega_0) of kkvim_reference.r4_energy_reference);
      discrete energy error e_rel_h of w_h against the full discrete solution omega_h (closed spectral form); Navier
      M = 2559 at the centre with its strict budget;
  V8  energy: the identity Delta P_s = Q_s - gamma_s (ЧВ4-14), E1/E2/UNRESOLVED of the test model tau_s^(N), monotone
      Pi_h by s, Pythagoras Pi_h(w) - Pi_h(omega_h) = |w - omega_h|_a^2 / 2, nested subspaces at s = 1 (Pi_h by N);
  V9  resolution: zeta^(N) = max over half-steps of lambda_max(A^-1 C)^(1/4) h of the given set (4.4) and its class
      Q/A/G (heuristic of D-039, NOT_QUALIFIED, not changed - D-048.9).
Statuses of a measurement (D-046): RESOLVED if the budget of its references (strict parts + rounding models) is <= 1/100 of
the measured quantity in the same relative norm, else UNRESOLVED; for the untruncated continuous method at s >= 2 at best
PROVISIONAL (Q-022; the group-B rule). The internal modal computation is never used as a reference for itself.

Reproduction (repository root):
    uv run python scripts/r4_n4_n6_study.py                  # full study (about an hour on the Builder machine)
    uv run python scripts/r4_n4_n6_study.py --tables-only    # tables.md from the CSV files
    uv run python scripts/r4_n4_n6_study.py --quick --out D  # small matrix for the tests (seconds)
Outputs (--out, default docs/results/r4_n4_n6): cells.csv, steps.csv, compare.csv, continuous.csv, orders.csv,
control_n23.csv, tables.md, meta.json.
"""

import argparse
import csv
import importlib.util
import json
import math
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import numpy as np
import scipy

import kkvim.coupled as kc
import kkvim.coupled_run as cr
from kkvim.coefficients import ERROR_DEGENERATE, ERROR_INPUT
from kkvim.coupled import (
    TRANSFORMS,
    CoupledHalfStep,
    CoupledHalfStepError,
    _solve_modes,
    coupled_coefficients,
    coupled_residual,
    odd_sine_set,
    refine_once,
    sine_eigenvalues,
    sine_transform,
)
from kkvim.coupled_energy import E1_VIOLATED, FACTOR_10, UNRESOLVED, field_from_sets, first_energy, pair_energy
from kkvim.coupled_rank import mode_conditions, orthogonalize_received, rank_diagnostics
from kkvim.energy import EnergyError, discrete_energy, rounding_threshold
from kkvim.first_halfstep import _positive_input
from kkvim.grid import Grid1D
from kkvim.linear_solver import ERROR_LINEAR_SOLVE, ERROR_NONFINITE
from kkvim.r3_input import validate_r3_sides
from kkvim_reference.levy import levy_field_bounds, levy_series_w_grid
from kkvim_reference.navier import NavierRectangle
from kkvim_reference.r4_continuous_sequence import continuous_sequence, field_at_nodes
from kkvim_reference.r4_levy_partial import navier_m_rounding_bound, navier_m_tail_bound, navier_m_truncated_at_nodes
from kkvim_reference.r4_spectral_control import SpectralGrid, run_control

ROOT = Path(__file__).resolve().parents[1]
OUT_DEFAULT = ROOT / "docs" / "results" / "r4_n4_n6"
U = 2.0**-53
MARK = cr.MARK  # PROVISIONAL / NOT_QUALIFIED

# ---- plan (fixed before the computations) ------------------------------------------------------------------------
N_STUDY = (4, 5, 6)
N_CONTROL = (2, 3)
N_RESEARCH = N_CONTROL + N_STUDY  # range of the research orchestration; the production range stays kc.N_SOLVE = (2, 3)
R_TEN = {"1/5": Fraction(1, 5), "1/3": Fraction(1, 3), "1/2": Fraction(1, 2), "2/3": Fraction(2, 3), "1": Fraction(1),
         "7/5": Fraction(7, 5), "3/2": Fraction(3, 2), "2": Fraction(2), "3": Fraction(3), "5": Fraction(5)}
FAMILY_I = (40, 80, 160, 320, 640, 1280)
FAMILY_II = {"1/5": (1600, 320), "1/3": (960, 320), "1/2": (640, 320), "2": (320, 640), "3": (320, 960), "5": (320, 1600)}
CONTROL_R = ("1/5", "1", "5")
CONTROL_N = (160, 640)
S_RUN = 40
S_LIST = (1, 2, 3, 4, 10, 20, 40)
TRANSFORM = "fft"  # D-048.6
Q_LOAD, D_RIGID = 1.0, 1.0
SAMPLE_DEN = 40  # common comparison points x_i = i a/40, y_j = j b/40 (nodes of every grid of the plan)
CONTROL_MAX_N = 640  # discrete control (V5) on grids with max(n_x, n_y) <= 640 and on family II
M_CONT, M_CHECK = 4001, 8001  # largest kept odd index of the continuous recursion and of its doubling diagnostic
M_NAVIER_M = 40001  # m-truncated Navier sum for s = 1 (odd; strict tail)
M_NAVIER = 2559  # NavierRectangle level of R3 (D-037)
NU_NAVIER = 0.3  # enters only the moments of NavierRectangle, not w
RULE = 0.01  # D-046: budget <= 1/100 of the measured quantity
ZETA_Q, ZETA_A = 0.039420, 0.157681  # zeta_A5^{R2}(80), zeta_A5^{R2}(20) (R3_NUMERICAL_SPEC 6.3; D-039), as in B8

QUICK = {"N": (4, 5, 6), "r": ("1",), "family_I": (24,), "family_II": {}, "control_r": ("1",), "control_n": (24,),
         "s_run": 4, "s_list": (1, 2, 3, 4), "M": 401, "M_check": 801, "M_navier_m": 4001, "control_max_n": 24}


def status(budget, measured):
    """RESOLVED if the budget of the references is <= 1/100 of the measured quantity (same relative norm), else UNRESOLVED."""
    if not (math.isfinite(budget) and math.isfinite(measured)) or measured == 0.0:
        return "UNRESOLVED"
    return "RESOLVED" if budget <= RULE * abs(measured) else "UNRESOLVED"


def continuous_status(s, meas_status, diagnostic, measured):
    """Statement on the untruncated continuous KK-VIM (group-B rule): s = 1 as measured (the budget then contains the
    strict tail of the s = 1 reference); s >= 2 at best PROVISIONAL (Q-022), when the doubling diagnostic is <= 1/100."""
    if s == 1:
        return meas_status
    if meas_status == "RESOLVED" and status(diagnostic, measured) == "RESOLVED":
        return "PROVISIONAL"
    return "UNRESOLVED"


def zeta_class(z, n_min):
    """Class of a level by zeta^(N) (heuristic of D-039 / R4_NUMERICAL_SPEC 4.4; NOT_QUALIFIED; not changed, D-048.9)."""
    if n_min < 20 or not math.isfinite(z):
        return "G"
    return "Q" if z <= ZETA_Q else ("A" if z <= ZETA_A else "G")


# ------------------------------------------------------------------------------------------------------------------
# Research orchestration (the working solver operations; N = 2-6)
# ------------------------------------------------------------------------------------------------------------------


def research_halfstep(fixed_grid, fixed, solve_grid, q, D, *, direction, transform):
    """ЧФ-1 (direction "Y") or ЧФ-2 ("X") by ЧФ-3: the operations of kkvim.coupled.solve_coupled_halfstep, in its order,
    for N in N_RESEARCH; the production function itself is not called because its N_SOLVE guard is D-047's range."""
    if direction not in ("X", "Y"):
        raise ValueError(f"direction must be 'X' or 'Y', got {direction!r}")
    if transform not in TRANSFORMS:
        raise ValueError(f"transform must be one of {TRANSFORMS}, got {transform!r}")
    D = _positive_input("D", D)
    coefficients = coupled_coefficients(fixed_grid, fixed, q)
    N, n = coefficients.N, solve_grid.n
    if N not in N_RESEARCH:
        raise CoupledHalfStepError(ERROR_INPUT, f"the research orchestration covers N in {N_RESEARCH}",
                                   direction=direction, N=N, n=n)
    mu = sine_eigenvalues(solve_grid)
    load = sine_transform(solve_grid, np.ones(solve_grid.interior_size), transform)  # psi_k^T 1
    modal = _solve_modes(coefficients, mu, load, D, direction=direction, n=n)
    solved = sine_transform(solve_grid, modal, transform)
    if not np.all(np.isfinite(solved)):
        raise CoupledHalfStepError(ERROR_NONFINITE, "non-finite solved set", direction=direction, N=N, n=n)
    return CoupledHalfStep(direction=direction, fixed_grid=fixed_grid, solve_grid=solve_grid,
                           fixed=np.array(fixed, dtype=float), solved=solved, modal=modal,
                           coefficients=coefficients, transform=transform, D=D, q=float(q))


def resolution(coefficients, grid):
    """kappa_s^(N) h = lambda_max(A^-1 C)^(1/4) h of a given set on its grid (R4_NUMERICAL_SPEC 4.4; as B8 of the QA)."""
    lam = np.linalg.eigvals(np.linalg.solve(coefficients.A, coefficients.C)).real
    return float(np.max(lam)) ** 0.25 * grid.h


@dataclass
class Cell:
    """One computed cell (N, r, grid): the logs of run_coupled's form, extra diagnostics and the kept fields."""

    N: int
    r_label: str
    family: str
    n_x: int
    n_y: int
    status: str = ""
    reason: str = ""
    steps: list = field(default_factory=list)  # kkvim.coupled_run.StepLog
    extra: list = field(default_factory=list)  # zeta, identity defect and the coefficient data per half-step
    fields: dict = field(default_factory=dict)  # s -> full field (n_x + 1, n_y + 1) for s in keep
    x: np.ndarray | None = None
    y: np.ndarray | None = None
    first_stabilization_s: int | None = None
    eps_floor: float = math.nan
    s2: dict = field(default_factory=dict)
    seconds: float = 0.0
    half_steps: int = S_RUN

    @property
    def r(self):
        return float(R_TEN[self.r_label]) if self.r_label in R_TEN else float(Fraction(self.r_label))

    @property
    def s_reached(self):
        return len(self.steps)


def run_cell(N, r_label, family, n_x, n_y, *, half_steps=S_RUN, transform=TRANSFORM, keep=S_LIST, r=None):
    """Coupled half-steps s = 1..half_steps at fixed N in the order of run_coupled(research_mode=True,
    stop_on_stabilization=False); every value PROVISIONAL / NOT_QUALIFIED."""
    t0 = time.perf_counter()
    cell = Cell(N=N, r_label=r_label, family=family, n_x=n_x, n_y=n_y, half_steps=half_steps)
    r = cell.r if r is None else r
    a, b = validate_r3_sides(1.0, r)
    q, D = Q_LOAD, D_RIGID
    grid_x, grid_y = Grid1D(a, n_x), Grid1D(b, n_y)
    floor = cr.eps_floor(n_x, n_y, N)
    cell.eps_floor = floor
    steps, extra = cell.steps, cell.extra
    state = {"X": None, "Y": None}

    def finish(status_, reason):
        cell.status, cell.reason, cell.seconds = status_, reason, time.perf_counter() - t0
        if state["Y"] is not None:
            cell.x, cell.y = state["X"], state["Y"]
        return cell

    for axis, grid in (("x", grid_x), ("y", grid_y)):  # S-1 before the first half-step (ChV4-8, ChV4-9)
        if grid.n < 2 * N:
            return finish(cr.STOPPED_RANK_LOSS_PROVEN, f"n_{axis} = {grid.n} < 2N = {2 * N}: rank W^1 < N is proven")
    if cr.TOL_E < 10.0 * floor:  # D-048.3 configuration rule of the research controller
        return finish(ERROR_INPUT, f"tol_E = {cr.TOL_E!r} < 10 eps_floor^(N) = {10.0 * floor!r} (D-048.3)")

    X = odd_sine_set(grid_x, N)
    Y = None
    previous_field, previous_ok, first = None, False, None
    for s in range(1, half_steps + 1):
        direction = "Y" if s % 2 else "X"
        given_grid, solve_grid = (grid_x, grid_y) if direction == "Y" else (grid_y, grid_x)
        given, companion = (X, Y) if direction == "Y" else (Y, X)
        try:
            step = research_halfstep(given_grid, given, solve_grid, q, D, direction=direction, transform=transform)
        except CoupledHalfStepError as exc:
            state.update(X=X, Y=Y)
            return finish(cr.STOPPED_RANK_LOSS_PROVEN if exc.code == ERROR_DEGENERATE else exc.code, f"half-step s={s}: {exc}")
        _, eta = coupled_residual(solve_grid, step.coefficients, D, step.solved)
        eta_initial, refined = eta, False
        if not (eta <= cr.ETA_MAX):  # D-048.2: exactly one refinement step
            try:
                step = refine_once(step)
            except CoupledHalfStepError as exc:
                state.update(X=X, Y=Y)
                return finish(exc.code, f"refinement of s={s}: {exc}")
            _, eta = coupled_residual(solve_grid, step.coefficients, D, step.solved)
            refined = True
            if not (eta <= cr.ETA_MAX):
                state.update(X=X, Y=Y)
                return finish(ERROR_LINEAR_SOLVE if math.isfinite(eta) else ERROR_NONFINITE,
                              f"s={s}: eta_N = {eta!r} > {cr.ETA_MAX!r} after one refinement step (before: {eta_initial!r})")
        received = step.solved
        new_X, new_Y = (X, received) if direction == "Y" else (received, Y)
        rank = rank_diagnostics(given_grid, given, solve_grid, received)
        cond_M, cond_A = mode_conditions(step.coefficients, solve_grid)
        zeta = resolution(step.coefficients, given_grid)
        try:
            if s == 1:
                energy = first_energy(grid_x, grid_y, new_X, new_Y, D, q)
            else:
                energy = pair_energy(grid_x, grid_y, X, Y, new_X, new_Y, D, q, solve_grid=solve_grid,
                                     coefficients=step.coefficients, received=received, companion_prev=companion)
        except EnergyError as exc:
            X, Y = new_X, new_Y
            steps.append(cr._log(s, direction, eta, rank, cond_A, cond_M, None, None, None, None, None, False, None, None,
                                 exc.code, eta_initial, refined))
            extra.append({"zeta": zeta})
            state.update(X=X, Y=Y)
            return finish(exc.code, f"energy of s={s}: {exc}")
        m_s = None if energy.tau is None else (energy.Q / (FACTOR_10 * energy.tau) if energy.tau > 0 else math.inf)
        if s == 2:
            mandatory = cr.is_mandatory_s2_grid(a, b, n_x, n_y) if N in kc.N_SOLVE else None  # D-048.1: N = 2-3 only
            cell.s2 = {"mandatory": mandatory, "status": energy.status, "m_2": m_s}
        w_now = field_from_sets(grid_x, grid_y, new_X, new_Y)
        eps_inf = eps_E = None
        if previous_field is not None:
            d = w_now - previous_field
            eps_inf = float(np.max(np.abs(d)) / np.max(np.abs(w_now)))
            eps_E = discrete_energy(grid_x, grid_y, d, D, q).norm_a / energy.norm_a
        stab_ok = None if eps_inf is None else bool(eps_inf <= cr.TOL_INF and eps_E <= cr.TOL_E)
        w_max = float(np.max(np.abs(w_now)))
        X, Y = new_X, new_Y
        previous_field = w_now
        if s in keep:
            cell.fields[s] = w_now
        defect = None if energy.delta_P is None else energy.delta_P - (energy.Q - energy.gamma)  # ЧВ4-14, item 3
        extra.append({"zeta": zeta, "identity_defect": defect, "centre": float(w_now[n_x // 2, n_y // 2]),
                      "rounding_scale": None if energy.R_prev is None else energy.R_prev + energy.R + energy.R_f})

        def log(guards, reason, qr_done=False, qr_error=None, P_qr=None):
            steps.append(cr._log(s, direction, eta, rank, cond_A, cond_M, energy, w_max, eps_inf, eps_E, guards, qr_done,
                                 qr_error, P_qr, reason, eta_initial, refined, m_s, stab_ok))

        state.update(X=X, Y=Y)
        if not (energy.P < 0.0):
            log(None, ERROR_DEGENERATE)
            return finish(ERROR_DEGENERATE, f"Pi_h(w_{s}) = {energy.P!r} >= 0")
        if energy.status == E1_VIOLATED:
            log(None, cr.PROVISIONAL_ENERGY_E1_STOP)
            return finish(cr.PROVISIONAL_ENERGY_E1_STOP, f"resolved energy test violates E1 at s={s} (research model)")
        guards = cr._guards(rank)
        if not math.isfinite(eta) or not rank.finite:
            log(guards, ERROR_NONFINITE)
            return finish(ERROR_NONFINITE, f"non-finite diagnostics at s={s}")
        if not (guards.applicable and guards.rank_ok):
            log(guards, cr.PROVISIONAL_RANK_GUARD_STOP)
            why = "K u kappa_rep >= 1" if not guards.applicable else "rho_N <= K u kappa_rep"
            return finish(cr.PROVISIONAL_RANK_GUARD_STOP, f"s={s}: {why}; no QR, N kept")
        if not guards.set_ok:
            log(guards, cr.PROVISIONAL_SET_GUARD_STOP)
            return finish(cr.PROVISIONAL_SET_GUARD_STOP, f"s={s}: sigma_min(R_V D_c) <= K u; no QR, N kept")
        stabilized = bool(s >= cr.S_MIN and stab_ok and previous_ok)
        if stabilized and first is None:
            first = s
            cell.first_stabilization_s = s
        previous_ok = bool(stab_ok)
        if s == cr.S_MAX:
            log(guards, cr.MAX_HALF_STEPS)
            return finish(cr.MAX_HALF_STEPS, f"s_max = {cr.S_MAX} reached")
        if s == half_steps:
            log(guards, cr.RESEARCH_LIMIT_REACHED)
            return finish(cr.RESEARCH_LIMIT_REACHED, f"{half_steps} half-steps done; this is not convergence")
        ortho = orthogonalize_received(solve_grid, received, given)
        if direction == "Y":
            X, Y = ortho.companion, ortho.received
        else:
            X, Y = ortho.received, ortho.companion
        state.update(X=X, Y=Y)
        P_qr = discrete_energy(grid_x, grid_y, field_from_sets(grid_x, grid_y, X, Y), D, q).pi
        log(guards, None, True, ortho.field_error, P_qr)
    raise AssertionError("unreachable")


def equals_run_coupled(cell):
    """N = 2-3: the cell equals run_coupled(research_mode=True, stop_on_stabilization=False) bitwise (sets and logs)."""
    run = cr.run_coupled(1.0, cell.r, Q_LOAD, D_RIGID, cell.n_x, cell.n_y, cell.N, transform=TRANSFORM,
                         half_steps=cell.half_steps,
                         research_mode=True, stop_on_stabilization=False)
    if run.status != cell.status or len(run.steps) != len(cell.steps):
        return False
    if not (np.array_equal(run.x, cell.x) and np.array_equal(run.y, cell.y)):
        return False
    for a_, b_ in zip(run.steps, cell.steps):
        for name in ("eta_N", "cond_A_given", "cond_M_max", "w_max", "eps_inf", "eps_E", "qr_field_error", "P_after_qr",
                     "stop_reason", "m_s", "stabilization_ok"):
            if getattr(a_, name) != getattr(b_, name):
                return False
        if not (np.array_equal(a_.rank.sigma, b_.rank.sigma) and a_.energy.P == b_.energy.P
                and a_.energy.tau == b_.energy.tau and a_.energy.status == b_.energy.status):
            return False
    return run.first_stabilization_s == cell.first_stabilization_s


# ------------------------------------------------------------------------------------------------------------------
# References (independent of the solver; D-015, D-028) and comparisons
# ------------------------------------------------------------------------------------------------------------------


def load_group_b():
    """The group-B module (QA, accepted for N = 2-3) for its continuous_case: e_rel and the centre error of w_cont with
    the budgets and statuses of group B, applied unchanged to N = 4-6."""
    spec = importlib.util.spec_from_file_location("r4_group_b_study", ROOT / "scripts" / "r4_group_b_study.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class References:
    """Caches of the reference quantities: per grid (Levy at the nodes, the discrete full solution omega_h), per (N, r)
    (continuous sequences M and 2M) and per (N, grid) (discrete control)."""

    def __init__(self, cfg):
        self.cfg = cfg
        self._grid = {}
        self._cont = {}

    def grid(self, r_label, n_x, n_y):
        key = (r_label, n_x, n_y)
        if key not in self._grid:
            self._grid.clear()  # one grid at a time (the cells are run grid by grid)
            r = float(R_TEN[r_label])
            g = SpectralGrid(1.0, r, n_x, n_y, Q_LOAD, D_RIGID)
            xs, ys = np.arange(n_x + 1) / n_x, np.arange(n_y + 1) * (r / n_y)
            W0 = levy_series_w_grid(xs, ys, a=1.0, b=r, q=Q_LOAD, D=D_RIGID)
            lb = levy_field_bounds(a=1.0, b=r, q=Q_LOAD, D=D_RIGID)
            self._grid[key] = {"g": g, "W0": W0, "levy_budget": lb.truncation + lb.rounding_model,
                               "omega_h": g.to_nodal(g.w_star), "pi_h_omega": g.omega_energy(), "control": {}}
        return self._grid[key]

    def continuous(self, N, r_label):
        key = (N, r_label)
        if key not in self._cont:
            r = float(R_TEN[r_label])
            s_run = self.cfg["s_run"]
            self._cont[key] = (continuous_sequence(1.0, r, N, s_run, self.cfg["M"], q=Q_LOAD, D=D_RIGID),
                               continuous_sequence(1.0, r, N, s_run, self.cfg["M_check"], q=Q_LOAD, D=D_RIGID))
        return self._cont[key]

    def control(self, N, r_label, n_x, n_y):
        ref = self.grid(r_label, n_x, n_y)
        if N not in ref["control"]:
            ref["control"][N] = run_control(ref["g"], N, self.cfg["s_run"])
        return ref["control"][N]


def compare_cell(cell, refs, cfg):
    """Rows of compare.csv for one cell: s in S_LIST that were completed (V5-V8 at fixed s)."""
    r, n_x, n_y, N = cell.r, cell.n_x, cell.n_y, cell.N
    ref = refs.grid(cell.r_label, n_x, n_y)
    W0, g = ref["W0"], ref["g"]
    W0i = W0[1:-1, 1:-1]
    W0max = float(np.max(np.abs(W0i)))
    ic, jc = n_x // 2, n_y // 2
    w0c = float(W0[ic, jc])
    levy_rel = ref["levy_budget"] / W0max
    levy_c_rel = ref["levy_budget"] / abs(w0c)
    full_grid = float(np.max(np.abs(ref["omega_h"] - W0i))) / W0max
    pi_h_omega = ref["pi_h_omega"]
    cont, cont2 = refs.continuous(N, cell.r_label)
    K = cont[0].x.shape[0]
    use_control = max(n_x, n_y) <= cfg["control_max_n"] or cell.family == "II"
    ctrl = refs.control(N, cell.r_label, n_x, n_y) if use_control else None
    modes = list(range(1, 2 * N, 2))
    grid_x, grid_y = Grid1D(1.0, n_x), Grid1D(r, n_y)
    nav = NavierRectangle(a=1.0, b=r, q=Q_LOAD, D=D_RIGID, nu=NU_NAVIER).at_points(0.5, r / 2.0, M=M_NAVIER,
                                                                                    diagnostics=False)
    nav_c = float(np.ravel(nav.w)[0])
    nav_budget = abs(nav.w_scale) * nav.w_error_bound_coef
    sub = (slice(0, None, n_x // SAMPLE_DEN), slice(0, None, n_y // SAMPLE_DEN)) if (
        n_x % SAMPLE_DEN == 0 and n_y % SAMPLE_DEN == 0) else None
    rows = []
    for s in cfg["s_list"]:
        if s not in cell.fields:
            continue
        log = cell.steps[s - 1]
        W = cell.fields[s]
        Wi = W[1:-1, 1:-1]
        Wc_full = field_at_nodes(cont[s - 1], n_x, n_y)
        Wc2_full = field_at_nodes(cont2[s - 1], n_x, n_y)
        Wc = Wc_full[1:-1, 1:-1]
        wc_c = float(Wc_full[ic, jc])
        e_disc = float(np.max(np.abs(Wi - Wc)))
        e_method = float(np.max(np.abs(Wc - W0i)))
        e_total = float(np.max(np.abs(Wi - W0i)))
        cont_round = K * U * float(np.max(np.abs(Wc)))  # model of the sine sums (group B)
        cont_diag = float(np.max(np.abs(Wc_full - Wc2_full)))  # doubling of M (diagnostic)
        tail_s1 = navier_m_tail_bound(1.0, r, Q_LOAD, D_RIGID, modes, cfg["M"]) if s == 1 else 0.0
        dc_disc = float(W[ic, jc]) - wc_c
        dc_method = wc_c - w0c
        dc_total = float(W[ic, jc]) - w0c
        cont_diag_c = abs(float(Wc_full[ic, jc]) - float(Wc2_full[ic, jc]))
        # discrete energy error against omega_h: energy route (R2 rounding model R(w), NUMERICAL_SPEC 10.3) and the
        # direct spectral route (non-negative sum) - their agreement is the exact identity Pi_h(w) - Pi_h(omega_h)
        # = |w - omega_h|_a^2 / 2 (V8)
        P = discrete_energy(grid_x, grid_y, W, D_RIGID, Q_LOAD).pi
        R_w = rounding_threshold(grid_x, grid_y, W, D_RIGID, Q_LOAD)
        e_h_E = (P - pi_h_omega) / (-pi_h_omega)
        d_hat = g.to_spectral(Wi) - g.w_star
        err2_D = g.D * g.hx * g.hy * math.fsum((g.lam * d_hat * d_hat).ravel().tolist())
        e_h_D = err2_D / (-2.0 * pi_h_omega)
        budget_E = R_w + 8.0 * U * abs(pi_h_omega)  # R(w) of the solver field + fsum and products of Pi_h(omega_h)
        row = {"N": N, "r_label": cell.r_label, "r": r, "family": cell.family, "nx": n_x, "ny": n_y, "s": s,
               "w_centre": float(W[ic, jc]), "w_max": log.w_max, "Pi_h": log.energy.P,
               "omega0_centre": w0c, "omega0_max": W0max, "levy_budget_rel": levy_rel,
               "E_disc_inf": e_disc / W0max, "E_method_inf": e_method / W0max, "E_total_inf": e_total / W0max,
               "E_full_grid_inf": full_grid,
               "d_disc_c": dc_disc / w0c, "d_method_c": dc_method / w0c, "d_total_c": dc_total / w0c,
               "cont_round_rel": cont_round / W0max, "cont_diag_rel": cont_diag / W0max, "cont_tail_s1_rel": tail_s1 / W0max,
               "e_rel_h": e_h_E, "e_rel_h_direct": e_h_D, "e_rel_h_budget_rel": budget_E / (P - pi_h_omega)
               if P > pi_h_omega else math.inf,
               "pythagoras_defect_over_budget": (2.0 * (P - pi_h_omega) - err2_D) / (2.0 * budget_E)}
        # statuses (D-046 rule; group-B logic for the continuous statement)
        meas_disc = status(cont_round + tail_s1, e_disc)
        row["status_meas_E_disc"] = meas_disc
        row["status_cont_E_disc"] = continuous_status(s, meas_disc, cont_diag, e_disc)
        meas_dc = status(cont_round + tail_s1, dc_disc)
        row["status_meas_d_disc_c"] = meas_dc
        row["status_cont_d_disc_c"] = continuous_status(s, meas_dc, cont_diag_c, dc_disc)
        meas_m = status(levy_rel * W0max + cont_round + tail_s1, e_method)
        row["status_meas_E_method"] = meas_m
        row["status_cont_E_method"] = continuous_status(s, meas_m, cont_diag, e_method)
        row["status_E_total"] = status(ref["levy_budget"], e_total)
        row["status_d_total_c"] = status(ref["levy_budget"], dc_total)
        row["status_e_rel_h"] = status(budget_E, P - pi_h_omega)
        # Navier M = 2559 at the centre (R3 reference): its strict budget against the measured total error
        row["navier_centre_minus_levy_rel"] = (nav_c - w0c) / w0c
        row["navier_budget_rel"] = nav_budget / abs(w0c)
        row["status_d_total_c_navier"] = status(nav_budget, float(W[ic, jc]) - nav_c)
        if s == 1:  # m-truncated Navier sum = exact continuous s = 1 (ВБ-23) with strict tail and rounding bound
            Wnm = navier_m_truncated_at_nodes(1.0, r, Q_LOAD, D_RIGID, modes, n_x, n_y, cfg["M_navier_m"])
            nm_budget = (navier_m_tail_bound(1.0, r, Q_LOAD, D_RIGID, modes, cfg["M_navier_m"])
                         + navier_m_rounding_bound(1.0, r, Q_LOAD, D_RIGID, modes, cfg["M_navier_m"]))
            e_nm = float(np.max(np.abs(W - Wnm)))
            row["E_disc_s1_navier_m_inf"] = e_nm / W0max
            row["navier_m_budget_rel"] = nm_budget / W0max
            row["status_E_disc_s1_navier_m"] = status(nm_budget, e_nm)
            row["cont_s1_minus_navier_m_rel"] = float(np.max(np.abs(Wc_full - Wnm))) / W0max
        if ctrl is not None:
            cs = ctrl[s - 1]
            cmax = float(np.max(np.abs(cs.W)))
            row["E_impl_inf"] = float(np.max(np.abs(Wi - cs.W))) / cmax
            row["E_impl_Pi"] = (log.energy.P - cs.energy_spectral) / abs(cs.energy_spectral)
            row["control_rho_N"] = cs.rho_N
            row["control_energy_identity_defect"] = cs.energy_identity_defect
        if sub is not None:
            row["_sample"] = (Wi.shape, W[sub], Wc_full[sub], W0[sub])  # for V3 (orders), not written
        rows.append(row)
    return rows


def orders(rows):
    """V3: observed order p of E_disc on the 41 x 41 common points for consecutive family-I levels (fixed N, r, s)."""
    out = []
    groups = {}
    for row in rows:
        if row["family"] == "I" and "_sample" in row:
            groups.setdefault((row["N"], row["r_label"], row["s"]), []).append(row)
    for (N, lab, s), grp in sorted(groups.items(), key=lambda t: (t[0][0], float(R_TEN[t[0][1]]), t[0][2])):
        grp = sorted(grp, key=lambda t: t["nx"])
        for k in range(len(grp) - 1):
            c, f = grp[k], grp[k + 1]
            if f["nx"] != 2 * c["nx"]:
                continue
            _, Wh1, Wc1, W01 = c["_sample"]
            _, Wh2, Wc2, _ = f["_sample"]
            W0max = float(np.max(np.abs(W01)))
            e1, e2 = float(np.max(np.abs(Wh1 - Wc1))), float(np.max(np.abs(Wh2 - Wc2)))
            d1, d2 = c["d_disc_c"], f["d_disc_c"]
            out.append({"N": N, "r_label": lab, "s": s, "n": c["nx"], "n2": f["nx"],
                        "E_disc_sample_n": e1 / W0max, "E_disc_sample_2n": e2 / W0max,
                        "p_sample": math.log2(e1 / e2) if e1 > 0 and e2 > 0 else math.nan,
                        "p_centre": math.log2(d1 / d2) if d1 * d2 > 0 else math.nan,
                        "status_cont_E_disc_n": c["status_cont_E_disc"], "status_cont_E_disc_2n": f["status_cont_E_disc"]})
    return out


# ------------------------------------------------------------------------------------------------------------------
# Rows of the logs
# ------------------------------------------------------------------------------------------------------------------


def cell_rows(cell):
    """cells.csv row and steps.csv rows of one cell."""
    zetas = [e["zeta"] for e in cell.extra]
    z_all = max(zetas) if zetas else math.nan
    z3 = max(zetas[:3]) if zetas else math.nan
    steps_rows = []
    for log, ex in zip(cell.steps, cell.extra):
        en, rk, gd = log.energy, log.rank, log.guards
        row = {"N": cell.N, "r_label": cell.r_label, "r": cell.r, "family": cell.family, "nx": cell.n_x, "ny": cell.n_y,
               "s": log.s, "k": log.k, "direction": log.direction, "w_centre": ex.get("centre", math.nan),
               "w_max": log.w_max, "Pi_h": None if en is None else en.P, "eta_initial": log.eta_initial,
               "eta_N": log.eta_N, "refined": log.refined, "rho_N": rk.rho_N, "kappa_rep": rk.kappa_rep,
               "cond_U": rk.cond_U, "cond_V": rk.cond_V, "set_sigma_min": rk.set_sigma_min,
               "cond_A_given": log.cond_A_given, "cond_M_max": log.cond_M_max, "zeta_s": ex["zeta"],
               "sigma": ";".join(f"{v:.6e}" for v in rk.sigma)}
        if en is not None:
            row.update({"delta_P": en.delta_P, "Q": en.Q, "gamma": en.gamma, "tau": en.tau, "m_s": log.m_s,
                        "energy_status": en.status, "identity_defect": ex.get("identity_defect"),
                        "identity_defect_over_rounding": None if ex.get("identity_defect") is None
                        else abs(ex["identity_defect"]) / ex["rounding_scale"],
                        "eps_inf": log.eps_inf, "eps_E": log.eps_E, "stabilization_ok": log.stabilization_ok})
        if gd is not None:
            row.update({"guard_applicable": gd.applicable, "guard_rank_ok": gd.rank_ok, "guard_set_ok": gd.set_ok,
                        "rho_over_threshold": rk.rho_N / gd.rank_threshold if gd.rank_threshold else None,
                        "set_over_threshold": rk.set_sigma_min / gd.set_threshold if gd.set_threshold else None})
        row.update({"qr_done": log.qr_done, "qr_field_error": log.qr_field_error, "P_after_qr": log.P_after_qr,
                    "stop_reason": log.stop_reason, "mark": log.mark})
        steps_rows.append(row)
    energies = [r["Pi_h"] for r in steps_rows if r["Pi_h"] is not None]
    taus = [r.get("tau") for r in steps_rows]
    increases = [i for i in range(1, len(energies)) if energies[i] > energies[i - 1]]
    tau_exceeded = [i for i in increases if energies[i] - energies[i - 1] > (taus[i] or 0.0)]
    defects = [r["identity_defect_over_rounding"] for r in steps_rows if r.get("identity_defect_over_rounding") is not None]
    cell_row = {"N": cell.N, "r_label": cell.r_label, "r": cell.r, "family": cell.family, "nx": cell.n_x, "ny": cell.n_y,
                "status": cell.status, "reason": cell.reason, "s_reached": cell.s_reached,
                "first_stabilization_s": cell.first_stabilization_s, "eps_floor": cell.eps_floor,
                "tolE_over_10floor": cr.TOL_E / (10.0 * cell.eps_floor),
                "zeta_N_s123": z3, "zeta_N_all": z_all, "class_s123": zeta_class(z3, min(cell.n_x, cell.n_y)),
                "class_all": zeta_class(z_all, min(cell.n_x, cell.n_y)),
                "zeta_estimate_s1_x": 2.0 * math.sin((2 * cell.N - 1) * math.pi / (2 * cell.n_x)),
                "min_rho_N": min((r["rho_N"] for r in steps_rows), default=math.nan),
                "min_rho_over_threshold": min((r["rho_over_threshold"] for r in steps_rows
                                               if r.get("rho_over_threshold") is not None), default=math.nan),
                "min_set_sigma": min((r["set_sigma_min"] for r in steps_rows), default=math.nan),
                "min_set_over_threshold": min((r["set_over_threshold"] for r in steps_rows
                                               if r.get("set_over_threshold") is not None), default=math.nan),
                "max_kappa_rep": max((r["kappa_rep"] for r in steps_rows), default=math.nan),
                "max_cond_M": max((r["cond_M_max"] for r in steps_rows), default=math.nan),
                "max_cond_A_given": max((r["cond_A_given"] for r in steps_rows), default=math.nan),
                "max_eta_N": max((r["eta_N"] for r in steps_rows), default=math.nan),
                "refinements": sum(bool(r["refined"]) for r in steps_rows),
                "max_qr_field_error": max((r["qr_field_error"] for r in steps_rows if r["qr_field_error"] is not None),
                                          default=math.nan),
                "energy_increases": len(increases), "energy_increases_beyond_tau": len(tau_exceeded),
                "max_identity_defect_over_rounding": max(defects, default=math.nan),
                "e1_e2_hold_steps": sum(r.get("energy_status") == "E1_E2_HOLD_PROVISIONAL" for r in steps_rows),
                "unresolved_steps": sum(r.get("energy_status") == UNRESOLVED for r in steps_rows),
                "s2_mandatory_D048": cell.s2.get("mandatory"), "s2_status": cell.s2.get("status"), "m_2": cell.s2.get("m_2"),
                "seconds": cell.seconds, "mark": MARK}
    return cell_row, steps_rows


# ------------------------------------------------------------------------------------------------------------------
# Continuous method error (group B function) and the N = 2-3 control against the existing results
# ------------------------------------------------------------------------------------------------------------------

CONT_KEYS = ("N", "s", "r", "M", "Pi_w", "Pi_0", "err_abs", "e_rel", "e_rel_energy_route", "D_strict_rel", "D_model_rel",
             "D_diag_rel", "delta_c", "c_strict_rel", "c_model_rel", "c_diag_rel", "E_inf_sampled", "L2_rel", "given_cond",
             "status_meas_e_rel", "status_cont_e_rel", "status_meas_delta_c", "status_cont_delta_c", "status_meas_E_inf",
             "status_cont_E_inf", "rel_energy_change_direct", "pythagoras_defect_rel", "increment_rel")


def continuous_rows(gb, N, r_label, cfg):
    r = float(R_TEN[r_label])
    out = []
    for row in gb.continuous_case(1.0, r, Q_LOAD, D_RIGID, N, M=cfg["M"], M_check=cfg["M_check"], s_max=cfg["s_run"]):
        rec = {"r_label": r_label}
        rec.update({k: row.get(k, math.nan) for k in CONT_KEYS})
        out.append(rec)
    return out


def read_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def rel(a, b):
    a, b = float(a), float(b)
    return abs(a - b) / abs(b) if b != 0 else (0.0 if a == 0 else math.inf)


def control_against_existing(cells_n23, compare_rows, cont_rows):
    """V4: N = 2-3 against docs/results/r4_numerical_prequal/b5_energy.csv (run_coupled diagnostics), the group-B grid
    table (QA discrete control) and the group-B continuous table; maximal relative differences per quantity and source."""
    out = []
    b5 = read_csv(ROOT / "docs" / "results" / "r4_numerical_prequal" / "b5_energy.csv")
    gb_grid = read_csv(ROOT / "docs" / "results" / "r4_group_b" / "grid.csv")
    gb_master = read_csv(ROOT / "docs" / "results" / "r4_group_b" / "master.csv")
    for cell, equal in cells_n23:
        rec = {"N": cell.N, "r_label": cell.r_label, "nx": cell.n_x, "ny": cell.n_y, "equal_run_coupled_bitwise": equal}
        # B5: rows of the same (r, n, N) and s
        diffs = {}
        for row in b5:
            if (row["family"], row["r"], int(row["nx"]), int(row["ny"]), int(row["N"])) != ("I", cell.r_label, cell.n_x,
                                                                                        cell.n_y, cell.N):
                continue
            s = int(row["s"])
            if s > cell.s_reached:
                continue
            log = cell.steps[s - 1]
            pairs = {"rho_N": log.rank.rho_N, "kappa_rep": log.rank.kappa_rep, "set_sigma_min": log.rank.set_sigma_min}
            if row["Q"]:
                pairs.update({"Q": log.energy.Q, "tau": log.energy.tau, "m": log.m_s})
            for k, v in pairs.items():
                diffs[k] = max(diffs.get(k, 0.0), rel(v, row[k]))
            rec["b5_s_compared"] = rec.get("b5_s_compared", 0) + 1
        for k, v in diffs.items():
            rec[f"b5_max_rel_{k}"] = v
        # group B grid table (QA discrete control): the same N, r, grid and s
        gdiff = {}
        mine = {(c["s"]): c for c in compare_rows if (c["N"], c["r_label"], c["nx"], c["ny"]) == (cell.N, cell.r_label,
                                                                                                  cell.n_x, cell.n_y)}
        for row in gb_grid:
            if (int(row["N"]), row["r_label"], int(row["nx"]), int(row["ny"]), row["family"]) != (
                    cell.N, cell.r_label, cell.n_x, cell.n_y, "I"):
                continue
            c = mine.get(int(row["s"]))
            if c is None:
                continue
            for k_mine, k_gb in (("d_total_c", "d_total_c"), ("d_disc_c", "d_disc_c"), ("d_method_c", "d_method_c"),
                                 ("e_rel_h_direct", "e_rel_h"), ("E_disc_inf", "E_inf_disc"), ("E_method_inf", "E_inf_method"),
                                 ("E_total_inf", "E_inf_total"), ("Pi_h", "Pi_h_w")):
                gdiff[k_mine] = max(gdiff.get(k_mine, 0.0), rel(c[k_mine], row[k_gb]))
            rec["group_b_grid_s_compared"] = rec.get("group_b_grid_s_compared", 0) + 1
        for k, v in gdiff.items():
            rec[f"group_b_grid_max_rel_{k}"] = v
        out.append(rec)
    # group B continuous table: e_rel and delta_c of w_cont (the same function, here for the control (N, r))
    cdiff = {}
    for row in gb_master:
        if row["family"] != "I" or int(row["n_x"]) != 40:
            continue
        lab, N, s = row["r"], int(row["N"]), int(row["s"])
        for c in cont_rows:
            if (c["N"], c["r_label"], c["s"]) == (N, lab, s):
                cdiff["e_rel"] = max(cdiff.get("e_rel", 0.0), rel(c["e_rel"], row["e_rel"]))
                cdiff["delta_c"] = max(cdiff.get("delta_c", 0.0), rel(c["delta_c"], row["delta_c_method"]))
                cdiff["count"] = cdiff.get("count", 0) + 1
    out.append({"N": "2,3", "r_label": "control", "nx": "", "ny": "", "continuous_compared": cdiff.get("count", 0),
                "group_b_master_max_rel_e_rel": cdiff.get("e_rel", math.nan),
                "group_b_master_max_rel_delta_c": cdiff.get("delta_c", math.nan)})
    return out


# ------------------------------------------------------------------------------------------------------------------
# Output
# ------------------------------------------------------------------------------------------------------------------


# values whose changes by s or N lie far below 1e-6 relative: 16 significant digits; the rest: 7
HIGH_PRECISION = {"w_centre", "w_max", "Pi_h", "delta_P", "Q", "gamma", "tau", "P_after_qr", "identity_defect",
                  "omega0_centre", "omega0_max", "Pi_w", "Pi_0"}


def write_csv(path, rows):
    keys = []
    for row in rows:
        keys += [k for k in row if k not in keys and not k.startswith("_")]

    def text(k, v):
        if isinstance(v, float):
            return f"{v:.15e}" if k in HIGH_PRECISION else f"{v:.6e}"
        return "" if v is None else v

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
        w.writeheader()
        for row in rows:
            w.writerow({k: text(k, v) for k, v in row.items() if not k.startswith("_")})


def fnum(v, spec=".3e"):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return str(v) if v not in (None, "") else "—"
    if math.isnan(x):
        return "—"
    return format(x, spec)


def md_table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines) + "\n"


def r_key(lab):
    return float(Fraction(lab))


def make_tables(out):
    """tables.md from the CSV files (deterministic text; LF line ends)."""
    cells = read_csv(out / "cells.csv")
    steps = read_csv(out / "steps.csv")
    comp = read_csv(out / "compare.csv")
    cont = read_csv(out / "continuous.csv")
    ords = read_csv(out / "orders.csv")
    ctl = read_csv(out / "control_n23.csv")
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    s_list = tuple(meta["plan"]["S_LIST"])
    parts = ["# R4, N = 4–6: исследовательские таблицы (PROVISIONAL / NOT_QUALIFIED)\n",
             "Сгенерировано `scripts/r4_n4_n6_study.py --tables-only` из CSV этого каталога. Это исследование, не приёмка "
             "N = 4–6; сетки исследования не нормативны. Отчёт — `docs/R4_N4_N6_STUDY.report.md`.\n"]

    # T1: cells overview
    rows = []
    for c in sorted(cells, key=lambda t: (int(t["N"]), t["family"], r_key(t["r_label"]), int(t["nx"]), int(t["ny"]))):
        rows.append([c["N"], c["r_label"], c["family"], f"{c['nx']}×{c['ny']}", c["status"], c["s_reached"],
                     c["first_stabilization_s"] or "—", fnum(c["zeta_N_all"], ".3f"), c["class_all"],
                     fnum(c["min_rho_N"], ".2e"), fnum(c["min_rho_over_threshold"], ".2e"),
                     fnum(c["min_set_over_threshold"], ".2e"), fnum(c["max_eta_N"], ".1e"), fnum(c["max_cond_M"], ".2e"),
                     c["energy_increases_beyond_tau"], fnum(c["max_identity_defect_over_rounding"], ".2e")])
    parts.append("## T1. Ячейки: статус расчёта, разрешение ζ^(N), ранг, обусловленность, невязка\n\n"
                 "ρ/порог = min_s ρ_N/(K·u·κ_rep), набор/порог = min_s σ_min(R_V·D_c)/(K·u), K = 10⁶ (D-048.5). ζ^(N) — максимум "
                 "по всем выполненным полушагам; класс — эвристика D-039 (NOT_QUALIFIED). «рост P > τ» — число полушагов, где "
                 "Π_h выросла больше τ_s. «тожд.» — max |ΔP − (Q − γ)| / (R_{s−1} + R_s + R_f).\n\n")
    parts.append(md_table(["N", "r", "сем.", "сетка", "статус", "s", "стаб. s", "ζ^(N)", "класс", "min ρ_N", "ρ/порог",
                           "набор/порог", "max η_N", "max cond M_k", "рост P > τ", "тожд."], rows))

    def comp_get(N, lab, fam, nx, ny, s):
        for c in comp:
            if (int(c["N"]), c["r_label"], c["family"], int(c["nx"]), int(c["ny"]), int(c["s"])) == (N, lab, fam, nx, ny, s):
                return c
        return None

    def cont_get(N, lab, s):
        for c in cont:
            if (int(c["N"]), c["r_label"], int(c["s"])) == (N, lab, s):
                return c
        return None

    labs = sorted({c["r_label"] for c in cells}, key=r_key)
    ns = sorted({int(c["N"]) for c in cells})
    fam_i = sorted({int(c["nx"]) for c in cells if c["family"] == "I"})
    top = max(fam_i)
    mid = 640 if 640 in fam_i else top

    # T2: by s at a fixed grid
    rows = []
    for lab in labs:
        for N in ns:
            for s in s_list:
                c = comp_get(N, lab, "I", mid, mid, s)
                if c is None:
                    continue
                st = next((t for t in steps if (int(t["N"]), t["r_label"], t["family"], int(t["nx"]), int(t["s"])) ==
                           (N, lab, "I", mid, s)), None)
                rows.append([N, lab, s, fnum(c["w_centre"], ".10e"), fnum(c["Pi_h"], ".12e"), fnum(c["e_rel_h"]),
                             c["status_e_rel_h"], fnum(st.get("eps_E") if st else None, ".2e"),
                             fnum(st["rho_N"] if st else None, ".2e"), (st or {}).get("energy_status") or "—",
                             fnum((st or {}).get("m_s"), ".2e")])
    parts.append(f"## T2 (V1). Изменение по s при фиксированных N и сетке (семейство I, n = {mid})\n\n"
                 "e_rel,h = (Π_h(w) − Π_h(ω_h))/(−Π_h(ω_h)) — энергетическая ошибка относительно полного дискретного решения той же "
                 "сетки; ε^E — изменение поля в энергетической норме; m_s = Q_s/(10τ_s).\n\n")
    parts.append(md_table(["N", "r", "s", "w(центр)", "Π_h", "e_rel,h", "статус", "ε^E", "ρ_N", "энерг. тест", "m_s"], rows))

    # T3: by N at fixed s and grid
    rows = []
    for lab in labs:
        for s in (1, 2, 10, max(s_list)):
            if s not in s_list:
                continue
            for N in ns:
                c = comp_get(N, lab, "I", mid, mid, s)
                cc = cont_get(N, lab, s)
                if c is None and cc is None:
                    continue
                rows.append([lab, s, N, fnum(c["e_rel_h"]) if c else "—", fnum(cc["e_rel"]) if cc else "—",
                             cc["status_cont_e_rel"] if cc else "—", fnum(c["d_total_c"]) if c else "—",
                             c["status_d_total_c"] if c else "—", fnum(c["E_total_inf"]) if c else "—",
                             fnum(cc["delta_c"]) if cc else "—"])
    parts.append(f"## T3 (V2). Изменение по N при фиксированных s и сетке (семейство I, n = {mid}); непрерывный метод — без сетки\n\n"
                 "e_rel(cont) — энергетическая ошибка непрерывной последовательности w_cont (функция `continuous_case` группы B, "
                 "M = 4001; статус непрерывного утверждения: s = 1 RESOLVED/UNRESOLVED, s ≥ 2 не выше PROVISIONAL, Q-022). "
                 "δ_c — относительная ошибка в центре со знаком (дискретная — против Леви в узле, непрерывная — против Леви (140)).\n\n")
    parts.append(md_table(["r", "s", "N", "e_rel,h", "e_rel(cont)", "статус cont", "δ_c (w_h)", "статус", "E_∞ (w_h)",
                           "δ_c (w_cont)"], rows))

    # T4: by grid at fixed N and s
    rows = []
    for lab in labs:
        for N in ns:
            for s in (1, 2, max(s_list)):
                if s not in s_list:
                    continue
                for n in fam_i:
                    c = comp_get(N, lab, "I", n, n, s)
                    if c is None:
                        continue
                    o = next((t for t in ords if (int(t["N"]), t["r_label"], int(t["s"]), int(t["n"])) == (N, lab, s, n)),
                             None)
                    cl = next((t for t in cells if (int(t["N"]), t["r_label"], t["family"], int(t["nx"])) == (N, lab, "I", n)),
                              None)
                    rows.append([lab, N, s, n, cl["class_all"] if cl else "—", fnum(c["E_disc_inf"]), fnum(c["d_disc_c"]),
                                 c["status_cont_E_disc"], fnum(o["p_sample"], ".3f") if o else "—",
                                 fnum(o["p_centre"], ".3f") if o else "—", fnum(c.get("E_impl_inf"), ".1e")])
    parts.append("## T4 (V3). Изменение по сетке при фиксированных N и s (семейство I)\n\n"
                 "E_disc = w_h − w_cont (дискретизация) по всем внутренним узлам, относительно max|ω₀|; p — наблюдаемый порядок на "
                 "паре (n, 2n) по общим точкам 41 × 41 и в центре (наблюдение, не критерий; Ричардсон не применяется). "
                 "E_impl — отличие от независимого дискретного контроля той же задачи (реализация).\n\n")
    parts.append(md_table(["r", "N", "s", "n", "класс ζ", "E_disc,∞", "δ_disc,c", "статус cont", "p (41×41)", "p (центр)",
                           "E_impl"], rows))

    # T5: family II
    rows = []
    for c in comp:
        if c["family"] != "II" or int(c["s"]) not in (1, 2, max(s_list)):
            continue
        rows.append([c["N"], c["r_label"], f"{c['nx']}×{c['ny']}", c["s"], fnum(c["E_disc_inf"]), fnum(c["d_total_c"]),
                     fnum(c["e_rel_h"]), fnum(c.get("E_impl_inf"), ".1e")])
    parts.append("## T5 (V3). Семейство II (h_x = h_y, шесть сеток D-048.1)\n\n")
    parts.append(md_table(["N", "r", "сетка", "s", "E_disc,∞", "δ_c (w_h)", "e_rel,h", "E_impl"], rows))

    # T6: decomposition at the finest family-I level
    rows = []
    for lab in labs:
        for N in ns:
            for s in (2, max(s_list)):
                c = comp_get(N, lab, "I", top, top, s) or comp_get(N, lab, "I", mid, mid, s)
                if c is None:
                    continue
                rows.append([lab, N, f"{c['nx']}", s, fnum(c.get("E_impl_inf"), ".1e"), fnum(c["E_disc_inf"]),
                             c["status_cont_E_disc"], fnum(c["E_method_inf"]), c["status_cont_E_method"],
                             fnum(c["E_total_inf"]), c["status_E_total"], fnum(c["E_full_grid_inf"]),
                             fnum(c["levy_budget_rel"], ".1e"), fnum(c["cont_diag_rel"], ".1e")])
    parts.append("## T6 (V5–V7). Разделение ошибок реализации, дискретизации и метода (наибольшая сетка семейства I)\n\n"
                 "Все величины — max по узлам, относительно max|ω₀|. E_full = ω_h − ω₀ — ошибка полной двумерной дискретной задачи "
                 "(ориентир, не оценка E_disc). Бюджеты: Леви (строгий хвост + модель округления); w_cont — диагностика удвоения M.\n\n")
    parts.append(md_table(["r", "N", "n", "s", "E_impl", "E_disc", "статус", "E_method", "статус", "E_total", "статус",
                           "E_full", "бюджет Леви", "диагн. w_cont"], rows))

    # T7: s = 1 against the m-truncated Navier sum
    rows = []
    for c in comp:
        if int(c["s"]) != 1 or c["family"] != "I":
            continue
        rows.append([c["N"], c["r_label"], c["nx"], fnum(c.get("E_disc_s1_navier_m_inf")), fnum(c.get("navier_m_budget_rel"), ".1e"),
                     c.get("status_E_disc_s1_navier_m", "—"), fnum(c.get("cont_s1_minus_navier_m_rel"), ".1e")])
    parts.append("## T7 (V6). s = 1 против m-усечённой суммы Навье (строгий хвост, граница округления)\n\n")
    parts.append(md_table(["N", "r", "n", "E_disc(s=1),∞", "бюджет Навье-m", "статус", "w_cont − Навье-m"], rows))

    # T8: energy behaviour (V8)
    rows = []
    for N in ns:
        sub = [c for c in cells if int(c["N"]) == N]
        if not sub:
            continue
        rows.append([N, len(sub), sum(int(c["energy_increases"]) for c in sub),
                     sum(int(c["energy_increases_beyond_tau"]) for c in sub),
                     fnum(max(float(c["max_identity_defect_over_rounding"]) for c in sub if c["max_identity_defect_over_rounding"]), ".2e"),
                     sum(int(c["e1_e2_hold_steps"]) for c in sub), sum(int(c["unresolved_steps"]) for c in sub),
                     fnum(max(abs(float(c["pythagoras_defect_over_budget"])) for c in comp if int(c["N"]) == N), ".2e")])
    parts.append("## T8 (V8). Энергетическое тождество и поведение по s\n\n"
                 "Рост Π_h — число полушагов с Π_h(s) > Π_h(s − 1) (на уровне округления), из них больше τ_s; тождество ЧВ4-14 — "
                 "max |ΔP − (Q − γ)| в единицах R_{s−1} + R_s + R_f; Пифагор — max |2(Π_h(w) − Π_h(ω_h)) − ‖w − ω_h‖²_a| в единицах "
                 "2(R(w) + 8u|Π_h(ω_h)|).\n\n")
    parts.append(md_table(["N", "ячеек", "рост Π_h", "рост > τ", "тожд./округл.", "полушагов E1_E2_HOLD", "UNRESOLVED",
                           "Пифагор/бюджет"], rows))
    # nested subspaces at s = 1: Pi_h non-increasing in N on the same grid
    nested = []
    for lab in labs:
        for fam_grid in sorted({(c["family"], int(c["nx"]), int(c["ny"])) for c in cells if c["r_label"] == lab}):
            vals = []
            for N in ns:
                c = comp_get(N, lab, fam_grid[0], fam_grid[1], fam_grid[2], 1)
                if c is not None:
                    vals.append((N, float(c["Pi_h"])))
            if len(vals) >= 2:
                viol = sum(1 for (n1, p1), (n2, p2) in zip(vals, vals[1:]) if p2 > p1)
                nested.append([lab, f"{fam_grid[0]} {fam_grid[1]}×{fam_grid[2]}", ",".join(str(v[0]) for v in vals), viol])
    parts.append("\nВложенность подпространств при s = 1 (нечётные синусы: span X⁰_N ⊂ span X⁰_{N+1}) ⇒ Π_h(w¹) не возрастает по N:\n\n")
    parts.append(md_table(["r", "сетка", "N", "нарушений"], nested))

    # T9: failures
    rows = [[c["N"], c["r_label"], c["family"], f"{c['nx']}×{c['ny']}", c["status"], c["s_reached"], c["reason"]]
            for c in cells if c["status"] != cr.RESEARCH_LIMIT_REACHED]
    parts.append("## T9. Ячейки без достижения s = S_RUN и причины\n\n")
    parts.append(md_table(["N", "r", "сем.", "сетка", "статус", "s", "причина"], rows) if rows else "Нет.\n")

    # T10: control N = 2, 3
    rows = []
    for c in ctl:
        rows.append([c["N"], c["r_label"], c.get("nx", ""), c.get("equal_run_coupled_bitwise", ""),
                     fnum(c.get("b5_max_rel_rho_N"), ".1e"), fnum(c.get("b5_max_rel_m"), ".1e"),
                     fnum(c.get("group_b_grid_max_rel_d_total_c"), ".1e"), fnum(c.get("group_b_grid_max_rel_e_rel_h_direct"), ".1e"),
                     fnum(c.get("group_b_grid_max_rel_Pi_h"), ".1e"), fnum(c.get("group_b_master_max_rel_e_rel"), ".1e"),
                     fnum(c.get("group_b_master_max_rel_delta_c"), ".1e")])
    parts.append("## T10 (V4). Контроль N = 2–3 против существующих результатов\n\n"
                 "Совпадение с `run_coupled` — побитовое (наборы и журналы). Остальное — max относительное отличие от сохранённых "
                 "значений (печать 7 значащих цифр, т. е. ~10⁻⁷ — уровень печати): B5 предквалификации (ρ_N, m_s), таблица `grid.csv` "
                 "группы B (дискретный контроль QA: δ_total,c, e_rel,h, Π_h), `master.csv` (непрерывный метод: e_rel, δ_c).\n\n")
    parts.append(md_table(["N", "r", "n", "= run_coupled", "B5 ρ_N", "B5 m_s", "grid δ_total,c", "grid e_rel,h", "grid Π_h",
                           "master e_rel", "master δ_c"], rows))
    text = "\n".join(parts)
    with open(out / "tables.md", "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


# ------------------------------------------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------------------------------------------


def plan(cfg):
    """List of cells (N, r_label, family, n_x, n_y) in the order of execution (grid by grid)."""
    cells = []
    for lab in cfg["r"]:
        for n in cfg["family_I"]:
            for N in cfg["N"]:
                cells.append((N, lab, "I", n, n))
            if lab in cfg["control_r"] and n in cfg["control_n"]:
                for N in N_CONTROL:
                    cells.append((N, lab, "I", n, n))
        if lab in cfg["family_II"]:
            nx, ny = cfg["family_II"][lab]
            for N in cfg["N"]:
                cells.append((N, lab, "II", nx, ny))
    return cells


def git_head():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, default=OUT_DEFAULT)
    p.add_argument("--quick", action="store_true", help="small matrix for the tests")
    p.add_argument("--tables-only", action="store_true")
    args = p.parse_args(argv)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    if args.tables_only:
        make_tables(out)
        return 0
    if args.quick:
        cfg = dict(QUICK)
    else:
        cfg = {"N": N_STUDY, "r": tuple(R_TEN), "family_I": FAMILY_I, "family_II": FAMILY_II, "control_r": CONTROL_R,
               "control_n": CONTROL_N, "s_run": S_RUN, "s_list": S_LIST, "M": M_CONT, "M_check": M_CHECK,
               "M_navier_m": M_NAVIER_M, "control_max_n": CONTROL_MAX_N}
    t_start = time.perf_counter()
    n_solve_before = kc.N_SOLVE
    gb = load_group_b()
    refs = References(cfg)
    cell_out, step_out, comp_out, n23 = [], [], [], []
    timings = {"cells": 0.0, "compare": 0.0, "continuous": 0.0}
    for N, lab, fam, nx, ny in plan(cfg):
        t0 = time.perf_counter()
        cell = run_cell(N, lab, fam, nx, ny, half_steps=cfg["s_run"], keep=cfg["s_list"])
        timings["cells"] += time.perf_counter() - t0
        crow, srows = cell_rows(cell)
        t1 = time.perf_counter()
        comp_out += compare_cell(cell, refs, cfg)
        timings["compare"] += time.perf_counter() - t1
        if N in N_CONTROL:
            eq = equals_run_coupled(cell)
            crow["equal_run_coupled_bitwise"] = eq
            n23.append((cell, eq))
        cell_out.append(crow)
        step_out += srows
        cell.fields.clear()
        print(f"N={N} r={lab} {fam} {nx}x{ny}: {cell.status} s={cell.s_reached} ({cell.seconds:.1f} s)", flush=True)
    t2 = time.perf_counter()
    cont_out = []
    for N in tuple(cfg["N"]) + N_CONTROL:
        for lab in cfg["r"] if N in cfg["N"] else cfg["control_r"]:
            cont_out += continuous_rows(gb, N, lab, cfg)
    timings["continuous"] = time.perf_counter() - t2
    ord_out = orders(comp_out)
    ctl_out = control_against_existing(n23, comp_out, cont_out) if not args.quick else [
        {"N": c.N, "r_label": c.r_label, "nx": c.n_x, "ny": c.n_y, "equal_run_coupled_bitwise": eq} for c, eq in n23]
    write_csv(out / "cells.csv", cell_out)
    write_csv(out / "steps.csv", step_out)
    write_csv(out / "compare.csv", comp_out)
    write_csv(out / "continuous.csv", cont_out)
    write_csv(out / "orders.csv", ord_out)
    write_csv(out / "control_n23.csv", ctl_out)
    meta = {
        "script": "scripts/r4_n4_n6_study.py", "git_head_at_run": git_head(), "quick": args.quick, "mark": MARK,
        "python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__,
        "platform": platform.platform(), "machine": platform.machine(),
        "N_SOLVE_before": list(n_solve_before), "N_SOLVE_after": list(kc.N_SOLVE),
        "plan": {"N_STUDY": list(cfg["N"]), "N_CONTROL": list(N_CONTROL), "r": list(cfg["r"]),
                 "FAMILY_I": list(cfg["family_I"]), "FAMILY_II": {k: list(v) for k, v in cfg["family_II"].items()},
                 "CONTROL_R": list(cfg["control_r"]), "CONTROL_N": list(cfg["control_n"]), "S_RUN": cfg["s_run"],
                 "S_LIST": list(cfg["s_list"]), "M_CONT": cfg["M"], "M_CHECK": cfg["M_check"],
                 "M_NAVIER_M": cfg["M_navier_m"], "M_NAVIER": M_NAVIER, "CONTROL_MAX_N": cfg["control_max_n"],
                 "SAMPLE_DEN": SAMPLE_DEN, "TRANSFORM": TRANSFORM, "RULE": RULE, "ZETA_Q": ZETA_Q, "ZETA_A": ZETA_A},
        "controller_constants": {"K_RESEARCH": cr.K_RESEARCH, "ETA_MAX": cr.ETA_MAX, "TOL_INF": cr.TOL_INF, "TOL_E": cr.TOL_E,
                                 "S_MIN": cr.S_MIN, "S_MAX": cr.S_MAX},
        "cells": len(cell_out), "control_cells_equal_run_coupled": all(eq for _, eq in n23),
        "seconds": {k: round(v, 1) for k, v in timings.items()} | {"total": round(time.perf_counter() - t_start, 1)},
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    make_tables(out)
    print("done", meta["seconds"], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
