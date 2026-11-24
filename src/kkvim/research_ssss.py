"""Research-only SSSS path N = 4-6 for ``python -m kkvim run-ssss`` (D-056.7, D-060). RESEARCH_ONLY, not a qualified solver.

The computation is the one of the accepted study ``scripts/r4_n4_n6_study.py`` (``research_halfstep``, ``run_cell``;
docs/R4_N4_N6_STUDY.report.md, QA ``ff9b5d1``): a = 1, b = r, q = D = 1, transform "fft", odd sine start, a fixed number of
half-steps, the stabilization criterion recorded and not stopping the run. The script stays as it is: it imports the
independent references and is not part of the package, so its orchestration is repeated here, operation by operation.
tests/cross/test_cli_ssss_research.py checks that this path equals ``run_cell`` for N = 4-6 and ``run_coupled`` for N = 2-3
bit for bit.

The production range is not touched: ``kkvim.coupled.N_SOLVE`` stays (2, 3), and ``solve_coupled_halfstep`` and
``run_coupled`` keep rejecting N = 4-6 (D-047). D-048 and K_RESEARCH do not extend to N = 4-6 (D-056.7, item 3): the imported
research controller is applied unchanged, as the research choice of the study, and every value is PROVISIONAL /
NOT_QUALIFIED.

Traceability (AGENTS.md, section 2):
- w = sum_j X_j Y_j^T, alternating half-steps, all N functions jointly, N fixed: level A (D-004, D-044, D-045);
- ChF-1/ChF-2, ChF-3, QR with the companion transformation, the energy identity: level B (R4_NUMERICAL_SPEC 2.2-2.4, 3.2, 5.4,
  6.2; D-046), by the functions of ``kkvim.coupled``, ``kkvim.coupled_rank``, ``kkvim.coupled_energy``;
- this loop, the domain check: level C (D-060); checked by the tests above. Limits: D-056.7 and the report, section 7.

Domain (``domain_refusal``): only what the study computed - the ten control r, family I n = 40..1280, the six family II grids
at their r, s <= 40. Anything else is refused and not extended.

Scaling (level B, own derivation of the project; R3_NUMERICAL_SPEC 3.7, NUMERICAL_SPEC ЧВ-14 for N = 1; candidate А4-12 of
R4_NUMERICAL_SPEC for all N). Physical a, b, q, D are mapped to the studied cell by w_h(i, j) = g w_bar_h(i, j; r, n_x, n_y),
g = q a^4 / D, r = b/a. Steps, by induction on s, with bars for a = 1, b = r, q = D = 1:
1. the start X0 depends only on i/n_x, and the operators on a grid of length l are L_l = l^-2 L_1 at the same n, h = l/n;
2. a half-step from a given set U = c U_bar on the grid of length a (the set on the grid of length r a is the same with
   r U_bar): A = c^2 a A_bar, B = c^2 a^-1 B_bar, C = c^2 a^-3 C_bar, F = c q a F_bar, and with the unknown on the grid of the
   same scale D [A L^2 - 2 B L + C] V = F 1 reduces to c^2 D a^-3 [bar system] V = c q a F_bar 1, so V = (q a^4 / (D c)) V_bar
   (ChF-1, ChF-2; R4_NUMERICAL_SPEC 2.2-2.4);
3. hence U V^T = g U_bar V_bar^T after every half-step (c = 1 at s = 1);
4. the QR of 5.4 replaces the set by Q / sqrt(h) and multiplies the companion by R^T; U V^T is unchanged, the columns of Q
   are those of the bar problem;
5. rho_N, kappa_rep, sigma_min(R_V D_c), eta_N, eps_inf and eps_E are dimensionless, so the guards see the same values
   (in exact arithmetic).
So no half-step is repeated for other units: the studied cell is computed and only the output is multiplied by g. Energies
in the history stay in the unit q^2 a^6 / D of the cell.
"""

import math
from fractions import Fraction

import numpy as np

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
from kkvim.coupled_energy import E1_VIOLATED, FACTOR_10, field_from_sets, first_energy, pair_energy
from kkvim.coupled_rank import mode_conditions, orthogonalize_received, rank_diagnostics
from kkvim.energy import EnergyError, discrete_energy
from kkvim.first_halfstep import _positive_input
from kkvim.grid import Grid1D
from kkvim.linear_solver import ERROR_LINEAR_SOLVE, ERROR_NONFINITE
from kkvim.r3_input import validate_r3_sides

N_RESEARCH = (4, 5, 6)
_N_ORCHESTRATION = (2, 3, 4, 5, 6)  # N = 2-3 only for the bitwise control against run_coupled
TRANSFORM = "fft"  # D-048.6, as the study

# the matrix of the study (scripts/r4_n4_n6_study.py, plan)
S_STUDY = 40
FAMILY_I = (40, 80, 160, 320, 640, 1280)
FAMILY_II = {Fraction(1, 5): (1600, 320), Fraction(1, 3): (960, 320), Fraction(1, 2): (640, 320), Fraction(2): (320, 640),
             Fraction(3): (320, 960), Fraction(5): (320, 1600)}


def control_ratio(a: float, b: float) -> Fraction | None:
    """The control r of D-048 equal to b/a within the representation of the input floats (4u, as D-038), or None."""
    fa, fb = Fraction(a), Fraction(b)
    tol = 4 * Fraction(cr.UNIT_ROUNDOFF)
    for r in cr.CONTROL_R:
        if abs(fb - r * fa) <= tol * max(fb, r * fa):
            return r
    return None


def domain_refusal(a: float, b: float, n_x: int, n_y: int, N: int, half_steps: int) -> str | None:
    """Why (a, b, n_x, n_y, half_steps) lies outside the studied domain of N = 4-6, or None if it is inside.

    n_x < 2N or n_y < 2N is not refused here: the run itself ends with the proven S-1 before the first half-step.
    """
    if half_steps > S_STUDY:
        return f"half_steps = {half_steps} > {S_STUDY}: the study of N = 4-6 computed s <= {S_STUDY}"
    if min(n_x, n_y) < 2 * N:
        return None
    r = control_ratio(a, b)
    if r is None:
        controls = ", ".join(str(value) for value in cr.CONTROL_R)
        return f"b/a = {b / a!r} is not one of the ten r of the study of N = 4-6: {controls}"
    if n_x == n_y and n_x in FAMILY_I:
        return None
    if FAMILY_II.get(r) == (n_x, n_y):
        return None
    allowed = f"n_x = n_y in {FAMILY_I}"
    if r in FAMILY_II:
        allowed += f" or (n_x, n_y) = {FAMILY_II[r]}"
    return f"the grid {n_x} x {n_y} at r = {r} was not computed by the study of N = 4-6: {allowed}"


def research_halfstep(fixed_grid, fixed, solve_grid, q, D, *, direction, transform):
    """ChF-1 (direction "Y") or ChF-2 ("X") by ChF-3: the operations of solve_coupled_halfstep, in its order, without its
    N_SOLVE guard, which is D-047's range and stays in force for the production path."""
    if direction not in ("X", "Y"):
        raise ValueError(f"direction must be 'X' or 'Y', got {direction!r}")
    if transform not in TRANSFORMS:
        raise ValueError(f"transform must be one of {TRANSFORMS}, got {transform!r}")
    D = _positive_input("D", D)
    coefficients = coupled_coefficients(fixed_grid, fixed, q)
    N, n = coefficients.N, solve_grid.n
    if N not in _N_ORCHESTRATION:
        raise CoupledHalfStepError(ERROR_INPUT, f"the research orchestration covers N in {_N_ORCHESTRATION}",
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


def run_research(r: float, n_x: int, n_y: int, N: int, *, half_steps: int) -> cr.CoupledRun:
    """The cell (N, r, n_x, n_y) of the study at a = 1, b = r, q = D = 1; statuses of the research controller."""
    if isinstance(N, bool) or not isinstance(N, (int, np.integer)) or N not in N_RESEARCH:
        return cr.CoupledRun(ERROR_INPUT, None, (), None, None, None, None,
                             reason=f"the research path covers N in {N_RESEARCH}, got {N!r}; N = 2-3 is run_coupled")
    return _orchestrate(r, n_x, n_y, N, half_steps)


def _orchestrate(r, n_x, n_y, N, half_steps) -> cr.CoupledRun:
    """The loop of ``run_cell``: solve -> eta_N -> rank and set diagnostics -> energy -> guards -> QR -> next half-step."""
    a, b = validate_r3_sides(1.0, r)
    q, D = 1.0, 1.0
    grid_x, grid_y = Grid1D(a, n_x), Grid1D(b, n_y)
    floor = cr.eps_floor(n_x, n_y, N)
    params = cr.CoupledRunParameters(a=a, b=b, D=D, q=q, n_x=int(n_x), n_y=int(n_y), N=int(N), transform=TRANSFORM,
                                     half_steps=int(half_steps), research_mode=True, K_research=cr.K_RESEARCH,
                                     eps_floor=floor, tol_inf=cr.TOL_INF, tol_E=cr.TOL_E, s_min=cr.S_MIN, s_max=cr.S_MAX,
                                     eta_max=cr.ETA_MAX, stop_on_stabilization=False)
    for axis, grid in (("x", grid_x), ("y", grid_y)):  # S-1 before the first half-step (ChV4-8, ChV4-9)
        if grid.n < 2 * N:
            return cr.CoupledRun(cr.STOPPED_RANK_LOSS_PROVEN, params, (), grid_x, grid_y, None, None,
                                 reason=f"n_{axis} = {grid.n} < 2N = {2 * N}: rank W^1 < N is proven (ChV4-8, ChV4-9)")
    if cr.TOL_E < 10.0 * floor:  # D-048.3 configuration rule of the research controller
        return cr.CoupledRun(ERROR_INPUT, params, (), grid_x, grid_y, None, None,
                             reason=f"tol_E = {cr.TOL_E!r} < 10 eps_floor^(N) = {10.0 * floor!r} (D-048.3)")

    X = odd_sine_set(grid_x, N)
    Y = None
    steps = []
    s2 = {}
    first_stabilization = None

    def stop(status, reason, error=None):
        return cr.CoupledRun(status, params, tuple(steps), grid_x, grid_y, X if Y is not None else None, Y, reason=reason,
                             error=error, energy_s2_checked=bool(s2), first_stabilization_s=first_stabilization, **s2)

    previous_field, previous_ok = None, False
    for s in range(1, half_steps + 1):
        direction = "Y" if s % 2 else "X"
        given_grid, solve_grid = (grid_x, grid_y) if direction == "Y" else (grid_y, grid_x)
        given, companion = (X, Y) if direction == "Y" else (Y, X)
        try:
            step = research_halfstep(given_grid, given, solve_grid, q, D, direction=direction, transform=TRANSFORM)
        except CoupledHalfStepError as exc:
            return stop(cr.STOPPED_RANK_LOSS_PROVEN if exc.code == ERROR_DEGENERATE else exc.code,
                        f"half-step s={s}: {exc}", exc)
        _, eta = coupled_residual(solve_grid, step.coefficients, D, step.solved)
        eta_initial, refined = eta, False
        if not (eta <= cr.ETA_MAX):  # D-048.2: exactly one refinement step
            try:
                step = refine_once(step)
            except CoupledHalfStepError as exc:
                return stop(exc.code, f"refinement of s={s}: {exc}", exc)
            _, eta = coupled_residual(solve_grid, step.coefficients, D, step.solved)
            refined = True
            if not (eta <= cr.ETA_MAX):
                return stop(ERROR_LINEAR_SOLVE if math.isfinite(eta) else ERROR_NONFINITE,
                            f"s={s}: eta_N = {eta!r} > {cr.ETA_MAX!r} after one refinement step (before: {eta_initial!r})")
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
            steps.append(cr._log(s, direction, eta, rank, cond_A, cond_M, None, None, None, None, None, False, None, None,
                                 exc.code, eta_initial, refined))
            return stop(exc.code, f"energy of s={s}: {exc}", exc)
        m_s = None if energy.tau is None else (energy.Q / (FACTOR_10 * energy.tau) if energy.tau > 0 else math.inf)
        if s == 2:  # the mandatory area of D-048.1 is N = 2-3 only: not applicable here
            s2 = dict(energy_s2_status=energy.status, energy_s2_m=m_s)
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

        def log(guards, reason, qr_done=False, qr_error=None, P_qr=None):
            steps.append(cr._log(s, direction, eta, rank, cond_A, cond_M, energy, w_max, eps_inf, eps_E, guards, qr_done,
                                 qr_error, P_qr, reason, eta_initial, refined, m_s, stab_ok))

        if not (energy.P < 0.0):
            log(None, ERROR_DEGENERATE)
            return stop(ERROR_DEGENERATE, f"Pi_h(w_{s}) = {energy.P!r} >= 0")
        if energy.status == E1_VIOLATED:
            log(None, cr.PROVISIONAL_ENERGY_E1_STOP)
            return stop(cr.PROVISIONAL_ENERGY_E1_STOP, f"resolved energy test violates E1 at s={s} (research model)")
        guards = cr._guards(rank)
        if not math.isfinite(eta) or not rank.finite:
            log(guards, ERROR_NONFINITE)
            return stop(ERROR_NONFINITE, f"non-finite diagnostics at s={s}")
        if not (guards.applicable and guards.rank_ok):
            log(guards, cr.PROVISIONAL_RANK_GUARD_STOP)
            why = "K u kappa_rep >= 1" if not guards.applicable else "rho_N <= K u kappa_rep"
            return stop(cr.PROVISIONAL_RANK_GUARD_STOP, f"s={s}: {why}; no QR, N kept")
        if not guards.set_ok:
            log(guards, cr.PROVISIONAL_SET_GUARD_STOP)
            return stop(cr.PROVISIONAL_SET_GUARD_STOP, f"s={s}: sigma_min(R_V D_c) <= K u; no QR, N kept")
        if s >= cr.S_MIN and stab_ok and previous_ok and first_stabilization is None:
            first_stabilization = s
        previous_ok = bool(stab_ok)
        if s == cr.S_MAX:
            log(guards, cr.MAX_HALF_STEPS)
            return stop(cr.MAX_HALF_STEPS, f"s_max = {cr.S_MAX} reached")
        if s == half_steps:
            log(guards, cr.RESEARCH_LIMIT_REACHED)
            return stop(cr.RESEARCH_LIMIT_REACHED, f"{half_steps} half-steps done; this is not convergence")
        ortho = orthogonalize_received(solve_grid, received, given)
        if direction == "Y":
            X, Y = ortho.companion, ortho.received
        else:
            X, Y = ortho.received, ortho.companion
        P_qr = discrete_energy(grid_x, grid_y, field_from_sets(grid_x, grid_y, X, Y), D, q).pi
        log(guards, None, True, ortho.field_error, P_qr)
    raise AssertionError("unreachable")
