"""Own tests of the D-048 implementation (R4, B-1 .. B-6): stopping, residual control, mandatory energy test,
dense DST-I reduction and metadata. Builder unit tests, not QA; every research result is PROVISIONAL / NOT_QUALIFIED.

D-048.1 .. D-048.6 (docs/DECISIONS.md); R4_NUMERICAL_SPEC 5.4.1, 6.3, 7. Fault injection (refinement) is done here
by monkeypatch only; the production code has no switch for it.
"""

import dataclasses
import math
from decimal import Decimal, getcontext

import numpy as np
import pytest

import kkvim.coupled as cp
import kkvim.coupled_run as cr
from kkvim.coefficients import ERROR_INPUT
from kkvim.coupled import coupled_residual, refine_once, sine_matrix, sine_transform, solve_coupled_halfstep
from kkvim.coupled_energy import UNRESOLVED
from kkvim.coupled_run import (
    HALF_STEP_LIMIT_REACHED,
    MARK,
    MAX_HALF_STEPS,
    NOT_QUALIFIED,
    PROVISIONAL_FIELD_STABILIZED,
    PROVISIONAL_RANK_GUARD_STOP,
    RESEARCH_LIMIT_REACHED,
    eps_floor,
    is_mandatory_s2_grid,
    run_coupled,
)
from kkvim.grid import Grid1D
from kkvim.linear_solver import ERROR_LINEAR_SOLVE, ERROR_NONFINITE

U = 2.0**-53
D, Q = 3.1, 2.3


def gamma(n):
    return n * U / (1 - n * U)


# B-1: stopping policy (research mode only) -------------------------------------------------

def test_constants_of_d048():
    assert (cr.TOL_INF, cr.TOL_E, cr.S_MIN, cr.S_MAX, cr.ETA_MAX, cr.K_RESEARCH) == (1e-10, 1e-8, 3, 100, 1e-12, 10**6)


@pytest.mark.parametrize("n_x, n_y, N", [(40, 40, 2), (160, 32, 3), (1280, 1280, 3), (320, 1600, 2)])
def test_eps_floor_formula(n_x, n_y, N):
    gamma_N = N * U / (1 - N * U)
    expected = 10 * U * (max(n_x, n_y) / math.pi) ** 2 * math.sqrt(N) / (1 - gamma_N * math.sqrt(N))
    assert eps_floor(n_x, n_y, N) == expected
    assert 10 * eps_floor(n_x, n_y, N) <= cr.TOL_E  # the checked grids satisfy the configuration rule


def test_configuration_rule_is_error_input_in_research_mode_only(monkeypatch):
    big = run_coupled(1.0, 1.0, D, Q, 2400, 2400, 3, transform="fft", half_steps=5, research_mode=True)
    assert 10 * eps_floor(2400, 2400, 3) > cr.TOL_E
    assert big.status == ERROR_INPUT and big.s == 0 and "eps_floor" in big.reason
    monkeypatch.setattr(cr, "TOL_E", 1e-14)  # temporary in the test only
    small = run_coupled(1.0, 2.0, D, Q, 20, 40, 2, transform="fft", half_steps=5, research_mode=True)
    assert small.status == ERROR_INPUT and small.s == 0
    normative = run_coupled(1.0, 2.0, D, Q, 20, 40, 2, transform="fft", half_steps=5)
    assert normative.status == NOT_QUALIFIED and normative.s == 1  # default behaviour unchanged


@pytest.mark.parametrize("a, b, n_x, n_y, N", [(1.0, 1.0, 40, 40, 2), (1.0, 0.2, 160, 32, 3), (1.0, 5.0, 32, 160, 3),
                                               (2.0, 3.0, 24, 36, 2)])
def test_stabilization_stops_with_provisional_field_stabilized(a, b, n_x, n_y, N):
    run = run_coupled(a, b, D, Q, n_x, n_y, N, transform="fft", half_steps=60, research_mode=True)
    assert run.status == PROVISIONAL_FIELD_STABILIZED and not run.converged and run.mark == MARK
    last, before = run.steps[-1], run.steps[-2]
    assert run.s >= cr.S_MIN and last.stabilization_ok and before.stabilization_ok
    assert last.eps_inf <= cr.TOL_INF and last.eps_E <= cr.TOL_E and before.eps_inf <= cr.TOL_INF and before.eps_E <= cr.TOL_E
    assert not any(x.stabilization_ok and y.stabilization_ok for x, y in zip(run.steps[1:-2], run.steps[2:-1]))
    assert all(step.guards.passed for step in run.steps) and not last.qr_done
    assert last.stop_reason == PROVISIONAL_FIELD_STABILIZED and "not convergence" in run.reason


def test_explicit_limit_before_stabilization_and_stabilization_on_the_last_allowed_step():
    full = run_coupled(1.0, 2.0, D, Q, 20, 40, 3, transform="fft", half_steps=60, research_mode=True)
    s_star = full.s
    early = run_coupled(1.0, 2.0, D, Q, 20, 40, 3, transform="fft", half_steps=s_star - 1, research_mode=True)
    assert early.status == RESEARCH_LIMIT_REACHED and early.s == s_star - 1
    exact = run_coupled(1.0, 2.0, D, Q, 20, 40, 3, transform="fft", half_steps=s_star, research_mode=True)
    assert exact.status == PROVISIONAL_FIELD_STABILIZED and exact.s == s_star
    assert np.array_equal(exact.field(), full.field())


@pytest.mark.parametrize("half_steps", [6, 50])
def test_max_half_steps_without_stabilization_and_no_step_beyond(half_steps, monkeypatch):
    monkeypatch.setattr(cr, "TOL_INF", 0.0)  # temporary: no stabilization possible
    monkeypatch.setattr(cr, "S_MAX", 6)
    run = run_coupled(1.0, 2.0, D, Q, 20, 40, 2, transform="fft", half_steps=half_steps, research_mode=True)
    assert run.status == MAX_HALF_STEPS and run.s == 6 and not run.steps[-1].qr_done
    assert all(step.guards.passed for step in run.steps)


def test_s_min_is_respected(monkeypatch):
    monkeypatch.setattr(cr, "TOL_INF", 1.0)  # temporary: the criterion holds from s = 2 on
    monkeypatch.setattr(cr, "TOL_E", 1.0)
    run = run_coupled(1.0, 2.0, D, Q, 20, 40, 2, transform="fft", half_steps=10, research_mode=True)
    assert run.status == PROVISIONAL_FIELD_STABILIZED and run.s == cr.S_MIN  # not at s = 2


def test_guards_precede_the_stabilization_status(monkeypatch):
    full = run_coupled(1.0, 2.0, D, Q, 20, 40, 2, transform="fft", half_steps=60, research_mode=True)
    real, calls = cr.rank_diagnostics, []

    def forced(*args):
        calls.append(1)
        r = real(*args)
        return dataclasses.replace(r, rho_N=1e-12) if len(calls) == full.s else r

    monkeypatch.setattr(cr, "rank_diagnostics", forced)
    run = run_coupled(1.0, 2.0, D, Q, 20, 40, 2, transform="fft", half_steps=60, research_mode=True)
    assert run.status == PROVISIONAL_RANK_GUARD_STOP and run.s == full.s and not run.steps[-1].qr_done


def test_normative_mode_has_no_d048_stopping():
    one = run_coupled(1.0, 2.0, D, Q, 20, 40, 2, transform="fft", half_steps=1)
    more = run_coupled(1.0, 2.0, D, Q, 20, 40, 2, transform="fft", half_steps=60)
    assert (one.status, more.status, more.s) == (HALF_STEP_LIMIT_REACHED, NOT_QUALIFIED, 1)
    p = more.parameters
    assert (p.eps_floor, p.tol_inf, p.tol_E, p.s_min, p.s_max, p.eta_max) == (None,) * 6
    assert more.steps[0].eta_initial is None and more.steps[0].stabilization_ok is None


# B-3: mandatory energy test at s = 2 -------------------------------------------------------

@pytest.mark.parametrize("a, b, n_x, n_y, expected", [
    (1.0, 1.0, 160, 160, True), (1.0, 1 / 3, 640, 640, True), (1.0, 1.4, 1280, 1280, True), (1.0, 1.0, 320, 320, False),
    (1.0, 1.0, 40, 40, False), (1.0, 0.2, 1600, 320, True), (1.0, 1 / 3, 960, 320, True), (1.0, 5.0, 320, 1600, True),
    (1.0, 1.0, 1600, 320, False), (1.0, 1.1, 160, 160, False), (2.0, 6.0, 320, 960, True),
])
def test_mandatory_s2_grids(a, b, n_x, n_y, expected):
    assert is_mandatory_s2_grid(a, b, n_x, n_y) is expected


def test_mandatory_s2_unresolved_is_reported_and_does_not_stop(monkeypatch):
    real, calls = cr.pair_energy, []

    def unresolved_at_s2(*args, **kwargs):  # the first pair_energy call is the test of s = 2
        calls.append(1)
        e = real(*args, **kwargs)
        return dataclasses.replace(e, status=UNRESOLVED) if len(calls) == 1 else e

    monkeypatch.setattr(cr, "pair_energy", unresolved_at_s2)
    run = run_coupled(1.0, 1.0, D, Q, 160, 160, 2, transform="fft", half_steps=4, research_mode=True)
    assert run.status == RESEARCH_LIMIT_REACHED and run.s == 4  # not stopped by UNRESOLVED
    assert run.energy_s2_checked and run.energy_s2_mandatory and run.energy_s2_mandatory_unresolved is True
    assert run.energy_s2_status == UNRESOLVED and run.energy_s2_m == run.steps[1].m_s and "m_2" in run.energy_s2_reason
    assert "PASS" not in run.status


def test_s2_record_resolved_not_mandatory_and_not_reached():
    resolved = run_coupled(1.0, 1.0, D, Q, 160, 160, 2, transform="fft", half_steps=3, research_mode=True)
    assert resolved.energy_s2_mandatory and resolved.energy_s2_mandatory_unresolved is False
    assert resolved.energy_s2_m >= 1 and resolved.energy_s2_reason is None
    free = run_coupled(1.0, 2.0, D, Q, 20, 40, 2, transform="fft", half_steps=3, research_mode=True)
    assert free.energy_s2_checked and free.energy_s2_mandatory is False and free.energy_s2_mandatory_unresolved is False
    none = run_coupled(1.0, 2.0, D, Q, 20, 40, 2, transform="fft", half_steps=1, research_mode=True)
    assert not none.energy_s2_checked and none.energy_s2_mandatory is None and none.energy_s2_mandatory_unresolved is None


# B-4: residual control and exactly one refinement step -------------------------------------

def perturbing_solver(monkeypatch, at_s, factor):
    real, count = cr.solve_coupled_halfstep, []

    def solver(*args, **kwargs):
        step = real(*args, **kwargs)
        count.append(1)
        if len(count) == at_s:
            rng = np.random.default_rng(1)
            step = dataclasses.replace(step, solved=step.solved * (1 + factor * rng.standard_normal(step.solved.shape)))
        return step

    monkeypatch.setattr(cr, "solve_coupled_halfstep", solver)


def test_no_refinement_below_eta_max():
    run = run_coupled(1.0, 2.0, D, Q, 20, 40, 3, transform="fft", half_steps=8, research_mode=True)
    assert all(not step.refined and step.eta_initial <= cr.ETA_MAX and step.eta_N == step.eta_initial for step in run.steps)


def test_perturbed_solution_is_repaired_by_one_refinement_step(monkeypatch):
    plain = run_coupled(1.0, 2.0, D, Q, 20, 40, 2, transform="fft", half_steps=4, research_mode=True)
    perturbing_solver(monkeypatch, at_s=2, factor=1e-6)
    run = run_coupled(1.0, 2.0, D, Q, 20, 40, 2, transform="fft", half_steps=4, research_mode=True)
    step = run.steps[1]
    assert step.refined and step.eta_initial > cr.ETA_MAX >= step.eta_N
    assert run.status == plain.status == RESEARCH_LIMIT_REACHED and not any(s.refined for i, s in enumerate(run.steps) if i != 1)
    cond = max(s.cond_M_max for s in plain.steps)
    assert np.max(np.abs(run.field() - plain.field())) <= 64 * gamma(64) * cond * np.max(np.abs(plain.field()))


def test_refinement_that_does_not_reduce_the_residual_is_error_linear_solve(monkeypatch):
    real = cp._correction_modes
    monkeypatch.setattr(cp, "_correction_modes", lambda *a, **k: 0.5 * real(*a, **k))  # injected faulty correction
    perturbing_solver(monkeypatch, at_s=2, factor=1e-6)
    run = run_coupled(1.0, 2.0, D, Q, 20, 40, 2, transform="fft", half_steps=4, research_mode=True)
    assert run.status == ERROR_LINEAR_SOLVE and run.s == 1 and "after one refinement step" in run.reason


def test_nonfinite_solution_is_error_nonfinite(monkeypatch):
    real = cr.solve_coupled_halfstep

    def nan_solver(*args, **kwargs):
        step = real(*args, **kwargs)
        solved = step.solved.copy()
        solved[0, 0] = np.nan
        return dataclasses.replace(step, solved=solved)

    monkeypatch.setattr(cr, "solve_coupled_halfstep", nan_solver)
    run = run_coupled(1.0, 2.0, D, Q, 20, 40, 2, transform="fft", half_steps=4, research_mode=True)
    assert run.status == ERROR_NONFINITE and run.s == 0


def test_refine_once_uses_the_nodal_residual_and_the_same_mode_systems():
    gx, gy = Grid1D(1.0, 20), Grid1D(2.0, 40)
    step = solve_coupled_halfstep(gx, odd_sine_set_x(gx), gy, Q, D, direction="Y", transform="fft")
    bad = dataclasses.replace(step, solved=step.solved * 1.000001)
    _, eta_bad = coupled_residual(gy, step.coefficients, D, bad.solved)
    fixed = refine_once(bad)
    _, eta_fixed = coupled_residual(gy, step.coefficients, D, fixed.solved)
    assert eta_bad > cr.ETA_MAX >= eta_fixed
    assert np.max(np.abs(fixed.solved - step.solved)) <= 64 * gamma(64) * np.max(np.abs(step.solved))


def odd_sine_set_x(grid):
    from kkvim.coupled import odd_sine_set
    return odd_sine_set(grid, 2)


# B-5: dense DST-I with exact integer reduction ----------------------------------------------

getcontext().prec = 60
PI = Decimal("3.14159265358979323846264338327950288419716939937510582097494459")


def dsin(x: Decimal) -> Decimal:
    x2, term, total, k = x * x, x, x, 1
    while abs(term) > Decimal(10) ** -58:
        term = -term * x2 / ((2 * k) * (2 * k + 1))
        total += term
        k += 1
    return total


def exact_psi(n):
    """psi_k(i) = sqrt(2/n) sin(pi i k / n) in 60 digits, via the exact reduction of i k (reference only)."""
    root = (Decimal(2) / Decimal(n)).sqrt()
    table = {m: dsin(PI * m / n) for m in range(n // 2 + 1)}
    out = np.empty((n - 1, n - 1), dtype=object)
    for i in range(1, n):
        for k in range(1, n):
            m = (i * k) % (2 * n)
            sign = -1 if m >= n else 1
            m = m - n if m >= n else m
            m = n - m if 2 * m > n else m
            out[i - 1, k - 1] = sign * root * table[m]
    return out


@pytest.mark.parametrize("n", [4, 5, 12, 40, 160])
def test_dense_matrix_exact_structure(n):
    P = sine_matrix(Grid1D(1.0, n))
    i = np.arange(1, n)
    assert np.array_equal(P, P.T)  # symmetry, exactly
    zero = (np.outer(i, i) % n) == 0
    assert np.all(P[zero] == 0.0)  # exact zeros
    k = i[np.newaxis, :]
    assert np.array_equal(P[::-1, :], P * np.where(k % 2 == 1, 1.0, -1.0))  # psi_k(n - i) = (-1)^(k+1) psi_k(i), exactly
    assert np.max(np.abs(P @ P - np.eye(n - 1))) <= 2 * gamma(n + 8)


@pytest.mark.parametrize("n", [5, 40, 160])
def test_dense_entries_against_the_7u_model(n):
    """QA model premise (sin within 1 ulp on [0, pi/2]) is checked by this comparison, not assumed."""
    P = sine_matrix(Grid1D(1.0, n))
    ref = exact_psi(n)
    err = np.vectorize(lambda a, b: abs(Decimal(float(a)) - b))(P, ref)
    bound = np.vectorize(lambda b: Decimal(7 * U) * abs(b))(ref)
    assert np.all(err <= bound)
    x = np.random.default_rng(n).standard_normal(n - 1)
    y = P @ x
    exact = [sum(ref[r, c] * Decimal(float(x[c])) for c in range(n - 1)) for r in range(n - 1)]
    scale = np.abs(P) @ np.abs(x)
    model = 7 * U + gamma(n - 1) * (1 + 7 * U)
    assert all(abs(Decimal(float(y[r])) - exact[r]) <= Decimal(model * scale[r]) for r in range(n - 1))


def test_fft_transform_path_is_unchanged_by_the_dense_fix():
    g = Grid1D(1.3, 160)
    x = np.random.default_rng(3).standard_normal((159, 3))
    import scipy.fft
    assert np.array_equal(sine_transform(g, x, "fft"), scipy.fft.dst(x, type=1, norm="ortho", axis=0))


# B-6: metadata and the fixed-step fft path ---------------------------------------------------

def test_metadata_of_a_research_run():
    run = run_coupled(1.0, 2.0, D, Q, 20, 40, 3, transform="fft", half_steps=60, research_mode=True)
    p = run.parameters
    assert (p.tol_inf, p.tol_E, p.s_min, p.s_max, p.eta_max) == (1e-10, 1e-8, 3, 100, 1e-12)
    assert p.eps_floor == eps_floor(20, 40, 3) and run.reason and run.mark == MARK
    for step in run.steps:
        assert step.rank.rho_N > 0 and step.rank.kappa_rep >= 1 - 1e-12 and step.rank.set_sigma_min > 0
        assert step.eta_initial is not None and step.mark == MARK
        assert (step.m_s is None) == (step.s == 1)
        if step.qr_done:
            assert step.qr_field_error is not None and step.P_after_qr is not None and step.energy.P is not None


def test_fixed_step_fft_path_equals_the_kernel_sequence():
    """Same platform and environment: run_coupled before the automatic stop is the plain sequence of kernel calls."""
    from kkvim.coupled import odd_sine_set
    from kkvim.coupled_rank import orthogonalize_received
    gx, gy = Grid1D(1.0, 20), Grid1D(2.0, 40)
    run = run_coupled(1.0, 2.0, D, Q, 20, 40, 3, transform="fft", half_steps=5, research_mode=True)
    X, Y = odd_sine_set(gx, 3), None
    for s in range(1, 6):
        if s % 2:
            step = solve_coupled_halfstep(gx, X, gy, Q, D, direction="Y", transform="fft")
            Y = step.solved
            if s < 5:
                o = orthogonalize_received(gy, Y, X)
                X, Y = o.companion, o.received
        else:
            step = solve_coupled_halfstep(gy, Y, gx, Q, D, direction="X", transform="fft")
            X = step.solved
            o = orthogonalize_received(gx, X, Y)
            X, Y = o.received, o.companion
    assert run.status == RESEARCH_LIMIT_REACHED
    assert np.array_equal(run.x, X) and np.array_equal(run.y, Y)


def test_s_min_larger_than_three_is_respected(monkeypatch):
    monkeypatch.setattr(cr, "TOL_INF", 1.0)  # temporary: the criterion holds from s = 2 on
    monkeypatch.setattr(cr, "TOL_E", 1.0)
    monkeypatch.setattr(cr, "S_MIN", 5)
    run = run_coupled(1.0, 2.0, D, Q, 20, 40, 2, transform="fft", half_steps=10, research_mode=True)
    assert run.status == PROVISIONAL_FIELD_STABILIZED and run.s == 5


def test_unresolved_s2_on_a_free_grid_is_not_a_mandatory_failure(monkeypatch):
    real, calls = cr.pair_energy, []

    def unresolved_at_s2(*args, **kwargs):
        calls.append(1)
        e = real(*args, **kwargs)
        return dataclasses.replace(e, status=UNRESOLVED) if len(calls) == 1 else e

    monkeypatch.setattr(cr, "pair_energy", unresolved_at_s2)
    run = run_coupled(1.0, 2.0, D, Q, 20, 40, 2, transform="fft", half_steps=3, research_mode=True)
    assert run.energy_s2_status == UNRESOLVED and run.energy_s2_mandatory is False
    assert run.energy_s2_mandatory_unresolved is False and run.energy_s2_reason is None
