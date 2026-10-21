"""Own tests of kkvim.energy: Pi_h and the energy test (R2, step 7; Q-021).

Definitions follow docs/NUMERICAL_SPEC.md 4.1, 10.1-10.4, 11.3 and table 18.1 (A-7).
The identity delta_P = Q - gamma holds for any finite approximate solution with
consistent fields (10.2), so FAIL branches are exercised by breaking the
consistency of the diagnostic data, not by spoiling the solver accuracy (that is A-6).
No reference implementations and no scratch numbers are used.
"""

import dataclasses
import math

import numpy as np
import pytest

import kkvim.energy as en
import kkvim.iteration as it
from kkvim.coefficients import ERROR_DEGENERATE
from kkvim.energy import (
    ERROR_ENERGY_TEST,
    FAIL,
    NOT_APPLICABLE,
    PASS,
    UNRESOLVED,
    EnergyError,
    check_history,
    check_pair,
    discrete_energy,
    first_check,
    rounding_threshold,
)
from kkvim.first_halfstep import HalfStepSolution
from kkvim.grid import Grid1D
from kkvim.iteration import as_record, next_halfstep, run_halfsteps
from kkvim.first_halfstep import first_halfstep
from kkvim.linear_solver import ERROR_NONFINITE

U = 2.0**-53


def random_field(n_x, n_y, seed=0, length_x=1.0, length_y=1.0):
    grid_x, grid_y = Grid1D(length_x, n_x), Grid1D(length_y, n_y)
    w = np.zeros((n_x + 1, n_y + 1))
    w[1:-1, 1:-1] = np.random.default_rng(seed).standard_normal((n_x - 1, n_y - 1))
    return grid_x, grid_y, w


def loop_laplacian(w, hx, hy, i, j):
    return (w[i - 1, j] - 2 * w[i, j] + w[i + 1, j]) / hx**2 + (w[i, j - 1] - 2 * w[i, j] + w[i, j + 1]) / hy**2


# Pi_h and R(w) by independent loops ---------------------------------------------------


@pytest.mark.parametrize("n_x, n_y", [(4, 5), (6, 4), (7, 9)])
def test_energy_by_independent_loops(n_x, n_y):
    grid_x, grid_y, w = random_field(n_x, n_y, seed=n_x, length_x=1.3, length_y=0.7)
    D, q = 2.1, -0.6
    hx, hy = grid_x.h, grid_y.h
    a = l = 0.0
    for i in range(1, n_x):
        for j in range(1, n_y):
            a += D * hx * hy * loop_laplacian(w, hx, hy, i, j) ** 2
            l += q * hx * hy * w[i, j]
    e = discrete_energy(grid_x, grid_y, w, D, q)
    assert e.a == pytest.approx(a, rel=1e-13, abs=0)
    assert e.l == pytest.approx(l, rel=1e-13, abs=0)
    assert e.pi == pytest.approx(0.5 * a - l, rel=1e-13, abs=0)
    assert e.norm_a == pytest.approx(math.sqrt(a), rel=1e-13, abs=0)


@pytest.mark.parametrize("n_x, n_y", [(4, 5), (7, 9), (20, 13)])
def test_rounding_threshold_by_independent_loops(n_x, n_y):
    grid_x, grid_y, w = random_field(n_x, n_y, seed=3, length_x=0.9, length_y=1.4)
    D, q = 1.7, -2.2
    hx, hy = grid_x.h, grid_y.h
    s_hat = 0.0
    s_hat_wrong = 0.0  # with |Delta_h w| in place of |Delta|_h w
    for i in range(1, n_x):
        for j in range(1, n_y):
            lap = loop_laplacian(w, hx, hy, i, j)
            abs_lap = (abs(w[i - 1, j]) + 2 * abs(w[i, j]) + abs(w[i + 1, j])) / hx**2 + (
                abs(w[i, j - 1]) + 2 * abs(w[i, j]) + abs(w[i, j + 1])
            ) / hy**2
            s_hat += D * hx * hy * abs(lap) * abs_lap + abs(q) * hx * hy * abs(w[i, j])
            s_hat_wrong += D * hx * hy * abs(lap) * abs(lap) + abs(q) * hx * hy * abs(w[i, j])
    N = (n_x - 1) * (n_y - 1)
    expected = U * (math.ceil(math.log2(N)) + 16) * s_hat
    R = rounding_threshold(grid_x, grid_y, w, D, q)
    assert R == pytest.approx(expected, rel=1e-13, abs=0)
    assert R > U * (math.ceil(math.log2(N)) + 16) * s_hat_wrong * (1 + 1e-6)  # |Delta|_h is not |Delta_h|


@pytest.mark.parametrize("N, expected", [(9, 4), (16, 4), (17, 5), (1, 0), (2, 1), (361, 9), (102400, 17)])
def test_ceil_log2_integer(N, expected):
    assert (N - 1).bit_length() == expected == math.ceil(math.log2(N))


def test_energy_input_checks():
    grid_x, grid_y, w = random_field(6, 5)
    with pytest.raises(ValueError):
        discrete_energy(grid_x, grid_y, w.T, 1.0, 1.0)
    bad = w.copy()
    bad[0, 2] = 1.0
    with pytest.raises(ValueError):
        discrete_energy(grid_x, grid_y, bad, 1.0, 1.0)
    for value in (math.nan, math.inf, -math.inf):
        bad = w.copy()
        bad[2, 2] = value
        with pytest.raises(EnergyError) as info:
            discrete_energy(grid_x, grid_y, bad, 1.0, 1.0)
        assert info.value.code == ERROR_NONFINITE
        with pytest.raises(EnergyError) as info:
            rounding_threshold(grid_x, grid_y, bad, 1.0, 1.0)
        assert info.value.code == ERROR_NONFINITE
    with pytest.raises(EnergyError):
        discrete_energy(grid_x, grid_y, w, math.nan, 1.0)
    with pytest.raises(EnergyError):  # overflow is not returned silently
        discrete_energy(grid_x, grid_y, 1e200 * w, 1.0, 1.0)


def test_energy_of_exact_halfstep_is_minus_half_load_work():
    # ChV-12 as an independent property of the solved half-steps (not used by the code).
    history = run_halfsteps(1.0, 1.0, 1.0, 20, 30, 4)
    for r in history.records:
        e = discrete_energy(r.grid_x, r.grid_y, r.field(), r.D, r.q)
        assert e.pi == pytest.approx(-0.5 * e.l, rel=1e-12, abs=0)


# Pair test quantities -----------------------------------------------------------------


def dense_k(grid, A, B, C, D):
    size = grid.interior_size
    lap = (np.diag(-2.0 * np.ones(size)) + np.diag(np.ones(size - 1), 1) + np.diag(np.ones(size - 1), -1)) / grid.h**2
    return D * (A * lap @ lap - 2 * B * lap + C * np.eye(size))


def inexact_record(previous, monkeypatch, relative=1e-6):
    """Next half-step with a deliberately inexact but finite raw solution; everything else consistent."""
    original = it.solve_halfstep

    def perturbed(fixed_grid, fixed_factor, solve_grid, q, D):
        step = original(fixed_grid, fixed_factor, solve_grid, q, D)
        bump = np.sin(np.pi * np.arange(1, solve_grid.n) / solve_grid.n) ** 2
        return HalfStepSolution(raw=step.raw * (1 + relative * bump), coefficients=step.coefficients,
                                solver_diagnostics=step.solver_diagnostics)

    with monkeypatch.context() as m:
        m.setattr(it, "solve_halfstep", perturbed)
        return next_halfstep(previous)


@pytest.mark.parametrize("n_x, n_y", [(20, 20), (20, 40), (21, 20)])
def test_gamma_tau_and_identity(n_x, n_y, monkeypatch):
    first = as_record(first_halfstep(1.2, 0.9, -1.4, n_x, n_y))
    for exact, current in ((True, next_halfstep(first)), (False, inexact_record(first, monkeypatch))):
        check = check_pair(first, current)
        if exact:
            # Exact solve: gamma is rounding noise only, far below the rounding model.
            assert abs(check.gamma) <= check.R_prev + check.R_now
        else:
            # gamma = h rho^T d_f with rho = F 1 - K raw, d_f = X1_scaled - raw (s = 2),
            # with an independently assembled dense K.
            c = current.coefficients
            K = dense_k(current.grid_x, c.A, c.B, c.C, current.D)
            rho = c.F - K @ current.raw
            gamma = current.grid_x.h * float(rho @ (first.x - current.raw))
            assert check.gamma == pytest.approx(gamma, rel=1e-6, abs=0)
        # R_f and tau from their parts.
        w_now, d = current.field(), first.field() - current.field()
        norm_now = discrete_energy(current.grid_x, current.grid_y, w_now, current.D, current.q).norm_a
        norm_d = discrete_energy(current.grid_x, current.grid_y, d, current.D, current.q).norm_a
        R_f = 4 * U * (max(n_x, n_y) / math.pi) ** 2 * norm_now * norm_d
        assert check.R_f == pytest.approx(R_f, rel=1e-12, abs=0)
        assert check.tau == pytest.approx(abs(check.gamma) + 10 * (check.R_prev + check.R_now + check.R_f), rel=1e-15, abs=0)
        assert check.R_prev == rounding_threshold(current.grid_x, current.grid_y, first.field(), current.D, current.q)
        assert check.Q == pytest.approx(0.5 * norm_d**2, rel=1e-15, abs=0)
        # delta_P = Q - gamma (10.2) up to rounding; the sign of gamma matters.
        assert abs(check.delta_P - (check.Q - check.gamma)) <= 10 * (check.R_prev + check.R_now)
        assert check.resolution_ratio == pytest.approx(check.Q / (10 * check.tau), rel=1e-15, abs=0)


def test_inexact_consistent_solution_gives_no_false_fail(monkeypatch):
    first = as_record(first_halfstep(1.0, 1.0, 1.0, 20, 20))
    for relative in (1e-8, 1e-6, 1e-4):
        current = inexact_record(first, monkeypatch, relative)
        check = check_pair(first, current)
        assert abs(check.gamma) > 0.0  # the solution is really inexact
        assert check.status in (PASS, UNRESOLVED)
        assert check.e1 and check.e2


# Statuses -------------------------------------------------------------------------------


def test_s1_not_applicable_and_s2_pass_s4_unresolved():
    checks = check_history(run_halfsteps(1.0, 1.0, 1.0, 20, 20, 4))
    assert checks[0].status == NOT_APPLICABLE and checks[0].P_now < 0.0 and checks[0].tau is None
    assert checks[1].status == PASS and checks[1].Q >= 10 * checks[1].tau
    s4 = checks[3]
    assert s4.status == UNRESOLVED and s4.status not in (PASS, FAIL)
    assert s4.Q < 10 * s4.tau and s4.resolution_ratio < 1.0


def test_resolved_e1_violation_raises_energy_test_error(monkeypatch):
    first = as_record(first_halfstep(1.0, 1.0, 1.0, 20, 20))
    current = next_halfstep(first)
    honest = check_pair(first, current)
    original = en.discrete_energy
    calls = []

    def inflated_now(grid_x, grid_y, w, D, q):
        calls.append(1)
        value = original(grid_x, grid_y, w, D, q)
        if len(calls) == 2:  # P(w_now): inconsistent two-dimensional energy
            return dataclasses.replace(value, pi=value.pi + 3 * honest.Q)
        return value

    monkeypatch.setattr(en, "discrete_energy", inflated_now)
    with pytest.raises(EnergyError) as info:
        check_pair(first, current)
    check = info.value.check
    assert info.value.code == ERROR_ENERGY_TEST
    assert check.status == FAIL and check.e1 is False
    assert check.Q >= 10 * check.tau  # resolved, otherwise UNRESOLVED would be right


def test_e2_only_violation_is_diagnostic_fail(monkeypatch):
    first = as_record(first_halfstep(1.0, 1.0, 1.0, 20, 20))
    current = inexact_record(first, monkeypatch, 1e-6)
    honest = check_pair(first, current)
    assert abs(honest.gamma) > 10 * (honest.R_prev + honest.R_now + honest.R_f)
    monkeypatch.setattr(en, "_gamma", lambda previous, now: 0.0)  # gamma inconsistent with the fields
    check = check_pair(first, current)  # no exception: E1 holds
    assert check.status == FAIL and check.e1 is True and check.e2 is False
    assert check.Q >= 10 * check.tau


def test_wrong_k_in_gamma_is_detected_or_unresolved(monkeypatch):
    first = as_record(first_halfstep(1.0, 1.0, 1.0, 20, 20))
    current = inexact_record(first, monkeypatch, 1e-6)
    original = en.apply_k
    monkeypatch.setattr(en, "apply_k", lambda grid, A, B, C, D, y: original(grid, A, B, C, 1.5 * D, y))
    check = check_pair(first, current)
    assert check.status in (FAIL, UNRESOLVED)
    if check.status == FAIL:
        assert check.Q >= 10 * check.tau and not check.e2


def test_positive_energy_is_degenerate():
    first = as_record(first_halfstep(1.0, 1.0, 1.0, 20, 20))
    flipped = dataclasses.replace(first, x=-first.x)  # field opposite to the load
    e = discrete_energy(first.grid_x, first.grid_y, flipped.field(), first.D, first.q)
    assert e.pi > 0.0  # the plain computation returns the value
    with pytest.raises(EnergyError) as info:
        first_check(flipped)
    assert info.value.code == ERROR_DEGENERATE


@pytest.mark.parametrize("which", ["previous", "current"])
def test_pair_rejects_nonnegative_energy_of_either_field(which):
    history = run_halfsteps(1.0, 1.0, 1.0, 20, 20, 3)
    r1, r2, r3 = history.records
    # Flip the factor that is not the fixed factor of the next half-step: the pair stays
    # valid (w(s-1) in M_s), but the flipped field works against the load, Pi_h > 0.
    if which == "previous":
        pair = (dataclasses.replace(r2, y=-r2.y), r3)
    else:
        pair = (r2, dataclasses.replace(r3, y=-r3.y))
    with pytest.raises(EnergyError) as info:
        check_pair(*pair)
    assert info.value.code == ERROR_DEGENERATE


@pytest.mark.parametrize("call", [1, 2])
def test_pair_rejects_exactly_zero_energy(monkeypatch, call):
    r1, r2 = run_halfsteps(1.0, 1.0, 1.0, 20, 20, 2).records
    original = en.discrete_energy
    calls = []

    def zero_energy(grid_x, grid_y, w, D, q):
        calls.append(1)
        value = original(grid_x, grid_y, w, D, q)
        return dataclasses.replace(value, pi=0.0) if len(calls) == call else value

    monkeypatch.setattr(en, "discrete_energy", zero_energy)
    with pytest.raises(EnergyError) as info:  # Pi_h >= 0, not only > 0 (11.3)
        check_pair(r1, r2)
    assert info.value.code == ERROR_DEGENERATE


def test_nonfinite_record_is_not_masked():
    first = as_record(first_halfstep(1.0, 1.0, 1.0, 20, 20))
    current = next_halfstep(first)
    broken = dataclasses.replace(current, y=np.where(np.arange(current.y.size) == 3, np.nan, current.y))
    with pytest.raises((EnergyError, ValueError)) as info:
        check_pair(first, broken)
    if isinstance(info.value, EnergyError):
        assert info.value.code == ERROR_NONFINITE


# Pair validation --------------------------------------------------------------------------


def test_pair_validation():
    history = run_halfsteps(1.0, 1.0, 1.0, 20, 20, 3)
    r1, r2, r3 = history.records
    with pytest.raises(ValueError):
        check_pair(r1, r3)  # not consecutive
    with pytest.raises(ValueError):
        check_pair(r2, r1)
    with pytest.raises(ValueError):
        check_pair(r1, dataclasses.replace(r2, q=2.0))
    with pytest.raises(ValueError):
        check_pair(r1, dataclasses.replace(r2, k=2))
    with pytest.raises(ValueError):
        check_pair(r1, dataclasses.replace(r2, direction="Y"))
    with pytest.raises(ValueError):  # wrongly scaled fixed factor: w(1) not in M_2
        check_pair(r1, dataclasses.replace(r2, y=r2.y * 1.001))
    other = run_halfsteps(1.0, 1.0, 1.0, 20, 40, 2).records[1]
    with pytest.raises(ValueError):
        check_pair(r1, other)
    with pytest.raises(TypeError):
        check_pair(first_halfstep(1.0, 1.0, 1.0, 20, 20), r2)
    with pytest.raises(TypeError):
        check_history(history.records)


# A-7 behaviour on the main grids ---------------------------------------------------------


@pytest.mark.parametrize("n", [20, 40, 80, 160, 320])
def test_a7_main_grids(n):
    checks = check_history(run_halfsteps(1.0, 1.0, 1.0, n, n, 6))
    assert [c.s for c in checks] == list(range(1, 7))
    assert checks[0].status == NOT_APPLICABLE
    assert checks[1].status == PASS  # s = 2 resolved on all grids
    if n <= 160:
        assert checks[2].status == PASS  # s = 3 resolved for n <= 160
    else:
        assert checks[2].status in (PASS, UNRESOLVED)
    for c in checks[1:]:
        assert c.status != FAIL and c.e1 and c.e2
        assert (c.status == UNRESOLVED) == (c.Q < 10 * c.tau)  # resolution threshold 10 tau
    assert all(c.P_now < 0.0 for c in checks)


def test_a7_diagnostic_grid_640():
    checks = check_history(run_halfsteps(1.0, 1.0, 1.0, 640, 640, 4))
    assert checks[1].status == PASS
    for c in checks[1:]:
        assert c.status != FAIL
        assert (c.status == UNRESOLVED) == (c.Q < 10 * c.tau)
    # s = 3 at n = 640: tau < Q < 10 tau, so only the threshold 10 tau makes it UNRESOLVED.
    assert checks[2].tau < checks[2].Q < 10 * checks[2].tau and checks[2].status == UNRESOLVED


@pytest.mark.parametrize("n_x, n_y, q", [(20, 40, -1.0), (21, 20, 2.5), (7, 9, -0.3)])
def test_unequal_grids_and_negative_load(n_x, n_y, q):
    checks = check_history(run_halfsteps(1.4, 0.6, q, n_x, n_y, 6))
    assert checks[1].status == PASS
    for c in checks[1:]:
        assert c.status != FAIL
        assert abs(c.delta_P - (c.Q - c.gamma)) <= 10 * (c.R_prev + c.R_now)
    assert all(c.P_now < 0.0 for c in checks)
