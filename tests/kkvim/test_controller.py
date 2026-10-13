"""Own tests of kkvim.controller: automatic stopping (R2, step 8; NUMERICAL_SPEC.md 11.1-11.3).

The stopping rule is tested as a pure function on synthetic diagnostics, without
the solver. Physical runs check the integration and the strict mode; their stop
numbers are not fixed as expected values. A-11 is qualified by QA.
"""

import math

import numpy as np
import pytest

import kkvim.controller as ctl
import kkvim.iteration as it
from kkvim.coefficients import ERROR_DEGENERATE, ERROR_INPUT, CoefficientError
from kkvim.controller import (
    CONVERGED,
    MAX_HALF_STEPS,
    STAGNATED,
    TAIL_ESTIMATE_NOTE,
    eps_floor,
    run_until_stopped,
    stopping_decision,
    strict_parameters,
    thetas,
)
from kkvim.energy import ERROR_ENERGY_TEST, NOT_APPLICABLE, PASS, UNRESOLVED, EnergyError, discrete_energy
from kkvim.iteration import run_halfsteps
from kkvim.linear_solver import ERROR_LINEAR_SOLVE, LinearSolveError

U = 2.0**-53
TOL_INF, TOL_E = 1e-10, 1e-8


def decide(s, rows, s_min=3, s_max=100, tol_inf=TOL_INF, tol_E=TOL_E):
    return stopping_decision(s, rows, tol_inf, tol_E, s_min, s_max)


def rows_from(eps_list, start=2):
    """(s, eps_inf, eps_E) rows from (eps_inf, eps_E) pairs for s = start, start + 1, ..."""
    return [(start + i, e_inf, e_E) for i, (e_inf, e_E) in enumerate(eps_list)]


# Pure decision function ------------------------------------------------------------------

GOOD = (1e-12, 1e-10)
BAD = (1e-6, 1e-4)


def test_converged_needs_two_consecutive_satisfied_steps():
    rows = rows_from([BAD, GOOD, GOOD])  # s = 2, 3, 4
    assert decide(4, rows) == CONVERGED
    assert decide(3, rows[:2]) is None  # one satisfied step: continue


def test_single_good_step_between_bad_ones_continues():
    rows = rows_from([BAD, GOOD, BAD, GOOD])  # s = 2..5
    for s in range(2, 6):
        assert decide(s, rows[: s - 1]) is None
    assert decide(6, rows_from([BAD, GOOD, BAD, GOOD, GOOD])) == CONVERGED


def test_both_tolerances_required():
    only_inf = (1e-12, 1e-6)
    only_e = (1e-8, 1e-10)
    assert decide(3, rows_from([only_inf, only_inf])) is None
    assert decide(3, rows_from([only_e, only_e])) is None
    assert decide(3, rows_from([(TOL_INF, TOL_E), (TOL_INF, TOL_E)])) == CONVERGED  # <= on the boundary


def test_not_converged_before_s_min():
    rows = rows_from([GOOD, GOOD, GOOD, GOOD])  # s = 2..5
    assert decide(3, rows[:2], s_min=5) is None
    assert decide(4, rows[:3], s_min=5) is None
    assert decide(5, rows, s_min=5) == CONVERGED


def stagnating(count, ratio=0.95, start=1e-3):
    return [(1.0, start * ratio**i) for i in range(count)]


def test_stagnated_after_ten_consecutive_steps():
    rows = rows_from(stagnating(11))  # s = 2..12: theta defined for s = 3..12 (ten values)
    assert decide(11, rows[:10]) is None  # nine thetas
    assert decide(12, rows) == STAGNATED


def test_stagnation_run_is_reset():
    seq = stagnating(6) + [(1.0, 1e-3 * 0.95**5 * 0.5)] + stagnating(9, start=1e-3 * 0.95**5 * 0.5 * 0.95)
    rows = rows_from(seq)
    assert all(decide(r[0], rows[: i + 1]) != STAGNATED for i, r in enumerate(rows))


def test_false_stagnation_below_tol_e():
    # theta >= 0.9 for ten steps, but eps_E <= tol_E: not STAGNATED (eps_inf blocks CONVERGED).
    seq = [(1.0, TOL_E * 0.99**i) for i in range(12)]
    rows = rows_from(seq)
    assert all(t is None or t >= 0.9 for t in thetas(r[2] for r in rows))
    assert all(decide(r[0], rows[: i + 1], s_max=1000) is None for i, r in enumerate(rows))


def test_undefined_theta_is_not_replaced():
    seq = stagnating(5) + [(1.0, 0.0), (1.0, 1e-3)] + stagnating(9, start=1e-3 * 0.95)
    rows = rows_from(seq)
    values = thetas(r[2] for r in rows)
    assert values[0] is None and values[6] is None  # s = 2 and zero denominator (s = 8)
    assert values[5] == 0.0
    assert all(isinstance(v, float) for i, v in enumerate(values) if i not in (0, 6))
    # The undefined theta breaks the run: no STAGNATED although the other thetas are >= 0.9 often.
    assert all(decide(r[0], rows[: i + 1]) != STAGNATED for i, r in enumerate(rows))


def test_max_half_steps():
    rows = rows_from([BAD] * 5)  # s = 2..6
    assert decide(5, rows[:4], s_max=6) is None
    assert decide(6, rows, s_max=6) == MAX_HALF_STEPS
    assert decide(1, [], s_max=1) == MAX_HALF_STEPS


def test_priorities():
    # CONVERGED over MAX_HALF_STEPS at the same s.
    assert decide(4, rows_from([BAD, GOOD, GOOD]), s_max=4) == CONVERGED
    # STAGNATED over MAX_HALF_STEPS at the same s.
    assert decide(12, rows_from(stagnating(11)), s_max=12) == STAGNATED
    # CONVERGED after a completed stagnation run, at s_max: the two last steps satisfy both
    # tolerances, so CONVERGED is returned. CONVERGED and STAGNATED cannot hold at the same s:
    # STAGNATED needs eps_E(s) > tol_E, CONVERGED needs eps_E(s) <= tol_E.
    seq = [(1e-12, 2e-8 * 0.95**i) for i in range(11)] + [(1e-12, 0.95e-8), (1e-12, 0.94e-8)]
    rows = rows_from(seq)  # s = 2..14
    assert decide(12, rows[:11]) == STAGNATED
    assert decide(14, rows, s_max=14) == CONVERGED
    for i, row in enumerate(rows):
        prefix = rows[: i + 1]
        stag = decide(row[0], prefix) == STAGNATED
        conv_possible = len(prefix) >= 2 and all(r[1] <= TOL_INF and r[2] <= TOL_E for r in prefix[-2:])
        assert not (stag and conv_possible)


def test_decision_rejects_inconsistent_diagnostics():
    with pytest.raises(ValueError):
        decide(4, rows_from([BAD, GOOD]))
    with pytest.raises(ValueError):
        decide(3, [(3, 1.0, 1.0), (2, 1.0, 1.0)])


def test_decision_does_not_modify_input():
    rows = rows_from([BAD, GOOD, GOOD])
    copy = list(rows)
    decide(4, rows)
    assert rows == copy


# eps_floor and strict mode ----------------------------------------------------------------


@pytest.mark.parametrize("n", [20, 80, 320, 640])
def test_eps_floor_formula(n):
    assert eps_floor(n, n) == 10 * U * (n / math.pi) ** 2
    assert eps_floor(n, 4) == eps_floor(4, n) == eps_floor(n, n)


@pytest.mark.parametrize("n", [80, 320, 640])
def test_strict_parameters(n):
    tol_inf_s, tol_e_s = strict_parameters(n, n)
    assert tol_inf_s == TOL_INF / 100
    assert tol_e_s == max(TOL_E / 100, 10 * eps_floor(n, n))
    assert tol_e_s >= 10 * eps_floor(n, n)  # never rejected by the configuration rule
    if n == 80:
        assert tol_e_s == TOL_E / 100
    else:
        assert tol_e_s == 10 * eps_floor(n, n) > TOL_E / 100


def test_unattainable_tol_e_is_error_input():
    n = 640
    bad = 10 * eps_floor(n, n) * 0.99
    r = run_until_stopped(1.0, 1.0, 1.0, n, n, tol_E=bad)
    assert r.status == ERROR_INPUT and r.s == 0 and not r.converged and r.last_field() is None
    ok = run_until_stopped(1.0, 1.0, 1.0, 20, 20, tol_E=10 * eps_floor(20, 20))
    assert ok.status != ERROR_INPUT


@pytest.mark.parametrize("kwargs", [{"tol_inf": 0.0}, {"tol_E": -1.0}, {"tol_E": math.nan}, {"s_min": 0},
                                    {"s_max": 2.0}, {"s_min": 5, "s_max": 4}, {"tol_inf": True}])
def test_invalid_parameters_are_error_input(kwargs):
    r = run_until_stopped(1.0, 1.0, 1.0, 20, 20, **kwargs)
    assert r.status == ERROR_INPUT and r.s == 0


@pytest.mark.parametrize("args", [(1.0, 1.0, 0.0, 20, 20), (0.0, 1.0, 1.0, 20, 20), (1.0, 1.0, 1.0, 3, 20)])
def test_invalid_physical_input_is_error_input(args):
    r = run_until_stopped(*args)
    assert r.status == ERROR_INPUT and not r.converged


# Physical runs -------------------------------------------------------------------------------


def field_changes_reference(w_prev, w_now, gx, gy, D, q):
    d = w_now - w_prev
    eps_inf = np.max(np.abs(d)) / np.max(np.abs(w_now))
    eps_E = discrete_energy(gx, gy, d, D, q).norm_a / discrete_energy(gx, gy, w_now, D, q).norm_a
    eps_L2 = math.sqrt(np.sum(d * d)) / math.sqrt(np.sum(w_now * w_now))
    return eps_inf, eps_E, eps_L2


@pytest.mark.parametrize("n_x, n_y, q", [(20, 20, 1.0), (20, 40, -1.0), (21, 20, 2.5)])
def test_quantities_match_section_11_1(n_x, n_y, q):
    r = run_until_stopped(1.3, 0.7, q, n_x, n_y)
    assert r.status == CONVERGED
    gx, gy = r.history.record(1).grid_x, r.history.record(1).grid_y
    for c in r.changes:
        w_prev, w_now = r.history.field(c.s - 1), r.history.field(c.s)
        e_inf, e_E, e_L2 = field_changes_reference(w_prev, w_now, gx, gy, 0.7, q)
        assert c.eps_inf == pytest.approx(e_inf, rel=1e-12, abs=0)
        assert c.eps_E == pytest.approx(e_E, rel=1e-12, abs=0)
        assert c.eps_L2 == pytest.approx(e_L2, rel=1e-12, abs=0)
        assert c.k == c.s // 2
    expected = thetas(c.eps_E for c in r.changes)
    assert [c.theta for c in r.changes] == expected
    assert r.changes[0].theta is None


@pytest.mark.parametrize("n", [20, 80, 160, 320])
def test_normal_and_strict_runs(n):
    normal = run_until_stopped(1.0, 1.0, 1.0, n, n)
    tol_inf_s, tol_e_s = strict_parameters(n, n)
    strict = run_until_stopped(1.0, 1.0, 1.0, n, n, tol_inf=tol_inf_s, tol_E=tol_e_s)
    for r, (ti, te) in ((normal, (TOL_INF, TOL_E)), (strict, (tol_inf_s, tol_e_s))):
        assert r.status == CONVERGED and r.converged
        assert r.s >= 3 and r.k == r.s // 2
        last_two = r.changes[-2:]
        assert all(c.eps_inf <= ti and c.eps_E <= te and c.satisfied for c in last_two)
        assert not (len(r.changes) >= 3 and r.changes[-3].satisfied and r.changes[-2].satisfied)  # stops at once
        assert [c.s for c in r.energy_checks] == list(range(1, r.s + 1))
        assert r.energy_checks[0].status == NOT_APPLICABLE and r.energy_checks[1].status == PASS
        np.testing.assert_array_equal(r.last_field(), r.history.field(r.s))
        assert r.parameters.tol_inf == ti and r.parameters.tol_E == te
    assert strict.s >= normal.s
    # Same half-steps up to the normal stop: the controller does not alter them.
    for s in range(1, normal.s + 1):
        np.testing.assert_array_equal(strict.history.field(s), normal.history.field(s))


def test_unresolved_energy_status_does_not_prevent_convergence():
    r = run_until_stopped(1.0, 1.0, 1.0, 20, 20)
    assert r.status == CONVERGED
    assert r.energy_checks[-1].status == UNRESOLVED


def test_diagnostic_grid_640_eps_floor():
    n = 640
    tol_inf_s, tol_e_s = strict_parameters(n, n)
    r = run_until_stopped(1.0, 1.0, 1.0, n, n, tol_inf=tol_inf_s, tol_E=tol_e_s)
    assert r.parameters.eps_floor == eps_floor(n, n)
    assert r.status == CONVERGED


def test_history_matches_run_halfsteps():
    r = run_until_stopped(1.0, 1.0, 1.0, 20, 30)
    fixed = run_halfsteps(1.0, 1.0, 1.0, 20, 30, r.s)
    assert len(fixed) == r.s
    for s in range(1, r.s + 1):
        np.testing.assert_array_equal(fixed.field(s), r.history.field(s))


def test_tail_estimate_only_with_three_defined_thetas():
    r = run_until_stopped(1.0, 1.0, 1.0, 20, 20)
    assert TAIL_ESTIMATE_NOTE == r.tail_estimate_note and "не доказанная" in TAIL_ESTIMATE_NOTE
    for i, c in enumerate(r.changes):
        last = [x.theta for x in r.changes[max(0, i - 2): i + 1]]
        if len(last) < 3 or any(t is None for t in last):
            assert c.tail_estimate is None
        else:
            median = sorted(last)[1]
            assert median < 1.0
            assert c.tail_estimate == pytest.approx(c.eps_E * median / (1 - median), rel=1e-14, abs=0)


# Controlled sequences through the real loop ------------------------------------------------


def test_controller_acts_on_stagnated_decision(monkeypatch):
    values = iter(stagnating(30))
    monkeypatch.setattr(ctl, "_field_changes", lambda prev, now: (*next(values), 0.0))
    r = run_until_stopped(1.0, 1.0, 1.0, 20, 20)
    assert r.status == STAGNATED and r.s == 12 and not r.converged
    assert len(r.history) == 12 and len(r.changes) == 11


def test_controller_max_half_steps(monkeypatch):
    monkeypatch.setattr(ctl, "_field_changes", lambda prev, now: (1e-3, 1e-3, 1e-3))  # theta = 1, eps_E > tol_E ...
    r = run_until_stopped(1.0, 1.0, 1.0, 20, 20, s_max=8)
    assert r.status == MAX_HALF_STEPS and r.s == 8
    r = run_until_stopped(1.0, 1.0, 1.0, 20, 20, s_max=20)
    assert r.status == STAGNATED and r.s == 12  # ... so with a larger s_max it stagnates first


def test_controller_does_not_stop_after_one_good_step(monkeypatch):
    seq = iter([BAD, GOOD, BAD, GOOD, GOOD])
    monkeypatch.setattr(ctl, "_field_changes", lambda prev, now: (*next(seq), 0.0))
    r = run_until_stopped(1.0, 1.0, 1.0, 20, 20)
    assert r.status == CONVERGED and r.s == 6


# Error exits ----------------------------------------------------------------------------------


def test_solver_error_keeps_completed_halfsteps(monkeypatch):
    original = it.solve_halfstep
    calls = []

    def failing(*args):
        calls.append(1)
        if len(calls) == 3:  # s = 4
            raise LinearSolveError(ERROR_LINEAR_SOLVE, "forced in test")
        return original(*args)

    monkeypatch.setattr(it, "solve_halfstep", failing)
    r = run_until_stopped(1.0, 1.0, 1.0, 20, 20)
    assert r.status == ERROR_LINEAR_SOLVE and not r.converged
    assert r.s == 3 and isinstance(r.error, LinearSolveError)
    assert len(r.changes) == 2 and len(r.energy_checks) == 3


def test_energy_test_error_is_kept(monkeypatch):
    def failing(previous, current):
        raise EnergyError(ERROR_ENERGY_TEST, "forced in test", check=None)

    monkeypatch.setattr(ctl, "check_pair", failing)
    r = run_until_stopped(1.0, 1.0, 1.0, 20, 20)
    assert r.status == ERROR_ENERGY_TEST and r.s == 2 and not r.converged


def test_degenerate_coefficients_are_kept(monkeypatch):
    import kkvim.first_halfstep as fh

    original = fh.compute_coefficients
    calls = []

    def failing(*args):
        calls.append(1)
        if len(calls) == 2:
            raise CoefficientError(ERROR_DEGENERATE, "forced in test")
        return original(*args)

    monkeypatch.setattr(fh, "compute_coefficients", failing)
    r = run_until_stopped(1.0, 1.0, 1.0, 20, 20)
    assert r.status == ERROR_DEGENERATE and r.s == 1 and r.error.code == ERROR_DEGENERATE


def test_run_halfsteps_unchanged():
    history = run_halfsteps(1.0, 1.0, 1.0, 20, 20, 15)
    assert len(history) == 15  # exactly count, no stopping
