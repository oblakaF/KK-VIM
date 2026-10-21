"""Own tests of kkvim.rectangle_run: KK-VIM on a x b until the stop (R3, stage B).

Checks follow docs/R3_NUMERICAL_SPEC.md 4 (R2 stopping unchanged), 1.2, table 9.1
(A-5d counters exact, A-6 eta <= 1e-12, A-9 <= 1e-11 relative to max |w_bar|) and
NUMERICAL_SPEC.md 10.4, 11.3, 11.5. r = 1 must reproduce the R2 controller exactly
(D-041). Energy errors stay visible; UNRESOLVED is never PASS. QA qualifies R3.
"""

import dataclasses
import math
import sys
from pathlib import Path

import numpy as np
import pytest

import kkvim.rectangle_run as rr
from kkvim.coefficients import ERROR_INPUT, compute_coefficients
from kkvim.controller import CONVERGED, MAX_HALF_STEPS, STAGNATED, run_until_stopped
from kkvim.energy import ERROR_ENERGY_TEST, FAIL, NOT_APPLICABLE, PASS, UNRESOLVED, EnergyError
from kkvim.first_halfstep import NOT_A_NODE_MESSAGE
from kkvim.linear_solver import ERROR_LINEAR_SOLVE, ERROR_NONFINITE, LinearSolveError
from kkvim.physical import PhysicalInputError, run_physical
from kkvim.rectangle_run import halfstep_report, run_rectangle_physical, run_rectangle_until_stopped

sys.path.insert(0, str(Path(__file__).parent))
from test_rectangle import A9_GRIDS, BASE, VARIANTS, records_equal, start_y_fields  # noqa: E402

A9_TOL = 1e-11
ETA_MAX = 1e-12
D, Q = 3.1, 2.3


def results_equal(r, s):
    assert (r.status, r.parameters, r.changes, r.energy_checks) == (s.status, s.parameters, s.changes, s.energy_checks)
    assert len(r.history) == len(s.history)
    for x, y in zip(r.history.records, s.history.records, strict=True):
        records_equal(x, y)


# r = 1: the R2 controller is reproduced exactly -----------------------------------------

@pytest.mark.parametrize("n_x, n_y, options", [(20, 20, {}), (40, 24, {}), (32, 32, {"s_max": 4, "tol_inf": 1e-20}),
                                               (80, 80, {"tol_E": 1e-6, "s_min": 5})])
def test_square_reproduces_r2_controller(n_x, n_y, options):
    old = run_until_stopped(1.3, D, Q, n_x, n_y, **options)
    new = run_rectangle_until_stopped(1.3, 1.3, D, Q, n_x, n_y, **options)
    results_equal(old, new)


def test_square_physical_reproduces_r2_physical():
    old = run_physical(**BASE, n_x=60, n_y=60)
    new = run_rectangle_physical(**BASE, b=BASE["a"], n_x=60, n_y=60)
    assert (new.status, new.D) == (old.status, old.D)
    results_equal(old.run, new.run)
    np.testing.assert_array_equal(new.dimensionless_field(), old.dimensionless_field())


# rectangles to the stop ------------------------------------------------------------------

RECTANGLES = [(1.0, 0.2, 160, 32), (1.0, 1 / 3, 96, 32), (1.0, 0.5, 80, 40), (1.0, 2 / 3, 60, 40), (1.0, 1.4, 40, 56),
              (1.0, 1.5, 40, 60), (1.0, 2.0, 40, 80), (1.0, 3.0, 32, 96), (1.0, 5.0, 32, 160), (2.0, 3.0, 20, 36)]


@pytest.mark.parametrize("a, b, n_x, n_y", RECTANGLES)
def test_rectangle_runs_to_the_stop(a, b, n_x, n_y):
    run = run_rectangle_until_stopped(a, b, D, Q, n_x, n_y)
    assert run.status == CONVERGED and run.error is None
    assert (run.s, run.k) == (len(run.history), run.s // 2)
    assert [c.s for c in run.changes] == list(range(2, run.s + 1))
    assert [c.s for c in run.energy_checks] == list(range(1, run.s + 1))
    assert run.changes[-1].satisfied and run.changes[-2].satisfied and run.s >= run.parameters.s_min
    assert all(r.solver_diagnostics.eta <= ETA_MAX for r in run.history.records)
    assert all(c.status != FAIL for c in run.energy_checks)
    assert run.history.records[-1].grid_y.length == b and run.last_field().shape == (n_x + 1, n_y + 1)


def test_max_half_steps_and_configuration_errors():
    run = run_rectangle_until_stopped(1.0, 2.0, D, Q, 20, 40, s_max=4, tol_inf=1e-20)
    assert run.status == MAX_HALF_STEPS and run.s == 4
    bad = run_rectangle_until_stopped(1.0, 2.0, D, Q, 20, 40, tol_E=1e-20)
    assert bad.status == ERROR_INPUT and bad.parameters is None and bad.s == 0


@pytest.mark.parametrize("a, b, D_", [(1.0, 6.0, D), (1.0, math.nan, D), (0.0, 1.0, D), (1.0, 2.0, -1.0)])
def test_invalid_sides_or_d_give_error_input_without_field(a, b, D_):
    run = run_rectangle_until_stopped(a, b, D_, Q, 20, 40)
    assert run.status == ERROR_INPUT and run.s == 0 and run.last_field() is None and run.parameters is not None


def test_stagnation_is_kept(monkeypatch):
    eps = iter([(1e-3, 1e-3 * 0.95**k, 1e-3) for k in range(40)])
    monkeypatch.setattr(rr, "_field_changes", lambda previous, current: next(eps))
    run = run_rectangle_until_stopped(1.0, 2.0, D, Q, 20, 40)
    assert run.status == STAGNATED and run.s == 12  # theta >= 0.9 at s = 3 .. 12


def test_linear_solve_error_keeps_completed_halfsteps(monkeypatch):
    real = rr.next_halfstep

    def failing(previous):
        if previous.s == 2:
            raise LinearSolveError(ERROR_LINEAR_SOLVE, "forced")
        return real(previous)

    monkeypatch.setattr(rr, "next_halfstep", failing)
    run = run_rectangle_until_stopped(1.0, 2.0, D, Q, 20, 40)
    assert run.status == ERROR_LINEAR_SOLVE and run.s == 2 and isinstance(run.error, LinearSolveError)


def test_nonfinite_changes_give_error_nonfinite(monkeypatch):
    monkeypatch.setattr(rr, "_field_changes", lambda previous, current: (math.nan, 1e-3, 1e-3))
    run = run_rectangle_until_stopped(1.0, 2.0, D, Q, 20, 40)
    assert run.status == ERROR_NONFINITE and run.s == 2 and len(run.energy_checks) == 2 and run.changes == ()


# energy test: errors are not hidden, UNRESOLVED is not PASS --------------------------------

def test_energy_statuses_follow_the_resolution_rule():
    run = run_rectangle_until_stopped(1.0, 2.0, D, Q, 40, 80)
    statuses = [c.status for c in run.energy_checks]
    assert statuses[0] == NOT_APPLICABLE and UNRESOLVED in statuses and PASS in statuses
    for check in run.energy_checks[1:]:
        assert (check.status == UNRESOLVED) == (check.Q < 10.0 * check.tau)
        assert (check.status == PASS) == (check.Q >= 10.0 * check.tau and check.e1 and check.e2)
    report = halfstep_report(run_rectangle_physical(**BASE, b=2.0, n_x=40, n_y=80))
    assert [row.energy.status for row in report.halfsteps] == statuses


def test_resolved_e1_failure_is_error_energy_test(monkeypatch):
    real = rr.check_pair

    def failing(previous, current):
        check = real(previous, current)
        if current.s == 3:
            check = dataclasses.replace(check, status=FAIL, e1=False)
            raise EnergyError(ERROR_ENERGY_TEST, "forced E1 failure", check)
        return check

    monkeypatch.setattr(rr, "check_pair", failing)
    run = run_rectangle_until_stopped(1.0, 2.0, D, Q, 40, 80)
    assert run.status == ERROR_ENERGY_TEST and run.s == 3 and isinstance(run.error, EnergyError)
    assert run.energy_checks[-1].status == FAIL and run.energy_checks[-1].s == 3


def test_diagnostic_e2_failure_stays_visible(monkeypatch):
    real = rr.check_pair

    def e2_fail(previous, current):
        check = real(previous, current)
        return dataclasses.replace(check, status=FAIL, e2=False) if current.s == 2 else check

    monkeypatch.setattr(rr, "check_pair", e2_fail)
    run = run_rectangle_until_stopped(1.0, 2.0, D, Q, 40, 80)
    assert run.status == CONVERGED and run.energy_checks[1].status == FAIL


def test_nonnegative_energy_is_not_hidden(monkeypatch):
    def degenerate(record):
        raise EnergyError("ERROR_DEGENERATE", "Pi_h >= 0")

    monkeypatch.setattr(rr, "first_check", degenerate)
    run = run_rectangle_until_stopped(1.0, 2.0, D, Q, 20, 40)
    assert run.status == "ERROR_DEGENERATE" and run.s == 1 and run.energy_checks == ()


# second start (ChV-R1) and scaling at the stop (A-9) ---------------------------------------

@pytest.mark.parametrize("a, b, n_x, n_y", [(1.0, 2.0, 20, 40), (1.0, 0.2, 100, 20), (2.0, 3.0, 20, 36)])
def test_swapped_plate_is_the_second_start_to_the_stop(a, b, n_x, n_y):
    swapped = run_rectangle_until_stopped(b, a, D, Q, n_y, n_x)
    assert swapped.status == CONVERGED
    for record, w in zip(swapped.history.records, start_y_fields(a, b, n_x, n_y, swapped.s), strict=True):
        np.testing.assert_array_equal(record.field().T, w)


@pytest.mark.parametrize("r", list(A9_GRIDS), ids=lambda r: f"r={r:.4g}")
@pytest.mark.parametrize("family", ["I", "II"])
def test_a9_scaling_to_the_stop(r, family):
    n_x, n_y = (320, 320) if family == "I" else A9_GRIDS[r]
    base = run_rectangle_physical(**BASE, b=r, n_x=n_x, n_y=n_y)
    assert base.status == CONVERGED
    for name, change in VARIANTS.items():
        change = dict(change)
        b = r * change.pop("b_factor", 1.0)
        run = run_rectangle_physical(**{**BASE, **change}, b=b, n_x=n_x, n_y=n_y)
        assert run.status == CONVERGED and run.run.s == base.run.s, name
        w_ref = base.dimensionless_field()
        deviation = np.max(np.abs(run.dimensionless_field() - w_ref)) / np.max(np.abs(w_ref))
        assert deviation <= A9_TOL, (name, deviation)


# physical input and the report of NUMERICAL_SPEC 11.5 ------------------------------------

@pytest.mark.parametrize("change", [{"b": 5.1}, {"b": 0.0}, {"b": "2"}, {"nu": 1.0}, {"q": 0.0}, {"n_y": 3}])
def test_physical_invalid_input(change):
    result = run_rectangle_physical(**{**BASE, "b": 2.0, "n_x": 20, "n_y": 40, **change})
    assert result.status == ERROR_INPUT and isinstance(result.error, PhysicalInputError)
    assert result.run is None and result.D is None and result.dimensionless_field() is None
    report = halfstep_report(result)
    assert report.halfsteps == () and report.status == ERROR_INPUT and report.parameters is None


def test_report_holds_the_quantities_of_11_5():
    result = run_rectangle_physical(**BASE, b=2.0, n_x=40, n_y=80, tol_inf=1e-10)
    report = halfstep_report(result, control_points=(("1/2", "1/2"), ("1/4", "3/8"), ("1/3", "1/2")))
    run = result.run
    assert report.status == CONVERGED and report.input == result.input and report.D == result.D
    assert report.parameters == run.parameters and set(report.versions) == {"python", "kkvim", "numpy", "scipy"}
    assert [row.s for row in report.halfsteps] == list(range(1, run.s + 1))
    for row, record in zip(report.halfsteps, run.history.records, strict=True):
        assert (row.k, row.direction) == (record.s // 2, record.direction)
        if row.direction == "Y":
            assert (row.solve_axis, row.solve_grid, row.fixed_grid) == ("y", (2.0, 80), (1.0, 40))
        else:
            assert (row.solve_axis, row.solve_grid, row.fixed_grid) == ("x", (1.0, 40), (2.0, 80))
        c = row.coefficients
        assert (row.beta, row.gamma, row.chi) == (c.B / c.A, c.C / c.A, c.F**2 / c.A)
        assert row.c == record.c and row.eta == record.solver_diagnostics.eta
        np.testing.assert_array_equal(row.x[1:-1], record.x)
        np.testing.assert_array_equal(row.y[1:-1], record.y)
        assert row.P == row.energy.P_now and row.P < 0.0 and row.norm_a > 0.0
        assert (row.changes is None) == (row.s == 1)
        w_bar = record.field() * result.D / (BASE["q"] * BASE["a"] ** 4)
        assert row.control_values[0][1] == w_bar[20, 40] and row.control_values[1][1] == w_bar[10, 30]
        assert row.control_values[2][1] == NOT_A_NODE_MESSAGE
    first, second = report.halfsteps[:2]
    assert first.coefficients_before_normalization is None
    previous = run.history.records[0]
    before = compute_coefficients(previous.grid_y, previous.raw, BASE["q"])
    assert second.coefficients_before_normalization == before  # Y1 before its normalization
    assert math.isclose(before.A, second.coefficients.A * previous.c**2, rel_tol=1e-14)
    assert report.halfsteps[-1].control_values[0][1] == result.dimensionless_field()[20, 40]
