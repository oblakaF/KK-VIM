"""Own tests of kkvim.iteration: fixed number of KK-VIM half-steps (R2, step 6).

Checks follow docs/MATH_SPEC.md 4.5, docs/NUMERICAL_SPEC.md 5.2-5.3, 9.1 (I-4),
10.1 (ChV-11 condition w(s-1) in M_s), D-030 and table 18.1 (A-5g, A-6, A-8, A-12).
No comparison with kkvim_reference and no scratch numbers.
"""

import math
from types import SimpleNamespace

import numpy as np
import pytest

import kkvim.first_halfstep as fh
import kkvim.iteration as it
from kkvim.coefficients import ERROR_DEGENERATE, CoefficientError, compute_coefficients
from kkvim.first_halfstep import HalfStepError, NotANodeError, first_halfstep, node_value
from kkvim.iteration import HalfStepHistory, HalfStepRecord, as_record, next_halfstep, run_halfsteps
from kkvim.linear_solver import ERROR_LINEAR_SOLVE, ERROR_NONFINITE, ETA_MAX, IMAG_MAX, LinearSolveError, apply_k
from kkvim.second_halfstep import second_halfstep

U = 2.0**-53
A5G_TOL = 1e-12  # table 18.1, A-5g
A8_TOL = 1e-12  # table 18.1, A-8
GRID_PAIRS = [(20, 20), (20, 40), (40, 20), (21, 20), (20, 21), (7, 9)]


def run(count=6, a=1.0, D=1.0, q=1.0, n_x=20, n_y=20):
    return run_halfsteps(a, D, q, n_x, n_y, count)


# Generic next half-step against the qualified s = 1, s = 2 --------------------------


def test_next_halfstep_reproduces_second_halfstep():
    first = first_halfstep(1.0, 1.0, 1.0, 20, 30)
    expected = second_halfstep(first)
    actual = next_halfstep(first)
    assert (actual.s, actual.k, actual.direction) == (2, 1, "X")
    scale = np.max(np.abs(expected.w2))
    assert np.max(np.abs(actual.field() - expected.w2)) <= A5G_TOL * scale
    np.testing.assert_allclose(actual.raw, expected.x2_raw, rtol=1e-14, atol=0)
    np.testing.assert_allclose(actual.x, expected.x2_normalized, rtol=1e-14, atol=0)
    np.testing.assert_allclose(actual.y, expected.y2_scaled, rtol=1e-14, atol=0)
    assert actual.c == pytest.approx(expected.c2, rel=1e-14, abs=0)
    for name in "ABCF":
        assert getattr(actual.coefficients, name) == pytest.approx(getattr(expected.coefficients, name), rel=1e-14, abs=0)
    assert actual.solver_diagnostics.refinements == expected.solver_diagnostics.refinements
    assert actual.solver_diagnostics.eta == pytest.approx(expected.solver_diagnostics.eta, rel=1e-6, abs=1e-20)


@pytest.mark.parametrize("count", [1, 2])
def test_short_runs_reproduce_existing_results(count):
    history = run(count=count, n_x=20, n_y=30)
    first = first_halfstep(1.0, 1.0, 1.0, 20, 30)
    assert len(history) == count
    np.testing.assert_array_equal(history.field(1), first.w1)
    if count == 2:
        second = second_halfstep(first)
        assert np.max(np.abs(history.field(2) - second.w2)) <= A5G_TOL * np.max(np.abs(second.w2))


def test_as_record_of_existing_results():
    first = first_halfstep(1.0, 1.0, 1.0, 20, 20)
    second = second_halfstep(first)
    r1, r2 = as_record(first), as_record(second)
    np.testing.assert_array_equal(r1.field(), first.w1)
    np.testing.assert_array_equal(r2.field(), second.w2)
    with pytest.raises(TypeError):
        as_record(SimpleNamespace(grid_x=first.grid_x, grid_y=first.grid_y, w1=first.w1))


# Alternation, counters, grids ---------------------------------------------------------


@pytest.mark.parametrize("count", [3, 6])
def test_alternation_and_counters(count):
    history = run(count=count)
    assert [r.s for r in history.records] == list(range(1, count + 1))
    assert [r.k for r in history.records] == [s // 2 for s in range(1, count + 1)]  # D-030
    assert [r.direction for r in history.records] == ["Y" if s % 2 else "X" for s in range(1, count + 1)]


@pytest.mark.parametrize("n_x, n_y", [(20, 40), (21, 20)])
def test_coefficients_on_fixed_grid_and_solve_on_other(monkeypatch, n_x, n_y):
    calls = []
    original_coefficients, original_solve = fh.compute_coefficients, fh.solve_k

    def spy_coefficients(grid, f, q):
        calls.append(("coefficients", grid, np.array(f)))
        return original_coefficients(grid, f, q)

    def spy_solve(grid, A, B, C, D, g):
        calls.append(("solve", grid, np.array(g)))
        return original_solve(grid, A, B, C, D, g)

    monkeypatch.setattr(fh, "compute_coefficients", spy_coefficients)
    monkeypatch.setattr(fh, "solve_k", spy_solve)
    history = run(count=6, n_x=n_x, n_y=n_y)
    grid_x, grid_y = history.record(1).grid_x, history.record(1).grid_y
    assert len(calls) == 12
    for s in range(1, 7):
        (kind_c, coeff_grid, factor), (kind_s, solve_grid, rhs) = calls[2 * s - 2], calls[2 * s - 1]
        assert (kind_c, kind_s) == ("coefficients", "solve")
        if s % 2:  # Y-half-step
            assert coeff_grid is grid_x and solve_grid is grid_y
        else:  # X-half-step
            assert coeff_grid is grid_y and solve_grid is grid_x
        assert np.all(rhs == history.record(s).coefficients.F)
        if s >= 2:  # fixed factor = normalized computed factor of s - 1 (ChV-11)
            np.testing.assert_array_equal(factor, history.record(s - 1).computed)


def test_fixed_factor_is_not_renormalized(monkeypatch):
    seen = []
    original = it.solve_halfstep

    def spy(fixed_grid, fixed_factor, solve_grid, q, D):
        seen.append(fixed_factor)
        return original(fixed_grid, fixed_factor, solve_grid, q, D)

    monkeypatch.setattr(it, "solve_halfstep", spy)
    history = run(count=6)
    assert len(seen) == 5
    for s in range(2, 7):
        previous = history.record(s - 1)
        assert seen[s - 2] is previous.computed  # the very stored array
        # w(s-1) lies in M_s: it is (fixed factor of s) (x) something, i.e. the field factor of s - 1.
        current = history.record(s)
        if s % 2:
            np.testing.assert_array_equal(current.x, previous.x * current.c)
        else:
            np.testing.assert_array_equal(current.y, previous.y * current.c)


# Normalization, fields -----------------------------------------------------------------


@pytest.mark.parametrize("n_x, n_y", GRID_PAIRS)
def test_normalization_and_fields(n_x, n_y):
    history = run(count=6, n_x=n_x, n_y=n_y)
    for r in history.records:
        computed, fixed_len = (r.y, n_x - 1) if r.direction == "Y" else (r.x, n_y - 1)
        assert math.isfinite(r.c) and r.c != 0.0
        assert np.max(np.abs(computed)) == pytest.approx(1.0, abs=2 * U)
        assert np.sum(computed) > 0.0
        np.testing.assert_array_equal(r.raw / r.c, computed)
        w = r.field()
        assert w.shape == (n_x + 1, n_y + 1)
        assert np.all(w[0, :] == 0.0) and np.all(w[-1, :] == 0.0)
        assert np.all(w[:, 0] == 0.0) and np.all(w[:, -1] == 0.0)
        np.testing.assert_array_equal(w[1:-1, 1:-1], np.outer(r.x, r.y))
        # The physical field does not depend on the normalization (<= a few u per node).
        if r.s >= 2:
            fixed = history.record(r.s - 1).computed
            raw = (np.outer(r.grid_x.with_boundary(fixed), r.grid_y.with_boundary(r.raw)) if r.direction == "Y"
                   else np.outer(r.grid_x.with_boundary(r.raw), r.grid_y.with_boundary(fixed)))
            assert np.all(np.abs(w - raw) <= 4 * U * np.abs(raw))


def test_fields_are_separate_and_stable():
    history = run(count=6)
    fields = [history.field(s) for s in range(1, 7)]
    for s in range(1, 7):
        np.testing.assert_array_equal(history.field(s), fields[s - 1])  # unchanged between calls
    for s in range(1, 6):
        assert not np.array_equal(fields[s - 1], fields[s])
    first, last = history.last_two_fields()
    np.testing.assert_array_equal(first, fields[4])
    np.testing.assert_array_equal(last, fields[5])
    record = history.record(3)
    with pytest.raises(ValueError):
        record.x[0] = 0.0  # stored factors are read-only
    with pytest.raises(IndexError):
        history.record(7)
    with pytest.raises(ValueError):
        run(count=1).last_two_fields()


@pytest.mark.parametrize("n_x, n_y", GRID_PAIRS)
def test_a6_every_halfstep(n_x, n_y):
    history = run(count=6, a=2.0, D=0.8, q=1.3, n_x=n_x, n_y=n_y)
    for r in history.records:
        d = r.solver_diagnostics
        assert d.eta <= ETA_MAX and d.imag_ratio <= IMAG_MAX and d.refinements <= 1
        c = r.coefficients
        grid = r.grid_y if r.direction == "Y" else r.grid_x
        residual = apply_k(grid, c.A, c.B, c.C, r.D, r.raw) - c.F
        assert np.max(np.abs(residual)) <= 1e-9 * abs(c.F)
        if r.s >= 2:  # coefficients of the stored fixed factor on its own grid
            fixed_grid = r.grid_x if r.direction == "Y" else r.grid_y
            assert c == compute_coefficients(fixed_grid, history.record(r.s - 1).computed, r.q)


@pytest.mark.parametrize("n", [20, 21, 80])
def test_a8_every_halfstep(n):
    history = run(count=6, n_x=n, n_y=n)
    for s in range(1, 7):
        w = history.field(s)
        scale = np.max(np.abs(w))
        assert np.max(np.abs(w - w[::-1, :])) <= A8_TOL * scale
        assert np.max(np.abs(w - w[:, ::-1])) <= A8_TOL * scale


def test_negative_load_and_scaling():
    base = run(count=6, n_x=20, n_y=30)
    neg = run(count=6, q=-1.0, n_x=20, n_y=30)
    by_q = run(count=6, q=3.7, n_x=20, n_y=30)
    by_d = run(count=6, D=2.0, n_x=20, n_y=30)
    for s in range(1, 7):
        w = base.field(s)
        assert np.all(w[1:-1, 1:-1] > 0.0) and np.all(neg.field(s)[1:-1, 1:-1] < 0.0)
        np.testing.assert_allclose(neg.field(s), -w, rtol=1e-15, atol=0)
        np.testing.assert_allclose(by_q.field(s), 3.7 * w, rtol=1e-12, atol=0)
        np.testing.assert_allclose(by_d.field(s), w / 2.0, rtol=1e-12, atol=0)


# node_value ---------------------------------------------------------------------------


def test_node_value_uses_the_field_of_its_own_step():
    first = first_halfstep(1.0, 1.0, 1.0, 20, 30)
    second = second_halfstep(first)
    assert node_value(first, "1/2", "1/2") == first.w1[10, 15]
    assert node_value(second, "1/2", "1/2") == second.w2[10, 15]
    assert node_value(second, "1/2", "1/2") != first.w1[10, 15]
    history = run(count=6, n_x=20, n_y=30)
    for s in range(1, 7):
        assert node_value(history.record(s), "1/2", "1/2") == history.field(s)[10, 15]


def test_node_value_rejects_unknown_types():
    first = first_halfstep(1.0, 1.0, 1.0, 20, 20)
    fake = SimpleNamespace(grid_x=first.grid_x, grid_y=first.grid_y, w1=first.w1, w2=first.w1)
    for bad in (fake, object(), run(count=2)):
        with pytest.raises(TypeError):
            node_value(bad, "1/2", "1/2")


@pytest.mark.parametrize("n_x, n_y", [(21, 20), (20, 21)])
def test_node_value_odd_grid_has_no_center(n_x, n_y):
    history = run(count=4, n_x=n_x, n_y=n_y)
    for s in range(1, 5):
        with pytest.raises(NotANodeError):
            node_value(history.record(s), "1/2", "1/2")


# Errors -------------------------------------------------------------------------------


@pytest.mark.parametrize("count", [0, -1, 2.0, True, "3"])
def test_invalid_count(count):
    with pytest.raises((ValueError, TypeError)):
        run(count=count)


def test_next_halfstep_rejects_unknown_types():
    with pytest.raises(TypeError):
        next_halfstep(object())


def test_errors_are_not_masked(monkeypatch):
    original = it.solve_halfstep
    calls = []

    def failing_at_s4(fixed_grid, fixed_factor, solve_grid, q, D):
        calls.append(1)
        if len(calls) == 3:  # s = 4
            raise LinearSolveError(ERROR_LINEAR_SOLVE, "forced in test")
        return original(fixed_grid, fixed_factor, solve_grid, q, D)

    monkeypatch.setattr(it, "solve_halfstep", failing_at_s4)
    result = None
    with pytest.raises(LinearSolveError) as info:
        result = run(count=6)
    assert info.value.code == ERROR_LINEAR_SOLVE and result is None and len(calls) == 3


def test_coefficient_and_normalization_errors_keep_codes(monkeypatch):
    first = first_halfstep(1.0, 1.0, 1.0, 20, 20)

    def degenerate(grid, f, q):
        raise CoefficientError(ERROR_DEGENERATE, "forced in test")

    with monkeypatch.context() as m:
        solver_calls = []
        m.setattr(fh, "compute_coefficients", degenerate)
        m.setattr(fh, "solve_k", lambda *args: solver_calls.append(args))
        with pytest.raises(CoefficientError) as info:
            next_halfstep(first)
        assert info.value.code == ERROR_DEGENERATE and solver_calls == []

    for raw, code in ((np.r_[1.0, -1.0, np.zeros(17)], ERROR_DEGENERATE), (np.r_[np.nan, np.ones(18)], ERROR_NONFINITE)):
        fake = SimpleNamespace(raw=raw, coefficients=None, solver_diagnostics=None)
        with monkeypatch.context() as m:
            m.setattr(it, "solve_halfstep", lambda *args, fake=fake: fake)
            with pytest.raises(HalfStepError) as info:
                next_halfstep(first)
            assert info.value.code == code


def test_history_type():
    history = run(count=3)
    assert isinstance(history, HalfStepHistory)
    assert all(isinstance(r, HalfStepRecord) for r in history.records)
