"""Own tests of kkvim.second_halfstep: half-step s = 2 and the field w2 (R2, step 5).

Checks follow docs/NUMERICAL_SPEC.md 5.3, 8, 9.1 (I-4), D-030 and table 18.1
(A-5g with 1e-12 for s = 2, A-6, A-8 with 1e-12). No comparison with kkvim_reference.
"""

import math

import numpy as np
import pytest

import kkvim.first_halfstep as fh
import kkvim.second_halfstep as sh
from kkvim.coefficients import ERROR_DEGENERATE, CoefficientError, compute_coefficients
from kkvim.first_halfstep import HalfStepError, first_halfstep, solve_halfstep
from kkvim.linear_solver import ERROR_LINEAR_SOLVE, ERROR_NONFINITE, ETA_MAX, IMAG_MAX, LinearSolveError, apply_k
from kkvim.second_halfstep import SecondHalfStepResult, second_halfstep

U = 2.0**-53
A5G_TOL = 1e-12  # table 18.1, A-5g
A8_TOL = 1e-12  # table 18.1, A-8
GRID_PAIRS = [(20, 20), (20, 40), (40, 20), (21, 20), (20, 21), (7, 9)]


def both(a=1.0, D=1.0, q=1.0, n_x=20, n_y=20):
    first = first_halfstep(a, D, q, n_x, n_y)
    return first, second_halfstep(first)


@pytest.mark.parametrize("n_x, n_y", GRID_PAIRS)
def test_x2_from_normalized_y1_and_counters(n_x, n_y):
    first, r = both(n_x=n_x, n_y=n_y)
    assert isinstance(r, SecondHalfStepResult)
    assert (r.s, r.k) == (2, 1) and r.k == r.s // 2  # D-030
    assert r.first is first and r.w1 is first.w1
    assert r.grid_x is first.grid_x and r.grid_y is first.grid_y
    # Coefficients of the normalized Y1 on grid_y (not Y1_raw, not X1_scaled).
    assert r.coefficients == compute_coefficients(first.grid_y, first.y1_normalized, first.q)
    assert r.coefficients != compute_coefficients(first.grid_y, first.y1_raw, first.q)
    assert r.x2_raw.shape == (n_x - 1,) and r.x2_normalized.shape == (n_x - 1,)
    assert r.y2_scaled.shape == (n_y - 1,)


def test_grid_order_and_rhs(monkeypatch):
    first = first_halfstep(1.0, 1.0, 1.0, 20, 40)
    seen = {}
    original_coefficients, original_solve = fh.compute_coefficients, fh.solve_k

    def spy_coefficients(grid, f, q):
        seen["coefficients_grid"], seen["factor"] = grid, np.array(f)
        return original_coefficients(grid, f, q)

    def spy_solve(grid, A, B, C, D, g):
        seen["solve_grid"], seen["rhs"] = grid, np.array(g)
        return original_solve(grid, A, B, C, D, g)

    monkeypatch.setattr(fh, "compute_coefficients", spy_coefficients)
    monkeypatch.setattr(fh, "solve_k", spy_solve)
    r = second_halfstep(first)
    assert seen["coefficients_grid"] is first.grid_y  # coefficients on y
    assert seen["solve_grid"] is first.grid_x  # system for X2 on x
    np.testing.assert_array_equal(seen["factor"], first.y1_normalized)
    assert seen["rhs"].shape == (first.grid_x.n - 1,)
    assert np.all(seen["rhs"] == r.coefficients.F)  # F_h in every interior node of x


def test_stored_d_and_q_are_used(monkeypatch):
    first = first_halfstep(1.5, 0.7, -2.5, 20, 30)
    seen = []
    original = sh.solve_halfstep

    def spy(fixed_grid, fixed_factor, solve_grid, q, D):
        seen.append((q, D))
        return original(fixed_grid, fixed_factor, solve_grid, q, D)

    monkeypatch.setattr(sh, "solve_halfstep", spy)
    second_halfstep(first)
    assert seen == [(-2.5, 0.7)]


@pytest.mark.parametrize("n_x, n_y", GRID_PAIRS)
def test_x2_solves_discrete_system(n_x, n_y):
    first, r = both(a=2.0, D=0.8, q=1.3, n_x=n_x, n_y=n_y)
    c = r.coefficients
    residual = apply_k(r.grid_x, c.A, c.B, c.C, r.D, r.x2_raw) - c.F
    assert np.max(np.abs(residual)) <= 1e-9 * abs(c.F)
    d = r.solver_diagnostics
    assert d.eta <= ETA_MAX and d.imag_ratio <= IMAG_MAX and d.refinements <= 1  # A-6


@pytest.mark.parametrize("n_x, n_y", GRID_PAIRS)
@pytest.mark.parametrize("q", [1.0, -2.5])
def test_normalization_i4_keeps_field(n_x, n_y, q):
    first, r = both(q=q, n_x=n_x, n_y=n_y)
    assert math.isfinite(r.c2) and r.c2 != 0.0
    assert np.max(np.abs(r.x2_normalized)) == pytest.approx(1.0, abs=2 * U)
    assert np.sum(r.x2_normalized) > 0.0
    np.testing.assert_array_equal(r.x2_raw / r.c2, r.x2_normalized)
    np.testing.assert_array_equal(first.y1_normalized * r.c2, r.y2_scaled)
    raw = np.outer(r.grid_x.with_boundary(r.x2_raw), r.grid_y.with_boundary(first.y1_normalized))
    assert np.all(np.abs(r.w2 - raw) <= 4 * U * np.abs(raw))


@pytest.mark.parametrize("n_x, n_y", GRID_PAIRS)
def test_field_shape_axes_and_zero_boundary(n_x, n_y):
    first, r = both(n_x=n_x, n_y=n_y)
    w = r.w2
    assert w.shape == (n_x + 1, n_y + 1)
    assert np.all(w[0, :] == 0.0) and np.all(w[-1, :] == 0.0)
    assert np.all(w[:, 0] == 0.0) and np.all(w[:, -1] == 0.0)
    np.testing.assert_array_equal(w[1:-1, 1:-1], np.outer(r.x2_normalized, r.y2_scaled))
    assert r.w1.shape == w.shape  # w1 is kept next to w2


def test_sign_of_deflection_follows_load():
    _, up = both(q=1.0)
    _, down = both(q=-1.0)
    assert np.all(up.w2[1:-1, 1:-1] > 0.0) and np.all(down.w2[1:-1, 1:-1] < 0.0)
    np.testing.assert_allclose(down.w2, -up.w2, rtol=1e-15, atol=0)


@pytest.mark.parametrize("factor", [3.7, -2.0])
def test_field_is_linear_in_q_and_inverse_in_d(factor):
    _, base = both(n_x=20, n_y=30)
    _, by_q = both(q=factor, n_x=20, n_y=30)
    _, by_d = both(D=abs(factor), n_x=20, n_y=30)
    np.testing.assert_allclose(by_q.w2, factor * base.w2, rtol=1e-12, atol=0)
    np.testing.assert_allclose(by_d.w2, base.w2 / abs(factor), rtol=1e-12, atol=0)


@pytest.mark.parametrize("n", [20, 21, 80])
def test_a8_mirror_symmetry(n):
    _, r = both(n_x=n, n_y=n)
    w = r.w2
    scale = np.max(np.abs(w))
    assert np.max(np.abs(w - w[::-1, :])) <= A8_TOL * scale
    assert np.max(np.abs(w - w[:, ::-1])) <= A8_TOL * scale


def test_a5g_scale_invariance_of_second_halfstep():
    """Test 16: X2 from the unnormalized Y1_raw gives the same field w2 (A-5g, s = 2)."""
    first, r = both(n_x=80, n_y=80)
    alt = solve_halfstep(first.grid_y, first.y1_raw, first.grid_x, first.q, first.D)
    w2_alt = np.outer(first.grid_x.with_boundary(alt.raw), first.grid_y.with_boundary(first.y1_raw))
    assert np.max(np.abs(w2_alt - r.w2)) / np.max(np.abs(r.w2)) <= A5G_TOL


@pytest.mark.parametrize("n_x, n_y", [(20, 20), (20, 40)])
def test_node_value_returns_w2_not_w1(n_x, n_y):
    first, r = both(n_x=n_x, n_y=n_y)
    assert fh.node_value(r, "1/2", "1/2") == r.w2[n_x // 2, n_y // 2]
    assert fh.node_value(r, "1/2", "1/2") != r.w1[n_x // 2, n_y // 2]
    assert fh.node_value(first, "1/2", "1/2") == first.w1[n_x // 2, n_y // 2]


# Errors -----------------------------------------------------------------------------


def test_requires_first_result():
    with pytest.raises(TypeError):
        second_halfstep(object())


def test_degenerate_coefficients_stop_before_solver(monkeypatch):
    first = first_halfstep(1.0, 1.0, 1.0, 20, 20)
    calls = []

    def degenerate(grid, f, q):
        raise CoefficientError(ERROR_DEGENERATE, "forced in test")

    monkeypatch.setattr(fh, "compute_coefficients", degenerate)
    monkeypatch.setattr(fh, "solve_k", lambda *args: calls.append(args))
    result = None
    with pytest.raises(CoefficientError) as info:
        result = second_halfstep(first)
    assert info.value.code == ERROR_DEGENERATE and calls == [] and result is None


def test_normalization_errors_return_no_field(monkeypatch):
    first = first_halfstep(1.0, 1.0, 1.0, 20, 20)

    class FakeStep:
        def __init__(self, raw):
            self.raw, self.coefficients, self.solver_diagnostics = raw, None, None

    for values, code in (([1.0, -1.0], ERROR_DEGENERATE), ([0.0, 0.0], ERROR_DEGENERATE), ([np.nan, 1.0], ERROR_NONFINITE)):
        raw = np.zeros(19)
        raw[:2] = values
        monkeypatch.setattr(sh, "solve_halfstep", lambda *args, raw=raw: FakeStep(raw))
        result = None
        with pytest.raises(HalfStepError) as info:
            result = second_halfstep(first)
        assert info.value.code == code and result is None


@pytest.mark.parametrize("code", [ERROR_LINEAR_SOLVE, ERROR_NONFINITE])
def test_solver_errors_are_not_masked(monkeypatch, code):
    first = first_halfstep(1.0, 1.0, 1.0, 20, 20)

    def failing(*args):
        raise LinearSolveError(code, "forced in test")

    monkeypatch.setattr(fh, "solve_k", failing)
    with pytest.raises(LinearSolveError) as info:
        second_halfstep(first)
    assert info.value.code == code
