"""Own tests of kkvim.first_halfstep: half-step s = 1 and the field w1 (R2, step 4).

Checks follow docs/NUMERICAL_SPEC.md 3.1-3.3, 5.2, 6.2-6.3, 8, 9.1 (I-4), D-030 and
table 18.1 (A-6, A-8 with 1e-12, A-12). No comparison with kkvim_reference here.
"""

import math
from fractions import Fraction

import numpy as np
import pytest

import kkvim.first_halfstep as fh
from kkvim.coefficients import ERROR_DEGENERATE, ERROR_INPUT, CoefficientError, compute_coefficients
from kkvim.first_halfstep import (
    NOT_A_NODE_MESSAGE,
    HalfStepError,
    NotANodeError,
    first_halfstep,
    node_value,
    solve_halfstep,
)
from kkvim.grid import Grid1D
from kkvim.linear_solver import ERROR_NONFINITE, ETA_MAX, IMAG_MAX, apply_k

U = 2.0**-53
A8_TOL = 1e-12  # table 18.1, A-8
GRID_PAIRS = [(20, 20), (20, 40), (40, 20), (21, 20), (20, 21), (7, 9)]


def full_raw_field(r):
    return np.outer(r.grid_x.with_boundary(r.x0), r.grid_y.with_boundary(r.y1_raw))


@pytest.mark.parametrize("n_x, n_y", GRID_PAIRS)
def test_y1_from_x0_and_counters(n_x, n_y):
    r = first_halfstep(1.0, 1.0, 1.0, n_x, n_y)
    assert (r.s, r.k) == (1, 0)  # D-030: k = floor(s / 2)
    assert r.k == r.s // 2
    np.testing.assert_array_equal(r.x0, np.sin(math.pi * np.arange(1, n_x) / n_x))
    assert r.grid_x.n == n_x and r.grid_y.n == n_y
    assert r.x0.shape == (n_x - 1,) and r.x1_scaled.shape == (n_x - 1,)
    assert r.y1_raw.shape == (n_y - 1,) and r.y1_normalized.shape == (n_y - 1,)
    assert r.coefficients == compute_coefficients(r.grid_x, r.x0, 1.0)


@pytest.mark.parametrize("n_x, n_y", GRID_PAIRS)
def test_y1_solves_system_with_constant_rhs(n_x, n_y):
    q, D = 1.3, 0.8
    r = first_halfstep(2.0, D, q, n_x, n_y)
    c = r.coefficients
    residual = apply_k(r.grid_y, c.A, c.B, c.C, D, r.y1_raw) - c.F
    assert np.max(np.abs(residual)) <= 1e-9 * abs(c.F)
    d = r.solver_diagnostics
    assert d.eta <= ETA_MAX and d.imag_ratio <= IMAG_MAX and d.refinements <= 1  # A-6


def test_coefficients_on_grid_x_and_solve_on_grid_y(monkeypatch):
    seen = {}
    original_coefficients, original_solve = fh.compute_coefficients, fh.solve_k

    def spy_coefficients(grid, f, q):
        seen["coefficients_grid"] = grid
        seen["factor_shape"] = np.shape(f)
        return original_coefficients(grid, f, q)

    def spy_solve(grid, A, B, C, D, g):
        seen["solve_grid"] = grid
        seen["rhs"] = np.array(g)
        return original_solve(grid, A, B, C, D, g)

    monkeypatch.setattr(fh, "compute_coefficients", spy_coefficients)
    monkeypatch.setattr(fh, "solve_k", spy_solve)
    for n_x, n_y in [(20, 40), (40, 20), (21, 20)]:
        r = first_halfstep(1.0, 1.0, 1.0, n_x, n_y)
        assert seen["coefficients_grid"] is r.grid_x and seen["coefficients_grid"].n == n_x
        assert seen["solve_grid"] is r.grid_y and seen["solve_grid"].n == n_y
        assert seen["factor_shape"] == (n_x - 1,)
        assert seen["rhs"].shape == (n_y - 1,)
        assert np.all(seen["rhs"] == r.coefficients.F)  # F_h in every interior node of y
        assert r.w1.shape == (n_x + 1, n_y + 1)


def test_solve_halfstep_uses_both_grids_explicitly():
    grid_x, grid_y = Grid1D(1.0, 12), Grid1D(1.0, 30)
    x0 = np.sin(math.pi * np.arange(1, 12) / 12)
    step = solve_halfstep(grid_x, x0, grid_y, 1.0, 1.0)
    assert step.raw.shape == (29,)
    assert step.coefficients == compute_coefficients(grid_x, x0, 1.0)
    with pytest.raises(ValueError):  # the factor must live on the fixed grid
        solve_halfstep(grid_y, x0, grid_x, 1.0, 1.0)


# Field ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_x, n_y", GRID_PAIRS)
def test_field_shape_axes_and_zero_boundary(n_x, n_y):
    r = first_halfstep(1.0, 1.0, 1.0, n_x, n_y)
    w = r.w1
    assert w.shape == (n_x + 1, n_y + 1)
    assert np.all(w[0, :] == 0.0) and np.all(w[-1, :] == 0.0)
    assert np.all(w[:, 0] == 0.0) and np.all(w[:, -1] == 0.0)
    # Axis 0 is x, axis 1 is y: w[i, j] = X1_scaled_i * Y1_normalized_j.
    np.testing.assert_array_equal(w[1:-1, 1:-1], np.outer(r.x1_scaled, r.y1_normalized))


@pytest.mark.parametrize("n_x, n_y", GRID_PAIRS)
@pytest.mark.parametrize("q", [1.0, -2.5])
def test_normalization_keeps_field_and_fixes_scale(n_x, n_y, q):
    r = first_halfstep(1.0, 1.0, q, n_x, n_y)
    assert math.isfinite(r.c) and r.c != 0.0
    assert np.max(np.abs(r.y1_normalized)) == pytest.approx(1.0, abs=2 * U)
    assert np.sum(r.y1_normalized) > 0.0
    np.testing.assert_array_equal(r.y1_raw / r.c, r.y1_normalized)
    np.testing.assert_array_equal(r.x0 * r.c, r.x1_scaled)
    raw = full_raw_field(r)
    # The physical field is unchanged by normalization (<= a few u per node).
    assert np.all(np.abs(r.w1 - raw) <= 4 * U * np.abs(raw))


def test_sign_of_deflection_follows_load():
    up = first_halfstep(1.0, 1.0, 1.0, 20, 20)
    down = first_halfstep(1.0, 1.0, -1.0, 20, 20)
    assert up.c > 0.0 and down.c < 0.0  # sigma < 0 for negative load
    assert np.all(up.w1[1:-1, 1:-1] > 0.0) and np.all(down.w1[1:-1, 1:-1] < 0.0)
    np.testing.assert_allclose(down.w1, -up.w1, rtol=1e-15, atol=0)
    np.testing.assert_allclose(down.y1_normalized, up.y1_normalized, rtol=1e-15, atol=0)


@pytest.mark.parametrize("factor", [3.7, 0.013, -2.0])
def test_field_is_linear_in_q(factor):
    base = first_halfstep(1.0, 1.0, 1.0, 20, 30)
    scaled = first_halfstep(1.0, 1.0, factor, 20, 30)
    np.testing.assert_allclose(scaled.w1, factor * base.w1, rtol=1e-12, atol=0)


@pytest.mark.parametrize("factor", [2.0, 0.37])
def test_field_is_inverse_in_d(factor):
    base = first_halfstep(1.0, 1.0, 1.0, 20, 30)
    scaled = first_halfstep(1.0, factor, 1.0, 20, 30)
    np.testing.assert_allclose(scaled.w1, base.w1 / factor, rtol=1e-12, atol=0)


@pytest.mark.parametrize("n", [20, 21, 80])
def test_a8_mirror_symmetry(n):
    w = first_halfstep(1.0, 1.0, 1.0, n, n).w1
    scale = np.max(np.abs(w))
    assert np.max(np.abs(w - w[::-1, :])) <= A8_TOL * scale  # x -> a - x
    assert np.max(np.abs(w - w[:, ::-1])) <= A8_TOL * scale  # y -> a - y


# Nodes (A-12) ----------------------------------------------------------------------


def test_center_is_node_for_even_n():
    r = first_halfstep(1.0, 1.0, 1.0, 20, 20)
    assert node_value(r, "1/2", "1/2") == r.w1[10, 10]
    assert node_value(r, Fraction(1, 4), 1) == r.w1[5, 20] == 0.0


@pytest.mark.parametrize("n_x, n_y", [(21, 21), (21, 20), (20, 21)])
def test_center_is_not_a_node_for_odd_n(n_x, n_y):
    r = first_halfstep(1.0, 1.0, 1.0, n_x, n_y)
    with pytest.raises(NotANodeError) as info:
        node_value(r, "1/2", "1/2")
    assert str(info.value) == NOT_A_NODE_MESSAGE


def test_node_coordinates_must_be_rational():
    r = first_halfstep(1.0, 1.0, 1.0, 20, 20)
    with pytest.raises(TypeError):
        node_value(r, 0.5, 0.5)
    with pytest.raises(ValueError):
        node_value(r, "3/2", "1/2")


# Errors before solve_k ---------------------------------------------------------


def test_degenerate_coefficients_stop_before_solver(monkeypatch):
    calls = []

    def degenerate(grid, f, q):
        raise CoefficientError(ERROR_DEGENERATE, "forced in test")

    monkeypatch.setattr(fh, "compute_coefficients", degenerate)
    monkeypatch.setattr(fh, "solve_k", lambda *args: calls.append(args))
    result = None
    with pytest.raises(CoefficientError) as info:
        result = first_halfstep(1.0, 1.0, 1.0, 20, 20)
    assert info.value.code == ERROR_DEGENERATE
    assert calls == []
    assert result is None


@pytest.mark.parametrize("code", [ERROR_NONFINITE, ERROR_INPUT])
def test_other_coefficient_errors_stop_before_solver(monkeypatch, code):
    calls = []

    def failing(grid, f, q):
        raise CoefficientError(code, "forced in test")

    monkeypatch.setattr(fh, "compute_coefficients", failing)
    monkeypatch.setattr(fh, "solve_k", lambda *args: calls.append(args))
    with pytest.raises(CoefficientError) as info:
        first_halfstep(1.0, 1.0, 1.0, 20, 20)
    assert info.value.code == code and calls == []


def test_normalization_errors(monkeypatch):
    class FakeSolution:
        def __init__(self, y):
            self.raw, self.coefficients, self.solver_diagnostics = y, None, None

    for values, code in (([1.0, -1.0, 0.0], ERROR_DEGENERATE), ([0.0, 0.0, 0.0], ERROR_DEGENERATE), ([np.nan, 1.0, 1.0], ERROR_NONFINITE)):
        y = np.zeros(19)
        y[:3] = values
        monkeypatch.setattr(fh, "solve_halfstep", lambda *args, y=y: FakeSolution(y))
        with pytest.raises(HalfStepError) as info:
            first_halfstep(1.0, 1.0, 1.0, 20, 20)
        assert info.value.code == code


# Invalid input --------------------------------------------------------------------


@pytest.mark.parametrize("name, value", [("a", 0.0), ("a", -1.0), ("a", math.inf), ("D", 0.0), ("D", -2.0), ("D", math.nan)])
def test_invalid_a_or_d_is_error_input(name, value):
    args = {"a": 1.0, "D": 1.0, "q": 1.0, "n_x": 20, "n_y": 20, name: value}
    with pytest.raises(HalfStepError) as info:
        first_halfstep(**args)
    assert info.value.code == ERROR_INPUT


@pytest.mark.parametrize("q", [0.0, math.nan, math.inf])
def test_invalid_q_is_error_input(q):
    with pytest.raises(CoefficientError) as info:
        first_halfstep(1.0, 1.0, q, 20, 20)
    assert info.value.code == ERROR_INPUT


@pytest.mark.parametrize("n_x, n_y", [(3, 20), (20, 3), (20.0, 20), (20, True)])
def test_invalid_grid_size_is_rejected(n_x, n_y):
    with pytest.raises((ValueError, TypeError)):
        first_halfstep(1.0, 1.0, 1.0, n_x, n_y)


@pytest.mark.parametrize("name", ["a", "D", "q"])
def test_non_numeric_parameter_is_rejected(name):
    args = {"a": 1.0, "D": 1.0, "q": 1.0, "n_x": 20, "n_y": 20, name: "1.0"}
    with pytest.raises(TypeError):
        first_halfstep(**args)
