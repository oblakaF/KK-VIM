"""Own tests of kkvim.coefficients: half-step coefficients A_h, B_h, C_h, F_h (R2, step 3).

Formulas and checks follow docs/NUMERICAL_SPEC.md, sections 5.2, 6.2 (ChV-6), 7.1,
8.3, 9.2, 11.3 and table 18.1 (A-3: relative error <= 1e-13 on n = 4, 7, 20, 320).
"""

import math

import numpy as np
import pytest

from kkvim.coefficients import (
    ERROR_DEGENERATE,
    ERROR_INPUT,
    CoefficientError,
    HalfStepCoefficients,
    compute_coefficients,
)
from kkvim.grid import Grid1D, inner, second_difference
from kkvim.linear_solver import ERROR_NONFINITE, solve_k

PI = math.pi
A3_TOL = 1e-13  # table 18.1, A-3
ROUNDING_RTOL = 1e-12


def x0(grid):
    return np.sin(PI * np.arange(1, grid.n) / grid.n)


def relative_error(value, expected):
    return abs(value - expected) / abs(expected)


# A-3: X0 against the closed formulas ChV-6 ----------------------------------------


@pytest.mark.parametrize("n", [4, 7, 20, 320])
@pytest.mark.parametrize("a", [1.0, 2.5])
@pytest.mark.parametrize("q", [1.0, -3.7])
def test_a3_x0_closed_formulas(n, a, q):
    grid = Grid1D(a, n)
    h = grid.h
    mu1 = 4.0 / h**2 * math.sin(PI / (2 * n)) ** 2
    c = compute_coefficients(grid, x0(grid), q)
    expected = {"A": a / 2, "B": a / 2 * mu1, "C": a / 2 * mu1**2, "F": q * h / math.tan(PI / (2 * n))}
    for name, value in expected.items():
        assert relative_error(getattr(c, name), value) <= A3_TOL, (name, getattr(c, name), value)
    # Double root of the X0 system: B_h^2 = A_h C_h (ChV-6).
    assert relative_error(c.B**2, c.A * c.C) <= A3_TOL


# Construction properties ----------------------------------------------------------


GRIDS = [(1.0, 4), (1.0, 7), (2.5, 20), (0.7, 101)]


@pytest.mark.parametrize("length, n", GRIDS)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_nonnegative_and_chv1(length, n, seed):
    grid = Grid1D(length, n)
    f = np.random.default_rng(seed).standard_normal(grid.interior_size) * 10.0 ** (seed - 1)
    c = compute_coefficients(grid, f, 1.0)
    assert c.A > 0.0 and c.B >= 0.0 and c.C >= 0.0
    # ChV-1: the sum of squares equals -<f, L f> up to rounding.
    assert math.isclose(c.B, -inner(grid, f, second_difference(grid, f)), rel_tol=ROUNDING_RTOL)
    assert math.isclose(c.A, inner(grid, f, f), rel_tol=ROUNDING_RTOL)
    lf = second_difference(grid, f)
    assert math.isclose(c.C, inner(grid, lf, lf), rel_tol=ROUNDING_RTOL)


@pytest.mark.parametrize("n", [4, 7, 20])
@pytest.mark.parametrize("position", ["first", "last"])
def test_b_includes_end_intervals(n, position):
    grid = Grid1D(1.3, n)
    f = np.zeros(grid.interior_size)
    f[0 if position == "first" else -1] = 1.0
    c = compute_coefficients(grid, f, 1.0)
    # Two nonzero differences +-1/h: the end interval and its neighbour.
    assert math.isclose(c.B, 2.0 / grid.h, rel_tol=1e-15)


def test_constant_multiplier_uses_end_intervals():
    grid = Grid1D(1.0, 10)
    c = compute_coefficients(grid, np.ones(grid.interior_size), 1.0)
    # Only the two end intervals contribute: (1 - 0)/h and (0 - 1)/h.
    assert math.isclose(c.B, 2.0 / grid.h, rel_tol=1e-15)


@pytest.mark.parametrize("q", [1.0, 2.5, -0.4, -7.0])
def test_f_is_linear_in_q(q):
    grid = Grid1D(1.0, 20)
    f = x0(grid) + 0.1 * np.cos(np.arange(1, grid.n))
    base = compute_coefficients(grid, f, 1.0)
    c = compute_coefficients(grid, f, q)
    assert math.isclose(c.F, q * base.F, rel_tol=1e-15)
    assert (c.A, c.B, c.C) == (base.A, base.B, base.C)


def test_returns_four_fields():
    c = compute_coefficients(Grid1D(1.0, 8), np.ones(7), 1.0)
    assert isinstance(c, HalfStepCoefficients)
    assert all(isinstance(getattr(c, name), float) for name in "ABCF")


# Errors ---------------------------------------------------------------------------


def test_wrong_shape():
    grid = Grid1D(1.0, 8)
    for bad in (np.ones(8), np.ones(6), np.ones((7, 1))):
        with pytest.raises(ValueError):
            compute_coefficients(grid, bad, 1.0)


@pytest.mark.parametrize("q", [0.0, -0.0, 0, math.nan, math.inf, -math.inf])
def test_invalid_load_is_error_input(q):
    grid = Grid1D(1.0, 8)
    with pytest.raises(CoefficientError) as info:
        compute_coefficients(grid, x0(grid), q)
    assert info.value.code == ERROR_INPUT


def test_non_numeric_load():
    grid = Grid1D(1.0, 8)
    for q in (True, "1.0", None):
        with pytest.raises(TypeError):
            compute_coefficients(grid, x0(grid), q)


def test_zero_multiplier_is_degenerate():
    grid = Grid1D(1.0, 8)
    with pytest.raises(CoefficientError) as info:
        compute_coefficients(grid, np.zeros(7), 1.0)
    assert info.value.code == ERROR_DEGENERATE


@pytest.mark.parametrize("q", [1.0, -2.0])
def test_exactly_zero_sum_is_degenerate_before_any_solve(q):
    grid = Grid1D(1.0, 8)
    f = np.zeros(grid.interior_size)
    f[:2] = [1.0, -1.0]  # nonzero multiplier, grid sum exactly 0.0 in float64
    assert np.sum(f) == 0.0 and np.any(f != 0.0)
    with pytest.raises(CoefficientError) as info:
        compute_coefficients(grid, f, q)
    assert info.value.code == ERROR_DEGENERATE

    # Why the check must happen here: the linear solver itself accepts the zero
    # right-hand side F_h * 1 = 0 and returns Y = 0 without any error.
    result = solve_k(grid, 1.0, 0.0, 0.0, 1.0, np.zeros(grid.interior_size))
    assert np.all(result.y == 0.0)


def test_observation_rounding_level_sum_is_not_detected():
    """Diagnostic observation only, not a criterion (no closeness threshold, as specified).

    A multiplier whose grid sum is zero mathematically but not in float64 gives a tiny
    nonzero F_h; the exact check F_h = 0 does not detect it.
    """
    grid = Grid1D(1.0, 4)
    f = np.array([0.1, 0.2, -0.3])  # 0.1 + 0.2 - 0.3 = 5.55e-17 in float64
    c = compute_coefficients(grid, f, 1.0)
    assert c.F != 0.0
    assert abs(c.F) <= 1e-16
    assert c.F == 1.0 * grid.h * float(np.sum(f))


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_nonfinite_multiplier(bad):
    grid = Grid1D(1.0, 8)
    f = x0(grid)
    f[3] = bad
    with pytest.raises(CoefficientError) as info:
        compute_coefficients(grid, f, 1.0)
    assert info.value.code == ERROR_NONFINITE


@pytest.mark.parametrize("scale", [1e160, 1e200, 1e300])
def test_overflow_is_not_returned_silently(scale):
    grid = Grid1D(1.0, 20)
    with pytest.raises(CoefficientError) as info:
        compute_coefficients(grid, scale * x0(grid), 1.0)
    assert info.value.code == ERROR_NONFINITE


def test_overflow_in_load_product():
    grid = Grid1D(1.0, 20)
    with pytest.raises(CoefficientError) as info:
        compute_coefficients(grid, 1e150 * x0(grid), 1e300)
    assert info.value.code == ERROR_NONFINITE
