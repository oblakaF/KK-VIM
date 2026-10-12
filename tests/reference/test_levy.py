"""Own tests of the analytic Levy reference for the first half-step (R2, s = 1).

Checks only the reference module kkvim_reference.levy (MATH_SPEC.md, В-29,
В-30; NUMERICAL_SPEC.md, section 14). Comparisons with the main solver are
written by QA (D-033) and are not here.

Tolerances are fixed before the comparison and are not tuned to results:
- RTOL_ROUND = 1000 u, u = 1.1e-16 (rounding model of NUMERICAL_SPEC.md,
  section 18): engineering bound for floating-point evaluation of the closed
  form (cancellation in the bracket of В-29 amplifies u by a factor < 10,
  rescaled arguments x/a add a few u; margin 10-100 on top);
- centre against 0.004109346839473: half a unit of the 15th decimal place;
- centre against 0.004109346839: А-4в, <= 5e-13 absolute (NUMERICAL_SPEC.md,
  table 18.1).
"""

import math

import numpy as np
import pytest

from kkvim_reference.levy import levy_w1, levy_x0, levy_y1

U = 1.1e-16
RTOL_ROUND = 1000 * U

CENTRE_15 = 0.004109346839473  # NUMERICAL_SPEC.md, section 14, item 6 [SCRATCH]
TOL_CENTRE_15 = 5e-16
CENTRE_12 = 0.004109346839  # MATH_SPEC.md, В-30
TOL_A4V = 5e-13  # А-4в

# Representative physical data, consistent units (D-009).
A, Q, D = 2.9, 3.7e3, 1.3e2


def scale(a=A, q=Q, d=D):
    return q * a**4 / d


def grid(a, n=40):
    return np.linspace(0.0, a, n + 1)


@pytest.mark.parametrize(("a", "q", "d"), [(1.0, 1.0, 1.0), (A, Q, D), (0.37, -2.5, 7.1)])
def test_zero_on_all_four_edges(a, q, d):
    s = grid(a)
    zero = np.zeros_like(s)
    full = np.full_like(s, a)
    bound = RTOL_ROUND * abs(scale(a, q, d))
    assert np.max(np.abs(levy_w1(zero, s, a=a, q=q, D=d))) <= bound  # x = 0
    assert np.max(np.abs(levy_w1(full, s, a=a, q=q, D=d))) <= bound  # x = a
    assert np.max(np.abs(levy_w1(s, zero, a=a, q=q, D=d))) <= bound  # y = 0
    assert np.max(np.abs(levy_w1(s, full, a=a, q=q, D=d))) <= bound  # y = a


@pytest.mark.parametrize("a", [1.0, A])
def test_symmetry_about_centre_in_both_coordinates(a):
    s = grid(a)
    x, y = np.meshgrid(s, s, indexing="ij")
    w = levy_w1(x, y, a=a, q=Q, D=D)
    mirror_x = levy_w1(a - x, y, a=a, q=Q, D=D)
    mirror_y = levy_w1(x, a - y, a=a, q=Q, D=D)
    ref = np.max(np.abs(w))
    assert np.max(np.abs(w - mirror_x)) <= RTOL_ROUND * ref
    assert np.max(np.abs(w - mirror_y)) <= RTOL_ROUND * ref
    # Maximum at the centre.
    assert abs(levy_w1(a / 2, a / 2, a=a, q=Q, D=D)) >= ref * (1 - RTOL_ROUND)


@pytest.mark.parametrize("factor", [3.7, -0.25, 1e-3])
def test_deflection_proportional_to_load(factor):
    s = grid(A)
    x, y = np.meshgrid(s, s, indexing="ij")
    w = levy_w1(x, y, a=A, q=Q, D=D)
    w_scaled = levy_w1(x, y, a=A, q=factor * Q, D=D)
    assert np.max(np.abs(w_scaled - factor * w)) <= RTOL_ROUND * abs(factor) * np.max(np.abs(w))


@pytest.mark.parametrize(("a", "q", "d"), [(A, Q, D), (0.37, 12.0, 0.013), (25.0, 1e-2, 4e5)])
def test_scaling_as_q_a4_over_d(a, q, d):
    xi = np.linspace(0.0, 1.0, 41)
    xi_x, xi_y = np.meshgrid(xi, xi, indexing="ij")
    w_bar_unit = levy_w1(xi_x, xi_y, a=1.0, q=1.0, D=1.0)
    w_bar = levy_w1(xi_x * a, xi_y * a, a=a, q=q, D=d) / scale(a, q, d)
    assert np.max(np.abs(w_bar - w_bar_unit)) <= RTOL_ROUND * np.max(np.abs(w_bar_unit))


def test_centre_value_unit_data():
    w_c = levy_w1(0.5, 0.5, a=1.0, q=1.0, D=1.0)
    assert abs(w_c - CENTRE_15) <= TOL_CENTRE_15


def test_centre_value_against_published_rounding_a4v():
    w_c = levy_w1(0.5, 0.5, a=1.0, q=1.0, D=1.0)
    assert abs(w_c - CENTRE_12) <= TOL_A4V
    # Same dimensionless coefficient for other data (В-30).
    w_bar = levy_w1(A / 2, A / 2, a=A, q=Q, D=D) / scale()
    assert abs(w_bar - CENTRE_12) <= TOL_A4V


def test_factors_x0_and_y1():
    # w1 = X0 Y1 (В-29); X0(a/2) = 1, Y1 at the centre equals the centre deflection.
    assert levy_x0(0.5, a=1.0) == 1.0
    assert levy_y1(0.5, a=1.0, q=1.0, D=1.0) == levy_w1(0.5, 0.5, a=1.0, q=1.0, D=1.0)
    s = grid(A)
    x, y = np.meshgrid(s, s, indexing="ij")
    np.testing.assert_array_equal(
        levy_w1(x, y, a=A, q=Q, D=D), levy_x0(x, a=A) * levy_y1(y, a=A, q=Q, D=D)
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"a": 0.0, "q": 1.0, "D": 1.0},
        {"a": -1.0, "q": 1.0, "D": 1.0},
        {"a": 1.0, "q": 1.0, "D": 0.0},
        {"a": 1.0, "q": math.inf, "D": 1.0},
        {"a": math.nan, "q": 1.0, "D": 1.0},
    ],
)
def test_rejects_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        levy_w1(0.5, 0.5, **kwargs)


@pytest.mark.parametrize(("x", "y"), [(-0.1, 0.5), (0.5, 1.1), (math.nan, 0.5)])
def test_rejects_points_outside_plate(x, y):
    with pytest.raises(ValueError):
        levy_w1(x, y, a=1.0, q=1.0, D=1.0)
