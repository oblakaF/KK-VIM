"""Own tests of the rectangular Navier reference NavierRectangle (R3; D-036).

Formulas: (131) of L-107 for a x b with r = b / a and the reference length a
(R3_NUMERICAL_SPEC_DRAFT.md, 4.1); strict bounds ЧВ-R2 (deflection) and
ЧВ-R2a (integral), 4.2-4.3; range 0.2 <= r <= 5 and level M = 2559 (Р-1, Р-5).

No new qualification tolerances. Comparisons use
- RTOL_CLOSED = 16 u for closed forms (as in test_navier.py);
- the rounding bounds reported by the module (rounding_w, rounding_mx,
  rounding_my, rounding_J) for exact analytic identities (axis exchange,
  w_c(1/r) = r^-4 w_c(r), summation orders, reflections);
- the strict bounds ЧВ-R2, ЧВ-R2a, which must cover differences of levels.
L-107, table 8 is not a criterion (Р-6): r = 1.4 is checked internally only.

Mathematical checks run at small M; the control checks at M = M_R3 = 2559 use
a few points and small grids.
"""

import ast
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from kkvim_reference.navier import (
    M_R2,
    M_R3,
    NAVIER_LEVELS,
    NAVIER_LEVELS_R3,
    NavierPlate,
    NavierRectangle,
    NavierRectangleConvergence,
    deflection_integral_coef,
    deflection_integral_coef_rect,
    deflection_integral_tail_bound,
    deflection_integral_tail_bound_rect,
    deflection_tail_bound,
    deflection_tail_bound_rect,
)

U = 2.0**-53
RTOL_CLOSED = 16 * U
NU = 0.3
# Control ratios of Р-1 (1.4 - diagnostic only) and the range ends.
R_CONTROL = (0.2, 1 / 3, 0.5, 2 / 3, 1.0, 1.4, 1.5, 2.0, 3.0, 5.0)
M_SMALL = 159

NAVIER_SOURCE = Path(__file__).resolve().parents[2] / "src" / "kkvim_reference" / "navier.py"


def plate(r, a=1.0, q=1.0, D=1.0, nu=NU):
    return NavierRectangle(a=a, b=r * a, q=q, D=D, nu=nu)


def centre(p, M, **kw):
    return p.at_points(p.a / 2, p.b / 2, M=M, **kw)


def random_points(p, count, seed):
    rng = np.random.default_rng(seed)
    return p.a * rng.random(count), p.b * rng.random(count)


# --- M = 1: exact first term -----------------------------------------------------------


@pytest.mark.parametrize("r", R_CONTROL)
def test_m1_centre_is_first_navier_term(r):
    # П-1 for a x b (4.5): w = 16 / (pi^6 (1 + r^-2)^2) in q a^4 / D
    expected = 16.0 / (math.pi**6 * (1.0 + 1.0 / (r * r)) ** 2)
    assert float(centre(plate(r), 1).w_coef) == pytest.approx(expected, rel=RTOL_CLOSED, abs=0)


@pytest.mark.parametrize("r", R_CONTROL)
def test_m1_moments_and_integral_are_first_terms(r):
    res = centre(plate(r), 1)
    den = (1.0 + 1.0 / (r * r)) ** 2
    assert float(res.mx_coef) == pytest.approx(16 / math.pi**4 * (1 + NU / (r * r)) / den, rel=RTOL_CLOSED)
    assert float(res.my_coef) == pytest.approx(16 / math.pi**4 * (NU + 1 / (r * r)) / den, rel=RTOL_CLOSED)
    J1 = 64.0 / math.pi**8 * r / den
    assert plate(r).deflection_integral(M=1).J_M == pytest.approx(J1, rel=RTOL_CLOSED, abs=0)


def test_m1_physical_value_with_both_lengths():
    # (131) with one term at the centre: 16 q / (pi^6 D (1/a^2 + 1/b^2)^2)
    a, b, q, D = 0.8, 2.0, 3.5e3, 41.0
    p = NavierRectangle(a=a, b=b, q=q, D=D, nu=NU)
    expected = 16 * q / (math.pi**6 * D * (1 / a**2 + 1 / b**2) ** 2)
    assert float(centre(p, 1).w) == pytest.approx(expected, rel=RTOL_CLOSED)


# --- r = 1: regression of R2 --------------------------------------------------------------


@pytest.mark.parametrize("M", [1, 9, M_SMALL])
def test_square_equals_r2_plate_bitwise(M):
    sq = NavierRectangle(a=1.3, b=1.3, q=2.0, D=0.7, nu=NU)
    r2 = NavierPlate(a=1.3, q=2.0, D=0.7, nu=NU)
    x, y = random_points(sq, 25, 1)
    for s in ("separable", "diagonal"):
        p, o = sq.at_points(x, y, M=M, summation=s), r2.at_points(x, y, M=M, summation=s)
        for f in ("w_coef", "mx_coef", "my_coef", "rounding_w", "rounding_mx", "rounding_my"):
            assert np.array_equal(getattr(p, f), getattr(o, f)), f
        assert np.array_equal(p.convergence.w, o.convergence.w)
    pn, on = sq.at_nodes(10, 7, M=M), r2.at_nodes(10, 7, M=M)
    for f in ("w_coef", "mx_coef", "my_coef", "x", "y"):
        assert np.array_equal(getattr(pn, f), getattr(on, f)), f
    assert pn.w_tail_bound_coef == on.w_tail_bound_coef == deflection_tail_bound(M)
    g, h = sq.deflection_integral(M=M), r2.deflection_integral(M=M)
    assert (g.J_M, g.I_M, g.Pi_M, g.J_tail_bound, g.rounding_J) == (h.J_M, h.I_M, h.Pi_M, h.J_tail_bound, h.rounding_J)


def test_square_at_r2_level_equals_r2_plate():
    sq, r2 = plate(1.0), NavierPlate(a=1.0, q=1.0, D=1.0, nu=NU)
    pts = (np.array([0.5, 0.25, 0.1]), np.array([0.5, 0.75, 0.6]))
    assert np.array_equal(sq.at_points(*pts, M=M_R2).w_coef, r2.at_points(*pts).w_coef)
    assert sq.deflection_integral(M=M_R2) == r2.deflection_integral()


def test_general_integral_path_at_r1_agrees_with_exact_square_form():
    from kkvim_reference.navier import _integral_coef_general

    for M in (1, 9, 159):
        exact = deflection_integral_coef(M)
        assert abs(_integral_coef_general(M, 1.0) - exact) <= 24 * U * exact + 12 * U * exact


def test_r2_defaults_unchanged():
    assert M_R2 == 1279 and NAVIER_LEVELS[-1] == 1279
    assert M_R3 == 2559 and NAVIER_LEVELS_R3 == NAVIER_LEVELS + (2559,)
    with pytest.raises(ValueError):
        NavierPlate(a=1.0, q=1.0, D=1.0, nu=NU, b=2.0)  # R2 class stays square-only


# --- axis exchange and the identity w_c(1/r) = r^-4 w_c(r) ---------------------------------


@pytest.mark.parametrize("r", [2.0, 0.5, 1.4, 5.0, 0.2, 3.0])
def test_axis_exchange_at_points(r):
    # ЧВ-R1: w(a, b; x, y) = w(b, a; y, x), M_x <-> M_y, same q, D
    p = NavierRectangle(a=1.1, b=1.1 * r, q=2.5, D=0.9, nu=NU)
    s = p.swapped()
    x, y = random_points(p, 30, 2)
    u, v = p.at_points(x, y, M=M_SMALL), s.at_points(y, x, M=M_SMALL)
    tol_w = abs(u.w_scale) * u.rounding_w + abs(v.w_scale) * v.rounding_w
    assert np.max(np.abs(u.w - v.w)) <= tol_w
    tol_m = abs(u.moment_scale) * u.rounding_mx + abs(v.moment_scale) * v.rounding_my
    assert np.max(np.abs(u.Mx - v.My)) <= tol_m
    tol_m = abs(u.moment_scale) * u.rounding_my + abs(v.moment_scale) * v.rounding_mx
    assert np.max(np.abs(u.My - v.Mx)) <= tol_m


@pytest.mark.parametrize(("r", "nx", "ny"), [(2.0, 8, 12), (0.5, 9, 6), (1.4, 10, 14)])
def test_axis_exchange_at_nodes(r, nx, ny):
    p = plate(r, a=1.7)
    u, v = p.at_nodes(nx, ny, M=M_SMALL), p.swapped().at_nodes(ny, nx, M=M_SMALL)
    tol = abs(u.w_scale) * u.rounding_w + abs(v.w_scale) * v.rounding_w
    assert np.max(np.abs(u.w - v.w.T)) <= tol


@pytest.mark.parametrize("r", [2.0, 1.4, 3.0, 5.0])
def test_centre_inverse_ratio_via_exchanged_plate(r):
    # plate a x b = 1 x r and its exchange r x 1 (ratio 1/r, reference length r):
    # w_c(1/r) = w D / (q r^4) = r^-4 w_c(r)
    p = plate(r)
    s = p.swapped()
    assert s.b / s.a == pytest.approx(1 / r, rel=U)
    for M in (1, 9, M_SMALL):
        wr, wi = centre(p, M), centre(s, M)
        lhs, rhs = float(wi.w_coef), float(wr.w_coef) / r**4
        assert abs(lhs - rhs) <= wi.rounding_w + wr.rounding_w / r**4 + 4 * U * abs(lhs)


@pytest.mark.parametrize("r", [2.0, 4.0])
def test_centre_inverse_ratio_with_exact_reciprocal(r):
    # 1/r exact in binary: plates 1 x r and 1 x (1/r) with the same a = 1
    for M in (1, 9, M_SMALL):
        wr, wi = centre(plate(r), M), centre(plate(1 / r), M)
        lhs, rhs = float(wi.w_coef), float(wr.w_coef) / r**4
        assert abs(lhs - rhs) <= wi.rounding_w + wr.rounding_w / r**4


# --- symmetry, maximum, edges, nu, scaling ----------------------------------------------------


@pytest.mark.parametrize("r", [0.2, 1.4, 2.0, 5.0])
def test_reflection_symmetry_at_nodes(r):
    res = plate(r).at_nodes(12, 10, M=M_SMALL)
    w = res.w_coef
    assert np.max(np.abs(w - w[::-1, :])) <= res.rounding_w
    assert np.max(np.abs(w - w[:, ::-1])) <= res.rounding_w


@pytest.mark.parametrize("r", [0.2, 0.5, 1.4, 2.0, 5.0])
def test_maximum_at_centre(r):
    # П-5 as in R2: the centre against its neighbours at 0.05a, 0.05b, one call
    p = plate(r)
    d = np.array([-0.05, 0.0, 0.05])
    x, y = np.meshgrid(p.a / 2 + d * p.a, p.b / 2 + d * p.b, indexing="ij")
    w = p.at_points(x, y, M=M_SMALL).w_coef
    others = np.delete(w.ravel(), 4)
    assert np.all(others < w[1, 1]) and w[1, 1] > 0


def test_edges_exactly_zero():
    p = plate(2.0, a=1.3)
    s = np.linspace(0, 1, 7)
    for x, y in ((0 * s, p.b * s), (p.a + 0 * s, p.b * s), (p.a * s, 0 * s), (p.a * s, p.b + 0 * s)):
        res = p.at_points(x, y, M=M_SMALL)
        assert np.all(res.w_coef == 0.0) and np.all(res.mx_coef == 0.0) and np.all(res.my_coef == 0.0)
    nodes = p.at_nodes(6, 9, M=M_SMALL).w_coef
    assert np.all(nodes[[0, -1], :] == 0.0) and np.all(nodes[:, [0, -1]] == 0.0)
    assert np.all(nodes[1:-1, 1:-1] > 0)


@pytest.mark.parametrize("r", [0.5, 1.4, 3.0])
def test_deflection_independent_of_nu(r):
    x, y = random_points(plate(r), 12, 3)
    w = [plate(r, nu=nu).at_points(x, y, M=M_SMALL).w_coef for nu in (0.0, 0.3, 0.49)]
    assert np.array_equal(w[0], w[1]) and np.array_equal(w[1], w[2])


def test_moments_of_centre_not_proportional_to_one_plus_nu_for_r_not_1():
    # 4.5, П-9: the square property does not hold for r != 1
    m = [float(centre(plate(2.0, nu=nu), M_SMALL).mx_coef) for nu in (0.0, 0.3)]
    assert abs(m[1] / m[0] - 1.3) > 1e-3


@pytest.mark.parametrize("lam", [2.0, 0.25, 3.7])
def test_scaling_at_fixed_ratio(lam):
    # П-8: w proportional to q a^4 / D at fixed r; the dimensionless field is unchanged
    p = NavierRectangle(a=1.0, b=1.4, q=1.0, D=1.0, nu=NU)
    s = NavierRectangle(a=lam, b=1.4 * lam, q=5.0, D=0.3, nu=NU)
    xi, eta = np.array([0.5, 0.2, 0.9]), np.array([0.5, 0.7, 0.1])
    u = p.at_points(xi, 1.4 * eta, M=M_SMALL)
    v = s.at_points(lam * xi, 1.4 * lam * eta, M=M_SMALL)
    if lam in (2.0, 0.25):  # exact binary scaling: the same floating-point operations
        assert np.array_equal(u.w_coef, v.w_coef)
    assert np.max(np.abs(u.w_coef - v.w_coef)) <= u.rounding_w + v.rounding_w
    assert np.allclose(v.w, 5.0 * lam**4 / 0.3 * v.w_coef, rtol=4 * U, atol=0)


# --- summation orders, nodes against points ----------------------------------------------------


@pytest.mark.parametrize("r", [0.2, 1.4, 2.0, 5.0])
def test_two_summation_orders(r):
    p = plate(r)
    x, y = random_points(p, 12, 4)
    s, d = p.at_points(x, y, M=M_SMALL), p.at_points(x, y, M=M_SMALL, summation="diagonal")
    for f, b in (("w_coef", "rounding_w"), ("mx_coef", "rounding_mx"), ("my_coef", "rounding_my")):
        assert np.max(np.abs(getattr(s, f) - getattr(d, f))) <= getattr(s, b) + getattr(d, b)
    ns, nd = p.at_nodes(8, 6, M=79), p.at_nodes(8, 6, M=79, summation="diagonal")
    assert np.max(np.abs(ns.w_coef - nd.w_coef)) <= ns.rounding_w + nd.rounding_w


@pytest.mark.parametrize(("r", "nx", "ny"), [(2.0, 8, 16), (1.4, 10, 14), (0.2, 15, 3)])
def test_nodal_field_equals_pointwise_series(r, nx, ny):
    p = plate(r, a=0.9)
    nodes = p.at_nodes(nx, ny, M=M_SMALL)
    x, y = np.meshgrid(nodes.x, nodes.y, indexing="ij")
    pts = p.at_points(x, y, M=M_SMALL)
    assert np.max(np.abs(nodes.w_coef - pts.w_coef)) <= nodes.rounding_w + pts.rounding_w
    assert nodes.y[-1] == p.b and nodes.x[-1] == p.a


# --- strict bounds ЧВ-R2, ЧВ-R2a ----------------------------------------------------------------


def test_deflection_bound_formula():
    for r in R_CONTROL:
        assert deflection_tail_bound_rect(2559, r) == pytest.approx(
            16 / math.pi**6 * (7 / 8 * 1.2020569031595942) * r * r / (8 * 2559**2), rel=RTOL_CLOSED
        )
        # R3_NUMERICAL_SPEC_DRAFT.md, 4.3: 3.341e-10 r^2 (half a unit of the printed digit)
        assert abs(deflection_tail_bound_rect(2559, r) / (r * r) - 3.341e-10) <= 0.0005e-10
    assert deflection_tail_bound_rect(1279, 1.0) == deflection_tail_bound(1279)
    assert deflection_integral_tail_bound_rect(1279, 1.0) == pytest.approx(
        deflection_integral_tail_bound(1279), rel=RTOL_CLOSED
    )


@pytest.mark.parametrize("r", [0.2, 1.4, 2.0, 5.0])
def test_deflection_bound_covers_level_differences(r):
    # |w_M - w_159| <= bound(M) + bound(159) + rounding for M on the levels below 159
    p = plate(r)
    x, y = random_points(p, 20, 5)
    res = p.at_points(x, y, M=M_SMALL)
    top = res.convergence.w[-1]
    for level, w in zip(res.convergence.levels[:-1], res.convergence.w[:-1]):
        allowed = deflection_tail_bound_rect(level, r) + deflection_tail_bound_rect(M_SMALL, r) + 2 * res.rounding_w
        assert np.max(np.abs(w - top)) <= allowed


@pytest.mark.parametrize("r", [Fraction(2), Fraction(1, 2), Fraction(7, 5), Fraction(5), Fraction(1, 5)])
def test_integral_against_exact_rational_sum(r):
    rf = float(r)
    for M in (1, 3, 9):
        odd = range(1, M + 1, 2)
        s = sum(Fraction(1, m * m * n * n) / (m * m + Fraction(n * n) / (r * r)) ** 2 for m in odd for n in odd)
        expected = 64 / math.pi**8 * float(r) * float(s)
        got = deflection_integral_coef_rect(M, rf)
        # the input r = float(7/5) differs from 7/5 by <= u relative; J is homogeneous of degree <= 5 in r
        assert got == pytest.approx(expected, rel=24 * U + 8 * U + 10 * U)


@pytest.mark.parametrize("r", [0.2, 1.4, 2.0, 5.0])
def test_integral_bound_covers_level_differences(r):
    J = {M: deflection_integral_coef_rect(M, r) for M in (1, 9, 39, 159, 639)}
    for M in (1, 9, 39, 159):
        diff = J[639] - J[M]
        assert -24 * U * J[639] <= diff <= deflection_integral_tail_bound_rect(M, r) + 48 * U * J[639]


@pytest.mark.parametrize("r", [2.0, 1.4, 5.0])
def test_integral_and_energy_axis_exchange(r):
    # the integral over the same plate does not depend on the axes: J(1/r) = r^-6 J(r)
    p = plate(r, q=1.7, D=0.4)
    g, h = p.deflection_integral(M=M_SMALL), p.swapped().deflection_integral(M=M_SMALL)
    assert h.I_M == pytest.approx(g.I_M, rel=2 * (24 + 8) * U)
    assert h.J_M == pytest.approx(g.J_M / r**6, rel=2 * (24 + 8) * U)
    assert g.Pi_M == -0.5 * p.q * g.I_M < 0
    assert g.Pi_tail_bound == pytest.approx(0.5 * p.q * abs(p.q * p.a**6 / p.D) * g.J_tail_bound, rel=4 * U)


# --- input checks (ERROR_INPUT) ------------------------------------------------------------------


# Representation tolerance of the input ratio (owner's decision of 2026-10-14):
# D-038 symmetric band: 1/[5(1+4u)] <= b/a <= 5(1+4u), original floats.
RATIO_HIGH = 5 * (1 + 4 * Fraction(1, 2**53))
RATIO_LOW = 1 / RATIO_HIGH


def ulps(x, k):
    """x moved by k ulps (k < 0 - downwards)."""
    for _ in range(abs(k)):
        x = math.nextafter(x, math.inf if k > 0 else 0.0)
    return x


@pytest.mark.parametrize(
    ("a", "b"),
    [
        (1.0, 5.0),
        (5.0, 1.0),
        (1.0, 0.2),
        (2.0, 10.0),
        (3.0, 3.0),
        (3.0, 0.6),  # float 0.6 < 3/5: b/a = 1/5 (1 - u/3), within 4u
        (0.6, 3.0),
        (1.0, ulps(5.0, 1)),  # 5 (1 + 1.6 u)
        (1.0, ulps(5.0, 2)),  # 5 (1 + 3.2 u)
        (1.0, ulps(0.2, -1)),  # 0.2 (1 - 0.75 u)
        (1.0, ulps(0.2, -3)),  # 0.2 (1 - 3.25 u)
    ],
)
def test_range_ends_accepted(a, b):
    p = NavierRectangle(a=a, b=b, q=1.0, D=1.0, nu=NU)
    assert RATIO_LOW <= Fraction(b) / Fraction(a) <= RATIO_HIGH
    assert 0.2 * (1 - 8 * U) <= p.r <= 5.0 * (1 + 8 * U)


@pytest.mark.parametrize(
    ("a", "b"),
    [
        (1.0, ulps(5.0, 3)),  # 5 (1 + 4.8 u): beyond 4u
        (1.0, ulps(0.2, -4)),  # 0.2 (1 - 4.5 u): beyond 4u
        (1.0, 0.19),
        (1.0, 5.01),
        (1.0, 10.0),
        (10.0, 1.0),
    ],
)
def test_ratio_outside_range_is_error_input(a, b):
    with pytest.raises(ValueError, match="ERROR_INPUT"):
        NavierRectangle(a=a, b=b, q=1.0, D=1.0, nu=NU)


@pytest.mark.parametrize("a", [1.0, 3.0, 0.7, 1.1, 9.7, 0.3])
@pytest.mark.parametrize("target", [0.2, 5.0])
def test_range_check_is_the_exact_4u_rule(a, target):
    """Acceptance equals the exact rule for b within +-12 ulps of target * a; bounds use the same rule on r."""
    for k in range(-12, 13):
        b = ulps(target * a, k)
        if RATIO_LOW <= Fraction(b) / Fraction(a) <= RATIO_HIGH:
            NavierRectangle(a=a, b=b, q=1.0, D=1.0, nu=NU)
        else:
            with pytest.raises(ValueError, match="ERROR_INPUT"):
                NavierRectangle(a=a, b=b, q=1.0, D=1.0, nu=NU)
        r = ulps(target, k)
        if RATIO_LOW <= Fraction(r) <= RATIO_HIGH:
            deflection_tail_bound_rect(9, r)
            deflection_integral_coef_rect(9, r)
            deflection_integral_tail_bound_rect(9, r)
        else:
            for f in (deflection_tail_bound_rect, deflection_integral_coef_rect, deflection_integral_tail_bound_rect):
                with pytest.raises(ValueError, match="ERROR_INPUT"):
                    f(9, r)


def test_accepted_plate_is_not_rejected_by_rounding_of_its_ratio():
    """a = 0.7, b = 0.13999999999999993: exact b/a = 0.2 (1 - 3.93 u) is accepted, fl(b/a) = 0.2 (1 - 4.5 u).

    The plate methods use the ratio checked in the constructor and must not reject it again.
    """
    a, b = 0.7, 0.13999999999999993
    assert RATIO_LOW <= Fraction(b) / Fraction(a) and Fraction(b / a) < RATIO_LOW
    p = NavierRectangle(a=a, b=b, q=1.0, D=1.0, nu=NU)
    res = p.at_points(a / 2, b / 2, M=9)
    assert res.w_tail_bound_coef > 0 and res.w_error_bound_coef > 0
    g = p.deflection_integral(M=9)
    assert g.J_M > 0 and g.J_tail_bound > 0
    p.at_nodes(4, 4, M=9)
    with pytest.raises(ValueError, match="ERROR_INPUT"):  # the public bound applies the rule to its own input r
        deflection_tail_bound_rect(9, b / a)


@pytest.mark.parametrize("r", [0.1, 5.5, ulps(5.0, 3), ulps(0.2, -4)])
def test_bounds_outside_range_are_error_input(r):
    with pytest.raises(ValueError, match="ERROR_INPUT"):
        deflection_tail_bound_rect(2559, r)
    with pytest.raises(ValueError, match="ERROR_INPUT"):
        deflection_integral_coef_rect(9, r)
    with pytest.raises(ValueError, match="ERROR_INPUT"):
        deflection_integral_tail_bound_rect(9, r)


@pytest.mark.parametrize(
    "kwargs", [{"a": 0.0}, {"b": -1.0}, {"b": math.nan}, {"q": math.inf}, {"D": 0.0}, {"nu": 1.0}, {"a": True}]
)
def test_invalid_plate(kwargs):
    with pytest.raises(ValueError):
        NavierRectangle(**({"a": 1.0, "b": 2.0, "q": 1.0, "D": 1.0, "nu": NU} | kwargs))


def test_points_outside_plate_and_invalid_level():
    p = plate(2.0)
    for x, y in ((0.5, 2.0 + 1e-12), (1.0 + 1e-12, 1.0), (-0.1, 0.5), (0.5, math.nan)):
        with pytest.raises(ValueError):
            p.at_points(x, y, M=9)
    assert p.at_points(0.5, 2.0, M=9).w_coef == 0.0  # y = b is inside (edge)
    for M in (0, 2, 3.0, True):
        with pytest.raises(ValueError):
            p.at_points(0.5, 0.5, M=M)
        with pytest.raises(ValueError):
            p.deflection_integral(M=M)
    with pytest.raises(ValueError):
        p.at_points(0.5, 0.5, M=9, summation="pairwise")
    with pytest.raises(ValueError):
        p.at_nodes(0, 4, M=9)


def test_from_material():
    p = NavierRectangle.from_material(a=1.0, b=2.0, q=1.0, E=2.1e11, thickness=0.01, nu=0.3)
    assert p.D == pytest.approx(2.1e11 * 1e-6 / (12 * 0.91), rel=RTOL_CLOSED)


def test_navier_module_imports_only_math_dataclasses_numpy():
    tree = ast.parse(NAVIER_SOURCE.read_text(encoding="utf-8"))
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            modules.add(node.module.split(".")[0])
    assert modules <= {"math", "dataclasses", "numpy", "kkvim_reference"}


# --- control checks at M = M_R3 = 2559 (few points, small grids) ---------------------------------

CONTROL_R = (0.2, 0.5, 1.0, 1.4, 2.0, 5.0)


@pytest.fixture(scope="module")
def control_centres():
    return {r: centre(plate(r), None) for r in CONTROL_R}


def test_default_level_is_m_r3(control_centres):
    for res in control_centres.values():
        assert res.M == M_R3
        assert isinstance(res.convergence, NavierRectangleConvergence)
        assert res.convergence.levels == NAVIER_LEVELS_R3
        assert not hasattr(res.convergence, "accepted_w")  # no acceptance rule for r != 1 (Р-5)


def test_control_bound_at_m_r3(control_centres):
    for r, res in control_centres.items():
        assert res.w_tail_bound_coef == deflection_tail_bound_rect(M_R3, r) <= 8.36e-9
        assert res.rounding_w < 1e-3 * res.w_tail_bound_coef


def test_control_convergence_over_levels(control_centres):
    # the last increment is covered by the strict bounds of its two levels
    for r, res in control_centres.items():
        d = res.convergence.delta_w
        assert d[-1] <= deflection_tail_bound_rect(1279, r) + deflection_tail_bound_rect(M_R3, r) + 2 * res.rounding_w
        assert d[-3] > d[-2] > d[-1] or d[-1] <= 2 * res.rounding_w


def test_control_square_equals_r2_plate_at_m_r3(control_centres):
    r2 = NavierPlate(a=1.0, q=1.0, D=1.0, nu=NU).at_points(0.5, 0.5, M=M_R3)
    assert np.array_equal(control_centres[1.0].w_coef, r2.w_coef)
    assert np.array_equal(control_centres[1.0].convergence.w, r2.convergence.w)


@pytest.mark.parametrize("r", [2.0, 5.0, 1.4])
def test_control_inverse_ratio_at_m_r3(control_centres, r):
    res = control_centres[r]
    inv = centre(plate(r).swapped(), M_R3)
    lhs, rhs = float(inv.w_coef), float(res.w_coef) / r**4
    assert abs(lhs - rhs) <= inv.rounding_w + res.rounding_w / r**4 + 4 * U * abs(lhs)


def test_control_r14_internal_checks():
    # r = 1.4: axis exchange, two summation orders, nodes against points - no table 8 criterion
    p = plate(1.4)
    pts = (np.array([0.5, 0.25, 0.5, 0.25]), np.array([0.7, 0.7, 0.35, 0.35]))
    u = p.at_points(*pts)
    v = p.swapped().at_points(pts[1], pts[0])
    assert np.max(np.abs(u.w - v.w)) <= abs(u.w_scale) * u.rounding_w + abs(v.w_scale) * v.rounding_w
    d = p.at_points(*pts, summation="diagonal", diagnostics=False)
    assert np.max(np.abs(u.w_coef - d.w_coef)) <= u.rounding_w + d.rounding_w
    nodes = p.at_nodes(4, 4, diagnostics=False)
    assert np.max(np.abs(nodes.w_coef[[2, 1, 2, 1], [2, 2, 1, 1]] - u.w_coef)) <= nodes.rounding_w + u.rounding_w


def test_control_integral_at_m_r3():
    for r in CONTROL_R:
        g = plate(r).deflection_integral()
        assert g.M == M_R3 and g.J_tail_bound == deflection_integral_tail_bound_rect(M_R3, r)
        assert g.J_tail_bound == pytest.approx(r**3 / (18 * math.pi**4 * M_R3**3), rel=RTOL_CLOSED)
        lower = plate(r).deflection_integral(M=M_R2).J_M
        assert -48 * U * g.J_M <= g.J_M - lower <= deflection_integral_tail_bound_rect(M_R2, r) + 48 * U * g.J_M


def test_control_grid_at_m_r3():
    # control grid r = 2: separable nodal evaluation at M = 2559
    res = plate(2.0).at_nodes(8, 16)
    assert res.w_coef.shape == (9, 17) and res.M == M_R3
    assert np.max(np.abs(res.w_coef - res.w_coef[::-1, ::-1])) <= res.rounding_w
    assert res.w_coef[4, 8] == np.max(res.w_coef)
