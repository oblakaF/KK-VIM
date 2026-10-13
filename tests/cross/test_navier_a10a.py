"""QA tests: independent qualification A-10a of the Navier reference (R2; D-034).

QA tests (D-033, item 8). The double series (131) and the moment series of
docs/NAVIER_REFERENCE_SPEC.md, section 5, are summed here with own code (centre by (132)
with the sign (-1)^((m+n)/2-1), points by an own loop, math.fsum); J_M by exact fractions.
No comparison with KK-VIM (that is group B, section 17 of NUMERICAL_SPEC.md).

Checks and tolerances (NAVIER_REFERENCE_SPEC.md 8.2, 8.4, 8.6, 9; NUMERICAL_SPEC.md 16; D-034):
- P-1 ... P-11; tau_tab = 5e-6 (deflection), 5e-5 (moments) for the printed tables;
- deflection: strict bound 8.2 at M = 1279 <= delta_w = 4.06e-9 (the basis of the qualification);
- moments: P-3, P-4 must pass; no strict delta_M guarantee is claimed (D-034, item 3);
- 8.4 item 3 at the designated control points (centre, midline x = 0.1a .. 0.5a); on dense grids
  non-monotone increments are diagnostics: they must not be reported as accepted, and
  Delta_last <= delta/10 is shown;
- J_M, I_M, Pi_M: values of the truncated series, remainder 0 <= J - J_M <= 1/(18 pi^4 M^3).
"""

import ast
import math
import pathlib
from fractions import Fraction
from functools import lru_cache

import numpy as np
import pytest

import kkvim_reference.navier as navier
from kkvim_reference.navier import (
    DELTA_M,
    DELTA_W,
    NavierPlate,
    deflection_integral_coef,
    deflection_integral_tail_bound,
    deflection_tail_bound,
)

U = 2.0**-53
PI = math.pi
TAU_W, TAU_M = 5e-6, 5e-5
PLATE = NavierPlate(a=1.0, q=1.0, D=1.0, nu=0.3)
MIDLINE_X = (0.1, 0.2, 0.3, 0.4, 0.5)
TABLE6_MX = (0.0209, 0.0343, 0.0424, 0.0466, 0.0479)
TABLE6_MY = (0.0168, 0.0303, 0.0400, 0.0459, 0.0479)


def sign_centre(m, n):
    return -1.0 if ((m + n) // 2 - 1) % 2 else 1.0


@lru_cache(maxsize=None)
def own_centre(M, nu=0.3):
    tw, tm = [], []
    for m in range(1, M + 1, 2):
        for n in range(1, M + 1, 2):
            den = m * n * (m * m + n * n) ** 2
            tw.append(sign_centre(m, n) / den)
            tm.append(sign_centre(m, n) * (m * m + nu * n * n) / den)
    return 16 / PI**6 * math.fsum(tw), 16 / PI**4 * math.fsum(tm)


def own_point(x, y, M, nu=0.3):
    tw, tx, ty = [], [], []
    for m in range(1, M + 1, 2):
        for n in range(1, M + 1, 2):
            s = math.sin(m * PI * x) * math.sin(n * PI * y)
            den = m * n * (m * m + n * n) ** 2
            tw.append(s / den)
            tx.append((m * m + nu * n * n) * s / den)
            ty.append((nu * m * m + n * n) * s / den)
    return 16 / PI**6 * math.fsum(tw), 16 / PI**4 * math.fsum(tx), 16 / PI**4 * math.fsum(ty)


# P-1 ... P-11 --------------------------------------------------------------------------------


def test_p1_single_term():
    r = PLATE.at_points(0.5, 0.5, M=1)
    assert abs(float(r.w_coef) - 4 / PI**6) <= 4 * U * 4 / PI**6
    moment = 4 * (1 + 0.3) / PI**4  # 16/pi^4 (1 + nu) / (1 * 1 * 4)
    assert abs(float(r.mx_coef) - moment) <= 4 * U * moment and abs(float(r.my_coef) - moment) <= 4 * U * moment


def test_p2_p3_centre_against_own_sum_and_tables():
    r = PLATE.at_points(0.5, 0.5)  # M = 1279
    w_own, m_own = own_centre(1279)
    assert abs(float(r.w_coef) - w_own) <= 1e-15
    assert abs(float(r.mx_coef) - m_own) <= 1e-13 and abs(float(r.my_coef) - m_own) <= 1e-13
    assert abs(float(r.w_coef) - 0.00406) <= TAU_W
    assert abs(float(r.mx_coef) - 0.0479) <= TAU_M and abs(float(r.my_coef) - 0.0479) <= TAU_M


def test_p4_midline_table6():
    r = PLATE.at_points(np.array(MIDLINE_X), np.full(5, 0.5))  # y = a/2 in Navier coordinates
    assert np.max(np.abs(r.mx_coef - np.array(TABLE6_MX))) <= TAU_M
    assert np.max(np.abs(r.my_coef - np.array(TABLE6_MY))) <= TAU_M
    for x in (0.1, 0.3):
        _, mx, my = own_point(x, 0.5, 159)
        low = PLATE.at_points(x, 0.5, M=159)
        assert abs(float(low.mx_coef) - mx) <= 1e-14 and abs(float(low.my_coef) - my) <= 1e-14


def test_p5_maximum_at_centre():
    w = PLATE.at_nodes(20, 20).w_coef
    assert w[10, 10] == w.max() and np.sum(w == w.max()) == 1


def test_p6_edges_and_continuity():
    t = np.linspace(0.0, 1.0, 21)
    zero, one = np.zeros(21), np.ones(21)
    for x, y, normal in ((zero, t, "mx_coef"), (one, t, "mx_coef"), (t, zero, "my_coef"), (t, one, "my_coef")):
        r = PLATE.at_points(x, y)
        assert not r.w_coef.any() and not getattr(r, normal).any()
    near = PLATE.at_points(np.full(21, 1e-7), t)
    assert np.max(np.abs(near.w_coef)) <= 5e-9 and np.max(np.abs(near.mx_coef)) <= 1e-6


def test_p7_symmetry():
    rng = np.random.default_rng(3)
    x, y = rng.uniform(0, 1, 30), rng.uniform(0, 1, 30)
    a = PLATE.at_points(x, y, M=159)
    swap = PLATE.at_points(y, x, M=159)
    mirror = PLATE.at_points(1 - x, y, M=159)
    assert np.max(np.abs(a.w_coef - swap.w_coef)) <= 1e-16 and np.max(np.abs(a.w_coef - mirror.w_coef)) <= 1e-16
    assert np.max(np.abs(a.mx_coef - swap.my_coef)) <= 1e-15 and np.max(np.abs(a.mx_coef - mirror.mx_coef)) <= 1e-15
    nodes = PLATE.at_nodes(24, 24, M=159)
    assert np.max(np.abs(nodes.w_coef - nodes.w_coef.T)) <= 1e-16
    assert np.max(np.abs(nodes.mx_coef - nodes.my_coef.T)) <= 1e-15


def test_p8_scaling():
    rng = np.random.default_rng(5)
    x, y = rng.uniform(0, 1, 20), rng.uniform(0, 1, 20)
    unit = PLATE.at_points(x, y, M=159)
    a, q, D = 2.7, -35.0, 0.013
    other = NavierPlate(a=a, q=q, D=D, nu=0.3).at_points(a * x, a * y, M=159)
    assert np.max(np.abs(other.w / (q * a**4 / D) - unit.w_coef)) <= 1e-13 * np.max(np.abs(unit.w_coef))
    assert np.max(np.abs(other.Mx / (q * a**2) - unit.mx_coef)) <= 1e-13 * np.max(np.abs(unit.mx_coef))


def test_p9_poisson_ratio():
    base = PLATE.at_points(0.5, 0.5)
    for nu in (0.0, 0.45):
        r = NavierPlate(a=1.0, q=1.0, D=1.0, nu=nu).at_points(0.5, 0.5)
        assert float(r.w_coef) == float(base.w_coef)
        assert abs(float(r.mx_coef) / float(base.mx_coef) - (1 + nu) / 1.3) <= 1e-14


def test_p10_packages_do_not_import_each_other():
    src = pathlib.Path(navier.__file__).parents[1]
    for package, forbidden in (("kkvim_reference", "kkvim"), ("kkvim", "kkvim_reference")):
        for path in (src / package).rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    names = [node.module]
                else:
                    continue
                assert all(name.split(".")[0] != forbidden for name in names), (path, names)


def test_p11_two_summation_orders():
    rng = np.random.default_rng(11)
    x, y = rng.uniform(0, 1, 15), rng.uniform(0, 1, 15)
    s = PLATE.at_points(x, y, M=159, summation="separable")
    d = PLATE.at_points(x, y, M=159, summation="diagonal")
    assert np.max(np.abs(s.w_coef - d.w_coef)) <= s.rounding_w + d.rounding_w
    assert np.max(np.abs(s.mx_coef - d.mx_coef)) <= s.rounding_mx + d.rounding_mx
    ns = PLATE.at_nodes(12, 10, M=79, summation="separable")
    nd = PLATE.at_nodes(12, 10, M=79, summation="diagonal")
    assert np.max(np.abs(ns.w_coef - nd.w_coef)) <= ns.rounding_w + nd.rounding_w


def test_points_and_nodes_are_direct_sums_of_odd_harmonics():
    r = PLATE.at_points(0.3, 0.7, M=3)
    four = 16 / PI**6 * sum(math.sin(m * PI * 0.3) * math.sin(n * PI * 0.7) / (m * n * (m * m + n * n) ** 2) for m in (1, 3) for n in (1, 3))
    assert abs(float(r.w_coef) - four) <= 4 * U * abs(four)
    nodes = PLATE.at_nodes(8, 6, M=39)
    for i, j in ((1, 1), (3, 2), (5, 4)):
        assert abs(nodes.w_coef[i, j] - own_point(i / 8, j / 6, 39)[0]) <= 1e-16


def test_zero_and_negative_load():
    z = NavierPlate(a=1.0, q=0.0, D=1.0, nu=0.3).at_points(0.4, 0.6)
    assert float(z.w) == 0.0 and float(z.Mx) == 0.0
    pos = NavierPlate(a=1.0, q=2.0, D=1.0, nu=0.3).at_points(0.4, 0.6)
    neg = NavierPlate(a=1.0, q=-2.0, D=1.0, nu=0.3).at_points(0.4, 0.6)
    assert float(neg.w) == -float(pos.w) and float(neg.My) == -float(pos.My)


@pytest.mark.parametrize("change", [{"a": 0.0}, {"D": -1.0}, {"nu": 1.0}, {"q": math.nan}, {"b": 2.0}])
def test_invalid_plate(change):
    with pytest.raises(ValueError):
        NavierPlate(**{**dict(a=1.0, q=1.0, D=1.0, nu=0.3), **change})


# Deflection: strict bound 8.2 (basis of the qualification, D-034) ------------------------------


def test_strict_deflection_bound_at_1279():
    lambda3 = 7 * 1.2020569031595942 / 8
    own = 16 / PI**6 * lambda3 / (8 * 1279**2)
    assert abs(deflection_tail_bound(1279) - own) <= 1e-15 * own
    assert own <= DELTA_W
    assert 16 / PI**6 * lambda3 / (8 * 639**2) > DELTA_W  # the previous level is not enough


# 8.4: control points mandatory, dense grids diagnostic (D-034, item 4) ---------------------------


def test_84_control_points():
    r = PLATE.at_points(np.array(MIDLINE_X), np.full(5, 0.5))
    c = r.convergence
    for deltas, delta in ((c.delta_w, DELTA_W), (c.delta_mx, DELTA_M), (c.delta_my, DELTA_M)):
        d3, d2, d1 = deltas[-3], deltas[-2], deltas[-1]
        assert np.all(d3 > d2) and np.all(d2 > d1) and np.all(d1 <= delta / 10)
    assert np.all(c.accepted_w()) and np.all(c.accepted_mx()) and np.all(c.accepted_my())


@pytest.mark.parametrize("n", [160, 320])
def test_84_dense_grid_diagnostics_are_not_false_pass(n):
    c = PLATE.at_nodes(n, n).convergence
    interior = np.zeros((n + 1, n + 1), dtype=bool)
    interior[1:-1, 1:-1] = True
    for deltas, delta, accepted in ((c.delta_w, DELTA_W, c.accepted_w()), (c.delta_mx, DELTA_M, c.accepted_mx())):
        d3, d2, d1 = deltas[-3], deltas[-2], deltas[-1]
        non_monotone = interior & ~((d3 >= d2) & (d2 >= d1))
        assert non_monotone.any()  # the documented diagnostic exists
        assert not accepted[non_monotone].any()  # and is not reported as accepted
        assert np.max(d1[interior]) <= delta / 10  # the last increment is small everywhere


# Integral and energy ---------------------------------------------------------------------------


def test_integral_small_levels_exact():
    assert abs(deflection_integral_coef(1) - 16 / PI**8) <= 4 * U * 16 / PI**8
    for M in (3, 9, 19):
        s = sum(Fraction(1, m * m * n * n * (m * m + n * n) ** 2) for m in range(1, M + 1, 2) for n in range(1, M + 1, 2))
        exact = 64 / PI**8 * (s.numerator / s.denominator)
        assert abs(deflection_integral_coef(M) - exact) <= 8 * U * exact


def test_integral_1279_and_remainder():
    k = np.arange(1, 1280, 2, dtype=float)
    rows = [math.fsum(1.0 / (m * m * k**2 * (m * m + k**2) ** 2)) for m in k]  # row-wise, other organisation
    own = 64 / PI**8 * math.fsum(reversed(rows))
    J = deflection_integral_coef(1279)
    assert abs(J - own) <= 12 * U * own
    bound = 1 / (18 * PI**4 * 1279**3)
    assert abs(deflection_integral_tail_bound(1279) - bound) <= 1e-15 * bound
    k5 = np.arange(1, 2560, 2, dtype=float)
    J_more = 64 / PI**8 * math.fsum(math.fsum(1.0 / (m * m * k5**2 * (m * m + k5**2) ** 2)) for m in k5)
    # the true remainder (~6.5e-19) is at the rounding level of J itself (u J ~ 4e-19): lower side up to rounding
    assert -16 * U * J <= J_more - J <= bound


@pytest.mark.parametrize("a, D, q", [(1.0, 1.0, 1.0), (2.3, 0.7, 4.1), (0.6, 13.0, -2.5)])
def test_energy_scaling_signs_and_bounds(a, D, q):
    g = NavierPlate(a=a, q=q, D=D, nu=0.2).deflection_integral()
    J, bound, scale = deflection_integral_coef(1279), deflection_integral_tail_bound(1279), q * a**6 / D
    assert abs(g.I_M - scale * J) <= 4 * U * abs(scale * J)
    assert abs(g.Pi_M + 0.5 * q * scale * J) <= 4 * U * abs(q * scale * J)
    assert g.Pi_M <= 0.0 and np.sign(g.I_M) == np.sign(q)
    assert abs(g.I_tail_bound - abs(scale) * bound) <= 4 * U * abs(scale) * bound
    assert abs(g.Pi_tail_bound - 0.5 * abs(q * scale) * bound) <= 4 * U * abs(q * scale) * bound


def test_energy_monotone_in_m_and_zero_load():
    levels = (1, 3, 9, 19, 39, 79, 159, 319, 639, 1279)
    J = [deflection_integral_coef(M) for M in levels]
    Pi = [NavierPlate(a=1.0, q=1.7, D=1.0, nu=0.3).deflection_integral(M=M).Pi_M for M in levels]
    assert all(a < b for a, b in zip(J, J[1:])) and all(a > b for a, b in zip(Pi, Pi[1:]))
    z = NavierPlate(a=1.0, q=0.0, D=1.0, nu=0.3).deflection_integral()
    assert z.I_M == 0.0 and z.Pi_M == 0.0


def test_trapezoid_of_truncated_series_converges_to_i_m():
    I = PLATE.deflection_integral(M=79).I_M
    errors = {n: abs(PLATE.at_nodes(n, n, M=79, diagnostics=False).w_coef.sum() / n**2 - I) / I for n in (80, 160, 320)}
    for n in (80, 160):
        assert 1.9 <= math.log2(errors[n] / errors[2 * n]) <= 2.1
