"""Own tests of kkvim_reference.navier (R2, independent Navier reference, A-10a).

Checks П-1 ... П-11 of docs/NAVIER_REFERENCE_SPEC.md, section 9, plus the
truncation levels and bounds of section 8, the practical control 8.4, the
load sign, input validation and the full nodal field. No KK-VIM results are
used; П-12 (Levy cross-check) is not part of A-10a (NUMERICAL_SPEC.md, 16).

Tolerances, fixed before any comparison (not tuned to results):
- TAU_TAB_W = 5e-6 for the printed deflection coefficient 0.00406 and
  TAU_TAB_M = 5e-5 for the printed moment coefficients (NAVIER_REFERENCE_SPEC.md,
  8.6: half a unit of the last printed digit); a check of the formulas against
  rounded tables, not the accuracy of the reference;
- DELTA_W = 4.06e-9 q a^4/D, DELTA_M = 4.79e-6 q a^2 - accuracy of the reference
  (8.6), from the module;
- rounding: the first-order bounds returned by the module (rounding_w,
  rounding_mx, rounding_my); two evaluations of the same truncated series must
  agree within the sum of their bounds;
- RTOL_CLOSED = 16 U for closed forms of a few operations (4/pi^6, D, scales).
- integral J_M (MATH_SPEC.md, В-27, item 6; NUMERICAL_SPEC.md, 16): against an
  independent exact-rational sum (fractions.Fraction, M <= 19) and a sum by
  columns of correctly rounded terms (M = 1279) within the module bound
  rounding_J plus 4 U J; trapezoidal quadrature of the nodal field of the same
  truncated series (M = 39, all m < 2n) on n = 20 ... 320: the error must
  decrease with every doubling and the ratio of successive errors must lie in
  [3.9, 4.1] (second order of the trapezoidal rule for the smooth truncated
  series; convergence is checked, not equality on a finite grid).
"""

import ast
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from kkvim_reference.navier import (
    DELTA_M,
    DELTA_W,
    M_R2,
    NAVIER_LEVELS,
    NavierPlate,
    deflection_integral_coef,
    deflection_integral_tail_bound,
    deflection_tail_bound,
    deflection_tail_bound_refined,
    diagnostic_levels,
    flexural_rigidity,
    moment_tail_bound,
    moment_tail_bound_refined,
    practical_control,
    select_level,
)

U = 2.0**-53
RTOL_CLOSED = 16 * U
TAU_TAB_W = 5e-6
TAU_TAB_M = 5e-5
ROUND_J = 12 * U  # rounding of J_M relative to J_M (module model)

# L-107, p. 137 and table 8, p. 143 (b/a = 1.0): alpha; table 8 and table 6,
# p. 139 (nu = 0.3): beta = beta_1.
ALPHA_TAB = 0.00406
BETA_TAB = 0.0479
# L-107, table 6, p. 139, nu = 0.3, b/a = 1.0, line y = 0 of section 30
# (y = b/2 in Navier coordinates): x/a -> (M_x, M_y) / (q a^2).
TABLE6 = {
    0.1: (0.0209, 0.0168),
    0.2: (0.0343, 0.0303),
    0.3: (0.0424, 0.0400),
    0.4: (0.0466, 0.0459),
    0.5: (0.0479, 0.0479),
}

NAVIER_SOURCE = Path(__file__).resolve().parents[2] / "src" / "kkvim_reference" / "navier.py"


@pytest.fixture(scope="module")
def unit_plate():
    return NavierPlate(a=1.0, q=1.0, D=1.0, nu=0.3)


@pytest.fixture(scope="module")
def steel_plate():
    # Dimensional input (D-009): a in m, q in Pa, E in Pa, thickness in m.
    return NavierPlate.from_material(a=2.5, q=1.3e4, E=2.1e11, thickness=0.012, nu=0.3)


@pytest.fixture(scope="module")
def control(unit_plate):
    """Centre, quarter points and the points of table 6 at M = 1279 with levels."""
    x = np.array([0.5, 0.25, 0.75, 0.25, 0.75, 0.25, 0.1, 0.2, 0.3, 0.4])
    y = np.array([0.5, 0.25, 0.25, 0.75, 0.75, 0.5, 0.5, 0.5, 0.5, 0.5])
    return unit_plate.at_points(x, y)


def random_points(count, seed):
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 1.0, count), rng.uniform(0.0, 1.0, count)


def assert_within(a, b, tol):
    assert np.max(np.abs(np.asarray(a) - np.asarray(b))) <= tol


# --- П-1 -------------------------------------------------------------------


def test_p1_single_term_centre(unit_plate, steel_plate):
    r = unit_plate.at_points(0.5, 0.5, M=1)
    assert r.w_coef == pytest.approx(4 / math.pi**6, rel=RTOL_CLOSED, abs=0)
    assert float(r.w_coef) == pytest.approx(0.00416, abs=5e-6)  # printed 0.00416 (p. 130)
    # structure of (f), p. 125 with a11 = 16 q / pi^2: M = 4 (1 + nu) q a^2 / pi^4
    assert r.mx_coef == pytest.approx(4 * 1.3 / math.pi**4, rel=RTOL_CLOSED, abs=0)
    assert r.my_coef == pytest.approx(4 * 1.3 / math.pi**4, rel=RTOL_CLOSED, abs=0)
    s = steel_plate.at_points(1.25, 1.25, M=1)
    expected = 4 * steel_plate.q * steel_plate.a**4 / (math.pi**6 * steel_plate.D)
    assert s.w == pytest.approx(expected, rel=RTOL_CLOSED, abs=0)


def test_p1_printed_example_with_e_and_h():
    # p. 130: w_max = 0.0454 q a^4 / (E h^3) at nu = 0.3 for the single term.
    plate = NavierPlate.from_material(a=1.0, q=1.0, E=1.0, thickness=1.0, nu=0.3)
    r = plate.at_points(0.5, 0.5, M=1)
    assert float(r.w) == pytest.approx(0.0454, abs=5e-5)


# --- П-2, П-3, П-4 ----------------------------------------------------------


def test_p2_centre_deflection_against_table(control):
    alpha = float(control.w_coef[0])
    assert control.M == M_R2
    assert abs(alpha - ALPHA_TAB) <= TAU_TAB_W
    assert control.w_error_bound_coef <= DELTA_W


def test_p3_centre_moments_against_table(control):
    beta, beta1 = float(control.mx_coef[0]), float(control.my_coef[0])
    assert abs(beta - BETA_TAB) <= TAU_TAB_M
    assert abs(beta1 - BETA_TAB) <= TAU_TAB_M
    assert abs(beta - beta1) <= control.rounding_mx + control.rounding_my


def test_p4_moments_along_middle_line_against_table6(unit_plate):
    xs = sorted(TABLE6)
    r = unit_plate.at_points(xs, 0.5)
    for k, x in enumerate(xs):
        mx_tab, my_tab = TABLE6[x]
        assert abs(r.mx_coef[k] - mx_tab) <= TAU_TAB_M, x
        assert abs(r.my_coef[k] - my_tab) <= TAU_TAB_M, x


# --- П-5 ---------------------------------------------------------------------


def test_p5_maximum_at_centre(unit_plate):
    steps = np.arange(21) * 0.05
    gx, gy = np.meshgrid(steps, steps, indexing="ij")
    r = unit_plate.at_points(gx, gy, diagnostics=False)
    centre = r.w_coef[10, 10]
    others = np.delete(r.w_coef.ravel(), 10 * 21 + 10)
    assert np.all(others < centre)
    g = unit_plate.at_nodes(20, 20, diagnostics=False)
    assert np.unravel_index(np.argmax(g.w_coef), g.w_coef.shape) == (10, 10)


# --- П-6 ---------------------------------------------------------------------


@pytest.mark.parametrize("M", [1, 9, M_R2])
def test_p6_boundary_values_exactly_zero(unit_plate, M):
    # Float sin(m pi) is not zero; the zeros come from the rule of the module.
    assert math.sin(math.pi) != 0.0
    t = np.linspace(0.0, 1.0, 41)
    zero, one = np.zeros_like(t), np.ones_like(t)
    for x, y in ((zero, t), (one, t), (t, zero), (t, one)):
        r = unit_plate.at_points(x, y, M=M)
        for field in (r.w_coef, r.mx_coef, r.my_coef, r.w, r.Mx, r.My):
            assert np.all(field == 0.0)
        for deltas in (r.convergence.delta_w, r.convergence.delta_mx, r.convergence.delta_my):
            assert np.all(deltas == 0.0)
    g = unit_plate.at_nodes(16, 12, M=M)
    for field in (g.w_coef, g.mx_coef, g.my_coef):
        assert np.all(field[[0, -1], :] == 0.0)
        assert np.all(field[:, [0, -1]] == 0.0)


def test_p6_moment_normal_to_each_edge(unit_plate):
    t = np.linspace(0.0, 1.0, 11)
    for x in (0.0, 1.0):
        assert np.all(unit_plate.at_points(x, t).Mx == 0.0)  # M_n on x = 0, a
    for y in (0.0, 1.0):
        assert np.all(unit_plate.at_points(t, y).My == 0.0)  # M_n on y = 0, a


def test_p6_values_near_edge_are_computed_not_masked(unit_plate):
    r = unit_plate.at_points([1e-9, 1.0 - 1e-9], [0.5, 0.5])
    assert np.all(r.w_coef > 0.0)
    # slope of w at the edge from the series: w ~ x * dw/dx(0, a/2)
    assert r.w_coef[0] == pytest.approx(r.w_coef[1], rel=1e-6)


# --- П-7 ---------------------------------------------------------------------


def test_p7_symmetry_at_points(unit_plate):
    x, y = random_points(40, seed=7)
    r = unit_plate.at_points(x, y)
    rt = unit_plate.at_points(y, x)
    rx = unit_plate.at_points(1.0 - x, y)
    ry = unit_plate.at_points(x, 1.0 - y)
    for other in (rt, rx, ry):
        assert_within(r.w_coef, other.w_coef, 2 * r.rounding_w)
    assert_within(r.mx_coef, rt.my_coef, r.rounding_mx + r.rounding_my)
    assert_within(r.my_coef, rt.mx_coef, r.rounding_mx + r.rounding_my)
    for other in (rx, ry):
        assert_within(r.mx_coef, other.mx_coef, 2 * r.rounding_mx)
        assert_within(r.my_coef, other.my_coef, 2 * r.rounding_my)


def test_p7_symmetry_at_nodes(unit_plate):
    g = unit_plate.at_nodes(40, 40)
    w, mx, my = g.w_coef, g.mx_coef, g.my_coef
    assert_within(w, w.T, 2 * g.rounding_w)
    assert_within(w, w[::-1, :], 2 * g.rounding_w)
    assert_within(w, w[:, ::-1], 2 * g.rounding_w)
    assert_within(mx, my.T, g.rounding_mx + g.rounding_my)
    assert_within(mx, mx[::-1, :], 2 * g.rounding_mx)


def test_p7_quarter_points_equal(control):
    # (a/4, a/4), (3a/4, a/4), (a/4, 3a/4), (3a/4, 3a/4)
    w = control.w_coef[1:5]
    assert_within(w, w[0], 2 * control.rounding_w)
    assert_within(control.mx_coef[1:5], control.my_coef[1:5], control.rounding_mx + control.rounding_my)


# --- П-8 ---------------------------------------------------------------------


def test_p8_scaling(unit_plate):
    big = NavierPlate(a=2.9, q=3.7, D=0.013, nu=0.3)
    x, y = random_points(25, seed=8)
    r1 = unit_plate.at_points(x, y, diagnostics=False)
    r2 = big.at_points(x * big.a, y * big.a, diagnostics=False)
    # x * a / a may differ from x by one rounding: the rounding bounds cover it.
    assert_within(r2.w / (big.q * big.a**4 / big.D), r1.w_coef, 2 * r1.rounding_w + RTOL_CLOSED * 0.005)
    assert_within(r2.Mx / (big.q * big.a**2), r1.mx_coef, 2 * r1.rounding_mx + RTOL_CLOSED * 0.06)
    assert_within(r2.My / (big.q * big.a**2), r1.my_coef, 2 * r1.rounding_my + RTOL_CLOSED * 0.06)
    # at grid nodes the dimensionless field does not depend on a, q, D at all
    g1 = unit_plate.at_nodes(10, 14, diagnostics=False)
    g2 = big.at_nodes(10, 14, diagnostics=False)
    assert np.array_equal(g1.w_coef, g2.w_coef)
    assert np.array_equal(g1.mx_coef, g2.mx_coef)
    assert np.allclose(g2.w, g1.w_coef * big.q * big.a**4 / big.D, rtol=RTOL_CLOSED, atol=0)


# --- П-9 ---------------------------------------------------------------------


@pytest.mark.parametrize("M", [1, M_R2])
def test_p9_deflection_independent_of_nu(M):
    x, y = random_points(10, seed=9)
    base = NavierPlate(a=1.0, q=1.0, D=1.0, nu=0.3).at_points(x, y, M=M, diagnostics=False)
    for nu in (0.0, 0.15, 0.45):
        r = NavierPlate(a=1.0, q=1.0, D=1.0, nu=nu).at_points(x, y, M=M, diagnostics=False)
        assert np.array_equal(r.w_coef, base.w_coef)
    # with D from E, h, nu the dimensionless deflection is still the same
    for nu in (0.0, 0.3, 0.45):
        plate = NavierPlate.from_material(a=1.0, q=1.0, E=7.0e10, thickness=0.01, nu=nu)
        r = plate.at_points(x, y, M=M, diagnostics=False)
        assert_within(r.w * plate.D / (plate.q * plate.a**4), base.w_coef, RTOL_CLOSED * 0.005)


@pytest.mark.parametrize("M", [1, M_R2])
def test_p9_centre_moments_proportional_to_one_plus_nu(M):
    ref = NavierPlate(a=1.0, q=1.0, D=1.0, nu=0.3).at_points(0.5, 0.5, M=M, diagnostics=False)
    for nu in (0.0, 0.1, 0.25, 0.45):
        r = NavierPlate(a=1.0, q=1.0, D=1.0, nu=nu).at_points(0.5, 0.5, M=M, diagnostics=False)
        expected = ref.mx_coef * (1 + nu) / 1.3
        tol = r.rounding_mx + ref.rounding_mx * (1 + nu) / 1.3 + RTOL_CLOSED * 0.06
        assert abs(r.mx_coef - expected) <= tol
        assert abs(r.my_coef - expected) <= tol
        if M == M_R2:
            # beta(nu) = 0.0479 (1 + nu) / 1.3 within the scaled table tolerance
            assert abs(r.mx_coef - BETA_TAB * (1 + nu) / 1.3) <= TAU_TAB_M * (1 + nu) / 1.3


# --- П-10 --------------------------------------------------------------------


def test_p10_navier_imports_only_math_dataclasses_numpy():
    tree = ast.parse(NAVIER_SOURCE.read_text(encoding="utf-8"))
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0
            modules.add(node.module.split(".")[0])
        elif isinstance(node, ast.Call):
            name = getattr(node.func, "attr", getattr(node.func, "id", None))
            assert name not in ("import_module", "__import__")
    assert modules <= {"math", "dataclasses", "numpy"}
    assert "kkvim" not in modules


# --- П-11 --------------------------------------------------------------------


def test_p11_two_summation_orders_at_points(unit_plate, control):
    x, y = random_points(30, seed=11)
    x = np.concatenate([control.x, x])
    y = np.concatenate([control.y, y])
    sep = unit_plate.at_points(x, y, diagnostics=False)
    diag = unit_plate.at_points(x, y, summation="diagonal", diagnostics=False)
    assert_within(sep.w_coef, diag.w_coef, sep.rounding_w + diag.rounding_w)
    assert_within(sep.mx_coef, diag.mx_coef, sep.rounding_mx + diag.rounding_mx)
    assert_within(sep.my_coef, diag.my_coef, sep.rounding_my + diag.rounding_my)


def test_p11_two_summation_orders_at_nodes(unit_plate):
    sep = unit_plate.at_nodes(20, 20, diagnostics=False)
    diag = unit_plate.at_nodes(20, 20, summation="diagonal", diagnostics=False)
    assert_within(sep.w_coef, diag.w_coef, sep.rounding_w + diag.rounding_w)
    assert_within(sep.mx_coef, diag.mx_coef, sep.rounding_mx + diag.rounding_mx)
    assert_within(sep.my_coef, diag.my_coef, sep.rounding_my + diag.rounding_my)


def test_p11_against_compensated_sum_at_centre(unit_plate):
    # Terms of (131) at the centre written out directly, summed by math.fsum:
    # s_m(a/2) = (-1)^((m - 1)/2) exactly.
    M = M_R2
    odd = range(1, M + 1, 2)
    terms = [
        (-1) ** ((m + n) // 2 - 1) / (m * n * (m * m + n * n) ** 2)
        for m in odd
        for n in odd
    ]
    exact = 16 / math.pi**6 * math.fsum(terms)
    r = unit_plate.at_points(0.5, 0.5, diagnostics=False)
    d = unit_plate.at_points(0.5, 0.5, summation="diagonal", diagnostics=False)
    tol_fsum = RTOL_CLOSED * 0.005
    assert abs(r.w_coef - exact) <= r.rounding_w + tol_fsum
    assert abs(d.w_coef - exact) <= d.rounding_w + tol_fsum


# --- truncation levels and bounds (section 8) ---------------------------------


def test_levels_and_default_level():
    assert NAVIER_LEVELS == (1, 3, 9, 19, 39, 79, 159, 319, 639, 1279)
    assert M_R2 == 1279
    assert diagnostic_levels(M_R2) == NAVIER_LEVELS
    assert diagnostic_levels(101) == (1, 3, 9, 19, 39, 79, 101)
    assert diagnostic_levels(1) == (1,)


def test_conservative_deflection_bound_and_threshold():
    # 8.2: (16/pi^6) lambda_3 / 8 = 0.0021881; M >= 735; first level 1279.
    assert deflection_tail_bound(1) == pytest.approx(0.0021881, rel=5e-5)
    lam3 = math.fsum(n**-3.0 for n in range(1, 200001, 2))
    assert lam3 <= 7 * 1.2020569031595942 / 8 <= lam3 + 1 / (4 * 200001**2)
    assert deflection_tail_bound(735) <= DELTA_W < deflection_tail_bound(733)
    assert deflection_tail_bound(1279) <= DELTA_W < deflection_tail_bound(639)
    assert select_level(DELTA_W) == 1279
    assert select_level(1e-6) == 79  # 0.0021881 / M^2 <= 1e-6 -> M >= 46.8
    for M, printed in ((79, 3.5e-7), (159, 8.7e-8), (639, 5.4e-9), (1279, 1.3e-9)):
        assert deflection_tail_bound(M) == pytest.approx(printed, rel=0.04)


def test_refined_and_moment_bounds_thresholds():
    # NAVIER_REFERENCE_SPEC.md 8.2, 8.3; recheck, section 4.
    assert deflection_tail_bound_refined(49) <= DELTA_W < deflection_tail_bound_refined(47)
    assert moment_tail_bound_refined(369) <= DELTA_M < moment_tail_bound_refined(367)
    assert moment_tail_bound(1) == pytest.approx(1 / math.pi**2, rel=RTOL_CLOSED)
    # the rough moment bound does not reach DELTA_M at the R2 level (8.3)
    assert moment_tail_bound(1279) == pytest.approx(7.9e-5, rel=0.01)
    assert moment_tail_bound(1279) > DELTA_M
    assert moment_tail_bound_refined(1279) == pytest.approx(4.6e-7, rel=0.02)
    assert deflection_tail_bound_refined(1279) == pytest.approx(1.3e-14, rel=0.05)


def test_bounds_cover_differences_of_levels(unit_plate, control):
    g = unit_plate.at_nodes(40, 40)
    for r in (control, g):
        conv = r.convergence
        for j, M in enumerate(conv.levels[:-1]):
            dw = np.abs(conv.w[j] - conv.w[-1])
            dm = np.maximum(np.abs(conv.mx[j] - conv.mx[-1]), np.abs(conv.my[j] - conv.my[-1]))
            assert np.all(dw <= deflection_tail_bound(M) + deflection_tail_bound(M_R2) + 2 * r.rounding_w)
            assert np.all(dm <= moment_tail_bound(M) + moment_tail_bound(M_R2) + 2 * r.rounding_mx)
            assert np.all(
                dm <= moment_tail_bound_refined(M) + moment_tail_bound_refined(M_R2) + 2 * r.rounding_mx
            )


def test_rounding_bounds_are_small(control):
    assert control.rounding_w <= 1e-14
    assert control.rounding_mx <= 1e-13
    assert control.w_error_bound_coef == control.w_tail_bound_coef + control.rounding_w


# --- practical control (8.4) --------------------------------------------------


def test_practical_control_at_control_points(control):
    conv = control.convergence
    assert conv.levels == NAVIER_LEVELS
    assert np.all(conv.accepted_w())
    assert np.all(conv.accepted_mx())
    assert np.all(conv.accepted_my())
    assert np.all(conv.delta_w[-1] <= DELTA_W / 10)
    assert np.all(conv.delta_mx[-1] <= DELTA_M / 10)
    assert conv.ratios(conv.delta_w).shape == (len(NAVIER_LEVELS) - 2, control.x.size)


def test_practical_control_rule():
    good = np.array([1e-3, 1e-5, 1e-7, 1e-8])
    assert practical_control(good, 1e-6)
    assert not practical_control(np.array([1e-5, 1e-8, 1e-7]), 1e-6)  # not monotone
    assert not practical_control(np.array([1e-5, 1e-6, 2e-7]), 1e-6)  # above delta/10
    assert not practical_control(np.array([1e-6, 1e-8]), 1e-6)  # fewer than three increments
    assert not practical_control(good, 1e-6, strict_bound=2e-6)  # strict bound above delta
    assert practical_control(np.zeros(5), 1e-6)  # exact zeros on the edges


def test_practical_control_needs_three_increments(unit_plate):
    r = unit_plate.at_points(0.5, 0.5, M=9)
    assert r.convergence.levels == (1, 3, 9)
    assert not r.convergence.accepted_w()


# --- load sign ----------------------------------------------------------------


def test_positive_and_negative_load():
    x, y = random_points(12, seed=12)
    up = NavierPlate(a=1.7, q=2.5e3, D=40.0, nu=0.3).at_points(1.7 * x, 1.7 * y)
    down = NavierPlate(a=1.7, q=-2.5e3, D=40.0, nu=0.3).at_points(1.7 * x, 1.7 * y)
    assert np.array_equal(up.w_coef, down.w_coef)
    assert np.array_equal(down.w, -up.w)
    assert np.array_equal(down.Mx, -up.Mx)
    assert np.array_equal(down.My, -up.My)
    assert np.all(up.w > 0) and np.all(down.w < 0)
    assert down.w_tail_bound == up.w_tail_bound > 0
    zero = NavierPlate(a=1.0, q=0.0, D=1.0, nu=0.3).at_points(0.5, 0.5)
    assert zero.w == 0.0 and zero.Mx == 0.0 and zero.w_coef > 0


# --- input --------------------------------------------------------------------


def test_flexural_rigidity():
    D = flexural_rigidity(2.1e11, 0.012, 0.3)
    assert D == pytest.approx(2.1e11 * 0.012**3 / (12 * (1 - 0.09)), rel=RTOL_CLOSED, abs=0)
    plate = NavierPlate.from_material(a=1.0, q=1.0, E=2.1e11, thickness=0.012, nu=0.3)
    assert plate.D == D


@pytest.mark.parametrize(
    "kwargs",
    [
        {"a": 0.0},
        {"a": -1.0},
        {"a": math.nan},
        {"a": True},
        {"q": math.inf},
        {"q": math.nan},
        {"D": 0.0},
        {"D": -2.0},
        {"nu": 1.0},
        {"nu": -1.0},
        {"nu": math.nan},
        {"b": 2.0},
        {"b": 0.0},
    ],
)
def test_invalid_plate(kwargs):
    args = {"a": 1.0, "q": 1.0, "D": 1.0, "nu": 0.3} | kwargs
    with pytest.raises(ValueError):
        NavierPlate(**args)


def test_square_b_equal_a_accepted():
    assert NavierPlate(a=2.0, q=1.0, D=1.0, nu=0.3, b=2.0).b == 2.0


@pytest.mark.parametrize(
    "kwargs",
    [{"E": 0.0}, {"E": -1.0}, {"thickness": 0.0}, {"thickness": math.nan}, {"nu": 1.0}],
)
def test_invalid_material(kwargs):
    args = {"a": 1.0, "q": 1.0, "E": 2.1e11, "thickness": 0.01, "nu": 0.3} | kwargs
    with pytest.raises(ValueError):
        NavierPlate.from_material(**args)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"M": 0},
        {"M": 2},
        {"M": -3},
        {"M": 3.0},
        {"M": True},
        {"M": 9, "w_tol": 1e-6},
        {"w_tol": 0.0},
        {"w_tol": -1e-9},
        {"w_tol": 1e-10},  # below the bound of the last level 1279
        {"summation": "pairwise"},
    ],
)
def test_invalid_level_and_summation(unit_plate, kwargs):
    with pytest.raises(ValueError):
        unit_plate.at_points(0.5, 0.5, **kwargs)
    with pytest.raises(ValueError):
        unit_plate.at_nodes(4, 4, **kwargs)


@pytest.mark.parametrize(
    ("x", "y"),
    [(-1e-12, 0.5), (1.0 + 1e-12, 0.5), (0.5, -0.1), (0.5, 2.0), (math.nan, 0.5), (0.5, math.inf)],
)
def test_points_outside_square(unit_plate, x, y):
    with pytest.raises(ValueError):
        unit_plate.at_points(x, y)


@pytest.mark.parametrize(("nx", "ny"), [(0, 4), (4, 0), (-2, 4), (4.0, 4), (True, 4)])
def test_invalid_grid(unit_plate, nx, ny):
    with pytest.raises(ValueError):
        unit_plate.at_nodes(nx, ny)


def test_tolerance_selects_level(unit_plate):
    r = unit_plate.at_points(0.5, 0.5, w_tol=1e-6)
    assert r.M == 79 and r.w_tail_bound_coef <= 1e-6
    assert unit_plate.at_points(0.5, 0.5).M == M_R2


# --- values at control points and the full nodal field ------------------------


def test_values_centre_quarters_edges(unit_plate, control):
    assert control.w_coef.shape == control.x.shape
    centre = control.w_coef[0]
    quarter = control.w_coef[1]
    assert 0 < quarter < centre
    edges = unit_plate.at_points([0.0, 1.0, 0.5, 0.5, 0.0, 1.0], [0.5, 0.5, 0.0, 1.0, 0.0, 1.0])
    assert np.all(edges.w_coef == 0.0)
    assert np.all(edges.mx_coef == 0.0) and np.all(edges.my_coef == 0.0)


@pytest.mark.parametrize(("nx", "ny"), [(16, 16), (12, 20), (21, 7)])
def test_nodal_field_equals_pointwise_series(unit_plate, nx, ny):
    g = unit_plate.at_nodes(nx, ny)
    assert g.w_coef.shape == (nx + 1, ny + 1)
    assert np.array_equal(g.x, np.arange(nx + 1) / nx)
    assert np.array_equal(g.y, np.arange(ny + 1) / ny)
    gx, gy = np.meshgrid(g.x, g.y, indexing="ij")
    p = unit_plate.at_points(gx, gy)
    assert_within(g.w_coef, p.w_coef, g.rounding_w + p.rounding_w)
    assert_within(g.mx_coef, p.mx_coef, g.rounding_mx + p.rounding_mx)
    assert_within(g.my_coef, p.my_coef, g.rounding_my + p.rounding_my)
    for name in ("w", "mx", "my"):
        assert_within(getattr(g.convergence, name), getattr(p.convergence, name), 2 * g.rounding_mx + 2 * p.rounding_mx)


def test_points_between_nodes_are_summed_directly(unit_plate):
    # value at a point that is not a node equals its own series, no interpolation
    r = unit_plate.at_points(0.123, 0.789, M=3, diagnostics=False)
    direct = 16 / math.pi**6 * math.fsum(
        math.sin(m * math.pi * 0.123) * math.sin(n * math.pi * 0.789) / (m * n * (m * m + n * n) ** 2)
        for m in (1, 3)
        for n in (1, 3)
    )
    assert r.w_coef == pytest.approx(direct, rel=64 * U, abs=0)


# --- integral of the deflection and energy (В-27, item 6; section 16) ----------


def test_integral_single_term():
    assert deflection_integral_coef(1) == pytest.approx(16 / math.pi**8, rel=RTOL_CLOSED, abs=0)


@pytest.mark.parametrize("M", [1, 3, 9, 19])
def test_integral_against_exact_rational_sum(M):
    odd = range(1, M + 1, 2)
    exact = sum(Fraction(1, m * m * n * n * (m * m + n * n) ** 2) for m in odd for n in odd)
    J = deflection_integral_coef(M)
    assert abs(J - 64 / math.pi**8 * float(exact)) <= (12 + 4) * U * J


def test_integral_against_sum_by_columns():
    # other order: math.fsum over n inside each column m, then fsum of columns
    odd = range(1, M_R2 + 1, 2)
    columns = [math.fsum(1 / (m * m * n * n * (m * m + n * n) ** 2) for n in odd) for m in odd]
    other = 64 / math.pi**8 * math.fsum(columns)
    r = NavierPlate(a=1.0, q=1.0, D=1.0, nu=0.3).deflection_integral()
    assert r.M == M_R2
    assert abs(r.J_M - other) <= r.rounding_J + 4 * U * other


def test_integral_tail_bound():
    assert deflection_integral_tail_bound(1) == pytest.approx(1 / (18 * math.pi**4), rel=RTOL_CLOSED, abs=0)
    # the same number from the lambda_4 form of section 16
    lam4 = math.pi**4 / 96
    for M in NAVIER_LEVELS:
        assert deflection_integral_tail_bound(M) == pytest.approx(
            64 / math.pi**8 * (lam4 / 4) * 2 / (6 * M**3), rel=RTOL_CLOSED, abs=0
        )
    bounds = [deflection_integral_tail_bound(M) for M in NAVIER_LEVELS]
    assert all(b2 < b1 for b1, b2 in zip(bounds, bounds[1:]))
    assert deflection_integral_tail_bound(M_R2) == pytest.approx(2.726e-13, rel=1e-3)
    # J_M grows with M (positive terms) and stays within the bound of the remainder
    values = [deflection_integral_coef(M) for M in NAVIER_LEVELS]
    assert all(j2 > j1 for j1, j2 in zip(values, values[1:]))
    for M, J in zip(NAVIER_LEVELS, values):
        assert 0 <= values[-1] - J <= deflection_integral_tail_bound(M) + 2 * ROUND_J * values[-1]



def test_integral_and_energy_scaling():
    unit = NavierPlate(a=1.0, q=1.0, D=1.0, nu=0.3).deflection_integral(M=79)
    for a, q, D in ((2.9, 3.7, 0.013), (0.4, -250.0, 7.5e3)):
        r = NavierPlate(a=a, q=q, D=D, nu=0.3).deflection_integral(M=79)
        assert r.J_M == unit.J_M
        assert r.I_M == pytest.approx(q * a**6 / D * unit.J_M, rel=RTOL_CLOSED, abs=0)
        assert r.Pi_M == pytest.approx(-0.5 * q**2 * a**6 / D * unit.J_M, rel=RTOL_CLOSED, abs=0)
        assert r.I_tail_bound == pytest.approx(abs(q) * a**6 / D * r.J_tail_bound, rel=RTOL_CLOSED, abs=0)
        assert r.Pi_tail_bound == pytest.approx(0.5 * q**2 * a**6 / D * r.J_tail_bound, rel=RTOL_CLOSED, abs=0)


@pytest.mark.parametrize("q", [1.0, -1.0, 3.2e4, -7.0e-3])
def test_energy_negative_for_nonzero_load(q):
    r = NavierPlate(a=1.3, q=q, D=2.0, nu=0.3).deflection_integral()
    assert r.Pi_M < 0
    assert (r.I_M > 0) == (q > 0)
    assert r.Pi_tail_bound > 0 and r.I_tail_bound > 0


def test_zero_load_integral_and_energy():
    r = NavierPlate(a=1.0, q=0.0, D=1.0, nu=0.3).deflection_integral()
    assert r.I_M == 0.0 and r.Pi_M == 0.0
    assert r.I_tail_bound == 0.0 and r.Pi_tail_bound == 0.0
    assert r.J_M > 0


@pytest.mark.parametrize("M", [0, 2, -1, 3.0, True])
def test_integral_invalid_level(M):
    with pytest.raises(ValueError):
        NavierPlate(a=1.0, q=1.0, D=1.0, nu=0.3).deflection_integral(M=M)


def test_integral_against_trapezoidal_quadrature_of_nodal_field():
    plate = NavierPlate(a=1.0, q=1.0, D=1.0, nu=0.3)
    M = 39
    J = plate.deflection_integral(M=M).J_M
    errors = []
    for n in (20, 40, 80, 160, 320):
        g = plate.at_nodes(n, n, M=M, diagnostics=False)
        # trapezoidal rule; edge nodes are exact zeros, so all weights are h^2
        errors.append(abs(np.sum(g.w_coef) / n**2 - J))
    assert all(e2 < e1 for e1, e2 in zip(errors, errors[1:]))
    ratios = [e1 / e2 for e1, e2 in zip(errors, errors[1:])]
    assert all(3.9 <= r <= 4.1 for r in ratios)
