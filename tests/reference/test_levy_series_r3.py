"""Own tests of the full Levy series of L-107, section 30, for R3 (kkvim_reference.levy).

Formulas (h), (i), (139), (j), (140), (141) [L-107, pp. 135-137], verified in
docs/R3_LEVY_SOURCE_TRANSCRIPTION.review.md; derivations, bounds and the
rounding model - docs/R3_LEVY_REFERENCE_DRAFT.md. Only the reference module
is checked; comparisons with the main solver are written by QA (D-033), and
the comparison with the Navier series (P-12) is not made here (no PASS/FAIL).

Tolerances are fixed before the comparison and are not tuned to results:
- U = 2^-53 (unit roundoff);
- closed-form comparisons of the first harmonic (V-29): 100 U C0 absolute,
  C0 = 4 q a_L^4 / (pi^5 D) (both evaluations err by <= about 30 U C0: the
  stable bracket <= 20 U, the textbook form with cosh, sinh <= 2 U (alpha + 2)
  <= 20 U for alpha <= 5 pi / 2; margin about 3);
- field and centre: the strict truncation bounds of the module plus its
  rounding model (sections 5, 6 of the draft) - an engineering check that the
  results are consistent with the bounds, not a proof of them;
- square: full centre 0.0040623527 q a^4 / D (MATH_SPEC.md, 9.3; review of
  the transcription, section 3) - half a unit of the 10th decimal place;
  first harmonic 0.004109346839 q a^4 / D - А-4в, 5e-13 absolute
  (NUMERICAL_SPEC.md, table 18.1).
Table 8 of L-107 is reference only (Р-6) and is not used here.
"""

import ast
import math
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import pytest

from kkvim_reference import levy
from kkvim_reference.levy import (
    LEVY_CENTRE_TOL,
    LEVY_FIELD_TOL,
    levy_alpha_141,
    levy_centre_140,
    levy_centre_bounds,
    levy_centre_j,
    levy_centre_tail_bound,
    levy_centre_wbar,
    levy_field_bounds,
    levy_field_tail_bound,
    levy_orientation,
    levy_select_m_centre,
    levy_select_m_field,
    levy_series_w,
    levy_series_w_grid,
    levy_w1,
)

U = 2.0**-53
TOL_V29 = 100 * U

# Control ratios r = b / a (D-036, Р-1) and the inverse ratios of the task.
R_L_CONTROL = [1.0, 1.4, 1.5, 2.0, 3.0, 5.0]
R_INVERSE = [1 / 5, 1 / 3, 1 / 2, 2 / 3]
R_ALL = R_INVERSE + R_L_CONTROL

CENTRE_SQUARE_FULL = 0.0040623527
TOL_CENTRE_SQUARE_FULL = 5e-11
CENTRE_SQUARE_FIRST = 0.004109346839
TOL_A4V = 5e-13

# Representative physical data, consistent units (D-009). A and the scale
# factors are binary fractions, so that b = r A and lam b stay inside the exactly
# checked range at r = 5 and r = 1/5 (with A = 2.9, 5 * A rounds above 5 A and is
# rejected; R3_LEVY_REFERENCE_DRAFT.md, 2.3).
A, Q, D = 2.5, 3.7e3, 1.3e2


def c0(a_l, q=1.0, d=1.0):
    return 4.0 * q * a_l**4 / (math.pi**5 * d)


def v29_first_halfstep(x, y, a, b, q, d):
    """First half-step by V-29 (MATH_SPEC.md, 9.1) for the plate a x b, start X0 = sin(pi x / a).

    Independent closed form with cosh, tanh, sinh; uses nothing of the module
    under test: alpha = pi b / (2 a), y_c = y - b / 2.
    """
    alpha = math.pi * b / (2.0 * a)
    p = (alpha * math.tanh(alpha) + 2.0) / (2.0 * math.cosh(alpha))
    rr = 1.0 / (2.0 * math.cosh(alpha))
    t = math.pi * (np.asarray(y, dtype=float) - b / 2.0) / a
    y1 = c0(a, q, d) * (1.0 - p * np.cosh(t) + rr * t * np.sinh(t))
    return np.sin(math.pi * np.asarray(x, dtype=float) / a) * y1


def nodes(length, n):
    return np.linspace(0.0, length, n + 1)


# --- 5A. Square: m_max = 1 against the accepted R2 function levy_w1 -------------


@pytest.mark.parametrize(("a", "q", "d"), [(1.0, 1.0, 1.0), (A, Q, D), (0.37, -2.5, 7.1)])
def test_first_harmonic_square_equals_r2_levy_w1(a, q, d):
    s = nodes(a, 40)
    x, y = np.meshgrid(s, s, indexing="ij")
    new = levy_series_w(x, y, a=a, b=a, q=q, D=d, M=1)
    old = levy_w1(x, y, a=a, q=q, D=d)
    tol = TOL_V29 * abs(c0(a, q, d))
    assert np.max(np.abs(new - old)) <= tol
    grid = levy_series_w_grid(s, s, a=a, b=a, q=q, D=d, M=1)
    assert np.max(np.abs(grid - old)) <= tol
    # interior points off the grid, the centre and the edges
    pts = np.array([0.0, 1e-9, 0.123, 0.5, 0.777, 1.0 - 1e-9, 1.0]) * a
    px, py = np.meshgrid(pts, pts, indexing="ij")
    assert np.max(np.abs(levy_series_w(px, py, a=a, b=a, q=q, D=d, M=1) - levy_w1(px, py, a=a, q=q, D=d))) <= tol


def test_first_harmonic_square_centre_value():
    w = float(levy_series_w(0.5, 0.5, a=1.0, b=1.0, q=1.0, D=1.0, M=1))
    assert abs(w - CENTRE_SQUARE_FIRST) <= TOL_A4V
    assert w == pytest.approx(float(levy_w1(0.5, 0.5, a=1.0, q=1.0, D=1.0)), rel=0, abs=TOL_V29 * c0(1.0))


# --- 5B. Rectangles r_L >= 1: first term against V-29 with alpha = pi b_L / (2 a_L) ---


@pytest.mark.parametrize("r_l", R_L_CONTROL)
@pytest.mark.parametrize(("a_l", "q", "d"), [(1.0, 1.0, 1.0), (A, Q, D)])
def test_first_term_equals_v29_for_r_l_at_least_one(r_l, a_l, q, d):
    b_l = r_l * a_l
    xs, ys = nodes(a_l, 32), nodes(b_l, 64)
    x, y = np.meshgrid(xs, ys, indexing="ij")
    expected = v29_first_halfstep(x, y, a_l, b_l, q, d)
    tol = TOL_V29 * abs(c0(a_l, q, d))
    assert np.max(np.abs(levy_series_w_grid(xs, ys, a=a_l, b=b_l, q=q, D=d, M=1) - expected)) <= tol
    assert np.max(np.abs(levy_series_w(x, y, a=a_l, b=b_l, q=q, D=d, M=1) - expected)) <= tol


# --- 5C. b/a < 1: m = 1 is the start along the shorter side, i.e. along y ----------


@pytest.mark.parametrize("r", R_INVERSE)
def test_first_term_for_b_less_than_a_is_start_along_y(r):
    a, b = A, r * A
    xs, ys = nodes(a, 48), nodes(b, 24)
    x, y = np.meshgrid(xs, ys, indexing="ij")
    first = levy_series_w(x, y, a=a, b=b, q=Q, D=D, M=1)
    # V-29 on the swapped plate b x a with start Y0 = sin(pi y / b) (ЧВ-R1).
    start_y = v29_first_halfstep(y, x, b, a, Q, D)
    assert np.max(np.abs(first - start_y)) <= TOL_V29 * c0(b, Q, D)
    # It is not the first half-step of the original plate with X0 = sin(pi x / a)
    # along the longer side: the two differ by far more than rounding.
    start_x = v29_first_halfstep(x, y, a, b, Q, D)
    assert np.max(np.abs(first - start_x)) > 0.01 * np.max(np.abs(first))


# --- 7. Square: full centre, three forms -----------------------------------------


def test_square_full_centre_and_first_term_are_different_values():
    full = levy_centre_140(a=1.0, b=1.0, q=1.0, D=1.0)
    assert abs(full - CENTRE_SQUARE_FULL) <= TOL_CENTRE_SQUARE_FULL
    assert levy_alpha_141(1.0) == full
    first = float(levy_series_w(0.5, 0.5, a=1.0, b=1.0, q=1.0, D=1.0, M=1))
    assert first - full > 4e-5  # 0.0041093 vs 0.0040624: not to be mixed


@pytest.mark.parametrize("r", R_ALL)
def test_consistency_of_139_j_and_140_at_centre(r):
    a, b = A, r * A
    w140 = levy_centre_140(a=a, b=b, q=Q, D=D)
    b140 = levy_centre_bounds(a=a, b=b, q=Q, D=D)
    wj = levy_centre_j(a=a, b=b, q=Q, D=D)
    bj = levy_centre_bounds(a=a, b=b, q=Q, D=D, form="j")
    assert abs(w140 - wj) <= b140.truncation + b140.rounding_model + bj.truncation + bj.rounding_model
    # (139) at the centre has the same terms as (j) at the same M.
    w139 = float(levy_series_w(a / 2, b / 2, a=a, b=b, q=Q, D=D, M=bj.M))
    fb = levy_field_bounds(a=a, b=b, q=Q, D=D, M=bj.M)
    assert abs(w139 - wj) <= fb.rounding_model + bj.rounding_model
    # The centre is the maximum on the grid (L-107, p. 136).
    xs, ys = nodes(a, 20), nodes(b, 20)
    assert np.max(levy_series_w_grid(xs, ys, a=a, b=b, q=Q, D=D)) <= w139 + fb.rounding_model


@pytest.mark.parametrize("r", R_ALL)
def test_centre_wbar_keeps_reference_length_a(r):
    a = A
    w = levy_centre_140(a=a, b=r * a, q=Q, D=D)
    assert levy_centre_wbar(r) == pytest.approx(w * D / (Q * a**4), rel=64 * U)
    if r < 1:
        # w_bar(r) = r^4 alpha_141(1/r) (ЧВ-R1): a_L = b, not the original a.
        assert levy_centre_wbar(r) == pytest.approx(r**4 * levy_alpha_141(1 / r), rel=64 * U)
        assert levy_centre_wbar(r) < 0.5 * levy_alpha_141(1 / r)
    else:
        assert levy_centre_wbar(r) == levy_alpha_141(r)


# --- Edges, reflections, permutation, scaling ------------------------------------


@pytest.mark.parametrize("r", R_ALL)
def test_zero_on_all_four_edges_exactly(r):
    # The stable bracket is exactly 0 at the edges eta_L = +-b_L/2 (s = 0) and
    # the reduced sine argument is exactly 0 at x_L = 0, a_L.
    a, b = A, r * A
    xs, ys = nodes(a, 30), nodes(b, 30)
    zx, zy = np.zeros_like(ys), np.zeros_like(xs)
    assert np.all(levy_series_w(zx, ys, a=a, b=b, q=Q, D=D) == 0.0)
    assert np.all(levy_series_w(zx + a, ys, a=a, b=b, q=Q, D=D) == 0.0)
    assert np.all(levy_series_w(xs, zy, a=a, b=b, q=Q, D=D) == 0.0)
    assert np.all(levy_series_w(xs, zy + b, a=a, b=b, q=Q, D=D) == 0.0)
    w = levy_series_w_grid(xs, ys, a=a, b=b, q=Q, D=D)
    assert np.all(w[0, :] == 0.0) and np.all(w[-1, :] == 0.0)
    assert np.all(w[:, 0] == 0.0) and np.all(w[:, -1] == 0.0)
    assert np.all(w[1:-1, 1:-1] > 0.0)  # q > 0: positive deflection inside


@pytest.mark.parametrize("r", R_ALL)
def test_reflections_x_and_y(r):
    a, b = A, r * A
    xs, ys = nodes(a, 24), nodes(b, 24)
    x, y = np.meshgrid(xs, ys, indexing="ij")
    w = levy_series_w(x, y, a=a, b=b, q=Q, D=D)
    tol = 2 * levy_field_bounds(a=a, b=b, q=Q, D=D).rounding_model
    assert np.max(np.abs(levy_series_w(a - x, y, a=a, b=b, q=Q, D=D) - w)) <= tol
    assert np.max(np.abs(levy_series_w(x, b - y, a=a, b=b, q=Q, D=D) - w)) <= tol


@pytest.mark.parametrize("r", R_ALL)
def test_permutation_of_sides_is_exact(r):
    # w_{a x b}(x, y) = w_{b x a}(y, x) (ЧВ-R1): one frame of L-107 for both.
    a, b = A, r * A
    xs, ys = nodes(a, 16), nodes(b, 12)
    x, y = np.meshgrid(xs, ys, indexing="ij")
    w = levy_series_w(x, y, a=a, b=b, q=Q, D=D)
    w_swapped = levy_series_w(y, x, a=b, b=a, q=Q, D=D)
    g = levy_series_w_grid(xs, ys, a=a, b=b, q=Q, D=D)
    g_swapped = levy_series_w_grid(ys, xs, a=b, b=a, q=Q, D=D).T
    if r == 1.0:
        # Square: no exchange of the frame; x <-> y symmetry holds up to rounding.
        tol = 2 * levy_field_bounds(a=a, b=b, q=Q, D=D).rounding_model
        assert np.max(np.abs(w - w_swapped)) <= tol
        assert np.max(np.abs(g - g_swapped)) <= tol
    else:
        assert np.array_equal(w, w_swapped)
        assert np.array_equal(g, g_swapped)
    assert levy_centre_140(a=a, b=b, q=Q, D=D) == levy_centre_140(a=b, b=a, q=Q, D=D)
    o, s = levy_orientation(a, b), levy_orientation(b, a)
    assert (o.a_L, o.b_L, o.r_L) == (s.a_L, s.b_L, s.r_L)
    assert o.swapped == (r < 1.0) and s.swapped == (r > 1.0)


@pytest.mark.parametrize("r", [1 / 5, 1 / 2, 1.0, 1.4, 5.0])
def test_grid_equals_points(r):
    a, b = A, r * A
    xs, ys = nodes(a, 20), nodes(b, 30)
    x, y = np.meshgrid(xs, ys, indexing="ij")
    tol = 2 * levy_field_bounds(a=a, b=b, q=Q, D=D).rounding_model
    assert np.max(np.abs(levy_series_w_grid(xs, ys, a=a, b=b, q=Q, D=D) - levy_series_w(x, y, a=a, b=b, q=Q, D=D))) <= tol


@pytest.mark.parametrize("r", [1 / 3, 1.0, 2.0, 5.0])
@pytest.mark.parametrize(("lam", "kq", "kd"), [(3.75, 1.0, 1.0), (0.0078125, -2.5, 7.0), (24.0, 1e-3, 4e5)])
def test_scaling_in_a_q_d(r, lam, kq, kd):
    a, b = A, r * A
    xs, ys = nodes(a, 12), nodes(b, 12)
    w = levy_series_w_grid(xs, ys, a=a, b=b, q=Q, D=D)
    ws = levy_series_w_grid(lam * xs, lam * ys, a=lam * a, b=lam * b, q=kq * Q, D=kd * D)
    factor = lam**4 * kq / kd
    tol = 4 * levy_field_bounds(a=lam * a, b=lam * b, q=kq * Q, D=kd * D).rounding_model
    assert np.max(np.abs(ws - factor * w)) <= tol
    assert levy_centre_140(a=lam * a, b=lam * b, q=kq * Q, D=kd * D) == pytest.approx(
        factor * levy_centre_140(a=a, b=b, q=Q, D=D), rel=64 * U
    )


# --- Convergence by the number of odd harmonics; bounds ---------------------------


@pytest.mark.parametrize("r_l", [1.0, 1.4, 5.0])
def test_field_convergence_within_uniform_bound(r_l):
    xs, ys = nodes(1.0, 80), nodes(r_l, 80)
    m_ref = 2 * levy_select_m_field() + 1
    ref = levy_series_w_grid(xs, ys, a=1.0, b=r_l, q=1.0, D=1.0, M=m_ref)
    ref_b = levy_field_bounds(a=1.0, b=r_l, q=1.0, D=1.0, M=m_ref)
    for M in (1, 3, 9, 19, 39, 79, 159, 319):
        w = levy_series_w_grid(xs, ys, a=1.0, b=r_l, q=1.0, D=1.0, M=M)
        bnd = levy_field_bounds(a=1.0, b=r_l, q=1.0, D=1.0, M=M)
        diff = np.max(np.abs(w - ref))
        assert diff <= bnd.truncation + bnd.rounding_model + ref_b.truncation + ref_b.rounding_model
        if M >= 9:
            # The field tail is algebraic, O(M^-4): the exponential bound of the
            # centre series (140) does not hold for the field.
            assert diff > 100 * levy_centre_tail_bound(M, r_l)


@pytest.mark.parametrize("r_l", R_L_CONTROL)
def test_centre_convergence_within_exponential_bound(r_l):
    ref = levy_alpha_141(r_l, M=101)
    for M in range(1, 31, 2):
        bnd = levy_centre_bounds(a=1.0, b=r_l, q=1.0, D=1.0, M=M)
        assert abs(levy_alpha_141(r_l, M=M) - ref) <= bnd.truncation + 2 * bnd.rounding_model


def test_selected_levels_are_minimal():
    assert LEVY_FIELD_TOL == 1e-15 and LEVY_CENTRE_TOL == 1e-19
    M = levy_select_m_field()
    assert M == 1131
    assert levy_field_tail_bound(M) <= LEVY_FIELD_TOL < levy_field_tail_bound(M - 2)
    for r_l in R_L_CONTROL:
        Mc = levy_select_m_centre(r_l)
        assert levy_centre_tail_bound(Mc, r_l) <= LEVY_CENTRE_TOL
        assert Mc == 1 or levy_centre_tail_bound(Mc - 2, r_l) > LEVY_CENTRE_TOL
    assert levy_field_tail_bound(1) == pytest.approx(4 / math.pi**5 / 8, rel=4 * U)


def test_no_overflow_at_largest_ratio_and_default_level():
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        xs, ys = nodes(1.0, 10), nodes(5.0, 50)
        w = levy_series_w_grid(xs, ys, a=1.0, b=5.0, q=1.0, D=1.0)
        w2 = levy_series_w_grid(ys, xs, a=5.0, b=1.0, q=1.0, D=1.0)
    assert np.all(np.isfinite(w)) and np.all(np.isfinite(w2))


# --- Near the edges: relative accuracy against 60-digit decimal arithmetic --------


def decimal_terms(r_l, s, M):
    """Terms m^-5 T_m(s) (-1)^((m-1)/2) of (139) at x_L = a_L/2 in 60-digit decimal, textbook form."""
    with localcontext() as ctx:
        ctx.prec = 60
        pi = Decimal("3.14159265358979323846264338327950288419716939937510582097494")
        s = Decimal(s)
        t = 1 - s
        out = []
        for m in range(1, M + 1, 2):
            al = m * pi * Decimal(r_l) / 2
            ch = (al.exp() + (-al).exp()) / 2
            th = ((al.exp() - (-al).exp()) / 2) / ch
            cht = ((al * t).exp() + (-al * t).exp()) / 2
            sht = ((al * t).exp() - (-al * t).exp()) / 2
            bracket = 1 - (al * th + 2) / (2 * ch) * cht + al / (2 * ch) * t * sht
            out.append((-1) ** ((m - 1) // 2) * bracket / m**5)
        return out


@pytest.mark.parametrize("r_l", [1.0, 2.0, 5.0])
@pytest.mark.parametrize("k", [1, 3, 6, 9, 12])
def test_relative_accuracy_close_to_the_edge(r_l, k):
    # y_L = b_L 10^-k / 2 gives s = 10^-k exactly up to one rounding; the terms
    # are O(alpha_m s), so a textbook form loses about k digits here.
    M = 7
    y = r_l * 10.0**-k / 2
    s = 2.0 * y / r_l
    terms = decimal_terms(r_l, s, M)
    expected = float(sum(terms)) * 4.0 / math.pi**5
    scale = float(sum(abs(t) for t in terms)) * 4.0 / math.pi**5
    w = float(levy_series_w(0.5, y, a=1.0, b=r_l, q=1.0, D=1.0, M=M))
    assert abs(w - expected) <= 64 * U * scale
    assert w > 0.0


# --- Input checks ------------------------------------------------------------------


@pytest.mark.parametrize(("a", "b"), [(1.0, 5.0), (5.0, 1.0), (1.0, 0.2), (0.2, 1.0), (2.0, 10.0)])
def test_range_boundary_accepted(a, b):
    o = levy_orientation(a, b)
    assert 1.0 <= o.r_L <= 5.0


@pytest.mark.parametrize(
    ("a", "b"),
    [
        (1.0, 5.0 * (1 + 2**-50)),
        (5.0 * (1 + 2**-50), 1.0),
        (1.0, 5.000001),
        (1.0, 0.19999),
        # Decimal intention 0.2, float ratio below 1/5 (0.6 < 3/5 as floats): rejected
        # by the exact check; the 4u band of D-037 is not carried over (draft, 2.3).
        (3.0, 0.6),
        # b = 5 * a computed in floats rounds above 5 a for a = 2.9 (same remark).
        (2.9, 5 * 2.9),
    ],
)
def test_range_outside_rejected(a, b):
    with pytest.raises(ValueError, match="ERROR_INPUT"):
        levy_series_w(0.0, 0.0, a=a, b=b, q=1.0, D=1.0)
    with pytest.raises(ValueError, match="ERROR_INPUT"):
        levy_centre_140(a=a, b=b, q=1.0, D=1.0)


@pytest.mark.parametrize("r_l", [0.5, 0.999, 5.0000001, math.inf, math.nan, -1.0])
def test_r_l_outside_rejected(r_l):
    with pytest.raises(ValueError):
        levy_alpha_141(r_l)


def test_invalid_arguments():
    with pytest.raises(ValueError):
        levy_series_w(1.1, 0.5, a=1.0, b=2.0, q=1.0, D=1.0)
    with pytest.raises(ValueError):
        levy_series_w(0.5, 2.1, a=1.0, b=2.0, q=1.0, D=1.0)
    with pytest.raises(ValueError):
        levy_series_w(0.5, 0.5, a=1.0, b=2.0, q=1.0, D=1.0, M=2)
    with pytest.raises(ValueError):
        levy_series_w(0.5, 0.5, a=1.0, b=2.0, q=math.nan, D=1.0)
    with pytest.raises(ValueError):
        levy_series_w(0.5, 0.5, a=1.0, b=2.0, q=1.0, D=0.0)
    with pytest.raises(ValueError):
        levy_centre_bounds(a=1.0, b=2.0, q=1.0, D=1.0, form="x")


# --- Independence -----------------------------------------------------------------


def test_levy_module_imports_neither_kkvim_nor_other_references():
    # The Levy reference must stay independent of the Navier series it is to be
    # compared with in P-12, and of the main solver (D-028).
    tree = ast.parse(Path(levy.__file__).read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "relative import in levy.py"
            imported.add(node.module.split(".")[0])
    assert imported <= {"math", "fractions", "typing", "numpy"}
