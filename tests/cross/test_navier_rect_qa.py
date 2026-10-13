"""QA tests: independent qualification of the rectangular Navier reference NavierRectangle (R3; D-036).

Written by QA from the formulas, not from the module: printed (131) of L-107 for a x b in physical form
16 q / (pi^6 D) sum sin(m pi x / a) sin(n pi y / b) / (m n (m^2/a^2 + n^2/b^2)^2), its term-by-term integral
64 q a b / (pi^8 D) sum 1 / (m^2 n^2 (m^2/a^2 + n^2/b^2)^2), the strict bounds ЧВ-R2, ЧВ-R2a (full formulas)
and the axis exchange ЧВ-R1 (R3_NUMERICAL_SPEC_DRAFT.md, 1.2, 4.1-4.3). Own sums use math.sin, exact rationals
(Fraction) or exactly formed integer terms with math.fsum; no value of the module is used as an expected answer.

Criterion of the owner (2026-10-14, fixed before this qualification; R3 Reference only): delta_w(r) = 1e-6 w_c(r)
in q a^4 / D. Sufficient condition checked: B_2559(r) + R_2559(r) <= 1e-6 (w_c,1279 - B_1279 - R_1279), with B the
strict bound ЧВ-R2 and R the documented rounding model of the implementation (a first-order estimate, not a
proved bound). It does not perform P-12 and does not qualify the moments (D-034, item 3).

Tolerances: rounding estimates of the module and own first-order rounding estimates (stated per test); exact
identities and integer-reduced values compared bitwise where the operations are the same.

Light mathematical tests use small M; the control computations at M = 2559 use the centre, few points and
small grids.
"""

import ast
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from kkvim_reference import navier as nv
from kkvim_reference.navier import (
    M_R2,
    M_R3,
    NavierPlate,
    NavierRectangle,
    deflection_integral_coef_rect,
    deflection_integral_tail_bound_rect,
    deflection_tail_bound_rect,
)

U = 2.0**-53
NU = 0.3
ZETA3 = 1.2020569031595942853997381  # Apery's constant [СТАНД.]
LAMBDA3 = 7.0 * ZETA3 / 8.0  # sum over odd n of n^-3
LAMBDA4 = math.pi**4 / 96.0  # sum over odd n of n^-4
# Control ratios (D-036, Р-1) as plates (a, b) with b/a exact as a ratio of the given floats.
CONTROL = {"0.2": (1.0, 0.2), "1/3": (3.0, 1.0), "1/2": (2.0, 1.0), "2/3": (3.0, 2.0), "1": (1.0, 1.0),
           "1.4": (1.0, 1.4), "1.5": (2.0, 3.0), "2": (1.0, 2.0), "3": (1.0, 3.0), "5": (1.0, 5.0)}
SRC = Path(__file__).resolve().parents[2] / "src" / "kkvim_reference"


def rect(a, b, q=1.0, D=1.0, nu=NU):
    return NavierRectangle(a=a, b=b, q=q, D=D, nu=nu)


def own_point_series(M, a, b, q, D, x, y):
    """(131) of L-107 in physical form at one point, math.sin per term; value and first-order rounding estimate."""
    C = 16.0 * q / (math.pi**6 * D)
    terms, err = [], 0.0
    for m in range(1, M + 1, 2):
        sm, am = math.sin(m * math.pi * x / a), m * m / (a * a)
        for n in range(1, M + 1, 2):
            c = C / (m * n * (am + n * n / (b * b)) ** 2)
            terms.append(c * sm * math.sin(n * math.pi * y / b))
            err += abs(c) * (24.0 + 2.0 * math.pi * (m + n)) * U
    return math.fsum(terms), 2.0 * err


def own_integral_exact(M, a, b, q, D):
    """64 q a b / (pi^8 D) sum 1/(m^2 n^2 (m^2/a^2 + n^2/b^2)^2), the sum exact in rationals."""
    A2, B2 = Fraction(a) ** 2, Fraction(b) ** 2
    s = sum(Fraction(1, m * m * n * n) / (m * m / A2 + n * n / B2) ** 2
            for m in range(1, M + 1, 2) for n in range(1, M + 1, 2))
    return 64.0 * q * a * b / (math.pi**8 * D) * float(s)


def own_centre_exact_terms(M, a, b):
    """w_c in q a^4 / D; terms (-1)^((m+n)/2-1) P^4 / (m n (m^2 P^2 + n^2 Q^2)^2) from exact integers (b/a = P/Q).

    Each term correctly rounded (<= u), math.fsum (<= u |S|), constant 16/pi^6 (<= 8u); estimate with margin 2.
    """
    P, Q = (Fraction(b) / Fraction(a)).as_integer_ratio()
    P2, Q2, P4 = P * P, Q * Q, P**4
    terms = []
    for m in range(1, M + 1, 2):
        mP2 = m * m * P2
        for n in range(1, M + 1, 2):
            t = P4 / (m * n * (mP2 + n * n * Q2) ** 2)
            terms.append(t if (m + n) % 4 == 2 else -t)
    S = math.fsum(terms)
    C = 16.0 / math.pi**6
    return C * S, 2.0 * (C * (U * math.fsum(map(abs, terms)) + U * abs(S)) + 8 * U * abs(C * S))


def own_centre_numpy(M, r):
    """Same centre sum with float terms (about 7 roundings each), math.fsum; first-order estimate, margin 2."""
    k = np.arange(1, M + 1, 2, dtype=float)
    s = np.where(np.arange(k.size) % 2 == 0, 1.0, -1.0)
    t = np.outer(s, s) / (k[:, None] * k[None, :] * (k[:, None] ** 2 + k[None, :] ** 2 / (r * r)) ** 2)
    S = math.fsum(t.ravel().tolist())
    C = 16.0 / math.pi**6
    return C * S, 2.0 * (C * (8 * U * float(np.abs(t).sum()) + U * abs(S)) + 8 * U * abs(C * S))


def chv_r2(M, r):
    """ЧВ-R2, full formula (16/pi^6) (7 zeta(3)/8) r^2 / (8 M^2), rounded upwards by 16u."""
    return 16.0 / math.pi**6 * LAMBDA3 * r * r / (8.0 * M * M) * (1 + 16 * U)


def chv_r2a(M, r):
    """ЧВ-R2a, full formula r^3 / (18 pi^4 M^3), rounded upwards by 16u."""
    return r**3 / (18.0 * math.pi**4 * M**3) * (1 + 16 * U)


# --- light mathematical tests (small M) ---------------------------------------------------------


@pytest.mark.parametrize("name", list(CONTROL))
def test_p1_first_term_physical(name):
    """П-1: M = 1 equals 16 q / (pi^6 D (1/a^2 + 1/b^2)^2) sin(pi x / a) sin(pi y / b) at points."""
    a0, b0 = CONTROL[name]
    a, b, q, D = 1.7 * a0, 1.7 * b0, -2.3, 0.61
    p = rect(a, b, q, D)
    x, y = a * np.array([0.5, 0.13, 0.71, 0.97]), b * np.array([0.5, 0.42, 0.08, 0.66])
    res = p.at_points(x, y, M=1, diagnostics=False)
    amp = 16.0 * q / (math.pi**6 * D * (1 / a**2 + 1 / b**2) ** 2)
    for i in range(x.size):
        expected = amp * math.sin(math.pi * x[i] / a) * math.sin(math.pi * y[i] / b)
        assert abs(res.w[i] - expected) <= 64 * U * abs(amp)
    centre = p.at_points(a / 2, b / 2, M=1, diagnostics=False)
    r = Fraction(b) / Fraction(a)
    assert abs(float(centre.w_coef) - 16.0 / math.pi**6 / float((1 + 1 / r**2) ** 2)) <= 16 * U * float(centre.w_coef)


@pytest.mark.parametrize("name", ["0.2", "1/3", "1.4", "2", "5"])
def test_formula_131_physical_at_points(name):
    """Printed (131) for a x b, summed independently in physical units (math.sin), M = 19."""
    a0, b0 = CONTROL[name]
    a, b, q, D = 2.3 * a0, 2.3 * b0, 3.7, 0.9
    p = rect(a, b, q, D)
    rng = np.random.default_rng(11)
    x, y = a * rng.random(6), b * rng.random(6)
    res = p.at_points(x, y, M=19, diagnostics=False)
    for i in range(x.size):
        value, err = own_point_series(19, a, b, q, D, x[i], y[i])
        assert abs(res.w[i] - value) <= err + abs(res.w_scale) * res.rounding_w


@pytest.mark.parametrize("M", [1, 3, 9])
@pytest.mark.parametrize(("a", "b"), [(1.3, 2.1), (2.0, 0.5), (1.0, 5.0), (3.0, 0.6)])
def test_integral_and_energy_against_exact_rational_sum(M, a, b):
    """I_M = integral of the truncated (131) over a x b (exact rational sum); Pi_M = -q I_M / 2 < 0 for q != 0."""
    q, D = -1.9, 0.7
    g = rect(a, b, q, D).deflection_integral(M=M)
    expected = own_integral_exact(M, a, b, q, D)
    assert abs(g.I_M - expected) <= 32 * U * abs(expected)
    assert g.Pi_M == -0.5 * q * g.I_M and g.Pi_M < 0.0
    assert abs(g.I_tail_bound - abs(q) * a**6 / D * chv_r2a(M, b / a)) <= 32 * U * g.I_tail_bound


@pytest.mark.parametrize("name", ["0.2", "1", "1.4", "5"])
def test_integral_is_the_integral_of_the_field(name):
    """Gauss-Legendre quadrature (48 x 48) of the module's own field at M = 9 reproduces I_M."""
    a, b = CONTROL[name]
    p = rect(a, b, q=1.3, D=0.8)
    t, wts = np.polynomial.legendre.leggauss(48)
    x, y = a * (t + 1) / 2, b * (t + 1) / 2
    X, Y = np.meshgrid(x, y, indexing="ij")
    res = p.at_points(X, Y, M=9, diagnostics=False)
    quad = float(np.sum(np.outer(wts, wts) * res.w)) * a * b / 4
    g = p.deflection_integral(M=9)
    assert abs(quad - g.I_M) <= abs(res.w_scale) * res.rounding_w * a * b + 64 * U * abs(g.I_M)


def test_chv_r2_lemmas():
    """Steps of ЧВ-R2: tail of odd n^-3 <= 1/(4 M^2); the majorant outside the square <= 2 lambda_3 / (4 M^2)."""
    odd3 = [1.0 / n**3 for n in range(1, 400001, 2)]
    assert abs(math.fsum(odd3) + 1 / (4 * 400001**2) - LAMBDA3) <= 1e-15  # lambda_3 = 7 zeta(3) / 8
    for M in (1, 3, 9, 39, 159, 639):
        head = math.fsum(1.0 / n**3 for n in range(1, M + 1, 2))
        tail = LAMBDA3 - head
        assert 0 < tail <= 1.0 / (4 * M * M)
        assert LAMBDA3**2 - head**2 <= 2 * LAMBDA3 / (4 * M * M)
    # (m^2 + n^2/r^2)^2 >= 4 m^2 n^2 / r^2: each term <= r^2 / (4 m^3 n^3) (checked on a grid of m, n, r)
    for r in (0.2, 1 / 3, 1.4, 5.0):
        for m in range(1, 60, 2):
            for n in range(1, 60, 2):
                assert 1.0 / (m * n * (m * m + n * n / r**2) ** 2) <= r * r / (4 * m**3 * n**3) * (1 + 8 * U)


@pytest.mark.parametrize("name", list(CONTROL))
def test_chv_r2_function_is_the_full_formula(name):
    a, b = CONTROL[name]
    r = b / a
    for M in (1, 9, 1279, 2559):
        assert abs(deflection_tail_bound_rect(M, r) - chv_r2(M, r)) <= 32 * U * chv_r2(M, r)
        assert abs(deflection_integral_tail_bound_rect(M, r) - chv_r2a(M, r)) <= 32 * U * chv_r2a(M, r)
    assert abs(deflection_tail_bound_rect(M_R3, r) / r**2 - 3.341e-10) <= 0.0005e-10  # R3 draft, 4.3


def test_chv_r2a_lemmas():
    """Steps of ЧВ-R2a: lambda_4 = pi^4/96; tail of odd n^-4 <= 1/(6 M^3)."""
    head = math.fsum(1.0 / n**4 for n in range(1, 200001, 2))
    assert abs(head - LAMBDA4) <= 1e-15
    for M in (1, 3, 9, 39, 159, 639):
        tail = LAMBDA4 - math.fsum(1.0 / n**4 for n in range(1, M + 1, 2))
        assert 0 < tail <= 1.0 / (6 * M**3)


@pytest.mark.parametrize("name", ["0.2", "1/3", "1.4", "2", "5"])
def test_strict_bounds_cover_truncation(name):
    """|w_M - w_639| <= B_M + B_639 + rounding at nodes; 0 <= J_639 - J_M <= ЧВ-R2a(M) + rounding."""
    a, b = CONTROL[name]
    r = b / a
    p = rect(a, b)
    ref = p.at_nodes(12, 12, M=639, diagnostics=False)
    for M in (1, 3, 9, 39):
        res = p.at_nodes(12, 12, M=M, diagnostics=False)
        diff = float(np.max(np.abs(res.w_coef - ref.w_coef)))
        assert diff <= chv_r2(M, r) + chv_r2(639, r) + res.rounding_w + ref.rounding_w
        J, Jref = deflection_integral_coef_rect(M, r), deflection_integral_coef_rect(639, r)
        assert -2 * nv.ROUND_INTEGRAL_RECT * Jref <= Jref - J <= chv_r2a(M, r) + 2 * nv.ROUND_INTEGRAL_RECT * Jref


@pytest.mark.parametrize("name", ["0.2", "1/3", "1.4", "5"])
def test_edges_are_exact_zeros(name):
    """П-6: w = M_x = M_y = 0 exactly on all edges, at points and nodes."""
    a, b = CONTROL[name]
    p = rect(a, b)
    s = np.linspace(0.0, 1.0, 9)
    x = np.concatenate([np.zeros(9), np.full(9, a), a * s, a * s])
    y = np.concatenate([b * s, b * s, np.zeros(9), np.full(9, b)])
    for M in (9, 159):
        res = p.at_points(x, y, M=M, diagnostics=False)
        for f in (res.w, res.Mx, res.My):
            assert np.all(f == 0.0)
        nodes = p.at_nodes(8, 6, M=M, diagnostics=False)
        for f in (nodes.w, nodes.Mx, nodes.My):
            assert np.all(f[[0, -1], :] == 0.0) and np.all(f[:, [0, -1]] == 0.0)


@pytest.mark.parametrize("name", list(CONTROL))
def test_reflection_symmetry(name):
    """П-7: w(a - x, y) = w(x, b - y) = w(x, y) at points and at nodes, within the rounding estimate."""
    a, b = CONTROL[name]
    p = rect(a, b)
    rng = np.random.default_rng(5)
    x, y = a * rng.random(8), b * rng.random(8)
    base = p.at_points(x, y, M=79, diagnostics=False)
    tol = 2 * base.rounding_w
    for xx, yy in ((a - x, y), (x, b - y), (a - x, b - y)):
        assert np.max(np.abs(p.at_points(xx, yy, M=79, diagnostics=False).w_coef - base.w_coef)) <= tol
    # nodes: the rows of S_x for i and n - i are identical, but BLAS may sum rows in different
    # blocks with a different order of operations (CPU dependent), so not bitwise
    nodes = p.at_nodes(10, 14, M=79, diagnostics=False)
    w = nodes.w_coef
    assert np.max(np.abs(w - w[::-1, :])) <= nodes.rounding_w and np.max(np.abs(w - w[:, ::-1])) <= nodes.rounding_w


@pytest.mark.parametrize("name", list(CONTROL))
def test_maximum_at_centre(name):
    """П-5: the centre is the strict maximum over all nodes of a 20 x 20 grid (M = 159)."""
    a, b = CONTROL[name]
    w = rect(a, b).at_nodes(20, 20, M=159, diagnostics=False).w_coef
    assert np.unravel_index(np.argmax(w), w.shape) == (10, 10)
    others = np.delete(w.ravel(), 10 * 21 + 10)
    assert w[10, 10] > others.max() > 0


@pytest.mark.parametrize("name", ["1/3", "1.4", "3"])
def test_deflection_independent_of_nu(name):
    """П-9: w does not depend on nu (bitwise), at points and nodes."""
    a, b = CONTROL[name]
    ref = rect(a, b, nu=0.3)
    for nu in (-0.6, 0.0, 0.49):
        p = rect(a, b, nu=nu)
        assert np.array_equal(p.at_nodes(6, 8, M=39).w_coef, ref.at_nodes(6, 8, M=39).w_coef)
        assert np.array_equal(p.at_points(0.3 * a, 0.6 * b, M=39).w_coef, ref.at_points(0.3 * a, 0.6 * b, M=39).w_coef)


@pytest.mark.parametrize("name", ["0.2", "1.4", "5"])
def test_physical_scaling(name):
    """П-8: at fixed r, w = (q a^4 / D) w_bar(x/a, y/b); the sign follows q."""
    a, b = CONTROL[name]
    base = rect(a, b).at_nodes(8, 8, M=79, diagnostics=False)
    for lam, q, D in ((3.7, 2.5, 0.3), (0.25, -4.0, 11.0)):
        res = rect(lam * a, lam * b, q, D).at_nodes(8, 8, M=79, diagnostics=False)
        scale = q * (lam * a) ** 4 / D
        assert np.max(np.abs(res.w - scale * base.w_coef)) <= abs(scale) * (base.rounding_w + res.rounding_w)
        assert np.all(np.sign(res.w[1:-1, 1:-1]) == np.sign(q))


@pytest.mark.parametrize("name", ["0.2", "2/3", "1.4", "5"])
def test_two_summation_orders(name):
    """П-11: separable and diagonal orders agree within the sum of their rounding estimates."""
    a, b = CONTROL[name]
    p = rect(a, b)
    rng = np.random.default_rng(3)
    x, y = a * rng.random(10), b * rng.random(10)
    s = p.at_points(x, y, M=79, diagnostics=False)
    d = p.at_points(x, y, M=79, summation="diagonal", diagnostics=False)
    for f, rf in (("w_coef", "rounding_w"), ("mx_coef", "rounding_mx"), ("my_coef", "rounding_my")):
        assert np.max(np.abs(getattr(s, f) - getattr(d, f))) <= getattr(s, rf) + getattr(d, rf)


@pytest.mark.parametrize("name", ["1/3", "1.4", "3"])
def test_points_and_nodes_agree(name):
    """Values at nodes (integer reduction) equal values at the same points (fmod reduction) within rounding."""
    a, b = CONTROL[name]
    p = rect(a, b)
    nodes = p.at_nodes(8, 12, M=159, diagnostics=False)
    X, Y = np.meshgrid(nodes.x, nodes.y, indexing="ij")
    pts = p.at_points(X, Y, M=159, diagnostics=False)
    assert np.max(np.abs(pts.w_coef - nodes.w_coef)) <= pts.rounding_w + nodes.rounding_w


@pytest.mark.parametrize("name", ["0.2", "1/3", "1.4", "2", "5"])
def test_axis_exchange(name):
    """ЧВ-R1: w(a, b; x, y) = w(b, a; y, x) in physical units, at points and at nodes (transposed)."""
    a, b = CONTROL[name]
    p, s = rect(a, b, q=1.4, D=0.7), rect(b, a, q=1.4, D=0.7)
    rng = np.random.default_rng(9)
    x, y = a * rng.random(8), b * rng.random(8)
    rp, rs = p.at_points(x, y, M=79, diagnostics=False), s.at_points(y, x, M=79, diagnostics=False)
    tol = abs(rp.w_scale) * rp.rounding_w + abs(rs.w_scale) * rs.rounding_w
    assert np.max(np.abs(rp.w - rs.w)) <= tol
    np_, ns = p.at_nodes(6, 10, M=79, diagnostics=False), s.at_nodes(10, 6, M=79, diagnostics=False)
    assert np.max(np.abs(np_.w - ns.w.T)) <= abs(np_.w_scale) * np_.rounding_w + abs(ns.w_scale) * ns.rounding_w


@pytest.mark.parametrize("k", [1.4, 1.5, 2.0, 3.0, 5.0])
def test_inverse_ratio_of_centre(k):
    """w_c(1/r) = r^-4 w_c(r) for the same rectangle turned: plates (a, b) = (k, 1) and (1, k), M = 159."""
    wide, tall = rect(k, 1.0).at_points(k / 2, 0.5, M=159), rect(1.0, k).at_points(0.5, k / 2, M=159)
    assert abs(float(wide.w_coef) - float(tall.w_coef) / k**4) <= wide.rounding_w + tall.rounding_w / k**4


def test_grid_field_is_separable(monkeypatch):
    """at_nodes uses one tensor product S_x C S_y^T per quantity, no per-point double loop over harmonics."""
    calls = {"tensor": 0, "other": 0}
    original = nv._sum_separable

    def counting(Sx, Sy, c, tensor):
        calls["tensor" if tensor else "other"] += 1
        return original(Sx, Sy, c, tensor)

    def forbidden(*args):
        raise AssertionError("diagonal summation used for the separable grid")

    monkeypatch.setattr(nv, "_sum_separable", counting)
    monkeypatch.setattr(nv, "_sum_diagonal", forbidden)
    rect(1.0, 5.0).at_nodes(16, 80, M=159, diagnostics=False)
    assert calls == {"tensor": 3, "other": 0}


def test_reference_package_does_not_import_kkvim():
    """П-10 (QA's own check): no module of kkvim_reference imports kkvim."""
    for path in sorted(SRC.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            assert not any(n == "kkvim" or n.startswith("kkvim.") for n in names), (path.name, names)


# --- R2 regression ------------------------------------------------------------------------------

R2_PUBLIC = (
    "DELTA_M", "DELTA_W", "M_R2", "NAVIER_LEVELS", "SUMMATIONS", "NavierConvergence", "NavierIntegral", "NavierPlate",
    "NavierResult", "deflection_integral_coef", "deflection_integral_tail_bound", "deflection_tail_bound",
    "deflection_tail_bound_refined", "diagnostic_levels", "flexural_rigidity", "moment_tail_bound",
    "moment_tail_bound_refined", "practical_control", "select_level",
)


def test_r2_api_and_constants_unchanged():
    assert set(R2_PUBLIC) <= set(nv.__all__)
    assert M_R2 == 1279 and nv.NAVIER_LEVELS == (1, 3, 9, 19, 39, 79, 159, 319, 639, 1279)
    assert nv.DELTA_W == 4.06e-9 and nv.DELTA_M == 4.79e-6 and nv.select_level(nv.DELTA_W) == 1279
    with pytest.raises(ValueError):
        NavierPlate(a=1.0, q=1.0, D=1.0, nu=NU, b=2.0)
    NavierPlate(a=1.0, q=1.0, D=1.0, nu=NU, b=1.0)


def test_rectangle_at_r1_reproduces_the_square():
    sq, re = NavierPlate(a=1.3, q=2.0, D=0.4, nu=NU), rect(1.3, 1.3, 2.0, 0.4)
    for M in (9, 1279):
        a, b = sq.at_nodes(8, 8, M=M), re.at_nodes(8, 8, M=M)
        assert np.array_equal(a.w, b.w) and np.array_equal(a.Mx, b.Mx) and a.rounding_w == b.rounding_w
        assert sq.deflection_integral(M=M).J_M == re.deflection_integral(M=M).J_M


# --- control computations at M = 2559 -----------------------------------------------------------


@pytest.mark.parametrize("name", list(CONTROL))
def test_control_centre_and_delta_w_criterion(name):
    """Centre at M = 1279 and 2559 against own sums; sufficient condition for delta_w(r) = 1e-6 w_c(r).

    Failure of the last assertion would mean the recorded PASS no longer holds (status UNRESOLVED, not FAIL).
    """
    a, b = CONTROL[name]
    p = rect(a, b)
    r = p.r
    c1279 = p.at_points(a / 2, b / 2, M=1279, diagnostics=False)
    c2559 = p.at_points(a / 2, b / 2, M=M_R3, diagnostics=False)
    own1279, e1279 = own_centre_exact_terms(1279, a, b)
    own2559, e2559 = own_centre_numpy(M_R3, r)
    w1279, R1279 = float(c1279.w_coef), c1279.rounding_w
    assert abs(w1279 - own1279) <= R1279 + e1279
    assert abs(float(c2559.w_coef) - own2559) <= c2559.rounding_w + e2559
    lower = w1279 - chv_r2(1279, r) - R1279
    assert lower > 0
    assert chv_r2(M_R3, r) + c2559.rounding_w <= 1e-6 * lower


@pytest.mark.parametrize("name", ["0.2", "1.4", "5"])
def test_control_integral_at_m_r3(name):
    """J_2559 against own exactly-formed terms; ЧВ-R2a at 2559; Pi = -q I / 2."""
    a, b = CONTROL[name]
    P, Q = (Fraction(b) / Fraction(a)).as_integer_ratio()
    P2, Q2, P5 = P * P, Q * Q, P**5
    terms = [P5 / (Q * m * m * n * n * (m * m * P2 + n * n * Q2) ** 2)
             for m in range(1, M_R3 + 1, 2) for n in range(1, M_R3 + 1, 2)]
    own = 64.0 / math.pi**8 * math.fsum(terms)
    g = rect(a, b).deflection_integral()
    assert g.M == M_R3
    assert abs(g.J_M - own) <= g.rounding_J + 24 * U * own
    assert abs(g.J_tail_bound - chv_r2a(M_R3, b / a)) <= 32 * U * g.J_tail_bound
    assert g.Pi_M == -0.5 * g.I_M


@pytest.mark.parametrize("k", [2.0, 3.0, 5.0])
def test_control_inverse_ratio_at_m_r3(k):
    wide, tall = rect(k, 1.0).at_points(k / 2, 0.5), rect(1.0, k).at_points(0.5, k / 2)
    assert wide.M == tall.M == M_R3
    assert abs(float(wide.w_coef) - float(tall.w_coef) / k**4) <= wide.rounding_w + tall.rounding_w / k**4


def test_control_accepted_boundary_plate_at_m_r3():
    """a = 3, b = 0.6 (accepted by the 4u rule) gives the centre of r = 0.2 within rounding and r^2 u."""
    edge, exact = rect(3.0, 0.6).at_points(1.5, 0.3), rect(1.0, 0.2).at_points(0.5, 0.1)
    assert abs(float(edge.w_coef) - float(exact.w_coef)) <= edge.rounding_w + exact.rounding_w + 8 * U * float(exact.w_coef)
