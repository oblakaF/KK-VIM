"""Own tests of kkvim_reference.parseval_recursion (R2, semi-analytic sequence).

Recursion G1 -> Y1 -> G2 -> X2 -> G3 -> Y3 by sine series (ЧВ-17) and Parseval
(ЧВ-18, ЧВ-19), choice of M (ЧВ-20, И-12) and error accumulation (ЧВ-21),
NUMERICAL_SPEC.md, sections 15.1-15.5, 15.7. No results of the finite-difference
solver, no grid quadratures inside the module, no constants from scratch runs.

Independent checks in this file:
- Parseval sums against Gauss-Legendre integration of an explicitly given finite
  sine combination and of the closed-form Levy profile (an independent method
  applied to an independent closed form);
- the recursion against the scalar form of section 15.3 (sums S_k, T);
- s = 1 field against the approved Levy reference levy_w1 (MATH_SPEC.md, В-29).

Tolerances, fixed before any comparison (not tuned to results):
- U = 2^-53 (unit roundoff of float64);
- RTOL_G1 = 32 U: G1 by another arrangement of the same closed forms (about
  8 roundings each side, margin 2);
- RTOL_QUAD = 1e-13: Gauss-Legendre (4 panels x 32 nodes) of trigonometric and
  hyperbolic integrands; the quadrature error is far below u, rounding of
  about 100 operations per value gives about 1e-14, margin about 10; in the
  G2 check the budget QUAD_ROUNDING = 2e-14 plus the module bound input_error
  of G2 must be <= RTOL_QUAD / 2 before the comparison;
- RTOL_SCALAR = 256 U: two arrangements of the same sums, each within
  ROUND_PARSEVAL = 96 U of the exact sums of the same terms, margin 1.3;
- TOL_LEVY_W1 = 1e-13 relative to max|w1| (NUMERICAL_SPEC.md, 15.7). Budget,
  checked to be <= TOL_LEVY_W1 / 2 before the comparison: field_error of
  s = 1 from the module plus Levy rounding RTOL_LEVY_W * 4 q a^4 / (pi^5 D)
  with RTOL_LEVY_W = 64 U (three terms of size <= 1.8 C0, cosh/sinh of an
  argument <= pi/2, about 35 U, margin 1.8);
- target of the reference (15.7): field and invariant errors <= 1e-12 for s <= 3;
- scale invariance: the sum of the two error bounds of the compared runs (both
  approximate the same exact fields).

Late half-steps s = 4 ... 10 (sections 15.5, 15.7, 17), additionally:
- the budget target 1e-12 is not required for s > 3; the budgets are checked
  for validity, growth and honest reporting only;
- PRACTICAL_CONTROL = 1e-13: practical control of section 15.5 (the whole
  recursion repeated with M at least doubled on every half-step): relative
  change of the centre, fields (n = 80, 160, 320), invariants and energy -
  a diagnostic, not a proof of accuracy;
- energy: Pi_s against Pi_full and Pi_s - Pi_{s-1} against -||w_s - w_{s-1}||^2/2
  within ROUND_ENERGY (|Pi_s| + |Pi_{s-1}|) of the module (both identities are
  exact for the computed truncated Galerkin sequence when M_s >= M_{s-2});
  monotonicity of Pi_s within the same rounding allowance;
- regression of the centre values s = 1 ... 4 against the 13-digit values of
  NUMERICAL_SPEC.md, 15.7 [SCRATCH]: half a unit of the 13th digit, 5e-14,
  plus the module field budget;
- Pi_1 against an independent Gauss-Legendre integral of the Levy profile
  within RTOL_QUAD; Pi_s >= Pi(omega_0) of the Navier reference (MATH_SPEC.md,
  В-23, В-27) - a reference-to-reference check, no KK-VIM results.
"""

import ast
import inspect
import math
from pathlib import Path

import numpy as np
import pytest

from kkvim_reference.levy import levy_w1
from kkvim_reference.navier import NavierPlate
from kkvim_reference.parseval_recursion import (
    EPS_TAIL,
    ROUND_ENERGY,
    ROUND_G1,
    ROUND_PARSEVAL,
    ROUND_VALUES,
    ROUND_X0,
    S_DEFAULT,
    Coefficients,
    ParsevalRecursion,
    first_coefficients,
    parseval_coefficients,
    select_odd_index,
)
from kkvim_reference.sine_halfstep import SineHalfStep

U = 2.0**-53
RTOL_G1 = 32 * U
RTOL_QUAD = 1e-13
QUAD_ROUNDING = 2e-14
RTOL_SCALAR = 256 * U
TOL_LEVY_W1 = 1e-13
RTOL_LEVY_W = 64 * U
TARGET_15_7 = 1e-12
PRACTICAL_CONTROL = 1e-13
S_LATE = 10

CASES = [
    {"a": 1.0, "q": 1.0, "D": 1.0},
    {"a": 2.3, "q": -0.6, "D": 0.7},
]


@pytest.fixture(scope="module")
def unit():
    return ParsevalRecursion(a=1.0, q=1.0, D=1.0)


def recursion(case, **extra):
    return ParsevalRecursion(**case, **extra)


def gauss_legendre(func, ell, panels=4, nodes=32):
    """Composite Gauss-Legendre integral of func over (0, ell) (math.fsum)."""
    t, w = np.polynomial.legendre.leggauss(nodes)
    h = ell / panels
    terms = []
    for p in range(panels):
        mid = (p + 0.5) * h
        for tk, wk in zip(t, w):
            terms.append(wk * (h / 2) * func(mid + (h / 2) * tk))
    return math.fsum(terms)


def rel(x, y):
    return abs(x - y) / abs(y)


# --- 1. G1 against the analytic formulas ------------------------------------------


@pytest.mark.parametrize("case", CASES)
def test_g1_matches_closed_forms(case):
    a, q = case["a"], case["q"]
    G1 = recursion(case, s_max=1).coefficients(1)
    expected = (a / 2, math.pi**2 / (2 * a), math.pi**4 / (2 * a**3), 2 * q * a / math.pi)
    for got, exp in zip((G1.A, G1.B, G1.C, G1.F), expected):
        assert rel(got, exp) <= RTOL_G1


def test_g1_equals_parseval_of_single_sine():
    # X0 = sin(pi x/a) is the series with c_1 = 1 and no other terms.
    a, q = 1.7, 0.9
    G1 = first_coefficients(a=a, q=q)
    P = parseval_coefficients([1.0], ell=a, q=q)
    for x, y in zip((G1.A, G1.B, G1.C, G1.F), (P.A, P.B, P.C, P.F)):
        assert rel(x, y) <= RTOL_G1


# --- 2. Parseval on an independently given finite sine combination ----------------

COMBINATION = [0.83, -0.21, 0.047, -0.0061, 0.0012]  # c_1, c_3, c_5, c_7, c_9
ELL_COMB = 1.37
Q_COMB = -2.4


def combination(xi, derivative):
    total = 0.0
    for j, c in enumerate(COMBINATION):
        lam = (2 * j + 1) * math.pi / ELL_COMB
        if derivative == 0:
            total += c * math.sin(lam * xi)
        elif derivative == 1:
            total += c * lam * math.cos(lam * xi)
        else:
            total += -c * lam * lam * math.sin(lam * xi)
    return total


def test_parseval_against_quadrature_of_finite_combination():
    P = parseval_coefficients(COMBINATION, ell=ELL_COMB, q=Q_COMB)
    A = gauss_legendre(lambda x: combination(x, 0) ** 2, ELL_COMB)
    B = gauss_legendre(lambda x: combination(x, 1) ** 2, ELL_COMB)
    C = gauss_legendre(lambda x: combination(x, 2) ** 2, ELL_COMB)
    F = Q_COMB * gauss_legendre(lambda x: combination(x, 0), ELL_COMB)
    for got, exp in zip((P.A, P.B, P.C, P.F), (A, B, C, F)):
        assert rel(got, exp) <= RTOL_QUAD


def test_parseval_single_term_closed_form():
    c, m, ell, q = 0.4, 7, 2.1, 3.3
    coeffs = [0.0] * ((m - 1) // 2) + [c]
    P = parseval_coefficients(coeffs, ell=ell, q=q)
    lam = m * math.pi / ell
    expected = (ell / 2 * c**2, ell / 2 * lam**2 * c**2, ell / 2 * lam**4 * c**2, q * c * 2 * ell / (m * math.pi))
    for got, exp in zip((P.A, P.B, P.C, P.F), expected):
        assert rel(got, exp) <= RTOL_G1


# --- 3. G2 against independent integration of the Levy closed form ----------------


def levy_derivatives(a, q, D):
    """Y1, Y1', Y1'' of В-29 (own differentiation of the closed form, a = b)."""
    alpha = math.pi / 2
    c0 = 4 * q * a**4 / (math.pi**5 * D)
    p = (alpha * math.tanh(alpha) + 2) / (2 * math.cosh(alpha))
    r = 1 / (2 * math.cosh(alpha))
    k = math.pi / a

    def t_of(y):
        return k * (y - a / 2)

    # Y = c0 [1 - p ch t + r t sh t], dt/dy = k
    y0 = lambda y: c0 * (1 - p * math.cosh(t_of(y)) + r * t_of(y) * math.sinh(t_of(y)))
    y1 = lambda y: c0 * k * (-p * math.sinh(t_of(y)) + r * (math.sinh(t_of(y)) + t_of(y) * math.cosh(t_of(y))))
    y2 = lambda y: c0 * k * k * ((2 * r - p) * math.cosh(t_of(y)) + r * t_of(y) * math.sinh(t_of(y)))
    return y0, y1, y2


@pytest.mark.parametrize("case", CASES)
def test_g2_matches_integrals_of_levy_profile(case):
    a, q, D = case["a"], case["q"], case["D"]
    rec = recursion(case, s_max=2)
    G2 = rec.coefficients(2)
    budget = rec.halfstep(2).budget.input_error + QUAD_ROUNDING
    assert budget <= RTOL_QUAD / 2
    y0, y1, y2 = levy_derivatives(a, q, D)
    expected = (
        gauss_legendre(lambda y: y0(y) ** 2, a),
        gauss_legendre(lambda y: y1(y) ** 2, a),
        gauss_legendre(lambda y: y2(y) ** 2, a),
        q * gauss_legendre(y0, a),
    )
    for got, exp in zip((G2.A, G2.B, G2.C, G2.F), expected):
        assert rel(got, exp) <= RTOL_QUAD


def test_levy_derivative_formulas_are_consistent():
    # The own derivatives vanish where the theory says: Y1 = Y1'' = 0 at y = 0, a,
    # Y1' = 0 at the centre (symmetry).
    a = 1.3
    y0, y1, y2 = levy_derivatives(a, 1.0, 1.0)
    scale = 4 * a**4 / math.pi**5
    for y in (0.0, a):
        assert abs(y0(y)) <= 64 * U * scale
        assert abs(y2(y)) <= 64 * U * scale * (math.pi / a) ** 2
    assert abs(y1(a / 2)) <= 64 * U * scale * math.pi / a


# --- 4. Recursion G1 -> G2 -> G3 --------------------------------------------------


def scalar_recursion(G, *, D, ell, q, M):
    """G_{s+1} by section 15.3: kappa = 4F/(D pi), S_k, T over odd m <= M."""
    kappa = 4 * G.F / (D * math.pi)
    s0, s2, s4, t = [], [], [], []
    for m in range(1, M + 1, 2):
        lam = m * math.pi / ell
        P = G.A * lam**4 + 2 * G.B * lam**2 + G.C
        s0.append(1 / (m * m * P * P))
        s2.append(lam**2 / (m * m * P * P))
        s4.append(lam**4 / (m * m * P * P))
        t.append(1 / (m * m * P))
    half = ell / 2 * kappa * kappa
    return Coefficients(
        A=half * math.fsum(s0),
        B=half * math.fsum(s2),
        C=half * math.fsum(s4),
        F=q * (2 * ell / math.pi) * kappa * math.fsum(t),
    )


@pytest.mark.parametrize("case", CASES)
def test_recursion_chain_and_scalar_form(case):
    rec = recursion(case)
    a, q, D = case["a"], case["q"], case["D"]
    for s in (1, 2, 3):
        step = rec.halfstep(s).step
        G = rec.coefficients(s)
        # The half-step is solved with G_s on the interval of length a.
        assert (step.A, step.B, step.C, step.F, step.D, step.ell) == (G.A, G.B, G.C, G.F, D, a)
        if s < 3:
            nxt = rec.coefficients(s + 1)
            assert nxt == parseval_coefficients(step.coefficients(), ell=a, q=q)
            alt = scalar_recursion(G, D=D, ell=a, q=q, M=step.M)
            for x, y in zip((nxt.A, nxt.B, nxt.C, nxt.F), (alt.A, alt.B, alt.C, alt.F)):
                assert rel(x, y) <= RTOL_SCALAR


def test_fields_are_products_of_the_right_multipliers(unit):
    n = 12
    x0 = [math.sin(math.pi * i / n) for i in range(n + 1)]
    y1 = unit.halfstep(1).step.nodal_values(n)
    x2 = unit.halfstep(2).step.nodal_values(n)
    y3 = unit.halfstep(3).step.nodal_values(n)
    assert np.array_equal(unit.field_on_grid(2, n, n), np.outer(x2, y1))
    assert np.array_equal(unit.field_on_grid(3, n, n), np.outer(x2, y3))
    w1 = unit.field_on_grid(1, n, n)
    # X0 by math.sin without quarter-period reduction: a few U per value.
    assert np.max(np.abs(w1 - np.outer(x0, y1))) <= 8 * U * np.max(np.abs(w1))


def test_s1_field_matches_levy_at_all_nodes():
    rec = ParsevalRecursion(a=1.0, q=1.0, D=1.0, s_max=1)
    for n in (16, 20):
        w = rec.field_on_grid(1, n, n)
        nodes = np.arange(n + 1) / n
        levy = levy_w1(nodes[:, None], nodes[None, :], a=1.0, q=1.0, D=1.0)
        scale = np.max(np.abs(w))
        budget = rec.halfstep(1).budget.field_error + RTOL_LEVY_W * (4 / math.pi**5) / scale
        assert budget <= TOL_LEVY_W1 / 2
        assert np.max(np.abs(w - levy)) / scale <= TOL_LEVY_W1


# --- 5. Positivity of A, B, C; Cauchy-Schwarz ------------------------------------------


@pytest.mark.parametrize("case", CASES)
def test_coefficients_positive_and_cauchy_schwarz(case):
    rec = recursion(case)
    for s in (1, 2, 3):
        G = rec.coefficients(s)
        assert G.A > 0 and G.B > 0 and G.C > 0
        # B^2 <= A C (15.6); equality only for a pure sine mode (s = 1).
        if s == 1:
            assert rel(G.B**2, G.A * G.C) <= 4 * RTOL_G1
        else:
            assert G.B**2 < G.A * G.C * (1 - 1e-6)


# --- 6. Sign of F and of the fields ---------------------------------------------------


@pytest.mark.parametrize("q", [2.5, -2.5])
def test_sign_of_f_follows_load_and_previous_multiplier(q):
    rec = ParsevalRecursion(a=1.0, q=q, D=1.0)
    sign_q = math.copysign(1.0, q)
    # F_s = q * integral of the given multiplier: X0 > 0, Y1 ~ q, X2 ~ q^2/q^2 > 0.
    assert math.copysign(1.0, rec.coefficients(1).F) == sign_q
    assert rec.coefficients(2).F > 0
    assert math.copysign(1.0, rec.coefficients(3).F) == sign_q
    n = 8
    for s in (1, 2, 3):
        w = rec.field_on_grid(s, n, n)
        assert np.all(np.sign(w[1:-1, 1:-1]) == sign_q)


def test_opposite_load_negates_fields_exactly():
    pos = ParsevalRecursion(a=1.0, q=1.3, D=1.0)
    neg = ParsevalRecursion(a=1.0, q=-1.3, D=1.0)
    assert pos.coefficients(2) == neg.coefficients(2)
    for s in (1, 3):
        Gp, Gn = pos.coefficients(s), neg.coefficients(s)
        assert (Gp.A, Gp.B, Gp.C, -Gp.F) == (Gn.A, Gn.B, Gn.C, Gn.F)
    for s in (1, 2, 3):
        assert np.array_equal(neg.field_on_grid(s, 10, 10), -pos.field_on_grid(s, 10, 10))


# --- 7. Scale invariance of the fields ----------------------------------------------------


@pytest.mark.parametrize("scale", [3.7, 1e-3, -2.5])
def test_fields_and_invariants_do_not_depend_on_x0_scale(unit, scale):
    other = ParsevalRecursion(a=1.0, q=1.0, D=1.0, x0_scale=scale)
    n = 16
    for s in (1, 2, 3):
        w, v = unit.field_on_grid(s, n, n), other.field_on_grid(s, n, n)
        tol = unit.halfstep(s).budget.field_error + other.halfstep(s).budget.field_error
        assert np.max(np.abs(w - v)) <= tol * np.max(np.abs(w))
        inv_tol = unit.halfstep(s).budget.invariant_error + other.halfstep(s).budget.invariant_error
        for x, y in zip(unit.coefficients(s).invariants(), other.coefficients(s).invariants()):
            assert rel(y, x) <= inv_tol


@pytest.mark.parametrize("kappa", [0.37, -5.0])
def test_normalisation_between_half_steps_with_compensating_factor(unit, kappa):
    # f -> kappa f: G -> (kappa^2 A, kappa^2 B, kappa^2 C, kappa F), the next
    # multiplier -> /kappa; the field X2 * Y1 is unchanged (В-13; 15.3).
    y1 = unit.halfstep(1).step
    G2 = unit.coefficients(2)
    G2k = parseval_coefficients([kappa * c for c in y1.coefficients()], ell=1.0, q=1.0)
    k2 = kappa * kappa
    for x, y in zip((G2k.A, G2k.B, G2k.C, G2k.F), (k2 * G2.A, k2 * G2.B, k2 * G2.C, kappa * G2.F)):
        assert rel(x, y) <= 2 * ROUND_PARSEVAL
    M = unit.halfstep(2).budget.M
    x2 = unit.halfstep(2).step
    x2k = SineHalfStep(A=G2k.A, B=G2k.B, C=G2k.C, F=G2k.F, D=1.0, ell=1.0, M=M)
    n = 16
    field = np.outer(x2.nodal_values(n), y1.nodal_values(n))
    sigma = kappa  # compensating scalar of the normalised Y1
    other = sigma * np.outer(x2k.nodal_values(n), y1.nodal_values(n))
    tol = 2 * unit.halfstep(2).budget.field_error
    assert np.max(np.abs(field - other)) <= tol * np.max(np.abs(field))


@pytest.mark.parametrize("case", CASES[1:])
def test_fields_scale_as_q_a4_over_d(unit, case):
    # Dimensional scaling (D-009): w_s(a, q, D) at the same nodes = q a^4 / D * w_s(1, 1, 1).
    rec = recursion(case)
    factor = case["q"] * case["a"] ** 4 / case["D"]
    n = 10
    for s in (1, 2, 3):
        w = rec.field_on_grid(s, n, n)
        ref = factor * unit.field_on_grid(s, n, n)
        tol = rec.halfstep(s).budget.field_error + unit.halfstep(s).budget.field_error + 8 * U
        assert np.max(np.abs(w - ref)) <= tol * np.max(np.abs(ref))


# --- 8. Exact zeros on the boundary ---------------------------------------------------------


@pytest.mark.parametrize("n", [7, 20])
def test_boundary_nodes_are_exact_zeros(unit, n):
    for s in (1, 2, 3):
        w = unit.field_on_grid(s, n, n + 3)
        assert np.all(w[0, :] == 0.0) and np.all(w[-1, :] == 0.0)
        assert np.all(w[:, 0] == 0.0) and np.all(w[:, -1] == 0.0)
        assert np.all(w[1:-1, 1:-1] != 0.0)


def test_requested_nodes_equal_slice_of_full_grid(unit):
    full = unit.field_on_grid(3, 12, 9)
    sub = unit.field_on_grid(3, 12, 9, i=[0, 6, 11], j=[2, 9])
    assert np.array_equal(sub, full[np.ix_([0, 6, 11], [2, 9])])


# --- 9. Counters --------------------------------------------------------------------------------


def test_counters_s_and_k(unit):
    expected = {1: (0, "Y"), 2: (1, "X"), 3: (1, "Y")}
    for s, (k, variable) in expected.items():
        record = unit.halfstep(s)
        assert record.s == s and record.k == k == s // 2 and record.variable == variable


@pytest.mark.parametrize("bad", [0, 4, 2.0, True])
def test_default_recursion_rejects_half_steps_outside_1_to_3(unit, bad):
    # the default s_max = 3 keeps the earlier contract
    assert unit.s_max == S_DEFAULT == 3
    with pytest.raises(ValueError):
        unit.halfstep(bad)


@pytest.mark.parametrize("bad", [0, -1, 2.0, True])
def test_rejects_invalid_s_max(bad):
    with pytest.raises(ValueError):
        ParsevalRecursion(a=1.0, q=1.0, D=1.0, s_max=bad)


@pytest.mark.parametrize("changes", [{"m_scale": 0}, {"m_scale": 1.5}, {"m_scale": True}, {"eps": 0.0}])
def test_rejects_invalid_m_scale_and_eps(changes):
    with pytest.raises(ValueError):
        ParsevalRecursion(a=1.0, q=1.0, D=1.0, **changes)


def test_s_max_limits_available_half_steps():
    rec = ParsevalRecursion(a=1.0, q=1.0, D=1.0, s_max=2)
    rec.halfstep(2)
    with pytest.raises(ValueError):
        rec.halfstep(3)


@pytest.mark.parametrize(
    "changes",
    [{"q": 0.0}, {"x0_scale": 0.0}, {"a": -1.0}, {"D": 0.0}, {"q": math.inf}],
)
def test_rejects_invalid_input(changes):
    with pytest.raises(ValueError):
        ParsevalRecursion(**{"a": 1.0, "q": 1.0, "D": 1.0, **changes})


def test_rejects_invalid_grid(unit):
    for args, kwargs in [((1, 0, 4), {}), ((1, 4, 4), {"i": [5]}), ((1, 4, 4), {"j": [-1]})]:
        with pytest.raises(ValueError):
            unit.field_on_grid(*args, **kwargs)


# --- 10. M from analytic bounds only, before any comparison ------------------------------


def own_relative_tails(G, *, D, ell, q, M):
    """Relative tails of 15.4, item 3, written out again in the test."""
    step = SineHalfStep(A=G.A, B=G.B, C=G.C, F=G.F, D=D, ell=ell, M=M)
    K = 4 * abs(G.F) * ell**4 / (D * G.A * math.pi**5)
    f_lower = abs(step.coefficient(1)) - abs(step.coefficient(3))
    kept = parseval_coefficients(step.coefficients(), ell=ell, q=q)
    r = math.pi / ell
    return (
        K / (8 * M**4) / f_lower,
        ell / 2 * K**2 / (18 * M**9) / kept.A,
        ell / 2 * K**2 * r**2 / (14 * M**7) / kept.B,
        ell / 2 * K**2 * r**4 / (10 * M**5) / kept.C,
        abs(q) * 2 * ell / math.pi * K / (10 * M**5) / abs(kept.F),
    )


@pytest.mark.parametrize("case", CASES)
def test_m_is_smallest_odd_index_meeting_all_tail_budgets(case):
    rec = recursion(case)
    a, q, D = case["a"], case["q"], case["D"]
    for s in (1, 2, 3):
        b = rec.halfstep(s).budget
        G = rec.coefficients(s)
        assert b.M % 2 == 1 and b.n_terms == (b.M + 1) // 2
        assert b.M == select_odd_index(G, D=D, ell=a, q=q)
        tails = own_relative_tails(G, D=D, ell=a, q=q, M=b.M)
        recorded = (b.tail_function_rel, b.tail_A_rel, b.tail_B_rel, b.tail_C_rel, b.tail_F_rel)
        for own, kept in zip(tails, recorded):
            assert own <= EPS_TAIL
            assert rel(kept, own) <= 1e-12
        assert max(own_relative_tails(G, D=D, ell=a, q=q, M=b.M - 2)) > EPS_TAIL


def test_m_choice_depends_only_on_g_and_problem_data():
    params = set(inspect.signature(select_odd_index).parameters)
    assert params == {"G", "D", "ell", "q", "eps"}


# --- 11. Separate accounting of tails and error accumulation --------------------------


@pytest.mark.parametrize("case", CASES)
def test_error_accumulation_rule_and_target(case):
    rec = recursion(case)
    budgets = [rec.halfstep(s).budget for s in (1, 2, 3)]
    assert budgets[0].input_error == ROUND_G1
    # tau_j of the first-order rule rho_{s+1} <= 4 rho_s + tau_{s+1} (15.5).
    taus = [ROUND_G1] + [
        max(b.tail_A_rel, b.tail_B_rel, b.tail_C_rel, b.tail_F_rel) + ROUND_PARSEVAL for b in budgets[:2]
    ]
    for s, b in enumerate(budgets, start=1):
        first_order = math.fsum(4 ** (s - j) * taus[j - 1] for j in range(1, s + 1))
        assert rel(b.first_order_input_error, first_order) <= 1e-12
        # The closed-form transfer is not smaller than the first-order rule and
        # differs from it only by higher-order terms.
        assert first_order * (1 - 1e-12) <= b.input_error <= first_order * (1 + 1e-9)
        assert b.input_error <= 21 * max(taus) * (1 + 1e-9)
        # Tails and rounding are recorded separately from the transfer.
        assert 0 < b.tail_function_rel <= EPS_TAIL
        assert b.rounding_coefficients == ROUND_PARSEVAL
        step = rec.halfstep(s).step
        assert b.rounding_values == ROUND_VALUES * step.abs_coefficient_sum()
        assert b.coefficient_error >= 2 * b.input_error
        assert b.multiplier_error > b.tail_function_rel
        # Target of section 15.7 for s <= 3.
        assert b.field_error <= TARGET_15_7
        assert b.invariant_error <= TARGET_15_7
    errors = [b.input_error for b in budgets]
    assert errors[0] < errors[1] < errors[2]


# --- 12. Independence from kkvim ----------------------------------------------------------------


def test_module_does_not_import_kkvim_or_grids():
    import kkvim_reference.parseval_recursion as module

    source = Path(module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.append(node.module)
    # D-038 adds one pure, reference-internal helper.
    assert set(names) == {"math", "dataclasses", "numpy", "kkvim_reference.sine_halfstep", "kkvim_reference._r3_range"}
    assert "Grid1D" not in source


# --- 13. Late half-steps s = 4 ... 10 ---------------------------------------------------


@pytest.fixture(scope="module")
def late():
    return ParsevalRecursion(a=1.0, q=1.0, D=1.0, s_max=S_LATE)


@pytest.fixture(scope="module")
def late_doubled():
    return ParsevalRecursion(a=1.0, q=1.0, D=1.0, s_max=S_LATE, m_scale=2)


def multiplier_nodal(rec, index, n):
    if index == 0:
        return [math.sin(math.pi * i / n) for i in range(n + 1)]
    return rec.halfstep(index).step.nodal_values(n)


def test_first_three_half_steps_unchanged_by_longer_run(unit, late):
    for s in (1, 2, 3):
        old, new = unit.halfstep(s), late.halfstep(s)
        assert old.coefficients == new.coefficients
        assert old.step == new.step
        assert old.budget == new.budget
        assert np.array_equal(unit.field_on_grid(s, 12, 9), late.field_on_grid(s, 12, 9))


def test_centre_values_against_numerical_spec_15_7(late):
    # NUMERICAL_SPEC.md, 15.7 [SCRATCH]: 13 significant digits for s = 1 ... 4.
    printed = {1: 0.0041093468395, 2: 0.0040584722659, 3: 0.0040581830898, 4: 0.0040581806852}
    for s, value in printed.items():
        centre = late.centre_value(s)
        assert abs(centre - value) <= 5e-14 + late.halfstep(s).budget.field_error * centre


def test_late_counters_alternation_and_inputs(late):
    a, q = late.a, late.q
    for s in range(1, S_LATE + 1):
        record = late.halfstep(s)
        assert record.s == s and record.k == s // 2
        assert record.variable == ("Y" if s % 2 else "X")
        if s > 1:
            previous = late.halfstep(s - 1).step.coefficients()
            assert record.coefficients == parseval_coefficients(previous, ell=a, q=q)


def test_late_fields_are_products_of_the_right_multipliers(late):
    n = 12
    for s in range(4, S_LATE + 1):
        if s % 2:
            expected = np.outer(multiplier_nodal(late, s - 1, n), multiplier_nodal(late, s, n))
        else:
            expected = np.outer(multiplier_nodal(late, s, n), multiplier_nodal(late, s - 1, n))
        assert np.array_equal(late.field_on_grid(s, n, n), expected)
    assert late.centre_value(7) == late.field_on_grid(7, 80, 80)[40, 40]


@pytest.mark.parametrize("case", CASES)
def test_late_recursion_against_scalar_form(case):
    rec = recursion(case, s_max=S_LATE)
    for s in range(3, S_LATE):
        G = rec.coefficients(s)
        nxt = rec.coefficients(s + 1)
        alt = scalar_recursion(G, D=case["D"], ell=case["a"], q=case["q"], M=rec.halfstep(s).budget.M)
        for x, y in zip((nxt.A, nxt.B, nxt.C, nxt.F), (alt.A, alt.B, alt.C, alt.F)):
            assert rel(x, y) <= RTOL_SCALAR


@pytest.mark.parametrize("scale", [3.7, -2.5])
def test_late_fields_and_invariants_do_not_depend_on_x0_scale(late, scale):
    other = ParsevalRecursion(a=1.0, q=1.0, D=1.0, s_max=S_LATE, x0_scale=scale)
    n = 16
    for s in range(4, S_LATE + 1):
        b, c = late.halfstep(s).budget, other.halfstep(s).budget
        w, v = late.field_on_grid(s, n, n), other.field_on_grid(s, n, n)
        assert np.max(np.abs(w - v)) <= (b.field_error + c.field_error) * np.max(np.abs(w))
        for x, y in zip(late.coefficients(s).invariants(), other.coefficients(s).invariants()):
            assert rel(y, x) <= b.invariant_error + c.invariant_error
        e, f = late.energy(s), other.energy(s)
        assert rel(f.energy, e.energy) <= e.error + f.error


def test_late_fields_and_energy_scale_with_a_q_d(late):
    case = CASES[1]
    rec = recursion(case, s_max=S_LATE)
    a, q, D = case["a"], case["q"], case["D"]
    n = 10
    for s in range(4, S_LATE + 1):
        w = rec.field_on_grid(s, n, n)
        ref = q * a**4 / D * late.field_on_grid(s, n, n)
        tol = rec.halfstep(s).budget.field_error + late.halfstep(s).budget.field_error + 8 * U
        assert np.max(np.abs(w - ref)) <= tol * np.max(np.abs(ref))
        # Pi ~ q^2 a^6 / D, I ~ q a^6 / D
        e, e1 = rec.energy(s), late.energy(s)
        assert rel(e.energy, q * q * a**6 / D * e1.energy) <= e.error + e1.error + 8 * U
        assert rel(e.integral, q * a**6 / D * e1.integral) <= e.error + e1.error + 8 * U


def test_late_negative_load():
    pos = ParsevalRecursion(a=1.0, q=1.3, D=1.0, s_max=S_LATE)
    neg = ParsevalRecursion(a=1.0, q=-1.3, D=1.0, s_max=S_LATE)
    for s in range(1, S_LATE + 1):
        assert np.array_equal(neg.field_on_grid(s, 10, 10), -pos.field_on_grid(s, 10, 10))
        assert neg.energy(s).integral == -pos.energy(s).integral
        assert neg.energy(s).energy == pos.energy(s).energy < 0


@pytest.mark.parametrize("case", CASES)
def test_two_forms_of_energy_agree(case):
    rec = recursion(case, s_max=S_LATE)
    for s in range(1, S_LATE + 1):
        e = rec.energy(s)
        assert e.rounding == ROUND_ENERGY
        assert abs(e.energy_full - e.energy) <= ROUND_ENERGY * abs(e.energy)
        assert e.energy_norm2 == pytest.approx(case["q"] * e.integral, rel=ROUND_ENERGY, abs=0)


@pytest.mark.parametrize("case", CASES)
def test_energy_non_increasing_and_change_identity(case):
    rec = recursion(case, s_max=S_LATE)
    for s in range(2, S_LATE + 1):
        e, p = rec.energy(s), rec.energy(s - 1)
        allowance = ROUND_ENERGY * (abs(e.energy) + abs(p.energy))
        # w_{s-1} lies in the truncated search space of half-step s when M_s >= M_{s-2}
        older = rec.halfstep(s - 2).budget.M if s > 2 else 1
        assert rec.halfstep(s).budget.M >= older
        assert e.delta <= allowance
        # exactly 0 once the float sequence repeats itself (a fixed point in rounding)
        assert e.change_norm2 >= 0
        assert abs(e.delta + 0.5 * e.change_norm2) <= allowance
    assert rec.energy(1).delta is None and rec.energy(1).change_norm2 is None


def test_energy_of_first_half_step_against_levy_integral():
    a, q, D = 1.0, 1.0, 1.0
    rec = ParsevalRecursion(a=a, q=q, D=D, s_max=1)
    y0, _, _ = levy_derivatives(a, q, D)
    expected = -0.5 * q * (2 * a / math.pi) * gauss_legendre(y0, a)
    assert rel(rec.energy(1).energy, expected) <= RTOL_QUAD


def test_energy_bounded_below_by_navier_energy(late):
    # Pi(w_s) >= Pi(omega_0) = -q I / 2 (В-23, В-27); I <= I_M + bound for q > 0.
    navier = NavierPlate(a=1.0, q=1.0, D=1.0, nu=0.3).deflection_integral()
    exact_lower = -0.5 * (navier.I_M + navier.I_tail_bound + navier.rounding_J)
    for s in range(1, S_LATE + 1):
        e = late.energy(s)
        assert e.energy * (1 - e.error) > exact_lower
    # the N = 1 sequence keeps a positive energy gap (MATH_SPEC.md, 7.4)
    assert late.energy(S_LATE).energy - exact_lower > 1e-4 * abs(exact_lower)


def test_field_and_energy_changes_decrease_with_s(late):
    changes = [late.field_change(s, 80, 80) for s in range(2, 8)]
    assert all(c2 < c1 for c1, c2 in zip(changes, changes[1:]))
    # above the rounding floor (about 1e-14) the changes decrease strictly
    rel_changes = [late.energy(s).change_rel for s in range(2, 9)]
    assert all(c2 < c1 for c1, c2 in zip(rel_changes, rel_changes[1:]))
    with pytest.raises(ValueError):
        late.field_change(1, 8, 8)


def test_doubled_m_practical_control(late, late_doubled):
    for s in range(1, S_LATE + 1):
        b, d = late.halfstep(s).budget, late_doubled.halfstep(s).budget
        assert b.M == b.M_rule and d.M == 2 * (d.M_rule + 1) - 1 >= 2 * b.M_rule
        c1, c2 = late.centre_value(s), late_doubled.centre_value(s)
        assert rel(c1, c2) <= PRACTICAL_CONTROL
        for n in (80, 160, 320):
            w1, w2 = late.field_on_grid(s, n, n), late_doubled.field_on_grid(s, n, n)
            assert np.max(np.abs(w1 - w2)) <= PRACTICAL_CONTROL * np.max(np.abs(w1))
        for x, y in zip(late.coefficients(s).invariants(), late_doubled.coefficients(s).invariants()):
            assert rel(x, y) <= PRACTICAL_CONTROL
        assert rel(late.energy(s).energy, late_doubled.energy(s).energy) <= PRACTICAL_CONTROL


def test_late_budgets_reported_honestly(late):
    budgets = [late.halfstep(s).budget for s in range(1, S_LATE + 1)]
    assert all(b.valid for b in budgets)
    errors = [b.input_error for b in budgets]
    assert all(e2 > e1 for e1, e2 in zip(errors, errors[1:]))
    for s, b in enumerate(budgets, start=1):
        assert b.first_order_input_error * (1 - 1e-12) <= b.input_error
        assert 0 < b.tail_function_rel <= EPS_TAIL
        assert b.integral_error >= b.coefficient_error
        assert late.energy(s).error >= b.integral_error
    # the target 1e-12 of 15.7 holds by the budget for s <= 3, not claimed later
    assert all(b.field_error <= TARGET_15_7 for b in budgets[:3])
    assert budgets[-1].field_error > TARGET_15_7


def test_budget_becomes_invalid_but_values_stay_finite():
    rec = ParsevalRecursion(a=1.0, q=1.0, D=1.0, s_max=30)
    flags = [rec.halfstep(s).budget.valid for s in range(1, 31)]
    first_invalid = flags.index(False) + 1
    assert all(flags[: first_invalid - 1]) and not any(flags[first_invalid - 1 :])
    b = rec.halfstep(30).budget
    assert b.field_error == b.multiplier_error == b.invariant_error == math.inf
    assert rec.energy(30).error == math.inf
    w = rec.field_on_grid(30, 8, 8)
    assert np.all(np.isfinite(w))
    assert math.isfinite(rec.energy(30).energy)


def test_unreliable_series_raises_floating_point_error():
    # q a^4 / D far below the normal range: the squared coefficients underflow
    with pytest.raises(FloatingPointError):
        ParsevalRecursion(a=1.0, q=1e-300, D=1e10, s_max=2)


def test_integral_of_x0_is_closed_form(unit):
    e = unit.energy(1)
    y1 = unit.halfstep(1).step
    lambdas = [(2 * j + 1) * math.pi for j in range(y1.n_terms)]
    integral_y1 = 2 * math.fsum(c / lam for c, lam in zip(y1.coefficients(), lambdas))
    assert rel(e.integral, (2 / math.pi) * integral_y1) <= 4 * U
    assert unit.halfstep(1).budget.integral_error > 0 and ROUND_X0 > 0
