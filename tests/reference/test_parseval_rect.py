"""Own tests of ParsevalRecursion for the rectangle a x b (R3; D-036, D-030).

Alternation of lengths (R3_NUMERICAL_SPEC_DRAFT.md, 2.1, 3.1; MATH_SPEC.md, 4.2-4.4;
NUMERICAL_SPEC.md, 15.1-15.5): half-step s is solved on ell_s = b (odd s) or
a (even s); G_{s+1} are the Parseval sums of the multiplier of half-step s on
its own length ell_s; X0 = sin(pi x/a) on (0, a). The R2 tests of the square
are not changed; this file adds independent checks for r != 1:

- s = 1 against the closed form В-29 with alpha = pi b/(2a) (field and centre);
- G2 against Gauss-Legendre integrals of the closed-form Y1 on (0, b), the
  integrand evaluated in decimal arithmetic (no cancellation in the check);
- multipliers of s = 2, 3, 4 against an independent closed-form solution of the
  half-step ODE D(A f'''' - 2B f'' + C f) = F on (0, ell_s) through the complex
  roots of A k^4 - 2B k^2 + C = 0 (another method than the sine series);
- the chain G_{s+1} = Parseval(multiplier s, ell_s) and the scalar form 15.3
  with ell_s; counters, labels and lengths of every half-step;
- fields at points and on rectangular grids, exact zeros on the edges,
  scaling, x0 scale, energy forms, the closed-form integral of Y1, the Navier
  lower bound of the energy, tails, error transfer, doubled M;
- the square: b = a and b = None give bitwise the R2 results.

Tolerances, fixed before any comparison (not tuned to results):
- U = 2^-53; RTOL_CLOSED = 16 U, RTOL_G1 = 32 U, RTOL_SCALAR = 256 U and
  RTOL_QUAD = 1e-13 with QUAD_ROUNDING = 2e-14 - the same models as in the R2
  test of this module (two arrangements of the same sums; Gauss-Legendre of
  smooth integrands with the integrand exact to about u);
- closed forms evaluated in float: CLOSED_ROUNDING = 32 U times the sum of the
  moduli of their terms (each term about 8 roundings, margin 4); the closed form
  of the ODE also multiplies by (cond + 1) of its 2 x 2 system;
- comparisons of two computations of the same exact field: the sum of the
  error bounds reported by the module (field_error, multiplier rounding);
- no PASS/FAIL against L-107 tables or against the research orientation
  values; the diagnostics A and B of docs/R3_PARSEVAL_REFERENCE.md are not
  asserted here.
"""

import ast
import math
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import pytest

from kkvim_reference.navier import NavierRectangle
from kkvim_reference.parseval_recursion import (
    EPS_TAIL,
    ROUND_ENERGY,
    ROUND_VALUES,
    ParsevalRecursion,
    first_coefficients,
    parseval_coefficients,
    select_odd_index,
)

U = 2.0**-53
RTOL_CLOSED = 16 * U
RTOL_G1 = 32 * U
RTOL_SCALAR = 256 * U
RTOL_QUAD = 1e-13
QUAD_ROUNDING = 2e-14
CLOSED_ROUNDING = 32 * U

R_CONTROL = (0.2, 1 / 3, 0.5, 2 / 3, 1.0, 1.4, 1.5, 2.0, 3.0, 5.0)
R_MAIN = (0.2, 0.5, 1.4, 2.0, 5.0)
S_R3 = 6


@pytest.fixture(scope="module")
def runs():
    """Unit data a = q = D = 1, b = r, s = 1 ... 6 for every control r."""
    return {r: ParsevalRecursion(a=1.0, b=r, q=1.0, D=1.0, s_max=S_R3) for r in R_CONTROL}


def rel(x, y):
    return abs(x - y) / abs(y)


# --- closed form В-29 with alpha = pi b / (2a) -----------------------------------------------


def levy_parts(a, b, q, D):
    alpha = math.pi * b / (2 * a)
    kappa = math.pi / a
    P = (alpha * math.tanh(alpha) + 2) / (2 * math.cosh(alpha))
    R = 1 / (2 * math.cosh(alpha))
    C0 = 4 * q * a**4 / (math.pi**5 * D)
    return alpha, kappa, P, R, C0


def levy_y1(y, a, b, q, D):
    """Y1(y) by В-29 and an absolute rounding allowance of its float evaluation."""
    alpha, kappa, P, R, C0 = levy_parts(a, b, q, D)
    t = kappa * (np.asarray(y, dtype=float) - b / 2)
    terms = (np.ones_like(t), P * np.cosh(t), R * t * np.sinh(t))
    value = C0 * (terms[0] - terms[1] + terms[2])
    allowance = CLOSED_ROUNDING * abs(C0) * (np.abs(terms[0]) + np.abs(terms[1]) + np.abs(terms[2]))
    return value, allowance


def levy_y1_decimal(y, a, b, q, D):
    """Y1, Y1', Y1'' at y in 40-digit decimal arithmetic (the float constants pi, a, b as given)."""
    with localcontext() as ctx:
        ctx.prec = 40
        pi = Decimal(math.pi)
        A, B, Q, DD = Decimal(a), Decimal(b), Decimal(q), Decimal(D)
        kappa = pi / A
        alpha = kappa * B / 2
        ea, ema = alpha.exp(), (-alpha).exp()
        ch_a, sh_a = (ea + ema) / 2, (ea - ema) / 2
        P = (alpha * sh_a / ch_a + 2) / (2 * ch_a)
        R = 1 / (2 * ch_a)
        C0 = 4 * Q * A**4 / (pi**5 * DD)
        t = kappa * (Decimal(y) - B / 2)
        et, emt = t.exp(), (-t).exp()
        ch, sh = (et + emt) / 2, (et - emt) / 2
        f = C0 * (1 - P * ch + R * t * sh)
        f1 = C0 * kappa * (-P * sh + R * (sh + t * ch))
        f2 = C0 * kappa**2 * (-P * ch + R * (2 * ch + t * sh))
        return float(f), float(f1), float(f2)


def gauss_legendre(values_at, ell, panels=16, nodes=32):
    """Composite Gauss-Legendre integrals over (0, ell) of the tuple returned by values_at."""
    t, w = np.polynomial.legendre.leggauss(nodes)
    h = ell / panels
    sums = None
    for p in range(panels):
        mid = (p + 0.5) * h
        for tk, wk in zip(t, w):
            vals = values_at(mid + (h / 2) * tk)
            parts = [wk * (h / 2) * v for v in vals]
            sums = [[x] for x in parts] if sums is None else [s + [x] for s, x in zip(sums, parts)]
    return [math.fsum(s) for s in sums]


@pytest.mark.parametrize("r", R_CONTROL)
def test_s1_field_against_closed_form_v29(runs, r):
    rec = runs[r]
    nx, ny = 8, 20
    W = rec.field_on_grid(1, nx, ny)
    x = rec.a * np.arange(nx + 1) / nx
    y = rec.b * np.arange(ny + 1) / ny
    Y, allowance = levy_y1(y, rec.a, rec.b, rec.q, rec.D)
    X = np.sin(np.pi * x / rec.a)
    expected = np.outer(X, Y)
    budget = rec.halfstep(1).budget.field_error * np.max(np.abs(W)) + np.outer(np.abs(X), allowance) + 4 * U * np.abs(expected)
    assert np.all(np.abs(W - expected) <= budget)


@pytest.mark.parametrize("r", R_CONTROL)
def test_s1_centre_against_v30(runs, r):
    rec = runs[r]
    alpha, _, P, R, C0 = levy_parts(rec.a, rec.b, rec.q, rec.D)
    centre = C0 * (1 - P * math.cosh(0.0))
    allowance = CLOSED_ROUNDING * C0 * (1 + P)
    assert abs(rec.centre_value(1) - centre) <= rec.halfstep(1).budget.field_error * abs(centre) + allowance


def test_s1_centres_reproduce_v14_example(runs):
    # MATH_SPEC.md, 4.1 (В-14), printed final values for b/a = 2, q a^4/D (5 digits, half a unit):
    # start by x 0.010179; start by y 0.010591 (= the plate b x a started by x, ЧВ-R1).
    # The intermediate 0.00066192 printed there is not used: the value is 0.000661929...
    # (see docs/R3_PARSEVAL_REFERENCE.md, 6).
    rec = runs[2.0]
    assert abs(rec.centre_value(1) - 0.010179) <= 0.5e-6
    assert abs(rec.swapped().centre_value(1) - 0.010591) <= 0.5e-6


# --- G2 against integrals of the closed form on (0, b) ---------------------------------------


@pytest.mark.parametrize("r", R_CONTROL)
def test_g2_against_gauss_legendre_of_y1_on_b(runs, r):
    rec = runs[r]
    a, b, q, D = rec.a, rec.b, rec.q, rec.D

    def values(y):
        f, f1, f2 = levy_y1_decimal(y, a, b, q, D)
        return (f * f, f1 * f1, f2 * f2, f)

    A, B, C, I = gauss_legendre(values, b)
    G2 = rec.coefficients(2)
    budget = rec.halfstep(2).budget.input_error + QUAD_ROUNDING
    assert budget <= RTOL_QUAD / 2
    for got, exp in zip((G2.A, G2.B, G2.C, G2.F), (A, B, C, q * I)):
        assert rel(got, exp) <= RTOL_QUAD
    if r != 1.0:
        # the same Parseval sums with the wrong length a are far off: the check sees the length
        wrong = parseval_coefficients(rec.halfstep(1).step.coefficients(), ell=a, q=q)
        assert rel(wrong.A, A) > 1e3 * RTOL_QUAD


def test_g1_is_taken_on_a(runs):
    for r, rec in runs.items():
        G1 = rec.coefficients(1)
        exp = first_coefficients(a=rec.a, q=rec.q)
        assert (G1.A, G1.B, G1.C, G1.F) == (exp.A, exp.B, exp.C, exp.F)
        assert rel(G1.A, rec.a / 2) <= RTOL_G1 and rel(G1.F, 2 * rec.q * rec.a / math.pi) <= RTOL_G1


# --- chain, scalar form, alternation, counters ------------------------------------------------


def scalar_recursion(G, *, D, ell, q, M):
    """G_{s+1} by the scalar form 15.3 (sums S_0, S_2, S_4, T over odd m <= M) on ell."""
    lam = [m * math.pi / ell for m in range(1, M + 1, 2)]
    odd = list(range(1, M + 1, 2))
    P = [G.A * l**4 + 2 * G.B * l**2 + G.C for l in lam]
    kappa = 4 * G.F / (D * math.pi)
    S = [math.fsum(l**k / (m * m * p * p) for l, m, p in zip(lam, odd, P)) for k in (0, 2, 4)]
    T = math.fsum(1 / (m * m * p) for m, p in zip(odd, P))
    half = ell / 2 * kappa * kappa
    return half * S[0], half * S[1], half * S[2], q * 2 * ell / math.pi * kappa * T


@pytest.mark.parametrize("r", R_MAIN)
def test_chain_lengths_and_scalar_form(runs, r):
    rec = runs[r]
    for s in range(1, S_R3 + 1):
        h = rec.halfstep(s)
        ell = rec.b if s % 2 else rec.a
        assert rec.length(s) == ell == h.step.ell
        assert h.variable == ("Y" if s % 2 else "X") and h.s == s and h.k == s // 2
        if s < S_R3:
            nxt = rec.coefficients(s + 1)
            assert nxt == parseval_coefficients(h.step.coefficients(), ell=ell, q=rec.q)
            alt = scalar_recursion(h.coefficients, D=rec.D, ell=ell, q=rec.q, M=h.budget.M)
            for x, y in zip((nxt.A, nxt.B, nxt.C, nxt.F), alt):
                assert rel(x, y) <= RTOL_SCALAR


@pytest.mark.parametrize("r", R_CONTROL)
def test_m_is_chosen_by_rule_i12_with_ell_s(runs, r):
    rec = runs[r]
    for s in range(1, S_R3 + 1):
        h = rec.halfstep(s)
        assert h.budget.M == h.budget.M_rule == select_odd_index(h.coefficients, D=rec.D, ell=rec.length(s), q=rec.q)
        G = h.coefficients
        K = 4 * abs(G.F) * rec.length(s) ** 4 / (rec.D * G.A * math.pi**5)
        assert rel(h.budget.tail_constant, K) <= RTOL_CLOSED
        for t in (h.budget.tail_function_rel, h.budget.tail_A_rel, h.budget.tail_B_rel, h.budget.tail_C_rel, h.budget.tail_F_rel):
            assert 0 < t <= EPS_TAIL


@pytest.mark.parametrize("r", R_CONTROL)
def test_tail_constant_ratio_at_s1_is_one_plus_r2_squared(runs, r):
    # R3_NUMERICAL_SPEC_DRAFT.md, Е-8: K / |c_1| = (1 + r^2)^2 at s = 1 (lambda_1 = pi / b)
    step = runs[r].halfstep(1).step
    assert rel(step.tail_constant / abs(step.coefficient(1)), (1 + r * r) ** 2) <= 64 * U


# --- independent closed-form solution of the half-step ODE (s = 2, 3, 4) -----------------------


def ode_closed_form(G, D, ell, xi):
    """f on (0, ell) solving D(A f'''' - 2B f'' + C f) = F, f = f'' = 0 at both ends, AC > B^2.

    Returns the values and an absolute rounding allowance of the float evaluation.
    """
    disc = G.A * G.C - G.B * G.B
    assert disc > 0
    k = np.sqrt(complex(G.B, math.sqrt(disc)) / G.A)
    fp = G.F / (D * G.C)
    L = ell / 2
    chL = np.cosh(k * L)
    Mx = np.array([[chL.real, chL.imag], [(k * k * chL).real, (k * k * chL).imag]])
    c1, c2 = np.linalg.solve(Mx, [-fp, 0.0])
    ch = np.cosh(k * (np.asarray(xi) - L))
    value = fp + c1 * ch.real + c2 * ch.imag
    size = abs(fp) + abs(c1) * np.abs(ch.real) + abs(c2) * np.abs(ch.imag)
    allowance = CLOSED_ROUNDING * (np.linalg.cond(Mx) + 1) * size
    return value, allowance


@pytest.mark.parametrize("r", R_MAIN)
@pytest.mark.parametrize("s", [2, 3, 4])
def test_later_multipliers_against_closed_form_ode(runs, r, s):
    rec = runs[r]
    h = rec.halfstep(s)
    ell = rec.length(s)
    n = 24
    xi = ell * np.arange(n + 1) / n
    series = np.array(h.step.nodal_values(n))
    exact, allowance = ode_closed_form(h.coefficients, rec.D, ell, xi)
    budget = h.budget.tail_function + h.budget.rounding_values + allowance
    assert np.all(np.abs(series - exact) <= budget)


# --- fields at points and on grids; edges ----------------------------------------------------


def point_rounding(rec, index, length):
    """Extra rounding of a multiplier value away from the nodes: pi u sum m |c_m| (sine_halfstep)."""
    if index == 0:
        return 4 * U * abs(rec.x0_scale)
    coefs = rec.halfstep(index).step.coefficients()
    return 2 * math.pi * U * math.fsum((2 * j + 1) * abs(c) for j, c in enumerate(coefs))


def multiplier_nodes(rec, index, n):
    if index == 0:
        return np.array([rec.x0_scale * math.sin(math.pi * i / n) for i in range(n + 1)])
    return np.array(rec.halfstep(index).step.nodal_values(n))


def multiplier_rounding(rec, index):
    """Nodal rounding plus the extra rounding of the argument away from the nodes."""
    if index == 0:
        return 8 * U * abs(rec.x0_scale)
    return rec.halfstep(index).budget.rounding_values + point_rounding(rec, index, None)


@pytest.mark.parametrize("r", R_MAIN)
@pytest.mark.parametrize("s", [1, 2, 5])
def test_points_equal_nodes(runs, r, s):
    rec = runs[r]
    nx, ny = 6, 10
    W = rec.field_on_grid(s, nx, ny)
    x = rec.a * np.arange(nx + 1) / nx
    y = rec.b * np.arange(ny + 1) / ny
    X, Y = np.meshgrid(x, y, indexing="ij")
    P = rec.field_at_points(s, X, Y)
    ix = s if s % 2 == 0 else s - 1
    iy = s if s % 2 == 1 else s - 1
    Fmax = np.max(np.abs(multiplier_nodes(rec, ix, nx)))
    Gmax = np.max(np.abs(multiplier_nodes(rec, iy, ny)))
    ex, ey = multiplier_rounding(rec, ix), multiplier_rounding(rec, iy)
    # |f_p g_p - f_n g_n| <= |f_p - f_n| |g_p| + |f_n| |g_p - g_n|, plus the product rounding
    tol = ex * (Gmax + ey) + Fmax * ey + 2 * U * Fmax * Gmax
    assert np.max(np.abs(P - W)) <= tol


@pytest.mark.parametrize("r", R_MAIN)
def test_edges_are_exact_zeros(runs, r):
    rec = runs[r]
    for s in (1, 2, 3):
        W = rec.field_on_grid(s, 7, 5)
        assert np.all(W[[0, -1], :] == 0.0) and np.all(W[:, [0, -1]] == 0.0)
        e = np.linspace(0, 1, 5)
        for x, y in ((0 * e, rec.b * e), (rec.a + 0 * e, rec.b * e), (rec.a * e, 0 * e), (rec.a * e, rec.b + 0 * e)):
            assert np.all(rec.field_at_points(s, x, y) == 0.0)


def test_points_outside_plate_rejected(runs):
    rec = runs[2.0]
    for x, y in ((0.5, 2.0 + 1e-12), (1.0 + 1e-12, 1.0), (-0.1, 0.5), (0.5, math.nan)):
        with pytest.raises(ValueError):
            rec.field_at_points(1, x, y)


@pytest.mark.parametrize("r", R_MAIN)
def test_reflection_symmetry(runs, r):
    rec = runs[r]
    for s in (1, 2, 3, 4):
        W = rec.field_on_grid(s, 12, 10)
        tol = 2 * rec.halfstep(s).budget.field_error * np.max(np.abs(W))
        assert np.max(np.abs(W - W[::-1, :])) <= tol and np.max(np.abs(W - W[:, ::-1])) <= tol


# --- scaling, x0 scale, load sign ------------------------------------------------------------------


@pytest.mark.parametrize("r", [0.5, 2.0])
def test_fields_scale_as_q_a4_over_d_at_fixed_r(runs, r):
    base = runs[r]
    lam, q, D = 2.9, -0.6, 0.7
    other = ParsevalRecursion(a=lam, b=lam * r, q=q, D=D, s_max=S_R3)
    for s in range(1, S_R3 + 1):
        w, v = base.field_on_grid(s, 6, 8), other.field_on_grid(s, 6, 8)
        tol = base.halfstep(s).budget.field_error + other.halfstep(s).budget.field_error + 8 * U
        assert np.max(np.abs(v - q * lam**4 / D * w)) <= tol * np.max(np.abs(v))


@pytest.mark.parametrize("scale", [3.7, -2.5])
def test_fields_do_not_depend_on_x0_scale(runs, scale):
    base = runs[2.0]
    other = ParsevalRecursion(a=1.0, b=2.0, q=1.0, D=1.0, s_max=S_R3, x0_scale=scale)
    for s in range(1, S_R3 + 1):
        b, c = base.halfstep(s).budget, other.halfstep(s).budget
        w, v = base.field_on_grid(s, 6, 8), other.field_on_grid(s, 6, 8)
        assert np.max(np.abs(w - v)) <= (b.field_error + c.field_error) * np.max(np.abs(w))
        for x, y in zip(base.coefficients(s).invariants(), other.coefficients(s).invariants()):
            assert rel(y, x) <= b.invariant_error + c.invariant_error


def test_opposite_load_negates_fields_exactly():
    pos = ParsevalRecursion(a=1.0, b=1.4, q=1.3, D=1.0, s_max=4)
    neg = ParsevalRecursion(a=1.0, b=1.4, q=-1.3, D=1.0, s_max=4)
    for s in range(1, 5):
        assert np.array_equal(neg.field_on_grid(s, 6, 7), -pos.field_on_grid(s, 6, 7))
        assert neg.energy(s).energy == pos.energy(s).energy < 0


# --- energy and integral ------------------------------------------------------------------------


@pytest.mark.parametrize("r", R_CONTROL)
def test_energy_forms_monotonicity_and_change_identity(runs, r):
    rec = runs[r]
    for s in range(1, S_R3 + 1):
        e = rec.energy(s)
        assert abs(e.energy_full - e.energy) <= ROUND_ENERGY * abs(e.energy)
        if s > 1:
            p = rec.energy(s - 1)
            allowance = ROUND_ENERGY * (abs(e.energy) + abs(p.energy))
            assert e.delta <= allowance
            assert abs(e.delta + 0.5 * e.change_norm2) <= allowance


@pytest.mark.parametrize("r", R_CONTROL)
def test_energy_of_s1_against_closed_form_integral(runs, r):
    rec = runs[r]
    a, b, q = rec.a, rec.b, rec.q
    alpha, kappa, P, R, C0 = levy_parts(a, b, q, rec.D)
    sh, ch = math.sinh(alpha), math.cosh(alpha)
    terms = (b, 2 * P * sh / kappa, 2 * R / kappa * alpha * ch, 2 * R / kappa * sh)
    I_y = C0 * (terms[0] - terms[1] + terms[2] - terms[3])
    I = 2 * a / math.pi * I_y
    allowance = CLOSED_ROUNDING * abs(C0) * sum(terms) * 2 * a / math.pi
    e = rec.energy(1)
    assert abs(e.integral - I) <= e.error * abs(e.integral) + allowance
    assert e.energy == -0.5 * q * e.integral


@pytest.mark.parametrize("r", R_MAIN)
def test_energy_bounded_below_by_navier_energy(runs, r):
    # Pi(w_s) >= Pi(omega_0) = -q I / 2 (В-23, В-27); I <= I_M + bound + rounding for q > 0
    rec = runs[r]
    nav = NavierRectangle(a=rec.a, b=rec.b, q=rec.q, D=rec.D, nu=0.3).deflection_integral()
    scale = rec.q * rec.a**6 / rec.D
    lower = -0.5 * rec.q * (nav.I_M + nav.I_tail_bound + nav.rounding_J * scale)
    for s in range(1, S_R3 + 1):
        e = rec.energy(s)
        assert e.energy * (1 - e.error) > lower


# --- error budgets, doubled M -------------------------------------------------------------------


@pytest.mark.parametrize("r", R_CONTROL)
def test_budgets_valid_and_growing(runs, r):
    budgets = [runs[r].halfstep(s).budget for s in range(1, S_R3 + 1)]
    assert all(b.valid for b in budgets)
    assert all(b2.input_error > b1.input_error for b1, b2 in zip(budgets, budgets[1:]))
    for b in budgets:
        assert b.first_order_input_error * (1 - 1e-12) <= b.input_error
        assert b.field_error >= b.multiplier_error > 0


@pytest.mark.parametrize("r", [0.2, 2.0, 5.0])
def test_doubled_m_within_both_budgets(runs, r):
    base = runs[r]
    doubled = ParsevalRecursion(a=1.0, b=r, q=1.0, D=1.0, s_max=S_R3, m_scale=2)
    for s in range(1, S_R3 + 1):
        b, d = base.halfstep(s).budget, doubled.halfstep(s).budget
        assert b.M == b.M_rule and d.M == 2 * (d.M_rule + 1) - 1
        w, v = base.field_on_grid(s, 8, 12), doubled.field_on_grid(s, 8, 12)
        assert np.max(np.abs(w - v)) <= (b.field_error + d.field_error) * np.max(np.abs(w))


def test_budget_becomes_invalid_but_values_stay_finite():
    rec = ParsevalRecursion(a=1.0, b=2.0, q=1.0, D=1.0, s_max=30)
    flags = [rec.halfstep(s).budget.valid for s in range(1, 31)]
    assert not flags[-1]
    first_invalid = flags.index(False) + 1
    assert all(flags[: first_invalid - 1]) and not any(flags[first_invalid - 1 :])
    assert rec.energy(30).error == math.inf
    assert np.all(np.isfinite(rec.field_on_grid(30, 4, 4)))


# --- second start by exchange of the sides (ЧВ-R1) -----------------------------------------------


@pytest.mark.parametrize("r", R_MAIN)
def test_swapped_plate_is_b_by_a_started_by_x(runs, r):
    rec = runs[r]
    sw = rec.swapped()
    assert (sw.a, sw.b, sw.q, sw.D, sw.s_max) == (rec.b, rec.a, rec.q, rec.D, rec.s_max)
    # s = 1 of the start by y on a x b: В-29 with a <-> b, alpha' = pi a / (2 b)
    alpha, _, P, R, C0 = levy_parts(rec.b, rec.a, rec.q, rec.D)
    centre = C0 * (1 - P)
    assert abs(sw.centre_value(1) - centre) <= sw.halfstep(1).budget.field_error * centre + CLOSED_ROUNDING * C0 * (1 + P)
    if r != 1.0:
        assert abs(sw.centre_value(1) - rec.centre_value(1)) > 1e3 * sw.halfstep(1).budget.field_error * centre


# --- square: regression of R2 --------------------------------------------------------------------


@pytest.mark.parametrize("kwargs", [{"a": 1.0, "q": 1.0, "D": 1.0}, {"a": 2.3, "q": -0.6, "D": 0.7, "x0_scale": 3.7}])
def test_square_with_b_equal_a_is_bitwise_r2(kwargs):
    r2 = ParsevalRecursion(**kwargs, s_max=6)
    sq = ParsevalRecursion(**kwargs, s_max=6, b=kwargs["a"])
    assert r2.b == r2.a == sq.b and r2.length(1) == r2.length(2) == r2.a
    for s in range(1, 7):
        assert r2.halfstep(s) == sq.halfstep(s)
        assert np.array_equal(r2.field_on_grid(s, 9, 7), sq.field_on_grid(s, 9, 7))
        assert r2.energy(s) == sq.energy(s)


def test_r2_signature_and_defaults_kept():
    rec = ParsevalRecursion(a=1.0, q=1.0, D=1.0)
    assert rec.s_max == 3 and rec.m_scale == 1 and rec.b == rec.a and rec.r == 1.0


# --- input range (ERROR_INPUT, exact, without the Navier tolerance of D-037) --------------------


@pytest.mark.parametrize(("a", "b"), [(1.0, 5.0), (5.0, 1.0), (1.0, 0.2), (2.0, 10.0)])
def test_range_ends_accepted(a, b):
    assert 0.2 <= ParsevalRecursion(a=a, b=b, q=1.0, D=1.0, s_max=1).r <= 5.0


@pytest.mark.parametrize(
    ("a", "b"),
    [(1.0, 5.000000000000001), (1.0, 0.19999999999999998), (1.0, 5.5), (10.0, 1.0), (3.0, 0.6)],
)
def test_ratio_outside_range_is_error_input(a, b):
    # (3.0, 0.6): the float 0.6 is below 3/5, so b/a < 1/5 exactly; see docs/R3_PARSEVAL_REFERENCE.md, 6
    with pytest.raises(ValueError, match="ERROR_INPUT"):
        ParsevalRecursion(a=a, b=b, q=1.0, D=1.0)


@pytest.mark.parametrize("b", [0.0, -1.0, math.nan, math.inf, True])
def test_invalid_b(b):
    with pytest.raises(ValueError):
        ParsevalRecursion(a=1.0, b=b, q=1.0, D=1.0)


def test_module_imports_unchanged():
    import kkvim_reference.parseval_recursion as module

    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module)
    assert names == {"math", "dataclasses", "numpy", "kkvim_reference.sine_halfstep"}
    assert ROUND_VALUES > 0
