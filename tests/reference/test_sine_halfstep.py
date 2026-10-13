"""Own tests of kkvim_reference.sine_halfstep (R2, step 3).

One 1D half-step by the sine series ЧВ-17 (NUMERICAL_SPEC.md, section 15.1)
with the truncation bound ЧВ-20 (section 15.4). The first half-step s = 1 is
checked with the CONTINUOUS coefficients of MATH_SPEC.md, section 5.1, against
the approved Levy reference levy_y1 (MATH_SPEC.md, В-29); grid coefficients
A_h, B_h, C_h, F_h are not used.

Tolerances, fixed before any comparison (not tuned to results):
- U = 2^-53 (unit roundoff of float64);
- RTOL_COEFF = 64 U: c_m costs about 15 roundings of positive quantities
  (lambda_m, lambda_m^2, lambda_m^4, three products, two sums, the
  denominator product, the division), the closed form for comparison about 6,
  the inputs B = pi^2/2, C = pi^4/2, F = 2/pi carry about 3 more; no
  subtraction, so the total relative error is below about 25 U; margin x2.5;
- TOL_PROFILE = 1e-13 absolute (task). Budget: tail <= TAIL_SHARE = 1e-15
  (M chosen from the analytic bound alone), series rounding
  ROUND_FACTOR_SERIES * U * sum|c_m| (section 15.5: sin <= ~u, fsum <= u,
  c_m <= ~8u per term; factor 32), Levy rounding RTOL_LEVY * (4/pi^5) with
  RTOL_LEVY = 1000 U as in tests/reference/test_levy.py; the budget must be
  <= TOL_PROFILE / 10 before the comparison is made.
"""

import ast
import math
from pathlib import Path

import pytest

from kkvim_reference.levy import levy_y1
from kkvim_reference.sine_halfstep import SineHalfStep, odd_index_for_tail

U = 2.0**-53
RTOL_COEFF = 64 * U
TOL_PROFILE = 1e-13
TAIL_SHARE = 1e-15
ROUND_FACTOR_SERIES = 32
RTOL_LEVY = 1000 * U

# Continuous coefficients of the first half-step, a = ell = q = D = 1 (MATH_SPEC.md, 5.1).
S1 = {"A": 0.5, "B": math.pi**2 / 2, "C": math.pi**4 / 2, "F": 2 / math.pi, "D": 1.0, "ell": 1.0}

# Generic data with B, C > 0 and all coefficients different.
GEN = {"A": 0.7, "B": 2.3, "C": 5.1, "F": 1.9, "D": 0.8, "ell": 1.7}


def series_rounding(step):
    return ROUND_FACTOR_SERIES * U * step.abs_coefficient_sum()


# --- coefficients -----------------------------------------------------------


@pytest.mark.parametrize("m", [1, 3, 5, 9, 31])
def test_s1_coefficients_equal_closed_form(m):
    # c_m = 16 / (pi^6 m (1 + m^2)^2) for the first half-step (task supplement, item 2).
    step = SineHalfStep(**S1, M=31)
    expected = 16.0 / (math.pi**6 * m * (1 + m * m) ** 2)
    assert abs(step.coefficient(m) - expected) <= RTOL_COEFF * expected


@pytest.mark.parametrize("m", [1, 3, 7, 21, 101])
def test_generic_coefficients_by_rearranged_formula(m):
    # c_m = 4 F ell^4 / (D m pi (A m^4 pi^4 + 2 B m^2 pi^2 ell^2 + C ell^4)), the same
    # expression multiplied through by ell^4.
    A, B, C, F, D, ell = (GEN[k] for k in ("A", "B", "C", "F", "D", "ell"))
    step = SineHalfStep(**GEN, M=101)
    mp = m * math.pi
    expected = 4 * F * ell**4 / (D * mp * (A * mp**4 + 2 * B * mp**2 * ell**2 + C * ell**4))
    assert abs(step.coefficient(m) - expected) <= RTOL_COEFF * abs(expected)


@pytest.mark.parametrize("m", [2, 4, 10, 32, 1000])
def test_even_coefficients_are_zero(m):
    assert SineHalfStep(**GEN, M=5).coefficient(m) == 0.0


def test_coefficients_list_matches_odd_indices():
    step = SineHalfStep(**GEN, M=9)
    assert step.n_terms == 5
    assert step.coefficients() == [step.coefficient(m) for m in (1, 3, 5, 7, 9)]


# --- s = 1 against Levy -------------------------------------------------------


def s1_step():
    """Series for s = 1 with M chosen only from the analytic tail bound."""
    M = odd_index_for_tail(TAIL_SHARE, A=S1["A"], F=S1["F"], D=S1["D"], ell=S1["ell"])
    return SineHalfStep(**S1, M=M)


def s1_budget(step):
    return step.tail_bound + series_rounding(step) + RTOL_LEVY * 4 / math.pi**5


def test_s1_m_from_tail_bound_and_budget():
    step = s1_step()
    assert step.tail_constant == pytest.approx(16 / math.pi**6, rel=RTOL_COEFF, abs=0)
    assert step.tail_bound <= TAIL_SHARE
    assert s1_budget(step) <= TOL_PROFILE / 10


def test_s1_profile_matches_levy_at_eighths():
    step = s1_step()
    diffs = [
        abs(step.value(k / 8) - float(levy_y1(k / 8, a=1.0, q=1.0, D=1.0))) for k in range(9)
    ]
    assert max(diffs) <= TOL_PROFILE, diffs


def test_s1_nodal_profile_matches_levy():
    step = s1_step()
    n = 64
    nodal = step.nodal_values(n)
    diffs = [abs(nodal[i] - float(levy_y1(i / n, a=1.0, q=1.0, D=1.0))) for i in range(n + 1)]
    assert max(diffs) <= TOL_PROFILE, max(diffs)


def test_nodal_values_agree_with_pointwise_values():
    step = SineHalfStep(**GEN, M=201)
    n = 20
    nodal = step.nodal_values(n)
    pointwise = [step.value(i * step.ell / n) for i in range(n + 1)]
    tol = 2 * series_rounding(step)
    assert max(abs(p - q) for p, q in zip(nodal, pointwise)) <= tol


# --- boundary conditions, symmetry, scaling, sign -------------------------------


def test_zero_at_both_ends():
    step = SineHalfStep(**GEN, M=401)
    nodal = step.nodal_values(10)
    assert nodal[0] == 0.0 and nodal[-1] == 0.0
    assert step.value(0.0) == 0.0
    # sin(m pi) evaluated from the rounded argument: |error| <= pi U m per term.
    weighted = math.fsum(m * abs(c) for m, c in zip(range(1, step.M + 1, 2), step.coefficients()))
    assert abs(step.value(step.ell)) <= 4 * math.pi * U * weighted


@pytest.mark.parametrize("n", [16, 21])
def test_profile_symmetric_about_middle(n):
    step = SineHalfStep(**GEN, M=401)
    nodal = step.nodal_values(n)
    tol = series_rounding(step)
    assert max(abs(nodal[i] - nodal[n - i]) for i in range(n + 1)) <= tol


@pytest.mark.parametrize("factor", [3.7, 0.01])
def test_scaling_in_f_and_d(factor):
    step = SineHalfStep(**GEN, M=101)
    by_f = SineHalfStep(**{**GEN, "F": GEN["F"] * factor}, M=101)
    by_d = SineHalfStep(**{**GEN, "D": GEN["D"] * factor}, M=101)
    base = step.nodal_values(12)
    tol = RTOL_COEFF * max(abs(v) for v in base)
    for b, f_val, d_val in zip(base, by_f.nodal_values(12), by_d.nodal_values(12)):
        assert abs(f_val - factor * b) <= factor * tol
        assert abs(d_val - b / factor) <= tol / factor


def test_negative_f_gives_negative_profile():
    pos = SineHalfStep(**GEN, M=101)
    neg = SineHalfStep(**{**GEN, "F": -GEN["F"]}, M=101)
    n = 12
    assert neg.nodal_values(n) == [-v for v in pos.nodal_values(n)]
    assert all(v < 0 for v in neg.nodal_values(n)[1:-1])
    assert neg.tail_bound == pos.tail_bound


# --- truncation bound --------------------------------------------------------------


def test_coefficients_obey_k_tail_over_m5():
    step = SineHalfStep(**GEN, M=2001)
    k = step.tail_constant
    for m, c in zip(range(1, step.M + 1, 2), step.coefficients()):
        assert abs(c) <= k / m**5 * (1 + RTOL_COEFF)


@pytest.mark.parametrize("M", [1, 5, 15, 41])
def test_tail_bound_covers_truncation(M):
    # Long series (its own bound and rounding are included) stands for f.
    short = SineHalfStep(**GEN, M=M)
    long = SineHalfStep(**GEN, M=4001)
    n = 17
    diff = max(abs(p - q) for p, q in zip(short.nodal_values(n), long.nodal_values(n)))
    allowance = long.tail_bound + series_rounding(short) + series_rounding(long)
    assert short.tail_bound == pytest.approx(
        4 * abs(GEN["F"]) * GEN["ell"] ** 4 / (GEN["D"] * GEN["A"] * math.pi**5) / (8 * M**4),
        rel=RTOL_COEFF,
        abs=0,
    )
    assert diff <= short.tail_bound + allowance


def test_tail_bound_decreases_with_m():
    bounds = [SineHalfStep(**GEN, M=M).tail_bound for M in range(1, 202, 2)]
    assert all(b1 > b2 for b1, b2 in zip(bounds, bounds[1:]))


def test_odd_index_for_tail_is_smallest_odd():
    for target in (1e-6, 1e-10, 1e-15):
        M = odd_index_for_tail(target, A=GEN["A"], F=GEN["F"], D=GEN["D"], ell=GEN["ell"])
        assert M % 2 == 1
        assert SineHalfStep(**GEN, M=M).tail_bound <= target
        if M > 1:
            assert SineHalfStep(**GEN, M=M - 2).tail_bound > target


# --- input validation and independence --------------------------------------------


@pytest.mark.parametrize(
    "changes",
    [
        {"A": 0.0},
        {"B": -1.0},
        {"C": -1.0},
        {"D": 0.0},
        {"ell": -1.0},
        {"F": math.nan},
        {"M": 4},
        {"M": 0},
        {"M": 3.0},
    ],
)
def test_rejects_invalid_input(changes):
    with pytest.raises(ValueError):
        SineHalfStep(**{**GEN, "M": 5, **changes})


def test_rejects_points_outside_interval():
    step = SineHalfStep(**GEN, M=5)
    for xi in (-1e-3, step.ell * 1.01, math.inf):
        with pytest.raises(ValueError):
            step.value(xi)


def test_module_does_not_import_kkvim():
    import kkvim_reference.sine_halfstep as module

    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.append(node.module)
    assert names and all(name.split(".")[0] != "kkvim" for name in names)
