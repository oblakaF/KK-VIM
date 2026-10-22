"""QA cross checks of the R5-V references A (Hencky-Inglis), B (Ritz, L-120), D2 and the published data C (CI part).

What is asserted here:
- program checks: A and B reproduce the printed digits of the published values of the same methods (C);
- the strict one-sided Ritz property: the load work of any exact solution is >= the Ritz work, so the work of A must not
  fall below the work of B (beyond rounding); a violation would show an error of A or B;
- D2: the refinement gives an own error estimate below the plan candidate and both orderings agree within it (plan 1.4).
What is NOT asserted: the accuracy of A, B or D2 relative to omega_0 (OBSERVATION, group B; D-049.8), the comparison of D2
with the external values (engineering candidate B_ext, not approved; scripts/r5_cccc_reference_study.py reports it).
"""

import math

import pytest

from kkvim_reference import r5_cccc_d2 as d2
from kkvim_reference import r5_cccc_hencky as hk
from kkvim_reference import r5_cccc_ritz as rz
from kkvim_reference import r5_cccc_tables as tb

U = 2.0**-53


@pytest.mark.parametrize("p", [p for p in tb.ALPHA if p.role == tb.CONTROL], ids=lambda p: f"{p.source[:5]} r={p.r}")
def test_hencky_reproduces_the_printed_control_values(p):
    """A (finite section N_Y = 200) rounded to the printed step equals the published value (L-120, L-058: Hencky)."""
    long_side = p.r
    s = hk.solve(hk.default_section(long_side, 1.0, 200))
    digits = round(-math.log10(p.step))
    assert round(s.alpha(), digits) == round(p.alpha, digits), (s.alpha(), p.alpha)


@pytest.mark.parametrize("r", [1.0, 2.0, 3.0, 5.0])
def test_hencky_work_is_not_below_the_ritz_lower_bound(r):
    a = hk.solve(hk.default_section(r, 1.0, 200))
    b = rz.solve_smw(rz.RitzProblem(r, 1.0, 120, max(1, round(120 / r))))
    assert a.work() >= b.work() * (1 - 64 * U)  # Ritz work is a lower bound of the exact work


def test_ritz_and_hencky_agree_in_the_printed_digits_of_l120_for_the_square():
    a = hk.solve(hk.HenckyProblem(1.0, 1.0, 200, 200))
    b = rz.solve_smw(rz.RitzProblem(1.0, 1.0, 1000, 1000))
    assert round(a.alpha() * 1e3, 9) == round(b.alpha() * 1e3, 9) == 1.265319087
    assert round(a.work_normalized() * 1e4, 9) == 3.891200775 and round(b.work_normalized() * 1e4, 9) == 3.891200772


@pytest.mark.parametrize("r, n", [(1, 64), (3, 48)])
def test_d2_own_error_estimate_and_orderings(r, n):
    p = d2.D2Problem(float(r), 1.0, r * n, n)
    s1, s2 = d2.solve(p, "MMD_AT_PLUS_A"), d2.solve(p, "COLAMD")
    alpha = s1.alpha()
    assert s1.e_D2 / alpha <= 1e-13 and s2.e_D2 / alpha <= 1e-13  # plan 1.4, item 3 (candidate 1e-13; UNRESOLVED otherwise)
    assert abs(s1.alpha() - s2.alpha()) <= 4 * s1.e_D2 + 4 * s2.e_D2 + 4 * U * alpha  # plan 1.4, item 4
