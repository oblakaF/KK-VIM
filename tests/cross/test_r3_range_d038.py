"""Integration tests for D-038: one symmetric 4u gate across all R3 references.

The expected answer is calculated using Fraction from the *input* floats,
independently of the shared integer-arithmetic validator. R2 contracts stay
covered by their original independent regression suites.
"""

import math
from fractions import Fraction

import numpy as np
import pytest

from kkvim_reference._r3_range import check_r3_ratio, in_r3_range
from kkvim_reference.levy import (
    levy_alpha_141,
    levy_centre_140,
    levy_centre_bounds,
    levy_centre_wbar,
    levy_orientation,
    levy_series_w,
)
from kkvim_reference.navier import NavierRectangle, deflection_tail_bound_rect
from kkvim_reference.parseval_recursion import ParsevalRecursion

U = Fraction(1, 2**53)
FACTOR = 5 * (1 + 4 * U)


def _expected(a, b):
    """Independent exact oracle for max(a,b) <= 5(1+4u) min(a,b)."""
    return Fraction(max(a, b)) <= FACTOR * Fraction(min(a, b))


def _ulps(x, count):
    for _ in range(abs(count)):
        x = math.nextafter(x, math.inf if count > 0 else 0.0)
    return x


def _all_three(a, b):
    NavierRectangle(a=a, b=b, q=1.0, D=1.0, nu=0.3)
    ParsevalRecursion(a=a, b=b, q=1.0, D=1.0, s_max=1)
    levy_orientation(a, b)


@pytest.mark.parametrize("a", [1.0, 3.0, 0.7, 2.9])
@pytest.mark.parametrize("target", [0.2, 5.0])
@pytest.mark.parametrize("k", [-12, -4, -3, -2, -1, 0, 1, 2, 3, 4, 12])
def test_exact_gate_and_all_references_agree_at_boundaries(a, target, k):
    b = _ulps(target * a, k)
    expected = _expected(a, b)
    assert in_r3_range(a, b) == expected
    assert in_r3_range(b, a) == expected
    if expected:
        check_r3_ratio(a, b)
        _all_three(a, b)
        _all_three(b, a)
    else:
        for aa, bb in ((a, b), (b, a)):
            with pytest.raises(ValueError, match="ERROR_INPUT"):
                check_r3_ratio(aa, bb)
            for ctor in (
                lambda: NavierRectangle(a=aa, b=bb, q=1.0, D=1.0, nu=0.3),
                lambda: ParsevalRecursion(a=aa, b=bb, q=1.0, D=1.0, s_max=1),
                lambda: levy_orientation(aa, bb),
            ):
                with pytest.raises(ValueError, match="ERROR_INPUT"):
                    ctor()


@pytest.mark.parametrize(
    ("a", "b"),
    [
        (3.0, 0.6),
        (0.6, 3.0),
        (2.9, 5 * 2.9),
        (5 * 2.9, 2.9),
        (0.7, 0.13999999999999993),
        (1.0, _ulps(5.0, 2)),
        (1.0, _ulps(0.2, -3)),
    ],
)
def test_accepted_borderline_plate_survives_swapped_and_internal_levy_paths(a, b):
    assert _expected(a, b)
    p = NavierRectangle(a=a, b=b, q=1.0, D=1.0, nu=0.3)
    s = ParsevalRecursion(a=a, b=b, q=1.0, D=1.0, s_max=1)
    assert p.swapped().swapped() == p
    assert s.swapped().swapped().a == s.a
    assert s.swapped().swapped().b == s.b
    # A rounded oriented ratio may lie outside the public gate. The original
    # a,b inputs were certified; internal tail selection must not reject them.
    c = levy_centre_140(a=a, b=b, q=1.0, D=1.0)
    bounds = levy_centre_bounds(a=a, b=b, q=1.0, D=1.0)
    assert math.isfinite(c) and c > 0
    assert bounds.truncation >= 0
    # Direct callers supply only a rounded ratio; for an already-certified
    # plate its rounded b/a can independently fall outside the 4u gate.
    if in_r3_range(1.0, b / a):
        assert math.isfinite(levy_centre_wbar(b / a))
    else:
        with pytest.raises(ValueError, match="ERROR_INPUT"):
            levy_centre_wbar(b / a)


@pytest.mark.parametrize(
    "r",
    [
        _ulps(5.0, 2),
        _ulps(5.0, 3),
        _ulps(0.2, -3),
        _ulps(0.2, -4),
        1.0,
    ],
)
def test_public_ratios_use_shared_gate(r):
    expected = _expected(1.0, r)
    if expected:
        assert deflection_tail_bound_rect(9, r) > 0
        assert levy_centre_wbar(r) > 0
        if r >= 1.0:
            assert levy_alpha_141(r) > 0
    else:
        with pytest.raises(ValueError, match="ERROR_INPUT"):
            deflection_tail_bound_rect(9, r)
        with pytest.raises(ValueError, match="ERROR_INPUT"):
            levy_centre_wbar(r)
        if r >= 1.0:
            with pytest.raises(ValueError, match="ERROR_INPUT"):
                levy_alpha_141(r)


@pytest.mark.parametrize("x", [None, True, math.inf, -1.0, 0.0, math.nan, "not-a-number"])
def test_invalid_input_is_rejected_by_shared_gate(x):
    assert not in_r3_range(1.0, x)
    with pytest.raises(ValueError, match="ERROR_INPUT"):
        check_r3_ratio(1.0, x)


def test_levy_harmless_exponential_underflow_is_ignored_locally():
    with np.errstate(all="raise"):
        val = levy_series_w(0.3, 2.5, a=1.0, b=5.0, q=1.0, D=1.0, M=1131)
    assert math.isfinite(float(val)) and val > 0
