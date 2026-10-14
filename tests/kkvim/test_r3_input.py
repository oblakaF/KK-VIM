"""Own tests of kkvim.r3_input: admissible sides a, b of R3 (R3, step 1).

Checks follow docs/R3_NUMERICAL_SPEC.md 1.2 and D-038, item 1:
max(a, b) <= 5 (1 + 4u) min(a, b), u = 2^-53, exactly on the rational values of the
input floats, symmetric in a <-> b; violations are ERROR_INPUT. The oracle below is
exact rational arithmetic (fractions.Fraction). Cross tests against the reference
validator are QA's (tests/cross).
"""

import math
import random
from decimal import Decimal
from fractions import Fraction

import numpy as np
import pytest

from kkvim.coefficients import ERROR_INPUT
from kkvim.physical import PhysicalInputError
from kkvim.r3_input import validate_r3_sides

BOUND = 5 * (1 + 4 * Fraction(1, 2**53))  # 5 (1 + 4u), D-038


def admissible(a: float, b: float) -> bool:
    small, large = sorted((Fraction(a), Fraction(b)))
    return large <= BOUND * small


def assert_rejected(a, b):
    with pytest.raises(PhysicalInputError) as info:
        validate_r3_sides(a, b)
    assert info.value.code == ERROR_INPUT and isinstance(info.value, ValueError)


def largest_admissible(small: float) -> float:
    """Largest float L with L <= 5 (1 + 4u) small, found by stepping from 5 small."""
    x = 5.0 * small
    while not admissible(small, x):
        x = math.nextafter(x, 0.0)
    while admissible(small, math.nextafter(x, math.inf)):
        x = math.nextafter(x, math.inf)
    return x


@pytest.mark.parametrize("a, b", [(1.0, 1.0), (1.0, 2.0), (2.0, 1.0), (3.0, 1.0), (1.0, 1.4), (1.0, 1.5),
                                  (2.0, 3.0), (3.0, 2.0), (1.0, 1 / 3), (1.0, 0.5), (1.0, 4.999), (0.25, 1.0)])
def test_admissible_sides(a, b):
    assert validate_r3_sides(a, b) == (a, b)


@pytest.mark.parametrize("small", [1.0, 3.0, 0.375, 2.0**-30, 1e-3, 7.0e5])
def test_exact_ratio_five_both_directions(small):
    large = 5.0 * small
    assert Fraction(large) == 5 * Fraction(small)  # b = 5a and a = 5b exactly
    assert validate_r3_sides(small, large) == (small, large)
    assert validate_r3_sides(large, small) == (large, small)


SCALES = [1.0, 0.2, 1 / 3, 0.75, 1.0e-300, 2.0**-1022, 5e-324, 3.0e-310, 1.0e300, 3.0e307]


@pytest.mark.parametrize("small", SCALES)
def test_neighbouring_floats_of_the_bound(small):
    top = largest_admissible(small)
    above = math.nextafter(top, math.inf)
    # 5 (1 + 4u) min(a, b) is never a float: the bound lies strictly between top and above
    assert Fraction(top) < BOUND * Fraction(small) < Fraction(above)
    for large in (math.nextafter(top, 0.0), top):
        assert validate_r3_sides(small, large) == (small, large)
        assert validate_r3_sides(large, small) == (large, small)
    for large in (above, math.nextafter(above, math.inf)):
        assert_rejected(small, large)
        assert_rejected(large, small)


def test_tolerance_admits_floats_above_exact_five():
    top = largest_admissible(1.0)
    assert top > 5.0  # b/a > 5 in floating point, yet inside the 4u band of D-038
    assert validate_r3_sides(1.0, top) == (1.0, top)
    assert validate_r3_sides(top, 1.0) == (top, 1.0)


@pytest.mark.parametrize("small, large", [("0x1.9755d4de3fa85p-14", "0x1.fd2b4a15cf92ap-12"),
                                          ("0x1.7af027a7c486ap+5", "0x1.d9ac3191b5a88p+7")])
def test_exact_check_where_rounded_float_bound_would_reject(small, large):
    small, large = float.fromhex(small), float.fromhex(large)
    assert admissible(small, large)
    assert not large <= 5.0 * (1.0 + 4.0 * 2.0**-53) * small  # rounded products give the wrong verdict
    assert validate_r3_sides(small, large) == (small, large)
    assert validate_r3_sides(large, small) == (large, small)


def test_ratio_one_fifth_and_decimal_inputs():
    for a, b in [(1.0, 0.2), (5.0, 1.0), (1.0, 5.0), (0.2, 1.0), (0.1, 0.5), (0.5, 0.1)]:
        assert admissible(a, b)
        assert validate_r3_sides(a, b) == (a, b)


def test_random_pairs_near_the_bound_against_exact_oracle():
    rng = random.Random(20261014)
    for _ in range(4000):
        small = math.ldexp(rng.uniform(1.0, 2.0), rng.randint(-1070, 1018))
        large = 5.0 * small
        for _ in range(rng.randint(-4, 4) + 4):
            large = math.nextafter(large, math.inf)
        for _ in range(rng.randint(0, 3)):
            large = math.nextafter(large, 0.0)
        for a, b in ((small, large), (large, small)):
            if admissible(a, b):
                assert validate_r3_sides(a, b) == (a, b)
            else:
                assert_rejected(a, b)


@pytest.mark.parametrize("a, b", [(1.0, 6.0), (6.0, 1.0), (1.0, 0.1), (0.1, 1.0), (1e-300, 1.0), (1.0, 1e300),
                                  (1.0, 5.000001), (5.000001, 1.0), (5e-324, 3e-323)])
def test_out_of_range_is_error_input(a, b):
    assert_rejected(a, b)


INVALID = [0, 0.0, -0.0, -1, -1.0, -1e-300, -5e-324, math.nan, math.inf, -math.inf, np.nan, np.float32("inf"),
           np.float64("-inf"), True, False, np.bool_(True), "1", "1.0", None, 1 + 0j, Fraction(1), Decimal("1"), [1.0],
           10**400, -(10**400)]


@pytest.mark.parametrize("bad", INVALID, ids=repr)
def test_invalid_side_is_error_input(bad):
    assert_rejected(bad, 1.0)
    assert_rejected(1.0, bad)
    assert_rejected(bad, bad)


def test_conversion_overflow_of_extended_float_is_error_input():
    huge = np.longdouble(np.finfo(np.longdouble).max)
    if np.finfo(np.longdouble).max <= np.finfo(np.float64).max:
        pytest.skip("long double is the same as double on this platform")
    assert_rejected(huge, 1.0)
    assert_rejected(1.0, huge)


@pytest.mark.parametrize(
    "a, b",
    [(1, 3), (np.int32(2), np.int64(3)), (np.uint16(4), np.int8(1)), (np.float16(0.5), np.float32(1.5)),
     (np.float64(1.0), 5), (np.int64(1), np.float32(0.2))],
)
def test_python_and_numpy_numeric_types(a, b):
    result = validate_r3_sides(a, b)
    assert result == (float(a), float(b))
    assert all(type(v) is float for v in result)


def test_numpy_float32_one_fifth_is_checked_on_its_float_value():
    b = np.float32(0.2)  # 0.200000002980..., ratio 1/b slightly below 5
    assert admissible(1.0, float(b))
    assert validate_r3_sides(1.0, b) == (1.0, float(b))
    assert_rejected(1.0, np.float32(0.19))
