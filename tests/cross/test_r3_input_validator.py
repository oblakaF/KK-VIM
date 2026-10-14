"""QA cross tests (R3, step 1): kkvim.r3_input.validate_r3_sides against kkvim_reference._r3_range.in_r3_range.

Both validators implement D-038, item 1 (docs/R3_NUMERICAL_SPEC.md 1.2, RV-2): max(a, b) <= 5 (1 + 4u) min(a, b),
u = 2^-53, exactly on the rational values of the input floats, symmetric in a <-> b; admissible input types by D-042:
int, float, numpy.integer, numpy.floating (not bool, numpy.bool_). The expected verdict is computed here independently:
the type policy is written out below and the inequality is checked with fractions.Fraction on float(x); neither
validator serves as the oracle of the other. Only accept/reject is compared, the exception classes may differ.
"""

import math
import random
import sys
import warnings
from decimal import Decimal
from fractions import Fraction

import numpy as np
import pytest

from kkvim.physical import PhysicalInputError
from kkvim.r3_input import validate_r3_sides
from kkvim_reference._r3_range import check_r3_ratio, in_r3_range

FACTOR = 5 * (1 + 4 * Fraction(1, 2**53))  # 5 (1 + 4u), D-038
FMAX = sys.float_info.max
INT_TYPES = (np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)
FLOAT_TYPES = (np.float16, np.float32, np.float64, np.longdouble)


def expected(a, b):
    """Independent oracle: D-042 type policy, conversion to float, exact rational D-038 inequality."""
    sides = []
    for v in (a, b):
        if type(v) is bool or isinstance(v, np.bool_):
            return False
        if not (type(v) in (int, float) or type(v) in INT_TYPES or type(v) in FLOAT_TYPES):
            return False
        try:
            with warnings.catch_warnings(), np.errstate(all="ignore"):
                warnings.simplefilter("ignore")
                f = float(v)
        except OverflowError:
            return False
        if not (math.isfinite(f) and f > 0.0):
            return False
        sides.append(Fraction(f))
    small, large = sorted(sides)
    return large <= FACTOR * small


def kkvim_accepts(a, b):
    try:
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            result = validate_r3_sides(a, b)
    except PhysicalInputError:
        return False
    assert result == (float(a), float(b))
    return True


def reference_accepts(a, b):
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        verdict = in_r3_range(a, b)
        try:
            check_r3_ratio(a, b)
            raised = False
        except ValueError as exc:
            assert "ERROR_INPUT" in str(exc)
            raised = True
    assert verdict is (not raised)
    return verdict


def assert_agree(a, b, want=None):
    """Both orientations: kkvim, reference and the oracle give the same accept/reject."""
    for x, y in ((a, b), (b, a)):
        oracle = expected(x, y)
        if want is not None:
            assert oracle is want, (x, y)
        assert kkvim_accepts(x, y) is oracle, ("kkvim", x, y)
        assert reference_accepts(x, y) is oracle, ("reference", x, y)


def step(x, k):
    for _ in range(abs(k)):
        x = math.nextafter(x, math.inf if k > 0 else 0.0)
    return x


def largest_admissible(small):
    """Largest float L with L <= 5 (1 + 4u) small (oracle only)."""
    x = min(5.0 * small, FMAX)
    while not Fraction(x) <= FACTOR * Fraction(small):
        x = math.nextafter(x, 0.0)
    while math.isfinite(nxt := math.nextafter(x, math.inf)) and Fraction(nxt) <= FACTOR * Fraction(small):
        x = nxt
    return x


def smallest_admissible(large):
    """Smallest float S with large <= 5 (1 + 4u) S (oracle only)."""
    x = large / 5.0
    while not Fraction(large) <= FACTOR * Fraction(x):
        x = math.nextafter(x, math.inf)
    while (prv := math.nextafter(x, 0.0)) > 0.0 and Fraction(large) <= FACTOR * Fraction(prv):
        x = prv
    return x


SCALES = [5e-324, 1.5e-323, 3.0e-310, 2.0**-1022, 1.0e-300, 2.0**-30, 1e-3, 0.2, 1 / 3, 0.375, 0.7, 1.0, 2.9, 3.0,
          7.0e5, 1.0e300, 3.0e307, FMAX / 5, 1.0e308, FMAX]


# 1-3. Both ends of the range, neighbouring floats, both orientations, subnormal to large finite scales.
@pytest.mark.parametrize("small", SCALES)
@pytest.mark.parametrize("k", [-3, -2, -1, 0, 1, 2, 3])
def test_upper_boundary_and_neighbours(small, k):
    edge = largest_admissible(small)
    x = step(edge, k)
    if not math.isfinite(x):
        assert_agree(small, x, want=False)
        return
    assert_agree(small, x, want=k <= 0)


@pytest.mark.parametrize("large", SCALES)
@pytest.mark.parametrize("k", [-3, -2, -1, 0, 1, 2, 3])
def test_lower_boundary_and_neighbours(large, k):
    edge = smallest_admissible(large)
    x = step(edge, k)
    if x == 0.0:
        assert_agree(large, x, want=False)
        return
    assert_agree(large, x, want=k >= 0)


@pytest.mark.parametrize("small", [s for s in SCALES if 5 * Fraction(s) <= Fraction(FMAX)])
def test_exact_five_both_directions(small):
    large = 5.0 * small
    if Fraction(large) == 5 * Fraction(small):  # b = 5a and a = 5b exactly
        assert_agree(small, large, want=True)
    else:  # 5 small rounded to a float: the oracle decides
        assert_agree(small, large)


def test_ratio_one_and_interior_points_on_all_scales():
    for s in SCALES:
        assert_agree(s, s, want=True)
        for r in (1.4, 1.5, 2.0, 3.0, 4.999):
            if math.isfinite(s * r):
                assert_agree(s, s * r, want=True)
        for r in (5.001, 6.0, 1e3):  # a subnormal product may round back to exactly 5 s: the oracle decides there
            if math.isfinite(s * r):
                assert_agree(s, s * r, want=False if s >= 2.0**-1022 else None)


# 4. Every admissible numeric type of Python and NumPy, also mixed.
@pytest.mark.parametrize("cast", [int, float, *INT_TYPES, *FLOAT_TYPES], ids=lambda c: c.__name__)
def test_admissible_numeric_types(cast):
    assert_agree(cast(1), cast(5), want=True)
    assert_agree(cast(2), cast(3), want=True)
    assert_agree(cast(7), cast(7), want=True)
    assert_agree(cast(1), cast(6), want=False)
    assert_agree(cast(0), cast(5), want=False)
    if cast in FLOAT_TYPES or cast is float:
        assert_agree(cast(0.25), cast(1.25), want=True)
        assert_agree(cast(-1), cast(-5), want=False)
    if cast is int or np.issubdtype(cast, np.signedinteger):
        assert_agree(cast(-1), cast(-5), want=False)


@pytest.mark.parametrize("cast_a", [int, float, np.int64, np.uint16, np.float16, np.float32, np.longdouble],
                         ids=lambda c: c.__name__)
@pytest.mark.parametrize("cast_b", [int, float, np.int32, np.uint64, np.float32, np.float64],
                         ids=lambda c: c.__name__)
def test_mixed_types(cast_a, cast_b):
    assert_agree(cast_a(4), cast_b(20), want=True)
    assert_agree(cast_a(4), cast_b(21), want=False)


def test_float32_and_float16_are_checked_on_their_exact_values():
    """0.2 and 5 in float32/float16 are other binary numbers than in float64; the oracle uses their exact values."""
    for cast in (np.float32, np.float16):
        for a, b in ((1.0, 0.2), (0.2, 1.0), (1.0, 5.0), (0.7, 3.5), (3.0, 0.6)):
            assert_agree(cast(a), cast(b))


# 5. Rejected inputs, in both positions.
BAD = ["1", "5", "nan", Fraction(1), Fraction(5, 1), Decimal("1"), Decimal("5"), True, False, np.bool_(True),
       np.bool_(False), None, math.nan, -math.nan, math.inf, -math.inf, np.float64("nan"), np.float32("inf"),
       np.float16("-inf"), complex(1, 0), np.complex128(1), np.array(1.0), [1.0], (1.0,), b"1", 0, 0.0, -0.0, -1,
       -1.0, 5e-324 * -1]


@pytest.mark.parametrize("bad", BAD, ids=repr)
@pytest.mark.parametrize("good", [1.0, 5.0, 0.2, 1])
def test_rejected_values_and_types(bad, good):
    assert_agree(good, bad, want=False)
    assert_agree(bad, bad, want=False)


# 6. Overflow of the conversion to float.
HUGE_INTS = [10**400, -(10**400), 2**1024, 2**1024 - 2**970]  # the last one rounds (ties-to-even) to 2^1024


@pytest.mark.parametrize("big", HUGE_INTS, ids=lambda v: f"int~2^{v.bit_length()}")
@pytest.mark.parametrize("good", [1.0, FMAX, 1])
def test_int_overflow_is_rejected(big, good):
    with pytest.raises(OverflowError):
        float(big)
    assert_agree(good, big, want=False)


def test_int_just_below_overflow_rounds_to_the_largest_float():
    big = 2**1024 - 2**970 - 1
    assert float(big) == FMAX
    assert_agree(big, FMAX, want=True)
    edge = smallest_admissible(FMAX)
    assert_agree(big, edge, want=True)
    assert_agree(big, math.nextafter(edge, 0.0), want=False)


@pytest.mark.skipif(np.finfo(np.longdouble).max <= FMAX, reason="numpy.longdouble is the binary64 double here")
def test_longdouble_beyond_double_range_is_rejected():
    with np.errstate(all="ignore"):
        big = np.longdouble(FMAX) * np.longdouble(16)
        tiny = np.longdouble(5e-324) / np.longdouble(16)
    assert np.isfinite(big) and tiny > 0
    assert_agree(big, 1.0, want=False)
    assert_agree(tiny, 1.0, want=False)


# 7. Integers above 2^53: the check is made after conversion to float.
def test_large_integers_are_checked_after_rounding_to_float():
    # min = 2^60 + 1 -> 2^60; max = 5 2^60 + 2565 -> 5 2^60 + 3072 (ulp 1024): exact integers admissible, floats not.
    a, b = 2**60 + 1, 5 * 2**60 + 2565
    assert Fraction(b) <= FACTOR * Fraction(a)
    assert (float(a), float(b)) == (2.0**60, 5 * 2.0**60 + 3072)
    assert_agree(a, b, want=False)
    assert_agree(np.uint64(a), np.uint64(b), want=False)
    # min = 2^60 - 1 -> 2^60; max = 5 2^60 + 2560 -> 5 2^60 + 2048 (tie to even): exact integers not admissible, floats are.
    a, b = 2**60 - 1, 5 * 2**60 + 2560
    assert not Fraction(b) <= FACTOR * Fraction(a)
    assert (float(a), float(b)) == (2.0**60, 5 * 2.0**60 + 2048)
    assert_agree(a, b, want=True)
    assert_agree(np.int64(a), np.int64(b), want=True)


def test_integers_around_two_to_the_53():
    for base in (2**53 - 1, 2**53, 2**53 + 1, 2**54 + 3, 2**63 - 1, 2**64 - 1, 3**40):
        for mult in (5, 6):
            assert_agree(base, mult * base)
            assert_agree(base, mult * base + 1)
            assert_agree(base, mult * base - 1)
        if base < 2**63:
            assert_agree(np.int64(base), base * 5)


# 8. Full agreement of accept/reject on identical inputs: seeded sweep of floats, integers and NumPy scalars.
def _random_side(rng, ref):
    kind = rng.randrange(6)
    if kind == 0:
        return step(ref * rng.choice([0.2, 5.0]), rng.randint(-4, 4))
    if kind == 1:
        return ref * rng.uniform(0.15, 6.5)
    if kind == 2:
        return rng.uniform(1.0, 2.0) * 2.0 ** rng.randint(-1074, 1023)
    if kind == 3:
        return rng.choice([float, np.float64, np.float32, np.float16])(ref * rng.choice([0.2, 1.0, 5.0]))
    if kind == 4:
        n = rng.randint(1, 2**70)
        return rng.choice([int, lambda v: np.int64(v % 2**63), lambda v: np.uint64(v % 2**64)])(n)
    return rng.choice(BAD)


def _random_pair(rng):
    ref = rng.uniform(1.0, 2.0) * 2.0 ** rng.randint(-1070, 1018)
    a = rng.choice([ref, int(ref) if 1.0 <= ref < 2.0**80 else ref, np.float64(ref)])
    return a, _random_side(rng, ref)


def test_seeded_sweep_full_agreement():
    rng = random.Random(20261014)
    cases = disagreements = 0
    with warnings.catch_warnings(), np.errstate(all="ignore"):  # float16/float32 casts may overflow to inf
        warnings.simplefilter("ignore")
        pairs = [_random_pair(rng) for _ in range(20000)]
    for a, b in pairs:
        for x, y in ((a, b), (b, a)):
            cases += 1
            oracle = expected(x, y)
            if kkvim_accepts(x, y) is not oracle or reference_accepts(x, y) is not oracle:
                disagreements += 1
    assert cases == 40000 and disagreements == 0
