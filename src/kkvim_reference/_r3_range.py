"""Exact, symmetric R3 aspect-ratio gate shared by all references (D-038).

The scientific scope remains 1/5 <= b/a <= 5 (D-036). The 4u band is a
technical tolerance for binary representation of input lengths, not a
new scientific range. No rounded division is performed during the check.
Admissible input types (D-042): int, float, numpy.integer, numpy.floating;
bool, numpy.bool_ and every other type are rejected before conversion.
"""

import math

import numpy as np

U = 2.0**-53
_U_INV = 2**53
_BAND_UNITS = 4
_RATIO_MAX = 5
_REAL_TYPES = (int, float, np.integer, np.floating)


def _admissible_type(value):
    return not isinstance(value, (bool, np.bool_)) and isinstance(value, _REAL_TYPES)


def in_r3_range(a, b):
    """Test max(a, b) <= 5(1+4u) min(a, b) exactly for positive finite floats."""
    if not (_admissible_type(a) and _admissible_type(b)):
        return False
    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError, OverflowError):
        return False
    if not (math.isfinite(a) and math.isfinite(b) and a > 0.0 and b > 0.0):
        return False
    lo, hi = min(a, b), max(a, b)
    high_num, high_den = hi.as_integer_ratio()
    low_num, low_den = lo.as_integer_ratio()
    return _U_INV * high_num * low_den <= _RATIO_MAX * (_U_INV + _BAND_UNITS) * low_num * high_den


def check_r3_ratio(a, b):
    """Raise ERROR_INPUT if the original lengths fail the symmetric D-038 gate."""
    if not in_r3_range(a, b):
        raise ValueError(
            "ERROR_INPUT: R3 reference requires positive finite a, b and "
            "max(a, b) <= 5(1 + 4u) min(a, b), u = 2^-53 (D-038)"
        )
