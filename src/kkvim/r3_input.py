"""Input sides of R3: the admissible range of r = b/a (D-038, D-040 RV-2, D-041).

docs/R3_NUMERICAL_SPEC.md 1.2: a and b are finite and > 0; the scientific range
0.2 <= b/a <= 5 (D-036) is checked by the exact symmetric condition of D-038, item 1,

    max(a, b) <= 5 (1 + 4u) min(a, b),  u = 2^-53,

on the rational values of the input floats (``as_integer_ratio``, integer products),
before any rounded division b/a. Both directions a <-> b are checked by the same
inequality. The 4u only covers the representation of the input floats and does not
widen the scientific range. A violation, a non-numeric value or a failed conversion to
float is ERROR_INPUT (``PhysicalInputError``).
"""

import math

import numpy as np

from kkvim.physical import PhysicalInputError

R_MAX = 5  # 0.2 <= b/a <= 5 (D-036)
_INV_U = 2**53  # 1/u, u = 2^-53


def validate_r3_sides(a, b) -> tuple[float, float]:
    """Return (a, b) as floats if both are finite, > 0 and within the range of D-038."""
    a, b = _side("a", a), _side("b", b)
    small, large = (a, b) if a <= b else (b, a)
    p_large, q_large = large.as_integer_ratio()
    p_small, q_small = small.as_integer_ratio()
    # large <= 5 (1 + 4u) small  <=>  p_large q_small / u <= 5 (1/u + 4) p_small q_large
    if p_large * q_small * _INV_U > R_MAX * (_INV_U + 4) * p_small * q_large:
        raise PhysicalInputError(f"sides a={a!r}, b={b!r} are outside 0.2 <= b/a <= 5 (D-038)")
    return a, b


def _side(name: str, value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise PhysicalInputError(f"{name} must be a real number, got {value!r}")
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            value = float(value)
    except (OverflowError, ValueError) as exc:
        raise PhysicalInputError(f"{name} cannot be converted to float: {exc}") from exc
    if not (math.isfinite(value) and value > 0.0):
        raise PhysicalInputError(f"{name} must be finite and positive, got {value!r}")
    return value
