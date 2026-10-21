"""Independent check of the full Levy profile Y1(y) by a sine series (R2, step 2).

For the square a = 1, q = 1, D = 1 the function Y1 of the first half-step
(MATH_SPEC.md, В-29) equals the m = 1 harmonic of the Navier sine series:

    Y1(y) = (16 / pi^6) * sum_{n odd >= 1} sin(n pi y) / [n (1 + n^2)^2].

These are the coefficients c_n of NUMERICAL_SPEC.md, section 15.1 (ЧВ-17)
with A = 1/2, B = pi^2/2, C = pi^4/2, F = 2/pi (MATH_SPEC.md, section 5.1):
A lambda_n^4 + 2 B lambda_n^2 + C = (pi^4 / 2)(1 + n^2)^2.

The expected series is computed here from scratch; kkvim_reference.levy is
used only as the function under test.

Fixed before the computation (not tuned to results):
- points y = k/8, k = 0..8;
- largest odd index M = 4095, i.e. 2048 odd terms;
- truncation bound (NUMERICAL_SPEC.md, section 15.4, item 2):
  |c_n| <= (16/pi^6) n^-5 and sum_{n odd > M} n^-5 <= 1/(8 M^4), so
  |tail| <= (16/pi^6) / (8 M^4) ~ 7.4e-18;
- tolerance: max absolute difference over the nine points <= 1e-15.
"""

import math

from kkvim_reference.levy import levy_y1

M = 4095
N_TERMS = (M + 1) // 2
TOL = 1e-15
PREFACTOR = 16.0 / math.pi**6
TAIL_BOUND = PREFACTOR / (8.0 * M**4)
POINTS_K = range(9)  # y = k/8


def sin_pi_r_over_8(r):
    """sin(pi r / 8) for an integer r, reduced to the first quarter period.

    Exact zeros at r = 0, 8 and exact +-1 at r = 4, 12; no large arguments.
    """
    r %= 16
    sign = 1.0 if r < 8 else -1.0
    r %= 8
    if r > 4:
        r = 8 - r
    return sign * math.sin(math.pi * r / 8)


SIN_TABLE = [sin_pi_r_over_8(r) for r in range(16)]


def m1_series(k):
    """Truncated m = 1 series at y = k/8; sin(n pi k/8) = sin(pi ((n k) mod 16) / 8)."""
    terms = [SIN_TABLE[(n * k) % 16] / (n * (1 + n * n) ** 2) for n in range(1, M + 1, 2)]
    assert len(terms) == N_TERMS
    return PREFACTOR * math.fsum(terms)


def profile_differences():
    """Absolute differences |levy_y1(k/8) - series(k/8)| for k = 0..8."""
    return [
        abs(float(levy_y1(k / 8, a=1.0, q=1.0, D=1.0)) - m1_series(k)) for k in POINTS_K
    ]


def test_truncation_bound_is_far_below_tolerance():
    assert N_TERMS == 2048
    assert TAIL_BOUND < 1e-17
    assert TAIL_BOUND <= TOL / 100


def test_sine_table_reduction():
    # Closed-form values in the first quarter period (relative 4u, u = 1.1e-16).
    exact = {
        1: math.sqrt(2.0 - math.sqrt(2.0)) / 2.0,
        2: math.sqrt(2.0) / 2.0,
        3: math.sqrt(2.0 + math.sqrt(2.0)) / 2.0,
    }
    for r, value in exact.items():
        assert abs(SIN_TABLE[r] - value) <= 4 * 1.1e-16 * value
    assert SIN_TABLE[0] == SIN_TABLE[8] == 0.0
    assert SIN_TABLE[4] == 1.0 and SIN_TABLE[12] == -1.0
    # Exact symmetries sin(pi - t) = sin t and sin(t + pi) = -sin t.
    for r in range(1, 8):
        assert SIN_TABLE[8 - r] == SIN_TABLE[r]
        assert SIN_TABLE[r + 8] == -SIN_TABLE[r]


def test_levy_profile_matches_m1_sine_series():
    differences = profile_differences()
    assert max(differences) <= TOL, differences
