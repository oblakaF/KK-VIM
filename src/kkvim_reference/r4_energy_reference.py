"""QA reference candidate for R4: the energy Pi(omega_0) of the exact SSSS solution with its own error budget.

R4_NUMERICAL_SPEC 9.5.2 (НБ-5; РВ4-8: separate energy reference, in particular for N >= 4; NOT_QUALIFIED until QA).
The budget of the deflection is NOT transferred to the energy. No import of ``kkvim`` (D-028).

Formula [MATH_SPEC В-27, item 6; Navier, L-107 (131)]: Pi(omega_0) = -q J / 2,
    J = int int omega_0 = (64 q a b / (pi^8 D)) S,   S = sum_{m, n odd} 1 / (m^2 n^2 (m^2/a^2 + n^2/b^2)^2).

Truncation (own, strict): with odd m, n <= M kept, the remainder splits into {m > M, any n} and {m <= M, n > M};
(m^2/a^2 + n^2/b^2)^2 >= m^4/a^4 and >= n^4/b^4, sum_{odd m > M} m^-6 <= 1/(10 M^5) (NUMERICAL_SPEC 15.4, item 2),
sum_{odd} m^-2 = pi^2/8, hence
    0 <= S - S_M <= (a^4 + b^4) pi^2 / (80 M^5).

Rounding (operation count to first order in the standard IEEE model: round to nearest, no underflow or overflow;
ASSUMPTION: x ** 8 of the double fl(pi) is within one ulp, <= 2u relative - checked by tests/cross/test_r4_dst_qa.py):
    term 1/((m^2 n^2) (m^2/a^2 + n^2/b^2)^2): a^2, b^2, m^2/a^2, n^2/b^2 and their sum <= 3u; the square <= 7u;
        the product with m^2 n^2 <= 8u (m^2 n^2 is an odd integer, exact in binary64 iff m^2 n^2 <= 2^53, guaranteed for
        odd M <= 9741 since 9741^4 < 2^53 < 9743^4; for 9741 < M <= 9999 some products are rounded: one more u, <= 9u);
        the reciprocal <= 9u (10u);
    S: rows and total by math.fsum, correctly rounded: + u each, <= 11u (12u);
    constant 64 a b / (pi^8 D): a b (u), pi^8 = fl(pi)^8 (8 x 0.35u for |pi - fl(pi)| <= 0.35u pi) and pow (2u), x D (u),
        the division (u): <= 7.8u;
    J = q * const * S and Pi = -0.5 q J: three multiplications, 3u;
    total <= 21.8u for odd M <= 9741 and <= 22.8u for 9741 < M <= 9999, both <= ROUNDING_REL = 24u (+ O(u^2)).
The count is not a machine-checked proof. Checked against an independent semi-analytic evaluation (closed form in n),
the qualified Navier integral and, in a temporary QA environment, 50-digit sums (Reviewer 3aaca8e: <= 6.2u at M = 4001).
"""

import math

import numpy as np

__all__ = ["energy_sum", "omega0_energy", "S_TAIL", "ROUNDING_REL", "closed_form_sum"]

ROUNDING_REL = 24 * 2.0**-53


def S_TAIL(a, b, M):
    """Strict bound of S - S_M (odd m, n <= M kept)."""
    return (a**4 + b**4) * math.pi**2 / (80.0 * M**5)


def energy_sum(a, b, M):
    """S_M = sum over odd m, n <= M of 1 / (m^2 n^2 (m^2/a^2 + n^2/b^2)^2), rows and total by math.fsum.

    Odd M <= 9999; the products m^2 n^2 are exact for M <= 9741 and rounded once for some terms above (module docstring).
    """
    if isinstance(M, bool) or not isinstance(M, int) or M < 1 or M % 2 == 0 or M > 9999:
        raise ValueError("M must be an odd integer in [1, 9999]")
    a2, b2 = a * a, b * b
    n = np.arange(1, M + 1, 2, dtype=float)
    n2 = n * n
    y = n2 / b2
    rows = []
    for m in range(1, M + 1, 2):
        m2 = float(m * m)
        terms = 1.0 / ((m2 * n2) * (m2 / a2 + y) ** 2)
        rows.append(math.fsum(terms.tolist()))
    return math.fsum(rows)


def omega0_energy(a, b, q=1.0, D=1.0, M=4001):
    """(Pi(omega_0), absolute error bound): Pi = -q J/2 with J = 64 q a b S/(pi^8 D); bound = tail + rounding."""
    S = energy_sum(a, b, M)
    const = 64.0 * a * b / (math.pi**8 * D)
    J = q * const * S
    pi0 = -0.5 * q * J
    tail = 0.5 * q * q * const * S_TAIL(a, b, M)
    return pi0, tail + ROUNDING_REL * abs(pi0)


def closed_form_sum(a, b, M):
    """Semi-analytic S: closed form of the n-sum for every odd m <= M plus the m-tail bound (independent check).

    With beta = 1/b^2 and c = m b / a: sum_{n odd} 1/(n^2 (m^2/a^2 + beta n^2)^2) = b^4 [ (pi^2/8 - S1(c))/c^4 - S2(c)/c^2 ],
    S1(c) = sum_{n odd} 1/(n^2 + c^2) = pi tanh(pi c/2)/(4c), S2(c) = sum_{n odd} 1/(n^2 + c^2)^2 = -S1'(c)/(2c).
    The subtractions lose accuracy for small c (here c >= b/a >= 0.2, measured in the report); the m-tail is
    <= a^4 pi^2/(80 M^5) (the n-sum is exact).
    """
    total = []
    for m in range(1, M + 1, 2):
        c = m * b / a
        t = math.tanh(math.pi * c / 2)
        S1 = math.pi * t / (4 * c)
        e = math.exp(-math.pi * c)
        sech2 = 4.0 * e / (1.0 + e) ** 2  # sech^2(pi c/2) without 1 - tanh^2
        S2 = math.pi * t / (8 * c**3) - math.pi**2 * sech2 / (16 * c**2)
        inner = b**4 * ((math.pi**2 / 8 - S1) / c**4 - S2 / c**2)
        total.append(inner / m**2)
    return math.fsum(total)
