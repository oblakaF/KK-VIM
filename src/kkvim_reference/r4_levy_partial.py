"""QA reference candidates for R4, s = 1: x-oriented partial sums of the Levy series and of the Navier series.

R4_NUMERICAL_SPEC 9.1 (ЧВ4-16, НБ-12; principle accepted by D-046, РВ4-8; qualification NOT_QUALIFIED until QA).
With the odd-sine start X0_j = sin(m_j pi x / a), m_j = 2j - 1, the first continuous half-step of KK-VIM is
w1 = sum_j s_{m_j}(x) Y_{m_j}(y) (MATH_SPEC ВБ-23), i.e. the terms m = 1, 3, ..., 2N - 1 of

    (139)  w = (4 q a^4 / (pi^5 D)) sum_m m^-5 T_m(y) sin(m pi x / a),   alpha_m = m pi b / (2a)

[L-107, eq. (139), p. 136] summed in the ORIGINAL orientation (sines along x, the side a, for every b/a); the existing
``levy_series_w`` sums along the shorter side and is not used for this purpose (R3_NUMERICAL_SPEC 9.2, R4 9.1).

T_m is evaluated in the stable form ЧВ-Л1 (docs/R3_LEVY_REFERENCE_DRAFT.md, 3.1-3.2; algebra verified by the audit
`9919f32`) implemented here independently: s = 2 min(y, b - y) / b, z = alpha s, g = exp(-2 alpha), E = exp(-alpha(2 - s)),
    (1 + g) T = [-expm1(-z) - (z/2) exp(-z)] + E [expm1(-z) + alpha expm1(-2z) / (1 + g) + z/2],
all exponents <= 0 (no overflow for any alpha), T = 0 exactly on the edges.

Independent cross-check: the same partial sum of the Navier double series [L-107, eq. (131)],
    w_m(x, y) = sin(m pi x/a) sum_{n odd <= M} 16 q / (pi^6 D m n (m^2/a^2 + n^2/b^2)^2) sin(n pi y / b),
whose tail is bounded strictly: (m^2/a^2 + n^2/b^2)^2 >= n^4/b^4 and sum_{n odd > M} n^-5 <= 1/(8 M^4) give
    |w - w_M| <= 16 |q| b^4 / (pi^6 D) * sum_j 1/m_j * 1/(8 M^4)   (uniformly in x, y).

Rounding of the Levy partial sum: first-order MODEL (not a strict bound) per term relative 12u (bracket <= 4u
observed in R3, sine and products), plus the summation of N terms: |err| <= C0 (12u + gamma_N) sum_j m_j^-5 |T_mj| |sin|,
C0 = 4|q| a^4 / (pi^5 D). DOMAIN: alpha_m >= pi/2 only, the range for which ЧВ-Л1 was checked in R3 (r_L >= 1). QA
measurement against 50-digit arithmetic (report, section 6.1): for alpha < pi/2 the bracket T is a small difference of
terms of order one and the stable form loses accuracy, about 200 u at alpha = 0.31 (r = 0.2, m = 1) and 1200 u at
alpha = 0.16; there the model does not apply (ValueError) and the m-truncated Navier sum is the s = 1 reference.
"""

import math

import numpy as np

__all__ = ["bracket", "levy_x_partial_at_nodes", "levy_x_partial_rounding_model", "navier_m_truncated_at_nodes",
           "navier_m_tail_bound"]

U_ROUND = 2.0**-53


def bracket(alpha, s):
    """T(alpha, s) of (139) in the stable form ЧВ-Л1; s = 1 - |2 eta / b| in [0, 1]."""
    s = np.asarray(s, dtype=float)
    z = alpha * s
    g = math.exp(-2.0 * alpha)
    E = np.exp(-alpha * (2.0 - s))
    em1 = np.expm1(-z)
    f = -em1 - 0.5 * z * np.exp(-z)
    rest = E * (em1 + alpha * np.expm1(-2.0 * z) / (1.0 + g) + 0.5 * z)
    return (f + rest) / (1.0 + g)


def _sin_nodes(n, m):
    """sin(m pi i / n), i = 0..n, integer reduction to [0, pi/2]."""
    r = np.mod(np.arange(n + 1) * m, 2 * n)
    sign = np.where(r < n, 1.0, -1.0)
    r1 = np.mod(r, n)
    return sign * np.sin(np.pi * np.minimum(r1, n - r1) / n)


def _s_nodes(ny):
    """s = 2 min(y_j, b - y_j)/b at y_j = j b / ny: exact rational 2 min(j, ny - j) / ny, one rounding."""
    j = np.arange(ny + 1)
    return 2.0 * np.minimum(j, ny - j) / ny


def levy_x_partial_at_nodes(a, b, q, D, modes, nx, ny):
    """sum over the given odd m of the (139) terms at the nodes x_i = i a/nx, y_j = j b/ny (edges included)."""
    C0 = 4.0 * q * a**4 / (math.pi**5 * D)
    s = _s_nodes(ny)
    W = np.zeros((nx + 1, ny + 1))
    for m in modes:
        alpha = m * math.pi * b / (2.0 * a)
        W += np.outer(_sin_nodes(nx, m), C0 * bracket(alpha, s) / m**5)
    return W


def levy_x_partial_rounding_model(a, b, q, D, modes, nx, ny):
    """First-order model of the rounding of ``levy_x_partial_at_nodes`` (absolute, per node); not a strict bound.

    Only for alpha_m = m pi b/(2a) >= pi/2 for every mode (see the module docstring); ValueError otherwise.
    """
    if min(modes) * b / a < 1.0:
        raise ValueError("rounding model not applicable: alpha_m < pi/2 (stable form loses accuracy, use the Navier sum)")
    C0 = 4.0 * abs(q) * a**4 / (math.pi**5 * D)
    s = _s_nodes(ny)
    N = len(modes)
    gamma = N * U_ROUND / (1 - N * U_ROUND)
    S = np.zeros((nx + 1, ny + 1))
    for m in modes:
        alpha = m * math.pi * b / (2.0 * a)
        S += np.outer(np.abs(_sin_nodes(nx, m)), C0 * np.abs(bracket(alpha, s)) / m**5)
    return (12.0 * U_ROUND + gamma) * S


def navier_m_tail_bound(a, b, q, D, modes, M):
    """Strict uniform bound of the n-tail of the m-truncated Navier sum (odd n > M)."""
    return 16.0 * abs(q) * b**4 / (math.pi**6 * D) * sum(1.0 / m for m in modes) / (8.0 * M**4)


def navier_m_truncated_at_nodes(a, b, q, D, modes, nx, ny, M):
    """sum_{m in modes} sum_{n odd <= M} w_mn sin(m pi x_i/a) sin(n pi y_j/b) at all nodes; separable evaluation."""
    n = np.arange(1, M + 1, 2)
    j = np.arange(ny + 1)
    r = np.mod(np.outer(j, n), 2 * ny)
    sign = np.where(r < ny, 1.0, -1.0)
    r1 = np.mod(r, ny)
    Sy = sign * np.sin(np.pi * np.minimum(r1, ny - r1) / ny)
    W = np.zeros((nx + 1, ny + 1))
    for m in modes:
        c = 16.0 * q / (math.pi**6 * D * m * n * (m**2 / a**2 + n**2 / b**2) ** 2)
        W += np.outer(_sin_nodes(nx, m), Sy @ c)
    return W
