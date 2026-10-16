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

Navier-m is checked for an odd M only (the bound 1/(8 M^4) needs n >= M + 2 for the omitted odd n; ValueError otherwise).

Rounding of the m-truncated Navier sum (``navier_m_rounding_bound``): STRICT in the standard IEEE model (round to nearest,
no overflow or underflow) under the ASSUMPTIONS that math.pow / ** of a double is within one ulp (<= 2u relative) and that
np.sin of an argument in [0, pi/2] is within one ulp (checked for the used grids by tests/cross/test_r4_dst_qa.py), with
classical dot products in any order. Factors (1 + delta) are composed exactly, not to first order:
    coefficient c_mn = 16 q / (pi^6 D m n (m^2/a^2 + n^2/b^2)^2): fl(pi)^6 (6 x 0.35u), pow (2u), x D, x m, x n, x square
        (1u each), m^2/a^2 and n^2/b^2 (a^2 or b^2 and the division: 2u each), their sum (u), the square (2 x 3u + u),
        final division (u);
    sine sin(pi r / n), r reduced to [0, n/2]: argument fl(pi) (0.35u), x r (u), / n (u); propagation |dx| <= eta x and
        x / sin(x) <= pi/2 on (0, pi/2] (with the second-order term), own error 2u;
    dot product of K = (M + 1)/2 terms: gamma_K; outer product with sin(m pi x/a): u and the sine; sum over N modes: gamma_N.
Hence |w_computed - w_M| <= theta * sum_m sum_n |c_mn| at every node, theta from the composition (about gamma_K + 26u).

Rounding of the Levy partial sum (``levy_x_partial_rounding_model``): MODEL_ONLY. The relative error eps_T of the stable
bracket T has NO a-priori bound here: R3 recorded <= 3.92u, the Reviewer measured 8.9u at alpha = pi/2 (3aaca8e, 2.1) and
QA 7.4u, and the sensitivity of T to the rounding of alpha and s is not bounded. eps_T must therefore be supplied by the
caller (e.g. measured in a temporary high-precision environment for the alpha and nodes used); the remaining operations
add 16.1u by count (C0 = 4 q a^4 / (pi^5 D): 8.75u; x C0 and / m^5: 2u; x sin(m pi x/a) and the outer product: 5.35u),
plus gamma_N for the sum of N modes. DOMAIN: alpha_m >= pi/2 only. For alpha < pi/2 the bracket T = O(alpha^4) is a
small difference of terms of order one (QA, 50 digits: about 200u at alpha = 0.31, r = 0.2; Reviewer 160u) and the
m-truncated Navier sum is the s = 1 reference (R4_NUMERICAL_SPEC 9.1; Reviewer note С-2).
"""

import math

import numpy as np

__all__ = ["bracket", "levy_x_partial_at_nodes", "levy_x_partial_rounding_model", "navier_m_truncated_at_nodes",
           "navier_m_tail_bound", "navier_m_rounding_bound"]

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


def levy_x_partial_rounding_model(a, b, q, D, modes, nx, ny, bracket_rel):
    """MODEL_ONLY rounding estimate of ``levy_x_partial_at_nodes`` (absolute, per node); not a bound.

    bracket_rel -- relative error of the bracket T supplied by the caller (no a-priori bound exists, module docstring);
    per term (bracket_rel + 16.1u), plus gamma_N for the sum of the N modes. Only for alpha_m >= pi/2 (ValueError).
    """
    if min(modes) * b / a < 1.0:
        raise ValueError("rounding model not applicable: alpha_m < pi/2 (stable form loses accuracy, use the Navier sum)")
    if not (math.isfinite(bracket_rel) and bracket_rel >= 0.0):
        raise ValueError("bracket_rel must be a finite non-negative number")
    C0 = 4.0 * abs(q) * a**4 / (math.pi**5 * D)
    s = _s_nodes(ny)
    N = len(modes)
    gamma = N * U_ROUND / (1 - N * U_ROUND)
    S = np.zeros((nx + 1, ny + 1))
    for m in modes:
        alpha = m * math.pi * b / (2.0 * a)
        S += np.outer(np.abs(_sin_nodes(nx, m)), C0 * np.abs(bracket(alpha, s)) / m**5)
    return (bracket_rel + 16.1 * U_ROUND + gamma) * S


def _check_odd_M(M):
    if isinstance(M, bool) or not isinstance(M, (int, np.integer)) or M < 1 or M % 2 == 0:
        raise ValueError(f"M must be a positive odd integer (largest kept odd n), got {M!r}")
    return int(M)


def navier_m_tail_bound(a, b, q, D, modes, M):
    """Strict uniform bound of the n-tail of the m-truncated Navier sum (omitted odd n > M; M odd)."""
    M = _check_odd_M(M)
    return 16.0 * abs(q) * b**4 / (math.pi**6 * D) * sum(1.0 / m for m in modes) / (8.0 * M**4)


def navier_m_truncated_at_nodes(a, b, q, D, modes, nx, ny, M):
    """sum_{m in modes} sum_{n odd <= M} w_mn sin(m pi x_i/a) sin(n pi y_j/b) at all nodes; separable evaluation."""
    M = _check_odd_M(M)
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


def navier_m_rounding_bound(a, b, q, D, modes, M):
    """Uniform bound of |w_computed - w_M| for ``navier_m_truncated_at_nodes`` (strict under the stated assumptions).

    Composition of the relative errors listed in the module docstring; returns theta * sum_m sum_n |c_mn| (absolute).
    """
    M = _check_odd_M(M)
    u = U_ROUND
    K = (M + 1) // 2
    N = len(modes)
    gamma_K = K * u / (1 - K * u)
    gamma_N = N * u / (1 - N * u)
    # coefficient: denominator pi^6 D m n (sq)^2 and the final division
    den = (1 + 0.35 * u) ** 6 * (1 + 2 * u) * (1 + u) ** 3 * (1 + u) ** 7 * (1 + u) - 1  # pi^6, pow, xD xm xn, square, x sq
    c_rel = (1 + den / (1 - den)) * (1 + u) - 1
    # sine: argument (fl(pi), x r, / n), propagation with x/sin(x) <= pi/2 and the second-order term, own error 2u
    eta = (1 + 0.35 * u) * (1 + u) ** 2 - 1
    s_rel = (1 + eta * math.pi / 2 + eta**2) * (1 + 2 * u) - 1
    theta = (1 + gamma_K) * (1 + c_rel) * (1 + s_rel) * (1 + u) * (1 + s_rel) * (1 + gamma_N) - 1
    n = np.arange(1, M + 1, 2)
    total = math.fsum(math.fsum((16.0 * abs(q) / (math.pi**6 * D * m * n * (m**2 / a**2 + n**2 / b**2) ** 2)).tolist())
                      for m in modes)
    return theta * total * (1 + 8 * u)  # the bound itself computed with a few roundings
