"""Analytic deflection of the first KK-VIM half-step by Levy's formula (R2).

Independent analytic reference for the first half-step s = 1 (D-030) of the
KK-VIM sequence for a square simply supported (SSSS) Kirchhoff-Love plate
under uniform load q, N = 1 (D-015, D-031, D-033).

Source: docs/MATH_SPEC.md, section 9.1, project derivation В-29, which
coincides with the term m = 1 of Levy's solution
[L-107, eq. (139), p. 136]
references/Plate_Benchmarks_Linear_BC/1966_Timoshenko_Woinowsky-Krieger_Plates_and_Shells_Nauka.djvu
(comparison accepted in MATH_SPEC.md, section 9.1, D-031); centre value -
MATH_SPEC.md, section 9.2, В-30; use as a control - docs/NUMERICAL_SPEC.md,
section 14.

Coordinates of the project (MATH_SPEC.md, section 1.1): 0 <= x <= a,
0 <= y <= b, origin at a corner. R2 is the square a = b, so alpha = pi/2.

    X0(x) = sin(pi x / a),
    Y1(y) = C0 [1 - (alpha th alpha + 2) / (2 ch alpha) ch(pi y_c / a)
                 + 1 / (2 ch alpha) (pi y_c / a) sh(pi y_c / a)],
    C0 = 4 q a^4 / (pi^5 D),  y_c = y - a/2,
    w1(x, y) = X0(x) Y1(y).

This is the first half-step only, not the Navier solution of the plate
(MATH_SPEC.md, section 9.3). Values are computed pointwise from closed-form
functions; no grid, no matrices, no KK-VIM iterations. The module must not
import ``kkvim`` (D-028).

R3 extension (D-036, Р-6; docs/R3_LEVY_REFERENCE_DRAFT.md): the full Levy
series of L-107, section 30, for the rectangle a x b, 0.2 <= b/a <= 5, SSSS,
uniform load q - the formulas (h), (i), (139), (j), (140), (141) [L-107,
pp. 135-137], verified in docs/R3_LEVY_SOURCE_TRANSCRIPTION.review.md
(VERIFIED_FOR_DEFLECTION). Moments are not implemented.

- Orientation: the series is summed in the frame of L-107 (fig. 62, footnote 2
  on p. 136): a_L = min(a, b) along x_L, b_L = max(a, b), r_L = b_L / a_L >= 1,
  eta_L = y_L - b_L / 2. If b < a the axes are exchanged, (x_L, y_L) = (y, x).
  Results are returned in the original coordinates (x, y) of the plate a x b;
  dimensionless values keep the reference length a of the original plate,
  w_bar = w D / (q a^4) (R3_NUMERICAL_SPEC_DRAFT.md, Р-3).
- Field: (139) term by term in a stable form of the bracket without cosh and
  sinh of large arguments (project transformation, R3_LEVY_REFERENCE_DRAFT.md,
  section 3). Centre: (140) (exponentially convergent correction series) and
  (j) (algebraic tail) - two forms of one quantity for a consistency check.
- Truncation: the bounds of R3_LEVY_REFERENCE_DRAFT.md, section 5 - a uniform
  bound O(M^-4) of the field and an exponential bound of the series (140);
  the exponential bound of the centre does not hold for the field.
- Rounding: a documented model, not a strict bound (section 6 there).

The comparison with the Navier series (P-12) is not part of this module.
"""

import math
from typing import NamedTuple

import numpy as np

from kkvim_reference._r3_range import check_r3_ratio

__all__ = [
    "levy_x0",
    "levy_y1",
    "levy_w1",
    "LEVY_RATIO_MIN",
    "LEVY_RATIO_MAX",
    "LEVY_FIELD_TOL",
    "LEVY_CENTRE_TOL",
    "LevyOrientation",
    "LevyBounds",
    "levy_orientation",
    "levy_field_tail_bound",
    "levy_centre_tail_bound",
    "levy_select_m_field",
    "levy_select_m_centre",
    "levy_series_w",
    "levy_series_w_grid",
    "levy_field_bounds",
    "levy_centre_140",
    "levy_centre_j",
    "levy_centre_bounds",
    "levy_alpha_141",
    "levy_centre_wbar",
]

# alpha = pi b / (2 a) with b = a (square, R2).
_ALPHA = math.pi / 2


def _check_positive(name, value):
    if not (math.isfinite(value) and value > 0):
        raise ValueError(f"{name} must be a finite positive number, got {value!r}")


def _check_coordinate(name, values, a):
    values = np.asarray(values, dtype=float)
    if not np.all((values >= 0.0) & (values <= a)):
        raise ValueError(f"{name} must lie in the closed interval [0, a] = [0, {a!r}]")
    return values


def levy_x0(x, *, a):
    """Initial function X0(x) = sin(pi x / a) (MATH_SPEC.md, section 4.6)."""
    _check_positive("a", a)
    x = _check_coordinate("x", x, a)
    return np.sin(math.pi * x / a)


def levy_y1(y, *, a, q, D):
    """Function Y1(y) of the first half-step by V-29 (MATH_SPEC.md, section 9.1).

    a -- side of the square plate, a = b > 0;
    q -- uniform load (constant);
    D -- flexural rigidity, D > 0.
    """
    _check_positive("a", a)
    _check_positive("D", D)
    if not math.isfinite(q):
        raise ValueError(f"q must be a finite number, got {q!r}")
    y = _check_coordinate("y", y, a)

    c0 = 4.0 * q * a**4 / (math.pi**5 * D)
    ch_alpha = math.cosh(_ALPHA)
    p = (_ALPHA * math.tanh(_ALPHA) + 2.0) / (2.0 * ch_alpha)
    r = 1.0 / (2.0 * ch_alpha)
    t = math.pi * (y - a / 2.0) / a  # pi y_c / a
    return c0 * (1.0 - p * np.cosh(t) + r * t * np.sinh(t))


def levy_w1(x, y, *, a, q, D):
    """Deflection after the first half-step w1(x, y) = X0(x) Y1(y).

    x, y -- coordinates in [0, a] (scalars or arrays, broadcast together);
    a, q, D -- as in ``levy_y1``. Units must be consistent (D-009).
    """
    return levy_x0(x, a=a) * levy_y1(y, a=a, q=q, D=D)


# ---------------------------------------------------------------------------
# R3: full Levy series of L-107, section 30 (rectangle a x b, 0.2 <= b/a <= 5).
# Formulas, transformations and bounds: docs/R3_LEVY_REFERENCE_DRAFT.md.
# ---------------------------------------------------------------------------

# Scientific range of b/a is D-036; all R3 references now use the
# exact symmetric 4u representation gate of D-038.
LEVY_RATIO_MIN = 0.2
LEVY_RATIO_MAX = 5.0

# Unit roundoff u = 2^-53 and the factor 4 / pi^5 of (139), (j), (140).
_U = 2.0**-53
_C5 = 4.0 / math.pi**5

# Default targets of the strict truncation bounds, in q a_L^4 / D [ИНЖЕНЕРНЫЙ
# ВЫБОР, fixed before any comparison; R3_LEVY_REFERENCE_DRAFT.md, 5.5]:
# field (139) - 1e-15, the order of the rounding model of its sum (M = 1131);
# centre (140) - 1e-19, below 1/8 of the unit in the last place of 5/384.
LEVY_FIELD_TOL = 1e-15
LEVY_CENTRE_TOL = 1e-19

# Rounding model of one term of (139) relative to its size, in units of u
# (R3_LEVY_REFERENCE_DRAFT.md, 6.1): stable bracket 4 (largest observed error
# 3.92 u against 80-digit decimal arithmetic); alpha_m = m (pi r_L / 2), three
# roundings with |d ln T / d ln alpha| <= 2.1 (observed): 7; s = 2 d / b_L,
# one rounding with |d ln T / d ln s| <= 1: 1; m^-5: 1; two products: 2;
# library sin: 1. The argument m theta (theta = pi x~ / a_L, relative error
# <= 4 u) adds 4 m theta u <= 2 pi m u absolutely. Summation of K terms in any
# order: (K - 1) u sum |terms| (first order). Scale 4 q a_L^4 / (pi^5 D): 6.
# Margin 2. This is a model, not a strict bound.
LEVY_TERM_ROUNDINGS = 16
LEVY_SCALE_ROUNDINGS = 6
LEVY_ROUNDING_MARGIN = 2.0
# Centre, per term m^-5 P_m relative: own evaluation of P_m about 10, alpha_m
# (3 roundings) times |d ln P / d ln alpha| <= alpha_m + 1, product and sign 2;
# fsum, 5/384, the factor 4/pi^5 and the difference: 2 (5/384) u.
_CENTRE_TERM_ROUNDINGS = 15
# Elements of the work arrays (terms x points) evaluated together.
_CHUNK_ELEMENTS = 1 << 21


class LevyOrientation(NamedTuple):
    """Frame of L-107, section 30, for the plate a x b (fig. 62; p. 136, footnote 2)."""

    a_L: float  # shorter side, along x_L (a of L-107)
    b_L: float  # longer side, along y_L (b of L-107), edges eta_L = +-b_L / 2
    r_L: float  # b_L / a_L, 1 <= r_L <= 5
    swapped: bool  # True if b < a: (x_L, y_L) = (y, x)


class LevyBounds(NamedTuple):
    """Error estimates of a truncated Levy series, in the units of the returned value."""

    M: int  # largest odd index of the truncated series
    truncation: float  # strict bound of the omitted tail (R3_LEVY_REFERENCE_DRAFT.md, 5)
    rounding_model: float  # rounding model, not a strict bound (section 6)


def _check_finite(name, value):
    if isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {value!r}")


def _check_odd(M):
    if isinstance(M, bool) or not isinstance(M, (int, np.integer)) or M < 1 or M % 2 == 0:
        raise ValueError(f"M must be a positive odd integer, got {M!r}")
    return int(M)


def _check_ratio_l(r_L):
    """Public r_L: 1 <= r_L <= 5(1+4u) in the short-side Levy frame."""
    _check_positive("r_L", r_L)
    if r_L < 1.0:
        raise ValueError("ERROR_INPUT: r_L must be >= 1 (shorter side is a_L)")
    check_r3_ratio(1.0, r_L)
    return float(r_L)


def _check_interval(name, values, length, side):
    values = np.asarray(values, dtype=float)
    if not np.all((values >= 0.0) & (values <= length)):
        raise ValueError(f"{name} must lie in the closed interval [0, {side}] = [0, {length!r}]")
    return values


def levy_orientation(a, b):
    """Frame of the Levy series for the plate a x b: a_L = min(a, b), b_L = max(a, b).

    The physical range is 0.2 <= b/a <= 5 (D-036, Р-1). Input floats use the
    shared rational D-038 gate max(a, b) <= 5(1+4u)min(a, b);
    outside it ``ValueError: ERROR_INPUT``. The gate is symmetric under a <-> b.
    """
    _check_positive("a", a)
    _check_positive("b", b)
    a, b = float(a), float(b)
    swapped = b < a
    a_L, b_L = (b, a) if swapped else (a, b)
    check_r3_ratio(a, b)
    return LevyOrientation(a_L, b_L, b_L / a_L, swapped)


def _odd(M):
    return np.arange(1, M + 1, 2)


def _inv_m5(M):
    """m^-5 for odd m <= M, each correctly rounded (int / int is correctly rounded)."""
    return np.array([1 / k**5 for k in range(1, M + 1, 2)])


def _alphas(M, r_L):
    """alpha_m = m pi b_L / (2 a_L) = m (pi r_L / 2) by (h) [L-107, p. 135]."""
    return _odd(M) * (math.pi * r_L / 2.0)


def levy_field_tail_bound(M):
    """Strict bound of |w - w_M| for the field (139), uniform over the plate, in q a_L^4 / D.

    (4 / pi^5) sum_{m odd >= M+2} m^-5 <= (4 / pi^5) / (8 M^4), because the
    bracket of (139) lies in [0, 1] on the whole closed plate (В-Л2) and
    |sin| <= 1 (R3_LEVY_REFERENCE_DRAFT.md, 5.2). Also bounds the tail of (j).
    Independent of r_L. In units of the original q a^4 / D multiply by
    (a_L / a)^4.
    """
    M = _check_odd(M)
    return _C5 / (8.0 * float(M) ** 4)


def _centre_tail_unchecked(M, r_L):
    """Internal tail formula after validation of the original plate or r_L."""
    m = M + 2
    alpha = m * (math.pi * r_L / 2.0)
    tau = (alpha + 2.0) * math.exp(-alpha) / float(m) ** 5
    return _C5 * tau / -math.expm1(-math.pi * r_L)


def levy_centre_tail_bound(M, r_L):
    """Strict correction-tail bound of (140), in |q| a_L^4 / D (D-038).

    To convert to the original |q| a^4 / D, multiply by (a_L/a)^4.
    This bound is for the centre correction series, not the full field.
    """
    return _centre_tail_unchecked(_check_odd(M), _check_ratio_l(r_L))


def levy_select_m_field(tol=LEVY_FIELD_TOL):
    """Smallest odd M with levy_field_tail_bound(M) <= tol (tol in q a_L^4 / D)."""
    _check_positive("tol", tol)
    M = max(1, math.ceil((_C5 / (8.0 * tol)) ** 0.25))
    M += 1 - M % 2
    while M > 1 and levy_field_tail_bound(M - 2) <= tol:
        M -= 2
    while levy_field_tail_bound(M) > tol:
        M += 2
    return M


def _select_m_centre_unchecked(r_L, tol):
    M = 1
    while _centre_tail_unchecked(M, r_L) > tol:
        M += 2
    return M


def levy_select_m_centre(r_L, tol=LEVY_CENTRE_TOL):
    """Smallest odd M meeting the centre correction-tail target."""
    _check_positive("tol", tol)
    return _select_m_centre_unchecked(_check_ratio_l(r_L), tol)


def _bracket(alpha, s):
    """Bracket T of (139) in the stable form (R3_LEVY_REFERENCE_DRAFT.md, 3.2).

    T = 1 - (alpha th alpha + 2) / (2 ch alpha) ch(alpha t)
          + alpha / (2 ch alpha) t sh(alpha t),   |t| = 1 - s,
    s = 1 - |2 eta_L / b_L| in [0, 1] (s = 0 at the edges, s = 1 on the centre
    line). With z = alpha s, g = exp(-2 alpha), E = exp(-alpha (2 - s)):

    (1 + g) T = [-expm1(-z) - (z / 2) exp(-z)]
                + E [expm1(-z) + alpha expm1(-2 z) / (1 + g) + z / 2].

    All exponents are <= 0 (no overflow); T = 0 exactly at s = 0.
    """
    z = alpha * s
    # exp(-large) correctly underflows to zero; external NumPy seterr(all="raise")
    # must not turn this harmless limiting behaviour into an exception.
    with np.errstate(under="ignore"):
        g = np.exp(-2.0 * alpha)
        em = np.expm1(-z)
        f = -em - 0.5 * z * np.exp(-z)
        e = np.exp(-alpha * (2.0 - s))
        return (f + e * (em + alpha * np.expm1(-2.0 * z) / (1.0 + g) + 0.5 * z)) / (1.0 + g)


def _p_coef(alpha):
    """(alpha th alpha + 2) / (2 ch alpha) of (i), (j), (140) without ch alpha (3.3)."""
    g = math.exp(-2.0 * alpha)
    th = -math.expm1(-2.0 * alpha) / (1.0 + g)
    return (alpha * th + 2.0) * math.exp(-alpha) / (1.0 + g)


def _levy_coords(xs, ys, orient):
    """Reduced coordinates in the frame of L-107: theta = pi x~ / a_L and s.

    x~ = min(x_L, a_L - x_L): sin(m pi x_L / a_L) = sin(m pi x~ / a_L) for odd m.
    s = 2 min(y_L, b_L - y_L) / b_L = 1 - |2 eta_L / b_L|, formed from the
    distance to the nearest edge without the difference 1 - |t| (3.4).
    """
    xl, yl = (ys, xs) if orient.swapped else (xs, ys)
    theta = math.pi * np.minimum(xl, orient.a_L - xl) / orient.a_L
    s = 2.0 * np.minimum(yl, orient.b_L - yl) / orient.b_L
    return theta, s


def _check_load(q, D):
    _check_finite("q", q)
    _check_positive("D", D)


def _resolve_m(M):
    return levy_select_m_field() if M is None else _check_odd(M)


def levy_series_w(x, y, *, a, b, q, D, M=None):
    """Deflection w(x, y) by the Levy series (139) [L-107, p. 136], truncated at m <= M.

    x in [0, a], y in [0, b] - original coordinates of the plate a x b (corner
    origin, MATH_SPEC.md, 1.1), scalars or arrays broadcast together; a, b,
    q, D - consistent units (D-009); 0.2 <= b/a <= 5. M - largest odd index;
    None selects levy_select_m_field() (strict bound <= LEVY_FIELD_TOL
    q a_L^4 / D). M = 1 is the first half-step of KK-VIM started along the
    shorter side a_L (В-29); with b < a this is the start along y.
    """
    orient = levy_orientation(a, b)
    _check_load(q, D)
    M = _resolve_m(M)
    x = _check_interval("x", x, float(a), "a")
    y = _check_interval("y", y, float(b), "b")
    x, y = np.broadcast_arrays(x, y)
    shape = x.shape
    theta, s = _levy_coords(x.ravel(), y.ravel(), orient)

    m = _odd(M)
    c = _inv_m5(M)[:, None]
    alpha = _alphas(M, orient.r_L)[:, None]
    out = np.empty(theta.size)
    chunk = max(1, _CHUNK_ELEMENTS // m.size)
    for i in range(0, theta.size, chunk):
        th = theta[None, i : i + chunk]
        terms = c * _bracket(alpha, s[None, i : i + chunk]) * np.sin(m[:, None] * th)
        out[i : i + chunk] = np.sum(terms, axis=0)
    return _C5 * (q * orient.a_L**4 / D) * out.reshape(shape)


def levy_series_w_grid(xs, ys, *, a, b, q, D, M=None):
    """Deflection by (139) at all nodes of the tensor grid xs x ys, shape (len(xs), len(ys)).

    xs in [0, a], ys in [0, b] (one-dimensional). The series is separable,
    w = S diag(m^-5) T, and is summed as one matrix product in the frame of
    L-107; the result is returned in the original orientation [i, j] = (xs[i], ys[j]).
    """
    orient = levy_orientation(a, b)
    _check_load(q, D)
    M = _resolve_m(M)
    xs = _check_interval("xs", np.ravel(xs), float(a), "a")
    ys = _check_interval("ys", np.ravel(ys), float(b), "b")
    xl, yl = (ys, xs) if orient.swapped else (xs, ys)
    theta = math.pi * np.minimum(xl, orient.a_L - xl) / orient.a_L
    s = 2.0 * np.minimum(yl, orient.b_L - yl) / orient.b_L

    m = _odd(M)
    sines = np.sin(theta[:, None] * m[None, :])  # (len xl, K)
    brackets = _inv_m5(M)[:, None] * _bracket(_alphas(M, orient.r_L)[:, None], s[None, :])  # (K, len yl)
    w = sines @ brackets
    if orient.swapped:
        w = w.T
    return _C5 * (q * orient.a_L**4 / D) * w


def levy_field_bounds(*, a, b, q, D, M=None):
    """Truncation bound and rounding model of levy_series_w(_grid), in the units of w.

    truncation - strict, uniform over the plate (levy_field_tail_bound times
    |q| a_L^4 / D); rounding_model - model of section 6.1, not a strict bound.
    """
    orient = levy_orientation(a, b)
    _check_load(q, D)
    M = _resolve_m(M)
    scale = abs(q) * orient.a_L**4 / D
    c = _inv_m5(M)
    k = c.size
    lam5 = math.fsum(c)
    lam4 = math.fsum(c * _odd(M))
    model = LEVY_ROUNDING_MARGIN * _U * _C5 * (
        (LEVY_TERM_ROUNDINGS + LEVY_SCALE_ROUNDINGS + k - 1) * lam5 + 2.0 * math.pi * lam4
    )
    return LevyBounds(M, levy_field_tail_bound(M) * scale, model * scale)


def _centre_140(r_L, M):
    """Centre deflection by (140) [L-107, p. 136] in q a_L^4 / D (alpha of (141))."""
    c = math.pi * r_L / 2.0
    terms = [(-1) ** ((m - 1) // 2) * _p_coef(m * c) / m**5 for m in range(1, M + 1, 2)]
    return 5.0 / 384.0 - _C5 * math.fsum(terms)


def _centre_j(r_L, M):
    """Centre deflection by (j) [L-107, p. 136] in q a_L^4 / D."""
    c = math.pi * r_L / 2.0
    terms = [(-1) ** ((m - 1) // 2) * (1.0 - _p_coef(m * c)) / m**5 for m in range(1, M + 1, 2)]
    return _C5 * math.fsum(terms)


def _centre_rounding_model(r_L, M, form):
    """Rounding model of _centre_140 (form "140") or _centre_j ("j") in q a_L^4 / D (6.2)."""
    total = 0.0
    for m in range(1, M + 1, 2):
        alpha = m * (math.pi * r_L / 2.0)
        p = _p_coef(alpha)
        total += p * (_CENTRE_TERM_ROUNDINGS + 3.0 * alpha) / m**5
        if form == "j":
            total += 2.0 * (1.0 - p) / m**5
    lead = 2.0 * 5.0 / 384.0 if form == "140" else 4.0 * _C5
    return LEVY_ROUNDING_MARGIN * _U * (lead + _C5 * total)


def levy_centre_140(*, a, b, q, D, M=None):
    """Deflection at the centre (a/2, b/2) by (140); M = None - levy_select_m_centre(r_L)."""
    orient = levy_orientation(a, b)
    _check_load(q, D)
    M = _select_m_centre_unchecked(orient.r_L, LEVY_CENTRE_TOL) if M is None else _check_odd(M)
    return _centre_140(orient.r_L, M) * (q * orient.a_L**4 / D)


def levy_centre_j(*, a, b, q, D, M=None):
    """Deflection at the centre by (j); algebraic tail, M = None - levy_select_m_field()."""
    orient = levy_orientation(a, b)
    _check_load(q, D)
    return _centre_j(orient.r_L, _resolve_m(M)) * (q * orient.a_L**4 / D)


def levy_centre_bounds(*, a, b, q, D, M=None, form="140"):
    """Truncation bound and rounding model of levy_centre_140 (form "140") or levy_centre_j ("j")."""
    if form not in ("140", "j"):
        raise ValueError(f"form must be '140' or 'j', got {form!r}")
    orient = levy_orientation(a, b)
    _check_load(q, D)
    if form == "140":
        M = _select_m_centre_unchecked(orient.r_L, LEVY_CENTRE_TOL) if M is None else _check_odd(M)
        trunc = _centre_tail_unchecked(M, orient.r_L)
    else:
        M = _resolve_m(M)
        trunc = levy_field_tail_bound(M)
    scale = abs(q) * orient.a_L**4 / D
    return LevyBounds(M, trunc * scale, _centre_rounding_model(orient.r_L, M, form) * scale)


def levy_alpha_141(r_L, M=None):
    """Coefficient alpha of (141) [L-107, p. 137]: w_max = alpha q a_L^4 / D, a_L - shorter side.

    1 <= r_L = b_L / a_L <= 5 (convention of L-107, p. 136, footnote 2). By
    (140); M = None - levy_select_m_centre(r_L).
    """
    r_L = _check_ratio_l(r_L)
    M = levy_select_m_centre(r_L) if M is None else _check_odd(M)
    return _centre_140(r_L, M)


def levy_centre_wbar(r, M=None):
    """Dimensionless centre deflection w D / (q a^4) of the plate a x b, r = b / a in [0.2, 5].

    Reference length - the original side a (Р-3). For r >= 1 it equals
    levy_alpha_141(r); for r < 1 the series is summed for r_L = 1 / r and
    multiplied by (a_L / a)^4 = r^4 (ЧВ-R1).
    """
    orient = levy_orientation(1.0, r)
    M = _select_m_centre_unchecked(orient.r_L, LEVY_CENTRE_TOL) if M is None else _check_odd(M)
    return _centre_140(orient.r_L, M) * orient.a_L**4
