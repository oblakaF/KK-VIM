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
"""

import math

import numpy as np

__all__ = ["levy_x0", "levy_y1", "levy_w1"]

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
