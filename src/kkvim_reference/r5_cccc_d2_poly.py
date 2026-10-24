"""D2-9 (plan PL 1, table D2): the polynomial field w = x^2 (a - x)^2 y^2 (b - y)^2 and an independent closed form of the discrete
13-point operator on it (R5-V CCCC; audit ce59753, item 1b). Research QA tool; no import of ``kkvim``.

w = X(x) Y(y), X(t) = t^2 (a - t)^2 = a^2 t^2 - 2 a t^3 + t^4, Y likewise with b; w = dw/dn = 0 on the boundary (CCCC). On the grid
x_i = i h (h = a/n EXACT rational of the input doubles) the operator of D2 (NS 3.3, Proposition D) is
    (A w)_ij = (delta4 X)_i Y_j + 2 (delta2 X)_i (delta2 Y)_j + X_i (delta4 Y)_j,
with delta2 the second difference (boundary values X_0 = X_n = 0 are the polynomial values) and delta4 the fourth difference with the
even ghost node X_{-1} = X_1, X_{n+1} = X_{n-1}. Closed forms (own derivation, level B):
- delta2 X_i = X''(x_i) + (h^2/12) X'''' = 12 x_i^2 - 12 a x_i + 2 a^2 + 2 h^2 for 1 <= i <= n - 1 (the central second difference is
  exact up to the fourth derivative for a quartic);
- delta4 X_i = X'''' = 24 for 2 <= i <= n - 2 (the five-point fourth difference is exact for a quartic);
- delta4 X_1 = (7 X_1 - 4 X_2 + X_3)/h^4 = 24 - 4 a/h, and by symmetry delta4 X_{n-1} = 24 - 4 a/h: with X(t) = a^2 t^2 - 2 a t^3 + t^4,
  7 X(h) - 4 X(2h) + X(3h) = (7 - 16 + 9) a^2 h^2 + (-14 + 64 - 54) a h^3 + (7 - 64 + 81) h^4 = -4 a h^3 + 24 h^4 (n >= 3).
The closed forms use no function of r5_cccc_d2 (not a tautological application of D2 to itself); the comparison with the
node-by-node D2 operator is done by the tests. ``solve_with_polynomial_load`` (second part of D2-9) uses the D2 matrix as the
object under test and refines with the exact residual of the exact right-hand side.
"""

from fractions import Fraction


def nodal_values(a: float, n: int) -> list[Fraction]:
    """X(x_i), i = 1..n-1, exactly (h = a/n)."""
    A = Fraction(a)
    h = A / n
    return [(i * h) ** 2 * (A - i * h) ** 2 for i in range(1, n)]


def delta2_closed(a: float, n: int) -> list[Fraction]:
    A = Fraction(a)
    h = A / n
    return [12 * (i * h) ** 2 - 12 * A * (i * h) + 2 * A**2 + 2 * h**2 for i in range(1, n)]


def delta4_closed(a: float, n: int) -> list[Fraction]:
    A = Fraction(a)
    h = A / n
    edge = 24 - 4 * A / h
    return [edge if i in (1, n - 1) else Fraction(24) for i in range(1, n)]


def operator_closed(lx: float, nx: int, ly: float, ny: int) -> list[Fraction]:
    """(A w)_k in the D2 ordering k = (i - 1)(n_y - 1) + (j - 1), exactly, from the closed forms."""
    X, Y = nodal_values(lx, nx), nodal_values(ly, ny)
    d2X, d2Y = delta2_closed(lx, nx), delta2_closed(ly, ny)
    d4X, d4Y = delta4_closed(lx, nx), delta4_closed(ly, ny)
    return [d4X[i] * Y[j] + 2 * d2X[i] * d2Y[j] + X[i] * d4Y[j] for i in range(nx - 1) for j in range(ny - 1)]


def field_values(lx: float, nx: int, ly: float, ny: int) -> list[Fraction]:
    """w_k = X_i Y_j in the D2 ordering, exactly."""
    X, Y = nodal_values(lx, nx), nodal_values(ly, ny)
    return [X[i] * Y[j] for i in range(nx - 1) for j in range(ny - 1)]


def solve_with_polynomial_load(lx: float, nx: int, ly: float, ny: int, steps: int = 4) -> dict:
    """Second part of D2-9: the D2 system A x = f with f = A w (exact, closed form) solved by the D2 matrix (sparse LU of
    r5_cccc_d2.assemble) and refined with the EXACT residual f - A x (node-by-node integer parts of D2, exact scales). The exact
    solution is x* = w, so the error is known exactly. Values are OBSERVATION (plan 1.4: no accepted threshold)."""
    import numpy as np
    import scipy.sparse.linalg as spla

    from kkvim_reference import r5_cccc_d2 as d2

    prob = d2.D2Problem(lx, ly, nx, ny)
    parts, scales = d2.nodewise_parts(nx, ny), prob.scales()
    f = operator_closed(lx, nx, ly, ny)
    w = field_values(lx, nx, ly, ny)
    lu = spla.splu(d2.assemble(prob))

    def residual(x):
        xs = [Fraction(float(v)) for v in x]
        out = list(f)
        for z, s in zip(parts, scales):
            z = z.tocsr()
            for r in range(len(xs)):
                out[r] -= s * sum((int(z.data[k]) * xs[z.indices[k]] for k in range(z.indptr[r], z.indptr[r + 1])), Fraction(0))
        return out

    def err(x):
        return max(abs(Fraction(float(v)) - e) for v, e in zip(x, w)) / max(abs(e) for e in w)

    x = lu.solve(np.array([float(v) for v in f]))
    errors = [err(x)]
    for _ in range(steps):
        x = x + lu.solve(np.array([float(v) for v in residual(x)]))
        errors.append(err(x))
    return {"relative_max_error": [float(e) for e in errors], "x": x}
