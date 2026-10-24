"""QA energy instrument for R5-V CCCC (audit ce59753, G-2 and G-4): exact Pi_h^C, ||.||_a, cross terms, the double assembly path of
VOK-3 and the model terms R^C, R_Q^C, R_f^C of VOK-8. Research QA tool (D-049.6, item 5; owner task); no import of ``kkvim``.

Definitions (NS 2.2; roundoff document RO 0.2, VOK-3, VOK-5, VOK-7, VOK-8):
- grid G(a, n_x) x G(b, n_y), h_x = a/n_x, h_y = b/n_y EXACT rationals of the input doubles; fields w on the interior nodes,
  rows = x index (array (n_x - 1) x (n_y - 1)), w = 0 on the boundary;
- L^ f: (f_{i-1} - 2 f_i + f_{i+1})/h^2 at i = 1..n-1, (L^ f)_0 = 2 f_1/h^2, (L^ f)_n = 2 f_{n-1}/h^2 (even ghost node); E embeds the
  interior values; Delta_h^C = L^_x (x) E_y + E_x (x) L^_y on all nodes 0..n_x x 0..n_y; W = diag(h/2, h, ..., h, h/2);
- a_h(v, w) = D (Delta_h^C v)^T (W_x (x) W_y)(Delta_h^C w); l_h(w) = q h_x h_y sum w; Pi_h^C(w) = a_h(w, w)/2 - l_h(w);
  ||w||_a^2 = a_h(w, w); ||w||_h^2 = h_x h_y sum w^2.
EXACT functions (``*_exact``) compute these in rational arithmetic from the double inputs (delta_ref = 0). ``bilinear_d2_exact`` is
an independent second route: D h_x h_y v^T A w with A from the node-by-node integer parts of r5_cccc_d2 and their exact scales
(Proposition D). Larger grids: the exact path only (cost grows like the number of nodes); no raised-precision path is
implemented, so no uncertified decimal or double-double value is offered as a reference.

DOUBLE path (``pi_double``, ``q_double``) - the path of VOK-3 with a FIXED order of operations: the terms are formed node by node and
added by ``pairwise_sum`` (explicit binary tree of depth ceil(log2 M)). Operation count per term (G-4, by this code; P-1...P-5,
pow with an integer exponent within 1 ulp counted as 2u, the rounding of h counted as a perturbation of the exact h):
- stencil value t = L^_x w + L^_y w: numerator of L^ w 2 additions (|L^| w form), h_d = fl(l/n) (1 rounding; enters squared: 2),
  h_d**2 by pow (2), the division (1): 7 per part; the sum of the two parts (1): c_Delta = 8; boundary rows 2 f/h^2: 5 (+1);
- t*t: 2 c_Delta + 1 = 17 relative to |Delta w|(|Delta| w) (first order);
- the factor (D/2) w_I w_J: w_I = h or h/2 (1 + the halving, exact), w_J (1), w_I w_J (1), times D/2 (1, the halving exact): 4;
  the product with t*t (1): quadratic term c = 17 + 4 + 1 = 22;
- load term -(q h_x h_y) w_ij: q h_x h_y (h_x 1, h_y 1, two products 2): 4, times w (1): c = 5;
- the sum of all M = (n_x + 1)(n_y + 1) + (n_x - 1)(n_y - 1) terms: depth ceil(log2 M) (``pairwise_sum``).
Hence R^C(w) = u (ceil(log2 M) + c_R) S^C(w) with c_R = 22 FOR THIS PATH (``C_R_COUNT``); it is a count of a first-order model, not
a proof that R^C majorizes the actual error (second-order terms dropped; RO VOK-3: MODEL). The owner's c_R, K_E, eps_floor^C are not
assigned here. ``model_R`` returns R^C split as base = u ceil(log2 M) S^C and per_cR = u S^C (R^C = base + c_R per_cR).
"""

import math
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from kkvim_reference import r5_cccc_d2 as d2
from kkvim_reference import r5_cccc_halfstep as hs

U = 2.0**-53
U_EXACT = Fraction(1, 2**53)
C_R_COUNT = 22  # first-order operation count of the double path of this module (docstring), not an owner's value
DBL_MAX = 1.7976931348623157e308


def _positive(name, v) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
        raise ValueError(f"{name} must be a finite number > 0, got {v!r}")
    return float(v)


@dataclass(frozen=True)
class Grid:
    """Plate a x b = l_x x l_y, grids n_x, n_y >= 3, stiffness D > 0, load q (finite)."""

    lx: float
    nx: int
    ly: float
    ny: int
    D: float = 1.0
    q: float = 1.0

    def __post_init__(self):
        for name in ("lx", "ly", "D"):
            object.__setattr__(self, name, _positive(name, getattr(self, name)))
        for name in ("nx", "ny"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v < 3:
                raise ValueError(f"{name} must be an integer >= 3, got {v!r}")
            object.__setattr__(self, name, int(v))
        if isinstance(self.q, bool) or not isinstance(self.q, (int, float)) or not math.isfinite(self.q):
            raise ValueError(f"q must be finite, got {self.q!r}")
        object.__setattr__(self, "q", float(self.q))

    @property
    def hx(self) -> Fraction:
        return Fraction(self.lx) / self.nx

    @property
    def hy(self) -> Fraction:
        return Fraction(self.ly) / self.ny

    @property
    def M(self) -> int:
        return (self.nx + 1) * (self.ny + 1) + (self.nx - 1) * (self.ny - 1)


def _frac(w) -> list[list[Fraction]]:
    return [[x if isinstance(x, Fraction) else Fraction(float(x)) for x in row] for row in w]


def _check_shape(g: Grid, w) -> None:
    if len(w) != g.nx - 1 or any(len(row) != g.ny - 1 for row in w):
        raise ValueError(f"field must be (n_x - 1) x (n_y - 1) = {g.nx - 1} x {g.ny - 1}")


# --- exact path ---------------------------------------------------------------------------------------------------------

def _hat_L(f: list[Fraction], h: Fraction) -> list[Fraction]:
    n = len(f) + 1
    v = [Fraction(0)] + list(f) + [Fraction(0)]
    return [2 * v[1] / h**2] + [(v[i - 1] - 2 * v[i] + v[i + 1]) / h**2 for i in range(1, n)] + [2 * v[n - 1] / h**2]


def delta_exact(g: Grid, w) -> list[list[Fraction]]:
    """(Delta_h^C w)_IJ on all nodes I = 0..n_x, J = 0..n_y, exactly."""
    W = _frac(w)
    _check_shape(g, W)
    nx, ny = g.nx, g.ny
    out = [[Fraction(0)] * (ny + 1) for _ in range(nx + 1)]
    for j in range(ny - 1):  # L^_x (x) E_y: only interior J carry L^_x
        col = _hat_L([W[i][j] for i in range(nx - 1)], g.hx)
        for I in range(nx + 1):
            out[I][j + 1] += col[I]
    for i in range(nx - 1):  # E_x (x) L^_y
        row = _hat_L(W[i], g.hy)
        for J in range(ny + 1):
            out[i + 1][J] += row[J]
    return out


def _weights(n: int, h: Fraction) -> list[Fraction]:
    return [h / 2] + [h] * (n - 1) + [h / 2]


def bilinear_exact(g: Grid, v, w) -> Fraction:
    """a_h(v, w) = D (Delta v)^T (W_x (x) W_y)(Delta w), exactly (NS 2.2)."""
    dv = delta_exact(g, v)
    dw = dv if w is v else delta_exact(g, w)
    wx, wy = _weights(g.nx, g.hx), _weights(g.ny, g.hy)
    return Fraction(g.D) * sum((wx[I] * wy[J] * dv[I][J] * dw[I][J] for I in range(g.nx + 1) for J in range(g.ny + 1)),
                               Fraction(0))


def load_exact(g: Grid, w) -> Fraction:
    """l_h(w) = q h_x h_y sum_interior w, exactly."""
    return Fraction(g.q) * g.hx * g.hy * sum((x for row in _frac(w) for x in row), Fraction(0))


def pi_exact(g: Grid, w) -> Fraction:
    """Pi_h^C(w) = a_h(w, w)/2 - l_h(w), exactly."""
    return bilinear_exact(g, w, w) / 2 - load_exact(g, w)


def norm_a_sq_exact(g: Grid, w) -> Fraction:
    return bilinear_exact(g, w, w)


def norm_h_sq_exact(g: Grid, w) -> Fraction:
    return g.hx * g.hy * sum((x * x for row in _frac(w) for x in row), Fraction(0))


def bilinear_d2_exact(g: Grid, v, w) -> Fraction:
    """Independent route: D h_x h_y v^T A w, A = s_x Z_x + s_xy Z_xy + s_y Z_y from r5_cccc_d2.nodewise_parts (integer parts, node
    by node with explicit ghost nodes) and the exact scales; D2 ordering k = (i - 1)(n_y - 1) + (j - 1)."""
    V, Wf = _frac(v), _frac(w)
    vs = [x for row in V for x in row]
    ws = [x for row in Wf for x in row]
    prob = d2.D2Problem(g.lx, g.ly, g.nx, g.ny)
    total = Fraction(0)
    for z, s in zip(d2.nodewise_parts(g.nx, g.ny), prob.scales()):
        z = z.tocsr()
        acc = Fraction(0)
        for row in range(len(vs)):
            if vs[row]:
                acc += vs[row] * sum((int(z.data[k]) * ws[z.indices[k]] for k in range(z.indptr[row], z.indptr[row + 1])),
                                     Fraction(0))
        total += s * acc
    return Fraction(g.D) * g.hx * g.hy * total


def sqrt_interval(x: Fraction) -> tuple[Fraction, Fraction]:
    if x < 0:
        raise ValueError("negative square")
    if x == 0:
        return Fraction(0), Fraction(0)
    P, Q = x.numerator, x.denominator
    r = math.isqrt(P * Q * 4**64)
    return Fraction(r, Q * 2**64), Fraction(r + 1, Q * 2**64)


def down(x: Fraction) -> float:
    if x > Fraction(DBL_MAX):
        return DBL_MAX
    if x < -Fraction(DBL_MAX):
        return -math.inf
    return hs._float_down(x)


def up(x: Fraction) -> float:
    if x < -Fraction(DBL_MAX):
        return -DBL_MAX
    return hs._float_up(x)


def interval(x: Fraction) -> tuple[float, float]:
    """Directed double interval [down, up] of an exact rational."""
    return down(x), up(x)


# --- double path (VOK-3, fixed order) ----------------------------------------------------------------------------------------

def pairwise_sum(a: np.ndarray) -> float:
    """Sum by an explicit binary tree: every term passes through at most ceil(log2 len(a)) additions."""
    v = np.asarray(a, dtype=float).ravel()
    if v.size == 0:
        return 0.0
    while v.size > 1:
        if v.size % 2:
            v = np.append(v, 0.0)  # adding an exact zero does not round
        v = v[0::2] + v[1::2]
    return float(v[0])


def _delta_double(g: Grid, w: np.ndarray) -> np.ndarray:
    hx, hy = g.lx / g.nx, g.ly / g.ny
    P = np.zeros((g.nx + 1, g.ny + 1))
    P[1:-1, 1:-1] = w
    Lx = np.zeros_like(P)
    Lx[1:-1, 1:-1] = (P[:-2, 1:-1] - 2.0 * P[1:-1, 1:-1] + P[2:, 1:-1]) / hx**2
    Lx[0, 1:-1], Lx[-1, 1:-1] = 2.0 * P[1, 1:-1] / hx**2, 2.0 * P[-2, 1:-1] / hx**2
    Ly = np.zeros_like(P)
    Ly[1:-1, 1:-1] = (P[1:-1, :-2] - 2.0 * P[1:-1, 1:-1] + P[1:-1, 2:]) / hy**2
    Ly[1:-1, 0], Ly[1:-1, -1] = 2.0 * P[1:-1, 1] / hy**2, 2.0 * P[1:-1, -2] / hy**2
    return Lx + Ly


def _quad_terms_double(g: Grid, w: np.ndarray) -> np.ndarray:
    hx, hy = g.lx / g.nx, g.ly / g.ny
    wx = np.full(g.nx + 1, hx)
    wx[[0, -1]] = hx / 2
    wy = np.full(g.ny + 1, hy)
    wy[[0, -1]] = hy / 2
    t = _delta_double(g, w)
    return (g.D / 2) * np.outer(wx, wy) * (t * t)


def pi_double(g: Grid, w: np.ndarray) -> float:
    """Pi_h^C(w) in double: quadratic terms (D/2) w_I w_J t^2 and load terms -(q h_x h_y) w_ij, one pairwise sum of all M terms."""
    w = np.asarray(w, dtype=float)
    load = (g.q * (g.lx / g.nx) * (g.ly / g.ny)) * w
    return pairwise_sum(np.concatenate([_quad_terms_double(g, w).ravel(), -load.ravel()]))


def q_double(g: Grid, d: np.ndarray) -> float:
    """Q = ||d||_a^2 / 2 in double (quadratic terms only, pairwise sum)."""
    return pairwise_sum(_quad_terms_double(g, np.asarray(d, dtype=float)))


# --- model terms (VOK-3, VOK-5, VOK-7, VOK-8) ------------------------------------------------------------------------------

def _abs_delta_exact(g: Grid, w) -> list[list[Fraction]]:
    """(|Delta| w)_IJ: the stencil with the moduli of the coefficients applied to |w| (VOK-3)."""
    A = [[abs(x) for x in row] for row in _frac(w)]
    nx, ny = g.nx, g.ny
    out = [[Fraction(0)] * (ny + 1) for _ in range(nx + 1)]

    def absL(f, h):
        n = len(f) + 1
        v = [Fraction(0)] + list(f) + [Fraction(0)]
        return [2 * v[1] / h**2] + [(v[i - 1] + 2 * v[i] + v[i + 1]) / h**2 for i in range(1, n)] + [2 * v[n - 1] / h**2]

    for j in range(ny - 1):
        col = absL([A[i][j] for i in range(nx - 1)], g.hx)
        for I in range(nx + 1):
            out[I][j + 1] += col[I]
    for i in range(nx - 1):
        row = absL(A[i], g.hy)
        for J in range(ny + 1):
            out[i + 1][J] += row[J]
    return out


def majorant_S(g: Grid, w) -> tuple[Fraction, Fraction]:
    """S^C(w) of VOK-3, exactly: (quadratic part (D/2) sum W |Delta w| (|Delta| w), load part |q| h_x h_y sum |w|)."""
    dw, aw = delta_exact(g, w), _abs_delta_exact(g, w)
    wx, wy = _weights(g.nx, g.hx), _weights(g.ny, g.hy)
    quad = Fraction(g.D) / 2 * sum((wx[I] * wy[J] * abs(dw[I][J]) * aw[I][J] for I in range(g.nx + 1) for J in range(g.ny + 1)),
                                   Fraction(0))
    load = abs(Fraction(g.q)) * g.hx * g.hy * sum((abs(x) for row in _frac(w) for x in row), Fraction(0))
    return quad, load


def model_R(g: Grid, w) -> dict:
    """R^C(w) = u (ceil(log2 M) + c_R) S^C(w) split as base + c_R per_cR (MODEL; c_R not assigned)."""
    quad, load = majorant_S(g, w)
    S = quad + load
    L = math.ceil(math.log2(g.M))
    return {"R_base_up": up(U_EXACT * L * S), "R_per_cR_up": up(U_EXACT * S), "ceil_log2_M": L, "S_C_up": up(S)}


def model_RQ(g: Grid, d) -> dict:
    """R_Q^C(d) = u (ceil(log2 M) + c_R) (quadratic part of S^C(d)) + u D^(1/2)(4/h_x^2 + 4/h_y^2)||d||_h ||d||_a (MODEL)."""
    quad, _ = majorant_S(g, d)
    L = math.ceil(math.log2(g.M))
    sub = U_EXACT * sqrt_interval(Fraction(g.D))[1] * (4 / g.hx**2 + 4 / g.hy**2) * sqrt_interval(norm_h_sq_exact(g, d))[1] * \
        sqrt_interval(norm_a_sq_exact(g, d))[1]
    return {"RQ_base_up": up(U_EXACT * L * quad + sub), "RQ_per_cR_up": up(U_EXACT * quad)}


def eps_a_C(g: Grid, X, Y) -> Fraction:
    """VOK-5 (upper, exact rational): gamma_N D^(1/2)(4/h_x^2 + 4/h_y^2)||S||_h, S = |X||Y|^T (formation of X Y^T)."""
    N = len(X[0])
    Xa = [[abs(x) for x in row] for row in _frac(X)]
    Ya = [[abs(x) for x in row] for row in _frac(Y)]
    S = [[sum((a * b for a, b in zip(x, y)), Fraction(0)) for y in Ya] for x in Xa]
    gam = N * U_EXACT / (1 - N * U_EXACT)
    return gam * sqrt_interval(Fraction(g.D))[1] * (4 / g.hx**2 + 4 / g.hy**2) * sqrt_interval(norm_h_sq_exact(g, S))[1]


def b_omega(g: Grid, w_double: np.ndarray, formation_bound: Fraction, field_cert: Fraction | None) -> tuple[Fraction, str]:
    """Upper bound of ||w~ - omega_h^C||_a for the exact product field w~ of the arrays whose double field is ``w_double`` (VOK-7):
    (b) ||D A w^ - q 1||_h/(D^(1/2)(mu_x + mu_y)) (exact residual of D2, mu_1 >= 4/l^2) plus ||w~ - w^||_a <= formation_bound;
    (a) |q|(ab)^(1/2)/(D^(1/2)(mu_x + mu_y)) plus ||w~ - w*||_a <= field_cert (VOK-14 energy certificate). The smaller one."""
    mu = Fraction(4) / Fraction(g.lx) ** 2 + Fraction(4) / Fraction(g.ly) ** 2  # lower bound of mu_1^x + mu_1^y
    denom = sqrt_interval(Fraction(g.D))[0] * mu
    prob = d2.D2Problem(g.lx, g.ly, g.nx, g.ny, D=g.D, q=g.q)
    r = d2.residual_exact(prob, np.asarray(w_double, dtype=float).reshape(-1))  # q/D - A w
    res_h = Fraction(g.D) * sqrt_interval(g.hx * g.hy * sum((x * x for x in r), Fraction(0)))[1]  # ||D A w - q 1||_h
    best, how = res_h / denom + formation_bound, "VOK-7 (b) + formation"
    if field_cert is not None:
        a = abs(Fraction(g.q)) * sqrt_interval(Fraction(g.lx) * Fraction(g.ly))[1] / denom + field_cert
        if a < best:
            best, how = a, "VOK-7 (a) + VOK-14 energy certificate"
    return best, how
