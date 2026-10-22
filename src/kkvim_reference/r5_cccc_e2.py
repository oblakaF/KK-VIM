"""QA instrument for the measurement E-2 (R5-V CCCC): the rounding error of FORMING the discrete field W = U V^T.

Preparation only: no Q3 sample, no thresholds (no eps_floor^C, K_E), no working solver; no import of ``kkvim`` (D-028).
Method: docs/R5_CCCC_ROUNDOFF_PREQUAL_DRAFT.md @ 96914c9, section 4.3, E-2 with the corrections N-1/N-2: direct measurement.

1. The double arrays are fixed (``FieldInputs``: read-only copies, sha256 of their bytes). X holds the x-factor ((n_x - 1) x N),
   Y the y-factor ((n_y - 1) x N); W = X Y^T on the interior nodes (x index first). For a Y-half-step X = U (given) and Y = V
   (computed); for an X-half-step X is computed and Y given - the product is the same.
2. w^ = fl(X Y^T) by the standard call ``X @ Y.T`` (numpy matmul, BLAS dgemm); w~ = X Y^T of the SAME doubles exactly
   (``fractions``). e = w^ - w~ is then exact. X, Y are common to w^ and w~, so e contains only the formation error (i) of VOK-9:
   no error of the half-step solution (ii) and no change of basis (iii).
3. Norms (roundoff document 0.2; NS 2.2): ||v||_h^2 = h_x h_y sum_interior v_ij^2; ||v||_a^2 = a_h(v, v) =
   D (Delta_h^C v)^T (W_x (x) W_y) (Delta_h^C v), Delta_h^C = L^_x (x) E_y + E_x (x) L^_y with the boundary rows (L^ f)_0 = 2 f_1/h^2,
   W = diag(h/2, h, ..., h, h/2) - the quadratic part of Pi_h^C exactly by its definition (h = l/n exactly). The squares are exact
   rationals; the norms are reported as intervals [down, up] (integer square root, directed conversion to double).
4. VOK-4 elementwise |e_ij| <= gamma_N (|X| |Y|^T)_ij and VOK-5 ||e||_a <= eps_a^C = gamma_N D^(1/2) (4/h_x^2 + 4/h_y^2) ||S||_h
   (= gamma_N kappa_W D^(1/2)(4/h_x^2 + 4/h_y^2) ||w^||_h) are checked EXACTLY (rational comparisons of squares).
5. Separately (component (iv)): ||w^||_a^2 evaluated in double by two QA paths (the definition of Pi_h^C; the sparse Hessian of
   r5_cccc_d2) against the exact value - OBSERVATION; a rigorous bound for these paths (VOK-3 is a MODEL, c_R open) is UNRESOLVED.
6. Undefined ratios (zero field) are UNRESOLVED. The swap of columns is only an auxiliary check (``swap_difference``) and is never
   used by ``measure``: with sequential summation and N = 2 it gives a false zero (N-1).
Provenance (versions, BLAS, platform, call, digests) is recorded with every measurement (D-025).
"""

import hashlib
import math
import platform
import sys
from dataclasses import dataclass, field
from fractions import Fraction
from importlib import metadata

import numpy as np

from kkvim_reference import r5_cccc_d2 as d2
from kkvim_reference import r5_cccc_halfstep as hs

UNRESOLVED = "UNRESOLVED"
U_EXACT = Fraction(1, 2**53)
STANDARD_CALL = "numpy.matmul: X @ Y.T (float64, BLAS dgemm)"


def gamma_exact(m: int) -> Fraction:
    return m * U_EXACT / (1 - m * U_EXACT)


# --- inputs and provenance ------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class FieldInputs:
    """Fixed double arrays of one field: X ((n_x - 1) x N), Y ((n_y - 1) x N) on the plate l_x x l_y with grids n_x, n_y."""

    X: np.ndarray
    Y: np.ndarray
    lx: float
    nx: int
    ly: float
    ny: int
    D: float = 1.0
    origin: str = ""

    def __post_init__(self):
        X = np.array(self.X, dtype=np.float64, order="C", copy=True)
        Y = np.array(self.Y, dtype=np.float64, order="C", copy=True)
        if X.ndim != 2 or Y.ndim != 2 or X.shape[1] != Y.shape[1] or not 1 <= X.shape[1] <= 3:
            raise ValueError("X, Y must be 2-D with the same number N = 1..3 of columns")
        if X.shape[0] != self.nx - 1 or Y.shape[0] != self.ny - 1 or self.nx < 3 or self.ny < 3:
            raise ValueError("X must be (n_x - 1) x N, Y (n_y - 1) x N, n >= 3")
        if not (np.all(np.isfinite(X)) and np.all(np.isfinite(Y))):
            raise ValueError("X, Y must be finite")
        X.flags.writeable = False
        Y.flags.writeable = False
        object.__setattr__(self, "X", X)
        object.__setattr__(self, "Y", Y)

    @property
    def N(self) -> int:
        return self.X.shape[1]

    def digest(self) -> str:
        h = hashlib.sha256()
        for a in (self.X, self.Y):
            h.update(str(a.shape).encode())
            h.update(a.tobytes())
        h.update(repr((self.lx, self.nx, self.ly, self.ny, self.D)).encode())
        return h.hexdigest()


def provenance() -> dict:
    """Environment of the measurement (D-025): versions, BLAS of numpy, platform, the standard call."""
    blas = "unknown"
    try:
        cfg = np.__config__.CONFIG  # numpy >= 1.26
        b = cfg.get("Build Dependencies", {}).get("blas", {})
        blas = f"{b.get('name', '?')} {b.get('version', '')}".strip()
    except AttributeError:
        pass
    return {"python": platform.python_version(), "implementation": sys.implementation.name, "platform": platform.platform(),
            "machine": platform.machine(), "numpy": metadata.version("numpy"), "scipy": metadata.version("scipy"),
            "blas": blas, "standard_call": STANDARD_CALL, "reference": "fractions.Fraction (exact)"}


# --- the field: standard, exact, auxiliary ---------------------------------------------------------------------------

def field_standard(inp: FieldInputs) -> np.ndarray:
    """w^ = fl(X Y^T) by the standard call."""
    return inp.X @ inp.Y.T


def field_exact(inp: FieldInputs) -> list[list[Fraction]]:
    """w~ = X Y^T of the same doubles, exactly."""
    X = [[Fraction(float(v)) for v in row] for row in inp.X]
    Y = [[Fraction(float(v)) for v in row] for row in inp.Y]
    return [[sum((a * b for a, b in zip(x, y)), Fraction(0)) for y in Y] for x in X]


def field_sequential(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """fl(sum_k X_:k Y_:k^T) with the terms added in the order k = 1..N (auxiliary)."""
    w = np.outer(X[:, 0], Y[:, 0])
    for k in range(1, X.shape[1]):
        w = w + np.outer(X[:, k], Y[:, k])
    return w


def swap_difference(inp: FieldInputs, sequential: bool = True) -> float:
    """AUXILIARY only (N-1): max |field(X, Y) - field(X P, Y P)| for the reversed column order P. Not a measure of the formation
    error: + and * are commutative, so with sequential summation and N = 2 (and always for N = 1) it is exactly 0."""
    f = field_sequential if sequential else (lambda X, Y: X @ Y.T)
    p = list(reversed(range(inp.N)))
    return float(np.max(np.abs(f(inp.X, inp.Y) - f(inp.X[:, p], inp.Y[:, p]))))


# --- exact norms ------------------------------------------------------------------------------------------------------

def _steps(inp: FieldInputs):
    return Fraction(inp.lx) / inp.nx, Fraction(inp.ly) / inp.ny


def norm_h_sq(inp: FieldInputs, v) -> Fraction:
    """||v||_h^2 = h_x h_y sum_interior v_ij^2, exactly (v: rows x, columns y; Fractions or doubles)."""
    hx, hy = _steps(inp)
    return hx * hy * sum((Fraction(x) ** 2 for row in v for x in row), Fraction(0))


def _hat_L(f: list[Fraction], h: Fraction) -> list[Fraction]:
    """L^ f on the nodes 0..n (NS 2.2): interior (f_{i-1} - 2 f_i + f_{i+1})/h^2 with f_0 = f_n = 0; (L^ f)_0 = 2 f_1/h^2,
    (L^ f)_n = 2 f_{n-1}/h^2 (even ghost node)."""
    n = len(f) + 1
    g = [Fraction(0)] + list(f) + [Fraction(0)]
    out = [2 * g[1] / h**2]
    out += [(g[i - 1] - 2 * g[i] + g[i + 1]) / h**2 for i in range(1, n)]
    out.append(2 * g[n - 1] / h**2)
    return out


def energy_sq(inp: FieldInputs, v) -> Fraction:
    """||v||_a^2 = a_h(v, v) = D sum_{nodes 0..n_x, 0..n_y} (W_x (x) W_y)_ij ((Delta_h^C v)_ij)^2, exactly (NS 2.2)."""
    hx, hy = _steps(inp)
    nx, ny = inp.nx, inp.ny
    V = [[Fraction(x) for x in row] for row in v]  # (n_x - 1) x (n_y - 1)
    Lx = [_hat_L([V[i][j] for i in range(nx - 1)], hx) for j in range(ny - 1)]  # per interior y-node: values at x-nodes 0..n_x
    Ly = [_hat_L(V[i], hy) for i in range(nx - 1)]  # per interior x-node: values at y-nodes 0..n_y
    wx = [hx / 2] + [hx] * (nx - 1) + [hx / 2]
    wy = [hy / 2] + [hy] * (ny - 1) + [hy / 2]
    total = Fraction(0)
    for I in range(nx + 1):
        for J in range(ny + 1):
            val = Fraction(0)
            if 1 <= J <= ny - 1:
                val += Lx[J - 1][I]  # (L^_x (x) E_y v)_IJ
            if 1 <= I <= nx - 1:
                val += Ly[I - 1][J]  # (E_x (x) L^_y v)_IJ
            if val:
                total += wx[I] * wy[J] * val * val
    return Fraction(inp.D) * total


def sqrt_interval(x: Fraction) -> tuple[Fraction, Fraction]:
    """[lo, hi] with lo <= sqrt(x) <= hi, rational, via an integer square root (x >= 0)."""
    if x == 0:
        return Fraction(0), Fraction(0)
    P, Q = x.numerator, x.denominator
    k = 64
    r = math.isqrt(P * Q * 4**k)
    return Fraction(r, Q * 2**k), Fraction(r + 1, Q * 2**k)


def _interval(lo: Fraction, hi: Fraction) -> tuple[float, float]:
    return hs._float_down(lo), hs._float_up(hi)


# --- double paths for component (iv) ------------------------------------------------------------------------------------

def energy_sq_double_definition(inp: FieldInputs, w: np.ndarray) -> float:
    """a_h(w, w) in double by the definition of Pi_h^C (Delta_h^C, trapezoid weights), numpy vectorised (OBSERVATION path)."""
    hx, hy = inp.lx / inp.nx, inp.ly / inp.ny
    P = np.zeros((inp.nx + 1, inp.ny + 1))
    P[1:-1, 1:-1] = w
    Lx = np.zeros_like(P)
    Lx[1:-1, :] = (P[:-2, :] - 2 * P[1:-1, :] + P[2:, :]) / hx**2
    Lx[0, :], Lx[-1, :] = 2 * P[1, :] / hx**2, 2 * P[-2, :] / hx**2
    Ly = np.zeros_like(P)
    Ly[:, 1:-1] = (P[:, :-2] - 2 * P[:, 1:-1] + P[:, 2:]) / hy**2
    Ly[:, 0], Ly[:, -1] = 2 * P[:, 1] / hy**2, 2 * P[:, -2] / hy**2
    Lx[:, 0] = Lx[:, -1] = 0.0  # E_y: only interior y-nodes carry L^_x
    Ly[0, :] = Ly[-1, :] = 0.0  # E_x: only interior x-nodes carry L^_y
    wx = np.full(inp.nx + 1, hx)
    wx[[0, -1]] = hx / 2
    wy = np.full(inp.ny + 1, hy)
    wy[[0, -1]] = hy / 2
    return float(inp.D * np.sum(np.outer(wx, wy) * (Lx + Ly) ** 2))


def energy_sq_double_hessian(inp: FieldInputs, w: np.ndarray) -> float:
    """a_h(w, w) = D h_x h_y w^T A2 w in double with the sparse Hessian of r5_cccc_d2 (Proposition D; OBSERVATION path)."""
    A2 = d2.energy_hessian(d2.D2Problem(inp.lx, inp.ly, inp.nx, inp.ny))
    x = np.asarray(w, dtype=float).reshape(-1)  # D2 ordering: x index slow
    return float(inp.D * (inp.lx / inp.nx) * (inp.ly / inp.ny) * (x @ (A2 @ x)))


# --- the measurement --------------------------------------------------------------------------------------------------

@dataclass
class E2Measurement:
    values: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)


def measure(inp: FieldInputs) -> E2Measurement:
    """E-2 for one fixed pair (X, Y). Exact quantities: e, VOK-4 and VOK-5 checks, the squares of the norms. Intervals [down, up] for
    the norms and their ratios. Ratios with a zero denominator are UNRESOLVED. The double evaluations of ||w^||_a^2 are
    OBSERVATION with their exact relative error (component (iv)); their rigorous bound is UNRESOLVED."""
    w_hat = field_standard(inp)
    w_ex = field_exact(inp)
    N = inp.N
    e = [[Fraction(float(w_hat[i, j])) - w_ex[i][j] for j in range(inp.ny - 1)] for i in range(inp.nx - 1)]
    Xa = [[abs(Fraction(float(v))) for v in row] for row in inp.X]
    Ya = [[abs(Fraction(float(v))) for v in row] for row in inp.Y]
    S = [[sum((a * b for a, b in zip(x, y)), Fraction(0)) for y in Ya] for x in Xa]
    g = gamma_exact(N)
    viol, worst = 0, Fraction(0)
    for i in range(inp.nx - 1):
        for j in range(inp.ny - 1):
            if abs(e[i][j]) > g * S[i][j]:
                viol += 1
            if S[i][j] > 0:
                worst = max(worst, abs(e[i][j]) / (g * S[i][j]))
    max_e = max(abs(x) for row in e for x in row)
    max_w = max(abs(Fraction(float(x))) for x in w_hat.reshape(-1))
    eh2, ea2 = norm_h_sq(inp, e), energy_sq(inp, e)
    wh2, wa2 = norm_h_sq(inp, w_hat), energy_sq(inp, w_hat)
    Sh2 = norm_h_sq(inp, S)
    hx, hy = _steps(inp)
    c2 = Fraction(inp.D) * (4 / hx**2 + 4 / hy**2) ** 2  # (D^(1/2)(4/h_x^2 + 4/h_y^2))^2
    eps_a2 = g * g * c2 * Sh2  # (eps_a^C)^2, exact
    v = {"N": N, "n_x": inp.nx, "n_y": inp.ny, "origin": inp.origin, "digest": inp.digest(),
         "max_abs_e_up": hs._float_up(max_e), "e_is_zero": max_e == 0,
         "max_rel_e_up": hs._float_up(max_e / max_w) if max_w else UNRESOLVED,
         "vok4_violations": viol, "vok4_max_ratio_up": hs._float_up(worst),
         "vok1_upper_holds": ea2 <= c2 * eh2,  # ||e||_a <= D^(1/2)(4/h_x^2 + 4/h_y^2)||e||_h (VOK-1, item 4), exact
         "vok5_holds": ea2 <= eps_a2}  # ||e||_a <= eps_a^C, exact
    for name, sq in (("e_h", eh2), ("e_a", ea2), ("w_h", wh2), ("w_a", wa2), ("S_h", Sh2), ("eps_a_C", eps_a2)):
        lo, hi = sqrt_interval(sq)
        v[f"{name}_down"], v[f"{name}_up"] = _interval(lo, hi)
    if wa2 > 0:
        (el, eu), (wl, wu) = sqrt_interval(ea2), sqrt_interval(wa2)
        v["rel_e_a_down"], v["rel_e_a_up"] = _interval(el / wu, eu / wl)
        el2, eu2 = sqrt_interval(eps_a2)
        v["eps_a_C_rel_down"], v["eps_a_C_rel_up"] = _interval(el2 / wu, eu2 / wl)
    else:
        v["rel_e_a_down"] = v["rel_e_a_up"] = v["eps_a_C_rel_down"] = v["eps_a_C_rel_up"] = UNRESOLVED
    if wh2 > 0:
        (sl, su), (wl, wu) = sqrt_interval(Sh2), sqrt_interval(wh2)
        v["kappa_W_down"], v["kappa_W_up"] = _interval(sl / wu, su / wl)
    else:
        v["kappa_W_down"] = v["kappa_W_up"] = UNRESOLVED
    if eps_a2 > 0:
        (el, eu), (bl, bu) = sqrt_interval(ea2), sqrt_interval(eps_a2)
        v["e_a_over_eps_a_C_down"], v["e_a_over_eps_a_C_up"] = _interval(el / bu, eu / bl)
    else:
        v["e_a_over_eps_a_C_down"] = v["e_a_over_eps_a_C_up"] = UNRESOLVED
    # component (iv): double evaluation of ||w^||_a^2 (OBSERVATION; bound UNRESOLVED)
    for path, fn in (("definition", energy_sq_double_definition), ("hessian", energy_sq_double_hessian)):
        val = fn(inp, w_hat)
        v[f"w_a_sq_double_{path}"] = val
        v[f"w_a_sq_double_{path}_rel_error"] = float(abs(Fraction(val) - wa2) / wa2) if wa2 else \
            ("0 (exact)" if val == 0 else UNRESOLVED)
    v["w_a_sq_double_bound"] = UNRESOLVED  # VOK-3 is a MODEL (c_R not counted): no rigorous bound for the double paths
    return E2Measurement(v, {**provenance(), "inputs_sha256": inp.digest()})
