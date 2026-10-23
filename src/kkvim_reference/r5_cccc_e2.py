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

Premises of the rounding model (roundoff document 0.4; review 907de22, IE-1...IE-3). VOK-4 and VOK-5 are bounds UNDER P-1...P-5;
the exact comparisons are normative only where the premises are established for the operations that formed w^:
- P-2 (no overflow): established iff w^ is finite. This uses P-1, rounding to nearest (N-b, review f5f85e1): an overflow then
  produces +-inf, which cannot disappear in later additions or FMA (inf - inf gives NaN); with rounding towards zero an overflow
  would give +-DBL_MAX and the conclusion would not hold. Otherwise OUTSIDE_P2: no error e, no VOK-4/VOK-5 check, no double path (iv).
- P-3 (no underflow), threshold 2^-1022 (smallest normal binary64): ``p3_premise``. All inputs normal or zero, and for every k and
  nonzero pair e(X_ik) + e(Y_jk) >= -918 (e = exponent, |x| in [2^e, 2^(e+1))) => P3_CONFIRMED. Proof (own derivation, level B):
  the exact product X_ik Y_jk is a multiple of 2^(e(X)+e(Y)-104) >= G := 2^-1022; sums of multiples of G are multiples of G, and
  rounding a multiple of G to the nearest double gives a multiple of G (either the result is exact, or its ulp is >= G); so every
  value of any evaluation order, with or without FMA, is 0 or has magnitude >= G - never subnormal. If this cannot be shown:
  OUTSIDE_P3 when an operation with a nonzero subnormal exact result certainly occurred (a nonzero subnormal w^_ij; a node whose
  only nonzero product is below 2^-1022 - with N = 1 that is the product itself) and, as a CONSERVATIVE restriction of the domain
  (N-a, review f5f85e1), for any nonzero subnormal input - even when no subnormal operation actually occurs (e.g. the subnormal
  entry meets only zeros of the other factor), otherwise
  UNRESOLVED_PREMISES (potential subnormal intermediates; possible exact sums of the products are listed). The exact
  comparisons remain available as OBSERVATION (``vok4_violations``, ``vok5_holds``); the normative results are ``vok4_status``,
  ``vok5_status``: HOLDS / VIOLATED only under P3_CONFIRMED, else the premise status.
- P-4, P-5: N <= 3 (no fast multiplication of length <= 3; N u < 1).
The input domain (IE-3) is checked in ``FieldInputs``: l_x, l_y, D finite and > 0 with the exact value of the argument equal to a
binary64 number (any int, float, numpy.integer, numpy.floating, including numpy.longdouble; IE-3a), n_x, n_y integers >= 3, X, Y
real numeric arrays exactly representable in float64, not masked arrays.
"""

import contextlib
import hashlib
import io
import math
import os
import platform
import sys
import warnings
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

DBL_MAX = sys.float_info.max
NORMAL_MIN = 2.0**-1022  # smallest normal binary64 (IEEE 754), the P-3 threshold
_NORMAL_MIN_EXACT = Fraction(1, 2**1022)


def _positive_real(name: str, v) -> float:
    """A finite positive real number, exactly representable as a double (IE-3)."""
    if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a real number (int or float, not bool, complex or str), got {v!r}")
    try:
        f = float(v)
    except OverflowError:
        raise ValueError(f"{name} is outside the double range: {v!r}") from None
    if not math.isfinite(f) or f <= 0:
        raise ValueError(f"{name} must be finite and > 0 (plate a x b, D > 0; NS 2.2), got {v!r}")
    # IE-3a (review f5f85e1): the EXACT value of the argument must equal the double, for every accepted type (numpy.longdouble may be
    # wider than binary64); no silent rounding, and no acceptance when the exact value cannot be determined
    try:
        exact = Fraction(int(v)) if isinstance(v, (int, np.integer)) else Fraction(*v.as_integer_ratio())
    except (AttributeError, TypeError, ValueError, OverflowError):
        raise ValueError(f"{name}: the exact value of {type(v).__name__} cannot be determined; not accepted") from None
    if Fraction.from_float(f) != exact:
        raise ValueError(f"{name} is not exactly representable as a double (binary64): {v!r}")
    return f


def _real_array(name: str, a) -> np.ndarray:
    """A real numeric array converted EXACTLY to a read-only C-contiguous float64 copy; complex, bool, object, str are refused
    (IE-3: no silent conversion). A numpy.ma.MaskedArray is refused: converting it would drop the mask silently (N-c)."""
    if isinstance(a, np.ma.MaskedArray):
        raise ValueError(f"{name} must not be a masked array (numpy.ma.MaskedArray): the mask would be lost silently")
    arr = np.asarray(a)
    if arr.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be a real numeric array (float or integer), got dtype {arr.dtype}")
    out = np.array(arr, dtype=np.float64, order="C", copy=True)
    with np.errstate(all="ignore"):
        exact = np.array_equal(out.astype(arr.dtype), arr) if arr.size else True
    if not exact:
        raise ValueError(f"{name} is not exactly representable in float64 (dtype {arr.dtype})")
    return out


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
        # IE-3: the domain of NS 2.2 (plate a x b, D > 0, grids n >= 3), checked before any computation
        for name in ("lx", "ly", "D"):
            object.__setattr__(self, name, _positive_real(name, getattr(self, name)))
        for name in ("nx", "ny"):
            v = getattr(self, name)
            if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or v < 3:
                raise ValueError(f"{name} must be an integer (int or numpy.integer, not bool) >= 3, got {v!r}")
            object.__setattr__(self, name, int(v))
        X, Y = _real_array("X", self.X), _real_array("Y", self.Y)
        if X.ndim != 2 or Y.ndim != 2 or X.shape[1] != Y.shape[1] or not 1 <= X.shape[1] <= 3:
            raise ValueError("X, Y must be 2-D with the same number N = 1..3 of columns")
        if X.shape[0] != self.nx - 1 or Y.shape[0] != self.ny - 1:
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
            "blas": blas, "standard_call": STANDARD_CALL, "reference": "fractions.Fraction (exact)", **_runtime()}


def _runtime() -> dict:
    """Run-time information available without new dependencies (N-1). Missing data is reported as missing, never as a
    confirmation of the IEEE mode; the FTZ/DAZ self-test covers one 1 x 1 matmul of this run only."""
    out = {"blas_runtime": "unavailable (threadpoolctl not installed): BLAS kernel and thread count unknown",
           "threads_env": {k: os.environ.get(k, "unset") for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
           "cpu_count": os.cpu_count()}
    try:
        import threadpoolctl  # optional
        out["blas_runtime"] = [{k: d.get(k) for k in ("internal_api", "version", "num_threads", "architecture", "threading_layer")}
                               for d in threadpoolctl.threadpool_info()]
    except ImportError:
        pass
    try:
        from numpy._core._multiarray_umath import __cpu_features__  # private: optional
        out["cpu_features_used_by_numpy"] = sorted(k for k, v in __cpu_features__.items() if v)
    except ImportError:
        out["cpu_features_used_by_numpy"] = "unavailable"
    a, b = np.array([[2.0**-540]]), np.array([[2.0**-530]])
    ftz = (a @ b)[0, 0] == 2.0**-1070  # subnormal RESULT kept?
    c, d = np.array([[2.0**-1060]]), np.array([[2.0**20]])
    daz = (c @ d)[0, 0] == 2.0**-1040  # subnormal INPUT kept?
    out["ftz_daz_selftest"] = ("subnormal result and input preserved in a 1 x 1 matmul of this run" if ftz and daz else
                               f"NOT preserved: result kept = {bool(ftz)}, input kept = {bool(daz)}") + \
        "; a self-test of one call, not a proof of the IEEE mode of every kernel"
    return out


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


def _down(x: Fraction) -> float:
    """Largest double <= x; a positive x above DBL_MAX gives DBL_MAX (a valid lower bound; IE-2). Local wrapper: the accepted
    hs._float_down of Q1b is not changed."""
    if x > Fraction(DBL_MAX):
        return DBL_MAX
    return hs._float_down(x)


def _interval(lo: Fraction, hi: Fraction) -> tuple[float, float]:
    return _down(lo), hs._float_up(hi)  # _float_up gives +inf above the double range


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


# --- premises P-2, P-3 (IE-1, IE-2) ------------------------------------------------------------------------------------

P3_CONFIRMED, OUTSIDE_P3, UNRESOLVED_PREMISES, OUTSIDE_P2 = "P3_CONFIRMED", "OUTSIDE_P3", "UNRESOLVED_PREMISES", "OUTSIDE_P2"


def _subnormal(a: np.ndarray) -> np.ndarray:
    return (a != 0) & (np.abs(a) < NORMAL_MIN)


def p3_premise(inp: FieldInputs, w_hat: np.ndarray) -> tuple[str, list[str]]:
    """Status of P-3 for the formation of w^ (finite) and the reasons (module docstring; threshold 2^-1022)."""
    X, Y = inp.X, inp.Y
    reasons = []
    if np.any(_subnormal(X)) or np.any(_subnormal(Y)):
        reasons.append("nonzero subnormal input")
    if np.any(_subnormal(w_hat)):
        reasons.append("nonzero subnormal result w^_ij")
    if reasons:
        return OUTSIDE_P3, reasons
    ex = np.frexp(X)[1] - 1  # |x| in [2^ex, 2^(ex+1)) for normal x
    ey = np.frexp(Y)[1] - 1
    big = 10**6
    min_ex = np.where(X != 0, ex, big).min(axis=0)  # per column k, over nonzero entries
    min_ey = np.where(Y != 0, ey, big).min(axis=0)
    active = (min_ex < big) & (min_ey < big)
    if not np.any(active) or np.all((min_ex + min_ey)[active] >= -918):
        return P3_CONFIRMED, ["every nonzero product has e(X) + e(Y) >= -918: all values are multiples of 2^-1022"]
    # cannot be shown: exact analysis of the products and of their possible sums (N <= 3)
    Xf = [[Fraction(float(v)) for v in row] for row in X]
    Yf = [[Fraction(float(v)) for v in row] for row in Y]
    N = inp.N
    subsets = [c for r in range(2, N + 1) for c in __import__("itertools").combinations(range(N), r)]
    certain, tiny_products, tiny_sums = False, 0, 0
    for xr in Xf:
        for yr in Yf:
            prods = [a * b for a, b in zip(xr, yr)]
            nz = [q for q in prods if q != 0]
            small = [q for q in nz if abs(q) < _NORMAL_MIN_EXACT]
            tiny_products += len(small)
            if len(nz) == 1 and small:
                certain = True  # the only nonzero contribution: its operation has a nonzero subnormal exact result
            for c in subsets:
                t = sum((prods[k] for k in c), Fraction(0))
                if t != 0 and abs(t) < _NORMAL_MIN_EXACT:
                    tiny_sums += 1
    reasons = [f"nonzero products below 2^-1022: {tiny_products}", f"nonzero exact partial sums below 2^-1022: {tiny_sums}",
               "products with e(X) + e(Y) < -918: absence of subnormal intermediates not provable"]
    if certain:
        return OUTSIDE_P3, ["a node whose only nonzero product is below 2^-1022"] + reasons
    return UNRESOLVED_PREMISES, reasons


# --- the measurement --------------------------------------------------------------------------------------------------

@dataclass
class E2Measurement:
    values: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)


def measure(inp: FieldInputs) -> E2Measurement:
    """E-2 for one fixed pair (X, Y). Exact quantities: e, the comparisons of VOK-1, VOK-4, VOK-5 (OBSERVATION), the squares of the
    norms; intervals [down, up] for norms and ratios; ratios with a zero denominator are UNRESOLVED. Normative statuses of VOK-4,
    VOK-5 only under the premises (``premises``: P3_CONFIRMED, OUTSIDE_P3, UNRESOLVED_PREMISES, OUTSIDE_P2). The double evaluations
    of ||w^||_a^2 are OBSERVATION with their exact relative error (component (iv)); their rigorous bound is UNRESOLVED. The sha256
    of the inputs is checked at the start and at the end (N-2)."""
    digest_start = inp.digest()
    with np.errstate(all="ignore"):  # only the reporting of overflow/underflow; the arithmetic of the call is unchanged
        w_hat = field_standard(inp)
    N = inp.N
    v = {"N": N, "n_x": inp.nx, "n_y": inp.ny, "origin": inp.origin, "digest": digest_start}
    prov = {**provenance(), "inputs_sha256": digest_start}
    if not np.all(np.isfinite(w_hat)):  # P-2 violated in the formation (IE-2)
        w_ex = field_exact(inp)
        v.update({"premises": OUTSIDE_P2, "premise_reasons": ["w^ not finite: overflow in the formation"],
                  "nonfinite_entries": int(np.sum(~np.isfinite(w_hat))),
                  "max_abs_exact_field_up": hs._float_up(max(abs(x) for row in w_ex for x in row)),
                  "vok4_status": OUTSIDE_P2, "vok5_status": OUTSIDE_P2, "w_a_sq_double_bound": UNRESOLVED,
                  "component_iv": "not evaluated: w^ not finite"})
        _check_digest(inp, digest_start)
        return E2Measurement(v, prov)
    premises, reasons = p3_premise(inp, w_hat)
    v["premises"], v["premise_reasons"] = premises, reasons
    w_ex = field_exact(inp)
    e = [[Fraction(float(w_hat[i, j])) - w_ex[i][j] for j in range(inp.ny - 1)] for i in range(inp.nx - 1)]
    Xa = [[abs(Fraction(float(x))) for x in row] for row in inp.X]
    Ya = [[abs(Fraction(float(x))) for x in row] for row in inp.Y]
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
    vok5 = ea2 <= eps_a2
    v.update({"max_abs_e_up": hs._float_up(max_e), "e_is_zero": max_e == 0,
              "max_rel_e_up": hs._float_up(max_e / max_w) if max_w else UNRESOLVED,
              # exact comparisons (OBSERVATION); normative only under the premises
              "vok4_violations": viol, "vok4_max_ratio_up": hs._float_up(worst),
              "vok1_upper_holds": ea2 <= c2 * eh2,  # ||e||_a <= D^(1/2)(4/h_x^2 + 4/h_y^2)||e||_h (VOK-1, item 4), exact algebra
              "vok5_holds": vok5,
              "vok4_status": ("HOLDS" if viol == 0 else "VIOLATED") if premises == P3_CONFIRMED else premises,
              "vok5_status": ("HOLDS" if vok5 else "VIOLATED") if premises == P3_CONFIRMED else premises})
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
    v.update(_component_iv(inp, w_hat, wa2))
    v["w_a_sq_double_bound"] = UNRESOLVED  # VOK-3 is a MODEL (c_R not counted): no rigorous bound for the double paths
    _check_digest(inp, digest_start)
    return E2Measurement(v, prov)


def _component_iv(inp: FieldInputs, w_hat: np.ndarray, wa2: Fraction) -> dict:
    """Component (iv), OBSERVATION: ||w^||_a^2 in double by two QA paths and its exact relative error. A grid step that is 0 or not
    finite in double, an overflow or any failure of a path gives UNRESOLVED for that path; the exact measurement (i) is kept."""
    out = {}
    hx, hy = inp.lx / inp.nx, inp.ly / inp.ny
    if not (math.isfinite(hx) and math.isfinite(hy) and hx > 0 and hy > 0):
        for path in ("definition", "hessian"):
            out[f"w_a_sq_double_{path}"] = UNRESOLVED
            out[f"w_a_sq_double_{path}_rel_error"] = UNRESOLVED
        out["component_iv"] = "not evaluated: grid step in double is 0 or not finite"
        return out
    for path, fn in (("definition", energy_sq_double_definition), ("hessian", energy_sq_double_hessian)):
        try:
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                val = fn(inp, w_hat)
        except (ArithmeticError, ValueError) as exc:
            out[f"w_a_sq_double_{path}"] = UNRESOLVED
            out[f"w_a_sq_double_{path}_rel_error"] = f"UNRESOLVED ({type(exc).__name__})"
            continue
        if not math.isfinite(val):
            out[f"w_a_sq_double_{path}"] = repr(val)  # 'inf' / 'nan' as an OBSERVATION
            out[f"w_a_sq_double_{path}_rel_error"] = UNRESOLVED
            continue
        out[f"w_a_sq_double_{path}"] = val
        out[f"w_a_sq_double_{path}_rel_error"] = float(abs(Fraction(val) - wa2) / wa2) if wa2 else \
            ("0 (exact)" if val == 0 else UNRESOLVED)
    return out


def _check_digest(inp: FieldInputs, digest_start: str) -> None:
    """N-2: the inputs must be unchanged between the start and the end of ``measure``."""
    if inp.digest() != digest_start:
        raise RuntimeError("inputs changed during the measurement (sha256 differs)")
