"""Certified reference of DST-I for R5-V CCCC (G-3 / OK-7; QA research tool, D-049.6 item 5). No import of ``kkvim``; no use of
math.sin, math.pi, numpy or scipy as a source of reference values (they appear only as the path under test and in diagnostics).

Transform (accepted, r5_cccc_bc.dst = scipy.fft.dst(type=1, norm="ortho")): Psi_ik = sqrt(2/n) sin(pi i k/n), i, k = 1..n-1.

Every reference quantity is an interval [lo, hi] of integers at the scale 2^-W (W = ``bits``) that CONTAINS the exact value. Proof
of containment, step by step (own derivation; classical facts named):
(P1) Alternating series. If t_0 >= t_1 >= ... >= 0 and t_k -> 0, the partial sums S_K = sum_{k<=K} (-1)^k t_k satisfy
     S_1 <= S_3 <= ... <= sum <= ... <= S_2 <= S_0, so sum is in [S_odd, S_even] for any odd/even indices. It suffices that the terms
     decrease from the first index used in the tail.
(P2) pi. Machin's identity pi = 16 arctan(1/5) - 4 arctan(1/239) (classical, exact). arctan(1/x) = sum (-1)^k/((2k+1) x^(2k+1)), x >= 2:
     terms decrease strictly, so (P1) brackets it. Each term is bounded by integers floor/ceil at the scale 2^-(W+G) (G guard bits);
     the lower bound of a partial sum uses the lower bounds of its positive terms and the upper bounds of its negative terms, and
     conversely. pi_lo = 16 atan5_lo - 4 atan239_hi, pi_hi = 16 atan5_hi - 4 atan239_lo; then floor/ceil to the scale 2^-W.
     Second, independent route (tests only): Euler's pi/4 = arctan(1/2) + arctan(1/3).
(P3) Exact angle reduction. sin(pi p/q) with integers p, q > 0: j = p mod 2q (period 2 pi); if j >= q: sign -1, j <- j - q
     (sin(t + pi) = -sin t); if 2j > q: j <- q - j (sin(pi - t) = sin t). Now 0 <= j/q <= 1/2 EXACTLY (integer arithmetic). j = 0 gives
     0 exactly, 2j = q gives 1 exactly. Otherwise t = pi j/q lies in [pi_lo j/q, pi_hi j/q] (j/q >= 0), rounded outward.
(P4) sin on [a, b], 0 <= a <= b <= 2 (< pi). Taylor sin x = sum (-1)^k x^(2k+1)/(2k+1)!: the ratio of consecutive terms is
     x^2/((2k+2)(2k+3)) <= 4/6 < 1, so (P1) brackets sin x at each endpoint; term bounds are directed integers as in (P2). sin is
     concave and >= 0 on [0, pi], so min over [a, b] >= max(0, min(sin a, sin b)); it is increasing on [0, pi/2], so max = sin b when
     b <= pi_lo/2 (proved by the integer comparison 2 b_hi <= pi_lo), otherwise max <= 1. Second route (tests only): sin t =
     cos(pi/2 - t), cos decreasing on [0, pi/2], Taylor of cos bracketed by (P1) (terms decrease from k = 1 for |x| <= 2).
(P5) sqrt(2/n) = sqrt(2n)/n: r = isqrt(2n 4^W) gives r <= sqrt(2n) 2^W < r + 1 (exact if r^2 = 2n 4^W); divided by n, floor/ceil.
(P6) Psi_ik = sign * c * s with c, s >= 0 in their intervals: the product of the lower bounds and of the upper bounds, floor/ceil to
     2^-W, encloses c s; the sign is exact (P3).
(P7) DST of exact binary64 inputs x_k = X_k 2^E (X_k integers, exact): y_i = sum_k Psi_ik x_k lies in [sum_k min(L_ik X_k, H_ik X_k),
     sum_k max(...)] 2^(E-W) (each term independently in its interval; the sign of X_k decides which bound enters) - exact integers.
(P8) Norms and ratios: squares of exact integers, square roots by isqrt with directed rounding, ratios [lo1/hi2, hi1/lo2]; a ratio
     whose denominator interval contains 0 is UNRESOLVED.
Numbers of terms only change the WIDTH, never the containment; the widths are recorded with every result. Cost limits give
UNRESOLVED_RESOURCES instead of a lower precision.
"""

import math
import time
from dataclasses import dataclass, field
from fractions import Fraction
from functools import lru_cache

import numpy as np

UNRESOLVED = "UNRESOLVED"
UNRESOLVED_RESOURCES = "UNRESOLVED_RESOURCES"
DEFAULT_BITS = 160
GUARD = 32
U = Fraction(1, 2**53)


# --- (P1), (P2): pi ---------------------------------------------------------------------------------------------------------

def _arctan_inv(x: int, scale_bits: int) -> tuple[int, int]:
    """[lo, hi] integers with lo <= arctan(1/x) 2^scale_bits <= hi, x >= 2 ((P1), (P2))."""
    one = 1 << scale_bits
    lo_sum = hi_sum = 0
    k, power = 0, x  # power = x^(2k+1)
    sums = []
    while True:
        d = (2 * k + 1) * power
        t_lo, t_hi = one // d, -(-one // d)  # floor, ceil of 2^s/((2k+1) x^(2k+1))
        if k % 2 == 0:
            lo_sum, hi_sum = lo_sum + t_lo, hi_sum + t_hi
        else:
            lo_sum, hi_sum = lo_sum - t_hi, hi_sum - t_lo
        sums.append((lo_sum, hi_sum))
        if t_hi <= 1 and k >= 1:
            break
        k += 1
        power *= x * x
    K = len(sums) - 1
    even = sums[K] if K % 2 == 0 else sums[K - 1]
    odd = sums[K] if K % 2 == 1 else sums[K - 1]
    return odd[0], even[1]  # S_odd (lower bound) <= arctan <= S_even (upper bound)


@lru_cache(maxsize=None)
def pi_bounds(bits: int = DEFAULT_BITS, route: str = "machin") -> tuple[int, int]:
    """[lo, hi] integers, lo <= pi 2^bits <= hi. route "machin" (reference) or "euler" (independent second route, tests)."""
    s = bits + GUARD
    if route == "machin":
        a_lo, a_hi = _arctan_inv(5, s)
        b_lo, b_hi = _arctan_inv(239, s)
        lo, hi = 16 * a_lo - 4 * b_hi, 16 * a_hi - 4 * b_lo
    elif route == "euler":
        a_lo, a_hi = _arctan_inv(2, s)
        b_lo, b_hi = _arctan_inv(3, s)
        lo, hi = 4 * (a_lo + b_lo), 4 * (a_hi + b_hi)
    else:
        raise ValueError("route must be 'machin' or 'euler'")
    return lo >> GUARD, -((-hi) >> GUARD)  # floor, ceil to the scale 2^bits


# --- (P4): sin and cos on intervals --------------------------------------------------------------------------------------------

def _series_at(A: int, bits: int, odd: bool) -> tuple[int, int]:
    """[lo, hi] for sin(x) (odd = True) or cos(x) (odd = False) at x = A 2^-bits, 0 <= x <= 2, by (P1) with directed term bounds."""
    one = 1 << bits
    if A == 0:
        return (0, 0) if odd else (one, one)
    A2, den2 = A * A, 1 << (2 * bits)
    t_lo = t_hi = A if odd else one  # term 0 at the scale 2^bits
    k = 0
    lo_s, hi_s = t_lo, t_hi
    sums = [(lo_s, hi_s)]
    while True:
        n1, n2 = (2 * k + 2, 2 * k + 3) if odd else (2 * k + 1, 2 * k + 2)
        d = den2 * n1 * n2
        t_lo, t_hi = (t_lo * A2) // d, -(-(t_hi * A2) // d)
        k += 1
        if k % 2 == 1:
            lo_s, hi_s = lo_s - t_hi, hi_s - t_lo
        else:
            lo_s, hi_s = lo_s + t_lo, hi_s + t_hi
        sums.append((lo_s, hi_s))
        if t_hi <= 1 and k >= 2:
            break
    K = len(sums) - 1
    even = sums[K] if K % 2 == 0 else sums[K - 1]
    odd_s = sums[K] if K % 2 == 1 else sums[K - 1]
    # terms decrease from k = 0 (sin, x <= 2) or from k = 1 (cos, x <= 2): (P1) with K >= 1
    return odd_s[0], even[1]


def _sin_interval(a_lo: int, b_hi: int, pi_lo: int, bits: int) -> tuple[int, int]:
    """sin on [a, b] (integers at 2^-bits, 0 <= a <= b <= 2): (P4)."""
    if not (0 <= a_lo <= b_hi <= 2 << bits):
        raise ValueError("sin interval outside [0, 2]")
    sa, sb = _series_at(a_lo, bits, True), _series_at(b_hi, bits, True)
    lo = max(0, min(sa[0], sb[0]))
    hi = sb[1] if 2 * b_hi <= pi_lo else (1 << bits)
    return lo, min(hi, 1 << bits)


def _reduce(p: int, q: int) -> tuple[int, int]:
    """(P3): sin(pi p/q) = sign * sin(pi j/q), 0 <= j/q <= 1/2."""
    if q <= 0:
        raise ValueError("q must be positive")
    j, sign = p % (2 * q), 1
    if j >= q:
        sign, j = -1, j - q
    if 2 * j > q:
        j = q - j
    return sign, j


def sin_pi_rational(p: int, q: int, bits: int = DEFAULT_BITS, route: str = "A") -> tuple[int, int, int]:
    """(sign, lo, hi): sin(pi p/q) = sign * s with lo <= s 2^bits <= hi, s >= 0. route "A" (reference: Machin pi, sin Taylor) or
    "B" (independent second route for tests: Euler pi, cos Taylor of the complementary angle)."""
    sign, j = _reduce(p, q)
    one = 1 << bits
    if j == 0:
        return 1, 0, 0
    if 2 * j == q:
        return sign, one, one
    if route == "A":
        plo, phi = pi_bounds(bits, "machin")
        a, b = (plo * j) // q, -(-(phi * j) // q)
        lo, hi = _sin_interval(a, b, plo, bits)
    elif route == "B":
        plo, phi = pi_bounds(bits, "euler")
        r = q - 2 * j  # complementary angle pi r/(2q) in (0, pi/2)
        a, b = (plo * r) // (2 * q), -(-(phi * r) // (2 * q))
        if 2 * b > plo:
            raise ValueError("complementary angle not proved below pi/2")
        lo = max(0, _series_at(b, bits, False)[0])  # cos decreasing on [0, pi/2]
        hi = min(one, _series_at(a, bits, False)[1])
    else:
        raise ValueError("route must be 'A' or 'B'")
    return sign, lo, hi


# --- (P5), (P6): Psi ---------------------------------------------------------------------------------------------------------------

def sqrt2n_bounds(n: int, bits: int = DEFAULT_BITS) -> tuple[int, int]:
    """[lo, hi] integers, lo <= sqrt(2/n) 2^bits <= hi (P5)."""
    r = math.isqrt(2 * n * (1 << (2 * bits)))
    exact = r * r == 2 * n * (1 << (2 * bits))
    return r // n, -(-(r if exact else r + 1) // n)


@dataclass(frozen=True)
class PsiTable:
    """Certified |Psi| values per reduced index j (Psi_ik depends on i k only): Psi_ik = sign(ik) * [lo_j, hi_j] 2^-bits."""

    n: int
    bits: int
    mag: dict  # j -> (lo, hi)
    seconds: float
    sqrt_bounds: tuple[int, int]
    sin_width_max: Fraction  # largest width of a sin interval (argument error from pi + Taylor truncation + outward rounding)

    def entry(self, i: int, k: int) -> tuple[int, int]:
        sign, j = _reduce(i * k, self.n)
        lo, hi = self.mag[j]
        return (lo, hi) if sign > 0 else (-hi, -lo)

    def entry_fraction(self, i: int, k: int) -> tuple[Fraction, Fraction]:
        lo, hi = self.entry(i, k)
        return Fraction(lo, 1 << self.bits), Fraction(hi, 1 << self.bits)


@lru_cache(maxsize=32)
def psi_table(n: int, bits: int = DEFAULT_BITS) -> PsiTable:
    """Certified magnitudes of all distinct Psi values of size n (only j = 0..n/2 are computed and reused)."""
    if not isinstance(n, int) or isinstance(n, bool) or n < 2:
        raise ValueError("n must be an integer >= 2")
    t0 = time.perf_counter()
    c_lo, c_hi = sqrt2n_bounds(n, bits)
    mag, wmax = {}, Fraction(0)
    for j in range(0, n // 2 + 1):
        _, s_lo, s_hi = sin_pi_rational(j, n, bits, "A")
        wmax = max(wmax, Fraction(s_hi - s_lo, 1 << bits))
        mag[j] = ((c_lo * s_lo) >> bits, -((-(c_hi * s_hi)) >> bits))
    return PsiTable(n, bits, mag, time.perf_counter() - t0, (c_lo, c_hi), wmax)


# --- (P7): certified DST of binary64 vectors -------------------------------------------------------------------------------------

def _exact_scaled(x: np.ndarray) -> tuple[list[int], int]:
    """x_k = X_k 2^E exactly (binary64 inputs)."""
    ratios = [float(v).as_integer_ratio() for v in x]
    E = min((-(q.bit_length() - 1) for p, q in ratios if p), default=0)
    return [p * (1 << (-(q.bit_length() - 1) - E)) for p, q in ratios], E


def _check_input(x) -> np.ndarray:
    a = np.asarray(x)
    if a.dtype != np.float64:
        raise ValueError(f"x must be a binary64 (float64) array, got dtype {a.dtype}")
    if a.ndim != 1 or a.size < 1:
        raise ValueError("x must be a non-empty 1-D array of length n - 1")
    if not np.all(np.isfinite(a)):
        raise ValueError("x must be finite")
    return a


@dataclass
class CertifiedDST:
    status: str
    n: int
    bits: int
    lo: list[int] = field(default_factory=list)  # y_i in [lo_i, hi_i] 2^exp
    hi: list[int] = field(default_factory=list)
    exp: int = 0
    seconds: float = 0.0
    note: str = ""

    def interval(self, i: int) -> tuple[Fraction, Fraction]:
        s = Fraction(2) ** self.exp
        return self.lo[i] * s, self.hi[i] * s


def certified_dst(x, bits: int = DEFAULT_BITS, max_entries: int = 1100**2, time_limit: float | None = None) -> CertifiedDST:
    """Intervals containing y = Psi x EXACTLY for the given binary64 vector x of length n - 1 ((P1)...(P7)). Over ``max_entries`` =
    (n - 1)^2 or ``time_limit`` seconds: UNRESOLVED_RESOURCES (no lower precision is used)."""
    a = _check_input(x)
    m = a.size
    n = m + 1
    if m * m > max_entries:
        return CertifiedDST(UNRESOLVED_RESOURCES, n, bits, note=f"(n - 1)^2 = {m * m} > max_entries = {max_entries}")
    t0 = time.perf_counter()
    tab = psi_table(n, bits)
    X, E = _exact_scaled(a)
    lo, hi = [], []
    for i in range(1, n):
        if time_limit is not None and time.perf_counter() - t0 > time_limit:
            return CertifiedDST(UNRESOLVED_RESOURCES, n, bits, note=f"time limit {time_limit} s exceeded at row {i}")
        sl = sh = 0
        for k in range(1, n):
            Xk = X[k - 1]
            if Xk == 0:
                continue
            L, H = tab.entry(i, k)
            if Xk > 0:
                sl += L * Xk
                sh += H * Xk
            else:
                sl += H * Xk
                sh += L * Xk
        lo.append(sl)
        hi.append(sh)
    return CertifiedDST("CERTIFIED", n, bits, lo, hi, E - bits, time.perf_counter() - t0)


# --- (P8): comparison of the tested path with the reference ------------------------------------------------------------------------

def _isqrt_bounds(v: int) -> tuple[int, int]:
    r = math.isqrt(v)
    return r, r if r * r == v else r + 1


def _sqrt_frac(x: Fraction, bits: int = 64) -> tuple[Fraction, Fraction]:
    if x == 0:
        return Fraction(0), Fraction(0)
    P, Q = x.numerator, x.denominator
    r = math.isqrt(P * Q * 4**bits)
    return Fraction(r, Q * 2**bits), Fraction(r + 1, Q * 2**bits)


def _down(x: Fraction) -> float:
    f = float(x)
    while Fraction(f) > x:
        f = math.nextafter(f, -math.inf)
    return f


def _up(x: Fraction) -> float:
    try:
        f = float(x)
    except OverflowError:
        return math.inf
    while math.isfinite(f) and Fraction(f) < x:
        f = math.nextafter(f, math.inf)
    return f


def compare(y_hat, ref: CertifiedDST, x=None, budget: Fraction = Fraction(1, 100)) -> dict:
    """Error of a computed DST y_hat (binary64) against the certified reference: per-component error intervals, absolute and relative
    2-norm intervals, the reference width, the budget rule (reference width <= budget * lower bound of the measured error, D-049.6
    item 4 form) and, with x, the ratio err/(u log2(n) ||x||_2) of RO 3.2 (OBSERVATION). Zero or not separated quantities are UNRESOLVED.
    """
    if ref.status != "CERTIFIED":
        return {"status": ref.status, "note": ref.note}
    yh = np.asarray(y_hat, dtype=np.float64)
    m = len(ref.lo)
    if yh.shape != (m,) or not np.all(np.isfinite(yh)):
        raise ValueError("y_hat must be a finite float64 vector of the reference length")
    Y, Ey = _exact_scaled(yh)
    e = min(Ey, ref.exp)
    Ys = [v << (Ey - e) for v in Y]
    L = [v << (ref.exp - e) for v in ref.lo]
    H = [v << (ref.exp - e) for v in ref.hi]
    err_lo2 = err_hi2 = wid2 = ylo2 = yhi2 = 0
    for yv, lv, hv in zip(Ys, L, H):
        d1, d2 = yv - hv, yv - lv  # e_i in [d1, d2]
        a_hi = max(abs(d1), abs(d2))
        a_lo = 0 if d1 <= 0 <= d2 else min(abs(d1), abs(d2))
        err_lo2 += a_lo * a_lo
        err_hi2 += a_hi * a_hi
        wid2 += (hv - lv) ** 2
        m_lo = 0 if lv <= 0 <= hv else min(abs(lv), abs(hv))
        ylo2 += m_lo * m_lo
        yhi2 += max(abs(lv), abs(hv)) ** 2
    scale = Fraction(2) ** e
    el, eh = _isqrt_bounds(err_lo2)[0] * scale, _isqrt_bounds(err_hi2)[1] * scale
    wh = _isqrt_bounds(wid2)[1] * scale
    yl, yhh = _isqrt_bounds(ylo2)[0] * scale, _isqrt_bounds(yhi2)[1] * scale
    out = {"status": "CERTIFIED", "n": ref.n, "bits": ref.bits,
           "abs_err_2_down": _down(el), "abs_err_2_up": _up(eh), "ref_width_2_up": _up(wh),
           "y_norm_2_down": _down(yl), "y_norm_2_up": _up(yhh)}
    if yl > 0:
        out["rel_err_2_down"], out["rel_err_2_up"] = _down(el / yhh), _up(eh / yl)
        out["ref_width_rel_up"] = _up(wh / yl)
    else:
        out["rel_err_2_down"] = out["rel_err_2_up"] = out["ref_width_rel_up"] = \
            UNRESOLVED + " (||y|| not separated from 0)"
    if el == 0:
        out["resolution"] = UNRESOLVED + " (measured error not separated from 0 by the reference width; the upper bound holds)"
    elif wh <= budget * el:
        out["resolution"] = f"RESOLVED (reference width <= {budget} of the measured error)"
    else:
        out["resolution"] = UNRESOLVED + f" (reference width above {budget} of the measured error: increase bits)"
    if x is not None:
        xa = _check_input(x)
        Xs, Ex = _exact_scaled(xa)
        xn2 = sum(v * v for v in Xs)
        if xn2 == 0:
            out["c_dst_ratio_down"] = out["c_dst_ratio_up"] = UNRESOLVED + " (x = 0)"
        else:
            xl, xh = _isqrt_bounds(xn2)
            xs = Fraction(2) ** Ex
            lg_lo, lg_hi = _log2_bounds(ref.n)
            out["c_dst_ratio_down"] = _down(el / (U * lg_hi * xh * xs))
            out["c_dst_ratio_up"] = _up(eh / (U * lg_lo * xl * xs)) if lg_lo > 0 else UNRESOLVED
    return out


def _log2_bounds(n: int) -> tuple[Fraction, Fraction]:
    """Rational bounds of log2(n) from integer arithmetic: log2 n in [floor(log2(n^K))/K, (floor(log2(n^K)) + 1)/K], K = 2^12."""
    if n <= 1:
        return Fraction(0), Fraction(0)
    K = 1 << 12
    b = (n**K).bit_length()  # 2^(b-1) <= n^K < 2^b
    return Fraction(b - 1, K), Fraction(b, K)


# --- quantities for E-4 ------------------------------------------------------------------------------------------------------------

def boundary_psi(n: int, bits: int = DEFAULT_BITS) -> list[dict]:
    """psi_k(1) = Psi_{1k}, psi_k(m) = Psi_{m k} (m = n - 1) and the products of z_k = (psi_k(1), psi_k(m)) needed by the modal path
    of S (E-4): intervals as Fractions."""
    tab = psi_table(n, bits)
    m, out = n - 1, []
    for k in range(1, n):
        a, b = tab.entry_fraction(1, k), tab.entry_fraction(m, k)
        out.append({"k": k, "psi_1": a, "psi_m": b, "psi_1_sq": _imul(a, a), "psi_1_psi_m": _imul(a, b), "psi_m_sq": _imul(b, b)})
    return out


def _imul(a, b):
    c = [x * y for x in a for y in b]
    return min(c), max(c)


def mu_bounds(n: int, l: float, bits: int = DEFAULT_BITS) -> list[tuple[Fraction, Fraction]]:
    """mu_k = (4/h^2) sin^2(k pi/(2n)), h = l/n EXACTLY from the double l (R4 3.2): certified intervals, k = 1..n-1."""
    h2 = (Fraction(l) / n) ** 2
    out = []
    for k in range(1, n):
        _, lo, hi = sin_pi_rational(k, 2 * n, bits)
        s = Fraction(1, 1 << bits)
        out.append((4 / h2 * (lo * s) ** 2, 4 / h2 * (hi * s) ** 2))
    return out


def sin_of_rational(x: Fraction, bits: int = DEFAULT_BITS) -> tuple[Fraction, Fraction]:
    """sin(x) for a rational 0 <= x <= pi_lo (diagnostics of double arguments): (P4) directly or via sin(pi - x)."""
    plo, phi = pi_bounds(bits)
    one = 1 << bits
    X_lo, X_hi = math.floor(x * one), math.ceil(x * one)
    if X_lo < 0 or X_hi > plo:
        raise ValueError("argument outside [0, pi_lo]")
    if 2 * X_hi <= plo:
        lo, hi = _sin_interval(X_lo, X_hi, plo, bits)
    else:  # pi - x in [pi_lo - X_hi, pi_hi - X_lo]
        a, b = max(0, plo - X_hi), phi - X_lo
        if b > 2 << bits:
            raise ValueError("reduced argument above 2")
        lo, hi = _sin_interval(a, b, plo, bits)
    return Fraction(lo, one), Fraction(hi, one)


def double_boundary_diagnostics(n: int, bits: int = DEFAULT_BITS) -> list[dict]:
    """Error of the double formula of psi_k(1) used by r5_cccc_bc.solve_bc (math.sqrt(2.0/n) * np.sin(k * math.pi / n)) split EXACTLY
    into: normalisation factor (c^ - c) sin t; argument c^ (sin t^ - sin t); sin evaluation c^ (s^ - sin t^); product rounding.
    t = pi k/n, t^ = fl(fl(k pi^)/n) (the double argument), c^ = fl(sqrt(fl(2/n))). Intervals (Fractions); diagnostics only."""
    k = np.arange(1, n)
    arg_d = k * math.pi / n
    c_d = math.sqrt(2.0 / n)
    s_d = np.sin(arg_d)
    psi_d = c_d * s_d
    tab = psi_table(n, bits)
    plo, phi = pi_bounds(bits)
    one = Fraction(1, 1 << bits)
    c_lo, c_hi = (Fraction(v, 1 << bits) for v in tab.sqrt_bounds)
    C = Fraction(c_d)
    out = []
    for idx, kk in enumerate(k):
        kk = int(kk)
        sign, slo, shi = sin_pi_rational(kk, n, bits)
        st = (sign * slo * one, sign * shi * one) if sign > 0 else (sign * shi * one, sign * slo * one)
        ta = Fraction(float(arg_d[idx]))
        sa = sin_of_rational(ta, bits)
        t_true = (Fraction(plo * kk, n) * one, Fraction(phi * kk, n) * one)
        f_norm = _imul((C - c_hi, C - c_lo), st)
        f_arg = (C * (sa[0] - st[1]), C * (sa[1] - st[0]))
        f_sin = (C * (Fraction(float(s_d[idx])) - sa[1]), C * (Fraction(float(s_d[idx])) - sa[0]))
        f_prod = Fraction(float(psi_d[idx])) - C * Fraction(float(s_d[idx]))
        out.append({"k": kk, "psi_double": float(psi_d[idx]), "arg_error": (ta - t_true[1], ta - t_true[0]),
                    "norm_part": f_norm, "arg_part": f_arg, "sin_part": f_sin, "product_part": f_prod,
                    "total": (Fraction(float(psi_d[idx])) - tab.entry_fraction(1, kk)[1],
                              Fraction(float(psi_d[idx])) - tab.entry_fraction(1, kk)[0])})
    return out


def orthogonality_check(n: int, bits: int = DEFAULT_BITS) -> dict:
    """Interval check of Psi Psi = I: every interval sum_k Psi_ik Psi_kj must contain delta_ij; reports the largest width."""
    tab = psi_table(n, bits)
    m = n - 1
    ok, wmax = True, Fraction(0)
    one2 = 1 << (2 * bits)
    for i in range(1, n):
        for j in range(i, n):
            lo = hi = 0
            for k in range(1, n):
                a, b = tab.entry(i, k), tab.entry(k, j)
                c = [x * y for x in a for y in b]
                lo, hi = lo + min(c), hi + max(c)
            target = one2 if i == j else 0
            ok &= lo <= target <= hi
            wmax = max(wmax, Fraction(hi - lo, one2))
    return {"n": n, "contains_identity": ok, "max_width": wmax, "size": m}
