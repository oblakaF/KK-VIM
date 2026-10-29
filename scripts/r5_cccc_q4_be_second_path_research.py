"""R5-V Q4 (D-049, D-053, D-054, D-055): independent SECOND computational path of the energy bound B_E,s of theorem E-2.
QA research tool: not the D1 campaign, not a qualification, no normative tolerance, no K_E, no approved SP-BE sample (OD-11).

Mathematics (not new): ST = docs/R5_CCCC_Q4_ENERGY_ERROR_BOUND.study.md (E-1, L-0...L-8, E-2, 5.3, 5.4, E-3, sect. 13); the bound is
a derivation of the project (levels B and V, AGENTS.md 2), NOT a statement of Kirichenko-Krysko. Independence here means an
independent IMPLEMENTATION of the same theorem: the statement of E-2 and the definitions of NS 2.2 are shared, by necessity.

Object bounded: the QA double path G-4 of src/kkvim_reference/r5_cccc_energy.py (pi_double, q_double; ST 1.2). Its values P^_p, P^_s,
Q^ are inputs; ``bind_double_path`` is the only place where that module is called (to bind the replication to it).

Implementation of THIS path (differs from QA-EB scripts/r5_cccc_q4_energy_bound_premises_audit.py and from r5_cccc_energy in the
arithmetic and the assembly; QA-EB is never called by the B_E functions, only by ``primary_qaeb`` for the comparison):
- exact rationals of binary64 by math.frexp (``exact``), not Fraction(float) and not the bit patterns of QA-EB;
- Delta_h^C as an explicit sparse matrix "all nodes x interior nodes" built row by row from the definition (``Mesh``), applied in
  Fraction arithmetic; |Delta|^C the same rows with the moduli of each part's coefficients (RO VOK-3);
- the D2 residual of B_omega through A = Delta^T W Delta/(h_x h_y) (the transpose of the same matrix: a_h(v, w) = D h_x h_y v^T A w,
  ST 3, Prop. D), not the 13-point operator of QA-EB and not r5_cccc_d2;
- eta, psi, psi_L by enumeration of the corners |delta_i| = u AND by the closed form of ST 4; the two must agree exactly;
- L, L_Q by simulating the pairwise tree of r5_cccc_energy AND by ceil(log2 M); must agree;
- square roots by an own directed rule: sqrt(P/Q) in [isqrt(P 4^k)/ceil(sqrt(Q 4^k)), ceil(sqrt(P 4^k))/isqrt(Q 4^k)], each pair
  certified by squaring (lo^2 <= x <= hi^2); numerators use the upper root, the denominator D^(1/2) mu_low the lower (ST 5.4);
- rounding to binary64 by an own integer rule (``round_up``, ``round_down``), checked against the neighbouring doubles;
- the exact defect E_s = (P^_p - P^_s - Q^) - f(w~_p, w~_s), f(v, w) = Pi(v) - Pi(w) - ||v - w||_a^2/2 (ST 1.1; RO VOK-2:
  f(w~_p, w~_s) = -gamma_s), from the exact product fields; no gamma tool is used. On request the parts E^(1)...E^(4) of E-1 with
  a_h(w~_s - omega, x) = a_h(w~_s, x) - l_h(x) (no omega needed) and the identity E^(1) + ... + E^(4) = E_s, exactly.

B_E,s (ST 5.1) = B1 + B2 + B3 + B4:
  B1 = eps_s (||d^||_a + beta) + B_omega (eps_p + eps_s)                       formation (L-8), second order inside B_omega
  B2 = B_P(w^_p) + B_P(w^_s), B_P = kappa_P S_q + kappa_L S_l + (1 + gamma_L) nu S_2   energies (L-3, L-5)
  B3 = beta ||d^||_a + beta^2/2                                               subtraction d^ (L-7)
  B4 = kappa_Q S_q(d^) + (1 + gamma_LQ) nu S_2(d^)                             Q^ (L-3, L-5)
  beta = u D^(1/2) Lambda ||d^||_h; eps = gamma_N D^(1/2) Lambda ||S||_h, S = |X||Y|^T (VOK-5); B_omega = ||D A w^_s - q 1||_h /
  (D^(1/2) mu_low) + eps_s (VOK-7 (b)); Lambda = 4/h_x^2 + 4/h_y^2, mu_low = 4/l_x^2 + 4/l_y^2. The only norm conversion is the
  one of VOK-7 (b) and L-7 (||.||_a >= D^(1/2) mu_low ||.||_h), stated in ST.

Premises of E-2 (ST 2, 13), checked here by own code; the certificate (status CERTIFIED) needs every one ESTABLISHED (P-1 is a
platform premise: ASSUMED with its probes recorded; a failed probe makes it NOT_ESTABLISHED):
  P-1 round to nearest (probes, this thread); P-3'(a) no FTZ/DAZ (probes, this thread); U-1 (replaces P-4' and P-3'(c) for the
  products) exactly on both fields; L-2 for d^ exactly; P-3'(c), P-2 and I-1 by an own operation-by-operation replication of G-4
  (every x and / with nonzero operands: |result| > 2^-1022 strictly; + and -: zero only for exact cancellation; all finite) that
  must be bitwise the module's values and the given P^, Q^; I-2 (h_d = fl(l/n), c = h_d**2 in this process, exact delta_h,
  delta_c, 1 + delta_c > 0); P-5 (k u < 1); D/2, h/2 normal; D^(1/2) mu_low > 0; B_E_up finite.
A premise not established, a domain error or a non-finite input: status UNRESOLVED with structured reasons (component, premise,
reason); B_E is then never offered as a certificate. |E_s| <= B_E and B_E <= K* R_Sigma are evaluated and reported; with the
premises established a violation is a CONTRADICTION (never hidden).

Usage (repository root; small non-D1 configurations; heavy output under scratch/):
    uv run python scripts/r5_cccc_q4_be_second_path_research.py run --profile synthetic --out scratch/be2
    uv run python scripts/r5_cccc_q4_be_second_path_research.py run --profile q3 --arrays <dir> --max-nodes 1024 --out scratch/be2q3
    uv run python scripts/r5_cccc_q4_be_second_path_research.py run --profile resources --out scratch/be2res
"""

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import platform
import struct
import sys
import threading
import time
import tracemalloc
from fractions import Fraction
from pathlib import Path

THREAD_VARS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
for _k in THREAD_VARS:
    os.environ[_k] = "1"  # one BLAS thread, as in Q3, EQ, QA-EB (effective only if NumPy is not loaded yet; recorded)

import numpy as np  # noqa: E402

from kkvim_reference import r5_cccc_energy as en  # noqa: E402  (object under test only: bind_double_path)

ROOT = Path(__file__).resolve().parents[1]
SELF = "scripts/r5_cccc_q4_be_second_path_research.py"
QAEB_TOOL = "scripts/r5_cccc_q4_energy_bound_premises_audit.py"
U = Fraction(1, 2**53)
TINY = Fraction(1, 2**1022)
C_R = 22  # operation count of the QA path G-4 in R_Sigma (ST 6.1, EN 16-28); not a normative c_R
ROOT_BITS = 96  # precision of the own directed roots (relative width ~ 2^-95); certified by squaring, not a tolerance
QAEB_ROOT_BITS = 80  # documented root rule of QA-EB (its docstring item 8): isqrt(P Q 4^80)/(Q 2^80), upper +1 unless exact
ESTABLISHED, NOT_ESTABLISHED, NOT_CHECKABLE, ASSUMED = "ESTABLISHED", "NOT_ESTABLISHED", "NOT_CHECKABLE", "ASSUMED"
CERTIFIED, UNRESOLVED = "CERTIFIED", "UNRESOLVED"
DATA_STATUS = "research observation of the second path of B_E (QA); NOT D1 data, never D1_RESULT; SP-BE not approved (OD-11)"


class SecondPathError(Exception):
    """A domain or input error of one component: the row is UNRESOLVED with (component, reason), never certified."""

    def __init__(self, component: str, reason: str):
        super().__init__(f"{component}: {reason}")
        self.component, self.reason = component, reason


# ---------------------------------------------------------------------------------------------------------------------------------
# exact rationals, directed roots, directed rounding (own implementations)
# ---------------------------------------------------------------------------------------------------------------------------------

def exact(x) -> Fraction:
    """The exact rational of a binary64 (math.frexp; subnormals included; -0.0 -> 0). Not finite: SecondPathError."""
    x = float(x)
    if not math.isfinite(x):
        raise SecondPathError("input", f"not finite binary64 {x!r}")
    if x == 0.0:
        return Fraction(0)
    m, e = math.frexp(x)  # x = m 2^e, 1/2 <= |m| < 1, m has at most 53 significant bits
    n = int(m * 9007199254740992.0)  # m 2^53: exact integer
    k = e - 53
    return Fraction(n << k) if k >= 0 else Fraction(n, 1 << -k)


def exact_list(a) -> list[Fraction]:
    return [exact(v) for v in np.asarray(a, dtype=float).ravel()]


def _isqrt_up(n: int) -> int:
    r = math.isqrt(n)
    return r if r * r == n else r + 1


def sqrt_bounds(x: Fraction, bits: int = ROOT_BITS) -> tuple[Fraction, Fraction]:
    """lo <= sqrt(x) <= hi, x >= 0: sqrt(P/Q) = sqrt(P 4^k)/sqrt(Q 4^k) with floor / ceiling of both integer roots; certified by
    squaring. Negative radicand: SecondPathError (an error, not a value; ST 5.4 item 2)."""
    if x < 0:
        raise SecondPathError("root", f"negative radicand {float(x)!r}")
    if x == 0:
        return Fraction(0), Fraction(0)
    P, Q = x.numerator, x.denominator
    a, b = P << (2 * bits), Q << (2 * bits)
    lo, hi = Fraction(math.isqrt(a), _isqrt_up(b)), Fraction(_isqrt_up(a), math.isqrt(b))
    if not (lo * lo <= x <= hi * hi):  # self-certification of the directed root
        raise SecondPathError("root", "directed root not certified by squaring")
    return lo, hi


def sqrt_bounds_qaeb(x: Fraction, bits: int = QAEB_ROOT_BITS) -> tuple[Fraction, Fraction]:
    """The documented representation rule of QA-EB (comparison only): r = isqrt(P Q 4^bits), lo = r/(Q 2^bits), hi = lo if r^2 is
    exact else (r + 1)/(Q 2^bits). Implemented here from the rule, not imported."""
    if x < 0:
        raise SecondPathError("root", f"negative radicand {float(x)!r}")
    if x == 0:
        return Fraction(0), Fraction(0)
    P, Q = x.numerator, x.denominator
    n = (P * Q) << (2 * bits)
    r = math.isqrt(n)
    lo = Fraction(r, Q << bits)
    return lo, (lo if r * r == n else Fraction(r + 1, Q << bits))


def _round(x: Fraction, upward: bool) -> float:
    """The nearest binary64 in the given direction (overflow: +-inf upward / +-DBL_MAX downward for positive x)."""
    if x == 0:
        return 0.0
    if x < 0:
        return -_round(-x, not upward)
    p, q = x.numerator, x.denominator
    e = p.bit_length() - q.bit_length()  # 2^(e-1) < x < 2^(e+1)
    if (p << max(-e, 0)) < (q << max(e, 0)):  # x < 2^e
        e -= 1
    if e > 1023:
        return math.inf if upward else 1.7976931348623157e308
    qe = max(e - 52, -1074)
    num, den = (p << -qe, q) if qe < 0 else (p, q << qe)
    m, r = divmod(num, den)
    if r and upward:
        m += 1
    if qe + m.bit_length() - 1 > 1023:
        return math.inf if upward else 1.7976931348623157e308
    return math.ldexp(m, qe)


def _certify(x: Fraction, v: float, upward: bool) -> float:
    """v is on the right side of x and its neighbour towards x is on the wrong side (tightest), checked exactly."""
    if not math.isfinite(v):
        return v
    side = Fraction(v) >= x if upward else Fraction(v) <= x
    nb = math.nextafter(v, -math.inf if upward else math.inf)
    tight = not math.isfinite(nb) or (Fraction(nb) < x if upward else Fraction(nb) > x)
    if not (side and tight):
        raise SecondPathError("rounding", "directed rounding to binary64 not certified")
    return v


def round_up(x: Fraction) -> float:
    return _certify(x, _round(x, True), True)


def round_down(x: Fraction) -> float:
    return _certify(x, _round(x, False), False)


def ftext(x: Fraction | None) -> str | None:
    return None if x is None else f"{x.numerator}/{x.denominator}"


def fsha(x: Fraction | None) -> str | None:
    return None if x is None else hashlib.sha256(ftext(x).encode()).hexdigest()


def gam(k: int) -> Fraction:
    if k * U >= 1:
        raise SecondPathError("P-5", f"k u >= 1 for k = {k}")
    return k * U / (1 - k * U)


# ---------------------------------------------------------------------------------------------------------------------------------
# the mesh and the sparse operator Delta_h^C (NS 2.2), built from the definition
# ---------------------------------------------------------------------------------------------------------------------------------

class Mesh:
    """Grid G(l_x, n_x) x G(l_y, n_y), exact h = l/n of the binary64 lengths, D, q. Interior field index k = (i - 1)(n_y - 1) +
    (j - 1), i = x index (rows of the arrays); node index (I, J) -> I (n_y + 1) + J."""

    def __init__(self, lx: float, nx: int, ly: float, ny: int, D: float = 1.0, q: float = 1.0):
        for name, v in (("nx", nx), ("ny", ny)):
            if isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v < 3:
                raise SecondPathError("mesh", f"{name} must be an integer >= 3, got {v!r}")
        self.lx, self.ly, self.D_f, self.q_f = float(lx), float(ly), float(D), float(q)
        self.nx, self.ny = int(nx), int(ny)
        self.Lx, self.Ly, self.D, self.q = exact(lx), exact(ly), exact(D), exact(q)
        if self.Lx <= 0 or self.Ly <= 0:
            raise SecondPathError("mesh", "plate sides must be > 0")
        self.hx, self.hy = self.Lx / self.nx, self.Ly / self.ny
        self.cx, self.cy = 1 / self.hx**2, 1 / self.hy**2
        self.Lam = 4 * self.cx + 4 * self.cy
        self.mu_low = 4 / self.Lx**2 + 4 / self.Ly**2
        self.n_int = (self.nx - 1) * (self.ny - 1)
        self.n_nodes = (self.nx + 1) * (self.ny + 1)
        wx = [self.hx / 2] + [self.hx] * (self.nx - 1) + [self.hx / 2]
        wy = [self.hy / 2] + [self.hy] * (self.ny - 1) + [self.hy / 2]
        self.W = [wx[I] * wy[J] for I in range(self.nx + 1) for J in range(self.ny + 1)]
        self.rows, self.abs_rows = self._build()

    def col(self, i: int, j: int) -> int:
        return (i - 1) * (self.ny - 1) + (j - 1)

    def _part(self, I: int, n: int, c: Fraction) -> list[tuple[int, Fraction]]:
        """(interior position, coefficient) of hat L along one direction at node I of 0..n (even ghost node at the ends)."""
        if I == 0:
            return [(1, 2 * c)]
        if I == n:
            return [(n - 1, 2 * c)]
        out = [(I, -2 * c)]
        if I - 1 >= 1:
            out.append((I - 1, c))
        if I + 1 <= n - 1:
            out.append((I + 1, c))
        return out

    def _build(self):
        rows, abs_rows = [], []
        for I in range(self.nx + 1):
            for J in range(self.ny + 1):
                r, a = {}, {}
                if 1 <= J <= self.ny - 1:  # hat L_x (x) E_y
                    for i, c in self._part(I, self.nx, self.cx):
                        k = self.col(i, J)
                        r[k] = r.get(k, 0) + c
                        a[k] = a.get(k, 0) + abs(c)
                if 1 <= I <= self.nx - 1:  # E_x (x) hat L_y
                    for j, c in self._part(J, self.ny, self.cy):
                        k = self.col(I, j)
                        r[k] = r.get(k, 0) + c
                        a[k] = a.get(k, 0) + abs(c)
                rows.append(sorted(r.items()))
                abs_rows.append(sorted(a.items()))
        return rows, abs_rows

    def apply(self, w: list[Fraction], absolute: bool = False) -> list[Fraction]:
        rows = self.abs_rows if absolute else self.rows
        v = [abs(x) for x in w] if absolute else w
        return [sum((c * v[k] for k, c in row), Fraction(0)) for row in rows]

    def apply_T(self, y: list[Fraction]) -> list[Fraction]:
        out = [Fraction(0)] * self.n_int
        for node, row in enumerate(self.rows):
            if y[node]:
                for k, c in row:
                    out[k] += c * y[node]
        return out


def field_quantities(m: Mesh, w: list[Fraction]) -> dict:
    """Exact S_q, S_l, S_2 (ST 4), ||w||_a^2 = D sum W t^2, ||w||_h^2, l_h(w), Pi_h^C(w) of one field."""
    t, A = m.apply(w), m.apply(w, absolute=True)
    W = m.W
    s_tt = sum((W[n] * t[n] * t[n] for n in range(m.n_nodes) if t[n]), Fraction(0))
    s_tA = sum((W[n] * abs(t[n]) * A[n] for n in range(m.n_nodes) if t[n]), Fraction(0))
    s_AA = sum((W[n] * A[n] * A[n] for n in range(m.n_nodes) if A[n]), Fraction(0))
    hh = m.hx * m.hy
    s_w = sum(w, Fraction(0))
    a2 = m.D * s_tt
    load = m.q * hh * s_w
    return {"a2": a2, "Sq": m.D / 2 * s_tA, "S2": m.D / 2 * s_AA, "Sl": abs(m.q) * hh * sum((abs(x) for x in w), Fraction(0)),
            "h2": hh * sum((x * x for x in w), Fraction(0)), "load": load, "Pi": a2 / 2 - load}


def bilinear(m: Mesh, v: list[Fraction], w: list[Fraction]) -> Fraction:
    tv = m.apply(v)
    tw = tv if w is v else m.apply(w)
    return m.D * sum((m.W[n] * tv[n] * tw[n] for n in range(m.n_nodes) if tv[n] and tw[n]), Fraction(0))


def load_of(m: Mesh, w: list[Fraction]) -> Fraction:
    return m.q * m.hx * m.hy * sum(w, Fraction(0))


def residual_h2(m: Mesh, w: list[Fraction]) -> Fraction:
    """||D A w - q 1||_h^2 with A = Delta^T W Delta/(h_x h_y) (transpose route), exactly."""
    t = m.apply(w)
    y = m.apply_T([m.W[n] * t[n] for n in range(m.n_nodes)])
    hh = m.hx * m.hy
    r = [m.D * v / hh - m.q for v in y]
    return hh * sum((x * x for x in r), Fraction(0))


# ---------------------------------------------------------------------------------------------------------------------------------
# formation (U-1, VOK-5) and the subtraction (L-2)
# ---------------------------------------------------------------------------------------------------------------------------------

def formation(m: Mesh, X, Y, W_hat) -> dict:
    """Exact product field X Y^T, S = |X||Y|^T and U-1: |W^ - X Y^T| <= gamma_N |X||Y|^T elementwise (ST 13 item 1)."""
    X, Y, W_hat = (np.asarray(a, dtype=float) for a in (X, Y, W_hat))
    if X.shape[0] != m.nx - 1 or Y.shape[0] != m.ny - 1 or W_hat.shape != (m.nx - 1, m.ny - 1) or X.shape[1] != Y.shape[1]:
        raise SecondPathError("formation", f"shapes {X.shape}, {Y.shape}, {W_hat.shape} do not fit the mesh")
    N = X.shape[1]
    gN = gam(N)
    Xe = [[exact(v) for v in row] for row in X]
    Ye = [[exact(v) for v in row] for row in Y]
    We = exact_list(W_hat)
    prod, S = [], []
    viol, worst = 0, Fraction(0)
    for i in range(m.nx - 1):
        for j in range(m.ny - 1):
            p = sum((Xe[i][k] * Ye[j][k] for k in range(N)), Fraction(0))
            s = sum((abs(Xe[i][k]) * abs(Ye[j][k]) for k in range(N)), Fraction(0))
            err = abs(We[i * (m.ny - 1) + j] - p)
            if err > gN * s:
                viol += 1
            if s:
                worst = max(worst, err / (gN * s))
            prod.append(p)
            S.append(s)
    return {"N": N, "exact": prod, "S": S, "computed": We, "holds": viol == 0, "violations": viol, "max_ratio": float(worst),
            "exact_zero": not any(prod), "S_zero": not any(S),
            "factor_zero": all(v == 0 for row in Xe for v in row) or all(v == 0 for row in Ye for v in row)}


def subtraction_check(wh_p, wh_s, dh) -> dict:
    """d^ is bitwise fl(w^_p - w^_s) here, and L-2: |d^ - (w^_p - w^_s)| <= u |d^| exactly, elementwise."""
    a, b, d = (np.asarray(x, dtype=float) for x in (wh_p, wh_s, dh))
    with np.errstate(all="ignore"):
        same = bool(np.array_equal((a - b).view(np.uint64), d.view(np.uint64)))
    bad = 0
    for x, y, z in zip(a.ravel(), b.ravel(), d.ravel()):
        if abs(exact(z) - (exact(x) - exact(y))) > U * abs(exact(z)):
            bad += 1
    return {"bitwise_fl_difference": same, "L2_holds": bad == 0, "L2_violations": bad}


# ---------------------------------------------------------------------------------------------------------------------------------
# grid constants (ST 4, 6.2): closed forms and an independent corner enumeration
# ---------------------------------------------------------------------------------------------------------------------------------

def _corners(base: Fraction, k: int, divisor: Fraction = Fraction(1)) -> Fraction:
    """max over |delta_i| = u, i = 1..k, of |base prod(1 + delta_i)/divisor - 1| (enumeration of the 2^k corners)."""
    best = Fraction(0)
    for mask in range(1 << k):
        p = base
        for i in range(k):
            p *= (1 + U) if mask >> i & 1 else (1 - U)
        best = max(best, abs(p / divisor - 1))
    return best


def pairwise_depth(M: int) -> int:
    """Depth of the explicit pairwise tree of r5_cccc_energy.pairwise_sum (odd lengths padded with an exact zero)."""
    d = 0
    while M > 1:
        M = (M + 1) // 2
        d += 1
    return d


def grid_constants(m: Mesh, N: int) -> dict:
    hxd, hyd = m.lx / m.nx, m.ly / m.ny  # h_d = fl(l/n), as in the path (I-2)
    cxd, cyd = hxd**2, hyd**2  # c = h_d**2 by Python pow, as in the path (I-2)
    dhx, dhy = exact(hxd) / m.hx - 1, exact(hyd) / m.hy - 1
    dcx, dcy = exact(cxd) / m.hx**2 - 1, exact(cyd) / m.hy**2 - 1
    if 1 + dcx <= 0 or 1 + dcy <= 0:
        raise SecondPathError("I-2", "1 + delta_c <= 0")
    eta_closed = max(max((1 + U) ** 3 / (1 + d) - 1, 1 - (1 - U) ** 3 / (1 + d)) for d in (dcx, dcy))
    eta_corner = max(_corners(Fraction(1), 3, 1 + d) for d in (dcx, dcy))
    phi = (1 + dhx) * (1 + dhy)
    psi_closed = max(phi * (1 + U) ** 4 - 1, 1 - phi * (1 - U) ** 4)
    psiL_closed = max(phi * (1 + U) ** 3 - 1, 1 - phi * (1 - U) ** 3)
    psi_corner, psiL_corner = _corners(phi, 4), _corners(phi, 3)
    eta, psi, psiL = eta_closed, psi_closed, psiL_closed
    eta_t = eta * (1 + U) + U
    c_q = psi + 2 * (1 + psi) * eta_t
    nu = (1 + psi) * eta_t**2
    M = m.n_nodes + m.n_int
    MQ = m.n_nodes
    L, LQ = pairwise_depth(M), pairwise_depth(MQ)
    gL, gLQ = gam(L), gam(LQ)
    kP, kL, kQ = gL + (1 + gL) * c_q, gL + (1 + gL) * psiL, gLQ + (1 + gLQ) * c_q
    Lbar = m.Lam / m.mu_low
    den = U * (L + C_R)
    KP = max(kP + (1 + gL) * nu * Lbar**2, kL) / den
    KQ = max(kQ / den, 1 + U * Lbar / 2 + (1 + gLQ) * nu * Lbar / (2 * U))
    Kf = 1 + U * Lbar
    for k in (L, LQ, N, 4):
        gam(k)  # P-5
    normal = {"h_dx": hxd, "h_dy": hyd, "c_x": cxd, "c_y": cyd, "h_dx/2": hxd / 2, "h_dy/2": hyd / 2, "D/2": m.D_f / 2}
    return {"h_dx": hxd, "h_dy": hyd, "c_x": cxd, "c_y": cyd, "pow_equals_mul": cxd == hxd * hxd and cyd == hyd * hyd,
            "delta_hx": dhx, "delta_hy": dhy, "delta_cx": dcx, "delta_cy": dcy, "eta": eta, "eta_t": eta_t, "psi": psi,
            "psi_L": psiL, "c_q": c_q, "nu": nu, "c_l": psiL, "M": M, "M_Q": MQ, "L": L, "L_Q": LQ, "gamma_L": gL, "gamma_LQ": gLQ,
            "kappa_P": kP, "kappa_L": kL, "kappa_Q": kQ, "Lambda": m.Lam, "mu_low": m.mu_low, "Lambda_bar": Lbar, "K_P": KP,
            "K_Q": KQ, "K_f": Kf, "K_star": max(KP, KQ, Kf),
            "self_checks": {"eta_closed_equals_corners": eta_closed == eta_corner, "psi_closed_equals_corners": psi_closed == psi_corner,
                            "psiL_closed_equals_corners": psiL_closed == psiL_corner,
                            "L_equals_ceil_log2": L == (M - 1).bit_length() and LQ == (MQ - 1).bit_length()},
            "normal": {k: (math.isfinite(v) and abs(v) > 2.0**-1022) for k, v in normal.items()}}


# ---------------------------------------------------------------------------------------------------------------------------------
# premises: probes and the own replication of the double path G-4 (ST 1.2, 2, 13 item 2)
# ---------------------------------------------------------------------------------------------------------------------------------

_MAG = np.uint64(0x7FFF_FFFF_FFFF_FFFF)
_SIGN = np.uint64(0x8000_0000_0000_0000)
_EXPALL = np.uint64(0x7FF0_0000_0000_0000)
_MINNORMAL = np.uint64(0x0010_0000_0000_0000)


def _bits(a) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(a, dtype=float)).view(np.uint64)


def probes() -> dict:
    """This thread: round to nearest (ties to even, both signs), gradual underflow (no FTZ) and subnormal operands kept (no DAZ),
    scalar and vector; operands built at run time; results compared bit by bit."""
    fails = []
    one, tiny = float(int("1")), struct.unpack("<d", struct.pack("<Q", 0x0010_0000_0000_0000))[0]
    half_ulp = math.ldexp(1.0, -53)
    sub = struct.unpack("<d", struct.pack("<Q", 0x0008_0000_0000_0000))[0]  # 2^-1023
    checks = (("round to nearest, tie to even up", one + half_ulp, one), ("tie to even, negative", -one - half_ulp, -one),
              ("round to nearest, above tie", one + math.ldexp(3.0, -54), math.nextafter(one, 2.0)),
              ("gradual underflow x", tiny * 0.5, sub), ("gradual underflow /", tiny / 2.0, sub),
              ("subnormal operand kept (no DAZ)", sub * 2.0, tiny), ("subnormal sum exact", sub + sub, tiny))
    for name, got, want in checks:
        if struct.pack("<d", got) != struct.pack("<d", want):
            fails.append(f"python {name}")
    for n in (1, 7, 64):
        v = np.full(n, tiny)
        s = np.full(n, sub)
        for name, got, want in (("vector gradual underflow", v * 0.5, sub), ("vector no DAZ", s * 2.0, tiny),
                                ("vector round to nearest", np.full(n, one) + half_ulp, one)):
            if not np.all(_bits(got) == _bits(np.array([want]))[0]):
                fails.append(f"numpy[{n}] {name}")
    return {"ok": not fails, "fails": fails, "thread": threading.get_native_id(),
            "main_thread": threading.current_thread() is threading.main_thread()}


class OpChecks:
    """Classification of every checked double operation by bit patterns (independent of the FTZ/DAZ mode itself)."""

    def __init__(self):
        self.violations, self.counts, self.subnormal_results = [], {}, 0

    def _count(self, name, n):
        self.counts[name] = self.counts.get(name, 0) + int(n)

    def finite(self, name, r):
        if np.any((_bits(r) & _EXPALL) == _EXPALL):
            self.violations.append(f"{name}: not finite result (P-2)")

    def mul_div(self, name, a, b, r):
        a, b, r = np.broadcast_arrays(_bits(a), _bits(b), _bits(r))
        self._count(name, r.size)
        nz = ((a & _MAG) != 0) & ((b & _MAG) != 0)
        if np.any((((r & _MAG) == 0) & nz)):
            self.violations.append(f"{name}: zero result of nonzero operands (P-3'(c))")
        if np.any(nz & ((r & _MAG) != 0) & ((r & _MAG) <= _MINNORMAL)):
            self.violations.append(f"{name}: |result| <= 2^-1022 with nonzero operands (P-3'(c))")
        self.finite(name, r.view(float))

    def add_sub(self, name, a, b, r, sub: bool):
        A, B, R = np.broadcast_arrays(_bits(a), _bits(b), _bits(r))
        self._count(name, R.size)
        cancel = (A & _MAG) == ((B if sub else (B ^ _SIGN)) & _MAG)  # magnitudes equal
        cancel &= ((A ^ (B if sub else B ^ _SIGN)) & _SIGN) == 0  # and signs make a - b (or a + b) an exact zero
        bothzero = ((A & _MAG) == 0) & ((B & _MAG) == 0)
        if np.any(((R & _MAG) == 0) & ~cancel & ~bothzero):
            self.violations.append(f"{name}: zero result without exact cancellation (FTZ suspected)")
        self.subnormal_results += int(np.count_nonzero(((R & _MAG) != 0) & ((R & _MAG) < _MINNORMAL)))
        self.finite(name, R.view(float))


def replicate_g4(m: Mesh, w: np.ndarray, with_load: bool, ops: OpChecks, tag: str) -> dict:
    """The G-4 path written again from ST 1.2 with every operation checked; must be bitwise r5_cccc_energy (``bind_double_path``)."""
    w = np.asarray(w, dtype=float)
    nx, ny = m.nx, m.ny
    with np.errstate(all="ignore"):
        hx, hy = m.lx / nx, m.ly / ny
        cx, cy = hx**2, hy**2
        P = np.zeros((nx + 1, ny + 1))
        P[1:-1, 1:-1] = w
        parts = []
        for axis, c in ((0, cx), (1, cy)):
            Pm = np.moveaxis(P, axis, 0)
            Lp = np.zeros_like(Pm)
            two_b = 2.0 * Pm[1:-1, 1:-1]
            ops.finite(f"{tag}: 2 w", two_b)
            s1 = Pm[:-2, 1:-1] - two_b
            ops.add_sub(f"{tag}: w_(i-1) - 2 w_i", Pm[:-2, 1:-1], two_b, s1, sub=True)
            s2 = s1 + Pm[2:, 1:-1]
            ops.add_sub(f"{tag}: + w_(i+1)", s1, Pm[2:, 1:-1], s2, sub=False)
            Lp[1:-1, 1:-1] = s2 / c
            ops.mul_div(f"{tag}: / h^2", s2, c, Lp[1:-1, 1:-1])
            e0, e1 = 2.0 * Pm[1, 1:-1], 2.0 * Pm[-2, 1:-1]
            Lp[0, 1:-1], Lp[-1, 1:-1] = e0 / c, e1 / c
            ops.mul_div(f"{tag}: boundary / h^2", np.concatenate([e0, e1]), c, np.concatenate([Lp[0, 1:-1], Lp[-1, 1:-1]]))
            parts.append(np.moveaxis(Lp, 0, axis))
        t = parts[0] + parts[1]
        ops.add_sub(f"{tag}: t = Lx + Ly", parts[0], parts[1], t, sub=False)
        wx = np.full(nx + 1, hx)
        wx[[0, -1]] = hx / 2
        wy = np.full(ny + 1, hy)
        wy[[0, -1]] = hy / 2
        Wo = np.outer(wx, wy)
        ops.mul_div(f"{tag}: w_I w_J", wx[:, None], wy[None, :], Wo)
        Dh = m.D_f / 2
        DW = Dh * Wo
        ops.mul_div(f"{tag}: (D/2) W", Dh, Wo, DW)
        tt = t * t
        ops.mul_div(f"{tag}: t t", t, t, tt)
        quad = DW * tt
        ops.mul_div(f"{tag}: (D/2) W t^2", DW, tt, quad)
        terms = quad.ravel()
        if with_load:
            qh = m.q_f * hx
            ops.mul_div(f"{tag}: q h_x", m.q_f, hx, qh)
            qhh = qh * hy
            ops.mul_div(f"{tag}: q h_x h_y", qh, hy, qhh)
            ld = qhh * w
            ops.mul_div(f"{tag}: q h_x h_y w", qhh, w, ld)
            terms = np.concatenate([terms, -ld.ravel()])
        v = terms.copy()
        depth = 0
        while v.size > 1:
            if v.size % 2:
                v = np.append(v, 0.0)
            s = v[0::2] + v[1::2]
            ops.add_sub(f"{tag}: pairwise level {depth + 1}", v[0::2], v[1::2], s, sub=False)
            v, depth = s, depth + 1
    return {"value": float(v[0]) if v.size else 0.0, "depth": depth, "t": t, "terms": terms}


def bind_double_path(m: Mesh, wh_p, wh_s, dh, reps: dict) -> dict:
    """I-1: the replication is bitwise the module r5_cccc_energy (the object of E-2) - the only call of that module."""
    g = en.Grid(m.lx, m.nx, m.ly, m.ny, D=m.D_f, q=m.q_f)
    mod = {"p": en.pi_double(g, np.asarray(wh_p, dtype=float)), "s": en.pi_double(g, np.asarray(wh_s, dtype=float)),
           "d": en.q_double(g, np.asarray(dh, dtype=float))}
    return {"module": mod, "bitwise": all(struct.pack("<d", reps[k]["value"]) == struct.pack("<d", mod[k]) for k in mod)}


# ---------------------------------------------------------------------------------------------------------------------------------
# assembly of B_E (ST 5.1, 5.4) for one root rule
# ---------------------------------------------------------------------------------------------------------------------------------

ROOT_RULES = {"own": sqrt_bounds, "qaeb80": sqrt_bounds_qaeb}


def assemble(rad: dict, kc: dict, N: int, rule: str = "own", bits: int | None = None, lower: bool = False) -> dict:
    """B_E from the exact radicands. Upper (default): roots up in numerators, D^(1/2) down in the denominator (ST 5.4). ``lower``:
    the opposite directions - a certified LOWER bound of the exact-root value of the same formula (comparison only)."""
    f = ROOT_RULES[rule]

    def r(x, numerator=True):
        lo, hi = f(x) if bits is None else f(x, bits)
        return (lo if lower else hi) if numerator else (hi if lower else lo)

    sD = r(rad["D"])
    gN = gam(N)
    eps_s = gN * sD * kc["Lambda"] * r(rad["S_h2_s"])
    eps_p = gN * sD * kc["Lambda"] * r(rad["S_h2_p"])
    dh_h, dh_a = r(rad["d_h2"]), r(rad["d_a2"])
    beta = U * sD * kc["Lambda"] * dh_h
    denom = r(rad["D"], numerator=False) * kc["mu_low"]
    if denom <= 0:
        raise SecondPathError("B_omega", "denominator D^(1/2) mu_low not positive")
    res_h = r(rad["res_h2"])
    Bw = res_h / denom + eps_s

    def BP(x):
        return kc["kappa_P"] * x["Sq"] + kc["kappa_L"] * x["Sl"] + (1 + kc["gamma_L"]) * kc["nu"] * x["S2"]

    sp, ss, sd = rad["p"], rad["s"], rad["d"]
    B1 = eps_s * (dh_a + beta) + Bw * (eps_p + eps_s)
    B2 = BP(sp) + BP(ss)
    B3 = beta * dh_a + beta**2 / 2
    B4 = kc["kappa_Q"] * sd["Sq"] + (1 + kc["gamma_LQ"]) * kc["nu"] * sd["S2"]
    BE = B1 + B2 + B3 + B4
    RS = (U * (kc["L"] + C_R) * (sp["Sq"] + sp["Sl"]) + U * (kc["L"] + C_R) * (ss["Sq"] + ss["Sl"])
          + U * (kc["L"] + C_R) * sd["Sq"] + beta * dh_a + eps_s * dh_a + Bw * (eps_p + eps_s))
    return {"sqrt_D": sD, "eps_s": eps_s, "eps_p": eps_p, "d_h": dh_h, "d_a": dh_a, "beta": beta, "res_h": res_h,
            "B_omega_denominator": denom, "B_omega": Bw, "B1": B1, "B2": B2, "B3": B3, "B4": B4, "BE": BE, "R_Sigma": RS,
            "B2_p": BP(sp), "B2_s": BP(ss), "B_nu_second_order": (1 + kc["gamma_L"]) * kc["nu"] * (sp["S2"] + ss["S2"])
            + (1 + kc["gamma_LQ"]) * kc["nu"] * sd["S2"], "B3_second_order": beta**2 / 2}


# components, in the order of the comparison (first divergence = first in this list); definition and source of each
COMPONENTS = [
    ("grid.delta_hx", "h_dx/h_x - 1 exactly, h_dx = fl(l_x/n_x) (I-2)", "ST 2 I-2, 4"),
    ("grid.delta_hy", "h_dy/h_y - 1 exactly", "ST 2 I-2, 4"),
    ("grid.delta_cx", "c_x/h_x^2 - 1 exactly, c_x = h_dx**2 (pow)", "ST 2 I-2, 4"),
    ("grid.delta_cy", "c_y/h_y^2 - 1 exactly", "ST 2 I-2, 4"),
    ("grid.eta", "max |(1 +- u)^3/(1 + delta_c) - 1| (stencil value)", "ST 4, L-4"),
    ("grid.eta_t", "eta (1 + u) + u", "ST 4, L-4"),
    ("grid.psi", "max |phi (1 +- u)^4 - 1| (quadratic factor)", "ST 4, L-5"),
    ("grid.psi_L", "max |phi (1 +- u)^3 - 1| (load factor)", "ST 4, L-5"),
    ("grid.c_q", "psi + 2 (1 + psi) eta_t", "ST 4, L-5"),
    ("grid.nu", "(1 + psi) eta_t^2 (second order of the stencil)", "ST 4, L-5, 5.3"),
    ("grid.L", "pairwise depth of the M = (n_x + 1)(n_y + 1) + (n_x - 1)(n_y - 1) energy terms", "ST 1.2, L-3"),
    ("grid.L_Q", "pairwise depth of the M_Q = (n_x + 1)(n_y + 1) quadratic terms", "ST 1.2, L-3"),
    ("grid.gamma_L", "L u/(1 - L u)", "ST L-1, L-3"),
    ("grid.gamma_LQ", "L_Q u/(1 - L_Q u)", "ST L-1, L-3"),
    ("grid.kappa_P", "gamma_L + (1 + gamma_L) c_q", "ST 5.1"),
    ("grid.kappa_L", "gamma_L + (1 + gamma_L) c_l", "ST 5.1"),
    ("grid.kappa_Q", "gamma_LQ + (1 + gamma_LQ) c_q", "ST 5.1"),
    ("grid.Lambda", "4/h_x^2 + 4/h_y^2", "ST 0"),
    ("grid.mu_low", "4/l_x^2 + 4/l_y^2", "ST 0"),
    ("grid.K_star", "max(K_P, K_Q, K_f)", "ST 6.2"),
] + [(f"{f}.{k}", d, s) for f, name in (("p", "previous field w^_p"), ("s", "current field w^_s"), ("d", "difference d^"))
     for k, d, s in (("Sq", f"S_q({name}) = (D/2) sum W |t| A", "ST 4, VOK-3"),
                     ("Sl", f"S_l({name}) = |q| h_x h_y sum |w|", "ST 4, VOK-3"),
                     ("S2", f"S_2({name}) = (D/2) sum W A^2", "ST 4, L-6"),
                     ("a2", f"||{name}||_a^2 = D sum W t^2", "NS 2.2"),
                     ("h2", f"||{name}||_h^2 = h_x h_y sum w^2", "NS 2.2"))] + [
    ("res_h2", "||D A w^_s - q 1||_h^2, exact D2 residual", "ST L-8, VOK-7 (b)"),
    ("eps_s", "gamma_N D^(1/2) Lambda ||S_s||_h (roots up)", "ST L-8, VOK-5, 5.4"),
    ("eps_p", "gamma_N D^(1/2) Lambda ||S_p||_h (roots up; exactly 0 for an exact zero product)", "ST L-8, VOK-5, 13 item 3"),
    ("d_a", "||d^||_a (root up)", "ST L-7, 5.4"),
    ("beta", "u D^(1/2) Lambda ||d^||_h (roots up)", "ST L-7"),
    ("B_omega", "||D A w^_s - q 1||_h/(D^(1/2) mu_low) + eps_s (numerator up, denominator down)", "ST L-8, VOK-7 (b), 5.4"),
    ("B1", "eps_s (||d^||_a + beta) + B_omega (eps_p + eps_s)", "ST L-8, 5.1"),
    ("B2", "B_P(w^_p) + B_P(w^_s)", "ST L-3, L-5, 5.1"),
    ("B3", "beta ||d^||_a + beta^2/2", "ST L-7, 5.1"),
    ("B4", "kappa_Q S_q(d^) + (1 + gamma_LQ) nu S_2(d^)", "ST L-3, L-5, 5.1"),
    ("BE", "B1 + B2 + B3 + B4", "ST E-2"),
]
ROOT_LEVEL = {"eps_s", "eps_p", "d_a", "beta", "B_omega", "B1", "B2", "B3", "B4", "BE"}  # depend on the root rule


def _component_values(kc: dict, rad: dict, asm: dict) -> dict:
    out = {f"grid.{k}": kc[k] for k in ("delta_hx", "delta_hy", "delta_cx", "delta_cy", "eta", "eta_t", "psi", "psi_L", "c_q", "nu",
                                        "L", "L_Q", "gamma_L", "gamma_LQ", "kappa_P", "kappa_L", "kappa_Q", "Lambda", "mu_low",
                                        "K_star")}
    for f in ("p", "s", "d"):
        for k in ("Sq", "Sl", "S2", "a2", "h2"):
            out[f"{f}.{k}"] = rad[f][k]
    out["res_h2"] = rad["res_h2"]
    for k in ("eps_s", "eps_p", "d_a", "beta", "B_omega", "B1", "B2", "B3", "B4", "BE"):
        out[k] = asm[k]
    return out


# ---------------------------------------------------------------------------------------------------------------------------------
# the second path for one row
# ---------------------------------------------------------------------------------------------------------------------------------

def _prem(state: str, reason: str) -> dict:
    return {"state": state, "reason": reason}


def second_path(mesh: dict, Xp, Yp, Xs, Ys, wh_p, wh_s, dh, P_hat: dict | None = None, parts: bool = False,
                timing: bool = False) -> dict:
    """B_E,s of one half-step by the second path. ``mesh`` = {lx, nx, ly, ny, D, q}; factors as oriented in the path
    (w^ = fl(X Y^T)); ``P_hat`` = {"P_p", "P_s", "Q"} (binary64 of G-4) to evaluate E_s; ``parts``: E^(1)...E^(4) of E-1."""
    t0 = time.perf_counter()
    clock = {}

    def tick(name, t=[t0]):
        now = time.perf_counter()
        clock[name] = clock.get(name, 0.0) + now - t[0]
        t[0] = now

    out = {"data_status": DATA_STATUS, "mesh": {k: mesh[k] for k in ("lx", "nx", "ly", "ny", "D", "q")}, "reasons": []}
    reasons = out["reasons"]
    premises = {}
    out["premises"] = premises
    try:
        m = Mesh(mesh["lx"], mesh["nx"], mesh["ly"], mesh["ny"], mesh.get("D", 1.0), mesh.get("q", 1.0))
        if m.D <= 0:
            raise SecondPathError("root", "D <= 0: D^(1/2) not defined or zero (domain of B_E; Grid requires D > 0)")
        arrays ={"Xp": Xp, "Yp": Yp, "Xs": Xs, "Ys": Ys, "wh_p": wh_p, "wh_s": wh_s, "dh": dh}
        for k, a in arrays.items():
            if not np.all(np.isfinite(np.asarray(a, dtype=float))):
                raise SecondPathError("input", f"{k} contains a value that is not finite")
        tick("mesh")
        # --- premises in this thread
        pb = probes()
        premises["P-1"] = _prem(ASSUMED if pb["ok"] else NOT_ESTABLISHED, "platform premise (IEEE binary64, round to nearest); "
                                "probes " + ("OK" if pb["ok"] else "FAILED: " + "; ".join(pb["fails"])))
        premises["P-3'(a)"] = _prem(ESTABLISHED if pb["ok"] else NOT_ESTABLISHED,
                                    "FTZ/DAZ probes in this thread " + ("OK" if pb["ok"] else "FAILED"))
        fp, fs = formation(m, Xp, Yp, wh_p), formation(m, Xs, Ys, wh_s)
        N = fs["N"]
        if fp["N"] != N:
            raise SecondPathError("formation", "previous and current factors have different N")
        premises["U-1"] = _prem(ESTABLISHED if fp["holds"] and fs["holds"] else NOT_ESTABLISHED,
                                f"|w^ - X Y^T| <= gamma_N |X||Y|^T: previous {fp['violations']}, current {fs['violations']} violations")
        tick("formation (U-1, S)")
        sc = subtraction_check(wh_p, wh_s, dh)
        premises["L-2 (d^)"] = _prem(ESTABLISHED if sc["bitwise_fl_difference"] and sc["L2_holds"] else NOT_ESTABLISHED,
                                     f"d^ bitwise fl(w^_p - w^_s): {sc['bitwise_fl_difference']}; |delta d| <= u |d^|: "
                                     f"{sc['L2_violations']} violations")
        ops = OpChecks()
        reps = {"p": replicate_g4(m, wh_p, True, ops, "P^_p"), "s": replicate_g4(m, wh_s, True, ops, "P^_s"),
                "d": replicate_g4(m, dh, False, ops, "Q^")}
        bind = bind_double_path(m, wh_p, wh_s, dh, reps)
        given_ok = None
        if P_hat is not None:
            given_ok = all(struct.pack("<d", float(P_hat[k])) == struct.pack("<d", reps[r]["value"])
                           for k, r in (("P_p", "p"), ("P_s", "s"), ("Q", "d")))
        premises["I-1"] = _prem(ESTABLISHED if bind["bitwise"] and given_ok is not False else NOT_ESTABLISHED,
                                f"own replication bitwise r5_cccc_energy: {bind['bitwise']}; bitwise the given P^, Q^: {given_ok}")
        premises["P-3'(c), P-2"] = _prem(ESTABLISHED if not ops.violations else NOT_ESTABLISHED,
                                         "all checked operations normal and finite" if not ops.violations else
                                         "; ".join(sorted(set(ops.violations))))
        tick("replication of G-4 (premises)")
        kc = grid_constants(m, N)
        sc_ok = all(kc["self_checks"].values())
        premises["I-2"] = _prem(ESTABLISHED if sc_ok and all(kc["normal"].values()) else NOT_ESTABLISHED,
                                f"delta_h, delta_c exact; closed forms = corner enumeration: {sc_ok}; D/2, h/2, h, c normal: "
                                f"{all(kc['normal'].values())}")
        premises["P-5"] = _prem(ESTABLISHED, f"k u < 1 for k = L {kc['L']}, L_Q {kc['L_Q']}, N {N}, 4")
        tick("grid constants")
        # --- exact radicands
        rad = {"D": m.D, "p": field_quantities(m, fp["computed"]), "s": field_quantities(m, fs["computed"])}
        tick("energy sums of the fields")
        rad["d"] = field_quantities(m, exact_list(dh))
        rad["d_h2"], rad["d_a2"] = rad["d"]["h2"], rad["d"]["a2"]
        tick("energy sums of d^")
        hh = m.hx * m.hy
        rad["S_h2_s"] = hh * sum((x * x for x in fs["S"]), Fraction(0))
        rad["S_h2_p"] = hh * sum((x * x for x in fp["S"]), Fraction(0))
        rad["res_h2"] = residual_h2(m, fs["computed"])
        tick("residual and formation norms")
        asm =assemble(rad, kc, N)
        asm_q = assemble(rad, kc, N, rule="qaeb80")
        asm_lo = assemble(rad, kc, N, lower=True)
        tick("assembly and roots")
        BE_up = round_up(asm["BE"])
        premises["B_E finite"] = _prem(ESTABLISHED if math.isfinite(BE_up) else NOT_ESTABLISHED, f"B_E_up = {BE_up!r}")
        premises["B_omega denominator"] = _prem(ESTABLISHED, "D^(1/2) mu_low > 0 (lower root)")
        premises["N-7"] = _prem(ESTABLISHED, "eps_p from the exact S = |X_p||Y_p|^T: 0 only for an exact zero product"
                                + (" (exact zero product, zero factor)" if fp["S_zero"] else ""))
        out.update({"N": N, "kc": kc, "radicands": rad, "assembly": asm, "assembly_qaeb80": asm_q, "assembly_lower": asm_lo,
                    "BE": asm["BE"], "BE_up": BE_up, "BE_lower_exact_roots": asm_lo["BE"], "formation": {
                        "p": {k: fp[k] for k in ("holds", "violations", "max_ratio", "exact_zero", "S_zero", "factor_zero")},
                        "s": {k: fs[k] for k in ("holds", "violations", "max_ratio", "exact_zero", "S_zero", "factor_zero")}},
                    "subtraction": sc, "op_checks": {"violations": ops.violations, "counts": ops.counts,
                                                     "subnormal_add_results": ops.subnormal_results},
                    "double_path": {"replication": {k: reps[k]["value"] for k in reps}, "module": bind["module"]},
                    "probes": pb, "components": _component_values(kc, rad, asm)})
        # --- exact E_s (and parts) from the exact product fields
        if P_hat is not None:
            Pp, Ps, Qd = (exact(P_hat[k]) for k in ("P_p", "P_s", "Q"))
            wt_p, wt_s = fp["exact"], fs["exact"]
            dt = [a - b for a, b in zip(wt_p, wt_s)]
            qp, qs = field_quantities(m, wt_p), field_quantities(m, wt_s)
            f_val = qp["Pi"] - qs["Pi"] - bilinear(m, dt, dt) / 2
            E_s = (Pp - Ps - Qd) - f_val
            out["E_s"] = E_s
            out["E_s_abs_up"] = round_up(abs(E_s))
            out["E_s_le_B_E"] = abs(E_s) <= asm["BE"]
            out["B_E_le_Kstar_R"] = asm["BE"] <= kc["K_star"] * asm["R_Sigma"]
            tick("exact E_s")
            if parts:
                out["parts"] = e1_parts(m, fp, fs, exact_list(dh), Pp, Ps, Qd, rad, E_s, asm)
                tick("parts E^(1)...E^(4)")
    except SecondPathError as exc:
        reasons.append({"component": exc.component, "premise": None, "reason": exc.reason})
        out["status"] = UNRESOLVED
        out["seconds"] = round(time.perf_counter() - t0, 4)
        out["timing"] = clock
        return out
    for p, st in premises.items():
        if st["state"] not in (ESTABLISHED, ASSUMED):
            reasons.append({"component": "premise", "premise": p, "reason": st["reason"]})
    out["status"] = CERTIFIED if not reasons else UNRESOLVED
    contra = []
    if out["status"] == CERTIFIED and P_hat is not None:
        if not out["E_s_le_B_E"]:
            contra.append("theorem E-2: |E_s| > B_E with the premises established")
        if not out["B_E_le_Kstar_R"]:
            contra.append("theorem E-3: B_E > K* R_Sigma with the premises established")
        if parts and not out["parts"]["ok"]:
            contra.append("parts of E-1: " + ", ".join(k for k, v in out["parts"]["checks"].items() if not v))
    out["contradictions"] = contra
    out["seconds"] = round(time.perf_counter() - t0, 4)
    out["timing"] = clock
    return out


def e1_parts(m: Mesh, fp: dict, fs: dict, dh: list[Fraction], Pp, Ps, Qd, rad: dict, E_s: Fraction, asm: dict) -> dict:
    """E^(1)...E^(4) of E-1 exactly (a_h(w~_s - omega, x) = a_h(w~_s, x) - l_h(x)) and their bounds; the identity with E_s."""
    wp, ws = fp["computed"], fs["computed"]
    tp, ts = fp["exact"], fs["exact"]
    e = [a - b for a, b in zip(ws, ts)]
    ep = [a - b for a, b in zip(wp, tp)]
    d = [a - b for a, b in zip(wp, ws)]
    dd = [a - b for a, b in zip(dh, d)]
    epe = [a - b for a, b in zip(ep, e)]
    E1 = bilinear(m, e, d) + bilinear(m, ts, epe) - load_of(m, epe)
    E2 = (Pp - rad["p"]["Pi"]) - (Ps - rad["s"]["Pi"])
    E3 = -bilinear(m, dh, dd) + bilinear(m, dd, dd) / 2
    E4 = -(Qd - rad["d"]["a2"] / 2)
    checks = {"identity": E1 + E2 + E3 + E4 == E_s, "E1_le_B1": abs(E1) <= asm["B1"], "E2_le_B2": abs(E2) <= asm["B2"],
              "E3_le_B3": abs(E3) <= asm["B3"], "E4_le_B4": abs(E4) <= asm["B4"],
              "e_a_le_eps_s": bilinear(m, e, e) <= asm["eps_s"] ** 2, "ep_a_le_eps_p": bilinear(m, ep, ep) <= asm["eps_p"] ** 2,
              "L6": all(rad[k]["S2"] <= m.D / 2 * m.Lam**2 * rad[k]["h2"] for k in ("p", "s", "d"))}
    return {"E1": E1, "E2": E2, "E3": E3, "E4": E4, "checks": checks, "ok": all(checks.values()),
            "ratios": {k: (float(abs(v) / asm[b]) if asm[b] else None) for k, v, b in
                       (("E1/B1", E1, "B1"), ("E2/B2", E2, "B2"), ("E3/B3", E3, "B3"), ("E4/B4", E4, "B4"))}}


# ---------------------------------------------------------------------------------------------------------------------------------
# comparison with the primary QA path (QA-EB), read-only
# ---------------------------------------------------------------------------------------------------------------------------------

def load_qaeb(name: str = "_be2_qaeb", source: str | None = None):
    """QA-EB as a module, read-only; ``source``: a modified text executed under ``name`` (mutation research), the file unchanged."""
    path = ROOT / QAEB_TOOL
    if source is None and name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    if source is None:
        spec.loader.exec_module(mod)
    else:
        exec(compile(source, str(path), "exec"), mod.__dict__)
    return mod


def primary_qaeb(mod, mesh: dict, Xp, Yp, Xs, Ys, dh, wh_p, wh_s) -> dict:
    """The B_E components of the primary path (QA-EB functions grid_constants, GridQ, formation_check, b_e) on the same inputs."""
    lx, nx, ly, ny, D, q = (mesh[k] for k in ("lx", "nx", "ly", "ny", "D", "q"))
    N = np.asarray(Xs).shape[1]
    kc = mod.grid_constants(lx, nx, ly, ny, D, q, N)
    gq = mod.GridQ(lx, nx, ly, ny, D, q)
    fp, fs = mod.formation_check(Xp, Yp, wh_p), mod.formation_check(Xs, Ys, wh_s)
    be = mod.b_e(gq, kc, N, fp, fs, np.asarray(dh, dtype=float))
    out = {f"grid.{k}": kc[k] for k in ("delta_hx", "delta_hy", "delta_cx", "delta_cy", "eta", "eta_t", "psi", "psi_L", "c_q", "nu",
                                        "L", "L_Q", "gamma_L", "gamma_LQ", "kappa_P", "kappa_L", "kappa_Q", "Lambda", "mu_low",
                                        "K_star")}
    for f in ("p", "s", "d"):
        for k, kk in (("Sq", "Sq"), ("Sl", "Sl"), ("S2", "S2"), ("a2", "a2"), ("h2", "h2")):
            out[f"{f}.{k}"] = be["sums"][f][kk]
    bnd = be["bnd"] or {}
    out.update({"res_h2": bnd.get("res_h2"), "eps_s": be["eps_s"], "eps_p": be["eps_p"], "d_a": be["dh_a"], "beta": be["beta"],
                "B_omega": be["Bw"], "B1": bnd.get("B1"), "B2": bnd.get("B2"), "B3": bnd.get("B3"), "B4": bnd.get("B4"),
                "BE": be["BE"]})
    out["_U1"] = (fp["holds"], fs["holds"])
    return out


def _exact_upper_decision(rad: dict, kc: dict, N: int, key: str, value: Fraction, max_bits: int = 4096) -> str:
    """Is ``value`` >= the exact-root value of component ``key``? Decided by refining the certified interval (no tolerance)."""
    bits = ROOT_BITS
    while bits <= max_bits:
        lo, hi = assemble(rad, kc, N, bits=bits, lower=True)[key], assemble(rad, kc, N, bits=bits)[key]
        if value < lo:
            return "BELOW"
        if value >= hi:
            return "ABOVE_OR_EQUAL"
        bits *= 2
    return "UNDECIDED"


def compare(sp: dict, primary: dict) -> dict:
    """Component by component, in COMPONENTS order: exact equality with this path (root-level components in the documented QA-EB
    representation, the same exact radicands). First divergence with inputs, definition, both values and the conclusion:
      MATCH; ERROR_VALUE (a root-free component differs: a different mathematical value); ERROR_UNSAFE (a root-level value BELOW
      the exact value of the formula: not an upper bound); ERROR_NOT_DOCUMENTED (a valid upper bound, but not the value of the
      documented formula and root rule: an implementation discrepancy, conservative). Own-rule B_E vs primary B_E: REPRESENTATION
      (admissible) when the primary equals the documented representation and both are upper bounds of the same exact formula."""
    if sp.get("status") is None or "components" not in sp:
        return {"comparable": False, "reason": "the second path produced no components"}
    mine_q = _component_values(sp["kc"], sp["radicands"], sp["assembly_qaeb80"])
    rows, first = [], None
    for key, definition, source in COMPONENTS:
        a, b = mine_q[key], primary.get(key)
        if b is None:
            st = "MISSING_IN_PRIMARY"
        elif a == b:
            st = "MATCH"
        elif key not in ROOT_LEVEL:
            st = "ERROR_VALUE"
        else:
            dec = _exact_upper_decision(sp["radicands"], sp["kc"], sp["N"], key, Fraction(b))
            st = "ERROR_UNSAFE" if dec == "BELOW" else ("ERROR_NOT_DOCUMENTED" if dec == "ABOVE_OR_EQUAL" else "UNDECIDED")
        row = {"component": key, "definition": definition, "source": source, "status": st,
               "second_path": float(a) if isinstance(a, Fraction) else a, "primary": float(b) if isinstance(b, Fraction) else b,
               "second_path_exact_sha256": fsha(a) if isinstance(a, Fraction) else None,
               "primary_exact_sha256": fsha(b) if isinstance(b, Fraction) else None}
        rows.append(row)
        if st != "MATCH" and first is None:
            first = {**row, "inputs": {"mesh": sp["mesh"], "N": sp["N"]},
                     "conclusion": {"ERROR_VALUE": "error: the primary computes a different mathematical value",
                                    "ERROR_UNSAFE": "error: the primary value is below the exact value of the formula (unsafe)",
                                    "ERROR_NOT_DOCUMENTED": "error of the implementation against its documented formula "
                                                            "(the value is still an upper bound)",
                                    "MISSING_IN_PRIMARY": "the primary did not produce this component",
                                    "UNDECIDED": "not decided at the maximal root precision"}.get(st)}
    rel = None
    if primary.get("BE") is not None:
        own = sp["assembly"]["BE"]
        dec_own = _exact_upper_decision(sp["radicands"], sp["kc"], sp["N"], "BE", own)
        dec_pri = _exact_upper_decision(sp["radicands"], sp["kc"], sp["N"], "BE", Fraction(primary["BE"]))
        rel = {"own_rule_BE_up": round_up(own), "primary_BE_up": round_up(Fraction(primary["BE"])),
               "own_rule_is_upper_bound": dec_own, "primary_is_upper_bound": dec_pri,
               "both_upper_bounds_of_the_exact_formula": dec_own == dec_pri == "ABOVE_OR_EQUAL",
               "primary_is_documented_representation": mine_q["BE"] == primary["BE"]}
        rel["relation"] = ("IDENTICAL" if own == primary["BE"] else "REPRESENTATION (admissible: same exact radicands, "
                           "different directed-root rule, both upper bounds)" if rel["primary_is_documented_representation"]
                           and rel["both_upper_bounds_of_the_exact_formula"] else "NOT_EXPLAINED_BY_REPRESENTATION")
    return {"comparable": True, "rows": rows, "first_divergence": first, "all_match": first is None,
            "BE_own_vs_primary": rel, "U1_primary": primary.get("_U1")}


# ---------------------------------------------------------------------------------------------------------------------------------
# inputs: synthetic rows and the published Q3 rows (read-only, sha256-checked)
# ---------------------------------------------------------------------------------------------------------------------------------

def synthetic_row(lx: float, nx: int, ly: float, ny: int, N: int = 1, seed: int = 0, D: float = 1.0, q: float = 1.0,
                  zero_prev: bool = False, scale: float = 1.0, near: int | None = None) -> dict:
    """A half-step-shaped input of fixed construction (NOT a B-C solution; E-2 holds for any arrays under its premises):
    factors of a fixed generator, fields w^ = fl(X Y^T) (numpy.matmul, the path), d^ = fl(w^_p - w^_s), P^, Q^ by r5_cccc_energy.
    ``zero_prev``: previous factor exactly 0 (s = 1); ``scale``: factors times scale (near the normal range); ``near``: current
    factors = previous ones changed in the last ``near`` bits (nearly cancelling difference)."""
    rng = np.random.default_rng(seed)
    U_ = np.round(rng.standard_normal((nx - 1, N)) * 2**20) / 2**20 * scale
    Cp = np.zeros((ny - 1, N)) if zero_prev else np.round(rng.standard_normal((ny - 1, N)) * 2**30) / 2**30 * 1e-3 * scale
    if near is not None:
        V = Cp.copy()
        for _ in range(near):  # the previous factor moved by ``near`` ulps: w^_p - w^_s nearly cancels
            V = np.nextafter(V, np.inf)
    else:
        V = np.round(rng.standard_normal((ny - 1, N)) * 2**30) / 2**30 * 1e-3 * scale
    Xp, Yp, Xs, Ys = U_, Cp, U_, V
    wh_p, wh_s = Xp @ Yp.T, Xs @ Ys.T
    dh = wh_p - wh_s
    g = en.Grid(lx, nx, ly, ny, D=D, q=q)
    mesh = {"lx": lx, "nx": nx, "ly": ly, "ny": ny, "D": D, "q": q}
    return {"mesh": mesh, "Xp": Xp, "Yp": Yp, "Xs": Xs, "Ys": Ys, "wh_p": wh_p, "wh_s": wh_s, "dh": dh,
            "P_hat": {"P_p": en.pi_double(g, wh_p), "P_s": en.pi_double(g, wh_s), "Q": en.q_double(g, dh)}}


def run_row(row: dict, parts: bool = False) -> dict:
    return second_path(row["mesh"], row["Xp"], row["Yp"], row["Xs"], row["Ys"], row["wh_p"], row["wh_s"], row["dh"],
                       row.get("P_hat"), parts=parts)


def _load_script(rel: str, name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def q3_rows(arrays: Path, max_nodes: int, limit: int | None = None) -> list[dict]:
    """Published Q3 rows (EQ sample, ST 7) on grids with n_x n_y <= max_nodes. Inputs reconstructed with the EQ functions (the object
    of the bound; read-only), after the sha256 of every used NPZ is checked against BOTH published meta.json lists; the fields are
    bound bitwise to the published P_hat_prev, P_hat_s, Q_hat. No input is replaced; a check that fails makes the row unusable."""
    eqm = _load_script("scripts/r5_cccc_q4_energy_bc_audit.py", "_be2_eq")
    ga, q3 = eqm._ga(), eqm._ga()._q3()
    from kkvim_reference import r5_cccc_bc as bc
    from kkvim_reference import r5_cccc_halfstep as hs
    meta_eq = json.loads((ROOT / "docs/results/r5_cccc_q4_energy_bc/meta.json").read_text(encoding="utf-8"))
    meta_ga = json.loads((ROOT / "docs/results/r5_cccc_q4_gamma_audit/meta.json").read_text(encoding="utf-8"))
    sha_eq = {e["file"]: e["sha256"] for e in meta_eq["npz"]}
    sha_ga = {e["file"]: e["sha256"] for e in meta_ga["npz"]}
    with open(ROOT / "docs/results/r5_cccc_q4_energy_bc/rows.csv", encoding="utf-8") as f:
        eq_rows = {r["id"]: r for r in csv.DictReader(f)}
    with open(ROOT / "docs/results/r5_cccc_q4_energy_error_bound/rows.csv", encoding="utf-8") as f:
        eb_rows = {r["id"]: r for r in csv.DictReader(f)}
    rows = ga.read_q3_rows(ROOT / "docs/results/r5_cccc_q3_prequalification/sequences_e1_e7_e9.csv")
    groups = {}
    for r in rows:
        if int(r["nx"]) * int(r["ny"]) <= max_nodes:
            groups.setdefault(ga.seq_key(r), []).append(r)
    out = []
    for rs in groups.values():
        spec = ga.spec_of(rs[0])
        npz = Path(arrays) / q3.seq_file(spec)
        got = sha256_file(npz)
        if not (sha_eq.get(npz.name) == got == sha_ga.get(npz.name)):
            out.append({"id": f"{npz.name}", "unusable": "NPZ sha256 differs from the published meta.json lists"})
            continue
        z = np.load(npz)
        steps = ga._steps_from_npz(z, spec)
        by_s = {int(r["j_s"]): r for r in rs}
        for i, st in enumerate(steps):
            rid = ga.row_id(by_s[st["s"]])
            N, s = st["given"].shape[1], st["s"]
            p_in, comp_in, _ = eqm.bc_input(spec, steps, i)
            rec = bc.run_halfstep(p_in, qr=N >= 2)
            if rec.solved is None:
                out.append({"id": rid, "unusable": "prototype: given set exactly dependent"})
                continue
            solved = rec.solved
            Cp = comp_in @ rec.R.T if rec.R is not None else np.asarray(comp_in, dtype=float)
            sol = eqm.solve_c3plus(solved, hs.exact_coefficients(solved)).pop("_sol")
            if sol.s is None:
                out.append({"id": rid, "unusable": f"B-C produced no solution ({sol.status})"})
                continue
            V = sol.s.reshape(N, solved.n_s - 1).T
            Xp, Yp = eqm.orient(st["direction"], solved.given, Cp)
            Xs, Ys = eqm.orient(st["direction"], solved.given, V)
            wh_p, wh_s = Xp @ Yp.T, Xs @ Ys.T
            dh = wh_p - wh_s
            g = en.Grid(spec["lx"], spec["nx"], spec["ly"], spec["ny"])
            P_hat = {"P_p": en.pi_double(g, wh_p), "P_s": en.pi_double(g, wh_s), "Q": en.q_double(g, dh)}
            h = eq_rows.get(rid, {})
            bound = all(h.get(k) not in (None, "") and ga.read_double(h[k]).hex() == float(P_hat[kk]).hex()
                        for k, kk in (("P_hat_prev", "P_p"), ("P_hat_s", "P_s"), ("Q_hat", "Q")))
            out.append({"id": rid, "N": N, "s": s, "npz": npz.name, "npz_sha256": got, "bound_to_published": bound,
                        "published_B_E_up": eb_rows.get(rid, {}).get("B_E_up"), "published_status": eb_rows.get(rid, {}).get("status"),
                        "mesh": {"lx": spec["lx"], "nx": spec["nx"], "ly": spec["ly"], "ny": spec["ny"], "D": 1.0, "q": 1.0},
                        "Xp": Xp, "Yp": Yp, "Xs": Xs, "Ys": Ys, "wh_p": wh_p, "wh_s": wh_s, "dh": dh, "P_hat": P_hat})
            if limit is not None and sum(1 for x in out if "unusable" not in x) >= limit:
                return out
    return out


# ---------------------------------------------------------------------------------------------------------------------------------
# reports, resources, command line
# ---------------------------------------------------------------------------------------------------------------------------------

def summary_of(res: dict, cmp: dict | None = None) -> dict:
    a = res.get("assembly") or {}
    s = {"status": res.get("status"), "reasons": res.get("reasons"), "mesh": res.get("mesh"), "N": res.get("N"),
         "BE_up": res.get("BE_up"), "BE_sha256": fsha(res.get("BE")), "E_s_abs_up": res.get("E_s_abs_up"),
         "E_s_le_B_E": res.get("E_s_le_B_E"), "B_E_le_Kstar_R": res.get("B_E_le_Kstar_R"),
         "parts": {k: round_up(a[k]) for k in ("B1", "B2", "B3", "B4", "B_omega", "eps_s", "eps_p", "beta", "d_a")} if a else None,
         "contradictions": res.get("contradictions"), "seconds": res.get("seconds"), "timing": res.get("timing")}
    if "parts" in res:
        s["E1_E4"] = {"checks": res["parts"]["checks"], "ratios": res["parts"]["ratios"]}
    if cmp is not None:
        s["comparison"] = {"all_match": cmp.get("all_match"), "first_divergence": cmp.get("first_divergence"),
                           "BE_own_vs_primary": cmp.get("BE_own_vs_primary")}
    return s


RESOURCE_CASES = ((8, 10, 1), (16, 16, 1), (32, 32, 1), (48, 40, 1), (64, 64, 1), (96, 96, 1), (128, 128, 1), (64, 64, 3))


def resources(cases=RESOURCE_CASES, parts: bool = True) -> list[dict]:
    """Time, peak Python memory (tracemalloc) and the cost of each stage on synthetic rows outside D1 (with E_s and E^(1)...E^(4))."""
    out = []
    for nx, ny, N in cases:
        row = synthetic_row(1.0, nx, 1.25, ny, N=N, seed=nx * 1000 + ny)
        tracemalloc.start()
        t0 = time.perf_counter()
        res = run_row(row, parts=parts)
        sec = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        t0 = time.perf_counter()
        prim = primary_qaeb(load_qaeb(), row["mesh"], row["Xp"], row["Yp"], row["Xs"], row["Ys"], row["dh"], row["wh_p"],
                            row["wh_s"])
        sec_primary = time.perf_counter() - t0
        out.append({"grid": f"{nx}x{ny}", "N": N, "nodes": (nx + 1) * (ny + 1), "seconds": round(sec, 3),
                    "peak_python_mb": round(peak / 2**20, 1), "stages": {k: round(v, 3) for k, v in res["timing"].items()},
                    "primary_qaeb_seconds": round(sec_primary, 3), "status": res["status"],
                    "matches_primary": compare(res, prim)["all_match"]})
    return out


def provenance() -> dict:
    def blob(p):
        data = (ROOT / p).read_bytes().replace(b"\r\n", b"\n")
        return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()

    return {"tool_blob": blob(SELF), "qaeb_blob": blob(QAEB_TOOL), "energy_module_blob": blob("src/kkvim_reference/r5_cccc_energy.py"),
            "python": sys.version.split()[0], "numpy": np.__version__, "platform": platform.platform(),
            "processor": platform.processor(), "blas_threads_env": {k: os.environ.get(k) for k in THREAD_VARS},
            "root_bits": ROOT_BITS, "qaeb_root_bits": QAEB_ROOT_BITS}


def guard_out(path: Path) -> None:
    p, docs = Path(path).resolve(), (ROOT / "docs").resolve()
    if p == docs or docs in p.parents:
        raise SystemExit("refused: output under docs/ (research data go to scratch/)")


def _json(o):
    if isinstance(o, Fraction):
        return {"float": float(o), "sha256": fsha(o)}
    if isinstance(o, (np.bool_, np.integer, np.floating)):
        return o.item()
    if isinstance(o, np.ndarray):
        return None
    return str(o)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--profile", choices=("synthetic", "q3", "resources"), required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--arrays", type=Path, default=Path(os.environ.get(
        "KKVIM_Q3_ARRAYS", r"D:\git\Test_Versions\KK-VIM\KK-VIM-Q3-QA\scratch\r5_cccc_q3_prequalification\arrays")))
    r.add_argument("--max-nodes", type=int, default=1024)
    r.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(argv)
    guard_out(a.out)
    a.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    doc = {"data_status": DATA_STATUS, "profile": a.profile, "provenance": provenance()}
    if a.profile == "synthetic":
        items = []
        for spec in ((1.0, 8, 1.25, 10, 1, 1, False), (1.0, 12, 0.75, 9, 2, 2, False), (1.0, 10, 1.0, 10, 3, 3, True),
                     (2.0, 9, 1.0, 7, 1, 4, False)):
            row = synthetic_row(spec[0], spec[1], spec[2], spec[3], N=spec[4], seed=spec[5], zero_prev=spec[6])
            res = run_row(row, parts=True)
            prim = primary_qaeb(load_qaeb(), row["mesh"], row["Xp"], row["Yp"], row["Xs"], row["Ys"], row["dh"], row["wh_p"],
                                row["wh_s"])
            items.append(summary_of(res, compare(res, prim)))
        doc["rows"] = items
    elif a.profile == "q3":
        items = []
        for row in q3_rows(a.arrays, a.max_nodes, a.limit):
            if "unusable" in row:
                items.append(row)
                continue
            res = run_row(row, parts=True)
            prim = primary_qaeb(load_qaeb(), row["mesh"], row["Xp"], row["Yp"], row["Xs"], row["Ys"], row["dh"], row["wh_p"],
                                row["wh_s"])
            s = summary_of(res, compare(res, prim))
            pub = row["published_B_E_up"]
            s.update({"id": row["id"], "npz": row["npz"], "npz_sha256": row["npz_sha256"], "bound_to_published": row["bound_to_published"],
                      "published_status": row["published_status"], "published_B_E_up": pub,
                      "qaeb80_BE_up_equals_published": pub not in (None, "") and round_up(res["assembly_qaeb80"]["BE"]).hex()
                      == float(pub).hex() if res.get("assembly_qaeb80") else None})
            items.append(s)
            print(row["id"], s["status"], s["comparison"]["all_match"], s.get("qaeb80_BE_up_equals_published"), flush=True)
        doc["arrays_dir"] = str(a.arrays)
        doc["rows"] = items
    else:
        doc["resources"] = resources()
    doc["seconds"] = round(time.time() - t0, 1)
    with open(a.out / f"{a.profile}.json", "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1, ensure_ascii=False, default=_json)
    print(json.dumps({"profile": a.profile, "seconds": doc["seconds"], "rows": len(doc.get("rows", doc.get("resources", [])))}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
