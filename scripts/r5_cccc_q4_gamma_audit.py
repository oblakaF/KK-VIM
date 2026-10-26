"""R5-V Q4, priority 1 (D-053.4): independent numerical study of gamma_s for CCCC. QA / research tool, not a production API; no
import of the working ``kkvim`` (D-028); the existing ``kkvim_reference`` modules and the Q3 data are used read-only.

Object (RO VOK-2, VOK-8; NS 4.2-4.3; Q3 report 4.1). For a half-step with given set G (fixed grid h_f = l_f/n_f), solved set V and
previous companion C (solve grid h_s = l_s/n_s), rho = F (x) 1 - K vec(V), Delta = C - V,

    gamma_s = h_s rho^T Delta,      Pi(w~^(s-1)) - Pi(w~^(s)) = ||d~||_a^2/2 - gamma_s   (VOK-2, exact, any V).

All "exact" values here refer to the FINITE DISCRETE problem with binary64 inputs: l_x, l_y, D, q and the stored binary64 factors are
taken exactly as rationals, h = l/n exactly (as the Q3 references; not fl(l/n)). Nothing here is a statement about the continuous
problem.

Paths (none uses another path's implementation of the operator under study):
- double path M0 (the object): the Q3 computed check, r5_cccc_sequence._energy: K, rhs by r5_cccc_halfstep.coefficient_system
  (dense Kronecker matrix, binary64 coefficients, h = fl(l/n)), rho^ = fl(rhs - fl(K s)), gamma^ = fl(fl(l_s/n_s) fl(rho^ . fl(C - V)));
- path A (exact residual): A = h_f G^T G, B = -h_f G^T L_f G, C = h_f G^T D4_f G, F = q h_f G^T 1 (NS 4.2) and K applied by the integer
  stencils L = (1, -2, 1)/h^2 (Dirichlet) and D4 = L^2 + (2/h^4)(e_1 e_1^T + e_m e_m^T), rows (7, -4, 1)/h^4, in exact integer /
  rational arithmetic (own implementation, not r5_cccc_halfstep.residual_exact);
- path B (exact two-dimensional energy): Pi_h^C by its definition NS 2.2 (L^ with the even ghost node, (L^ f)_0 = 2 f_1/h^2,
  trapezoid weights), on the exact products w~ = X Y^T, and gamma_B = Q~ - (Pi(w~^(s-1)) - Pi(w~^(s))); no coefficient, no K;
- path M2 (raised precision): the defining formulas in decimal arithmetic with p digits (round half even), with the conditional
  bound below; p = 34 for the campaign, small p only to test the bound itself.

Error of gamma^ (all parts EXACT, telescoping to gamma^ - gamma): coefficients (A^, B^, C^, F^ taken exactly, exact operators),
operator assembly (K^ against K), matrix-vector product, subtraction, rounding of C - V, dot product, multiplication by fl(h_s).

Bound B_gamma (QA derivation, report section 5; status CONDITIONAL): under P-1...P-5 of RO 0.4, the premise that pow(h, k) is within
1 ulp, classical dot products in BLAS (P-4) and the operation count of the code path (counts in ``double_counts``), with the standard
dot-product bound |fl(x^T y) - x^T y| <= gamma_m |x|^T |y| for any order of summation [STD.; no source in the project library]:

    |gamma^ - gamma| <= gamma_2 h_s |g| + h_s [gamma_M RD + E + gamma_1 (RD + E)],

g = fl(rho^ . Delta^), RD >= |rho^|^T |Delta^|, E >= E_rho^T |Delta^| with E_rho the componentwise bound of |rho^ - rho| built from the
majorants A_maj = h_f |G|^T|G|, B_maj = h_f |G|^T|L||G|, C_maj = h_f |G|^T|D4|maj|G|, F_maj = |q| h_f |G|^T 1. The same formula with
u_p = 10^(1 - p)/2 and the counts of ``decimal_counts`` bounds the decimal path. A bound that is not applicable (non-finite or
subnormal values, P-2/P-3 not verified) is UNRESOLVED, never replaced by an observation.

Statistics for K_E are given separately for (i) the test form |dP^ - Q^| <= tau = |gamma_x| + K_E R, (ii) the identity control
|dP^ - Q^ + gamma^| <= K_E R (Q3 m_s), (iii) the identity control with the exact gamma or with gamma^ and its bound. K_E is NOT assigned;
classification is given as a function of K_E (intervals), no threshold of R4 is used.

Usage (repository root):
    uv run python scripts/r5_cccc_q4_gamma_audit.py run --profile full [--workers 8]
    uv run python scripts/r5_cccc_q4_gamma_audit.py run --profile ci --out scratch/r5_cccc_q4_ci
"""

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import multiprocessing
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from decimal import ROUND_HALF_EVEN, Context, Decimal
from fractions import Fraction
from pathlib import Path

THREAD_VARS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
for _k in THREAD_VARS:
    os.environ[_k] = "1"  # as in Q3: one BLAS thread, so that the double path is reproduced bitwise

import numpy as np  # noqa: E402
import scipy  # noqa: E402

from kkvim_reference import r5_cccc_energy as en  # noqa: E402
from kkvim_reference import r5_cccc_halfstep as hs  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "results" / "r5_cccc_q4_gamma_audit"
Q3_DIR = ROOT / "docs" / "results" / "r5_cccc_q3_prequalification"
Q3_CSV = Q3_DIR / "sequences_e1_e7_e9.csv"
Q3_SCRIPT = ROOT / "scripts" / "r5_cccc_q3_prequalification.py"
Q3_DATA_COMMIT = "98631164b164b7a8852fb00069f691cf8aafbbfb"  # Q3 campaign commit (D-053, basis)
Q3_ARRAYS_DEFAULT = Path(os.environ.get("KKVIM_Q3_ARRAYS", r"D:\git\KK-VIM-Q3-QA\scratch\r5_cccc_q3_prequalification\arrays"))
U = Fraction(1, 2**53)
C_R = 22  # Q3 path count of r5_cccc_energy (G-4) used in the Q3 denominators; NOT an owner's c_R
DEC_P = 34  # campaign precision of the decimal path
DEC_LOW = (6, 8, 12, 16)  # small precisions: a test of the bound derivation only
TINY = 2.0**-1022
UNRESOLVED = "UNRESOLVED"


def _q3():
    """The Q3 campaign script as a module (sample definitions, seq_job for the CI profile, peak memory); read-only use."""
    if not hasattr(_q3, "mod"):
        spec = importlib.util.spec_from_file_location("r5_cccc_q3_prequalification", Q3_SCRIPT)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _q3.mod = mod
    return _q3.mod


def gam(k: int, u: Fraction = U) -> Fraction:
    """gamma_k = k u/(1 - k u) (exact rational); requires k u < 1."""
    if k * u >= 1:
        raise ValueError("k u >= 1: gamma_k undefined")
    return k * u / (1 - k * u)


# ---------------------------------------------------------------------------------------------------------------------------------
# reading rules of the Q3 CSV (Q3 meta: "*_up rounded up and *_down rounded down to 17 significant digits; other floats repr")
# ---------------------------------------------------------------------------------------------------------------------------------

def read_double(text: str) -> float:
    """A repr-printed binary64: float(text) is that double exactly (round trip of repr)."""
    v = float(text)
    if repr(v) != text:
        raise ValueError(f"not a repr-printed double: {text!r}")
    return v


def read_bound(text: str) -> Fraction:
    """A directed 17-digit decimal (*_up / *_down): the decimal itself, exactly, as the bound (NOT float(text): rounding to nearest
    may move it to the other side of the quantity)."""
    return Fraction(Decimal(text))


def read_q3_rows(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def spec_of(row: dict) -> dict:
    """Input of the sequence of a Q3 row: chain points by plate_grid (exact ratio doubles of RATIOS), pairs 1 x 1."""
    q3 = _q3()
    if row["group"] == "chain":
        lx, nx, ly, ny = q3.plate_grid(row["r"], int(row["n"]))
    else:
        lx, nx, ly, ny = 1.0, int(row["nx"]), 1.0, int(row["ny"])
    assert (nx, ny) == (int(row["nx"]), int(row["ny"])) and q3.grid_label(lx, nx, ly, ny) == row["grid"]
    return {"group": row["group"], "r": row["r"], "n": row["n"], "lx": lx, "nx": nx, "ly": ly, "ny": ny, "N": int(row["N"]),
            "first": row["first"]}


def seq_key(row: dict) -> tuple:
    return (row["group"], row["grid"], row["N"], row["first"])


# ---------------------------------------------------------------------------------------------------------------------------------
# exact dyadic integers (binary64 values carried over exactly: value = int * 2**E)
# ---------------------------------------------------------------------------------------------------------------------------------

def dyadic(*arrays):
    """Columns of each (m, N) array as Python int lists with ONE common exponent E; exact."""
    flat = [np.asarray(a, dtype=np.float64) for a in arrays]
    for a in flat:
        if not np.all(np.isfinite(a)):
            raise ValueError("non-finite input")
    ratios = [[x.as_integer_ratio() for x in a.ravel(order="F").tolist()] for a in flat]
    K = max((q.bit_length() - 1 for rs in ratios for _, q in rs), default=0)
    out = []
    for a, rs in zip(flat, ratios):
        ints = [p << (K - (q.bit_length() - 1)) for p, q in rs]
        m = a.shape[0]
        out.append([ints[j * m:(j + 1) * m] for j in range(a.shape[1] if a.ndim == 2 else 1)])
    return out, -K


def vec_dyadic(v):
    (cols,), E = dyadic(np.asarray(v, dtype=float).reshape(-1, 1))
    return cols[0], E


def p2(e: int) -> Fraction:
    return Fraction(2) ** e


def dot(a, b) -> int:
    return sum(x * y for x, y in zip(a, b))


def st_L(c):
    """Integer (1, -2, 1) stencil with zero Dirichlet ends (h not included)."""
    p = [0] + list(c) + [0]
    return [p[k] - 2 * p[k + 1] + p[k + 2] for k in range(len(c))]


def st_D4(c):
    """Integer D4 = L^2 + 2 (e_1 e_1^T + e_m e_m^T): rows (7, -4, 1), interior (1, -4, 6, -4, 1) (h^4 not included)."""
    out = st_L(st_L(c))
    out[0] += 2 * c[0]
    out[-1] += 2 * c[-1]
    return out


def st_absL(c):
    p = [0] + [abs(x) for x in c] + [0]
    return [p[k] + 2 * p[k + 1] + p[k + 2] for k in range(len(c))]


def st_absD4(c):
    """|L|^2 |c| + 2|c| at both ends: a majorant of |D4| |c| entrywise."""
    out = st_absL(st_absL(c))
    out[0] += 2 * abs(c[0])
    out[-1] += 2 * abs(c[-1])
    return out


# ---------------------------------------------------------------------------------------------------------------------------------
# path A: exact residual
# ---------------------------------------------------------------------------------------------------------------------------------

def coefficients_exact(G, EG, h_f: Fraction, q: Fraction):
    """A, B, C, F (Fractions) of the given set (int columns G, exponent EG) by the definitions of NS 4.2."""
    N = len(G)
    LG = [st_L(c) for c in G]
    D4G = [st_D4(c) for c in G]
    s = p2(2 * EG)
    A = [[h_f * dot(G[i], G[k]) * s for k in range(N)] for i in range(N)]
    B = [[-dot(G[i], LG[k]) * s / h_f for k in range(N)] for i in range(N)]
    C = [[dot(G[i], D4G[k]) * s / h_f**3 for k in range(N)] for i in range(N)]
    F = [q * h_f * sum(G[i]) * p2(EG) for i in range(N)]
    return A, B, C, F


def gamma_from_coefficients(coef, V, Vp, EV, h_s: Fraction, D: Fraction) -> Fraction:
    """h_s sum_i rho_i^T Delta_i with rho = F (x) 1 - D [A (x) D4 - 2 B (x) L + C (x) I] vec(V), Delta = Vp - V, exactly."""
    A, B, C, F = coef
    N = len(V)
    L4 = [st_D4(c) for c in V]
    L2 = [st_L(c) for c in V]
    Dl = [[a - b for a, b in zip(Vp[i], V[i])] for i in range(N)]
    total = Fraction(0)
    for i in range(N):
        acc = F[i] * sum(Dl[i]) * p2(EV)
        kp = Fraction(0)
        for k in range(N):
            kp += (A[i][k] / h_s**4 * dot(L4[k], Dl[i]) - 2 * B[i][k] / h_s**2 * dot(L2[k], Dl[i])
                   + C[i][k] * dot(V[k], Dl[i])) * p2(2 * EV)
        total += acc - D * kp
    return h_s * total


# ---------------------------------------------------------------------------------------------------------------------------------
# path B: exact two-dimensional energy by the definition NS 2.2
# ---------------------------------------------------------------------------------------------------------------------------------

def _int_field(X, Y):
    """W = X Y^T as an object array of Python ints (exponent returned by the caller's dyadic)."""
    Xo = np.array(X, dtype=object).T  # (m_x, N)
    Yo = np.array(Y, dtype=object).T  # (m_y, N)
    return Xo.dot(Yo.T)


def _hat_parts(W, nx, ny):
    """Integer parts of L^_x (x) E_y and E_x (x) L^_y on all nodes 0..n_x x 0..n_y (h^2 not included)."""
    P = np.zeros((nx + 1, ny + 1), dtype=object)
    P[1:nx, 1:ny] = W
    Sx = np.zeros((nx + 1, ny + 1), dtype=object)
    Sx[1:nx, :] = P[:-2, :] - 2 * P[1:-1, :] + P[2:, :]
    Sx[0, :] = 2 * P[1, :]
    Sx[nx, :] = 2 * P[nx - 1, :]
    Sy = np.zeros((nx + 1, ny + 1), dtype=object)
    Sy[:, 1:ny] = P[:, :-2] - 2 * P[:, 1:-1] + P[:, 2:]
    Sy[:, 0] = 2 * P[:, 1]
    Sy[:, ny] = 2 * P[:, ny - 1]
    return Sx, Sy


def _weights4(nx, ny):
    wx = np.full(nx + 1, 2, dtype=object)
    wx[0] = wx[-1] = 1
    wy = np.full(ny + 1, 2, dtype=object)
    wy[0] = wy[-1] = 1
    return np.outer(wx, wy)  # 4 w_I w_J / (h_x h_y)


def _a_form(Sx, Sy, c4, hx, hy, D, E):
    sxx = int((c4 * Sx * Sx).sum())
    sxy = int((c4 * Sx * Sy).sum())
    syy = int((c4 * Sy * Sy).sum())
    return D * hx * hy / 4 * (sxx / hx**4 + 2 * Fraction(sxy) / (hx**2 * hy**2) + syy / hy**4) * p2(2 * E)


def energies_path_b(lx, nx, ly, ny, D, q, Xp, Yp, Xs, Ys):
    """(Pi(w~^(s-1)), Pi(w~^(s)), Q~ = ||w~^(s-1) - w~^(s)||_a^2/2), exactly, from the stored factor arrays."""
    hx, hy = Fraction(lx) / nx, Fraction(ly) / ny
    (Xpi, Xsi), EX = dyadic(Xp, Xs)
    (Ypi, Ysi), EY = dyadic(Yp, Ys)
    E = EX + EY
    Wp, Ws = _int_field(Xpi, Ypi), _int_field(Xsi, Ysi)
    Sxp, Syp = _hat_parts(Wp, nx, ny)
    Sxs, Sys = _hat_parts(Ws, nx, ny)
    c4 = _weights4(nx, ny)
    Dq, qq = Fraction(D), Fraction(q)
    a_p, a_s = _a_form(Sxp, Syp, c4, hx, hy, Dq, E), _a_form(Sxs, Sys, c4, hx, hy, Dq, E)
    a_d = _a_form(Sxp - Sxs, Syp - Sys, c4, hx, hy, Dq, E)
    l_p = qq * hx * hy * int(Wp.sum()) * p2(E)
    l_s = qq * hx * hy * int(Ws.sum()) * p2(E)
    return a_p / 2 - l_p, a_s / 2 - l_s, a_d / 2


# ---------------------------------------------------------------------------------------------------------------------------------
# the double path M0 (object of the study) and its exact decomposition
# ---------------------------------------------------------------------------------------------------------------------------------

def double_path(lx, nx, ly, ny, direction, given, companion, V, D=1.0, q=1.0) -> dict:
    """gamma^ exactly as r5_cccc_sequence._energy computes it, keeping every intermediate array."""
    p = hs.HalfStep(lx, nx, ly, ny, direction, given, D=D, q=q)
    K, rhs = hs.coefficient_system(p)
    s_vec = V.T.reshape(-1)
    Ks = K @ s_vec
    rho = rhs - Ks
    rho_ref = rhs - K @ s_vec  # the expression of the Q3 code
    Delta = (companion - V).T.reshape(-1)
    g = float(rho @ Delta)
    h_d = float(p.l_s / p.n_s)
    return {"p": p, "K": K, "rhs": rhs, "s": s_vec, "Ks": Ks, "rho": rho, "rho_same": bool(np.array_equal(rho, rho_ref)),
            "Delta": Delta, "g": g, "h_d": h_d, "gamma_hat": h_d * g, "coef": hs.coefficients(p)}


def _frac_vec_dot(a, b) -> Fraction:
    """Exact a^T b of two float vectors."""
    ai, ea = vec_dyadic(a)
    bi, eb = vec_dyadic(b)
    return Fraction(dot(ai, bi)) * p2(ea + eb)


def stages(dp: dict, gamma_exact: Fraction, gamma_dcoef: Fraction, Delta_exact, EV, h_s: Fraction) -> dict:
    """Exact telescoping parts of gamma^ - gamma (module docstring)."""
    K, s, rhs = dp["K"], dp["s"], dp["rhs"]
    rows, cols = np.nonzero(K)
    vals = K[rows, cols]
    ki, ek = vec_dyadic(vals)
    si, es = vec_dyadic(s)
    Dl = [x for col in Delta_exact for x in col]  # term-first, exponent EV
    Ks_exact = sum(Dl[r] * k * si[c] for r, c, k in zip(rows.tolist(), cols.tolist(), ki))  # Delta^T K^ s
    ri, er = vec_dyadic(rhs)
    rhs_dot = Fraction(dot(ri, Dl)) * p2(er + EV)
    g1 = h_s * (rhs_dot - Fraction(Ks_exact) * p2(ek + es + EV))
    ksi, eks = vec_dyadic(dp["Ks"])
    g2 = h_s * (rhs_dot - Fraction(dot(ksi, Dl)) * p2(eks + EV))
    rhoi, erho = vec_dyadic(dp["rho"])
    g3 = h_s * Fraction(dot(rhoi, Dl)) * p2(erho + EV)
    g4 = h_s * _frac_vec_dot(dp["rho"], dp["Delta"])
    g5 = h_s * Fraction(dp["g"])
    gh = Fraction(dp["gamma_hat"])
    parts = {"coefficients": gamma_dcoef - gamma_exact, "operator_assembly": g1 - gamma_dcoef, "matvec": g2 - g1,
             "subtraction": g3 - g2, "delta_rounding": g4 - g3, "dot_product": g5 - g4, "h_multiplication": gh - g5}
    assert sum(parts.values(), Fraction(0)) == gh - gamma_exact  # exact telescoping
    return parts


# ---------------------------------------------------------------------------------------------------------------------------------
# majorants and the conditional bounds
# ---------------------------------------------------------------------------------------------------------------------------------

def majorant_sums(G, EG, V, Vp, EV, h_f: Fraction, h_s: Fraction, D: Fraction, q: Fraction) -> dict:
    """Exact SF, SA, SB, SC: (F_maj (x) 1)^T|Delta|, |D| (A_maj (x) |D4|maj |V|)^T|Delta| / h_s^4 ... (module docstring)."""
    N = len(G)
    aG = [[abs(x) for x in c] for c in G]
    aLG = [st_absL(c) for c in aG]
    aD4G = [st_absD4(c) for c in aG]
    s = p2(2 * EG)
    Am = [[h_f * dot(aG[i], aG[k]) * s for k in range(N)] for i in range(N)]
    Bm = [[dot(aG[i], aLG[k]) * s / h_f for k in range(N)] for i in range(N)]
    Cm = [[dot(aG[i], aD4G[k]) * s / h_f**3 for k in range(N)] for i in range(N)]
    Fm = [abs(q) * h_f * sum(aG[i]) * p2(EG) for i in range(N)]
    aV = [[abs(x) for x in c] for c in V]
    aL4 = [st_absD4(c) for c in aV]
    aL2 = [st_absL(c) for c in aV]
    aDl = [[abs(a - b) for a, b in zip(Vp[i], V[i])] for i in range(N)]
    SF = sum(Fm[i] * sum(aDl[i]) for i in range(N)) * p2(EV)
    SA = SB = SC = Fraction(0)
    for i in range(N):
        for k in range(N):
            SA += Am[i][k] / h_s**4 * dot(aL4[k], aDl[i])
            SB += 2 * Bm[i][k] / h_s**2 * dot(aL2[k], aDl[i])
            SC += Cm[i][k] * dot(aV[k], aDl[i])
    sc = abs(D) * p2(2 * EV)
    return {"SF": SF, "SA": SA * sc, "SB": SB * sc, "SC": SC * sc}


def double_counts(n_f: int, n_s: int, N: int) -> dict:
    """Operation counts of the double path (QA derivation, report 5.2; P-1...P-5, pow within 1 ulp)."""
    m_f, m_s = n_f - 1, n_s - 1
    kA, kB, kC, kF = m_f + 2, 2 * m_f + 7, 3 * m_f + 13, m_f + 3
    kL_s, kD4_s = 5, m_s + 11
    return {"kF": kF, "k1": kA + kD4_s + 4, "k2": kB + kL_s + 4, "k3": kC + 2, "M": N * m_s}


def decimal_counts(n_f: int, n_s: int, N: int) -> dict:
    """Operation counts of the decimal path ``decimal_gamma`` (report 5.3)."""
    m_f, m_s = n_f - 1, n_s - 1
    kA, kB, kC, kF = m_f + 2, m_f + 5, m_f + 12, m_f + 3
    tail = 3 * N + 1  # sum of 3N terms, the factor D, the subtraction from F
    return {"kF": kF + 1, "k1": kA + 17 + tail, "k2": kB + 9 + tail, "k3": kC + 1 + tail, "M": N * m_s}


def bound(ms: dict, counts: dict, u: Fraction, h_s: Fraction, g_abs: Fraction, rd: Fraction) -> Fraction:
    """|gamma_computed - gamma| <= gamma_2 h_s |g| + h_s [gamma_M RD + E + gamma_1 (RD + E)] (module docstring); Delta^ replaced by
    the exact Delta with the factor (1 + u) in E and RD (|Delta^| <= (1 + u) |Delta|)."""
    SF, SA, SB, SC = ms["SF"], ms["SA"], ms["SB"], ms["SC"]
    kmax = max(counts["k1"], counts["k2"], counts["k3"])
    g = lambda k: gam(k, u)  # noqa: E731
    E = (1 + u) * (g(counts["kF"]) * SF + g(counts["k1"]) * SA + g(counts["k2"]) * SB + g(counts["k3"]) * SC
                   + g(counts["M"] + 1) * ((1 + g(counts["kF"])) * SF + (1 + g(kmax)) * (SA + SB + SC)))
    RD = (1 + u) * rd
    return g(2) * h_s * g_abs + h_s * (g(counts["M"]) * RD + E + g(1) * (RD + E))


def coefficient_bound(ms: dict, counts: dict, h_s: Fraction) -> Fraction:
    """|g1 - gamma| (binary64 coefficients and operator, exact evaluation): h_s [gamma_kF SF + gamma_k1 SA + gamma_k2 SB + gamma_k3 SC]."""
    return h_s * (gam(counts["kF"]) * ms["SF"] + gam(counts["k1"]) * ms["SA"] + gam(counts["k2"]) * ms["SB"]
                  + gam(counts["k3"]) * ms["SC"])


def premises_double(dp: dict) -> str | None:
    """P-2/P-3 checks of the double path on the arrays that exist (None if they hold): finiteness, no subnormal nonzero entry, no
    possible underflow of the products K s and rho Delta (min nonzero |factors| product >= 2^-1022). Sums never underflow inexactly."""
    arrays = {k: dp[k] for k in ("K", "rhs", "s", "Ks", "rho", "Delta")}
    for name, a in arrays.items():
        if not np.all(np.isfinite(a)):
            return f"P-2: non-finite {name}"
        nz = np.abs(a[a != 0])
        if nz.size and float(nz.min()) < TINY:
            return f"P-3: subnormal entry in {name}"

    def mn(a):
        nz = np.abs(a[a != 0])
        return float(nz.min()) if nz.size else math.inf

    if mn(dp["K"]) * mn(dp["s"]) < TINY or mn(dp["rho"]) * mn(dp["Delta"]) < TINY:
        return "P-3: a product may underflow"
    return None


# ---------------------------------------------------------------------------------------------------------------------------------
# path M2: raised precision (decimal, p digits), the same defining formulas with a fixed sequential order
# ---------------------------------------------------------------------------------------------------------------------------------

def decimal_gamma(lf, nf, ls, ns, given, companion, V, D, q, prec: int):
    """gamma in decimal arithmetic with ``prec`` digits (ROUND_HALF_EVEN); returns (gamma as Fraction, |g|, RD = sum |rho^||Delta|)."""
    ctx = Context(prec=prec, rounding=ROUND_HALF_EVEN, Emin=-999999, Emax=999999)
    d = Decimal  # Decimal(float) is exact (context-free constructor)
    add, sub, mul, div = ctx.add, ctx.subtract, ctx.multiply, ctx.divide
    two = d(2)

    def L(c):
        p = [d(0)] + c + [d(0)]
        return [add(sub(p[k], mul(two, p[k + 1])), p[k + 2]) for k in range(len(c))]

    def D4(c):
        out = L(L(c))
        out[0] = add(out[0], mul(two, c[0]))
        out[-1] = add(out[-1], mul(two, c[-1]))
        return out

    def sdot(a, b):
        acc = d(0)
        for x, y in zip(a, b):
            acc = add(acc, mul(x, y))
        return acc

    N = given.shape[1]
    G = [[d(float(x)) for x in given[:, k]] for k in range(N)]
    Vd = [[d(float(x)) for x in V[:, k]] for k in range(N)]
    Cd = [[d(float(x)) for x in companion[:, k]] for k in range(N)]
    h_f, h_s = div(d(float(lf)), d(nf)), div(d(float(ls)), d(ns))
    Dd, qd = d(float(D)), d(float(q))
    LG, D4G = [L(c) for c in G], [D4(c) for c in G]
    A = [[mul(h_f, sdot(G[i], G[k])) for k in range(N)] for i in range(N)]
    B = [[ctx.minus(div(sdot(G[i], LG[k]), h_f)) for k in range(N)] for i in range(N)]
    h_f3 = mul(mul(h_f, h_f), h_f)
    C = [[div(sdot(G[i], D4G[k]), h_f3) for k in range(N)] for i in range(N)]
    F = []
    for i in range(N):
        acc = d(0)
        for x in G[i]:
            acc = add(acc, x)
        F.append(mul(mul(qd, h_f), acc))
    h_s2 = mul(h_s, h_s)
    h_s4 = mul(mul(h_s2, h_s), h_s)
    c4 = [[div(A[i][k], h_s4) for k in range(N)] for i in range(N)]
    c2 = [[div(mul(two, B[i][k]), h_s2) for k in range(N)] for i in range(N)]
    LV, D4V = [L(c) for c in Vd], [D4(c) for c in Vd]
    g = d(0)
    rd = Fraction(0)
    m = len(Vd[0])
    for i in range(N):
        for j in range(m):
            acc = d(0)
            for k in range(N):
                acc = add(acc, mul(c4[i][k], D4V[k][j]))
                acc = sub(acc, mul(c2[i][k], LV[k][j]))
                acc = add(acc, mul(C[i][k], Vd[k][j]))
            rho = sub(F[i], mul(Dd, acc))
            delta = sub(Cd[i][j], Vd[i][j])
            g = add(g, mul(rho, delta))
            rd += abs(Fraction(rho)) * abs(Fraction(float(companion[j, i])) - Fraction(float(V[j, i])))
    gamma = mul(h_s, g)
    return Fraction(gamma), abs(Fraction(g)), rd


# ---------------------------------------------------------------------------------------------------------------------------------
# K_E statistics: separate test forms (no K_E assigned)
# ---------------------------------------------------------------------------------------------------------------------------------

def k_test_form(dP: Fraction, Q: Fraction, allowance: Fraction, R: Fraction) -> dict:
    """Test form |dP - Q| <= tau, dP >= -tau, tau = allowance + K_E R, resolved iff Q >= K_E tau. Returns the K_E interval
    [K_low, K_max] of a resolved HOLD (empty if K_low > K_max): K_low from E2 and E1, K_max the root of K^2 R + K a - Q = 0."""
    k_e2 = max(Fraction(0), abs(dP - Q) - allowance) / R
    k_e1 = max(Fraction(0), -dP - allowance) / R
    a = float(allowance)
    disc = a * a + 4 * float(R) * float(Q)
    k_max = (-a + math.sqrt(disc)) / (2 * float(R)) if Q > 0 else 0.0
    return {"K_low": max(k_e2, k_e1), "K_E2": k_e2, "K_E1": k_e1, "K_max": k_max}


# ---------------------------------------------------------------------------------------------------------------------------------
# per-sequence job
# ---------------------------------------------------------------------------------------------------------------------------------

def control_rule(rows: list[dict]) -> set:
    """Control sample, FIXED before any new calculation from the Q3 CSV only (report 3.1):
    1) N = 1, 2, 3 x s = 1, 2, 8 on the grids 1x1:32x32 (smallest chain), 1x1:512x512, 1x2:512x1024, 1x0.2:1280x256 (largest chains)
       and the pair 1x1:47x53 with first = X;
    2) the 6 rows with the largest Q3 m_s (E1_m_s_cR22) overall and the 2 largest for each N not yet included;
    3) the 3 rows with the largest |gamma^ - gamma|/|gamma| (Q3 columns) not yet included."""
    pick = set()
    grids = {("chain", "1x1:32x32", "Y"), ("chain", "1x1:512x512", "Y"), ("chain", "1x2:512x1024", "Y"),
             ("chain", "1x0.2:1280x256", "Y"), ("pairs", "1x1:47x53", "X")}
    for r in rows:
        if (r["group"], r["grid"], r["first"]) in grids and r["j_s"] in ("1", "2", "8"):
            pick.add(row_id(r))
    by_m = sorted(rows, key=lambda r: -float(r["E1_m_s_cR22"]))
    pick.update(row_id(r) for r in by_m[:6])
    for N in ("1", "2", "3"):
        extra = [r for r in by_m if r["N"] == N and row_id(r) not in pick][:2]
        pick.update(row_id(r) for r in extra)

    def rel(r):
        gh, ge = read_double(r["j_gamma_double"]), read_double(r["j_gamma_exact"])
        return abs(gh - ge) / abs(ge) if ge != 0 else math.inf

    pick.update(row_id(r) for r in sorted((r for r in rows if row_id(r) not in pick), key=lambda r: -rel(r))[:3])
    return pick


def row_id(r: dict) -> str:
    return f"{r['group']}|{r['grid']}|N{r['N']}|{r['first']}|s{r['j_s']}"


def _steps_from_npz(z, spec) -> list[dict]:
    """(direction, given, companion, V, (Xp, Yp), (Xs, Ys)) of every stored half-step; bitwise consistency checks."""
    nx, ny, N = spec["nx"], spec["ny"], spec["N"]
    out = []
    prev = None
    s = 1
    while f"s{s}_Xs" in z.files:
        direction = "Y" if int(z[f"s{s}_dir"][0]) == 0 else "X"
        Xs, Ys = z[f"s{s}_Xs"], z[f"s{s}_Ys"]
        if direction == "Y":
            given, V = Xs, Ys
            companion = prev[1] if prev is not None else np.zeros((ny - 1, N))
            expected_given = prev[0] if prev is not None else hs.start_set(nx, N)
        else:
            given, V = Ys, Xs
            companion = prev[0] if prev is not None else np.zeros((nx - 1, N))
            expected_given = prev[1] if prev is not None else hs.start_set(ny, N)
        Xp, Yp = prev if prev is not None else (np.zeros((nx - 1, N)), np.zeros((ny - 1, N)))
        out.append({"s": s, "direction": direction, "given": given, "companion": companion, "V": V, "prev": (Xp, Yp),
                    "now": (Xs, Ys), "given_consistent": bool(np.array_equal(given, expected_given))})
        prev = (z[f"s{s}_Xn"], z[f"s{s}_Yn"]) if f"s{s}_Xn" in z.files else None
        s += 1
    return out


def analyse_step(spec: dict, st: dict, row: dict, control: bool, low_prec: bool, path_b_all: bool = False) -> dict:
    lx, nx, ly, ny = spec["lx"], spec["nx"], spec["ly"], spec["ny"]
    D, q = 1.0, 1.0
    direction, given, companion, V = st["direction"], st["given"], st["companion"], st["V"]
    lf, nf, ls, ns = (lx, nx, ly, ny) if direction == "Y" else (ly, ny, lx, nx)
    h_f, h_s = Fraction(lf) / nf, Fraction(ls) / ns
    Dq, qq = Fraction(D), Fraction(q)
    N = given.shape[1]
    out = {"id": row_id(row), "group": spec["group"], "grid": row["grid"], "N": N, "first": spec["first"], "s": st["s"],
           "direction": direction, "n_f": nf, "n_s": ns, "control": control, "given_consistent": st["given_consistent"]}
    # provenance: the stored arrays reproduce the Q3 journal (double values bitwise)
    g = en.Grid(lx, nx, ly, ny, D, q)
    w_s = st["now"][0] @ st["now"][1].T
    out["P_double_s_bitwise"] = en.pi_double(g, w_s) == read_double(row["j_P_double_s"])
    dp = double_path(lx, nx, ly, ny, direction, given, companion, V, D, q)
    gh = dp["gamma_hat"]
    out["gamma_hat_bitwise"] = gh == read_double(row["j_gamma_double"]) and dp["rho_same"]
    out["rhs_is_repeat_F"] = bool(np.array_equal(dp["rhs"], np.repeat(dp["coef"][3], ns - 1)))
    # path A
    (Gi,), EG = dyadic(given)
    (Vi, Vpi), EV = dyadic(V, companion)
    coef = coefficients_exact(Gi, EG, h_f, qq)
    gA = gamma_from_coefficients(coef, Vi, Vpi, EV, h_s, Dq)
    out["gamma_A"] = float(gA)
    out["gamma_A_equals_q3_exact"] = float(gA) == read_double(row["j_gamma_exact"])
    # stages with the binary64 coefficients of the double path taken exactly
    Ad, Bd, Cd, Fd = dp["coef"]
    coef_d = ([[Fraction(float(Ad[i, k])) for k in range(N)] for i in range(N)],
              [[Fraction(float(Bd[i, k])) for k in range(N)] for i in range(N)],
              [[Fraction(float(Cd[i, k])) for k in range(N)] for i in range(N)], [Fraction(float(x)) for x in Fd])
    g_dc = gamma_from_coefficients(coef_d, Vi, Vpi, EV, h_s, Dq)
    Dl = [[a - b for a, b in zip(Vpi[i], Vi[i])] for i in range(N)]
    parts = stages(dp, gA, g_dc, Dl, EV, h_s)
    err = Fraction(gh) - gA
    out["gamma_hat"] = gh
    out["err_hat"] = float(err)
    for k, v in parts.items():
        out[f"part_{k}"] = float(v)
    # majorants, model scale and bounds
    ms = majorant_sums(Gi, EG, Vi, Vpi, EV, h_f, h_s, Dq, qq)
    S_gamma = h_s * (ms["SF"] + ms["SA"] + ms["SB"] + ms["SC"])
    out["S_gamma"] = float(S_gamma)
    out["err_hat_over_u_S"] = float(abs(err) / (U * S_gamma)) if S_gamma else math.inf
    rd = abs(Fraction(_frac_vec_dot(np.abs(dp["rho"]), np.abs(dp["Delta"]))))
    why = premises_double(dp)
    cnt = double_counts(nf, ns, N)
    if why is None:
        Bg = bound(ms, cnt, U, h_s, abs(Fraction(dp["g"])), rd)
        out["B_gamma"], out["B_gamma_status"] = en.up(Bg), "CONDITIONAL (P-1...P-5, pow, P-4, counts by code)"
        out["err_hat_over_B"] = float(abs(err) / Bg)
        out["B_holds"] = abs(err) <= Bg
    else:
        Bg = None
        out["B_gamma"], out["B_gamma_status"], out["err_hat_over_B"], out["B_holds"] = None, f"{UNRESOLVED}: {why}", None, None
    Bc = coefficient_bound(ms, cnt, h_s)
    out["B_coef"] = en.up(Bc)
    out["coef_err"] = float(g_dc + parts["operator_assembly"] - gA)  # g1 - gamma: binary64 operator, exact evaluation
    out["coef_err_over_B_coef"] = float(abs(g_dc + parts["operator_assembly"] - gA) / Bc)
    # decimal path M2
    gD, gabs, rdD = decimal_gamma(lf, nf, ls, ns, given, companion, V, D, q, DEC_P)
    uP = Fraction(1, 2 * 10 ** (DEC_P - 1))
    BD = bound(ms, decimal_counts(nf, ns, N), uP, h_s, gabs, rdD)
    out["gamma_dec34"] = float(gD)
    out["err_dec34"] = float(gD - gA)
    out["B_dec34"] = en.up(BD)
    out["dec34_holds"] = abs(gD - gA) <= BD
    out["err_dec34_over_B"] = float(abs(gD - gA) / BD) if BD else 0.0
    if low_prec:
        for pr in DEC_LOW:
            gL, gabsL, rdL = decimal_gamma(lf, nf, ls, ns, given, companion, V, D, q, pr)
            uL = Fraction(1, 2 * 10 ** (pr - 1))
            try:
                BL = bound(ms, decimal_counts(nf, ns, N), uL, h_s, gabsL, rdL)
                out[f"dec{pr}_err_over_B"] = float(abs(gL - gA) / BL)
                out[f"dec{pr}_holds"] = abs(gL - gA) <= BL
            except ValueError:
                out[f"dec{pr}_err_over_B"], out[f"dec{pr}_holds"] = None, "NOT_APPLICABLE (k u_p >= 1)"
    # path B (control sample; with --path-b all also on every other row as an extended check)
    if control or path_b_all:
        Pp, Ps, Qe = energies_path_b(lx, nx, ly, ny, D, q, *st["prev"], *st["now"])
        gB = Qe - (Pp - Ps)
        out["gamma_B"] = float(gB)
        out["A_equals_B"] = gA == gB
        out["P_s_B_in_q3_interval"] = read_bound(row["j_P_exact_s_down"]) <= Ps <= read_bound(row["j_P_exact_s_up"])
        out["Q_B_float_equals_q3"] = float(Qe) == read_double(row["j_Q_exact"])
    # energy test data (Q3 journal values; dP^, Q^ exact rationals of the stored doubles)
    dP = Fraction(read_double(row["j_P_double_prev"])) - Fraction(read_double(row["j_P_double_s"]))
    Qh = Fraction(read_double(row["j_Q_double"]))
    R = read_bound(row["j_R_base_up"]) + C_R * read_bound(row["j_R_per_cR_up"]) + read_bound(row["j_R_f_up"])
    defect = dP - Qh
    out["R_sum_cR22"] = float(R)
    out["Q_over_R"] = float(Qh / R)
    out["defect_over_R"] = float(defect / R)
    out["dP_negative"] = dP < 0
    out["coef_err_over_R"] = float(abs(g_dc + parts["operator_assembly"] - gA) / R)
    out["gamma_over_R"] = float(gA / R)
    out["err_hat_over_R"] = float(err / R)
    out["B_gamma_over_R"] = float(Bg / R) if Bg is not None else None
    out["B_dec34_over_R"] = float(BD / R)
    # (ii) identity control with gamma^ (Q3 m_s with this R), (iii) with exact gamma, gamma^ + bound, decimal + bound
    out["K_id_hat"] = float(abs(defect + Fraction(gh)) / R)
    out["K_id_exact"] = float(abs(defect + gA) / R)
    out["K_id_hat_cert"] = float(max(Fraction(0), abs(defect + Fraction(gh)) - Bg) / R) if Bg is not None else None
    out["K_id_dec34_cert"] = float(max(Fraction(0), abs(defect + gD) - BD) / R)
    # (i) test form with the different allowances
    variants = {"hat": abs(Fraction(gh)), "exact": abs(gA), "dec34_cert": abs(gD) + BD}
    if Bg is not None:
        variants["hat_cert"] = abs(Fraction(gh)) + Bg
    for name, allowance in variants.items():
        k = k_test_form(dP, Qh, allowance, R)
        out[f"T_{name}_K_low"] = float(k["K_low"])
        out[f"T_{name}_K_max"] = k["K_max"]
    return out


def seq_job(spec: dict, rows: list[dict], npz_path: str, control_ids: list[str], low_prec_ids: list[str],
            path_b_all: bool = False) -> dict:
    t0 = time.perf_counter()
    z = np.load(npz_path)
    steps = _steps_from_npz(z, spec)
    by_s = {int(r["j_s"]): r for r in rows}
    out = []
    for st in steps:
        row = by_s[st["s"]]
        rid = row_id(row)
        out.append(analyse_step(spec, st, row, rid in control_ids, rid in low_prec_ids, path_b_all))
    return {"rows": out, "spec": spec, "npz": str(npz_path), "npz_sha256": sha256(Path(npz_path)), "steps": len(steps),
            "seconds": round(time.perf_counter() - t0, 2), "peak_mb": _q3().peak_mb()}


# ---------------------------------------------------------------------------------------------------------------------------------
# step 1: arithmetic of the 984 Q3 rows (control of the stored data, NOT an independent reference)
# ---------------------------------------------------------------------------------------------------------------------------------

def q3_arithmetic(rows: list[dict], summary: dict | None) -> dict:
    res = {"rows": len(rows), "repr_fields_round_trip": 0, "checks": {}, "stats": {}}
    repr_cols = ("j_gamma_exact", "j_gamma_double", "j_P_double_prev", "j_P_double_s", "j_Q_double", "j_defect_computed",
                 "j_m_s_numerator", "j_part_solution", "j_Q_exact")
    ok = {k: 0 for k in ("repr", "defect", "m_num", "part_solution", "gamma_err", "m_s_cR22_rel4u", "m_s_cR22_bitwise",
                         "Q_over_tau_rel4u", "parts_sum_rel")}
    m_s, m_hat_gt1, per_s_unres = [], {}, {}
    norm = {"err_rel_hat": [], "gamma_over_R": [], "gamma_le_hat": 0, "test_form_K": [], "identity_exact_K": []}
    for r in rows:
        if all(repr(float(r[c])) == r[c] for c in repr_cols):
            ok["repr"] += 1
        Pp, Ps, Qh = (Fraction(read_double(r[c])) for c in ("j_P_double_prev", "j_P_double_s", "j_Q_double"))
        gh, ge = read_double(r["j_gamma_double"]), read_double(r["j_gamma_exact"])
        defect = Pp - Ps - Qh
        ok["defect"] += float(defect) == read_double(r["j_defect_computed"])
        ok["m_num"] += float(abs(defect + Fraction(gh))) == read_double(r["j_m_s_numerator"])
        ok["part_solution"] += read_double(r["j_part_solution"]) == -ge
        ge_ulp = math.ulp(ge)
        ok["gamma_err"] += abs(read_double(r["j_gamma_computation_error"]) - (gh - ge)) <= 2 * ge_ulp + math.ulp(gh - ge)
        parts = sum(read_double(r[c]) for c in ("j_part_solution", "j_part_formation", "j_part_assembly"))
        ok["parts_sum_rel"] += abs(parts - float(defect)) <= 8 * 2**-53 * max(abs(float(defect)), abs(parts), 1e-300) \
            + 4 * max(math.ulp(read_double(r[c])) for c in ("j_part_solution", "j_part_formation", "j_part_assembly"))
        den22 = float(r["j_R_base_up"]) + C_R * float(r["j_R_per_cR_up"]) + float(r["j_R_f_up"])
        m = read_double(r["j_m_s_numerator"]) / den22
        stored = float(r["E1_m_s_cR22"])
        ok["m_s_cR22_bitwise"] += m == stored
        ok["m_s_cR22_rel4u"] += abs(m - stored) <= 4 * 2**-53 * abs(stored)
        qt = read_double(r["j_Q_double"]) / (abs(gh) + den22)
        ok["Q_over_tau_rel4u"] += abs(qt - float(r["E1_Q_over_tau_KE1_cR22"])) <= 4 * 2**-53 * abs(qt)
        m_s.append((int(r["N"]), int(r["j_s"]), stored))
        if stored > 1:
            m_hat_gt1[int(r["j_s"])] = m_hat_gt1.get(int(r["j_s"]), 0) + 1
        if read_double(r["j_Q_double"]) < abs(gh) + den22:
            per_s_unres[int(r["j_s"])] = per_s_unres.get(int(r["j_s"]), 0) + 1
        R = read_bound(r["j_R_base_up"]) + C_R * read_bound(r["j_R_per_cR_up"]) + read_bound(r["j_R_f_up"])
        norm["err_rel_hat"].append(abs(gh - ge) / abs(gh))
        norm["gamma_over_R"].append(float(abs(Fraction(ge)) / R))
        norm["gamma_le_hat"] += abs(ge) <= abs(gh)
        norm["test_form_K"].append(float(max(Fraction(0), abs(defect) - abs(Fraction(gh))) / R))
        norm["identity_exact_K"].append(float(abs(defect + Fraction(ge)) / R))
    res["checks"] = ok
    for N in (1, 2, 3):
        v = [m for n, _, m in m_s if n == N]
        res["stats"][f"m_s_cR22_N{N}"] = stats(v)
        if summary is not None:
            ref = summary.get(f"E1_m_s_cR22_N{N}", {})
            res["stats"][f"m_s_cR22_N{N}_equals_summary"] = all(
                res["stats"][f"m_s_cR22_N{N}"][k] == ref.get(k) for k in ("count", "min", "median", "q95", "max"))
    res["stats"]["m_s_by_s_median"] = {s: statistics.median([m for _, ss, m in m_s if ss == s]) for s in sorted({x[1] for x in m_s})}
    res["stats"]["m_s_gt_1_by_s"] = dict(sorted(m_hat_gt1.items()))
    res["stats"]["unresolved_Q_lt_tau_KE1_by_s"] = dict(sorted(per_s_unres.items()))
    res["stats"]["err_rel_hat_range"] = [min(norm["err_rel_hat"]), max(norm["err_rel_hat"])]
    res["stats"]["max_gamma_over_R"] = max(norm["gamma_over_R"])
    res["stats"]["gamma_le_gamma_hat"] = norm["gamma_le_hat"]
    res["stats"]["max_test_form_K_hat"] = max(norm["test_form_K"])
    res["stats"]["max_identity_K_exact"] = max(norm["identity_exact_K"])
    return res


def stats(values) -> dict:
    v = sorted(float(x) for x in values if x is not None and not isinstance(x, (bool, str)) and math.isfinite(float(x)))
    if not v:
        return {"count": 0}
    q = lambda f: v[min(len(v) - 1, int(math.ceil(f * len(v))) - 1)]  # noqa: E731 - nearest rank (as Q3)
    return {"count": len(v), "min": v[0], "median": statistics.median(v), "q95": q(0.95), "q99": q(0.99), "max": v[-1]}


# ---------------------------------------------------------------------------------------------------------------------------------
# campaign
# ---------------------------------------------------------------------------------------------------------------------------------

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=False).stdout.strip()


def provenance(csv_path: Path) -> dict:
    rel = csv_path.relative_to(ROOT).as_posix()
    blob_now = git("hash-object", str(csv_path))
    return {"csv": rel, "sha256": sha256(csv_path), "blob": blob_now, "blob_in_HEAD": git("rev-parse", f"HEAD:{rel}"),
            "blob_in_q3_campaign_commit": git("rev-parse", f"{Q3_DATA_COMMIT}:{rel}"),
            "unchanged_since_q3_campaign": blob_now == git("rev-parse", f"{Q3_DATA_COMMIT}:{rel}")}


def fmt(v):
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, Fraction):
        return repr(float(v))
    if isinstance(v, (float, np.floating)):
        return repr(float(v))
    return "" if v is None else v


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: fmt(r.get(k)) for k in keys})


def summarize(rows: list[dict]) -> dict:
    def col(k, sub=None):
        return [r[k] for r in (sub if sub is not None else rows) if isinstance(r.get(k), (int, float)) and not isinstance(r.get(k), bool)]

    out = {"rows": len(rows)}
    for k in ("given_consistent", "P_double_s_bitwise", "gamma_hat_bitwise", "rhs_is_repeat_F", "gamma_A_equals_q3_exact",
              "B_holds", "dec34_holds"):
        out[k] = sum(1 for r in rows if r.get(k) is True)
    ctrl = [r for r in rows if r["control"]]
    out["control_rows"] = len(ctrl)
    for k in ("A_equals_B", "P_s_B_in_q3_interval", "Q_B_float_equals_q3"):
        out[f"control_{k}"] = sum(1 for r in ctrl if r.get(k) is True)
        out[f"all_{k}"] = f"{sum(1 for r in rows if r.get(k) is True)}/{sum(1 for r in rows if k in r)}"
    out["dP_negative_rows"] = sum(1 for r in rows if r.get("dP_negative") is True)
    out["B_gamma_unresolved"] = sum(1 for r in rows if str(r.get("B_gamma_status", "")).startswith(UNRESOLVED))
    for k in ("coef_err_over_R", "err_hat_over_u_S", "err_hat_over_B", "coef_err_over_B_coef", "err_dec34_over_B", "err_hat_over_R",
              "B_gamma_over_R", "B_dec34_over_R", "gamma_over_R", "Q_over_R", "K_id_hat", "K_id_exact", "K_id_hat_cert", "K_id_dec34_cert"):
        vals = col(k)
        if k in ("err_hat_over_R", "gamma_over_R"):
            vals = [abs(v) for v in vals]
        out[k] = stats(vals)
    # share of the parts in |gamma^ - gamma|
    for p in ("coefficients", "operator_assembly", "matvec", "subtraction", "delta_rounding", "dot_product", "h_multiplication"):
        out[f"part_{p}_over_err"] = stats(abs(r[f"part_{p}"]) / abs(r["err_hat"]) for r in rows if r["err_hat"] != 0)
    for pr in DEC_LOW:
        lows = [r for r in rows if f"dec{pr}_holds" in r]
        out[f"dec{pr}_bound_holds"] = f"{sum(1 for r in lows if r[f'dec{pr}_holds'] is True)}/{len(lows)}"
        out[f"dec{pr}_err_over_B"] = stats(col(f"dec{pr}_err_over_B", lows))
    # by s and N
    for s in sorted({r["s"] for r in rows}):
        sub = [r for r in rows if r["s"] == s]
        out[f"s{s}"] = {k: stats(col(k, sub)) for k in ("K_id_hat", "K_id_exact", "K_id_hat_cert", "B_gamma_over_R",
                                                         "B_dec34_over_R", "Q_over_R", "err_hat_over_R", "err_hat_over_u_S",
                                                         "coef_err_over_R", "T_hat_K_low", "T_exact_K_low", "T_hat_cert_K_low",
                                                         "T_dec34_cert_K_low", "T_hat_K_max", "T_exact_K_max",
                                                         "T_hat_cert_K_max", "T_dec34_cert_K_max")}
        out[f"s{s}"]["empty_hold_interval"] = {v: sum(1 for r in sub if r.get(f"T_{v}_K_low") is not None
                                                      and r[f"T_{v}_K_low"] > r[f"T_{v}_K_max"])
                                               for v in ("hat", "exact", "hat_cert", "dec34_cert")}
    for N in (1, 2, 3):
        sub = [r for r in rows if r["N"] == N]
        out[f"N{N}"] = {k: stats(col(k, sub)) for k in ("K_id_hat", "K_id_exact", "K_id_hat_cert", "B_gamma_over_R", "err_hat_over_B")}
    # classification as a function of K_E: false HOLD and lost resolution of the test form against the exact allowance
    fh, lost_hat, lost_cert = [], [], []
    for r in rows:
        if r["T_hat_K_low"] < r["T_exact_K_low"] and r["T_hat_K_low"] <= r["T_hat_K_max"]:
            fh.append((r["T_hat_K_low"], min(r["T_exact_K_low"], r["T_hat_K_max"])))
        if r["T_hat_K_max"] < r["T_exact_K_max"]:
            lost_hat.append((r["T_hat_K_max"], r["T_exact_K_max"]))
        if r.get("T_hat_cert_K_max") is not None and r["T_hat_cert_K_max"] < r["T_exact_K_max"]:
            lost_cert.append((r["T_hat_cert_K_max"], r["T_exact_K_max"]))
    out["false_hold_rows"] = len(fh)
    out["false_hold_K_E_upper_max"] = max((b for _, b in fh), default=None)
    out["lost_resolution_hat_rows"] = len(lost_hat)
    out["lost_resolution_hat_max_rel"] = max(((b - a) / b for a, b in lost_hat), default=None)
    out["lost_resolution_cert_rows"] = len(lost_cert)
    out["lost_resolution_cert_max_rel"] = max(((b - a) / b for a, b in lost_cert), default=None)
    return out


def ci_data(tmp: Path) -> tuple[Path, Path]:
    """Deterministic small CI data in the Q3 format: Q3 seq_job on small grids (N = 1-3, both first directions)."""
    q3 = _q3()
    specs = [{"group": "ci", "r": "1.3", "n": 9, "lx": 1.0, "nx": 10, "ly": 1.3, "ny": 9, "N": N, "first": "Y", "s_max": 4}
             for N in (1, 2, 3)]
    specs += [{"group": "ci", "r": "0.6", "n": 9, "lx": 1.0, "nx": 12, "ly": 0.6, "ny": 9, "N": 2, "first": "X", "s_max": 4}]
    arrays = tmp / "arrays"
    rows = []
    for spec in specs:
        rows += q3.seq_job(spec, str(arrays))["seq"]
    path = tmp / "sequences.csv"
    q3.write_csv(path, rows)
    return path, arrays


def run_campaign(profile: str, out: Path, workers: int, arrays: Path, path_b_all: bool = False) -> dict:
    t0 = time.time()
    out.mkdir(parents=True, exist_ok=True)
    q3 = _q3()
    tmp = None
    if profile == "ci":
        tmp = Path(tempfile.mkdtemp(prefix="r5q4ci_"))
        csv_path, arrays = ci_data(tmp)
        summary_q3 = None
        prov = {"csv": "generated by the Q3 seq_job (CI profile)", "sha256": sha256(csv_path)}
    else:
        csv_path = Q3_CSV
        summary_q3 = json.loads((Q3_DIR / "summary.json").read_text(encoding="utf-8"))
        prov = provenance(csv_path)
    rows = read_q3_rows(csv_path)
    arith = q3_arithmetic(rows, summary_q3)
    control = control_rule(rows) if profile == "full" else {row_id(r) for r in rows}
    low = {i for i in control if i.split("|")[1] in ("1x1:32x32", "1x1:47x53", "1x1.3:10x9", "1x0.6:12x9")}
    groups = {}
    for r in rows:
        groups.setdefault(seq_key(r), []).append(r)
    jobs = []
    for rs in groups.values():
        spec = spec_of(rs[0]) if profile == "full" else _ci_spec(rs[0])
        npz = arrays / q3.seq_file(spec)
        jobs.append((spec, rs, str(npz), sorted(control), sorted(low), path_b_all))
    jobs.sort(key=lambda j: -(j[0]["nx"] * j[0]["ny"] * j[0]["N"]))  # largest first
    if workers > 1:
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(workers) as pool:
            results = pool.starmap(seq_job, jobs, chunksize=1)
    else:
        results = [seq_job(*j) for j in jobs]
    rows_out = [r for res in results for r in res["rows"]]
    rows_out.sort(key=lambda r: (r["group"], r["grid"], r["N"], r["first"], r["s"]))
    write_csv(out / "gamma_rows.csv", rows_out)
    summ = summarize(rows_out)
    summ["q3_arithmetic"] = arith
    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summ, f, indent=1, ensure_ascii=False, default=_json)
        f.write("\n")
    meta = {
        "study": "R5-V Q4 priority 1 (D-053.4): gamma_s of CCCC; QA research tool, not a qualification; K_E not assigned",
        "profile": profile, "git_commit": git("rev-parse", "HEAD"), "git_status_porcelain": git("status", "--porcelain").splitlines(),
        "python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__, "platform": platform.platform(),
        "machine": platform.machine(), "processor": platform.processor(),
        "threads": {k: os.environ.get(k) for k in THREAD_VARS}, "workers": workers, "decimal_precision": DEC_P,
        "path_b": "all rows (extended check) + fixed control sample" if path_b_all else "fixed control sample",
        "decimal_low_precisions_bound_test": list(DEC_LOW), "c_R_denominator": C_R,
        "data": prov, "q3_arrays_dir": str(arrays) if profile == "full" else "temporary (CI)",
        "q3_arrays_used_read_only": True, "npz": [{"file": Path(r["npz"]).name, "sha256": r["npz_sha256"]} for r in results],
        "control_rule": control_rule.__doc__, "control_ids": sorted(control), "low_precision_ids": sorted(low),
        "jobs": [{"grid": r["spec"].get("nx"), "ny": r["spec"].get("ny"), "N": r["spec"]["N"], "first": r["spec"]["first"],
                  "seconds": r["seconds"], "peak_mb": r["peak_mb"]} for r in results],
        "resource_rule_PL_6_3": {"peak_job_mb": max((r["peak_mb"] or 0) for r in results),
                                 "limit_mb_50pct_ram": (q3.total_ram_mb() or 0) / 2,
                                 "longest_job_s": max(r["seconds"] for r in results), "limit_s": 1800},
        "seconds_total": round(time.time() - t0, 1),
    }
    with open(out / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1, ensure_ascii=False, default=_json)
        f.write("\n")
    return {"summary": summ, "meta": meta}


def _ci_spec(row: dict) -> dict:
    ly = {"1.3": 1.3, "0.6": 0.6}[row["r"]]
    return {"group": "ci", "r": row["r"], "n": row["n"], "lx": 1.0, "nx": int(row["nx"]), "ly": ly, "ny": int(row["ny"]),
            "N": int(row["N"]), "first": row["first"]}


def _json(o):
    if isinstance(o, (np.bool_, np.integer, np.floating)):
        return o.item()
    if isinstance(o, Fraction):
        return float(o)
    raise TypeError(type(o).__name__)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--profile", choices=("full", "ci"), default="full")
    r.add_argument("--out", type=Path, default=None)
    r.add_argument("--workers", type=int, default=8)
    r.add_argument("--arrays", type=Path, default=Q3_ARRAYS_DEFAULT)
    r.add_argument("--path-b", choices=("control", "all"), default="control",
                   help="rows of the exact 2-D path B: the fixed control sample, or every row (extended check)")
    a = ap.parse_args(argv)
    out = a.out or (OUT if a.profile == "full" else ROOT / "scratch" / "r5_cccc_q4_ci")
    res = run_campaign(a.profile, out, a.workers if a.profile == "full" else 1, a.arrays, a.path_b == "all")
    s = res["summary"]
    print(json.dumps({k: s[k] for k in ("rows", "given_consistent", "gamma_hat_bitwise", "gamma_A_equals_q3_exact", "B_holds",
                                         "dec34_holds", "control_rows", "control_A_equals_B")}, default=_json))
    ok = (s["given_consistent"] == s["rows"] == s["gamma_hat_bitwise"] == s["gamma_A_equals_q3_exact"]
          and s["control_A_equals_B"] == s["control_rows"] and s["dec34_holds"] == s["rows"])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
