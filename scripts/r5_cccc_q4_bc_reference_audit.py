"""R5-V Q4, priority 2 (D-053.4, item 2): a raised-precision certified reference of the CCCC half-step and the actual error of the
research B-C on the E-4 sample of Q3 with the reference budget 1/100 (D-049.6, item 4). QA / research tool, not a production API; no
import of the working ``kkvim`` (D-028). The existing ``kkvim_reference`` modules, the Q3 script and the Q3 data are used read-only.

Problem. For the half-step system actually solved by B-C (``r5_cccc_bc.run_halfstep``: after QR the solved set U~), the FINITE
DISCRETE problem with binary64 inputs: the given set, l_x, l_y, D, q taken exactly, h = l/n exactly (NS 4.2-4.3):

    K x = b,   K = D [A (x) D4 - 2 B (x) L + C (x) I],   b = F (x) 1,   A = h_f G^T G, B = -h_f G^T L_f G, C = h_f G^T D4_f G,
    F = q h_f G^T 1,   L = (1, -2, 1)/h^2,   D4 = L^2 + (2/h^4)(e_1 e_1^T + e_m e_m^T)  (boundary rows (7, -4, 1)/h^4).

Reference (does not call B-C, its modal solution or the DST):
1. K and b are assembled EXACTLY (rational, own implementation: integer stencils, exact coefficients; never rounded to double),
   node-term ordering j N + i, half bandwidth 3N - 1;
2. a banded Cholesky factor of K rounded to p decimal digits (``decimal``, p from the ladder P_LADDER), forward/back substitution;
3. iterative refinement: r = b - K x EXACTLY (rational, the original operator), correction by the decimal factor, x accumulated as an
   exact rational; refinement stops when r = 0, when ||r||_inf does not fall by at least STAGNATION, or after MAX_REFINE steps;
4. CERTIFICATE (not the last correction): lambda_low < lambda_min(K) from the Q1b certificate
   (r5_cccc_halfstep.certify_lambda_min: decimal Cholesky of K - t I, exact defect E, lambda_min(K) > t - ||E||_inf), t = lambda_est/2
   (the Q3 rule); then for the exact solution x*
       ||x - x*||_2 <= ||r||_2 / lambda_low,     ||x - x*||_K <= ||r||_2 / lambda_low^(1/2),
   since x - x* = -K^-1 r, ||K^-1||_2 = 1/lambda_min, ||K^-1/2||_2 = lambda_min^-1/2; the field of the solved set:
   ||U~(V - V*)^T||_a = h_s^(1/2) ||x - x*||_K (NS 4.3, P-S1). Square roots are bounded by integer square roots, every bound rounded
   outwards (directed) and printed by directed 17-digit decimals;
5. precision rule (fixed before the campaign): p = 40; if the factorization fails, or the budget 1/100 is not met in the set or the
   field norm, the next p of P_LADDER is used; after the last p the point is UNRESOLVED. Without a certificate: UNRESOLVED.

Error of B-C (x^ = its binary64 set, term-first, permuted exactly): d = ||x^ - x||_2 and d_K^2 = (x^ - x)^T (r(x) - r(x^)) exactly,
true errors in [d - b, d + b] and [d_K - b_K, d_K + b_K]; budget: 100 b <= max(0, d - b) in the same norm (exact comparison), else
UNRESOLVED. Relative values use certified bounds of ||x*||_2 and ||x*||_K (||x||_K^2 = x^T (b - r) exactly).

Independent paths (same discrete problem, different algorithms): the exact rational solution on N m <= 36 (Q3 exact tier, own
exact elimination is NOT used: r5_cccc_q3_e4e6 exact system), and R-H (restriction of the two-dimensional Hessian, double, certified
by its own exact residual) on grids with (n_x - 1)(n_y - 1) <= RH_LIMIT.

N >= 2: RESEARCH_ONLY / PROVISIONAL / NOT_QUALIFIED; all certificates refer to the SOLVED set (after QR), not to the original set.

Usage (repository root):
    uv run python scripts/r5_cccc_q4_bc_reference_audit.py probe
    uv run python scripts/r5_cccc_q4_bc_reference_audit.py run --profile full [--workers 8]
    uv run python scripts/r5_cccc_q4_bc_reference_audit.py run --profile ci --out scratch/r5_cccc_q4_bc_ci
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
import subprocess
import sys
import time
from decimal import ROUND_HALF_EVEN, Context, Decimal
from fractions import Fraction
from pathlib import Path

THREAD_VARS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
for _k in THREAD_VARS:
    os.environ[_k] = "1"  # as in Q3: B-C is reproduced bitwise with one BLAS thread

import numpy as np  # noqa: E402
import scipy  # noqa: E402
import scipy.linalg  # noqa: E402

from kkvim_reference import r5_cccc_bc as bc  # noqa: E402
from kkvim_reference import r5_cccc_halfstep as hs  # noqa: E402
from kkvim_reference import r5_cccc_q3_e4e6 as e46  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "results" / "r5_cccc_q4_bc_reference_audit"
Q3_DIR = ROOT / "docs" / "results" / "r5_cccc_q3_prequalification"
Q3_E4 = Q3_DIR / "e4.csv"
Q3_SCRIPT = ROOT / "scripts" / "r5_cccc_q3_prequalification.py"
Q3_DATA_COMMIT = "98631164b164b7a8852fb00069f691cf8aafbbfb"
Q3_ARRAYS_DEFAULT = Path(os.environ.get("KKVIM_Q3_ARRAYS", r"D:\git\KK-VIM-Q3-QA\scratch\r5_cccc_q3_prequalification\arrays"))
P_LADDER = (40, 80, 160)  # decimal digits of the factor (rule fixed before the campaign)
MAX_REFINE = 6
STAGNATION = 10  # a refinement step must reduce ||r||_inf at least by this factor, otherwise refinement stops
RH_LIMIT = 4096  # (n_x - 1)(n_y - 1) up to which R-H (dense 2-D Hessian) is compared
EXACT_LIMIT = e46.EXACT_LIMIT  # 36: Q3 exact tier
RESEARCH = "RESEARCH_ONLY / PROVISIONAL / NOT_QUALIFIED (D-049.5)"
UNRESOLVED = "UNRESOLVED"


def _q3():
    if not hasattr(_q3, "mod"):
        spec = importlib.util.spec_from_file_location("r5_cccc_q3_prequalification", Q3_SCRIPT)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _q3.mod = mod
    return _q3.mod


# ---------------------------------------------------------------------------------------------------------------------------------
# directed rationals
# ---------------------------------------------------------------------------------------------------------------------------------

def sqrt_down(x: Fraction) -> Fraction:
    if x <= 0:
        return Fraction(0)
    P, Q = x.numerator, x.denominator
    return Fraction(math.isqrt(P * Q * 4**64), Q * 2**64)


def sqrt_up(x: Fraction) -> Fraction:
    if x <= 0:
        return Fraction(0)
    P, Q = x.numerator, x.denominator
    return Fraction(math.isqrt(P * Q * 4**64) + 1, Q * 2**64)


def up(x: Fraction) -> float:
    return hs._float_up(x)


def down(x: Fraction) -> float:
    return e46._down(x)


def dec_up(x: Fraction) -> str:
    return hs.decimal_up(up(x), 17) if x > 0 else "0.0000000000000000e+00"


def dec_down(x: Fraction) -> str:
    return hs.decimal_down(down(x), 17) if x > 0 else "0.0000000000000000e+00"


# ---------------------------------------------------------------------------------------------------------------------------------
# exact system (own implementation; equality with the accepted Q1b band and R-HP is tested)
# ---------------------------------------------------------------------------------------------------------------------------------

def _st_L(c):
    p = [0] + list(c) + [0]
    return [p[k] - 2 * p[k + 1] + p[k + 2] for k in range(len(c))]


def _st_D4(c):
    out = _st_L(_st_L(c))
    out[0] += 2 * c[0]
    out[-1] += 2 * c[-1]
    return out


def exact_coefficients(p: hs.HalfStep):
    """A, B, C, F of the given binary64 set exactly (integers with one binary exponent; NS 4.2)."""
    G = np.asarray(p.given, dtype=float)
    ratios = [[float(G[i, k]).as_integer_ratio() for i in range(G.shape[0])] for k in range(p.N)]
    K2 = max((q.bit_length() - 1 for col in ratios for _, q in col), default=0)
    cols = [[a << (K2 - (q.bit_length() - 1)) for a, q in col] for col in ratios]
    s = Fraction(1, 2 ** (2 * K2))
    h = Fraction(p.l_f) / p.n_f
    dot = lambda a, b: sum(x * y for x, y in zip(a, b, strict=True))  # noqa: E731
    L = [_st_L(c) for c in cols]
    D4 = [_st_D4(c) for c in cols]
    N = p.N
    A = [[h * dot(cols[i], cols[k]) * s for k in range(N)] for i in range(N)]
    B = [[-dot(cols[i], L[k]) * s / h for k in range(N)] for i in range(N)]
    C = [[dot(cols[i], D4[k]) * s / h**3 for k in range(N)] for i in range(N)]
    F = [Fraction(p.q) * h * sum(cols[i]) / 2**K2 for i in range(N)]
    return A, B, C, F


def exact_band(p: hs.HalfStep, coef) -> dict:
    """Upper band of K (node-term ordering j N + i) as {(row, col): Fraction}, zero entries omitted."""
    A, B, C, _ = coef
    N, m = p.N, p.n_s - 1
    h = Fraction(p.l_s) / p.n_s
    D = Fraction(p.D)
    h2, h4 = h * h, h**4
    out = {}
    for j in range(m):
        for dj in (0, 1, 2):
            jj = j + dj
            if jj >= m:
                continue
            d4 = (7 if j in (0, m - 1) else 6) if dj == 0 else (-4 if dj == 1 else 1)
            lv = -2 if dj == 0 else (1 if dj == 1 else 0)
            iv = 1 if dj == 0 else 0
            for i in range(N):
                for k in range(N):
                    row, col = j * N + i, jj * N + k
                    if col < row:
                        continue
                    v = D * (A[i][k] * d4 / h4 - 2 * B[i][k] * lv / h2 + C[i][k] * iv)
                    if v != 0:
                        out[(row, col)] = v
    return out


def band_rows(band: dict, n: int) -> list[list[tuple[int, Fraction]]]:
    """Full symmetric rows of K as lists of (col, value)."""
    rows = [[] for _ in range(n)]
    for (r, c), v in band.items():
        rows[r].append((c, v))
        if c != r:
            rows[c].append((r, v))
    return rows


def rhs_exact(p: hs.HalfStep, coef) -> list[Fraction]:
    """b = F (x) 1 in the node-term ordering."""
    F = coef[3]
    return [F[i] for _ in range(p.n_s - 1) for i in range(p.N)]


def residual(rows, b, x) -> list[Fraction]:
    """r = b - K x exactly (rational)."""
    return [b[i] - sum(v * x[c] for c, v in rows[i]) for i in range(len(b))]


def to_node_term(s_term_first, N: int, m: int) -> list:
    """Term-first (i m + j) -> node-term (j N + i), exact (no arithmetic)."""
    return [s_term_first[i * m + j] for j in range(m) for i in range(N)]


# ---------------------------------------------------------------------------------------------------------------------------------
# raised-precision banded Cholesky
# ---------------------------------------------------------------------------------------------------------------------------------

def dec_cholesky(band: dict, n: int, pb: int, prec: int):
    """Upper banded factor R (R^T R ~ K) in decimal arithmetic with ``prec`` digits; None if a pivot is not positive."""
    ctx = Context(prec=prec, rounding=ROUND_HALF_EVEN, Emin=-999999, Emax=999999)
    a = {k: ctx.divide(Decimal(v.numerator), Decimal(v.denominator)) for k, v in band.items()}
    R = [[Decimal(0)] * (pb + 1) for _ in range(n)]
    for i in range(n):
        s = a.get((i, i), Decimal(0))
        for k in range(max(0, i - pb), i):
            s = ctx.subtract(s, ctx.multiply(R[k][i - k], R[k][i - k]))
        if s <= 0:
            return None
        rii = ctx.sqrt(s)
        R[i][0] = rii
        for j in range(i + 1, min(n, i + pb + 1)):
            s = a.get((i, j), Decimal(0))
            for k in range(max(0, j - pb), i):
                s = ctx.subtract(s, ctx.multiply(R[k][i - k], R[k][j - k]))
            R[i][j - i] = ctx.divide(s, rii)
    return R, ctx


def dec_solve(fac, rhs) -> list[Fraction]:
    """Solve R^T R y = rhs with the decimal factor; returns exact Fractions of the decimal result."""
    R, ctx = fac
    n, pb = len(R), len(R[0]) - 1
    y = [Decimal(0)] * n
    for i in range(n):
        s = ctx.divide(Decimal(rhs[i].numerator), Decimal(rhs[i].denominator)) if isinstance(rhs[i], Fraction) else rhs[i]
        for k in range(max(0, i - pb), i):
            s = ctx.subtract(s, ctx.multiply(R[k][i - k], y[k]))
        y[i] = ctx.divide(s, R[i][0])
    x = [Decimal(0)] * n
    for i in range(n - 1, -1, -1):
        s = y[i]
        for j in range(i + 1, min(n, i + pb + 1)):
            s = ctx.subtract(s, ctx.multiply(R[i][j - i], x[j]))
        x[i] = ctx.divide(s, R[i][0])
    return [Fraction(v) for v in x]


def refine(rows, b, fac):
    """Iterative refinement with the exact residual; returns (x exact rational, r exact, log)."""
    x = dec_solve(fac, b)
    r = residual(rows, b, x)
    log = [float(max(abs(v) for v in r))]
    for _ in range(MAX_REFINE):
        if log[-1] == 0:
            break
        dx = dec_solve(fac, r)
        xn = [a + d for a, d in zip(x, dx, strict=True)]
        rn = residual(rows, b, xn)
        nr = float(max(abs(v) for v in rn))
        if not nr * STAGNATION <= log[-1]:
            if nr < log[-1]:
                x, r = xn, rn
                log.append(nr)
            break
        x, r = xn, rn
        log.append(nr)
    return x, r, log


def tau_trial(p: hs.HalfStep) -> float | None:
    """t = lambda_min(K in double)/2 from the banded double matrix (the Q3 rule; validity only through the certificate)."""
    K, _ = hs.coefficient_system(p)
    N, m = p.N, p.n_s - 1
    perm = np.array([i * m + j for j in range(m) for i in range(N)])
    Kp = K[np.ix_(perm, perm)]
    pb = 3 * N - 1
    ab = np.zeros((pb + 1, Kp.shape[0]))
    for d in range(pb + 1):
        ab[pb - d, d:] = np.diagonal(Kp, d)
    lam = float(scipy.linalg.eig_banded(ab, lower=False, eigvals_only=True, select="i", select_range=(0, 0))[0])
    return lam / 2 if lam > 0 else None


def gershgorin_up(rows) -> Fraction:
    """lambda_max(K) <= max_i sum_j |K_ij| (exact)."""
    return max(sum(abs(v) for _, v in r) for r in rows)


# ---------------------------------------------------------------------------------------------------------------------------------
# one E-4 point
# ---------------------------------------------------------------------------------------------------------------------------------

def budget_ok(ref_bound: Fraction, measured_lower: Fraction) -> bool:
    """D-049.6, item 4: reference error <= 1/100 of the measured error in the same norm, with the GUARANTEED lower bound of the
    measured error (already reduced by the reference error); a measured lower bound 0 is never resolved."""
    return measured_lower > 0 and 100 * ref_bound <= measured_lower


def norm2_sq(v) -> Fraction:
    return sum((x * x for x in v), Fraction(0))


def analyse_point(p: hs.HalfStep, qr: bool, q3row: dict | None, with_rh: bool) -> dict:
    t0 = time.perf_counter()
    rec = bc.run_halfstep(p, qr=qr)
    out = {"N": p.N, "direction": p.direction, "n_f": p.n_f, "n_s": p.n_s, "qr": rec.log["qr"], "status": rec.status,
           "marking": RESEARCH if p.N >= 2 else "N = 1",
           "certificates_refer_to": "solved set (span U~ after QR)" if rec.log["qr"] else "given set",
           "cond_A_before_QR": rec.log.get("cond_A_before_QR"), "cond_A_after_QR": rec.log.get("cond_A_after_QR")}
    if rec.bc is None or rec.bc.s is None:
        out["result"] = f"{UNRESOLVED}: B-C produced no solution ({rec.status})"
        return out
    ps = rec.solved
    N, m = ps.N, ps.n_s - 1
    n, pb = N * m, 3 * N - 1
    h_s = Fraction(ps.l_s) / ps.n_s
    coef = exact_coefficients(ps)
    q3coef = hs.exact_coefficients(ps)
    out["coefficients_equal_q3"] = all(coef[t] == q3coef[t] for t in range(4))
    band = exact_band(ps, coef)
    out["band_equal_q1b"] = band == hs._exact_band_matrix(ps, q3coef)
    rows = band_rows(band, n)
    b = rhs_exact(ps, coef)
    s_bc = to_node_term([Fraction(float(v)) for v in rec.bc.s], N, m)
    r_bc = residual(rows, b, s_bc)
    # provenance: the B-C solution reproduces the Q3 row (exact residual interval and eta_qa)
    if q3row is not None:
        lo, hi = e46._norm2_interval(to_term_first_frac(r_bc, N, m))
        out["q3_reproduced"] = (hs.decimal_down(lo, 17) == q3row["rhp_residual_2_down"]
                                and hs.decimal_up(hi, 17) == q3row["rhp_residual_2_up"]
                                and repr(bc._eta(ps, rec.bc.s)) == q3row["eta_qa"])
    # certificate of lambda_min(K) (Q1b) and conditioning
    tau = tau_trial(ps)
    lam = None if tau is None else hs.certify_lambda_min(ps, tau, q3coef)
    lmax = gershgorin_up(rows)
    out["lambda_low"] = lam
    out["lambda_max_up"] = up(lmax)
    if lam is None:
        out["result"] = f"{UNRESOLVED}: no certificate of lambda_min(K) (Q1b)"
        out["seconds"] = round(time.perf_counter() - t0, 2)
        return out
    L = Fraction(lam)
    out["cond_K_up"] = up(lmax / L)
    # raised-precision reference with the precision ladder
    attempts, used = [], None
    for prec in P_LADDER:
        fac = dec_cholesky(band, n, pb, prec)
        if fac is None:
            attempts.append(f"p={prec}: factorization failed")
            continue
        x, r, log = refine(rows, b, fac)
        r2 = norm2_sq(r)
        b2 = sqrt_up(r2) / L  # >= ||x - x*||_2
        bK = sqrt_up(r2) / sqrt_down(L)  # >= ||x - x*||_K
        dv = [a - c for a, c in zip(s_bc, x, strict=True)]
        d2 = norm2_sq(dv)
        dK2 = sum((a * (rr - rb) for a, rr, rb in zip(dv, r, r_bc, strict=True)), Fraction(0))  # (x^ - x)^T K (x^ - x)
        set_lo = max(Fraction(0), sqrt_down(d2) - b2)
        set_hi = sqrt_up(d2) + b2
        K_lo = max(Fraction(0), sqrt_down(dK2) - bK)
        K_hi = sqrt_up(dK2) + bK
        ok_set, ok_field = budget_ok(b2, set_lo), budget_ok(bK, K_lo)
        used = prec
        attempts.append(f"p={prec}: refinement {len(log) - 1} steps, ||r||_inf {log[0]:.2e} -> {log[-1]:.2e}, budget set "
                        f"{ok_set}, field {ok_field}")
        if ok_set and ok_field or prec == P_LADDER[-1]:
            break
    out["precision_log"] = " | ".join(attempts)
    if used is None:
        out["result"] = f"{UNRESOLVED}: no factorization at any precision of the ladder"
        return out
    out["precision"] = used
    out["refine_steps"] = len(log) - 1
    out["refine_converged"] = len(log) == 1 or log[-1] < log[0]
    sh_up, sh_lo = sqrt_up(h_s), sqrt_down(h_s)
    xs2 = norm2_sq(x)
    xK2 = sum((a * (bb - rr) for a, bb, rr in zip(x, b, r, strict=True)), Fraction(0))  # x^T K x = x^T (b - r)
    xs_lo = max(Fraction(0), sqrt_down(xs2) - b2)  # <= ||x*||_2
    xK_lo = max(Fraction(0), sqrt_down(xK2) - bK)  # <= ||x*||_K
    out["ref_bound_2_up"] = dec_up(b2)
    out["ref_bound_K_up"] = dec_up(bK)
    out["ref_bound_field_a_up"] = dec_up(sh_up * bK)
    out["ref_bound_2_rel_up"] = dec_up(b2 / xs_lo) if xs_lo > 0 else UNRESOLVED
    out["ref_bound_field_rel_up"] = dec_up(bK / xK_lo) if xK_lo > 0 else UNRESOLVED
    out["bc_minus_ref_2_down"], out["bc_minus_ref_2_up"] = dec_down(sqrt_down(d2)), dec_up(sqrt_up(d2))
    out["set_error_2_down"], out["set_error_2_up"] = dec_down(set_lo), dec_up(set_hi)
    out["field_error_a_down"], out["field_error_a_up"] = dec_down(sh_lo * K_lo), dec_up(sh_up * K_hi)
    out["set_error_rel_down"] = dec_down(set_lo / (sqrt_up(xs2) + b2)) if xs2 else UNRESOLVED
    out["set_error_rel_up"] = dec_up(set_hi / xs_lo) if xs_lo > 0 else UNRESOLVED
    out["field_error_rel_down"] = dec_down(K_lo / (sqrt_up(xK2) + bK)) if xK2 else UNRESOLVED
    out["field_error_rel_up"] = dec_up(K_hi / xK_lo) if xK_lo > 0 else UNRESOLVED
    out["budget_set"] = "SATISFIED" if ok_set else f"NOT_MET: comparison {UNRESOLVED}"
    out["budget_field"] = "SATISFIED" if ok_field else f"NOT_MET: comparison {UNRESOLVED}"
    out["result"] = ("RESOLVED (both norms)" if ok_set and ok_field else
                     f"{UNRESOLVED}: budget 1/100 not met (set {ok_set}, field {ok_field})")
    # consistency with Q3 (both intervals contain the true error: they must intersect)
    if q3row is not None and q3row.get("set_error_2_down") not in (None, ""):
        q_lo, q_hi = Fraction(Decimal(q3row["set_error_2_down"])), Fraction(Decimal(q3row["set_error_2_up"]))
        out["q3_set_interval_intersects"] = q_lo <= set_hi and set_lo <= q_hi
        f_lo, f_hi = Fraction(Decimal(q3row["field_error_a_down"])), Fraction(Decimal(q3row["field_error_a_up"]))
        out["q3_field_interval_intersects"] = f_lo <= sh_up * K_hi and sh_lo * K_lo <= f_hi
    # independent paths: exact rational solution (small) and R-H (moderate grids)
    if n <= EXACT_LIMIT:
        Ke, be = e46.exact_system(ps, q3coef)
        x_star_tf = e46._solve_many(Ke, [be])[0]  # term-first
        x_star = to_node_term(x_star_tf, N, m)
        out["exact_ref_within_bound"] = norm2_sq([a - c for a, c in zip(x, x_star, strict=True)]) <= b2 * b2
        e_true = sqrt_down(norm2_sq([a - c for a, c in zip(s_bc, x_star, strict=True)]))
        out["exact_bc_error_in_interval"] = set_lo <= sqrt_up(norm2_sq([a - c for a, c in zip(s_bc, x_star, strict=True)])) \
            and e_true <= set_hi
    if with_rh and (ps.nx - 1) * (ps.ny - 1) <= RH_LIMIT:
        st, xh = hs.solve_hessian(ps)
        if xh is not None:
            xh_nt = to_node_term([Fraction(float(v)) for v in xh], N, m)
            rh = residual(rows, b, xh_nt)
            bh = sqrt_up(norm2_sq(rh)) / L
            dh = norm2_sq([a - c for a, c in zip(xh_nt, x, strict=True)])
            out["rh_consistent"] = dh <= (bh + b2) ** 2  # both enclose x*: must hold
            out["rh_bound_2_up"] = dec_up(bh)
            out["rh_minus_ref_2_up"] = dec_up(sqrt_up(dh))
        else:
            out["rh_consistent"] = f"{UNRESOLVED}: {st}"
    out["seconds"] = round(time.perf_counter() - t0, 2)
    return out


def to_term_first_frac(v_node_term, N: int, m: int) -> list:
    return [v_node_term[j * N + i] for i in range(N) for j in range(m)]


# ---------------------------------------------------------------------------------------------------------------------------------
# the E-4 sample of Q3 (same loops as hs_job and seqstep_job; one-to-one with e4.csv)
# ---------------------------------------------------------------------------------------------------------------------------------

def e4_points(profile: str, arrays: Path) -> list[dict]:
    q3 = _q3()
    pts = []
    for spec in q3.halfstep_specs("full" if profile == "full" else "ci"):
        for direction in hs.DIRECTIONS:
            for N in (1, 2, 3):
                for qr in (spec["qr_modes"] if N >= 2 else (False,)):
                    pts.append({"group": spec["group"], "r": spec.get("r"), "n": spec.get("n"), "lx": spec["lx"], "nx": spec["nx"],
                                "ly": spec["ly"], "ny": spec["ny"], "s": 1, "set": "T_N (sin^2)", "first": "", "N": N,
                                "direction": direction, "qr": qr,
                                "grid": q3.grid_label(spec["lx"], spec["nx"], spec["ly"], spec["ny"])})
    if profile == "full":
        for nx, ny in q3.PAIRS:
            for N in (1, 2, 3):
                for first in ("Y", "X"):
                    for s in range(2, 9):
                        pts.append({"group": "pairs", "r": "1", "n": None, "lx": 1.0, "nx": nx, "ly": 1.0, "ny": ny, "s": s,
                                    "set": "sequence set", "first": first, "N": N, "qr": N >= 2,
                                    "grid": q3.grid_label(1.0, nx, 1.0, ny), "npz": str(arrays / q3.seq_file(
                                        {"nx": nx, "ny": ny, "ly": 1.0, "N": N, "first": first}))})
    return pts


def point_key(d: dict) -> tuple:
    return (d["group"], d["grid"], str(d["s"]), d.get("set", ""), str(d["N"]), d.get("direction", ""), str(d["qr"]),
            d.get("first") or "")


def halfstep_of(pt: dict) -> hs.HalfStep:
    if pt["s"] == 1:
        n_f = pt["nx"] if pt["direction"] == "Y" else pt["ny"]
        return hs.HalfStep(pt["lx"], pt["nx"], pt["ly"], pt["ny"], pt["direction"], hs.start_set(n_f, pt["N"]))
    data = np.load(pt["npz"])
    s, N = pt["s"], pt["N"]
    prev_dir = "Y" if int(data[f"s{s - 1}_dir"][0]) == 0 else "X"
    direction = "X" if prev_dir == "Y" else "Y"
    pt["direction"] = direction
    before = data[f"s{s - 1}_Ys"] if prev_dir == "Y" else data[f"s{s - 1}_Xs"]
    after = data[f"s{s - 1}_Yn"] if prev_dir == "Y" else data[f"s{s - 1}_Xn"]
    return hs.HalfStep(pt["lx"], pt["nx"], pt["ly"], pt["ny"], direction, before if N >= 2 else after)


def point_job(pt: dict, q3row: dict | None, with_rh: bool) -> dict:
    t0 = time.perf_counter()
    p = halfstep_of(pt)
    res = analyse_point(p, pt["qr"], q3row, with_rh)
    tag = {k: pt.get(k) for k in ("group", "r", "n", "grid", "s", "set", "first")}
    return {**tag, **res, "job_seconds": round(time.perf_counter() - t0, 2), "peak_mb": _q3().peak_mb()}


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


def fmt(v):
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
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
    def cnt(pred):
        return sum(1 for r in rows if pred(r))

    out = {"points": len(rows)}
    for k in ("q3_reproduced", "coefficients_equal_q3", "band_equal_q1b", "refine_converged", "q3_set_interval_intersects",
              "q3_field_interval_intersects", "exact_ref_within_bound", "exact_bc_error_in_interval", "rh_consistent"):
        out[k] = f"{cnt(lambda r, k=k: r.get(k) is True)}/{cnt(lambda r, k=k: k in r)}"
    out["certificate_missing"] = cnt(lambda r: str(r.get("result", "")).startswith(f"{UNRESOLVED}: no certificate"))
    out["budget_set_satisfied"] = cnt(lambda r: r.get("budget_set") == "SATISFIED")
    out["budget_field_satisfied"] = cnt(lambda r: r.get("budget_field") == "SATISFIED")
    out["resolved_both"] = cnt(lambda r: r.get("result") == "RESOLVED (both norms)")
    out["precision_used"] = {str(p): cnt(lambda r, p=p: r.get("precision") == p) for p in P_LADDER}
    out["by_group"] = {}
    for g in sorted({r["group"] for r in rows}):
        sub = [r for r in rows if r["group"] == g]
        out["by_group"][g] = {"points": len(sub), "resolved_both": sum(1 for r in sub if r.get("result") == "RESOLVED (both norms)")}
    return out


def run_campaign(profile: str, out: Path, workers: int, arrays: Path) -> dict:
    t0 = time.time()
    out.mkdir(parents=True, exist_ok=True)
    pts = e4_points(profile, arrays)
    q3rows = {}
    if profile == "full":
        with open(Q3_E4, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                key = point_key({**r, "qr": r["qr"]})
                if r["s"] != "1":
                    key = point_key({**r, "direction": "", "qr": r["qr"]})
                assert key not in q3rows, key
                q3rows[key] = r
    jobs = []
    for pt in pts:
        key = point_key({**pt, "direction": pt["direction"] if pt["s"] == 1 else "", "qr": pt["qr"] if pt["N"] >= 2 else False})
        jobs.append((pt, q3rows.get(key) if profile == "full" else None, True))
    if profile == "full":
        assert len(jobs) == len(q3rows) == 476 and all(j[1] is not None for j in jobs), "E-4 sample is not one-to-one with Q3"
    jobs.sort(key=lambda j: -(max(j[0]["nx"], j[0]["ny"]) * j[0]["N"]))
    if workers > 1:
        with multiprocessing.get_context("spawn").Pool(workers) as pool:
            rows = pool.starmap(point_job, jobs, chunksize=1)
    else:
        rows = [point_job(*j) for j in jobs]
    rows.sort(key=lambda r: (r["group"], r["grid"], int(r["s"]), r.get("first") or "", r["N"], r["direction"], str(r["qr"])))
    timing = [{k: r.get(k) for k in ("group", "grid", "s", "N", "direction", "qr", "first", "job_seconds", "peak_mb")} for r in rows]
    data_rows = [{k: v for k, v in r.items() if k not in ("job_seconds", "peak_mb", "seconds")} for r in rows]
    write_csv(out / "e4_reference.csv", data_rows)
    summ = summarize(rows)
    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summ, f, indent=1, ensure_ascii=False)
        f.write("\n")
    q3 = _q3()
    meta = {"study": "R5-V Q4 priority 2 (D-053.4, item 2): certified raised-precision half-step reference and the B-C error on the "
                     "E-4 sample; QA research tool, no qualification; tol_solve not assigned",
            "profile": profile, "git_commit": git("rev-parse", "HEAD"), "git_status_porcelain": git("status", "--porcelain").splitlines(),
            "python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__, "platform": platform.platform(),
            "machine": platform.machine(), "processor": platform.processor(), "threads": {k: os.environ.get(k) for k in THREAD_VARS},
            "workers": workers, "precision_ladder": list(P_LADDER), "max_refine": MAX_REFINE, "stagnation_factor": STAGNATION,
            "rh_limit": RH_LIMIT, "exact_limit": EXACT_LIMIT,
            "q3_e4": {"path": Q3_E4.relative_to(ROOT).as_posix(), "sha256": sha256(Q3_E4),
                      "unchanged_since_q3_campaign": git("hash-object", str(Q3_E4)) == git("rev-parse",
                                                                                          f"{Q3_DATA_COMMIT}:docs/results/r5_cccc_q3_prequalification/e4.csv")}
            if profile == "full" else "CI profile: no Q3 comparison",
            "q3_arrays_dir": str(arrays) if profile == "full" else None, "q3_arrays_used_read_only": True,
            "npz_sha256": {Path(j[0]["npz"]).name: sha256(Path(j[0]["npz"])) for j in jobs if "npz" in j[0]},
            "timing": timing,
            "resource_rule_PL_6_3": {"peak_job_mb": max((r.get("peak_mb") or 0) for r in rows),
                                     "limit_mb_50pct_ram": (q3.total_ram_mb() or 0) / 2,
                                     "longest_job_s": max(r["job_seconds"] for r in rows), "limit_factorization_s": 1800},
            "seconds_total": round(time.time() - t0, 1)}
    with open(out / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1, ensure_ascii=False)
        f.write("\n")
    return {"summary": summ, "meta": meta}


def probe(arrays: Path) -> dict:
    """Resource probe (PL 6.3) on the largest E-4 points before the campaign."""
    pts = [p for p in e4_points("full", arrays) if p["s"] == 1 and max(p["nx"], p["ny"]) == 1280 and p["N"] == 3]
    out = []
    for pt in pts[:2]:
        t0 = time.perf_counter()
        r = point_job(pt, None, False)
        out.append({"grid": pt["grid"], "direction": pt["direction"], "qr": pt["qr"], "seconds": round(time.perf_counter() - t0, 1),
                    "peak_mb": r["peak_mb"], "result": r.get("result"), "precision_log": r.get("precision_log")})
    return {"probe": out, "ram_mb": _q3().total_ram_mb()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--profile", choices=("full", "ci"), default="full")
    r.add_argument("--out", type=Path, default=None)
    r.add_argument("--workers", type=int, default=8)
    r.add_argument("--arrays", type=Path, default=Q3_ARRAYS_DEFAULT)
    pr = sub.add_parser("probe")
    pr.add_argument("--arrays", type=Path, default=Q3_ARRAYS_DEFAULT)
    a = ap.parse_args(argv)
    if a.cmd == "probe":
        print(json.dumps(probe(a.arrays), indent=1))
        return 0
    out = a.out or (OUT if a.profile == "full" else ROOT / "scratch" / "r5_cccc_q4_bc_ci")
    res = run_campaign(a.profile, out, a.workers if a.profile == "full" else 1, a.arrays)
    print(json.dumps(res["summary"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
