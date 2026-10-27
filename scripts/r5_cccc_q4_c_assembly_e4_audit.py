"""R5-V Q4, priority 2 (D-053.4): the assembly C-3+ of the coefficient C (Literature c66f4d2, docs/R5_CCCC_Q4_C_ASSEMBLY.study.md 3.3;
Reviewer 31f921a: PASS_WITH_LIMITATIONS) checked on the 476 half-steps of E-4 against the certified reference of the exact system.
QA research function; the prototype B-C (src/kkvim_reference/r5_cccc_bc.py), the working kkvim and all existing data are NOT changed;
no import of the working ``kkvim``.

Fixed BEFORE the campaign (this docstring and the constants are the protocol):

Object. For each E-4 point the solved half-step system of r5_cccc_bc.run_halfstep (after QR for N >= 2: the stored set U~ of that run;
the same sets as the certified reference audit) - the finite discrete problem with binary64 inputs, h = l/n exact.

Solutions (the same solve_bc, the same A, B, F computed by the prototype's own bc_coefficients; ONLY C differs):
- OLD: C of the prototype, h * (G^T @ _apply_D4(G, h)) (current path);
- NEW: C-3+ (``c3plus``): Y3 = rd(Z3 G) (exact sum of the three exactly representable terms g_{i-1} - 2 g_i + g_{i+1}, ONE correct
  rounding), h = fl(l_f/n_f), h2 = fl(h*h), X = fl(Y3/h2), C_L = fl(h * fl(X^T X)), h3 = fl(h2*h), s = fl(2/h3),
  E = fl(s * fl(fl(g1 g1^T) + fl(gm gm^T))), C = fl(C_L + E). Powers by explicit multiplication: no premise on ``pow``;
- CONTROL: variant 1 of the study (second algorithmic path): Y5 = rd(Z5 G) (exact five-point D4 stencil incl. the (7, -4, 1) rows, one
  rounding), h4 = fl(h2*h2), C = fl(h * fl(G^T fl(Y5/h4)));
- REFERENCE: the certified raised-precision solution x of the exact system (scripts/r5_cccc_q4_bc_reference_audit.py functions:
  exact assembly, decimal Cholesky, exact-residual refinement, Q1b lambda_low), bounds ||x - x*||_2 <= ||r||_2/lambda_low,
  ||x - x*||_K <= ||r||_2/lambda_low^(1/2).

Count k of the C-3+ code (QA derivation, report 3; u = 2^-53, gamma_j = j u/(1 - j u)): X^ = X (1 + theta), |theta| <= gamma_5 (h: 1,
squared: 2, h*h: 1, rounding of Y3: 1, division: 1 -> c_x = 5); C_L: Gram gamma_m (any order) and (1 + theta)^2 -> gamma_{m + 10},
times h (2) -> gamma_{m + 12}; E: 2/h^3 with h (3), two products (2), division (1), outer products (1), their sum (1), the scaling (1)
-> c_E = 9; the final sum (1) -> k = m + 13 (m = n_f - 1). Theorem C-3+: |C^_ij - C_ij| <= gamma_k sqrt(C_ii C_jj),
||C^ - C||_F <= gamma_k tr C, checked EXACTLY (squared, rational).

Premises (P-1...P-5 of RO 0.4) checked per point; if one fails the inequality is NOT_APPLICABLE (never PROVEN for that point):
finite values; no nonzero subnormal Y3, X^, g_1, g_m; no possible underflow of the Gram products X^_li X^_lj and g g (min nonzero
|X^|^2, |g|^2 >= 2^-1022) and no overflow (m max|X^|^2, 2 max|g|^2 s < DBL_MAX) (review N-A2).

Positive definiteness of the computed C (review N-A1) is checked separately and exactly (symmetric part, exact LDL^T); no
regularization. Modal factorization success and the prototype's diagnostics (cond_S, max_cond_Mk) are recorded, no threshold.

Errors (each in the set 2-norm and in the K-norm, the field a-norm being h_s^(1/2) ||.||_K; P-S1):
- total: s - x* (s = OLD, NEW or CONTROL); interval [d - b, d + b], d = ||s - x||, b the reference bound;
- assembly: x*_c - x*, x*_c the exact solution of the system with the coefficients (A^, B^, C^, F^) of that solution taken exactly,
  matrices by their exact symmetric parts (``frac_coef``; correction after the first run, report 4) (the same certified solver);
  modal: s - x*_c (B-C error relative to its own rounded coefficients, incl. the antisymmetric part of B^, C^);
- relative values use the certified bounds of ||x*||.
Budget (D-049.6, item 4) for EVERY quantitative comparison separately, in its norm: 100 * (reference bound) <= guaranteed lower bound
of the measured difference (already reduced by the reference bounds); otherwise that comparison is UNRESOLVED.
NEW vs OLD: BETTER if the NEW interval lies below the OLD interval, WORSE if above, EQUAL if the solutions are bitwise equal,
OVERLAP otherwise (not decided).

Precision rule of the references: P_LADDER of the reference audit (40, 80, 160 digits): the next precision when a factorization fails
or a budget of the point is not met; after 160 the comparison stays UNRESOLVED.

Usage (repository root):
    uv run python scripts/r5_cccc_q4_c_assembly_e4_audit.py run --profile full [--workers 8]
    uv run python scripts/r5_cccc_q4_c_assembly_e4_audit.py run --profile ci --out scratch/r5_cccc_q4_c_e4_ci
"""

import argparse
import contextlib
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
from fractions import Fraction
from pathlib import Path

THREAD_VARS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
for _k in THREAD_VARS:
    os.environ[_k] = "1"

import numpy as np  # noqa: E402
import scipy  # noqa: E402

from kkvim_reference import r5_cccc_bc as bc  # noqa: E402
from kkvim_reference import r5_cccc_halfstep as hs  # noqa: E402
from kkvim_reference import r5_cccc_q3_e4e6 as e46  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "results" / "r5_cccc_q4_c_assembly_e4"
REF_SCRIPT = ROOT / "scripts" / "r5_cccc_q4_bc_reference_audit.py"
REF_ROWS = ROOT / "docs" / "results" / "r5_cccc_q4_bc_reference_audit" / "e4_reference.csv"
LITERATURE_SHA = "c66f4d2e9f06e7303be0aaadb1d2162b7ae2f8bb"
REVIEWER_SHA = "31f921ad6b9ca1b46dcfc3295a51cf5a9dd50431"
U = Fraction(1, 2**53)
TINY = 2.0**-1022
DBL_MAX = 1.7976931348623157e308
K_EXTRA = 13  # k = m + 13 for the C-3+ code of this tool (docstring)
VARIANTS = ("OLD", "NEW", "CONTROL")
UNRESOLVED = "UNRESOLVED"


def _ra():
    if not hasattr(_ra, "mod"):
        spec = importlib.util.spec_from_file_location("r5_cccc_q4_bc_reference_audit", REF_SCRIPT)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _ra.mod = mod
    return _ra.mod


def gam(k: int) -> Fraction:
    return k * U / (1 - k * U)


# ---------------------------------------------------------------------------------------------------------------------------------
# assemblies of C
# ---------------------------------------------------------------------------------------------------------------------------------

def _exact_stencil_rounded(G: np.ndarray, coeffs: dict) -> np.ndarray:
    """rd(sum_k c_k g_{i+k}) column by column: exact rational sum of the stored doubles (zeros outside the grid), ONE correct rounding
    (float(Fraction) is correctly rounded); ``coeffs`` maps an offset to an integer, the diagonal may differ at the two end rows."""
    m, N = G.shape
    out = np.empty_like(G, dtype=float)
    F = [[Fraction(float(G[i, j])) for j in range(N)] for i in range(m)]
    for j in range(N):
        for i in range(m):
            s = Fraction(0)
            for off, c in coeffs["off"].items():
                k = i + off
                if 0 <= k < m:
                    s += c * F[k][j]
            if i in (0, m - 1) and "end_diag" in coeffs:
                s += (coeffs["end_diag"] - coeffs["off"][0]) * F[i][j]
            out[i, j] = float(s)
    return out


Z3 = {"off": {-1: 1, 0: -2, 1: 1}}
Z5 = {"off": {-2: 1, -1: -4, 0: 6, 1: -4, 2: 1}, "end_diag": 7}


def c3plus(p: hs.HalfStep) -> dict:
    """C-3+ (docstring) with every intermediate kept for the premise checks."""
    G = np.asarray(p.given, dtype=float)
    h = p.l_f / p.n_f
    Y3 = _exact_stencil_rounded(G, Z3)
    h2 = h * h
    h3 = h2 * h
    s = 2.0 / h3
    g1, gm = G[0], G[-1]
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):  # inf/NaN are detected by premises_c3plus (P-2)
        X = Y3 / h2
        CL = h * (X.T @ X)
        E = s * (np.outer(g1, g1) + np.outer(gm, gm))
        C = CL + E
    return {"C": C, "Y3": Y3, "X": X, "g1": g1, "gm": gm, "s": s}


def c_control(p: hs.HalfStep) -> np.ndarray:
    """Variant 1: exact five-point D4 stencil rounded once, Gram in double."""
    G = np.asarray(p.given, dtype=float)
    h = p.l_f / p.n_f
    Y5 = _exact_stencil_rounded(G, Z5)
    h2 = h * h
    h4 = h2 * h2
    return h * (G.T @ (Y5 / h4))


def premises_c3plus(parts: dict) -> str | None:
    """P-2/P-3 for C-3+ (docstring); None if they hold."""
    for name in ("Y3", "X", "g1", "gm", "C"):
        a = np.asarray(parts[name])
        if not np.all(np.isfinite(a)):
            return f"P-2: non-finite {name}"
        nz = np.abs(a[a != 0])
        if nz.size and float(nz.min()) < TINY:
            return f"P-3: subnormal {name}"

    def mn(a):
        nz = np.abs(np.asarray(a)[np.asarray(a) != 0])
        return float(nz.min()) if nz.size else math.inf

    X, g = parts["X"], np.concatenate([parts["g1"], parts["gm"]])
    if mn(X) ** 2 < TINY or mn(g) ** 2 < TINY or mn(g) ** 2 * parts["s"] < TINY:
        return "P-3: a Gram or corner product may underflow"
    if X.shape[0] * float(np.max(np.abs(X))) ** 2 >= DBL_MAX or 2 * float(np.max(np.abs(g))) ** 2 * parts["s"] >= DBL_MAX:
        return "P-2: a Gram or corner sum may overflow"
    return None


def check_c3plus(C_hat: np.ndarray, C_exact, k: int) -> dict:
    """Exact check of |C^_ij - C_ij| <= gamma_k sqrt(C_ii C_jj) and ||C^ - C||_F <= gamma_k tr C (squared, rational)."""
    N = C_hat.shape[0]
    g = gam(k)
    worst = Fraction(0)
    ok = True
    fro = Fraction(0)
    for i in range(N):
        for j in range(N):
            d = Fraction(float(C_hat[i, j])) - C_exact[i][j]
            fro += d * d
            rhs = g * g * C_exact[i][i] * C_exact[j][j]
            ok &= d * d <= rhs
            if rhs > 0:
                worst = max(worst, d * d / rhs)
    tr = sum(C_exact[i][i] for i in range(N))
    ok_f = fro <= (g * tr) ** 2
    return {"elementwise": ok, "frobenius": ok_f, "max_ratio_sq": worst, "fro_ratio_sq": fro / (g * tr) ** 2 if tr else None}


def c_error(C_hat: np.ndarray, C_exact) -> float:
    """max_ij |C^_ij - C_ij| / sqrt(C_ii C_jj) (OBSERVATION; the scale of the theorem), rounded upwards."""
    N = C_hat.shape[0]
    worst = Fraction(0)
    for i in range(N):
        for j in range(N):
            d = Fraction(float(C_hat[i, j])) - C_exact[i][j]
            den = C_exact[i][i] * C_exact[j][j]
            if den > 0:
                worst = max(worst, d * d / den)
    return hs._float_up(_ra().sqrt_up(worst))


def spd_exact(C_hat: np.ndarray) -> bool:
    """Positive definiteness of the symmetric part of the stored double matrix (exact LDL^T; review N-A1)."""
    N = C_hat.shape[0]
    M = [[(Fraction(float(C_hat[i, j])) + Fraction(float(C_hat[j, i]))) / 2 for j in range(N)] for i in range(N)]
    return e46._spd_exact(M)


@contextlib.contextmanager
def replaced_c(C_new: np.ndarray):
    """solve_bc with ONLY C replaced: A, B, F from the prototype's own bc_coefficients (runtime wrapper; the file is not changed)."""
    orig = bc.bc_coefficients

    def wrapped(p):
        A, B, C, F = orig(p)
        return A, B, C_new, F

    bc.bc_coefficients = wrapped
    try:
        yield
    finally:
        bc.bc_coefficients = orig


# ---------------------------------------------------------------------------------------------------------------------------------
# certified solutions of exact systems (functions of the reference audit)
# ---------------------------------------------------------------------------------------------------------------------------------

def certified(ps: hs.HalfStep, coef, prec: int, tau):
    """(x, r, b2, bK, lambda_low) of the exact system with the coefficients ``coef`` (Fractions), or a reason string."""
    ra = _ra()
    N, m = ps.N, ps.n_s - 1
    n = N * m
    band = ra.exact_band(ps, coef)
    lam = None if tau is None else hs.certify_lambda_min(ps, tau, coef)
    if lam is None:
        return f"{UNRESOLVED}: no lambda_min certificate (Q1b)"
    fac = ra.dec_cholesky(band, n, 3 * N - 1, prec)
    if fac is None:
        return f"{UNRESOLVED}: decimal factorization failed at p = {prec}"
    rows = ra.band_rows(band, n)
    b = ra.rhs_exact(ps, coef)
    x, r, log = ra.refine(rows, b, fac)
    r2 = ra.norm2_sq(r)
    L = Fraction(lam)
    return {"x": x, "r": r, "rows": rows, "b": b, "b2": ra.sqrt_up(r2) / L, "bK": ra.sqrt_up(r2) / ra.sqrt_down(L), "lam": lam,
            "refine": log}


def diff_interval(ra, u_vec, v_vec, ru, rv, bu2, bv2, buK, bvK):
    """Intervals of ||u* - v*|| (2- and K-norm) from approximations u, v of exact solutions with bounds; residuals of the SAME K."""
    dv = [a - c for a, c in zip(u_vec, v_vec, strict=True)]
    d2 = ra.norm2_sq(dv)
    dK2 = sum((a * (rv_ - ru_) for a, ru_, rv_ in zip(dv, ru, rv, strict=True)), Fraction(0))  # (u - v)^T K (u - v)
    b2, bK = bu2 + bv2, buK + bvK
    return {"lo2": max(Fraction(0), ra.sqrt_down(d2) - b2), "hi2": ra.sqrt_up(d2) + b2, "b2": b2,
            "loK": max(Fraction(0), ra.sqrt_down(max(dK2, Fraction(0))) - bK), "hiK": ra.sqrt_up(max(dK2, Fraction(0))) + bK, "bK": bK}


def frac_coef(A, B, C, F):
    """Exact SYMMETRIC parts (M + M^T)/2 of the binary64 coefficient matrices and F exactly. The band assembly of an SPD system
    (and the Q1b certificate) uses one triangle per node block and both at the neighbouring blocks, so a non-symmetric M^ (B^ of the
    prototype is not exactly symmetric) must be symmetrized explicitly; the antisymmetric part, which B-C sees through the LU solve
    of the full modal blocks, is thereby attributed to the modal part (correction of this tool, report 4)."""
    N = len(F)
    sym = lambda M: [[(Fraction(float(M[i, k])) + Fraction(float(M[k, i]))) / 2 for k in range(N)] for i in range(N)]  # noqa: E731
    return sym(A), sym(B), sym(C), [Fraction(float(x)) for x in F]


# ---------------------------------------------------------------------------------------------------------------------------------
# one E-4 point
# ---------------------------------------------------------------------------------------------------------------------------------

def analyse_point(p: hs.HalfStep, qr: bool, stored: dict | None) -> dict:
    ra = _ra()
    rec = bc.run_halfstep(p, qr=qr)  # OLD, and the fixed solved set
    out = {"N": p.N, "direction": p.direction, "n_f": p.n_f, "n_s": p.n_s, "qr": rec.log["qr"], "status_OLD": rec.status,
           "marking": "RESEARCH_ONLY / PROVISIONAL / NOT_QUALIFIED (D-049.5)" if p.N >= 2 else "N = 1",
           "certificates_refer_to": "solved set (span U~ after QR)" if rec.log["qr"] else "given set"}
    if rec.bc is None or rec.bc.s is None:
        out["result"] = f"{UNRESOLVED}: OLD B-C produced no solution"
        return out
    ps = rec.solved
    m_f = ps.n_f - 1
    k = m_f + K_EXTRA
    out["k"] = k
    A, B, C_old, F = bc.bc_coefficients(ps)
    exact = hs.exact_coefficients(ps)
    Cx = exact[2]
    parts = c3plus(ps)
    C_new, C_ctl = parts["C"], c_control(ps)
    why = premises_c3plus(parts)
    out["c3plus_premises"] = "HOLD" if why is None else why
    chk = check_c3plus(C_new, Cx, k)
    out["c3plus_ratio_up"] = hs._float_up(ra.sqrt_up(chk["max_ratio_sq"]))  # max |dC_ij| / (gamma_k sqrt(C_ii C_jj))
    out["c3plus_fro_ratio_up"] = hs._float_up(ra.sqrt_up(chk["fro_ratio_sq"])) if chk["fro_ratio_sq"] is not None else None
    if why is None:
        out["c3plus_inequality"] = "PASS" if chk["elementwise"] and chk["frobenius"] else "FAIL (violation with premises)"
    else:
        out["c3plus_inequality"] = "NOT_APPLICABLE (premise not verified)"
    for name, Ch in (("OLD", C_old), ("NEW", C_new), ("CONTROL", C_ctl)):
        out[f"C_err_{name}_up"] = c_error(Ch, Cx)  # max |dC_ij|/sqrt(C_ii C_jj)
        out[f"C_spd_{name}"] = spd_exact(Ch)
        out[f"C_asym_{name}"] = float(np.max(np.abs(Ch - Ch.T)))
    out["C_NEW_equals_OLD_bitwise"] = bool(np.array_equal(C_new, C_old))
    # solutions on the same fixed set: only C differs
    sols = {"OLD": rec.bc}
    for name, Ch in (("NEW", C_new), ("CONTROL", C_ctl)):
        with replaced_c(Ch):
            sols[name] = bc.solve_bc(ps)
        A2, B2, C2, F2 = (bc.bc_coefficients(ps))  # unwrapped again: the prototype's own A, B, F
        assert np.array_equal(A2, A) and np.array_equal(B2, B) and np.array_equal(F2, F)
    for name in VARIANTS:
        out[f"bc_status_{name}"] = sols[name].status
        for dk in ("cond_S", "max_cond_Mk", "min_eig_Mk"):
            out[f"{dk}_{name}"] = sols[name].diagnostics.get(dk) if sols[name].s is not None else None
    # certified references: exact system and the three systems with the binary64 coefficients of each variant
    tau = ra.tau_trial(ps)
    coefs = {"OLD": frac_coef(A, B, C_old, F), "NEW": frac_coef(A, B, C_new, F), "CONTROL": frac_coef(A, B, C_ctl, F)}
    used = None
    for prec in ra.P_LADDER:
        ref = certified(ps, exact, prec, tau)
        if isinstance(ref, str):
            out["result"] = ref
            continue
        refc = {v: certified(ps, coefs[v], prec, tau) for v in VARIANTS}
        res = _errors(out, ps, ref, refc, sols)
        used = prec
        if res["all_budgets"] or prec == ra.P_LADDER[-1]:
            break
    out["precision"] = used
    if used is None:
        return out
    out["lambda_low"] = ref["lam"]
    out["ref_bound_2_rel_up"] = res["ref_rel2"]
    out["ref_bound_K_rel_up"] = res["ref_relK"]
    # provenance: OLD reproduces the stored interval of the reference audit
    if stored is not None:
        out["old_matches_reference_audit"] = (out["OLD_set_down"] == stored["set_error_2_down"]
                                              and out["OLD_set_up"] == stored["set_error_2_up"])
    out["result"] = "COMPLETE" if res["all_budgets"] else f"{UNRESOLVED}: a budget 1/100 not met (see budget_* columns)"
    return out


def _errors(out, ps, ref, refc, sols) -> dict:
    ra = _ra()
    N, m = ps.N, ps.n_s - 1
    x, r = ref["x"], ref["r"]
    xs2 = ra.norm2_sq(x)
    xK2 = sum((a * (bb - rr) for a, bb, rr in zip(x, ref["b"], r, strict=True)), Fraction(0))
    xs_lo, xs_hi = max(Fraction(0), ra.sqrt_down(xs2) - ref["b2"]), ra.sqrt_up(xs2) + ref["b2"]
    xK_lo, xK_hi = max(Fraction(0), ra.sqrt_down(xK2) - ref["bK"]), ra.sqrt_up(xK2) + ref["bK"]
    all_ok = True
    bounds = {}

    def put(prefix, iv):
        nonlocal all_ok
        out[f"{prefix}_set_down"], out[f"{prefix}_set_up"] = ra.dec_down(iv["lo2"]), ra.dec_up(iv["hi2"])
        out[f"{prefix}_set_rel_down"] = ra.dec_down(iv["lo2"] / xs_hi) if xs_hi else UNRESOLVED
        out[f"{prefix}_set_rel_up"] = ra.dec_up(iv["hi2"] / xs_lo) if xs_lo else UNRESOLVED
        out[f"{prefix}_field_rel_down"] = ra.dec_down(iv["loK"] / xK_hi) if xK_hi else UNRESOLVED
        out[f"{prefix}_field_rel_up"] = ra.dec_up(iv["hiK"] / xK_lo) if xK_lo else UNRESOLVED
        ok2, okK = ra.budget_ok(iv["b2"], iv["lo2"]), ra.budget_ok(iv["bK"], iv["loK"])
        out[f"{prefix}_budget_set"] = "SATISFIED" if ok2 else UNRESOLVED
        out[f"{prefix}_budget_field"] = "SATISFIED" if okK else UNRESOLVED
        all_ok = all_ok and ok2 and okK
        bounds[prefix] = iv

    zero = Fraction(0)
    for v in VARIANTS:
        s = sols[v].s
        if s is None:
            out[f"{v}_set_down"] = f"{UNRESOLVED}: no B-C solution ({sols[v].status})"
            all_ok = False
            continue
        sv = ra.to_node_term([Fraction(float(t)) for t in s], N, m)
        rs = ra.residual(ref["rows"], ref["b"], sv)
        put(v, diff_interval(ra, sv, x, rs, r, zero, ref["b2"], zero, ref["bK"]))  # total error s - x*
        rc = refc[v]
        if isinstance(rc, str):
            out[f"{v}_assembly_set_down"] = rc
            all_ok = False
            continue
        rcx = ra.residual(ref["rows"], ref["b"], rc["x"])  # residual of x_c in the EXACT system (for the K-norm of x*_c - x*)
        # assembly x*_c - x*: x_c approximates x*_c with bound rc; the K-norm is that of the exact K
        put(f"{v}_assembly", diff_interval(ra, rc["x"], x, rcx, r, rc["b2"], ref["b2"],
                                           rc["b2"] * ra.sqrt_up(ra.gershgorin_up(ref["rows"])), ref["bK"]))
        put(f"{v}_modal", diff_interval(ra, sv, rc["x"], rs, rcx, zero, rc["b2"], zero,
                                        rc["b2"] * ra.sqrt_up(ra.gershgorin_up(ref["rows"]))))
    for nrm, lo, hi in (("set", "lo2", "hi2"), ("field", "loK", "hiK")):
        if "OLD" in bounds and "NEW" in bounds:
            o, nw = bounds["OLD"], bounds["NEW"]
            if np.array_equal(sols["OLD"].s, sols["NEW"].s):
                cmp_ = "EQUAL (bitwise)"
            elif nw[hi] < o[lo]:
                cmp_ = "BETTER"
            elif nw[lo] > o[hi]:
                cmp_ = "WORSE"
            else:
                cmp_ = "OVERLAP (not decided)"
            out[f"NEW_vs_OLD_{nrm}"] = cmp_
    return {"all_budgets": all_ok, "ref_rel2": ra.dec_up(ref["b2"] / xs_lo) if xs_lo else UNRESOLVED,
            "ref_relK": ra.dec_up(ref["bK"] / xK_lo) if xK_lo else UNRESOLVED}


# ---------------------------------------------------------------------------------------------------------------------------------
# campaign
# ---------------------------------------------------------------------------------------------------------------------------------

def point_job(pt: dict, stored: dict | None) -> dict:
    ra = _ra()
    t0 = time.perf_counter()
    p = ra.halfstep_of(pt)
    res = analyse_point(p, pt["qr"], stored)
    tag = {k: pt.get(k) for k in ("group", "r", "n", "grid", "s", "set", "first")}
    return {**tag, **res, "job_seconds": round(time.perf_counter() - t0, 2), "peak_mb": ra._q3().peak_mb()}


def _sha(path: Path) -> str:
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


def _ref_key(r: dict) -> tuple:
    return (r["group"], r["grid"], str(r["s"]), r.get("first") or "", str(r["N"]), r["direction"], str(r["qr"]))


def summarize(rows: list[dict]) -> dict:
    def cnt(k, v):
        return sum(1 for r in rows if r.get(k) == v)

    out = {"points": len(rows), "complete": cnt("result", "COMPLETE"),
           "c3plus_inequality": {s: sum(1 for r in rows if str(r.get("c3plus_inequality", "")).startswith(s))
                                 for s in ("PASS", "FAIL", "NOT_APPLICABLE")},
           "old_matches_reference_audit": f"{cnt('old_matches_reference_audit', True)}/"
                                          f"{sum(1 for r in rows if 'old_matches_reference_audit' in r)}",
           "precision_used": {str(p): cnt("precision", p) for p in (40, 80, 160)}}
    for v in VARIANTS:
        out[f"bc_status_{v}"] = {s: cnt(f"bc_status_{v}", s) for s in sorted({str(r.get(f'bc_status_{v}')) for r in rows})}
        out[f"C_spd_{v}"] = f"{cnt(f'C_spd_{v}', True)}/{len(rows)}"
        for pre in (v, f"{v}_assembly", f"{v}_modal"):
            for nrm in ("set", "field"):
                out[f"budget_{pre}_{nrm}"] = f"{cnt(f'{pre}_budget_{nrm}', 'SATISFIED')}/{len(rows)}"
    for nrm in ("set", "field"):
        out[f"NEW_vs_OLD_{nrm}"] = {s: sum(1 for r in rows if str(r.get(f"NEW_vs_OLD_{nrm}", "")).startswith(s))
                                   for s in ("BETTER", "EQUAL", "WORSE", "OVERLAP")}
    ratios = [r["c3plus_ratio_up"] for r in rows if isinstance(r.get("c3plus_ratio_up"), float)]
    out["c3plus_max_ratio"] = max(ratios) if ratios else None
    out["c3plus_max_fro_ratio"] = max((r["c3plus_fro_ratio_up"] for r in rows if isinstance(r.get("c3plus_fro_ratio_up"), float)),
                                      default=None)
    return out


def run_campaign(profile: str, out: Path, workers: int, arrays: Path) -> dict:
    ra = _ra()
    t0 = time.time()
    out.mkdir(parents=True, exist_ok=True)
    pts = ra.e4_points(profile, arrays)
    stored = {}
    if profile == "full":
        with open(REF_ROWS, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                stored[_ref_key(r)] = r
    jobs = []
    for pt in pts:
        jobs.append((pt, None))
    if workers > 1:
        with multiprocessing.get_context("spawn").Pool(workers) as pool:
            rows = pool.starmap(point_job, jobs, chunksize=1)
    else:
        rows = [point_job(*j) for j in jobs]
    if profile == "full":  # provenance against the stored reference audit (OLD intervals, string equality)
        for r in rows:
            st = stored.get(_ref_key(r))
            r["old_matches_reference_audit"] = st is not None and r.get("OLD_set_down") == st["set_error_2_down"] \
                and r.get("OLD_set_up") == st["set_error_2_up"]
        assert len(rows) == len(stored) == 476, "E-4 sample is not one-to-one with the reference audit"
    rows.sort(key=lambda r: (r["group"], r["grid"], int(r["s"]), r.get("first") or "", r["N"], r["direction"], str(r["qr"])))
    timing = [{k: r.get(k) for k in ("group", "grid", "s", "N", "direction", "qr", "first", "job_seconds", "peak_mb")} for r in rows]
    write_csv(out / "rows.csv", [{k: v for k, v in r.items() if k not in ("job_seconds", "peak_mb")} for r in rows])
    summ = summarize(rows)
    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summ, f, indent=1, ensure_ascii=False)
        f.write("\n")
    inputs = {p.relative_to(ROOT).as_posix(): _sha(p) for p in (REF_SCRIPT, REF_ROWS, ROOT / "src" / "kkvim_reference" / "r5_cccc_bc.py",
                                                                  ROOT / "src" / "kkvim_reference" / "r5_cccc_halfstep.py",
                                                                  ra.Q3_E4)}
    meta = {"study": "C-3+ assembly of C on the 476 E-4 half-steps against the certified reference (QA research; no Builder choice, "
                     "no thresholds)",
            "profile": profile, "git_commit": git("rev-parse", "HEAD"), "git_status_porcelain": git("status", "--porcelain").splitlines(),
            "literature_sha": LITERATURE_SHA, "reviewer_sha": REVIEWER_SHA,
            "literature_sha_present": git("cat-file", "-t", LITERATURE_SHA) == "commit",
            "reviewer_sha_present": git("cat-file", "-t", REVIEWER_SHA) == "commit",
            "python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__, "platform": platform.platform(),
            "machine": platform.machine(), "processor": platform.processor(), "threads": {k: os.environ.get(k) for k in THREAD_VARS},
            "workers": workers, "k_rule": "k = m_f + 13 (docstring)", "precision_ladder": list(ra.P_LADDER),
            "input_sha256": inputs, "q3_arrays_dir": str(arrays) if profile == "full" else None,
            "npz_sha256": {Path(p["npz"]).name: _sha(Path(p["npz"])) for p in pts if "npz" in p},
            "timing": timing,
            "resource_rule_PL_6_3": {"peak_job_mb": max((r.get("peak_mb") or 0) for r in rows),
                                     "limit_mb_50pct_ram": (ra._q3().total_ram_mb() or 0) / 2,
                                     "longest_job_s": max(r["job_seconds"] for r in rows), "limit_factorization_s": 1800},
            "seconds_total": round(time.time() - t0, 1)}
    with open(out / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1, ensure_ascii=False)
        f.write("\n")
    return {"summary": summ, "meta": meta}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--profile", choices=("full", "ci"), default="full")
    r.add_argument("--out", type=Path, default=None)
    r.add_argument("--workers", type=int, default=8)
    r.add_argument("--arrays", type=Path, default=None)
    a = ap.parse_args(argv)
    arrays = a.arrays or _ra().Q3_ARRAYS_DEFAULT
    out = a.out or (OUT if a.profile == "full" else ROOT / "scratch" / "r5_cccc_q4_c_e4_ci")
    res = run_campaign(a.profile, out, a.workers if a.profile == "full" else 1, arrays)
    print(json.dumps(res["summary"], indent=1))
    s = res["summary"]
    return 0 if s["c3plus_inequality"]["FAIL"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
