"""QA - R4, N = 4-6: independent checks of the Builder study (docs/R4_N4_N6_STUDY.report.md; G3.3, G4; G2).

Object: research/r4-n4-n6 @ 6c7a80de00140fbb714e6cb0577bed232f3bc644 (results computed on 69e26bb, table generator fixed
in 6c7a80d). This file is QA code; it changes nothing in src/, in the Builder script or in the committed Builder results.
The heavy parts need python-flint (arb ball arithmetic) and mpmath in a temporary QA environment (as D-040, O-2); they are
not project dependencies. Reproduction (repository root, PYTHONPATH=src):
    python scripts/r4_n4_n6_study_qa.py integrity        # K-1, K-2: committed data, tables, report numbers, controller
    python scripts/r4_n4_n6_study_qa.py rho1             # K-3: exact rank of the first half-step field (all 210 cells)
    python scripts/r4_n4_n6_study_qa.py nodal            # K-4: exact discrete sequence by 2D Galerkin, small grids
    python scripts/r4_n4_n6_study_qa.py cont             # K-5: continuous sequence in arb, closed form s = 1
    python scripts/r4_n4_n6_study_qa.py repro            # K-6, K-7: rerun of committed rows, run_coupled, orders
Outputs: docs/results/r4_n4_n6_qa/ (CSV, JSON).

What is independent of what (task, item 2В: a control is not called independent if it uses the same operator and the
same computational path):
- K-3 (rho1): closed form of the exact discrete first half-step (R4_NUMERICAL_SPEC 4.2-4.3, ЧВ4-8, ЧВ4-9: diagonal A, B,
  C, F of the odd sines; V^_ik = F_i c_k / (D (a/2) (mu_k + lambda_i)^2)) in arb, singular values certified by Sylvester
  inertia. Shares with the solver only the accepted discrete problem; no solver code, no DST, no LAPACK.
- K-4 (nodal): the exact discrete KK-VIM sequence of the same finite problem in arb, each half-step the stationarity of
  Pi_h = 1/2 a_h - l_h, a_h(w, v) = D h_x h_y sum (Delta_h w)(Delta_h v), Delta_h = L_x (x) I + I (x) L_y (2.1), on the
  half-step space span(U) (x) V_h^y (or V_h^x (x) span(V)), assembled by the two-dimensional five-point stencil applied to
  every basis field and the two-dimensional grid sums. No coefficient matrices A, B, C, F, no modal splitting ЧФ-3, no
  DST, no QR/SVD of the solver; the given set is replaced by an orthonormal basis of its span (field unchanged, 5.4).
  Common with the solver: the definition of the discrete problem (grid, L, Pi_h, odd sine start) and the alternation
  Y -> X -> Y with all N functions found jointly (method level A). Feasible only on small grids.
- K-5 (cont): the continuous recursion of R4_NUMERICAL_SPEC 9.2 coded anew in arb (same formula as
  kkvim_reference.r4_continuous_sequence, different code and arithmetic); the energy error by the projection identity
  |w - omega_0|^2_{H_T} = q J(omega_0) - q J(w) (J = integral of the field; exact for every half-step field, which is the
  energy projection of omega_0), J(omega_0) by the Navier double sum of 9.5.2 with the strict tail
  (a^4 + b^4) pi^2 / (80 M^5); s = 1 also untruncated by the closed form with a strict tail. Shares the formula 9.2.
- K-6 (repro): reruns the Builder functions (NOT independent: reproducibility only).

Criteria, fixed in this file before the QA computations (no value is chosen from the results; AGENTS.md, section 4):
K-1  integrity: 210 cells (198 + 12), 7 854 half-step rows, 1 386 comparison rows; the 14 stops are N = 6, r in
     {1/5, 1/3}, s = 1, PROVISIONAL_RANK_GUARD_STOP on all seven grids each; tables.md is the generator's function of
     the CSV files; every number of the report checked here equals the recomputed one within half a unit of its last
     printed digit. FAIL otherwise (a wrong number is a finding, its weight is judged in the QA report).
K-2  controller: the guard decisions in steps.csv follow D-048.5 from the logged rho_N, kappa_rep, sigma_min(R_V D_c)
     (decisions within 1e-5 relative of the threshold are not judged: 7 printed digits); no S-2, no
     PROVISIONAL_FIELD_STABILIZED, no CONVERGED; the D-048.1 mandatory flag is empty for N >= 4; the energy test is
     resolved iff m_s >= 1; every energy increase lies after the first stabilization and below tau_s.
K-3  rank at s = 1: rho*_N(W*^1) > 0 certified in every cell (theorem ЧВ4-9: n_x, n_y >= 2N). The stop is called a
     property of the exact discrete field if the guard decision with the exact values (kappa_rep* = 1) equals the solver's
     decision in every cell. Agreement |rho^ - rho*| / rho* is reported (observation, no bound is proven: D-046).
K-4  implementation (small grids): true relative field error E_impl(s) = max|W_solver - W*| / max|W*| of the solver fields
     (Builder's run_cell) for all completed s; the Builder V5 statement "<= 1.5e-12" is CONFIRMED at these grids if
     max_s E_impl <= 1.5e-12, CONSISTENT_IN_ORDER if <= 1.5e-11, else DISCREPANCY. Energy: |Pi_h(solver) - Pi_h*| /
     |Pi_h*| <= 4.9e-15 (the Builder V5 energy statement) CONFIRMED, else as above with factor 10. The exact sequence
     energy is certified non-increasing (ВБ-17) where the ball width allows.
K-5  continuous: (а) Builder's double recursion (M = 4001) agrees with the arb recursion of the same truncated problem
     within its own rounding model K u max|w_cont| (K = number of odd modes) on the 41 x 41 points; (б) the committed
     e_rel equals the QA value within the committed budget (D_strict_rel + D_model_rel) e_rel plus the ball radius;
     (в) the continuous rho_N quoted in the report (s = 1: 1.741e-12 at r = 1/5, 7.39e-11 at r = 1/3; s = 2: 9.8e-9 at
     r = 1/5, N = 6) agrees with the certified value to the printed digits; (г) the committed d_disc_c and d_method_c at
     the centre are reproduced from the arb w_cont within 1e-5 relative (+ 1e-14 absolute: 7 printed digits).
K-6  reproducibility: rerun cells reproduce the committed rows of cells.csv (without seconds), steps.csv and compare.csv
     exactly as text on this platform (same Python, NumPy, SciPy); otherwise the largest relative difference is
     reported. Control N = 2-3: every field of every step log equal to run_coupled (deep comparison, not a field list).
K-7  order: observed self-convergence order of the solver fields alone, p_self = log2(|w_n - w_2n| / |w_2n - w_4n|) on
     the 41 x 41 points, in [1.9, 2.1] and within 0.05 of the committed p (observation; no Richardson, РВ4-7); s = 1:
     E_disc on the 41 x 41 points from the two exact closed forms (discrete and continuous, arb) equals the committed
     orders.csv value within 1e-6 relative.
Statuses: PASS / FAIL per check, NOT_RUN if a check could not be run; nothing here is a normative acceptance of N = 4-6.
"""

import argparse
import csv
import dataclasses
import hashlib
import importlib.util
import json
import math
import os
import platform
import subprocess
import sys
import time
from fractions import Fraction
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "docs" / "results" / "r4_n4_n6"
OUT_DEFAULT = ROOT / "docs" / "results" / "r4_n4_n6_qa"
U = 2.0**-53
K_RESEARCH = 1_000_000
R_TEN = ("1/5", "1/3", "1/2", "2/3", "1", "7/5", "3/2", "2", "3", "5")
FAMILY_I = (40, 80, 160, 320, 640, 1280)
FAMILY_II = {"1/5": (1600, 320), "1/3": (960, 320), "1/2": (640, 320), "2": (320, 640), "3": (320, 960), "5": (320, 1600)}
S_LIST = (1, 2, 3, 4, 10, 20, 40)
SAMPLE = 40

# nodal (K-4): small grids; family-II-like h_x = h_y grids for r = 1/5 and 5
NODAL_CELLS = [(N, lab, 40, 40) for N in (4, 5, 6) for lab in ("1/5", "1/3", "1", "5")] + \
              [(N, "1/5", 60, 12) for N in (4, 5, 6)] + [(N, "5", 12, 60) for N in (4, 5, 6)]
NODAL_S = 40
# The span of a received set with small rho_N is sensitive like 1/rho_N, so the ball radius grows by roughly that factor
# per half-step in ANY rigorous evaluation of the sequence (not an artefact of the solve); the working precision is
# therefore started by N and raised once if the final relative radius exceeds 2^-100.
NODAL_PREC = {4: 1536, 5: 2048, 6: 2560}
# cont (K-5)
CONT_PAIRS = [(N, lab) for N in (4, 5, 6) for lab in ("1/5", "1", "5")] + [(6, "1/3")]
CONT_M = 4001
OMEGA0_M = 12001  # J(omega_0): strict tail (a^4 + b^4) pi^2 / (80 M^5) far below the e_rel to be compared
# repro (K-6, K-7): chains for the order, the 14 stops partly, family II, control N = 2-3
CHAINS = [(6, "1"), (5, "5"), (4, "1/5"), (6, "1/2"), (5, "1/3")]
CHAIN_N = (160, 320, 640, 1280)
REPRO_EXTRA = [(6, "1/5", "I", 1280, 1280), (6, "1/3", "II", 960, 320), (6, "1/2", "II", 640, 320),
               (5, "3", "II", 320, 960), (4, "5", "II", 320, 1600), (6, "5", "I", 1280, 1280), (5, "5", "I", 40, 40)]
CONTROL = [(N, lab, "I", n, n) for lab in ("1/5", "1", "5") for n in (160, 640) for N in (2, 3)]


# ======================================================================================== common
def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    keys = []
    for row in rows:
        keys += [k for k in row if k not in keys]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
        w.writeheader()
        for row in rows:
            w.writerow({k: _text(v) for k, v in row.items()})


def _text(v):
    if isinstance(v, float):
        return repr(v)
    return "" if v is None else v


def fl(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return math.nan


def up(x):
    """Upper end of a ball as a float rounded outward."""
    return math.nextafter(float(x.upper()), math.inf)


def lo(x):
    return math.nextafter(float(x.lower()), -math.inf)


def builder_study():
    spec = importlib.util.spec_from_file_location("r4_n4_n6_study", ROOT / "scripts" / "r4_n4_n6_study.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def git_head():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def env_info():
    info = {"python": sys.version.split()[0], "numpy": np.__version__, "platform": platform.platform(),
            "git_head": git_head(), "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    try:
        import scipy

        info["scipy"] = scipy.__version__
    except ImportError:
        pass
    try:
        import flint
        import mpmath

        info["python_flint"] = flint.__version__
        info["mpmath"] = mpmath.__version__
    except ImportError:
        pass
    import kkvim

    info["kkvim_path"] = str(Path(kkvim.__file__).resolve().parent)
    return info


# ======================================================================================== arb: certified eigenvalues
def _flint():
    import flint

    return flint


def _n_positive(G, t):
    """Number of eigenvalues of the symmetric ball matrix G above t (Sylvester inertia of LDL^T); None if a pivot
    contains 0."""
    f = _flint()
    n = G.nrows()
    A = [[G[i, k] - (t if i == k else 0) for k in range(n)] for i in range(n)]
    L = [[f.arb(0)] * n for _ in range(n)]
    d = []
    for i in range(n):
        for k in range(i):
            s = A[i][k]
            for m in range(k):
                s -= L[i][m] * L[k][m] * d[m]
            L[i][k] = s / d[k]
        di = A[i][i]
        for m in range(i):
            di -= L[i][m] * L[i][m] * d[m]
        if di.contains(0):
            return None
        d.append(di)
    return sum(1 for x in d if x > 0)


def eig_brackets(G):
    """Certified brackets (t_lo, t_hi) of all eigenvalues of the symmetric ball matrix G (descending): at least j
    eigenvalues > t_lo and at most j - 1 eigenvalues > t_hi. Guesses by mpmath on the midpoints."""
    import mpmath as mp

    f = _flint()
    prec = f.ctx.prec
    n = G.nrows()
    mp.mp.prec = prec + 64
    digits = int(prec * 0.302) + 10
    A = mp.matrix(n, n)
    for i in range(n):
        for k in range(n):
            A[i, k] = mp.mpf(G[i, k].mid().str(digits, radius=False))
    A = (A + A.T) / 2
    ev = sorted(mp.eigsy(A, eigvals_only=True), reverse=True)
    out = []
    for j, lam in enumerate(ev, start=1):
        for e in range(prec // 3, 8, -8):
            eps = mp.mpf(2) ** (-e)
            t_lo = f.arb(mp.nstr(lam * (1 - eps) if lam > 0 else lam - eps * abs(ev[0]), digits))
            t_hi = f.arb(mp.nstr(lam * (1 + eps) + eps * abs(ev[0]) * mp.mpf(2) ** (-prec // 3), digits))
            n_lo, n_hi = _n_positive(G, t_lo), _n_positive(G, t_hi)
            if n_lo is not None and n_hi is not None and n_lo >= j and n_hi <= j - 1:
                out.append((t_lo, t_hi))
                break
        else:
            return None
    return out


def sv_ratio(G):
    """(rho_lo, rho_hi) of sqrt(lambda_min / lambda_max) for a Gram matrix G (certified), and the brackets."""
    br = eig_brackets(G)
    if br is None:
        return None, None, None
    (l1_lo, l1_hi), (lN_lo, lN_hi) = br[0], br[-1]
    if not lN_lo > 0:
        rho_lo = 0.0
    else:
        rho_lo = lo((lN_lo / l1_hi).sqrt())
    rho_hi = up((lN_hi / l1_lo).sqrt())
    return rho_lo, rho_hi, br


def normalized(G):
    """D G D with D = diag(1/sqrt(G_ii)) (unit columns)."""
    f = _flint()
    n = G.nrows()
    d = [1 / G[i, i].sqrt() for i in range(n)]
    return f.arb_mat([[G[i, k] * d[i] * d[k] for k in range(n)] for i in range(n)])


def chol(G):
    f = _flint()
    n = G.nrows()
    L = [[f.arb(0)] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = G[i, j]
            for k in range(j):
                s -= L[i][k] * L[j][k]
            L[i][j] = s.sqrt() if i == j else s / L[j][j]
    return f.arb_mat(L)


def rel_rad(M):
    """max radius / max |mid| of an arb_mat."""
    rmax = max(float(M[i, j].rad()) for i in range(M.nrows()) for j in range(M.ncols()))
    amax = max(abs(float(M[i, j].mid())) for i in range(M.nrows()) for j in range(M.ncols()))
    return rmax / amax if amax > 0 else math.inf


# ======================================================================================== K-3: rank of W*^1, closed form
def first_halfstep_hat(N, r, nx, ny, q=1, D=1):
    """V^ (N x odd k) of the exact discrete first half-step (ЧВ4-8, ЧВ4-9): V^_ik = F_i c_k / (D (a/2) (mu_k + lam_i)^2),
    lam_i = mu^x_{2i-1}, F_i = q h_x cot((2i-1) pi / (2 n_x)), c_k = sqrt(2/n_y) cot(k pi / (2 n_y)), k odd."""
    f = _flint()
    a, b = f.arb(1), f.arb(f.fmpq(Fraction(r).numerator, Fraction(r).denominator))
    hx, hy = a / nx, b / ny
    lam = [4 / (hx * hx) * f.arb(f.fmpq(2 * i + 1, 2 * nx)).sin_pi() ** 2 for i in range(N)]
    F = [q * hx * f.arb(f.fmpq(2 * i + 1, 2 * nx)).cot_pi() for i in range(N)]
    ks = list(range(1, ny, 2))
    mu = [4 / (hy * hy) * f.arb(f.fmpq(k, 2 * ny)).sin_pi() ** 2 for k in ks]
    c = [(f.arb(2) / ny).sqrt() * f.arb(f.fmpq(k, 2 * ny)).cot_pi() for k in ks]
    return f.arb_mat([[F[i] * c[j] / (D * a / 2 * (mu[j] + lam[i]) ** 2) for j in range(len(ks))] for i in range(N)]), ks


def rho1_cell(args):
    N, lab, fam, nx, ny = args
    f = _flint()
    for prec in (384, 768, 1536):
        f.ctx.prec = prec
        V, _ = first_halfstep_hat(N, Fraction(lab), nx, ny)
        G = V * V.transpose()
        rho_lo, rho_hi, br = sv_ratio(G)
        set_lo, set_hi, sbr = sv_ratio(normalized(G))
        if br is not None and sbr is not None:
            set_lo = lo(sbr[-1][0].sqrt()) if sbr[-1][0] > 0 else 0.0
            set_hi = up(sbr[-1][1].sqrt())
            return {"N": N, "r_label": lab, "family": fam, "nx": nx, "ny": ny, "prec": prec, "rho_star_lo": rho_lo,
                    "rho_star_hi": rho_hi, "set_star_lo": set_lo, "set_star_hi": set_hi}
    return {"N": N, "r_label": lab, "family": fam, "nx": nx, "ny": ny, "prec": None}


def cont_first_hat(N, r, M, q=1, D=1):
    """c^_in of the untruncated continuous first half-step, odd n <= M (R4_NUMERICAL_SPEC 9.2 with the diagonal
    coefficients of the odd sines, MATH_SPEC Б.16): c^_in = 16 q / (pi^2 D n m_i (nu_n^2 + alpha_i^2)^2), and the strict
    bound of |T|_F^2 of the columns n > M: sum_i (16 q b^4 / (pi^6 D m_i))^2 / (18 M^9)."""
    f = _flint()
    pi = f.arb.pi()
    b = f.arb(f.fmpq(Fraction(r).numerator, Fraction(r).denominator))
    ms = [2 * i + 1 for i in range(N)]
    ns = list(range(1, M + 1, 2))
    al2 = [(m * pi) ** 2 for m in ms]
    nu2 = [(n * pi / b) ** 2 for n in ns]
    C = f.arb_mat([[16 * q / (pi * pi * D * n * m * (nu2[j] + al2[i]) ** 2) for j, n in enumerate(ns)]
                   for i, m in enumerate(ms)])
    tail = sum((16 * q * b**4 / (pi**6 * D * m)) ** 2 for m in ms) / (18 * f.arb(M) ** 9)
    return C, tail


def cont_rho1(args):
    N, lab, M = args
    f = _flint()
    for prec in (384, 768, 1536):
        f.ctx.prec = prec
        C, tail = cont_first_hat(N, Fraction(lab), M)
        br = eig_brackets(C * C.transpose())
        if br is None:
            continue
        (l1_lo, _), (lN_lo, lN_hi) = br[0], br[-1]
        l1_hi = br[0][1] + tail  # lambda_k(C C^T) <= lambda_k(C_M C_M^T) + |T|_F^2 (Weyl, T T^T >= 0)
        lN_hi = lN_hi + tail
        return {"N": N, "r_label": lab, "M": M, "prec": prec, "tail_F2_up": up(tail),
                "rho_cont_lo": lo((lN_lo / l1_hi).sqrt()) if lN_lo > 0 else 0.0, "rho_cont_hi": up((lN_hi / l1_lo).sqrt())}
    return {"N": N, "r_label": lab, "M": M, "prec": None}


def _decision(v_lo, v_hi, thr):
    """Guard decision 'value > thr' from a certified bracket: True, False, or 'UNCERTAIN'."""
    if v_lo > thr:
        return True
    if v_hi <= thr:
        return False
    return "UNCERTAIN"


def _max_up(values):
    return max(up(v) for v in values)


def _max_lo(values):
    return max(lo(v) for v in values)


def rho1(out, workers):
    steps = read_csv(BUILDER / "steps.csv")
    first = {(s["N"], s["r_label"], s["family"], s["nx"], s["ny"]): s for s in steps if s["s"] == "1"}
    cells = [(int(c["N"]), c["r_label"], c["family"], int(c["nx"]), int(c["ny"])) for c in read_csv(BUILDER / "cells.csv")]
    extra = [(6, lab, "X", n, n) for lab in ("1/5", "1/3") for n in (2560, 5120)]  # finer grids, not in the plan
    with Pool(workers) as pool:
        rows = pool.map(rho1_cell, cells + extra)
        crow = pool.map(cont_rho1, [(N, lab, CONT_M) for N in (4, 5, 6) for lab in R_TEN] + [(N, lab, 2 * CONT_M + 1)
                                                                                           for N in (6,) for lab in ("1/5", "1/3")])
    thr = K_RESEARCH * U
    for row in rows:
        s = first.get((str(row["N"]), row["r_label"], row["family"], str(row["nx"]), str(row["ny"])))
        if s is None:
            continue
        rho_hat, kap, set_hat = fl(s["rho_N"]), fl(s["kappa_rep"]), fl(s["set_sigma_min"])
        mid = 0.5 * (row["rho_star_lo"] + row["rho_star_hi"])
        smid = 0.5 * (row["set_star_lo"] + row["set_star_hi"])
        row.update({"rho_hat": rho_hat, "rho_rel_err": (rho_hat - mid) / mid, "kappa_rep_hat": kap,
                    "set_hat": set_hat, "set_rel_err": (set_hat - smid) / smid,
                    "guard_rank_solver": s.get("guard_rank_ok", ""), "guard_set_solver": s.get("guard_set_ok", ""),
                    "guard_rank_exact": _decision(row["rho_star_lo"], row["rho_star_hi"], thr),
                    "guard_set_exact": _decision(row["set_star_lo"], row["set_star_hi"], thr),
                    "rho_star_over_threshold": mid / thr, "set_star_over_threshold": smid / thr})
    write_csv(out / "rho_s1.csv", rows)
    write_csv(out / "rho_s1_continuous.csv", crow)
    return rows, crow


# ======================================================================================== K-4: nodal 2D Galerkin in arb
class NodalProblem:
    """The discrete problem of R4_NUMERICAL_SPEC 2.1 on G(1, n_x) x G(r, n_y) in arb (q = D = 1)."""

    def __init__(self, r, nx, ny):
        f = _flint()
        self.f = f
        self.nx, self.ny = nx, ny
        self.nxi, self.nyi = nx - 1, ny - 1
        b = f.arb(f.fmpq(Fraction(r).numerator, Fraction(r).denominator))
        self.hx, self.hy = f.arb(1) / nx, b / ny
        self.ihx2, self.ihy2 = 1 / (self.hx * self.hx), 1 / (self.hy * self.hy)
        self.c0 = self.hx * self.hy  # D h_x h_y

    def lap(self, F):
        """Delta_h F for a sparse field {(p, q): value} (five-point stencil, zero boundary; symmetric, so scattering
        each value equals the stencil sum)."""
        out = {}
        cxy = -2 * (self.ihx2 + self.ihy2)
        for (p, q), v in F.items():
            for pp, qq, c in ((p, q, cxy), (p - 1, q, self.ihx2), (p + 1, q, self.ihx2), (p, q - 1, self.ihy2),
                              (p, q + 1, self.ihy2)):
                if 0 <= pp < self.nxi and 0 <= qq < self.nyi:
                    out[(pp, qq)] = out.get((pp, qq), 0) + c * v
        return out

    def halfstep(self, given, direction):
        """Stationarity of Pi_h on span(given) (x) V_h (direction "Y": given x-set, unknown y-set) or V_h (x) span(given)
        ("X"): returns the unknown set (rows = nodes of the unknown axis, columns = N)."""
        f = self.f
        N = given.ncols()
        n_unk = self.nyi if direction == "Y" else self.nxi
        n_giv = self.nxi if direction == "Y" else self.nyi
        cols, rhs = [], []
        for j in range(N):
            gsum = sum((given[p, j] for p in range(n_giv)), f.arb(0))
            for k in range(n_unk):
                F = {(p, k): given[p, j] for p in range(n_giv)} if direction == "Y" else \
                    {(k, p): given[p, j] for p in range(n_giv)}
                cols.append(self.lap(F))
                rhs.append(self.c0 * gsum)  # l_h(basis) = q h_x h_y sum of the basis field
        size = self.nxi * self.nyi
        zero = f.arb(0)
        B = f.arb_mat([[c.get((idx // self.nyi, idx % self.nyi), zero) for c in cols] for idx in range(size)])
        G = (B.transpose() * B) * self.c0
        x = G.solve(f.arb_mat([[v] for v in rhs]))
        return f.arb_mat([[x[j * n_unk + k, 0] for j in range(N)] for k in range(n_unk)])

    def field(self, X, Y):
        return X * Y.transpose()

    def energy(self, W):
        """Pi_h(W) = 1/2 D h_x h_y sum (Delta_h W)^2 - q h_x h_y sum W (dense interior field)."""
        f = self.f
        nxi, nyi = self.nxi, self.nyi
        s2, s1 = f.arb(0), f.arb(0)
        for p in range(nxi):
            for q in range(nyi):
                w = W[p, q]
                v = -2 * (self.ihx2 + self.ihy2) * w
                if p > 0:
                    v += self.ihx2 * W[p - 1, q]
                if p < nxi - 1:
                    v += self.ihx2 * W[p + 1, q]
                if q > 0:
                    v += self.ihy2 * W[p, q - 1]
                if q < nyi - 1:
                    v += self.ihy2 * W[p, q + 1]
                s2 += v * v
                s1 += w
        return self.c0 * s2 / 2 - self.c0 * s1, self.c0 * s2

    def odd_sines(self, N):
        f = self.f
        return f.arb_mat([[f.arb(f.fmpq(((2 * i + 1) * l) % (2 * self.nx), self.nx)).sin_pi() for i in range(N)]
                          for l in range(1, self.nx)])


def orthonormal(S):
    """S L^{-T} with S^T S = L L^T (span unchanged)."""
    L = chol(S.transpose() * S)
    return S * L.transpose().inv()


def nodal_sequence(N, lab, nx, ny, S, prec):
    """Exact discrete sequence W*(s), s = 1..S, with certified rho_N, set sigma_min, Pi_h and eps_E per half-step."""
    f = _flint()
    f.ctx.prec = prec
    P = NodalProblem(Fraction(lab), nx, ny)
    X = P.odd_sines(N)
    Y = None
    out = []
    W_prev, norm_prev = None, None
    for s in range(1, S + 1):
        if s % 2:
            given = X if s == 1 else orthonormal(X)
            Y = P.halfstep(given, "Y")
            X = given
            received = Y
        else:
            given = orthonormal(Y)
            X = P.halfstep(given, "X")
            Y = given
            received = X
        W = P.field(X, Y)
        Gr = received.transpose() * received
        rho_lo, rho_hi, _ = sv_ratio(Gr)
        sbr = eig_brackets(normalized(Gr))
        Pi, a2 = P.energy(W)
        row = {"s": s, "W": W, "Pi": Pi, "rho_lo": rho_lo, "rho_hi": rho_hi,
               "set_lo": lo(sbr[-1][0].sqrt()) if sbr and sbr[-1][0] > 0 else (0.0 if sbr else None),
               "set_hi": up(sbr[-1][1].sqrt()) if sbr else None, "rel_rad_W": rel_rad(W)}
        if W_prev is not None:
            d = {(p, q): W[p, q] - W_prev[p, q] for p in range(P.nxi) for q in range(P.nyi)}
            ld = P.lap(d)
            dn2 = P.c0 * sum((v * v for v in ld.values()), f.arb(0))
            row["eps_E_up"] = up((dn2 / a2).sqrt()) if a2 > 0 else math.inf
        out.append(row)
        W_prev = W
    return out, P


def nodal_cell(args):
    """K-4 for one small cell: exact sequence (adaptive precision) against the solver (Builder's run_cell, the research
    orchestration) and against the discrete control kkvim_reference.r4_spectral_control used by Builder in V5."""
    N, lab, nx, ny = args
    t0 = time.perf_counter()
    study = builder_study()
    from kkvim_reference.r4_spectral_control import SpectralGrid, run_control

    cell = study.run_cell(N, lab, "I" if nx == ny else "II", nx, ny, half_steps=NODAL_S, keep=tuple(range(1, NODAL_S + 1)))
    try:
        ctrl = run_control(SpectralGrid(1.0, float(Fraction(lab)), nx, ny), N, NODAL_S)
    except (np.linalg.LinAlgError, ValueError):
        ctrl = None
    prec = NODAL_PREC[N]
    for _ in range(3):
        try:
            seq, P = nodal_sequence(N, lab, nx, ny, NODAL_S, prec)
            rad = seq[-1]["rel_rad_W"]
        except (ZeroDivisionError, ValueError, ArithmeticError):
            seq, rad = None, math.inf
        if seq is not None and rad < 2.0**-100:
            break
        if math.isfinite(rad) and rad > 0:
            prec += 256 * math.ceil((math.log2(rad) + 140) / 256)
        else:
            prec *= 2
    f = _flint()
    rows = []
    for e in seq:
        s, W = e["s"], e["W"]
        idx = [(p, q) for p in range(P.nxi) for q in range(P.nyi)]
        wmax = _max_lo(abs(W[p, q]) for p, q in idx)  # lower bound of max|W*|
        row = {"N": N, "r_label": lab, "nx": nx, "ny": ny, "s": s, "prec": prec, "rel_rad_W": e["rel_rad_W"],
               "Pi_star": float(e["Pi"].mid()), "Pi_star_rad": float(e["Pi"].rad()), "rho_star_lo": e["rho_lo"],
               "rho_star_hi": e["rho_hi"], "set_star_lo": e["set_lo"], "set_star_hi": e["set_hi"],
               "eps_E_star_up": e.get("eps_E_up"), "solver_status": cell.status, "solver_s_reached": cell.s_reached}
        if s >= 2:
            dP = e["Pi"] - seq[s - 2]["Pi"]
            row["dPi_star_up"] = up(dP)
            row["energy_decrease_certified"] = bool(dP < 0)
        if s in cell.fields:
            Ws = cell.fields[s][1:-1, 1:-1]
            dmax = _max_up(abs(f.arb(float(Ws[p, q])) - W[p, q]) for p, q in idx)
            row["E_impl_true_up"] = math.nextafter(dmax / wmax, math.inf)
            log = cell.steps[s - 1]
            Pi_s = f.arb(float(log.energy.P))
            row["Pi_solver"] = float(log.energy.P)
            row["Pi_rel_err_up"] = up(abs(Pi_s - e["Pi"]) / abs(e["Pi"]))
            row["rho_hat"] = float(log.rank.rho_N)
            if e["rho_lo"] is not None and e["rho_hi"] is not None:
                mid = 0.5 * (e["rho_lo"] + e["rho_hi"])
                row["rho_rel_err"] = (float(log.rank.rho_N) - mid) / mid
            row["w_centre_solver"] = float(cell.fields[s][nx // 2, ny // 2])
        if ctrl is not None and s <= len(ctrl):
            Wc = ctrl[s - 1].W
            dmax = _max_up(abs(f.arb(float(Wc[p, q])) - W[p, q]) for p, q in idx)
            row["E_control_true_up"] = math.nextafter(dmax / wmax, math.inf)
            if s in cell.fields:
                row["E_solver_minus_control"] = float(np.max(np.abs(cell.fields[s][1:-1, 1:-1] - Wc)) / np.max(np.abs(Wc)))
        row["w_centre_star"] = float(W[nx // 2 - 1, ny // 2 - 1].mid())
        rows.append(row)
    print(f"nodal N={N} r={lab} {nx}x{ny}: prec {prec}, solver {cell.status} s={cell.s_reached}, "
          f"{time.perf_counter() - t0:.0f} s", flush=True)
    return rows


def nodal(out, workers):
    with Pool(min(workers, len(NODAL_CELLS))) as pool:
        parts = pool.map(nodal_cell, NODAL_CELLS, chunksize=1)
    rows = [r for p in parts for r in p]
    write_csv(out / "nodal.csv", rows)
    return rows


# ======================================================================================== K-5: continuous, arb
def _sin_table(modes, den, npts):
    """sin(m pi p / den) for p = 0..npts, odd m in modes (exact integer reduction of the argument)."""
    f = _flint()
    return f.arb_mat([[f.arb(f.fmpq((m * p) % (2 * den), den)).sin_pi() for m in modes] for p in range(npts + 1)])


def omega0_integral(r, M):
    """J(omega_0) = int omega_0 = (64 q a b / (pi^8 D)) S, S = sum_{m,n odd} 1/(m^2 n^2 (m^2/a^2 + n^2/b^2)^2) (9.5.2);
    S in [S_M, S_M + (a^4 + b^4) pi^2 / (80 M^5)] (odd m or n > M; sum_{odd n} n^-2 = pi^2/8)."""
    f = _flint()
    a, b = f.arb(1), f.arb(f.fmpq(Fraction(r).numerator, Fraction(r).denominator))
    ib2 = 1 / (b * b)
    odd = list(range(1, M + 1, 2))
    n2 = [f.arb(n * n) for n in odd]
    v = [x * ib2 for x in n2]
    S = f.arb(0)
    for m in odd:
        A = f.arb(m * m)
        part = f.arb(0)
        for k in range(len(odd)):
            t = A + v[k]
            part += 1 / (n2[k] * t * t)
        S += part / A
    pi = f.arb.pi()
    tail = (a**4 + b**4) * pi**2 / (80 * f.arb(M) ** 5)
    return 64 * a * b / pi**8 * (S + f.arb(0).union(tail))


def cont_sequence_arb(N, lab, S, M):
    """Truncated continuous recursion of R4_NUMERICAL_SPEC 9.2 in arb (odd modes <= M both directions; the given set is
    orthonormalized before each half-step, as in the Builder reference; field invariant)."""
    f = _flint()
    pi = f.arb.pi()
    a, b = f.arb(1), f.arb(f.fmpq(Fraction(lab).numerator, Fraction(lab).denominator))
    modes = list(range(1, M + 1, 2))
    K = len(modes)
    k_x = [m * pi / a for m in modes]
    k_y = [m * pi / b for m in modes]
    w_x = [2 * a / (m * pi) for m in modes]
    w_y = [2 * b / (m * pi) for m in modes]
    x = f.arb_mat([[1 if (i == j) else 0 for j in range(N)] for i in range(K)])
    y = None
    steps = []

    def gram(c, length, k):
        c2 = f.arb_mat([[c[i, j] * k[i] ** 2 for j in range(N)] for i in range(K)])
        c4 = f.arb_mat([[c2[i, j] * k[i] ** 2 for j in range(N)] for i in range(K)])
        ct = c.transpose()
        return (ct * c) * (length / 2), (ct * c2) * (length / 2), (ct * c4) * (length / 2)

    def half(c, length, k_g, w_g, k_new):
        A_, B_, C_ = gram(c, length, k_g)
        F = f.arb_mat([[sum((c[i, j] * w_g[i] for i in range(K)), f.arb(0))] for j in range(N)])
        new = []
        for n, nu in zip(modes, k_new):
            nu2 = nu * nu
            Mn = A_ * (nu2 * nu2) + B_ * (2 * nu2) + C_
            v = Mn.solve(F) * (4 / (n * pi))
            new.append([v[j, 0] for j in range(N)])
        return f.arb_mat(new)

    for s in range(1, S + 1):
        if s % 2:
            x = orthonormal(x) if s > 1 else x
            y = half(x, a, k_x, w_x, k_y)
            received = y
        else:
            y = orthonormal(y)
            x = half(y, b, k_y, w_y, k_x)
            received = x
        Ix = [sum((x[i, j] * w_x[i] for i in range(K)), f.arb(0)) for j in range(N)]
        Iy = [sum((y[i, j] * w_y[i] for i in range(K)), f.arb(0)) for j in range(N)]
        J = sum((Ix[j] * Iy[j] for j in range(N)), f.arb(0))
        Gr = received.transpose() * received  # the given set is orthonormal (in the coefficient metric) for s >= 2
        if s == 1:
            Gr = Gr * (a / 2)  # odd sines: x^T x = I already; scale irrelevant for the ratio
        rho_lo, rho_hi, _ = sv_ratio(Gr)
        steps.append({"s": s, "x": x, "y": y, "J": J, "rho_lo": rho_lo, "rho_hi": rho_hi})
    return steps, modes


def cont_pair(args):
    N, lab = args
    t0 = time.perf_counter()
    f = _flint()
    from kkvim_reference.r4_continuous_sequence import continuous_sequence, field_at_nodes

    r = float(Fraction(lab))
    modes = list(range(1, CONT_M + 1, 2))
    f.ctx.prec = 192
    J0 = omega0_integral(Fraction(lab), OMEGA0_M)
    Sx = _sin_table(modes, SAMPLE, SAMPLE)  # x_p = p a / 40, y_q = q b / 40: sin(m pi p / 40) on both axes
    prec = NODAL_PREC[N]  # same sensitivity of the spans as in K-4
    for _ in range(3):
        f.ctx.prec = prec
        try:
            seq, modes = cont_sequence_arb(N, lab, 40, CONT_M)
            ok = rel_rad(seq[-1]["x"]) < 2.0**-100 and rel_rad(seq[-1]["y"]) < 2.0**-100
        except (ZeroDivisionError, ValueError, ArithmeticError):
            ok = False
        if ok:
            break
        prec *= 2
    dbl = continuous_sequence(1.0, r, N, 40, CONT_M)
    comp = {(c["s"], c["family"], c["nx"]): c for c in read_csv(BUILDER / "compare.csv") if (c["N"], c["r_label"]) == (str(N), lab)}
    cont = {c["s"]: c for c in read_csv(BUILDER / "continuous.csv") if (c["N"], c["r_label"]) == (str(N), lab)}
    rows = []
    for st in seq:
        s = st["s"]
        Wq = (Sx * st["x"]) * (Sx * st["y"]).transpose()
        Wd = field_at_nodes(dbl[s - 1], SAMPLE, SAMPLE)
        pts = [(p, q) for p in range(SAMPLE + 1) for q in range(SAMPLE + 1)]
        wmax = _max_lo(abs(Wq[p, q]) for p, q in pts)
        dmax = _max_up(abs(f.arb(float(Wd[p, q])) - Wq[p, q]) for p, q in pts)
        e_rel = 1 - st["J"] / J0  # |w - omega_0|^2 / |omega_0|^2 for the energy projection w (q = 1)
        cc = cont.get(str(s), {})
        wc = Wq[SAMPLE // 2, SAMPLE // 2]
        row = {"N": N, "r_label": lab, "s": s, "M": CONT_M, "prec": prec,
               "rho_cont_trunc_lo": st["rho_lo"], "rho_cont_trunc_hi": st["rho_hi"],
               "w_cont_centre": float(wc.mid()), "double_minus_arb_max_rel_up": math.nextafter(dmax / wmax, math.inf),
               "double_rounding_model_rel": len(modes) * U, "e_rel_trunc_lo": lo(e_rel), "e_rel_trunc_hi": up(e_rel),
               "e_rel_builder": fl(cc.get("e_rel")),
               "e_rel_builder_budget_rel": fl(cc.get("D_strict_rel")) + fl(cc.get("D_model_rel")),
               "e_rel_builder_diag_rel": fl(cc.get("D_diag_rel"))}
        if s >= 2:
            row["J_increase_certified"] = bool(st["J"] > seq[s - 2]["J"])
        for (ss, fam, nx), c in comp.items():
            if int(ss) != s or fam != "I" or nx != "1280":
                continue
            w0c = fl(c["omega0_centre"])
            row["d_method_c_qa"] = (float(wc.mid()) - w0c) / w0c
            row["d_method_c_builder"] = fl(c["d_method_c"])
            row["d_disc_c_qa"] = (fl(c["w_centre"]) - float(wc.mid())) / w0c
            row["d_disc_c_builder"] = fl(c["d_disc_c"])
        rows.append(row)
    # s = 1 untruncated: J(w^1) = q sum_i (2a/(m_i pi)) sum_n c^_in 2b/(n pi), tail of n > M strict
    C, _ = cont_first_hat(N, Fraction(lab), CONT_M)
    pi = f.arb.pi()
    b = f.arb(f.fmpq(Fraction(lab).numerator, Fraction(lab).denominator))
    ns = list(range(1, CONT_M + 1, 2))
    J1 = f.arb(0)
    tail = f.arb(0)
    for i in range(N):
        m = 2 * i + 1
        J1 += 2 / (m * pi) * sum((C[i, j] * 2 * b / (n * pi) for j, n in enumerate(ns)), f.arb(0))
        # c^_in 2b/(n pi) <= 32 b^5 / (pi^7 m n^6); sum_{odd n > M} n^-6 <= 1/(10 M^5)
        tail += 2 / (m * pi) * 32 * b**5 / (pi**7 * m) / (10 * f.arb(CONT_M) ** 5)
    e1_hi = 1 - J1 / J0
    e1_lo = 1 - (J1 + tail) / J0
    rows[0]["e_rel_untrunc_s1_lo"] = lo(e1_lo)
    rows[0]["e_rel_untrunc_s1_hi"] = up(e1_hi)
    print(f"cont N={N} r={lab}: {time.perf_counter() - t0:.0f} s", flush=True)
    return rows


def cont(out, workers):
    with Pool(min(workers, len(CONT_PAIRS))) as pool:
        parts = pool.map(cont_pair, CONT_PAIRS, chunksize=1)
    rows = [r for p in parts for r in p]
    write_csv(out / "continuous_arb.csv", rows)
    return rows


# ======================================================================================== K-7 (s = 1): exact E_disc
def disc_s1_sample(args):
    """max over the 41 x 41 points of |W*^1 - w^1| for one family-I cell, both exact (arb), and the committed value."""
    N, lab, n = args
    f = _flint()
    f.ctx.prec = 192
    V, ks = first_halfstep_hat(N, Fraction(lab), n, n)
    step = n // SAMPLE
    # discrete: W*(x_p, y_q) = sum_i sin(m_i pi p'/n) sum_k V^_ik sqrt(2/n) sin(k pi q'/n), p' = p n/40
    Sx = f.arb_mat([[f.arb(f.fmpq(((2 * i + 1) * p * step) % (2 * n), n)).sin_pi() for i in range(N)] for p in range(SAMPLE + 1)])
    Sy = f.arb_mat([[f.arb(f.fmpq((k * q * step) % (2 * n), n)).sin_pi() for k in ks] for q in range(SAMPLE + 1)])
    Wd = Sx * (Sy * V.transpose()).transpose() * (f.arb(2) / n).sqrt()
    C, _ = cont_first_hat(N, Fraction(lab), CONT_M)
    ns = list(range(1, CONT_M + 1, 2))
    Sy_c = _sin_table(ns, SAMPLE, SAMPLE)
    Wc = Sx * (Sy_c * C.transpose()).transpose()
    # strict pointwise tail of the continuous field: sum_i sum_{n > M} c^_in <= sum_i 16 b^4 / (pi^6 m_i) / (8 M^4)
    pi = f.arb.pi()
    b = f.arb(f.fmpq(Fraction(lab).numerator, Fraction(lab).denominator))
    tail = sum((16 * b**4 / (pi**6 * (2 * i + 1)) for i in range(N)), f.arb(0)) / (8 * f.arb(CONT_M) ** 4)
    diffs = [abs(Wd[p, q] - Wc[p, q]) for p in range(SAMPLE + 1) for q in range(SAMPLE + 1)]
    d_lo, d_hi = _max_lo(diffs), _max_up(diffs)  # max_pq |.|: lower and upper bounds
    t_up = up(tail)
    return {"N": N, "r_label": lab, "n": n, "E_disc_s1_abs_lo": max(0.0, math.nextafter(d_lo - t_up, -math.inf)),
            "E_disc_s1_abs_hi": math.nextafter(d_hi + t_up, math.inf)}


# ======================================================================================== K-6, K-7: reruns
def deep_equal(a, b, path="", diffs=None, skip=()):
    diffs = [] if diffs is None else diffs
    if dataclasses.is_dataclass(a) and not isinstance(a, type):
        if type(a) is not type(b):
            diffs.append(path)
            return diffs
        for fld in dataclasses.fields(a):
            if fld.name in skip:
                continue
            deep_equal(getattr(a, fld.name), getattr(b, fld.name), f"{path}.{fld.name}", diffs, skip)
    elif isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        if not (isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and a.shape == b.shape
                and np.array_equal(a, b, equal_nan=True)):
            diffs.append(path)
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            diffs.append(path + "[len]")
        for i, (x, y) in enumerate(zip(a, b)):
            deep_equal(x, y, f"{path}[{i}]", diffs, skip)
    elif isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        pass
    elif a != b:
        diffs.append(path)
    return diffs


def repro_cell(args):
    """Rerun one cell with the Builder functions; rows formatted by the Builder writer; samples of the fields."""
    N, lab, fam, nx, ny = args
    t0 = time.perf_counter()
    study = builder_study()
    cfg = {"N": study.N_STUDY, "r": tuple(study.R_TEN), "family_I": study.FAMILY_I, "family_II": study.FAMILY_II,
           "control_r": study.CONTROL_R, "control_n": study.CONTROL_N, "s_run": study.S_RUN, "s_list": study.S_LIST,
           "M": study.M_CONT, "M_check": study.M_CHECK, "M_navier_m": study.M_NAVIER_M, "control_max_n": study.CONTROL_MAX_N}
    cell = study.run_cell(N, lab, fam, nx, ny, half_steps=cfg["s_run"], keep=cfg["s_list"])
    crow, srows = study.cell_rows(cell)
    refs = study.References(cfg)
    comp = study.compare_cell(cell, refs, cfg)
    sub = (slice(0, None, nx // SAMPLE), slice(0, None, ny // SAMPLE))
    samples = {s: cell.fields[s][sub].copy() for s in cell.fields}
    deep = None
    if N in (2, 3):
        import kkvim.coupled_run as cr

        run = cr.run_coupled(1.0, cell.r, 1.0, 1.0, nx, ny, N, transform="fft", half_steps=cfg["s_run"], research_mode=True,
                             stop_on_stabilization=False)
        diffs = []
        if run.status != cell.status or run.first_stabilization_s != cell.first_stabilization_s:
            diffs.append("status/first_stabilization_s")
        if not (np.array_equal(run.x, cell.x) and np.array_equal(run.y, cell.y)):
            diffs.append("x/y")
        deep_equal(list(run.steps), list(cell.steps), "steps", diffs)
        deep = diffs
    for row in comp:
        row.pop("_sample", None)
    return {"key": (N, lab, fam, nx, ny), "cell": crow, "steps": srows, "compare": comp, "samples": samples, "deep": deep,
            "seconds": time.perf_counter() - t0}


def _formatted(study, rows, path):
    study.write_csv(path, rows)
    return read_csv(path)


def repro(out, workers):
    study = builder_study()
    keys = [(N, lab, "I", n, n) for N, lab in CHAINS for n in CHAIN_N] + REPRO_EXTRA + CONTROL
    with Pool(workers) as pool:
        res = pool.map(repro_cell, keys, chunksize=1)
    tmp = out / "_tmp"
    tmp.mkdir(exist_ok=True)
    committed = {"cells": read_csv(BUILDER / "cells.csv"), "steps": read_csv(BUILDER / "steps.csv"),
                 "compare": read_csv(BUILDER / "compare.csv")}

    def key_of(row):
        return (row["N"], row["r_label"], row["family"], row["nx"], row["ny"])

    index = {name: {} for name in committed}
    for name, rows in committed.items():
        for row in rows:
            index[name].setdefault(key_of(row), []).append(row)
    rep_rows = []
    for item in res:
        N, lab, fam, nx, ny = item["key"]
        k = (str(N), lab, fam, str(nx), str(ny))
        rec = {"N": N, "r_label": lab, "family": fam, "nx": nx, "ny": ny, "seconds": round(item["seconds"], 1)}
        for name in ("cells", "steps", "compare"):
            mine = item["cell"] if name == "cells" else item[name]
            mine = [mine] if isinstance(mine, dict) else mine
            fresh = _formatted(study, mine, tmp / f"{name}.csv")
            old = index[name].get(k, [])
            if name == "cells":
                for row in fresh:
                    row.pop("seconds", None)
                    row.pop("equal_run_coupled_bitwise", None)
                old = [{kk: vv for kk, vv in r.items() if kk not in ("seconds", "equal_run_coupled_bitwise")} for r in old]
            n_diff, max_rel = 0, 0.0
            if len(fresh) != len(old):
                n_diff = -1
            for a, b in zip(fresh, old):
                for kk in a:
                    if a[kk] != b.get(kk, ""):
                        n_diff += 1
                        x, y = fl(a[kk]), fl(b.get(kk))
                        if math.isfinite(x) and math.isfinite(y) and y != 0:
                            max_rel = max(max_rel, abs(x - y) / abs(y))
                        else:
                            max_rel = math.inf
            rec[f"{name}_rows"] = len(fresh)
            rec[f"{name}_text_diffs"] = n_diff
            rec[f"{name}_max_rel_diff"] = max_rel
        if item["deep"] is not None:
            rec["run_coupled_deep_diffs"] = len(item["deep"])
            rec["run_coupled_deep_paths"] = ";".join(sorted({p.split("]")[-1] for p in item["deep"]}))[:300]
        rep_rows.append(rec)
    for p in tmp.iterdir():
        p.unlink()
    tmp.rmdir()
    write_csv(out / "repro.csv", rep_rows)
    # K-7 order: self-convergence of the solver fields
    samples = {item["key"]: item["samples"] for item in res}
    orders_c = {(o["N"], o["r_label"], o["s"], o["n"]): o for o in read_csv(BUILDER / "orders.csv")}
    ord_rows = []
    for N, lab in CHAINS:
        for s in (1, 2, 40):
            ws = [samples.get((N, lab, "I", n, n), {}).get(s) for n in CHAIN_N]
            for i in range(len(CHAIN_N) - 2):
                a, b, c = ws[i:i + 3]
                if a is None or b is None or c is None:
                    continue
                d1, d2 = float(np.max(np.abs(a - b))), float(np.max(np.abs(b - c)))
                p = math.log2(d1 / d2) if d1 > 0 and d2 > 0 else math.nan
                o1 = orders_c.get((str(N), lab, str(s), str(CHAIN_N[i])))
                o2 = orders_c.get((str(N), lab, str(s), str(CHAIN_N[i + 1])))
                ord_rows.append({"N": N, "r_label": lab, "s": s, "n": CHAIN_N[i], "p_self": p,
                                 "p_builder_n": fl(o1["p_sample"]) if o1 else math.nan,
                                 "p_builder_2n": fl(o2["p_sample"]) if o2 else math.nan})
    # s = 1: exact discrete minus exact continuous on the 41 x 41 points, all family-I cells N = 4-6
    keys1 = [(N, lab, n) for N in (4, 5, 6) for lab in R_TEN for n in FAMILY_I]
    with Pool(workers) as pool:
        exact = pool.map(disc_s1_sample, keys1, chunksize=4)
    w0 = {(c["N"], c["r_label"], c["nx"]): fl(c["omega0_centre"]) for c in read_csv(BUILDER / "compare.csv")
          if c["family"] == "I" and c["s"] == "1"}
    s1_rows = []
    for e in exact:
        o = orders_c.get((str(e["N"]), e["r_label"], "1", str(e["n"])))
        o_prev = orders_c.get((str(e["N"]), e["r_label"], "1", str(e["n"] // 2)))
        W0 = w0[(str(e["N"]), e["r_label"], str(e["n"]))]
        committed_val = fl(o["E_disc_sample_n"]) if o else (fl(o_prev["E_disc_sample_2n"]) if o_prev else math.nan)
        # orders.csv normalizes by max|omega_0| of the coarser grid's sample: the centre value (Levy), same point
        e.update({"E_disc_s1_rel_lo": e["E_disc_s1_abs_lo"] / W0, "E_disc_s1_rel_hi": e["E_disc_s1_abs_hi"] / W0,
                  "E_disc_s1_builder": committed_val})
        s1_rows.append(e)
    by = {(e["N"], e["r_label"], e["n"]): e for e in s1_rows}
    for e in s1_rows:
        nxt = by.get((e["N"], e["r_label"], 2 * e["n"]))
        if nxt:
            e["p_exact_s1"] = math.log2((e["E_disc_s1_abs_lo"] + e["E_disc_s1_abs_hi"]) /
                                        (nxt["E_disc_s1_abs_lo"] + nxt["E_disc_s1_abs_hi"]))
            o = orders_c.get((str(e["N"]), e["r_label"], "1", str(e["n"])))
            e["p_builder_s1"] = fl(o["p_sample"]) if o else math.nan
    write_csv(out / "orders_qa.csv", ord_rows)
    write_csv(out / "disc_s1_exact.csv", s1_rows)
    return rep_rows, ord_rows, s1_rows


# ======================================================================================== K-1, K-2: integrity
def _claim(table, name, value, printed, digits_rel=None, abs_half=None):
    """value equals the printed number within half a unit of its last printed digit (abs_half), or a relative bound."""
    if abs_half is None:
        abs_half = abs(printed) * (digits_rel if digits_rel is not None else 0.0)
    ok = abs(value - printed) <= abs_half * (1 + 1e-12)
    table.append({"claim": name, "recomputed": value, "printed": printed, "half_unit": abs_half, "ok": ok})
    return ok


def integrity(out):
    cells = read_csv(BUILDER / "cells.csv")
    steps = read_csv(BUILDER / "steps.csv")
    comp = read_csv(BUILDER / "compare.csv")
    cont = read_csv(BUILDER / "continuous.csv")
    ords = read_csv(BUILDER / "orders.csv")
    ctl = read_csv(BUILDER / "control_n23.csv")
    meta = json.loads((BUILDER / "meta.json").read_text(encoding="utf-8"))
    chk = {}
    study = builder_study()
    # ---- K-1: plan coverage and counts
    full = {"N": study.N_STUDY, "r": tuple(study.R_TEN), "family_I": study.FAMILY_I, "family_II": study.FAMILY_II,
            "control_r": study.CONTROL_R, "control_n": study.CONTROL_N}
    plan = study.plan(full)
    got = [(int(c["N"]), c["r_label"], c["family"], int(c["nx"]), int(c["ny"])) for c in cells]
    chk["plan_equals_cells"] = got == plan
    chk["cells_210_198_12"] = (len(cells), sum(int(c["N"]) >= 4 for c in cells), sum(int(c["N"]) <= 3 for c in cells)) == (210, 198, 12)
    chk["steps_7854"] = len(steps) == 7854 == sum(int(c["s_reached"]) for c in cells)
    expected_comp = sum(sum(1 for s in S_LIST if s <= int(c["s_reached"])) for c in cells)
    chk["compare_1386"] = len(comp) == expected_comp == 1386
    stops = [c for c in cells if c["status"] != "RESEARCH_LIMIT_REACHED"]
    grids = lambda lab: {("I", n, n) for n in FAMILY_I} | {("II",) + FAMILY_II[lab]}
    chk["stops_14_N6_r15_r13_s1"] = (len(stops) == 14 and all(
        c["N"] == "6" and c["r_label"] in ("1/5", "1/3") and c["s_reached"] == "1" and c["status"] == "PROVISIONAL_RANK_GUARD_STOP"
        for c in stops) and all({(c["family"], int(c["nx"]), int(c["ny"])) for c in stops if c["r_label"] == lab} == grids(lab)
                                for lab in ("1/5", "1/3")))
    stop_steps = [s for s in steps if s["stop_reason"] == "PROVISIONAL_RANK_GUARD_STOP"]
    chk["stop_rows_before_qr"] = len(stop_steps) == 14 and all(
        s["s"] == "1" and s["qr_done"] == "False" and s["guard_applicable"] == "True" and s["guard_rank_ok"] == "False"
        for s in stop_steps)
    chk["meta_N_SOLVE_unchanged"] = meta["N_SOLVE_before"] == [2, 3] == meta["N_SOLVE_after"]
    chk["meta_run_commit_69e26bb"] = meta["git_head_at_run"].startswith("69e26bb")
    # ---- K-2: controller rules from the logged values
    thr0 = K_RESEARCH * U
    bad_rank, bad_set, judged = [], [], 0
    for s in steps:
        if not s.get("guard_rank_ok"):
            continue
        rho, kap, sset = fl(s["rho_N"]), fl(s["kappa_rep"]), fl(s["set_sigma_min"])
        t = thr0 * kap
        if abs(rho / t - 1) > 1e-5:
            judged += 1
            if (rho > t) != (s["guard_rank_ok"] == "True"):
                bad_rank.append((s["N"], s["r_label"], s["nx"], s["s"]))
            if abs(fl(s["rho_over_threshold"]) / (rho / t) - 1) > 2e-6:
                bad_rank.append(("ratio", s["N"], s["r_label"], s["nx"], s["s"]))
        if s["guard_set_ok"] and abs(sset / thr0 - 1) > 1e-5 and (sset > thr0) != (s["guard_set_ok"] == "True"):
            bad_set.append((s["N"], s["r_label"], s["nx"], s["s"]))
        if (t < 1) != (s["guard_applicable"] == "True"):
            bad_rank.append(("applicable", s["N"], s["s"]))
    chk["guards_follow_D048_5"] = not bad_rank and not bad_set and judged > 7000
    forbidden = {"STOPPED_RANK_NUMERICALLY_UNRESOLVED", "PROVISIONAL_FIELD_STABILIZED", "CONVERGED", "MAX_HALF_STEPS"}
    chk["no_forbidden_statuses"] = not any(c["status"] in forbidden for c in cells) and not any(
        s["stop_reason"] in forbidden for s in steps)
    chk["s2_mandatory_only_N23"] = all((c["s2_mandatory_D048"] == "") == (int(c["N"]) >= 4) for c in cells)
    chk["mark_everywhere"] = all(c["mark"] == "PROVISIONAL / NOT_QUALIFIED" for c in cells) and all(
        s["mark"] == "PROVISIONAL / NOT_QUALIFIED" for s in steps)
    bad_m = [s for s in steps if s["energy_status"] in ("E1_E2_HOLD_PROVISIONAL", "UNRESOLVED")
             and (s["energy_status"] == "UNRESOLVED") != (fl(s["m_s"]) < 1.0) and abs(fl(s["m_s"]) - 1) > 1e-6]
    chk["energy_resolved_iff_m_ge_1"] = not bad_m
    first_stab = {(c["N"], c["r_label"], c["family"], c["nx"], c["ny"]): fl(c["first_stabilization_s"]) for c in cells}
    # increases of Pi_h by s from the printed 16 digits: rounding to 16 digits is monotone, so a printed increase is an
    # increase of the doubles (the exact count of the doubles, cells.csv, can only be larger)
    prev, n_inc, inc_beyond = {}, 0, 0
    inc_rel = {"before": 0, "at": 0, "after": 0}
    inc_rel_N46 = {"before": 0, "at": 0, "after": 0}
    min_offset = 0
    for s in steps:
        key = (s["N"], s["r_label"], s["family"], s["nx"], s["ny"])
        P = fl(s["Pi_h"])
        if key in prev and P > prev[key]:
            n_inc += 1
            fs = first_stab[key]
            off = int(s["s"]) - fs if math.isfinite(fs) else -999
            where = "after" if off > 0 else ("at" if off == 0 else "before")
            inc_rel[where] += 1
            if int(s["N"]) >= 4:
                inc_rel_N46[where] += 1
                min_offset = min(min_offset, off)
            if P - prev[key] > fl(s["tau"]):
                inc_beyond += 1
        prev[key] = P
    chk["energy_increases_below_tau"] = inc_beyond == 0
    chk["energy_increases_all_after_stabilization_as_stated"] = inc_rel_N46["before"] + inc_rel_N46["at"] == 0
    # ---- K-1: tables.md is the function of the CSV files (Builder generator) and T8
    tmp = out / "_tables"
    tmp.mkdir(exist_ok=True)
    for p in BUILDER.iterdir():
        (tmp / p.name).write_bytes(p.read_bytes())
    study.make_tables(tmp)
    regen = (tmp / "tables.md").read_bytes().replace(b"\r\n", b"\n")
    committed_tables = (BUILDER / "tables.md").read_bytes().replace(b"\r\n", b"\n")
    chk["tables_are_function_of_csv"] = regen == committed_tables
    for p in tmp.iterdir():
        p.unlink()
    tmp.rmdir()
    t8 = {}
    for N in "23456":
        vals = [fl(c["max_identity_defect_over_rounding"]) for c in cells if c["N"] == N]
        t8[N] = max(v for v in vals if not math.isnan(v))
    t8_line = [ln for ln in committed_tables.decode("utf-8").splitlines() if ln.startswith("| 6 | 66 |")]
    chk["T8_N6_after_fix"] = bool(t8_line) and f"{t8['6']:.2e}" in t8_line[0]
    # ---- K-1: numbers of the report
    claims = []
    st4 = [s for s in steps if int(s["N"]) >= 4]
    _claim(claims, "max eta_N (5.1)", max(fl(s["eta_N"]) for s in steps), 4.9e-16, abs_half=0.05e-16)
    _claim(claims, "refinements (5.1)", sum(s["refined"] == "True" for s in steps), 0, abs_half=0)
    _claim(claims, "max cond M_k, N = 6 (5.1)", max(fl(s["cond_M_max"]) for s in steps if s["N"] == "6"), 2.0e5, abs_half=0.05e5)
    _claim(claims, "max kappa_rep (5.1: = 1)", max(fl(s["kappa_rep"]) for s in st4), 1.0, abs_half=1e-6)
    _claim(claims, "max QR field error (1)", max(fl(s["qr_field_error"]) for s in st4 if s["qr_field_error"]), 5.5e-15,
           abs_half=0.05e-15)
    for N, v, cnt in (("4", 2.8e-13, 392), ("5", 1.5e-12, 392), ("6", 6.3e-14, 320)):
        rows = [c for c in comp if c["N"] == N and c["E_impl_inf"]]
        _claim(claims, f"V5 max E_impl, N = {N} (5.5)", max(fl(c["E_impl_inf"]) for c in rows), v, abs_half=0.05 * 10 ** math.floor(math.log10(v)))
        _claim(claims, f"V5 comparisons, N = {N} (5.5)", len(rows), cnt, abs_half=0)
    _claim(claims, "V5 max |E_impl_Pi| (5.5)", max(abs(fl(c["E_impl_Pi"])) for c in comp if int(c["N"]) >= 4 and c["E_impl_Pi"]),
           4.9e-15, abs_half=0.05e-15)
    r15 = [c for c in stops if c["r_label"] == "1/5"]
    r13 = [c for c in stops if c["r_label"] == "1/3"]
    _claim(claims, "rho_N, N = 6, r = 1/5, n = 1280 (5.1)", fl([c for c in r15 if c["nx"] == "1280"][0]["min_rho_N"]), 1.7404e-12,
           abs_half=0.00005e-12)
    _claim(claims, "rho_N, N = 6, r = 1/5, n = 40 (5.1)", fl([c for c in r15 if c["nx"] == "40"][0]["min_rho_N"]), 1.452e-12,
           abs_half=0.0005e-12)
    _claim(claims, "rho/threshold, r = 1/5, max over grids (5.1: 0.016)", max(fl(c["min_rho_over_threshold"]) for c in r15), 0.016,
           abs_half=0.0005)
    _claim(claims, "set/threshold, r = 1/5, max over grids (5.1: 0.28)", max(fl(c["min_set_over_threshold"]) for c in r15), 0.28,
           abs_half=0.005)
    _claim(claims, "rho_N, N = 6, r = 1/3, n = 1280 (5.1: 7.4e-11)", fl([c for c in r13 if c["nx"] == "1280"][0]["min_rho_N"]),
           7.4e-11, abs_half=0.05e-11)
    _claim(claims, "rho/threshold, r = 1/3, n = 1280 (5.1: 0.67)", fl([c for c in r13 if c["nx"] == "1280"][0]["min_rho_over_threshold"]),
           0.67, abs_half=0.005)
    for N, lab, v, h in (("5", "1/5", 1.27, 0.005), ("6", "5", 1.47, 0.005), ("5", "5", 81, 0.5), ("6", "1/2", 5.2, 0.05)):
        m = min(fl(c["min_rho_over_threshold"]) for c in cells if (c["N"], c["r_label"]) == (N, lab))
        _claim(claims, f"min rho/threshold N = {N}, r = {lab} (5.1)", m, v, abs_half=h)
    argmin_s1 = True
    for c in cells:
        if int(c["N"]) < 4:
            continue
        rows = [s for s in steps if (s["N"], s["r_label"], s["family"], s["nx"], s["ny"]) ==
                (c["N"], c["r_label"], c["family"], c["nx"], c["ny"]) and s["rho_over_threshold"]]
        if rows and min(rows, key=lambda s: fl(s["rho_over_threshold"]))["s"] != "1":
            argmin_s1 = False
    chk["min_rho_margin_at_s1"] = argmin_s1
    _claim(claims, "rho_N(s=1), N = 6, r = 5, n = 1280 (5.1: 3.6e-9)",
           fl([s for s in steps if (s["N"], s["r_label"], s["family"], s["nx"], s["s"]) == ("6", "5", "I", "1280", "1")][0]["rho_N"]),
           3.6e-9, abs_half=0.05e-9)
    p4 = [fl(o["p_sample"]) for o in ords if int(o["N"]) >= 4]
    _claim(claims, "min p (5.4)", min(p4), 1.974, abs_half=0.0005)
    _claim(claims, "max p (5.4)", max(p4), 2.003, abs_half=0.0005)
    _claim(claims, "p for n >= 320, max |p - 2| (5.4)", max(abs(fl(o["p_sample"]) - 2) for o in ords if int(o["N"]) >= 4 and int(o["n"]) >= 320),
           0.0, abs_half=0.0005)
    c4 = [c for c in comp if int(c["N"]) >= 4]
    _claim(claims, "T7 RESOLVED (5.4)", sum(c["status_E_disc_s1_navier_m"] == "RESOLVED" for c in c4 if c["s"] == "1"), 198, abs_half=0)
    _claim(claims, "T7 budget max (5.4: 3.4e-12)", max(fl(c["navier_m_budget_rel"]) for c in c4 if c["s"] == "1"), 3.4e-12, abs_half=0.05e-12)
    _claim(claims, "recursion vs Navier-m max (5.4: 1.3e-13)", max(fl(c["cont_s1_minus_navier_m_rel"]) for c in c4 if c["s"] == "1"),
           1.3e-13, abs_half=0.05e-13)
    _claim(claims, "E_total RESOLVED (5.5)", sum(c["status_E_total"] == "RESOLVED" for c in c4), 1302, abs_half=0)
    _claim(claims, "Navier UNRESOLVED (6)", sum(c["status_d_total_c_navier"] == "UNRESOLVED" for c in c4), 863, abs_half=0)
    _claim(claims, "e_rel_h energy route UNRESOLVED (6)", sum(c["status_e_rel_h"] == "UNRESOLVED" for c in c4), 1092, abs_half=0)
    _claim(claims, "Levy budget max (5.5: 6.7e-13)", max(fl(c["levy_budget_rel"]) for c in c4), 6.7e-13, abs_half=0.05e-13)
    _claim(claims, "w_cont rounding model max (5.5: 2.2e-13)", max(fl(c["cont_round_rel"]) for c in c4), 2.2e-13, abs_half=0.05e-13)
    _claim(claims, "w_cont doubling max (5.5: 1.3e-13)", max(fl(c["cont_diag_rel"]) for c in c4), 1.3e-13, abs_half=0.05e-13)
    _claim(claims, "identity defect N = 4-6 (5.6: 2.0e-4)", max(t8[N] for N in "456"), 2.0e-4, abs_half=0.05e-4)
    _claim(claims, "identity defect N = 2-3 (5.6: 1.7e-5)", max(t8[N] for N in "23"), 1.7e-5, abs_half=0.05e-5)
    _claim(claims, "energy increases N = 4-6 (5.6)", sum(int(c["energy_increases"]) for c in cells if int(c["N"]) >= 4), 2483, abs_half=0)
    _claim(claims, "increases beyond tau (5.6)", sum(int(c["energy_increases_beyond_tau"]) for c in cells), 0, abs_half=0)
    _claim(claims, "Pythagoras max (5.6: 3.7e-4)", max(abs(fl(c["pythagoras_defect_over_budget"])) for c in comp), 3.7e-4, abs_half=0.05e-4)
    for N, v in (("4", 66), ("5", 66), ("6", 47)):
        _claim(claims, f"s = 2 E1_E2_HOLD, N = {N} (5.6)", sum(c["s2_status"] == "E1_E2_HOLD_PROVISIONAL" for c in cells if c["N"] == N), v,
               abs_half=0)
    un = [c for c in cells if c["N"] == "6" and c["s2_status"] == "UNRESOLVED"]
    chk["s2_unresolved_N6_1280_five_r"] = sorted(c["r_label"] for c in un) == sorted(["7/5", "3/2", "2", "3", "5"]) and all(
        c["nx"] == "1280" for c in un)
    for N, v, h in (("4", 3.2, 0.05), ("5", 1.08, 0.005), ("6", 0.44, 0.005)):
        rows = [c for c in cells if c["N"] == N and c["m_2"]]
        best = min(rows, key=lambda c: fl(c["m_2"]))
        _claim(claims, f"min m_2, N = {N} at r = 5, n = 1280 (5.6)", fl(best["m_2"]), v, abs_half=h)
        chk[f"min_m2_location_N{N}"] = (best["r_label"], best["nx"]) == ("5", "1280")
    first = [fl(c["first_stabilization_s"]) for c in cells if int(c["N"]) >= 4 and c["first_stabilization_s"]]
    _claim(claims, "first stabilization s min (5.2: 4)", min(first), 4, abs_half=0)
    _claim(claims, "first stabilization s max (5.2: 15)", max(first), 15, abs_half=0)
    ctl_max = max(fl(v) for c in ctl for k, v in c.items() if "max_rel" in k and v)
    _claim(claims, "control N = 2-3 max rel (5.7: 4.1e-7)", ctl_max, 4.1e-7, abs_half=0.05e-7)
    chk["control_equal_run_coupled_12"] = sum(c["equal_run_coupled_bitwise"] == "True" for c in ctl) == 12
    cfg_min = min(cells, key=lambda c: fl(c["tolE_over_10floor"]))
    _claim(claims, "min tol_E/(10 eps_floor) (4: 1.42)", fl(cfg_min["tolE_over_10floor"]), 1.42, abs_half=0.005)
    chk["config_margin_1_42_at_n1280_as_stated"] = cfg_min["nx"] == cfg_min["ny"] == "1280"
    claims.append({"claim": "grid of the minimal configuration margin", "recomputed": f"{cfg_min['family']} {cfg_min['nx']}x{cfg_min['ny']}",
                   "printed": "N = 6, n = 1280", "half_unit": "", "ok": cfg_min["nx"] == cfg_min["ny"] == "1280"})
    m1280 = {(c["N"], c["r_label"]): fl(c["tolE_over_10floor"]) for c in cells if c["family"] == "I" and c["nx"] == "1280"}
    claims.append({"claim": "tol_E/(10 eps_floor) at N = 6, n = 1280", "recomputed": m1280[("6", "1")], "printed": "",
                   "half_unit": "", "ok": True})
    _claim(claims, "Navier budget min (6: 8.2e-8)", min(fl(c["navier_budget_rel"]) for c in comp), 8.2e-8, abs_half=0.05e-8)
    _claim(claims, "Navier budget max (6: 6.4e-7)", max(fl(c["navier_budget_rel"]) for c in comp), 6.4e-7, abs_half=0.05e-7)
    f2 = [fl(c["E_disc_inf"]) for c in comp if c["family"] == "II"]
    _claim(claims, "family II E_disc min (5.4: 3.4e-6)", min(f2), 3.4e-6, abs_half=0.05e-6)
    _claim(claims, "family II E_disc max (5.4: 7.6e-6)", max(f2), 7.6e-6, abs_half=0.05e-6)
    c2 = [c for c in cont if int(c["N"]) >= 4 and c["s"] != "1"]
    _claim(claims, "continuous e_rel budget s >= 2 (6: 3.3e-5)", max(fl(c["D_strict_rel"]) + fl(c["D_model_rel"]) for c in c2), 3.3e-5,
           abs_half=0.05e-5)
    _claim(claims, "continuous e_rel doubling s >= 2 (6: 6.8e-7)", max(fl(c["D_diag_rel"]) for c in c2), 6.8e-7, abs_half=0.05e-7)
    # strict decrease of e_rel(cont) in N at every s (5.3)
    viol, tot = 0, 0
    for lab in R_TEN:
        for s in range(1, 41):
            vals = sorted((int(c["N"]), fl(c["e_rel"])) for c in cont if c["r_label"] == lab and c["s"] == str(s))
            for (n1, v1), (n2, v2) in zip(vals, vals[1:]):
                tot += 1
                viol += not v2 < v1
    chk["e_rel_strictly_decreasing_in_N"] = viol == 0 and tot == 1040
    # nested subspaces at s = 1 (5.3, T8): 144 pairs
    nested, nviol = 0, 0
    groups = {}
    for c in comp:
        if c["s"] == "1":
            groups.setdefault((c["r_label"], c["family"], c["nx"], c["ny"]), []).append((int(c["N"]), fl(c["Pi_h"])))
    for vals in groups.values():
        vals.sort()
        for (n1, p1), (n2, p2) in zip(vals, vals[1:]):
            nested += 1
            nviol += p2 > p1
    _claim(claims, "nested pairs at s = 1 (5.3: 144 of 144)", nested - nviol, 144, abs_half=0)
    # zeta classes (4)
    cl = {(c["N"], c["r_label"], c["family"], c["nx"]): (c["class_all"], fl(c["zeta_N_all"])) for c in cells}
    chk["zeta_N4_1280_AQ_all_r"] = all(cl[("4", lab, "I", "1280")][0] in ("A", "Q") for lab in R_TEN)
    chk["zeta_N5_1280"] = all((cl[("5", lab, "I", "1280")][0] == "G") == (lab in ("1/5", "5")) for lab in R_TEN)
    chk["zeta_N6_1280"] = all(cl[("6", lab, "I", "1280")][0] in ("A", "Q") for lab in ("1/2", "2/3", "1", "7/5", "3/2", "2", "3")) and \
        cl[("6", "5", "I", "1280")][0] == "G"
    chk["zeta_N6_familyII_G"] = {lab for lab in FAMILY_II if cl[("6", lab, "II", str(FAMILY_II[lab][0]))][0] == "G"} >= {"1/2", "2", "3", "5"}
    _claim(claims, "zeta N = 5, r = 5, n = 1280 (4: 0.158)", cl[("5", "5", "I", "1280")][1], 0.158, abs_half=0.0005)
    _claim(claims, "zeta N = 6, r = 5, n = 1280 (4: 0.250)", cl[("6", "5", "I", "1280")][1], 0.250, abs_half=0.0005)
    # report tables 5.2, 5.3, 5.5 (spot values; printed digits)
    def cp(N, lab, n, s):
        return [c for c in comp if (c["N"], c["r_label"], c["family"], c["nx"], c["s"]) == (N, lab, "I", n, s)][0]

    def half(v, k):  # half a unit of the last of k printed significant digits
        return 0.5 * 10.0 ** (math.floor(math.log10(abs(v))) - (k - 1))

    for N, s, w, e, k in (("4", "1", 4.062187431e-3, 1.25e-5, 3), ("4", "2", 4.062351328e-3, 2.28e-9, 3),
                          ("4", "40", 4.062349326e-3, 1.014e-9, 4), ("5", "40", 4.062351109e-3, 6.31e-11, 3),
                          ("6", "2", 4.062351537e-3, 4.05e-11, 3), ("6", "40", 4.062351439e-3, 5.15e-12, 3)):
        c = cp(N, "1", "640", s)
        _claim(claims, f"5.2 w(centre) N = {N}, s = {s}", fl(c["w_centre"]), w, abs_half=half(w, 10))
        _claim(claims, f"5.2 e_rel,h N = {N}, s = {s}", fl(c["e_rel_h_direct"]), e, abs_half=half(e, k))
    for lab, s, vals in (("1/5", "40", (3.32e-7, 6.42e-9, 2.49e-10, 1.47e-11, 1.16e-12)),
                         ("1", "40", (1.05e-6, 2.40e-8, 1.02e-9, 6.35e-11, 5.22e-12)),
                         ("5", "2", (3.46e-7, 8.05e-9, 4.55e-10, 4.50e-11, 6.51e-12))):
        for N, v in zip("23456", vals):
            c = [x for x in cont if (x["N"], x["r_label"], x["s"]) == (N, lab, s)][0]
            _claim(claims, f"5.3 e_rel(cont) r = {lab}, s = {s}, N = {N}", fl(c["e_rel"]), v, abs_half=half(v, 3))
    for lab, N, d, kd, m, t in (("1", "4", 6.9e-8, 2, 5.45e-7, 6.14e-7), ("1", "6", 6.9e-8, 2, 2.47e-8, 9.37e-8),
                                ("5", "6", 6.66e-7, 3, 1.78e-8, 6.55e-7), ("1/5", "5", 6.66e-7, 3, 8.01e-8, 6.53e-7)):
        c = cp(N, lab, "1280", "40")
        _claim(claims, f"5.5 E_disc r = {lab}, N = {N}", fl(c["E_disc_inf"]), d, abs_half=half(d, kd))
        _claim(claims, f"5.5 E_method r = {lab}, N = {N}", fl(c["E_method_inf"]), m, abs_half=half(m, 3))
        _claim(claims, f"5.5 E_total r = {lab}, N = {N}", fl(c["E_total_inf"]), t, abs_half=half(t, 3))
    chk["report_numbers"] = all(c["ok"] for c in claims if c["claim"] != "grid of the minimal configuration margin")
    result = {"checks": chk, "claims_failed": [c for c in claims if not c["ok"]], "n_claims": len(claims),
              "guard_rows_judged": judged, "energy_increases_printed_all_N": n_inc,
              "energy_increases_printed_by_position_all_N": inc_rel, "energy_increases_printed_by_position_N4_6": inc_rel_N46,
              "energy_increase_earliest_offset_N4_6": min_offset, "T8_max_identity_defect": t8,
              "env": env_info()}
    write_csv(out / "integrity_claims.csv", claims)
    (out / "integrity.json").write_text(json.dumps(result, indent=1, default=str) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "env"}, indent=1, default=str)[:6000])
    return result


# ======================================================================================== source check of the isolation
def isolation(out):
    """The research half-step against the production half-step: the function bodies differ only by the N_SOLVE guard;
    the study script patches nothing in kkvim."""
    import ast
    import inspect

    import kkvim.coupled as kc

    study = builder_study()
    prod = inspect.getsource(kc.solve_coupled_halfstep)
    res = inspect.getsource(study.research_halfstep)

    def body(src):
        tree = ast.parse(src)
        fn = tree.body[0]
        stmts = [s for s in fn.body if not (isinstance(s, ast.Expr) and isinstance(getattr(s, "value", None), ast.Constant))]
        return [ast.dump(s) for s in stmts]

    bp, br = body(prod), body(res)
    only_p = [s for s in bp if s not in br]
    only_r = [s for s in br if s not in bp]
    text = (ROOT / "scripts" / "r4_n4_n6_study.py").read_text(encoding="utf-8")
    tree = ast.parse(text)
    assigns = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id in ("kc", "cr"):
                    assigns.append(ast.unparse(t))
    patch_calls = [n for n in ("setattr(", "monkeypatch", "importlib.reload") if n in text]
    out_d = {"statements_only_in_production": len(only_p), "statements_only_in_research": len(only_r),
             "difference_is_the_N_guard": len(only_p) == 1 and len(only_r) == 1 and "N_SOLVE" in only_p[0]
             and "N_RESEARCH" in only_r[0], "attribute_assignments_to_kkvim": assigns, "patch_calls": patch_calls,
             "statements_compared": len(bp)}
    (out / "isolation.json").write_text(json.dumps(out_d, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(out_d, indent=1))
    return out_d


# ======================================================================================== main
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("mode", choices=["integrity", "isolation", "rho1", "nodal", "cont", "repro", "meta"])
    p.add_argument("--out", type=Path, default=OUT_DEFAULT)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    if args.mode == "integrity":
        integrity(args.out)
    elif args.mode == "isolation":
        isolation(args.out)
    elif args.mode == "rho1":
        rho1(args.out, args.workers)
    elif args.mode == "nodal":
        nodal(args.out, args.workers)
    elif args.mode == "cont":
        cont(args.out, args.workers)
    elif args.mode == "repro":
        repro(args.out, args.workers)
    elif args.mode == "meta":
        info = env_info()
        info["object"] = "research/r4-n4-n6 @ 6c7a80de00140fbb714e6cb0577bed232f3bc644"
        (args.out / f"meta_{info['python']}.json").write_text(json.dumps(info, indent=1) + "\n", encoding="utf-8")
        print(json.dumps(info, indent=1))
    print(f"{args.mode}: {time.perf_counter() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
