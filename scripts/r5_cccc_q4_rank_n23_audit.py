"""R5-V Q4, priority 3 (D-053.4, item 3): numerical rank of CCCC at N = 2, 3 - exact rank of the stored binary64 factors, numerical
diagnostics rho^_N, kappa_rep, cond(A) before/after QR, cond(S) of B-C (C-3+), control constructions of known rank and the
separability of the candidate form rho_N > K u kappa_rep (PL 4.5). QA research tool, not a qualification: K, c_d, c_rank, tau_set,
cond(S) and other thresholds are NOT assigned; no rank regularisation, no lowering of N, no working solver. No import of ``kkvim``; the
QA references r5_cccc_sequence, r5_cccc_bc, r5_cccc_halfstep, r5_cccc_energy, the research C-3+ assembly and the earlier Q4 tools are
used read-only (no file of them is changed).

Fixed BEFORE the campaign (this docstring, the constants and ``control_cases`` are the protocol).

Objects (strictly distinguished; never mixed in the report):
 (O1) the stored binary64 factors U^ (given set, fixed direction) and V^ (solved set) of a half-step, and the EXACT product
      W~ = U^ V^^T of their rational values ("field defined by the stored factors"); its exact rank is decided here;
 (O2) the binary64 matrix fl(U^ V^^T) formed in double (a different matrix: it is not of rank N in general);
 (O3) the exact solution V* of the discrete half-step system with the exact coefficients of the STORED given set U^ (Q1b system),
      field W* = U^ V*^T; its rank is decided only through an accepted error certificate (Q3 / PR #41 set bounds, read, not
      recomputed);
 (O4) the exact-arithmetic discrete KK-MVI sequence and the theoretical sequence: NOT decided here (the error of the stored given set
      with respect to them, accumulated over s, has no bound: R4 5.2.1 (6), UNRESOLVED).
 A computed SVD rank is never a proof of an exact rank; numpy.linalg.matrix_rank and SVD thresholds are not used for any exact claim.

Sample. All half-steps with N = 2, 3 of the 123 Q3 sequences (docs/results/r5_cccc_q3_prequalification/sequences_e1_e7_e9.csv):
82 sequences, 656 half-steps, both first directions, s = 1..8, all ratios of Q3, chains and pairs. The Q3 NPZ arrays are read-only;
the sha256 of every used file is checked against docs/results/r5_cccc_q4_gamma_audit/meta.json BEFORE any calculation (mismatch:
campaign stopped). Per half-step two fields (one CSV row each):
 - AC: the stored R-AC field of Q3 (given set and solved set before the controller's QR, as stored);
 - BC: the research B-C half-step with the C-3+ assembly on the SAME input state, paired design of PR #41 (input mapping
   ``bc_input`` of scripts/r5_cccc_q4_energy_bc_audit.py; prototype QR ``r5_cccc_bc.run_halfstep`` on that input; only C replaced by
   C-3+ (``c3plus``/``replaced_c`` of scripts/r5_cccc_q4_c_assembly_e4_audit.py); at s >= 2 the prototype's U~ must equal the Q3
   given set bitwise, at s = 1 U~ = QR of T_N). The link to PR #41 is checked by the bitwise equality of cond(S), max cond(M_k),
   cond(A) of the solved set and the orthonormality defect with docs/results/r5_cccc_q4_energy_bc/rows.csv. No new B-C sequences;
   gamma, energies and B-C certificates of PR #39-41 are not recomputed.

Exact rank of a stored factor M (m x N, binary64) - own implementation (EXACT):
 1. every value is a dyadic rational; M = M_int 2^E with ONE exponent E for the whole matrix (scalar scaling keeps the rank, rho_N and
    kappa_rep); M_int a Python integer matrix (no rounding);
 2. fraction-free (Bareiss) elimination with row pivoting, every division checked to be exact (remainder 0, else FAIL); rank r,
    pivot rows I and pivot columns J;
 3. certificate of rank N: the N x N minor M[I, :]; its determinant is recomputed independently by the Leibniz formula over Fraction of
    the stored doubles and must equal det(M_int[I, :]) 2^(N E) exactly and be != 0 (the witness rows I are written to the CSV; anyone
    can recompute the minor from the stored arrays);
 4. certificate of rank < N: an integer null vector z != 0 (from the exact RREF of the pivot rows) with M_int z = 0 verified exactly on
    ALL rows;
 5. second implementation (agreement required, else FAIL): r5_cccc_halfstep.exact_rank (Fraction Gauss-Jordan).
 Field rank: rank(U^ V^^T) = N iff both U^ and V^ have exact rank N (U^ injective, V^^T onto R^N); if one is dependent, rank < N.
 Second exact path for the field: rank(U V^T) = rank(G_U G_V), G = Gram matrices (exact integers) - must agree; the N x N minor
 W~[I_U, I_V] formed from the exact products must equal det(U^[I_U]) det(V^[I_V]) (Cauchy-Binet; checked exactly).
 Small matrices (controls): the factor rank is also checked by brute force over ALL N x N minors (Leibniz, Fraction).

Exact (certified) singular values of W~ (EXACT): the squares sigma_k^2(W~_w) are the eigenvalues of the N x N integer matrix
G_U G_V (times h_U h_V 2^(2E_U + 2E_V)); its characteristic polynomial is formed exactly (Faddeev-LeVerrier over Fraction); the roots
are isolated by a Sturm sequence of the square-free part and refined by exact bisection to relative width 2^-ROOT_BITS. Hence certified
enclosures of rho_N(W~) = sigma_N/sigma_1, kappa_rep(U^, V^) = ||U_w|| ||V_w|| / ||W~_w|| (lambda_max of G_U, G_V, G_U G_V), cond(A_U),
cond(A_V) (lambda_max/lambda_min of the Gram matrices). The proven inequality R4 5.4.1 item 1, rho_N/cond2(U_w) <= 1/cond2(V_w) <=
cond2(U_w) rho_N, is checked on the enclosures (a certain violation is a FAIL: contradiction with a proven statement). With it the
Q3 E-7 identity cond(A) before QR = 1/rho^_N^2 gets an exact frame for the stored arrays: 1/cond(A_U) <= cond(A_V) rho_N^2 <= cond(A_U).

Numerical diagnostics (OBSERVATION; no bitwise equality between different BLAS paths is required):
 - P1: replica of the Q3 controller (r5_cccc_sequence._rank_diagnostic and its transformation): numpy.linalg.qr of sqrt(h) U, sqrt(h) V,
   numpy SVD of R_U R_V^T -> rho^_N, kappa_rep, sigma_N(R_V D_c), cond(A) before QR = cond(R_V)^2; transformation V_n = Q_V/sqrt(h_s),
   G_n = U R_V^T, cond(A) after QR = eigvalsh(h_s V_n^T V_n). For AC every value must equal the Q3 CSV bitwise and V_n, G_n must equal
   the stored given set and companion of s + 1 bitwise (same computation, same platform: "where the systems really coincide");
   otherwise the row is UNRESOLVED (input identification);
 - P2: other LAPACK path: scipy.linalg.qr with column pivoting (geqp3) and scipy.linalg.svd(lapack_driver="gesvd");
 - P3 (pre-selected small grids: n_x <= 64 and n_y <= 64): numpy SVD of the explicitly formed field fl(U V^T) (orientation x, y):
   rho^_N, and sigma_{N+1}/sigma_1 (the rounding floor of O2);
 - errors against the certified rho_N(W~): relative error of P1, P2, P3; c_d_obs = |rho^ - rho(W~)|/(u kappa^_rep) (observed constant
   of the diagnostic part (d) of R4 5.2 item 6; relative to W~ the factor error delta_f is 0 by definition);
 - after QR: exact rank of V_n and G_n, exact rho_N and kappa_rep of the transformed pair, exact orthonormality defect
   max|h_s V_n^T V_n - I| and exact cond(A) after QR, exact relative change of rho_N by the transformation (part (iii) of VOK-9);
   for BC the transformation is the prototype's (``r5_cccc_bc.orthonormalise`` of the next half-step), hypothetical (no B-C sequence);
 - BC only: cond(S), max cond(M_k), cond(A) of the input before QR (prototype log), after QR, exact orthonormality defect of U~.
 - O3 (certificate route): sigma_N(W*_w) >= sigma_N(W~_w) - ||U_w||_2 sqrt(h_s) b, b the accepted set bound ||x^ - x*||_2 (Q3
   j_set_bound_2 for AC, PR #41 bc_set_bound_2_up for BC; Weyl, ||delta V_w||_2 <= ||delta V_w||_F); "rank(W*) = N CERTIFIED" iff the
   right-hand side is > 0 in exact arithmetic; eps_step = ||U_w|| sqrt(h_s) b / sigma_1(W~_w) (only the current half-step part of
   delta_f; accumulation over s is NOT covered).
 - small grids: exact rank of the binary64 matrix fl(U V^T) (O2) by the exact Bareiss elimination of its dyadic integer matrix;
   cross-check: the rank modulo the prime 2^61 - 1 (a certified lower bound) must not exceed it; the numbers of distinct rows and
   columns of fl(U V^T) are recorded (exact symmetry of the plate problem duplicates rows/columns).

Control constructions of known rank (fixed here, ``control_cases``; small factors m = 15, n = 11; deterministic integers; the
mathematical rank of every construction is computed exactly from its EXACT rational source (Fraction), independently of any SVD):
 F1 FULL_EXACT_UNIMOD(a)     integer U0, V0 of rank N, unimodular integer T_a (cond ~ a^2): U = U0 T_a, V = V0 T_a^-T stored EXACTLY;
 F2 DEFICIENT_EXACT_UNIMOD(a) as F1 with V0 of rank N - 1 (last column = combination of the others): stored exactly deficient;
 F3 FULL_TINY_SIGMA(k)       integer U0; V with last column = combination + 2^-k e_j at an entry where the combination is 0
                             (exactly representable): mathematically and stored exactly of rank N, sigma_N/sigma_1 ~ 2^-k;
 F4 DEFICIENT_ROUNDED_T(a)   rank N - 1 source with rational non-dyadic T_a (entries a/3): U = rd(U0 T_a), V = rd(V0 T_a^-T)
                             correctly rounded from the exact rationals: mathematically deficient, stored rank decided exactly;
 F5 FULL_ROUNDED_T(a)        as F4 with a rank-N source;
 F6 DEFICIENT_ROUNDED_COMB   v_N = rd((v_1 + 2 v_2)/3) (N = 3), v_2 = rd(v_1/3) (N = 2): mathematically deficient, rounded;
 F7 DEFICIENT_SIN2_SUM       real sin^2 factors (T_N doubles): v_N = fl(v_1 + v_2) (N = 3), fl(v_1/3) (N = 2) computed in double;
                             the mathematical construction (exact sum / quotient of the stored doubles) is deficient;
 F8 DEFICIENT_REPEAT_SIN2    real sin^2 factors with v_N = v_1 exactly: stored exactly deficient; fl(U V^T) is formed in double.
 Parameters: a = 2^0, 2^3, ..., 2^24; k = 10, 20, 30, 40, 50, 60, 80, 100. Per construction: exact source rank, exact stored ranks
 (both factors, W~, fl(U V^T) - all exact), certified rho_N(W~), rho^ by P1, P2, P3, kappa^_rep, and rho^ after QR of the stored
 factors (U = Q R, pair (Q, V R^T) in double; R4 5.2 item 8 claim: QR of stored factors does not remove the stored error).

Rank-guard form (PL 4.5; R4 5.2 items 6-9): t = rho^_N/(u kappa^_rep) with u = 2^-53 and rho^, kappa^ of P1, for the classes
"mathematically full", "mathematically deficient" (controls) and "computed fields" (AC, BC). Reported: per class min/max of t, the
overlap of classes, and whether any value of K could separate them ON THESE DATA; K is NOT chosen, the R4 threshold is not carried over,
S-1/S-2 are not assigned. A separation on data is an OBSERVATION, not a qualification: the strict constant of the QR/SVD diagnostic
error, the factor error delta_f with its accumulation and the validity of the model for the computational path are listed as missing.

Arithmetic correctness criteria (fixed here; any failure is FAIL and the result would be BLOCKED): own exact rank = second
implementation; witness determinant equal by two exact paths and != 0; null vector verified on all rows; field rank by factors = rank
of G_U G_V; minor of W~ = product of the factor minors; every root enclosure verified by exact sign checks with relative width <=
2^-ROOT_BITS; R4 5.4.1 item 1 not certainly violated; for controls: the designed mathematical rank equals the exact source rank, and the
brute-force factor rank equals the own rank. Input identification (UNRESOLVED if violated): NPZ sha256, Q3 replica bitwise (AC), Q3
state reproduced bitwise by the prototype (BC, s >= 2), PR #41 values bitwise (BC).

Usage (repository root):
    uv run python scripts/r5_cccc_q4_rank_n23_audit.py run --profile full [--workers 12]
    uv run python scripts/r5_cccc_q4_rank_n23_audit.py run --profile ci --out scratch/r5_cccc_q4_rank_n23_ci
"""

import argparse
import csv
import hashlib
import importlib.util
import itertools
import json
import math
import multiprocessing
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from fractions import Fraction
from pathlib import Path

THREAD_VARS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
for _k in THREAD_VARS:
    os.environ[_k] = "1"  # as in Q3 and PR #41: one BLAS thread, the order of operations within a job is fixed

import numpy as np  # noqa: E402
import scipy  # noqa: E402
import scipy.linalg  # noqa: E402

from kkvim_reference import r5_cccc_bc as bc  # noqa: E402
from kkvim_reference import r5_cccc_energy as en  # noqa: E402
from kkvim_reference import r5_cccc_halfstep as hs  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "results" / "r5_cccc_q4_rank_n23"
GAMMA_AUDIT = ROOT / "scripts" / "r5_cccc_q4_gamma_audit.py"
ENERGY_BC = ROOT / "scripts" / "r5_cccc_q4_energy_bc_audit.py"
C_ASSEMBLY = ROOT / "scripts" / "r5_cccc_q4_c_assembly_e4_audit.py"
GA_META = ROOT / "docs" / "results" / "r5_cccc_q4_gamma_audit" / "meta.json"
PR41_ROWS = ROOT / "docs" / "results" / "r5_cccc_q4_energy_bc" / "rows.csv"
Q3_CSV = ROOT / "docs" / "results" / "r5_cccc_q3_prequalification" / "sequences_e1_e7_e9.csv"
Q3_ARRAYS_DEFAULT = Path(os.environ.get("KKVIM_Q3_ARRAYS", r"D:\git\KK-VIM-Q3-QA\scratch\r5_cccc_q3_prequalification\arrays"))
U_ROUND = Fraction(1, 2**53)  # unit roundoff of binary64
ROOT_BITS = 72  # relative width of the certified root enclosures
P_MOD = 2**61 - 1  # prime of the modular rank (certified lower bound)
SMALL_MAX_N = 64  # pre-selected small grids: n_x <= 64 and n_y <= 64
CONTROL_M, CONTROL_N = 15, 11
CONTROL_SEED = 20261028
A_PARAMS = tuple(2**j for j in range(0, 25, 3))
K_PARAMS = (10, 20, 30, 40, 50, 60, 80, 100)
UNRESOLVED = "UNRESOLVED"
RESEARCH = "RESEARCH_ONLY / PROVISIONAL / NOT_QUALIFIED (D-049.5)"
Q3_P1_KEYS = ("rho_N", "kappa_rep", "sigma_N_RV_Dc", "cond_A_before_QR", "cond_A_after_QR")
PR41_KEYS = ("cond_S", "max_cond_Mk", "cond_A_solved", "orthonormality_defect_up")


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ga():
    """Q4 gamma audit (Q3 reader, NPZ steps, CI data); read-only use."""
    if not hasattr(_ga, "mod"):
        _ga.mod = _load(GAMMA_AUDIT, "r5_cccc_q4_gamma_audit")
    return _ga.mod


def _eb():
    """PR #41 tool (input mapping of the paired B-C half-step); read-only use."""
    if not hasattr(_eb, "mod"):
        _eb.mod = _load(ENERGY_BC, "r5_cccc_q4_energy_bc_audit")
    return _eb.mod


def _ca():
    """C-3+ audit (c3plus, premises, replaced_c); read-only use."""
    if not hasattr(_ca, "mod"):
        _ca.mod = _load(C_ASSEMBLY, "r5_cccc_q4_c_assembly_e4_audit")
    return _ca.mod


# ---------------------------------------------------------------------------------------------------------------------------------
# exact integer linear algebra
# ---------------------------------------------------------------------------------------------------------------------------------

def dyadic_int(M) -> tuple[list[list[int]], int]:
    """M = M_int * 2**E exactly, one exponent E for the whole matrix (binary64 values are dyadic rationals)."""
    A = np.asarray(M, dtype=float)
    if A.ndim != 2 or not np.all(np.isfinite(A)):
        raise ValueError("finite 2-D array expected")
    rats = [[float(v).as_integer_ratio() for v in row] for row in A]
    k = max((d.bit_length() - 1 for row in rats for (_, d) in row), default=0)
    return [[n << (k - (d.bit_length() - 1)) for (n, d) in row] for row in rats], -k


def _exact_div(a: int, b: int) -> int:
    q, r = divmod(a, b)
    if r:
        raise ArithmeticError("Bareiss division not exact")  # would be an algorithm error (FAIL)
    return q


def bareiss(M: list[list[int]]) -> tuple[int, list[int], list[int]]:
    """Fraction-free elimination with row pivoting: (rank, pivot rows (original indices), pivot columns); exact divisions checked."""
    A = [list(r) for r in M]
    m = len(A)
    n = len(A[0]) if m else 0
    rows = list(range(m))
    rank, prev = 0, 1
    prow, pcol = [], []
    for c in range(n):
        p = next((i for i in range(rank, m) if A[i][c] != 0), None)
        if p is None:
            continue
        A[rank], A[p] = A[p], A[rank]
        rows[rank], rows[p] = rows[p], rows[rank]
        piv = A[rank][c]
        for i in range(rank + 1, m):
            aic = A[i][c]
            Ai, Ar = A[i], A[rank]
            for j in range(c + 1, n):
                Ai[j] = _exact_div(Ai[j] * piv - aic * Ar[j], prev)
            Ai[c] = 0
        prev = piv
        prow.append(rows[rank])
        pcol.append(c)
        rank += 1
        if rank == m:
            break
    return rank, prow, pcol


def det_leibniz(M) -> Fraction:
    """Determinant by the Leibniz formula over Fraction (independent of the integer path)."""
    n = len(M)
    total = Fraction(0)
    for perm in itertools.permutations(range(n)):
        inv = sum(1 for i in range(n) for j in range(i + 1, n) if perm[i] > perm[j])
        prod = Fraction(1)
        for i in range(n):
            prod *= Fraction(M[i][perm[i]])
        total += -prod if inv % 2 else prod
    return total


def det_int(M: list[list[int]]) -> int:
    """Determinant of a square integer matrix (Bareiss)."""
    n = len(M)
    A = [list(r) for r in M]
    sign, prev = 1, 1
    for k in range(n - 1):
        if A[k][k] == 0:
            sw = next((i for i in range(k + 1, n) if A[i][k] != 0), None)
            if sw is None:
                return 0
            A[k], A[sw] = A[sw], A[k]
            sign = -sign
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                A[i][j] = _exact_div(A[i][j] * A[k][k] - A[i][k] * A[k][j], prev)
        prev = A[k][k]
    return sign * A[n - 1][n - 1]


def null_vector(Mi: list[list[int]], prow: list[int], pcol: list[int]) -> list[int]:
    """Integer z != 0 with M z = 0 from the exact RREF of the pivot rows (rank < number of columns)."""
    ncol = len(Mi[0])
    free = next(c for c in range(ncol) if c not in pcol)
    R = [[Fraction(v) for v in Mi[r]] for r in prow]
    r = len(R)
    # Gauss-Jordan on the pivot rows in the pivot columns
    for k, c in enumerate(pcol):
        p = next(i for i in range(k, r) if R[i][c] != 0)
        R[k], R[p] = R[p], R[k]
        R[k] = [v / R[k][c] for v in R[k]]
        for i in range(r):
            if i != k and R[i][c] != 0:
                f = R[i][c]
                R[i] = [a - f * b for a, b in zip(R[i], R[k])]
    z = [Fraction(0)] * ncol
    z[free] = Fraction(1)
    for k, c in enumerate(pcol):
        z[c] = -R[k][free]
    den = math.lcm(*[v.denominator for v in z])
    return [int(v * den) for v in z]


def exact_rank_cert(M) -> dict:
    """Exact rank of a stored binary64 matrix with a checkable certificate (module docstring)."""
    Mi, E = dyadic_int(M)
    m, ncol = len(Mi), len(Mi[0])
    rank, prow, pcol = bareiss(Mi)
    out = {"rank": rank, "rows": prow, "cols": pcol, "E": E}
    if rank == ncol:
        sub = [[Mi[r][c] for c in range(ncol)] for r in prow]
        d_int = det_int(sub)
        d_frac = det_leibniz([[Fraction(float(M[r, c])) for c in range(ncol)] for r in prow])
        out["det"] = d_frac
        out["det_two_paths_equal"] = d_frac == Fraction(d_int) * Fraction(2) ** (ncol * E)
        out["det_nonzero"] = d_frac != 0
        out["certificate"] = f"MINOR rows={prow}"
        out["ok"] = out["det_two_paths_equal"] and out["det_nonzero"]
    else:
        z = null_vector(Mi, prow, pcol)
        ok = any(z) and all(sum(a * b for a, b in zip(row, z)) == 0 for row in Mi)
        out["null"] = z
        out["certificate"] = f"NULL z={z}"
        out["ok"] = ok
    out["m"], out["n"] = m, ncol
    return out


def rank_brute_force(M) -> int:
    """Rank of a small m x N matrix by ALL k x k minors (Leibniz over Fraction); independent control."""
    F = [[Fraction(float(v)) for v in row] for row in np.asarray(M)]
    m, n = len(F), len(F[0])
    for k in range(min(m, n), 0, -1):
        for rs in itertools.combinations(range(m), k):
            for cs in itertools.combinations(range(n), k):
                if det_leibniz([[F[r][c] for c in cs] for r in rs]) != 0:
                    return k
    return 0


def rank_mod_p(Mi: list[list[int]], p: int = P_MOD) -> tuple[int, list[int], list[int]]:
    """Rank modulo the prime p (certified lower bound of the exact rank) with pivot rows/columns."""
    A = [[v % p for v in row] for row in Mi]
    m = len(A)
    n = len(A[0]) if m else 0
    rows = list(range(m))
    rank = 0
    prow, pcol = [], []
    for c in range(n):
        piv = next((i for i in range(rank, m) if A[i][c]), None)
        if piv is None:
            continue
        A[rank], A[piv] = A[piv], A[rank]
        rows[rank], rows[piv] = rows[piv], rows[rank]
        inv = pow(A[rank][c], p - 2, p)
        Ar = [(v * inv) % p for v in A[rank]]
        A[rank] = Ar
        for i in range(rank + 1, m):
            f = A[i][c]
            if f:
                Ai = A[i]
                for j in range(c, n):
                    Ai[j] = (Ai[j] - f * Ar[j]) % p
        prow.append(rows[rank])
        pcol.append(c)
        rank += 1
        if rank == m:
            break
    return rank, prow, pcol


def gram_int(Mi: list[list[int]]) -> list[list[int]]:
    cols = list(zip(*Mi))
    n = len(cols)
    G = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            G[i][j] = G[j][i] = sum(a * b for a, b in zip(cols[i], cols[j]))
    return G


def matmul_int(A, B):
    return [[sum(a * b for a, b in zip(r, c)) for c in zip(*B)] for r in A]


# ---------------------------------------------------------------------------------------------------------------------------------
# certified eigenvalues of small integer matrices with real nonnegative spectrum (Sturm sequence, exact bisection)
# ---------------------------------------------------------------------------------------------------------------------------------

def charpoly(M) -> list[Fraction]:
    """Characteristic polynomial det(lambda I - M), coefficients high -> low (Faddeev-LeVerrier over Fraction)."""
    n = len(M)
    A = [[Fraction(v) for v in r] for r in M]
    c = [Fraction(1)]
    Mk = [[Fraction(0)] * n for _ in range(n)]
    for k in range(1, n + 1):
        AM = [[sum(A[i][t] * Mk[t][j] for t in range(n)) for j in range(n)] for i in range(n)]
        Mk = [[AM[i][j] + (c[-1] if i == j else 0) for j in range(n)] for i in range(n)]
        AMk = [[sum(A[i][t] * Mk[t][j] for t in range(n)) for j in range(n)] for i in range(n)]
        c.append(-sum(AMk[i][i] for i in range(n)) / k)
    return c


def _ptrim(p):
    i = 0
    while i < len(p) - 1 and p[i] == 0:
        i += 1
    return p[i:]


def _prem(a, b):
    a = list(a)
    while len(a) >= len(b) and any(a):
        f = a[0] / b[0]
        for i in range(len(b)):
            a[i] -= f * b[i]
        a = a[1:]
    return _ptrim(a) if a else [Fraction(0)]


def _pdiv(a, b):
    a = list(a)
    q = []
    while len(a) >= len(b):
        f = a[0] / b[0]
        q.append(f)
        for i in range(len(b)):
            a[i] -= f * b[i]
        a = a[1:]
    if any(a):
        raise ArithmeticError("polynomial division not exact")
    return q


def _pderiv(p):
    n = len(p) - 1
    return [c * (n - i) for i, c in enumerate(p[:-1])] or [Fraction(0)]


def _peval(p, x):
    v = Fraction(0)
    for c in p:
        v = v * x + c
    return v


def _sturm(p):
    seq = [p, _pderiv(p)]
    while len(seq[-1]) > 1 or seq[-1][0] != 0:
        r = _prem(seq[-2], seq[-1])
        if not any(r):
            break
        seq.append([-v for v in r])
        if len(r) == 1:
            break
    return seq


def _changes(seq, x):
    signs = [s for s in (_peval(q, x) for q in seq) if s != 0]
    return sum(1 for a, b in itertools.pairwise(signs) if (a > 0) != (b > 0))


def roots_certified(M, bits: int = ROOT_BITS) -> dict:
    """Distinct real roots of det(lambda I - M) for a small integer matrix with real nonnegative spectrum: exact zero-root flag and
    the positive roots in ascending order as verified enclosures [lo, hi] (exact rationals) with (hi - lo) <= lo 2^-bits."""
    p = charpoly(M)
    g = _sturm(p)[-1]
    sf = _pdiv(p, g) if len(g) > 1 else list(p)  # square-free part
    sf = [c / sf[0] for c in sf]
    zero = sf[-1] == 0
    while sf[-1] == 0 and len(sf) > 1:
        sf = sf[:-1]
    if len(sf) == 1:
        return {"zero_root": zero, "roots": [], "ok": True}
    seq = _sturm(sf)
    B = Fraction(1)
    bound = 1 + max(abs(c) for c in sf[1:])
    while B < bound:
        B *= 2
    total = _changes(seq, Fraction(0)) - _changes(seq, B)
    # isolation: (a, b] intervals whose end points are never roots (0 and B are not; a split point that is a root is moved)
    stack, iso = [(Fraction(0), B)], []
    while stack:
        a, b = stack.pop()
        cnt = _changes(seq, a) - _changes(seq, b)
        if cnt == 0:
            continue
        if cnt == 1:
            iso.append((a, b))
            continue
        mid = (a + b) / 2
        while _peval(sf, mid) == 0:
            mid = (a + mid) / 2
        stack += [(a, mid), (mid, b)]
    roots, ok = [], True
    for a, b in sorted(iso):
        sa = _peval(sf, a) > 0
        while a == 0 or (b - a) > a / 2**bits:
            mid = (a + b) / 2
            v = _peval(sf, mid)
            if v == 0:
                a = b = mid
                break
            if (v > 0) == sa:
                a = mid
            else:
                b = mid
        if a != b:
            fa, fb = _peval(sf, a), _peval(sf, b)
            ok &= (fa > 0) != (fb > 0) and (b - a) <= a / 2**bits
        roots.append((a, b))
    # the roots of the matrix (with the zero root removed) must be all counted: the spectrum is real and nonnegative
    ok &= len(roots) == total
    return {"zero_root": zero, "roots": roots, "ok": ok}


def sqrt_bounds(x: Fraction, bits: int = 90) -> tuple[Fraction, Fraction]:
    if x <= 0:
        return Fraction(0), Fraction(0)
    n, d = x.numerator, x.denominator
    k = max(0, bits - (n * d).bit_length() // 2 + 1)
    r = math.isqrt(n * d * 4**k)
    lo = Fraction(r, d * 2**k)
    hi = lo if r * r == n * d * 4**k else Fraction(r + 1, d * 2**k)
    return lo, hi


def _lam(res: dict, which: str) -> tuple[Fraction, Fraction]:
    """(lo, hi) of the smallest ('min', 0 if the zero root is present) or largest ('max') eigenvalue."""
    if which == "min":
        if res["zero_root"]:
            return Fraction(0), Fraction(0)
        return res["roots"][0]
    return res["roots"][-1] if res["roots"] else (Fraction(0), Fraction(0))


# ---------------------------------------------------------------------------------------------------------------------------------
# exact analysis of a pair of stored factors
# ---------------------------------------------------------------------------------------------------------------------------------

def field_exact(U, V, hU: Fraction, hV: Fraction) -> dict:
    """Exact rank of U, V and of W~ = U V^T with certificates, certified rho_N, kappa_rep, cond(A_U), cond(A_V), R4 5.4.1 item 1."""
    N = U.shape[1]
    out, fails = {}, []
    ru, rv = exact_rank_cert(U), exact_rank_cert(V)
    for tag, r, M in (("U", ru, U), ("V", rv, V)):
        out[f"{tag}_rank_exact"] = r["rank"]
        out[f"{tag}_rank_certificate"] = r["certificate"]
        if r["rank"] == N:
            d = r["det"]
            out[f"{tag}_minor_det_log2abs"] = math.log2(abs(d.numerator)) - math.log2(d.denominator)
            out[f"{tag}_minor_det_sign"] = 1 if d > 0 else -1
        r2 = hs.exact_rank(M)
        out[f"{tag}_rank_second_impl"] = r2
        if r2 != r["rank"]:
            fails.append(f"exact rank of {tag}: own {r['rank']} != second implementation {r2}")
        if not r["ok"]:
            fails.append(f"certificate of {tag} not verified")
    Ui, EU = dyadic_int(U)
    Vi, EV = dyadic_int(V)
    GU, GV = gram_int(Ui), gram_int(Vi)
    Mprod = matmul_int(GU, GV)
    rM = bareiss(Mprod)[0]
    by_factors = N if (ru["rank"] == N and rv["rank"] == N) else min(ru["rank"], rv["rank"])
    out["field_rank_exact"] = N if by_factors == N else rM
    out["field_rank_gram_path"] = rM
    if (by_factors == N) != (rM == N) or (by_factors < N and rM > by_factors):
        fails.append(f"field rank: factors give {by_factors}, Gram product gives {rM}")
    out["det_gram_product_positive"] = det_int(Mprod) > 0
    if out["det_gram_product_positive"] != (rM == N):
        fails.append("det(G_U G_V) sign inconsistent with the rank")
    if ru["rank"] == N and rv["rank"] == N:  # minor of W~ from the exact products = product of the factor minors
        Uf = [[Fraction(float(v)) for v in U[r]] for r in ru["rows"]]
        Vf = [[Fraction(float(v)) for v in V[r]] for r in rv["rows"]]
        Wm = [[sum((a * b for a, b in zip(x, y)), Fraction(0)) for y in Vf] for x in Uf]
        out["field_minor_equals_product"] = det_leibniz(Wm) == ru["det"] * rv["det"]
        out["field_rank_certificate"] = f"MINOR rows_U={ru['rows']} rows_V={rv['rows']}"
        if not out["field_minor_equals_product"]:
            fails.append("minor of W~ != product of the factor minors")
    else:
        out["field_rank_certificate"] = f"rank < N: {'U' if ru['rank'] < N else 'V'} {ru['certificate'] if ru['rank'] < N else rv['certificate']}"
    # certified spectra
    eM, eU, eV = roots_certified(Mprod), roots_certified(GU), roots_certified(GV)
    if not (eM["ok"] and eU["ok"] and eV["ok"]):
        fails.append("root enclosure not verified")
    lmin, lmax = _lam(eM, "min"), _lam(eM, "max")
    uU, uV = _lam(eU, "max"), _lam(eV, "max")
    nU, nV = _lam(eU, "min"), _lam(eV, "min")
    rho2 = (lmin[0] / lmax[1], lmin[1] / lmax[0]) if lmax[0] > 0 else (Fraction(0), Fraction(0))
    rho = (sqrt_bounds(rho2[0])[0], sqrt_bounds(rho2[1])[1])
    kap2 = (uU[0] * uV[0] / lmax[1], uU[1] * uV[1] / lmax[0]) if lmax[0] > 0 else None
    kap = (sqrt_bounds(kap2[0])[0], sqrt_bounds(kap2[1])[1]) if kap2 else None
    cU = (uU[0] / nU[1], uU[1] / nU[0]) if nU[0] > 0 else None
    cV = (uV[0] / nV[1], uV[1] / nV[0]) if nV[0] > 0 else None
    out.update({"rho_exact_down": en.down(rho[0]), "rho_exact_up": en.up(rho[1]),
                "kappa_exact_down": en.down(kap[0]) if kap else UNRESOLVED, "kappa_exact_up": en.up(kap[1]) if kap else UNRESOLVED,
                "condA_U_exact_up": en.up(cU[1]) if cU else math.inf, "condA_V_exact_up": en.up(cV[1]) if cV else math.inf,
                "condA_V_exact_down": en.down(cV[0]) if cV else math.inf,
                "condA_U_minus1_exact_up": en.up(cU[1] - 1) if cU else math.inf,
                "rho_rel_width": float((rho[1] - rho[0]) / rho[0]) if rho[0] > 0 else (0.0 if rho[1] == 0 else math.inf)})
    if rho[0] > 0 and (rho[1] - rho[0]) > rho[0] / 2**60:
        fails.append("rho enclosure too wide")
    # R4 5.4.1 item 1 (squared): rho^2/cond(A_U) <= 1/cond(A_V) <= cond(A_U) rho^2, i.e. X := cond(A_V) rho^2 in
    # [1/cond(A_U), cond(A_U)] (exact frame of the E-7 identity). Certainly violated -> FAIL; undecided by the enclosures -> recorded.
    if cU and cV and rho2[0] > 0:
        lo_id, hi_id = cV[0] * rho2[0], cV[1] * rho2[1]
        out["E7_exact_condAV_rho2_down"], out["E7_exact_condAV_rho2_up"] = en.down(lo_id), en.up(hi_id)
        if lo_id > cU[1] or hi_id < 1 / cU[1]:
            out["R4_541_inequality"] = "VIOLATED"
            fails.append("R4 5.4.1 item 1 violated (contradiction with a proven inequality)")
        elif hi_id <= cU[0] and lo_id >= 1 / cU[0]:
            out["R4_541_inequality"] = "HOLDS"
        else:
            out["R4_541_inequality"] = "UNDECIDED (enclosures overlap the bound)"
        out["E7_exact_frame_holds"] = out["R4_541_inequality"] == "HOLDS"
    else:
        out["R4_541_inequality"] = "NOT_APPLICABLE (rank < N)"
    # weighted extreme singular values (for the certificate route O3)
    S = Fraction(2) ** (2 * EU + 2 * EV) * hU * hV
    out["_sN_w"] = (sqrt_bounds(lmin[0] * S)[0], sqrt_bounds(lmin[1] * S)[1])
    out["_s1_w"] = (sqrt_bounds(lmax[0] * S)[0], sqrt_bounds(lmax[1] * S)[1])
    out["_Uw_norm_up"] = sqrt_bounds(uU[1] * Fraction(2) ** (2 * EU) * hU)[1]
    out["_rho"] = rho
    out["_kappa"] = kap
    out["_fails"] = fails
    return out


def orth_defect_exact(Q, h: Fraction) -> Fraction:
    """max |h Q^T Q - I| exactly."""
    Qi, E = dyadic_int(Q)
    G = gram_int(Qi)
    s = h * Fraction(2) ** (2 * E)
    n = len(G)
    return max(abs(s * G[i][j] - (1 if i == j else 0)) for i in range(n) for j in range(n))


# ---------------------------------------------------------------------------------------------------------------------------------
# numerical diagnostics (double)
# ---------------------------------------------------------------------------------------------------------------------------------

def diag_p1(G, hg: float, V, hv: float) -> dict:
    """Replica of r5_cccc_sequence._rank_diagnostic and of the controller's transformation (same numpy calls, same order)."""
    _, RG = np.linalg.qr(math.sqrt(hg) * G)
    QV, RV = np.linalg.qr(math.sqrt(hv) * V)
    sig = np.linalg.svd(RG @ RV.T, compute_uv=False)
    sv = np.linalg.svd(RV, compute_uv=False)
    Dc = np.diag(1.0 / np.linalg.norm(RV, axis=0))
    out = {"sigma": [float(x) for x in sig], "rho_N": float(sig[-1] / sig[0]) if sig[0] > 0 else 0.0,
           "kappa_rep": float(np.linalg.norm(RG, 2) * np.linalg.norm(RV, 2) / sig[0]) if sig[0] > 0 else math.inf,
           "sigma_N_RV_Dc": float(np.linalg.svd(RV @ Dc, compute_uv=False)[-1]),
           "cond_A_before_QR": float((sv[0] / sv[-1]) ** 2) if sv[-1] > 0 else math.inf}
    Vn = QV / math.sqrt(hv)
    Gn = G @ RV.T
    eA = np.linalg.eigvalsh(hv * (Vn.T @ Vn))
    out["cond_A_after_QR"] = float(eA[-1] / eA[0])
    return out, Vn, Gn


def diag_p2(G, hg: float, V, hv: float) -> dict:
    """Other LAPACK path: column-pivoted QR (geqp3) and gesvd."""
    _, RG, pg = scipy.linalg.qr(math.sqrt(hg) * G, mode="economic", pivoting=True)
    _, RV, pv = scipy.linalg.qr(math.sqrt(hv) * V, mode="economic", pivoting=True)
    RG, RV = RG[:, np.argsort(pg)], RV[:, np.argsort(pv)]
    sig = scipy.linalg.svd(RG @ RV.T, compute_uv=False, lapack_driver="gesvd")
    nG = scipy.linalg.svd(RG, compute_uv=False, lapack_driver="gesvd")[0]
    nV = scipy.linalg.svd(RV, compute_uv=False, lapack_driver="gesvd")[0]
    return {"rho_N": float(sig[-1] / sig[0]) if sig[0] > 0 else 0.0,
            "kappa_rep": float(nG * nV / sig[0]) if sig[0] > 0 else math.inf}


def diag_full(W) -> dict:
    """SVD of the explicitly formed double field (ratios are invariant under the constant weights)."""
    s = np.linalg.svd(np.asarray(W, dtype=float), compute_uv=False)
    return s


def rel_err_up(x: float, iv: tuple[Fraction, Fraction]) -> float:
    """max over the enclosure of |x - rho| / rho (rounded upwards); inf if rho = 0 and x != 0."""
    lo, hi = iv
    xf = Fraction(x)
    if lo == 0:
        return 0.0 if xf == 0 else math.inf
    return en.up(max(abs(xf - lo), abs(xf - hi)) / lo)


def abs_err_up(x: float, iv: tuple[Fraction, Fraction]) -> Fraction:
    xf = Fraction(x)
    return max(abs(xf - iv[0]), abs(xf - iv[1]))


# ---------------------------------------------------------------------------------------------------------------------------------
# one field (AC, BC or control)
# ---------------------------------------------------------------------------------------------------------------------------------

def analyse_field(U, V, hU: float, hV: float, hUq: Fraction, hVq: Fraction, X=None, Y=None, small: bool = False) -> dict:
    """Exact analysis, P1, P2 and (small) P3 of the field U V^T; hU, hV the float weights of the double paths (as the Q3
    controller), hUq, hVq the exact weights l/n; X, Y: the orientation used for fl(X Y^T) (default U, V)."""
    ex = field_exact(U, V, hUq, hVq)
    p1, Vn, Gn = diag_p1(U, hU, V, hV)
    p2 = diag_p2(U, hU, V, hV)
    out = {k: v for k, v in ex.items()}
    out.update({f"p1_{k}": v for k, v in p1.items() if k != "sigma"})
    out["p1_sigma"] = ";".join(repr(x) for x in p1["sigma"])
    out.update({"p2_rho_N": p2["rho_N"], "p2_kappa_rep": p2["kappa_rep"]})
    rho = ex["_rho"]
    out["p1_rho_relerr_up"] = rel_err_up(p1["rho_N"], rho)
    out["p2_rho_relerr_up"] = rel_err_up(p2["rho_N"], rho)
    out["p1_p2_rho_reldiff"] = abs(p1["rho_N"] - p2["rho_N"]) / p1["rho_N"] if p1["rho_N"] > 0 else UNRESOLVED
    kap = p1["kappa_rep"]
    if math.isfinite(kap) and kap > 0:
        out["t_ratio"] = float(Fraction(p1["rho_N"]) / (U_ROUND * Fraction(kap)))
        out["c_d_obs_p1"] = float(abs_err_up(p1["rho_N"], rho) / (U_ROUND * Fraction(kap)))
    if ex["_kappa"] is not None and math.isfinite(kap):
        out["p1_kappa_relerr_up"] = rel_err_up(kap, ex["_kappa"])
    if math.isfinite(p1["cond_A_before_QR"]) and isinstance(ex.get("condA_V_exact_up"), float) and math.isfinite(ex["condA_V_exact_up"]):
        cv = (Fraction(ex["condA_V_exact_down"]), Fraction(ex["condA_V_exact_up"]))
        out["p1_condA_before_relerr_up"] = rel_err_up(p1["cond_A_before_QR"], cv)
        if p1["rho_N"] > 0:
            out["p1_condA_over_inv_rho2"] = p1["cond_A_before_QR"] * p1["rho_N"] ** 2
    if small:
        Xf, Yf = (U, V) if X is None else (X, Y)
        Wd = Xf @ Yf.T
        s = diag_full(Wd)
        N = U.shape[1]
        out["p3_rho_N"] = float(s[N - 1] / s[0]) if s[0] > 0 else 0.0
        out["p3_rho_relerr_up"] = rel_err_up(out["p3_rho_N"], rho)
        out["p3_floor_sigma_N1_over_sigma1"] = float(s[N] / s[0]) if len(s) > N and s[0] > 0 else UNRESOLVED
        Wi, _ = dyadic_int(Wd)
        rk_exact = bareiss(Wi)[0]
        rk_p, _, _ = rank_mod_p(Wi)
        out["flW_rank_exact"] = rk_exact
        out["flW_rank_mod_p"] = rk_p
        if rk_p > rk_exact:  # rank mod p can never exceed the exact rank
            out["_fails"] = out.get("_fails", []) + ["rank of fl(U V^T) mod p > exact rank"]
        out["flW_shape"] = f"{Wd.shape[0]}x{Wd.shape[1]}"
        out["flW_distinct_rows"] = len({tuple(r) for r in Wd.tolist()})
        out["flW_distinct_cols"] = len({tuple(c) for c in Wd.T.tolist()})
    out["_Vn"], out["_Gn"] = Vn, Gn
    return out


def post_qr_exact(Gn, Vn, hgq: Fraction, hvq: Fraction, rho_pre) -> dict:
    """Exact rank and rho of the transformed pair (G_n, V_n), exact orthonormality defect and cond(A) after QR."""
    ex = field_exact(Gn, Vn, hgq, hvq)
    out = {"post_G_rank_exact": ex["U_rank_exact"], "post_V_rank_exact": ex["V_rank_exact"],
           "post_field_rank_exact": ex["field_rank_exact"], "post_rho_exact_down": ex["rho_exact_down"],
           "post_rho_exact_up": ex["rho_exact_up"], "post_kappa_exact_up": ex["kappa_exact_up"]}
    od = orth_defect_exact(Vn, hvq)
    out["post_orthonormality_defect_exact_up"] = en.up(od)
    Vi, _ = dyadic_int(Vn)
    eg = roots_certified(gram_int(Vi))
    lo, hi = _lam(eg, "min"), _lam(eg, "max")
    out["post_condA_after_exact_minus1_up"] = en.up(hi[1] / lo[0] - 1) if lo[0] > 0 else math.inf
    r0, r1 = ex["_rho"], rho_pre
    if r1[0] > 0:
        out["post_rho_change_rel_up"] = en.up(max(abs(r0[1] - r1[0]), abs(r0[0] - r1[1])) / r1[0])
    out["_fails"] = ex["_fails"]
    return out


def halfstep_certificate(ex: dict, hs_q: Fraction, b: float | None) -> dict:
    """Route O3: sigma_N(W*_w) >= sigma_N(W~_w) - ||U_w|| sqrt(h_s) b (Weyl); b = accepted set bound (read, not recomputed)."""
    if b is None or not math.isfinite(b):
        return {"halfstep_rank_certificate": "NOT_APPLICABLE (no accepted set bound)"}
    e = ex["_Uw_norm_up"] * sqrt_bounds(hs_q)[1] * Fraction(b)
    sN, s1 = ex["_sN_w"], ex["_s1_w"]
    out = {"set_bound_2": b, "eps_step_up": en.up(e / s1[0]) if s1[0] > 0 else math.inf}
    if sN[0] - e > 0:
        out["halfstep_rank_certificate"] = "CERTIFIED: rank(W*) = N (exact half-step solution on the stored given set)"
        out["rho_halfstep_exact_down"] = en.down((sN[0] - e) / (s1[1] + e))
        out["rho_over_eps_step_down"] = en.down(ex["_rho"][0] * s1[0] / e) if e > 0 else math.inf
    else:
        out["halfstep_rank_certificate"] = "NOT_CERTIFIED (bound does not separate sigma_N from 0)"
    return out


# ---------------------------------------------------------------------------------------------------------------------------------
# one half-step: AC and BC fields
# ---------------------------------------------------------------------------------------------------------------------------------

def _h(spec, direction):
    """(h of the fixed direction, h of the solved direction) as the Q3 controller computes them (float divisions)."""
    hx, hy = spec["lx"] / spec["nx"], spec["ly"] / spec["ny"]
    return (hx, hy) if direction == "Y" else (hy, hx)


def _hq(spec, direction):
    """The same weights exactly: l/n with l the stored double (as the Q1b certificates and r5_cccc_bc.orthonormality_defect)."""
    hx, hy = Fraction(spec["lx"]) / spec["nx"], Fraction(spec["ly"]) / spec["ny"]
    return (hx, hy) if direction == "Y" else (hy, hx)


def _base(spec, row, st, kind):
    N = st["given"].shape[1]
    small = spec["nx"] <= SMALL_MAX_N and spec["ny"] <= SMALL_MAX_N
    return {"id": _ga().row_id(row), "row_type": kind, "group": spec["group"], "grid": row["grid"], "N": N, "first": spec["first"],
            "s": st["s"], "direction": st["direction"], "n_f": int(row["j_n_f"]), "n_s": int(row["j_n_s"]), "small_grid": small,
            "marking": RESEARCH}


def _finish(out: dict, fails: list, unres: list) -> dict:
    fails = fails + out.pop("_fails", [])
    for k in [k for k in out if k.startswith("_")]:
        out.pop(k)
    if fails:
        out["status"] = "FAIL: " + "; ".join(fails)
    elif unres:
        out["status"] = f"{UNRESOLVED}: " + "; ".join(unres)
    else:
        out["status"] = "COMPLETE"
    return out


def analyse_ac(spec, steps, i, row) -> dict:
    st = steps[i]
    out = _base(spec, row, st, "AC")
    fails, unres = [], []
    hg, hv = _h(spec, st["direction"])
    hgq, hvq = _hq(spec, st["direction"])
    X, Y = st["now"]
    a = analyse_field(st["given"], st["V"], hg, hv, hgq, hvq, X, Y, out["small_grid"])
    fails += a.pop("_fails")
    Vn, Gn = a.pop("_Vn"), a.pop("_Gn")
    out.update(a)
    out["given_consistent"] = st["given_consistent"]
    if not st["given_consistent"]:
        unres.append("stored given set inconsistent with the previous half-step")
    # Q3 replica (bitwise)
    eq = {k: (a[f"p1_{k}"] == float(row[f"j_{k}"])) for k in Q3_P1_KEYS}
    out["q3_p1_bitwise"] = all(eq.values())
    if not out["q3_p1_bitwise"]:
        unres.append("P1 replica differs from Q3: " + ",".join(k for k, v in eq.items() if not v))
    if i + 1 < len(steps):
        nxt = steps[i + 1]
        out["q3_next_state_bitwise"] = bool(np.array_equal(Vn, nxt["given"]) and np.array_equal(Gn, nxt["companion"]))
        if not out["q3_next_state_bitwise"]:
            unres.append("controller transformation not reproduced bitwise")
    else:
        out["q3_next_state_bitwise"] = "NOT_APPLICABLE (s = s_max: no stored next state)"
    pq = post_qr_exact(Gn, Vn, hgq, hvq, a["_rho"])
    fails += pq.pop("_fails")
    out.update(pq)
    out["given_orthonormality_defect_exact_up"] = en.up(orth_defect_exact(st["given"], hgq))
    b = float(row["j_set_bound_2"]) if row.get("j_certificate", "").startswith("CERTIFIED") else None
    out["set_bound_source"] = "Q3 j_set_bound_2 (Q1b certificate, read)" if b is not None else "none"
    out.update(halfstep_certificate(a, hvq, b))
    return _finish(out, fails, unres)


def analyse_bc(spec, steps, i, row, pr41: dict | None) -> dict:
    eb, ca = _eb(), _ca()
    st = steps[i]
    out = _base(spec, row, st, "BC")
    fails, unres = [], []
    lx, nx, ly, ny = spec["lx"], spec["nx"], spec["ly"], spec["ny"]
    direction, N, s = st["direction"], st["given"].shape[1], st["s"]
    p_in, comp_in, kind = eb.bc_input(spec, steps, i)
    out["bc_input"] = kind
    rec = bc.run_halfstep(p_in, qr=True)  # prototype order NS 5.1: exact rank, QR, diagnostics (its own solve is not used)
    if rec.solved is None:
        out["status"] = f"{UNRESOLVED}: given set exactly dependent (prototype)"
        return out
    solved = rec.solved
    out["prototype_condA_input_before_QR"] = rec.log.get("cond_A_before_QR")
    out["prototype_condA_after_QR"] = rec.log.get("cond_A_after_QR")
    out["prototype_orthonormality_defect_up"] = rec.log.get("orthonormality_defect_up")
    out["prototype_sin_angle_span"] = rec.log.get("sin_angle_span_G_Utilde")
    if s >= 2:
        Cp = comp_in @ rec.R.T
        out["q3_state_reproduced_bitwise"] = bool(np.array_equal(solved.given, st["given"]) and np.array_equal(Cp, st["companion"]))
        if not out["q3_state_reproduced_bitwise"]:
            unres.append("input state of B-C does not reproduce the Q3 state")
    else:
        out["q3_state_reproduced_bitwise"] = "NOT_APPLICABLE (s = 1: U~ = QR of T_N)"
    # B-C with only C replaced by C-3+ (as PR #41)
    parts = ca.c3plus(solved)
    prem = ca.premises_c3plus(parts)
    out["c3plus_premises"] = "HOLD" if prem is None else f"NOT_APPLICABLE: {prem}"
    with ca.replaced_c(parts["C"]):
        A, B, C, F = bc.bc_coefficients(solved)
        sol = bc.solve_bc(solved)
    A0, B0, _, F0 = bc.bc_coefficients(solved)
    out["only_C_replaced"] = bool(np.array_equal(A, A0) and np.array_equal(B, B0) and np.array_equal(F, F0)
                                  and np.array_equal(C, parts["C"]))
    if not out["only_C_replaced"]:
        fails.append("only C replaced")
    out["bc_status"] = sol.status
    if sol.s is None:
        out["status"] = f"{UNRESOLVED}: B-C produced no solution ({sol.status})"
        return out
    out.update({"cond_S": sol.diagnostics.get("cond_S"), "max_cond_Mk": sol.diagnostics.get("max_cond_Mk"),
                "cond_A_solved": sol.diagnostics.get("cond_A")})
    if pr41 is not None:
        got = {"cond_S": out["cond_S"], "max_cond_Mk": out["max_cond_Mk"], "cond_A_solved": out["cond_A_solved"],
               "orthonormality_defect_up": out["prototype_orthonormality_defect_up"]}
        eq = {k: got[k] == float(pr41[k]) for k in PR41_KEYS}
        out["pr41_bitwise"] = all(eq.values())
        if not out["pr41_bitwise"]:
            unres.append("PR #41 values not reproduced: " + ",".join(k for k, v in eq.items() if not v))
        b = float(pr41["bc_set_bound_2_up"]) if pr41.get("bc_certificate", "").startswith("CERTIFIED") else None
        out["set_bound_source"] = "PR #41 bc_set_bound_2_up (Q1b certificate, read)" if b is not None else "none"
    else:
        out["pr41_bitwise"] = "NOT_APPLICABLE (no PR #41 row: CI profile)"
        b = None
        out["set_bound_source"] = "none (CI profile)"
    m = solved.n_s - 1
    Vbc = sol.s.reshape(N, m).T
    hg, hv = _h(spec, direction)
    hgq, hvq = _hq(spec, direction)
    X, Y = (solved.given, Vbc) if direction == "Y" else (Vbc, solved.given)
    a = analyse_field(solved.given, Vbc, hg, hv, hgq, hvq, X, Y, out["small_grid"])
    fails += a.pop("_fails")
    a.pop("_Vn")
    a.pop("_Gn")
    out.update(a)
    out["given_orthonormality_defect_exact_up"] = en.up(orth_defect_exact(solved.given, hgq))
    if out["given_orthonormality_defect_exact_up"] != out["prototype_orthonormality_defect_up"]:
        fails.append("orthonormality defect of U~: own exact != prototype exact")
    # hypothetical transformation for the next half-step by the prototype's QR
    nd = "X" if direction == "Y" else "Y"
    Un, Rn = bc.orthonormalise(hs.HalfStep(lx, nx, ly, ny, nd, Vbc))
    Gn = solved.given @ Rn.T
    pq = post_qr_exact(Gn, Un, hgq, hvq, a["_rho"])
    fails += pq.pop("_fails")
    out.update(pq)
    out["post_orthonormality_defect_prototype_up"] = bc.orthonormality_defect(hs.HalfStep(lx, nx, ly, ny, nd, Un))
    if out["post_orthonormality_defect_prototype_up"] != out["post_orthonormality_defect_exact_up"]:
        fails.append("orthonormality defect after QR: own exact != prototype exact")
    out.update(halfstep_certificate(a, hvq, b))
    return _finish(out, fails, unres)


def seq_job(spec: dict, rows: list[dict], npz_path: str, pr41_rows: dict) -> dict:
    ga = _ga()
    t0 = time.perf_counter()
    z = np.load(npz_path)
    steps = ga._steps_from_npz(z, spec)
    by_s = {int(r["j_s"]): r for r in rows}
    out = []
    for i, st in enumerate(steps):
        row = by_s[st["s"]]
        rid = ga.row_id(row)
        for kind in ("AC", "BC"):
            try:
                out.append(analyse_ac(spec, steps, i, row) if kind == "AC" else analyse_bc(spec, steps, i, row, pr41_rows.get(rid)))
            except Exception as exc:  # recorded, never hidden: the row is UNRESOLVED with the reason
                out.append({"id": rid, "row_type": kind, "group": spec["group"], "grid": row["grid"], "N": spec["N"],
                            "first": spec["first"], "s": st["s"], "direction": st["direction"],
                            "status": f"{UNRESOLVED}: exception {type(exc).__name__}: {exc}"})
    return {"rows": out, "spec": spec, "npz": Path(npz_path).name, "npz_sha256": sha256(Path(npz_path)), "steps": len(steps),
            "seconds": round(time.perf_counter() - t0, 2), "peak_mb": ga._q3().peak_mb()}


# ---------------------------------------------------------------------------------------------------------------------------------
# control constructions (fixed before the campaign)
# ---------------------------------------------------------------------------------------------------------------------------------

def _int_matrix(rng, m, n):
    while True:
        M = rng.integers(-9, 10, size=(m, n))
        if np.linalg.matrix_rank(M) == n:  # only a selection aid; the exact rank is verified below
            return [[Fraction(int(v)) for v in row] for row in M]


def _t_matrix(N: int, a: Fraction) -> list[list[Fraction]]:
    T = [[Fraction(1 if i == j else 0) for j in range(N)] for i in range(N)]
    for i in range(N - 1):
        T[i][i + 1] = a
    return T


def _inv_t(T):
    """Exact inverse of a unit upper triangular matrix."""
    N = len(T)
    X = [[Fraction(1 if i == j else 0) for j in range(N)] for i in range(N)]
    for i in reversed(range(N)):
        for j in range(N):
            X[i][j] -= sum(T[i][k] * X[k][j] for k in range(i + 1, N))
    return X


def _mm(A, B):
    return [[sum((a * b for a, b in zip(r, c)), Fraction(0)) for c in zip(*B)] for r in A]


def _tr(A):
    return [list(c) for c in zip(*A)]


def _deficient(V0: list[list[Fraction]], N: int) -> list[list[Fraction]]:
    """Last column replaced by an exact combination: v3 = v1 + v2 (N = 3), v2 = 2 v1 (N = 2)."""
    out = [list(r) for r in V0]
    for r in out:
        r[N - 1] = r[0] + r[1] if N == 3 else 2 * r[0]
    return out


def control_cases() -> list[dict]:
    """The fixed control constructions (module docstring). Each: exact source (U_src, V_src Fraction), stored doubles U, V, the
    designed mathematical rank and the storage kind."""
    rng = np.random.default_rng(CONTROL_SEED)
    m, n = CONTROL_M, CONTROL_N
    cases = []
    for N in (2, 3):
        U0, V0 = _int_matrix(rng, m, N), _int_matrix(rng, n, N)
        V0d = _deficient(V0, N)
        for a in A_PARAMS:
            T = _t_matrix(N, Fraction(a))
            Tit = _tr(_inv_t(T))
            for fam, Vsrc, rank in (("F1_FULL_EXACT_UNIMOD", V0, N), ("F2_DEFICIENT_EXACT_UNIMOD", V0d, N - 1)):
                Us, Vs = _mm(U0, T), _mm(Vsrc, Tit)
                cases.append({"family": fam, "N": N, "param": f"a=2^{int(math.log2(a))}", "U_src": Us, "V_src": Vs,
                              "math_rank": rank, "storage": "exact (integers)"})
            T3 = _t_matrix(N, Fraction(a, 3))
            T3it = _tr(_inv_t(T3))
            for fam, Vsrc, rank in (("F5_FULL_ROUNDED_T", V0, N), ("F4_DEFICIENT_ROUNDED_T", V0d, N - 1)):
                cases.append({"family": fam, "N": N, "param": f"a=2^{int(math.log2(a))}/3", "U_src": _mm(U0, T3),
                              "V_src": _mm(Vsrc, T3it), "math_rank": rank, "storage": "correctly rounded rationals"})
        # F3: tiny sigma_N, exactly representable perturbation at an entry where the combination vanishes
        Vz = [list(r) for r in V0]
        j = 0
        if N == 3:
            Vz[j][0], Vz[j][1] = Fraction(1), Fraction(-1)
        else:
            Vz[j][0] = Fraction(0)
        Vzd = _deficient(Vz, N)
        for k in K_PARAMS:
            Vk = [list(r) for r in Vzd]
            Vk[j][N - 1] += Fraction(1, 2**k)
            cases.append({"family": "F3_FULL_TINY_SIGMA", "N": N, "param": f"k={k}", "U_src": U0, "V_src": Vk, "math_rank": N,
                          "storage": "exact (dyadic)"})
        # F6: rounded rational combination
        V6 = [list(r) for r in V0]
        for r in V6:
            r[N - 1] = (r[0] + 2 * r[1]) / 3 if N == 3 else r[0] / 3
        cases.append({"family": "F6_DEFICIENT_ROUNDED_COMB", "N": N, "param": "(v1+2v2)/3" if N == 3 else "v1/3", "U_src": U0,
                      "V_src": V6, "math_rank": N - 1, "storage": "correctly rounded rationals"})
        # F7, F8: real sin^2 factors (T_N doubles)
        Ud = hs.start_set(m + 1, N)
        Yd = hs.start_set(n + 1, N)
        Y7 = Yd.copy()
        Y7[:, N - 1] = Yd[:, 0] + Yd[:, 1] if N == 3 else Yd[:, 0] / 3  # computed in double
        Y7src = [[Fraction(float(v)) for v in r] for r in Yd]
        for r in Y7src:
            r[N - 1] = r[0] + r[1] if N == 3 else r[0] / 3  # exact construction of the stored doubles
        cases.append({"family": "F7_DEFICIENT_SIN2_SUM", "N": N, "param": "fl(v1+v2)" if N == 3 else "fl(v1/3)",
                      "U_src": [[Fraction(float(v)) for v in r] for r in Ud], "V_src": Y7src, "math_rank": N - 1,
                      "storage": "computed in double", "U": Ud, "V": Y7})
        Y8 = Yd.copy()
        Y8[:, N - 1] = Yd[:, 0]
        cases.append({"family": "F8_DEFICIENT_REPEAT_SIN2", "N": N, "param": "v_N = v_1", "U_src": [[Fraction(float(v)) for v in r] for r in Ud],
                      "V_src": [[Fraction(float(v)) for v in r] for r in Y8], "math_rank": N - 1, "storage": "exact (stored doubles)",
                      "U": Ud, "V": Y8})
    for c in cases:
        if "U" not in c:
            c["U"] = np.array([[float(v) for v in r] for r in c["U_src"]])  # float(Fraction) is correctly rounded
            c["V"] = np.array([[float(v) for v in r] for r in c["V_src"]])
    return cases


def rank_fraction(M: list[list[Fraction]]) -> int:
    """Exact rank of a rational matrix (Fraction Gauss elimination); used for the exact SOURCE of the controls."""
    A = [list(r) for r in M]
    m, n = len(A), len(A[0])
    rank = 0
    for c in range(n):
        p = next((i for i in range(rank, m) if A[i][c] != 0), None)
        if p is None:
            continue
        A[rank], A[p] = A[p], A[rank]
        for i in range(rank + 1, m):
            if A[i][c] != 0:
                f = A[i][c] / A[rank][c]
                A[i] = [x - f * y for x, y in zip(A[i], A[rank])]
        rank += 1
    return rank


def analyse_control(c: dict) -> dict:
    N = c["N"]
    U, V = c["U"], c["V"]
    out = {"id": f"control|{c['family']}|N{N}|{c['param']}", "row_type": "CONTROL", "group": "control", "N": N,
           "family": c["family"], "param": c["param"], "storage": c["storage"], "small_grid": True}
    fails = []
    src_rank = rank_fraction(_mm(c["U_src"], _tr(c["V_src"])))
    out["math_rank_design"] = c["math_rank"]
    out["math_rank_exact_source"] = src_rank
    if src_rank != c["math_rank"]:
        fails.append(f"designed mathematical rank {c['math_rank']} != exact source rank {src_rank}")
    out["math_class"] = "FULL" if src_rank == N else "DEFICIENT"
    a = analyse_field(U, V, 1.0, 1.0, Fraction(1), Fraction(1), small=True)
    fails += a.pop("_fails")
    a.pop("_Vn")
    a.pop("_Gn")
    out.update(a)
    for tag, M in (("U", U), ("V", V)):
        bf = rank_brute_force(M)
        out[f"{tag}_rank_brute_force"] = bf
        if bf != out[f"{tag}_rank_exact"]:
            fails.append(f"brute-force rank of {tag} {bf} != own {out[f'{tag}_rank_exact']}")
    r_fl = out["flW_rank_exact"]
    out["flW_rank_second_impl"] = hs.exact_rank(U @ V.T)
    if out["flW_rank_second_impl"] != r_fl:
        fails.append("exact rank of fl(U V^T): own != second implementation")
    out["stored_class"] = "FULL" if out["field_rank_exact"] == N else "DEFICIENT"
    # QR of the stored factors (R4 5.2 item 8): U = Q R, pair (Q, V R^T) in double
    Q, R = np.linalg.qr(U)
    p1q, _, _ = diag_p1(Q, 1.0, V @ R.T, 1.0)
    out["p1_rho_after_QR_of_stored"] = p1q["rho_N"]
    out["p1_kappa_after_QR_of_stored"] = p1q["kappa_rep"]
    return _finish(out, fails, [])


def control_job(cases: list[dict]) -> list[dict]:
    res = []
    for c in cases:
        try:
            res.append(analyse_control(c))
        except Exception as exc:
            res.append({"id": f"control|{c['family']}|N{c['N']}|{c['param']}", "row_type": "CONTROL",
                        "status": f"{UNRESOLVED}: exception {type(exc).__name__}: {exc}"})
    return res


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
    if isinstance(v, Fraction):
        return repr(float(v))
    if isinstance(v, (float, np.floating)):
        return repr(float(v))
    if isinstance(v, (list, tuple)):
        return ";".join(str(x) for x in v)
    return "" if v is None else v


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys and not k.startswith("_")]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: fmt(r.get(k)) for k in keys})


def stats(values) -> dict:
    v = sorted(float(x) for x in values if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x)))
    if not v:
        return {"count": 0}
    q = lambda f: v[min(len(v) - 1, math.ceil(f * len(v)) - 1)]  # noqa: E731 - nearest rank (as Q3)
    return {"count": len(v), "min": v[0], "median": statistics.median(v), "q95": q(0.95), "max": v[-1]}


def _group(rows, key_fn):
    groups = {}
    for r in rows:
        groups.setdefault(key_fn(r), []).append(r)
    return groups


FIELD_STATS = ("p1_rho_N", "rho_exact_down", "p1_kappa_rep", "kappa_exact_up", "p1_cond_A_before_QR", "p1_cond_A_after_QR",
               "p1_sigma_N_RV_Dc", "condA_U_minus1_exact_up", "given_orthonormality_defect_exact_up",
               "post_orthonormality_defect_exact_up", "post_condA_after_exact_minus1_up", "post_rho_change_rel_up",
               "p1_rho_relerr_up", "p2_rho_relerr_up", "p3_rho_relerr_up", "p1_p2_rho_reldiff", "c_d_obs_p1", "p1_kappa_relerr_up",
               "p1_condA_before_relerr_up", "p1_condA_over_inv_rho2", "t_ratio", "p3_floor_sigma_N1_over_sigma1", "eps_step_up",
               "rho_over_eps_step_down", "rho_halfstep_exact_down", "cond_S", "max_cond_Mk", "cond_A_solved",
               "prototype_condA_input_before_QR", "prototype_orthonormality_defect_up")


def _block(sub: list[dict]) -> dict:
    blk = {"rows": len(sub)}
    for k in FIELD_STATS:
        st = stats(r.get(k) for r in sub)
        if st["count"]:
            blk[k] = st
    return blk


def _tally(rows, key):
    have = [r for r in rows if key in r]
    return f"{sum(1 for r in have if r.get(key) is True)}/{len(have)}"


def _log10(x):
    return math.log10(x) if isinstance(x, float) and x > 0 else None


def summarize(rows: list[dict], controls: list[dict]) -> dict:
    out = {"field_rows": len(rows), "control_rows": len(controls)}
    allr = rows + controls
    out["status_counts"] = {k: sum(1 for r in allr if r.get("status") == k) for k in sorted({str(r.get("status")) for r in allr})}
    out["failures"] = [{"id": r["id"], "row_type": r.get("row_type"), "status": r.get("status")} for r in allr
                       if r.get("status") != "COMPLETE"]
    done = [r for r in rows if r.get("status") == "COMPLETE"]
    for t in ("AC", "BC"):
        sub = [r for r in done if r["row_type"] == t]
        blk = {"rows_complete": len(sub)}
        N_of = lambda r: r["N"]  # noqa: E731
        blk["U_rank_exact_eq_N"] = sum(1 for r in sub if r["U_rank_exact"] == r["N"])
        blk["V_rank_exact_eq_N"] = sum(1 for r in sub if r["V_rank_exact"] == r["N"])
        blk["field_rank_exact_eq_N"] = sum(1 for r in sub if r["field_rank_exact"] == N_of(r))
        blk["field_rank_gram_path_eq_N"] = sum(1 for r in sub if r["field_rank_gram_path"] == N_of(r))
        blk["field_minor_equals_product"] = _tally(sub, "field_minor_equals_product")
        blk["post_field_rank_exact_eq_N"] = sum(1 for r in sub if r.get("post_field_rank_exact") == N_of(r))
        blk["halfstep_rank_certified"] = sum(1 for r in sub if str(r.get("halfstep_rank_certificate", "")).startswith("CERTIFIED"))
        blk["R4_541_inequality"] = {v: sum(1 for r in sub if r.get("R4_541_inequality") == v)
                                    for v in sorted({str(r.get("R4_541_inequality")) for r in sub})}
        blk["E7_exact_frame_holds"] = _tally(sub, "E7_exact_frame_holds")
        small = [r for r in sub if "flW_rank_exact" in r]
        blk["flW_small_rows"] = len(small)
        blk["flW_rank_exact_gt_N"] = sum(1 for r in small if r["flW_rank_exact"] > r["N"])
        blk["flW_rank_eq_min_distinct"] = sum(1 for r in small if r["flW_rank_exact"] == min(r["flW_distinct_rows"],
                                                                                             r["flW_distinct_cols"]))
        blk["flW_rank_exact"] = stats(r["flW_rank_exact"] for r in small)
        for k in ("q3_p1_bitwise", "q3_next_state_bitwise", "q3_state_reproduced_bitwise", "pr41_bitwise", "only_C_replaced",
                  "given_consistent"):
            if any(k in r for r in sub):
                blk[k] = _tally(sub, k)
        blk["all"] = _block(sub)
        blk["by_N"] = {str(k): _block(v) for k, v in sorted(_group(sub, N_of).items())}
        blk["by_s_N"] = {f"s{k[0]}_N{k[1]}": _block(v) for k, v in sorted(_group(sub, lambda r: (r["s"], r["N"])).items())}
        blk["by_direction_N"] = {f"{k[0]}_N{k[1]}": _block(v) for k, v in
                                 sorted(_group(sub, lambda r: (r["direction"], r["N"])).items())}
        blk["by_group_N"] = {f"{k[0]}_N{k[1]}": _block(v) for k, v in sorted(_group(sub, lambda r: (r["group"], r["N"])).items())}
        blk["by_grid_N"] = {f"{k[0]}_N{k[1]}": _block(v) for k, v in sorted(_group(sub, lambda r: (r["grid"], r["N"])).items())}
        blk["s1_vs_s_ge2_N"] = {f"{'s1' if k[0] else 's>=2'}_N{k[1]}": _block(v) for k, v in
                                sorted(_group(sub, lambda r: (r["s"] == 1, r["N"])).items())}
        out[t] = blk
    # paired comparison BC vs AC (same id)
    ac = {r["id"]: r for r in done if r["row_type"] == "AC"}
    pairs = [(ac[r["id"]], r) for r in done if r["row_type"] == "BC" and r["id"] in ac]
    out["pairs"] = len(pairs)
    out["BC_over_AC_rho_exact"] = {f"N{N}": stats(b["rho_exact_down"] / a["rho_exact_down"] for a, b in pairs
                                                  if a["N"] == N and a["rho_exact_down"] > 0)
                                   for N in (2, 3)}
    out["BC_over_AC_rho_exact_by_s"] = {f"s{s}_N{N}": stats(b["rho_exact_down"] / a["rho_exact_down"] for a, b in pairs
                                                            if a["N"] == N and a["s"] == s and a["rho_exact_down"] > 0)
                                        for s in range(1, 9) for N in (2, 3)}
    # controls
    cdone = [r for r in controls if r.get("status") == "COMPLETE"]
    out["controls"] = [{k: r.get(k) for k in ("id", "family", "N", "param", "storage", "math_rank_exact_source", "U_rank_exact",
                                               "V_rank_exact", "field_rank_exact", "flW_rank_exact", "rho_exact_down", "rho_exact_up",
                                               "p1_rho_N", "p2_rho_N", "p3_rho_N", "p1_kappa_rep", "kappa_exact_up", "t_ratio",
                                               "c_d_obs_p1", "p1_rho_after_QR_of_stored", "p1_kappa_after_QR_of_stored",
                                               "p3_floor_sigma_N1_over_sigma1", "math_class", "stored_class")} for r in cdone]
    # rank-guard form: t = rho^/(u kappa^)
    classes = {
        "control_math_FULL": [r for r in cdone if r["math_class"] == "FULL"],
        "control_math_DEFICIENT": [r for r in cdone if r["math_class"] == "DEFICIENT"],
        "control_math_DEFICIENT_stored_DEFICIENT": [r for r in cdone if r["math_class"] == "DEFICIENT" and r["stored_class"] == "DEFICIENT"],
        "control_math_DEFICIENT_stored_FULL": [r for r in cdone if r["math_class"] == "DEFICIENT" and r["stored_class"] == "FULL"],
        "control_F1_F5_full_moderate": [r for r in cdone if r["family"] in ("F1_FULL_EXACT_UNIMOD", "F5_FULL_ROUNDED_T")],
        "control_F3_full_tiny_sigma": [r for r in cdone if r["family"] == "F3_FULL_TINY_SIGMA"],
        "AC": [r for r in done if r["row_type"] == "AC"], "BC": [r for r in done if r["row_type"] == "BC"],
    }
    rg = {}
    for k, sub in classes.items():
        t = [r.get("t_ratio") for r in sub]
        rg[k] = {"t": stats(t), "c_d_obs_p1": stats(r.get("c_d_obs_p1") for r in sub)}
        for N in (2, 3):
            rg[k][f"t_N{N}"] = stats(r.get("t_ratio") for r in sub if r["N"] == N)
    real_min = min((r["t_ratio"] for r in classes["AC"] + classes["BC"] if isinstance(r.get("t_ratio"), float)), default=None)
    def_max = max((r["t_ratio"] for r in classes["control_math_DEFICIENT"] if isinstance(r.get("t_ratio"), float)), default=None)
    full_min = min((r["t_ratio"] for r in classes["control_math_FULL"] if isinstance(r.get("t_ratio"), float)), default=None)
    rg["separation"] = {
        "max_t_math_deficient_controls": def_max, "min_t_computed_fields": real_min, "min_t_math_full_controls": full_min,
        "gap_log10_computed_over_deficient": (_log10(real_min) - _log10(def_max)) if real_min and def_max else None,
        "math_full_controls_below_max_deficient": sum(1 for r in classes["control_math_FULL"]
                                                      if isinstance(r.get("t_ratio"), float) and def_max is not None
                                                      and r["t_ratio"] <= def_max),
        "note": "OBSERVATION on these data; K is not chosen (PL 4.5, D-053.5)"}
    out["rank_guard_form"] = rg
    return out


def ci_specs_rows(tmp: Path):
    ga = _ga()
    csv_path, arrays = ga.ci_data(tmp)
    rows = [r for r in ga.read_q3_rows(csv_path) if r["N"] in ("2", "3")]
    return rows, arrays, csv_path


def run_campaign(profile: str, out: Path, workers: int, arrays: Path) -> dict:
    t0 = time.time()
    head_at_start, dirty_at_start = git("rev-parse", "HEAD"), git("status", "--porcelain").splitlines()
    out.mkdir(parents=True, exist_ok=True)
    ga = _ga()
    q3 = ga._q3()
    tmp = None
    if profile == "ci":
        tmp = Path(tempfile.mkdtemp(prefix="r5q4rank_"))
        rows, arrays, csv_path = ci_specs_rows(tmp)
        pr41, npz_expected = {}, {}
        prov = {"csv": "generated by the Q3 seq_job (CI profile)", "sha256": sha256(csv_path)}
    else:
        rows = [r for r in ga.read_q3_rows(Q3_CSV) if r["N"] in ("2", "3")]
        meta_ga = json.loads(GA_META.read_text(encoding="utf-8"))
        npz_expected = {e["file"]: e["sha256"] for e in meta_ga["npz"]}
        with open(PR41_ROWS, encoding="utf-8") as f:
            pr41 = {r["id"]: r for r in csv.DictReader(f)}
        prov = {"q3_csv": ga.provenance(Q3_CSV), "gamma_audit_meta_sha256": sha256(GA_META), "pr41_rows_sha256": sha256(PR41_ROWS),
                "pr41_rows": len(pr41)}
    groups = {}
    for r in rows:
        groups.setdefault(ga.seq_key(r), []).append(r)
    jobs, npz_check = [], []
    for rs in groups.values():
        spec = ga.spec_of(rs[0]) if profile == "full" else ga._ci_spec(rs[0])
        npz = arrays / q3.seq_file(spec)
        if profile == "full":  # sha256 of every NPZ before any calculation
            got = sha256(npz)
            npz_check.append({"file": npz.name, "sha256": got, "equals_gamma_audit_meta": npz_expected.get(npz.name) == got})
            if npz_expected.get(npz.name) != got:
                raise SystemExit(f"NPZ {npz.name}: sha256 differs from the gamma audit meta - inputs not verified, campaign stopped")
        jobs.append((spec, rs, str(npz), {ga.row_id(r): pr41[ga.row_id(r)] for r in rs if ga.row_id(r) in pr41}))
    jobs.sort(key=lambda j: -(j[0]["nx"] * j[0]["ny"] * j[0]["N"]))  # largest first
    cases = control_cases()
    if workers > 1:
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(workers) as pool:
            ctrl_async = pool.apply_async(control_job, (cases,))
            results = pool.starmap(seq_job, jobs, chunksize=1)
            controls = ctrl_async.get()
    else:
        results = [seq_job(*j) for j in jobs]
        controls = control_job(cases)
    rows_out = [r for res in results for r in res["rows"]]
    rows_out.sort(key=lambda r: (r["group"], r["grid"], r["N"], r["first"], r["s"], r["row_type"]))
    write_csv(out / "rows.csv", rows_out + controls)
    summ = summarize(rows_out, controls)
    summ["sample_half_steps_expected"] = len(rows)
    summ["sequences"] = len(groups)
    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summ, f, indent=1, ensure_ascii=False, default=_json)
        f.write("\n")
    meta = {
        "study": "R5-V Q4 priority 3 (D-053.4): numerical rank of CCCC at N = 2, 3 - exact rank of stored factors, diagnostics, "
                 "controls, separability of rho_N > K u kappa_rep; QA research tool, paired design; K, c_d, cond(S) not assigned",
        "profile": profile, "git_commit_at_start": head_at_start, "git_status_porcelain_at_start": dirty_at_start,
        "git_commit_at_end": git("rev-parse", "HEAD"), "git_status_porcelain_at_end": git("status", "--porcelain").splitlines(),
        "python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__, "platform": platform.platform(),
        "machine": platform.machine(), "processor": platform.processor(),
        "threads": {k: os.environ.get(k) for k in THREAD_VARS}, "workers": workers, "u": float(U_ROUND), "root_bits": ROOT_BITS,
        "p_mod": P_MOD, "small_grid_rule": f"n_x <= {SMALL_MAX_N} and n_y <= {SMALL_MAX_N}",
        "control": {"m": CONTROL_M, "n": CONTROL_N, "seed": CONTROL_SEED, "a": list(A_PARAMS), "k": list(K_PARAMS), "cases": len(cases)},
        "data": prov, "q3_arrays_dir": str(arrays) if profile == "full" else "temporary (CI)", "q3_arrays_used_read_only": True,
        "npz_checked_before_calculation": npz_check,
        "npz": [{"file": r["npz"], "sha256": r["npz_sha256"]} for r in results],
        "jobs": [{"nx": r["spec"]["nx"], "ny": r["spec"]["ny"], "N": r["spec"]["N"], "first": r["spec"]["first"],
                  "seconds": r["seconds"], "peak_mb": r["peak_mb"]} for r in results],
        "resource_rule_PL_6_3": {"peak_job_mb": max((r["peak_mb"] or 0) for r in results),
                                 "limit_mb_50pct_ram": (q3.total_ram_mb() or 0) / 2,
                                 "longest_job_s": max(r["seconds"] for r in results)},
        "seconds_total": round(time.time() - t0, 1),
    }
    with open(out / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1, ensure_ascii=False, default=_json)
        f.write("\n")
    if tmp is not None:
        shutil.rmtree(tmp, ignore_errors=True)
    return {"summary": summ, "meta": meta}


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
    r.add_argument("--workers", type=int, default=12)
    r.add_argument("--arrays", type=Path, default=Q3_ARRAYS_DEFAULT)
    a = ap.parse_args(argv)
    out = a.out or (OUT if a.profile == "full" else ROOT / "scratch" / "r5_cccc_q4_rank_n23_ci")
    res = run_campaign(a.profile, out, a.workers if a.profile == "full" else 1, a.arrays)
    s = res["summary"]
    print(json.dumps({"field_rows": s["field_rows"], "control_rows": s["control_rows"], "status_counts": s["status_counts"]},
                     default=_json))
    return 0 if all(k == "COMPLETE" for k in s["status_counts"]) else 1


if __name__ == "__main__":
    sys.exit(main())
