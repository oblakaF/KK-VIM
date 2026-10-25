"""R5-V Q2 (QA): does operator assembly rounding and missing iterative refinement explain the external D2 values?

Independent QA control for the five external D2 values that the Q1 study did not reproduce within the B_ext candidate,
(r, n) = (1, 48), (3, 96), (5, 48), (5, 96), (5, 192), with dyadic controls n = 32, 64 (h = 1/n exact in binary) and the
other external points as a supplement. Plan docs/R5_CCCC_PREQUALIFICATION.plan.md, sections 1.4, 1.5 and 7 (Q2); external
values docs/R5_CCCC_REFERENCE_MATH.pre-review.md, section 9.8 (UNVERIFIED_EXTERNAL_NUMERICAL_EVIDENCE).

Independence: no import of ``kkvim`` and no import of ``kkvim_reference``. The 13-point stencil, the CCCC ghost-node rule,
the assemblies, the exact residual and the refinement are written here. The comparison with the stored kkvim_reference
results (d2_external.csv, column alpha_ref) is a separate step that only reads the CSV.

Problem (RM 9.3). Short side l = 1, long side r, q = D = 1, n intervals on the short side, h = 1/n on both axes (h_x = r/(rn)
is the same real number as 1/n). In every interior node (delta_x^4 + 2 delta_x^2 delta_y^2 + delta_y^4) w = q/D; w = 0 on
the boundary; clamped edge by the even ghost node w_-1 = w_1. With Z the integer 13-point operator (centre 20, the four
neighbours -8, the four diagonals 2, the four nodes at distance 2 equal 1, mirrored ghosts summed): A = Z / h^4 = n^4 Z, so
Z y = 1 gives w = y / n^4 and alpha = w_c D/(q l^4) = y_c / n^4 (exact scaling, l = 1).

Variants (fixed before the comparison):
    V1  F-assembly, no refinement: spsolve(A_F, 1), SuperLU, ordering COLAMD (the stated set-up of the consultant).
    V2  the same A_F, LU of A_F, refinement with the EXACT residual of A_F itself: converges to the exact solution of the
        rounded operator, so V2 - V4 is the effect of the assembly alone.
    V2X the same LU of A_F, refinement with the exact residual of n^4 Z (the operator is rounded only inside the LU).
    V3  integer Z, no refinement: spsolve(Z, 1), alpha = y_c / n^4.
    V4  integer Z, LU of Z, refinement with the exact residual of Z: alpha_ref QA.
Three F-assemblies with h = 1.0/n in double, all D4 (x) I + 2 L (x) L + I (x) D4 with L = tridiag(1, -2, 1)/h**2 unless
stated: F_KRON with D4 = L L + (2/h**4)(e_1 e_1^T + e_m e_m^T); F_D4H4 with D4 = pentadiag(1, -4, 6, -4, 1)/h**4 and the
boundary entries 7/h**4; F_DIV = Z / h**4 entry by entry. They are representative textbook assemblies, NOT the code of the
consultant (unavailable). At n = 3 * 2^k, 1/h**2 = n^2 exactly but 1/h**4 != n^4 in double, so the three differ there.

Equivalent diagonal shift (OBSERVATION, a one-parameter summary, not a model of the consultant): if the consultant's operator
were n^4 (Z + eps I), to first order alpha_ext - alpha_ref = -eps (Z^-2 1)_c / n^4; eps_equiv = -(alpha_ext - alpha_ref) n^4 /
(Z^-2 1)_c, reported in units of u = 2^-53. A systematic operator perturbation gives eps_equiv independent of r at fixed h;
independent solve errors do not.

Exact arithmetic: matrix entries and iterates are dyadic rationals; the residual b - A x is computed in Python integers with
a common power-of-two denominator and rounded once to double (int true division is correctly rounded). Refinement: up to
MAX_REFINE steps, stop when max|dx| no longer decreases or is zero (plan 1.4, item 2).

Usage:
    uv run python scripts/r5_cccc_q2_integer_repro.py              # target 5 + dyadic controls, ~1 min
    uv run python scripts/r5_cccc_q2_integer_repro.py --all        # + the other external points (n <= 256), ~5 min
Output (not committed, scratch/ is outside git): scratch/r5_cccc_q2_integer_repro/{runs.csv, meta.json}.
"""

import argparse
import csv
import json
import math
import platform
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

import numpy as np
import scipy
import scipy.sparse as sp
import scipy.sparse.linalg as spla

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "scratch" / "r5_cccc_q2_integer_repro"
EXTERNAL_CSV = ROOT / "docs" / "results" / "r5_cccc_references" / "d2_external.csv"

# RM 9.8, raw consultant values as printed (strings, unchanged)
EXTERNAL = {
    (1, 32): "1.2759787033576e-3", (1, 48): "1.2700711562290e-3", (1, 64): "1.2679950248813e-3",
    (1, 96): "1.2665093207759e-3", (1, 128): "1.2659887771808e-3", (1, 192): "1.2656167846025e-3",
    (1, 256): "1.2654865531591e-3",
    (2, 32): "2.5518770709064e-3", (2, 48): "2.5413696856572e-3", (2, 64): "2.5376894721877e-3",
    (2, 96): "2.5350599141738e-3", (2, 128): "2.5341394052183e-3", (2, 192): "2.5334818417406e-3",
    (2, 256): "2.5332516902792e-3",
    (3, 32): "2.6376772204295e-3", (3, 48): "2.6263173160792e-3", (3, 64): "2.6223412711801e-3",
    (3, 96): "2.6195012156523e-3", (3, 128): "2.6185071923402e-3", (3, 192): "2.6177971682155e-3",
    (3, 256): "2.6175486667662e-3",
    (5, 32): "2.6245775748442e-3", (5, 48): "2.6132718443996e-3", (5, 64): "2.6093148563303e-3",
    (5, 96): "2.6064884413949e-3", (5, 128): "2.6054991977135e-3", (5, 192): "2.6047925893927e-3",
}
TARGET = [(1, 48), (3, 96), (5, 48), (5, 96), (5, 192)]
DYADIC = [(r, n) for r in (1, 2, 3, 5) for n in (32, 64)]
ORDERING = "COLAMD"
SENSITIVITY_ORDERING = "MMD_AT_PLUS_A"  # unrefined V1/V3 only, target points
MAX_REFINE = 5
ASSEMBLIES = ("F_KRON", "F_D4H4", "F_DIV")
U = 2.0**-53
PRINTED_DIGITS = 14


# --- the integer 13-point operator ------------------------------------------------------------------------------

def stencil13() -> dict[tuple[int, int], int]:
    """Offsets (di, dj) -> integer coefficient of h^4 (delta_x^4 + 2 delta_x^2 delta_y^2 + delta_y^4)."""
    d4 = {-2: 1, -1: -4, 0: 6, 1: -4, 2: 1}
    d2 = {-1: 1, 0: -2, 1: 1}
    out: dict[tuple[int, int], int] = {}
    for k, c in d4.items():
        out[(k, 0)] = out.get((k, 0), 0) + c
        out[(0, k)] = out.get((0, k), 0) + c
    for a, ca in d2.items():
        for b, cb in d2.items():
            out[(a, b)] = out.get((a, b), 0) + 2 * ca * cb
    return out


def mirror(k: np.ndarray, n: int) -> np.ndarray:
    """Node index -1..n+1 -> unknown index 1..n-1, or 0 for a boundary node (w = 0). Ghost: w_-1 = w_1, w_{n+1} = w_{n-1}."""
    k = np.where(k == -1, 1, np.where(k == n + 1, n - 1, k))
    return np.where((k >= 1) & (k <= n - 1), k, 0)


def z_nodewise(nx: int, ny: int) -> sp.csr_matrix:
    """Integer Z assembled node by node from the 13-point stencil with ghost nodes; ordering k = (i-1)(ny-1) + (j-1)."""
    i, j = np.meshgrid(np.arange(1, nx), np.arange(1, ny), indexing="ij")
    i, j = i.ravel(), j.ravel()
    row = (i - 1) * (ny - 1) + (j - 1)
    rows, cols, vals = [], [], []
    for (di, dj), c in stencil13().items():
        ti, tj = mirror(i + di, nx), mirror(j + dj, ny)
        keep = (ti > 0) & (tj > 0)
        rows.append(row[keep])
        cols.append((ti[keep] - 1) * (ny - 1) + (tj[keep] - 1))
        vals.append(np.full(int(keep.sum()), c, dtype=np.int64))
    size = (nx - 1) * (ny - 1)
    z = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(size, size)).tocsr()
    z.sum_duplicates()
    z.eliminate_zeros()
    return z


def _d4_int(m: int) -> sp.csr_matrix:
    d = sp.diags([1, -4, 6, -4, 1], [-2, -1, 0, 1, 2], shape=(m, m), dtype=np.int64).tolil()
    d[0, 0] = d[m - 1, m - 1] = 7  # 6 + 1 from the mirrored ghost
    return d.tocsr()


def _t_int(m: int) -> sp.csr_matrix:
    return sp.diags([1, -2, 1], [-1, 0, 1], shape=(m, m), dtype=np.int64).tocsr()


def z_kron(nx: int, ny: int) -> sp.csr_matrix:
    """Second integer assembly: D4 (x) I + 2 T (x) T + I (x) D4 (exact in int64)."""
    mx, my = nx - 1, ny - 1
    ix, iy = sp.identity(mx, dtype=np.int64), sp.identity(my, dtype=np.int64)
    z = (sp.kron(_d4_int(mx), iy) + 2 * sp.kron(_t_int(mx), _t_int(my)) + sp.kron(ix, _d4_int(my))).tocsr()
    z.sum_duplicates()
    z.eliminate_zeros()
    return z


# --- F-assemblies with the rounded h = 1/n ----------------------------------------------------------------------

def f_kron(nx: int, ny: int, lx: float, ly: float) -> sp.csc_matrix:
    hx, hy = lx / nx, ly / ny

    def lap(m, h):
        return (sp.diags([np.ones(m - 1), -2.0 * np.ones(m), np.ones(m - 1)], [-1, 0, 1]) / h**2).tocsr()

    def d4(m, h):
        el = lap(m, h)
        corner = np.zeros(m)
        corner[0] = corner[-1] = 2.0 / h**4
        return (el @ el + sp.diags(corner)).tocsr()

    mx, my = nx - 1, ny - 1
    a = sp.kron(d4(mx, hx), sp.identity(my)) + 2.0 * sp.kron(lap(mx, hx), lap(my, hy)) + sp.kron(sp.identity(mx), d4(my, hy))
    return a.tocsc()


def f_d4h4(nx: int, ny: int, lx: float, ly: float) -> sp.csc_matrix:
    hx, hy = lx / nx, ly / ny

    def lap(m, h):
        return (sp.diags([np.ones(m - 1), -2.0 * np.ones(m), np.ones(m - 1)], [-1, 0, 1]) / h**2).tocsr()

    def d4(m, h):
        d = sp.diags([np.ones(m - 2), -4.0 * np.ones(m - 1), 6.0 * np.ones(m), -4.0 * np.ones(m - 1), np.ones(m - 2)],
                     [-2, -1, 0, 1, 2]).tolil()
        d[0, 0] = d[m - 1, m - 1] = 7.0
        return (d.tocsr() / h**4).tocsr()

    mx, my = nx - 1, ny - 1
    a = sp.kron(d4(mx, hx), sp.identity(my)) + 2.0 * sp.kron(lap(mx, hx), lap(my, hy)) + sp.kron(sp.identity(mx), d4(my, hy))
    return a.tocsc()


def f_div(z: sp.csr_matrix, n: int) -> sp.csc_matrix:
    h = 1.0 / n
    return (z.astype(float) / h**4).tocsc()


def assemble_f(kind: str, z, nx, ny, lx, ly, n) -> sp.csc_matrix:
    if kind == "F_KRON":
        return f_kron(nx, ny, lx, ly)
    if kind == "F_D4H4":
        return f_d4h4(nx, ny, lx, ly)
    return f_div(z, n)


def matrix_difference(a: sp.spmatrix, z: sp.spmatrix, n: int) -> tuple[float, int]:
    """Exact max_ij |a_ij - n^4 z_ij| / (n^4 |z_ij|) and the number of differing entries. n^4 z_ij < 2^53 is exact; a_ij and
    n^4 z_ij agree in sign and are within a factor 2, so the double subtraction is exact (Sterbenz)."""
    ref = (z.astype(float) * float(n**4)).tocsr()
    a = a.tocsr()
    if a.nnz != ref.nnz or not (np.array_equal(a.indptr, ref.indptr) and np.array_equal(a.indices, ref.indices)):
        a.sort_indices()
        ref.sort_indices()
    if not (np.array_equal(a.indptr, ref.indptr) and np.array_equal(a.indices, ref.indices)):
        raise ValueError("sparsity pattern of the F-assembly differs from Z")
    diff = a.data - ref.data
    rel = np.abs(diff) / np.abs(ref.data)
    return float(rel.max()), int(np.count_nonzero(diff))


# --- exact residual ---------------------------------------------------------------------------------------------

def _dyadic(values: np.ndarray) -> tuple[list[int], int]:
    """Doubles -> (integers p_k, e) with values_k = p_k / 2^e exactly (common power-of-two denominator)."""
    pairs = [float(v).as_integer_ratio() for v in values]
    e = max(q.bit_length() - 1 for _, q in pairs)
    return [p << (e - (q.bit_length() - 1)) for p, q in pairs], e


class ExactOperator:
    """Sparse matrix with dyadic-rational entries; exact residual b - factor * A x rounded once to double."""

    def __init__(self, a: sp.spmatrix):
        a = a.tocsr()
        a.sort_indices()
        self.size = a.shape[0]
        self.indptr = a.indptr
        self.cols = a.indices
        p, self.ea = _dyadic(a.data)
        self.p = np.array(p, dtype=object)

    def residual(self, x: np.ndarray, factor: int = 1) -> np.ndarray:
        """r = 1 - factor * A x (right-hand side 1), exact, rounded to nearest double."""
        m, ex = _dyadic(x)
        prod = self.p * np.array(m, dtype=object)[self.cols]
        s = np.add.reduceat(prod, self.indptr[:-1])
        den = 1 << (self.ea + ex)
        return np.array([(den - factor * int(v)) / den for v in s], dtype=float)


# --- solution ---------------------------------------------------------------------------------------------------

def refine(a: sp.csc_matrix, op: ExactOperator, factor: int = 1, ordering: str = ORDERING):
    """LU of the double matrix a, refinement with the exact residual of op (scaled by factor). Returns the first solve, the
    refined iterate, the corrections and the factor nnz."""
    lu = spla.splu(a, permc_spec=ordering)
    x0 = lu.solve(np.ones(a.shape[0]))
    x = x0.copy()
    corrections: list[float] = []
    previous = math.inf
    for _ in range(MAX_REFINE):
        dx = lu.solve(op.residual(x, factor))
        size = float(np.max(np.abs(dx)))
        x = x + dx
        corrections.append(size)
        if size >= previous or size == 0.0:
            break
        previous = size
    return x0, x, corrections, int(lu.L.nnz + lu.U.nnz)


def second_moment(zf: sp.csc_matrix, y: np.ndarray, steps: int = 3) -> np.ndarray:
    """g = Z^-1 y = Z^-2 1 (y the refined solution of Z y = 1). Only the first-order coefficient of eps_equiv: LU and a few
    double-residual refinement steps (relative accuracy far better than the 2 significant digits used)."""
    lu = spla.splu(zf, permc_spec=ORDERING)
    g = lu.solve(y)
    for _ in range(steps):
        g = g + lu.solve(y - zf @ g)
    return g


def unrefined(a: sp.csc_matrix, ordering: str = ORDERING) -> np.ndarray:
    return spla.spsolve(a, np.ones(a.shape[0]), permc_spec=ordering, use_umfpack=False)


def geometry(r: int, n: int, orientation: str):
    """orientation 'x': long side along x (plan); 'y': axes exchanged."""
    if orientation == "x":
        nx, ny, lx, ly = r * n, n, float(r), 1.0
    else:
        nx, ny, lx, ly = n, r * n, 1.0, float(r)
    centre = (nx // 2 - 1) * (ny - 1) + (ny // 2 - 1)
    return nx, ny, lx, ly, centre


def printed(value: float) -> str:
    return f"{value:.{PRINTED_DIGITS - 1}e}"


def reproduced(value: float, external: str) -> bool:
    """Plan 1.5: rounded to 14 significant digits equals the printed value."""
    return float(printed(value)) == float(external)


def peak_rss_mb() -> float | None:
    """Peak resident set of this process (cumulative), MB; None if unavailable."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

        c = Counters()
        c.cb = ctypes.sizeof(c)
        psapi = ctypes.WinDLL("psapi")
        kernel = ctypes.WinDLL("kernel32")
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        if psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(c), c.cb):
            return c.PeakWorkingSetSize / 2**20
        return None
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # kB on Linux
    except (ImportError, OSError):
        return None


def run_point(r: int, n: int, group: str, sensitivity: bool) -> list[dict]:
    rows = []
    n4 = n**4
    for orientation in ("x", "y"):
        nx, ny, lx, ly, c = geometry(r, n, orientation)
        z = z_nodewise(nx, ny)
        if (z != z_kron(nx, ny)).nnz:
            raise AssertionError(f"integer assemblies differ at r={r}, n={n}, {orientation}")
        zf = z.astype(float).tocsc()
        z_op = ExactOperator(z)
        z_norm = float(abs(z).sum(axis=1).max())  # ||Z||_inf (64)
        base = dict(r=r, n=n, nx=nx, ny=ny, orientation=orientation, group=group, external=EXTERNAL.get((r, n), ""))

        def record(variant, assembly, ordering, solver, alpha, y_norm, residual, seconds, mdiff=0.0, mcount=0,
                   corrections=(), factor_nnz="", **extra):
            """residual: exact 1 - Z y of the final iterate (y = n^4 w), rounded to double; backward error
            max|r| / (||Z||_inf ||y||_inf + 1)."""
            res = float(np.max(np.abs(residual)))
            rows.append(dict(base, variant=variant, assembly=assembly, refined=bool(corrections), ordering=ordering,
                             solver=solver, alpha=repr(float(alpha)), matrix_maxrel_vs_n4Z=mdiff,
                             matrix_entries_differing=mcount, residual_Z_max=res,
                             backward_error_Z=res / (z_norm * y_norm + 1.0), refine_steps=len(corrections),
                             last_correction_rel=corrections[-1] / abs(alpha) if corrections else "",
                             factor_nnz=factor_nnz, seconds=round(seconds, 3), **extra))

        def ynorm(v, scale=1):
            return scale * float(np.max(np.abs(v)))

        # integer operator: V3 (no refinement), V4 (refinement with the exact residual of Z) = alpha_ref QA
        orderings = (ORDERING, SENSITIVITY_ORDERING) if sensitivity else (ORDERING,)
        for ordering in orderings:
            t = time.perf_counter()
            y3 = unrefined(zf, ordering)
            record("V3", "INT", ordering, "spsolve", y3[c] / n4, ynorm(y3), z_op.residual(y3), time.perf_counter() - t)
        t = time.perf_counter()
        _, y4, corr, fnnz = refine(zf, z_op)
        record("V4", "INT", ORDERING, "splu+exact_refinement(Z)", y4[c] / n4, ynorm(y4), z_op.residual(y4),
               time.perf_counter() - t, corrections=[v / n4 for v in corr], factor_nnz=fnnz,
               z2_centre=repr(float(second_moment(zf, y4)[c])))

        # F-assemblies with the rounded h = 1/n: V1 (no refinement), V2 (refinement against A_F), V2X (against n^4 Z)
        for kind in ASSEMBLIES:
            a = assemble_f(kind, z, nx, ny, lx, ly, n)
            mdiff, mcount = matrix_difference(a, z, n)
            x1 = None
            for ordering in orderings:
                t = time.perf_counter()
                x = unrefined(a, ordering)
                record("V1", kind, ordering, "spsolve", float(x[c]), ynorm(x, n4), z_op.residual(x, n4),
                       time.perf_counter() - t, mdiff, mcount)
                x1 = x if ordering == ORDERING else x1
            t = time.perf_counter()
            x0, x2, corr, fnnz = refine(a, ExactOperator(a))
            record("V2", kind, ORDERING, "splu+exact_refinement(A_F)", float(x2[c]), ynorm(x2, n4), z_op.residual(x2, n4),
                   time.perf_counter() - t, mdiff, mcount, corr, fnnz, first_solve_equals_spsolve=bool(np.array_equal(x0, x1)))
            t = time.perf_counter()
            _, x2x, corr, fnnz = refine(a, z_op, n4)
            record("V2X", kind, ORDERING, "splu(A_F)+exact_refinement(n^4 Z)", float(x2x[c]), ynorm(x2x, n4),
                   z_op.residual(x2x, n4), time.perf_counter() - t, mdiff, mcount, corr, fnnz)
    rss = peak_rss_mb()
    for row in rows:
        row["peak_rss_mb_cumulative"] = round(rss, 1) if rss is not None else ""
    return rows


def annotate(rows: list[dict]) -> None:
    """Differences: against alpha_ref QA (V4 of the same orientation), against the consultant, against kkvim_reference
    (the stored CSV, separate step)."""
    stored = {}
    if EXTERNAL_CSV.exists():
        with EXTERNAL_CSV.open(newline="") as f:
            for line in csv.DictReader(f):
                stored[(int(line["r"]), int(line["n"]))] = line
    ref = {(row["r"], row["n"], row["orientation"]): float(row["alpha"]) for row in rows if row["variant"] == "V4"}
    for row in rows:
        a = float(row["alpha"])
        a_ref = ref[(row["r"], row["n"], row["orientation"])]
        row["rel_vs_alpha_ref_QA"] = (a - a_ref) / a_ref
        ext = row["external"]
        row["rel_vs_external"] = (a - float(ext)) / float(ext) if ext else ""
        row["reproduced_to_printed"] = reproduced(a, ext) if ext else ""
        if ext and row["variant"] == "V4":
            g = float(row["z2_centre"])
            row["eps_equiv_u"] = -(float(ext) - a_ref) * row["n"]**4 / g / U
        line = stored.get((row["r"], row["n"]))
        if line and row["variant"] == "V4":
            other = float(line["alpha_ref"] if row["orientation"] == "x" else line["alpha_ref_transposed"])
            e_other = float(line["e_D2"] if row["orientation"] == "x" else line["e_D2_transposed"])
            row["alpha_ref_kkvim_reference"] = repr(other)
            row["rel_vs_kkvim_reference"] = (a - other) / other
            # PL 1.4, item 4 rule (two refined solutions): |a - b| <= 4 u a + 4 (e_a + e_b), fixed before comparison
            e_self = float(row["last_correction_rel"]) * a
            bound = 4 * 2.0**-53 * a + 4 * (e_self + e_other)
            row["kkvim_reference_within_PL14_rule"] = abs(a - other) <= bound


FIELDS = ["group", "r", "n", "nx", "ny", "orientation", "variant", "assembly", "refined", "ordering", "solver", "alpha",
          "rel_vs_alpha_ref_QA", "external", "rel_vs_external", "reproduced_to_printed", "matrix_maxrel_vs_n4Z",
          "matrix_entries_differing", "residual_Z_max", "backward_error_Z", "refine_steps", "last_correction_rel", "first_solve_equals_spsolve",
          "alpha_ref_kkvim_reference", "rel_vs_kkvim_reference", "kkvim_reference_within_PL14_rule", "z2_centre",
          "eps_equiv_u", "factor_nnz",
          "seconds", "peak_rss_mb_cumulative"]


def environment() -> dict:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except OSError:
        commit = ""
    return dict(commit=commit, python=sys.version.split()[0], numpy=np.__version__, scipy=scipy.__version__,
                superlu=f"bundled with SciPy {scipy.__version__} (version not exposed by the API)",
                ordering=ORDERING, sensitivity_ordering=SENSITIVITY_ORDERING, max_refine=MAX_REFINE,
                platform=platform.platform(), machine=platform.machine(),
                pytest=_version("pytest"))


def _version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return ""


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--all", action="store_true", help="also the other external points (supplement)")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    points = [(p, "TARGET") for p in TARGET] + [(p, "DYADIC_CONTROL") for p in DYADIC]
    if args.all:
        points += [(p, "OTHER_EXTERNAL") for p in EXTERNAL if p not in TARGET and p not in DYADIC]
    rows = []
    start = time.perf_counter()
    for (r, n), group in points:
        t = time.perf_counter()
        rows += run_point(r, n, group, sensitivity=(group == "TARGET"))
        print(f"r={r} n={n} {group}: {time.perf_counter() - t:.1f} s", flush=True)
    annotate(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "runs.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in FIELDS})
    meta = environment()
    meta.update(points=len(points), seconds=round(time.perf_counter() - start, 1), peak_rss_mb=peak_rss_mb())
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
