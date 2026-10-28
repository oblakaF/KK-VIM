"""R5-V Q4, priority 1 (D-053.4): QA check of the arithmetic premises of the proven bound |E_s| <= B_E,s <= K* R_Sigma on the data of
the Q4 Energy B-C campaign, and the computation of B_E on those rows. QA research tool, not a qualification: K_E, numerical
tolerances and the form of the working energy test are NOT assigned; no B-C trajectory is created; the working ``kkvim`` is not
imported; r5_cccc_energy, the EQ script, the gamma tools and all Q3 / Q4 data are used read-only (no file of them is changed).

Sources: ST = docs/R5_CCCC_Q4_ENERGY_ERROR_BOUND.study.md (edition 47ea3ba: P-1, P-3', P-4', I-1, I-2, L-0...L-8, E-1...E-3, sect. 13);
RV = docs/R5_CCCC_Q4_ENERGY_ERROR_BOUND.independent-review.md (sect. 6, 7, 11); RO = docs/R5_CCCC_ROUNDOFF_PREQUAL_DRAFT.md (VOK-1...
VOK-8); EQ = scripts/r5_cccc_q4_energy_bc_audit.py and docs/results/r5_cccc_q4_energy_bc/ (historical campaign, commit cad6da7; the
code of src/ and of the EQ, gamma and C-3+ scripts is unchanged between cad6da7 and the base of this tool). The mathematical review is
NOT repeated: E-1...E-3 are taken as proven under their premises; this tool checks the premises on the actual numbers and evaluates B_E.

PROTOCOL - fixed in this docstring and the constants BEFORE the full run (separate local commit):

1. Inputs. The 123 NPZ of the Q3 campaign, read-only; sha256 of every file equal to docs/results/r5_cccc_q4_gamma_audit/meta.json AND
   to docs/results/r5_cccc_q4_energy_bc/meta.json, checked before any calculation (otherwise the campaign stops). Directory: --arrays
   (default KKVIM_Q3_ARRAYS, else Q3_ARRAYS_DEFAULT); the directory actually used is recorded. Sample: the 984 ids of the EQ rows.csv
   (= the keys of the Q3 CSV); the sha256 of both CSV files is recorded.
2. Environment recorded: Python, NumPy, SciPy, the BLAS library with its core and thread count read in the process from the library
   itself, CPU, OS, 12 spawn workers (one sequence per job), BLAS threads 1 (environment as in EQ), git HEAD at the start and at the
   end and the dirty paths at the start. Expected: the EQ environment (Python 3.13.15, NumPy 2.5.4, SciPy 1.18.1, scipy-openblas
   0.3.34, Windows 10, Intel64 Family 6 Model 183); a different environment is recorded, and the bitwise binding of 3 decides.
3. Reproduction and binding (ST 13 item 2; RV 6 G, 11 item 1). Every row runs the EQ path with the EQ FUNCTIONS THEMSELVES, in the
   EQ order (bc_input, run_halfstep, C' = fl(C R^T), exact_coefficients, solve_c3plus, orient, gamma_b1, w^ = fl(X Y^T), pi_double,
   pi_double, q_double on d^ = fl(w^_p - w^_s), model_terms), in the main thread of the worker process. Bitwise comparison with the
   historical rows.csv of P_hat_prev, P_hat_s, Q_hat, R_Sigma_up (required) and, for information, of gamma_down, gamma_up, Gamma_up
   and the R_Sigma terms. A difference makes the row UNRESOLVED; inputs and EQ data are never replaced; no tolerance.
4. U-1 (ST 13 item 1; RV H-5). For both fields of the row, previous w^_p = fl(X_p Y_p^T) and current w^_s = fl(X_s Y_s^T) - the
   SAME arrays as in 3 - exactly, elementwise, in integers from the bit patterns: (2^53 - N)|w^ - X Y^T| <= N |X||Y|^T, i.e.
   |e| <= gamma_N |X||Y|^T. Reported separately: previous / current field, N, zero entries of the factors, zero |X||Y|^T, strong
   compensation (2^20 |X Y^T| <= |X||Y|^T, including X Y^T = 0 < |X||Y|^T), the largest ratio |e|/(gamma_N |X||Y|^T). A second
   product X Y^T in the same process must be bitwise the same array. The check is a property of these arrays, not of the BLAS.
5. U-2 (ST 2 P-3', 13 item 2; RV H-2, H-4). In the SAME process and thread as 3:
   (a) a mode probe before and after every row: Python float and NumPy vectors of lengths 1, 3, 8, 257 and one BLAS product; FTZ
       (subnormal exact results of x, /, +, -), DAZ (subnormal operands), round to nearest (ties to even, both directions); bit-level
       comparisons only; operands built at run time (no constant folding); the SSE control word read by _controlfp (Windows);
   (b) a replication of pi_double / q_double operation by operation, as read in r5_cccc_energy (_delta_double, _quad_terms_double,
       pi_double, q_double, pairwise_sum); the stencil array t, the quadratic terms and the energies must be bitwise those of the
       module, which binds the checked operations to the module's values;
   (c) every x and / with nonzero finite operands: computed |result| > 2^-1022 strictly (P-3'(c) as corrected by H-2); a zero
       operand gives a zero result; x2 and negation: finite and zero only for a zero operand; + and -: a zero result only for
       a = -b (H-4); subnormal results of + and - counted (exact under gradual underflow); every intermediate finite (P-2 directly,
       besides L-0); D/2 and h/2 normal; the subtraction d^ = fl(w^_p - w^_s) checked like -;
   (d) subnormal operands of the checked operations counted: with none, DAZ cannot act; with some, the certificate relies on (a).
   All classifications use bit patterns (integer operations), so they do not depend on the FTZ/DAZ mode themselves.
6. N-7 (ST 13 item 3; RV H-7). For every row with eps_a_C_prev_up = 0 in the historical rows.csv or in the reproduction, exactly from
   the bit patterns: C' and U~ entries, the exact product X_p Y_p^T, the computed field. Zero only "by the double product" while
   the exact product is nonzero -> UNRESOLVED. The R-AC branch of EQ (AU 461, rows s = 1 with route (a) in Q3) is checked the same
   way on the stored R-AC previous factors.
7. U-3 (ST 4, 6.2, 6.4; RV 7). For the 36 grids, in the main process and again in every worker (bitwise equal): h_d = fl(l/n),
   c = h_d**2 (Python pow, as in EN), exact delta_h, delta_c, pow == fl(h_d h_d); eta, eta_t, psi, psi_L, c_q, nu, c_l;
   L = ceil(log2 M), L_Q by integer arithmetic (EN's float log2 compared, pairwise depth simulated); gamma_L, gamma_LQ; kappa_P,
   kappa_L, kappa_Q; Lambda, mu_low, Lambda_bar; K_P, K_Q, K_f, K* - exact rationals; normality of h_d, c, h_d/2, D/2, q h_dx,
   q h_dx h_dy and (h_dx/2)(h_dy/2); denominators 1 + delta_c > 0, u (L + 22) > 0, 1 - k u > 0 for k = L, L_Q, N, 4; the Reviewer's
   max(K* - 1) = 3.9581e-10 compared.
8. B_E (ST 5.1, 5.4; RV 11 item 5) by an own exact implementation (integer stencils of NS 2.2 with exact h = l/n): S_q, S_l, S_2 of
   w^_p, w^_s; S_q, S_2 of d^; ||d^||_h^2, ||d^||_a^2; ||S||_h^2 of both formation majorants (the integers of 4); the D2 residual
   of w^_s by the 13-point operator A = D4_x (x) I + 2 L_x (x) L_y + I (x) D4_y (not r5_cccc_d2). Roots: upper bounds in numerators,
   lower bounds in denominators (integer isqrt at 2^-80); D^(1/2) mu_low > 0 checked. B_E is an exact rational upper bound;
   then, exactly, |E_s| <= B_E,s with E_s = (P^_p - P^_s) - Q^ + gamma_s (gamma exact, from 3) and B_E,s <= K* R_Sigma with
   R_Sigma = the exact sum of the upward-rounded terms of model_terms. Every row also: own R^C, R_Q formulas <= the EN terms; own
   eps_a^C(s), eps_a^C(s - 1), ||d^||_a, B_omega <= the EN bounds (same radicands, finer roots); Prop. D (two routes of
   ||w^_s||_a^2) exactly.
9. Separate mathematical path on the fixed CONTROL set: every row whose grid has n_x n_y <= 4096 (288 rows: 32x32, 47x53, 40x64,
   64x40, 64x32, 32x64, 64x64, 96x32, 32x96; N = 1-3, s = 1-8, both directions, l_y in {1, 1/2, 2, 1/3, 3}). Exact parts
   E^(1)...E^(4) of E-1 with the 13-point form a_h(v, w) = D h_x h_y v^T A w (E^(1) without omega, by l_h = a_h(omega, .)); the
   identity E^(1) + E^(2) + E^(3) + E^(4) = E_s (gamma of path B1); each |E^(k)| <= its own bound of E-2; ||e||_a <= eps_a^C(s),
   ||e'||_a <= eps_a^C(s - 1); node by node L-4, L-5 (quadratic, load) and L-3 for the three double energies; L-6; own S, norms
   and residual = the exact functions of r5_cccc_energy and r5_cccc_d2.
10. Status of a row (no tolerance anywhere):
   VERIFIED - inputs (sha256, Q3 state reproduced), binding 4/4 bitwise, U-1 on both fields, U-2 (probes before and after OK, no
     violation, replication bitwise), U-3 of the grid, N-7 where applicable, B_E computed (finite, positive denominators), and
     |E_s| <= B_E,s and B_E,s <= K* R_Sigma; on control rows also every item of 9;
   UNRESOLVED - a premise not established or not checkable (each reason listed); the inequalities are still evaluated and reported,
     but not as verified;
   FAIL - all premises established and an inequality of 8 or 9 violated, or a cross-check of 8 or 9 contradicted.
   Campaign: QA_READY_FOR_REVIEW if every row is VERIFIED; PARTIAL if some rows are UNRESOLVED and none FAIL; BLOCKED on any FAIL, an
   unverified input or a contradiction with the proof.
Known limitation kept (not a FAIL): rows with K_max < K* (313 in RV 8) are not resolved by the test at K_E >= K*; they are counted
from the historical rows.csv with the K* of this run.

Usage (repository root):
    uv run python scripts/r5_cccc_q4_energy_bound_premises_audit.py run --profile full --arrays <dir> [--workers 12]
    uv run python scripts/r5_cccc_q4_energy_bound_premises_audit.py run --profile ci --out scratch/r5_cccc_q4_energy_bound_ci
"""

import argparse
import csv
import ctypes
import hashlib
import importlib.util
import json
import math
import multiprocessing
import operator
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from fractions import Fraction
from pathlib import Path

THREAD_VARS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
for _k in THREAD_VARS:
    os.environ[_k] = "1"  # as in Q3 and EQ: one BLAS thread

import numpy as np  # noqa: E402
import scipy  # noqa: E402

from kkvim_reference import r5_cccc_bc as bc  # noqa: E402
from kkvim_reference import r5_cccc_d2 as d2  # noqa: E402
from kkvim_reference import r5_cccc_energy as en  # noqa: E402
from kkvim_reference import r5_cccc_halfstep as hs  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "results" / "r5_cccc_q4_energy_error_bound"
EQ_SCRIPT = ROOT / "scripts" / "r5_cccc_q4_energy_bc_audit.py"
EQ_DIR = ROOT / "docs" / "results" / "r5_cccc_q4_energy_bc"
EQ_ROWS = EQ_DIR / "rows.csv"
EQ_META = EQ_DIR / "meta.json"
GA_META = ROOT / "docs" / "results" / "r5_cccc_q4_gamma_audit" / "meta.json"
Q3_CSV = ROOT / "docs" / "results" / "r5_cccc_q3_prequalification" / "sequences_e1_e7_e9.csv"
Q3_ARRAYS_DEFAULT = Path(os.environ.get(
    "KKVIM_Q3_ARRAYS", r"D:\git\Test_Versions\KK-VIM\KK-VIM-Q3-QA\scratch\r5_cccc_q3_prequalification\arrays"))
U = Fraction(1, 2**53)
C_R = 22  # operation count of the QA path G-4 in R_Sigma (EQ, EN); NOT a normative c_R
BIND_KEYS = ("P_hat_prev", "P_hat_s", "Q_hat", "R_Sigma_up")
EXT_KEYS = ("gamma_down", "gamma_up", "Gamma_up", "R_base_up", "R_per_cR_up", "R_f_up", "B_omega_up", "eps_a_C_s_up",
            "eps_a_C_prev_up", "d_hat_a_up", "R_C_prev_up", "R_C_s_up", "R_Q_up")
CONTROL_MAX_NODES = 4096  # control rule of item 9: n_x n_y <= 4096
SQRT_BITS = 80
COMPENSATION_BITS = 20
REVIEWER_KSTAR_MINUS_1 = "3.9581e-10"  # RV 7
KNOWN_KMAX_BELOW_KSTAR = 313  # RV 8
UNRESOLVED, VERIFIED, FAIL = "UNRESOLVED", "VERIFIED", "FAIL"
RESEARCH = "RESEARCH_ONLY / PROVISIONAL / NOT_QUALIFIED (D-049.5)"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _eq():
    """The EQ campaign script as a module (its functions define the reproduced path); read-only use."""
    if not hasattr(_eq, "mod"):
        _eq.mod = _load(EQ_SCRIPT, "r5_cccc_q4_energy_bc_audit")
    return _eq.mod


# ---------------------------------------------------------------------------------------------------------------------------------
# binary64 by bit patterns (integer operations only: independent of FTZ / DAZ)
# ---------------------------------------------------------------------------------------------------------------------------------

_ABS = np.uint64(0x7FFF_FFFF_FFFF_FFFF)
_EXP = np.uint64(0x7FF0_0000_0000_0000)
_TINY = np.uint64(0x0010_0000_0000_0000)  # 2^-1022
_SIGN = np.uint64(0x8000_0000_0000_0000)
_FRAC = np.uint64(0x000F_FFFF_FFFF_FFFF)


def _u64(a) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(a, dtype=np.float64)).view(np.uint64)


def abs_bits(a) -> np.ndarray:
    return _u64(a) & _ABS


def is_zero(a) -> np.ndarray:
    return abs_bits(a) == 0


def is_finite(a) -> np.ndarray:
    return (abs_bits(a) & _EXP) != _EXP


def gt_tiny(a) -> np.ndarray:
    """|a| > 2^-1022 strictly (for finite a)."""
    return abs_bits(a) > _TINY


def is_subnormal(a) -> np.ndarray:
    ab = abs_bits(a)
    return (ab != 0) & (ab < _TINY)


def negate(a) -> np.ndarray:
    """-a by flipping the sign bit (no floating-point operation)."""
    return (_u64(a) ^ _SIGN).view(np.float64)


def same_value(a, b) -> np.ndarray:
    """a = b as real numbers (+0 = -0), by bit patterns."""
    ua, ub = np.broadcast_arrays(_u64(a), _u64(b))
    return (ua == ub) | (((ua & _ABS) == 0) & ((ub & _ABS) == 0))


def same_bits(a, b) -> bool:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return a.shape == b.shape and bool(np.array_equal(_u64(a), _u64(b)))


def frac_of(x: float) -> Fraction:
    """Exact rational of a finite double, from its bit pattern."""
    (I,), E = dyadic(np.array([x]))
    return Fraction(int(I)) * p2(E)


def p2(e: int) -> Fraction:
    return Fraction(1 << e) if e >= 0 else Fraction(1, 1 << -e)


def dyadic(a) -> tuple[np.ndarray, int]:
    """a = I 2^E exactly: I an object array of Python ints, ONE exponent E for the whole array (from the bit patterns)."""
    arr = np.asarray(a, dtype=np.float64)
    if not np.all(is_finite(arr)):
        raise ValueError("non-finite entry")
    u = _u64(arr).ravel()
    ex = ((u >> np.uint64(52)) & np.uint64(0x7FF)).astype(np.int64)
    m = (u & _FRAC).astype(np.int64)
    m = np.where(ex == 0, m, m | (1 << 52))
    e = np.where(ex == 0, -1074, ex - 1075)
    nz = m != 0
    if not nz.any():
        return np.zeros(arr.shape, dtype=object), 0
    E = int(e[nz].min())
    sh = np.where(nz, e - E, 0)
    ints = np.array([int(mi) << int(si) for mi, si in zip(m.tolist(), sh.tolist())], dtype=object)
    neg = (u >> np.uint64(63)).astype(bool)
    ints = np.where(neg, -ints, ints)
    return ints.reshape(arr.shape), E


def at_exp(I: np.ndarray, E: int, E0: int) -> np.ndarray:
    """I 2^E written as an integer array at exponent E0 <= E."""
    if E < E0:
        raise ValueError("E < E0")
    return I * (1 << (E - E0)) if E > E0 else I


def isum(a) -> int:
    return int(np.sum(a)) if np.size(a) else 0


# ---------------------------------------------------------------------------------------------------------------------------------
# U-2 (a): arithmetic-mode probe of the calling thread
# ---------------------------------------------------------------------------------------------------------------------------------

_PROBES = (  # (name, op, a, b, expected): operands and results as hex strings, parsed at run time
    ("FTZ x: 2^-1022 * 1/2", "mul", "0x1p-1022", "0x1p-1", "0x0.8p-1022"),
    ("FTZ /: 2^-1022 / 4", "div", "0x1p-1022", "0x1p+2", "0x0.4p-1022"),
    ("FTZ -: 1.5 2^-1022 - 2^-1022", "sub", "0x1.8p-1022", "0x1p-1022", "0x0.8p-1022"),
    ("FTZ +: 2^-1074 + 2^-1074", "add", "0x0.0000000000001p-1022", "0x0.0000000000001p-1022", "0x0.0000000000002p-1022"),
    ("DAZ x: 2^-1060 * 2^60", "mul", "0x1p-1060", "0x1p+60", "0x1p-1000"),
    ("DAZ +: 2^-1074 + 0", "add", "0x0.0000000000001p-1022", "0x0p+0", "0x0.0000000000001p-1022"),
    ("DAZ /: 2^-1073 / 2^-1000", "div", "0x0.0000000000002p-1022", "0x1p-1000", "0x1p-73"),
    ("RN tie to even: 1 + 2^-53", "add", "0x1p+0", "0x1p-53", "0x1p+0"),
    ("RN tie to even up: (1 + 2^-52) + 2^-53", "add", "0x1.0000000000001p+0", "0x1p-53", "0x1.0000000000002p+0"),
    ("RN up: 1 + 1.5 2^-53", "add", "0x1p+0", "0x1.8p-53", "0x1.0000000000001p+0"),
    ("RN down: 1 + 2^-54", "add", "0x1p+0", "0x1p-54", "0x1p+0"),
    ("RN neg: -1 - 1.5 2^-53", "sub", "-0x1p+0", "0x1.8p-53", "-0x1.0000000000001p+0"),
    ("RN neg toward zero: -1 - 2^-54", "sub", "-0x1p+0", "0x1p-54", "-0x1p+0"),
    ("RN x: (1 + 2^-52)^2", "mul", "0x1.0000000000001p+0", "0x1.0000000000001p+0", "0x1.0000000000002p+0"),
)
_PY_OPS = {"add": operator.add, "sub": operator.sub, "mul": operator.mul, "div": operator.truediv}
_NP_OPS = {"add": np.add, "sub": np.subtract, "mul": np.multiply, "div": np.divide}


def _controlfp() -> dict:
    """SSE control word of the calling thread (Windows _controlfp(0, 0)); denormal control and rounding control decoded."""
    if sys.platform != "win32":
        return {"available": False}
    try:
        f = ctypes.CDLL("ucrtbase")._controlfp
        f.restype, f.argtypes = ctypes.c_uint, [ctypes.c_uint, ctypes.c_uint]
        cw = f(0, 0)
    except (OSError, AttributeError) as exc:
        return {"available": False, "error": str(exc)}
    dn = {0x0: "_DN_SAVE (no FTZ, no DAZ)", 0x01000000: "_DN_FLUSH (FTZ+DAZ)", 0x02000000: "_DN_FLUSH_OPERANDS_SAVE_RESULTS (DAZ)",
          0x03000000: "_DN_SAVE_OPERANDS_FLUSH_RESULTS (FTZ)"}[cw & 0x03000000]
    rc = {0x0: "_RC_NEAR", 0x100: "_RC_DOWN", 0x200: "_RC_UP", 0x300: "_RC_CHOP"}[cw & 0x300]
    return {"available": True, "word": hex(cw), "denormal_control": dn, "rounding": rc,
            "ok": (cw & 0x03000000) == 0 and (cw & 0x300) == 0}


def probe_mode() -> dict:
    """Behaviour of the CALLING thread: FTZ, DAZ and round-to-nearest probes (Python float, NumPy vectors, one BLAS product)."""
    fails = []
    for name, op, a, b, exp in _PROBES:
        x, y, z = float.fromhex(a), float.fromhex(b), float.fromhex(exp)
        if _PY_OPS[op](x, y).hex() != z.hex():
            fails.append(f"python {name}")
        for n in (1, 3, 8, 257):
            r = _NP_OPS[op](np.full(n, x), np.full(n, y))
            if not np.all(_u64(r) == _u64(np.array([z]))[0]):
                fails.append(f"numpy[{n}] {name}")
    t = float.fromhex("0x1p-1022")
    X = np.tile(np.array([[t, t, negate(np.array([t]))[0]]]), (8, 1))
    Y = np.full((5, 3), float.fromhex("0x1p-2"))
    if not np.all(_u64(X @ Y.T) == _u64(np.array([float.fromhex("0x0.4p-1022")]))[0]):
        fails.append("BLAS product with subnormal results")
    pw = float.fromhex("0x1p-537") ** 2  # informational: pow with a subnormal result
    cf = _controlfp()
    return {"ok": not fails and cf.get("ok", True), "fails": fails, "controlfp": cf, "thread": threading.get_native_id(),
            "main_thread": threading.current_thread() is threading.main_thread(),
            "pow_subnormal_exact": pw.hex() == float.fromhex("0x0.0000000000001p-1022").hex()}


def blas_info() -> dict:
    """BLAS library of NumPy, its core and thread count, read in this process from the loaded library."""
    out = {"numpy_blas": None}
    try:
        cfg = np.show_config(mode="dicts")
        out["numpy_blas"] = {k: cfg["Build Dependencies"]["blas"].get(k) for k in ("name", "version", "openblas configuration")}
    except Exception as exc:  # recorded, never hidden
        out["numpy_blas_error"] = str(exc)
    libs = Path(np.__file__).resolve().parent.parent / "numpy.libs"
    dll = next(iter(sorted(libs.glob("*openblas*"))), None) if libs.exists() else None
    if dll is None:
        out["library"] = "not found"
        return out
    try:
        lib = ctypes.CDLL(str(dll))
        nt = lib.scipy_openblas_get_num_threads64_
        nt.restype = ctypes.c_int
        core = lib.scipy_openblas_get_corename64_
        core.restype = ctypes.c_char_p
        out.update({"library": dll.name, "threads": nt(), "core": core().decode()})
    except (OSError, AttributeError) as exc:
        out["library_error"] = str(exc)
    return out


# ---------------------------------------------------------------------------------------------------------------------------------
# U-2 (b)-(d): replication of the double path G-4 operation by operation, with the premise checks
# ---------------------------------------------------------------------------------------------------------------------------------

class OpLog:
    """Counts and violations of the checked operations (ST 2, P-3'; RV H-2, H-4); classifications by bit patterns."""

    def __init__(self):
        self.n = {"mul": 0, "div": 0, "add": 0, "exact2": 0, "neg": 0, "pow": 0}
        self.violations: list[str] = []
        self.cancellations = 0  # + / - with zero result and nonzero operands (exact cancellation a = -b)
        self.subnormal_results = 0  # + / - with subnormal result (exact under gradual underflow)
        self.subnormal_operands = 0

    def _flag(self, name: str, what: str, mask) -> None:
        k = int(np.count_nonzero(mask))
        if k:
            self.violations.append(f"{name}: {what} ({k})")

    def mul(self, name, a, b, r, div=False):
        a, b, r = np.broadcast_arrays(np.asarray(a, dtype=float), np.asarray(b, dtype=float), np.asarray(r, dtype=float))
        self.n["div" if div else "mul"] += r.size
        self.subnormal_operands += int(np.count_nonzero(is_subnormal(a)) + np.count_nonzero(is_subnormal(b)))
        fin = is_finite(r) & is_finite(a) & is_finite(b)
        self._flag(name, "non-finite operand or result", ~fin)
        if div:
            self._flag(name, "division by zero", is_zero(b))
            nz = ~is_zero(a)
        else:
            nz = ~(is_zero(a) | is_zero(b))
        self._flag(name, "nonzero operands, |result| <= 2^-1022", nz & ~gt_tiny(r) & fin)
        self._flag(name, "zero operand, nonzero result", ~nz & ~is_zero(r))

    def div(self, name, a, b, r):
        self.mul(name, a, b, r, div=True)

    def add(self, name, a, b, r, sub=False):
        a, b, r = np.broadcast_arrays(np.asarray(a, dtype=float), np.asarray(b, dtype=float), np.asarray(r, dtype=float))
        c = negate(b) if sub else b  # r = a + c
        self.n["add"] += r.size
        self.subnormal_operands += int(np.count_nonzero(is_subnormal(a)) + np.count_nonzero(is_subnormal(c)))
        self._flag(name, "non-finite operand or result", ~(is_finite(r) & is_finite(a) & is_finite(c)))
        zr = is_zero(r)
        self._flag(name, "zero result with a != -b (flush to zero)", zr & ~same_value(a, negate(c)))
        self.cancellations += int(np.count_nonzero(zr & ~(is_zero(a) & is_zero(c))))
        self.subnormal_results += int(np.count_nonzero(is_subnormal(r)))

    def exact2(self, name, b, r):
        b, r = np.broadcast_arrays(np.asarray(b, dtype=float), np.asarray(r, dtype=float))
        self.n["exact2"] += r.size
        self.subnormal_operands += int(np.count_nonzero(is_subnormal(b)))
        self._flag(name, "x2: non-finite", ~(is_finite(r) & is_finite(b)))
        self._flag(name, "x2: zero pattern changed", is_zero(r) != is_zero(b))

    def neg(self, name, b, r):
        self.n["neg"] += np.size(r)
        self._flag(name, "negation is not the sign flip", ~(_u64(r) == _u64(negate(b))))

    def power(self, name, b, r):
        self.n["pow"] += 1
        self._flag(name, "pow: not finite or |result| <= 2^-1022", ~(is_finite(r) & gt_tiny(r)))

    def as_dict(self) -> dict:
        return {"ops": dict(self.n), "violations": list(self.violations), "cancellations": self.cancellations,
                "subnormal_results_add": self.subnormal_results, "subnormal_operands": self.subnormal_operands}


def checked_pairwise_sum(terms: np.ndarray, log: OpLog, name: str) -> tuple[float, int]:
    """en.pairwise_sum operation by operation (same tree), every addition checked; returns (value, number of levels)."""
    v = np.asarray(terms, dtype=float).ravel()
    if v.size == 0:
        return 0.0, 0
    lvl = 0
    while v.size > 1:
        if v.size % 2:
            v = np.append(v, 0.0)
        a, b = v[0::2], v[1::2]
        r = a + b
        log.add(f"{name}: pairwise level {lvl}", a, b, r)
        v, lvl = r, lvl + 1
    return float(v[0]), lvl


def replicate_energy(g: en.Grid, w: np.ndarray, log: OpLog, name: str, with_load: bool, keep: bool = False) -> dict:
    """pi_double (with_load) or q_double of ``w`` replicated operation by operation as read in r5_cccc_energy (EN 203-247), each
    operation checked; the stencil array, the quadratic terms and the value are compared bitwise with the module."""
    w = np.asarray(w, dtype=float)
    lx, nx, ly, ny = g.lx, g.nx, g.ly, g.ny
    hx = lx / nx
    log.div(f"{name}: h_x = l_x/n_x", lx, float(nx), hx)
    hy = ly / ny
    log.div(f"{name}: h_y = l_y/n_y", ly, float(ny), hy)
    cx = hx**2
    log.power(f"{name}: h_x**2", hx, cx)
    cy = hy**2
    log.power(f"{name}: h_y**2", hy, cy)
    P = np.zeros((nx + 1, ny + 1))
    P[1:-1, 1:-1] = w
    parts = []
    for ax, c in ((0, cx), (1, cy)):
        Pa = P if ax == 0 else P.T
        a, b, cc = Pa[:-2, 1:-1], Pa[1:-1, 1:-1], Pa[2:, 1:-1]
        b2 = 2.0 * b
        log.exact2(f"{name}: 2 P (interior, axis {ax})", b, b2)
        s1 = a - b2
        log.add(f"{name}: P_- - 2 P (axis {ax})", a, b2, s1, sub=True)
        s2 = s1 + cc
        log.add(f"{name}: (P_- - 2 P) + P_+ (axis {ax})", s1, cc, s2)
        li = s2 / c
        log.div(f"{name}: interior / h^2 (axis {ax})", s2, c, li)
        e0, e1 = 2.0 * Pa[1, 1:-1], 2.0 * Pa[-2, 1:-1]
        log.exact2(f"{name}: 2 P_1 (boundary, axis {ax})", Pa[1, 1:-1], e0)
        log.exact2(f"{name}: 2 P_n-1 (boundary, axis {ax})", Pa[-2, 1:-1], e1)
        b0, b1 = e0 / c, e1 / c
        log.div(f"{name}: boundary / h^2 (axis {ax})", e0, c, b0)
        log.div(f"{name}: boundary / h^2 (axis {ax})", e1, c, b1)
        La = np.zeros_like(Pa)
        La[1:-1, 1:-1] = li
        La[0, 1:-1], La[-1, 1:-1] = b0, b1
        parts.append(La if ax == 0 else La.T)
    t = parts[0] + parts[1]
    log.add(f"{name}: t = L_x + L_y", parts[0], parts[1], t)
    out = {"t_bitwise_module": same_bits(t, en._delta_double(g, w)), "hx": hx, "hy": hy, "cx": cx, "cy": cy}
    hx2, hy2 = hx / 2, hy / 2
    log.div(f"{name}: h_x/2", hx, 2.0, hx2)
    log.div(f"{name}: h_y/2", hy, 2.0, hy2)
    wx = np.full(nx + 1, hx)
    wx[[0, -1]] = hx2
    wy = np.full(ny + 1, hy)
    wy[[0, -1]] = hy2
    O = np.outer(wx, wy)
    log.mul(f"{name}: outer weights", wx[:, None], wy[None, :], O)
    D2 = g.D / 2
    log.div(f"{name}: D/2", g.D, 2.0, D2)
    F = D2 * O
    log.mul(f"{name}: (D/2) W", D2, O, F)
    tt = t * t
    log.mul(f"{name}: t t", t, t, tt)
    Tq = F * tt
    log.mul(f"{name}: ((D/2) W)(t t)", F, tt, Tq)
    out["quad_bitwise_module"] = same_bits(Tq, en._quad_terms_double(g, w))
    if with_load:
        hxl, hyl = lx / nx, ly / ny
        qh = g.q * hxl
        log.mul(f"{name}: q h_x", g.q, hxl, qh)
        qhh = qh * hyl
        log.mul(f"{name}: (q h_x) h_y", qh, hyl, qhh)
        ld = qhh * w
        log.mul(f"{name}: load (q h_x h_y) w", qhh, w, ld)
        nld = -ld.ravel()
        log.neg(f"{name}: -load", ld.ravel(), nld)
        terms = np.concatenate([Tq.ravel(), nld])
        value, lvl = checked_pairwise_sum(terms, log, name)
        out["value_bitwise_module"] = value.hex() == en.pi_double(g, w).hex()
    else:
        ld = None
        terms = Tq.ravel()
        value, lvl = checked_pairwise_sum(terms, log, name)
        out["value_bitwise_module"] = value.hex() == en.q_double(g, w).hex()
    out.update({"value": value, "levels": lvl, "subnormal_inputs": int(np.count_nonzero(is_subnormal(w)))})
    if keep:
        out.update({"_t": t, "_Tq": Tq, "_load": ld, "_terms": terms})
    return out


# ---------------------------------------------------------------------------------------------------------------------------------
# U-1: exact elementwise formation check
# ---------------------------------------------------------------------------------------------------------------------------------

def formation_check(X, Y, W) -> dict:
    """(2^53 - N)|W - X Y^T| <= N |X||Y|^T exactly, elementwise (gamma_N = N u/(1 - N u)); X Y^T and |X||Y|^T as integers."""
    X, Y, W = (np.asarray(a, dtype=float) for a in (X, Y, W))
    N = X.shape[1]
    XI, EX = dyadic(X)
    YI, EY = dyadic(Y)
    WI, EW = dyadic(W)
    Wt = XI.dot(YI.T) if N else np.zeros(W.shape, dtype=object)
    S = np.abs(XI).dot(np.abs(YI).T) if N else np.zeros(W.shape, dtype=object)
    EP = EX + EY
    E0 = min(EP, EW)
    Wt0, S0, W0 = at_exp(Wt, EP, E0), at_exp(S, EP, E0), at_exp(WI, EW, E0)
    err = np.abs(W0 - Wt0)
    lhs, rhs = err * (2**53 - N), S0 * N
    ok = lhs <= rhs
    pos = S0 > 0
    ratio = max((float(x) for x in (lhs[pos] / rhs[pos]).ravel()), default=0.0)
    comp = pos & (np.abs(Wt0) * (1 << COMPENSATION_BITS) <= S0)
    ratio_comp = max((float(x) for x in (lhs[comp] / rhs[comp]).ravel()), default=0.0) if np.any(comp) else None
    # exact products of nonzero factor entries: all of modulus >= 2^-1022 ? (informational; U-1 replaces P-3'(c) for them)
    prod_normal = True
    for k in range(N):
        xs = [abs(int(v)) for v in XI[:, k] if v != 0]
        ys = [abs(int(v)) for v in YI[:, k] if v != 0]
        if xs and ys and Fraction(min(xs) * min(ys)) * p2(EP) < p2(-1022):
            prod_normal = False
    return {"holds": bool(np.all(ok)), "violations": int(np.count_nonzero(~ok)), "elements": int(W.size),
            "max_ratio": ratio, "zero_S": int(np.count_nonzero(~pos)), "exact_errors_zero": int(np.count_nonzero(err == 0)),
            "compensation": int(np.count_nonzero(comp)), "max_ratio_compensation": ratio_comp,
            "zero_factor_entries": int(np.count_nonzero(is_zero(X)) + np.count_nonzero(is_zero(Y))),
            "factor_all_zero": bool(np.all(is_zero(X)) or np.all(is_zero(Y))), "exact_product_zero": bool(np.all(Wt == 0)),
            "computed_field_zero": bool(np.all(is_zero(W))), "exact_products_normal": prod_normal,
            "_S": S, "_ES": EP, "_Wt": Wt, "_EWt": EP, "_W": WI, "_EW": EW}


# ---------------------------------------------------------------------------------------------------------------------------------
# U-3: grid constants (exact rationals; ST 4, 6.2)
# ---------------------------------------------------------------------------------------------------------------------------------

def gam(k: int) -> Fraction:
    if k * U >= 1:
        raise ValueError("k u >= 1")
    return k * U / (1 - k * U)


def ceil_log2(m: int) -> int:
    return (m - 1).bit_length()


def pairwise_levels(m: int) -> int:
    lvl = 0
    while m > 1:
        m = (m + 1) // 2
        lvl += 1
    return lvl


def grid_constants(lx: float, nx: int, ly: float, ny: int, D: float = 1.0, q: float = 1.0, N: int = 3) -> dict:
    """K* and its ingredients for one grid on THIS platform (h_d, c = h_d**2 computed here as in EN)."""
    hxd, hyd = lx / nx, ly / ny
    cx, cy = hxd**2, hyd**2
    hx, hy = Fraction(lx) / nx, Fraction(ly) / ny
    dhx, dhy = frac_of(hxd) / hx - 1, frac_of(hyd) / hy - 1
    dcx, dcy = frac_of(cx) / hx**2 - 1, frac_of(cy) / hy**2 - 1
    one = Fraction(1)
    up_, dn_ = one + U, one - U
    eta = max(max(up_**3 / (1 + d) - 1, 1 - dn_**3 / (1 + d)) for d in (dcx, dcy))
    eta_t = eta * (1 + U) + U
    phi = (1 + dhx) * (1 + dhy)
    psi = max(phi * up_**4 - 1, 1 - phi * dn_**4)
    psiL = max(phi * up_**3 - 1, 1 - phi * dn_**3)
    cq = psi + 2 * (1 + psi) * eta_t
    nu = (1 + psi) * eta_t**2
    M = (nx + 1) * (ny + 1) + (nx - 1) * (ny - 1)
    MQ = (nx + 1) * (ny + 1)
    L, LQ = ceil_log2(M), ceil_log2(MQ)
    gL, gLQ = gam(L), gam(LQ)
    kP, kL, kQ = gL + (1 + gL) * cq, gL + (1 + gL) * psiL, gLQ + (1 + gLQ) * cq
    Lam = 4 / hx**2 + 4 / hy**2
    mu_low = 4 / Fraction(lx) ** 2 + 4 / Fraction(ly) ** 2
    Lbar = Lam / mu_low
    den = U * (L + C_R)
    KP = max(kP + (1 + gL) * nu * Lbar**2, kL) / den
    KQ = max(kQ / den, 1 + U * Lbar / 2 + (1 + gLQ) * nu * Lbar / (2 * U))
    Kf = 1 + U * Lbar
    Kstar = max(KP, KQ, Kf)
    normal = {"h_dx": hxd, "h_dy": hyd, "c_x": cx, "c_y": cy, "h_dx/2": hxd / 2, "h_dy/2": hyd / 2, "D/2": D / 2,
              "q h_dx": q * hxd, "q h_dx h_dy": q * hxd * hyd, "(h_dx/2)(h_dy/2)": (hxd / 2) * (hyd / 2)}
    normal_ok = all(bool(gt_tiny(np.array([v]))[0] and is_finite(np.array([v]))[0]) for k, v in normal.items()
                    if not (k.startswith("q") and q == 0))
    denominators_ok = (1 + dcx > 0 and 1 + dcy > 0 and den > 0 and mu_low > 0
                       and all(k * U < 1 for k in (L, LQ, N, 4)))
    return {"lx": lx, "nx": nx, "ly": ly, "ny": ny, "h_dx": hxd, "h_dy": hyd, "c_x": cx, "c_y": cy,
            "pow_equals_mul": cx.hex() == (hxd * hxd).hex() and cy.hex() == (hyd * hyd).hex(),
            "delta_hx": dhx, "delta_hy": dhy, "delta_cx": dcx, "delta_cy": dcy, "eta": eta, "eta_t": eta_t, "psi": psi,
            "psi_L": psiL, "c_q": cq, "nu": nu, "c_l": psiL, "M": M, "M_Q": MQ, "L": L, "L_Q": LQ,
            "L_equals_EN_log2": L == math.ceil(math.log2(M)) and LQ == math.ceil(math.log2(MQ)),
            "L_equals_pairwise_depth": L == pairwise_levels(M) and LQ == pairwise_levels(MQ),
            "gamma_L": gL, "gamma_LQ": gLQ, "kappa_P": kP, "kappa_L": kL, "kappa_Q": kQ, "Lambda": Lam, "mu_low": mu_low,
            "Lambda_bar": Lbar, "K_P": KP, "K_Q": KQ, "K_f": Kf, "K_star": Kstar, "normal_ok": normal_ok,
            "denominators_ok": denominators_ok}


def grid_bits(c: dict) -> tuple:
    return tuple(float(c[k]).hex() for k in ("h_dx", "h_dy", "c_x", "c_y"))


def grid_ok(c: dict) -> bool:
    return bool(c["normal_ok"] and c["denominators_ok"] and c["L_equals_pairwise_depth"])


# ---------------------------------------------------------------------------------------------------------------------------------
# exact integer energy quantities (NS 2.2 with exact h = l/n; 13-point operator A of RO 0.2)
# ---------------------------------------------------------------------------------------------------------------------------------

class GridQ:
    """Exact grid scalars: 1/h_x^2 = k_x/Z, 1/h_y^2 = k_y/Z (integers), weights 4 w_I w_J/(h_x h_y) in {1, 2, 4}."""

    def __init__(self, lx: float, nx: int, ly: float, ny: int, D: float = 1.0, q: float = 1.0):
        self.nx, self.ny = nx, ny
        self.hx, self.hy = Fraction(lx) / nx, Fraction(ly) / ny
        rx, ry = 1 / self.hx**2, 1 / self.hy**2
        self.Z = math.lcm(rx.denominator, ry.denominator)
        self.kx, self.ky = int(rx * self.Z), int(ry * self.Z)
        self.D, self.q = Fraction(D), Fraction(q)
        self.Lam = 4 * rx + 4 * ry
        self.mu_low = 4 / Fraction(lx) ** 2 + 4 / Fraction(ly) ** 2
        wx = np.full(nx + 1, 2, dtype=object)
        wx[0] = wx[-1] = 1
        wy = np.full(ny + 1, 2, dtype=object)
        wy[0] = wy[-1] = 1
        self.c4 = np.outer(wx, wy)


def _hat(I: np.ndarray, nx: int, ny: int, absolute: bool):
    """Integer parts of L^_x (x) E_y and E_x (x) L^_y (h^2 not included) on all nodes; |L^| with |w| if absolute (VOK-3)."""
    P = np.zeros((nx + 1, ny + 1), dtype=object)
    P[1:nx, 1:ny] = np.abs(I) if absolute else I
    k = 2 if absolute else -2
    Sx = np.zeros((nx + 1, ny + 1), dtype=object)
    Sx[1:nx, :] = P[:-2, :] + k * P[1:-1, :] + P[2:, :]
    Sx[0, :], Sx[nx, :] = 2 * P[1, :], 2 * P[nx - 1, :]
    Sy = np.zeros((nx + 1, ny + 1), dtype=object)
    Sy[:, 1:ny] = P[:, :-2] + k * P[:, 1:-1] + P[:, 2:]
    Sy[:, 0], Sy[:, ny] = 2 * P[:, 1], 2 * P[:, ny - 1]
    return Sx, Sy


def stencil_ints(gq: GridQ, I: np.ndarray):
    """t = 2^E T/Z and A = |Delta|^C|w| = 2^E A/Z for w = 2^E I (integers T, A on all nodes)."""
    Sx, Sy = _hat(I, gq.nx, gq.ny, False)
    T = Sx * gq.kx + Sy * gq.ky
    Ax, Ay = _hat(I, gq.nx, gq.ny, True)
    A = Ax * gq.kx + Ay * gq.ky
    return T, A


def field_sums(gq: GridQ, I: np.ndarray, E: int, keep: bool = False) -> dict:
    """Exact S_q, S_l, S_2, ||w||_a^2, ||w||_h^2 of w = 2^E I (ST 4: S_q, S_l = parts of S^C; S_2 = (D/2) sum W A^2)."""
    T, A = stencil_ints(gq, I)
    c4 = gq.c4
    base = gq.hx * gq.hy / 4 * p2(2 * E) / gq.Z**2
    out = {"Sq": gq.D / 2 * base * isum(c4 * np.abs(T) * A), "S2": gq.D / 2 * base * isum(c4 * A * A),
           "a2": gq.D * base * isum(c4 * T * T), "Sl": abs(gq.q) * gq.hx * gq.hy * p2(E) * isum(np.abs(I)),
           "h2": gq.hx * gq.hy * p2(2 * E) * isum(I * I)}
    if keep:
        out.update({"_T": T, "_A": A, "_E": E})
    return out


def _L1(P: np.ndarray, axis: int) -> np.ndarray:
    """Integer (1, -2, 1) along ``axis`` with zero Dirichlet ends (interior arrays)."""
    Q = np.moveaxis(P, axis, 0)
    R = np.zeros_like(Q)
    R[:] = -2 * Q
    R[1:] += Q[:-1]
    R[:-1] += Q[1:]
    return np.moveaxis(R, 0, axis)


def _D4(P: np.ndarray, axis: int) -> np.ndarray:
    """Integer D4 = L^2 + 2(e_1 e_1^T + e_m e_m^T) along ``axis``: rows (7, -4, 1), (-4, 6, -4, 1), (1, -4, 6, -4, 1)."""
    R = np.moveaxis(_L1(_L1(P, axis), axis), axis, 0).copy()
    Q = np.moveaxis(P, axis, 0)
    R[0] += 2 * Q[0]
    R[-1] += 2 * Q[-1]
    return np.moveaxis(R, 0, axis)


def apply_A(gq: GridQ, I: np.ndarray) -> np.ndarray:
    """Z^2 (A w)/2^E for w = 2^E I: A = D4_x (x) I + 2 L_x (x) L_y + I (x) D4_y (RO 0.2), integers."""
    return _D4(I, 0) * (gq.kx * gq.kx) + 2 * _L1(_L1(I, 0), 1) * (gq.kx * gq.ky) + _D4(I, 1) * (gq.ky * gq.ky)


def a13(gq: GridQ, v: tuple, w: tuple) -> Fraction:
    """a_h(v, w) = D h_x h_y v^T A w for v = 2^Ev Iv, w = 2^Ew Iw (13-point route, Prop. D)."""
    (Iv, Ev), (Iw, Ew) = v, w
    return gq.D * gq.hx * gq.hy * p2(Ev + Ew) * isum(Iv * apply_A(gq, Iw)) / gq.Z**2


def residual_h2(gq: GridQ, I: np.ndarray, E: int, AI: np.ndarray | None = None) -> Fraction:
    """||D A w - q 1||_h^2 for w = 2^E I (VOK-7 (b)), exactly."""
    AI = apply_A(gq, I) if AI is None else AI
    c1 = gq.D * p2(E) / gq.Z**2  # D A w = c1 AI
    num = c1.numerator * gq.q.denominator
    sub = gq.q.numerator * c1.denominator
    R = AI * num - sub
    return gq.hx * gq.hy * Fraction(isum(R * R), (c1.denominator * gq.q.denominator) ** 2)


def sqrt_bounds(x: Fraction, bits: int = SQRT_BITS) -> tuple[Fraction, Fraction]:
    """lo <= sqrt(x) <= hi (exact when x is a square of the grid); x >= 0."""
    if x < 0:
        raise ValueError("negative radicand")
    if x == 0:
        return Fraction(0), Fraction(0)
    P, Q = x.numerator, x.denominator
    n = P * Q << (2 * bits)
    r = math.isqrt(n)
    lo = Fraction(r, Q << bits)
    return lo, (lo if r * r == n else Fraction(r + 1, Q << bits))


def directed(x: Fraction) -> tuple[float, float]:
    return _eq()._gi().directed(x)


def b_e(gq: GridQ, kc: dict, N: int, fp: dict, fs_: dict, dh: np.ndarray, keep: bool = False) -> dict:
    """B_E,s of E-2 (ST 5.1) as an exact rational UPPER bound (roots up in numerators, down in denominators; ST 5.4) from the
    computed fields (integers of formation_check) and d^; also the parts B1 (formation), B2 (energies), B3 (subtraction), B4 (Q)."""
    wp, ws = (fp["_W"], fp["_EW"]), (fs_["_W"], fs_["_EW"])
    dd = dyadic(dh)
    sums = {"p": field_sums(gq, *wp, keep=keep), "s": field_sums(gq, *ws, keep=keep), "d": field_sums(gq, *dd, keep=keep)}
    sD_lo, sD_hi = sqrt_bounds(gq.D)
    S2p = gq.hx * gq.hy * p2(2 * fp["_ES"]) * isum(fp["_S"] * fp["_S"])
    S2s = gq.hx * gq.hy * p2(2 * fs_["_ES"]) * isum(fs_["_S"] * fs_["_S"])
    eps_s = gam(N) * sD_hi * gq.Lam * sqrt_bounds(S2s)[1]
    eps_p = gam(N) * sD_hi * gq.Lam * sqrt_bounds(S2p)[1]
    dh_h = sqrt_bounds(sums["d"]["h2"])[1]
    dh_a = sqrt_bounds(sums["d"]["a2"])[1]
    beta = U * sD_hi * gq.Lam * dh_h
    AI = apply_A(gq, ws[0])
    res_h2 = residual_h2(gq, ws[0], ws[1], AI)
    denom = sD_lo * gq.mu_low
    out = {"wp": wp, "ws": ws, "dd": dd, "sums": sums, "eps_s": eps_s, "eps_p": eps_p, "dh_a": dh_a, "beta": beta,
           "denominator_positive": denom > 0, "Bw": None, "bnd": None, "BE": None,
           "PropD": gq.D * gq.hx * gq.hy * p2(2 * ws[1]) * isum(ws[0] * AI) / gq.Z**2 == sums["s"]["a2"]}
    if denom <= 0:
        return out
    Bw = sqrt_bounds(res_h2)[1] / denom + eps_s

    def BP(f):
        return kc["kappa_P"] * f["Sq"] + kc["kappa_L"] * f["Sl"] + (1 + kc["gamma_L"]) * kc["nu"] * f["S2"]

    BQ = kc["kappa_Q"] * sums["d"]["Sq"] + (1 + kc["gamma_LQ"]) * kc["nu"] * sums["d"]["S2"]
    bnd = {"B2": BP(sums["p"]) + BP(sums["s"]), "B4": BQ, "B3": beta * dh_a + beta**2 / 2,
           "B1": eps_s * (dh_a + beta) + Bw * (eps_p + eps_s), "eps_s": eps_s, "eps_p": eps_p, "res_h2": res_h2}
    out.update({"Bw": Bw, "bnd": bnd, "BE": bnd["B1"] + bnd["B2"] + bnd["B3"] + bnd["B4"]})
    return out


# ---------------------------------------------------------------------------------------------------------------------------------
# separate mathematical path (control rows; item 9)
# ---------------------------------------------------------------------------------------------------------------------------------

def lin_le(lhs: list, rhs: list) -> tuple[bool, float]:
    """Elementwise |sum_i a_i X_i| <= sum_j b_j Y_j exactly (a_i, b_j rationals, X_i, Y_j integer arrays); (holds, max ratio)."""
    den = 1
    for a, _ in lhs + rhs:
        den = math.lcm(den, Fraction(a).denominator)
    L = sum((int(Fraction(a) * den) * X for a, X in lhs), 0)
    R = sum((int(Fraction(b) * den) * Y for b, Y in rhs), 0)
    L, R = np.abs(np.asarray(L, dtype=object)), np.asarray(R, dtype=object)
    L, R = np.broadcast_arrays(L, R)
    ok = L <= R
    pos = R > 0
    ratio = max((float(x) for x in (L[pos] / R[pos]).ravel()), default=0.0)
    if np.any(~pos & (L > 0)):
        ratio = math.inf
    return bool(np.all(ok)), ratio


def node_checks(gq: GridQ, kc: dict, rep: dict, fs: dict, I: np.ndarray, E: int, value: float, with_load: bool) -> dict:
    """L-4 (stencil), L-5 (quadratic, load), L-3 (pairwise sum) node by node, exactly, for one double energy."""
    T, A = fs["_T"], fs["_A"]
    out = {}
    ti, Et = dyadic(rep["_t"])
    E0 = min(Et, E)
    # L-4: |t^ - t| <= eta_t A, t = 2^E T/Z, A = 2^E A/Z
    out["L4"] = lin_le([(p2(Et - E0) * gq.Z, ti), (-p2(E - E0), T)], [(kc["eta_t"] * p2(E - E0), A)])
    # L-5 quadratic: |T^ - (D/2) W t^2| <= (D/2) W [c_q |t| A + nu A^2], (D/2) W = k0 c4 Z^2/2^(2E)
    qi, Eq = dyadic(rep["_Tq"])
    k0 = gq.D / 2 * gq.hx * gq.hy / 4 * p2(2 * E) / gq.Z**2
    c4 = gq.c4
    out["L5_quadratic"] = lin_le([(p2(Eq), qi), (-k0, c4 * T * T)], [(k0 * kc["c_q"], c4 * np.abs(T) * A), (k0 * kc["nu"], c4 * A * A)])
    if with_load:
        li, El = dyadic(rep["_load"])
        qhh = gq.q * gq.hx * gq.hy
        out["L5_load"] = lin_le([(p2(El), li), (-qhh * p2(E), I)], [(kc["c_l"] * abs(qhh) * p2(E), np.abs(I))])
    xi, Ex = dyadic(rep["_terms"])
    vi, Ev = dyadic(np.array([value]))
    gL = kc["gamma_L"] if with_load else kc["gamma_LQ"]
    out["L3"] = lin_le([(p2(Ev), vi), (-p2(Ex), np.array([isum(xi)], dtype=object))], [(gL * p2(Ex), np.array([isum(np.abs(xi))], dtype=object))])
    return out


def _sub_fields(a: tuple, b: tuple) -> tuple:
    (Ia, Ea), (Ib, Eb) = a, b
    E0 = min(Ea, Eb)
    return at_exp(Ia, Ea, E0) - at_exp(Ib, Eb, E0), E0


def control_path(gq: GridQ, kc: dict, g: en.Grid, fm: dict, reps: dict, sums: dict, ints: dict, bnd: dict, E_s: Fraction,
                 Pp: float, Ps: float, Qd: float) -> dict:
    """Item 9: exact parts of E-1 by the 13-point form, the identity, the bounds of each part, L-3...L-6 node by node, and own exact
    quantities against the exact functions of r5_cccc_energy / r5_cccc_d2."""
    wp, ws, dd = ints["wp"], ints["ws"], ints["dh"]
    tp, ts = ints["tp"], ints["ts"]
    e = _sub_fields(ws, ts)
    ep = _sub_fields(wp, tp)
    d = _sub_fields(wp, ws)
    dl = _sub_fields(dd, d)
    epe = _sub_fields(ep, e)

    def load(x):
        return gq.q * gq.hx * gq.hy * p2(x[1]) * isum(x[0])

    def pi13(x):
        return a13(gq, x, x) / 2 - load(x)

    E1 = a13(gq, e, d) + a13(gq, ts, epe) - load(epe)
    E2 = (frac_of(Pp) - pi13(wp)) - (frac_of(Ps) - pi13(ws))
    E3 = -a13(gq, dd, dl) + a13(gq, dl, dl) / 2
    E4 = -(frac_of(Qd) - a13(gq, dd, dd) / 2)
    out = {"E1": E1, "E2": E2, "E3": E3, "E4": E4, "identity_E1_exact": E1 + E2 + E3 + E4 == E_s}
    B1, B2, B3, B4 = bnd["B1"], bnd["B2"], bnd["B3"], bnd["B4"]
    out.update({"E1_le_bound": abs(E1) <= B1, "E2_le_bound": abs(E2) <= B2, "E3_le_bound": abs(E3) <= B3, "E4_le_bound": abs(E4) <= B4,
                "E1_over_bound": float(abs(E1) / B1) if B1 else None, "E2_over_bound": float(abs(E2) / B2) if B2 else None,
                "E3_over_bound": float(abs(E3) / B3) if B3 else None, "E4_over_bound": float(abs(E4) / B4) if B4 else None})
    out["e_a_le_eps_s"] = a13(gq, e, e) <= bnd["eps_s"] ** 2
    out["e_prev_a_le_eps_prev"] = a13(gq, ep, ep) <= bnd["eps_p"] ** 2
    out["dhat_a2_two_routes"] = a13(gq, dd, dd) == sums["d"]["a2"]
    # L-6 (both inequalities) on the three fields
    L6 = True
    for key in ("p", "s", "d"):
        f = sums[key]
        L6 &= f["S2"] <= gq.D / 2 * gq.Lam**2 * f["h2"] and f["S2"] <= kc["Lambda_bar"] ** 2 * f["Sq"]
    out["L6"] = L6
    # node by node L-3, L-4, L-5
    nodes = {}
    for key, fld, with_load, val in (("p", wp, True, Pp), ("s", ws, True, Ps), ("d", dd, False, Qd)):
        nodes[key] = node_checks(gq, kc, reps[key], sums[key], fld[0], fld[1], val, with_load)
    out["nodes_ok"] = all(v[0] for n in nodes.values() for v in n.values())
    out["nodes_max_ratio"] = {f"{k}_{name}": v[1] for k, n in nodes.items() for name, v in n.items()}
    # own exact quantities = the exact functions of r5_cccc_energy / r5_cccc_d2
    xs = {}
    for key, w in (("p", fm["wh_prev"]), ("s", fm["wh_s"]), ("d", fm["dh"])):
        sq, sl = en.majorant_S(g, w)
        xs[key] = sq == sums[key]["Sq"] and sl == sums[key]["Sl"]
    xs["d_norms"] = en.norm_a_sq_exact(g, fm["dh"]) == sums["d"]["a2"] and en.norm_h_sq_exact(g, fm["dh"]) == sums["d"]["h2"]
    prob = d2.D2Problem(g.lx, g.ly, g.nx, g.ny, D=g.D, q=g.q)
    r = d2.residual_exact(prob, np.asarray(fm["wh_s"], dtype=float).reshape(-1))
    xs["residual"] = Fraction(g.D) ** 2 * gq.hx * gq.hy * sum((x * x for x in r), Fraction(0)) == bnd["res_h2"]
    out["own_equals_EN_exact"] = xs
    out["ok"] = bool(out["identity_E1_exact"] and out["E1_le_bound"] and out["E2_le_bound"] and out["E3_le_bound"]
                     and out["E4_le_bound"] and out["e_a_le_eps_s"] and out["e_prev_a_le_eps_prev"] and out["dhat_a2_two_routes"]
                     and L6 and out["nodes_ok"] and all(xs.values()))
    return out


# ---------------------------------------------------------------------------------------------------------------------------------
# one half-step
# ---------------------------------------------------------------------------------------------------------------------------------

def _bind(hist: dict | None, key: str, value) -> bool | str:
    if hist is None or key not in hist or hist[key] in ("", None):
        return "NOT_AVAILABLE"
    if isinstance(hist[key], float):  # CI profile: the EQ functions' own value in this process
        return float(value).hex() == hist[key].hex()
    try:
        h = _eq()._ga().read_double(hist[key])  # rows.csv: repr-printed double, exact round trip
    except ValueError:
        return False
    return float(value).hex() == h.hex()


def analyse_row(spec: dict, steps: list[dict], i: int, rid: str, hist: dict | None, control: bool, grid_ref: tuple | None) -> dict:
    eq = _eq()
    gi = eq._gi()
    st = steps[i]
    lx, nx, ly, ny = spec["lx"], spec["nx"], spec["ly"], spec["ny"]
    direction, N, s = st["direction"], st["given"].shape[1], st["s"]
    g = en.Grid(lx, nx, ly, ny)
    out = {"id": rid, "group": spec["group"], "grid": rid.split("|")[1], "N": N, "first": spec["first"], "s": s,
           "direction": direction, "control": control, "marking": RESEARCH if N >= 2 else "N = 1 (OBSERVATION)"}
    unres, fails = [], []
    t0 = time.perf_counter()
    pb = probe_mode()
    out["main_thread"] = pb["main_thread"]
    out["mode_before_ok"] = pb["ok"]
    out["controlfp_before"] = pb["controlfp"].get("word")
    if not pb["ok"]:
        unres.append("U-2: mode probe before the row: " + "; ".join(pb["fails"]))
    # --- 3. the EQ path with the EQ functions, in the EQ order
    out["given_consistent"] = st["given_consistent"]
    if st["given_consistent"] is not True:
        unres.append("input: given set inconsistent with the stored arrays")
    p_in, comp_in, _ = eq.bc_input(spec, steps, i)
    rec = bc.run_halfstep(p_in, qr=N >= 2)
    if rec.solved is None:
        out["status"] = f"{UNRESOLVED}: given set exactly dependent (prototype)"
        return out
    solved = rec.solved
    Cp = comp_in @ rec.R.T if rec.R is not None else np.asarray(comp_in, dtype=float)
    if N >= 2 and s == 1:
        out["q3_state_reproduced_bitwise"] = "NOT_APPLICABLE (s = 1, N >= 2: QR basis U~ != T_N)"
    else:
        out["q3_state_reproduced_bitwise"] = bool(np.array_equal(solved.given, st["given"])
                                                  and (s == 1 or np.array_equal(Cp, st["companion"])))
        if out["q3_state_reproduced_bitwise"] is not True:
            unres.append("input: B-C state does not reproduce the Q3 state")
    coeffs = hs.exact_coefficients(solved)
    bcres = eq.solve_c3plus(solved, coeffs)
    sol = bcres.pop("_sol")
    out["bc_status"] = bcres.get("bc_status")
    if sol.s is None:
        out["status"] = f"{UNRESOLVED}: B-C produced no solution ({sol.status})"
        return out
    m = solved.n_s - 1
    V = sol.s.reshape(N, m).T
    Xp, Yp = eq.orient(direction, solved.given, Cp)
    Xs, Ys = eq.orient(direction, solved.given, V)
    gamma_s = gi.gamma_b1(lx, nx, ly, ny, direction, solved.given, Cp, V)
    gd, gu = gi.directed(gamma_s)
    G_up = gi.gamma_up(gd, gu)
    wh_prev, wh_s = Xp @ Yp.T, Xs @ Ys.T
    dh = wh_prev - wh_s
    Pp, Ps, Qd = en.pi_double(g, wh_prev), en.pi_double(g, wh_s), en.q_double(g, dh)
    mt = eq.model_terms(g, Xp, Yp, Xs, Ys)
    R = mt.pop("_R_Sigma")
    eps_s_en, eps_p_en, da_en, Bw_en = mt.pop("_Rf_parts")
    R_up = en.up(R)
    vals = {"P_hat_prev": Pp, "P_hat_s": Ps, "Q_hat": Qd, "R_Sigma_up": R_up, "gamma_down": gd, "gamma_up": gu, "Gamma_up": G_up}
    vals.update({k: mt[k] for k in EXT_KEYS if k in mt})
    out.update({k: vals[k] for k in BIND_KEYS})
    # --- binding to the historical rows.csv
    for k in BIND_KEYS + EXT_KEYS:
        out[f"bind_{k}"] = _bind(hist, k, vals[k])
    req = [out[f"bind_{k}"] for k in BIND_KEYS]
    out["binding_required_4"] = all(v is True for v in req)
    if not out["binding_required_4"]:
        unres.append("binding: " + ", ".join(k for k in BIND_KEYS if out[f"bind_{k}"] is not True) + " differ from rows.csv")
    out["binding_extended_all"] = all(out[f"bind_{k}"] is True for k in EXT_KEYS)
    # --- U-1 on both fields (the same arrays), determinism of the product
    out["product_deterministic"] = same_bits(Xp @ Yp.T, wh_prev) and same_bits(Xs @ Ys.T, wh_s)
    if not out["product_deterministic"]:
        unres.append("U-1: a second product X Y^T differs bitwise")
    fp = formation_check(Xp, Yp, wh_prev)
    fs_ = formation_check(Xs, Ys, wh_s)
    for tag, f in (("prev", fp), ("s", fs_)):
        for k, v in f.items():
            if not k.startswith("_"):
                out[f"U1_{tag}_{k}"] = v
        if not f["holds"]:
            unres.append(f"U-1: |e| <= gamma_N |X||Y|^T violated on the {'previous' if tag == 'prev' else 'current'} field")
    # --- U-2 (b)-(d): replication with operation checks, the subtraction d^
    log = OpLog()
    dh2 = wh_prev - wh_s
    log.add("d^ = w^_p - w^_s", wh_prev, wh_s, dh2, sub=True)
    reps = {"p": replicate_energy(g, wh_prev, log, "P^_prev", True, keep=control),
            "s": replicate_energy(g, wh_s, log, "P^_s", True, keep=control),
            "d": replicate_energy(g, dh, log, "Q^", False, keep=control)}
    rep_ok = (same_bits(dh2, dh) and all(r["t_bitwise_module"] and r["quad_bitwise_module"] and r["value_bitwise_module"]
                                         for r in reps.values())
              and reps["p"]["value"].hex() == Pp.hex() and reps["s"]["value"].hex() == Ps.hex() and reps["d"]["value"].hex() == Qd.hex())
    out["U2_replication_bitwise"] = rep_ok
    ld = log.as_dict()
    out["U2_ops"] = json.dumps(ld["ops"], sort_keys=True)
    out["U2_violations"] = "; ".join(ld["violations"])
    out["U2_cancellations"] = ld["cancellations"]
    out["U2_subnormal_results_add"] = ld["subnormal_results_add"]
    out["U2_subnormal_operands"] = ld["subnormal_operands"]
    out["U2_subnormal_field_entries"] = sum(r["subnormal_inputs"] for r in reps.values())
    out["U2_pairwise_levels"] = f"{reps['p']['levels']}/{reps['s']['levels']}/{reps['d']['levels']}"
    if not rep_ok:
        unres.append("U-2: replication is not bitwise the module's computation")
    if ld["violations"]:
        unres.append("U-2: " + "; ".join(ld["violations"]))
    # --- U-3 of the grid, in this process
    kc = grid_constants(lx, nx, ly, ny, g.D, g.q, N)
    out["U3_grid_bits_equal_main"] = grid_ref is None or grid_bits(kc) == grid_ref
    out["U3_grid_ok"] = grid_ok(kc)
    out["U3_h_equals_replication"] = all(float(kc[k]).hex() == float(reps["p"][r]).hex() for k, r in
                                         (("h_dx", "hx"), ("h_dy", "hy"), ("c_x", "cx"), ("c_y", "cy")))
    out["U3_levels_equal_L"] = reps["p"]["levels"] == kc["L"] and reps["s"]["levels"] == kc["L"] and reps["d"]["levels"] == kc["L_Q"]
    if not (out["U3_grid_bits_equal_main"] and out["U3_grid_ok"] and out["U3_h_equals_replication"] and out["U3_levels_equal_L"]):
        unres.append("U-3: grid constants not established in this process")
    Kstar = kc["K_star"]
    out["K_star_minus_1"] = float(Kstar - 1)
    # --- N-7: zero previous-field formation bound only for an exact zero product
    hist_eps0 = hist is not None and hist.get("eps_a_C_prev_up") not in (None, "") and float(hist["eps_a_C_prev_up"]) == 0.0
    if hist_eps0 or eps_p_en == 0:
        exact0 = fp["exact_product_zero"] and fp["computed_field_zero"]
        out["N7_applicable"] = True
        out["N7_hist_eps_prev_zero"] = hist_eps0
        out["N7_EN_eps_prev_zero"] = eps_p_en == 0
        out["N7_companion_exact_zero"] = bool(np.all(is_zero(Cp)))
        out["N7_given_exact_zero"] = bool(np.all(is_zero(solved.given)))
        out["N7_exact_product_zero"] = fp["exact_product_zero"]
        out["N7_ok"] = bool(exact0 and (out["N7_companion_exact_zero"] or out["N7_given_exact_zero"]) and eps_p_en == 0)
        if not out["N7_ok"]:
            unres.append("N-7: eps_a^C(s - 1) = 0 without an exact zero product")
    else:
        out["N7_applicable"] = False
    if s == 1:  # R-AC branch of EQ (AU 461): prev factors of the stored R-AC half-step
        Xa, Ya = st["prev"]
        out["N7_RAC_prev_exact_zero"] = bool(np.all(is_zero(Xa)) or np.all(is_zero(Ya)))
    # --- 8. B_E (own exact implementation)
    gq = GridQ(lx, nx, ly, ny, g.D, g.q)
    be = b_e(gq, kc, N, fp, fs_, dh, keep=control)
    sums, bnd, BE = be["sums"], be["bnd"], be["BE"]
    eps_s, eps_p, dh_a, beta, Bw = be["eps_s"], be["eps_p"], be["dh_a"], be["beta"], be["Bw"]
    out["B_omega_denominator_positive"] = be["denominator_positive"]
    out["PropD_two_routes_ws"] = be["PropD"]
    E_s = frac_of(Pp) - frac_of(Ps) - frac_of(Qd) + gam_exact(gamma_s)
    out["E_s_abs_up"] = directed(abs(E_s))[1]
    if BE is None:
        unres.append("B_E: denominator D^(1/2) mu_low not positive")
    else:
        be_u = directed(BE)[1]
        out["B_E_up"] = be_u
        out["B_E_finite"] = math.isfinite(be_u) and math.isfinite(R_up)
        if not out["B_E_finite"]:
            unres.append("B_E or R_Sigma not finite in binary64")
        out["E_s_le_B_E"] = abs(E_s) <= BE
        out["B_E_le_Kstar_R"] = BE <= Kstar * R
        out["E_s_over_B_E"] = float(abs(E_s) / BE) if BE else None
        out["B_E_over_R"] = float(BE / R) if R else None
        out["B_E_over_Kstar_R"] = float(BE / (Kstar * R)) if R else None
        out["m_star_up"] = directed(abs(E_s) / R)[1] if R else None
        out["m_star_le_Kstar"] = abs(E_s) <= Kstar * R
        for k in ("B1", "B2", "B3", "B4"):
            out[f"{k}_over_B_E"] = float(bnd[k] / BE) if BE else None
    # cross-checks against r5_cccc_energy (same radicands, finer roots; formulas below the upward-rounded EN terms)
    L = kc["L"]
    RCp = U * (L + C_R) * (sums["p"]["Sq"] + sums["p"]["Sl"])
    RCs = U * (L + C_R) * (sums["s"]["Sq"] + sums["s"]["Sl"])
    RQ = U * (L + C_R) * sums["d"]["Sq"] + beta * dh_a
    xc = {"R_C_prev": RCp <= Fraction(mt["R_C_prev_up"]), "R_C_s": RCs <= Fraction(mt["R_C_s_up"]),
          "R_Q": RQ <= Fraction(mt["R_Q_up"]), "eps_s": eps_s <= eps_s_en, "eps_prev": eps_p <= eps_p_en, "d_hat_a": dh_a <= da_en,
          "B_omega": Bw is not None and Bw <= Bw_en}
    out["crosscheck_EN_all"] = all(xc.values())
    out["crosscheck_EN_failed"] = ",".join(k for k, v in xc.items() if not v)
    rel = []
    for mine, theirs in ((RCp, Fraction(mt["R_C_prev_up"])), (RCs, Fraction(mt["R_C_s_up"])), (RQ, Fraction(mt["R_Q_up"])),
                         (eps_s, eps_s_en), (eps_p, eps_p_en), (dh_a, da_en), (Bw, Bw_en)):
        if mine is not None and theirs:
            rel.append(float(abs(theirs - mine) / theirs))
    out["crosscheck_EN_max_rel_gap"] = max(rel, default=0.0)
    if not out["PropD_two_routes_ws"]:
        fails.append("Prop. D: two routes of ||w^_s||_a^2 differ")
    if not out["crosscheck_EN_all"]:
        fails.append("own B_E ingredients exceed the EN bounds: " + out["crosscheck_EN_failed"])
    # --- 9. separate mathematical path on control rows
    if control and BE is not None:
        ints = {"wp": be["wp"], "ws": be["ws"], "dh": be["dd"], "tp": (fp["_Wt"], fp["_EWt"]), "ts": (fs_["_Wt"], fs_["_EWt"])}
        fm = {"wh_prev": wh_prev, "wh_s": wh_s, "dh": dh}
        cp = control_path(gq, kc, g, fm, reps, sums, ints, bnd, E_s, Pp, Ps, Qd)
        out["ctl_ok"] = cp["ok"]
        for k in ("identity_E1_exact", "E1_le_bound", "E2_le_bound", "E3_le_bound", "E4_le_bound", "E1_over_bound", "E2_over_bound",
                  "E3_over_bound", "E4_over_bound", "e_a_le_eps_s", "e_prev_a_le_eps_prev", "dhat_a2_two_routes", "L6", "nodes_ok"):
            out[f"ctl_{k}"] = cp[k]
        out["ctl_nodes_max_ratio"] = json.dumps(cp["nodes_max_ratio"], sort_keys=True)
        out["ctl_own_equals_EN_exact"] = all(cp["own_equals_EN_exact"].values())
        if not cp["ok"]:
            fails.append("control path: " + ", ".join(k for k in ("identity_E1_exact", "E1_le_bound", "E2_le_bound", "E3_le_bound",
                                                                   "E4_le_bound", "e_a_le_eps_s", "e_prev_a_le_eps_prev",
                                                                   "dhat_a2_two_routes", "L6", "nodes_ok") if not cp[k])
                         + ("" if all(cp["own_equals_EN_exact"].values()) else " own != EN exact"))
    # --- mode probe after the row, same thread
    pa = probe_mode()
    out["mode_after_ok"] = pa["ok"]
    out["controlfp_after"] = pa["controlfp"].get("word")
    out["same_thread"] = pa["thread"] == pb["thread"] and pb["main_thread"]
    if not pa["ok"]:
        unres.append("U-2: mode probe after the row: " + "; ".join(pa["fails"]))
    if not out["same_thread"]:
        unres.append("U-2: the probes and the computation are not in one (main) thread")
    # --- status
    if BE is not None and not unres:
        if not out["E_s_le_B_E"]:
            fails.append("|E_s| > B_E,s")
        if not out["B_E_le_Kstar_R"]:
            fails.append("B_E,s > K* R_Sigma")
    if fails:
        out["status"] = f"{FAIL}: " + "; ".join(fails)
    elif unres:
        out["status"] = f"{UNRESOLVED}: " + "; ".join(unres)
    else:
        out["status"] = VERIFIED
    out["_seconds"] = round(time.perf_counter() - t0, 2)  # timing: meta.json, not rows.csv (rows.csv is deterministic)
    return out


def gam_exact(x) -> Fraction:
    if not isinstance(x, Fraction):
        raise TypeError("gamma must be an exact rational")
    return x


def seq_job(spec: dict, rows: list[dict], npz_path: str, hist: dict, control_ids: list[str], grid_ref: dict) -> dict:
    ga = _eq()._ga()
    t0 = time.perf_counter()
    start = {"probe": probe_mode(), "blas": blas_info(), "pid": os.getpid()}
    z = np.load(npz_path)
    steps = ga._steps_from_npz(z, spec)
    by_s = {int(r["j_s"]): r for r in rows}
    gkey = (spec["lx"], spec["nx"], spec["ly"], spec["ny"])
    out = []
    for i, st in enumerate(steps):
        rid = ga.row_id(by_s[st["s"]])
        try:
            out.append(analyse_row(spec, steps, i, rid, hist.get(rid), rid in control_ids, grid_ref.get(str(gkey))))
        except Exception as exc:  # recorded, never hidden: the row is UNRESOLVED with the reason
            out.append({"id": rid, "group": spec["group"], "grid": rid.split("|")[1], "N": spec["N"], "first": spec["first"],
                        "s": st["s"], "direction": st["direction"], "control": rid in control_ids,
                        "status": f"{UNRESOLVED}: exception {type(exc).__name__}: {exc}"})
    end = probe_mode()
    return {"rows": out, "spec": spec, "npz": Path(npz_path).name, "npz_sha256": sha256(Path(npz_path)), "steps": len(steps),
            "seconds": round(time.perf_counter() - t0, 2), "peak_mb": ga._q3().peak_mb(), "start": start,
            "end_probe_ok": end["ok"], "end_thread": end["thread"]}


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
    q = lambda f: v[min(len(v) - 1, int(math.ceil(f * len(v))) - 1)]  # noqa: E731 - nearest rank (as Q3)
    return {"count": len(v), "min": v[0], "median": statistics.median(v), "q99": q(0.99), "max": v[-1]}


def _count(rows, key, val=True) -> str:
    return f"{sum(1 for r in rows if r.get(key) is val or r.get(key) == val)}/{sum(1 for r in rows if key in r)}"


def _status(r) -> str:
    return str(r.get("status", "")).split(":")[0]


def summarize(rows: list[dict], grids: dict, hist_rows: list[dict]) -> dict:
    out = {"rows": len(rows), "status": {k: sum(1 for r in rows if _status(r) == k) for k in (VERIFIED, UNRESOLVED, FAIL)},
           "reasons": {}}
    for r in rows:
        if _status(r) != VERIFIED:
            out["reasons"][r["status"]] = out["reasons"].get(r["status"], 0) + 1
    for k in ("given_consistent", "binding_required_4", "binding_extended_all", "product_deterministic", "U1_prev_holds",
              "U1_s_holds", "U2_replication_bitwise", "mode_before_ok", "mode_after_ok", "same_thread", "main_thread",
              "U3_grid_bits_equal_main", "U3_grid_ok", "U3_h_equals_replication", "U3_levels_equal_L", "B_omega_denominator_positive",
              "B_E_finite", "E_s_le_B_E", "B_E_le_Kstar_R", "m_star_le_Kstar", "PropD_two_routes_ws", "crosscheck_EN_all",
              "N7_applicable", "N7_ok", "N7_RAC_prev_exact_zero", "ctl_ok", "ctl_identity_E1_exact", "ctl_E1_le_bound",
              "ctl_E2_le_bound", "ctl_E3_le_bound", "ctl_E4_le_bound", "ctl_e_a_le_eps_s", "ctl_e_prev_a_le_eps_prev",
              "ctl_dhat_a2_two_routes", "ctl_L6", "ctl_nodes_ok", "ctl_own_equals_EN_exact"):
        out[k] = _count(rows, k)
    for k in BIND_KEYS + EXT_KEYS:
        out[f"bind_{k}"] = _count(rows, f"bind_{k}")
    out["q3_state_reproduced_bitwise"] = {str(v): sum(1 for r in rows if str(r.get("q3_state_reproduced_bitwise")) == str(v))
                                          for v in sorted({str(r.get("q3_state_reproduced_bitwise")) for r in rows})}
    # U-1 by field and N
    u1 = {}
    for tag in ("prev", "s"):
        for N in (1, 2, 3):
            sub = [r for r in rows if r.get("N") == N and f"U1_{tag}_holds" in r]
            u1[f"{tag}_N{N}"] = {
                "rows": len(sub), "holds": sum(1 for r in sub if r[f"U1_{tag}_holds"] is True),
                "elements": sum(r[f"U1_{tag}_elements"] for r in sub), "violations": sum(r[f"U1_{tag}_violations"] for r in sub),
                "max_ratio": max((r[f"U1_{tag}_max_ratio"] for r in sub), default=None),
                "zero_S": sum(r[f"U1_{tag}_zero_S"] for r in sub), "exact_errors_zero": sum(r[f"U1_{tag}_exact_errors_zero"] for r in sub),
                "compensation": sum(r[f"U1_{tag}_compensation"] for r in sub),
                "max_ratio_compensation": max((r[f"U1_{tag}_max_ratio_compensation"] for r in sub
                                               if r[f"U1_{tag}_max_ratio_compensation"] is not None), default=None),
                "rows_with_zero_factor_entries": sum(1 for r in sub if r[f"U1_{tag}_zero_factor_entries"] > 0),
                "rows_factor_all_zero": sum(1 for r in sub if r[f"U1_{tag}_factor_all_zero"] is True),
                "rows_exact_products_normal": sum(1 for r in sub if r[f"U1_{tag}_exact_products_normal"] is True)}
    out["U1"] = u1
    out["U2"] = {"cancellations": stats(r.get("U2_cancellations") for r in rows),
                 "rows_with_cancellations": sum(1 for r in rows if (r.get("U2_cancellations") or 0) > 0),
                 "subnormal_results_add_total": sum(r.get("U2_subnormal_results_add") or 0 for r in rows),
                 "subnormal_operands_total": sum(r.get("U2_subnormal_operands") or 0 for r in rows),
                 "subnormal_field_entries_total": sum(r.get("U2_subnormal_field_entries") or 0 for r in rows),
                 "rows_with_violations": sum(1 for r in rows if r.get("U2_violations")),
                 "controlfp_words": sorted({str(r.get(k)) for r in rows for k in ("controlfp_before", "controlfp_after")}),
                 "ops_example": next((r.get("U2_ops") for r in rows if r.get("U2_ops")), None)}
    out["U3"] = grid_summary(grids)
    done = [r for r in rows if "E_s_le_B_E" in r]
    out["B_E"] = {"rows": len(done), "E_s_over_B_E": stats(r.get("E_s_over_B_E") for r in done),
                  "B_E_over_R": stats(r.get("B_E_over_R") for r in done), "B_E_over_Kstar_R": stats(r.get("B_E_over_Kstar_R") for r in done),
                  "m_star_up": stats(r.get("m_star_up") for r in done),
                  "crosscheck_EN_max_rel_gap": stats(r.get("crosscheck_EN_max_rel_gap") for r in done),
                  "share_B1": stats(r.get("B1_over_B_E") for r in done), "share_B2": stats(r.get("B2_over_B_E") for r in done),
                  "share_B3": stats(r.get("B3_over_B_E") for r in done), "share_B4": stats(r.get("B4_over_B_E") for r in done)}
    out["B_E_by_s"] = {str(s): {"E_s_over_B_E_max": stats(r.get("E_s_over_B_E") for r in done if r["s"] == s).get("max"),
                               "B_E_over_R": stats(r.get("B_E_over_R") for r in done if r["s"] == s)} for s in range(1, 9)}
    out["B_E_by_N"] = {str(N): {"E_s_over_B_E_max": stats(r.get("E_s_over_B_E") for r in done if r["N"] == N).get("max"),
                               "B_E_over_R": stats(r.get("B_E_over_R") for r in done if r["N"] == N)} for N in (1, 2, 3)}
    out["max_rows"] = {k: max(((r.get(k), r["id"]) for r in done if isinstance(r.get(k), float)), default=None)
                       for k in ("E_s_over_B_E", "B_E_over_R", "B_E_over_Kstar_R", "m_star_up")}
    ctl = [r for r in rows if r.get("control")]
    out["control"] = {"rows": len(ctl), "grids": sorted({r["grid"] for r in ctl}),
                      "N": sorted({r["N"] for r in ctl}), "s": sorted({r["s"] for r in ctl}),
                      "parts_over_bound_max": {k: stats(r.get(f"ctl_{k}_over_bound") for r in ctl).get("max") for k in ("E1", "E2", "E3", "E4")}}
    nm = {}
    for r in ctl:
        if r.get("ctl_nodes_max_ratio"):
            for k, v in json.loads(r["ctl_nodes_max_ratio"]).items():
                nm[k] = max(nm.get(k, 0.0), v)
    out["control"]["nodes_max_ratio"] = nm
    # known limitation: rows with K_max < K* (historical rows.csv, K* of this run)
    ks = {f"{c['lx']}|{c['nx']}|{c['ly']}|{c['ny']}": c["K_star"] for c in grids.values()}
    lim = []
    for h in hist_rows:
        key = h.get("_grid_key")
        if key in ks and h.get("BC_K_max_up") not in (None, "") and Fraction(float(h["BC_K_max_up"])) < ks[key]:
            lim.append(h)
    out["K_max_below_Kstar"] = {"rows": len(lim), "expected_RV8": KNOWN_KMAX_BELOW_KSTAR,
                                "by_s": {str(s): sum(1 for h in lim if int(h["s"]) == s) for s in range(1, 9)},
                                "EMPTY_among": sum(1 for h in lim if h.get("BC_K_class") == "EMPTY"),
                                "all_have_K_max_below_1": all(float(h["BC_K_max_up"]) < 1 for h in lim)}
    out["failures"] = [{"id": r["id"], "status": r["status"]} for r in rows if _status(r) != VERIFIED]
    return out


def grid_summary(grids: dict) -> dict:
    cs = list(grids.values())
    if not cs:
        return {}
    km = max(cs, key=lambda c: c["K_star"])
    rev = Fraction(REVIEWER_KSTAR_MINUS_1)
    return {"grids": len(cs), "pow_equals_mul": f"{sum(2 for c in cs if c['pow_equals_mul'])}/{2 * len(cs)}",
            "exact_h_and_h2": sum(1 for c in cs if c["delta_hx"] == 0 and c["delta_hy"] == 0 and c["delta_cx"] == 0 and c["delta_cy"] == 0),
            "max_abs_delta_h_over_u": float(max(max(abs(c["delta_hx"]), abs(c["delta_hy"])) for c in cs) / U),
            "max_abs_delta_c_over_u": float(max(max(abs(c["delta_cx"]), abs(c["delta_cy"])) for c in cs) / U),
            "c_q_over_u": [float(min(c["c_q"] for c in cs) / U), float(max(c["c_q"] for c in cs) / U)],
            "c_l_over_u": [float(min(c["c_l"] for c in cs) / U), float(max(c["c_l"] for c in cs) / U)],
            "K_P": [float(min(c["K_P"] for c in cs)), float(max(c["K_P"] for c in cs))],
            "max_K_Q_minus_1": float(max(c["K_Q"] for c in cs) - 1), "max_K_f_minus_1": float(max(c["K_f"] for c in cs) - 1),
            "K_star_equals_K_Q": sum(1 for c in cs if c["K_star"] == c["K_Q"]),
            "max_K_star_minus_1": float(km["K_star"] - 1), "max_K_star_minus_1_up": directed(km["K_star"] - 1)[1],
            "max_K_star_grid": f"{km['lx']}x{km['ly']}:{km['nx']}x{km['ny']}", "Lambda_bar_max": float(km["Lambda_bar"]),
            "reviewer_max_K_star_minus_1": REVIEWER_KSTAR_MINUS_1,
            "reviewer_value_matches_to_5_digits": f"{float(km['K_star'] - 1):.4e}" == f"{float(rev):.4e}",
            "all_normal_ok": all(c["normal_ok"] for c in cs), "all_denominators_ok": all(c["denominators_ok"] for c in cs),
            "L_equals_EN_log2": all(c["L_equals_EN_log2"] for c in cs), "L_equals_pairwise_depth": all(c["L_equals_pairwise_depth"] for c in cs),
            "max_L": max(c["L"] for c in cs)}


def grid_rows(grids: dict) -> list[dict]:
    out = []
    for c in grids.values():
        out.append({"grid": f"{c['lx']}x{c['ly']}:{c['nx']}x{c['ny']}", "h_dx": c["h_dx"], "h_dy": c["h_dy"], "c_x": c["c_x"],
                    "c_y": c["c_y"], "pow_equals_mul": c["pow_equals_mul"], "delta_hx_over_u": float(c["delta_hx"] / U),
                    "delta_hy_over_u": float(c["delta_hy"] / U), "delta_cx_over_u": float(c["delta_cx"] / U),
                    "delta_cy_over_u": float(c["delta_cy"] / U), "c_q_over_u": float(c["c_q"] / U), "c_l_over_u": float(c["c_l"] / U),
                    "L": c["L"], "L_Q": c["L_Q"], "Lambda_bar": float(c["Lambda_bar"]), "K_P": float(c["K_P"]),
                    "K_Q_minus_1": float(c["K_Q"] - 1), "K_f_minus_1": float(c["K_f"] - 1), "K_star_minus_1": float(c["K_star"] - 1),
                    "K_star_minus_1_up": directed(c["K_star"] - 1)[1], "normal_ok": c["normal_ok"], "denominators_ok": c["denominators_ok"]})
    return sorted(out, key=lambda r: r["grid"])


def _hist_rows(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def run_campaign(profile: str, out: Path, workers: int, arrays: Path) -> dict:
    t0 = time.time()
    head_start, dirty_start = git("rev-parse", "HEAD"), git("status", "--porcelain").splitlines()
    out.mkdir(parents=True, exist_ok=True)
    eq = _eq()
    ga = eq._ga()
    q3 = ga._q3()
    tmp = None
    main_probe_start = probe_mode()
    if profile == "ci":
        tmp = Path(tempfile.mkdtemp(prefix="r5q4ebp_"))
        rows, arrays, csv_path = eq.ci_specs_rows(tmp)
        control = {ga.row_id(r) for r in rows}
        groups = {}
        for r in rows:
            groups.setdefault(ga.seq_key(r), []).append(r)
        hist = []
        for rs in groups.values():  # reference = the EQ functions on the same CI data (binding of the CI profile)
            spec = ga._ci_spec(rs[0])
            hist += eq.seq_job(spec, rs, str(arrays / q3.seq_file(spec)), sorted(control), {}, False)["rows"]
        prov = {"csv": "generated by the Q3 seq_job (CI profile)", "sha256": sha256(csv_path),
                "reference": "EQ seq_job on the same CI data, in this process"}
        npz_expected = {}
    else:
        rows = ga.read_q3_rows(Q3_CSV)
        hist = _hist_rows(EQ_ROWS)
        meta_ga = json.loads(GA_META.read_text(encoding="utf-8"))
        meta_eq = json.loads(EQ_META.read_text(encoding="utf-8"))
        npz_expected = {e["file"]: e["sha256"] for e in meta_ga["npz"]}
        npz_eq = {e["file"]: e["sha256"] for e in meta_eq["npz"]}
        if npz_expected != npz_eq or len(npz_expected) != 123:
            raise SystemExit("sha256 lists of the gamma audit and EQ meta.json differ - inputs not verified, campaign stopped")
        control = {ga.row_id(r) for r in rows if int(r["nx"]) * int(r["ny"]) <= CONTROL_MAX_NODES}
        prov = {"q3_csv": ga.provenance(Q3_CSV), "eq_rows_sha256": sha256(EQ_ROWS), "eq_meta_sha256": sha256(EQ_META),
                "gamma_audit_meta_sha256": sha256(GA_META), "eq_rows": len(hist), "eq_git_commit": meta_eq.get("git_commit")}
    hist_by_id = {h["id"]: h for h in hist}
    groups = {}
    for r in rows:
        groups.setdefault(ga.seq_key(r), []).append(r)
    if profile == "full":
        ids = {ga.row_id(r) for r in rows}
        if ids != set(hist_by_id) or len(ids) != 984:
            raise SystemExit("the sample of the Q3 CSV and the ids of the EQ rows.csv differ - campaign stopped")
    # U-3 in the main process (36 grids)
    grids = {}
    specs = []
    for rs in groups.values():
        spec = ga.spec_of(rs[0]) if profile == "full" else ga._ci_spec(rs[0])
        specs.append((spec, rs))
        key = (spec["lx"], spec["nx"], spec["ly"], spec["ny"])
        if str(key) not in grids:
            grids[str(key)] = grid_constants(*key)
    for h in hist:
        h["_grid_key"] = None
    for spec, rs in specs:
        k = f"{spec['lx']}|{spec['nx']}|{spec['ly']}|{spec['ny']}"
        for r in rs:
            if ga.row_id(r) in hist_by_id:
                hist_by_id[ga.row_id(r)]["_grid_key"] = k
    grid_ref = {k: grid_bits(c) for k, c in grids.items()}
    jobs, npz_check = [], []
    for spec, rs in specs:
        npz = arrays / q3.seq_file(spec)
        if profile == "full":  # sha256 of every NPZ before any calculation
            got = sha256(npz)
            npz_check.append({"file": npz.name, "sha256": got, "equals_meta": npz_expected.get(npz.name) == got})
            if npz_expected.get(npz.name) != got:
                raise SystemExit(f"NPZ {npz.name}: sha256 differs from the gamma audit / EQ meta - inputs not verified, campaign stopped")
        jobs.append((spec, rs, str(npz), {ga.row_id(r): hist_by_id.get(ga.row_id(r)) for r in rs}, sorted(control), grid_ref))
    jobs.sort(key=lambda j: -(j[0]["nx"] * j[0]["ny"] * j[0]["N"]))  # largest first
    t_calc = time.time()
    if workers > 1:
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(workers) as pool:
            results = pool.starmap(seq_job, jobs, chunksize=1)
    else:
        results = [seq_job(*j) for j in jobs]
    rows_out = [r for res in results for r in res["rows"]]
    rows_out.sort(key=lambda r: (r["group"], r["grid"], r["N"], r["first"], r["s"]))
    write_csv(out / "rows.csv", rows_out)
    summ = summarize(rows_out, grids, hist)
    summ["sample_rows_expected"] = len(rows)
    summ["grids_table"] = grid_rows(grids)
    summ["workers_start"] = {"probe_ok": sum(1 for r in results if r["start"]["probe"]["ok"]), "jobs": len(results),
                             "end_probe_ok": sum(1 for r in results if r["end_probe_ok"]),
                             "blas_threads": sorted({str(r["start"]["blas"].get("threads")) for r in results}),
                             "blas_core": sorted({str(r["start"]["blas"].get("core")) for r in results}),
                             "controlfp": sorted({str(r["start"]["probe"]["controlfp"].get("word")) for r in results}),
                             "pids": len({r["start"]["pid"] for r in results})}
    st = summ["status"]
    if st[FAIL] or (profile == "full" and not all(c["equals_meta"] for c in npz_check)):
        verdict = "BLOCKED"
    elif st[UNRESOLVED]:
        verdict = "PARTIAL"
    else:
        verdict = "QA_READY_FOR_REVIEW" if st[VERIFIED] == len(rows) else "PARTIAL"
    summ["verdict_by_protocol"] = verdict
    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summ, f, indent=1, ensure_ascii=False, default=_json)
        f.write("\n")
    meta = {
        "study": "R5-V Q4 priority 1 (D-053.4): arithmetic premises of |E_s| <= B_E,s <= K* R_Sigma on the EQ rows and B_E; QA research "
                 "tool; K_E, tolerances and the test form not assigned",
        "profile": profile, "git_commit_start": head_start, "git_status_start": dirty_start, "git_commit_end": git("rev-parse", "HEAD"),
        "python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__, "platform": platform.platform(),
        "machine": platform.machine(), "processor": platform.processor(), "blas_main": blas_info(),
        "threads_env": {k: os.environ.get(k) for k in THREAD_VARS}, "workers": workers,
        "mode_probe_main_start": main_probe_start, "mode_probe_main_end": probe_mode(),
        "data": prov, "q3_arrays_dir": str(arrays) if profile == "full" else "temporary (CI)", "q3_arrays_used_read_only": True,
        "npz_checked_before_calculation": npz_check, "npz": [{"file": r["npz"], "sha256": r["npz_sha256"]} for r in results],
        "control_rule": f"n_x n_y <= {CONTROL_MAX_NODES}", "control_rows": len(control), "sqrt_bits": SQRT_BITS,
        "compensation_bits": COMPENSATION_BITS,
        "jobs": [{"grid": f"{r['spec']['nx']}x{r['spec']['ny']}", "N": r["spec"]["N"], "first": r["spec"]["first"],
                  "seconds": r["seconds"], "row_seconds": [x.get("_seconds") for x in r["rows"]], "peak_mb": r["peak_mb"],
                  "pid": r["start"]["pid"],
                  "blas_threads": r["start"]["blas"].get("threads"), "probe_start_ok": r["start"]["probe"]["ok"],
                  "probe_end_ok": r["end_probe_ok"]} for r in results],
        "resource_rule_PL_6_3": {"peak_job_mb": max((r["peak_mb"] or 0) for r in results),
                                 "limit_mb_50pct_ram": (q3.total_ram_mb() or 0) / 2,
                                 "longest_job_s": max(r["seconds"] for r in results)},
        "seconds_calculation": round(time.time() - t_calc, 1), "seconds_total": round(time.time() - t0, 1),
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
    out = a.out or (OUT if a.profile == "full" else ROOT / "scratch" / "r5_cccc_q4_energy_bound_ci")
    res = run_campaign(a.profile, out, a.workers if a.profile == "full" else 1, a.arrays)
    s = res["summary"]
    print(json.dumps({"rows": s["rows"], "status": s["status"], "verdict": s["verdict_by_protocol"],
                      "max_K_star_minus_1": s["U3"].get("max_K_star_minus_1")}, default=_json))
    return 0 if s["status"][VERIFIED] == s["rows"] else 1


if __name__ == "__main__":
    sys.exit(main())
