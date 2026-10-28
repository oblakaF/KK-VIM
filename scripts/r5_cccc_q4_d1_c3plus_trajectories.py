"""R5-V Q4, D-054, OP-8: QA research tool - OWN half-step sequences of the research B-C with the C-3+ assembly of C.

QA research tool (owner's permission OP-8, D1 protocol docs/R5_CCCC_Q4_D1_PROTOCOL.md 3.1, 12, item 2), NOT the working CCCC solver: no
import of the working ``kkvim``; the prototype B-C (src/kkvim_reference/r5_cccc_bc.py), the half-step module, the Q3 sequence module and
the C-3+ functions of scripts/r5_cccc_q4_c_assembly_e4_audit.py (PR #40) are used read-only, none of them is changed. The tool does NOT
run the D1 campaign (blocks A, B, C, D, S; D1_START = BLOCKED), has no energy classifier, assigns no threshold (K_E, tol_solve,
eta_max^C, TS-1, K, c_d, c_rank, tau_set, cond(S), S-2 criteria) and changes no discretisation. Its outputs are tool data
(OBSERVATION), not D1 data.

Trajectory (fixed here; the order is the approved one, nothing of it is new):
- plate l_x x l_y, grids n_x, n_y; start set T_N = span{sin^2(i pi x/a)} in the sin^2 basis (hs.start_set; D-049.3, item 1); first
  half-step by y with given functions of x, then Y, X, Y, X, ... (D-049.3, item 2); s = 1..s_max (s_max <= 8), no stop criterion;
- half-step s receives (given, companion): the given set G of the fixed direction and the other factor C of the previous field
  w^(s-1) = G C^T; s = 1: G = T_N, C = 0 (w^(0) = 0 exactly, H-7);
- s >= 2: (given, companion) are the arrays ``next_given``, ``next_companion`` of half-step s - 1 OF THIS TRAJECTORY, i.e. formed from
  the B-C solution of s - 1 (checked bitwise per step). No stored R-AC set of Q3 (or any other R-AC / R-H solution) enters the
  trajectory; R-H and the exact system appear only in the separate small-grid check ``independent_check``.

Order of one half-step (NS 5.1; D-049.2, item 3; the same functions as ``r5_cccc_bc.run_halfstep``, but the solve uses C-3+):
1. rank diagnostic of the given set: exact rank (rational, no threshold); s = 1: the proven conditions of D-049.3, item 3 on the start
   grid (independence N <= floor(n_x/2), exact formulas 2N < n_x) - statements about the EXACT T_N, applied only when the given set
   is bitwise hs.start_set(n_x, N) and the companion is 0 (``run_step`` refuses any other input at s = 1; the stored doubles of T_N
   are not claimed dependent by the theorem); s >= 2, N >= 2: the field diagnostic of w^(s-1) (rho_N, kappa_rep,
   cond(A) before QR; r5_cccc_sequence._rank_diagnostic, OBSERVATION). A PROVEN dependence (exact rank < N, or N > floor(n_x/2) at
   s = 1) stops the trajectory BEFORE QR (S-1 only for a proven dependence, D-049.5, item 2). Otherwise the numerical rank decision is
   UNRESOLVED (no approved criterion; no S-2, no K, c_rank, c_d, tau_set, no rank reduction, no regularisation) and the run goes on as
   RESEARCH_ONLY (N >= 2). QR is not a proof that the true rank is preserved (O4 UNRESOLVED, D-054.5);
2. N >= 2: QR of the given set by the prototype (``bc.orthonormalise``: sqrt(h_f) G = Q R, U~ = Q/sqrt(h_f)); companion C' = fl(C R^T)
   (w = G C^T = U~ (C R^T)^T, R4 5.4); N = 1: no QR (NS 5.1);
3. diagnostics of the prototype (values only): cond(A) before QR = cond(R)^2, cond(A) after QR, exact orthonormality defect, angle
   between span(U~) and span(G);
4. C-3+ of the SOLVED set (after QR if QR was applied): ``c3plus`` of PR #40, C = h X^T X + (2/h^3)(g_1 g_1^T + g_m g_m^T),
   X = rd(Z_3 G)/h^2. Premises P-1...P-5 (RO 0.4) are classified one by one (``c3plus_premises``; premises_c3plus() covers only a
   part of P-2/P-3) as ESTABLISHED (checked), ASSUMED (an explicitly accepted model) or UNRESOLVED (a check failed): P-1
   round-to-nearest probes and the FTZ/DAZ probes of P-3 in this thread before the assembly and after the solve (they show the mode at
   the moments of probing); P-2/P-3 on every intermediate (premises_c3plus, plus h, h^2, h^3, 2/h^3, C_L, E and the decomposition
   C = fl(C_L + E)); P-5 k u < 1, k = m_f + 13, exactly; P-4 (classical dot products in BLAS) cannot be checked by the tool: it is
   ASSUMED, never declared established. The summary is ESTABLISHED_CONDITIONAL_ON_P4 (P-1, P-2, P-3, P-5 established; P-4 assumed)
   or UNRESOLVED. The inequality of theorem C-3+ |C^_ij - C_ij| <= gamma_k sqrt(C_ii C_jj) (and the Frobenius form) is then checked
   EXACTLY for THIS assembled matrix on THIS input (the stored doubles) against the exact C - a check of one matrix, not a proof or a
   refutation of the universal theorem, which holds under all of P-1...P-5.
   A premise UNRESOLVED -> UNRESOLVED_C3PLUS_PREMISE, NO solve, the trajectory stops (no switch to the OLD C, no CONTROL);
   a violated inequality with P-1, P-2, P-3, P-5 established and P-4 assumed -> FAIL_C3PLUS_BOUND_CHECK, the trajectory stops (not an
   unconditional refutation of the theorem: P-4 is not verified);
5. B-C solve of the solved set with ONLY C replaced (``solve_with_c``; A, B, F and the modal algorithm are the prototype's): the
   UNCHANGED code object of bc.solve_bc is executed as a new function with a PRIVATE globals dictionary (types.FunctionType; a copy of
   the namespace of r5_cccc_bc taken at import) in which only the name ``bc_coefficients`` is bound to a local function of this call:
   it calls the prototype's bc_coefficients and replaces C by the C-3+ matrix. Nothing global is replaced (bc.solve_bc and
   bc.bc_coefficients keep their identity; no monkey-patch, no lock), so concurrent ordinary bc.solve_bc calls in other threads use the
   prototype's C. The local function records what the call consumed (one call, made by the code of solve_bc with the private globals,
   the solved set, A, B, F bitwise the prototype's, C bitwise the C-3+ matrix);
6. residual eta_QA (exact R-HP residual of the original operator with D4; UNRESOLVED for 0/0 at q = 0);
7. rank diagnostic of the new field w^(s) = U~ V^T: exact rank of V; N >= 2: rho_N etc. (OBSERVATION);
8. transformation for s + 1:
   - N = 1: I-4 (D-049.1, item 4; NS 4.4): c = sigma ||f||_inf, sigma = sign <f, 1> (sign of the EXACT sum of the stored doubles,
     >= 0 -> +1), f <- f/c, g <- g c; f = 0 exactly -> STOP ZERO_FIELD. The factors before and after I-4 are both stored;
   - N >= 2: next given = V (the B-C solved factor, untransformed), next companion = U~; the QR of V is step 2 of half-step s + 1.

Statuses of a half-step (no thresholds): OK; STOP_BEFORE_QR (proven dependence, S-1); STOP_ZERO_FIELD; PROTECTIVE_STOP (QR factor or
solved set exactly singular, B-C factorization failed); UNRESOLVED_C3PLUS_PREMISE; FAIL_C3PLUS_BOUND_CHECK; FAIL_TOOL (only-C
replacement not confirmed, broken input link, exception). N >= 2 is RESEARCH_ONLY / PROVISIONAL / NOT_QUALIFIED (D-049.5, D-054.5).

Journal (``write_journal``): journal.json (meta + one record per half-step) and arrays.npz (keys ``s{s}_{name}``) with the given set
and companion as received, R, the solved set, the transformed companion, the C-3+ matrix, the B-C solved factor and the next input
(next_given, next_companion), so that the input of s + 1 can be restored independently. Fields before and after QR and before and
after I-4 are recorded separately (sha256 of fl(X Y^T)). Provenance: git commit and status, git blob ids of the tool and of the
listed used modules (compared with the base d537941; a difference is recorded, it does not stop the run), the project modules loaded
in the process that are NOT in the list (provenance NOT_CHECKED), Python, NumPy, SciPy (versions only), BLAS build, platform, and the
values of the BLAS thread environment variables as found (the tool does not set them; they are not the effective BLAS thread count,
which is not verified). Bitwise reproducibility is claimed only on the same platform, BLAS build and thread settings, never across
BLAS platforms (D-025; AGENTS.md 4).

Guard against D1: the command line refuses the D1 grids and output under docs/. This is a guard of the command line only, NOT a
mechanism that prevents a D1 run: the Python API (run_trajectory, run_step) has no such guard (D1_START = BLOCKED is an owner's
decision, not enforced by this tool).

Independent check of small half-steps (``independent_check``; R-H, exact system; BR 4; D-053.4): on the SAME solved set (after QR)
the exact system (exact coefficients of the stored doubles, h = l/n exact) is solved in rational arithmetic; the Q1b certificate
lambda_low gives ||s - x*||_2 <= ||r||_2/lambda_low for any double vector s (exact residual). Checked exactly: K x* = b; the B-C
solution and the R-H solution lie within their certified bounds of x*; the pair is consistent (||s_BC - s_H|| <= b_BC + b_H). No
threshold taken from an observed difference; the relative difference to x* is an OBSERVATION.

Usage (repository root; only small non-D1 configurations, the D1 grids are refused):
    uv run python scripts/r5_cccc_q4_d1_c3plus_trajectories.py run --lx 1 --nx 8 --ly 1.25 --ny 10 --N 2 --out scratch/traj
"""

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import subprocess
import sys
import time
import types
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import numpy as np
import scipy

from kkvim_reference import r5_cccc_bc as bc
from kkvim_reference import r5_cccc_halfstep as hs
from kkvim_reference import r5_cccc_sequence as seq

# recorded as found in the environment (the tool does not set them; not the effective BLAS thread count)
THREAD_VARS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
ROOT = Path(__file__).resolve().parents[1]
C_ASSEMBLY = ROOT / "scripts" / "r5_cccc_q4_c_assembly_e4_audit.py"
BASE_COMMIT = "d537941e9fd7f5b3218dc4db919ee7f3f0ccf6fc"
# git blob ids at BASE_COMMIT of the project modules this tool uses read-only, directly or through their imports (LF content;
# ``git rev-parse d537941:<path>``); other project modules loaded in the process are reported as NOT_CHECKED (``loaded_modules_check``)
USED_MODULES = {
    "src/kkvim_reference/__init__.py": "661a7c6e7771cca2d95485b2a7e3c1affa8e7f1b",
    "src/kkvim_reference/r5_cccc_bc.py": "1303c7d301cd8e5f799813795eb8fe1c5f5ed10c",
    "src/kkvim_reference/r5_cccc_halfstep.py": "619e63926015e6214ee923b27424fca977519113",
    "src/kkvim_reference/r5_cccc_sequence.py": "a175b5071bfa376042b62815781adba3701a902b",
    "src/kkvim_reference/r5_cccc_energy.py": "a2085bebf3d3879d3ccae26118c78f26c5e5a21c",
    "src/kkvim_reference/r5_cccc_d2.py": "a910fd291fdafdb1b8a6d568dd7b4bf589409758",
    "src/kkvim_reference/r5_cccc_q3_e4e6.py": "1282a02cf7433ec7fb8d9db6673e1ee944ef345a",
    "scripts/r5_cccc_q4_c_assembly_e4_audit.py": "89b009b463324775e139d4781f60ab3665e60aa3",
}
U = Fraction(1, 2**53)
TINY = 2.0**-1022
K_EXTRA = 13  # k = m_f + 13: operation count of the C-3+ code of PR #40 (its docstring), not a new constant
EXACT_CHECK_MAX_UNKNOWNS = 64  # cost limit of the rational solve in ``independent_check`` (size of the check, not a threshold)
RH_MAX_NODES = 4096  # R-H only for (n_x - 1)(n_y - 1) <= 4096, as in BR 4

OK = "OK"
STOP_START = ("STOP_BEFORE_QR: S-1, start set T_N dependent (N > floor(n_x/2), D-049.3 item 3; proven for the exact T_N, not a "
              "statement about the stored doubles)")
STOP_DEPENDENT = "STOP_BEFORE_QR: S-1, given set exactly dependent (exact rank < N; D-049.5 item 2)"
STOP_ZERO = "STOP_ZERO_FIELD: computed factor exactly 0 (N = 1, I-4 undefined)"
PROTECTIVE = "PROTECTIVE_STOP"
UNRES_PREMISE = "UNRESOLVED_C3PLUS_PREMISE"
FAIL_THEOREM = ("FAIL_C3PLUS_BOUND_CHECK: the exact check of the C-3+ inequality for this assembled C on this input is violated, with "
                "P-1, P-2, P-3, P-5 established and P-4 assumed; not an unconditional refutation of the theorem (P-4 not verified)")
PREMISES_OK = "ESTABLISHED_CONDITIONAL_ON_P4"
PREMISES_OK_SUMMARY = "P-1, P-2, P-3, P-5 established; P-4 assumed (accepted model, not proven)"
BOUND_CHECK_SCOPE = ("exact check of the inequality of theorem C-3+ for this assembled matrix C on this input (the stored doubles); "
                     "not a proof or a refutation of the universal theorem, which holds under P-1...P-5; here P-4 is assumed")
D1_GUARD = ("the command line refuses the D1 grids and output under docs/; the Python API has no such guard - not a mechanism that "
            "prevents a D1 run (D1_START = BLOCKED is the owner's decision)")
FAIL_TOOL = "FAIL_TOOL"
UNRESOLVED = "UNRESOLVED"
RESEARCH = ("RESEARCH_ONLY / PROVISIONAL / NOT_QUALIFIED (D-049.5, D-054.5): QR without a qualified numerical rank decision; QR is not "
            "a proof of preservation of the true rank (O4 UNRESOLVED)")
DATA_STATUS = "QA tool data (OBSERVATION); NOT D1 data, not part of the D1 sample (protocol 10, 12; D1_START = BLOCKED)"

# D1 sample grids (protocol 2.2, 2.3, 3): refused by the command line only (D1_GUARD; D1_START = BLOCKED)
D1_GRIDS = frozenset([
    (50, 50), (100, 100), (200, 200), (400, 400), (48, 72), (100, 150), (200, 300), (400, 600), (48, 32), (96, 64), (198, 132),
    (396, 264), (48, 120), (100, 250), (200, 500), (400, 1000), (50, 20), (100, 40), (200, 80), (400, 160), (40, 160), (80, 320),
    (160, 640), (320, 1280), (48, 12), (100, 25), (200, 50), (400, 100),
    (72, 40), (40, 72), (61, 37), (37, 61), (150, 48), (48, 150),
    (9, 11), (11, 15), (14, 6),
    (400, 1600), (1600, 400)])


def _ca():
    """C-3+ audit of PR #40 (c3plus, premises_c3plus, check_c3plus, spd_exact); read-only use. Its replaced_c (a global
    replacement of bc.bc_coefficients) is NOT used by this tool (``solve_with_c``)."""
    if not hasattr(_ca, "mod"):
        spec = importlib.util.spec_from_file_location("r5_cccc_q4_c_assembly_e4_audit", C_ASSEMBLY)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _ca.mod = mod
    return _ca.mod


def _prototype_function(name: str):
    """bc.<name> as DEFINED in r5_cccc_bc.py; refused if the module attribute is a replacement when this tool is imported."""
    f = getattr(bc, name)
    code = getattr(f, "__code__", None)
    if not (isinstance(f, types.FunctionType) and f.__module__ == bc.__name__ and f.__qualname__ == name and code is not None
            and Path(code.co_filename).resolve() == Path(bc.__file__).resolve()):
        raise RuntimeError(f"bc.{name} is not the function defined in {bc.__file__} (replaced when this tool was imported)")
    return f


_BC_COEFFICIENTS = _prototype_function("bc_coefficients")  # the prototype's own functions, captured at import
_BC_SOLVE = _prototype_function("solve_bc")
_BC_NAMESPACE = types.MappingProxyType(dict(vars(bc)))  # read-only snapshot of the namespace of r5_cccc_bc at import
ISOLATION = ("unchanged code object of bc.solve_bc run with a private globals dictionary of the call (types.FunctionType); only "
             "the name bc_coefficients bound to a local function (prototype's A, B, F; C -> C-3+); no global replacement, no lock")


# ---------------------------------------------------------------------------------------------------------------------------------
# configuration, hashes
# ---------------------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class TrajectoryConfig:
    lx: float
    nx: int
    ly: float
    ny: int
    N: int
    D: float = 1.0
    q: float = 1.0
    s_max: int = 8

    def __post_init__(self):
        for name in ("lx", "ly", "D"):
            v = getattr(self, name)
            if not (isinstance(v, (int, float)) and math.isfinite(v) and v > 0):
                raise ValueError(f"{name} must be a finite positive number, got {v!r}")
        if not (isinstance(self.q, (int, float)) and math.isfinite(self.q)):
            raise ValueError("q must be finite")
        for name in ("nx", "ny"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int) or v < 3:
                raise ValueError(f"{name} must be an integer >= 3, got {v!r}")
        if isinstance(self.N, bool) or self.N not in (1, 2, 3):
            raise ValueError("N must be 1, 2 or 3")
        if isinstance(self.s_max, bool) or not isinstance(self.s_max, int) or not 1 <= self.s_max <= 8:
            raise ValueError("s_max must be an integer 1..8")

    @property
    def trajectory_id(self) -> str:
        return f"BC-C3P:{float(self.lx)!r}x{float(self.ly)!r}:{self.nx}x{self.ny}:N{self.N}:D{float(self.D)!r}:q{float(self.q)!r}:" \
               f"s1-{self.s_max}"


def direction_of(s: int) -> str:
    """Y, X, Y, X, ... from s = 1 (D-049.3, item 2: the first half-step is by y)."""
    return "Y" if s % 2 == 1 else "X"


def array_sha(a: np.ndarray) -> str:
    a = np.ascontiguousarray(np.asarray(a, dtype="<f8"))
    return hashlib.sha256(f"{a.shape}".encode() + a.tobytes()).hexdigest()


def _orient(direction: str, given: np.ndarray, other: np.ndarray):
    """(x-factor, y-factor) of the field given (fixed direction) x other (solved direction)."""
    return (given, other) if direction == "Y" else (other, given)


def _field(direction: str, given: np.ndarray, other: np.ndarray) -> np.ndarray:
    X, Y = _orient(direction, given, other)
    return X @ Y.T  # fl(X Y^T) on the (n_x - 1) x (n_y - 1) interior grid


def _max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b))) if a.size else 0.0


def git_blob_id(path: Path) -> str:
    """git blob id of the file content with LF line endings (independent of a CRLF checkout)."""
    data = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def used_modules_check() -> dict:
    return {rel: {"blob": git_blob_id(ROOT / rel), "base_blob": blob, "unchanged": git_blob_id(ROOT / rel) == blob}
            for rel, blob in USED_MODULES.items()}


def loaded_modules_check() -> dict:
    """Project files loaded as modules in this process that are not in USED_MODULES (and are not this tool): NOT_CHECKED."""
    out = {}
    for m in list(sys.modules.values()):
        f = getattr(m, "__file__", None)
        if not f:
            continue
        try:
            rel = Path(f).resolve().relative_to(ROOT)
        except ValueError:
            continue
        if rel.parts and rel.parts[0] not in (".venv", "venv") and rel.as_posix() not in USED_MODULES \
                and Path(f).resolve() != Path(__file__).resolve():
            out[rel.as_posix()] = "NOT_CHECKED (no base blob recorded in the tool)"
    return out


# ---------------------------------------------------------------------------------------------------------------------------------
# premises of C-3+ (RO 0.4: P-1...P-5)
# ---------------------------------------------------------------------------------------------------------------------------------

def fp_environment_probe() -> dict:
    """P-1 / P-3 probes in this thread: round to nearest (Python floats, NumPy element-wise, a matrix product np.matmul) and no
    FTZ/DAZ. A probe shows the mode at the moment of the probe; it is taken before the assembly and after the solve of every half-step."""
    u = 2.0**-53
    a = np.array([1.0, -1.0, 1.0, -1.0])
    b = np.array([1.5 * u, -1.5 * u, 0.5 * u, -0.5 * u])
    want = [1.0 + 2 * u, -1.0 - 2 * u, 1.0, -1.0]  # 1 + 0.75 ulp -> 1 + ulp, 1 + 0.25 ulp -> 1 (nearest only)
    py = [x + y for x, y in zip(a.tolist(), b.tolist())]
    mm = float((np.array([[1.0, 1.5 * u]]) @ np.array([[1.0], [1.0]]))[0, 0])
    sub = np.array([TINY, 3 * 2.0**-1074])
    return {"nearest_python": py == want, "nearest_numpy": (a + b).tolist() == want, "nearest_matmul": mm == 1.0 + 2 * u,
            "no_ftz": float((sub / 2.0)[0]) == 2.0**-1023 and TINY / 2.0 == 2.0**-1023,
            "no_daz": float((sub + sub)[1]) == 6 * 2.0**-1074 and float((sub * 1.0)[1]) != 0.0}


def _subnormal_or_nonfinite(name: str, a) -> str | None:
    a = np.asarray(a, dtype=float)
    if not np.all(np.isfinite(a)):
        return f"P-2: non-finite {name}"
    nz = np.abs(a[a != 0])
    if nz.size and float(nz.min()) < TINY:
        return f"P-3: subnormal {name}"
    return None


def c3plus_premises(solved: hs.HalfStep, parts: dict, probe: dict) -> dict:
    """P-1...P-5 for the C-3+ code of PR #40 on the solved set, item by item (module docstring, step 4). Each premise is ESTABLISHED
    (checked here), ASSUMED (P-4 only: an explicitly accepted model, never declared established) or UNRESOLVED (a check failed). The
    summary ``status`` is ESTABLISHED_CONDITIONAL_ON_P4 when P-1, P-2, P-3, P-5 are established, otherwise UNRESOLVED."""
    ca = _ca()
    out = {}
    p1 = [f"P-1: probe {k}" for k in ("nearest_python", "nearest_numpy", "nearest_matmul") if not probe[k]]
    p2, p3 = [], []
    r = ca.premises_c3plus(parts)  # finite / subnormal Y3, X, g1, gm, C; Gram and corner products
    if r is not None:
        (p2 if r.startswith("P-2") else p3).append(r)
    h = solved.l_f / solved.n_f
    h2 = h * h
    h3 = h2 * h
    for name, v in (("h", h), ("h^2", h2), ("h^3", h3), ("2/h^3", parts["s"])):
        r = _subnormal_or_nonfinite(name, v)
        if r is not None:
            (p2 if r.startswith("P-2") else p3).append(r)
        elif v == 0:
            p3.append(f"P-3: {name} = 0")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):  # the same operations as c3plus, to see the intermediates
        CL = h * (parts["X"].T @ parts["X"])
        E = parts["s"] * (np.outer(parts["g1"], parts["g1"]) + np.outer(parts["gm"], parts["gm"]))
        Csum = CL + E
    for name, v in (("C_L", CL), ("E", E)):
        r = _subnormal_or_nonfinite(name, v)
        if r is not None:
            (p2 if r.startswith("P-2") else p3).append(r)
    out["decomposition_C_eq_fl_CL_plus_E"] = bool(np.array_equal(Csum, parts["C"]))
    if not out["decomposition_C_eq_fl_CL_plus_E"]:  # the intermediates of C are not seen: neither P-2 nor P-3 is established
        p2.append("P-2: C differs from fl(C_L + E) recomputed (intermediates not seen)")
        p3.append("P-3: C differs from fl(C_L + E) recomputed (intermediates not seen)")
    if not (probe["no_ftz"] and probe["no_daz"]):
        p3.append("P-3: FTZ/DAZ probe")
    k = solved.n_f - 1 + K_EXTRA
    out["k"] = k
    p5 = [] if k * U < 1 else ["P-5: k u >= 1"]
    established = {"P-1": "ESTABLISHED (round to nearest: Python, NumPy, matmul probes, at the moments of probing)",
                   "P-2": "ESTABLISHED (no overflow: all intermediates finite, Gram and corner sums below DBL_MAX)",
                   "P-3": "ESTABLISHED (no subnormal intermediate, no underflow of Gram or corner products; FTZ/DAZ probes)",
                   "P-5": "ESTABLISHED (k u < 1, exact)"}
    failed = {"P-1": p1, "P-2": p2, "P-3": p3, "P-5": p5}
    for name, reasons in failed.items():
        out[name] = established[name] if not reasons else "UNRESOLVED: " + "; ".join(reasons)
    out["P-4"] = "ASSUMED: classical dot products in BLAS (accepted model [STAND.]; not checkable by the tool, not proven)"
    out["classes"] = {name: ("ESTABLISHED" if not failed[name] else "UNRESOLVED") for name in ("P-1", "P-2", "P-3", "P-5")}
    out["classes"]["P-4"] = "ASSUMED"
    out["reasons"] = [x for name in ("P-1", "P-2", "P-3", "P-5") for x in failed[name]]
    out["status"] = PREMISES_OK if not out["reasons"] else UNRESOLVED
    out["summary"] = PREMISES_OK_SUMMARY if not out["reasons"] else "not all of P-1, P-2, P-3, P-5 established; P-4 assumed"
    return out


# ---------------------------------------------------------------------------------------------------------------------------------
# B-C with only C replaced
# ---------------------------------------------------------------------------------------------------------------------------------

def solve_with_c(solved: hs.HalfStep, C: np.ndarray):
    """bc.solve_bc on the solved set with ONLY C replaced and NO global replacement (module docstring, step 5). The unchanged code
    object of solve_bc runs as a new function whose globals are a private copy of the r5_cccc_bc namespace (taken at import) in which
    only ``bc_coefficients`` is the local function of this call. Nothing shared is written: bc.solve_bc, bc.bc_coefficients and the
    module namespace keep their identity, other threads (and nested calls) are not affected. The local function records what the call
    consumed."""
    C_new = np.array(C, dtype=float, copy=True)
    A0, B0, C0, F0 = _BC_COEFFICIENTS(solved)
    namespace = dict(_BC_NAMESPACE)
    calls = []

    def bc_coefficients_c3plus(p):
        A, B, _, F = _BC_COEFFICIENTS(p)
        caller = sys._getframe(1)
        calls.append({"p": p, "A": A, "B": B, "C": C_new, "F": F, "caller_is_solve_bc_code": caller.f_code is _BC_SOLVE.__code__,
                      "caller_globals_private": caller.f_globals is namespace})
        return A, B, C_new, F

    namespace["bc_coefficients"] = bc_coefficients_c3plus
    isolated = types.FunctionType(_BC_SOLVE.__code__, namespace, _BC_SOLVE.__name__, _BC_SOLVE.__defaults__, _BC_SOLVE.__closure__)
    isolated.__kwdefaults__ = _BC_SOLVE.__kwdefaults__
    sol = isolated(solved)
    private_only_c = bool(namespace.keys() == _BC_NAMESPACE.keys()
                          and all(namespace[k] is _BC_NAMESPACE[k] for k in namespace if k != "bc_coefficients"))
    if len(calls) == 1:
        c = calls[0]
        only_c = bool(c["p"] is solved and c["caller_is_solve_bc_code"] and c["caller_globals_private"]
                      and np.array_equal(c["A"], A0) and np.array_equal(c["B"], B0) and np.array_equal(c["F"], F0)
                      and np.array_equal(c["C"], C))
    else:
        only_c = False
    same_code = isolated.__code__ is _BC_SOLVE.__code__
    info = {"only_C_replaced": bool(only_c and private_only_c and same_code), "bc_coefficients_calls": len(calls),
            "solve_isolation": ISOLATION, "solve_bc_code_is_prototype": same_code,
            "private_globals_differ_only_in_bc_coefficients": private_only_c,
            "A_sha": array_sha(A0), "B_sha": array_sha(B0), "F_sha": array_sha(F0), "C_old_sha": array_sha(C0),
            "C_old_equals_C3plus_bitwise": bool(np.array_equal(C0, C))}
    return sol, info


# ---------------------------------------------------------------------------------------------------------------------------------
# one half-step
# ---------------------------------------------------------------------------------------------------------------------------------

@dataclass
class StepResult:
    record: dict
    arrays: dict
    next_given: np.ndarray | None = None
    next_companion: np.ndarray | None = None


def _stop(rec: dict, arrays: dict, status: str) -> StepResult:
    rec["status"] = status
    return StepResult(rec, arrays)


def run_step(cfg: TrajectoryConfig, s: int, given: np.ndarray, companion: np.ndarray) -> StepResult:
    """Half-step s of the trajectory from (given, companion) (module docstring, steps 1-8)."""
    ca = _ca()
    direction = direction_of(s)
    given = np.array(given, dtype=float, copy=True)
    companion = np.array(companion, dtype=float, copy=True)
    if given.ndim == 1:
        given = given[:, None]
    p = hs.HalfStep(cfg.lx, cfg.nx, cfg.ly, cfg.ny, direction, given, D=cfg.D, q=cfg.q)
    N, n_f, n_s = p.N, p.n_f, p.n_s
    if N != cfg.N or companion.shape != (n_s - 1, N) or not np.all(np.isfinite(companion)):
        raise ValueError(f"companion must be a finite ({n_s - 1}, {N}) array and N = {cfg.N}")
    h_f, h_s = p.l_f / n_f, p.l_s / n_s
    rec = {"trajectory_id": cfg.trajectory_id, "s": s, "direction": direction, "lx": float(cfg.lx), "ly": float(cfg.ly), "nx": cfg.nx,
           "ny": cfg.ny, "n_f": n_f, "n_s": n_s, "h_f_double": h_f, "h_s_double": h_s, "N": N, "D": float(cfg.D), "q": float(cfg.q),
           "marking": RESEARCH if N >= 2 else "N = 1: OBSERVATION (QA tool)", "data_status": DATA_STATUS,
           "given_in_sha": array_sha(given), "companion_in_sha": array_sha(companion)}
    arrays = {"given_in": given, "companion_in": companion}
    # --- 1. rank diagnostic of the given set and of the previous field
    prev_zero = not np.any(companion)
    rec["prev_field"] = "ZERO (w^(s-1) = 0 exactly, H-7)" if prev_zero else "nonzero"
    rec["prev_field_before_QR_sha"] = array_sha(_field(direction, given, companion))
    if s == 1:
        if not (np.array_equal(given, hs.start_set(n_f, N)) and prev_zero):  # the start theorem is about T_N only (D-049.3)
            raise ValueError("s = 1 accepts only the start set T_N = hs.start_set(n_x, N) with a zero companion (D-049.3, H-7)")
        rec["start_set"] = "T_N, sin^2 basis (hs.start_set; D-049.3 item 1); given bitwise hs.start_set(n_x, N), companion 0"
        rec["start_independence_condition"] = f"N <= floor(n_x/2): {N} <= {n_f // 2} -> {N <= n_f // 2}"
        rec["start_exact_formulas_condition"] = f"2N < n_x: {2 * N} < {n_f} -> {2 * N < n_f}"
        if N > n_f // 2:
            return _stop(rec, arrays, STOP_START)
    rank = hs.exact_rank(given)
    rec["given_exact_rank"] = rank
    if rank < N:
        return _stop(rec, arrays, STOP_DEPENDENT)
    if N >= 2 and s >= 2 and not prev_zero:
        diag, _, _ = seq._rank_diagnostic(companion, h_s, given, h_f)
        rec.update({f"prev_field_{k}": v for k, v in diag.items()})
    rec["rank_decision"] = "UNRESOLVED (no approved numerical rank criterion; D-049.5, D-049.6)" if N >= 2 else "N = 1"
    # --- 2./3. QR (N >= 2) and the prototype's diagnostics
    if N >= 2:
        Ut, R = bc.orthonormalise(p)
        solved = hs.HalfStep(cfg.lx, cfg.nx, cfg.ly, cfg.ny, direction, Ut, D=cfg.D, q=cfg.q)
        comp_solved = companion @ R.T
        sv = np.linalg.svd(R, compute_uv=False)
        eA = np.linalg.eigvalsh(_BC_COEFFICIENTS(solved)[0])
        rec.update({"qr": True, "qr_function": "bc.orthonormalise (prototype Householder QR, R4 5.4)",
                    "cond_A_before_QR": float((sv[0] / sv[-1]) ** 2) if sv[-1] > 0 else math.inf,
                    "cond_A_after_QR": float(eA[-1] / eA[0]) if eA[0] > 0 else math.inf,
                    "R_diag_min_abs": float(np.min(np.abs(np.diag(R)))),
                    "orthonormality_defect_up": bc.orthonormality_defect(solved),
                    "sin_angle_span_G_Utilde": bc.span_change_sin(given, Ut, h_f)})
        arrays["R"] = R
        if not np.all(np.diag(R) != 0) or hs.exact_rank(Ut) < N:
            arrays.update({"solved_set": Ut, "companion_solved": comp_solved})
            return _stop(rec, arrays, f"{PROTECTIVE}: QR factor R or the solved set exactly singular in binary64")
    else:
        solved, comp_solved = p, companion
        rec.update({"qr": False, "cond_A_before_QR": 1.0})
    arrays.update({"solved_set": solved.given, "companion_solved": comp_solved})
    rec["solved_set_sha"] = array_sha(solved.given)
    rec["companion_solved_sha"] = array_sha(comp_solved)
    W_after_qr = _field(direction, solved.given, comp_solved)
    rec["prev_field_after_QR_sha"] = array_sha(W_after_qr)
    rec["prev_field_QR_max_abs_change"] = _max_abs_diff(W_after_qr, _field(direction, given, companion))  # OBSERVATION
    # --- 4. C-3+ of the solved set, premises, exact theorem check
    probe_before = fp_environment_probe()
    parts = ca.c3plus(solved)
    prem = c3plus_premises(solved, parts, probe_before)
    rec["C_assembly"] = "C-3+ (scripts/r5_cccc_q4_c_assembly_e4_audit.py::c3plus, PR #40) on the solved set"
    rec["C_assembled_on_solved_set_sha"] = array_sha(np.asarray(solved.given))
    rec["c3plus_premises"] = {k: v for k, v in prem.items() if k not in ("status", "summary", "reasons")}
    rec["c3plus_premises_status"] = prem["status"]
    rec["c3plus_premises_summary"] = prem["summary"]
    rec["fp_probe_before"] = probe_before
    arrays["C3plus"] = parts["C"]
    rec["C3plus_sha"] = array_sha(parts["C"])
    if prem["status"] != PREMISES_OK:
        return _stop(rec, arrays, f"{UNRES_PREMISE}: " + "; ".join(prem["reasons"]))
    coeffs = hs.exact_coefficients(solved)
    chk = ca.check_c3plus(parts["C"], coeffs[2], prem["k"])
    rec["c3plus_bound_check_exact"] = "PASS" if (chk["elementwise"] and chk["frobenius"]) else "FAIL"
    rec["c3plus_bound_check_scope"] = BOUND_CHECK_SCOPE
    rec["c3plus_bound_max_ratio_obs"] = math.sqrt(float(chk["max_ratio_sq"]))  # max |dC_ij|/(gamma_k sqrt(C_ii C_jj)), OBSERVATION
    rec["c3plus_C_spd_exact"] = bool(ca.spd_exact(parts["C"]))
    if rec["c3plus_bound_check_exact"] != "PASS":
        return _stop(rec, arrays, FAIL_THEOREM)
    # --- 5. B-C with only C replaced
    sol, info = solve_with_c(solved, parts["C"])
    rec.update(info)
    probe_after = fp_environment_probe()
    rec["fp_probe_after"] = probe_after
    rec["bc_status"] = sol.status
    if not info["only_C_replaced"]:
        return _stop(rec, arrays, f"{FAIL_TOOL}: only-C replacement not confirmed")
    if not all(probe_after.values()):
        return _stop(rec, arrays, f"{UNRES_PREMISE}: P-1/P-3 probe after the solve failed")
    if sol.s is None:
        return _stop(rec, arrays, f"{PROTECTIVE}: B-C status {sol.status}")
    rec["C_assembly_applied"] = ("C-3+ (confirmed: the isolated call of the code of solve_bc consumed exactly the C-3+ matrix and "
                                 "A, B, F of the prototype)")
    for k in ("cond_S", "cond_S_modal", "cond_S_rows", "cond_A", "max_cond_Mk", "min_eig_Mk", "rho_0", "kappa_bar_c", "M_norm_2"):
        rec[f"bc_{k}"] = sol.diagnostics.get(k)
    m = n_s - 1
    V = sol.s.reshape(N, m).T
    arrays["V_bc"] = V
    rec["V_bc_sha"] = array_sha(V)
    # --- 6. residual of the original operator
    rec["eta_qa"] = bc._eta(solved, sol.s, coeffs)
    # --- 7. rank diagnostic of the new field
    W_s = _field(direction, solved.given, V)
    rec["field_s_sha"] = array_sha(W_s)
    rec["V_exact_rank"] = hs.exact_rank(V)
    if N >= 2 and np.any(V):
        diag, _, _ = seq._rank_diagnostic(solved.given, h_f, V, h_s)
        rec.update({f"field_s_{k}": v for k, v in diag.items()})
    # --- 8. transformation for s + 1
    if N == 1:
        f = V[:, 0]
        if not np.any(f):
            return _stop(rec, arrays, STOP_ZERO)
        exact_sum = sum((Fraction(float(v)) for v in f), Fraction(0))
        sigma = 1.0 if exact_sum >= 0 else -1.0
        c = sigma * float(np.max(np.abs(f)))
        f_next = (f / c)[:, None]
        g_next = solved.given * c
        rec.update({"normalisation": "I-4 (D-049.1 item 4; NS 4.4): f <- f/c, g <- g c", "normalisation_c": c, "sigma": sigma,
                    "sigma_double_sum_agrees": (1.0 if float(np.sum(f)) >= 0 else -1.0) == sigma,
                    "f_after_max_abs": float(np.max(np.abs(f_next)))})
        W_norm = _field(direction, g_next, f_next)
        rec["field_s_after_normalisation_sha"] = array_sha(W_norm)
        rec["field_s_normalisation_max_abs_change"] = _max_abs_diff(W_norm, W_s)  # OBSERVATION
        nxt_g, nxt_c = f_next, g_next
        rec["next_input"] = "N = 1: next given = f/c (normalised B-C factor), next companion = g c"
    else:
        nxt_g, nxt_c = V, solved.given
        rec["next_input"] = "N >= 2: next given = V (B-C factor, untransformed; QR at step 2 of s + 1), next companion = U~"
        if rec["V_exact_rank"] < N:
            rec["next_step_note"] = "V exactly dependent: half-step s + 1 stops before QR"
    arrays.update({"next_given": nxt_g, "next_companion": nxt_c})
    rec["next_given_sha"], rec["next_companion_sha"] = array_sha(nxt_g), array_sha(nxt_c)
    rec["status"] = OK
    return StepResult(rec, arrays, nxt_g, nxt_c)


# ---------------------------------------------------------------------------------------------------------------------------------
# trajectory
# ---------------------------------------------------------------------------------------------------------------------------------

@dataclass
class Trajectory:
    config: TrajectoryConfig
    records: list[dict] = field(default_factory=list)
    arrays: list[dict] = field(default_factory=list)
    status: str = "RUNNING"
    timing: list[float] = field(default_factory=list)


def run_trajectory(cfg: TrajectoryConfig, independent: bool = False) -> Trajectory:
    """Own B-C (C-3+) trajectory s = 1..s_max (module docstring). ``independent``: run ``independent_check`` on every solved small
    half-step (cost limits EXACT_CHECK_MAX_UNKNOWNS, RH_MAX_NODES)."""
    tr = Trajectory(cfg)
    given = hs.start_set(cfg.nx, cfg.N)
    companion = np.zeros((cfg.ny - 1, cfg.N))
    for s in range(1, cfg.s_max + 1):
        t0 = time.perf_counter()
        try:
            res = run_step(cfg, s, given, companion)
        except Exception as exc:  # recorded, never hidden
            res = StepResult({"trajectory_id": cfg.trajectory_id, "s": s, "direction": direction_of(s),
                              "status": f"{FAIL_TOOL}: exception {type(exc).__name__}: {exc}"}, {})
        if s == 1:
            res.record["input_source"] = "start: T_N (hs.start_set) and zero companion"
            res.record["input_link_bitwise"] = True
        elif "given_in" in res.arrays:
            prev = tr.arrays[-1]
            res.record["input_source"] = f"B-C result of half-step {s - 1} of this trajectory (next_given, next_companion of s - 1)"
            res.record["input_link_bitwise"] = bool(np.array_equal(res.arrays["given_in"], prev["next_given"])
                                                    and np.array_equal(res.arrays["companion_in"], prev["next_companion"]))
            if not res.record["input_link_bitwise"]:
                res.record["status"] = f"{FAIL_TOOL}: input of s is not the B-C result of s - 1"
                res.next_given = None
        if independent and res.record.get("status") == OK:
            res.record["independent_check"] = independent_check(cfg, s, res.arrays)
        tr.records.append(res.record)
        tr.arrays.append(res.arrays)
        tr.timing.append(round(time.perf_counter() - t0, 3))
        if res.next_given is None:
            tr.status = f"STOPPED at s = {s}: {res.record['status']}"
            return tr
        given, companion = res.next_given, res.next_companion
    tr.status = f"COMPLETED s = 1..{cfg.s_max} (no stop criterion applied)"
    return tr


# ---------------------------------------------------------------------------------------------------------------------------------
# independent check of small half-steps (R-H, exact system)
# ---------------------------------------------------------------------------------------------------------------------------------

def _exact_system(p: hs.HalfStep, coeffs):
    """Exact K and b = F (x) 1 of the half-step system (term-first) from the exact R-HP residual: b = r(0), K e_j = r(0) - r(e_j)."""
    n = p.size
    b = hs.residual_exact(p, np.zeros(n), coeffs)
    K = [[None] * n for _ in range(n)]
    for j in range(n):
        e = np.zeros(n)
        e[j] = 1.0
        rj = hs.residual_exact(p, e, coeffs)
        for i in range(n):
            K[i][j] = b[i] - rj[i]
    return K, b


def _dist2(v: np.ndarray, x: list[Fraction]) -> Fraction:
    return sum(((Fraction(float(a)) - b) ** 2 for a, b in zip(v, x, strict=True)), Fraction(0))


def independent_check(cfg: TrajectoryConfig, s: int, arrays: dict) -> dict:
    """Module docstring, last paragraph. Exact statements only; 'CONSISTENT' iff all exact checks hold."""
    solved = hs.HalfStep(cfg.lx, cfg.nx, cfg.ly, cfg.ny, direction_of(s), arrays["solved_set"], D=cfg.D, q=cfg.q)
    s_bc = np.asarray(arrays["V_bc"]).T.reshape(-1)  # term-first
    if solved.size > EXACT_CHECK_MAX_UNKNOWNS:
        return {"status": f"NOT_PERFORMED (size {solved.size} > {EXACT_CHECK_MAX_UNKNOWNS} unknowns)"}
    out = {"object": "the solved set of this half-step (after QR if applied); exact system from its doubles, h = l/n exact"}
    coeffs = hs.exact_coefficients(solved)
    K, b = _exact_system(solved, coeffs)
    x = bc._solve_exact(K, b)
    out["exact_solution_identity_Kx_eq_b"] = all(sum((K[i][j] * x[j] for j in range(len(x))), Fraction(0)) == b[i]
                                                 for i in range(len(x)))
    tau = bc._tau(solved)
    low = None if tau is None else hs.certify_lambda_min(solved, tau, coeffs)
    if low is None:
        out["status"] = f"{UNRESOLVED}: no lambda_min certificate (Q1b)"
        return out
    out["lambda_low"] = low
    b_bc = hs.error_bound(solved, s_bc, low, coeffs)
    d2 = _dist2(s_bc, x)
    out["bc_bound_2_up"] = b_bc
    out["bc_within_certified_bound"] = math.isfinite(b_bc) and d2 <= Fraction(b_bc) ** 2
    xs2 = sum((v * v for v in x), Fraction(0))
    out["bc_rel_error_to_exact_obs"] = math.sqrt(float(d2 / xs2)) if xs2 else UNRESOLVED  # OBSERVATION
    checks = [out["exact_solution_identity_Kx_eq_b"], out["bc_within_certified_bound"]]
    if (cfg.nx - 1) * (cfg.ny - 1) <= RH_MAX_NODES:
        st_h, s_h = hs.solve_hessian(solved)
        out["rh_status"] = st_h
        if s_h is not None:
            b_h = hs.error_bound(solved, s_h, low, coeffs)
            out["rh_bound_2_up"] = b_h
            out["rh_within_certified_bound"] = math.isfinite(b_h) and _dist2(s_h, x) <= Fraction(b_h) ** 2
            out["bc_rh_certified_consistent"] = bool(bc._certified_pair_consistent(s_bc, b_bc, s_h, b_h))
            checks += [out["rh_within_certified_bound"], out["bc_rh_certified_consistent"]]
        else:
            checks.append(False)
    else:
        out["rh_status"] = f"NOT_PERFORMED ((n_x - 1)(n_y - 1) > {RH_MAX_NODES})"
    out["status"] = "CONSISTENT" if all(checks) else "CONTRADICTION"
    return out


# ---------------------------------------------------------------------------------------------------------------------------------
# journal and provenance
# ---------------------------------------------------------------------------------------------------------------------------------

def git(*args) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=False).stdout.strip()
    except OSError:
        return ""


def _blas_info() -> dict:
    try:
        cfg = np.show_config(mode="dicts")
        return {k: cfg["Build Dependencies"].get(k) for k in ("blas", "lapack")}
    except Exception as exc:  # recorded, the journal does not depend on it
        return {"error": f"{type(exc).__name__}: {exc}"}


def provenance() -> dict:
    used = used_modules_check()
    changed = [rel for rel, v in used.items() if not v["unchanged"]]
    return {"git_commit": git("rev-parse", "HEAD"), "git_status_porcelain": git("status", "--porcelain").splitlines(),
            "base_commit": BASE_COMMIT, "tool_blob": git_blob_id(Path(__file__)), "used_modules": used,
            "used_modules_status": ("listed modules: blob = base" if not changed else "CHANGED against the base: " + ", ".join(changed))
                                   + "; recorded, the run is not stopped; packages (NumPy, SciPy, BLAS): versions only, not checked",
            "loaded_project_modules_not_checked": loaded_modules_check(),
            "python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__, "blas": _blas_info(),
            "platform": platform.platform(), "machine": platform.machine(), "processor": platform.processor(),
            "blas_thread_env_vars_as_found": {k: os.environ.get(k) for k in THREAD_VARS},
            "blas_threads_effective": "NOT_VERIFIED (the values above are environment variables, not the effective thread count)",
            "solve_isolation": ISOLATION, "d1_guard": D1_GUARD,
            "reproducibility": "bitwise only on the same platform, BLAS build and thread settings; not across BLAS platforms "
                               "(D-025; AGENTS.md 4)"}


def _json(o):
    if isinstance(o, (np.bool_, np.integer, np.floating)):
        return o.item()
    if isinstance(o, Fraction):
        return f"{o.numerator}/{o.denominator}"
    raise TypeError(type(o).__name__)


def write_journal(tr: Trajectory, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    npz = {f"s{rec['s']}_{k}": np.asarray(v) for rec, arrs in zip(tr.records, tr.arrays) for k, v in arrs.items()}
    np.savez(out / "arrays.npz", **npz)
    cfg = tr.config
    meta = {"tool": "R5-V Q4 OP-8: own B-C trajectories with the C-3+ assembly (QA research tool)", "data_status": DATA_STATUS,
            "d1_guard": D1_GUARD,
            "config": {k: getattr(cfg, k) for k in ("lx", "nx", "ly", "ny", "N", "D", "q", "s_max")}, "first": "Y",
            "trajectory_id": cfg.trajectory_id, "status": tr.status, "half_steps": len(tr.records),
            "arrays_npz": "arrays.npz", "arrays_sha256": {k: array_sha(v) for k, v in npz.items()},
            "timing_seconds": tr.timing, "provenance": provenance()}
    with open(out / "journal.json", "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "steps": tr.records}, f, indent=1, ensure_ascii=False, default=_json)
        f.write("\n")
    return meta


def load_arrays(out: Path) -> dict:
    with np.load(out / "arrays.npz") as z:
        return {k: z[k] for k in z.files}


# ---------------------------------------------------------------------------------------------------------------------------------
# command line (single small non-D1 trajectory)
# ---------------------------------------------------------------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--lx", type=float, required=True)
    r.add_argument("--nx", type=int, required=True)
    r.add_argument("--ly", type=float, required=True)
    r.add_argument("--ny", type=int, required=True)
    r.add_argument("--N", type=int, required=True)
    r.add_argument("--D", type=float, default=1.0)
    r.add_argument("--q", type=float, default=1.0)
    r.add_argument("--s-max", type=int, default=8)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--independent-check", action="store_true")
    a = ap.parse_args(argv)
    if (a.nx, a.ny) in D1_GRIDS:
        print(f"refused: ({a.nx}, {a.ny}) is a grid of the D1 sample; D1_START = BLOCKED (protocol 12)", file=sys.stderr)
        return 2
    out = a.out.resolve()
    if (ROOT / "docs").resolve() in out.parents or out == (ROOT / "docs").resolve():
        print("refused: tool output is not written under docs/ (no published data)", file=sys.stderr)
        return 2
    cfg = TrajectoryConfig(a.lx, a.nx, a.ly, a.ny, a.N, a.D, a.q, a.s_max)
    tr = run_trajectory(cfg, independent=a.independent_check)
    meta = write_journal(tr, out)
    print(json.dumps({"trajectory_id": meta["trajectory_id"], "status": tr.status,
                      "steps": [r["status"] for r in tr.records]}, ensure_ascii=False))
    return 0 if not any(str(r["status"]).startswith(FAIL_TOOL) or str(r["status"]).startswith("FAIL_C3PLUS")
                        for r in tr.records) else 1


if __name__ == "__main__":
    sys.exit(main())
