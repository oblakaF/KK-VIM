"""R5-V CCCC, stage Q3: pre-qualification measurement campaign E-1...E-9 on the QA references.

Basis: docs/R5_CCCC_PREQUALIFICATION.plan.md (PL) 2.2, 4, 6.3, 7; docs/R5_CCCC_ROUNDOFF_PREQUAL_DRAFT.md (RO) 4.1-4.5, VOK-1...VOK-14;
D-049.5, D-049.6 (item 4: reference budget 1/100), D-051.4, D-052.3. QA research measurements, NOT a qualification: no threshold,
no K_E, c_R, c_DST, eps_floor^C, tol_solve, eta_max^C, K, admissible cond(S) is assigned; no outcome makes Q3 ACCEPTED. No import of
``kkvim``; only kkvim_reference (r5_cccc_sequence, r5_cccc_energy, r5_cccc_e2, r5_cccc_bc, r5_cccc_q3_e4e6, r5_cccc_halfstep,
r5_cccc_dst_cert, r5_cccc_d2). N >= 2: RESEARCH_ONLY / PROVISIONAL / NOT_QUALIFIED (D-049.5), rank decision UNRESOLVED.

SAMPLE (fixed before the runs; RO 4.1 after Z-4 = PL 2.2 and BC-2). Plate a x b = 1 x r (r = b/a), D = q = 1:
- ratios r in {1, 2, 3, 5, 1/2, 1/3, 1/5};
- pairs (n_x, n_y) = (40, 64), (64, 40), (47, 53), (160, 32), (32, 160) on the plate 1 x 1 (h_x != h_y);
- chain: n = intervals on the SHORT side, the long side r n (or n/r) intervals (equal steps up to the rounding of b); n = 32...512
  for r in {1, 2, 1/2} and n = 32...256 for r in {3, 1/3, 5, 1/5} (BC-2: r <= 2 up to 512, r = 5 up to 256; r = 3 is not named in
  BC-2 - the conservative limit of r = 5 is used, an interpretation recorded in the report);
- N = 1, 2, 3; both half-step directions; sequences s = 1..8 from T_N (D-049.3, first half-step Y; for the pairs also first X).
Extended OBSERVATION (RO 4.1, not part of candidate bases): the exact tier of E-4 on small grids (N m <= 36) and n = 16, 1024, 2048
in E-5.

MEASUREMENTS (RO 4.3; statuses: PROVEN = exact algebra/theorem check, CERTIFIED = rigorous bound computed with directed rounding,
MODEL = first-order model, OBSERVATION = value, UNRESOLVED = not established; reference budget per D-049.6, item 4):
- E-1, E-7, E-9: sequences (r5_cccc_sequence with energy): exact VOK-2 identity, exact parts of the computed defect (solution,
  formation, assembly), gamma^, R^C, R_Q^C, R_f^C (MODEL), m_s = |dP^ - Q^ + gamma^|/(R^C + R^C + R_Q^C + R_f^C) with the operation
  count c_R = 22 of the double path of r5_cccc_energy (G-4 count, not an owner's value) and with c_R = 0 (base only); R_Q^C against
  R^C on the same pairs; cond(A) before QR against 1/rho^_N^2 (no model fixed: comparison UNRESOLVED, values OBSERVATION).
- E-2: e2.measure on every solved field (X, Y as stored): exact formation error, VOK-4/VOK-5 statuses, double norm paths (iv).
- E-3: the same half-step in two bases of one span (T_N: sin^2, its QR, the product basis, a unit-triangular combination; at
  s >= 2: the solved set before and after QR / I-4), solved by R-AC and B-C; exact difference of the exact product fields and
  certified solve bounds (Q1b certificate on each system): transformation part = interval.
- E-4, E-6: r5_cccc_q3_e4e6.e4_case / e6_case (QR for N >= 2; without QR on the pairs as BC-6 analogue), at s = 1 on the chain, the
  pairs and the exact tier, and at s = 2..8 on the sets of the pair sequences.
- E-5: scipy DST-I (r5_cccc_bc.dst) against the certified DST-I (r5_cccc_dst_cert); subnormal inputs or results are excluded from
  the c_DST statistic (condition U-P3, D-051.4, item 4).
- E-8: distance of every sequence field to omega_h^C = D1 = D2: computed EXACTLY between the double arrays, plus the CERTIFIED
  reference bound ||omega_D2 - omega_h||_a <= ||D A omega_D2 - q 1||_h/(D^(1/2)(4/a^2 + 4/b^2)) (VOK-7 (b), exact residual of
  r5_cccc_d2, mu_x + mu_y >= 4/a^2 + 4/b^2); budget 1/100 checked exactly; B_omega of the journal against the measured distance.

Not available (UNRESOLVED, missing capability named): E-4 components eps_0, eps_Y, eps_S, eps_k, T_1...T_4 (r5_cccc_bc does not expose
x0^, Y0^, c^, G^-1^; no raised-precision B-C reference for N m > 36); rigorous bound of the double energy paths (VOK-3 MODEL);
strict majorization by tau_s^C (OK-1); QR model (OK-5).

Execution: one process per job (multiprocessing, spawn), BLAS/OpenMP threads = 1 (fixed order of operations within a job), peak
working set recorded per job; resource rule PL 6.3 (peak <= 50 % RAM, factorization <= 30 min, refinement converges).

    uv run python scripts/r5_cccc_q3_prequalification.py run              # full campaign (hours, parallel workers)
    uv run python scripts/r5_cccc_q3_prequalification.py run --profile ci # tiny configuration (seconds)
Writes docs/results/r5_cccc_q3_prequalification/ (or --out); stored double arrays of the sequences go to scratch/ (not in git).
"""

import argparse
import csv
import json
import math
import multiprocessing
import os
import platform
import statistics
import subprocess
import sys
import time
from fractions import Fraction
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "results" / "r5_cccc_q3_prequalification"
ARRAYS = ROOT / "scratch" / "r5_cccc_q3_prequalification" / "arrays"
THREAD_VARS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")

import numpy as np  # noqa: E402

from kkvim_reference import r5_cccc_bc as bc  # noqa: E402
from kkvim_reference import r5_cccc_d2 as d2  # noqa: E402
from kkvim_reference import r5_cccc_dst_cert as dc  # noqa: E402
from kkvim_reference import r5_cccc_e2 as e2  # noqa: E402
from kkvim_reference import r5_cccc_energy as en  # noqa: E402
from kkvim_reference import r5_cccc_halfstep as hs  # noqa: E402
from kkvim_reference import r5_cccc_q3_e4e6 as e46  # noqa: E402
from kkvim_reference import r5_cccc_sequence as sq  # noqa: E402

UNRESOLVED = "UNRESOLVED"
C_R_PATH = en.C_R_COUNT  # 22: operation count of the double path of r5_cccc_energy (G-4), NOT an owner's c_R
SEED = 20261025
BIG_GRID = 150_000  # n_x n_y from which the E-8 D2 jobs run one at a time (memory; PL 6.3)
RESEARCH = "RESEARCH_ONLY / PROVISIONAL / NOT_QUALIFIED (D-049.5)"

# --- sample (fixed before the runs) ---------------------------------------------------------------------------------------------

RATIOS = {"1": 1.0, "2": 2.0, "3": 3.0, "5": 5.0, "1/2": 0.5, "1/3": 1 / 3, "1/5": 0.2}
PAIRS = [(40, 64), (64, 40), (47, 53), (160, 32), (32, 160)]
CHAIN = [32, 64, 128, 256, 512]
E5_MAIN = [32, 64, 128, 256, 512]
E5_EXTENDED = [16, 1024, 2048]
EXACT_TIER = [(1.0, 8, 1.3, 7), (1.0, 10, 0.6, 9), (1.0, 12, 1.0, 13)]


def chain_limit(r: str) -> int:
    """BC-2: r <= 2 up to 512, r = 5 up to 256; r = 3 (not named) with the limit of r = 5."""
    return 512 if RATIOS[r] in (1.0, 2.0, 0.5) else 256


def plate_grid(r: str, n: int) -> tuple[float, int, float, int]:
    """(l_x, n_x, l_y, n_y) of the chain point: plate 1 x r, n intervals on the short side, equal steps."""
    ly = RATIOS[r]
    if ly >= 1:
        return 1.0, n, ly, int(round(ly * n))
    return 1.0, int(round(n / ly)), ly, n


def grid_label(lx, nx, ly, ny) -> str:
    return f"{lx:g}x{ly:.6g}:{nx}x{ny}"


def sequence_specs(profile: str = "full") -> list[dict]:
    if profile == "ci":
        return [{"group": "ci", "r": "1.3", "lx": 1.0, "nx": 10, "ly": 1.3, "ny": 9, "N": N, "first": "Y"} for N in (1, 2)]
    out = []
    for nx, ny in PAIRS:
        for N in (1, 2, 3):
            for first in ("Y", "X"):
                out.append({"group": "pairs", "r": "1", "lx": 1.0, "nx": nx, "ly": 1.0, "ny": ny, "N": N, "first": first})
    for r in RATIOS:
        for n in CHAIN:
            if n > chain_limit(r):
                continue
            lx, nx, ly, ny = plate_grid(r, n)
            for N in (1, 2, 3):
                out.append({"group": "chain", "r": r, "n": n, "lx": lx, "nx": nx, "ly": ly, "ny": ny, "N": N, "first": "Y"})
    return out


def halfstep_specs(profile: str = "full") -> list[dict]:
    """s = 1 half-steps from T_N for E-4/E-6: chain and pairs (main sample), exact tier (extended OBSERVATION)."""
    if profile == "ci":
        return [{"group": "ci", "lx": 1.0, "nx": 8, "ly": 1.3, "ny": 7, "qr_modes": (True,)}]
    out = []
    for r in RATIOS:
        for n in CHAIN:
            if n <= chain_limit(r):
                lx, nx, ly, ny = plate_grid(r, n)
                out.append({"group": "chain", "r": r, "n": n, "lx": lx, "nx": nx, "ly": ly, "ny": ny, "qr_modes": (True,)})
    for nx, ny in PAIRS:
        out.append({"group": "pairs", "r": "1", "lx": 1.0, "nx": nx, "ly": 1.0, "ny": ny, "qr_modes": (True, False)})
    for lx, nx, ly, ny in EXACT_TIER:
        out.append({"group": "exact_tier (OBSERVATION)", "r": f"{ly / lx:g}", "lx": lx, "nx": nx, "ly": ly, "ny": ny,
                    "qr_modes": (True, False)})
    return out


# --- helpers -------------------------------------------------------------------------------------------------------------------

def peak_mb() -> float | None:
    """Peak working set of this process in MB (Windows: GetProcessMemoryInfo; POSIX: ru_maxrss); None if unavailable."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t), ("a", ctypes.c_size_t), ("b", ctypes.c_size_t),
                        ("c", ctypes.c_size_t), ("d", ctypes.c_size_t), ("e", ctypes.c_size_t), ("f", ctypes.c_size_t)]

        c = Counters()
        c.cb = ctypes.sizeof(c)
        k = ctypes.WinDLL("kernel32")
        k.GetCurrentProcess.restype = wintypes.HANDLE
        p = ctypes.WinDLL("psapi")
        p.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        p.GetProcessMemoryInfo.restype = wintypes.BOOL
        return round(c.PeakWorkingSetSize / 2**20, 1) if p.GetProcessMemoryInfo(k.GetCurrentProcess(), ctypes.byref(c), c.cb) \
            else None
    try:
        import resource
        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    except (ImportError, OSError):
        return None


def total_ram_mb() -> float | None:
    if sys.platform == "win32":
        import ctypes

        class Status(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        s = Status()
        s.dwLength = ctypes.sizeof(s)
        return round(s.ullTotalPhys / 2**20, 1) if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s)) else None
    try:
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**20, 1)
    except (ValueError, OSError, AttributeError):
        return None


def ratio(num, den):
    """num/den as an OBSERVATION; UNRESOLVED when a value is missing, not finite or the denominator is 0."""
    if not isinstance(num, (int, float)) or not isinstance(den, (int, float)) or isinstance(num, bool) or isinstance(den, bool):
        return UNRESOLVED
    if den == 0 or not (math.isfinite(num) and math.isfinite(den)):
        return UNRESOLVED
    return float(num) / float(den)


def budget(ref_upper: Fraction, measured_lower: Fraction) -> str:
    """D-049.6, item 4: reference error <= measured/100 in the same norm (exact rational comparison)."""
    if measured_lower <= 0:
        return "UNRESOLVED (measured quantity not separated from 0)"
    return "SATISFIED" if 100 * ref_upper <= measured_lower else "NOT_MET: comparison UNRESOLVED"


def frac_field(w) -> list[list[Fraction]]:
    return [[Fraction(float(v)) for v in row] for row in np.asarray(w, dtype=float)]


def exact_product(X, Y) -> list[list[Fraction]]:
    Xf = [[Fraction(float(v)) for v in row] for row in X]
    Yf = [[Fraction(float(v)) for v in row] for row in Y]
    return [[sum((a * b for a, b in zip(x, y)), Fraction(0)) for y in Yf] for x in Xf]


def orient(direction: str, given, solved):
    """(x-factor, y-factor) of a half-step field: Y-half-step - given is the x-factor."""
    return (given, solved) if direction == "Y" else (solved, given)


def solved_factor(p: hs.HalfStep, s) -> np.ndarray:
    return np.asarray(s, dtype=float).reshape(p.N, p.n_s - 1).T


def certificate(p: hs.HalfStep, s, coeffs=None) -> dict:
    """Q1b certificate of the system of ``p`` for the vector s: lambda_low, the set bound and the energy-field bound
    ||w~ - w*||_a <= h_s^(1/2) ||r||_2/lambda_low^(1/2) (NS 4.3, P-S1); None entries if no certificate."""
    coeffs = coeffs or hs.exact_coefficients(p)
    tau = bc._tau(p)
    low = None if tau is None else hs.certify_lambda_min(p, tau, coeffs)
    if low is None:
        return {"lambda_low": None, "field_bound_a": None}
    r = hs.residual_exact(p, s, coeffs)
    rK = en.sqrt_interval(sum((x * x for x in r), Fraction(0)))[1] / en.sqrt_interval(Fraction(low))[0]
    return {"lambda_low": low, "field_bound_a": en.sqrt_interval(Fraction(p.l_s) / p.n_s)[1] * rK}


def _job_meta(t0: float) -> dict:
    return {"seconds": round(time.perf_counter() - t0, 2), "peak_mb": peak_mb(), "pid": os.getpid()}


# --- sequences: E-1, E-2, E-7, E-9 (and stored arrays for E-3, E-4, E-6, E-8) ----------------------------------------------------

class Recorder:
    """Records the (x-factor, y-factor) pairs passed to energy.eps_a_C by r5_cccc_sequence: per half-step first the solved field,
    then the field after the transformation (QR / I-4). An observation hook only; the association is verified bitwise against the
    journal (P_double_s, Q_double, R_base_up) before any use."""

    def __init__(self):
        self.calls = []
        self._orig = en.eps_a_C

    def __enter__(self):
        def hook(g, X, Y):
            self.calls.append((np.array(X, dtype=float, copy=True), np.array(Y, dtype=float, copy=True)))
            return self._orig(g, X, Y)

        en.eps_a_C = hook
        return self

    def __exit__(self, *exc):
        en.eps_a_C = self._orig
        return False


def seq_job(spec: dict, arrays_dir: str | None = None) -> dict:
    t0 = time.perf_counter()
    cfg = sq.SequenceConfig(spec["lx"], spec["nx"], spec["ly"], spec["ny"], spec["N"], first=spec["first"],
                            s_max=spec.get("s_max", 8))
    g = en.Grid(cfg.lx, cfg.nx, cfg.ly, cfg.ny, cfg.D, cfg.q)
    with Recorder() as rec:
        res = sq.run_sequence(cfg)
    journal = sq.journal(res)
    tag = {k: spec.get(k) for k in ("group", "r", "n", "nx", "ny", "N", "first")}
    tag["grid"] = grid_label(cfg.lx, cfg.nx, cfg.ly, cfg.ny)
    seq_rows, e2_rows, stored = [], [], {}
    calls = list(rec.calls)
    w_prev = np.zeros((cfg.nx - 1, cfg.ny - 1))
    lam_h = float((4 * cfg.nx**2 / cfg.lx**2 + 4 * cfg.ny**2 / cfg.ly**2) /
                  (bc.modal_eigenvalues(cfg.nx, cfg.lx / cfg.nx)[0] + bc.modal_eigenvalues(cfg.ny, cfg.ly / cfg.ny)[0]))
    for step, row in zip(res.steps, journal):
        out = {**tag, "sequence_status": res.status, **{f"j_{k}": v for k, v in row.items()}}
        out["marking"] = RESEARCH if cfg.N >= 2 else "N = 1 (OBSERVATION)"
        if "P_double_s" not in step or not calls:
            out["association"] = "NOT_APPLICABLE (no energy analysis in this step)"
            seq_rows.append(out)
            continue
        Xs, Ys = calls.pop(0)
        w_s = Xs @ Ys.T
        dh = w_prev - w_s
        ok = en.pi_double(g, w_s) == step["P_double_s"] and en.q_double(g, dh) == step["Q_double"]
        Rp, Rs, RQ = en.model_R(g, w_prev), en.model_R(g, w_s), en.model_RQ(g, dh)
        ok = ok and en.up(Fraction(Rp["R_base_up"]) + Fraction(Rs["R_base_up"]) + Fraction(RQ["RQ_base_up"])) == step["R_base_up"]
        out["association"] = "VERIFIED (bitwise: P_double_s, Q_double, R_base_up)" if ok else UNRESOLVED
        if not ok:
            seq_rows.append(out)
            continue
        s = step["s"]
        # E-1: m_s with the path count c_R = 22 (MODEL count) and with c_R = 0; resolution Q^/tau(K_E = 1)
        num = step["m_s_numerator"]
        den22 = step["R_base_up"] + C_R_PATH * step["R_per_cR_up"] + step["R_f_up"]
        den0 = step["R_base_up"] + step["R_f_up"]
        tau1 = abs(step["gamma_double"]) + den22
        out.update({"E1_m_s_cR22": ratio(num, den22), "E1_m_s_cR0": ratio(num, den0),
                    "E1_Q_over_tau_KE1_cR22": ratio(step["Q_double"], tau1),
                    "E1_formation_over_Rf": ratio(abs(step["part_formation"]), step["R_f_up"]),
                    "E1_assembly_over_R_cR22": ratio(abs(step["part_assembly"]),
                                                     step["R_base_up"] + C_R_PATH * step["R_per_cR_up"]),
                    "E1_status": "m_s: MODEL denominators (c_R = 22 is the G-4 path count, not an owner's value); parts exact"})
        # E-9: R_Q^C against R^C on the same pair
        R_both22 = (Rp["R_base_up"] + C_R_PATH * Rp["R_per_cR_up"]) + (Rs["R_base_up"] + C_R_PATH * Rs["R_per_cR_up"])
        RQ22 = RQ["RQ_base_up"] + C_R_PATH * RQ["RQ_per_cR_up"]
        out.update({"E9_RQ_cR22": RQ22, "E9_R_prev_plus_s_cR22": R_both22, "E9_RQ_over_R_cR22": ratio(RQ22, R_both22),
                    "E9_RQ_over_R_base": ratio(RQ["RQ_base_up"], Rp["R_base_up"] + Rs["R_base_up"]),
                    "E9_status": "OBSERVATION (ratio of two MODEL terms)"})
        # E-7: cond(A) before QR of the solved set against 1/rho^_N^2 of the same field; the identity is expected when the given
        # set of the half-step is orthonormal (s >= 2: after the QR of s - 1); s = 1 (given T_N) is checked in E-6 (1 + 2N)
        if cfg.N >= 2 and s == 1:
            out["E7_status"] = "NOT_APPLICABLE (s = 1: given T_N not orthonormal; the s = 1 check 1 + 2N is in E-6)"
        elif cfg.N >= 2 and isinstance(step.get("cond_A_before_QR"), float):
            rho = step.get("rho_N")
            out["E7_inv_rho_N_sq"] = ratio(1.0, rho * rho) if isinstance(rho, float) else UNRESOLVED
            out["E7_condA_over_inv_rho_sq"] = ratio(step["cond_A_before_QR"], out["E7_inv_rho_N_sq"]) \
                if isinstance(out["E7_inv_rho_N_sq"], float) else UNRESOLVED
            out["E7_status"] = ("UNRESOLVED (no rounding model of the identity fixed before the comparison, PL BC-5); values "
                                "OBSERVATION")
        # E-2 on the stored solved field
        m = e2.measure(e2.FieldInputs(Xs, Ys, cfg.lx, cfg.nx, cfg.ly, cfg.ny, origin=f"{tag['grid']} N={cfg.N} s={s}")).values
        e2_row = {**tag, "s": s, "direction": step["direction"], "Lambda_h": lam_h,
                  "marking": out["marking"]}
        for k in ("premises", "vok4_status", "vok5_status", "vok4_violations", "vok4_max_ratio_up", "max_rel_e_up", "e_is_zero",
                  "rel_e_a_down", "rel_e_a_up", "eps_a_C_rel_up", "e_a_over_eps_a_C_up", "kappa_W_down", "kappa_W_up",
                  "w_a_sq_double_definition_rel_error", "w_a_sq_double_hessian_rel_error", "vok1_upper_holds"):
            e2_row[k] = m.get(k)
        kw = m.get("kappa_W_up")
        model_i = (cfg.N * e2.U_EXACT / (1 - cfg.N * e2.U_EXACT)) * Fraction(lam_h) * Fraction(kw) \
            if isinstance(kw, float) else None
        e2_row["H_F1_model_gammaN_Lambda_kappaW"] = float(model_i) if model_i is not None else UNRESOLVED
        e2_row["H_F1_rel_e_a_over_model"] = ratio(m.get("rel_e_a_up"), e2_row["H_F1_model_gammaN_Lambda_kappaW"])
        e2_row["E2_status"] = ("exact reference (delta_ref = 0): budget SATISFIED; VOK-4/VOK-5 normative only under "
                               "P3_CONFIRMED; ratios OBSERVATION")
        e2_rows.append(e2_row)
        stored[f"s{s}_Xs"], stored[f"s{s}_Ys"] = Xs, Ys
        if calls and step.get("status") == "OK":
            Xn, Yn = calls.pop(0)
            stored[f"s{s}_Xn"], stored[f"s{s}_Yn"] = Xn, Yn
            w_prev = Xn @ Yn.T
        stored[f"s{s}_dir"] = np.array([0 if step["direction"] == "Y" else 1])
        seq_rows.append(out)
    if arrays_dir is not None:
        Path(arrays_dir).mkdir(parents=True, exist_ok=True)
        np.savez(Path(arrays_dir) / seq_file(spec), **stored)
    return {"kind": "seq", "spec": spec, "seq": seq_rows, "e2": e2_rows, **_job_meta(t0)}


def seq_file(spec: dict) -> str:
    return f"seq_{spec['nx']}x{spec['ny']}_ly{spec['ly']:.6g}_N{spec['N']}_{spec['first']}.npz"


# --- E-8: distance to omega_h^C = D1 = D2 ------------------------------------------------------------------------------------------

def e8_job(spec: dict, arrays_dir: str) -> dict:
    """All sequences on one grid: exact distance of each stored field to the refined D2 solution and the certified reference bound."""
    t0 = time.perf_counter()
    lx, nx, ly, ny = spec["lx"], spec["nx"], spec["ly"], spec["ny"]
    g = en.Grid(lx, nx, ly, ny)
    prob = d2.D2Problem(lx, ly, nx, ny)
    t1 = time.perf_counter()
    sol = d2.solve(prob)
    d2_seconds = time.perf_counter() - t1
    omega = sol.x.reshape(nx - 1, ny - 1)
    r = d2.residual_exact(prob, sol.x)
    mu = Fraction(4) / Fraction(lx) ** 2 + Fraction(4) / Fraction(ly) ** 2
    ref = en.sqrt_interval(g.hx * g.hy * sum((x * x for x in r), Fraction(0)))[1] / mu  # D = q = 1
    omega_a = en.sqrt_interval(en.norm_a_sq_exact(g, omega))
    rows = []
    for sspec in spec["sequences"]:
        path = Path(arrays_dir) / seq_file(sspec)
        if not path.exists():
            continue
        data = np.load(path)
        jrows = {row["j_s"]: row for row in spec["journal"].get(seq_file(sspec), [])}
        for s in range(1, 9):
            if f"s{s}_Xs" not in data:
                break
            w_exact = exact_product(data[f"s{s}_Xs"], data[f"s{s}_Ys"])  # w~: the field bounded by B_omega (VOK-7, VOK-8)
            diff = [[a - Fraction(float(b)) for a, b in zip(ra, rb)] for ra, rb in zip(w_exact, omega)]
            lo, hi = en.sqrt_interval(en.norm_a_sq_exact(g, diff))
            true_lo, true_hi = max(Fraction(0), lo - ref), hi + ref
            j = jrows.get(s, {})
            b_om = j.get("j_B_omega_up")
            row = {"group": sspec["group"], "r": sspec.get("r"), "n": sspec.get("n"), "grid": grid_label(lx, nx, ly, ny),
                   "N": sspec["N"], "first": sspec["first"], "s": s,
                   "marking": RESEARCH if sspec["N"] >= 2 else "N = 1 (OBSERVATION)",
                   "dist_wexact_to_omegaD2_a_down": en.down(lo), "dist_wexact_to_omegaD2_a_up": en.up(hi),
                   "ref_bound_a_up": en.up(ref), "dist_wexact_to_omega_h_a_down": en.down(true_lo),
                   "dist_wexact_to_omega_h_a_up": en.up(true_hi),
                   "rel_dist_to_omega_h_up": en.up(true_hi / omega_a[0]) if omega_a[0] > 0 else UNRESOLVED,
                   "budget_1_100": budget(ref, lo), "e_D2_estimate": sol.e_D2,
                   "B_omega_up": b_om if isinstance(b_om, float) else UNRESOLVED,
                   "B_omega_route": j.get("j_B_omega_route")}
            if isinstance(b_om, float):
                row["B_omega_over_dist_down"] = en.down(Fraction(b_om) / true_hi) if true_hi > 0 else UNRESOLVED
                # B_omega is a proven upper bound of ||w~ - omega_h||_a (route (a) with the Q1b certificate; route (b) under VOK-4):
                # the certified lower bound of the true distance must not exceed it
                row["B_omega_bound_consistent"] = true_lo <= Fraction(b_om)
            else:
                row["B_omega_over_dist_down"] = row["B_omega_bound_consistent"] = UNRESOLVED
            row["E8_status"] = ("distance: EXACT (exact product w~ of the stored arrays against the double omega_D2); reference: "
                                "CERTIFIED (VOK-7 (b), exact residual, mu >= 4/a^2 + 4/b^2); B_omega ratio OBSERVATION")
            rows.append(row)
    return {"kind": "e8", "spec": {k: spec[k] for k in ("lx", "nx", "ly", "ny")}, "e8": rows, "d2_seconds": round(d2_seconds, 2),
            **_job_meta(t0)}


# --- s = 1 half-steps: E-4, E-6 ----------------------------------------------------------------------------------------------------

def e4_relative(p: hs.HalfStep, qr: bool, row: dict) -> dict:
    """OBSERVATION denominators for the certified tier: ||s^||_2 of the B-C set and ||w||_a ~ (h_s s^T b)^(1/2) (x^T K x = x^T b
    for the exact solution), both in double."""
    rec = bc.run_halfstep(p, qr=qr)
    if rec.bc is None or rec.bc.s is None:
        return row
    s = rec.bc.s
    F = bc.bc_coefficients(rec.solved)[3]
    b = np.repeat(F, rec.solved.n_s - 1)
    wa = math.sqrt(max(0.0, (rec.solved.l_s / rec.solved.n_s) * float(s @ b)))
    row["obs_set_norm_2"], row["obs_field_norm_a"] = float(np.linalg.norm(s)), wa
    row["obs_set_error_rel_up"] = ratio(row.get("set_error_2_up"), row["obs_set_norm_2"])
    row["obs_field_error_a_rel_up"] = ratio(row.get("field_error_a_up"), wa)
    row["obs_vok14_bound_a_rel_up"] = ratio(row.get("vok14_field_a_bound_up"), wa)
    return row


def hs_job(spec: dict) -> dict:
    t0 = time.perf_counter()
    rows4, rows6 = [], []
    for direction in hs.DIRECTIONS:
        for N in (1, 2, 3):
            n_f = spec["nx"] if direction == "Y" else spec["ny"]
            p = hs.HalfStep(spec["lx"], spec["nx"], spec["ly"], spec["ny"], direction, hs.start_set(n_f, N))
            for qr in (spec["qr_modes"] if N >= 2 else (False,)):
                tag = {"group": spec["group"], "r": spec.get("r"), "n": spec.get("n"),
                       "grid": grid_label(spec["lx"], spec["nx"], spec["ly"], spec["ny"]), "s": 1, "set": "T_N (sin^2)"}
                rows4.append(e4_relative(p, qr, {**tag, **e46.e4_case(p, qr=qr)}))
                rows6.append({**tag, **e46.e6_case(p, qr=qr)})
    return {"kind": "hs", "spec": spec, "e4": rows4, "e6": rows6, **_job_meta(t0)}


# --- s >= 2 from the stored pair sequences: E-4, E-6, E-3 ------------------------------------------------------------------------

def seqstep_job(spec: dict, arrays_dir: str) -> dict:
    """Half-steps s = 2..8 of a stored sequence: given set = solved factor of s - 1 (N >= 2: before QR, the half-step applies the same
    Householder QR as the controller; N = 1: after I-4). E-4/E-6 with QR (N >= 2); E-3: the same half-step with the set before and
    after the transformation (QR / I-4)."""
    t0 = time.perf_counter()
    data = np.load(Path(arrays_dir) / seq_file(spec))
    lx, nx, ly, ny, N = spec["lx"], spec["nx"], spec["ly"], spec["ny"], spec["N"]
    rows4, rows6, rows3 = [], [], []
    for s in range(2, 9):
        if f"s{s - 1}_Xn" not in data or f"s{s}_Xs" not in data:
            break
        prev_dir = "Y" if int(data[f"s{s - 1}_dir"][0]) == 0 else "X"
        direction = "X" if prev_dir == "Y" else "Y"
        before = data[f"s{s - 1}_Ys"] if prev_dir == "Y" else data[f"s{s - 1}_Xs"]
        after = data[f"s{s - 1}_Yn"] if prev_dir == "Y" else data[f"s{s - 1}_Xn"]
        tag = {"group": spec["group"], "grid": grid_label(lx, nx, ly, ny), "first": spec["first"], "s": s}
        p = hs.HalfStep(lx, nx, ly, ny, direction, before if N >= 2 else after)
        rows4.append(e4_relative(p, N >= 2, {**tag, "set": "sequence set", **e46.e4_case(p, qr=N >= 2)}))
        rows6.append({**tag, "set": "sequence set", **e46.e6_case(p, qr=N >= 2)})
        rows3.append({**tag, **e3_pair(lx, nx, ly, ny, direction, before, after,
                                       "solved set of s-1 before / after " + ("QR" if N >= 2 else "I-4"))})
    return {"kind": "seqstep", "spec": spec, "e4": rows4, "e6": rows6, "e3": rows3, **_job_meta(t0)}


# --- E-3: two bases of one span -----------------------------------------------------------------------------------------------------

def _solutions(p: hs.HalfStep) -> dict:
    out = {}
    ac = hs.solve_ac(p)
    if ac.s is not None:
        out["R-AC"] = ac.s
    sol = bc.solve_bc(p)
    if sol.s is not None:
        out["B-C"] = sol.s
    return out


def e3_pair(lx, nx, ly, ny, direction, G1, G2, label) -> dict:
    """Same half-step with the given sets G1, G2 (same span in exact arithmetic up to the transformation rounding). For each solver:
    Delta = ||w~1 - w~2||_a of the EXACT products of the stored arrays, certified bounds b_i >= ||w~i - w*_i||_a (Q1b, P-S1), the
    transformation part ||w*1 - w*2||_a in [Delta - b1 - b2, Delta + b1 + b2]; formation is excluded (exact products)."""
    g = en.Grid(lx, nx, ly, ny)
    p1 = hs.HalfStep(lx, nx, ly, ny, direction, G1)
    p2 = hs.HalfStep(lx, nx, ly, ny, direction, G2)
    out = {"direction": direction, "N": p1.N, "bases": label, "marking": RESEARCH if p1.N >= 2 else "N = 1 (OBSERVATION)",
           "sin_angle_spans": bc.span_change_sin(np.asarray(G1, dtype=float),
                                                 np.linalg.qr(math.sqrt(p2.l_f / p2.n_f) * np.asarray(G2, dtype=float))[0]
                                                 / math.sqrt(p2.l_f / p2.n_f), p2.l_f / p2.n_f)}
    if hs.given_set_status(p1) != hs.INDEPENDENT or hs.given_set_status(p2) != hs.INDEPENDENT:
        out["E3_status"] = "NOT_APPLICABLE (dependent set)"
        return out
    s1, s2 = _solutions(p1), _solutions(p2)
    c1, c2 = hs.exact_coefficients(p1), hs.exact_coefficients(p2)
    for solver in ("R-AC", "B-C"):
        if solver not in s1 or solver not in s2:
            out[f"{solver}_status"] = UNRESOLVED + " (solver failed)"
            continue
        w1 = exact_product(*orient(direction, p1.given, solved_factor(p1, s1[solver])))
        w2 = exact_product(*orient(direction, p2.given, solved_factor(p2, s2[solver])))
        diff = [[a - b for a, b in zip(ra, rb)] for ra, rb in zip(w1, w2)]
        dlo, dhi = en.sqrt_interval(en.norm_a_sq_exact(g, diff))
        wlo, whi = en.sqrt_interval(en.norm_a_sq_exact(g, w1))
        b1, b2 = certificate(p1, s1[solver], c1)["field_bound_a"], certificate(p2, s2[solver], c2)["field_bound_a"]
        out[f"{solver}_Delta_a_rel_up"] = en.up(dhi / wlo) if wlo > 0 else UNRESOLVED
        if b1 is None or b2 is None:
            out[f"{solver}_transformation_part"] = UNRESOLVED + " (no Q1b certificate)"
            continue
        out[f"{solver}_solve_bounds_rel_up"] = en.up((b1 + b2) / wlo) if wlo > 0 else UNRESOLVED
        out[f"{solver}_transformation_rel_down"] = en.down(max(Fraction(0), dlo - b1 - b2) / (whi + b1)) if wlo > 0 else UNRESOLVED
        out[f"{solver}_transformation_rel_up"] = en.up((dhi + b1 + b2) / max(wlo - b1, Fraction(0))) \
            if wlo > b1 else UNRESOLVED
        out[f"{solver}_separation"] = budget(b1 + b2, dlo)
    out["E3_status"] = ("Delta EXACT (exact products; formation excluded); solve bounds CERTIFIED (Q1b); transformation part = "
                        "interval; separation per D-049.6, item 4")
    return out


def e3_job(spec: dict) -> dict:
    """s = 1, N = 2, 3, both directions: T_N (sin^2) against its QR (normative), the product basis and a unit-triangular combination."""
    t0 = time.perf_counter()
    rows = []
    lx, nx, ly, ny = spec["lx"], spec["nx"], spec["ly"], spec["ny"]
    for direction in hs.DIRECTIONS:
        n_f = nx if direction == "Y" else ny
        h_f = (lx / nx) if direction == "Y" else (ly / ny)
        for N in (2, 3):
            G = hs.start_set(n_f, N)
            U = np.linalg.qr(math.sqrt(h_f) * G)[0] / math.sqrt(h_f)
            T = np.tril(np.ones((N, N)))
            for label, G1 in (("sin^2 vs QR(sin^2)", G), ("product basis vs QR(sin^2)", hs.start_set(n_f, N, "product")),
                              ("sin^2 T (unit lower triangular) vs QR(sin^2)", G @ T)):
                rows.append({"group": spec["group"], "grid": grid_label(lx, nx, ly, ny), "s": 1,
                             **e3_pair(lx, nx, ly, ny, direction, G1, U, label)})
    return {"kind": "e3", "spec": spec, "e3": rows, **_job_meta(t0)}


# --- E-5: DST-I -------------------------------------------------------------------------------------------------------------------

def e5_vectors(n: int):
    m = n - 1
    rng = np.random.default_rng(SEED + n)
    yield "e_1", np.eye(m)[0]
    yield "e_m", np.eye(m)[-1]
    yield "ones (F (x) 1 profile)", np.ones(m)
    yield "random normal", rng.standard_normal(m)
    yield "mixed scales 10^U(-8,8)", rng.standard_normal(m) * 10.0 ** rng.uniform(-8, 8, m)
    yield "psi_1 in double", bc.dst(np.eye(m)[0])
    x = np.linspace(0.0, 1.0, n + 1)[1:-1]
    yield "smooth clamped profile x^2 (1-x)^2", x**2 * (1 - x) ** 2


def subnormal(a: np.ndarray) -> bool:
    a = np.asarray(a, dtype=float)
    return bool(np.any((a != 0) & (np.abs(a) < 2.0**-1022)))


def e5_job(spec: dict) -> dict:
    t0 = time.perf_counter()
    n = spec["n"]
    rows = []
    for label, x in e5_vectors(n):
        y = bc.dst(x)
        ref = dc.certified_dst(x, max_entries=spec.get("max_entries", 2100**2), time_limit=spec.get("time_limit", 1800.0))
        row = {"n": n, "sample": spec["sample"], "vector": label, "reference_status": ref.status,
               "subnormal_input": subnormal(x), "subnormal_result": subnormal(y)}
        if ref.status == "CERTIFIED":
            c = dc.compare(y, ref, x)
            for k in ("abs_err_2_down", "abs_err_2_up", "rel_err_2_down", "rel_err_2_up", "ref_width_rel_up", "resolution",
                      "c_dst_ratio_down", "c_dst_ratio_up"):
                row[k] = c.get(k)
        else:
            row["resolution"] = f"{ref.status}: {ref.note}"
        row["in_c_dst_statistic"] = (ref.status == "CERTIFIED" and not row["subnormal_input"] and not row["subnormal_result"]
                                     and str(row.get("resolution", "")).startswith("RESOLVED"))
        row["E5_status"] = ("reference CERTIFIED (intervals); c_DST ratio OBSERVATION; outside P-3 (subnormal input or result) "
                            "excluded (U-P3)")
        rows.append(row)
    return {"kind": "e5", "spec": spec, "e5": rows, **_job_meta(t0)}


# --- output -------------------------------------------------------------------------------------------------------------------------

def fmt(key: str, v):
    if v is None:
        return ""
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, (float, np.floating)):
        v = float(v)
        if key.endswith("_up"):
            return hs.decimal_up(v, 17)
        if key.endswith("_down"):
            return hs.decimal_down(v, 17)
        return repr(v)
    return v


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys and not k.startswith("_")]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: fmt(k, r.get(k)) for k in keys})


def stats(values) -> dict:
    v = sorted(float(x) for x in values if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x))
    if not v:
        return {"count": 0}
    q = lambda f: v[min(len(v) - 1, int(math.ceil(f * len(v))) - 1)]  # noqa: E731 - nearest-rank quantile
    return {"count": len(v), "min": v[0], "median": statistics.median(v), "q95": q(0.95), "max": v[-1]}


def summarize(res: dict) -> dict:
    """Distributions (OBSERVATION) of the main measured quantities; no threshold is derived."""
    seq, e2r, e4, e6, e5, e8, e3 = (res.get(k, []) for k in ("seq", "e2", "e4", "e6", "e5", "e8", "e3"))
    out = {}
    for N in (1, 2, 3):
        rows = [r for r in seq if r.get("N") == N and r.get("association", "").startswith("VERIFIED")]
        out[f"E1_m_s_cR22_N{N}"] = stats(r.get("E1_m_s_cR22") for r in rows)
        out[f"E1_m_s_cR0_N{N}"] = stats(r.get("E1_m_s_cR0") for r in rows)
        out[f"E1_Q_over_tau_KE1_N{N}"] = stats(r.get("E1_Q_over_tau_KE1_cR22") for r in rows)
        out[f"E9_RQ_over_R_cR22_N{N}"] = stats(r.get("E9_RQ_over_R_cR22") for r in rows)
        out[f"E2_rel_e_a_up_N{N}"] = stats(r.get("rel_e_a_up") for r in e2r if r.get("N") == N)
        out[f"E2_e_a_over_eps_a_C_up_N{N}"] = stats(r.get("e_a_over_eps_a_C_up") for r in e2r if r.get("N") == N)
        out[f"E2_rel_e_a_over_HF1_model_N{N}"] = stats(r.get("H_F1_rel_e_a_over_model") for r in e2r if r.get("N") == N)
    out["E1_vok2_identity_exact_all"] = all(r.get("j_vok2_identity_exact") in (True, "True") for r in seq
                                            if "j_vok2_identity_exact" in r)
    out["E1_identity_rows"] = sum(1 for r in seq if "j_vok2_identity_exact" in r)
    out["E2_status_counts"] = {k: sum(1 for r in e2r if r.get("vok5_status") == k) for k in {r.get("vok5_status") for r in e2r}}
    out["E2_vok4_status_counts"] = {k: sum(1 for r in e2r if r.get("vok4_status") == k) for k in {r.get("vok4_status") for r in e2r}}
    out["E7_condA_over_inv_rho_sq"] = stats(r.get("E7_condA_over_inv_rho_sq") for r in seq)
    s1 = [r for r in e6 if r.get("s") == 1 and r.get("set") == "T_N (sin^2)" and r.get("N", 1) >= 2
          and isinstance(r.get("cond_A_before_QR"), float)]
    out["E7_s1_condA_before_QR_minus_1_plus_2N_rel"] = stats(abs(r["cond_A_before_QR"] - (1 + 2 * r["N"])) / (1 + 2 * r["N"])
                                                              for r in s1)
    for key in ("set_error_2_up", "field_error_a_up", "vok14_bound_2_up"):
        out[f"E4_{key}"] = stats(r.get(key) for r in e4)
    out["E4_budget_set_counts"] = {k: sum(1 for r in e4 if r.get("budget_set") == k) for k in {r.get("budget_set") for r in e4}}
    out["E4_vok14_consistent_counts"] = {str(k): sum(1 for r in e4 if r.get("vok14_consistent") == k)
                                         for k in {r.get("vok14_consistent") for r in e4}}
    for key in ("cond_S", "M_norm_2", "rho_0", "kappa_bar_c", "max_cond_Mk", "beta_U", "chi_U", "varrho", "cond_A_after_QR"):
        out[f"E6_{key}"] = stats(r.get(key) for r in e6)
    for key in ("set_error_rel_up", "field_error_a_rel_up"):
        out[f"E4_exact_tier_{key}"] = stats(r.get(key) for r in e4)
    for key in ("obs_set_error_rel_up", "obs_field_error_a_rel_up", "obs_vok14_bound_a_rel_up"):
        out[f"E4_{key}"] = stats(r.get(key) for r in e4)
    for key in ("vok10_check", "vok11_check", "vok12_check"):
        out[f"E6_{key}_counts"] = {k: sum(1 for r in e6 if str(r.get(key, "")).startswith(k))
                                   for k in ("HOLDS", "UNRESOLVED_ROUNDING", "NOT_APPLICABLE")}
    out["E5_c_dst_ratio_up_main"] = stats(r.get("c_dst_ratio_up") for r in e5 if r.get("in_c_dst_statistic")
                                          and r.get("sample") == "main")
    out["E5_c_dst_ratio_up_extended"] = stats(r.get("c_dst_ratio_up") for r in e5 if r.get("in_c_dst_statistic")
                                              and r.get("sample") != "main")
    out["E5_excluded"] = sum(1 for r in e5 if not r.get("in_c_dst_statistic"))
    out["E8_budget_counts"] = {k: sum(1 for r in e8 if r.get("budget_1_100") == k) for k in {r.get("budget_1_100") for r in e8}}
    out["E8_B_omega_over_true"] = stats(r.get("B_omega_over_true_down") for r in e8)
    out["E3_separation_counts"] = {k: sum(1 for r in e3 for s in ("R-AC", "B-C") if r.get(f"{s}_separation") == k)
                                   for k in {r.get(f"{s}_separation") for r in e3 for s in ("R-AC", "B-C")}}
    out["E3_RAC_transformation_rel_up"] = stats(r.get("R-AC_transformation_rel_up") for r in e3)
    out["hypotheses"] = hypotheses(e6, e5)
    return out


def _slope(points):
    """Least-squares slope of log y against log x (OBSERVATION); UNRESOLVED with fewer than 3 points."""
    pts = [(math.log(x), math.log(y)) for x, y in points if isinstance(y, float) and y > 0 and x > 0]
    if len(pts) < 3:
        return UNRESOLVED
    mx = sum(p[0] for p in pts) / len(pts)
    my = sum(p[1] for p in pts) / len(pts)
    return sum((p[0] - mx) * (p[1] - my) for p in pts) / sum((p[0] - mx) ** 2 for p in pts)


def hypotheses(e6, e5) -> dict:
    """RO 4.4 on the chain (s = 1, T_N, QR for N >= 2): H-M1 slope of ||M||_2 against n_s, H-S1 cond(S) after QR against n_s, H-S3
    ranges of rho_0 and kappa_bar_c, H-D1 c_DST against n. All OBSERVATION on the sample; no strict statement follows."""
    out = {}
    chain = [r for r in e6 if r.get("group") == "chain" and isinstance(r.get("M_norm_2"), float)]
    keys = sorted({(r["r"], r["N"], r["direction"]) for r in chain})
    out["H_M1_slope_M_norm_vs_n_s"] = {f"r={k[0]} N={k[1]} {k[2]}": _slope([(r["n_s"], r["M_norm_2"]) for r in chain
                                                                            if (r["r"], r["N"], r["direction"]) == k])
                                       for k in keys}
    out["H_S1_slope_cond_S_vs_n_s"] = {f"r={k[0]} N={k[1]} {k[2]}": _slope([(r["n_s"], r["cond_S"]) for r in chain
                                                                            if (r["r"], r["N"], r["direction"]) == k])
                                       for k in keys}
    out["H_S1_cond_S_by_N"] = {N: stats(r.get("cond_S") for r in chain if r["N"] == N) for N in (1, 2, 3)}
    out["H_S3_rho_0"] = stats(r.get("rho_0") for r in e6)
    out["H_S3_kappa_bar_c"] = stats(r.get("kappa_bar_c") for r in e6)
    out["H_D1_c_dst_by_n"] = {n: stats(r.get("c_dst_ratio_up") for r in e5 if r.get("n") == n and r.get("in_c_dst_statistic"))
                              for n in sorted({r.get("n") for r in e5})}
    return out


def environment() -> dict:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = "unknown", None
    prov = e2.provenance()
    return {"git_commit": commit, "git_dirty": dirty, "python": platform.python_version(), "platform": platform.platform(),
            "machine": platform.machine(), "processor": platform.processor(), "numpy": metadata.version("numpy"),
            "scipy": metadata.version("scipy"), "blas": prov.get("blas"), "fft": "scipy.fft (pocketfft) DST-I, norm='ortho'",
            "threads": {k: os.environ.get(k, "unset") for k in THREAD_VARS}, "cpu_count": os.cpu_count(),
            "ram_mb": total_ram_mb(), "ftz_daz_selftest": prov.get("ftz_daz_selftest"), "seed": SEED, "c_R_path_count": C_R_PATH,
            "e46_exact_limit": e46.EXACT_LIMIT}


# --- driver -------------------------------------------------------------------------------------------------------------------------

def _run(pool, fn, specs, *extra):
    results = pool.starmap(fn, [(s, *extra) for s in specs], chunksize=1)
    return results


def run_campaign(out: Path, profile: str = "full", workers: int = 8, arrays: Path = ARRAYS) -> dict:
    for k in THREAD_VARS:
        os.environ[k] = "1"  # inherited by the spawned workers: one BLAS thread, fixed order within a job
    t0 = time.time()
    arrays.mkdir(parents=True, exist_ok=True)
    ctx = multiprocessing.get_context("spawn")
    res = {k: [] for k in ("seq", "e2", "e3", "e4", "e5", "e6", "e8")}
    jobs = []
    seq_specs = sequence_specs(profile)
    if profile == "ci":
        for s in seq_specs:
            s["s_max"] = 3
    order = sorted(seq_specs, key=lambda s: -(s["nx"] * s["ny"]))  # largest first
    with ctx.Pool(workers, maxtasksperchild=1) as pool:
        seq_results = _run(pool, seq_job, order, str(arrays))
        hs_specs = halfstep_specs(profile)
        hs_results = _run(pool, hs_job, hs_specs)
        e3_specs = [{"group": "pairs", "lx": 1.0, "nx": nx, "ly": 1.0, "ny": ny} for nx, ny in PAIRS] if profile == "full" else \
            [{"group": "ci", "lx": 1.0, "nx": 10, "ly": 1.3, "ny": 9}]
        e3_results = _run(pool, e3_job, e3_specs)
        step_specs = [s for s in seq_specs if s["group"] in ("pairs", "ci")]
        step_results = _run(pool, seqstep_job, step_specs, str(arrays))
        e5_specs = [{"n": n, "sample": "main" if profile == "full" else "ci"} for n in (E5_MAIN if profile == "full" else [16])] + \
            ([{"n": n, "sample": "extended (OBSERVATION)"} for n in E5_EXTENDED] if profile == "full" else [])
        e5_results = _run(pool, e5_job, e5_specs)
    for r in seq_results:
        res["seq"] += r["seq"]
        res["e2"] += r["e2"]
    # E-8 per grid (sequential pool of 2: D2 factorizations are the largest memory users)
    grids = {}
    for s in seq_specs:
        grids.setdefault((s["lx"], s["nx"], s["ly"], s["ny"]), []).append(s)
    journal = {}
    for r in seq_results:
        journal[seq_file(r["spec"])] = r["seq"]
    e8_specs = [{"lx": k[0], "nx": k[1], "ly": k[2], "ny": k[3], "sequences": v,
                 "journal": {seq_file(s): journal.get(seq_file(s), []) for s in v}}
                for k, v in sorted(grids.items(), key=lambda kv: -(kv[0][1] * kv[0][3]))]
    big = [s for s in e8_specs if s["nx"] * s["ny"] >= BIG_GRID]
    small = [s for s in e8_specs if s["nx"] * s["ny"] < BIG_GRID]
    with ctx.Pool(1, maxtasksperchild=1) as pool:  # largest D2 factorizations one at a time (PL 6.3 memory rule)
        e8_results = _run(pool, e8_job, big, str(arrays))
    with ctx.Pool(min(4, workers), maxtasksperchild=1) as pool:
        e8_results += _run(pool, e8_job, small, str(arrays))
    for name, results in (("e4", hs_results), ("e6", hs_results), ("e3", e3_results), ("e5", e5_results), ("e8", e8_results)):
        for r in results:
            res[name] += r.get(name, [])
    for r in step_results:
        for name in ("e4", "e6", "e3"):
            res[name] += r.get(name, [])
    for group in (seq_results, hs_results, e3_results, step_results, e5_results, e8_results):
        for r in group:
            jobs.append({"kind": r["kind"], "spec": json.dumps({k: v for k, v in r["spec"].items()
                                                                 if k not in ("sequences", "journal")}, sort_keys=True),
                         "seconds": r["seconds"], "peak_mb": r["peak_mb"], "d2_seconds": r.get("d2_seconds")})
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "sequences_e1_e7_e9.csv", res["seq"])
    for name in ("e2", "e3", "e4", "e5", "e6", "e8"):
        write_csv(out / f"{name}.csv", res[name])
    write_csv(out / "resources.csv", jobs)
    summary = summarize(res)
    ram = total_ram_mb()
    peak = max((j["peak_mb"] or 0) for j in jobs) if jobs else 0
    meta = {"study": "R5-V Q3 pre-qualification measurements E-1...E-9 (QA references; not a qualification)",
            "profile": profile, "workers": workers, **environment(), "seconds_total": round(time.time() - t0),
            "resource_rule_PL_6_3": {"peak_job_mb": peak, "limit_mb_50pct_ram": ram / 2 if ram else None,
                                     "within_limit": (peak <= ram / 2) if ram else UNRESOLVED,
                                     "longest_job_s": max((j["seconds"] for j in jobs), default=0),
                                     "factorization_limit_s": 1800},
            "sample": {"ratios": list(RATIOS), "pairs": PAIRS, "chain": CHAIN, "chain_limit": {r: chain_limit(r) for r in RATIOS},
                       "e5_main": E5_MAIN, "e5_extended": E5_EXTENDED, "exact_tier": EXACT_TIER},
            "counts": {k: len(v) for k, v in res.items()},
            "printing": "*_up rounded up and *_down rounded down to 17 significant digits; other floats repr (exact double)",
            "arrays": str(arrays.relative_to(ROOT)) if arrays.is_relative_to(ROOT) else str(arrays)}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str) + "\n", encoding="utf-8")
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=str) + "\n", encoding="utf-8")
    return {"res": res, "summary": summary, "meta": meta}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("command", choices=("run",))
    parser.add_argument("--profile", choices=("full", "ci"), default="full")
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--arrays", default=str(ARRAYS))
    parser.add_argument("--workers", type=int, default=8)
    a = parser.parse_args(argv)
    r = run_campaign(Path(a.out), a.profile, a.workers, Path(a.arrays))
    print(json.dumps(r["meta"]["counts"]), r["meta"]["seconds_total"], "s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
