"""R5-V Q4, priority 1 (D-053.4): the energy control on the solutions of the research B-C with the C-3+ assembly - statistics m*_s, the
model R_Sigma and the resolvability of E1/E2 on the 984 fixed half-step inputs of Q3. QA research tool, not a qualification: K_E,
eps_floor^C, c_R and other thresholds are NOT assigned. No import of the working ``kkvim``; the prototype B-C, the energy instrument,
the earlier Q4 tools and all Q3 / Q4 data are used read-only (no file of them is changed).

Fixed BEFORE the campaign (this docstring and the constants are the protocol; owner's research route of 2026-10-27: C-3+ as the
research assembly of C, path B1 with the directed Gamma_up, the candidate tau = Gamma_up + K_E R_Sigma admitted for analysis as a
MODEL, m*_s and E1/E2 studied without K_E).

Sample. The 984 half-steps (123 sequences) of Q3 (docs/results/r5_cccc_q3_prequalification/sequences_e1_e7_e9.csv) with the stored
NPZ arrays of the Q3 campaign (read-only; sha256 of every file checked against docs/results/r5_cccc_q4_gamma_audit/meta.json before
any calculation). PAIRED design: every B-C half-step starts from the SAME input state as the R-AC half-step of Q3 (given set,
companion, previous field); it is NOT a sequence propagated by B-C, and no statistic here is a statistic of a B-C trajectory.

Input of the B-C half-step (prototype order NS 5.1, ``r5_cccc_bc.run_halfstep``; QR for N >= 2 as the prototype provides it):
- N = 1 (any s) and s = 1 (any N): the given set of the Q3 half-step as stored (s = 1: the start set T_N); companion as stored
  (s = 1: zero). For N >= 2 at s = 1 the prototype's QR yields U~ != T_N; the previous field is the zero field, so it is preserved
  exactly whatever the basis; the companion is transformed as C' = C R^T (= 0).
- N >= 2, s >= 2: the solved factor of s - 1 BEFORE the controller's transformation (stored s-1 arrays), on which the prototype
  applies its own Householder QR (U~, R); the companion is transformed as C' = fl(C_{s-1} R^T) (C_{s-1} the given set of s - 1),
  exactly as the Q3 controller transformed it. U~ and C' must reproduce the stored Q3 given set and companion BITWISE (checked per
  row); then the previous field is the Q3 previous field bitwise. If not, the row is UNRESOLVED (inputs are never replaced).
Only C is replaced (``replaced_c`` of scripts/r5_cccc_q4_c_assembly_e4_audit.py): C = C-3+ (``c3plus`` of the same script) of the
solved set; A, B, F and the modal algorithm are the prototype's. Per row: premises of C-3+, the exact check of the theorem
|C^_ij - C_ij| <= gamma_k sqrt(C_ii C_jj), k = m_f + 13, positive definiteness of C^ (exact), the B-C status (modal factorization).
Fields are compared ONLY as exact products on the grid (never factor arrays of different bases).

Exact gamma (path B1, proposal 4.5; QA-gamma3 tool ``gamma_b1``): gamma_s = h_s rho^T Delta with the exact R-HP residual rho of the
solved system (exact coefficients of U~) at the B-C solution and Delta = C' - V_BC, all binary64 arrays taken exactly. Directed
interval [gamma_down, gamma_up] (QA-gamma3 ``directed``, exact verification), Gamma_up = max(|gamma_down|, |gamma_up|). Second
implementation on every row: path A of the gamma audit (own integer arithmetic). Second MATHEMATICAL path (VOK-2 identity through
the independent two-dimensional energy, ``energies_path_b`` of the gamma audit): on the fixed control sample of the gamma audit (the
57 ids of its meta.json, fixed in e11f268 before that campaign) and, as an extended check, on every row. The paths are different
computations by ONE executor (QA); they are not an independent executor.

Computed check (the double path G-4 of r5_cccc_energy, unchanged: pi_double, q_double, fields fl(X Y^T)): P^_prev, P^_s, Q^ on
d^ = fl(w^_prev - w^_s); dP^ - Q^; the exact numerator |dP^ - Q^ + gamma_s| from the exact rationals of the stored doubles; its exact
split into formation (f(w^) - f(w~)) and assembly ((dP^ - Q^) - f(w^)) parts, f(v, w) = Pi(v) - Pi(w) - ||v - w||_a^2/2 (exact,
integer path on the double fields).

Model R_Sigma = R^C(w^_prev) + R^C(w^_s) + R_Q^C(d^) + R_f^C (VOK-3, VOK-8; MODEL), every term from the ACTUAL fields of this row with
the functions of r5_cccc_energy (model_R, model_RQ, eps_a_C, b_omega); c_R = 22 is the operation count of that double path
(en.C_R_COUNT, the path is the same as in Q3), not a normative coefficient. R_f^C = eps_a^C(s) ||d^||_a + B_omega (eps_a^C(s - 1) +
eps_a^C(s)); B_omega by VOK-7 (b) only (exact two-dimensional residual of the double field plus the formation bound; ``b_omega`` with
no certificate): the R-AC certificate is NOT used for the B-C field. Q3 grouping: R_Sigma = R_base_up + c_R R_per_cR_up + R_f_up (the
upward-rounded floats taken exactly). R_Sigma = 0: no division (status).

Statistics (OBSERVATION): m*_s = |dP^ - Q^ + gamma_s|/R_Sigma (directed interval). Test forms (D-048.1 form with the candidate tau):
E2 |dP^ - Q^| <= tau, E1 dP^ >= -tau, tau = Gamma_up + K_E R_Sigma, resolved iff Q^ >= K_E tau. Per row K_low = max(K_E1, K_E2) (exact
rational), K_max = the positive root of R K^2 + Gamma_up K - Q^ = 0 (binary64 enclosure verified by exact sign checks); classes:
INTERVAL (K_low <= K_max: E1/E2 hold and the test is resolved for K_low <= K_E <= K_max), EMPTY (K_low > K_max: for 0 < K_E <= K_max
the test is resolved and E1/E2 are violated; for K_E > K_max it is UNRESOLVED - NOT "unresolvable for every K_E"), BOUNDARY (K_low
inside the enclosure of K_max), NOT_APPLICABLE (R_Sigma = 0 or Q^ <= 0). No K_E is recommended.

Comparison with R-AC (Q3, published): dP^, Q^ and the R_Sigma terms from the Q3 CSV (directed decimals taken exactly), gamma exactly
(B1 recomputed and checked against QA-gamma3 gamma_intervals.csv), Gamma_up of QA-gamma3. Q3 chose B_omega = min(route (a) with the
VOK-14 certificate of R-AC, route (b)); route (a) was taken in 84 rows (all s = 1). For those rows the R-AC denominator with route (b)
is recomputed here, so that the comparison of the same model exists; both are reported and labelled. On the control sample the R-AC
denominators of Q3 are recomputed with this tool (Q3 route rule incl. the certificate) and must reproduce the Q3 CSV strings.

Diagnostics of the B-C solution (not used in B_omega): Q1b certificate lambda_low of the solved system, set bound, field bound
||w~_BC - w*||_a <= h_s^(1/2) ||r||_2/lambda_low^(1/2); eta_QA; exact ||w~_BC - w~_AC||_a; when the solved system is the R-AC system
(N = 1 or s >= 2) the exact consistency ||w~_BC - w~_AC||_a <= bound_BC + bound_AC (both bound the distance to the same w*).

Usage (repository root):
    uv run python scripts/r5_cccc_q4_energy_bc_audit.py run --profile full [--workers 12]
    uv run python scripts/r5_cccc_q4_energy_bc_audit.py run --profile ci --out scratch/r5_cccc_q4_energy_bc_ci
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
    os.environ[_k] = "1"  # as in Q3: one BLAS thread, the order of operations within a job is fixed

import numpy as np  # noqa: E402
import scipy  # noqa: E402

from kkvim_reference import r5_cccc_bc as bc  # noqa: E402
from kkvim_reference import r5_cccc_energy as en  # noqa: E402
from kkvim_reference import r5_cccc_halfstep as hs  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "results" / "r5_cccc_q4_energy_bc"
GAMMA_AUDIT = ROOT / "scripts" / "r5_cccc_q4_gamma_audit.py"
GAMMA_INTERVALS = ROOT / "scripts" / "r5_cccc_q4_gamma_interval_audit.py"
C_ASSEMBLY = ROOT / "scripts" / "r5_cccc_q4_c_assembly_e4_audit.py"
GA_META = ROOT / "docs" / "results" / "r5_cccc_q4_gamma_audit" / "meta.json"
GI_ROWS = ROOT / "docs" / "results" / "r5_cccc_q4_gamma_intervals" / "gamma_intervals.csv"
Q3_DIR = ROOT / "docs" / "results" / "r5_cccc_q3_prequalification"
Q3_CSV = Q3_DIR / "sequences_e1_e7_e9.csv"
Q3_ARRAYS_DEFAULT = Path(os.environ.get("KKVIM_Q3_ARRAYS", r"D:\git\KK-VIM-Q3-QA\scratch\r5_cccc_q3_prequalification\arrays"))
C_R = 22  # operation count of the double path G-4 of r5_cccc_energy (Q3); NOT a normative c_R
UNRESOLVED = "UNRESOLVED"
RESEARCH = "RESEARCH_ONLY / PROVISIONAL / NOT_QUALIFIED (D-049.5)"
CLASSES = ("INTERVAL", "EMPTY", "BOUNDARY", "NOT_APPLICABLE")


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ga():
    """Q4 gamma audit (Q3 reader, path A, path B, control rule, CI data); read-only use."""
    if not hasattr(_ga, "mod"):
        _ga.mod = _load(GAMMA_AUDIT, "r5_cccc_q4_gamma_audit")
    return _ga.mod


def _gi():
    """QA-gamma3 (path B1 gamma_b1, directed rounding); read-only use."""
    if not hasattr(_gi, "mod"):
        _gi.mod = _load(GAMMA_INTERVALS, "r5_cccc_q4_gamma_interval_audit")
    return _gi.mod


def _ca():
    """C-3+ audit (c3plus, premises, theorem check, SPD check, replaced_c); read-only use."""
    if not hasattr(_ca, "mod"):
        _ca.mod = _load(C_ASSEMBLY, "r5_cccc_q4_c_assembly_e4_audit")
    return _ca.mod


# ---------------------------------------------------------------------------------------------------------------------------------
# exact helpers
# ---------------------------------------------------------------------------------------------------------------------------------

def frac(x) -> Fraction:
    return Fraction(float(x))


def energies_fields(lx, nx, ly, ny, D, q, Wp: np.ndarray, Ws: np.ndarray):
    """(Pi(Wp), Pi(Ws), ||Wp - Ws||_a^2/2) EXACTLY for two double fields (the exact difference, not fl(Wp - Ws)); definition NS 2.2
    by the integer stencils of the gamma audit's path B (no coefficient, no K)."""
    ga = _ga()
    hx, hy = Fraction(lx) / nx, Fraction(ly) / ny
    (Pc, Sc), E = ga.dyadic(np.asarray(Wp, dtype=float), np.asarray(Ws, dtype=float))
    P = np.array(Pc, dtype=object).T
    S = np.array(Sc, dtype=object).T
    Sxp, Syp = ga._hat_parts(P, nx, ny)
    Sxs, Sys = ga._hat_parts(S, nx, ny)
    c4 = ga._weights4(nx, ny)
    Dq, qq = Fraction(D), Fraction(q)
    a_p = ga._a_form(Sxp, Syp, c4, hx, hy, Dq, E)
    a_s = ga._a_form(Sxs, Sys, c4, hx, hy, Dq, E)
    a_d = ga._a_form(Sxp - Sxs, Syp - Sys, c4, hx, hy, Dq, E)
    l_p = qq * hx * hy * int(P.sum()) * ga.p2(E)
    l_s = qq * hx * hy * int(S.sum()) * ga.p2(E)
    return a_p / 2 - l_p, a_s / 2 - l_s, a_d / 2


def k_interval(dP: Fraction, Q: Fraction, allowance: Fraction, R: Fraction) -> dict:
    """K_E data of the test form (module docstring): K_low exact, K_max enclosed in [K_max_down, K_max_up] by exact sign checks of
    f(K) = Q - allowance K - R K^2 (decreasing for K >= 0), and the class."""
    if R <= 0:
        return {"K_class": "NOT_APPLICABLE", "K_note": "R_Sigma = 0: no division"}
    k_e2 = max(Fraction(0), abs(dP - Q) - allowance) / R
    k_e1 = max(Fraction(0), -dP - allowance) / R
    k_low = max(k_e2, k_e1)
    out = {"K_E2": k_e2, "K_E1": k_e1, "K_low": k_low}
    if Q <= 0:
        out.update({"K_max_down": 0.0, "K_max_up": 0.0, "K_max_bracket_ok": True, "K_class": "NOT_APPLICABLE",
                    "K_note": "Q^ <= 0: resolved only for K_E = 0"})
        return out

    def f(k: float) -> Fraction:
        K = Fraction(k)
        return Q - allowance * K - R * K * K

    a, Rf, Qf = float(allowance), float(R), float(Q)
    k0 = 2 * Qf / (a + math.sqrt(a * a + 4 * Rf * Qf))  # stable form of the positive root
    lo = k0
    while f(lo) < 0:
        lo = math.nextafter(lo, 0.0)
    hi = lo if f(lo) == 0 else math.nextafter(lo, math.inf)
    while f(hi) > 0:
        lo, hi = hi, math.nextafter(hi, math.inf)
    ok = f(lo) >= 0 >= f(hi) and (lo == hi or math.nextafter(lo, math.inf) == hi)
    out.update({"K_max_down": lo, "K_max_up": hi, "K_max_bracket_ok": ok})
    if k_low <= Fraction(lo):
        out["K_class"] = "INTERVAL"
    elif k_low > Fraction(hi):
        out["K_class"] = "EMPTY"
    else:
        out["K_class"] = "BOUNDARY"
    return out


def directed_ratio(num: Fraction, den: Fraction) -> tuple[float, float]:
    """Tightest binary64 enclosure of num/den (den > 0), verified exactly (QA-gamma3 ``directed``)."""
    return _gi().directed(num / den)


# ---------------------------------------------------------------------------------------------------------------------------------
# B-C half-step on the Q3 input state
# ---------------------------------------------------------------------------------------------------------------------------------

def orient(direction: str, given, other):
    return (given, other) if direction == "Y" else (other, given)


def bc_input(spec: dict, steps: list[dict], i: int):
    """(HalfStep for run_halfstep, companion to transform, kind) of the paired B-C half-step i (module docstring)."""
    st = steps[i]
    N = st["given"].shape[1]
    lx, nx, ly, ny = spec["lx"], spec["nx"], spec["ly"], spec["ny"]
    if N == 1 or st["s"] == 1:
        kind = "N = 1: Q3 given set as stored" if N == 1 else "s = 1: start set T_N, prototype QR (U~ != T_N), zero companion"
        return hs.HalfStep(lx, nx, ly, ny, st["direction"], st["given"]), st["companion"], kind
    prev = steps[i - 1]
    return (hs.HalfStep(lx, nx, ly, ny, st["direction"], prev["V"]), prev["given"],
            "N >= 2, s >= 2: solved set of s - 1 before the controller's QR; prototype QR; companion C' = fl(C R^T)")


def solve_c3plus(solved: hs.HalfStep, coeffs) -> dict:
    """B-C with ONLY C replaced by C-3+ of the solved set; premises, exact theorem check, exact SPD check."""
    ca = _ca()
    parts = ca.c3plus(solved)
    prem = ca.premises_c3plus(parts)
    out = {"c3plus_premises": "HOLD" if prem is None else f"NOT_APPLICABLE: {prem}"}
    if prem is None:
        chk = ca.check_c3plus(parts["C"], coeffs[2], solved.n_f - 1 + ca.K_EXTRA)
        out["c3plus_theorem"] = "PASS" if (chk["elementwise"] and chk["frobenius"]) else "FAIL"
    else:
        out["c3plus_theorem"] = "NOT_APPLICABLE"
    out["c3plus_C_spd_exact"] = ca.spd_exact(parts["C"])
    with ca.replaced_c(parts["C"]):
        A, B, C, F = bc.bc_coefficients(solved)
        sol = bc.solve_bc(solved)
    A0, B0, _, F0 = bc.bc_coefficients(solved)
    out["only_C_replaced"] = bool(np.array_equal(A, A0) and np.array_equal(B, B0) and np.array_equal(F, F0)
                                  and np.array_equal(C, parts["C"]))
    out["bc_status"] = sol.status
    out["_sol"] = sol
    if sol.s is not None:
        out.update({"cond_S": sol.diagnostics.get("cond_S"), "max_cond_Mk": sol.diagnostics.get("max_cond_Mk"),
                    "cond_A_solved": sol.diagnostics.get("cond_A")})
    return out


def bc_certificate(solved: hs.HalfStep, s_vec: np.ndarray, coeffs) -> dict:
    """Q1b certificate of the solved system at the B-C solution (diagnostic; not used in B_omega)."""
    tau = bc._tau(solved)
    low = None if tau is None else hs.certify_lambda_min(solved, tau, coeffs)
    if low is None:
        return {"bc_certificate": "CERT_UNRESOLVED (lambda_min not certified)"}
    r = hs.residual_exact(solved, s_vec, coeffs)
    r2 = sum((x * x for x in r), Fraction(0))
    field = en.sqrt_interval(Fraction(solved.l_s) / solved.n_s)[1] * en.sqrt_interval(r2)[1] / en.sqrt_interval(Fraction(low))[0]
    return {"bc_certificate": "CERTIFIED (Q1b)", "bc_lambda_low": low, "bc_set_bound_2_up": hs.error_bound(solved, s_vec, low, coeffs),
            "_bc_field_bound": field, "bc_field_bound_a_up": en.up(field), "bc_eta_qa": hs.eta_qa(solved, s_vec, coeffs)}


# ---------------------------------------------------------------------------------------------------------------------------------
# model R_Sigma of a pair of fields (r5_cccc_energy, as in r5_cccc_sequence._energy)
# ---------------------------------------------------------------------------------------------------------------------------------

def model_terms(g: en.Grid, Xp, Yp, Xs, Ys, field_cert: Fraction | None = None) -> dict:
    """R^C(w^_prev), R^C(w^_s), R_Q^C(d^), R_f^C with B_omega by VOK-7 (b) (field_cert None) or by the Q3 rule (min with route (a)
    and the given VOK-14 certificate, only to reproduce Q3). Returns exact values and the Q3-grouped upward floats."""
    wh_prev, wh_s = Xp @ Yp.T, Xs @ Ys.T
    dh = wh_prev - wh_s
    Rp, Rs, RQ = en.model_R(g, wh_prev), en.model_R(g, wh_s), en.model_RQ(g, dh)
    eps_s = en.eps_a_C(g, Xs, Ys)
    eps_prev = Fraction(0) if not np.any(Xp @ Yp.T) else en.eps_a_C(g, Xp, Yp)
    Bw, how = en.b_omega(g, wh_s, eps_s, field_cert)
    d_a = en.sqrt_interval(en.norm_a_sq_exact(g, [[Fraction(float(x)) for x in row] for row in dh]))[1]
    Rf = eps_s * d_a + Bw * (eps_prev + eps_s)
    out = {"R_base_up": en.up(Fraction(Rp["R_base_up"]) + Fraction(Rs["R_base_up"]) + Fraction(RQ["RQ_base_up"])),
           "R_per_cR_up": en.up(Fraction(Rp["R_per_cR_up"]) + Fraction(Rs["R_per_cR_up"]) + Fraction(RQ["RQ_per_cR_up"])),
           "R_f_up": en.up(Rf), "B_omega_up": en.up(Bw), "B_omega_route": how, "eps_a_C_s_up": en.up(eps_s),
           "eps_a_C_prev_up": en.up(eps_prev), "d_hat_a_up": en.up(d_a),
           "R_C_prev_up": en.up(Fraction(Rp["R_base_up"]) + C_R * Fraction(Rp["R_per_cR_up"])),
           "R_C_s_up": en.up(Fraction(Rs["R_base_up"]) + C_R * Fraction(Rs["R_per_cR_up"])),
           "R_Q_up": en.up(Fraction(RQ["RQ_base_up"]) + C_R * Fraction(RQ["RQ_per_cR_up"]))}
    out["_R_Sigma"] = Fraction(out["R_base_up"]) + C_R * Fraction(out["R_per_cR_up"]) + Fraction(out["R_f_up"])
    out["_Rf_parts"] = (eps_s, eps_prev, d_a, Bw)
    return out


def q3_R(row: dict) -> Fraction:
    ga = _ga()
    return ga.read_bound(row["j_R_base_up"]) + C_R * ga.read_bound(row["j_R_per_cR_up"]) + ga.read_bound(row["j_R_f_up"])


# ---------------------------------------------------------------------------------------------------------------------------------
# one half-step
# ---------------------------------------------------------------------------------------------------------------------------------

def analyse_step(spec: dict, steps: list[dict], i: int, row: dict, gi_row: dict | None, control: bool, path_b_all: bool) -> dict:
    ga, gi = _ga(), _gi()
    st = steps[i]
    lx, nx, ly, ny = spec["lx"], spec["nx"], spec["ly"], spec["ny"]
    direction, N, s = st["direction"], st["given"].shape[1], st["s"]
    g = en.Grid(lx, nx, ly, ny)
    rid = ga.row_id(row)
    out = {"id": rid, "group": spec["group"], "grid": row["grid"], "N": N, "first": spec["first"], "s": s, "direction": direction,
           "n_f": int(row["j_n_f"]), "n_s": int(row["j_n_s"]), "control": control,
           "marking": RESEARCH if N >= 2 else "N = 1 (OBSERVATION)"}
    fails, unres = [], []
    # --- 1. the input belongs to the fixed Q3 sample and reproduces the stored data
    out["in_q3_sample"] = True
    out["given_consistent"] = st["given_consistent"]
    wh_prev_ac, wh_s_ac = st["prev"][0] @ st["prev"][1].T, st["now"][0] @ st["now"][1].T
    Pp_ac, Ps_ac = en.pi_double(g, wh_prev_ac), en.pi_double(g, wh_s_ac)
    Q_ac_d = en.q_double(g, wh_prev_ac - wh_s_ac)
    out["q3_P_double_bitwise"] = (Pp_ac == ga.read_double(row["j_P_double_prev"]) and Ps_ac == ga.read_double(row["j_P_double_s"])
                                  and Q_ac_d == ga.read_double(row["j_Q_double"]))
    g_ac = gi.gamma_b1(lx, nx, ly, ny, direction, st["given"], st["companion"], st["V"])
    out["q3_gamma_exact_nearest"] = float(g_ac) == ga.read_double(row["j_gamma_exact"])
    if gi_row is not None:
        out["gamma_intervals_exact_equal"] = gi_row["gamma_exact"] == f"{g_ac.numerator}/{g_ac.denominator}"
    else:
        out["gamma_intervals_exact_equal"] = "NOT_APPLICABLE (CI profile)"
    for k in ("given_consistent", "q3_P_double_bitwise", "q3_gamma_exact_nearest"):
        if out[k] is not True:
            unres.append(f"input mapping: {k}")
    if out["gamma_intervals_exact_equal"] is False:
        unres.append("input mapping: gamma_intervals_exact_equal")
    # --- 2. the B-C half-step on that input
    p_in, comp_in, kind = bc_input(spec, steps, i)
    out["bc_input"] = kind
    rec = bc.run_halfstep(p_in, qr=N >= 2)
    out["prototype_qr"] = rec.log["qr"]
    out["prototype_set_status"] = rec.log.get("set_status")
    if rec.solved is None:
        out["status"] = f"{UNRESOLVED}: given set exactly dependent (prototype)"
        return out
    solved = rec.solved
    Cp = comp_in @ rec.R.T if rec.R is not None else np.asarray(comp_in, dtype=float)
    if N >= 2 and s >= 2:
        out["q3_state_reproduced_bitwise"] = bool(np.array_equal(solved.given, st["given"]) and np.array_equal(Cp, st["companion"]))
        out["prev_field_check"] = "BITWISE (arrays = Q3)" if out["q3_state_reproduced_bitwise"] else UNRESOLVED
    elif s == 1:
        out["q3_state_reproduced_bitwise"] = bool(np.array_equal(solved.given, st["given"])) if N == 1 else \
            "NOT_APPLICABLE (s = 1, N >= 2: QR basis U~ != T_N)"
        out["prev_field_check"] = "ZERO FIELD (exactly)" if not np.any(Cp) and not np.any(st["companion"]) else UNRESOLVED
    else:
        out["q3_state_reproduced_bitwise"] = bool(np.array_equal(solved.given, st["given"]) and np.array_equal(Cp, st["companion"]))
        out["prev_field_check"] = "BITWISE (arrays = Q3)" if out["q3_state_reproduced_bitwise"] else UNRESOLVED
    if out["prev_field_check"] == UNRESOLVED or out["q3_state_reproduced_bitwise"] is False:
        unres.append("input state of B-C does not reproduce the Q3 state")
    if rec.log.get("qr"):
        out["orthonormality_defect_up"] = rec.log.get("orthonormality_defect_up")
        out["sin_angle_span_G_Utilde"] = rec.log.get("sin_angle_span_G_Utilde")
    coeffs = hs.exact_coefficients(solved)
    bcres = solve_c3plus(solved, coeffs)
    sol = bcres.pop("_sol")
    out.update(bcres)
    if sol.s is None:
        out["status"] = f"{UNRESOLVED}: B-C produced no solution ({sol.status})"
        return out
    if out["only_C_replaced"] is not True:
        fails.append("only C replaced")
    if out["c3plus_theorem"] == "FAIL":
        fails.append("C-3+ theorem")
    m = solved.n_s - 1
    V = sol.s.reshape(N, m).T
    Xp, Yp = orient(direction, solved.given, Cp)
    Xs, Ys = orient(direction, solved.given, V)
    # --- 3. exact gamma (B1), directed interval, second implementation (path A), second path (2-D energy)
    gam = gi.gamma_b1(lx, nx, ly, ny, direction, solved.given, Cp, V)
    gd, gu = gi.directed(gam)
    G_up = gi.gamma_up(gd, gu)
    out.update({"gamma_down": gd, "gamma_up": gu, "Gamma_up": G_up, "gamma_inclusion_exact": gi.verify(gam, gd, gu) is True,
                "Gamma_up_holds": gi.gamma_up_holds(gam, G_up)})
    if not (out["gamma_inclusion_exact"] and out["Gamma_up_holds"]):
        fails.append("gamma inclusion")
    lf, nf, ls, ns = (lx, nx, ly, ny) if direction == "Y" else (ly, ny, lx, nx)
    (Gi,), EG = ga.dyadic(solved.given)
    (Vi, Ci), EV = ga.dyadic(V, Cp)
    gA = ga.gamma_from_coefficients(ga.coefficients_exact(Gi, EG, Fraction(lf) / nf, Fraction(1)), Vi, Ci, EV,
                                    Fraction(ls) / ns, Fraction(1))
    out["gamma_equals_path_A"] = gA == gam
    if not out["gamma_equals_path_A"]:
        fails.append("gamma B1 != path A")
    if control or path_b_all:
        Pp_b, Ps_b, Q_b = ga.energies_path_b(lx, nx, ly, ny, 1.0, 1.0, Xp, Yp, Xs, Ys)
        out["vok2_identity_energy_path"] = Q_b - (Pp_b - Ps_b) == gam
        out["P_exact_prev_in_q3_interval"] = ga.read_bound(row["j_P_exact_prev_down"]) <= Pp_b <= ga.read_bound(row["j_P_exact_prev_up"])
        if not out["vok2_identity_energy_path"]:
            fails.append("VOK-2 identity (2-D energy path)")
        if not out["P_exact_prev_in_q3_interval"]:
            fails.append("previous exact energy differs from Q3")
    # --- 4. computed check in double and the exact numerator
    wh_prev, wh_s = Xp @ Yp.T, Xs @ Ys.T
    dh = wh_prev - wh_s
    Pp, Ps, Qd = en.pi_double(g, wh_prev), en.pi_double(g, wh_s), en.q_double(g, dh)
    out["P_hat_prev_equals_q3"] = Pp == ga.read_double(row["j_P_double_prev"])
    if not out["P_hat_prev_equals_q3"]:
        unres.append("P^_prev differs from Q3")
    dP = frac(Pp) - frac(Ps)
    Qh = frac(Qd)
    computed = dP - Qh
    num = abs(computed + gam)
    Pp_x, Ps_x, Qx = energies_fields(lx, nx, ly, ny, 1.0, 1.0, wh_prev, wh_s)
    f_hat = Pp_x - Ps_x - Qx
    part_form, part_asm = f_hat + gam, computed - f_hat
    out.update({"P_hat_prev": Pp, "P_hat_s": Ps, "Q_hat": Qd, "dP_hat": float(dP), "dP_hat_minus_Q_hat": float(computed),
                "dP_negative": dP < 0, "gamma_over_Q": float(gam / Qh) if Qh else UNRESOLVED,
                "numerator": float(num), "part_formation": float(part_form), "part_assembly": float(part_asm),
                "numerator_split_exact": num == abs(part_form + part_asm)})
    if not out["numerator_split_exact"]:
        fails.append("numerator split")
    # --- 5. model R_Sigma of the actual fields (B_omega by VOK-7 (b))
    mt = model_terms(g, Xp, Yp, Xs, Ys)
    R = mt.pop("_R_Sigma")
    mt.pop("_Rf_parts")
    out.update(mt)
    out["R_Sigma_up"] = en.up(R)
    out["model_status"] = "MODEL (VOK-3, VOK-8); c_R = 22 path count, K_E not assigned"
    if R > 0:
        md, mu = directed_ratio(num, R)
        out.update({"m_star_down": md, "m_star_up": mu, "m_star_interval_ok": gi.verify(num / R, md, mu) is True,
                    "Gamma_up_over_R": float(Fraction(G_up) / R), "Q_over_R": float(Qh / R),
                    "formation_over_Rf": float(abs(part_form) / Fraction(out["R_f_up"])) if out["R_f_up"] else UNRESOLVED,
                    "assembly_over_R_C": float(abs(part_asm) / (Fraction(out["R_base_up"]) + C_R * Fraction(out["R_per_cR_up"]))),
                    "test_excess_over_R": float(max(Fraction(0), abs(computed) - Fraction(G_up)) / R)})
        if not out["m_star_interval_ok"]:
            fails.append("m* interval")
    else:
        out["m_star"] = "NOT_APPLICABLE: R_Sigma = 0"
    out["Gamma_up_over_Q"] = float(Fraction(G_up) / Qh) if Qh > 0 else UNRESOLVED
    kb = k_interval(dP, Qh, Fraction(G_up), R)
    out.update({f"BC_{k}": (float(v) if isinstance(v, Fraction) else v) for k, v in kb.items()})
    if kb.get("K_max_bracket_ok") is False:
        fails.append("K_max enclosure")
    # --- 6. R-AC comparison (Q3 values; route (b) recomputed where Q3 took route (a))
    dP_ac = frac(Pp_ac) - frac(Ps_ac)
    Q_ac = frac(Q_ac_d)  # = the Q3 double bitwise (checked above); a repr string is NOT read as an exact decimal
    num_ac = abs(dP_ac - Q_ac + g_ac)
    G_ac = gi.gamma_up(*gi.directed(g_ac))
    if gi_row is not None and gi_row.get("Gamma_up"):
        out["gamma_intervals_Gamma_up_equal"] = float(gi_row["Gamma_up"]) == G_ac
    R_ac_q3 = q3_R(row)
    out["AC_B_omega_route_q3"] = row["j_B_omega_route"]
    if row["j_B_omega_route"].startswith("VOK-7 (b)"):
        R_ac_b = R_ac_q3
        out["AC_R_Sigma_b_source"] = "Q3 CSV (route (b))"
    else:  # route (a) in Q3: recompute R_f^C of the R-AC field with route (b); the other terms from the Q3 CSV
        eps_s_ac = en.eps_a_C(g, *st["now"])
        eps_p_ac = Fraction(0) if not np.any(wh_prev_ac) else en.eps_a_C(g, *st["prev"])
        Bw_ac, _ = en.b_omega(g, wh_s_ac, eps_s_ac, None)
        d_ac = en.sqrt_interval(en.norm_a_sq_exact(g, [[Fraction(float(x)) for x in r] for r in wh_prev_ac - wh_s_ac]))[1]
        Rf_b = en.up(eps_s_ac * d_ac + Bw_ac * (eps_p_ac + eps_s_ac))
        R_ac_b = ga.read_bound(row["j_R_base_up"]) + C_R * ga.read_bound(row["j_R_per_cR_up"]) + Fraction(Rf_b)
        out["AC_B_omega_b_up"] = en.up(Bw_ac)
        out["AC_R_f_b_up"] = Rf_b
        out["AC_R_Sigma_b_source"] = "Q3 CSV R_base, R_per_cR; R_f^C recomputed with route (b)"
    out.update({"AC_dP_hat": float(dP_ac), "AC_Q_hat": float(Q_ac), "AC_Gamma_up": G_ac, "AC_numerator": float(num_ac),
                "AC_R_Sigma_q3_up": en.up(R_ac_q3), "AC_R_Sigma_b_up": en.up(R_ac_b),
                "AC_m_star_q3": float(num_ac / R_ac_q3), "AC_m_star_b": float(num_ac / R_ac_b),
                "AC_Gamma_up_over_R_b": float(Fraction(G_ac) / R_ac_b), "AC_Q_over_R_b": float(Q_ac / R_ac_b),
                "BC_over_AC_gamma_abs": float(abs(gam) / abs(g_ac)) if g_ac else UNRESOLVED,
                "BC_over_AC_R_Sigma_b": float(R / R_ac_b), "BC_over_AC_numerator": float(num / num_ac) if num_ac else UNRESOLVED,
                "BC_minus_AC_Q_rel": float((Qh - Q_ac) / Q_ac) if Q_ac else UNRESOLVED})
    for tag, Rx in (("AC_b", R_ac_b), ("AC_q3", R_ac_q3)):
        kx = k_interval(dP_ac, Q_ac, Fraction(G_ac), Rx)
        out.update({f"{tag}_{k}": (float(v) if isinstance(v, Fraction) else v) for k, v in kx.items()
                    if k in ("K_low", "K_max_down", "K_max_up", "K_class")})
    # control: the tool reproduces the Q3 R-AC denominators (Q3 route rule, incl. the R-AC certificate for route (a))
    if control:
        p_ac = hs.HalfStep(lx, nx, ly, ny, direction, st["given"])
        low = ga.read_double(row["j_lambda_low"])
        r_ac = hs.residual_exact(p_ac, st["V"].T.reshape(-1))
        cert = en.sqrt_interval(Fraction(p_ac.l_s) / p_ac.n_s)[1] * en.sqrt_interval(sum((x * x for x in r_ac), Fraction(0)))[1] \
            / en.sqrt_interval(Fraction(low))[0]
        mac = model_terms(g, *st["prev"], *st["now"], field_cert=cert)
        rep = all(hs.decimal_up(mac[k], 17) == row[f"j_{k}"] for k in ("R_base_up", "R_per_cR_up", "R_f_up", "B_omega_up"))
        out["control_q3_R_reproduced"] = rep and mac["B_omega_route"] == row["j_B_omega_route"]
        out["control_R_C_prev_same_as_AC"] = mac["R_C_prev_up"] == out["R_C_prev_up"] and mac["eps_a_C_prev_up"] == out["eps_a_C_prev_up"]
        if not out["control_q3_R_reproduced"]:
            fails.append("Q3 R-AC denominators not reproduced")
        if not out["control_R_C_prev_same_as_AC"]:
            fails.append("R^C(w^_prev) of B-C differs from R-AC")
    # --- 7. diagnostics of the B-C solution
    cert = bc_certificate(solved, sol.s, coeffs)
    fb = cert.pop("_bc_field_bound", None)
    out.update(cert)
    _, _, half_d2 = ga.energies_path_b(lx, nx, ly, ny, 1.0, 1.0, *st["now"], Xs, Ys)
    lo, hi = en.sqrt_interval(2 * half_d2)
    out["field_diff_BC_AC_a_down"], out["field_diff_BC_AC_a_up"] = en.down(lo), en.up(hi)
    same_system = N == 1 or s >= 2
    if same_system and fb is not None:
        b_ac = ga.read_bound(row["j_field_bound_a_exact_up"])
        out["bc_ac_certified_consistent"] = lo <= fb + b_ac
        if not out["bc_ac_certified_consistent"]:
            fails.append("B-C / R-AC fields inconsistent with the certificates")
    else:
        out["bc_ac_certified_consistent"] = "NOT_APPLICABLE (different solved systems: s = 1, N >= 2)" if not same_system else \
            UNRESOLVED
    # --- status
    if fails:
        out["status"] = "FAIL: " + "; ".join(fails)
    elif unres:
        out["status"] = f"{UNRESOLVED}: " + "; ".join(unres)
    else:
        out["status"] = "COMPLETE"
    return out


def seq_job(spec: dict, rows: list[dict], npz_path: str, control_ids: list[str], gi_rows: dict, path_b_all: bool) -> dict:
    ga = _ga()
    t0 = time.perf_counter()
    z = np.load(npz_path)
    steps = ga._steps_from_npz(z, spec)
    by_s = {int(r["j_s"]): r for r in rows}
    out = []
    for i, st in enumerate(steps):
        row = by_s[st["s"]]
        rid = ga.row_id(row)
        try:
            out.append(analyse_step(spec, steps, i, row, gi_rows.get(rid), rid in control_ids, path_b_all))
        except Exception as exc:  # recorded, never hidden: the row is UNRESOLVED with the reason
            out.append({"id": rid, "group": spec["group"], "grid": row["grid"], "N": spec["N"], "first": spec["first"],
                        "s": st["s"], "direction": st["direction"], "status": f"{UNRESOLVED}: exception {type(exc).__name__}: {exc}"})
    return {"rows": out, "spec": spec, "npz": Path(npz_path).name, "npz_sha256": sha256(Path(npz_path)), "steps": len(steps),
            "seconds": round(time.perf_counter() - t0, 2), "peak_mb": ga._q3().peak_mb()}


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
    return {"count": len(v), "min": v[0], "median": statistics.median(v), "q95": q(0.95), "q99": q(0.99), "max": v[-1]}


STAT_KEYS = ("m_star_up", "R_Sigma_up", "Gamma_up_over_R", "Gamma_up_over_Q", "Q_over_R", "test_excess_over_R", "formation_over_Rf",
             "assembly_over_R_C", "BC_K_low", "BC_K_max_down", "AC_m_star_b", "AC_m_star_q3", "AC_Gamma_up_over_R_b", "AC_Q_over_R_b",
             "AC_b_K_low", "AC_b_K_max_down", "BC_over_AC_gamma_abs", "BC_over_AC_R_Sigma_b", "BC_over_AC_numerator",
             "BC_minus_AC_Q_rel", "bc_eta_qa", "cond_S")


def _group(rows, key_fn):
    groups = {}
    for r in rows:
        groups.setdefault(key_fn(r), []).append(r)
    return groups


def _block(sub: list[dict]) -> dict:
    blk = {"rows": len(sub)}
    for k in ("m_star_up", "R_Sigma_up", "Gamma_up_over_R", "Q_over_R", "BC_K_low", "BC_K_max_down", "AC_m_star_b", "AC_b_K_low",
              "AC_b_K_max_down", "BC_over_AC_gamma_abs", "BC_over_AC_R_Sigma_b"):
        blk[k] = stats(r.get(k) for r in sub)
    blk["BC_K_class"] = {c: sum(1 for r in sub if r.get("BC_K_class") == c) for c in CLASSES}
    blk["AC_b_K_class"] = {c: sum(1 for r in sub if r.get("AC_b_K_class") == c) for c in CLASSES}
    blk["dP_negative"] = sum(1 for r in sub if r.get("dP_negative") is True)
    return blk


def summarize(rows: list[dict]) -> dict:
    done = [r for r in rows if r.get("status") == "COMPLETE"]
    out = {"rows": len(rows), "complete": len(done),
           "status_counts": {k: sum(1 for r in rows if r.get("status") == k) for k in sorted({r.get("status") for r in rows})}}
    for k in ("in_q3_sample", "given_consistent", "q3_P_double_bitwise", "q3_gamma_exact_nearest", "gamma_intervals_exact_equal",
              "gamma_intervals_Gamma_up_equal", "q3_state_reproduced_bitwise", "only_C_replaced", "c3plus_C_spd_exact",
              "gamma_inclusion_exact", "Gamma_up_holds", "gamma_equals_path_A", "vok2_identity_energy_path",
              "P_exact_prev_in_q3_interval", "P_hat_prev_equals_q3", "numerator_split_exact", "m_star_interval_ok",
              "BC_K_max_bracket_ok", "control_q3_R_reproduced", "control_R_C_prev_same_as_AC", "bc_ac_certified_consistent"):
        out[k] = f"{sum(1 for r in rows if r.get(k) is True)}/{sum(1 for r in rows if k in r)}"
    ctrl = [r for r in rows if r.get("control")]
    out["control_rows"] = len(ctrl)
    out["control_vok2_identity_energy_path"] = f"{sum(1 for r in ctrl if r.get('vok2_identity_energy_path') is True)}/{len(ctrl)}"
    for k in ("c3plus_premises", "c3plus_theorem", "bc_status", "bc_certificate", "prev_field_check", "bc_input",
              "AC_B_omega_route_q3", "B_omega_route"):
        out[k] = {v: sum(1 for r in rows if r.get(k) == v) for v in sorted({str(r.get(k)) for r in rows if k in r})}
    out["all"] = {k: stats(r.get(k) for r in done) for k in STAT_KEYS}
    out["BC_K_class"] = {c: sum(1 for r in done if r.get("BC_K_class") == c) for c in CLASSES}
    out["AC_b_K_class"] = {c: sum(1 for r in done if r.get("AC_b_K_class") == c) for c in CLASSES}
    out["AC_q3_K_class"] = {c: sum(1 for r in done if r.get("AC_q3_K_class") == c) for c in CLASSES}
    out["dP_negative"] = {"BC": sum(1 for r in done if r.get("dP_negative") is True),
                          "AC": sum(1 for r in done if isinstance(r.get("AC_dP_hat"), float) and r["AC_dP_hat"] < 0)}
    out["by_s"] = {str(k): _block(v) for k, v in sorted(_group(done, lambda r: r["s"]).items())}
    out["by_N"] = {str(k): _block(v) for k, v in sorted(_group(done, lambda r: r["N"]).items())}
    out["by_direction"] = {k: _block(v) for k, v in sorted(_group(done, lambda r: r["direction"]).items())}
    out["by_first"] = {k: _block(v) for k, v in sorted(_group(done, lambda r: r["first"]).items())}
    out["by_group"] = {k: _block(v) for k, v in sorted(_group(done, lambda r: r["group"]).items())}
    out["by_grid"] = {k: _block(v) for k, v in sorted(_group(done, lambda r: r["grid"]).items())}
    out["by_s_N"] = {f"s{k[0]}_N{k[1]}": _block(v) for k, v in sorted(_group(done, lambda r: (r["s"], r["N"])).items())}
    # class transitions R-AC (route (b)) -> B-C, and the change of K_max
    trans = {}
    for r in done:
        key = f"{r.get('AC_b_K_class')} -> {r.get('BC_K_class')}"
        trans[key] = trans.get(key, 0) + 1
    out["class_transitions_AC_b_to_BC"] = dict(sorted(trans.items()))
    rel = [(r["AC_b_K_max_down"] - r["BC_K_max_down"]) / r["AC_b_K_max_down"] for r in done
           if isinstance(r.get("AC_b_K_max_down"), float) and isinstance(r.get("BC_K_max_down"), float) and r["AC_b_K_max_down"] > 0]
    out["K_max_relative_loss_AC_to_BC"] = stats(rel)
    out["K_max_relative_loss_by_s"] = {str(s): stats((r["AC_b_K_max_down"] - r["BC_K_max_down"]) / r["AC_b_K_max_down"]
                                                     for r in sub if isinstance(r.get("BC_K_max_down"), float)
                                                     and isinstance(r.get("AC_b_K_max_down"), float) and r["AC_b_K_max_down"] > 0)
                                       for s, sub in sorted(_group(done, lambda r: r["s"]).items())}
    empty = [r for r in done if r.get("BC_K_class") == "EMPTY"]
    out["BC_empty_rows"] = [{"id": r["id"], "K_low": r["BC_K_low"], "K_max_up": r["BC_K_max_up"], "dP_negative": r["dP_negative"]}
                            for r in empty]
    out["max_rows"] = {k: max(((r.get(k), r["id"]) for r in done if isinstance(r.get(k), float)), default=None)
                       for k in ("m_star_up", "Gamma_up_over_R", "AC_m_star_b", "BC_over_AC_gamma_abs")}
    out["failures"] = [{"id": r["id"], "status": r["status"]} for r in rows if r.get("status") != "COMPLETE"]
    return out


def ci_specs_rows(tmp: Path):
    """Deterministic small CI data in the Q3 format (gamma audit ci_data: Q3 seq_job on small grids, N = 1-3, both directions)."""
    ga = _ga()
    csv_path, arrays = ga.ci_data(tmp)
    return ga.read_q3_rows(csv_path), arrays, csv_path


def run_campaign(profile: str, out: Path, workers: int, arrays: Path, path_b_all: bool) -> dict:
    t0 = time.time()
    out.mkdir(parents=True, exist_ok=True)
    ga = _ga()
    q3 = ga._q3()
    tmp = None
    if profile == "ci":
        tmp = Path(tempfile.mkdtemp(prefix="r5q4ebc_"))
        rows, arrays, csv_path = ci_specs_rows(tmp)
        control = {ga.row_id(r) for r in rows}
        gi_rows, npz_expected = {}, {}
        prov = {"csv": "generated by the Q3 seq_job (CI profile)", "sha256": sha256(csv_path)}
    else:
        rows = ga.read_q3_rows(Q3_CSV)
        meta_ga = json.loads(GA_META.read_text(encoding="utf-8"))
        control = set(meta_ga["control_ids"])
        npz_expected = {e["file"]: e["sha256"] for e in meta_ga["npz"]}
        with open(GI_ROWS, encoding="utf-8") as f:
            gi_rows = {r["id"]: r for r in csv.DictReader(f)}
        prov = {"q3_csv": ga.provenance(Q3_CSV), "gamma_audit_meta_sha256": sha256(GA_META),
                "gamma_intervals_csv_sha256": sha256(GI_ROWS), "gamma_intervals_rows": len(gi_rows)}
    groups = {}
    for r in rows:
        groups.setdefault(ga.seq_key(r), []).append(r)
    jobs = []
    npz_check = []
    for rs in groups.values():
        spec = ga.spec_of(rs[0]) if profile == "full" else ga._ci_spec(rs[0])
        npz = arrays / q3.seq_file(spec)
        if profile == "full":  # sha256 of every NPZ before any calculation
            got = sha256(npz)
            npz_check.append({"file": npz.name, "sha256": got, "equals_gamma_audit_meta": npz_expected.get(npz.name) == got})
            if npz_expected.get(npz.name) != got:
                raise SystemExit(f"NPZ {npz.name}: sha256 differs from the gamma audit meta - inputs not verified, campaign stopped")
        jobs.append((spec, rs, str(npz), sorted(control), {ga.row_id(r): gi_rows[ga.row_id(r)] for r in rs if ga.row_id(r) in gi_rows},
                     path_b_all))
    jobs.sort(key=lambda j: -(j[0]["nx"] * j[0]["ny"] * j[0]["N"]))  # largest first
    if workers > 1:
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(workers) as pool:
            results = pool.starmap(seq_job, jobs, chunksize=1)
    else:
        results = [seq_job(*j) for j in jobs]
    rows_out = [r for res in results for r in res["rows"]]
    rows_out.sort(key=lambda r: (r["group"], r["grid"], r["N"], r["first"], r["s"]))
    write_csv(out / "rows.csv", rows_out)
    summ = summarize(rows_out)
    summ["sample_rows_expected"] = len(rows)
    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summ, f, indent=1, ensure_ascii=False, default=_json)
        f.write("\n")
    meta = {
        "study": "R5-V Q4 priority 1 (D-053.4): energy control on B-C (C-3+) solutions on the fixed Q3 inputs; QA research tool, "
                 "paired design (not a B-C trajectory); K_E, eps_floor^C, c_R not assigned",
        "profile": profile, "git_commit": git("rev-parse", "HEAD"), "git_status_porcelain": git("status", "--porcelain").splitlines(),
        "python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__, "platform": platform.platform(),
        "machine": platform.machine(), "processor": platform.processor(),
        "threads": {k: os.environ.get(k) for k in THREAD_VARS}, "workers": workers, "c_R_path_count": C_R,
        "c_R_energy_module": en.C_R_COUNT, "path_b": "all rows (extended) + fixed control sample" if path_b_all else "control sample",
        "data": prov, "q3_arrays_dir": str(arrays) if profile == "full" else "temporary (CI)", "q3_arrays_used_read_only": True,
        "npz_checked_before_calculation": npz_check,
        "npz": [{"file": r["npz"], "sha256": r["npz_sha256"]} for r in results],
        "control_ids": sorted(control), "control_source": "gamma audit meta.json control_ids (rule fixed in e11f268)",
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
    r.add_argument("--path-b", choices=("control", "all"), default="all",
                   help="rows of the exact 2-D energy path: the fixed control sample, or every row (extended check)")
    a = ap.parse_args(argv)
    out = a.out or (OUT if a.profile == "full" else ROOT / "scratch" / "r5_cccc_q4_energy_bc_ci")
    res = run_campaign(a.profile, out, a.workers if a.profile == "full" else 1, a.arrays, a.path_b == "all")
    s = res["summary"]
    print(json.dumps({k: s[k] for k in ("rows", "complete", "control_rows", "control_vok2_identity_energy_path",
                                         "gamma_inclusion_exact", "BC_K_class")}, default=_json))
    return 0 if s["complete"] == s["rows"] else 1


if __name__ == "__main__":
    sys.exit(main())
