"""R5-V Q4 PRELAUNCH (protocol docs/R5_CCCC_Q4_D1_PROTOCOL.md 4, 5, 9, 12): research pipeline of the energy control of ONE row,
N = 1, from the journal of an own B-C (C-3+) half-step of the OP-8 tool to the status of the protocol classifier.

QA research tool, NOT the D1 campaign and NOT the working CCCC solver: no import of the working ``kkvim``; the OP-8 tool
(scripts/r5_cccc_q4_d1_c3plus_trajectories.py, D-055.1), QA-EB (scripts/r5_cccc_q4_energy_bound_premises_audit.py, PR #44), EQ
(scripts/r5_cccc_q4_energy_bc_audit.py, PR #41), QA-gamma3 (scripts/r5_cccc_q4_gamma_interval_audit.py), the gamma audit
(scripts/r5_cccc_q4_gamma_audit.py) and the isolated classifier (scripts/r5_cccc_q4_d1_preflight.py, PR #48) are used read-only;
none of them is changed. No new formula, threshold, tolerance or status is introduced; every value is the one of the protocol.

Row (protocol 4.1) - half-step s of an OP-8 trajectory (journal.json + arrays.npz written by the OP-8 ``write_journal``):
1. binding of the inputs (П-а, first part): every array of the journal against its sha256 in the meta and in the record of its step;
   the chain s - 1 -> s (given_in(s) = next_given(s - 1), companion_in(s) = next_companion(s - 1), bitwise, and input_link_bitwise of
   the record); s = 1: the start set T_1 = hs.start_set(n_x, 1) and an all-+0.0 companion, bitwise; the fields formed here are bitwise
   the fields recorded by OP-8 (prev_field_after_QR_sha, field_s_sha);
2. fields from the journal arrays only: previous w^_p = fl(X_p Y_p^T) of (solved set, companion), current w^_s = fl(X_s Y_s^T) of
   (solved set, V_bc), d^ = fl(w^_p - w^_s) (EQ order; N = 1: solved set = given set, no QR);
3. path G-4: P^_p = pi_double(w^_p), P^_s = pi_double(w^_s), Q^ = q_double(d^) (r5_cccc_energy, as EQ and QA-EB);
4. Gamma_up of path B1: gamma_s exactly (QA-gamma3 ``gamma_b1`` on the journal arrays), tightest directed interval, Gamma_up =
   max(|gamma_down|, |gamma_up|) (D-054.1, item 1);
5. premises П-б...П-е by the QA-EB functions: П-б U-1 (``formation_check`` on both fields, determinism of the product); П-в U-2
   (``probe_mode`` before and after the row in this thread, the replication of the three energies operation by operation with
   ``OpLog``, the subtraction d^, finiteness of P^_p, P^_s, Q^); П-г U-3 (``grid_constants``, ``grid_ok``, h and pairwise depth of
   the replication); П-д N-7 (applicable when the computed previous field is 0: exact product 0 and a factor exactly 0); П-е B_E
   (``b_e``: root domain, D^(1/2) mu_low > 0, B_E_up finite). B_E is the exact upper bound of E-2, recorded as B_E_up = its upward
   binary64;
6. dP = P^_p - P^_s and tau' = Gamma_up + B_E_up EXACTLY from the recorded components (protocol 4.1; Z-3). A dP or tau' recorded with
   the row is accepted only if it equals its definition exactly; otherwise the row is REJECTED before classification: the value is
   withheld, the classifier gives UNRESOLVED_CERTIFICATE with source UNDETERMINED, never PASS. A missing or not finite component is
   never replaced by an assumed value: dP / tau' are then not built;
7. П-а (second part): the row is recomputed in a SEPARATE PROCESS from the same inputs (for a journal row the OP-8 half-step itself is
   re-run there: V_bc bitwise) and every reproduced value (binary64 by hex, exact rationals by p/q, array hashes, premise states) must
   be bitwise the same; values the row is bound to (``bound_values``) must be bitwise equal too. Otherwise П-а is not established;
8. ``classify_row`` of the PR #48 tool, unchanged (protocol 4.2).

Diagnostics (not premises; listed as contradictions, never hidden; they do not change the protocol status): second path of gamma (VOK-2
identity on the exact products: gamma_B = Q~ - (Pi(w~_p) - Pi(w~_s)), path B of the gamma audit) equal to the gamma of path B1 and in
[gamma_down, gamma_up] (cost limit (n_x - 1)(n_y - 1) <= SECOND_PATH_MAX_NODES, as the small-grid rule of protocol 5 item 9); Prop. D;
|E_s| <= B_E (theorem E-2) with E_s = dP - Q^ + gamma_s exact when П-б...П-е hold; B_E <= K* R_Sigma and the own ingredients against
the bounds of r5_cccc_energy (QA-EB 8).

A half-step not executed by OP-8 (STOP_*, PROTECTIVE_STOP, UNRESOLVED_C3PLUS_PREMISE, FAIL_C3PLUS_BOUND_CHECK, FAIL_TOOL) or absent
after a stop: the row is NOT executed -> UNRESOLVED with the reason (protocol 4.2), never FAIL_COMPUTATION. How C-3+ premises enter
the row status is owner decision OD-8.

Controls (scripts/r5_cccc_q4_d1_controls_prelaunch.py) use ``compute_row`` with a named injection or mutation (INJECTIONS,
MUTATIONS); such rows are marked and never enter a campaign.

Not done here: N >= 2 (PRE_D1_RANK), blocks B, C, D, the separate recount of B_E (OD-11), the D1 rows and any campaign verdict.
Guard (API and command line): the D1 grids of protocol 2.2, 2.3, 3, N != 1 and output under docs/ are refused. Data of this tool:
PRELAUNCH research observation, NOT D1 data, never D1_RESULT.

Usage (repository root; small non-D1 configurations only):
    uv run python scripts/r5_cccc_q4_d1_row_pipeline_prelaunch.py run --lx 1 --nx 8 --ly 1.25 --ny 10 --s-max 3 --out scratch/pl
"""

import argparse
import ctypes
import hashlib
import importlib.util
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import threading
from fractions import Fraction
from pathlib import Path

THREAD_VARS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
for _k in THREAD_VARS:
    os.environ[_k] = "1"  # as in Q3, EQ, QA-EB, BR: one BLAS thread (effective only if NumPy is not loaded yet; recorded)

import numpy as np  # noqa: E402

from kkvim_reference import r5_cccc_energy as en  # noqa: E402
from kkvim_reference import r5_cccc_halfstep as hs  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SELF = "scripts/r5_cccc_q4_d1_row_pipeline_prelaunch.py"
OP8_TOOL = "scripts/r5_cccc_q4_d1_c3plus_trajectories.py"
QAEB_TOOL = "scripts/r5_cccc_q4_energy_bound_premises_audit.py"
PREFLIGHT_TOOL = "scripts/r5_cccc_q4_d1_preflight.py"
USED_TOOLS = (OP8_TOOL, QAEB_TOOL, "scripts/r5_cccc_q4_energy_bc_audit.py", "scripts/r5_cccc_q4_gamma_interval_audit.py",
              "scripts/r5_cccc_q4_gamma_audit.py", "scripts/r5_cccc_q4_c_assembly_e4_audit.py", PREFLIGHT_TOOL)
SECOND_PATH_MAX_NODES = 4096  # cost limit of the exact second path of gamma (small-grid rule of protocol 5 item 9); not a threshold
D1_START = "BLOCKED"
DATA_STATUS = "PRELAUNCH research observation (N = 1); NOT D1 data, never D1_RESULT (D1_START = BLOCKED, protocol 12)"
ESTABLISHED, NOT_ESTABLISHED, NOT_CHECKABLE = "ESTABLISHED", "NOT_ESTABLISHED", "NOT_CHECKABLE"
UNDETERMINED = "UNDETERMINED"
PREMISES_COMPUTED = ("P-b", "P-c", "P-d", "P-e", "P-f")  # П-б...П-е; П-а is set from binding and reproduction
INJECTIONS = ("perturb_field_s_rel_2^-40", "scale_field_s_2^1000", "subnormal_field_s_node", "grid_D_2^-1074", "prev_factors_2^-600",
              "mu_low_zero", "B_E_sqrt_negative_D")
MUTATIONS = ("gamma_sign_flip", "gamma_from_step", "Q_from_field_s", "P_prev_from_field_s", "B_E_drop_B1")
MODES = {"round_up": (0x200, 0x300), "ftz_daz": (0x01000000, 0x03000000)}  # _controlfp(new, mask): _RC_UP; _DN_FLUSH


class D1Refused(ValueError):
    """The configuration or the output is refused (D1_START = BLOCKED; N = 1 only)."""


def _load(rel: str, name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def op8():
    return _load(OP8_TOOL, "_pl_op8")


def qa():
    return _load(QAEB_TOOL, "_pl_qaeb")


def eq():
    return qa()._eq()


def gi():
    return eq()._gi()


def ga():
    return eq()._ga()


def pf():
    return _load(PREFLIGHT_TOOL, "_pl_preflight")


def git_blob_id(path: Path) -> str:
    data = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


# ---------------------------------------------------------------------------------------------------------------------------------
# guard and encodings
# ---------------------------------------------------------------------------------------------------------------------------------

def guard_config(cfg) -> None:
    if (cfg.nx, cfg.ny) in op8().D1_GRIDS:
        raise D1Refused(f"({cfg.nx}, {cfg.ny}) is a grid of the D1 sample (protocol 2.2, 2.3, 3); D1_START = BLOCKED")
    if cfg.N != 1:
        raise D1Refused("N = 1 only: N = 2, 3 are blocked by PRE_D1_RANK (D-049.5; QR without a qualified rank decision)")


def guard_out(path: Path) -> None:
    p = Path(path).resolve()
    docs = (ROOT / "docs").resolve()
    if p == docs or docs in p.parents:
        raise D1Refused("output under docs/ is refused (no published data; D1_START = BLOCKED)")


def fhex(x) -> str | None:
    return None if x is None else float(x).hex()


def ftext(x: Fraction | None) -> str | None:
    return None if x is None else f"{x.numerator}/{x.denominator}"


def fsha(x: Fraction | None) -> str | None:
    return None if x is None else hashlib.sha256(ftext(x).encode()).hexdigest()


def parse_recorded(v) -> Fraction:
    """A recorded value: Fraction, int, float (exactly), 'p/q' or decimal text, or 'hex:<float.hex>'."""
    if isinstance(v, bool):
        raise TypeError("a boolean is not a value")
    if isinstance(v, Fraction):
        return v
    if isinstance(v, int):
        return Fraction(v)
    if isinstance(v, float):
        if not math.isfinite(v):
            raise ValueError("a recorded dP / tau' must be finite")
        return Fraction(v)
    if isinstance(v, str):
        if v.startswith("hex:"):
            return parse_recorded(float.fromhex(v[4:]))
        return Fraction(v)
    raise TypeError(f"unsupported recorded value {v!r}")


def _float_of(v) -> float:
    """A component: binary64 as float, hex text or 'inf' / 'nan' text."""
    if isinstance(v, bool):
        raise TypeError("a boolean is not a value")
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        t = v.strip().lower()
        if t.lstrip("+-") in ("inf", "infinity", "nan"):
            return float(t)
        return float.fromhex(v)
    raise TypeError(f"unsupported component {v!r}")


# ---------------------------------------------------------------------------------------------------------------------------------
# 6. the boundary: exact dP and tau' from the recorded components, Z-3, classification
# ---------------------------------------------------------------------------------------------------------------------------------

COMPONENTS = ("P_p", "P_s", "Q", "Gamma_up", "B_E")


def assemble(components: dict, premises: dict, recorded: dict | None = None) -> dict:
    """Protocol 4.1 / Z-3: dP = P^_p - P^_s and tau' = Gamma_up + B_E exactly from the recorded binary64 components (a component
    absent, None or not finite is never replaced; dP / tau' are then not built). A recorded dP or tau' must equal the definition
    exactly, otherwise the row is rejected (the value withheld). Then classify_row (PR #48) with the components."""
    unknown = set(components) - set(COMPONENTS)
    if unknown:
        raise ValueError(f"unknown components {sorted(unknown)}")
    c = {k: _float_of(v) for k, v in components.items() if v is not None}

    def fin(k):
        return k in c and math.isfinite(c[k])

    dP = Fraction(c["P_p"]) - Fraction(c["P_s"]) if fin("P_p") and fin("P_s") else None
    tau = Fraction(c["Gamma_up"]) + Fraction(c["B_E"]) if fin("Gamma_up") and fin("B_E") else None
    rejections = []
    for name, built in (("dP", dP), ("tau", tau)):
        if recorded is None or recorded.get(name) is None:
            continue
        r = parse_recorded(recorded[name])
        if built is None:
            rejections.append(f"recorded {name} given while its components are not all recorded finite values: not accepted")
        elif r != built:
            rejections.append(f"recorded {name} = {ftext(r)} differs from its definition {ftext(built)} (protocol 4.1; Z-3)")
            if name == "dP":
                dP = None
            else:
                tau = None
    inp = {"executed": True, "premises": premises, "dP": dP, "tau": tau}
    if "Q" in c:
        inp["Q"] = c["Q"]
    for k in ("P_p", "P_s", "Gamma_up", "B_E"):
        if k in c:
            inp[k] = c[k]
    res = pf().classify_row(inp)
    res = dict(res)
    if rejections:
        res["reasons"] = list(res["reasons"]) + [f"REJECTED at the pipeline boundary: {x}" for x in rejections]
    res.update({"rejected": bool(rejections), "rejections": rejections, "dP_exact": ftext(dP), "dP_sha256": fsha(dP),
                "tau_exact": ftext(tau), "tau_sha256": fsha(tau), "components_hex": {k: fhex(v) for k, v in c.items()}})
    return res


# ---------------------------------------------------------------------------------------------------------------------------------
# 2.-5. the row computation (in the calling thread)
# ---------------------------------------------------------------------------------------------------------------------------------

def _prem(ok: bool, reasons: list[str], checkable: bool = True) -> dict:
    state = ESTABLISHED if ok else (NOT_ESTABLISHED if checkable else NOT_CHECKABLE)
    return {"state": state, "reason": "; ".join(reasons) if reasons else "all checks hold"}


def _node(a: np.ndarray) -> tuple[int, int]:
    return a.shape[0] // 2, a.shape[1] // 2


def compute_row(cfg, s: int, given: np.ndarray, companion: np.ndarray, V: np.ndarray, injection: str | None = None,
                mutation: str | None = None, gamma_other: Fraction | None = None) -> dict:
    """Steps 2-5 of the module docstring on the arrays of one half-step (N = 1: given = solved set). ``injection`` (control B) or
    ``mutation`` (control G) names one of INJECTIONS / MUTATIONS; None for an ordinary row."""
    guard_config(cfg)
    if injection is not None and injection not in INJECTIONS:
        raise ValueError(f"unknown injection {injection!r}")
    if mutation is not None and mutation not in MUTATIONS:
        raise ValueError(f"unknown mutation {mutation!r}")
    Q_ = qa()
    lx, nx, ly, ny, D, q = float(cfg.lx), cfg.nx, float(cfg.ly), cfg.ny, float(cfg.D), float(cfg.q)
    direction = op8().direction_of(s)
    given, companion, V = (np.array(a, dtype=float, copy=True) for a in (given, companion, V))
    N = given.shape[1]
    g = en.Grid(lx, nx, ly, ny, D=D, q=q)
    out = {"s": s, "direction": direction, "N": N, "injection": injection, "mutation": mutation}
    reasons = {p: [] for p in PREMISES_COMPUTED}
    checkable = {p: True for p in PREMISES_COMPUTED}
    pb = Q_.probe_mode()
    out["mode_before_ok"], out["controlfp_before"] = pb["ok"], pb["controlfp"].get("word")
    if not pb["ok"]:
        reasons["P-c"].append("U-2: mode probe before the row: " + "; ".join(pb["fails"]))
    # --- 2. fields
    Xp, Yp = eq().orient(direction, given, companion)
    Xs, Ys = eq().orient(direction, given, V)
    if injection == "prev_factors_2^-600":
        Xp, Yp = np.full_like(Xp, 2.0**-600), np.full_like(Yp, 2.0**-600)
    with np.errstate(all="ignore"):
        wh_prev, wh_s = Xp @ Yp.T, Xs @ Ys.T
        if injection == "perturb_field_s_rel_2^-40":
            i, j = _node(wh_s)
            wh_s[i, j] = wh_s[i, j] * (1.0 + 2.0**-40)
        elif injection == "scale_field_s_2^1000":
            wh_s = wh_s * 2.0**1000
        elif injection == "subnormal_field_s_node":
            i, j = _node(wh_s)
            wh_s[i, j] = 2.0**-1060
        dh = wh_prev - wh_s
        p_field = wh_s if mutation == "P_prev_from_field_s" else wh_prev
        q_field = wh_s if mutation == "Q_from_field_s" else dh
        # --- 3. path G-4
        Pp, Ps, Qd = en.pi_double(g, p_field), en.pi_double(g, wh_s), en.q_double(g, q_field)
    sha = op8().array_sha
    out.update({"field_prev_sha": sha(wh_prev), "field_s_sha": sha(wh_s), "d_hat_sha": sha(dh), "P_hat_p": fhex(Pp),
                "P_hat_s": fhex(Ps), "Q_hat": fhex(Qd)})
    # --- 4. gamma of path B1
    gamma = None
    try:
        gamma = gi().gamma_b1(lx, nx, ly, ny, direction, given, companion, V, D=D, q=q)
    except (ValueError, ArithmeticError) as exc:
        out["gamma_error"] = f"{type(exc).__name__}: {exc}"
    if gamma is not None and mutation == "gamma_sign_flip":
        gamma = -gamma
    if mutation == "gamma_from_step":
        if gamma_other is None:
            raise ValueError("gamma_from_step needs gamma_other")
        gamma = gamma_other
    G_up = gd = gu = None
    if gamma is not None:
        gd, gu = gi().directed(gamma)
        G_up = gi().gamma_up(gd, gu)
    out.update({"gamma_exact": ftext(gamma), "gamma_exact_sha256": fsha(gamma), "gamma_down": fhex(gd), "gamma_up": fhex(gu),
                "Gamma_up": fhex(G_up)})
    # --- 5. premises: П-б U-1
    with np.errstate(all="ignore"):
        det = Q_.same_bits(Xp @ Yp.T, wh_prev) and Q_.same_bits(Xs @ Ys.T, wh_s)
    out["product_deterministic"] = det
    if not det:
        reasons["P-b"].append("U-1: a second product X Y^T differs bitwise from the field of the row")
    forms = {}
    for tag, (X, Y, W) in (("prev", (Xp, Yp, wh_prev)), ("s", (Xs, Ys, wh_s))):
        try:
            f = Q_.formation_check(X, Y, W)
        except (ValueError, ArithmeticError, OverflowError) as exc:  # a detector that cannot be evaluated: not checkable
            forms[tag] = None
            checkable["P-b"] = False
            reasons["P-b"].append(f"U-1 on the {tag} field not checkable: {type(exc).__name__}: {exc}")
            continue
        forms[tag] = f
        out[f"U1_{tag}_holds"], out[f"U1_{tag}_max_ratio"] = f["holds"], f["max_ratio"]
        if not f["holds"]:
            reasons["P-b"].append(f"U-1: |e| <= gamma_N |X||Y|^T violated on the {tag} field ({f['violations']} entries)")
    fp, fs_ = forms["prev"], forms["s"]
    # П-в U-2: replication with checked operations, finiteness
    log = Q_.OpLog()
    with np.errstate(all="ignore"):
        dh2 = wh_prev - wh_s
        log.add("d^ = w^_p - w^_s", wh_prev, wh_s, dh2, sub=True)
        reps = {"p": Q_.replicate_energy(g, p_field, log, "P^_prev", True), "s": Q_.replicate_energy(g, wh_s, log, "P^_s", True),
                "d": Q_.replicate_energy(g, q_field, log, "Q^", False)}
    rep_ok = bool(Q_.same_bits(dh2, dh) and all(r["t_bitwise_module"] and r["quad_bitwise_module"] and r["value_bitwise_module"]
                                                for r in reps.values())
                  and reps["p"]["value"].hex() == Pp.hex() and reps["s"]["value"].hex() == Ps.hex() and reps["d"]["value"].hex()
                  == Qd.hex())
    ld = log.as_dict()
    out.update({"U2_replication_bitwise": rep_ok, "U2_violations": ld["violations"], "U2_subnormal_operands": ld["subnormal_operands"],
                "U2_ops": ld["ops"]})
    if not rep_ok:
        reasons["P-c"].append("U-2: the replication is not bitwise the module's computation")
    if ld["violations"]:
        reasons["P-c"].append("U-2: " + "; ".join(ld["violations"]))
    for name, v in (("P^_p", Pp), ("P^_s", Ps), ("Q^", Qd)):
        if not math.isfinite(v):
            reasons["P-c"].append(f"finiteness: {name} = {v!r}")
    # П-г U-3
    D_grid = 2.0**-1074 if injection == "grid_D_2^-1074" else D
    kc = Q_.grid_constants(lx, nx, ly, ny, D_grid, q, N)
    out["U3_grid_bits"] = list(Q_.grid_bits(kc))
    out["U3_grid_ok"] = Q_.grid_ok(kc)
    out["U3_h_equals_replication"] = all(float(kc[k]).hex() == float(reps["p"][r]).hex() for k, r in
                                         (("h_dx", "hx"), ("h_dy", "hy"), ("c_x", "cx"), ("c_y", "cy")))
    out["U3_levels_equal_L"] = reps["p"]["levels"] == kc["L"] and reps["s"]["levels"] == kc["L"] and reps["d"]["levels"] == kc["L_Q"]
    if not (out["U3_grid_ok"] and out["U3_h_equals_replication"] and out["U3_levels_equal_L"]):
        reasons["P-d"].append("U-3: grid constants not established (normality, denominators, pairwise depth or h of the replication)")
    # П-д N-7 (the branch eps_a^C(s - 1) = 0 of r5_cccc_energy: the computed previous field is 0)
    out["N7_applicable"] = not bool(np.any(wh_prev))
    if out["N7_applicable"]:
        if fp is None:
            checkable["P-e"] = False
            reasons["P-e"].append("N-7 not checkable: exact product of the previous field not available (U-1 not checkable)")
        else:
            factor0 = bool(np.all(Q_.is_zero(Xp)) or np.all(Q_.is_zero(Yp)))
            out["N7_ok"] = bool(fp["exact_product_zero"] and fp["computed_field_zero"] and factor0)
            if not out["N7_ok"]:
                reasons["P-e"].append("N-7: computed previous field 0 without an exact zero product of an exactly zero factor")
    # П-е B_E
    BE, be = None, None
    if fp is None or fs_ is None:
        checkable["P-f"] = False
        reasons["P-f"].append("B_E not checkable: the formation integers of U-1 are not available")
    else:
        try:
            gq = Q_.GridQ(lx, nx, ly, ny, D, q)
            if injection == "mu_low_zero":
                gq.mu_low = Fraction(0)
            elif injection == "B_E_sqrt_negative_D":
                gq.D = Fraction(-1)
            be = Q_.b_e(gq, kc, N, fp, fs_, dh)
            BE = be["BE"]
            if BE is None:
                reasons["P-f"].append("B_E: denominator D^(1/2) mu_low not positive")
        except ValueError as exc:
            checkable["P-f"] = False
            reasons["P-f"].append(f"B_E not checkable (root domain): {exc}")
    if BE is not None and mutation == "B_E_drop_B1":
        BE = BE - be["bnd"]["B1"]
    BE_up = None
    if BE is not None:
        BE_up = gi().directed(BE)[1]
        if not math.isfinite(BE_up):
            reasons["P-f"].append("B_E_up not finite in binary64")
    out.update({"B_E_exact_sha256": fsha(BE), "B_E_up": fhex(BE_up)})
    # --- mode probe after the row, same thread
    pa = Q_.probe_mode()
    out["mode_after_ok"], out["controlfp_after"] = pa["ok"], pa["controlfp"].get("word")
    out["same_main_thread"] = pa["thread"] == pb["thread"] and pb["main_thread"]
    if not pa["ok"]:
        reasons["P-c"].append("U-2: mode probe after the row: " + "; ".join(pa["fails"]))
    if not out["same_main_thread"]:
        reasons["P-c"].append("U-2: the probes and the computation are not in one (main) thread")
    out["premises"] = {p: _prem(not reasons[p], reasons[p], checkable[p]) for p in PREMISES_COMPUTED}
    out["components"] = {"P_p": Pp, "P_s": Ps, "Q": Qd, "Gamma_up": G_up, "B_E": BE_up}
    # --- diagnostics (contradictions)
    contra, diag = [], {}
    if (nx - 1) * (ny - 1) <= SECOND_PATH_MAX_NODES and gamma is not None:
        try:
            Pe, Se, Qe = ga().energies_path_b(lx, nx, ly, ny, D, q, Xp, Yp, Xs, Ys)
            gB = Qe - (Pe - Se)
            ok = gB == gamma and Fraction(gd) <= gB <= Fraction(gu)
            diag["second_path_gamma"] = {"performed": True, "consistent": bool(ok), "gamma_B_sha256": fsha(gB)}
            if not ok:
                contra.append("second path of gamma (VOK-2 identity, path B) differs from the gamma used (path B1)")
        except (ValueError, ArithmeticError, OverflowError) as exc:
            diag["second_path_gamma"] = {"performed": False, "consistent": None, "reason": f"{type(exc).__name__}: {exc}"}
    else:
        diag["second_path_gamma"] = {"performed": False, "consistent": None,
                                     "reason": f"NOT_PERFORMED ((n_x - 1)(n_y - 1) > {SECOND_PATH_MAX_NODES}) or no gamma"}
    if be is not None:
        diag["PropD_two_routes_ws"] = bool(be["PropD"])
        if not be["PropD"]:
            contra.append("Prop. D: two routes of ||w^_s||_a^2 differ")
    established = all(out["premises"][p]["state"] == ESTABLISHED for p in PREMISES_COMPUTED)
    if BE is not None and gamma is not None and math.isfinite(Pp) and math.isfinite(Ps) and math.isfinite(Qd):
        E_s = Fraction(Pp) - Fraction(Ps) - Fraction(Qd) + gamma
        diag["E_s_le_B_E"] = abs(E_s) <= BE
        if established and not diag["E_s_le_B_E"]:
            contra.append("theorem E-2: |E_s| > B_E with П-б...П-е established")
    try:
        mt = eq().model_terms(g, Xp, Yp, Xs, Ys)
        R = mt.pop("_R_Sigma")
        eps_s_en, eps_p_en, da_en, Bw_en = mt.pop("_Rf_parts")
        if be is not None and BE is not None:
            L = kc["L"]
            sums, beta = be["sums"], be["beta"]
            RCp = Q_.U * (L + Q_.C_R) * (sums["p"]["Sq"] + sums["p"]["Sl"])
            RCs = Q_.U * (L + Q_.C_R) * (sums["s"]["Sq"] + sums["s"]["Sl"])
            RQ = Q_.U * (L + Q_.C_R) * sums["d"]["Sq"] + beta * be["dh_a"]
            xc = {"R_C_prev": RCp <= Fraction(mt["R_C_prev_up"]), "R_C_s": RCs <= Fraction(mt["R_C_s_up"]),
                  "R_Q": RQ <= Fraction(mt["R_Q_up"]), "eps_s": be["eps_s"] <= eps_s_en, "eps_prev": be["eps_p"] <= eps_p_en,
                  "d_hat_a": be["dh_a"] <= da_en, "B_omega": be["Bw"] is not None and be["Bw"] <= Bw_en,
                  "B_E_le_Kstar_R": BE <= kc["K_star"] * R}
            diag["crosscheck_EN"] = {k: bool(v) for k, v in xc.items()}
            bad = [k for k, v in xc.items() if not v]
            if bad and established:
                contra.append("own B_E ingredients against r5_cccc_energy: " + ", ".join(bad))
    except (ValueError, ArithmeticError, OverflowError) as exc:
        diag["crosscheck_EN"] = f"NOT_AVAILABLE ({type(exc).__name__}: {exc})"
    out["diagnostics"] = diag
    out["contradictions"] = contra
    return out


REPRO_KEYS = ("field_prev_sha", "field_s_sha", "d_hat_sha", "P_hat_p", "P_hat_s", "Q_hat", "gamma_exact", "gamma_exact_sha256",
              "gamma_down", "gamma_up", "Gamma_up", "B_E_exact_sha256", "B_E_up", "U3_grid_bits", "N7_applicable")


def reproduced_values(row: dict) -> dict:
    """The values a second run must reproduce bitwise (text forms) and the premise states."""
    out = {k: row.get(k) for k in REPRO_KEYS}
    out["premise_states"] = {p: row["premises"][p]["state"] for p in PREMISES_COMPUTED}
    out["contradictions"] = list(row["contradictions"])
    out["second_path_gamma_consistent"] = row["diagnostics"].get("second_path_gamma", {}).get("consistent")
    return out


# ---------------------------------------------------------------------------------------------------------------------------------
# 1. binding of a journal (П-а, first part)
# ---------------------------------------------------------------------------------------------------------------------------------

STEP_ARRAYS = (("given_in", "given_in_sha"), ("companion_in", "companion_in_sha"), ("solved_set", "solved_set_sha"),
               ("companion_solved", "companion_solved_sha"), ("C3plus", "C3plus_sha"), ("V_bc", "V_bc_sha"),
               ("next_given", "next_given_sha"), ("next_companion", "next_companion_sha"))


def load_journal(jdir: Path) -> tuple[dict, dict]:
    with open(Path(jdir) / "journal.json", encoding="utf-8") as f:
        j = json.load(f)
    return j, op8().load_arrays(Path(jdir))


def config_of(meta: dict):
    c = meta["config"]
    return op8().TrajectoryConfig(float(c["lx"]), int(c["nx"]), float(c["ly"]), int(c["ny"]), int(c["N"]), float(c["D"]),
                                  float(c["q"]), int(c["s_max"]))


def bind_journal(j: dict, arrays: dict) -> dict:
    """Per half-step: hashes of its arrays against the meta and the record, the chain and the start (module docstring, 1)."""
    sha = op8().array_sha
    meta, steps = j["meta"], j["steps"]
    cfg = config_of(meta)
    listed = meta.get("arrays_sha256", {})
    whole = [] if set(listed) == set(arrays) else ["the arrays of arrays.npz differ from the list of the meta"]
    per_key = {k: f"{k}: sha256 differs from the meta" for k, v in arrays.items() if listed.get(k) != sha(v)}
    out = {}
    for rec in steps:
        s = rec["s"]
        r = whole + [m for k, m in per_key.items() if k.startswith(f"s{s}_")]
        for name, key in STEP_ARRAYS:
            a = arrays.get(f"s{s}_{name}")
            if key in rec:
                if a is None:
                    r.append(f"{name}: recorded hash without an array")
                elif rec[key] != sha(a):
                    r.append(f"{name}: sha256 differs from the record of s = {s}")
        gi_, ci_ = arrays.get(f"s{s}_given_in"), arrays.get(f"s{s}_companion_in")
        if s == 1:
            if gi_ is None or not np.array_equal(gi_, hs.start_set(cfg.nx, cfg.N)) or gi_.tobytes() != hs.start_set(cfg.nx, cfg.N).tobytes():
                r.append("s = 1: given set is not bitwise the start set T_N")
            if ci_ is None or ci_.tobytes() != np.zeros_like(ci_).tobytes():
                r.append("s = 1: companion is not bitwise +0.0")
        else:
            pg, pc = arrays.get(f"s{s - 1}_next_given"), arrays.get(f"s{s - 1}_next_companion")
            if pg is None or pc is None or gi_ is None or ci_ is None or gi_.tobytes() != pg.tobytes() \
                    or ci_.tobytes() != pc.tobytes() or gi_.shape != pg.shape or ci_.shape != pc.shape:
                r.append(f"chain: input of s = {s} is not bitwise the next input of s = {s - 1}")
            if rec.get("input_link_bitwise") is not True:
                r.append("chain: input_link_bitwise of the record is not True")
        out[s] = {"ok": not r, "reasons": r}
    return out


# ---------------------------------------------------------------------------------------------------------------------------------
# 7. separate process (П-а, second part)
# ---------------------------------------------------------------------------------------------------------------------------------

def _set_mode(mode: str | None) -> dict:
    if mode is None:
        return {"mode": None}
    if sys.platform != "win32":
        raise RuntimeError("floating-point mode injection needs _controlfp (Windows)")
    new, mask = MODES[mode]
    f = ctypes.CDLL("ucrtbase")._controlfp
    f.restype, f.argtypes = ctypes.c_uint, [ctypes.c_uint, ctypes.c_uint]
    return {"mode": mode, "controlfp": hex(f(new, mask))}


def _inputs_from_spec(spec: dict):
    """(cfg, s, given, companion, V, input binding reasons, OP-8 reproduction) of a row specification (journal or control arrays)."""
    sha = op8().array_sha
    s = int(spec["s"])
    reasons, rerun = [], None
    if spec["kind"] == "journal":
        j, arrays = load_journal(Path(spec["journal"]))
        cfg = config_of(j["meta"])
        b = bind_journal(j, arrays)
        if s not in b:
            raise ValueError(f"no half-step {s} in the journal")
        reasons += b[s]["reasons"]
        given, companion, V = arrays[f"s{s}_solved_set"], arrays[f"s{s}_companion_solved"], arrays[f"s{s}_V_bc"]
        if spec.get("rerun_op8", True):
            res = op8().run_step(cfg, s, arrays[f"s{s}_given_in"], arrays[f"s{s}_companion_in"])
            rerun = {"status": res.record.get("status"), "V_bc_sha": res.record.get("V_bc_sha"),
                     "reproduced": res.record.get("status") == op8().OK and "V_bc" in res.arrays
                     and res.arrays["V_bc"].tobytes() == np.asarray(V).tobytes()}
    else:
        with np.load(spec["arrays_npz"]) as z:
            given, companion, V = z["given"], z["companion"], z["V"]
        cfg = op8().TrajectoryConfig(**spec["config"])
        for name, a in (("given", given), ("companion", companion), ("V", V)):
            if spec["arrays_sha256"].get(name) != sha(a):
                reasons.append(f"control input {name}: sha256 differs from the construction")
    return cfg, s, given, companion, V, reasons, rerun


def _row_job(spec: dict) -> dict:
    """The row in this process (child entry and in-process primary)."""
    mode = _set_mode(spec.get("mode"))
    cfg, s, given, companion, V, reasons, rerun = _inputs_from_spec(spec)
    go = parse_recorded(spec["gamma_other"]) if spec.get("gamma_other") is not None else None
    row = compute_row(cfg, s, given, companion, V, spec.get("injection"), spec.get("mutation"), go)
    return {"row": row, "input_binding_reasons": reasons, "op8_rerun": rerun, "mode": mode, "pid": os.getpid()}


def _json_default(o):
    if isinstance(o, (np.bool_, np.integer, np.floating)):
        return o.item()
    if isinstance(o, Fraction):
        return ftext(o)
    if isinstance(o, float):
        return o.hex()
    raise TypeError(type(o).__name__)


def run_child(spec: dict, timeout: float = 1800) -> dict:
    with tempfile.TemporaryDirectory() as td:
        sp = Path(td) / "spec.json"
        sp.write_text(json.dumps(spec), encoding="utf-8")
        p = subprocess.run([sys.executable, str(ROOT / SELF), "_row", str(sp)], cwd=ROOT, capture_output=True, text=True,
                           timeout=timeout)
    if p.returncode != 0:
        raise RuntimeError(f"child row failed ({p.returncode}): {p.stderr[-3000:]}")
    return json.loads(p.stdout.strip().splitlines()[-1])


def _comparable(job: dict) -> dict:
    row = dict(job["row"])
    comp = {k: fhex(v) for k, v in row.pop("components").items()}
    vals = reproduced_values({**row, "premises": row["premises"]})
    vals["components"] = comp
    vals["op8_rerun"] = job.get("op8_rerun")
    return json.loads(json.dumps(vals, default=_json_default))


# ---------------------------------------------------------------------------------------------------------------------------------
# the row: binding, computation, reproduction, П-а, classification
# ---------------------------------------------------------------------------------------------------------------------------------

def analyse(spec: dict, reproduce: bool = True, bound_values: dict | None = None, recorded: dict | None = None) -> dict:
    """One row from a specification: {"kind": "journal", "journal": dir, "s": s} or {"kind": "arrays", "arrays_npz": path,
    "arrays_sha256": {...}, "config": {...}, "s": s}, optional "injection", "mutation", "gamma_other", "mode"."""
    try:
        if spec.get("mode") is not None:
            primary = run_child(spec)
        else:
            primary = json.loads(json.dumps(_row_job(spec), default=_json_default))
    except D1Refused:
        raise
    except Exception as exc:  # recorded, never hidden: the row is not executed (protocol 4.2), never PASS
        res = pf().classify_row({"executed": False, "not_executed": {"reason": f"pipeline exception {type(exc).__name__}: "
                                                                               f"{str(exc)[-500:]}"}})
        return {"data_status": DATA_STATUS, "spec": {k: v for k, v in spec.items() if k != "arrays_sha256"}, "row": None,
                "reproduction": None, "premises": {}, "classification": res, "status": res["status"], "contradictions": [],
                "op8_rerun": None, "exception": f"{type(exc).__name__}: {exc}"}
    row = primary["row"]
    pa = list(primary["input_binding_reasons"])
    if spec["kind"] == "journal":
        j, _ = load_journal(Path(spec["journal"]))
        rec = next(r for r in j["steps"] if r["s"] == spec["s"])
        if row["field_s_sha"] != rec.get("field_s_sha") and spec.get("injection") is None:
            pa.append("the current field formed here is not bitwise the field of the journal (field_s_sha)")
        if row["field_prev_sha"] != rec.get("prev_field_after_QR_sha") and spec.get("injection") is None:
            pa.append("the previous field formed here is not bitwise the field of the journal (prev_field_after_QR_sha)")
        rr = primary.get("op8_rerun")
        if rr is not None and not rr["reproduced"]:
            pa.append(f"the OP-8 half-step re-run does not reproduce V_bc bitwise ({rr['status']})")
    repro = None
    if reproduce:
        try:
            second = run_child(spec)
            a, b = _comparable(primary), _comparable(second)
            repro = {"bitwise_equal": a == b, "differences": sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k)),
                     "child_pid": second["pid"]}
            if not repro["bitwise_equal"]:
                pa.append("separate process: values differ: " + ", ".join(repro["differences"]))
        except (RuntimeError, subprocess.TimeoutExpired) as exc:
            repro = {"bitwise_equal": None, "error": str(exc)[-500:]}
    if bound_values:
        for k, v in bound_values.items():
            if row.get(k) != v:
                pa.append(f"bound value {k}: {row.get(k)} != {v}")
    if reproduce and repro is not None and repro.get("bitwise_equal") is None:
        pa_state = {"state": NOT_CHECKABLE, "reason": "separate process failed: " + repro.get("error", "")}
    elif not reproduce:
        pa_state = {"state": NOT_CHECKABLE, "reason": "no reproduction in a separate process (П-а requires it)" +
                    ("; " + "; ".join(pa) if pa else "")}
    else:
        pa_state = {"state": ESTABLISHED if not pa else NOT_ESTABLISHED, "reason": "; ".join(pa) if pa else
                    "inputs bound by sha256 (journal / construction, chain, start), fields bitwise the journal's, values reproduced "
                    "bitwise in a separate process"}
    premises = {"P-a": pa_state, **row["premises"]}
    comps = {k: (float.fromhex(v) if isinstance(v, str) else v) for k, v in row["components"].items()}
    res = assemble(comps, premises, recorded)
    return {"data_status": DATA_STATUS, "spec": {k: v for k, v in spec.items() if k != "arrays_sha256"}, "row": row,
            "reproduction": repro, "premises": premises, "classification": res, "status": res["status"],
            "contradictions": row["contradictions"], "op8_rerun": primary.get("op8_rerun")}


def not_executed(rec: dict | None, s: int) -> dict:
    """A half-step without a B-C solution of OP-8 (or absent): not executed -> UNRESOLVED (protocol 4.2), never FAIL_COMPUTATION."""
    st = "absent (trajectory stopped before)" if rec is None else str(rec.get("status"))
    note = " (status of C-3+ premises in the row: owner decision OD-8)" if "C3PLUS" in st else ""
    res = pf().classify_row({"executed": False, "not_executed": {"reason": f"OP-8 half-step s = {s}: {st}{note}"}})
    return {"data_status": DATA_STATUS, "spec": {"s": s}, "status": res["status"], "classification": res, "contradictions": []}


def analyse_journal(jdir: Path, s_max: int | None = None, reproduce: bool = True) -> list[dict]:
    j, _ = load_journal(Path(jdir))
    cfg = config_of(j["meta"])
    guard_config(cfg)
    by_s = {r["s"]: r for r in j["steps"]}
    out = []
    for s in range(1, (s_max or cfg.s_max) + 1):
        rec = by_s.get(s)
        if rec is None or rec.get("status") != op8().OK:
            out.append(not_executed(rec, s))
        else:
            out.append(analyse({"kind": "journal", "journal": str(jdir), "s": s}, reproduce=reproduce))
    return out


def run_trajectory_journal(cfg, out: Path) -> Path:
    """OP-8 trajectory of a non-D1 N = 1 configuration, journal written to out/journal (OP-8 write_journal)."""
    guard_config(cfg)
    guard_out(out)
    tr = op8().run_trajectory(cfg)
    jdir = Path(out) / "journal"
    op8().write_journal(tr, jdir)
    return jdir


def write_control_arrays(path: Path, given, companion, V) -> dict:
    """Arrays of a control row (control V, mutations) with their sha256 (binding to the construction)."""
    guard_out(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrs = {"given": np.asarray(given, dtype=float), "companion": np.asarray(companion, dtype=float), "V": np.asarray(V, dtype=float)}
    np.savez(path, **arrs)
    return {k: op8().array_sha(v) for k, v in arrs.items()}


def provenance() -> dict:
    return {"tool_blob": git_blob_id(ROOT / SELF), "used_tools": {t: git_blob_id(ROOT / t) for t in USED_TOOLS},
            "python": sys.version.split()[0], "numpy": np.__version__, "platform": platform.platform(),
            "machine": platform.machine(), "processor": platform.processor(), "sys_platform": sys.platform,
            "blas": qa().blas_info(), "blas_thread_env_vars": {k: os.environ.get(k) for k in THREAD_VARS},
            "main_thread": threading.current_thread() is threading.main_thread(),
            "reproducibility": "bitwise only on the same platform, BLAS build and thread settings (D-025; AGENTS.md 4)"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    for k in ("lx", "ly"):
        r.add_argument(f"--{k}", type=float, required=True)
    for k in ("nx", "ny"):
        r.add_argument(f"--{k}", type=int, required=True)
    r.add_argument("--D", type=float, default=1.0)
    r.add_argument("--q", type=float, default=1.0)
    r.add_argument("--s-max", type=int, default=3)
    r.add_argument("--out", type=Path, required=True)
    c = sub.add_parser("_row")
    c.add_argument("spec", type=Path)
    a = ap.parse_args(argv)
    if a.cmd == "_row":
        spec = json.loads(a.spec.read_text(encoding="utf-8"))
        print(json.dumps(_row_job(spec), default=_json_default))
        return 0
    try:
        cfg = op8().TrajectoryConfig(a.lx, a.nx, a.ly, a.ny, 1, a.D, a.q, a.s_max)
        guard_config(cfg)
        guard_out(a.out)
    except D1Refused as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    jdir = run_trajectory_journal(cfg, a.out)
    rows = analyse_journal(jdir)
    doc = {"data_status": DATA_STATUS, "d1_start": D1_START, "config": a.__dict__ | {"out": str(a.out)}, "provenance": provenance(),
           "rows": rows}
    with open(Path(a.out) / "rows.json", "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1, ensure_ascii=False, default=_json_default)
    print(json.dumps({"rows": [x["status"] for x in rows]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
