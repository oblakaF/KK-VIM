"""R5-V Q4, D-054 / D-055: PRELAUNCH preparation of the D1 campaign - conditions 3-5 of sect. 12 of the protocol
docs/R5_CCCC_Q4_D1_PROTOCOL.md (APPROVED_PROTOCOL, D-055.2). QA research tool.

THIS IS NOT THE D1 CAMPAIGN. D1_START = BLOCKED. The tool runs no trajectory beyond the first half-step, classifies no D1 row, gives
no verdict of the candidate and produces no D1 data. No import of the working ``kkvim``; the CCCC solver of Builder, the prototype
B-C, the QA tools of PR #39-#46 and all published data are used read-only (no file of them is changed). No threshold is assigned
(K_E, tol_solve, eta_max^C, TS-1, K, c_d, c_rank, tau_set, S-2, ...); the protocol, its statuses, the strict and non-strict
inequalities, the candidate c = 1 and the area s = 1, 2, 3 are taken as written.

Contents (protocol 12, items 3-5):
1. MANIFEST (12.3, task A). ``build_manifest``: git blob ids of the protocol and of every tool of the campaign (a component that does
   not exist is MISSING, with no id); the 102 mandatory rows of block A (34 grids of 2.2-2.3 x N = 1 x s = 1, 2, 3) with keys,
   geometry, configuration and the sha256 of the inputs known BEFORE the run (start set T_1 and zero companion of s = 1, the
   configuration); the procedure that records the dynamic hashes (s = 2, 3) during a future run - NOT executed here; the second-path
   sample (OP-4, 5 item 9) and the B_E recount sample (sect. 9) chosen by declared features of the rows only; the proposals for the
   parameters the accepted documents do not fix, each OWNER_DECISION_REQUIRED. The manifest is INCOMPLETE and NOT FROZEN.
2. CLASSIFIER (12.4, task B). ``classify_row`` (protocol 4.2), ``campaign_verdict`` (4.3, 4.4): an ISOLATED QA classifier for the
   synthetic harness of controls A and B; it is not wired to any computation. ``run_controls`` compares it with the expectations
   frozen BEFORE this code was written (docs/results/r5_cccc_q4_d1_prelaunch/control_ab_expected.json, sha256 EXPECTED_SHA256).
   Control B has two parts: synthetic premise flags (classifier branch) and injections into the premise DETECTOR FUNCTIONS of the QA-EB
   tool (PR #44) on minimal hand-made inputs, each in a child process. Neither part qualifies the detectors of a D1 row: the pipeline
   OP-8 trajectory -> G-4 / B1 / B_E / premises -> classifier does not exist (MISSING).
3. RESOURCE PROBES (12.5, task C; PL 6.3). ``probe``: one half-step s = 1, N = 1 of a configuration, in a child process, with the
   stages (a) OP-8 tool, first half-step (B-C with C-3+ and its premises), (b) QA-EB ``analyse_row`` on the same input (EQ path G-4,
   gamma of path B1, U-1, U-2, U-3, N-7, exact B_E), (c) BR ``analyse_point`` (certified reference: exact assembly, decimal Cholesky
   ladder, refinement, lambda_low). Recorded: stage times, peak working set, total RAM, the factorization / refinement flags of BR,
   the bitwise identity of the B-C solution of (a) and of the EQ path of (b). NO numerical value of the row (P^, Q^, gamma, B_E, tau',
   statuses) is recorded or printed: the probe result is PRELAUNCH_RESOURCE_PROBE, never D1_RESULT. N = 2, 3 are refused
   (PRE_D1_RANK: D-049.5 requires a stop before QR, OP-8 continues QR as RESEARCH_ONLY); block D is refused (geometry not fixed).

Usage (repository root):
    uv run python scripts/r5_cccc_q4_d1_preflight.py manifest            # writes docs/results/r5_cccc_q4_d1_prelaunch/manifest.*
    uv run python scripts/r5_cccc_q4_d1_preflight.py controls            # writes .../control_ab_results.json
    uv run python scripts/r5_cccc_q4_d1_preflight.py probe A:320x1280 [...] --out scratch/r5_cccc_q4_d1_prelaunch
"""

import argparse
import ast
import csv
import hashlib
import importlib.util
import io
import json
import math
import os
import platform
import re
import subprocess
import sys
import threading
import time
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "results" / "r5_cccc_q4_d1_prelaunch"
EXPECTED = OUT / "control_ab_expected.json"
EXPECTED_SHA256 = "19225c8e131701824f0ea5529b8edc06ca6b07a872eedc6cd8744263739cb454"  # frozen before the classifier was written
BASE_COMMIT = "9998575aa97198e52cc971df739c75886ba0474d"  # main at the start of the task; CI 40775140813: 5/5 SUCCESS
PROTOCOL = "docs/R5_CCCC_Q4_D1_PROTOCOL.md"
OP8_TOOL = "scripts/r5_cccc_q4_d1_c3plus_trajectories.py"
QAEB_TOOL = "scripts/r5_cccc_q4_energy_bound_premises_audit.py"
BR_TOOL = "scripts/r5_cccc_q4_bc_reference_audit.py"
SELF = "scripts/r5_cccc_q4_d1_preflight.py"
D1_START = "BLOCKED"
PROBE_LABEL = "PRELAUNCH_RESOURCE_PROBE (not D1_RESULT; no row value recorded)"
NOT_D1 = "PRELAUNCH preparation, not D1 data (D1_START = BLOCKED; protocol 12)"

# ---------------------------------------------------------------------------------------------------------------------------------
# the D1 sample as written in the protocol (2.2, 2.3, 3); transcribed, checked against the protocol text by the tests
# ---------------------------------------------------------------------------------------------------------------------------------

FAMILIES = (  # nominal r = l_y/l_x (rational, as written) and the grids (n_x, n_y) of levels 1-4 (protocol 2.2)
    ("1", ((50, 50), (100, 100), (200, 200), (400, 400))),
    ("3/2", ((48, 72), (100, 150), (200, 300), (400, 600))),
    ("2/3", ((48, 32), (96, 64), (198, 132), (396, 264))),
    ("5/2", ((48, 120), (100, 250), (200, 500), (400, 1000))),
    ("2/5", ((50, 20), (100, 40), (200, 80), (400, 160))),
    ("4", ((40, 160), (80, 320), (160, 640), (320, 1280))),
    ("1/4", ((48, 12), (100, 25), (200, 50), (400, 100))),
)
ANISOTROPIC = ((72, 40), (40, 72), (61, 37), (37, 61), (150, 48), (48, 150))  # protocol 2.3, geometry 1 x 1
BLOCK_D = ((9, 11, 1), (11, 15, 1), (14, 6, 1), (14, 6, 3))  # protocol 3, OP-3: geometry (l_x, l_y) NOT fixed
BLOCK_S = (("4", 400, 1600), ("1/4", 1600, 400))  # protocol 3, OP-6
MANDATORY_S = (1, 2, 3)  # D-054.2, item 2
S_MAX = 8
RH_NODES = 4096  # protocol 5 item 9: R-H where (n_x - 1)(n_y - 1) <= 4096
PROTOCOL_RH_GRIDS = ((50, 50), (48, 72), (48, 32), (50, 20), (100, 40), (48, 12), (100, 25), (72, 40), (40, 72), (61, 37), (37, 61))
PROPOSED_D, PROPOSED_Q = 1.0, 1.0  # OD-5: proposal (Q3 / D0, PL 1.5); the protocol does not state D, q - not an owner decision

# ---------------------------------------------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------------------------------------------


def git_blob_id(path: Path) -> str:
    """git blob id of the file content with LF line endings (as the OP-8 tool; independent of a CRLF checkout)."""
    data = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def canonical(o) -> bytes:
    return json.dumps(o, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _load(rel: str, name: str):
    """A QA script as a module (read-only use; executes its module-level code in THIS process)."""
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def platform_fingerprint() -> dict:
    import numpy as np

    return {"sys_platform": sys.platform, "machine": platform.machine(), "processor": platform.processor(),
            "python": platform.python_version(), "numpy": np.__version__}


def _same_hash_platform(fp: dict) -> bool:
    """The input hashes of s = 1 depend on numpy's sin (start set T_1): they are compared only on the recorded platform."""
    now = platform_fingerprint()
    return all(now[k] == fp.get(k) for k in ("sys_platform", "machine", "processor", "numpy"))


# ---------------------------------------------------------------------------------------------------------------------------------
# 1. SHA inventory (12.3, first item; task A1)
# ---------------------------------------------------------------------------------------------------------------------------------

COMPONENTS = (  # (id, role in the campaign, path or None = MISSING)
    ("protocol_D1", "protocol of the D1 campaign (APPROVED_PROTOCOL, D-055.2)", PROTOCOL),
    ("plan_PL", "QA plan PL (PLAN, D-049.6): rule of resources 6.3", "docs/R5_CCCC_PREQUALIFICATION.plan.md"),
    ("op8_tool", "QA path B-C with C-3+: own trajectories (OP-8, D-055.1)", OP8_TOOL),
    ("op8_tests", "tests of the OP-8 tool", "tests/reference/test_r5_cccc_q4_d1_c3plus_trajectories.py"),
    ("c3plus_and_control", "C-3+ assembly, its premises / exact check, variant CONTROL (second path for C; D-054.3)",
     "scripts/r5_cccc_q4_c_assembly_e4_audit.py"),
    ("classifier_campaign", "energy classifier of the campaign wired to D1 rows (protocol 4)", None),
    ("classifier_prelaunch_qa", "isolated QA classifier 4.2-4.4 + synthetic harness of controls A, B (this tool)", SELF),
    ("gamma_B1", "Gamma_up of path B1: gamma_b1, directed, gamma_up (GP 5.3; QA-gamma3)", "scripts/r5_cccc_q4_gamma_interval_audit.py"),
    ("gamma_audit", "gamma audit: helpers of B1; second path for gamma (VOK-2 identity, energies_path_b)",
     "scripts/r5_cccc_q4_gamma_audit.py"),
    ("energy_path_G4", "EQ path G-4 (pi_double, q_double, solve_c3plus, model_terms; PR #41)", "scripts/r5_cccc_q4_energy_bc_audit.py"),
    ("B_E_and_premises", "B_E (ST 5.1, 5.4) and premises U-1, U-2, U-3, N-7, binding (QA-EB, PR #44)", QAEB_TOOL),
    ("BR_reference", "certified reference of the half-step (BR, PR #39): BC-1 / TS-1 observations, second path for B-C", BR_TOOL),
    ("q3_script", "Q3 helpers loaded by BR and the gamma audit", "scripts/r5_cccc_q3_prequalification.py"),
    ("second_path_RH_RAC_RHP", "R-H, R-AC, R-HP, lambda_low (Q1b) of the half-step", "src/kkvim_reference/r5_cccc_halfstep.py"),
    ("row_pipeline", "pipeline of a D1 row: OP-8 trajectory -> G-4, B1, B_E, premises P-a...P-f -> classifier", None),
    ("block_C_driver", "R-AC chains on the 7 level-2 grids and the paired B-C (C-3+) of block C", None),
    ("B_E_independent_recount", "separate recount of B_E on a control sample (sect. 9, recommendation N-10)", None),
    ("control_V_harness", "positive control V (consistent perturbation of the factors; protocol 6)", None),
    ("control_G_harness", "control G: mutations of the computing path with declared detection channels (protocol 6)", None),
)
TOOL_ROOTS = (OP8_TOOL, "scripts/r5_cccc_q4_c_assembly_e4_audit.py", "scripts/r5_cccc_q4_gamma_interval_audit.py",
              "scripts/r5_cccc_q4_gamma_audit.py", "scripts/r5_cccc_q4_energy_bc_audit.py", QAEB_TOOL, BR_TOOL)


def _static_deps(rel: str) -> set[str]:
    """Project files a file uses: ``from kkvim_reference import x`` / ``from . import x`` / relative imports, and the scripts it
    loads by path (``ROOT / "scripts" / "y.py"``). Static (AST and string constants): nothing is executed."""
    src = (ROOT / rel).read_text(encoding="utf-8")
    out = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.ImportFrom):
            if node.module == "kkvim_reference" or (node.level == 1 and rel.startswith("src/kkvim_reference/") and not node.module):
                out.update(f"src/kkvim_reference/{a.name}.py" for a in node.names)
            elif node.module and node.module.startswith("kkvim_reference."):
                out.add("src/kkvim_reference/" + node.module.split(".", 1)[1].replace(".", "/") + ".py")
            elif node.level == 1 and node.module and rel.startswith("src/kkvim_reference/"):
                out.add(f"src/kkvim_reference/{node.module}.py")
        elif isinstance(node, ast.Import):
            out.update("src/kkvim_reference/" + a.name.split(".", 1)[1] + ".py" for a in node.names
                       if a.name.startswith("kkvim_reference."))
    out.update(f"scripts/{m}" for m in re.findall(r'"scripts"\s*/\s*"([A-Za-z0-9_]+\.py)"', src))
    if any(p.startswith("src/kkvim_reference/") for p in out):
        out.add("src/kkvim_reference/__init__.py")
    return {p for p in out if (ROOT / p).exists() and p != rel}


def dependency_closure(roots=TOOL_ROOTS) -> list[str]:
    seen, todo = set(), list(roots)
    while todo:
        rel = todo.pop()
        if rel in seen:
            continue
        seen.add(rel)
        todo.extend(_static_deps(rel) - seen)
    return sorted(seen)


def sha_inventory() -> dict:
    comps = []
    for cid, role, rel in COMPONENTS:
        if rel is None:
            comps.append({"id": cid, "role": role, "path": None, "state": "MISSING", "blob": None})
        else:
            comps.append({"id": cid, "role": role, "path": rel, "state": "PRESENT" if rel != SELF else "PRESENT (this branch, new)",
                          "blob": git_blob_id(ROOT / rel)})
    deps = dependency_closure()
    listed = {c["path"] for c in comps if c["path"]}
    modules = [{"path": p, "blob": git_blob_id(ROOT / p), "used_by_closure_of": "TOOL_ROOTS"} for p in deps if p not in listed]
    return {"base_commit": BASE_COMMIT, "blob_rule": "git blob id of the LF content (= git rev-parse <commit>:<path> for these files)",
            "components": comps, "dependency_closure": deps, "other_used_files": modules,
            "missing": [c["id"] for c in comps if c["state"] == "MISSING"]}


# ---------------------------------------------------------------------------------------------------------------------------------
# 2. the 102 mandatory rows (12.3, second item; task A2)
# ---------------------------------------------------------------------------------------------------------------------------------


def block_a_grids() -> list[dict]:
    out = []
    for r, grids in FAMILIES:
        for level, (nx, ny) in enumerate(grids, start=1):
            out.append({"family": f"r={r}", "r": r, "level": level, "nx": nx, "ny": ny, "lx": 1.0, "ly": float(Fraction(r))})
    for nx, ny in ANISOTROPIC:
        out.append({"family": "1x1", "r": "1 (anisotropic grid)", "level": "aniso", "nx": nx, "ny": ny, "lx": 1.0, "ly": 1.0})
    return out


def row_key(block: str, family: str, nx: int, ny: int, N: int, s: int) -> str:
    return f"D1-{block}|{family}|{nx}x{ny}|N{N}|s{s}"


def direction_of(s: int) -> str:
    return "Y" if s % 2 == 1 else "X"  # D-049.3 item 2: the first half-step by y with given functions of x


def geometry(g: dict) -> dict:
    lx, ly, nx, ny = g["lx"], g["ly"], g["nx"], g["ny"]
    hx, hy = Fraction(lx) / nx, Fraction(ly) / ny
    return {"lx": repr(lx), "lx_hex": lx.hex(), "ly": repr(ly), "ly_hex": ly.hex(),
            "ly_equals_r_exactly": Fraction(ly) == (Fraction(g["r"]) if g["family"] != "1x1" else Fraction(1)),
            "h_double_equal": (lx / nx) == (ly / ny), "h_exact_equal": hx == hy, "cells": nx * ny,
            "rh_eligible": (nx - 1) * (ny - 1) <= RH_NODES}


def config_record(g: dict) -> dict:
    """Configuration of the trajectory of one block-A grid (all rows s = 1..8 share it). D and q are PROPOSALS (OD-5)."""
    return {"tool": OP8_TOOL, "block": "A", "lx_hex": g["lx"].hex(), "ly_hex": g["ly"].hex(), "nx": g["nx"], "ny": g["ny"], "N": 1,
            "D_hex": PROPOSED_D.hex(), "q_hex": PROPOSED_Q.hex(), "D_q_status": "PROPOSAL OD-5 (OWNER_DECISION_REQUIRED)",
            "s_max": S_MAX, "orientation": "first half-step Y with given functions of x (D-049.3 item 2)",
            "start": "T_1 = sin^2(pi x/a) on the interior x nodes (hs.start_set(n_x, 1); D-049.3 item 1)",
            "stop_rule": "none: s = 1..8 fixed (protocol 2.1)"}


def known_input_hashes(g: dict, tj) -> dict:
    """sha256 (OP-8 ``array_sha``: shape + little-endian bytes) of the inputs of s = 1 known before the run."""
    import numpy as np

    from kkvim_reference import r5_cccc_halfstep as hs

    given = hs.start_set(g["nx"], 1)
    comp = np.zeros((g["ny"] - 1, 1))
    return {"given_in_sha": tj.array_sha(given), "companion_in_sha": tj.array_sha(comp)}


DYNAMIC_PROCEDURE = {
    "status": "DEFINED, NOT EXECUTED (no trajectory is computed to obtain future hashes)",
    "per_half_step_fields_of_the_OP8_journal": [
        "given_in_sha", "companion_in_sha", "solved_set_sha", "companion_solved_sha", "prev_field_before_QR_sha",
        "prev_field_after_QR_sha", "C3plus_sha", "A_sha", "B_sha", "F_sha", "C_old_sha", "V_bc_sha", "field_s_sha",
        "field_s_after_normalisation_sha", "next_given_sha", "next_companion_sha"],
    "hash_function": "OP-8 array_sha: sha256(str(shape) + little-endian float64 bytes, C order)",
    "link_rule": "s >= 2: given_in_sha(s) == next_given_sha(s - 1) and companion_in_sha(s) == next_companion_sha(s - 1) of the SAME "
                 "trajectory journal (OP-8 records input_link_bitwise); s = 1: given_in_sha and companion_in_sha == this manifest",
    "row_values_to_record_by_the_row_pipeline (MISSING)": [
        "P_hat_prev, P_hat_s, Q_hat (float.hex)", "d_hat sha (array_sha of fl(w_prev - w_s))", "gamma_down, gamma_up, Gamma_up "
        "(float.hex)", "B_E,s (exact rational: sha256 of 'p/q')", "tau'_s (exact rational: sha256 of 'p/q')",
        "premise states P-a...P-f with reasons", "status of the row (protocol 4.2)"],
    "binding_P_a": "a second run of the row in a separate process reproduces every recorded value bitwise (QA-EB item 3 pattern); "
                   "the comparison is made against the first run's journal, never against expected numbers",
    "before_the_run": "every s = 1 input recomputed and compared with this manifest; a mismatch is BLOCKED_CAMPAIGN (protocol 4.4)",
    "platform_note": "the s = 1 hashes depend on numpy's sin (start set): valid on the platform recorded in 'hash_platform' only",
}


def mandatory_rows(tj=None) -> list[dict]:
    rows = []
    for g in block_a_grids():
        geo = geometry(g)
        cfg = config_record(g)
        cfg_sha = sha256_bytes(canonical(cfg))
        known = known_input_hashes(g, tj) if tj is not None else {}
        for s in MANDATORY_S:
            d = direction_of(s)
            n_f, n_s = (g["nx"], g["ny"]) if d == "Y" else (g["ny"], g["nx"])
            row = {"key": row_key("A", g["family"], g["nx"], g["ny"], 1, s), "block": "A", "family": g["family"], "r": g["r"],
                   "level": g["level"], "nx": g["nx"], "ny": g["ny"], **geo, "N": 1, "s": s, "direction": d, "n_f": n_f, "n_s": n_s,
                   "mandatory": True, "config_sha256": cfg_sha}
            if s == 1:
                row["inputs"] = {"state": "KNOWN_BEFORE_RUN", **known}
            else:
                row["inputs"] = {"state": "DYNAMIC", "given_in_sha": f"= next_given_sha of {row_key('A', g['family'], g['nx'], g['ny'], 1, s - 1)}",
                                 "companion_in_sha": f"= next_companion_sha of {row_key('A', g['family'], g['nx'], g['ny'], 1, s - 1)}"}
            rows.append(row)
    return rows


def planned_row_counts() -> dict:
    """Arithmetic of protocol 3 (checked, not executed)."""
    a = len(block_a_grids()) * S_MAX
    b = 2 * 3 * len(FAMILIES) * S_MAX
    c = len(FAMILIES) * 3 * S_MAX
    d = len(BLOCK_D) * S_MAX
    s_ = len(BLOCK_S) * S_MAX
    return {"A": a, "B": b, "C_paired": c, "C_RAC_reference_chain": c, "D": d, "S": s_, "classified": a + b + c + d + s_,
            "mandatory": len(block_a_grids()) * len(MANDATORY_S)}


# ---------------------------------------------------------------------------------------------------------------------------------
# 3. second-path sample (OP-4) and B_E recount sample (sect. 9); task A3
# ---------------------------------------------------------------------------------------------------------------------------------

SAMPLE_FEATURES = ("block", "family", "level", "nx", "ny", "s")
SAMPLE_RULE = {
    "selected_by": "features of the row only (block, family, level, n_x, n_y, s); no value of any computation is read",
    "SP-RH": "R-H direct with its own exact-residual bound: block A rows with (n_x - 1)(n_y - 1) <= 4096 (the 11 grids of protocol 5 "
             "item 9), s = 1, 2, 3",
    "SP-LARGE": "large grids: exact A, B, C, F against the doubles and the exact residual R-HP with certified lambda_low: block A "
                "level-4 grids of the 7 families and the anisotropic grids with (n_x - 1)(n_y - 1) > 4096, s = 1, 2, 3",
    "SP-GAMMA": "exact gamma_s by a different path (VOK-2 identity, energies_path_b, or the 13-point form a_h) against [gamma_down, "
                "gamma_up] of B1: SP-RH union SP-LARGE",
    "SP-BR": "certified reference BR for the B-C solution: SP-RH union SP-LARGE",
    "SP-C": "exact rational C from the stored binary64 set (reference) and variant CONTROL in double (D-054.3 item 2): SP-RH union "
            "SP-LARGE",
    "SP-BE": "separate recount of B_E (recommendation, sect. 9): level-1 grids of the 7 families and the anisotropic pair (61, 37), "
             "(37, 61); s = 2, 3 (both fields of the row nonzero)",
    "block_D": "R-H on every block-D N = 1 row (protocol 5 item 9): PENDING the geometry (OD-1); (14, 6), N = 3 blocked (PRE_D1_RANK)",
}


def sample_paths(f: dict) -> list[str]:
    """Second-path memberships of one row from its declared features only (SAMPLE_FEATURES)."""
    if f["block"] != "A" or f["s"] not in MANDATORY_S:
        return []
    rh = (f["nx"] - 1) * (f["ny"] - 1) <= RH_NODES
    large = f["level"] == 4 or (f["level"] == "aniso" and not rh)
    out = []
    if rh:
        out.append("SP-RH")
    if large:
        out.append("SP-LARGE")
    if rh or large:
        out += ["SP-GAMMA", "SP-BR", "SP-C"]
    if f["s"] in (2, 3) and (f["level"] == 1 or (f["family"] == "1x1" and (f["nx"], f["ny"]) in ((61, 37), (37, 61)))):
        out.append("SP-BE")
    return sorted(out)


def second_path_sample(rows: list[dict]) -> dict:
    sel = []
    for r in rows:
        paths = sample_paths({k: r[k] for k in SAMPLE_FEATURES})
        if paths:
            sel.append({"key": r["key"], "paths": paths})
    counts = {p: sum(p in x["paths"] for x in sel) for p in ("SP-RH", "SP-LARGE", "SP-GAMMA", "SP-BR", "SP-C", "SP-BE")}
    return {"rule": SAMPLE_RULE, "rows": sel, "counts": counts, "sha256": sha256_bytes(canonical(sel)),
            "status": "PROPOSAL fixed by rule before any D1 result; adoption OWNER_DECISION_REQUIRED (OD-11)"}


# ---------------------------------------------------------------------------------------------------------------------------------
# 4. proposals for the parameters the accepted documents do not fix (task A4)
# ---------------------------------------------------------------------------------------------------------------------------------

OWNER_DECISIONS = (
    {"id": "OD-1", "topic": "geometry (l_x, l_y) of block D (OP-3; protocol 3, 12.3)",
     "proposal": "l_x = l_y = 1 for (9, 11), (11, 15), (14, 6) (as the anisotropic grids 2.3: every input exact, h_x != h_y by "
                 "construction)", "alternative": "l_x = 1, l_y = fl(n_y/n_x) (h_x ~ h_y; fl inexact for 11/9, 15/11, 3/7)"},
    {"id": "OD-2", "topic": "infinity norm of the set for N >= 2 (form BC-1; protocol 5.1, 12.3)",
     "proposal": "max |V_ij| over all entries of the solved set (n_s - 1) x N in the basis U~ after QR, the basis recorded",
     "alternative": "max over the nodes of the field |(U~ V^T)_ij| (basis-independent)", "note": "N >= 2 runs are blocked (PRE_D1_RANK)"},
    {"id": "OD-3", "topic": "path and inputs of the paired half-steps of block C (protocol 3.1, 12.3)",
     "proposal": "R-AC chain r5_cccc_sequence.run_sequence(SequenceConfig(1.0, n_x, fl(r), n_y, N, D, q, s_max=8, first='Y', "
                 "qr=True)) on the 7 level-2 grids, stored NPZ hashed before the paired run; paired B-C by the EQ functions bc_input, "
                 "run_halfstep, solve_c3plus (blob of scripts/r5_cccc_q4_energy_bc_audit.py), single-threaded (solve_c3plus replaces "
                 "bc.bc_coefficients globally); N = 2, 3 only after PRE_D1_RANK is resolved", "missing": "block_C_driver"},
    {"id": "OD-4", "topic": "detection channels of control G (protocol 6, 12.3)",
     "proposal": "G-1 sign of gamma: second path SP-GAMMA (E1/E2 do not see it: Gamma_up = |gamma|); G-2 gamma of another half-step: "
                 "binding P-a + SP-GAMMA; G-3 Q^ from another d^: binding P-a (d^ hash) + E2; G-4 energy of the wrong field: binding "
                 "P-a + E1/E2; G-5 a term of B_E dropped: SP-BE recount only", "gap": "outside SP-GAMMA / SP-BE the mutations G-1 "
                 "and G-5 have no declared channel (UNRESOLVED of control G there)", "missing": "control_G_harness"},
    {"id": "OD-5", "topic": "material and load (not stated in the D1 protocol)",
     "proposal": "D = q = 1 (Q3 / D0 report 'D = q = 1', PL 1.5; default of the OP-8 tool - a code default is NOT an owner decision)"},
    {"id": "OD-6", "topic": "platform, versions, BLAS (12.3)",
     "proposal": "the QA machine of this manifest (hash_platform) with the uv.lock versions; Python of the run fixed (D0 / EQ / QA-EB "
                 "used 3.13.15, this preflight 3.14.8); OPENBLAS_NUM_THREADS = OMP_NUM_THREADS = MKL_NUM_THREADS = 1 set by the "
                 "launcher and recorded (the OP-8 tool does not set them, N-4 of its review); one process per trajectory"},
    {"id": "OD-7", "topic": "conditions of PL 6.3 for a B-C row (12.3, 12.5)",
     "proposal": "RAM of the machine = total physical memory (GlobalMemoryStatusEx); peak = peak working set of the row process; "
                 "'factorization' = wall time of each stage containing a factorization (OP-8 B-C half-step; BR: exact assembly, "
                 "decimal Cholesky ladder, refinement, lambda_low) <= 30 min each; 'refinement converges' = BR refine_converged and a "
                 "certified lambda_low (BR's own rule)"},
    {"id": "OD-8", "topic": "premises of C-3+ (P-1...P-5, OP-8) in the row status",
     "proposal": "the classifier of protocol 4 knows only P-a...P-f; a row whose C-3+ premises are UNRESOLVED or whose exact bound "
                 "check fails needs a declared status (e.g. UNRESOLVED_CERTIFICATE with reason 'C-3+') - the protocol is silent"},
    {"id": "OD-9", "topic": "resource probes beyond s = 1",
     "proposal": "permission to probe s = 2 of the largest configurations (needs the s = 1 B-C result as input); s = 1 has a zero "
                 "previous field and does not bound the cost of s >= 2"},
    {"id": "OD-10", "topic": "the D1 row pipeline (MISSING)",
     "proposal": "a separate QA task: adapter OP-8 trajectory -> QA-EB analyse_row-equivalent per row with binding by a second run, "
                 "-> classifier of this tool (blob fixed); its own review; controls A and B repeated on it"},
    {"id": "OD-11", "topic": "second-path sample and B_E recount sample", "proposal": "adopt SAMPLE_RULE (manifest 'second_path')",
     "missing": "B_E_independent_recount (no second implementation of B_E exists)"},
    {"id": "OD-12", "topic": "platform dependence of the input hashes",
     "proposal": "the s = 1 hashes are valid on the recorded platform only (numpy sin); either the platform is fixed (OD-6) or the "
                 "start sets are stored as fixed arrays and hashed"},
)


BLOCKED_PROBES_N3 = (("B", "4", 160, 640), ("B", "5/2", 200, 500), ("C", "4", 80, 320), ("C", "5/2", 100, 250))  # protocol 8


def blocked_probe_plan(tj) -> dict:
    """Inputs (known before the run) and plan of the N = 3 probes of blocks B and C; NOT executed (PRE_D1_RANK)."""
    import numpy as np

    from kkvim_reference import r5_cccc_halfstep as hs

    out = []
    for block, r, nx, ny in BLOCKED_PROBES_N3:
        ly = float(Fraction(r))
        out.append({"probe": f"{block}:{nx}x{ny}:N3", "block": block, "r": r, "lx_hex": (1.0).hex(), "ly_hex": ly.hex(), "nx": nx,
                    "ny": ny, "N": 3, "s": 1, "start": "T_3 (hs.start_set(n_x, 3))",
                    "given_in_sha": tj.array_sha(hs.start_set(nx, 3)), "companion_in_sha": tj.array_sha(np.zeros((ny - 1, 3))),
                    "plan": "same stages as the N = 1 probe; block C adds the R-AC half-step of r5_cccc_sequence on the same input",
                    "state": "BLOCKED: PRE_D1_RANK (D-049.5 requires a stop before QR at lost or indistinguishable rank; the OP-8 tool "
                             "continues QR with rank_decision = UNRESOLVED as RESEARCH_ONLY) - needs a separate owner permission"})
    return {"status": "NOT_EXECUTED", "probes": out}


def build_manifest() -> dict:
    tj = _load(OP8_TOOL, "_op8_tool_for_manifest")
    rows = mandatory_rows(tj)
    keys = [r["key"] for r in rows]
    if len(keys) != 102 or len(set(keys)) != 102:
        raise RuntimeError("the mandatory list must have 102 unique keys")
    return {
        "document": "R5-V Q4 D1 PRELAUNCH manifest (protocol 12, item 3)", "manifest_state": "INCOMPLETE - NOT FROZEN (OWNER_DECISION_"
        "REQUIRED items open; components MISSING)", "d1_start": D1_START, "data_status": NOT_D1,
        "sha_inventory": sha_inventory(),
        "hash_platform": platform_fingerprint(),
        "sample_arithmetic": planned_row_counts(),
        "mandatory_rows": rows, "mandatory_keys_sha256": sha256_bytes(canonical(keys)),
        "dynamic_hash_procedure": DYNAMIC_PROCEDURE,
        "second_path": second_path_sample(rows),
        "owner_decisions": [{**d, "status": "OWNER_DECISION_REQUIRED"} for d in OWNER_DECISIONS],
        "blocked_probes_N23": blocked_probe_plan(tj),
        "controls_expected_sha256": EXPECTED_SHA256,
    }


def manifest_csv(rows: list[dict]) -> str:
    cols = ("key", "block", "family", "level", "nx", "ny", "lx", "ly", "ly_hex", "ly_equals_r_exactly", "h_double_equal",
            "h_exact_equal", "N", "s", "direction", "n_f", "n_s", "rh_eligible", "config_sha256", "input_state", "given_in_sha",
            "companion_in_sha")
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(cols)
    for r in rows:
        flat = {**r, "input_state": r["inputs"]["state"], "given_in_sha": r["inputs"]["given_in_sha"],
                "companion_in_sha": r["inputs"]["companion_in_sha"]}
        w.writerow([flat[c] for c in cols])
    return buf.getvalue()


def write_manifest(out: Path = OUT) -> dict:
    m = build_manifest()
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(m, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    (out / "mandatory_rows.csv").write_text(manifest_csv(m["mandatory_rows"]), encoding="utf-8", newline="\n")
    return m


# ---------------------------------------------------------------------------------------------------------------------------------
# 5. isolated QA classifier (protocol 4.2-4.4); task B
# ---------------------------------------------------------------------------------------------------------------------------------

PREMISES = ("P-a", "P-b", "P-c", "P-d", "P-e", "P-f")
PREMISE_NAMES = {"P-a": "П-а data binding", "P-b": "П-б U-1", "P-c": "П-в U-2, mode, finiteness", "P-d": "П-г U-3 grid constants",
                 "P-e": "П-д N-7 exact zero", "P-f": "П-е applicability of B_E"}
ESTABLISHED, NOT_ESTABLISHED, NOT_CHECKABLE = "ESTABLISHED", "NOT_ESTABLISHED", "NOT_CHECKABLE"
PASS, FAIL_COMPUTATION = "PASS", "FAIL_COMPUTATION"
UNRES_RESOLVABILITY, UNRES_CERT = "UNRESOLVED_RESOLVABILITY", "UNRESOLVED_CERTIFICATE"
UNRES_RESOURCES, UNRESOLVED = "UNRESOLVED_RESOURCES", "UNRESOLVED"
BLOCKED_CAMPAIGN, FAIL_CANDIDATE = "BLOCKED_CAMPAIGN", "FAIL_CANDIDATE"
UNRESOLVED_CANDIDATE, PASS_CANDIDATE = "UNRESOLVED_CANDIDATE", "PASS_CANDIDATE"


def exact_value(v) -> Fraction | None:
    """Exact rational of a recorded value; None if missing or not finite. Strings 'p/q' or decimal; floats taken exactly."""
    if v is None:
        return None
    if isinstance(v, bool):
        raise TypeError("a boolean is not a value")
    if isinstance(v, Fraction):
        return v
    if isinstance(v, int):
        return Fraction(v)
    if isinstance(v, float):
        return Fraction(v) if math.isfinite(v) else None
    if isinstance(v, str):
        return None if v.strip().lower().lstrip("+-") in ("inf", "infinity", "nan") else Fraction(v)
    raise TypeError(f"unsupported value {v!r}")


def classify_row(inp: dict) -> dict:
    """Status of one row, protocol 4.2 (the decision tree, in this order; exact comparisons, no tolerance)."""
    if not inp.get("executed", False):
        ne = inp.get("not_executed") or {}
        reason = ne.get("reason") or "not executed (no reason recorded)"
        if ne.get("resources") and ne.get("measured"):
            return {"status": UNRES_RESOURCES, "e1": None, "e2": None, "e_diag": False, "reason_premises": [],
                    "reasons": [f"PL 6.3: {reason}"], "measured": dict(ne["measured"])}
        if ne.get("resources"):
            reason += " (resource rule claimed without measured probe values: not the rule of PL 6.3)"
        return {"status": UNRESOLVED, "e1": None, "e2": None, "e_diag": False, "reason_premises": [], "reasons": [reason]}
    prem = inp.get("premises") or {}
    unknown = set(prem) - set(PREMISES)
    if unknown:
        raise ValueError(f"unknown premise ids {sorted(unknown)}")
    bad, reasons = {}, {}
    for p in PREMISES:
        st = prem.get(p)
        state, why = (NOT_CHECKABLE, "not provided") if st is None else (st.get("state"), st.get("reason", ""))
        if state not in (ESTABLISHED, NOT_ESTABLISHED, NOT_CHECKABLE):
            raise ValueError(f"{p}: unknown state {state!r}")
        if state != ESTABLISHED:
            bad[p] = state
            reasons[p] = f"{p} ({PREMISE_NAMES[p]}): {state}: {why}"
    dP, Q, tau = (exact_value(inp.get(k)) for k in ("dP", "Q", "tau"))
    if dP is None or Q is None or tau is None:
        bad.setdefault("P-c", NOT_ESTABLISHED)
        reasons["P-c"] = reasons.get("P-c", f"P-c ({PREMISE_NAMES['P-c']}): {NOT_ESTABLISHED}") + "; dP, Q^ or tau' missing or not finite"
        order = [p for p in PREMISES if p in bad]
        return {"status": UNRES_CERT, "e1": None, "e2": None, "e_diag": False, "reason_premises": order,
                "reasons": [reasons[p] for p in order]}
    if tau < 0:
        raise ValueError("tau' < 0: tau' is a sum of upper bounds of absolute values")
    e1 = dP >= -tau  # E1, non-strict
    e2 = abs(dP - Q) <= tau  # E2, non-strict
    if bad:  # 4.2 item 1: E1/E2 and resolvability are evaluated and recorded, not as verified
        order = [p for p in PREMISES if p in bad]
        return {"status": UNRES_CERT, "e1": e1, "e2": e2, "e_diag": not (e1 and e2), "resolvable_unverified": Q > tau,
                "reason_premises": order, "reasons": [reasons[p] for p in order]}
    if not (e1 and e2):
        status = FAIL_COMPUTATION  # 4.2 item 2
    elif Q <= tau:
        status = UNRES_RESOLVABILITY  # 4.2 item 3 (strict sign of resolvability; Q = 0 included)
    else:
        status = PASS  # 4.2 item 4
    return {"status": status, "e1": e1, "e2": e2, "e_diag": False, "reason_premises": [], "reasons": []}


def mandatory_index(rows: list[dict]) -> dict:
    return {r["key"]: (r["block"], r["N"], r["s"]) for r in rows}


def campaign_verdict(rows: list[dict], mandatory: dict, input_hash_mismatch=(), control_v_fail: bool = False) -> dict:
    """Protocol 4.3, 4.4. ``rows``: dicts with key, block, N, s, mutated (injected control row) and input (classify_row).
    ``mandatory``: key -> (block, N, s) of the 102 rows of the manifest."""
    if len(mandatory) != 102:
        raise ValueError("the mandatory list must have exactly 102 rows (protocol 4.3)")
    seen, done = set(), []
    for r in rows:
        k = r["key"]
        if k in seen:
            raise ValueError(f"duplicate row key {k}")
        seen.add(k)
        if k in mandatory and not r.get("mutated") and (r["block"], r["N"], r["s"]) != mandatory[k]:
            raise ValueError(f"mandatory key with block/N/s different from the manifest: {k}")
        done.append({**r, "result": classify_row(r["input"])})
    camp = [d for d in done if not d.get("mutated")]
    injected = sorted(d["key"] for d in done if d.get("mutated"))
    e_diag = sorted(d["key"] for d in camp if d["result"]["e_diag"])
    block = [f"input hash mismatch before the run: {k}" for k in input_hash_mismatch]
    if control_v_fail:
        block.append("FAIL of the positive control V (protocol 6)")
    block += [f"FAIL_COMPUTATION on a non-mutated row: {d['key']}" for d in camp if d["result"]["status"] == FAIL_COMPUTATION]
    unresolved_all = sorted((d["key"], d["result"]["status"], d["result"]["reasons"]) for d in camp
                            if d["result"]["status"] not in (PASS, FAIL_COMPUTATION))
    if block:
        return {"campaign_verdict": BLOCKED_CAMPAIGN, "candidate_verdict": None, "block_reasons": block, "e_diag_keys": e_diag,
                "injected_keys": injected, "unresolved_all": unresolved_all}
    by_key = {d["key"]: d["result"]["status"] for d in camp}
    st = {k: by_key.get(k, UNRESOLVED) for k in mandatory}  # an absent row stays in the denominator as UNRESOLVED
    if any(v == UNRES_RESOLVABILITY for v in st.values()):
        cand = FAIL_CANDIDATE
    elif all(v == PASS for v in st.values()):
        cand = PASS_CANDIDATE
    else:
        cand = UNRESOLVED_CANDIDATE
    counts = {v: sum(1 for x in st.values() if x == v) for v in sorted(set(st.values()))}
    return {"campaign_verdict": cand, "candidate_verdict": cand, "denominator": len(mandatory), "pass": counts.get(PASS, 0),
            "counts": counts, "unresolved_keys": sorted(k for k, v in st.items() if v != PASS),
            "absent_keys": sorted(k for k in mandatory if k not in by_key), "e_diag_keys": e_diag, "injected_keys": injected,
            "unresolved_all": unresolved_all}


# ---------------------------------------------------------------------------------------------------------------------------------
# 6. harness of controls A and B against the frozen expectations
# ---------------------------------------------------------------------------------------------------------------------------------


def load_expected(path: Path = EXPECTED) -> dict:
    raw = path.read_bytes().replace(b"\r\n", b"\n")  # LF content: a CRLF checkout (core.autocrlf) is the same file
    if sha256_bytes(raw) != EXPECTED_SHA256:
        raise RuntimeError("the frozen expectations were modified (sha256 differs): controls are not run")
    return json.loads(raw)


def _template_row(tpl: dict) -> dict:
    t = json.loads(json.dumps(tpl))
    if t.get("executed"):
        spec = t["premises"]
        t["premises"] = {p: {"state": ESTABLISHED} for p in PREMISES} if spec == "ALL_ESTABLISHED" else {
            p: {"state": spec.get(p, ESTABLISHED), "reason": "template"} for p in PREMISES}
    return t


def aggregation_rows(case: dict, templates: dict, mandatory_rows_: list[dict]) -> list[dict]:
    spec = case["rows"]
    rows = []
    if spec.get("mandatory"):
        for r in mandatory_rows_:
            if r["key"] in spec.get("drop", []):
                continue
            tpl = spec.get("set", {}).get(r["key"], spec["mandatory"])
            row = {"key": r["key"], "block": "A", "N": 1, "s": r["s"], "mutated": False, "input": _template_row(templates[tpl])}
            row.update(spec.get("set_fields", {}).get(r["key"], {}))
            rows.append(row)
    for x in spec.get("extra", []):
        rows.append({"key": x["key"], "block": x["block"], "N": x["N"], "s": x["s"], "mutated": bool(x.get("mutated")),
                     "input": _template_row(templates[x["template"]])})
    return rows


def _compare_row(exp: dict, act: dict) -> bool:
    return all(act.get(k) == v for k, v in exp.items())


def _compare_agg(exp: dict, act: dict) -> bool:
    if "error" in exp:
        return act.get("error") is not None and exp["error"] in act["error"]
    for k, v in exp.items():
        if k == "block_reasons_contain":
            if not all(any(s in r for r in act.get("block_reasons", [])) for s in v):
                return False
        elif k == "unresolved_count":
            if len(act.get("unresolved_keys", [])) != v:
                return False
        elif act.get(k) != v:
            return False
    return True


def _child(cmd: list[str], timeout: float = 600) -> dict:
    p = subprocess.run([sys.executable, str(ROOT / SELF), *cmd], cwd=ROOT, capture_output=True, text=True, timeout=timeout)
    if p.returncode != 0:
        raise RuntimeError(f"child {cmd} failed ({p.returncode}): {p.stderr[-2000:]}")
    return json.loads(p.stdout.strip().splitlines()[-1])


def run_controls(path: Path = EXPECTED) -> dict:
    exp = load_expected(path)
    rows_manifest = mandatory_rows(None)
    mand = mandatory_index(rows_manifest)
    res_rows = []
    for c in exp["row_cases"]:
        act = classify_row(c["input"])
        sub = {k: act.get(k) for k in c["expected"]}
        res_rows.append({"id": c["id"], "control": c["control"], "expected": c["expected"], "actual": sub,
                         "reasons": act["reasons"], "match": _compare_row(c["expected"], act)})
    res_agg = []
    for c in exp["aggregation_cases"]:
        try:
            act = campaign_verdict(aggregation_rows(c, exp["templates"], rows_manifest), mand, c.get("input_hash_mismatch", ()),
                                   c.get("control_v_fail", False))
        except ValueError as e:
            act = {"error": str(e)}
        small = {k: act.get(k) for k in ("campaign_verdict", "candidate_verdict", "denominator", "pass", "e_diag_keys", "injected_keys",
                                         "block_reasons", "error")}
        small["unresolved_keys"] = act.get("unresolved_keys", [])[:5] + (["..."] if len(act.get("unresolved_keys", [])) > 5 else [])
        res_agg.append({"id": c["id"], "expected": c["expected"], "actual": small, "match": _compare_agg(c["expected"], act)})
    det = detector_cases()
    pass_like = exp["value_situations"]["PASS_LIKE"]
    res_det = []
    for c in exp["detector_cases"]:
        d = det.get(c["id"])
        if d is None or d.get("not_executed"):
            res_det.append({"id": c["id"], "result": "NOT_EXECUTED", "reason": (d or {}).get("not_executed", "no detector"),
                            "match": None})
            continue
        out = {}
        for kind in ("injected", "clean"):
            prem = {p: {"state": ESTABLISHED} for p in PREMISES}
            prem.update(d[kind]["premises"])
            act = classify_row({"executed": True, "premises": prem, **pass_like})
            e = c[f"expected_{kind}"]
            ok = act["status"] == e["status"] and act["reason_premises"] == e["reason_premises"] and act["status"] not in e.get(
                "not_status", [])
            out[kind] = {"status": act["status"], "reason_premises": act["reason_premises"], "reasons": act["reasons"], "match": ok}
        res_det.append({"id": c["id"], "premise": c["premise"], "injection": c["injection"], "evidence": d.get("evidence"),
                        **out, "match": out["injected"]["match"] and out["clean"]["match"]})
    a_ok = all(r["match"] for r in res_rows if r["control"] == "A") and all(r["match"] for r in res_agg)
    b_syn_ok = all(r["match"] for r in res_rows if r["control"] == "B")
    b_det = [r for r in res_det if r["match"] is not None]
    b_det_ok = bool(b_det) and all(r["match"] for r in b_det)
    return {"document": "R5-V Q4 D1 PRELAUNCH: results of controls A and B (synthetic harness)", "data_status": NOT_D1,
            "expected_file": str(path.relative_to(ROOT)).replace("\\", "/"), "expected_sha256": EXPECTED_SHA256,
            "classifier": {"path": SELF, "blob": git_blob_id(ROOT / SELF)}, "environment": platform_fingerprint(),
            "summary": {"row_cases": len(res_rows), "row_cases_matching": sum(r["match"] for r in res_rows),
                        "aggregation_cases": len(res_agg), "aggregation_matching": sum(r["match"] for r in res_agg),
                        "detector_cases": len(res_det), "detector_matching": sum(1 for r in res_det if r["match"]),
                        "detector_not_executed": [r["id"] for r in res_det if r["match"] is None],
                        "control_A_all_match": a_ok, "control_B_synthetic_all_match": b_syn_ok,
                        "control_B_detector_functions_all_match": b_det_ok},
            "row_cases": res_rows, "aggregation_cases": res_agg, "detector_cases": res_det}


# --- detector injections (child processes; they import the QA-EB tool, whose import sets BLAS thread variables) ------------------

def _st(ok: bool, why: str, checkable: bool = True) -> dict:
    return {"state": ESTABLISHED if ok else (NOT_ESTABLISHED if checkable else NOT_CHECKABLE), "reason": why}


def _detectors_child() -> dict:
    """All detector cases except the arithmetic-mode ones, in this (child) process."""
    import numpy as np

    from kkvim_reference import r5_cccc_energy as en
    from kkvim_reference import r5_cccc_halfstep as hs

    qa = _load(QAEB_TOOL, "_qaeb_for_controls")
    tj = _load(OP8_TOOL, "_op8_for_controls")
    out = {}
    # P-a: binding of a row value (QA-EB _bind with a repr-printed historical value, as rows.csv)
    v = 0.1 + 0.2
    for kind, val in (("injected", float(np.nextafter(v, np.inf))), ("clean", v)):
        b = qa._bind({"P_hat_s": repr(v)}, "P_hat_s", val)
        out.setdefault("B-D-a1", {})[kind] = {"premises": {"P-a": _st(b is True, f"QA-EB _bind -> {b}")}}
    # P-a: input array with one entry moved by 1 ulp (hash fixed before the run)
    g0 = hs.start_set(10, 1)
    fixed = tj.array_sha(g0)
    g1 = g0.copy()
    g1[3, 0] = np.nextafter(g1[3, 0], np.inf)
    for kind, arr in (("injected", g1), ("clean", g0.copy())):
        h = tj.array_sha(arr)
        out.setdefault("B-D-a2", {})[kind] = {"premises": {"P-a": _st(h == fixed, f"array_sha {h[:12]} vs fixed {fixed[:12]}")}}
    # P-b: U-1
    X = np.array([[1.0], [0.75], [-0.5], [0.3125], [2.0]])
    Y = np.array([[0.5], [-1.25], [3.0], [0.625]])
    W = X @ Y.T
    W1 = W.copy()
    W1[1, 2] = W1[1, 2] * (1 + 2.0**-40)
    for kind, w in (("injected", W1), ("clean", W)):
        f = qa.formation_check(X, Y, w)
        out.setdefault("B-D-b", {})[kind] = {"premises": {"P-b": _st(f["holds"], f"formation_check holds={f['holds']}, "
                                                                                 f"violations={f['violations']}")}}
    # P-c: subnormal operand in a checked multiplication
    for kind, (a, b) in (("injected", (2.0**-1060, 1.0)), ("clean", (1.5, 2.0))):
        log = qa.OpLog()
        log.mul("injection", np.array([a]), np.array([b]), np.array([a]) * np.array([b]))
        out.setdefault("B-D-c1", {})[kind] = {"premises": {"P-c": _st(not log.violations, "; ".join(log.violations) or "no violation")}}
    # P-c: FTZ imitation (a zero result of + with a != -b)
    a, b = float.fromhex("0x1.8p-1022"), -float.fromhex("0x1p-1022")
    for kind, r in (("injected", 0.0), ("clean", a + b)):
        log = qa.OpLog()
        log.add("injection", np.array([a]), np.array([b]), np.array([r]))
        out.setdefault("B-D-c2", {})[kind] = {"premises": {"P-c": _st(not log.violations, "; ".join(log.violations) or "no violation")}}
    # P-c: finiteness of P^ (pi_double of r5_cccc_energy on a field that overflows)
    g = en.Grid(1.0, 4, 1.0, 4)
    with np.errstate(over="ignore", invalid="ignore"):
        for kind, w in (("injected", np.full((3, 3), 1e200)), ("clean", np.ones((3, 3)))):
            P = en.pi_double(g, w)
            fin = bool(qa.is_finite(np.array([P]))[0])
            out.setdefault("B-D-c5", {})[kind] = {"premises": {"P-c": _st(fin, f"pi_double finite={fin}")}}
    # P-d: U-3 grid constants with D = 2^-1074
    for kind, D in (("injected", 5e-324), ("clean", 1.0)):
        kc = qa.grid_constants(1.0, 8, 1.0, 8, D, 1.0, 1)
        ok = qa.grid_ok(kc)
        out.setdefault("B-D-d", {})[kind] = {"premises": {"P-d": _st(ok, f"grid_ok={ok}, normal_ok={kc['normal_ok']}")}}
    # P-e: N-7 (previous field 2^-600 x 2^-600 computed as 0, eps_a^C(s - 1) = 0 recorded); predicate of QA-EB analyse_row
    g3 = en.Grid(1.0, 3, 1.0, 3)
    t = 2.0**-600
    for kind, (Xp, Yp) in (("injected", (np.full((2, 1), t), np.full((2, 1), t))), ("clean", (np.zeros((2, 1)), np.full((2, 1), 0.5)))):
        Wp = Xp @ Yp.T
        fp = qa.formation_check(Xp, Yp, Wp)
        eps_p_en = en.eps_a_C(g3, Xp, Yp)
        hist_eps0 = True  # injected record: eps_a_C_prev_up = 0
        applicable = hist_eps0 or eps_p_en == 0
        n7_ok = bool(fp["exact_product_zero"] and fp["computed_field_zero"]
                     and (bool(np.all(qa.is_zero(Yp))) or bool(np.all(qa.is_zero(Xp)))) and eps_p_en == 0) if applicable else True
        out.setdefault("B-D-e", {})[kind] = {"premises": {
            "P-b": _st(fp["holds"], f"formation_check holds={fp['holds']}"),
            "P-e": _st(n7_ok, f"N-7: exact product zero={fp['exact_product_zero']}, computed zero={fp['computed_field_zero']}, "
                              f"EN eps_prev == 0: {eps_p_en == 0}")}}
    # P-f: B_E not finite in binary64
    gq = qa.GridQ(1.0, 4, 1.0, 4)
    kc = qa.grid_constants(1.0, 4, 1.0, 4, 1.0, 1.0, 1)
    for kind, val in (("injected", 2.0**500), ("clean", 1.0)):
        Xs, Ys = np.full((3, 1), val), np.full((3, 1), val)
        Xp, Yp = np.zeros((3, 1)), np.zeros((3, 1))
        Wp, Ws = Xp @ Yp.T, Xs @ Ys.T
        be = qa.b_e(gq, kc, 1, qa.formation_check(Xp, Yp, Wp), qa.formation_check(Xs, Ys, Ws), Wp - Ws)
        fin = be["BE"] is not None and math.isfinite(qa.directed(be["BE"])[1])
        out.setdefault("B-D-f1", {})[kind] = {"premises": {"P-f": _st(fin, f"B_E up finite={fin}")}}
    # P-f: root domain (negative radicand -> exception -> not checkable)
    for kind, x in (("injected", Fraction(-1)), ("clean", Fraction(4))):
        try:
            qa.sqrt_bounds(x)
            st = _st(True, "sqrt_bounds evaluated")
        except ValueError as e:
            st = _st(False, f"sqrt_bounds: ValueError: {e}", checkable=False)
        out.setdefault("B-D-f2", {})[kind] = {"premises": {"P-f": st}}
    return out


_MODES = {"round_up": (0x200, 0x300), "ftz_daz": (0x01000000, 0x03000000)}  # (_RC_UP, _MCW_RC), (_DN_FLUSH, _MCW_DN)


def _mode_probe_child(mode: str) -> dict:
    """QA-EB probe_mode in this (child) process after setting the SSE control word of the main thread."""
    qa = _load(QAEB_TOOL, "_qaeb_for_mode")
    if mode != "none":
        import ctypes

        f = ctypes.CDLL("ucrtbase")._controlfp
        f.restype, f.argtypes = ctypes.c_uint, [ctypes.c_uint, ctypes.c_uint]
        f(*_MODES[mode])
    pb = qa.probe_mode()
    return {"mode": mode, "ok": bool(pb["ok"]), "fails": len(pb["fails"]), "controlfp": pb["controlfp"].get("word")}


def detector_cases() -> dict:
    det = _child(["_detectors"])
    for cid, mode in (("B-D-c3", "round_up"), ("B-D-c4", "ftz_daz")):
        if sys.platform != "win32":
            det[cid] = {"not_executed": "the mode injection uses _controlfp (Windows); not executed on this platform"}
            continue
        inj, clean = _child(["_mode_probe", mode]), _child(["_mode_probe", "none"])
        det[cid] = {"injected": {"premises": {"P-c": _st(inj["ok"], f"probe_mode ok={inj['ok']}, fails={inj['fails']}, "
                                                                    f"controlfp={inj['controlfp']}")}},
                    "clean": {"premises": {"P-c": _st(clean["ok"], f"probe_mode ok={clean['ok']}, controlfp={clean['controlfp']}")}},
                    "evidence": {"injected": inj, "clean": clean}}
    return det


# ---------------------------------------------------------------------------------------------------------------------------------
# 7. resource probes (12.5; PL 6.3); task C
# ---------------------------------------------------------------------------------------------------------------------------------

PROBES = {  # configuration -> (l_x, n_x, l_y, n_y); N = 1, s = 1 only
    "A:320x1280": (1.0, 320, 4.0, 1280), "A:400x1000": (1.0, 400, 2.5, 1000), "A:400x600": (1.0, 400, 1.5, 600),
    "S:400x1600": (1.0, 400, 4.0, 1600), "S:1600x400": (1.0, 1600, 0.25, 400),
}
SMOKE = {"SMOKE:8x8": (1.0, 8, 1.0, 8)}  # harness test only; not a D1 configuration
REFUSED = {
    "D:11x15": "block D: geometry (l_x, l_y) not fixed (OD-1); largest small N = 1 configuration by cells",
    "B:160x640:N3": "N = 3: PRE_D1_RANK (D-049.5 stop before QR vs QR continued by OP-8 as RESEARCH_ONLY)",
    "B:200x500:N3": "N = 3: PRE_D1_RANK", "C:80x320:N3": "N = 3: PRE_D1_RANK", "C:100x250:N3": "N = 3: PRE_D1_RANK",
}
PL63_RAM_FRACTION = Fraction(1, 2)  # PL 6.3: peak memory <= 50 % of the RAM of the machine
PL63_FACTORIZATION_S = 30 * 60  # PL 6.3: factorization <= 30 min
SAFETY_TIMEOUT_S = 4 * 3600  # operational safety of THIS probe (not a limit of PL 6.3, which sets none for B_E)
PROBE_ENV = {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}  # launcher environment of the child (OD-6
# proposal); set for the child process only, recorded; the actual BLAS thread count is read from the library in the child


def _mem() -> dict:
    """(working set, peak working set) of this process in bytes (Windows); POSIX: peak from ru_maxrss."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class C(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t), ("a", ctypes.c_size_t), ("b", ctypes.c_size_t), ("c", ctypes.c_size_t),
                        ("d", ctypes.c_size_t), ("e", ctypes.c_size_t), ("f", ctypes.c_size_t)]

        c = C()
        c.cb = ctypes.sizeof(c)
        k = ctypes.WinDLL("kernel32")
        k.GetCurrentProcess.restype = wintypes.HANDLE
        ps = ctypes.WinDLL("psapi")
        ps.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(C), wintypes.DWORD]
        if ps.GetProcessMemoryInfo(k.GetCurrentProcess(), ctypes.byref(c), c.cb):
            return {"ws": int(c.WorkingSetSize), "peak": int(c.PeakWorkingSetSize)}
        return {"ws": None, "peak": None}
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    return {"ws": peak, "peak": peak}


def total_ram_bytes() -> int | None:
    if sys.platform == "win32":
        import ctypes

        class S(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        s = S()
        s.dwLength = ctypes.sizeof(s)
        return int(s.ullTotalPhys) if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s)) else None
    try:
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (ValueError, OSError, AttributeError):
        return None


def _watchdog(limit: int, flag: Path, stop: threading.Event) -> None:
    while not stop.is_set():
        m = _mem()
        if m["ws"] is not None and m["ws"] > limit:
            flag.write_text(json.dumps({"aborted": "memory", "working_set_bytes": m["ws"], "limit_bytes": limit}), encoding="utf-8")
            os._exit(97)
        stop.wait(0.2)


def _probe_child(name: str, limit: int, flag: Path) -> dict:
    """One half-step s = 1, N = 1 with the stages of the module docstring (3). Values of the row are discarded."""
    lx, nx, ly, ny = {**PROBES, **SMOKE}[name]
    stop = threading.Event()
    threading.Thread(target=_watchdog, args=(limit, flag, stop), daemon=True).start()
    t0 = time.perf_counter()
    import numpy as np

    from kkvim_reference import r5_cccc_bc as bc
    from kkvim_reference import r5_cccc_halfstep as hs

    tj = _load(OP8_TOOL, "_op8_probe")
    qa = _load(QAEB_TOOL, "_qaeb_probe")
    br = _load(BR_TOOL, "_br_probe")
    out = {"probe": name, "label": PROBE_LABEL, "N": 1, "s": 1, "direction": "Y", "nx": nx, "ny": ny, "lx": repr(lx), "ly": repr(ly),
           "cells": nx * ny, "import_s": round(time.perf_counter() - t0, 2),
           "blas_threads_read_from_library": qa.blas_info().get("threads")}
    # (a) OP-8 tool, first half-step only
    t = time.perf_counter()
    tr = tj.run_trajectory(tj.TrajectoryConfig(lx, nx, ly, ny, 1, s_max=1))
    rec, arr = tr.records[0], tr.arrays[0]
    out["stage_op8"] = {"seconds": round(time.perf_counter() - t, 2), "completed": rec.get("status") == tj.OK,
                        "tool_status_class": "OK" if rec.get("status") == tj.OK else str(rec.get("status", "")).split(":")[0]}
    # (b) QA-EB analyse_row on the same s = 1 input (EQ path, gamma B1, U-1, U-2, U-3, N-7, exact B_E)
    given = hs.start_set(nx, 1)
    step = {"s": 1, "direction": "Y", "given": given, "companion": np.zeros((ny - 1, 1)), "V": None,
            "prev": (np.zeros((nx - 1, 1)), np.zeros((ny - 1, 1))), "given_consistent": True}
    spec = {"lx": lx, "nx": nx, "ly": ly, "ny": ny, "group": "PRELAUNCH_RESOURCE_PROBE", "first": "Y", "N": 1}
    t = time.perf_counter()
    try:
        row = qa.analyse_row(spec, [step], 0, f"PRELAUNCH|{name}|N1|s1", None, nx * ny <= qa.CONTROL_MAX_NODES, None)
        out["stage_qaeb"] = {"seconds": round(time.perf_counter() - t, 2), "completed": "status" in row,
                             "B_E_evaluated": "B_E_up" in row, "control_path_rows": nx * ny <= qa.CONTROL_MAX_NODES}
    except Exception as exc:  # recorded, never hidden
        out["stage_qaeb"] = {"seconds": round(time.perf_counter() - t, 2), "completed": False,
                             "exception": f"{type(exc).__name__}: {exc}"}
    row = None
    # bitwise identity of the B-C solution: OP-8 (a) vs the EQ path used in (b)
    p = hs.HalfStep(lx, nx, ly, ny, "Y", given)
    rec2 = bc.run_halfstep(p, qr=False)
    sol = qa._eq().solve_c3plus(rec2.solved, hs.exact_coefficients(rec2.solved))["_sol"]
    V_eq = None if sol.s is None else sol.s.reshape(1, ny - 1).T
    out["op8_V_equals_EQ_path_V_bitwise"] = V_eq is not None and "V_bc" in arr and bool(np.array_equal(arr["V_bc"], V_eq))
    # (c) BR certified reference of the same half-step
    t = time.perf_counter()
    try:
        b = br.analyse_point(p, False, None, (nx - 1) * (ny - 1) <= br.RH_LIMIT)
        log = str(b.get("precision_log", ""))
        out["stage_br"] = {"seconds": round(time.perf_counter() - t, 2), "lambda_low_certified": b.get("lambda_low") is not None,
                           "factorization_attempts": log.count("p="), "factorization_failed_at": log.count("factorization failed"),
                           "precision_used": b.get("precision"), "refine_steps": b.get("refine_steps"),
                           "refine_converged": b.get("refine_converged")}
    except Exception as exc:  # recorded, never hidden
        out["stage_br"] = {"seconds": round(time.perf_counter() - t, 2), "exception": f"{type(exc).__name__}: {exc}"}
    stop.set()
    m = _mem()
    out["peak_working_set_bytes"] = m["peak"]
    out["total_seconds"] = round(time.perf_counter() - t0, 2)
    return out


PROBE_CODE = ("_mem", "total_ram_bytes", "_watchdog", "_probe_child", "evaluate_probe", "run_probes")


def probe_code_sha() -> str:
    """sha256 of the source of the probe functions and constants: shows that later edits elsewhere left the probe unchanged."""
    import inspect

    mod = sys.modules[__name__]
    parts = [inspect.getsource(getattr(mod, n)) for n in PROBE_CODE]
    parts.append(repr((PROBES, SMOKE, REFUSED, PL63_RAM_FRACTION, PL63_FACTORIZATION_S, SAFETY_TIMEOUT_S, PROBE_ENV, PROBE_LABEL)))
    return sha256_bytes("\n".join(parts).encode("utf-8"))


def evaluate_probe(res: dict, ram: int | None) -> dict:
    """PL 6.3 conditions on one probe under the proposed definitions OD-7 (not owner decisions)."""
    ev = {}
    lim = None if ram is None else ram * PL63_RAM_FRACTION
    pk = res.get("peak_working_set_bytes")
    ev["peak_le_50pct_ram"] = None if (lim is None or pk is None) else pk <= lim
    st = [res.get("stage_op8", {}).get("seconds"), res.get("stage_br", {}).get("seconds")]
    ev["factorization_stages_le_30min"] = None if None in st else all(x <= PL63_FACTORIZATION_S for x in st)
    br = res.get("stage_br", {})
    ev["refinement_converges"] = None if "refine_converged" not in br else bool(br["refine_converged"] and br["lambda_low_certified"])
    ev["B_E_and_premises_included"] = bool(res.get("stage_qaeb", {}).get("completed") and res["stage_qaeb"].get("B_E_evaluated"))
    ev["all_within"] = all(v is True for v in ev.values())
    return ev


def run_probes(names: list[str], out_dir: Path) -> dict:
    # provenance taken BEFORE the probes: the file may be edited while they run (inspect reads the source from disk)
    blob, code_sha = git_blob_id(ROOT / SELF), probe_code_sha()
    ram = total_ram_bytes()
    limit = int(ram * PL63_RAM_FRACTION) if ram else 0
    results = []
    for name in names:
        if name in REFUSED:
            results.append({"probe": name, "result": "NOT_EXECUTED", "reason": REFUSED[name]})
            continue
        if name not in PROBES and name not in SMOKE:
            raise ValueError(f"unknown probe {name}")
        if not limit:
            results.append({"probe": name, "result": "UNRESOLVED", "reason": "RAM of the machine not measurable"})
            continue
        flag = out_dir / f"abort_{name.replace(':', '_')}.json"
        flag.unlink(missing_ok=True)
        t0 = time.perf_counter()
        try:
            p = subprocess.run([sys.executable, str(ROOT / SELF), "_probe_child", name, str(limit), str(flag)], cwd=ROOT,
                               capture_output=True, text=True, timeout=SAFETY_TIMEOUT_S, env={**os.environ, **PROBE_ENV})
        except subprocess.TimeoutExpired:
            results.append({"probe": name, "result": "UNRESOLVED", "reason": f"safety stop after {SAFETY_TIMEOUT_S} s (not a PL 6.3 "
                                                                            "limit)", "elapsed_s": round(time.perf_counter() - t0, 1)})
            continue
        if flag.exists():
            ab = json.loads(flag.read_text(encoding="utf-8"))
            results.append({"probe": name, "result": "UNRESOLVED_RESOURCES", "reason": "peak memory above 50 % of RAM: probe "
                            "stopped", **ab, "elapsed_s": round(time.perf_counter() - t0, 1)})
            continue
        if p.returncode != 0:
            results.append({"probe": name, "result": "UNRESOLVED", "reason": f"child failed ({p.returncode})",
                            "stderr_tail": p.stderr[-1500:]})
            continue
        res = json.loads(p.stdout.strip().splitlines()[-1])
        ev = evaluate_probe(res, ram)
        res["pl63_under_OD7"] = ev
        res["result"] = "WITHIN_PL63_UNDER_PROPOSED_DEFINITIONS" if ev["all_within"] else "UNRESOLVED_RESOURCES"
        results.append(res)
    return {"document": "R5-V Q4 D1 PRELAUNCH resource probes (protocol 8, 12.5; PL 6.3)", "label": PROBE_LABEL,
            "data_status": NOT_D1, "total_ram_bytes": ram, "limit_bytes_50pct": limit, "environment": platform_fingerprint(),
            "probe_tool_blob": blob, "probe_code_sha256": code_sha, "definitions": "OD-7 (proposal)", "material_load": "D = q = 1 (OD-5 proposal)",
            "child_environment_set_by_the_launcher": PROBE_ENV,
            "results": results}


# ---------------------------------------------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("manifest")
    m.add_argument("--out", type=Path, default=OUT)
    c = sub.add_parser("controls")
    c.add_argument("--out", type=Path, default=OUT / "control_ab_results.json")
    p = sub.add_parser("probe")
    p.add_argument("names", nargs="+")
    p.add_argument("--out", type=Path, default=ROOT / "scratch" / "r5_cccc_q4_d1_prelaunch")
    pc = sub.add_parser("_probe_child")
    pc.add_argument("name")
    pc.add_argument("limit", type=int)
    pc.add_argument("flag", type=Path)
    sub.add_parser("_detectors")
    mp = sub.add_parser("_mode_probe")
    mp.add_argument("mode", choices=("none", *_MODES))
    a = ap.parse_args(argv)
    if a.cmd == "manifest":
        man = write_manifest(a.out)
        print(json.dumps({"rows": len(man["mandatory_rows"]), "missing": man["sha_inventory"]["missing"],
                          "second_path": man["second_path"]["counts"], "d1_start": D1_START}))
    elif a.cmd == "controls":
        r = run_controls()
        a.out.write_text(json.dumps(r, ensure_ascii=False, indent=1, default=str) + "\n", encoding="utf-8", newline="\n")
        print(json.dumps(r["summary"]))
    elif a.cmd == "probe":
        a.out.mkdir(parents=True, exist_ok=True)
        r = run_probes(a.names, a.out)
        (a.out / "resource_probes.json").write_text(json.dumps(r, indent=1) + "\n", encoding="utf-8", newline="\n")
        print(json.dumps([{k: x.get(k) for k in ("probe", "result")} for x in r["results"]]))
    elif a.cmd == "_probe_child":
        print(json.dumps(_probe_child(a.name, a.limit, a.flag)))
    elif a.cmd == "_detectors":
        print(json.dumps(_detectors_child(), default=str))
    elif a.cmd == "_mode_probe":
        print(json.dumps(_mode_probe_child(a.mode)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
