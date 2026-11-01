"""R4 owner-readiness audit: integrity, recomputation and targeted independent controls (Auditor; G2, G3.3, G4).

Object: integration branch ``integration/r4-qa-n46-roundoff`` @ a5d89c3 on ``main`` @ a2307cd (package A: N = 4-6 study and
its QA; package B: N = 2-3 round-off / rank qualification and its review). The tool changes nothing in the solver, the
specifications or the packages; it reads the committed data and git objects and recomputes.

Subcommands (from the repository root, environment of uv.lock):

- ``integrity``  - git history, patch-ids, file set, byte equality with the sources, protected paths, meta.json vs code;
- ``n46``        - counts, statuses, guards and the 1/100 rule recomputed from the CSV of package A; tables.md regenerated;
- ``roundoff``   - counts, maxima, rank/energy certificate consistency from the CSV of package B; outward rounding of
                   table 3.1 / section 15 of the QA report checked against cells_fft_M2.csv;
- ``controls``   - (needs python-flint and mpmath) targeted controls: own nodal Galerkin half-step in ball arithmetic
                   (five-point Delta_h applied to every basis field, no A/B/C, no DST, no modal split) against the solver
                   half-step; exact s = 1 rank by two routes; fresh reruns against the committed rows; delta_f(1) of
                   package B by a closed form in mpmath;
- ``config``     - fingerprint of a small sample for the current interpreter and BLAS thread setting;
- ``config-compare`` - compares the fingerprints by the criteria CFG_* fixed below before the runs;
- ``collect``    - meta.json with the hashes of the outputs.

Criteria of ``config-compare`` (fixed before any fingerprint was produced; AGENTS.md section 4, D-025):
- classifications identical: cell status, half-steps done, first stabilization s, per-step energy status, guard decisions,
  stabilization flags (CFG_CLASSIFICATION);
- w(centre) and Pi_h: relative difference <= 1e-10 (CFG_REL; the reproducibility criterion of the Builder report, 7.8);
- rho_N: absolute difference <= 1e-12 (CFG_RHO_ABS). Basis: by Weyl |rho(W1) - rho(W2)| <= 2 |W1 - W2|_2 / sigma_1, and the
  certified field error of the recorded N = 2-3 runs is <= 1.39e-13 (package B), the true implementation error of N = 4-6 on
  small grids <= 3.3e-14 (package A QA); 1e-12 is about 7 times the larger of them. Bitwise equality is recorded as
  information only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from decimal import ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_EVEN, Decimal
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "results" / "r4_owner_readiness"
RES_A = ROOT / "docs" / "results" / "r4_n4_n6"
RES_AQ = ROOT / "docs" / "results" / "r4_n4_n6_qa"
RES_B = ROOT / "docs" / "results" / "r4_roundoff_rank_qualification"

BASE = "a2307cd8030db812ff1769bc775c0e5485b8f9e2"
HEAD = "a5d89c35ff08b85270d08685d0624b8930298bd9"
PKG_A = ["d1b956336db0efe0d517723fc36437c176d895b8", "69e26bb79f25c91e894b8f4b72da579d38cd9c9d",
         "6c7a80de00140fbb714e6cb0577bed232f3bc644", "58c35d747af64a9b62225a871c5b535440c8867a",
         "ff9b5d1933798fef54ae815641d9d0d7d7649320"]
PKG_B = [("2e53a485ac4d59075fb4bce6ca2f8f47e60750fe", "df376524ffa638bca9de2cd9f31cff5e5c1cb81d"),
         ("64685456979fdffbaa370ca81e4e0142cead62d1", "a2649450805620caad284789d79f8ab56a22ab59"),
         ("28046d628776ca198fb6631c4542d32d95c52dc8", "bc6ae13ac736f535b1705e681fd7d091eb41793c"),
         ("d4563dd986a3ca6c131d836f0ed5ba5acc63ff70", "a4add731eb645e4cde5f10f32f3fb98a7b9dda22"),
         ("d0a95c75e080cb3cf54f773c45163416ed9da374", "0cfe5f9b58638afcaa5a709941c6c22ca478dd77"),
         ("92e9fb9dc2d8f38752ef5c42746b8133fbe9d781", "4b7094a352f6497034e9aa42d8af3a87703e467f")]
QA_A_HEAD, REVIEW_B = PKG_A[-1], PKG_B[-1][1]
RUN_A, TOOL_B = PKG_A[1], PKG_B[1][1]  # commit of the N = 4-6 computation; round-off data tool
INTEGRATION_REPORT = "docs/R4_QA_N46_ROUNDOFF_INTEGRATION.report.md"
EDITED = {"docs/R4_N4_N6_STUDY.report.md", "docs/R4_ROUNDOFF_RANK_QUALIFICATION.qa-report.md"}
PROTECTED = ["src/kkvim", "src/kkvim_reference", "docs/MATH_SPEC.md", "docs/R4_NUMERICAL_SPEC.md", "docs/DECISIONS.md",
             "docs/STATUS.md", "docs/OPEN_QUESTIONS.md", "docs/ROADMAP.md", "docs/PROJECT_GOALS.md", "docs/SOURCE_MAP.md",
             "docs/R4_SOURCE_NOTES.md", "docs/R4_Q022_ERROR_TRANSFER.md", "AGENTS.md", "CLAUDE.md", "pyproject.toml",
             "uv.lock", ".github", ".githooks", ".gitignore", ".gitattributes", "references"]
FORBIDDEN_EXT = (".pdf", ".djvu", ".doc", ".docx", ".zip", ".7z", ".rar", ".png", ".jpg", ".jpeg", ".tif", ".tiff",
                 ".npz", ".npy", ".pkl")
SECRET_RE = re.compile(r"(ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY"
                       r"|xox[baprs]-[A-Za-z0-9-]{10,}|(?i:password|passwd|secret|token)\s*[:=]\s*['\"][^'\"]{6,})")

U = 2.0**-53
K_RESEARCH = 1_000_000
RULE = 0.01
CFG_REL = 1e-10
CFG_RHO_ABS = 1e-12
CFG_CLASSIFICATION = ("status", "s_reached", "first_stabilization_s", "energy_status", "guard_rank_ok", "guard_set_ok",
                      "guard_applicable", "stabilization_ok", "stop_reason")


# ----------------------------------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------------------------------

def git(*args, binary=False):
    r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, check=True)
    return r.stdout if binary else r.stdout.decode("utf-8").replace("\r\n", "\n")


def blob(rev, path):
    return git("show", f"{rev}:{path}", binary=True)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def patch_id(rev):
    show = subprocess.run(["git", "show", rev], cwd=ROOT, capture_output=True, check=True).stdout
    pid = subprocess.run(["git", "patch-id", "--stable"], cwd=ROOT, input=show, capture_output=True, check=True).stdout
    return pid.decode().split()[0]


def rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fl(v):
    if v is None or v == "" or v.lower() == "nan":
        return math.nan
    return float(v)


def write_json(name, data, compact=False):
    OUT.mkdir(parents=True, exist_ok=True)
    text = (json.dumps(data, separators=(",", ":"), ensure_ascii=False, default=str) if compact
            else json.dumps(data, indent=1, ensure_ascii=False, default=str))
    (OUT / name).write_text(text + "\n", encoding="utf-8", newline="\n")


def write_csv(name, data):
    OUT.mkdir(parents=True, exist_ok=True)
    keys = []
    for r in data:
        keys += [k for k in r if k not in keys]
    with open(OUT / name, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
        w.writeheader()
        for r in data:
            w.writerow({k: (f"{v:.9e}" if isinstance(v, float) else ("" if v is None else v)) for k, v in r.items()})


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ----------------------------------------------------------------------------------------------------------------------
# Part I - integrity
# ----------------------------------------------------------------------------------------------------------------------

def changed_lines(rev_a, rev_b, path):
    """Line numbers (in rev_b) of added/changed lines and the count of removed lines of path."""
    diff = git("diff", "-U0", rev_a, rev_b, "--", path)
    new_lines, removed = [], 0
    for line in diff.splitlines():
        m = re.match(r"@@ -\d+(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line)
        if m:
            removed += int(m.group(1)) if m.group(1) is not None else 1
            start, count = int(m.group(2)), int(m.group(3)) if m.group(3) is not None else 1
            new_lines += list(range(start, start + count))
    return new_lines, removed


def function_range(source, name):
    lines = source.splitlines()
    start = next(i for i, l in enumerate(lines, 1) if re.match(rf"def {name}\(", l))
    end = next((i for i, l in enumerate(lines[start:], start + 1) if re.match(r"(def |class |[A-Za-z_])", l)), len(lines) + 1)
    return start, end - 1


def integrity():
    res = {"base": BASE, "head": HEAD}
    chain = [l.split() for l in git("rev-list", "--parents", "--reverse", f"{BASE}..{HEAD}").splitlines() if l]
    expected = PKG_A + [n for n, _ in PKG_B] + [HEAD]
    res["commits"] = len(chain)
    res["linear_single_parents"] = all(len(c) == 2 for c in chain) and chain[0][1] == BASE and all(
        chain[i][1] == chain[i - 1][0] for i in range(1, len(chain)))
    res["order_as_reported"] = [c[0] for c in chain] == expected
    res["base_is_origin_main"] = git("rev-parse", "origin/main").strip() == BASE
    # package A: original SHAs; parent chain of the sources
    res["A_qa_head_is_ancestor"] = subprocess.run(["git", "merge-base", "--is-ancestor", QA_A_HEAD, HEAD], cwd=ROOT).returncode == 0
    # package B: source chain order, patch-ids, -x trailers
    src_chain = [l.split() for l in git("rev-list", "--parents", "--reverse", f"{BASE}..{REVIEW_B}").splitlines() if l]
    res["B_source_chain_linear_from_base"] = [c[0] for c in src_chain] == [s for _, s in PKG_B] and src_chain[0][1] == BASE
    picks = []
    for new, src in PKG_B:
        body = git("log", "-1", "--format=%B", new)
        picks.append({"new": new[:7], "source": src[:7], "patch_id_equal": patch_id(new) == patch_id(src),
                      "x_trailer": f"cherry picked from commit {src}" in body,
                      "author_kept": git("log", "-1", "--format=%an|%ae|%ad", new) == git("log", "-1", "--format=%an|%ae|%ad", src)})
    res["B_cherry_picks"] = picks
    # files
    status = [l.split("\t") for l in git("diff", "--name-status", BASE, HEAD).splitlines() if l]
    files = sorted(p for _, p in status)
    res["files"] = len(files)
    res["all_added"] = all(s == "A" for s, _ in status)
    files_a = set(git("diff", "--name-only", BASE, QA_A_HEAD).split())
    files_b = set(git("diff", "--name-only", BASE, REVIEW_B).split())
    res["file_set_is_A_plus_B_plus_report"] = set(files) == files_a | files_b | {INTEGRATION_REPORT}
    res["A_files"], res["B_files"] = len(files_a), len(files_b)
    diffs = []
    for p in files:
        if p == INTEGRATION_REPORT:
            continue
        src = QA_A_HEAD if p in files_a else REVIEW_B
        if git("rev-parse", f"{HEAD}:{p}").strip() != git("rev-parse", f"{src}:{p}").strip():
            diffs.append(p)
    res["files_differing_from_source"] = diffs
    res["only_the_two_reports_edited"] = set(diffs) == EDITED
    res["edited_numstat"] = {p: git("diff", "--numstat", QA_A_HEAD if p in files_a else REVIEW_B, HEAD, "--", p).split()[:2]
                             for p in diffs}
    res["final_commit_files"] = git("diff", "--name-only", PKG_B[-1][0], HEAD).split()
    res["protected_paths_unchanged"] = {p: git("diff", "--name-only", BASE, HEAD, "--", p).strip() == "" for p in PROTECTED}
    res["no_forbidden_extensions"] = not [p for p in files if p.lower().endswith(FORBIDDEN_EXT)]
    res["no_scratch"] = not [p for p in files if p.startswith("scratch/") or "/scratch/" in p]
    patch_text = git("diff", BASE, HEAD)
    res["secret_pattern_hits"] = sorted({m.group(0)[:24] for m in SECRET_RE.finditer(patch_text)})
    res["binary_files"] = [l.split("\t")[2] for l in git("diff", "--numstat", BASE, HEAD).splitlines() if l.startswith("-\t-\t")]
    # meta.json against the code that produced the data
    meta_a = json.loads(blob(HEAD, "docs/results/r4_n4_n6/meta.json"))
    lines, removed = changed_lines(RUN_A, HEAD, "scripts/r4_n4_n6_study.py")
    lo, hi = function_range(blob(HEAD, "scripts/r4_n4_n6_study.py").decode(), "make_tables")
    res["A_meta"] = {"git_head_at_run": meta_a["git_head_at_run"], "run_commit_is_69e26bb": meta_a["git_head_at_run"] == RUN_A,
                     "script_changed_lines_after_run": lines, "removed_lines": removed,
                     "changes_only_in_make_tables": bool(lines) and all(lo <= n <= hi for n in lines),
                     "N_SOLVE_unchanged": meta_a["N_SOLVE_before"] == meta_a["N_SOLVE_after"] == [2, 3]}
    st = load_script("r4_n4_n6_study")
    import kkvim.coupled_run as cr
    plan = meta_a["plan"]
    res["A_meta"]["plan_equals_script"] = (
        plan["N_STUDY"] == list(st.N_STUDY) and plan["N_CONTROL"] == list(st.N_CONTROL) and plan["r"] == list(st.R_TEN)
        and plan["FAMILY_I"] == list(st.FAMILY_I) and plan["FAMILY_II"] == {k: list(v) for k, v in st.FAMILY_II.items()}
        and plan["S_RUN"] == st.S_RUN and plan["S_LIST"] == list(st.S_LIST) and plan["M_CONT"] == st.M_CONT
        and plan["TRANSFORM"] == st.TRANSFORM and plan["RULE"] == st.RULE)
    cc = meta_a["controller_constants"]
    res["A_meta"]["controller_constants_equal_main"] = cc == {"K_RESEARCH": cr.K_RESEARCH, "ETA_MAX": cr.ETA_MAX,
                                                             "TOL_INF": cr.TOL_INF, "TOL_E": cr.TOL_E, "S_MIN": cr.S_MIN,
                                                             "S_MAX": cr.S_MAX}
    qa_sha = sha256(blob(HEAD, "scripts/r4_n4_n6_study_qa.py"))
    res["AQ_meta"] = {}
    for f in ("meta_3.13.15.json", "meta_3.14.8.json"):
        m = json.loads(blob(HEAD, f"docs/results/r4_n4_n6_qa/{f}"))
        res["AQ_meta"][f] = {"git_head": m["git_head"][:7], "script_sha256_equals_committed_tool": m["script_sha256"] == qa_sha,
                             "object": m.get("object")}
    meta_b = json.loads(blob(HEAD, "docs/results/r4_roundoff_rank_qualification/meta.json"))
    lines_b, removed_b = changed_lines(TOOL_B, HEAD, "scripts/r4_roundoff_rank_qualification.py")
    res["B_meta"] = {"data_tool_commit_is_a264945": meta_b["data_tool_commit"] == TOOL_B,
                     "data_tool_sha256_equals_blob": meta_b["data_tool_sha256"] == sha256(blob(TOOL_B, "scripts/r4_roundoff_rank_qualification.py")),
                     "script_changed_lines_after_data": lines_b, "removed_lines": removed_b,
                     "changed_lines_text": [blob(HEAD, "scripts/r4_roundoff_rank_qualification.py").decode().splitlines()[n - 1].strip()
                                            for n in lines_b],
                     "plan_commit_is_df37652": meta_b["plan_commit"] == PKG_B[0][1],
                     "blas_threads_bitwise": meta_b["blas_threads_bitwise"]["equal"]}
    res["B_meta"]["only_mode_one_changed"] = lines_b and all(
        "campaign_cell(" in t or "aggregate(" in t for t in res["B_meta"]["changed_lines_text"])
    checks = [res["linear_single_parents"], res["order_as_reported"], res["base_is_origin_main"], res["A_qa_head_is_ancestor"],
              res["B_source_chain_linear_from_base"], all(p["patch_id_equal"] and p["x_trailer"] for p in picks),
              res["files"] == 49, res["all_added"], res["file_set_is_A_plus_B_plus_report"], res["only_the_two_reports_edited"],
              all(res["protected_paths_unchanged"].values()), res["no_forbidden_extensions"], res["no_scratch"],
              not res["secret_pattern_hits"], not res["binary_files"], res["A_meta"]["run_commit_is_69e26bb"],
              res["A_meta"]["changes_only_in_make_tables"], res["A_meta"]["plan_equals_script"],
              res["A_meta"]["controller_constants_equal_main"],
              all(v["script_sha256_equals_committed_tool"] for v in res["AQ_meta"].values()),
              res["B_meta"]["data_tool_commit_is_a264945"], res["B_meta"]["data_tool_sha256_equals_blob"],
              res["B_meta"]["only_mode_one_changed"], res["B_meta"]["plan_commit_is_df37652"]]
    res["INTEGRITY"] = "INTEGRITY_PASS" if all(checks) else "INTEGRITY_DEFECT"
    res["checks_passed"] = f"{sum(map(bool, checks))}/{len(checks)}"
    write_json("integrity.json", res)
    print(res["INTEGRITY"], res["checks_passed"])
    return res


# ----------------------------------------------------------------------------------------------------------------------
# Part III - package A recomputation from the CSV
# ----------------------------------------------------------------------------------------------------------------------

def rule(budget, measured):
    if not (math.isfinite(budget) and math.isfinite(measured)) or measured == 0.0:
        return "UNRESOLVED"
    return "RESOLVED" if budget <= RULE * abs(measured) else "UNRESOLVED"


def n46():
    cells, steps, comp = rows(RES_A / "cells.csv"), rows(RES_A / "steps.csv"), rows(RES_A / "compare.csv")
    cont = rows(RES_A / "continuous.csv")
    res = {}
    study = [c for c in cells if int(c["N"]) >= 4]
    ctrl = [c for c in cells if int(c["N"]) <= 3]
    res["cells"], res["study_cells"], res["control_cells"] = len(cells), len(study), len(ctrl)
    res["steps_rows"], res["sum_s_reached"] = len(steps), sum(int(c["s_reached"]) for c in cells)
    res["N_study"] = sorted({int(c["N"]) for c in study})
    res["N_control"] = sorted({int(c["N"]) for c in ctrl})
    res["r_values"] = sorted({c["r_label"] for c in study}, key=lambda s: float(Fraction(s)))
    res["grids_I"] = sorted({int(c["nx"]) for c in study if c["family"] == "I"})
    res["grids_II"] = sorted({f'{c["nx"]}x{c["ny"]}' for c in study if c["family"] == "II"})
    stops = [c for c in cells if c["status"] != "RESEARCH_LIMIT_REACHED"]
    res["non_limit_cells"] = len(stops)
    res["stops"] = sorted({(c["N"], c["r_label"], c["status"], c["s_reached"]) for c in stops})
    res["stops_14_N6_r15_r13_s1"] = (len(stops) == 14 and all(c["N"] == "6" and c["r_label"] in ("1/5", "1/3") and
                                     c["status"] == "PROVISIONAL_RANK_GUARD_STOP" and c["s_reached"] == "1" for c in stops))
    stop_rows = [s for s in steps if s["stop_reason"] == "PROVISIONAL_RANK_GUARD_STOP"]
    res["stop_rows_before_qr_rank_guard"] = len(stop_rows) == 14 and all(
        s["qr_done"] == "False" and s["guard_rank_ok"] == "False" and s["s"] == "1" for s in stop_rows)
    res["stop_rows_set_guard_also_fails"] = sorted({(s["r_label"], s["guard_set_ok"]) for s in stop_rows})
    # guard decisions recomputed from the logged diagnostics (D-048.5 applied without change)
    mism, judged, close = 0, 0, 0
    for s in steps:
        if s["guard_rank_ok"] == "":
            continue
        judged += 1
        rho, kap, sig = fl(s["rho_N"]), fl(s["kappa_rep"]), fl(s["set_sigma_min"])
        thr = K_RESEARCH * U * kap
        ok = (str(thr < 1.0) == s["guard_applicable"] and str(rho > thr) == s["guard_rank_ok"]
              and str(sig > K_RESEARCH * U) == s["guard_set_ok"])
        mism += not ok
        close += abs(rho / thr - 1) < 1e-5
    res["guards_judged"], res["guard_mismatches"], res["guard_within_1e-5_of_threshold"] = judged, mism, close
    passed = [c for c in study if c["status"] == "RESEARCH_LIMIT_REACHED"]
    margins = {}
    for N in (4, 5, 6):
        for lab in res["r_values"]:
            sub = [c for c in passed if int(c["N"]) == N and c["r_label"] == lab]
            if sub:
                best = min(sub, key=lambda c: fl(c["min_rho_over_threshold"]))
                margins[f"{N}:{lab}"] = (round(fl(best["min_rho_over_threshold"]), 4), f'{best["family"]}:{best["nx"]}x{best["ny"]}')
    res["min_guard_margin_passed"] = dict(sorted(margins.items(), key=lambda t: t[1][0])[:6])
    rows_min = {}
    for key in ("5:1/5", "6:5"):
        N, lab = key.split(":")
        sub = [s for s in steps if s["N"] == N and s["r_label"] == lab and s["rho_over_threshold"] != ""]
        m = min(sub, key=lambda s: fl(s["rho_over_threshold"]))
        rows_min[key] = {"value": fl(m["rho_over_threshold"]), "s": int(m["s"]), "grid": f'{m["family"]}:{m["nx"]}x{m["ny"]}'}
    res["min_margin_rows"] = rows_min
    # statuses and marks
    forbidden = ("CONVERGED", "PROVISIONAL_FIELD_STABILIZED", "MAX_HALF_STEPS", "S-2", "STOPPED_NUMERICALLY_INDISTINGUISHABLE")
    res["forbidden_status_hits_N_ge_4"] = sum(any(f in (c["status"] + c["reason"]) for f in forbidden) for c in study) + sum(
        any(f in s["stop_reason"] for f in forbidden) for s in steps if int(s["N"]) >= 4)
    res["mark_every_row"] = all(c["mark"] == "PROVISIONAL / NOT_QUALIFIED" for c in cells) and all(
        s["mark"] == "PROVISIONAL / NOT_QUALIFIED" for s in steps)
    res["s2_mandatory_flag_empty_for_N_ge_4"] = all(c["s2_mandatory_D048"] == "" for c in study)
    res["s2_mandatory_flag_set_for_controls"] = sorted({c["s2_mandatory_D048"] for c in ctrl})
    res["errors"] = sum(c["status"].startswith("ERROR") for c in cells)
    res["refinements"] = sum(int(c["refinements"]) for c in cells)
    res["max_eta_N"] = max(fl(c["max_eta_N"]) for c in cells)
    res["energy_increases_beyond_tau"] = sum(int(c["energy_increases_beyond_tau"]) for c in cells)
    res["energy_increases_N_ge_4"] = sum(int(c["energy_increases"]) for c in study)
    res["max_identity_defect_over_rounding_N_ge_4"] = max(
        (fl(c["max_identity_defect_over_rounding"]) for c in study if not math.isnan(fl(c["max_identity_defect_over_rounding"]))))
    res["identity_defect_nan_cells"] = sum(math.isnan(fl(c["max_identity_defect_over_rounding"])) for c in cells)
    res["min_tolE_over_10floor"] = min((fl(c["tolE_over_10floor"]), f'{c["N"]}:{c["family"]}:{c["nx"]}x{c["ny"]}') for c in cells)
    s2 = [c for c in study if c["s2_status"]]
    res["s2_hold"] = {N: sum(c["s2_status"].startswith("E1_E2_HOLD") for c in s2 if c["N"] == str(N)) for N in (4, 5, 6)}
    res["s2_unresolved"] = sorted((c["N"], c["r_label"], c["nx"]) for c in s2 if c["s2_status"] == "UNRESOLVED")
    # compare.csv: counts and the 1/100 rule recomputed from the stored budgets
    cs = [r for r in comp if int(r["N"]) >= 4]
    res["compare_rows"], res["compare_rows_N_ge_4"] = len(comp), len(cs)
    res["compare_rows_expected"] = 184 * 7 + 14 * 1 + 12 * 7
    res["V5_rows_all_N"] = sum(r["E_impl_inf"] != "" for r in comp)  # includes the 84 rows of the N = 2-3 controls
    res["V5_rows_by_N"] = {N: sum(r["E_impl_inf"] != "" for r in comp if r["N"] == str(N)) for N in (4, 5, 6)}
    res["max_E_impl_inf_by_N"] = {N: max(fl(r["E_impl_inf"]) for r in comp if r["N"] == str(N) and r["E_impl_inf"] != "")
                                  for N in (4, 5, 6)}
    recomputed = {k: 0 for k in ("status_E_total", "status_meas_E_disc", "status_cont_E_disc", "status_meas_E_method",
                                 "status_cont_E_method", "status_d_total_c_navier", "status_E_disc_s1_navier_m", "status_e_rel_h")}
    border = 0
    for r in comp:
        s = int(r["s"])
        W0max, w0c = fl(r["omega0_max"]), fl(r["omega0_centre"])
        levy = fl(r["levy_budget_rel"])
        tail = fl(r["cont_tail_s1_rel"]) if r["cont_tail_s1_rel"] else 0.0
        rnd = fl(r["cont_round_rel"])
        e_disc, e_meth, e_tot = fl(r["E_disc_inf"]), fl(r["E_method_inf"]), fl(r["E_total_inf"])
        got = {"status_E_total": rule(levy, e_tot), "status_meas_E_disc": rule(rnd + tail, e_disc),
               "status_meas_E_method": rule(levy + rnd + tail, e_meth)}
        got["status_cont_E_disc"] = got["status_meas_E_disc"] if s == 1 else (
            "PROVISIONAL" if got["status_meas_E_disc"] == "RESOLVED" and rule(fl(r["cont_diag_rel"]), e_disc) == "RESOLVED"
            else "UNRESOLVED")
        got["status_cont_E_method"] = got["status_meas_E_method"] if s == 1 else (
            "PROVISIONAL" if got["status_meas_E_method"] == "RESOLVED" and rule(fl(r["cont_diag_rel"]), e_meth) == "RESOLVED"
            else "UNRESOLVED")
        got["status_d_total_c_navier"] = rule(fl(r["navier_budget_rel"]) * abs(w0c),
                                              (fl(r["d_total_c"]) - fl(r["navier_centre_minus_levy_rel"])) * w0c)
        if r["status_E_disc_s1_navier_m"]:
            got["status_E_disc_s1_navier_m"] = rule(fl(r["navier_m_budget_rel"]), fl(r["E_disc_s1_navier_m_inf"]))
        got["status_e_rel_h"] = "RESOLVED" if fl(r["e_rel_h_budget_rel"]) <= RULE else "UNRESOLVED"
        for k, v in got.items():
            if v != r[k]:
                recomputed[k] += 1
                b = {"status_E_total": (levy, e_tot)}.get(k)
                if b and abs(b[0] / (RULE * b[1]) - 1) < 1e-5:
                    border += 1
    res["status_rule_mismatches"] = recomputed
    res["status_rule_mismatches_on_printing_border"] = border
    res["E_total_RESOLVED_N_ge_4"] = sum(r["status_E_total"] == "RESOLVED" for r in cs)
    res["navier_UNRESOLVED_N_ge_4"] = sum(r["status_d_total_c_navier"] == "UNRESOLVED" for r in cs)
    res["e_rel_h_UNRESOLVED_N_ge_4"] = sum(r["status_e_rel_h"] == "UNRESOLVED" for r in cs)
    res["T7_s1_navier_m_RESOLVED"] = sum(r["status_E_disc_s1_navier_m"] == "RESOLVED" for r in cs if r["s"] == "1")
    res["T7_s1_rows"] = sum(1 for r in cs if r["s"] == "1")
    res["cont_status_s_ge_2_values"] = sorted({r["status_cont_E_method"] for r in cs if r["s"] != "1"})
    big = max(cs, key=lambda r: fl(r["E_disc_inf"]) / max(fl(r["E_total_inf"]), 1e-300))
    res["largest_discretization_share"] = {k: big[k] for k in ("N", "r_label", "family", "nx", "ny", "s", "E_disc_inf",
                                                               "E_method_inf", "E_total_inf", "status_E_total")}
    # continuous e_rel strictly decreasing in N (observation of the report)
    ce = {(int(r["N"]), r["r_label"], int(r["s"])): fl(r["e_rel"]) for r in cont}
    pairs = viol = 0
    for (N, lab, s), v in ce.items():
        if (N + 1, lab, s) in ce:
            pairs += 1
            viol += not (ce[(N + 1, lab, s)] < v)
    res["e_rel_N_pairs"], res["e_rel_N_not_decreasing"] = pairs, viol
    # NaN handling: tables.md regenerated from the committed CSV by the Builder generator (LF)
    st = load_script("r4_n4_n6_study")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for name in ("cells.csv", "steps.csv", "compare.csv", "continuous.csv", "orders.csv", "control_n23.csv", "meta.json"):
            (tmp / name).write_bytes(blob(HEAD, f"docs/results/r4_n4_n6/{name}"))
        st.make_tables(tmp)
        regen = (tmp / "tables.md").read_bytes().replace(b"\r\n", b"\n")
    res["tables_md_regenerated_equal"] = regen == blob(HEAD, "docs/results/r4_n4_n6/tables.md")
    nan_cols = {}
    for r in steps:
        for k in ("rho_N", "Pi_h", "w_centre"):
            if math.isnan(fl(r[k])):
                nan_cols[k] = nan_cols.get(k, 0) + 1
    res["nan_in_core_step_columns"] = nan_cols
    expected_steps = {(c["N"], c["r_label"], c["family"], c["nx"], c["ny"]): int(c["s_reached"]) for c in cells}
    got_steps = {}
    for s in steps:
        key = (s["N"], s["r_label"], s["family"], s["nx"], s["ny"])
        got_steps.setdefault(key, []).append(int(s["s"]))
    res["steps_contiguous_1_to_s_reached"] = all(sorted(v) == list(range(1, expected_steps[k] + 1)) for k, v in got_steps.items())
    checks = [res["cells"] == 210, res["study_cells"] == 198, res["control_cells"] == 12, res["steps_rows"] == 7854,
              res["sum_s_reached"] == 7854, res["stops_14_N6_r15_r13_s1"], res["stop_rows_before_qr_rank_guard"],
              len(res["r_values"]) == 10, res["N_study"] == [4, 5, 6], res["N_control"] == [2, 3], res["guard_mismatches"] == 0,
              res["forbidden_status_hits_N_ge_4"] == 0, res["mark_every_row"], res["s2_mandatory_flag_empty_for_N_ge_4"],
              res["compare_rows"] == res["compare_rows_expected"], sum(res["V5_rows_by_N"].values()) == 1104,
              all(v == 0 for v in res["status_rule_mismatches"].values()), res["tables_md_regenerated_equal"],
              res["steps_contiguous_1_to_s_reached"], res["energy_increases_beyond_tau"] == 0, res["errors"] == 0]
    res["N46_RECOUNT"] = "PASS" if all(checks) else "CHECK"
    res["checks_passed"] = f"{sum(map(bool, checks))}/{len(checks)}"
    write_json("n46_recount.json", res)
    print(res["N46_RECOUNT"], res["checks_passed"])
    return res


# ----------------------------------------------------------------------------------------------------------------------
# Part IV - package B recomputation from the CSV; outward rounding of table 3.1 / section 15
# ----------------------------------------------------------------------------------------------------------------------

SUP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻", "0123456789-")
TABLE_COLS = [("max_delta_f", "up"), ("max_delta_f_s41_100", "up"), ("max_transfer", "up"), ("max_eps_cert", "up"),
              ("max_c_d_obs", "up"), ("min_rho_hat", "down"), ("min_rho_hat_over_eps_ratio", "down"),
              ("min_energy_margin_s_ge2", "down")]


def parse_printed(text):
    """'2,50·10⁻¹⁵' -> (Decimal value, Decimal ulp of the last printed digit)."""
    t = text.strip()
    m = re.fullmatch(r"(\d+),?(\d*)(?:·10([⁰¹²³⁴⁵⁶⁷⁸⁹⁻]+))?", t)
    if not m:
        raise ValueError(t)
    ip, dp, ex = m.group(1), m.group(2), m.group(3)
    e = int(ex.translate(SUP)) if ex else 0
    mant = Decimal(f"{ip}.{dp}" if dp else ip)
    return mant.scaleb(e), Decimal(1).scaleb(e - len(dp))


def tables_31_15(text):
    """The two copies of table 3.1 in the QA report: (original, section 15)."""
    out, cur = [], None
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("| Сетки | max δ_f"):
            cur = {}
            out.append(cur)
            continue
        if cur is not None:
            if not s.startswith("|"):
                cur = None
                continue
            cells = [c.strip() for c in s.strip("|").split("|")]
            if cells[0].startswith("---"):
                continue
            cur[cells[0]] = cells[1:]
    return out


def group_value(cells_b, label, col, direction):
    m = re.fullmatch(r"(I|II), N = (\d)(?:, n = (\d+))?(?: \(6 сеток\))?", label)
    fam, N, n = m.group(1), m.group(2), m.group(3)
    sel = [c for c in cells_b if c["cell"].startswith(f"{fam}:N{N}:") and (n is None or c["cell"].endswith(f":{n}x{n}"))]
    vals = [Decimal(c[col]) for c in sel]
    return (max(vals) if direction == "up" else min(vals)), len(sel)


def roundoff():
    cb = rows(RES_B / "cells_fft_M2.csv")
    st_rows = rows(RES_B / "steps_fft_M2.csv")
    summ = json.loads((RES_B / "summary.json").read_text(encoding="utf-8"))
    res = {"cells": len(cb), "n_steps_certified": sum(int(c["n_steps_certified"]) for c in cb),
           "n_steps_unresolved": sum(int(c["n_steps_unresolved"]) for c in cb),
           "status_recorded": sorted({c["status_recorded"] for c in cb}), "status_M1": sorted({c["status_M1"] for c in cb}),
           "status_M0": sorted({c["status_M0"] for c in cb}),
           "prefixes_and_records": all(c["M1_prefix"] == c["M0_prefix"] == c["records_ok"] == "True" for c in cb),
           "s_stop_range": [min(int(c["s_stop"]) for c in cb), max(int(c["s_stop"]) for c in cb)]}
    agg = {"max_delta_f": max, "max_eps_cert": max, "max_transfer": max, "max_L_s": max, "max_c_d_obs": max,
           "min_rho_hat": min, "min_rho_star_lo": min, "min_rho_hat_over_eps_ratio": min, "min_energy_margin_s_ge2": min,
           "min_set_sigma_cert_lo": min, "max_eta_N": max}
    res["extrema_from_cells_csv"] = {k: str(f(Decimal(c[k]) for c in cb)) for k, f in agg.items()}
    res["argext"] = {k: (f(cb, key=lambda c: Decimal(c[k])))["cell"] for k, f in
                     {"max_delta_f": max, "max_eps_cert": max, "min_rho_hat": min, "min_rho_hat_over_eps_ratio": min,
                      "min_energy_margin_s_ge2": min}.items()}
    allsum = summ["fft_M2"]["all"]
    res["summary_json_agrees"] = all(Decimal(str(allsum[k])) == Decimal(res["extrema_from_cells_csv"][k])
                                     for k in res["extrema_from_cells_csv"] if k in allsum)
    res["energy_rank_not_certified_s_ge2"] = sum(int(c["energy_rank_not_certified_s_ge2"]) for c in cb)
    res["energy_rank_certified_steps"] = sum(int(c["energy_rank_certified_steps"]) for c in cb)
    res["energy_tests"] = sum(int(c["energy_tests"]) for c in cb)
    res["energy_status_flips"] = sum(int(c["energy_status_flips"]) for c in cb)
    res["D_low_positive_all_cells"] = all(Decimal(c["D_low_N1_lo"]) > 0 for c in cb)
    # statements of the task/report against the extrema
    ex = {k: Decimal(v) for k, v in res["extrema_from_cells_csv"].items()}
    res["claims"] = {
        "delta_f <= 1.3857e-13": ex["max_delta_f"] <= Decimal("1.3857e-13"),
        "eps_cert <= 1.3772e-13": ex["max_eps_cert"] <= Decimal("1.3772e-13"),
        "delta_f <= 1.39e-13 (verdict)": ex["max_delta_f"] <= Decimal("1.39e-13"),
        "eps_cert <= 1.38e-13 (verdict)": ex["max_eps_cert"] <= Decimal("1.38e-13"),
        "rho_hat >= 2.18e-6 (verdict after Z-3)": ex["min_rho_hat"] >= Decimal("2.18e-6"),
        "rho_hat >= 2.19e-6 (before Z-3)": ex["min_rho_hat"] >= Decimal("2.19e-6"),
        "margin >= 1.04e8": ex["min_rho_hat_over_eps_ratio"] >= Decimal("1.04e8"),
        "energy margin >= 0.741": ex["min_energy_margin_s_ge2"] >= Decimal("0.741"),
        "t(s) <= 9.2e-15 (sections 3.2, 5 printed)": ex["max_transfer"] <= Decimal("9.2e-15"),
        "L_s <= 3.8 (section 5 / integration table)": ex["max_L_s"] <= Decimal("3.8"),
        "L_s <= 3.83 (sections 3.2 / review)": ex["max_L_s"] <= Decimal("3.83"),
    }
    # step rows: three different rank statements and the certificates
    n_dec = sum(r["src_total"] != "" for r in st_rows)
    s1 = [r for r in st_rows if r["s"] == "1"]
    sge2 = [r for r in st_rows if int(r["s"]) >= 2]
    rule_ok = all((r["rank_distinguishable"] == "True") == (fl(r["eps_cert"]) < 1 and
                                                            fl(r["rho_hat"]) > fl(r["eps_cert"]) / (1 - fl(r["eps_cert"])))
                  for r in st_rows)
    tri = all(fl(r["delta_f"]) <= (fl(r["e_step"]) + fl(r["transfer"])) * (1 + 1e-6) for r in st_rows)
    route_ok, route_bad = 0, []
    for r in sge2:
        e, d, mg = fl(r["energy_err_up"]), fl(r["dlow_sqrt_lo"]), fl(r["energy_margin"])
        ok = r["energy_rank_certified"] == "True" and e < d and mg > 0 and (1 - e / d) <= mg * (1 + 1e-7) + 1e-9
        route_ok += ok
        if not ok:
            route_bad.append(r["cell"] + ":" + r["s"])
    res["steps_rows"], res["decomposition_rows"] = len(st_rows), n_dec
    res["three_rank_statements"] = {
        "W_star_rank_N (rho_star_lo > 0) rows": f'{sum(fl(r["rho_star_lo"]) > 0 for r in st_rows)}/{len(st_rows)}',
        "W_tilde_rank_N (rho_tilde_lo > 0) rows": f'{sum(fl(r["rho_tilde_lo"]) > 0 for r in st_rows)}/{len(st_rows)}',
        "W_tilde energy route s>=2 rows": f"{route_ok}/{len(sge2)}",
        "W_tilde energy route s=1 rows certified": sum(r["energy_rank_certified"] == "True" for r in s1),
        "W_tilde energy route s=1 cells certified": sorted({r["cell"] for r in s1 if r["energy_rank_certified"] == "True"}),
        "SVD diagnostic rank_distinguishable rows": f'{sum(r["rank_distinguishable"] == "True" for r in st_rows)}/{len(st_rows)}',
        "rank rule reproduced from eps_cert and rho_hat": rule_ok,
        "min rho_tilde_lo": min(fl(r["rho_tilde_lo"]) for r in st_rows),
        "min rho_star_lo": min(fl(r["rho_star_lo"]) for r in st_rows)}
    res["route_failures"] = route_bad[:10]
    res["delta_f_le_e_step_plus_transfer"] = tri
    res["weyl_ok_all"] = all(r["weyl_ok"] == "True" for r in st_rows)
    res["triangle_ok_all"] = all(r["triangle_ok"] == "True" for r in st_rows)
    dlow = {c["cell"]: Decimal(c["D_low_N1_lo"]) for c in cb}
    res["dlow_sqrt_consistent"] = all(Decimal(r["dlow_sqrt_lo"]) ** 2 <= dlow[r["cell"]] * Decimal("1.00000001")
                                      for r in st_rows if r["dlow_sqrt_lo"])
    # dense control
    for tag in ("dense_M1", "dense_M2"):
        cd = rows(RES_B / f"cells_{tag}.csv")
        res[tag] = {"cells": len(cd), "n_steps_certified": sum(int(c["n_steps_certified"]) for c in cd),
                    "unresolved": sum(int(c["n_steps_unresolved"]) for c in cd),
                    "energy_not_certified_s_ge2": sum(int(c["energy_rank_not_certified_s_ge2"] or 0) for c in cd),
                    "max_delta_f": max(fl(c["max_delta_f"]) for c in cd), "max_c_d_obs": max(fl(c["max_c_d_obs"]) for c in cd),
                    "dense_bound_PASS": sum(c["dense_bound_status"] == "PASS" for c in cd)}
    # table 3.1 / section 15 of the QA report
    report = blob(HEAD, "docs/R4_ROUNDOFF_RANK_QUALIFICATION.qa-report.md").decode()
    t31, t15 = tables_31_15(report)
    entries, n_out_15, n_near_31, n_31_not_out = [], 0, 0, 0
    for label, printed31 in t31.items():
        printed15 = t15[label]
        for (col, direction), p31, p15 in zip(TABLE_COLS, printed31, printed15):
            exact, ncell = group_value(cb, label, col, direction)
            v31, ulp31 = parse_printed(p31)
            v15, ulp15 = parse_printed(p15)
            outward = (exact / ulp15).to_integral_value(ROUND_CEILING if direction == "up" else ROUND_FLOOR) * ulp15
            near31 = (exact / ulp31).to_integral_value(ROUND_HALF_EVEN) * ulp31
            out31 = (exact / ulp31).to_integral_value(ROUND_CEILING if direction == "up" else ROUND_FLOOR) * ulp31
            ok15 = v15 == outward
            n_out_15 += ok15
            n_near_31 += v31 == near31
            n_31_not_out += v31 != out31
            entries.append({"group": label, "column": col, "direction": direction, "cells": ncell, "csv_value": str(exact),
                            "printed_3_1": p31, "printed_15": p15, "outward_15_ok": ok15, "nearest_3_1": v31 == near31,
                            "3_1_is_outward": v31 == out31,
                            "3_1_safe_side": (v31 >= exact) if direction == "up" else (v31 <= exact)})
    res["table_entries"] = len(entries)
    res["section15_outward_correct"] = n_out_15
    res["table31_nearest"] = n_near_31
    res["table31_not_outward"] = n_31_not_out
    res["table31_on_wrong_side_of_bound"] = sum(not e["3_1_safe_side"] for e in entries)
    write_csv("table31_rounding.csv", entries)
    checks = [res["cells"] == 132, res["n_steps_certified"] == 13200, res["n_steps_unresolved"] == 0,
              res["prefixes_and_records"], res["summary_json_agrees"], res["energy_rank_not_certified_s_ge2"] == 0,
              res["energy_tests"] == 13068, res["energy_status_flips"] == 0, res["D_low_positive_all_cells"],
              rule_ok, tri, res["weyl_ok_all"], res["triangle_ok_all"], route_ok == len(sge2), n_dec == 2772,
              res["dlow_sqrt_consistent"], res["section15_outward_correct"] == res["table_entries"] == 80,
              res["table31_not_outward"] == 40]
    res["ROUNDOFF_RECOUNT"] = "PASS" if all(checks) else "CHECK"
    res["checks_passed"] = f"{sum(map(bool, checks))}/{len(checks)}"
    write_json("roundoff_recount.json", res)
    print(res["ROUNDOFF_RECOUNT"], res["checks_passed"])
    return res


# ----------------------------------------------------------------------------------------------------------------------
# Part III/IV - targeted controls (python-flint, mpmath)
# ----------------------------------------------------------------------------------------------------------------------

def nodal_halfstep(Ug, hu, hv, nv, prec):
    """Exact Galerkin half-step on span(U) (x) V_h^v: Pi_h(W) = D/2 hu hv |Delta_h W|^2 - q hu hv sum W, D = q = 1.

    Ug - (nu - 1) x N list of arb (given set on the u-grid, exact); hu, hv - fmpq steps; nv - intervals of the solved grid.
    Every basis field phi_(k,i) = U_i (x) e_k gets the five-point Delta_h applied node by node; G = hu hv Phi^T Phi
    (sparse accumulation over shared nodes), f_(k,i) = hu hv sum_l U_li. Returns (G, f, z) as arb_mat; z[k N + i] = V_ki."""
    from flint import arb, arb_mat, ctx
    ctx.prec = prec
    nu1, N = len(Ug), len(Ug[0])
    nv1 = nv - 1
    iu2, iv2 = arb(1) / (arb(hu) * arb(hu)), arb(1) / (arb(hv) * arb(hv))
    LU = [[((Ug[l - 1][i] if l > 0 else 0) - 2 * Ug[l][i] + (Ug[l + 1][i] if l < nu1 - 1 else 0)) * iu2 for i in range(N)]
          for l in range(nu1)]
    cols = []  # column p = k N + i -> {node (l, kk): value}
    for k in range(nv1):
        for i in range(N):
            c = {}
            for l in range(nu1):
                c[(l, k)] = LU[l][i] - 2 * iv2 * Ug[l][i]
                if k > 0:
                    c[(l, k - 1)] = iv2 * Ug[l][i]
                if k < nv1 - 1:
                    c[(l, k + 1)] = iv2 * Ug[l][i]
            cols.append(c)
    P = len(cols)
    w = arb(hu) * arb(hv)
    G = arb_mat(P, P)
    for p in range(P):
        kp = p // N
        for q_ in range(p, P):
            kq = q_ // N
            if kq - kp > 2:
                break
            cp, cq = cols[p], cols[q_]
            acc = arb(0)
            for kk in range(max(kp, kq) - 1, min(kp, kq) + 2):
                if 0 <= kk < nv1:
                    for l in range(nu1):
                        a, b = cp.get((l, kk)), cq.get((l, kk))
                        if a is not None and b is not None:
                            acc += a * b
            G[p, q_] = G[q_, p] = w * acc
    usum = [sum((Ug[l][i] for l in range(nu1)), arb(0)) for i in range(N)]
    f = arb_mat(P, 1)
    for k in range(nv1):
        for i in range(N):
            f[k * N + i, 0] = w * usum[i]
    z = G.solve(f)
    return G, f, z


def mp_sv(Gu, Gv, mp):
    """Singular values of W = U V^T from the Gram matrices (mpmath; midpoints of high-precision balls)."""
    L = mp.cholesky(Gu)
    S = L.T * Gv * L
    ev = sorted(mp.eigsy(S, eigvals_only=True), reverse=True)
    return [mp.sqrt(max(e, 0)) for e in ev]


def to_mp(a, mp):
    return mp.mpf(a.mid().str(80, radius=False)) if hasattr(a, "mid") else mp.mpf(a)


def exact_sine_set(n, N, prec):
    from flint import arb, ctx, fmpq
    ctx.prec = prec
    return [[arb(fmpq((2 * i + 1) * l, n)).sin_pi() for i in range(N)] for l in range(1, n)]


def float_set(A):
    from flint import arb
    return [[arb(float(v)) for v in row] for row in np.asarray(A, dtype=float)]


def closed_form_rho1(N, r, n_x, n_y, dps=50):
    """rho_N of the exact discrete W*(1) for the odd-sine start (decoupled modes, R4_NUMERICAL_SPEC 4.2, 3.2):
    V_ik = F_i psi_k^T 1 / (D (a/2) (mu_k + lambda_i)^2); rho = sigma_N / sigma_1 of the N x (n_y - 1) matrix V."""
    import mpmath as mp
    mp.mp.dps = dps
    a, b = mp.mpf(1), mp.mpf(float(r))
    hx, hy = a / n_x, b / n_y
    rows_ = []
    for i in range(N):
        m = 2 * i + 1
        lam = 4 / hx**2 * mp.sin(m * mp.pi / (2 * n_x)) ** 2
        F = hx * mp.cot(m * mp.pi / (2 * n_x))
        row = []
        for k in range(1, n_y):
            if k % 2 == 0:
                row.append(mp.mpf(0))
                continue
            mu = 4 / hy**2 * mp.sin(k * mp.pi / (2 * n_y)) ** 2
            row.append(F * mp.sqrt(mp.mpf(2) / n_y) * mp.cot(k * mp.pi / (2 * n_y)) / ((a / 2) * (mu + lam) ** 2))
        rows_.append(row)
    V = mp.matrix(rows_)
    G = V * V.T
    ev = sorted(mp.eigsy(G, eigvals_only=True), reverse=True)
    return float(mp.sqrt(ev[-1] / ev[0]))


def closed_form_field1(N, r, n_x, n_y, mp):
    """Exact discrete W*(1) (n_x - 1) x (n_y - 1) for the odd-sine start (same closed form), mpmath."""
    a, b = mp.mpf(1), mp.mpf(float(r))
    hx, hy = a / n_x, b / n_y
    Ys, Xs = [], []
    for i in range(N):
        m = 2 * i + 1
        lam = 4 / hx**2 * mp.sin(m * mp.pi / (2 * n_x)) ** 2
        F = hx * mp.cot(m * mp.pi / (2 * n_x))
        coef = {}
        for k in range(1, n_y, 2):
            mu = 4 / hy**2 * mp.sin(k * mp.pi / (2 * n_y)) ** 2
            coef[k] = F * mp.sqrt(mp.mpf(2) / n_y) * mp.cot(k * mp.pi / (2 * n_y)) / ((a / 2) * (mu + lam) ** 2)
        Ys.append([mp.fsum(c * mp.sqrt(mp.mpf(2) / n_y) * mp.sin(k * mp.pi * j / n_y) for k, c in coef.items())
                   for j in range(1, n_y)])
        Xs.append([mp.sin(m * mp.pi * l / n_x) for l in range(1, n_x)])
    return Xs, Ys


def compare_cell_rows(cell, committed):
    """Fresh run_cell against the committed steps.csv rows (values printed with the Builder formats)."""
    st = sys.modules["r4_n4_n6_study"]
    _, srows = st.cell_rows(cell)
    buf = io.StringIO()
    keys = list(srows[0].keys())
    w = csv.DictWriter(buf, fieldnames=keys, lineterminator="\n")
    w.writeheader()
    for r in srows:
        w.writerow({k: (f"{v:.15e}" if k in st.HIGH_PRECISION else f"{v:.6e}") if isinstance(v, float) else
                    ("" if v is None else v) for k, v in r.items()})
    fresh = list(csv.DictReader(io.StringIO(buf.getvalue())))
    diffs = 0
    for a, b in zip(fresh, committed):
        diffs += sum(a[k] != b.get(k, a[k]) for k in a)
    return len(fresh), len(committed), diffs


NODAL_CONTROLS = [  # (label, N, r, family, n_x, n_y, s list) - small grids, own nodal ball-arithmetic half-step
    ("N4 normal", 4, "1", "I", 40, 40, (1, 2, 3, 10, 40)),
    ("N4 coarse grid r=5 (large discretization error)", 4, "5", "I", 40, 40, (1, 2, 40)),
    ("N5 smallest guard margin", 5, "1/5", "I", 40, 40, (1, 2, 3, 40)),
    ("N6 r=1/5 stop", 6, "1/5", "I", 40, 40, (1,)),
    ("N6 r=1/3 stop", 6, "1/3", "I", 40, 40, (1,)),
    ("N6 r=5 small margin", 6, "5", "I", 40, 40, (1, 2, 40)),
    ("N6 r=1/2 small margin", 6, "1/2", "I", 40, 40, (1, 2)),
    ("N4 rectangular h_x = h_y, 12x60", 4, "5", "R", 12, 60, (1, 2, 3, 40)),
    ("N6 rectangular h_x = h_y, 60x12 (stop)", 6, "1/5", "R", 60, 12, (1,)),
    ("N3 control", 3, "5", "I", 40, 40, (1, 2, 40)),
]
RERUN_CONTROLS = [  # fresh reruns of committed cells (solver path), compared with steps.csv / cells.csv
    (4, "1", "I", 160, 160), (5, "1/5", "I", 40, 40), (6, "1/5", "I", 1280, 1280), (6, "1/3", "II", 960, 320),
    (6, "5", "I", 1280, 1280), (6, "1/2", "II", 640, 320), (4, "5", "II", 320, 1600), (5, "1/3", "I", 640, 640),
]
RHO1_CONTROLS = [(6, "1/5", 40, 40), (6, "1/5", 1280, 1280), (6, "1/5", 1600, 320), (6, "1/3", 40, 40), (6, "1/3", 1280, 1280),
                 (6, "1/3", 960, 320), (5, "1/5", 40, 40), (6, "5", 40, 40), (6, "5", 1280, 1280), (4, "1", 40, 40)]
DELTA_F1_CELLS = [(N, r, n) for N in (2, 3) for r in ("1/5", "1", "5") for n in (40, 160)]


def controls():
    import mpmath as mp
    from flint import arb, fmpq
    st = load_script("r4_n4_n6_study")
    import kkvim.coupled_run as cr
    steps_c = rows(RES_A / "steps.csv")
    cells_c = rows(RES_A / "cells.csv")
    rho_qa = {(r["N"], r["r_label"], r["nx"], r["ny"]): r for r in rows(RES_AQ / "rho_s1.csv")}
    out, t0 = [], time.perf_counter()
    prec = 640
    mp.mp.dps = 120
    # (1) own nodal half-step in ball arithmetic against the solver half-step
    for label, N, lab, fam, nx, ny, s_list in NODAL_CONTROLS:
        r = Fraction(lab)
        b = Fraction(float(r))
        hx, hy = fmpq(1, nx), fmpq(b.numerator, b.denominator * ny)
        for s in s_list:
            cell = st.run_cell(N, lab, fam, nx, ny, half_steps=s, keep=())
            if cell.s_reached < s:
                break
            log = cell.steps[-1]
            direction = log.direction
            if direction == "Y":
                given, received, hu, hv, nv = cell.x, cell.y, hx, hy, ny
            else:
                given, received, hu, hv, nv = cell.y, cell.x, hy, hx, nx
            G, f, z = nodal_halfstep(float_set(given), hu, hv, nv, prec)
            Nn = given.shape[1]
            zh = float_set(received.reshape(-1, 1) if False else received)  # (nv-1) x N
            from flint import arb_mat
            zhat = arb_mat(len(zh) * Nn, 1)
            for k in range(len(zh)):
                for i in range(Nn):
                    zhat[k * Nn + i, 0] = zh[k][i]
            e = zhat - z
            Ge = G * e
            err_a = (e.transpose() * Ge)[0, 0]
            nz = (z.transpose() * (G * z))[0, 0]
            resid = G * zhat - f
            fmax = max(abs(f[p, 0]) for p in range(f.nrows()))
            rmax = max(abs(resid[p, 0]) for p in range(resid.nrows()))
            # fields: W* = U V*^T and W^ = U V^^T; relative max-norm error at the nodes
            Uf = float_set(given)
            Vs = [[z[k * Nn + i, 0] for i in range(Nn)] for k in range(len(zh))]
            Wd = mp.matrix(len(Uf), len(Vs))
            Wsm = 0
            for l in range(len(Uf)):
                for k in range(len(Vs)):
                    ws = sum((Uf[l][i] * Vs[k][i] for i in range(Nn)), arb(0))
                    wh = sum((Uf[l][i] * zh[k][i] for i in range(Nn)), arb(0))
                    Wd[l, k] = to_mp(wh - ws, mp)
                    Wsm = max(Wsm, abs(to_mp(ws, mp)))
            e_field = max(abs(Wd[l, k]) for l in range(len(Uf)) for k in range(len(Vs))) / Wsm
            Gu = mp.matrix([[to_mp(sum((Uf[l][i] * Uf[l][j] for l in range(len(Uf))), arb(0)), mp) for j in range(Nn)] for i in range(Nn)])
            Gv = mp.matrix([[to_mp(sum((Vs[k][i] * Vs[k][j] for k in range(len(Vs))), arb(0)), mp) for j in range(Nn)] for i in range(Nn)])
            sv = mp_sv(Gu, Gv, mp)
            rho_star = float(sv[-1] / sv[0])
            Dn = mp.diag([1 / mp.sqrt(Gv[i, i]) for i in range(Nn)])
            set_star = float(mp.sqrt(min(mp.eigsy(Dn * Gv * Dn, eigvals_only=True))))
            # exact energies: Pi_h(W) = 1/2 z^T G z - f^T z
            pi_star = to_mp(-(f.transpose() * z)[0, 0] / 2, mp)
            pi_hat = to_mp(((zhat.transpose() * (G * zhat))[0, 0]) / 2 - (f.transpose() * zhat)[0, 0], mp)
            thr = K_RESEARCH * U * log.rank.kappa_rep
            row = {"control": label, "N": N, "r": lab, "grid": f"{nx}x{ny}", "s": s, "direction": direction,
                   "solver_status": cell.status, "solver_s_reached": cell.s_reached,
                   "field_err_rel_max": float(e_field),
                   "energy_norm_err_rel": float(mp.sqrt(to_mp(err_a, mp) / to_mp(nz, mp))),
                   "galerkin_residual_rel": float(to_mp(rmax, mp) / to_mp(fmax, mp)),
                   "Pi_solver_log": log.energy.P if log.energy else None,
                   "Pi_exact_of_solver_field_rel_diff": float(abs(pi_hat - to_mp(arb(log.energy.P), mp)) / abs(pi_hat)) if log.energy else None,
                   "Pi_solver_field_minus_Pi_star_rel": float((pi_hat - pi_star) / abs(pi_star)),
                   "rho_hat": log.rank.rho_N, "rho_star_halfstep": rho_star,
                   "rho_rel_diff": abs(log.rank.rho_N - rho_star) / rho_star,
                   "rho_over_threshold_exact": rho_star / thr, "guard_rank_solver": log.guards.rank_ok if log.guards else None,
                   "guard_rank_exact": rho_star > thr,
                   "set_hat": log.rank.set_sigma_min, "set_star": set_star,
                   "guard_set_solver": log.guards.set_ok if log.guards else None, "guard_set_exact": set_star > K_RESEARCH * U,
                   "eta_N": log.eta_N, "ball_rel_radius_z": float(max(to_mp(z[p, 0].rad(), mp) for p in range(z.nrows())) /
                                                                 max(abs(to_mp(z[p, 0], mp)) for p in range(z.nrows())))}
            out.append(row)
            print(label, s, f"{row['field_err_rel_max']:.2e}", f"{row['rho_star_halfstep']:.6e}",
                  f"{time.perf_counter() - t0:.0f}s", flush=True)
    # (2) global exact W*(1) with exact sines: nodal ball arithmetic and closed form (two routes), vs QA and solver
    rho1 = []
    for N, lab, nx, ny in RHO1_CONTROLS:
        cf = closed_form_rho1(N, Fraction(lab), nx, ny)
        row = {"N": N, "r": lab, "grid": f"{nx}x{ny}", "rho_closed_form": cf}
        if nx <= 60 and ny <= 60:
            b = Fraction(float(Fraction(lab)))
            G, f, z = nodal_halfstep(exact_sine_set(nx, N, prec), fmpq(1, nx), fmpq(b.numerator, b.denominator * ny), ny, prec)
            Us = exact_sine_set(nx, N, prec)
            Vs = [[z[k * N + i, 0] for i in range(N)] for k in range(ny - 1)]
            Gu = mp.matrix([[to_mp(sum((Us[l][i] * Us[l][j] for l in range(nx - 1)), arb(0)), mp) for j in range(N)] for i in range(N)])
            Gv = mp.matrix([[to_mp(sum((Vs[k][i] * Vs[k][j] for k in range(ny - 1)), arb(0)), mp) for j in range(N)] for i in range(N)])
            sv = mp_sv(Gu, Gv, mp)
            row["rho_nodal_exact_sines"] = float(sv[-1] / sv[0])
            row["two_routes_rel_diff"] = abs(row["rho_nodal_exact_sines"] - cf) / cf
        fam = "I" if nx == ny else "II"
        q = rho_qa.get((str(N), lab, str(nx), str(ny)))
        if q:
            row["rho_qa_lo"], row["rho_qa_hi"] = fl(q["rho_star_lo"]), fl(q["rho_star_hi"])
            row["closed_form_inside_qa_interval_1e-9"] = row["rho_qa_lo"] * (1 - 1e-9) <= cf <= row["rho_qa_hi"] * (1 + 1e-9)
        srow = next((x for x in steps_c if x["N"] == str(N) and x["r_label"] == lab and x["nx"] == str(nx) and
                     x["ny"] == str(ny) and x["s"] == "1"), None)
        if srow:
            row["rho_hat_committed"] = fl(srow["rho_N"])
            row["rho_hat_rel_diff"] = abs(row["rho_hat_committed"] - cf) / cf
            row["rho_over_threshold_exact"] = cf / (K_RESEARCH * U * fl(srow["kappa_rep"]))
            row["guard_rank_committed"] = srow["guard_rank_ok"]
            row["guard_rank_exact"] = cf > K_RESEARCH * U * fl(srow["kappa_rep"])
        rho1.append(row)
        print("rho1", N, lab, nx, ny, f"{cf:.8e}", flush=True)
    # (3) fresh reruns of committed cells: steps.csv rows as text, cells.csv core fields
    reruns = []
    for N, lab, fam, nx, ny in RERUN_CONTROLS:
        t1 = time.perf_counter()
        cell = st.run_cell(N, lab, fam, nx, ny, half_steps=st.S_RUN, keep=())
        committed = [x for x in steps_c if x["N"] == str(N) and x["r_label"] == lab and x["family"] == fam and
                     x["nx"] == str(nx) and x["ny"] == str(ny)]
        crow = next(x for x in cells_c if x["N"] == str(N) and x["r_label"] == lab and x["family"] == fam and
                    x["nx"] == str(nx) and x["ny"] == str(ny))
        nf, nc, d = compare_cell_rows(cell, committed)
        reruns.append({"N": N, "r": lab, "family": fam, "grid": f"{nx}x{ny}", "status": cell.status,
                       "status_committed": crow["status"], "s_reached": cell.s_reached, "s_committed": int(crow["s_reached"]),
                       "first_stab": cell.first_stabilization_s, "first_stab_committed": crow["first_stabilization_s"],
                       "rows_fresh": nf, "rows_committed": nc, "text_field_diffs": d,
                       "seconds": round(time.perf_counter() - t1, 1)})
        print("rerun", N, lab, fam, nx, ny, cell.status, d, flush=True)
    # (4) package B: delta_f(1) = |W~(1) - W*(1)|_F / sigma_1(W*(1)) by the closed form, vs steps_fft_M2.csv
    mp.mp.dps = 40
    srb = {(r["cell"], r["s"]): r for r in rows(RES_B / "steps_fft_M2.csv")}
    df1 = []
    for N, lab, n in DELTA_F1_CELLS:
        run = cr.run_coupled(1.0, float(Fraction(lab)), 1.0, 1.0, n, n, N, transform="fft", half_steps=1, research_mode=True)
        X, Y = run.x, run.y
        Xs, Ys = closed_form_field1(N, Fraction(lab), n, n, mp)
        num = mp.mpf(0)
        for l in range(n - 1):
            for j in range(n - 1):
                wt = mp.fsum(mp.mpf(float(X[l, i])) * mp.mpf(float(Y[j, i])) for i in range(N))
                ws = mp.fsum(Xs[i][l] * Ys[i][j] for i in range(N))
                num += (wt - ws) ** 2
        Gx = mp.matrix([[mp.fsum(Xs[i][l] * Xs[k][l] for l in range(n - 1)) for k in range(N)] for i in range(N)])
        Gy = mp.matrix([[mp.fsum(Ys[i][l] * Ys[k][l] for l in range(n - 1)) for k in range(N)] for i in range(N)])
        s1 = mp_sv(Gx, Gy, mp)[0]
        val = float(mp.sqrt(num) / s1)
        key = (f"I:N{N}:r{lab}:{n}x{n}", "1")
        qa = fl(srb[key]["delta_f"]) if key in srb else math.nan
        df1.append({"cell": key[0], "delta_f1_audit": val, "delta_f1_qa": qa, "rel_diff": abs(val - qa) / qa,
                    "qa_is_upper_within_1e-8": qa >= val * (1 - 1e-8)})
        print("delta_f1", key[0], f"{val:.9e}", f"{qa:.9e}", flush=True)
    write_csv("controls_nodal.csv", out)
    write_csv("controls_rho1.csv", rho1)
    write_csv("controls_rerun.csv", reruns)
    write_csv("controls_delta_f1.csv", df1)
    summary = {
        "nodal_rows": len(out), "max_field_err_rel": max(r["field_err_rel_max"] for r in out),
        "max_energy_norm_err_rel": max(r["energy_norm_err_rel"] for r in out),
        "max_galerkin_residual_rel": max(r["galerkin_residual_rel"] for r in out),
        "max_Pi_rel_diff": max(r["Pi_exact_of_solver_field_rel_diff"] for r in out if r["Pi_exact_of_solver_field_rel_diff"] is not None),
        "max_rho_rel_diff": max(r["rho_rel_diff"] for r in out),
        "guard_decisions_equal": all(r["guard_rank_solver"] == r["guard_rank_exact"] and r["guard_set_solver"] == r["guard_set_exact"]
                                     for r in out if r["guard_rank_solver"] is not None),
        "max_ball_rel_radius": max(r["ball_rel_radius_z"] for r in out),
        "rho1_rows": len(rho1), "rho1_two_routes_max_rel_diff": max(r.get("two_routes_rel_diff", 0) for r in rho1),
        "rho1_inside_qa": all(r.get("closed_form_inside_qa_interval_1e-9", True) for r in rho1),
        "rho1_guard_equal": all(str(r["guard_rank_exact"]) == r["guard_rank_committed"] for r in rho1 if "guard_rank_exact" in r),
        "rho1_max_solver_rel_diff": max(r.get("rho_hat_rel_diff", 0) for r in rho1),
        "reruns": len(reruns), "reruns_text_equal": all(r["text_field_diffs"] == 0 and r["rows_fresh"] == r["rows_committed"]
                                                         for r in reruns),
        "reruns_status_equal": all(r["status"] == r["status_committed"] and r["s_reached"] == r["s_committed"] for r in reruns),
        "delta_f1_cells": len(df1), "delta_f1_max_rel_diff": max(r["rel_diff"] for r in df1),
        "delta_f1_qa_upper": all(r["qa_is_upper_within_1e-8"] for r in df1),
        "seconds": round(time.perf_counter() - t0, 1), "prec_bits": prec}
    write_json("controls_summary.json", summary)
    print(json.dumps(summary, indent=1))
    return summary


# ----------------------------------------------------------------------------------------------------------------------
# Part VI - two configurations
# ----------------------------------------------------------------------------------------------------------------------

CONFIG_CELLS = [(5, "1/5", "I", 40, 40), (6, "5", "I", 160, 160), (6, "1/5", "I", 160, 160), (4, "5", "II", 320, 1600),
                (6, "1", "I", 640, 640)]
CONFIG_N23 = [(3, "5", 160), (2, "1/5", 640), (3, "1/3", 320)]


def config(tag):
    st = load_script("r4_n4_n6_study")
    import kkvim.coupled_run as cr
    fp = {"python": sys.version.split()[0], "numpy": np.__version__, "blas_threads": os.environ.get("OPENBLAS_NUM_THREADS", "default"),
          "cells": {}}
    def step_rec(log):
        g = log.guards
        return {"s": log.s, "rho_N": log.rank.rho_N, "Pi_h": log.energy.P if log.energy else None,
                "energy_status": log.energy.status if log.energy else None, "guard_rank_ok": g.rank_ok if g else None,
                "guard_set_ok": g.set_ok if g else None, "guard_applicable": g.applicable if g else None,
                "stabilization_ok": log.stabilization_ok, "stop_reason": log.stop_reason}
    for N, lab, fam, nx, ny in CONFIG_CELLS:
        cell = st.run_cell(N, lab, fam, nx, ny, half_steps=st.S_RUN, keep=(st.S_RUN,))
        last = max(cell.fields) if cell.fields else None
        fp["cells"][f"{N}:{lab}:{fam}:{nx}x{ny}"] = {
            "status": cell.status, "s_reached": cell.s_reached, "first_stabilization_s": cell.first_stabilization_s,
            "centre": [e.get("centre") for e in cell.extra], "steps": [step_rec(l) for l in cell.steps],
            "sha_sets": sha256(np.ascontiguousarray(cell.x).tobytes() + np.ascontiguousarray(cell.y).tobytes()) if cell.x is not None else None}
    for N, lab, n in CONFIG_N23:
        run = cr.run_coupled(1.0, float(Fraction(lab)), 1.0, 1.0, n, n, N, transform="fft", half_steps=100, research_mode=True,
                             stop_on_stabilization=False)
        fp["cells"][f"run_coupled:{N}:{lab}:{n}"] = {
            "status": run.status, "s_reached": len(run.steps), "first_stabilization_s": run.first_stabilization_s,
            "centre": None, "steps": [step_rec(l) for l in run.steps],
            "sha_sets": sha256(np.ascontiguousarray(run.x).tobytes() + np.ascontiguousarray(run.y).tobytes())}
    write_json(f"config_{tag}.json", fp, compact=True)
    print(tag, {k: (v["status"], v["s_reached"]) for k, v in fp["cells"].items()})


def config_compare(tags):
    fps = {t: json.loads((OUT / f"config_{t}.json").read_text(encoding="utf-8")) for t in tags}
    ref_tag = tags[0]
    ref = fps[ref_tag]
    res = {"reference": ref_tag, "criteria": {"classification": CFG_CLASSIFICATION, "rel_centre_Pi": CFG_REL,
                                              "abs_rho": CFG_RHO_ABS}, "pairs": {}}
    for t in tags[1:]:
        o = fps[t]
        cls_diff, max_rel, max_rho, bitwise = 0, 0.0, 0.0, 0
        for key, a in ref["cells"].items():
            b = o["cells"][key]
            for k in ("status", "s_reached", "first_stabilization_s"):
                cls_diff += a[k] != b[k]
            bitwise += a["sha_sets"] == b["sha_sets"]
            for sa, sb in zip(a["steps"], b["steps"]):
                for k in ("energy_status", "guard_rank_ok", "guard_set_ok", "guard_applicable", "stabilization_ok", "stop_reason"):
                    cls_diff += sa[k] != sb[k]
                if sa["Pi_h"] is not None and sb["Pi_h"] is not None:
                    max_rel = max(max_rel, abs(sa["Pi_h"] - sb["Pi_h"]) / abs(sa["Pi_h"]))
                max_rho = max(max_rho, abs(sa["rho_N"] - sb["rho_N"]))
            if a["centre"] and b["centre"]:
                max_rel = max(max_rel, max(abs(x - y) / abs(x) for x, y in zip(a["centre"], b["centre"]) if x))
        res["pairs"][f"{ref_tag} vs {t}"] = {"classification_differences": cls_diff, "max_rel_centre_Pi": max_rel,
                                             "max_abs_rho": max_rho, "bitwise_equal_sets": f"{bitwise}/{len(ref['cells'])}",
                                             "PASS": cls_diff == 0 and max_rel <= CFG_REL and max_rho <= CFG_RHO_ABS}
    res["ALL_PASS"] = all(p["PASS"] for p in res["pairs"].values())
    write_json("config_compare.json", res)
    print(json.dumps(res, indent=1))


def collect():
    meta = {"tool": "scripts/r4_owner_readiness_audit.py", "object_head": HEAD, "base": BASE,
            "git_head_at_collect": git("rev-parse", "HEAD").strip(),
            "tool_sha256_lf": sha256((ROOT / "scripts" / "r4_owner_readiness_audit.py").read_bytes().replace(b"\r\n", b"\n")),
            "python": sys.version.split()[0], "numpy": np.__version__, "platform": platform.platform(),
            "outputs": {}}
    for p in sorted(OUT.iterdir()):
        if p.name != "meta.json":
            meta["outputs"][p.name] = sha256(p.read_bytes().replace(b"\r\n", b"\n"))
    write_json("meta.json", meta)
    print(json.dumps(meta, indent=1))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("mode", choices=["integrity", "n46", "roundoff", "controls", "config", "config-compare", "collect"])
    p.add_argument("--tag", default=None)
    p.add_argument("--tags", nargs="*", default=None)
    a = p.parse_args(argv)
    if a.mode == "integrity":
        integrity()
    elif a.mode == "n46":
        n46()
    elif a.mode == "roundoff":
        roundoff()
    elif a.mode == "controls":
        controls()
    elif a.mode == "config":
        config(a.tag or f"py{sys.version_info.major}{sys.version_info.minor}")
    elif a.mode == "config-compare":
        config_compare(a.tags)
    else:
        collect()
    return 0


if __name__ == "__main__":
    sys.exit(main())
