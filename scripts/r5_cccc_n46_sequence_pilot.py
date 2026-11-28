"""Isolated research pilot of CCCC N = 4-6 beyond s = 2 (task R5-CCCC-N46-FULL-SEQUENCE-PILOT-01; goals G3.4, G4, G5; R5-V).

EXPERIMENTAL. Every result is RESEARCH_ONLY / PROVISIONAL / NOT_QUALIFIED. This is not a production computation: ``run-cccc-n46`` and
``kkvim.cccc_n46`` still stop before s = 3 (D-068, 7 G); nothing in ``src/kkvim`` is changed or bypassed.

Authorization: the owner's task R5-CCCC-N46-FULL-SEQUENCE-PILOT-01 (2026-11-25) - a narrow research-only exception to the stopping rules of
D-057.1 and D-068.5 for this standalone script only: N = 4, 5, 6, a budget of at most 32 half-steps, grids within the D-068 small-grid domain
(n_x, n_y <= 32). No new decision number is assigned; the scope is recorded in every result.json for the later integration. D-066 (the
N = 2-3 pilot) is NOT extended by this script and is not its authorization.

Three separate execution paths:

    run      (imports ``kkvim`` only)            the half-steps s = 1..K, K <= 32, Y, X, Y, X, ...
    compare  (imports ``kkvim_reference`` only)  the independent D2 reference on the same grid; run as a separate process
    summary  (imports neither)                   one CSV / JSON and SVG plots over several saved runs

    uv run python scripts/r5_cccc_n46_sequence_pilot.py run --N 4 --nx 16 --ny 16 --half-steps 32 --out scratch/n46_pilot/n4_16
    uv run python scripts/r5_cccc_n46_sequence_pilot.py compare --out scratch/n46_pilot/n4_16       (``run`` starts it unless --no-compare)
    uv run python scripts/r5_cccc_n46_sequence_pilot.py summary --out scratch/n46_pilot/summary scratch/n46_pilot/n4_16 ...

s = 1 and s = 2 are those of ``kkvim.cccc_n46.run_cccc_n46_physical(..., half_steps=min(K, 2))`` unchanged (D-067; D-068: RN-5 before the QR
of V, RT-7 after it, grid <= 32). A stop there ends the pilot: nothing is continued after a failed RN-5 or RT-7. Every half-step s >= 3 is
``pilot_step`` of scripts/r5_cccc_n23_sequence_pilot.py, imported and not copied: the operations of s = 2 with the roles of x and y
alternating (exact dependence and nonfinite input -> stop before QR; paired weighted QR and companion G_{s-1} R^T; C-3+ assembly with the
rounding probes and premises -> stop before the solve; B-C solve; eta, exact rank of the stored factors, rank diagnostic, field preservation,
P_hat, alpha_centre written as OBSERVATIONS). The same function needs no number of terms: no threshold of N = 2-3 is used (pilot_step has no
eta guard and no rank cutoff; the premises are those of the D-067 production s = 1 of N = 4-6).

No RN-5 / RT-7 at s >= 3 and none is invented: the rank of the exact discrete sequence beyond the certified initialization is UNRESOLVED; the
exact full rank of the stored binary64 factors is an algebraic diagnostic only, never a certificate of the exact discrete sequence. No rank
reduction, regularization, fallback solver or fabricated row. Stops:

    RESEARCH_LIMIT_REACHED          the budget K is reached
    STOPPED_BITWISE_REPEATED_STATE  the stored (G, V) of a done half-step equal bitwise those of an earlier one of the same phase;
                                    a computational observation, not convergence
    PILOT_STOPPED                   a protective stop of production s = 1, 2 or of ``pilot_step``; the reason names it

No CONVERGED status exists. The D2 comparison (``compare``) is the D-066 procedure of the N = 2-3 pilot (``command_compare``, imported) with
the marking of this pilot; D is recomputed there from t, E, nu. Every difference is an OBSERVATION of the discrete problem (D-049.1), not
a grid-error or continuous-model accuracy measure; no acceptable-error threshold is used.
"""

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
N23_PILOT = ROOT / "scripts" / "r5_cccc_n23_sequence_pilot.py"
TASK = "R5-CCCC-N46-FULL-SEQUENCE-PILOT-01"
MARKING = (f"RESEARCH_ONLY / PROVISIONAL / NOT_QUALIFIED (owner task {TASK}, experimental N = 4-6 pilot; not a production computation; "
           "not D-066)")
AUTHORIZATION = (f"s = 1: D-067; s = 2: D-068 (RN-5 before QR, RT-7 after QR, n_x, n_y <= 32); s = 3..32: owner task {TASK} "
                 "(2026-11-25), a research-only exception to D-057.1 and D-068.5 for this script only; no new decision number; "
                 "production continuation beyond s = 2 is not authorized")
N_VALUES = (4, 5, 6)
GRID_MAX = 32  # the D-068 small-grid domain of the continuation; a bound of the authorization, not an accuracy certificate
ARTIFACTS = ("result.json", "history.csv", "fields.npz", "factors.npz", "comparison.csv", "comparison.json")
CHECKPOINTS = (1, 2, 8, 16, 32)


def _load_n23_pilot():
    """scripts/r5_cccc_n23_sequence_pilot.py as a module (its top level imports neither kkvim nor kkvim_reference)."""
    spec = importlib.util.spec_from_file_location("r5_cccc_n23_sequence_pilot", N23_PILOT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


p23 = _load_n23_pilot()
RANK_S3 = p23.RANK_S3
RESEARCH_LIMIT_REACHED = p23.RESEARCH_LIMIT_REACHED
STOPPED_BITWISE_REPEATED_STATE = p23.STOPPED_BITWISE_REPEATED_STATE
PILOT_STOPPED = p23.PILOT_STOPPED
BUDGET_MAX = p23.BUDGET_MAX
HISTORY_COLUMNS = p23.HISTORY_COLUMNS
RANK_STATUS = {1: "s = 1: exact rank of the stored factors; D-067 (OBSERVATION)", 2: "s = 2: RN-5 + RT-7 (D-068)"}


def provenance() -> dict:
    import scipy

    out = p23.provenance()
    out["script"] = "scripts/r5_cccc_n46_sequence_pilot.py (reuses scripts/r5_cccc_n23_sequence_pilot.py: pilot_step, repeated_state, _row, " \
                    "command_compare)"
    for name, module in (("numpy", np), ("scipy", scipy)):
        try:
            blas = module.show_config(mode="dicts")["Build Dependencies"]["blas"]
            out[f"{name}_blas"] = {k: blas.get(k) for k in ("name", "version", "openblas configuration")}
        except Exception as exc:  # noqa: BLE001 - provenance only
            out[f"{name}_blas"] = f"UNAVAILABLE ({type(exc).__name__})"
    return out


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --- the working path ----------------------------------------------------------------------------------------------------------------

def refusal(N, n_x, n_y, half_steps) -> str | None:
    """Why the input is outside this pilot's authorization, or None (the production domain of D-067 is checked by kkvim.cccc_n46)."""
    if isinstance(N, bool) or N not in N_VALUES:
        return f"N must be 4, 5 or 6, got {N!r}"
    if not (isinstance(half_steps, int) and not isinstance(half_steps, bool) and 1 <= half_steps <= BUDGET_MAX):
        return f"half_steps must be an integer 1..{BUDGET_MAX}, got {half_steps!r}"
    for name, value in (("n_x", n_x), ("n_y", n_y)):
        if isinstance(value, bool) or not isinstance(value, int) or value > GRID_MAX:
            return f"{name} must be an integer <= {GRID_MAX} (D-068 small-grid domain), got {value!r}"
    return None


def run_pilot(a, b, t, E, nu, q, n_x, n_y, N, half_steps):
    """(status, reason, production result of s = 1..2, all rows). s = 1, 2: production unchanged; s >= 3: ``p23.pilot_step``."""
    import kkvim.cccc_n46 as c46

    reason = refusal(N, n_x, n_y, half_steps)
    if reason is not None:
        raise ValueError(reason)
    result = c46.run_cccc_n46_physical(a, b, t, E, nu, q, n_x, n_y, N, min(half_steps, 2))
    rows = list(result.steps)
    if result.status != c46.COMPLETED_FIXED_HALF_STEPS:
        return PILOT_STOPPED, f"s = {max(len(rows), 1)}: {result.status}: {result.reason}", result, rows
    for step in rows:
        step.status = c46.HALF_STEP_DONE
    prev = rows[-1]
    for s in range(3, half_steps + 1):
        prev = p23.pilot_step(prev, s, result.a, result.b, result.D, result.q, result.n_x, result.n_y)
        rows.append(prev)
        if not prev.done:
            return PILOT_STOPPED, f"s = {s}: {prev.status}: {prev.reason}", result, rows
        prev.repeated_state_of = p23.repeated_state(rows, prev)
        if prev.repeated_state_of is not None:
            reason = (f"s = {s}: G и V побитово равны сохранённым на s = {prev.repeated_state_of} той же фазы "
                      "(повтор вычислительного состояния; не доказательство сходимости)")
            return STOPPED_BITWISE_REPEATED_STATE, reason, result, rows
    return RESEARCH_LIMIT_REACHED, None, result, rows


def history_row(step, D, q, a, b, n_x, n_y, previous_field, phase_field):
    """``p23._row`` with the N = 4-6 rank status of s = 1, 2 (D-067, D-068 instead of D-063, D-065)."""
    row, field = p23._row(step, D, q, a, b, n_x, n_y, previous_field, phase_field)
    row["rank_status"] = RANK_STATUS.get(step.s, RANK_S3)
    return row, field


def command_run(args) -> int:
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        print(f"refused: {out} exists and is not empty (no overwrite)", file=sys.stderr)
        return 2
    reason = refusal(args.N, args.nx, args.ny, args.half_steps)
    if reason is not None:
        print(f"refused: {reason}", file=sys.stderr)
        return 2
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    status, reason, result, rows = run_pilot(args.a, args.b, args.t, args.E, args.nu, args.q, args.nx, args.ny, args.N, args.half_steps)
    elapsed = time.perf_counter() - started
    D = result.D
    history, fields, factors = [], {}, {}
    previous = np.zeros((args.nx + 1, args.ny + 1))
    for step in rows:
        row, field = history_row(step, D, args.q, args.a, args.b, args.nx, args.ny, previous, fields.get(f"w_s{step.s - 2}"))
        history.append(row)
        if field is not None:
            fields[f"w_s{step.s}"] = field
            previous = field
        for name in ("G", "V", "R", "companion", "given"):
            value = getattr(step, name, None)
            if value is not None:
                factors[f"{name}_s{step.s}"] = value
    np.savez(out / "fields.npz", **fields)
    np.savez(out / "factors.npz", **factors)
    with open(out / "history.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(history)
    digest = hashlib.sha256(b"".join(fields[k].tobytes() for k in sorted(fields))).hexdigest()
    done = sum(1 for step in rows if step.done)
    summary = {"task": TASK, "decision": "D-067 (s = 1), D-068 (s = 2); s >= 3: owner task scope, no decision number",
               "authorization": AUTHORIZATION, "marking": MARKING, "status": status, "reason": reason,
               "stops": [RESEARCH_LIMIT_REACHED, STOPPED_BITWISE_REPEATED_STATE, PILOT_STOPPED],
               "convergence": "NOT ASSESSED (no CONVERGED status)",
               "half_steps_requested": args.half_steps, "half_steps_completed": done,
               "last_completed_s": max((step.s for step in rows if step.done), default=None),
               "production_status_s1_s2": result.status, "production_reason_s1_s2": result.reason,
               "sequence": "Y, X, Y, X, ... (s = 1 solves Y on the start set T_N in x, D-049.3)",
               "input": {"a": args.a, "b": args.b, "t": args.t, "E": args.E, "nu": args.nu, "q": args.q, "n_x": args.nx, "n_y": args.ny,
                         "N": args.N, "D_solver": D},
               "input_note": "demonstration parameters of PR #74, not P82 material parameters, not an author reproduction",
               "s1_s2": "kkvim.cccc_n46.run_cccc_n46_physical(..., half_steps=min(K, 2)) unchanged",
               "s_ge_3": RANK_S3, "elapsed_s": elapsed, "fields_sha256": digest, "provenance": provenance(), "history": history}
    (out / "result.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"{status}: {reason or ''}\n  completed {done}/{args.half_steps} in {elapsed:.1f} s, fields sha256 {digest[:16]}, out {out}", flush=True)
    code = 6 if status == PILOT_STOPPED else 0
    if args.no_compare:
        write_manifest(out)
        return code
    import subprocess

    finished = subprocess.run([sys.executable, str(Path(__file__).resolve()), "compare", "--out", str(out)], cwd=ROOT, check=False)
    return finished.returncode or code


def write_manifest(out: Path) -> dict:
    manifest = {name: sha256(out / name) for name in ARTIFACTS if (out / name).exists()}
    (out / "artifacts.sha256.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


# --- the reference path (separate process; no kkvim import) ------------------------------------------------------------------------

def command_compare(args) -> int:
    """The D-066 comparison procedure (``p23.command_compare``) with this pilot's marking, then its N = 4-6 metadata and the manifest."""
    out = Path(args.out)
    p23.MARKING = MARKING  # the comparison files carry the marking of this pilot, not that of D-066
    started = time.perf_counter()
    code = p23.command_compare(args)
    comparison = json.loads((out / "comparison.json").read_text(encoding="utf-8"))
    result = json.loads((out / "result.json").read_text(encoding="utf-8"))
    comparison.update(task=TASK, authorization=AUTHORIZATION, N=result["input"]["N"], grid=[result["input"]["n_x"], result["input"]["n_y"]],
                      procedure="D-066 comparison procedure of scripts/r5_cccc_n23_sequence_pilot.py (command_compare), unchanged",
                      elapsed_s=time.perf_counter() - started, kkvim_imported_in_this_process="kkvim" in sys.modules)
    (out / "comparison.json").write_text(json.dumps(comparison, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    write_manifest(out)
    return code


# --- the summary path (no kkvim, no kkvim_reference) --------------------------------------------------------------------------------

def _at(rows, s, key):
    for row in rows:
        if row["s"] == s:
            return row.get(key)
    return None


def _svg(title, series, y_label, path, log=False):
    """A compact line plot: series = [(label, [(s, value)], dashed)]; unavailable values are left out."""
    width, height, left, right, top, bottom = 640, 380, 70, 170, 30, 40
    points = [(s, v) for _, data, _ in series for s, v in data if v is not None and (not log or v > 0)]
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" font-family="sans-serif" font-size="11">',
             '<rect width="100%" height="100%" fill="white"/>', f'<text x="{left}" y="18" font-size="13">{title}</text>']
    if not points:
        lines.append(f'<text x="{left}" y="{height / 2}">no data</text></svg>')
        path.write_text("\n".join(lines), encoding="utf-8")
        return
    tf = (lambda v: math.log10(v)) if log else (lambda v: v)
    xs = [s for s, _ in points]
    ys = [tf(v) for _, v in points]
    x0, x1 = min(xs), max(max(xs), min(xs) + 1)
    y0, y1 = min(ys), max(ys)
    if y1 == y0:
        y0, y1 = y0 - 1 if log else y0 - abs(y0) * 1e-3 - 1e-30, y1 + 1 if log else y1 + abs(y1) * 1e-3 + 1e-30
    pw, ph = width - left - right, height - top - bottom

    def px(s):
        return left + (s - x0) / (x1 - x0) * pw

    def py(v):
        return top + (1 - (tf(v) - y0) / (y1 - y0)) * ph

    lines.append(f'<rect x="{left}" y="{top}" width="{pw}" height="{ph}" fill="none" stroke="#888"/>')
    for k in range(5):
        value = y0 + (y1 - y0) * k / 4
        y = top + (1 - k / 4) * ph
        text = f"1e{value:.1f}" if log else f"{value:.6g}"
        lines.append(f'<text x="{left - 4}" y="{y + 4}" text-anchor="end">{text}</text>')
    for s in sorted(set(xs)):
        if s in (1, 2) or s % 4 == 0:
            lines.append(f'<text x="{px(s)}" y="{top + ph + 14}" text-anchor="middle">{s}</text>')
    lines.append(f'<text x="{left + pw / 2}" y="{height - 6}" text-anchor="middle">half-step s</text>')
    lines.append(f'<text x="14" y="{top + ph / 2}" transform="rotate(-90 14 {top + ph / 2})" text-anchor="middle">{y_label}</text>')
    colours = ["#1f6fb4", "#d1661c", "#2a9a3a", "#9b3fb0", "#b0302e", "#555555"]
    for i, (label, data, dashed) in enumerate(series):
        data = [(s, v) for s, v in data if v is not None and (not log or v > 0)]
        if not data:
            continue
        colour = colours[i % len(colours)]
        dash = ' stroke-dasharray="5,3"' if dashed else ""
        coords = " ".join(f"{px(s):.1f},{py(v):.1f}" for s, v in data)
        lines.append(f'<polyline fill="none" stroke="{colour}" stroke-width="1.5"{dash} points="{coords}"/>')
        for s, v in data:
            lines.append(f'<circle cx="{px(s):.1f}" cy="{py(v):.1f}" r="2" fill="{colour}"/>')
        ly = top + 14 * i + 8
        lines.append(f'<line x1="{left + pw + 10}" y1="{ly}" x2="{left + pw + 30}" y2="{ly}" stroke="{colour}" stroke-width="2"{dash}/>')
        lines.append(f'<text x="{left + pw + 34}" y="{ly + 4}">{label}</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def command_summary(args) -> int:
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        print(f"refused: {out} exists and is not empty (no overwrite)", file=sys.stderr)
        return 2
    out.mkdir(parents=True, exist_ok=True)
    runs = []
    for folder in map(Path, args.runs):
        result = json.loads((folder / "result.json").read_text(encoding="utf-8"))
        comparison = json.loads((folder / "comparison.json").read_text(encoding="utf-8")) if (folder / "comparison.json").exists() else None
        manifest = json.loads((folder / "artifacts.sha256.json").read_text(encoding="utf-8")) if (folder / "artifacts.sha256.json").exists() else {}
        runs.append((folder, result, comparison, manifest))
    table = []
    for folder, result, comparison, manifest in runs:
        history = result["history"]
        crow = comparison["rows"] if comparison else []
        done = [h for h in history if h["s"] <= (result["last_completed_s"] or 0)]  # the stopped row, if any, is s = last + 1
        last = done[-1] if done else None
        completion = {RESEARCH_LIMIT_REACHED: "fixed-budget completion", STOPPED_BITWISE_REPEATED_STATE: "repeated-state observation",
                      PILOT_STOPPED: "protective stop"}[result["status"]]
        entry = {"N": result["input"]["N"], "n_x": result["input"]["n_x"], "n_y": result["input"]["n_y"],
                 "half_steps_requested": result["half_steps_requested"], "half_steps_completed": result["half_steps_completed"],
                 "status": result["status"], "completion_kind": completion, "reason": result["reason"],
                 "last_completed_s": result["last_completed_s"]}
        for s in CHECKPOINTS:
            entry[f"w_centre_s{s}"] = _at(history, s, "w_centre")
            entry[f"rel_max_vs_D2_s{s}"] = _at(crow, s, "rel_max_diff_vs_D2")
            entry[f"rel_l2_vs_D2_s{s}"] = _at(crow, s, "rel_l2_diff_vs_D2")
            entry[f"rel_energy_vs_D2_s{s}"] = _at(crow, s, "rel_energy_diff_vs_D2")
            entry[f"P_hat_minus_Pi_D2_s{s}"] = _at(crow, s, "P_hat_minus_Pi_D2")
        entry.update({"last_w_centre": last.get("w_centre") if last else None, "last_alpha_centre": last.get("alpha_centre") if last else None,
                      "last_P_hat": last.get("P_hat") if last else None,
                      "last_rel_max_vs_D2": _at(crow, entry["last_completed_s"], "rel_max_diff_vs_D2"),
                      "last_rel_energy_vs_D2": _at(crow, entry["last_completed_s"], "rel_energy_diff_vs_D2"),
                      "last_V_exact_rank": last.get("V_exact_rank") if last else None,
                      "min_V_exact_rank": min((h["V_exact_rank"] for h in done if h.get("V_exact_rank") is not None), default=None),
                      "max_rho_N": max((h["rho_N"] for h in done if h.get("rho_N") is not None), default=None),
                      "max_kappa_rep": max((h["kappa_rep"] for h in done if h.get("kappa_rep") is not None), default=None),
                      "max_cond_S": max((h["cond_S"] for h in done if h.get("cond_S") is not None), default=None),
                      "max_cond_A_after_QR": max((h["cond_A_after_QR"] for h in done if h.get("cond_A_after_QR") is not None), default=None),
                      "max_eta": max((h["eta"] for h in done if h.get("eta") is not None), default=None),
                      "rn5_s2": _at(history, 2, "rn5_verdict"), "rt7_s2": _at(history, 2, "rt7_passed"),
                      "D_bitwise_equal": comparison["reference"]["D_bitwise_equal"] if comparison else None,
                      "D2_alpha_centre": comparison["reference"]["alpha_centre"] if comparison else None,
                      "D2_w_centre": comparison["reference"]["w_centre"] if comparison else None,
                      "D2_residual_max_dd": comparison["reference"]["residual_max_dd"] if comparison else None,
                      "D2_e_last_correction": comparison["reference"]["e_D2_last_correction"] if comparison else None,
                      "run_elapsed_s": result.get("elapsed_s"), "compare_elapsed_s": comparison.get("elapsed_s") if comparison else None,
                      "git_commit": result["provenance"]["git_commit"], "folder": str(folder), "artifacts_sha256": manifest})
        table.append(entry)
    columns = [k for k in table[0] if k != "artifacts_sha256"] if table else ["N"]
    with open(out / "summary.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(table)
    (out / "summary.json").write_text(json.dumps({"task": TASK, "marking": MARKING, "authorization": AUTHORIZATION,
                                                  "convergence": "NOT ASSESSED (no CONVERGED status)",
                                                  "unavailable": "None = not available (the run did not reach that s)",
                                                  "runs": table}, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    grids = sorted({(e["n_x"], e["n_y"]) for e in table})
    for grid in grids:
        chosen = [(f, r, c) for f, r, c, _ in runs if (r["input"]["n_x"], r["input"]["n_y"]) == grid]
        tag = f"{grid[0]}x{grid[1]}"

        def label(r):
            kind = {RESEARCH_LIMIT_REACHED: "budget", STOPPED_BITWISE_REPEATED_STATE: "repeated", PILOT_STOPPED: "STOP"}[r["status"]]
            return f"N={r['input']['N']} ({kind} s={r['last_completed_s']})"

        _svg(f"CCCC {tag}: centre deflection (RESEARCH_ONLY)", [(label(r), [(h["s"], h.get("w_centre")) for h in r["history"]],
             r["status"] == PILOT_STOPPED) for _, r, _ in chosen] + [("D2", [(1, c["reference"]["w_centre"]), (BUDGET_MAX, c["reference"]["w_centre"])],
             True) for _, _, c in chosen[:1] if c], "w_centre, m", out / f"w_centre_{tag}.svg")
        _svg(f"CCCC {tag}: discrete energy P_hat (RESEARCH_ONLY)", [(label(r), [(h["s"], h.get("P_hat")) for h in r["history"]],
             r["status"] == PILOT_STOPPED) for _, r, _ in chosen] + [("D2 Pi_h", [(1, c["reference"]["Pi_h"]), (BUDGET_MAX, c["reference"]["Pi_h"])],
             True) for _, _, c in chosen[:1] if c], "P_hat, J", out / f"energy_{tag}.svg")
        _svg(f"CCCC {tag}: rel. max difference vs D2 (OBSERVATION)", [(label(r), [(row["s"], row["rel_max_diff_vs_D2"]) for row in (c or {"rows": []})["rows"]],
             r["status"] == PILOT_STOPPED) for _, r, c in chosen], "max|w - w_D2| / max|w_D2| (log)", out / f"rel_diff_D2_{tag}.svg", log=True)
    print(f"summary of {len(table)} runs -> {out}")
    for e in table:
        print(f"  N={e['N']} {e['n_x']}x{e['n_y']}: {e['status']} ({e['completion_kind']}), completed {e['half_steps_completed']}/"
              f"{e['half_steps_requested']}, last w_centre {e['last_w_centre']}, rel_max vs D2 {e['last_rel_max_vs_D2']}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--N", type=int, required=True, choices=N_VALUES)
    run.add_argument("--nx", type=int, required=True)
    run.add_argument("--ny", type=int, required=True)
    run.add_argument("--half-steps", type=int, default=8, choices=range(1, BUDGET_MAX + 1), metavar="1..32")
    run.add_argument("--a", type=float, default=1.0)
    run.add_argument("--b", type=float, default=1.0)
    run.add_argument("--t", type=float, default=0.01)
    run.add_argument("--E", type=float, default=2.1e11)
    run.add_argument("--nu", type=float, default=0.3)
    run.add_argument("--q", type=float, default=1000.0)
    run.add_argument("--out", required=True)
    run.add_argument("--no-compare", action="store_true")
    compare = sub.add_parser("compare")
    compare.add_argument("--out", required=True)
    summary = sub.add_parser("summary")
    summary.add_argument("--out", required=True)
    summary.add_argument("runs", nargs="+")
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] not in ("run", "compare", "summary", "-h", "--help"):
        argv = ["run", *argv]
    args = parser.parse_args(argv)
    return {"run": command_run, "compare": command_compare, "summary": command_summary}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
