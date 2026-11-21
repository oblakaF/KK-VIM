"""SSSS, N = 1...6 with one physical input: ``python -m kkvim run-ssss`` six times, each in its own process (task R4-SSSS-N16-BATCH-01).

    uv run --frozen python scripts/r4_ssss_n16_batch.py --a 1 --b 1 --t 0.01 --E 2.1e11 --nu 0.3 --q 1000 --nx 40 --ny 40 --out scratch/ssss16

An engineering command (level В), not a solver: it computes no quantity of the method. ``run-ssss`` routes every N itself and applies
its own domains and guards; this command only chooses, per N, the mode the task fixes and keeps the modes distinct:

    N = 1        run-ssss --N 1                                  the accepted R3 (D-043), its own stopping rule (NUMERICAL_SPEC 11.3);
                                                                 no --half-steps, --fixed-half-steps or --research-only
    N = 2, 3     run-ssss --N 2|3 --half-steps 8 --fixed-half-steps   research mode of D-048 with fixed half-steps; PROVISIONAL / NOT_QUALIFIED
    N = 4, 5, 6  run-ssss --N 4|5|6 --half-steps 8 --research-only    the study of D-056.7, only inside its domain; RESEARCH_ONLY

N = 1 stops by the criterion of R3 (CONVERGED is its own status); N >= 2 runs exactly the fixed half-steps and is never CONVERGED
(D-048.4, D-056.8): a CONVERGED status at N >= 2 is reported as an inconsistent result.

The files of each run are kept unchanged in <out>/N1 ... <out>/N6 (result.json, history.csv and field.csv, field_research.csv or
field_diagnostic.csv); stdout and stderr in <out>/logs. <out>/summary.json and <out>/summary.csv copy, per N, what its result.json states:
route and mode, input and grid, scientific applicability and mark, solver status, reason, half-steps requested and completed, field
file, role and availability, the centre coefficient w_bar_centre = w D / (q a^4), max|w| in metres, decisions, limits, provenance, and the
SHA256 of every saved file. Nothing is recomputed: a value the result does not state stays empty (R3 writes no max|w|; its field is in
field.csv). Agreement or difference between N values is an observation, not evidence of convergence, accuracy or correctness.

<out> must not exist or must be an empty folder. The summary is written after the sixth run; an interrupted batch leaves the folders
of the runs done so far and no summary.

Exit codes: 0 - every run ended as requested (N = 1 CONVERGED, N >= 2 the 8 half-steps); 6 - at least one stop by a rule or guard of its
route or a refusal outside the domain, no failure; 1 - a failure (no result.json, ERROR_*, an inconsistent result.json); 2 - wrong
arguments or a populated <out>.
"""

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
N_VALUES = (1, 2, 3, 4, 5, 6)
HALF_STEPS = 8  # N >= 2, fixed by the task; N = 1 has the stopping rule of R3 instead
PHYSICAL = ("a", "b", "t", "E", "nu", "q", "nx", "ny")
MODES = {1: "R3_ACCEPTED_SOLVER", 2: "R4_LIMITED_RESEARCH_MODE", 3: "R4_LIMITED_RESEARCH_MODE", 4: "RESEARCH_ONLY_N4_N6",
         5: "RESEARCH_ONLY_N4_N6", 6: "RESEARCH_ONLY_N4_N6"}
EXPECTED_END = {1: ("CONVERGED",), **{N: ("RESEARCH_LIMIT_REACHED",) for N in (2, 3, 4, 5, 6)}}
EXIT_DONE, EXIT_FAILED, EXIT_USAGE, EXIT_STOPPED = 0, 1, 2, 6
COMMAND_EXIT_MEANING = {0: "ended as requested (N = 1: CONVERGED; N >= 2: not convergence)", 1: "MAX_HALF_STEPS or STAGNATED",
                        2: "outside the domain or refused input, nothing computed", 3: "ERROR_*", 4: "output error",
                        5: "internal error", 6: "guard, energy or proven rank stop", 130: "interrupted by the user"}
NOTE = ("N = 1 is the accepted R3 with its own stopping rule; N = 2-3 the D-048 research mode with 8 fixed half-steps "
        "(PROVISIONAL / NOT_QUALIFIED); N = 4-6 the D-056.7 study, 8 fixed half-steps (RESEARCH_ONLY / PROVISIONAL / NOT_QUALIFIED). "
        "Agreement or difference between N values is an observation, not evidence of convergence, accuracy or correctness. "
        "A single run is not a scientific acceptance.")
CSV_COLUMNS = ("N", "mode", "route", "decisions", "command", "exit_code", "execution", "status", "reason", "half_steps_requested",
               "fixed_half_steps", "half_steps_completed", "accepted_solver", "research_only", "applicability_status", "mark",
               "field_file", "field_role", "field_half_step", "field_available", "w_bar_centre", "w_max_m", "w_max_x", "w_max_y",
               "n_x", "n_y", "git_commit", "git_worktree_modified", "result_json_sha256", "field_sha256", "folder", "problem")


def options_of(N: int) -> list[str]:
    """The mode options of N; N = 1 gets none (R3 stops by its own rule)."""
    if N == 1:
        return []
    if N in (2, 3):
        return ["--half-steps", str(HALF_STEPS), "--fixed-half-steps"]
    if N in (4, 5, 6):
        return ["--half-steps", str(HALF_STEPS), "--research-only"]
    raise ValueError(f"N must be 1...6, got {N!r}")


def command_of(N: int, physical: dict, folder: Path) -> list[str]:
    """argv after ``python -m kkvim``: the same texts of the input for every N."""
    argv = ["run-ssss", "--N", str(N)]
    for name in PHYSICAL:
        argv += [f"--{name}", physical[name]]
    return [*argv, *options_of(N), "--out", str(folder)]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _empty_or_missing(path: Path) -> bool:
    return not path.exists() or (path.is_dir() and not any(path.iterdir()))


def _consistency(N: int, physical: dict, document: dict) -> list[str]:
    """Where result.json does not describe the run that was asked for; empty if it does."""
    problems = []
    if document.get("N") != N:
        problems.append(f"N = {document.get('N')!r} != {N}")
    if document.get("boundary_conditions") != "SSSS":
        problems.append(f"boundary_conditions = {document.get('boundary_conditions')!r}")
    mode = (document.get("solver_path") or {}).get("mode")
    if document.get("execution") != "NOT_EXECUTED" and mode != MODES[N]:
        problems.append(f"solver_path.mode = {mode!r} != {MODES[N]}")
    given = document.get("input") or {}
    for name in ("a", "b", "t", "E", "nu", "q"):
        value = given.get(name, {}).get("value") if isinstance(given.get(name), dict) else given.get(name)
        if value != float(physical[name]):
            problems.append(f"input.{name} = {value!r} != {physical[name]}")
    for name, key in (("nx", "n_x"), ("ny", "n_y")):
        if given.get(key) != int(physical[name]):
            problems.append(f"input.{key} = {given.get(key)!r} != {physical[name]}")
    if N >= 2:
        if given.get("half_steps_requested") != HALF_STEPS or given.get("fixed_half_steps") is not True:
            problems.append(f"half-steps {given.get('half_steps_requested')!r}, fixed {given.get('fixed_half_steps')!r}")
        if document.get("status") == "CONVERGED" or document.get("converged"):
            problems.append("CONVERGED at N >= 2 (D-048.4, D-056.8)")
    return problems


def run_one(N: int, physical: dict, out: Path) -> dict:
    """Run ``run-ssss`` for N in its own process; the row of the summary from its result.json and the hashes of its files."""
    folder = out / f"N{N}"
    argv = command_of(N, physical, folder)
    done = subprocess.run([sys.executable, "-m", "kkvim", *argv], cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", check=False)
    (out / "logs" / f"N{N}.stdout.txt").write_text(done.stdout, encoding="utf-8")
    (out / "logs" / f"N{N}.stderr.txt").write_text(done.stderr, encoding="utf-8")
    row = {"N": N, "mode_requested": MODES[N], "options": options_of(N), "command": "python -m kkvim " + " ".join(argv),
           "exit_code": done.returncode, "exit_meaning": COMMAND_EXIT_MEANING.get(done.returncode, "unknown exit code"),
           "folder": folder.name, "files": {}, "problem": None}
    files = sorted(p for p in folder.iterdir() if p.is_file()) if folder.is_dir() else []
    row["files"] = {p.name: sha256(p) for p in files}
    result = folder / "result.json"
    if not result.is_file():
        row["problem"] = f"no result.json (exit code {done.returncode}: {row['exit_meaning']}); stderr: {done.stderr.strip()[-400:]}"
        return row
    document = json.loads(result.read_text(encoding="utf-8"))
    given, path = document.get("input") or {}, document.get("solver_path") or {}
    applicability, field = document.get("scientific_applicability") or {}, document.get("field") or {}
    w_max = field.get("w_max") or {}
    row.update(schema=document.get("schema"), mode=path.get("mode"), route=path.get("route"), decisions=path.get("decisions"),
               execution=document.get("execution"), status=document.get("status"), status_note=document.get("status_note"),
               reason=document.get("reason"), converged=document.get("converged"),
               half_steps_requested=given.get("half_steps_requested"), fixed_half_steps=given.get("fixed_half_steps"),
               stopping_parameters=document.get("stopping_parameters"), half_steps_completed=document.get("half_steps_completed"),
               n_x=given.get("n_x"), n_y=given.get("n_y"), input=given,
               accepted_solver=applicability.get("accepted_solver"), accepted_by=applicability.get("accepted_by"),
               research_only=applicability.get("research_only"), qualified=applicability.get("qualified"),
               applicability_status=applicability.get("status") or applicability.get("implementation"),
               mark=applicability.get("mark"), domain=applicability.get("domain"), limits=applicability.get("limits"),
               field_file=field.get("file"), field_role=field.get("role"), field_half_step=field.get("half_step"),
               field_available=bool(field.get("file")) and field.get("file") in row["files"], field_note=field.get("note"),
               w_bar_centre=field.get("w_bar_centre"), w_bar_definition=field.get("w_bar_definition"),
               w_max_m=w_max.get("value"), w_max_x=w_max.get("x"), w_max_y=w_max.get("y"),
               w_max_note=None if w_max else "not stated by this result.json (not recomputed here)",
               provenance=document.get("provenance"))
    problems = _consistency(N, physical, document)
    if field.get("file") and not row["field_available"]:
        problems.append(f"the field file {field['file']} named in result.json is missing")
    if w_max and w_max.get("unit") not in (None, "m"):
        problems.append(f"field.w_max.unit = {w_max.get('unit')!r}")
    if problems:
        row["problem"] = "inconsistent result.json: " + "; ".join(problems)
    return row


def outcome(rows: list[dict]) -> tuple[int, str]:
    """Exit code and wording of the batch; a stop or a refusal is never counted as a run that ended as requested."""
    failed = [r["N"] for r in rows if r["problem"] or r["exit_code"] not in (0, 1, 2, 6) or str(r.get("status", "")).startswith("ERROR")]
    ended = [r["N"] for r in rows if r["N"] not in failed and r["exit_code"] == 0 and r.get("status") in EXPECTED_END[r["N"]]]
    other = [r["N"] for r in rows if r["N"] not in failed and r["N"] not in ended]
    if failed:
        return EXIT_FAILED, f"FAILED for N = {failed}" + (f"; stopped or refused for N = {other}" if other else "")
    if other:
        return EXIT_STOPPED, f"stopped or refused for N = {other}; ended as requested for N = {ended}"
    return EXIT_DONE, "every run ended as requested (N = 1 by the R3 rule; N >= 2 the 8 fixed half-steps, not convergence)"


def write_summary(out: Path, physical: dict, rows: list[dict], code: int, text: str, started: str) -> None:
    summary = {"task": "R4-SSSS-N16-BATCH-01", "note": NOTE, "script": "scripts/r4_ssss_n16_batch.py", "argv": sys.argv,
               "started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "python": sys.version, "platform": platform.platform(), "input_as_given": physical,
               "modes": {N: {"mode": MODES[N], "options": options_of(N)} for N in N_VALUES},
               "batch_exit_code": code, "batch_outcome": text, "rows": rows}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    with open(out / "summary.csv", "w", newline="", encoding="utf-8") as handle:
        handle.write(f"# {NOTE}\n")
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            flat = dict(row)
            flat["decisions"] = "; ".join(row.get("decisions") or [])
            flat["git_commit"] = (row.get("provenance") or {}).get("git_commit")
            flat["git_worktree_modified"] = (row.get("provenance") or {}).get("git_worktree_modified")
            flat["result_json_sha256"] = row["files"].get("result.json")
            flat["field_sha256"] = row["files"].get(row.get("field_file")) if row.get("field_file") else None
            writer.writerow({k: ("" if flat.get(k) is None else repr(flat[k]) if isinstance(flat.get(k), float) else flat[k])
                             for k in CSV_COLUMNS})


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):  # the texts of the results are Russian; a Windows pipe defaults to cp1252
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in PHYSICAL:
        parser.add_argument(f"--{name}", required=True, help="passed unchanged to every run; run-ssss checks it")
    parser.add_argument("--out", required=True, help="new or empty folder; a populated one is refused")
    args = parser.parse_args(argv)
    out = Path(args.out).resolve()  # the runs have cwd = the repository root
    if not _empty_or_missing(out):
        print(f"ERROR: {out} exists and is not an empty folder; earlier calculations are not overwritten. Choose a new --out.",
              file=sys.stderr)
        return EXIT_USAGE
    physical = {name: getattr(args, name) for name in PHYSICAL}
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    (out / "logs").mkdir(parents=True, exist_ok=True)
    rows = []
    for N in N_VALUES:
        row = run_one(N, physical, out)
        rows.append(row)
        print(f"N = {N}: {row.get('status') or 'no result'} (exit {row['exit_code']}), s {row.get('half_steps_completed')}, "
              f"field {row.get('field_role')}{' PROBLEM: ' + row['problem'] if row['problem'] else ''}", flush=True)
    code, text = outcome(rows)
    write_summary(out, physical, rows, code, text, started)
    print(f"\n{'N':>2} {'mode':<25} {'status':<24} {'s':>3} {'field':<10} {'w_bar_centre':>22} {'max|w|, m':>23}  mark")
    for r in rows:
        centre = "" if r.get("w_bar_centre") is None else repr(r["w_bar_centre"])
        w = "not stated" if r.get("w_max_m") is None else repr(r["w_max_m"])
        if r.get("accepted_solver"):
            mark = f"accepted solver ({r.get('accepted_by')}); a single run is not an acceptance"
        else:
            mark = "; ".join(str(x) for x in (r.get("applicability_status"), r.get("mark")) if x)
        print(f"{r['N']:>2} {str(r.get('mode')):<25} {str(r.get('status')):<24} {str(r.get('half_steps_completed')):>3} "
              f"{str(r.get('field_role')):<10} {centre:>22} {w:>23}  {mark}")
    print(f"{text}\n{NOTE}\nsummary: {out / 'summary.json'}, {out / 'summary.csv'}")
    return code


if __name__ == "__main__":
    sys.exit(main())
