"""CCCC, N = 1...6 with one physical input: the six existing research commands, each in its own process (task R5-CCCC-N16-BATCH-01).

    uv run --frozen python scripts/r5_cccc_n16_batch.py --a 1 --b 1 --t 0.01 --E 2.1e11 --nu 0.3 --q 1000 --nx 16 --ny 16 --out scratch/n16

An engineering command (level В), not a solver: it computes no quantity of the method. For every N it starts the existing command of
that N with the same input and exactly two requested half-steps, as a separate process with a folder of its own:

    N = 1        python -m kkvim run-cccc-n1                 (D-061)
    N = 2, 3     python -m kkvim run-cccc-n23 --N 2 | 3      (D-063, D-065: s = 2 only after RN-5 and RT-7, n_x, n_y <= 32)
    N = 4, 5, 6  python -m kkvim run-cccc-n46 --N 4 | 5 | 6  (D-067, D-068: the same s = 2 guards)

The commands validate the input, run the half-steps with their own guards and save result.json, history.csv, factors.csv and the field
file of their status (field_research.csv or field_diagnostic.csv). Their files are kept unchanged in <out>/N1 ... <out>/N6; their stdout and
stderr go to <out>/logs. Then <out>/summary.json and <out>/summary.csv copy, per N, what each result.json states (route, half-steps
requested and completed, status and reason, field role and file, alpha_centre, max|w| with its unit, decisions, limits) and the SHA256 of
every saved file. Nothing is recomputed, interpolated or taken from another N; a value the command did not write stays empty.

Every result is RESEARCH_ONLY / PROVISIONAL / NOT_QUALIFIED. A protective stop is a stop, not a completed result: its status, reason and
diagnostic field are reported as the command wrote them. Agreement or difference between the N values is an observation: it is not
evidence of convergence, accuracy or correctness. CCCC s >= 3 for N >= 2 is not requested (D-065.6, D-066.3). D1_START = BLOCKED.

<out> must not exist or must be an empty folder: earlier calculations are never overwritten.

Exit codes: 0 - all six commands did the two half-steps; 6 - at least one protective stop, no failure; 1 - at least one command failed
(no result.json, an ERROR_* status, a refused input, an inconsistent result.json); 2 - wrong arguments or a populated <out>.
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
MARKING = "RESEARCH_ONLY / PROVISIONAL / NOT_QUALIFIED"
HALF_STEPS = 2  # fixed by the task: s = 1, 2; s >= 3 is not executable for N >= 2 in the production commands (D-065.6, D-066.3)
N_VALUES = (1, 2, 3, 4, 5, 6)
PHYSICAL = ("a", "b", "t", "E", "nu", "q", "nx", "ny")
DONE = "COMPLETED_FIXED_HALF_STEPS"
EXIT_DONE, EXIT_COMMAND_FAILED, EXIT_USAGE, EXIT_PROTECTIVE_STOP = 0, 1, 2, 6
COMMAND_EXIT_MEANING = {0: "the half-steps asked for done (not convergence)", 2: "outside the domain, nothing computed", 3: "ERROR_*",
                        4: "input or usage error", 5: "output error", 6: "guard or protective stop", 130: "interrupted by the user"}
NOTE = ("Each N is a separate run of its own command with its own guards. Agreement or difference between N values is an observation, "
        "not evidence of convergence, accuracy or correctness. No CONVERGED status exists for these runs. D1_START = BLOCKED.")
CSV_COLUMNS = ("N", "route", "command", "exit_code", "half_steps_requested", "half_steps_completed", "status", "reason",
               "application_status", "field_role", "field_file", "field_half_step", "alpha_centre", "w_max_abs", "w_max_unit",
               "w_max_x", "w_max_y", "decision", "result_json_sha256", "field_sha256", "folder", "problem")


def route_of(N: int) -> list[str]:
    """The sub-command (and --N) of the existing command of N."""
    if N == 1:
        return ["run-cccc-n1"]
    if N in (2, 3):
        return ["run-cccc-n23", "--N", str(N)]
    if N in (4, 5, 6):
        return ["run-cccc-n46", "--N", str(N)]
    raise ValueError(f"N must be 1...6, got {N!r}")


def command_of(N: int, physical: dict, folder: Path) -> list[str]:
    """argv after ``python -m kkvim``: the same texts of the input for every N, exactly two half-steps."""
    argv = route_of(N)
    for name in PHYSICAL:
        argv += [f"--{name}", physical[name]]
    return [*argv, "--half-steps", str(HALF_STEPS), "--out", str(folder)]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _empty_or_missing(path: Path) -> bool:
    return not path.exists() or (path.is_dir() and not any(path.iterdir()))


def _consistency(N: int, physical: dict, document: dict) -> list[str]:
    """Where result.json does not describe the run that was asked for (wrong route, other input); empty if it does."""
    problems = []
    if (document.get("model") or {}).get("N") != N:
        problems.append(f"model.N = {(document.get('model') or {}).get('N')!r} != {N}")
    if (document.get("model") or {}).get("boundary") != "CCCC":
        problems.append(f"model.boundary = {(document.get('model') or {}).get('boundary')!r}")
    given = document.get("input") or {}
    for name, key in (("a", "a"), ("b", "b"), ("t", "t"), ("E", "E"), ("nu", "nu"), ("q", "q")):
        value = given.get(key, {}).get("value") if isinstance(given.get(key), dict) else given.get(key)
        if value != float(physical[name]):
            problems.append(f"input.{key} = {value!r} != {physical[name]}")
    for name, key in (("nx", "n_x"), ("ny", "n_y")):
        if given.get(key) != int(physical[name]):
            problems.append(f"input.{key} = {given.get(key)!r} != {physical[name]}")
    if given.get("half_steps") != HALF_STEPS or document.get("half_steps_requested") != HALF_STEPS:
        problems.append(f"half-steps requested {document.get('half_steps_requested')!r} != {HALF_STEPS}")
    if document.get("application_status") != MARKING:
        problems.append(f"application_status = {document.get('application_status')!r}")
    return problems


def run_one(N: int, physical: dict, out: Path) -> dict:
    """Run the command of N in its own process; the row of the summary from its result.json and the hashes of its files."""
    folder = out / f"N{N}"
    argv = command_of(N, physical, folder)
    done = subprocess.run([sys.executable, "-m", "kkvim", *argv], cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", check=False)
    logs = out / "logs"
    (logs / f"N{N}.stdout.txt").write_text(done.stdout, encoding="utf-8")
    (logs / f"N{N}.stderr.txt").write_text(done.stderr, encoding="utf-8")
    row = {"N": N, "route": argv[0], "command": "python -m kkvim " + " ".join(argv), "exit_code": done.returncode,
           "exit_meaning": COMMAND_EXIT_MEANING.get(done.returncode, "unknown exit code"), "folder": folder.name, "files": {},
           "problem": None}
    files = sorted(p for p in folder.iterdir() if p.is_file()) if folder.is_dir() else []
    row["files"] = {p.name: sha256(p) for p in files}
    result = folder / "result.json"
    if not result.is_file():
        row["problem"] = f"no result.json (exit code {done.returncode}: {row['exit_meaning']}); stderr: {done.stderr.strip()[-400:]}"
        return row
    document = json.loads(result.read_text(encoding="utf-8"))
    field = document.get("field") or {}
    w_max = field.get("w_max") or {}
    row.update(schema=document.get("schema"), half_steps_requested=document.get("half_steps_requested"),
               half_steps_completed=document.get("half_steps_completed"), status=document.get("status"),
               status_note=document.get("status_note"), reason=document.get("reason"),
               application_status=document.get("application_status"), field_role=field.get("role"), field_file=field.get("file"),
               field_half_step=field.get("half_step"), alpha_centre=field.get("alpha_centre"), alpha_note=field.get("alpha_note"),
               alpha_definition=field.get("alpha_definition"), w_max_abs=w_max.get("value"),
               w_max_unit=(field.get("units") or {}).get("w"), w_max_x=w_max.get("x"), w_max_y=w_max.get("y"),
               decision=(document.get("model") or {}).get("decision"), rank_guard_rule=document.get("rank_guard_rule"),
               limits=document.get("limits"), provenance=document.get("provenance"))
    problems = _consistency(N, physical, document)
    if field.get("file") and field["file"] not in row["files"]:
        problems.append(f"the field file {field['file']} named in result.json is missing")
    if problems:
        row["problem"] = "inconsistent result.json: " + "; ".join(problems)
    return row


def outcome(rows: list[dict]) -> tuple[int, str]:
    """Exit code and wording of the batch from the six rows; a protective stop is never counted as a completed result."""
    failed = [r["N"] for r in rows if r["problem"] or r["exit_code"] not in (0, 6) or str(r.get("status", "")).startswith("ERROR")]
    stopped = [r["N"] for r in rows if r["N"] not in failed and (r["exit_code"] == 6 or r.get("status") != DONE)]
    if failed:
        return EXIT_COMMAND_FAILED, f"FAILED for N = {failed}" + (f"; protective stops for N = {stopped}" if stopped else "")
    if stopped:
        return EXIT_PROTECTIVE_STOP, f"protective stops for N = {stopped}; the other N did the {HALF_STEPS} half-steps"
    return EXIT_DONE, f"all six commands did the {HALF_STEPS} half-steps asked for (not convergence)"


def write_summary(out: Path, physical: dict, rows: list[dict], code: int, text: str, started: str) -> None:
    summary = {"task": "R5-CCCC-N16-BATCH-01", "marking": MARKING, "note": NOTE, "script": "scripts/r5_cccc_n16_batch.py",
               "argv": sys.argv, "started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "python": sys.version, "platform": platform.platform(), "input_as_given": physical,
               "half_steps_requested": HALF_STEPS, "routes": {N: route_of(N)[0] for N in N_VALUES},
               "batch_exit_code": code, "batch_outcome": text, "convergence": "NOT ASSESSED (no CONVERGED status)",
               "D1_START": "BLOCKED", "rows": rows}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    with open(out / "summary.csv", "w", newline="", encoding="utf-8") as handle:
        handle.write(f"# {MARKING}; {NOTE}\n")
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            flat = dict(row)
            flat["result_json_sha256"] = row["files"].get("result.json")
            flat["field_sha256"] = row["files"].get(row.get("field_file")) if row.get("field_file") else None
            writer.writerow({k: ("" if flat.get(k) is None else repr(flat[k]) if isinstance(flat.get(k), float) else flat[k])
                             for k in CSV_COLUMNS})


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in PHYSICAL:
        parser.add_argument(f"--{name}", required=True, help="passed unchanged to every command; the commands check it")
    parser.add_argument("--out", required=True, help="new or empty folder; a populated one is refused")
    args = parser.parse_args(argv)
    out = Path(args.out).resolve()  # the commands run with cwd = the repository root
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
        print(f"N = {N}: {row.get('status') or 'no result'} (exit {row['exit_code']}), s {row.get('half_steps_completed')}/"
              f"{HALF_STEPS}, field {row.get('field_role')}{' PROBLEM: ' + row['problem'] if row['problem'] else ''}", flush=True)
    code, text = outcome(rows)
    write_summary(out, physical, rows, code, text, started)
    print(f"\n{MARKING}\n{'N':>2} {'route':<13} {'status':<32} {'s':>3} {'field':<11} {'alpha_centre':>22} {'max|w|':>22}")
    for r in rows:
        alpha = "" if r.get("alpha_centre") is None else repr(r["alpha_centre"])
        w = "" if r.get("w_max_abs") is None else f"{r['w_max_abs']!r} {r.get('w_max_unit') or ''}"
        print(f"{r['N']:>2} {r['route']:<13} {str(r.get('status')):<32} {str(r.get('half_steps_completed')):>3} "
              f"{str(r.get('field_role')):<11} {alpha:>22} {w:>22}")
    print(f"{text}\n{NOTE}\nsummary: {out / 'summary.json'}, {out / 'summary.csv'}")
    return code


if __name__ == "__main__":
    sys.exit(main())
