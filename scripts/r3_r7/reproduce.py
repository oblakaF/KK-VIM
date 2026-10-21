"""Reproduce the R-7 study (Q-029) from the versioned original scripts in a temporary directory.

New driver added for reproducibility; the original scripts are NOT modified. They expect to be run from a
directory containing scratch/r7/ (their original location), so this driver:

1. checks the SHA256 of the original files against docs/results/r3/r7/metadata.json;
2. copies them byte for byte into <workdir>/scratch/r7/;
3. runs, with the current interpreter and cwd = <workdir>, in the original order:
   calibrate.py, margins.py, summarize.py, extra.py, selfcheck.py, export.py (logs in <workdir>/logs/);
4. compares <workdir>/docs/results/r3/r7/{calibration_r1.csv, margins.csv, metadata.json} with the committed files:
   byte equality first, otherwise a numeric comparison and equality of all text columns (categories, flags);
5. checks the control counts (2178 grid pairs, 6534 rows) and writes <workdir>/reproduction_report.json.

Usage (repository root):  uv run --no-sync python scripts/r3_r7/reproduce.py [--workdir DIR]
"""

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS = ROOT / "docs" / "results" / "r3" / "r7"
ORIGINALS = ["surrogate.py", "calibrate.py", "margins.py", "summarize.py", "extra.py", "selfcheck.py", "export.py"]
RULES = "RULES_fixed_before_R3.md"
ORDER = ["calibrate.py", "margins.py", "summarize.py", "extra.py", "selfcheck.py", "export.py"]
VOLATILE_METADATA = {"base_commit", "date_utc", "python", "numpy", "platform"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_hashes(meta):
    result = {}
    for name in ORIGINALS:
        result[name] = sha(HERE / name) == meta["scripts_sha256"][name]
    result[RULES] = sha(HERE / RULES) == meta["rules_sha256"]
    return result


def compare_csv(new, old):
    if new.read_bytes() == old.read_bytes():
        return {"bytes_equal": True}
    a, b = list(csv.reader(new.open())), list(csv.reader(old.open()))
    out = {"bytes_equal": False, "rows_new": len(a), "rows_old": len(b), "header_equal": a[0] == b[0]}
    if len(a) != len(b) or a[0] != b[0]:
        return out
    max_rel, text_diff = {}, 0
    for ra, rb in zip(a[1:], b[1:]):
        for col, x, y in zip(a[0], ra, rb):
            try:
                fx, fy = float(x), float(y)
            except ValueError:
                text_diff += x != y
                continue
            rel = abs(fx - fy) / max(abs(fy), 1e-300) if fx != fy else 0.0
            max_rel[col] = max(max_rel.get(col, 0.0), rel)
    out.update({"text_cells_different": text_diff, "max_relative_difference": {k: v for k, v in max_rel.items() if v > 0}})
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", default=None)
    args = parser.parse_args()
    work = Path(args.workdir) if args.workdir else Path(tempfile.mkdtemp(prefix="kkvim_r7_"))
    meta = json.loads((RESULTS / "metadata.json").read_text(encoding="utf-8"))
    report = {"workdir": str(work), "hashes_match": check_hashes(meta)}
    if not all(report["hashes_match"].values()):
        print(json.dumps(report, indent=2))
        sys.exit("SHA256 of the versioned scripts differ from metadata.json; stop")

    target = work / "scratch" / "r7"
    target.mkdir(parents=True, exist_ok=True)
    for name in ORIGINALS + [RULES]:
        shutil.copyfile(HERE / name, target / name)
    logs = work / "logs"
    logs.mkdir(exist_ok=True)
    report["runs"] = {}
    for name in ORDER:
        done = subprocess.run([sys.executable, f"scratch/r7/{name}"], cwd=work, capture_output=True, text=True)
        (logs / f"{name}.txt").write_text(done.stdout + done.stderr, encoding="utf-8")
        report["runs"][name] = done.returncode
        print(f"{name}: exit {done.returncode}", flush=True)
        if done.returncode != 0:
            print(done.stderr[-2000:])
            sys.exit(f"{name} failed")

    out = work / "docs" / "results" / "r3" / "r7"
    report["calibration_r1.csv"] = compare_csv(out / "calibration_r1.csv", RESULTS / "calibration_r1.csv")
    report["margins.csv"] = compare_csv(out / "margins.csv", RESULTS / "margins.csv")
    new_meta = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
    report["metadata_differences"] = {
        k: {"new": new_meta.get(k), "committed": meta.get(k)}
        for k in sorted(set(new_meta) | (set(meta) - {"versioned_scripts", "reproduction"}))
        if new_meta.get(k) != meta.get(k)
    }
    report["metadata_differences_non_volatile"] = sorted(set(report["metadata_differences"]) - VOLATILE_METADATA)
    rows = list(csv.DictReader((out / "margins.csv").open()))
    report["rows"] = len(rows)
    report["grid_pairs"] = len({(r["r"], r["family"], r["nx"], r["ny"]) for r in rows})
    report["categories"] = {c: sum(1 for r in rows if r["category"] == c) for c in sorted({r["category"] for r in rows})}
    (work / "reproduction_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
