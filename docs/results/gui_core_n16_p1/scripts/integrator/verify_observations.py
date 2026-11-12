"""GUI-CORE-N16-P1-INTEGRATION-PREP (Integrator): reproducible check of this evidence package, using only its own files.

It does not run the GUI or a solver. It checks:
  1. the package integrity: MANIFEST.sha256, REPACKAGING.csv against MANIFEST.qa_original.sha256, every *.json parses;
  2. that gui_check.json keeps the QA verdict_checks = FAIL and that exactly three case checks are not true, all explained by
     observations N1 and N2 of QA_REPORT.md (section 6);
  3. N1: R4 field files GUI vs CLI - every column bitwise equal except w_bar [1], which differs by at most 1 ulp;
  4. N2: SSSS N = 4 history - the GUI column "w_max [1]" is the dimensionless max|w_bar| of the studied cell (equal to the CLI
     column w_bar_max), and multiplied by w_scale = q a^4 / D it equals the CLI column w_max [m] bitwise;
  5. N3: R4 normative (D-047) - NOT_QUALIFIED after s = 1, recorded as half_steps {completed 1, requested 100} and as
     "s = 1 из запрошенных 100" in report.html; 100 is the half-step budget, not an instruction to run 100 half-steps.
Usage: python -I -X utf8 verify_observations.py [<package_dir>]   (default: the package that contains this script)
Writes <package_dir>/results/integrator/verify_observations.json (deterministic: no time stamps) and exits 0 only if all checks hold.
"""
import csv
import hashlib
import io
import json
import struct
import sys
from pathlib import Path

PKG = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parents[2]
OUT = PKG / "results" / "integrator" / "verify_observations.json"
CHECK = PKG / "results" / "check"
report, problems = {}, []


def expect(name, ok, detail=None):
    report[name] = {"ok": bool(ok), "detail": detail}
    if not ok:
        problems.append(name)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table(path):
    lines = [line for line in path.read_text(encoding="utf-8").splitlines(True) if line.strip() and not line.startswith("#")]
    return list(csv.DictReader(io.StringIO("".join(lines))))


def ulps(a, b):
    ia, ib = (struct.unpack("<q", struct.pack("<d", float(v)))[0] for v in (a, b))
    return abs(ia - ib)


# 1. integrity
manifest = {}
for line in (PKG / "MANIFEST.sha256").read_text(encoding="utf-8").splitlines():
    digest, name = line.split(maxsplit=1)
    manifest[name.lstrip("*")] = digest
manifest.pop(OUT.relative_to(PKG).as_posix(), None)  # the output of this script itself; checked by sha256sum -c MANIFEST.sha256
bad =sorted(n for n, d in manifest.items() if not (PKG / n).is_file() or sha(PKG / n) != d)
listed = {p.relative_to(PKG).as_posix() for p in PKG.rglob("*") if p.is_file()}
unlisted = sorted(listed - set(manifest) - {"MANIFEST.sha256", OUT.relative_to(PKG).as_posix()})
expect("manifest_sha256_all_match", not bad and not unlisted, {"entries": len(manifest), "mismatch": bad, "unlisted": unlisted})

original = {}
for line in (PKG / "MANIFEST.qa_original.sha256").read_text(encoding="utf-8").splitlines():
    digest, name = line.split(maxsplit=1)
    original[name.lstrip("*").removeprefix("./")] = digest
rows = list(csv.DictReader(io.StringIO((PKG / "REPACKAGING.csv").read_text(encoding="utf-8"))))
rp_bad = []
for r in rows:
    in_original = r["qa_path"] == "MANIFEST.sha256" or original.get(r["qa_path"]) == r["qa_sha256"]
    committed_ok = sha(PKG / r["committed_path"]) == r["committed_sha256"]
    unchanged_ok = r["transformations"] != "none" or r["qa_sha256"] == r["committed_sha256"]
    if not (in_original and committed_ok and unchanged_ok):
        rp_bad.append(r["qa_path"])
expect("repackaging_consistent_with_qa_manifest", not rp_bad,
       {"rows": len(rows), "qa_original_entries": len(original), "problems": rp_bad,
        "transformations": sorted({r["transformations"] for r in rows})})

json_bad = []
for p in sorted(PKG.rglob("*.json")):
    if p == OUT:
        continue
    try:
        json.loads(p.read_text(encoding="utf-8"))
    except ValueError as exc:
        json_bad.append(f"{p.relative_to(PKG).as_posix()}: {exc}")
expect("all_json_parse", not json_bad, json_bad)

# 2. the machine-readable FAIL of gui_check.json is kept and fully explained
g = json.loads((CHECK / "gui_check.json").read_text(encoding="utf-8"))
not_true = sorted(f"{case}:{name}" for case, v in g["cases"].items() for name, c in v.get("checks", {}).items()
                  if not (c is True or (isinstance(c, dict) and c.get("ok") is True)))
total = sum(len(v.get("checks", {})) for v in g["cases"].values())
EXPECTED = ["S3_R4_N2_research:field_csv_data_bitwise_vs_cli", "S4_R4_N3_fixed4:field_csv_data_bitwise_vs_cli",
            "S6_SSSS_N4_research:history_common_columns_text_equal"]
expect("gui_check_verdict_checks_is_FAIL_as_published", g["verdict_checks"] == "FAIL", g["verdict_checks"])
expect("gui_check_not_true_checks_are_exactly_N1_N2", not_true == EXPECTED and len(g["failures"]) == 3,
       {"case_checks": total, "not_true": not_true, "failures_entries": len(g["failures"])})
roles = {c: (v["outcome"].get("status"), v["outcome"].get("field_role")) for c, v in g["cases"].items() if "outcome" in v}
expect("field_roles_accepted_only_for_R3_CONVERGED",
       all((role == "accepted") == (status == "CONVERGED") for status, role in roles.values())
       and all(role == "diagnostic" for status, role in roles.values() if status in ("MAX_HALF_STEPS", "STOPPED_RANK_GUARD_UNCERTIFIED")),
       {c: list(v) for c, v in roles.items()})
qt = json.loads((PKG / "results" / "qt" / "qt_smoke.json").read_text(encoding="utf-8"))
expect("qt_smoke_verdict_PASS", qt.get("verdict") == "PASS" and qt.get("failures") == [], qt.get("verdict"))

# 3. N1
for case in ("S3_R4_N2_research", "S4_R4_N3_fixed4"):
    cli, gui = table(CHECK / "cli" / case / "field_research.csv"), table(CHECK / "export" / case / "field_research.csv")
    cols = list(cli[0])
    per = {c: [ulps(a[c], b[c]) for a, b in zip(cli, gui)] for c in cols}
    detail = {"nodes": len(cli), "columns": cols,
              "differing_nodes": {c: sum(u > 0 for u in per[c]) for c in cols}, "max_ulp": {c: max(per[c]) for c in cols}}
    expect(f"N1_{case}_only_w_bar_differs_by_at_most_1_ulp",
           len(cli) == len(gui) and cols == list(gui[0])
           and all(detail["max_ulp"][c] == 0 for c in cols if c != "w_bar [1]") and detail["max_ulp"]["w_bar [1]"] <= 1,
           detail)

# 4. N2
case = "S6_SSSS_N4_research"
cli, gui = table(CHECK / "cli" / case / "history.csv"), table(CHECK / "export" / case / "history.csv")
result = json.loads((CHECK / "cli" / case / "result.json").read_text(encoding="utf-8"))
gm = json.loads((CHECK / "export" / case / "manifest.json").read_text(encoding="utf-8"))
w_scale = result["w_scale_q_a4_over_D"]["value"]
column = [c for c in gm["history_columns"] if c["key"] == "w_max"]
rows2 = [{"s": a["s"], "cli_w_max_m": a["w_max"], "cli_w_bar_max": a["w_bar_max"], "gui_w_max_[1]": b["w_max [1]"],
          "gui_equals_cli_w_bar_max": b["w_max [1]"] == a["w_bar_max"],
          "ulp(gui*w_scale, cli_w_max)": ulps(float(b["w_max [1]"]) * w_scale, a["w_max"])} for a, b in zip(cli, gui)]
expect("N2_gui_w_max_is_dimensionless_cell_value",
       len(cli) == len(gui) and all(r["gui_equals_cli_w_bar_max"] and r["ulp(gui*w_scale, cli_w_max)"] == 0 for r in rows2)
       and column == [{"key": "w_max", "title": "max|w̄| (ячейка)", "kind": "number", "unit": "1"}]
       and gm["solver"]["parameters"]["w_scale_q_a4_over_D"] == w_scale,
       {"w_scale_q_a4_over_D_m": w_scale, "gui_manifest_column": column, "rows": rows2})

# 5. N3
m5 = json.loads((CHECK / "export" / "S5_R4_N2_normative" / "manifest.json").read_text(encoding="utf-8"))
html = (CHECK / "export" / "S5_R4_N2_normative" / "report.html").read_text(encoding="utf-8")
n3 = {"r4_mode": m5["input"]["method"]["r4_mode"], "solver_status": m5["statuses"]["solver_status"],
      "half_steps": m5["results"]["half_steps"], "requested_run_completed": m5["statuses"]["requested_run_completed"],
      "field_role": m5["statuses"]["field_role"], "report_phrase_present": "s = 1 из запрошенных 100" in html}
expect("N3_normative_R4_budget_wording_as_reported",
       n3 == {"r4_mode": "normative", "solver_status": "NOT_QUALIFIED", "half_steps": {"completed": 1, "requested": 100},
              "requested_run_completed": True, "field_role": "research", "report_phrase_present": True}, n3)

summary = {"task": "GUI-CORE-N16-P1-INTEGRATION-PREP", "verdict": "PASS" if not problems else "FAIL", "problems": problems,
           "checks": report,
           "note": "PASS means the package is intact and the QA FAIL entries are exactly the documented observations N1 and N2; "
                   "it does not turn gui_check.json verdict_checks into PASS and is not a scientific acceptance."}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
print(summary["verdict"], problems)
sys.exit(0 if not problems else 1)
