"""GUI-CORE-N16-P1-QA — independent functional checks of the GUI routes (QA scratch, not in git).

usage (QA worktree root, detached at d48c10e):  QT_QPA_PLATFORM=offscreen .venv/Scripts/python.exe -X utf8 scratch/gui_n16/gui_check.py

For every case:
  GUI path   = kkvim_gui.backend.process.SolverProcess (the real spawn child process of the GUI) -> RunOutcome;
  CLI path   = `python -m kkvim <command>` in a subprocess (the existing CLI of the same environment), and
  direct     = the kkvim library entry written here from its public signature (not through the adapter).
Compared: status, reason, completed s, field (bitwise), w_bar (bitwise), history values (text of repr), units, field file name,
manifest schema 2, read_manifest, report.html, no-overwrite. No numerical tolerance is used: same environment -> bitwise.
The D-068 published QA control (docs/results/r5_cccc_n46_s2_d068, D_N5_neg) and the D-068 run files (scratch/d068_final/runs) are
compared bitwise for N = 5, a = 1, b = 0.6, 20 x 13, s = 2.
"""
import csv
import hashlib
import io
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SRC = (ROOT / "src").resolve()
OUT = HERE / "check"
PY = sys.executable

import kkvim  # noqa: E402
import kkvim_gui  # noqa: E402
from kkvim import cccc_n1, cccc_n23, cccc_n46, coupled_run, rectangle_run  # noqa: E402
from kkvim_gui.backend import export, kkvim_adapter, process, report  # noqa: E402
from kkvim_gui.backend.request import R4_NORMATIVE, R4_RESEARCH_FIXED, R4_RESEARCH_STOP, RunRequest  # noqa: E402
from kkvim_gui.model.setup_state import SetupState, to_request, validate  # noqa: E402

for mod in (kkvim, kkvim_gui):
    assert Path(mod.__file__).resolve().is_relative_to(SRC), mod.__file__

PHYS = dict(t=0.01, E=2.1e11, nu=0.3)
DEF = kkvim_adapter.DEFAULTS
results = {"cases": {}, "refusals": {}, "setup_state": {}, "schema1": {}, "notes": []}
FAIL = []


def check(tag, name, cond, detail=None):
    results["cases"].setdefault(tag, {}).setdefault("checks", {})[name] = bool(cond) if detail is None else {"ok": bool(cond), "detail": detail}
    if not cond:
        FAIL.append(f"{tag}: {name} {detail if detail is not None else ''}")


def req(a, b, q, nx, ny, N, supports="SSSS", **kw):
    return RunRequest(entered=(), a=a, b=b, q=q, n_x=nx, n_y=ny, N=N, supports=supports, **PHYS, **kw)


def gui_run(request):
    """The GUI computation: the spawn child process of kkvim_gui.backend.process."""
    p = process.SolverProcess(request)
    t0 = time.time()
    while p.poll() is None:
        time.sleep(0.05)
        if time.time() - t0 > 1200:
            p.terminate()
            raise TimeoutError
    p.terminate()
    return p.message


def cli_run(tag, argv):
    out = OUT / "cli" / tag
    shutil.rmtree(out, ignore_errors=True)
    proc = subprocess.run([PY, "-X", "utf8", "-m", "kkvim", *argv, "--out", str(out)], cwd=ROOT, capture_output=True,
                          text=True, encoding="utf-8")
    (OUT / "cli" / f"{tag}.log").write_text(proc.stdout + proc.stderr, encoding="utf-8")
    return proc.returncode, out


def data_lines(path):
    return [line for line in Path(path).read_text(encoding="utf-8").splitlines() if not line.startswith("#")]


def table(path):
    rows = list(csv.reader(io.StringIO("\n".join(data_lines(path)))))
    head = [h.split(" [")[0] for h in rows[0]]
    return [dict(zip(head, r)) for r in rows[1:]]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compare_history(tag, gui_hist, cli_hist):
    g, c = table(gui_hist), table(cli_hist)
    check(tag, "history_rows_equal", len(g) == len(c), (len(g), len(c)))
    common = sorted(set(g[0]) & set(c[0])) if g and c else []
    diffs = [(k, i, g[i][k], c[i][k]) for i in range(min(len(g), len(c))) for k in common if g[i][k] != c[i][k]]
    check(tag, "history_common_columns_text_equal", not diffs, {"columns": common, "diffs": diffs[:5]})


def export_checks(tag, outcome, expect_file, expect_role, cli_out=None, cli_field=None):
    folder = OUT / "export" / tag
    shutil.rmtree(folder, ignore_errors=True)
    paths = report.write_package(outcome, folder)
    names = sorted(p.name for p in folder.iterdir())
    expected = sorted(n for n in ("manifest.json", "history.csv", "report.html", expect_file) if n)
    check(tag, "export_files", names == expected, names)
    m = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    st, rs = m["statuses"], m["results"]
    check(tag, "manifest_schema_2", m["schema_version"] == 2)
    check(tag, "manifest_field_role", st["field_role"] == expect_role and rs["field_role_code"] == expect_role, st["field_role"])
    check(tag, "manifest_field_file", st["field_file"] == expect_file and m["files"].get("field") == expect_file, st["field_file"])
    check(tag, "manifest_solver_status", st["solver_status"] == outcome.status and st["solver_reason"] == outcome.reason)
    check(tag, "manifest_scientific", bool(st["scientific_status"]) and st["scientific"]["text"] == outcome.science_text)
    check(tag, "manifest_half_steps", rs["half_steps"] == {"completed": outcome.s, "requested": outcome.s_requested}, rs["half_steps"])
    check(tag, "manifest_requested_completed", st["requested_run_completed"] == outcome.requested_completed)
    check(tag, "manifest_model_route", m["model"]["route"] == outcome.request.route and m["model"]["supports"] == outcome.request.supports)
    check(tag, "manifest_provenance", bool(m.get("program")) or bool(m.get("versions")) or "kkvim" in json.dumps(m)[:20000])
    check(tag, "no_CONVERGED_unless_R3", outcome.status != "CONVERGED" or outcome.request.route == "ssss_r3")
    if outcome.status != "CONVERGED":
        check(tag, "accepted_role_only_R3_CONVERGED", expect_role != "accepted")
    rm = export.read_manifest(folder / "manifest.json")
    check(tag, "read_manifest_v2", rm["field_role"] == expect_role and rm["field_file"] == expect_file and not rm["legacy"]
          and rm["solver_status"] == outcome.status and rm["s_completed"] == outcome.s and rm["s_requested"] == outcome.s_requested)
    html = (folder / "report.html").read_text(encoding="utf-8")
    check(tag, "report_agrees", outcome.status in html and (expect_file or "—") in html
          and report.ROLE_TITLE.get(expect_role, "поле не получено") in html)
    if expect_role == "diagnostic":
        check(tag, "report_diagnostic_banner", "ДИАГНОСТИКА" in html)
        check(tag, "field_header_diagnostic", any("DIAGNOSTIC" in line for line in
                                                  (folder / expect_file).read_text(encoding="utf-8").splitlines()[:4]))
    if expect_file:
        w = np.asarray(outcome.w)
        exported = np.array([[float(r["w"]) for r in table(folder / expect_file)]]).reshape(w.shape)
        check(tag, "exported_w_equals_solver_bitwise", np.array_equal(exported, w))
    if cli_out is not None:
        check(tag, "field_file_name_as_cli", (expect_file is None and cli_field is None) or expect_file == cli_field, (expect_file, cli_field))
        if expect_file and cli_field:
            check(tag, "field_csv_data_bitwise_vs_cli", data_lines(folder / expect_file) == data_lines(cli_out / cli_field))
        compare_history(tag, folder / "history.csv", cli_out / "history.csv")
    # a second export into the same folder is refused and leaves the first package unchanged
    before = {p.name: sha(p) for p in folder.iterdir()}
    try:
        report.write_package(outcome, folder)
        refused = False
    except FileExistsError:
        refused = True
    after = {p.name: sha(p) for p in folder.iterdir()}
    check(tag, "second_export_refused_unchanged", refused and before == after)
    return m


def common_outcome_checks(tag, o, status_expect=None, role=None, s=None):
    if status_expect is not None:
        check(tag, "status", o.status == status_expect, o.status)
    if role is not None:
        check(tag, "field_role", o.field_role == role, o.field_role)
    if s is not None:
        check(tag, "s_completed", o.s == s, o.s)
    results["cases"][tag]["outcome"] = {"status": o.status, "reason": o.reason, "s": o.s, "s_requested": o.s_requested,
                                        "field_role": o.field_role, "field_file": export.field_file_name(o), "mark": o.mark,
                                        "requested_completed": o.requested_completed, "route": o.route, "solver": o.solver,
                                        "diagnostic_only": o.diagnostic_only, "stop_kind": o.stop_kind, "D": o.D}


def field_eq_direct(tag, o, w, w_bar):
    check(tag, "w_bitwise_vs_direct", w is not None and np.array_equal(np.asarray(o.w), np.asarray(w)))
    check(tag, "w_bar_bitwise_vs_direct", w_bar is not None and np.array_equal(np.asarray(o.w_bar), np.asarray(w_bar)))


def cli_argv(cmd, a, b, q, nx, ny, extra=()):
    return [cmd, "--a", repr(a), "--b", repr(b), "--t", "0.01", "--E", "2.1e11", "--nu", "0.3", "--q", repr(q),
            "--nx", str(nx), "--ny", str(ny), *extra]


def cli_field(out):
    for name in ("field.csv", "field_research.csv", "field_diagnostic.csv"):
        if (out / name).exists():
            return name
    return None


# ---------------------------------------------------------------- SSSS
def ssss_r3(tag, s_min, s_max, expect_status, expect_role, expect_file):
    r = req(1.0, 1.0, 1000.0, 16, 16, 1, tol_inf=DEF["tol_inf"], tol_E=DEF["tol_E"], s_min=s_min, s_max=s_max)
    kind, o = gui_run(r)
    assert kind == process.OK, o
    common_outcome_checks(tag, o, expect_status, expect_role)
    d = rectangle_run.run_rectangle_physical(1.0, 1.0, 0.01, 2.1e11, 0.3, 1000.0, 16, 16, tol_inf=DEF["tol_inf"], tol_E=DEF["tol_E"],
                                             s_min=s_min, s_max=s_max)
    check(tag, "status_vs_direct", d.status == o.status)
    field_eq_direct(tag, o, d.run.last_field(), d.dimensionless_field())
    code, out = cli_run(tag, cli_argv("run", 1.0, 1.0, 1000.0, 16, 16, ("--s-min", str(s_min), "--s-max", str(s_max))))
    results["cases"][tag]["cli_exit"] = code
    j = json.loads((out / "result.json").read_text(encoding="utf-8"))
    check(tag, "status_vs_cli", json.dumps(j).count(o.status) > 0)
    export_checks(tag, o, expect_file, expect_role, out, cli_field(out))


ssss_r3("S1_R3_converged", DEF["s_min"], DEF["s_max"], "CONVERGED", "accepted", "field.csv")
ssss_r3("S2_R3_max_half_steps", 1, 2, "MAX_HALF_STEPS", "diagnostic", "field_diagnostic.csv")

# R4 N = 2 research (stop on stabilization) and N = 3 fixed: CLI run-ssss
for tag, N, mode, hs, extra in (("S3_R4_N2_research", 2, R4_RESEARCH_STOP, 100, ()),
                                ("S4_R4_N3_fixed4", 3, R4_RESEARCH_FIXED, 4, ("--fixed-half-steps",))):
    r = req(1.0, 1.0, 1000.0, 16, 16, N, r4_mode=mode, transform="fft", half_steps=hs)
    kind, o = gui_run(r)
    assert kind == process.OK, o
    common_outcome_checks(tag, o, role="research")
    code, out = cli_run(tag, cli_argv("run-ssss", 1.0, 1.0, 1000.0, 16, 16, ("--N", str(N), "--half-steps", str(hs), *extra)))
    results["cases"][tag]["cli_exit"] = code
    j = json.loads((out / "result.json").read_text(encoding="utf-8"))
    check(tag, "status_vs_cli", j["status"] == o.status, (j["status"], o.status))
    check(tag, "s_vs_cli", j.get("half_steps_completed", j.get("s")) in (o.s, None), j.get("half_steps_completed"))
    export_checks(tag, o, "field_research.csv", "research", out, cli_field(out))

# R4 N = 2 normative (GUI-only mode; no CLI): direct run_coupled(research_mode=False)
tag = "S5_R4_N2_normative"
r = req(1.0, 1.0, 1000.0, 16, 16, 2, r4_mode=R4_NORMATIVE, transform="fft", half_steps=DEF["half_steps"])
kind, o = gui_run(r)
assert kind == process.OK, o
common_outcome_checks(tag, o, "NOT_QUALIFIED", "research", 1)
D = o.D
d = coupled_run.run_coupled(1.0, 1.0, D, 1000.0, 16, 16, 2, transform="fft", half_steps=DEF["half_steps"], research_mode=False)
check(tag, "status_vs_direct", d.status == o.status, d.status)
field_eq_direct(tag, o, d.field(), d.field() * D / (1000.0 * 1.0**4))
export_checks(tag, o, "field_research.csv", "research")
results["notes"].append("S5: R4 normative (D-047) ends after s = 1 with NOT_QUALIFIED; the GUI treats it as the requested end "
                        "(field_research.csv, PROVISIONAL / NOT_QUALIFIED). run-ssss has no normative mode, so no CLI reference.")

# SSSS N = 4 research, a = b = 1, 40 x 40, s = 2 (inside the studied domain): CLI run-ssss --research-only
tag = "S6_SSSS_N4_research"
r = req(1.0, 1.0, 1000.0, 40, 40, 4, half_steps=2, ssss_research=True)
kind, o = gui_run(r)
assert kind == process.OK, o
common_outcome_checks(tag, o, "RESEARCH_LIMIT_REACHED", "research", 2)
check(tag, "mark_research", o.mark == "RESEARCH_ONLY / PROVISIONAL / NOT_QUALIFIED", o.mark)
code, out = cli_run(tag, cli_argv("run-ssss", 1.0, 1.0, 1000.0, 40, 40, ("--N", "4", "--research-only", "--half-steps", "2")))
results["cases"][tag]["cli_exit"] = code
j = json.loads((out / "result.json").read_text(encoding="utf-8"))
check(tag, "status_vs_cli", j["status"] == o.status, (j["status"], o.status))
export_checks(tag, o, "field_research.csv", "research", out, cli_field(out))


# ---------------------------------------------------------------- CCCC
def cccc(tag, N, a, b, q, nx, ny, s, expect_status, expect_role, expect_file, s_done, cmd, extra=()):
    r = req(a, b, q, nx, ny, N, supports="CCCC", half_steps=s)
    kind, o = gui_run(r)
    assert kind == process.OK, (tag, o)
    common_outcome_checks(tag, o, expect_status, expect_role, s_done)
    check(tag, "mark_research", o.mark == "RESEARCH_ONLY / PROVISIONAL / NOT_QUALIFIED", o.mark)
    check(tag, "never_accepted", o.field_role != "accepted" and o.solver != "R3")
    D = o.D
    if N == 1:
        d = cccc_n1.run_cccc_n1_physical(a, b, 0.01, 2.1e11, 0.3, q, nx, ny, s)
    elif N <= 3:
        d = cccc_n23.run_cccc_n23_physical(a, b, 0.01, 2.1e11, 0.3, q, nx, ny, N, s)
    else:
        d = cccc_n46.run_cccc_n46_physical(a, b, 0.01, 2.1e11, 0.3, q, nx, ny, N, s)
    check(tag, "status_reason_s_vs_direct", (d.status, d.reason or "", d.half_steps_completed) == (o.status, o.reason, o.s))
    check(tag, "D_vs_direct", d.D == D)
    if d.field() is not None:
        field_eq_direct(tag, o, d.field(), d.dimensionless_field())
    else:
        check(tag, "no_field", o.w is None)
    qr = [bool(getattr(st, "qr_executed", False)) for st in d.steps]
    results["cases"][tag]["direct_steps"] = [{"s": st.s, "status": st.status, "qr_executed": getattr(st, "qr_executed", None),
                                              "rn5": getattr(st, "rn5_verdict", None) if hasattr(st, "rn5_verdict") else None}
                                             for st in d.steps]
    code, out = cli_run(tag, cli_argv(cmd, a, b, q, nx, ny, (*extra, "--half-steps", str(s))))
    results["cases"][tag]["cli_exit"] = code
    j = json.loads((out / "result.json").read_text(encoding="utf-8"))
    check(tag, "status_reason_s_vs_cli", (j["status"], j["reason"] or "", j["half_steps_completed"]) == (o.status, o.reason, o.s),
          (j["status"], j["half_steps_completed"]))
    export_checks(tag, o, expect_file, expect_role, out, cli_field(out))
    return o, d, out, qr


cccc("C1_CCCC_N1_s3", 1, 1.0, 1.0, 1000.0, 16, 16, 3, "COMPLETED_FIXED_HALF_STEPS", "research", "field_research.csv", 3, "run-cccc-n1")
cccc("C2_CCCC_N2_s2", 2, 1.0, 1.0, 1000.0, 16, 16, 2, "COMPLETED_FIXED_HALF_STEPS", "research", "field_research.csv", 2,
     "run-cccc-n23", ("--N", "2"))
o4, d4, out4, _ = cccc("C3_CCCC_N4_32_s2", 4, 1.0, 1.0, 1000.0, 32, 32, 2, "COMPLETED_FIXED_HALF_STEPS", "research",
                       "field_research.csv", 2, "run-cccc-n46", ("--N", "4"))
o5p, d5p, out5p, _ = cccc("C4_CCCC_N5_b06_s2_pos", 5, 1.0, 0.6, 1000.0, 20, 13, 2, "COMPLETED_FIXED_HALF_STEPS", "research",
                          "field_research.csv", 2, "run-cccc-n46", ("--N", "5"))
o5n, d5n, out5n, _ = cccc("C5_CCCC_N5_b06_s2_neg", 5, 1.0, 0.6, -1000.0, 20, 13, 2, "COMPLETED_FIXED_HALF_STEPS", "research",
                          "field_research.csv", 2, "run-cccc-n46", ("--N", "5"))
o6, d6, out6, qr6 = cccc("C6_CCCC_N6_a5_32x13_s2", 6, 5.0, 1.0, 1000.0, 32, 13, 2, "STOPPED_RANK_GUARD_UNCERTIFIED", "diagnostic",
                         "field_diagnostic.csv", 1, "run-cccc-n46", ("--N", "6"))
o33, d33, out33, qr33 = cccc("C7_CCCC_N4_33x9_s2", 4, 1.0, 1.0, 1000.0, 33, 9, 2, None, "diagnostic", "field_diagnostic.csv", 1,
                             "run-cccc-n46", ("--N", "4"))

# sign of the load (N = 5, 20 x 13): w(-q) = -w(q) exactly, w_bar equal
tag = "C5_CCCC_N5_b06_s2_neg"
check(tag, "w_neg_equals_minus_w_pos", np.array_equal(np.asarray(o5n.w), -np.asarray(o5p.w)))
check(tag, "w_bar_neg_equals_w_bar_pos", np.array_equal(np.asarray(o5n.w_bar), np.asarray(o5p.w_bar)))
check(tag, "deflection_sign_follows_q", float(np.max(o5p.w)) > 0 and float(np.min(o5n.w)) < 0 and float(np.max(o5n.w)) <= 0)

# the D-068 QA control (published: docs/results/r5_cccc_n46_s2_d068; run files: scratch/d068_final/runs)
ctl = json.loads((ROOT / "docs/results/r5_cccc_n46_s2_d068/results/new/final_check.json").read_text(encoding="utf-8"))
pub = ctl["s2"]["D_N5_neg"]
tag = "C5_CCCC_N5_b06_s2_neg"
check(tag, "published_D068_control_inputs", (pub["N"], pub["a"], pub["b"], pub["nx"], pub["ny"], pub["q"]) == (5, 1.0, 0.6, 20, 13, -1000.0))
check(tag, "published_D068_control_all_checks_true", all(v is True for v in pub["checks"].values()))
check(tag, "D_equals_published_D068", o5n.D == pub["values"]["D"], (o5n.D, pub["values"]["D"]))
d068_runs = ROOT / "scratch/d068_final/runs"
for tg, run_tag, gui_out in (("C5_CCCC_N5_b06_s2_neg", "D_N5_neg", "C5_CCCC_N5_b06_s2_neg"),
                             ("C4_CCCC_N5_b06_s2_pos", "R_C7", "C4_CCCC_N5_b06_s2_pos"),
                             ("C3_CCCC_N4_32_s2", "R_C1", "C3_CCCC_N4_32_s2"),
                             ("C6_CCCC_N6_a5_32x13_s2", "C_pos", "C6_CCCC_N6_a5_32x13_s2")):
    src = d068_runs / run_tag
    if src.exists():
        fname = cli_field(src)
        check(tg, f"field_bitwise_vs_D068_QA_run_{run_tag}", data_lines(OUT / "export" / gui_out / fname) == data_lines(src / fname))
        compare_history(tg + f"__vs_D068_{run_tag}", OUT / "export" / gui_out / "history.csv", src / "history.csv")
    else:
        results["notes"].append(f"D-068 run files {src} not available")

# C6: RN-5 refusal, no QR / X at s = 2, diagnostic s = 1 field
tag = "C6_CCCC_N6_a5_32x13_s2"
s2 = [st for st in d6.steps if st.s == 2]
check(tag, "rn5_refused", "RN5_UNRESOLVED" in (o6.reason or ""), o6.reason)
check(tag, "no_qr_at_s2", not any(bool(getattr(st, "qr_executed", False)) for st in s2), [getattr(st, "qr_executed", None) for st in s2])
check(tag, "history_has_no_completed_s2", all(row.get("status") != "HALF_STEP_DONE" for row in o6.history if row.get("s") == 2))
check(tag, "field_is_s1_G_VT", o6.s == 1 and o6.diagnostic_only)
check(tag, "cli_exit_6", results["cases"][tag]["cli_exit"] == 6)
# C7: grid 33 > 32 -> stop before RN-5 and QR
tag = "C7_CCCC_N4_33x9_s2"
s2 = [st for st in d33.steps if st.s == 2]
check(tag, "stop_before_rn5_qr", not any(bool(getattr(st, "qr_executed", False)) for st in s2)
      and all(getattr(st, "rn5_verdict", None) in (None, "") for st in s2), [(getattr(st, "rn5_verdict", None)) for st in s2])
check(tag, "status_protective", o33.status in kkvim_adapter.CCCC_PROTECTIVE and o33.stop_kind == "not_met", (o33.status, o33.stop_kind))
check(tag, "grid_note", any("32" in n and "RN-5" in n for n in o33.notes))
check(tag, "cli_exit_6", results["cases"][tag]["cli_exit"] == 6)


# ---------------------------------------------------------------- refusals before calculation (nothing computed)
def refusal(tag, request, needle):
    problems = kkvim_adapter.check_request(request)
    kind, text = gui_run(request)
    ok = bool(problems) and kind == process.INPUT and needle in text
    results["refusals"][tag] = {"ok": ok, "problems": problems, "process": kind}
    if not ok:
        FAIL.append(f"refusal {tag}: {problems} {kind}")


refusal("SSSS_N5_without_research_choice", req(1.0, 1.0, 1000.0, 40, 40, 5, half_steps=2, ssss_research=False), "D-056.7")
refusal("SSSS_N4_grid_outside_study", req(1.0, 1.0, 1000.0, 32, 32, 4, half_steps=2, ssss_research=True), "NOT_EXECUTED_OUTSIDE_DOMAIN")
refusal("SSSS_N6_s41_outside_study", req(1.0, 1.0, 1000.0, 40, 40, 6, half_steps=41, ssss_research=True), "NOT_EXECUTED_OUTSIDE_DOMAIN")
refusal("SSSS_N4_ratio_outside_study", req(1.0, 1.1, 1000.0, 40, 40, 4, half_steps=2, ssss_research=True), "NOT_EXECUTED_OUTSIDE_DOMAIN")
for N in (2, 3, 4, 5, 6):
    refusal(f"CCCC_N{N}_s3", req(1.0, 1.0, 1000.0, 16, 16, N, supports="CCCC", half_steps=3), "s ≥ 3")
refusal("CCCC_N1_s9", req(1.0, 1.0, 1000.0, 16, 16, 1, supports="CCCC", half_steps=9), "недоступно")
refusal("CCCC_N4_grid_below_2N+1", req(1.0, 1.0, 1000.0, 8, 8, 4, supports="CCCC", half_steps=1), "NOT_EXECUTED_OUTSIDE_DOMAIN")
refusal("CCCC_N2_q0", req(1.0, 1.0, 0.0, 16, 16, 2, supports="CCCC", half_steps=1), "")
refusal("N7_no_route", req(1.0, 1.0, 1000.0, 16, 16, 7), "N = 7")
refusal("supports_SCSC_unknown", req(1.0, 1.0, 1000.0, 16, 16, 1, supports="SCSC"), "SCSC")

# scope probe: the GUI applies the R3 input checks of validated_rectangle_input to CCCC as well (stricter, never looser)
probe = {}
for label, a, b, nx, ny, N, s in (("CCCC_N1_b/a=6", 1.0, 6.0, 16, 16, 1, 1), ("CCCC_N1_grid_700x8", 1.0, 1.0, 700, 8, 1, 1)):
    r = req(a, b, 1000.0, nx, ny, N, supports="CCCC", half_steps=s)
    probe[label] = {"gui_check_request": kkvim_adapter.check_request(r),
                    "solver_domain_refusal": cccc_n1.domain_refusal(a, b, nx, ny, s)}
results["scope_probe"] = probe

# ---------------------------------------------------------------- form state: no implicit research continuation
st = SetupState()
st4 = st.with_choice(N=4)
st4r = st4.with_choice(ssss_research=True)
st5 = st4r.with_choice(N=5)
stc = st4r.with_choice(supports="CCCC")
stc2 = SetupState().with_choice(supports="CCCC", N=4).with_choice(cccc_half_steps=2)
stc2n = stc2.with_choice(N=5)
st2r = SetupState().with_choice(N=2, r4_mode=R4_RESEARCH_STOP)
st3 = st2r.with_choice(N=3)
try:
    SetupState().with_choice(supports="CCCC", N=4).with_choice(cccc_half_steps=3)
    s3_refused = False
except ValueError:
    s3_refused = True
results["setup_state"] = {
    "N4_default_research_off": st4.ssss_research is False and "research" in validate(st4),
    "N4_research_on_then_N5_reset": st4r.ssss_research is True and st5.ssss_research is False,
    "research_reset_on_supports_change": stc.ssss_research is False,
    "cccc_s2_reset_on_N_change": stc2.cccc_half_steps == 2 and stc2n.cccc_half_steps == 1,
    "r4_research_reset_on_N_change": st2r.r4_mode == R4_RESEARCH_STOP and st3.r4_mode == R4_NORMATIVE,
    "cccc_s3_not_offered": s3_refused,
    "cccc_request_from_form": to_request(stc2).half_steps == 2 and to_request(stc2).supports == "CCCC",
}
for k, v in results["setup_state"].items():
    if not v:
        FAIL.append(f"setup_state {k}")

# ---------------------------------------------------------------- schema 1 (written by the base export 306e85a)
expect1 = {"r3_converged": "accepted", "r3_max_half_steps": "diagnostic", "r4_normative": "research", "r4_research": "research"}
for tag, role in expect1.items():
    m = export.read_manifest(HERE / "schema1" / tag / "manifest.json")
    ok = m["legacy"] and m["schema_version"] == 1 and m["field_role"] == role and m["field_file"] == "field.csv" \
        and m["requested_run_completed"] is None
    results["schema1"][tag] = {"ok": ok, "read": m}
    if not ok:
        FAIL.append(f"schema1 {tag}: {m['field_role']}")
try:
    bad = OUT / "schema3.json"
    bad.write_text(json.dumps({"schema_version": 3}), encoding="utf-8")
    export.read_manifest(bad)
    results["schema1"]["schema3_refused"] = False
    FAIL.append("schema3 accepted")
except ValueError:
    results["schema1"]["schema3_refused"] = True

# ---------------------------------------------------------------- demo marks
from kkvim_gui.backend.outcome import ORIGIN_DEMO  # noqa: E402
from dataclasses import replace  # noqa: E402

demo = replace(o4, origin=ORIGIN_DEMO)
folder = OUT / "export" / "DEMO"
shutil.rmtree(folder, ignore_errors=True)
paths = report.write_package(demo, folder)
names = sorted(p.name for p in folder.iterdir())
md = json.loads(paths["manifest"].read_text(encoding="utf-8"))
results["demo"] = {"files": names, "marks": md["marks"],
                   "ok": all("TEST-DATA" in n for n in names) and "TEST DATA" in md["marks"]
                   and "TEST DATA" in (folder / "report_TEST-DATA.html").read_text(encoding="utf-8")}
if not results["demo"]["ok"]:
    FAIL.append("demo marks")

results["failures"] = FAIL
results["verdict_checks"] = "PASS" if not FAIL else "FAIL"
(OUT / "gui_check.json").write_text(json.dumps(results, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
print(json.dumps({"failures": FAIL, "cases": {k: sum(1 for c in v["checks"].values() if c is True or (isinstance(c, dict) and c["ok"]))
                                               for k, v in results["cases"].items()}}, ensure_ascii=False, indent=1))
print("VERDICT_CHECKS", results["verdict_checks"])
