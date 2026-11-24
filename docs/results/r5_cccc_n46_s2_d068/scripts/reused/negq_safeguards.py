"""R5-RN5-NEGATIVE-Q-FIX-P0A-INDEPENDENT: protective stops for q = -1000 (and +1000 for comparison), CLI outputs, signed zeros by array.
Each scenario is a separate process of inject_n23.py on the real kkvim of the QA worktree (detached at the hotfix).
"""
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
QA = HERE.parents[1]
PY = str(QA / ".venv" / "Scripts" / "python.exe")
EV = HERE / "events"
CLI = HERE / "cli"
EV.mkdir(exist_ok=True)
CLI.mkdir(exist_ok=True)
ENV = {**os.environ, "PYTHONIOENCODING": "utf-8"}
GEO = ("1", "0.6", "10", "9", "3")  # a b nx ny N, D = 1


def api(tag, inject, q, nx="10", ny="9", hs="2"):
    out = EV / f"{tag}_q{q}.json"
    env = {**ENV, "QA_MODE": "api", "QA_INJECT": inject, "QA_EVENTS": str(out)}
    p = subprocess.run([PY, str(HERE / "inject_n23.py"), "1", "0.6", nx, ny, "3", hs, q], env=env, cwd=QA, capture_output=True, text=True,
                       encoding="utf-8")
    assert p.returncode == 0, p.stderr
    return json.loads(out.read_text(encoding="utf-8"))


def kinds(ev):
    return [e[0] for e in ev["events"] if e[0] in ("qr", "solve_bc", "rn5", "rt7")]


R = {}
S1_ONLY = ["qr", "solve_bc"]
UP_TO_RN5 = ["qr", "solve_bc", "rn5"]
UP_TO_RT7 = ["qr", "solve_bc", "rn5", "qr", "rt7"]
FULL = ["qr", "solve_bc", "rn5", "qr", "rt7", "solve_bc"]
SCEN = [
    ("A0_success", "", {}, "COMPLETED_FIXED_HALF_STEPS", FULL, "s2"),
    ("A1_dependence", "dep", {}, "STOP_BEFORE_QR_S1", S1_ONLY, "s1"),
    ("A2_rn5_refusal", "rn5_refuse", {}, "STOPPED_RANK_GUARD_UNCERTIFIED", UP_TO_RN5, "s1"),
    ("A3_grid_nx33", "", {"nx": "33"}, "STOPPED_RANK_GUARD_UNCERTIFIED", S1_ONLY, "s1"),
    ("A3_grid_ny33", "", {"ny": "33"}, "STOPPED_RANK_GUARD_UNCERTIFIED", S1_ONLY, "s1"),
    ("A4_rt7_refusal", "rt7_refuse", {}, "STOPPED_RANK_GUARD_UNCERTIFIED", UP_TO_RT7, "s1"),
    ("A5_rn5_float_eps", "rn5_float_eps", {}, "STOPPED_RANK_GUARD_UNCERTIFIED", UP_TO_RT7, "s1"),
    ("A5_rn5_missing_eps", "rn5_missing_eps", {}, "STOPPED_RANK_GUARD_UNCERTIFIED", UP_TO_RT7, "s1"),
    ("A6_nonfinite_s2", "nonfinite_s2", {}, "ERROR_NONFINITE", FULL, "s1"),
    ("A7_hs3", "", {"hs": "3"}, "STOPPED_RANK_GUARD_UNCERTIFIED", FULL, "s2"),
]
for tag, inj, kw, status, calls, field_from in SCEN:
    res = {}
    for q in ("-1000", "1000"):
        ev = api(tag, inj, q, **kw)
        s2 = ev["steps"][1] if len(ev["steps"]) > 1 else None
        ok = ev["status"] == status and kinds(ev) == calls and ev["field_from"] == field_from
        if tag == "A4_rt7_refusal" or tag.startswith("A5"):
            ok &= s2["qr_executed"] is True and s2["G_none"] is False and s2["A_none"] and s2["V_none"] and not s2["done"]
            ok &= ev["rejected_G2_used"] is False
        if tag in ("A1_dependence", "A2_rn5_refusal") or tag.startswith("A3"):
            ok &= s2["qr_executed"] is False and s2["G_none"] and s2["A_none"] and s2["V_none"]
        if tag in ("A0_success", "A7_hs3"):
            ok &= ev["completed"] == 2
        elif tag != "A6_nonfinite_s2":
            ok &= ev["completed"] == 1
        if tag == "A6_nonfinite_s2":
            ok &= ev["completed"] == 1 and not s2["done"]
        if tag == "A7_hs3":
            ok &= len(ev["steps"]) == 3 and ev["steps"][2]["qr_executed"] is False and ev["steps"][2]["status"] == status
        rn5_q = [e[1] for e in ev["events"] if e[0] == "rn5"]
        ok &= all(v == float(q) for v in rn5_q)  # RN-5 receives the actual (signed) q of the run
        res[q] = {"ok": ok, "status": ev["status"], "calls": kinds(ev), "field_from": ev["field_from"], "completed": ev["completed"],
                  "rejected_G2_used": ev["rejected_G2_used"], "s2": s2, "field_sum": ev["field_sum"],
                  "rt7_reason": next((e[2] for e in ev["events"] if e[0] == "rt7_result"), None)}
    res["same_outcome_both_signs"] = (res["-1000"]["status"], res["-1000"]["calls"], res["-1000"]["field_from"]) == \
        (res["1000"]["status"], res["1000"]["calls"], res["1000"]["field_from"])
    res["field_odd"] = res["-1000"]["field_sum"] == -res["1000"]["field_sum"] if res["1000"]["field_sum"] is not None else None
    R[tag] = res
    print(tag, res["-1000"]["ok"], res["1000"]["ok"], res["same_outcome_both_signs"], res["field_odd"], res["-1000"]["calls"], flush=True)


# --- CLI ------------------------------------------------------------------------------------------------------------------------------
def cli(tag, inject, q, nx="10", ny="9", extra=()):
    out = CLI / f"{tag}_q{q}"
    if out.exists():
        shutil.rmtree(out)
    ev = CLI / f"{tag}_q{q}.events.json"
    env = {**ENV, "QA_MODE": "cli", "QA_INJECT": inject, "QA_EVENTS": str(ev)}
    args = ["run-cccc-n23", "--N", "3", "--a", "1", "--b", "0.6", "--t", "0.01", "--E", "2.1e11", "--nu", "0.3", "--q", q, "--nx", nx,
            "--ny", ny, "--half-steps", "2", "--out", str(out), *extra]
    p = subprocess.run([PY, str(HERE / "inject_n23.py"), *args], env=env, cwd=QA, capture_output=True, text=True, encoding="utf-8")
    (CLI / f"{tag}_q{q}.log").write_text(f"exit={p.returncode}\n--- stdout\n{p.stdout}\n--- stderr\n{p.stderr}", encoding="utf-8")
    files = sorted(x.name for x in out.iterdir()) if out.exists() else []
    return p.returncode, out, files


def read_field(path):
    rows = [r for r in csv.reader(path.read_text(encoding="utf-8").splitlines()) if r and not r[0].startswith("#")]
    head, body = rows[0], rows[1:]
    return head, np.array([[float(v) for v in r] for r in body])


C = {}
for tag, inj, kw, code, fname in (("C0_success", "", {}, 0, "field_research.csv"), ("C1_grid33", "", {"nx": "33"}, 6, "field_diagnostic.csv"),
                                  ("C2_rt7_refusal", "rt7_refuse", {}, 6, "field_diagnostic.csv"),
                                  ("C3_rn5_refusal", "rn5_refuse", {}, 6, "field_diagnostic.csv")):
    res = {}
    for q in ("-1000", "1000"):
        rc, out, files = cli(tag, inj, q, **kw)
        doc = json.loads((out / "result.json").read_text(encoding="utf-8"))
        other = "field_diagnostic.csv" if fname == "field_research.csv" else "field_research.csv"
        head, F = read_field(out / fname)
        res[q] = {"exit": rc, "files": files, "ok": rc == code and fname in files and other not in files, "status": doc.get("status"),
                  "header": head[:6], "field": F, "diag_text": "DIAGNOSTIC" in (out / fname).read_text(encoding="utf-8")}
        res[q]["ok"] &= res[q]["diag_text"] == (fname == "field_diagnostic.csv")
        text = (out / "result.json").read_text(encoding="utf-8")
        res[q]["forbidden"] = [w for w in ("CONVERGED", "ACCEPTED") if w in text]
        res[q]["ok"] &= not res[q]["forbidden"]
    Fm, Fp = res["-1000"]["field"], res["1000"]["field"]
    # the files hold coordinates + values; columns that do not depend on q must be equal, the value column(s) odd or even by name
    res["equal_shape"] = Fm.shape == Fp.shape
    res["cols_equal_or_negated"] = [("equal" if np.array_equal(Fm[:, j], Fp[:, j]) else "negated" if np.array_equal(Fm[:, j], -Fp[:, j])
                                     else "OTHER") for j in range(Fm.shape[1])]
    for q in res:
        if isinstance(res[q], dict) and "field" in res[q]:
            res[q]["field"] = None
    C[tag] = res
    print(tag, res["-1000"]["exit"], res["1000"]["exit"], res["-1000"]["ok"], res["1000"]["ok"], res["-1000"]["header"],
          res["cols_equal_or_negated"], flush=True)

# --- signed zeros by array (from the runs of runner_negq.py) ------------------------------------------------------------------------
ODD = ("s1_V", "s1_F", "s2_R", "s2_companion", "s2_given", "s2_F", "s2_V", "field")
Z = {}
for p in sorted((HERE / "runs").glob("hotfix_*q+*.npz")):
    zp, zn = np.load(p), np.load(Path(str(p).replace("q+", "q-")))
    per = {}
    for k in zp.files:
        y = -zp[k] if k in ODD else zp[k]
        d = zn[k].view(np.uint64) != y.view(np.uint64)
        if np.any(d):
            both0 = (zn[k] == 0) & (y == 0)
            where = "boundary" if k in ("field", "wbar") and not np.any(d[1:-1, 1:-1]) else ("strict lower triangle" if k == "s2_R" and not
                                                                                             np.any(np.triu(d)) else "other")
            per[k] = {"count": int(d.sum()), "all_zero_values": bool(np.all(both0[d])), "where": where}
    Z[p.stem] = per
    print(p.stem, per)

summary = {"api_all_ok": all(v["-1000"]["ok"] and v["1000"]["ok"] and v["same_outcome_both_signs"] for v in R.values()),
           "cli_all_ok": all(v["-1000"]["ok"] and v["1000"]["ok"] and "OTHER" not in v["cols_equal_or_negated"] for v in C.values()),
           "signed_zero_only": all(all(x["all_zero_values"] and x["where"] != "other" for x in per.values()) for per in Z.values()),
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           "inject_sha256": hashlib.sha256((HERE / "inject_n23.py").read_bytes()).hexdigest()}
(HERE / "negq_safeguards.json").write_text(json.dumps({"api": R, "cli": C, "signed_zeros": Z, "summary": summary}, ensure_ascii=False,
                                                      indent=1, default=str), encoding="utf-8")
print(json.dumps(summary, indent=1))
sys.exit(0 if summary["api_all_ok"] and summary["cli_all_ok"] and summary["signed_zero_only"] else 1)
