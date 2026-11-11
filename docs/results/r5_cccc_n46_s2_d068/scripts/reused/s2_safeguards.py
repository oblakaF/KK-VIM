"""QA R5-CCCC-N46-S2-D068-INDEPENDENT: focused safeguard controls (QA scratch). Each scenario is a fresh process of inject_s2.py
(real kkvim, one injected fault). Expectations are written here before the runs. Also: parent 0551141 vs 754dcfe on the N1, N2-3,
SSSS and N4-6 s = 1 interfaces (parent sources from ``git archive`` into scratch; no worktree)."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

QA = Path(r"<QA_WORKTREE>")
PY = str(QA / ".venv" / "Scripts" / "python.exe")
S = QA / "scratch" / "d068_s2"
DRIVER = str(S / "inject_s2.py")
EV = S / "events"
CLI = S / "cli"
PAR = S / "parent_src"
CMP = S / "cmp"
ENV = {**os.environ, "PYTHONIOENCODING": "utf-8"}
PHYS = ["--a", "1", "--t", "0.01", "--E", "2.1e11", "--nu", "0.3", "--q", "1000"]
results = {}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def api(name, inject, a=1.0, b=1.0, nx=9, ny=9, N=4, hs=2):
    path = EV / f"api_{name}.json"
    subprocess.run([PY, "-I", DRIVER, str(a), str(b), str(nx), str(ny), str(N), str(hs)], cwd=S, check=True,
                   env={**ENV, "QA_MODE": "api", "QA_INJECT": inject, "QA_EVENTS": str(path)})
    return json.loads(path.read_text(encoding="utf-8"))


def cli(name, inject, args, out=None, fresh=True):
    out = out or CLI / name
    if fresh and out.exists():
        shutil.rmtree(out)
    path = EV / f"cli_{name}.json"
    p = subprocess.run([PY, "-I", DRIVER, *args, "--out", str(out)], cwd=S, capture_output=True, text=True, encoding="utf-8",
                       env={**ENV, "QA_MODE": "cli", "QA_INJECT": inject, "QA_EVENTS": str(path)})
    files = sorted(f.name for f in out.iterdir()) if out.exists() else []
    return {"exit": p.returncode, "files": files, "stderr_tail": p.stderr[-600:], "out": out}


def kinds(ev):
    return [e[0] if e[0] not in ("solve_given", "solve_bc") else f"{e[0]}{e[1]}" for e in ev["events"]]


def n46_args(N=4, b="1", nx=9, ny=9, hs=2):
    return ["run-cccc-n46", "--N", str(N), *PHYS, "--b", b, "--nx", str(nx), "--ny", str(ny), "--half-steps", str(hs)]


def record(name, ok, detail):
    results[name] = {"pass": bool(ok), "detail": detail}
    print(("PASS " if ok else "FAIL ") + name, flush=True)


def main():
    EV.mkdir(exist_ok=True)
    CLI.mkdir(exist_ok=True)
    STOP, BEFORE = "STOPPED_RANK_GUARD_UNCERTIFIED", "STOP_BEFORE_QR_S1"

    # ---- API-level, call order
    r = api("none", "")
    k = kinds(r)
    record("A0 baseline order: QR(T), s1 solve, RN-5, QR(V), RT-7 immediately, s2 solve", r["status"] == "COMPLETED_FIXED_HALF_STEPS"
           and k == ["qr", "solve_given1", "solve_bc1", "rn5", "qr", "rt7", "solve_given2", "solve_bc2"]
           and r["events"][4][1] == [8, 4] and r["field_from"] == "s2", {"events": k, "status": r["status"]})
    r = api("dep", "dep")
    record("A1 proven dependence stops before QR (no RN-5, no 2nd QR)", r["status"] == BEFORE and kinds(r).count("qr") == 1
           and "rn5" not in kinds(r) and r["steps"][1]["qr_executed"] is False and r["steps"][1]["G_none"] and r["field_from"] == "s1",
           {"events": kinds(r), "status": r["status"], "step2": r["steps"][1]})
    r = api("rn5_refuse", "rn5_refuse")
    record("A2 RN-5 refusal stops before QR", r["status"] == STOP and kinds(r) == ["qr", "solve_given1", "solve_bc1", "rn5"]
           and r["steps"][1]["qr_executed"] is False and r["steps"][1]["G_none"] and r["field_from"] == "s1" and r["completed"] == 1,
           {"events": kinds(r), "status": r["status"], "reason": r["reason"]})
    for nm, nx, ny in (("A3a grid 33x33", 33, 33), ("A3b grid 9x33", 9, 33), ("A3c grid 33x9", 33, 9)):
        r = api(nm.split()[0], "", nx=nx, ny=ny)
        record(f"{nm}: > 32 stops before RN-5 and QR, s1 diagnostic", r["status"] == STOP and kinds(r) == ["qr", "solve_given1", "solve_bc1"]
               and r["steps"][1]["qr_executed"] is False and r["field_from"] == "s1" and r["completed"] == 1, {"events": kinds(r), "reason": r["reason"]})
    r = api("rt7_refuse", "rt7_refuse")
    s2 = r["steps"][1]
    record("A4 RT-7 refusal after QR, before s2 assembly/solve; G2 not used", r["status"] == STOP
           and kinds(r) == ["qr", "solve_given1", "solve_bc1", "rn5", "qr", "rt7"] and s2["qr_executed"] is True and not s2["G_none"]
           and s2["A_none"] and s2["V_none"] and not s2["done"] and r["field_from"] == "s1" and r["completed"] == 1,
           {"events": kinds(r), "step2": s2})
    for nm, inj in (("A5a RN-5 certified but eps/sigma missing", "rn5_missing_eps"), ("A5b RN-5 eps/sigma as float (malformed)", "rn5_float_eps")):
        r = api(inj, inj)
        record(f"{nm}: RT-7 R7-2 refuses, no s2", r["status"] == STOP and "solve_given2" not in kinds(r) and r["field_from"] == "s1"
               and r["steps"][1]["rt7"] is False, {"events": kinds(r), "reason": r["reason"]})
    r = api("rn5_empty", "rn5_empty")
    record("A5c RN-5 record empty: no continuation (exception, no QR of V, no s2)", "exception" in r and kinds(r).count("qr") == 1
           and "solve_given2" not in kinds(r), {"events": kinds(r), "exception": r.get("exception")})
    for nm, inj, st in (("A6a C-3+ premise failure at s2", "premise_s2", "UNRESOLVED_C3PLUS_PREMISE"),
                        ("A6b nonfinite s2 solution", "nonfinite_s2", "ERROR_NONFINITE"),
                        ("A6c factorization failure at s2 (real Cholesky on -A)", "factor_s2", "PROTECTIVE_STOP_FACTORIZATION")):
        r = api(inj, inj)
        record(f"{nm} -> {st}, s2 not completed", r["status"] == st and r["completed"] == 1 and not r["steps"][1]["done"] and r["field_from"] == "s1",
               {"events": kinds(r), "status": r["status"], "reason": r["reason"]})
    for hs in (3, 8):
        r = api(f"hs{hs}", "", hs=hs)
        record(f"A7 half_steps={hs}: stop before the QR of s3; s2 diagnostic", r["status"] == STOP and len(r["steps"]) == 3 and r["completed"] == 2
               and kinds(r).count("qr") == 2 and kinds(r).count("solve_given2") == 1 and kinds(r)[-1] == "solve_bc2"
               and r["steps"][2]["qr_executed"] is False and r["field_from"] == "s2", {"events": kinds(r), "step3": r["steps"][2]})

    # ---- CLI-level files and exit codes
    c = cli("ok", "", n46_args())
    record("C0 CLI s2 success: exit 0, field_research.csv", c["exit"] == 0 and c["files"] == ["factors.csv", "field_research.csv", "history.csv", "result.json"], c["files"])
    for nm, inj, extra in (("C1 dependence", "dep", {}), ("C2 RN-5 refusal", "rn5_refuse", {}), ("C3 grid 33", "", {"nx": 33, "ny": 33}),
                           ("C4 RT-7 refusal", "rt7_refuse", {}), ("C5 half_steps 3", "", {"hs": 3})):
        c = cli(nm.split()[0], inj, n46_args(**extra))
        out = c["out"]
        detail = {"exit": c["exit"], "files": c["files"]}
        ok = c["exit"] == 6 and "field_diagnostic.csv" in c["files"] and "field_research.csv" not in c["files"]
        if ok:
            doc = json.loads((out / "result.json").read_text(encoding="utf-8"))
            head = (out / "field_diagnostic.csv").read_text(encoding="utf-8").splitlines()[0]
            want_s = 2 if "half_steps" in nm else 1
            detail.update(status=doc["status"], completed=doc["half_steps_completed"], field_comment=head, field_half_step=doc["field"]["half_step"])
            ok = doc["field"]["half_step"] == want_s and doc["half_steps_completed"] == want_s and "DIAGNOSTIC" in head and f"s = {want_s}" in head
            if nm.startswith("C4"):
                h2 = doc["rank"]["half_step_2"]
                fac = (out / "factors.csv").read_text(encoding="utf-8").splitlines()
                roles2 = sorted({ln.split(",")[2] for ln in fac if ln.startswith("2,")})
                hist = (out / "history.csv").read_text(encoding="utf-8").splitlines()
                detail.update(h2_done=h2.get("done"), h2_qr=h2.get("qr_executed"), h2_rt7=h2["rt7"]["passed"], h2_reason=h2.get("reason"),
                              s2_roles_in_factors=roles2, factors_comment=fac[0], factors_note=doc["factors"]["note"],
                              history_row2_status=hist[3].split(",")[3] if len(hist) > 3 else None)
                ok = ok and h2.get("done") is False and h2["rt7"]["passed"] is False and "V" not in roles2 and "A" not in roles2
        record(f"{nm}: exit 6, diagnostic field of the last completed half-step only", ok, detail)

    # ---- transactional replacement on one folder
    T = CLI / "trans"
    if T.exists():
        shutil.rmtree(T)
    c = cli("trans_ok", "", n46_args(), out=T)
    base = {f: sha(T / f) for f in c["files"]}
    c2 = cli("trans_rn5", "rn5_refuse", n46_args() + ["--overwrite"], out=T, fresh=False)
    record("T1 stop run with --overwrite replaces the set: no stale field_research.csv", c2["exit"] == 6
           and c2["files"] == ["factors.csv", "field_diagnostic.csv", "history.csv", "result.json"], c2["files"])
    snap = {f: sha(T / f) for f in c2["files"]}
    checks = [("T2 outside domain N=7 keeps earlier files", "", n46_args(N=7) + ["--overwrite"], 2),
              ("T3 existing results without --overwrite: refused, unchanged", "", n46_args(), 2),
              ("T4 Ctrl+C (KeyboardInterrupt) exit 130, unchanged", "interrupt", n46_args() + ["--overwrite"], 130),
              ("T5 ERROR_NONFINITE at s2 keeps earlier files", "nonfinite_s2", n46_args() + ["--overwrite"], 3)]
    for nm, inj, args, code in checks:
        c3 = cli(nm.split()[0], inj, args, out=T, fresh=False)
        now = {f.name: sha(f) for f in T.iterdir()}
        record(nm, c3["exit"] == code and now == snap, {"exit": c3["exit"], "unchanged": now == snap, "stderr": c3["stderr_tail"][-300:]})
    c4 = cli("trans_ok2", "", n46_args() + ["--overwrite"], out=T, fresh=False)
    record("T6 success with --overwrite after a stop: field_research.csv, no diagnostic", c4["exit"] == 0
           and c4["files"] == ["factors.csv", "field_research.csv", "history.csv", "result.json"]
           and all(sha(T / f) == base[f] for f in ("factors.csv", "history.csv", "field_research.csv")), c4["files"])

    # ---- unchanged interfaces: parent 0551141 vs 754dcfe
    if PAR.exists():
        shutil.rmtree(PAR)
    PAR.mkdir()
    arch = subprocess.run(["git", "-C", str(QA), "archive", "0551141bc67c1e1b12c0cd7eba2028d89e56607b", "src"], capture_output=True, check=True).stdout
    (S / "parent_src.tar").write_bytes(arch)
    subprocess.run(["tar", "-xf", str(S / "parent_src.tar"), "-C", str(PAR)], check=True)
    (S / "parent_src.tar").unlink()
    cmds = {
        "n1_hs3": ["run-cccc-n1", *PHYS, "--b", "1", "--nx", "12", "--ny", "12", "--half-steps", "3"],
        "n23_N3_hs2": ["run-cccc-n23", "--N", "3", *PHYS, "--b", "1", "--nx", "12", "--ny", "12", "--half-steps", "2"],
        "n23_N2_hs3": ["run-cccc-n23", "--N", "2", *PHYS, "--b", "0.8", "--nx", "10", "--ny", "12", "--half-steps", "3"],
        "ssss_N2": ["run-ssss", "--N", "2", *PHYS, "--b", "1", "--nx", "16", "--ny", "16"],
        "ssss_N1": ["run-ssss", "--N", "1", *PHYS, "--b", "1", "--nx", "16", "--ny", "16"],
        "n46_hs1_C4": n46_args(hs=1),
        "n46_hs1_40": n46_args(N=5, nx=40, ny=40, hs=1),
        "n23_refuse_N4": ["run-cccc-n23", "--N", "4", *PHYS, "--b", "1", "--nx", "12", "--ny", "12", "--half-steps", "2"],
        "ssss_bc_CCCC": ["run-ssss", "--N", "2", "--bc", "CCCC", *PHYS, "--b", "1", "--nx", "16", "--ny", "16"],
    }
    cmp = {}
    for name, args in cmds.items():
        outs = {}
        for tag, env in (("parent", {**ENV, "PYTHONPATH": str(PAR / "src")}), ("commit", ENV)):
            out = CMP / tag / name
            if out.exists():
                shutil.rmtree(out)
            pyflags = [] if tag == "parent" else ["-I"]
            p = subprocess.run([PY, *pyflags, "-c", "import sys,kkvim; from kkvim.cli import main; print(kkvim.__file__, file=sys.stderr); sys.exit(main(sys.argv[1:]))",
                                *args, "--out", str(out)], cwd=S, capture_output=True, text=True, encoding="utf-8", env=env)
            files = {f.name: f for f in out.iterdir()} if out.exists() else {}
            outs[tag] = {"exit": p.returncode, "files": files, "module": p.stderr.splitlines()[0] if p.stderr else "", "stderr": p.stderr}
        P, C = outs["parent"], outs["commit"]
        same_files = sorted(P["files"]) == sorted(C["files"])
        bitwise = {f: sha(P["files"][f]) == sha(C["files"][f]) for f in P["files"] if f != "result.json" and f in C["files"]}
        rj = None
        if "result.json" in P["files"] and "result.json" in C["files"]:
            a = json.loads(P["files"]["result.json"].read_text(encoding="utf-8"))
            b = json.loads(C["files"]["result.json"].read_text(encoding="utf-8"))
            diff = sorted(k for k in set(a) | set(b) if k not in ("program", "provenance") and a.get(k) != b.get(k))
            rj = diff
        cmp[name] = {"exit": [P["exit"], C["exit"]], "same_file_set": same_files, "bitwise": bitwise, "result_json_diff_keys": rj,
                     "parent_module": P["module"], "commit_module": C["module"],
                     "stderr_equal": P["stderr"].splitlines()[1:] == C["stderr"].splitlines()[1:]}
        print(name, cmp[name]["exit"], same_files, bitwise, rj, cmp[name]["stderr_equal"], flush=True)
    results["interfaces_parent_vs_commit"] = cmp
    (S / "s2_safeguards.json").write_text(json.dumps(results, ensure_ascii=False, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
