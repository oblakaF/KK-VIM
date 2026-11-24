"""D-059 corrective pass, B-1, B-3, B-4 and the resource evidence: real child processes with injected failures, the real parsing of their
results, the real restart. No D1 row is computed: the campaign runs on non-D1 grids, the real-run path is stopped before the first unit.
No production source and no frozen data is modified: the faults live in child scripts under the test's temporary directory."""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = ROOT / "scripts" / "r5_cccc_q4_d1_campaign.py"
CONTROLS = ROOT / "scripts" / "r5_cccc_q4_d1_ready_controls.py"


def _load(path: Path, name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


c = _load(CAMPAIGN, "_d1_campaign_test")
rd = c.rd()
pl = c.pl()
pf = c.pf()


@pytest.fixture(autouse=True)
def _fresh_gate(monkeypatch):
    monkeypatch.delenv(pl.CAMPAIGN_AUTH_ENV, raising=False)
    monkeypatch.delenv(pl.STAGE_LOG_ENV, raising=False)
    pl._GATE_CACHE.clear()
    yield
    pl._GATE_CACHE.clear()


@pytest.fixture
def dry_out():
    out = ROOT / "scratch" / f"test_d1_faults_{os.getpid()}"
    shutil.rmtree(out, ignore_errors=True)
    yield out
    shutil.rmtree(out, ignore_errors=True)


def _plan():
    return [rd._row("A", "r=1", "1", 1, 10, 12, 1.0, 1.0, 1, s) for s in (1, 2)]


# --- B-1: the controls child, read whatever its exit code was ------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def real_controls():
    """The real controls tool once, in a real child: the valid report every case below starts from."""
    out = ROOT / "scratch" / f"test_d1_real_controls_{os.getpid()}"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    p = subprocess.run(c._controls_command(out), cwd=ROOT, capture_output=True, text=True, env=c.child_env())
    assert p.returncode == 0, p.stderr[-800:]
    yield json.loads((out / "controls_results.json").read_text(encoding="utf-8"))
    shutil.rmtree(out, ignore_errors=True)


FAKE_CONTROLS = """
    import json, sys
    src, mode, rc, out = sys.argv[1:5]
    res = json.load(open(src, encoding="utf-8"))
    path = out + "/controls_results.json"
    if mode == "v_fail":
        res["control_V"][0].update(verdict="UNEXPECTED ['status']", match=False)
        res["summary"]["control_V"]["match"] -= 1
        res["summary"]["control_V"]["mismatch"] = [res["control_V"][0]["id"]]
        res["all_match"] = False
    elif mode == "v_empty":
        res["control_V"][0].update(verdict="UNRESOLVED (empty control)", match=False)
        res["summary"]["control_V"]["match"] -= 1
        res["all_match"] = False
    elif mode == "other_control_fails":
        res["summary"]["control_B"]["mismatch"] = ["x"]
        res["all_match"] = False
    elif mode == "all_match_false":
        res["all_match"] = False
    elif mode == "other_expectations":
        res["expected_sha256"] = "0" * 64
    elif mode == "case_missing":
        res["control_V"].pop()
    elif mode == "summary_disagrees":
        res["summary"]["control_V"]["match"] = 0
    elif mode == "other_tools":
        res["provenance"]["tool_blobs"]["scripts/r5_cccc_q4_d1_ready.py"] = "0" * 40
    elif mode == "no_verdict":
        del res["control_V"][0]["verdict"]
    if mode == "missing":
        pass
    elif mode == "bad_json":
        open(path, "w").write("{not json")
    else:
        json.dump(res, open(path, "w", encoding="utf-8"))
    if mode == "crash":
        raise RuntimeError("injected crash after the results")
    sys.exit(int(rc))
"""


def _fake_command(monkeypatch, tmp_path, real, mode, rc):
    src, script = tmp_path / "real.json", tmp_path / "fake_controls.py"
    src.write_text(json.dumps(real), encoding="utf-8")
    script.write_text(textwrap.dedent(FAKE_CONTROLS), encoding="utf-8")
    monkeypatch.setattr(c, "_controls_command", lambda cdir: [sys.executable, str(script), str(src), mode, str(rc), str(cdir)])


def _prerun(work):
    return c.prerun_checks(rd.row_plan(), c.frozen_manifest(), work)


VALID_CASES = [("pass", 0, True, False), ("v_fail", 1, False, True), ("v_empty", 1, False, False), ("other_control_fails", 1, False, False)]
INVALID_CASES = [("missing", 1, "no results"), ("bad_json", 1, "not JSON"), ("crash", 0, "do not agree"), ("pass", 1, "do not agree"),
                 ("all_match_false", 0, "do not agree"), ("pass", 139, "exited with 139"), ("other_expectations", 0, "other expectations"),
                 ("case_missing", 0, "exactly the cases"), ("summary_disagrees", 0, "does not agree"), ("other_tools", 0, "other tools"),
                 ("no_verdict", 0, "exactly the cases")]


@pytest.mark.parametrize("mode, rc, ready, v_fail", VALID_CASES)
def test_valid_report_is_read_whatever_the_exit_code(mode, rc, ready, v_fail, monkeypatch, tmp_path, real_controls):
    _fake_command(monkeypatch, tmp_path, real_controls, mode, rc)
    r = _prerun(tmp_path / "w")
    assert r["controls_state"] == c.CONTROLS_VALID and r["controls_returncode"] == rc and r["tool_defects"] == [], r
    assert r["ready_controls_all_match"] is ready and r["control_v_fail"] is v_fail


@pytest.mark.parametrize("mode, rc, reason", INVALID_CASES)
def test_unreadable_or_inconsistent_controls_are_a_tool_defect(mode, rc, reason, monkeypatch, tmp_path, real_controls):
    _fake_command(monkeypatch, tmp_path, real_controls, mode, rc)
    r = _prerun(tmp_path / "w")
    assert r["controls_state"] == c.CONTROLS_INVALID and r["ready_controls_all_match"] is False and r["control_v_fail"] is False
    assert any(reason in x for x in r["tool_defects"]), r


def test_results_of_an_earlier_attempt_are_never_read(monkeypatch, tmp_path, real_controls):
    cdir = tmp_path / "w" / "controls"
    cdir.mkdir(parents=True)
    (cdir / "controls_results.json").write_text(json.dumps(real_controls), encoding="utf-8")
    _fake_command(monkeypatch, tmp_path, real_controls, "missing", 1)
    assert _prerun(tmp_path / "w")["controls_state"] == c.CONTROLS_INVALID


def _authorised_real_run(monkeypatch):
    """The real pre-run on the real plan; the gate and the machine are granted here, and a started unit fails the test: no D1 row."""
    monkeypatch.setattr(pl, "campaign_gate", lambda *a, **k: {"authorized": True, "state": "AUTHORIZED", "reasons": [], "approved": {"code": {}, "environment": {}}})
    monkeypatch.setattr(c, "platform_defects", lambda: [])

    def no_unit(*a, **k):
        raise AssertionError("a unit of the campaign was started")
    monkeypatch.setattr(c, "run_unit", no_unit)


def test_control_v_fail_of_a_real_child_reaches_blocked_campaign(monkeypatch, tmp_path, dry_out):
    """Decisive: the real controls tool in a real child, control V really failing inside it, exit code 1, the real pre-run, the verdict."""
    script = tmp_path / "failing_controls.py"
    script.write_text(textwrap.dedent(f"""
        import importlib.util, sys
        spec = importlib.util.spec_from_file_location("ct", {str(CONTROLS)!r})
        ct = importlib.util.module_from_spec(spec)
        sys.modules["ct"] = ct
        spec.loader.exec_module(ct)
        real = ct.control_v
        def failing(exp, journals, paired, work, clean):
            real_ok = ct._clean_ok
            ct._clean_ok = lambda r, E: (False, ["injected into control V"])
            try:
                return real(exp, journals, paired, work, clean)
            finally:
                ct._clean_ok = real_ok
        ct.control_v = failing
        sys.exit(ct.main(["run", "--out", sys.argv[1]]))
    """), encoding="utf-8")
    monkeypatch.setattr(c, "_controls_command", lambda cdir: [sys.executable, str(script), str(cdir)])
    _authorised_real_run(monkeypatch)
    doc = c.run_campaign(dry_out)
    assert doc["prerun"]["controls_returncode"] == 1 and doc["prerun"]["controls_state"] == c.CONTROLS_VALID
    assert doc["prerun"]["control_v_fail"] is True and doc["state"] == "BLOCKED_BEFORE_THE_RUN"
    assert doc["verdict"]["verdict"] == "BLOCKED_CAMPAIGN" and "FAIL of the positive control V (protocol 6)" in doc["verdict"]["classifier"]["block_reasons"]
    assert not (dry_out / "units").exists()


@pytest.mark.parametrize("mode, rc", [("missing", 1), ("bad_json", 1), ("pass", 139)])
def test_unreadable_control_never_proceeds_and_repair_runs_it_again(mode, rc, monkeypatch, tmp_path, dry_out, real_controls):
    _fake_command(monkeypatch, tmp_path, real_controls, mode, rc)
    _authorised_real_run(monkeypatch)
    doc = c.run_campaign(dry_out)
    assert doc["state"].startswith("NOT_STARTED_CONTROLS_INVALID") and doc["verdict"]["verdict"] == rd.VERDICT_NOT_ISSUED and doc["reasons"]
    assert not (dry_out / "units").exists()
    _fake_command(monkeypatch, tmp_path, real_controls, "pass", 0)  # after the repair: not reused, read again; the run goes on to its first unit
    with pytest.raises(AssertionError, match="a unit of the campaign was started"):
        c.run_campaign(dry_out)


def test_v_empty_does_not_start_and_is_not_a_fail_of_v(monkeypatch, tmp_path, dry_out, real_controls):
    _fake_command(monkeypatch, tmp_path, real_controls, "v_empty", 1)
    _authorised_real_run(monkeypatch)
    doc = c.run_campaign(dry_out)
    assert doc["state"].startswith("NOT_STARTED_CONTROLS_MISMATCH") and "verdict" not in doc


def test_mismatched_input_hashes_reach_blocked_campaign(monkeypatch, tmp_path, dry_out, real_controls):
    _fake_command(monkeypatch, tmp_path, real_controls, "pass", 0)
    _authorised_real_run(monkeypatch)
    frozen = json.loads(json.dumps(c.frozen_manifest()))
    frozen["d057"]["input_hashes_s1"][0]["given_in_sha"] = "0" * 64
    monkeypatch.setattr(c, "frozen_manifest", lambda: frozen)
    monkeypatch.setattr(c, "plan_defects", lambda *a: [])
    doc = c.run_campaign(dry_out)
    assert doc["state"] == "BLOCKED_BEFORE_THE_RUN" and doc["verdict"]["verdict"] == "BLOCKED_CAMPAIGN"
    assert any("input hash mismatch" in x for x in doc["verdict"]["classifier"]["block_reasons"])


def test_a_pr48_harness_that_crashes_is_a_tool_defect(monkeypatch, tmp_path, real_controls):
    _fake_command(monkeypatch, tmp_path, real_controls, "pass", 0)

    def boom():
        raise OSError("injected")
    monkeypatch.setattr(pf, "run_controls", boom)
    r = _prerun(tmp_path / "w")
    assert r["pr48_controls_all_match"] is False and any("PR #48" in x for x in r["tool_defects"])


# --- B-3: the environment --------------------------------------------------------------------------------------------------------------------

def _env():
    return {"sys_platform": "win32", "platform": "Windows-10", "machine": "AMD64", "processor": "Intel", "python": "3.13.15", "numpy": "2.5.4",
            "scipy": "1.18.1", "numpy_blas": {"name": "scipy-openblas", "version": "0.3"}, "scipy_blas": {"name": "scipy-openblas", "version": "0.3"},
            "numpy_blas_library_sha256": "a", "scipy_blas_library_sha256": "b", "blas_threads": {"numpy": 1, "scipy": 1},
            "thread_env": {k: "1" for k in pl.THREAD_VARS}, "total_ram_bytes": 8, "node_sha256": "n"}


@pytest.mark.parametrize("field, value", [("numpy", "2.0.0"), ("scipy", "1.0.0"), ("python", "3.14.8"), ("processor", "other"), ("sys_platform", "linux"),
                                          ("total_ram_bytes", 4), ("node_sha256", "m"), ("numpy_blas", {"name": "mkl", "version": "1"}),
                                          ("scipy_blas_library_sha256", "c"), ("blas_threads", {"numpy": 2, "scipy": 1}),
                                          ("thread_env", {k: "4" for k in pl.THREAD_VARS})])
def test_a_mismatch_is_a_defect_naming_the_field(field, value):
    assert pl.environment_defects(_env(), _env()) == []
    bad = pl.environment_defects({**_env(), field: value}, _env())
    assert len(bad) == 1 and bad[0].startswith(f"OD-6: {field} = ")


@pytest.mark.parametrize("field", ["numpy_blas", "scipy_blas", "numpy_blas_library_sha256", "blas_threads", "numpy"])
def test_unavailable_evidence_is_not_a_match(field):
    rec = {**_env(), field: None}
    assert any(f"{field} is unavailable" in x and "NOT VERIFIED" in x for x in pl.environment_defects(rec, _env()))
    assert any(f"{field} is not pinned" in x for x in pl.environment_defects(_env(), {**_env(), field: None}))
    assert pl.environment_defects(None, _env()) and pl.environment_defects(_env(), None)


def test_missing_blas_identity_of_a_library():
    class NoConfig:
        @staticmethod
        def show_config(mode="dicts"):
            return {"Build Dependencies": {"blas": {"name": "x"}}}

    assert pl._build_identity(NoConfig) is None and pl._build_identity(object()) is None
    assert pl.environment_defects({**_env(), "numpy_blas": pl._build_identity(NoConfig)}, _env())


def test_real_record_of_a_subprocess_and_its_threads(tmp_path):
    script = tmp_path / "rec.py"
    script.write_text(textwrap.dedent(f"""
        import importlib.util, json, os, sys
        spec = importlib.util.spec_from_file_location("p", {str(ROOT / pl.SELF)!r})
        p = importlib.util.module_from_spec(spec)
        sys.modules["p"] = p
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        spec.loader.exec_module(p)
        print(json.dumps(p.environment_record()))
    """), encoding="utf-8")
    p = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, env=c.child_env())
    rec = json.loads(p.stdout.strip().splitlines()[-1])
    assert sorted(rec) == sorted(pl.ENVIRONMENT_KEYS)
    assert rec["blas_threads"] == {"numpy": 1, "scipy": 1} and rec["numpy_blas"]["name"] and rec["scipy_blas"]["name"]
    assert len(rec["numpy_blas_library_sha256"]) == len(rec["scipy_blas_library_sha256"]) == 64
    assert rec["numpy_blas_library_sha256"] != rec["scipy_blas_library_sha256"]  # two separate libraries


@pytest.mark.parametrize("field, value", [("numpy", "0.0.0"), ("scipy", "0.0.0"), ("numpy_blas", {"name": "other"})])
def test_a_child_not_on_the_expected_environment_stops_the_run(field, value, monkeypatch, dry_out):
    """A real child reports its real environment; the run began on another one: refused, nothing committed."""
    real = pl.environment_record
    monkeypatch.setattr(pl, "environment_record", lambda: {**real(), field: value})
    with pytest.raises(c.CampaignRefused, match=f"OD-6: {field} = "):
        c.run_campaign(dry_out, plan=_plan(), dry_run=True)
    assert not list((dry_out / "units").glob("*.json")) if (dry_out / "units").exists() else True
    assert (dry_out / "last_refusal.json").exists()


def test_the_real_gate_path_refuses_a_wrong_environment_and_no_remote(monkeypatch):
    monkeypatch.setattr(pl, "campaign_gate", lambda *a, **k: {"authorized": True, "state": "AUTHORIZED", "reasons": [], "approved": {"code": {}, "environment": None}})
    bad = c.platform_defects()
    assert any("approves no environment" in x for x in bad) and any("nothing is approved" in x for x in bad)


# --- B-4: tool failures against unresolved numerical results ------------------------------------------------------------------------------------

FAULT = """
    import json, os, runpy, subprocess, sys, tempfile, time

    def plant(p, what, row_s=None):
        # a failure inside a boundary of the tools: name:exception
        import numpy as np
        name, cls = what.rsplit(":", 1)
        exc = dict(TypeError=TypeError("injected"), MemoryError=MemoryError("injected"), KeyError=KeyError("injected"), ValueError=ValueError("injected"),
                   LinAlgError=np.linalg.LinAlgError("injected"), OverflowError=OverflowError("injected"))[cls]
        def bad(*a, **k):
            raise exc
        if name in ("op8", "traj"):
            p.op8().run_step = bad
        elif name == "be":
            p.be2().second_path = bad
        elif name == "seq" and row_s == 2:  # the predecessor s = 1 of the row s = 2
            real = p.op8().run_step
            def only(cfg, k, *a, **kw):
                if k == 1:
                    raise exc
                return real(cfg, k, *a, **kw)
            p.op8().run_step = only

    if sys.argv[1] == "_replay":  # a real replay process with the failure planted in it
        import importlib.util
        what, tool, sp = sys.argv[2:5]
        s = importlib.util.spec_from_file_location("_d1_pipeline_replay", tool)
        m = importlib.util.module_from_spec(s)
        sys.modules["_d1_pipeline_replay"] = m
        s.loader.exec_module(m)
        plant(m, what)
        sys.exit(m.main(["_row", sp]))

    mode, target, runner, spec = sys.argv[1:5]
    kind = json.load(open(spec, encoding="utf-8"))["kind"]
    if kind == target:
        if mode == "crash":
            raise RuntimeError("injected crash")
        if mode == "killed":
            os._exit(137)
        if mode == "refused":
            print("refused: injected refusal of the gate", file=sys.stderr)
            sys.exit(2)
        if mode == "exit2_no_refusal":
            sys.exit(2)
        if mode == "garbage":
            print("this is not JSON")
            sys.exit(0)
        if mode == "empty":
            sys.exit(0)
        if mode == "other_process":
            print(json.dumps({{"process": {{"pid": 1}}, "journal": "x"}}))
            sys.exit(0)
        if mode == "no_analysis":
            print(json.dumps({{"process": {{"pid": os.getpid()}}}}))
            sys.exit(0)
        if mode == "hang":
            child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"])
            open(os.environ["PIDFILE"], "w").write(str(child.pid))
            time.sleep(300)
        if mode == "orphan":
            child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"], stdin=subprocess.DEVNULL)
            open(os.environ["PIDFILE"], "w").write(str(child.pid))
    if mode.startswith("patch:") and kind == target:
        import importlib.util
        s = importlib.util.spec_from_file_location("_d1_campaign_child", runner)
        m = importlib.util.module_from_spec(s)
        sys.modules["_d1_campaign_child"] = m
        s.loader.exec_module(m)
        what = mode.split(":", 1)[1]
        p = m.pl()
        if what.startswith(("op8:", "seq:", "traj:", "be:")):
            plant(p, what, (json.load(open(spec, encoding="utf-8")).get("spec") or {{}}).get("s"))
        if what.startswith("replay:"):
            def child(spec_, timeout=1800):
                with tempfile.TemporaryDirectory() as td:
                    sp = os.path.join(td, "spec.json")
                    json.dump(spec_, open(sp, "w", encoding="utf-8"))
                    r = subprocess.run([sys.executable, __file__, "_replay", what.split(":", 1)[1], p.__file__, sp], capture_output=True, text=True)
                if r.returncode:
                    raise RuntimeError(f"child row failed ({{r.returncode}}): {{r.stderr[-3000:]}}")
                return json.loads(r.stdout.strip().splitlines()[-1])
            p.run_child = child
        if what == "replay_crash":
            def bad(*a, **k):
                raise RuntimeError("child row failed (1): injected")
            p.run_child = bad
        elif what == "replay_refused":
            def bad(*a, **k):
                raise p.D1Refused("refused: injected in the replay")
            p.run_child = bad
        elif what == "replay_other_code":
            real = p.run_child
            def other(*a, **k):
                r = real(*a, **k)
                r["loaded_code"]["outside"] = ["kkvim_reference: somewhere else"]
                return r
            p.run_child = other
        elif what in ("row_type_error", "row_linalg"):
            import numpy as np
            exc = TypeError("injected") if what == "row_type_error" else np.linalg.LinAlgError("injected")
            def bad(*a, **k):
                raise exc
            p._row_job = bad
        elif what in ("check_type_error", "check_float_error"):
            exc = TypeError("injected") if what == "check_type_error" else FloatingPointError("injected")
            def bad(*a, **k):
                raise exc
            m.br_reference = bad
        sys.argv = [runner, "_job", spec]
        sys.exit(m.main(["_job", spec]))
    sys.argv = [runner, "_job", spec]
    runpy.run_path(runner, run_name="__main__")
"""


def _inject(monkeypatch, tmp_path, mode, target):
    script = tmp_path / "fault.py"
    script.write_text(textwrap.dedent(FAULT.format()), encoding="utf-8")
    monkeypatch.setattr(c, "_child_command", lambda spec: [sys.executable, str(script), mode, target, str(CAMPAIGN), str(spec)])


def _clean(monkeypatch):
    monkeypatch.setattr(c, "_child_command", lambda spec: [sys.executable, str(CAMPAIGN), "_job", str(spec)])


def _units(out):
    return list((out / "units").glob("*.json")) if (out / "units").exists() else []


@pytest.mark.parametrize("mode, target, kind", [
    ("crash", "journal", "CHILD_CRASH"), ("killed", "journal", "CHILD_CRASH"), ("exit2_no_refusal", "journal", "CHILD_CRASH"),
    ("garbage", "journal", "MALFORMED_OUTPUT"), ("empty", "journal", "MALFORMED_OUTPUT"), ("other_process", "journal", "MALFORMED_OUTPUT"),
    ("crash", "row", "CHILD_CRASH"), ("killed", "row", "CHILD_CRASH"), ("garbage", "row", "MALFORMED_OUTPUT"), ("no_analysis", "row", "MALFORMED_OUTPUT"),
    ("crash", "second_paths", "CHILD_CRASH"), ("garbage", "second_paths", "MALFORMED_OUTPUT"),
    ("patch:replay_crash", "row", "REPLAY_FAILED"), ("patch:row_type_error", "row", c.INTERNAL_EXCEPTION),
    ("patch:check_type_error", "second_paths", c.INTERNAL_EXCEPTION)])
def test_tool_failure_stops_is_not_stored_and_restart_after_repair(mode, target, kind, monkeypatch, tmp_path, dry_out):
    _inject(monkeypatch, tmp_path, mode, target)
    doc = c.run_campaign(dry_out, plan=_plan(), dry_run=True)
    assert doc["state"] == c.STOPPED_TOOL_FAILURE and doc["failure"]["kind"] == kind, doc.get("failure")
    assert doc["verdict"]["verdict"] == rd.VERDICT_NOT_ISSUED and "rows" not in doc and not _units(dry_out)
    assert json.loads((dry_out / "campaign_result.json").read_text(encoding="utf-8"))["state"] == c.STOPPED_TOOL_FAILURE
    assert len(list((dry_out / "failures").glob("*.json"))) == 1
    _clean(monkeypatch)
    after = c.run_campaign(dry_out, plan=_plan(), dry_run=True)
    assert after["state"] == c.COMPLETED and {r["status"] for r in after["rows"]} == {"PASS"} and len(after["restarts"]) == 2


@pytest.mark.parametrize("mode, target", [("refused", "journal"), ("refused", "row"), ("refused", "second_paths"), ("patch:replay_refused", "row"),
                                          ("patch:replay_other_code", "row")])
def test_refusal_stops_the_run_before_anything_is_committed(mode, target, monkeypatch, tmp_path, dry_out):
    _inject(monkeypatch, tmp_path, mode, target)
    with pytest.raises(c.CampaignRefused):
        c.run_campaign(dry_out, plan=_plan(), dry_run=True)
    assert not _units(dry_out) and not (dry_out / "campaign_result.json").exists() and (dry_out / "last_refusal.json").exists()
    _clean(monkeypatch)
    assert c.run_campaign(dry_out, plan=_plan(), dry_run=True)["state"] == c.COMPLETED


def test_numerical_exception_stays_with_the_row_and_is_not_a_tool_defect(monkeypatch, tmp_path, dry_out):
    _inject(monkeypatch, tmp_path, "patch:row_linalg", "row")
    doc = c.run_campaign(dry_out, plan=_plan(), dry_run=True)
    assert doc["state"] == c.COMPLETED and {r["status"] for r in doc["rows"]} == {"UNRESOLVED"} and all(not r["tool_defects"] for r in doc["rows"])


def test_numerical_exception_of_a_check_stays_an_observation(monkeypatch, tmp_path, dry_out):
    _inject(monkeypatch, tmp_path, "patch:check_float_error", "second_paths")
    doc = c.run_campaign(dry_out, plan=_plan(), dry_run=True)
    assert doc["state"] == c.COMPLETED and {r["status"] for r in doc["rows"]} == {"PASS"}
    assert doc["second_path_coverage"][c.ACCURACY]["resolved"] == 0


# --- exceptions recorded instead of raised: OP-8, replay, second path of B_E ------------------------------------------------------------------

PATHS = [("op8", "row"), ("seq", "row"), ("be", "row"), ("replay:op8", "row"), ("replay:be", "row"), ("traj", "journal")]


def _be_subsets(plan):
    return {r["key"]: ["SP-BE"] for r in plan}


@pytest.mark.parametrize("cls", ["TypeError", "MemoryError", "KeyError", "ValueError"])
@pytest.mark.parametrize("path, target", PATHS)
def test_non_numerical_exception_of_each_path_stops_the_run(path, target, cls, monkeypatch, tmp_path, dry_out):
    """Each of them used to end as a completed row (UNRESOLVED, or PASS with SP-BE not performed) without a tool defect."""
    _inject(monkeypatch, tmp_path, f"patch:{path}:{cls}", target)
    plan = _plan()
    doc = c.run_campaign(dry_out, plan=plan, dry_run=True, subsets=_be_subsets(plan))
    assert doc["state"] == c.STOPPED_TOOL_FAILURE and doc["failure"]["kind"] == c.INTERNAL_EXCEPTION and cls in doc["failure"]["reason"], doc.get("failure")
    assert doc["verdict"]["verdict"] == rd.VERDICT_NOT_ISSUED and "rows" not in doc and not _units(dry_out)
    _clean(monkeypatch)
    after = c.run_campaign(dry_out, plan=plan, dry_run=True, subsets=_be_subsets(plan))
    assert after["state"] == c.COMPLETED and {r["status"] for r in after["rows"]} == {"PASS"} and after["second_path_coverage_complete"] is True


@pytest.mark.parametrize("path, target, cls, statuses, covered", [
    ("op8", "row", "LinAlgError", {"UNRESOLVED_CERTIFICATE"}, True), ("seq", "row", "OverflowError", {"PASS", "UNRESOLVED_CERTIFICATE"}, True),
    # only the first process fails: the replay computes B_E, so the values of the two processes differ and П-а is not established (protocol 4.2)
    ("be", "row", "OverflowError", {"UNRESOLVED_CERTIFICATE"}, False), ("be", "row", "LinAlgError", {"UNRESOLVED_CERTIFICATE"}, False),
    ("replay:be", "row", "OverflowError", {"UNRESOLVED_CERTIFICATE", "PASS"}, True), ("traj", "journal", "LinAlgError", {"UNRESOLVED"}, False)])
def test_numerical_exception_of_each_path_stays_with_the_row(path, target, cls, statuses, covered, monkeypatch, tmp_path, dry_out):
    _inject(monkeypatch, tmp_path, f"patch:{path}:{cls}", target)
    plan = _plan()
    doc = c.run_campaign(dry_out, plan=plan, dry_run=True, subsets=_be_subsets(plan))
    assert doc["state"] == c.COMPLETED and {r["status"] for r in doc["rows"]} <= statuses, doc.get("failure") or doc["rows"]
    assert all(not r["tool_defects"] for r in doc["rows"]) and len(_units(dry_out)) == 1
    if not covered:
        assert doc["second_path_coverage_complete"] is False


def test_failure_in_the_repeat_of_z4_is_seen(monkeypatch):
    repeat = {"row": None, "premises": {}, "status": "UNRESOLVED", "exception": None, "reproduction": None,
              "caught_exceptions": [{"where": "replay process: second path of B_E", "type": "TypeError", "message": "injected"}]}
    monkeypatch.setattr(pl, "analyse", lambda *a, **k: repeat)
    z4 = rd.z4_confirm({"kind": "journal", "journal": "x", "s": 1}, {"row": {}, "premises": {}})
    assert c._analysis_tool_failure({"z4": z4})[0] == c.INTERNAL_EXCEPTION
    repeat["caught_exceptions"] = [{"where": "x", "type": "OverflowError", "message": "m"}]
    assert c._analysis_tool_failure({"z4": rd.z4_confirm({"kind": "journal", "journal": "x", "s": 1}, {"row": {}, "premises": {}})}) is None
    repeat.update(caught_exceptions=[], reproduction={"bitwise_equal": None, "error": "child row failed (1)"})
    assert c._analysis_tool_failure({"z4": rd.z4_confirm({"kind": "journal", "journal": "x", "s": 1}, {"row": {}, "premises": {}})})[0] == "REPLAY_FAILED"


@pytest.mark.parametrize("status, kind", [("FAIL_TOOL: exception TypeError: x", "INTERNAL_EXCEPTION"), ("FAIL_TOOL: exception MemoryError: ", "INTERNAL_EXCEPTION"),
                                          ("FAIL_TOOL: only-C replacement not confirmed", "FAIL_TOOL"), ("FAIL_TOOL: exception LinAlgError: x", None),
                                          ("OK", None), ("STOP_ZERO_FIELD: computed factor exactly 0 (N = 1, I-4 undefined)", None)])
def test_status_of_a_journal(status, kind, tmp_path):
    (tmp_path / "journal.json").write_text(json.dumps({"meta": {}, "steps": [{"s": 1, "status": status}]}), encoding="utf-8")
    got = c._journal_tool_failure(tmp_path)
    assert (got[0] if got else None) == kind


def test_unreadable_journal_is_a_tool_failure(tmp_path):
    assert c._journal_tool_failure(tmp_path)[0] == "UNREADABLE_DATA"
    (tmp_path / "journal.json").write_text("{not json", encoding="utf-8")
    assert c._journal_tool_failure(tmp_path)[0] == "UNREADABLE_DATA"


def _gone(pid: int, seconds: float = 10) -> bool:
    end = time.time() + seconds
    while time.time() < end:
        if pid not in c._process_table():
            return True
        time.sleep(0.1)
    return False


def test_timeout_is_recorded_and_the_tree_is_stopped(monkeypatch, tmp_path, dry_out):
    pidfile = tmp_path / "pid"
    monkeypatch.setenv("PIDFILE", str(pidfile))
    monkeypatch.setattr(rd, "SAFETY_TIMEOUT_S", 8)
    _inject(monkeypatch, tmp_path, "hang", "journal")
    doc = c.run_campaign(dry_out, plan=_plan(), dry_run=True)
    assert doc["state"] == c.STOPPED_TOOL_FAILURE and doc["failure"]["kind"] == "TIMEOUT" and "killed" in doc["failure"]["reason"]
    assert _gone(int(pidfile.read_text())) and not _units(dry_out)
    rec = json.loads(next((dry_out / "failures").glob("*.json")).read_text(encoding="utf-8"))
    assert rec["resources"]["safety_timeout"] is True and rec["resources"]["stopped_by_monitor"] is True and rec["resources"]["orphans_surviving"] == []


def test_orphan_of_a_job_is_a_tool_failure_and_is_cleaned_up(monkeypatch, tmp_path, dry_out):
    pidfile = tmp_path / "pid"
    monkeypatch.setenv("PIDFILE", str(pidfile))
    _inject(monkeypatch, tmp_path, "orphan", "journal")
    doc = c.run_campaign(dry_out, plan=_plan(), dry_run=True)
    assert doc["state"] == c.STOPPED_TOOL_FAILURE and doc["failure"]["kind"] == "ORPHAN_PROCESSES", doc.get("failure")
    assert _gone(int(pidfile.read_text())) and not _units(dry_out)


@pytest.mark.skipif(sys.platform != "win32", reason="job object: Windows")
def test_killed_runner_takes_its_job_tree_with_it(tmp_path):
    """Interruption of the runner itself: the job object ends the tree (kill on close), no orphan computes on."""
    pidfile = tmp_path / "pid"
    (tmp_path / "grand.py").write_text(textwrap.dedent(f"""
        import os, subprocess, sys, time
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"])
        open({str(pidfile)!r}, "w").write(str(child.pid))
        time.sleep(300)
    """), encoding="utf-8")
    (tmp_path / "runner.py").write_text(textwrap.dedent(f"""
        import importlib.util, subprocess, sys
        spec = importlib.util.spec_from_file_location("r", {str(CAMPAIGN)!r})
        r = importlib.util.module_from_spec(spec)
        sys.modules["r"] = r
        spec.loader.exec_module(r)
        p = subprocess.Popen([sys.executable, {str(tmp_path / "grand.py")!r}])
        r.monitor_job(p, 0, 1e9, r.Path({str(tmp_path / "none.jsonl")!r}))
    """), encoding="utf-8")
    run = subprocess.Popen([sys.executable, str(tmp_path / "runner.py")], env=c.child_env())
    end = time.time() + 60
    while not pidfile.exists() and time.time() < end:
        time.sleep(0.1)
    grand = int(pidfile.read_text())
    assert grand in c._process_table()
    subprocess.run(["taskkill", "/PID", str(run.pid), "/F", "/T"], capture_output=True)
    run.wait()
    assert _gone(grand)


# --- B-4: stored blocks ------------------------------------------------------------------------------------------------------------------------

def _interrupted(out):
    with pytest.raises(c.CampaignInterrupted):
        c.run_campaign(out, plan=_plan() + [rd._row("A", "r=1", "1", 1, 12, 14, 1.0, 1.0, 1, 1)], dry_run=True, stop_after_jobs=2)
    return next((out / "units").glob("*.json"))


def _commit_edit(out, path, edit):
    """A stored unit changed AND its ledger entry rewritten: the ledger alone would not see it."""
    doc = json.loads(path.read_text(encoding="utf-8"))
    edit(doc)
    path.write_text(json.dumps(doc), encoding="utf-8")
    led = json.loads((out / "ledger.json").read_text(encoding="utf-8"))
    led[path.stem] = c._sha_file(path)
    (out / "ledger.json").write_text(json.dumps(led), encoding="utf-8")


CORRUPT = {
    "status changed": (lambda d: d["rows"][0].update(status="FAIL_COMPUTATION"), "the classifier gives"),
    "other unit": (lambda d: d.update(unit="x"), "the result of"),
    "other run": (lambda d: d.update(run_provenance_sha256="0" * 64), "another provenance"),
    "row removed": (lambda d: d["rows"].pop(), "not the rows of the plan"),
    "input removed": (lambda d: d["rows"][0].pop("input"), "not readable"),
    "failure inside": (lambda d: d["rows"][0]["checks"][c.ACCURACY].update(failure_kind=c.INTERNAL_EXCEPTION), "failure of the tools"),
    "exception of the tools recorded in the row": (lambda d: d["rows"][0]["analysis"].update(
        caught_exceptions=[{"where": "second path of B_E", "type": "TypeError", "message": "x"}]), "the analysis holds a failure of the tools"),
}


@pytest.mark.parametrize("name", sorted(CORRUPT))
def test_corrupted_block_with_a_consistent_ledger_is_refused(name, dry_out):
    path = _interrupted(dry_out)
    edit, reason = CORRUPT[name]
    _commit_edit(dry_out, path, edit)
    with pytest.raises(c.CampaignRefused, match=reason):
        c.run_campaign(dry_out, plan=_plan() + [rd._row("A", "r=1", "1", 1, 12, 14, 1.0, 1.0, 1, 1)], dry_run=True)


def test_truncated_block_refused_and_block_outside_the_ledger_computed_again(dry_out):
    plan = _plan() + [rd._row("A", "r=1", "1", 1, 12, 14, 1.0, 1.0, 1, 1)]
    path = _interrupted(dry_out)
    saved = path.read_bytes()
    path.write_bytes(saved[:100])
    with pytest.raises(c.CampaignRefused, match="differ from the ledger"):
        c.run_campaign(dry_out, plan=plan, dry_run=True)
    path.write_bytes(saved)
    (dry_out / "ledger.json").write_text("{}", encoding="utf-8")  # written, not committed: interrupted between the two
    assert c.run_campaign(dry_out, plan=plan, dry_run=True)["state"] == c.COMPLETED


def test_changed_environment_refuses_the_restart(monkeypatch, dry_out):
    _interrupted(dry_out)
    real = pl.environment_record
    monkeypatch.setattr(pl, "environment_record", lambda: {**real(), "scipy": "0.0.0"})
    with pytest.raises(c.CampaignRefused, match="restart refused: provenance or plan"):
        c.run_campaign(dry_out, plan=_plan() + [rd._row("A", "r=1", "1", 1, 12, 14, 1.0, 1.0, 1, 1)], dry_run=True)


def test_source_change_during_the_run_gives_no_result(monkeypatch, dry_out):
    real, calls = c._provenance, []

    def changing(*a):
        calls.append(1)
        out = real(*a)
        if len(calls) > 1:
            out["tool_blobs"] = {**out["tool_blobs"], c.HS_MODULE: "0" * 40}
        return out
    monkeypatch.setattr(c, "_provenance", changing)
    doc = c.run_campaign(dry_out, plan=_plan(), dry_run=True)
    assert doc["state"] == c.STOPPED_CHANGED and doc["verdict"]["verdict"] == rd.VERDICT_NOT_ISSUED and not _units(dry_out)
    assert any(c.HS_MODULE in x for x in doc["reasons"]) and "rows" not in doc


def test_authorisation_lost_during_a_real_run_is_seen(monkeypatch):
    """On the real repository the act does not exist: as after a withdrawal, the check of the run gives the reasons of the gate."""
    prov = c._provenance(False, None)
    bad = c._changed_since_start(prov, False, None)
    assert any(x.startswith("gate: ") for x in bad)


def test_a_stale_result_never_stands_for_a_stopped_attempt(monkeypatch, tmp_path, dry_out):
    assert c.run_campaign(dry_out, plan=_plan(), dry_run=True)["state"] == c.COMPLETED
    shutil.rmtree(dry_out / "units")
    (dry_out / "ledger.json").write_text("{}", encoding="utf-8")
    _inject(monkeypatch, tmp_path, "crash", "journal")
    assert c.run_campaign(dry_out, plan=_plan(), dry_run=True)["state"] == c.STOPPED_TOOL_FAILURE
    assert json.loads((dry_out / "campaign_result.json").read_text(encoding="utf-8"))["state"] == c.STOPPED_TOOL_FAILURE


@pytest.mark.parametrize("doc, code, shown", [
    ({"state": c.STOPPED_TOOL_FAILURE}, c.EXIT_NOT_COMPLETED, False),
    ({"state": "BLOCKED_BEFORE_THE_RUN", "verdict": {"verdict": "BLOCKED_CAMPAIGN"}}, c.EXIT_NOT_COMPLETED, False),
    ({"state": c.COMPLETED, "second_path_coverage_complete": True, "second_path_coverage_gaps": {}}, c.EXIT_OK, True),
    ({"state": c.COMPLETED, "second_path_coverage_complete": False, "verdict": {"verdict": "PASS"},
      "second_path_coverage_gaps": {"SP-C": {"not_established": ["a"], "contradictions": [], "selected_without_a_row": ["b"]}}}, c.EXIT_COVERAGE_INCOMPLETE, False),
    ({"state": c.COMPLETED}, c.EXIT_COVERAGE_INCOMPLETE, False)])
def test_exit_code_of_the_command_line_and_what_it_displays(doc, code, shown, monkeypatch, capsys):
    monkeypatch.setattr(c, "run_campaign", lambda out: doc)
    assert c.main(["run", "--out", "scratch/x"]) == code
    line = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert line["second_path_coverage_complete"] is shown and line["state"] == doc["state"]
    assert line["verdict"] == (doc.get("verdict") or {}).get("verdict")
    if doc.get("second_path_coverage_gaps"):
        assert line["second_path_coverage_gaps"] == {"SP-C": 2}


_GAPS = {"SP-RH": {"not_established": ["k1"], "contradictions": [], "selected_without_a_row": ["k2", "k3"]}}


@pytest.mark.parametrize("doc, code", [
    ({"state": c.COMPLETED, "verdict": {"verdict": "PASS_CANDIDATE"}, "second_path_coverage_complete": False,
      "second_path_coverage_gaps": _GAPS}, c.EXIT_COVERAGE_INCOMPLETE),
    ({"state": c.STOPPED_TOOL_FAILURE, "verdict": {"verdict": rd.VERDICT_NOT_ISSUED}}, c.EXIT_NOT_COMPLETED),
    ({"state": c.COMPLETED, "verdict": {"verdict": "PASS_CANDIDATE"}, "second_path_coverage_complete": True,
      "second_path_coverage_gaps": {}}, c.EXIT_OK),
    ({"state": c.COMPLETED, "verdict": {"verdict": "PASS_CANDIDATE"}}, c.EXIT_COVERAGE_INCOMPLETE),  # missing evidence: never success
    ({"state": c.STOPPED_TOOL_FAILURE, "second_path_coverage_complete": True}, c.EXIT_NOT_COMPLETED),  # the state comes first
    ({"state": "BLOCKED_BEFORE_THE_RUN", "verdict": {"verdict": "BLOCKED_CAMPAIGN"}}, c.EXIT_NOT_COMPLETED)])
def test_r1d_both_entry_points_give_the_same_exit_status(doc, code, monkeypatch, capsys):
    """R1-D: ready.py campaign returns campaign.py's exit_code(doc) for the same document and shows the coverage outcome; the document is
    produced once and neither re-evaluated nor completed (stubbed runner: nothing is computed, the real gate is not opened)."""
    calls = []
    monkeypatch.setattr(rd, "campaign", lambda out: calls.append(out) or doc)
    monkeypatch.setattr(rd, "verdict", lambda *a, **k: (_ for _ in ()).throw(AssertionError("the verdict was recalculated")))
    monkeypatch.setattr(c, "run_campaign", lambda out: doc)
    snapshot = json.dumps(doc, sort_keys=True)
    rc_ready = rd.main(["campaign", "--out", "scratch/never"])
    shown_ready = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    rc_camp = c.main(["run", "--out", "scratch/never"])
    shown_camp = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert rc_ready == rc_camp == code == c.exit_code(doc) and len(calls) == 1
    assert json.dumps(doc, sort_keys=True) == snapshot
    assert shown_ready["exit_code"] == code
    for k in ("state", "verdict", "second_path_coverage_complete"):
        assert shown_ready[k] == shown_camp[k], k
    assert shown_ready["verdict"] == (doc.get("verdict") or {}).get("verdict")
    assert shown_ready["second_path_coverage_complete"] is (doc.get("second_path_coverage_complete") is True)
    assert shown_ready["second_path_coverage_gaps"] == (doc.get("second_path_coverage_gaps") or {})
    assert {k: sum(map(len, v.values())) for k, v in shown_ready["second_path_coverage_gaps"].items()} == shown_camp["second_path_coverage_gaps"]


def test_r1d_a_refusal_of_the_runner_is_exit_refused_on_both_entry_points(monkeypatch, capsys):
    cm = rd._load(rd.CAMPAIGN_TOOL, "_d1_campaign")

    def refuse(out):
        raise cm.CampaignRefused("stub platform defect")
    monkeypatch.setattr(rd, "campaign", refuse)
    monkeypatch.setattr(c, "run_campaign", lambda out: (_ for _ in ()).throw(c.CampaignRefused("stub platform defect")))
    assert rd.main(["campaign", "--out", "scratch/never"]) == c.main(["run", "--out", "scratch/never"]) == c.EXIT_REFUSED
    err = capsys.readouterr().err
    assert err.count("refused: stub platform defect") == 2
    monkeypatch.setattr(rd, "campaign", lambda out: (_ for _ in ()).throw(RuntimeError("other")))
    with pytest.raises(RuntimeError, match="other"):  # any other failure is neither a refusal nor a result
        rd.main(["campaign", "--out", "scratch/never"])


@pytest.mark.parametrize("signal", [None, "D-059:" + "0" * 40])
def test_r1d_the_gate_still_refuses_before_the_runner_is_loaded(signal, monkeypatch, capsys):
    """The real gate (no owner act): both entry points refuse with 2; ready.py loads no campaign module; nothing is written."""
    if signal:
        monkeypatch.setenv(pl.CAMPAIGN_AUTH_ENV, signal)
    real = rd._load
    monkeypatch.setattr(rd, "_load", lambda rel, name: (_ for _ in ()).throw(AssertionError("runner loaded"))
                        if rel == rd.CAMPAIGN_TOOL else real(rel, name))
    assert rd.main(["campaign", "--out", "scratch/never"]) == 2
    assert "D1_START = BLOCKED" in capsys.readouterr().err
    assert c.main(["run", "--out", "scratch/never"]) == c.EXIT_REFUSED
    assert "D1_START = BLOCKED" in capsys.readouterr().err
    with pytest.raises(pl.D1Refused, match="D1_START = BLOCKED"):  # a child command validates the gate in its own process
        pl.guard_config(pl.op8().TrajectoryConfig(1.0, 14, 1.0, 6, 3, s_max=1))
    assert not (ROOT / "scratch" / "never").exists()


def test_only_complete_coverage_is_the_ordinary_success():
    assert {c.EXIT_OK, c.EXIT_REFUSED, c.EXIT_NOT_COMPLETED, c.EXIT_COVERAGE_INCOMPLETE} == {0, 2, 3, 4}


# --- resource evidence ---------------------------------------------------------------------------------------------------------------------------

ALLOC = """
    import time
    x = bytearray({n} * 2**20)
    for i in range(0, len(x), 4096):
        x[i] = 1
    time.sleep({sleep})
"""


def _spawn(tmp_path: Path, code: str):
    src = tmp_path / "child.py"
    src.write_text(textwrap.dedent(code), encoding="utf-8")
    return subprocess.Popen([sys.executable, str(src)], cwd=tmp_path)


def test_evidence_statuses_do_not_overstate():
    acc = {"total_processes": 3, "active_processes": 0, "peak_process_commit_bytes": 50, "peak_job_commit_bytes": 90}
    assert c.memory_evidence(40, 3, acc, 100, 0.2)["status"] == "SAMPLED_AND_OS_BOUNDED"
    assert c.memory_evidence(40, 3, acc, 100, 0.2)["compliance"] == "WITHIN_LIMIT_BY_THIS_EVIDENCE"
    e = c.memory_evidence(40, 2, acc, 100, 0.2)
    assert e["processes_not_sampled"] == 1 and "NOT_MEASURED" in e["reason"]
    assert c.memory_evidence(40, 3, None, 100, 0.2)["status"] == "NOT_MEASURED" and c.memory_evidence(40, 3, None, 100, 0.2)["compliance"] == "NOT_CLAIMED"
    assert c.memory_evidence(None, 0, acc, 100, 0.2)["status"] == "NOT_MEASURED"
    over = c.memory_evidence(40, 3, {**acc, "peak_process_commit_bytes": 200}, 100, 0.2)
    assert over["status"] == "COMMIT_OVER_LIMIT_RECORDED" and over["compliance"] == "NOT_CLAIMED"


@pytest.mark.skipif(sys.platform != "win32", reason="job object: Windows")
def test_short_lived_process_escapes_the_samples_not_the_os_high_water_mark(tmp_path):
    (tmp_path / "grand.py").write_text(textwrap.dedent(ALLOC.format(n=200, sleep=0)), encoding="utf-8")
    p = _spawn(tmp_path, f"""
        import subprocess, sys
        subprocess.run([sys.executable, {str(tmp_path / "grand.py")!r}])
    """)
    mon = c.monitor_job(p, 100 * 2**20, 1e9, tmp_path / "none.jsonl", poll_s=4)  # the poll is longer than the life of the process
    ev = mon["memory_evidence"]
    assert mon["violation"] is None and ev["sampled_max_process_peak_bytes"] < 100 * 2**20  # the samples alone would look compliant
    assert ev["processes_not_sampled"] >= 1 and ev["os_high_water"]["peak_process_commit_bytes"] > 200 * 2**20
    assert ev["status"] == "COMMIT_OVER_LIMIT_RECORDED" and ev["compliance"] == "NOT_CLAIMED"


def test_concurrent_processes_over_the_limit_together_are_recorded_not_stopped(tmp_path):
    (tmp_path / "grand.py").write_text(textwrap.dedent(ALLOC.format(n=70, sleep=2)), encoding="utf-8")
    p = _spawn(tmp_path, f"""
        import subprocess, sys
        ps = [subprocess.Popen([sys.executable, {str(tmp_path / "grand.py")!r}]) for _ in range(2)]
        [x.wait() for x in ps]
    """)
    limit = 100 * 2**20
    mon = c.monitor_job(p, limit, 1e9, tmp_path / "none.jsonl")
    assert mon["violation"] is None and mon["max_process_peak_bytes"] < limit < mon["max_concurrent_working_set_bytes"]
    assert mon["tree_measurement_over_limit"] is True and "owner decision I3" in mon["tree_measurement_note"]


# --- process identity: pid and creation time --------------------------------------------------------------------------------------------------

def test_members_by_identity_simulated_pid_reuse():
    t = {100: 10, 200: 20, 300: 30, 400: 5, 500: 50, 600: 60}
    when = t.get
    # real descendants, grandchild included
    known = {(100, 10)}
    assert c.tree_members({100: 1, 200: 100, 300: 200}, known, when) == {100: 10, 200: 20, 300: 30}
    assert known == {(100, 10), (200, 20), (300, 30)}
    # the pid of a member that ended now belongs to someone else: neither it nor its child is a member
    known = {(100, 7)}
    assert c.tree_members({100: 1, 200: 100}, known, when) == {}
    assert known == {(100, 7)}
    # the parent pid of an old process names a younger member by accident
    assert c.tree_members({100: 1, 400: 100}, {(100, 10)}, when) == {100: 10}
    # the parent has ended and its pid is free: nobody is adopted through it
    assert c.tree_members({200: 100}, {(100, 10)}, when) == {}
    # a process that no longer runs (no identity) is not a member, whatever the table says
    assert c.tree_members({100: 1, 200: 100}, {(100, 10)}, {100: 10}.get) == {100: 10}
    # the job object lists exactly its members, a dead intermediate does not lose them
    known = {(100, 10)}
    assert c.tree_members({500: 999, 600: 500}, known, when, {500: 50}) == {500: 50, 600: 60}
    assert (500, 50) in known and (600, 60) in known


def _unrelated(seconds=120):
    """A process of nobody's tree, with a large working set."""
    u = subprocess.Popen([sys.executable, "-c", f"import time; x = bytearray(150 * 2**20); [x.__setitem__(i, 1) for i in range(0, len(x), 4096)]; time.sleep({seconds})"])
    time.sleep(1.0)
    assert u.poll() is None
    return u


def test_reused_pid_is_not_measured_stopped_or_an_orphan(monkeypatch, tmp_path):
    """The unrelated process first looks like a young child of the job, as a member would; then its pid has the real, other creation time, as
    if the member ended and the system gave its pid to someone else."""
    u = _unrelated()
    try:
        root = _spawn(tmp_path, "import time; time.sleep(1.5)")
        table, when, created = c._process_table, c._creation_time, c._creation_time(root.pid)
        end = time.time() + 0.7

        def fake_table():
            out = table()
            if time.time() < end:
                out[u.pid] = root.pid
            return out

        monkeypatch.setattr(c, "_process_table", fake_table)
        monkeypatch.setattr(c, "_creation_time", lambda pid: created + 1 if pid == u.pid and time.time() < end else when(pid))
        mon = c.monitor_job(root, 0, 1e9, tmp_path / "none.jsonl", poll_s=0.05)
        assert mon["orphans_found"] == [] and mon["orphans_surviving"] == [] and u.poll() is None
        if sys.platform == "win32":  # the memory is read through the handle of the real process
            assert str(u.pid) not in mon["processes"]
    finally:
        u.kill()
        u.wait()


def test_older_process_naming_the_job_as_parent_is_left_alone(monkeypatch, tmp_path):
    u = _unrelated()
    try:
        root = _spawn(tmp_path, "import time; time.sleep(1.5)")
        table = c._process_table
        monkeypatch.setattr(c, "_process_table", lambda: {**table(), u.pid: root.pid})
        limit = 100 * 2**20  # the memory of u would be a violation of the job
        mon = c.monitor_job(root, limit, 1e9, tmp_path / "none.jsonl", poll_s=0.05)
        assert mon["violation"] is None and not mon["stopped_by_monitor"] and mon["orphans_found"] == [] and u.poll() is None
        assert str(u.pid) not in mon["processes"]
    finally:
        u.kill()
        u.wait()


def test_stopping_verifies_the_identity(tmp_path):
    u = _unrelated()
    try:
        created = c._creation_time(u.pid)
        assert created is not None and c._memory_of(u.pid, created) is not None and c._memory_of(u.pid, created + 1) is None
        assert c._kill_identified(u.pid, created + 1) is False
        c._stop_tree({(u.pid, created + 1)})
        time.sleep(0.3)
        assert u.poll() is None
        assert c._kill_identified(u.pid, created) is True
        assert u.wait(timeout=20) is not None
    finally:
        u.kill()
        u.wait()


def test_an_ended_process_has_no_identity_even_when_a_handle_holds_it():
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    assert p.wait(timeout=60) == 0  # Popen keeps its handle on Windows
    assert c._creation_time(p.pid) is None and c._memory_of(p.pid) is None


def test_real_descendants_are_measured_and_a_clean_exit_has_no_orphan(tmp_path):
    (tmp_path / "grand.py").write_text(textwrap.dedent(ALLOC.format(n=50, sleep=1)), encoding="utf-8")
    p = _spawn(tmp_path, f"""
        import subprocess, sys
        subprocess.run([sys.executable, {str(tmp_path / "grand.py")!r}])
    """)
    mon = c.monitor_job(p, 0, 1e9, tmp_path / "none.jsonl", poll_s=0.05)
    assert mon["returncode"] == 0 and mon["orphans_found"] == [] and mon["orphans_surviving"] == []
    assert mon["max_process_peak_bytes"] > 50 * 2**20 and len(mon["processes"]) >= 2


def test_many_clean_exits_with_process_churn_have_no_false_orphan(tmp_path):
    """The intermittent ORPHAN_PROCESSES: short trees, ended processes held by handles, unrelated processes started all the time, pids reused."""
    import threading
    stop = threading.Event()

    def churn():
        while not stop.is_set():
            subprocess.run([sys.executable, "-c", "pass"], capture_output=True)

    th = threading.Thread(target=churn)
    th.start()
    try:
        for i in range(25):
            p = _spawn(tmp_path, f"""
                import subprocess, sys
                ps = [subprocess.Popen([sys.executable, "-c", "pass"]) for _ in range({1 + i % 3})]
                [x.wait() for x in ps]
            """)
            mon = c.monitor_job(p, 0, 1e9, tmp_path / "none.jsonl", poll_s=0.02)
            assert mon["orphans_found"] == [] and mon["orphans_surviving"] == [] and mon["returncode"] == 0, mon
    finally:
        stop.set()
        th.join()


def test_campaign_summary_carries_the_evidence(dry_out):
    doc = c.run_campaign(dry_out, plan=_plan(), dry_run=True)
    ev = doc["resource_evidence"]
    assert ev["jobs"] >= 2 and ev["max_process_peak_bytes"] > 0 and ev["compliance"] in ("WITHIN_LIMIT_BY_THIS_EVIDENCE", "NOT_CLAIMED")
    if sys.platform != "win32":
        assert ev["compliance"] == "NOT_CLAIMED" and set(ev["by_status"]) == {"NOT_MEASURED"}


# --- coverage ------------------------------------------------------------------------------------------------------------------------------------

def test_coverage_of_the_real_subsets_is_complete_as_selection():
    plan = rd.row_plan()
    frozen = c.frozen_manifest()
    subsets = c.frozen_subsets(plan, frozen)
    counts = frozen["d057"]["subsets"]["counts"]
    keys = {r["key"] for r in plan if r["kind"] == "row"}
    for name, n in counts.items():
        if name in (*c.SECOND_PATHS, *c.EMBEDDED_PATHS):
            have = sum(name in v for v in subsets.values())
            assert have == (n + 25 if name == "SP-RH" else n), name  # block D adds 25 rows of R-H, nothing is removed
    assert set(subsets) <= keys
    for row in frozen["d057"]["subsets"]["rows"]:
        assert set(row["paths"]) <= set(subsets[row["key"]])


def test_every_frozen_selection_can_be_established_by_the_runner():
    """Complete coverage must be reachable: a mandatory check on a row that is never executed, or a journal-only check on a paired row, would turn
    every real run into incomplete coverage."""
    plan = {r["key"]: r for r in rd.row_plan() if r["kind"] == "row"}
    subsets = c.frozen_subsets(list(plan.values()) + [r for r in rd.row_plan() if r["kind"] != "row"], c.frozen_manifest())
    gaps = {k: v for k, v in subsets.items() if not plan[k]["executable"] or (plan[k]["block"] == "C" and set(v) & set(c.SECOND_PATHS))}
    assert gaps == {}


def _coverage_run(out, subsets):
    return c.run_campaign(out, plan=_plan(), dry_run=True, subsets=subsets)


def test_complete_synthetic_coverage_is_the_ordinary_success(dry_out):
    keys = [r["key"] for r in _plan()]
    doc = _coverage_run(dry_out, {keys[0]: ["SP-OP", "SP-BE"], keys[1]: ["SP-GAMMA"]})
    assert doc["state"] == c.COMPLETED and doc["second_path_coverage_complete"] is True and doc["second_path_coverage_gaps"] == {}
    assert c.exit_code(doc) == c.EXIT_OK


def test_incomplete_coverage_is_not_the_ordinary_success_and_changes_no_status(dry_out):
    keys = [r["key"] for r in _plan()]
    full = _coverage_run(dry_out, {keys[0]: ["SP-BE"]})
    shutil.rmtree(dry_out)
    doc = _coverage_run(dry_out, {keys[0]: ["SP-BE"], "absent selected row": ["SP-C"]})
    assert doc["state"] == c.COMPLETED and doc["second_path_coverage_complete"] is False and c.exit_code(doc) == c.EXIT_COVERAGE_INCOMPLETE
    assert doc["second_path_coverage_gaps"] == {"SP-C": {"not_established": [], "contradictions": [], "selected_without_a_row": ["absent selected row"]}}
    assert [(r["key"], r["status"], r["tool_defects"]) for r in doc["rows"]] == [(r["key"], r["status"], r["tool_defects"]) for r in full["rows"]]
    assert "state" in doc and doc["state"] not in (c.STOPPED_TOOL_FAILURE, c.STOPPED_CHANGED)


def test_a_selected_row_not_established_is_listed_as_a_gap():
    rec = {"key": "x", "execution_status": c.EXEC_DONE, "status": "PASS", "checks": {"SP-C": {"status": c.UNRESOLVED, "reason": "r"}}}
    cov = c.second_path_coverage([rec, {"key": "y", "execution_status": c.EXEC_DONE, "status": "PASS", "checks": {"SP-C": {"status": c.CONTRADICTION, "reason": "r"}}}],
                                 {"x": ["SP-C"], "y": ["SP-C"], "z": ["SP-C"]})
    assert c.coverage_gaps(cov)["SP-C"] == {"not_established": ["x"], "contradictions": ["y"], "selected_without_a_row": ["z"]}


def test_selected_row_without_a_row_and_embedded_paths_are_never_covered(dry_out):
    plan = _plan()
    k = plan[0]["key"]
    doc = c.run_campaign(dry_out, plan=plan, dry_run=True, subsets={k: ["SP-GAMMA", "SP-BE"], "absent": ["SP-C"]})
    cov = doc["second_path_coverage"]
    assert cov["SP-C"]["selected_without_a_row"] == ["absent"] and cov["SP-C"]["complete"] is False
    assert doc["second_path_coverage_complete"] is False
    assert cov["SP-GAMMA"]["pass"] == 1 and cov["SP-BE"]["pass"] == 1 and cov["SP-BE"]["complete"] is True
    rec = {"key": "x", "execution_status": c.EXEC_DONE, "status": "PASS", "analysis": {"row": {"diagnostics": {}}}}
    out = c.second_path_coverage([rec], {"x": ["SP-GAMMA", "SP-BE"]})
    assert out["SP-GAMMA"]["complete"] is False and out["SP-BE"]["not_established"][0]["status"] == "NOT_RUN"
