"""D-059, gap closure of the campaign runner: second paths (SP-RH, SP-LARGE, SP-C), BC-1 / TS-1, PL 6.3 at runtime, provenance and labels.

No D1 row is computed: every computation is on non-D1 grids. The mutation tests patch functions in this process only; no source and no
frozen data are modified."""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = ROOT / "scripts" / "r5_cccc_q4_d1_campaign.py"
D1_CFG = {"lx": 1.0, "nx": 50, "ly": 1.0, "ny": 50, "N": 1, "D": 1.0, "q": 1.0, "s_max": 1}


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
hs = c.hs


@pytest.fixture(autouse=True)
def _fresh_gate(monkeypatch):
    monkeypatch.delenv(pl.CAMPAIGN_AUTH_ENV, raising=False)
    monkeypatch.delenv(pl.STAGE_LOG_ENV, raising=False)
    pl._GATE_CACHE.clear()
    yield
    pl._GATE_CACHE.clear()


@pytest.fixture(scope="module")
def journal(tmp_path_factory):
    TC = pl.op8().TrajectoryConfig
    return pl.run_trajectory_journal(TC(1.0, 10, 1.0, 12, 1, s_max=3), tmp_path_factory.mktemp("j") / "j")


@pytest.fixture
def inp(journal):
    return c.row_arrays({"kind": "journal", "journal": str(journal), "s": 2})


@pytest.fixture
def dry_out():
    out = ROOT / "scratch" / f"test_d1_gaps_{os.getpid()}"
    shutil.rmtree(out, ignore_errors=True)
    yield out
    shutil.rmtree(out, ignore_errors=True)


def _plan():
    return [rd._row("A", "r=1", "1", 1, 10, 12, 1.0, 1.0, 1, s) for s in (1, 2)] + [rd._row("B", "r=1", "1", 1, 12, 10, 1.0, 1.0, 2, s) for s in (1, 2)]


# --- selection --------------------------------------------------------------------------------------------------------------------------

def test_frozen_selection_is_used_without_reduction():
    frozen = c.frozen_manifest()
    sub = {x["key"]: x["paths"] for x in frozen["d057"]["subsets"]["rows"]}
    counts = frozen["d057"]["subsets"]["counts"]
    for name in ("SP-RH", "SP-LARGE", "SP-C", "SP-BR", "SP-OP"):
        assert sum(name in p for p in sub.values()) == counts[name] and name in c.SECOND_PATHS
    d = c.block_d_rh(rd.row_plan())
    assert len(d) == 25 and all(v == ["SP-RH"] for v in d.values())  # 24 rows N = 1 and (14, 6), N = 3 at s = 1
    assert not set(d) & set(sub)
    assert "D1-D|1x1|14x6|N3|s1" in d and "D1-D|1x1|14x6|N3|s2" not in d


def test_selected_and_nonselected_rows(dry_out):
    plan = _plan()
    sel = {plan[1]["key"]: ["SP-RH", "SP-LARGE", "SP-C", "SP-BR", "SP-OP"]}
    doc = c.run_campaign(dry_out, plan=plan, dry_run=True, subsets=sel)
    recs = {r["key"]: r for f in (dry_out / "units").glob("*.json") for r in json.loads(f.read_text(encoding="utf-8"))["rows"]}
    chosen, other = recs[plan[1]["key"]], recs[plan[0]["key"]]
    assert set(chosen["checks"]) == {"SP-RH", "SP-LARGE", "SP-C", "SP-BR", "SP-OP", c.ACCURACY}
    assert set(other["checks"]) == {c.ACCURACY}  # BC-1 / TS-1 on every executed row, a second path only where selected
    for name in ("SP-RH", "SP-LARGE", "SP-C", "SP-BR", "SP-OP"):
        x = chosen["checks"][name]
        assert x["status"] == c.PASS and x["row_key"] == plan[1]["key"] and x["input_hashes"]["solved_set_sha"] and x["implementation"]
        assert x["pass_means"] == c.PASS_MEANS
    cov = doc["second_path_coverage"]
    assert all(cov[n]["selected"] == cov[n]["pass"] == 1 and cov[n]["complete"] for n in ("SP-RH", "SP-LARGE", "SP-C", "SP-BR", "SP-OP"))
    assert cov[c.ACCURACY]["executed_rows"] == 3 and cov[c.ACCURACY]["semantics"] == "OBSERVATION"
    guard = recs["D1-B|r=1|12x10|N2|s2"]
    assert guard["execution_status"] == c.EXEC_GUARD and guard["status"] == "UNRESOLVED" and "checks" not in guard


def test_selected_row_not_run_is_listed_never_covered():
    rec = c._not_executed_record({"key": "k", "block": "A", "N": 1, "s": 1, "mandatory": True}, "stopped", None, c.EXEC_FAILED)
    cov = c.second_path_coverage([rec], {"k": ["SP-RH"]})["SP-RH"]
    assert cov["complete"] is False and cov["pass"] == 0 and cov["not_established"][0]["status"] == "NOT_RUN"
    assert c.EXEC_FAILED in cov["not_established"][0]["reason"]


def test_second_path_is_never_a_call_of_the_solver(monkeypatch, inp):
    def no(*a, **k):
        raise AssertionError("the B-C solver was called by a second path")
    monkeypatch.setattr(pl.op8(), "solve_with_c", no)
    monkeypatch.setattr(c.bc, "solve_bc", no)
    monkeypatch.setattr(c.bc, "run_halfstep", no)
    for fn in (c.sp_rh, c.sp_large, c.sp_c):
        assert fn(inp)["status"] == c.PASS


# --- SP-RH ------------------------------------------------------------------------------------------------------------------------------

def test_sp_rh_failure_paths(monkeypatch, inp):
    r = c.sp_rh(inp)
    assert r["status"] == c.PASS and r["lambda_low"] > 0 and r["R_H_bound_2_up"] >= 0 and r["V_bound_2_up"] >= 0
    with monkeypatch.context() as m:
        m.setattr(c.adapter().br(), "RH_LIMIT", 10)
        assert "outside its rule" in c.sp_rh(inp)["reason"]
    with monkeypatch.context() as m:
        m.setattr(hs, "certify_lambda_min", lambda *a, **k: None)
        x = c.sp_rh(inp)
        assert x["status"] == c.UNRESOLVED and "certificate" in x["reason"]
    with monkeypatch.context() as m:
        m.setattr(hs, "solve_hessian", lambda p: (hs.FACTORIZATION_FAILED, None))
        assert c.sp_rh(inp)["status"] == c.UNRESOLVED
    with monkeypatch.context() as m:
        m.setattr(hs, "error_bound", lambda *a, **k: float("inf"))
        assert "overflowed" in c.sp_rh(inp)["reason"]


def test_mutation_sp_rh_detects_a_deflated_bound(monkeypatch, inp):
    """Mutation (in this process): the bound by the exact residual made too small on a V that really differs."""
    bad = dict(inp, V=inp["V"] * (1 + 1e-6))
    assert c.sp_rh(bad)["status"] == c.PASS  # the true bound of a wrong V is wide: the error is measured, not hidden
    assert float(c.sp_rh(bad)["V_error_2_up"]) > float(c.sp_rh(inp)["V_error_2_up"]) * 100
    monkeypatch.setattr(hs, "error_bound", lambda *a, **k: 1e-300)
    r = c.sp_rh(bad)
    assert r["status"] == c.CONTRADICTION and "certified bound is wrong" in r["reason"]


# --- SP-LARGE ---------------------------------------------------------------------------------------------------------------------------

def test_sp_large_exact_comparisons(inp):
    r = c.sp_large(inp)
    assert r["status"] == c.PASS and r["C3plus_theorem_bound"] == "HOLDS" and r["eta_qa"] == r["eta_qa_recorded"]
    for k in "ABCF":
        assert 0 <= r[k]["max_abs_diff_up"] < 1e-10 and r[k]["relative_up"] is not None
    assert r["lambda_low"] > 0 and r["V_bound_2_up"] >= 0


def test_sp_large_failure_paths(monkeypatch, inp):
    bad_c = inp["C3plus"].copy()
    bad_c[0, 0] = float("nan")
    assert c.sp_large(dict(inp, C3plus=bad_c))["status"] == c.UNRESOLVED
    assert c.sp_large(dict(inp, record={}))["status"] == c.UNRESOLVED  # no k of the theorem
    wrong = c.sp_large(dict(inp, C3plus=inp["C3plus"] * (1 + 1e-9)))
    assert wrong["status"] == c.CONTRADICTION and "theorem bound" in wrong["reason"]
    v = c.sp_large(dict(inp, V=inp["V"] * (1 + 1e-9)))
    assert v["status"] == c.CONTRADICTION and "eta_QA" in v["reason"]  # V is not the one of the record
    with monkeypatch.context() as m:
        m.setattr(hs, "certify_lambda_min", lambda *a, **k: None)
        assert "certificate" in c.sp_large(inp)["reason"]


def test_mutation_sp_large_detects_wrong_unknown_ordering(monkeypatch, journal):
    """Mutation (in this process): the unknowns shifted by one position: the exact residual no longer reproduces eta_QA of the record."""
    inp = c.row_arrays({"kind": "journal", "journal": str(journal), "s": 2})
    monkeypatch.setattr(c, "_term_first", lambda V: np.roll(np.asarray(V, dtype=float).T.reshape(-1), 1))
    assert c.sp_large(inp)["status"] == c.CONTRADICTION


def test_input_binding_of_the_checks(monkeypatch, journal):
    real = pl.load_journal

    def tampered(d):
        j, a = real(d)
        a = dict(a)
        a["s2_V_bc"] = a["s2_V_bc"] * (1 + 2.0**-40)
        return j, a

    spec = {"kind": "journal", "journal": str(journal), "s": 2}
    monkeypatch.setattr(pl, "load_journal", tampered)
    assert c.row_arrays(spec)["binding_defects"]
    out = c.second_paths({"spec": spec, "row_key": "k", "subset_paths": ["SP-RH", "SP-LARGE", "SP-C", "SP-BR", "SP-OP"]})
    for name in ("SP-RH", "SP-LARGE", "SP-C", "SP-OP"):
        assert out[name]["status"] == c.UNRESOLVED and out[name]["reason"].startswith("input binding"), name
    assert out["SP-BR"]["status"] == c.UNRESOLVED and out[c.ACCURACY]["status"] == c.UNRESOLVED


# --- SP-C -------------------------------------------------------------------------------------------------------------------------------

def test_sp_c_agreement_and_inconsistent_double(monkeypatch, inp):
    r = c.sp_c(inp)
    assert r["status"] == c.PASS and r["C3plus_bitwise_the_assembly"] is True and r["C3plus_theorem_bound"] == "HOLDS" and r["k"]
    assert r["CONTROL_error_up"] >= 0 and r["C3plus_spd_exact"] is True
    wrong = c.sp_c(dict(inp, C3plus=inp["C3plus"] * (1 + 1e-9)))
    assert wrong["status"] == c.CONTRADICTION and "theorem bound" in wrong["reason"]
    one_ulp = inp["C3plus"].copy()
    one_ulp[0, 0] = np.nextafter(one_ulp[0, 0], np.inf)  # inside the theorem bound, but not the assembly of the stored set
    x = c.sp_c(dict(inp, C3plus=one_ulp))
    assert x["status"] == c.CONTRADICTION and "assembly" in x["reason"]
    assert c.sp_c(dict(inp, C3plus=None))["status"] == c.UNRESOLVED
    with monkeypatch.context() as m:
        m.setattr(c._ca(), "c_control", lambda p: np.full_like(inp["C3plus"], np.inf))
        assert "CONTROL" in c.sp_c(inp)["reason"]


def test_mutation_sp_c_detects_a_wrong_exact_reference(monkeypatch, inp):
    """Mutation (in this process): the exact reference C shifted: the theorem bound must fail."""
    real = hs.exact_coefficients

    def shifted(p):
        A, B, C, F = real(p)
        return A, B, [[x * 2 for x in row] for row in C], F

    monkeypatch.setattr(hs, "exact_coefficients", shifted)
    assert c.sp_c(inp)["status"] == c.CONTRADICTION


# --- BC-1 / TS-1 ------------------------------------------------------------------------------------------------------------------------

def test_accuracy_observation_valid_and_unresolved(monkeypatch, journal, inp):
    spec = {"kind": "journal", "journal": str(journal), "s": 2}
    r = c.accuracy_record(c.br_reference(inp, spec), True)
    assert r["status"] == "RESOLVED" and r["semantics"].startswith("OBSERVATION") and r["basis"] == "frozen subset SP-BR"
    assert float(r["BC1_rel_inf_down"]) <= float(r["BC1_rel_inf_up"]) and float(r["TS1_e_field_down"]) <= float(r["TS1_e_field_up"])
    assert r["TS1_observation"].endswith("(OBSERVATION)") and r["budget_inf"] == "SATISFIED"
    assert "FAIL" not in json.dumps(r).replace("FAIL_COMPUTATION", "")
    with monkeypatch.context() as m:  # no positive certified denominator of the reference
        m.setattr(hs, "certify_lambda_min", lambda *a, **k: None)
        x = c.accuracy_record(c.br_reference(inp, spec), False)
        assert x["status"] == c.UNRESOLVED and "certificate" in x["reason"] and "BC1_rel_inf_up" not in x
        assert x["basis"].startswith("protocol 5.1")


def test_exceedance_of_the_candidate_is_only_an_observation():
    ref = {"result": "RESOLVED (set 2-norm, set inf-norm, field energy norm)", "TS1_observation": "ABOVE the candidate (OBSERVATION)"}
    r = c.accuracy_record(ref, False)
    assert r["status"] == "RESOLVED" and r["TS1_observation"].startswith("ABOVE")
    rec = {"key": "k", "status": "PASS", "execution_status": c.EXEC_DONE, "checks": {c.ACCURACY: r}}
    assert c.second_path_coverage([rec], {})[c.ACCURACY]["resolved"] == 1 and rec["status"] == "PASS"


def test_accuracy_on_paired_and_n2_rows(tmp_path):
    TC = pl.op8().TrajectoryConfig
    jd = pl.run_trajectory_journal(TC(1.0, 10, 1.0, 12, 2, s_max=1), tmp_path / "n2")
    spec = {"kind": "journal", "journal": str(jd), "s": 1}
    out = c.second_paths({"spec": spec, "row_key": "k", "subset_paths": ["SP-RH"]})
    assert out["SP-RH"]["status"] == c.PASS and out[c.ACCURACY]["status"] == "RESOLVED"  # OD-2: elementwise norm of the set, N = 2
    assert out[c.ACCURACY]["norm"].startswith("OD-2") and out[c.ACCURACY]["input_hashes"]["solved_set_sha"]
    doc = rd.bcd().write_paired(TC(1.0, 12, 1.0, 10, 1, s_max=2), tmp_path / "p")
    s2 = next(x for x in doc["rows"] if x["s"] == 2)
    out = c.second_paths({"spec": s2["spec"], "row_key": "p", "subset_paths": ["SP-RH"]})
    assert out[c.ACCURACY]["status"] == "RESOLVED"
    assert out["SP-RH"]["status"] == c.BLOCKED_FOR_OWNER and "no approved path" in out["SP-RH"]["reason"]  # exact capability named


# --- status of the row against the checks (OD-7), contradictions --------------------------------------------------------------------------

def _fake_jobs(monkeypatch, second):
    def fake(job, work, limit, signal, factorization_limit_s=None):
        if job["kind"] == "row":
            return {"analysis": {"row": {"inputs_sha": {"given": "x"}}, "status": "PASS", "classification": {"reasons": []}, "tool_defects": [],
                                 "reproduction": {"child_pid": 1}}, "input": {"executed": True}, "process": {}, "resources": {}, "seconds": 0}
        return second
    monkeypatch.setattr(c, "run_job", fake)


ROW = {"key": "k", "block": "A", "N": 1, "s": 1, "mandatory": True}


def test_violation_in_the_checks_leaves_the_row_status(monkeypatch, tmp_path):
    _fake_jobs(monkeypatch, {"aborted": {"aborted": "memory", "pid": 7}, "resources": {"violation": 1}, "seconds": 1})
    r = c._row_job(ROW, {"kind": "journal"}, {"k": ["SP-RH"]}, tmp_path, 1, None, True)
    assert r["status"] == "PASS" and r["execution_status"] == c.EXEC_DONE and r["tool_defects"] == []
    assert r["checks"]["SP-RH"]["status"] == c.UNRESOLVED_RESOURCES_OF_CHECK and r["checks"][c.ACCURACY]["status"] == c.UNRESOLVED_RESOURCES_OF_CHECK
    assert not c.second_path_coverage([r], {"k": ["SP-RH"]})["SP-RH"]["complete"]


def test_contradiction_is_a_tool_defect_and_stops_the_verdict(monkeypatch, tmp_path):
    _fake_jobs(monkeypatch, {"checks": {"SP-C": {"status": c.CONTRADICTION, "reason": "bound"}, c.ACCURACY: {"status": "RESOLVED"}},
                             "resources": {}, "seconds": 1})
    r = c._row_job(ROW, {"kind": "journal"}, {"k": ["SP-C"]}, tmp_path, 1, None, True)
    assert r["status"] == "PASS" and r["tool_defects"] == ["SECOND_PATH_CONTRADICTION SP-C: bound"]
    ct = _load(ROOT / "scripts" / "r5_cccc_q4_d1_ready_controls.py", "_ct_gaps")
    mand = pf.mandatory_index(c.frozen_manifest()["mandatory_rows"])
    rows = [{"key": k, "block": b, "N": N, "s": s, "mutated": False, "input": c.encode_input(ct._syn("PASS")), "z4": None, "tool_defects": [],
             "mandatory": True} for k, (b, N, s) in mand.items()]
    assert c.campaign_verdict(rows)["verdict"] == "PASS_CANDIDATE"
    rows[0]["tool_defects"] = r["tool_defects"]
    v = c.campaign_verdict(rows)
    assert v["verdict"] == rd.VERDICT_NOT_ISSUED and v["classifier"]["campaign_verdict"] == "PASS_CANDIDATE"  # classification unchanged


# --- PL 6.3 at runtime ------------------------------------------------------------------------------------------------------------------

def _spawn(tmp_path: Path, code: str, env=None):
    src = tmp_path / "child.py"
    src.write_text(textwrap.dedent(code), encoding="utf-8")
    return subprocess.Popen([sys.executable, str(src)], env={**os.environ, **(env or {})}, cwd=tmp_path)


ALLOC = """
    import time
    x = bytearray({n} * 2**20)
    for i in range(0, len(x), 4096):
        x[i] = 1
    time.sleep({sleep})
"""


def test_memory_of_a_real_subprocess_child_stops_the_tree(tmp_path):
    (tmp_path / "grand.py").write_text(textwrap.dedent(ALLOC.format(n=200, sleep=60)), encoding="utf-8")
    p = _spawn(tmp_path, f"""
        import subprocess, sys
        subprocess.run([sys.executable, {str(tmp_path / "grand.py")!r}])
    """)
    mon = c.monitor_job(p, 100 * 2**20, 1e9, tmp_path / "none.jsonl")
    v = mon["violation"]
    assert v["aborted"] == "memory" and v["working_set_bytes"] > v["limit_bytes"] == 100 * 2**20 and "PL 6.3" in v["rule"]
    assert v["pid"] != p.pid and len(mon["processes"]) >= 2  # the child, not the process started by the runner
    time.sleep(0.5)
    table = c._process_table()
    assert p.poll() is not None and not {int(x) for x in mon["processes"]} & set(table)  # the whole tree is stopped


def test_limit_is_per_process_the_sum_is_a_measurement(tmp_path):
    (tmp_path / "grand.py").write_text(textwrap.dedent(ALLOC.format(n=70, sleep=2)), encoding="utf-8")
    p = _spawn(tmp_path, f"""
        import subprocess, sys
        ps = [subprocess.Popen([sys.executable, {str(tmp_path / "grand.py")!r}]) for _ in range(2)]
        [x.wait() for x in ps]
    """)
    limit = 150 * 2**20
    mon = c.monitor_job(p, limit, 1e9, tmp_path / "none.jsonl")
    assert mon["violation"] is None and mon["max_process_peak_bytes"] < limit < mon["max_tree_sum_bytes"]


STAGE = """
    import json, os, time
    def ev(name, event, f=True):
        with open(os.environ["L"], "a", encoding="utf-8") as h:
            h.write(json.dumps({{"pid": os.getpid(), "stage": name, "event": event, "factorization": f, "t": time.time()}}) + chr(10))
    {body}
"""


def test_factorization_time_is_enforced_per_stage(tmp_path):
    log = tmp_path / "s.jsonl"
    body = 'ev("a factorization", "start"); time.sleep(60)'
    p = _spawn(tmp_path, STAGE.format(body=body), {"L": str(log)})
    t0 = time.time()
    mon = c.monitor_job(p, 0, 1.0, log)
    v = mon["violation"]
    assert time.time() - t0 < 30 and v["aborted"] == "factorization_time" and v["stage"] == "a factorization" and v["completed"] is False
    assert v["seconds"] > v["limit_seconds"] == 1.0 and "PL 6.3" in v["rule"] and p.poll() is not None


def test_completed_factorization_over_the_limit_and_other_stages(tmp_path):
    log = tmp_path / "s.jsonl"
    p = _spawn(tmp_path, STAGE.format(body='ev("f", "start"); time.sleep(1.5); ev("f", "end")'), {"L": str(log)})
    v = c.monitor_job(p, 0, 1.0, log, poll_s=5.0)["violation"]
    assert v["completed"] is True and v["stage"] == "f" and v["seconds"] > 1.0
    log2 = tmp_path / "s2.jsonl"
    body = 'ev("B_E", "start", False); time.sleep(1.5); ev("short", "start"); ev("short", "end")'
    p = _spawn(tmp_path, STAGE.format(body=body), {"L": str(log2)})
    mon = c.monitor_job(p, 0, 1.0, log2)  # a stage without a factorization has no time limit (protocol 8)
    assert mon["violation"] is None and mon["factorization_stages"]["short"]["count"] == 1


def test_factorization_refusal_gives_unresolved_resources_with_evidence(dry_out):
    doc = c.run_campaign(dry_out, plan=_plan(), dry_run=True, factorization_limit_s=0.0)
    st = {r["key"]: r for r in doc["rows"]}
    assert st["D1-A|r=1|10x12|N1|s1"]["status"] == "UNRESOLVED_RESOURCES" and st["D1-A|r=1|10x12|N1|s1"]["execution_status"] == c.EXEC_RESOURCES
    assert st["D1-B|r=1|12x10|N2|s2"]["status"] == "UNRESOLVED"
    recs = [r for f in (dry_out / "units").glob("*.json") for r in json.loads(f.read_text(encoding="utf-8"))["rows"]]
    m = next(r for r in recs if r["status"] == "UNRESOLVED_RESOURCES")["input"]["not_executed"]["measured"]
    assert m["aborted"] == "factorization_time" and m["stage"] == "B-C solve (OP-8 half-step)" and m["limit_seconds"] == 0.0 and m["pid"]


def test_replay_has_no_total_limit_under_the_monitor(monkeypatch):
    """Protocol 8: no time limit for the exact B_E; under the monitor the replay is limited per factorization stage only."""
    seen = []

    def run(*a, **k):
        seen.append(k.get("timeout"))
        raise RuntimeError("stop here")

    monkeypatch.setattr(pl.subprocess, "run", run)
    spec = {"kind": "arrays", "arrays_npz": "absent.npz", "arrays_sha256": {}, "config": {**D1_CFG, "nx": 10, "ny": 12}, "s": 1}
    with pytest.raises(RuntimeError):
        pl.run_child(spec)
    monkeypatch.setenv(pl.STAGE_LOG_ENV, "log")
    with pytest.raises(RuntimeError):
        pl.run_child(spec)
    assert seen == [1800, None]


def test_real_run_refuses_changed_limits_and_unmeasurable_platform(monkeypatch, dry_out):
    with pytest.raises(c.CampaignRefused, match="PL 6.3 memory limit"):
        c.run_campaign(dry_out, factorization_limit_s=1.0)
    monkeypatch.setattr(c, "memory_measurement_works", lambda: False)
    assert any("PL 6.3 monitor" in x for x in c.platform_defects())
    monkeypatch.undo()
    assert c.memory_measurement_works() is True or sys.platform not in ("win32", "linux")


def test_stage_timers_do_not_change_the_numbers_and_replay_is_measured(dry_out, tmp_path):
    """The job tree really contains the replay child, its stage and its threads; the journal is bitwise the one without timers."""
    plan = _plan()[:2]
    c.run_campaign(dry_out, plan=plan, dry_run=True)
    f = next((dry_out / "units").glob("*.json"))
    u = json.loads(f.read_text(encoding="utf-8"))
    r = u["rows"][0]
    assert r["replay"]["bitwise_equal"] is True and r["replay"]["child_blas_threads"] == {"numpy": 1, "scipy": 1} and r["replay"]["child_peak"]["peak"]
    assert "replay process of the row" in r["resources"]["last_stage_by_process"].values() or any(
        "B-C" in x or "replay" in x for x in r["resources"]["last_stage_by_process"].values())
    assert len(r["resources"]["processes"]) >= 2 and r["resources"]["factorization_stages"]["B-C solve (OP-8 half-step)"]["count"] >= 1
    TC = pl.op8().TrajectoryConfig
    jd = pl.run_trajectory_journal(TC(1.0, 10, 1.0, 12, 1, s_max=8), tmp_path / "plain")
    a = json.loads((jd / "journal.json").read_text(encoding="utf-8"))["meta"]["arrays_sha256"]
    timed = next(dry_out.glob("work/*/data/journal/journal.json"))
    b = json.loads(timed.read_text(encoding="utf-8"))["meta"]["arrays_sha256"]
    assert a == b


# --- real D1 grids, authorisation, labels, restart --------------------------------------------------------------------------------------

def test_no_d1_row_through_a_job_child(tmp_path, dry_out):
    for signal in (None, "D-059:" + "0" * 40):
        r = c.run_job({"kind": "journal", "config": D1_CFG, "out": str(dry_out / "d1"), "dry_run": False}, tmp_path / "w", 0, signal)
        assert "failed" in r and "D1_START = BLOCKED" in r["tail"] and not (dry_out / "d1").exists()
    env = {**os.environ, pl.STAGE_LOG_ENV: str(tmp_path / "x.jsonl")}  # the stage log authorises nothing
    p = subprocess.run([sys.executable, str(CAMPAIGN), "run", "--out", "scratch/never"], cwd=ROOT, capture_output=True, text=True, env=env)
    assert p.returncode == 2 and "D1_START = BLOCKED" in p.stderr


def test_relabel_keeps_the_tool_label():
    doc = {"data_status": "PRELAUNCH ... NOT D1 data", "row": {"data_status": "tool", "x": [{"data_status": "tool2"}]}, "y": 1}
    r = c.relabel(doc, c.DATA_STATUS_CAMPAIGN)
    assert r["data_status"] == r["row"]["data_status"] == r["row"]["x"][0]["data_status"] == c.DATA_STATUS_CAMPAIGN
    assert r["tool_data_status"].startswith("PRELAUNCH") and r["row"]["x"][0]["tool_data_status"] == "tool2" and r["y"] == 1
    assert "NOT D1" not in c.DATA_STATUS_CAMPAIGN and c.relabel({"data_status": c.DATA_STATUS_CAMPAIGN}, c.DATA_STATUS_CAMPAIGN) == {"data_status": c.DATA_STATUS_CAMPAIGN}


def _walk(o):
    if isinstance(o, dict):
        yield o
        for v in o.values():
            yield from _walk(v)
    elif isinstance(o, list):
        for v in o:
            yield from _walk(v)


def test_labels_ledger_and_restart(dry_out):
    plan = _plan()
    with pytest.raises(c.CampaignInterrupted):
        c.run_campaign(dry_out, plan=plan, dry_run=True, stop_after_jobs=2)
    first = next((dry_out / "units").glob("*.json"))
    saved = first.read_bytes()
    first.write_bytes(saved + b" ")  # a stored result changed between the runs
    with pytest.raises(c.CampaignRefused, match="differ from the ledger"):
        c.run_campaign(dry_out, plan=plan, dry_run=True)
    first.write_bytes(saved)
    other = [dict(r) for r in plan]
    other[0]["reason"] = "changed plan"
    with pytest.raises(c.CampaignRefused, match="provenance or plan"):
        c.run_campaign(dry_out, plan=other, dry_run=True)
    doc = c.run_campaign(dry_out, plan=plan, dry_run=True)
    assert len(doc["restarts"]) == 2 and doc["provenance"]["tool_blobs"][c.HS_MODULE] and doc["state"] == "COMPLETED"
    assert (dry_out / "PROVENANCE_README.json").exists()
    seen = 0
    for f in (dry_out / "units").glob("*.json"):
        for d in _walk(json.loads(f.read_text(encoding="utf-8"))):
            if "data_status" in d:
                seen += 1
                assert d["data_status"] == c.DATA_STATUS_DRY
            if "tool_data_status" in d:
                assert "NOT D1 data" in d["tool_data_status"]
    assert seen
    assert "no owner acceptance" in doc["scope"]
    assert pl.git_blob_id(ROOT / pl.FROZEN_MANIFEST) == pl.FROZEN_MANIFEST_BLOB
