"""Tests of QA-D1-READY (D-057.1...D-057.6): the integrated layer scripts/r5_cccc_q4_d1_ready.py, the block-C driver
scripts/r5_cccc_q4_d1_block_c_driver.py and the integrated controls scripts/r5_cccc_q4_d1_ready_controls.py against the expectations
fixed before the code (tests/reference/r5_cccc_q4_d1_ready_controls_expected.json, own commit). Small non-D1 configurations only;
D1_START = BLOCKED."""

import ast
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
READY = ROOT / "scripts" / "r5_cccc_q4_d1_ready.py"
CONTROLS = ROOT / "scripts" / "r5_cccc_q4_d1_ready_controls.py"
BLOCK_C = ROOT / "scripts" / "r5_cccc_q4_d1_block_c_driver.py"
EXPECTED = ROOT / "tests" / "reference" / "r5_cccc_q4_d1_ready_controls_expected.json"
RESULTS = ROOT / "docs" / "results" / "r5_cccc_q4_d1_ready"
EXPECTED_COMMIT = "bbb90f65645efe093eb80116bba390c125c0847b"  # the expectations, fixed before the code (their own commit)
CODE_FILES = ("scripts/r5_cccc_q4_d1_ready.py", "scripts/r5_cccc_q4_d1_ready_controls.py", "scripts/r5_cccc_q4_d1_block_c_driver.py",
              "tests/reference/test_r5_cccc_q4_d1_ready.py")


def _load(path: Path, name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ct = _load(CONTROLS, "_ct_ready_controls")
rd = ct.rd()
pl = rd.pl()


@pytest.fixture(scope="module")
def controls(tmp_path_factory):
    """The harness in a fresh process, as the launcher runs it (OD-6: BLAS threads set before NumPy is imported)."""
    out = tmp_path_factory.mktemp("ready_controls")
    p = subprocess.run([sys.executable, str(CONTROLS), "run", "--out", str(out)], cwd=ROOT, capture_output=True, text=True,
                       timeout=7200)
    assert p.returncode in (0, 1), p.stderr[-3000:]
    return json.loads((out / "controls_results.json").read_text(encoding="utf-8"))


# --- expectations fixed before the code -------------------------------------------------------------------------------------------

def test_expectations_are_frozen(tmp_path):
    exp = ct.load_expected()
    assert exp["fixed_before_execution"] is True and len(exp["control_B"]) == 12 and len(exp["control_G"]) == 11
    bad = tmp_path / "e.json"
    bad.write_bytes(EXPECTED.read_bytes().replace(b'"PASS_CANDIDATE"', b'"FAIL_CANDIDATE"', 1))
    with pytest.raises(RuntimeError, match="modified"):
        ct.load_expected(bad)
    crlf = tmp_path / "crlf.json"
    crlf.write_bytes(EXPECTED.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
    assert ct.load_expected(crlf) == exp


def _git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)


def test_expectations_committed_before_the_code():
    """The expectations commit holds the expectations and none of the code; the file is today the blob of that commit."""
    if shutil.which("git") is None or _git("rev-parse", "--git-dir").returncode:
        pytest.skip("git not available: historical check NOT performed")
    if _git("cat-file", "-e", f"{EXPECTED_COMMIT}^{{commit}}").returncode:
        if _git("rev-parse", "--is-shallow-repository").stdout.strip() == "true":
            pytest.skip("shallow checkout: historical check NOT performed")
        pytest.fail(f"full clone without {EXPECTED_COMMIT}")
    tree = _git("ls-tree", "-r", "--name-only", EXPECTED_COMMIT).stdout.splitlines()
    rel = EXPECTED.relative_to(ROOT).as_posix()
    assert rel in tree and not set(CODE_FILES) & set(tree)
    b = _git("rev-parse", f"{EXPECTED_COMMIT}:{rel}").stdout.strip()
    assert b == rd.git_blob_id(EXPECTED)


# --- the integrated controls ------------------------------------------------------------------------------------------------------

def test_integrated_controls_match_the_expectations(controls):
    s = controls["summary"]
    for k, v in s.items():
        assert not v.get("mismatch"), (k, v)
    assert s["control_A"]["pr48_harness_all_match"] and s["control_A"]["z4_aggregation_match"] == 8
    for k in ("row_plan", "rt17", "platform"):
        assert s[k]["match"], k
    assert controls["all_match"]
    # every declared channel of control G fired (a gap would be UNRESOLVED of the control, never PASS)
    assert all(x["detected"] for x in controls["control_G"])
    not_exec = [x["id"] for x in controls["control_B"] if x["match"] is None]
    assert not_exec == [] or sys.platform != "win32"


def test_control_v_change_is_confirmed(controls):
    for c in controls["control_V"]:
        assert all(c["changed"].values()) and all(c["unchanged"].values()) and c["verdict"] == "PASS"


# --- D-057.1 in the pipeline: rows N >= 2 ------------------------------------------------------------------------------------------

def test_rows_n23_at_s2_are_unresolved_with_the_reason(tmp_path):
    cfg = pl.op8().TrajectoryConfig(1.0, 8, 1.25, 10, 2, s_max=3)
    jd = pl.run_trajectory_journal(cfg, tmp_path / "n2")
    for s in (2, 3):
        r = rd.analyse_ready({"kind": "journal", "journal": str(jd), "s": s})
        assert r["status"] == "UNRESOLVED" and "STOPPED_RANK_GUARD_UNCERTIFIED" in r["not_executed_reason"] and r.get("row") is None
    with pytest.raises(pl.RankGuardRefused):
        pl.guard_row(cfg, 2)
    pl.guard_row(cfg, 1)


def test_rank_guard_reason_whatever_the_journal_says(tmp_path, monkeypatch):
    """A journal claiming an executed s = 2 for N = 2 is never computed (no binding, no child)."""
    cfg = pl.op8().TrajectoryConfig(1.0, 8, 1.25, 10, 2, s_max=2)
    jd = pl.run_trajectory_journal(cfg, tmp_path / "n2")
    monkeypatch.setattr(pl, "load_journal", lambda *a: (_ for _ in ()).throw(AssertionError("loaded")))
    r = pl.analyse({"kind": "journal", "journal": str(jd), "s": 2})
    assert r["status"] == "UNRESOLVED" and "D-057.1" in r["not_executed_reason"]


# --- F-2 binding of the actual inputs ----------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def t1(tmp_path_factory):
    cfg = pl.op8().TrajectoryConfig(1.0, 8, 1.25, 10, 1, s_max=2)
    jd = pl.run_trajectory_journal(cfg, tmp_path_factory.mktemp("t1"))
    return cfg, jd, rd.analyse_ready({"kind": "journal", "journal": str(jd), "s": 2}, sp_be=True)


def test_f2_holds_on_a_clean_row_and_sees_each_rebinding(t1):
    cfg, _, r = t1
    row = r["row"]
    assert pl.actual_input_defects(row, cfg) == [] and r["f2_defects"] == []
    if r["blas_threads_read"] == 1:
        assert r["status"] in ("PASS", "UNRESOLVED_RESOLVABILITY")
    edits = {"q_double input": ("q_double", "field", "0" * 64), "pi_double input": ("pi_double_p", "field", "0" * 64),
             "gamma used": ("gamma_b1", "used", "x"), "gamma inputs": ("gamma_b1", "V", "0" * 64), "b_e used": ("b_e", "used", "x"),
             "b_e grid": ("b_e", "grid_constants", "0" * 64), "formation": ("formation_s", "out", "0" * 64),
             "second path B_E P^": ("second_path_BE", "P_hat", ["0x0p+0"] * 3)}
    for name, (k, f, v) in edits.items():
        bad = json.loads(json.dumps(row))
        bad["actual_inputs"][k][f] = v
        assert pl.actual_input_defects(bad, cfg), name
    gone = json.loads(json.dumps(row))
    del gone["actual_inputs"]
    assert pl.actual_input_defects(gone, cfg)


def test_f2_defect_makes_the_row_unresolved_certificate(t1):
    _, _, r = t1
    out = rd._reassemble(r, ["F-2: test"], None)
    assert out["status"] == "UNRESOLVED_CERTIFICATE" and "P-a" in out["classification"]["reason_premises"]


# --- F-3, SP-BE, Z-4 -------------------------------------------------------------------------------------------------------------

def test_f3_uses_an_independent_rounding_and_is_recorded(t1):
    _, _, r = t1
    spg = r["row"]["diagnostics"]["sp_gamma"]
    assert spg["performed"] and spg["inclusion"] and spg["tight"] and spg["equals_B1_used"]
    g = Fraction(r["row"]["gamma_exact"])
    assert float.fromhex(spg["RD_gamma_B"]) == pl.be2().round_down(g) and float.fromhex(spg["RU_gamma_B"]) == pl.be2().round_up(g)
    assert r["row"]["diagnostics"]["second_path_BE"]["match"] is True


def test_z4_required_rule():
    assert rd.z4_required({"mandatory": True, "status": "UNRESOLVED_RESOLVABILITY"})
    assert rd.z4_required({"mandatory": False, "status": "FAIL_COMPUTATION"})
    assert not rd.z4_required({"mandatory": False, "status": "UNRESOLVED_RESOLVABILITY"})
    assert not rd.z4_required({"mandatory": True, "status": "PASS"})


# --- plan, geometry, norm, subsets, archive, platform -------------------------------------------------------------------------------

def test_row_plan_under_d057_1():
    plan = rd.row_plan()
    c = rd.plan_counts(plan)
    assert c == ct.load_expected()["row_plan"]
    assert len({r["key"] for r in plan}) == len(plan)
    assert all(r["executable"] for r in plan if r["mandatory"])
    assert all("STOPPED_RANK_GUARD_UNCERTIFIED" in r["reason"] for r in plan if r["N"] >= 2 and r["s"] >= 2)
    assert all(not rd.geometry_check(r) for r in plan)
    d = [r for r in plan if r["block"] == "D"]
    assert {(r["lx_hex"], r["ly_hex"]) for r in d} == {((1.0).hex(), (1.0).hex())}


def test_set_inf_norm_od2():
    assert rd.set_inf_norm([[1.0, -3.0], [2.0, 0.5]]) == 3.0
    with pytest.raises(ValueError):
        rd.set_inf_norm([[float("nan")]])
    rec = rd.bc1_record(np.array([[1.0, 2.0]]), np.array([[1.0, 2.0 + 2.0**-50]]), np.eye(2))
    assert rec["d_inf_exact"] == f"{Fraction(2.0**-50).numerator}/{Fraction(2.0**-50).denominator}" and len(rec["basis_sha"]) == 64


def test_subsets_od11():
    s = rd.subsets()
    assert s["counts"]["SP-GAMMA"] == 102 and s["counts"]["SP-BE"] == 20 and s["counts"]["SP-OP"] == s["counts"]["SP-LARGE"]
    be = {x["key"] for x in s["rows"] if "SP-BE" in x["paths"]}
    assert "D1-A|r=4|320x1280|N1|s2" in be and "D1-A|r=5/2|400x1000|N1|s2" in be
    assert s["sha256"] == hashlib.sha256(rd.pf().canonical(s["rows"])).hexdigest()


def test_start_archive_round_trip_and_tamper():
    plan = rd.row_plan()
    a = rd.start_archive(plan)
    assert rd.verify_archive(a) == []
    assert {(e["nx"], e["N"]) for e in a["sets"]} == {(r["nx"], r["N"]) for r in plan if r["s"] == 1 and r["executable"]}
    bad = json.loads(json.dumps(a))
    x = float.fromhex(bad["sets"][0]["columns_hex"][0][0])
    bad["sets"][0]["columns_hex"][0][0] = np.nextafter(x, np.inf).hex()
    assert rd.verify_archive(bad)


def test_freeze_rule_od6():
    rec = rd.platform_record()
    ok = rd.freeze_allowed(rec)
    assert ok["allowed"] is (sys.platform == "win32" and rec["python"] == "3.13.15" and rec["blas_threads_read_from_library"] == 1)
    for change in ({"python": "3.14.8"}, {"sys_platform": "linux"}, {"blas_threads_read_from_library": 2}):
        f = rd.freeze_allowed({**rec, **change})
        assert f["allowed"] is False and f["state"].startswith("UNRESOLVED")


def test_campaign_is_refused():
    with pytest.raises(rd.D1Blocked):
        rd.campaign()
    p = subprocess.run([sys.executable, str(READY), "campaign"], cwd=ROOT, capture_output=True, text=True)
    assert p.returncode == 2 and "D1_START = BLOCKED" in p.stderr


def test_d1_grids_refused_before_computation(tmp_path, monkeypatch):
    monkeypatch.setattr(pl, "_compute_row_core", lambda *a, **k: (_ for _ in ()).throw(AssertionError("computed")))
    TC = pl.op8().TrajectoryConfig
    for nx, ny in ((100, 100), (80, 320), (14, 6)):
        with pytest.raises(pl.D1Refused):
            rd.bcd().write_paired(TC(1.0, nx, 1.0, ny, 1, s_max=1), tmp_path / "x")
    with pytest.raises(pl.D1Refused):
        pl.compute_row(TC(1.0, 50, 1.0, 50, 1, s_max=1), 1, np.zeros((49, 1)), np.zeros((49, 1)), np.zeros((49, 1)))


# --- SP-OP --------------------------------------------------------------------------------------------------------------------------

def test_sp_op_exact_identity_and_negative_control(t1):
    cfg, jd, _ = t1
    _, arrays = pl.load_journal(jd)
    for s in (1, 2):
        r = rd.sp_op_check(cfg, s, arrays[f"s{s}_solved_set"], columns=list(range(cfg.ny - 1 if s == 1 else cfg.nx - 1)))
        assert r["status"] == "HOLDS" and r["entries_checked"] == (cfg.ny - 1 if s == 1 else cfg.nx - 1) ** 2
    from kkvim_reference import r5_cccc_halfstep as hs
    p = hs.HalfStep(cfg.lx, cfg.nx, cfg.ly, cfg.ny, "Y", arrays["s1_solved_set"])
    A, B, C, F = hs.exact_coefficients(p)
    wrong = (A, B, [[C[0][0] * (1 + Fraction(1, 2**40))]], F)
    assert rd.sp_op_check(cfg, 1, arrays["s1_solved_set"], coeffs_override=wrong)["status"] == "VIOLATED"


# --- block C ----------------------------------------------------------------------------------------------------------------------

def test_block_c_paired_rows_bind_to_the_chain(tmp_path):
    cfg = pl.op8().TrajectoryConfig(1.0, 8, 1.25, 10, 1, s_max=2)
    doc = rd.bcd().write_paired(cfg, tmp_path / "c")
    assert all(x["executed"] and x["crosscheck"]["bitwise"] for x in doc["rows"])
    spec = doc["rows"][1]["spec"]
    assert rd.bcd().paired_chain_binding(spec) == []
    other = {**spec, "paired_sha256": {**spec["paired_sha256"], "given_in": "0" * 64}}
    assert rd.bcd().paired_chain_binding(other)
    code = ("import importlib.util, json, sys; s = importlib.util.spec_from_file_location('r', sys.argv[1]); r = "
            "importlib.util.module_from_spec(s); sys.modules['r'] = r; s.loader.exec_module(r); x = r.analyse_ready(json.loads(sys.argv[2]));"
            " print(json.dumps([x['premises']['P-a']['state'], x['c3plus']['P-4'], x['status'], x['blas_threads_read']]))")
    p = subprocess.run([sys.executable, "-c", code, str(READY), json.dumps(spec)], cwd=ROOT, capture_output=True, text=True, timeout=600)
    assert p.returncode == 0, p.stderr[-2000:]
    pa, p4, st, thr = json.loads(p.stdout.strip().splitlines()[-1])
    assert (pa, p4, thr) == ("ESTABLISHED", "ASSUMED", 1) and st in ("PASS", "UNRESOLVED_RESOLVABILITY")


def test_od6_thread_count_other_than_one_is_not_a_binding(t1, monkeypatch):
    """OD-6: a row analysed in a process whose effective BLAS thread count is not 1 is UNRESOLVED_CERTIFICATE (П-а), never PASS."""
    cfg, jd, _ = t1
    monkeypatch.setattr(rd, "blas_threads_read", lambda: 24)
    r = rd.analyse_ready({"kind": "journal", "journal": str(jd), "s": 1}, reproduce=False)
    assert r["status"] == "UNRESOLVED_CERTIFICATE" and "OD-6" in r["premises"]["P-a"]["reason"]


def test_block_c_rac_chain_n23_has_no_qr_transform():
    cfg = pl.op8().TrajectoryConfig(1.0, 8, 1.25, 10, 3, s_max=4)
    ch = rd.bcd().rac_chain(cfg)
    assert ch["records"][0]["status"] == "OK" and ch["records"][0]["qr_transform"] is False
    assert [r["status"].split(":")[0] for r in ch["records"][1:]] == ["STOPPED_RANK_GUARD_UNCERTIFIED"] * 3
    assert sorted(ch["states"]) == [1]
    neg = rd.bcd().rac_chain(pl.op8().TrajectoryConfig(1.0, 12, 1.0, 4, 3, s_max=4))
    assert neg["records"][0]["status"] == pl.op8().STOP_CONFIG and not neg["states"]


# --- probes: resources only ------------------------------------------------------------------------------------------------------

def test_probe_smoke_records_no_row_value(tmp_path):
    doc = rd.run_probes(["SMOKE:8x10:N1:s2", "SMOKE:8x10:N3:s1"], tmp_path)
    text = json.dumps(doc)
    for k in ("P_hat", "Q_hat", "gamma_exact", "B_E_up", "tau"):
        assert f'"{k}' not in text, k
    assert all(r["result"] == "WITHIN_PL63" for r in doc["results"])
    assert doc["surrogate_definition"] == ct.load_expected()["probes"]["surrogate_definition"]
    with pytest.raises(Exception):
        rd._probe_child("B:160x640:N3:s2", 1, str(tmp_path / "f"))


def test_probe_list_is_the_expected_one():
    exp = ct.load_expected()["probes"]
    assert list(rd.PROBES) == exp["list"]
    assert list(rd.PROBE_SP_BE) == exp["second_paths_in_probes"]["SP-BE"] and list(rd.PROBE_SP_OP) == exp["second_paths_in_probes"]["SP-OP"]


# --- committed results ------------------------------------------------------------------------------------------------------------

def test_committed_results_are_consistent():
    s = json.loads((RESULTS / "controls_summary.json").read_text(encoding="utf-8"))
    assert s["all_match"] is True and s["platform"]["python"] == "3.13.15" and s["freeze"]["allowed"] is True
    assert s["expected_sha256"] == ct.EXPECTED_SHA256
    p = json.loads((RESULTS / "resource_probes.json").read_text(encoding="utf-8"))
    assert p["platform"]["python"] == "3.13.15" and p["freeze"]["allowed"] is True
    assert [r["probe"] for r in p["results"]] == list(rd.PROBES) and all(r["result"] == "WITHIN_PL63" for r in p["results"])
    a = json.loads((RESULTS / "start_archive.json").read_text(encoding="utf-8"))
    if rd.pf()._same_hash_platform(s["hash_platform"]):
        assert rd.verify_archive(a) == []  # platform dependent (OD-12): strict on the recorded platform only
    else:
        pytest.skip("start sets depend on numpy.sin: compared on the recorded platform only (OD-12)")


# --- no import of the working kkvim ----------------------------------------------------------------------------------------------

def test_no_import_of_the_working_kkvim():
    for path in (READY, CONTROLS, BLOCK_C):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names = [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
        names += [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        assert not [m for m in names if m == "kkvim" or m.startswith("kkvim.")], path
