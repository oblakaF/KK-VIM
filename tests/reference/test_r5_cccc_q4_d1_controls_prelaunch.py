"""Tests of the integrated controls harness scripts/r5_cccc_q4_d1_controls_prelaunch.py against the expectations fixed before the code
(tests/reference/r5_cccc_q4_d1_controls_prelaunch_expected.json): control A (PR #48 harness + pipeline-boundary cases), control B
(real detectors), control V (positive, change confirmed), control G (mutations; a mutation detected by no channel is a gap, never
PASS), the BR adapter on the clean rows."""

import copy
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "scripts" / "r5_cccc_q4_d1_controls_prelaunch.py"
EXPECTED = ROOT / "tests" / "reference" / "r5_cccc_q4_d1_controls_prelaunch_expected.json"


def _load():
    spec = importlib.util.spec_from_file_location("_ct_controls", TOOL)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


ct = sys.modules.get("_ct_controls") or _load()


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    return ct.run_all(tmp_path_factory.mktemp("controls"))


def test_expectations_are_frozen(tmp_path):
    exp = ct.load_expected()
    assert len(exp["pipeline_boundary_cases"]) == 25 and len(exp["control_B"]) == 12 and len(exp["control_V"]) == 2
    assert len(exp["control_G"]) == 7
    bad = tmp_path / "e.json"
    bad.write_bytes(EXPECTED.read_bytes().replace(b'"PASS"', b'"FAIL_COMPUTATION"', 1))
    with pytest.raises(RuntimeError, match="modified"):
        ct.load_expected(bad)
    crlf = tmp_path / "crlf.json"
    crlf.write_bytes(EXPECTED.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
    assert ct.load_expected(crlf) == exp


def test_expectations_committed_before_the_code():
    if shutil.which("git") is None:
        pytest.skip("git not available")

    def first_commit(path):
        p = subprocess.run(["git", "log", "--diff-filter=A", "--format=%H", "--", path], cwd=ROOT, capture_output=True, text=True)
        return p.stdout.split()[-1] if p.returncode == 0 and p.stdout.split() else None

    e = first_commit("tests/reference/r5_cccc_q4_d1_controls_prelaunch_expected.json")
    if e is None:
        pytest.skip("history not available (shallow checkout)")
    tree = subprocess.run(["git", "ls-tree", "-r", "--name-only", e], cwd=ROOT, capture_output=True, text=True).stdout.split()
    for code in ("scripts/r5_cccc_q4_d1_row_pipeline_prelaunch.py", "scripts/r5_cccc_q4_d1_controls_prelaunch.py",
                 "scripts/r5_cccc_q4_d1_bc_reference_adapter.py"):
        assert code not in tree, code


def test_controls_match_the_expectations(result):
    s = result["summary"]
    assert s["control_A"]["status"] == "PASS" and s["control_A"]["boundary"] == "25/25"
    assert s["clean_rows"] == "5/5"
    b = result["control_B"]
    if sys.platform == "win32":
        assert b["not_executed"] == [] and b["matching"] == b["executed"] == 12
    else:
        assert b["not_executed"] == ["B-c1", "B-c2"] and b["matching"] == b["executed"] == 10
    for c in b["cases"]:
        if c["match"] is not None:
            assert c["actual"]["status"] == "UNRESOLVED_CERTIFICATE" and c["premise"] in c["actual"]["reason_premises"]
    assert s["control_V"]["status"] == "PASS"
    for c in result["control_V"]["cases"]:
        assert c["change_confirmed"] and c["verdict"] == "PASS" and all(c["changed"].values()) and all(c["unchanged"].values())
    g = result["control_G"]
    assert g["total"] == 7 and g["declared_channel"] == 6 and g["detected"] == 6 and g["matching"] == 6
    assert g["no_channel"] == ["G-5"] and g["gaps"] == ["G-5"] and g["control_status"].startswith("UNRESOLVED")
    assert s["control_G"]["detected"] == "6/7" and s["control_G"]["matching_declared"] == "6/6"
    assert set(g["channels_pending_OD4"]) == {"G-1", "G-2", "G-3", "G-4"}
    assert any("OD-11" in x for x in g["decisions_needed"])
    by = {c["id"]: c for c in g["cases"]}
    assert by["G-3"]["actual"]["status"] == by["G-4"]["actual"]["status"] == "FAIL_COMPUTATION"
    assert by["G-6a"]["actual"]["rejected"] and by["G-6b"]["actual"]["rejected"]
    # sources of the registered contradictions, not only the flags
    assert by["G-1"]["contradiction_sources"] == by["G-2"]["contradiction_sources"] == ["second_path_gamma"]
    assert by["G-3"]["contradiction_sources"] == by["G-4"]["contradiction_sources"] == ["theorem_E2"]
    # G-5: the mutation is effective (B_E changed), no channel sees it, the row stays PASS; not counted as a detection or a match
    g5 = by["G-5"]
    assert g5["match"] is None and g5["detected"] is None and g5["mutation_effective"]["B_E_exact"] is True
    assert g5["actual"]["status"] == "PASS" and g5["outcome"].startswith("GAP") and "MISSING" in g5["outcome"]
    for c in result["clean_rows"]:
        assert c["actual"]["c3plus"]["P-4"] == "ASSUMED" and c["actual"]["faults"] == []
    assert all(r.startswith("RESOLVED") for r in s["reference_adapter"])
    assert "NOT D1 data" in result["data_status"]


def test_an_undetected_mutation_is_a_gap_not_pass(result, tmp_path):
    exp = copy.deepcopy(ct.load_expected())
    g1 = next(c for c in exp["control_G"] if c["id"] == "G-1")
    g1["expected"] = {"detected_by": "E1/E2"}  # a channel that does not see the sign of gamma
    exp["control_G"] = [g1]
    journals = ct.make_journals({"rows": {"T1": exp["rows"]["T1"]}}, tmp_path)
    clean = {("T1", 2): ct.pl().analyse(ct._spec(journals, ["T1", 2]))}
    out = ct.control_g(exp, journals, tmp_path, clean)
    c = out["cases"][0]
    assert c["match"] is False and "gap" in c["outcome"] and out["control_status"].startswith("UNRESOLVED")


def test_the_second_path_channel_needs_the_registered_contradiction(tmp_path, monkeypatch):
    """A flag without the registered contradiction (its registration lost) is not a detection of G-1."""
    exp = copy.deepcopy(ct.load_expected())
    exp["control_G"] = [next(c for c in exp["control_G"] if c["id"] == "G-1")]
    journals = ct.make_journals({"rows": {"T1": exp["rows"]["T1"]}}, tmp_path)
    clean = {("T1", 2): ct.pl().analyse(ct._spec(journals, ["T1", 2]))}
    real = ct.pl().analyse

    def unregistered(spec, **kw):
        r = real(spec, **kw)
        r["contradictions"] = []
        return r

    monkeypatch.setattr(ct.pl(), "analyse", unregistered)
    out = ct.control_g(exp, journals, tmp_path, clean)
    c = out["cases"][0]
    assert c["fired"]["second_path_gamma"] is False and c["match"] is False and out["detected"] == 0 and out["gaps"] == ["G-1"]


def test_an_unconfirmed_change_makes_control_v_empty(tmp_path):
    exp = copy.deepcopy(ct.load_expected())
    v2 = next(c for c in exp["control_V"] if c["id"] == "V-2")
    v2["expected"]["changed"] = v2["expected"]["changed"] + ["P_hat_p"]  # P^_p of a zero previous field cannot change
    exp["control_V"] = [v2]
    journals = ct.make_journals({"rows": {"T1": exp["rows"]["T1"]}}, tmp_path)
    clean = {("T1", 1): ct.pl().analyse(ct._spec(journals, ["T1", 1]))}
    out = ct.control_v(exp, journals, tmp_path, clean)
    assert out["cases"][0]["verdict"].startswith("UNRESOLVED (empty control") and out["matching"] == 0


def test_output_guard():
    p = subprocess.run([sys.executable, str(TOOL), "run", "--out", "docs/results/x"], cwd=ROOT, capture_output=True, text=True)
    assert p.returncode == 2 and "refused" in p.stderr
