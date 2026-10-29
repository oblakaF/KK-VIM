"""Tests of the D1 PRELAUNCH QA tool scripts/r5_cccc_q4_d1_preflight.py (protocol docs/R5_CCCC_Q4_D1_PROTOCOL.md 12, items 3-5).

The tool prepares the conditions of the D1 campaign; it does not run it (D1_START = BLOCKED). These tests check: the manifest against
the protocol text and against git; the 102 mandatory keys; the reproducibility of the hashes; the integrity of the second-path sample;
the isolated classifier against the expectations frozen before it was written (controls A and B); the refusals and the smoke run of the
resource probe; that no row value of a probe is recorded.
"""

import ast
import csv
import hashlib
import importlib.util
import json
import re
import shutil
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "scripts" / "r5_cccc_q4_d1_preflight.py"
OUT = ROOT / "docs" / "results" / "r5_cccc_q4_d1_prelaunch"
PROTOCOL = (ROOT / "docs" / "R5_CCCC_Q4_D1_PROTOCOL.md").read_text(encoding="utf-8")


def _load():
    spec = importlib.util.spec_from_file_location("r5_cccc_q4_d1_preflight", TOOL)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


pf = _load()


def _pairs(text: str) -> list[tuple[int, int]]:
    return [(int(a), int(b)) for a, b in re.findall(r"\((\d+), (\d+)\)", text)]


def _section(start: str, end: str) -> str:
    i = PROTOCOL.index(start)
    return PROTOCOL[i:PROTOCOL.index(end, i)]


@pytest.fixture(scope="module")
def manifest():
    return pf.build_manifest()


@pytest.fixture(scope="module")
def committed():
    return json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))


# --- manifest against the protocol text ---------------------------------------------------------------------------------------------

def test_grids_are_the_protocol_tables():
    sec = _section("### 2.2.", "### 2.3.")
    rows = [ln for ln in sec.splitlines() if ln.startswith("| ") and "(" in ln]
    table = {ln.split("|")[1].strip(): _pairs(ln) for ln in rows}
    assert table == {r: list(g) for r, g in pf.FAMILIES}
    aniso = _pairs(_section("### 2.3.", "## 3."))
    assert aniso[:6] == list(pf.ANISOTROPIC) and len(aniso) == 6
    sec3 = _section("## 3. Блоки выборки", "**Арифметическая сверка**")
    assert "(9, 11), (11, 15), (14, 6) при N = 1" in sec3 and "(14, 6) при N = 3" in sec3
    assert "(400, 1600)" in sec3 and "(1600, 400)" in sec3
    i = PROTOCOL.index("≤ 4096, как в BR")
    rh = _pairs(PROTOCOL[i:PROTOCOL.index("и все", i)])
    assert rh == list(pf.PROTOCOL_RH_GRIDS)
    computed = [(g["nx"], g["ny"]) for g in pf.block_a_grids() if (g["nx"] - 1) * (g["ny"] - 1) <= pf.RH_NODES]
    assert sorted(computed) == sorted(pf.PROTOCOL_RH_GRIDS)


def test_sample_arithmetic_of_section_3():
    assert pf.planned_row_counts() == {"A": 272, "B": 336, "C_paired": 168, "C_RAC_reference_chain": 168, "D": 32, "S": 16,
                                       "classified": 824, "mandatory": 102}


def test_geometry_table_of_section_2_1():
    """Protocol 2.1: fl(r) = r exactly except 2/3, 2/5 (relative deviation -5.55e-17, +5.55e-17); fl(h_x) != fl(h_y) only at (198, 132),
    (396, 264); exact h_x = h_y on all grids of the exact families, never for 2/3, 2/5 and the anisotropic grids."""
    g = pf.block_a_grids()
    for x in g:
        geo = pf.geometry(x)
        if x["family"] == "1x1":
            assert geo["h_exact_equal"] is False
            continue
        r = x["r"]
        exact = r not in ("2/3", "2/5")
        assert geo["ly_equals_r_exactly"] is exact
        assert geo["h_exact_equal"] is exact
        assert geo["h_double_equal"] is ((x["nx"], x["ny"]) not in ((198, 132), (396, 264)))
    for r, dev in (("2/3", "-5.55e-17"), ("2/5", "5.55e-17")):
        rel = (Fraction(float(Fraction(r))) - Fraction(r)) / Fraction(r)
        assert f"{float(rel):.2e}" == dev


def test_no_grid_of_d1_is_a_grid_of_d0():
    with open(ROOT / "docs" / "results" / "r5_cccc_q3_prequalification" / "sequences_e1_e7_e9.csv", encoding="utf-8") as f:
        labels = {r["grid"] for r in csv.DictReader(f)}
    assert len(labels) == 36  # the 36 grids of D0 (geometry and grid); as (n_x, n_y) pairs alone they are 34
    d0 = {tuple(int(v) for v in g.split(":")[1].split("x")) for g in labels}
    assert not d0 & {(x["nx"], x["ny"]) for x in pf.block_a_grids()}  # stronger than the protocol: no common (n_x, n_y) at all


# --- the 102 mandatory rows ---------------------------------------------------------------------------------------------------------

def test_102_unique_keys(manifest):
    rows = manifest["mandatory_rows"]
    keys = [r["key"] for r in rows]
    assert len(keys) == 102 == len(set(keys))
    assert {(r["nx"], r["ny"]) for r in rows} == {(x["nx"], x["ny"]) for x in pf.block_a_grids()}
    assert all(r["block"] == "A" and r["N"] == 1 and r["mandatory"] for r in rows)
    assert sorted({r["s"] for r in rows}) == [1, 2, 3]
    assert all(r["direction"] == ("Y" if r["s"] != 2 else "X") for r in rows)
    assert manifest["mandatory_keys_sha256"] == hashlib.sha256(pf.canonical(keys)).hexdigest()
    for r in rows:  # known inputs only at s = 1; s = 2, 3 are links to the journal of s - 1, never computed values
        if r["s"] == 1:
            assert r["inputs"]["state"] == "KNOWN_BEFORE_RUN" and len(r["inputs"]["given_in_sha"]) == 64
        else:
            assert r["inputs"]["state"] == "DYNAMIC" and r["inputs"]["given_in_sha"].startswith("= next_given_sha of D1-A|")


def test_input_hashes_reproducible(manifest, committed):
    tj = pf._load(pf.OP8_TOOL, "_op8_test")
    for g in pf.block_a_grids()[:5] + pf.block_a_grids()[-3:]:
        assert pf.known_input_hashes(g, tj) == pf.known_input_hashes(g, tj)
    if not pf._same_hash_platform(committed["hash_platform"]):
        pytest.skip("s = 1 hashes depend on numpy's sin: compared on the recorded platform only (OD-12)")
    assert [r["inputs"] for r in manifest["mandatory_rows"]] == [r["inputs"] for r in committed["mandatory_rows"]]


def _strip_platform(m: dict) -> dict:
    m = json.loads(json.dumps(m))
    m.pop("hash_platform")
    for r in m["mandatory_rows"]:
        if r["s"] == 1:
            r["inputs"].pop("given_in_sha")
            r["inputs"].pop("companion_in_sha")
    return m


def test_committed_manifest_is_reproduced(manifest, committed):
    if pf._same_hash_platform(committed["hash_platform"]):
        assert manifest == {**committed, "hash_platform": manifest["hash_platform"]}
        assert (OUT / "mandatory_rows.csv").read_text(encoding="utf-8") == pf.manifest_csv(manifest["mandatory_rows"])
    else:
        assert _strip_platform(manifest) == _strip_platform(committed)


def test_sha_inventory_against_git(manifest):
    inv = manifest["sha_inventory"]
    have = subprocess.run(["git", "cat-file", "-e", pf.BASE_COMMIT], cwd=ROOT, capture_output=True).returncode == 0
    for c in inv["components"] + inv["other_used_files"]:
        if not c.get("path"):
            continue
        assert c["blob"] == pf.git_blob_id(ROOT / c["path"])
        if have and c["path"] != pf.SELF:
            b = subprocess.run(["git", "rev-parse", f"{pf.BASE_COMMIT}:{c['path']}"], cwd=ROOT, capture_output=True, text=True)
            assert b.stdout.strip() == c["blob"], c["path"]
    missing = {c["id"] for c in inv["components"] if c["state"] == "MISSING"}
    assert missing == {"classifier_campaign", "row_pipeline", "block_C_driver", "B_E_independent_recount", "control_V_harness",
                       "control_G_harness"}
    assert all(c["blob"] is None and c["path"] is None for c in inv["components"] if c["state"] == "MISSING")


def test_dependency_closure_contains_the_op8_list(manifest):
    tj = pf._load(pf.OP8_TOOL, "_op8_test_closure")
    closure = set(manifest["sha_inventory"]["dependency_closure"])
    assert set(tj.USED_MODULES) <= closure
    for p in closure:
        assert (ROOT / p).exists()
    assert "src/kkvim/__init__.py" not in closure and not any(p.startswith("src/kkvim/") for p in closure)


# --- second-path sample -------------------------------------------------------------------------------------------------------------

class _OnlyFeatures(dict):
    def __getitem__(self, k):
        if k not in pf.SAMPLE_FEATURES:
            raise AssertionError(f"the sample rule read a non-feature field {k!r}")
        return super().__getitem__(k)


def test_second_path_sample_integrity(manifest, committed):
    sp = manifest["second_path"]
    assert sp == committed["second_path"]
    assert sp["sha256"] == hashlib.sha256(pf.canonical(sp["rows"])).hexdigest()
    keys = {r["key"] for r in manifest["mandatory_rows"]}
    assert {x["key"] for x in sp["rows"]} <= keys
    assert sp["counts"] == {"SP-RH": 33, "SP-LARGE": 27, "SP-GAMMA": 60, "SP-BR": 60, "SP-C": 60, "SP-BE": 18}
    rh = {(r["nx"], r["ny"]) for r in manifest["mandatory_rows"] if "SP-RH" in next(
        (x["paths"] for x in sp["rows"] if x["key"] == r["key"]), [])}
    assert rh == set(pf.PROTOCOL_RH_GRIDS)
    for r in manifest["mandatory_rows"]:  # features only: a row with result-like fields gives the same selection
        f = _OnlyFeatures({**{k: r[k] for k in pf.SAMPLE_FEATURES}, "Q_hat": 0.0, "status": "PASS"})
        assert pf.sample_paths(f) == pf.sample_paths({k: r[k] for k in pf.SAMPLE_FEATURES})


def test_owner_decisions_are_not_frozen(manifest):
    assert manifest["manifest_state"].startswith("INCOMPLETE - NOT FROZEN")
    assert manifest["d1_start"] == "BLOCKED"
    assert {d["id"] for d in manifest["owner_decisions"]} == {f"OD-{i}" for i in range(1, 13)}
    assert all(d["status"] == "OWNER_DECISION_REQUIRED" for d in manifest["owner_decisions"])
    assert "FROZEN" not in manifest["manifest_state"].replace("NOT FROZEN", "")


def test_n3_probes_are_planned_not_executed(manifest):
    b = manifest["blocked_probes_N23"]
    assert b["status"] == "NOT_EXECUTED"
    assert {p["probe"] for p in b["probes"]} == {k for k in pf.REFUSED if k.endswith(":N3")}
    sec8 = _section("**Обязательные пробы перед запуском**", "Для планирования")
    for p in b["probes"]:
        assert f"({p['nx']}, {p['ny']})" in sec8 and p["N"] == 3 and p["state"].startswith("BLOCKED: PRE_D1_RANK")


# --- classifier and controls A, B ---------------------------------------------------------------------------------------------------

def test_expectations_are_frozen(tmp_path):
    raw = (OUT / "control_ab_expected.json").read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(raw).hexdigest() == pf.EXPECTED_SHA256
    crlf = tmp_path / "crlf.json"  # a CRLF checkout of the same content is accepted
    crlf.write_bytes(raw.replace(b"\n", b"\r\n"))
    assert pf.load_expected(crlf)["row_cases"]
    bad = tmp_path / "e.json"
    bad.write_bytes(raw.replace(b'"PASS"', b'"FAIL_COMPUTATION"', 1))
    with pytest.raises(RuntimeError, match="modified"):
        pf.load_expected(bad)


@pytest.fixture(scope="module")
def controls():
    return pf.run_controls()


def test_controls_match_the_frozen_expectations(controls):
    s = controls["summary"]
    assert s["row_cases"] == s["row_cases_matching"] == 213
    assert s["aggregation_cases"] == s["aggregation_matching"] == 18
    assert s["control_A_all_match"] and s["control_B_synthetic_all_match"] and s["control_B_detector_functions_all_match"]
    if sys.platform == "win32":
        assert s["detector_matching"] == 12 and s["detector_not_executed"] == []
    else:
        assert s["detector_not_executed"] == ["B-D-c3", "B-D-c4"] and s["detector_matching"] == 10
    for d in controls["detector_cases"]:  # every injection gives UNRESOLVED_CERTIFICATE, never PASS or FAIL_COMPUTATION
        if d["match"] is not None:
            assert d["injected"]["status"] == "UNRESOLVED_CERTIFICATE" and d["clean"]["status"] == "PASS"


def test_committed_control_results_are_reproduced(controls):
    old = json.loads((OUT / "control_ab_results.json").read_text(encoding="utf-8"))
    assert old["expected_sha256"] == pf.EXPECTED_SHA256
    assert [(r["id"], r["actual"], r["match"]) for r in old["row_cases"]] == [(r["id"], r["actual"], r["match"])
                                                                           for r in controls["row_cases"]]
    assert [(r["id"], r["actual"], r["match"]) for r in old["aggregation_cases"]] == [
        (r["id"], json.loads(json.dumps(r["actual"])), r["match"]) for r in controls["aggregation_cases"]]
    if sys.platform == "win32":
        assert old["summary"] == controls["summary"]


def test_row_tree_boundaries():
    ok = {p: {"state": "ESTABLISHED"} for p in pf.PREMISES}
    c = pf.classify_row
    assert c({"executed": True, "premises": ok, "dP": "-1", "Q": "0", "tau": "1"})["status"] == "UNRESOLVED_RESOLVABILITY"  # E1 =
    assert c({"executed": True, "premises": ok, "dP": "-1", "Q": "1", "tau": "1"})["status"] == "FAIL_COMPUTATION"  # |dP - Q| = 2
    assert c({"executed": True, "premises": ok, "dP": 0.5, "Q": 0.5, "tau": 0.25})["status"] == "PASS"  # floats taken exactly
    assert c({"executed": True, "premises": ok, "dP": 0.1, "Q": 0.1, "tau": 0.1})["status"] == "UNRESOLVED_RESOLVABILITY"
    with pytest.raises(ValueError):
        c({"executed": True, "premises": ok, "dP": "1", "Q": "1", "tau": "-1/2"})
    with pytest.raises(ValueError):
        c({"executed": True, "premises": {**ok, "P-z": {"state": "ESTABLISHED"}}, "dP": "1", "Q": "1", "tau": "0"})
    with pytest.raises(ValueError):
        c({"executed": True, "premises": {**ok, "P-a": {"state": "OK"}}, "dP": "1", "Q": "1", "tau": "0"})
    with pytest.raises(TypeError):
        c({"executed": True, "premises": ok, "dP": True, "Q": "1", "tau": "0"})


def test_verdict_needs_the_102_rows(manifest):
    mand = pf.mandatory_index(manifest["mandatory_rows"])
    with pytest.raises(ValueError, match="102"):
        pf.campaign_verdict([], dict(list(mand.items())[:101]))


# --- resource probes ----------------------------------------------------------------------------------------------------------------

def test_probe_refusals(tmp_path):
    r = pf.run_probes(["D:11x15", "B:160x640:N3", "C:100x250:N3"], tmp_path)
    assert [x["result"] for x in r["results"]] == ["NOT_EXECUTED"] * 3
    assert "OD-1" in r["results"][0]["reason"] and "PRE_D1_RANK" in r["results"][1]["reason"]
    with pytest.raises(ValueError):
        pf.run_probes(["A:50x50"], tmp_path)
    assert {k.split(":")[0] for k in pf.PROBES} == {"A", "S"}  # N = 1 configurations of blocks A and S only


_VALUE_KEYS = re.compile(r"P_hat|Q_hat|gamma|B_E_up|tau|E_s|R_Sigma|^status$|field_error|set_error|bc_minus_ref|lambda_low$")


def _keys(o, acc):
    if isinstance(o, dict):
        for k, v in o.items():
            acc.add(k)
            _keys(v, acc)
    elif isinstance(o, list):
        for v in o:
            _keys(v, acc)
    return acc


def test_probe_smoke_records_no_row_value(tmp_path):
    r = pf.run_probes(["SMOKE:8x8"], tmp_path)
    x = r["results"][0]
    assert x["result"] == "WITHIN_PL63_UNDER_PROPOSED_DEFINITIONS", x
    assert x["stage_op8"]["completed"] and x["stage_qaeb"]["completed"] and x["stage_qaeb"]["B_E_evaluated"]
    assert x["op8_V_equals_EQ_path_V_bitwise"] is True
    assert x["blas_threads_read_from_library"] in (1, None)
    assert not [k for k in _keys(r, set()) if _VALUE_KEYS.search(k)]
    assert r["label"].startswith("PRELAUNCH_RESOURCE_PROBE")


def test_committed_probe_results():
    p = OUT / "resource_probes.json"
    r = json.loads(p.read_text(encoding="utf-8"))
    assert r["label"].startswith("PRELAUNCH_RESOURCE_PROBE") and "not D1 data" in r["data_status"]
    assert r["probe_code_sha256"] == pf.probe_code_sha()  # the probe code is the code that produced the file
    assert not [k for k in _keys(r, set()) if _VALUE_KEYS.search(k)]
    names = [x["probe"] for x in r["results"]]
    assert set(names) >= set(pf.PROBES) and not any(n.startswith("SMOKE") for n in names)
    for x in r["results"]:
        if x["probe"] in pf.REFUSED:
            assert x["result"] == "NOT_EXECUTED"
        else:
            assert x["N"] == 1 and x["s"] == 1


# --- the tool does not run D1 and changes nothing ------------------------------------------------------------------------------------

def test_tool_source_runs_no_campaign():
    tree = ast.parse(TOOL.read_text(encoding="utf-8"))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    for n in calls:
        f = n.func
        name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
        if name == "TrajectoryConfig":
            kw = {k.arg: k.value for k in n.keywords}
            assert isinstance(kw.get("s_max"), ast.Constant) and kw["s_max"].value == 1
            assert isinstance(n.args[4], ast.Constant) and n.args[4].value == 1  # N = 1
        assert name not in ("run_sequence", "run_campaign", "write_journal"), name
        if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Attribute) and f.value.attr == "environ":
            assert f.attr in ("get",), f"os.environ.{f.attr} is called"
    for n in ast.walk(tree):
        if isinstance(n, (ast.Assign, ast.AugAssign)):
            for t in (n.targets if isinstance(n, ast.Assign) else [n.target]):
                assert not (isinstance(t, ast.Subscript) and "environ" in ast.unparse(t)), "os.environ is modified"
                assert not (isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id in (
                    "bc", "hs", "en", "tj", "qa", "br")), f"module attribute written: {ast.unparse(t)}"


def test_only_new_files_in_the_branch():
    if shutil.which("git") is None or subprocess.run(["git", "cat-file", "-e", pf.BASE_COMMIT], cwd=ROOT).returncode:
        pytest.skip("base commit not available")
    d = subprocess.run(["git", "diff", "--name-status", pf.BASE_COMMIT, "--", "."], cwd=ROOT, capture_output=True, text=True).stdout
    for ln in d.splitlines():
        st, path = ln.split("\t", 1)
        assert st == "A", ln
        assert path in ("scripts/r5_cccc_q4_d1_preflight.py", "tests/reference/test_r5_cccc_q4_d1_preflight.py",
                        "docs/R5_CCCC_Q4_D1_PRELAUNCH.qa-report.md") or path.startswith("docs/results/r5_cccc_q4_d1_prelaunch/"), ln
