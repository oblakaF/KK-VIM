"""Tests of the PRELAUNCH N = 1 row pipeline scripts/r5_cccc_q4_d1_row_pipeline_prelaunch.py (protocol 4.1-4.2, 5, 12; task: Z-3).

Small non-D1 configurations only (T1 of the frozen control expectations: 8 x 10 on 1 x 1.25, N = 1). Checked: the full path from an
OP-8 journal to the classification; exact dP and tau' at the boundary with the Z-3 counterexamples; no false PASS with an incomplete
certificate; E1/E2, E_DIAG and the strict sign of resolvability; real detectors; binding, hash chain and reproduction in a separate
process; the D0 regression against the published EQ / QA-EB rows; the guard against the D1 campaign.
"""

import ast
import csv
import importlib.util
import json
import math
import shutil
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "scripts" / "r5_cccc_q4_d1_row_pipeline_prelaunch.py"


def _load():
    spec = importlib.util.spec_from_file_location("_pl_pipeline", TOOL)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


pl = sys.modules.get("_pl_pipeline") or _load()
OK = {p: {"state": "ESTABLISHED", "reason": "synthetic"} for p in ("P-a", "P-b", "P-c", "P-d", "P-e", "P-f")}


@pytest.fixture(scope="module")
def journal(tmp_path_factory):
    cfg = pl.op8().TrajectoryConfig(1.0, 8, 1.25, 10, 1, 1.0, 1.0, 3)
    return pl.run_trajectory_journal(cfg, tmp_path_factory.mktemp("pl_T1"))


@pytest.fixture(scope="module")
def rows(journal):
    return pl.analyse_journal(journal)


def _states(r):
    return {k: v["state"] for k, v in r["premises"].items()}


# --- 1. full path ----------------------------------------------------------------------------------------------------------------

def test_full_path_from_the_op8_journal_to_the_classification(journal, rows):
    j, arrays = pl.load_journal(journal)
    assert [r["status"] for r in rows] == ["PASS"] * 3
    for s, r in enumerate(rows, start=1):
        rec = j["steps"][s - 1]
        assert set(_states(r).values()) == {"ESTABLISHED"}
        assert r["reproduction"]["bitwise_equal"] is True and r["op8_rerun"]["reproduced"] is True
        assert r["row"]["field_s_sha"] == rec["field_s_sha"] and r["row"]["field_prev_sha"] == rec["prev_field_after_QR_sha"]
        c = r["classification"]
        # dP and tau' are the exact rationals of the recorded binary64 components (protocol 4.1)
        assert Fraction(c["dP_exact"]) == Fraction(float.fromhex(r["row"]["P_hat_p"])) - Fraction(float.fromhex(r["row"]["P_hat_s"]))
        assert Fraction(c["tau_exact"]) == Fraction(float.fromhex(r["row"]["Gamma_up"])) + Fraction(float.fromhex(r["row"]["B_E_up"]))
        assert c["e1"] is True and c["e2"] is True and r["contradictions"] == []
        assert r["row"]["diagnostics"]["second_path_gamma"]["consistent"] is True
        assert r["data_status"].startswith("PRELAUNCH") and "never D1_RESULT" in r["data_status"]


# --- 2. Z-3 at the boundary ------------------------------------------------------------------------------------------------------

def test_z3_exact_dP_and_tau_and_rejection_of_recorded_values():
    big = "0x1.0000000000001p+53"  # 2^53 + 2 = fl((2^53 + 2) - 0.5)
    comp = {"P_p": big, "P_s": 0.5, "Q": big, "Gamma_up": 0.125, "B_E": 0.125}
    r = pl.assemble(comp, OK)
    assert r["dP_exact"] == f"{2**54 + 3}/2" and r["status"] == "FAIL_COMPUTATION"  # exact |dP - Q^| = 1/2 > 1/4
    q = 0.1 + 0.2
    r = pl.assemble({"P_p": q, "P_s": 0.0, "Q": q, "Gamma_up": 0.1, "B_E": 0.2}, OK)
    assert Fraction(r["tau_exact"]) == Fraction(0.1) + Fraction(0.2) < Fraction(q) and r["status"] == "PASS"
    # counterexamples of Z-3: a recorded value that is not the definition is never accepted, never PASS
    for rec in ({"dP": "hex:" + big}, {"dP": str(Fraction(2**54 + 3, 2) + Fraction(1, 2**1080))}):
        x = pl.assemble(comp, OK, rec)
        assert x["rejected"] and x["status"] == "UNRESOLVED_CERTIFICATE" and x["source_undetermined"] and x["dP_exact"] is None
    x = pl.assemble({"P_p": q, "P_s": 0.0, "Q": q, "Gamma_up": 0.1, "B_E": 0.2}, OK, {"tau": q})
    assert x["rejected"] and x["status"] == "UNRESOLVED_CERTIFICATE" and x["tau_exact"] is None
    x = pl.assemble({"P_p": 3.0, "P_s": 2.0, "Q": 0.25, "Gamma_up": 0.25, "B_E": 0.25}, OK, {"tau": "1"})  # would hide FAIL
    assert x["rejected"] and x["status"] not in ("PASS", "FAIL_COMPUTATION")
    x = pl.assemble({"P_p": 3.0, "P_s": 2.0, "Q": 1.0, "Gamma_up": 0.25}, OK, {"tau": "1/2"})  # recorded tau' with B_E absent
    assert x["rejected"] and x["status"] == "UNRESOLVED_CERTIFICATE"
    x = pl.assemble({"P_p": 3.0, "P_s": 2.0, "Q": 1.0, "Gamma_up": 0.25, "B_E": 0.25}, OK, {"dP": 1.0, "tau": "1/2"})
    assert not x["rejected"] and x["status"] == "PASS"
    with pytest.raises(ValueError):
        pl.assemble({"P_p": 1.0, "P_s": 0.0, "Q": 1.0, "Gamma_up": 0.0, "B_E": 0.0}, OK, {"tau": float("inf")})


# --- 3. no false PASS with an incomplete certificate ---------------------------------------------------------------------------

def test_no_false_pass_with_an_incomplete_certificate():
    good = {"P_p": 3.0, "P_s": 2.0, "Q": 1.0, "Gamma_up": 0.25, "B_E": 0.25}
    assert pl.assemble(good, OK)["status"] == "PASS"
    for p in OK:
        for st in ("NOT_ESTABLISHED", "NOT_CHECKABLE"):
            r = pl.assemble(good, {**OK, p: {"state": st, "reason": "x"}})
            assert r["status"] == "UNRESOLVED_CERTIFICATE" and r["reason_premises"] == [p]
    for k in good:
        for v in (None, "inf", "nan"):
            r = pl.assemble({**good, k: v}, OK)
            assert r["status"] == "UNRESOLVED_CERTIFICATE", (k, v)
    r = pl.assemble({k: v for k, v in good.items() if k != "B_E"}, OK)
    assert r["tau_exact"] is None and r["status"] == "UNRESOLVED_CERTIFICATE"  # no assumed component
    r = pl.not_executed({"status": "UNRESOLVED_C3PLUS_PREMISE: P-3: FTZ/DAZ probe"}, 2)
    assert r["status"] == "UNRESOLVED" and "OD-8" in r["classification"]["reasons"][0]
    for st in ("FAIL_C3PLUS_BOUND_CHECK: x", "PROTECTIVE_STOP: x", "STOP_ZERO_FIELD: x", "FAIL_TOOL: x"):
        assert pl.not_executed({"status": st}, 3)["status"] == "UNRESOLVED"
    assert pl.not_executed(None, 4)["status"] == "UNRESOLVED"


def test_reproduction_is_required_for_p_a(journal):
    r = pl.analyse({"kind": "journal", "journal": str(journal), "s": 2}, reproduce=False)
    assert r["premises"]["P-a"]["state"] == "NOT_CHECKABLE" and r["status"] == "UNRESOLVED_CERTIFICATE"
    v = float.fromhex(pl.analyse({"kind": "journal", "journal": str(journal), "s": 2}, reproduce=False)["row"]["Q_hat"])
    r = pl.analyse({"kind": "journal", "journal": str(journal), "s": 2}, bound_values={"Q_hat": math.nextafter(v, 0).hex()})
    assert r["premises"]["P-a"]["state"] == "NOT_ESTABLISHED" and r["status"] == "UNRESOLVED_CERTIFICATE"


def test_a_value_not_reproduced_in_the_separate_process_breaks_p_a(journal, monkeypatch):
    real = pl.run_child

    def differing(spec, timeout=1800):
        job = real(spec, timeout)
        v = math.nextafter(float.fromhex(job["row"]["Q_hat"]), 0.0)
        job["row"]["Q_hat"], job["row"]["components"]["Q"] = v.hex(), v
        return job

    monkeypatch.setattr(pl, "run_child", differing)
    r = pl.analyse({"kind": "journal", "journal": str(journal), "s": 2})
    assert r["reproduction"]["bitwise_equal"] is False and "Q_hat" in r["reproduction"]["differences"]
    assert r["premises"]["P-a"]["state"] == "NOT_ESTABLISHED" and r["status"] == "UNRESOLVED_CERTIFICATE"


# --- 4. E1/E2, E_DIAG, strict sign -----------------------------------------------------------------------------------------------

def test_e1_e2_e_diag_and_the_strict_sign():
    r = pl.assemble({"P_p": 2.5, "P_s": 2.0, "Q": 0.5, "Gamma_up": 0.25, "B_E": 0.25}, OK)
    assert r["status"] == "UNRESOLVED_RESOLVABILITY"  # Q^ = tau'
    r = pl.assemble({"P_p": "0x1.0000000000001p-1", "P_s": 0.0, "Q": "0x1.0000000000001p-1", "Gamma_up": 0.25, "B_E": 0.25}, OK)
    assert r["status"] == "PASS"  # Q^ = tau' + 2^-53
    r = pl.assemble({"P_p": 0.0, "P_s": 0.5, "Q": 0.0, "Gamma_up": 0.25, "B_E": 0.25}, OK)
    assert (r["e1"], r["e2"], r["status"]) == (True, True, "UNRESOLVED_RESOLVABILITY")  # E1, E2 non-strict boundaries
    r = pl.assemble({"P_p": 3.0, "P_s": 2.0, "Q": 0.25, "Gamma_up": 0.25, "B_E": 0.25}, {**OK, "P-d": {"state": "NOT_ESTABLISHED"}})
    assert (r["status"], r["e1"], r["e2"], r["e_diag"]) == ("UNRESOLVED_CERTIFICATE", True, False, True)


# --- 5. real detectors ------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("injection, premise", [("perturb_field_s_rel_2^-40", "P-b"), ("subnormal_field_s_node", "P-c"),
                                                ("grid_D_2^-1074", "P-d"), ("mu_low_zero", "P-f"), ("B_E_sqrt_negative_D", "P-f")])
def test_injections_reach_the_real_detectors(journal, injection, premise):
    r = pl.analyse({"kind": "journal", "journal": str(journal), "s": 2, "injection": injection})
    assert r["status"] == "UNRESOLVED_CERTIFICATE" and premise in r["classification"]["reason_premises"]


def test_n7_detector_and_source_of_the_fault(journal):
    r = pl.analyse({"kind": "journal", "journal": str(journal), "s": 1, "injection": "prev_factors_2^-600"})
    assert "P-e" in r["classification"]["reason_premises"] and r["row"]["N7_applicable"] is True
    r = pl.analyse({"kind": "journal", "journal": str(journal), "s": 2, "injection": "scale_field_s_2^1000"})
    assert {"P-c", "P-f"} <= set(r["classification"]["reason_premises"]) and r["status"] == "UNRESOLVED_CERTIFICATE"


# --- 9. binding, hash chain, binary representation -------------------------------------------------------------------------------

def test_hash_chain_and_binding(journal, tmp_path):
    j, arrays = pl.load_journal(journal)
    assert all(v["ok"] for v in pl.bind_journal(j, arrays).values())
    assert pl.bind_journal(j, arrays)[2]["ok"] and j["steps"][1]["given_in_sha"] == j["steps"][0]["next_given_sha"]
    # a significant input changed by 1 ulp is always detected
    a = dict(arrays)
    a["s1_next_given"] = a["s1_next_given"].copy()
    a["s1_next_given"].flat[0] = math.nextafter(float(a["s1_next_given"].flat[0]), math.inf)
    b = pl.bind_journal(j, a)
    assert not b[1]["ok"] and not b[2]["ok"] and any("chain" in x for x in b[2]["reasons"])
    # a hash binding is not a coincidence of values: -0.0 for +0.0 leaves the computed values but breaks the binding
    d = tmp_path / "negzero"
    shutil.copytree(journal, d)
    arr = pl.op8().load_arrays(d)
    arr["s1_companion_in"] = arr["s1_companion_in"].copy()
    arr["s1_companion_in"].flat[0] = -0.0
    np.savez(d / "arrays.npz", **arr)
    clean = pl.analyse({"kind": "journal", "journal": str(journal), "s": 1}, reproduce=False)
    neg = pl.analyse({"kind": "journal", "journal": str(d), "s": 1})
    assert all(neg["row"][k] == clean["row"][k] for k in ("P_hat_p", "P_hat_s", "Q_hat", "Gamma_up", "B_E_up"))
    assert neg["premises"]["P-a"]["state"] == "NOT_ESTABLISHED" and neg["status"] == "UNRESOLVED_CERTIFICATE"


def test_binary_representation_is_kept():
    for x in (0.1, -0.0, 2.0**-1074, 1.7976931348623157e308, -3.5):
        assert float.fromhex(pl.fhex(x)).hex() == x.hex()
    f = Fraction(3, 2**1080) + 1
    assert Fraction(pl.ftext(f)) == f and pl.parse_recorded("hex:" + (0.1).hex()) == Fraction(0.1)
    assert pl.parse_recorded(pl.ftext(f)) == f


# --- D0 regression (published EQ and QA-EB rows; bitwise on the platform of their run) -------------------------------------------

@pytest.mark.skipif(sys.platform != "win32", reason="published D0 values are bound to the Windows platform of their run (D-025)")
def test_d0_regression_bitwise():
    ga = pl.ga()
    q3 = ga.read_q3_rows(ROOT / "docs" / "results" / "r5_cccc_q3_prequalification" / "sequences_e1_e7_e9.csv")
    with open(ROOT / "docs" / "results" / "r5_cccc_q4_energy_error_bound" / "rows.csv", encoding="utf-8") as f:
        qaeb = {r["id"]: r for r in csv.DictReader(f)}
    with open(ROOT / "docs" / "results" / "r5_cccc_q4_energy_bc" / "rows.csv", encoding="utf-8") as f:
        eqr = {r["id"]: r for r in csv.DictReader(f)}
    sel = [r for r in q3 if r["N"] == "1" and r["j_s"] == "1" and r["first"] == "Y" and int(r["nx"]) * int(r["ny"]) <= 4096]
    assert len(sel) == 9
    for r in sel:
        sp, rid = ga.spec_of(r), ga.row_id(r)
        cfg = pl.op8().TrajectoryConfig(sp["lx"], sp["nx"], sp["ly"], sp["ny"], 1, 1.0, 1.0, 1)
        st = pl.op8().run_step(cfg, 1, pl.hs.start_set(sp["nx"], 1), np.zeros((sp["ny"] - 1, 1)))
        row = pl.compute_row(cfg, 1, st.arrays["solved_set"], st.arrays["companion_solved"], st.arrays["V_bc"])
        h, e = qaeb[rid], eqr[rid]
        assert h["status"] == "VERIFIED"
        assert row["P_hat_p"] == ga.read_double(h["P_hat_prev"]).hex() and row["P_hat_s"] == ga.read_double(h["P_hat_s"]).hex()
        assert row["Q_hat"] == ga.read_double(h["Q_hat"]).hex() and row["B_E_up"] == ga.read_double(h["B_E_up"]).hex()
        assert row["Gamma_up"] == ga.read_double(e["Gamma_up"]).hex() and row["gamma_down"] == ga.read_double(e["gamma_down"]).hex()
        assert set(v["state"] for v in row["premises"].values()) == {"ESTABLISHED"} and row["contradictions"] == []


# --- 10. guard against the D1 campaign -------------------------------------------------------------------------------------------

def test_guard_against_d1(tmp_path):
    TC = pl.op8().TrajectoryConfig
    for nx, ny in pl.op8().D1_GRIDS:
        if nx >= 3 and ny >= 3:
            with pytest.raises(pl.D1Refused):
                pl.guard_config(TC(1.0, nx, 1.0, ny, 1, s_max=1))
    with pytest.raises(pl.D1Refused):
        pl.guard_config(TC(1.0, 8, 1.25, 10, 2, s_max=1))
    with pytest.raises(pl.D1Refused):
        pl.guard_out(ROOT / "docs" / "results" / "x")
    with pytest.raises(pl.D1Refused):
        pl.compute_row(TC(1.0, 50, 1.0, 50, 1, s_max=1), 1, np.zeros((49, 1)), np.zeros((49, 1)), np.zeros((49, 1)))
    for args in (["--nx", "50", "--ny", "50", "--out", str(tmp_path / "x")], ["--nx", "8", "--ny", "10", "--out", "docs/x"]):
        p = subprocess.run([sys.executable, str(TOOL), "run", "--lx", "1", "--ly", "1", *args], cwd=ROOT, capture_output=True,
                           text=True)
        assert p.returncode == 2 and "refused" in p.stderr
    assert not (ROOT / "docs" / "x").exists()
    tree = ast.parse(TOOL.read_text(encoding="utf-8"))
    names = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)} | {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    assert not names & {"campaign_verdict", "mandatory_rows", "build_manifest", "run_probes", "write_manifest", "D1_RESULT"}
    assert "NOT D1 data" in pl.DATA_STATUS and pl.D1_START == "BLOCKED"


def test_tools_used_read_only():
    out = subprocess.run(["git", "status", "--porcelain", "--", *pl.USED_TOOLS, "src"], cwd=ROOT, capture_output=True, text=True)
    assert out.returncode != 0 or out.stdout.strip() == ""
    assert json.loads(json.dumps(pl.provenance(), default=pl._json_default))["tool_blob"] == pl.git_blob_id(TOOL)
