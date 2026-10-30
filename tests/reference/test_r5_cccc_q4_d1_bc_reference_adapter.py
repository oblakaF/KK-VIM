"""Tests of scripts/r5_cccc_q4_d1_bc_reference_adapter.py: the certified BR reference applied to the ACTUAL V_bc of an OP-8 C-3+
half-step (N-1), never to the OLD assembly and never through a B-C call; refusal (UNRESOLVED) without the certificate, with an
unconfirmed input or without the budget 1/100. Small non-D1 configuration (T1), N = 1."""

import importlib.util
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "scripts" / "r5_cccc_q4_d1_bc_reference_adapter.py"


def _load():
    spec = importlib.util.spec_from_file_location("_ad_adapter", TOOL)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


ad = sys.modules.get("_ad_adapter") or _load()


@pytest.fixture(scope="module")
def journal(tmp_path_factory):
    cfg = ad.pl().op8().TrajectoryConfig(1.0, 8, 1.25, 10, 1, 1.0, 1.0, 2)
    return ad.pl().run_trajectory_journal(cfg, tmp_path_factory.mktemp("ad_T1"))


@pytest.fixture
def no_bc_calls(monkeypatch):
    """The adapter must never call the B-C solver (a B-C call is not an independent reference)."""
    from kkvim_reference import r5_cccc_bc as bc

    def boom(*a, **k):
        raise AssertionError("B-C solver called by the reference adapter")

    for name in ("run_halfstep", "solve_bc"):
        monkeypatch.setattr(bc, name, boom)
    monkeypatch.setattr(ad.pl().op8(), "solve_with_c", boom)
    monkeypatch.setattr(ad.pl().op8(), "run_step", boom)


def test_reference_of_the_c3plus_output_not_old(journal, no_bc_calls):
    j, arrays = ad.pl().load_journal(journal)
    for s in (1, 2):
        rec = j["steps"][s - 1]
        r = ad.reference_for_journal(journal, s)
        assert r["input_confirmed"] and r["result"].startswith("RESOLVED")
        assert r["V_sha"] == rec["V_bc_sha"] and r["C3plus_sha"] == rec["C3plus_sha"]
        assert r["K_SPD_certified"] and r["lambda_low"] > 0 and r["refine_converged"]
        assert r["budget_set"] == r["budget_field"] == r["budget_inf"] == "SATISFIED"
        assert r["exact_ref_within_bound"] and r["exact_bc_error_in_interval"] and r["rh_consistent"]
        assert ad.decimal_value(r["set_error_2_down"]) <= ad.decimal_value(r["set_error_2_up"])
        assert r["C_old_equals_C3plus_bitwise"] is False  # the assemblies differ on this input: the measured object is C-3+


def test_the_old_solution_is_a_different_object(journal):
    """On the same fixed input the OLD solve of the prototype gives another binary64 solution; the adapter measured V_bc of C-3+."""
    from kkvim_reference import r5_cccc_bc as bc

    j, arrays = ad.pl().load_journal(journal)
    cfg = ad.pl().config_of(j["meta"])
    p = ad.hs.HalfStep(cfg.lx, cfg.nx, cfg.ly, cfg.ny, "Y", arrays["s1_solved_set"])
    old = bc.run_halfstep(p, qr=False).bc.s.reshape(1, -1).T
    assert old.tobytes() != arrays["s1_V_bc"].tobytes()
    r_old = ad.reference(cfg, 1, arrays["s1_solved_set"], old)
    r_new = ad.reference_for_journal(journal, 1)
    assert r_old["V_sha"] != r_new["V_sha"] and r_old["set_error_2_up"] != r_new["set_error_2_up"]


def test_refusal_without_certificate(journal, no_bc_calls):
    r = ad.reference_for_journal(journal, 1, tau_override=1e30)  # t far above lambda_min: K - t I not SPD, no certificate
    assert r["result"].startswith("UNRESOLVED") and r["K_SPD_certified"] is False and "RESOLVED (" not in r["result"]


def test_refusal_without_budget(journal, no_bc_calls):
    r = ad.reference_for_journal(journal, 1, ladder=(2,))  # 2-digit factor: no proven budget (or no factorization)
    assert r["result"].startswith("UNRESOLVED")


def test_refusal_of_an_unconfirmed_input(journal, tmp_path, no_bc_calls):
    d = tmp_path / "j"
    shutil.copytree(journal, d)
    j = json.loads((d / "journal.json").read_text(encoding="utf-8"))
    j["steps"][0]["only_C_replaced"] = False
    (d / "journal.json").write_text(json.dumps(j), encoding="utf-8")
    r = ad.reference_for_journal(d, 1)
    assert r["result"].startswith("UNRESOLVED") and r["input_confirmed"] is False and "C-3+" in r["result"]
    shutil.rmtree(d)
    shutil.copytree(journal, d)
    arr = ad.pl().op8().load_arrays(d)
    arr["s1_V_bc"] = arr["s1_V_bc"].copy()
    arr["s1_V_bc"].flat[0] = math.nextafter(float(arr["s1_V_bc"].flat[0]), math.inf)
    np.savez(d / "arrays.npz", **arr)
    r = ad.reference_for_journal(d, 1)
    assert r["result"].startswith("UNRESOLVED") and "V_bc" in r["result"]


def test_guard(journal):
    cfg = ad.pl().op8().TrajectoryConfig(1.0, 50, 1.0, 50, 1, s_max=1)
    with pytest.raises(ad.pl().D1Refused):
        ad.reference(cfg, 1, np.ones((49, 1)), np.ones((49, 1)))
