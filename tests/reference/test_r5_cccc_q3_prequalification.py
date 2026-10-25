"""Tests of the Q3 pre-qualification measurement script (scripts/r5_cccc_q3_prequalification.py): small configurations only (PL 6.3).

Checked: the fixed sample against PL 2.2 / BC-2 / RO 4.1; the association of the recorded arrays with the journal (bitwise); exact
identities and statuses; zero and undefined denominators; the reference budget rule; E-3 separation logic; E-5 exclusion (U-P3);
E-8 certified reference bound; output formatting (directed rounding, exact repr); reproducibility of a job; metadata; independence
from the working solver. No thresholds are tested (none exist).
"""

import ast
import csv
import importlib.util
import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from kkvim_reference import r5_cccc_energy as en
from kkvim_reference import r5_cccc_halfstep as hs

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "r5_cccc_q3_prequalification.py"
_spec = importlib.util.spec_from_file_location("r5_cccc_q3_prequalification", SCRIPT)
q3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(q3)

SMALL = {"group": "test", "r": "1.3", "lx": 1.0, "nx": 10, "ly": 1.3, "ny": 9, "first": "Y", "s_max": 3}


# --- sample ----------------------------------------------------------------------------------------------------------------------

def test_sample_matches_the_plan():
    assert q3.PAIRS == [(40, 64), (64, 40), (47, 53), (160, 32), (32, 160)]
    assert set(q3.RATIOS) == {"1", "2", "3", "5", "1/2", "1/3", "1/5"}
    assert q3.CHAIN == [32, 64, 128, 256, 512] and q3.E5_MAIN == q3.CHAIN and q3.E5_EXTENDED == [16, 1024, 2048]
    assert {r: q3.chain_limit(r) for r in q3.RATIOS} == {"1": 512, "2": 512, "1/2": 512, "3": 256, "1/3": 256, "5": 256,
                                                       "1/5": 256}
    specs = q3.sequence_specs()
    assert {s["N"] for s in specs} == {1, 2, 3}
    assert {(s["nx"], s["ny"]) for s in specs if s["group"] == "pairs"} == set(q3.PAIRS)
    assert {s["first"] for s in specs if s["group"] == "pairs"} == {"Y", "X"}
    assert all(s["first"] == "Y" for s in specs if s["group"] == "chain")  # D-049.3
    assert all(s["n"] <= q3.chain_limit(s["r"]) for s in specs if s["group"] == "chain")
    assert len(specs) == 5 * 3 * 2 + (3 * 5 + 4 * 4) * 3


@pytest.mark.parametrize("r, n, expected", [("1", 32, (1.0, 32, 1.0, 32)), ("2", 64, (1.0, 64, 2.0, 128)),
                                            ("1/5", 32, (1.0, 160, 0.2, 32)), ("1/3", 32, (1.0, 96, 1 / 3, 32))])
def test_plate_grid_has_equal_steps_and_n_on_the_short_side(r, n, expected):
    lx, nx, ly, ny = q3.plate_grid(r, n)
    assert (lx, nx, ly, ny) == expected
    assert min(nx, ny) == n
    assert math.isclose(lx / nx, ly / ny, rel_tol=1e-15)


def test_exact_tier_is_within_the_exact_limit():
    for lx, nx, ly, ny in q3.EXACT_TIER:
        for direction in hs.DIRECTIONS:
            n_s = ny if direction == "Y" else nx
            assert 3 * (n_s - 1) <= q3.e46.EXACT_LIMIT


# --- helpers ----------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("num, den, expected", [(1.0, 2.0, 0.5), (1.0, 0.0, "UNRESOLVED"), ("UNRESOLVED", 1.0, "UNRESOLVED"),
                                                (math.inf, 1.0, "UNRESOLVED"), (None, 1.0, "UNRESOLVED"), (True, 1.0, "UNRESOLVED")])
def test_ratio_handles_missing_and_zero_denominators(num, den, expected):
    assert q3.ratio(num, den) == expected


def test_budget_rule_is_exact_and_relative():
    assert q3.budget(Fraction(1), Fraction(100)) == "SATISFIED"
    assert q3.budget(Fraction(1), Fraction(99)) == "NOT_MET: comparison UNRESOLVED"
    assert q3.budget(Fraction(0), Fraction(0)).startswith("UNRESOLVED")


def test_formatting_is_directed_or_exact():
    x = 0.1
    assert Fraction(q3.fmt("a_up", x)) >= Fraction(x) and Fraction(q3.fmt("a_down", x)) <= Fraction(x)
    assert float(q3.fmt("a", x)) == x
    assert q3.fmt("k", None) == "" and q3.fmt("k", True) == "True" and q3.fmt("k", "UNRESOLVED") == "UNRESOLVED"
    assert q3.fmt("b_up", math.inf) == "inf"


def test_stats_ignore_non_numbers():
    s = q3.stats([1.0, "UNRESOLVED", None, 3.0, math.nan, True])
    assert s["count"] == 2 and s["min"] == 1.0 and s["max"] == 3.0
    assert q3.stats([]) == {"count": 0}


# --- sequences ------------------------------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def seq_n2(tmp_path_factory):
    d = tmp_path_factory.mktemp("arrays")
    return q3.seq_job({**SMALL, "N": 2}, str(d)), d


def test_sequence_job_association_and_exact_identity(seq_n2):
    out, d = seq_n2
    rows = out["seq"]
    assert len(rows) == 3
    assert all(r["association"].startswith("VERIFIED") for r in rows)
    assert all(r["j_vok2_identity_exact"] is True for r in rows)
    assert all(r["marking"].startswith("RESEARCH_ONLY") for r in rows)
    assert all(isinstance(r["E1_m_s_cR22"], float) and r["E1_m_s_cR22"] <= r["E1_m_s_cR0"] for r in rows)
    assert rows[0]["E7_status"].startswith("NOT_APPLICABLE") and isinstance(rows[1]["E7_condA_over_inv_rho_sq"], float)
    assert (d / q3.seq_file({**SMALL, "N": 2})).exists()


def test_e9_split_reproduces_the_merged_model_term(seq_n2):
    """R_base of the journal = up(R(w_prev) + R(w_s) + R_Q(d)) of the recorded arrays (bitwise) - checked inside the job; here the
    recorded arrays reproduce the journal fields directly."""
    out, d = seq_n2
    data = np.load(d / q3.seq_file({**SMALL, "N": 2}))
    g = en.Grid(1.0, 10, 1.3, 9)
    w1 = data["s1_Xs"] @ data["s1_Ys"].T
    assert en.pi_double(g, w1) == out["seq"][0]["j_P_double_s"]
    assert out["seq"][0]["E9_RQ_over_R_cR22"] > 0


def test_e2_rows_have_exact_reference_and_premises(seq_n2):
    out, _ = seq_n2
    for r in out["e2"]:
        assert r["premises"] == "P3_CONFIRMED" and r["vok4_status"] == "HOLDS" and r["vok5_status"] == "HOLDS"
        assert r["e_a_over_eps_a_C_up"] <= 1.0 and r["E2_status"].startswith("exact reference")


def test_zero_load_gives_unresolved_ratios_not_errors(tmp_path):
    """q = 0: the sequence stops (ZERO_FIELD) or yields zero fields; no ratio with a zero denominator is reported as a number."""
    import kkvim_reference.r5_cccc_sequence as sq

    cfg = sq.SequenceConfig(1.0, 8, 1.3, 7, 1, q=0.0, s_max=2)
    res = sq.run_sequence(cfg)
    assert res.status.startswith("STOP")
    assert q3.ratio(0.0, 0.0) == "UNRESOLVED"


def test_sequence_job_is_reproducible(tmp_path):
    a = q3.seq_job({**SMALL, "N": 1}, str(tmp_path / "a"))
    b = q3.seq_job({**SMALL, "N": 1}, str(tmp_path / "b"))
    strip = lambda rows: [{k: v for k, v in r.items() if k != "seconds"} for r in rows]  # noqa: E731
    assert json.dumps(strip(a["seq"]), default=str) == json.dumps(strip(b["seq"]), default=str)
    assert json.dumps(a["e2"], default=str) == json.dumps(b["e2"], default=str)


# --- E-8, E-3, E-4/E-6, E-5 ----------------------------------------------------------------------------------------------------------

def test_e8_reference_bound_is_certified_and_budget_checked(seq_n2):
    out, d = seq_n2
    spec = {"lx": 1.0, "nx": 10, "ly": 1.3, "ny": 9, "sequences": [{**SMALL, "N": 2}],
            "journal": {q3.seq_file({**SMALL, "N": 2}): out["seq"]}}
    rows = q3.e8_job(spec, str(d))["e8"]
    assert len(rows) == 3
    for r in rows:
        assert r["budget_1_100"] == "SATISFIED"
        assert r["dist_wexact_to_omega_h_a_down"] <= r["dist_wexact_to_omega_h_a_up"]
        assert r["B_omega_bound_consistent"] is True  # proven upper bound against the certified lower bound
        assert r["ref_bound_a_up"] < 1e-12


def test_e3_separation_logic():
    G = hs.start_set(10, 2)
    U = np.linalg.qr(math.sqrt(0.1) * G)[0] / math.sqrt(0.1)
    row = q3.e3_pair(1.0, 10, 1.3, 9, "Y", G, U, "sin^2 vs QR")
    for solver in ("R-AC", "B-C"):
        assert row[f"{solver}_transformation_rel_down"] <= row[f"{solver}_transformation_rel_up"]
        assert row[f"{solver}_separation"] in ("SATISFIED", "NOT_MET: comparison UNRESOLVED") or \
            row[f"{solver}_separation"].startswith("UNRESOLVED")
    same = q3.e3_pair(1.0, 10, 1.3, 9, "Y", U, U, "identical sets")
    assert same["R-AC_Delta_a_rel_up"] == 0.0 and same["R-AC_separation"].startswith("UNRESOLVED")


def test_halfstep_job_rows():
    out = q3.hs_job({"group": "test", "lx": 1.0, "nx": 8, "ly": 1.3, "ny": 7, "qr_modes": (True, False)})
    assert len(out["e4"]) == len(out["e6"]) == 2 * (1 + 2 + 2)
    for r in out["e4"]:
        assert r["reference"].startswith("EXACT") and r["vok14_consistent"] is True
        if r["N"] >= 2:
            assert r["marking"].startswith("PROVISIONAL")
    for r in out["e6"]:
        assert r["vok10_check"].startswith("HOLDS") and r["vok11_check"].startswith("HOLDS")


def test_e5_excludes_subnormal_inputs_and_results():
    assert q3.subnormal(np.array([0.0, 2.0**-1074])) and not q3.subnormal(np.array([0.0, 1.0]))
    out = q3.e5_job({"n": 16, "sample": "test"})
    assert len(out["e5"]) == 7 and all(r["in_c_dst_statistic"] for r in out["e5"])


def test_seqstep_job_uses_the_controller_qr(tmp_path):
    out = q3.seq_job({**SMALL, "N": 2}, str(tmp_path))
    st = q3.seqstep_job({**SMALL, "N": 2}, str(tmp_path))
    assert [r["s"] for r in st["e4"]] == [2, 3] and len(st["e3"]) == 2
    data = np.load(tmp_path / q3.seq_file({**SMALL, "N": 2}))
    # the half-step QR of the solved set reproduces the controller's transformed set bitwise
    import kkvim_reference.r5_cccc_bc as bc

    p = hs.HalfStep(1.0, 10, 1.3, 9, "X", data["s1_Ys"])
    U, _ = bc.orthonormalise(p)
    assert np.array_equal(U, data["s1_Yn"])
    assert out["seq"][1]["j_direction"] == "X"


# --- outputs, metadata, independence -------------------------------------------------------------------------------------------------

def test_ci_campaign_outputs_and_metadata(tmp_path):
    """The command line (spawned worker processes) in the ci profile: all outputs, complete metadata, the resource record."""
    import subprocess
    import sys

    done = subprocess.run([sys.executable, str(SCRIPT), "run", "--profile", "ci", "--workers", "2", "--out", str(tmp_path / "out"),
                           "--arrays", str(tmp_path / "arrays")], cwd=ROOT, capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stderr
    meta = json.loads((tmp_path / "out" / "meta.json").read_text(encoding="utf-8"))
    for key in ("git_commit", "python", "platform", "numpy", "scipy", "blas", "fft", "threads", "seed", "ram_mb",
                "resource_rule_PL_6_3", "sample", "counts", "printing", "c_R_path_count"):
        assert key in meta
    assert meta["threads"]["OPENBLAS_NUM_THREADS"] == "1" and meta["resource_rule_PL_6_3"]["within_limit"] is True
    for name in ("sequences_e1_e7_e9", "e2", "e3", "e4", "e5", "e6", "e8", "resources"):
        rows = list(csv.DictReader(open(tmp_path / "out" / f"{name}.csv", encoding="utf-8")))
        assert rows, name
    summary = json.loads((tmp_path / "out" / "summary.json").read_text(encoding="utf-8"))
    assert summary["E1_vok2_identity_exact_all"] is True and "hypotheses" in summary
    assert meta["counts"]["seq"] == 6 and meta["profile"] == "ci"


def test_script_does_not_import_kkvim():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            names.add((node.module or "").split(".")[0])
    assert "kkvim" not in names
