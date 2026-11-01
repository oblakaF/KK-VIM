"""QA checks of the R4 N = 4-6 study (scripts/r4_n4_n6_study_qa.py; docs/R4_N4_N6_STUDY.qa-report.md).

Fast checks (CI): the integrity and isolation checks are recomputed live from the committed Builder results and compared
with the committed QA results; the committed QA tables follow the criteria K-1..K-7 of the QA script. The checks that need
python-flint and mpmath (a temporary QA environment, not project dependencies) are skipped when they are absent: certified
eigenvalue brackets with a negative control (an exactly rank-deficient field is never certified), the exact first
half-step against the committed row, the exact 2D Galerkin sequence against the solver with an injected error and a wrong
stencil as negative controls. One cell is rerun and compared with the committed rows as text only with
KKVIM_QA_HEAVY=1.
"""

import csv
import importlib.util
import json
import math
import os
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "r4_n4_n6_study_qa.py"
QA = ROOT / "docs" / "results" / "r4_n4_n6_qa"


def _load():
    spec = importlib.util.spec_from_file_location("r4_n4_n6_study_qa", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


Q = _load()


def _rows(name):
    with open(QA / name, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# deviations of the Builder report found by K-1 (documented in the QA report, section 4)
EXPECTED_FALSE = {"energy_increases_all_after_stabilization_as_stated", "config_margin_1_42_at_n1280_as_stated",
                  "report_numbers"}
EXPECTED_CLAIMS_FAILED = {"grid of the minimal configuration margin", "family II E_disc max (5.4: 7.6e-6)",
                          "5.3 e_rel(cont) r = 1/5, s = 40, N = 6"}


# ------------------------------------------------------------------------------------------------ live recomputation
def test_integrity_is_reproduced_from_the_committed_builder_results(tmp_path):
    live = Q.integrity(tmp_path)
    committed = json.loads((QA / "integrity.json").read_text(encoding="utf-8"))
    assert live["checks"] == committed["checks"]
    assert {k for k, v in live["checks"].items() if v is not True} == EXPECTED_FALSE
    assert {c["claim"] for c in live["claims_failed"]} == EXPECTED_CLAIMS_FAILED
    assert live["n_claims"] == committed["n_claims"] == 100


def test_research_halfstep_differs_from_the_production_halfstep_only_by_the_n_guard(tmp_path):
    out = Q.isolation(tmp_path)
    assert out["difference_is_the_N_guard"] and not out["attribute_assignments_to_kkvim"] and not out["patch_calls"]


# ------------------------------------------------------------------------------------------------ committed QA results
def test_first_halfstep_rank_is_certified_and_the_guard_decisions_are_those_of_the_exact_field():
    rows = [r for r in _rows("rho_s1.csv") if r["family"] in ("I", "II")]
    assert len(rows) == 210
    for r in rows:
        assert float(r["rho_star_lo"]) > 0  # rank W*^1 = N (theorem ЧВ4-9), certified
        assert r["guard_rank_exact"] == r["guard_rank_solver"] and r["guard_set_exact"] == r["guard_set_solver"]
    stops = [r for r in rows if r["guard_rank_solver"] == "False"]
    assert len(stops) == 14 and {(r["N"], r["r_label"]) for r in stops} == {("6", "1/5"), ("6", "1/3")}
    assert all(float(r["rho_star_over_threshold"]) < 1 for r in stops)


def test_exact_sequence_on_small_grids_confirms_the_implementation():
    rows = _rows("nodal.csv")
    assert len({(r["N"], r["r_label"], r["nx"], r["ny"]) for r in rows}) == len(Q.NODAL_CELLS)
    impl = [float(r["E_impl_true_up"]) for r in rows if r["E_impl_true_up"]]
    assert impl and max(impl) <= 1.5e-12  # K-4: CONFIRMED level of the Builder V5 statement
    pis = [float(r["Pi_rel_err_up"]) for r in rows if r["Pi_rel_err_up"]]
    assert max(pis) <= 4.9e-15
    for r in rows:
        assert float(r["rel_rad_W"]) < 2.0**-100
        assert float(r["rho_star_lo"]) > 0


def test_first_halfstep_residual_and_solve_error_on_small_grids():
    rows = _rows("residual_s1.csv")
    assert len(rows) == len(Q.NODAL_CELLS)
    for r in rows:
        assert float(r["solve_error_rel_up"]) <= 1.5e-12  # K-4 level; the residual itself is only an observation
        assert float(r["galerkin_residual_of_rounded_exact_rel_up"]) > 0


def test_continuous_recursion_in_arb():
    rows = _rows("continuous_arb.csv")
    assert len({(r["N"], r["r_label"]) for r in rows}) == len(Q.CONT_PAIRS)
    for r in rows:
        assert float(r["double_minus_arb_max_rel_up"]) <= float(r["double_rounding_model_rel"])  # K-5 (а)
        lo_, hi_ = float(r["e_rel_trunc_lo"]), float(r["e_rel_trunc_hi"])
        b, budget = float(r["e_rel_builder"]), float(r["e_rel_builder_budget_rel"])
        printed = 0.5 * 10.0 ** (math.floor(math.log10(b)) - 6)  # continuous.csv keeps 7 significant digits
        slack = budget * b * (1 + 1e-6) + printed
        assert lo_ - slack <= b <= hi_ + slack  # K-5 (б), with the print precision of the committed value (QA report, 6)
        if r["d_method_c_qa"]:
            for k in ("d_method_c", "d_disc_c"):
                qa_, bu = float(r[f"{k}_qa"]), float(r[f"{k}_builder"])
                assert abs(qa_ - bu) <= 1e-5 * abs(bu) + 1e-14  # K-5 (г)


def test_reruns_reproduce_the_committed_rows():
    rows = _rows("repro.csv")
    assert len(rows) == len(Q.CHAINS) * len(Q.CHAIN_N) + len(Q.REPRO_EXTRA) + len(Q.CONTROL)
    for r in rows:
        assert int(r["cells_text_diffs"]) == int(r["steps_text_diffs"]) == int(r["compare_text_diffs"]) == 0
        if r.get("run_coupled_deep_diffs"):
            # the only difference to run_coupled: the s = 2 mandatory flag is not written into the step log
            assert r["run_coupled_deep_paths"] == ".energy_s2_mandatory_unresolved"


def test_orders():
    for r in _rows("orders_qa.csv"):
        assert 1.9 <= float(r["p_self"]) <= 2.1
    for r in _rows("disc_s1_exact.csv"):
        lo_, hi_, b = float(r["E_disc_s1_rel_lo"]), float(r["E_disc_s1_rel_hi"]), float(r["E_disc_s1_builder"])
        assert lo_ * (1 - 1e-6) <= b <= hi_ * (1 + 1e-6)


# ------------------------------------------------------------------------------------------------ arb (python-flint)
def _arb():
    pytest.importorskip("flint")
    pytest.importorskip("mpmath")
    f = Q._flint()
    f.ctx.prec = 384
    return f


def test_eigen_brackets_and_an_exactly_rank_deficient_first_halfstep():
    f = _arb()
    G = f.arb_mat([[4, 1, 0], [1, 3, 1e-20], [0, 1e-20, 1e-30]])
    br = Q.eig_brackets(G)
    ev = np.linalg.eigvalsh(np.array([[4, 1, 0], [1, 3, 0], [0, 0, 0]], dtype=float))
    assert br is not None and abs(float(br[0][0].mid()) - ev[-1]) <= 1e-14 and abs(float(br[0][1].mid()) - ev[-1]) <= 1e-14
    assert 0.99e-30 < float(br[2][0].lower()) <= float(br[2][1].upper()) < 1.01e-30  # lambda_3 = 1e-30 - O(1e-40)
    # n_y = 2N - 1: rank W^1 = N - 1 exactly (ЧВ4-9); the N x N Gram matrix of the (N - 1)-column V^ is singular
    V, ks = Q.first_halfstep_hat(3, Fraction(1), 12, 5)
    assert len(ks) == 2
    rho_lo, rho_hi, _ = Q.sv_ratio(V * V.transpose())
    assert rho_lo == 0.0 and rho_hi < 1e-30  # never certified positive; the bracket width is the working precision


def test_first_halfstep_closed_form_matches_the_committed_row():
    _arb()
    row = Q.rho1_cell((6, "1/5", "I", 40, 40))
    ref = [r for r in _rows("rho_s1.csv") if (r["N"], r["r_label"], r["family"], r["nx"]) == ("6", "1/5", "I", "40")][0]
    assert row["rho_star_lo"] == float(ref["rho_star_lo"]) and row["rho_star_hi"] == float(ref["rho_star_hi"])


def test_exact_sequence_detects_an_injected_error_and_a_wrong_stencil():
    f = _arb()
    study = Q.builder_study()
    cell = study.run_cell(4, "5", "II", 12, 60, half_steps=3, keep=(1, 2, 3))
    seq, P = Q.nodal_sequence(4, "5", 12, 60, 3, 384)
    W, Ws = seq[2]["W"], cell.fields[3][1:-1, 1:-1]
    idx = [(p, q) for p in range(P.nxi) for q in range(P.nyi)]
    wmax = Q._max_lo(abs(W[p, q]) for p, q in idx)
    base = Q._max_up(abs(f.arb(float(Ws[p, q])) - W[p, q]) for p, q in idx) / wmax
    pert = Ws + 1e-10 * float(np.max(np.abs(Ws))) * np.outer(np.sin(np.pi * np.arange(1, 12) / 12),
                                                            np.sin(np.pi * np.arange(1, 60) / 60))
    meas = Q._max_up(abs(f.arb(float(pert[p, q])) - W[p, q]) for p, q in idx) / wmax
    assert base < 1e-14 and 0.5e-10 < meas < 2e-10
    # negative control: a wrong y spacing in the five-point stencil (on a grid with h_x != h_y) is visible at once
    good_P, bad_P = Q.NodalProblem(Fraction(1), 12, 24), Q.NodalProblem(Fraction(1), 12, 24)
    bad_P.ihy2 = bad_P.ihx2
    good = good_P.halfstep(good_P.odd_sines(4), "Y")
    Wbad = bad_P.halfstep(bad_P.odd_sines(4), "Y")
    assert max(abs(float((Wbad[k, j] - good[k, j]).mid())) for k in range(good.nrows()) for j in range(4)) > \
        1e-3 * max(abs(float(good[k, j].mid())) for k in range(good.nrows()) for j in range(4))


@pytest.mark.skipif(os.environ.get("KKVIM_QA_HEAVY") != "1", reason="set KKVIM_QA_HEAVY=1")
def test_one_cell_is_reproduced_as_text(tmp_path):
    item = Q.repro_cell((5, "5", "I", 40, 40))
    study = Q.builder_study()
    fresh = Q._formatted(study, item["steps"], tmp_path / "steps.csv")
    with open(ROOT / "docs" / "results" / "r4_n4_n6" / "steps.csv", newline="", encoding="utf-8") as f:
        old = [r for r in csv.DictReader(f) if (r["N"], r["r_label"], r["family"], r["nx"]) == ("5", "5", "I", "40")]
    assert len(fresh) == len(old) == 40
    for a, b in zip(fresh, old):
        assert all(a[k] == b[k] for k in a)
