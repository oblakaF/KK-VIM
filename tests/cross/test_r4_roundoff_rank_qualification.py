"""QA checks of the R4 round-off / rank qualification (docs/R4_ROUNDOFF_RANK_QUALIFICATION.plan.md, .qa-report.md).

Fast checks (CI): the recording of the solver arrays is transparent; the rigorous nodal enclosure (TwoProduct/TwoSum) is exact
against rational arithmetic; the committed result tables are complete and follow the rules of the plan, sec. 7. The checks
that need python-flint (a temporary QA tool, not a project dependency) are skipped when it is absent: the nodal path against
the spectral path, the injection control, rank-deficient fields never certified. A small part of the campaign is recomputed
and compared with the committed rows only with KKVIM_ROUNDOFF_HEAVY=1.
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
SCRIPT = ROOT / "scripts" / "r4_roundoff_rank_qualification.py"
RESULTS = ROOT / "docs" / "results" / "r4_roundoff_rank_qualification"


def _load():
    spec = importlib.util.spec_from_file_location("r4_roundoff_rank_qualification", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


Q = _load()


def _rows(name):
    with open(RESULTS / name, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _cell(N, r, nx, ny=None):
    return {"family": "I", "N": N, "r": r, "nx": nx, "ny": ny or nx}


# ------------------------------------------------------------------------------------------------ recording and enclosures
@pytest.mark.parametrize("N,r", [(2, "1"), (3, "1/5")])
def test_recording_does_not_change_the_run(N, r):
    cell = _cell(N, r, 16)
    rec_run, recs = Q.run_solver(cell, "fft", "M2", record=True)
    run, _ = Q.run_solver(cell, "fft", "M2")
    assert Q.log_signature(rec_run.steps) == Q.log_signature(run.steps)
    assert Q.check_records(rec_run, recs)
    assert np.array_equal(rec_run.x, run.x) and np.array_equal(rec_run.y, run.y)


def test_stopping_and_normative_runs_are_prefixes_of_the_fixed_run():
    cell = _cell(3, "5", 16)
    run2, _ = Q.run_solver(cell, "fft", "M2")
    run1, _ = Q.run_solver(cell, "fft", "M1")
    run0, _ = Q.run_solver(cell, "fft", "M0")
    assert all(Q.num_signature(a) == Q.num_signature(b) for a, b in zip(run1.steps, run2.steps))
    assert Q.num_signature(run0.steps[0]) == Q.num_signature(run2.steps[0])


def test_nodal_enclosure_contains_the_exact_rational_value():
    rng = np.random.default_rng(11)
    X = rng.standard_normal((7, 3)) * np.array([1.0, 1e-5, 1e3])
    Y = rng.standard_normal((5, 3))
    w = X @ Y.T
    lo_, hi_ = Q.enclose_sum(Q.exact_terms(X, Y) + [-w])
    for i in range(7):
        for j in range(5):
            exact = sum(Fraction(float(X[i, k])) * Fraction(float(Y[j, k])) for k in range(3)) - Fraction(float(w[i, j]))
            assert Fraction(float(lo_[i, j])) <= exact <= Fraction(float(hi_[i, j]))
    assert np.max(hi_ - lo_) <= 1e-28 * np.max(np.abs(X) @ np.abs(Y).T)


def test_formation_bound_holds_for_a_solver_field():
    from kkvim.grid import Grid1D

    cell = _cell(3, "1", 40)
    _, recs = Q.run_solver(cell, "fft", "M2", record=True)
    rec = recs[4]
    X, Y = (rec["solved"], rec["given"]) if rec["step"].direction == "X" else (rec["given"], rec["solved"])
    out = Q.formation_check(X, Y, Grid1D(1.0, 40), Grid1D(1.0, 40))
    assert out["form_status"] == "PASS" and out["form_ratio_max"] < 1


def test_outward_rounding_of_the_tables():
    x = 1.23456789123e-15
    assert float(Q.fmt_value("delta_f", x)) >= x
    assert float(Q.fmt_value("rho_star_lo", x)) <= x
    assert float(Q.fmt_value("max_delta_f", -x)) >= -x


# ------------------------------------------------------------------------------------------------ ball arithmetic (python-flint)
def test_nodal_path_agrees_with_the_spectral_path_on_a_small_grid():
    pytest.importorskip("flint")
    f = Q.flint()
    cell = {"family": "V", "N": 3, "r": "1/5", "nx": 12, "ny": 8}
    nseq, _, prec = Q.nodal_sequence_adaptive(cell, 6)
    f.ctx.prec = prec
    P = Q.Problem(cell)
    seq, _ = Q.exact_sequence_at(P, 6)
    for s in range(6):
        ov, wid = Q._overlap_width(Q._nodal_from_coeff(P, *seq[s]), nseq[s][0] * nseq[s][1].transpose())
        assert ov and wid < 1e-30


def test_wrong_sign_of_B_is_detected():
    pytest.importorskip("flint")
    f = Q.flint()
    cell = {"family": "V", "N": 2, "r": "5", "nx": 12, "ny": 12}
    f.ctx.prec = 512
    P = Q.Problem(cell)
    seq, _ = Q.exact_sequence_at(P, 2)
    try:
        bad, _ = Q.nodal_sequence(cell, 2, sign_B=-1)
    except ZeroDivisionError:
        return
    assert not Q._overlap_width(Q._nodal_from_coeff(P, *seq[1]), bad[1][0] * bad[1][1].transpose())[0]


def test_injected_field_error_is_measured():
    pytest.importorskip("flint")
    f = Q.flint()
    cell = _cell(2, "1", 16)
    _, recs = Q.run_solver(cell, "fft", "M2", record=True)
    f.ctx.prec = 256
    P = Q.Problem(cell)
    seq, _ = Q.exact_sequence_at(P, 3)
    rec = recs[2]  # s = 3, Y-half-step
    Gc = P.ax.transform(rec["given"])
    Xe, Ye = P.ax.embed(seq[2][0]), P.ay.embed(seq[2][1])
    s1 = Q.lo(Q.sigmas(Xe, Ye)[0])
    base = Q.up(Q.frob_diff(Gc, P.ay.transform(rec["solved"]), Xe, Ye)) / s1
    pert = rec["solved"].copy()
    pert[:, 1] += 1e-11 * np.max(np.abs(pert)) * np.sin(3 * np.pi * np.arange(1, 16) / 16)
    meas = Q.up(Q.frob_diff(Gc, P.ay.transform(pert), Xe, Ye)) / s1
    assert base < 1e-14 and 1e-12 < meas < 1e-10


def test_rank_deficient_field_is_never_certified():
    pytest.importorskip("flint")
    from kkvim.coupled_rank import rank_diagnostics
    from kkvim.grid import Grid1D
    from kkvim_reference.r4_rank_diagnostics import cancellation_set

    f = Q.flint()
    f.ctx.prec = 512
    P = Q.Problem(_cell(3, "1", 24))
    g = Grid1D(1.0, 24)
    hxy = (P.ax.h * P.ay.h).sqrt()
    for K in (1.0, 1e6):
        U, V = cancellation_set(23, 23, K, seed=2, N=3)
        rk = rank_diagnostics(g, U, g, V)
        sg = Q.sigmas(P.ax.transform(U), P.ay.transform(V))
        assert sg[-1].contains(0)
        eps = max(Q.up(abs(f.arb(float(rk.sigma[k])) - sg[k] * hxy)) for k in range(3)) / Q.lo(sg[0] * hxy)
        assert not (eps < 1 and rk.rho_N > eps / (1 - eps))


# ------------------------------------------------------------------------------------------------ committed results
def test_tool_validation_passed():
    summary = json.loads((RESULTS / "validate_summary.json").read_text(encoding="utf-8"))
    assert all(summary[k] for k in ("V1", "V2", "V3", "V4", "V5"))
    assert summary["V1_cases"] == 720 and summary["V4_cells"] == 12


def test_main_campaign_is_complete_and_consistent():
    cells = _rows("cells_fft_M2.csv")
    assert len(cells) == 132
    assert {c["cell"] for c in cells} == {Q.cell_key(c) for c in Q.all_cells()}
    for c in cells:
        assert c["result"] == "DONE", c["cell"]
        assert c["records_ok"] == "True" and c["M1_prefix"] == "True" and c["M0_prefix"] == "True"
        assert c["all_triangle_ok"] == "True" and c["all_weyl_ok"] == "True"
        assert int(c["n_steps_certified"]) == int(c["s_end"])
        for key in ("assembly_bound_status", "assembly_influence_status", "modal_bound_status", "form_status"):
            assert c[key] == "PASS", (c["cell"], key)


def test_step_rows_follow_the_rank_rule():
    for r in _rows("steps_fft_M2.csv"):
        if r["status"] != "OK":
            continue
        eps, rho = float(r["eps_cert"]), float(r["rho_hat"])
        assert (r["rank_distinguishable"] == "True") == (eps < 1 and rho > eps / (1 - eps))
        assert float(r["delta_f"]) <= (float(r["e_step"]) + float(r["transfer"])) * (1 + 1e-6)


@pytest.mark.skipif(os.environ.get("KKVIM_ROUNDOFF_HEAVY") != "1", reason="set KKVIM_ROUNDOFF_HEAVY=1")
def test_campaign_rows_are_reproduced(tmp_path):
    pytest.importorskip("flint")
    committed = {(r["cell"], r["s"]): r for r in _rows("steps_fft_M2.csv")}
    cells = [_cell(2, "1/5", 40), _cell(3, "5", 40)]
    Q.run_campaign(cells, "fft", 100, tmp_path / "work", 1, "fft_M2", tmp_path)
    with open(tmp_path / "steps_fft_M2.csv", newline="", encoding="utf-8") as f:
        fresh = list(csv.DictReader(f))
    assert fresh
    for r in fresh:
        old = committed[(r["cell"], r["s"])]
        assert {k: v for k, v in r.items() if not k.startswith("t_")} == {k: v for k, v in old.items() if not k.startswith("t_")
                                                                          and k in r}
