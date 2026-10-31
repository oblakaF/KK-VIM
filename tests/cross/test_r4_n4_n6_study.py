"""Minimal checks of the research script scripts/r4_n4_n6_study.py (R4, N = 4-6; Builder).

The study measures the behaviour of the original coupled KK-VIM at N = 4-6; there is deliberately no threshold on the
accuracy of the method (D-025, D-046). The tests check that the research orchestration is the working solver path and
nothing else: it equals run_coupled bitwise for N = 2-3, it leaves the production range N_SOLVE = (2, 3) and its guards
untouched, its half-steps agree with the independent discrete control of the same discrete problem for N = 4-6, and the
s = 1 field equals the closed form of the odd-sine start. Bounds in these tests are smoke bounds of the test (a formula
error gives differences of order 1e-4 and more), not scientific tolerances.
"""

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

import kkvim.coupled as kc
import kkvim.coupled_run as cr
from kkvim.coefficients import ERROR_INPUT
from kkvim.grid import Grid1D
from kkvim_reference.r4_spectral_control import SpectralGrid, first_halfstep_closed_form, run_control

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "r4_n4_n6_study.py"
_spec = importlib.util.spec_from_file_location("r4_n4_n6_study", SCRIPT)
study = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(study)


def test_production_range_and_guards_are_unchanged():
    assert kc.N_SOLVE == (2, 3) and cr.N_SOLVE == (2, 3)
    cell = study.run_cell(4, "1", "I", 24, 24, half_steps=2, keep=())
    assert cell.status == cr.RESEARCH_LIMIT_REACHED and cell.s_reached == 2
    assert kc.N_SOLVE == (2, 3) and cr.N_SOLVE == (2, 3)
    run = cr.run_coupled(1.0, 1.0, 1.0, 1.0, 24, 24, 4, transform="fft", half_steps=2, research_mode=True,
                         stop_on_stabilization=False)
    assert run.status == ERROR_INPUT
    g = Grid1D(1.0, 24)
    with pytest.raises(kc.CoupledHalfStepError):
        kc.solve_coupled_halfstep(g, kc.odd_sine_set(g, 4), g, 1.0, 1.0, direction="Y", transform="fft")


@pytest.mark.parametrize("N, lab, nx, ny", [(2, "1", 24, 24), (3, "1", 24, 24), (2, "1/5", 40, 20), (3, "5", 16, 48),
                                            (3, "3/2", 30, 36)])
def test_orchestration_equals_run_coupled_for_n2_n3(N, lab, nx, ny):
    cell = study.run_cell(N, lab, "I", nx, ny, half_steps=7, keep=())
    assert cell.s_reached == 7
    assert study.equals_run_coupled(cell)


def test_rank_loss_condition_before_the_first_halfstep():
    cell = study.run_cell(6, "1", "I", 11, 24, half_steps=3, keep=())
    assert cell.status == cr.STOPPED_RANK_LOSS_PROVEN and cell.s_reached == 0


@pytest.mark.parametrize("N", [4, 5, 6])
def test_halfsteps_agree_with_the_independent_discrete_control(N):
    for lab, nx, ny in (("1", 24, 24), ("1/3", 30, 20), ("5", 16, 40)):
        r = float(study.R_TEN[lab])
        cell = study.run_cell(N, lab, "I", nx, ny, half_steps=4, keep=(1, 2, 3, 4))
        assert cell.s_reached >= 1 and set(cell.fields) == set(range(1, cell.s_reached + 1))  # a guard may stop early
        ctrl = run_control(SpectralGrid(1.0, r, nx, ny), N, 4)
        for s in cell.fields:
            W = cell.fields[s][1:-1, 1:-1]
            assert float(np.max(np.abs(W - ctrl[s - 1].W))) <= 1e-10 * float(np.max(np.abs(ctrl[s - 1].W)))


@pytest.mark.parametrize("N", [4, 5, 6])
def test_first_halfstep_equals_the_closed_form_of_the_odd_sine_start(N):
    for lab, nx, ny in (("1", 24, 24), ("1/5", 40, 20), ("2", 20, 30)):
        g = SpectralGrid(1.0, float(study.R_TEN[lab]), nx, ny)
        cell = study.run_cell(N, lab, "I", nx, ny, half_steps=1, keep=(1,))
        W1 = first_halfstep_closed_form(g, N)
        assert float(np.max(np.abs(cell.fields[1][1:-1, 1:-1] - W1))) <= 1e-12 * float(np.max(np.abs(W1)))


@pytest.mark.parametrize("N", [4, 6])
def test_energy_identity_and_nonincreasing_energy(N):
    cell = study.run_cell(N, "2/3", "I", 30, 20, half_steps=8, keep=())
    _, rows = study.cell_rows(cell)
    for prev, row in zip(rows, rows[1:]):
        assert row["identity_defect_over_rounding"] <= 1.0  # Delta P = Q - gamma up to the rounding model (ЧВ4-14)
        assert row["Pi_h"] - prev["Pi_h"] <= row["tau"]  # E1 of the test model


def test_quick_run_writes_consistent_outputs(tmp_path):
    assert study.main(["--quick", "--out", str(tmp_path)]) == 0
    meta = json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))
    assert meta["N_SOLVE_before"] == [2, 3] and meta["N_SOLVE_after"] == [2, 3]
    assert meta["control_cells_equal_run_coupled"] is True
    cells = study.read_csv(tmp_path / "cells.csv")
    assert {int(c["N"]) for c in cells} == {2, 3, 4, 5, 6}
    assert all(c["status"] == cr.RESEARCH_LIMIT_REACHED for c in cells)
    comp = study.read_csv(tmp_path / "compare.csv")
    for row in comp:
        # the D-046 rule as written: RESOLVED only if the budget is <= 1/100 of the measured quantity
        if row["status_E_total"] == "RESOLVED":
            assert float(row["levy_budget_rel"]) <= 0.01 * float(row["E_total_inf"])
        if int(row["s"]) >= 2:
            assert row["status_cont_E_disc"] != "RESOLVED" and row["status_cont_E_method"] != "RESOLVED"
        assert math.isfinite(float(row["E_impl_inf"]))
    text = (tmp_path / "tables.md").read_text(encoding="utf-8")
    before = text
    study.make_tables(tmp_path)
    assert (tmp_path / "tables.md").read_text(encoding="utf-8") == before  # tables are a function of the CSV files
