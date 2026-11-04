"""``run-ssss`` for N = 4-6 against the existing research path and the independent references (D-056.7, D-060).

- the research route equals ``scripts/r4_n4_n6_study.py::run_cell`` bit for bit in one process, and on N = 2-3 ``run_coupled``;
- the command reproduces the committed study results (docs/results/r4_n4_n6) with the existing criterion of
  tests/cross/test_r4_n4_n6_study.py: statuses and s exactly, w(centre) and Pi_h to 1e-10 relative (AGENTS.md, section 4);
- physical input: the field of the command agrees with the independent discrete control at the same physical a, b, q, D, with
  the existing smoke bound 1e-10 of that test, and on N = 2-3 the scaling of the study cell agrees with run_coupled at the same
  physical input within the bound 1e-11 of А4-12 (D-048.7; N = 2, 3, r = 1/5, 1, 5, n <= 160, s = 6).
The bounds are existing ones; these tests choose no tolerance and qualify nothing.
"""

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

import kkvim.coupled_run as cr
from kkvim import cli, research_ssss
from kkvim.coefficients import ERROR_INPUT
from kkvim.physical import bending_stiffness
from kkvim_reference.r4_spectral_control import SpectralGrid, run_control

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "docs" / "results" / "r4_n4_n6"
_spec = importlib.util.spec_from_file_location("r4_n4_n6_study", ROOT / "scripts" / "r4_n4_n6_study.py")
study = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(study)

PHYSICAL = dict(t=0.01, E=2.1e11, nu=0.3)
D = bending_stiffness(PHYSICAL["E"], PHYSICAL["t"], PHYSICAL["nu"])


def run_command(out, N, a, b, q, n_x, n_y, half_steps):
    args = ["run-ssss", "--N", str(N), "--a", repr(a), "--b", repr(b), "--q", repr(q), "--nx", str(n_x), "--ny", str(n_y),
            "--half-steps", str(half_steps), "--out", str(out)]
    for name, value in PHYSICAL.items():
        args += [f"--{name}", repr(value)]
    if N >= 4:
        args.append("--research-only")
    return cli.main(args)


def read_rows(path):
    return list(csv.DictReader(line for line in path.read_text(encoding="utf-8").splitlines() if not line.startswith("#")))


def read_field(path, n_x, n_y, column):
    values = np.empty((n_x + 1, n_y + 1))
    for row in read_rows(path):
        values[int(row["i"]), int(row["j"])] = float(row[column])
    return values


def assert_equal_logs(mine, theirs):
    assert len(mine) == len(theirs)
    for a, b in zip(mine, theirs):
        for name in ("s", "k", "direction", "eta_N", "eta_initial", "refined", "cond_A_given", "cond_M_max", "w_max", "eps_inf",
                     "eps_E", "qr_done", "qr_field_error", "P_after_qr", "stop_reason", "m_s", "stabilization_ok", "mark"):
            assert getattr(a, name) == getattr(b, name), name
        assert np.array_equal(a.rank.sigma, b.rank.sigma) and np.array_equal(a.rank.set_sigma, b.rank.set_sigma)
        assert a.rank.rho_N == b.rank.rho_N and a.rank.kappa_rep == b.rank.kappa_rep
        assert a.guards == b.guards
        for name in ("P", "delta_P", "Q", "gamma", "tau", "status", "R", "R_f"):
            assert getattr(a.energy, name) == getattr(b.energy, name), name


@pytest.mark.parametrize("N, lab, nx, ny, s", [(4, "1", 40, 40, 40), (5, "1", 40, 40, 40), (6, "1", 40, 40, 40), (4, "1/5", 40, 40, 12),
                                               (5, "5", 40, 40, 12), (6, "5", 40, 40, 12), (6, "1/5", 40, 40, 8), (6, "1/3", 80, 80, 8),
                                               (6, "1", 24, 24, 4), (4, "2", 320, 640, 3)])
def test_the_route_equals_the_cell_of_the_study_bitwise(N, lab, nx, ny, s):
    cell = study.run_cell(N, lab, "I", nx, ny, half_steps=s, keep=())
    run = research_ssss.run_research(float(study.R_TEN[lab]), nx, ny, N, half_steps=s)
    assert run.status == cell.status and run.reason == cell.reason and run.s == cell.s_reached
    assert run.first_stabilization_s == cell.first_stabilization_s
    assert np.array_equal(run.x, cell.x) and np.array_equal(run.y, cell.y)
    assert_equal_logs(run.steps, cell.steps)
    assert run.parameters.N == N and run.parameters.eps_floor == cell.eps_floor


@pytest.mark.parametrize("N, r, nx, ny", [(2, 1.0, 24, 24), (3, 1.0, 24, 24), (2, 0.2, 40, 20), (3, 5.0, 16, 48), (3, 1.5, 30, 36)])
def test_the_orchestration_equals_run_coupled_on_n2_n3_bitwise(N, r, nx, ny):
    """The loop is the working path, as the study checked for its own: statuses, sets and logs of run_coupled."""
    run = research_ssss._orchestrate(r, nx, ny, N, 7)
    direct = cr.run_coupled(1.0, r, 1.0, 1.0, nx, ny, N, transform="fft", half_steps=7, research_mode=True,
                            stop_on_stabilization=False)
    assert run.status == direct.status and run.s == direct.s == 7
    assert np.array_equal(run.x, direct.x) and np.array_equal(run.y, direct.y)
    assert_equal_logs(run.steps, direct.steps)  # without the s = 2 flag of D-048.1: N = 2-3 only (report of the study, Н-5)
    assert run.first_stabilization_s == direct.first_stabilization_s


def test_the_route_refuses_the_production_range():
    for N in (1, 2, 3, 7, True):
        assert research_ssss.run_research(1.0, 40, 40, N, half_steps=1).status == ERROR_INPUT
    assert research_ssss.run_research(1.0, 40, 40, 4, half_steps=1).status == cr.RESEARCH_LIMIT_REACHED


@pytest.mark.parametrize("N", [4, 5, 6])
@pytest.mark.parametrize("lab, b", [("1/5", 0.2), ("1", 1.0), ("5", 5.0)])
def test_the_command_reproduces_the_committed_study_values(tmp_path, N, lab, b):
    """n = 40, s = 40, a = 1, q = D (so q a^4 / D = 1): statuses, reasons and s exactly; the cell values to 1e-10."""
    out = tmp_path / "out"
    code = run_command(out, N, 1.0, b, D, 40, 40, 40)
    key = (str(N), lab, "I", "40")
    ref = next(c for c in study.read_csv(RESULTS / "cells.csv") if (c["N"], c["r_label"], c["family"], c["nx"]) == key)
    result = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert result["status"] == ref["status"] and result["half_steps_completed"] == int(ref["s_reached"])
    assert result["first_stabilization_s"] == (int(ref["first_stabilization_s"]) if ref["first_stabilization_s"] else None)
    assert code == (cli.EXIT_CONVERGED if ref["status"] == cr.RESEARCH_LIMIT_REACHED else 6)
    steps = [t for t in study.read_csv(RESULTS / "steps.csv") if (t["N"], t["r_label"], t["family"], t["nx"]) == key]
    rows = read_rows(out / "history.csv")
    assert len(rows) == len(steps)
    for mine, committed in zip(rows, steps):
        assert mine["stop_reason"] == committed["stop_reason"] and mine["direction"] == committed["direction"]
        assert abs(float(mine["P"]) - float(committed["Pi_h"])) <= 1e-10 * abs(float(committed["Pi_h"]))
        assert abs(float(mine["w_bar_max"]) - float(committed["w_max"])) <= 1e-10 * abs(float(committed["w_max"]))
    name = "field_research.csv" if ref["status"] == cr.RESEARCH_LIMIT_REACHED else "field_diagnostic.csv"
    centre = float(steps[-1]["w_centre"])
    assert abs(read_field(out / name, 40, 40, "w_bar [1]")[20, 20] - centre) <= 1e-10 * abs(centre)


@pytest.mark.parametrize("N, a, b, q, nx, ny", [(4, 2.0, 2.0, 1000.0, 40, 40), (5, 0.5, 1.5, -300.0, 40, 40), (6, 3.0, 1.5, 2500.0, 80, 80),
                                                (5, 0.4, 2.0, 7.0, 320, 1600)])
def test_physical_fields_agree_with_the_independent_control(tmp_path, N, a, b, q, nx, ny):
    """The control solves the same discrete problem at the physical a, b, q, D by another route (no call of the solver)."""
    half_steps = 4
    out = tmp_path / "out"
    assert run_command(out, N, a, b, q, nx, ny, half_steps) == cli.EXIT_CONVERGED
    w = read_field(out / "field_research.csv", nx, ny, "w [m]")
    control = run_control(SpectralGrid(a, b, nx, ny, q, D), N, half_steps)[half_steps - 1].W
    assert float(np.max(np.abs(w[1:-1, 1:-1] - control))) <= 1e-10 * float(np.max(np.abs(control)))
    rows = read_rows(out / "field_research.csv")
    assert max(float(row["x [m]"]) for row in rows) == pytest.approx(a, rel=1e-15)
    assert max(float(row["y [m]"]) for row in rows) == pytest.approx(b, rel=1e-15)


@pytest.mark.parametrize("N", [2, 3])
@pytest.mark.parametrize("r", [0.2, 1.0, 5.0])
def test_scaling_of_the_cell_agrees_with_the_physical_run_on_n2_n3(N, r):
    """s = 6, n = 160 (the area of А4-12, D-048.7), the only N where both ways exist: g w_bar against the physical run."""
    a, q = 2.9, 3700.0
    direct = cr.run_coupled(a, r * a, D, q, 160, 160, N, transform="fft", half_steps=6, research_mode=True,
                            stop_on_stabilization=False)
    cell = research_ssss._orchestrate(r, 160, 160, N, 6)
    assert direct.status == cell.status == cr.RESEARCH_LIMIT_REACHED
    scaled = cell.field() * (q * a**4 / D)
    assert float(np.max(np.abs(scaled - direct.field()))) <= 1e-11 * float(np.max(np.abs(direct.field())))
    assert all(step.guards.passed for step in direct.steps) and all(step.guards.passed for step in cell.steps)
