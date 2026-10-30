"""QA: end-to-end checks of the R3 (N = 1) and R4 (N = 2-3) calculation core for a future GUI adapter.

Object: the export of scripts/gui_backend_cases.py, which only calls the kkvim API. Criteria are the accepted ones of
the project, no new tolerance:
- same process, same operations: bitwise (AGENTS.md, section 4);
- R3 at r = 1 against R2: bitwise (STATUS, R3, item H-6);
- D: within 4u of the exact rational value (tests/cross/test_physical_a9.py);
- A-9 (R3): <= 1e-11 relative to max|w_bar| (NUMERICAL_SPEC 18.1, R3_NUMERICAL_SPEC 9.1, D-039);
- A4-12 scaling (R4): <= 1e-11 max|w_bar| in its area N = 2, 3; r = 1/5, 1, 5; n <= 160; s = 6 (D-048.7);
- R4 fields against the QA control: the first-order field model ``_model`` of test_r4_controller_comparison.py;
  dense only up to n = 40 (D-048.6); the stored golden fields across platforms: the doubled model (K-1 of D-048);
- R3 status and stopping half-step: as in the accepted qualification table docs/results/r3/solver/qual/levels.csv.
These are integration regression checks, not independent scientific references; the GUI itself is not tested here.
"""

import csv
import dataclasses
import importlib.util
import json
import math
import os
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

import kkvim.coupled as kc
import kkvim.coupled_run as cr
import kkvim.rectangle_run as rr
from kkvim.energy import ERROR_ENERGY_TEST, EnergyError
from kkvim.linear_solver import ERROR_LINEAR_SOLVE, LinearSolveError
from kkvim.physical import bending_stiffness, run_physical
from kkvim_reference.r4_spectral_control import SpectralGrid, run_control

ROOT = Path(__file__).resolve().parents[2]
U = 2.0**-53


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gb = _load("gui_backend_cases", ROOT / "scripts" / "gui_backend_cases.py")
cmp = _load("r4cmp_for_gui", Path(__file__).with_name("test_r4_controller_comparison.py"))

_CACHE = {}


def result_of(name):
    if name not in _CACHE:
        _CACHE[name] = gb.run_case(gb.CASES[name])
    return _CACHE[name]


WITH_FIELD = [n for n, s in gb.CASES.items() if not n.startswith(("r3_error", "r4_error")) and n != "r4_n3_rank_loss"]
R3_FIELD = [n for n in WITH_FIELD if gb.CASES[n]["model"] == "R3"]
R4_FIELD = [n for n in WITH_FIELD if gb.CASES[n]["model"] == "R4"]


def exact_stiffness(E, t, nu):
    E, t, nu = Fraction(E), Fraction(t), Fraction(nu)
    return float(E * t**3 / (12 * (1 - nu * nu)))


# --- 1. physical input, D, field layout -----------------------------------------------------------------------

@pytest.mark.parametrize("name", WITH_FIELD + ["r4_n3_rank_loss"])
def test_bending_stiffness_is_the_library_value(name):
    p, rec = gb.CASES[name]["inputs"], result_of(name)["record"]
    assert rec["D"] == bending_stiffness(p["E"], p["t"], p["nu"])
    exact = exact_stiffness(p["E"], p["t"], p["nu"])
    assert abs(rec["D"] - exact) <= 4 * U * exact


@pytest.mark.parametrize("name", WITH_FIELD)
def test_field_shape_orientation_boundary_maximum_and_centre(name):
    spec, res = gb.CASES[name], result_of(name)
    p, rec, arr = spec["inputs"], res["record"], res["arrays"]
    w, w_bar, x, y = arr["w"], arr["w_bar"], arr["x"], arr["y"]
    assert w.shape == (p["n_x"] + 1, p["n_y"] + 1) == tuple(rec["field"]["shape"])  # axis 0 = x, axis 1 = y
    assert x.size == p["n_x"] + 1 and y.size == p["n_y"] + 1 and x[0] == y[0] == 0.0
    assert abs(x[-1] - p["a"]) <= 2 * U * p["a"] and abs(y[-1] - p["b"]) <= 2 * U * p["b"]
    assert np.all(w[0, :] == 0) and np.all(w[-1, :] == 0) and np.all(w[:, 0] == 0) and np.all(w[:, -1] == 0)
    assert rec["field"]["boundary_max_abs"] == 0.0
    f = rec["field"]
    i, j = f["argmax_abs"]["i"], f["argmax_abs"]["j"]
    assert f["max_abs_w"] == float(np.max(np.abs(w))) == abs(w[i, j]) and f["w_at_max_abs"] == w[i, j]
    assert (f["argmax_abs"]["x"], f["argmax_abs"]["y"]) == (x[i], y[j])
    assert np.sign(w[i, j]) == np.sign(p["q"])  # deflection in the direction of the load
    ic, jc = p["n_x"] // 2, p["n_y"] // 2
    assert f["centre"]["w"] == w[ic, jc] and f["centre"]["w_bar"] == w_bar[ic, jc]
    if spec["model"] == "R4":  # the API of R4 knows the sets on G(a, n_x) x G(b, n_y)
        run = res["api"]["run"]
        assert run.grid_x.length == p["a"] and run.grid_y.length == p["b"] and run.x.shape[0] == p["n_x"] - 1
    else:
        last = res["api"]["result"].run.history.records[-1]
        assert last.grid_x.length == p["a"] and last.grid_y.length == p["b"] and last.x.size == p["n_x"] - 1


@pytest.mark.parametrize("name", WITH_FIELD)
def test_w_bar_is_w_D_over_q_a4(name):
    p, res = gb.CASES[name]["inputs"], result_of(name)
    D, w = res["record"]["D"], res["arrays"]["w"]
    np.testing.assert_array_equal(res["arrays"]["w_bar"], w * D / (p["q"] * p["a"] ** 4))


# --- 2. export = direct API (bitwise, same process) -----------------------------------------------------------

@pytest.mark.parametrize("name", R3_FIELD)
def test_r3_export_equals_the_direct_api(name, tmp_path):
    p, res = gb.CASES[name]["inputs"], result_of(name)
    direct = rr.run_rectangle_physical(p["a"], p["b"], p["t"], p["E"], p["nu"], p["q"], p["n_x"], p["n_y"])
    report = rr.halfstep_report(direct)
    rec = res["record"]
    assert rec["status"] == direct.status and rec["D"] == direct.D and rec["half_steps_done"] == len(report.halfsteps)
    np.testing.assert_array_equal(res["arrays"]["w"], direct.run.last_field())
    np.testing.assert_array_equal(res["arrays"]["w_bar"], direct.dimensionless_field())
    for row, h in zip(rec["history"], report.halfsteps):
        assert (row["s"], row["direction"], row["eta"], row["P"], row["c"]) == (h.s, h.direction, h.eta, h.P, h.c)
        assert row["eps_E"] == (None if h.changes is None else h.changes.eps_E)
        assert row["energy_status"] == (None if h.energy is None else h.energy.status)
        assert row["w_bar_centre"] == dict(h.control_values)[("1/2", "1/2")]
    _round_trip(name, res, tmp_path)


@pytest.mark.parametrize("name", R4_FIELD)
def test_r4_export_equals_the_direct_api(name, tmp_path):
    spec, res = gb.CASES[name], result_of(name)
    p = spec["inputs"]
    D = bending_stiffness(p["E"], p["t"], p["nu"])
    kwargs = {"transform": spec["transform"], "half_steps": spec["half_steps"]}
    if spec["mode"] != "normative":
        kwargs["research_mode"] = True
    if spec["mode"] == "research_fixed":
        kwargs["stop_on_stabilization"] = False
    direct = cr.run_coupled(p["a"], p["b"], D, p["q"], p["n_x"], p["n_y"], spec["N"], **kwargs)
    rec = res["record"]
    assert rec["status"] == direct.status and rec["reason"] == direct.reason and rec["half_steps_done"] == direct.s
    np.testing.assert_array_equal(res["arrays"]["w"], direct.field())
    for row, st in zip(rec["history"], direct.steps):
        assert (row["s"], row["eta_N"], row["rho_N"], row["P"], row["eps_E"], row["stop_reason"]) == \
            (st.s, st.eta_N, st.rank.rho_N, st.energy.P, st.eps_E, st.stop_reason)
    _round_trip(name, res, tmp_path)


def _round_trip(name, res, tmp_path):
    """The written files give back the same numbers: JSON and CSV by repr, NPZ and NPY exactly."""
    for fmt in ("csv", "npz", "npy"):
        out = tmp_path / fmt
        files = gb.export(name, res, out, fmt)
        rec = json.loads((out / files["json"]).read_text(encoding="utf-8"))
        assert rec["status"] == res["record"]["status"] and rec["D"] == res["record"]["D"]
        assert rec["history"] == res["record"]["history"] and rec["field"] == res["record"]["field"]
        if fmt == "csv":
            for key in ("w", "w_bar"):
                x, y, values = gb.read_field_csv(out / files[f"{key}_csv"])
                np.testing.assert_array_equal(values, res["arrays"][key])
                np.testing.assert_array_equal(x, res["arrays"]["x"])
                np.testing.assert_array_equal(y, res["arrays"]["y"])
            with open(out / files["history_csv"], encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            assert len(rows) == res["record"]["half_steps_done"]
        elif fmt == "npz":
            with np.load(out / files["npz"]) as data:
                for key, value in res["arrays"].items():
                    np.testing.assert_array_equal(data[key], value)
        else:
            for key, value in res["arrays"].items():
                np.testing.assert_array_equal(np.load(out / files[f"{key}_npy"]), value)


# --- 3. R3 against the accepted implementation ----------------------------------------------------------------

def test_r3_square_is_bitwise_the_r2_path():
    p = gb.CASES["r3_square"]["inputs"]
    r2 = run_physical(p["a"], p["t"], p["E"], p["nu"], p["q"], p["n_x"], p["n_y"])
    res = result_of("r3_square")
    assert res["record"]["status"] == r2.status and res["record"]["half_steps_done"] == len(r2.run.history)
    assert res["record"]["D"] == r2.D
    np.testing.assert_array_equal(res["arrays"]["w"], r2.run.last_field())
    np.testing.assert_array_equal(res["arrays"]["w_bar"], r2.dimensionless_field())


LEVELS = ROOT / "docs" / "results" / "r3" / "solver" / "qual" / "levels.csv"
LEVEL_KEYS = [("1", "I", 20, 20), ("1/5", "II", 160, 32), ("7/5", "II", 20, 28), ("5", "I", 40, 40), ("3", "II", 24, 72), ("5", "II", 32, 160)]


@pytest.mark.parametrize("key", LEVEL_KEYS, ids=str)
def test_r3_status_and_stopping_half_step_as_in_the_accepted_qualification(key):
    with open(LEVELS, encoding="utf-8") as fh:
        rows = {(r["r"], r["family"], int(r["nx"]), int(r["ny"])): r for r in csv.DictReader(fh)}
    if key not in rows:
        pytest.fail(f"{key} is not in the accepted qualification table {LEVELS}")
    stored = rows[key]
    lab, _, nx, ny = key
    spec = gb.case("R3", {"a": 1.0, "b": float(Fraction(lab)), **gb.UNIT, "n_x": nx, "n_y": ny}, "levels.csv")
    rec = gb.run_case(spec)["record"]
    assert rec["D"] == 1.0  # the dimensionless plate of the table: q = D = 1
    assert rec["status"] == stored["status"] and rec["half_steps_done"] == int(stored["s_stop"])


# --- 4. units and normalizations (accepted A-9 and A4-12) ------------------------------------------------------

_NU2 = 0.1
A9_VARIANTS = {
    "q x 3.7": {"q": 1000.0 * 3.7},
    "E x 0.013": {"E": 2.1e11 * 0.013},
    "t x 1.9": {"t": 0.01 * 1.9},
    "nu 0.3 -> 0.1 at the same D": {"nu": _NU2, "E": 2.1e11 * (1 - _NU2**2) / (1 - 0.3**2)},
    "a, b x 2.9": "scale",
}


@pytest.mark.parametrize("grid", [(1.0, (320, 320)), (2.0, (160, 320))], ids=["I r=1", "II r=2"])
def test_r3_a9_through_the_export(grid):
    r, (nx, ny) = grid
    base = gb.plate(1.0, r, nx, ny)
    ref = gb.run_case(gb.case("R3", base, "A-9 base"))["arrays"]["w_bar"]
    for name, change in A9_VARIANTS.items():
        inputs = {**base, "a": 2.9, "b": 2.9 * r} if change == "scale" else {**base, **change}
        w_bar = gb.run_case(gb.case("R3", inputs, name))["arrays"]["w_bar"]
        assert np.max(np.abs(w_bar - ref)) <= 1e-11 * np.max(np.abs(ref)), name


@pytest.mark.parametrize("N, r", [(2, 0.2), (2, 1.0), (3, 1.0), (3, 5.0)])
def test_r4_a4_12_scaling_through_the_export(N, r):
    nx, ny = {0.2: (160, 32), 1.0: (40, 40), 5.0: (32, 160)}[r]  # h_x = h_y, n <= 160 (area of D-048.7)
    base = gb.case("R4", gb.plate(1.0, r, nx, ny), "A4-12 base", N=N, mode="research_fixed", half_steps=6)
    scaled = gb.case("R4", gb.plate(2.9, 2.9 * r, nx, ny, q=3700.0, E=2.73e9, t=0.019), "A4-12 scaled", N=N,
                     mode="research_fixed", half_steps=6)
    a, b = gb.run_case(base)["arrays"]["w_bar"], gb.run_case(scaled)["arrays"]["w_bar"]
    assert np.max(np.abs(a - b)) <= 1e-11 * np.max(np.abs(a))


# --- 5. R4: modes, marks, dense against fft -------------------------------------------------------------------

def test_r4_normative_mode_does_not_switch_on_the_research_mode(monkeypatch):
    seen = []
    real = cr.run_coupled

    def spy(*args, **kwargs):
        seen.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(cr, "run_coupled", spy)
    for name in ("r4_n2_normative", "r4_n2_normative_one"):
        res = gb.run_case(gb.CASES[name])
        run = res["api"]["run"]
        assert "research_mode" not in seen[-1] and "stop_on_stabilization" not in seen[-1]
        assert run.parameters.research_mode is False and res["record"]["parameters"]["research_mode"] is False
        assert run.status in (cr.NOT_QUALIFIED, cr.HALF_STEP_LIMIT_REACHED) and run.s == 1
        assert res["record"]["status_class"] == "NOT_QUALIFIED" and not res["record"]["completed"]
        assert not run.steps[0].qr_done and run.steps[0].stabilization_ok is None


@pytest.mark.parametrize("name", [n for n in gb.CASES if gb.CASES[n]["model"] == "R4"])
def test_r4_keeps_provisional_not_qualified(name):
    rec = result_of(name)["record"]
    assert rec["scientific_status"] == cr.MARK == "PROVISIONAL / NOT_QUALIFIED" and rec["mark"] == cr.MARK
    assert rec["completed"] is False and rec["status_class"] != "COMPLETED"
    assert rec.get("converged", False) is False
    assert all(row["mark"] == cr.MARK for row in rec["history"])
    assert any("PROVISIONAL / NOT_QUALIFIED" in text for text in rec["limitations"])


DENSE_GRIDS = [(2, 1.0, (40, 40)), (3, 1.0, (40, 40)), (3, 0.5, (40, 20)), (2, 2.0, (20, 40))]  # n <= 40 (D-048.6)


@pytest.mark.parametrize("N, r, grid", DENSE_GRIDS, ids=str)
def test_r4_dense_and_fft_within_the_field_model_of_the_qa_control(N, r, grid):
    nx, ny = grid
    s = 10
    qa = run_control(SpectralGrid(1.0, r, nx, ny), N, s)
    Wq = qa[s - 1].W
    eps, W = {}, {}
    for tr in ("dense", "fft"):
        res = gb.run_case(gb.case("R4", gb.plate(1.0, r, nx, ny, gb.UNIT), "dense/fft", N=N, mode="research_fixed",
                                  transform=tr, half_steps=s))
        run = res["api"]["run"]
        assert run.status == cr.RESEARCH_LIMIT_REACHED and run.s == s
        W[tr] = res["arrays"]["w"][1:-1, 1:-1]
        eps[tr] = cmp._model(run.steps, qa, s, nx, ny)
        assert np.max(np.abs(W[tr] - Wq)) <= eps[tr] * np.max(np.abs(Wq)), tr
    assert np.max(np.abs(W["dense"] - W["fft"])) <= (eps["dense"] + eps["fft"]) * np.max(np.abs(Wq))


# --- 6. stops without a field, errors are never a success ------------------------------------------------------

@pytest.mark.parametrize("name", ["r3_error_nu", "r3_error_ratio", "r3_error_grid", "r4_n3_rank_loss", "r4_error_E"])
def test_stop_without_a_computed_field(name, tmp_path):
    res = result_of(name)
    rec = res["record"]
    assert rec["field_available"] is False and rec["field"] is None and res["arrays"] == {}
    assert rec["completed"] is False and rec["half_steps_done"] == 0
    files = gb.export(name, res, tmp_path)
    assert set(files) == {"json"} and sorted(p.name for p in tmp_path.iterdir()) == [files["json"]]
    if name == "r4_n3_rank_loss":
        assert rec["status"] == cr.STOPPED_RANK_LOSS_PROVEN and rec["is_error"] is False
        assert res["api"]["run"].field() is None
    else:
        assert rec["status"] == "ERROR_INPUT" and rec["is_error"] is True and rec["D"] is None


def test_r4_physical_input_is_checked_like_r3():
    """run_coupled receives only D: E < 0 and t < 0 give D > 0 and would be accepted by run_coupled alone. The tool
    validates E, t, nu with the R3 library function, so the GUI must do the same (observation for the GUI adapter)."""
    p = gb.CASES["r4_error_E"]["inputs"]
    D = bending_stiffness(p["E"], p["t"], p["nu"])
    assert D > 0
    assert cr.run_coupled(p["a"], p["b"], D, p["q"], p["n_x"], p["n_y"], 2, transform="fft", half_steps=1).s == 1
    assert result_of("r4_error_E")["record"]["status"] == "ERROR_INPUT"


ALL_STATUSES = {
    "R3": ["CONVERGED", "STAGNATED", "MAX_HALF_STEPS", "ERROR_INPUT", "ERROR_LINEAR_SOLVE", "ERROR_NONFINITE",
           "ERROR_DEGENERATE", "ERROR_ENERGY_TEST"],
    "R4": [cr.PROVISIONAL_FIELD_STABILIZED, cr.RESEARCH_LIMIT_REACHED, cr.MAX_HALF_STEPS, cr.NOT_QUALIFIED,
           cr.HALF_STEP_LIMIT_REACHED, cr.PROVISIONAL_RANK_GUARD_STOP, cr.PROVISIONAL_SET_GUARD_STOP,
           cr.PROVISIONAL_ENERGY_E1_STOP, cr.STOPPED_RANK_LOSS_PROVEN, cr.STOPPED_RANK_NUMERICALLY_UNRESOLVED,
           "ERROR_INPUT", "ERROR_LINEAR_SOLVE", "ERROR_NONFINITE", "ERROR_DEGENERATE", "ERROR_ENERGY_TEST"],
}


def test_status_classes_never_promote_a_status():
    for model, statuses in ALL_STATUSES.items():
        for status in statuses:
            cls = gb.status_class(model, status)
            assert cls != "UNKNOWN", (model, status)
            assert (cls == "COMPLETED") == (model == "R3" and status == "CONVERGED")
            assert (cls == "ERROR") == status.startswith("ERROR_")
    for model, status in (("R3", "SOMETHING_NEW"), ("R4", "CONVERGED"), ("R4", None), ("R5", "CONVERGED")):
        assert gb.status_class(model, status) == "UNKNOWN"  # unknown = error, never a success
    assert gb.scientific_status("R4", cr.PROVISIONAL_FIELD_STABILIZED) == cr.MARK


def _r3_failure(monkeypatch, where):
    if where == "linear solve at s = 3":
        real, count = rr.next_halfstep, {"n": 0}

        def failing(record):
            count["n"] += 1
            if count["n"] == 2:
                raise LinearSolveError(ERROR_LINEAR_SOLVE, "injected by QA")
            return real(record)

        monkeypatch.setattr(rr, "next_halfstep", failing)
    else:
        monkeypatch.setattr(rr, "check_pair", lambda *a, **k: (_ for _ in ()).throw(
            EnergyError(ERROR_ENERGY_TEST, "injected by QA")))


@pytest.mark.parametrize("where, status, steps", [("linear solve at s = 3", ERROR_LINEAR_SOLVE, 2),
                                                  ("energy test at s = 2", ERROR_ENERGY_TEST, 2)])
def test_r3_failures_are_errors_with_their_partial_history(monkeypatch, tmp_path, where, status, steps):
    _r3_failure(monkeypatch, where)
    res = gb.run_case(gb.CASES["r3_square"])
    rec = res["record"]
    assert rec["status"] == status and rec["status_class"] == "ERROR" and rec["is_error"] and not rec["completed"]
    assert rec["half_steps_done"] == steps and rec["field"]["role"] == "field of the last completed half-step"
    assert "no result is claimed" in rec["scientific_status"] and status in rec["reason"]
    gb.export("failed", res, tmp_path)
    saved = json.loads((tmp_path / "failed.json").read_text(encoding="utf-8"))
    assert saved["completed"] is False and saved["is_error"] is True and saved["status"] == status


def test_r4_failures_are_not_reported_as_success(monkeypatch):
    real_pair = cr.pair_energy
    with monkeypatch.context() as m:
        m.setattr(cr, "pair_energy", lambda *a, **k: dataclasses.replace(real_pair(*a, **k), status=cr.E1_VIOLATED,
                                                                         e1=False))
        rec = gb.run_case(gb.CASES["r4_n2_research"])["record"]
    assert rec["status"] == cr.PROVISIONAL_ENERGY_E1_STOP and rec["status_class"] == "PROVISIONAL_STOPPED"
    assert not rec["completed"] and rec["half_steps_done"] == 2
    real_s, real_c = kc._solve_modes, kc._correction_modes
    with monkeypatch.context() as m:  # persistent error after the one refinement step (D-048.2)
        m.setattr(kc, "_solve_modes", lambda *a, **k: real_s(*a, **k) + 1e-6)
        m.setattr(kc, "_correction_modes", lambda *a, **k: real_c(*a, **k) + 1e-6)
        res = gb.run_case(gb.CASES["r4_n2_research"])
    rec = res["record"]
    assert rec["status"] == ERROR_LINEAR_SOLVE and rec["is_error"] and not rec["completed"]
    assert rec["field_available"] is False and res["arrays"] == {}


# --- 7. reproducibility ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["r3_rect_r5_unequal", "r4_n3_research_rect"])
def test_same_input_same_environment_gives_identical_results(name):
    first, second = gb.run_case(gb.CASES[name]), gb.run_case(gb.CASES[name])
    assert json.dumps(first["record"], sort_keys=True) == json.dumps(second["record"], sort_keys=True)
    for key in first["arrays"]:
        np.testing.assert_array_equal(first["arrays"][key], second["arrays"][key])


def test_command_line_runs_and_repeats_byte_identical(tmp_path):
    outs = []
    for k in (1, 2):
        out = tmp_path / str(k)
        proc = subprocess.run([sys.executable, str(ROOT / "scripts" / "gui_backend_cases.py"), "run", "r3_square",
                               "r4_n2_normative", "r4_n3_rank_loss", "--out", str(out)],
                              capture_output=True, text=True, cwd=ROOT)
        assert proc.returncode == 0, proc.stderr
        outs.append(out)
    names = sorted(p.name for p in outs[0].iterdir())
    assert "r3_square_w.csv" in names and "r4_n3_rank_loss.json" in names and "r4_n3_rank_loss_w.csv" not in names
    for n in names:
        assert (outs[0] / n).read_bytes() == (outs[1] / n).read_bytes(), n


# --- 8. stored golden cases for the GUI adapter ---------------------------------------------------------------

GOLDEN = json.loads((gb.GOLDEN / "cases.json").read_text(encoding="utf-8"))
BITWISE = os.environ.get("KKVIM_GUI_BITWISE") == "1"  # only on the reference machine of the stored values


@pytest.mark.parametrize("name", sorted(GOLDEN["cases"]))
def test_golden_case_is_reproduced(name, capsys):
    stored = GOLDEN["cases"][name]
    assert stored["spec"] == gb.CASES[name]  # the golden set describes the current cases
    res = gb.run_case(gb.CASES[name])
    rec = res["record"]
    for key in ("expected_status", "expected_status_class"):
        assert rec[key.replace("expected_", "")] == stored[key], key
    assert rec["is_error"] == stored["is_error"] and rec["half_steps_done"] == stored["half_steps_done"]
    assert rec["scientific_status"] == stored["scientific_status"] and rec["field_available"] == stored["field_available"]
    if stored["D"] is not None:
        assert abs(rec["D"] - stored["D"]) <= 8 * U * stored["D"]  # both within 4u of the exact value
    if not stored["field_available"]:
        return
    _, _, w_stored = gb.read_field_csv(gb.GOLDEN / stored["files"]["w_csv"])
    w = res["arrays"]["w"]
    assert w.shape == w_stored.shape and np.all(w[0] == 0) and np.all(w[:, -1] == 0)
    diff = float(np.max(np.abs(w - w_stored)) / np.max(np.abs(w_stored)))
    if BITWISE:
        np.testing.assert_array_equal(w, w_stored)
    elif gb.CASES[name]["model"] == "R4":
        spec = gb.CASES[name]
        p = spec["inputs"]
        run = res["api"]["run"]
        qa = run_control(SpectralGrid(1.0, p["b"] / p["a"], p["n_x"], p["n_y"]), spec["N"], run.s)
        assert diff <= 2 * cmp._model(run.steps, qa, run.s, p["n_x"], p["n_y"]), diff
    with capsys.disabled():
        print(f"\n{name}: max relative difference to the stored field {diff:.3e}"
              + ("" if BITWISE or gb.CASES[name]["model"] == "R4" else " (R3: informative, no cross-platform criterion)"))


def test_golden_set_covers_n_1_2_3_both_r4_modes_and_stops():
    cases = GOLDEN["cases"]
    assert {c["spec"]["N"] for c in cases.values()} == {1, 2, 3}
    assert {c["spec"].get("mode") for c in cases.values()} >= {"normative", "research", "research_fixed"}
    classes = {c["expected_status_class"] for c in cases.values()}
    assert classes >= {"COMPLETED", "PROVISIONAL", "NOT_QUALIFIED", "STOPPED_RANK_LOSS", "ERROR"}
    for c in cases.values():
        assert c["limitations"] and c["provenance"] and c["api_call"]
        assert (c["expected_status_class"] == "COMPLETED") == (c["spec"]["model"] == "R3"
                                                               and c["expected_status"] == "CONVERGED")
    assert GOLDEN["source"]["git_commit"] and not math.isnan(len(cases))
