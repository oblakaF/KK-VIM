"""Independent QA acceptance of the GUI backend (Builder ``gui/kkvim-backend`` @ 2b9b199).

Role QA, goals G3.2, G3.3, G4 (G5 interface preparation). Not a scientific acceptance of R4.

Independence:
- references are the QA control data of PR #24 (``qa/gui-backend-acceptance`` @ 1d6af27,
  docs/results/gui_backend/): read from that git commit (or from the folder in KKVIM_QA_CONTROL_DIR),
  never regenerated through the GUI and never taken from Builder's tests or adapter checks;
- direct calls of kkvim are written here from the ``api_call`` of each control case;
- comparison rules are those of cases.json -> comparison (fixed in PR #24 before this review):
  same environment -> bitwise; other platform R3 -> status, s, D, shape exactly (field informative);
  other platform R4 -> field within the doubled first-order model ``_model`` of
  tests/cross/test_r4_controller_comparison.py (as K-1 of D-048). No new tolerance is introduced.

Qt-free checks always run; Qt window checks need the "gui" extra and the "gui-test" group.
"""

import ast
import csv
import importlib.util
import io
import json
import math
import multiprocessing
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import scipy

import kkvim.coupled_run as cr
import kkvim.rectangle_run as rr
from kkvim.linear_solver import ERROR_LINEAR_SOLVE, LinearSolveError
from kkvim.rectangle import validated_rectangle_input

from kkvim_gui.backend import export, kkvim_adapter, process
from kkvim_gui.backend.request import R4_NORMATIVE, R4_RESEARCH_FIXED, R4_RESEARCH_STOP, RunRequest
from kkvim_gui.model.setup_state import SetupState, to_request, validate

ROOT = Path(__file__).resolve().parents[2]
CONTROL_COMMIT = "1d6af27e3a4e3434080e074d0d21c848f2261362"
CONTROL_PATH = "docs/results/gui_backend"
U = 2.0**-53
MARK = "PROVISIONAL / NOT_QUALIFIED"
MODE_OF_CASE = {"normative": R4_NORMATIVE, "research": R4_RESEARCH_STOP, "research_fixed": R4_RESEARCH_FIXED}
HAVE_QT = all(importlib.util.find_spec(m) is not None for m in ("PySide6", "pytestqt", "pyqtgraph"))
needs_qt = pytest.mark.skipif(not HAVE_QT, reason="Qt checks need: uv sync --extra gui --group gui-test")


# --- control data of PR #24 ------------------------------------------------------------------------------------

def _control_text(relative):
    folder = os.environ.get("KKVIM_QA_CONTROL_DIR")
    if folder:
        return (Path(folder) / relative).read_text(encoding="utf-8")
    out = subprocess.run(["git", "show", f"{CONTROL_COMMIT}:{CONTROL_PATH}/{relative}"], cwd=ROOT,
                         capture_output=True)
    if out.returncode != 0:
        pytest.skip(f"QA control data of PR #24 ({CONTROL_COMMIT[:7]}) not available: set KKVIM_QA_CONTROL_DIR")
    return out.stdout.decode("utf-8")


def control():
    if not hasattr(control, "data"):
        control.data = json.loads(_control_text("cases.json"))
    return control.data


def control_grid(relative):
    """(x, y, W) of a control CSV: header 'x\\y', y nodes; each row x, values."""
    rows = list(csv.reader(io.StringIO(_control_text(relative))))
    y = np.array([float(v) for v in rows[0][1:]])
    x = np.array([float(r[0]) for r in rows[1:]])
    w = np.array([[float(v) for v in r[1:]] for r in rows[1:]])
    return x, y, w


def control_history(relative):
    return list(csv.DictReader(io.StringIO(_control_text(relative))))


def same_environment():
    """Source environment of the control data (cases.json -> source) equals this one."""
    src = control()["source"]
    return (src["python"], src["platform"], src["numpy"], src["scipy"]) == (
        platform.python_version(), platform.platform(), np.__version__, scipy.__version__)


PLATE = {"a": 1.0, "b": 1.0, "t": 0.01, "E": 2.1e11, "nu": 0.3, "q": 1000.0}
CASE_NAMES = ["r3_square", "r3_rect_b2a", "r3_rect_a2b", "r3_rect_r5_unequal", "r3_negative_q", "r3_error_nu",
              "r3_error_ratio", "r3_error_grid", "r4_n2_normative", "r4_n2_normative_one", "r4_n2_research",
              "r4_n3_research_rect", "r4_n3_research_fixed", "r4_n2_dense", "r4_n3_rank_loss", "r4_error_E"]


# --- the form path of the GUI (SetupState -> RunRequest), text as a user types it ------------------------------

def form_state(inp, N, mode=None, transform="fft", half_steps=None, units=None):
    """SetupState filled like the form: every value typed as text with a unit (default m and Pa)."""
    units = units or {}
    state = SetupState().with_choice(N=N)
    texts = {"a": inp["a"], "b": inp["b"], "h": inp["t"], "E": inp["E"], "nu": inp["nu"], "q": inp["q"],
             "n_x": inp["n_x"], "n_y": inp["n_y"]}
    base = {"a": "m", "b": "m", "h": "m", "E": "Pa", "q": "Pa"}
    for key, value in texts.items():
        unit, text = units.get(key, (base.get(key), None))
        state = state.with_text(key, text if text is not None else (repr(value) if isinstance(value, float) else str(value)))
        if key in base:
            state = state.with_unit(key, unit)
    if N != 1:
        state = state.with_choice(transform=transform)
        if mode is not None and mode != R4_NORMATIVE:
            state = state.with_choice(r4_mode=mode)  # the explicit choice (research switch)
        if half_steps is not None:
            state = state.with_text("half_steps", str(half_steps))
    return state


def case_state(name):
    c = control()["cases"][name]
    inp = c["input"]
    if c["spec"]["model"] == "R3":
        return form_state(inp, 1)
    return form_state(inp, c["spec"]["N"], MODE_OF_CASE[inp["mode"]], inp["transform"], inp["half_steps"])


def case_request(name):
    """RunRequest of the case: through the form, or (inputs the form refuses: t, E < 0) built directly."""
    c = control()["cases"][name]
    request = to_request(case_state(name))
    if request is not None:
        return request
    inp = c["input"]
    return RunRequest(entered=(), a=inp["a"], b=inp["b"], t=inp["t"], E=inp["E"], nu=inp["nu"], q=inp["q"],
                      n_x=inp["n_x"], n_y=inp["n_y"], N=c["spec"]["N"], r4_mode=MODE_OF_CASE[inp["mode"]],
                      transform=inp["transform"], half_steps=inp["half_steps"])


def direct_api(name):
    """The api_call of the control case, written here (not the adapter)."""
    c = control()["cases"][name]
    inp = c["input"]
    if c["spec"]["model"] == "R3":
        return rr.run_rectangle_physical(inp["a"], inp["b"], inp["t"], inp["E"], inp["nu"], inp["q"], inp["n_x"], inp["n_y"])
    _, D = validated_rectangle_input(inp["a"], inp["b"], inp["t"], inp["E"], inp["nu"], inp["q"], inp["n_x"], inp["n_y"])
    kwargs = {"transform": inp["transform"], "half_steps": inp["half_steps"]}
    if inp["mode"] != "normative":  # the normative call passes no research_mode (library default False)
        kwargs.update(research_mode=True, stop_on_stabilization=inp["mode"] == "research")
    return D, cr.run_coupled(inp["a"], inp["b"], D, inp["q"], inp["n_x"], inp["n_y"], c["spec"]["N"], **kwargs)


def r4_model(name, s):
    """Doubled first-order field model of K-1 (tests/cross/test_r4_controller_comparison.py, _model) for the control
    case: log quantities from the control history, QA spectral control for ls_cond (unit plate a = 1, b = b/a)."""
    spec = importlib.util.spec_from_file_location("r4cmp", ROOT / "tests" / "cross" / "test_r4_controller_comparison.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from kkvim_reference.r4_spectral_control import SpectralGrid, run_control

    c = control()["cases"][name]
    inp, N = c["input"], c["spec"]["N"]
    hist = control_history(c["files"]["history_csv"])
    log = [SimpleNamespace(direction=h["direction"], cond_M_max=float(h["cond_M_max"]),
                           rank=SimpleNamespace(sigma=np.zeros(N), kappa_rep=float(h["kappa_rep"]))) for h in hist]
    qa = run_control(SpectralGrid(1.0, inp["b"] / inp["a"], inp["n_x"], inp["n_y"]), N, s)
    return 2.0 * module._model(log, qa, s, inp["n_x"], inp["n_y"])


# =================================================================================================================
# 1. Normative and research modes
# =================================================================================================================

def test_default_is_normative_and_research_mode_false():
    state = SetupState()
    assert state.r4_mode == R4_NORMATIVE and not state.research
    for N in (2, 3):
        request = to_request(form_state({**PLATE, "n_x": 24, "n_y": 24}, N))
        assert request.r4_mode == R4_NORMATIVE and request.research_mode is False


@pytest.mark.parametrize("path", [(1, 2), (2, 3), (3, 2), (2, 1, 2), (3, 1, 3), (1, 3, 2)])
@pytest.mark.parametrize("mode", [R4_RESEARCH_STOP, R4_RESEARCH_FIXED])
def test_every_change_of_N_resets_the_research_choice(path, mode):
    state = SetupState().with_choice(N=path[0])
    if path[0] != 1:
        state = state.with_choice(r4_mode=mode)
        assert state.research
    for N in path[1:]:
        state = state.with_choice(N=N)  # what the N combo does (setup_view: set_choice(N=...))
        assert state.r4_mode == R4_NORMATIVE and not state.research
    request = to_request(state)
    assert not request.research_mode


def test_changing_N_alone_never_enables_qr(monkeypatch):
    """Spy on run_coupled: after any sequence of N changes the call carries research_mode=False."""
    calls = []
    real = cr.run_coupled

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(cr, "run_coupled", spy)
    state = form_state({**PLATE, "n_x": 24, "n_y": 24}, 2, R4_RESEARCH_STOP).with_choice(N=3).with_choice(N=2)
    outcome = kkvim_adapter.execute(to_request(state))
    assert calls and calls[-1]["research_mode"] is False and outcome.status == cr.NOT_QUALIFIED
    assert all(row["qr_done"] is False for row in outcome.history)


def test_request_flags_of_the_three_modes():
    base = dict(entered=(), a=1.0, b=1.0, t=0.01, E=2.1e11, nu=0.3, q=1000.0, n_x=24, n_y=24, N=2)
    assert (RunRequest(**base).research_mode, RunRequest(**base).stop_on_stabilization) == (False, True)
    r = RunRequest(**base, r4_mode=R4_RESEARCH_STOP)
    assert (r.research_mode, r.stop_on_stabilization) == (True, True)
    r = RunRequest(**base, r4_mode=R4_RESEARCH_FIXED)
    assert (r.research_mode, r.stop_on_stabilization) == (True, False)


@needs_qt
@pytest.mark.parametrize("name", ["r4_n2_normative", "r4_n2_normative_one"])
def test_normative_shows_not_qualified_after_s1(name):
    from kkvim_gui.analysis.results import statuses

    c = control()["cases"][name]
    outcome = kkvim_adapter.execute(case_request(name))
    assert outcome.status == c["expected_status"] and outcome.s == 1 == c["half_steps_done"]
    assert outcome.mark == MARK and outcome.stop_kind == "none" and not outcome.is_error
    assert outcome.request.research_mode is False
    assert [row["qr_done"] for row in outcome.history] == [h["qr_done"] == "True" for h in
                                                           control_history(c["files"]["history_csv"])] == [False]
    execution, stop, science = statuses(outcome, None)
    assert stop[0] == "neutral" and science[0] == "warning" and MARK in science[1]  # NOT_QUALIFIED is shown
    if outcome.status == cr.NOT_QUALIFIED:
        assert cr.NOT_QUALIFIED in stop[1]
    # HALF_STEP_LIMIT_REACHED: the stop line has no status code (observation Н-2 of the QA report)


def _all_texts(outcome):
    from kkvim_gui.analysis.results import statuses

    texts = [outcome.stop_text, outcome.science_text, *outcome.notes, *(t for _, t in statuses(outcome, None))]
    manifest = export.manifest(outcome)
    texts += [json.dumps(manifest, ensure_ascii=False)]
    return texts


@needs_qt
@pytest.mark.parametrize("name", ["r4_n2_research", "r4_n3_research_rect", "r4_n2_dense"])
def test_field_stabilized_is_never_shown_as_proven_convergence(name, tmp_path):
    from kkvim_gui.analysis.results import statuses

    outcome = kkvim_adapter.execute(case_request(name))
    assert outcome.status == cr.PROVISIONAL_FIELD_STABILIZED
    execution, stop, science = statuses(outcome, None)
    assert stop[0] == "warning" and science[0] == "warning"  # never "success"
    for text in _all_texts(outcome):
        low = text.lower()
        assert "converged" not in low
        for i in [k for k in range(len(low)) if low.startswith("сходимост", k)]:
            assert low[max(0, i - 20):i].rstrip().endswith(("не", "не доказанная")), text[max(0, i - 40):i + 20]
    paths = export.write_result(outcome, tmp_path)
    assert paths["history"].read_text(encoding="utf-8").splitlines()[0] == f"# {MARK}"
    assert paths["field"].read_text(encoding="utf-8").splitlines()[0] == f"# {MARK}"
    saved = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert MARK in saved["marks"] and saved["statuses"]["mark"] == MARK
    assert saved["statuses"]["scientific"]["kind"] == "warning"


def test_help_and_form_texts_never_claim_convergence_for_r4():
    texts = []
    for path in (ROOT / "src" / "kkvim_gui").rglob("*"):
        if path.suffix in (".py", ".md", ".json"):
            texts.append((path, path.read_text(encoding="utf-8")))
    for path, text in texts:
        for line in text.splitlines():
            if "PROVISIONAL_FIELD_STABILIZED" in line and "сходим" in line.lower():
                assert "не сходимость" in line.lower() or "не доказанная сходимость" in line.lower(), (path, line)


# =================================================================================================================
# 2. dense only while max(n_x, n_y) <= 40 (D-048.6): form, check_request, dispatcher, computing process
# =================================================================================================================

GRIDS = [((40, 40), True), ((40, 41), False), ((41, 40), False), ((40, 160), False), ((160, 40), False)]


def _dense_request(N, nx, ny, mode=R4_NORMATIVE):
    return RunRequest(entered=(), **PLATE, n_x=nx, n_y=ny, N=N, r4_mode=mode, transform="dense", half_steps=5)


@pytest.mark.parametrize("N", [2, 3])
@pytest.mark.parametrize("grid, allowed", GRIDS)
def test_dense_rule_in_the_form(N, grid, allowed):
    nx, ny = grid
    for mode in (None, R4_RESEARCH_STOP, R4_RESEARCH_FIXED):
        state = form_state({**PLATE, "n_x": nx, "n_y": ny}, N, mode, "dense", 5)
        errors = validate(state)
        assert ("transform" in errors) is (not allowed)
        assert (to_request(state) is None) is (not allowed)
        fft = state.with_choice(transform="fft")
        assert "transform" not in validate(fft) and to_request(fft) is not None


@pytest.mark.parametrize("N", [2, 3])
@pytest.mark.parametrize("grid, allowed", GRIDS)
def test_dense_rule_in_check_request_and_in_the_process_before_kkvim(N, grid, allowed, monkeypatch):
    nx, ny = grid
    request = _dense_request(N, nx, ny)
    problems = kkvim_adapter.check_request(request)
    assert any("dense" in p for p in problems) is (not allowed)
    calls = []
    monkeypatch.setattr(cr, "run_coupled", lambda *a, **k: calls.append((a, k)) or (_ for _ in ()).throw(
        RuntimeError("QA spy: kkvim reached")))
    receiver, sender = multiprocessing.Pipe(duplex=False)
    process.compute(request, sender)  # the body of the computing process, in this process
    kind, payload = receiver.recv()
    if allowed:
        assert calls and calls[0][1]["transform"] == "dense"  # reached kkvim (the spy then raises -> crash)
        assert kind == process.CRASH
    else:
        assert not calls and kind == process.INPUT and "dense" in payload


@needs_qt
@pytest.mark.parametrize("N", [2, 3])
@pytest.mark.parametrize("grid, allowed", GRIDS)
def test_dense_rule_in_the_dispatcher_before_a_process_exists(qapp, N, grid, allowed):
    from kkvim_gui.model.runs import RunManager

    created = []

    def factory(request):
        created.append(request)
        return SimpleNamespace(poll=lambda: None, terminate=lambda timeout=2.0: True)

    manager = RunManager(process_factory=factory)
    request = _dense_request(N, *grid)
    snapshot = form_state({**PLATE, "n_x": grid[0], "n_y": grid[1]}, N, None, "fft", 5)
    if allowed:
        run = manager.start(snapshot, request)
        assert created == [request] and run.active
        manager.stop(run.run_id)
    else:
        with pytest.raises(kkvim_adapter.AdapterInputError, match="dense"):
            manager.start(snapshot, request)
        assert created == [] and manager.runs == [] and manager.active is None


@pytest.mark.parametrize("grid", [(40, 160), (160, 40)])
def test_dense_rule_in_a_real_computing_process(grid):
    p = process.SolverProcess(_dense_request(2, *grid))
    t0 = time.monotonic()
    while p.poll() is None and time.monotonic() - t0 < 60:
        time.sleep(0.05)
    assert p.message[0] == process.INPUT and "dense" in p.message[1]
    p.terminate()


def test_dense_40x40_is_computed_and_matches_the_direct_call():
    request = _dense_request(3, 40, 40, R4_RESEARCH_FIXED)
    outcome = kkvim_adapter.execute(request)
    _, D = validated_rectangle_input(*PLATE.values(), 40, 40)
    run = cr.run_coupled(1.0, 1.0, D, 1000.0, 40, 40, 3, transform="dense", half_steps=5, research_mode=True,
                         stop_on_stabilization=False)
    assert outcome.status == run.status and outcome.s == run.s
    assert np.array_equal(outcome.w, run.field())


# =================================================================================================================
# 3. The 16 control cases of PR #24 through the form path
# =================================================================================================================

@pytest.mark.parametrize("name", CASE_NAMES)
def test_control_case_through_the_gui_path(name):
    c = control()["cases"][name]
    request = case_request(name)
    inp = c["input"]
    for key, attr in (("a", "a"), ("b", "b"), ("t", "t"), ("E", "E"), ("nu", "nu"), ("q", "q"), ("n_x", "n_x"),
                      ("n_y", "n_y")):
        assert getattr(request, attr) == inp[key]  # physical input reaches the solver unchanged (bitwise)
    if c["expected_status"] == "ERROR_INPUT":
        problems = kkvim_adapter.check_request(request)
        assert problems and any("ERROR_INPUT" in p for p in problems)
        with pytest.raises(kkvim_adapter.AdapterInputError):
            kkvim_adapter.execute(request)
        assert c["field_available"] is False and c["half_steps_done"] == 0
        return
    outcome = kkvim_adapter.execute(request)
    assert outcome.status == c["expected_status"] and outcome.s == c["half_steps_done"]
    assert outcome.is_error is c["is_error"] and outcome.has_field is c["field_available"]
    assert (outcome.mark == MARK) is (c["spec"]["model"] == "R4") and (outcome.solver == c["spec"]["model"])
    if same_environment():
        assert outcome.D == c["D"]
    else:
        assert abs(outcome.D - c["D"]) <= 8 * U * abs(c["D"])
    if not c["field_available"]:
        assert outcome.w is None
        return
    f = c["field"]
    assert list(outcome.w.shape) == f["shape"]
    x, y, w = control_grid(c["files"]["w_csv"])
    _, _, w_bar = control_grid(c["files"]["w_bar_csv"])
    assert np.array_equal(outcome.x, x) and np.array_equal(outcome.y, y)
    assert np.all(outcome.w[0, :] == 0) and np.all(outcome.w[-1, :] == 0) and np.all(outcome.w[:, 0] == 0)
    if same_environment():
        assert np.array_equal(outcome.w, w) and np.array_equal(outcome.w_bar, w_bar)
        ext = outcome.extremum()
        assert (ext["i"], ext["j"], ext["w"]) == (f["argmax_abs"]["i"], f["argmax_abs"]["j"], f["w_at_max_abs"])
    elif c["spec"]["model"] == "R4":
        eps = r4_model(name, outcome.s)
        assert np.max(np.abs(outcome.w - w)) <= eps * np.max(np.abs(w)), eps
    # R3 on another platform: field informative only (no accepted cross-platform criterion)


@pytest.mark.parametrize("name", [n for n in CASE_NAMES if n.startswith("r4_") and "error" not in n and "rank" not in n])
def test_r4_history_against_the_control_history(name):
    c = control()["cases"][name]
    outcome = kkvim_adapter.execute(case_request(name))
    hist = control_history(c["files"]["history_csv"])
    assert [row["s"] for row in outcome.history] == [int(h["s"]) for h in hist]
    assert [row["direction"] for row in outcome.history] == [h["direction"] for h in hist]
    if not same_environment():
        pytest.skip("history values: bitwise only in the source environment of the control data")
    gui_to_control = {"eps_inf": "eps_inf", "eps_E": "eps_E", "w_max": "w_max", "P": "P", "delta_P": "delta_P",
                      "Q": "Q", "tau": "tau", "m_s": "m_s", "eta": "eta_N", "rho_N": "rho_N", "kappa_rep": "kappa_rep",
                      "qr_field_error": "qr_field_error"}
    for row, h in zip(outcome.history, hist):
        for gui, ctl in gui_to_control.items():
            expected = None if h[ctl] == "" else float(h[ctl])
            assert row[gui] == expected, (row["s"], gui)
        assert str(row["qr_done"]) == h["qr_done"] and str(row["stop_reason"] or "") == h["stop_reason"]


@pytest.mark.parametrize("name", [n for n in CASE_NAMES if n.startswith("r3_") and "error" not in n])
def test_r3_history_against_the_control_history(name):
    c = control()["cases"][name]
    outcome = kkvim_adapter.execute(case_request(name))
    hist = control_history(c["files"]["history_csv"])
    assert [row["s"] for row in outcome.history] == [int(h["s"]) for h in hist]
    if not same_environment():
        pytest.skip("history values: bitwise only in the source environment of the control data")
    for row, h in zip(outcome.history, hist):
        for key in ("eps_inf", "eps_E", "eps_L2", "theta", "P", "delta_P", "Q", "tau", "eta", "w_bar_centre"):
            expected = None if h[key] == "" else float(h[key])
            assert row[key] == expected, (row["s"], key)
        assert row["energy_status"] == h["energy_status"]


@pytest.mark.parametrize("name", [n for n in CASE_NAMES if "error" not in n])
def test_gui_path_equals_the_direct_api_call_bitwise(name):
    """cases.json -> comparison -> same_environment: the GUI reproduces the direct API call bitwise."""
    c = control()["cases"][name]
    outcome = kkvim_adapter.execute(case_request(name))
    if c["spec"]["model"] == "R3":
        result = direct_api(name)
        assert (outcome.status, outcome.D, outcome.s) == (result.status, result.D, result.run.s)
        assert np.array_equal(outcome.w, result.run.last_field())
        assert np.array_equal(outcome.w_bar, result.dimensionless_field())
    else:
        D, run = direct_api(name)
        assert (outcome.status, outcome.D, outcome.s) == (run.status, D, run.s)
        field = run.field()
        assert (outcome.w is None and field is None) or np.array_equal(outcome.w, field)
        if field is not None:
            inp = c["input"]
            assert np.array_equal(outcome.w_bar, field * D / (inp["q"] * inp["a"] ** 4))  # MATH_SPEC 2.5, a = x side


def test_units_of_the_form_reach_the_solver_exactly():
    """r3_rect_a2b entered in mm, GPa and kPa gives the control SI input and the control result."""
    c = control()["cases"]["r3_rect_a2b"]
    units = {"a": ("mm", "2000"), "b": ("mm", "1000"), "h": ("mm", "10"), "E": ("GPa", "210"), "q": ("kPa", "1")}
    request = to_request(form_state(c["input"], 1, units=units))
    assert (request.a, request.b, request.t, request.E, request.q) == (2.0, 1.0, 0.01, 2.1e11, 1000.0)
    assert ("h", "10", "mm") in request.entered and ("E", "210", "GPa") in request.entered
    outcome = kkvim_adapter.execute(request)
    reference = kkvim_adapter.execute(case_request("r3_rect_a2b"))
    assert outcome.status == c["expected_status"] and np.array_equal(outcome.w, reference.w)
    assert outcome.D == reference.D and np.array_equal(outcome.w_bar, reference.w_bar)


def test_decimal_comma_and_unicode_minus_give_the_same_request():
    inp = control()["cases"]["r3_negative_q"]["input"]
    a = to_request(form_state(inp, 1))
    b = to_request(form_state(inp, 1, units={"b": ("m", "1,5"), "q": ("Pa", "−500"), "nu": (None, "0,3")}))
    assert (a.b, a.q, a.nu) == (b.b, b.q, b.nu) == (1.5, -500.0, 0.3)


def test_unit_conversion_is_within_two_roundings_but_not_always_the_decimal_value():
    """Finding Н-1 (non-blocking): units.to_base multiplies by 1e-2 / 1e-3, which are not exact binary numbers.
    Two roundings (the factor and the product), so the SI value is within gamma_2 = 2u/(1 - 2u) of the exact decimal
    value (standard bound), but not always the double nearest to it: e.g. 70 cm -> 0.7000000000000001 m."""
    from decimal import Decimal

    from kkvim_gui.backend.units import to_base

    worst = 0.0
    for unit, exponent in (("cm", -2), ("mm", -3)):
        for n in range(1, 2001):
            got = to_base(float(n), unit)
            exact = Decimal(n).scaleb(exponent)
            worst = max(worst, abs(float((Decimal(got) - exact) / exact)))
    assert worst <= 2 * U / (1 - 2 * U)  # gamma_2: fl(n * fl(10^-k)), two roundings
    assert to_base(70.0, "cm") != 0.7  # not the double nearest to 0.7: the documented "exact prefixes" do not hold


# =================================================================================================================
# 4. Errors, partial history, stop, export, protocol
# =================================================================================================================

def _inject_r3_linear_failure(monkeypatch, at_call=2):
    real, count = rr.next_halfstep, {"n": 0}

    def failing(record):
        count["n"] += 1
        if count["n"] == at_call:
            raise LinearSolveError(ERROR_LINEAR_SOLVE, "injected by QA")
        return real(record)

    monkeypatch.setattr(rr, "next_halfstep", failing)


@needs_qt
def test_solver_error_with_partial_history_is_diagnostic_only(monkeypatch, tmp_path):
    from kkvim_gui.analysis.results import statuses

    _inject_r3_linear_failure(monkeypatch)
    outcome = kkvim_adapter.execute(case_request("r3_square"))
    assert outcome.status == ERROR_LINEAR_SOLVE and outcome.is_error and outcome.diagnostic_only
    assert outcome.s == len(outcome.history) == 2  # half-steps completed before the failure
    execution, stop, _ = statuses(outcome, None)
    assert execution[0] == "error" and stop[0] == "error"
    paths = export.write_result(outcome, tmp_path)
    saved = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert saved["results"]["completed_without_error"] is False
    assert saved["results"]["field_role"].startswith("diagnostic only")
    assert any(m.startswith("DIAGNOSTIC") for m in saved["marks"])
    assert paths["field"].read_text(encoding="utf-8").splitlines()[0].startswith("# DIAGNOSTIC")


@needs_qt
def test_r4_failures_are_never_success(monkeypatch):
    import dataclasses

    from kkvim_gui.analysis.results import statuses

    real_pair = cr.pair_energy
    monkeypatch.setattr(cr, "pair_energy", lambda *a, **k: dataclasses.replace(real_pair(*a, **k), status=cr.E1_VIOLATED,
                                                                               e1=False))
    outcome = kkvim_adapter.execute(case_request("r4_n2_research"))
    assert outcome.status == cr.PROVISIONAL_ENERGY_E1_STOP and outcome.s == 2
    assert outcome.stop_kind == "not_met" and statuses(outcome, None)[1][0] == "warning" and outcome.mark == MARK


def test_crash_of_the_computing_process_is_an_error_message():
    request = RunRequest(entered=(), **PLATE, n_x=24, n_y=24, N=2, transform="xyz", half_steps=5)
    p = process.SolverProcess(request)  # a request the form cannot build: the process crashes inside kkvim
    t0 = time.monotonic()
    while p.poll() is None and time.monotonic() - t0 < 60:
        time.sleep(0.05)
    assert p.message[0] == process.CRASH and "Traceback" in p.message[1]
    p.terminate()


@pytest.mark.parametrize("name", ["r3_square", "r4_n2_research", "r4_n3_rank_loss"])
def test_real_computing_process_returns_the_same_outcome(name):
    request = case_request(name)
    p = process.SolverProcess(request)
    t0 = time.monotonic()
    while p.poll() is None and time.monotonic() - t0 < 120:
        time.sleep(0.05)
    kind, outcome = p.message
    p.terminate()
    inline = kkvim_adapter.execute(request)
    assert kind == process.OK and (outcome.status, outcome.D, outcome.s) == (inline.status, inline.D, inline.s)
    assert (outcome.w is None and inline.w is None) or np.array_equal(outcome.w, inline.w)
    assert repr(outcome.history) == repr(inline.history)  # bitwise (repr round trip), NaN-safe


@pytest.mark.parametrize("name", ["r3_rect_r5_unequal", "r4_n3_research_fixed", "r4_n3_rank_loss"])
def test_export_files_reproduce_the_control_data(name, tmp_path):
    c = control()["cases"][name]
    outcome = kkvim_adapter.execute(case_request(name))
    paths = export.write_result(outcome, tmp_path / name)
    saved = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    si = saved["input"]["si"]
    inp = c["input"]
    assert {k: (si[k]["value"], si[k]["unit"]) for k in si} == {
        "a": (inp["a"], "m"), "b": (inp["b"], "m"), "t": (inp["t"], "m"), "E": (inp["E"], "Pa"), "nu": (inp["nu"], "1"),
        "q": (inp["q"], "Pa")}
    assert saved["results"]["D"] == {"value": outcome.D, "unit": "N·m"}
    assert saved["statuses"]["solver_status"] == c["expected_status"] and saved["results"]["s"] == c["half_steps_done"]
    assert saved["results"]["field_available"] is c["field_available"] and ("field" in paths) is c["field_available"]
    if not c["field_available"]:
        return
    lines = paths["field"].read_text(encoding="utf-8").splitlines()
    body = [line for line in lines if not line.startswith("#")]
    assert body[0] == "i,j,x [m],y [m],w [m],w_bar [1]"
    rows = list(csv.reader(body[1:]))
    w = np.zeros(outcome.w.shape)
    w_bar = np.zeros(outcome.w.shape)
    for i, j, _, _, wv, wb in rows:
        w[int(i), int(j)], w_bar[int(i), int(j)] = float(wv), float(wb)
    assert np.array_equal(w, outcome.w) and np.array_equal(w_bar, outcome.w_bar)  # repr round trip
    if same_environment():
        _, _, w_ctl = control_grid(c["files"]["w_csv"])
        assert np.array_equal(w, w_ctl)
    header = [line for line in paths["history"].read_text(encoding="utf-8").splitlines() if not line.startswith("#")][0]
    for col in outcome.columns:
        if col.kind == "number" and col.key not in ("s", "k"):
            assert f"{col.key} [{col.unit or '1'}]" in header


# =================================================================================================================
# 5. kkvim_reference is not on the user computing path
# =================================================================================================================

def test_no_kkvim_reference_import_in_kkvim_gui_sources():
    for path in (ROOT / "src" / "kkvim_gui").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = ([a.name for a in node.names] if isinstance(node, ast.Import) else
                     [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
            assert not any(n.split(".")[0] == "kkvim_reference" for n in names), path


def test_no_kkvim_reference_module_at_run_time_of_the_computation():
    """A fresh interpreter does what the computing process does (process.compute) for R3 and R4 and imports the GUI
    application module: kkvim_reference must not be in sys.modules."""
    code = (
        "import sys, multiprocessing, threading\n"
        "from kkvim_gui.backend import process\n"
        "from kkvim_gui.backend.request import RunRequest\n"
        "for N, mode in ((1, 'normative'), (2, 'research_stop')):\n"
        "    r = RunRequest(entered=(), a=1.0, b=1.0, t=0.01, E=2.1e11, nu=0.3, q=1000.0, n_x=24, n_y=24, N=N,\n"
        "                   tol_inf=1e-10, tol_E=1e-8, s_min=3, s_max=100, r4_mode=mode, transform='fft', half_steps=100)\n"
        "    a, b = multiprocessing.Pipe(duplex=False)\n"
        "    got = []\n"
        "    reader = threading.Thread(target=lambda: got.append(a.recv()))\n"
        "    reader.start()  # read while compute writes: a large outcome does not fit the pipe buffer\n"
        "    process.compute(r, b)\n"
        "    reader.join()\n"
        "    assert got[0][0] == 'ok', got\n"
        "try:\n"
        "    import kkvim_gui.app\n"
        "except ImportError:\n"
        "    pass\n"
        "print(sorted(m for m in sys.modules if m.split('.')[0] == 'kkvim_reference'))\n")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]"


# =================================================================================================================
# 6. Real Qt: the main window, the form, the computing process, stop, analysis and export
# =================================================================================================================

@pytest.fixture
def window(qtbot, tmp_path):
    from kkvim_gui.main_window.window import MainWindow
    from kkvim_gui.session import Session

    w = MainWindow(Session(settings=None), protocol_dir=tmp_path / "runs")
    qtbot.addWidget(w)
    w.set_state("setup")
    yield w
    w.runs.shutdown()


def _type_case(view, inp, N, transform="fft", half_steps=None, units=None):
    from kkvim_gui.model.setup_state import N_VALUES

    view.N.setCurrentIndex(N_VALUES.index(N))
    units = units or {}
    texts = {"a": inp["a"], "b": inp["b"], "h": inp["t"], "E": inp["E"], "nu": inp["nu"], "q": inp["q"],
             "n_x": inp["n_x"], "n_y": inp["n_y"]}
    base = {"a": "m", "b": "m", "h": "m", "E": "Pa", "q": "Pa"}
    for key, value in texts.items():
        field = getattr(view, key)
        unit, text = units.get(key, (base.get(key), None))
        if field.unit_combo is not None:
            field.unit_combo.setCurrentText(unit)
        field.edit.setText(text if text is not None else (repr(value) if isinstance(value, float) else str(value)))
    if N != 1:
        view.transform.setCurrentIndex(["fft", "dense"].index(transform))
        if half_steps is not None:
            view.half_steps.edit.setText(str(half_steps))


def _run_and_wait(qtbot, window, timeout=180_000):
    from PySide6.QtCore import Qt

    qtbot.mouseClick(window.setup_view.run_button, Qt.MouseButton.LeftButton)
    run = window.runs.active
    assert run is not None, window.setup_view.check_text.text()
    qtbot.waitUntil(lambda: window.runs.active is None, timeout=timeout)
    return run


@needs_qt
def test_qt_research_switch_is_explicit_and_reset_by_N(window):
    view = window.setup_view
    _type_case(view, {**PLATE, "n_x": 24, "n_y": 24}, 2)
    assert not view.research.isChecked() and not view.research_rule.isEnabled()
    assert window.setup_model.state.r4_mode == R4_NORMATIVE
    view.research.setChecked(True)
    assert window.setup_model.state.r4_mode == R4_RESEARCH_STOP and view.research_rule.isEnabled()
    view.research_rule.setCurrentIndex(1)
    assert window.setup_model.state.r4_mode == R4_RESEARCH_FIXED
    from kkvim_gui.model.setup_state import N_VALUES

    view.N.setCurrentIndex(N_VALUES.index(3))
    assert not view.research.isChecked() and window.setup_model.state.r4_mode == R4_NORMATIVE
    assert not view.research_rule.isEnabled()
    view.N.setCurrentIndex(N_VALUES.index(1))
    view.N.setCurrentIndex(N_VALUES.index(2))
    assert not view.research.isChecked() and to_request(window.setup_model.state).research_mode is False


@needs_qt
@pytest.mark.parametrize("grid, allowed", GRIDS)
def test_qt_dense_is_refused_in_the_form_at_once(window, grid, allowed):
    view = window.setup_view
    _type_case(view, {**PLATE, "n_x": grid[0], "n_y": grid[1]}, 2, transform="dense", half_steps=5)
    assert view.run_button.isEnabled() is allowed
    assert ("transform" in view.shown_errors) is (not allowed)
    if not allowed:
        assert not view.try_run() and window.runs.active is None and not window.runs.runs


@needs_qt
def test_qt_real_run_r4_research_matches_the_control_case_and_exports(window, qtbot, tmp_path):
    c = control()["cases"]["r4_n2_research"]
    view = window.setup_view
    _type_case(view, c["input"], 2, half_steps=100)
    view.research.setChecked(True)  # explicit research choice, stop on stabilization
    run = _run_and_wait(qtbot, window)
    assert run.status == "finished" and run.outcome.status == c["expected_status"] and run.s == c["half_steps_done"]
    assert run.outcome.request.research_mode is True and run.outcome.request.stop_on_stabilization is True
    criterion = window.run_view._criterion(run)
    assert criterion[0] == "warning" and "не сходимость" in criterion[1]
    assert window.run_view._science(run)[0] == "warning"
    if same_environment():
        _, _, w = control_grid(c["files"]["w_csv"])
        assert np.array_equal(run.outcome.w, w)
    assert run.protocol_path is not None and (run.protocol_path / "manifest.json").exists()
    paths = window.export_run(run.run_id, tmp_path / "export")
    assert {"manifest", "history", "field"} <= set(paths)
    from kkvim_gui.analysis.window import AnalysisWindow
    from kkvim_gui.model.setup_state import summary_rows

    analysis = AnalysisWindow(window.session, summary_rows(run.snapshot), run=run)
    qtbot.addWidget(analysis)
    assert analysis.tabs.count() >= 5


@needs_qt
def test_qt_real_run_r3_in_mm_gpa_kpa_matches_the_control_case(window, qtbot):
    c = control()["cases"]["r3_rect_a2b"]
    units = {"a": ("mm", "2000"), "b": ("mm", "1000"), "h": ("mm", "10"), "E": ("GPa", "210"), "q": ("kPa", "1")}
    _type_case(window.setup_view, c["input"], 1, units=units)
    run = _run_and_wait(qtbot, window)
    assert run.status == "finished" and run.outcome.status == c["expected_status"] and run.s == c["half_steps_done"]
    assert (run.outcome.request.a, run.outcome.request.t, run.outcome.request.E) == (2.0, 0.01, 2.1e11)
    assert window.run_view._criterion(run)[0] == "success" and window.run_view._science(run)[0] == "neutral"
    if same_environment():
        _, _, w = control_grid(c["files"]["w_csv"])
        assert np.array_equal(run.outcome.w, w)


@needs_qt
def test_qt_input_rejected_by_kkvim_is_not_computed(window, qtbot):
    c = control()["cases"]["r3_error_ratio"]
    _type_case(window.setup_view, c["input"], 1)
    from PySide6.QtCore import Qt

    qtbot.mouseClick(window.setup_view.run_button, Qt.MouseButton.LeftButton)
    assert window.runs.active is None and not window.runs.runs
    assert "ERROR_INPUT" in window.setup_view.check_text.text()


@needs_qt
def test_qt_stop_terminates_the_real_process(window, qtbot):
    view = window.setup_view
    _type_case(view, {**PLATE, "n_x": 1000, "n_y": 1000}, 3, half_steps=100)
    view.research.setChecked(True)
    view.research_rule.setCurrentIndex(1)  # fixed number of half-steps: a long run
    from PySide6.QtCore import Qt

    qtbot.mouseClick(view.run_button, Qt.MouseButton.LeftButton)
    run = window.runs.active
    assert run is not None
    qtbot.wait(1500)
    assert run.active
    child = run._process._process
    window.runs.stop(run.run_id)  # what "Да" in the confirmation does
    assert run.status == "stopped" and run.outcome is None and not child.is_alive()
    assert not run.has_results and not run.has_diagnostic_field and window.runs.active is None
    assert not view.locked and view.run_button.isEnabled()
