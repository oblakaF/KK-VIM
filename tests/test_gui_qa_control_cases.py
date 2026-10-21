"""The GUI path against the 16 control cases of QA (docs/GUI_BACKEND_ACCEPTANCE.qa-report.md). Qt-free.

The GUI path is the one of the "Выполнить расчёт" button without the window: the values are typed into the
form model (SetupState, units m and Pa), formally checked, converted to a RunRequest, checked by the
library's own input validation and computed by the adapter. The QA script is not used as a backend: only
its stored golden data are read, from ``docs/results/gui_backend`` (after QA is integrated) or from the
folder given by ``KKVIM_QA_GUI_CASES``; without them the module is skipped.

Comparison rules of ``cases.json -> comparison`` (no new tolerance):
- status, half-step count, error flag and the presence of a field exactly; D within 8u (both within 4u);
- same environment as the stored values: the field bitwise; otherwise R4 within the doubled first-order field
  model ``_model`` of tests/cross/test_r4_controller_comparison.py, R3 informative only;
- always: the GUI path equals the direct API call of the same process bitwise (status, D, history, field).
"""

import importlib.util
import json
import os
import platform
from pathlib import Path

import numpy as np
import pytest
import scipy

import kkvim.coupled_run as cr
import kkvim.rectangle_run as rr
from kkvim.rectangle import validated_rectangle_input

from kkvim_gui.backend import kkvim_adapter
from kkvim_gui.backend.request import R4_NORMATIVE, R4_RESEARCH_FIXED, R4_RESEARCH_STOP
from kkvim_gui.model.setup_state import SetupState, to_request, validate

ROOT = Path(__file__).resolve().parents[1]
GOLDEN_DIR = Path(os.environ.get("KKVIM_QA_GUI_CASES", ROOT / "docs" / "results" / "gui_backend"))
if not (GOLDEN_DIR / "cases.json").is_file():
    pytest.skip(f"QA control cases not found in {GOLDEN_DIR} (set KKVIM_QA_GUI_CASES)", allow_module_level=True)
GOLDEN = json.loads((GOLDEN_DIR / "cases.json").read_text(encoding="utf-8"))
CASES = GOLDEN["cases"]
U = 2.0**-53
MODES = {"normative": R4_NORMATIVE, "research": R4_RESEARCH_STOP, "research_fixed": R4_RESEARCH_FIXED}
SOURCE = GOLDEN["source"]
SAME_ENVIRONMENT = (SOURCE["python"] == platform.python_version() and SOURCE["numpy"] == np.__version__
                    and SOURCE["scipy"] == scipy.__version__ and SOURCE["platform"] == platform.platform())


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def form_state(spec):
    p = spec["inputs"]
    state = SetupState()
    for key, name in (("a", "a"), ("b", "b"), ("h", "t"), ("E", "E"), ("nu", "nu"), ("q", "q")):
        state = state.with_text(key, repr(float(p[name])))
    for key, unit in (("a", "m"), ("b", "m"), ("h", "m"), ("E", "Pa"), ("q", "Pa")):
        state = state.with_unit(key, unit)
    state = state.with_text("n_x", str(p["n_x"])).with_text("n_y", str(p["n_y"])).with_choice(N=spec["N"])
    if spec["N"] != 1:
        state = state.with_choice(r4_mode=MODES[spec["mode"]], transform=spec["transform"])
        state = state.with_text("half_steps", str(spec["half_steps"]))
    return state


def gui_path(spec):
    """("form", errors) | ("check", problems) | ("ok", RunOutcome), as after pressing "Выполнить расчёт"."""
    state = form_state(spec)
    errors = validate(state)
    if errors:
        return "form", errors
    request = to_request(state)
    problems = kkvim_adapter.check_request(request)
    if problems:
        return "check", problems
    return "ok", kkvim_adapter.execute(request)


def direct_api(spec):
    """The documented API call of the case (QA report, section 1), in this process."""
    p = spec["inputs"]
    args = (p["a"], p["b"], p["t"], p["E"], p["nu"], p["q"], p["n_x"], p["n_y"])
    if spec["N"] == 1:
        return rr.run_rectangle_physical(*args)
    _, D = validated_rectangle_input(*args)
    mode = spec["mode"]
    return D, cr.run_coupled(p["a"], p["b"], D, p["q"], p["n_x"], p["n_y"], spec["N"], transform=spec["transform"],
                             half_steps=spec["half_steps"], research_mode=mode != "normative",
                             stop_on_stabilization=mode != "research_fixed")


def read_field_csv(path):
    rows = [line.split(",") for line in path.read_text(encoding="utf-8").splitlines()]
    return np.array([[float(v) for v in r[1:]] for r in rows[1:]])


def test_the_golden_set_has_16_cases_for_n_1_2_3():
    assert len(CASES) == 16 and {c["spec"]["N"] for c in CASES.values()} == {1, 2, 3}


@pytest.mark.parametrize("name", sorted(CASES))
def test_gui_path_reproduces_the_qa_control_case(name, capsys):
    stored = CASES[name]
    spec = stored["spec"]
    kind, result = gui_path(spec)
    if stored["expected_status"] == "ERROR_INPUT":
        # the GUI refuses the run: by the form (a, b, h, E > 0) or by the library check (ERROR_INPUT)
        assert kind in ("form", "check"), name
        assert kind == "form" or any("ERROR_INPUT" in p for p in result), result
        assert not stored["field_available"] and stored["expected_status_class"] == "ERROR"
        return
    assert kind == "ok", result
    out = result
    assert out.status == stored["expected_status"]
    assert out.s == stored["half_steps_done"] and out.is_error == stored["is_error"]
    assert out.has_field == stored["field_available"]
    assert abs(out.D - stored["D"]) <= 8 * U * stored["D"]
    cls = stored["expected_status_class"]
    assert (cls == "COMPLETED") == (out.solver == "R3" and out.status == "CONVERGED" and out.stop_kind == "met")
    if spec["N"] != 1:
        assert out.mark == cr.MARK and cr.MARK in out.science_text  # R4 is never COMPLETED
    if not out.has_field:
        return
    w_stored = read_field_csv(GOLDEN_DIR / stored["files"]["w_csv"])
    assert out.w.shape == w_stored.shape == tuple(stored["field"]["shape"])
    diff = float(np.max(np.abs(out.w - w_stored)) / np.max(np.abs(w_stored)))
    if SAME_ENVIRONMENT:
        np.testing.assert_array_equal(out.w, w_stored)
        np.testing.assert_array_equal(out.w_bar, read_field_csv(GOLDEN_DIR / stored["files"]["w_bar_csv"]))
    elif spec["N"] != 1:
        from kkvim_reference.r4_spectral_control import SpectralGrid, run_control

        cmp = _load("r4cmp_for_gui_cases", ROOT / "tests" / "cross" / "test_r4_controller_comparison.py")
        p = spec["inputs"]
        _, run = direct_api(spec)
        qa = run_control(SpectralGrid(1.0, p["b"] / p["a"], p["n_x"], p["n_y"]), spec["N"], run.s)
        assert diff <= 2 * cmp._model(run.steps, qa, run.s, p["n_x"], p["n_y"]), diff
    e = out.extremum()
    assert (e["i"], e["j"]) == (stored["field"]["argmax_abs"]["i"], stored["field"]["argmax_abs"]["j"])
    with capsys.disabled():
        print(f"\n{name}: {out.status}, s = {out.s}, max relative difference to QA {diff:.3e}"
              + (" (bitwise environment)" if SAME_ENVIRONMENT else ""))


@pytest.mark.parametrize("name", sorted(n for n, c in CASES.items() if c["expected_status"] != "ERROR_INPUT"))
def test_gui_path_equals_the_direct_api_call_bitwise(name):
    spec = CASES[name]["spec"]
    _, out = gui_path(spec)
    if spec["N"] == 1:
        direct = direct_api(spec)
        report = rr.halfstep_report(direct)
        assert out.status == direct.status and out.D == direct.D and out.s == direct.run.s
        assert np.array_equal(out.w, direct.run.last_field())
        assert [r["P"] for r in out.history] == [h.P for h in report.halfsteps]
        assert [r["eta"] for r in out.history] == [h.eta for h in report.halfsteps]
        return
    D, run = direct_api(spec)
    assert out.status == run.status and out.reason == run.reason and out.D == D and out.s == run.s
    field = run.field()
    assert (out.w is None) == (field is None)
    if field is not None:
        assert np.array_equal(out.w, field) and np.array_equal(out.w_bar, field * D / (spec["inputs"]["q"]
                                                                                       * spec["inputs"]["a"] ** 4))
    assert [r["P"] for r in out.history] == [step.energy.P for step in run.steps]
    assert [r["eps_E"] for r in out.history] == [step.eps_E for step in run.steps]
