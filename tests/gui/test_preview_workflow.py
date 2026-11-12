"""Functional preview of the GUI (SSSS, N = 1): input -> calculation -> visualization -> save, through the windows.

- the preview offers only Kirchhoff–Love / SSSS / N = 1; N = 2-6, CCCC and other theories are shown as unavailable
  and cannot be run;
- the field of a run from the form is the field of a direct call of the accepted R3 solver
  (``kkvim.rectangle_run.run_rectangle_physical``), bit for bit, for square and rectangular plates;
- the 3D view draws the stored field with a factor that is shown; values never change;
- the saved package (manifest.json, history.csv, field.csv, PNG figures, report.html) holds the displayed values;
- invalid input and solver failures are shown and never computed or saved as results.
Bounds here are exact equalities of copies, not scientific tolerances.
"""

import csv
import json
import math

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")
pytest.importorskip("pyqtgraph")

from PySide6.QtCore import Qt  # noqa: E402

import kkvim.rectangle_run as rr  # noqa: E402

from kkvim_gui.analysis import surface  # noqa: E402
from kkvim_gui.analysis.window import AnalysisWindow  # noqa: E402
from kkvim_gui.app import GuiApplication, parse_args  # noqa: E402
from kkvim_gui.backend import kkvim_adapter, report  # noqa: E402
from kkvim_gui.backend.process import CRASH, OK  # noqa: E402
from kkvim_gui.backend.request import RunRequest  # noqa: E402
from kkvim_gui.main_window.window import PREVIEW_N, MainWindow  # noqa: E402
from kkvim_gui.model.setup_state import N_VALUES, summary_rows  # noqa: E402
from kkvim_gui.session import Session  # noqa: E402

# (texts and units as typed, SI values for the direct call)
SQUARE = ({"a": ("1", "m"), "b": ("1", "m"), "h": ("10", "mm"), "E": ("210", "GPa"), "nu": ("0,3", ""),
           "q": ("1", "kPa"), "n_x": ("24", ""), "n_y": ("24", "")},
          dict(a=1.0, b=1.0, t=0.01, E=2.1e11, nu=0.3, q=1000.0, n_x=24, n_y=24))
RECTANGLE = ({"a": ("2000", "mm"), "b": ("1000", "mm"), "h": ("1", "cm"), "E": ("70", "GPa"), "nu": ("0.33", ""),
              "q": ("5", "kPa"), "n_x": ("40", ""), "n_y": ("20", "")},
             dict(a=2.0, b=1.0, t=0.01, E=7e10, nu=0.33, q=5000.0, n_x=40, n_y=20))
NARROW = ({"a": ("1", "m"), "b": ("5", "m"), "h": ("20", "mm"), "E": ("200", "GPa"), "nu": ("0,25", ""),
           "q": ("-2", "kPa"), "n_x": ("12", ""), "n_y": ("60", "")},
          dict(a=1.0, b=5.0, t=0.02, E=2e11, nu=0.25, q=-2000.0, n_x=12, n_y=60))


class InlineProcess:
    """Computes in the test process on the first poll (the dispatcher path without spawning)."""

    def __init__(self, request):
        self.request, self.message = request, None

    def poll(self):
        if self.message is None:
            self.message = (OK, kkvim_adapter.execute(self.request))
        return self.message

    def terminate(self, timeout=2.0):
        return True


class Crashing(InlineProcess):
    def poll(self):
        return (CRASH, "Traceback (most recent call last):\nMemoryError: simulated failure of the computation")


@pytest.fixture
def preview(qtbot, tmp_path):
    window = MainWindow(Session(settings=None), protocol_dir=tmp_path / "runs", process_factory=InlineProcess,
                        preview=True)
    qtbot.addWidget(window)
    window.set_state("setup")
    yield window
    window.runs.shutdown()


def type_case(view, texts):
    for key, (text, unit) in texts.items():
        field = getattr(view, key)
        if field.unit_combo is not None:
            field.unit_combo.setCurrentText(unit)
        field.edit.setText(text)


def run_case(qtbot, window, texts):
    type_case(window.setup_view, texts)
    qtbot.mouseClick(window.setup_view.run_button, Qt.MouseButton.LeftButton)
    run = window.runs.active
    assert run is not None, window.setup_view.check_text.text()
    qtbot.waitUntil(lambda: window.runs.active is None, timeout=120_000)
    return run


def direct(si):
    return rr.run_rectangle_physical(si["a"], si["b"], si["t"], si["E"], si["nu"], si["q"], si["n_x"], si["n_y"])


def read_field_csv(path):
    with open(path, encoding="utf-8") as f:
        lines = [line for line in f if not line.startswith("#")]
    rows = list(csv.DictReader(lines))
    return rows


# --- pure parts ------------------------------------------------------------------------------------------

def test_auto_factor_is_a_1_2_5_value_not_above_the_target_height():
    x, y = np.linspace(0, 2.0, 5), np.linspace(0, 1.0, 3)
    w = np.zeros((5, 3))
    assert surface.auto_factor(x, y, w) == 1.0
    w[2, 1] = 3.1e-4
    f = surface.auto_factor(x, y, w)
    raw = surface.AUTO_HEIGHT * 2.0 / 3.1e-4
    assert f <= raw < 2.5 * f and f / 10 ** math.floor(math.log10(f)) in (1.0, 2.0, 5.0)
    w[2, 1] = 10.0  # already visible: never shrunk below the true scale
    assert surface.auto_factor(x, y, w) == 1.0


def test_surface_lines_are_the_projected_nodes_of_the_stored_field():
    x, y = np.linspace(0, 1, 4), np.linspace(0, 2, 3)
    w = np.arange(12.0).reshape(4, 3) * 1e-3
    u, v = surface.surface_polylines(x, y, w, 50.0)
    finite = ~np.isnan(u)
    assert finite.sum() == 2 * w.size  # every node on one line of each direction
    pu, pv = surface.project(x[:, None] + 0 * y, 0 * x[:, None] + y, -50.0 * w)
    got = set(zip(np.round(u[finite], 12), np.round(v[finite], 12)))
    assert got == set(zip(np.round(pu.ravel(), 12), np.round(pv.ravel(), 12)))
    assert surface.line_indices(1001)[0] == 0 and surface.line_indices(1001)[-1] == 1000
    assert len(surface.line_indices(1001)) == surface.MAX_LINES
    assert "только на изображении" in surface.exaggeration_text(100.0)
    assert "истинный" in surface.exaggeration_text(1.0)


def test_report_and_package_without_figures_hold_the_outcome_values(tmp_path):
    si = SQUARE[1]
    request = RunRequest(entered=(("a", "1", "m"),), **si, N=1, tol_inf=kkvim_adapter.DEFAULTS["tol_inf"],
                         tol_E=kkvim_adapter.DEFAULTS["tol_E"], s_min=kkvim_adapter.DEFAULTS["s_min"],
                         s_max=kkvim_adapter.DEFAULTS["s_max"])
    o = kkvim_adapter.execute(request)
    paths = report.write_package(o, tmp_path / "pkg", {"run_id": 1})
    assert set(paths) == {"manifest", "history", "field", "report"}
    data = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    e, c = o.extremum(), report.centre_node(o)
    assert c is not None and (c["i"], c["j"]) == (12, 12) and c["w"] == float(o.w[12, 12])
    assert data["report_summary"]["w_max_abs_node"]["w"] == e["w"]
    assert data["results"]["extremum_by_nodes"]["w"]["value"] == e["w"]
    assert data["files"]["report"] == "report.html"
    text = paths["report"].read_text(encoding="utf-8")
    for value in (e["w"], e["w_bar"], c["w"], o.D, si["E"]):
        assert repr(float(value)) in text
    assert o.status in text and report.PREVIEW_NOTICE in text and "0,2 ≤ b/a ≤ 5" in text


# --- the preview offers N = 1 only --------------------------------------------------------------------------

def test_preview_offers_only_kirchhoff_love_ssss_and_n1(preview):
    view = preview.setup_view
    model = view.N.model()
    enabled = [n for i, n in enumerate(N_VALUES) if model.item(i).isEnabled()]
    assert enabled == list(PREVIEW_N) == [1]
    for n in (2, 3, 4, 5, 6):
        assert "недоступно" in view.N.itemText(N_VALUES.index(n))
    theories = view.theory.model()
    assert theories.item(0).isEnabled() and not any(theories.item(i).isEnabled() for i in (1, 2))
    assert "недоступно" in view.supports.buttons["CCCC"].text() and not view.supports.buttons["CCCC"].isEnabled()
    assert view.preview_note is not None
    assert any("N = 1" in t and "недоступные" in t for t in label_texts(view.preview_note))
    assert not preview.header.badge.isHidden() and "ПРЕДВАРИТЕЛЬНАЯ" in preview.header.badge.text()


def test_preview_refuses_n2_and_cccc_even_if_set_programmatically(preview):
    view = preview.setup_view
    view.N.setCurrentIndex(N_VALUES.index(2))  # not possible with the mouse: the item is disabled
    assert preview._start_run() is None and preview.runs.active is None and not preview.runs.runs
    assert any("недоступно" in t for t in label_texts(view))
    view.N.setCurrentIndex(0)
    view.supports.set_current("CCCC")  # not possible with the mouse: the button is disabled in the preview
    view.try_run()
    assert not preview.runs.runs and any("недоступно" in t for t in label_texts(view))


def test_the_preview_window_is_restricted_to_n1():
    """``preview=True`` remains the restricted window (SSSS, N = 1); the application entry runs every route (P1)."""
    args, _ = parse_args(["--smoke-test", "x"])
    assert args.smoke_test == "x"
    gui = GuiApplication(Session(settings=None), preview=True)
    try:
        assert gui.main_window.preview and gui.main_window.setup_view.n_selectable == PREVIEW_N
    finally:
        gui.main_window.close()


def label_texts(widget):
    from PySide6.QtWidgets import QLabel

    return [w.text() for w in widget.findChildren(QLabel) if not w.isHidden()]


# --- the workflow against the accepted solver -----------------------------------------------------------------

@pytest.mark.parametrize("case", [SQUARE, RECTANGLE, NARROW], ids=["square", "rect_2x1_mm", "rect_1x5_negative_q"])
def test_workflow_matches_the_accepted_solver_and_saves_what_is_shown(qtbot, preview, tmp_path, case):
    texts, si = case
    run = run_case(qtbot, preview, texts)
    o = run.outcome
    ref = direct(si)
    assert run.status == "finished" and o.status == ref.status == "CONVERGED" and o.solver == "R3"
    assert (o.request.a, o.request.b, o.request.t, o.request.E, o.request.q) == (
        si["a"], si["b"], si["t"], si["E"], si["q"])
    assert np.array_equal(o.w, ref.run.last_field()) and np.array_equal(o.w_bar, ref.dimensionless_field())
    assert o.s == ref.run.s and o.D == ref.D

    window = AnalysisWindow(preview.session, summary_rows(run.snapshot), run=run)
    qtbot.addWidget(window)
    fields = window.fields
    assert np.array_equal(fields.figure.image.image, o.w * 1000.0)  # the colour field is the solver array in mm
    assert len(fields.figure.isocurves) == fields.figure.ISOLINES
    levels = [c.level for c in fields.figure.isocurves]
    assert min(levels) > float(o.w.min()) * 1000.0 and max(levels) < float(o.w.max()) * 1000.0
    fields.isolines.setChecked(False)
    assert not fields.figure.isocurves
    fields.isolines.setChecked(True)

    fields.set_view("surface")
    assert fields.stack.currentWidget() is fields.surface
    assert not fields.surface_controls[0].isHidden() and all(w.isHidden() for w in fields.map_controls)
    auto = fields.surface.factor
    assert auto == surface.auto_factor(o.x, o.y, o.w) and "только на изображении" in fields.surface.scale_note.text()
    assert fields.surface.title.text() and fields.surface.mark.isHidden()  # no empty mark; R3 carries none
    fields.factor.setCurrentIndex(fields.factor.findData(1.0))
    assert fields.surface.factor == 1.0 and "истинный" in fields.surface.scale_note.text()
    fields.factor.setCurrentIndex(fields.factor.findData(1e3))
    u, v = fields.surface.mesh.getData()
    pu, pv = surface.surface_polylines(o.x, o.y, o.w, 1e3)
    assert np.allclose(u[~np.isnan(u)], pu[~np.isnan(pu)], rtol=0, atol=1e-12) and np.array_equal(
        np.isnan(v), np.isnan(pv))
    e = o.extremum()
    assert np.sign(e["w"]) == np.sign(si["q"])
    assert f"{e['i']}, j = {e['j']}" in " ".join(label_texts(window.fields))

    w_before = o.w.copy()
    paths = window.save_package(tmp_path / "saved")
    assert np.array_equal(o.w, w_before)  # the drawing factor never touches the values
    assert set(paths) == {"manifest", "history", "field", "field_png", "surface_png", "report"}
    for key in ("field_png", "surface_png"):
        assert paths[key].stat().st_size > 10_000 and paths[key].read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    data = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert data["statuses"]["solver_status"] == o.status and data["input"]["N"] == 1
    assert data["visualization"]["exaggeration_factor"] == 1e3 and data["visualization"]["values_scaled"] is False
    assert data["results"]["extremum_by_nodes"]["w"]["value"] == e["w"]
    assert [d["text"] for d in data["input"]["entered"][:6]] == [texts[k][0] for k in ("a", "b", "h", "E", "nu", "q")]
    rows = read_field_csv(paths["field"])
    assert len(rows) == o.w.size
    w_csv = np.array([float(r["w [m]"]) for r in rows]).reshape(o.w.shape)
    assert np.array_equal(w_csv, o.w)  # saved = displayed, bit for bit
    text = paths["report"].read_text(encoding="utf-8")
    assert repr(e["w"]) in text and repr(e["w_bar"]) in text and "CONVERGED" in text
    assert 'src="field.png"' in text and 'src="surface.png"' in text
    assert surface.exaggeration_text(1e3) in text


def test_run_view_saves_the_package_of_the_selected_run(qtbot, preview, tmp_path):
    run = run_case(qtbot, preview, SQUARE[0])
    paths = preview.save_package(run.run_id, tmp_path / "from_run_view")
    assert {"field_png", "surface_png", "report"} <= set(paths)
    data = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert data["visualization"]["exaggeration_factor"] == surface.auto_factor(run.outcome.x, run.outcome.y,
                                                                                 run.outcome.w)


# --- invalid input and failures -------------------------------------------------------------------------------

@pytest.mark.parametrize("key, text, expected", [
    ("a", "abc", None),
    ("h", "0", "больше нуля"),
    ("n_x", "2,5", "целое"),
])
def test_invalid_form_input_is_shown_and_not_computed(preview, key, text, expected):
    view = preview.setup_view
    getattr(view, key).edit.setText(text)
    assert not view.try_run() and not preview.runs.runs
    shown = view.errors[key].text()
    assert shown and (expected is None or expected in shown)


@pytest.mark.parametrize("texts", [{"nu": ("1,2", "")}, {"q": ("0", "kPa")}, {"b": ("6", "m"), "a": ("1", "m")}],
                         ids=["abs_nu_not_below_1", "zero_load", "ratio_above_5"])
def test_input_rejected_by_the_solver_is_shown_and_not_computed(preview, texts):
    view = preview.setup_view
    type_case(view, texts)
    assert view.try_run()  # the form is formally correct
    assert preview.runs.active is None and not preview.runs.runs
    assert any("ERROR_INPUT" in t for t in label_texts(view))


def test_solver_failure_is_visible_and_saves_no_result(qtbot, tmp_path):
    window = MainWindow(Session(settings=None), protocol_dir=tmp_path / "runs", process_factory=Crashing,
                        preview=True)
    qtbot.addWidget(window)
    window.set_state("setup")
    try:
        qtbot.mouseClick(window.setup_view.run_button, Qt.MouseButton.LeftButton)
        qtbot.waitUntil(lambda: window.runs.active is None, timeout=10_000)
        run = window.runs.runs[-1]
        assert run.status == "error" and run.outcome is None and not run.has_results
        assert window.run_view.status_run.property("status") == "error"
        assert "MemoryError" in window.run_view.status_run.text() or any(
            "MemoryError" in t for t in label_texts(window.run_view))
        assert not window.run_view.analysis_button.isEnabled() or window.run_view.analysis_button.isHidden()
        assert window.run_view.export_button.isHidden()
    finally:
        window.runs.shutdown()
