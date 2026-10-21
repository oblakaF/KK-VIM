"""The working path of the GUI with the KK-VIM solver: input -> Run -> computation -> figures -> history -> export.

- the main window starts the real solver in a separate process and is not blocked;
- statuses of the library and PROVISIONAL / NOT_QUALIFIED of R4 are shown unchanged;
- the analysis window draws the solver arrays (field, sections, values, history, energy) and exports them;
- input rejected by kkvim is never computed; a second run cannot start while one is active;
- an error run with a field opens only as diagnostics of an unfinished run;
- the demonstration backend exists only with demo=True and stays marked TEST DATA.
"""

import json

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")
pytest.importorskip("pyqtgraph")

import kkvim.coupled_run as cr  # noqa: E402
import kkvim.rectangle_run as rr  # noqa: E402
from kkvim.linear_solver import ERROR_LINEAR_SOLVE, LinearSolveError  # noqa: E402

from kkvim_gui.analysis.window import SOLVER_TABS, AnalysisWindow  # noqa: E402
from kkvim_gui.app import GuiApplication  # noqa: E402
from kkvim_gui.backend import kkvim_adapter  # noqa: E402
from kkvim_gui.backend.process import CRASH, INPUT, OK  # noqa: E402
from kkvim_gui.model.runs import RunManager  # noqa: E402
from kkvim_gui.model.setup_state import N_VALUES, SetupState, to_request  # noqa: E402
from kkvim_gui.session import Session  # noqa: E402

MARK = cr.MARK


class InlineProcess:
    """Computes in the test process on the first poll (dispatcher logic without spawning)."""

    def __init__(self, request):
        self.request, self.message, self.terminated = request, None, False

    def poll(self):
        if self.message is None:
            try:
                self.message = (OK, kkvim_adapter.execute(self.request))
            except kkvim_adapter.AdapterInputError as exc:
                self.message = (INPUT, str(exc))
        return self.message

    def terminate(self, timeout=2.0):
        self.terminated = True
        return True


class Scripted(InlineProcess):
    """Never finishes (message None) or returns a fixed message."""

    def __init__(self, request, message=None):
        super().__init__(request)
        self.scripted = message

    def poll(self):
        return self.scripted


def small(state=None, N=1, **texts):
    state = state or SetupState()
    texts = {"a": "1", "b": "1,5", "n_x": "24", "n_y": "36", **texts}
    for key, text in texts.items():
        state = state.with_text(key, text)
    return state.with_choice(N=N)


# --- dispatcher -----------------------------------------------------------------------------------------

@pytest.fixture
def manager(qapp, tmp_path):
    return RunManager(process_factory=InlineProcess, protocol_dir=tmp_path / "runs")


def test_solver_run_returns_the_library_result_and_writes_the_protocol(manager, tmp_path):
    state = small()
    run = manager.start(state, to_request(state))
    assert run.active and run.s is None and run.data_origin == "solver"
    manager.tick()
    assert run.status == "finished" and run.has_results and run.outcome.status == "CONVERGED"
    assert run.s == run.outcome.s and run.k == run.outcome.s // 2 and run.stop_criterion == "met"
    assert any("CONVERGED" in t for _, _, t in run.important)
    assert any("ACCEPTED (D-043)" in t for _, _, t in run.important)
    names = sorted(p.name for p in run.protocol_path.iterdir())
    assert names == ["history.csv", "manifest.json"]          # the protocol has no field
    data = json.loads((run.protocol_path / "manifest.json").read_text(encoding="utf-8"))
    assert data["run"]["execution_status"] == "finished" and data["data_origin"] == "solver"
    assert any("Расчёт запущен" in e["text"] for e in data["log"])
    assert any("→ 1.0 m" in t for _, _, t in run.detailed)    # unit conversion is in the protocol


def test_r4_run_is_marked_provisional_not_qualified_everywhere(manager):
    state = small(N=3)
    run = manager.start(state, to_request(state))
    assert any(MARK in t for _, _, t in run.important)       # already at the start
    manager.tick()
    assert run.outcome.status == cr.PROVISIONAL_FIELD_STABILIZED and run.outcome.mark == MARK
    assert any("не сходимость" in t for _, _, t in run.important)
    data = json.loads((run.protocol_path / "manifest.json").read_text(encoding="utf-8"))
    assert data["marks"] == [MARK] and data["statuses"]["mark"] == MARK
    assert (run.protocol_path / "history.csv").read_text(encoding="utf-8").startswith(f"# {MARK}\n")


def test_only_one_run_and_stop_terminates_the_process(qapp):
    manager = RunManager(process_factory=Scripted)
    state = small()
    run = manager.start(state, to_request(state))
    manager.tick()
    assert run.active
    with pytest.raises(RuntimeError):
        manager.start(state, to_request(state))
    manager.stop(run.run_id)
    assert run._process.terminated and run.status == "stopped" and not run.has_results
    assert manager.active is None


@pytest.mark.parametrize(("message", "text"), [
    ((CRASH, "Traceback (most recent call last):\nValueError: broken"), "Сбой вычислительного процесса: ValueError"),
    ((INPUT, "Ввод отклонён решателем (ERROR_INPUT): |nu| must be < 1"), "Ввод отклонён до расчёта"),
])
def test_crash_and_rejected_input_are_errors(qapp, message, text):
    manager = RunManager(process_factory=lambda request: Scripted(request, message))
    state = small()
    run = manager.start(state, to_request(state))
    manager.tick()
    assert run.status == "error" and not run.has_results and run.outcome is None
    assert any(text in t for _, _, t in run.important)


# --- main window and analysis window with the real process ---------------------------------------------

@pytest.fixture
def gui(qtbot, qapp, tmp_path):
    session = Session(settings=None)
    session.apply_theme()
    g = GuiApplication(session, protocol_dir=tmp_path / "runs")
    qtbot.addWidget(g.main_window)
    g.main_window.show()
    g.main_window.set_state("setup")
    yield g
    for w in list(g.analysis_windows):
        w.close()
    g.main_window.runs.shutdown()
    g.help.close()


def run_from_form(qtbot, gui, state):
    w = gui.main_window
    w.setup_model._set(state)
    assert w.setup_view.try_run() is True
    run = w.runs.runs[-1]
    assert w.state == "running" and run.active
    assert not w.setup_view.run_button.isEnabled()          # no second start during the run
    qtbot.waitUntil(lambda: not run.active, timeout=120_000)
    assert w.state == "finished"
    return run


def test_r4_from_the_form_to_figures_history_and_export(qtbot, gui, tmp_path):
    w = gui.main_window
    run = run_from_form(qtbot, gui, small(N=2))
    o = run.outcome
    assert run.status == "finished" and o.status == cr.PROVISIONAL_FIELD_STABILIZED and o.mark == MARK
    rv = w.run_view
    assert MARK in rv.status_accept.text() and cr.PROVISIONAL_FIELD_STABILIZED in rv.status_stop.text()
    assert rv.details_tag.text() == MARK and MARK in rv.summary_text.text()
    assert rv.analysis_button.isVisibleTo(rv) and rv.analysis_button.isEnabled()
    w0, wb0 = o.w.copy(), o.w_bar.copy()
    with qtbot.waitSignal(w.analysisRequested):
        rv.analysis_button.click()
    a = gui.analysis_windows[-1]
    assert a.outcome is o and [a.tabs.tabText(i) for i in range(a.tabs.count())] == list(SOLVER_TABS)
    assert not a.tabs.isTabEnabled(SOLVER_TABS.index("Сравнение"))
    assert a.header.badge.isVisibleTo(a.header) and "PROVISIONAL" in a.header.badge.text()
    assert MARK in a.status_accept.text() and MARK in a.fields.figure.mark.text()
    np.testing.assert_array_equal(a.fields.figure.image.image, o.w * 1000.0)   # the solver array, in mm
    image = a.fields.figure.image
    placed = image.mapRectToParent(image.boundingRect())       # nodes centred on pixels, coordinates in m
    hx, hy = o.x[1] - o.x[0], o.y[1] - o.y[0]
    assert placed.left() == pytest.approx(-hx / 2) and placed.right() == pytest.approx(o.x[-1] + hx / 2)
    assert placed.top() == pytest.approx(-hy / 2) and placed.bottom() == pytest.approx(o.y[-1] + hy / 2)
    assert rv.status_stop.property("status") != "success"      # a stabilization of R4 is not shown as success
    a.fields.quantity.set_current("w_bar")
    np.testing.assert_array_equal(a.fields.figure.image.image, o.w_bar)
    e = o.extremum()
    xs, ys = a.sections.curve_x.getData()
    np.testing.assert_array_equal(xs, o.x) and np.testing.assert_array_equal(ys, o.w[:, e["j"]] * 1000.0)
    assert a.values.model.rowCount() == o.w.size
    assert a.history.model.rowCount() == len(o.history) == o.s
    assert a.energy.curve.getData()[1].tolist() == [r["P"] for r in o.history]
    paths = a.export_all(tmp_path / "export")
    data = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert data["statuses"]["solver_status"] == o.status and data["marks"] == [MARK]
    assert sorted(p.name for p in paths["manifest"].parent.iterdir()) == ["field.csv", "history.csv", "manifest.json"]
    for theme in ("dark", "light"):
        gui.session.set_theme(theme, persist=False)
        assert not a.grab().isNull()
    gui.session.set_figure_language("en")
    assert a.fields.figure.title.text().startswith("Dimensionless deflection")
    gui.session.set_figure_language("ru")
    np.testing.assert_array_equal(o.w, w0) and np.testing.assert_array_equal(o.w_bar, wb0)  # never changed


def test_r3_from_the_form_is_the_accepted_solver_without_a_provisional_mark(qtbot, gui):
    run = run_from_form(qtbot, gui, small())
    o = run.outcome
    assert o.status == "CONVERGED" and o.mark is None and o.solver == "R3"
    rv = gui.main_window.run_view
    assert "ACCEPTED (D-043)" in rv.status_accept.text() and "PROVISIONAL" not in rv.status_accept.text()
    a = gui.open_analysis()
    assert not a.header.badge.isVisibleTo(a.header) and a.fields.figure.mark.text() == ""
    assert not a.fields.figure.mark.isVisibleTo(a.fields.figure)
    direct = rr.run_rectangle_physical(1.0, 1.5, 0.01, 2.1e11, 0.3, 5000.0, 24, 36)
    np.testing.assert_array_equal(o.w, direct.run.last_field())   # same process, same result


def test_input_rejected_by_kkvim_is_not_computed(qtbot, gui):
    w = gui.main_window
    w.setup_model._set(small(nu="1"))
    assert w.setup_view.try_run() is True                 # the form accepts any number for ν ...
    assert w.runs.runs == [] and w.state == "setup"        # ... but the library check refuses the run
    assert "ERROR_INPUT" in w.setup_view.check_text.text()
    w.setup_model._set(small(N=2, **{"n_x": "48", "n_y": "48"}).with_choice(transform="dense"))
    w.setup_view.try_run()
    assert w.runs.runs == [] and "D-048.6" in w.setup_view.check_text.text()


def test_cccc_and_n_4_6_are_shown_as_not_implemented(qtbot, gui):
    sv = gui.main_window.setup_view
    assert "не реализовано" in sv.supports.buttons["CCCC"].text()
    model = sv.N.model()
    assert [model.item(i).isEnabled() for i in range(len(N_VALUES))] == [n <= 3 for n in N_VALUES]
    sv.supports.set_current("CCCC")
    assert sv.try_run() is False and gui.main_window.runs.runs == []
    assert sv.errors_supports.isVisibleTo(sv) and "CCCC" in sv.errors_supports.text()


def test_r4_parameters_are_shown_only_for_n_2_3(qtbot, gui):
    sv = gui.main_window.setup_view
    assert sv.tol_E.isVisibleTo(sv) and not sv.half_steps.isVisibleTo(sv)
    sv.N.setCurrentIndex(2)
    assert not sv.tol_E.isVisibleTo(sv) and sv.half_steps.isVisibleTo(sv) and sv.r4_mode.isVisibleTo(sv)
    assert sv.r4_note.isVisibleTo(sv) and MARK in sv.r4_note.text.text()
    assert gui.main_window.setup_model.state.r4_mode == "research_stop"


def test_error_with_a_field_opens_only_as_diagnostics(qtbot, qapp, tmp_path, monkeypatch):
    original = rr.next_halfstep

    def failing(record):
        if record.s == 3:
            raise LinearSolveError(ERROR_LINEAR_SOLVE, "injected")
        return original(record)

    monkeypatch.setattr(rr, "next_halfstep", failing)
    session = Session(settings=None)
    g = GuiApplication(session)
    g.main_window.runs.process_factory = InlineProcess
    qtbot.addWidget(g.main_window)
    w = g.main_window
    w.setup_model._set(small())
    w.setup_view.try_run()
    w.runs.tick()
    run = w.runs.runs[0]
    assert run.status == "error" and run.outcome.diagnostic_only and not run.has_results
    assert w.run_view.analysis_button.text().startswith("Открыть диагностику")
    a = g.open_analysis()
    assert "ДИАГНОСТИКА" in a.header.badge.text() and "не завершён" in a.status_run.text()
    assert a.fields.figure.mark.text() in ("DIAGNOSTIC · RUN NOT COMPLETED", "ДИАГНОСТИКА · РАСЧЁТ НЕ ЗАВЕРШЁН")
    a.close()
    g.help.close()


def test_demo_backend_only_on_request_and_marked_test_data(qtbot, qapp):
    session = Session(settings=None)
    real = GuiApplication(session)
    demo = GuiApplication(session, demo=True)
    for g in (real, demo):
        qtbot.addWidget(g.main_window)
    assert not real.main_window.header.badge.isVisibleTo(real.main_window.header)
    assert real.main_window.start_view.table.rowCount() == 0          # no invented projects without --demo
    assert demo.main_window.start_view.table.rowCount() > 0
    assert demo.main_window.header.badge.text().startswith("TEST DATA")
    assert AnalysisWindow(session).outcome is None           # without a solver run: placeholders only
    for g in (real, demo):
        g.help.close()
