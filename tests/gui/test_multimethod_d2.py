"""CCCC with the independent D2 through the windows (GUI-MULTIMETHOD-CCCC-D2-01).

- setup: the clear choice «МКР — двумерный (D2)» for CCCC only (on by default; locked during a run); SSSS shows the
  automatic Navier line and never offers D2; the choice is in the configuration and the method summary;
- dispatcher: D2 runs after the KK-VIM process of the same run (never at the same time), with the physical input and
  grid only; «Стоп» during D2 stops D2 only and keeps the KK-VIM result; crash, unavailable worker and a missing factory
  are stated; a load series runs D2 at every q and advances only after it; no process is left;
- analysis: the «МКР D2» tab, the D2 row of «Сравнение с эталоном» next to the saved rows, the package with D2 files;
  a D2 that arrives after the window opened rebuilds the comparison; the series graph has the D2 curve;
- O-1 in the dispatcher: a killed computing process ends the run with «Ошибка», the series goes on by its policy;
- O-3, O-4: the stopping rule texts; the highlighted index entry is always the article shown.
"""

import json
import time
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")
pytest.importorskip("pyqtgraph")

from kkvim_gui.analysis.window import AnalysisWindow  # noqa: E402
from kkvim_gui.app import GuiApplication  # noqa: E402
from kkvim_gui.backend import d2_job, kkvim_adapter  # noqa: E402
from kkvim_gui.backend.process import INPUT, OK, SolverProcess  # noqa: E402
from kkvim_gui.help.articles import get  # noqa: E402
from kkvim_gui.main_window.window import MainWindow  # noqa: E402
from kkvim_gui.model.series import series_data  # noqa: E402
from kkvim_gui.model.setup_state import SetupState, method_text, to_config  # noqa: E402
from kkvim_gui.session import Session  # noqa: E402


class InlineProcess:
    def __init__(self, request):
        self.request, self.message, self.terminated = request, None, False
        self.started, self.received = time.perf_counter(), None

    def poll(self):
        if self.message is None:
            try:
                self.message = (OK, kkvim_adapter.execute(self.request))
            except kkvim_adapter.AdapterInputError as exc:
                self.message = (INPUT, str(exc))
            self.received = time.perf_counter()
        return self.message

    def terminate(self, timeout=2.0):
        self.terminated = True
        return True


class InlineD2:
    """The D2 worker contract evaluated in the test process after ``polls`` polls (the real process: test_gui_d2.py)."""

    created = []

    def __init__(self, inputs, polls=1):
        self.inputs, self.polls, self.terminated = inputs, polls, False
        InlineD2.created.append(self)

    def poll(self):
        if self.terminated:
            return None
        self.polls -= 1
        if self.polls > 0:
            return None
        from kkvim_d2_worker import compute

        return d2_job.parse_output(json.dumps(compute(self.inputs)), self.inputs)

    def terminate(self, timeout=5.0):
        self.terminated = True
        return True


class SlowD2(InlineD2):
    def __init__(self, inputs):
        super().__init__(inputs, polls=10 ** 9)


def cccc(model, n="16", q="1000", N=1, s=8):
    model.set_choice(supports="CCCC", N=N)
    model.set_choice(cccc_half_steps=s)
    for key, text in (("a", "1"), ("b", "1"), ("h", "0,01"), ("E", "210"), ("nu", "0,3"), ("q", q), ("n_x", n),
                      ("n_y", n)):
        model.set_text(key, text)
    model.set_unit("a", "m")
    model.set_unit("b", "m")
    model.set_unit("h", "m")
    model.set_unit("E", "GPa")
    model.set_unit("q", "Pa")


@pytest.fixture
def main(qtbot, tmp_path):
    InlineD2.created = []
    w = MainWindow(Session(settings=None), protocol_dir=tmp_path / "runs", process_factory=InlineProcess)
    w.runs.d2_factory = InlineD2
    w.runs.navier_factory = None
    qtbot.addWidget(w)
    w.set_state("setup")
    return w


def finish(main, qtbot, timeout=60000):
    qtbot.waitUntil(lambda: main.runs.active is None and not main.runs.references_pending
                    and not any(s.active for s in main.runs.series), timeout=timeout)


# --- setup ------------------------------------------------------------------------------------------------------------

def test_d2_is_a_clear_choice_for_cccc_only(main):
    sv, m = main.setup_view, main.setup_model
    assert SetupState().with_d2 is True
    m.set_choice(supports="SSSS", N=1)
    assert not sv.d2.isVisibleTo(sv) and sv.ssss_independent.isVisibleTo(sv)
    assert "Навье рассчитывается автоматически" in sv.ssss_independent.text() and "не применяется" in sv.ssss_independent.text()
    assert not m.state.d2  # never for SSSS, whatever the switch
    m.set_choice(supports="CCCC")
    assert sv.d2.isVisibleTo(sv) and sv.d2.isChecked() and not sv.ssss_independent.isVisibleTo(sv)
    assert sv.d2.text().startswith("МКР — двумерный (D2)") and sv.d2.property("helpTopic") == "verification.d2"
    assert m.state.d2 and "МКР — двумерный (D2)" in method_text(m.state)
    assert to_config(m.state)["method"]["independent_d2"] is True
    sv.d2.setChecked(False)
    assert m.state.with_d2 is False and not m.state.d2 and to_config(m.state)["method"]["independent_d2"] is False
    m.set_choice(N=2)  # a change of N keeps the user's choice of the independent method
    assert m.state.with_d2 is False and not sv.d2.isChecked()
    with pytest.raises(ValueError):
        m.state.with_choice(with_d2="yes")


def test_preview_never_offers_d2(qtbot):
    w = MainWindow(Session(settings=None), preview=True, process_factory=InlineProcess)
    qtbot.addWidget(w)
    w.set_state("setup")
    assert not w.setup_view.d2.isVisibleTo(w.setup_view)


# --- dispatcher -------------------------------------------------------------------------------------------------------

def test_d2_runs_after_kkvim_with_the_physical_input_only(main, qtbot):
    cccc(main.setup_model)
    InlineD2.created = []
    main.runs.d2_factory = lambda inputs: InlineD2(inputs, polls=3)
    run = main._start_run()
    assert run.d2_requested and run.d2 is None and run.outcome is None  # KK-VIM first: D2 not started yet
    main.runs.tick()  # the KK-VIM process answers
    assert run.outcome is not None and run.active and run.d2["state"] == "pending" and len(InlineD2.created) == 1
    assert "МКР — двумерный (D2)" in run.stage
    sent = InlineD2.created[0].inputs
    assert sent == {"contract": d2_job.CONTRACT, "supports": "CCCC", "a": 1.0, "b": 1.0, "t": 0.01, "E": 210e9, "nu": 0.3,
                    "q": 1000.0, "n_x": 16, "n_y": 16}  # no field, no KK-VIM value
    finish(main, qtbot)
    assert run.status == "finished" and run.d2["state"] == "ok" and run.has_results
    assert set(run.timing) == {"kkvim_import_s", "kkvim_solver_s", "kkvim_wall_s"}
    assert run.meta(0.0)["timing_s"] == run.timing
    texts = [t for _, _, t in run.important]
    assert any("МКР — двумерный (D2): w в центре" in t for t in texts)
    assert "МКР D2" in run.params_text()
    assert "w в центре" in main.run_view.d2_text(run)


def test_without_the_choice_and_for_ssss_there_is_no_d2(main, qtbot):
    cccc(main.setup_model)
    main.setup_model.set_choice(with_d2=False)
    run = main._start_run()
    finish(main, qtbot)
    assert not run.d2_requested and run.d2 is None and not InlineD2.created
    assert main.run_view.d2_text(run) == "не выбран"
    main.set_state("setup")
    main.setup_model.set_choice(supports="SSSS", N=1, with_d2=True)
    for key, text in (("n_x", "16"), ("n_y", "8"), ("a", "2")):
        main.setup_model.set_text(key, text)
    run = main._start_run()
    finish(main, qtbot)
    assert not run.d2_requested and run.d2 is None and not InlineD2.created


def test_stop_during_d2_stops_d2_only_and_keeps_kkvim(main, qtbot, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    asked = []
    monkeypatch.setattr(QMessageBox, "question",
                        lambda *args, **kw: asked.append(args[2]) or QMessageBox.StandardButton.Yes)
    cccc(main.setup_model)
    main.runs.d2_factory = SlowD2
    run = main._start_run()
    main.runs.tick()
    assert run._d2_job is not None and run.active
    job = run._d2_job
    main.run_view.select(run.run_id)
    main.run_view.stop_button.click()
    assert asked and "Остановить расчёт МКР D2? Результат КК-ВИМ этого запуска сохраняется" in asked[0]
    assert job.terminated and run._d2_job is None
    assert run.status == "finished" and run.outcome is not None and run.has_results  # KK-VIM result kept
    assert run.d2["state"] == "cancelled"
    assert any("остановлен пользователем; процесс D2 завершён; результат КК-ВИМ сохранён" in t for _, _, t in run.important)


def test_unavailable_or_missing_d2_is_stated(main, qtbot):
    cccc(main.setup_model)

    def broken(inputs):
        raise OSError("нет файла")

    main.runs.d2_factory = broken
    run = main._start_run()
    finish(main, qtbot)
    assert run.status == "finished" and run.d2["state"] == "failed" and "нет файла" in run.d2["reason"]
    main.set_state("setup")
    main.runs.d2_factory = None
    run = main._start_run()
    finish(main, qtbot)
    assert run.d2["state"] == "not_run" and run.status == "finished"


def test_series_runs_d2_at_every_q_and_waits_for_it(main, qtbot):
    m = main.setup_model
    cccc(m)
    m.set_choice(load_mode="series")
    m.set_text("q_series", "500; 1000; 2000")
    main.runs.d2_factory = lambda inputs: InlineD2(inputs, polls=2)
    series = main._start_run()
    main.runs.tick()  # point 1: KK-VIM answers, D2 starts
    assert len(series.run_ids) == 1  # the next point waits for the D2 of this one
    finish(main, qtbot)
    data = series_data(series, main.runs)
    assert series.status == "finished" and [p.d2_state for p in data.points] == ["ok", "ok", "ok"]
    assert [c.inputs["q"] for c in InlineD2.created] == [500.0, 1000.0, 2000.0]
    gui_tab = AnalysisWindow(Session(settings=None), run=main.runs.get(series.run_ids[-1]), series=data)
    qtbot.addWidget(gui_tab)
    dx, dy = gui_tab.series_tab.figure.d2_curve()
    assert dx == [0.5, 1.0, 2.0] and dy == [p.d2_w_m * 1000.0 for p in data.points]
    assert gui_tab.series_tab.model.columnCount() == 17


def test_stop_during_d2_of_a_series_point_cancels_the_series(main, qtbot):
    m = main.setup_model
    cccc(m)
    m.set_choice(load_mode="series")
    m.set_text("q_series", "500; 1000; 2000")
    main.runs.d2_factory = SlowD2
    series = main._start_run()
    main.runs.tick()
    run = main.runs.get(series.run_ids[0])
    main.runs.stop(run.run_id)
    finish(main, qtbot)
    data = series_data(series, main.runs)
    assert series.status == "cancelled" and len(series.run_ids) == 1
    assert [p.d2_state for p in data.points] == ["cancelled", "not_run", "not_run"]
    assert data.points[0].included and not data.points[0].on_d2_curve  # KK-VIM point kept, no D2 point


def test_a_killed_computing_process_ends_the_point_and_the_series_goes_on(main, qtbot, monkeypatch):
    """O-1 in the dispatcher: the real SolverProcess of point 1 is killed (no message); no BrokenPipeError escapes."""
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parent.parent))
    import process_targets

    m = main.setup_model
    cccc(m)
    m.set_choice(load_mode="series")
    m.set_text("q_series", "500; 1000")
    processes = []

    def factory(request):
        p = SolverProcess(request, target=process_targets.hang) if not processes else InlineProcess(request)
        processes.append(p)
        return p

    main.runs.process_factory = factory
    series = main._start_run()
    time.sleep(0.5)
    processes[0]._process.kill()
    finish(main, qtbot)
    first, second = (main.runs.get(i) for i in series.run_ids)
    assert first.status == "error" and "Сбой вычислительного процесса" in " ".join(t for _, _, t in first.important)
    assert first.d2["state"] == "not_run"
    assert second.status == "finished" and second.d2["state"] == "ok" and series.status == "finished"
    assert not processes[0]._process.is_alive()


# --- analysis ---------------------------------------------------------------------------------------------------------

def finished_run(main, qtbot, **kw):
    cccc(main.setup_model, **kw)
    run = main._start_run()
    finish(main, qtbot)
    return run


def test_analysis_has_the_d2_tab_and_the_executed_row(main, qtbot, tmp_path):
    run = finished_run(main, qtbot, n="32")
    w = AnalysisWindow(Session(settings=None), run=run)
    qtbot.addWidget(w)
    titles = [w.tabs.tabText(i) for i in range(w.tabs.count())]
    assert titles[titles.index("Сравнение с эталоном") + 1] == "МКР D2"
    c = w.comparison.comparison
    assert [r.reference.kind for r in c.rows] == ["D2_live", "published", "D2", "FEM"]
    assert c.rows[0].reference.beta == 0.0012759787033572092 and c.rows[0].beta_kkvim == 0.0012741269462195456
    assert w.overview.comparison_rows[1][0].startswith("МКР — двумерный (D2), рассчитан")
    tab = w.d2_tab
    assert tab.state_label.text().endswith("МКР D2 рассчитан") and tab.difference is not None
    assert [name for name, _, _ in tab.timing][:3] == ["kkvim_solver_s", "kkvim_import_s", "kkvim_wall_s"]
    assert tab.values.model().rowCount() == 33 * 33
    paths = w.save_package(tmp_path)
    assert {"d2_field", "d2_result", "comparison", "report"} <= set(paths)
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert manifest["independent_d2"]["state"] == "ok"


def test_a_failed_d2_tab_states_the_reason(main, qtbot):
    main.runs.d2_factory = None
    run = finished_run(main, qtbot)
    w = AnalysisWindow(Session(settings=None), run=run)
    qtbot.addWidget(w)
    assert w.d2_tab is not None and w.d2_tab.values is None and "не выполнялся" in w.d2_tab.state_label.text()
    assert not any(r.reference.kind == "D2_live" for r in w.comparison.comparison.rows)


def test_a_d2_arriving_after_the_window_opened_rebuilds_the_comparison(main, qtbot):
    cccc(main.setup_model, n="32")
    main.runs.d2_factory = lambda inputs: InlineD2(inputs, polls=10)
    run = main._start_run()
    main.runs.tick()
    assert run.d2["state"] == "pending"
    w = AnalysisWindow(Session(settings=None), run=run)  # opened while D2 still runs (e.g. from a test or a script)
    qtbot.addWidget(w)
    assert w.d2_tab is None and not any(r.reference.kind == "D2_live" for r in w.comparison.comparison.rows)
    finish(main, qtbot)
    qtbot.waitUntil(lambda: w.d2_tab is not None, timeout=10000)
    assert w.comparison.comparison.rows[0].reference.kind == "D2_live"


# --- O-3, O-4 ---------------------------------------------------------------------------------------------------------

def test_stopping_rule_texts_name_s_min_both_tolerances_and_two_half_steps():
    for topic in ("status.overview", "iterations.overview", "param.stop.s_min", "param.stop.tol_inf", "param.stop.tol_E"):
        a = get(topic)
        text = a.summary + " " + a.body
        assert "s ≥ s_min" in text and "подряд" in text and "tol_∞" in text and "tol_E" in text, topic
    assert "не больше допусков)" not in get("status.overview").body


def test_iterations_tab_states_the_rule_with_s_min(main, qtbot):
    main.setup_model.set_choice(supports="SSSS", N=1)
    for key, text in (("a", "2"), ("b", "1"), ("h", "10"), ("E", "210"), ("nu", "0,3"), ("q", "5"), ("n_x", "16"),
                      ("n_y", "8")):
        main.setup_model.set_text(key, text)
    run = main._start_run()
    finish(main, qtbot)
    from kkvim_gui.analysis.results import HistoryTab

    intro = HistoryTab._intro(run.outcome)
    assert "s ≥ s_min" in intro and "s − 1" in intro and "11.3" in intro


@pytest.fixture
def gui(qtbot, qapp):
    g = GuiApplication(Session(settings=None), demo=True)
    qtbot.addWidget(g.main_window)
    yield g
    for window in list(g.analysis_windows):
        window.close()
    g.help.close()


def test_the_highlighted_index_entry_is_always_the_article_shown(gui):
    hw = gui.help.open("verification.navier")
    assert hw.highlighted_topic() == hw.current_topic == "verification.navier"
    hw.show_search("Навье")  # was: help.using shown while verification.navier stayed highlighted
    assert hw.current_topic == hw.highlighted_topic() == "verification.navier"
    hw.show_topic("param.N")
    hw.search.setText("Пуассон")  # the article shown is not among the results: nothing highlighted
    assert hw.current_topic == "param.N" and hw.highlighted_topic() is None
    hw.search.setText("")
    assert hw.highlighted_topic() == "param.N"
    hw.show_search("")  # F1 without context: the start article, highlighted
    assert hw.current_topic == hw.highlighted_topic() == "help.using" and hw.mode == "search"
    hw.show_search("несуществующееслово")
    assert hw.current_topic == "help.using" and hw.highlighted_topic() is None and hw.topics_in_index() == []
    hw.show_search("D2")
    assert hw.current_topic == hw.highlighted_topic() and "D2" in get(hw.current_topic).title
