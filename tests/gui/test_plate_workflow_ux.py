"""The plate workflow of GUI-PLATE-WORKFLOW-UX-01 through the windows.

- setup: plain labels, N = 1…6 only, the route status shown separately and never "research" for every SSSS route,
  advanced stopping parameters collapsed (opened by an error there), every guard of the form kept;
- load series: preview of the loads, one real run per q, sequential, cancellable, every point checked first;
- analysis: plain tab names, the live Navier comparison (also when it arrives after the window opened), centre
  cross-sections chosen by node, the colour-mapped 3D view with an approved scale, the «Нагрузка — прогиб» graph;
- start screen and help without developer placeholders; links, search and F1 topics resolve.
"""

import json

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")
pytest.importorskip("pyqtgraph")

import pyqtgraph as pg  # noqa: E402
from PySide6.QtWidgets import QLabel  # noqa: E402

from kkvim_gui.analysis import surface  # noqa: E402
from kkvim_gui.analysis.window import SOLVER_TABS, AnalysisWindow  # noqa: E402
from kkvim_gui.app import GuiApplication  # noqa: E402
from kkvim_gui.backend import benchmarks, kkvim_adapter, navier_job  # noqa: E402
from kkvim_gui.backend.process import INPUT, OK  # noqa: E402
from kkvim_gui.help.articles import get, load_articles, search  # noqa: E402
from kkvim_gui.help.render import CITATIONS_TITLE, STUB_NOTE, article_markdown  # noqa: E402
from kkvim_gui.main_window.window import MainWindow  # noqa: E402
from kkvim_gui.model.series import series_data  # noqa: E402
from kkvim_gui.session import Session  # noqa: E402


class InlineProcess:
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


class Hanging(InlineProcess):
    def poll(self):
        return None


class InlineNavier:
    """The worker contract evaluated in the test process (the real worker runs in tests/test_gui_navier_series.py)."""

    def __init__(self, inputs):
        self.inputs, self.terminated = inputs, False

    def poll(self):
        from kkvim_navier_worker import compute

        return navier_job.parse_output(json.dumps(compute(self.inputs)), self.inputs)

    def terminate(self, timeout=2.0):
        self.terminated = True
        return True


def texts_of(widget):
    return [w.text() for w in widget.findChildren(QLabel) if not w.isHidden()]


def plate(model, n_x="16", n_y="8"):
    for key, text in (("a", "2"), ("b", "1"), ("h", "10"), ("E", "210"), ("nu", "0,3"), ("q", "5"), ("n_x", n_x),
                      ("n_y", n_y)):
        model.set_text(key, text)


@pytest.fixture
def main(qtbot, tmp_path):
    w = MainWindow(Session(settings=None), protocol_dir=tmp_path / "runs", process_factory=InlineProcess)
    w.runs.navier_factory = InlineNavier
    qtbot.addWidget(w)
    w.set_state("setup")
    return w


def finish(main, qtbot):
    qtbot.waitUntil(lambda: main.runs.active is None and not main.runs.references_pending
                    and not any(s.active for s in main.runs.series), timeout=60000)


# --- setup ------------------------------------------------------------------------------------------------------------

def test_n_list_is_plain_and_the_route_status_is_separate(main):
    sv, m = main.setup_view, main.setup_model
    assert [sv.N.itemText(i) for i in range(sv.N.count())] == ["1", "2", "3", "4", "5", "6"]
    expected = {("SSSS", 1): "принятый решатель R3", ("SSSS", 2): "PROVISIONAL / NOT_QUALIFIED",
                ("SSSS", 4): "RESEARCH_ONLY / PROVISIONAL / NOT_QUALIFIED", ("CCCC", 1): "RESEARCH_ONLY"}
    for (supports, N), text in expected.items():
        m.set_choice(supports=supports, N=N)
        assert text in sv.route_status.text()
        assert sv.cccc_warning.isVisibleTo(sv) is (supports == "CCCC")
        assert [sv.N.itemText(i) for i in range(sv.N.count())] == ["1", "2", "3", "4", "5", "6"]
    m.set_choice(supports="SSSS", N=1)
    assert "RESEARCH" not in sv.route_status.text() and sv.route_status.property("status") == "success"
    assert sv.cccc_warning.text.text() == "Исследовательский режим — научная квалификация не завершена."
    labels = " ".join(texts_of(sv))
    assert "Разбиение по X (nₓ)" in labels and "Число членов N" in labels and "SSSS — шарнирное опирание" in \
        sv.supports.buttons["SSSS"].text() and "CCCC — жёсткая заделка" in sv.supports.buttons["CCCC"].text()
    for dev in ("D-043", "D-047", "R4,", "NOT_QUALIFIED"):
        assert not any(dev in sv.N.itemText(i) for i in range(sv.N.count()))


def test_stopping_parameters_are_collapsed_and_open_on_an_error(main):
    sv, m = main.setup_view, main.setup_model
    assert sv.advanced_toggle.isVisibleTo(sv) and not sv.advanced_box.isVisibleTo(sv)
    assert not sv.tol_E.isVisibleTo(sv) and not sv.details_box.isVisibleTo(sv)
    assert to_defaults(m.state)  # the approved defaults are used without opening the section
    m.set_text("tol_E", "abc")
    sv.check_field("tol_E")
    assert sv.advanced_box.isVisibleTo(sv) and sv.errors["tol_E"].isVisibleTo(sv)
    assert not sv.run_button.isEnabled()
    m.set_choice(N=4)
    assert not sv.advanced_toggle.isVisibleTo(sv) and sv.half_steps.isVisibleTo(sv) and sv.ssss_research.isVisibleTo(sv)
    m.set_choice(supports="CCCC", N=2)
    assert sv.cccc_half_steps.isVisibleTo(sv) and not sv.half_steps.isVisibleTo(sv)


def to_defaults(state):
    d = kkvim_adapter.DEFAULTS
    return (float(state.texts["tol_inf"]) == d["tol_inf"] and float(state.texts["tol_E"]) == d["tol_E"]
            and int(state.texts["s_min"]) == d["s_min"] and int(state.texts["s_max"]) == d["s_max"])


def test_series_mode_previews_the_loads(main):
    sv, m = main.setup_view, main.setup_model
    m.set_choice(load_mode="series")
    m.set_text("q_series", "1; 2,5; 5")
    assert sv.q_series.isVisibleTo(sv) and not sv.q.isVisibleTo(sv)
    assert "3" in sv.series_preview.text() and "2,5 kPa" in sv.series_preview.text()
    assert sv.run_button.text().endswith("Рассчитать серию")
    m.set_text("q_series", "5")
    assert "не заданы" in sv.series_preview.text()


# --- series ----------------------------------------------------------------------------------------------------------

def test_series_runs_every_point_and_draws_the_graph(main, qtbot):
    m = main.setup_model
    plate(m)
    m.set_choice(load_mode="series")
    m.set_text("q_series", "1; 5; 2,5")
    series = main._start_series(m.state)
    assert series is not None and len(series.run_ids) == 1   # sequential: one point at a time
    finish(main, qtbot)
    assert series.status == "finished" and len(series.run_ids) == 3
    runs = [main.runs.get(i) for i in series.run_ids]
    assert [r.request.q for r in runs] == [1000.0, 5000.0, 2500.0]
    assert all(r.outcome.status == "CONVERGED" and r.navier["state"] == "ok" for r in runs)
    single = kkvim_adapter.execute(runs[1].request)  # a series point is the ordinary run of its q
    assert np.array_equal(runs[1].outcome.w, single.w)
    data = series_data(series, main.runs)
    gui = GuiApplication.__new__(GuiApplication)
    gui.session, gui.main_window, gui.analysis_windows = main.session, main, []
    gui.help = type("H", (), {"install": lambda self, w: None, "open_search": lambda self: None})()
    window = gui.open_series(series.series_id)
    qtbot.addWidget(window)
    assert window.header.badge.isHidden()  # real runs of the accepted route: no TEST DATA or research badge
    kx, ky, nx, ny = window.tab.figure.curves()
    assert kx == [1.0, 2.5, 5.0] and ky == [p.w_centre_m * 1000.0 for p in sorted(data.points, key=lambda p: p.q)]
    assert nx == kx and ny == [p.navier_w_m * 1000.0 for p in sorted(data.points, key=lambda p: p.q)]
    window.tab.quantity.set_current("maxabs")
    window.tab.figure.set_quantity("maxabs")
    assert window.tab.figure.curves()[2] == []  # the reference is evaluated at the centre only
    main.run_view.select(series.run_ids[0])
    assert main.run_view.series_button.isVisibleTo(main.run_view) and main.run_view.series_button.isEnabled()
    analysis = gui.open_analysis()
    qtbot.addWidget(analysis)
    # a run of a completed series opens on «Нагрузка — прогиб» (GUI-OWNER-UX-01)
    assert analysis.tabs.tabText(0) == "Нагрузка — прогиб" and analysis.tabs.currentIndex() == 0


def test_series_is_cancelled_by_stop_and_leaves_no_process(main, qtbot):
    created = []
    main.runs.process_factory = lambda request: created.append(Hanging(request)) or created[-1]
    m = main.setup_model
    plate(m)
    m.set_choice(load_mode="series")
    m.set_text("q_series", "1; 2; 3")
    series = main._start_series(m.state)
    main.runs.stop(series.run_ids[0])
    assert series.status == "cancelled" and len(series.run_ids) == 1 and created[0].terminated
    assert main.runs.active is None and len(created) == 1
    data = series_data(series, main.runs)
    assert [p.run_status for p in data.points] == ["stopped", "not_run", "not_run"] and not data.included


def test_a_rejected_point_refuses_the_whole_series(main, monkeypatch):
    from kkvim_gui.main_window import window as window_module

    real = window_module.check_request
    monkeypatch.setattr(window_module, "check_request",
                        lambda r: ["отклонено"] if r.q == 3000.0 else real(r))
    m = main.setup_model
    plate(m)
    m.set_choice(load_mode="series")
    m.set_text("q_series", "1; 2; 3")
    assert main._start_series(m.state) is None and not main.runs.runs
    assert "3000.0" in main.setup_view.check_text.text()


# --- analysis ---------------------------------------------------------------------------------------------------------

def run_single(main, qtbot, **texts):
    plate(main.setup_model, **texts)
    run = main._start_run()
    finish(main, qtbot)
    main.run_view.select(run.run_id)
    return run


def test_analysis_tabs_and_live_navier_comparison(main, qtbot):
    run = run_single(main, qtbot)
    window = AnalysisWindow(main.session, main.run_rows(), run=run)
    qtbot.addWidget(window)
    assert [window.tabs.tabText(i) for i in range(window.tabs.count())] == [*SOLVER_TABS, "Циклы VIM-1–3"]
    assert SOLVER_TABS == ("Результат", "Прогиб", "Сечения", "Значения в узлах", "Итерации", "Диагностика",
                           "Сравнение с эталоном")
    c = window.comparison.comparison
    assert c.state == benchmarks.AVAILABLE and c.live and c.rows[0].reference.w_centre_m == run.navier["w_m"]
    assert window.overview.comparison_rows and window.status_run.text().endswith("Расчёт завершён")
    assert "CONVERGED" in window.status_stop.text() and "s = " in window.status_stop.text()
    assert window.status_accept.text().endswith("Научный статус: принятый решатель R3 (ACCEPTED)")
    assert "D-043" in window.status_accept.toolTip()  # the full provenance stays in the tooltip


def test_reference_arriving_after_the_window_opened_rebuilds_the_comparison(main, qtbot):
    run = run_single(main, qtbot)
    final = run.navier
    run.navier = navier_job.pending_record()
    window = AnalysisWindow(main.session, main.run_rows(), run=run)
    qtbot.addWidget(window)
    assert window.comparison.comparison.state == benchmarks.PENDING
    run.navier = final
    qtbot.waitUntil(lambda: window.comparison.comparison.state == benchmarks.AVAILABLE, timeout=5000)
    assert window.tabs.tabText(window.tabs.indexOf(window.comparison)) == "Сравнение с эталоном"
    assert window.overview.comparison_rows


def test_sections_default_to_the_centre_lines_by_node(main, qtbot):
    run = run_single(main, qtbot)
    window = AnalysisWindow(main.session, main.run_rows(), run=run)
    qtbot.addWidget(window)
    s, o = window.sections, run.outcome
    assert (s.at_x.index(), s.at_y.index()) == (8, 4)
    assert np.array_equal(s.curve_x.getData()[1], o.w[:, 4] * 1000.0)
    assert np.array_equal(s.curve_y.getData()[1], o.w[8, :] * 1000.0)
    s.at_y.slider.setValue(2)
    assert s.at_y.index() == 2 and np.array_equal(s.curve_x.getData()[1], o.w[:, 2] * 1000.0)
    assert "y = 0,25 m" in s.at_y.text.text()


def test_surface_is_colour_mapped_by_the_true_deflection(main, qtbot):
    run = run_single(main, qtbot)
    window = AnalysisWindow(main.session, main.run_rows(), run=run)
    qtbot.addWidget(window)
    f, o = window.fields.surface, run.outcome
    assert f.bar.levels() == (float(o.w.min()) * 1000.0, float(o.w.max()) * 1000.0)
    cells = surface.surface_cells(o.x, o.y, o.w, f.factor)
    expected = pg.colormap.get("viridis").map(
        (np.array([c[2] for c in cells]) * 1000.0 - f.low) / (f.high - f.low), mode="qcolor")
    assert [c.name() for *_, c in f.surface.cells] == [c.name() for c in expected]
    colours = [c.name() for *_, c in f.surface.cells]
    window.fields.factor.setCurrentIndex(window.fields.factor.findData(1.0))  # the factor never changes the colours
    assert [c.name() for *_, c in f.surface.cells] == colours
    window.fields.cmap.setCurrentText("cividis")
    assert f.cmap_name == "cividis"
    with pytest.raises(ValueError):
        f.set_colormap("jet")
    window.fields.mesh.setChecked(False)
    assert not f.surface.mesh
    assert np.array_equal(run.outcome.w, kkvim_adapter.execute(run.request).w)


# --- start screen and help -------------------------------------------------------------------------------------------

def test_start_screen_has_no_developer_placeholders(qtbot):
    w = MainWindow(Session(settings=None))
    qtbot.addWidget(w)
    text = " ".join(texts_of(w.start_view))
    assert w.start_view.new_button.text().endswith("Новый расчёт") and w.start_view.table is None
    for dev in ("D-043", "R4", "следующий этап", "отдельном процессе —", ".kkv", "P2", "P4", "NOT_QUALIFIED"):
        assert dev not in text, dev
    assert "Готово к расчёту" in text


@pytest.mark.parametrize("topic", sorted(load_articles()))
def test_help_articles_show_no_developer_text(topic):
    article = get(topic)
    md = article_markdown(article)
    for dev in ("Черновик", "Заготовка", STUB_NOTE, "## Основание", "docs/", "будет добавлен", "в подготовке",
                "В первой реализации", "Placeholder", "Draft"):
        assert dev not in md, (topic, dev)
    assert article.body, topic


def test_method_article_cites_the_primary_sources_and_states_vim_terms_honestly():
    md = article_markdown(get("method.overview"))
    assert f"## {CITATIONS_TITLE}" in md and "Substantiation of the variational iteration method" in md
    assert "He's Variational Iteration Method" in md and "VIM-1" not in md and "VIM-2" not in md


def test_new_widget_topics_exist_and_search_finds_them():
    for topic in ("param.N", "param.grid.nx", "param.grid.ny", "param.stop.tol_inf", "param.stop.tol_E",
                  "param.stop.s_min", "param.stop.s_max", "param.stop.strict", "param.halfsteps", "param.load.q",
                  "param.load.series", "results.series", "verification.navier", "status.overview"):
        assert get(topic) is not None, topic
    assert "verification.navier" in [a.id for a in search("Навье")]
    assert "param.N" in [a.id for a in search("число членов")]
    assert "не число итераций" in get("param.N").summary


# --- GUI-OWNER-UX-01: series generator, chart view, point runs, layout ------------------------------------------------

def test_generator_gives_exact_decimal_loads_and_refuses_with_a_reason():
    from kkvim_gui.model.setup_state import generate_series

    texts, error = generate_series("0,1", "2", "0,1")
    assert error is None and len(texts) == 20 and texts[2] == "0,3" and texts[-1] == "2"
    assert [float(t.replace(",", ".")) for t in texts] == [round(0.1 * k, 1) for k in range(1, 21)]  # no stepping drift
    assert generate_series("10", "1", "-1,5")[0] == ["10", "8,5", "7", "5,5", "4", "2,5", "1"]
    assert generate_series("1e3", "5e3", "1e3")[0] == ["1000", "2000", "3000", "4000", "5000"]
    assert generate_series("1.5", "4.5", "1.5")[0] == ["1.5", "3", "4.5"]  # decimal point kept
    for args, words in ((("1", "10", "0"), "нулю"), (("1", "10", "-1"), "направлен"), (("1", "10", "4"), "не достигается"),
                        (("0", "100", "1"), "101"), (("1", "1", "1"), "совпадают"), (("1", "x", "1"), "Конец"),
                        (("1", "1.0000000000000000001", "0.00000000000000000005"), "двоичное")):
        texts, error = generate_series(*args)
        assert texts is None and words in error, (args, error)
    assert "13" in generate_series("1", "10", "4")[1]  # the end is not moved: the next value is named


def test_series_dialog_writes_the_list_and_the_requests_are_the_ordinary_ones(main, qtbot):
    from kkvim_gui.main_window.setup_view import SeriesDialog
    from kkvim_gui.model.setup_state import to_request, to_series_requests

    sv, m = main.setup_view, main.setup_model
    plate(m)
    m.set_choice(load_mode="series")
    assert sv.series_button.isVisibleTo(sv)
    dialog = SeriesDialog(m, sv)
    qtbot.addWidget(dialog)
    ok = dialog.buttons.button(dialog.buttons.StandardButton.Ok)
    dialog.start.setText("1")
    dialog.end.setText("10")
    dialog.step.setText("4")
    assert not ok.isEnabled() and "не достигается" in dialog.error.text()
    dialog.step.setText("0,5")
    dialog.end.setText("3")
    dialog.unit.setCurrentText("kPa")
    assert ok.isEnabled() and "Будет выполнено расчётов: 5" in dialog.preview.text()
    assert "1; 1,5; 2; 2,5; 3 kPa" in dialog.preview.text()
    before = dict(m.state.texts)
    dialog.accept()
    assert m.state.texts["q_series"] == "1; 1,5; 2; 2,5; 3" and m.state.units["q"] == "kPa"
    assert {k: v for k, v in m.state.texts.items() if k != "q_series"} == {k: v for k, v in before.items() if k != "q_series"}
    requests = to_series_requests(m.state)
    assert [r.q for r in requests] == [1000.0, 1500.0, 2000.0, 2500.0, 3000.0]
    single = to_request(m.state.with_choice(load_mode="single").with_text("q", "2,5"))
    assert requests[3] == single  # exactly the ordinary run of that q
    assert "Будет выполнено расчётов: 5" in sv.series_preview.text()
    m.set_text("q_series", "1; 2; 7")  # a manual, non-uniform list stays possible
    assert [r.q for r in to_series_requests(m.state)] == [1000.0, 2000.0, 7000.0]


@pytest.fixture
def finished_series(main, qtbot):
    m = main.setup_model
    plate(m)
    m.set_choice(load_mode="series")
    m.set_text("q_series", "1; 5; 2,5")
    series = main._start_series(m.state)
    finish(main, qtbot)
    return series


def counting(main):
    """Every new solver or reference process from now on is recorded (none may start)."""
    started = []
    process, navier = main.runs.process_factory, main.runs.navier_factory
    main.runs.process_factory = lambda request: started.append(request) or process(request)
    main.runs.navier_factory = lambda inputs: started.append(inputs) or navier(inputs)
    return started


def test_toggling_curves_language_and_theme_keep_the_view_and_the_selection(main, qtbot, finished_series):
    from kkvim_gui.analysis.series import SeriesTab

    started = counting(main)
    tab = SeriesTab(main.session, series_data(finished_series, main.runs))
    qtbot.addWidget(tab)
    tab.resize(1200, 800)
    fig = tab.figure
    xs, ys = fig.all_points()
    (x0, x1), (y0, y1) = fig.view_range()
    assert x0 < 0 < max(xs) < x1 and y0 < 0 < max(ys) < y1  # the initial view contains zero and every point
    fig.plot.getViewBox().setRange(xRange=(2.0, 4.0), yRange=(0.4, 0.9), padding=0)
    user = fig.view_range()
    tab.select_q(2500.0)
    assert fig.marker.data["x"].tolist() == [2.5]
    for key, box in tab.toggles.items():
        box.setChecked(False)
        assert fig.view_range() == user, key
        box.setChecked(True)
        assert fig.view_range() == user, key
    tab.toggles["kkvim"].setChecked(False)
    assert fig.marker.data["x"].tolist() == [2.5] and fig.selected_q == 2500.0  # on the next visible curve
    for box in tab.toggles.values():
        box.setChecked(False)
    assert len(fig.marker.data) == 0 and fig.selected_q == 2500.0  # hidden curves are never marked; selection kept
    for box in tab.toggles.values():
        box.setChecked(True)
    main.session.set_figure_language("ru" if main.session.figure_language == "en" else "en")
    main.session.set_theme("dark", persist=False)
    assert fig.view_range() == user and fig.selected_q == 2500.0
    assert fig.curves()[0] == [1.0, 2.5, 5.0]  # every curve keeps its data
    tab.reset_button.click()
    assert fig.view_range() == (fig.home[0], fig.home[1]) != user
    assert started == []  # nothing was computed


def test_a_selected_point_opens_its_stored_run_without_a_new_calculation(main, qtbot, finished_series):
    series = finished_series
    started = counting(main)
    gui = GuiApplication.__new__(GuiApplication)
    gui.session, gui.main_window, gui.analysis_windows = main.session, main, []
    gui.help = type("H", (), {"install": lambda self, w: None, "open_search": lambda self: None})()
    main.run_view.select(series.run_ids[0])
    first = gui.open_analysis()
    qtbot.addWidget(first)
    assert first.tabs.tabText(0) == "Нагрузка — прогиб" and first.tabs.currentWidget() is first.series_tab
    assert first.series_tab.selected.q == 1000.0  # the point of the opened run
    tab = first.series_tab
    tab.select_q(2500.0)
    assert "q = 2,5 kPa" in tab.selected_label.text() and tab.open_button.isEnabled()
    tab.open_button.click()
    opened = gui.analysis_windows[-1]
    qtbot.addWidget(opened)
    run = main.runs.get(series.run_ids[2])
    assert opened is not first and run.request.q == 2500.0
    assert opened.run is run and opened.outcome is run.outcome  # the stored outcome itself, not a new calculation
    assert opened.series_tab.selected.q == 2500.0
    assert opened.windowTitle().endswith(f"запуск {run.run_id} · {run.started:%d.%m.%Y %H:%M:%S}")
    assert started == []
    # the dispatcher shows the q of every row with its unit and the point of the series
    texts = [main.run_view.table.item(r, 2).text() for r in range(main.run_view.table.rowCount())]
    assert [t.split(" · ")[0] for t in texts] == ["q = 1 kPa", "q = 5 kPa", "q = 2,5 kPa"]
    assert all(f"серия {series.series_id}, точка {k} из 3" in t for k, t in enumerate(texts, start=1))


def test_a_mouse_click_on_a_graph_point_selects_and_opens_that_stored_run(main, qtbot):
    """QA GUI-OWNER-UX-VIEW-QA-01: 8,1 kPa clicked after the 8 kPa row — ring, row, label and the opened run agree."""
    from PySide6.QtCore import QPoint, QPointF, Qt
    from PySide6.QtTest import QTest

    m = main.setup_model
    plate(m)
    m.set_choice(load_mode="series")
    m.set_text("q_series", "8; 8,1; 8,2; 16,1")
    series = main._start_series(m.state)
    finish(main, qtbot)
    started = counting(main)
    gui = GuiApplication.__new__(GuiApplication)
    gui.session, gui.main_window, gui.analysis_windows = main.session, main, []
    gui.help = type("H", (), {"install": lambda self, w: None, "open_search": lambda self: None})()
    main.run_view.select(series.run_ids[0])
    window = gui.open_analysis()
    qtbot.addWidget(window)
    window.resize(1440, 900)
    window.show()
    qtbot.waitExposed(window)
    tab, fig = window.series_tab, window.series_tab.figure
    box = fig.plot.getViewBox()
    points = tab.data.points
    assert [p.q for p in points[:3]] == [8000.0, 8100.0, 8200.0]
    for row, target in ((0, points[1]), (1, points[3])):  # 8,1 kPa after the 8 kPa row; 16,1 kPa after 8,1 kPa
        tab.view.selectRow(row)  # the table row chosen by the user first
        assert tab.selected is points[row]
        kx, ky = fig.curves()[:2]
        k = next(i for i, p in enumerate(sorted(tab.data.included, key=lambda p: p.q)) if p is target)  # drawing order
        x, y = kx[k], ky[k]
        box.setRange(xRange=(x - 0.15, x + 0.15), yRange=(y - 0.05, y + 0.05), padding=0)  # neighbours apart
        qtbot.wait(20)
        pixel = fig.plot.mapFromScene(box.mapViewToScene(QPointF(x, y)))
        QTest.mouseClick(fig.plot.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                         QPoint(int(pixel.x()), int(pixel.y())))
        p = tab.selected
        assert p is target, (target.q, p.q)  # the clicked record, not the previously chosen row
        assert fig.selected_point is p  # one record: the table, the label and the ring
        assert tab.view.selectionModel().currentIndex().row() == points.index(p)
        assert fig.marker.data["x"].tolist() == [x] and f"расчёт № {p.run_id} " in tab.selected_label.text()
    tab.view.selectRow(0)
    box.setRange(xRange=(7.95, 8.25), yRange=(ky[1] - 0.05, ky[1] + 0.05), padding=0)
    qtbot.wait(20)
    pixel = fig.plot.mapFromScene(box.mapViewToScene(QPointF(kx[1], ky[1])))
    QTest.mouseClick(fig.plot.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                     QPoint(int(pixel.x()), int(pixel.y())))
    assert tab.selected.q == 8100.0 and "q = 8,1 kPa" in tab.selected_label.text()
    tab.open_button.click()
    opened = gui.analysis_windows[-1]
    qtbot.addWidget(opened)
    run = main.runs.get(series.run_ids[1])
    assert opened.run is run and opened.outcome is run.outcome and run.request.q == 8100.0  # the stored run itself
    assert opened.series_tab.selected.q == 8100.0
    assert started == []  # no solver and no Navier process


def test_a_point_stopped_by_the_half_step_limit_is_not_called_converged(main, qtbot):
    from kkvim_gui.analysis.series import KKVIM_TOGGLE, SeriesTab

    m = main.setup_model
    plate(m)
    m.set_text("s_max", "5")
    m.set_choice(load_mode="series")
    m.set_text("q_series", "1; 2")
    series = main._start_series(m.state)
    finish(main, qtbot)
    data = series_data(series, main.runs)
    assert [p.solver_status for p in data.points] == ["MAX_HALF_STEPS", "MAX_HALF_STEPS"] and not data.included
    tab = SeriesTab(main.session, data)
    qtbot.addWidget(tab)
    assert [tab.model.data(tab.model.index(r, 2)) for r in range(2)] == ["MAX_HALF_STEPS", "MAX_HALF_STEPS"]
    assert tab.toggles["kkvim"].text() == KKVIM_TOGGLE and "не сходимость" in tab.toggles["kkvim"].toolTip()
    tab.select_q(1000.0)
    assert "диагностическое, не принятый научный результат" in tab.selected_label.text()
    assert "CONVERGED" not in tab.selected_label.text()


@pytest.mark.parametrize("size", [(1440, 900), (1280, 720), (960, 540)])
def test_scheme_keeps_a_to_b_and_does_not_move_when_sections_open(main, qtbot, size):
    from PySide6.QtCore import QRectF
    from PySide6.QtGui import QFont, QFontMetrics

    sv, m = main.setup_view, main.setup_model

    def plan():
        return sv.plate.compute_layout(QRectF(sv.plate.rect()), sv.plate.data()).plan

    main.resize(*size)
    main.show()
    qtbot.waitExposed(main)
    m.set_choice(load_mode="series")
    qtbot.wait(50)
    for a, b in (("2", "1"), ("1", "3")):
        m.set_text("a", a)
        m.set_text("b", b)
        qtbot.wait(50)
        rect = plan()
        assert abs(rect.width() / rect.height() - float(a) / float(b)) < 0.02  # true proportion a : b
    before = (sv.scheme.geometry(), plan())
    sv.advanced_toggle.setChecked(True)
    sv.details_toggle.setChecked(True)
    qtbot.wait(50)
    if sv.mode == "wide":  # the right-hand scheme stays where it is
        assert (sv.scheme.geometry(), plan()) == before
    else:  # below the method card it moves down with it, but keeps its size and drawing
        assert (sv.scheme.size(), plan()) == (before[0].size(), before[1])
    for control in (sv.load_mode, sv.supports):  # «Серия нагрузок», «SSSS — шарнирное опирание» … are never cut
        for button in control.buttons.values():
            bold = QFont(button.font())
            bold.setBold(True)
            assert button.width() >= QFontMetrics(bold).horizontalAdvance(button.text()), button.text()
    for card in sv.cards.values():  # no card is squeezed below its content (horizontal scrolling instead of clipping)
        assert card.width() >= card.minimumSizeHint().width()
    assert sv.run_button.isVisible() and not sv.run_button.visibleRegion().isEmpty()


def test_setup_explains_grid_intervals_references_and_statuses(main):
    sv, m = main.setup_view, main.setup_model
    plate(m, "16", "8")
    assert "интервалов" in sv.grid_note.text() and "17 × 9" in sv.grid_note.text()
    assert "проверяет решатель" in sv.grid_note.text()
    assert "МКЭ в программе не рассчитывается" in sv.fem_note.text()
    assert "не являются принятым научным результатом" in sv.diagnostic_note.text()
    m.set_choice(N=2)
    assert "PROVISIONAL / NOT_QUALIFIED" in sv.route_status.text() and "не ошибка решателя" in sv.route_status.text()
    m.set_choice(N=1)
    assert "Навье рассчитывается автоматически" in sv.ssss_independent.text()
