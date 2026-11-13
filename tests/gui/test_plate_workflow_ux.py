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
    assert analysis.tabs.tabText(analysis.tabs.count() - 1) == "Нагрузка — прогиб"


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
    assert [window.tabs.tabText(i) for i in range(window.tabs.count())] == list(SOLVER_TABS)
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
