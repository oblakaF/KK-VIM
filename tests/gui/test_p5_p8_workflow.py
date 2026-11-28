"""GUI-P5-P8-COMPLETE-WORKFLOW-01: chart tools on every X–Y graph (P5), publication export (P6), the report constructor
(P7) and the window workflow (P8), on real solver results evaluated in the test process.

Checked (GUI_DESIGN.md 16.8 G2–G3, G5–G8, G12–G16; 20.6 W4–W6, W8; 18.4 R1–R6): showing and hiding curves keeps the data;
Snap only on the visible active curve; the values window and the CSV file hold the stored solver values (SI, full
precision), never a Snap position or a plotted coordinate; enlarging by a double click on the title; «Оформление» and its
reset; the publication figure is white, theme independent, has only the visible curves with the chosen styles and no
interaction overlay; PNG 300/600 dpi, SVG and PDF with metadata; TEST DATA for demonstration data; the report folder holds
only the chosen and available components, a manifest with inputs, versions, statuses and SHA256 of every file, refuses to
overwrite and never changes the run; CCCC N >= 2 keeps s <= 2 and states the missing cycles.
"""

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")
pytest.importorskip("pyqtgraph")

from PySide6.QtCore import QPoint, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QGuiApplication, QImage  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402

from kkvim_gui.analysis import report_dialog as rd  # noqa: E402
from kkvim_gui.analysis.plot_tools import (  # noqa: E402
    ChartInfo,
    ChartStyle,
    ChartTools,
    Curve,
    CurveOverride,
    FieldPublisher,
    PublicationDialog,
    StyleDialog,
    default_style,
)
from kkvim_gui.analysis.results import Figure, FieldFigure  # noqa: E402
from kkvim_gui.analysis.window import AnalysisWindow  # noqa: E402
from kkvim_gui.backend import chart_export, report_builder  # noqa: E402
from kkvim_gui.backend.outcome import ORIGIN_DEMO, RunOutcome, frozen_copy  # noqa: E402
from kkvim_gui.backend.request import RunRequest  # noqa: E402
from kkvim_gui.figures import publication as pub  # noqa: E402
from kkvim_gui.main_window.window import MainWindow  # noqa: E402
from kkvim_gui.session import Session  # noqa: E402
from tests.gui.test_multimethod_d2 import InlineD2, cccc  # noqa: E402
from tests.gui.test_plate_workflow_ux import InlineNavier, InlineProcess, finish, finished_series, plate  # noqa: E402, F401


@pytest.fixture
def main(qtbot, tmp_path):
    InlineD2.created = []
    w = MainWindow(Session(settings=None), protocol_dir=tmp_path / "runs", process_factory=InlineProcess)
    w.runs.navier_factory = InlineNavier
    w.runs.d2_factory = InlineD2
    qtbot.addWidget(w)
    w.set_state("setup")
    return w


def analysed(main, qtbot, setup=None, size=(1440, 900)):
    (setup or (lambda m: plate(m, "16", "8")))(main.setup_model)
    run = main._start_run()
    finish(main, qtbot)
    window = AnalysisWindow(main.session, main.run_rows(), run=run)
    qtbot.addWidget(window)
    window.resize(*size)
    window.show()
    qtbot.waitExposed(window)
    return run, window


@pytest.fixture
def ssss(main, qtbot):
    return analysed(main, qtbot)


def counting(main):
    started = []
    process, navier = main.runs.process_factory, main.runs.navier_factory
    main.runs.process_factory = lambda request: started.append(request) or process(request)
    main.runs.navier_factory = lambda inputs: started.append(inputs) or navier(inputs)
    return started


def demo_outcome():
    """Explicitly synthetic demonstration data (GUI_DESIGN.md 9): a smooth artificial function, not a plate solution."""
    x, y = np.linspace(0.0, 2.0, 9), np.linspace(0.0, 1.0, 5)
    w = np.outer(np.sin(np.pi * x / 2.0), np.sin(np.pi * y)) * 1e-3
    request = RunRequest((), 2.0, 1.0, 0.01, 2.1e11, 0.3, 5000.0, 8, 4, 1)
    return RunOutcome(ORIGIN_DEMO, request, "DEMO", "demo", "DEMO", "", None, "none", "demonstration", "neutral",
                      "TEST DATA", None, 1.0, frozen_copy(x), frozen_copy(y), frozen_copy(w), frozen_copy(w * 2.0), 1,
                      field_role="research")


# ---------------------------------------------------------------------------------------- publication (P6)


def xy_spec(test_data=False, lang="en"):
    xs = tuple(np.linspace(0.0, 2.0, 9))
    return pub.XYSpec("Section at y = 0.500 m", "x (m)", "w (mm)",
                      (pub.Series("KK-VIM, N = 1", xs, tuple(np.sin(np.pi * np.array(xs) / 2)), default_style(0)),),
                      test_data=test_data, lang=lang, meta={"run": "kkvim_test_run001"})


@pytest.mark.parametrize("dpi", pub.DPI_CHOICES)
def test_png_size_resolution_metadata_and_white_background_follow_the_choice_only(qapp, tmp_path, dpi):
    path = pub.save(xy_spec(), tmp_path / "f.png", "png", dpi, pub.WIDTHS_MM["one_column"])
    img = QImage(str(path))
    assert img.width() == round(85.0 / 25.4 * 72 * (dpi / 72))  # the width in mm and the dpi only (D6)
    assert round(img.dotsPerMeterX() * 0.0254) == dpi
    assert img.pixelColor(1, 1).name() == "#ffffff" and img.pixelColor(img.width() - 2, img.height() - 2).name() == "#ffffff"
    assert img.text("figure_language") == "en" and img.text("data_origin") == "solver" and img.text("kkvim_gui")


def test_svg_and_pdf_are_vector_files_with_metadata(qapp, tmp_path):
    svg = pub.save(xy_spec(lang="ru"), tmp_path / "f.svg").read_text(encoding="utf-8")
    assert "<svg" in svg and "<title>Section at y = 0.500 m</title>" in svg and "figure_language" in svg and "ru" in svg
    assert "<image" not in svg  # axes, lines and texts are vector
    pdf = pub.save(xy_spec(), tmp_path / "f.pdf").read_bytes()
    assert pdf.startswith(b"%PDF") and b"KK-VIM GUI" in pdf


def test_test_data_watermark_and_file_name(qapp):
    plain, marked = pub.image(xy_spec(), 150), pub.image(xy_spec(test_data=True), 150)
    assert plain != marked and marked.text("data_origin") == "demo"
    assert pub.file_name("kkvim_x_run001", "section_x", "ru", True, "png") == "kkvim_x_run001_section_x_ru_TEST-DATA.png"


def test_tick_labels_write_the_step_exactly_in_the_figure_language():
    ticks, step = pub.nice_ticks(-0.06, 1.06)
    assert step == 0.25 and [pub.tick_text(t, step, "ru") for t in ticks] == ["0", "0,25", "0,50", "0,75", "1,00"]
    ticks, step = pub.nice_ticks(0.0, 6.6e-5)
    assert pub.tick_text(ticks[1], step, "en") == "2×10⁻⁵"


def test_demonstration_field_export_is_marked_test_data(qtbot, tmp_path):
    session = Session(settings=None)
    fig = FieldFigure(session, demo_outcome(), isolines=True)
    qtbot.addWidget(fig)
    publisher = FieldPublisher(session, fig)
    spec = publisher.spec("ru")
    assert spec.test_data and spec.lang == "ru" and np.array_equal(spec.values, demo_outcome().w * 1000.0)
    dialog = PublicationDialog(session, publisher, fig)
    qtbot.addWidget(dialog)
    assert dialog.file_name().endswith("_field_en_TEST-DATA.png") or dialog.file_name().endswith("_field_ru_TEST-DATA.png")
    path = dialog.save(tmp_path / dialog.file_name())
    assert QImage(str(path)).text("data_origin") == "demo"


def test_isolines_pass_through_the_nodes_of_the_field(qtbot):
    """pyqtgraph's isocurve puts index i at i + 0.5: the transform maps it back to the node x_i (a fixed display defect)."""
    o = demo_outcome()
    fig = FieldFigure(Session(settings=None), o, isolines=True)
    qtbot.addWidget(fig)
    for i, j in ((0, 0), (3, 2), (8, 4)):
        p = fig.node_transform.map(QPointF(i + 0.5, j + 0.5))
        assert p.x() == pytest.approx(o.x[i]) and p.y() == pytest.approx(o.y[j])


# ---------------------------------------------------------------------------------------- chart tools (P5, P6)


def test_lines_menu_hides_and_shows_curves_and_keeps_the_data(ssss, main):
    run, w = ssss
    started = counting(main)
    t = w.sections.tools_x
    item = t.curve("kkvim").item
    data = [a.copy() for a in item.getOriginalDataset()]
    menu = t.lines_button.menu()
    menu.aboutToShow.emit()
    action = next(a for a in menu.actions() if a.text() == t.curve("kkvim").name)
    action.setChecked(False)
    assert not t.is_shown("kkvim") and t.interaction.active is None and not t.spec().series  # G5, G6, G14
    t.show_all(True)
    assert t.is_shown("kkvim") and all(np.array_equal(a, b) for a, b in zip(data, item.getOriginalDataset()))
    t.show_all(False)
    assert not t.visible_curves()
    t.show_all(True)
    assert started == [] and np.array_equal(run.outcome.w, w.outcome.w)


def test_snap_values_window_and_csv_hold_only_stored_values(ssss, tmp_path):
    run, w = ssss
    o = run.outcome
    w.tabs.setCurrentWidget(w.sections)
    t = w.sections.tools_x
    assert t.interaction.active == "kkvim"  # the first drawn curve is the active Snap curve
    t.interaction.set_snap(True)
    xs, ys = t.curve("kkvim").item.getData()
    t.interaction.move_to(t.view_box.mapViewToScene(QPointF((xs[2] + xs[3]) / 2, (ys[2] + ys[3]) / 2)))
    m = t.interaction.fix()
    assert m.snap.node is None  # an interpolated measurement on the drawn segment (G7)
    window = t.show_values("kkvim")
    j = w.sections.at_y.index()
    expected = [(i, j, float(o.x[i]), float(o.y[j]), float(o.w[i, j])) for i in range(len(o.x))]
    assert list(window.rows) == expected  # metres, exactly the solver array (W5)
    window.close()
    text = t.write_csv(tmp_path / "s.csv").read_text(encoding="utf-8")
    lines = text.splitlines()
    assert lines[0].startswith("# ") and "curve,i,j,x [m],y [m],w [m]" in lines
    import csv

    rows = list(csv.reader(line for line in lines if not line.startswith("#")))[1:]
    assert {r[0] for r in rows} == {"KK-VIM, N = 1"} and [float(r[-1]) for r in rows] == [e[-1] for e in expected]
    assert repr(m.y) not in text and repr(m.y / 1000.0) not in text  # no Snap position (G16)


def test_values_choice_when_two_curves_are_under_the_pointer(main, qtbot):
    run, w = analysed(main, qtbot, lambda m: (cccc(m, n="16", N=1, s=8), m.set_choice(with_d2=True)))
    fig = w.d2_tab.figure_x
    t = fig.tools
    assert [c.key for c in t.curves()] == ["kkvim", "d2"]
    xs, ys = t.curve("kkvim").item.getData()
    hits = t.curves_at(t.view_box.mapViewToScene(QPointF(xs[0], ys[0])))
    assert set(hits) == {"kkvim", "d2"}  # both start at w = 0 on the clamped edge (W6)
    d2_rows = t.curve("d2").rows()
    assert [r[-1] for r in d2_rows] == [float(v) for v in run.d2["field"][:, w.d2_tab.at_y.index()]]


def test_double_click_on_the_title_enlarges_and_restores(ssss, qtbot):
    _, w = ssss
    w.tabs.setCurrentWidget(w.sections)
    qtbot.wait(50)
    fx, fy = w.sections.fig_x, w.sections.fig_y
    QTest.mouseDClick(fx.plot.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(40, 40))
    assert fy.isVisible()  # a double click on the plot area does not enlarge (20.2)
    QTest.mouseDClick(fx.title, Qt.MouseButton.LeftButton)
    assert fx.isVisible() and not fy.isVisible() and w.sections.at_y.isVisible()  # the tab toolbar stays (W4)
    QTest.mouseDClick(fx.title, Qt.MouseButton.LeftButton)
    assert fy.isVisible()


def test_home_and_zoom_about_the_centre(ssss):
    _, w = ssss
    t = w.history.tools
    home = t.view_range()
    (x0, x1), _ = home
    t.zoom(0.5)
    (a0, a1), _ = t.view_range()
    assert a1 - a0 == pytest.approx((x1 - x0) * 0.5, rel=1e-6) and (a0 + a1) / 2 == pytest.approx((x0 + x1) / 2)
    t.go_home()
    assert [v for r in t.view_range() for v in r] == pytest.approx([v for r in home for v in r])


def test_style_applies_on_screen_and_in_the_file_and_resets(ssss, qtbot):
    _, w = ssss
    t = w.sections.tools_x
    item = t.curve("kkvim").item
    pen0 = item.opts["pen"]
    t.set_style(ChartStyle(plot_bg="#FFFBEA", curves={"kkvim": CurveOverride("#D55E00", 3.0, "dash", False)}))
    pen = item.opts["pen"]
    assert pen.color().name() == "#d55e00" and pen.widthF() == 3.0 and pen.style() == Qt.PenStyle.DashLine
    assert item.opts["symbol"] is None
    s = t.spec().series[0].style
    assert (s.colour, s.width, s.dash, s.marker) == ("#D55E00", 3.0, "dash", None)  # G14
    dialog = StyleDialog(t)
    qtbot.addWidget(dialog)
    assert dialog.chosen().curves["kkvim"].colour == "#D55E00"
    t.reset_style()
    assert item.opts["pen"] == pen0 and item.opts["symbol"] == "o"
    assert t.spec().series[0].style == default_style(0, "solid", "o")  # the publication standard (G13)


def test_export_is_identical_in_light_and_dark_theme_and_has_no_overlay(ssss, tmp_path):
    _, w = ssss
    t = w.sections.tools_x
    t.interaction.set_snap(True)
    t.interaction.set_projections(True)
    xs, ys = t.curve("kkvim").item.getData()
    t.interaction.move_to(t.view_box.mapViewToScene(QPointF(xs[4], ys[4])))
    t.interaction.fix()
    light = pub.image(t.spec("en"), 150)
    w.session.set_theme("dark", persist=False)
    dark = pub.image(t.spec("en"), 150)
    w.session.set_theme("light", persist=False)
    assert light == dark  # G15
    spec = t.spec("en")
    assert [s.label for s in spec.series] == ["KK-VIM, N = 1"] and spec.lang == "en"  # the drawn curve only, no marker


def test_copy_figure_puts_raster_and_svg_in_the_clipboard(ssss):
    _, w = ssss
    w.tabs.setCurrentWidget(w.sections)
    w.copy_current()
    data = QGuiApplication.clipboard().mimeData()
    assert data.hasImage() and bytes(data.data("image/svg+xml")).startswith(b"<?xml")


def test_ctrl_e_and_ctrl_tab_in_the_analysis_window(ssss, qtbot):
    _, w = ssss
    w.tabs.setCurrentWidget(w.history)
    w.activateWindow()
    qtbot.wait(20)
    index = w.tabs.currentIndex()
    QTest.keyClick(w, Qt.Key.Key_Tab, Qt.KeyboardModifier.ControlModifier)
    assert w.tabs.currentIndex() == (index + 1) % w.tabs.count()
    w.tabs.setCurrentWidget(w.history)
    dialog = w.publish_current()
    qtbot.addWidget(dialog)
    assert isinstance(dialog, PublicationDialog) and dialog.source is w.history.tools
    assert dialog.file_name() == f"{w.export_base()}_history_en.png"
    assert any(s.key().toString() == "Ctrl+E" for s in w.shortcuts)


# ---------------------------------------------------------------------------------------- cycles: CCCC N >= 2 keeps s <= 2


def test_cccc_n2_cycles_offer_only_what_was_computed(main, qtbot):
    run, w = analysed(main, qtbot, lambda m: cccc(m, n="12", N=2, s=2))
    o = run.outcome
    assert o.s <= 2 and o.s_requested == 2
    t = w.cycles.figure.tools
    menu = t.lines_button.menu()
    menu.aboutToShow.emit()
    enabled = {a.text(): a.isEnabled() for a in menu.actions() if a.isCheckable()}
    assert enabled["VIM-1"] and not enabled["VIM-2"] and not enabled["VIM-3"]
    assert [s.label for s in t.spec("en").series if s.label][0].startswith("KK-VIM VIM-1")
    components = {c.id: c for c in rd.components(w)}
    assert components["vim_cycles"].available
    reasons = [c.reason for c in w.cycles.table.cycles if not c.on_graph]
    assert all("s ≤ 2" in r or "s = 2" in r or "не вычисляется" in r for r in reasons)
    pix = w.cycles.figure.grab()
    assert pix.width() > 0 and t.toolbar.isVisibleTo(w.cycles.figure)  # the saved figure hid the tools only while grabbed


# ---------------------------------------------------------------------------------------- report constructor (P7)


def test_report_folder_has_only_the_chosen_components_and_a_complete_manifest(ssss, tmp_path):
    run, w = ssss
    o = run.outcome
    before = report_builder.source_digest(o)
    chosen = {"section_x", "history", "history_table", "chart_data"}
    result = rd.build_report(w, tmp_path, chosen, "ru", formats=("png", "svg"), dpi=600)
    folder = result["folder"]
    files = sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file())
    assert "figures/section_x_ru.png" in files and "figures/section_x_ru.svg" in files
    assert "figures/history_ru.png" in files and not any("section_y" in f or "field_ru" in f for f in files)  # R1
    assert "report.pdf" in files and "tables/history.csv" in files and "data/manifest.json" in files
    png = QImage(str(folder / "figures/section_x_ru.png"))
    assert round(png.dotsPerMeterX() * 0.0254) == 600 and png.text("figure_language") == "ru"  # R6
    m = json.loads(result["manifest"].read_text(encoding="utf-8"))
    assert m["run_id"] == run.run_id and m["input"]["si"]["a"] == {"value": 2.0, "unit": "m"}  # R2
    assert m["statuses"]["solver_status"] == o.status and m["program"]["versions"] and m["report"]["figure_language"] == "ru"
    listed = {f["path"]: f for f in m["files"]}
    assert set(listed) == set(files) - {"manifest.json"}
    for path, f in listed.items():
        assert f["sha256"] == report_builder.sha256(folder / path)
    assert m["source_data"]["unchanged"] and m["source_data"]["sha256_before"] == before  # R5
    assert report_builder.source_digest(o) == before
    unavailable = {u["id"] for u in m["report"]["unavailable"]}
    assert {"d2_section_x", "load_series", "d2_field_table"} <= unavailable  # R3: listed with a reason, no file
    assert m["report"]["xlsx"]["status"].startswith("NOT_BUILT") and not any(f.endswith(".xlsx") for f in files)


def test_report_never_overwrites(ssss, tmp_path):
    run, w = ssss
    first = rd.build_report(w, tmp_path, {"section_x"}, "en", pdf=False, raw=False)
    second = rd.build_report(w, tmp_path, {"section_x"}, "en", pdf=False, raw=False)
    assert first["folder"] != second["folder"] and second["folder"].name.endswith("_2")
    with pytest.raises(FileExistsError):
        report_builder.build(rd.plan(w, {"section_x"}, "en"), first["folder"])


def test_report_of_a_protective_stop_is_diagnostic(main, qtbot, tmp_path):
    """CCCC N = 2 above the grid limit 32: the solver's rank guard stops at s = 1; the field is diagnostic only."""
    run, w = analysed(main, qtbot, lambda m: cccc(m, n="40", N=2, s=2))
    o = run.outcome
    assert o.diagnostic_only and o.status.startswith("STOPPED")
    spec = w.fields.publisher.spec("en")
    assert any("DIAGNOSTIC" in mark for mark in spec.marks)
    result = rd.build_report(w, tmp_path, {"field", "field_table"}, "en", pdf=False)
    m = json.loads(result["manifest"].read_text(encoding="utf-8"))
    assert any("DIAGNOSTIC" in mark for mark in m["marks"]) and m["statuses"]["field_role"] == "diagnostic"
    assert any(f["path"] == "tables/field_diagnostic.csv" for f in m["files"])
    comp = {c.id: c for c in rd.components(w)}
    assert not comp["comparison_table"].available


def test_demonstration_report_is_marked_test_data(qapp, tmp_path):
    from kkvim_gui.backend import export

    o = demo_outcome()

    def field(folder):
        path = Path(folder) / "tables" / export.field_file_name(o)
        return [(export.write_field_csv(o, path), "csv", None)]

    plan = report_builder.ReportPlan(o, "en", [report_builder.Item("field_table", "table", "field", field)],
                                     {"run_id": 7, "data_origin": "demo"})
    name = report_builder.folder_name("kkvim_demo", "en", o.is_demo)
    result = report_builder.build(plan, tmp_path / name)
    m = json.loads(result["manifest"].read_text(encoding="utf-8"))
    assert name.endswith("_TEST-DATA") and m["data_origin"] == "demo" and m["test_data"]  # R4
    assert m["files"][0]["path"].endswith("_TEST-DATA.csv")
    assert (result["folder"] / m["files"][0]["path"]).read_text(encoding="utf-8").startswith("# TEST DATA")


def test_chart_csv_of_demonstration_data_starts_with_test_data(tmp_path):
    path = chart_export.write_curves_csv(tmp_path / "c.csv", (("x", "m"), ("w", "m")), [("a", [(0.1, 1e-3)])],
                                         ["figure: test"], test_data=True)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "# TEST DATA" and lines[-1] == "a,0.1,0.001" and lines[-2] == "curve,x [m],w [m]"


def test_a_figure_with_its_own_curve_tools(qtbot):
    """ChartTools on any X–Y Figure: the default active curve, the coordinate field and the publication texts."""
    session = Session(settings=None)
    o = demo_outcome()
    fig = Figure(session, o, "section.x.title", {"y": 0.5})
    fig.set_axes("axis.x", "axis.w")
    qtbot.addWidget(fig)
    item = fig.plot.plot(list(o.x), list(o.w[:, 2] * 1000.0))
    curves = [Curve("c", item, "кривая", lambda lang: "curve", lambda: [(float(v),) for v in o.w[:, 2]])]
    t = ChartTools(session, fig, fig.plot, fig.title, ChartInfo("section_x", fig.texts, lambda: (("x", "m"), ("w", "mm")),
                                                                (("w", "m"),), test_data=True), lambda: curves)
    assert t.interaction.active == "c" and t.readout is not None
    assert t.spec("ru").title == "Сечение при y = 0,500 m" and t.spec("ru").test_data


def test_series_lines_menu_and_switches_are_one_state_and_the_saved_graph_has_no_tools(main, qtbot, finished_series):  # noqa: F811
    from kkvim_gui.analysis.series import SeriesTab
    from kkvim_gui.model.series import series_data

    series = finished_series
    tab = SeriesTab(main.session, series_data(series, main.runs))
    qtbot.addWidget(tab)
    tab.resize(1300, 900)
    tab.show()
    qtbot.waitExposed(tab)
    fig = tab.figure
    t = fig.tools
    t.set_shown("navier", False)
    assert not tab.toggles["navier"].isChecked() and not fig.shown["navier"] and not fig.navier.isVisible()
    tab.toggles["navier"].setChecked(True)
    assert t.is_shown("navier")
    labels = [s.label for s in t.spec("en").series]
    assert "KK-VIM (a run at every q)" in labels and any(label.startswith("Navier") for label in labels)
    rows = t.curve("kkvim").rows()
    assert [r[0] for r in rows] == [p.q for p in fig.source_records("kkvim")]  # Pa, the stored order
    pix = fig.export_pixmap()
    assert pix.height() > 0 and t.toolbar.isVisibleTo(fig) and fig.interaction.snap_button.isVisibleTo(fig)
