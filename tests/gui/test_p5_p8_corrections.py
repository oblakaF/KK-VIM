"""GUI-P5-P8-FINAL-CORRECTIONS-01: the defects B-1, B-2 and B-3 of the QA GUI-P5-P8-FULL-WORKFLOW-QA-01.

B-1: the page header of the PDF report is measured; complete marks wrap onto further lines, never clipped, shortened
or left out; the body starts below the header; a heading is never left alone at the foot of a page.
B-2: a number in a PDF table is never shortened; its sign, digits and exponent are kept as given.
B-3: Ctrl+E is one shortcut of the analysis window. A real key event opens exactly one publication dialog for the
chart of the current tab, and nothing on a tab without a figure; Qt reports no ambiguous shortcut.
"""

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")
pytest.importorskip("pyqtgraph")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtGui import QAction, QGuiApplication, QShortcut  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from kkvim_gui.analysis import report_dialog as rd  # noqa: E402
from kkvim_gui.analysis.plot_tools import PublicationDialog, tools_of  # noqa: E402
from kkvim_gui.figures import report_pdf  # noqa: E402
from kkvim_gui.figures.report_pdf import MIN_TABLE_PT, PdfReport, cell_lines, is_number, table_layout  # noqa: E402
from tests.gui.test_multimethod_d2 import cccc  # noqa: E402
from tests.gui.test_p5_p8_workflow import analysed, main, ssss  # noqa: E402, F401

LONG_MARKS = ("RESEARCH_ONLY / PROVISIONAL / NOT_QUALIFIED: CCCC N = 2, s ≤ 2 (D-057.1); not a scientific acceptance",
              "DIAGNOSTIC: the run stopped with STOPPED_RANK_GUARD_UNCERTIFIED; field of the last completed half-step, "
              "not a result")
REPORTED = ("-1.9434287072469126e-06", "-4.5410867186511904e-07")  # the QA examples (CCCC, SSSS)
VALUES = (*REPORTED, "0.0", "-0.0", "1.0", "-1.0", "6.02214076e+23", "-1.7976931348623157e+308", "5e-324",
          "2.2250738585072014e-308", "123456789.12345679", "−1,94343·10⁻⁶", "4,54109·10⁻⁷", "12,5 %", "inf", "nan")


class Recording(PdfReport):
    made = []

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        Recording.made.append(self)


def header_texts(doc):
    return [t for line in doc.header_lines for t in line if t is not None]


# ---------------------------------------------------------------------------------------- B-1: page header


@pytest.mark.parametrize("marks,test_data", [((), False), (("RESEARCH_ONLY",), False), (LONG_MARKS, False),
                                             (LONG_MARKS, True)])
def test_header_keeps_every_mark_complete_within_the_page_and_above_the_body(qapp, tmp_path, marks, test_data):
    header = "KK-VIM · kkvim_20261126-152253_run006 · CCCC, N = 6, 32 × 13"
    doc = PdfReport(tmp_path / "r.pdf", "t", header, marks, test_data)
    try:
        texts = header_texts(doc)
        widths = [doc.pp.rich_width(t, report_pdf.HEADER_PT) for t in texts]
        assert all(w <= doc.width + 0.01 for w in widths)  # inside the margins, nothing clipped at the page edge
        flat = " ".join(" ".join(texts).split())
        assert header in flat
        for mark in (*marks, *(["TEST DATA"] if test_data else [])):
            assert " ".join(mark.split()) in flat  # complete: no mark shortened, dropped or ellipsised
        assert "…" not in flat
        if len(doc.header_lines) == 1 and marks:
            left, right, _ = doc.header_lines[0]
            gap = doc.width - doc.pp.rich_width(left, report_pdf.HEADER_PT) - doc.pp.rich_width(right, report_pdf.HEADER_PT)
            assert gap >= report_pdf.HEADER_GAP  # identification and marks do not overlap
        assert doc.top >= doc.header_bottom + report_pdf.HEADER_BODY_GAP - 1e-9 and doc.y == doc.top
        if marks == LONG_MARKS:
            assert len(doc.header_lines) >= 3  # the long combination wraps onto further lines
        for _ in range(3):  # every page carries the same complete header and the body starts below it
            doc.paragraph("x " * 4000)
        assert doc.page >= 3 and doc.top >= doc.header_bottom
    finally:
        doc.close()
    assert (tmp_path / "r.pdf").stat().st_size > 0


def test_heading_is_never_left_alone_at_the_foot_of_a_page(qapp, tmp_path):
    doc = PdfReport(tmp_path / "r.pdf", "t", "KK-VIM", ("RESEARCH_ONLY",))
    try:
        doc.y = doc.bottom - 20  # room for the heading but not for it and the paragraph after it
        doc.heading("5. Section that must stay with its text", 1)
        doc.paragraph("First line of the section. " * 30)
        page, baseline, text = doc.headings[-1]
        assert text.startswith("5.") and page == 2 and baseline < doc.top + 40
        doc.y = doc.bottom - 30
        doc.heading("6. Table section", 1)
        doc.table(("a", "b"), [("1", "2")] * 3)
        assert doc.headings[-1][0] == 3
    finally:
        doc.close()


# ---------------------------------------------------------------------------------------- B-2: numbers in tables


@pytest.mark.parametrize("text", VALUES)
def test_numeric_cells_are_recognised(text):
    assert is_number(text)


@pytest.mark.parametrize("text", ["VIM-1", "s = 2", "да", "—", "1.2.3", "неполный цикл: только Y¹; не VIM-1", "Navier"])
def test_texts_are_not_numbers(text):
    assert not is_number(text)


def test_numbers_are_whole_in_narrow_columns_and_texts_wrap(qapp, tmp_path):
    doc = PdfReport(tmp_path / "r.pdf", "t", "KK-VIM", ())
    try:
        headers = ("Эталон", "β эталона", "β КК-ВИМ", "Δβ", "Δβ/β, %", "Источник")
        rows = [("Navier, double series with a long explanation of the reference", v, v, v, v, "executed")
                for v in VALUES]
        layout = table_layout(doc.pp, headers, rows, doc.width, 7.0)
        # with the real fonts (the "offscreen" test platform has none: every glyph is a wide box) each number fits
        real_fonts = QGuiApplication.platformName() != "offscreen"
        assert layout["size"] >= MIN_TABLE_PT and (layout["numbers_whole"] or not real_fonts)
        for row in rows:
            for v, w in zip(row, layout["widths"]):
                lines = cell_lines(doc.pp, v, w - 2 * report_pdf.CELL_PAD, layout["size"])
                if is_number(v):
                    assert "".join(lines) == v and (lines == [v] or not layout["numbers_whole"])  # every character
                else:
                    assert "".join(lines).replace(" ", "") == v.replace(" ", "")  # the text wraps, nothing dropped
        doc.table(headers, rows)
        cells = doc.tables[-1]["cells"]
        for row, drawn in zip(rows, cells[1:]):
            for v, lines in zip(row, drawn):
                assert "…" not in "".join(lines)
                assert "".join(lines).replace(" ", "") == v.replace(" ", "")  # every character, in order
    finally:
        doc.close()


def test_a_number_too_wide_for_any_column_continues_without_losing_a_character(qapp, tmp_path):
    doc = PdfReport(tmp_path / "r.pdf", "t", "KK-VIM", ())
    try:
        v = "-1.9434287072469126e-06"
        layout = table_layout(doc.pp, [f"c{k}" for k in range(40)], [[v] * 40], doc.width, 7.0)
        assert not layout["numbers_whole"] and layout["size"] == MIN_TABLE_PT
        lines = cell_lines(doc.pp, v, layout["widths"][0] - 2 * report_pdf.CELL_PAD, layout["size"])
        assert len(lines) > 1 and "".join(lines) == v
    finally:
        doc.close()


def drawn_tables(window, tmp_path, monkeypatch):
    Recording.made = []
    monkeypatch.setattr(report_pdf, "PdfReport", Recording)
    chosen = {c.id for c in rd.components(window) if c.available}
    result = rd.build_report(window, tmp_path, chosen, "ru", formats=("png",))
    return Recording.made[-1], result


def assert_tables_complete(doc):
    assert doc.tables
    numbers = 0
    for layout in doc.tables:
        for row in layout["cells"]:
            for lines in row:
                joined = "".join(lines)
                assert "…" not in joined
                if is_number(joined):
                    numbers += 1
                    assert len(lines) == 1 or not layout["numbers_whole"]
    return numbers


def test_report_tables_of_a_real_ssss_run_keep_every_number(ssss, tmp_path, monkeypatch):  # noqa: F811
    run, w = ssss
    before = rd.report_builder.source_digest(run.outcome)
    doc, result = drawn_tables(w, tmp_path, monkeypatch)
    assert assert_tables_complete(doc) > 0 and all(t["numbers_whole"] for t in doc.tables)
    shown = {"".join(lines) for t in doc.tables for row in t["cells"] for lines in row}
    for x in w.comparison.comparison.rows:  # the comparison table shows repr() of the stored values, exponent kept
        assert repr(x.delta) in shown and repr(x.beta_kkvim) in shown
    assert rd.report_builder.source_digest(run.outcome) == before
    assert result["folder"].joinpath("report.pdf").is_file()


def test_report_of_a_protective_stop_has_the_complete_diagnostic_header(main, qtbot, tmp_path, monkeypatch):  # noqa: F811
    run, w = analysed(main, qtbot, lambda m: cccc(m, n="40", N=2, s=2))
    o = run.outcome
    assert o.diagnostic_only
    doc, result = drawn_tables(w, tmp_path, monkeypatch)
    flat = " ".join(" ".join(header_texts(doc)).split())
    marks = doc.all_marks()
    assert any(m.startswith("DIAGNOSTIC") for m in marks)
    assert all(" ".join(m.split()) in flat for m in marks)
    assert all(doc.pp.rich_width(t, report_pdf.HEADER_PT) <= doc.width + 0.01 for t in header_texts(doc))
    assert doc.page >= 2
    assert_tables_complete(doc)


# ---------------------------------------------------------------------------------------- B-3: Ctrl+E


def dialogs(window):
    return [d for d in window.findChildren(PublicationDialog) if d.isVisible()]


def press_ctrl_e(window, qtbot):
    window.activateWindow()
    window.raise_()
    qtbot.waitUntil(lambda: QApplication.activeWindow() is window, timeout=3000)
    QTest.keyClick(window, Qt.Key.Key_E, Qt.KeyboardModifier.ControlModifier)
    qtbot.wait(30)


def close_all(window):
    for d in dialogs(window):
        d.reject()
        d.deleteLater()
    QApplication.processEvents()


def test_ctrl_e_key_event_opens_exactly_one_dialog_for_the_current_chart(ssss, qtbot, qtlog):  # noqa: F811
    _, w = ssss
    for t in tools_of(w):  # fill every chart menu as when it is opened: no action may register Ctrl+E
        if hasattr(t, "figure_button"):
            t._fill_figure_menu(t.figure_button.menu())
    window_keys = [s for s in w.findChildren(QShortcut) if s.key().toString() == "Ctrl+E"]
    assert len(window_keys) == 1 and window_keys[0] in w.shortcuts
    assert not [a for a in w.findChildren(QAction) if "Ctrl+E" in a.shortcut().toString()]

    for tab in [t for t in (w.history, w.sections, w.energy, w.fields) if w.tabs.indexOf(t) >= 0]:
        w.tabs.setCurrentWidget(tab)
        qtbot.wait(20)
        expected = w.current_tools()
        assert expected is not None and expected in tools_of(tab)
        press_ctrl_e(w, qtbot)
        opened = dialogs(w)
        assert len(opened) == 1 and opened[0].source is expected  # exactly once, the chart of this tab
        close_all(w)

    w.tabs.setCurrentWidget(w.sections)
    w.sections.tools_y.used.emit(w.sections.tools_y)  # the second chart of the tab used last
    press_ctrl_e(w, qtbot)
    opened = dialogs(w)
    assert len(opened) == 1 and opened[0].source is w.sections.tools_y
    close_all(w)

    plain = [w.tabs.widget(k) for k in range(w.tabs.count()) if not tools_of(w.tabs.widget(k))]
    assert plain, "the analysis window has a tab without a figure"
    w.tabs.setCurrentWidget(plain[0])
    qtbot.wait(20)
    press_ctrl_e(w, qtbot)
    assert dialogs(w) == [] and "нет рисунка" in w.statusBar().currentMessage()  # no unrelated export
    assert not [r for r in qtlog.records if "Ambiguous" in r.message]


def test_export_menu_still_opens_the_publication_dialog(ssss, qtbot, qtlog):  # noqa: F811
    _, w = ssss
    w.tabs.setCurrentWidget(w.history)
    button = w._export_button()
    action = next(a for a in button.menu().actions() if a.text().startswith("Рисунок для публикации"))
    assert action.text().endswith("\tCtrl+E") and action.shortcut().isEmpty()
    action.trigger()
    opened = dialogs(w)
    assert len(opened) == 1 and opened[0].source is w.history.tools
    close_all(w)
    t = w.history.tools
    t._fill_figure_menu(t.figure_button.menu())
    chart_action = next(a for a in t.figure_button.menu().actions() if a.text().startswith("Рисунок для публикации"))
    assert chart_action.shortcut().isEmpty()
    chart_action.trigger()
    assert len(dialogs(w)) == 1
    close_all(w)
    assert not [r for r in qtlog.records if "Ambiguous" in r.message]
