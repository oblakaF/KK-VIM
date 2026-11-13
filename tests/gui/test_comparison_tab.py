"""The "Сравнение" tab with real solver runs from the form (GUI-BENCHMARK-COMPARISON-01).

- a completed CCCC run on a stored benchmark grid shows the published, D2 and FEM references with signed differences
  at full precision, next to the unchanged solver statuses; no verdict word appears;
- a run without reference data states it and shows no numbers;
- a protective stop (diagnostic field) is never compared;
- damaged data are reported, and the demonstration window keeps the tab disabled.
"""

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")
pytest.importorskip("pyqtgraph")

from PySide6.QtWidgets import QLabel  # noqa: E402

from kkvim_gui.analysis.comparison import ComparisonTab, full  # noqa: E402
from kkvim_gui.analysis.window import SOLVER_TABS, TABS, AnalysisWindow  # noqa: E402
from kkvim_gui.app import GuiApplication, type_case  # noqa: E402
from kkvim_gui.backend import benchmarks as bm  # noqa: E402
from kkvim_gui.session import Session  # noqa: E402

CASE = dict(supports="CCCC", N=1, a=1.0, b=1.0, h=0.01, E=2.1e11, nu=0.3, q=1000, n_x=16, n_y=16, half_steps=1)
VERDICTS = ("ACCURATE", "INACCURATE", "PASS", "FAIL", "ТОЧНО", "НЕТОЧНО", "ACCEPTED")


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


def run_case(qtbot, gui, **change):
    w = gui.main_window
    type_case(w.setup_model, {**CASE, **change})
    assert w.setup_view.try_run() is True
    run = w.runs.runs[-1]
    qtbot.waitUntil(lambda: not run.active, timeout=300_000)
    return run, gui.open_analysis()


def cell_texts(tab):
    m = tab.table.model()
    return [[m.data(m.index(i, j)) for j in range(m.columnCount())] for i in range(m.rowCount())]


def test_cccc_benchmark_case_shows_all_references_at_full_precision(qtbot, gui):
    run, a = run_case(qtbot, gui)
    o = run.outcome
    w0 = o.w.copy()
    index = SOLVER_TABS.index("Сравнение")
    assert a.tabs.isTabEnabled(index) and a.tabs.widget(index) is a.comparison
    a.tabs.setCurrentIndex(index)
    c = a.comparison.comparison
    assert c.state == bm.AVAILABLE and [r.reference.kind for r in c.rows] == ["published", "D2", "FEM"]
    shown = cell_texts(a.comparison)
    for row, r in zip(shown, c.rows, strict=True):
        assert row[1] == repr(r.reference.beta) and row[3] == repr(r.delta) and row[4] == repr(r.rel_pct)
        assert float(row[3]) == r.delta                  # the text round-trips to the exact binary64 value
    assert shown[0][2] == "—"                            # the published table gives beta only
    text = " ".join(lab.text() for lab in a.comparison.findChildren(QLabel))
    assert "той же дискретной задачи" in text and "другая дискретизация" in text
    assert "порога согласия" in text and "порядка сходимости" in text and "не всегда положителен" in text
    assert o.mark in text and o.status in text                       # the solver statuses, unchanged
    for word in VERDICTS:
        assert all(word not in cell for row in shown for cell in row)
    assert (o.w == w0).all() and not o.w.flags.writeable
    assert full(c.beta) == repr(c.beta)


def test_full_provenance_is_readable_in_the_tooltip(qtbot, gui):
    """O-2: the provenance column is cut by the window width; its tooltip gives the full read-only text."""
    from PySide6.QtCore import Qt

    run, a = run_case(qtbot, gui)
    m = a.comparison.table.model()
    for i, r in enumerate(a.comparison.comparison.rows):
        tip = m.data(m.index(i, bm.DETAIL_COLUMN), Qt.ItemDataRole.ToolTipRole)
        assert tip == f"{r.reference.title}\n{r.reference.detail}"
        assert m.data(m.index(i, 3), Qt.ItemDataRole.ToolTipRole) == repr(r.delta)
    assert a.comparison.table.editTriggers() == a.comparison.table.EditTrigger.NoEditTriggers


def test_saved_report_and_csv_hold_the_table_shown_in_the_tab(qtbot, gui, tmp_path):
    """GUI-BENCHMARK-REPORT-01: the package saved from the analysis window repeats the tab, value by value."""
    import csv
    import html
    import json

    run, a = run_case(qtbot, gui)
    shown = cell_texts(a.comparison)
    paths = a.save_package(tmp_path / "saved")
    text = paths["report"].read_text(encoding="utf-8")
    for row in shown:
        assert "<tr>" + "".join(f"<td>{html.escape(t)}</td>" for t in row) + "</tr>" in text
    lines = [x for x in paths["comparison"].read_text(encoding="utf-8").splitlines() if not x.startswith("#")]
    rows = list(csv.DictReader(lines))
    assert [[r["beta_reference [1]"], r["delta_beta [1]"], r["delta_beta_relative [%]"]] for r in rows] == \
        [[row[1], row[3], row[4]] for row in shown]
    assert [r["reference_source_provenance"] for r in rows] == [row[5] for row in shown]
    data = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert data["files"]["comparison"] == "comparison.csv" and data["statuses"]["mark"] == run.outcome.mark


def test_case_without_reference_data_states_it_without_numbers(qtbot, gui, tmp_path):
    run, a = run_case(qtbot, gui, b=1.5)
    c = a.comparison.comparison
    assert c.state == bm.UNAVAILABLE and a.comparison.table is None and c.beta is None
    assert "Эталонных данных для этих входных данных нет" in a.comparison.reason.text()
    assert a.tabs.isTabEnabled(SOLVER_TABS.index("Сравнение"))
    paths = a.save_package(tmp_path / "saved")  # saving works without references
    assert "comparison" not in paths and c.reason.split(":")[0] in paths["report"].read_text(encoding="utf-8")


def test_protective_stop_is_not_compared(qtbot, gui):
    """CCCC N = 2, s = 2 on a grid above 32: STOPPED_RANK_GUARD_UNCERTIFIED with the s = 1 field as diagnostics."""
    run, a = run_case(qtbot, gui, N=2, n_x=34, n_y=34, half_steps=2)
    o = run.outcome
    assert o.diagnostic_only and o.status == "STOPPED_RANK_GUARD_UNCERTIFIED"
    c = a.comparison.comparison
    assert c.state == bm.DIAGNOSTIC and a.comparison.table is None and c.beta is None and not c.rows
    assert "диагностическое" in a.comparison.state_label.text()


def test_damaged_data_are_reported_in_the_tab(qtbot, gui):
    run, _ = run_case(qtbot, gui)
    tab = ComparisonTab(run.outcome, run, text='{"payload": {"schema": "kkvim-gui-benchmarks/1"}, "payload_sha256": "0"}')
    qtbot.addWidget(tab)
    assert tab.comparison.state == bm.DATA_ERROR and tab.table is None
    assert "повреждены" in tab.state_label.text()


def test_demonstration_window_keeps_the_tab_disabled(qtbot, qapp):
    w = AnalysisWindow(Session(settings=None))
    qtbot.addWidget(w)
    assert not w.tabs.isTabEnabled(TABS.index("Сравнение"))
