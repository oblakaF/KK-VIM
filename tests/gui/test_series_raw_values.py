"""«Показать исходные значения» on the «Нагрузка — прогиб» graph (GUI_DESIGN.md 20.3, GUI-P5-SERIES-RAW-VALUES-01).

Real load series computed by the KK-VIM adapter in the test process: a right click with the mouse on a drawn curve offers
the command (several coinciding curves by name), the window shows the stored records of that curve in their stored order
and exact values, copying gives those values back, marks are kept, hidden or unavailable curves offer nothing, and no
calculation is started.
"""

import json

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")
pytest.importorskip("pyqtgraph")

from PySide6.QtCore import QItemSelectionModel, QPoint, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QGuiApplication  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QAbstractItemView  # noqa: E402

from kkvim_gui.analysis.series import SHOW_VALUES, SeriesTab  # noqa: E402
from kkvim_gui.backend import kkvim_adapter, navier_job  # noqa: E402
from kkvim_gui.backend.process import INPUT, OK  # noqa: E402
from kkvim_gui.main_window.window import MainWindow  # noqa: E402
from kkvim_gui.model.series import series_data  # noqa: E402
from kkvim_gui.session import Session  # noqa: E402


class InlineProcess:
    def __init__(self, request):
        self.request, self.message = request, None

    def poll(self):
        if self.message is None:
            try:
                self.message = (OK, kkvim_adapter.execute(self.request))
            except kkvim_adapter.AdapterInputError as exc:
                self.message = (INPUT, str(exc))
        return self.message

    def terminate(self, timeout=2.0):
        return True


class InlineNavier:
    def __init__(self, inputs):
        self.inputs = inputs

    def poll(self):
        from kkvim_navier_worker import compute

        return navier_job.parse_output(json.dumps(compute(self.inputs)), self.inputs)

    def terminate(self, timeout=2.0):
        return True


@pytest.fixture
def main(qtbot, tmp_path):
    w = MainWindow(Session(settings=None), protocol_dir=tmp_path / "runs", process_factory=InlineProcess)
    w.runs.navier_factory = InlineNavier
    qtbot.addWidget(w)
    w.set_state("setup")
    return w


def run_series(main, qtbot, n=1, loads="1; 5; 2,5"):
    m = main.setup_model
    for key, text in (("a", "2"), ("b", "1"), ("h", "10"), ("E", "210"), ("nu", "0,3"), ("q", "5"), ("n_x", "16"),
                      ("n_y", "8")):
        m.set_text(key, text)
    if n != 1:
        m.set_choice(N=n)
    m.set_choice(load_mode="series")
    m.set_text("q_series", loads)
    series = main._start_series(m.state)
    qtbot.waitUntil(lambda: main.runs.active is None and not main.runs.references_pending
                    and not any(s.active for s in main.runs.series), timeout=60000)
    return series


def counting(main):
    started = []
    process, navier = main.runs.process_factory, main.runs.navier_factory
    main.runs.process_factory = lambda request: started.append(request) or process(request)
    main.runs.navier_factory = lambda inputs: started.append(inputs) or navier(inputs)
    return started


def shown_tab(main, qtbot, series):
    tab = SeriesTab(main.session, series_data(series, main.runs))
    qtbot.addWidget(tab)
    tab.resize(1400, 900)
    tab.show()
    qtbot.waitExposed(tab)
    return tab, tab.figure


def mid_segment(fig, key):
    """Scene point in the middle of the first drawn segment of the curve (between two stored points)."""
    xs, ys = fig._curve(key)
    return fig.plot.getViewBox().mapViewToScene(QPointF((xs[0] + xs[1]) / 2, (ys[0] + ys[1]) / 2))


def right_click(fig, scene):
    pixel = fig.plot.mapFromScene(scene)
    QTest.mouseClick(fig.plot.viewport(), Qt.MouseButton.RightButton, Qt.KeyboardModifier.NoModifier,
                     QPoint(int(pixel.x()), int(pixel.y())))


def test_right_click_on_a_curve_shows_and_copies_its_stored_values(main, qtbot):
    series = run_series(main, qtbot)
    started = counting(main)
    tab, fig = shown_tab(main, qtbot, series)
    data = tab.data
    right_click(fig, mid_segment(fig, "kkvim"))
    menu = fig.values_menu
    assert menu is not None and menu.actions()[0].text() == SHOW_VALUES
    # SSSS N = 1 converges at s = 2: VIM-1…3 and Navier coincide with KK-VIM there; the user chooses, KK-VIM first
    choice = menu.actions()[0].menu()
    names = [a.text() for a in choice.actions()]
    assert names[0] == fig.curve_name("kkvim") and len(names) == len(fig.curves_at(mid_segment(fig, "kkvim"))) > 1
    menu.close()
    choice.actions()[0].trigger()
    window = fig.value_windows[-1]
    qtbot.addWidget(window)
    records = data.included
    assert [p.q for p in records] == [1000.0, 5000.0, 2500.0]  # stored (entered) order, not the drawing order
    w_col, x_col = window.headers.index("w в центре КК-ВИМ, m"), window.headers.index("x узла центра, m")
    for row, p in zip(window.rows, records, strict=True):
        assert (row[0], row[1], row[2], row[w_col]) == (p.index, p.run_id, p.q, p.w_centre_m)  # the record itself
        assert row[x_col] == p.centre["x"] == 1.0
    shown = window.model.index(0, w_col).data()
    assert shown == repr(records[0].w_centre_m).replace(".", ",")  # displayed digits are the copied digits
    assert window.view.editTriggers() == QAbstractItemView.EditTrigger.NoEditTriggers
    # copy: everything, then two selected cells with Ctrl+C
    lines = window.copy_all().splitlines()
    assert lines[0].startswith(f"# Серия {data.series_id} · SSSS") and lines[1].split("\t") == list(window.headers)
    assert [float(line.split("\t")[w_col]) for line in lines[2:]] == [p.w_centre_m for p in records]
    sel = window.view.selectionModel()
    for c in (2, w_col):
        sel.select(window.model.index(2, c), QItemSelectionModel.SelectionFlag.Select)
    QGuiApplication.clipboard().clear()
    window.activateWindow()
    qtbot.waitUntil(window.isActiveWindow, timeout=2000)
    window.view.setFocus()
    QTest.keyClick(window.view, Qt.Key.Key_C, Qt.KeyboardModifier.ControlModifier)
    copied = QGuiApplication.clipboard().text().splitlines()
    assert copied[-2].split("\t") == ["q, Pa", "w в центре КК-ВИМ, m"]
    assert [float(v) for v in copied[-1].split("\t")] == [2500.0, records[2].w_centre_m]
    assert started == []  # opening and copying never start the solver or the reference
    assert series_data(series, main.runs) == data  # the stored series is unchanged


def test_hidden_and_unavailable_curves_offer_no_values(main, qtbot):
    series = run_series(main, qtbot)
    started = counting(main)
    tab, fig = shown_tab(main, qtbot, series)
    scene = mid_segment(fig, "kkvim")
    for key in ("kkvim", 1, 2, 3):
        tab.toggles[key].setChecked(False)
    assert fig.curves_at(scene) == ["navier"]  # hidden curves are never under the pointer
    assert fig.show_values("kkvim") is None and fig.value_windows == []
    tab.toggles["navier"].setChecked(False)
    assert fig.curves_at(scene) == []
    right_click(fig, scene)
    assert fig.values_menu is None  # nothing invented: the plot keeps its own menu
    for box in tab.toggles.values():
        box.setChecked(True)
    tab.quantity.changed.emit("maxabs")  # the reference and the cycle states exist at the centre only
    assert fig.source_records("navier") == [] and fig.show_values("navier") is None
    assert fig.curves_at(mid_segment(fig, "kkvim")) == ["kkvim"]
    window = fig.show_values("kkvim")
    qtbot.addWidget(window)
    col = window.headers.index("w в узле max|w| КК-ВИМ (знаковый), m")
    assert [r[col] for r in window.rows] == [p.w_at_max_abs_m for p in tab.data.included]
    assert started == []


def test_marks_of_the_runs_stay_on_the_values_and_the_copy(main, qtbot):
    series = run_series(main, qtbot, n=2, loads="1; 2")
    started = counting(main)
    _tab, fig = shown_tab(main, qtbot, series)
    window = fig.show_values("kkvim")
    qtbot.addWidget(window)
    marks = {p.mark for p in window.records}
    assert marks == {"PROVISIONAL / NOT_QUALIFIED"} and window.marks == ("PROVISIONAL / NOT_QUALIFIED",)
    col = window.headers.index("Отметка")
    assert all(r[col] == "PROVISIONAL / NOT_QUALIFIED" for r in window.rows)
    assert "# PROVISIONAL / NOT_QUALIFIED" in window.copy_all().splitlines()
    assert started == []
