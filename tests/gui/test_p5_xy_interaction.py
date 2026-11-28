"""Snap, «Проекции» and the fixed point on «Нагрузка — прогиб» (GUI_DESIGN.md 16.4–16.5, G6–G12, G16; GUI-P5-XY-INTERACTION-01).

The geometry is checked on its own (display coordinates, hysteresis, single and empty polylines); the figure is checked on
real load series computed by the KK-VIM adapter in the test process (as in test_series_raw_values): the marker on the
drawn line of the active visible curve only, interpolation on the drawn segment and labelled as such, no marker beyond
the threshold, no jumps along the curve, projections from the cursor or the marker, a click fixes and a drag does not,
and nothing of it reaches the series data, «Исходные значения», the saved CSV or figure, or a solver.
"""

import math
import sys
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")
pytest.importorskip("pyqtgraph")

from PySide6.QtCore import QPoint, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QCursor  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QScrollArea  # noqa: E402

from kkvim_gui.analysis import xy_interaction as xy  # noqa: E402
from kkvim_gui.model.series import series_data  # noqa: E402
from kkvim_gui.theme.tokens import THEMES  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from test_series_raw_values import counting, main, mid_segment, right_click, run_series, shown_tab  # noqa: E402, F401


# ---------------------------------------------------------------- geometry alone

def test_snap_geometry_in_display_coordinates():
    line = [(0.0, 0.0), (100.0, 0.0), (100.0, 10.0), (0.0, 10.0)]  # a hairpin: segments 0 and 2 are 10 px apart
    s = xy.snap(line, (40.0, 3.0))
    assert (s.segment, s.node, s.t, s.distance) == (0, None, 0.4, 3.0)  # interpolation on the segment, G7
    assert s.describe() == "интерполяция между узлами 1 и 2" and s.figure_words("en") == "interpolated 1–2"
    near = xy.snap(line, (98.0, -2.0))
    assert (near.node, near.t, near.describe(), near.figure_words("ru")) == (1, 1.0, "узел 2", "узел 2")
    assert xy.snap(line, (40.0, -20.5)) is None and xy.snap(line, (40.0, -19.5)) is not None  # threshold 20 px, G8
    # hysteresis, G9: from segment 0 the closer branch (segment 2) wins only when closer by more than 6 px
    assert xy.snap(line, (50.0, 5.5), previous=0).segment == 0
    assert xy.snap(line, (50.0, 5.5)).segment == 2
    assert xy.snap(line, (50.0, 9.0), previous=0).segment == 2
    single = xy.snap([(5.0, 5.0)], (8.0, 9.0))
    assert (single.segment, single.node, single.distance) == (0, 0, 5.0)
    assert xy.snap([(5.0, 5.0)], (40.0, 40.0)) is None and xy.snap([], (0.0, 0.0)) is None
    assert xy.format_value(2.5, "ru") == "2,5" and xy.format_value(2.5, "en") == "2.5"


# ---------------------------------------------------------------- the figure

def scene_of(fig, x, y):
    return fig.plot.getViewBox().mapViewToScene(QPointF(x, y))


def pixel(fig, scene):
    p = fig.plot.mapFromScene(scene)
    return QPoint(round(p.x()), round(p.y()))


def offset(fig, key, segment, t, px):
    """Scene point at t on the drawn segment of the curve key, moved px logical pixels across the segment."""
    xs, ys = fig._curve(key)
    a, b = scene_of(fig, xs[segment], ys[segment]), scene_of(fig, xs[segment + 1], ys[segment + 1])
    d = b - a
    length = math.hypot(d.x(), d.y())
    return a + d * t + QPointF(-d.y() / length, d.x() / length) * px


def on_segment(fig, m, key):
    """Pixel distance of the measurement position from the drawn segment of its snap (0: on the line)."""
    xs, ys = fig._curve(key)
    k = m.snap.segment
    a, b, p = scene_of(fig, xs[k], ys[k]), scene_of(fig, xs[k + 1], ys[k + 1]), scene_of(fig, m.x, m.y)
    cross = (b.x() - a.x()) * (p.y() - a.y()) - (b.y() - a.y()) * (p.x() - a.x())
    return abs(cross) / math.hypot(b.x() - a.x(), b.y() - a.y())


@pytest.fixture
def figure(main, qtbot):  # noqa: F811
    series = run_series(main, qtbot)  # SSSS N = 1, q = 1; 5; 2.5 kPa: KK-VIM, VIM-1…3 coincide, Navier close
    started = counting(main)
    tab, fig = shown_tab(main, qtbot, series)
    return main, series, tab, fig, started


def test_G6_the_marker_is_only_on_the_active_visible_curve(figure):
    owner, series, tab, fig, started = figure
    data, ia = tab.data, fig.interaction
    keys = [ia.combo.itemData(i) for i in range(1, ia.combo.count())]
    assert keys == ["kkvim", 1, 2, 3, "navier"] and ia.active == "kkvim"  # only drawn curves are offered
    ia.set_snap(True)
    pos = offset(fig, "kkvim", 0, 0.5, 5)
    for key in ("kkvim", 1, "navier"):  # coinciding curves are chosen one by one
        ia.combo.activated.emit(ia.combo.findData(key))
        ia.move_to(pos)
        assert ia.active == key and ia.measurement.key == key and ia.marker.isVisible()
    tab.toggles["navier"].setChecked(False)  # the active curve is hidden: no Snap until another is chosen
    assert ia.active is None and ia.combo.currentIndex() == 0 and "navier" not in [
        ia.combo.itemData(i) for i in range(ia.combo.count())]
    ia.move_to(pos)
    assert ia.measurement is None and not ia.marker.isVisible() and "выберите активную кривую" in fig.readout.text()
    tab.toggles["navier"].setChecked(True)
    assert ia.active is None  # showing it again does not choose it
    ia.combo.activated.emit(ia.combo.findData(2))
    ia.move_to(pos)
    assert ia.measurement.key == 2
    assert fig.source_records("navier") == [p for p in data.points if p.navier_w_m is not None]  # hiding kept the data
    assert series_data(series, owner.runs) == data and started == []


def test_G7_G8_interpolation_on_the_drawn_segment_and_the_threshold(figure):
    owner, series, tab, fig, started = figure
    ia = fig.interaction
    ia.set_snap(True)
    xs, ys = fig._curve("kkvim")
    ia.move_to(offset(fig, "kkvim", 0, 0.4, 5))
    m = ia.measurement
    assert m.snap.node is None and m.snap.segment == 0 and 0 < m.snap.t < 1 and abs(m.snap.distance - 5) < 1e-6
    assert on_segment(fig, m, "kkvim") < 1e-6 and xs[0] < m.x < xs[1]  # on the drawn segment, between the nodes
    assert m.x not in xs and m.y not in ys  # a graphical position, not a stored value
    assert "интерполяция между узлами 1 и 2" in fig.readout.text() and "не значение решателя" in fig.readout.text()
    ia.move_to(offset(fig, "kkvim", 0, 1.0, 2))  # on the node 2 (2.5 kPa) within its drawn symbol
    assert (ia.measurement.snap.node, ia.measurement.x, ia.measurement.y) == (1, xs[1], ys[1])
    assert "узел 2 —" in fig.readout.text()
    ia.move_to(offset(fig, "kkvim", 0, 0.5, 15))
    assert ia.measurement is not None
    ia.set_projections(True)
    ia.move_to(offset(fig, "kkvim", 0, 0.5, 25))  # farther than 20 px: no marker and no projection
    assert ia.measurement is None and not ia.marker.isVisible()
    assert not any(item.isVisible() for item in ia.projection.items())
    assert "маркера нет" in fig.readout.text() and fig.readout.text().startswith("Курсор: q = ")
    ia.move_to(None)
    assert fig.readout.text() == ""  # outside the plot the field is empty (16.2)
    assert started == []


def test_G9_the_marker_follows_the_curve_without_jumps(figure):
    _main, _series, _tab, fig, _started = figure
    ia = fig.interaction
    ia.set_snap(True)
    xs, ys = fig._curve("kkvim")
    track, segments = [], []
    for k in range(len(xs) - 1):
        for i in range(41):
            ia.move_to(offset(fig, "kkvim", k, i / 40, 3 if i % 2 else -3))  # a hand along the line, ±3 px
            m = ia.measurement
            assert m is not None and m.key == "kkvim"
            track.append(scene_of(fig, m.x, m.y))
            segments.append(m.snap.segment)
    assert segments == sorted(segments)  # never back to an earlier segment
    lengths = [math.hypot(b.x() - a.x(), b.y() - a.y()) for a, b in zip(track, track[1:])]
    step = max(math.hypot((scene_of(fig, xs[k + 1], ys[k + 1]) - scene_of(fig, xs[k], ys[k])).x(),
                          (scene_of(fig, xs[k + 1], ys[k + 1]) - scene_of(fig, xs[k], ys[k])).y()) / 40
               for k in range(len(xs) - 1))
    assert max(lengths) <= step + 1e-6  # every move of the marker is at most one step of the hand


def test_projections_follow_the_cursor_or_the_marker_view_language_and_theme(figure):
    owner, series, tab, fig, started = figure
    ia, session = fig.interaction, owner.session
    view_box = fig.plot.getViewBox()
    pos = offset(fig, "kkvim", 0, 0.5, 8)
    ia.move_to(pos)
    assert not any(item.isVisible() for item in ia.projection.items())  # off by default
    ia.set_projections(True)  # without Snap: from the cursor
    cx, cy = ia.cursor()
    vertical, horizontal = ia.projection.lines
    (x0, _), (y0, _) = fig.view_range()
    assert list(vertical.getData()[0]) == [cx, cx] and list(vertical.getData()[1]) == [cy, y0]
    assert list(horizontal.getData()[0]) == [cx, x0] and list(horizontal.getData()[1]) == [cy, cy]
    assert vertical.opts["pen"].style() == Qt.PenStyle.DashLine
    x_text, y_text = (t.toPlainText() for t in ia.projection.texts)
    assert x_text.startswith("q = ") and x_text.endswith(" kPa") and y_text.startswith("w = ") and y_text.endswith(" mm")
    ia.set_snap(True)  # with Snap: from the marker
    m = ia.measurement
    assert list(vertical.getData()[0]) == [m.x, m.x] and list(horizontal.getData()[1]) == [m.y, m.y]
    view_box.scaleBy((0.5, 0.5))  # zoom: the same pointer is measured again in the new view
    (x0, _), (y0, _) = fig.view_range()
    assert vertical.getData()[1][1] == y0 and horizontal.getData()[0][1] == x0
    assert ia.measurement is not None and on_segment(fig, ia.measurement, "kkvim") < 1e-6
    session.set_figure_language("ru")
    assert "," in ia.projection.texts[0].toPlainText() and "." not in ia.projection.texts[0].toPlainText()
    session.set_figure_language("en")
    assert "," not in ia.projection.texts[0].toPlainText()
    other = next(name for name in THEMES if name != session.theme)
    session.set_theme(other, persist=False)
    assert vertical.opts["pen"].color().name().lower() == THEMES[other]["text.primary"].lower()
    assert series_data(series, owner.runs) == tab.data and started == []


def test_G12_a_click_fixes_the_point_and_a_drag_does_not(figure, qtbot):
    owner, series, tab, fig, started = figure
    ia, viewport = fig.interaction, fig.plot.viewport()
    first = pixel(fig, offset(fig, "kkvim", 0, 0.4, 4))
    QTest.mouseClick(viewport, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, first)
    assert ia.fixed is None  # neither Snap nor Projections: a click only selects, as before
    ia.set_snap(True)
    QTest.mouseClick(viewport, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, first)
    fixed = ia.fixed
    assert fixed.key == "kkvim" and fixed.snap.node is None and ia.unfix_button.isEnabled()
    assert ia.fixed_marker.isVisible() and all(item.isVisible() for item in ia.fixed_projection.items())
    assert "Фиксация: Snap" in fig.readout.text()
    second = pixel(fig, offset(fig, "kkvim", 1, 0.5, 3))
    QTest.mouseClick(viewport, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, second)
    assert ia.fixed.snap.segment == 1 and ia.fixed != fixed  # another click moves the fixation
    moved = ia.fixed
    view = fig.view_range()
    # a small slip (6 px, more than 4 px): not a click
    QTest.mousePress(viewport, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, first)
    qtbot.wait(30)
    QTest.mouseMove(viewport, first + QPoint(6, 0))
    qtbot.wait(30)
    QTest.mouseRelease(viewport, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, first + QPoint(6, 0))
    assert ia.fixed == moved
    # a drag: the graph is moved, nothing is fixed
    QTest.mousePress(viewport, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, first)
    for i in range(1, 9):
        qtbot.wait(30)
        QTest.mouseMove(viewport, first + QPoint(10 * i, 5 * i))
    qtbot.wait(30)
    QTest.mouseRelease(viewport, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, first + QPoint(80, 40))
    assert fig.view_range() != view and ia.fixed == moved
    # the fixed point stays where it was on the curve when the view moves
    assert (ia.fixed_marker.data["x"][0], ia.fixed_marker.data["y"][0]) == (moved.x, moved.y)
    QTest.keyClick(fig.plot, Qt.Key.Key_Escape)
    assert ia.fixed is None and not ia.fixed_marker.isVisible() and not ia.unfix_button.isEnabled()
    ia.set_projections(True)
    ia.set_snap(False)  # projections from the cursor: a click fixes the cursor position
    QTest.mouseClick(viewport, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, second)
    assert ia.fixed is not None and ia.fixed.snap is None
    ia.unfix_button.click()
    assert ia.fixed is None
    assert series_data(series, owner.runs) == tab.data and started == []


def test_the_overlay_never_reaches_values_saved_files_or_a_solver(figure, qtbot, tmp_path, monkeypatch):
    owner, series, tab, fig, started = figure
    data, ia = tab.data, fig.interaction
    moves = []
    monkeypatch.setattr(QCursor, "setPos", lambda *a: moves.append(a))
    clean_csv = tab.save(tmp_path / "clean")["load_series"].read_bytes()
    ia.set_snap(True)
    ia.set_projections(True)
    ia.move_to(offset(fig, "kkvim", 0, 0.4, 4))
    ia.fix()
    ia.move_to(offset(fig, "kkvim", 1, 0.3, 4))
    assert ia.fixed is not None and ia.marker.isVisible()
    # W5–W6: the right click still offers the curves under the pointer, and the values are the stored records
    right_click(fig, mid_segment(fig, "kkvim"))
    menu = fig.values_menu
    choice = menu.actions()[0].menu()
    assert [a.text() for a in choice.actions()] == [fig.curve_name(k) for k in fig.curves_at(mid_segment(fig, "kkvim"))]
    menu.close()
    window = fig.show_values("kkvim")
    qtbot.addWidget(window)
    w_col = window.headers.index("w в центре КК-ВИМ, m")
    assert [r[w_col] for r in window.rows] == [p.w_centre_m for p in data.included]
    assert ia.fixed.y / 1e3 not in [r[w_col] for r in window.rows]
    # G16: the saved CSV and figure are the same with the overlay as without it
    saved = tab.save(tmp_path / "overlay")
    assert saved["load_series"].read_bytes() == clean_csv
    with_overlay = fig.export_pixmap().toImage()
    assert ia.fixed is not None and ia.marker.isVisible() and ia.controls.isVisible()  # the screen keeps them
    assert fig.grab().toImage() == with_overlay  # the report package grabs the figure: also without the overlay
    assert fig.grab(fig.rect()).toImage() != with_overlay  # on the screen the overlay is drawn
    ia.clear_fixed()
    ia.set_snap(False)
    ia.set_projections(False)
    ia.move_to(None)
    assert with_overlay == fig.export_pixmap().toImage()  # the saved figure is the same as without any overlay
    assert series_data(series, owner.runs) == data and started == [] and moves == []  # G10: the pointer is never moved


def test_a_curve_of_one_point_snaps_to_its_node_and_an_empty_curve_is_not_offered(qtbot):
    """A load series has at least two loads, so a one-point and an empty curve are drawn on a plain plot here."""
    import pyqtgraph as pg

    from kkvim_gui.session import Session

    session = Session(settings=None)
    plot = pg.PlotWidget()
    qtbot.addWidget(plot)
    plot.resize(600, 400)
    plot.show()
    qtbot.waitExposed(plot)
    one, empty = plot.plot([2.0], [3.0], symbol="o"), plot.plot([], [])
    plot.getViewBox().setRange(xRange=(0, 4), yRange=(0, 6), padding=0)
    ia = xy.XYInteraction(plot, lambda: [("one", "one point", one), ("empty", "no point", empty)],
                          lambda: (("q", "kPa"), ("w", "mm")), lambda: "en", lambda: session.tokens)
    assert [ia.combo.itemData(i) for i in range(ia.combo.count())] == [None, "one"]
    ia.set_active("empty")
    assert ia.active is None  # a curve without points cannot be active
    ia.set_active("one")
    ia.set_snap(True)
    node = plot.getViewBox().mapViewToScene(QPointF(2.0, 3.0))
    ia.move_to(node + QPointF(6, -6))
    assert (ia.measurement.snap.node, ia.measurement.x, ia.measurement.y) == (0, 2.0, 3.0)
    assert "node 1" in ia._label(ia.measurement) and "узел 1 —" in ia.readout()
    ia.move_to(node + QPointF(30, 0))
    assert ia.measurement is None and not ia.marker.isVisible()


@pytest.mark.parametrize("width", [960, 1400])
def test_the_coordinate_field_never_widens_the_figure(figure, qtbot, width):
    """QA 8.1: at the minimum window width (960, GUI_DESIGN.md 19) the long Snap / fixation text is wrapped inside the
    figure; showing or hiding it changes neither the plot size nor the window, and the whole text stays readable.

    Before the correction the unwrapped field set the figure minimum: at 1400 the plot grew from 1340 to 2136 px and
    a 960 window could not be kept (the active-curve list with its long names alone needed 930 px)."""
    owner, series, tab, fig, started = figure
    ia = fig.interaction
    width = max(width, tab.minimumSizeHint().width())  # offscreen fonts: the tab toolbar may need a little more
    tab.resize(width, 700)
    qtbot.waitUntil(lambda: tab.width() == width)

    def sizes():
        QTest.qWait(20)
        bars = [a for a in tab.findChildren(QScrollArea) if a.horizontalScrollBar().isVisible()]
        return tab.width(), fig.minimumSizeHint().width(), fig.plot.width(), fig.plot.height(), bars

    ia.move_to(None)
    empty = sizes()
    ia.set_snap(True)
    ia.set_projections(True)
    ia.move_to(offset(fig, "kkvim", 0, 0.4, 5))  # unfixed: the cursor and the Snap lines
    assert "интерполяция" in fig.readout.text() and sizes() == empty
    ia.fix()  # fixed: the cursor, the Snap and the fixation lines (the longest text)
    ia.move_to(offset(fig, "kkvim", 1, 0.5, 5))
    assert ia.fixed is not None and fig.readout.text().count("\n") == 2 and sizes() == empty
    lines = fig.readout.text().split("\n")
    assert fig.readout.height() >= fig.readout.heightForWidth(fig.readout.width())  # every wrapped line is shown
    assert fig.readout.wordWrap() and fig.readout.toolTip() == fig.readout.text() and len(lines) == 3
    ia.move_to(None)  # the Snap text disappears, the fixation stays
    assert sizes() == empty
    ia.clear_fixed()
    assert fig.readout.text() == "" and sizes() == empty and fig.width() <= width
    # the active-curve list is narrower than its longest name, which the open list and the tooltip show whole
    longest = max(ia.combo.view().fontMetrics().horizontalAdvance(ia.combo.itemText(i)) for i in range(ia.combo.count()))
    assert ia.combo.width() < longest < ia.combo.view().minimumWidth()
    assert ia.combo.currentText() in ia.combo.toolTip()
    assert series_data(series, owner.runs) == tab.data and started == []
