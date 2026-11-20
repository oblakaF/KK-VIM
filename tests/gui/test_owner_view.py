"""Node inspection on the deflection map and the side view of the mid-surface (GUI-OWNER-VIEW-02).

- a real mouse click on the colour map selects the nearest EXISTING node (checked against a brute-force search over all
  nodes) and shows its stored x, y and w without interpolation; the mark survives the colour scale, the quantity, the
  figure language and the theme;
- the side view draws exactly the stored node row (x at y_j) or column (y at x_i), the undeformed mid-surface w = 0, the
  maximum of the section; ×1 locks equal axes, a larger factor multiplies only the drawn w; positive w is drawn down;
- nothing of this starts a calculation; a diagnostic field keeps its diagnostic label.
"""

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")
pytest.importorskip("pyqtgraph")

from PySide6.QtCore import QPoint, QPointF, Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402

from kkvim_gui.analysis.window import AnalysisWindow  # noqa: E402
from kkvim_gui.main_window.window import MainWindow  # noqa: E402
from kkvim_gui.session import Session  # noqa: E402
from tests.gui.test_plate_workflow_ux import InlineNavier, InlineProcess, finish, plate  # noqa: E402


@pytest.fixture
def main(qtbot, tmp_path):
    """The main window with the solver and the Navier worker evaluated in the test process."""
    w = MainWindow(Session(settings=None), protocol_dir=tmp_path / "runs", process_factory=InlineProcess)
    w.runs.navier_factory = InlineNavier
    qtbot.addWidget(w)
    w.set_state("setup")
    return w


def counting(main):
    started = []
    process, navier = main.runs.process_factory, main.runs.navier_factory
    main.runs.process_factory = lambda request: started.append(request) or process(request)
    main.runs.navier_factory = lambda inputs: started.append(inputs) or navier(inputs)
    return started


def analysed(main, qtbot, **texts):
    plate(main.setup_model, "16", "8")
    for key, text in texts.items():
        main.setup_model.set_text(key, text)
    run = main._start_run()
    finish(main, qtbot)
    window = AnalysisWindow(main.session, main.run_rows(), run=run)
    qtbot.addWidget(window)
    window.resize(1440, 900)
    window.show()
    window.tabs.setCurrentWidget(window.fields)
    qtbot.waitExposed(window)
    qtbot.wait(50)
    return run, window, window.fields


def click(fig, x, y):
    box = fig.plot.getViewBox()
    point = fig.plot.mapFromScene(box.mapViewToScene(QPointF(x, y)))
    QTest.mouseClick(fig.plot.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                     QPoint(int(point.x()), int(point.y())))


def nearest(outcome, x, y):
    """Independent brute force: the node of the smallest distance among all (x_i, y_j)."""
    X, Y = np.meshgrid(outcome.x, outcome.y, indexing="ij")
    k = int(np.argmin(np.hypot(X - x, Y - y)))
    return np.unravel_index(k, X.shape)


def test_a_click_selects_the_nearest_existing_node_with_its_stored_values(main, qtbot):
    run, window, tab = analysed(main, qtbot)
    started = counting(main)
    o, fig = run.outcome, tab.figure
    box = fig.plot.getViewBox()
    for x, y in ((0.53, 0.31), (1.0, 0.5), (0.07, 0.93), (1.96, 0.04), (1.31, 0.66)):
        click(fig, x, y)
        # where the click actually landed (pixel rounding), mapped back exactly as the figure does
        landed = box.mapSceneToView(fig.plot.mapToScene(fig.plot.mapFromScene(box.mapViewToScene(QPointF(x, y)))))
        i, j = nearest(o, landed.x(), landed.y())
        assert fig.selected == (i, j), (x, y)
        text = tab.node_info.text()
        assert f"узел i = {i}, j = {j}" in text and f"x = {float(o.x[i])!r} m" in text and f"y = {float(o.y[j])!r} m" in text
        assert f"w = {float(o.w[i, j])!r} m" in text and "без интерполяции" in text
        assert fig.pick_outer.data["x"].tolist() == [float(o.x[i])] and fig.pick_outer.data["y"].tolist() == [float(o.y[j])]
        assert tab.side_at_y.index() == j and tab.side_at_x.index() == i  # the side view follows the node
    selected = fig.selected
    click(fig, 2.3, 0.5)  # outside the plate: nothing changes
    assert fig.selected == selected
    tab.cmap.setCurrentText("cividis")
    tab.quantity.set_current("w_bar")
    tab.figure.set_quantity("w_bar")
    main.session.set_figure_language("ru" if main.session.figure_language == "en" else "en")
    main.session.set_theme("dark", persist=False)
    assert fig.selected == selected and fig.pick_outer.data["x"].tolist() == [float(o.x[selected[0]])]
    assert started == []


def test_side_view_draws_the_stored_row_or_column_at_every_factor(main, qtbot):
    run, window, tab = analysed(main, qtbot)
    started = counting(main)
    o, side = run.outcome, tab.side
    tab.set_view("side")
    assert tab.stack.currentWidget() is side and tab.side_bar.isVisible() and not tab.node_info.isVisible()
    assert side.plot.getViewBox().yInverted()  # w > 0 is drawn downward (MATH_SPEC.md 8.1)
    for j in (4, 2, 7):
        tab.side_at_y.set_index(j)
        xs, ys = side.profile.getData()
        assert np.array_equal(xs, o.x) and np.array_equal(ys, o.w[:, j])  # ×1: the stored values themselves
        assert side.zero.getData()[1].tolist() == [0.0, 0.0] and side.zero.getData()[0].tolist() == [o.x[0], o.x[-1]]
        k = int(np.argmax(np.abs(o.w[:, j])))
        assert side.max_point.data["x"].tolist() == [float(o.x[k])] and side.max_point.data["y"].tolist() == [float(o.w[k, j])]
        assert f"(j = {j})" in tab.side_info.text() and f"i = {k}" in tab.side_info.text()
    assert side.plot.getViewBox().state["aspectLocked"] == 1.0  # ×1: equal scale of both axes
    for index, factor in enumerate(side.FACTORS):
        tab.side_factor.setCurrentIndex(index)
        xs, ys = side.profile.getData()
        assert side.factor == factor and np.array_equal(ys, factor * o.w[:, 7])  # the same numbers, only drawn larger
        assert f"× {factor:g}" in side.legend.items[1][1].text and f"w × {factor:g}, m" in side.plot.getAxis("left").labelText
        assert (side.plot.getViewBox().state["aspectLocked"] == 1.0) is (factor == 1.0)
    tab.side_direction.set_current("y")
    tab._side_direction("y")
    tab.side_at_x.set_index(5)
    xs, ys = side.profile.getData()
    assert np.array_equal(xs, o.y) and np.array_equal(ys, side.factor * o.w[5, :])
    tab.side_thickness.setChecked(True)
    h = o.request.t
    assert side.upper.getData()[1].tolist() == [-h / 2, -h / 2] and side.lower.getData()[1].tolist() == [h / 2, h / 2]
    assert any(word in side.legend.items[-1][1].text for word in ("схема", "schematic"))  # labelled, never a computation
    main.session.set_figure_language("ru")
    assert "вниз" in side.plot.getAxis("left").labelText and "вдоль y" in side.title.text()
    assert started == []  # sections, factors, language: nothing computed


def test_a_diagnostic_field_keeps_its_label_in_the_node_and_side_texts(main, qtbot):
    run, window, tab = analysed(main, qtbot, s_max="5")
    o = run.outcome
    assert o.status == "MAX_HALF_STEPS" and o.diagnostic_only
    click(tab.figure, 1.0, 0.5)
    assert tab.node_info.text().startswith("ДИАГНОСТИКА") and "не результат" in tab.node_info.text()
    tab.set_view("side")
    assert tab.side_info.text().startswith("ДИАГНОСТИКА") and tab.side.mark.text()  # the figure keeps its mark
