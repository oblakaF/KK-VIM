"""Contextual tooltips, F1, help centre, adaptive layout (GUI_DESIGN.md, 17 and 19)."""

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

from PySide6.QtCore import QEvent, QPoint, QPointF, QRect, QSize, Qt, QUrl  # noqa: E402
from PySide6.QtGui import QEnterEvent, QGuiApplication, QShortcut  # noqa: E402
from PySide6.QtWidgets import QApplication, QWidget  # noqa: E402

from kkvim_gui.app import GuiApplication  # noqa: E402
from kkvim_gui.help.articles import get  # noqa: E402
from kkvim_gui.help.tooltip import DELAY_MS, popup_position  # noqa: E402
from kkvim_gui.help.window import START_TOPIC  # noqa: E402
from kkvim_gui.main_window.setup_view import layout_mode  # noqa: E402
from kkvim_gui.session import Session  # noqa: E402
from kkvim_gui.widgets.common import InfoButton  # noqa: E402

CTRL = Qt.KeyboardModifier.ControlModifier


@pytest.fixture
def gui(qtbot, qapp):
    session = Session(settings=None)
    session.apply_theme()
    g = GuiApplication(session, demo=True)
    qtbot.addWidget(g.main_window)
    g.main_window.show()
    qtbot.waitExposed(g.main_window)
    yield g
    for w in list(g.analysis_windows):
        w.close()
    g.help.close()


def hover(widget):
    pos = QPointF(widget.width() / 2, widget.height() / 2)
    QApplication.sendEvent(widget, QEnterEvent(pos, pos, QPointF(widget.mapToGlobal(pos.toPoint()))))


def leave(widget):
    QApplication.sendEvent(widget, QEvent(QEvent.Type.Leave))


# Popup placement (pure geometry)

@pytest.mark.parametrize("target", [
    QRect(100, 100, 300, 32),   # room below
    QRect(100, 860, 300, 32),   # bottom of the screen: goes above
    QRect(100, 5, 300, 880),    # very tall: goes beside
])
def test_popup_never_covers_the_target_and_stays_on_screen(target):
    avail = QRect(0, 0, 1280, 900)
    size = QSize(300, 90)
    pos = popup_position(target, target.center(), size, avail)
    popup = QRect(pos, size)
    assert not popup.intersects(target)
    assert avail.contains(popup)


# Tooltips

def test_tooltip_appears_after_delay_next_to_the_field(qtbot, gui):
    w = gui.main_window
    w.set_state("setup")
    field = w.setup_view.a
    controller = gui.help.tooltips
    hover(field.edit)
    assert not controller.popup.isVisible()  # small delay first
    qtbot.waitUntil(controller.popup.isVisible, timeout=DELAY_MS + 1500)
    assert controller.popup.summary.text() == get("param.geometry.a").summary
    popup_rect = controller.popup.frameGeometry()
    field_rect = QRect(field.mapToGlobal(QPoint(0, 0)), field.size())
    assert not popup_rect.intersects(field_rect)
    assert controller.popup.testAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
    leave(field.edit)
    assert not controller.popup.isVisible()


@pytest.mark.parametrize(("getter", "topic"), [
    (lambda v: v.h, "param.geometry.h"),
    (lambda v: v.E, "param.material.E"),
    (lambda v: v.nu, "param.material.nu"),
    (lambda v: v.theory, "param.theory"),
])
def test_tooltips_for_geometry_material_and_theory(qtbot, gui, getter, topic):
    gui.main_window.set_state("setup")
    widget = getter(gui.main_window.setup_view)
    hover(widget)
    qtbot.waitUntil(gui.help.tooltips.popup.isVisible, timeout=DELAY_MS + 1500)
    assert gui.help.tooltips.visible_topic == topic


def test_typing_hides_the_tooltip_and_is_not_consumed(qtbot, gui):
    gui.main_window.set_state("setup")
    edit = gui.main_window.setup_view.b.edit
    edit.setFocus()
    edit.clear()
    hover(edit)
    qtbot.waitUntil(gui.help.tooltips.popup.isVisible, timeout=DELAY_MS + 1500)
    qtbot.keyClicks(edit, "1,5")
    assert edit.text() == "1,5"
    assert not gui.help.tooltips.popup.isVisible()


def test_standard_editing_keys_work_with_help_installed(qtbot, gui):
    gui.main_window.set_state("setup")
    edit = gui.main_window.setup_view.a.edit
    edit.setFocus()
    edit.clear()
    qtbot.keyClicks(edit, "-2.5e-3")
    for key, expected in ((Qt.Key.Key_A, "-2.5e-3"),):
        qtbot.keyClick(edit, key, CTRL)
        assert edit.selectedText() == expected
    qtbot.keyClick(edit, Qt.Key.Key_C, CTRL)
    assert QGuiApplication.clipboard().text() == "-2.5e-3"
    qtbot.keyClick(edit, Qt.Key.Key_X, CTRL)
    assert edit.text() == ""
    qtbot.keyClick(edit, Qt.Key.Key_Z, CTRL)
    assert edit.text() == "-2.5e-3"
    qtbot.keyClick(edit, Qt.Key.Key_A, CTRL)
    qtbot.keyClick(edit, Qt.Key.Key_V, CTRL)
    assert edit.text() == "-2.5e-3"
    assert gui.help.window is None  # no editing key opened the help


# F1

def test_f1_on_selected_parameter_opens_its_article(qtbot, gui):
    w = gui.main_window
    w.set_state("setup")
    edit = w.setup_view.E.edit
    edit.setFocus()
    before = edit.text()
    gui.help.on_f1(w)
    assert gui.help.window.current_topic == "param.material.E"
    assert gui.help.window.mode == "topic"
    assert edit.text() == before


def test_f1_key_in_a_field_opens_help(qtbot, gui):
    w = gui.main_window
    w.set_state("setup")
    w.activateWindow()
    edit = w.setup_view.nu.edit
    edit.setFocus()
    qtbot.keyClick(edit, Qt.Key.Key_F1)
    qtbot.waitUntil(lambda: gui.help.window is not None and gui.help.window.isVisible(), timeout=2000)
    assert gui.help.window.current_topic == "param.material.nu"


def test_f1_without_context_opens_search(qtbot, gui):
    w = gui.main_window
    w.set_state("start")
    w.start_view.new_button.setFocus()
    gui.help.on_f1(w)
    assert gui.help.window.mode == "search"
    assert gui.help.window.current_topic == START_TOPIC


def test_f1_in_analysis_window_opens_current_tab_article(qtbot, gui):
    a = gui.open_analysis()
    qtbot.waitExposed(a)
    a.tabs.setCurrentIndex(4)  # "Сходимость"
    a.tabs.tabBar().setFocus()
    gui.help.on_f1(a)
    assert gui.help.window.current_topic == "iterations.overview"
    assert any(s.key().toString() == "F1" for s in a.findChildren(QShortcut))


def test_info_button_opens_article(qtbot, gui):
    gui.main_window.set_state("setup")
    buttons = gui.main_window.setup_view.findChildren(InfoButton)
    topics = {b.property("helpTopic") for b in buttons}
    assert {"param.geometry.a", "param.material.nu", "param.theory"} <= topics
    button = next(b for b in buttons if b.property("helpTopic") == "param.theory")
    button.click()
    assert gui.help.window.current_topic == "param.theory"


def test_all_gui_help_references_exist(qtbot, gui):
    a = gui.open_analysis()
    for window in (gui.main_window, a):
        for widget in window.findChildren(QWidget):
            for prop in ("helpTopic", "helpContext"):
                topic = widget.property(prop)
                if topic:
                    assert get(topic) is not None, (prop, topic)


# Help centre

def test_help_centre_search_and_links(qtbot, gui):
    hw = gui.help.open_search()
    all_topics = hw.topics_in_index()
    assert all_topics == sorted(all_topics, key=lambda t: get(t).title.casefold().replace("ё", "е"))
    hw.search.setText("Пуассон")
    assert hw.topics_in_index() == ["param.material.nu"]
    assert hw.found.text() == "Найдено: 1"
    hw.search.setText("")
    assert hw.topics_in_index() == all_topics
    hw.view.anchorClicked.emit(QUrl("topic:param.geometry.b"))
    assert hw.current_topic == "param.geometry.b"


def test_help_window_is_reused(qtbot, gui):
    first = gui.help.open("param.geometry.a")
    second = gui.help.open_search()
    assert first is second


# Adaptive layout

@pytest.mark.parametrize(("width", "mode"), [(1500, "wide"), (1100, "wide"), (900, "medium"), (700, "narrow")])
def test_layout_mode_thresholds(width, mode):
    assert layout_mode(width) == mode


def test_setup_page_relayouts_by_width(qtbot, gui):
    w = gui.main_window
    w.set_state("setup")
    view = w.setup_view
    grid = view.grid
    for width, mode in ((1400, "wide"), (900, "medium"), (700, "narrow")):
        view._apply_mode(layout_mode(width))
        assert view.mode == mode
        row, col, _, _ = grid.getItemPosition(grid.indexOf(view.scheme))
        assert (col == 2) == (mode == "wide")
        assert view.run_button.isVisibleTo(view)


def test_header_shortens_on_narrow_windows(qtbot, gui):
    header = gui.main_window.header
    header.resize(1000, header.height())
    header._adapt()
    assert header.badge.text() == "TEST DATA"
    header.resize(1400, header.height())
    header._adapt()
    assert header.badge.text().startswith("TEST DATA ·")

