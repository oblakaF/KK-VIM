"""Settings dialog with segmented controls; gear menu instead of the "?" button."""

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

from PySide6.QtCore import QSettings, Qt  # noqa: E402
from PySide6.QtWidgets import QApplication, QRadioButton, QToolButton  # noqa: E402

from kkvim_gui.analysis.window import AnalysisWindow  # noqa: E402
from kkvim_gui.app import GuiApplication  # noqa: E402
from kkvim_gui.help.window import HelpWindow  # noqa: E402
from kkvim_gui.main_window.window import MainWindow  # noqa: E402
from kkvim_gui.session import Session  # noqa: E402
from kkvim_gui.settings_dialog import SettingsDialog  # noqa: E402
from kkvim_gui.widgets.header import AppHeader  # noqa: E402


@pytest.fixture
def session(qapp, tmp_path):
    s = Session(QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat))
    s.apply_theme()
    yield s
    s.set_theme("light", persist=False)


@pytest.fixture
def dialog(qtbot, session):
    d = SettingsDialog(session)
    qtbot.addWidget(d)
    d.show()
    qtbot.waitExposed(d)
    return d


def test_segmented_controls_replace_radio_buttons(dialog):
    assert dialog.findChildren(QRadioButton) == []
    assert [b.text() for b in dialog.theme.buttons.values()] == ["Light", "Dark"]
    assert [b.text() for b in dialog.language.buttons.values()] == ["RU", "EN"]  # RU left, EN right
    assert dialog.theme.current() == "light" and dialog.language.current() == "en"
    for b in (*dialog.theme.buttons.values(), *dialog.language.buttons.values()):
        assert b.property("strong") is True  # accent-filled selection


def test_ok_applies_and_stores_both_choices(session, dialog):
    dialog.theme.buttons["dark"].click()
    dialog.language.buttons["ru"].click()
    assert session.theme == "light"  # nothing changes before OK
    dialog.accept()
    assert session.theme == "dark" and session.default_figure_language == "ru"
    session._settings.sync()
    again = Session(QSettings(session._settings.fileName(), QSettings.Format.IniFormat))
    assert (again.theme, again.default_figure_language) == ("dark", "ru")


def test_cancel_changes_nothing(session, dialog):
    dialog.theme.buttons["dark"].click()
    dialog.reject()
    assert session.theme == "light"


def test_keyboard_arrows_space_and_tab(qtbot, dialog):
    light = dialog.theme.buttons["light"]
    light.setFocus()
    qtbot.keyClick(light, Qt.Key.Key_Right)
    assert dialog.theme.current() == "dark"
    assert dialog.theme.buttons["dark"].hasFocus()
    qtbot.keyClick(dialog.theme.buttons["dark"], Qt.Key.Key_Left)
    assert dialog.theme.current() == "light"
    ru = dialog.language.buttons["ru"]
    ru.setFocus()
    qtbot.keyClick(ru, Qt.Key.Key_Space)
    assert dialog.language.current() == "ru"
    assert ru.focusPolicy() & Qt.FocusPolicy.TabFocus


def test_focus_and_selection_styles_exist_in_both_themes(session):
    for theme in ("light", "dark"):
        session.set_theme(theme, persist=False)
        qss = QApplication.instance().styleSheet()
        assert 'QPushButton[segment][strong="true"]:checked' in qss
        assert 'QPushButton[segment][strong="true"]:focus' in qss


@pytest.mark.parametrize("make", [lambda s: MainWindow(s), lambda s: AnalysisWindow(s)])
def test_gear_menu_replaces_the_question_mark(qtbot, session, make):
    w = make(session)
    qtbot.addWidget(w)
    header = w.header
    assert not [b for b in header.findChildren(QToolButton) if b.text() == "?"]
    w.show()
    qtbot.waitExposed(w)
    qtbot.mouseClick(header.settings_button, Qt.MouseButton.LeftButton)
    assert header.menu.isVisible()
    gear = header.settings_button
    assert header.menu.geometry().right() <= gear.mapToGlobal(gear.rect().bottomRight()).x() + 1
    header.menu.hide()
    texts = [a.text() for a in header.menu.actions() if a.isVisible()]
    assert texts == ["Настройки", "Справка\tF1"]
    assert all(a.shortcut().isEmpty() for a in header.menu.actions())  # F1 stays contextual
    with qtbot.waitSignal(w.settingsRequested):
        header.settings_action.trigger()
    with qtbot.waitSignal(w.helpRequested):
        header.help_action.trigger()


def test_help_menu_item_opens_the_help_centre(qtbot, session):
    g = GuiApplication(session)
    qtbot.addWidget(g.main_window)
    g.main_window.header.help_action.trigger()
    assert g.help.window is not None and g.help.window.mode == "search"
    g.help.close()


def test_help_window_has_no_gear(qtbot, session):
    w = HelpWindow(session)
    qtbot.addWidget(w)
    header = w.findChild(AppHeader)
    assert not header.settings_button.isVisibleTo(header)
