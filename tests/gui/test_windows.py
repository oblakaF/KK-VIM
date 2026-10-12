"""P1 skeleton: main window states, analysis window, themes, figure language, SI units."""

import re

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

from PySide6.QtCore import QSettings  # noqa: E402
from PySide6.QtGui import QShortcut  # noqa: E402
from PySide6.QtWidgets import QApplication, QComboBox, QLabel, QPushButton, QTabWidget  # noqa: E402

from kkvim_gui.analysis.window import TABS, AnalysisWindow  # noqa: E402
from kkvim_gui.app import GuiApplication  # noqa: E402
from kkvim_gui.main_window.window import STATES, MainWindow  # noqa: E402
from kkvim_gui.session import Session  # noqa: E402
from kkvim_gui.theme.tokens import DARK, LIGHT, THEMES  # noqa: E402
from kkvim_gui.widgets.common import (  # noqa: E402
    SI_UNITS,
    FigureLanguageSwitch,
    FigurePlaceholder,
    UnitField,
)
from kkvim_gui.widgets.header import TEST_DATA_BADGE  # noqa: E402


@pytest.fixture
def session(qapp):
    s = Session(settings=None)
    s.apply_theme()
    yield s
    s.set_theme("light", persist=False)


@pytest.fixture
def main_window(qtbot, session):
    w = MainWindow(session)
    qtbot.addWidget(w)
    return w


@pytest.fixture
def dev_window(qtbot, session):
    w = MainWindow(session, dev=True)
    qtbot.addWidget(w)
    return w


class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def finish_run(window, clock, seconds=60.0):
    clock.t += seconds
    window.runs.tick()


def test_main_window_has_four_states(dev_window):
    w = dev_window
    assert STATES == ("start", "setup", "running", "finished")
    expected = {
        "start": w.start_view,
        "setup": w.setup_view,
        "running": w.run_view,   # both run states are shown by the run dispatcher
        "finished": w.run_view,
    }
    for state, page in expected.items():
        w.set_state(state)
        assert w.stack.currentWidget() is page
        assert w.state_selector.currentData() == state


def test_state_selector_only_in_developer_mode(main_window, dev_window):
    assert main_window.state_selector is None
    assert not [s for s in main_window.findChildren(QShortcut) if s.key().toString().startswith("Ctrl+1")]
    dev_window.state_selector.setCurrentIndex(STATES.index("finished"))
    assert dev_window.state == "finished"
    assert any(s.key().toString() == "Ctrl+1" for s in dev_window.findChildren(QShortcut))


def test_navigation_between_states(qtbot, main_window):
    clock = FakeClock()
    main_window.runs.clock = clock
    main_window.start_view.new_button.click()
    assert main_window.state == "setup"
    main_window.setup_view.run_button.click()
    assert main_window.state == "running"
    finish_run(main_window, clock)
    assert main_window.state == "finished"
    main_window.nav.buttons["setup"].click()        # back to the parameters via the top navigation
    assert main_window.state == "setup"


def test_finished_run_has_no_back_to_parameters_button(main_window):
    clock = FakeClock()
    main_window.runs.clock = clock
    main_window.set_state("setup")
    main_window.setup_view.try_run()
    finish_run(main_window, clock)
    rv = main_window.run_view
    texts = [b.text() for b in rv.findChildren(QPushButton)]
    assert "Вернуться к параметрам" not in texts
    assert rv.analysis_button.isVisibleTo(rv)


def test_header_navigation_stays_available_during_a_run(main_window):
    clock = FakeClock()
    main_window.runs.clock = clock
    nav = main_window.nav
    assert [b.text() for b in nav.buttons.values()] == ["Главная", "Настройка расчёта", "Ход расчёта"]
    assert nav.buttons["start"].isChecked()
    assert not nav.buttons["run"].isEnabled()  # no run yet
    nav.buttons["setup"].click()
    main_window.setup_view.try_run()
    assert main_window.state == "running" and nav.buttons["run"].isChecked()
    # All sections stay reachable while the run is active.
    assert nav.buttons["start"].isEnabled() and nav.buttons["setup"].isEnabled()
    nav.buttons["start"].click()
    assert main_window.state == "start"
    nav.buttons["run"].click()
    assert main_window.state == "running"
    finish_run(main_window, clock)
    assert main_window.state == "finished" and nav.buttons["run"].isChecked()


@pytest.fixture
def gui(qtbot, session):
    g = GuiApplication(session)
    qtbot.addWidget(g.main_window)
    yield g
    for w in list(g.analysis_windows):
        w.close()
    g.help.close()


def test_analysis_opens_only_by_button_and_only_for_finished_runs(qtbot, gui):
    w = gui.main_window
    clock = FakeClock()
    w.runs.clock = clock
    w.set_state("setup")
    w.setup_view.try_run()
    w.run_view.analysis_button.click()      # running: no analysis
    assert gui.analysis_windows == []
    finish_run(w, clock)
    assert gui.analysis_windows == []       # never opened automatically
    with qtbot.waitSignal(w.analysisRequested):
        w.run_view.analysis_button.click()
    assert len(gui.analysis_windows) == 1


def test_report_button_is_planned_but_disabled(qtbot, session):
    w = AnalysisWindow(session)
    qtbot.addWidget(w)
    assert w.report_button.text() == "Собрать отчёт…"
    assert not w.report_button.isEnabled()


def test_test_data_badge_in_both_windows(qtbot, session, main_window):
    analysis = AnalysisWindow(session)
    qtbot.addWidget(analysis)
    for header in (main_window.header, analysis.header):
        # Full text on wide windows, shortened on narrow ones; TEST DATA is always there.
        assert header.badge.text() in (TEST_DATA_BADGE, "TEST DATA")
        assert header.badge.isVisibleTo(header)


def test_analysis_window_has_horizontal_tabs(qtbot, session):
    w = AnalysisWindow(session)
    qtbot.addWidget(w)
    assert w.tabs.tabPosition() == QTabWidget.TabPosition.North
    assert [w.tabs.tabText(i) for i in range(w.tabs.count())] == list(TABS)
    comparison = TABS.index("Сравнение")
    assert not w.tabs.isTabEnabled(comparison)
    assert all(w.tabs.isTabEnabled(i) for i in range(w.tabs.count()) if i != comparison)


def test_three_statuses_are_separate(qtbot, session):
    w = AnalysisWindow(session)
    qtbot.addWidget(w)
    texts = [w.status_run.text(), w.status_stop.text(), w.status_accept.text()]
    assert "Расчёт выполнен" in texts[0]
    assert "Критерий остановки" in texts[1]
    assert "Научная приёмка: не проводилась" in texts[2]


def test_figure_language_is_shared_by_all_windows(qtbot, session):
    first, second = AnalysisWindow(session), AnalysisWindow(session)
    qtbot.addWidget(first)
    qtbot.addWidget(second)
    assert session.figure_language == "en"

    def titles(window):
        return {p.title.text() for p in window.findChildren(FigurePlaceholder)}

    assert "Deflection w(x, y)" in titles(second)
    first.findChildren(FigureLanguageSwitch)[0].control.buttons["ru"].click()
    assert session.figure_language == "ru"
    assert "Прогиб w(x, y)" in titles(second)
    marks = {p.mark.text() for p in second.findChildren(FigurePlaceholder)}
    assert marks == {"ТЕСТОВЫЕ ДАННЫЕ · TEST DATA"}
    for sw in second.findChildren(FigureLanguageSwitch):
        assert sw.control.current() == "ru"


def test_themes_share_tokens_and_light_is_default(qapp, tmp_path):
    assert set(LIGHT) == set(DARK)
    assert set(THEMES) == {"light", "dark"}
    s = Session(settings=None)
    assert s.theme == "light"
    s.set_theme("dark", persist=False)
    assert DARK["window.bg"] in qapp.styleSheet()
    s.set_theme("light", persist=False)
    assert LIGHT["window.bg"] in qapp.styleSheet()


def test_settings_are_persisted(qapp, tmp_path):
    path = str(tmp_path / "kkvim-gui.ini")
    s = Session(QSettings(path, QSettings.Format.IniFormat))
    assert (s.theme, s.default_figure_language) == ("light", "en")
    s.set_theme("dark")
    s.set_default_figure_language("ru")
    s._settings.sync()
    again = Session(QSettings(path, QSettings.Format.IniFormat))
    assert (again.theme, again.default_figure_language, again.figure_language) == ("dark", "ru", "ru")
    again.set_theme("light", persist=False)


CYRILLIC = re.compile(r"[а-яА-ЯёЁ]")


def test_setup_units_are_si_symbols(main_window):
    for field in main_window.setup_view.unit_fields():
        units = [field.unit()] if field.unit() else []
        if field.unit_combo is not None:
            units = [field.unit_combo.itemText(i) for i in range(field.unit_combo.count())]
        for unit in units:
            assert unit in SI_UNITS
            assert not CYRILLIC.search(unit)


def test_unit_field_rejects_non_si_symbols(qapp):
    with pytest.raises(ValueError):
        UnitField("1", "мм")


def test_changing_unit_does_not_change_the_number(main_window):
    a = main_window.setup_view.a
    before = a.text()
    a.unit_combo.setCurrentText("mm")
    assert a.text() == before


def test_window_fits_available_screen(main_window, qapp):
    avail = QApplication.primaryScreen().availableGeometry()
    assert main_window.minimumWidth() <= avail.width()
    assert main_window.minimumHeight() <= avail.height()
    assert main_window.width() <= avail.width()


def test_interface_labels_use_no_cyrillic_unit_symbols(qtbot, session, main_window):
    analysis = AnalysisWindow(session)
    qtbot.addWidget(analysis)
    pattern = re.compile(r"(?<![а-яА-ЯёЁ])(мм|см|кПа|МПа|ГПа)(?![а-яА-ЯёЁ])|\d\s?м(?![а-яА-ЯёЁ])")
    for window in (main_window, analysis):
        for lab in window.findChildren(QLabel):
            assert not pattern.search(lab.text()), lab.text()
        for combo in window.findChildren(QComboBox):
            for i in range(combo.count()):
                assert not pattern.search(combo.itemText(i)), combo.itemText(i)


def test_header_is_dark_blue_in_both_themes(session):
    for name, tokens in THEMES.items():
        session.set_theme(name, persist=False)
        assert f"QFrame#AppHeader {{\n    background: {tokens['header.bg']};" in QApplication.instance().styleSheet()


def test_theme_icons_exist_for_both_themes():
    from kkvim_gui.theme.qss import ICONS

    for tokens in THEMES.values():
        assert (ICONS / tokens["icon.chevron"]).is_file()


def test_figure_language_switch_order_is_ru_en(qtbot, session):
    w = AnalysisWindow(session)
    qtbot.addWidget(w)
    switches = w.findChildren(FigureLanguageSwitch)
    assert switches
    for sw in switches:
        assert [b.text() for b in sw.control.buttons.values()] == ["RU", "EN"]
        assert sw.control.current() == "en"           # English stays the default
