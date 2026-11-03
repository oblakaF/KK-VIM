"""Run dispatcher: individual timers, counters, stop, selection, journal, setup lock."""

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtGui import QGuiApplication  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from kkvim_gui.main_window.setup_view import LOCKED_TEXT  # noqa: E402
from kkvim_gui.main_window.window import MainWindow  # noqa: E402
from kkvim_gui.model.runs import (  # noqa: E402
    DEMO_FINISH,
    DEMO_HALF_STEP,
    DEMO_K_STOP,
    DEMO_PREPARE,
    RunManager,
    format_duration,
)
from kkvim_gui.model.setup_state import SetupState  # noqa: E402
from kkvim_gui.session import Session  # noqa: E402
from kkvim_gui.widgets.common import FigurePlaceholder  # noqa: E402

FULL = DEMO_PREPARE + 2 * DEMO_K_STOP * DEMO_HALF_STEP + DEMO_FINISH


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t

    def advance(self, dt, manager):
        self.t += dt
        manager.tick()


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def manager(qapp, clock):
    return RunManager(clock=clock)


@pytest.mark.parametrize(("seconds", "text"), [(0.4, "0 с"), (5, "5 с"), (30.9, "30 с"),
                                               (65, "1 мин 05 с"), (3725, "1 ч 02 мин")])
def test_duration_format(seconds, text):
    assert format_duration(seconds) == text


def test_each_run_has_its_own_timer(manager, clock):
    first = manager.start_demo(SetupState())
    clock.advance(3, manager)
    assert first.active and manager.elapsed(first) == pytest.approx(3)
    clock.advance(FULL, manager)
    assert first.status == "finished"
    frozen = manager.elapsed(first)
    second = manager.start_demo(SetupState())
    clock.advance(2, manager)
    assert manager.elapsed(second) == pytest.approx(2)        # measured separately
    assert manager.elapsed(first) == pytest.approx(frozen)    # frozen after the end
    clock.advance(50, manager)
    assert manager.elapsed(first) == pytest.approx(frozen)


def test_counters_follow_d030(manager, clock):
    run = manager.start_demo(SetupState())
    clock.advance(DEMO_PREPARE + 5 * DEMO_HALF_STEP + 0.01, manager)
    assert run.s == 5 and run.k == 2          # k = floor(s / 2)
    clock.advance(FULL, manager)
    assert run.s == 2 * DEMO_K_STOP and run.k == DEMO_K_STOP
    assert run.stop_criterion == "met"


def test_non_iterative_method_has_no_counters(manager, clock):
    run = manager.start_demo(SetupState(), method="МКР")
    assert run.s is None and run.k is None and not run.iterative
    clock.advance(10, manager)
    assert run.status == "finished" and run.stop_criterion == ""


def test_stop_freezes_time_and_disables_analysis(manager, clock):
    run = manager.start_demo(SetupState())
    clock.advance(3, manager)
    manager.stop(run.run_id)
    assert run.status == "stopped" and not run.has_results
    clock.advance(30, manager)
    assert manager.elapsed(run) == pytest.approx(3)
    assert manager.active is None


def test_only_one_active_run(manager):
    manager.start_demo(SetupState())
    with pytest.raises(RuntimeError):
        manager.start_demo(SetupState())


def test_journal_has_no_per_iteration_lines(manager, clock):
    run = manager.start_demo(SetupState())
    clock.advance(FULL + 1, manager)
    texts = [text for _, _, text in run.detailed]
    assert not any(text.startswith("Итерация") for text in texts)
    assert len(run.important) < len(run.detailed)
    assert all(entry in run.detailed for entry in run.important)
    assert any("TEST DATA" in text for _, _, text in run.important)


# Main window

@pytest.fixture
def window(qtbot, qapp, clock):
    session = Session(settings=None)
    session.apply_theme()
    w = MainWindow(session, demo=True)   # dispatcher mechanics on the demonstration backend
    w.runs.clock = clock
    qtbot.addWidget(w)
    w.show()
    w.set_state("setup")
    yield w
    session.set_theme("light", persist=False)


def table_row(w, row):
    return [w.run_view.table.item(row, c).text() for c in range(w.run_view.table.columnCount())]


def test_dispatcher_rows_and_individual_durations(window, clock):
    w, rv = window, window.run_view
    w.setup_view.try_run()
    clock.advance(5, w.runs)
    row = table_row(w, 0)
    assert row[1] == "КК-ВИМ" and "N = 1" in row[2] and "демонстрация" in row[2]
    assert row[3] == "Выполняется" and row[4] == "5 с"
    assert rv.summary_title.text() == "Выполняется — 5 с"
    clock.advance(FULL, w.runs)
    first = table_row(w, 0)
    assert first[3] == "Завершён"
    w.set_state("setup")
    w.setup_view.try_run()
    clock.advance(2, w.runs)
    assert table_row(w, 1)[4] == "2 с"
    assert table_row(w, 0)[4] == first[4]      # the finished run keeps its own time
    assert rv.table.rowCount() == 2


def test_selecting_a_run_shows_its_details(window, clock):
    w, rv = window, window.run_view
    w.setup_model.set_text("a", "3")
    w.setup_view.try_run()
    clock.advance(FULL + 1, w.runs)
    w.set_state("setup")
    w.setup_model.set_text("a", "4")
    w.setup_view.try_run()
    clock.advance(1, w.runs)
    rv.table.selectRow(0)
    assert "a 3 m" in rv.details["Пластина"].text()
    assert rv.summary_title.text().startswith("Завершён")
    assert rv.analysis_button.isVisibleTo(rv) and not rv.stop_button.isVisibleTo(rv)
    assert "s = 16" in rv.details["Счётчики"].text() and "k = 8" in rv.details["Счётчики"].text()
    rv.table.selectRow(1)
    assert "a 4 m" in rv.details["Пластина"].text()
    assert rv.stop_button.isVisibleTo(rv) and not rv.analysis_button.isVisibleTo(rv)


def test_no_convergence_plot_in_the_main_window(window):
    assert window.run_view.findChildren(FigurePlaceholder) == []


def test_detailed_journal_is_collapsed_by_default(window, clock):
    rv = window.run_view
    window.setup_view.try_run()
    assert not rv.journal.isVisibleTo(rv)
    rv.journal_toggle.setChecked(True)
    assert rv.journal.isVisibleTo(rv) and rv.journal.rowCount() >= 2
    assert "Важн" not in rv.journal.item(0, 2).text()


@pytest.mark.parametrize(("answer", "status"), [(QMessageBox.StandardButton.Yes, "stopped"),
                                                (QMessageBox.StandardButton.No, "running")])
def test_stop_asks_for_confirmation(window, clock, monkeypatch, answer, status):
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: answer)
    window.setup_view.try_run()
    clock.advance(3, window.runs)
    window.run_view.stop_button.click()
    assert window.runs.runs[0].status == status


def test_parameters_are_read_only_during_the_run(qtbot, window, clock):
    w, sv = window, window.setup_view
    sv.try_run()
    w.nav.buttons["setup"].click()                 # navigation allowed during the run
    assert w.state == "setup" and sv.locked
    assert sv.a.edit.isReadOnly() and not sv.a.unit_combo.isEnabled()
    assert not sv.supports.buttons["CCCC"].isEnabled() and not sv.N.isEnabled()
    assert not sv.run_button.isEnabled() and sv.check_text.text() == LOCKED_TEXT
    assert sv.try_run() is False
    qtbot.keyClicks(sv.a.edit, "9")
    assert sv.a.text() == "2,000"                  # cannot be changed
    sv.a.edit.setFocus()
    qtbot.keyClick(sv.a.edit, Qt.Key.Key_A, Qt.KeyboardModifier.ControlModifier)
    qtbot.keyClick(sv.a.edit, Qt.Key.Key_C, Qt.KeyboardModifier.ControlModifier)
    assert QGuiApplication.clipboard().text() == "2,000"   # viewing and copying work
    clock.advance(FULL + 1, w.runs)                # finished: editing again
    assert not sv.locked and not sv.a.edit.isReadOnly() and sv.run_button.isEnabled()


def test_both_themes_redraw_the_dispatcher(window, clock):
    window.setup_view.try_run()
    clock.advance(2, window.runs)
    for theme in ("dark", "light"):
        window.session.set_theme(theme, persist=False)
        assert window.run_view.grab().isNull() is False


def test_stopped_by_user_wording_and_no_pause_button(window, clock, monkeypatch):
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)
    window.setup_view.try_run()
    clock.advance(4, window.runs)
    window.run_view.stop_button.click()
    row = table_row(window, 0)
    assert row[3] == "Остановлен пользователем" and row[4] == "4 с"
    assert window.run_view.summary_title.text() == "Остановлен пользователем — 4 с"
    texts = [b.text() for b in window.findChildren(type(window.run_view.stop_button))]
    assert not any("Пауза" in t or "Продолжить" in t for t in texts)


def test_lock_banner_only_during_an_active_run(window, clock, monkeypatch):
    from kkvim_gui.main_window.setup_view import LOCK_BANNER

    sv = window.setup_view
    assert not sv.lock_banner.isVisibleTo(sv)
    sv.try_run()
    window.nav.buttons["setup"].click()
    assert sv.lock_banner.isVisibleTo(sv)
    assert sv.lock_banner.text.text() == LOCK_BANNER
    assert QApplication.activeModalWidget() is None          # a banner, not a pop-up
    clock.advance(FULL + 1, window.runs)                    # finished: banner hidden
    assert not sv.lock_banner.isVisibleTo(sv)
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)
    sv.try_run()
    assert sv.lock_banner.isVisibleTo(sv)
    window.runs.stop(window.runs.active.run_id)             # stopped: banner hidden
    assert not sv.lock_banner.isVisibleTo(sv)
