"""Main window: header with navigation + four states in one window (GUI_DESIGN.md, 3.5, 5, 17.1).

Navigation: "Главная" (start), "Настройка расчёта" (setup), "Ход расчёта" (the run
dispatcher; "running" and "finished" are both shown there). All sections stay reachable
while a run is active; the parameters of the active run are read-only until it finishes
or is stopped. Runs of the session are kept by a RunManager (individual timers).

Developer mode (dev=True, `kkvim-gui --dev`) adds the state selector in the status bar
and Ctrl+1..Ctrl+4 for design review; ordinary users do not see them. Application
shortcuts never use the standard editing keys (Ctrl+C/V/X/Z/Y/A; GUI_DESIGN.md, 12.1).
"""

from PySide6.QtCore import Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QComboBox,
    QMainWindow,
    QMessageBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from kkvim_gui import placeholders, program_version
from kkvim_gui.main_window.run_view import RunView
from kkvim_gui.main_window.setup_view import SetupView
from kkvim_gui.main_window.start_view import StartView
from kkvim_gui.model.runs import RunManager
from kkvim_gui.model.setup_model import SetupModel
from kkvim_gui.model.setup_state import summary_rows
from kkvim_gui.widgets.common import fit_to_screen, label
from kkvim_gui.widgets.header import AppHeader, HeaderNav, app_icon

STATES = ("start", "setup", "running", "finished")
STATE_NAMES = {
    "start": "Старт",
    "setup": "Настройка",
    "running": "Выполнение",
    "finished": "Завершение",
}
NAV_SECTIONS = {"start": "Главная", "setup": "Настройка расчёта", "run": "Ход расчёта"}
NAV_OF_STATE = {"start": "start", "setup": "setup", "running": "run", "finished": "run"}
STATUS_TEXT = {
    "start": "●  Готово",
    "setup": "●  Готово",
    "running": "●  Выполняется (демонстрация)",
    "finished": "●  Расчёт выполнен (TEST DATA)",
}


class MainWindow(QMainWindow):
    analysisRequested = Signal()
    settingsRequested = Signal()
    helpRequested = Signal()

    def __init__(self, session, dev=False, parent=None):
        super().__init__(parent)
        self.session = session
        self.dev = dev
        self.setWindowIcon(app_icon())
        self.state = None
        self.runs = RunManager(parent=self)
        # One setup model per main window: values survive switching sections (GUI_DESIGN.md, 5.2).
        self.setup_model = SetupModel(parent=self)
        # Snapshot of the setup taken when a run starts; "Ход расчёта" and the analysis show it.
        self.run_snapshot = self.setup_model.state

        central = QWidget()
        central.setObjectName("Page")
        lay = QVBoxLayout(central)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self.nav = HeaderNav(NAV_SECTIONS)
        self.nav.navigate.connect(self._navigate)
        self.header = AppHeader(session, "КК-ВИМ", nav=self.nav)
        self.header.settingsClicked.connect(self.settingsRequested)
        self.header.helpClicked.connect(self.helpRequested)
        lay.addWidget(self.header)

        self.stack = QStackedWidget()
        self.start_view = StartView()
        self.setup_view = SetupView(self.setup_model, session)
        self.run_view = RunView(session, self.runs)
        for w in (self.start_view, self.setup_view, self.run_view):
            self.stack.addWidget(w)
        lay.addWidget(self.stack, 1)
        self.setCentralWidget(central)

        self.start_view.newRequested.connect(lambda: self.set_state("setup"))
        self.setup_view.runRequested.connect(self._start_run)
        self.run_view.stopRequested.connect(self._confirm_stop)
        self.run_view.analysisRequested.connect(self._open_analysis)
        self.runs.activeChanged.connect(self._on_active_changed)

        bar = self.statusBar()
        bar.setSizeGripEnabled(True)
        self.status_label = label("")
        bar.addWidget(self.status_label, 1)
        self.state_selector = None
        if dev:
            bar.addPermanentWidget(label("Состояние (разработка):"))
            self.state_selector = QComboBox()
            for key in STATES:
                self.state_selector.addItem(STATE_NAMES[key], key)
            self.state_selector.setToolTip("Только режим разработки: Ctrl+1 … Ctrl+4")
            self.state_selector.currentIndexChanged.connect(
                lambda i: self.set_state(self.state_selector.itemData(i))
            )
            bar.addPermanentWidget(self.state_selector)
            for i, key in enumerate(STATES, start=1):
                QShortcut(QKeySequence(f"Ctrl+{i}"), self, activated=lambda k=key: self.set_state(k))
        bar.addPermanentWidget(label(f"Бэкенд: демонстрационный · KK-VIM {program_version()}"))

        QShortcut(QKeySequence("Ctrl+N"), self, activated=self._new_shortcut)
        QShortcut(QKeySequence("F5"), self, activated=self._run_shortcut)

        fit_to_screen(self)
        self.set_state("start")

    @property
    def run_started(self):
        return bool(self.runs.runs)

    def _run_state(self):
        return "running" if self.runs.active is not None else "finished"

    def _navigate(self, section):
        self.set_state(self._run_state() if section == "run" else section)

    def set_state(self, state):
        if state not in STATES:
            raise ValueError(f"unknown state: {state!r}")
        self.state = state
        if state == "start":
            self.stack.setCurrentWidget(self.start_view)
            self.header.set_project("")
        elif state == "setup":
            self.stack.setCurrentWidget(self.setup_view)
            self.header.set_project(placeholders.PROJECT_NAME)
        else:
            self.stack.setCurrentWidget(self.run_view)
            self.header.set_project(placeholders.PROJECT_NAME)
        self.nav.set_current(NAV_OF_STATE[state])
        self.nav.set_enabled("run", self.run_started, "" if self.run_started else "Расчёт ещё не запускался")
        self.status_label.setText(STATUS_TEXT["running"] if self.runs.active else STATUS_TEXT[state])
        project = "" if state == "start" else f" — {placeholders.PROJECT_NAME}"
        self.setWindowTitle(f"КК-ВИМ{project}")
        if self.state_selector is not None:
            index = STATES.index(state)
            if self.state_selector.currentIndex() != index:
                self.state_selector.blockSignals(True)
                self.state_selector.setCurrentIndex(index)
                self.state_selector.blockSignals(False)

    def help_context(self):
        """F1 topic of the current page; None on the start page (opens the search)."""
        return self.stack.currentWidget().property("helpContext") or None

    def _new_shortcut(self):
        self.set_state("setup")

    def _run_shortcut(self):
        if self.state == "setup":
            self.setup_view.try_run()

    def _start_run(self):
        """Called only after the formal input check passed (SetupView.try_run)."""
        self.run_snapshot = self.setup_model.state
        self.runs.start_demo(self.run_snapshot)
        self.set_state("running")

    def _on_active_changed(self, active):
        """Lock the parameters while a run is active; unlock when it ends."""
        self.setup_view.set_locked(active)
        if self.state in ("running", "finished"):
            self.set_state(self._run_state())
        else:
            self.status_label.setText(STATUS_TEXT["running"] if active else STATUS_TEXT[self.state])

    def _open_analysis(self, run_id):
        """Analysis only for a finished run (stopped or failed runs have no results)."""
        run = self.runs.get(run_id)
        if run is not None and run.has_results:
            self.analysisRequested.emit()

    def selected_run(self):
        return self.run_view.selected()

    def run_rows(self):
        """Input parameters of the selected run as entered (for the analysis window)."""
        run = self.selected_run()
        return summary_rows(run.snapshot if run is not None else self.run_snapshot)

    def _confirm_stop(self, run_id):
        answer = QMessageBox.question(
            self,
            "Остановить расчёт",
            "Остановить расчёт? Анализ прерванного расчёта недоступен.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer == QMessageBox.StandardButton.Yes:
            self.runs.stop(run_id)
