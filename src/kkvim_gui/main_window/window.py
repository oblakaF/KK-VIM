"""Main window: header with navigation + four states in one window (GUI_DESIGN.md, 3.5, 5, 17.1).

Navigation: "Главная" (start), "Настройка расчёта" (setup), "Ход расчёта" (the run
dispatcher; "running" and "finished" are both shown there). All sections stay reachable
while a run is active; the parameters of the active run are read-only until it finishes
or is stopped. Runs of the session are kept by a RunManager (individual timers).

Developer mode (dev=True, `kkvim-gui --dev`) adds the state selector in the status bar
and Ctrl+1..Ctrl+4 for design review; ordinary users do not see them. Application
shortcuts never use the standard editing keys (Ctrl+C/V/X/Z/Y/A; GUI_DESIGN.md, 12.1).

Backend: the KK-VIM solver in a separate process (default) or, only with demo=True
(`kkvim-gui --demo`), the demonstration backend marked TEST DATA. Before a solver run the input is
checked by kkvim's own validation (``kkvim_adapter.check_request``); a rejected input is shown on the
setup page and no run is started.

A load series («Серия нагрузок») starts ``RunManager.start_series``: one ordinary solver run per load q, each checked by
kkvim before the first one starts; «Нагрузка — прогиб…» in «Ход расчёта» opens its graph (``seriesRequested``).

Project .kkv (P2.3; ``model.project_file``): «Сохранить проект» Ctrl+S, «Сохранить как…» Ctrl+Shift+S, «Открыть
проект…» Ctrl+O, «Новый расчёт» Ctrl+N. A project holds the setup only (entered values and units, method, research
and display choices), never results. Opening checks the whole file first: a defective file is refused with its reason
and the current setup is kept. Opening or a new calculation replaces only the setup: no solver, Navier or D2 process
is started, and the runs of the session keep their own snapshots. Unsaved changes (the setup differs from the one
last saved or opened) are marked «●» in the header; before they are discarded the user chooses «Сохранить», «Не
сохранять» or «Отмена». ``confirm_close`` (set by ``app.main``) asks the same when the window is closed.
"""

from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from kkvim_gui import placeholders, program_version
from kkvim_gui.backend import export
from kkvim_gui.backend.kkvim_adapter import AdapterInputError, check_request
from kkvim_gui.backend.process import SolverProcess
from kkvim_gui.main_window.run_view import RunView
from kkvim_gui.main_window.setup_view import SetupView
from kkvim_gui.main_window.start_view import StartView
from kkvim_gui.model import project_file
from kkvim_gui.model.runs import RunManager
from kkvim_gui.model.setup_model import SetupModel
from kkvim_gui.model.setup_state import SetupState, summary_rows, to_request, to_series_requests
from kkvim_gui.widgets.common import fit_to_screen, label
from kkvim_gui.widgets.header import AppHeader, HeaderNav, app_icon

PREVIEW_N = (1,)  # functional preview: only the accepted R3 solver (D-043)
PREVIEW_BADGE = "ПРЕДВАРИТЕЛЬНАЯ ВЕРСИЯ · SSSS, N = 1"
PREVIEW_TOOLTIP = ("Функциональная предварительная версия GUI, не официальный выпуск R7. Доступно: Кирхгоф–Ляв, SSSS, "
                   "N = 1 (решатель R3, ACCEPTED, D-043). Отдельный расчёт — не научная приёмка")
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
    "running": "●  Выполняется расчёт",
    "finished": "●  Готово",
}
UNTITLED = "Без имени"
PROJECT_FILTER = "Проект КК-ВИМ (*.kkv)"
ACTIVE_RUN_TEXT = "Во время расчёта параметры не меняются: откройте проект или начните новый расчёт после его завершения."


class MainWindow(QMainWindow):
    analysisRequested = Signal()
    seriesRequested = Signal(int)
    settingsRequested = Signal()
    helpRequested = Signal()

    def __init__(self, session, dev=False, demo=False, protocol_dir=None, process_factory=SolverProcess,
                 preview=False, parent=None):
        """preview=True: the functional preview — only Kirchhoff–Love, SSSS, N = 1 (R3) can be chosen and run."""
        super().__init__(parent)
        self.session = session
        self.dev = dev
        self.demo = demo
        self.preview = preview
        self.setWindowIcon(app_icon())
        self.state = None
        self.runs = RunManager(process_factory=process_factory, protocol_dir=protocol_dir, parent=self)
        # One setup model per main window: values survive switching sections (GUI_DESIGN.md, 5.2).
        self.setup_model = SetupModel(parent=self)
        # Snapshot of the setup taken when a run starts; "Ход расчёта" and the analysis show it.
        self.run_snapshot = self.setup_model.state
        # Project .kkv: the file, its creation stamp and the setup as last saved or opened (unsaved changes: differs).
        self.project_path = None
        self.project_created = None
        self.saved_state = self.setup_model.state
        self.confirm_close = False  # app.main: ask before closing with unsaved changes (not in tests and smoke runs)
        self.last_dir = None

        central = QWidget()
        central.setObjectName("Page")
        lay = QVBoxLayout(central)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self.nav = HeaderNav(NAV_SECTIONS)
        self.nav.navigate.connect(self._navigate)
        self.header = AppHeader(session, "КК-ВИМ", nav=self.nav, show_badge=demo or preview)
        if preview and not demo:
            self.header.set_badge(PREVIEW_BADGE, "ПРЕДВАРИТЕЛЬНАЯ", PREVIEW_TOOLTIP)
        self.header.settingsClicked.connect(self.settingsRequested)
        self.header.helpClicked.connect(self.helpRequested)
        lay.addWidget(self.header)

        self.stack = QStackedWidget()
        self.start_view = StartView(demo=demo)
        self.setup_view = SetupView(self.setup_model, session, demo=demo, n_selectable=PREVIEW_N if preview else None)
        self.run_view = RunView(session, self.runs)
        for w in (self.start_view, self.setup_view, self.run_view):
            self.stack.addWidget(w)
        lay.addWidget(self.stack, 1)
        self.setCentralWidget(central)

        self.start_view.newRequested.connect(self.new_project)
        self.start_view.openRequested.connect(self.open_project)
        self.setup_view.runRequested.connect(self._start_run)
        self.setup_view.saveRequested.connect(self.save_project)
        self.setup_view.saveAsRequested.connect(self.save_project_as)
        self.setup_model.changed.connect(self._show_project)
        self.run_view.stopRequested.connect(self._confirm_stop)
        self.run_view.analysisRequested.connect(self._open_analysis)
        self.run_view.exportRequested.connect(self._choose_export_folder)
        self.run_view.seriesRequested.connect(self.seriesRequested)
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
        backend = "демонстрационные данные (TEST DATA)" if demo else "решатель KK-VIM"
        bar.addPermanentWidget(label(f"{backend} · версия {program_version()}"))

        QShortcut(QKeySequence("Ctrl+N"), self, activated=self._new_shortcut)
        QShortcut(QKeySequence("Ctrl+O"), self, activated=self.open_project)
        QShortcut(QKeySequence("Ctrl+S"), self, activated=self.save_project)
        QShortcut(QKeySequence("Ctrl+Shift+S"), self, activated=self.save_project_as)
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
        elif state == "setup":
            self.stack.setCurrentWidget(self.setup_view)
        else:
            self.stack.setCurrentWidget(self.run_view)
        self.nav.set_current(NAV_OF_STATE[state])
        self.nav.set_enabled("run", self.run_started, "" if self.run_started else "Расчёт ещё не запускался")
        self.status_label.setText(STATUS_TEXT["running"] if self.runs.active else STATUS_TEXT[state])
        self._show_project()
        if self.state_selector is not None:
            index = STATES.index(state)
            if self.state_selector.currentIndex() != index:
                self.state_selector.blockSignals(True)
                self.state_selector.setCurrentIndex(index)
                self.state_selector.blockSignals(False)

    def _project_name(self):
        """The file name of the project; «Без имени» before the first save (the demonstration: its example name)."""
        if self.project_path is not None:
            return self.project_path.name
        return placeholders.PROJECT_NAME if self.demo else UNTITLED

    @property
    def modified(self):
        """True if the setup differs from the one last saved or opened (or, for a new one, from the initial form)."""
        return self.setup_model.state != self.saved_state

    def _show_project(self):
        """Project name and «●» (unsaved changes) in the header and the window title; on the start page only if a
        project is open or the setup was changed."""
        if self.state is None:
            return
        shown = self.state != "start" or self.project_path is not None or self.modified
        name = self._project_name() if shown else ""
        self.header.set_project(name, modified=bool(name) and self.modified)
        self.setWindowTitle(f"КК-ВИМ — {name}{' ●' if self.modified else ''}" if name else "КК-ВИМ")

    # --- project .kkv ---------------------------------------------------------------------------------------------

    def ask_unsaved(self):
        """«save», «discard» or «cancel» for unsaved changes (a dialog; replaced in tests)."""
        box = QMessageBox(QMessageBox.Icon.Question, "Несохранённые изменения",
                          f"Параметры проекта «{self._project_name()}» изменены и не сохранены. Сохранить их?",
                          parent=self)
        save = box.addButton("Сохранить", QMessageBox.ButtonRole.AcceptRole)
        discard = box.addButton("Не сохранять", QMessageBox.ButtonRole.DestructiveRole)
        cancel = box.addButton("Отмена", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(save)
        box.setEscapeButton(cancel)
        box.exec()
        return {save: "save", discard: "discard"}.get(box.clickedButton(), "cancel")

    def ask_open_path(self):
        """Path of the project to open, or "" (the native file dialog; replaced in tests)."""
        path, _ = QFileDialog.getOpenFileName(self, "Открыть проект", self.last_dir or "", PROJECT_FILTER)
        return path

    def ask_save_path(self, suggested):
        """Path to save the project to, or "" (the native dialog asks before replacing a file; replaced in tests)."""
        path, _ = QFileDialog.getSaveFileName(self, "Сохранить проект как", suggested, PROJECT_FILTER)
        return path

    def ask_replace(self, path):
        """True to replace an existing file whose name got the extension .kkv here (the dialog did not ask)."""
        answer = QMessageBox.question(self, "Сохранить проект как", f"Файл «{path.name}» уже существует. Заменить его?",
                                      QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                      QMessageBox.StandardButton.No)
        return answer == QMessageBox.StandardButton.Yes

    def show_project_error(self, title, text):
        """A refused open or save: the status line and a message box (replaced in tests)."""
        self.status_label.setText(f"●  {title}")
        QMessageBox.warning(self, title, text)

    def confirm_discard(self):
        """True if the current setup may be replaced: no unsaved changes, saved now, or the user discards them."""
        if not self.modified:
            return True
        answer = self.ask_unsaved()
        if answer == "save":
            return self.save_project()
        return answer == "discard"

    def _replace_setup(self, state, path, created):
        """Show another setup; the runs and their snapshots are not touched and nothing is computed."""
        self.setup_model.load(state)
        self.project_path, self.project_created = path, created
        self.saved_state = self.setup_model.state
        self.setup_view.reset_checks()
        self.set_state("setup")

    def new_project(self):
        """«Новый расчёт»: the initial form of the program, after the unsaved changes are saved or discarded."""
        fresh = SetupState()
        if self.project_path is None and self.setup_model.state == fresh:
            self.set_state("setup")  # nothing to replace: the initial form is shown
            return True
        if self.runs.active is not None:
            self.set_state("setup")
            self.status_label.setText(f"●  {ACTIVE_RUN_TEXT}")
            return False
        if not self.confirm_discard():
            return False
        self._replace_setup(fresh, None, None)
        return True

    def open_project(self, path=None):
        """«Открыть проект…»: the whole file is checked before the setup is replaced; nothing is computed."""
        if self.runs.active is not None:
            self.show_project_error("Проект не открыт", ACTIVE_RUN_TEXT)
            return False
        if not self.confirm_discard():
            return False
        path = path or self.ask_open_path()
        if not path:
            return False
        path = Path(path)
        self.last_dir = str(path.parent)
        try:
            project = project_file.read(path)
        except project_file.ProjectError as exc:
            self.show_project_error("Проект не открыт", f"{path}\n\n{exc}\n\nТекущие параметры не изменены.")
            return False
        st = project.state
        if self.preview and (st.N not in PREVIEW_N or st.supports != "SSSS"):
            self.show_project_error("Проект не открыт", f"{path}\n\n{st.supports}, N = {st.N}: недоступно в "
                                    "функциональной предварительной версии; доступно SSSS, N = 1 (R3). Текущие параметры "
                                    "не изменены.")
            return False
        self._replace_setup(st, path, project.created)
        research = (" Исследовательский выбор восстановлен из файла (см. «Метод»)."
                    if st.research or st.ssss_research or st.scope == "cccc" else "")
        self.status_label.setText(f"●  Проект открыт: {path.name}. Расчёт не запускался.{research}")
        return True

    def save_project(self):
        """«Сохранить проект»: to the open file, or «Сохранить как…» for a new one."""
        if self.project_path is None:
            return self.save_project_as()
        return self._write_project(self.project_path, self.project_created)

    def save_project_as(self):
        suggested = str(self.project_path or Path(self.last_dir or "") / f"{UNTITLED}{project_file.SUFFIX}")
        path = self.ask_save_path(suggested)
        if not path:
            return False
        path = Path(path)
        if path.suffix.lower() != project_file.SUFFIX:
            path = path.with_name(path.name + project_file.SUFFIX)
            if path.exists() and not self.ask_replace(path):
                return False
        self.last_dir = str(path.parent)
        return self._write_project(path, self.project_created if path == self.project_path else None)

    def _write_project(self, path, created):
        state = self.setup_model.state
        try:
            project = project_file.write(path, state, created=created)
        except (OSError, project_file.ProjectError) as exc:
            reason = exc.strerror if isinstance(exc, OSError) and exc.strerror else str(exc)
            self.show_project_error("Проект не сохранён", f"{path}\n\n{reason}\n\nПрежний файл проекта не изменён.")
            return False
        self.project_path, self.project_created = path, project.created
        self.saved_state = state
        self._show_project()
        self.status_label.setText(f"●  Проект сохранён: {path.name} (параметры расчёта, без результатов)")
        return True

    def help_context(self):
        """F1 topic of the current page; None on the start page (opens the search)."""
        return self.stack.currentWidget().property("helpContext") or None

    def _new_shortcut(self):
        self.new_project()

    def _run_shortcut(self):
        if self.state == "setup":
            self.setup_view.try_run()

    def _start_run(self):
        """Called only after the formal input check passed (SetupView.try_run); never twice at once."""
        if self.runs.active is not None:
            return None
        snapshot = self.setup_model.state
        if snapshot.series:
            return self._start_series(snapshot)
        if self.demo:
            self.run_snapshot = snapshot
            run = self.runs.start_demo(snapshot)
            self.set_state("running")
            return run
        request = to_request(snapshot)
        problems = check_request(request)
        if self.preview and (request.N not in PREVIEW_N or request.supports != "SSSS"):
            problems.append(f"{request.supports}, N = {request.N}: недоступно в функциональной предварительной версии; "
                            "доступно SSSS, N = 1 (R3)")
        if problems:
            self.setup_view.show_solver_problems(problems)
            return None
        self.run_snapshot = snapshot
        run = self.runs.start(snapshot, request)
        self.set_state("running")
        return run

    def _start_series(self, snapshot):
        """A load series: every point is checked by kkvim first; a rejected point refuses the whole series."""
        if self.demo:
            self.setup_view.show_solver_problems(["Серия нагрузок выполняется только решателем KK-VIM, не в "
                                                  "демонстрационном режиме."])
            return None
        requests = to_series_requests(snapshot)
        problems = []
        for k, request in enumerate(requests, start=1):
            problems += [f"q = {request.q!r} Pa (точка {k}): {p}" for p in check_request(request)]
        if self.preview and (snapshot.N not in PREVIEW_N or snapshot.supports != "SSSS"):
            problems.append(f"{snapshot.supports}, N = {snapshot.N}: недоступно в функциональной предварительной версии; "
                            "доступно SSSS, N = 1 (R3)")
        if problems:
            self.setup_view.show_solver_problems(problems)
            return None
        try:
            series = self.runs.start_series(snapshot, requests)
        except AdapterInputError as exc:
            self.setup_view.show_solver_problems([str(exc)])
            return None
        self.run_snapshot = snapshot
        self.set_state("running")
        return series

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
        if run is not None and (run.has_results or run.has_diagnostic_field):
            self.analysisRequested.emit()

    def _choose_export_folder(self, run_id):
        run = self.runs.get(run_id)
        if run is None or run.outcome is None:
            return
        folder = QFileDialog.getExistingDirectory(self, "Папка для расчёта и отчёта")
        if not folder:
            return
        try:
            paths = self.save_package(run_id, folder)
        except OSError as exc:
            QMessageBox.warning(self, "Сохранение не выполнено", f"Расчёт не сохранён: {exc}")
            return
        self.status_label.setText(f"●  Расчёт и отчёт сохранены: {paths['report'].parent}")

    def save_package(self, run_id, folder):
        """Export files, PNG figures (if the run has a field) and report.html of a run in a new subfolder.

        An existing package is never overwritten: a subfolder that already holds files is refused (FileExistsError)."""
        from kkvim_gui.backend import report

        run = self.runs.get(run_id)
        target = Path(folder) / f"kkvim_{run.started:%Y%m%d-%H%M%S}_run{run.run_id:03d}"
        render = view = None
        if run.outcome.has_field:
            from kkvim_gui.analysis import snapshots

            render = snapshots.renderer(self.session, run.outcome)
            view = snapshots.view_record(run.outcome)
        paths = report.write_package(run.outcome, target, run.meta(self.runs.clock()), run.detailed, render, view)
        run.log("ИНФО", f"Расчёт и отчёт сохранены: {target}")
        return paths

    def export_run(self, run_id, folder):
        """Write manifest.json, history.csv and the field file of its role into a new subfolder of ``folder``."""
        run = self.runs.get(run_id)
        target = Path(folder) / f"kkvim_{run.started:%Y%m%d-%H%M%S}_run{run.run_id:03d}"
        paths = export.write_result(run.outcome, target, run.meta(self.runs.clock()), run.detailed)
        run.log("ИНФО", f"Результаты сохранены: {target}")
        return paths

    def closeEvent(self, event):
        if self.confirm_close and not self.confirm_discard():
            event.ignore()
            return
        self.runs.shutdown()
        super().closeEvent(event)

    def selected_run(self):
        return self.run_view.selected()

    def run_rows(self):
        """Input parameters of the selected run as entered (for the analysis window)."""
        run = self.selected_run()
        return summary_rows(run.snapshot if run is not None else self.run_snapshot)

    def analysis_run(self):
        """The selected run if it has results (for the analysis window), else None."""
        run = self.selected_run()
        return run if run is not None and (run.has_results or run.has_diagnostic_field) else None

    def _confirm_stop(self, run_id):
        run = self.runs.get(run_id)
        d2_running = run is not None and getattr(run, "_d2_job", None) is not None
        answer = QMessageBox.question(
            self,
            "Остановить расчёт",
            ("Остановить расчёт МКР D2? Результат КК-ВИМ этого запуска сохраняется; значение D2 не будет получено."
             if d2_running else "Остановить расчёт? Анализ прерванного расчёта недоступен.")
            + (" Следующие точки серии нагрузок не будут рассчитаны." if run is not None and run.series_id is not None
               else ""),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer == QMessageBox.StandardButton.Yes:
            self.runs.stop(run_id)
