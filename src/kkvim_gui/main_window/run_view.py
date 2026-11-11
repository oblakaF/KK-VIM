"""«Ход расчёта»: run dispatcher of the session (GUI_DESIGN.md, 5.3-5.4).

- Top: summary of the selected run — status with its own duration, the three separate
  statuses (execution, stopping criterion, scientific acceptance), available results and
  «Открыть анализ результатов» (finished runs only) or «Стоп» (active run).
- Middle: the dispatcher table (date and time, method, method parameters, status,
  individual duration) and details of the selected run.
- Bottom: collapsible «Подробный журнал»; normally only important messages are shown
  (in the details), and there are no per-iteration lines.
No convergence plot here: scientific figures live in the analysis window. No remaining-time
estimate. The demonstration backend is marked TEST DATA everywhere; every R4 result carries
PROVISIONAL / NOT_QUALIFIED and every research route (SSSS N = 4-6, CCCC) RESEARCH_ONLY / PROVISIONAL /
NOT_QUALIFIED, with the limits of its route. The scientific status comes from the solver outcome (or, while
the run is active, from the route of its request), never from N alone; a diagnostic field is never shown
as a result.
"""

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from kkvim_gui.backend.kkvim_adapter import LIMITATIONS, MARK, RESEARCH_MARK
from kkvim_gui.backend.request import ROUTE_R3, ROUTE_R4
from kkvim_gui.model.runs import STATUS_TEXT, format_duration
from kkvim_gui.model.setup_state import summary_rows
from kkvim_gui.widgets.common import Card, Note, StatusLabel, label, set_help_context

COLUMNS = ("Дата и время", "Метод", "Параметры", "Статус", "Длительность")
STATUS_KIND = {"running": "warning", "finished": "success", "stopped": "neutral", "error": "error"}
LEVEL_TOKEN = {"ПРЕДУПРЕЖДЕНИЕ": "status.warning", "ОШИБКА": "status.error"}
RESULTS_TEXT = (
    "Доступно в окне анализа: заготовки поля прогиба, сечений и сходимости с демонстрационными "
    "значениями — TEST DATA, не результаты решателя."
)
RESULTS_SOLVER = (
    "Доступно в окне анализа: поле прогиба, максимальный прогиб, безразмерный прогиб, сечения, история "
    "полушагов, энергия и диагностика — по массивам решателя KK-VIM."
)
RESULTS_R4 = RESULTS_SOLVER + " Все величины R4 — PROVISIONAL / NOT_QUALIFIED."
RESULTS_RESEARCH = RESULTS_SOLVER + f" Исследовательский расчёт: {RESEARCH_MARK}; не квалифицированный результат."
DIAGNOSTIC_FIELD = ("Запрошенный расчёт не выполнен: поле последнего выполненного полушага доступно только как диагностика "
                    "(field_diagnostic.csv), не как результат.")
LIMITS_SHOWN = 3  # the first limits of the route in the run summary; all of them in the analysis and the report
NO_FIELD = "Поле прогиба не получено. Протокол и диагностика доступны через «Сохранить результаты…»."


def _table(columns):
    t = QTableWidget(0, len(columns))
    t.setHorizontalHeaderLabels(list(columns))
    t.verticalHeader().setVisible(False)
    t.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    t.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    t.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
    t.setShowGrid(False)
    t.setWordWrap(False)
    t.horizontalHeader().setDefaultAlignment(Qt.AlignmentFlag.AlignLeft)
    return t


class RunView(QWidget):
    stopRequested = Signal(int)
    analysisRequested = Signal(int)
    exportRequested = Signal(int)

    def __init__(self, session, runs, parent=None):
        super().__init__(parent)
        self.setObjectName("Page")
        set_help_context(self, "iterations.overview")
        self.session = session
        self.runs = runs
        self.selected_id = None

        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 16, 16, 12)
        lay.setSpacing(12)
        lay.addWidget(self._summary_card())

        middle = QHBoxLayout()
        middle.setSpacing(12)
        dispatcher = Card("Диспетчер запусков", right=label("история текущего сеанса", "tag"))
        self.table = _table(COLUMNS)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        for c in (0, 1, 3, 4):
            header.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
        self.table.itemSelectionChanged.connect(self._on_selection)
        dispatcher.body.addWidget(self.table)
        middle.addWidget(dispatcher, 3)
        middle.addWidget(self._details_card(), 2)
        lay.addLayout(middle, 1)
        lay.addWidget(self._journal())

        runs.runAdded.connect(self._on_added)
        runs.runChanged.connect(self._on_changed)
        session.themeChanged.connect(self._on_theme)
        self._show_selected()

    # Construction
    def _summary_card(self):
        card = Card()
        self.summary_card = card
        head = QHBoxLayout()
        self.summary_title = label("Расчёты ещё не запускались", "pageTitle")
        head.addWidget(self.summary_title)
        head.addStretch(1)
        self.stop_button = QPushButton("■  Стоп")
        self.stop_button.setProperty("variant", "danger")
        self.stop_button.setToolTip("Остановить расчёт (с подтверждением)")
        self.stop_button.clicked.connect(lambda: self.stopRequested.emit(self.selected_id))
        self.analysis_button = QPushButton("Открыть анализ результатов")
        self.analysis_button.setProperty("variant", "primary")
        self.analysis_button.clicked.connect(lambda: self.analysisRequested.emit(self.selected_id))
        self.export_button = QPushButton("Сохранить результаты…")
        self.export_button.setToolTip("Исходные данные, статусы, история полушагов и поле: JSON и CSV с единицами")
        self.export_button.clicked.connect(lambda: self.exportRequested.emit(self.selected_id))
        for b in (self.stop_button, self.export_button, self.analysis_button):
            head.addWidget(b)
        card.body.addLayout(head)
        statuses = QHBoxLayout()
        statuses.setSpacing(24)
        self.status_run = StatusLabel("neutral", "")
        self.status_stop = StatusLabel("neutral", "")
        self.status_accept = StatusLabel("neutral", "Научная приёмка: не проводилась")
        for s in (self.status_run, self.status_stop, self.status_accept):
            statuses.addWidget(s)
        statuses.addStretch(1)
        card.body.addLayout(statuses)
        self.summary_text = label("", "secondary", wrap=True)
        card.body.addWidget(self.summary_text)
        self.results_note = Note(RESULTS_TEXT, "testdata")
        card.body.addWidget(self.results_note)
        self.limits_note = Note("", "testdata")  # the limits of a non-accepted route (the same texts as the report)
        self.limits_note.setVisible(False)
        card.body.addWidget(self.limits_note)
        return card

    def _details_card(self):
        self.details_tag = label("", "tag")
        card = Card("Выбранный запуск", right=self.details_tag)
        self.details_card = card
        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(5)
        grid.setColumnStretch(1, 1)
        self.details = {}
        rows = ("Проект", "Метод", "Параметры", "Пластина", "Этап", "Счётчики", "Прошло времени", "Протокол")
        for i, name in enumerate(rows):
            grid.addWidget(label(name, "fieldLabel"), i, 0, Qt.AlignmentFlag.AlignTop)
            value = label("", wrap=True)
            value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            grid.addWidget(value, i, 1)
            self.details[name] = value
        self.counter_label = grid.itemAtPosition(5, 0).widget()
        card.body.addLayout(grid)
        card.body.addWidget(label("Важные сообщения", "cardTitle"))
        self.important = label("", wrap=True)
        self.important.setTextFormat(Qt.TextFormat.RichText)
        self.important.setAlignment(Qt.AlignmentFlag.AlignTop)
        card.body.addWidget(self.important, 1)
        return card

    def _journal(self):
        box = QFrame()
        lay = QVBoxLayout(box)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        self.journal_toggle = QToolButton()
        self.journal_toggle.setCheckable(True)
        self.journal_toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.journal_toggle.setArrowType(Qt.ArrowType.RightArrow)
        self.journal_toggle.setText("Подробный журнал")
        self.journal_toggle.toggled.connect(self._toggle_journal)
        lay.addWidget(self.journal_toggle)
        self.journal = _table(("Время", "Уровень", "Событие"))
        self.journal.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.journal.setMaximumHeight(180)
        self.journal.setVisible(False)
        lay.addWidget(self.journal)
        return box

    # Updates
    def _toggle_journal(self, on):
        self.journal_toggle.setArrowType(Qt.ArrowType.DownArrow if on else Qt.ArrowType.RightArrow)
        self.journal.setVisible(on)

    def _on_added(self, run_id):
        row = self.table.rowCount()
        self.table.insertRow(row)
        for c in range(len(COLUMNS)):
            self.table.setItem(row, c, QTableWidgetItem())
        self._fill_row(row, self.runs.get(run_id))
        self.select(run_id)

    def _row_of(self, run_id):
        for r in range(self.table.rowCount()):
            if self.table.item(r, 0).data(Qt.ItemDataRole.UserRole) == run_id:
                return r
        return None

    def _fill_row(self, row, run):
        elapsed = format_duration(self.runs.elapsed(run))
        values = (
            run.started.strftime("%d.%m.%Y %H:%M:%S"),
            run.method,
            run.params_text(),
            STATUS_TEXT[run.status],
            elapsed,
        )
        colour = QColor(self.session.tokens[f"status.{STATUS_KIND[run.status]}"])
        for c, value in enumerate(values):
            item = self.table.item(row, c)
            item.setText(value)
            item.setData(Qt.ItemDataRole.UserRole, run.run_id)
            if c == 3:
                item.setForeground(colour)
            if c == 4:
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

    def _on_changed(self, run_id):
        row = self._row_of(run_id)
        if row is not None:
            self._fill_row(row, self.runs.get(run_id))
        if run_id == self.selected_id:
            self._show_selected()

    def _on_theme(self, _name):
        for run in self.runs.runs:
            self._on_changed(run.run_id)

    def _on_selection(self):
        items = self.table.selectedItems()
        if items:
            self.selected_id = items[0].data(Qt.ItemDataRole.UserRole)
            self._show_selected()

    def select(self, run_id):
        row = self._row_of(run_id)
        if row is not None:
            self.table.selectRow(row)
        self.selected_id = run_id
        self._show_selected()

    def selected(self):
        return self.runs.get(self.selected_id) if self.selected_id is not None else None

    def _show_selected(self):
        run = self.selected()
        has = run is not None
        for w in (self.status_run, self.status_stop, self.status_accept, self.summary_text):
            w.setVisible(has)
        self.stop_button.setVisible(has and run.active)
        diagnostic = has and run.has_diagnostic_field
        self.analysis_button.setVisible(has and (run.status == "finished" or diagnostic))
        self.analysis_button.setEnabled(has and (run.has_results or diagnostic))
        self.analysis_button.setText("Открыть диагностику (запрошенный расчёт не выполнен)" if diagnostic
                                     else "Открыть анализ результатов")
        self.export_button.setVisible(has and run.outcome is not None)
        outcome = run.outcome if has else None
        if has and run.is_demo:
            self.results_note.text.setText(RESULTS_TEXT)
        elif outcome is not None:
            self.results_note.text.setText(
                NO_FIELD if not outcome.has_field else DIAGNOSTIC_FIELD if outcome.diagnostic_only else
                RESULTS_R4 if outcome.mark == MARK else RESULTS_RESEARCH if outcome.mark else RESULTS_SOLVER)
        self.results_note.setVisible(has and (run.has_results or (outcome is not None and (not outcome.has_field
                                                                                          or outcome.diagnostic_only))))
        self.details_tag.setText(self._tag(run) if has else "")
        limits = self._limitations(run) if has else ()
        shown = [f"• {t}" for t in limits[:LIMITS_SHOWN]]
        if len(limits) > LIMITS_SHOWN:
            shown.append(f"… ещё {len(limits) - LIMITS_SHOWN}: полный перечень — в окне анализа и в report.html")
        self.limits_note.text.setText("Ограничения маршрута:\n" + "\n".join(shown))
        self.limits_note.setVisible(bool(limits))
        if not has:
            self.summary_title.setText("Расчёты ещё не запускались")
            for v in self.details.values():
                v.setText("—")
            self.important.setText("")
            self._fill_journal([])
            return
        elapsed = format_duration(self.runs.elapsed(run))
        self.summary_title.setText(f"{STATUS_TEXT[run.status]} — {elapsed}")
        self.status_run.set_status(STATUS_KIND[run.status], f"Выполнение: {STATUS_TEXT[run.status].lower()}")
        self.status_stop.set_status(*self._criterion(run))
        self.status_accept.set_status(*self._science(run))
        rows = dict(summary_rows(run.snapshot))
        mark = self._mark(run)
        tail = " · TEST DATA" if run.is_demo else (f" · {mark}" if mark else "")
        self.summary_text.setText(f"{run.method} · {run.params_text()} · {rows['Закрепление']} · "
                                  f"{rows['Геометрия']}{tail}")
        d = self.details
        d["Проект"].setText(run.project)
        d["Метод"].setText(f"{run.method} · {run.snapshot.theory}")
        d["Параметры"].setText(f"{run.params_text()} (N — число членов)")
        d["Пластина"].setText(f"{rows['Геометрия']} · {rows['Закрепление']} · {rows['Нагрузка']}")
        d["Этап"].setText(run.stage)
        self.counter_label.setVisible(run.iterative)
        d["Счётчики"].setVisible(run.iterative)
        if run.iterative:
            if run.s is None:
                d["Счётчики"].setText("s и k — после завершения: решатель не передаёт промежуточные полушаги")
            else:
                d["Счётчики"].setText(
                    f"s = {run.s} — выполненные координатные полушаги; "
                    f"k = {run.k} — завершённые циклы (D-030)"
                )
        d["Прошло времени"].setText(elapsed)
        if run.protocol_path:
            d["Протокол"].setText(str(run.protocol_path))
        else:
            d["Протокол"].setText("—" if run.is_demo or run.active else "не сохранён")
        t = self.session.tokens
        lines = []
        for time_, level, text in run.important:
            colour = t.get(LEVEL_TOKEN.get(level, "text.primary"))
            lines.append(f'<span style="color:{t["text.secondary"]}">{time_}</span> '
                         f'<span style="color:{colour}">{text}</span>')
        self.important.setText("<br>".join(lines))
        self._fill_journal(run.detailed)

    @classmethod
    def _limitations(cls, run):
        """Limits of a research or PROVISIONAL route (none for R3 and the demonstration)."""
        if run.is_demo or cls._mark(run) is None:
            return ()
        if run.outcome is not None:
            return run.outcome.limitations
        route = run.request.route if run.request is not None else run.snapshot.route
        return LIMITATIONS.get(route, ())

    @staticmethod
    def _mark(run):
        """The status mark of the run: from its outcome, or from the route of its request while it runs."""
        if run.outcome is not None:
            return run.outcome.mark
        route = run.request.route if run.request is not None else run.snapshot.route
        return None if route == ROUTE_R3 else MARK if route == ROUTE_R4 else RESEARCH_MARK

    @classmethod
    def _tag(cls, run):
        if run.is_demo:
            return "TEST DATA"
        mark = cls._mark(run)
        if run.outcome is not None and run.outcome.diagnostic_only:
            return f"ДИАГНОСТИКА · {mark}" if mark else "ДИАГНОСТИКА"
        return mark or "KK-VIM"

    @staticmethod
    def _criterion(run):
        if not run.iterative:
            return "neutral", "Критерий остановки: не применяется"
        if run.status == "running":
            return "neutral", "Критерий остановки: ещё не выполнен"
        if run.is_demo:
            if run.stop_criterion == "met":
                return "success", f"Критерий остановки выполнен на k = {run.k} (s = {run.s}) — демонстрация"
            return "neutral", "Критерий остановки: не выполнен"
        outcome = run.outcome
        if outcome is None:
            return "neutral", "Критерий остановки: не выполнен (результата нет)"
        kind = {"met": "success", "not_met": "warning", "error": "error", "none": "neutral"}[outcome.stop_kind]
        if outcome.mark and kind == "success":
            kind = "warning"  # an R4 stabilization is PROVISIONAL, not a success of the method
        return kind, f"{outcome.stop_text} (s = {outcome.s}, k = {outcome.k})"

    @classmethod
    def _science(cls, run):
        if run.is_demo:
            return "neutral", "Научная приёмка: не проводилась"
        mark = cls._mark(run)
        if mark == MARK:
            return "warning", "Научный статус: PROVISIONAL / NOT_QUALIFIED (R4, D-047, D-048)"
        if mark:
            decisions = ", ".join(run.outcome.decisions) if run.outcome is not None else ""
            return "warning", f"Научный статус: {mark}" + (f" ({decisions})" if decisions else "")
        return "neutral", "Научный статус: решатель R3 ACCEPTED (D-043); расчёт — не научная приёмка"

    def _fill_journal(self, rows):
        self.journal.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, value in enumerate(row):
                self.journal.setItem(r, c, QTableWidgetItem(value))
        self.journal.resizeColumnToContents(0)
        self.journal.resizeColumnToContents(1)
