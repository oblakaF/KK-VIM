"""Analysis window (GUI_DESIGN.md, section 6).

Opened only by the "Открыть анализ результатов" button; several windows may be open.
Horizontal tabs at the top, no side menu. Each tab has its own toolbar; tabs with
figures carry the session-wide "Графики RU | EN" switch.

- A solver run (``run`` with a ``RunOutcome``): every tab is drawn from the solver arrays and rows
  (``analysis.results``); «Экспорт ▾» writes JSON and CSV with units and statuses. The window keeps
  its run: later changes in the main window do not affect it.
- Without a solver run (``--demo``): the layout with TEST DATA placeholders, never results.
"""

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from kkvim_gui import placeholders
from kkvim_gui.backend import export
from kkvim_gui.theme.tokens import SIZES
from kkvim_gui.widgets.common import (
    Card,
    FigureLanguageSwitch,
    FigurePlaceholder,
    SegmentedControl,
    StatusLabel,
    fit_to_screen,
    label,
    set_help_context,
)
from kkvim_gui.widgets.header import AppHeader, app_icon

TABS = ("Обзор", "Поля", "Сечения", "Численные значения", "Сходимость", "Сравнение")
SOLVER_TABS = ("Обзор", "Поля", "Сечения", "Численные значения", "История полушагов", "Энергия и диагностика",
               "Сравнение")
SOLVER_TAB_HELP = ("results.overview", "results.overview", "results.overview", "results.overview",
                   "iterations.overview", "iterations.overview", "verification.overview")
TAB_HELP = (
    "results.overview", "results.overview", "results.overview",
    "results.overview", "iterations.overview", "verification.overview",
)
COMPARISON_TOOLTIP = "Сравнение с эталонами — после появления независимых эталонов (G4)"
REPORT_TOOLTIP = (
    "Конструктор отчёта: выбор рисунков, полей и таблиц, PDF/PNG/SVG, XLSX, язык рисунков, "
    "папка отчёта с manifest.json — после появления графиков и таблиц (GUI_DESIGN.md, 18)"
)


def _toolbar(*widgets, session=None, export=True):
    bar = QFrame()
    bar.setObjectName("TabToolbar")
    lay = QHBoxLayout(bar)
    lay.setContentsMargins(16, 6, 16, 6)
    lay.setSpacing(10)
    for w in widgets:
        lay.addWidget(w)
    lay.addStretch(1)
    if session is not None:
        lay.addWidget(FigureLanguageSwitch(session))
    if export:
        b = QPushButton("Экспорт ▾")
        b.setEnabled(False)
        b.setToolTip("Данные CSV — этап P5; рисунок для публикации — этап P6")
        lay.addWidget(b)
    return bar


def _labelled(text, widget):
    box = QWidget()
    lay = QHBoxLayout(box)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(6)
    lay.addWidget(label(text, "fieldLabel"))
    lay.addWidget(widget)
    return box


def _page(toolbar, content):
    page = QWidget()
    page.setObjectName("Page")
    lay = QVBoxLayout(page)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(0)
    lay.addWidget(toolbar)
    lay.addWidget(content, 1)
    return page


def _padded(layout):
    w = QWidget()
    layout.setContentsMargins(16, 14, 16, 16)
    layout.setSpacing(12)
    w.setLayout(layout)
    return w


def _key_value_grid(rows):
    grid = QGridLayout()
    grid.setHorizontalSpacing(16)
    grid.setVerticalSpacing(5)
    grid.setColumnStretch(1, 1)
    for i, (k, v) in enumerate(rows):
        grid.addWidget(label(k, "fieldLabel"), i, 0)
        grid.addWidget(label(v), i, 1)
    return grid


MODEL_ROWS = (
    ("Пластина", "a 2,000 m · b 1,000 m · h 10,0 mm"),
    ("Материал", "E 210 GPa · ν 0,300"),
    ("Нагрузка", "равномерная, q 5,000 kPa"),
    ("Закрепление", "SSSS"),
    ("Метод", "КК-ВИМ, N = 3 · Кирхгоф–Ляв"),
)


class AnalysisWindow(QMainWindow):
    settingsRequested = Signal()
    helpRequested = Signal()

    def __init__(self, session, model_rows=None, run=None, parent=None):
        """model_rows: input parameters of the analysed run; run: a finished solver run (None: demonstration)."""
        super().__init__(parent)
        self.session = session
        self.model_rows = tuple(model_rows) if model_rows else MODEL_ROWS
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setWindowIcon(app_icon())
        has_outcome = run is not None and not run.is_demo and run.outcome is not None and run.outcome.has_field
        # an error exit with a field is opened only as diagnostics, marked as an unfinished run
        self.run = run if has_outcome else None
        self.outcome = self.run.outcome if has_outcome else None
        if self.outcome is not None:
            self._build_solver()
            return
        self.setWindowTitle(f"КК-ВИМ: анализ — {placeholders.RUN_TITLE}")

        central = QWidget()
        central.setObjectName("Page")
        lay = QVBoxLayout(central)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self.header = AppHeader(session, "КК-ВИМ: анализ результатов")
        self.header.set_project(placeholders.RUN_TITLE)
        self.header.settingsClicked.connect(self.settingsRequested)
        self.header.helpClicked.connect(self.helpRequested)
        lay.addWidget(self.header)
        lay.addWidget(self._status_strip())

        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.TabPosition.North)
        self.tabs.setDocumentMode(True)
        self.tabs.addTab(self._overview_tab(), TABS[0])
        self.tabs.addTab(self._fields_tab(), TABS[1])
        self.tabs.addTab(self._sections_tab(), TABS[2])
        self.tabs.addTab(self._values_tab(), TABS[3])
        self.tabs.addTab(self._convergence_tab(), TABS[4])
        self.tabs.addTab(QWidget(), TABS[5])
        self.tabs.setTabEnabled(5, False)
        self.tabs.setTabToolTip(5, COMPARISON_TOOLTIP)
        for i, topic in enumerate(TAB_HELP):
            set_help_context(self.tabs.widget(i), topic)
        self.report_button = QPushButton("Собрать отчёт…")
        self.report_button.setEnabled(False)
        self.report_button.setToolTip(REPORT_TOOLTIP)
        corner = QWidget()
        corner_lay = QHBoxLayout(corner)
        corner_lay.setContentsMargins(0, 4, 16, 4)
        corner_lay.addWidget(self.report_button)
        self.tabs.setCornerWidget(corner, Qt.Corner.TopRightCorner)
        self.tabs.setCurrentIndex(1)
        lay.addWidget(self.tabs, 1)
        self.setCentralWidget(central)

        self.statusBar().addWidget(label("●  Результаты загружены · TEST DATA"), 1)
        fit_to_screen(self)

    def help_context(self):
        """F1 topic of the current tab."""
        return self.tabs.currentWidget().property("helpContext") or None

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Narrow window (high Windows scaling): the data note duplicates the badge.
        self.data_note.setVisible(self.width() >= 1100)

    # Solver result -----------------------------------------------------------------------------------

    def _build_solver(self):
        from kkvim_gui.analysis import results
        from kkvim_gui.model.runs import format_duration

        run, outcome = self.run, self.outcome
        title = f"запуск {run.run_id} · {run.started:%d.%m.%Y %H:%M:%S}"
        self.setWindowTitle(f"КК-ВИМ: анализ — {title}")
        central = QWidget()
        central.setObjectName("Page")
        lay = QVBoxLayout(central)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self.header = AppHeader(self.session, "КК-ВИМ: анализ результатов",
                                show_badge=bool(outcome.mark) or outcome.is_error or outcome.diagnostic_only)
        if outcome.is_error or outcome.diagnostic_only:
            cause = f"Решатель завершился ошибкой {outcome.status}" if outcome.is_error else                 f"Запрошенный расчёт не выполнен ({outcome.status})"
            self.header.set_badge("ДИАГНОСТИКА · расчёт не завершён", "ДИАГНОСТИКА",
                                  f"{cause}: поле последнего выполненного полушага показано только для диагностики, "
                                  "это не результат")
        elif outcome.mark:
            decisions = ", ".join(outcome.decisions)
            self.header.set_badge(outcome.mark, "RESEARCH" if outcome.mark.startswith("RESEARCH_ONLY") else "PROVISIONAL",
                                  f"Решатель {outcome.solver} ({decisions}): результаты не квалифицированы, не научная приёмка")
        self.header.set_project(title)
        self.header.settingsClicked.connect(self.settingsRequested)
        self.header.helpClicked.connect(self.helpRequested)
        lay.addWidget(self.header)

        strip = QFrame()
        strip.setObjectName("StatusStrip")
        strip_lay = QHBoxLayout(strip)
        strip_lay.setContentsMargins(16, 6, 16, 6)
        strip_lay.setSpacing(24)
        (k1, t1), (k2, t2), (k3, t3) = results.statuses(outcome, run)
        self.status_run, self.status_stop, self.status_accept = (StatusLabel(k1, t1), StatusLabel(k2, t2),
                                                                 StatusLabel(k3, t3))
        for w in (self.status_run, self.status_stop, self.status_accept):
            strip_lay.addWidget(w)
        strip_lay.addStretch(1)
        self.data_note = label(f"Данные: решатель KK-VIM {outcome.solver}" + (f" · {outcome.mark}" if outcome.mark else ""),
                               "secondary")
        strip_lay.addWidget(self.data_note)
        lay.addWidget(strip)

        elapsed = format_duration(run._elapsed_final or 0.0)
        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.TabPosition.North)
        self.tabs.setDocumentMode(True)
        s = self.session
        self.overview = results.OverviewTab(s, outcome, run, self.model_rows, elapsed)
        self.fields = results.FieldsTab(s, outcome, self.model_rows, trailing=self._trailing())
        self.sections = results.SectionsTab(s, outcome, trailing=self._trailing())
        self.values = results.ValuesTab(outcome, trailing=(self._export_button(),))
        self.history = results.HistoryTab(s, outcome, trailing=self._trailing())
        self.energy = results.EnergyTab(s, outcome, trailing=self._trailing())
        for widget, name in zip((self.overview, self.fields, self.sections, self.values, self.history, self.energy,
                                 QWidget()), SOLVER_TABS, strict=True):
            self.tabs.addTab(widget, name)
        self.tabs.setTabEnabled(6, False)
        self.tabs.setTabToolTip(6, COMPARISON_TOOLTIP)
        for i, topic in enumerate(SOLVER_TAB_HELP):
            set_help_context(self.tabs.widget(i), topic)
        self.report_button = QPushButton("Собрать отчёт…")
        self.report_button.setEnabled(False)
        self.report_button.setToolTip(REPORT_TOOLTIP)
        self.save_button = QPushButton("Сохранить расчёт и отчёт…")
        self.save_button.setToolTip("Папка расчёта: manifest.json, history.csv, field.csv, рисунки PNG и "
                                    "читаемый отчёт report.html")
        self.save_button.clicked.connect(self._save_package_dialog)
        corner = QWidget()
        corner_lay = QHBoxLayout(corner)
        corner_lay.setContentsMargins(0, 4, 16, 4)
        corner_lay.addWidget(self.save_button)
        corner_lay.addWidget(self.report_button)
        self.tabs.setCornerWidget(corner, Qt.Corner.TopRightCorner)
        self.tabs.setCurrentIndex(0)
        lay.addWidget(self.tabs, 1)
        self.setCentralWidget(central)
        mark = f" · {outcome.mark}" if outcome.mark else ""
        if outcome.is_error or outcome.diagnostic_only:
            cause = "расчёт завершился ошибкой" if outcome.is_error else "запрошенный расчёт не выполнен,"
            self.statusBar().addWidget(label(f"●  ДИАГНОСТИКА: {cause} {outcome.status}; "
                                             f"поле — последнего выполненного полушага, не результат{mark}"), 1)
        else:
            self.statusBar().addWidget(label(f"●  Результаты решателя KK-VIM загружены{mark}"), 1)
        fit_to_screen(self)

    def _trailing(self):
        return (FigureLanguageSwitch(self.session), self._export_button())

    def _export_button(self):
        button = QToolButton()
        button.setText("Экспорт ▾")
        button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        menu = QMenu(button)
        menu.addAction("Все результаты: JSON + CSV…", self._export_all_dialog)
        menu.addAction("История полушагов, CSV…", self._export_history_dialog)
        menu.addAction("Поле прогиба, CSV…", self._export_field_dialog)
        button.setMenu(menu)
        button.setToolTip("Исходные данные, результаты и история вычислений с единицами и научными статусами")
        return button

    def _base_name(self):
        return f"kkvim_{self.run.started:%Y%m%d-%H%M%S}_run{self.run.run_id:03d}"

    def export_all(self, folder):
        meta = self.run.meta(self.run._t0 + (self.run._elapsed_final or 0.0))
        return export.write_result(self.outcome, Path(folder) / self._base_name(), meta, self.run.detailed)

    def save_package(self, folder):
        """Calculation package of this run (export files, PNG figures, report.html) in a new subfolder."""
        from kkvim_gui.analysis import snapshots
        from kkvim_gui.backend import report

        factor = self.fields.surface.factor
        meta = self.run.meta(self.run._t0 + (self.run._elapsed_final or 0.0))
        return report.write_package(self.outcome, Path(folder) / self._base_name(), meta, self.run.detailed,
                                    render=snapshots.renderer(self.session, self.outcome, factor),
                                    view=snapshots.view_record(self.outcome, factor))

    def _save_package_dialog(self):
        folder = QFileDialog.getExistingDirectory(self, "Папка для расчёта и отчёта")
        if not folder:
            return
        try:
            paths = self.save_package(folder)
        except OSError as exc:
            QMessageBox.warning(self, "Сохранение не выполнено", f"Расчёт не сохранён: {exc}")
            return
        self.statusBar().showMessage(f"Сохранено: {paths['report'].parent}", 10000)

    def _export_all_dialog(self):
        folder = QFileDialog.getExistingDirectory(self, "Папка для результатов")
        if folder:
            try:
                paths = self.export_all(folder)
            except OSError as exc:  # an existing package is never overwritten (export.new_folder)
                QMessageBox.warning(self, "Сохранение не выполнено", f"Результаты не сохранены: {exc}")
                return
            self.statusBar().showMessage(f"Сохранено: {paths['manifest'].parent}", 8000)

    def _export_history_dialog(self):
        path, _ = QFileDialog.getSaveFileName(self, "История полушагов", f"{self._base_name()}_history.csv",
                                              "CSV (*.csv)")
        if path:
            export.write_history_csv(self.outcome, path)
            self.statusBar().showMessage(f"Сохранено: {path}", 8000)

    def _export_field_dialog(self):
        name = export.field_file_name(self.outcome)  # the name of its role: field / field_research / field_diagnostic
        path, _ = QFileDialog.getSaveFileName(self, "Поле прогиба", f"{self._base_name()}_{name}", "CSV (*.csv)")
        if path:
            export.write_field_csv(self.outcome, path)
            self.statusBar().showMessage(f"Сохранено: {path}", 8000)

    def _status_strip(self):
        strip = QFrame()
        strip.setObjectName("StatusStrip")
        lay = QHBoxLayout(strip)
        lay.setContentsMargins(16, 6, 16, 6)
        lay.setSpacing(24)
        self.status_run = StatusLabel("success", "Расчёт выполнен")
        self.status_stop = StatusLabel(
            "success", f"Критерий остановки: k = {placeholders.FINISHED['k_stop']}"
        )
        self.status_accept = StatusLabel("neutral", "Научная приёмка: не проводилась")
        for s in (self.status_run, self.status_stop, self.status_accept):
            lay.addWidget(s)
        lay.addStretch(1)
        self.data_note = label("Данные: демонстрационные (TEST DATA)", "secondary")
        lay.addWidget(self.data_note)
        return strip

    def _overview_tab(self):
        grid = QGridLayout()
        inputs = Card("Входные данные")
        inputs.body.addLayout(_key_value_grid(self.model_rows))
        inputs.body.addStretch(1)
        statuses = Card("Статусы")
        for status, text in (
            ("success", "Расчёт выполнен"),
            ("success", f"Критерий остановки выполнен на k = {placeholders.FINISHED['k_stop']}"),
            ("neutral", "Научная приёмка: не проводилась"),
        ):
            statuses.body.addWidget(StatusLabel(status, text))
        statuses.body.addStretch(1)
        f = placeholders.FINISHED
        key = Card("Ключевые значения", right=label("TEST DATA", "tag"))
        key.body.addLayout(_key_value_grid((
            ("Экстремум прогиба (по узлам)", f["extremum"]),
            ("Итоговая невязка (предварительно)", f["residual"]),
            ("Время расчёта", f["elapsed"]),
        )))
        key.body.addStretch(1)
        grid.addWidget(inputs, 0, 0)
        grid.addWidget(statuses, 0, 1)
        grid.addWidget(key, 0, 2)
        grid.addWidget(FigurePlaceholder(self.session, "field.w.title", "Миниатюра поля — этап P4"), 1, 0, 1, 2)
        grid.addWidget(
            FigurePlaceholder(self.session, "conv.title", "Миниатюра сходимости — этап P4"), 1, 2
        )
        for c in range(3):
            grid.setColumnStretch(c, 1)
        grid.setRowStretch(1, 1)
        return _page(_toolbar(session=self.session, export=False), _padded(grid))

    def _fields_tab(self):
        quantity = QComboBox()
        quantity.addItem("Прогиб w, mm")
        quantity.addItem("Изгибающие моменты — планируется")
        quantity.model().item(1).setEnabled(False)
        # The quantity list may shrink on narrow windows; the toggles may not.
        quantity.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        quantity.setMinimumContentsLength(10)
        mode = SegmentedControl({"field": "Поле", "iso": "Изолинии", "both": "Оба"}, "field")
        cmap = QComboBox()
        cmap.addItems(["viridis", "cividis"])
        toolbar = _toolbar(
            _labelled("Величина", quantity), mode, _labelled("Шкала", cmap), session=self.session
        )

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        figure_box = QVBoxLayout()
        figure_box.setContentsMargins(16, 14, 16, 16)
        figure_box.addWidget(FigurePlaceholder(
            self.session, "field.w.title",
            "Цветовое поле w(x, y) с изолиниями и цветовой шкалой — этап P4.\n"
            "Поле статично: интерактивное управление полем не реализуется (GUI-D13).",
        ))
        body.addLayout(figure_box, 1)
        body.addWidget(self._inspector())
        content = QWidget()
        content.setLayout(body)
        return _page(toolbar, content)

    def _inspector(self):
        panel = QFrame()
        panel.setObjectName("Inspector")
        panel.setFixedWidth(SIZES["inspector.width"])
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(16, 14, 16, 16)
        lay.setSpacing(6)
        lay.addWidget(label("Экстремум (по узлам)", "cardTitle"))
        lay.addWidget(label(f"w {placeholders.FINISHED['extremum']}"))
        lay.addWidget(label("координаты узла — этап P4", "secondary"))
        lay.addSpacing(10)
        lay.addWidget(label("Параметры модели", "cardTitle"))
        for k, v in self.model_rows:
            lay.addWidget(label(k, "fieldLabel"))
            lay.addWidget(label(v, wrap=True))
        lay.addStretch(1)
        lay.addWidget(label("TEST DATA", "figureMark"), 0, Qt.AlignmentFlag.AlignLeft)
        return panel

    def _sections_tab(self):
        position = QComboBox()
        position.addItems(["y = 0,500 m", "y = 0,250 m"])
        position.setEnabled(False)
        position.setToolTip("Положение сечения задаётся числом или списком узлов — этап P5")
        toolbar = _toolbar(_labelled("Сечение", position), session=self.session)
        row = QHBoxLayout()
        row.addWidget(FigurePlaceholder(
            self.session, "section.x.title", "График w(x) — этап P5", {"y": 0.5}, xy=True
        ))
        row.addWidget(FigurePlaceholder(
            self.session, "section.y.title", "График w(y) — этап P5", {"x": 1.0}, xy=True
        ))
        return _page(toolbar, _padded(row))

    def _values_tab(self):
        table = QTableWidget(0, 3)
        table.setHorizontalHeaderLabels(["x, m", "y, m", "w, mm"])
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        card = Card("Значения в узлах сетки результата", right=label("этап P5", "tag"))
        card.body.addWidget(table)
        box = QVBoxLayout()
        box.addWidget(card)
        return _page(_toolbar(export=True), _padded(box))

    def _convergence_tab(self):
        row = QHBoxLayout()
        row.addWidget(FigurePlaceholder(
            self.session, "conv.title",
            "Невязка по итерациям k, логарифмическая шкала — этап P3/P5", xy=True,
        ), 2)
        table = QTableWidget(0, 2)
        table.setHorizontalHeaderLabels(["k", "Невязка (предварительно)"])
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        card = Card("Итерации", right=label("этап P5", "tag"))
        card.body.addWidget(table)
        row.addWidget(card, 1)
        return _page(_toolbar(session=self.session), _padded(row))
