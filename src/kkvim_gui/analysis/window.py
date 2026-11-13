"""Analysis window (GUI_DESIGN.md, section 6).

Opened only by the "Открыть анализ результатов" button; several windows may be open.
Horizontal tabs at the top, no side menu. Each tab has its own toolbar; tabs with
figures carry the session-wide "Графики RU | EN" switch.

- A solver run (``run`` with a ``RunOutcome``): every tab is drawn from the solver arrays and rows
  (``analysis.results``); «Экспорт ▾» writes JSON and CSV with units and statuses. The window keeps
  its run: later changes in the main window do not affect it.
- «Сравнение с эталоном» (``analysis.comparison``): the run next to independent references — for SSSS the Navier
  reference computed for its own inputs in a separate process (``RunRecord.navier``; while it is computed the tab says so
  and is rebuilt when it arrives), for CCCC the saved references read as data (``backend.benchmarks``); for a
  diagnostic field or inputs without a reference it states why, with no numbers.
- «Нагрузка — прогиб» (``analysis.series``): only for a run of a finished load series.
- Without a solver run (``--demo``): the layout with TEST DATA placeholders, never results.
"""

from pathlib import Path

from PySide6.QtCore import Qt, QTimer, Signal
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

TABS = ("Результат", "Прогиб", "Сечения", "Значения в узлах", "Итерации", "Сравнение с эталоном")
SOLVER_TABS = ("Результат", "Прогиб", "Сечения", "Значения в узлах", "Итерации", "Диагностика", "Сравнение с эталоном")
SERIES_TAB = "Нагрузка — прогиб"
SOLVER_TAB_HELP = ("results.overview", "results.overview", "results.overview", "results.overview",
                   "iterations.overview", "iterations.overview", "verification.overview")
TAB_HELP = (
    "results.overview", "results.overview", "results.overview",
    "results.overview", "iterations.overview", "verification.overview",
)
COMPARISON_TOOLTIP = "Сравнение с эталонами — только для расчёта решателя, не для демонстрационных данных"
COMPARISON_SOLVER_TOOLTIP = ("Текущий расчёт рядом с независимым эталоном: SSSS — решение Навье для этих входных данных; "
                             "CCCC — сохранённые эталоны (опубликованное решение, D2, МКЭ). Знаковые разности без порога")
SECTIONS_TOOLTIP = "Профили прогиба вдоль выбранных линий узлов сетки: w(x) при y = const и w(y) при x = const"
TAB_TOOLTIPS = {"Прогиб": "Поле прогиба w(x, y): карта с изолиниями или цветной 3D-вид",
                "Сечения": SECTIONS_TOOLTIP, "Значения в узлах": "Таблица w во всех узлах сетки результата",
                "Итерации": "Как изменялся результат по полушагам s; подробная таблица — по кнопке",
                "Диагностика": "Энергия Π_h по полушагам, невязка, параметры решателя и ограничения маршрута"}
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

    def __init__(self, session, model_rows=None, run=None, series=None, parent=None):
        """model_rows: input parameters of the analysed run; run: a finished solver run (None: demonstration);
        series: ``model.series.SeriesData`` of the finished load series of the run, if any."""
        super().__init__(parent)
        self.session = session
        self.series = series
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
        self.status_run, self.status_stop, self.status_accept = results.status_labels(outcome, run)
        for w in (self.status_run, self.status_stop, self.status_accept):
            strip_lay.addWidget(w, 1)  # long statuses (diagnostics) wrap instead of being cut
        self.data_note = label(f"Решатель KK-VIM {outcome.solver}" + (f" · {outcome.mark}" if outcome.mark else ""),
                               "secondary")
        strip_lay.addWidget(self.data_note)
        lay.addWidget(strip)

        elapsed = format_duration(run._elapsed_final or 0.0)
        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.TabPosition.North)
        self.tabs.setDocumentMode(True)
        s = self.session
        from kkvim_gui.analysis.comparison import ComparisonTab

        self._navier = getattr(run, "navier", None)  # None: a run without a reference record
        self.comparison = ComparisonTab(outcome, run, navier=self._navier)  # states itself when no reference applies
        self.overview = results.OverviewTab(s, outcome, run, self.model_rows, elapsed, comparison=self.comparison.comparison)
        self.fields = results.FieldsTab(s, outcome, self.model_rows, trailing=self._trailing())
        self.sections = results.SectionsTab(s, outcome, trailing=self._trailing())
        self.values = results.ValuesTab(outcome, trailing=(self._export_button(),))
        self.history = results.HistoryTab(s, outcome, trailing=self._trailing())
        self.energy = results.EnergyTab(s, outcome, trailing=self._trailing())
        for widget, name in zip((self.overview, self.fields, self.sections, self.values, self.history, self.energy,
                                 self.comparison), SOLVER_TABS, strict=True):
            self.tabs.addTab(widget, name)
            if name in TAB_TOOLTIPS:
                self.tabs.setTabToolTip(self.tabs.count() - 1, TAB_TOOLTIPS[name])
        self.tabs.setTabToolTip(6, COMPARISON_SOLVER_TOOLTIP)
        for i, topic in enumerate(SOLVER_TAB_HELP):
            set_help_context(self.tabs.widget(i), topic)
        self.series_tab = None
        if self.series is not None:
            from kkvim_gui.analysis.series import SeriesTab

            self.series_tab = SeriesTab(s, self.series)
            self.tabs.addTab(self.series_tab, SERIES_TAB)
        self._navier_timer = None
        if self._navier is not None and self._navier.get("state") == "pending":
            self._navier_timer = QTimer(self)
            self._navier_timer.setInterval(300)
            self._navier_timer.timeout.connect(self._check_navier)
            self._navier_timer.start()
        self.report_button = None  # the report constructor (GUI-D18, P7) is not offered as a disabled button
        self.save_button = QPushButton("Сохранить расчёт и отчёт…")
        self.save_button.setProperty("variant", "primary")
        self.save_button.setToolTip("Папка расчёта: manifest.json, history.csv, field.csv, рисунки PNG, report.html и, "
                                    "если есть сравнение с эталоном, comparison.csv")
        self.save_button.clicked.connect(self._save_package_dialog)
        corner = QWidget()
        corner_lay = QHBoxLayout(corner)
        corner_lay.setContentsMargins(0, 4, 16, 4)
        corner_lay.addWidget(self.save_button)
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

    def _check_navier(self):
        """The Navier reference of an SSSS run arrived after the window opened: rebuild the comparison with it."""
        navier = getattr(self.run, "navier", None)
        if navier is None or navier is self._navier or navier.get("state") == "pending":
            return
        self._navier_timer.stop()
        self.set_navier(self.run.navier)

    def set_navier(self, navier):
        from kkvim_gui.analysis.comparison import ComparisonTab

        self._navier = navier
        index = self.tabs.indexOf(self.comparison)
        current = self.tabs.currentIndex()
        old = self.comparison
        self.comparison = ComparisonTab(self.outcome, self.run, navier=navier)
        self.tabs.removeTab(index)
        self.tabs.insertTab(index, self.comparison, SOLVER_TABS[6])
        self.tabs.setTabToolTip(index, COMPARISON_SOLVER_TOOLTIP)
        set_help_context(self.comparison, SOLVER_TAB_HELP[6])
        self.tabs.setCurrentIndex(current)
        old.deleteLater()
        self.overview.set_comparison(self.comparison.comparison)

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
        """Calculation package of this run (export files, PNG figures, report.html and, when the "Сравнение" tab
        shows a comparison, comparison.csv: the tab's own Comparison object) in a new subfolder."""
        from kkvim_gui.analysis import snapshots
        from kkvim_gui.backend import report

        factor = self.fields.surface.factor
        meta = self.run.meta(self.run._t0 + (self.run._elapsed_final or 0.0))
        return report.write_package(self.outcome, Path(folder) / self._base_name(), meta, self.run.detailed,
                                    render=snapshots.renderer(self.session, self.outcome, factor),
                                    view=snapshots.view_record(self.outcome, factor),
                                    comparison=self.comparison.comparison)

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
