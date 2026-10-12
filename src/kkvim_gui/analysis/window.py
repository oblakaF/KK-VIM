"""Analysis window (GUI_DESIGN.md, section 6).

Opened only by the "Открыть анализ результатов" button; several windows may be open.
Horizontal tabs at the top, no side menu. Each tab has its own toolbar; tabs with
figures carry the session-wide "Графики RU | EN" switch. P1 shows the layout with
TEST DATA placeholders; real figures arrive at P3-P5.
"""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QMainWindow,
    QPushButton,
    QTableWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from kkvim_gui import placeholders
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

    def __init__(self, session, model_rows=None, parent=None):
        """model_rows: input parameters of the analysed run; default: demonstration rows."""
        super().__init__(parent)
        self.session = session
        self.model_rows = tuple(model_rows) if model_rows else MODEL_ROWS
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setWindowIcon(app_icon())
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
        # Narrow window (high Windows scaling): the data note duplicates the TEST DATA badge.
        self.data_note.setVisible(self.width() >= 1100)

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
