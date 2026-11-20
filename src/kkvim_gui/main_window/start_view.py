"""Start state (GUI_DESIGN.md, 5.1; GUI-PLATE-WORKFLOW-UX-01).

«Новый расчёт», a short truthful readiness line and what the program does, step by step. Opening or saving projects
is not implemented and is not offered. The demonstration list of recent projects is shown only in the demonstration
mode (``--demo``, TEST DATA). The scientific status of a calculation is shown where the calculation is configured and
displayed, not here.
"""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from kkvim_gui import placeholders
from kkvim_gui.widgets.common import Card, StatusLabel, label

READY = "Готово к расчёту"
READY_TEXT = ("Прямоугольная пластина Кирхгофа–Лява под равномерной нагрузкой: шарнирное опирание (SSSS) или жёсткая "
              "заделка (CCCC), N = 1–6 членов. Статус выбранного варианта расчёта показывается при настройке и в результатах.")
DEMO_TEXT = "Демонстрационный режим: решатель не вызывается, все данные помечены TEST DATA."
STEPS = (
    ("1. Настройка", "Размеры, материал, нагрузка (одна или серия), закрепление краёв, число членов N и сетка."),
    ("2. Расчёт", "Решатель КК-ВИМ считает в отдельном процессе; окно остаётся доступным, расчёт можно остановить."),
    ("3. Результат", "Прогиб в центре и максимальный, поле и 3D-вид, сечения, итерации; для SSSS — сравнение с "
                     "независимым решением Навье; для серии — график «Нагрузка — прогиб»."),
    ("4. Сохранение", "Папка расчёта: отчёт report.html, CSV с полной точностью и manifest.json; для серии — "
                      "load_series.csv."),
)


class StartView(QWidget):
    newRequested = Signal()

    def __init__(self, demo=False, parent=None):
        super().__init__(parent)
        self.setObjectName("Page")
        # no help context: F1 on the start page opens the help search (GUI_DESIGN.md, 17.3)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.setSpacing(12)

        left = QVBoxLayout()
        left.setSpacing(12)
        actions = Card("Начало работы")
        self.new_button = QPushButton("+  Новый расчёт")
        self.new_button.setProperty("variant", "primary")
        self.new_button.setMinimumHeight(44)
        self.new_button.setToolTip("Новый расчёт (Ctrl+N)")
        self.new_button.clicked.connect(self.newRequested)
        actions.body.addWidget(self.new_button)
        self.ready = StatusLabel("neutral" if demo else "success", "Демонстрационный режим" if demo else READY)
        actions.body.addWidget(self.ready)
        self.ready_text = label(DEMO_TEXT if demo else READY_TEXT, "secondary", wrap=True)
        actions.body.addWidget(self.ready_text)
        actions.body.addWidget(label("Справка: F1 или меню ⚙ → «Справка» в верхней панели.", "secondary", wrap=True))
        left.addWidget(actions)
        left.addStretch(1)
        left_box = QWidget()
        left_box.setLayout(left)
        left_box.setFixedWidth(340)
        lay.addWidget(left_box)

        right = QVBoxLayout()
        right.setSpacing(12)
        steps = Card("Как проходит расчёт")
        for title, text in STEPS:
            steps.body.addWidget(label(title, "cardTitle"))
            steps.body.addWidget(label(text, wrap=True))
        right.addWidget(steps)
        self.table = None
        if demo:
            projects = placeholders.RECENT_PROJECTS
            recent = Card("Недавние проекты", right=label("демонстрационные записи", "tag"))
            self.table = QTableWidget(len(projects), 6)
            self.table.setHorizontalHeaderLabels(["Имя", "Пластина", "Закрепление", "N", "Изменён", "Результат"])
            for r, row in enumerate(projects):
                for c, value in enumerate(row):
                    item = QTableWidgetItem(value)
                    if c == 3:
                        item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                    self.table.setItem(r, c, item)
            self.table.verticalHeader().setVisible(False)
            self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
            self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
            self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
            self.table.setShowGrid(False)
            self.table.setAlternatingRowColors(True)
            self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
            self.table.horizontalHeader().setDefaultAlignment(Qt.AlignmentFlag.AlignLeft)
            self.table.setToolTip("Двойной щелчок открывает настройку расчёта")
            self.table.cellDoubleClicked.connect(lambda *_: self.newRequested.emit())
            recent.body.addWidget(self.table)
            right.addWidget(recent, 1)
        else:
            right.addStretch(1)
        lay.addLayout(right, 1)
