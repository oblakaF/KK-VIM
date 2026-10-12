"""Start state (GUI_DESIGN.md, 5.1)."""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from kkvim_gui import placeholders, program_version
from kkvim_gui.widgets.common import Card, label


class StartView(QWidget):
    newRequested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Page")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.setSpacing(12)

        left = QVBoxLayout()
        left.setSpacing(12)
        actions = Card("Начало работы")
        self.new_button = QPushButton("+  Новый расчёт")
        self.new_button.setProperty("variant", "primary")
        self.new_button.setToolTip("Новый расчёт (Ctrl+N)")
        self.new_button.clicked.connect(self.newRequested)
        self.open_project_button = QPushButton("Открыть проект…")
        self.open_project_button.setEnabled(False)
        self.open_project_button.setToolTip("Чтение файлов .kkv — этап P2")
        self.open_results_button = QPushButton("Открыть результаты…")
        self.open_results_button.setEnabled(False)
        self.open_results_button.setToolTip("Открытие сохранённых результатов — после этапа P4")
        for b in (self.new_button, self.open_project_button, self.open_results_button):
            actions.body.addWidget(b)
        left.addWidget(actions)

        status = Card("Состояние программы")
        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(6)
        rows = (
            ("Решатель", "не подключён"),
            ("Режим", "демонстрационный (TEST DATA)"),
            ("Версия", f"KK-VIM {program_version()}"),
        )
        for i, (k, v) in enumerate(rows):
            grid.addWidget(label(k, "fieldLabel"), i, 0)
            grid.addWidget(label(v), i, 1)
        grid.setColumnStretch(1, 1)
        status.body.addLayout(grid)
        left.addWidget(status)
        left.addStretch(1)

        left_box = QWidget()
        left_box.setLayout(left)
        left_box.setFixedWidth(320)
        lay.addWidget(left_box)

        recent = Card("Недавние проекты", right=label("демонстрационные записи", "tag"))
        self.table = QTableWidget(len(placeholders.RECENT_PROJECTS), 6)
        self.table.setHorizontalHeaderLabels(
            ["Имя", "Пластина", "Закрепление", "N", "Изменён", "Результат"]
        )
        for r, row in enumerate(placeholders.RECENT_PROJECTS):
            for c, value in enumerate(row):
                item = QTableWidgetItem(value)
                if c in (3,):
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
        lay.addWidget(recent, 1)
