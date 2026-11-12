"""Start state (GUI_DESIGN.md, 5.1).

The state of the program is the real backend (the KK-VIM solver) unless the demonstration mode was
requested; the demonstration list of recent projects is shown only in that mode (saving projects .kkv
is not implemented yet).
"""

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

    def __init__(self, demo=False, parent=None):
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
        if demo:
            rows = (("Решатель", "не вызывается"), ("Режим", "демонстрационный (TEST DATA)"))
        else:
            rows = (("Решатель", ("KK-VIM: SSSS N = 1 — R3 (ACCEPTED, D-043); SSSS N = 2–3 — R4 (PROVISIONAL / NOT_QUALIFIED); "
                                  "SSSS N = 4–6 и CCCC N = 1–6 — только исследование (RESEARCH_ONLY / PROVISIONAL / NOT_QUALIFIED)")),
                    ("Режим", "расчёт в отдельном процессе"))
        rows = (*rows, ("Версия", f"KK-VIM {program_version()}"))
        for i, (k, v) in enumerate(rows):
            grid.addWidget(label(k, "fieldLabel"), i, 0, Qt.AlignmentFlag.AlignTop)
            grid.addWidget(label(v, wrap=True), i, 1)
        grid.setColumnStretch(1, 1)
        status.body.addLayout(grid)
        left.addWidget(status)
        left.addStretch(1)

        left_box = QWidget()
        left_box.setLayout(left)
        left_box.setFixedWidth(320)
        lay.addWidget(left_box)

        projects = placeholders.RECENT_PROJECTS if demo else ()
        recent = Card("Недавние проекты", right=label("демонстрационные записи" if demo else
                                                       "сохранение проектов .kkv — следующий этап", "tag"))
        self.table = QTableWidget(len(projects), 6)
        self.table.setHorizontalHeaderLabels(
            ["Имя", "Пластина", "Закрепление", "N", "Изменён", "Результат"]
        )
        for r, row in enumerate(projects):
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
