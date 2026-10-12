"""Task setup: all parameters on one page in compact blocks (GUI_DESIGN.md, 5.2).

The available theory, supports and load reflect the approved first implementation
(D-003, D-005-D-007). From P3 the composition comes from the backend capability
registry. Method parameters other than N are marked preliminary: they are defined by
NUMERICAL_SPEC (Q-009, Q-010), not by the GUI.

All values live in one SetupModel shared with the main window, so they survive switching
between sections (GUI_DESIGN.md, 5.2). Numeric fields keep exactly the text the user typed;
it is parsed ("." or ",", sign, exponent) and formally checked when the value is committed
(Enter or leaving the field) and before a run (11.2). Errors appear under the field; a field
shown as invalid is re-checked while typing so the message disappears as soon as the value
is correct. Units are SI symbols and are never converted (11.1). This is a formal check of
the input only, not a check by the scientific solver.

Adaptive layout (GUI_DESIGN.md, 19): "wide" — parameter blocks on the left, plate scheme
on the right; "medium" — the scheme below the blocks; "narrow" — one column. The page
scrolls when needed; the action bar with "Выполнить расчёт" always stays visible.
"""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from kkvim_gui.main_window.plate_scheme import PlateScheme
from kkvim_gui.model.setup_model import SetupModel
from kkvim_gui.model.setup_state import FIELDS, N_VALUES, validate_field
from kkvim_gui.session import Session
from kkvim_gui.widgets.common import (
    Card,
    FieldGrid,
    Note,
    SegmentedControl,
    UnitField,
    error_label,
    label,
    repolish,
    label_with_info,
    set_help,
    set_help_context,
)

SUPPORT_TEXT = {
    "SSSS": "Шарнирное опирание всех четырёх краёв",
    "CCCC": "Жёсткая заделка всех четырёх краёв",
}
THEORIES = (
    ("Кирхгоф–Ляв", True),
    ("Тимошенко — планируется", False),
    ("Редди — планируется", False),
)
STATUS_OK = (
    "✓  Формальная проверка ввода пройдена. Научные ограничения не проверялись: "
    "решатель не подключён, расчёт демонстрационный (TEST DATA)."
)
LOCK_BANNER = (
    "Выполняется расчёт. Изменение параметров временно заблокировано. "
    "Просмотр и копирование доступны"
)
LOCKED_TEXT = (
    "Параметры активного расчёта заблокированы до его завершения или остановки. "
    "Просмотр и копирование доступны."
)
# Layout thresholds in logical pixels of the page width.
WIDE_MIN = 1100
MEDIUM_MIN = 760


def _disable_items(combo, enabled_flags):
    model = combo.model()
    for i, enabled in enumerate(enabled_flags):
        model.item(i).setEnabled(enabled)


def layout_mode(width):
    if width >= WIDE_MIN:
        return "wide"
    return "medium" if width >= MEDIUM_MIN else "narrow"


class SetupView(QWidget):
    runRequested = Signal()

    def __init__(self, model=None, session=None, parent=None):
        super().__init__(parent)
        self.setObjectName("Page")
        set_help_context(self, "model.overview")
        self.model = model if model is not None else SetupModel(parent=self)
        self.session = session if session is not None else Session(settings=None, parent=self)
        st = self.model.state
        t, u = st.texts, st.units

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        geometry = Card("Геометрия")
        g = FieldGrid()
        self.a = UnitField(t["a"], u["a"], FIELDS["a"].units)
        self.b = UnitField(t["b"], u["b"], FIELDS["b"].units)
        self.h = UnitField(t["h"], u["h"], FIELDS["h"].units)
        g.add_row("Длина a", self.a, topic="param.geometry.a")
        g.add_row("Ширина b", self.b, topic="param.geometry.b")
        g.add_row("Толщина h", self.h, topic="param.geometry.h")
        geometry.body.addLayout(g)
        geometry.body.addWidget(label("Прямоугольная пластина постоянной толщины", "secondary", wrap=True))

        material = Card("Материал")
        m = FieldGrid()
        self.E = UnitField(t["E"], u["E"], FIELDS["E"].units)
        self.nu = UnitField(t["nu"])
        m.add_row("Модуль упругости E", self.E, topic="param.material.E")
        m.add_row("Коэффициент Пуассона ν", self.nu, topic="param.material.nu")
        material.body.addLayout(m)
        material.body.addWidget(label("Однородный изотропный линейно-упругий", "secondary", wrap=True))
        material.body.addStretch(1)

        load = Card("Нагрузка")
        self.load_uniform = QRadioButton("Равномерно распределённая")
        self.load_uniform.setChecked(True)
        self.load_uniform.setToolTip("Единственный вид нагрузки первой реализации (D-005)")
        load.body.addWidget(self.load_uniform)
        lg = FieldGrid()
        self.q = UnitField(t["q"], u["q"], FIELDS["q"].units)
        lg.add_row("Интенсивность q", self.q)
        load.body.addLayout(lg)
        load.body.addStretch(1)

        supports = Card("Граничные условия")
        self.supports = SegmentedControl({"SSSS": "SSSS", "CCCC": "CCCC"}, st.supports)
        row = QHBoxLayout()
        row.addWidget(self.supports)
        row.addStretch(1)
        supports.body.addLayout(row)
        self.support_text = label(SUPPORT_TEXT[st.supports], "secondary", wrap=True)
        supports.body.addWidget(self.support_text)
        supports.body.addWidget(label("Другие закрепления — в последующих этапах проекта", "secondary", wrap=True))
        supports.body.addStretch(1)

        method = Card("Расчётная модель и метод КК-ВИМ", right=label("предварительно", "tag"))
        self.method_grid = QGridLayout()
        self.method_grid.setHorizontalSpacing(12)
        self.method_grid.setVerticalSpacing(8)
        self.theory = QComboBox()
        for text, _ in THEORIES:
            self.theory.addItem(text)
        _disable_items(self.theory, [enabled for _, enabled in THEORIES])
        set_help(self.theory, "param.theory")
        self.N = QComboBox()
        for n in N_VALUES:
            self.N.addItem(f"{n}" if n <= 3 else f"{n} (исследование)", n)
        self.N.setCurrentIndex(N_VALUES.index(st.N))
        self.N.setToolTip("N — число членов разделённого представления; 1–3 — основной диапазон, 4–6 — исследование (D-003)")
        self.k_max = UnitField(t["k_max"])
        self.k_max.setToolTip("k — число итераций (D-003); предварительно")
        self.grid_n = UnitField(t["grid"], placeholder="по NUMERICAL_SPEC (Q-010)")
        self.stop_rule = UnitField(t["stop_rule"], placeholder="по NUMERICAL_SPEC (Q-009)")
        self.fields = {
            "a": self.a, "b": self.b, "h": self.h, "E": self.E, "nu": self.nu, "q": self.q,
            "k_max": self.k_max, "grid": self.grid_n, "stop_rule": self.stop_rule,
        }
        by_widget = {**g.errors, **m.errors, **lg.errors}
        self.errors = {key: by_widget[f] for key, f in self.fields.items() if f in by_widget}
        self._method_cells = []
        for text, w, topic in (
            ("Теория", self.theory, "param.theory"),
            ("Число членов N", self.N, None),
            ("Макс. итераций k_max", self.k_max, None),
            ("Сетка МКР", self.grid_n, None),
            ("Критерий остановки", self.stop_rule, None),
        ):
            cell = QWidget()
            box = QVBoxLayout(cell)
            box.setContentsMargins(0, 0, 0, 0)
            box.setSpacing(4)
            box.addWidget(label_with_info(text, topic) if topic else label(text, "fieldLabel"))
            box.addWidget(w)
            if w is self.k_max:
                self.errors["k_max"] = error_label()
                box.addWidget(self.errors["k_max"])
            box.addStretch(1)
            self._method_cells.append(cell)
        method.body.addLayout(self.method_grid)
        method.body.addWidget(Note(
            "Параметры метода, кроме N, предварительные: их состав и смысл определяет "
            "NUMERICAL_SPEC (Q-009, Q-010). Решатель не подключён — расчёт демонстрационный."
        ))

        summary = Card("Сводка")
        self.summary_grid = QGridLayout()
        self.summary_grid.setHorizontalSpacing(16)
        self.summary_grid.setVerticalSpacing(6)
        self.summary_grid.setColumnStretch(1, 1)
        self.summary_grid.setColumnStretch(3, 1)
        self._summary_values = {}
        for i, key in enumerate(("Геометрия", "Материал", "Нагрузка", "Закрепление", "Метод")):
            r, pair = divmod(i, 2)
            self.summary_grid.addWidget(label(key, "fieldLabel"), r, 2 * pair)
            value = label("", wrap=True)
            self.summary_grid.addWidget(value, r, 2 * pair + 1)
            self._summary_values[key] = value
        summary.body.addLayout(self.summary_grid)
        summary.body.addStretch(1)

        self.scheme = Card("Схема модели", right=label("схема условная", "tag"))
        # The scheme shows the current input (this model), never the run snapshot.
        self.plate = PlateScheme(self.session, self.model)
        self.scheme.body.addWidget(self.plate, 1)

        self.cards = {
            "geometry": geometry, "material": material, "load": load,
            "supports": supports, "method": method, "summary": summary, "scheme": self.scheme,
        }
        content = QWidget()
        self.grid = QGridLayout(content)
        self.grid.setContentsMargins(16, 16, 16, 16)
        self.grid.setSpacing(12)
        self.scroll = scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(content)
        # Lock banner above the cards: shown only while a run is active (no pop-up).
        self.lock_banner = Note(LOCK_BANNER, "lock")
        self.lock_banner.setVisible(False)
        banner_box = QWidget()
        banner_lay = QVBoxLayout(banner_box)
        banner_lay.setContentsMargins(16, 12, 16, 0)
        banner_lay.addWidget(self.lock_banner)
        self.banner_box = banner_box
        banner_box.setVisible(False)
        outer.addWidget(banner_box)
        outer.addWidget(scroll, 1)
        self.mode = None
        self._apply_mode("wide")

        # Bottom action bar: outside the scroll area, always visible.
        bar = QFrame()
        bar.setObjectName("ActionBar")
        bl = QHBoxLayout(bar)
        bl.setContentsMargins(16, 8, 16, 8)
        self.check_text = label(STATUS_OK, "secondary", wrap=True)
        bl.addWidget(self.check_text, 1)
        self.goto_error = QToolButton()
        self.goto_error.setText("Перейти к полю")
        self.goto_error.setVisible(False)
        self.goto_error.clicked.connect(self.focus_first_error)
        bl.addWidget(self.goto_error)
        self.save_button = QPushButton("Сохранить проект")
        self.save_button.setEnabled(False)
        self.save_button.setToolTip("Сохранение .kkv — следующий этап")
        self.run_button = QPushButton("▶  Выполнить расчёт")
        self.run_button.setProperty("variant", "primary")
        self.run_button.setToolTip("Выполнить расчёт (F5). Демонстрационный режим")
        self.run_button.clicked.connect(self.try_run)
        bl.addWidget(self.save_button)
        bl.addWidget(self.run_button)
        outer.addWidget(bar)

        self.shown_errors = {}
        self.locked = False
        for key, f in self.fields.items():
            f.edit.textChanged.connect(lambda text, k=key: self._on_text(k, text))
            f.edit.editingFinished.connect(lambda k=key: self.check_field(k))
            if f.unit_combo is not None:
                f.unit_combo.currentTextChanged.connect(lambda unit, k=key: self.model.set_unit(k, unit))
        self.supports.changed.connect(lambda key: self.model.set_choice(supports=key))
        self.N.currentIndexChanged.connect(lambda i: self.model.set_choice(N=self.N.itemData(i)))
        self.model.changed.connect(self.refresh)
        self.refresh()

    # Model binding and validation
    def _on_text(self, key, text):
        self.model.set_text(key, text)
        if key in self.shown_errors:  # re-check only fields already shown as invalid
            self.check_field(key)

    def refresh(self):
        """Show the model: the summary, and widget values that differ from it.

        Equal values are not written back, so the undo history of the fields is kept.
        """
        st = self.model.state
        for key, f in self.fields.items():
            if f.text() != st.texts[key]:
                f.edit.setText(st.texts[key])
            if f.unit_combo is not None and f.unit() != st.units[key]:
                f.unit_combo.setCurrentText(st.units[key])
        if self.supports.current() != st.supports:
            self.supports.set_current(st.supports)
        self.support_text.setText(SUPPORT_TEXT[st.supports])
        if self.N.currentData() != st.N:
            self.N.setCurrentIndex(N_VALUES.index(st.N))
        for name, text in self.model.summary():
            self._summary_values[name].setText(text)

    def check_field(self, key):
        _, error = validate_field(key, self.model.state.texts[key])
        self._show_error(key, error)
        self._update_status()
        return error is None

    def check_all(self):
        errors = self.model.errors()
        for key in self.fields:
            self._show_error(key, errors.get(key))
        self._update_status()
        return errors

    def _show_error(self, key, error):
        field = self.fields[key]
        err = self.errors.get(key)
        field.set_invalid(bool(error), error or "")
        if err is not None:
            err.setText(error or "")
            err.setVisible(bool(error))
        if error:
            self.shown_errors[key] = error
        else:
            self.shown_errors.pop(key, None)

    def set_locked(self, locked):
        """Read-only view of the parameters of the active run (GUI_DESIGN.md, 5.3).

        Text stays selectable and copyable; nothing can be changed until the run
        finishes or is stopped.
        """
        self.locked = locked
        self.lock_banner.setVisible(locked)
        self.banner_box.setVisible(locked)
        for f in self.fields.values():
            f.edit.setReadOnly(locked)
            f.setProperty("locked", locked)
            repolish(f)
            if f.unit_combo is not None:
                f.unit_combo.setEnabled(not locked)
        for b in self.supports.buttons.values():
            b.setEnabled(not locked)
        for w in (self.theory, self.N, self.load_uniform):
            w.setEnabled(not locked)
        self._update_status()

    def _update_status(self):
        n = len(self.shown_errors)
        if self.locked:
            self.check_text.setText(LOCKED_TEXT)
            self.check_text.setProperty("role", "secondary")
            self.check_text.style().unpolish(self.check_text)
            self.check_text.style().polish(self.check_text)
            self.goto_error.setVisible(False)
            self.run_button.setEnabled(False)
            return
        if n:
            first = FIELDS[next(k for k in self.fields if k in self.shown_errors)].label
            self.check_text.setText(f"⚠  Исправьте поля: {n} (первое — «{first}»). Расчёт не запускается.")
            self.check_text.setProperty("role", "fieldError")
        else:
            self.check_text.setText(STATUS_OK)
            self.check_text.setProperty("role", "secondary")
        self.check_text.style().unpolish(self.check_text)
        self.check_text.style().polish(self.check_text)
        self.goto_error.setVisible(bool(n))
        self.run_button.setEnabled(not n)

    def focus_first_error(self):
        for key, f in self.fields.items():
            if key in self.shown_errors:
                self.scroll.ensureWidgetVisible(f, 16, 48)
                f.edit.setFocus(Qt.FocusReason.OtherFocusReason)
                return f
        return None

    def try_run(self):
        """Check everything; run only if the input is formally correct and no run is active."""
        if self.locked:
            return False
        if self.check_all():
            self.focus_first_error()
            return False
        self.runRequested.emit()
        return True

    # Adaptive layout
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._apply_mode(layout_mode(self.width()))

    def _apply_mode(self, mode):
        if mode == self.mode:
            return
        self.mode = mode
        grid = self.grid
        for w in self.cards.values():
            grid.removeWidget(w)
        for c in range(3):
            grid.setColumnStretch(c, 0)
        for r in range(8):
            grid.setRowStretch(r, 0)
        c = self.cards
        if mode == "narrow":
            order = ("geometry", "material", "load", "supports", "method", "summary", "scheme")
            for r, key in enumerate(order):
                grid.addWidget(c[key], r, 0)
            grid.setColumnStretch(0, 1)
            c["scheme"].setMinimumHeight(360)
        else:
            grid.addWidget(c["geometry"], 0, 0)
            grid.addWidget(c["material"], 0, 1)
            grid.addWidget(c["load"], 1, 0)
            grid.addWidget(c["supports"], 1, 1)
            grid.addWidget(c["method"], 2, 0, 1, 2)
            grid.addWidget(c["summary"], 3, 0, 1, 2)
            grid.setColumnStretch(0, 3)
            grid.setColumnStretch(1, 3)
            if mode == "wide":
                grid.addWidget(c["scheme"], 0, 2, 4, 1)
                grid.setColumnStretch(2, 4)
                grid.setRowStretch(3, 1)
                c["scheme"].setMinimumHeight(0)
            else:
                grid.addWidget(c["scheme"], 4, 0, 1, 2)
                c["scheme"].setMinimumHeight(360)
        # Method fields: three per row, or one per row in the narrow layout.
        per_row = 1 if mode == "narrow" else 3
        for cell in self._method_cells:
            self.method_grid.removeWidget(cell)
        for i, cell in enumerate(self._method_cells):
            r, col = divmod(i, per_row)
            self.method_grid.addWidget(cell, r, col)
        for col in range(3):
            self.method_grid.setColumnStretch(col, 1 if col < per_row else 0)

    def unit_fields(self):
        return list(self.fields.values())
