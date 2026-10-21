"""Task setup: all parameters on one page in compact blocks (GUI_DESIGN.md, 5.2).

The composition follows what the solver implements (``backend.kkvim_adapter.CAPABILITIES``):
Kirchhoff–Love, uniform load, SSSS; CCCC is shown as not implemented (no owner decision);
N = 1 (R3) and N = 2–3 (R4, D-047); N = 4–6 are shown as not implemented. The method block shows the
parameters of the selected solver: for N = 1 the stopping parameters of NUMERICAL_SPEC.md 11.2 with
the defaults of kkvim; for N = 2–3 the mode, the DST path and the number of half-steps (the R4
tolerances are fixed by D-048).

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
    QCheckBox,
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

from kkvim_gui.backend.kkvim_adapter import CAPABILITIES, DENSE_N_MAX, N_AVAILABLE, R4_FIXED
from kkvim_gui.backend.request import R4_MODES, TRANSFORMS
from kkvim_gui.main_window.plate_scheme import PlateScheme
from kkvim_gui.model.setup_model import SetupModel
from kkvim_gui.model.setup_state import FIELDS, N_VALUES, R4_MODE_TEXT, UNSUPPORTED, validate_field
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
    "CCCC": "Жёсткая заделка всех четырёх краёв — не реализовано",
}
CCCC_TOOLTIP = "CCCC не реализовано: расчёт не выполняется до решения владельца проекта"
THEORIES = (
    ("Кирхгоф–Ляв", True),
    ("Тимошенко — планируется", False),
    ("Редди — планируется", False),
)
STATUS_OK = (
    "✓  Формальная проверка ввода пройдена. Научные проверки ввода (диапазон b/a, ν, сетка) "
    "выполняет решатель KK-VIM перед запуском."
)
STATUS_OK_DEMO = "✓  Формальная проверка ввода пройдена. Демонстрационный режим: решатель не вызывается (TEST DATA)."
R3_NOTE = (
    "N = 1 — принятый решатель R3 (D-043). Параметры остановки — NUMERICAL_SPEC.md 11.2; по умолчанию — "
    "значения kkvim, с которыми выполнена приёмка. Строгий режим: tol_∞/100, max(tol_E/100, 10·ε_floor) (11.4)."
)
R4_NOTE = (
    "N = 2–3 — исследовательский решатель R4 (D-047, D-048). Все результаты — PROVISIONAL / NOT_QUALIFIED. "
    f"Допуски фиксированы D-048: tol_∞ = {R4_FIXED['tol_inf']:g}, tol_E = {R4_FIXED['tol_E']:g}, "
    f"s_min = {R4_FIXED['s_min']}, s_max = {R4_FIXED['s_max']}, η_max = {R4_FIXED['eta_max']:g}. "
    f"Путь DST: fft — основной; dense — только при n ≤ {DENSE_N_MAX} (D-048.6)."
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

    def __init__(self, model=None, session=None, demo=False, parent=None):
        super().__init__(parent)
        self.setObjectName("Page")
        set_help_context(self, "model.overview")
        self.demo = demo
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
        self.supports = SegmentedControl({"SSSS": "SSSS", "CCCC": "CCCC — не реализовано"}, st.supports)
        self.supports.buttons["CCCC"].setToolTip(CCCC_TOOLTIP)
        row = QHBoxLayout()
        row.addWidget(self.supports)
        row.addStretch(1)
        supports.body.addLayout(row)
        self.support_text = label(SUPPORT_TEXT[st.supports], "secondary", wrap=True)
        supports.body.addWidget(self.support_text)
        self.errors_supports = error_label()
        supports.body.addWidget(self.errors_supports)
        supports.body.addWidget(label("Другие закрепления — в последующих этапах проекта", "secondary", wrap=True))
        supports.body.addStretch(1)

        method = Card("Расчётная модель и метод КК-ВИМ")
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
            text = {1: "1 — R3, принятый", 2: "2 — R4, исследование", 3: "3 — R4, исследование"}.get(
                n, f"{n} — не реализовано")
            self.N.addItem(text, n)
        _disable_items(self.N, [n in N_AVAILABLE for n in N_VALUES])
        self.N.setCurrentIndex(N_VALUES.index(st.N))
        self.N.setToolTip("N — число членов разделённого представления (D-003). Реализовано: N = 1 (R3), "
                          "N = 2–3 (R4, D-047); N = 4–6 не реализовано")
        self.n_x = UnitField(t["n_x"])
        self.n_y = UnitField(t["n_y"])
        for w in (self.n_x, self.n_y):
            w.setToolTip(f"Число интервалов равномерной сетки (не меньше {CAPABILITIES['n_min']}); "
                         "при N = 2–3 — не меньше 2N")
        self.tol_inf = UnitField(t["tol_inf"])
        self.tol_E = UnitField(t["tol_E"])
        self.s_min = UnitField(t["s_min"])
        self.s_max = UnitField(t["s_max"])
        self.strict = QCheckBox("Строгий режим (11.4)")
        self.strict.setChecked(st.strict)
        self.strict.setToolTip("tol_∞/100 и max(tol_E/100, 10·ε_floor) — NUMERICAL_SPEC.md 11.2, 11.4")
        self.r4_mode = QComboBox()
        for key in R4_MODES:
            self.r4_mode.addItem(R4_MODE_TEXT[key], key)
        self.r4_mode.setCurrentIndex(R4_MODES.index(st.r4_mode))
        self.transform = QComboBox()
        for key in TRANSFORMS:
            self.transform.addItem({"fft": "fft — основной путь", "dense": f"dense — только n ≤ {DENSE_N_MAX}"}[key], key)
        self.transform.setCurrentIndex(TRANSFORMS.index(st.transform))
        for combo in (self.N, self.r4_mode, self.transform):
            # long item texts must not widen the method block beyond the page (no horizontal scroll)
            combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(16)
        self.r4_mode.setToolTip("\n".join(R4_MODE_TEXT.values()))
        self.half_steps = UnitField(t["half_steps"])
        self.half_steps.setToolTip(f"Предел полушагов; не больше s_max = {R4_FIXED['s_max']} (D-048.3). В режиме "
                                   "фиксированного числа — точное число полушагов")
        self.fields = {
            "a": self.a, "b": self.b, "h": self.h, "E": self.E, "nu": self.nu, "q": self.q,
            "n_x": self.n_x, "n_y": self.n_y, "tol_inf": self.tol_inf, "tol_E": self.tol_E,
            "s_min": self.s_min, "s_max": self.s_max, "half_steps": self.half_steps,
        }
        by_widget = {**g.errors, **m.errors, **lg.errors}
        self.errors = {key: by_widget[f] for key, f in self.fields.items() if f in by_widget}
        self._method_cells = []
        self._cells_by_scope = {"all": [], "r3": [], "r4": []}
        for text, w, topic, scope in (
            ("Теория", self.theory, "param.theory", "all"),
            ("Число членов N", self.N, None, "all"),
            ("Интервалов по x n_x", self.n_x, None, "all"),
            ("Интервалов по y n_y", self.n_y, None, "all"),
            ("Допуск tol_∞", self.tol_inf, None, "r3"),
            ("Допуск tol_E", self.tol_E, None, "r3"),
            ("Мин. полушагов s_min", self.s_min, None, "r3"),
            ("Макс. полушагов s_max", self.s_max, None, "r3"),
            ("Режим остановки", self.strict, None, "r3"),
            ("Режим R4", self.r4_mode, None, "r4"),
            ("Преобразование DST", self.transform, None, "r4"),
            ("Полушагов (предел)", self.half_steps, None, "r4"),
        ):
            cell = QWidget()
            box = QVBoxLayout(cell)
            box.setContentsMargins(0, 0, 0, 0)
            box.setSpacing(4)
            box.addWidget(label_with_info(text, topic) if topic else label(text, "fieldLabel"))
            box.addWidget(w)
            key = next((k for k, f in self.fields.items() if f is w), None)
            if key is not None:
                self.errors[key] = error_label()
                box.addWidget(self.errors[key])
            box.addStretch(1)
            self._method_cells.append(cell)
            self._cells_by_scope[scope].append(cell)
        method.body.addLayout(self.method_grid)
        self.r3_note = Note(R3_NOTE)
        self.r4_note = Note(R4_NOTE, "testdata")
        method.body.addWidget(self.r3_note)
        method.body.addWidget(self.r4_note)

        summary = Card("Сводка")
        self.summary_grid = QGridLayout()
        self.summary_grid.setHorizontalSpacing(16)
        self.summary_grid.setVerticalSpacing(6)
        self.summary_grid.setColumnStretch(1, 1)
        self.summary_grid.setColumnStretch(3, 1)
        self._summary_values = {}
        for i, key in enumerate(("Геометрия", "Материал", "Нагрузка", "Закрепление", "Метод", "Сетка")):
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
        self.run_button.setToolTip("Выполнить расчёт (F5). Демонстрационный режим: TEST DATA" if demo else
                                   "Выполнить расчёт решателем KK-VIM в отдельном процессе (F5)")
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
        self.strict.toggled.connect(lambda on: self.model.set_choice(strict=on))
        self.r4_mode.currentIndexChanged.connect(lambda i: self.model.set_choice(r4_mode=self.r4_mode.itemData(i)))
        self.transform.currentIndexChanged.connect(
            lambda i: self.model.set_choice(transform=self.transform.itemData(i)))
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
        if self.strict.isChecked() != st.strict:
            self.strict.setChecked(st.strict)
        if self.r4_mode.currentData() != st.r4_mode:
            self.r4_mode.setCurrentIndex(R4_MODES.index(st.r4_mode))
        if self.transform.currentData() != st.transform:
            self.transform.setCurrentIndex(TRANSFORMS.index(st.transform))
        for name, text in self.model.summary():
            self._summary_values[name].setText(text)
        scope = "r3" if st.N == 1 else "r4"
        if scope != getattr(self, "_scope", None):
            self._scope = scope
            self.r3_note.setVisible(scope == "r3")
            self.r4_note.setVisible(scope == "r4")
            for key in list(self.shown_errors):  # errors of fields that no longer take part in the run
                if key in self.fields and key not in st.active_fields():
                    self._show_error(key, None)
            self._layout_method_cells()
            self._update_status()
        if "supports" in self.shown_errors and st.supports == "SSSS":
            self._show_error("supports", None)
            self._update_status()

    def check_field(self, key):
        _, error = validate_field(key, self.model.state.texts[key])
        self._show_error(key, error)
        self._update_status()
        return error is None

    def check_all(self):
        errors = self.model.errors()
        for key in self.fields:
            self._show_error(key, errors.get(key))
        self._show_error("supports", errors.get("supports"))
        self._update_status()
        return errors

    def _show_error(self, key, error):
        if key == "supports":
            self.errors_supports.setText(error or "")
            self.errors_supports.setVisible(bool(error))
            if error:
                self.shown_errors[key] = error
            else:
                self.shown_errors.pop(key, None)
            return
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
        for w in (self.theory, self.N, self.load_uniform, self.strict, self.r4_mode, self.transform):
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
            key = next((k for k in self.fields if k in self.shown_errors), "supports")
            first = FIELDS[key].label if key in FIELDS else "Граничные условия"
            self.check_text.setText(f"⚠  Исправьте поля: {n} (первое — «{first}»). Расчёт не запускается.")
            self.check_text.setProperty("role", "fieldError")
        else:
            self.check_text.setText(STATUS_OK_DEMO if self.demo else STATUS_OK)
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
        if "supports" in self.shown_errors:
            button = self.supports.buttons["SSSS"]
            self.scroll.ensureWidgetVisible(button, 16, 48)
            button.setFocus(Qt.FocusReason.OtherFocusReason)
            return button
        return None

    def show_solver_problems(self, problems):
        """Input rejected by kkvim's own checks before a run: shown in the action bar, no run is started."""
        self.check_text.setText("⚠  Расчёт не запущен. " + " ".join(problems))
        self.check_text.setProperty("role", "fieldError")
        self.check_text.style().unpolish(self.check_text)
        self.check_text.style().polish(self.check_text)
        self.solver_problems = list(problems)

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
        self._layout_method_cells()

    def _layout_method_cells(self):
        """Method fields of the selected solver: three per row, or one per row in the narrow layout."""
        per_row = 1 if self.mode == "narrow" else 3
        scope = getattr(self, "_scope", "r3" if self.model.state.N == 1 else "r4")
        visible = self._cells_by_scope["all"] + self._cells_by_scope[scope]
        for cell in self._method_cells:
            self.method_grid.removeWidget(cell)
        for i, cell in enumerate(visible):
            r, col = divmod(i, per_row)
            self.method_grid.addWidget(cell, r, col)
        # visibility after the layout: a cell is shown only once it belongs to the card (never as a window)
        for cell in self._method_cells:
            cell.setVisible(cell in visible)
        for col in range(3):
            self.method_grid.setColumnStretch(col, 1 if col < per_row else 0)

    def unit_fields(self):
        return list(self.fields.values())
