"""Task setup: all parameters on one page in compact blocks (GUI_DESIGN.md, 5.2).

The composition follows what the solvers implement (``backend.kkvim_adapter.CAPABILITIES``, ``request.route_of``):
Kirchhoff–Love, uniform load; SSSS — N = 1 (R3), N = 2–3 (R4, D-047), N = 4–6 research only (D-056.7);
CCCC — research only, N = 1 (D-061), N = 2–3 (D-063, D-065), N = 4–6 (D-067, D-068). The method block shows only
the parameters of the selected route: for SSSS N = 1 the stopping parameters of NUMERICAL_SPEC.md 11.2 with the
defaults of kkvim; for SSSS N = 2–3 the mode, the DST path and the number of half-steps (the R4 tolerances are
fixed by D-048); for SSSS N = 4–6 the explicit research switch and the number of half-steps; for CCCC the number
of half-steps (N = 1: 1…8; N ≥ 2: 1 or 2, s = 2 under the solver's RN-5 and RT-7). Every research choice is off
by default and is reset when N or the supports change.

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

from kkvim_gui.backend.kkvim_adapter import CAPABILITIES, DENSE_N_MAX, N_AVAILABLE, R4_FIXED, RESEARCH_MARK
from kkvim_gui.backend.request import R4_NORMATIVE, R4_RESEARCH_MODES, R4_RESEARCH_STOP, TRANSFORMS
from kkvim_gui.main_window.plate_scheme import PlateScheme
from kkvim_gui.model.setup_model import SetupModel
from kkvim_gui.model.setup_state import (FIELDS, N_VALUES, R4_MODE_TEXT, RESEARCH_RULE_TEXT, cccc_half_steps_of, scope_of,
                                         validate_field)
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
    "CCCC": "Жёсткая заделка всех четырёх краёв — только исследование (RESEARCH_ONLY / PROVISIONAL / NOT_QUALIFIED)",
}
CCCC_TOOLTIP = ("CCCC — исследовательские режимы: N = 1 (D-061), N = 2–3 (D-063, D-065), N = 4–6 (D-067, D-068). "
                "Результат никогда не квалифицирован")
CCCC_PREVIEW_TEXT = "CCCC — недоступно в этой версии"
N_TEXT = {
    "SSSS": {1: "1 — R3, принятый", 2: "2 — R4, NOT_QUALIFIED", 3: "3 — R4, NOT_QUALIFIED",
             4: "4 — исследование", 5: "5 — исследование", 6: "6 — исследование"},
    "CCCC": {1: "1 — исследование, s ≤ 8", 2: "2 — исследование, s ≤ 2", 3: "3 — исследование, s ≤ 2",
             4: "4 — исследование, s ≤ 2", 5: "5 — исследование, s ≤ 2", 6: "6 — исследование, s ≤ 2"},
}
N_TOOLTIP = ("N — число членов разделённого представления (D-003). SSSS: N = 1 — R3 (ACCEPTED), N = 2–3 — R4 "
             "(PROVISIONAL / NOT_QUALIFIED), N = 4–6 — только исследование (D-056.7). CCCC: N = 1–6 — только исследование "
             "(D-061, D-063, D-065, D-067, D-068). N никогда не уменьшается")
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
    f"Путь DST: fft — основной; dense — только при max(n_x, n_y) ≤ {DENSE_N_MAX} (D-048.6). "
    "По умолчанию — нормативный режим D-047 (research_mode=False): после s = 1 — NOT_QUALIFIED. "
    "Исследовательское продолжение через QR включается только флажком (D-048.5) и сбрасывается при смене N."
)
PREVIEW_NOTE = (
    "Функциональная предварительная версия: доступна только модель Кирхгофа–Лява, SSSS, равномерная нагрузка, "
    "N = 1 — принятый решатель R3 (D-043). N = 2–6, CCCC, теории Тимошенко и Редди показаны как недоступные."
)
PREVIEW_N_TOOLTIP = ("N — число членов разделённого представления (D-003). В этой версии доступно только N = 1 "
                     "(R3, ACCEPTED); N = 2–6 и CCCC — в полной версии, все они не квалифицированы")
R46_NOTE = (
    f"SSSS, N = 4–6 — только исследование (D-056.7, D-060): {RESEARCH_MARK}. Расчёт выполняется лишь при включённом "
    "исследовательском режиме и только в исследованной области: десять контрольных b/a, сетки n_x = n_y из 40…1280 "
    f"или шесть сеток семейства II, не больше {CAPABILITIES['ssss_research_s_max']} полушагов; вне неё — отказ без расчёта. "
    "Энергии в истории — в единицах q²a⁶/D ячейки исследования.")
CCCC_NOTE = (
    f"CCCC — только исследование: {RESEARCH_MARK}. N = 1 — фиксированные полушаги s = 1…8 (D-061); N = 2–3 и 4–6 — "
    f"полушаг s = 1, полушаг s = 2 только при n_x, n_y ≤ {CAPABILITIES['cccc_s2_grid_max']} после сертификатов RN-5 и RT-7 "
    "этого расчёта (D-065, D-068). Проверки выполняет решатель: при отказе сертификата — защитная остановка, поле s = 1 "
    "сохраняется только как диагностика. s ≥ 3 не исполняется. По умолчанию — s = 1.")
SSSS_RESEARCH_TEXT = "Исследовательский режим N = 4–6 (RESEARCH_ONLY, D-056.7)"
CCCC_STEPS_TEXT = "Полушагов (ровно)"
RESEARCH_TEXT = "Исследовательское продолжение через QR (D-048.5)"
RESEARCH_TOOLTIP = (
    "Выключено — нормативный режим D-047: research_mode=False, после s = 1 — NOT_QUALIFIED.\n"
    "Включено — research_mode=True: политика остановки D-048.3; результат PROVISIONAL / NOT_QUALIFIED; "
    "PROVISIONAL_FIELD_STABILIZED — стабилизация поля конечной задачи, не доказанная сходимость (D-048.4).\n"
    "Выбор сбрасывается при смене N."
)
TRANSFORM_LABEL = "Преобразование DST"
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

    def __init__(self, model=None, session=None, demo=False, n_selectable=None, parent=None):
        """n_selectable: values of N the user may choose (the functional preview passes PREVIEW_N = (1,));
        None keeps every N that the adapter implements."""
        super().__init__(parent)
        self.setObjectName("Page")
        set_help_context(self, "model.overview")
        self.demo = demo
        self.n_selectable = tuple(N_AVAILABLE if n_selectable is None else n_selectable)
        self.preview = n_selectable is not None
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
        self.supports = SegmentedControl({"SSSS": "SSSS", "CCCC": CCCC_PREVIEW_TEXT if self.preview else "CCCC — исследование"},
                                         st.supports)
        self.supports.buttons["CCCC"].setToolTip(CCCC_TOOLTIP)
        self.supports.buttons["CCCC"].setEnabled(not self.preview)
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
            self.N.addItem(self._n_text(st.supports, n), n)
        _disable_items(self.N, [n in self.n_selectable for n in N_VALUES])
        self.N.setCurrentIndex(N_VALUES.index(st.N))
        self._n_supports = st.supports
        self.N.setToolTip(PREVIEW_N_TOOLTIP if self.preview else N_TOOLTIP)
        self.n_x = UnitField(t["n_x"])
        self.n_y = UnitField(t["n_y"])
        for w in (self.n_x, self.n_y):
            w.setToolTip(f"Число интервалов равномерной сетки (не меньше {CAPABILITIES['n_min']}); "
                         "при N ≥ 2 — не меньше 2N (CCCC N ≥ 4: 2N + 1); CCCC: 7…1280")
        self.tol_inf = UnitField(t["tol_inf"])
        self.tol_E = UnitField(t["tol_E"])
        self.s_min = UnitField(t["s_min"])
        self.s_max = UnitField(t["s_max"])
        self.strict = QCheckBox("Строгий режим (11.4)")
        self.strict.setChecked(st.strict)
        self.strict.setToolTip("tol_∞/100 и max(tol_E/100, 10·ε_floor) — NUMERICAL_SPEC.md 11.2, 11.4")
        self.research = QCheckBox(RESEARCH_TEXT)
        self.research.setChecked(st.research)
        self.research.setToolTip(RESEARCH_TOOLTIP)
        self.research_rule = QComboBox()
        for key in R4_RESEARCH_MODES:
            self.research_rule.addItem(RESEARCH_RULE_TEXT[key], key)
        self.research_rule.setCurrentIndex(R4_RESEARCH_MODES.index(st.r4_mode) if st.research else 0)
        self.research_rule.setEnabled(st.research)
        self.research_rule.setToolTip("Правило остановки исследовательского продолжения; доступно только при "
                                      "включённом исследовательском режиме")
        self.r4_mode_text = label(R4_MODE_TEXT[st.r4_mode], "secondary")
        self.r4_mode_text.setWordWrap(True)
        self.transform = QComboBox()
        for key in TRANSFORMS:
            self.transform.addItem({"fft": "fft — основной путь", "dense": f"dense — max(n_x, n_y) ≤ {DENSE_N_MAX}"}[key], key)
        self.transform.setCurrentIndex(TRANSFORMS.index(st.transform))
        for combo in (self.N, self.research_rule, self.transform):
            # long item texts must not widen the method block beyond the page (no horizontal scroll)
            combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(16)
        self.ssss_research = QCheckBox(SSSS_RESEARCH_TEXT)
        self.ssss_research.setChecked(st.ssss_research)
        self.ssss_research.setToolTip("Выключено — расчёт SSSS при N = 4–6 не выполняется. Включено — исследовательский путь "
                                      "kkvim.research_ssss (RESEARCH_ONLY / PROVISIONAL / NOT_QUALIFIED). Сбрасывается при смене N")
        self.cccc_half_steps = QComboBox()
        self._fill_cccc_steps(st.N if st.supports == "CCCC" else 1, st.cccc_half_steps)
        self.cccc_half_steps.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.cccc_half_steps.setMinimumContentsLength(16)
        self.cccc_half_steps.setToolTip("Точное число полушагов CCCC: N = 1 — 1…8; N = 2–6 — 1 или 2 (s = 2 только после RN-5 "
                                        "и RT-7 этого расчёта, на сетках до 32 × 32). По умолчанию 1; сбрасывается при смене N")
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
        self._cell_scopes = {}
        for text, w, topic, scopes in (
            ("Теория", self.theory, "param.theory", ("all",)),
            ("Число членов N", self.N, None, ("all",)),
            ("Интервалов по x n_x", self.n_x, None, ("all",)),
            ("Интервалов по y n_y", self.n_y, None, ("all",)),
            ("Допуск tol_∞", self.tol_inf, None, ("r3",)),
            ("Допуск tol_E", self.tol_E, None, ("r3",)),
            ("Мин. полушагов s_min", self.s_min, None, ("r3",)),
            ("Макс. полушагов s_max", self.s_max, None, ("r3",)),
            ("Режим остановки", self.strict, None, ("r3",)),
            ("Режим R4", self.research, None, ("r4",)),
            ("Остановка исследования", self.research_rule, None, ("r4",)),
            (TRANSFORM_LABEL, self.transform, None, ("r4",)),
            ("Режим N = 4–6", self.ssss_research, None, ("r46",)),
            ("Полушагов (предел)", self.half_steps, None, ("r4", "r46")),
            (CCCC_STEPS_TEXT, self.cccc_half_steps, None, ("cccc",)),
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
            if w is self.research:
                box.addWidget(self.r4_mode_text)
            if w is self.transform:
                self.errors_transform = error_label()
                box.addWidget(self.errors_transform)
            if w is self.ssss_research:
                self.errors_research = error_label()
                box.addWidget(self.errors_research)
            box.addStretch(1)
            self._method_cells.append(cell)
            self._cell_scopes[cell] = scopes
        method.body.addLayout(self.method_grid)
        self.preview_note = Note(PREVIEW_NOTE) if self.preview else None
        if self.preview_note is not None:
            method.body.addWidget(self.preview_note)
        self.r3_note = Note(R3_NOTE)
        self.r4_note = Note(R4_NOTE, "testdata")
        self.r46_note = Note(R46_NOTE, "testdata")
        self.cccc_note = Note(CCCC_NOTE, "testdata")
        self.scope_notes = {"r3": self.r3_note, "r4": self.r4_note, "r46": self.r46_note, "cccc": self.cccc_note}
        for note in self.scope_notes.values():
            method.body.addWidget(note)

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
        self.research.toggled.connect(self._on_research)
        self.ssss_research.toggled.connect(lambda on: self.model.set_choice(ssss_research=on))
        self.cccc_half_steps.currentIndexChanged.connect(self._on_cccc_steps)
        self.research_rule.currentIndexChanged.connect(self._on_research_rule)
        self.transform.currentIndexChanged.connect(
            lambda i: self.model.set_choice(transform=self.transform.itemData(i)))
        self.model.changed.connect(self.refresh)
        self.refresh()

    # Model binding and validation
    def _on_research(self, on):
        """The research continuation is chosen only by this explicit switch (D-048.5)."""
        self.model.set_choice(r4_mode=self.research_rule.currentData() if on else R4_NORMATIVE)

    def _n_text(self, supports, n):
        if self.preview and n not in self.n_selectable:
            return f"{n} — недоступно в этой версии"
        return N_TEXT[supports if supports in N_TEXT else "SSSS"][n]

    def _fill_cccc_steps(self, N, current):
        self.cccc_half_steps.blockSignals(True)
        self.cccc_half_steps.clear()
        for value in cccc_half_steps_of(N):
            self.cccc_half_steps.addItem(f"s = {value}" + (" (по умолчанию)" if value == 1 else ""), value)
        self.cccc_half_steps.setCurrentIndex(max(0, self.cccc_half_steps.findData(current)))
        self.cccc_half_steps.blockSignals(False)
        self._cccc_steps_N = N

    def _on_cccc_steps(self, index):
        value = self.cccc_half_steps.itemData(index)
        if value is not None and self.model.state.supports == "CCCC":
            self.model.set_choice(cccc_half_steps=value)

    def _on_research_rule(self, index):
        if self.research.isChecked():
            self.model.set_choice(r4_mode=self.research_rule.itemData(index))

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
        if self._n_supports != st.supports:  # the N items name the route of the selected supports
            self._n_supports = st.supports
            for i, n in enumerate(N_VALUES):
                self.N.setItemText(i, self._n_text(st.supports, n))
        if self.ssss_research.isChecked() != st.ssss_research:
            self.ssss_research.blockSignals(True)
            self.ssss_research.setChecked(st.ssss_research)
            self.ssss_research.blockSignals(False)
        steps_N = st.N if st.supports == "CCCC" else 1
        if steps_N != self._cccc_steps_N or self.cccc_half_steps.currentData() != st.cccc_half_steps:
            self._fill_cccc_steps(steps_N, st.cccc_half_steps)
        if self.N.currentData() != st.N:
            self.N.setCurrentIndex(N_VALUES.index(st.N))
        if self.strict.isChecked() != st.strict:
            self.strict.setChecked(st.strict)
        # the model is the source: showing it must not choose a mode back (no implicit research run)
        rule = st.r4_mode if st.research else R4_RESEARCH_STOP  # the choice is reset with the mode
        for w in (self.research, self.research_rule):
            w.blockSignals(True)
        self.research.setChecked(st.research)
        self.research_rule.setCurrentIndex(R4_RESEARCH_MODES.index(rule))
        for w in (self.research, self.research_rule):
            w.blockSignals(False)
        self.research_rule.setEnabled(st.research and not self.locked)
        self.r4_mode_text.setText(R4_MODE_TEXT[st.r4_mode])
        self.r4_mode_text.setProperty("role", "figureMark" if st.research else "secondary")
        repolish(self.r4_mode_text)
        if self.transform.currentData() != st.transform:
            self.transform.setCurrentIndex(TRANSFORMS.index(st.transform))
        for name, text in self.model.summary():
            self._summary_values[name].setText(text)
        scope = scope_of(st.N, st.supports)
        if scope != getattr(self, "_scope", None):
            self._scope = scope
            for key, note in self.scope_notes.items():
                note.setVisible(key == scope)
            for key in list(self.shown_errors):  # errors of fields that no longer take part in the run
                if key in self.fields and key not in st.active_fields():
                    self._show_error(key, None)
            self._layout_method_cells()
            self._update_status()
        if "supports" in self.shown_errors and not self.model.errors().get("supports"):
            self._show_error("supports", None)
            self._update_status()
        if "research" in self.shown_errors and not self.model.errors().get("research"):
            self._show_error("research", None)
            self._update_status()
        transform_error = self.model.errors().get("transform")  # a choice: shown at once (D-048.6)
        if transform_error != self.shown_errors.get("transform"):
            self._show_error("transform", transform_error)
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
        self._show_error("transform", errors.get("transform"))
        self._show_error("research", errors.get("research"))
        self._update_status()
        return errors

    def _show_error(self, key, error):
        if key in ("supports", "transform", "research"):
            text = {"supports": self.errors_supports, "transform": self.errors_transform,
                    "research": self.errors_research}[key]
            text.setText(error or "")
            text.setVisible(bool(error))
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
        for key, b in self.supports.buttons.items():
            b.setEnabled(not locked and not (self.preview and key == "CCCC"))
        for w in (self.theory, self.N, self.load_uniform, self.strict, self.research, self.transform, self.ssss_research,
                  self.cccc_half_steps):
            w.setEnabled(not locked)
        self.research_rule.setEnabled(not locked and self.model.state.research)
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
            key = next(k for k in (*self.fields, "supports", "transform", "research") if k in self.shown_errors)
            first = FIELDS[key].label if key in FIELDS else {"supports": "Граничные условия",
                                                             "transform": TRANSFORM_LABEL, "research": "Режим N = 4–6"}[key]
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
        if "transform" in self.shown_errors:
            self.scroll.ensureWidgetVisible(self.transform, 16, 48)
            self.transform.setFocus(Qt.FocusReason.OtherFocusReason)
            return self.transform
        if "research" in self.shown_errors:
            self.scroll.ensureWidgetVisible(self.ssss_research, 16, 48)
            self.ssss_research.setFocus(Qt.FocusReason.OtherFocusReason)
            return self.ssss_research
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
        scope = getattr(self, "_scope", None) or self.model.state.scope
        visible = [cell for cell in self._method_cells if "all" in self._cell_scopes[cell] or scope in self._cell_scopes[cell]]
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
