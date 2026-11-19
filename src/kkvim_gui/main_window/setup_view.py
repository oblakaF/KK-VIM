"""Task setup: all parameters on one page in compact blocks (GUI_DESIGN.md, 5.2).

The composition follows what the solvers implement (``backend.kkvim_adapter.CAPABILITIES``, ``request.route_of``):
Kirchhoff–Love, uniform load; SSSS — N = 1 (R3), N = 2–3 (R4, D-047), N = 4–6 research only (D-056.7);
CCCC — research only, N = 1 (D-061), N = 2–3 (D-063, D-065), N = 4–6 (D-067, D-068). The method block shows only
the parameters of the selected route: for SSSS N = 1 the stopping parameters of NUMERICAL_SPEC.md 11.2 with the
defaults of kkvim; for SSSS N = 2–3 the mode, the DST path and the number of half-steps (the R4 tolerances are
fixed by D-048); for SSSS N = 4–6 the explicit research switch and the number of half-steps; for CCCC the number
of half-steps (N = 1: 1…8; N ≥ 2: 1 or 2, s = 2 under the solver's RN-5 and RT-7). Every research choice is off
by default and is reset when N or the supports change.

User-facing layout (GUI-PLATE-WORKFLOW-UX-01): plain labels («SSSS — шарнирное опирание», «Число членов N», «Разбиение по X
(nₓ)»); the list of N shows only 1…6 and the scientific status of the chosen route is a separate line; the stopping
parameters of R3 and the R4 mode, DST path and half-step limit are in the collapsed «Дополнительные параметры расчёта»
(the approved defaults are kept without opening it; a field with an error opens it); the decision numbers, guards and
limits of the route are in the collapsed «Технические подробности маршрута». No guard or check is removed: the same
model, the same validation and the same solver checks run. Load: «Одна нагрузка» or «Серия нагрузок» (several q, one
run each); the loads of a series are shown before the run.

«Независимый расчёт для сравнения» (in the supports card): for CCCC the switch «МКР — двумерный (D2)» (on by default) —
D2 of the same plate and grid in its own process after KK-VIM; for SSSS the line that the Navier reference is computed
automatically and that D2 is not applied to SSSS.

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
from PySide6.QtGui import QFont, QFontMetrics
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLineEdit,
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
from kkvim_gui.model.setup_state import (DISPLAY_ALL, DISPLAY_KEYS, DISPLAY_TITLES, FIELDS, N_VALUES, R4_MODE_TEXT,
                                         RESEARCH_RULE_TEXT, SERIES_MAX, SERIES_MIN, cccc_half_steps_of, display_options,
                                         generate_series, parse_series, scope_of, series_preview, series_texts,
                                         validate_field)
from kkvim_gui.session import Session
from kkvim_gui.widgets.common import (
    Card,
    FieldGrid,
    Note,
    SegmentedControl,
    StatusLabel,
    UnitField,
    error_label,
    label,
    repolish,
    label_with_info,
    set_help,
    set_help_context,
)

SUPPORT_TEXT = {
    "SSSS": "Все четыре края шарнирно оперты: прогиб на краях равен нулю, края могут поворачиваться.",
    "CCCC": "Все четыре края жёстко заделаны: на краях равны нулю прогиб и угол поворота.",
}
SUPPORT_LABELS = {"SSSS": "SSSS — шарнирное опирание", "CCCC": "CCCC — жёсткая заделка"}
CCCC_WARNING = "Исследовательский режим — научная квалификация не завершена."
ROUTE_STATUS = {  # the scientific status of the chosen route, shown next to N (never in the list of N)
    "r3": ("success", "SSSS, N = 1: принятый решатель R3 в утверждённой области (0,2 ≤ b/a ≤ 5)"),
    "r4": ("warning", "SSSS, N = 2–3: PROVISIONAL / NOT_QUALIFIED — квалификация ограничена; это не ошибка решателя"),
    "r46": ("warning", "SSSS, N = 4–6: RESEARCH_ONLY / PROVISIONAL / NOT_QUALIFIED"),
    "cccc": ("warning", "CCCC, N = 1–6: RESEARCH_ONLY / PROVISIONAL / NOT_QUALIFIED"),
}
ADVANCED_TITLE = "Дополнительные параметры расчёта"
ADVANCED_NOTE = ("По умолчанию — утверждённые значения решателя; для обычного расчёта их менять не нужно. Наведите указатель "
                 "на название параметра или нажмите ⓘ — объяснение откроется в справке.")
DETAILS_TITLE = "Технические подробности маршрута"
LOAD_MODES_TEXT = {"single": "Одна нагрузка", "series": "Серия нагрузок"}
SERIES_HINT = ("Список значений q через «;» или пробел (можно неравномерный); десятичный разделитель — точка или запятая. "
               "От 2 до 20 значений. Равномерную серию проще задать кнопкой «Задать серию…».")
SERIES_BUTTON = "Задать серию: начало, конец, шаг…"
GRID_NOTE = ("nₓ и nᵧ — число интервалов сетки, а не узлов: узлов (nₓ + 1) × (nᵧ + 1){nodes}. Допустимые сетки маршрута "
             "проверяет решатель KK-VIM по решениям проекта; форма своих ограничений не вводит.")
FEM_NOTE = "МКЭ в программе не рассчитывается: в сравнении показываются только сохранённые значения прежних проверок."
DIAGNOSTIC_NOTE = "Диагностические значения (запрошенный расчёт не выполнен) не являются принятым научным результатом."
CCCC_TOOLTIP = ("CCCC — исследовательские режимы: N = 1 (D-061), N = 2–3 (D-063, D-065), N = 4–6 (D-067, D-068). "
                "Результат никогда не квалифицирован")
CCCC_PREVIEW_TEXT = "CCCC — недоступно в этой версии"
D2_TEXT = "МКР — двумерный (D2): рассчитать для тех же входных данных"
D2_TOOLTIP = ("Независимая реализация той же дискретной задачи CCCC (13-точечная схема, D-049.1, D-049.8) в отдельном "
              "процессе после КК-ВИМ, для тех же a, b, h, E, ν, q и той же сетки; в серии — при каждом q. Совпадение с D2 "
              "не проверяет сеточную погрешность и не повышает научный статус КК-ВИМ")
INDEPENDENT_TITLE = "Независимый расчёт для сравнения"
SSSS_INDEPENDENT = ("SSSS: эталон Навье рассчитывается автоматически для тех же входных данных. МКР D2 — схема "
                    "жёсткой заделки; для SSSS не применяется.")
N_TEXT = {n: str(n) for n in N_VALUES}  # the list shows N only; the status of the route is a separate line
THEORIES = (
    ("Кирхгоф–Ляв", True),
    ("Тимошенко — планируется", False),
    ("Редди — планируется", False),
)
STATUS_OK = "✓  Ввод проверен. Перед запуском решатель KK-VIM проверит отношение сторон, ν и сетку."
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
N_LABEL = "Число членов N"
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


RESULTS_TITLE = "Результаты и сравнение"
RESULTS_NOTE = ("Какие кривые показать в окне анализа после расчёта. Это только показ: флажки не запускают и не отменяют "
                "расчёты (КК-ВИМ, Навье, D2), не меняют статус и не включают исследовательский режим.")
RESULTS_AFTER = ("После расчёта во вкладках «Циклы VIM-1–3» и «Нагрузка — прогиб» видно, что фактически рассчитано и "
                 "сохранено, а чего нет — с причиной; видимость кривых там можно менять без пересчёта.")
DISPLAY_ALL_TEXT = "Все доступные кривые"
POSSIBLE_PREFIX, UNAVAILABLE_PREFIX = "○ возможно после расчёта — ", "✕ недоступно — "


def _disable_items(combo, enabled_flags):
    model = combo.model()
    for i, enabled in enumerate(enabled_flags):
        model.item(i).setEnabled(enabled)


def layout_mode(width):
    if width >= WIDE_MIN:
        return "wide"
    return "medium" if width >= MEDIUM_MIN else "narrow"


def fit_selected_text(control):
    """Segment buttons are drawn in bold when selected: never narrower than the bold text (no clipped labels)."""
    for button in control.buttons.values():
        bold = QFont(button.font())
        bold.setBold(True)
        chrome = max(16, button.sizeHint().width() - button.fontMetrics().horizontalAdvance(button.text()))
        button.setMinimumWidth(QFontMetrics(bold).horizontalAdvance(button.text()) + chrome)


class SeriesDialog(QDialog):
    """«Серия нагрузок»: start, end and increment in the unit of q; the generated loads are previewed before applying.

    The values come from ``setup_state.generate_series`` (decimal arithmetic, the end reached exactly, 2…20 values) and
    are written as the ordinary text of "q_series", so they pass the same validation and the same solver checks as a
    typed list. Nothing is computed here."""

    def __init__(self, model, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Серия нагрузок")
        self.model = model
        st = model.state
        texts = series_texts(st.texts.get("q_series", ""))
        valid = parse_series(st.texts.get("q_series", ""))[1] is None
        lay = QVBoxLayout(self)
        lay.addWidget(label("Равномерная серия: q = начало, начало + шаг, …, конец. Каждое значение — отдельный расчёт "
                            "тем же решателем с теми же остальными параметрами.", "secondary", wrap=True))
        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        self.start = QLineEdit(texts[0] if valid else "")
        self.end = QLineEdit(texts[-1] if valid else "")
        self.step = QLineEdit(self._step_of(texts) if valid else "")
        self.unit = QComboBox()
        self.unit.addItems(list(FIELDS["q"].units))
        self.unit.setCurrentText(st.units["q"])
        self.unit.setToolTip("Единица q для всех значений серии; смена единицы не пересчитывает числа")
        for r, (text, w) in enumerate((("Начальная нагрузка", self.start), ("Конечная нагрузка", self.end),
                                       ("Шаг нагрузки", self.step), ("Единица q", self.unit))):
            grid.addWidget(label(text, "fieldLabel"), r, 0)
            grid.addWidget(w, r, 1)
        lay.addLayout(grid)
        self.preview = label("", wrap=True)
        self.preview.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        lay.addWidget(self.preview)
        self.error = error_label()
        lay.addWidget(self.error)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Применить")
        self.buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("Отмена")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        lay.addWidget(self.buttons)
        for w in (self.start, self.end, self.step):
            w.textChanged.connect(self._update)
        self.unit.currentTextChanged.connect(self._update)
        self.setMinimumWidth(460)
        self._update()

    @staticmethod
    def _step_of(texts):
        """The increment of a uniform list as typed ("" if the list is not uniform)."""
        from decimal import Decimal

        values = [Decimal(t.replace("−", "-").replace(",", ".")) for t in texts]
        steps = {b - a for a, b in zip(values, values[1:])}
        if len(steps) != 1:
            return ""
        text = format(steps.pop().normalize(), "f")
        return text.replace(".", ",") if any("," in t for t in texts) else text

    def generated(self):
        """(texts, error) of the current entries."""
        return generate_series(self.start.text(), self.end.text(), self.step.text())

    def _update(self, *_):
        texts, error = self.generated()
        if error is None:
            _, error = validate_field("q_series", "; ".join(texts))  # the same check as a typed list
        ok = error is None
        self.preview.setText(f"Будет выполнено расчётов: {len(texts)} — q = " + "; ".join(texts) + f" {self.unit.currentText()}"
                             if ok else f"Серия не задана: значения q появятся здесь (от {SERIES_MIN} до {SERIES_MAX}).")
        self.error.setText(error or "")
        self.error.setVisible(not ok)
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(ok)

    def accept(self):
        texts, error = self.generated()
        if error is not None:
            return
        self.model.set_unit("q", self.unit.currentText())
        self.model.set_text("q_series", "; ".join(texts))
        super().accept()


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
        self.load_uniform.setToolTip("Нагрузка q одинакова во всех точках пластины и направлена по прогибу")
        load.body.addWidget(self.load_uniform)
        self.load_mode = SegmentedControl(LOAD_MODES_TEXT, st.load_mode)
        set_help(self.load_mode, "param.load.series")
        mode_row = QHBoxLayout()
        mode_row.addWidget(self.load_mode)
        mode_row.addStretch(1)
        load.body.addLayout(mode_row)
        lg = FieldGrid()
        self.q = UnitField(t["q"], u["q"], FIELDS["q"].units)
        self.q_label = lg.add_row("Интенсивность q", self.q, topic="param.load.q")
        self.q_series = UnitField(t["q_series"])
        self.q_series_label = lg.add_row("Значения q серии", self.q_series, topic="param.load.series")
        self.q_series_unit = label(u["q"], "secondary")
        load.body.addLayout(lg)
        self.series_button = QPushButton(SERIES_BUTTON)
        self.series_button.setToolTip("Начальная и конечная нагрузка, шаг и единица q; перед применением показываются все "
                                      "значения и число расчётов")
        self.series_button.clicked.connect(self.open_series_dialog)
        series_row = QHBoxLayout()
        series_row.addWidget(self.series_button)
        series_row.addStretch(1)
        load.body.addLayout(series_row)
        self.series_hint = label(SERIES_HINT, "secondary", wrap=True)
        load.body.addWidget(self.series_hint)
        self.series_preview = label("", wrap=True)
        self.series_preview.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        load.body.addWidget(self.series_preview)
        load.body.addStretch(1)

        supports = Card("Граничные условия")
        self.supports = SegmentedControl({"SSSS": SUPPORT_LABELS["SSSS"],
                                          "CCCC": CCCC_PREVIEW_TEXT if self.preview else SUPPORT_LABELS["CCCC"]}, st.supports)
        set_help(self.supports.buttons["SSSS"], "bc.overview")
        set_help(self.supports.buttons["CCCC"], "bc.overview")
        self.supports.buttons["CCCC"].setEnabled(not self.preview)
        row = QHBoxLayout()
        row.addWidget(self.supports)
        row.addStretch(1)
        supports.body.addLayout(row)
        self.support_text = label(SUPPORT_TEXT[st.supports], "secondary", wrap=True)
        supports.body.addWidget(self.support_text)
        self.cccc_warning = Note(CCCC_WARNING, "testdata")
        supports.body.addWidget(self.cccc_warning)
        supports.body.addWidget(label_with_info(INDEPENDENT_TITLE, "verification.d2"))
        self.d2 = QCheckBox(D2_TEXT)
        self.d2.setChecked(st.with_d2)
        self.d2.setToolTip(D2_TOOLTIP)
        set_help(self.d2, "verification.d2")
        supports.body.addWidget(self.d2)
        self.ssss_independent = label(SSSS_INDEPENDENT, "secondary", wrap=True)
        supports.body.addWidget(self.ssss_independent)
        self.fem_note = label(FEM_NOTE, "secondary", wrap=True)
        self.fem_note.setVisible(not self.preview)
        supports.body.addWidget(self.fem_note)
        self.errors_supports = error_label()
        supports.body.addWidget(self.errors_supports)
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
        if self.preview:
            self.N.setToolTip(PREVIEW_N_TOOLTIP)
        else:
            set_help(self.N, "param.N")
        self.n_x = UnitField(t["n_x"])
        self.n_y = UnitField(t["n_y"])
        set_help(self.n_x, "param.grid.nx")
        set_help(self.n_y, "param.grid.ny")
        self.tol_inf = UnitField(t["tol_inf"])
        self.tol_E = UnitField(t["tol_E"])
        self.s_min = UnitField(t["s_min"])
        self.s_max = UnitField(t["s_max"])
        for w, topic in ((self.tol_inf, "param.stop.tol_inf"), (self.tol_E, "param.stop.tol_E"),
                         (self.s_min, "param.stop.s_min"), (self.s_max, "param.stop.s_max")):
            set_help(w, topic)
        self.strict = QCheckBox("Строгий режим")
        self.strict.setChecked(st.strict)
        set_help(self.strict, "param.stop.strict")
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
            "q_series": self.q_series, "n_x": self.n_x, "n_y": self.n_y, "tol_inf": self.tol_inf, "tol_E": self.tol_E,
            "s_min": self.s_min, "s_max": self.s_max, "half_steps": self.half_steps,
        }
        by_widget = {**g.errors, **m.errors, **lg.errors}
        self.errors = {key: by_widget[f] for key, f in self.fields.items() if f in by_widget}
        self._method_cells = []
        self._cell_scopes = {}
        self._advanced_cells = set()
        half_steps_cells = []
        for text, w, topic, scopes, advanced in (
            ("Теория", self.theory, "param.theory", ("all",), False),
            (N_LABEL, self.N, None if self.preview else "param.N", ("all",), False),
            ("Разбиение по X (nₓ)", self.n_x, "param.grid.nx", ("all",), False),
            ("Разбиение по Y (nᵧ)", self.n_y, "param.grid.ny", ("all",), False),
            ("Режим N = 4–6", self.ssss_research, None, ("r46",), False),
            ("Полушагов (ровно)", self.half_steps, "param.halfsteps", ("r46",), False),
            (CCCC_STEPS_TEXT, self.cccc_half_steps, "param.halfsteps", ("cccc",), False),
            ("Допуск tol_∞", self.tol_inf, "param.stop.tol_inf", ("r3",), True),
            ("Допуск tol_E", self.tol_E, "param.stop.tol_E", ("r3",), True),
            ("Мин. полушагов s_min", self.s_min, "param.stop.s_min", ("r3",), True),
            ("Макс. полушагов s_max", self.s_max, "param.stop.s_max", ("r3",), True),
            ("Режим остановки", self.strict, "param.stop.strict", ("r3",), True),
            ("Режим R4", self.research, None, ("r4",), True),
            ("Остановка исследования", self.research_rule, None, ("r4",), True),
            (TRANSFORM_LABEL, self.transform, None, ("r4",), True),
            ("Полушагов (предел)", self.half_steps, "param.halfsteps", ("r4",), True),
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
            if w is self.N:
                self.route_status = StatusLabel(*ROUTE_STATUS[st.scope])
                self.route_status.setWordWrap(True)
                box.addWidget(self.route_status)
            box.addStretch(1)
            if w is self.half_steps:  # the same field: research N = 4-6 (main block), R4 (advanced)
                half_steps_cells.append(cell)
            self._method_cells.append(cell)
            self._cell_scopes[cell] = scopes
            if advanced:
                self._advanced_cells.add(cell)
        self._half_steps_cells = tuple(half_steps_cells)  # (research N = 4-6 cell, R4 cell)
        method.body.addLayout(self.method_grid)
        self.grid_note = label("", "secondary", wrap=True)
        method.body.addWidget(self.grid_note)
        self.diagnostic_note = label(DIAGNOSTIC_NOTE, "secondary", wrap=True)
        method.body.addWidget(self.diagnostic_note)
        self.advanced_toggle = self._toggle(ADVANCED_TITLE)
        self.advanced_box = QWidget()
        advanced_lay = QVBoxLayout(self.advanced_box)
        advanced_lay.setContentsMargins(0, 0, 0, 0)
        advanced_lay.addWidget(label(ADVANCED_NOTE, "secondary", wrap=True))
        self.advanced_grid = QGridLayout()
        self.advanced_grid.setHorizontalSpacing(12)
        self.advanced_grid.setVerticalSpacing(8)
        advanced_lay.addLayout(self.advanced_grid)
        self.advanced_box.setVisible(False)
        self.advanced_toggle.toggled.connect(self._show_advanced)
        method.body.addWidget(self.advanced_toggle)
        method.body.addWidget(self.advanced_box)
        self.preview_note = Note(PREVIEW_NOTE) if self.preview else None
        if self.preview_note is not None:
            method.body.addWidget(self.preview_note)
        self.details_toggle = self._toggle(DETAILS_TITLE)
        self.details_box = QWidget()
        details_lay = QVBoxLayout(self.details_box)
        details_lay.setContentsMargins(0, 0, 0, 0)
        self.r3_note = Note(R3_NOTE)
        self.r4_note = Note(R4_NOTE, "testdata")
        self.r46_note = Note(R46_NOTE, "testdata")
        self.cccc_note = Note(CCCC_NOTE, "testdata")
        self.scope_notes = {"r3": self.r3_note, "r4": self.r4_note, "r46": self.r46_note, "cccc": self.cccc_note}
        for note in self.scope_notes.values():
            details_lay.addWidget(note)
        self.details_box.setVisible(False)
        self.details_toggle.toggled.connect(lambda on: self._expand(self.details_toggle, self.details_box, on))
        method.body.addWidget(self.details_toggle)
        method.body.addWidget(self.details_box)

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

        results = Card(RESULTS_TITLE, right=label("только показ", "tag"))
        results.body.addWidget(label(RESULTS_NOTE, "secondary", wrap=True))
        self.display_all = QCheckBox(DISPLAY_ALL_TEXT)
        self.display_all.setToolTip("Отметить все кривые, которые выбранный маршрут может дать")
        self.display_all.clicked.connect(self._on_display_all)
        results.body.addWidget(self.display_all)
        display_grid = QGridLayout()
        display_grid.setHorizontalSpacing(12)
        display_grid.setVerticalSpacing(4)
        display_grid.setColumnStretch(1, 1)
        self.display_boxes, self.display_status = {}, {}
        for r, key in enumerate(DISPLAY_KEYS):
            box = QCheckBox(DISPLAY_TITLES[key])
            box.toggled.connect(lambda on, k=key: self._on_display(k, on))
            status = label("", "secondary", wrap=True)
            display_grid.addWidget(box, r, 0, Qt.AlignmentFlag.AlignTop)
            display_grid.addWidget(status, r, 1)
            self.display_boxes[key], self.display_status[key] = box, status
        results.body.addLayout(display_grid)
        results.body.addWidget(label(RESULTS_AFTER, "secondary", wrap=True))

        self.scheme = Card("Схема модели", right=label("схема условная", "tag"))
        # The scheme shows the current input (this model), never the run snapshot.
        self.plate = PlateScheme(self.session, self.model)
        self.scheme.body.addWidget(self.plate, 1)

        self.cards = {
            "geometry": geometry, "material": material, "load": load,
            "supports": supports, "method": method, "results": results, "summary": summary, "scheme": self.scheme,
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
        self.run_button = QPushButton("▶  Выполнить расчёт")
        self.run_button.setProperty("variant", "primary")
        self.run_button.setToolTip("Выполнить расчёт (F5). Демонстрационный режим: TEST DATA" if demo else
                                   "Выполнить расчёт решателем KK-VIM (F5)")
        self.run_button.clicked.connect(self.try_run)
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
        self.load_mode.changed.connect(lambda key: self.model.set_choice(load_mode=key))
        self.N.currentIndexChanged.connect(lambda i: self.model.set_choice(N=self.N.itemData(i)))
        self.strict.toggled.connect(lambda on: self.model.set_choice(strict=on))
        self.research.toggled.connect(self._on_research)
        self.ssss_research.toggled.connect(lambda on: self.model.set_choice(ssss_research=on))
        self.d2.toggled.connect(lambda on: self.model.set_choice(with_d2=on))
        self.cccc_half_steps.currentIndexChanged.connect(self._on_cccc_steps)
        self.research_rule.currentIndexChanged.connect(self._on_research_rule)
        self.transform.currentIndexChanged.connect(
            lambda i: self.model.set_choice(transform=self.transform.itemData(i)))
        self.model.changed.connect(self.refresh)
        self.refresh()

    @staticmethod
    def _toggle(text):
        button = QToolButton()
        button.setCheckable(True)
        button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        button.setArrowType(Qt.ArrowType.RightArrow)
        button.setText(text)
        return button

    @staticmethod
    def _expand(button, box, on):
        button.setArrowType(Qt.ArrowType.DownArrow if on else Qt.ArrowType.RightArrow)
        box.setVisible(on)

    def _show_advanced(self, on):
        self._expand(self.advanced_toggle, self.advanced_box, on)

    def open_advanced(self):
        """Open «Дополнительные параметры расчёта» (a field with an error there must be visible)."""
        if not self.advanced_toggle.isChecked():
            self.advanced_toggle.setChecked(True)

    ADVANCED_KEYS = ("tol_inf", "tol_E", "s_min", "s_max", "transform")

    def _advanced_error(self):
        scope = self.model.state.scope
        keys = [k for k in self.ADVANCED_KEYS if k in self.shown_errors]
        return bool(keys) or ("half_steps" in self.shown_errors and scope == "r4")

    # Model binding and validation
    def _on_research(self, on):
        """The research continuation is chosen only by this explicit switch (D-048.5)."""
        self.model.set_choice(r4_mode=self.research_rule.currentData() if on else R4_NORMATIVE)

    def _n_text(self, supports, n):
        if self.preview and n not in self.n_selectable:
            return f"{n} — недоступно в этой версии"
        return N_TEXT[n]

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
        if self.load_mode.current() != st.load_mode:
            self.load_mode.set_current(st.load_mode)
        self.support_text.setText(SUPPORT_TEXT[st.supports])
        self.cccc_warning.setVisible(st.supports == "CCCC")
        self.d2.setVisible(st.supports == "CCCC" and not self.preview)
        self.ssss_independent.setVisible(st.supports == "SSSS" and not self.preview)
        if self.d2.isChecked() != st.with_d2:
            self.d2.blockSignals(True)
            self.d2.setChecked(st.with_d2)
            self.d2.blockSignals(False)
        self.route_status.set_status(*ROUTE_STATUS[st.scope])
        self._show_load(st)
        self._show_display(st)
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
        nx, ny = validate_field("n_x", st.texts["n_x"]), validate_field("n_y", st.texts["n_y"])
        nodes = (f" = {int(nx[0]) + 1} × {int(ny[0]) + 1}" if nx[1] is None and ny[1] is None
                 and float(nx[0]).is_integer() and float(ny[0]).is_integer() else "")
        self.grid_note.setText(GRID_NOTE.format(nodes=nodes))
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
        if self._advanced_error():
            self.open_advanced()

    def _show_display(self, st):
        """«Результаты и сравнение»: the choice of the model and, for every curve, whether the route can give it."""
        options = display_options(st)
        possible = {o.key for o in options if o.possible}
        for o in options:
            box = self.display_boxes[o.key]
            box.blockSignals(True)
            box.setChecked(o.key in st.display and o.possible)  # a curve the route cannot give is never shown as chosen
            box.blockSignals(False)
            box.setEnabled(o.possible and not self.locked)
            box.setToolTip(o.text)
            self.display_status[o.key].setText((POSSIBLE_PREFIX if o.possible else UNAVAILABLE_PREFIX) + o.text)
        chosen = possible & st.display
        state = (Qt.CheckState.Checked if chosen == possible else
                 Qt.CheckState.Unchecked if not chosen else Qt.CheckState.PartiallyChecked)
        self.display_all.blockSignals(True)
        self.display_all.setCheckState(state)
        self.display_all.blockSignals(False)
        self.display_all.setEnabled(not self.locked)

    def _on_display(self, key, on):
        display = self.model.state.display
        self.model.set_choice(display=display | {key} if on else display - {key})

    def _on_display_all(self, *_):
        """All curves the route can give; a second click (all already chosen) clears the choice."""
        st = self.model.state
        possible = {o.key for o in display_options(st) if o.possible}
        everything = possible <= st.display
        self.model.set_choice(display=st.display - possible if everything else DISPLAY_ALL)
        self._show_display(self.model.state)  # the model may be unchanged: the box state is restored from it

    def _show_load(self, st):
        series = st.series
        for w in (self.q, self.q_label):
            w.setVisible(not series)
        for w in (self.q_series, self.q_series_label, self.series_hint, self.series_preview, self.series_button):
            w.setVisible(series)
        if "q" in self.shown_errors and series:
            self._show_error("q", None)
        if "q_series" in self.shown_errors and not series:
            self._show_error("q_series", None)
        loads = series_preview(st) if series else None
        if series:
            self.series_preview.setText(
                f"Будет выполнено расчётов: {len(loads)} — q = " + "; ".join(f"{t} {u}" for t, u in loads)
                if loads else "Значения q серии ещё не заданы правильно.")
        self.run_button.setText("▶  Рассчитать серию" if series else "▶  Выполнить расчёт")

    def check_field(self, key):
        _, error = validate_field(key, self.model.state.texts[key])
        self._show_error(key, error)
        self._update_status()
        if error and self._advanced_error():
            self.open_advanced()
        return error is None

    def check_all(self):
        errors = self.model.errors()
        for key in self.fields:
            self._show_error(key, errors.get(key))
        self._show_error("supports", errors.get("supports"))
        self._show_error("transform", errors.get("transform"))
        self._show_error("research", errors.get("research"))
        self._update_status()
        if self._advanced_error():
            self.open_advanced()
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
                  self.cccc_half_steps, self.load_mode, self.d2, self.series_button):
            w.setEnabled(not locked)
        self.research_rule.setEnabled(not locked and self.model.state.research)
        self._show_display(self.model.state)
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
                if key in self.ADVANCED_KEYS or (key == "half_steps" and self.model.state.scope == "r4"):
                    self.open_advanced()
                self.scroll.ensureWidgetVisible(f, 16, 48)
                f.edit.setFocus(Qt.FocusReason.OtherFocusReason)
                return f
        if "supports" in self.shown_errors:
            button = self.supports.buttons["SSSS"]
            self.scroll.ensureWidgetVisible(button, 16, 48)
            button.setFocus(Qt.FocusReason.OtherFocusReason)
            return button
        if "transform" in self.shown_errors:
            self.open_advanced()
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

    def open_series_dialog(self):
        """«Задать серию…»: generate start…end by step; applied only with «Применить» (``SeriesDialog``)."""
        if self.locked:
            return None
        dialog = SeriesDialog(self.model, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.check_field("q_series")
        return dialog

    # Adaptive layout
    def showEvent(self, event):
        super().showEvent(event)
        for control in (self.load_mode, self.supports):  # after the style sheet: the real button padding is known
            fit_selected_text(control)

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
            order = ("geometry", "material", "load", "supports", "method", "results", "summary", "scheme")
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
            grid.addWidget(c["results"], 3, 0, 1, 2)
            grid.addWidget(c["summary"], 4, 0, 1, 2)
            grid.setColumnStretch(0, 3)
            grid.setColumnStretch(1, 3)
            if mode == "wide":
                # top-aligned at the height for its width: the scheme neither stretches nor moves when the method card
                # grows («Дополнительные параметры расчёта», «Технические подробности маршрута»)
                grid.addWidget(c["scheme"], 0, 2, 5, 1, Qt.AlignmentFlag.AlignTop)
                grid.setColumnStretch(2, 4)
                grid.setRowStretch(5, 1)
                c["scheme"].setMinimumHeight(0)
            else:
                grid.addWidget(c["scheme"], 5, 0, 1, 2)
                c["scheme"].setMinimumHeight(360)
                grid.setRowStretch(6, 1)
        self._layout_method_cells()

    def _layout_method_cells(self):
        """Method fields of the selected solver: three per row, or one per row in the narrow layout; the stopping and
        mode parameters in «Дополнительные параметры расчёта»."""
        per_row = 1 if self.mode == "narrow" else 3
        scope = getattr(self, "_scope", None) or self.model.state.scope
        visible = [cell for cell in self._method_cells if "all" in self._cell_scopes[cell] or scope in self._cell_scopes[cell]]
        main = [cell for cell in visible if cell not in self._advanced_cells]
        advanced = [cell for cell in visible if cell in self._advanced_cells]
        for cell in self._method_cells:
            self.method_grid.removeWidget(cell)
            self.advanced_grid.removeWidget(cell)
        # the field "half_steps" belongs to one cell at a time: the research cell (N = 4-6) or the R4 cell
        main_cell, r4_cell = self._half_steps_cells
        holder = r4_cell if scope == "r4" else main_cell
        holder.layout().insertWidget(1, self.half_steps)
        holder.layout().insertWidget(2, self.errors["half_steps"])
        for grid, cells in ((self.method_grid, main), (self.advanced_grid, advanced)):
            for i, cell in enumerate(cells):
                r, col = divmod(i, per_row)
                grid.addWidget(cell, r, col)
            for col in range(3):
                grid.setColumnStretch(col, 1 if col < per_row else 0)
        # visibility after the layout: a cell is shown only once it belongs to the card (never as a window)
        for cell in self._method_cells:
            cell.setVisible(cell in visible)
        has_advanced = bool(advanced)
        self.advanced_toggle.setVisible(has_advanced)
        self.advanced_box.setVisible(has_advanced and self.advanced_toggle.isChecked())

    def unit_fields(self):
        return list(self.fields.values())
