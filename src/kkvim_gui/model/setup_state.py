"""Setup state of the main window and its formal validation (GUI_DESIGN.md, 5.2, 11). Qt-free.

The GUI checks only the form: a value is present, is a number and is finite; a, b, h and E are
greater than zero; counts are whole numbers not less than 1; tolerances are greater than zero. The
scientific checks (ranges of ν and b/a, grid sizes, configuration rules) are those of kkvim itself:
``backend.kkvim_adapter.check_request`` calls the solver's input validation before a run.

Values keep the text and the SI unit exactly as entered (GUI_DESIGN.md, 11.1). ``to_request``
converts them to the base units m and Pa for the solver (``backend.units``; GUI_DESIGN.md, 3.3).

Method parameters (from kkvim, the single source of their defaults):
- N = 1 — R3 (``run_rectangle_physical``): tol_∞, tol_E, s_min, s_max and the strict mode of
  NUMERICAL_SPEC.md 11.2 and 11.4;
- N = 2–3 — R4 (``run_coupled``): the mode, the DST path (fft; dense only while max(n_x, n_y) <= 40,
  D-048.6) and the number of half-steps. The tolerances of R4 are fixed by D-048 and are not editable.
  The mode is normative D-047 (``research_mode=False``) unless the research continuation through a QR is
  chosen explicitly (D-048.5): with stopping on stabilization or with a fixed number of half-steps.
  A change of N resets the choice to normative, so that a research run is never started implicitly.
- SSSS, N = 4–6 — research only (``kkvim.research_ssss``, D-056.7): the run needs the explicit research
  choice ``ssss_research`` and the exact number of half-steps (the field "half_steps");
- CCCC, N = 1–6 — research only: the number of half-steps is a choice (N = 1: s = 1…8; N = 2–6: s = 1 or 2;
  s = 2 runs under the solver's RN-5 and RT-7, s >= 3 is not offered).
A change of N or of the supports resets every research choice (normative R4, no N = 4–6 research, s = 1 for CCCC).

Load: one uniform load q, or a load series (``load_mode == "series"``): the field "q_series" holds several values of q
in the unit chosen for q, separated by ";" or spaces ("," is the decimal comma). ``to_series_requests`` gives one
RunRequest per value: the request of the same form with only the text of q replaced, so geometry, material, supports,
N, grid and every method parameter are exactly those of a single run. ``generate_series`` (GUI-OWNER-UX-01) writes such
a list from a start, an end and an increment: the values start + i·step are computed in decimal arithmetic (no
accumulated binary rounding), the end must be reached exactly (it is never moved), and the number of values must be
within SERIES_MIN…SERIES_MAX; the result is the ordinary text of "q_series", checked again by the same path.

Independent method for comparison (GUI-MULTIMETHOD-CCCC-D2-01): ``with_d2`` — for CCCC, also compute «МКР — двумерный
(D2)» (the QA reference D2 of the same discrete problem, ``backend.d2_job``) for the same plate and grid, in its own
process after the KK-VIM run (for a series: at every q). On by default; it applies to CCCC only: SSSS keeps the Navier
reference and D2 (the clamped-plate scheme) is never offered for SSSS. It does not enter the RunRequest of the solver.

Results and comparison (GUI-OWNER-COMPARISON-SELECT-01): ``display`` — the curves shown after the run (KK-VIM last
half-step, VIM-1/2/3 = the completed full cycles k = 1, 2, 3, s = 2k of the same run (D-030), Navier, D2). It is a
DISPLAY choice only: it never enters the RunRequest or the configuration, never starts or stops a computation, never
changes a research choice and is not reset with N. ``display_options`` states before the run which curves the chosen
route can give and why the others cannot (from the route's recorded history; nothing is promised by a checkbox).
"""

from dataclasses import dataclass, field, replace
from decimal import Decimal, InvalidOperation, localcontext

from kkvim_gui import placeholders
from kkvim_gui.backend import units
from kkvim_gui.backend.kkvim_adapter import DEFAULTS, N_AVAILABLE, dense_problem
from kkvim_gui.backend.request import (CCCC_HALF_STEPS, CCCC_ROUTES, R4_MODES, R4_NORMATIVE, R4_RESEARCH_FIXED,
                                      R4_RESEARCH_MODES, R4_RESEARCH_STOP, ROUTE_CCCC_N1, ROUTE_R3, ROUTE_R4, ROUTE_SSSS_N46,
                                      SUPPORTS, TRANSFORMS, RunRequest,
                                      route_of)
from kkvim_gui.model.numbers import parse_number

POSITIVE = "Значение должно быть больше нуля."
COUNT = "Нужно целое число не меньше 1."
SERIES_MIN, SERIES_MAX = 2, 20
SERIES_SEPARATORS = (";", "\n", "\t", " ")
RESEARCH_REQUIRED = ("SSSS, N = 4–6 — только исследование (RESEARCH_ONLY, D-056.7): включите исследовательский режим "
                     "N = 4–6, иначе расчёт не выполняется.")


@dataclass(frozen=True)
class FieldSpec:
    key: str
    label: str
    rule: str  # "positive" | "number" | "count"
    units: tuple = ()
    scope: tuple = ("all",)  # "all" or the scopes of scope_of: "r3", "r4", "r46", "cccc"


FIELDS = {
    spec.key: spec
    for spec in (
        FieldSpec("a", "Длина a", "positive", ("m", "cm", "mm")),
        FieldSpec("b", "Ширина b", "positive", ("m", "cm", "mm")),
        FieldSpec("h", "Толщина h", "positive", ("m", "cm", "mm")),
        FieldSpec("E", "Модуль упругости E", "positive", ("GPa", "MPa", "Pa")),
        FieldSpec("nu", "Коэффициент Пуассона ν", "number"),
        FieldSpec("q", "Интенсивность q", "number", ("kPa", "Pa", "MPa")),
        FieldSpec("q_series", "Значения q серии", "series"),
        FieldSpec("n_x", "Разбиение по X (nₓ)", "count"),
        FieldSpec("n_y", "Разбиение по Y (nᵧ)", "count"),
        FieldSpec("tol_inf", "Допуск tol_∞", "positive", scope=("r3",)),
        FieldSpec("tol_E", "Допуск tol_E", "positive", scope=("r3",)),
        FieldSpec("s_min", "Мин. полушагов s_min", "count", scope=("r3",)),
        FieldSpec("s_max", "Макс. полушагов s_max", "count", scope=("r3",)),
        FieldSpec("half_steps", "Полушагов (предел)", "count", scope=("r4", "r46")),
    )
}
THEORY = "Кирхгоф–Ляв"
LOAD_MODES = ("single", "series")
N_VALUES = (1, 2, 3, 4, 5, 6)  # every N has a route (request.route_of); each route has its own domain and status
R4_MODE_TEXT = {
    R4_NORMATIVE: "Нормативный D-047: только s = 1, далее NOT_QUALIFIED",
    R4_RESEARCH_STOP: "Исследовательский D-048: остановка по стабилизации поля",
    R4_RESEARCH_FIXED: "Исследовательский: фиксированное число полушагов (дополнение к D-048)",
}
DISPLAY_KEYS = ("kkvim", 1, 2, 3, "navier", "d2")  # the series / cycles tab curve keys (k = cycle)
DISPLAY_ALL = frozenset(DISPLAY_KEYS)
DISPLAY_TITLES = {
    "kkvim": "КК-ВИМ: последний полушаг",
    1: "ВИМ-1 — k=1, s=2",
    2: "ВИМ-2 — k=2, s=4",
    3: "ВИМ-3 — k=3, s=6",
    "navier": "Навье — эталон для SSSS",
    "d2": "МКР — двумерный (D2) — сравнение для CCCC",
}
RESEARCH_RULE_TEXT = {  # stopping of the research continuation; shown only while it is chosen
    R4_RESEARCH_STOP: "Остановка по стабилизации поля (D-048.3)",
    R4_RESEARCH_FIXED: "Фиксированное число полушагов (дополнение к D-048)",
}


def _number_text(value):
    return repr(float(value)) if isinstance(value, float) and value < 1e-3 else str(value)


def _default_texts():
    s = placeholders.SETUP  # initial values of the form (an example plate, not a result)
    return {"a": s["a"][0], "b": s["b"][0], "h": s["h"][0], "E": s["E"][0], "nu": s["nu"], "q": s["q"][0],
            "q_series": "1; 2,5; 5; 7,5; 10", "n_x": "80", "n_y": "40",
            "tol_inf": _number_text(DEFAULTS["tol_inf"]), "tol_E": _number_text(DEFAULTS["tol_E"]),
            "s_min": str(DEFAULTS["s_min"]), "s_max": str(DEFAULTS["s_max"]),
            "half_steps": str(DEFAULTS["half_steps"])}


def _default_units():
    s = placeholders.SETUP
    return {"a": s["a"][1], "b": s["b"][1], "h": s["h"][1], "E": s["E"][1], "q": s["q"][1]}


SCOPE_OF_ROUTE = {ROUTE_R3: "r3", ROUTE_R4: "r4", ROUTE_SSSS_N46: "r46", **{route: "cccc" for route in CCCC_ROUTES}}


def scope_of(N, supports="SSSS"):
    """The parameter group of (supports, N): "r3", "r4", "r46" (SSSS research N = 4-6) or "cccc"."""
    return SCOPE_OF_ROUTE[route_of(supports, N)]


def cccc_half_steps_of(N):
    """The numbers of half-steps the form offers for CCCC with N."""
    return CCCC_HALF_STEPS[route_of("CCCC", N)]


def series_texts(text):
    """The value texts of a load-series field, in the order entered (separators: ";" and white space)."""
    for sep in SERIES_SEPARATORS[:-1]:
        text = text.replace(sep, " ")
    return [part for part in text.split(" ") if part]


def parse_series(text):
    """(values, error) of a load-series field: SERIES_MIN...SERIES_MAX distinct numbers."""
    parts = series_texts(text)
    values = []
    for part in parts:
        parsed = parse_number(part)
        if not parsed.ok:
            return None, f"«{part}»: {parsed.error}"
        values.append(parsed.value)
    if not SERIES_MIN <= len(values) <= SERIES_MAX:
        return None, (f"Нужно от {SERIES_MIN} до {SERIES_MAX} значений q через «;» или пробел (десятичный разделитель — "
                      "точка или запятая).")
    if len(set(values)) != len(values):
        return None, "Значения q серии не должны повторяться."
    return values, None


def _decimal(text):
    """(Decimal, error) of a number typed with "." or "," (the rules of parse_number; the value kept exactly)."""
    parsed = parse_number(text)
    if not parsed.ok:
        return None, parsed.error
    return Decimal(text.strip().replace("−", "-").replace(",", ".")), None


def _decimal_text(value, comma):
    """Plain decimal text without an exponent or trailing zeros: 2.50 -> "2,5" (comma) or "2.5"."""
    text = format(value.normalize(), "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    text = "0" if text in ("-0", "") else text
    return text.replace(".", ",") if comma else text


def generate_series(start, end, step):
    """(texts, error): the loads start, start + step, …, end as texts of "q_series" (in the unit of q).

    Decimal arithmetic: start + i·step exactly as typed, without accumulated binary rounding. Refused with a reason: a
    zero increment, an increment pointing away from the end, an end that the increment does not reach exactly (the end
    is never moved), a number of values outside SERIES_MIN…SERIES_MAX and values that coincide after conversion to
    binary64. The texts use the decimal comma if any of the three entries did.
    """
    values = {}
    for name, text in (("Начало", start), ("Конец", end), ("Шаг", step)):
        value, error = _decimal(text)
        if error:
            return None, f"{name}: {error}"
        values[name] = value
    first, last, inc = values["Начало"], values["Конец"], values["Шаг"]
    if inc == 0:
        return None, "Шаг не может быть равен нулю."
    if first == last:
        return None, f"Начало и конец совпадают: в серии нужно от {SERIES_MIN} до {SERIES_MAX} разных значений q."
    if (last - first) * inc < 0:
        direction = "положительным" if last > first else "отрицательным"
        return None, f"Шаг направлен от конца: при начале {'меньше' if last > first else 'больше'} конца шаг должен быть {direction}."
    with localcontext() as ctx:
        ctx.prec = 60
        try:
            count_minus_one, remainder = divmod(last - first, inc)
        except InvalidOperation:
            return None, f"Слишком много значений: от {SERIES_MIN} до {SERIES_MAX}."
        count = int(count_minus_one) + 1
        if remainder != 0:
            below = first + count_minus_one * inc
            return None, (f"Конец не достигается шагом: после {_decimal_text(below, False)} следующее значение "
                          f"{_decimal_text(below + inc, False)}. Измените конец или шаг — конец не подгоняется.")
        if not SERIES_MIN <= count <= SERIES_MAX:
            return None, f"Получается {count} значений q; допускается от {SERIES_MIN} до {SERIES_MAX}. Измените шаг."
        series = [first + i * inc for i in range(count)]
    if len({float(v) for v in series}) != count:
        return None, "Значения q совпадают после перевода в двоичное число: увеличьте шаг."
    comma = any("," in text for text in (start, end, step))
    return [_decimal_text(v, comma) for v in series], None


def validate_field(key, text):
    """(value, error) for one field."""
    spec = FIELDS[key]
    if spec.rule == "series":
        return parse_series(text)
    parsed = parse_number(text)
    if not parsed.ok:
        return None, parsed.error
    value = parsed.value
    if spec.rule == "positive" and not value > 0:
        return None, POSITIVE
    if spec.rule == "count":
        if value < 1 or not float(value).is_integer():
            return None, COUNT
        return int(value), None
    return value, None


@dataclass(frozen=True)
class SetupState:
    texts: dict = field(default_factory=_default_texts)
    units: dict = field(default_factory=_default_units)
    supports: str = "SSSS"
    theory: str = THEORY
    N: int = 1
    r4_mode: str = DEFAULTS["r4_mode"]
    transform: str = DEFAULTS["transform"]
    strict: bool = False
    ssss_research: bool = False  # SSSS N = 4-6: the explicit research choice (D-056.7)
    cccc_half_steps: int = 1  # CCCC: the exact number of half-steps; s = 1 by default (no continuation)
    load_mode: str = "single"  # "single": one load q; "series": the loads of "q_series", one run each
    with_d2: bool = True  # CCCC: also «МКР — двумерный (D2)» for the same plate and grid (never for SSSS)
    display: frozenset = DISPLAY_ALL  # curves shown after the run: a display choice only, never a computation

    def with_text(self, key, text):
        return replace(self, texts={**self.texts, key: text})

    def with_unit(self, key, unit):
        if unit not in FIELDS[key].units:
            raise ValueError(f"{unit!r} is not an allowed unit of {key}")
        return replace(self, units={**self.units, key: unit})

    def with_choice(self, **choices):
        if "supports" in choices and choices["supports"] not in SUPPORTS:
            raise ValueError(f"unknown supports {choices['supports']!r}")
        if "N" in choices and choices["N"] not in N_AVAILABLE:
            raise ValueError(f"N has no route: {choices['N']!r}")
        if "r4_mode" in choices and choices["r4_mode"] not in R4_MODES:
            raise ValueError(f"unknown R4 mode {choices['r4_mode']!r}")
        if "transform" in choices and choices["transform"] not in TRANSFORMS:
            raise ValueError(f"unknown transform {choices['transform']!r}")
        if "with_d2" in choices and not isinstance(choices["with_d2"], bool):
            raise ValueError(f"with_d2 must be True or False, got {choices['with_d2']!r}")
        if "load_mode" in choices and choices["load_mode"] not in LOAD_MODES:
            raise ValueError(f"unknown load mode {choices['load_mode']!r}")
        if "display" in choices:
            choices["display"] = frozenset(choices["display"])
            if not choices["display"] <= DISPLAY_ALL:
                raise ValueError(f"unknown curves {sorted(map(str, choices['display'] - DISPLAY_ALL))}")
        changed = any(key in choices and choices[key] != getattr(self, key) for key in ("N", "supports"))
        if changed:  # no implicit research continuation after a change of N or of the supports (D-048.5, D-056.7)
            choices.setdefault("r4_mode", R4_NORMATIVE)
            choices.setdefault("ssss_research", False)
            choices.setdefault("cccc_half_steps", 1)
        state = replace(self, **choices)
        if state.supports == "CCCC" and state.cccc_half_steps not in cccc_half_steps_of(state.N):
            raise ValueError(f"CCCC, N = {state.N}: {state.cccc_half_steps!r} half-steps are not offered")
        return state

    @property
    def route(self):
        return route_of(self.supports, self.N)

    @property
    def scope(self):
        return scope_of(self.N, self.supports)

    @property
    def research(self):
        """True only if the research continuation through a QR is chosen explicitly (D-048.5)."""
        return self.r4_mode in R4_RESEARCH_MODES

    @property
    def series(self):
        return self.load_mode == "series"

    @property
    def d2(self):
        """True if «МКР — двумерный (D2)» is computed with the run: CCCC and the choice is on."""
        return self.supports == "CCCC" and self.with_d2

    def active_fields(self):
        """Fields that take part in a run of the current supports, N and load mode, in form order."""
        scope = self.scope
        skip = "q" if self.series else "q_series"
        return [key for key, spec in FIELDS.items() if ("all" in spec.scope or scope in spec.scope) and key != skip]


@dataclass(frozen=True)
class DisplayOption:
    key: object  # a DISPLAY_KEYS item
    title: str
    possible: bool  # the chosen route and parameters can give this curve (before the run: possible, not computed)
    text: str  # what decides it, or why it cannot be shown


def display_of(state):
    """The display choice stored with a run (``RunRecord.snapshot``); all curves for an older snapshot without it."""
    return frozenset(getattr(state, "display", DISPLAY_ALL))


def display_options(state):
    """[DisplayOption] of DISPLAY_KEYS for the current form, before any run (Qt-free).

    Cycles: VIM-k is the state s = 2k of THIS run, read after the run from its half-step history; only the routes that
    record the centre value per half-step can give it (R3 for SSSS N = 1: w̄ at the centre; CCCC: α at the centre). The
    centre (a/2, b/2) must be a grid node. The run may still end before s = 2k: the analysis window then states why.
    """
    route, N = state.route, state.N
    nx, ny = validate_field("n_x", state.texts.get("n_x", "")), validate_field("n_y", state.texts.get("n_y", ""))
    odd = (nx[1] is None and ny[1] is None and float(nx[0]).is_integer() and float(ny[0]).is_integer()
           and (int(nx[0]) % 2 or int(ny[0]) % 2))
    out = [DisplayOption("kkvim", DISPLAY_TITLES["kkvim"], True,
                         "поле последнего выполненного полушага этого расчёта; статус — по решателю (CONVERGED — критерий "
                         "остановки выполнен; остановка по пределу полушагов — не сходимость)")]

    def cycle(k):
        s = 2 * k
        if route in (ROUTE_R4, ROUTE_SSSS_N46):
            name = "R4 (N = 2–3)" if route == ROUTE_R4 else "исследования SSSS N = 4–6"
            return False, (f"маршрут {name} записывает на каждом полушаге только max|w|, не прогиб в центре; "
                           "VIM-1–3 не вычисляются и не подменяются итоговым полем")
        if odd:
            return False, "центр (a/2, b/2) не узел сетки (nₓ или nᵧ нечётно); значение не интерполируется"
        if route == ROUTE_R3:
            s_max, error = validate_field("s_max", state.texts.get("s_max", ""))
            if error is None and s_max < s:
                return False, f"s_max = {int(s_max)} < {s}"
            return True, (f"w̄ в центре на полушаге s = {s} из истории R3, если расчёт выполнит s = {s} "
                          "(критерий остановки может остановить его раньше)")
        steps = state.cccc_half_steps
        if route != ROUTE_CCCC_N1 and s > 2:
            return False, f"маршрут CCCC, N = {N} допускает не более s = 2 (RN-5, RT-7); s = {s} не вычисляется"
        if steps < s:
            return False, f"выбрано полушагов s = {steps} < {s}"
        guard = " после проверок RN-5 и RT-7 этого расчёта" if route != ROUTE_CCCC_N1 else ""
        return True, f"α в центре на полушаге s = {s} из истории CCCC{guard}"

    for k in (1, 2, 3):
        out.append(DisplayOption(k, DISPLAY_TITLES[k], *cycle(k)))
    if state.supports == "SSSS":
        out.append(DisplayOption("navier", DISPLAY_TITLES["navier"], True,
                                 "рассчитывается автоматически в отдельном процессе для каждого q; флажок — только показ"))
        out.append(DisplayOption("d2", DISPLAY_TITLES["d2"], False,
                                 "для SSSS: D2 — сравнение для CCCC; эталон SSSS — Навье"))
    else:
        out.append(DisplayOption("navier", DISPLAY_TITLES["navier"], False,
                                 "решение Навье относится только к шарнирному опиранию SSSS"))
        out.append(DisplayOption("d2", DISPLAY_TITLES["d2"], state.with_d2,
                                 "будет рассчитан в отдельном процессе (флажок «МКР — двумерный (D2)» в «Граничные условия»)"
                                 if state.with_d2 else
                                 "не будет рассчитан: выключен флажок «МКР — двумерный (D2)» в «Граничные условия»"))
    return out


def validate(state):
    """{key: error} for every invalid active field (and the supports), in form order."""
    errors = {}
    for key in state.active_fields():
        _, error = validate_field(key, state.texts.get(key, ""))
        if error:
            errors[key] = error
    if state.supports not in SUPPORTS:
        errors["supports"] = f"Неизвестное закрепление {state.supports!r}"
    if state.scope == "r46" and not state.ssss_research:
        errors["research"] = RESEARCH_REQUIRED
    if state.scope == "r4" and "n_x" not in errors and "n_y" not in errors:
        dense = dense_problem(state.transform, _value(state, "n_x"), _value(state, "n_y"))
        if dense:
            errors["transform"] = dense
    return errors


def _value(state, key):
    value, _ = validate_field(key, state.texts[key])
    return value


def to_config(state):
    """Parsed configuration (plain numbers, SI units as entered), or None if invalid.

    The same values give the same configuration whichever decimal separator was typed.
    """
    if validate(state):
        return None

    def quantity(key):
        value = _value(state, key)
        return {"value": value, "unit": state.units[key]} if FIELDS[key].units else value

    method = {"N": state.N, "n_x": quantity("n_x"), "n_y": quantity("n_y"), "route": state.route}
    if state.scope == "r3":
        method.update({k: quantity(k) for k in ("tol_inf", "tol_E", "s_min", "s_max")}, strict=state.strict)
    elif state.scope == "r4":
        method.update(r4_mode=state.r4_mode, transform=state.transform, half_steps=quantity("half_steps"))
    elif state.scope == "r46":
        method.update(ssss_research=state.ssss_research, half_steps=quantity("half_steps"))
    else:
        method.update(half_steps=state.cccc_half_steps)
        method["independent_d2"] = state.d2  # «МКР — двумерный (D2)» for the same plate and grid
    return {
        "geometry": {k: quantity(k) for k in ("a", "b", "h")},
        "material": {"E": quantity("E"), "nu": quantity("nu")},
        "load": ({"type": "uniform", "series": {"values": _value(state, "q_series"), "unit": state.units["q"]}}
                 if state.series else {"type": "uniform", "q": quantity("q")}),
        "supports": state.supports,
        "theory": state.theory,
        "method": method,
    }


def to_series_requests(state):
    """One RunRequest per load of the series (only the text of q differs), or None if the form is invalid."""
    if not state.series or validate(state):
        return None
    single = replace(state, load_mode="single")
    return [to_request(single.with_text("q", text)) for text in series_texts(state.texts["q_series"])]


def series_preview(state):
    """The loads of the series as entered, for the preview before the run: [(text, unit)], or None if invalid."""
    values, error = parse_series(state.texts.get("q_series", ""))
    if error:
        return None
    return [(text, state.units["q"]) for text in series_texts(state.texts["q_series"])]


def to_request(state):
    """RunRequest in the base SI units for the solver, or None if the form is invalid (a series: ``to_series_requests``)."""
    if validate(state) or state.series:
        return None
    si = {key: units.to_base(_value(state, key), state.units[key]) for key in ("a", "b", "h", "E", "q")}
    entered = tuple((key, state.texts[key], state.units.get(key, "")) for key in state.active_fields())
    common = dict(entered=entered, a=si["a"], b=si["b"], t=si["h"], E=si["E"], nu=_value(state, "nu"), q=si["q"],
                  n_x=_value(state, "n_x"), n_y=_value(state, "n_y"), N=state.N, supports=state.supports)
    if state.scope == "r3":
        return RunRequest(**common, tol_inf=_value(state, "tol_inf"), tol_E=_value(state, "tol_E"),
                          s_min=_value(state, "s_min"), s_max=_value(state, "s_max"), strict=state.strict)
    if state.scope == "r46":
        return RunRequest(**common, half_steps=_value(state, "half_steps"), ssss_research=state.ssss_research)
    if state.scope == "cccc":
        return RunRequest(**common, half_steps=state.cccc_half_steps)
    return RunRequest(**common, r4_mode=state.r4_mode, transform=state.transform,
                      half_steps=_value(state, "half_steps"))


def method_text(state):
    if state.scope == "cccc":
        d2 = " · + МКР — двумерный (D2) для сравнения" if state.d2 else ""
        return (f"КК-ВИМ, N = {state.N} · CCCC, исследование, s = {state.cccc_half_steps} · RESEARCH_ONLY / PROVISIONAL / "
                f"NOT_QUALIFIED · {state.theory}{d2}")
    if state.scope == "r46":
        chosen = "выбрано" if state.ssss_research else "не выбрано"
        return f"КК-ВИМ, N = {state.N} · SSSS, исследование N = 4–6 ({chosen}) · RESEARCH_ONLY · {state.theory}"
    if state.N == 1:
        strict = " · строгий режим" if state.strict else ""
        return f"КК-ВИМ, N = 1 · R3 (принятый решатель){strict} · {state.theory}"
    mode = {R4_RESEARCH_STOP: "исследовательский D-048", R4_RESEARCH_FIXED: "фиксированное число полушагов",
            R4_NORMATIVE: "нормативный D-047"}[state.r4_mode]
    return f"КК-ВИМ, N = {state.N} · R4, {mode}, {state.transform} · {state.theory}"


def summary_rows(state):
    """Text summary: the entered text and unit, without parsing or conversion."""
    def show(key):
        text = state.texts[key].strip() or "—"
        unit = state.units.get(key, "")
        return f"{text} {unit}".strip()

    return [
        ("Геометрия", f"a {show('a')} · b {show('b')} · h {show('h')}"),
        ("Материал", f"E {show('E')} · ν {show('nu')}"),
        ("Нагрузка", (f"равномерная, серия q: {'; '.join(series_texts(state.texts['q_series'])) or '—'} {state.units['q']}"
                      if state.series else f"равномерная, q {show('q')}")),
        ("Закрепление", state.supports),
        ("Метод", method_text(state)),
        ("Сетка", f"n_x {show('n_x')} · n_y {show('n_y')}"),
    ]
