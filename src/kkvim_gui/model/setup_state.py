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
"""

from dataclasses import dataclass, field, replace

from kkvim_gui import placeholders
from kkvim_gui.backend import units
from kkvim_gui.backend.kkvim_adapter import DEFAULTS, N_AVAILABLE, dense_problem
from kkvim_gui.backend.request import (CCCC_HALF_STEPS, CCCC_ROUTES, R4_MODES, R4_NORMATIVE, R4_RESEARCH_FIXED,
                                      R4_RESEARCH_MODES, R4_RESEARCH_STOP, ROUTE_R3, ROUTE_R4, ROUTE_SSSS_N46, SUPPORTS, TRANSFORMS,
                                      RunRequest, route_of)
from kkvim_gui.model.numbers import parse_number

POSITIVE = "Значение должно быть больше нуля."
COUNT = "Нужно целое число не меньше 1."
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
        FieldSpec("n_x", "Интервалов по x n_x", "count"),
        FieldSpec("n_y", "Интервалов по y n_y", "count"),
        FieldSpec("tol_inf", "Допуск tol_∞", "positive", scope=("r3",)),
        FieldSpec("tol_E", "Допуск tol_E", "positive", scope=("r3",)),
        FieldSpec("s_min", "Мин. полушагов s_min", "count", scope=("r3",)),
        FieldSpec("s_max", "Макс. полушагов s_max", "count", scope=("r3",)),
        FieldSpec("half_steps", "Полушагов (предел)", "count", scope=("r4", "r46")),
    )
}
THEORY = "Кирхгоф–Ляв"
N_VALUES = (1, 2, 3, 4, 5, 6)  # every N has a route (request.route_of); each route has its own domain and status
R4_MODE_TEXT = {
    R4_NORMATIVE: "Нормативный D-047: только s = 1, далее NOT_QUALIFIED",
    R4_RESEARCH_STOP: "Исследовательский D-048: остановка по стабилизации поля",
    R4_RESEARCH_FIXED: "Исследовательский: фиксированное число полушагов (дополнение к D-048)",
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
            "n_x": "80", "n_y": "40",
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


def validate_field(key, text):
    """(value, error) for one field."""
    spec = FIELDS[key]
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

    def active_fields(self):
        """Fields that take part in a run of the current supports and N, in form order."""
        scope = self.scope
        return [key for key, spec in FIELDS.items() if "all" in spec.scope or scope in spec.scope]


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
    return {
        "geometry": {k: quantity(k) for k in ("a", "b", "h")},
        "material": {"E": quantity("E"), "nu": quantity("nu")},
        "load": {"type": "uniform", "q": quantity("q")},
        "supports": state.supports,
        "theory": state.theory,
        "method": method,
    }


def to_request(state):
    """RunRequest in the base SI units for the solver, or None if the form is invalid."""
    if validate(state):
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
        return (f"КК-ВИМ, N = {state.N} · CCCC, исследование, s = {state.cccc_half_steps} · RESEARCH_ONLY / PROVISIONAL / "
                f"NOT_QUALIFIED · {state.theory}")
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
        ("Нагрузка", f"равномерная, q {show('q')}"),
        ("Закрепление", state.supports),
        ("Метод", method_text(state)),
        ("Сетка", f"n_x {show('n_x')} · n_y {show('n_y')}"),
    ]
