"""Setup state of the main window and its formal validation (GUI_DESIGN.md, 5.2, 11). Qt-free.

Only rules already approved for the GUI are checked (GUI_DESIGN.md, 5.2): a value is
present, is a number and is finite; a, b, h and E are greater than zero; k_max, a number
of iterations (D-003), is a whole number not less than 1. Scientific ranges (for example
of ν) and other limits come from the specification and are not invented here. This is a
formal check of the input, not a check by the scientific solver.

Values keep the text and the SI unit exactly as entered; units are never converted
(GUI_DESIGN.md, 11.1). The parsed configuration contains plain floats.
"""

from dataclasses import dataclass, field, replace

from kkvim_gui import placeholders
from kkvim_gui.model.numbers import parse_number

POSITIVE = "Значение должно быть больше нуля."
COUNT = "Нужно целое число не меньше 1."


@dataclass(frozen=True)
class FieldSpec:
    key: str
    label: str
    rule: str  # "positive" | "number" | "count" | "preliminary"
    units: tuple = ()


FIELDS = {
    spec.key: spec
    for spec in (
        FieldSpec("a", "Длина a", "positive", ("m", "cm", "mm")),
        FieldSpec("b", "Ширина b", "positive", ("m", "cm", "mm")),
        FieldSpec("h", "Толщина h", "positive", ("m", "cm", "mm")),
        FieldSpec("E", "Модуль упругости E", "positive", ("GPa", "MPa", "Pa")),
        FieldSpec("nu", "Коэффициент Пуассона ν", "number"),
        FieldSpec("q", "Интенсивность q", "number", ("kPa", "Pa", "MPa")),
        FieldSpec("k_max", "Макс. итераций k_max", "count"),
        # Content defined by NUMERICAL_SPEC (Q-009, Q-010): kept as text, not validated.
        FieldSpec("grid", "Сетка МКР", "preliminary"),
        FieldSpec("stop_rule", "Критерий остановки", "preliminary"),
    )
}
SUPPORTS = ("SSSS", "CCCC")
THEORY = "Кирхгоф–Ляв"
N_VALUES = (1, 2, 3, 4, 5, 6)  # 1-3 main range, 4-6 research (D-003)


def validate_field(key, text):
    """(value, error) for one field; value is None for preliminary fields."""
    spec = FIELDS[key]
    if spec.rule == "preliminary":
        return None, None
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


def _default_texts():
    s = placeholders.SETUP
    return {"a": s["a"][0], "b": s["b"][0], "h": s["h"][0], "E": s["E"][0], "nu": s["nu"],
            "q": s["q"][0], "k_max": s["k_max"], "grid": "", "stop_rule": ""}


def _default_units():
    s = placeholders.SETUP
    return {"a": s["a"][1], "b": s["b"][1], "h": s["h"][1], "E": s["E"][1], "q": s["q"][1]}


@dataclass(frozen=True)
class SetupState:
    texts: dict = field(default_factory=_default_texts)
    units: dict = field(default_factory=_default_units)
    supports: str = "SSSS"
    theory: str = THEORY
    N: int = 3

    def with_text(self, key, text):
        return replace(self, texts={**self.texts, key: text})

    def with_unit(self, key, unit):
        if unit not in FIELDS[key].units:
            raise ValueError(f"{unit!r} is not an allowed unit of {key}")
        return replace(self, units={**self.units, key: unit})

    def with_choice(self, **choices):
        if "supports" in choices and choices["supports"] not in SUPPORTS:
            raise ValueError(f"unknown supports {choices['supports']!r}")
        if "N" in choices and choices["N"] not in N_VALUES:
            raise ValueError(f"N out of the approved range: {choices['N']!r}")
        return replace(self, **choices)


def validate(state):
    """{key: error} for every invalid field, in form order."""
    errors = {}
    for key in FIELDS:
        _, error = validate_field(key, state.texts.get(key, ""))
        if error:
            errors[key] = error
    return errors


def to_config(state):
    """Parsed configuration (plain numbers, SI units as entered), or None if invalid.

    The same values give the same configuration whichever decimal separator was typed.
    """
    if validate(state):
        return None

    def quantity(key):
        value, _ = validate_field(key, state.texts[key])
        return {"value": value, "unit": state.units[key]} if FIELDS[key].units else value

    return {
        "geometry": {k: quantity(k) for k in ("a", "b", "h")},
        "material": {"E": quantity("E"), "nu": quantity("nu")},
        "load": {"type": "uniform", "q": quantity("q")},
        "supports": state.supports,
        "theory": state.theory,
        "method": {
            "N": state.N,
            "k_max": quantity("k_max"),
            "preliminary": {"grid": state.texts["grid"], "stop_rule": state.texts["stop_rule"]},
        },
    }


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
        ("Метод", f"КК-ВИМ, N = {state.N} · {state.theory}"),
    ]

