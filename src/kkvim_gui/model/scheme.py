"""Data of the plate scheme on the setup page (GUI_DESIGN.md, 5.2). Qt-free.

The scheme shows the current input, not a computation. Proportions use the lengths
converted to metres only for drawing (m, cm, mm are SI prefixes); the labels show the
text and unit exactly as entered, and the stored values are never changed.

Not shown until the KK-VIM formulation is accepted (MATH_SPEC.md, part A; D-029):
coordinate axes and the direction (sign) of the load q.
"""

from dataclasses import dataclass

from kkvim_gui.model.setup_state import FIELDS, validate_field

TO_METRES = {"m": 1.0, "cm": 1e-2, "mm": 1e-3}
GEOMETRY = ("a", "b", "h")


@dataclass(frozen=True)
class SchemeData:
    valid: bool                 # geometry a, b, h formally valid
    waiting_for: tuple          # labels of invalid geometry fields
    a_m: float | None = None
    b_m: float | None = None
    h_m: float | None = None
    a_label: str = ""
    b_label: str = ""
    h_label: str = ""
    q_valid: bool = False
    q_label: str = ""
    supports: str = "SSSS"

    @property
    def aspect(self):
        """a / b of the plan (true proportion)."""
        return self.a_m / self.b_m if self.valid else None

    @property
    def h_over_a(self):
        return self.h_m / self.a_m if self.valid else None


def _label(state, key, symbol):
    return f"{symbol} = {state.texts[key].strip()} {state.units[key]}"


def scheme_data(state):
    values, waiting = {}, []
    for key in GEOMETRY:
        value, error = validate_field(key, state.texts[key])
        if error:
            waiting.append(FIELDS[key].label)
        else:
            values[key] = value * TO_METRES[state.units[key]]
    _, q_error = validate_field("q", state.texts["q"])
    q_label = _label(state, "q", "q") if q_error is None else "q — ожидает исправления значения"
    common = dict(supports=state.supports, q_valid=q_error is None, q_label=q_label)
    if waiting:
        return SchemeData(valid=False, waiting_for=tuple(waiting), **common)
    return SchemeData(
        valid=True, waiting_for=(),
        a_m=values["a"], b_m=values["b"], h_m=values["h"],
        a_label=_label(state, "a", "a"), b_label=_label(state, "b", "b"),
        h_label=_label(state, "h", "h"),
        **common,
    )
