"""SI units of the input fields and their conversion to the base units m and Pa (D-009, GUI-D14).

The form keeps every value with the unit chosen by the user (GUI_DESIGN.md, 11.1). The solver needs
consistent units (``kkvim.physical``: no conversion is done there), so the adapter passes all lengths
in m and all moduli and loads in Pa (GUI_DESIGN.md, 3.3). The factors are exact decimal SI prefixes;
the result of every conversion is recorded in the run protocol next to the entered value.
"""

# factor to the base unit: value_in_base = value * FACTOR[unit]
LENGTH = {"m": 1.0, "cm": 1e-2, "mm": 1e-3}
PRESSURE = {"Pa": 1.0, "kPa": 1e3, "MPa": 1e6, "GPa": 1e9}
BASE = {"length": "m", "pressure": "Pa"}
KIND_OF_UNIT = {**{u: "length" for u in LENGTH}, **{u: "pressure" for u in PRESSURE}}


def to_base(value: float, unit: str) -> float:
    """Value in m (lengths) or Pa (moduli, loads)."""
    if unit in LENGTH:
        return value * LENGTH[unit]
    if unit in PRESSURE:
        return value * PRESSURE[unit]
    raise ValueError(f"unknown unit {unit!r}")


def base_unit(unit: str) -> str:
    return BASE[KIND_OF_UNIT[unit]]
