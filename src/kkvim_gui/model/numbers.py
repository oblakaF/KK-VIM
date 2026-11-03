"""Parsing of numbers typed or pasted by the user (GUI_DESIGN.md, 11.2). Qt-free.

- "." and "," are both decimal separators: 0.1 == 0,1; -0,25; 1,5e-3; -2.5E+3.
- Sign: "+", "-" or the Unicode minus U+2212 (often pasted from documents).
- No thousands separators: inner spaces, or both "." and "," in one number, are errors
  ("1,000" is 1.0 because "," is a decimal separator).
- The text itself is never rewritten; parsing happens when the value is committed.
- The result is a plain Python float; machine formats write it with a dot.
"""

import math
import re
from dataclasses import dataclass

NUMBER = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$")
SPACES = re.compile(r"\s")

EMPTY = "Введите значение."
NOT_A_NUMBER = "Не число. Примеры: 0,25; -0.25; 1,5e-3."
THOUSANDS = "Разделители тысяч не допускаются: вводите число без пробелов и без второго разделителя."
NOT_FINITE = "Слишком большое по модулю значение."


@dataclass(frozen=True)
class Parsed:
    value: float | None
    error: str | None

    @property
    def ok(self):
        return self.error is None


def parse_number(text):
    s = text.strip().replace("−", "-")
    if not s:
        return Parsed(None, EMPTY)
    if SPACES.search(s):
        return Parsed(None, THOUSANDS)
    if "." in s and "," in s:
        return Parsed(None, THOUSANDS)
    if s.count(",") > 1 or s.count(".") > 1:
        return Parsed(None, THOUSANDS)
    s = s.replace(",", ".")
    if not NUMBER.match(s):
        return Parsed(None, NOT_A_NUMBER)
    value = float(s)
    if not math.isfinite(value):
        return Parsed(None, NOT_FINITE)
    return Parsed(value, None)
