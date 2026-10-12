"""Figure text catalog EN/RU (GUI_DESIGN.md, 7.3-7.4). Qt-free.

Figures contain only keys; texts live in text/en.json and text/ru.json, which have the
same keys. The language changes texts and the decimal separator only: numerical
values and SI unit symbols are identical in both languages (GUI-D14).
"""

import json
from functools import cache
from importlib.resources import files

LANGUAGES = ("en", "ru")


@cache
def catalog(lang):
    if lang not in LANGUAGES:
        raise ValueError(f"unknown figure language: {lang!r}")
    return json.loads(files("kkvim_gui.figures").joinpath("text", f"{lang}.json").read_text("utf-8"))


def format_number(value, lang, decimals=3):
    """Fixed-point text of a number: EN "0.842", RU "0,842". The value is not changed."""
    text = f"{value:.{decimals}f}"
    return text.replace(".", ",") if lang == "ru" else text


def fig_text(lang, key, decimals=3, **values):
    """Text for `key`; float placeholders are formatted for the figure language."""
    fields = {
        name: format_number(v, lang, decimals) if isinstance(v, float) else v
        for name, v in values.items()
    }
    return catalog(lang)[key].format(**fields)
