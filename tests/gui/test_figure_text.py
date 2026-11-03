"""Figure text catalog EN/RU (GUI_DESIGN.md, 7.3-7.4, 11.1). Qt-free."""

import re

import pytest

from kkvim_gui.figures.text import LANGUAGES, catalog, fig_text, format_number

CYRILLIC_UNITS = re.compile(r"(?<![а-яА-ЯёЁ])(мм|см|м|кПа|МПа|ГПа|Па)(?![а-яА-ЯёЁ])")
SI_UNIT = re.compile(r"(?<![A-Za-z])(mm|cm|m|Pa|kPa|MPa|GPa)(?![A-Za-z])")


def test_languages_have_the_same_keys():
    assert set(catalog("en")) == set(catalog("ru"))


def test_default_language_is_listed_first():
    assert LANGUAGES[0] == "en"


@pytest.mark.parametrize("key", sorted(catalog("en")))
def test_unit_symbols_are_identical_si_in_both_languages(key):
    en, ru = catalog("en")[key], catalog("ru")[key]
    assert CYRILLIC_UNITS.search(ru) is None, ru
    assert SI_UNIT.findall(en) == SI_UNIT.findall(ru)


def test_decimal_separator_follows_the_language_and_value_is_unchanged():
    assert format_number(0.842, "en") == "0.842"
    assert format_number(0.842, "ru") == "0,842"
    assert fig_text("en", "section.x.title", y=0.5) == "Section at y = 0.500 m"
    assert fig_text("ru", "section.x.title", y=0.5) == "Сечение при y = 0,500 m"


def test_test_data_marks():
    assert fig_text("en", "mark.test_data") == "TEST DATA"
    assert fig_text("ru", "mark.test_data") == "ТЕСТОВЫЕ ДАННЫЕ · TEST DATA"
