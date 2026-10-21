"""Number parsing, formal validation and setup configuration (GUI_DESIGN.md, 5.2, 11). Qt-free."""

import json

import pytest

from kkvim_gui.model.numbers import EMPTY, NOT_A_NUMBER, NOT_FINITE, THOUSANDS, parse_number
from kkvim_gui.model.setup_state import (
    COUNT,
    POSITIVE,
    SetupState,
    summary_rows,
    to_config,
    validate,
    validate_field,
)


@pytest.mark.parametrize(("text", "value"), [
    ("0.1", 0.1), ("0,1", 0.1),
    ("-0.25", -0.25), ("-0,25", -0.25),
    ("1.5e-3", 1.5e-3), ("1,5e-3", 1.5e-3),
    ("−2.5E+3", -2500.0),          # Unicode minus, pasted from documents
    ("+3", 3.0), (" 2,000 ", 2.0),
    ("1,000", 1.0),                 # "," is a decimal separator, not thousands
    (".5", 0.5), (",5", 0.5), ("5.", 5.0), ("1e3", 1000.0),
])
def test_numbers_with_point_or_comma(text, value):
    parsed = parse_number(text)
    assert parsed.ok and parsed.value == value


@pytest.mark.parametrize(("text", "error"), [
    ("", EMPTY), ("   ", EMPTY),
    ("-", NOT_A_NUMBER), ("1e-", NOT_A_NUMBER), ("1,5E", NOT_A_NUMBER), ("abc", NOT_A_NUMBER),
    ("e3", NOT_A_NUMBER), ("--1", NOT_A_NUMBER), ("nan", NOT_A_NUMBER), ("inf", NOT_A_NUMBER),
    ("1 000", THOUSANDS), ("1 000", THOUSANDS), ("1,000.5", THOUSANDS), ("1.000,5", THOUSANDS),
    ("1.000.000", THOUSANDS),
    ("1e400", NOT_FINITE),
])
def test_invalid_numbers(text, error):
    parsed = parse_number(text)
    assert not parsed.ok and parsed.error == error and parsed.value is None


@pytest.mark.parametrize(("key", "text", "error"), [
    ("a", "0", POSITIVE), ("b", "-1", POSITIVE), ("h", "0,0", POSITIVE), ("E", "-210", POSITIVE),
    ("n_x", "0", COUNT), ("n_y", "2,5", COUNT), ("s_max", "-3", COUNT), ("half_steps", "0", COUNT),
    ("tol_E", "0", POSITIVE), ("tol_inf", "-1e-10", POSITIVE),
    ("a", "", EMPTY), ("E", "двести", NOT_A_NUMBER),
])
def test_formal_rules(key, text, error):
    assert validate_field(key, text) == (None, error)


@pytest.mark.parametrize(("key", "text", "value"), [
    ("nu", "-0,3", -0.3),   # no invented physical range: the specification defines it
    ("nu", "0,7", 0.7),
    ("q", "-5", -5.0),      # load sign follows the specification
    ("s_max", "1e2", 100),
    ("n_x", "3", 3),        # N_MIN is checked by the solver, not invented in the form
    ("tol_E", "1e-8", 1e-8),
])
def test_values_accepted_without_invented_limits(key, text, value):
    assert validate_field(key, text) == (value, None)


def test_default_state_is_valid():
    assert validate(SetupState()) == {}


def test_point_and_comma_give_identical_configuration():
    comma = SetupState().with_text("h", "0,1").with_text("nu", "0,25")
    point = SetupState().with_text("h", "0.1").with_text("nu", "0.25")
    assert to_config(comma) == to_config(point)
    assert json.dumps(to_config(comma)) == json.dumps(to_config(point))
    assert '"value": 0.1' in json.dumps(to_config(comma))  # machine format: point


def test_unit_change_never_converts_the_number():
    state = SetupState().with_text("h", "10,0").with_unit("h", "m")
    assert to_config(state)["geometry"]["h"] == {"value": 10.0, "unit": "m"}


def test_invalid_state_has_no_configuration():
    state = SetupState().with_text("a", "abc")
    assert validate(state) == {"a": NOT_A_NUMBER}
    assert to_config(state) is None


def test_choices_are_limited_to_approved_values():
    with pytest.raises(ValueError):
        SetupState().with_choice(supports="SFSF")
    with pytest.raises(ValueError):
        SetupState().with_choice(N=7)
    with pytest.raises(ValueError):
        SetupState().with_unit("h", "inch")


def test_summary_echoes_entered_text_and_units():
    state = SetupState().with_text("a", "3,5").with_unit("h", "cm").with_choice(supports="CCCC", N=2)
    rows = dict(summary_rows(state))
    assert "a 3,5 m" in rows["Геометрия"] and "cm" in rows["Геометрия"]
    assert rows["Закрепление"] == "CCCC"
    assert rows["Метод"].startswith("КК-ВИМ, N = 2")
    assert dict(summary_rows(state.with_text("b", "")))["Геометрия"].count("—") == 1
