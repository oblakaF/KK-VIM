"""P2.2: plate scheme data (GUI_DESIGN.md, 5.2). Qt-free."""

import pytest

from kkvim_gui.model.scheme import scheme_data
from kkvim_gui.model.setup_state import SetupState


def state(**kw):
    s = SetupState()
    for key, value in kw.items():
        if key == "supports":
            s = s.with_choice(supports=value)
        elif key.endswith("_unit"):
            s = s.with_unit(key[:-5], value)
        else:
            s = s.with_text(key, value)
    return s


# Data (Qt-free)

def test_square_and_rectangle_proportions():
    assert scheme_data(state(a="1", b="1")).aspect == 1.0
    assert scheme_data(state(a="2,000", b="1,000")).aspect == 2.0


def test_units_are_respected_only_for_drawing():
    d = scheme_data(state(a="2", b="50", b_unit="cm", h="0,02", h_unit="m"))
    assert d.aspect == pytest.approx(4.0)
    assert d.b_label == "b = 50 cm" and d.h_label == "h = 0,02 m"  # text as entered


def test_labels_follow_input_text():
    d = scheme_data(state(h="12,5", q="-7.5"))
    assert d.h_label == "h = 12,5 mm"
    assert d.q_label == "q = -7.5 kPa"


@pytest.mark.parametrize("bad", ["", "-", "1e-", "abc", "0", "-2", "1 000"])
def test_invalid_geometry_draws_no_geometry(bad):
    d = scheme_data(state(b=bad))
    assert not d.valid and d.waiting_for == ("Ширина b",)
    assert d.aspect is None and d.a_m is None


def test_several_invalid_fields_are_listed():
    d = scheme_data(state(a="x", h=""))
    assert d.waiting_for == ("Длина a", "Толщина h")


def test_invalid_q_keeps_geometry_but_marks_the_load():
    d = scheme_data(state(q="1e-"))
    assert d.valid and not d.q_valid
    assert "ожидает исправления" in d.q_label


def test_supports_follow_the_state():
    assert scheme_data(state(supports="CCCC")).supports == "CCCC"
