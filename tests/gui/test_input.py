"""P2.1: working input fields, session model and validation in the GUI (GUI_DESIGN.md, 5.2, 11.2)."""

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

from PySide6.QtCore import QEvent, Qt  # noqa: E402
from PySide6.QtGui import QGuiApplication, QKeyEvent  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from kkvim_gui.app import GuiApplication  # noqa: E402
from kkvim_gui.main_window.setup_view import STATUS_OK  # noqa: E402
from kkvim_gui.model.numbers import EMPTY, NOT_A_NUMBER  # noqa: E402
from kkvim_gui.model.setup_state import COUNT, POSITIVE  # noqa: E402
from kkvim_gui.session import Session  # noqa: E402

CTRL = Qt.KeyboardModifier.ControlModifier


@pytest.fixture
def gui(qtbot, qapp):
    session = Session(settings=None)
    session.apply_theme()
    g = GuiApplication(session)
    qtbot.addWidget(g.main_window)
    g.main_window.show()
    qtbot.waitExposed(g.main_window)
    g.main_window.set_state("setup")
    yield g
    for w in list(g.analysis_windows):
        w.close()
    g.help.close()


@pytest.fixture
def view(gui):
    return gui.main_window.setup_view


def retype(qtbot, field, text, commit=True):
    edit = field.edit
    edit.setFocus()
    qtbot.keyClick(edit, Qt.Key.Key_A, CTRL)
    if text:
        qtbot.keyClicks(edit, text)
    else:
        qtbot.keyClick(edit, Qt.Key.Key_Delete)
    if commit:
        qtbot.keyClick(edit, Qt.Key.Key_Return)


def shown(view, key):
    err = view.errors[key]
    return err.text() if err.isVisibleTo(view) else None


def test_invalid_value_shows_message_next_to_the_field_and_keeps_text(qtbot, view):
    retype(qtbot, view.h, "abc")
    assert shown(view, "h") == NOT_A_NUMBER
    assert view.h.text() == "abc"
    assert view.h.property("invalid") is True
    assert not view.run_button.isEnabled()
    assert "Исправьте поля: 1" in view.check_text.text()
    assert view.goto_error.isVisibleTo(view)


def test_no_modal_dialog_on_invalid_input(qtbot, view):
    retype(qtbot, view.a, "x")
    assert QApplication.activeModalWidget() is None


def test_intermediate_input_is_not_reported(qtbot, view):
    for partial in ("-", "0,", "1e-"):
        retype(qtbot, view.q, partial, commit=False)
        assert shown(view, "q") is None


def test_error_disappears_while_correcting(qtbot, view):
    retype(qtbot, view.E, "-210")
    assert shown(view, "E") == POSITIVE
    retype(qtbot, view.E, "210", commit=False)  # no Enter needed once the field is flagged
    assert shown(view, "E") is None
    assert view.run_button.isEnabled()
    assert view.check_text.text() == STATUS_OK


@pytest.mark.parametrize(("attr", "key", "text", "error"), [
    ("a", "a", "", EMPTY),
    ("b", "b", "0", POSITIVE),
    ("h", "h", "-1,5", POSITIVE),
    ("k_max", "k_max", "2,5", COUNT),
])
def test_formal_checks_in_the_form(qtbot, view, attr, key, text, error):
    retype(qtbot, getattr(view, attr), text)
    assert shown(view, key) == error


def test_no_invented_scientific_limits(qtbot, view):
    retype(qtbot, view.nu, "-0,3")
    retype(qtbot, view.q, "-5e3")
    assert shown(view, "nu") is None and shown(view, "q") is None


def test_points_commas_negatives_and_exponents_are_accepted(qtbot, view):
    for text in ("0.1", "0,1", "1,5e-3", "1.5E-3", "2,5e+1"):
        retype(qtbot, view.h, text)
        assert shown(view, "h") is None, text
    QGuiApplication.clipboard().setText("−0,25")
    retype(qtbot, view.q, "")
    qtbot.keyClick(view.q.edit, Qt.Key.Key_V, CTRL)
    qtbot.keyClick(view.q.edit, Qt.Key.Key_Return)
    assert view.q.text() == "−0,25" and shown(view, "q") is None


def test_run_is_blocked_and_focus_goes_to_the_first_error(qtbot, gui, view):
    retype(qtbot, view.k_max, "0")
    retype(qtbot, view.b, "abc")
    view.k_max.edit.setFocus()
    with qtbot.assertNotEmitted(view.runRequested):
        assert view.try_run() is False
    assert gui.main_window.state == "setup"
    assert view.b.edit.hasFocus()  # "b" comes before "k_max" in the form
    assert "Исправьте поля: 2" in view.check_text.text()


def test_f5_uses_the_same_check(qtbot, gui, view):
    retype(qtbot, view.a, "")
    gui.main_window._run_shortcut()
    assert gui.main_window.state == "setup"


def test_values_survive_switching_sections(qtbot, gui, view):
    w = gui.main_window
    retype(qtbot, view.b, "1,25")
    view.h.unit_combo.setCurrentText("cm")
    w.nav.buttons["start"].click()
    w.nav.buttons["setup"].click()
    assert view.b.text() == "1,25" and view.h.unit() == "cm"
    assert w.setup_model.state.texts["b"] == "1,25"
    assert w.setup_model.state.units["h"] == "cm"


def test_summary_follows_the_fields(qtbot, view):
    retype(qtbot, view.a, "3,5", commit=False)
    view.supports.set_current("CCCC")
    view.N.setCurrentIndex(1)
    texts = {k: v.text() for k, v in view._summary_values.items()}
    assert "a 3,5 m" in texts["Геометрия"]
    assert texts["Закрепление"] == "CCCC"
    assert texts["Метод"].startswith("КК-ВИМ, N = 2")


def test_model_changes_are_shown_in_the_form(view):
    view.model.set_text("a", "4")
    assert view.a.text() == "4"
    assert "a 4 m" in view._summary_values["Геометрия"].text()


def test_undo_restores_text_and_clears_the_error(qtbot, view):
    edit = view.h.edit
    edit.setFocus()
    qtbot.keyClick(edit, Qt.Key.Key_End)
    qtbot.keyClicks(edit, "x")
    qtbot.keyClick(edit, Qt.Key.Key_Return)
    assert shown(view, "h") == NOT_A_NUMBER
    qtbot.keyClick(edit, Qt.Key.Key_Z, CTRL)
    assert edit.text() == "10,0"
    assert shown(view, "h") is None
    qtbot.keyClick(edit, Qt.Key.Key_Y, CTRL)
    assert edit.text() == "10,0x"


def test_editing_shortcuts_match_by_key_not_by_layout_text(qtbot, view):
    """With the Russian layout Windows reports the same key code for Ctrl+A/C/V;
    Qt matches shortcuts by key code, so the Cyrillic character does not matter."""
    edit = view.nu.edit
    edit.setFocus()
    for key, cyr in ((Qt.Key.Key_A, "ф"), (Qt.Key.Key_C, "с")):
        QApplication.sendEvent(edit, QKeyEvent(QEvent.Type.KeyPress, key, CTRL, cyr))
    assert edit.selectedText() == "0,300"
    assert QGuiApplication.clipboard().text() == "0,300"
    QApplication.sendEvent(edit, QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_V, CTRL, "м"))
    assert edit.text() == "0,300"


def test_run_snapshot_reaches_progress_and_analysis(qtbot, gui, view):
    w = gui.main_window
    retype(qtbot, view.a, "3,5")
    view.try_run()
    assert w.state == "running"
    assert "a 3,5 m" in w.run_view.summary_text.text()
    assert "TEST DATA" in w.run_view.summary_text.text()
    snapshot = w.runs.active.snapshot
    w.runs.stop(w.runs.active.run_id)          # editing is possible again after the run
    w.set_state("setup")
    retype(qtbot, view.a, "9")
    assert snapshot.texts["a"] == "3,5"        # the run keeps its own snapshot
    analysis = gui.open_analysis()
    assert ("Геометрия", "a 3,5 m · b 1,000 m · h 10,0 mm") in analysis.model_rows


def test_errors_visible_in_the_narrow_layout(qtbot, view):
    view._apply_mode("narrow")
    retype(qtbot, view.k_max, "abc")
    assert view.errors["k_max"].isVisibleTo(view)
    assert view.errors["k_max"].wordWrap()
    view._apply_mode("wide")
