"""Standard Windows editing in input fields (GUI_DESIGN.md, 12.1; checks E1, E3-E5 of 14.1).

Key events are sent through Qt, so the result does not depend on the keyboard layout;
the RU/EN layout check itself (E2) is manual. Number parsing is stage P2: here the
text must stay exactly as typed or pasted, with "." or ",".
"""

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtGui import QGuiApplication, QKeySequence, QShortcut  # noqa: E402

from kkvim_gui.analysis.window import AnalysisWindow  # noqa: E402
from kkvim_gui.main_window.window import MainWindow  # noqa: E402
from kkvim_gui.session import Session  # noqa: E402

CTRL = Qt.KeyboardModifier.ControlModifier
SHIFT = Qt.KeyboardModifier.ShiftModifier

EDIT_KEYS = (
    QKeySequence.StandardKey.Copy,
    QKeySequence.StandardKey.Paste,
    QKeySequence.StandardKey.Cut,
    QKeySequence.StandardKey.Undo,
    QKeySequence.StandardKey.Redo,
    QKeySequence.StandardKey.SelectAll,
)


@pytest.fixture
def window(qtbot, qapp):
    session = Session(settings=None)
    session.apply_theme()
    w = MainWindow(session)
    qtbot.addWidget(w)
    w.set_state("setup")
    w.show()
    qtbot.waitExposed(w)
    return w


@pytest.fixture
def edit(window, qtbot):
    e = window.setup_view.q.edit
    e.setFocus()
    e.clear()
    return e


def test_select_all_copy_cut_paste(edit, qtbot):
    qtbot.keyClicks(edit, "-0,25")
    qtbot.keyClick(edit, Qt.Key.Key_A, CTRL)
    assert edit.selectedText() == "-0,25"
    qtbot.keyClick(edit, Qt.Key.Key_C, CTRL)
    assert QGuiApplication.clipboard().text() == "-0,25"
    qtbot.keyClick(edit, Qt.Key.Key_X, CTRL)
    assert edit.text() == ""
    qtbot.keyClick(edit, Qt.Key.Key_V, CTRL)
    assert edit.text() == "-0,25"


def test_undo_and_both_redo_shortcuts(edit, qtbot):
    qtbot.keyClicks(edit, "1,5e-3")
    qtbot.keyClick(edit, Qt.Key.Key_A, CTRL)
    qtbot.keyClick(edit, Qt.Key.Key_X, CTRL)
    assert edit.text() == ""
    qtbot.keyClick(edit, Qt.Key.Key_Z, CTRL)
    assert edit.text() == "1,5e-3"
    redo_keys = [ks.toString() for ks in QKeySequence.keyBindings(QKeySequence.StandardKey.Redo)]
    if "Ctrl+Y" in redo_keys:
        qtbot.keyClick(edit, Qt.Key.Key_Y, CTRL)
        assert edit.text() == ""
        qtbot.keyClick(edit, Qt.Key.Key_Z, CTRL)
        assert edit.text() == "1,5e-3"
    qtbot.keyClick(edit, Qt.Key.Key_Z, CTRL | SHIFT)
    assert edit.text() == ""


def test_backspace_and_delete(edit, qtbot):
    qtbot.keyClicks(edit, "0,125")
    qtbot.keyClick(edit, Qt.Key.Key_Backspace)
    assert edit.text() == "0,12"
    qtbot.keyClick(edit, Qt.Key.Key_Home)
    qtbot.keyClick(edit, Qt.Key.Key_Delete)
    assert edit.text() == ",12"


@pytest.mark.parametrize("text", ["0.1", "0,1", "-0.25", "-0,25", "1.5e-3", "1,5e-3", "−2.5E+3"])
def test_pasted_numbers_are_kept_exactly(edit, qtbot, text):
    QGuiApplication.clipboard().setText(text)
    qtbot.keyClick(edit, Qt.Key.Key_V, CTRL)
    assert edit.text() == text


@pytest.mark.parametrize("partial", ["-", "0,", "0.", "1e-", "1,5E"])
def test_intermediate_input_is_not_blocked(edit, qtbot, partial):
    qtbot.keyClicks(edit, partial)
    assert edit.text() == partial
    assert edit.validator() is None
    assert edit.inputMask() == ""


def test_application_shortcuts_do_not_take_editing_keys(window, qtbot):
    analysis = AnalysisWindow(window.session)
    qtbot.addWidget(analysis)
    editing = {ks.toString() for key in EDIT_KEYS for ks in QKeySequence.keyBindings(key)}
    editing |= {"Ctrl+C", "Ctrl+V", "Ctrl+X", "Ctrl+Z", "Ctrl+Y", "Ctrl+Shift+Z", "Ctrl+A", "Del", "Backspace"}
    for w in (window, analysis):
        for shortcut in w.findChildren(QShortcut):
            assert shortcut.key().toString() not in editing
