"""Project .kkv in the main window (GUI_DESIGN.md, 4.1, 5.1, 12.2; P2.3; GUI-PROJECT-KKV-P23-01).

«Сохранить проект» Ctrl+S, «Сохранить как…» Ctrl+Shift+S, «Открыть проект…» Ctrl+O, «Новый расчёт» Ctrl+N; the
project name and «●» in the header; the question before unsaved changes are discarded; a defective file changes
nothing; save and open never start a solver, Navier or D2 process and never change the runs of the session.
"""

import sys
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtGui import QShortcut  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from kkvim_gui.main_window.window import UNTITLED, MainWindow  # noqa: E402
from kkvim_gui.model import project_file  # noqa: E402
from kkvim_gui.model.setup_state import SetupState, to_request, to_series_requests  # noqa: E402
from kkvim_gui.session import Session  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from test_project_file import CONFIGS, DEFECTS  # noqa: E402

CTRL = Qt.KeyboardModifier.ControlModifier


class Forbidden:
    """A process factory that must never be called by saving or opening."""

    def __init__(self, started):
        self.started = started

    def __call__(self, request):
        self.started.append(request)
        raise AssertionError("a computation was started")


@pytest.fixture
def main(qtbot, tmp_path):
    w = MainWindow(Session(settings=None), protocol_dir=tmp_path / "runs")
    qtbot.addWidget(w)
    w.started = []
    w.runs.process_factory = w.runs.navier_factory = w.runs.d2_factory = Forbidden(w.started)
    w.dialogs = []
    w.errors = []
    w.answers = []
    w.ask_open_path = lambda: w.dialogs.append("open") or str(w.open_path)
    w.ask_save_path = lambda suggested: w.dialogs.append(("save", suggested)) or str(w.save_path)
    w.ask_unsaved = lambda: w.dialogs.append("unsaved") or w.answers.pop(0)
    w.ask_replace = lambda path: w.dialogs.append(("replace", path.name)) or False
    w.show_project_error = lambda title, text: w.errors.append((title, text))
    w.set_state("setup")
    return w


def shortcut(window, keys):
    found = [s for s in window.findChildren(QShortcut) if s.key().toString() == keys]
    assert len(found) == 1, keys
    return found[0]


def header_text(main):
    return main.header.project.text()


def type_plate(main, **texts):
    for key, text in texts.items():
        main.setup_view.fields[key].edit.setText(text)


def test_shortcuts_of_the_project_are_installed(main):
    for keys in ("Ctrl+N", "Ctrl+O", "Ctrl+S", "Ctrl+Shift+S"):
        shortcut(main, keys)


def test_save_with_ctrl_s_then_unsaved_mark_and_save_again_without_a_dialog(main, qtbot, tmp_path):
    main.save_path = tmp_path / "plate_ssss"  # typed without the extension: .kkv is added
    assert header_text(main) == UNTITLED and not main.modified
    type_plate(main, a="2,5", n_x="16", n_y="8")
    assert main.modified and header_text(main) == f"{UNTITLED}  ●" and main.windowTitle().endswith("●")
    main.show()
    qtbot.waitExposed(main)
    main.activateWindow()
    qtbot.waitUntil(lambda: QApplication.activeWindow() is main)  # a window shortcut needs the active window
    edit = main.setup_view.fields["a"].edit
    edit.setFocus()
    qtbot.keyClick(edit, Qt.Key.Key_S, CTRL)  # Ctrl+S while typing in a field
    saved = tmp_path / "plate_ssss.kkv"
    assert saved.exists() and main.dialogs == [("save", f"{UNTITLED}.kkv")]  # no folder chosen yet
    assert header_text(main) == "plate_ssss.kkv" and main.windowTitle() == "КК-ВИМ — plate_ssss.kkv"
    assert project_file.read(saved).state == main.setup_model.state
    type_plate(main, q="7,5")
    assert header_text(main) == "plate_ssss.kkv  ●"
    shortcut(main, "Ctrl+S").activated.emit()
    assert len(main.dialogs) == 1 and project_file.read(saved).state.texts["q"] == "7,5"
    assert not main.modified and main.started == []


def test_save_as_writes_a_new_file_and_keeps_the_old_one(main, tmp_path):
    main.save_path = tmp_path / "first.kkv"
    main.setup_view.save_button.click()  # the button of the action bar
    first = (tmp_path / "first.kkv").read_bytes()
    type_plate(main, h="12")
    main.save_path = tmp_path / "second.kkv"
    main.setup_view.save_as_button.click()
    assert main.project_path == tmp_path / "second.kkv" and (tmp_path / "first.kkv").read_bytes() == first
    shortcut(main, "Ctrl+Shift+S").activated.emit()  # asks again even with an open project
    assert [d[0] for d in main.dialogs] == ["save", "save", "save"]


def test_save_as_without_extension_asks_before_replacing_an_existing_file(main, tmp_path):
    (tmp_path / "old.kkv").write_text("old", encoding="utf-8")
    main.save_path = tmp_path / "old"
    assert not main.save_project_as()
    assert (tmp_path / "old.kkv").read_text(encoding="utf-8") == "old" and ("replace", "old.kkv") in main.dialogs


def test_open_restores_the_saved_setup_requests_and_display_in_the_form(main, tmp_path):
    for name in ("r4_n3_research_fixed_dense", "cccc_n3_s2_guarded", "ssss_n1_units", "ssss_n5_research",
                 "series_ssss_n1"):
        state = CONFIGS[name]
        path = tmp_path / f"{name}.kkv"
        project_file.write(path, state)
        main.open_path = path
        assert main.open_project(), name
        st = main.setup_model.state
        assert st == state and to_request(st) == to_request(state), name
        assert to_series_requests(st) == to_series_requests(state), name
        view = main.setup_view
        assert all(view.fields[k].text() == state.texts[k] for k in view.fields), name
        assert all(view.fields[k].unit() == state.units[k] for k in state.units), name
        assert view.N.currentData() == state.N and view.supports.current() == state.supports, name
        assert view.research.isChecked() == state.research and view.ssss_research.isChecked() == state.ssss_research
        assert view.d2.isChecked() == state.with_d2, name
        assert {k for k, box in view.display_boxes.items() if box.isChecked()} <= set(state.display), name
        assert header_text(main) == path.name and not main.modified and main.state == "setup"
    assert main.started == [] and main.errors == []
    assert "Расчёт не запускался" in main.status_label.text()


def test_open_from_the_start_screen_and_ctrl_o(main, tmp_path):
    path = tmp_path / "cccc.kkv"
    project_file.write(path, CONFIGS["cccc_n1_s5_no_d2"])
    main.open_path = path
    main.set_state("start")
    main.start_view.open_button.click()
    assert main.state == "setup" and main.setup_model.state == CONFIGS["cccc_n1_s5_no_d2"]
    main.setup_model.load(SetupState())
    main.saved_state = main.setup_model.state
    shortcut(main, "Ctrl+O").activated.emit()
    assert main.setup_model.state == CONFIGS["cccc_n1_s5_no_d2"] and main.dialogs == ["open", "open"]


def test_a_defective_file_changes_nothing(main, tmp_path):
    good = tmp_path / "good.kkv"
    project_file.write(good, CONFIGS["r4_n2_normative"])
    main.open_path = good
    main.open_project()
    type_plate(main, q="9")
    before, path_before = main.setup_model.state, main.project_path
    main.answers = ["discard"]
    for name, (text, _) in sorted(DEFECTS.items())[:6] + [("schema_2", DEFECTS["schema_2"]),
                                                         ("research", DEFECTS["research_r4_for_n1"])]:
        bad = tmp_path / f"{name}.kkv"
        bad.write_text(text, encoding="utf-8")
        main.open_path = bad
        main.answers = ["discard"]
        assert not main.open_project(), name
        assert main.setup_model.state == before and main.project_path == path_before and main.modified, name
    assert len(main.errors) == 8 and all(title == "Проект не открыт" and "Текущие параметры не изменены" in text
                                         for title, text in main.errors)
    assert main.started == []


def test_unsaved_changes_cancel_save_or_discard_before_open(main, tmp_path):
    other = tmp_path / "other.kkv"
    project_file.write(other, CONFIGS["cccc_n1_s5_no_d2"])
    main.open_path = other
    type_plate(main, a="3")
    edited_state = main.setup_model.state
    main.answers = ["cancel"]
    assert not main.open_project()
    assert main.dialogs == ["unsaved"] and main.setup_model.state == edited_state  # no file dialog after «Отмена»
    main.answers = ["save"]
    main.save_path = tmp_path / "mine.kkv"
    assert main.open_project()
    assert project_file.read(tmp_path / "mine.kkv").state == edited_state  # saved first, then opened
    assert main.setup_model.state == CONFIGS["cccc_n1_s5_no_d2"] and main.project_path == other


def test_a_cancelled_save_dialog_keeps_the_changes_and_the_setup(main, tmp_path):
    type_plate(main, a="3")
    edited_state = main.setup_model.state
    main.ask_save_path = lambda suggested: ""  # «Отмена» in the file dialog
    main.answers = ["save"]
    main.open_path = tmp_path / "x.kkv"
    assert not main.open_project() and main.setup_model.state == edited_state and main.modified
    assert not main.save_project() and list(tmp_path.iterdir()) == []


def test_a_failed_save_is_reported_and_keeps_the_unsaved_mark(main, tmp_path, monkeypatch):
    main.save_path = tmp_path / "p.kkv"
    main.save_project()
    before = (tmp_path / "p.kkv").read_bytes()
    type_plate(main, a="4")
    monkeypatch.setattr(project_file.os, "replace", lambda *a: (_ for _ in ()).throw(PermissionError(13, "Отказано")))
    assert not main.save_project()
    assert main.modified and header_text(main).endswith("●") and (tmp_path / "p.kkv").read_bytes() == before
    assert main.errors[-1][0] == "Проект не сохранён" and "Прежний файл проекта не изменён" in main.errors[-1][1]


def test_new_calculation_asks_and_returns_to_the_initial_form(main, tmp_path):
    type_plate(main, a="3")
    edited_state = main.setup_model.state
    main.answers = ["cancel"]
    shortcut(main, "Ctrl+N").activated.emit()
    assert main.setup_model.state == edited_state
    main.answers = ["discard"]
    main.set_state("start")
    main.start_view.new_button.click()
    assert main.setup_model.state == SetupState() and main.project_path is None and not main.modified
    assert main.state == "setup" and header_text(main) == UNTITLED
    main.dialogs.clear()
    shortcut(main, "Ctrl+N").activated.emit()  # the initial form: nothing to ask
    assert main.dialogs == [] and main.state == "setup"


def test_closing_with_unsaved_changes_asks_in_the_application(main):
    main.confirm_close = True
    main.show()
    type_plate(main, a="3")
    main.answers = ["cancel"]
    assert not main.close() and main.isVisible()
    main.answers = ["discard"]
    assert main.close()


def test_closing_without_changes_does_not_ask(main):
    main.confirm_close = True
    main.show()
    assert main.close() and main.dialogs == []


def test_existing_runs_keep_their_snapshots_and_statuses(qtbot, tmp_path):
    from kkvim_gui.backend import kkvim_adapter
    from kkvim_gui.backend.process import OK

    class Inline:
        def __init__(self, request):
            self.request, self.message = request, None

        def poll(self):
            if self.message is None:
                self.message = (OK, kkvim_adapter.execute(self.request))
            return self.message

        def terminate(self, timeout=2.0):
            return True

    w = MainWindow(Session(settings=None), protocol_dir=tmp_path / "runs", process_factory=Inline)
    qtbot.addWidget(w)
    w.runs.navier_factory = None
    w.errors = []
    w.show_project_error = lambda title, text: w.errors.append(title)
    w.set_state("setup")
    w.setup_model.load(CONFIGS["cccc_n3_s2_guarded"])
    run = w._start_run()
    qtbot.waitUntil(lambda: w.runs.active is None and not w.runs.references_pending, timeout=120000)
    snapshot, status, request, count = run.snapshot, run.outcome.status, run.request, len(w.runs.runs)
    started = []
    w.runs.process_factory = w.runs.navier_factory = w.runs.d2_factory = Forbidden(started)
    w.ask_unsaved = lambda: "discard"
    path = tmp_path / "n1.kkv"
    project_file.write(path, CONFIGS["ssss_n1_units"])
    assert w.open_project(path)
    assert w.new_project()
    assert w.open_project(path)
    assert run.snapshot is snapshot and run.snapshot == CONFIGS["cccc_n3_s2_guarded"]
    assert run.outcome.status == status and run.request == request and len(w.runs.runs) == count
    assert started == [] and w.setup_model.state == CONFIGS["ssss_n1_units"]


def test_open_and_new_are_refused_while_a_run_is_active(main, tmp_path):
    class Hanging:
        def __init__(self, request):
            pass

        def poll(self):
            return None

        def terminate(self, timeout=2.0):
            return True

    path = tmp_path / "p.kkv"
    project_file.write(path, CONFIGS["cccc_n1_s5_no_d2"])
    main.runs.process_factory = Hanging
    main.runs.navier_factory = None
    type_plate(main, a="2", b="1", h="10", E="210", nu="0,3", q="5", n_x="16", n_y="8")
    run = main._start_run()
    assert run is not None and main.runs.active is run
    running_state = main.setup_model.state
    main.open_path = path
    assert not main.open_project() and main.errors[-1][0] == "Проект не открыт"
    assert not main.new_project()
    assert main.setup_model.state == running_state and run.snapshot == running_state
    main.save_path = tmp_path / "during.kkv"
    assert main.save_project() and project_file.read(tmp_path / "during.kkv").state == running_state
    main.runs.stop(run.run_id)
