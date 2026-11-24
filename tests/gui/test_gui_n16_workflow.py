"""GUI-CORE-N16-P1 in the real windows: SSSS and CCCC, N = 1-6, from the form through the computing process to the run
view, the analysis window and the saved package.

The solver runs in its own process (``backend.process.SolverProcess``, the application default). The cases are the small
control cases of D-063, D-065 and D-068 and the studied cell of SSSS N = 4-6. Checked: the route chosen by the form, the
status shown everywhere, the field role (a research field is never shown as a qualified result, a diagnostic field never
as a result), the file of the field, TEST DATA of the demonstration and that nothing is written over an earlier package.
"""

import json
import os

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")
pytest.importorskip("pyqtgraph")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QLabel  # noqa: E402

import kkvim.cccc_n23 as n23  # noqa: E402
import kkvim.cccc_n46 as n46  # noqa: E402
from kkvim_gui.app import GuiApplication  # noqa: E402
from kkvim_gui.backend.kkvim_adapter import RESEARCH_MARK  # noqa: E402
from kkvim_gui.backend.process import SolverProcess  # noqa: E402
from kkvim_gui.main_window.window import MainWindow  # noqa: E402
from kkvim_gui.model.setup_state import N_VALUES  # noqa: E402
from kkvim_gui.session import Session  # noqa: E402


@pytest.fixture
def gui(qtbot, qapp, tmp_path):
    session = Session(settings=None)
    session.apply_theme()
    g = GuiApplication(session, protocol_dir=tmp_path / "runs")
    qtbot.addWidget(g.main_window)
    g.main_window.show()
    g.main_window.set_state("setup")
    yield g
    for w in list(g.analysis_windows):
        w.close()
    g.main_window.runs.shutdown()
    g.help.close()


def fill(view, supports, N, a, b, n_x, n_y, q=1000.0, s=None, ssss_research=False):
    """Type a case into the form as a user does: supports, N, values in m and Pa, the half-step choice."""
    view.supports.set_current(supports)
    view.N.setCurrentIndex(N_VALUES.index(N))
    for key, value, unit in (("a", a, "m"), ("b", b, "m"), ("h", 0.01, "m"), ("E", 2.1e11, "Pa"), ("nu", 0.3, None),
                             ("q", q, "Pa"), ("n_x", n_x, None), ("n_y", n_y, None)):
        field = getattr(view, key)
        if unit is not None:
            field.unit_combo.setCurrentText(unit)
        field.edit.setText(repr(value) if isinstance(value, float) else str(value))
    if supports == "CCCC" and s is not None:
        view.cccc_half_steps.setCurrentIndex(view.cccc_half_steps.findData(s))
    if supports == "SSSS" and N >= 4:
        view.ssss_research.setChecked(ssss_research)
        if s is not None:
            view.half_steps.edit.setText(str(s))


def start_and_wait(qtbot, main, timeout=180_000):
    qtbot.mouseClick(main.setup_view.run_button, Qt.MouseButton.LeftButton)
    run = main.runs.active
    assert run is not None, main.setup_view.check_text.text()
    qtbot.waitUntil(lambda: main.runs.active is None, timeout=timeout)
    return run


def texts(widget):
    return [w.text() for w in widget.findChildren(QLabel) if not w.isHidden()]


def test_real_gui_smoke_cccc_s2_in_a_child_process_shows_the_field_and_saves_a_package(qtbot, gui, tmp_path):
    main = gui.main_window
    fill(main.setup_view, "CCCC", 2, 1.0, 0.6, 10, 9, s=2)
    assert main.setup_model.state.route == "cccc_n23" and main.setup_model.state.cccc_half_steps == 2
    qtbot.mouseClick(main.setup_view.run_button, Qt.MouseButton.LeftButton)
    run = main.runs.active
    assert isinstance(run._process, SolverProcess) and run._process.pid not in (None, os.getpid())  # a child process
    qtbot.waitUntil(lambda: main.runs.active is None, timeout=180_000)
    o = run.outcome
    direct = n23.run_cccc_n23_physical(1.0, 0.6, 0.01, 2.1e11, 0.3, 1000.0, 10, 9, 2, 2)
    assert run.status == "finished" and run.has_results and not run.has_diagnostic_field
    assert o.status == n23.COMPLETED_FIXED_HALF_STEPS and o.s == 2 and o.field_role == "research"
    assert np.array_equal(o.w, direct.field())  # the child process returned the solver array unchanged
    rv = main.run_view
    assert RESEARCH_MARK in rv.status_accept.text() and rv.details_tag.text() == RESEARCH_MARK
    assert rv.limits_note.isVisibleTo(rv) and "Ограничения маршрута" in rv.limits_note.text.text()
    assert rv.analysis_button.text() == "Открыть анализ результатов" and rv.analysis_button.isEnabled()
    assert not any("CONVERGED" in t for t in texts(rv))
    qtbot.mouseClick(rv.analysis_button, Qt.MouseButton.LeftButton)
    window = gui.analysis_windows[-1]
    qtbot.addWidget(window)
    assert window.outcome is o and "RESEARCH" in window.header.badge.text()
    np.testing.assert_array_equal(window.fields.figure.image.image, o.w * 1000.0)  # the displayed field, in mm
    assert "RESEARCH_ONLY" in window.fields.figure.mark.text() and window.status_run.property("status") == "success"
    paths = main.save_package(run.run_id, tmp_path / "packages")
    names = sorted(p.name for p in paths["manifest"].parent.iterdir())
    assert "field_research.csv" in names and "report.html" in names and "field.csv" not in names
    assert any(n.endswith(".png") for n in names)
    data = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert data["statuses"]["field_role"] == "research" and data["statuses"]["field_file"] == "field_research.csv"
    assert data["statuses"]["solver_status"] == o.status and data["results"]["half_steps"] == {"completed": 2, "requested": 2}
    assert data["model"]["supports"] == "CCCC" and data["model"]["decisions"] == ["D-063", "D-065"]
    protocol = run.protocol_path
    assert protocol is not None and (protocol / "manifest.json").exists() and not list(protocol.glob("field*"))


def test_a_protective_rn5_refusal_is_shown_only_as_diagnostics(qtbot, gui, tmp_path):
    """N = 6, a = 5, b = 1, 32 x 13, s = 2: the solver's RN-5 is not certified; the field of s = 1 is diagnostic only."""
    main = gui.main_window
    fill(main.setup_view, "CCCC", 6, 5.0, 1.0, 32, 13, s=2)
    run = start_and_wait(qtbot, main)
    o = run.outcome
    assert o.status == n46.STOPPED_RANK_GUARD_UNCERTIFIED and o.s == 1 and o.s_requested == 2
    assert run.status == "finished" and not run.has_results and run.has_diagnostic_field and o.diagnostic_only
    rv = main.run_view
    assert rv.analysis_button.text().startswith("Открыть диагностику") and rv.details_tag.text().startswith("ДИАГНОСТИКА")
    assert rv.status_stop.property("status") != "success" and "Защитная остановка" in rv.status_stop.text()
    shown = texts(rv)
    assert not any("CONVERGED" in t for t in shown) and any("только как диагностика" in t for t in shown)
    rv.analysis_button.click()
    window = gui.analysis_windows[-1]
    qtbot.addWidget(window)
    assert "ДИАГНОСТИКА" in window.header.badge.text() and window.status_run.property("status") == "warning"
    assert "ДИАГНОСТИКА" in window.fields.figure.mark.text() or "DIAGNOSTIC" in window.fields.figure.mark.text()
    paths = main.save_package(run.run_id, tmp_path / "packages")
    names = sorted(p.name for p in paths["manifest"].parent.iterdir())
    assert "field_diagnostic.csv" in names and not {"field.csv", "field_research.csv"} & set(names)
    data = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert data["statuses"]["requested_run_completed"] is False and data["statuses"]["field_role"] == "diagnostic"
    assert "CONVERGED" not in json.dumps(data) and "CONVERGED" not in paths["report"].read_text(encoding="utf-8")


def test_ssss_n4_runs_only_with_the_explicit_research_choice(qtbot, gui):
    main = gui.main_window
    fill(main.setup_view, "SSSS", 4, 1.0, 1.0, 40, 40, s=4, ssss_research=False)
    assert main.setup_view.try_run() is False and main.runs.runs == []
    main.setup_view.ssss_research.setChecked(True)
    run = start_and_wait(qtbot, main)
    o = run.outcome
    assert o.route == "ssss_n46" and o.s == 4 and o.field_role == "research" and o.mark == RESEARCH_MARK
    assert run.has_results and RESEARCH_MARK in main.run_view.status_accept.text()
    main.set_state("setup")
    fill(main.setup_view, "SSSS", 5, 1.0, 1.0, 41, 41, s=4, ssss_research=True)  # outside the studied grids
    assert main.setup_view.try_run() is True and len(main.runs.runs) == 1               # refused before a run
    assert "NOT_EXECUTED_OUTSIDE_DOMAIN" in main.setup_view.check_text.text()


def test_negative_q_is_shown_with_its_sign_and_the_same_normalized_field(qtbot, gui):
    main = gui.main_window
    fill(main.setup_view, "CCCC", 2, 1.0, 0.6, 10, 9, q=-1000.0, s=2)
    o = start_and_wait(qtbot, main).outcome
    direct = n23.run_cccc_n23_physical(1.0, 0.6, 0.01, 2.1e11, 0.3, 1000.0, 10, 9, 2, 2)
    assert np.array_equal(o.w, -direct.field()) and np.array_equal(o.w_bar, direct.dimensionless_field())
    main.run_view.analysis_button.click()
    window = gui.analysis_windows[-1]
    qtbot.addWidget(window)
    np.testing.assert_array_equal(window.fields.figure.image.image, o.w * 1000.0)
    assert o.extremum()["w"] < 0 < o.extremum()["w_bar"]


def test_rejected_input_and_a_second_save_never_overwrite_a_package(qtbot, gui, tmp_path):
    main = gui.main_window
    fill(main.setup_view, "CCCC", 1, 1.0, 1.0, 16, 16, s=4)
    run = start_and_wait(qtbot, main)
    paths = main.save_package(run.run_id, tmp_path / "packages")
    folder = paths["manifest"].parent
    before = {p.name: p.read_bytes() for p in folder.iterdir()}
    main.set_state("setup")
    main.setup_view.n_x.edit.setText("5")  # below the CCCC domain 7...1280: refused by the solver's own check
    main.setup_view.try_run()
    assert len(main.runs.runs) == 1 and "NOT_EXECUTED_OUTSIDE_DOMAIN" in main.setup_view.check_text.text()
    with pytest.raises(FileExistsError):
        main.save_package(run.run_id, tmp_path / "packages")
    assert {p.name: p.read_bytes() for p in folder.iterdir()} == before


def test_a_demonstration_with_cccc_selected_is_test_data(qtbot, qapp):
    window = MainWindow(Session(settings=None), demo=True)
    qtbot.addWidget(window)
    window.setup_model.set_choice(supports="CCCC", N=3)
    run = window._start_run()
    assert run.is_demo and window.run_view._tag(run) == "TEST DATA" and "TEST DATA" in window.run_view.summary_text.text()
    window.runs.stop(run.run_id)
