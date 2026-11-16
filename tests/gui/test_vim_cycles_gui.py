"""VIM-1, VIM-2, VIM-3 in the windows (GUI-VIM-CYCLES-123-01).

- the «Циклы VIM-1–3» tab of a real SSSS N = 1 run: the graph and the table use the same records (s = 2, 4, 6 of the
  run's history), the Navier line is the run's reference, every state and the reference can be hidden;
- a reference that arrives after the window opened rebuilds the tab; closing the window deletes its Qt objects;
- the load series: VIM-1 … VIM-3 curves from every point's own run, selectable with the other curves;
- the calculation package of a series run: vim_cycles.csv/png, load_series_vim_cycles.csv, load_series.png with SHA256.
"""

import hashlib
import json

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")
pytest.importorskip("pyqtgraph")

import shiboken6  # noqa: E402
from PySide6.QtCore import QCoreApplication, QEvent  # noqa: E402

from kkvim_gui.analysis.cycles_tab import TAB_TITLE, CyclesTab  # noqa: E402
from kkvim_gui.analysis.series import SeriesFigure  # noqa: E402
from kkvim_gui.analysis.window import SOLVER_TABS, AnalysisWindow  # noqa: E402
from kkvim_gui.backend import kkvim_adapter, navier_job, vim_cycles  # noqa: E402
from kkvim_gui.backend.process import INPUT, OK  # noqa: E402
from kkvim_gui.main_window.window import MainWindow  # noqa: E402
from kkvim_gui.model.series import series_data  # noqa: E402
from kkvim_gui.session import Session  # noqa: E402

MM = 1000.0


class InlineProcess:
    def __init__(self, request):
        self.request, self.message = request, None

    def poll(self):
        if self.message is None:
            try:
                self.message = (OK, kkvim_adapter.execute(self.request))
            except kkvim_adapter.AdapterInputError as exc:
                self.message = (INPUT, str(exc))
        return self.message

    def terminate(self, timeout=2.0):
        return True


class InlineNavier:
    """The Navier worker contract evaluated in the test process (the real worker: tests/test_gui_navier_series.py)."""

    def __init__(self, inputs):
        self.inputs = inputs

    def poll(self):
        from kkvim_navier_worker import compute

        return navier_job.parse_output(json.dumps(compute(self.inputs)), self.inputs)

    def terminate(self, timeout=2.0):
        return True


@pytest.fixture
def main(qtbot, tmp_path):
    w = MainWindow(Session(settings=None), protocol_dir=tmp_path / "runs", process_factory=InlineProcess)
    w.runs.navier_factory = InlineNavier
    qtbot.addWidget(w)
    w.set_state("setup")
    return w


def finish(main, qtbot):
    qtbot.waitUntil(lambda: main.runs.active is None and not main.runs.references_pending
                    and not any(s.active for s in main.runs.series), timeout=120000)


def single(main, qtbot, n_x="80", n_y="40"):
    m = main.setup_model
    for key, text in (("a", "2"), ("b", "1"), ("h", "10"), ("E", "210"), ("nu", "0,3"), ("q", "5"), ("n_x", n_x),
                      ("n_y", n_y)):
        m.set_text(key, text)
    run = main._start_run()
    finish(main, qtbot)
    return run


def test_cycles_tab_of_a_real_run_draws_the_table_records(main, qtbot):
    run = single(main, qtbot)
    w = AnalysisWindow(main.session, main.run_rows(), run=run)
    qtbot.addWidget(w)
    titles = [w.tabs.tabText(i) for i in range(w.tabs.count())]
    assert titles == [*SOLVER_TABS, TAB_TITLE]
    tab = w.cycles
    t = tab.table
    assert t == vim_cycles.cycle_table(run.outcome, navier=run.navier)
    assert (t.solver_status, t.s, t.k) == ("CONVERGED", 8, 4) and [c.s for c in t.cycles] == [2, 4, 6]
    pts, ref = tab.figure.drawn()
    assert pts == {c.k: (c.k, c.w_m * MM) for c in t.cycles} and ref == run.navier["w_m"] * MM
    assert tab.figure.points[2].getData()[1][0] == t.cycles[1].w_m * MM
    rows = [[tab.model.index(r, c).data(256) for c in range(tab.model.columnCount())] for r in range(tab.model.rowCount())]
    assert [r[0] for r in rows] == ["s = 1 (k = 0)", "VIM-1", "VIM-2", "VIM-3"]
    for r, c in zip(rows, (t.partial, *t.cycles), strict=True):
        assert (r[1], r[2], r[5], r[8]) == (c.k, c.s, c.w_m, c.delta_w_m * MM) and r[10] is c.on_graph
    tab.toggles[2].setChecked(False)
    tab.toggles["ref"].setChecked(False)
    pts, ref = tab.figure.drawn()
    assert sorted(pts) == [1, 3] and ref is None and not tab.figure.points[2].isVisible()
    assert not tab.figure.reference.isVisible()


def test_late_reference_rebuilds_the_tab_and_closing_deletes_it(main, qtbot):
    run = single(main, qtbot, "16", "8")
    final = run.navier
    run.navier = navier_job.pending_record()
    w = AnalysisWindow(main.session, main.run_rows(), run=run)
    qtbot.addWidget(w)
    first = w.cycles
    assert first.table.ref_w_m is None and first.figure.drawn()[1] is None
    run.navier = final
    qtbot.waitUntil(lambda: w.cycles is not first, timeout=5000)
    assert w.cycles.table.ref_w_m == final["w_m"] and w.tabs.tabText(w.tabs.indexOf(w.cycles)) == TAB_TITLE
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert not shiboken6.isValid(first)  # the replaced tab is deleted, not left alive
    second = w.cycles
    w.close()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert not shiboken6.isValid(second)


def test_series_has_three_cycle_curves_from_every_point(main, qtbot, tmp_path):
    m = main.setup_model
    for key, text in (("a", "2"), ("b", "1"), ("h", "10"), ("E", "210"), ("nu", "0,3"), ("n_x", "16"), ("n_y", "8")):
        m.set_text(key, text)
    m.set_choice(load_mode="series")
    m.set_text("q_series", "1; 5; 2,5")
    series = main._start_series(m.state)
    finish(main, qtbot)
    data = series_data(series, main.runs)
    assert data.cycles_supported and all(p.cycles.solver_status == "CONVERGED" for p in data.points)
    figure = SeriesFigure(main.session, data)
    qtbot.addWidget(figure)
    for k in (1, 2, 3):
        xs, ys = figure.cycle_curve(k)
        pts = sorted(data.points, key=lambda p: p.q)
        assert xs == [p.q / 1000.0 for p in pts] and ys == [p.cycle(k).w_m * MM for p in pts]
        assert all(p.cycle(k).w_m == vim_cycles.cycle_table(main.runs.get(p.run_id).outcome).cycles[k - 1].w_m
                   for p in pts)  # each point from its own run
        assert figure.cycle_items[k].isVisible()
    figure.set_shown(2, False)
    figure.set_shown("kkvim", False)
    assert not figure.cycle_items[2].isVisible() and not figure.kkvim.isVisible() and figure.navier.isVisible()
    figure.set_quantity("maxabs")
    assert figure.cycle_curve(1) == ([], [])  # recorded at the centre only
    run = main.runs.get(series.run_ids[1])
    w = AnalysisWindow(main.session, main.run_rows(), run=run, series=data)
    qtbot.addWidget(w)
    assert [w.tabs.tabText(i) for i in range(w.tabs.count())][-2:] == [TAB_TITLE, "Нагрузка — прогиб"]
    assert set(w.series_tab.toggles) == {"kkvim", 1, 2, 3, "navier"}
    paths = w.save_package(tmp_path)
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    sha = manifest["vim_cycles"]["files_sha256"]
    for key in ("vim_cycles", "vim_cycles_png", "load_series_vim_cycles", "load_series_png"):
        assert sha[paths[key].name] == hashlib.sha256(paths[key].read_bytes()).hexdigest(), key
    html = paths["report"].read_text(encoding="utf-8")
    assert "vim_cycles.png" in html and "load_series.png" in html and "6г. Нагрузка — прогиб" in html
    assert manifest["vim_cycles"]["series_id"] == series.series_id


def test_cycles_tab_states_an_unsupported_route(qapp):
    from PySide6.QtWidgets import QLabel

    from kkvim_gui.model.setup_state import SetupState, to_request

    o = kkvim_adapter.execute(to_request(SetupState().with_choice(N=2).with_text("n_x", "16").with_text("n_y", "8")))
    tab = CyclesTab(Session(settings=None), vim_cycles.cycle_table(o))
    texts = " ".join(lb.text() for lb in tab.findChildren(QLabel))
    assert "max|w|" in texts and not tab.figure.drawn()[0]
    assert not any(tab.toggles[k].isEnabled() for k in (1, 2, 3))
    tab.deleteLater()
