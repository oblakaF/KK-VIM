"""R6-1: the moments of a diagnostic field are never compared with a moment reference (same rule as the main comparison,
``benchmarks.compare``: a diagnostic field, an error or a run that did not end as requested is not compared).

Checked on real runs: the protective stop of QA R6-1 (CCCC, N = 6, a/b = 5, grid 32 x 13, requested s = 2 ->
STOPPED_RANK_GUARD_UNCERTIFIED with the s = 1 field as diagnostics) and an SSSS R3 run stopped at s_max (MAX_HALF_STEPS,
CLI folder, reopened); the GUI package of the protective stop reopened. For each: the diagnostic moments stay available
and marked; the comparison button is disabled with the RU and EN explanation; ``start_reference`` launches no process;
``MomentsRefJob`` and ``compare`` refuse; a late reference answer is dropped; the report holds the diagnostic moment
files and no moment comparison (CSV, component, PDF section). Ordinary research fields keep their comparison and its
report table. The run data (field, history) are unchanged.
"""

import hashlib
import json
import subprocess

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")
pytest.importorskip("pyqtgraph")

from kkvim import cli  # noqa: E402
from kkvim_gui.analysis import report_dialog as rd  # noqa: E402
from kkvim_gui.analysis.window import AnalysisWindow  # noqa: E402
from kkvim_gui.backend import export, moments_view, saved_run  # noqa: E402
from kkvim_gui.session import Session  # noqa: E402
from tests.gui.test_p5_p8_workflow import analysed, main, ssss  # noqa: E402, F401

FAKE_OK = {"state": "ok", "contract": moments_view.D2_CONTRACT}  # a reference answer arriving after the refusal


def protective(model):
    """The failing case of QA R6-1 (scratch/r6_qa_01/qa_gui.py, case PROTECTIVE)."""
    model.set_choice(load_mode="single")
    model.set_choice(supports="CCCC", N=6)
    model.set_choice(cccc_half_steps=2)
    for key, text, unit in (("a", "5.0", "m"), ("b", "1.0", "m"), ("h", "0.01", "m"), ("E", "210", "GPa"), ("nu", "0.3", None),
                            ("q", "1000.0", "Pa"), ("n_x", "32", None), ("n_y", "13", None)):
        model.set_text(key, text)
        if unit:
            model.set_unit(key, unit)
    model.set_choice(with_d2=False)


def digest(outcome):
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(outcome.w).tobytes())
    h.update(json.dumps(outcome.history, sort_keys=True, default=repr).encode("utf-8"))
    return h.hexdigest()


class Recorder:
    """A PDF document stand-in that records what ``report_dialog._moments_section`` writes."""

    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        return lambda *a, **k: self.calls.append((name, a))


def no_process(monkeypatch):
    started = []

    def popen(*a, **k):
        started.append(a)
        raise AssertionError("a moment reference process was started for a diagnostic field")

    monkeypatch.setattr(moments_view.subprocess, "Popen", popen)
    return started


def assert_refused(window, tmp_path, monkeypatch, qtbot):
    """Every path of the moment comparison of a diagnostic field (window: an AnalysisWindow of a diagnostic outcome)."""
    o = window.outcome
    before = digest(o)
    tab = window.moments
    assert not tab.unavailable_reason and tab.data.available and tab.data.diagnostic  # the moments stay available
    assert "ДИАГНОСТИКА" in tab.info.text()
    started = no_process(monkeypatch)
    # the GUI action
    assert not tab.compare_button.isEnabled()
    assert moments_view.REFERENCE_DIAGNOSTIC_RU in tab.ref_info.text()
    assert moments_view.REFERENCE_DIAGNOSTIC_EN in tab.ref_info.text()
    assert moments_view.REFERENCE_DIAGNOSTIC_RU in tab.compare_button.toolTip()
    # direct invocation of the action and of the execution boundary
    tab.start_reference()
    assert tab.job is None and not started and tab.record is None and not tab.refs
    with pytest.raises(moments_view.ReferenceRefused):
        moments_view.MomentsRefJob(o)
    assert not started
    # a late or forged answer is dropped, never compared
    ref = {k: np.zeros_like(o.w) for k in moments_view.MOMENT_KEYS}
    with pytest.raises(moments_view.ReferenceRefused):
        moments_view.compare(tab.data, ref)
    tab.set_reference(dict(FAKE_OK, d2={"Mx": ref["Mx"].tolist(), "My": ref["My"].tolist(), "Mxy": ref["Mxy"].tolist()}))
    assert tab.record is None and not tab.refs and tab.comparison_rows() == [] and tab.compare_model.rowCount() == 0
    assert not tab.ref_choice.isEnabled() and not tab.sec_x.ref_curve.isVisible()
    # the report: the comparison component is unavailable with the reason, even when chosen explicitly
    comps = {c.id: c for c in rd.components(window)}
    assert not comps["moments_comparison_table"].available
    assert moments_view.REFERENCE_DIAGNOSTIC_RU in comps["moments_comparison_table"].reason
    assert comps["moments_table"].available and comps["moments_Mx"].available
    chosen = {c.id for c in comps.values() if c.available} | {"moments_comparison_table"}
    result = rd.build_report(window, tmp_path, chosen, "en", formats=("png",))
    folder = result["folder"]
    files = sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file())
    assert not [f for f in files if "moments_comparison" in f]
    assert "tables/moments_diagnostic.csv" in files and "data/moments_diagnostic.csv" in files and "report.pdf" in files
    m = json.loads(result["manifest"].read_text(encoding="utf-8"))
    assert m["source_data"]["unchanged"]
    assert "moments_comparison_table" in {u["id"] for u in m["report"]["unavailable"]}
    doc = Recorder()
    rd._moments_section(doc, window, o)
    tables = [a[0] for name, a in doc.calls if name == "table"]
    assert all("Эталон, N" not in header for header in tables)  # the summary table only, no comparison table
    paragraphs = " ".join(a[0] for name, a in doc.calls if name == "paragraph")
    assert moments_view.REFERENCE_DIAGNOSTIC_RU in paragraphs and "ДИАГНОСТИКА" in paragraphs
    assert not started and digest(o) == before


def test_protective_stop_of_r6_1_is_not_compared_through_any_path(main, qtbot, tmp_path, monkeypatch):  # noqa: F811
    run, w = analysed(main, qtbot, protective)
    o = run.outcome
    assert o.status == "STOPPED_RANK_GUARD_UNCERTIFIED" and o.field_role == "diagnostic" and o.s == 1
    assert (o.request.N, o.request.n_x, o.request.n_y, o.s_requested) == (6, 32, 13, 2)
    assert_refused(w, tmp_path / "live", monkeypatch, qtbot)
    # the saved GUI package of the protective stop, reopened
    folder = tmp_path / "pkg"
    paths = export.write_result(o, folder, run.meta(0.0), run.detailed)
    assert paths["moments"].name == "moments_diagnostic.csv" and "moments_comparison" not in paths
    loaded = saved_run.load(folder)
    assert (loaded.status, loaded.field_role, loaded.requested_completed) == (o.status, "diagnostic", False)
    window = AnalysisWindow(Session(settings=None), saved_run.summary_rows(loaded), run=saved_run.saved_record(loaded, folder))
    qtbot.addWidget(window)
    assert np.array_equal(loaded.w, o.w)
    assert_refused(window, tmp_path / "reopened", monkeypatch, qtbot)


def test_diagnostic_ssss_cli_folder_is_not_compared(qtbot, tmp_path, monkeypatch):
    out = tmp_path / "cli"
    code = cli.main(["run", "--a", "2.0", "--b", "1.0", "--t", "0.01", "--E", "2.1e11", "--nu", "0.3", "--q", "1000",
                     "--nx", "16", "--ny", "8", "--s-min", "1", "--s-max", "2", "--out", str(out), "--moments"])
    result = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert result["status"] == "MAX_HALF_STEPS" and code != 0  # the CLI convention of a run that did not converge
    assert (out / "moments_diagnostic.csv").is_file()
    loaded = saved_run.load(out)
    assert loaded.request.supports == "SSSS" and loaded.field_role == "diagnostic" and not loaded.requested_completed
    window = AnalysisWindow(Session(settings=None), saved_run.summary_rows(loaded), run=saved_run.saved_record(loaded, out))
    qtbot.addWidget(window)
    assert_refused(window, tmp_path / "report", monkeypatch, qtbot)


def test_refusal_follows_the_rule_of_the_main_comparison(ssss):  # noqa: F811
    import dataclasses

    run, _ = ssss
    o = run.outcome
    assert moments_view.reference_refusal(o) is None
    for change in ({"field_role": "diagnostic", "requested_completed": False}, {"requested_completed": False},
                   {"stop_kind": "error"}):
        other = dataclasses.replace(o, **change)
        assert moments_view.REFERENCE_DIAGNOSTIC_RU in moments_view.reference_refusal(other)
    assert moments_view.reference_refusal(dataclasses.replace(o, origin="demo")) is not None
    from kkvim_gui.backend import benchmarks

    assert moments_view.REFERENCE_DIAGNOSTIC_RU == benchmarks.STATE_TITLE[benchmarks.DIAGNOSTIC]  # the existing text


def test_ordinary_research_field_keeps_its_comparison_in_the_report(ssss, qtbot, tmp_path):  # noqa: F811
    run, w = ssss
    tab = w.moments
    assert tab.compare_button.isEnabled() and moments_view.reference_refusal(run.outcome) is None
    tab.start_reference()
    assert isinstance(tab.job, moments_view.MomentsRefJob) and isinstance(tab.job._process, subprocess.Popen)
    qtbot.waitUntil(lambda: tab.job is None, timeout=120_000)
    assert tab.record["state"] == "ok" and tab.comparison_rows()
    comps = {c.id: c for c in rd.components(w)}
    assert comps["moments_comparison_table"].available
    result = rd.build_report(w, tmp_path, {"moments_comparison_table", "moments_table"}, "en", formats=("png",))
    folder = result["folder"]
    assert (folder / "tables/moments_comparison.csv").is_file()
    doc = Recorder()
    rd._moments_section(doc, w, run.outcome)
    assert any("Эталон, N" in a[0] for name, a in doc.calls if name == "table")
