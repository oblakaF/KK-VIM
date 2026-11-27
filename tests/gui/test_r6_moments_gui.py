"""R6 research workflow in the GUI: «Моменты и напряжения», exports, report, reopening a saved calculation.

Checked: the values shown are those of kkvim.moments of the stored field (the GUI has no formula; display = unit scale);
stresses at the surfaces and at a chosen z; the raw values of the sections are the SI values; the publication figure
carries the R6 mark; the report holds the moment figures, tables and the PDF section while the run data stay unchanged;
the export package holds the moment files of the field role (diagnostic for a diagnostic field) and TEST DATA for demo
data; a saved package and a CLI folder reopen with their statuses unchanged; the independent references run in their
own processes (Navier and Levy for SSSS; D2 and Ritz for CCCC) and give observations only.
"""

import json

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")
pytest.importorskip("pyqtgraph")

from kkvim import cli  # noqa: E402
from kkvim import moments as km  # noqa: E402
from kkvim_gui.analysis import report_dialog as rd  # noqa: E402
from kkvim_gui.analysis.moments_tab import MomentsTab  # noqa: E402
from kkvim_gui.analysis.window import AnalysisWindow  # noqa: E402
from kkvim_gui.backend import export, moments_view, report_builder, saved_run  # noqa: E402
from kkvim_gui.session import Session  # noqa: E402
from tests.gui.test_multimethod_d2 import cccc  # noqa: E402
from tests.gui.test_p5_p8_workflow import analysed, demo_outcome, main, ssss  # noqa: E402, F401


def kk(outcome):
    r = outcome.request
    return km.plate_moments(outcome.w, r.a, r.b, outcome.D, r.nu, km.BC_OF_SUPPORTS[r.supports])


def choose(tab, key, surface="bottom", z_mm=None):
    tab.quantity.setCurrentIndex(tab.quantity.findData(key))
    tab.surface.setCurrentIndex(tab.surface.findData(surface))
    if z_mm is not None:
        tab.z.setValue(z_mm)


def wait_reference(tab, qtbot):
    tab.start_reference()
    qtbot.waitUntil(lambda: tab.job is None, timeout=120_000)
    return tab.record


def test_values_are_those_of_kkvim_moments_and_the_field_is_unchanged(ssss, qtbot):  # noqa: F811
    run, w = ssss
    o = run.outcome
    before = o.w.copy()
    tab = w.moments
    assert isinstance(tab, MomentsTab) and not tab.unavailable_reason
    w.tabs.setCurrentWidget(tab)
    M = kk(o)
    choose(tab, "Mx")
    assert np.array_equal(tab.field.values(), M.Mx)  # N·m/m = N: scale 1
    choose(tab, "Mxy")
    assert np.array_equal(tab.field.values(), M.Mxy)
    choose(tab, "sigma_x", "bottom")
    assert np.array_equal(tab.field.values(), km.stresses_at(M, o.request.t, o.request.t / 2)[0] * 1e-6)
    choose(tab, "tau_xy", "top")
    assert np.array_equal(tab.field.values(), km.stresses_at(M, o.request.t, -o.request.t / 2)[2] * 1e-6)
    choose(tab, "sigma_y", "z", z_mm=2.0)
    assert np.allclose(tab.field.values(), km.stresses_at(M, o.request.t, 0.002)[1] * 1e-6, rtol=1e-15, atol=0)
    choose(tab, "w")
    assert np.array_equal(tab.field.values(), o.w * 1e3)
    assert np.array_equal(o.w, before) and not o.w.flags.writeable


def test_section_raw_values_are_si_and_the_figure_has_the_r6_mark(ssss, qtbot):  # noqa: F811
    run, w = ssss
    tab = w.moments
    choose(tab, "My")
    M = kk(run.outcome)
    rows = tab.section_rows(tab.sec_x, None)
    j = tab.at_y.index()
    assert [r[4] for r in rows] == [float(v) for v in M.My[:, j]]
    spec = tab.publisher.spec("en")
    assert any("R6" in m for m in spec.marks) and "M_y" in spec.title and spec.meta["quantity"] == "My"
    assert tab.publisher.info.figure_id == "field_My"
    chart = tab.tools_x.spec("ru")
    assert any("R6" in m for m in chart.marks)
    summary = tab.data.summary("My")
    assert summary["max"]["value"] == float(np.max(M.My)) and summary["centre"]["value"] == float(M.My[km.centre_index(M.My.shape)])
    assert "RESEARCH / NOT_QUALIFIED" in tab.info.text() and "CONVERGED" in tab.info.text()


def test_report_holds_the_moments_and_never_changes_the_run(ssss, tmp_path):  # noqa: F811
    run, w = ssss
    o = run.outcome
    before = report_builder.source_digest(o)
    comps = {c.id: c for c in rd.components(w)}
    assert comps["moments_Mx"].available and comps["moments_table"].available
    assert not comps["moments_comparison_table"].available and "не запрошен" in comps["moments_comparison_table"].reason
    chosen = {"moments_Mx", "moments_sigma_x", "moments_section_x", "moments_table", "field"}
    result = rd.build_report(w, tmp_path, chosen, "en", formats=("png",))
    folder = result["folder"]
    files = sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file())
    assert "figures/moments_Mx_en.png" in files and "figures/moments_sigma_x_en.png" in files
    assert "tables/moments_research.csv" in files and "tables/moments.json" in files
    assert "data/moments_research.csv" in files and "data/moments.json" in files and "report.pdf" in files
    m = json.loads(result["manifest"].read_text(encoding="utf-8"))
    assert m["source_data"]["unchanged"] and report_builder.source_digest(o) == before
    unavailable = {u["id"] for u in m["report"]["unavailable"]}
    assert "moments_comparison_table" in unavailable
    doc = json.loads((folder / "tables/moments.json").read_text(encoding="utf-8"))
    assert doc["status"] == "RESEARCH / NOT_QUALIFIED" and doc["source_field"]["solver_status"] == "CONVERGED"


def test_ssss_reference_navier_and_levy_in_their_own_process(ssss, qtbot):  # noqa: F811
    run, w = ssss
    tab = w.moments
    record = wait_reference(tab, qtbot)
    assert record["state"] == "ok" and set(tab.refs) == {"navier"}
    choose(tab, "Mx")
    rows = tab.comparison_rows()
    centre = next(r for r in rows if r[0] == "Mx" and r[1] == "центр")
    assert centre[2] == float(kk(run.outcome).Mx[km.centre_index(run.outcome.w.shape)]) and centre[4] == centre[2] - centre[3]
    levy = [r for r in rows if r[1].startswith("Навье − Леви")]
    assert levy and all(abs(r[4]) <= 1e-6 * max(1.0, abs(r[2])) for r in levy)  # two series, same plate
    assert tab.sec_x.ref_curve.isVisible()
    comps = {c.id: c for c in rd.components(w)}
    assert comps["moments_comparison_table"].available


def test_cccc_reference_d2_and_ritz_and_the_clamped_edges(main, qtbot):  # noqa: F811
    run, w = analysed(main, qtbot, lambda m: cccc(m, n="16", N=1, s=8))
    tab = w.moments
    M = kk(run.outcome)
    assert np.all(M.Mxy[0] == 0.0) and np.all(M.Mxy[:, -1] == 0.0)
    record = wait_reference(tab, qtbot)
    assert record["state"] == "ok" and set(tab.refs) == {"d2", "ritz"}
    assert "согласованность" in tab.refs["d2"]["title"] and "NOT_QUALIFIED" in tab.refs["ritz"]["status"]
    tab.ref_choice.setCurrentIndex(tab.ref_choice.findData("ritz"))
    assert any(r[1] == "центр" for r in tab.comparison_rows())


def test_diagnostic_field_gives_diagnostic_moments(main, qtbot, tmp_path):  # noqa: F811
    run, w = analysed(main, qtbot, lambda m: cccc(m, n="40", N=2, s=2))
    o = run.outcome
    assert o.diagnostic_only
    tab = w.moments
    assert not tab.unavailable_reason and "ДИАГНОСТИКА" in tab.info.text()
    paths = export.write_result(o, tmp_path / "pkg")
    assert paths["moments"].name == "moments_diagnostic.csv"
    m = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert m["moments"]["role"] == "diagnostic" and m["moments"]["available"]


def test_demo_moments_are_marked_test_data(qapp, tmp_path):
    import dataclasses

    from kkvim_gui.backend.outcome import frozen_copy

    raw = demo_outcome()
    refused = export.write_result(raw, tmp_path / "raw")  # sin(pi) ~ 1e-16 on an edge: not an SSSS/CCCC field (8.1)
    m = json.loads(refused["manifest"].read_text(encoding="utf-8"))
    assert not m["moments"]["available"] and "boundary nodes" in m["moments"]["reason"] and "moments" not in refused
    w = np.array(raw.w)
    w[[0, -1], :] = 0.0
    w[:, [0, -1]] = 0.0
    o = dataclasses.replace(raw, w=frozen_copy(w))
    paths = export.write_result(o, tmp_path / "demo")
    text = paths["moments"].read_text(encoding="utf-8")
    assert paths["moments"].name.endswith("_TEST-DATA.csv") and text.startswith("# TEST DATA")
    assert json.loads(paths["moments_json"].read_text(encoding="utf-8"))["data_origin"] == "demo"


def test_saved_package_reopens_with_its_statuses_and_moments(ssss, qtbot, tmp_path):  # noqa: F811
    run, w = ssss
    o = run.outcome
    folder = tmp_path / "pkg"
    export.write_result(o, folder, run.meta(0.0), run.detailed)
    loaded = saved_run.load(folder)
    assert np.array_equal(loaded.w, o.w) and np.array_equal(loaded.x, o.x) and loaded.D == o.D
    assert (loaded.status, loaded.field_role, loaded.mark, loaded.s) == (o.status, o.field_role, o.mark, o.s)
    assert len(loaded.history) == len(o.history)
    window = AnalysisWindow(Session(settings=None), saved_run.summary_rows(loaded), run=saved_run.saved_record(loaded, folder))
    qtbot.addWidget(window)
    assert np.array_equal(window.moments.data.moments.Mx, kk(o).Mx)


def test_cli_folder_reopens_and_its_moments_equal_the_cli_files(qtbot, tmp_path):
    out = tmp_path / "cli"
    code = cli.main(["run-cccc-n1", "--a", "1.2", "--b", "1.0", "--t", "0.01", "--E", "2.1e11", "--nu", "0.3", "--q", "1000",
                     "--nx", "12", "--ny", "10", "--half-steps", "4", "--out", str(out), "--moments"])
    assert code == 0
    loaded = saved_run.load(out)
    assert loaded.field_role == "research" and loaded.mark.startswith("RESEARCH_ONLY")
    data = moments_view.MomentsData.of(loaded)
    lines = [ln for ln in (out / "moments_research.csv").read_text(encoding="utf-8").splitlines() if not ln.startswith("#")]
    header = lines[0].split(",")
    cx = header.index("Mx [N]")
    for line in lines[1:]:
        cells = line.split(",")
        assert float(cells[cx]) == data.moments.Mx[int(cells[0]), int(cells[1])]


def test_not_a_saved_calculation_is_refused(tmp_path):
    with pytest.raises(saved_run.SavedRunError):
        saved_run.load(tmp_path)
