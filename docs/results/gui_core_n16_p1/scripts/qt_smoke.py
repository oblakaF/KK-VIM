"""GUI-CORE-N16-P1-QA — real Qt workflow (offscreen platform), QA scratch only.

usage: QT_QPA_PLATFORM=offscreen .venv/Scripts/python.exe -X utf8 scratch/gui_n16/qt_smoke.py <out_dir>
The application window (GuiApplication, as kkvim-gui creates it, preview=False) is driven through its form widgets:
case 1 SSSS N = 1 R3; case 2 CCCC N = 4, 32 x 32, s = 2 (child process); case 3 CCCC N = 6, a = 5, b = 1, 32 x 13, s = 2
(RN-5 refusal). For each: run view texts, the analysis window (field, history, status), screenshots, the saved package,
the field file and manifest, and a second save that must be refused without changing the first package.
Invalid input after a run must not start a run or replace the package.
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

if __name__ == "__main__":
    import numpy as np
    from PySide6.QtWidgets import QApplication, QLabel

    from kkvim_gui.app import GuiApplication, create_qapplication
    from kkvim_gui.backend.process import SolverProcess
    from kkvim_gui.model.setup_state import N_VALUES
    from kkvim_gui.session import Session

    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    app = create_qapplication([])
    session = Session(settings=None)
    session.apply_theme()
    gui = GuiApplication(session, protocol_dir=out / "protocols")
    main = gui.main_window
    main.resize(1400, 900)
    main.show()
    report, failures = {"platform": QApplication.platformName()}, []

    def ok(case, name, cond, detail=None):
        report.setdefault(case, {})[name] = bool(cond) if detail is None else {"ok": bool(cond), "detail": detail}
        if not cond:
            failures.append(f"{case}: {name} {detail if detail is not None else ''}")

    def labels(widget):
        return [w.text() for w in widget.findChildren(QLabel) if not w.isHidden()]

    def fill(supports, N, a, b, nx, ny, q=1000.0, s=None):
        v = main.setup_view
        v.supports.set_current(supports)
        v.N.setCurrentIndex(N_VALUES.index(N))
        for key, value, unit in (("a", a, "m"), ("b", b, "m"), ("h", 0.01, "m"), ("E", 2.1e11, "Pa"), ("nu", 0.3, None),
                                 ("q", q, "Pa"), ("n_x", nx, None), ("n_y", ny, None)):
            f = getattr(v, key)
            if unit:
                f.unit_combo.setCurrentText(unit)
            f.edit.setText(repr(value) if isinstance(value, float) else str(value))
        if supports == "CCCC" and s is not None:
            v.cccc_half_steps.setCurrentIndex(v.cccc_half_steps.findData(s))
        QApplication.processEvents()

    def run_case(case, expect_role, expect_file, timeout=900):
        main.set_state("setup")
        QApplication.processEvents()
        started = main.setup_view.try_run()
        run = main.runs.active
        ok(case, "started", started is not False and run is not None, main.setup_view.check_text.text())
        child = isinstance(run._process, SolverProcess) and run._process.pid not in (None, os.getpid())
        ok(case, "child_process", child, run._process.pid)
        t0 = time.monotonic()
        while main.runs.active is not None and time.monotonic() - t0 < timeout:
            QApplication.processEvents()
            time.sleep(0.05)
        for _ in range(20):
            QApplication.processEvents()
        o = run.outcome
        rv = main.run_view
        shown = labels(rv)
        report[case]["outcome"] = {"status": o.status, "s": o.s, "s_requested": o.s_requested, "field_role": o.field_role,
                                   "run_status": run.status, "has_results": run.has_results,
                                   "has_diagnostic_field": run.has_diagnostic_field, "seconds": round(time.monotonic() - t0, 1)}
        report[case]["run_view_texts"] = [t for t in shown if t.strip()][:60]
        ok(case, "field_role", o.field_role == expect_role, o.field_role)
        ok(case, "status_shown", any(o.status in t for t in shown))
        if o.status != "CONVERGED":
            ok(case, "no_CONVERGED_shown", not any("CONVERGED" in t for t in shown))
        main.grab().save(str(out / f"{case}_run_view.png"))
        rv.analysis_button.click()
        for _ in range(20):
            QApplication.processEvents()
        window = gui.analysis_windows[-1]
        window.resize(1400, 900)
        ok(case, "analysis_outcome_is_run", window.outcome is o)
        ok(case, "analysis_field_shown_mm", np.array_equal(np.asarray(window.fields.figure.image.image), np.asarray(o.w) * 1000.0))
        badge = window.header.badge.text()
        report[case]["analysis_badge"] = badge
        report[case]["analysis_mark"] = window.fields.figure.mark.text()
        report[case]["analysis_texts"] = [t for t in labels(window) if t.strip()][:60]
        if expect_role == "diagnostic":
            ok(case, "analysis_diagnostic_badge", "ДИАГНОСТИКА" in badge or "DIAGNOSTIC" in badge, badge)
            ok(case, "run_view_diagnostic_button", rv.analysis_button.text().startswith("Открыть диагностику"), rv.analysis_button.text())
        elif expect_role == "research":
            ok(case, "analysis_research_badge", "RESEARCH" in badge, badge)
        ok(case, "analysis_status_shown", any(o.status in t for t in labels(window)))
        for tab in ("fields", "history") if hasattr(window, "history") else ("fields",):
            window.tabs.setCurrentWidget(getattr(window, tab))
            QApplication.processEvents()
            window.grab().save(str(out / f"{case}_analysis_{tab}.png"))
        pkg_root = out / "packages" / case
        paths = window.save_package(pkg_root)
        folder = paths["manifest"].parent
        names = sorted(p.name for p in folder.iterdir())
        report[case]["package_files"] = names
        ok(case, "package_field_file", expect_file in names and not ({"field.csv", "field_research.csv", "field_diagnostic.csv"}
                                                                    - {expect_file}) & set(names), names)
        m = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        ok(case, "manifest", m["schema_version"] == 2 and m["statuses"]["field_role"] == expect_role
           and m["statuses"]["field_file"] == expect_file and m["statuses"]["solver_status"] == o.status)
        html = paths["report"].read_text(encoding="utf-8")
        ok(case, "report_html", o.status in html and expect_file in html)
        before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir()}
        try:
            window.save_package(pkg_root)
            refused = False
        except FileExistsError:
            refused = True
        after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir()}
        ok(case, "second_save_refused_unchanged", refused and before == after)
        try:
            main.save_package(run.run_id, pkg_root / "main")
            main.save_package(run.run_id, pkg_root / "main")
            refused2 = False
        except FileExistsError:
            refused2 = True
        ok(case, "main_window_second_save_refused", refused2)
        window.close()
        return run, folder, before

    fill("SSSS", 1, 1.0, 1.0, 16, 16)
    r1, f1, b1 = run_case("Q1_SSSS_R3", "accepted", "field.csv")
    fill("CCCC", 4, 1.0, 1.0, 32, 32, s=2)
    ok("Q2_CCCC_N4_s2", "route_from_form", main.setup_model.state.route == "cccc_n46" and main.setup_model.state.cccc_half_steps == 2)
    r2, f2, b2 = run_case("Q2_CCCC_N4_s2", "research", "field_research.csv")
    fill("CCCC", 6, 5.0, 1.0, 32, 13, s=2)
    r3, f3, b3 = run_case("Q3_CCCC_N6_rn5_refusal", "diagnostic", "field_diagnostic.csv")
    # invalid input after the runs: no new run, no package changed
    main.set_state("setup")
    n_runs = len(main.runs.runs)
    main.setup_view.n_x.edit.setText("abc")
    QApplication.processEvents()
    main.setup_view.try_run()
    QApplication.processEvents()
    ok("Q4_invalid_input", "no_new_run", len(main.runs.runs) == n_runs and main.runs.active is None)
    fill("CCCC", 4, 1.0, 1.0, 16, 16, s=1)
    main.setup_view.cccc_half_steps.setCurrentIndex(main.setup_view.cccc_half_steps.findData(2))
    items = [main.setup_view.cccc_half_steps.itemData(i) for i in range(main.setup_view.cccc_half_steps.count())]
    ok("Q4_invalid_input", "cccc_N4_offers_only_s1_s2", items == [1, 2], items)
    for folder, before in ((f1, b1), (f2, b2), (f3, b3)):
        now = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir()}
        ok("Q4_invalid_input", f"package_unchanged_{folder.parent.name}", now == before)
    main.grab().save(str(out / "Q4_setup_after_invalid.png"))
    main.runs.shutdown()
    report["failures"] = failures
    report["verdict"] = "PASS" if not failures else "FAIL"
    (out / "qt_smoke.json").write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps({"failures": failures, "verdict": report["verdict"]}, ensure_ascii=False, indent=1))
    sys.exit(0 if not failures else 1)
