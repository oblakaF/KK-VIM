"""Application entry point: `kkvim-gui` or `python -m kkvim_gui`.

Calculations run the KK-VIM solver in a separate process; the protocol of every run is written to
the application data folder (``runs/``). Kirchhoff–Love, uniform load: SSSS N = 1 (the accepted R3 solver, D-043),
SSSS N = 2–3 (R4, PROVISIONAL / NOT_QUALIFIED), SSSS N = 4–6 and CCCC N = 1–6 (research only, RESEARCH_ONLY /
PROVISIONAL / NOT_QUALIFIED), each on the domain and with the guards of its own solver. ``MainWindow(preview=True)``
remains the restricted functional preview (SSSS, N = 1).
Options:
  --demo               demonstration backend only (TEST DATA, the solver is not called)
  --protocol-dir DIR   folder for run protocols (default: application data folder)
  --smoke-test DIR     check of an installed application: run the default case through the windows, open the
                       analysis and its "Сравнение" tab, save the calculation package and screenshots to DIR, write
                       DIR/smoke.json, exit
  --smoke-case FILE    with --smoke-test: the case to type into the form instead of the default (JSON: supports, N,
                       a, b, h, E, nu, q in m and Pa, n_x, n_y, half_steps; optional series (loads in Pa: a load series
                       instead of one run), expect_status, expect_field_role, expect_comparison, expect_navier,
                       expect_series_points, screenshots (also the start, setup, help and every analysis tab))
Options for design review (do not change saved settings):
  --theme light|dark   start with this theme
  --state start|setup|running|finished
  --analysis           also open an analysis window
  --dev                developer mode: state selector and Ctrl+1..Ctrl+4 in the main window
Qt options (for example -platform offscreen) are passed through to Qt.
"""

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

from PySide6.QtCore import QLibraryInfo, QLocale, QStandardPaths, Qt, QTimer, QTranslator
from PySide6.QtGui import QFont, QGuiApplication
from PySide6.QtWidgets import QApplication, QMessageBox

from kkvim_gui.analysis.window import AnalysisWindow
from kkvim_gui.help.service import HelpService
from kkvim_gui.main_window.window import STATES, MainWindow
from kkvim_gui.session import Session, user_settings
from kkvim_gui.settings_dialog import SettingsDialog
from kkvim_gui.theme.tokens import THEMES

UI_FONT_FAMILIES = ["Segoe UI Variable Text", "Segoe UI"]
UI_FONT_POINT_SIZE = 9.5


def ui_font():
    font = QFont()
    font.setFamilies(UI_FONT_FAMILIES)
    font.setPointSizeF(UI_FONT_POINT_SIZE)
    return font


def install_qt_translations(app):
    """Russian texts of Qt's own widgets: context menus of fields, standard dialog buttons."""
    translator = QTranslator(app)
    path = QLibraryInfo.path(QLibraryInfo.LibraryPath.TranslationsPath)
    if translator.load(QLocale(QLocale.Language.Russian), "qtbase", "_", path):
        app.installTranslator(translator)
        return True
    return False


class GuiApplication:
    """Owns the session, the help service, the main window and the open analysis windows."""

    def __init__(self, session, dev=False, demo=False, protocol_dir=None, preview=False):
        self.session = session
        self.help = HelpService(session)
        self.analysis_windows = []
        self.main_window = MainWindow(session, dev=dev, demo=demo, protocol_dir=protocol_dir, preview=preview)
        self.main_window.analysisRequested.connect(self.open_analysis)
        self.main_window.seriesRequested.connect(self.open_series)
        self.main_window.settingsRequested.connect(lambda: self.open_settings(self.main_window))
        self.main_window.helpRequested.connect(self.help.open_search)
        self.help.install(self.main_window)

    def series_of(self, run):
        """SeriesData of the finished load series of ``run`` (None if the run is not in a series or it still runs)."""
        from kkvim_gui.model.series import series_data

        runs = self.main_window.runs
        series_id = getattr(run, "series_id", None)
        series = None if series_id is None else runs.get_series(series_id)
        return None if series is None or series.active else series_data(series, runs)

    def open_analysis(self):
        run = self.main_window.analysis_run()
        window = AnalysisWindow(self.session, self.main_window.run_rows(), run=run, series=self.series_of(run))
        return self._show(window)

    def open_series(self, series_id):
        """The «Нагрузка — прогиб» window of a finished load series."""
        from kkvim_gui.analysis.series import SeriesWindow
        from kkvim_gui.model.series import series_data

        runs = self.main_window.runs
        series = runs.get_series(series_id)
        if series is None or series.active:
            return None
        return self._show(SeriesWindow(self.session, series_data(series, runs)))

    def _show(self, window):
        window.settingsRequested.connect(lambda: self.open_settings(window))
        window.helpRequested.connect(self.help.open_search)
        self.help.install(window)
        window.destroyed.connect(lambda *_: self._forget(window))
        self.analysis_windows.append(window)
        window.show()
        window.raise_()
        window.activateWindow()
        return window

    def _forget(self, window):
        if window in self.analysis_windows:
            self.analysis_windows.remove(window)

    def open_settings(self, parent):
        SettingsDialog(self.session, parent).exec()


def parse_args(argv):
    parser = argparse.ArgumentParser(prog="kkvim-gui", description="KK-VIM graphical interface")
    parser.add_argument("--theme", choices=sorted(THEMES), help="theme for this session only")
    parser.add_argument("--state", choices=STATES, default="start", help="initial main window state")
    parser.add_argument("--analysis", action="store_true", help="also open an analysis window")
    parser.add_argument("--dev", action="store_true", help="developer mode (state selector, Ctrl+1..4)")
    parser.add_argument("--demo", action="store_true", help="demonstration backend (TEST DATA, no solver)")
    parser.add_argument("--protocol-dir", help="folder for run protocols")
    parser.add_argument("--smoke-test", metavar="DIR", help="run, analyse and save the default case, then exit")
    parser.add_argument("--smoke-case", metavar="FILE", help="with --smoke-test: JSON case typed into the form")
    return parser.parse_known_args(argv)


def show_unexpected_error(exc_type, exc, tb):
    """Uncaught exception of the interface: shown in a window (the installed GUI has no console), never hidden."""
    text = "".join(traceback.format_exception(exc_type, exc, tb))
    if sys.stderr is not None:
        sys.stderr.write(text)
    if QApplication.instance() is not None:
        QMessageBox.critical(None, "KK-VIM: ошибка интерфейса",
                             f"Непредвиденная ошибка интерфейса. Расчёт и сохранённые файлы не изменены.\n\n{text}")


class SmokeTest:
    """--smoke-test: the user workflow of the installed application, driven through its own windows."""

    TIMEOUT = 600.0

    def __init__(self, gui, app, folder, case=None):
        self.gui, self.app, self.folder = gui, app, Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.t0 = time.monotonic()
        self.case = case
        self.report = {"stage": "start", "case": case}
        self.timer = QTimer()
        self.timer.setInterval(250)
        self.timer.timeout.connect(self._poll)

    def _shot(self, widget, name):
        if self.case is not None and self.case.get("screenshots"):
            QApplication.processEvents()
            widget.grab().save(str(self.folder / name))

    def start(self):
        main = self.gui.main_window
        self._shot(main, "screen_start.png")
        main.set_state("setup")
        if self.case is not None:
            type_case(main.setup_model, self.case)
        QApplication.processEvents()
        self._shot(main, "screen_setup.png")
        if self.case is not None and self.case.get("screenshots"):
            main.setup_view.advanced_toggle.setChecked(True)
            main.setup_view.details_toggle.setChecked(True)
            self._shot(main, "screen_setup_details.png")
            main.setup_view.advanced_toggle.setChecked(False)
            main.setup_view.details_toggle.setChecked(False)
        self.report.update(installation=installation_record(), preview=main.preview, demo=main.demo,
                           route=main.setup_model.state.route, route_status=main.setup_view.route_status.text(),
                           n_items=[main.setup_view.N.itemText(i) for i in range(main.setup_view.N.count())],
                           series_preview=main.setup_view.series_preview.text() if self.series else None)
        main.setup_view.try_run()
        self.report["stage"] = "running"
        if main.runs.active is None:
            return self._finish(1, "the run did not start: " + main.setup_view.check_text.text())
        self.timer.start()

    @property
    def series(self):
        return self.case is not None and bool(self.case.get("series"))

    def _poll(self):
        main = self.gui.main_window
        busy = main.runs.active is not None or main.runs.references_pending or (
            self.series and main.runs.series and main.runs.series[-1].active)
        if busy:
            if time.monotonic() - self.t0 > self.TIMEOUT:
                self._finish(2, "timeout")
            return
        self.timer.stop()
        try:
            if self.series:
                self._after_series()
            else:
                self._after_run()
        except Exception:  # noqa: BLE001 - reported in smoke.json and by the exit code
            self._finish(1, traceback.format_exc())

    def _after_series(self):
        from kkvim_gui.model.series import series_data

        main = self.gui.main_window
        series = main.runs.series[-1]
        main.run_view.select(series.run_ids[-1])
        QApplication.processEvents()
        main.grab().save(str(self.folder / "screen_main.png"))
        window = self.gui.open_series(series.series_id)
        QApplication.processEvents()
        window.grab().save(str(self.folder / "screen_series.png"))
        kx, ky, nx, ny = window.tab.figure.curves()
        paths = window.tab.save(self.folder)
        data = series_data(series, main.runs)
        manifest = json.loads(Path(paths["manifest"]).read_text(encoding="utf-8"))
        points = [{"q": p.q, "run_id": p.run_id, "run_status": p.run_status, "solver_status": p.solver_status,
                   "field_role": p.field_role, "included": p.included, "reason": p.reason, "w_centre_m": p.w_centre_m,
                   "w_at_max_abs_m": p.w_at_max_abs_m, "navier_state": p.navier_state, "navier_w_m": p.navier_w_m,
                   "delta_w_m": p.delta_w_m, "rel_pct": p.rel_pct, "scientific_status": p.scientific_status}
                  for p in data.points]
        self.report.update(stage="finished", series_status=series.status, series_points=points,
                           curves={"kkvim_q_kPa": kx, "kkvim_w_mm": ky, "navier_q_kPa": nx, "navier_w_mm": ny},
                           navier_state=None if series.navier is None else series.navier.get("state"),
                           package={k: str(v) for k, v in paths.items()}, manifest_files=manifest["files"])
        expected = self.case.get("expect_series_points", len(points))
        ok = (series.status == "finished" and len(data.included) == expected
              and self.case.get("expect_navier", self.report["navier_state"]) == self.report["navier_state"])
        self._finish(0 if ok else 1, "ok" if ok else "unexpected series result")

    def _after_run(self):
        main = self.gui.main_window
        run = main.runs.runs[-1]
        o = run.outcome
        self.report.update(stage="finished", run_status=run.status, solver_status=None if o is None else o.status)
        main.grab().save(str(self.folder / "screen_main.png"))
        if o is None or not o.has_field:
            return self._finish(1, "no field")
        main.run_view.select(run.run_id)
        window = self.gui.open_analysis()
        QApplication.processEvents()
        window.grab().save(str(self.folder / "screen_analysis_overview.png"))
        window.tabs.setCurrentWidget(window.fields)
        window.fields.set_view("surface")
        QApplication.processEvents()
        window.grab().save(str(self.folder / "screen_analysis_surface.png"))
        window.fields.set_view("map")
        QApplication.processEvents()
        window.grab().save(str(self.folder / "screen_analysis_map.png"))
        for name, tab in (("sections", window.sections), ("history", window.history), ("diagnostics", window.energy)):
            window.tabs.setCurrentWidget(tab)
            self._shot(window, f"screen_analysis_{name}.png")
        comparison = self._comparison(window)
        self.report["overview_comparison"] = window.overview.comparison_rows
        self.report["navier"] = run.navier
        self.report["tab_titles"] = [window.tabs.tabText(i) for i in range(window.tabs.count())]
        self.report["statuses"] = [window.status_run.text(), window.status_stop.text(), window.status_accept.text()]
        if self.case is not None and self.case.get("screenshots"):
            help_window = self.gui.help.open("verification.navier")
            self._shot(help_window, "screen_help_navier.png")
            help_window.show_search("Навье")
            self._shot(help_window, "screen_help_search.png")
            help_window.show_topic("method.overview")
            self._shot(help_window, "screen_help_method.png")
            self.report["help_search_navier"] = help_window.topics_in_index()
        paths = window.save_package(self.folder)
        e = o.extremum()
        manifest = json.loads(Path(paths["manifest"]).read_text(encoding="utf-8"))
        self.report.update(N=o.request.N, supports=o.request.supports, grid=[o.request.n_x, o.request.n_y], s=o.s,
                           s_requested=o.s_requested, route=o.route, solver=o.solver, call=o.call, mark=o.mark,
                           field_role=o.field_role, requested_completed=o.requested_completed, w_max=e["w"],
                           w_bar_at_max=e["w_bar"], x=e["x"], y=e["y"], analysis_badge=window.header.badge.text(),
                           analysis_badge_visible=not window.header.badge.isHidden(),
                           package={k: str(v) for k, v in paths.items()},
                           manifest_statuses=manifest["statuses"], manifest_marks=manifest["marks"],
                           manifest_files=manifest["files"], comparison=comparison)
        if self.case is None:
            ok = o.status == "CONVERGED"
        else:
            navier_state = None if run.navier is None else run.navier.get("state")
            ok = (self.case.get("expect_status", o.status) == o.status
                  and self.case.get("expect_field_role", o.field_role) == o.field_role
                  and self.case.get("expect_comparison", comparison["state"]) == comparison["state"]
                  and self.case.get("expect_navier", navier_state) == navier_state
                  and ("comparison" in manifest["files"]) == (comparison["state"] == "available")
                  and manifest["statuses"]["solver_status"] == o.status
                  and manifest["statuses"]["field_role"] == o.field_role)
        self._finish(0 if ok else 1, "ok" if ok else "unexpected status or field role")

    def _comparison(self, window):
        """Open "Сравнение" as a user would: screenshot, the comparison record and the texts shown in its table."""
        from kkvim_gui.backend import benchmarks

        tab = window.comparison
        window.tabs.setCurrentWidget(tab)
        QApplication.processEvents()
        window.grab().save(str(self.folder / "screen_analysis_comparison.png"))
        shown, tips = [], []
        if tab.table is not None:
            model = tab.table.model()
            shown = [[model.data(model.index(i, j)) for j in range(model.columnCount())] for i in range(model.rowCount())]
            tips = [model.data(model.index(i, benchmarks.DETAIL_COLUMN), Qt.ItemDataRole.ToolTipRole)
                    for i in range(model.rowCount())]
        return {**benchmarks.as_record(tab.comparison), "tab_enabled": window.tabs.isTabEnabled(window.tabs.indexOf(tab)),
                "state_text": tab.state_label.text(), "table_shown": shown, "provenance_tooltips": tips}

    def _finish(self, code, text):
        self.report.update(exit_code=code, result=text, seconds=round(time.monotonic() - self.t0, 2))
        (self.folder / "smoke.json").write_text(json.dumps(self.report, ensure_ascii=False, indent=2),
                                                encoding="utf-8")
        for window in list(self.gui.analysis_windows):
            window.close()
        self.gui.main_window.runs.shutdown()
        self.app.exit(code)


def create_qapplication(qt_args):
    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication([sys.argv[0], *qt_args])
    app.setOrganizationName("KK-VIM")
    app.setApplicationName("kkvim-gui")
    app.setStyle("Fusion")
    app.setFont(ui_font())
    install_qt_translations(app)
    return app


def default_protocol_dir():
    """<application data>/runs (for example %LOCALAPPDATA%/KK-VIM/kkvim-gui/runs on Windows)."""
    base = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppLocalDataLocation)
    return Path(base) / "runs" if base else None


def type_case(model, case):
    """Type a smoke case into the setup form as a user would (values in m and Pa); research choices only if asked."""
    model.set_choice(supports=case["supports"], N=int(case["N"]))
    for key in ("a", "b", "h", "E", "nu", "q", "n_x", "n_y"):
        model.set_text(key, str(case[key]))
    for key, unit in (("a", "m"), ("b", "m"), ("h", "m"), ("E", "Pa"), ("q", "Pa")):
        model.set_unit(key, unit)
    if case.get("series"):
        model.set_choice(load_mode="series")
        model.set_text("q_series", "; ".join(repr(float(q)) for q in case["series"]))
    if case["supports"] == "CCCC":
        model.set_choice(cccc_half_steps=int(case["half_steps"]))
    elif "half_steps" in case:
        model.set_text("half_steps", str(case["half_steps"]))
        if int(case["N"]) >= 4:
            model.set_choice(ssss_research=bool(case.get("ssss_research", False)))


def installation_record():
    """Where the running program and its modules come from (the check of an installed application).

    The location of kkvim is looked up without importing it: only the adapter imports kkvim (test_gui_independence)."""
    import importlib.util

    import kkvim_gui
    from kkvim_gui.backend import benchmarks
    from kkvim_gui.help.articles import load_articles

    return {"executable": sys.executable, "prefix": sys.prefix, "kkvim": importlib.util.find_spec("kkvim").origin,
            "kkvim_gui": kkvim_gui.__file__,
            "kkvim_navier_worker": importlib.util.find_spec("kkvim_navier_worker").origin,
            "kkvim_reference": importlib.util.find_spec("kkvim_reference").origin,
            "help_articles": len(load_articles()), "benchmarks_payload_ok": bool(benchmarks.load())}


def main(argv=None):
    args, qt_args = parse_args(sys.argv[1:] if argv is None else argv)
    app = create_qapplication(qt_args)
    session = Session(user_settings())
    if args.theme:
        session.set_theme(args.theme, persist=False)
    else:
        session.apply_theme()
    protocol_dir = args.protocol_dir or default_protocol_dir()
    sys.excepthook = show_unexpected_error
    gui = GuiApplication(session, dev=args.dev, demo=args.demo, protocol_dir=None if args.demo else protocol_dir)
    gui.main_window.set_state(args.state)
    gui.main_window.show()
    if args.analysis:
        gui.open_analysis()
    if args.smoke_test:
        case = json.loads(Path(args.smoke_case).read_text(encoding="utf-8")) if args.smoke_case else None
        smoke = SmokeTest(gui, app, args.smoke_test, case)
        QTimer.singleShot(500, smoke.start)
    return app.exec()
