"""Application entry point: `kkvim-gui` or `python -m kkvim_gui`.

Calculations run the KK-VIM solver in a separate process; the protocol of every run is written to
the application data folder (``runs/``). Options:
  --demo               demonstration backend only (TEST DATA, the solver is not called)
  --protocol-dir DIR   folder for run protocols (default: application data folder)
Options for design review (do not change saved settings):
  --theme light|dark   start with this theme
  --state start|setup|running|finished
  --analysis           also open an analysis window
  --dev                developer mode: state selector and Ctrl+1..Ctrl+4 in the main window
Qt options (for example -platform offscreen) are passed through to Qt.
"""

import argparse
import sys
from pathlib import Path

from PySide6.QtCore import QLibraryInfo, QLocale, QStandardPaths, Qt, QTranslator
from PySide6.QtGui import QFont, QGuiApplication
from PySide6.QtWidgets import QApplication

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

    def __init__(self, session, dev=False, demo=False, protocol_dir=None):
        self.session = session
        self.help = HelpService(session)
        self.analysis_windows = []
        self.main_window = MainWindow(session, dev=dev, demo=demo, protocol_dir=protocol_dir)
        self.main_window.analysisRequested.connect(self.open_analysis)
        self.main_window.settingsRequested.connect(lambda: self.open_settings(self.main_window))
        self.main_window.helpRequested.connect(self.help.open_search)
        self.help.install(self.main_window)

    def open_analysis(self):
        window = AnalysisWindow(self.session, self.main_window.run_rows(), run=self.main_window.analysis_run())
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
    return parser.parse_known_args(argv)


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


def main(argv=None):
    args, qt_args = parse_args(sys.argv[1:] if argv is None else argv)
    app = create_qapplication(qt_args)
    session = Session(user_settings())
    if args.theme:
        session.set_theme(args.theme, persist=False)
    else:
        session.apply_theme()
    protocol_dir = args.protocol_dir or default_protocol_dir()
    gui = GuiApplication(session, dev=args.dev, demo=args.demo, protocol_dir=None if args.demo else protocol_dir)
    gui.main_window.set_state(args.state)
    gui.main_window.show()
    if args.analysis:
        gui.open_analysis()
    return app.exec()
