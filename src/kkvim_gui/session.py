"""Session state shared by all windows: theme and figure language.

- Theme (GUI_DESIGN.md, 10): Light by default, Dark optional; stored in the settings.
- Figure language (GUI_DESIGN.md, 7.2): one value for all windows of the session,
  initialised from the "default figure language" setting (EN). Switching it only
  redraws texts; it never triggers a computation and never converts units or data.
"""

from PySide6.QtCore import QObject, QSettings, Signal
from PySide6.QtWidgets import QApplication

from kkvim_gui.theme.qss import build_palette, build_stylesheet
from kkvim_gui.theme.tokens import DEFAULT_THEME, THEMES

FIGURE_LANGUAGES = ("en", "ru")
DEFAULT_FIGURE_LANGUAGE = "en"


def user_settings():
    """Persistent user settings (INI file in the user's application data folder)."""
    return QSettings(QSettings.Format.IniFormat, QSettings.Scope.UserScope, "KK-VIM", "kkvim-gui")


class Session(QObject):
    themeChanged = Signal(str)
    figureLanguageChanged = Signal(str)

    def __init__(self, settings=None, parent=None):
        """settings: a QSettings object, or None to keep everything in memory (tests)."""
        super().__init__(parent)
        self._settings = settings
        self._theme = self._read("theme", DEFAULT_THEME, THEMES)
        self._default_figure_language = self._read(
            "figure_language_default", DEFAULT_FIGURE_LANGUAGE, FIGURE_LANGUAGES
        )
        self._figure_language = self._default_figure_language

    def _read(self, key, default, allowed):
        if self._settings is None:
            return default
        value = self._settings.value(key, default)
        return value if value in allowed else default

    def _write(self, key, value):
        if self._settings is not None:
            self._settings.setValue(key, value)

    # Theme
    @property
    def theme(self):
        return self._theme

    @property
    def tokens(self):
        return THEMES[self._theme]

    def set_theme(self, name, persist=True):
        if name not in THEMES:
            raise ValueError(f"unknown theme: {name!r}")
        self._theme = name
        if persist:
            self._write("theme", name)
        self.apply_theme()
        self.themeChanged.emit(name)

    def apply_theme(self):
        app = QApplication.instance()
        if app is not None:
            app.setPalette(build_palette(self.tokens))
            app.setStyleSheet(build_stylesheet(self.tokens))

    # Figure language
    @property
    def figure_language(self):
        return self._figure_language

    def set_figure_language(self, lang):
        """Session-wide figure language; not persisted (the default is a setting)."""
        if lang not in FIGURE_LANGUAGES:
            raise ValueError(f"unknown figure language: {lang!r}")
        if lang != self._figure_language:
            self._figure_language = lang
            self.figureLanguageChanged.emit(lang)

    @property
    def default_figure_language(self):
        return self._default_figure_language

    def set_default_figure_language(self, lang):
        if lang not in FIGURE_LANGUAGES:
            raise ValueError(f"unknown figure language: {lang!r}")
        self._default_figure_language = lang
        self._write("figure_language_default", lang)
