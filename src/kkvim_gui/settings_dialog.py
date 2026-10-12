"""Settings dialog (GUI_DESIGN.md, 4.3): theme and default figure language.

Segmented controls Light / Dark and RU / EN (clearly marked selection in both themes).
Nothing changes until OK; the theme then applies to all windows and both choices are
stored in the user settings.
"""

from PySide6.QtWidgets import QDialog, QDialogButtonBox, QHBoxLayout, QVBoxLayout

from kkvim_gui.widgets.common import Card, SegmentedControl, label


class SettingsDialog(QDialog):
    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        self.setWindowTitle("Настройки")
        self.setModal(True)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 16, 16, 12)
        lay.setSpacing(12)

        theme = Card("Тема интерфейса")
        self.theme = SegmentedControl({"light": "Light", "dark": "Dark"}, session.theme, strong=True)
        self.theme.setAccessibleName("Тема интерфейса")
        theme.body.addLayout(self._row(self.theme))
        theme.body.addWidget(label(
            "Light — светлая (по умолчанию), Dark — тёмная. Тема не влияет на публикационный "
            "экспорт рисунков.", "secondary", wrap=True))
        lay.addWidget(theme)

        lang = Card("Язык графиков по умолчанию")
        self.language = SegmentedControl(
            {"ru": "RU", "en": "EN"}, session.default_figure_language, strong=True)
        self.language.setAccessibleName("Язык графиков по умолчанию")
        lang.body.addLayout(self._row(self.language))
        lang.body.addWidget(label(
            "RU — русский, EN — английский (по умолчанию). Применяется при следующем запуске; "
            "язык графиков текущего сеанса переключается в окне анализа («Графики RU │ EN»). "
            "Единицы СИ и данные от языка не зависят.", "secondary", wrap=True))
        lay.addWidget(lang)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)
        self.setMinimumWidth(440)
        self.theme.buttons[session.theme].setFocus()

    @staticmethod
    def _row(control):
        row = QHBoxLayout()
        row.addWidget(control)
        row.addStretch(1)
        return row

    def accept(self):
        theme = self.theme.current()
        if theme != self.session.theme:
            self.session.set_theme(theme)
        self.session.set_default_figure_language(self.language.current())
        super().accept()
