"""Dark-blue application header under the native Windows frame (GUI_DESIGN.md, 4.1)."""

from PySide6.QtCore import QPoint, QPointF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QMenu, QToolButton, QWidget

from kkvim_gui.theme.tokens import LIGHT, SIZES

TEST_DATA_BADGE = "TEST DATA · демо-данные"
TEST_DATA_TOOLTIP = (
    "Демонстрационный режим: решатель не подключён, результаты не получены "
    "вычислительным алгоритмом"
)


def paint_logo(painter, rect, colour):
    """Program mark: a plate drawn as a parallelogram (a schematic, not a model)."""
    w, h = rect.width(), rect.height()
    x0, y0 = rect.x(), rect.y()
    plate = QPolygonF([
        QPointF(x0 + 0.30 * w, y0 + 0.22 * h),
        QPointF(x0 + 0.95 * w, y0 + 0.22 * h),
        QPointF(x0 + 0.70 * w, y0 + 0.78 * h),
        QPointF(x0 + 0.05 * w, y0 + 0.78 * h),
    ])
    pen = QPen(QColor(colour), max(1.2, w / 14))
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    painter.setPen(pen)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawPolygon(plate)
    painter.drawLine(QPointF(x0 + 0.50 * w, y0 + 0.22 * h), QPointF(x0 + 0.38 * w, y0 + 0.78 * h))
    painter.drawLine(QPointF(x0 + 0.175 * w, y0 + 0.50 * h), QPointF(x0 + 0.825 * w, y0 + 0.50 * h))


class LogoMark(QWidget):
    def __init__(self, colour, parent=None):
        super().__init__(parent)
        self._colour = colour
        self.setFixedSize(22, 22)

    def set_colour(self, colour):
        self._colour = colour
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        paint_logo(p, self.rect().adjusted(1, 1, -1, -1), self._colour)


def app_icon():
    """Window icon drawn at several sizes (sharp on HiDPI)."""
    icon = QIcon()
    for size in (16, 24, 32, 48, 64, 128, 256):
        pm = QPixmap(size, size)
        pm.fill(QColor(LIGHT["header.bg"]))
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        m = size // 8
        paint_logo(p, pm.rect().adjusted(m, m, -m, -m), LIGHT["header.fg"])
        p.end()
        icon.addPixmap(pm)
    return icon


class HeaderNav(QWidget):
    """Top navigation of the main window: one checkable button per section."""

    navigate = Signal(str)

    def __init__(self, sections, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        self.buttons = {}
        for key, text in sections.items():
            b = QToolButton()
            b.setText(text)
            b.setCheckable(True)
            b.setProperty("nav", True)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.clicked.connect(lambda _=False, k=key: self.navigate.emit(k))
            self.buttons[key] = b
            lay.addWidget(b)

    def set_current(self, key):
        for k, b in self.buttons.items():
            b.setChecked(k == key)

    def set_enabled(self, key, enabled, tooltip=""):
        self.buttons[key].setEnabled(enabled)
        self.buttons[key].setToolTip(tooltip)


class AppHeader(QFrame):
    """Logo, title, project, optional navigation, TEST DATA badge and the gear menu.

    The gear opens a small menu: "Настройки" and "Справка" (F1). The F1 hint in the menu is
    text only: F1 itself stays the contextual shortcut of each window (GUI_DESIGN.md, 17.3);
    registering it twice would make the shortcut ambiguous.

    Adapts to the width: on narrow windows (high Windows scaling) the project name is
    hidden and the badge is shortened, so that navigation and actions stay visible.
    """

    settingsClicked = Signal()
    helpClicked = Signal()
    NARROW_WIDTH = 1100

    def __init__(self, session, title, nav=None, show_badge=True, show_settings=True,
                 show_help=True, parent=None):
        super().__init__(parent)
        self.setObjectName("AppHeader")
        self.setFixedHeight(SIZES["header.height"])
        self._session = session
        self._project_text = ""

        lay = QHBoxLayout(self)
        lay.setContentsMargins(14, 0, 10, 0)
        lay.setSpacing(10)

        self.logo = LogoMark(session.tokens["header.fg"])
        self.title = QLabel(title)
        self.title.setObjectName("HeaderTitle")
        self.separator = sep = QFrame()
        sep.setObjectName("HeaderSeparator")
        sep.setFixedSize(1, 18)
        self.project = QLabel("")
        self.project.setObjectName("HeaderProject")
        self.nav = nav

        self._badge_full, self._badge_short = TEST_DATA_BADGE, "TEST DATA"
        self.badge = QLabel(TEST_DATA_BADGE)
        self.badge.setObjectName("TestDataBadge")
        self.badge.setToolTip(TEST_DATA_TOOLTIP)
        self.badge.setVisible(show_badge)

        self.menu = QMenu(self)
        self.menu.setObjectName("HeaderMenu")
        self.settings_action = self.menu.addAction("Настройки")
        self.settings_action.triggered.connect(self.settingsClicked)
        self.help_action = self.menu.addAction("Справка\tF1")  # hint only, not a shortcut
        self.help_action.triggered.connect(self.helpClicked)
        self.settings_action.setVisible(show_settings)
        self.help_action.setVisible(show_help)
        self.settings_button = self._icon_button("⚙", "Настройки и справка")
        # Opened by click, Space or Enter; right-aligned under the gear so it stays in the window.
        self.settings_button.clicked.connect(self.show_menu)
        self.settings_button.setVisible(show_settings or show_help)

        lay.addWidget(self.logo)
        lay.addWidget(self.title)
        lay.addWidget(sep)
        lay.addWidget(self.project)
        if nav is not None:
            lay.addSpacing(10)
            lay.addWidget(nav)
        lay.addStretch(1)
        lay.addWidget(self.badge, 0, Qt.AlignmentFlag.AlignVCenter)
        lay.addWidget(self.settings_button)

        # A bound method (not a lambda): Qt drops the connection when the header is deleted.
        session.themeChanged.connect(self._on_theme_changed)

    @staticmethod
    def _icon_button(text, tooltip):
        b = QToolButton()
        b.setText(text)
        b.setToolTip(tooltip)
        b.setAccessibleName(tooltip)
        b.setIconSize(QSize(16, 16))
        b.setCursor(Qt.CursorShape.PointingHandCursor)
        return b

    def show_menu(self):
        b = self.settings_button
        width = self.menu.sizeHint().width()
        self.menu.popup(b.mapToGlobal(QPoint(b.width() - width, b.height())))

    def _on_theme_changed(self, _name):
        self.logo.set_colour(self._session.tokens["header.fg"])

    def set_badge(self, full, short, tooltip):
        """Another mark in the badge (e.g. PROVISIONAL / NOT_QUALIFIED of an R4 result)."""
        self._badge_full, self._badge_short = full, short
        self.badge.setToolTip(tooltip)
        self.badge.setVisible(True)
        self._adapt()

    def set_project(self, text, modified=False):
        self._project_text = f"{text}  ●" if modified else text
        self.project.setText(self._project_text)
        self._adapt()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._adapt()

    def _adapt(self):
        narrow = self.width() < self.NARROW_WIDTH
        show_project = bool(self._project_text) and not narrow
        self.project.setVisible(show_project)
        self.separator.setVisible(show_project)
        self.badge.setText(self._badge_short if narrow else self._badge_full)
