"""Contextual help tooltips (GUI_DESIGN.md, 17.2).

A widget takes part by having the dynamic property "helpTopic" (see widgets.common.set_help);
children of such a widget (for example the line edit inside a UnitField) share its topic.
One application-wide event filter shows the article summary after a short delay, next to
the widget but never over it. The filter never consumes input events: keyboard and mouse
work as usual; inside help targets it only replaces Qt's standard tooltip.
"""

from PySide6.QtCore import QEvent, QObject, QPoint, QRect, Qt, QTimer
from PySide6.QtGui import QCursor
from PySide6.QtWidgets import QApplication, QFrame, QLabel, QVBoxLayout, QWidget

from kkvim_gui.help.articles import get

DELAY_MS = 500
GAP = 6
MAX_WIDTH = 360
HIDE_EVENTS = {
    QEvent.Type.MouseButtonPress,
    QEvent.Type.MouseButtonDblClick,
    QEvent.Type.KeyPress,
    QEvent.Type.Wheel,
    QEvent.Type.WindowDeactivate,
}


def help_target(widget, prop="helpTopic"):
    """The widget itself or the nearest ancestor in the same window with the property."""
    w = widget
    while w is not None:
        if w.property(prop):
            return w
        if w.isWindow():
            return None
        w = w.parentWidget()
    return None


def popup_position(target, cursor, size, avail, gap=GAP):
    """Top-left corner of the popup: below the target, else above, else beside it.

    target and avail are global QRects, cursor a global QPoint, size a QSize. The result
    never overlaps the target rectangle and stays inside the available screen area.
    """
    w, h = size.width(), size.height()
    x = min(max(cursor.x() - 16, avail.left()), avail.right() - w + 1)
    below = target.bottom() + 1 + gap
    above = target.top() - gap - h
    if below + h - 1 <= avail.bottom():
        return QPoint(x, below)
    if above >= avail.top():
        return QPoint(x, above)
    # Very tall target: place the popup to the right (or left) of it.
    y = min(max(cursor.y() - h // 2, avail.top()), avail.bottom() - h + 1)
    right = target.right() + 1 + gap
    if right + w - 1 <= avail.right():
        return QPoint(right, y)
    return QPoint(max(avail.left(), target.left() - gap - w), y)


class HelpPopup(QFrame):
    """Tooltip window: does not take focus and is transparent for the mouse."""

    def __init__(self):
        super().__init__(None, Qt.WindowType.ToolTip | Qt.WindowType.FramelessWindowHint)
        self.setObjectName("HelpPopup")
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 8, 10, 8)
        lay.setSpacing(4)
        self.title = QLabel()
        self.title.setObjectName("HelpPopupTitle")
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        self.hint = QLabel("F1 или ⓘ — подробная справка")
        self.hint.setObjectName("HelpPopupHint")
        for w in (self.title, self.summary, self.hint):
            lay.addWidget(w)
        self.setMaximumWidth(MAX_WIDTH)
        self.topic = None

    def show_for(self, article, target):
        self.topic = article.id
        self.title.setText(article.title)
        self.summary.setText(article.summary)
        self.adjustSize()
        top_left = target.mapToGlobal(QPoint(0, 0))
        rect = QRect(top_left, target.size())
        screen = target.screen() or QApplication.primaryScreen()
        self.move(popup_position(rect, QCursor.pos(), self.sizeHint(), screen.availableGeometry()))
        self.show()


class TooltipController(QObject):
    """Application-wide event filter for help tooltips."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.popup = HelpPopup()
        self._target = None
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._show)

    @property
    def visible_topic(self):
        return self.popup.topic if self.popup.isVisible() else None

    def hide(self):
        self._timer.stop()
        self.popup.hide()
        self.popup.topic = None

    def _cursor_inside_target(self):
        t = self._target
        return t is not None and t.isVisible() and t.rect().contains(t.mapFromGlobal(QCursor.pos()))

    def _show(self):
        t = self._target
        if t is None or not t.isVisible():
            return
        article = get(t.property("helpTopic"))
        if article is not None:
            self.popup.show_for(article, t)

    def eventFilter(self, obj, event):
        etype = event.type()
        if not isinstance(obj, QWidget) or obj is self.popup:
            return False
        if etype == QEvent.Type.Enter:
            target = help_target(obj)
            if target is not None and target is not self._target:
                self.hide()
                self._target = target
                self._timer.start(DELAY_MS)
        elif etype == QEvent.Type.Leave:
            if self._target is not None and not self._cursor_inside_target():
                self.hide()
                self._target = None
        elif etype in HIDE_EVENTS:
            if self._target is not None:
                self.hide()
                self._target = None
        elif etype == QEvent.Type.ToolTip and help_target(obj) is not None:
            return True  # the help popup replaces Qt's standard tooltip inside help targets
        return False
