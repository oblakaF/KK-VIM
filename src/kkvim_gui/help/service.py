"""Help service: F1 context resolution and the single help centre window (GUI_DESIGN.md, 17.3).

F1 picks the topic in this order:
1. the topic of the visible help tooltip;
2. "helpTopic" of the hovered widget or its ancestors;
3. "helpTopic" / "helpContext" of the focused widget or its ancestors (selected parameter);
4. "helpContext" of the hovered widget's page;
5. window.help_context() (the current page or tab);
6. none: the help centre opens with the search field.
"""

from PySide6.QtCore import QObject
from PySide6.QtGui import QCursor, QKeySequence, QShortcut
from PySide6.QtWidgets import QApplication

from kkvim_gui.help.articles import get
from kkvim_gui.help.tooltip import TooltipController, help_target

F1 = QKeySequence("F1")
_instance = None


def help_service():
    """The application's HelpService, or None (for example in isolated widget tests)."""
    return _instance


class HelpService(QObject):
    def __init__(self, session, parent=None):
        global _instance
        super().__init__(parent)
        self.session = session
        self.tooltips = TooltipController(self)
        QApplication.instance().installEventFilter(self.tooltips)
        self.window = None
        _instance = self

    def close(self):
        global _instance
        QApplication.instance().removeEventFilter(self.tooltips)
        self.tooltips.hide()
        if self.window is not None:
            self.window.close()
        if _instance is self:
            _instance = None

    def install(self, window):
        """F1 for one top-level window. F1 is not an editing key (GUI_DESIGN.md, 12)."""
        return QShortcut(F1, window, activated=lambda: self.on_f1(window))

    def context_topic(self, window):
        topic = self.tooltips.visible_topic
        if topic:
            return topic
        hovered = QApplication.widgetAt(QCursor.pos())
        if hovered is not None and hovered.window() is window:
            target = help_target(hovered)
            if target is not None:
                return target.property("helpTopic")
        focused = QApplication.focusWidget()
        if focused is not None and focused.window() is window:
            for prop in ("helpTopic", "helpContext"):
                target = help_target(focused, prop)
                if target is not None:
                    return target.property(prop)
        if hovered is not None and hovered.window() is window:
            target = help_target(hovered, "helpContext")
            if target is not None:
                return target.property("helpContext")
        context = getattr(window, "help_context", None)
        return context() if callable(context) else None

    def on_f1(self, window):
        topic = self.context_topic(window)
        self.tooltips.hide()
        if get(topic) is not None:
            self.open(topic)
        else:
            self.open_search()

    def _window(self):
        if self.window is None:
            from kkvim_gui.help.window import HelpWindow

            self.window = HelpWindow(self.session)
        return self.window

    def _present(self, w):
        w.show()
        w.raise_()
        w.activateWindow()

    def open(self, topic):
        w = self._window()
        w.show_topic(topic)
        self._present(w)
        return w

    def open_search(self, query=""):
        w = self._window()
        w.show_search(query)
        self._present(w)
        return w
