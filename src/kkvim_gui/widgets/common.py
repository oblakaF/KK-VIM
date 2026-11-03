"""Common building blocks: cards, unit fields, segmented controls, statuses, figure placeholders.

Numeric fields are plain QLineEdit widgets with Qt's standard editing (Ctrl+C/V/X/Z/Y/A,
Backspace, Delete; GUI_DESIGN.md, 12.1). They have no input masks, validators or
character replacement; number parsing on commit is stage P2 (GUI_DESIGN.md, 11.2).
"""

from PySide6.QtCore import QEvent, QObject, Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from kkvim_gui.figures.text import fig_text
from kkvim_gui.help.service import help_service
from kkvim_gui.theme.tokens import SIZES

# SI unit symbols allowed in the interface (GUI_DESIGN.md, 11.1; GUI-D14).
SI_UNITS = ("m", "cm", "mm", "Pa", "kPa", "MPa", "GPa", "N·m", "kN·m")


def repolish(widget):
    """Re-apply the stylesheet after a dynamic property change."""
    widget.style().unpolish(widget)
    widget.style().polish(widget)
    widget.update()


def set_help(widget, topic):
    """Attach a help topic: tooltip on hover, article on F1 (GUI_DESIGN.md, 17.2)."""
    widget.setProperty("helpTopic", topic)
    return widget


def set_help_context(widget, topic):
    """F1 topic of a page or area without its own tooltip."""
    widget.setProperty("helpContext", topic)
    return widget


class InfoButton(QToolButton):
    """"ⓘ" next to a parameter name: opens the full help article."""

    def __init__(self, topic, parent=None):
        super().__init__(parent)
        self.setObjectName("InfoButton")
        self.setText("ⓘ")
        self.setAccessibleName("Подробная справка")
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        set_help(self, topic)
        self.clicked.connect(lambda: help_service() and help_service().open(topic))


def label_with_info(text, topic, role="fieldLabel"):
    """Parameter name with the help tooltip and an "ⓘ" button."""
    box = QWidget()
    lay = QHBoxLayout(box)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(4)
    lab = set_help(label(text, role), topic)
    lay.addWidget(lab)
    lay.addWidget(InfoButton(topic))
    lay.addStretch(1)
    return box


def label(text, role=None, wrap=False):
    lab = QLabel(text)
    if role:
        lab.setProperty("role", role)
    lab.setWordWrap(wrap)
    return lab


class Card(QFrame):
    """White block with a title row; content goes into `self.body`."""

    def __init__(self, title=None, right=None, parent=None):
        super().__init__(parent)
        self.setProperty("card", True)
        pad = SIZES["card.padding"]
        outer = QVBoxLayout(self)
        outer.setContentsMargins(pad, pad - 2, pad, pad)
        outer.setSpacing(10)
        if title or right is not None:
            row = QHBoxLayout()
            row.setSpacing(8)
            if title:
                self.title_label = label(title, "cardTitle")
                row.addWidget(self.title_label)
            row.addStretch(1)
            if right is not None:
                row.addWidget(right)
            outer.addLayout(row)
        self.body = QVBoxLayout()
        self.body.setSpacing(8)
        outer.addLayout(self.body, 1)


class _FocusWatcher(QObject):
    """Highlights the UnitField frame while its line edit has focus (QSS has no :focus-within)."""

    def __init__(self, frame):
        super().__init__(frame)
        self._frame = frame

    def eventFilter(self, obj, event):
        if event.type() in (QEvent.Type.FocusIn, QEvent.Type.FocusOut):
            self._frame.setProperty("focused", event.type() == QEvent.Type.FocusIn)
            repolish(self._frame)
        return False


class UnitField(QFrame):
    """Number input with its SI unit inside the field (fixed label or explicit unit list).

    The text is never rewritten automatically: the value is parsed on commit (P2).
    Choosing another unit never converts the number (GUI_DESIGN.md, 11.1).
    """

    def __init__(self, text="", unit=None, units=None, placeholder="", parent=None):
        super().__init__(parent)
        self.setObjectName("UnitField")
        for u in ([unit] if unit else []) + list(units or []):
            if u not in SI_UNITS:
                raise ValueError(f"not an SI unit symbol allowed in the GUI: {u!r}")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self.edit = QLineEdit(text)
        self.edit.setPlaceholderText(placeholder)
        self.edit.installEventFilter(_FocusWatcher(self))
        lay.addWidget(self.edit, 1)
        self.unit_combo = None
        self.unit_label = None
        if units:
            self.unit_combo = QComboBox()
            self.unit_combo.addItems(list(units))
            if unit:
                self.unit_combo.setCurrentText(unit)
            self.unit_combo.setToolTip("Единица СИ; смена единицы не пересчитывает число")
            lay.addWidget(self.unit_combo)
        elif unit:
            self.unit_label = QLabel(unit)
            lay.addWidget(self.unit_label)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def text(self):
        return self.edit.text()

    def set_invalid(self, invalid, message=""):
        """Red frame for an invalid value; the message is also given to screen readers."""
        self.setProperty("invalid", bool(invalid))
        self.edit.setAccessibleDescription(message if invalid else "")
        repolish(self)

    def unit(self):
        if self.unit_combo is not None:
            return self.unit_combo.currentText()
        return self.unit_label.text() if self.unit_label is not None else ""


def error_label():
    """Validation message under a field: hidden until there is an error."""
    lab = label("", "fieldError", wrap=True)
    lab.setVisible(False)
    return lab


class FieldGrid(QGridLayout):
    """Two-column grid: label on the left, input on the right, error message under the input."""

    def __init__(self):
        super().__init__()
        self.setHorizontalSpacing(12)
        self.setVerticalSpacing(6)
        self.setColumnStretch(1, 1)
        self.errors = {}

    def add_row(self, text, widget, tooltip=None, topic=None):
        """topic: help topic id — tooltip, "ⓘ" and F1 for the label and the input."""
        row = self.rowCount()
        if topic:
            lab = label_with_info(text, topic)
            set_help(widget, topic)
        else:
            lab = label(text, "fieldLabel")
            if tooltip:
                lab.setToolTip(tooltip)
                widget.setToolTip(tooltip)
        self.addWidget(lab, row, 0)
        self.addWidget(widget, row, 1)
        err = error_label()
        self.addWidget(err, row + 1, 1)
        self.errors[widget] = err
        return lab


class SegmentedControl(QWidget):
    """Exclusive row of checkable buttons.

    Keyboard: Tab reaches the buttons, Space selects, Left/Right move the selection.
    strong=True fills the selected segment with the accent colour (settings dialog).
    """

    changed = Signal(str)

    def __init__(self, options, current=None, strong=False, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self.buttons = {}
        keys = list(options)
        for i, key in enumerate(keys):
            b = QPushButton(options[key])
            b.setCheckable(True)
            b.setProperty("segment", "first" if i == 0 else "last" if i == len(keys) - 1 else "middle")
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            # Never narrower than the text (no clipped labels at high Windows scaling).
            b.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
            if strong:
                b.setProperty("strong", True)
            b.installEventFilter(self)
            self._group.addButton(b)
            self.buttons[key] = b
            lay.addWidget(b)
            b.toggled.connect(lambda on, k=key: on and self.changed.emit(k))
        self.set_current(current if current is not None else keys[0])

    def current(self):
        for key, b in self.buttons.items():
            if b.isChecked():
                return key
        return None

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.KeyPress and event.key() in (Qt.Key.Key_Left, Qt.Key.Key_Right):
            keys = list(self.buttons)
            i = keys.index(self.current()) if self.current() in keys else 0
            i = max(0, min(len(keys) - 1, i + (1 if event.key() == Qt.Key.Key_Right else -1)))
            self.set_current(keys[i])
            self.buttons[keys[i]].setFocus(Qt.FocusReason.TabFocusReason)
            return True
        return False

    def set_current(self, key):
        b = self.buttons[key]
        if not b.isChecked():
            b.setChecked(True)


STATUS_SYMBOLS = {"success": "✓", "warning": "◐", "error": "✕", "neutral": "○"}


class StatusLabel(QLabel):
    """One of the three separate statuses: execution, stopping criterion, scientific acceptance."""

    def __init__(self, status, text, parent=None):
        super().__init__(parent)
        self.set_status(status, text)

    def set_status(self, status, text):
        self.setProperty("status", status)
        self.setText(f"{STATUS_SYMBOLS[status]}  {text}")
        repolish(self)


class MetricTile(QFrame):
    def __init__(self, caption, value, parent=None):
        super().__init__(parent)
        self.setProperty("metric", True)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 8, 12, 10)
        lay.setSpacing(2)
        lay.addWidget(label(caption, "metricLabel"))
        self.value = label(value, "metricValue")
        self.value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        lay.addWidget(self.value)


class Note(QFrame):
    """Information plate ("info"), the TEST DATA warning ("testdata") or the lock banner ("lock")."""

    def __init__(self, text, kind="info", parent=None):
        super().__init__(parent)
        self.setProperty("note", kind)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 7, 10, 7)
        lay.setSpacing(8)
        icon = QLabel({"testdata": "⚠", "lock": "⏸"}.get(kind, "ⓘ"))
        icon.setAlignment(Qt.AlignmentFlag.AlignTop)
        lay.addWidget(icon)
        self.text = label(text, "secondary" if kind == "info" else None, wrap=True)
        if kind == "lock":
            self.text.setProperty("role", "lockText")
        lay.addWidget(self.text, 1)


class FigureLanguageSwitch(QWidget):
    """"Графики RU | EN": session-wide figure language (GUI_DESIGN.md, 7.2; order RU, EN — 21)."""

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self._session = session
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)
        lay.addWidget(label("Графики", "fieldLabel"))
        self.control = SegmentedControl({"ru": "RU", "en": "EN"}, session.figure_language)
        self.control.setToolTip(
            "Язык подписей рисунков. Меняет только тексты; данные, единицы СИ и расчёт не меняются"
        )
        lay.addWidget(self.control)
        self.control.changed.connect(session.set_figure_language)
        session.figureLanguageChanged.connect(self.control.set_current)


# Interactive X-Y plot tools (GUI_DESIGN.md, 16.8): (text, tooltip, stage).
PLOT_TOOLS = (
    ("⌂", "Исходный вид", "P3"),
    ("+", "Приблизить", "P3"),
    ("−", "Отдалить", "P3"),
    ("Snap", "Привязка маркера к активной кривой", "P5"),
    ("Проекции", "Проекции точки на оси X и Y", "P5"),
    ("Линии ▾", "Видимость кривых и активная кривая", "P5"),
    ("Оформление ▾", "Оформление графика", "P6"),
)


class PlotToolbarPlaceholder(QWidget):
    """Disabled X-Y plot toolbar: reserves the place and shows the planned tools."""

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        for i, (text, tip, stage) in enumerate(PLOT_TOOLS):
            if i in (3, 5):
                lay.addSpacing(8)
            b = QToolButton()
            b.setText(text)
            b.setToolTip(f"{tip} — этап {stage}")
            b.setEnabled(False)
            b.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            lay.addWidget(b)
        lay.addStretch(1)


class FigurePlaceholder(QFrame):
    """Reserved place of a scientific figure (real plots come at P3-P5).

    The title and the TEST DATA mark follow the session figure language; the note is
    interface text (Russian). xy=True adds the planned interactive X-Y plot toolbar.
    """

    def __init__(self, session, title_key, note, title_values=None, xy=False, parent=None):
        super().__init__(parent)
        self.setProperty("figure", True)
        self._session = session
        self._key = title_key
        self._values = title_values or {}
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 10, 14, 12)
        lay.setSpacing(8)
        top = QHBoxLayout()
        self.title = label("", "figureTitle")
        top.addWidget(self.title)
        top.addStretch(1)
        self.mark = label("", "figureMark")
        top.addWidget(self.mark)
        lay.addLayout(top)
        # The X-Y plot toolbar has its own row so that it is never squeezed.
        self.toolbar = PlotToolbarPlaceholder() if xy else None
        if self.toolbar is not None:
            lay.addWidget(self.toolbar)
        area = QFrame()
        area.setObjectName("FigureArea")
        area_lay = QVBoxLayout(area)
        self.note = label(note, "secondary", wrap=True)
        self.note.setAlignment(Qt.AlignmentFlag.AlignCenter)
        area_lay.addWidget(self.note)
        lay.addWidget(area, 1)
        self.setMinimumHeight(140)
        self._retranslate(session.figure_language)
        session.figureLanguageChanged.connect(self._retranslate)

    def _retranslate(self, lang):
        self.title.setText(fig_text(lang, self._key, **self._values))
        self.mark.setText(fig_text(lang, "mark.test_data"))


def fit_to_screen(window, design=(1440, 900), minimum=(960, 540)):
    """Initial and minimum size in logical pixels, never larger than the available work area.

    Windows scaling changes the logical screen size: Full HD gives 1920x1080 at 100 %,
    about 1280x720 at 150 % and 960x540 at 200 %. The minimum 960x540 keeps the window
    usable everywhere; pages adapt their layout and scroll (GUI_DESIGN.md, 4.4 and 19).
    """
    screen = window.screen()
    if screen is None:
        window.setMinimumSize(*minimum)
        window.resize(*design)
        return
    avail = screen.availableGeometry()
    min_w, min_h = min(minimum[0], avail.width()), min(minimum[1], avail.height() - 40)
    window.setMinimumSize(min_w, min_h)
    w = min(design[0], avail.width())
    h = min(design[1], avail.height() - 40)
    window.resize(max(w, min_w), max(h, min_h))
    frame = window.frameGeometry()
    frame.moveCenter(avail.center())
    window.move(max(frame.left(), avail.left()), max(frame.top(), avail.top()))
