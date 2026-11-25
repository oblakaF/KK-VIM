"""Tools of every interactive X–Y graph of the analysis window (GUI_DESIGN.md 16, 20; GUI-D15, GUI-D20; P5–P6).

One ``ChartTools`` is attached to a graph (sections, history, energy, cycles, D2 sections, load–deflection). It adds
the chart toolbar of 16.8 — ``[⌂ Исходный вид] [+] [−] │ [Snap] [Проекции] … │ [Линии ▾] [Оформление ▾] [Рисунок ▾]`` —
and only ever changes what is drawn:

* **Линии ▾** (16.3): every curve shown or hidden on its own, «Показать все», «Скрыть все»; hiding keeps the data.
* **Snap, Проекции, фиксация** (16.4–16.5): the verified ``xy_interaction.XYInteraction`` of PR #77 on the drawn
  curves; the coordinate field under the plot (16.2).
* **Исходный вид, +, −** (16.1): the initial view, zoom about the centre.
* **Показать исходные значения** (20.3): a right click on a curve (a choice when several are under the pointer) opens
  a read-only table of the STORED values of that curve — solver arrays and records in SI units at full precision, never
  the plotted (mm, kPa) coordinates and never a Snap position.
* **Данные графика, CSV…** (8.4): the stored values of the visible curves (``backend.chart_export``).
* **Развернуть** (20.2): a double click on the title of the graph shows only this graph (with the toolbars of the tab)
  in the tab; another double click restores the layout. A double click on the plot area does not.
* **Оформление ▾** (16.6): colours of the figure, the plot area, the axes and the grid; colour, width, line type and
  markers of every curve; «Сбросить оформление». The screen defaults are the theme tokens; the choices live as long
  as the window.
* **Рисунок для публикации…, Копировать рисунок** (8, 16.7, 20.4): ``figures.publication`` — the visible curves, the
  explicit curve choices of «Оформление», the figure language, white background, the marks of the result and TEST DATA;
  no interaction overlay.

Nothing here computes a result or starts a calculation; the data of the figure are only read.
"""

import math
from dataclasses import dataclass, field, replace
from pathlib import Path

import pyqtgraph as pg
from PySide6.QtCore import QEvent, QObject, QPoint, QPointF, QRect, QSize, Qt, Signal
from PySide6.QtGui import QColor, QGuiApplication, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLayout,
    QMenu,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QSpinBox,
    QStackedWidget,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from kkvim_gui.analysis.xy_interaction import XYInteraction
from kkvim_gui.backend import chart_export
from kkvim_gui.figures import publication as pub

SHOW_VALUES = "Показать исходные значения"
COMBO_CHARS = 12  # [инж.] the active-curve list of a figure of the analysis tabs (the whole name: list and tooltip)
FIGURE_MIN_HEIGHT = 380  # [инж.] below this a figure with its tools scrolls in the tab instead of being squeezed
CURVE_HIT_PX = 8.0  # a right click this close (logical px) to a drawn curve is on it [инж.]
ZOOM_IN, ZOOM_OUT = 0.8, 1.25


# ------------------------------------------------------------------------------------------------ description


@dataclass
class Curve:
    """One curve of a graph: the drawn item, its names and its stored values.

    ``rows()`` returns the stored values of the curve as tuples of the chart's ``columns`` (SI, full precision): read
    from the result, never from the plotted coordinates. ``listed`` False: a decoration (e.g. a connector line) that is
    drawn in the figure but is not a curve of the lists, Snap, the values or the CSV file.
    """

    key: object
    item: object
    name: str
    legend: object  # lang -> legend text ("" for none)
    rows: object = None
    pub: pub.CurveStyle = field(default_factory=pub.CurveStyle)
    listed: bool = True


@dataclass
class ChartInfo:
    figure_id: str  # file name part: section_x, history, …
    texts: object  # lang -> (title, x label, y label)
    axes: object  # () -> ((x name, x unit), (y name, y unit)) of the drawn coordinates (coordinate field)
    columns: tuple = ()  # (name, unit) of the stored values of every curve
    marks: object = None  # lang -> tuple of marks of the result
    test_data: bool = False
    identity: str = ""
    meta: dict = field(default_factory=dict)
    ref_lines: object = None  # lang -> tuple of publication.RefLine
    x_ticks: object = None  # lang -> ((value, text), …) or None
    ranges: object = None  # () -> (x range, y range) of the publication, or None (from the data)


@dataclass
class CurveOverride:
    colour: str | None = None
    width: float | None = None
    dash: str | None = None
    markers: bool | None = None

    def empty(self):
        return self.colour is None and self.width is None and self.dash is None and self.markers is None


@dataclass
class ChartStyle:
    """«Оформление графика» (16.6): explicit choices only; None keeps the standard (theme on screen, publication style
    in the file)."""

    figure_bg: str | None = None
    plot_bg: str | None = None
    axis: str | None = None
    grid: str | None = None
    curves: dict = field(default_factory=dict)  # key -> CurveOverride

    def curve(self, key):
        return self.curves.get(key, CurveOverride())


def segment_distance(p, a, b):
    dx, dy = b.x() - a.x(), b.y() - a.y()
    length2 = dx * dx + dy * dy
    t = 0.0 if length2 == 0 else max(0.0, min(1.0, ((p.x() - a.x()) * dx + (p.y() - a.y()) * dy) / length2))
    return math.hypot(p.x() - a.x() - t * dx, p.y() - a.y() - t * dy)


def pen_dash(pen):
    style = pen.style() if pen is not None else Qt.PenStyle.SolidLine
    return {Qt.PenStyle.DashLine: "dash", Qt.PenStyle.DotLine: "dot", Qt.PenStyle.DashDotLine: "dashdot",
            Qt.PenStyle.NoPen: "none"}.get(style, "solid")


SYMBOL_MAP = {"o": "o", "s": "s", "t": "t", "t1": "t", "d": "d", "+": "+", "x": "x"}


# ------------------------------------------------------------------------------------------------ flow layout


class FlowLayout(QLayout):
    """A layout that wraps its widgets onto the next line when the width is not enough (narrow windows, two graphs)."""

    def __init__(self, parent=None, spacing=6):
        super().__init__(parent)
        self._items = []
        self._spacing = spacing
        self.setContentsMargins(0, 0, 0, 0)

    def addItem(self, item):  # noqa: N802 - Qt API
        self._items.append(item)

    def count(self):
        return len(self._items)

    def itemAt(self, index):  # noqa: N802 - Qt API
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index):  # noqa: N802 - Qt API
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self):  # noqa: N802 - Qt API
        return Qt.Orientation(0)

    def hasHeightForWidth(self):  # noqa: N802 - Qt API
        return True

    def heightForWidth(self, width):  # noqa: N802 - Qt API
        return self._place(QRect(0, 0, width, 0), dry=True)

    def setGeometry(self, rect):  # noqa: N802 - Qt API
        super().setGeometry(rect)
        self._place(rect, dry=False)

    def sizeHint(self):  # noqa: N802 - Qt API
        return self.minimumSize()

    def minimumSize(self):  # noqa: N802 - Qt API
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        return size

    def _place(self, rect, dry):
        x, y, line = rect.x(), rect.y(), 0
        for item in self._items:
            if item.widget() is not None and not item.widget().isVisibleTo(self.parentWidget()):
                continue
            hint = item.sizeHint()
            if x + hint.width() > rect.right() + 1 and line > 0:
                x, y, line = rect.x(), y + line + self._spacing, 0
            if not dry:
                item.setGeometry(QRect(QPoint(x, y), hint))
            x += hint.width() + self._spacing
            line = max(line, hint.height())
        return y + line - rect.y()


# ------------------------------------------------------------------------------------------------ raw values


class RawValuesWindow(QDialog):
    """«Исходные значения»: the stored values of one curve, read only (GUI_DESIGN.md 20.3; never a calculation)."""

    def __init__(self, identity, curve, headers, rows, marks=(), test_data=False, parent=None):
        super().__init__(parent)
        from kkvim_gui.analysis.results import RowsModel, table_view
        from kkvim_gui.widgets.common import Note, label

        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setWindowTitle(f"Исходные значения — {curve}")
        self.curve, self.headers, self.rows = curve, tuple(headers), tuple(tuple(r) for r in rows)
        self.marks = tuple(marks) + (("TEST DATA",) if test_data else ())
        self.identity = f"{identity} · кривая: {curve}" if identity else f"кривая: {curve}"
        lay = QVBoxLayout(self)
        lay.addWidget(label(self.identity, "figureTitle", wrap=True))
        if self.marks:
            lay.addWidget(Note("Отметки результата: " + " · ".join(self.marks) + ". Значения не являются квалифицированным "
                               "научным результатом.", "testdata"))
        lay.addWidget(label(f"Исходные значения: {len(self.rows)} точек кривой — значения решателя и записей расчёта в "
                            "единицах СИ с полной точностью. Интерполированные значения Snap, координаты графика (mm, kPa) "
                            "и пересчёты не включаются.", "secondary", wrap=True))

        class Model(RowsModel):
            def data(self, index, role=Qt.ItemDataRole.DisplayRole):
                if role == Qt.ItemDataRole.DisplayRole and index.isValid():
                    value = self._row(index.row())[index.column()]
                    if isinstance(value, float):
                        return raw_text(value).replace(".", ",")
                return super().data(index, role)

        self.model = Model(self.headers, len(self.rows), lambda i: self.rows[i])
        self.view = table_view(self.model)
        self.view.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectItems)
        self.view.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        QShortcut(QKeySequence.StandardKey.Copy, self.view, self.copy_selection,
                  context=Qt.ShortcutContext.WidgetWithChildrenShortcut)
        lay.addWidget(self.view, 1)
        buttons = QHBoxLayout()
        for text, slot in (("Копировать выделенное", self.copy_selection), ("Копировать всё", self.copy_all)):
            b = QPushButton(text)
            b.clicked.connect(slot)
            buttons.addWidget(b)
        buttons.addStretch(1)
        close = QPushButton("Закрыть")
        close.clicked.connect(self.close)
        buttons.addWidget(close)
        lay.addLayout(buttons)
        self.resize(760, 420)

    def _lines(self, rows, columns, picked=None):
        lines = [f"# {self.identity}"] + ([f"# {' · '.join(self.marks)}"] if self.marks else [])
        lines.append("\t".join(self.headers[c] for c in columns))
        lines += ["\t".join(raw_text(self.rows[r][c]) if picked is None or (r, c) in picked else "" for c in columns)
                  for r in rows]
        return "\n".join(lines) + "\n"

    def text(self):
        return self._lines(range(len(self.rows)), range(len(self.headers)))

    def copy_selection(self):
        picked = {(i.row(), i.column()) for i in self.view.selectionModel().selectedIndexes()}
        if not picked:
            return ""
        text = self._lines(sorted({r for r, _ in picked}), sorted({c for _, c in picked}), picked)
        QGuiApplication.clipboard().setText(text)
        return text

    def copy_all(self):
        text = self.text()
        QGuiApplication.clipboard().setText(text)
        return text


def raw_text(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return repr(value)
    if hasattr(value, "item"):
        return raw_text(value.item())
    return str(value)


# ------------------------------------------------------------------------------------------------ the tools


class ChartTools(QObject):
    """The chart tools of one X–Y graph (see the module text)."""

    shownChanged = Signal(object, bool)
    styleChanged = Signal()
    used = Signal(object)  # this chart was used last (the window's Ctrl+E acts on it)

    def __init__(self, session, figure, plot, title, info, curves, *, interaction=None, readout=True,
                 shown_hook=None, is_shown=None, home=None, context_menu=True, values_hook=None, parent=None):
        """``figure``: the frame of the graph; ``plot``: its pg.PlotWidget; ``title``: the title label (double click
        enlarges); ``curves()``: the Curve descriptions in a fixed order. ``interaction``: an existing XYInteraction
        (the load–deflection graph of PR #77), else one is made with a coordinate field. ``shown_hook(key, on)`` and
        ``is_shown(key)``: the figure manages visibility itself (its own switches stay in step); ``home()``: the
        figure's own «Исходный вид». ``context_menu`` False: the figure keeps its own right-click menu; ``values_hook(key)``:
        the figure's own «Исходные значения» window."""
        super().__init__(parent or figure)
        self.session, self.figure, self.plot, self.title, self.info = session, figure, plot, title, info
        self._curves, self._shown_hook, self._is_shown, self._home_fn = curves, shown_hook, is_shown, home
        self._values_hook = values_hook
        self.style = ChartStyle()
        self._defaults = {}  # key -> captured screen opts of the item (pen, symbol, …)
        self.home = None
        self.enlarged = []  # widgets hidden by «Развернуть»
        self.value_windows = []
        self.view_box = plot.getViewBox()
        own_interaction = interaction is None
        if own_interaction:
            interaction = XYInteraction(plot, self.drawn_curves, info.axes, lambda: session.figure_language,
                                        lambda: session.tokens, self)
            interaction.install_shortcut(figure)
        self.interaction = interaction
        self.toolbar = self._build_toolbar(own_interaction)
        self.readout = None
        if own_interaction and readout:
            from kkvim_gui.widgets.common import label

            self.readout = label("", "secondary")
            self.readout.setWordWrap(True)
            self.readout.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop)
            policy = QSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
            policy.setRetainSizeWhenHidden(True)
            self.readout.setSizePolicy(policy)
            self.readout.setFixedHeight(self.readout.fontMetrics().lineSpacing() * 3 + 4)
            figure.layout().addWidget(self.readout)
            interaction.changed.connect(self._show_readout)
        self._plot_menu = self.view_box.raiseContextMenu
        if context_menu:
            self.view_box.raiseContextMenu = self._raise_context_menu
        self.values_menu = None
        title.installEventFilter(self)
        plot.viewport().installEventFilter(self)
        session.themeChanged.connect(self.reapply)
        self.rehome()
        if own_interaction:  # the first drawn curve is the active Snap curve when the graph is built (as PR #77)
            drawn = interaction.drawn()
            if interaction.active is None and drawn:
                interaction.set_active(drawn[0][0])

    # ------------------------------------------------------------ toolbar (16.8)
    def _build_toolbar(self, own_interaction):
        bar = QWidget()
        bar.setObjectName("ChartToolbar")
        flow = FlowLayout(bar)
        self.home_button = self._tool("⌂ Исходный вид", "Исходный масштаб и положение графика (16.1)", self.go_home)
        self.zoom_in_button = self._tool("+", "Приблизить относительно центра области построения", lambda: self.zoom(ZOOM_IN))
        self.zoom_out_button = self._tool("−", "Отдалить относительно центра области построения", lambda: self.zoom(ZOOM_OUT))
        for w in (self.home_button, self.zoom_in_button, self.zoom_out_button):
            flow.addWidget(w)
        if own_interaction:
            # the controls of PR #77 one by one, so that a narrow figure wraps them instead of cutting them (19.2)
            it = self.interaction
            pair = QWidget()
            pair_lay = QHBoxLayout(pair)
            pair_lay.setContentsMargins(0, 0, 0, 0)
            pair_lay.setSpacing(0)
            pair_lay.addWidget(it.snap_button)
            pair_lay.addWidget(it.projection_button)
            it.combo.setMinimumContentsLength(COMBO_CHARS)
            for w in (pair, it.combo, it.unfix_button):
                flow.addWidget(w)
        self.lines_button = self._menu_button("Линии ▾", "Показать или скрыть кривые; скрытие не удаляет данные (16.3)",
                                              self._fill_lines_menu)
        self.style_button = self._menu_button("Оформление ▾", "Цвета, толщина и тип линий, маркеры; сброс (16.6)",
                                              self._fill_style_menu)
        self.figure_button = self._menu_button("Рисунок ▾", "Рисунок для публикации, копирование, данные графика CSV, "
                                               "исходные значения, развернуть", self._fill_figure_menu)
        for w in (self.lines_button, self.style_button, self.figure_button):
            flow.addWidget(w)
        layout = self.figure.layout()
        layout.insertWidget(1, bar)
        return bar

    def _tool(self, text, tip, slot):
        b = QToolButton()
        b.setText(text)
        b.setToolTip(tip)
        b.clicked.connect(slot)
        b.clicked.connect(lambda: self.used.emit(self))
        return b

    def _menu_button(self, text, tip, fill):
        b = QToolButton()
        b.setText(text)
        b.setToolTip(tip)
        b.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        menu = QMenu(b)
        menu.aboutToShow.connect(lambda m=menu: fill(m))
        menu.aboutToShow.connect(lambda: self.used.emit(self))
        b.setMenu(menu)
        return b

    def _fill_lines_menu(self, menu):
        menu.clear()
        for c in self.curves():
            action = menu.addAction(c.name)
            action.setCheckable(True)
            action.setChecked(self.is_shown(c.key))
            action.setEnabled(self.has_points(c))
            if not self.has_points(c):
                action.setToolTip("Нет значений для этой кривой")
            action.toggled.connect(lambda on, k=c.key: self.set_shown(k, on))
        menu.addSeparator()
        menu.addAction("Показать все", lambda: self.show_all(True))
        menu.addAction("Скрыть все", lambda: self.show_all(False))

    def _fill_style_menu(self, menu):
        menu.clear()
        menu.addAction("Настроить оформление…", self.open_style_dialog)
        reset = menu.addAction("Сбросить оформление", self.reset_style)
        reset.setEnabled(not self.style_is_default())

    def _fill_figure_menu(self, menu):
        menu.clear()
        menu.addAction("Рисунок для публикации…", self.open_publication_dialog).setShortcut(QKeySequence("Ctrl+E"))
        menu.addAction("Копировать рисунок", self.copy_figure)
        menu.addAction("Данные графика, CSV…", self._csv_dialog)
        values = menu.addMenu(SHOW_VALUES)
        shown = [c for c in self.curves() if self.is_shown(c.key) and self.has_points(c) and c.rows is not None]
        for c in shown:
            values.addAction(c.name, lambda k=c.key: self.show_values(k))
        values.setEnabled(bool(shown))
        menu.addSeparator()
        menu.addAction("Восстановить расположение" if self.enlarged else "Развернуть график", self.toggle_enlarged)

    # ------------------------------------------------------------ curves and visibility
    def curves(self):
        return [c for c in self._curves() if c.listed]

    def all_curves(self):
        return list(self._curves())

    def curve(self, key):
        return next((c for c in self.all_curves() if c.key == key), None)

    @staticmethod
    def has_points(c):
        xs, _ = c.item.getOriginalDataset() if hasattr(c.item, "getOriginalDataset") else c.item.getData()
        if xs is not None and len(xs):
            return True
        return c.rows is not None and bool(c.rows())

    def drawn_curves(self):
        """(key, name, item) of the listed curves for Snap (XYInteraction keeps only the drawn ones)."""
        return [(c.key, c.name, c.item) for c in self.curves()]

    def is_shown(self, key):
        if self._is_shown is not None:
            return bool(self._is_shown(key))
        c = self.curve(key)
        return c is not None and c.item.isVisible()

    def set_shown(self, key, on):
        """Show or hide one curve: drawing only; the data, the view and the other curves stay."""
        on = bool(on)
        if self._shown_hook is not None:
            self._shown_hook(key, on)
        else:
            c = self.curve(key)
            if c is None:
                return
            c.item.setVisible(on)
        self.interaction.curves_changed()
        self.shownChanged.emit(key, on)

    def show_all(self, on):
        for c in self.curves():
            if self.has_points(c) and self.is_shown(c.key) != on:
                self.set_shown(c.key, on)

    def visible_curves(self):
        return [c for c in self.curves() if self.is_shown(c.key)]

    # ------------------------------------------------------------ view (16.1)
    def rehome(self):
        """The data changed (built, other node line): the initial view is the full range of the new data."""
        if self._home_fn is not None:
            self.home = None
            return
        self.view_box.autoRange()
        self.home = tuple(tuple(r) for r in self.view_box.viewRange())

    def go_home(self):
        if self._home_fn is not None:
            self._home_fn()
            return
        if self.home is None:
            self.rehome()
            return
        (x0, x1), (y0, y1) = self.home
        self.view_box.setRange(xRange=(x0, x1), yRange=(y0, y1), padding=0)

    def zoom(self, factor):
        centre = self.view_box.viewRect().center()
        self.view_box.scaleBy((factor, factor), center=centre)

    def view_range(self):
        return tuple(tuple(r) for r in self.view_box.viewRange())

    # ------------------------------------------------------------ coordinate field, context menu, enlarging
    def _show_readout(self):
        if self.readout is not None:
            text = self.interaction.readout()
            self.readout.setText(text)
            self.readout.setToolTip(text)

    def curves_at(self, scene_pos, radius=CURVE_HIT_PX):
        hits = []
        for c in self.curves():
            if not self.is_shown(c.key):
                continue
            xs, ys = c.item.getData()
            if xs is None or not len(xs):
                continue
            pts = [self.view_box.mapViewToScene(QPointF(float(x), float(y))) for x, y in zip(xs, ys, strict=True)]
            if min(segment_distance(scene_pos, a, b) for a, b in zip(pts, pts[1:] or pts)) <= radius:
                hits.append(c.key)
        return hits

    def _raise_context_menu(self, ev):
        self.used.emit(self)
        keys = [k for k in self.curves_at(ev.scenePos()) if self.curve(k).rows is not None]
        menu = QMenu(self.plot)
        if keys:
            if len(keys) == 1:
                menu.addAction(SHOW_VALUES, lambda k=keys[0]: self.show_values(k)).setToolTip(self.curve(keys[0]).name)
            else:
                sub = menu.addMenu(SHOW_VALUES)
                for k in keys:
                    sub.addAction(self.curve(k).name, lambda key=k: self.show_values(key))
            menu.addSeparator()
        self._fill_figure_actions(menu)
        menu.addSeparator()
        menu.addAction("Меню графика pyqtgraph…", lambda: self._plot_menu(ev))
        self.values_menu = menu
        menu.popup(ev.screenPos().toPoint())
        return None

    def _fill_figure_actions(self, menu):
        menu.addAction("Копировать рисунок", self.copy_figure)
        menu.addAction("Рисунок для публикации…", self.open_publication_dialog)
        menu.addAction("Данные графика, CSV…", self._csv_dialog)
        menu.addAction("Восстановить расположение" if self.enlarged else "Развернуть график", self.toggle_enlarged)

    def eventFilter(self, obj, ev):
        kind = ev.type()
        if obj is self.title and kind == QEvent.Type.MouseButtonDblClick:
            self.toggle_enlarged()
            return True
        if kind == QEvent.Type.MouseButtonPress:
            self.used.emit(self)
        return False

    def _tab_page(self):
        """The page of the tab that holds the figure (its parent is the QStackedWidget of a QTabWidget)."""
        w = self.figure
        while w.parentWidget() is not None:
            parent = w.parentWidget()
            if isinstance(parent, QStackedWidget) and isinstance(parent.parentWidget(), QTabWidget):
                return w
            w = parent
        return w

    def toggle_enlarged(self):
        """20.2: only this graph (and the tab toolbars) in the tab; again: the previous layout."""
        if self.enlarged:
            for w in self.enlarged:
                w.setVisible(True)
            self.enlarged = []
            return False
        root = self._tab_page()
        w = self.figure
        while w is not root and w.parentWidget() is not None:
            parent = w.parentWidget()
            for sibling in parent.findChildren(QWidget, options=Qt.FindChildOption.FindDirectChildrenOnly):
                if sibling is w or not sibling.isVisible() or sibling.isWindow():
                    continue
                if sibling.objectName() == "TabToolbar" or sibling.inherits("QSplitterHandle"):
                    continue
                sibling.setVisible(False)
                self.enlarged.append(sibling)
            w = parent
        return True

    # ------------------------------------------------------------ values and CSV (20.3, 8.4)
    def headers(self):
        return tuple(chart_export.header_cell(n, u).replace("[", "").replace("]", "").replace(" ", ", ", 1)
                     if u else n for n, u in self.info.columns)

    def marks(self, lang=None):
        return tuple(self.info.marks(lang or self.session.figure_language)) if self.info.marks else ()

    def show_values(self, key):
        """The read-only window of the stored values of the shown curve ``key`` (None: hidden or without values)."""
        c = self.curve(key)
        if c is None or c.rows is None or not self.is_shown(key):
            return None
        if self._values_hook is not None:
            return self._values_hook(key)
        rows = c.rows()
        if not rows:
            return None
        window = RawValuesWindow(self.info.identity, c.name, self.headers(), rows, self.marks("en"),
                                 self.info.test_data, self.figure)
        self.value_windows = [w for w in self.value_windows if w.isVisible()] + [window]
        window.show()
        return window

    def write_csv(self, path):
        """The stored values of the visible curves (backend.chart_export)."""
        curves = [(c.legend("en") or c.name, c.rows()) for c in self.visible_curves() if c.rows is not None]
        title = self.info.texts("en")[0]
        comments = [f"# {self.info.identity}" if self.info.identity else "", f"# figure: {title}",
                    *(f"# mark: {m}" for m in self.marks("en"))]
        return chart_export.write_curves_csv(path, self.info.columns, curves, [c for c in comments if c],
                                             self.info.test_data)

    def base_name(self):
        window = self.figure.window()
        base = getattr(window, "export_base", None)
        return base() if callable(base) else "kkvim"

    def _csv_dialog(self):
        name = f"{self.base_name()}_{self.info.figure_id}{'_TEST-DATA' if self.info.test_data else ''}.csv"
        path, _ = QFileDialog.getSaveFileName(self.figure, "Данные графика", name, "CSV (*.csv)")
        if path:
            self.write_csv(path)

    # ------------------------------------------------------------ style (16.6)
    def _capture(self, c):
        if c.key not in self._defaults:
            opts = c.item.opts
            self._defaults[c.key] = {k: opts.get(k) for k in ("pen", "symbol", "symbolBrush", "symbolPen", "symbolSize")}

    def style_is_default(self):
        s = self.style
        return (s.figure_bg is None and s.plot_bg is None and s.axis is None and s.grid is None
                and all(o.empty() for o in s.curves.values()))

    def set_style(self, style):
        """Apply «Оформление графика»: every explicit choice on the screen (the rest keeps the theme)."""
        self.style = style
        self.reapply()
        self.styleChanged.emit()

    def reset_style(self):
        """«Сбросить оформление»: the standard settings (theme tokens on screen, publication style in files)."""
        for c in self.all_curves():
            d = self._defaults.get(c.key)
            if d is not None:
                c.item.setPen(d["pen"])
                c.item.setSymbol(d["symbol"])
                c.item.setSymbolBrush(d["symbolBrush"])
                c.item.setSymbolPen(d["symbolPen"])
                c.item.setSymbolSize(d["symbolSize"])
        self.style = ChartStyle()
        self.figure.setStyleSheet("")
        restyle = getattr(self.figure, "restyle", None)
        if restyle is not None:
            restyle()
        self.styleChanged.emit()

    def reapply(self, *_):
        s = self.style
        if s.figure_bg:
            self.figure.setStyleSheet(f'QFrame[figure="true"] {{ background: {s.figure_bg}; }}')
        if s.plot_bg:
            self.plot.setBackground(s.plot_bg)
        for side in ("left", "bottom"):
            axis = self.plot.getAxis(side)
            if s.axis:
                axis.setPen(s.axis)
                axis.setTextPen(s.axis)
            if s.grid:
                axis.setTickPen(pg.mkPen(s.grid))
        for c in self.all_curves():
            o = s.curve(c.key)
            if o.empty():
                continue
            self._capture(c)
            d = self._defaults[c.key]
            base = pg.mkPen(d["pen"]) if d["pen"] is not None else pg.mkPen(None)
            colour = o.colour or base.color().name()
            width = o.width if o.width is not None else base.widthF()
            dash = o.dash or pen_dash(base)
            pen = pg.mkPen(colour, width=width, style=pub.DASHES.get(dash, Qt.PenStyle.SolidLine)) if dash != "none" \
                else pg.mkPen(None)
            c.item.setPen(pen)
            markers = o.markers if o.markers is not None else d["symbol"] is not None
            c.item.setSymbol((d["symbol"] or "o") if markers else None)
            if markers:
                c.item.setSymbolBrush(colour if o.colour else d["symbolBrush"])
                c.item.setSymbolPen(colour if o.colour else d["symbolPen"])

    def open_style_dialog(self):
        dialog = StyleDialog(self, self.figure)
        dialog.open()
        return dialog

    # ------------------------------------------------------------ publication (8, 16.7, 20.4)
    def publication_style(self, c, index):
        """The style of a curve in the file: the explicit choices of «Оформление», else the publication standard."""
        base = c.pub
        o = self.style.curve(c.key)
        marker = base.marker
        if o.markers is not None:
            item_symbol = c.item.opts.get("symbol") if self._defaults.get(c.key) is None else self._defaults[c.key]["symbol"]
            marker = (SYMBOL_MAP.get(item_symbol, "o") if item_symbol else base.marker or "o") if o.markers else None
        return replace(base, colour=o.colour or base.colour, width=o.width if o.width is not None else base.width,
                       dash=o.dash or base.dash, marker=marker)

    def spec(self, lang=None):
        """The publication description of the graph as shown: visible curves, chosen styles, the figure language."""
        lang = lang or self.session.figure_language
        title, x_label, y_label = self.info.texts(lang)
        series = []
        for index, c in enumerate(self.all_curves()):
            if c.listed and not self.is_shown(c.key):
                continue
            if not c.listed and not c.item.isVisible():
                continue
            xs, ys = c.item.getOriginalDataset() if hasattr(c.item, "getOriginalDataset") else c.item.getData()
            if xs is None or not len(xs):
                continue
            series.append(pub.Series(c.legend(lang) if c.listed else "", tuple(float(v) for v in xs),
                                     tuple(float(v) for v in ys), self.publication_style(c, index)))
        axis = self.plot.getAxis("left")
        ranges = self.info.ranges() if self.info.ranges else None
        return pub.XYSpec(
            title=title, x_label=x_label, y_label=y_label, series=tuple(series),
            ref_lines=tuple(self.info.ref_lines(lang)) if self.info.ref_lines else (),
            log_y=bool(axis.logMode), invert_y=bool(self.view_box.yInverted()),
            x_ticks=self.info.x_ticks(lang) if self.info.x_ticks else None,
            x_range=ranges[0] if ranges else None, y_range=ranges[1] if ranges else None,
            marks=self.marks(lang), test_data=self.info.test_data, lang=lang,
            meta={**self.info.meta, "figure": self.info.figure_id, "run": self.base_name()})

    def copy_figure(self, lang=None):
        """20.4: the publication figure (raster and SVG) in the clipboard; no interaction overlay."""
        data = pub.mime_data(self.spec(lang))
        QGuiApplication.clipboard().setMimeData(data)
        return data

    def open_publication_dialog(self):
        dialog = PublicationDialog(self.session, self, self.figure)
        dialog.open()
        return dialog


# ------------------------------------------------------------------------------------------------ dialogs


class ColourButton(QPushButton):
    """A colour choice: the swatch and the hex code; «стандарт» when nothing is chosen."""

    changed = Signal()

    def __init__(self, colour=None, parent=None):
        super().__init__(parent)
        self.colour = colour
        self.clicked.connect(self._pick)
        self._show()

    def _pick(self):
        chosen = QColorDialog.getColor(QColor(self.colour or "#000000"), self, "Цвет")
        if chosen.isValid():
            self.set_colour(chosen.name())

    def set_colour(self, colour):
        self.colour = colour
        self._show()
        self.changed.emit()

    def _show(self):
        if self.colour:
            pix = QPixmap(14, 14)
            pix.fill(QColor(self.colour))
            from PySide6.QtGui import QIcon

            self.setIcon(QIcon(pix))
            self.setText(self.colour)
        else:
            from PySide6.QtGui import QIcon

            self.setIcon(QIcon())
            self.setText("стандарт")


class StyleDialog(QDialog):
    """«Оформление графика» (16.6): explicit choices of the figure and of every curve; «Сбросить оформление»."""

    def __init__(self, tools, parent=None):
        super().__init__(parent)
        self.tools = tools
        self.setWindowTitle("Оформление графика")
        s = tools.style
        lay = QVBoxLayout(self)
        grid = QGridLayout()
        self.colours = {}
        for row, (key, text) in enumerate((("figure_bg", "Фон всего рисунка"), ("plot_bg", "Фон области построения"),
                                           ("axis", "Цвет осей и подписей"), ("grid", "Цвет линий сетки"))):
            grid.addWidget(QLabel(text), row, 0)
            b = ColourButton(getattr(s, key))
            self.colours[key] = b
            grid.addWidget(b, row, 1)
        lay.addLayout(grid)
        curves = QGridLayout()
        for col, text in enumerate(("Кривая", "Цвет", "Толщина", "Тип линии", "Маркеры")):
            curves.addWidget(QLabel(text), 0, col)
        self.rows = {}
        for r, c in enumerate(tools.curves(), start=1):
            o = s.curve(c.key)
            colour = ColourButton(o.colour)
            width = QDoubleSpinBox()
            width.setRange(0.0, 6.0)
            width.setSingleStep(0.25)
            width.setSpecialValueText("стандарт")
            width.setValue(o.width or 0.0)
            dash = QComboBox()
            dash.addItem("стандарт", None)
            for k, v in pub.DASH_NAMES.items():
                dash.addItem(v, k)
            dash.setCurrentIndex(max(dash.findData(o.dash), 0))
            markers = QComboBox()
            for text, value in (("стандарт", None), ("вкл.", True), ("выкл.", False)):
                markers.addItem(text, value)
            markers.setCurrentIndex(max(markers.findData(o.markers), 0) if o.markers is not None else 0)
            curves.addWidget(QLabel(c.name), r, 0)
            for col, w in enumerate((colour, width, dash, markers), start=1):
                curves.addWidget(w, r, col)
            self.rows[c.key] = (colour, width, dash, markers)
        lay.addLayout(curves)
        lay.addWidget(QLabel("Экран: стандарт — цвета темы. Файл публикации: стандарт — публикационный стиль "
                             "(белый фон, палитра Okabe–Ito); явно выбранные цвета, толщины, типы линий и маркеры кривых "
                             "переносятся в экспорт; фоны экспорта всегда белые (16.7)."))
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
                                   | QDialogButtonBox.StandardButton.Apply | QDialogButtonBox.StandardButton.Reset)
        buttons.button(QDialogButtonBox.StandardButton.Reset).setText("Сбросить оформление")
        buttons.button(QDialogButtonBox.StandardButton.Apply).setText("Применить")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("Отмена")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(self.apply)
        buttons.button(QDialogButtonBox.StandardButton.Reset).clicked.connect(self.reset)
        lay.addWidget(buttons)

    def chosen(self):
        curves = {}
        for key, (colour, width, dash, markers) in self.rows.items():
            o = CurveOverride(colour.colour, width.value() or None, dash.currentData(), markers.currentData())
            if not o.empty():
                curves[key] = o
        return ChartStyle(*(self.colours[k].colour for k in ("figure_bg", "plot_bg", "axis", "grid")), curves=curves)

    def apply(self):
        self.tools.reset_style()
        self.tools.set_style(self.chosen())

    def accept(self):
        self.apply()
        super().accept()

    def reset(self):
        self.tools.reset_style()
        for b in self.colours.values():
            b.set_colour(None)
        for colour, width, dash, markers in self.rows.values():
            colour.set_colour(None)
            width.setValue(0.0)
            dash.setCurrentIndex(0)
            markers.setCurrentIndex(0)


class PublicationDialog(QDialog):
    """«Рисунок для публикации…» (GUI_DESIGN.md 8.3): format, dpi, width, figure language, preview, save."""

    def __init__(self, session, source, parent=None):
        """``source``: an object with ``spec(lang)``, ``base_name()`` and ``info.figure_id``/``info.test_data``."""
        super().__init__(parent)
        self.session, self.source = session, source
        self.setWindowTitle("Рисунок для публикации")
        self.lang = session.figure_language
        lay = QVBoxLayout(self)
        form = QGridLayout()
        self.format = QComboBox()
        for f in ("png", "pdf", "svg"):
            self.format.addItem(f.upper(), f)
        self.dpi = QComboBox()
        for d in pub.DPI_CHOICES:
            self.dpi.addItem(f"{d} dpi", d)
        self.width_box = QComboBox()
        self.width_box.addItem("85 mm — одна колонка", pub.WIDTHS_MM["one_column"])
        self.width_box.addItem("175 mm — две колонки", pub.WIDTHS_MM["two_columns"])
        self.width_box.addItem("своя ширина", None)
        self.custom = QSpinBox()
        self.custom.setRange(40, 300)
        self.custom.setValue(120)
        self.custom.setSuffix(" mm")
        self.lang_ru = QRadioButton("RU")
        self.lang_en = QRadioButton("EN")  # 21 F1: RU on the left, EN on the right
        (self.lang_ru if self.lang == "ru" else self.lang_en).setChecked(True)
        lang_box = QHBoxLayout()
        lang_box.addWidget(self.lang_ru)
        lang_box.addWidget(self.lang_en)
        lang_box.addStretch(1)
        for row, (text, w) in enumerate((("Формат", self.format), ("Разрешение PNG", self.dpi), ("Ширина", self.width_box),
                                         ("", self.custom))):
            form.addWidget(QLabel(text), row, 0)
            form.addWidget(w, row, 1)
        form.addWidget(QLabel("Язык подписей"), 4, 0)
        form.addLayout(lang_box, 4, 1)
        lay.addLayout(form)
        self.preview = QLabel()
        self.preview.setMinimumSize(420, 280)
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setFrameShape(QFrame.Shape.StyledPanel)
        lay.addWidget(self.preview, 1)
        note = QLabel("Белый фон независимо от темы; только видимые кривые с выбранным оформлением; без курсора, Snap, "
                      "проекций и фиксации. Демонстрационные данные — водяной знак TEST DATA.")
        note.setWordWrap(True)
        lay.addWidget(note)
        buttons = QDialogButtonBox()
        self.save_button = buttons.addButton("Сохранить…", QDialogButtonBox.ButtonRole.AcceptRole)
        self.copy_button = buttons.addButton("Копировать рисунок", QDialogButtonBox.ButtonRole.ActionRole)
        buttons.addButton("Закрыть", QDialogButtonBox.ButtonRole.RejectRole)
        self.save_button.clicked.connect(self._save_dialog)
        self.copy_button.clicked.connect(lambda: QGuiApplication.clipboard().setMimeData(
            pub.mime_data(self.spec(), self.dpi.currentData(), self.width_mm())))
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)
        for signal in (self.format.currentIndexChanged, self.width_box.currentIndexChanged, self.custom.valueChanged,
                       self.lang_ru.toggled, self.dpi.currentIndexChanged):
            signal.connect(self.refresh)
        self.refresh()

    def chosen_lang(self):
        return "ru" if self.lang_ru.isChecked() else "en"

    def width_mm(self):
        return self.width_box.currentData() or float(self.custom.value())

    def spec(self):
        return self.source.spec(self.chosen_lang())

    def file_name(self):
        info = self.source.info
        return pub.file_name(self.source.base_name(), info.figure_id, self.chosen_lang(), info.test_data,
                             self.format.currentData())

    def refresh(self, *_):
        self.dpi.setEnabled(self.format.currentData() == "png")
        self.custom.setEnabled(self.width_box.currentData() is None)
        ratio = self.devicePixelRatioF() or 1.0
        box = self.preview.size() - QSize(8, 8)
        # rendered at the pixel density of the screen: the preview is sharp at every Windows scale (19.1)
        dpi = max(60.0, min(600.0, box.width() * ratio / (self.width_mm() / pub.MM_PER_IN)))
        pix = QPixmap.fromImage(pub.image(self.spec(), dpi, self.width_mm()))
        pix = pix.scaled(box * ratio, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        pix.setDevicePixelRatio(ratio)
        self.preview.setPixmap(pix)

    def save(self, path):
        return pub.save(self.spec(), path, self.format.currentData(), self.dpi.currentData(), self.width_mm())

    def _save_dialog(self):
        fmt = self.format.currentData()
        path, _ = QFileDialog.getSaveFileName(self, "Рисунок для публикации", self.file_name(), f"{fmt.upper()} (*.{fmt})")
        if not path:
            return
        try:
            self.save(path)
        except OSError as exc:
            QMessageBox.warning(self, "Рисунок не сохранён", str(exc))
            return
        window = self.parent().window() if self.parent() is not None else None
        if window is not None and hasattr(window, "statusBar"):
            window.statusBar().showMessage(f"Сохранено: {Path(path)}", 8000)


class FieldPublisher(QObject):
    """Publication export of the static colour field (no X–Y tools: GUI-D13)."""

    def __init__(self, session, figure, identity="", parent=None):
        super().__init__(parent or figure)
        self.session, self.figure = session, figure
        outcome = figure.outcome
        self.info = ChartInfo("field", None, None, test_data=outcome.is_demo, identity=identity)

    def base_name(self):
        window = self.figure.window()
        base = getattr(window, "export_base", None)
        return base() if callable(base) else "kkvim"

    def spec(self, lang=None):
        from kkvim_gui.figures.text import fig_text

        lang = lang or self.session.figure_language
        f, o = self.figure, self.figure.outcome
        wq = f.quantity == "w"
        return pub.FieldSpec(
            title=fig_text(lang, "field.w.title" if wq else "field.wbar.title"),
            x_label=fig_text(lang, "axis.x"), y_label=fig_text(lang, "axis.y"),
            colorbar_label=fig_text(lang, "colorbar.w" if wq else "colorbar.wbar"),
            x=tuple(float(v) for v in o.x), y=tuple(float(v) for v in o.y), values=f.values().copy(),
            cmap=f.cmap_name, isolines=f.ISOLINES if f.show_isolines else 0, marks=figure_marks(o, lang),
            test_data=o.is_demo, lang=lang,
            meta={"figure": "field", "run": self.base_name(), "solver": o.solver, "data_origin": o.origin,
                  "values": "stored solver field on the result nodes; colour scale only"})

    def copy_figure(self, lang=None):
        data = pub.mime_data(self.spec(lang))
        QGuiApplication.clipboard().setMimeData(data)
        return data

    def open_publication_dialog(self):
        dialog = PublicationDialog(self.session, self, self.figure)
        dialog.open()
        return dialog


def figure_marks(outcome, lang):
    """The marks of a result on a figure (diagnostic, research, provisional) in the figure language."""
    from kkvim_gui.figures.text import fig_text

    marks = []
    if outcome.is_error or outcome.diagnostic_only:
        marks.append(fig_text(lang, "mark.diagnostic"))
    if outcome.mark:
        marks.append(fig_text(lang, "mark.research" if outcome.mark.startswith("RESEARCH_ONLY") else "mark.provisional"))
    return tuple(marks)


def tools_of(widget):
    """The chart tools and field publishers inside ``widget`` (a tab), in creation order."""
    return [t for t in widget.findChildren(QObject) if isinstance(t, (ChartTools, FieldPublisher))]


def default_style(index, dash="solid", marker="o", reference=False):
    """The publication standard of the curve ``index`` of a graph: Okabe–Ito colours; references dashed, black."""
    if reference:
        return pub.CurveStyle(pub.BLACK, 0.9, "dash", marker if marker else None)
    return pub.CurveStyle(pub.OKABE_ITO[index % len(pub.OKABE_ITO)], 1.0, dash, marker)


def check_box_sync(boxes):
    """A slot keeping figure switches (QCheckBox per key) in step with the chart tools (signals blocked)."""
    def sync(key, on):
        box = boxes.get(key)
        if isinstance(box, QCheckBox) and box.isChecked() != on:
            box.blockSignals(True)
            box.setChecked(on)
            box.blockSignals(False)
    return sync


def outcome_identity(outcome):
    """One line naming the result of a figure (values windows and CSV comments)."""
    r = outcome.request
    return (f"{r.supports}, N = {r.N}, сетка {r.n_x} × {r.n_y}, q = {r.q!r} Pa; решатель {outcome.solver}, "
            f"статус {outcome.status}, s = {outcome.s}")


def outcome_meta(outcome):
    return {"solver": outcome.solver, "solver_status": outcome.status, "data_origin": outcome.origin,
            "mark": outcome.mark, "field_role": outcome.field_role, "s": outcome.s}
