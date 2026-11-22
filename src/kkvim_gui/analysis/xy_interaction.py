"""Snap, projections and a fixed point on an interactive X–Y plot (GUI_DESIGN.md 16.4–16.5, GUI-D15; P5).

Everything here is a graphical measurement on what is drawn — never a solver value and never a record of a result:

* **Snap** puts a measurement marker on the drawn polyline of one *active* curve (the data the plot item displays,
  ``PlotDataItem.getData()``), at the nearest point by distance on the screen in logical pixels; between two drawn
  nodes the marker moves along the segment (linear interpolation in display coordinates, as the line is drawn). The
  search starts at the segments next to the previous marker position and moves to a remote part of the curve only when
  that part is closer to the pointer by more than ``HYSTERESIS_PX`` [инж.]. Beyond ``SNAP_PX`` there is no marker
  [инж., 16.4: «около 20 логических пикселей»]. The system pointer is never moved.
* **Projections** are two dashed lines from the measurement position (the marker with Snap, the cursor without it)
  to the X and Y axes with the coordinates written on the axes.
* **Fixation**: a left click without dragging (the press and release within ``CLICK_PX``, 16.1 «около 4 логических
  пикселей», and no change of the view) fixes the current measurement position; another click moves it; Esc or
  «Снять фиксацию» removes it.

Nothing of this enters a table, a CSV file, a saved figure or a solver artifact; the drawn data are only read.
"""

import math
from contextlib import contextmanager
from dataclasses import dataclass

import pyqtgraph as pg
from PySide6.QtCore import QEvent, QObject, QPointF, Qt, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QPushButton, QSizePolicy, QWidget

SNAP_PX = 20.0  # no snap farther than this from the active curve (logical px) [инж., GUI_DESIGN.md 16.4]
HYSTERESIS_PX = 6.0  # a remote segment wins only when closer by more than this [инж.]: above hand jitter, below SNAP_PX
NODE_PX = 4.0  # within the drawn node symbol (8 px) the marker is on the node itself [инж.]
CLICK_PX = 4.0  # press and release closer than this, without a change of the view, is a click [инж., 16.1]
NO_ACTIVE = "— активная кривая не выбрана —"
NOT_A_RESULT = "графическое измерение на линии графика, не значение решателя"
FIGURE_WORDS = {"en": {"node": "node {i}", "interp": "interpolated {i}–{j}", "cursor": "cursor"},
                "ru": {"node": "узел {i}", "interp": "интерполяция {i}–{j}", "cursor": "курсор"}}


# ---------------------------------------------------------------- geometry in display (screen) coordinates, Qt-free

def segment_projection(p, a, b):
    """(t, distance) of the point p projected onto the segment a–b (t in [0, 1]; a point when a == b)."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    length2 = dx * dx + dy * dy
    t = 0.0 if length2 == 0 else max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / length2))
    return t, math.hypot(p[0] - a[0] - t * dx, p[1] - a[1] - t * dy)


def _segments(n):
    """Segment indices of a polyline of n nodes (one degenerate segment for a single node, none for no node)."""
    return range(max(n - 1, 1)) if n else range(0)


def _best(pixels, p, indices):
    best = None
    for k in indices:
        t, d = segment_projection(p, pixels[k], pixels[min(k + 1, len(pixels) - 1)])
        if best is None or d < best[2]:
            best = (k, t, d)
    return best


@dataclass(frozen=True)
class SnapPoint:
    """The marker on the drawn polyline: segment k (nodes k and k + 1, 0-based), parameter t, node or None."""

    segment: int
    t: float
    node: int | None  # 0-based drawn node when the marker coincides with it, else None (interpolation)
    distance: float  # from the pointer, logical px

    def describe(self):
        """«узел i» or «интерполяция между узлами i и i+1» (1-based, in the drawn order)."""
        if self.node is not None:
            return f"узел {self.node + 1}"
        return f"интерполяция между узлами {self.segment + 1} и {self.segment + 2}"

    def figure_words(self, lang):
        words = FIGURE_WORDS.get(lang, FIGURE_WORDS["en"])
        if self.node is not None:
            return words["node"].format(i=self.node + 1)
        return words["interp"].format(i=self.segment + 1, j=self.segment + 2)


def snap(pixels, p, previous=None, threshold=SNAP_PX, hysteresis=HYSTERESIS_PX, node_px=NODE_PX):
    """The SnapPoint of the pointer p on the polyline ``pixels`` (screen points of the drawn nodes), or None.

    ``previous`` is the segment of the previous marker on the same curve: its neighbourhood (segments previous − 1 …
    previous + 1) is kept unless another part of the curve is closer by more than ``hysteresis``. None beyond
    ``threshold`` and for an empty polyline.
    """
    n = len(pixels)
    if n == 0:
        return None
    best = _best(pixels, p, _segments(n))
    if previous is not None and 0 <= previous < len(_segments(n)):
        local = _best(pixels, p, range(max(previous - 1, 0), min(previous + 2, len(_segments(n)))))
        if local[2] <= threshold and local[2] - best[2] <= hysteresis:
            best = local
    k, t, d = best
    if d > threshold:
        return None
    a, b = pixels[k], pixels[min(k + 1, n - 1)]
    on = (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
    if n == 1 or math.hypot(on[0] - a[0], on[1] - a[1]) <= node_px:
        return SnapPoint(k, 0.0, k, d)
    if math.hypot(on[0] - b[0], on[1] - b[1]) <= node_px:
        return SnapPoint(k, 1.0, k + 1, d)
    return SnapPoint(k, t, None, d)


def format_value(value, lang, digits=6):
    """A coordinate for the figure language: 6 significant digits, decimal comma in RU (GUI_DESIGN.md 7.3, 16.2)."""
    text = f"{value:.{digits}g}"
    return text.replace(".", ",") if lang == "ru" else text


# ---------------------------------------------------------------- Qt part

@dataclass(frozen=True)
class Measurement:
    """A measurement position in view coordinates: on the active curve (snap) or the cursor itself (snap None)."""

    x: float
    y: float
    key: object = None
    name: str = ""
    snap: SnapPoint | None = None


class _Inert:
    """An overlay item that is drawn only: never a click, drag or hover target (the curves keep their clicks)."""

    def _inert(self):
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.setAcceptHoverEvents(False)

    def mouseClickEvent(self, ev):
        ev.ignore()

    def mouseDragEvent(self, ev):
        ev.ignore()

    def hoverEvent(self, ev):
        pass


class OverlayScatter(_Inert, pg.ScatterPlotItem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._inert()


class OverlayLine(_Inert, pg.PlotCurveItem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._inert()


class OverlayText(_Inert, pg.TextItem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._inert()


class _Projection:
    """Two dashed lines from a point to the axes and the two coordinates on the axes."""

    def __init__(self, plot, z):
        self.lines = (OverlayLine(), OverlayLine())
        self.texts = (OverlayText(anchor=(0.5, 1.0)), OverlayText(anchor=(0.0, 0.5)))
        for item in (*self.lines, *self.texts):
            item.setZValue(z)
            plot.addItem(item, ignoreBounds=True)
        self.hide()

    def items(self):
        return (*self.lines, *self.texts)

    def hide(self):
        for item in self.items():
            item.setVisible(False)

    def show(self, x, y, view_range, x_text, y_text):
        (x0, _), (y0, _) = view_range
        self.lines[0].setData([x, x], [y, y0])
        self.lines[1].setData([x, x0], [y, y])
        self.texts[0].setText(x_text)
        self.texts[0].setPos(x, y0)
        self.texts[1].setText(y_text)
        self.texts[1].setPos(x0, y)
        for item in self.items():
            item.setVisible(True)

    def style(self, colour, background):
        for line in self.lines:
            line.setPen(pg.mkPen(colour, width=1, style=Qt.PenStyle.DashLine))
        for text in self.texts:
            text.setColor(colour)
            text.fill = pg.mkBrush(background)
            text.update()


class XYInteraction(QObject):
    """Snap, projections and the fixed point of one pg.PlotWidget; the plot and its curves are only read.

    ``curves()`` gives, in a fixed order, (key, name, plot item) of the curves that are drawn now (visible, with
    points); ``axes()`` gives ((x name, x unit), (y name, y unit)) of the drawn quantity; ``language()`` the figure
    language. ``changed`` is emitted after every change of the measurement, the fixation or the controls.
    """

    changed = Signal()

    def __init__(self, plot, curves, axes, language, tokens, parent=None):
        super().__init__(parent)
        self.plot, self._curves, self._axes, self._language, self._tokens = plot, curves, axes, language, tokens
        self.view_box = plot.getViewBox()
        self.snap_on = self.projections_on = False
        self.active = None  # key of the active curve; None: Snap does nothing until a curve is chosen
        self.measurement = None  # the current measurement position (None: none valid)
        self.fixed = None  # the fixed Measurement (an interface overlay only)
        self._pos = None  # last pointer position (scene) inside the plot area
        self._previous = None  # (key, segment) of the last marker, for continuity along the curve
        self._press = None  # (position, view range) of a left press on the plot
        self.marker = OverlayScatter(size=12, symbol="o", brush=None)
        self.fixed_marker = OverlayScatter(size=12, symbol="+")
        self.fixed_text = OverlayText(anchor=(-0.04, -0.1))  # below right of the point, no fill: the curves stay visible
        for item in (self.marker, self.fixed_marker, self.fixed_text):
            item.setZValue(20)
            plot.addItem(item, ignoreBounds=True)
        self.projection = _Projection(plot, 19)
        self.fixed_projection = _Projection(plot, 18)
        self._build_controls()
        plot.scene().sigMouseMoved.connect(self.move_to)
        self.view_box.sigRangeChanged.connect(lambda *_: self.update())
        plot.viewport().installEventFilter(self)
        self.restyle()
        self.curves_changed()

    # ------------------------------------------------------------ controls (GUI_DESIGN.md 16.8)
    def _build_controls(self):
        self.controls = QWidget()
        lay = QHBoxLayout(self.controls)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        self.snap_button = QPushButton("Snap")
        self.snap_button.setCheckable(True)
        self.snap_button.setToolTip("Привязка измерительного маркера к активной кривой (ближайшая точка отображаемой "
                                    f"линии, не дальше {SNAP_PX:g} px). Указатель мыши не ограничивается; значения "
                                    "между узлами — графическая интерполяция, не результат решателя")
        self.snap_button.toggled.connect(self.set_snap)
        self.projection_button = QPushButton("Проекции")
        self.projection_button.setCheckable(True)
        self.projection_button.setToolTip("Пунктирные линии от маркера (Snap) или от курсора до осей и значения на осях. "
                                          "Щелчок без перетаскивания фиксирует точку, Esc снимает фиксацию")
        self.projection_button.toggled.connect(self.set_projections)
        self.combo = QComboBox()
        self.combo.setToolTip("Активная кривая Snap: только видимые кривые с точками")
        self.combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.combo.activated.connect(lambda i: self.set_active(self.combo.itemData(i)))
        self.unfix_button = QPushButton("Снять фиксацию")
        self.unfix_button.setToolTip("Убрать зафиксированную точку (также Esc)")
        self.unfix_button.clicked.connect(self.clear_fixed)
        toggles = QHBoxLayout()
        toggles.setSpacing(0)
        for button, segment in ((self.snap_button, "first"), (self.projection_button, "last")):
            button.setProperty("segment", segment)  # the theme's style of a pressed toggle (light and dark)
            toggles.addWidget(button)
        lay.addLayout(toggles)
        for w in (self.combo, self.unfix_button):
            lay.addWidget(w)
        policy = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        policy.setRetainSizeWhenHidden(True)  # hidden for a saved figure without moving the plot
        self.controls.setSizePolicy(policy)

    def install_shortcut(self, widget):
        """Esc on ``widget`` (and its children) removes the fixation."""
        self.escape = QShortcut(QKeySequence(Qt.Key.Key_Escape), widget, self.clear_fixed,
                                context=Qt.ShortcutContext.WidgetWithChildrenShortcut)

    def _fill_combo(self):
        drawn = self.drawn()
        self.combo.blockSignals(True)
        self.combo.clear()
        self.combo.addItem(NO_ACTIVE, None)
        for key, name, _item in drawn:
            self.combo.addItem(name, key)
        self.combo.setCurrentIndex(max(self.combo.findData(self.active), 0) if self.active is not None else 0)
        self.combo.blockSignals(False)
        self.unfix_button.setEnabled(self.fixed is not None)

    # ------------------------------------------------------------ state
    def drawn(self):
        """(key, name, item) of the curves drawn now: visible and with at least one drawn point."""
        out = []
        for key, name, item in self._curves():
            xs, _ = item.getData()
            if item.isVisible() and xs is not None and len(xs):
                out.append((key, name, item))
        return out

    def set_snap(self, on):
        self.snap_on = bool(on)
        self.snap_button.setChecked(self.snap_on)
        self._previous = None
        self.update()

    def set_projections(self, on):
        self.projections_on = bool(on)
        self.projection_button.setChecked(self.projections_on)
        self.update()

    def set_active(self, key):
        """Choose the active curve (only a drawn curve; anything else leaves no active curve)."""
        self.active = key if any(k == key for k, _, _ in self.drawn()) else None
        self._previous = None
        self._fill_combo()
        self.update()

    def curves_changed(self):
        """The drawn curves changed (shown, hidden, other quantity): a hidden active curve is no longer active."""
        keys = [k for k, _, _ in self.drawn()]
        if self.active not in keys:
            self.active = None
        if self.fixed is not None and self.fixed.key is not None and self.fixed.key not in keys:
            self.fixed = None  # its curve is not drawn any more
        self._previous = None
        self._fill_combo()
        self.update()

    def clear_fixed(self):
        self.fixed = None
        self._fill_combo()
        self.update()

    # ------------------------------------------------------------ pointer
    def inside(self, scene_pos):
        return self.view_box.sceneBoundingRect().contains(scene_pos)

    def move_to(self, scene_pos):
        """The pointer is at scene_pos (logical px of the plot scene); None: it left the plot."""
        self._pos = None if scene_pos is None or not self.inside(scene_pos) else QPointF(scene_pos)
        self.update()

    def eventFilter(self, obj, ev):
        kind = ev.type()
        if kind == QEvent.Type.Leave:
            self.move_to(None)
        elif kind == QEvent.Type.MouseButtonPress and ev.button() == Qt.MouseButton.LeftButton:
            self._press = (ev.position(), self.view_range())
            self.plot.setFocus(Qt.FocusReason.MouseFocusReason)
        elif kind == QEvent.Type.MouseButtonRelease and ev.button() == Qt.MouseButton.LeftButton and self._press:
            (start, view), self._press = self._press, None
            moved = math.hypot(ev.position().x() - start.x(), ev.position().y() - start.y())
            if moved <= CLICK_PX and self.view_range() == view:  # a click; a drag (pan) never fixes
                self.move_to(self.plot.mapToScene(ev.position().toPoint()))
                self.fix()
        return False  # the plot keeps its own pan, zoom, point clicks and menu

    def fix(self):
        """Fix the current measurement position (with Snap or Projections on); nothing valid: no change."""
        if self.measurement is not None and (self.snap_on or self.projections_on):
            self.fixed = self.measurement
            self._fill_combo()
            self.update()
        return self.fixed

    def view_range(self):
        return tuple(tuple(r) for r in self.view_box.viewRange())

    def cursor(self):
        """The pointer in view coordinates, or None outside the plot area."""
        if self._pos is None:
            return None
        v = self.view_box.mapSceneToView(self._pos)
        return v.x(), v.y()

    def _measure(self):
        if self._pos is None:
            self._previous = None
            return None
        if not self.snap_on:
            c = self.cursor()
            return Measurement(*c) if self.projections_on else None
        item = next(((k, n, it) for k, n, it in self.drawn() if k == self.active), None)
        if item is None:
            self._previous = None
            return None
        key, name, curve = item
        xs, ys = curve.getData()
        nodes = [(float(x), float(y)) for x, y in zip(xs, ys, strict=True)]
        pixels = [(s.x(), s.y()) for s in (self.view_box.mapViewToScene(QPointF(x, y)) for x, y in nodes)]
        previous = self._previous[1] if self._previous and self._previous[0] == key else None
        s = snap(pixels, (self._pos.x(), self._pos.y()), previous)
        if s is None:
            self._previous = None
            return None
        self._previous = (key, s.segment)
        if s.node is not None:
            x, y = nodes[s.node]
        else:
            (ax, ay), (bx, by) = nodes[s.segment], nodes[s.segment + 1]
            x, y = ax + s.t * (bx - ax), ay + s.t * (by - ay)  # on the drawn segment, display coordinates
        return Measurement(x, y, key, name, s)

    # ------------------------------------------------------------ values and texts
    def values(self, x, y):
        """Quantity values of a view point (on a logarithmic axis the value, not its logarithm, 16.2)."""
        def value(v, side):
            return 10.0 ** v if self.plot.getAxis(side).logMode else v
        return value(x, "bottom"), value(y, "left")

    def coordinate_texts(self, x, y):
        """('q = 2.5 kPa', 'w = 1.23 mm') in the figure language."""
        lang = self._language()
        (xn, xu), (yn, yu) = self._axes()
        vx, vy = self.values(x, y)
        return f"{xn} = {format_value(vx, lang)} {xu}", f"{yn} = {format_value(vy, lang)} {yu}"

    def describe(self, m):
        """Interface text of a measurement position."""
        x_text, y_text = self.coordinate_texts(m.x, m.y)
        if m.snap is None:
            return f"курсор: {x_text}   {y_text}"
        return f"Snap · {m.name}: {m.snap.describe()} — {x_text}   {y_text} ({NOT_A_RESULT})"

    def readout(self):
        """The coordinate field under the plot (16.2): cursor, Snap and the fixation; empty outside the plot."""
        parts = []
        c = self.cursor()
        if c is not None:
            x_text, y_text = self.coordinate_texts(*c)
            parts.append(f"Курсор: {x_text}   {y_text}")
            if self.snap_on:
                if self.active is None:
                    parts.append("Snap: выберите активную кривую")
                elif self.measurement is None:
                    parts.append(f"Snap: дальше {SNAP_PX:g} px от активной кривой — маркера нет")
                else:
                    parts.append(self.describe(self.measurement))
        if self.fixed is not None:
            parts.append("Фиксация: " + self.describe(self.fixed))
        return "\n".join(parts)

    def _label(self, m):
        x_text, y_text = self.coordinate_texts(m.x, m.y)
        words = m.snap.figure_words(self._language()) if m.snap else FIGURE_WORDS.get(
            self._language(), FIGURE_WORDS["en"])["cursor"]
        return f"{words}\n{x_text}\n{y_text}"

    # ------------------------------------------------------------ drawing
    def restyle(self, *_):
        t = self._tokens()
        colour, background = t["text.primary"], t["plot.bg"]
        self.marker.setPen(pg.mkPen(colour, width=2))
        self.fixed_marker.setPen(pg.mkPen(t["status.error"], width=2))
        self.fixed_marker.setBrush(pg.mkBrush(t["status.error"]))
        self.fixed_text.setColor(colour)
        self.projection.style(colour, background)
        self.fixed_projection.style(t["status.error"], background)
        self.update()

    def update(self):
        """Measure at the last pointer position and redraw the overlay (pan and zoom keep the pointer, 16.4)."""
        self.measurement = self._measure()
        m, view = self.measurement, self.view_range()
        if m is not None and m.snap is not None:
            self.marker.setData([m.x], [m.y])
            self.marker.setVisible(True)
        else:
            self.marker.setData([], [])
            self.marker.setVisible(False)
        if m is not None and self.projections_on:
            self.projection.show(m.x, m.y, view, *self.coordinate_texts(m.x, m.y))
        else:
            self.projection.hide()
        f = self.fixed
        if f is not None:
            self.fixed_marker.setData([f.x], [f.y])
            self.fixed_text.setText(self._label(f))
            self.fixed_text.setPos(f.x, f.y)
            self.fixed_projection.show(f.x, f.y, view, *self.coordinate_texts(f.x, f.y))
        else:
            self.fixed_projection.hide()
        for item in (self.fixed_marker, self.fixed_text):
            item.setVisible(f is not None)
        self.unfix_button.setEnabled(f is not None)
        self.changed.emit()

    def items(self):
        return (self.marker, self.fixed_marker, self.fixed_text, *self.projection.items(), *self.fixed_projection.items())

    @contextmanager
    def hidden(self, *widgets):
        """Nothing of the interaction (overlay items, the controls, ``widgets``) is drawn inside the block."""
        saved = [(o, o.isVisible()) for o in (*self.items(), self.controls, *widgets)]
        for o, _ in saved:
            o.setVisible(False)
        try:
            yield
        finally:
            for o, shown in saved:
                o.setVisible(shown)
