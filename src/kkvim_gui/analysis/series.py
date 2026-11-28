"""«Нагрузка — прогиб»: the graph and table of a load series (GUI-PLATE-WORKFLOW-UX-01).

Drawn from ``model.series.SeriesData`` only: every KK-VIM point is a real run at its q (the stored field at the centre
node, or the separately labelled signed w at the node of the largest |w|); runs that did not end as requested stay in
the table with the reason and are not points of the curve. For SSSS the Navier reference computed in its own process at
every q is a second curve (never for CCCC). For a CCCC series with «МКР — двумерный (D2)», D2 computed in its own process
at every q is the second curve: only the points where D2 answered with a centre value; the others stay in the table
with the state and reason of their D2. q is shown in kPa and w in mm (SI multiples, stated on the axes); the saved
file keeps Pa and m at full precision. The graph is an interactive pyqtgraph X–Y plot (pan, zoom, reset); its texts
follow the figure language (GUI-D03).

VIM-1, VIM-2, VIM-3 (GUI-VIM-CYCLES-123-01) are three more curves: at every q the centre deflection after the completed
full cycles k = 1, 2, 3 of that point's own run, read from its stored half-step history (``backend.vim_cycles``); a point
whose run did not complete s = 2k, or did not end as requested, has no point on that curve (reason in the table). Every
curve can be shown or hidden.

Interaction (GUI-OWNER-UX-01): showing or hiding a curve, the figure language and the theme change only what is drawn —
the current view (pan, zoom) and the selected point stay; «Исходный вид» returns to the initial view, which contains
q = 0 and w = 0 and every point of every curve (visible or not, GUI_DESIGN.md 16.1, 16.3). A point is selected by a
click on a visible curve or by a row of the table; «Открыть расчёт точки» (or a double click on the row) opens the stored
run of that q in its own analysis window — the solver is never called again, and the run keeps its status, field role,
warnings and provenance. The final KK-VIM curve is the field of the last completed half-step of each point's own run.

«Показать исходные значения» (GUI_DESIGN.md 20.3, GUI-P5-SERIES-RAW-VALUES-01): a right click on a visible curve opens a
read-only table of the stored records of that curve, in the order the loads were entered, in Pa and m at full precision,
with the run, status, field role and mark of each point; several curves under the pointer are offered by name. The values
are read from the series records themselves — never from the plotted coordinates, never interpolated (no Snap position),
never recomputed; a hidden curve or a curve without points offers nothing. Opening and copying never start a calculation.

Snap, «Проекции» and the fixed point (GUI_DESIGN.md 16.4–16.5, GUI-P5-XY-INTERACTION-01; ``xy_interaction``): the
measurement marker follows the drawn line of the active curve, chosen in a list of the drawn curves; between drawn nodes
it is a graphical interpolation, never a solver value. The coordinate field under the plot shows the cursor, the marker
and the fixation (16.2). None of it enters the table, «Исходные значения», the saved CSV files or the saved figure.
"""

from datetime import datetime
from pathlib import Path

import pyqtgraph as pg
from PySide6.QtCore import QItemSelectionModel, QPointF, Qt, Signal
from PySide6.QtGui import QGuiApplication, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from kkvim_gui.analysis.results import MM, RowsModel, fmt, padded, page, table_view, toolbar
from kkvim_gui.analysis.xy_interaction import XYInteraction
from kkvim_gui.backend import series_export, vim_cycles
from kkvim_gui.figures.text import fig_text
from kkvim_gui.model.series import RUN_STATUS_TEXT
from kkvim_gui.widgets.common import Card, FigureLanguageSwitch, Note, SegmentedControl, fit_to_screen, label, set_help_context
from kkvim_gui.widgets.header import AppHeader, app_icon

KPA = 1e-3  # q in kPa for the axis (q in Pa · 1e-3)
HEADERS = ("q, kPa", "Расчёт", "Статус решателя", "Роль поля", "На графике", "w в центре КК-ВИМ, mm",
           "w в узле max|w|, mm", "w в центре Навье, mm", "Δw = w_КК-ВИМ − w_Навье, mm", "Δw / w_Навье, %", "Причина исключения")
D2_HEADERS = ("МКР D2", "w в центре D2, mm", "Δw = w_КК-ВИМ − w_D2, mm", "Δw / w_D2, %", "Время КК-ВИМ (решатель), s",
              "Время D2 (solve), s")
ROLE_SHORT = {"accepted": "результат R3", "research": "исследование", "diagnostic": "диагностика", None: "—"}
QUANTITIES = {"centre": "Прогиб в центре", "maxabs": "w в узле max|w|"}
CYCLE_HEADERS = tuple(h for k in range(1, vim_cycles.K_MAX + 1)
                      for h in (f"{vim_cycles.label(k)} (s = {2 * k}): w в центре, mm",
                                f"{vim_cycles.label(k)}: Δw / w_эталона, %"))
CYCLE_SYMBOLS = {1: "o", 2: "s", 3: "d"}
CYCLE_COLOURS = {1: "accent", 2: "status.success", 3: "status.error"}
KKVIM_TOGGLE = "КК-ВИМ: последний полушаг"
KKVIM_TOOLTIP = ("Итог расчёта каждой точки — поле последнего выполненного полушага её расчёта (не отдельный метод). "
                 "Как расчёт остановился, показывает столбец «Статус решателя»: CONVERGED — выполнен критерий остановки; "
                 "остановка по пределу полушагов или другой статус — это не сходимость, и такая точка показана только "
                 "в таблице с причиной")
HOME_PAD = 0.06  # relative margin of the initial view
SHOW_VALUES = "Показать исходные значения"
CURVE_HIT_PX = 8.0  # a right click this close (logical px) to a drawn segment or point is on the curve [инж.]
CURVE_KEYS = ("kkvim", 1, 2, 3, "navier", "d2")  # order of the choice when several curves are under the pointer


def _cycle_cells(p):
    cells = []
    for k in range(1, vim_cycles.K_MAX + 1):
        c = p.cycle(k)
        drawn = c is not None and c.on_graph
        cells += [c.w_m * MM if drawn else None, c.delta_pct if drawn else None]
    notes = sorted({f"{c.label}: {c.reason}" for c in (p.cycles.cycles if p.cycles else ()) if not c.on_graph and c.reason})
    return (*cells, "; ".join(notes))


def _row(p, d2=False, cycles=False):
    return _row_methods(p, d2) + (_cycle_cells(p) if cycles else ())


def _row_methods(p, d2=False):
    extra = () if not d2 else (
        p.d2_state if not p.d2_reason else f"{p.d2_state}: {p.d2_reason}", None if p.d2_w_m is None else p.d2_w_m * MM,
        None if p.d2_delta_w_m is None else p.d2_delta_w_m * MM, p.d2_rel_pct, p.kkvim_solver_s, p.d2_solve_s)
    return _base_row(p) + extra


def _base_row(p):
    return (p.q * KPA, f"№ {p.run_id}" if p.run_id else "—", p.solver_status or RUN_STATUS_TEXT.get(p.run_status, p.run_status),
            ROLE_SHORT.get(p.field_role, p.field_role), "да" if p.included else "нет",
            None if p.w_centre_m is None else p.w_centre_m * MM, None if p.w_at_max_abs_m is None else p.w_at_max_abs_m * MM,
            None if p.navier_w_m is None else p.navier_w_m * MM, None if p.delta_w_m is None else p.delta_w_m * MM,
            p.rel_pct, p.reason or "")


def home_range(xs, ys, pad=HOME_PAD):
    """((x0, x1), (y0, y1)) of the initial view: every value and zero on both axes, with a relative margin."""
    def span(values):
        lo, hi = min([0.0, *values]), max([0.0, *values])
        width = hi - lo or 1.0
        return lo - pad * width, hi + pad * width
    return span(xs), span(ys)


class SelectionRing(pg.ScatterPlotItem):
    """The ring of the selected point: drawn only, never a click target.

    pyqtgraph's scene gives a click to the topmost item whose mouseClickEvent accepts it and does not look at
    acceptedMouseButtons; a plain ScatterPlotItem accepts every click within its 18 px symbol and so took the clicks
    meant for a neighbouring load 7–8 px away. This item ignores clicks and hover, so they reach the curve points.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.setAcceptHoverEvents(False)

    def mouseClickEvent(self, ev):
        ev.ignore()

    def mouseDragEvent(self, ev):
        ev.ignore()

    def hoverEvent(self, ev):
        pass  # never claims the clicks of a hover


class SeriesFigure(QFrame):
    """Interactive X–Y graph: KK-VIM points of the series and, for SSSS, the Navier reference."""

    pointSelected = Signal(object)  # the series point record chosen on the graph (never rebuilt from a coordinate)

    def __init__(self, session, data, parent=None):
        super().__init__(parent)
        self.setProperty("figure", True)
        self.session, self.data = session, data
        self.quantity = "centre"
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 10, 14, 12)
        top = QHBoxLayout()
        self.title = label("", "figureTitle")
        top.addWidget(self.title)
        top.addStretch(1)
        marks = sorted({p.mark for p in data.points if p.mark})
        self.mark = label(" · ".join(marks), "figureMark")
        self.mark.setVisible(bool(marks))
        top.addWidget(self.mark)
        lay.addLayout(top)
        self.plot = pg.PlotWidget()
        for side in ("left", "bottom"):
            self.plot.getAxis(side).enableAutoSIPrefix(False)
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        self.legend = self.plot.addLegend(offset=(10, 10))
        lay.addWidget(self.plot, 1)
        t = session.tokens
        self.kkvim = self.plot.plot(pen=pg.mkPen(t["accent"], width=2), symbol="o", symbolSize=8, symbolBrush=t["accent"])
        self.navier = self.plot.plot(pen=pg.mkPen(t["status.warning"], width=2, style=Qt.PenStyle.DashLine), symbol="t",
                                     symbolSize=8, symbolBrush=t["status.warning"])
        self.d2 = self.plot.plot(pen=pg.mkPen(t["status.warning"], width=2, style=Qt.PenStyle.DashLine), symbol="s",
                                 symbolSize=8, symbolBrush=t["status.warning"])
        self.cycle_items = {k: self.plot.plot(pen=pg.mkPen(t[CYCLE_COLOURS[k]], width=1, style=Qt.PenStyle.DotLine),
                                              symbol=CYCLE_SYMBOLS[k], symbolSize=9, symbolBrush=t[CYCLE_COLOURS[k]])
                            for k in range(1, vim_cycles.K_MAX + 1)}
        self.shown = {"kkvim": True, "navier": True, "d2": True, **{k: True for k in self.cycle_items}}
        self.selected_point = None  # the selected series point record; kept while curves are shown or hidden
        self.marker = SelectionRing(size=18, pen=pg.mkPen(t["text.primary"], width=2), brush=None, symbol="o")
        self.marker.setZValue(10)
        self.plot.addItem(self.marker)
        for item in (self.kkvim, self.navier, self.d2, *self.cycle_items.values()):
            item.sigPointsClicked.connect(self._clicked)
        self.plot.getViewBox().disableAutoRange()  # data changes never move the view; «Исходный вид» does
        # A right click on a curve offers its stored values; elsewhere the plot keeps its own menu (this instance only).
        view_box = self.plot.getViewBox()
        self._plot_menu = view_box.raiseContextMenu
        view_box.raiseContextMenu = self._raise_context_menu
        self.values_menu = None  # the last menu offered on a curve (kept alive for its actions)
        self.value_windows = []  # open «Исходные значения» windows
        self.home = None
        # Snap, projections and the fixed point (16.4–16.5): an overlay of the drawn curves, data only read
        self.interaction = XYInteraction(self.plot, self.drawn_curves, lambda: (("q", "kPa"), ("w", "mm")),
                                         lambda: self.session.figure_language, lambda: self.session.tokens, self)
        self.interaction.install_shortcut(self)
        top.addWidget(self.interaction.controls)
        # The coordinate field is wrapped to the figure width and keeps one height for that width (the longest text
        # it can show), so the Snap and fixation lines never widen or shrink the plot in a narrow window (QA 8.1).
        self.readout = label("", "secondary")
        self.readout.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop)
        self.readout.setWordWrap(True)
        policy = QSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        policy.setRetainSizeWhenHidden(True)
        self.readout.setSizePolicy(policy)
        self._readout_probe = label("", "secondary")  # never shown: measures the kept height
        self._readout_probe.setWordWrap(True)
        self._readout_probe.setParent(self)
        self._readout_probe.hide()
        lay.addWidget(self.readout)
        self.interaction.changed.connect(self._show_readout)
        self.restyle()
        session.themeChanged.connect(self.restyle)
        session.figureLanguageChanged.connect(self.redraw)
        self.redraw()
        self.reset_view()
        self.choose_default_active()

    def drawn_curves(self):
        """(key, interface name, plot item) of every curve in the fixed order of CURVE_KEYS (drawn or not)."""
        items = {"kkvim": self.kkvim, "navier": self.navier, "d2": self.d2, **self.cycle_items}
        return [(key, self.curve_name(key), items[key]) for key in CURVE_KEYS]

    def choose_default_active(self):
        """The first drawn curve is the active Snap curve when the graph is built (later only the user changes it)."""
        drawn = self.interaction.drawn()
        if self.interaction.active is None and drawn:
            self.interaction.set_active(drawn[0][0])

    def _show_readout(self):
        text = self.interaction.readout()
        self.readout.setText(text)
        self.readout.setToolTip(text)
        self._keep_readout_height()

    def _keep_readout_height(self):
        """The height of the longest field text at the field width (at least three lines); the same for every text."""
        margins = self.layout().contentsMargins()
        width = max(self.width() - margins.left() - margins.right(), 1)
        self._readout_probe.setText(self.interaction.longest_readout())
        lines = self.readout.fontMetrics().lineSpacing() * 3
        height = max(lines, self._readout_probe.heightForWidth(width), self.readout.heightForWidth(width))
        if self.readout.height() != height or self.readout.minimumHeight() != height:
            self.readout.setFixedHeight(height)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._keep_readout_height()

    def export_pixmap(self):
        """The figure for a saved file: without the marker, projections, fixation, coordinate field and controls."""
        with self.interaction.hidden(self.readout):
            return super().grab()

    def grab(self, *args):
        """A whole-figure grab is a saved figure (the report package grabs it too), so it has no overlay (16.7)."""
        return super().grab(*args) if args else self.export_pixmap()

    def restyle(self, *_):
        t = self.session.tokens
        self.plot.setBackground(t["plot.bg"])
        for side in ("left", "bottom"):
            axis = self.plot.getAxis(side)
            axis.setPen(t["plot.axis"])
            axis.setTextPen(t["plot.axis"])
        self.marker.setPen(pg.mkPen(t["text.primary"], width=2))  # style only: the view and the selection stay
        self.interaction.restyle()

    def set_quantity(self, quantity):
        """Another quantity has other values: drawn and shown from its own initial view (a fixed point is removed)."""
        self.quantity = quantity
        self.interaction.clear_fixed()
        self.redraw()
        self.reset_view()

    def set_shown(self, key, shown):
        """Show or hide one curve: "kkvim", "navier", "d2" or the cycle k (1 … 3). The view and the selection stay."""
        self.shown[key] = bool(shown)
        self.redraw()

    def all_points(self):
        """(xs, ys) of every curve of the current quantity, visible or not (the initial view covers all of them)."""
        kx, ky, nx, ny = self.curves()
        xs, ys = [*kx, *nx], [*ky, *ny]
        for k in self.cycle_items:
            cx, cy = self.cycle_curve(k)
            xs += cx
            ys += cy
        dx, dy = self.d2_curve()
        return xs + dx, ys + dy

    def reset_view(self):
        """«Исходный вид»: q from 0, w from 0, and all points of all curves."""
        self.home = home_range(*self.all_points())
        (x0, x1), (y0, y1) = self.home
        self.plot.getViewBox().setRange(xRange=(x0, x1), yRange=(y0, y1), padding=0)

    def view_range(self):
        return tuple(tuple(r) for r in self.plot.getViewBox().viewRange())

    @property
    def selected_q(self):
        """q in Pa of the selected record (None: nothing selected)."""
        return None if self.selected_point is None else self.selected_point.q

    def _clicked(self, item, points, *_):
        """The clicked spot carries its own series record (set in redraw): no q is rebuilt from the kPa coordinate."""
        record = points[0].data() if len(points) else None
        if record is not None:
            self.select(record)
            self.pointSelected.emit(record)

    def select(self, point):
        """Select the series point record (or nothing: None); the view does not move."""
        self.selected_point = point
        self._draw_marker()

    def _draw_marker(self):
        """A ring on the selected record on the first visible curve that has it (hidden curves are never selected)."""
        point = self.selected_point
        candidates = []
        if point is not None:
            for key in ("kkvim", *self.cycle_items, "navier", "d2"):
                if self.shown[key]:
                    xs, ys = self._curve(key)
                    candidates += [(x, y) for p, x, y in zip(self.records(key), xs, ys) if p is point]
        self.marker.setData([c[0] for c in candidates[:1]], [c[1] for c in candidates[:1]])

    def records(self, key):
        """The series point records drawn on the curve key ("kkvim", "navier", "d2" or the cycle k), in drawing order."""
        return sorted(self.source_records(key), key=lambda p: p.q)

    def source_records(self, key):
        """The same records in the stored order of the series (the order the loads were entered)."""
        if key == "kkvim":
            return self.data.included
        if self.quantity != "centre":
            return []  # the reference, D2 and the cycle states are evaluated at the centre only
        if key == "navier":
            return [p for p in self.data.points if p.navier_w_m is not None]
        if key == "d2":
            return self.data.d2_included
        return self.data.cycle_included(key)

    def curve_name(self, key):
        """The name of the curve key in the interface (as the curve switches, GUI_DESIGN.md 16.3)."""
        if key == "kkvim":
            return f"{KKVIM_TOGGLE} — {QUANTITIES[self.quantity].lower()}"
        if key == "navier":
            return f"Навье (эталон, M = {self.data.navier.get('M')})" if self.data.navier else "Навье"
        if key == "d2":
            return "МКР D2"
        return vim_cycles.cycle_text(key)

    def curves_at(self, scene_pos, radius=CURVE_HIT_PX):
        """The visible curves drawn within ``radius`` logical px of the scene point, in the fixed order of CURVE_KEYS.

        The distance is measured to the drawn polyline (the segments between neighbouring points, or the point alone);
        a hidden curve or a curve without points is never under the pointer. Coinciding curves (VIM-k and the final
        KK-VIM field of a run converged at s = 2k) are all offered, always in the same order.
        """
        view_box = self.plot.getViewBox()
        hits = []
        for key in CURVE_KEYS:
            if not self.shown.get(key):
                continue
            xs, ys = self._curve(key)
            if not xs:
                continue
            pts = [view_box.mapViewToScene(QPointF(x, y)) for x, y in zip(xs, ys)]
            if min(_segment_distance(scene_pos, a, b) for a, b in zip(pts, pts[1:] or pts)) <= radius:
                hits.append(key)
        return hits

    def _raise_context_menu(self, ev):
        keys = self.curves_at(ev.scenePos())
        if not keys:
            return self._plot_menu(ev)
        self.values_menu = self.curve_menu(keys)
        self.values_menu.popup(ev.screenPos().toPoint())
        return None

    def curve_menu(self, keys):
        """«Показать исходные значения» for one curve; for several curves under the pointer, the choice of the curve."""
        menu = QMenu(self)
        if len(keys) == 1:
            action = menu.addAction(SHOW_VALUES)
            action.setToolTip(self.curve_name(keys[0]))
            action.triggered.connect(lambda _=False, k=keys[0]: self.show_values(k))
        else:
            sub = menu.addMenu(SHOW_VALUES)
            for key in keys:
                sub.addAction(self.curve_name(key)).triggered.connect(lambda _=False, k=key: self.show_values(k))
        return menu

    def show_values(self, key):
        """Open the read-only window of the stored values of the visible curve key (None: hidden or without points)."""
        records = self.source_records(key)
        if not self.shown.get(key) or not records:
            return None
        window = SourceValuesWindow(self.data, self.curve_name(key), records, value_columns(key, self.quantity), self)
        self.value_windows = [w for w in self.value_windows if w.isVisible()] + [window]
        window.show()
        return window

    def _curve(self, key):
        if key == "kkvim":
            return self.curves()[:2]
        if key == "navier":
            return self.curves()[2:]
        return self.d2_curve() if key == "d2" else self.cycle_curve(key)

    def cycle_curve(self, k):
        """(xs, ys) in kPa and mm of VIM-k: the points whose own run completed the cycle k (centre only)."""
        pts = self.records(k)
        return [p.q * KPA for p in pts], [p.cycle(k).w_m * MM for p in pts]

    def d2_curve(self):
        """(D2 xs, ys) in kPa and mm: the D2 points actually computed (centre only)."""
        pts = self.records("d2")
        return [p.q * KPA for p in pts], [p.d2_w_m * MM for p in pts]

    def curves(self):
        """(KK-VIM xs, ys, Navier xs, ys) in kPa and mm: the plotted coordinates."""
        pts = self.records("kkvim")
        key = "w_centre_m" if self.quantity == "centre" else "w_at_max_abs_m"
        kx = [p.q * KPA for p in pts]
        ky = [getattr(p, key) * MM for p in pts]
        nav = self.records("navier")
        return kx, ky, [p.q * KPA for p in nav], [p.navier_w_m * MM for p in nav]

    def redraw(self, *_):
        lang = self.session.figure_language
        self.title.setText(fig_text(lang, "series.title"))
        self.plot.setLabel("bottom", fig_text(lang, "axis.q"))
        self.plot.setLabel("left", fig_text(lang, "axis.w_centre" if self.quantity == "centre" else "axis.w_maxabs"))
        kx, ky, nx, ny = self.curves()
        if not self.shown["kkvim"]:
            kx, ky = [], []
        if not self.shown["navier"]:
            nx, ny = [], []
        self.legend.clear()
        self.kkvim.setData(kx, ky, data=self.records("kkvim") if kx else [])  # each spot carries its record
        self.kkvim.setVisible(bool(kx))
        if kx:
            self.legend.addItem(self.kkvim, fig_text(lang, "legend.series.kkvim"))
        for k, item in self.cycle_items.items():
            cx, cy = self.cycle_curve(k) if self.shown[k] else ([], [])
            item.setData(cx, cy, data=self.records(k) if cx else [])
            item.setVisible(bool(cx))
            if cx:
                self.legend.addItem(item, fig_text(lang, "legend.series.cycle", label=vim_cycles.label(k), k=k, s=2 * k))
        self.navier.setData(nx, ny, data=self.records("navier") if nx else [])
        self.navier.setVisible(bool(nx))
        if nx:
            self.legend.addItem(self.navier, fig_text(lang, "legend.series.navier", M=self.data.navier.get("M")))
        dx, dy = self.d2_curve() if self.shown["d2"] else ([], [])
        self.d2.setData(dx, dy, data=self.records("d2") if dx else [])
        self.d2.setVisible(bool(dx))
        if dx:
            self.legend.addItem(self.d2, fig_text(lang, "legend.series.d2"))
        self._draw_marker()  # no autoRange: the user's view is kept (GUI-OWNER-UX-01)
        self.interaction.curves_changed()  # a hidden active curve stops Snap until another is chosen
        self._keep_readout_height()  # names and figure language of the shown curves


def _segment_distance(p, a, b):
    """Distance (scene px) from the point p to the segment a–b."""
    dx, dy = b.x() - a.x(), b.y() - a.y()
    length2 = dx * dx + dy * dy
    t = 0.0 if length2 == 0 else max(0.0, min(1.0, ((p.x() - a.x()) * dx + (p.y() - a.y()) * dy) / length2))
    return ((p.x() - a.x() - t * dx) ** 2 + (p.y() - a.y() - t * dy) ** 2) ** 0.5


def _centre(p, key):
    return None if not p.centre else p.centre.get(key)


def value_columns(key, quantity):
    """(header, record -> stored value) of the curve key: SI units, each value exactly as stored in the record."""
    if key == "kkvim":
        if quantity == "centre":
            return (("w в центре КК-ВИМ, m", lambda p: p.w_centre_m), ("x узла центра, m", lambda p: _centre(p, "x")),
                    ("y узла центра, m", lambda p: _centre(p, "y")))
        return (("w в узле max|w| КК-ВИМ (знаковый), m", lambda p: p.w_at_max_abs_m),)
    if key == "navier":
        return (("w в центре Навье, m", lambda p: p.navier_w_m), ("Состояние Навье", lambda p: p.navier_state))
    if key == "d2":
        return (("w в центре МКР D2, m", lambda p: p.d2_w_m), ("Состояние D2", lambda p: p.d2_state))
    return ((f"w в центре после {vim_cycles.label(key)}, m", lambda p: p.cycle(key).w_m),
            ("Полушаг s", lambda p: p.cycle(key).s))


SOURCE_COLUMNS = (("№ q (порядок ввода)", lambda p: p.index), ("Расчёт №", lambda p: p.run_id), ("q, Pa", lambda p: p.q))
STATUS_COLUMNS = (("Статус решателя", lambda p: p.solver_status or RUN_STATUS_TEXT.get(p.run_status, p.run_status)),
                  ("Роль поля", lambda p: ROLE_SHORT.get(p.field_role, p.field_role)),
                  ("Отметка", lambda p: p.mark or ""), ("Научный статус", lambda p: p.scientific_status))


def raw_text(value):
    """A stored value for the clipboard: a float in its shortest round-trip form with a decimal point, None empty."""
    if value is None:
        return ""
    if isinstance(value, float):
        return repr(value)
    return str(value)


class SourceValuesModel(RowsModel):
    """The stored values shown with exactly the digits that are copied (shortest round-trip form, decimal comma)."""

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole and index.isValid():
            value = self._row(index.row())[index.column()]
            if isinstance(value, float):
                return raw_text(value).replace(".", ",")
        return super().data(index, role)


class SourceValuesWindow(QDialog):
    """«Исходные значения»: the stored records of one curve, read only (GUI_DESIGN.md 20.3; never a calculation)."""

    def __init__(self, data, curve, records, columns, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setWindowTitle(f"Исходные значения — {curve}")
        self.curve, self.records = curve, tuple(records)
        self.columns = (*SOURCE_COLUMNS, *columns, *STATUS_COLUMNS)
        self.headers = tuple(h for h, _ in self.columns)
        self.rows = tuple(tuple(get(p) for _, get in self.columns) for p in self.records)  # read once from the records
        marks = sorted({p.mark for p in self.records if p.mark})
        if any(p.field_role == "diagnostic" for p in self.records):
            marks.append("DIAGNOSTIC")
        self.marks = tuple(marks)
        self.identity = (f"Серия {data.series_id} · {data.supports} · N = {data.N} · сетка {data.grid[0]} × {data.grid[1]}"
                         f" · кривая: {curve}")
        lay = QVBoxLayout(self)
        lay.addWidget(label(self.identity, "figureTitle", wrap=True))
        if self.marks:
            lay.addWidget(Note("Отметки расчётов: " + " · ".join(self.marks) + ". Значения не являются квалифицированным "
                               "научным результатом.", "testdata"))
        missing = len(data.points) - len(self.records)
        lay.addWidget(label(
            f"Исходные значения: {len(self.records)} точек кривой в порядке ввода q, из записей расчётов, в единицах СИ "
            "с полной точностью. Интерполированные значения Snap, координаты графика и пересчёты не включаются."
            + (f" Без точки на этой кривой: {missing} значений q (не подставляются)." if missing else ""),
            "secondary", wrap=True))
        self.model = SourceValuesModel(self.headers, len(self.rows), lambda i: self.rows[i])
        self.view = table_view(self.model)
        self.view.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectItems)
        self.view.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        QShortcut(QKeySequence.StandardKey.Copy, self.view, self.copy_selection,
                  context=Qt.ShortcutContext.WidgetWithChildrenShortcut)
        lay.addWidget(self.view, 1)
        self.copy_selected_button = QPushButton("Копировать выделенное")
        self.copy_selected_button.setToolTip("Выделенные ячейки с заголовками столбцов, через табуляцию (Ctrl+C)")
        self.copy_selected_button.clicked.connect(self.copy_selection)
        self.copy_all_button = QPushButton("Копировать всё")
        self.copy_all_button.setToolTip("Вся таблица: строки происхождения «#», заголовки и значения через табуляцию")
        self.copy_all_button.clicked.connect(self.copy_all)
        close = QPushButton("Закрыть")
        close.clicked.connect(self.close)
        buttons = QHBoxLayout()
        buttons.addWidget(self.copy_selected_button)
        buttons.addWidget(self.copy_all_button)
        buttons.addStretch(1)
        buttons.addWidget(close)
        lay.addLayout(buttons)
        self.resize(980, 420)

    def _lines(self, rows, columns, picked=None):
        lines = [f"# {self.identity}"] + ([f"# {' · '.join(self.marks)}"] if self.marks else [])
        lines.append("\t".join(self.headers[c] for c in columns))
        lines += ["\t".join(raw_text(self.rows[r][c]) if picked is None or (r, c) in picked else "" for c in columns)
                  for r in rows]
        return "\n".join(lines) + "\n"

    def text(self):
        """The whole table: «#» lines of the curve and its marks, the headers, the stored values (tab-separated)."""
        return self._lines(range(len(self.rows)), range(len(self.headers)))

    def selected_text(self):
        """The selected cells as a block of their rows and columns (unselected cells inside the block empty)."""
        picked = {(i.row(), i.column()) for i in self.view.selectionModel().selectedIndexes()}
        if not picked:
            return ""
        return self._lines(sorted({r for r, _ in picked}), sorted({c for _, c in picked}), picked)

    def copy_selection(self):
        text = self.selected_text()
        if text:
            QGuiApplication.clipboard().setText(text)
        return text

    def copy_all(self):
        text = self.text()
        QGuiApplication.clipboard().setText(text)
        return text


class SeriesTab(QWidget):
    runRequested = Signal(int)  # run_id of a stored point run to open (never a new calculation)

    def __init__(self, session, data, trailing=(), display=None, parent=None):
        super().__init__(parent)
        set_help_context(self, "results.series")
        self.session, self.data = session, data
        self.figure = SeriesFigure(session, data)
        self.quantity = SegmentedControl(QUANTITIES, "centre")
        self.quantity.setToolTip("Прогиб в центре (a/2, b/2) или знаковый w в узле с наибольшим |w|: это разные величины")
        self.quantity.changed.connect(self.figure.set_quantity)
        self.save_button = QPushButton("Сохранить серию…")
        self.save_button.setToolTip("Отдельная папка серии: load_series.csv, series_manifest.json и график PNG"
                                    + ("; для D2 — его значения и время каждого метода в load_series.csv" if data.d2 else ""))
        self.save_button.clicked.connect(self._save_dialog)
        self.save_button.setEnabled(data.complete)
        self.toggles = {"kkvim": QCheckBox(KKVIM_TOGGLE)}
        self.toggles["kkvim"].setToolTip(KKVIM_TOOLTIP)
        if data.cycles_supported:
            self.toggles.update({k: QCheckBox(vim_cycles.label(k)) for k in range(1, vim_cycles.K_MAX + 1)})
        if data.navier is not None:
            self.toggles["navier"] = QCheckBox("Навье")
        if data.d2:
            self.toggles["d2"] = QCheckBox("МКР D2")
        for key, box in self.toggles.items():
            shown = display is None or key in display  # «Результаты и сравнение» of the run: the initial visibility only
            box.setChecked(shown)
            self.figure.shown[key] = shown
            box.toggled.connect(lambda shown, k=key: self.figure.set_shown(k, shown))
            if isinstance(key, int):
                box.setToolTip(f"{vim_cycles.cycle_text(key)}: при каждом q — из истории полушагов расчёта этой точки")
        self.figure.redraw()  # no view change: the initial view covers every curve, shown or not
        self.figure.choose_default_active()  # the initial visibility is not a hiding by the user
        self.reset_button = QPushButton("Исходный вид")
        self.reset_button.setToolTip("Вернуть исходный масштаб: от q = 0 и w = 0, все точки всех кривых")
        self.reset_button.clicked.connect(self.figure.reset_view)
        self.open_button = QPushButton("Открыть расчёт точки")
        self.open_button.setToolTip("Окно анализа сохранённого расчёта выбранного q (решатель не запускается); "
                                    "также двойной щелчок по строке таблицы")
        self.open_button.clicked.connect(self.open_selected)
        # two rows: the quantity and the actions; the curves (nothing is cut on a narrow window)
        top = toolbar(self.quantity, trailing=(*trailing, self.reset_button, FigureLanguageSwitch(session), self.save_button))
        curves = toolbar(label("Кривые:", "fieldLabel"), *self.toggles.values())
        bar = QWidget()
        bar_lay = QVBoxLayout(bar)
        bar_lay.setContentsMargins(0, 0, 0, 0)
        bar_lay.setSpacing(0)
        bar_lay.addWidget(top)
        bar_lay.addWidget(curves)
        cycles = data.cycles_supported
        headers = HEADERS + (D2_HEADERS if data.d2 else ()) + ((*CYCLE_HEADERS, "VIM-1–3: примечание") if cycles else ())
        self.model = RowsModel(headers, len(data.points), lambda i: _row(data.points[i], data.d2, cycles), digits=10)
        table = Card(f"Расчёты серии: {len(data.points)} значений q, на графике — {len(data.included)}",
                     right=label(f"{data.supports} · N = {data.N} · сетка {data.grid[0]} × {data.grid[1]}", "tag"))
        excluded = [p for p in data.points if not p.included]
        if excluded:
            table.body.addWidget(Note(f"Не на графике: {len(excluded)} — причина в последнем столбце; значения не "
                                      "подставляются и не интерполируются.", "testdata"))
        self.selected_label = label("Точка не выбрана: щёлкните точку на графике или строку таблицы.", "secondary", wrap=True)
        pick = QHBoxLayout()
        pick.addWidget(self.selected_label, 1)
        pick.addWidget(self.open_button)
        table.body.addLayout(pick)
        self.view = table_view(self.model)
        self.view.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.view.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.view.selectionModel().currentRowChanged.connect(lambda current, _prev: self._row_chosen(current.row()))
        self.view.doubleClicked.connect(lambda index: (self._row_chosen(index.row()), self.open_selected()))
        table.body.addWidget(self.view)
        self.figure.pointSelected.connect(self.select_point)
        self.selected = None
        self._show_selection()
        notes = [("Каждая точка КК-ВИМ — отдельный расчёт при своём q, без пересчёта по линейности."), ]
        if data.navier is not None:
            notes.append("Навье — независимый эталон, рассчитанный в отдельном процессе при каждом q (M = 2559); "
                         "показываются знаковые разности, порога согласия нет." if data.navier_ok else
                         f"Эталон Навье недоступен: {data.navier.get('reason') or data.navier.get('state')}.")
        else:
            notes.append("Для CCCC кривой Навье нет: решение Навье относится только к шарнирному опиранию SSSS.")
        if data.d2:
            missing = len(data.points) - len(data.d2_included)
            notes.append(f"МКР D2 рассчитан в отдельном процессе при каждом q для той же пластины и сетки (та же дискретная "
                         f"задача, D-049.1): точек D2 на графике — {len(data.d2_included)}"
                         + (f"; без значения — {missing} (состояние и причина в таблице)" if missing else "")
                         + ". Разность КК-ВИМ − D2 — не ошибка модели пластины. Время — одно исполнение каждого метода, "
                         "наблюдение, а не сравнение алгоритмов.")
        if data.cycles_supported:
            notes.append(f"{vim_cycles.OWNER_NOTE}. При каждом q значения VIM-k — из истории полушагов расчёта этой точки "
                         "(s = 2k), не итоговое поле и не пересчёт по нагрузке; если расчёт точки не дошёл до s = 2k, точки "
                         "на кривой VIM-k нет (причина в последнем столбце).")
        for n in notes:
            table.body.addWidget(label(n, "secondary", wrap=True))
        self.figure.setMinimumHeight(300)
        self.view.setMinimumHeight(150)  # several rows stay visible; a small window scrolls the page instead
        split = QSplitter(Qt.Orientation.Vertical)
        split.addWidget(self.figure)
        split.addWidget(table)
        split.setChildrenCollapsible(False)
        split.setSizes([460, 420])
        box = QVBoxLayout()
        box.addWidget(split)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(padded(box))
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(page(bar, scroll))

    @staticmethod
    def openable(p):
        """A point whose stored run has a field (a result, or a diagnostic field marked as such) can be opened."""
        return p is not None and p.run_id is not None and p.field_role is not None

    def _row_chosen(self, row):
        if 0 <= row < len(self.data.points):
            self.selected = self.data.points[row]
            self.figure.select(self.selected)
            self._show_selection()

    def select_q(self, q):
        """Select the point of the stored load q (Pa, exactly as stored in the run request) in the table and graph."""
        row = next((i for i, p in enumerate(self.data.points) if p.q == q), None)
        if row is not None:
            self._select_row(row)

    def select_point(self, point):
        """Select the series point record itself (a click on the graph): its row, label and ring."""
        row = next((i for i, p in enumerate(self.data.points) if p is point), None)
        if row is not None:
            self._select_row(row)

    def _select_row(self, row):
        index = self.model.index(row, 0)
        self.view.selectionModel().setCurrentIndex(
            index, QItemSelectionModel.SelectionFlag.ClearAndSelect | QItemSelectionModel.SelectionFlag.Rows)
        self._row_chosen(row)

    def _show_selection(self):
        p = self.selected
        ok = self.openable(p)
        self.open_button.setEnabled(ok)
        if p is None:
            return
        role = ROLE_SHORT.get(p.field_role, p.field_role)
        status = p.solver_status or RUN_STATUS_TEXT.get(p.run_status, p.run_status)
        tail = "" if ok else " — сохранённого поля нет, открывать нечего"
        if p.field_role == "diagnostic":
            tail = " — поле только диагностическое, не принятый научный результат"
        self.selected_label.setText(f"Выбрано: q = {fmt(p.q * KPA, 8)} kPa · расчёт № {p.run_id or '—'} · {status} · "
                                    f"{role}{tail}")

    def open_selected(self):
        """Ask for the stored run of the selected point (the analysis window of that run; no new calculation)."""
        if self.openable(self.selected):
            self.runRequested.emit(self.selected.run_id)
            return self.selected.run_id
        return None

    def folder_name(self):
        return f"kkvim_series_{datetime.now():%Y%m%d-%H%M%S}_s{self.data.series_id:03d}"

    def save(self, folder):
        """Save the series package into a new subfolder of ``folder``."""
        def render(target):
            path = Path(target) / "load_deflection.png"
            if not self.figure.export_pixmap().save(str(path), "PNG"):  # interaction overlay not exported (16.7)
                raise OSError(f"figure not saved: {path}")
            return {"graph_png": path}

        return series_export.write_series(self.data, Path(folder) / self.folder_name(), render)

    def _save_dialog(self):
        folder = QFileDialog.getExistingDirectory(self, "Папка для серии нагрузок")
        if not folder:
            return
        try:
            paths = self.save(folder)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Сохранение не выполнено", f"Серия не сохранена: {exc}")
            return
        QMessageBox.information(self, "Серия сохранена", f"Сохранено: {paths['manifest'].parent}")


class SeriesWindow(QMainWindow):
    settingsRequested = Signal()
    helpRequested = Signal()
    runRequested = Signal(int)

    def __init__(self, session, data, display=None, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setWindowIcon(app_icon())
        self.setWindowTitle(f"КК-ВИМ: нагрузка — прогиб, серия {data.series_id}")
        central = QWidget()
        central.setObjectName("Page")
        lay = QVBoxLayout(central)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        marks = sorted({p.mark for p in data.points if p.mark})
        self.header = AppHeader(session, "КК-ВИМ: нагрузка — прогиб", show_badge=bool(marks))  # never TEST DATA: real runs
        if marks:
            mark = marks[0]
            self.header.set_badge(mark, "RESEARCH" if mark.startswith("RESEARCH_ONLY") else "PROVISIONAL",
                                  "Расчёты серии не квалифицированы: " + " · ".join(marks))
        self.header.set_project(f"серия {data.series_id} · {data.started}")
        self.header.settingsClicked.connect(self.settingsRequested)
        self.header.helpClicked.connect(self.helpRequested)
        lay.addWidget(self.header)
        self.tab = SeriesTab(session, data, display=display)
        self.tab.runRequested.connect(self.runRequested)
        lay.addWidget(self.tab, 1)
        self.setCentralWidget(central)
        self.statusBar().addWidget(label(f"●  Серия {data.series_id}: {len(data.points)} значений q, "
                                         f"на графике {len(data.included)}"), 1)
        fit_to_screen(self)

    def help_context(self):
        return "results.series"


def describe(p) -> str:
    """One-line text of a point (journal, tooltips)."""
    w = "—" if p.w_centre_m is None else f"{fmt(p.w_centre_m * MM, 6)} mm"
    return f"q = {fmt(p.q * KPA, 6)} kPa: w в центре {w}" + (f" ({p.reason})" if p.reason else "")
