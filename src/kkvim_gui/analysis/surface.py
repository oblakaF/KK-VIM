"""Axonometric view of the deformed plate surface with a labelled visual exaggeration (functional preview).

The view is a drawing of the stored solver field, nothing else: every drawn node is (x_i, y_j, w_ij) of the
``RunOutcome`` arrays. The only change is the vertical scale of the image: the drawn height is
``-factor · w`` (the sag is drawn in the direction of the load q), so that deflections of order 1e-4 a become
visible. The factor is shown on the figure and written to the saved files; values, tables and exports always
use the unscaled solver field.

Colour (GUI-PLATE-WORKFLOW-UX-01): the surface is filled cell by cell between the drawn node lines; the colour of a cell
is the mean of the TRUE w (mm) at its four corner nodes, mapped by the approved viridis or cividis scale between the
smallest and largest w of the field; the colour bar shows that scale in mm and never depends on the exaggeration. An
optional thin mesh overlays the node lines. Cells are painted from the far side to the near side (painter's order).
This is an image of the 2D Kirchhoff–Love result, not a three-dimensional plate model. Pure functions
(``auto_factor``, ``project``, ``surface_polylines``, ``surface_cells``) are Qt-free; ``SurfaceFigure`` draws with
pyqtgraph without OpenGL.
"""

import math

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QPainter, QPen, QPolygonF

from kkvim_gui.analysis.results import MM, Figure, fmt
from kkvim_gui.figures.text import fig_text
from kkvim_gui.widgets.common import label

AZIMUTH = math.radians(-35.0)  # rotation of the plate about the vertical axis
ELEVATION = math.radians(35.0)  # view from above the plate
AUTO_HEIGHT = 0.1  # auto factor: the largest drawn sag is 0.1 · max(a, b)
MAX_LINES = 41  # at most this many node lines per direction are drawn (every node line on coarse grids)
PRESETS = (1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6)
COLORMAPS = ("viridis", "cividis")  # GUI-D11: no rainbow scale


def auto_factor(x, y, w) -> float:
    """1-2-5 factor not larger than the one that draws max|w| as AUTO_HEIGHT · max(a, b); 1 for a zero field."""
    w_max = float(np.max(np.abs(w)))
    span = max(float(x[-1] - x[0]), float(y[-1] - y[0]))
    if not w_max > 0 or not math.isfinite(w_max):
        return 1.0
    raw = AUTO_HEIGHT * span / w_max
    if raw <= 1.0:
        return 1.0
    decade = 10.0 ** math.floor(math.log10(raw))
    for step in (5.0, 2.0, 1.0):
        if step * decade <= raw:
            return step * decade
    return decade


def line_indices(n_nodes: int) -> list[int]:
    """Node lines drawn in one direction: all of them up to MAX_LINES, else evenly spread, ends included."""
    if n_nodes <= MAX_LINES:
        return list(range(n_nodes))
    return sorted({round(k * (n_nodes - 1) / (MAX_LINES - 1)) for k in range(MAX_LINES)})


def project(x, y, z):
    """Screen coordinates (u, v) of points (x, y, z) for the fixed azimuth and elevation (arrays broadcast)."""
    ca, sa = math.cos(AZIMUTH), math.sin(AZIMUTH)
    u = x * ca - y * sa
    depth = x * sa + y * ca
    v = depth * math.sin(ELEVATION) + z * math.cos(ELEVATION)
    return u, v


def depth_of(x, y):
    """Distance along the viewing direction in the plane (larger: farther, drawn first)."""
    return x * math.sin(AZIMUTH) + y * math.cos(AZIMUTH)


def surface_polylines(x, y, w, factor):
    """(u, v) of the node lines of the drawn surface, separated by NaN: lines y = y_j, then lines x = x_i."""
    X, Y = np.meshgrid(x, y, indexing="ij")
    Z = -factor * np.asarray(w)
    us, vs = [], []
    for j in line_indices(len(y)):
        u, v = project(X[:, j], Y[:, j], Z[:, j])
        us += [u, [np.nan]]
        vs += [v, [np.nan]]
    for i in line_indices(len(x)):
        u, v = project(X[i, :], Y[i, :], Z[i, :])
        us += [u, [np.nan]]
        vs += [v, [np.nan]]
    return np.concatenate(us), np.concatenate(vs)


def surface_cells(x, y, w, factor):
    """Cells between the drawn node lines, far to near: (u corners, v corners, mean TRUE w of the corners in m)."""
    w = np.asarray(w)
    ii, jj = line_indices(len(x)), line_indices(len(y))
    cells = []
    for a, b in zip(ii[:-1], ii[1:], strict=True):
        for c, d in zip(jj[:-1], jj[1:], strict=True):
            corners = ((a, c), (b, c), (b, d), (a, d))
            cx = np.array([x[i] for i, _ in corners], dtype=float)
            cy = np.array([y[j] for _, j in corners], dtype=float)
            cw = np.array([w[i, j] for i, j in corners], dtype=float)
            u, v = project(cx, cy, -factor * cw)
            cells.append((float(depth_of(cx.mean(), cy.mean())), u, v, float(cw.mean())))
    cells.sort(key=lambda cell: -cell[0])
    return [(u, v, mean) for _, u, v, mean in cells]


def exaggeration_text(factor) -> str:
    if factor == 1.0:
        return "Масштаб прогиба ×1 — истинный масштаб изображения; цвет показывает истинный w"
    return (f"Прогиб увеличен ×{fmt(factor, 3)} только на изображении; цвет и шкала показывают истинный w; значения w "
            "в таблицах, отчёте и файлах не изменены")


class SurfaceItem(pg.GraphicsObject):
    """Filled cells of the drawn surface (painted far to near) and an optional thin mesh."""

    def __init__(self):
        super().__init__()
        self.cells, self.mesh = [], True
        self._rect = QRectF()
        self.mesh_colour = None

    def set_cells(self, cells, mesh_colour):
        self.prepareGeometryChange()
        self.cells = cells
        self.mesh_colour = mesh_colour
        if cells:
            us = np.concatenate([np.asarray(u) for u, _, _ in cells])
            vs = np.concatenate([np.asarray(v) for _, v, _ in cells])
            self._rect = QRectF(float(us.min()), float(vs.min()), float(us.max() - us.min()), float(vs.max() - vs.min()))
        self.update()

    def set_mesh(self, on):
        self.mesh = bool(on)
        self.update()

    def boundingRect(self):  # noqa: N802 - Qt API
        return self._rect

    def paint(self, painter, *_):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        for u, v, colour in self.cells:
            polygon = QPolygonF([QPointF(float(a), float(b)) for a, b in zip(u, v, strict=True)])
            pen = QPen(self.mesh_colour if self.mesh else colour)
            pen.setCosmetic(True)
            pen.setWidthF(0.6 if self.mesh else 1.0)
            painter.setPen(pen)
            painter.setBrush(QBrush(colour))
            painter.drawPolygon(polygon)


class SurfaceFigure(Figure):
    """Colour-mapped deformed surface: the stored field, drawn with a vertical factor that is always shown."""

    def __init__(self, session, outcome, factor=None, cmap="viridis", parent=None):
        super().__init__(session, outcome, "surface.title", interactive=False, parent=parent)
        self.auto = auto_factor(outcome.x, outcome.y, outcome.w)
        self.factor = self.auto if factor is None else float(factor)
        if cmap not in COLORMAPS:
            raise ValueError(f"colour scale {cmap!r} is not approved")
        self.cmap_name = cmap
        self.scale_note = label("", "figureMark")
        self.scale_note.setWordWrap(True)
        self.layout().insertWidget(1, self.scale_note)
        self.plot.setAspectLocked(True)
        for side in ("left", "bottom"):
            self.plot.hideAxis(side)  # projected coordinates carry no unit; the edges are labelled on the plate
        self.plot.showGrid(x=False, y=False)
        t = session.tokens
        self.outline = pg.PlotCurveItem(pen=pg.mkPen(t["text.secondary"], width=1, style=Qt.PenStyle.DashLine))
        self.surface = SurfaceItem()
        self.extremum = pg.ScatterPlotItem(size=9, brush=pg.mkBrush(t["status.warning"]), pen=pg.mkPen(t["plot.bg"]))
        self.edge_x = pg.TextItem(anchor=(0.5, -0.3), color=t["text.secondary"])
        self.edge_y = pg.TextItem(anchor=(1.05, 0.5), color=t["text.secondary"])
        for item in (self.outline, self.surface, self.extremum, self.edge_x, self.edge_y):
            self.plot.addItem(item)
        self.extremum.setZValue(10)
        values = np.asarray(outcome.w) * MM
        self.low, self.high = float(values.min()), float(values.max())
        if self.low == self.high:
            self.low, self.high = self.low - 0.5, self.high + 0.5
        self.bar = pg.ColorBarItem(values=(self.low, self.high), colorMap=pg.colormap.get(cmap), interactive=False,
                                   width=14)
        layout = self.plot.getPlotItem().layout
        layout.addItem(self.bar, 2, 5)  # after the right-hand axis, as ColorBarItem.setImageItem(insert_in=...)
        layout.setColumnFixedWidth(4, 5)
        x, y = outcome.x, outcome.y
        cx = np.array([x[0], x[-1], x[-1], x[0], x[0]])
        cy = np.array([y[0], y[0], y[-1], y[-1], y[0]])
        self.outline.setData(*project(cx, cy, np.zeros(5)))  # the undeformed contour of the plate
        u, v = project(np.array([(x[0] + x[-1]) / 2]), np.array([y[0]]), np.zeros(1))
        self.edge_x.setPos(float(u[0]), float(v[0]))
        u, v = project(np.array([x[0]]), np.array([(y[0] + y[-1]) / 2]), np.zeros(1))
        self.edge_y.setPos(float(u[0]), float(v[0]))
        self.retranslate()  # title and marks (no axes here, so set_axes does not do it)
        self.set_factor(self.factor)

    def retranslate(self, *_):
        super().retranslate()
        if getattr(self, "bar", None) is None:
            return
        lang = self.session.figure_language
        o = self.outcome
        self.bar.axis.setLabel(fig_text(lang, "surface.colorbar"))
        self.edge_x.setText(fig_text(lang, "surface.edge.x", a=float(o.x[-1] - o.x[0])))
        self.edge_y.setText(fig_text(lang, "surface.edge.y", b=float(o.y[-1] - o.y[0])))

    def colours(self, cells):
        """QColor of every cell: the approved scale at the mean TRUE w of its corners (independent of the factor)."""
        cmap = pg.colormap.get(self.cmap_name)
        means = np.array([mean for _, _, mean in cells]) * MM
        return cmap.map((means - self.low) / (self.high - self.low), mode="qcolor")

    def set_factor(self, factor):
        self.factor = float(factor)
        o = self.outcome
        cells = surface_cells(o.x, o.y, o.w, self.factor)
        colours = self.colours(cells)
        mesh = pg.mkColor(self.session.tokens["plot.bg"])
        mesh.setAlpha(110)
        self.surface.set_cells([(u, v, c) for (u, v, _), c in zip(cells, colours, strict=True)], mesh)
        e = o.extremum()
        u, v = project(np.array([e["x"]]), np.array([e["y"]]), np.array([-self.factor * e["w"]]))
        self.extremum.setData(u, v)
        self.scale_note.setText(exaggeration_text(self.factor))
        self.plot.autoRange(padding=0.06)

    def set_colormap(self, name):
        if name not in COLORMAPS:
            raise ValueError(f"colour scale {name!r} is not approved")
        self.cmap_name = name
        self.bar.setColorMap(pg.colormap.get(name))
        self.set_factor(self.factor)

    def set_mesh(self, on):
        self.surface.set_mesh(on)

    def restyle(self, *_):
        super().restyle()
        self.plot.showGrid(x=False, y=False)
