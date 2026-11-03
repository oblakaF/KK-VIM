"""Axonometric view of the deformed plate surface with a labelled visual exaggeration (functional preview).

The view is a drawing of the stored solver field, nothing else: every drawn node is (x_i, y_j, w_ij) of the
``RunOutcome`` arrays. The only change is the vertical scale of the image: the drawn height is
``-factor · w`` (the sag is drawn in the direction of the load q), so that deflections of order 1e-4 a become
visible. The factor is shown on the figure and written to the saved files; values, tables and exports always
use the unscaled solver field. Pure functions (``exaggeration_presets``, ``auto_factor``, ``project``) are
Qt-free; ``SurfaceFigure`` draws with pyqtgraph without OpenGL.
"""

import math

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt

from kkvim_gui.analysis.results import Figure, fmt
from kkvim_gui.widgets.common import label

AZIMUTH = math.radians(-35.0)  # rotation of the plate about the vertical axis
ELEVATION = math.radians(35.0)  # view from above the plate
AUTO_HEIGHT = 0.1  # auto factor: the largest drawn sag is 0.1 · max(a, b)
MAX_LINES = 41  # at most this many node lines per direction are drawn (every node line on coarse grids)
PRESETS = (1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6)


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


def exaggeration_text(factor) -> str:
    if factor == 1.0:
        return "Масштаб прогиба ×1 — истинный масштаб изображения"
    return (f"Прогиб увеличен ×{fmt(factor, 3)} только на изображении; значения w в таблицах, отчёте и файлах "
            "не изменены")


class SurfaceFigure(Figure):
    """Wireframe of the deformed surface: the stored field, drawn with a vertical factor that is always shown."""

    def __init__(self, session, outcome, factor=None, parent=None):
        super().__init__(session, outcome, "surface.title", interactive=False, parent=parent)
        self.auto = auto_factor(outcome.x, outcome.y, outcome.w)
        self.factor = self.auto if factor is None else float(factor)
        self.scale_note = label("", "figureMark")
        self.scale_note.setWordWrap(True)
        self.layout().insertWidget(1, self.scale_note)
        self.plot.setAspectLocked(True)
        for side in ("left", "bottom"):
            self.plot.hideAxis(side)  # projected coordinates carry no unit
        self.plot.showGrid(x=False, y=False)
        t = session.tokens
        self.outline = pg.PlotCurveItem(pen=pg.mkPen(t["text.secondary"], width=1, style=Qt.PenStyle.DashLine))
        self.mesh = pg.PlotCurveItem(pen=pg.mkPen(t["accent"], width=1), connect="finite")
        self.extremum = pg.ScatterPlotItem(size=9, brush=pg.mkBrush(t["status.warning"]), pen=pg.mkPen(None))
        for item in (self.outline, self.mesh, self.extremum):
            self.plot.addItem(item)
        x, y = outcome.x, outcome.y
        cx = np.array([x[0], x[-1], x[-1], x[0], x[0]])
        cy = np.array([y[0], y[0], y[-1], y[-1], y[0]])
        self.outline.setData(*project(cx, cy, np.zeros(5)))  # the undeformed contour of the plate
        self.retranslate()  # title and marks (no axes here, so set_axes does not do it)
        self.set_factor(self.factor)

    def set_factor(self, factor):
        self.factor = float(factor)
        o = self.outcome
        self.mesh.setData(*surface_polylines(o.x, o.y, o.w, self.factor))
        e = o.extremum()
        u, v = project(np.array([e["x"]]), np.array([e["y"]]), np.array([-self.factor * e["w"]]))
        self.extremum.setData(u, v)
        self.scale_note.setText(exaggeration_text(self.factor))
        self.plot.autoRange(padding=0.04)

    def restyle(self, *_):
        super().restyle()
        self.plot.showGrid(x=False, y=False)
