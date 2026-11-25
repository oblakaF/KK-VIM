"""PublicationRenderer: figures for publication — PNG 300/600 dpi, SVG, PDF and the clipboard (GUI_DESIGN.md 8, 16.6–16.7,
20.4; GUI-D04, P6).

A figure is described by a ``XYSpec`` (an X–Y graph) or a ``FieldSpec`` (the colour field of w on the result nodes):
already translated texts, the drawn values and the style. The renderer paints the description with QPainter in points
(1/72 in) on any paint device, so the screen figure and the publication figure are two drawings of one description and
the theme of the interface never reaches the file:

* a fixed publication style — white background, black axes and texts, a sans-serif font with Cyrillic (Arial, then
  Liberation Sans or DejaVu Sans), 8 pt text, 0.75–1.0 pt lines, no grid; curve colours by default from the Okabe–Ito
  palette; explicit choices of «Оформление графика» (16.6) replace the defaults of a curve; the backgrounds are always
  white (16.7: the owner's open question, until decided the export is white);
* only the curves given (the visible ones); no interaction overlay ever (cursor, Snap marker, projections, fixed point,
  coordinate field and buttons are not part of a description);
* TEST DATA: a diagonal watermark that cannot be switched off; the research, provisional and diagnostic marks of the
  result are written above the plot;
* formats: PNG at the chosen dpi (the pixel size follows the width in mm and the dpi only, never the screen), SVG and
  PDF with vector axes, lines and texts — the colour field is an embedded raster image at the chosen dpi («где
  применимо»); metadata in every file: program version, run, data origin, language (and git commit when known).

Mathematical texts: Unicode symbols; ``^x`` and ``_x`` (or ``^{…}``, ``_{…}``) of the figure catalog are drawn as a
superscript and a subscript. matplotlib (mathtext) is not used: it is a dependency subject to the owner's approval
(PROPOSED_CHANGES.md 1); Qt (PySide6-Essentials, GUI-D01) is used instead [инж.].

Nothing here reads or changes solver arrays: a description holds copies of the drawn numbers.
"""

import json
import math
from dataclasses import dataclass, field, replace

import numpy as np
from PySide6.QtCore import QBuffer, QByteArray, QIODevice, QMarginsF, QMimeData, QPointF, QRectF, QSize, QSizeF, Qt
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QFontMetricsF,
    QImage,
    QPageLayout,
    QPageSize,
    QPainter,
    QPainterPath,
    QPdfWriter,
    QPen,
    QPolygonF,
)

from kkvim_gui import program_version

MM_PER_IN = 25.4
PT_PER_IN = 72.0
WIDTHS_MM = {"one_column": 85.0, "two_columns": 175.0}
DPI_CHOICES = (300, 600)
FORMATS = ("png", "svg", "pdf")
ASPECT = 0.70  # height / width of an X–Y figure [инж.]
FONT_FAMILIES = ("Arial", "Liberation Sans", "DejaVu Sans", "Segoe UI")
FONT_PT = 8.0
SMALL_PT = 6.5
OKABE_ITO = ("#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7", "#56B4E9", "#000000", "#F0E442")
DASHES = {"solid": Qt.PenStyle.SolidLine, "dash": Qt.PenStyle.DashLine, "dot": Qt.PenStyle.DotLine,
          "dashdot": Qt.PenStyle.DashDotLine}
DASH_NAMES = {"solid": "сплошная", "dash": "штриховая", "dot": "пунктирная", "dashdot": "штрихпунктирная"}
MARKERS = ("o", "s", "t", "d", "+", "x")
WATERMARK = "TEST DATA"
WHITE, BLACK = "#FFFFFF", "#000000"


@dataclass(frozen=True)
class CurveStyle:
    """Style of one curve in the file: colour, line width in pt, line type, marker (None: no marker)."""

    colour: str = BLACK
    width: float = 1.0
    dash: str = "solid"
    marker: str | None = None
    marker_size: float = 3.5  # pt


@dataclass(frozen=True)
class Series:
    """One curve as drawn: the legend text and the drawn values (in the units of the axis labels)."""

    label: str
    xs: tuple
    ys: tuple
    style: CurveStyle = CurveStyle()


@dataclass(frozen=True)
class RefLine:
    """A horizontal reference line (y in the units of the axis), e.g. a tolerance or a reference value."""

    y: float
    label: str
    style: CurveStyle = CurveStyle(dash="dash", width=0.75)


@dataclass(frozen=True)
class XYSpec:
    title: str
    x_label: str
    y_label: str
    series: tuple = ()
    ref_lines: tuple = ()
    log_y: bool = False
    invert_y: bool = False
    x_ticks: tuple | None = None  # ((value, text), …): named ticks (e.g. VIM-k)
    x_range: tuple | None = None
    y_range: tuple | None = None
    marks: tuple = ()  # research / provisional / diagnostic marks of the result, figure language
    test_data: bool = False
    lang: str = "en"
    meta: dict = field(default_factory=dict)


@dataclass(frozen=True)
class FieldSpec:
    """The colour field of a quantity on the result nodes (axis 0 = x, axis 1 = y), node-centred cells."""

    title: str
    x_label: str
    y_label: str
    colorbar_label: str
    x: tuple
    y: tuple
    values: np.ndarray
    cmap: str = "viridis"
    isolines: int = 10
    marks: tuple = ()
    test_data: bool = False
    lang: str = "en"
    meta: dict = field(default_factory=dict)


# ------------------------------------------------------------------------------------------------ text helpers


def decimal(text, lang):
    return text.replace(".", ",") if lang == "ru" else text


SUP = str.maketrans("0123456789-+−", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺⁻")


def step_decimals(step):
    """The decimals that write every multiple of ``step`` exactly (0.25 -> 2, 0.2 -> 1, 5 -> 0)."""
    if not step:
        return 3
    for d in range(0, 15):
        scaled = abs(step) * 10 ** d
        if abs(round(scaled) - scaled) <= 1e-9 * max(1.0, scaled):
            return d
    return 15


def tick_text(value, step, lang):
    """A tick label: fixed point with the decimals of the step, or m×10ⁿ for very small or large numbers."""
    if value == 0:
        return "0"
    big = max(abs(value), abs(step))
    if big >= 1e5 or big < 1e-3:
        exponent = math.floor(math.log10(abs(step))) if step else math.floor(math.log10(abs(value)))
        mantissa = value / 10 ** exponent
        text = f"{mantissa:.{step_decimals(step / 10 ** exponent) if step else 0}f}"
        return decimal(f"{text}×10{str(exponent).translate(SUP)}", lang)
    return decimal(f"{value:.{step_decimals(step)}f}", lang)


def nice_ticks(lo, hi, count=5):
    """Ticks at 1, 2, 2.5 or 5 × 10ⁿ inside [lo, hi] and their step."""
    if not (math.isfinite(lo) and math.isfinite(hi)) or hi <= lo:
        return [lo], 0.0
    raw = (hi - lo) / count
    power = 10 ** math.floor(math.log10(raw))
    step = next(m * power for m in (1, 2, 2.5, 5, 10) if m * power >= raw)
    first = math.ceil(lo / step - 1e-9) * step
    ticks, k = [], 0
    while first + k * step <= hi + step * 1e-9:
        ticks.append(round(first + k * step, 12))
        k += 1
    return ticks, step


def _runs(text):
    """[(text, level)] of a figure text: level +1 after ^, -1 after _ (one character or {…}); 0 otherwise."""
    runs, i, plain = [], 0, ""
    while i < len(text):
        ch = text[i]
        if ch in "^_" and i + 1 < len(text) and text[i + 1] not in " ,)":
            if plain:
                runs.append((plain, 0))
                plain = ""
            if text[i + 1] == "{" and "}" in text[i + 2:]:
                end = text.index("}", i + 2)
                runs.append((text[i + 2:end], 1 if ch == "^" else -1))
                i = end + 1
            else:
                runs.append((text[i + 1], 1 if ch == "^" else -1))
                i += 2
            continue
        plain += ch
        i += 1
    if plain:
        runs.append((plain, 0))
    return runs


class Painter:
    """QPainter in points on any device, with fonts sized in points independently of the device resolution."""

    def __init__(self, painter):
        self.p = painter
        device = painter.device()
        self.dpi = device.logicalDpiY() or PT_PER_IN

    def font(self, size=FONT_PT, bold=False):
        f = QFont()
        f.setFamilies(list(FONT_FAMILIES))
        f.setPointSizeF(size * PT_PER_IN / self.dpi)  # after the painter's scale the text is ``size`` pt
        f.setBold(bold)
        f.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
        return f

    def metrics(self, font):
        return QFontMetricsF(font, self.p.device())

    def rich_width(self, text, size=FONT_PT, bold=False):
        return sum(self.metrics(self.font(size * (0.72 if level else 1.0), bold)).horizontalAdvance(t)
                   for t, level in _runs(text))

    def rich(self, x, y, text, size=FONT_PT, colour=BLACK, align="left", bold=False, plain=False):
        """Draw a text with super- and subscripts (``plain``: literally); y is the baseline; align left/centre/right."""
        runs = [(text, 0)] if plain else _runs(text)
        width = sum(self.metrics(self.font(size * (0.72 if lv else 1.0), bold)).horizontalAdvance(t) for t, lv in runs)
        x0 = x - width if align == "right" else x - width / 2 if align == "centre" else x
        self.p.setPen(QPen(QColor(colour)))
        for t, level in runs:
            f = self.font(size * (0.72 if level else 1.0), bold)
            self.p.setFont(f)
            dy = -size * 0.38 if level > 0 else size * 0.18 if level < 0 else 0.0
            self.p.drawText(QPointF(x0, y + dy), t)
            x0 += self.metrics(f).horizontalAdvance(t)
        return width

    def wrap(self, text, width, size=FONT_PT):
        """Lines of a plain text wrapped to ``width`` points."""
        m = self.metrics(self.font(size))
        lines = []
        for paragraph in str(text).split("\n"):
            line = ""
            for word in paragraph.split(" "):
                probe = f"{line} {word}".strip()
                if line and m.horizontalAdvance(probe) > width:
                    lines.append(line)
                    line = word
                else:
                    line = probe
            lines.append(line)
        return lines


def pen(colour, width, dash="solid"):
    p = QPen(QColor(colour))
    p.setWidthF(width)
    p.setStyle(DASHES.get(dash, Qt.PenStyle.SolidLine))
    p.setCapStyle(Qt.PenCapStyle.FlatCap)
    p.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    return p


def draw_marker(p, x, y, style):
    s = style.marker_size / 2
    p.setPen(pen(style.colour, 0.6))
    p.setBrush(QBrush(QColor(style.colour)))
    m = style.marker
    if m == "s":
        p.drawRect(QRectF(x - s, y - s, 2 * s, 2 * s))
    elif m == "t":
        p.drawPolygon(QPolygonF([QPointF(x, y - s * 1.15), QPointF(x - s, y + s * 0.85), QPointF(x + s, y + s * 0.85)]))
    elif m == "d":
        p.drawPolygon(QPolygonF([QPointF(x, y - s * 1.2), QPointF(x + s, y), QPointF(x, y + s * 1.2), QPointF(x - s, y)]))
    elif m in ("+", "x"):
        p.setBrush(Qt.BrushStyle.NoBrush)
        if m == "+":
            p.drawLine(QPointF(x - s, y), QPointF(x + s, y))
            p.drawLine(QPointF(x, y - s), QPointF(x, y + s))
        else:
            p.drawLine(QPointF(x - s, y - s), QPointF(x + s, y + s))
            p.drawLine(QPointF(x - s, y + s), QPointF(x + s, y - s))
    else:
        p.drawEllipse(QPointF(x, y), s, s)
    p.setBrush(Qt.BrushStyle.NoBrush)


def watermark(pp, rect):
    """The diagonal TEST DATA mark over the whole figure (GUI_DESIGN.md 9: cannot be switched off)."""
    p = pp.p
    p.save()
    p.translate(rect.center())
    p.rotate(-math.degrees(math.atan2(rect.height(), rect.width())))
    size = max(10.0, min(rect.width(), rect.height()) * 0.16)
    f = pp.font(size, bold=True)
    p.setFont(f)
    colour = QColor(200, 30, 30, 70)
    p.setPen(QPen(colour))
    w = pp.metrics(f).horizontalAdvance(WATERMARK)
    p.drawText(QPointF(-w / 2, size * 0.35), WATERMARK)
    p.restore()


def _header(pp, spec, rect):
    """Title and the marks of the result; returns the top of the plot area."""
    top = rect.top() + FONT_PT * 1.25
    if spec.title:
        for line in pp.wrap(spec.title, rect.width(), FONT_PT + 1)[:2]:
            pp.rich(rect.center().x(), top, line, FONT_PT + 1, align="centre", bold=True)
            top += (FONT_PT + 1) * 1.3
    marks = list(spec.marks) + (["TEST DATA"] if spec.test_data else [])
    if marks:
        pp.rich(rect.right(), top, " · ".join(dict.fromkeys(marks)), SMALL_PT, colour="#8A1C1C", align="right", plain=True)
        top += SMALL_PT * 1.4
    return top


# ------------------------------------------------------------------------------------------------ X–Y figure


def _finite(values):
    return [float(v) for v in values if v is not None and math.isfinite(float(v))]


def data_ranges(spec):
    xs, ys = [], []
    for s in spec.series:
        pairs = [(x, y) for x, y in zip(s.xs, s.ys, strict=True) if x is not None and y is not None]
        xs += _finite(x for x, _ in pairs)
        ys += _finite(y for _, y in pairs if not spec.log_y or y > 0)
    ys += _finite(r.y for r in spec.ref_lines)
    if spec.x_ticks:
        xs += [t[0] for t in spec.x_ticks]

    def span(values, log=False):
        if not values:
            return (0.0, 1.0)
        if log:
            values = [math.log10(v) for v in values if v > 0] or [0.0]
        lo, hi = min(values), max(values)
        pad = (hi - lo) * 0.06 or (abs(lo) * 0.1 or 1.0)
        return lo - pad, hi + pad

    x_range = spec.x_range or span(xs)
    y_range = spec.y_range or span(ys, spec.log_y)
    if spec.y_range and spec.log_y:
        y_range = tuple(math.log10(v) for v in spec.y_range)
    return x_range, y_range


def _y_ticks(spec, y0, y1):
    if not spec.log_y:
        ticks, step = nice_ticks(y0, y1)
        return [(t, tick_text(t, step, spec.lang)) for t in ticks]
    lo, hi = math.ceil(y0 - 1e-9), math.floor(y1 + 1e-9)
    if hi - lo >= 1:
        stride = max(1, math.ceil((hi - lo + 1) / 7))
        return [(e, "10" + str(int(e)).translate(SUP)) for e in range(lo, hi + 1, stride)]
    ticks, step = nice_ticks(10 ** y0, 10 ** y1)
    return [(math.log10(t), tick_text(t, step, spec.lang)) for t in ticks if t > 0]


def render_xy(spec, painter, rect):
    """Paint the X–Y figure ``spec`` into ``rect`` (points) of ``painter``."""
    pp = Painter(painter)
    p = painter
    p.save()
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    p.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
    p.fillRect(rect, QColor(WHITE))
    top = _header(pp, spec, rect)
    (x0, x1), (y0, y1) = data_ranges(spec)
    y_ticks = _y_ticks(spec, y0, y1)
    x_ticks = list(spec.x_ticks) if spec.x_ticks else None
    if x_ticks is None:
        ticks, step = nice_ticks(x0, x1)
        x_ticks = [(t, tick_text(t, step, spec.lang)) for t in ticks]
    tick_w = max((pp.rich_width(t) for _, t in y_ticks), default=10.0)
    left = rect.left() + FONT_PT * 1.6 + tick_w + 6
    bottom = rect.bottom() - FONT_PT * 3.1
    right = rect.right() - max(6.0, pp.rich_width(x_ticks[-1][1]) / 2 if x_ticks else 6.0)
    plot = QRectF(QPointF(left, top + 4), QPointF(right, bottom))

    def X(v):
        return plot.left() + (v - x0) / (x1 - x0) * plot.width()

    def Y(v):
        v = math.log10(v) if spec.log_y else v
        f = (v - y0) / (y1 - y0)
        return plot.top() + f * plot.height() if spec.invert_y else plot.bottom() - f * plot.height()

    def Yraw(v):  # already in axis coordinates (log10 for a log axis)
        f = (v - y0) / (y1 - y0)
        return plot.top() + f * plot.height() if spec.invert_y else plot.bottom() - f * plot.height()

    axis = pen(BLACK, 0.75)
    p.setPen(axis)
    p.drawRect(plot)
    for v, text in x_ticks:
        if x0 - 1e-12 <= v <= x1 + 1e-12:
            x = X(v)
            p.setPen(axis)
            p.drawLine(QPointF(x, plot.bottom()), QPointF(x, plot.bottom() - 3))
            pp.rich(x, plot.bottom() + FONT_PT * 1.15, text, align="centre")
    for v, text in y_ticks:
        if y0 - 1e-12 <= v <= y1 + 1e-12:
            y = Yraw(v)
            p.setPen(axis)
            p.drawLine(QPointF(plot.left(), y), QPointF(plot.left() + 3, y))
            pp.rich(plot.left() - 3, y + FONT_PT * 0.35, text, align="right")
    pp.rich(plot.center().x(), rect.bottom() - FONT_PT * 0.5, spec.x_label, align="centre")
    p.save()
    p.translate(rect.left() + FONT_PT * 1.0, plot.center().y())
    p.rotate(-90)
    pp.rich(0, 0, spec.y_label, align="centre")
    p.restore()

    p.save()
    p.setClipRect(plot)
    for r in spec.ref_lines:
        if spec.log_y and r.y <= 0:
            continue
        y = Y(r.y)
        p.setPen(pen(r.style.colour, r.style.width, r.style.dash))
        p.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y))
        pp.rich(plot.right() - 3, y - 2, r.label, SMALL_PT, colour=r.style.colour, align="right")
    for s in spec.series:
        pts = [(X(x), Y(y)) for x, y in zip(s.xs, s.ys, strict=True)
               if x is not None and y is not None and math.isfinite(x) and math.isfinite(y) and (not spec.log_y or y > 0)]
        if s.style.width > 0 and s.style.dash != "none" and len(pts) > 1:
            path = QPainterPath(QPointF(*pts[0]))
            for q in pts[1:]:
                path.lineTo(QPointF(*q))
            p.setPen(pen(s.style.colour, s.style.width, s.style.dash))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPath(path)
        if s.style.marker:
            for q in pts:
                draw_marker(p, *q, s.style)
    p.restore()
    _legend(pp, spec, plot, X, Y)
    if spec.test_data:
        watermark(pp, rect)
    p.restore()


def _legend(pp, spec, plot, X, Y):
    items = [(s.label, s.style, True) for s in spec.series if s.label] + \
            [(r.label, r.style, False) for r in spec.ref_lines if r.label and False]
    if not items:
        return
    sample = 16.0
    line_h = SMALL_PT * 1.45
    width = max(pp.rich_width(t, SMALL_PT) for t, _, _ in items) + sample + 10
    width = min(width, plot.width() * 0.7)
    height = line_h * len(items) + 4
    # the corner with the fewest drawn points [инж.]
    pts = [(X(x), Y(y)) for s in spec.series for x, y in zip(s.xs, s.ys, strict=True)
           if x is not None and y is not None and (not spec.log_y or y > 0)]
    corners = {"tr": QRectF(plot.right() - width - 4, plot.top() + 4, width, height),
               "tl": QRectF(plot.left() + 4, plot.top() + 4, width, height),
               "br": QRectF(plot.right() - width - 4, plot.bottom() - height - 4, width, height),
               "bl": QRectF(plot.left() + 4, plot.bottom() - height - 4, width, height)}
    box = min(corners.values(), key=lambda r: sum(r.contains(QPointF(*q)) for q in pts))
    p = pp.p
    p.setPen(pen("#666666", 0.5))
    p.setBrush(QBrush(QColor(WHITE)))
    p.drawRect(box)
    p.setBrush(Qt.BrushStyle.NoBrush)
    y = box.top() + 2 + line_h * 0.72
    for text, style, _ in items:
        mid = y - SMALL_PT * 0.33
        if style.width > 0:
            p.setPen(pen(style.colour, style.width, style.dash))
            p.drawLine(QPointF(box.left() + 4, mid), QPointF(box.left() + 4 + sample, mid))
        if style.marker:
            draw_marker(p, box.left() + 4 + sample / 2, mid, style)
        pp.rich(box.left() + sample + 8, y, text, SMALL_PT)
        y += line_h


# ------------------------------------------------------------------------------------------------ colour field


def colour_image(values, cmap, low, high):
    """RGBA QImage of the values (rows = y from the top, columns = x), the colour map of the screen (viridis/cividis)."""
    import pyqtgraph as pg

    v = np.asarray(values, dtype=float)
    span = high - low if high > low else 1.0
    norm = np.clip((v - low) / span, 0.0, 1.0)
    rgba = pg.colormap.get(cmap).map(norm.T[::-1, :], mode="byte")  # y up: the last row of the image is y_0
    rgba = np.ascontiguousarray(rgba, dtype=np.uint8)
    h, w = rgba.shape[:2]
    image = QImage(rgba.data, w, h, 4 * w, QImage.Format.Format_RGBA8888)
    return image.copy()


def render_field(spec, painter, rect, dpi=300):
    pp = Painter(painter)
    p = painter
    p.save()
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    p.fillRect(rect, QColor(WHITE))
    top = _header(pp, spec, rect)
    x, y = np.asarray(spec.x, float), np.asarray(spec.y, float)
    hx, hy = x[1] - x[0], y[1] - y[0]
    X0, X1, Y0, Y1 = x[0] - hx / 2, x[-1] + hx / 2, y[0] - hy / 2, y[-1] + hy / 2
    values = np.asarray(spec.values, float)
    low, high = float(values.min()), float(values.max())
    if low == high:
        low, high = low - 0.5, high + 0.5
    bar_w = 8.0
    xt, x_step = nice_ticks(X0 + hx / 2, X1 - hx / 2, 4)
    yt, y_step = nice_ticks(Y0 + hy / 2, Y1 - hy / 2, 4)
    tick_w = max((pp.rich_width(tick_text(t, y_step, spec.lang)) for t in yt), default=10.0)
    bt, b_step = nice_ticks(low, high, 5)
    bar_tick_w = max((pp.rich_width(tick_text(t, b_step, spec.lang)) for t in bt), default=10.0)
    avail = QRectF(QPointF(rect.left() + FONT_PT * 1.6 + tick_w + 6, top + 4),
                   QPointF(rect.right() - bar_w - bar_tick_w - FONT_PT * 2.6 - 14, rect.bottom() - FONT_PT * 3.1))
    aspect = (Y1 - Y0) / (X1 - X0)
    w = min(avail.width(), avail.height() / aspect)
    plot = QRectF(avail.left(), avail.top(), w, w * aspect)

    def X(v):
        return plot.left() + (v - X0) / (X1 - X0) * plot.width()

    def Y(v):
        return plot.bottom() - (v - Y0) / (Y1 - Y0) * plot.height()

    image = colour_image(values, spec.cmap, low, high)
    px = max(1, int(plot.width() / PT_PER_IN * dpi)), max(1, int(plot.height() / PT_PER_IN * dpi))
    p.drawImage(plot, image.scaled(QSize(*px), Qt.AspectRatioMode.IgnoreAspectRatio,
                                   Qt.TransformationMode.FastTransformation))
    if spec.isolines and high > low:
        import pyqtgraph as pg

        p.save()
        p.setClipRect(plot)
        p.setPen(pen(WHITE, 0.5))
        for k in range(1, spec.isolines + 1):
            level = low + (high - low) * k / (spec.isolines + 1)
            for line in pg.functions.isocurve(values, level, connected=True):
                # pyqtgraph puts node (i, j) at (i + 0.5, j + 0.5) (pixel centres): the node itself is drawn at (x_i, y_j)
                pts = [QPointF(X(x[0] + (i - 0.5) * hx), Y(y[0] + (j - 0.5) * hy)) for i, j in line]
                if len(pts) > 1:
                    p.drawPolyline(QPolygonF(pts))
        p.restore()
    axis = pen(BLACK, 0.75)
    p.setPen(axis)
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawRect(plot)
    for t in xt:
        p.setPen(axis)
        p.drawLine(QPointF(X(t), plot.bottom()), QPointF(X(t), plot.bottom() + 3))
        pp.rich(X(t), plot.bottom() + FONT_PT * 1.4, tick_text(t, x_step, spec.lang), align="centre")
    for t in yt:
        p.setPen(axis)
        p.drawLine(QPointF(plot.left(), Y(t)), QPointF(plot.left() - 3, Y(t)))
        pp.rich(plot.left() - 4, Y(t) + FONT_PT * 0.35, tick_text(t, y_step, spec.lang), align="right")
    pp.rich(plot.center().x(), plot.bottom() + FONT_PT * 2.7, spec.x_label, align="centre")
    p.save()
    p.translate(rect.left() + FONT_PT * 1.0, plot.center().y())
    p.rotate(-90)
    pp.rich(0, 0, spec.y_label, align="centre")
    p.restore()
    bar = QRectF(plot.right() + 10, plot.top(), bar_w, plot.height())
    grad = colour_image(np.linspace(low, high, 256).reshape(1, -1), spec.cmap, low, high)
    p.drawImage(bar, grad.scaled(QSize(4, 256)))
    p.setPen(axis)
    p.drawRect(bar)
    for t in bt:
        yy = bar.bottom() - (t - low) / (high - low) * bar.height()
        p.setPen(axis)
        p.drawLine(QPointF(bar.right(), yy), QPointF(bar.right() + 3, yy))
        pp.rich(bar.right() + 4, yy + FONT_PT * 0.35, tick_text(t, b_step, spec.lang))
    p.save()
    p.translate(bar.right() + bar_tick_w + FONT_PT * 1.8 + 4, bar.center().y())
    p.rotate(90)
    pp.rich(0, 0, spec.colorbar_label, align="centre")
    p.restore()
    if spec.test_data:
        watermark(pp, rect)
    p.restore()


# ------------------------------------------------------------------------------------------------ files


def render(spec, painter, rect, dpi=300):
    if isinstance(spec, FieldSpec):
        render_field(spec, painter, rect, dpi)
    else:
        render_xy(spec, painter, rect)


def size_pt(spec, width_mm):
    width = width_mm / MM_PER_IN * PT_PER_IN
    if isinstance(spec, FieldSpec):
        x, y = np.asarray(spec.x, float), np.asarray(spec.y, float)
        plate = (y[-1] - y[0]) / (x[-1] - x[0])
        plot_w = max(width - 120.0, width * 0.45)  # tick labels, the colour bar and its labels take ≈ 120 pt
        head = FONT_PT * 2.6 + (SMALL_PT * 1.4 if spec.marks or spec.test_data else 0.0)
        return QSizeF(width, min(plot_w * plate, width * 1.1) + head + FONT_PT * 3.6)
    return QSizeF(width, width * ASPECT)


def metadata(spec, fmt=None, dpi=None, width_mm=None):
    """Metadata of a file (GUI_DESIGN.md 8.3): program version, run, data origin, figure language; never results."""
    meta = {"program": "KK-VIM GUI", "kkvim_gui": program_version(), "renderer": "PublicationRenderer (Qt QPainter)",
            "data_origin": "demo" if spec.test_data else spec.meta.get("data_origin", "solver"),
            "figure_language": spec.lang, **{k: v for k, v in spec.meta.items() if v is not None}}
    if fmt:
        meta["format"] = fmt
    if dpi:
        meta["dpi"] = dpi
    if width_mm:
        meta["width_mm"] = width_mm
    if spec.marks:
        meta["marks"] = list(spec.marks)
    return meta


def image(spec, dpi=300, width_mm=WIDTHS_MM["one_column"]):
    """The figure as a QImage at ``dpi`` (pixel size = width in mm and dpi only), white background."""
    s = size_pt(spec, width_mm)
    w, h = round(s.width() / PT_PER_IN * dpi), round(s.height() / PT_PER_IN * dpi)
    img = QImage(w, h, QImage.Format.Format_ARGB32)
    dpm = round(dpi / MM_PER_IN * 1000)
    img.setDotsPerMeterX(dpm)
    img.setDotsPerMeterY(dpm)
    img.fill(QColor(WHITE))
    p = QPainter(img)
    p.scale(dpi / PT_PER_IN, dpi / PT_PER_IN)
    render(spec, p, QRectF(0, 0, s.width(), s.height()), dpi)
    p.end()
    for key, value in metadata(spec, "png", dpi, width_mm).items():
        img.setText(str(key), value if isinstance(value, str) else json.dumps(value, ensure_ascii=False))
    return img


def svg_bytes(spec, width_mm=WIDTHS_MM["one_column"], dpi=300):
    from PySide6.QtSvg import QSvgGenerator

    s = size_pt(spec, width_mm)
    buffer = QBuffer()
    buffer.open(QIODevice.OpenModeFlag.WriteOnly)
    gen = QSvgGenerator()
    gen.setOutputDevice(buffer)
    gen.setResolution(int(PT_PER_IN))
    gen.setSize(QSize(math.ceil(s.width()), math.ceil(s.height())))
    gen.setViewBox(QRectF(0, 0, s.width(), s.height()))
    gen.setTitle(spec.title)
    gen.setDescription(json.dumps(metadata(spec, "svg", dpi, width_mm), ensure_ascii=False))
    p = QPainter(gen)
    render(spec, p, QRectF(0, 0, s.width(), s.height()), dpi)
    p.end()
    return bytes(buffer.data())


def write_pdf(spec, path, width_mm=WIDTHS_MM["one_column"], dpi=300):
    s = size_pt(spec, width_mm)
    writer = QPdfWriter(str(path))
    writer.setResolution(720)
    writer.setPageSize(QPageSize(s, QPageSize.Unit.Point, "figure"))
    writer.setPageMargins(QMarginsF(0, 0, 0, 0), QPageLayout.Unit.Point)
    writer.setTitle(spec.title)
    meta = metadata(spec, "pdf", dpi, width_mm)
    writer.setCreator(f"KK-VIM GUI {meta['kkvim_gui']} · {json.dumps(meta, ensure_ascii=False)}")
    if hasattr(writer, "setDocumentXmpMetadata"):
        writer.setDocumentXmpMetadata(xmp(meta))
    p = QPainter(writer)
    p.scale(720 / PT_PER_IN, 720 / PT_PER_IN)
    render(spec, p, QRectF(0, 0, s.width(), s.height()), dpi)
    p.end()


def xmp(meta):
    from xml.sax.saxutils import escape

    text = escape(json.dumps(meta, ensure_ascii=False))
    return QByteArray(("<?xpacket begin='' id='W5M0MpCehiHzreSzNTczkc9d'?><x:xmpmeta xmlns:x='adobe:ns:meta/'>"
                       "<rdf:RDF xmlns:rdf='http://www.w3.org/1999/02/22-rdf-syntax-ns#'><rdf:Description "
                       "xmlns:dc='http://purl.org/dc/elements/1.1/'><dc:description><rdf:Alt><rdf:li xml:lang='x-default'>"
                       f"{text}</rdf:li></rdf:Alt></dc:description></rdf:Description></rdf:RDF></x:xmpmeta>"
                       "<?xpacket end='w'?>").encode("utf-8"))


def save(spec, path, fmt=None, dpi=300, width_mm=WIDTHS_MM["one_column"]):
    """Write the figure: PNG (``dpi``), SVG or PDF by ``fmt`` or the file extension. Returns the path."""
    from pathlib import Path

    path = Path(path)
    fmt = (fmt or path.suffix.lstrip(".")).lower()
    if fmt not in FORMATS:
        raise ValueError(f"unknown figure format: {fmt!r}")
    if fmt == "png":
        if not image(spec, dpi, width_mm).save(str(path), "PNG"):
            raise OSError(f"figure not saved: {path}")
    elif fmt == "svg":
        path.write_bytes(svg_bytes(spec, width_mm, dpi))
    else:
        write_pdf(spec, path, width_mm, dpi)
    return path


def mime_data(spec, dpi=300, width_mm=120.0):
    """Clipboard content (GUI_DESIGN.md 20.4): a raster image and, for programs that accept it, SVG."""
    data = QMimeData()
    data.setImageData(image(spec, dpi, width_mm))
    data.setData("image/svg+xml", QByteArray(svg_bytes(spec, width_mm, dpi)))
    return data


def file_name(run_id, figure, lang, test_data, ext):
    """``<run_id>_<figure>_<en|ru>[_TEST-DATA].<ext>`` (GUI_DESIGN.md 8.3)."""
    return f"{run_id}_{figure}_{lang}{'_TEST-DATA' if test_data else ''}.{ext}"


def with_style(spec, styles):
    """The description with explicit curve styles {index: CurveStyle} applied (others keep theirs)."""
    return replace(spec, series=tuple(replace(s, style=styles.get(i, s.style)) for i, s in enumerate(spec.series)))
