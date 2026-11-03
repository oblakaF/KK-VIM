"""Plate scheme on the setup page (GUI_DESIGN.md, 5.2; stage P2.2).

Vector drawing with QPainter from the current setup model (not the run snapshot):
- top view of the rectangular plate in true a : b proportion, with dimension lines;
- small side view with the thickness h (marked "не в масштабе" when exaggerated);
- uniform load q as a comb of lines perpendicular to the plate, WITHOUT arrowheads:
  the sign convention of the KK-VIM formulation is not accepted yet (D-029);
- SSSS (triangles) and CCCC (hatched edge) symbols with a short legend.
Coordinate axes are not drawn until the formulation fixes their directions.
When a, b or h is invalid the scheme says what it is waiting for and draws no geometry.
Colours come from the theme tokens; sizes are logical pixels (sharp on HiDPI).
"""

from dataclasses import dataclass

from PySide6.QtCore import QPointF, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QFont, QFontMetricsF, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QSizePolicy, QWidget

from kkvim_gui.model.scheme import scheme_data

SUPPORT_LEGEND = {
    "SSSS": "SSSS — шарнирное опирание всех четырёх краёв",
    "CCCC": "CCCC — жёсткая заделка всех четырёх краёв",
}
LOAD_LEGEND = "равномерно распределённая по всей площади"
AXES_NOTE = "Оси и направление нагрузки не показаны: постановка КК-ВИМ ещё не утверждена."
WAITING = "Схема ожидает исправления параметров:"
SIDE_TITLE = "вид сбоку"
NOT_TO_SCALE = "не в масштабе"
EXTREME_ASPECT = 20.0

PAD = 12.0          # outer padding
SUPPORT = 12.0      # room for support symbols around the plan
SIDE_MIN_LEN = 120.0
SIDE_T_MIN, SIDE_T_MAX = 6.0, 14.0
COMB_H = 16.0


@dataclass
class SchemeLayout:
    plan: QRectF
    side: QRectF | None
    side_to_scale: bool
    legend_top: float
    a_label_pos: QPointF | None
    b_label_pos: QPointF | None


class PlateScheme(QWidget):
    def __init__(self, session, model, parent=None):
        super().__init__(parent)
        self.session = session
        self.model = model
        self.layout_info = None
        self.setMinimumSize(260, 260)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAccessibleName("Схема пластины")
        model.changed.connect(self._on_change)
        session.themeChanged.connect(self._on_change)
        self._on_change()

    def sizeHint(self):
        return QSize(420, 420)

    def data(self):
        return scheme_data(self.model.state)

    def _on_change(self, *_):
        d = self.data()
        if d.valid:
            desc = f"{d.a_label}, {d.b_label}, {d.h_label}, {d.q_label}, {d.supports}"
        else:
            desc = f"{WAITING} {', '.join(d.waiting_for)}"
        self.setAccessibleDescription(desc)
        self.update()

    # Layout
    def _fonts(self):
        base = QFont(self.font())
        small = QFont(base)
        small.setPointSizeF(max(7.5, base.pointSizeF() * 0.9))
        return base, small

    def compute_layout(self, rect, d):
        base, small = self._fonts()
        fm, fs = QFontMetricsF(base), QFontMetricsF(small)
        legend_h = 2 * fm.height() + fs.height() * 2 + 14
        side_h = COMB_H + SIDE_T_MAX + fs.height() + 14
        dim_a_h = fm.height() + 24  # dimension line, label and a gap before the side view
        dim_b_w = (fm.horizontalAdvance(d.b_label) + 22) if d.valid else 0.0
        x0 = rect.left() + PAD + SUPPORT
        y0 = rect.top() + PAD + SUPPORT
        w_avail = max(10.0, rect.width() - 2 * (PAD + SUPPORT) - dim_b_w)
        h_avail = max(10.0, rect.height() - 2 * PAD - 2 * SUPPORT - dim_a_h - side_h - legend_h)
        legend_top = rect.bottom() - PAD - legend_h + 6
        if not d.valid:
            area = QRectF(rect.left() + PAD, rect.top() + PAD,
                          rect.width() - 2 * PAD, legend_top - rect.top() - 2 * PAD)
            return SchemeLayout(area, None, True, legend_top, None, None)
        scale = min(w_avail / d.a_m, h_avail / d.b_m)
        pw, ph = d.a_m * scale, d.b_m * scale  # true proportion a : b
        px = x0 + (w_avail - pw) / 2
        # Plan, dimension and side view form one group, centred above the legend (no gaps).
        group = ph + SUPPORT + dim_a_h + side_h
        py = y0 + max(0.0, (h_avail + dim_a_h + side_h - group) / 2)
        plan = QRectF(px, py, pw, ph)
        a_pos = QPointF(plan.center().x(), plan.bottom() + SUPPORT + 8)
        b_pos = QPointF(plan.right() + SUPPORT + 8, plan.center().y())
        # Side view: same length as the plan when readable, otherwise a schematic length.
        side_len = min(max(pw, SIDE_MIN_LEN), rect.width() - 2 * PAD - fm.horizontalAdvance(d.h_label) - 16)
        true_t = d.h_over_a * side_len
        t = min(max(true_t, SIDE_T_MIN), SIDE_T_MAX)
        to_scale = abs(side_len - pw) < 0.5 and abs(t - true_t) < 0.5
        side_top = plan.bottom() + SUPPORT + dim_a_h + COMB_H
        sx = max(rect.left() + PAD, min(px, rect.right() - PAD - side_len))
        side = QRectF(sx, side_top, side_len, t)
        return SchemeLayout(plan, side, to_scale, legend_top, a_pos, b_pos)

    # Painting
    def paintEvent(self, event):
        t = self.session.tokens
        d = self.data()
        rect = QRectF(self.rect())
        lay = self.compute_layout(rect, d)
        self.layout_info = lay
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        base, small = self._fonts()
        p.setFont(base)
        if d.valid:
            self._draw_plan(p, lay, d, t)
            self._draw_side(p, lay, d, t, small)
        else:
            self._draw_waiting(p, lay, d, t)
        self._draw_legend(p, lay, d, t, base, small, rect)
        p.end()

    @staticmethod
    def _pen(colour, width=1.0, style=Qt.PenStyle.SolidLine):
        pen = QPen(QColor(colour), width, style)
        pen.setCosmetic(True)
        return pen

    def _draw_waiting(self, p, lay, d, t):
        p.setPen(self._pen(t["status.warning"], 1.2, Qt.PenStyle.DashLine))
        p.setBrush(Qt.BrushStyle.NoBrush)
        area = lay.plan.adjusted(8, 8, -8, -8)
        p.drawRoundedRect(area, 6, 6)
        p.setPen(QColor(t["status.warning"]))
        text = WAITING + "\n" + ", ".join(d.waiting_for) + "."
        p.drawText(area.adjusted(12, 12, -12, -12),
                   int(Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap), text)

    def _draw_plan(self, p, lay, d, t):
        plan = lay.plan
        fill = QColor(t["accent.soft"])
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(fill)
        p.drawRect(plan)
        # Uniform load over the whole area: neutral dot pattern (no direction).
        if d.q_valid:
            p.setPen(self._pen(t["accent"], 1.6))
            step = 12.0
            y = plan.top() + step / 2
            while y < plan.bottom():
                x = plan.left() + step / 2
                while x < plan.right():
                    p.drawPoint(QPointF(x, y))
                    x += step
                y += step
        p.setPen(self._pen(t["text.primary"], 1.6))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(plan)
        self._draw_supports(p, plan, d.supports, t)
        self._draw_dimensions(p, lay, d, t)

    def _edges(self, r):
        """(start, end, outward normal) for the four edges."""
        return (
            (r.topLeft(), r.topRight(), QPointF(0, -1)),
            (r.topRight(), r.bottomRight(), QPointF(1, 0)),
            (r.bottomRight(), r.bottomLeft(), QPointF(0, 1)),
            (r.bottomLeft(), r.topLeft(), QPointF(-1, 0)),
        )

    def _draw_supports(self, p, r, supports, t):
        colour = t["text.secondary"]
        for a, b, n in self._edges(r):
            length = ((b.x() - a.x()) ** 2 + (b.y() - a.y()) ** 2) ** 0.5
            ux, uy = (b.x() - a.x()) / length, (b.y() - a.y()) / length
            if supports == "SSSS":
                self._triangles(p, a, ux, uy, n, length, colour)
            else:
                self._hatch(p, a, b, ux, uy, n, length, colour)

    def _triangles(self, p, a, ux, uy, n, length, colour, size=7.0):
        p.setPen(self._pen(colour, 1.1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        count = max(1, int(length // 24))
        for i in range(count):
            s = length * (i + 0.5) / count
            apex = QPointF(a.x() + ux * s, a.y() + uy * s)
            base = QPointF(apex.x() + n.x() * size, apex.y() + n.y() * size)
            half = size * 0.6
            p.drawPolygon(QPolygonF([
                apex,
                QPointF(base.x() + ux * half, base.y() + uy * half),
                QPointF(base.x() - ux * half, base.y() - uy * half),
            ]))

    def _hatch(self, p, a, b, ux, uy, n, length, colour, depth=7.0, step=6.0):
        p.setPen(self._pen(colour, 2.2))
        p.drawLine(a, b)
        p.setPen(self._pen(colour, 1.0))
        s = step / 2
        while s < length:
            start = QPointF(a.x() + ux * s, a.y() + uy * s)
            end = QPointF(start.x() + n.x() * depth - ux * depth * 0.6,
                          start.y() + n.y() * depth - uy * depth * 0.6)
            p.drawLine(start, end)
            s += step

    def _draw_dimensions(self, p, lay, d, t):
        plan = lay.plan
        colour = QColor(t["text.secondary"])
        p.setPen(self._pen(t["text.secondary"], 1.0))
        tick = 4.0
        # a: below the plan
        y = lay.a_label_pos.y()
        p.drawLine(QPointF(plan.left(), y), QPointF(plan.right(), y))
        for x in (plan.left(), plan.right()):
            p.drawLine(QPointF(x - tick, y + tick), QPointF(x + tick, y - tick))
            p.drawLine(QPointF(x, plan.bottom() + SUPPORT - 2), QPointF(x, y + tick))
        fm = QFontMetricsF(p.font())
        p.setPen(colour)
        text_w = fm.horizontalAdvance(d.a_label)
        tx = min(max(plan.center().x() - text_w / 2, PAD), self.width() - PAD - text_w)
        p.drawText(QPointF(tx, y + fm.ascent() + 4), d.a_label)
        # b: right of the plan
        p.setPen(self._pen(t["text.secondary"], 1.0))
        x = lay.b_label_pos.x()
        p.drawLine(QPointF(x, plan.top()), QPointF(x, plan.bottom()))
        for yy in (plan.top(), plan.bottom()):
            p.drawLine(QPointF(x - tick, yy + tick), QPointF(x + tick, yy - tick))
            p.drawLine(QPointF(plan.right() + SUPPORT - 2, yy), QPointF(x + tick, yy))
        p.setPen(colour)
        p.drawText(QPointF(x + 8, plan.center().y() + fm.ascent() / 2 - 2), d.b_label)

    def _draw_side(self, p, lay, d, t, small):
        side = lay.side
        # Load comb: lines perpendicular to the plate, no arrowheads (sign not fixed).
        if d.q_valid:
            p.setPen(self._pen(t["accent"], 1.2))
            top = side.top() - COMB_H
            p.drawLine(QPointF(side.left(), top), QPointF(side.right(), top))
            count = max(2, int(side.width() // 8))
            for i in range(count + 1):
                xx = side.left() + side.width() * i / count
                p.drawLine(QPointF(xx, top), QPointF(xx, side.top() - 1))
        p.setPen(self._pen(t["text.primary"], 1.4))
        p.setBrush(QColor(t["accent.soft"]))
        p.drawRect(side)
        colour = t["text.secondary"]
        if d.supports == "SSSS":
            for x in (side.left(), side.right()):
                apex = QPointF(x, side.bottom())
                p.setPen(self._pen(colour, 1.1))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawPolygon(QPolygonF([apex, QPointF(x - 5, side.bottom() + 8),
                                         QPointF(x + 5, side.bottom() + 8)]))
        else:
            for x, n in ((side.left(), QPointF(-1, 0)), (side.right(), QPointF(1, 0))):
                a = QPointF(x, side.top() - 6)
                b = QPointF(x, side.bottom() + 6)
                self._hatch(p, a, b, 0.0, 1.0, n, b.y() - a.y(), colour, depth=6.0, step=5.0)
        fm = QFontMetricsF(p.font())
        p.setPen(QColor(t["text.primary"]))
        p.drawText(QPointF(side.right() + 12, side.center().y() + fm.ascent() / 2 - 2), d.h_label)
        p.setFont(small)
        fs = QFontMetricsF(small)
        caption = SIDE_TITLE if lay.side_to_scale else f"{SIDE_TITLE} · {NOT_TO_SCALE}"
        p.setPen(QColor(t["text.secondary"]))
        p.drawText(QPointF(side.left(), side.bottom() + 10 + fs.ascent()), caption)
        p.setFont(self._fonts()[0])

    def _draw_legend(self, p, lay, d, t, base, small, rect):
        x = rect.left() + PAD
        width = rect.width() - 2 * PAD
        fm, fs = QFontMetricsF(base), QFontMetricsF(small)
        y = lay.legend_top
        # Supports
        sample = QRectF(x, y + 3, 22, fm.height() - 6)
        if d.supports == "SSSS":
            self._triangles(p, QPointF(sample.left(), sample.bottom()), 1.0, 0.0, QPointF(0, 1),
                            sample.width(), t["text.secondary"], size=6.0)
            p.setPen(self._pen(t["text.primary"], 1.4))
            p.drawLine(QPointF(sample.left(), sample.bottom()), QPointF(sample.right(), sample.bottom()))
        else:
            self._hatch(p, QPointF(sample.left(), sample.bottom()), QPointF(sample.right(), sample.bottom()),
                        1.0, 0.0, QPointF(0, 1), sample.width(), t["text.secondary"], depth=5.0, step=5.0)
        p.setFont(base)
        p.setPen(QColor(t["text.primary"]))
        text_x = x + 30
        p.drawText(QPointF(text_x, y + fm.ascent()),
                   fm.elidedText(SUPPORT_LEGEND[d.supports], Qt.TextElideMode.ElideRight, width - 30))
        # Load
        y += fm.height() + 4
        if d.q_valid:
            p.setPen(self._pen(t["accent"], 1.2))
            for i in range(4):
                xx = x + 2 + i * 6
                p.drawLine(QPointF(xx, y + 3), QPointF(xx, y + fm.height() - 4))
            p.drawLine(QPointF(x + 2, y + 3), QPointF(x + 20, y + 3))
            colour, text = t["text.primary"], f"{d.q_label} — {LOAD_LEGEND}"
        else:
            colour, text = t["status.warning"], d.q_label
        p.setPen(QColor(colour))
        p.drawText(QPointF(text_x, y + fm.ascent()),
                   fm.elidedText(text, Qt.TextElideMode.ElideRight, width - 30))
        # Note on axes and load direction (and on extreme proportions)
        y += fm.height() + 6
        p.setFont(small)
        p.setPen(QColor(t["text.secondary"]))
        note = AXES_NOTE
        if d.valid and (d.aspect > EXTREME_ASPECT or d.aspect < 1 / EXTREME_ASPECT):
            note = "Пропорции a : b сохранены. " + note
        p.drawText(QRectF(x, y, width, fs.height() * 2 + 2),
                   int(Qt.AlignmentFlag.AlignLeft | Qt.TextFlag.TextWordWrap), note)
        p.setFont(base)
