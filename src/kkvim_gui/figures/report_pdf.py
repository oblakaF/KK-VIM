"""The combined PDF report (GUI_DESIGN.md 18.1 «Общий PDF-отчёт»; P7): A4 pages drawn with Qt (QPdfWriter).

A simple flowing layout in points: headings, paragraphs, key–value tables, tables with a repeated header, and figures
drawn by ``figures.publication`` (vector axes, lines and texts; the colour field as an embedded raster). Every page
carries the identification of the run and its marks; demonstration data get the TEST DATA watermark on every page.
No value is computed here: the caller passes the texts and numbers of the run.
"""

from PySide6.QtCore import QMarginsF, QPointF, QRectF, QSizeF
from PySide6.QtGui import QColor, QPageLayout, QPageSize, QPainter, QPdfWriter

from kkvim_gui.figures import publication as pub

A4 = QSizeF(595.28, 841.89)  # pt
MARGIN = 48.0
BODY_PT = 8.5
RESOLUTION = 720


class PdfReport:
    def __init__(self, path, title, header, marks=(), test_data=False, meta=None):
        self.path, self.title, self.header, self.marks, self.test_data = path, title, header, tuple(marks), test_data
        self.writer = QPdfWriter(str(path))
        self.writer.setResolution(RESOLUTION)
        self.writer.setPageSize(QPageSize(QPageSize.PageSizeId.A4))
        self.writer.setPageMargins(QMarginsF(0, 0, 0, 0), QPageLayout.Unit.Point)
        self.writer.setTitle(title)
        self.writer.setCreator(f"KK-VIM GUI {pub.program_version()}")
        if meta and hasattr(self.writer, "setDocumentXmpMetadata"):
            self.writer.setDocumentXmpMetadata(pub.xmp(meta))
        self.painter = QPainter(self.writer)
        self.painter.scale(RESOLUTION / pub.PT_PER_IN, RESOLUTION / pub.PT_PER_IN)
        self.pp = pub.Painter(self.painter)
        self.page = 1
        self.width = A4.width() - 2 * MARGIN
        self._start_page()

    # ------------------------------------------------------------ pages
    def _start_page(self):
        p = self.pp
        self.painter.fillRect(QRectF(0, 0, A4.width(), A4.height()), QColor(pub.WHITE))
        p.rich(MARGIN, MARGIN - 18, self.header, 6.5, colour="#555555", plain=True)
        marks = list(self.marks) + (["TEST DATA"] if self.test_data else [])
        if marks:
            p.rich(A4.width() - MARGIN, MARGIN - 18, " · ".join(marks), 6.5, colour="#8A1C1C", align="right", plain=True)
        p.rich(A4.width() / 2, A4.height() - MARGIN + 22, f"— {self.page} —", 6.5, colour="#555555", align="centre",
               plain=True)
        if self.test_data:
            pub.watermark(p, QRectF(0, 0, A4.width(), A4.height()))
        self.y = MARGIN

    def new_page(self):
        self.writer.newPage()
        self.page += 1
        self._start_page()

    def need(self, height):
        if self.y + height > A4.height() - MARGIN:
            self.new_page()

    # ------------------------------------------------------------ blocks
    def heading(self, text, level=1):
        size = {0: 15.0, 1: 11.5, 2: 9.5}[level]
        self.need(size * 3)
        self.y += size * (0.6 if level else 0.2)
        for line in self.pp.wrap(text, self.width, size):
            self.y += size * 1.25
            self.pp.rich(MARGIN, self.y, line, size, bold=True, plain=True)
        self.y += size * 0.45

    def paragraph(self, text, size=BODY_PT, colour=pub.BLACK):
        for line in self.pp.wrap(text, self.width, size):
            self.need(size * 1.4)
            self.y += size * 1.35
            self.pp.rich(MARGIN, self.y, line, size, colour=colour, plain=True)
        self.y += size * 0.4

    def key_values(self, rows, key_width=0.34):
        kw = self.width * key_width
        for key, value in rows:
            lines_k = self.pp.wrap(str(key), kw - 6, BODY_PT)
            lines_v = self.pp.wrap(str(value), self.width - kw, BODY_PT)
            height = max(len(lines_k), len(lines_v)) * BODY_PT * 1.3 + 3
            self.need(height)
            y = self.y
            for k, line in enumerate(lines_k):
                self.pp.rich(MARGIN, y + BODY_PT * 1.1 + k * BODY_PT * 1.3, line, BODY_PT, colour="#333333", bold=True,
                             plain=True)
            for k, line in enumerate(lines_v):
                self.pp.rich(MARGIN + kw, y + BODY_PT * 1.1 + k * BODY_PT * 1.3, line, BODY_PT, plain=True)
            self.y += height
            self.painter.setPen(pub.pen("#DDDDDD", 0.4))
            self.painter.drawLine(QPointF(MARGIN, self.y), QPointF(MARGIN + self.width, self.y))
        self.y += 4

    def table(self, headers, rows, size=7.0):
        n = len(headers)
        col = self.width / n
        row_h = size * 1.45

        def head():
            self.painter.fillRect(QRectF(MARGIN, self.y, self.width, row_h + 2), QColor("#EEF1F4"))
            for c, h in enumerate(headers):
                self.pp.rich(MARGIN + c * col + 2, self.y + size * 1.15, _clip(self.pp, h, col - 4, size), size, bold=True,
                             plain=True)
            self.y += row_h + 2

        self.need(row_h * 3)
        head()
        for row in rows:
            if self.y + row_h > A4.height() - MARGIN:
                self.new_page()
                head()
            for c, v in enumerate(row):
                self.pp.rich(MARGIN + c * col + 2, self.y + size * 1.1, _clip(self.pp, str(v), col - 4, size), size,
                             plain=True)
            self.y += row_h
        self.y += 6

    def figure(self, spec, caption="", width_mm=160.0):
        size = pub.size_pt(spec, width_mm)
        scale = min(1.0, self.width / size.width())
        w, h = size.width() * scale, size.height() * scale
        self.need(h + 30)
        self.painter.save()
        self.painter.translate(MARGIN + (self.width - w) / 2, self.y)
        self.painter.scale(scale, scale)
        pub.render(spec, self.painter, QRectF(0, 0, size.width(), size.height()), dpi=300)
        self.painter.restore()
        self.y += h + 4
        if caption:
            self.paragraph(caption, 7.5, "#333333")

    def close(self):
        self.painter.end()


def _clip(pp, text, width, size):
    m = pp.metrics(pp.font(size))
    if m.horizontalAdvance(text) <= width:
        return text
    while text and m.horizontalAdvance(text + "…") > width:
        text = text[:-1]
    return text + "…"

