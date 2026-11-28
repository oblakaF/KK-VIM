"""The combined PDF report (GUI_DESIGN.md 18.1 «Общий PDF-отчёт»; P7): A4 pages drawn with Qt (QPdfWriter).

A simple flowing layout in points: headings, paragraphs, key–value tables, tables with a repeated header, and figures
drawn by ``figures.publication`` (vector axes, lines and texts; the colour field as an embedded raster). Every page
carries the identification of the run and its marks; demonstration data get the TEST DATA watermark on every page.
No value is computed here: the caller passes the texts and numbers of the run.

Nothing is cut to fit. The page header is measured and wraps onto as many lines as the complete marks need; the body
starts below its actual height. Table columns are sized from their contents and a numeric cell is never shortened:
its sign, digits and exponent stay as given. A table whose numbers do not fit is set in a smaller font; only below
the smallest size does a number continue on the next line of its cell, character for character. Texts wrap. A heading
is kept on the page of the block that follows it.
"""

import re

from PySide6.QtCore import QMarginsF, QPointF, QRectF, QSizeF
from PySide6.QtGui import QColor, QPageLayout, QPageSize, QPainter, QPdfWriter

from kkvim_gui.figures import publication as pub

A4 = QSizeF(595.28, 841.89)  # pt
MARGIN = 48.0
BODY_PT = 8.5
RESOLUTION = 720
HEADER_PT = 6.5
HEADER_LINE = HEADER_PT * 1.3
HEADER_TOP = 16.0  # pt from the top edge to the top of the first header line
HEADER_GAP = 18.0  # minimum gap between the identification and the marks on one line
HEADER_BODY_GAP = 10.0  # from the rule under the header to the body
HEADER_GREY = "#555555"
MARK_RED = "#8A1C1C"
CELL_PAD = 2.0
MIN_TABLE_PT = 5.0
# 1.5, -4.54e-07, −1,94343·10⁻⁶, 12 %, inf, nan: the forms of repr() and of the interface ``fmt``
NUMBER = re.compile(r"^[+\-−]?(\d[\d  ]*([.,]\d*)?|[.,]\d+)"
                    r"([eE][+\-−]?\d+|·10[⁺⁻]?[⁰¹²³⁴-⁹]+)?( ?%)?$"
                    r"|^[+\-−]?(inf|nan)$", re.IGNORECASE)


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
        self.bottom = A4.height() - MARGIN
        self._pending = []  # headings waiting for the block that follows them (kept on its page)
        self.tables = []  # layouts of the drawn tables (font size, column widths, cell lines), for checks
        self.headings = []  # (page, baseline, text) of the drawn headings, for checks
        self.header_lines = self._header_layout()
        self._start_page()

    # ------------------------------------------------------------ pages
    def all_marks(self):
        return list(self.marks) + (["TEST DATA"] if self.test_data and "TEST DATA" not in self.marks else [])

    def _header_layout(self):
        """The header lines [(left text, right text)]: the run identification and the complete marks.

        The marks share the identification line (right-aligned) only when both fit with a gap. Otherwise the
        identification gets its own line(s) and the marks follow, joined by « · » and wrapped between marks; a single
        mark longer than the line wraps between its words. Nothing is shortened or left out.
        """
        marks = self.all_marks()
        joined = " · ".join(marks)
        ident = self.pp.wrap(self.header, self.width, HEADER_PT)
        if marks and len(ident) == 1 and (self.pp.rich_width(self.header, HEADER_PT) + HEADER_GAP
                                          + self.pp.rich_width(joined, HEADER_PT) <= self.width):
            return [(self.header, joined, None)]
        lines = [(line, None, None) for line in ident]
        current = ""
        for mark in marks:
            pieces = self.pp.wrap(mark, self.width, HEADER_PT)
            probe = f"{current} · {pieces[0]}" if current else pieces[0]
            if len(pieces) == 1 and self.pp.rich_width(probe + " ·", HEADER_PT) <= self.width:
                current = probe
                continue
            if current:
                lines.append((None, None, current + " ·"))
            lines.extend((None, None, piece) for piece in pieces[:-1])
            current = pieces[-1]
        if current:
            lines.append((None, None, current))
        return lines

    def header_height(self):
        return len(self.header_lines) * HEADER_LINE

    def _start_page(self):
        p = self.pp
        self.painter.fillRect(QRectF(0, 0, A4.width(), A4.height()), QColor(pub.WHITE))
        y = HEADER_TOP
        for left, right, mark_line in self.header_lines:
            y += HEADER_LINE
            if left is not None:
                p.rich(MARGIN, y, left, HEADER_PT, colour=HEADER_GREY, plain=True)
            if right is not None:
                p.rich(A4.width() - MARGIN, y, right, HEADER_PT, colour=MARK_RED, align="right", plain=True)
            if mark_line is not None:
                p.rich(MARGIN, y, mark_line, HEADER_PT, colour=MARK_RED, plain=True)
        y += HEADER_PT * 0.6
        self.painter.setPen(pub.pen("#BBBBBB", 0.4))
        self.painter.drawLine(QPointF(MARGIN, y), QPointF(MARGIN + self.width, y))
        self.header_bottom = y
        p.rich(A4.width() / 2, A4.height() - MARGIN + 22, f"— {self.page} —", HEADER_PT, colour=HEADER_GREY,
               align="centre", plain=True)
        if self.test_data:
            pub.watermark(p, QRectF(0, 0, A4.width(), A4.height()))
        self.top = max(MARGIN, y + HEADER_BODY_GAP)
        self.y = self.top

    def new_page(self):
        self.writer.newPage()
        self.page += 1
        self._start_page()

    def need(self, height):
        """Room for ``height`` points and the pending headings; the headings are then drawn above the block."""
        extra = sum(h for _, _, _, h in self._pending)
        if self.y + extra + height > self.bottom and self.y > self.top:
            self.new_page()
        pending, self._pending = self._pending, []
        for lines, size, before, _ in pending:
            self.y += before
            for line in lines:
                self.y += size * 1.25
                self.pp.rich(MARGIN, self.y, line, size, bold=True, plain=True)
                self.headings.append((self.page, self.y, line))
            self.y += size * 0.45

    # ------------------------------------------------------------ blocks
    def heading(self, text, level=1):
        """A heading, drawn together with the first part of the next block (never alone at the foot of a page)."""
        size = {0: 15.0, 1: 11.5, 2: 9.5}[level]
        lines = self.pp.wrap(text, self.width, size)
        before = size * (0.6 if level else 0.2)
        self._pending.append((lines, size, before, before + len(lines) * size * 1.25 + size * 0.45))

    def paragraph(self, text, size=BODY_PT, colour=pub.BLACK):
        lines = self.pp.wrap(text, self.width, size)
        for k, line in enumerate(lines):
            self.need(size * 1.4 * (min(2, len(lines)) if k == 0 else 1))  # a pending heading keeps two lines
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
        """A table: column widths from the contents, numbers complete, texts wrapped, the header row repeated."""
        headers = [str(h) for h in headers]
        rows = [[str(v) for v in row] for row in rows]
        layout = table_layout(self.pp, headers, rows, self.width, size)
        size, widths = layout["size"], layout["widths"]
        line_h = size * 1.3
        xs = [MARGIN]
        for w in widths[:-1]:
            xs.append(xs[-1] + w)

        def cells(values, bold=False):
            return [cell_lines(self.pp, v, w - 2 * CELL_PAD, size, bold) for v, w in zip(values, widths)]

        def draw(row_cells, bold=False):
            for x, lines in zip(xs, row_cells):
                for k, line in enumerate(lines):
                    self.pp.rich(x + CELL_PAD, self.y + size * 1.15 + k * line_h, line, size, bold=bold, plain=True)

        head_cells = cells(headers, bold=True)
        head_h = max(len(c) for c in head_cells) * line_h + 4

        def head():
            self.painter.fillRect(QRectF(MARGIN, self.y, self.width, head_h), QColor("#EEF1F4"))
            draw(head_cells, bold=True)
            self.y += head_h

        body = [cells(row) for row in rows]
        heights = [max(len(c) for c in rc) * line_h + 2 for rc in body]
        self.need(head_h + sum(heights[:2]))
        head()
        for row_cells, h in zip(body, heights):
            if self.y + h > self.bottom:
                self.new_page()
                head()
            draw(row_cells)
            self.y += h
        layout["cells"] = [head_cells, *body]
        self.tables.append(layout)
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
        if self._pending:
            self.need(0)
        self.painter.end()


# ------------------------------------------------------------------------------------------------ tables


def is_number(text):
    """A numeric cell (sign, digits, decimal part, exponent, optional %): one token, never broken or shortened."""
    return bool(NUMBER.match(str(text).strip()))


def _words(pp, text, size, bold=False):
    m = pp.metrics(pp.font(size, bold))
    return [m.horizontalAdvance(w) for w in str(text).split()] or [0.0]


def table_layout(pp, headers, rows, width, size):
    """The font size and the column widths of a table.

    A column needs at least its widest number and its longest word of text. What remains of the width is shared in
    proportion to how much each column would need to put its full contents on one line. If the minima do not fit at
    ``size``, the font shrinks in 0.25 pt steps down to MIN_TABLE_PT. Past that the columns are proportional to their
    minima, and an over-long number continues on the next line of its cell (``cell_lines``) with every character kept.
    """
    n = len(headers)
    columns = [[r[c] if c < len(r) else "" for r in rows] for c in range(n)]
    s = size
    while True:
        m, mb = pp.metrics(pp.font(s)), pp.metrics(pp.font(s, True))
        minimum, natural = [], []
        for c in range(n):
            need = max(_words(pp, headers[c], s, True))
            full = mb.horizontalAdvance(headers[c])
            for v in columns[c]:
                adv = m.horizontalAdvance(v)
                need = max(need, adv if is_number(v) else max(_words(pp, v, s)))
                full = max(full, adv)
            minimum.append(need + 2 * CELL_PAD + 0.5)
            natural.append(full + 2 * CELL_PAD + 0.5)
        if sum(minimum) <= width or s <= MIN_TABLE_PT:
            break
        s = max(MIN_TABLE_PT, s - 0.25)
    if sum(minimum) > width:
        widths, whole = [width * w / sum(minimum) for w in minimum], False
    elif sum(natural) <= width:
        widths, whole = [w + (width - sum(natural)) * w / sum(natural) for w in natural], True
    else:
        grow = [a - b for a, b in zip(natural, minimum)]
        spare = width - sum(minimum)
        widths, whole = [b + spare * g / sum(grow) for b, g in zip(minimum, grow)], True
    return {"size": s, "widths": widths, "numbers_whole": whole}


def cell_lines(pp, text, width, size, bold=False):
    """The lines of a cell: a number whole on one line (else continued by characters, none lost); a text wrapped."""
    text = str(text)
    m = pp.metrics(pp.font(size, bold))
    if m.horizontalAdvance(text) <= width + 0.01:
        return [text]
    if is_number(text):
        return _split_chars(m, text, width)
    lines = []
    for paragraph in text.splitlines() or [""]:
        line = ""
        for word in paragraph.split(" "):
            probe = f"{line} {word}".strip()
            if line and m.horizontalAdvance(probe) > width + 0.01:
                lines.append(line)
                line = word
            else:
                line = probe
        lines.append(line)
    return [piece for line in lines
            for piece in ([line] if m.horizontalAdvance(line) <= width + 0.01 else _split_chars(m, line, width))]


def _split_chars(m, text, width):
    lines, line = [], ""
    for ch in text:
        if line and m.horizontalAdvance(line + ch) > width:
            lines.append(line)
            line = ch
        else:
            line += ch
    return lines + [line]
