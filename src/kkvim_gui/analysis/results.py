"""Tabs of the analysis window for a solver result (GUI_DESIGN.md, 6).

Every figure and table is drawn from the arrays and rows of the ``RunOutcome`` (copies of the solver
result); nothing is recomputed here except unit scaling for display (w in mm = w in m · 1000) and the
search of the node with the largest |w|. The field figure is static (GUI-D13); X–Y plots keep the
pan and zoom of pyqtgraph. Every R4 figure carries PROVISIONAL / NOT_QUALIFIED; texts of figures
follow the session figure language (GUI_DESIGN.md, 7).
"""

import math

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QAbstractTableModel, QModelIndex, QRectF, Qt
from PySide6.QtGui import QTransform
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QScrollArea,
    QSplitter,
    QStackedWidget,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from kkvim_gui.figures.text import fig_text
from kkvim_gui.theme.tokens import SIZES
from kkvim_gui.widgets.common import Card, Note, SegmentedControl, StatusLabel, label

MM = 1000.0  # display of w in mm (SI submultiple, shown next to every value)
SUPERSCRIPT = str.maketrans("0123456789-+", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺")
STOP_STATUS = {"met": "success", "not_met": "warning", "error": "error", "none": "neutral"}


def fmt(value, digits=4):
    """Interface form of a value: decimal comma, 6,7·10⁻⁴, `digits` significant digits (GUI_DESIGN.md, 11)."""
    if value is None:
        return "—"
    if isinstance(value, (bool, np.bool_)):
        return "да" if value else "нет"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, str):
        return value
    v = float(value)
    if not math.isfinite(v):
        return str(v)
    text = f"{v:.{digits}g}"
    if "e" in text:
        mantissa, exponent = text.split("e")
        text = f"{mantissa}·10{str(int(exponent)).translate(SUPERSCRIPT)}"
    return text.replace(".", ",")


class RowsModel(QAbstractTableModel):
    """Read-only table: headers and a function row -> tuple of values (no copy of large arrays)."""

    def __init__(self, headers, count, row, digits=6, parent=None):
        super().__init__(parent)
        self._headers, self._count, self._row, self._digits = list(headers), count, row, digits

    def rowCount(self, parent=QModelIndex()):  # noqa: B008, N802 - Qt API
        return 0 if parent.isValid() else self._count

    def columnCount(self, parent=QModelIndex()):  # noqa: B008, N802 - Qt API
        return 0 if parent.isValid() else len(self._headers)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        value = self._row(index.row())[index.column()]
        if role == Qt.ItemDataRole.DisplayRole:
            return fmt(value, self._digits)
        if role == Qt.ItemDataRole.UserRole:
            return value
        if role == Qt.ItemDataRole.TextAlignmentRole and isinstance(value, (int, float, np.number)):
            return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):  # noqa: N802 - Qt API
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return self._headers[section]
        return None


def table_view(model):
    view = QTableView()
    view.setModel(model)
    view.verticalHeader().setVisible(False)
    view.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    view.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    view.setAlternatingRowColors(True)
    view.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
    view.horizontalHeader().setStretchLastSection(True)
    return view


def key_value(rows):
    grid = QGridLayout()
    grid.setHorizontalSpacing(16)
    grid.setVerticalSpacing(5)
    grid.setColumnStretch(1, 1)
    for i, (k, v) in enumerate(rows):
        grid.addWidget(label(k, "fieldLabel"), i, 0, Qt.AlignmentFlag.AlignTop)
        value = label(v, wrap=True)
        value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        grid.addWidget(value, i, 1)
    return grid


class Figure(QFrame):
    """Figure frame: title (figure language), the scientific mark of the result and a pyqtgraph plot."""

    def __init__(self, session, outcome, title_key, title_values=None, interactive=True, parent=None):
        super().__init__(parent)
        self.setProperty("figure", True)
        self.session, self.outcome = session, outcome
        self._title_key, self._title_values = title_key, title_values or {}
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 10, 14, 12)
        lay.setSpacing(6)
        top = QHBoxLayout()
        self.title = label("", "figureTitle")
        top.addWidget(self.title)
        top.addStretch(1)
        self.mark = label("", "figureMark")
        top.addWidget(self.mark)
        lay.addLayout(top)
        self.plot = pg.PlotWidget()
        self.plot.setMinimumHeight(180)
        for side in ("left", "bottom"):
            # values are shown in the unit of the axis label; no automatic SI prefix scaling (e.g. "x0.001")
            self.plot.getAxis(side).enableAutoSIPrefix(False)
        if not interactive:  # the field is static (GUI-D13)
            self.plot.setMouseEnabled(x=False, y=False)
            self.plot.hideButtons()
            self.plot.setMenuEnabled(False)
        lay.addWidget(self.plot, 1)
        self._axes = {}
        self.restyle()
        session.themeChanged.connect(self.restyle)
        session.figureLanguageChanged.connect(self.retranslate)

    def set_title(self, key, **values):
        self._title_key, self._title_values = key, values
        self.retranslate()

    def set_axes(self, bottom, left):
        self._axes = {"bottom": bottom, "left": left}
        self.retranslate()

    def retranslate(self, *_):
        lang = self.session.figure_language
        self.title.setText(fig_text(lang, self._title_key, **self._title_values))
        marks = []
        if self.outcome.is_error:
            marks.append(fig_text(lang, "mark.diagnostic"))
        if self.outcome.mark:
            marks.append(fig_text(lang, "mark.provisional"))
        self.mark.setText(" · ".join(marks))
        self.mark.setVisible(bool(marks))
        for side, key in self._axes.items():
            self.plot.setLabel(side, fig_text(lang, key))

    def restyle(self, *_):
        t = self.session.tokens
        self.plot.setBackground(t["plot.bg"])
        for side in ("left", "bottom"):
            axis = self.plot.getAxis(side)
            axis.setPen(t["plot.axis"])
            axis.setTextPen(t["plot.axis"])
        self.plot.showGrid(x=True, y=True, alpha=0.25)


class FieldFigure(Figure):
    """Static colour field of w (mm) or w̄ on the nodes of the result grid with a colour bar."""

    ISOLINES = 10  # contour levels evenly inside (min, max) of the shown quantity

    def __init__(self, session, outcome, compact=False, isolines=False, parent=None):
        super().__init__(session, outcome, "field.w.title", interactive=False, parent=parent)
        self.quantity, self.cmap_name = "w", "viridis"
        self.show_isolines = isolines
        self.isocurves = []
        self.image = pg.ImageItem()
        self.plot.addItem(self.image)
        x, y = outcome.x, outcome.y
        hx, hy = x[1] - x[0], y[1] - y[0]
        # every image pixel is centred on its node (axis 0 = x, axis 1 = y: column-major, as the solver array)
        self.rect = QRectF(x[0] - hx / 2, y[0] - hy / 2, x[-1] - x[0] + hx, y[-1] - y[0] + hy)
        # isolines are traced through the node values: node (i, j) is drawn at (x_i, y_j)
        self.node_transform = QTransform.fromTranslate(float(x[0]), float(y[0])).scale(float(hx), float(hy))
        self.plot.setAspectLocked(True)
        self.bar = None if compact else pg.ColorBarItem(interactive=False, width=14)
        if self.bar is not None:
            self.bar.setImageItem(self.image, insert_in=self.plot.getPlotItem())
        self.set_axes("axis.x", "axis.y")
        self.update_image()

    def values(self):
        return self.outcome.w * MM if self.quantity == "w" else self.outcome.w_bar

    def set_quantity(self, quantity):
        self.quantity = quantity
        self.set_title("field.w.title" if quantity == "w" else "field.wbar.title")
        self.update_image()

    def set_colormap(self, name):
        self.cmap_name = name
        self.update_image()

    def set_isolines(self, on):
        self.show_isolines = bool(on)
        self.update_isolines()

    def isoline_levels(self):
        values = self.values()
        low, high = float(values.min()), float(values.max())
        if not high > low:
            return []
        return [low + (high - low) * k / (self.ISOLINES + 1) for k in range(1, self.ISOLINES + 1)]

    def update_isolines(self):
        for curve in self.isocurves:
            self.plot.removeItem(curve)
        self.isocurves = []
        if not self.show_isolines:
            return
        values = np.asarray(self.values())
        pen = pg.mkPen(self.session.tokens["plot.bg"], width=1)
        for level in self.isoline_levels():
            curve = pg.IsocurveItem(data=values, level=level, pen=pen)
            curve.setTransform(self.node_transform)
            curve.setZValue(10)
            self.plot.addItem(curve)
            self.isocurves.append(curve)

    def update_image(self):
        values = self.values()
        low, high = float(values.min()), float(values.max())
        if low == high:
            low, high = low - 0.5, high + 0.5
        cmap = pg.colormap.get(self.cmap_name)
        self.image.setImage(values, levels=(low, high), autoLevels=False)
        self.image.setRect(self.rect)  # after setImage: the scale needs the image size
        self.image.setColorMap(cmap)
        self.plot.setRange(rect=self.rect, padding=0.02)
        if self.bar is not None:
            self.bar.setColorMap(cmap)
            self.bar.setLevels((low, high))
            lang = self.session.figure_language
            self.bar.axis.setLabel(fig_text(lang, "colorbar.w" if self.quantity == "w" else "colorbar.wbar"))
        self.update_isolines()

    def retranslate(self, *_):
        super().retranslate()
        if getattr(self, "bar", None) is not None:
            lang = self.session.figure_language
            self.bar.axis.setLabel(fig_text(lang, "colorbar.w" if self.quantity == "w" else "colorbar.wbar"))


def toolbar(*widgets, trailing=()):
    bar = QFrame()
    bar.setObjectName("TabToolbar")
    lay = QHBoxLayout(bar)
    lay.setContentsMargins(16, 6, 16, 6)
    lay.setSpacing(10)
    for w in widgets:
        lay.addWidget(w)
    lay.addStretch(1)
    for w in trailing:
        lay.addWidget(w)
    return bar


def labelled(text, widget):
    box = QWidget()
    lay = QHBoxLayout(box)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(6)
    lay.addWidget(label(text, "fieldLabel"))
    lay.addWidget(widget)
    return box


def page(bar, content):
    w = QWidget()
    w.setObjectName("Page")
    lay = QVBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(0)
    if bar is not None:
        lay.addWidget(bar)
    lay.addWidget(content, 1)
    return w


def padded(layout):
    w = QWidget()
    layout.setContentsMargins(16, 14, 16, 16)
    layout.setSpacing(12)
    w.setLayout(layout)
    return w


def statuses(outcome, run):
    """(kind, text) of execution, stop and scientific status of the result."""
    execution = (("error", f"Расчёт не завершён: ошибка решателя {outcome.status}; поле — только диагностика")
                 if outcome.is_error else ("success", "Расчёт выполнен"))
    kind = STOP_STATUS[outcome.stop_kind]
    if outcome.mark and kind == "success":
        kind = "warning"  # an R4 stabilization is PROVISIONAL, not a success of the method
    stop = (kind, f"{outcome.stop_text} (s = {outcome.s}, k = {outcome.k})")
    science = (outcome.science_kind if outcome.mark else "neutral",
               f"Научный статус: {outcome.mark}" if outcome.mark else
               "Научный статус: решатель R3 ACCEPTED (D-043); расчёт — не научная приёмка")
    return execution, stop, science


def input_rows(outcome, model_rows):
    r = outcome.request
    rows = list(model_rows)
    rows.append(("В единицах СИ для решателя",
                 f"a {fmt(r.a, 6)} m · b {fmt(r.b, 6)} m · t {fmt(r.t, 6)} m · E {fmt(r.E, 6)} Pa · "
                 f"ν {fmt(r.nu, 6)} · q {fmt(r.q, 6)} Pa"))
    rows.append(("Жёсткость D (kkvim)", f"{fmt(outcome.D, 6)} N·m"))
    return rows


def key_values(outcome, elapsed_text):
    e = outcome.extremum()
    r = outcome.request
    rows = []
    if e is not None:
        rows += [
            ("Максимальный |w| (по узлам)", f"{fmt(e['w'] * MM, 6)} mm  ({fmt(e['w'], 6)} m)"),
            ("Положение узла", f"x = {fmt(e['x'], 6)} m, y = {fmt(e['y'], 6)} m (i = {e['i']}, j = {e['j']})"),
            ("Безразмерный прогиб w̄ в узле", f"{fmt(e['w_bar'], 6)}  (w̄ = wD/(qa⁴), MATH_SPEC 2.5)"),
        ]
    rows += [
        ("Статус решателя", outcome.status),
        ("Полушаги", f"s = {outcome.s}, k = {outcome.k} (D-030)"),
        ("Сетка, N", f"n_x = {r.n_x}, n_y = {r.n_y} · N = {r.N}"),
        ("Решатель", f"{outcome.solver}: {outcome.call}"),
        ("Время расчёта", elapsed_text),
    ]
    return rows


class OverviewTab(QWidget):
    def __init__(self, session, outcome, run, model_rows, elapsed_text, parent=None):
        super().__init__(parent)
        grid = QGridLayout()
        inputs = Card("Входные данные")
        inputs.body.addLayout(key_value(input_rows(outcome, model_rows)))
        inputs.body.addStretch(1)
        status_card = Card("Статусы")
        for kind, text in statuses(outcome, run):
            s = StatusLabel(kind, text)
            s.setWordWrap(True)
            status_card.body.addWidget(s)
        for note in outcome.notes:
            status_card.body.addWidget(Note(note, "testdata" if outcome.mark else "info"))
        status_card.body.addStretch(1)
        key = Card("Ключевые значения", right=label(outcome.mark or "", "tag"))
        key.body.addLayout(key_value(key_values(outcome, elapsed_text)))
        key.body.addStretch(1)
        grid.addWidget(inputs, 0, 0)
        grid.addWidget(status_card, 0, 1)
        grid.addWidget(key, 0, 2)
        self.field = FieldFigure(session, outcome, compact=True) if outcome.has_field else None
        if self.field is not None:
            grid.addWidget(self.field, 1, 0, 1, 3)
            grid.setRowStretch(1, 1)
        for c in range(3):
            grid.setColumnStretch(c, 1)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(padded(grid))


class FieldsTab(QWidget):
    """Colour field with isolines, or the axonometric view of the deformed surface with a labelled factor."""

    def __init__(self, session, outcome, model_rows, trailing=(), parent=None):
        super().__init__(parent)
        from kkvim_gui.analysis.surface import PRESETS, SurfaceFigure

        self.figure = FieldFigure(session, outcome, isolines=True)
        self.surface = SurfaceFigure(session, outcome)
        self.view = SegmentedControl({"map": "Карта и изолинии", "surface": "3D-вид"}, "map")
        self.quantity = SegmentedControl({"w": "Прогиб w, mm", "w_bar": "Безразмерный w̄"}, "w")
        self.quantity.changed.connect(self.figure.set_quantity)
        self.cmap = QComboBox()
        self.cmap.addItems(["viridis", "cividis"])
        self.cmap.currentTextChanged.connect(self.figure.set_colormap)
        self.isolines = QCheckBox("Изолинии")
        self.isolines.setChecked(True)
        self.isolines.toggled.connect(self.figure.set_isolines)
        self.factor = QComboBox()
        self.factor.addItem(f"Авто: ×{fmt(self.surface.auto, 3)}", self.surface.auto)
        for value in PRESETS:
            self.factor.addItem("×1 (истинный)" if value == 1.0 else f"×{fmt(value, 3)}", value)
        self.factor.setToolTip("Масштаб прогиба только на изображении 3D-вида; числовые значения не изменяются")
        self.factor.currentIndexChanged.connect(lambda i: self.surface.set_factor(self.factor.itemData(i)))
        self.map_controls = (self.quantity, labelled("Шкала", self.cmap), self.isolines)
        self.surface_controls = (labelled("Масштаб прогиба", self.factor),)
        bar = toolbar(self.view, *self.map_controls, *self.surface_controls, trailing=trailing)
        self.stack = QStackedWidget()
        self.stack.addWidget(self.figure)
        self.stack.addWidget(self.surface)
        self.view.changed.connect(self.set_view)
        self.set_view("map")
        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        box = QVBoxLayout()
        box.setContentsMargins(16, 14, 16, 16)
        box.addWidget(self.stack)
        body.addLayout(box, 1)
        body.addWidget(self._inspector(outcome, model_rows))
        content = QWidget()
        content.setLayout(body)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(page(bar, content))

    def set_view(self, view):
        surface = view == "surface"
        self.stack.setCurrentWidget(self.surface if surface else self.figure)
        for w in self.map_controls:
            w.setVisible(not surface)
        for w in self.surface_controls:
            w.setVisible(surface)
        if self.view.current() != view:
            self.view.set_current(view)

    @staticmethod
    def _inspector(outcome, model_rows):
        panel = QFrame()
        panel.setObjectName("Inspector")
        panel.setFixedWidth(SIZES["inspector.width"])
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(16, 14, 16, 16)
        lay.setSpacing(6)
        e = outcome.extremum()
        lay.addWidget(label("Максимальный |w| (по узлам)", "cardTitle"))
        lay.addWidget(label(f"w = {fmt(e['w'] * MM, 6)} mm"))
        lay.addWidget(label(f"w̄ = {fmt(e['w_bar'], 6)}"))
        lay.addWidget(label(f"x = {fmt(e['x'], 6)} m, y = {fmt(e['y'], 6)} m", "secondary"))
        lay.addWidget(label(f"узел i = {e['i']}, j = {e['j']}", "secondary"))
        lay.addSpacing(10)
        lay.addWidget(label("Сетка и метод", "cardTitle"))
        r = outcome.request
        lay.addWidget(label(f"n_x = {r.n_x}, n_y = {r.n_y}; N = {r.N}; {outcome.solver}", wrap=True))
        lay.addWidget(label(f"s = {outcome.s}, k = {outcome.k}; {outcome.status}", wrap=True))
        lay.addSpacing(10)
        lay.addWidget(label("Параметры модели", "cardTitle"))
        for k, v in model_rows:
            lay.addWidget(label(k, "fieldLabel"))
            lay.addWidget(label(v, wrap=True))
        lay.addStretch(1)
        if outcome.mark:
            lay.addWidget(label(outcome.mark, "figureMark"), 0, Qt.AlignmentFlag.AlignLeft)
        return panel


class SectionsTab(QWidget):
    """w(x) at a node row y_j and w(y) at a node column x_i, both from the stored field."""

    def __init__(self, session, outcome, trailing=(), parent=None):
        super().__init__(parent)
        self.outcome = outcome
        e = outcome.extremum()
        self.at_y, self.at_x = QComboBox(), QComboBox()
        for j, y in enumerate(outcome.y):
            self.at_y.addItem(f"y = {fmt(y, 6)} m (j = {j})", j)
        for i, x in enumerate(outcome.x):
            self.at_x.addItem(f"x = {fmt(x, 6)} m (i = {i})", i)
        self.fig_x = Figure(session, outcome, "section.x.title", {"y": float(outcome.y[e["j"]])})
        self.fig_y = Figure(session, outcome, "section.y.title", {"x": float(outcome.x[e["i"]])})
        self.fig_x.set_axes("axis.x", "axis.w")
        self.fig_y.set_axes("axis.y", "axis.w")
        self.curve_x = self.fig_x.plot.plot(pen=pg.mkPen(session.tokens["accent"], width=2))
        self.curve_y = self.fig_y.plot.plot(pen=pg.mkPen(session.tokens["accent"], width=2))
        self.at_y.currentIndexChanged.connect(lambda _: self.update_x())
        self.at_x.currentIndexChanged.connect(lambda _: self.update_y())
        self.at_y.setCurrentIndex(e["j"])
        self.at_x.setCurrentIndex(e["i"])
        self.update_x()
        self.update_y()
        bar = toolbar(labelled("Сечение w(x) при", self.at_y), labelled("Сечение w(y) при", self.at_x),
                      trailing=trailing)
        row = QHBoxLayout()
        row.addWidget(self.fig_x)
        row.addWidget(self.fig_y)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(page(bar, padded(row)))

    def update_x(self):
        j = self.at_y.currentData()
        self.curve_x.setData(self.outcome.x, self.outcome.w[:, j] * MM)
        self.fig_x.set_title("section.x.title", y=float(self.outcome.y[j]))

    def update_y(self):
        i = self.at_x.currentData()
        self.curve_y.setData(self.outcome.y, self.outcome.w[i, :] * MM)
        self.fig_y.set_title("section.y.title", x=float(self.outcome.x[i]))


class ValuesTab(QWidget):
    """Values at all nodes of the result grid (i, j, x, y, w, w̄)."""

    HEADERS = ("i", "j", "x, m", "y, m", "w, mm", "w, m", "w̄")

    def __init__(self, outcome, trailing=(), parent=None):
        super().__init__(parent)
        o = outcome
        ny = len(o.y)

        def row(r):
            i, j = divmod(r, ny)
            w = o.w[i, j]
            return (i, j, o.x[i], o.y[j], w * MM, w, o.w_bar[i, j])

        self.model = RowsModel(self.HEADERS, len(o.x) * ny, row, digits=10)
        card = Card(f"Значения во всех узлах сетки результата: {len(o.x)} × {ny}",
                    right=label(o.mark or "KK-VIM", "tag"))
        card.body.addWidget(table_view(self.model))
        box = QVBoxLayout()
        box.addWidget(card)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(page(toolbar(trailing=trailing), padded(box)))


def positive_series(s_values, values):
    pairs = [(s, v) for s, v in zip(s_values, values, strict=True) if v is not None and v > 0 and math.isfinite(v)]
    return [p[0] for p in pairs], [p[1] for p in pairs]


class HistoryTab(QWidget):
    """Half-step history: ε^∞ and ε^E (log scale) with the tolerances, and the full table of rows."""

    def __init__(self, session, outcome, trailing=(), parent=None):
        super().__init__(parent)
        self.figure = Figure(session, outcome, "hist.title")
        self.figure.set_axes("axis.halfstep", "axis.change")
        plot = self.figure.plot
        plot.setLogMode(y=True)
        plot.addLegend(offset=(-10, 10))
        lang = session.figure_language
        t = session.tokens
        s = outcome.history_values("s")
        self.curves = {}
        for key, colour, symbol in (("eps_inf", t["accent"], "o"), ("eps_E", t["status.warning"], "t")):
            xs, ys = positive_series(s, outcome.history_values(key))
            self.curves[key] = plot.plot(xs, ys, pen=pg.mkPen(colour, width=2), symbol=symbol, symbolSize=6,
                                         symbolBrush=colour, name=fig_text(lang, f"legend.{key}"))
        params = dict(outcome.parameters)
        for key, style, at in (("tol_inf", Qt.PenStyle.DashLine, 0.04), ("tol_E", Qt.PenStyle.DotLine, 0.14)):
            value = params.get(key)
            if value:
                line = pg.InfiniteLine(pos=math.log10(value), angle=0,
                                       pen=pg.mkPen(t["text.secondary"], width=1, style=style),
                                       label=fig_text(lang, f"legend.{key}"),
                                       labelOpts={"position": at, "color": t["text.secondary"]})
                plot.addItem(line)
        headers = [c.title + (f", {c.unit}" if c.unit else "") for c in outcome.columns]
        keys = [c.key for c in outcome.columns]
        self.model = RowsModel(headers, len(outcome.history), lambda r: tuple(outcome.history[r].get(k) for k in keys))
        table = Card("История полушагов", right=label(outcome.mark or "KK-VIM", "tag"))
        table.body.addWidget(table_view(self.model))
        split = QSplitter(Qt.Orientation.Vertical)
        split.addWidget(self.figure)
        split.addWidget(table)
        split.setSizes([300, 300])
        box = QVBoxLayout()
        box.addWidget(split)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(page(toolbar(trailing=trailing), padded(box)))


class EnergyTab(QWidget):
    """Discrete energy Π_h by half-steps and the run-level diagnostics of the solver."""

    def __init__(self, session, outcome, trailing=(), parent=None):
        super().__init__(parent)
        self.figure = Figure(session, outcome, "energy.title")
        self.figure.set_axes("axis.halfstep", "axis.energy")
        s = outcome.history_values("s")
        P = outcome.history_values("P")
        pairs = [(a, b) for a, b in zip(s, P, strict=True) if b is not None]
        colour = session.tokens["accent"]
        self.curve = self.figure.plot.plot([p[0] for p in pairs], [p[1] for p in pairs],
                                           pen=pg.mkPen(colour, width=2), symbol="o", symbolSize=6,
                                           symbolBrush=colour)
        diag = Card("Диагностика невязки и остановки", right=label(outcome.mark or "KK-VIM", "tag"))
        last = outcome.history[-1] if outcome.history else {}
        rows = [("Статус решателя", outcome.status), ("Остановка", outcome.stop_text)]
        if outcome.reason:
            rows.append(("Причина (решатель)", outcome.reason))
        if outcome.error:
            rows.append(("Ошибка", outcome.error))
        rows.append(("Невязка η последнего полушага", fmt(last.get("eta"), 6)))
        rows.append(("Энергетический тест последнего полушага", fmt(last.get("energy_status"))))
        rows += [(f"Параметр {k}", fmt(v, 6)) for k, v in outcome.parameters]
        rows += [(f"Диагностика {k}", fmt(v, 6)) for k, v in outcome.diagnostics if k not in ("status", "reason")]
        rows += [(f"Версия {k}", v) for k, v in outcome.versions.items()]
        content = QWidget()
        box = QVBoxLayout(content)
        box.setContentsMargins(0, 0, 8, 0)
        box.addLayout(key_value(rows))
        for note in outcome.notes:
            box.addWidget(Note(note, "testdata" if outcome.mark else "info"))
        box.addStretch(1)
        scroll = QScrollArea()  # long texts keep their wrapped height; the list scrolls instead
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(content)
        diag.body.addWidget(scroll, 1)
        row = QHBoxLayout()
        row.addWidget(self.figure, 3)
        row.addWidget(diag, 2)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(page(toolbar(trailing=trailing), padded(row)))
