"""Tabs of the analysis window for a solver result (GUI_DESIGN.md, 6).

Every figure and table is drawn from the arrays and rows of the ``RunOutcome`` (copies of the solver
result); nothing is recomputed here except unit scaling for display (w in mm = w in m · 1000) and the
search of the node with the largest |w|. The field figure is static (GUI-D13); X–Y plots keep the
pan and zoom of pyqtgraph. Every R4 figure carries PROVISIONAL / NOT_QUALIFIED; texts of figures
follow the session figure language (GUI_DESIGN.md, 7).

User-facing layout (GUI-PLATE-WORKFLOW-UX-01): the status strip and the «Результат» tab state three things separately
and briefly — whether the run completed, the numerical stopping result and the scientific applicability — with the full
solver texts in tooltips and in «Технические подробности»; a research, provisional or diagnostic result always carries
its visible mark. Cross-sections are chosen by node index (slider and spin box) with the actual coordinate shown; the
default is the centre line when it is a node. The half-step history shows the change of the field first and the full
table behind «Подробная таблица».

Inspection (GUI-OWNER-VIEW-02): a click on the colour map selects the nearest EXISTING grid node (no interpolation) and
shows its indices, x, y and the stored w; the mark stays when the colour scale, the quantity or the figure language
change. «Вид сбоку» draws the stored mid-surface deflection along a node row (x at y_j) or a node column (y at x_i)
against the undeformed mid-surface w = 0, with w positive downward (MATH_SPEC.md 8.1). At ×1 both axes are in metres
with equal scale (the physical shape); a larger factor multiplies only the drawn w and is stated on the axis and in the
legend, the numbers shown are always the stored values. The nominal thickness h can be drawn as a labelled schematic
around the undeformed mid-surface; deformed faces, strains or stresses through the thickness are not computed or drawn.
Nothing here starts a calculation.
"""

import math

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QAbstractTableModel, QModelIndex, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QFont, QFontMetrics, QTransform
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTableView,
    QToolButton,
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

    def __init__(self, headers, count, row, digits=6, parent=None, tooltip=None):
        """tooltip(row, column) -> text: optional full text of a cell (read-only access to a narrowed column)."""
        super().__init__(parent)
        self._headers, self._count, self._row, self._digits = list(headers), count, row, digits
        self._tooltip = tooltip

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
        if role == Qt.ItemDataRole.ToolTipRole and self._tooltip is not None:
            return self._tooltip(index.row(), index.column())
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
        if self.outcome.is_error or self.outcome.diagnostic_only:
            marks.append(fig_text(lang, "mark.diagnostic"))
        if self.outcome.mark:
            marks.append(fig_text(lang, "mark.research" if self.outcome.mark.startswith("RESEARCH_ONLY") else "mark.provisional"))
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
    """Static colour field of w (mm) or w̄ on the nodes of the result grid with a colour bar.

    A left click on the map selects the nearest existing node (``nodeSelected(i, j)``); the view itself stays static
    (GUI-D13)."""

    ISOLINES = 10  # contour levels evenly inside (min, max) of the shown quantity
    nodeSelected = Signal(int, int)

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
        self.selected = None  # (i, j) of the selected node
        self.pick_outer = pg.ScatterPlotItem(size=16, pen=pg.mkPen("#000000", width=3), brush=None, symbol="o")
        self.pick_inner = pg.ScatterPlotItem(size=16, pen=pg.mkPen("#ffffff", width=1.5), brush=None, symbol="o")
        for item in (self.pick_outer, self.pick_inner):
            item.setZValue(20)  # above the isolines; visible on every colour map
            self.plot.addItem(item)
        self.plot.scene().sigMouseClicked.connect(self._on_click)
        self.update_image()

    def node_at(self, x, y):
        """(i, j) of the existing node nearest to the point (x, y) in metres, or None outside the drawn plate."""
        if not self.rect.contains(QPointF(x, y)):
            return None
        return int(np.argmin(np.abs(self.outcome.x - x))), int(np.argmin(np.abs(self.outcome.y - y)))

    def _on_click(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            return
        box = self.plot.getViewBox()
        if not box.sceneBoundingRect().contains(event.scenePos()):
            return
        point = box.mapSceneToView(event.scenePos())
        node = self.node_at(point.x(), point.y())
        if node is not None:
            self.select_node(*node)
            self.nodeSelected.emit(*node)

    def select_node(self, i, j):
        """Mark the node (i, j) at its own coordinates (x_i, y_j)."""
        self.selected = (int(i), int(j))
        x, y = [float(self.outcome.x[i])], [float(self.outcome.y[j])]
        self.pick_outer.setData(x, y)
        self.pick_inner.setData(x, y)

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


FIELD_ROLE_TEXT = {
    "accepted": "результат принятого решателя R3 (field.csv)",
    "research": "исследовательское поле, не квалифицированный результат (field_research.csv)",
    "diagnostic": "только диагностика: поле последнего выполненного полушага (field_diagnostic.csv)",
}


def statuses(outcome, run):
    """(kind, text) of execution, stop and scientific status of the result: short texts (the full ones: status_details)."""
    if outcome.is_error:
        execution = ("error", f"Расчёт не завершён: ошибка решателя {outcome.status}; поле только диагностическое")
    elif not outcome.requested_completed:
        execution = ("warning", f"Расчёт не завершён как запрошено ({outcome.status}): поле только диагностическое")
    else:
        execution = ("success", "Расчёт завершён")
    kind = STOP_STATUS[outcome.stop_kind]
    if outcome.mark and kind == "success":
        kind = "warning"  # an R4 stabilization is PROVISIONAL, not a success of the method
    stop = (kind, f"{outcome.stop_text} · s = {outcome.s}, k = {outcome.k}")
    science = (outcome.science_kind if outcome.mark else "neutral",
               f"Научный статус: {outcome.mark}" if outcome.mark else "Научный статус: принятый решатель R3 (ACCEPTED)")
    return execution, stop, science


def status_details(outcome):
    """The full texts behind the three short statuses (tooltips and «Технические подробности»)."""
    decisions = ", ".join(outcome.decisions)
    return (f"Статус решателя: {outcome.status}" + (f". Причина: {outcome.reason}" if outcome.reason else "")
            + (f". Ошибка: {outcome.error}" if outcome.error else ""),
            f"{outcome.stop_text}. s — выполненные полушаги, k = ⌊s/2⌋ — завершённые циклы (D-030). Выполнение критерия "
            "остановки не доказывает точности модели или метода",
            f"{outcome.science_text}" + (f" ({decisions})" if decisions else ""))


def status_labels(outcome, run):
    """StatusLabel widgets of the three statuses with the full texts as tooltips."""
    out = []
    for (kind, text), detail in zip(statuses(outcome, run), status_details(outcome), strict=True):
        w = StatusLabel(kind, text)
        w.setWordWrap(True)
        w.setToolTip(detail)
        out.append(w)
    return out


def expander(title, content, expanded=False):
    """A collapsible section: a toggle button and its content (returns the button and a container)."""
    box = QWidget()
    lay = QVBoxLayout(box)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(6)
    button = QToolButton()
    button.setCheckable(True)
    button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
    button.setText(title)
    lay.addWidget(button)
    lay.addWidget(content)

    def show(on):
        button.setArrowType(Qt.ArrowType.DownArrow if on else Qt.ArrowType.RightArrow)
        content.setVisible(on)

    button.toggled.connect(show)
    button.setChecked(expanded)
    show(expanded)
    return button, box


def centre_value(outcome):
    """The stored w at the centre node (a/2, b/2), or None if the centre is not a node (no interpolation)."""
    from kkvim_gui.backend.benchmarks import centre_node

    c = centre_node(outcome) if outcome.has_field else None
    if c is None:
        return None
    return {**c, "w": float(outcome.w[c["i"], c["j"]]),
            "w_bar": None if outcome.w_bar is None else float(outcome.w_bar[c["i"], c["j"]])}


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
        ("Полушаги", f"s = {outcome.s}" + (f" из запрошенных {outcome.s_requested}" if outcome.s_requested else "")
                     + f", k = {outcome.k} (D-030)"),
        ("Роль поля", FIELD_ROLE_TEXT.get(outcome.field_role, "поле не получено")),
        ("Закрепление, сетка, N", f"{r.supports} · n_x = {r.n_x}, n_y = {r.n_y} · N = {r.N}"),
        ("Решатель", f"{outcome.solver}: {outcome.call}"),
        ("Время расчёта", elapsed_text),
    ]
    return rows


APPLICABILITY = {
    None: "Принятый решатель R3 в утверждённой области. Отдельный расчёт не является научной приёмкой.",
}
SCIENCE_NOTE = {  # a visible warning of every non-accepted route (the full texts are in the details)
    "PROVISIONAL / NOT_QUALIFIED": "Результат не квалифицирован: PROVISIONAL / NOT_QUALIFIED.",
    "RESEARCH_ONLY / PROVISIONAL / NOT_QUALIFIED": ("Исследовательский режим — научная квалификация не завершена "
                                                   "(RESEARCH_ONLY / PROVISIONAL / NOT_QUALIFIED)."),
}


class OverviewTab(QWidget):
    """«Результат»: the plate and load, the deflection, the completion, the applicability and the comparison."""

    def __init__(self, session, outcome, run, model_rows, elapsed_text, comparison=None, parent=None):
        super().__init__(parent)
        self.outcome = outcome
        rows = dict(model_rows)
        r = outcome.request
        plate = Card("Пластина и нагрузка")
        plate.body.addLayout(key_value([(k, rows[k]) for k in ("Геометрия", "Материал", "Нагрузка", "Закрепление")
                                        if k in rows] + [("Метод и сетка", f"КК-ВИМ, N = {r.N} · сетка {r.n_x} × {r.n_y}")]))
        plate.body.addStretch(1)
        deflection = Card("Прогиб", right=label(outcome.mark or "", "tag"))
        self.deflection_rows = self._deflection_rows(outcome)
        deflection.body.addLayout(key_value(self.deflection_rows))
        if outcome.diagnostic_only or outcome.is_error:
            deflection.body.addWidget(Note("Поле только диагностическое (последний выполненный полушаг): это не результат.",
                                           "testdata"))
        deflection.body.addStretch(1)
        completion = Card("Расчёт")
        self.status_widgets = status_labels(outcome, run)
        for w in self.status_widgets[:2]:
            completion.body.addWidget(w)
        completion.body.addWidget(label(f"Время расчёта: {elapsed_text}", "secondary"))
        completion.body.addStretch(1)
        science = Card("Научная применимость")
        science.body.addWidget(self.status_widgets[2])
        science.body.addWidget(label(APPLICABILITY[None] if outcome.mark is None else
                                     SCIENCE_NOTE.get(outcome.mark, outcome.mark), "secondary", wrap=True))
        science.body.addStretch(1)
        self.comparison_card = Card("Сравнение с эталоном")
        self.comparison_body = QVBoxLayout()
        self.comparison_card.body.addLayout(self.comparison_body)
        self.comparison_card.body.addStretch(1)
        self.set_comparison(comparison)
        details = QWidget()
        dl = QVBoxLayout(details)
        dl.setContentsMargins(0, 0, 0, 0)
        dl.addLayout(key_value(input_rows(outcome, model_rows)[len(model_rows):] + key_values(outcome, elapsed_text)
                               + [("Полный статус", " · ".join(status_details(outcome)))]))
        for note in outcome.notes:
            dl.addWidget(Note(note, "testdata" if outcome.mark else "info"))
        if outcome.limitations:
            dl.addWidget(Note(f"Ограничения маршрута ({len(outcome.limitations)}): вкладка «Диагностика» и report.html",
                              "testdata" if outcome.mark else "info"))
        self.details_button, details_box = expander("Технические подробности", details)
        grid = QGridLayout()
        grid.addWidget(plate, 0, 0)
        grid.addWidget(deflection, 0, 1)
        grid.addWidget(completion, 0, 2)
        grid.addWidget(science, 1, 0)
        grid.addWidget(self.comparison_card, 1, 1, 1, 2)
        grid.addWidget(details_box, 2, 0, 1, 3)
        self.field = FieldFigure(session, outcome, compact=True) if outcome.has_field else None
        if self.field is not None:
            self.field.setMinimumHeight(260)
            grid.addWidget(self.field, 3, 0, 1, 3)
            grid.setRowStretch(3, 1)
        for c in range(3):
            grid.setColumnStretch(c, 1)
        content = padded(grid)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(content)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(scroll)

    @staticmethod
    def _deflection_rows(outcome):
        c = centre_value(outcome)
        e = outcome.extremum()
        rows = []
        if c is not None:
            rows.append(("w в центре (a/2, b/2)", f"{fmt(c['w'] * MM, 6)} mm"))
        elif outcome.has_field:
            rows.append(("w в центре", "центр не является узлом сетки — значение не интерполируется"))
        if e is not None:
            rows += [("Максимальный |w| (по узлам)", f"{fmt(e['w'] * MM, 6)} mm"),
                     ("Узел максимума", f"x = {fmt(e['x'], 6)} m, y = {fmt(e['y'], 6)} m")]
        if c is not None and c["w_bar"] is not None:
            rows.append(("w̄ = wD/(qa⁴) в центре", fmt(c["w_bar"], 6)))
        return rows or [("Поле прогиба", "не получено")]

    def set_comparison(self, comparison):
        """Show the state of the comparison with the independent reference (it may arrive after the window opened)."""
        while self.comparison_body.count():
            item = self.comparison_body.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        self.comparison_rows = []
        if comparison is None:
            self.comparison_body.addWidget(label("Сравнение недоступно.", "secondary", wrap=True))
            return
        from kkvim_gui.backend import benchmarks

        if comparison.has_numbers:
            row = comparison.rows[0]
            ref = row.reference
            rows = [("КК-ВИМ — w в центре", f"{fmt(comparison.w_centre_m * MM, 8)} mm")]
            if ref.w_centre_m is not None:
                rows.append((f"{ref.title} — w в центре", f"{fmt(ref.w_centre_m * MM, 8)} mm"))
            if row.delta_w_m is not None:
                rows.append(("Разность Δw = w_КК-ВИМ − w_эталона", f"{fmt(row.delta_w_m * MM, 6)} mm"))
            rows += [("Относительная разность Δβ/β", f"{fmt(row.rel_pct, 6)} %"),
                     ("β КК-ВИМ / β эталона", f"{fmt(row.beta_kkvim, 8)} / {fmt(ref.beta, 8)}")]
            if len(comparison.rows) > 1:
                rows.append(("Другие эталоны", f"ещё {len(comparison.rows) - 1}: вкладка «Сравнение с эталоном»"))
            self.comparison_rows = rows
            self.comparison_body.addLayout(key_value(rows))
            if self.outcome.mark and comparison.navier is not None and comparison.live:
                self.comparison_body.addWidget(Note(f"Для этого маршрута ({self.outcome.mark}) решение Навье — справочные "
                                                    "данные: статус расчёта не меняется.", "testdata"))
            if ref.kind == "D2_live":
                self.comparison_body.addWidget(Note("МКР D2 рассчитан для этих входных данных в отдельном процессе: та же "
                                                    "дискретная задача (D-049.1); разность — отличие приближения КК-ВИМ от "
                                                    "полного дискретного решения, не ошибка модели пластины. Статус "
                                                    "расчёта не меняется.", "info"))
            self.comparison_body.addWidget(label("Знаковые разности без порога согласия; сравнение не меняет научный "
                                                 "статус и не является оценкой точности метода.", "secondary", wrap=True))
        else:
            kind = "warning" if comparison.state != benchmarks.PENDING else "neutral"
            w = StatusLabel(kind, benchmarks.STATE_TITLE[comparison.state])
            w.setWordWrap(True)
            self.comparison_body.addWidget(w)
            if comparison.reason:
                self.comparison_body.addWidget(label(comparison.reason, "secondary", wrap=True))


class SideFigure(Figure):
    """Side view of the stored mid-surface deflection along a node row or column (GUI-OWNER-VIEW-02).

    Profile: the nodes of the result grid and the stored w (no interpolation, nothing recomputed). y of the plot = factor
    · w in metres, the axis inverted so that positive w (downward, MATH_SPEC.md 8.1) is drawn below w = 0. At factor 1
    the aspect ratio is locked to 1: the physical shape."""

    FACTORS = (1.0, 10.0, 100.0, 1000.0)

    def __init__(self, session, outcome, i0, j0, parent=None):
        super().__init__(session, outcome, "side.title.x", {"y": float(outcome.y[j0])}, interactive=True, parent=parent)
        self.direction, self.i, self.j, self.factor, self.show_h = "x", int(i0), int(j0), 1.0, False
        t = session.tokens
        self.plot.invertY(True)  # w positive downward
        self.legend = self.plot.addLegend(offset=(10, -10))
        self.zero = self.plot.plot(pen=pg.mkPen(t["plot.axis"], width=1.5, style=Qt.PenStyle.DashLine))
        self.upper = self.plot.plot(pen=pg.mkPen(t["status.warning"], width=1, style=Qt.PenStyle.DotLine))
        self.lower = self.plot.plot(pen=pg.mkPen(t["status.warning"], width=1, style=Qt.PenStyle.DotLine))
        self.profile = self.plot.plot(pen=pg.mkPen(t["accent"], width=2), symbol="o", symbolSize=4, symbolBrush=t["accent"])
        self.max_point = pg.ScatterPlotItem(size=12, pen=pg.mkPen(t["status.error"], width=2), brush=None, symbol="o")
        self.plot.addItem(self.max_point)
        self.update_view()

    def section(self):
        """(name, coordinates, w in m) of the current node row or column, exactly the stored arrays."""
        if self.direction == "x":
            return "x", self.outcome.x, self.outcome.w[:, self.j]
        return "y", self.outcome.y, self.outcome.w[self.i, :]

    def maximum(self):
        """(index, coordinate, w in m) of the node of the largest |w| in the section."""
        _, coords, w = self.section()
        k = int(np.argmax(np.abs(w)))
        return k, float(coords[k]), float(w[k])

    def set_section(self, direction=None, i=None, j=None):
        self.direction = direction or self.direction
        self.i = self.i if i is None else int(i)
        self.j = self.j if j is None else int(j)
        self.update_view()

    def set_factor(self, factor):
        self.factor = float(factor)
        self.update_view()

    def set_thickness(self, on):
        self.show_h = bool(on)
        self.update_view()

    def update_view(self):
        name, coords, w = self.section()
        drawn = self.factor * w
        lo, hi = float(coords[0]), float(coords[-1])
        self.profile.setData(coords, drawn)
        self.zero.setData([lo, hi], [0.0, 0.0])
        half = self.outcome.request.t / 2
        self.upper.setData([lo, hi], [-half, -half] if self.show_h else [])
        self.lower.setData([lo, hi], [half, half] if self.show_h else [])
        self.upper.setVisible(self.show_h)
        self.lower.setVisible(self.show_h)
        k, c, wk = self.maximum()
        self.max_point.setData([c], [self.factor * wk])
        self.plot.setAspectLocked(self.factor == 1.0, ratio=1.0)  # ×1: equal scale of both axes, the physical shape
        self.retranslate()
        self.plot.autoRange()

    def retranslate(self, *_):
        lang = self.session.figure_language
        f = f"{self.factor:g}"
        if self.direction == "x":
            self._title_key, self._title_values = "side.title.x", {"y": float(self.outcome.y[self.j])}
        else:
            self._title_key, self._title_values = "side.title.y", {"x": float(self.outcome.x[self.i])}
        self._axes = {"bottom": "axis.side.x" if self.direction == "x" else "axis.side.y"}
        super().retranslate()
        self.plot.setLabel("left", fig_text(lang, "axis.side.w", f=f))
        if getattr(self, "legend", None) is None:
            return
        self.legend.clear()
        self.legend.addItem(self.zero, fig_text(lang, "legend.side.undeformed"))
        self.legend.addItem(self.profile, fig_text(lang, "legend.side.profile", f=f))
        self.legend.addItem(self.max_point, fig_text(lang, "legend.side.max"))
        if self.show_h:
            self.legend.addItem(self.upper, fig_text(lang, "legend.side.thickness"))


def fit_bold(control):
    """Segment buttons are drawn in bold when selected: never narrower than the bold text (no clipped labels)."""
    for button in control.buttons.values():
        bold = QFont(button.font())
        bold.setBold(True)
        chrome = max(16, button.sizeHint().width() - button.fontMetrics().horizontalAdvance(button.text()))
        button.setMinimumWidth(QFontMetrics(bold).horizontalAdvance(button.text()) + chrome)


def node_text(outcome, i, j):
    """Text of the stored values at node (i, j); diagnostic and research fields keep their mark."""
    x, y, w = float(outcome.x[i]), float(outcome.y[j]), float(outcome.w[i, j])
    head = ""
    if outcome.is_error or outcome.diagnostic_only:
        head = "ДИАГНОСТИКА (поле последнего выполненного полушага, не результат): "
    elif outcome.mark:
        head = f"{outcome.mark}: "
    wbar = "" if outcome.w_bar is None else f"; w̄ = {fmt(float(outcome.w_bar[i, j]), 10)}"
    return (f"{head}узел i = {i}, j = {j}: x = {x!r} m, y = {y!r} m; w = {w!r} m (= {fmt(w * MM, 7)} mm){wbar} "
            "— значение решателя в узле, без интерполяции")


class FieldsTab(QWidget):
    """Colour field with isolines, the axonometric view of the deformed surface with a labelled factor, or the side view
    of the mid-surface along a node line; a click on the map selects a node."""

    def __init__(self, session, outcome, model_rows, trailing=(), parent=None):
        super().__init__(parent)
        from kkvim_gui.analysis.surface import PRESETS, SurfaceFigure

        self.outcome = outcome
        self.figure = FieldFigure(session, outcome, isolines=True)
        self.surface = SurfaceFigure(session, outcome)
        e = outcome.extremum()
        c = centre_value(outcome)
        i0, j0 = (c["i"], c["j"]) if c is not None else (e["i"], e["j"])  # the centre lines when they are nodes
        self.side = SideFigure(session, outcome, i0, j0)
        self.view = SegmentedControl({"map": "Карта и изолинии", "surface": "3D-вид (цвет)", "side": "Вид сбоку"}, "map")
        self.quantity = SegmentedControl({"w": "Прогиб w, mm", "w_bar": "Безразмерный w̄"}, "w")
        self.quantity.changed.connect(self.figure.set_quantity)
        self.cmap = QComboBox()
        self.cmap.addItems(["viridis", "cividis"])
        self.cmap.currentTextChanged.connect(self.figure.set_colormap)
        self.cmap.currentTextChanged.connect(self.surface.set_colormap)
        self.mesh = QCheckBox("Сетка")
        self.mesh.setChecked(True)
        self.mesh.setToolTip("Тонкие линии узлов сетки поверх цветной поверхности")
        self.mesh.toggled.connect(self.surface.set_mesh)
        self.isolines = QCheckBox("Изолинии")
        self.isolines.setChecked(True)
        self.isolines.toggled.connect(self.figure.set_isolines)
        self.factor = QComboBox()
        self.factor.addItem(f"Авто: ×{fmt(self.surface.auto, 3)}", self.surface.auto)
        for value in PRESETS:
            self.factor.addItem("×1 (истинный)" if value == 1.0 else f"×{fmt(value, 3)}", value)
        self.factor.setToolTip("Масштаб прогиба только на изображении 3D-вида; числовые значения не изменяются")
        self.factor.currentIndexChanged.connect(lambda i: self.surface.set_factor(self.factor.itemData(i)))
        self.map_controls = (self.quantity, self.isolines)
        self.surface_controls = (labelled("Масштаб прогиба", self.factor), self.mesh)
        # side view: direction, the node line (the selectors of «Сечения»), the drawing factor and the thickness schematic
        self.side_direction = SegmentedControl({"x": "Вдоль x", "y": "Вдоль y"}, "x")
        self.side_direction.setToolTip("Вдоль x — строка узлов при выбранном y_j; вдоль y — столбец узлов при выбранном x_i")
        self.side_at_y = NodeSelector("y", outcome.y, j0)
        self.side_at_x = NodeSelector("x", outcome.x, i0)
        self.side_factor = QComboBox()
        for value in SideFigure.FACTORS:
            self.side_factor.addItem("×1 — физический масштаб" if value == 1.0 else f"×{value:g} — только изображение", value)
        self.side_factor.setToolTip("×1: обе оси в метрах в одном масштабе (истинная форма). Больший множитель увеличивает "
                                    "только изображение прогиба; числа — значения решателя")
        self.side_thickness = QCheckBox("Толщина h (схема)")
        self.side_thickness.setToolTip("Номинальная толщина h вокруг исходной срединной поверхности — схема; деформированные "
                                       "лицевые поверхности, деформации и напряжения по толщине не рассчитываются")
        self.side_direction.changed.connect(self._side_direction)
        self.side_at_y.spin.valueChanged.connect(lambda j: self.side.set_section(j=j))
        self.side_at_x.spin.valueChanged.connect(lambda i: self.side.set_section(i=i))
        self.side_factor.currentIndexChanged.connect(lambda k: self.side.set_factor(self.side_factor.itemData(k)))
        self.side_thickness.toggled.connect(self.side.set_thickness)
        for signal in (self.side_at_y.spin.valueChanged, self.side_at_x.spin.valueChanged,
                       self.side_factor.currentIndexChanged, self.side_thickness.toggled):
            signal.connect(self._show_side_info)  # after the figure: the text follows the drawn section
        self.side_y_box = labelled("при", self.side_at_y)
        self.side_x_box = labelled("при", self.side_at_x)
        self.side_bar = QWidget()  # two rows: the node line; the drawing factor and the thickness schematic
        side_lay = QVBoxLayout(self.side_bar)
        side_lay.setContentsMargins(0, 0, 0, 0)
        side_lay.setSpacing(0)
        side_lay.addWidget(toolbar(self.side_direction, self.side_y_box, self.side_x_box))
        side_lay.addWidget(toolbar(labelled("Масштаб", self.side_factor), self.side_thickness))
        self.side_controls = (self.side_bar,)
        bar = toolbar(self.view, labelled("Шкала", self.cmap), *self.map_controls, *self.surface_controls,
                      trailing=trailing)
        self.stack = QStackedWidget()
        self.stack.addWidget(self.figure)
        self.stack.addWidget(self.surface)
        self.stack.addWidget(self.side)
        self.node_info = label("Щёлкните карту: будет выбран ближайший узел сетки (без интерполяции).", "secondary", wrap=True)
        self.node_info.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.side_info = label("", "secondary", wrap=True)
        self.side_info.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.figure.nodeSelected.connect(self.select_node)
        self.view.changed.connect(self.set_view)
        self._side_direction("x")
        self.set_view("map")
        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        box = QVBoxLayout()
        box.setContentsMargins(16, 14, 16, 16)
        box.addWidget(self.side_bar)
        box.addWidget(self.stack, 1)
        box.addWidget(self.node_info)
        box.addWidget(self.side_info)
        body.addLayout(box, 1)
        inspector = QScrollArea()  # a low window scrolls the panel instead of overlapping its lines
        inspector.setWidgetResizable(True)
        inspector.setFrameShape(QFrame.Shape.NoFrame)
        inspector.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        panel = self._inspector(outcome, model_rows)
        inspector.setWidget(panel)
        inspector.setFixedWidth(panel.width() + 14)
        body.addWidget(inspector)
        content = QWidget()
        content.setLayout(body)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(page(bar, content))

    def showEvent(self, event):
        super().showEvent(event)
        for control in (self.view, self.quantity, self.side_direction):  # after the style sheet: real padding known
            fit_bold(control)

    def set_view(self, view):
        surface, side = view == "surface", view == "side"
        self.stack.setCurrentWidget(self.surface if surface else self.side if side else self.figure)
        for w in self.map_controls:
            w.setVisible(view == "map")
        for w in self.surface_controls:
            w.setVisible(surface)
        for w in self.side_controls:
            w.setVisible(side)
        self.cmap.parentWidget().setVisible(not side)
        self.node_info.setVisible(view == "map")
        self.side_info.setVisible(side)
        if self.view.current() != view:
            self.view.set_current(view)

    def select_node(self, i, j):
        """The node (i, j) chosen on the map: its stored values; the side view moves to its row and column."""
        self.figure.select_node(i, j)
        self.node_info.setText(node_text(self.outcome, i, j))
        self.side_at_y.set_index(j)
        self.side_at_x.set_index(i)

    def _side_direction(self, direction):
        self.side.set_section(direction=direction)
        self.side_y_box.setVisible(direction == "x")
        self.side_x_box.setVisible(direction == "y")
        self._show_side_info()

    def _show_side_info(self, *_):
        o, side = self.outcome, self.side
        name, coords, _ = side.section()
        k, c, wk = side.maximum()
        line = (f"вдоль x при y = {fmt(float(o.y[side.j]), 10)} m (j = {side.j})" if name == "x" else
                f"вдоль y при x = {fmt(float(o.x[side.i]), 10)} m (i = {side.i})")
        scale = ("×1: обе оси в метрах в одном масштабе — истинная форма" if side.factor == 1.0 else
                 f"×{side.factor:g}: увеличен только рисунок прогиба; числа — значения решателя")
        status = ("ДИАГНОСТИКА: поле последнего выполненного полушага, не результат. " if o.is_error or o.diagnostic_only
                  else f"{o.mark}. " if o.mark else "")
        self.side_info.setText(f"{status}Срединная поверхность {line}; {len(coords)} узлов; max |w| в сечении: "
                               f"w = {fmt(wk * MM, 10)} mm в узле {'i' if name == 'x' else 'j'} = {k} "
                               f"({name} = {fmt(c, 10)} m). w > 0 — вниз (MATH_SPEC 8.1); {scale}. Толщина h — только схема.")

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


class NodeSelector(QWidget):
    """A node line of the result grid: slider and spin box of the index with the actual coordinate (no interpolation)."""

    def __init__(self, name, nodes, index, parent=None):
        super().__init__(parent)
        self.name, self.nodes = name, nodes
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, len(nodes) - 1)
        self.slider.setMinimumWidth(120)
        self.spin = QSpinBox()
        self.spin.setRange(0, len(nodes) - 1)
        self.spin.setPrefix(f"{'j' if name == 'y' else 'i'} = ")
        self.text = label("", "secondary")
        self.text.setMinimumWidth(110)
        lay.addWidget(self.slider, 1)
        lay.addWidget(self.spin)
        lay.addWidget(self.text)
        self.slider.valueChanged.connect(self.spin.setValue)
        self.spin.valueChanged.connect(self.slider.setValue)
        self.spin.valueChanged.connect(self._show)
        self.set_index(index)

    def _show(self, index):
        self.text.setText(f"{self.name} = {fmt(float(self.nodes[index]), 6)} m")

    def set_index(self, index):
        self.spin.setValue(index)
        self._show(index)

    def index(self):
        return self.spin.value()


class SectionsTab(QWidget):
    """w(x) at a node row y_j and w(y) at a node column x_i, both from the stored field."""

    TOOLTIP = "Профили прогиба вдоль выбранных линий узлов сетки: w(x) при y = const и w(y) при x = const"

    def __init__(self, session, outcome, trailing=(), parent=None):
        super().__init__(parent)
        self.outcome = outcome
        e = outcome.extremum()
        c = centre_value(outcome)
        i0, j0 = (c["i"], c["j"]) if c is not None else (e["i"], e["j"])  # the centre line when it is a node
        self.at_y = NodeSelector("y", outcome.y, j0)
        self.at_x = NodeSelector("x", outcome.x, i0)
        self.fig_x = Figure(session, outcome, "section.x.title", {"y": float(outcome.y[j0])})
        self.fig_y = Figure(session, outcome, "section.y.title", {"x": float(outcome.x[i0])})
        self.fig_x.set_axes("axis.x", "axis.w")
        self.fig_y.set_axes("axis.y", "axis.w")
        self.curve_x = self.fig_x.plot.plot(pen=pg.mkPen(session.tokens["accent"], width=2), symbol="o", symbolSize=4,
                                            symbolBrush=session.tokens["accent"])
        self.curve_y = self.fig_y.plot.plot(pen=pg.mkPen(session.tokens["accent"], width=2), symbol="o", symbolSize=4,
                                            symbolBrush=session.tokens["accent"])
        self.at_y.spin.valueChanged.connect(lambda _: self.update_x())
        self.at_x.spin.valueChanged.connect(lambda _: self.update_y())
        self.update_x()
        self.update_y()
        bar = toolbar(labelled("w(x) при", self.at_y), labelled("w(y) при", self.at_x), trailing=trailing)
        bar.setToolTip(self.TOOLTIP)
        row = QHBoxLayout()
        row.addWidget(self.fig_x)
        row.addWidget(self.fig_y)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(page(bar, padded(row)))

    def update_x(self):
        j = self.at_y.index()
        self.curve_x.setData(self.outcome.x, self.outcome.w[:, j] * MM)
        self.fig_x.set_title("section.x.title", y=float(self.outcome.y[j]))

    def update_y(self):
        i = self.at_x.index()
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

    TITLE = "Как изменялся результат при итерациях"

    @staticmethod
    def _intro(outcome):
        order = "Порядок полушагов: Y¹, X², Y³, X⁴, … — на каждом полушаге одна функция задана, другая ищется."
        counters = (f"s — число выполненных полушагов (здесь s = {outcome.s}), k = ⌊s/2⌋ — число завершённых циклов "
                    f"(k = {outcome.k}).")
        if outcome.route == "ssss_r3":
            rule = ("График: ε^∞ и ε^E — относительное изменение поля между соседними полушагами (по максимуму и в "
                    "энергетической норме). Остановка CONVERGED — при s ≥ s_min, когда на двух полушагах подряд (s − 1 "
                    "и s) одновременно ε^∞ ≤ tol_∞ и ε^E ≤ tol_E (NUMERICAL_SPEC.md, разд. 11.3); это правило остановки, "
                    "а не доказательство точности.")
        else:
            rule = (f"Как закончился расчёт: {outcome.stop_text}. Выполнение заданного числа полушагов — не сходимость. "
                    "График показывает изменение поля между полушагами, если оно вычислено.")
        return " ".join((counters, order, rule))

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
        intro = Card(self.TITLE, right=label(outcome.mark or "KK-VIM", "tag"))
        intro.body.addWidget(label(self._intro(outcome), "secondary", wrap=True))
        self.table = table_view(self.model)
        self.table_button, table_box = expander(f"Подробная таблица: {len(outcome.history)} полушагов, "
                                                f"{len(outcome.columns)} величин", self.table)
        intro.body.addWidget(table_box)
        split = QSplitter(Qt.Orientation.Vertical)
        split.addWidget(self.figure)
        split.addWidget(intro)
        split.setSizes([360, 220])
        box = QVBoxLayout()
        box.addWidget(split)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(page(toolbar(trailing=trailing), padded(box)))


class EnergyTab(QWidget):
    """Discrete energy Π_h by half-steps and the run-level diagnostics of the solver."""

    def __init__(self, session, outcome, trailing=(), parent=None):
        super().__init__(parent)
        energy = next((c for c in outcome.columns if c.key in ("P", "P_hat")), None)  # R3/R4: P; CCCC: P_hat
        self.figure = Figure(session, outcome, "energy.title")
        self.figure.set_axes("axis.halfstep", "axis.energy_cell" if energy is not None and energy.unit == "q²a⁶/D"
                             else "axis.energy")
        s = outcome.history_values("s")
        P = outcome.history_values(energy.key if energy is not None else "P")
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
        if outcome.limitations:
            box.addWidget(Note("Ограничения маршрута:\n" + "\n".join(f"• {t}" for t in outcome.limitations),
                               "testdata" if outcome.mark else "info"))
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
