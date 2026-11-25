"""«Циклы VIM-1–3»: the completed full cycles k = 1, 2, 3 of the analysed run (GUI-VIM-CYCLES-123-01).

Drawn from ``backend.vim_cycles.CycleTable`` only: the graph and the table use the same records. X is the completed full
cycle k (s = 2k), Y the centre deflection in mm; the independent reference of the run (Navier for SSSS, D2 for CCCC) is a
horizontal line. Every VIM state and the reference can be shown or hidden. The partial state s = 1 (k = 0) is a separate
table row, never labelled VIM-1 and not drawn. States the run did not complete, or states of a run that did not end as
requested, are in the table with the reason and are not drawn.

``display`` (GUI-OWNER-COMPARISON-SELECT-01): the curves chosen in «Результаты и сравнение» before the run set the
initial visibility only; what was actually computed and saved, and why the rest was not, is stated above the table.
"""

import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QCheckBox, QFrame, QHBoxLayout, QSplitter, QVBoxLayout, QWidget

from kkvim_gui.analysis.results import MM, RowsModel, fmt, padded, page, table_view, toolbar
from kkvim_gui.backend import vim_cycles
from kkvim_gui.figures.text import fig_text
from kkvim_gui.widgets.common import Card, FigureLanguageSwitch, Note, label, set_help_context

TAB_TITLE = "Циклы VIM-1–3"
TAB_TOOLTIP = ("Прогиб в центре после завершённых полных циклов k = 1, 2, 3 этого расчёта (s = 2, 4, 6) рядом с "
               "независимым эталоном; VIM-k — обозначения интерфейса, не отдельные методы и не N")
HEADERS = ("Состояние", "k", "s", "Полушаг", "w в центре КК-ВИМ, mm", "w в центре КК-ВИМ, m", "Нормированное значение",
           "Эталон, mm", "Δw = w_КК-ВИМ − w_эталона, mm", "Δw / w_эталона, %", "На графике", "Примечание")
SYMBOLS = {1: "o", 2: "s", 3: "d"}
COLOURS = {1: "accent", 2: "status.success", 3: "status.error"}
REF_KEYS = {vim_cycles.NAVIER: "Навье", vim_cycles.D2: "МКР D2"}


def table_row(c, table):
    return (c.label, c.k, c.s, f"{c.direction}{vim_cycles._sup(c.s)}" if c.direction else "—",
            None if c.w_m is None else c.w_m * MM, c.w_m, c.w_norm,
            None if c.ref_w_m is None else c.ref_w_m * MM, None if c.delta_w_m is None else c.delta_w_m * MM, c.delta_pct,
            c.on_graph, c.reason or ("неполный цикл: только Y¹; не VIM-1" if c.k == 0 else ""))


class CycleFigure(QFrame):
    """X–Y graph: w at the centre after the completed full cycles k (VIM-1 … VIM-3) and the reference line."""

    def __init__(self, session, table, parent=None):
        super().__init__(parent)
        self.setProperty("figure", True)
        self.session, self.table = session, table
        self.visible = {k: True for k in range(1, vim_cycles.K_MAX + 1)}
        self.visible["ref"] = True
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 10, 14, 12)
        top = QHBoxLayout()
        self.title = label("", "figureTitle")
        top.addWidget(self.title)
        top.addStretch(1)
        self.mark = label(table.mark or "", "figureMark")
        self.mark.setVisible(bool(table.mark))
        top.addWidget(self.mark)
        lay.addLayout(top)
        self.plot = pg.PlotWidget()
        for side in ("left", "bottom"):
            self.plot.getAxis(side).enableAutoSIPrefix(False)
        # the cycles differ in the 5th-7th digit: tick labels need ~9 characters, which pyqtgraph drops when they do
        # not fit its default axis width (seen in the saved PNG of a CCCC N = 2 run)
        self.plot.getAxis("left").setWidth(110)
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        self.plot.getAxis("bottom").setTicks([[(k, f"{vim_cycles.label(k)} (s = {2 * k})")
                                               for k in range(1, vim_cycles.K_MAX + 1)]])
        self.legend = self.plot.addLegend(offset=(10, -10))  # bottom left, away from the reference line on top
        self.plot.getViewBox().setDefaultPadding(0.4)  # room for the legend: the points stay in the middle band
        lay.addWidget(self.plot, 1)
        t = session.tokens
        self.connector = self.plot.plot(pen=pg.mkPen(t["plot.axis"], width=1, style=Qt.PenStyle.DotLine))
        self.points = {k: self.plot.plot(pen=None, symbol=SYMBOLS[k], symbolSize=11, symbolBrush=t[COLOURS[k]])
                       for k in range(1, vim_cycles.K_MAX + 1)}
        self.reference = self.plot.plot(pen=pg.mkPen(t["status.warning"], width=2, style=Qt.PenStyle.DashLine))
        self.restyle()
        session.themeChanged.connect(self.restyle)
        session.figureLanguageChanged.connect(self.redraw)
        self.tools = None
        self.redraw()
        self._attach_tools()

    COLUMNS = (("k", ""), ("s", ""), ("w_centre", "m"))

    def _attach_tools(self):
        """Chart tools (16, 20): the VIM-k points and the reference line of the same records as the table."""
        from kkvim_gui.analysis.plot_tools import ChartInfo, ChartTools, Curve, default_style
        from kkvim_gui.figures import publication as pub

        table = self.table

        def state_rows(k):
            c = table.cycles[k - 1]
            return [(c.k, c.s, c.w_m)] if c.on_graph else []

        def ref_rows():
            return [(None, None, table.ref_w_m)] if table.ref_w_m is not None and table.drawn else []

        ref_name = "Навье" if table.reference == vim_cycles.NAVIER else "МКР D2"
        ref_key = "legend.cycle.navier" if table.reference == vim_cycles.NAVIER else "legend.cycle.d2"
        curves = [Curve(k, self.points[k], vim_cycles.label(k),
                        lambda lang, k=k: fig_text(lang, "legend.cycle", label=vim_cycles.label(k), k=k, s=2 * k),
                        lambda k=k: state_rows(k), replace_width(default_style(k - 1, "solid", SYMBOLS[k])))
                  for k in range(1, vim_cycles.K_MAX + 1)]
        curves.append(Curve("ref", self.reference, ref_name, lambda lang: fig_text(lang, ref_key), ref_rows,
                            default_style(0, marker=None, reference=True)))
        curves.append(Curve("connector", self.connector, "", lambda lang: "", None,
                            pub.CurveStyle("#666666", 0.6, "dot", None), listed=False))
        info = ChartInfo(
            "vim_cycles", lambda lang: (fig_text(lang, "cycles.title"), fig_text(lang, "axis.cycle_k"),
                                        fig_text(lang, "axis.w_centre")),
            lambda: (("k", ""), ("w", "mm")), self.COLUMNS,
            marks=lambda lang: (table.mark,) if table.mark else (), test_data=False,
            identity=f"{table.solver} · N = {table.N} · q = {table.q!r} Pa",
            meta={"solver": table.solver, "solver_status": table.solver_status},
            x_ticks=lambda lang: tuple((k, f"{vim_cycles.label(k)} (s = {2 * k})") for k in range(1, vim_cycles.K_MAX + 1)),
            ranges=lambda: ((0.5, vim_cycles.K_MAX + 0.5), None))
        self.tools = ChartTools(self.session, self, self.plot, self.title, info, lambda: curves,
                                shown_hook=self.set_visible, is_shown=lambda key: self.visible.get(key, False),
                                home=self.go_home)

    def go_home(self):
        self.plot.setXRange(0.5, vim_cycles.K_MAX + 0.5)
        self.plot.enableAutoRange(axis="y")

    def grab(self, *args):
        """A whole-figure grab is a saved figure (the calculation package): no chart tools, no interaction (16.7)."""
        if args or self.tools is None:
            return super().grab(*args)
        with self.tools.interaction.hidden(self.tools.toolbar, *([self.tools.readout] if self.tools.readout else [])):
            return super().grab()

    def restyle(self, *_):
        t = self.session.tokens
        self.plot.setBackground(t["plot.bg"])
        for side in ("left", "bottom"):
            axis = self.plot.getAxis(side)
            axis.setPen(t["plot.axis"])
            axis.setTextPen(t["plot.axis"])

    def set_visible(self, key, shown):
        self.visible[key] = bool(shown)
        self.redraw()
        if self.tools is not None:
            self.tools.interaction.curves_changed()

    def drawn(self):
        """{k: (k, w in mm)} of the VIM states on the graph (the same records as the table) and the reference in mm."""
        pts = {c.k: (c.k, c.w_m * MM) for c in self.table.drawn if self.visible[c.k]}
        ref = self.table.ref_w_m * MM if self.table.ref_w_m is not None and self.visible["ref"] and self.table.drawn else None
        return pts, ref

    def redraw(self, *_):
        lang = self.session.figure_language
        self.title.setText(fig_text(lang, "cycles.title"))
        self.plot.setLabel("bottom", fig_text(lang, "axis.cycle_k"))
        self.plot.setLabel("left", fig_text(lang, "axis.w_centre"))
        pts, ref = self.drawn()
        self.legend.clear()
        for k, item in self.points.items():
            item.setData([pts[k][0]] if k in pts else [], [pts[k][1]] if k in pts else [])
            item.setVisible(k in pts)
            if k in pts:
                self.legend.addItem(item, fig_text(lang, "legend.cycle", label=vim_cycles.label(k), k=k, s=2 * k))
        ordered = [pts[k] for k in sorted(pts)]
        self.connector.setData([p[0] for p in ordered], [p[1] for p in ordered])
        self.reference.setVisible(ref is not None)
        if ref is not None:
            self.reference.setData([0.75, vim_cycles.K_MAX + 0.25], [ref, ref])
            key = "legend.cycle.navier" if self.table.reference == vim_cycles.NAVIER else "legend.cycle.d2"
            self.legend.addItem(self.reference, fig_text(lang, key))
        self.plot.setXRange(0.5, vim_cycles.K_MAX + 0.5)
        self.plot.enableAutoRange(axis="y")


def replace_width(style):
    """A VIM-k state is a single point: a marker without a line in the file too."""
    from dataclasses import replace

    return replace(style, width=0.0, marker_size=4.5)


def availability_text(table):
    """What this run actually computed and saved for VIM-1 … VIM-3 and its reference, and why the rest is missing."""
    done = [f"{c.label} (s = {c.s})" for c in table.cycles if c.on_graph]
    missing = [f"{c.label} — {c.reason}" for c in table.cycles if not c.on_graph]
    parts = ["Рассчитано и сохранено в истории этого расчёта: " + (", ".join(done) if done else "ни одного полного цикла")]
    if missing:
        parts.append("Нет на графике: " + "; ".join(missing))
    name = REF_KEYS.get(table.reference, "эталон")
    parts.append(f"{name}: рассчитан" if table.ref_w_m is not None else f"{name}: нет — {table.reference_reason}")
    return ". ".join(parts) + "."


class CyclesTab(QWidget):
    def __init__(self, session, table, trailing=(), display=None, parent=None):
        super().__init__(parent)
        set_help_context(self, "iterations.overview")
        self.session, self.table = session, table
        self.display = set(display) if display is not None else {1, 2, 3, "navier", "d2"}
        self.ref_key = "navier" if table.reference == vim_cycles.NAVIER else "d2"
        self.figure = CycleFigure(session, table)
        self.toggles = {}
        for k in range(1, vim_cycles.K_MAX + 1):
            box = QCheckBox(vim_cycles.label(k))
            box.setChecked(k in self.display)  # the choice made before the run; changed here without a calculation
            self.figure.visible[k] = k in self.display
            box.setEnabled(table.cycles[k - 1].on_graph)
            box.setToolTip(table.cycles[k - 1].reason or vim_cycles.cycle_text(k))
            box.toggled.connect(lambda shown, key=k: self._toggle(key, key, shown))
            self.toggles[k] = box
        # «Линии ▾» of the chart tools and these switches are one state: the menu sets the switch, which redraws
        ref_name = REF_KEYS.get(table.reference, "эталон")
        self.toggles["ref"] = QCheckBox(ref_name)
        self.toggles["ref"].setChecked(self.ref_key in self.display)
        self.figure.visible["ref"] = self.ref_key in self.display
        self.toggles["ref"].setEnabled(table.ref_w_m is not None)
        self.toggles["ref"].setToolTip(table.reference_reason or vim_cycles.REFERENCE_TITLE.get(table.reference, ""))
        self.toggles["ref"].toggled.connect(lambda shown: self._toggle("ref", self.ref_key, shown))
        self.figure.tools._shown_hook = lambda key, on: self.toggles[key].setChecked(on)
        self.figure.redraw()
        bar = toolbar(label("Показать:", "fieldLabel"), *self.toggles.values(),
                      trailing=(*trailing, FigureLanguageSwitch(session)))
        states = (table.partial, *table.cycles)
        self.model = RowsModel(HEADERS, len(states), lambda i: table_row(states[i], table), digits=10)

        info = Card("Остановка и счётчики этого расчёта", right=label(f"{table.solver} · N = {table.N}", "tag"))
        if table.criterion_met:
            met = "да"
        elif table.stop_kind == "none":
            met = "нет — заданное число полушагов без критерия остановки"
        else:
            met = "нет — запрошенный расчёт не выполнен" if not table.requested_completed else "нет"
        rows = [("Статус решателя", f"{table.solver_status} — {table.stop_text}"), ("Выполнено", table.termination),
                ("Утверждённый критерий остановки выполнен", met), ("Научный статус", table.scientific_status)]
        names = {"tol_inf": "tol_∞", "tol_E": "tol_E", "s_min": "s_min", "s_max": "s_max"}
        params = [f"{names[k]} = {fmt(v, 6)}" for k, v in table.parameters if k in names]
        if params:
            rows.append(("Параметры остановки", "; ".join(params)))
        if table.last_changes:
            rows.append(("ε^∞; ε^E последних полушагов", "   ".join(f"s = {s}: {fmt(ei, 4)}; {fmt(ee, 4)}"
                                                                     for s, ei, ee in table.last_changes)))
        for k, v in rows:
            line = QHBoxLayout()
            line.addWidget(label(k, "fieldLabel"))
            line.addWidget(label(str(v), wrap=True), 1)
            info.body.addLayout(line)
        # what this run actually computed and saved (not the choice made before the run), beside how it stopped
        self.availability = label(availability_text(table), wrap=True)
        info.body.addWidget(self.availability)
        info.body.addStretch(1)

        card = Card(f"Полные циклы: {len(table.drawn)} из {vim_cycles.K_MAX} на графике",
                    right=label(f"q = {fmt(table.q, 8)} Pa", "tag"))
        if not table.supported:
            card.body.addWidget(Note(table.route_reason, "testdata"))
        elif table.ref_w_m is None:
            card.body.addWidget(Note(f"Эталон недоступен: {table.reference_reason}. Разности не вычисляются.", "testdata"))
        view = table_view(self.model)
        view.setMinimumHeight(180)  # the four states stay visible
        card.body.addWidget(view)
        for text in (vim_cycles.OWNER_NOTE + ".",
                     "Происхождение значений: " + (table.provenance or "—") + "; это не итоговое поле и не пересчёт. "
                     "Наличие VIM-3 — не сходимость: остановка — по статусу решателя выше.",
                     "Разности знаковые, порога согласия нет; более поздний цикл не объявляется более точным."):
            card.body.addWidget(label(text, "secondary", wrap=True))
        split = QSplitter(Qt.Orientation.Vertical)
        upper = QSplitter(Qt.Orientation.Horizontal)  # the graph and, beside it, how the run stopped
        upper.addWidget(self.figure)
        upper.addWidget(info)
        upper.setSizes([1300, 600])
        split.addWidget(upper)
        split.addWidget(card)
        self.figure.setMinimumHeight(300)
        split.setSizes([460, 440])
        box = QVBoxLayout()
        box.addWidget(split)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(page(bar, padded(box)))

    def _toggle(self, figure_key, display_key, shown):
        """Show or hide one curve: display only (the history and the reference are not touched)."""
        (self.display.add if shown else self.display.discard)(display_key)
        self.figure.set_visible(figure_key, shown)

    def chosen(self):
        """The current visibility as display keys (kept when the tab is rebuilt for a late reference)."""
        return frozenset(self.display)
