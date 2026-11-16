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
"""

from datetime import datetime
from pathlib import Path

import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from kkvim_gui.analysis.results import MM, RowsModel, fmt, padded, page, table_view, toolbar
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


class SeriesFigure(QFrame):
    """Interactive X–Y graph: KK-VIM points of the series and, for SSSS, the Navier reference."""

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
        self.restyle()
        session.themeChanged.connect(self.restyle)
        session.figureLanguageChanged.connect(self.redraw)
        self.redraw()

    def restyle(self, *_):
        t = self.session.tokens
        self.plot.setBackground(t["plot.bg"])
        for side in ("left", "bottom"):
            axis = self.plot.getAxis(side)
            axis.setPen(t["plot.axis"])
            axis.setTextPen(t["plot.axis"])

    def set_quantity(self, quantity):
        self.quantity = quantity
        self.redraw()

    def set_shown(self, key, shown):
        """Show or hide one curve: "kkvim", "navier", "d2" or the cycle k (1 … 3)."""
        self.shown[key] = bool(shown)
        self.redraw()

    def cycle_curve(self, k):
        """(xs, ys) in kPa and mm of VIM-k: the points whose own run completed the cycle k (centre only)."""
        if self.quantity != "centre":
            return [], []  # the cycle states are recorded at the centre node only
        pts = sorted(self.data.cycle_included(k), key=lambda p: p.q)
        return [p.q * KPA for p in pts], [p.cycle(k).w_m * MM for p in pts]

    def d2_curve(self):
        """(D2 xs, ys) in kPa and mm: the D2 points actually computed (centre only)."""
        if self.quantity != "centre":
            return [], []  # D2 is reported at the centre node only
        pts = sorted(self.data.d2_included, key=lambda p: p.q)
        return [p.q * KPA for p in pts], [p.d2_w_m * MM for p in pts]

    def curves(self):
        """(KK-VIM xs, ys, Navier xs, ys) in kPa and mm: the plotted coordinates."""
        pts = sorted(self.data.included, key=lambda p: p.q)
        key = "w_centre_m" if self.quantity == "centre" else "w_at_max_abs_m"
        kx = [p.q * KPA for p in pts]
        ky = [getattr(p, key) * MM for p in pts]
        nav = sorted((p for p in self.data.points if p.navier_w_m is not None), key=lambda p: p.q)
        if self.quantity != "centre":
            nav = []  # the reference is evaluated at the centre only
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
        self.kkvim.setData(kx, ky)
        self.kkvim.setVisible(bool(kx))
        if kx:
            self.legend.addItem(self.kkvim, fig_text(lang, "legend.series.kkvim"))
        for k, item in self.cycle_items.items():
            cx, cy = self.cycle_curve(k) if self.shown[k] else ([], [])
            item.setData(cx, cy)
            item.setVisible(bool(cx))
            if cx:
                self.legend.addItem(item, fig_text(lang, "legend.series.cycle", label=vim_cycles.label(k), k=k, s=2 * k))
        self.navier.setData(nx, ny)
        self.navier.setVisible(bool(nx))
        if nx:
            self.legend.addItem(self.navier, fig_text(lang, "legend.series.navier", M=self.data.navier.get("M")))
        dx, dy = self.d2_curve() if self.shown["d2"] else ([], [])
        self.d2.setData(dx, dy)
        self.d2.setVisible(bool(dx))
        if dx:
            self.legend.addItem(self.d2, fig_text(lang, "legend.series.d2"))
        self.plot.autoRange()


class SeriesTab(QWidget):
    def __init__(self, session, data, trailing=(), parent=None):
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
        self.toggles = {"kkvim": QCheckBox("КК-ВИМ (итог)")}
        if data.cycles_supported:
            self.toggles.update({k: QCheckBox(vim_cycles.label(k)) for k in range(1, vim_cycles.K_MAX + 1)})
        if data.navier is not None:
            self.toggles["navier"] = QCheckBox("Навье")
        if data.d2:
            self.toggles["d2"] = QCheckBox("МКР D2")
        for key, box in self.toggles.items():
            box.setChecked(True)
            box.toggled.connect(lambda shown, k=key: self.figure.set_shown(k, shown))
            if isinstance(key, int):
                box.setToolTip(f"{vim_cycles.cycle_text(key)}: при каждом q — из истории полушагов расчёта этой точки")
        bar = toolbar(self.quantity, *self.toggles.values(),
                      trailing=(*trailing, FigureLanguageSwitch(session), self.save_button))
        cycles = data.cycles_supported
        headers = HEADERS + (D2_HEADERS if data.d2 else ()) + ((*CYCLE_HEADERS, "VIM-1–3: примечание") if cycles else ())
        self.model = RowsModel(headers, len(data.points), lambda i: _row(data.points[i], data.d2, cycles), digits=10)
        table = Card(f"Расчёты серии: {len(data.points)} значений q, на графике — {len(data.included)}",
                     right=label(f"{data.supports} · N = {data.N} · сетка {data.grid[0]} × {data.grid[1]}", "tag"))
        excluded = [p for p in data.points if not p.included]
        if excluded:
            table.body.addWidget(Note(f"Не на графике: {len(excluded)} — причина в последнем столбце; значения не "
                                      "подставляются и не интерполируются.", "testdata"))
        table.body.addWidget(table_view(self.model))
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
        split = QSplitter(Qt.Orientation.Vertical)
        split.addWidget(self.figure)
        split.addWidget(table)
        split.setSizes([420, 300])
        box = QVBoxLayout()
        box.addWidget(split)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(page(bar, padded(box)))

    def folder_name(self):
        return f"kkvim_series_{datetime.now():%Y%m%d-%H%M%S}_s{self.data.series_id:03d}"

    def save(self, folder):
        """Save the series package into a new subfolder of ``folder``."""
        def render(target):
            path = Path(target) / "load_deflection.png"
            if not self.figure.grab().save(str(path), "PNG"):
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

    def __init__(self, session, data, parent=None):
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
        self.tab = SeriesTab(session, data)
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
