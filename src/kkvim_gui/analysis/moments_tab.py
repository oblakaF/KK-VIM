"""«Моменты и напряжения» (R6 research; GUI_DESIGN.md 8; R6 review 8.5): the quantities of the plate field next to
the deflection, with their sections, values, independent comparisons and limitations.

- Quantity: w, M_x, M_y, M_xy (L-107 convention), M_1, M_2, σ_x, σ_y, τ_xy; stresses at the bottom surface (z = +t/2),
  the top surface (z = −t/2) or a chosen z in [−t/2, t/2] (z downward). The values are those of ``kkvim.moments``
  through ``backend.moments_view`` (no formula here); the display only scales the unit (mm, N·m/m, MPa).
- Colour field with isolines (``results.FieldFigure``; nodes at their own coordinates, the orientation of the solver
  array: axis 0 = x), its publication export (``plot_tools.FieldPublisher``), and two sections along node lines with the
  chart tools of P5 (Snap, raw values, CSV, publication); the raw values are the stored values in SI.
- Panel: R6 status (RESEARCH / NOT_QUALIFIED; DIAGNOSTIC for a diagnostic field), the displacement status unchanged,
  units, the centre value and the sampled extrema with their nodes, conventions, operators, what is not computed.
- «Сравнение с эталоном моментов»: started on request in a separate process (Navier and Levy for SSSS; D2 and Ritz for
  CCCC); the reference is drawn in the sections and the differences are listed as OBSERVATIONS (no threshold).
"""

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from kkvim_gui.analysis.results import (
    MM,
    Figure,
    FieldFigure,
    NodeSelector,
    RowsModel,
    centre_value,
    fmt,
    labelled,
    page,
    scrolled,
    table_view,
    toolbar,
)
from kkvim_gui.backend import moments_view as mv
from kkvim_gui.figures.text import fig_text
from kkvim_gui.widgets.common import label

TAB_TITLE = "Моменты и напряжения"
TAB_TOOLTIP = ("R6, исследование: изгибающие и крутящий моменты, главные моменты и напряжения в плоскости на поверхностях "
               "и по толщине — постобработка поля прогиба (kkvim.moments); RESEARCH / NOT_QUALIFIED")
ORDER = ("w", "Mx", "My", "Mxy", "M1", "M2", "sigma_x", "sigma_y", "tau_xy")


def _key(quantity, surface):
    return f"q.{quantity}.{surface}" if mv.QUANTITIES[quantity].kind == "stress" else f"q.{quantity}"


class QuantityFieldFigure(FieldFigure):
    """The colour field of one registered quantity (display units), nodes at their own coordinates."""

    def __init__(self, session, outcome, data, parent=None):
        super().__init__(session, outcome, isolines=True, parent=parent)
        self.data, self.key, self.surface, self.z = data, "w", "bottom", None
        self.set_key("w")

    def values(self):
        data = getattr(self, "data", None)
        if data is None:  # during the construction of the base class (the deflection)
            return super().values()
        return data.display(self.key, self.surface, self.z)

    def set_key(self, key, surface=None, z=None):
        self.key = key
        self.quantity = key
        if surface is not None:
            self.surface = surface
        self.z = z
        self.retranslate()
        self.update_image()

    def _z_text(self):
        return {"z": fmt(self.z * MM, 6)} if self.surface == "z" and self.z is not None else {}

    def field_texts(self, lang):
        base = _key(self.key, self.surface)
        return (fig_text(lang, f"{base}.field", **self._z_text()), fig_text(lang, f"q.{self.key}.unit"))

    def extra_marks(self, lang):
        return () if self.key == "w" else (fig_text(lang, "mark.r6"),)

    def extra_meta(self):
        return {"quantity": self.key, "unit": mv.QUANTITIES[self.key].display_unit,
                "surface": self.surface if mv.QUANTITIES[self.key].kind == "stress" else "",
                "z_m": self.z if self.surface == "z" else None,
                "r6_status": "" if self.key == "w" else mv.STATUS}

    def retranslate(self, *_):
        super().retranslate()
        if getattr(self, "data", None) is None:
            return
        lang = self.session.figure_language
        title, bar = self.field_texts(lang)
        self.title.setText(title)
        marks = [m for m in (self.mark.text(),) if m] + list(self.extra_marks(lang))
        self.mark.setText(" · ".join(marks))
        self.mark.setVisible(bool(marks))
        if self.bar is not None:
            self.bar.axis.setLabel(bar)

    def update_image(self):
        super().update_image()
        if getattr(self, "data", None) is not None and self.bar is not None:
            self.bar.axis.setLabel(self.field_texts(self.session.figure_language)[1])


class QuantitySection(Figure):
    """One section of the chosen quantity along a node line (x at y_j or y at x_i), with an optional reference curve."""

    def __init__(self, session, outcome, direction, parent=None):
        super().__init__(session, outcome, "section.x.title" if direction == "x" else "section.y.title",
                         {"y": 0.0} if direction == "x" else {"x": 0.0}, parent=parent)
        self.direction = direction
        self.key, self.surface, self.z, self.index = "w", "bottom", None, 0
        self.ref_name = None
        t = session.tokens
        self.curve = self.plot.plot(pen=pg.mkPen(t["accent"], width=2), symbol="o", symbolSize=4, symbolBrush=t["accent"])
        self.ref_curve = self.plot.plot(pen=pg.mkPen(t["status.warning"], width=2, style=Qt.PenStyle.DashLine),
                                        symbol="t", symbolSize=5, symbolBrush=t["status.warning"])
        self.ref_curve.setVisible(False)

    def texts(self, lang):
        base = _key(self.key, self.surface)
        z = {"z": fmt(self.z * MM, 6)} if self.surface == "z" and self.z is not None else {}
        coord = ({"y": float(self.outcome.y[self.index])} if self.direction == "x" else {"x": float(self.outcome.x[self.index])})
        return (fig_text(lang, f"{base}.section.{self.direction}", **coord, **z),
                fig_text(lang, "axis.x" if self.direction == "x" else "axis.y"), fig_text(lang, f"q.{self.key}.unit"))

    def retranslate(self, *_):
        super().retranslate()
        if not hasattr(self, "direction"):
            return
        lang = self.session.figure_language
        title, xl, yl = self.texts(lang)
        self.title.setText(title)
        self.plot.setLabel("bottom", xl)
        self.plot.setLabel("left", yl)
        if self.key != "w":
            marks = [m for m in (self.mark.text(),) if m] + [fig_text(lang, "mark.r6")]
            self.mark.setText(" · ".join(marks))
            self.mark.setVisible(True)


class MomentsTab(QWidget):
    """The R6 quantities of one outcome (see the module text)."""

    def __init__(self, session, outcome, trailing=(), parent=None):
        super().__init__(parent)
        self.session, self.outcome = session, outcome
        self.data = mv.MomentsData.of(outcome) if outcome.has_field else None
        self.job, self.record, self.refs = None, None, {}
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        if self.data is None or not self.data.available:
            reason = "поле прогиба не вычислено" if self.data is None else self.data.reason
            note = label(f"Моменты и напряжения недоступны: {reason}. Ничего не подставляется.", "secondary", wrap=True)
            box = QVBoxLayout()
            box.setContentsMargins(16, 16, 16, 16)
            box.addWidget(note)
            box.addStretch(1)
            content = QWidget()
            content.setLayout(box)
            lay.addWidget(page(toolbar(trailing=trailing), content))
            self.unavailable_reason = reason
            return
        self.unavailable_reason = ""
        self._build(trailing)

    # ------------------------------------------------------------------------------------------------ layout

    def _build(self, trailing):
        o = self.outcome
        self.quantity = QComboBox()
        for key in ORDER:
            q = mv.QUANTITIES[key]
            self.quantity.addItem(f"{q.name_ru}, {q.display_unit}", key)
        self.quantity.setCurrentIndex(ORDER.index("Mx"))
        self.surface = QComboBox()
        for key in mv.SURFACES:
            self.surface.addItem(mv.SURFACE_TEXT[key], key)
        self.z = QDoubleSpinBox()
        half = o.request.t / 2 * MM
        self.z.setRange(-half, half)
        self.z.setDecimals(6)
        self.z.setSingleStep(half / 4 if half > 0 else 0.1)
        self.z.setSuffix(" mm")
        self.z.setValue(half / 2)
        self.z.setToolTip("Координата по толщине, ось z вниз (нагрузка сверху): −t/2 … +t/2")
        self.isolines = QCheckBox("Изолинии")
        self.isolines.setChecked(True)
        self.cmap = QComboBox()
        self.cmap.addItems(["viridis", "cividis"])
        self.field = QuantityFieldFigure(self.session, o, self.data)
        from kkvim_gui.analysis.plot_tools import FIGURE_MIN_HEIGHT, FieldPublisher

        self.publisher = FieldPublisher(self.session, self.field, figure_id="field_Mx")
        self.isolines.toggled.connect(self.field.set_isolines)
        self.cmap.currentTextChanged.connect(self.field.set_colormap)
        c = centre_value(o)
        e = o.extremum()
        i0, j0 = (c["i"], c["j"]) if c is not None else (e["i"], e["j"])
        self.at_y = NodeSelector("y", o.y, j0)
        self.at_x = NodeSelector("x", o.x, i0)
        self.sec_x = QuantitySection(self.session, o, "x")
        self.sec_y = QuantitySection(self.session, o, "y")
        for fig in (self.field, self.sec_x, self.sec_y):
            fig.setMinimumHeight(FIGURE_MIN_HEIGHT)
        self.tools_x = self._tools(self.sec_x)
        self.tools_y = self._tools(self.sec_y)
        self.compare_button = QPushButton("Сравнить с эталоном моментов")
        self.compare_button.setToolTip("Независимый эталон в отдельном процессе: SSSS — Навье и Леви; CCCC — МКР D2 (та же "
                                       "дискретная задача) и Ритц. Разности — наблюдения, без порогов")
        self.compare_button.clicked.connect(self.start_reference)
        self.ref_choice = QComboBox()
        self.ref_choice.setEnabled(False)
        self.ref_choice.setMinimumContentsLength(10)
        self.ref_choice.currentIndexChanged.connect(lambda _: self.update_view())
        bar1 = toolbar(labelled("Величина", self.quantity), labelled("Поверхность", self.surface), labelled("z", self.z),
                       self.isolines, labelled("Шкала", self.cmap), trailing=trailing)
        bar2 = toolbar(labelled("сечение по x при", self.at_y), labelled("по y при", self.at_x), self.compare_button,
                       labelled("Эталон на графиках", self.ref_choice))
        for signal in (self.quantity.currentIndexChanged, self.surface.currentIndexChanged, self.z.valueChanged,
                       self.at_y.spin.valueChanged, self.at_x.spin.valueChanged):
            signal.connect(lambda *_: self.update_view())
        self.info = label("", "secondary", wrap=True)
        self.info.setTextFormat(Qt.TextFormat.RichText)
        self.info.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.ref_info = label("Эталон моментов не запрошен.", "secondary", wrap=True)
        self.compare_model = None
        self.compare_view = table_view(RowsModel(("Величина", "Место", "КК-ВИМ, N", "Эталон, N", "Разность, N", "Отн."), 0,
                                                 lambda r: ()))
        self.compare_view.setMinimumHeight(180)
        grid = QGridLayout()  # field | section along x; section along y | the comparison with the reference
        grid.addWidget(self.field, 0, 0)
        grid.addWidget(self.sec_x, 0, 1)
        grid.addWidget(self.sec_y, 1, 0)
        compare = QVBoxLayout()
        compare.addWidget(self.ref_info)
        compare.addWidget(self.compare_view, 1)
        grid.addLayout(compare, 1, 1)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        left = QWidget()
        left.setLayout(grid)
        panel = QScrollArea()
        panel.setWidgetResizable(True)
        panel.setFrameShape(QFrame.Shape.NoFrame)
        inner = QWidget()
        inner_lay = QVBoxLayout(inner)
        inner_lay.addWidget(self.info)
        inner_lay.addStretch(1)
        panel.setWidget(inner)
        panel.setFixedWidth(360)
        row = QHBoxLayout()  # the figures scroll on a low window; the panel keeps its own scroll bar beside them
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(scrolled(left), 1)
        row.addWidget(panel)
        content = QWidget()
        content.setLayout(row)
        bars = QWidget()
        bars_lay = QVBoxLayout(bars)
        bars_lay.setContentsMargins(0, 0, 0, 0)
        bars_lay.setSpacing(0)
        bars_lay.addWidget(bar1)
        bars_lay.addWidget(bar2)
        self.layout().addWidget(page(bars, content))
        self.timer = QTimer(self)
        self.timer.setInterval(300)
        self.timer.timeout.connect(self._poll)
        self.update_view()

    def _tools(self, fig):
        from kkvim_gui.analysis.plot_tools import Curve, default_style

        n = self.outcome.request.N

        def curves():
            out = [Curve("kkvim", fig.curve, f"КК-ВИМ, N = {n}", lambda lang: fig_text(lang, "legend.kkvim.q", N=n),
                         lambda: self.section_rows(fig, None), default_style(0, "solid", "o"))]
            if self.refs and fig.ref_name is not None:
                name = fig.ref_name
                out.append(Curve("reference", fig.ref_curve, self.refs[name]["short"],
                                 lambda lang: fig_text(lang, f"legend.ref.{name}", M=self.refs[name].get("M", "")),
                                 lambda: self.section_rows(fig, name), default_style(1, "dash", "t")))
            return out

        from kkvim_gui.analysis.plot_tools import figure_marks

        tools = fig.attach_tools(f"moments_section_{fig.direction}", curves, self.columns(),
                                 lambda: ((fig.direction, "m"), (mv.QUANTITIES[fig.key].symbol,
                                                                mv.QUANTITIES[fig.key].display_unit)))
        o = self.outcome
        tools.info.marks = lambda lang: tuple(figure_marks(o, lang)) + (
            () if fig.key == "w" else (fig_text(lang, "mark.r6"),))
        return tools

    def columns(self):
        q = mv.QUANTITIES[self.current()[0]] if hasattr(self, "quantity") else mv.QUANTITIES["w"]
        return (("i", ""), ("j", ""), ("x", "m"), ("y", "m"), (q.key, q.unit))

    # ------------------------------------------------------------------------------------------------ state

    def current(self):
        key = self.quantity.currentData()
        surface = self.surface.currentData()
        z = self.z.value() / MM if surface == "z" else None
        return key, surface, z

    def values(self, name=None):
        """SI values of the current quantity: KK-VIM (name None) or a reference moment field (moments only)."""
        key, surface, z = self.current()
        if name is None:
            return self.data.values(key, surface, z)
        return self.refs[name][key]

    def section_rows(self, fig, name):
        values = self.values(name)
        o = self.outcome
        if fig.direction == "x":
            j = self.at_y.index()
            return [(i, j, float(o.x[i]), float(o.y[j]), float(values[i, j])) for i in range(len(o.x))]
        i = self.at_x.index()
        return [(i, j, float(o.x[i]), float(o.y[j]), float(values[i, j])) for j in range(len(o.y))]

    def update_view(self):
        key, surface, z = self.current()
        stress = mv.QUANTITIES[key].kind == "stress"
        self.surface.setEnabled(stress)
        self.z.setEnabled(stress and surface == "z")
        self.field.set_key(key, surface, z)
        self.publisher.info.figure_id = f"field_{key}" + (f"_{surface}" if stress else "")
        q = mv.QUANTITIES[key]
        ref_name = self.ref_choice.currentData() if self.refs and key in mv.MOMENT_KEYS else None
        for fig, idx in ((self.sec_x, self.at_y.index()), (self.sec_y, self.at_x.index())):
            fig.key, fig.surface, fig.z, fig.index = key, surface, z, idx
            fig.ref_name = ref_name
            values = self.data.display(key, surface, z)
            o = self.outcome
            if fig.direction == "x":
                fig.curve.setData(o.x, values[:, idx])
            else:
                fig.curve.setData(o.y, values[idx, :])
            if ref_name is not None:
                ref = self.refs[ref_name][key] * q.scale
                fig.ref_curve.setData(o.x, ref[:, idx]) if fig.direction == "x" else fig.ref_curve.setData(o.y, ref[idx, :])
            fig.ref_curve.setVisible(ref_name is not None)
            fig.retranslate()
            if getattr(fig, "tools", None) is not None:
                fig.tools.info.columns = self.columns()
                fig.tools.rehome()
        self.info.setText(self.info_html())
        self._fill_comparison()

    # ------------------------------------------------------------------------------------------------ texts

    def info_html(self):
        o, d = self.outcome, self.data
        key, surface, z = self.current()
        q = mv.QUANTITIES[key]
        s = d.summary(key, surface, z)

        def place(p):
            return f"{fmt(p['value'] * q.scale + 0.0, 6)} {q.display_unit} в узле i = {p['i']}, j = {p['j']} " \
                   f"(x = {fmt(p['x'], 6)} m, y = {fmt(p['y'], 6)} m)"

        status = ("<b>ДИАГНОСТИКА</b>: моменты поля последнего выполненного полушага расчёта, который не завершился как "
                  "запрошено; это не результат." if d.diagnostic else "")
        lines = [f"<b>R6: {mv.STATUS}</b>. Точность моментов и напряжений не квалифицирована; R5 и R6 не приняты.",
                 status,
                 f"Статус прогиба (не переносится на моменты): {o.status}" + (f" · {o.mark}" if o.mark else ""),
                 f"<b>{q.name_ru}</b>, {q.display_unit} (в файлах — {q.unit}).",
                 ("Поверхность: " + mv.SURFACE_TEXT[surface] + (f" = {fmt(z * MM, 6)} mm" if surface == "z" else ""))
                 if q.kind == "stress" else "",
                 "Центр: " + (place(s["centre"]) if s["centre"] else "центр пластины — не узел сетки (нечётное n), "
                                                                     "значение не интерполируется"),
                 "Наибольшее по узлам: " + place(s["max"]),
                 "Наименьшее по узлам: " + place(s["min"]),
                 f"Соглашения: {d.texts['conventions']}.",
                 f"Граничная кривизна CCCC: правило {d.texts['cc_edge_rule']}." if o.request.supports == "CCCC" else
                 "SSSS: на краях M_x = M_y = 0 (точно), M_xy ≠ 0.",
                 f"Модель округления моментов: {fmt(d.moments.rounding_model, 3)} N (модель, не граница).",
                 "Не вычисляется: " + "; ".join(mv.NOT_COMPUTED) + ".",
                 "Ограничения: " + " ".join(mv.LIMITATIONS_RU)]
        return "<br><br>".join(line for line in lines if line)

    def spec_source(self, key, surface="bottom", z=None):
        return QuantitySpecSource(self, key, surface, z)

    # ------------------------------------------------------------------------------------------------ references

    def start_reference(self):
        if self.job is not None:
            return
        if mv.reference_kind(self.outcome) is None:
            self.ref_info.setText("Для этого закрепления эталона моментов нет.")
            return
        self.job = mv.MomentsRefJob(self.outcome)
        self.compare_button.setEnabled(False)
        self.ref_info.setText("Эталон моментов вычисляется в отдельном процессе…")
        self.timer.start()

    def _poll(self):
        record = self.job.poll() if self.job is not None else None
        if record is None:
            return
        self.timer.stop()
        self.set_reference(record)

    def set_reference(self, record):
        self.record = record
        self.refs = mv.reference_fields(record)
        self.ref_choice.blockSignals(True)
        self.ref_choice.clear()
        for name, ref in self.refs.items():
            self.ref_choice.addItem(ref["short"], name)
        self.ref_choice.setEnabled(bool(self.refs))
        self.ref_choice.blockSignals(False)
        if record.get("state") != "ok":
            self.ref_info.setText(f"Эталон моментов недоступен: {record.get('reason') or record.get('state')}.")
        else:
            notes = [f"{ref['title']}: {ref['status']}" for ref in self.refs.values()]
            self.ref_info.setText("Эталон моментов рассчитан. " + " ".join(notes) +
                                  " Разности ниже — наблюдения, без порога и без PASS/FAIL.")
        self.compare_button.setEnabled(True)
        self.job = None
        self.update_view()

    def comparison_rows(self):
        """Rows of the comparison table for the chosen reference (node fields) and the Navier/Levy points."""
        name = self.ref_choice.currentData() if self.refs else None
        rows = []
        if name is not None:
            rows += [(r["quantity"], r["place"], r["kkvim"], r["reference"], r["difference"], r.get("relative"))
                     for r in mv.compare(self.data, self.refs[name])]
        if name == "navier":
            rows += [(r["quantity"], f"Навье − Леви, {r['place']}", r["navier"], r["levy"], r["difference"], None)
                     for r in mv.levy_rows(self.record)]
        return rows

    def _fill_comparison(self):
        rows = self.comparison_rows() if self.refs else []
        self.compare_model = RowsModel(("Величина", "Место", "КК-ВИМ, N", "Эталон, N", "Разность, N", "Отн."), len(rows),
                                       lambda k: rows[k], digits=8)
        self.compare_view.setModel(self.compare_model)


# ------------------------------------------------------------------------------------------------ report sources


def field_spec(tab, key, surface, z, lang):
    """FieldSpec of one quantity of the tab's outcome for the publication / report (the stored kkvim values, scaled)."""
    from kkvim_gui.analysis.plot_tools import figure_marks
    from kkvim_gui.figures import publication as pub

    o, d = tab.outcome, tab.data
    base = _key(key, surface)
    zt = {"z": fmt(z * MM, 6)} if surface == "z" and z is not None else {}
    marks = tuple(figure_marks(o, lang)) + (() if key == "w" else (fig_text(lang, "mark.r6"),))
    return pub.FieldSpec(
        title=fig_text(lang, f"{base}.field", **zt), x_label=fig_text(lang, "axis.x"), y_label=fig_text(lang, "axis.y"),
        colorbar_label=fig_text(lang, f"q.{key}.unit"), x=tuple(float(v) for v in o.x), y=tuple(float(v) for v in o.y),
        values=np.array(d.display(key, surface, z), dtype=float), cmap="viridis", isolines=FieldFigure.ISOLINES,
        marks=marks, test_data=o.is_demo, lang=lang,
        meta={"figure": f"field_{key}", "solver": o.solver, "data_origin": o.origin, "quantity": key,
              "unit": mv.QUANTITIES[key].display_unit, "surface": surface if mv.QUANTITIES[key].kind == "stress" else "",
              "r6_status": "" if key == "w" else mv.STATUS,
              "values": "kkvim.moments of the stored solver field on the result nodes; colour scale only"})


class QuantitySpecSource:
    """A report figure of one quantity (``spec(lang)``), independent of what the tab currently shows."""

    def __init__(self, tab, key, surface="bottom", z=None):
        self.tab, self.key, self.surface, self.z = tab, key, surface, z

    def spec(self, lang):
        return field_spec(self.tab, self.key, self.surface, self.z, lang)


REPORT_FIGURES = (("Mx", "bottom"), ("My", "bottom"), ("Mxy", "bottom"), ("sigma_x", "bottom"))


def summary_rows(tab):
    """(quantity, unit, centre, max with node, min with node) for every registered quantity except w (report table)."""
    rows = []
    for key in ORDER[1:]:
        q = mv.QUANTITIES[key]
        surfaces = ("bottom", "top") if q.kind == "stress" else ("bottom",)
        for surface in surfaces:
            s = tab.data.summary(key, surface)
            name = q.name_ru + (f" ({mv.SURFACE_TEXT[surface]})" if q.kind == "stress" else "")
            centre = s["centre"]["value"] if s["centre"] else None
            rows.append((name, q.unit, centre, s["max"]["value"], f"i = {s['max']['i']}, j = {s['max']['j']}",
                         s["min"]["value"], f"i = {s['min']['i']}, j = {s['min']['j']}"))
    return rows
