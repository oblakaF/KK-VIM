"""The «МКР D2» tab of the analysis window: the D2 solution computed for this CCCC run (GUI-MULTIMETHOD-CCCC-D2-01).

Everything shown comes from the D2 record of the run (``RunRecord.d2``, ``backend.d2_job``: a separate process running
``kkvim_reference.r5_cccc_d2``) and from the KK-VIM outcome; nothing is recomputed, interpolated or scaled here:

- the state of D2 and, when it did not run, the reason (no value is substituted);
- method, source and the exact input it received (the same a, b, h, E, ν, q and grid as KK-VIM), its own error estimate
  e_D2 and the refinement record;
- dimensional (w_c, m and mm) and dimensionless values: β_a = w_c D/(q a⁴) (length a, as the comparison) and the
  reference's own α = w_c D/(q l⁴) with the shorter side l — labelled apart, never compared with each other;
- KK-VIM and D2 along one node line of the common grid (w(x) at y_j, w(y) at x_i), and the largest |w_KK-VIM − w_D2| over
  the nodes — only for a KK-VIM result field, never a diagnostic one;
- the D2 values at all nodes (table);
- the computation times of both methods with the boundaries of every timer (``backend.d2_job.TIMER_TEXT``): one
  execution each — an observation, never a ranking of the algorithms.
"""

import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QScrollArea, QSplitter, QVBoxLayout, QWidget

from kkvim_gui.analysis.results import (
    MM,
    Figure,
    NodeSelector,
    RowsModel,
    fmt,
    key_value,
    labelled,
    padded,
    page,
    table_view,
    toolbar,
)
from kkvim_gui.backend import d2_job
from kkvim_gui.figures.text import fig_text
from kkvim_gui.widgets.common import Card, Note, StatusLabel, label, set_help_context

TAB_TITLE = "МКР D2"
TAB_TOOLTIP = ("Независимый расчёт МКР — двумерный (D2) для тех же входных данных: поле, сечения рядом с КК-ВИМ, "
               "значения в узлах и время вычислений")
VALUE_HEADERS = ("i", "j", "x, m", "y, m", "w D2, m", "w КК-ВИМ, m", "Δw = w_КК-ВИМ − w_D2, m")
SCOPE_NOTE = ("D2 — независимая реализация той же дискретной задачи, что и КК-ВИМ CCCC (D-049.1, D-049.8). Разность "
              "КК-ВИМ − D2 — отличие приближения КК-ВИМ (N членов, s полушагов) от полного дискретного решения на той же "
              "сетке; она не является ошибкой континуальной модели пластины и не проверяет сеточную погрешность. "
              "Порога согласия нет; научный статус КК-ВИМ не меняется.")


def _s(value):
    return "—" if value is None else f"{value:.4f} с"


def timing_rows(run):
    """(timer, value, definition) of both methods of ``run`` (the same texts as report.html)."""
    rows = []
    kk = getattr(run, "timing", None) or {}
    for key in ("kkvim_solver_s", "kkvim_import_s", "kkvim_wall_s"):
        rows.append((key, kk.get(key), d2_job.TIMER_TEXT[key]))
    d2 = getattr(run, "d2", None) or {}
    t = d2.get("timing") or {}
    for key, name in (("solve_s", "d2_solve_s"), ("import_s", "d2_import_s"), ("post_s", "d2_post_s"),
                      ("worker_s", "d2_worker_s"), ("wall_s", "d2_wall_s"), ("parse_s", "d2_parse_s")):
        rows.append((name, t.get(key), d2_job.TIMER_TEXT[name]))
    return rows


class D2Tab(QWidget):
    def __init__(self, session, outcome, run, trailing=(), parent=None):
        super().__init__(parent)
        set_help_context(self, "verification.d2")
        self.session, self.outcome, self.run = session, outcome, run
        self.record = run.d2
        d2 = self.record
        ok = d2.get("state") == d2_job.OK
        body = QVBoxLayout()
        head = Card(d2_job.TITLE, right=label("рассчитан для этих входных данных" if ok else d2.get("state", ""), "tag"))
        self.state_label = StatusLabel("neutral" if ok else "warning",
                                       d2_job.STATE_TEXT.get(d2.get("state"), d2.get("state", "")))
        self.state_label.setWordWrap(True)
        head.body.addWidget(self.state_label)
        if d2.get("reason"):
            reason = label(d2["reason"], wrap=True)
            head.body.addWidget(reason)
        head.body.addWidget(Note(SCOPE_NOTE, "info"))
        body.addWidget(head)
        self.figure_x = self.figure_y = None
        self.values = None
        self.difference = None
        self.result_rows = []
        if ok:
            self.result_rows = self._result_rows()
            results = Card("Результаты D2 (размерные и безразмерные)")
            results.body.addLayout(key_value(self.result_rows))
            body.addWidget(results)
            method = Card("Метод, вход и уточнение")
            method.body.addLayout(key_value(self._method_rows()))
            body.addWidget(method)
        timing = Card("Время вычислений (одно исполнение; наблюдение, не сравнение алгоритмов)")
        self.timing = timing_rows(run)
        timing.body.addLayout(key_value([(name, f"{_s(value)} — {text}") for name, value, text in self.timing]))
        timing.body.addLayout(key_value(self._runtime_rows()))
        timing.body.addWidget(label(d2_job.TIMING_ROLE, "secondary", wrap=True))
        body.addWidget(timing)
        body.addStretch(1)
        info = QWidget()
        info.setObjectName("Page")
        info_lay = QVBoxLayout(info)
        info_lay.setContentsMargins(0, 0, 0, 0)
        info_lay.addWidget(padded(body))
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setWidget(info)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        if not ok:
            lay.addWidget(page(toolbar(trailing=trailing), scroll))
            return
        field = d2["field"]
        self.x = [i * outcome.request.a / outcome.request.n_x for i in range(field.shape[0])]
        self.y = [j * outcome.request.b / outcome.request.n_y for j in range(field.shape[1])]
        self.kk_field = outcome.w if outcome.has_field and not outcome.diagnostic_only and outcome.w.shape == field.shape \
            else None
        i0, j0 = field.shape[0] // 2, field.shape[1] // 2  # the centre line (a node line; not interpolated)
        self.at_y = NodeSelector("y", self.y, j0)
        self.at_x = NodeSelector("x", self.x, i0)
        self.figure_x = Figure(session, outcome, "d2.section.x.title", {"y": self.y[j0]})
        self.figure_y = Figure(session, outcome, "d2.section.y.title", {"x": self.x[i0]})
        t = session.tokens
        self.curves = {}
        for fig, axis in ((self.figure_x, "axis.x"), (self.figure_y, "axis.y")):
            fig.set_axes(axis, "axis.w")
            legend = fig.plot.addLegend(offset=(10, 10))
            d2_curve = fig.plot.plot(pen=pg.mkPen(t["status.warning"], width=2, style=Qt.PenStyle.DashLine),
                                     symbol="t", symbolSize=6, symbolBrush=t["status.warning"])
            kk_curve = fig.plot.plot(pen=pg.mkPen(t["accent"], width=2), symbol="o", symbolSize=4, symbolBrush=t["accent"])
            self.curves[fig] = (d2_curve, kk_curve, legend)
        self.at_y.spin.valueChanged.connect(lambda _: self.update_x())
        self.at_x.spin.valueChanged.connect(lambda _: self.update_y())
        session.figureLanguageChanged.connect(self._relegend)
        self.update_x()
        self.update_y()
        self._relegend()
        figures = QHBoxLayout()
        figures.addWidget(self.figure_x)
        figures.addWidget(self.figure_y)
        fig_box = QWidget()
        fig_box.setLayout(figures)
        model = RowsModel(VALUE_HEADERS, field.size, self._value_row, digits=10)
        self.values = table_view(model)
        values = Card(f"Значения D2 в узлах: {field.shape[0]} × {field.shape[1]}",
                      right=label("поле D2 сохраняется в d2_field.csv пакета расчёта", "tag"))
        values.body.addWidget(self.values)
        split = QSplitter()
        split.setOrientation(Qt.Orientation.Vertical)
        split.addWidget(fig_box)
        split.addWidget(values)
        split.addWidget(scroll)
        split.setSizes([360, 220, 360])
        bar = toolbar(labelled("w(x) при", self.at_y), labelled("w(y) при", self.at_x), trailing=trailing)
        bar.setToolTip(TAB_TOOLTIP)
        lay.addWidget(page(bar, split))

    def _value_row(self, k):
        field = self.record["field"]
        i, j = divmod(k, field.shape[1])
        w = float(field[i, j])
        kk = None if self.kk_field is None else float(self.kk_field[i, j])
        return (i, j, self.x[i], self.y[j], w, kk, None if kk is None else kk - w)

    def _result_rows(self):
        d2, o = self.record, self.outcome
        rows = []
        if d2["w_centre_m"] is None:
            rows.append(("w в центре", d2.get("centre_reason") or "центр не является узлом сетки: не интерполируется"))
        else:
            c = d2["centre"]
            rows += [("w в центре (узел a/2, b/2)", f"{fmt(d2['w_centre_m'] * MM, 10)} mm = {d2['w_centre_m']!r} m"),
                     ("Узел центра", f"i = {c['i']}, j = {c['j']}: x = {c['x']!r} m, y = {c['y']!r} m"),
                     ("β_a = w_c·D/(q·a⁴) (длина a)", repr(d2["beta_a"])),
                     (f"α = w_c·D/(q·l⁴), l = min(a, b) = {d2['short_side_m']!r} m (эталон)", repr(d2["alpha_short_side"]))]
        m = d2.get("w_max_abs_node") or {}
        if m:
            rows.append(("Наибольший |w| D2 (по узлам)", f"{m['w_m']!r} m в узле i = {m['i']}, j = {m['j']}"))
        rows.append(("Жёсткость D (эталон, flexural_rigidity)", f"{d2['D']!r} N·m; D КК-ВИМ = {o.D!r} N·m"))
        self.difference = d2_job.field_difference(d2, o)
        if self.difference is not None:
            dd = self.difference
            rows.append(("max |w_КК-ВИМ − w_D2| по узлам", f"{dd['delta_w_m']!r} m (знаковое) в узле i = {dd['i']}, "
                                                          f"j = {dd['j']}; относительно max|w_D2|: "
                                                          f"{fmt(dd['relative_to_max_abs_d2'], 6)}"))
        else:
            rows.append(("Разность полей", "не вычисляется: поле КК-ВИМ — диагностическое или не получено"))
        return rows

    def _method_rows(self):
        d2 = self.record
        inp = d2["inputs"]
        p = d2.get("problem") or {}
        return [
            ("Метод", d2["method"]),
            ("Источник", d2["source"]),
            ("Вход (СИ, точные значения)", f"a = {inp['a']!r} m · b = {inp['b']!r} m · h = {inp['t']!r} m · E = {inp['E']!r} Pa"
                                          f" · ν = {inp['nu']!r} · q = {inp['q']!r} Pa · {inp['supports']}"),
            ("Сетка", f"n_x = {inp['n_x']}, n_y = {inp['n_y']} интервалов; {p.get('unknowns')} неизвестных; "
                      f"упорядочение {p.get('ordering')}; ненулевых в LU {d2['factor_nnz']}"),
            ("Собственная оценка ошибки e_D2", f"{d2['e_D2_m']!r} m (последняя поправка уточнения; не ошибка сетки)"),
            ("Уточнение", f"{d2['refinement_steps']} шагов; поправки, m: {', '.join(repr(v) for v in d2['corrections_m'])}"),
            ("max |r| перед шагами", ", ".join(repr(v) for v in d2["residual_max"])),
            ("Версии", ", ".join(f"{k} {v}" for k, v in (d2.get("versions") or {}).items()) or "—"),
        ]

    def _runtime_rows(self):
        d2 = self.record or {}
        rt = d2.get("runtime") or {}
        p = d2.get("problem") or {}
        r = self.outcome.request
        rows = [("Сетка и неизвестные", f"{r.n_x} × {r.n_y} интервалов; D2: {p.get('unknowns', '—')} неизвестных "
                                        f"(внутренние узлы); КК-ВИМ: N = {r.N}, s = {self.outcome.s}")]
        if rt:
            rows.append(("Компьютер и среда (процесс D2)", f"{rt.get('platform')}; {rt.get('processor') or rt.get('machine')}; "
                                                           f"логических процессоров {rt.get('cpu_count')}; "
                                                           f"Python {rt.get('python')}"))
        return rows

    def update_x(self):
        j = self.at_y.index()
        d2_curve, kk_curve, _ = self.curves[self.figure_x]
        d2_curve.setData(self.x, self.record["field"][:, j] * MM)
        if self.kk_field is not None:
            kk_curve.setData(self.x, self.kk_field[:, j] * MM)
        self.figure_x.set_title("d2.section.x.title", y=self.y[j])

    def update_y(self):
        i = self.at_x.index()
        d2_curve, kk_curve, _ = self.curves[self.figure_y]
        d2_curve.setData(self.y, self.record["field"][i, :] * MM)
        if self.kk_field is not None:
            kk_curve.setData(self.y, self.kk_field[i, :] * MM)
        self.figure_y.set_title("d2.section.y.title", x=self.x[i])

    def _relegend(self, *_):
        lang = self.session.figure_language
        for d2_curve, kk_curve, legend in self.curves.values():
            legend.clear()
            legend.addItem(d2_curve, fig_text(lang, "legend.d2"))
            if self.kk_field is not None:
                legend.addItem(kk_curve, fig_text(lang, "legend.kkvim", N=self.outcome.request.N))
