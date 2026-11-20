"""The «Сравнение с эталоном» tab of the analysis window: one solver run next to independent references (GUI_DESIGN.md, 6).

All numbers come from ``backend.benchmarks.compare`` (the run arrays, the packaged reference data, for SSSS the Navier
reference and for CCCC with «МКР — двумерный (D2)» the D2 solution computed for the run's own inputs in a separate
process); this module only lays them out. A row computed for this run and a row of stored data are titled apart. The tab
keeps four things apart: the solver route and its scientific status, the numerical stopping result, the independent
reference, and the observed signed difference (which does not by itself establish the accuracy of the method).
Values are shown at full precision (Python repr of the binary64 number): the comparison is a set of
signed differences, never a verdict. Without an exact reference case, for a diagnostic field or with damaged data the
tab states why and shows no comparison numbers.
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QScrollArea, QVBoxLayout, QWidget

from kkvim_gui.analysis import results
from kkvim_gui.backend import benchmarks
from kkvim_gui.widgets.common import Card, Note, StatusLabel, label

HEADERS = benchmarks.HEADERS  # one table layout for this tab and report.html
STATE_KIND = {benchmarks.AVAILABLE: "neutral", benchmarks.UNAVAILABLE: "warning", benchmarks.DIAGNOSTIC: "warning",
              benchmarks.DATA_ERROR: "error", benchmarks.NO_RUN: "neutral", benchmarks.PENDING: "neutral"}
WHAT_IS_COMPARED = (
    ("Маршрут решателя", "какой решатель КК-ВИМ выполнил расчёт и его научный статус (ниже, «Текущий расчёт»)"),
    ("Итог остановки", "как закончились итерации; выполнение критерия остановки не доказывает точности"),
    ("Независимый эталон", "решение другим методом, полученное без КК-ВИМ и без его поля"),
    ("Разность", "знаковая разность КК-ВИМ − эталон; утверждённого порога согласия нет, это наблюдение, не оценка точности"),
)
STATE_TEXT = {state: (kind, benchmarks.STATE_TITLE[state]) for state, kind in STATE_KIND.items()}
full = benchmarks.full  # full-precision text of a number (shortest repr that round-trips the binary64 value)


class ComparisonTab(QWidget):
    def __init__(self, outcome, run=None, text=None, navier=None, d2=None, parent=None):
        """text: benchmark data instead of the packaged file (tests of damaged data); navier: the point record of the
        Navier reference computed for this SSSS run (``backend.navier_job``), "pending" while it is computed; d2: the D2
        record computed for this CCCC run (``backend.d2_job``)."""
        super().__init__(parent)
        self.outcome = outcome
        self.comparison = benchmarks.compare(outcome, text, navier=navier, d2=d2)
        c = self.comparison
        body = QVBoxLayout()
        guide = Card("Что сравнивается")
        guide.body.addLayout(results.key_value(WHAT_IS_COMPARED))
        body.addWidget(guide)
        kind, _ = STATE_TEXT[c.state]
        self.state_label = StatusLabel(kind, benchmarks.state_title(c))
        self.state_label.setWordWrap(True)
        head = Card("Состояние сравнения")
        head.body.addWidget(self.state_label)
        if c.reason:
            self.reason = label(c.reason, wrap=True)
            self.reason.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            head.body.addWidget(self.reason)
        body.addWidget(head)

        run_card = Card("Текущий расчёт КК-ВИМ", right=label(outcome.mark or "", "tag") if outcome is not None else None)
        if outcome is not None:
            for s in results.status_labels(outcome, run):
                run_card.body.addWidget(s)
            run_card.body.addLayout(results.key_value(self._run_rows()))
        body.addWidget(run_card)

        self.table = None
        if c.has_numbers:
            rows = c.rows
            model = results.RowsModel(HEADERS, len(rows), lambda i: benchmarks.cells(rows[i]), tooltip=self._tooltip)
            self.table = results.table_view(model)
            self.table.setMinimumHeight(40 + 34 * len(rows))
            refs = Card("Независимый эталон, рассчитанный для этих входных данных" if c.live else
                        "Рассчитанный для этих входных данных D2 и сохранённые эталоны (сохранённые не пересчитываются)"
                        if c.executed else "Эталоны (сохранённые данные, не пересчитываются)")
            refs.body.addWidget(self.table)
            for kind_, reason in c.missing:
                refs.body.addWidget(Note(f"{benchmarks.KIND_TITLE[kind_]}: {reason}.", "info"))
            body.addWidget(refs)
            notes = Card("Как читать сравнение")
            for n in c.notes:
                notes.body.addWidget(Note(n, "info"))
            body.addWidget(notes)
        body.addStretch(1)
        content = QWidget()
        content.setObjectName("Page")
        lay = results.padded(body)
        inner = QVBoxLayout(content)
        inner.setContentsMargins(0, 0, 0, 0)
        inner.addWidget(lay)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setWidget(content)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    def _tooltip(self, i, j):
        """Full text of a cell: the provenance column is cut by the window width, its tooltip is not (read-only)."""
        row = self.comparison.rows[i]
        text = benchmarks.cells(row)[j]
        return f"{row.reference.title}\n{text}" if j == benchmarks.DETAIL_COLUMN else text

    def _run_rows(self):
        o, c = self.outcome, self.comparison
        r = o.request
        rows = [
            ("Закрепление, N, полушаги", f"{r.supports} · N = {r.N} · s = {o.s}"
                                         + (f" (запрошено {o.s_requested})" if o.s_requested else "")),
            ("Вход в единицах СИ", f"a = {r.a!r} m · b = {r.b!r} m · t = {r.t!r} m · E = {r.E!r} Pa · ν = {r.nu!r} · "
                                   f"q = {r.q!r} Pa"),
            ("Сетка МКР КК-ВИМ", f"n_x = {r.n_x}, n_y = {r.n_y} интервалов"),
            ("Статус решателя", f"{o.status} · {o.solver}: {o.call}"),
            ("Роль поля", results.FIELD_ROLE_TEXT.get(o.field_role, "поле не получено")),
        ]
        if c.state == benchmarks.AVAILABLE:
            ce = c.centre
            if c.navier is not None and c.live:
                nav = c.navier
                rows += [("Эталон Навье — w в центре", f"{full(nav['w_m'])} m (M = {nav['M']}, отдельный процесс)"),
                         ("Разность Δw = w_КК-ВИМ − w_Навье", f"{full(c.rows[0].delta_w_m)} m"),
                         ("Относительная разность Δβ / β_Навье", f"{full(c.rows[0].rel_pct)} %")]
            d2_row = next((x for x in c.rows if x.reference.kind == "D2_live"), None)
            if d2_row is not None:
                rows += [("МКР D2 — w в центре", f"{full(c.d2['w_centre_m'])} m (рассчитан в отдельном процессе)"),
                         ("Разность Δw = w_КК-ВИМ − w_D2", f"{full(d2_row.delta_w_m)} m"),
                         ("Относительная разность Δβ / β_D2", f"{full(d2_row.rel_pct)} %")]
            rows += [
                ("Узел центра", f"i = {ce['i']}, j = {ce['j']}: x = {ce['x']!r} m, y = {ce['y']!r} m (точно a/2, b/2)"),
                ("Прогиб в центре w_c", f"{full(c.w_centre_m)} m"),
                ("Жёсткость D (kkvim)", f"{full(c.D)} N·m"),
                ("β = w_c·D/(q·a⁴)", full(c.beta)),
                ("w̄ решателя в этом узле", f"{full(c.w_bar_solver)} (MATH_SPEC 2.5, контроль)"),
            ]
        return rows
