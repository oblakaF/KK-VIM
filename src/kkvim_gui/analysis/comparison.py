"""The "Сравнение" tab of the analysis window: one solver run next to saved independent references (GUI_DESIGN.md, 6).

All numbers come from ``backend.benchmarks.compare`` (the run arrays and the packaged reference data); this module only
lays them out. Values are shown at full precision (Python repr of the binary64 number): the comparison is a set of
signed differences, never a verdict. Without an exact reference case, for a diagnostic field or with damaged data the
tab states why and shows no comparison numbers.
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QScrollArea, QVBoxLayout, QWidget

from kkvim_gui.analysis import results
from kkvim_gui.backend import benchmarks
from kkvim_gui.widgets.common import Card, Note, StatusLabel, label

HEADERS = ("Эталон", "β эталона", "w эталона в центре, m", "Δβ = β_КК-ВИМ − β_эталона", "Δβ / β_эталона, %",
           "Метод, сетка, происхождение")
STATE_TEXT = {
    benchmarks.AVAILABLE: ("neutral", "Сравнение с сохранёнными эталонами: знаковые разности, без порога согласия"),
    benchmarks.UNAVAILABLE: ("warning", "Эталонные данные для этого расчёта недоступны"),
    benchmarks.DIAGNOSTIC: ("warning", "Сравнение не выполняется: поле диагностическое"),
    benchmarks.DATA_ERROR: ("error", "Данные эталонов повреждены или отсутствуют"),
    benchmarks.NO_RUN: ("neutral", "Нет результата решателя"),
}


def full(value):
    """Full-precision text of a number (shortest repr that round-trips the binary64 value)."""
    return "—" if value is None else repr(float(value))


class ComparisonTab(QWidget):
    def __init__(self, outcome, run=None, text=None, parent=None):
        """text: benchmark data instead of the packaged file (tests of damaged data)."""
        super().__init__(parent)
        self.outcome = outcome
        self.comparison = benchmarks.compare(outcome, text)
        c = self.comparison
        body = QVBoxLayout()
        kind, title = STATE_TEXT[c.state]
        self.state_label = StatusLabel(kind, title)
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
            for k, t in results.statuses(outcome, run):
                s = StatusLabel(k, t)
                s.setWordWrap(True)
                run_card.body.addWidget(s)
            run_card.body.addLayout(results.key_value(self._run_rows()))
        body.addWidget(run_card)

        self.table = None
        if c.has_numbers:
            rows = c.rows
            model = results.RowsModel(HEADERS, len(rows), lambda i: (
                rows[i].reference.title, full(rows[i].reference.beta), full(rows[i].reference.w_centre_m),
                full(rows[i].delta), full(rows[i].rel_pct), rows[i].reference.detail))
            self.table = results.table_view(model)
            self.table.setMinimumHeight(40 + 34 * len(rows))
            refs = Card("Эталоны (сохранённые данные, не пересчитываются)")
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
            rows += [
                ("Узел центра", f"i = {ce['i']}, j = {ce['j']}: x = {ce['x']!r} m, y = {ce['y']!r} m (точно a/2, b/2)"),
                ("Прогиб в центре w_c", f"{full(c.w_centre_m)} m"),
                ("Жёсткость D (kkvim)", f"{full(c.D)} N·m"),
                ("β = w_c·D/(q·a⁴)", full(c.beta)),
                ("w̄ решателя в этом узле", f"{full(c.w_bar_solver)} (MATH_SPEC 2.5, контроль)"),
            ]
        return rows
