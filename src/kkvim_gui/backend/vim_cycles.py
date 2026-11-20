"""VIM-1, VIM-2, VIM-3: completed full cycles of ONE KK-VIM computation (GUI-VIM-CYCLES-123-01). Qt-free.

The labels are display labels requested by the project owner (2026-11-16) for the states after the completed full cycles
k = 1, 2, 3 of a single run (D-030; MATH_SPEC 0.3, 4.1, 4.5):

- VIM-1 = k = 1, s = 2 (after Y¹ and X²); VIM-2 = k = 2, s = 4 (after Y³, X⁴); VIM-3 = k = 3, s = 6 (after Y⁵, X⁶).

They are neither separate methods nor N = 1, 2, 3 (N is the number of separated terms, D-003), and they are not author
nomenclature. The initial state s = 1 (k = 0, Y¹ only) is a partial cycle and is never labelled VIM-1.

Values are read from the history the solver already returned (``RunOutcome.history``); nothing is computed again, scaled
from another load or taken from the final field:

- SSSS, N = 1 (R3): ``w_bar_centre`` of the row s = 2k (``kkvim.rectangle_run.halfstep_report``, w̄ = w D / (q a⁴),
  MATH_SPEC 2.5) and w_c = w̄ · q · a⁴ / D;
- CCCC (research routes): ``alpha_centre`` of the row s = 2k (w D / (q min(a, b)⁴) at the centre node, computed by the
  solver) and w_c = α · q · min(a, b)⁴ / D;
- other routes record no centre value per half-step: the cycles are unavailable (stated, never substituted).

A value exists only if the solver completed the half-step s = 2k and recorded a centre value (the centre (a/2, b/2) is a
grid node); it is drawn as a result only if the run ended as requested with a result field (not diagnostic). The
reference is the independent one already associated with the run: Navier for SSSS (``RunRecord.navier``), D2 for CCCC
(``RunRecord.d2``). Differences are signed, Δw = w_KK-VIM − w_ref and 100 Δw / w_ref; no agreement threshold exists, and
a later cycle is not claimed to be more accurate than an earlier one.
"""

from dataclasses import dataclass

from kkvim_gui.backend import export
from kkvim_gui.backend.outcome import ROLE_ACCEPTED, ROLE_RESEARCH
from kkvim_gui.backend.request import CCCC_HALF_STEPS

K_MAX = 3  # the owner's VIM-1 … VIM-3
OWNER_NOTE = ("VIM-1, VIM-2, VIM-3 — обозначения интерфейса (решение владельца, 2026-11-16) для состояний после "
              "завершённых полных циклов k = 1, 2, 3 одного расчёта КК-ВИМ (s = 2k, D-030): не отдельные методы, не N = 1, "
              "2, 3 (N — число членов разделённого представления) и не обозначения авторов метода")
# history key of the centre value and the reference length of its normalisation, by solver
SOURCES = {"R3": ("w_bar_centre", "a"), "CCCC-N1": ("alpha_centre", "min(a, b)"),
           "CCCC-N23": ("alpha_centre", "min(a, b)"), "CCCC-N46": ("alpha_centre", "min(a, b)")}
NAVIER, D2 = "Navier", "D2"
REFERENCE_TITLE = {NAVIER: "Навье (независимый эталон, рассчитан для этих входных данных)",
                   D2: "МКР — двумерный (D2), рассчитан для этих входных данных"}


def label(k: int) -> str:
    return f"VIM-{k}"


def cycle_text(k: int) -> str:
    """«VIM-k (k = …, s = …: Y^{2k−1}, X^{2k})»."""
    return f"{label(k)} (k = {k}, s = {2 * k}: Y{_sup(2 * k - 1)}, X{_sup(2 * k)})"


def _sup(n: int) -> str:
    return str(n).translate(str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹"))


def termination_text(s: int) -> str:
    """What the completed half-steps mean in cycles (D-030): s = 7 is three full cycles and one Y half-step."""
    k = s // 2
    if s == 0:
        return "s = 0: ни одного полушага"
    if s == 1:
        return "s = 1: только Y¹, полного цикла нет (k = 0)"
    cycles = f"k = {k} {_cycles_word(k)}"
    if s % 2:
        return f"s = {s}: {cycles} и ещё один Y-полушаг (Y{_sup(s)})"
    return f"s = {s}: {cycles}, последний полушаг — X{_sup(s)}"


def _cycles_word(k: int) -> str:
    if k % 10 == 1 and k % 100 != 11:
        return "полный цикл"
    if k % 10 in (2, 3, 4) and k % 100 not in (12, 13, 14):
        return "полных цикла"
    return "полных циклов"


@dataclass(frozen=True)
class CycleValue:
    k: int  # 0 for the partial state s = 1
    s: int
    label: str  # "VIM-k", or "s = 1 (k = 0)" for the partial state
    direction: str | None  # the half-step that produced the state ("X" for s = 2k)
    available: bool  # the solver completed s and recorded the centre value
    on_graph: bool  # available and the run ended as requested with a result field
    w_m: float | None  # centre deflection, m (from the stored normalised value)
    w_norm: float | None  # the stored normalised value (w̄ or α)
    reason: str  # why not available / not drawn ("" otherwise)
    ref_w_m: float | None = None
    delta_w_m: float | None = None  # w_m - ref_w_m
    delta_pct: float | None = None  # 100 delta / ref_w_m


@dataclass(frozen=True)
class CycleTable:
    supported: bool  # the route records centre values per half-step
    route_reason: str
    key: str | None  # history key of the centre value
    norm: str | None  # reference length of the normalisation: "a" or "min(a, b)"
    q: float
    N: int
    solver: str
    solver_status: str
    s: int
    k: int
    stop_kind: str
    stop_text: str
    criterion_met: bool  # the approved stopping criterion was met (R3 CONVERGED); fixed half-steps are not
    requested_completed: bool
    field_role: str | None
    scientific_status: str
    mark: str | None
    parameters: tuple
    last_changes: tuple  # (s, eps_inf, eps_E) of the last two rows that have them
    reference: str | None  # NAVIER, D2 or None
    reference_state: str | None
    reference_reason: str
    ref_w_m: float | None
    partial: CycleValue | None  # s = 1
    cycles: tuple  # CycleValue for k = 1 … K_MAX

    @property
    def termination(self) -> str:
        return termination_text(self.s)

    @property
    def drawn(self):
        return [c for c in self.cycles if c.on_graph]

    @property
    def provenance(self) -> str:
        if not self.supported:
            return ""
        what = "w̄ в центре = w D / (q a⁴)" if self.key == "w_bar_centre" else "α в центре = w D / (q min(a, b)⁴)"
        return (f"история полушагов этого расчёта ({self.solver}, {what}); w_c = значение · q · {self.norm}⁴ / D; "
                "решатель заново не вызывался")


def _reference(outcome, navier, d2):
    """(kind, state, reason, w_m) of the independent reference associated with the run."""
    supports = outcome.request.supports
    if supports == "SSSS":
        if navier is None:
            return NAVIER, None, "эталон Навье для этого расчёта не рассчитывался", None
        state = navier.get("state")
        w = navier.get("w_m") if state == "ok" else None
        return NAVIER, state, "" if w is not None else (navier.get("reason") or str(state)), w
    if supports == "CCCC":
        if d2 is None:
            return D2, None, "МКР D2 для этого расчёта не выбран (флажок «МКР — двумерный (D2)»)", None
        state = d2.get("state")
        w = d2.get("w_centre_m") if state == "ok" else None
        reason = "" if w is not None else (d2.get("centre_reason") or d2.get("reason") or str(state))
        return D2, state, reason, w
    return None, None, f"независимого эталона для {supports} нет", None


def cycle_table(outcome, navier=None, d2=None) -> CycleTable:
    """The VIM-1 … VIM-3 states of ``outcome`` with the reference of its run (``navier``/``d2`` records, or None)."""
    r = outcome.request
    source = SOURCES.get(outcome.solver)
    key, norm = source if source else (None, None)
    length = None if norm is None else (r.a if norm == "a" else min(r.a, r.b))
    ref_kind, ref_state, ref_reason, ref_w = _reference(outcome, navier, d2)
    rows = {row.get("s"): row for row in outcome.history}
    result_field = outcome.requested_completed and outcome.field_role in (ROLE_ACCEPTED, ROLE_RESEARCH)
    centre_node = r.n_x % 2 == 0 and r.n_y % 2 == 0
    route_reason = "" if source else (f"маршрут {outcome.solver} не записывает прогиб в центре на каждом полушаге "
                                      "(в истории только max|w|): VIM-1…VIM-3 не показываются, расчёт заново не выполняется")

    def state(k, s):
        name = label(k) if k else "s = 1 (k = 0)"
        direction = "X" if s % 2 == 0 else "Y"
        if not source:
            return CycleValue(k, s, name, direction, False, False, None, None, route_reason)
        row = rows.get(s)
        limit = max(CCCC_HALF_STEPS.get(outcome.route, ()), default=None)
        if row is None and limit is not None and s > limit:
            return CycleValue(k, s, name, direction, False, False, None, None,
                              f"маршрут {outcome.solver} (N = {r.N}) допускает не более s = {limit}"
                              + (" (защиты RN-5/RT-7; s ≥ 3 не предлагается)" if r.N >= 2 else "")
                              + f": состояние s = {s} не вычисляется и не подставляется")
        if row is None:
            return CycleValue(k, s, name, direction, False, False, None, None,
                              f"полушаг s = {s} не выполнялся: расчёт закончился на {termination_text(outcome.s)} "
                              f"({outcome.status}); более позднее поле не подставляется")
        value = row.get(key)
        if value is None:
            why = ("центр (a/2, b/2) не является узлом сетки (n_x или n_y нечётно): значение не интерполируется"
                   if not centre_node else f"решатель не записал значение в центре на полушаге s = {s}")
            return CycleValue(k, s, name, direction, False, False, None, None, why)
        w = value * r.q * length ** 4 / outcome.D
        reason = "" if result_field else (f"расчёт не завершён как запрошено ({outcome.status}): значение полушага — только "
                                          "диагностика, на графике не показывается")
        delta = None if ref_w is None else w - ref_w
        drawn = result_field and k >= 1  # the partial state s = 1 is a table row only, never a VIM point
        return CycleValue(k, s, name, direction, True, drawn, w, value, reason, ref_w, delta,
                          None if delta is None else delta / ref_w * 100.0)

    changes = [(row["s"], row.get("eps_inf"), row.get("eps_E")) for row in outcome.history
               if row.get("eps_inf") is not None or row.get("eps_E") is not None][-2:]
    return CycleTable(
        supported=source is not None, route_reason=route_reason, key=key, norm=norm, q=r.q, N=r.N, solver=outcome.solver,
        solver_status=outcome.status, s=outcome.s, k=outcome.k, stop_kind=outcome.stop_kind, stop_text=outcome.stop_text,
        criterion_met=outcome.stop_kind == "met", requested_completed=outcome.requested_completed,
        field_role=outcome.field_role, scientific_status=export.scientific_status(outcome), mark=outcome.mark,
        parameters=tuple(outcome.parameters), last_changes=tuple(changes), reference=ref_kind, reference_state=ref_state,
        reference_reason=ref_reason, ref_w_m=ref_w, partial=state(0, 1), cycles=tuple(state(k, 2 * k) for k in
                                                                                       range(1, K_MAX + 1)))


# ---- CSV ---------------------------------------------------------------------------------------------------------

CSV_FILE = "vim_cycles.csv"
SERIES_CSV_FILE = "load_series_vim_cycles.csv"
COLUMNS = (  # (name, unit)
    ("run_id", ""), ("q_Pa", "Pa"), ("N", ""), ("label", ""), ("k", ""), ("s", ""), ("half_step", ""),
    ("available", ""), ("on_graph", ""), ("w_KKVIM_m", "m"), ("w_normalised", "1"), ("normalisation", ""),
    ("reference_method", ""), ("reference_state", ""), ("w_Navier_m", "m"), ("w_D2_m", "m"), ("delta_w_m", "m"),
    ("delta_percent", "%"), ("solver_status", ""), ("s_completed", ""), ("k_completed", ""), ("criterion_met", ""),
    ("requested_run_completed", ""), ("field_role", ""), ("scientific_status", ""), ("provenance", ""), ("reason", ""),
)
DEFINITIONS = (
    f"# {OWNER_NOTE}",
    "# VIM-k: k = s/2 completed full cycles (Y^(2k-1), X^(2k)); the row s = 1 (k = 0) is the partial initial state",
    "# w_KKVIM_m: stored centre value of the half-step s of this run times q * L^4 / D (L = a for SSSS N = 1, w_bar = w D / "
    "(q a^4); L = min(a, b) for CCCC, alpha = w D / (q min(a, b)^4)); never the final field, never scaled from another q",
    "# delta_w_m = w_KKVIM_m - w_reference; delta_percent = 100 * delta_w_m / w_reference; signed differences only, no "
    "agreement threshold; on_graph = available and the run ended as requested with a result field (not diagnostic)",
)


def rows(table: CycleTable, run_id) -> list:
    out = []
    for c in (table.partial, *table.cycles):
        out.append((run_id, table.q, table.N, c.label, c.k, c.s, c.direction, c.available, c.on_graph, c.w_m, c.w_norm,
                    None if table.key is None else f"{table.key} (L = {table.norm})", table.reference or "none",
                    table.reference_state or "", c.ref_w_m if table.reference == NAVIER else None,
                    c.ref_w_m if table.reference == D2 else None, c.delta_w_m, c.delta_pct, table.solver_status, table.s,
                    table.k, table.criterion_met, table.requested_completed, table.field_role, table.scientific_status,
                    table.provenance, c.reason))
    return out


def header() -> list:
    return [f"{name} [{unit}]" if unit else name for name, unit in COLUMNS]


def write_csv(path, tables, comments=()) -> None:
    """``tables``: iterable of (run_id, CycleTable)."""
    from pathlib import Path

    body = [row for run_id, table in tables for row in rows(table, run_id)]
    export._write_csv(Path(path), [*comments, *DEFINITIONS], header(), body)


def plain(table: CycleTable, run_id=None) -> dict:
    """JSON record of a table (manifest)."""
    return export._plain({
        "labels": OWNER_NOTE, "run_id": run_id, "supported": table.supported, "route_reason": table.route_reason,
        "solver": table.solver, "solver_status": table.solver_status, "s": table.s, "k": table.k,
        "termination": table.termination, "criterion_met": table.criterion_met,
        "requested_run_completed": table.requested_completed, "reference": table.reference,
        "reference_state": table.reference_state, "reference_w_m": table.ref_w_m, "provenance": table.provenance,
        "states": [dict(zip((n for n, _ in COLUMNS), r, strict=True)) for r in rows(table, run_id)],
    })
