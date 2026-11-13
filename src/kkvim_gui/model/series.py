"""Data of a load series for the «Нагрузка — прогиб» graph and its export (GUI-PLATE-WORKFLOW-UX-01). Qt-free.

``series_data`` turns a ``runs.SeriesRecord`` and its runs into an immutable ``SeriesData``. Every point is one real
KK-VIM run at its own q (``RunRecord.outcome``); nothing is scaled, interpolated or fitted:

- a point enters the KK-VIM curve only if its run ended as requested with a result field (field role "accepted" or
  "research", never "diagnostic") and the plate centre (a/2, b/2) is a node of its grid; the value is the stored
  field at that node (``benchmarks.centre_node``). Every other run stays in the table with the reason it is excluded;
- the node of the largest |w| is a separately labelled quantity (``RunOutcome.extremum``: the signed w at that node);
- the Navier reference (SSSS only) is the record of the series job: the reference evaluated at every q of the series in
  its own process (``backend.navier_job``); it is shown for every q, also where the KK-VIM run failed, and is never
  derived from the KK-VIM field. CCCC series have no Navier curve;
- D2 (CCCC with «МКР — двумерный (D2)» chosen, GUI-MULTIMETHOD-CCCC-D2-01) is the record of each run (``RunRecord.d2``):
  D2 computed in its own process at that q for the same plate and grid. A point is on the D2 curve only if that D2
  answered with a centre value; a failed, cancelled or not started D2 stays in the table with its state and reason.
  Timings of both methods are kept per point (one execution each: observations, never a ranking).
"""

from dataclasses import dataclass

from kkvim_gui.backend import benchmarks, d2_job, export, navier_job
from kkvim_gui.backend.outcome import ROLE_ACCEPTED, ROLE_RESEARCH

RUN_STATUS_TEXT = {"finished": "завершён", "error": "ошибка", "stopped": "остановлен", "running": "выполняется",
                   "not_run": "не выполнялся"}


@dataclass(frozen=True)
class SeriesPoint:
    index: int  # 1-based, in the order entered
    q: float  # Pa, exactly the q of the request
    run_id: int | None
    run_status: str  # "finished" | "error" | "stopped" | "running" | "not_run"
    solver_status: str | None
    field_role: str | None
    requested_completed: bool
    scientific_status: str
    mark: str | None
    included: bool  # the point is on the KK-VIM curve
    reason: str  # why it is not ("" if included)
    w_centre_m: float | None  # KK-VIM w at the centre node
    centre: dict | None  # i, j, x, y of the centre node
    w_at_max_abs_m: float | None  # signed w at the node of the largest |w| (eligible fields only)
    navier_state: str | None  # None: no Navier for this series (CCCC)
    navier_w_m: float | None
    delta_w_m: float | None  # w_KK-VIM - w_Navier at the centre
    rel_pct: float | None  # delta / w_Navier * 100
    d2_state: str | None = None  # None: D2 not requested for this series; otherwise the state of the D2 record
    d2_reason: str = ""
    d2_w_m: float | None = None  # D2 w at the centre node (its own run at this q)
    d2_beta_a: float | None = None  # w_c D / (q a^4) of D2
    d2_e_m: float | None = None  # e_D2 of that run
    d2_delta_w_m: float | None = None  # w_KK-VIM - w_D2 at the centre (both computed at this q)
    d2_rel_pct: float | None = None  # d2_delta_w_m / w_D2 * 100
    kkvim_solver_s: float | None = None  # KK-VIM: kkvim_adapter.execute in the computing process
    kkvim_wall_s: float | None = None  # KK-VIM: start of the process to the message read by the GUI
    d2_solve_s: float | None = None  # D2: the call r5_cccc_d2.solve in its process
    d2_wall_s: float | None = None  # D2: start of the process to the complete answer read

    @property
    def on_d2_curve(self) -> bool:
        return self.d2_w_m is not None


@dataclass(frozen=True)
class SeriesData:
    series_id: int
    status: str  # "running" | "finished" | "cancelled"
    started: str
    supports: str
    N: int
    route: str
    plate: dict  # a, b, t, E, nu (SI)
    grid: tuple  # (n_x, n_y)
    entered: tuple  # the entered (key, text, unit) of the first request without q
    points: tuple
    navier: dict | None  # job record of the reference (None for CCCC)
    d2: bool = False  # D2 requested for this series (CCCC with «МКР — двумерный (D2)»)
    d2_method: str = ""  # the method text of the D2 worker (from the first answered point)

    @property
    def navier_ok(self) -> bool:
        return self.navier is not None and self.navier.get("state") == navier_job.OK

    @property
    def included(self):
        return [p for p in self.points if p.included]

    @property
    def d2_included(self):
        return [p for p in self.points if p.on_d2_curve]

    @property
    def complete(self) -> bool:
        return self.status != "running"


def _d2_fields(run, requested):
    """The D2 columns of a point: state, centre value, its signed difference with KK-VIM is added by the caller."""
    if not requested:
        return {}
    d2 = None if run is None else getattr(run, "d2", None)
    if d2 is None:
        return {"d2_state": d2_job.NOT_RUN, "d2_reason": "расчёт D2 не выполнялся (точка серии не запускалась)"}
    timing = d2.get("timing") or {}
    ok = d2.get("state") == d2_job.OK
    w = d2.get("w_centre_m") if ok else None
    reason = d2.get("reason", "") if not ok else (d2.get("centre_reason", "") if w is None else "")
    return {"d2_state": d2.get("state"), "d2_reason": reason, "d2_w_m": w, "d2_beta_a": d2.get("beta_a") if ok else None,
            "d2_e_m": d2.get("e_D2_m") if ok else None, "d2_solve_s": timing.get("solve_s"),
            "d2_wall_s": timing.get("wall_s")}


def _point(index, request, run, navier, d2_requested=False):
    nav = None if navier is None else (navier_job.pending_record() if navier.get("state") == navier_job.PENDING
                                       else navier_job.point_record(navier, request))
    nav_state = None if nav is None else nav["state"]
    nav_w = nav["w_m"] if nav is not None and nav_state == navier_job.OK else None
    common = dict(index=index, q=request.q, navier_state=nav_state, navier_w_m=nav_w, **_d2_fields(run, d2_requested))
    if run is not None:
        timing = getattr(run, "timing", None) or {}
        common.update(kkvim_solver_s=timing.get("kkvim_solver_s"), kkvim_wall_s=timing.get("kkvim_wall_s"))
    if run is None:
        return SeriesPoint(run_id=None, run_status="not_run", solver_status=None, field_role=None, requested_completed=False,
                           scientific_status="", mark=None, included=False, reason="расчёт не выполнялся (серия остановлена)",
                           w_centre_m=None, centre=None, w_at_max_abs_m=None, delta_w_m=None, rel_pct=None, **common)
    o = run.outcome
    if o is None:
        reason = {"stopped": "остановлен пользователем", "running": "выполняется"}.get(run.status, "ошибка: результата нет")
        return SeriesPoint(run_id=run.run_id, run_status=run.status, solver_status=None, field_role=None,
                           requested_completed=False, scientific_status="", mark=None, included=False, reason=reason,
                           w_centre_m=None, centre=None, w_at_max_abs_m=None, delta_w_m=None, rel_pct=None, **common)
    eligible = (run.status == "finished" and o.requested_completed and o.field_role in (ROLE_ACCEPTED, ROLE_RESEARCH))
    centre = benchmarks.centre_node(o) if eligible else None
    if not eligible:
        reason = (f"поле только диагностическое ({o.status})" if o.diagnostic_only else
                  f"запрошенный расчёт не выполнен ({o.status})" if o.has_field else f"поле не получено ({o.status})")
    elif centre is None:
        reason = "центр (a/2, b/2) не является узлом сетки: значение не интерполируется"
    else:
        reason = ""
    w_c = None if centre is None else float(o.w[centre["i"], centre["j"]])
    w_max = o.extremum()["w"] if eligible else None
    delta = None if w_c is None or nav_w is None else w_c - nav_w
    d2_w = common.get("d2_w_m")
    if w_c is not None and d2_w is not None:
        common.update(d2_delta_w_m=w_c - d2_w, d2_rel_pct=(w_c - d2_w) / d2_w * 100.0)
    return SeriesPoint(run_id=run.run_id, run_status=run.status, solver_status=o.status, field_role=o.field_role,
                       requested_completed=o.requested_completed, scientific_status=export.scientific_status(o), mark=o.mark,
                       included=w_c is not None, reason=reason, w_centre_m=w_c, centre=centre, w_at_max_abs_m=w_max,
                       delta_w_m=delta, rel_pct=None if delta is None else delta / nav_w * 100.0, **common)


def series_data(series, manager) -> SeriesData:
    """The SeriesData of ``series`` (runs.SeriesRecord) with the runs of ``manager`` (runs.RunManager)."""
    first = series.requests[0]
    d2_requested = first.supports == "CCCC" and bool(getattr(getattr(series, "snapshot", None), "with_d2", False))
    points, method = [], ""
    for k, request in enumerate(series.requests):
        run = manager.get(series.run_ids[k]) if k < len(series.run_ids) else None
        points.append(_point(k + 1, request, run, series.navier, d2_requested))
        d2 = getattr(run, "d2", None)
        if not method and d2 is not None and d2.get("state") == d2_job.OK:
            method = d2.get("method", "")
    entered = tuple(e for e in first.entered if e[0] != "q")
    return SeriesData(series_id=series.series_id, status=series.status, started=series.started.isoformat(timespec="seconds"),
                      supports=first.supports, N=first.N, route=first.route,
                      plate={"a": first.a, "b": first.b, "t": first.t, "E": first.E, "nu": first.nu},
                      grid=(first.n_x, first.n_y), entered=entered, points=tuple(points), navier=series.navier,
                      d2=d2_requested, d2_method=method)
