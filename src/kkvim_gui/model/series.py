"""Data of a load series for the «Нагрузка — прогиб» graph and its export (GUI-PLATE-WORKFLOW-UX-01). Qt-free.

``series_data`` turns a ``runs.SeriesRecord`` and its runs into an immutable ``SeriesData``. Every point is one real
KK-VIM run at its own q (``RunRecord.outcome``); nothing is scaled, interpolated or fitted:

- a point enters the KK-VIM curve only if its run ended as requested with a result field (field role "accepted" or
  "research", never "diagnostic") and the plate centre (a/2, b/2) is a node of its grid; the value is the stored
  field at that node (``benchmarks.centre_node``). Every other run stays in the table with the reason it is excluded;
- the node of the largest |w| is a separately labelled quantity (``RunOutcome.extremum``: the signed w at that node);
- the Navier reference (SSSS only) is the record of the series job: the reference evaluated at every q of the series in
  its own process (``backend.navier_job``); it is shown for every q, also where the KK-VIM run failed, and is never
  derived from the KK-VIM field. CCCC series have no Navier curve.
"""

from dataclasses import dataclass

from kkvim_gui.backend import benchmarks, export, navier_job
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

    @property
    def navier_ok(self) -> bool:
        return self.navier is not None and self.navier.get("state") == navier_job.OK

    @property
    def included(self):
        return [p for p in self.points if p.included]

    @property
    def complete(self) -> bool:
        return self.status != "running"


def _point(index, request, run, navier):
    nav = None if navier is None else (navier_job.pending_record() if navier.get("state") == navier_job.PENDING
                                       else navier_job.point_record(navier, request))
    nav_state = None if nav is None else nav["state"]
    nav_w = nav["w_m"] if nav is not None and nav_state == navier_job.OK else None
    common = dict(index=index, q=request.q, navier_state=nav_state, navier_w_m=nav_w)
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
    return SeriesPoint(run_id=run.run_id, run_status=run.status, solver_status=o.status, field_role=o.field_role,
                       requested_completed=o.requested_completed, scientific_status=export.scientific_status(o), mark=o.mark,
                       included=w_c is not None, reason=reason, w_centre_m=w_c, centre=centre, w_at_max_abs_m=w_max,
                       delta_w_m=delta, rel_pct=None if delta is None else delta / nav_w * 100.0, **common)


def series_data(series, manager) -> SeriesData:
    """The SeriesData of ``series`` (runs.SeriesRecord) with the runs of ``manager`` (runs.RunManager)."""
    first = series.requests[0]
    points = []
    for k, request in enumerate(series.requests):
        run = manager.get(series.run_ids[k]) if k < len(series.run_ids) else None
        points.append(_point(k + 1, request, run, series.navier))
    entered = tuple(e for e in first.entered if e[0] != "q")
    return SeriesData(series_id=series.series_id, status=series.status, started=series.started.isoformat(timespec="seconds"),
                      supports=first.supports, N=first.N, route=first.route,
                      plate={"a": first.a, "b": first.b, "t": first.t, "E": first.E, "nu": first.nu},
                      grid=(first.n_x, first.n_y), entered=entered, points=tuple(points), navier=series.navier)
