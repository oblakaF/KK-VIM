"""Run dispatcher model: runs of the current session (GUI_DESIGN.md, 5.3, 3.6).

Two backends:
- the solver (default): ``backend.process.SolverProcess`` runs ``backend.kkvim_adapter.execute`` in a
  separate process; the main window is never blocked, a crash of the computation becomes the status
  «Ошибка», «Стоп» terminates the process. The solver reports no intermediate half-steps, so s and
  k are shown when the run ends;
- the demonstration backend (only with ``--demo``): synthetic stages, marked TEST DATA, never a result.

At most one run is active; a second start while a run is active is refused.

Independent Navier reference (SSSS): with a run of SSSS supports the manager starts ``backend.navier_job.NavierJob`` —
a separate process with the physical input only (never the KK-VIM result) — and keeps its point record in
``RunRecord.navier`` ("pending" until it answers). The KK-VIM result is never changed by it.

Load series (``start_series``): the same setup with several loads q, run one after another, each as an ordinary
solver run of the dispatcher (its own process, check, outcome and statuses); only q differs between the requests.
Every request is checked by ``check_request`` before the first run starts; a rejected point refuses the series.
«Стоп» on a run of a series stops that run and cancels the points not yet started. The Navier reference of an SSSS
series is one job for all loads (each load evaluated by the reference itself). Each run has its own
timer. Results of a finished solver run are kept as a ``RunOutcome`` (immutable copies of the solver
arrays). The protocol of every solver run (manifest.json and history.csv) is written to
``protocol_dir`` when it is set. KK-VIM counters follow D-030: s — completed coordinate half-steps;
k = floor(s / 2) — completed full cycles; N — number of terms (D-003).

Independent D2 (CCCC, GUI-MULTIMETHOD-CCCC-D2-01): with «МКР — двумерный (D2)» chosen in the setup (``with_d2``) a CCCC
run computes, after its KK-VIM process has ended, the QA reference D2 of exactly the same plate and grid in a separate
process (``backend.d2_job.D2Job``; the physical input only, never the KK-VIM result). The run stays active until D2 has
answered, so the two computations never run at the same time (their timings are not measured under each other's load)
and a load series advances only after the D2 of its point. «Стоп» while D2 runs stops D2 only: the KK-VIM result is kept,
the D2 record says "cancelled" and the series is cancelled as before. D2 is never started after a crash, a rejected
input or a stop of the KK-VIM process; its record then says why ("not_run"). Timings: ``RunRecord.timing`` (KK-VIM:
``kkvim_import_s``, ``kkvim_solver_s``, ``kkvim_wall_s``) and ``RunRecord.d2["timing"]`` (``backend.d2_job``).
"""

import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QObject, QTimer, Signal

from kkvim_gui import placeholders
from kkvim_gui.backend import d2_job, export, navier_job
from kkvim_gui.backend.kkvim_adapter import (
    RESEARCH_MARK,
    ROUTES,
    AdapterInputError,
    check_request,
)
from kkvim_gui.backend.outcome import (
    ORIGIN_DEMO,
    ORIGIN_SOLVER,
    ROLE_ACCEPTED,
    ROLE_DIAGNOSTIC,
    ROLE_RESEARCH,
)
from kkvim_gui.backend.process import CRASH, INPUT, OK, SolverProcess
from kkvim_gui.backend.request import ROUTE_R3, ROUTE_R4
from kkvim_gui.backend.units import base_unit

STATUS_TEXT = {
    "running": "Выполняется",
    "finished": "Завершён",
    "stopped": "Остановлен пользователем",
    "error": "Ошибка",
}
ITERATIVE_METHODS = {"КК-ВИМ"}  # future methods (МКР, МКЭ, Навье) are not iterative here

# Demonstration schedule (seconds of real time); synthetic, TEST DATA.
DEMO_PREPARE = 0.6
DEMO_HALF_STEP = 0.4
DEMO_FINISH = 0.3
DEMO_K_STOP = 8


def format_duration(seconds):
    """Whole seconds in Russian short form: "5 с", "1 мин 05 с", "1 ч 02 мин"."""
    s = int(seconds)
    if s < 60:
        return f"{s} с"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m} мин {s:02d} с"
    h, m = divmod(m, 60)
    return f"{h} ч {m:02d} мин"


@dataclass
class RunRecord:
    run_id: int
    started: datetime
    method: str
    snapshot: object                  # SetupState at the start (never changes)
    project: str = placeholders.PROJECT_NAME
    status: str = "running"
    stop_criterion: str = ""          # stop kind of the outcome ("met", "not_met", "error", "none"); "" while running
    stage: str = ""
    s: int | None = None              # completed coordinate half-steps (iterative methods)
    k_stop: int | None = None         # demonstration only
    data_origin: str = ORIGIN_DEMO
    request: object = None            # RunRequest (solver runs)
    outcome: object = None            # RunOutcome of a solver run that returned
    protocol_path: Path | None = None
    important: list = field(default_factory=list)   # (time, level, text)
    detailed: list = field(default_factory=list)
    navier: dict | None = None        # point record of the Navier reference (SSSS; backend.navier_job), None otherwise
    series_id: int | None = None      # the load series of this run, if any
    d2_requested: bool = False        # CCCC with «МКР — двумерный (D2)» chosen: D2 runs after the KK-VIM process
    d2: dict | None = None            # D2 record (backend.d2_job.d2_record), None if D2 was not requested
    timing: dict = field(default_factory=dict)  # KK-VIM timings, s: kkvim_import_s, kkvim_solver_s, kkvim_wall_s
    _t0: float = 0.0
    _elapsed_final: float | None = None
    _process: object = None
    _d2_job: object = None

    @property
    def iterative(self):
        return self.method in ITERATIVE_METHODS

    @property
    def k(self):
        return None if self.s is None else self.s // 2

    @property
    def active(self):
        return self.status == "running"

    @property
    def is_demo(self):
        return self.data_origin == ORIGIN_DEMO

    @property
    def has_results(self):
        """Analysis: finished runs whose field is a result of its route (accepted or research; demo: finished runs).

        Never stopped or failed runs, and never a diagnostic field (``RunOutcome.field_role``)."""
        if self.status != "finished":
            return False
        return self.is_demo or (self.outcome is not None and self.outcome.field_role in (ROLE_ACCEPTED, ROLE_RESEARCH))

    @property
    def has_diagnostic_field(self):
        """A run that did not end as requested and kept the field of its last half-step: diagnostics only."""
        return self.status in ("finished", "error") and self.outcome is not None and self.outcome.field_role == ROLE_DIAGNOSTIC

    def elapsed(self, now):
        return self._elapsed_final if self._elapsed_final is not None else max(0.0, now - self._t0)

    def params_text(self):
        st = self.snapshot
        grid = f"сетка {st.texts['n_x'].strip()}×{st.texts['n_y'].strip()}"
        if self.is_demo:
            return f"N = {st.N} · {grid} · демонстрация"
        route = st.route
        if route == ROUTE_R3:
            return f"N = 1 · R3 · {grid}" + (" · строгий режим" if st.strict else "")
        if route == ROUTE_R4:
            return f"N = {st.N} · R4 {st.r4_mode}, {st.transform} · {grid}"
        if st.supports == "CCCC":
            d2 = " · + МКР D2" if self.d2_requested else ""
            return f"CCCC · N = {st.N} · s = {st.cccc_half_steps} · исследование · {grid}{d2}"
        return f"SSSS · N = {st.N} · исследование, s = {st.texts['half_steps'].strip()} · {grid}"

    def log(self, level, text, important=False):
        entry = (datetime.now().strftime("%H:%M:%S"), level, text)
        self.detailed.append(entry)
        if important:
            self.important.append(entry)

    def meta(self, now):
        meta = {"run_id": self.run_id, "started": self.started.isoformat(timespec="seconds"),
                "execution_status": self.status, "elapsed_s": round(self.elapsed(now), 3),
                "data_origin": self.data_origin}
        if self.series_id is not None:
            meta["series_id"] = self.series_id
        if self.timing:
            meta["timing_s"] = dict(self.timing)
        return meta


SERIES_MAX = 20  # points of one load series
SERIES_STATUS_TEXT = {"running": "Выполняется", "finished": "Завершена", "cancelled": "Остановлена пользователем"}


@dataclass
class SeriesRecord:
    """A load series: the same setup at several loads q, every point an ordinary run of the dispatcher."""

    series_id: int
    started: datetime
    snapshot: object                   # SetupState at the start
    requests: tuple                    # RunRequest of every point, in the order entered
    run_ids: list = field(default_factory=list)   # runs started so far (point k -> run_ids[k])
    status: str = "running"            # "running" | "finished" | "cancelled"
    navier: dict | None = None         # job record of the Navier reference of all loads (SSSS), or None
    _job: object = None

    @property
    def loads(self):
        return tuple(r.q for r in self.requests)

    @property
    def active(self):
        return self.status == "running"


class RunManager(QObject):
    """Runs of the session; at most one active run (the setup is locked meanwhile)."""

    runAdded = Signal(int)
    runChanged = Signal(int)
    activeChanged = Signal(bool)
    seriesChanged = Signal(int)

    def __init__(self, clock=time.monotonic, process_factory=SolverProcess, protocol_dir=None, parent=None,
                 navier_factory=navier_job.NavierJob, d2_factory=d2_job.D2Job):
        """navier_factory(inputs) -> job with poll()/terminate() (``backend.navier_job.NavierJob``); None: no reference.
        d2_factory(inputs) -> job with poll()/terminate() (``backend.d2_job.D2Job``); None: D2 is not available."""
        super().__init__(parent)
        self.clock = clock
        self.process_factory = process_factory
        self.navier_factory = navier_factory
        self.d2_factory = d2_factory
        self.protocol_dir = Path(protocol_dir) if protocol_dir else None
        self.runs = []
        self.series = []
        # (job, callback(manager, record)) still running. The callback gets the manager as an argument and never
        # closes over it: a closure would make a reference cycle, and a parentless manager would then be freed by
        # the cyclic garbage collector, possibly in a reader thread of a reference job (a QObject destroyed outside
        # the GUI thread while its timer is registered there — a native crash; GUI-QT-LIFETIME-REGRESSION-FIX-01).
        self._navier_jobs = []
        self._timer = QTimer(self)
        self._timer.setInterval(200)
        self._timer.timeout.connect(self.tick)

    @property
    def active(self):
        return next((r for r in self.runs if r.active), None)

    def get(self, run_id):
        return next((r for r in self.runs if r.run_id == run_id), None)

    def elapsed(self, run):
        return run.elapsed(self.clock())

    def _new_run(self, snapshot, method, origin, stage, **extra):
        if self.active is not None:
            raise RuntimeError("a run is already active")
        run = RunRecord(run_id=len(self.runs) + 1, started=datetime.now(), method=method, snapshot=snapshot,
                        data_origin=origin, stage=stage, _t0=self.clock(), **extra)
        return run

    def _register(self, run):
        self.runs.append(run)
        self._timer.start()
        self.runAdded.emit(run.run_id)
        self.activeChanged.emit(True)
        return run

    # Solver -----------------------------------------------------------------------------------------

    def start(self, snapshot, request, series=None):
        """Start the solver in a separate process for a request that passed ``check_request``.

        The check is repeated here: a rejected request raises ``AdapterInputError`` and no process is created
        (the computing process checks it once more before calling kkvim).
        """
        problems = check_request(request)
        if problems:
            raise AdapterInputError("; ".join(problems))
        run = self._new_run(snapshot, "КК-ВИМ", ORIGIN_SOLVER, "Расчёт в отдельном процессе", request=request)
        run.d2_requested = request.supports == "CCCC" and bool(getattr(snapshot, "with_d2", False))
        if series is not None:
            run.series_id = series.series_id
            run.navier = self._series_point(series, request)
        elif request.supports == "SSSS":
            self._start_navier(request, [request.q], lambda manager, record, r=run: manager._set_navier(r, record))
        route = ROUTES[request.route]
        solver = f"{route['solver']} ({route['call']}; {', '.join(route['decisions'])})"
        run.log("ИНФО", "Формальная проверка ввода и проверка ввода решателем выполнены")
        for key, text, unit in request.entered:
            if unit:
                value = getattr(request, "t" if key == "h" else key)
                run.log("ИНФО", f"{key} = {text} {unit} → {value!r} {base_unit(unit)} (СИ для решателя)")
        run.log("ИНФО", f"Расчёт запущен: {route['title']}; KK-VIM {solver}, {request.supports}, N = {request.N}, "
                        f"сетка {request.n_x}×{request.n_y}", important=True)
        if run.d2_requested:
            run.log("ИНФО", f"Выбран независимый расчёт «{d2_job.TITLE}»: после КК-ВИМ, в отдельном процессе, для тех же "
                            "a, b, h, E, ν, q и той же сетки; поле КК-ВИМ ему не передаётся", important=True)
        if request.route == ROUTE_R4:
            run.log("ПРЕДУПРЕЖДЕНИЕ", "R4: все результаты PROVISIONAL / NOT_QUALIFIED (D-047, D-048)", important=True)
        elif request.route != ROUTE_R3:
            run.log("ПРЕДУПРЕЖДЕНИЕ", f"Только исследование: все результаты {RESEARCH_MARK}", important=True)
        if series is not None:
            series.run_ids.append(run.run_id)  # before runAdded: the dispatcher shows the point of its series
            run.log("ИНФО", f"Серия нагрузок {series.series_id}: точка {len(series.run_ids)} из {len(series.requests)}, "
                            f"q = {request.q!r} Pa", important=True)
        run.log("ЭТАП", run.stage)
        run._process = self.process_factory(request)
        return self._register(run)

    # Independent Navier reference (SSSS) ------------------------------------------------------------

    def _start_navier(self, request, loads, callback):
        """Start the reference job of ``loads``; the record goes to ``callback(self, record)`` (pending until it
        answers)."""
        if self.navier_factory is None:
            return None
        callback(self, navier_job.pending_record())
        try:
            job = self.navier_factory(navier_job.worker_input(request, loads))
        except OSError as exc:
            callback(self, {"state": navier_job.FAILED, "reason": f"процесс эталона Навье не запущен: {exc}"})
            return None
        self._navier_jobs.append((job, callback))
        self._timer.start()
        return job

    def _set_navier(self, run, job_record):
        run.navier = job_record if job_record.get("state") == navier_job.PENDING else \
            navier_job.point_record(job_record, run.request)
        if run.navier["state"] == navier_job.OK:
            run.log("ИНФО", f"Эталон Навье (отдельный процесс, M = {run.navier['M']}): w в центре = {run.navier['w_m']!r} m")
        elif run.navier["state"] != navier_job.PENDING:
            run.log("ИНФО", f"Эталон Навье не получен: {run.navier['reason']}", important=True)
        self.runChanged.emit(run.run_id)

    def _poll_navier(self):
        for job, callback in list(self._navier_jobs):
            record = job.poll()
            if record is not None:
                self._navier_jobs.remove((job, callback))
                callback(self, record)

    @property
    def references_pending(self):
        return bool(self._navier_jobs)

    # Load series ---------------------------------------------------------------------------------------

    def start_series(self, snapshot, requests):
        """Start a load series: every request is checked first; the points run one after another."""
        requests = tuple(requests)
        if not 2 <= len(requests) <= SERIES_MAX:
            raise AdapterInputError(f"серия нагрузок: нужно от 2 до {SERIES_MAX} значений q")
        if self.active is not None:
            raise RuntimeError("a run is already active")
        problems = []
        for k, request in enumerate(requests, start=1):
            problems += [f"q = {request.q!r} Pa (точка {k}): {p}" for p in check_request(request)]
        if problems:
            raise AdapterInputError("; ".join(problems))
        series = SeriesRecord(series_id=len(self.series) + 1, started=datetime.now(), snapshot=snapshot, requests=requests)
        self.series.append(series)
        if requests[0].supports == "SSSS":
            series._job = self._start_navier(requests[0], [r.q for r in requests],
                                             lambda manager, record, s=series: manager._set_series_navier(s, record))
        self._start_point(series)
        return series

    def _series_point(self, series, request):
        if series.navier is None:
            return None
        if series.navier.get("state") == navier_job.PENDING:
            return navier_job.pending_record()
        return navier_job.point_record(series.navier, request)

    def _set_series_navier(self, series, record):
        series.navier = record
        for run_id in series.run_ids:
            run = self.get(run_id)
            run.navier = self._series_point(series, run.request)
            self.runChanged.emit(run_id)
        self.seriesChanged.emit(series.series_id)

    def _start_point(self, series):
        request = series.requests[len(series.run_ids)]
        run = self.start(series.snapshot, request, series=series)
        self.seriesChanged.emit(series.series_id)
        return run

    def _advance_series(self, run):
        """After a run of a series ended: start the next point, or close the series."""
        series = self.get_series(run.series_id)
        if series is None or not series.active:
            return
        if len(series.run_ids) < len(series.requests):
            self._start_point(series)
        else:
            series.status = "finished"
            self.seriesChanged.emit(series.series_id)

    def get_series(self, series_id):
        return next((s for s in self.series if s.series_id == series_id), None)

    def stop_series(self, series_id):
        """Cancel a series: the active point is stopped, the points not yet started are not run."""
        series = self.get_series(series_id)
        if series is None or not series.active:
            return
        series.status = "cancelled"
        active = self.active
        if active is not None and active.series_id == series_id:
            self.stop(active.run_id)
        self.seriesChanged.emit(series_id)

    def _poll_solver(self, run):
        message = run._process.poll()
        if message is None:
            return
        kind, payload = message
        if kind == OK:
            self._solver_returned(run, payload)
        elif kind == INPUT:
            self._fail(run, f"Ввод отклонён до расчёта: {payload}")
        else:
            first = payload.strip().splitlines()[-1] if payload.strip() else payload
            run.log("ОШИБКА", payload)
            self._fail(run, f"Сбой вычислительного процесса: {first}")

    def _solver_returned(self, run, outcome):
        """The KK-VIM process answered: keep the result, then run D2 (if chosen) or end the run."""
        process = run._process
        run.outcome = outcome
        run.s = outcome.s
        run.stop_criterion = outcome.stop_kind
        started, received = getattr(process, "started", None), getattr(process, "received", None)
        timing = getattr(outcome, "timing", None) or {}
        run.timing = {"kkvim_import_s": timing.get("import_s"), "kkvim_solver_s": timing.get("solver_s"),
                      "kkvim_wall_s": None if started is None or received is None else received - started}
        self._log_outcome(run, outcome)
        if run.d2_requested and self._start_d2(run):
            self._stage(run, f"Независимый расчёт «{d2_job.TITLE}» в отдельном процессе")
            return
        self._complete(run)

    def _complete(self, run):
        """End a run whose KK-VIM process answered (and whose D2, if any, ended): status, protocol, next series point."""
        outcome = run.outcome
        self._end(run, "error" if outcome.is_error else "finished", "Ошибка решателя" if outcome.is_error else "Завершён")
        self._save_protocol(run)
        self._advance_series(run)

    def _log_outcome(self, run, outcome):
        level = "ОШИБКА" if outcome.is_error else "ИНФО"
        run.log(level, f"Решатель вернул {outcome.status} за {format_duration(self.clock() - run._t0)}: "
                       f"s = {outcome.s}, k = {outcome.k}", important=True)
        run.log("ИНФО" if outcome.stop_kind == "met" else "ПРЕДУПРЕЖДЕНИЕ", outcome.stop_text, important=True)
        if outcome.reason:
            run.log("ИНФО", f"Причина (решатель): {outcome.reason}")
        if outcome.error:
            run.log("ОШИБКА", outcome.error, important=True)
        run.log("ИНФО", f"Научный статус: {outcome.science_text}", important=True)
        for note in outcome.notes:
            run.log("ПРЕДУПРЕЖДЕНИЕ", note, important=True)
        if not outcome.has_field:
            run.log("ПРЕДУПРЕЖДЕНИЕ", "Поле прогиба не получено: анализ поля недоступен", important=True)
        elif outcome.field_role == ROLE_DIAGNOSTIC:
            run.log("ПРЕДУПРЕЖДЕНИЕ", f"Запрошенный расчёт не выполнен ({outcome.status}): поле последнего выполненного полушага "
                                      f"s = {outcome.s} — только диагностика, не результат", important=True)

    # Independent D2 (CCCC) -------------------------------------------------------------------------

    def _start_d2(self, run):
        """Start the D2 process of ``run``; True if it runs (its record is "pending" until it answers)."""
        if self.d2_factory is None:
            run.d2 = d2_job.not_run_record("процесс D2 недоступен в этой конфигурации")
            run.log("ПРЕДУПРЕЖДЕНИЕ", run.d2["reason"], important=True)
            return False
        run.d2 = d2_job.pending_record()
        try:
            run._d2_job = self.d2_factory(d2_job.worker_input(run.request))
        except OSError as exc:
            run.d2 = {"state": d2_job.FAILED, "reason": f"{d2_job.STATE_TEXT[d2_job.FAILED]}: процесс не запущен: {exc}"}
            run.log("ПРЕДУПРЕЖДЕНИЕ", run.d2["reason"], important=True)
            return False
        run.log("ИНФО", f"«{d2_job.TITLE}» запущен в отдельном процессе (вход: a, b, h, E, ν, q, n_x, n_y; без поля КК-ВИМ)",
                important=True)
        return True

    def _poll_d2(self, run):
        record = run._d2_job.poll()
        if record is None:
            return
        run._d2_job = None
        run.d2 = d2_job.d2_record(record, run.request)
        d2 = run.d2
        if d2["state"] == d2_job.OK:
            centre = ("нет узла в центре" if d2["w_centre_m"] is None else
                      f"w в центре = {d2['w_centre_m']!r} m, β_a = {d2['beta_a']!r}")
            timing = d2["timing"]
            run.log("ИНФО", f"{d2_job.TITLE}: {centre}; e_D2 = {d2['e_D2_m']!r} m; вызов solve "
                            f"{timing.get('solve_s', 0.0):.3f} с, процесс D2 {timing.get('wall_s', 0.0):.3f} с", important=True)
        else:
            run.log("ПРЕДУПРЕЖДЕНИЕ", f"{d2_job.TITLE} не получен: {d2['reason']}", important=True)
        self._complete(run)

    def _save_protocol(self, run):
        if self.protocol_dir is None or run.outcome is None:
            return
        folder = self.protocol_dir / f"{run.started:%Y%m%d-%H%M%S}_run{run.run_id:03d}"
        try:
            export.write_result(run.outcome, folder, run.meta(self.clock()), run.detailed, include_field=False)
        except OSError as exc:
            run.log("ПРЕДУПРЕЖДЕНИЕ", f"Протокол не сохранён: {exc}", important=True)
            return
        run.protocol_path = folder
        run.log("ИНФО", f"Протокол расчёта сохранён: {folder}")

    # Demonstration ------------------------------------------------------------------------------------

    def start_demo(self, snapshot, method="КК-ВИМ"):
        run = self._new_run(snapshot, method, ORIGIN_DEMO, "Подготовка (демонстрация)", k_stop=DEMO_K_STOP,
                            s=0 if method in ITERATIVE_METHODS else None)
        run.log("ИНФО", "Формальная проверка ввода в интерфейсе выполнена")
        run.log("ИНФО", "Расчёт запущен: демонстрационный бэкенд, решатель не вызывается (TEST DATA)",
                important=True)
        run.log("ЭТАП", run.stage)
        return self._register(run)

    def _tick_demo(self, run):
        t = self.clock() - run._t0
        if run.iterative:
            s_stop = 2 * run.k_stop
            if t >= DEMO_PREPARE:
                s = min(s_stop, int((t - DEMO_PREPARE) / DEMO_HALF_STEP))
                if run.s == 0 and s > 0:
                    self._stage(run, "Координатные полушаги (демонстрация)")
                run.s = s
            done = run.s >= s_stop and t >= DEMO_PREPARE + s_stop * DEMO_HALF_STEP + DEMO_FINISH
        else:
            done = t >= DEMO_PREPARE + DEMO_FINISH
        if done:
            if run.iterative:
                run.log("ИНФО", f"Выполнено: s = {run.s} полушагов, k = {run.k} циклов (TEST DATA)")
                run.stop_criterion = "met"
            self._end(run, "finished", "Завершён (демонстрация)")
            run.log("ИНФО", f"Расчёт завершён за {format_duration(run._elapsed_final)} (TEST DATA)",
                    important=True)

    # Common -------------------------------------------------------------------------------------------

    def tick(self):
        self._poll_navier()
        run = self.active
        if run is None:
            if not self._navier_jobs:
                self._timer.stop()
            return
        if run.is_demo:
            self._tick_demo(run)
        elif run._d2_job is not None:
            self._poll_d2(run)
        else:
            self._poll_solver(run)
        self.runChanged.emit(run.run_id)

    def _stage(self, run, stage):
        run.stage = stage
        run.log("ЭТАП", stage)

    def _end(self, run, status, stage):
        run._elapsed_final = self.clock() - run._t0
        run.status = status
        run.stage = stage
        if self.active is None:
            if not self._navier_jobs:
                self._timer.stop()
            self.activeChanged.emit(False)

    def _fail(self, run, message):
        self._end(run, "error", "Ошибка")
        run.stop_criterion = "error"
        run.log("ОШИБКА", message, important=True)
        if run.d2_requested and run.d2 is None:
            run.d2 = d2_job.not_run_record("расчёт КК-ВИМ завершился ошибкой до запуска D2")
        self._advance_series(run)

    def stop(self, run_id):
        run = self.get(run_id)
        if run is None or not run.active:
            return
        series = self.get_series(run.series_id)
        if run._d2_job is not None:  # KK-VIM has answered, D2 runs: stop D2 only; the KK-VIM result is kept
            run._d2_job.terminate()
            run._d2_job = None
            run.d2 = d2_job.cancelled_record()
            run.log("ПРЕДУПРЕЖДЕНИЕ", f"«{d2_job.TITLE}» остановлен пользователем; процесс D2 завершён; результат КК-ВИМ "
                                      "сохранён", important=True)
            if series is not None and series.active:  # «Стоп» of a point cancels the rest of its series
                series.status = "cancelled"
                self.seriesChanged.emit(series.series_id)
            self._complete(run)
            self.runChanged.emit(run.run_id)
            return
        if run._process is not None:
            run._process.terminate()
        if series is not None and series.active:  # «Стоп» of a point cancels the rest of its series
            series.status = "cancelled"
            self.seriesChanged.emit(series.series_id)
        if run.d2_requested and run.d2 is None:
            run.d2 = d2_job.not_run_record("расчёт КК-ВИМ остановлен пользователем")
        self._end(run, "stopped", "Остановлен пользователем")
        run.log("ПРЕДУПРЕЖДЕНИЕ",
                f"Остановлен пользователем через {format_duration(run._elapsed_final)}; "
                "вычислительный процесс завершён; анализ прерванного расчёта недоступен", important=True)
        self.runChanged.emit(run.run_id)

    def fail(self, run_id, message):
        """Error status of an active run (used by tests and by backends)."""
        run = self.get(run_id)
        if run is None or not run.active:
            return
        if run._process is not None:
            run._process.terminate()
        if run._d2_job is not None:
            run._d2_job.terminate()
            run._d2_job = None
            run.d2 = d2_job.not_run_record("запуск завершён с ошибкой; процесс D2 остановлен")
        self._fail(run, message)
        self.runChanged.emit(run.run_id)

    def shutdown(self):
        """Terminate the process of an active run and every reference process (application exit)."""
        for series in self.series:
            if series.active:
                series.status = "cancelled"
        run = self.active
        if run is not None and run._process is not None:
            run._process.terminate()
        if run is not None and run._d2_job is not None:
            run._d2_job.terminate()
            run._d2_job = None
        for job, _ in self._navier_jobs:
            job.terminate()
        self._navier_jobs.clear()
