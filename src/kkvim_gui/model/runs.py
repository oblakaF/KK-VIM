"""Run dispatcher model: runs of the current session (GUI_DESIGN.md, 5.3, 3.6).

Two backends:
- the solver (default): ``backend.process.SolverProcess`` runs ``backend.kkvim_adapter.execute`` in a
  separate process; the main window is never blocked, a crash of the computation becomes the status
  «Ошибка», «Стоп» terminates the process. The solver reports no intermediate half-steps, so s and
  k are shown when the run ends;
- the demonstration backend (only with ``--demo``): synthetic stages, marked TEST DATA, never a result.

At most one run is active; a second start while a run is active is refused. Each run has its own
timer. Results of a finished solver run are kept as a ``RunOutcome`` (immutable copies of the solver
arrays). The protocol of every solver run (manifest.json and history.csv) is written to
``protocol_dir`` when it is set. KK-VIM counters follow D-030: s — completed coordinate half-steps;
k = floor(s / 2) — completed full cycles; N — number of terms (D-003).
"""

import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QObject, QTimer, Signal

from kkvim_gui import placeholders
from kkvim_gui.backend import export
from kkvim_gui.backend.outcome import ORIGIN_DEMO, ORIGIN_SOLVER
from kkvim_gui.backend.process import CRASH, INPUT, OK, SolverProcess
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
    _t0: float = 0.0
    _elapsed_final: float | None = None
    _process: object = None

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
        """Analysis: finished runs with a field (demo: finished runs); never stopped or failed runs."""
        if self.status != "finished":
            return False
        return self.is_demo or (self.outcome is not None and self.outcome.has_field)

    @property
    def has_diagnostic_field(self):
        """A solver error that kept the field of its last half-step: may be opened only as diagnostics."""
        return self.status == "error" and self.outcome is not None and self.outcome.diagnostic_only

    def elapsed(self, now):
        return self._elapsed_final if self._elapsed_final is not None else max(0.0, now - self._t0)

    def params_text(self):
        st = self.snapshot
        grid = f"сетка {st.texts['n_x'].strip()}×{st.texts['n_y'].strip()}"
        if self.is_demo:
            return f"N = {st.N} · {grid} · демонстрация"
        if st.N == 1:
            return f"N = 1 · R3 · {grid}" + (" · строгий режим" if st.strict else "")
        return f"N = {st.N} · R4 {st.r4_mode}, {st.transform} · {grid}"

    def log(self, level, text, important=False):
        entry = (datetime.now().strftime("%H:%M:%S"), level, text)
        self.detailed.append(entry)
        if important:
            self.important.append(entry)

    def meta(self, now):
        return {"run_id": self.run_id, "started": self.started.isoformat(timespec="seconds"),
                "execution_status": self.status, "elapsed_s": round(self.elapsed(now), 3),
                "data_origin": self.data_origin}


class RunManager(QObject):
    """Runs of the session; at most one active run (the setup is locked meanwhile)."""

    runAdded = Signal(int)
    runChanged = Signal(int)
    activeChanged = Signal(bool)

    def __init__(self, clock=time.monotonic, process_factory=SolverProcess, protocol_dir=None, parent=None):
        super().__init__(parent)
        self.clock = clock
        self.process_factory = process_factory
        self.protocol_dir = Path(protocol_dir) if protocol_dir else None
        self.runs = []
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

    def start(self, snapshot, request):
        """Start the solver in a separate process for a request that passed ``check_request``."""
        run = self._new_run(snapshot, "КК-ВИМ", ORIGIN_SOLVER, "Расчёт в отдельном процессе", request=request)
        solver = "R3 (run_rectangle_physical)" if request.N == 1 else "R4 (run_coupled)"
        run.log("ИНФО", "Формальная проверка ввода и проверка ввода решателем выполнены")
        for key, text, unit in request.entered:
            if unit:
                value = getattr(request, "t" if key == "h" else key)
                run.log("ИНФО", f"{key} = {text} {unit} → {value!r} {base_unit(unit)} (СИ для решателя)")
        run.log("ИНФО", f"Расчёт запущен: KK-VIM {solver}, N = {request.N}, сетка {request.n_x}×{request.n_y}",
                important=True)
        if request.N != 1:
            run.log("ПРЕДУПРЕЖДЕНИЕ", "R4: все результаты PROVISIONAL / NOT_QUALIFIED (D-047, D-048)", important=True)
        run.log("ЭТАП", run.stage)
        run._process = self.process_factory(request)
        return self._register(run)

    def _poll_solver(self, run):
        message = run._process.poll()
        if message is None:
            return
        kind, payload = message
        if kind == OK:
            self._finish_solver(run, payload)
        elif kind == INPUT:
            self._fail(run, f"Ввод отклонён до расчёта: {payload}")
        else:
            first = payload.strip().splitlines()[-1] if payload.strip() else payload
            run.log("ОШИБКА", payload)
            self._fail(run, f"Сбой вычислительного процесса: {first}")

    def _finish_solver(self, run, outcome):
        run.outcome = outcome
        run.s = outcome.s
        run.stop_criterion = outcome.stop_kind
        status = "error" if outcome.is_error else "finished"
        self._end(run, status, "Ошибка решателя" if outcome.is_error else "Завершён")
        level = "ОШИБКА" if outcome.is_error else "ИНФО"
        run.log(level, f"Решатель вернул {outcome.status} за {format_duration(run._elapsed_final)}: "
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
        self._save_protocol(run)

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
        run = self.active
        if run is None:
            self._timer.stop()
            return
        if run.is_demo:
            self._tick_demo(run)
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
            self._timer.stop()
            self.activeChanged.emit(False)

    def _fail(self, run, message):
        self._end(run, "error", "Ошибка")
        run.stop_criterion = "error"
        run.log("ОШИБКА", message, important=True)

    def stop(self, run_id):
        run = self.get(run_id)
        if run is None or not run.active:
            return
        if run._process is not None:
            run._process.terminate()
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
        self._fail(run, message)
        self.runChanged.emit(run.run_id)

    def shutdown(self):
        """Terminate the process of an active run (application exit)."""
        run = self.active
        if run is not None and run._process is not None:
            run._process.terminate()
