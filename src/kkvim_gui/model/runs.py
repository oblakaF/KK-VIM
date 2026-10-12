"""Run dispatcher model: runs of the current session (GUI_DESIGN.md, 5.3).

Stop is demonstrational now: it only ends the demonstration run (no computing process
exists yet; GUI_DESIGN.md, 3.6). Pause is not offered without a safe backend mechanism.

Each run has its own timer: the elapsed time grows while it runs and is frozen when it
finishes, is stopped or fails. No remaining-time estimate is computed. History lives
only in the current session (persistent storage is a separate task).

The only backend now is the demonstration one: stages and counters are synthetic,
marked TEST DATA, and are not results of the KK-VIM solver. KK-VIM counters follow
D-030: s — completed coordinate half-steps; k = floor(s / 2) — completed full cycles;
N — number of terms (D-003). Methods without iterations show no counters.
"""

import time
from dataclasses import dataclass, field
from datetime import datetime

from PySide6.QtCore import QObject, QTimer, Signal

from kkvim_gui import placeholders
from kkvim_gui.model.setup_state import validate_field

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
    stop_criterion: str = ""          # "met", "k_max" or "" (not reached)
    stage: str = ""
    s: int | None = None              # completed coordinate half-steps (iterative methods)
    k_max: int | None = None
    k_stop: int | None = None
    data_origin: str = "demo"
    important: list = field(default_factory=list)   # (time, level, text)
    detailed: list = field(default_factory=list)
    _t0: float = 0.0
    _elapsed_final: float | None = None

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
    def has_results(self):
        """Analysis is available for finished runs only (not for stopped or failed ones)."""
        return self.status == "finished"

    def elapsed(self, now):
        return self._elapsed_final if self._elapsed_final is not None else max(0.0, now - self._t0)

    def params_text(self):
        st = self.snapshot
        grid = st.texts["grid"].strip() or "по NUMERICAL_SPEC"
        return f"N = {st.N} · k_max = {st.texts['k_max'].strip()} · сетка МКР: {grid}"

    def log(self, level, text, important=False):
        entry = (datetime.now().strftime("%H:%M:%S"), level, text)
        self.detailed.append(entry)
        if important:
            self.important.append(entry)


class RunManager(QObject):
    """Runs of the session; at most one active run (the setup is locked meanwhile)."""

    runAdded = Signal(int)
    runChanged = Signal(int)
    activeChanged = Signal(bool)

    def __init__(self, clock=time.monotonic, parent=None):
        super().__init__(parent)
        self.clock = clock
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

    def start_demo(self, snapshot, method="КК-ВИМ"):
        if self.active is not None:
            raise RuntimeError("a run is already active")
        k_max, _ = validate_field("k_max", snapshot.texts["k_max"])
        run = RunRecord(
            run_id=len(self.runs) + 1,
            started=datetime.now(),
            method=method,
            snapshot=snapshot,
            k_max=k_max,
            k_stop=min(DEMO_K_STOP, k_max) if k_max else DEMO_K_STOP,
            s=0 if method in ITERATIVE_METHODS else None,
            stage="Подготовка (демонстрация)",
            _t0=self.clock(),
        )
        run.log("ИНФО", "Формальная проверка ввода в интерфейсе выполнена")
        run.log("ИНФО", "Расчёт запущен: демонстрационный бэкенд, решатель не подключён (TEST DATA)",
                important=True)
        run.log("ЭТАП", run.stage)
        self.runs.append(run)
        self._timer.start()
        self.runAdded.emit(run.run_id)
        self.activeChanged.emit(True)
        return run

    def tick(self):
        run = self.active
        if run is None:
            self._timer.stop()
            return
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
            self._finish(run)
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

    def _finish(self, run):
        if run.iterative:
            run.log("ИНФО", f"Выполнено: s = {run.s} полушагов, k = {run.k} циклов (TEST DATA)")
            if run.k_max is not None and run.k_max < DEMO_K_STOP:
                run.stop_criterion = "k_max"
                run.log("ПРЕДУПРЕЖДЕНИЕ",
                        "Достигнуто k_max: критерий остановки не выполнен (демонстрация)", important=True)
            else:
                run.stop_criterion = "met"
        self._end(run, "finished", "Завершён (демонстрация)")
        run.log("ИНФО", f"Расчёт завершён за {format_duration(run._elapsed_final)} (TEST DATA)",
                important=True)

    def stop(self, run_id):
        run = self.get(run_id)
        if run is None or not run.active:
            return
        self._end(run, "stopped", "Остановлен пользователем")
        run.log("ПРЕДУПРЕЖДЕНИЕ",
                f"Остановлен пользователем через {format_duration(run._elapsed_final)}; "
                "анализ прерванного расчёта недоступен", important=True)
        self.runChanged.emit(run.run_id)

    def fail(self, run_id, message):
        """Error status (used by real backends later; not produced by the demonstration)."""
        run = self.get(run_id)
        if run is None or not run.active:
            return
        self._end(run, "error", "Ошибка")
        run.log("ОШИБКА", message, important=True)
        self.runChanged.emit(run.run_id)
