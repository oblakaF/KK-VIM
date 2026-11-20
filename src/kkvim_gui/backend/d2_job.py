"""The independent two-dimensional finite-difference solution D2 of a CCCC run, computed in its own process. Qt-free.

The GUI does not import ``kkvim_reference``: ``D2Job`` starts ``python -I -m kkvim_d2_worker`` (a separate package that
imports only ``kkvim_reference.r5_cccc_d2`` and ``kkvim_reference.navier.flexural_rigidity``), sends it the physical
input of the plate and the grid as JSON and reads one JSON answer (contract ``kkvim-d2-worker/1``). The worker never
receives a KK-VIM field or result, and the KK-VIM result is never changed by it: the D2 record is kept next to the run
(``RunRecord.d2``).

D2 is an independent implementation of the SAME discrete problem as KK-VIM CCCC (D-049.1, D-049.8): a difference between
KK-VIM and D2 is the difference of the low-rank KK-VIM approximation from the full discrete solution on the same grid, not
an error of the continuous plate model, and agreement does not check the grid error. D2 is offered for CCCC only; SSSS
keeps the Navier reference (``backend.navier_job``).

Record of this module (``d2_record``; plain dict, JSON-ready): ``state`` "pending" | "ok" | "out_of_scope" | "rejected" |
"error" | "failed" | "cancelled" | "not_run", ``reason``, and for "ok": the echoed ``inputs``, ``method``, ``source``,
``problem``, ``D`` (N·m), ``w_centre_m``, ``centre`` (i, j, x, y), ``beta_a`` (= w_c D/(q a^4)), ``alpha_short_side``
(= w_c D/(q l^4), l = min(a, b)), ``short_side_m``, ``w_max_abs_node``, ``e_D2_m``, ``refinement_steps``,
``corrections_m``, ``residual_max``, ``factor_nnz``, ``timing``, ``runtime``, ``versions``. The field is kept apart
(``field``: a read-only array, axis 0 = x, metres) and is written to its own CSV, never into the manifest.

Timing (``timing``): the worker's own timers (``import_s``, ``solve_s``, ``post_s``, ``worker_s``; see the worker package)
and two of this process: ``wall_s`` — from the start request of the worker process (``Popen``) to the moment its complete
answer was read (the reader thread records it), i.e. interpreter start, imports, computation, serialization and pipe
transfer; ``parse_s`` — reading the JSON answer in the GUI. Every value is one execution: an observation, not a ranking.
"""

import json
import math
import subprocess
import sys
import threading
import time

import numpy as np

WORKER_MODULE = "kkvim_d2_worker"
CONTRACT = "kkvim-d2-worker/1"
INPUT_KEYS = ("a", "b", "t", "E", "nu", "q")
GRID_KEYS = ("n_x", "n_y")
TIMEOUT = 900.0  # s; the largest allowed grid takes seconds; a hung worker is terminated and reported, never waited for
PENDING, OK, FAILED, CANCELLED, NOT_RUN = "pending", "ok", "failed", "cancelled", "not_run"
STATE_TEXT = {
    PENDING: "МКР D2 вычисляется в отдельном процессе",
    OK: "МКР D2 рассчитан",
    "out_of_scope": "МКР D2 не выполнялся: вне инженерного предела сетки",
    "rejected": "ввод не принят МКР D2",
    "error": "ошибка расчёта МКР D2",
    FAILED: "процесс МКР D2 не дал ответа",
    CANCELLED: "МКР D2 остановлен пользователем",
    NOT_RUN: "МКР D2 не выполнялся",
}
TITLE = "МКР — двумерный (D2)"
TIMER_TEXT = {  # the definitions shown next to every timing (the same in the GUI, report.html and the manifest)
    "kkvim_solver_s": "КК-ВИМ, вызов решателя: проверка ввода kkvim, вызов маршрута kkvim и копирование результата в "
                      "RunOutcome, внутри вычислительного процесса (time.perf_counter); без запуска процесса и импорта",
    "kkvim_import_s": "КК-ВИМ, импорт адаптера (kkvim, NumPy, SciPy) в вычислительном процессе",
    "kkvim_wall_s": "КК-ВИМ, полное время процесса: от запуска процесса до получения результата интерфейсом (опрос "
                    "каждые 0,2 с); включает запуск интерпретатора, импорт, расчёт и передачу",
    "d2_solve_s": "D2, вызов r5_cccc_d2.solve: поузловая сборка, LU-разложение SuperLU, первое решение и итерационное "
                  "уточнение (эталон не разделяет эти фазы), внутри процесса D2 (time.perf_counter)",
    "d2_import_s": "D2, импорт NumPy, SciPy и модулей эталона в процессе D2",
    "d2_post_s": "D2, поле, центр и α из решения и их преобразование в списки",
    "d2_worker_s": "D2, от входа в main процесса D2 до конца вычислений (без запуска интерпретатора, сериализации и передачи)",
    "d2_wall_s": "D2, полное время процесса: от запуска процесса до чтения полного ответа (поток чтения); включает запуск "
                 "интерпретатора, импорт, расчёт, сериализацию JSON и передачу",
    "d2_parse_s": "D2, чтение ответа JSON в интерфейсе",
}
TIMING_ROLE = ("Одно исполнение каждого метода: наблюдение, а не сравнение быстродействия алгоритмов. Границы таймеров "
               "КК-ВИМ и D2 разные (см. определения); D2 запускается после окончания процесса КК-ВИМ, поэтому их "
               "вычисления не шли одновременно.")


def worker_input(request) -> dict:
    """The worker request for the plate and grid of ``request`` (RunRequest, SI)."""
    return {"contract": CONTRACT, "supports": request.supports, **{key: getattr(request, key) for key in INPUT_KEYS},
            **{key: getattr(request, key) for key in GRID_KEYS}}


def worker_command() -> list:
    """The interpreter of this program with the worker module; -I: no current directory or user site on sys.path."""
    return [sys.executable, "-I", "-m", WORKER_MODULE]


def parse_output(text: str, sent: dict) -> dict:
    """The job record of the worker answer ``text`` for the request ``sent``; any defect gives state "failed"."""
    try:
        data = json.loads(text)
    except ValueError as exc:
        return {"state": FAILED, "reason": f"ответ процесса D2 не прочитан: {exc}", "inputs": sent}
    if not isinstance(data, dict) or data.get("contract") != CONTRACT:
        return {"state": FAILED, "reason": "ответ процесса D2 не соответствует контракту", "inputs": sent}
    if data.get("inputs") != sent:  # the answer must be for exactly these inputs (exact equality of the binary64 values)
        return {"state": FAILED, "reason": "процесс D2 ответил для других входных данных", "inputs": sent}
    if data.get("state") == OK:
        result = data.get("result") or {}
        field = result.get("field_m")
        shape = (sent["n_x"] + 1, sent["n_y"] + 1)
        if not isinstance(field, list) or np.shape(field) != shape:
            return {"state": FAILED, "reason": f"в ответе D2 нет поля размера {shape[0]} × {shape[1]}", "inputs": sent}
    return data


class D2Job:
    """One D2 computation in its own process; ``poll`` never blocks, ``terminate`` leaves no process behind."""

    def __init__(self, inputs: dict, command=None, clock=time.perf_counter, timeout=TIMEOUT):
        self.inputs = inputs
        self.clock, self.timeout = clock, timeout
        self.record = None
        self._answer = None
        self._t_answer = None
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self.t0 = clock()
        self._process = subprocess.Popen(command or worker_command(), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                         stderr=subprocess.PIPE, creationflags=flags)
        payload = json.dumps(inputs).encode("utf-8")
        # communicate() in a thread: the answer (the field) exceeds the pipe buffer, so it is read while the worker writes
        self._reader = threading.Thread(target=self._communicate, args=(payload,), daemon=True)
        self._reader.start()

    def _communicate(self, payload):
        try:
            self._answer = self._process.communicate(payload)
        except (OSError, ValueError) as exc:
            self._answer = (b"", str(exc).encode("utf-8", "replace"))
        self._t_answer = self.clock()

    @property
    def pid(self):
        return self._process.pid

    def poll(self):
        """The job record, or None while the worker is still computing."""
        if self.record is not None:
            return self.record
        if self._reader.is_alive():
            if self.clock() - self.t0 > self.timeout:
                self.terminate()
                self.record = {"state": FAILED, "reason": f"процесс D2 не ответил за {self.timeout:g} с и остановлен",
                               "inputs": self.inputs}
            return self.record
        out, err = self._answer or (b"", b"")
        if self._process.returncode != 0 or not out:
            tail = err.decode("utf-8", "replace").strip().splitlines()[-1:] or [f"код {self._process.returncode}"]
            self.record = {"state": FAILED, "reason": f"процесс D2 завершился без ответа: {tail[0]}", "inputs": self.inputs}
            return self.record
        t_parse = self.clock()
        record = parse_output(out.decode("utf-8"), self.inputs)
        t_parsed = self.clock()
        if record.get("state") == OK:
            record["timing"] = {**record.get("timing", {}), "wall_s": self._t_answer - self.t0, "parse_s": t_parsed - t_parse}
        self.record = record
        return self.record

    def terminate(self, timeout=5.0):
        """Stop the worker; True if no process is left."""
        if self._process.poll() is None:
            self._process.kill()
            try:
                self._process.wait(timeout)
            except subprocess.TimeoutExpired:
                return False
        return True


def pending_record() -> dict:
    return {"state": PENDING, "reason": STATE_TEXT[PENDING]}


def not_run_record(reason: str) -> dict:
    return {"state": NOT_RUN, "reason": f"{STATE_TEXT[NOT_RUN]}: {reason}"}


def cancelled_record() -> dict:
    return {"state": CANCELLED, "reason": STATE_TEXT[CANCELLED] + "; значение не получено"}


def d2_record(job_record: dict, request) -> dict:
    """The D2 record of one run (``request``) from a job record; the field is a separate read-only array."""
    state = job_record.get("state")
    if state != OK:
        reason = job_record.get("reason") or ""
        return {"state": state if state in STATE_TEXT else FAILED,
                "reason": f"{STATE_TEXT.get(state, STATE_TEXT[FAILED])}: {reason}".rstrip(": ")}
    sent = job_record["inputs"]
    if any(sent[key] != getattr(request, key) for key in (*INPUT_KEYS, *GRID_KEYS)) or sent["supports"] != request.supports:
        return {"state": FAILED, "reason": "D2 рассчитан для других входных данных"}
    res = job_record["result"]
    field = np.array(res["field_m"], dtype=float)
    field.setflags(write=False)
    h_x, h_y = request.a / request.n_x, request.b / request.n_y
    centre = res.get("centre")
    if centre is not None:  # the coordinates of the node, as the CCCC routes give them (i h_x, j h_y)
        centre = {**centre, "x": centre["i"] * h_x, "y": centre["j"] * h_y}
    record = {"state": OK, "reason": "", "inputs": sent, "method": job_record.get("method", ""),
              "source": job_record.get("source", ""), "problem": job_record.get("problem"), "D": job_record["D"],
              "w_centre_m": res.get("w_centre_m"), "centre": centre, "centre_reason": res.get("centre_reason", ""),
              "beta_a": res.get("beta_a"), "alpha_short_side": res.get("alpha_short_side"),
              "short_side_m": res.get("short_side_m"), "w_max_abs_node": res.get("w_max_abs_node"),
              "e_D2_m": res.get("e_D2_m"), "refinement_steps": res.get("refinement_steps"),
              "corrections_m": res.get("corrections_m"), "residual_max": res.get("residual_max"),
              "factor_nnz": res.get("factor_nnz"), "timing": job_record.get("timing", {}),
              "runtime": job_record.get("runtime", {}), "versions": job_record.get("versions", {})}
    record["field"] = field
    return record


def plain(record: dict | None) -> dict | None:
    """The record without the field array (JSON-ready: the manifest, smoke.json)."""
    if record is None:
        return None
    return {k: v for k, v in record.items() if k != "field"}


def field_difference(record: dict, outcome) -> dict | None:
    """max |w_KK-VIM − w_D2| over the nodes of the common grid (signed value at that node), or None.

    Only for an outcome whose field is a result (never a diagnostic field) on exactly the grid of the D2 record."""
    if record is None or record.get("state") != OK or outcome is None or not outcome.has_field or outcome.diagnostic_only:
        return None
    w = np.asarray(outcome.w)
    if w.shape != record["field"].shape:
        return None
    diff = w - record["field"]
    k = int(np.argmax(np.abs(diff)))
    i, j = divmod(k, w.shape[1])
    scale = float(np.max(np.abs(record["field"])))
    value = float(diff[i, j])
    return {"i": i, "j": j, "x": float(outcome.x[i]), "y": float(outcome.y[j]), "delta_w_m": value,
            "relative_to_max_abs_d2": value / scale if scale and math.isfinite(scale) else None}
