"""The independent Navier reference for SSSS runs, computed in its own process (GUI_DESIGN.md, 3.1, 3.6). Qt-free.

The GUI does not import ``kkvim_reference``: ``NavierJob`` starts ``python -I -m kkvim_navier_worker`` (a separate
package that imports only ``kkvim_reference.navier``), sends it the physical input of the plate as JSON and reads one
JSON answer (contract ``kkvim-navier-worker/1``). The worker never receives a KK-VIM field or result, and the KK-VIM
result is never changed by it: the reference record is kept next to the run.

Records of this module (plain dicts, JSON-ready):

- job record (``NavierJob.poll``): ``state`` "ok" | "out_of_scope" | "rejected" | "error" | "failed", ``reason``, the
  echoed ``inputs`` and, for "ok", ``M``, ``x``, ``y``, ``D`` and one point per load q;
- point record (``point_record``): the reference for one run (one q) — ``state`` as above or "pending" while the job
  runs; for "ok": ``a``, ``b``, ``t``, ``E``, ``nu``, ``q`` (exactly the request values), ``w_m``, ``w_coef``
  (= w D/(q a^4)), ``error_bound_coef``, ``M``, ``x``, ``y``, ``D``, ``method``, ``source``, ``versions``.
"""

import json
import subprocess
import sys
import threading
import time

WORKER_MODULE = "kkvim_navier_worker"
CONTRACT = "kkvim-navier-worker/1"
INPUT_KEYS = ("a", "b", "t", "E", "nu")
TIMEOUT = 180.0  # s; the reference takes about a second; a hung worker is terminated and reported, never waited for
PENDING = "pending"
OK = "ok"
FAILED = "failed"
STATE_TEXT = {
    PENDING: "эталон Навье вычисляется в отдельном процессе",
    OK: "эталон Навье рассчитан",
    "out_of_scope": "вне области эталона Навье",
    "rejected": "ввод не принят эталоном Навье",
    "error": "ошибка расчёта эталона Навье",
    FAILED: "процесс эталона Навье не дал ответа",
}


def worker_input(request, loads) -> dict:
    """The worker request for the plate of ``request`` (RunRequest, SI) and the loads ``loads`` (Pa)."""
    return {"contract": CONTRACT, **{key: getattr(request, key) for key in INPUT_KEYS}, "q": [float(q) for q in loads]}


def worker_command() -> list:
    """The interpreter of this program with the worker module; -I: no current directory or user site on sys.path."""
    return [sys.executable, "-I", "-m", WORKER_MODULE]


def parse_output(text: str, sent: dict) -> dict:
    """The job record of the worker answer ``text`` for the request ``sent``; any defect gives state "failed"."""
    try:
        data = json.loads(text)
    except ValueError as exc:
        return {"state": FAILED, "reason": f"ответ процесса эталона не прочитан: {exc}", "inputs": sent}
    if not isinstance(data, dict) or data.get("contract") != CONTRACT:
        return {"state": FAILED, "reason": "ответ процесса эталона не соответствует контракту", "inputs": sent}
    echo = data.get("inputs")
    if echo != sent:  # the answer must be for exactly these inputs (exact equality of the binary64 values)
        return {"state": FAILED, "reason": "процесс эталона ответил для других входных данных", "inputs": sent}
    if data.get("state") == OK:
        points = data.get("points")
        if not isinstance(points, list) or [p.get("q") for p in points] != sent["q"]:
            return {"state": FAILED, "reason": "в ответе эталона нет значений для запрошенных нагрузок", "inputs": sent}
    return data


class NavierJob:
    """One reference computation in its own process; ``poll`` never blocks, ``terminate`` leaves no process behind."""

    def __init__(self, inputs: dict, command=None, clock=time.monotonic, timeout=TIMEOUT):
        self.inputs = inputs
        self.clock, self.timeout = clock, timeout
        self.t0 = clock()
        self.record = None
        self._answer = None
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self._process = subprocess.Popen(command or worker_command(), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                         stderr=subprocess.PIPE, creationflags=flags)
        payload = json.dumps(inputs).encode("utf-8")
        # communicate() in a thread: the answer may exceed the pipe buffer, so it is read while the worker writes
        self._reader = threading.Thread(target=self._communicate, args=(payload,), daemon=True)
        self._reader.start()

    def _communicate(self, payload):
        try:
            self._answer = self._process.communicate(payload)
        except (OSError, ValueError) as exc:
            self._answer = (b"", str(exc).encode("utf-8", "replace"))

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
                self.record = {"state": FAILED, "reason": f"процесс эталона не ответил за {self.timeout:g} с и остановлен",
                               "inputs": self.inputs}
            return self.record
        out, err = self._answer or (b"", b"")
        if self._process.returncode != 0 or not out:
            tail = err.decode("utf-8", "replace").strip().splitlines()[-1:] or [f"код {self._process.returncode}"]
            self.record = {"state": FAILED, "reason": f"процесс эталона завершился без ответа: {tail[0]}",
                           "inputs": self.inputs}
        else:
            self.record = parse_output(out.decode("utf-8"), self.inputs)
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


def point_record(job_record: dict, request) -> dict:
    """The reference of one run (``request``) from a job record: its own load q, the job's plate."""
    state = job_record.get("state")
    if state != OK:
        reason = job_record.get("reason") or ""
        return {"state": state if state in STATE_TEXT else FAILED,
                "reason": f"{STATE_TEXT.get(state, STATE_TEXT[FAILED])}: {reason}".rstrip(": ")}
    sent = job_record["inputs"]
    plate = {key: sent[key] for key in INPUT_KEYS}
    if any(plate[key] != getattr(request, key) for key in INPUT_KEYS):
        return {"state": FAILED, "reason": "эталон рассчитан для другой пластины"}
    point = next((p for p in job_record["points"] if p["q"] == request.q), None)
    if point is None:
        return {"state": FAILED, "reason": "эталон не рассчитан для этой нагрузки q"}
    return {"state": OK, "reason": "", **plate, "q": point["q"], "w_m": point["w_m"], "w_coef": point["w_coef"],
            "error_bound_coef": point["error_bound_coef"], "M": job_record["M"], "x": job_record["x"],
            "y": job_record["y"], "D": job_record["D"], "method": job_record.get("method", ""),
            "source": job_record.get("source", ""), "versions": job_record.get("versions", {})}
