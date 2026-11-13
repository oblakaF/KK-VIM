"""The computation in a separate process (GUI_DESIGN.md, 3.6). Qt-free.

The interface starts one process per run, sends it the RunRequest and receives exactly one message:
("ok", RunOutcome), ("input", text) for an input rejected by ``check_request`` or ("crash", traceback).
A crash of the solver or of the process never reaches the GUI process. The solver has no cooperative
cancellation point, so "Стоп" terminates the process (``terminate``; ``kill`` if it is still alive after
the timeout). The process starts no processes of its own.

A process that dies without a message (killed, out of memory, a crash of the interpreter) is a crash of the computation:
on Windows reading the pipe of a dead writer raises ``BrokenPipeError`` (also ``EOFError`` or another ``OSError``
elsewhere); ``poll`` turns every such failure into ("crash", text with the exit code and the error) instead of raising
it into the interface (GUI-MULTIMETHOD-CCCC-D2-01, O-1).

Timing (``RunOutcome.timing``; seconds, ``time.perf_counter`` of the computing process; one execution, an observation):
``import_s`` — import of the adapter (kkvim, NumPy, SciPy); ``solver_s`` — ``kkvim_adapter.execute``: the input check of
kkvim, the call of the kkvim route and the copy of its result into RunOutcome. ``SolverProcess`` itself records when it
was started and when its message was read (``started``, ``received``: ``time.perf_counter`` of the GUI process; the
message is read by polling, so ``received`` is late by at most one polling interval).
"""

import dataclasses
import multiprocessing
import time
import traceback

_CONTEXT = multiprocessing.get_context("spawn")  # the same on Windows and Linux; no inherited Qt state

OK = "ok"
INPUT = "input"
CRASH = "crash"


def compute(request, connection):
    """Body of the computing process: run the adapter and send one message."""
    t0 = time.perf_counter()
    try:
        from kkvim_gui.backend import kkvim_adapter

        t1 = time.perf_counter()
        try:
            outcome = kkvim_adapter.execute(request)
            t2 = time.perf_counter()
            message = (OK, dataclasses.replace(outcome, timing={"import_s": t1 - t0, "solver_s": t2 - t1}))
        except kkvim_adapter.AdapterInputError as exc:
            message = (INPUT, str(exc))
    except BaseException:  # noqa: BLE001 - everything is reported to the GUI as a crash of the computation
        message = (CRASH, traceback.format_exc())
    try:
        connection.send(message)
    finally:
        connection.close()


class SolverProcess:
    """One computation in its own process; ``poll`` never blocks and never raises."""

    def __init__(self, request, target=compute, clock=time.perf_counter):
        receiver, sender = _CONTEXT.Pipe(duplex=False)
        self._receiver = receiver
        self._clock = clock
        self._process = _CONTEXT.Process(target=target, args=(request, sender), daemon=True,
                                         name="kkvim-computation")
        self.started = clock()
        self._process.start()
        sender.close()  # only the child writes
        self.message = None
        self.received = None

    @property
    def pid(self):
        return self._process.pid

    @property
    def exitcode(self):
        return self._process.exitcode

    def poll(self):
        """The message of the process, or None while it is still computing.

        A dead process without a message gives ("crash", text): the broken pipe is reported, never raised (O-1)."""
        if self.message is not None:
            return self.message
        try:
            if self._receiver.poll():
                try:
                    self.message = self._receiver.recv()
                except EOFError:
                    self.message = (CRASH, self._ended_text())
                self._process.join(timeout=5)
            elif not self._process.is_alive():
                self._process.join(timeout=5)
                self.message = self._receiver.recv() if self._receiver.poll() else (CRASH, self._ended_text())
        except (OSError, EOFError) as exc:  # BrokenPipeError and ConnectionResetError are OSError
            self._process.join(timeout=5)
            if self._process.is_alive():  # the pipe failed while the process still runs: stop it, never leave it behind
                self.terminate()
            self.message = (CRASH, f"{self._ended_text()}; канал связи с процессом закрыт: {type(exc).__name__}: {exc}")
        if self.message is not None and self.received is None:
            self.received = self._clock()
        return self.message

    def _ended_text(self):
        return f"Вычислительный процесс завершился (код {self._process.exitcode}) без результата"

    def terminate(self, timeout=2.0):
        """Stop the computation; True if the process is gone."""
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout)
            if self._process.is_alive():
                self._process.kill()
                self._process.join(timeout)
        self._receiver.close()
        return not self._process.is_alive()
