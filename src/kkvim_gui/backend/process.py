"""The computation in a separate process (GUI_DESIGN.md, 3.6). Qt-free.

The interface starts one process per run, sends it the RunRequest and receives exactly one message:
("ok", RunOutcome), ("input", text) for an input rejected by ``check_request`` or ("crash", traceback).
A crash of the solver or of the process never reaches the GUI process. The solver has no cooperative
cancellation point, so "Стоп" terminates the process (``terminate``; ``kill`` if it is still alive after
the timeout). The process starts no processes of its own.
"""

import multiprocessing
import traceback

_CONTEXT = multiprocessing.get_context("spawn")  # the same on Windows and Linux; no inherited Qt state

OK = "ok"
INPUT = "input"
CRASH = "crash"


def compute(request, connection):
    """Body of the computing process: run the adapter and send one message."""
    try:
        from kkvim_gui.backend import kkvim_adapter

        try:
            message = (OK, kkvim_adapter.execute(request))
        except kkvim_adapter.AdapterInputError as exc:
            message = (INPUT, str(exc))
    except BaseException:  # noqa: BLE001 - everything is reported to the GUI as a crash of the computation
        message = (CRASH, traceback.format_exc())
    try:
        connection.send(message)
    finally:
        connection.close()


class SolverProcess:
    """One computation in its own process; ``poll`` never blocks."""

    def __init__(self, request, target=compute):
        receiver, sender = _CONTEXT.Pipe(duplex=False)
        self._receiver = receiver
        self._process = _CONTEXT.Process(target=target, args=(request, sender), daemon=True,
                                         name="kkvim-computation")
        self._process.start()
        sender.close()  # only the child writes
        self.message = None

    @property
    def pid(self):
        return self._process.pid

    def poll(self):
        """The message of the process, or None while it is still computing."""
        if self.message is not None:
            return self.message
        if self._receiver.poll():
            try:
                self.message = self._receiver.recv()
            except EOFError:
                self.message = (CRASH, self._ended_text())
            self._process.join(timeout=5)
        elif not self._process.is_alive():
            self._process.join(timeout=5)
            self.message = self._receiver.recv() if self._receiver.poll() else (CRASH, self._ended_text())
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
