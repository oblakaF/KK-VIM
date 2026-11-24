"""Targets of a computing process that end abnormally (GUI-MULTIMETHOD-CCCC-D2-01, O-1). Not a test module.

``backend.process.SolverProcess`` starts its target with the "spawn" method, so the target must be importable by name in
the child: the tests put this directory on ``sys.path`` before importing this module.
"""

import os
import time


def die_at_once(request, connection):
    """The process ends at once with exit code 3 and never sends a message (no exception, no cleanup)."""
    os._exit(3)


def die_later(request, connection):
    """The process ends with exit code 7 after a short time, without a message."""
    time.sleep(0.5)
    os._exit(7)


def hang(request, connection):
    """The process computes 'forever' (until it is killed from outside)."""
    time.sleep(600)
