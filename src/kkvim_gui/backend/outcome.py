"""RunOutcome: an immutable copy of one solver result for the interface (GUI_DESIGN.md, 3.3). Qt-free.

The arrays are copies of the solver arrays, made read-only: the interface can neither change the
solver result nor be changed by it. Nothing here is computed from the field except the node of the
largest |w| (a search over the stored values, not a new quantity) and lookups of stored rows.

``origin`` is the only source of the TEST DATA mark (GUI_DESIGN.md, 9): "solver" for a computation
by kkvim, "demo" for the demonstration backend.
"""

from dataclasses import dataclass, field

import numpy as np

from kkvim_gui.backend.request import RunRequest

ORIGIN_SOLVER = "solver"
ORIGIN_DEMO = "demo"

# stop: "met" — the stopping rule of the mode is satisfied; "not_met" — the run ended otherwise;
# "error" — an error exit of the solver; "none" — no stopping rule applies (e.g. the normative mode).
STOP_KINDS = ("met", "not_met", "error", "none")


def frozen_copy(values) -> np.ndarray | None:
    if values is None:
        return None
    array = np.array(values, dtype=float, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class Column:
    key: str
    title: str
    unit: str = ""  # SI symbol; "" for dimensionless numbers
    kind: str = "number"  # "number" | "text" | "flag" (text and flags have no unit)


@dataclass(frozen=True)
class RunOutcome:
    origin: str
    request: RunRequest
    solver: str  # "R3" (N = 1) or "R4" (N = 2-3)
    call: str  # the kkvim function that produced the result
    status: str  # status code of the solver, unchanged
    reason: str  # reason text of the solver ("" if it gives none)
    error: str | None
    stop_kind: str
    stop_text: str
    science_kind: str  # "success" | "warning" | "neutral"
    science_text: str
    mark: str | None  # "PROVISIONAL / NOT_QUALIFIED" for every R4 result, None otherwise
    D: float | None  # N·m
    x: np.ndarray | None  # node coordinates, m
    y: np.ndarray | None
    w: np.ndarray | None  # deflection on all nodes, m; axis 0 = x, axis 1 = y
    w_bar: np.ndarray | None  # w D / (q a^4) (MATH_SPEC 2.5)
    s: int  # completed half-steps (D-030)
    columns: tuple = ()  # Column of the history rows
    history: tuple = ()  # one dict per completed half-step
    parameters: tuple = ()  # (name, value) of the solver parameters actually used
    diagnostics: tuple = ()  # (name, value) of the run-level diagnostics
    notes: tuple = ()  # warnings for the user
    versions: dict = field(default_factory=dict)

    @property
    def k(self) -> int:
        return self.s // 2  # D-030

    @property
    def is_error(self) -> bool:
        return self.stop_kind == "error"

    @property
    def has_field(self) -> bool:
        return self.w is not None

    @property
    def diagnostic_only(self) -> bool:
        """An error exit that kept the field of its last completed half-step: diagnostics, never a result."""
        return self.is_error and self.has_field

    @property
    def is_demo(self) -> bool:
        return self.origin == ORIGIN_DEMO

    def extremum(self) -> dict | None:
        """Node with the largest |w| (by nodes of the result grid): indices, coordinates, w and w_bar."""
        if self.w is None:
            return None
        i, j = np.unravel_index(int(np.argmax(np.abs(self.w))), self.w.shape)
        return {"i": int(i), "j": int(j), "x": float(self.x[i]), "y": float(self.y[j]),
                "w": float(self.w[i, j]), "w_bar": float(self.w_bar[i, j]) if self.w_bar is not None else None}

    def history_values(self, key: str) -> list:
        return [row.get(key) for row in self.history]
