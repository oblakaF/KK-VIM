"""RunOutcome: an immutable copy of one solver result for the interface (GUI_DESIGN.md, 3.3). Qt-free.

The arrays are copies of the solver arrays, made read-only: the interface can neither change the
solver result nor be changed by it. Nothing here is computed from the field except the node of the
largest |w| (a search over the stored values, not a new quantity) and lookups of stored rows.

``origin`` is the only source of the TEST DATA mark (GUI_DESIGN.md, 9): "solver" for a computation
by kkvim, "demo" for the demonstration backend.

``field_role`` is the single statement of what the field is (set by the adapter from the solver status,
``kkvim_adapter.FIELD_ROLE_RULE``); every view, the export and the report read it:
- "accepted" — the field of an accepted R3 run with CONVERGED (field.csv);
- "research" — the field of a research or PROVISIONAL run that ended as requested (field_research.csv);
- "diagnostic" — the field of the last completed half-step of a run that did not end as requested: a
  protective stop, a stop of its stopping rule or a solver error (field_diagnostic.csv); never a result;
- None — no field was computed (no field file).
"""

from dataclasses import dataclass, field

import numpy as np

from kkvim_gui.backend.request import RunRequest

ORIGIN_SOLVER = "solver"
ORIGIN_DEMO = "demo"

ROLE_ACCEPTED = "accepted"
ROLE_RESEARCH = "research"
ROLE_DIAGNOSTIC = "diagnostic"
FIELD_ROLES = (ROLE_ACCEPTED, ROLE_RESEARCH, ROLE_DIAGNOSTIC)
FIELD_FILE = {ROLE_ACCEPTED: "field", ROLE_RESEARCH: "field_research", ROLE_DIAGNOSTIC: "field_diagnostic"}  # + .csv

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
    solver: str  # "R3", "R4", "SSSS-RESEARCH", "CCCC-N1", "CCCC-N23", "CCCC-N46"
    call: str  # the kkvim function that produced the result
    status: str  # status code of the solver, unchanged
    reason: str  # reason text of the solver ("" if it gives none)
    error: str | None
    stop_kind: str
    stop_text: str
    science_kind: str  # "success" | "warning" | "neutral"
    science_text: str
    mark: str | None  # status mark of every non-accepted route (PROVISIONAL / NOT_QUALIFIED, RESEARCH_ONLY / ...), None for R3
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
    route: str = ""  # request.route_of(supports, N)
    field_role: str | None = None  # see the module text; None without a field
    requested_completed: bool = False  # the run ended as requested (R3: CONVERGED; others: the requested end of the mode)
    s_requested: int | None = None  # the requested (fixed or limiting) number of half-steps; None for R3 (s_max is a parameter)
    decisions: tuple = ()  # decisions that authorize the route
    limitations: tuple = ()  # limits of the route, shown in the run screen, the analysis and the report

    def __post_init__(self):
        if self.field_role is not None and self.field_role not in FIELD_ROLES:
            raise ValueError(f"unknown field role {self.field_role!r}")
        if (self.field_role is None) != (self.w is None):
            raise ValueError("a field needs a field role and a field role needs a field")

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
        """The field of the last completed half-step of a run that did not end as requested: diagnostics, never a result."""
        return self.field_role == ROLE_DIAGNOSTIC

    @property
    def field_file(self) -> str | None:
        """Base name of the field file (without the TEST DATA suffix and .csv), or None without a field."""
        return None if self.field_role is None else FIELD_FILE[self.field_role]

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
