"""Fixed number of KK-VIM coordinate half-steps (R2, step 6; D-030 .. D-033).

s = 1 is ``first_halfstep`` (X0 -> Y1). Every later half-step, starting with s = 2,
is the same ``next_halfstep``: the direction follows from s (docs/MATH_SPEC.md 4.5,
docs/NUMERICAL_SPEC.md 5.2-5.3):

- s odd, Y-half-step: fixed factor X (grid_x), system for Y on grid_y;
  field X_scaled (x) Y_normalized;
- s even, X-half-step: fixed factor Y (grid_y), system for X on grid_x;
  field X_normalized (x) Y_scaled.

The fixed factor is exactly the stored normalized factor computed by the previous
half-step, never renormalized, so w(s-1) lies in the subspace of half-step s
(ChV-11). Each computed factor is normalized by I-4 (9.1). k = s // 2.

The history stores factors, coefficients, normalization and diagnostics; the field
w(s) = X_full (x) Y_full is formed on request. There is no automatic stopping, no
Pi_h and no energy test here. Errors are raised, never replaced by a result.
"""

from dataclasses import dataclass

import numpy as np

from kkvim.coefficients import HalfStepCoefficients
from kkvim.first_halfstep import FirstHalfStepResult, first_halfstep, normalization_factor, solve_halfstep
from kkvim.grid import Grid1D
from kkvim.linear_solver import SolveDiagnostics
from kkvim.second_halfstep import SecondHalfStepResult


@dataclass(frozen=True)
class HalfStepRecord:
    """One half-step s: the factor pair forming w(s) and the data of its computation."""

    s: int
    k: int
    direction: str  # "Y" for odd s, "X" for even s: the computed factor
    grid_x: Grid1D
    grid_y: Grid1D
    D: float
    q: float
    x: np.ndarray  # X factor of w(s), interior values on grid_x
    y: np.ndarray  # Y factor of w(s), interior values on grid_y
    raw: np.ndarray  # computed factor before normalization
    c: float  # I-4 normalization factor of the computed factor
    coefficients: HalfStepCoefficients  # of the fixed factor, on its own grid
    solver_diagnostics: SolveDiagnostics

    def __post_init__(self):
        for name in ("x", "y", "raw"):
            values = np.array(getattr(self, name), dtype=float)
            values.setflags(write=False)
            object.__setattr__(self, name, values)

    @property
    def computed(self) -> np.ndarray:
        """Normalized computed factor; it is the fixed factor of half-step s + 1."""
        return self.y if self.direction == "Y" else self.x

    def field(self) -> np.ndarray:
        """w(s) on all tensor nodes, shape (n_x + 1, n_y + 1), axis 0 = x, axis 1 = y."""
        return np.outer(self.grid_x.with_boundary(self.x), self.grid_y.with_boundary(self.y))


@dataclass(frozen=True)
class HalfStepHistory:
    """Ordered half-steps s = 1 .. count."""

    records: tuple[HalfStepRecord, ...]

    def __len__(self) -> int:
        return len(self.records)

    def record(self, s: int) -> HalfStepRecord:
        if isinstance(s, bool) or not isinstance(s, int) or not 1 <= s <= len(self.records):
            raise IndexError(f"half-step s={s!r} is not in the history 1..{len(self.records)}")
        return self.records[s - 1]

    def field(self, s: int) -> np.ndarray:
        return self.record(s).field()

    def last_two_fields(self) -> tuple[np.ndarray, np.ndarray]:
        """(w(s-1), w(s)) for the last s; requires at least two half-steps."""
        if len(self.records) < 2:
            raise ValueError("at least two half-steps are needed")
        return self.records[-2].field(), self.records[-1].field()


def as_record(result) -> HalfStepRecord:
    """Record of a half-step result; types are checked explicitly."""
    if type(result) is HalfStepRecord:
        return result
    if type(result) is FirstHalfStepResult:
        return HalfStepRecord(
            s=1, k=0, direction="Y", grid_x=result.grid_x, grid_y=result.grid_y, D=result.D, q=result.q,
            x=result.x1_scaled, y=result.y1_normalized, raw=result.y1_raw, c=result.c,
            coefficients=result.coefficients, solver_diagnostics=result.solver_diagnostics,
        )
    if type(result) is SecondHalfStepResult:
        return HalfStepRecord(
            s=2, k=1, direction="X", grid_x=result.grid_x, grid_y=result.grid_y, D=result.D, q=result.q,
            x=result.x2_normalized, y=result.y2_scaled, raw=result.x2_raw, c=result.c2,
            coefficients=result.coefficients, solver_diagnostics=result.solver_diagnostics,
        )
    raise TypeError(f"unsupported half-step result type {type(result).__name__}")


def next_halfstep(previous) -> HalfStepRecord:
    """Half-step s + 1 after the given half-step s >= 1, with the same grids, D and q."""
    prev = as_record(previous)
    s = prev.s + 1
    fixed = prev.computed
    if s % 2 == 1:  # Y-half-step: X fixed on grid_x, Y solved on grid_y
        step = solve_halfstep(prev.grid_x, fixed, prev.grid_y, prev.q, prev.D)
    else:  # X-half-step: Y fixed on grid_y, X solved on grid_x
        step = solve_halfstep(prev.grid_y, fixed, prev.grid_x, prev.q, prev.D)
    c = normalization_factor(step.raw)
    computed = step.raw / c
    scaled = fixed * c
    x, y = (scaled, computed) if s % 2 == 1 else (computed, scaled)
    return HalfStepRecord(
        s=s, k=s // 2, direction="Y" if s % 2 == 1 else "X", grid_x=prev.grid_x, grid_y=prev.grid_y,
        D=prev.D, q=prev.q, x=x, y=y, raw=step.raw, c=c,
        coefficients=step.coefficients, solver_diagnostics=step.solver_diagnostics,
    )


def run_halfsteps(a: float, D: float, q: float, n_x: int, n_y: int, count: int) -> HalfStepHistory:
    """Exactly ``count`` half-steps s = 1 .. count; no automatic stopping."""
    if isinstance(count, bool) or not isinstance(count, (int, np.integer)):
        raise TypeError(f"count must be an integer, got {count!r}")
    if count < 1:
        raise ValueError(f"count must be >= 1, got {count}")
    records = [as_record(first_halfstep(a, D, q, n_x, n_y))]
    while len(records) < count:
        records.append(next_halfstep(records[-1]))
    return HalfStepHistory(records=tuple(records))
