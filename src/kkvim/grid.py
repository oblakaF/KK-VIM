"""One-dimensional grid G(length, n) and its difference operators (R2).

Definitions follow docs/NUMERICAL_SPEC.md, sections 3.1, 4.2 and 5.1 (D-032).

A grid function on G(length, n) is given by its n - 1 interior values
f_1, ..., f_{n-1} as a 1-D array of shape (n - 1,). The end values
f_0 = f_n = 0 are exact (principal condition) and are never stored.

Operators work on interior values and never assemble a matrix:

- ``forward_difference``: (D+ f)_{i+1/2} = (f_{i+1} - f_i) / h on all n intervals,
  shape (n,);
- ``second_difference``: (L f)_i = (f_{i-1} - 2 f_i + f_{i+1}) / h^2 at interior
  nodes, shape (n - 1,);
- ``fourth_difference``: L^2 f = L(L f), where the intermediate L f is taken with
  zero end values (L f)_0 = (L f)_n = 0 (natural condition f'' = 0), shape (n - 1,);
- ``inner``: <u, v> = h * sum_{i=1}^{n-1} u_i v_i.
"""

import math
from dataclasses import dataclass

import numpy as np

N_MIN = 4
"""Minimum number of intervals in the program (NUMERICAL_SPEC.md, 3.3, I-1)."""


@dataclass(frozen=True)
class Grid1D:
    """Uniform grid G(length, n) on [0, length]: n intervals, n + 1 nodes."""

    length: float
    n: int

    def __post_init__(self):
        if isinstance(self.n, bool) or not isinstance(self.n, (int, np.integer)):
            raise TypeError(f"n must be an integer, got {self.n!r}")
        if self.n < N_MIN:
            raise ValueError(f"n must be >= {N_MIN}, got {self.n}")
        if isinstance(self.length, bool) or not isinstance(self.length, (int, float, np.integer, np.floating)):
            raise TypeError(f"length must be a real number, got {self.length!r}")
        if not math.isfinite(self.length) or self.length <= 0:
            raise ValueError(f"length must be finite and positive, got {self.length!r}")
        object.__setattr__(self, "n", int(self.n))
        object.__setattr__(self, "length", float(self.length))

    @property
    def h(self) -> float:
        """Grid step length / n."""
        return self.length / self.n

    @property
    def interior_size(self) -> int:
        """Number of interior nodes, n - 1."""
        return self.n - 1

    @property
    def nodes(self) -> np.ndarray:
        """All n + 1 node coordinates i * length / n, i = 0..n (not accumulated)."""
        return np.arange(self.n + 1) * self.length / self.n

    def with_boundary(self, f) -> np.ndarray:
        """Return all n + 1 nodal values of f, with the exact zeros f_0 = f_n = 0."""
        values = np.zeros(self.n + 1)
        values[1:-1] = self._interior(f)
        return values

    def _interior(self, f) -> np.ndarray:
        values = np.asarray(f, dtype=float)
        if values.shape != (self.interior_size,):
            raise ValueError(
                f"expected interior values of shape ({self.interior_size},), got {values.shape}"
            )
        return values


def forward_difference(grid: Grid1D, f) -> np.ndarray:
    """(D+ f) on all n intervals, including the two end intervals; shape (n,)."""
    full = grid.with_boundary(f)
    return (full[1:] - full[:-1]) / grid.h


def second_difference(grid: Grid1D, f) -> np.ndarray:
    """(L f) at the interior nodes with f_0 = f_n = 0; shape (n - 1,)."""
    full = grid.with_boundary(f)
    return (full[:-2] - 2.0 * full[1:-1] + full[2:]) / grid.h**2


def fourth_difference(grid: Grid1D, f) -> np.ndarray:
    """L^2 f = L(L f) with (L f)_0 = (L f)_n = 0; shape (n - 1,)."""
    return second_difference(grid, second_difference(grid, f))


def inner(grid: Grid1D, u, v) -> float:
    """Grid inner product <u, v> = h * sum of u_i v_i over the interior nodes."""
    return grid.h * float(np.dot(grid._interior(u), grid._interior(v)))
