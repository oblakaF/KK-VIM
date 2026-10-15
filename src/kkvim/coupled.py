"""Coupled N-term half-step of KK-VIM, N >= 2 (R4, step 15; D-046, D-047). Internal, not public API.

Three levels are kept apart (R4, step 15, addendum):

- A. Authors' method (P81/P82; R4_SOURCE_NOTES D-044; MATH_SPEC part B, D-045; D-004):
  w = sum_{j=1}^N X_j(x) Y_j(y); in each half-step one set is fixed and the other is
  found from the variational condition, all N functions jointly from one coupled system;
  the next half-step changes direction and uses the set just found; N is fixed.
- B. Proven discretization of the project (R4_NUMERICAL_SPEC 2.2-2.6, 3.2; ChV4-1 ..
  ChV4-6): grid sums A, B, C, F, the coupled systems ChF-1/ChF-2 (stationarity of Pi_h on
  M_h(U)) and their exact modal splitting ChF-3.
- C. Engineering implementation (not attributed to the authors): array layout, NumPy and
  SciPy, the dense or fast DST-I, the Cholesky factorization of the N x N mode matrices,
  error handling and diagnostics. They solve the same discrete system and change neither
  the system nor the half-step spaces.

docs/R4_NUMERICAL_SPEC.md (D-046):

- 2.2 (ChV4-1): coefficients of the given set U (columns U_j, interior values on its grid)
  A_ij = <U_i, U_j>, B_ij = sum_l h (D+U_i)(D+U_j) = -<U_i, L U_j>, C_ij = <L U_i, L U_j>,
  F_i = q <U_i, 1>; A, B, C are assembled symmetrically (one sum per pair i <= j,
  mirrored; A4-2(a)); the diagonal is the R2 sum of each column.
- 2.3, 2.4 (ChF-1, ChF-2): sum_j D [A_ij L^2 - 2 B_ij L + C_ij I] V_j = F_i 1 on the grid of
  the unknown set. The superscript of A, B, C names the set the coefficients come from:
  a Y-half-step takes them from X on G(a, n_x) and solves on G(b, n_y), an X-half-step the
  other way round; one solver for both directions.
- 3.2 (ChV4-6, ChF-3): with -L = Psi Lambda Psi^T, psi_k(i) = sqrt(2/n) sin(k pi i / n),
  mu_k = (4/h^2) sin^2(k pi / (2n)), the system splits exactly into the mode systems
  D M_k v_k = F (psi_k^T 1), M_k = mu_k^2 A + 2 mu_k B + C, k = 1 .. n - 1, and
  V = Psi V_hat. Each M_k is factored by Cholesky; a failure or a non-finite value is an
  explicit error, never replaced by a pseudo-inverse, least squares, a shifted matrix,
  a dropped column or a smaller N.
- the sine transform is an explicit choice: "dense" (the matrix Psi) or "fft"
  (scipy.fft.dst type I, norm="ortho"); both solve the same discrete problem. Neither is
  normative before QA-delta2.
- 3.3: residual of ChF-1/ChF-2 in the nodal space, F (x) 1 - K vec(V), computed with the
  difference operators of ``kkvim.grid`` and without the sine transform; eta_N is a
  diagnostic (eta_max is NOT_QUALIFIED) and is not the factor error delta_f.
- 4.1-4.3: odd sine start X0_i = sin((2i - 1) pi x / a); n_x >= 2N and n_y >= 2N are
  checked before the first half-step (necessary and sufficient for the full rank of the
  first half-step in exact arithmetic, ChV4-8, ChV4-9); they do not certify the rank
  in double.

A successful Cholesky factorization does not prove that the given set is independent,
and a failure does not prove an exact loss of rank; no rank status (S-1/S-2), QR,
normalization or stopping is done here. N = 1 stays on the R2/R3 path: the half-step
solver accepts only N = 2-3 (D-047).
"""

import math
from dataclasses import dataclass

import numpy as np
import scipy.fft
import scipy.linalg

from kkvim.coefficients import ERROR_DEGENERATE, ERROR_INPUT
from kkvim.first_halfstep import _positive_input
from kkvim.grid import Grid1D, forward_difference, fourth_difference, second_difference
from kkvim.linear_solver import ERROR_LINEAR_SOLVE, ERROR_NONFINITE
from kkvim.r3_input import validate_r3_sides

N_SOLVE = (2, 3)  # D-047: production range of this stage; N = 1 is the R2/R3 path
TRANSFORMS = ("dense", "fft")


class CoupledHalfStepError(RuntimeError):
    """Rejected input or failed coupled half-step, with a specification code and its context."""

    def __init__(self, code: str, reason: str, *, direction=None, mode=None, N=None, n=None):
        context = ", ".join(f"{k}={v}" for k, v in (("direction", direction), ("mode", mode), ("N", N), ("n", n))
                            if v is not None)
        super().__init__(f"{code}: {reason}" + (f" ({context})" if context else ""))
        self.code = code
        self.reason = reason
        self.direction = direction
        self.mode = mode
        self.N = N
        self.n = n


@dataclass(frozen=True)
class CoupledCoefficients:
    """A, B, C (symmetric N x N) and F (N,) of a given set on its own grid."""

    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    F: np.ndarray

    def __post_init__(self):
        for name in ("A", "B", "C", "F"):
            values = np.array(getattr(self, name), dtype=float)
            values.setflags(write=False)
            object.__setattr__(self, name, values)

    @property
    def N(self) -> int:
        return self.F.shape[0]


@dataclass(frozen=True)
class CoupledHalfStep:
    """One coupled half-step: given set on fixed_grid, solved set on solve_grid."""

    direction: str  # "Y" (solve on the y grid) or "X" (solve on the x grid)
    fixed_grid: Grid1D
    solve_grid: Grid1D
    fixed: np.ndarray  # given set, shape (fixed_grid.n - 1, N)
    solved: np.ndarray  # computed set, shape (solve_grid.n - 1, N)
    modal: np.ndarray  # mode coefficients V_hat, shape (solve_grid.n - 1, N)
    coefficients: CoupledCoefficients  # of the given set
    transform: str
    D: float
    q: float

    @property
    def N(self) -> int:
        return self.coefficients.N


@dataclass(frozen=True)
class CoupledPair:
    """Sets X (grid_x) and Y (grid_y) of w = sum_j X_j (x) Y_j after a half-step."""

    s: int
    grid_x: Grid1D
    grid_y: Grid1D
    x: np.ndarray  # shape (n_x - 1, N)
    y: np.ndarray  # shape (n_y - 1, N)
    step: CoupledHalfStep

    @property
    def k(self) -> int:
        return self.s // 2

    def field(self) -> np.ndarray:
        """w on all tensor nodes, shape (n_x + 1, n_y + 1), axis 0 = x, axis 1 = y."""
        return _with_boundary(self.grid_x, self.x) @ _with_boundary(self.grid_y, self.y).T


# coefficients -----------------------------------------------------------------------

def coupled_coefficients(grid: Grid1D, factors, q: float) -> CoupledCoefficients:
    """A, B, C, F of the set ``factors`` (interior values, shape (n - 1, N)) on ``grid``."""
    q = _load(q)
    U = _checked_set(grid, factors)
    N = U.shape[1]
    h = grid.h
    dU = np.column_stack([forward_difference(grid, U[:, j]) for j in range(N)])
    lU = np.column_stack([second_difference(grid, U[:, j]) for j in range(N)])
    A, B, C = (np.empty((N, N)) for _ in range(3))
    with np.errstate(over="ignore", invalid="ignore"):
        for i in range(N):
            for j in range(i, N):
                A[i, j] = A[j, i] = h * float(np.sum(U[:, i] * U[:, j]))
                B[i, j] = B[j, i] = h * float(np.sum(dU[:, i] * dU[:, j]))
                C[i, j] = C[j, i] = h * float(np.sum(lU[:, i] * lU[:, j]))
        F = np.array([q * h * float(np.sum(U[:, i])) for i in range(N)])
    if not all(np.all(np.isfinite(M)) for M in (A, B, C, F)):
        raise CoupledHalfStepError(ERROR_NONFINITE, "non-finite coefficients", N=N, n=grid.n)
    return CoupledCoefficients(A=A, B=B, C=C, F=F)


# sine transform ---------------------------------------------------------------------

def sine_eigenvalues(grid: Grid1D) -> np.ndarray:
    """mu_k = (4 / h^2) sin^2(k pi / (2n)), k = 1 .. n - 1: eigenvalues of -L."""
    k = np.arange(1, grid.n)
    return 4.0 / grid.h**2 * np.sin(k * math.pi / (2 * grid.n)) ** 2


def sine_matrix(grid: Grid1D) -> np.ndarray:
    """Psi with psi_k(i) = sqrt(2/n) sin(k pi i / n), i, k = 1 .. n - 1 (symmetric, orthogonal)."""
    i = np.arange(1, grid.n)
    return math.sqrt(2.0 / grid.n) * np.sin(np.outer(i, i) * math.pi / grid.n)


def sine_transform(grid: Grid1D, values, method: str) -> np.ndarray:
    """Psi @ values along axis 0. Psi^T = Psi^-1 = Psi, so this is the forward and the inverse transform."""
    values = np.asarray(values, dtype=float)
    if values.shape[0] != grid.interior_size:
        raise ValueError(f"expected {grid.interior_size} interior values along axis 0, got {values.shape}")
    if method == "dense":
        return sine_matrix(grid) @ values
    if method == "fft":
        return scipy.fft.dst(values, type=1, norm="ortho", axis=0)
    raise ValueError(f"transform must be one of {TRANSFORMS}, got {method!r}")


# coupled half-step ------------------------------------------------------------------

def solve_coupled_halfstep(fixed_grid: Grid1D, fixed, solve_grid: Grid1D, q: float, D: float, *,
                           direction: str, transform: str) -> CoupledHalfStep:
    """ChF-1 (direction "Y") or ChF-2 ("X") by the modal algorithm ChF-3."""
    if direction not in ("X", "Y"):
        raise ValueError(f"direction must be 'X' or 'Y', got {direction!r}")
    if transform not in TRANSFORMS:
        raise ValueError(f"transform must be one of {TRANSFORMS}, got {transform!r}")
    D = _positive_input("D", D)
    coefficients = coupled_coefficients(fixed_grid, fixed, q)
    N, n = coefficients.N, solve_grid.n
    if N not in N_SOLVE:
        raise CoupledHalfStepError(ERROR_INPUT, f"the coupled solver covers N in {N_SOLVE} (D-047); N = 1 is the R2/R3 path",
                                   direction=direction, N=N, n=n)
    mu = sine_eigenvalues(solve_grid)
    load = sine_transform(solve_grid, np.ones(solve_grid.interior_size), transform)  # psi_k^T 1
    modal = _solve_modes(coefficients, mu, load, D, direction=direction, n=n)
    solved = sine_transform(solve_grid, modal, transform)
    if not np.all(np.isfinite(solved)):
        raise CoupledHalfStepError(ERROR_NONFINITE, "non-finite solved set", direction=direction, N=N, n=n)
    return CoupledHalfStep(direction=direction, fixed_grid=fixed_grid, solve_grid=solve_grid,
                           fixed=np.array(fixed, dtype=float), solved=solved, modal=modal,
                           coefficients=coefficients, transform=transform, D=D, q=float(q))


def _solve_modes(coefficients: CoupledCoefficients, mu: np.ndarray, load: np.ndarray, D: float, *,
                 direction: str, n: int) -> np.ndarray:
    """D M_k v_k = F load_k for every mode k by Cholesky; explicit errors, no fallback."""
    A, B, C, F = coefficients.A, coefficients.B, coefficients.C, coefficients.F
    N = coefficients.N
    if not (np.all(np.isfinite(mu)) and np.all(np.isfinite(load))):
        raise CoupledHalfStepError(ERROR_NONFINITE, "non-finite eigenvalues or modal load", direction=direction, N=N, n=n)
    modal = np.empty((mu.size, N))
    for k in range(mu.size):
        M = mu[k] ** 2 * A + 2.0 * mu[k] * B + C
        if not np.all(np.isfinite(M)):
            raise CoupledHalfStepError(ERROR_NONFINITE, "non-finite mode matrix", direction=direction, mode=k + 1, N=N, n=n)
        try:
            factor = scipy.linalg.cho_factor(M, lower=True, check_finite=False)
        except np.linalg.LinAlgError as exc:
            raise CoupledHalfStepError(ERROR_LINEAR_SOLVE, f"Cholesky factorization of M_k failed: {exc}",
                                       direction=direction, mode=k + 1, N=N, n=n) from exc
        v = scipy.linalg.cho_solve(factor, F * load[k], check_finite=False) / D
        if not np.all(np.isfinite(v)):
            raise CoupledHalfStepError(ERROR_NONFINITE, "non-finite mode solution", direction=direction, mode=k + 1, N=N, n=n)
        modal[k] = v
    return modal


# residual without the sine transform (3.3) ------------------------------------------------

def coupled_operator(solve_grid: Grid1D, coefficients: CoupledCoefficients, D: float, V) -> np.ndarray:
    """Block i of K vec(V): sum_j D [A_ij L^2 V_j - 2 B_ij L V_j + C_ij V_j], shape (n - 1, N)."""
    V = _checked_shape(solve_grid, V, coefficients.N)
    l2V = np.column_stack([fourth_difference(solve_grid, V[:, j]) for j in range(coefficients.N)])
    lV = np.column_stack([second_difference(solve_grid, V[:, j]) for j in range(coefficients.N)])
    return D * (l2V @ coefficients.A.T - 2.0 * (lV @ coefficients.B.T) + V @ coefficients.C.T)


def coupled_residual(solve_grid: Grid1D, coefficients: CoupledCoefficients, D: float, V) -> tuple[np.ndarray, float]:
    """rho = F (x) 1 - K vec(V) (nodal) and the diagnostic eta_N of R4_NUMERICAL_SPEC 3.3 (eta_max NOT_QUALIFIED)."""
    rho = coefficients.F[np.newaxis, :] - coupled_operator(solve_grid, coefficients, D, V)
    h = solve_grid.h
    norm = lambda M: float(np.max(np.sum(np.abs(M), axis=1)))  # noqa: E731 - matrix infinity norm
    scale = (D * (16.0 * norm(coefficients.A) / h**4 + 8.0 * norm(coefficients.B) / h**2 + norm(coefficients.C))
             * float(np.max(np.abs(V))) + float(np.max(np.abs(coefficients.F))))
    return rho, float(np.max(np.abs(rho))) / scale


# start and the first two half-steps --------------------------------------------------

def odd_sine_set(grid: Grid1D, N: int) -> np.ndarray:
    """X0_i = sin((2i - 1) pi l / n) at the interior nodes, i = 1 .. N; requires n >= 2N (ChV4-8)."""
    _check_start_grid("x", grid, N)
    l = np.arange(1, grid.n)
    return np.column_stack([np.sin((2 * i - 1) * math.pi * l / grid.n) for i in range(1, N + 1)])


def coupled_first_halfstep(a: float, b: float, D: float, q: float, n_x: int, n_y: int, N: int, *,
                           transform: str) -> CoupledPair:
    """s = 1: odd sine set X0 on G(a, n_x), coupled Y-half-step on G(b, n_y) (ChF-1)."""
    a, b = validate_r3_sides(a, b)
    if isinstance(N, bool) or not isinstance(N, (int, np.integer)) or N not in N_SOLVE:
        raise CoupledHalfStepError(ERROR_INPUT, f"N must be one of {N_SOLVE} (D-047), got {N!r}")
    grid_x, grid_y = Grid1D(a, n_x), Grid1D(b, n_y)
    _check_start_grid("x", grid_x, N)
    _check_start_grid("y", grid_y, N)
    x0 = odd_sine_set(grid_x, N)
    step = solve_coupled_halfstep(grid_x, x0, grid_y, q, D, direction="Y", transform=transform)
    return CoupledPair(s=1, grid_x=grid_x, grid_y=grid_y, x=x0, y=step.solved, step=step)


def coupled_second_halfstep(first: CoupledPair, *, transform: str) -> CoupledPair:
    """s = 2: the set Y of s = 1 is given on G(b, n_y), coupled X-half-step on G(a, n_x) (ChF-2); no QR."""
    if type(first) is not CoupledPair or first.s != 1:
        raise TypeError("expected the CoupledPair of s = 1")
    step = solve_coupled_halfstep(first.grid_y, first.y, first.grid_x, first.step.q, first.step.D,
                                  direction="X", transform=transform)
    return CoupledPair(s=2, grid_x=first.grid_x, grid_y=first.grid_y, x=step.solved, y=first.y, step=step)


# helpers ------------------------------------------------------------------------------

def _load(q) -> float:
    if isinstance(q, bool) or not isinstance(q, (int, float, np.integer, np.floating)):
        raise TypeError(f"q must be a real number, got {q!r}")
    q = float(q)
    if not math.isfinite(q) or q == 0.0:
        raise CoupledHalfStepError(ERROR_INPUT, f"load q must be finite and nonzero, got {q!r}")
    return q


def _checked_shape(grid: Grid1D, values, N=None) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[0] != grid.interior_size or (N is not None and values.shape[1] != N):
        expected = f"({grid.interior_size}, {'N' if N is None else N})"
        raise ValueError(f"expected a set of shape {expected}, got {values.shape}")
    return values


def _checked_set(grid: Grid1D, factors) -> np.ndarray:
    U = _checked_shape(grid, factors)
    N = U.shape[1]
    if N < 1:
        raise ValueError("the set must have at least one column")
    if not np.all(np.isfinite(U)):
        raise CoupledHalfStepError(ERROR_NONFINITE, "the set contains NaN or infinity", N=N, n=grid.n)
    for j in range(N):
        if not np.any(U[:, j]):
            raise CoupledHalfStepError(ERROR_DEGENERATE, f"column {j + 1} is zero", N=N, n=grid.n)
        for i in range(j):
            if np.array_equal(U[:, i], U[:, j]) or np.array_equal(U[:, i], -U[:, j]):
                raise CoupledHalfStepError(ERROR_DEGENERATE, f"columns {i + 1} and {j + 1} are equal up to sign",
                                           N=N, n=grid.n)
    return U


def _check_start_grid(axis: str, grid: Grid1D, N: int) -> None:
    if grid.n < 2 * N:
        raise CoupledHalfStepError(
            ERROR_INPUT,
            f"n_{axis} = {grid.n} < 2N = {2 * N}: the odd sine start has no full rank at s = 1 (ChV4-8, ChV4-9); "
            "N and the harmonics are not changed",
            N=N, n=grid.n)


def _with_boundary(grid: Grid1D, values: np.ndarray) -> np.ndarray:
    full = np.zeros((grid.n + 1, values.shape[1]))
    full[1:-1] = values
    return full
