"""QA reference D2 for the clamped rectangle (CCCC): the 13-point finite-difference scheme with even ghost nodes (R5-V).

Independent of the main solver: no import of ``kkvim`` (D-028); nothing of Builder is used (D-049.8). Basis:
docs/R5_CCCC_REFERENCE_MATH.pre-review.md, sections 9.2-9.5 (standard construction; Proposition D, Reviewer 62db9f2
R-18, R-19), plan docs/R5_CCCC_PREQUALIFICATION.plan.md, section 1 (D2-1...D2-12, 1.4), D-049.1 and D-049.8.

Problem. Plate [0, l_x] x [0, l_y], D lap^2 w = q, w = dw/dn = 0 on the boundary. Grid h_x = l_x/n_x, h_y = l_y/n_y,
unknowns w_ij at the interior nodes, ordering k = (i - 1)(n_y - 1) + (j - 1) (x index slow). In every interior node

    (delta_x^4 + 2 delta_x^2 delta_y^2 + delta_y^4) w_ij = q / D,

delta^4 = (1, -4, 6, -4, 1)/h^4, delta_x^2 delta_y^2 = (1, -2, 1) x (1, -2, 1)/(h_x^2 h_y^2); w = 0 on boundary nodes; the
clamped condition by the central difference (w_1 - w_-1)/(2h) = 0, i.e. the ghost node w_-1 = w_1 (and w_{n+1} = w_{n-1}).
Ghost nodes only enter delta^4; the mixed term next to the edge reaches boundary nodes only (w = 0).

The matrix is assembled node by node from this stencil (``nodewise_parts``), as three integer parts Z_x, Z_xy, Z_y with
the exact scales (n_x/l_x)^4, (n_x n_y/(l_x l_y))^2, (n_y/l_y)^4. Two control assemblies are separate functions:
``kron_matrix`` (D4 (x) I + 2 L (x) L + I (x) D4) and ``energy_hessian`` (the Hessian of Pi_h^C, the problem D1;
Proposition D: Hess = D h_x h_y A). With D-049.1, D1 = D2: D2 is an independent implementation of the same discrete
problem, NOT an independent check of the grid error (only A and B check it).

Solution (plan 1.4): sparse LU (SuperLU) and iterative refinement with the residual of the EXACT operator: the integer
parts are applied in double-double arithmetic (error-free transformations, Dekker split, no FMA) and the scales are the
exact rationals rounded to double-double (relative 1e-32; their effect on the solution is far below u). An exact
rational residual (``residual_exact``, fractions) is kept for small grids and checks the double-double one. The estimate
of the own error of the reference is the last refinement correction e_D2 (absolute, max norm).

Not established here (AGENTS.md, section 2): the global second order of D2 and Richardson (H-D2-2, OV-M9); D2 says
nothing about the error of the grid relative to omega_0 (group B).
"""

import math
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

D4_STENCIL = ((-2, 1), (-1, -4), (0, 6), (1, -4), (2, 1))
SECOND_STENCIL = ((-1, 1), (0, -2), (1, 1))
ORDERINGS = ("MMD_AT_PLUS_A", "COLAMD")


@dataclass(frozen=True)
class D2Problem:
    """Plate l_x x l_y (full sides), grid n_x x n_y intervals, D, q (consistent units)."""

    lx: float
    ly: float
    nx: int
    ny: int
    D: float = 1.0
    q: float = 1.0

    def __post_init__(self):
        for name in ("nx", "ny"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v < 3:
                raise ValueError(f"{name} must be an integer >= 3, got {v!r}")
        for name in ("lx", "ly", "D"):
            v = getattr(self, name)
            if not (isinstance(v, (int, float)) and math.isfinite(v) and v > 0):
                raise ValueError(f"{name} must be finite and > 0, got {v!r}")
        if not (isinstance(self.q, (int, float)) and math.isfinite(self.q)):
            raise ValueError(f"q must be finite, got {self.q!r}")

    @property
    def size(self) -> int:
        return (self.nx - 1) * (self.ny - 1)

    def scales(self) -> tuple[Fraction, Fraction, Fraction]:
        """Exact rational scales of Z_x, Z_xy, Z_y: (n_x/l_x)^4, (n_x n_y/(l_x l_y))^2, (n_y/l_y)^4."""
        ix = Fraction(self.nx) / Fraction(self.lx)
        iy = Fraction(self.ny) / Fraction(self.ly)
        return ix**4, (ix * iy) ** 2, iy**4

    def load(self) -> Fraction:
        return Fraction(self.q) / Fraction(self.D)


# --- assembly ---------------------------------------------------------------------------------------------------

def _ghost(index: np.ndarray, n: int) -> np.ndarray:
    """Map a node index of -1..n+1 to the unknown it stands for (1..n-1) or 0 for a boundary node (w = 0)."""
    out = index.copy()
    out[index == -1] = 1  # w_-1 = w_1
    out[index == n + 1] = n - 1  # w_{n+1} = w_{n-1}
    out[(out <= 0) | (out >= n)] = 0
    return out


def nodewise_parts(nx: int, ny: int):
    """Integer stencil parts Z_x, Z_xy, Z_y (CSR, int64) assembled node by node with explicit ghost nodes."""
    I, J = np.meshgrid(np.arange(1, nx), np.arange(1, ny), indexing="ij")
    I, J = I.ravel(), J.ravel()
    rows = (I - 1) * (ny - 1) + (J - 1)
    size = (nx - 1) * (ny - 1)

    def part(terms):
        r, c, v = [], [], []
        for di, dj, coef in terms:
            ii, jj = _ghost(I + di, nx), _ghost(J + dj, ny)
            ok = (ii > 0) & (jj > 0)
            r.append(rows[ok])
            c.append((ii[ok] - 1) * (ny - 1) + (jj[ok] - 1))
            v.append(np.full(int(ok.sum()), coef, dtype=np.int64))
        m = sp.coo_matrix((np.concatenate(v), (np.concatenate(r), np.concatenate(c))), shape=(size, size))
        return m.tocsr()  # duplicates (ghost = mirrored node) are summed

    zx = part([(k, 0, c) for k, c in D4_STENCIL])
    zy = part([(0, k, c) for k, c in D4_STENCIL])
    zxy = part([(a, b, 2 * ca * cb) for a, ca in SECOND_STENCIL for b, cb in SECOND_STENCIL])
    return zx, zxy, zy


def assemble(problem: D2Problem) -> sp.csc_matrix:
    """A = s_x Z_x + s_xy Z_xy + s_y Z_y, scales rounded once to double (A vec(w) = q/D 1)."""
    zx, zxy, zy = nodewise_parts(problem.nx, problem.ny)
    sx, sxy, sy = (float(s) for s in problem.scales())
    return (sx * zx.astype(float) + sxy * zxy.astype(float) + sy * zy.astype(float)).tocsc()


def _second_difference(n: int, h: float) -> sp.csr_matrix:
    m = n - 1
    return (sp.diags([np.ones(m - 1), -2 * np.ones(m), np.ones(m - 1)], [-1, 0, 1]) / h**2).tocsr()


def kron_matrix(problem: D2Problem) -> sp.csc_matrix:
    """Control assembly D4_x (x) I + 2 L_x (x) L_y + I (x) D4_y with D4 = L^2 + (2/h^4)(e_1 e_1^T + e_{n-1} e_{n-1}^T)."""
    hx, hy = problem.lx / problem.nx, problem.ly / problem.ny

    def d4(n, h):
        L = _second_difference(n, h)
        corner = sp.diags([np.r_[2.0 / h**4, np.zeros(n - 3), 2.0 / h**4]], [0])
        return (L @ L + corner).tocsr(), L

    D4x, Lx = d4(problem.nx, hx)
    D4y, Ly = d4(problem.ny, hy)
    Ix, Iy = sp.identity(problem.nx - 1), sp.identity(problem.ny - 1)
    return (sp.kron(D4x, Iy) + 2.0 * sp.kron(Lx, Ly) + sp.kron(Ix, D4y)).tocsc()


def energy_hessian(problem: D2Problem) -> sp.csc_matrix:
    """Control assembly D1: Hessian of Pi_h^C / (D h_x h_y) with trapezoid weights (1/2 at boundary nodes), the even ghost
    second difference (L f)_0 = 2 f_1/h^2 and the embedding E (RM 9.4). Proposition D: equals A."""
    hx, hy = problem.lx / problem.nx, problem.ly / problem.ny

    def factors(n, h):
        m = n - 1
        Lb = sp.lil_matrix((n + 1, m))  # (n + 1) x (n - 1): second difference on all nodes, ghost w_-1 = w_1
        for i in range(1, n):
            Lb[i, i - 1] = -2.0 / h**2
            if i - 1 >= 1:
                Lb[i, i - 2] = 1.0 / h**2
            if i + 1 <= n - 1:
                Lb[i, i] = 1.0 / h**2
        Lb[0, 0] = 2.0 / h**2
        Lb[n, m - 1] = 2.0 / h**2
        E = sp.lil_matrix((n + 1, m))
        for i in range(1, n):
            E[i, i - 1] = 1.0
        w = np.full(n + 1, h)
        w[0] = w[-1] = h / 2
        return Lb.tocsr(), E.tocsr(), sp.diags(w)

    Lx, Ex, Wx = factors(problem.nx, hx)
    Ly, Ey, Wy = factors(problem.ny, hy)
    lap = sp.kron(Lx, Ey) + sp.kron(Ex, Ly)
    return ((lap.T @ sp.kron(Wx, Wy) @ lap) / (hx * hy)).tocsc()


# --- double-double residual -------------------------------------------------------------------------------------

_SPLIT = 134217729.0  # 2^27 + 1


def _two_sum(a, b):
    s = a + b
    bb = s - a
    return s, (a - (s - bb)) + (b - bb)


def _split(a):
    c = _SPLIT * a
    hi = c - (c - a)
    return hi, a - hi


def _two_prod(a, b):
    p = a * b
    ah, al = _split(a)
    bh, bl = _split(b)
    return p, ((ah * bh - p) + ah * bl + al * bh) + al * bl


def _dd_add(xh, xl, yh, yl):
    s, t = _two_sum(xh, yh)
    t = t + xl + yl
    return _two_sum(s, t)


def _dd_mul(xh, xl, yh, yl):
    p, e = _two_prod(xh, yh)
    e = e + xh * yl + xl * yh
    return _two_sum(p, e)


def _dd(value: Fraction) -> tuple[float, float]:
    hi = float(value)
    return hi, float(value - Fraction(hi))


def _apply_dd(z: sp.csr_matrix, x: np.ndarray):
    """Z x in double-double for an integer matrix Z (every product exact by TwoProd, sums by TwoSum)."""
    hi = np.zeros(z.shape[0])
    lo = np.zeros(z.shape[0])
    coo = z.tocoo()
    order = np.argsort(coo.row, kind="stable")
    rows, cols, vals = coo.row[order], coo.col[order], coo.data[order].astype(float)
    # process the nonzeros in "layers": the k-th nonzero of every row, so each layer touches every row at most once
    starts = np.searchsorted(rows, np.arange(z.shape[0]))
    position = np.arange(rows.size) - starts[rows]
    for k in range(int(position.max()) + 1):
        sel = position == k
        r = rows[sel]
        p, e = _two_prod(vals[sel], x[cols[sel]])
        h, l_ = _dd_add(hi[r], lo[r], p, e)
        hi[r], lo[r] = h, l_
    return hi, lo


def residual_dd(problem: D2Problem, x: np.ndarray, parts=None) -> np.ndarray:
    """r = q/D 1 - A x of the exact operator, in double-double, rounded to double."""
    parts = parts or nodewise_parts(problem.nx, problem.ny)
    acc_h = np.zeros(problem.size)
    acc_l = np.zeros(problem.size)
    for z, s in zip(parts, problem.scales()):
        th, tl = _apply_dd(z, x)
        sh, sl = _dd(s)
        ph, pl = _dd_mul(th, tl, sh, sl)
        acc_h, acc_l = _dd_add(acc_h, acc_l, ph, pl)
    fh, fl = _dd(problem.load())
    rh, rl = _dd_add(np.full(problem.size, fh), np.full(problem.size, fl), -acc_h, -acc_l)
    return rh + rl


def residual_exact(problem: D2Problem, x: np.ndarray, parts=None) -> list[Fraction]:
    """r = q/D 1 - A x in exact rational arithmetic (small grids only)."""
    parts = parts or nodewise_parts(problem.nx, problem.ny)
    xs = [Fraction(float(v)) for v in x]
    f = problem.load()
    total = [Fraction(0)] * problem.size
    for z, s in zip(parts, problem.scales()):
        z = z.tocsr()
        for row in range(problem.size):
            acc = Fraction(0)
            for idx in range(z.indptr[row], z.indptr[row + 1]):
                acc += int(z.data[idx]) * xs[z.indices[idx]]
            total[row] += s * acc
    return [f - t for t in total]


# --- solution ---------------------------------------------------------------------------------------------------

@dataclass
class D2Solution:
    problem: D2Problem
    ordering: str
    x: np.ndarray  # refined interior values, ordering k = (i - 1)(n_y - 1) + (j - 1)
    x_unrefined: np.ndarray  # first solve, no refinement (as an unrefined SuperLU solve)
    corrections: list[float] = field(default_factory=list)  # max |dx| of every refinement step
    residual_max: list[float] = field(default_factory=list)  # max |r| before every step (double-double residual)
    factor_nnz: int = 0

    @property
    def e_D2(self) -> float:
        """Own error estimate of the reference: the last refinement correction (absolute, max norm)."""
        return self.corrections[-1] if self.corrections else math.inf

    def field(self, x=None) -> np.ndarray:
        """w on all nodes, shape (n_x + 1, n_y + 1), axis 0 = x, zeros on the boundary."""
        p = self.problem
        w = np.zeros((p.nx + 1, p.ny + 1))
        w[1:-1, 1:-1] = (self.x if x is None else x).reshape(p.nx - 1, p.ny - 1)
        return w

    def centre(self, x=None) -> float | None:
        p = self.problem
        if p.nx % 2 or p.ny % 2:
            return None
        return float(self.field(x)[p.nx // 2, p.ny // 2])

    def alpha(self, x=None) -> float | None:
        """alpha = w(centre) D / (q l^4), l = the shorter full side."""
        p = self.problem
        c = self.centre(x)
        return None if c is None else c * p.D / (p.q * min(p.lx, p.ly) ** 4)

    def work(self, x=None) -> float:
        """W_h = h_x h_y sum of interior w (the discrete load work; Pi_h(omega_h) = -q W_h / 2)."""
        p = self.problem
        return float((p.lx / p.nx) * (p.ly / p.ny) * math.fsum((self.x if x is None else x).tolist()))


def solve(problem: D2Problem, ordering: str = "MMD_AT_PLUS_A", max_refine: int = 5) -> D2Solution:
    """Sparse LU of the nodewise matrix and iterative refinement with the double-double residual of the exact operator.
    Stops when max|dx| no longer decreases or after ``max_refine`` steps."""
    if ordering not in ORDERINGS:
        raise ValueError(f"ordering must be one of {ORDERINGS}")
    parts = nodewise_parts(problem.nx, problem.ny)
    sx, sxy, sy = (float(s) for s in problem.scales())
    A = (sx * parts[0].astype(float) + sxy * parts[1].astype(float) + sy * parts[2].astype(float)).tocsc()
    lu = spla.splu(A, permc_spec=ordering)
    f = float(problem.load())
    x = lu.solve(np.full(problem.size, f))
    sol = D2Solution(problem, ordering, x.copy(), x.copy(), factor_nnz=int(lu.L.nnz + lu.U.nnz))
    previous = math.inf
    for _ in range(max_refine):
        r = residual_dd(problem, x, parts)
        sol.residual_max.append(float(np.max(np.abs(r))))
        dx = lu.solve(r)
        size = float(np.max(np.abs(dx)))
        x = x + dx
        sol.corrections.append(size)
        if size >= previous or size == 0.0:
            break
        previous = size
    sol.x = x
    return sol


def transpose_problem(problem: D2Problem) -> D2Problem:
    """The same plate with the axes exchanged (l_x <-> l_y, n_x <-> n_y)."""
    return D2Problem(problem.ly, problem.lx, problem.ny, problem.nx, problem.D, problem.q)


def transpose_field(problem: D2Problem, x_transposed: np.ndarray) -> np.ndarray:
    """Interior values of the transposed problem back in the ordering of ``problem``."""
    return x_transposed.reshape(problem.ny - 1, problem.nx - 1).T.ravel()
