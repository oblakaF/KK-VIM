"""Own tests of kkvim.coupled: the coupled N-term half-step kernel (R4, step 15; D-046, D-047).

Checks follow docs/R4_NUMERICAL_SPEC.md 2.2-2.6 (ChV4-1 .. ChV4-5, ChF-1, ChF-2), 3.2-3.3
(ChV4-6, ChF-3, residual), 4.1-4.3 (ChV4-8, ChV4-9). Bounds are rounding models of the
computed expressions (absolute-value evaluation, gamma_n = n u, condition numbers), not
the NOT_QUALIFIED constants of D-046 (c_DST, eta_max, A4 tolerances), and not chosen
from observed values. Variant A (the assembled block matrix) is built here only as an
independent control of the same discrete system. These are Builder unit tests, not QA.
"""

import math

import numpy as np
import pytest
import scipy.linalg

import kkvim.coupled as cp
from kkvim.coefficients import ERROR_DEGENERATE, ERROR_INPUT, compute_coefficients
from kkvim.coupled import (
    TRANSFORMS,
    CoupledCoefficients,
    CoupledHalfStepError,
    coupled_coefficients,
    coupled_first_halfstep,
    coupled_operator,
    coupled_residual,
    coupled_second_halfstep,
    odd_sine_set,
    sine_eigenvalues,
    sine_matrix,
    sine_transform,
    solve_coupled_halfstep,
)
from kkvim.grid import Grid1D
from kkvim.linear_solver import ERROR_LINEAR_SOLVE, ERROR_NONFINITE

U = 2.0**-53
D, Q = 3.1, 2.3


def dense_L(grid):
    """(n - 1) x (n - 1) second difference with Dirichlet ends, written out independently of kkvim.grid."""
    m = grid.interior_size
    return (np.diag(-2.0 * np.ones(m)) + np.diag(np.ones(m - 1), 1) + np.diag(np.ones(m - 1), -1)) / grid.h**2


def assembled_K(grid, c, Dval):
    """Variant A: K = D [A (x) L^2 - 2 B (x) L + C (x) I], vec(V) = columns of V one after another (ChF-1/ChF-2)."""
    L = dense_L(grid)
    return Dval * (np.kron(c.A, L @ L) - 2.0 * np.kron(c.B, L) + np.kron(c.C, np.eye(grid.interior_size)))


def vec(V):
    return V.T.reshape(-1)


def unvec(v, N):
    return v.reshape(N, -1).T


def random_set(grid, N, seed):
    """A smooth non-orthogonal set: random combinations of sines plus a polynomial bump."""
    rng = np.random.default_rng(seed)
    x = grid.nodes[1:-1] / grid.length
    basis = np.column_stack([np.sin(m * math.pi * x) for m in range(1, 6)] + [x * (1 - x)])
    return basis @ rng.normal(size=(basis.shape[1], N))


def abs_bounds(grid, V):
    """Absolute-value evaluations of the sums of A, B, C (rounding scale of each entry)."""
    full = np.vstack([np.zeros(V.shape[1]), np.abs(V), np.zeros(V.shape[1])])
    d = (full[1:] + full[:-1]) / grid.h
    l = (full[:-2] + 2.0 * full[1:-1] + full[2:]) / grid.h**2
    h = grid.h
    return h * np.abs(V).T @ np.abs(V), h * d.T @ d, h * l.T @ l


def gamma(n):
    return n * U / (1 - n * U)


# 1-4: coefficients ---------------------------------------------------------------------

@pytest.mark.parametrize("n, N, seed", [(20, 2, 1), (37, 3, 2), (64, 3, 3), (12, 2, 4)])
def test_coefficients_symmetric_and_cross_terms_kept(n, N, seed):
    grid = Grid1D(1.7, n)
    V = random_set(grid, N, seed)
    c = coupled_coefficients(grid, V, Q)
    for M in (c.A, c.B, c.C):
        assert np.array_equal(M, M.T)  # symmetric assembly, A4-2(a): exactly symmetric
    off = ~np.eye(N, dtype=bool)
    assert np.all(np.abs(c.A[off]) > 1e-3 * np.max(np.abs(c.A)))  # cross terms are not lost
    assert np.all(np.abs(c.B[off]) > 1e-3 * np.max(np.abs(c.B)))
    full = np.vstack([np.zeros(N), V, np.zeros(N)])
    dV, lV = np.diff(full, axis=0) / grid.h, (full[:-2] - 2 * full[1:-1] + full[2:]) / grid.h**2
    bA, bB, bC = abs_bounds(grid, V)
    g = gamma(n + 8)
    assert np.all(np.abs(c.A - grid.h * V.T @ V) <= 2 * g * bA)
    assert np.all(np.abs(c.B - grid.h * dV.T @ dV) <= 2 * g * bB)
    assert np.all(np.abs(c.C - grid.h * lV.T @ lV) <= 2 * g * bC)
    assert np.allclose(c.F, Q * grid.h * V.sum(axis=0), rtol=0, atol=2 * g * Q * grid.h * np.abs(V).sum(axis=0).max())


@pytest.mark.parametrize("n, N, seed", [(20, 2, 5), (41, 3, 6), (160, 3, 7)])
def test_summation_by_parts_off_diagonal(n, N, seed):
    """<D+U_i, D+U_j> = -<U_i, L U_j> for i != j (ChV-1, ChV4-1), within the rounding of both sums."""
    grid = Grid1D(2.5, n)
    V = random_set(grid, N, seed)
    c = coupled_coefficients(grid, V, Q)
    L = dense_L(grid)
    other = -grid.h * V.T @ (L @ V)
    _, bB, _ = abs_bounds(grid, V)
    scale_other = grid.h * np.abs(V).T @ (np.abs(L) @ np.abs(V))
    for i in range(N):
        for j in range(N):
            if i != j:
                assert abs(c.B[i, j]) > 1e-3 * np.max(np.abs(c.B))  # a genuinely non-zero cross term
                assert abs(c.B[i, j] - other[i, j]) <= 2 * gamma(n + 8) * (bB[i, j] + scale_other[i, j])


@pytest.mark.parametrize("a, n, N", [(1.0, 4, 2), (1.3, 6, 3), (2.0, 20, 3), (0.7, 64, 2), (1.0, 320, 3)])
def test_odd_sines_give_the_diagonal_coefficients_of_chv4_8(a, n, N):
    grid = Grid1D(a, n)
    X0 = odd_sine_set(grid, N)
    c = coupled_coefficients(grid, X0, Q)
    m = 2 * np.arange(1, N + 1) - 1
    mu = 4 / grid.h**2 * np.sin(m * math.pi / (2 * n)) ** 2
    exact = [a / 2 * np.eye(N), a / 2 * np.diag(mu), a / 2 * np.diag(mu**2)]
    bounds = abs_bounds(grid, X0)
    g = gamma(n + 16)
    for value, ref, b in zip((c.A, c.B, c.C), exact, bounds, strict=True):
        assert np.all(np.abs(value - ref) <= 2 * g * b + 16 * U * np.abs(ref))
    F = Q * grid.h / np.tan(m * math.pi / (2 * n))
    assert np.all(np.abs(c.F - F) <= 2 * g * Q * grid.h * np.abs(X0).sum(axis=0) + 16 * U * np.abs(F))


def test_one_column_diagonal_is_the_r2_coefficient_bitwise():
    grid = Grid1D(1.9, 30)
    f = random_set(grid, 1, 8)[:, 0]
    c, r2 = coupled_coefficients(grid, f[:, None], Q), compute_coefficients(grid, f, Q)
    assert (c.A[0, 0], c.B[0, 0], c.C[0, 0], c.F[0]) == (r2.A, r2.B, r2.C, r2.F)


# 5, 9, 10: start conditions and explicit rejection ------------------------------------------

@pytest.mark.parametrize("N, n_x, n_y", [(3, 5, 20), (3, 20, 5), (3, 4, 4), (3, 4, 40)])  # N = 2 needs n < 4 < N_MIN
def test_start_requires_n_at_least_2N_before_any_solve(N, n_x, n_y, monkeypatch):
    calls = []
    monkeypatch.setattr(cp, "solve_coupled_halfstep", lambda *a, **k: calls.append(1))
    with pytest.raises(CoupledHalfStepError) as info:
        coupled_first_halfstep(1.0, 2.0, D, Q, n_x, n_y, N, transform="dense")
    assert info.value.code == ERROR_INPUT and info.value.N == N and calls == []


@pytest.mark.parametrize("n", [4, 5])
def test_aliased_odd_sines_are_rejected_by_the_start_check(n):
    """n < 2N: a zero column (m = n) or a column equal to another up to sign (ChV4-8, item 2).

    In double these columns are equal only up to rounding (sin(k pi) != 0), so an equality check of
    the set cannot see this dependence; the start check n >= 2N rejects it before any solve.
    """
    grid = Grid1D(1.0, n)
    with pytest.raises(CoupledHalfStepError) as info:
        odd_sine_set(grid, 3)
    assert info.value.code == ERROR_INPUT
    l = np.arange(1, n)
    s3, s5 = (np.sin(m * math.pi * l / n) for m in (3, 5))
    dependent = s5 if n == 5 else s5 + s3  # s_5 = 0 for n = 5, s_5 = -s_3 for n = 4 in exact arithmetic
    assert np.max(np.abs(dependent)) <= 8 * U * 5 * n  # rounding of sin at arguments <= 5 pi
    assert np.any(dependent != 0.0)  # not exactly zero: the equality check alone would miss it


@pytest.mark.parametrize("make", ["zero", "duplicate", "negated"])
def test_zero_and_duplicated_columns_are_rejected(make):
    grid = Grid1D(1.0, 20)
    V = random_set(grid, 3, 9)
    V[:, 2] = {"zero": 0.0 * V[:, 0], "duplicate": V[:, 0], "negated": -V[:, 1]}[make]
    with pytest.raises(CoupledHalfStepError) as info:
        solve_coupled_halfstep(grid, V, Grid1D(2.0, 30), Q, D, direction="Y", transform="dense")
    assert info.value.code == ERROR_DEGENERATE


def test_nonfinite_or_invalid_input_is_explicit():
    grid = Grid1D(1.0, 20)
    V = random_set(grid, 2, 10)
    V[3, 1] = np.nan
    with pytest.raises(CoupledHalfStepError) as info:
        coupled_coefficients(grid, V, Q)
    assert info.value.code == ERROR_NONFINITE
    with pytest.raises(CoupledHalfStepError) as info:
        coupled_coefficients(grid, random_set(grid, 2, 10), 0.0)
    assert info.value.code == ERROR_INPUT
    with pytest.raises(ValueError):
        solve_coupled_halfstep(grid, random_set(grid, 2, 10), grid, Q, D, direction="Y", transform="auto")


@pytest.mark.parametrize("N", [1, 4])
def test_solver_covers_only_n_2_and_3(N):
    """N = 1 stays on the R2/R3 path; N >= 4 is a later research stage (D-047)."""
    grid = Grid1D(1.0, 40)
    with pytest.raises(CoupledHalfStepError) as info:
        solve_coupled_halfstep(grid, random_set(grid, N, 11), grid, Q, D, direction="Y", transform="dense")
    assert info.value.code == ERROR_INPUT


# 7, 8, 10: Cholesky of the mode matrices, explicit failure, no fallback --------------------

def test_failed_cholesky_is_an_explicit_error_with_context():
    indefinite = CoupledCoefficients(A=np.array([[1.0, 2.0], [2.0, 1.0]]), B=np.zeros((2, 2)), C=np.zeros((2, 2)),
                                     F=np.array([1.0, 1.0]))
    grid = Grid1D(1.0, 10)
    with pytest.raises(CoupledHalfStepError) as info:
        cp._solve_modes(indefinite, sine_eigenvalues(grid), np.ones(9), D, direction="X", n=10)
    e = info.value
    assert (e.code, e.direction, e.mode, e.N, e.n) == (ERROR_LINEAR_SOLVE, "X", 1, 2, 10)


def test_no_fallback_when_the_factorization_fails(monkeypatch):
    def failing(*args, **kwargs):
        raise np.linalg.LinAlgError("forced")

    monkeypatch.setattr(scipy.linalg, "cho_factor", failing)
    for name in ("pinv", "lstsq", "solve"):  # any silent fallback would call one of these
        monkeypatch.setattr(scipy.linalg, name, lambda *a, **k: pytest.fail(f"fallback {name}"))
        monkeypatch.setattr(np.linalg, name, lambda *a, **k: pytest.fail(f"fallback {name}"))
    with pytest.raises(CoupledHalfStepError) as info:
        coupled_first_halfstep(1.0, 2.0, D, Q, 20, 40, 3, transform="fft")
    assert info.value.code == ERROR_LINEAR_SOLVE and info.value.mode == 1 and info.value.N == 3


def test_positive_definite_modes_are_solved_by_cholesky():
    grid_x, grid_y = Grid1D(1.0, 24), Grid1D(1.5, 36)
    step = solve_coupled_halfstep(grid_x, random_set(grid_x, 3, 12), grid_y, Q, D, direction="Y", transform="dense")
    c, mu = step.coefficients, sine_eigenvalues(grid_y)
    load = sine_matrix(grid_y) @ np.ones(35)
    for k in range(35):
        M = mu[k] ** 2 * c.A + 2 * mu[k] * c.B + c.C
        assert np.all(np.linalg.eigvalsh(M) > 0)
        r = c.F * load[k] - D * M @ step.modal[k]
        assert np.max(np.abs(r)) <= 8 * gamma(3 * 3 + 8) * D * np.abs(M).sum(axis=1).max() * np.max(np.abs(step.modal[k])) \
            + 8 * U * np.max(np.abs(c.F * load[k]))


def test_nearly_dependent_set_keeps_N_or_fails_explicitly():
    """Limitation of this stage: Cholesky neither certifies nor refutes the rank; N is never reduced."""
    grid = Grid1D(1.0, 32)
    V = random_set(grid, 3, 13)
    V[:, 2] = V[:, 0] + V[:, 1]  # exact dependence in exact arithmetic, not an exact duplicate
    try:
        step = solve_coupled_halfstep(grid, V, Grid1D(1.0, 32), Q, D, direction="Y", transform="dense")
    except CoupledHalfStepError as exc:
        assert exc.code in (ERROR_LINEAR_SOLVE, "ERROR_NONFINITE")
    else:
        assert step.solved.shape == (31, 3) and step.N == 3  # no dropped column, no smaller N


# 11, 12: the sine transform ---------------------------------------------------------------

def transform_bound(n, x):
    """Rounding of Psi @ x: dense gamma_n |Psi| |x| (|Psi|_2 <= sqrt(2n)), FFT c log2(n) u |x|_2."""
    nrm = np.linalg.norm(x, axis=0)
    return (gamma(n + 8) * math.sqrt(2 * n) + 16 * math.log2(n) * U) * nrm


@pytest.mark.parametrize("n", [4, 5, 20, 37, 160, 640])
def test_dense_and_fast_sine_transforms_agree_and_invert(n):
    grid = Grid1D(1.3, n)
    Psi = sine_matrix(grid)
    assert np.array_equal(Psi, Psi.T)
    assert np.max(np.abs(Psi @ Psi - np.eye(n - 1))) <= 2 * gamma(n + 8)
    x = random_set(grid, 3, n)
    dense, fast = sine_transform(grid, x, "dense"), sine_transform(grid, x, "fft")
    assert np.all(np.linalg.norm(dense - fast, axis=0) <= 2 * transform_bound(n, x))
    for method in TRANSFORMS:  # forward and inverse transforms are the same map, no extra factor n
        back = sine_transform(grid, sine_transform(grid, x, method), method)
        assert np.all(np.linalg.norm(back - x, axis=0) <= 2 * transform_bound(n, x))


@pytest.mark.parametrize("n", [6, 21, 40, 320])
def test_eigenvectors_eigenvalues_and_load_of_the_modes(n):
    grid = Grid1D(0.9, n)
    Psi, mu, L = sine_matrix(grid), sine_eigenvalues(grid), dense_L(grid)
    assert np.all(mu > 0) and np.all(np.diff(mu) > 0)
    assert np.max(np.abs(L @ Psi + Psi * mu)) <= 8 * gamma(n + 8) * (4 / grid.h**2) * math.sqrt(2 / n) * 4
    k = np.arange(1, n)
    load = sine_transform(grid, np.ones(n - 1), "dense")
    odd = k % 2 == 1
    closed = math.sqrt(2 / n) / np.tan(k[odd] * math.pi / (2 * n))
    assert np.all(np.abs(load[odd] - closed) <= transform_bound(n, np.ones((n - 1, 1)))[0] + 8 * U * closed)
    assert np.all(np.abs(load[~odd]) <= transform_bound(n, np.ones((n - 1, 1)))[0])  # zero for even k


# 6, 13, 14, 16: variant B against the assembled variant A; residual without DST ---------------

CASES = [(1.0, 2.0, 12, 16, 2, 14), (1.0, 0.2, 40, 8, 3, 15), (2.0, 3.0, 10, 18, 3, 16), (1.0, 5.0, 8, 40, 2, 17)]


@pytest.mark.parametrize("transform", TRANSFORMS)
@pytest.mark.parametrize("a, b, n_x, n_y, N, seed", CASES)
@pytest.mark.parametrize("direction", ["Y", "X"])
def test_modal_solution_equals_the_assembled_system(direction, a, b, n_x, n_y, N, seed, transform):
    gx, gy = Grid1D(a, n_x), Grid1D(b, n_y)
    fixed_grid, solve_grid = (gx, gy) if direction == "Y" else (gy, gx)
    step = solve_coupled_halfstep(fixed_grid, random_set(fixed_grid, N, seed), solve_grid, Q, D,
                                  direction=direction, transform=transform)
    K = assembled_K(solve_grid, step.coefficients, D)
    size = N * solve_grid.interior_size  # K^X: N (n_y - 1) for a Y-half-step, K^Y: N (n_x - 1) for an X-half-step
    assert K.shape == (size, size) and np.allclose(K, K.T, rtol=0, atol=0)
    rhs = np.kron(step.coefficients.F, np.ones(solve_grid.interior_size))
    v_A = np.linalg.solve(K, rhs)
    cond = np.linalg.cond(K)
    err = np.linalg.norm(vec(step.solved) - v_A) / np.linalg.norm(v_A)
    assert err <= 64 * gamma(size + 8) * cond
    # residual in the nodal space without the sine transform == assembled K, within rounding of both
    Kv = unvec(K @ vec(step.solved), N)
    absK = D * (np.kron(np.abs(step.coefficients.A), np.abs(dense_L(solve_grid)) @ np.abs(dense_L(solve_grid)))
                + 2 * np.kron(np.abs(step.coefficients.B), np.abs(dense_L(solve_grid)))
                + np.kron(np.abs(step.coefficients.C), np.eye(solve_grid.interior_size)))
    scale = unvec(absK @ np.abs(vec(step.solved)), N)
    assert np.all(np.abs(coupled_operator(solve_grid, step.coefficients, D, step.solved) - Kv) <= 4 * gamma(N + 16) * scale)
    rho, eta = coupled_residual(solve_grid, step.coefficients, D, step.solved)
    assert rho.shape == (solve_grid.interior_size, N)
    n = solve_grid.n
    assert eta <= 16 * gamma(n + 3 * N + 8) * math.sqrt(n)  # backward-stable modes and transforms; eta_max NOT_QUALIFIED


def two_d_galerkin_residual(gx, gy, X, Y, direction):
    """a_h(w, test) - l_h(test) from the two-dimensional Delta_h (path of Pi_h), not from ChF-1/ChF-2.

    Tests U_i (x) e_k for a Y-half-step (U = X fixed) and e_k (x) V_i for an X-half-step.
    Returns the residual and its absolute-value scale.
    """
    W = X @ Y.T
    Lx, Ly = dense_L(gx), dense_L(gy)
    lap = Lx @ W + W @ Ly
    G = D * gx.h * gy.h * (Lx @ lap + lap @ Ly)
    lap_abs = np.abs(Lx) @ np.abs(W) + np.abs(W) @ np.abs(Ly)
    G_abs = D * gx.h * gy.h * (np.abs(Lx) @ lap_abs + lap_abs @ np.abs(Ly))
    cell = Q * gx.h * gy.h
    if direction == "Y":
        r = X.T @ G - cell * X.sum(axis=0)[:, None]
        s = np.abs(X).T @ G_abs + cell * np.abs(X).sum(axis=0)[:, None]
    else:
        r = G @ Y - cell * Y.sum(axis=0)[None, :]
        s = G_abs @ np.abs(Y) + cell * np.abs(Y).sum(axis=0)[None, :]
    return r, s


# 15, 16, 17, 18: first Y-half-step with the odd sine start and the X-half-step -------------

@pytest.mark.parametrize("transform", TRANSFORMS)
@pytest.mark.parametrize("a, b, n_x, n_y, N", [(1.0, 2.0, 20, 40, 2), (1.0, 2.0, 20, 40, 3), (1.0, 0.2, 100, 20, 3),
                                               (2.0, 3.0, 24, 36, 3), (1.0, 5.0, 16, 80, 2), (1.0, 1.0, 6, 6, 3)])
def test_first_y_halfstep_matches_the_closed_mode_formula(a, b, n_x, n_y, N, transform):
    """ChV4-8/ChV4-9: v_ik = F_i c_k / (D (a/2) (mu_k^y + lambda_i)^2), c_k = psi_k^T 1, only odd k."""
    first = coupled_first_halfstep(a, b, D, Q, n_x, n_y, N, transform=transform)
    assert (first.s, first.k) == (1, 0) and first.step.direction == "Y"
    assert first.step.fixed_grid == Grid1D(a, n_x) and first.step.solve_grid == Grid1D(b, n_y)
    m = 2 * np.arange(1, N + 1) - 1
    lam = 4 / (a / n_x) ** 2 * np.sin(m * math.pi / (2 * n_x)) ** 2
    F = Q * (a / n_x) / np.tan(m * math.pi / (2 * n_x))
    k = np.arange(1, n_y)
    mu = 4 / (b / n_y) ** 2 * np.sin(k * math.pi / (2 * n_y)) ** 2
    odd = k % 2 == 1
    c = np.where(odd, math.sqrt(2 / n_y) / np.tan(k * math.pi / (2 * n_y)), 0.0)
    closed = F[None, :] * c[:, None] / (D * a / 2 * (mu[:, None] + lam[None, :]) ** 2)
    rel = (gamma(n_x + 16) + gamma(n_y + 16)) * 8 + transform_bound(n_y, np.ones((n_y - 1, 1)))[0] / np.min(c[odd]) * 2
    assert np.all(np.abs(first.step.modal[odd] - closed[odd]) <= rel * np.abs(closed[odd]))
    assert np.max(np.abs(first.step.modal[~odd]), initial=0.0) <= rel * np.max(np.abs(closed))
    W = first.field()
    assert W.shape == (n_x + 1, n_y + 1) and not W[0].any() and not W[-1].any() and not W[:, 0].any() and not W[:, -1].any()
    assert np.linalg.matrix_rank(first.y) == N  # rank N in double here (not a certification, 4.3, item 4)
    r, s = two_d_galerkin_residual(first.grid_x, first.grid_y, first.x, first.y, "Y")
    assert np.all(np.abs(r) <= 16 * gamma(n_x + n_y + N + 16) * s)


@pytest.mark.parametrize("transform", TRANSFORMS)
@pytest.mark.parametrize("a, b, n_x, n_y, N", [(1.0, 2.0, 20, 40, 2), (1.0, 2.0, 20, 40, 3), (2.0, 3.0, 24, 36, 3),
                                               (1.0, 0.2, 60, 12, 2)])
def test_x_halfstep_with_the_given_y_set(a, b, n_x, n_y, N, transform):
    first = coupled_first_halfstep(a, b, D, Q, n_x, n_y, N, transform=transform)
    second = coupled_second_halfstep(first, transform=transform)
    assert (second.s, second.k) == (2, 1) and second.step.direction == "X"
    assert second.step.fixed_grid == Grid1D(b, n_y) and second.step.solve_grid == Grid1D(a, n_x)
    assert np.array_equal(second.y, first.y) and second.x.shape == (n_x - 1, N)
    ref = coupled_coefficients(Grid1D(b, n_y), first.y, Q)  # coefficients from Y on G(b, n_y)
    for name in ("A", "B", "C", "F"):
        assert np.array_equal(getattr(second.step.coefficients, name), getattr(ref, name))
    K = assembled_K(second.step.solve_grid, second.step.coefficients, D)
    assert K.shape == (N * (n_x - 1),) * 2  # K^Y: N (n_x - 1)
    r, s = two_d_galerkin_residual(second.grid_x, second.grid_y, second.x, second.y, "X")
    assert np.all(np.abs(r) <= 16 * gamma(n_x + n_y + N + 16) * s)


def test_cross_terms_of_the_y_set_change_the_x_halfstep():
    first = coupled_first_halfstep(1.0, 2.0, D, Q, 20, 40, 3, transform="dense")
    second = coupled_second_halfstep(first, transform="dense")
    c = second.step.coefficients
    off = ~np.eye(3, dtype=bool)
    assert np.all(c.A[off] != 0) and np.all(c.B[off] != 0) and np.all(c.C[off] != 0)
    diagonal = CoupledCoefficients(A=np.diag(np.diag(c.A)), B=np.diag(np.diag(c.B)), C=np.diag(np.diag(c.C)), F=c.F)
    grid = second.step.solve_grid
    mu, load = sine_eigenvalues(grid), sine_transform(grid, np.ones(grid.interior_size), "dense")
    without = sine_transform(grid, cp._solve_modes(diagonal, mu, load, D, direction="X", n=grid.n), "dense")
    assert np.max(np.abs(without - second.x)) > 1e3 * gamma(64) * np.max(np.abs(second.x))


@pytest.mark.parametrize("transform", TRANSFORMS)
def test_same_span_gives_the_same_field_with_off_diagonal_coefficients(transform):
    """A4-6 idea: U = (s1, s3) against U' = (s1, s1 + s3), without orthogonalization (VB-16: field depends on the span)."""
    gx, gy = Grid1D(1.0, 24), Grid1D(1.5, 30)
    U0 = odd_sine_set(gx, 2)
    U1 = np.column_stack([U0[:, 0], U0[:, 0] + U0[:, 1]])
    s0 = solve_coupled_halfstep(gx, U0, gy, Q, D, direction="Y", transform=transform)
    s1 = solve_coupled_halfstep(gx, U1, gy, Q, D, direction="Y", transform=transform)
    assert s1.coefficients.A[0, 1] != 0 and s1.coefficients.B[0, 1] != 0
    W0, W1 = U0 @ s0.solved.T, U1 @ s1.solved.T
    cond = max(np.linalg.cond(s0.coefficients.A), np.linalg.cond(s1.coefficients.A))
    assert np.max(np.abs(W0 - W1)) <= 64 * gamma(24 + 30 + 16) * cond * np.max(np.abs(W0))


# 19, 20: N = 1 and R2/R3 are untouched -------------------------------------------------------

def test_n1_path_is_not_intercepted():
    """kkvim.coupled imports nothing that replaces the R2/R3 half-steps; their modules are unchanged objects."""
    import kkvim.first_halfstep as first
    import kkvim.iteration as iteration
    import kkvim.rectangle as rectangle

    assert first.first_halfstep.__module__ == "kkvim.first_halfstep"
    assert iteration.next_halfstep.__module__ == "kkvim.iteration"
    assert rectangle.rectangle_first_halfstep.__module__ == "kkvim.rectangle"
    assert not any(name in vars(cp) for name in ("first_halfstep", "next_halfstep", "rectangle_first_halfstep"))
