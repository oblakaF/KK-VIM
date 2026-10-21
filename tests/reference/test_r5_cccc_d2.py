"""Program checks of the QA reference D2 (kkvim_reference.r5_cccc_d2), plan docs/R5_CCCC_PREQUALIFICATION.plan.md 1.2 (D2-1...D2-12).

These are checks of the implementation of the discrete problem (group A of the reference), not a qualification of the grid
error (that is group B, through A and B). Tolerances are rounding models of a few operations, stated at each check; bitwise
equality is required only where every value is an exact binary number (h = 1/2^k, integer stencil).
"""

import math
from fractions import Fraction

import numpy as np
import pytest
import scipy.linalg

from kkvim_reference import r5_cccc_d2 as d2

U = 2.0**-53
RECT = [(6, 9, 0.37, 1.13), (12, 8, 1.0, 0.7), (9, 14, 0.6, 1.3), (40, 200, 1.0, 5.0)]


def row(problem, i, j):
    """Nonzeros of the row of node (i, j) as {(di, dj): value}."""
    A = d2.assemble(problem).tocsr()
    k = (i - 1) * (problem.ny - 1) + (j - 1)
    out = {}
    for idx in range(A.indptr[k], A.indptr[k + 1]):
        col = A.indices[idx]
        out[(col // (problem.ny - 1) + 1 - i, col % (problem.ny - 1) + 1 - j)] = A.data[idx]
    return out


def test_d2_1_interior_13_point_stencil_bitwise():
    p = d2.D2Problem(1.0, 1.0, 16, 16)  # h = 1/16: every coefficient c n^4 is exact
    s = 16.0**4
    expected = {(0, 0): 20 * s, (1, 0): -8 * s, (-1, 0): -8 * s, (0, 1): -8 * s, (0, -1): -8 * s,
                (2, 0): s, (-2, 0): s, (0, 2): s, (0, -2): s,
                (1, 1): 2 * s, (1, -1): 2 * s, (-1, 1): 2 * s, (-1, -1): 2 * s}
    assert row(p, 8, 8) == expected


def test_d2_2_3_clamped_rows_by_the_even_ghost_node():
    p = d2.D2Problem(1.0, 1.0, 16, 16)
    s = 16.0**4
    edge = row(p, 1, 8)  # x part (7, -4, 1), y part interior (1, -4, 6, -4, 1), mixed term without ghost nodes
    assert edge[(0, 0)] == (7 + 6 + 8) * s and edge[(1, 0)] == -8 * s and edge[(2, 0)] == s
    assert (-1, 0) not in edge and (-2, 0) not in edge
    corner = row(p, 1, 1)  # both parts 7, mixed 8
    assert corner[(0, 0)] == (7 + 7 + 8) * s and corner[(1, 1)] == 2 * s and len(corner) == 6
    zx, _, _ = d2.nodewise_parts(10, 3)  # Z_x restricted to the line j = 1 is the 1D D4 with rows (7, -4, 1) at both ends
    line = np.arange(9) * 2  # unknowns (i, 1), i = 1..9, ordering (i - 1)(n_y - 1)
    dense = zx.toarray()[np.ix_(line, line)]
    assert list(dense[0, :3]) == [7, -4, 1] and list(dense[-1, -3:]) == [1, -4, 7]
    assert list(dense[4, 2:7]) == [1, -4, 6, -4, 1]


@pytest.mark.parametrize("n", [8, 16])
def test_d2_4_5_equal_to_kron_and_to_the_energy_hessian_d1_bitwise_on_dyadic_grids(n):
    p = d2.D2Problem(1.0, 1.0, n, n)
    A = d2.assemble(p).toarray()
    assert np.array_equal(A, d2.kron_matrix(p).toarray())
    assert np.array_equal(A, d2.energy_hessian(p).toarray())  # Proposition D: Hess Pi_h^C = D h_x h_y A
    assert np.array_equal(A, A.T)


@pytest.mark.parametrize("grid", RECT, ids=str)
def test_d2_4_5_7_rectangular_grids_within_the_rounding_of_the_assembly(grid):
    """h_x != h_y and non-dyadic steps: every element is a sum of at most three terms, each a scale rounded once;
    rounding model 8u max|A| (plan D2-4, D2-7)."""
    nx, ny, lx, ly = grid
    p = d2.D2Problem(lx, ly, nx, ny)
    A = d2.assemble(p)
    scale = abs(A).max()
    for other in (d2.kron_matrix(p), d2.energy_hessian(p)):
        assert abs(A - other).max() <= 8 * U * scale
    assert (A != A.T).nnz == 0


@pytest.mark.parametrize("grid", RECT[:3] + [(10, 10, 1.0, 1.0)], ids=str)
def test_d2_6_positive_definite_above_the_proved_bound(grid):
    """lambda_min(A) >= (mu_1x + mu_1y)^2 (plan 1.3, proved: Reviewer 403755b 4.1); computed eigenvalue error model
    e_lambda = c m u lambda_max with c = 1 here (stricter than any larger candidate c_lambda; plan 1.3a)."""
    nx, ny, lx, ly = grid
    p = d2.D2Problem(lx, ly, nx, ny)
    A = d2.assemble(p).toarray()
    scipy.linalg.cholesky(A)  # no failure
    lam = np.linalg.eigvalsh(A)
    hx, hy = lx / nx, ly / ny
    mu = lambda h, n: 4 / h**2 * math.sin(math.pi / (2 * n)) ** 2  # noqa: E731
    bound = (mu(hx, nx) + mu(hy, ny)) ** 2
    assert lam[0] >= bound - A.shape[0] * U * lam[-1]
    assert lam[-1] <= (16 / hx**4 + 32 / (hx**2 * hy**2) + 16 / hy**4) * (1 + A.shape[0] * U)


@pytest.mark.parametrize("r, n", [(2, 8), (3, 6)])
def test_d2_8_side_permutation(r, n):
    p = d2.D2Problem(float(r), 1.0, r * n, n)
    pt = d2.transpose_problem(p)
    A, At = d2.assemble(p).toarray(), d2.assemble(pt).toarray()
    perm = np.arange(p.size).reshape(p.nx - 1, p.ny - 1).T.ravel()  # index of the transposed ordering
    assert np.array_equal(At, A[np.ix_(perm, perm)])  # Pi A Pi^T, bitwise
    s, st = d2.solve(p), d2.solve(pt)
    xt = d2.transpose_field(p, st.x)
    assert np.max(np.abs(s.x - xt)) <= 4 * U * np.max(np.abs(s.x)) + s.e_D2 + st.e_D2
    assert abs(s.alpha() - st.alpha()) <= 4 * U * s.alpha() + s.e_D2 + st.e_D2


def test_d2_9_double_double_residual_equals_the_exact_rational_residual():
    p = d2.D2Problem(0.37, 1.13, 6, 9, D=2.5, q=0.7)  # non-dyadic scales
    rng = np.random.default_rng(3)
    xs = [d2.solve(p).x, rng.standard_normal(p.size) * 1e-3]
    for x in xs:
        dd = d2.residual_dd(p, x)
        exact = d2.residual_exact(p, x)
        terms = float(sum(abs(s) for s in p.scales())) * 32 * float(np.max(np.abs(x))) + float(p.load())
        for a, b in zip(dd, exact):
            assert abs(Fraction(float(a)) - b) <= Fraction(2 * U) * abs(b) + Fraction(64 * U * U * terms)


def test_d2_10_boundary_row_residual_is_O_of_1_over_h():
    """tau h = -1/6 for u = x^2 (1 - x)^2 / 24 (u'''' = 1) in the first row, exactly (Reviewer R-19)."""
    for n in (8, 16, 64):
        h = Fraction(1, n)
        u = lambda x: x**2 * (1 - x) ** 2 / 24  # noqa: E731
        tau = (7 * u(h) - 4 * u(2 * h) + u(3 * h)) / h**4 - 1
        assert tau * h == Fraction(-1, 6)


def test_d2_11_centre_alpha_work_and_refinement():
    p = d2.D2Problem(1.0, 1.0, 32, 32)
    s = d2.solve(p)
    w = s.field()
    assert w.shape == (33, 33) and np.all(w[0] == 0) and np.all(w[:, -1] == 0)
    assert s.centre() == w[16, 16] and s.alpha() == w[16, 16]  # q = D = l = 1
    assert np.allclose(w, w.T, rtol=0, atol=8 * U * np.max(np.abs(w)) + 2 * s.e_D2)  # symmetry of the square
    assert s.corrections[0] > s.corrections[-1] and s.e_D2 / s.alpha() <= 1e-13  # e_D2: plan 1.4 (candidate 1e-13)
    assert s.work() == pytest.approx((1 / 32) ** 2 * w.sum(), rel=4 * U * 33 * 33)
    assert d2.D2Problem(1.0, 1.0, 31, 32).nx == 31 and d2.solve(d2.D2Problem(1.0, 1.0, 9, 8)).alpha() is None


def test_d2_12_scale_invariance_of_alpha():
    """alpha is invariant in D, q and the length (MATH_SPEC 2.5); model 16u alpha + e_D2(1) + e_D2(2) (plan D2-12, AP-8)."""
    base = d2.solve(d2.D2Problem(2.0, 1.0, 32, 16))
    scaled = d2.solve(d2.D2Problem(6.0, 3.0, 32, 16, D=2.5, q=0.7))
    assert abs(base.alpha() - scaled.alpha()) <= 16 * U * base.alpha() + base.e_D2 * 1.0 + scaled.e_D2 * 2.5 / (0.7 * 81)


def test_d2_input_checks():
    for bad in [dict(nx=2), dict(lx=-1.0), dict(D=0.0), dict(q=math.inf), dict(nx=True)]:
        args = dict(lx=1.0, ly=1.0, nx=8, ny=8) | bad
        with pytest.raises(ValueError):
            d2.D2Problem(**args)
    with pytest.raises(ValueError):
        d2.solve(d2.D2Problem(1.0, 1.0, 8, 8), ordering="NATURAL")
