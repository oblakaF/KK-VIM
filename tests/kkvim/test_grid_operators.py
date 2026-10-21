"""Own tests of kkvim.grid: grid G(length, n) and operators D+, L, L^2 (R2, D-033).

Checks follow docs/NUMERICAL_SPEC.md, sections 3.1, 4.2 (ChV-1) and 5.1 (ChV-2).
Tolerances allow for rounding; bitwise equality is not required. Comparisons of
identically defined quantities use a tolerance relative to their size. Comparisons
with analytic values (sine eigenvector) use the rounding scale of the operator:
the error of L^p f grows like u * max|f| * (4 / h^2)^p (cf. NUMERICAL_SPEC.md, 3.5).
"""

import math

import numpy as np
import pytest

from kkvim.grid import (
    N_MIN,
    Grid1D,
    forward_difference,
    fourth_difference,
    inner,
    second_difference,
)

RTOL = 1e-12
UNIT_ROUNDOFF = 2.0**-53
GRIDS = [(1.0, 4), (1.0, 5), (1.0, 20), (2.5, 21), (0.7, 101)]


def grid_id(case):
    length, n = case
    return f"length={length}-n={n}"


def random_interior(grid, seed=0):
    return np.random.default_rng(seed).standard_normal(grid.interior_size)


def assert_close(actual, expected, rtol=RTOL):
    scale = max(np.max(np.abs(expected)), np.max(np.abs(actual)), 1.0)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=rtol * scale)


def rounding_atol(grid, f, order):
    """Rounding scale of L^order applied to f, with a safety factor of 64."""
    return 64 * UNIT_ROUNDOFF * float(np.max(np.abs(f))) * (4.0 / grid.h**2) ** order


def operator_matrix(grid, operator):
    """Columns of the operator applied to unit vectors; for tests only."""
    size = grid.interior_size
    return np.column_stack([operator(grid, np.eye(size)[:, j]) for j in range(size)])


# 1. Grid structure ---------------------------------------------------------


@pytest.mark.parametrize("case", GRIDS, ids=grid_id)
def test_grid_nodes_and_step(case):
    length, n = case
    grid = Grid1D(length, n)
    assert grid.n == n
    assert grid.interior_size == n - 1
    assert grid.h == length / n
    nodes = grid.nodes
    assert nodes.shape == (n + 1,)
    assert nodes[0] == 0.0
    assert math.isclose(nodes[-1], length, rel_tol=4e-16)
    np.testing.assert_array_equal(nodes, np.array([i * length / n for i in range(n + 1)]))


@pytest.mark.parametrize("n", [4, 5, 20, 21])
def test_center_is_node_only_for_even_n(n):
    grid = Grid1D(1.0, n)
    assert (0.5 in set(grid.nodes.tolist())) == (n % 2 == 0)


@pytest.mark.parametrize("n", [N_MIN - 1, 2, 0, -4])
def test_rejects_too_few_intervals(n):
    with pytest.raises(ValueError):
        Grid1D(1.0, n)


@pytest.mark.parametrize("n", [4.0, True, "8"])
def test_rejects_non_integer_n(n):
    with pytest.raises(TypeError):
        Grid1D(1.0, n)


@pytest.mark.parametrize("length", [0.0, -1.0, math.inf, math.nan])
def test_rejects_invalid_length(length):
    with pytest.raises(ValueError):
        Grid1D(length, 8)


def test_rejects_wrong_shape_of_grid_function():
    grid = Grid1D(1.0, 8)
    for bad in (np.zeros(8), np.zeros(6), np.zeros((7, 1)), np.zeros(9)):
        with pytest.raises(ValueError):
            second_difference(grid, bad)


# 2. Zero boundary values ---------------------------------------------------


@pytest.mark.parametrize("case", GRIDS, ids=grid_id)
def test_zero_boundary_values(case):
    grid = Grid1D(*case)
    f = random_interior(grid)
    h = grid.h
    full = grid.with_boundary(f)
    assert full[0] == 0.0 and full[-1] == 0.0
    np.testing.assert_array_equal(full[1:-1], f)

    d = forward_difference(grid, f)
    assert d.shape == (grid.n,)
    assert_close(d[[0, -1]], np.array([f[0], -f[-1]]) / h)

    lf = second_difference(grid, f)
    assert_close(lf[[0, -1]], np.array([f[1] - 2 * f[0], f[-2] - 2 * f[-1]]) / h**2)

    # L^2 uses (L f)_0 = (L f)_n = 0 for the intermediate L f.
    l2f = fourth_difference(grid, f)
    assert_close(l2f[[0, -1]], np.array([lf[1] - 2 * lf[0], lf[-2] - 2 * lf[-1]]) / h**2)


# 3. Boundary row of L^2 ----------------------------------------------------


@pytest.mark.parametrize("case", GRIDS, ids=grid_id)
def test_fourth_difference_boundary_rows_are_5_minus4_1(case):
    grid = Grid1D(*case)
    h4 = grid.h**4
    rows = operator_matrix(grid, fourth_difference) * h4
    size = grid.interior_size

    first = np.zeros(size)
    first[:3] = [5.0, -4.0, 1.0]
    assert_close(rows[0], first)
    assert_close(rows[-1], first[::-1])
    assert not math.isclose(rows[0, 0], 7.0, abs_tol=1e-6)  # 7 would be the clamped (CCCC) row

    for i in range(2, size - 2):
        expected = np.zeros(size)
        expected[i - 2 : i + 3] = [1.0, -4.0, 6.0, -4.0, 1.0]
        assert_close(rows[i], expected)


# 4. Sine as an eigenvector of L ------------------------------------------


@pytest.mark.parametrize("case", GRIDS, ids=grid_id)
@pytest.mark.parametrize("k", [1, 2, 3])
def test_sine_is_eigenvector_of_second_difference(case, k):
    grid = Grid1D(*case)
    if k > grid.interior_size:
        pytest.skip("mode does not exist on this grid")
    i = np.arange(1, grid.n)
    f = np.sin(k * np.pi * i / grid.n)
    mu = 4.0 / grid.h**2 * np.sin(k * np.pi / (2 * grid.n)) ** 2
    np.testing.assert_allclose(second_difference(grid, f), -mu * f, rtol=0, atol=rounding_atol(grid, f, 1))
    np.testing.assert_allclose(fourth_difference(grid, f), mu**2 * f, rtol=0, atol=rounding_atol(grid, f, 2))
    # The tolerance is far below the eigenvalue itself, so a wrong operator cannot pass.
    assert rounding_atol(grid, f, 2) < 1e-6 * mu**2


# 5. Summation by parts (ChV-1) ----------------------------------------------


@pytest.mark.parametrize("case", GRIDS, ids=grid_id)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_summation_by_parts(case, seed):
    grid = Grid1D(*case)
    f = random_interior(grid, seed)
    g = random_interior(grid, seed + 100)
    d = forward_difference(grid, f)
    assert d.shape == (grid.n,)  # all n intervals, including both end intervals

    lhs = -inner(grid, f, second_difference(grid, f))
    rhs = grid.h * float(np.sum(d**2))
    assert math.isclose(lhs, rhs, rel_tol=RTOL)

    # Symmetry: <f, L g> = <L f, g> = -h * sum(D+f * D+g).
    cross = -grid.h * float(np.sum(d * forward_difference(grid, g)))
    scale = max(abs(cross), 1.0) * RTOL
    assert math.isclose(inner(grid, f, second_difference(grid, g)), cross, abs_tol=scale)
    assert math.isclose(inner(grid, second_difference(grid, f), g), cross, abs_tol=scale)


def test_inner_product_uses_weight_h():
    grid = Grid1D(2.0, 8)
    u = np.ones(grid.interior_size)
    assert math.isclose(inner(grid, u, u), grid.h * grid.interior_size)


# 6. L^2 f = L(L f) and the five-point stencil (ChV-2) -----------------------


@pytest.mark.parametrize("case", GRIDS, ids=grid_id)
def test_fourth_difference_is_l_of_l_and_matches_stencil(case):
    grid = Grid1D(*case)
    f = random_interior(grid, 7)
    l2f = fourth_difference(grid, f)
    assert_close(l2f, second_difference(grid, second_difference(grid, f)))

    # Independent five-point stencil with f_0 = f_n = 0 and ghosts f_{-1} = -f_1, f_{n+1} = -f_{n-1}.
    n = grid.n
    ext = np.zeros(n + 3)  # indices -1..n+1 stored at 0..n+2
    ext[2:n + 1] = f
    ext[0] = -f[0]
    ext[n + 2] = -f[-1]
    stencil = np.array(
        [ext[i - 1] - 4 * ext[i] + 6 * ext[i + 1] - 4 * ext[i + 2] + ext[i + 3] for i in range(1, n)]
    ) / grid.h**4
    assert_close(l2f, stencil)
