"""Own tests of kkvim.linear_solver: 1D solver K Y = g (R2, step 2; D-032, D-033).

Manufactured solutions and tolerances follow docs/NUMERICAL_SPEC.md, sections 8, 12,
13 and table 18.1 (A-1a, A-1b, A-2a, A-2b, A-2v, A-6). Tolerances are the approved
ones; they are not adjusted to the results. The Richardson check A-2b is an
empirical criterion (D-032), not a proved property.
"""

import math
import tracemalloc

import numpy as np
import pytest

import kkvim.linear_solver as ls
from kkvim.grid import Grid1D
from kkvim.linear_solver import (
    ERROR_LINEAR_SOLVE,
    ERROR_NONFINITE,
    ETA_MAX,
    IMAG_MAX,
    LinearSolveError,
    apply_k,
    backward_error,
    characteristic_roots,
    solve_k,
)

PI = math.pi
SINE_SETS = {
    "A1-B0-C0": (1.0, 0.0, 0.0),
    "A1-B3-C5": (1.0, 3.0, 5.0),
    "double": (0.5, PI**2 / 2, PI**4 / 2),
}
NEAR_DOUBLE = (0.5, PI**2 / 2, 1.001 * PI**4 / 2)
POLY_SETS = {"A1-B0-C0": (1.0, 0.0, 0.0), "A1-B3-C5": (1.0, 3.0, 5.0)}
LENGTHS = (1.0, 2.5)
MAIN_GRIDS = (20, 40, 80, 160, 320)
ALL_GRIDS = MAIN_GRIDS + (640,)

A1A_TOL = 1e-11  # table 18.1, A-1a
A1B_WINDOW = (1.95, 2.05)  # A-1b
A2A_WINDOW = (1.9, 2.1)  # A-2a
A2B_RATIO = 0.05  # A-2b
A2V_TOL = 1e-14  # A-2v


def assert_a6(diagnostics):
    """A-6: eta <= 1e-12 and Im <= 1e-12 after at most one refinement."""
    assert diagnostics.eta <= ETA_MAX
    assert diagnostics.imag_ratio <= IMAG_MAX
    assert diagnostics.refinements <= 1


# T-1D-1: sine (section 13.1) -----------------------------------------------------


def sine_case(length, n, coeffs, D=1.0):
    A, B, C = coeffs
    grid = Grid1D(length, n)
    kappa = PI / length
    f = np.sin(PI * np.arange(1, n) / n)  # sin(kappa * y_i), y_i = i * length / n
    mu1 = 4.0 / grid.h**2 * math.sin(PI / (2 * n)) ** 2
    symbol = A * kappa**4 + 2 * B * kappa**2 + C
    rho = symbol / (A * mu1**2 + 2 * B * mu1 + C)
    result = solve_k(grid, A, B, C, D, D * symbol * f)
    return grid, f, rho, result


@pytest.mark.parametrize("length", LENGTHS)
@pytest.mark.parametrize("name", SINE_SETS)
def test_sine_exact_discrete_solution_and_order(name, length):
    coeffs = SINE_SETS[name]
    errors = {}
    for n in ALL_GRIDS:
        grid, f, rho, result = sine_case(length, n, coeffs)
        max_f = np.max(np.abs(f))
        # A-1a: agreement with the exact discrete solution rho_h * f.
        assert np.max(np.abs(result.y - rho * f)) / max_f <= A1A_TOL
        assert_a6(result.diagnostics)
        errors[n] = np.max(np.abs(result.y - f)) / max_f
        # The error is exactly |1 - rho_h| (chV-16) up to the A-1a level.
        assert abs(errors[n] - abs(1.0 - rho)) <= A1A_TOL
    # A-1b: observed order on all pairs up to 320.
    for n in MAIN_GRIDS[:-1]:
        p = math.log2(errors[n] / errors[2 * n])
        assert A1B_WINDOW[0] <= p <= A1B_WINDOW[1], (n, p)


# Root regimes ----------------------------------------------------------------------


def test_distinct_real_roots_branch():
    A, B, C = POLY_SETS["A1-B3-C5"]
    assert B * B > A * C
    r1, r2 = characteristic_roots(A, B, C)
    assert r1.imag == 0.0 and r2.imag == 0.0
    assert r1.real == pytest.approx(5.0) and r2.real == pytest.approx(1.0)
    assert r1.real >= 0.0 and r2.real >= 0.0


def test_exact_double_root_zero():
    r1, r2 = characteristic_roots(1.0, 0.0, 0.0)
    assert r1 == 0.0 and r2 == 0.0


def test_double_root_up_to_rounding_is_distinguished():
    A, B, C = SINE_SETS["double"]
    # Mathematically B^2 = AC = pi^4/4; in floating point the discriminant is only
    # rounding noise, not an exact zero, and the computed roots may split.
    disc = B * B - A * C
    assert abs(disc) <= 8 * np.finfo(float).eps * B * B
    r1, r2 = characteristic_roots(A, B, C)
    assert abs(r1 - r2) <= 4 * math.sqrt(np.finfo(float).eps) * abs(r1)
    # Either sign of the rounding-level discriminant gives the same accurate solution.
    for c in (C * (1 - 4 * np.finfo(float).eps), C, C * (1 + 4 * np.finfo(float).eps)):
        grid, f, rho, result = sine_case(1.0, 640, (A, B, c))
        assert np.max(np.abs(result.y - rho * f)) <= A1A_TOL
        assert_a6(result.diagnostics)


@pytest.mark.parametrize("length", LENGTHS)
def test_near_double_complex_roots(length):
    A, B, C = NEAR_DOUBLE
    assert B * B < A * C
    r1, r2 = characteristic_roots(A, B, C)
    assert r1 == r2.conjugate() and r1.imag != 0.0
    assert r1.real >= 0.0
    for n in ALL_GRIDS:
        grid, f, rho, result = sine_case(length, n, NEAR_DOUBLE)
        assert np.max(np.abs(result.y - rho * f)) / np.max(np.abs(f)) <= A1A_TOL
        assert_a6(result.diagnostics)


@pytest.mark.parametrize("name", list(SINE_SETS) + ["near-double"])
def test_accuracy_on_finest_grid_640(name):
    coeffs = NEAR_DOUBLE if name == "near-double" else SINE_SETS[name]
    grid, f, rho, result = sine_case(1.0, 640, coeffs)
    relative_error = np.max(np.abs(result.y - rho * f)) / np.max(np.abs(f))
    d = result.diagnostics
    assert relative_error <= A1A_TOL
    assert d.eta <= ETA_MAX and d.imag_ratio <= IMAG_MAX
    assert d.refinements <= 1


# T-1D-2: polynomial (section 13.2) -------------------------------------------------


def poly_exact(y, length):
    return (y**4 - 2 * length * y**3 + length**3 * y) / 24.0


def poly_rhs(y, length, A, B, C, D):
    return D * (A - 2 * B * (y**2 - length * y) / 2 + C * poly_exact(y, length))


@pytest.mark.parametrize("name", POLY_SETS)
def test_polynomial_order_and_richardson(name):
    A, B, C = POLY_SETS[name]
    length, D = 1.0, 1.0
    solutions, errors = {}, {}
    for n in MAIN_GRIDS:
        grid = Grid1D(length, n)
        y = grid.nodes[1:-1]
        exact = poly_exact(y, length)
        result = solve_k(grid, A, B, C, D, poly_rhs(y, length, A, B, C, D))
        assert_a6(result.diagnostics)
        full = grid.with_boundary(result.y)
        assert full[0] == 0.0 and full[-1] == 0.0  # SSSS: principal condition exact
        solutions[n] = full
        errors[n] = np.max(np.abs(result.y - exact)) / np.max(np.abs(exact))

    # A-2a: error decreases from n = 20; order on (80,160), (160,320).
    assert all(errors[n] > errors[2 * n] for n in MAIN_GRIDS[:-1])
    for n in (80, 160):
        p = math.log2(errors[n] / errors[2 * n])
        assert A2A_WINDOW[0] <= p <= A2A_WINDOW[1], (n, p)

        # A-2b: empirical Richardson criterion (D-032) on the nodes of grid n.
        exact_n = poly_exact(Grid1D(length, n).nodes, length)
        fine_on_coarse = solutions[2 * n][::2]
        extrapolated = (4 * fine_on_coarse - solutions[n]) / 3
        assert np.max(np.abs(extrapolated - exact_n)) <= A2B_RATIO * np.max(np.abs(fine_on_coarse - exact_n))


# A-2v: SSSS boundary row ----------------------------------------------------------


@pytest.mark.parametrize("n", [4, 20])
@pytest.mark.parametrize("coeffs", [(1.0, 0.0, 0.0), (1.0, 3.0, 5.0), (0.5, PI**2 / 2, PI**4 / 2)])
@pytest.mark.parametrize("D", [1.0, 2.3])
def test_k_first_column_matches_sss_row(n, coeffs, D):
    A, B, C = coeffs
    grid = Grid1D(1.0, n)
    h = grid.h
    e1 = np.zeros(n - 1)
    e1[0] = 1.0
    column = apply_k(grid, A, B, C, D, e1)
    expected = np.zeros(n - 1)
    expected[:3] += A * np.array([5.0, -4.0, 1.0]) / h**4
    expected[:2] += 2 * B * np.array([2.0, -1.0]) / h**2
    expected[0] += C
    expected *= D
    assert np.max(np.abs(column - expected)) <= A2V_TOL * np.max(np.abs(expected))
    assert column[0] != pytest.approx(D * (A * 7.0 / h**4 + 4 * B / h**2 + C))  # not the clamped row


# Residual, imaginary part, no assembly of K (A-6) -----------------------------


def dense_k(grid, A, B, C, D):
    """Independent dense K for checks only; the solver never assembles it."""
    size = grid.interior_size
    lap = (np.diag(-2.0 * np.ones(size)) + np.diag(np.ones(size - 1), 1) + np.diag(np.ones(size - 1), -1)) / grid.h**2
    return D * (A * lap @ lap - 2 * B * lap + C * np.eye(size))


@pytest.mark.parametrize("coeffs", [(1.0, 3.0, 5.0), NEAR_DOUBLE])
def test_backward_error_formula(coeffs):
    A, B, C = coeffs
    D = 1.7
    grid = Grid1D(1.3, 20)
    rng = np.random.default_rng(3)
    y = rng.standard_normal(grid.interior_size)
    g = rng.standard_normal(grid.interior_size)
    residual = g - dense_k(grid, A, B, C, D) @ y
    h = grid.h
    expected = np.max(np.abs(residual)) / (D * (16 * A / h**4 + 8 * B / h**2 + C) * np.max(np.abs(y)) + np.max(np.abs(g)))
    eta, residual_norm = backward_error(grid, A, B, C, D, y, g)
    assert residual_norm == pytest.approx(np.max(np.abs(residual)), rel=1e-12)
    assert eta == pytest.approx(expected, rel=1e-12)
    np.testing.assert_allclose(apply_k(grid, A, B, C, D, y), dense_k(grid, A, B, C, D) @ y, rtol=0, atol=1e-12 * np.max(np.abs(dense_k(grid, A, B, C, D) @ y)))


def test_random_rhs_solution_satisfies_dense_system():
    grid = Grid1D(2.0, 40)
    A, B, C, D = NEAR_DOUBLE + (1.4,)
    g = np.random.default_rng(5).standard_normal(grid.interior_size)
    result = solve_k(grid, A, B, C, D, g)
    assert_a6(result.diagnostics)
    exact = np.linalg.solve(dense_k(grid, A, B, C, D), g)
    assert np.max(np.abs(result.y - exact)) <= 1e-10 * np.max(np.abs(exact))


def test_imaginary_part_is_controlled(monkeypatch):
    grid, f, rho, result = sine_case(1.0, 80, NEAR_DOUBLE)
    assert 0.0 <= result.diagnostics.imag_ratio <= IMAG_MAX

    original = ls._factored_solve

    def with_imaginary_noise(*args):
        y = original(*args)
        return y + 1e-6j * np.max(np.abs(y)) * np.ones_like(y)

    monkeypatch.setattr(ls, "_factored_solve", with_imaginary_noise)
    with pytest.raises(LinearSolveError) as info:
        sine_case(1.0, 80, NEAR_DOUBLE)
    assert info.value.code == ERROR_LINEAR_SOLVE
    assert info.value.diagnostics.imag_ratio > IMAG_MAX
    assert info.value.diagnostics.eta <= ETA_MAX  # rejected for the imaginary part alone


def test_solver_does_not_assemble_k():
    n = 4000
    grid = Grid1D(1.0, n)
    g = np.sin(PI * np.arange(1, n) / n)
    tracemalloc.start()
    try:
        result = solve_k(grid, *NEAR_DOUBLE, 1.0, g)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert_a6(result.diagnostics)
    dense_bytes = (n - 1) ** 2 * 8
    assert peak < dense_bytes / 50  # banded work arrays only, O(n) memory


# Failures and refinement --------------------------------------------------------


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_nonfinite_rhs(bad):
    grid = Grid1D(1.0, 20)
    g = np.ones(grid.interior_size)
    g[5] = bad
    with pytest.raises(LinearSolveError) as info:
        solve_k(grid, 1.0, 3.0, 5.0, 1.0, g)
    assert info.value.code == ERROR_NONFINITE


def test_nonfinite_computed_solution_is_not_accepted(monkeypatch):
    monkeypatch.setattr(ls, "_factored_solve", lambda grid, *args: np.full(grid.interior_size, np.nan, dtype=complex))
    with pytest.raises(LinearSolveError) as info:
        sine_case(1.0, 20, (1.0, 3.0, 5.0))
    assert info.value.code == ERROR_NONFINITE


def counting_solver(monkeypatch, corrupt_calls):
    """Wrap the internal factored solve; corrupt the result on the given call numbers."""
    original = ls._factored_solve
    calls = []

    def wrapped(*args):
        calls.append(1)
        y = original(*args)
        if len(calls) in corrupt_calls:
            y = y + 1e-3 * np.max(np.abs(y)) * np.cos(np.arange(y.size))
        return y

    monkeypatch.setattr(ls, "_factored_solve", wrapped)
    return calls


def test_persistent_failure_gives_error_after_one_refinement(monkeypatch):
    calls = counting_solver(monkeypatch, corrupt_calls={1, 2, 3, 4})
    with pytest.raises(LinearSolveError) as info:
        sine_case(1.0, 40, (1.0, 3.0, 5.0))
    assert info.value.code == ERROR_LINEAR_SOLVE
    assert info.value.diagnostics.eta > ETA_MAX
    assert info.value.diagnostics.refinements == 1
    assert len(calls) == 2  # initial solve plus exactly one refinement


def test_single_refinement_recovers(monkeypatch):
    calls = counting_solver(monkeypatch, corrupt_calls={1})
    grid, f, rho, result = sine_case(1.0, 40, (1.0, 3.0, 5.0))
    assert result.diagnostics.refinements == 1 and result.diagnostics.refined
    assert len(calls) == 2
    assert_a6(result.diagnostics)
    assert np.max(np.abs(result.y - rho * f)) <= A1A_TOL


def test_no_refinement_needed_normally():
    _, _, _, result = sine_case(1.0, 320, (1.0, 3.0, 5.0))
    assert result.diagnostics.refinements == 0 and not result.diagnostics.refined


# Input validation ----------------------------------------------------------------


@pytest.mark.parametrize(
    "coeffs",
    [(0.0, 0.0, 0.0, 1.0), (-1.0, 0.0, 0.0, 1.0), (1.0, -0.1, 0.0, 1.0), (1.0, 0.0, -0.1, 1.0), (1.0, 0.0, 0.0, 0.0), (1.0, math.nan, 0.0, 1.0), (1.0, 0.0, math.inf, 1.0)],
)
def test_rejects_invalid_coefficients(coeffs):
    grid = Grid1D(1.0, 8)
    with pytest.raises(ValueError):
        solve_k(grid, *coeffs, np.ones(7))


def test_rejects_non_numeric_coefficient_and_wrong_shape():
    grid = Grid1D(1.0, 8)
    with pytest.raises(TypeError):
        solve_k(grid, True, 0.0, 0.0, 1.0, np.ones(7))
    with pytest.raises(ValueError):
        solve_k(grid, 1.0, 0.0, 0.0, 1.0, np.ones(8))
