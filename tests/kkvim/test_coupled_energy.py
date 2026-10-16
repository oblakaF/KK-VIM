"""Own tests of kkvim.coupled_energy: energy diagnostics of coupled half-steps (R4, step 16).

R4_NUMERICAL_SPEC 6.2 (ChV4-14) and 6.3 (ChV4-15). The two-dimensional discrete problem is
assembled here independently (dense Delta_h) to obtain omega_h on small grids. The factor
10 and the resolution rule are NOT_QUALIFIED for N >= 2; statuses are research diagnostics.
"""

import math

import numpy as np
import pytest

from kkvim.coupled import coupled_first_halfstep, coupled_second_halfstep
from kkvim.coupled_energy import (
    E1_E2_HOLD,
    E1_VIOLATED,
    NOT_APPLICABLE,
    UNRESOLVED,
    eps_a,
    field_from_sets,
    first_energy,
    omega_bounds,
    pair_energy,
)
from kkvim.grid import Grid1D

U = 2.0**-53
D, Q = 3.1, 2.3


def gamma(n):
    return n * U / (1 - n * U)


def dense_L(grid):
    m = grid.interior_size
    return (np.diag(-2.0 * np.ones(m)) + np.diag(np.ones(m - 1), 1) + np.diag(np.ones(m - 1), -1)) / grid.h**2


def lap2d(gx, gy):
    return np.kron(dense_L(gx), np.eye(gy.interior_size)) + np.kron(np.eye(gx.interior_size), dense_L(gy))


def omega_h(gx, gy):
    """Discrete exact minimizer of Pi_h: D Delta_h^2 omega = q 1 on the interior (row-major x, y)."""
    Lap = lap2d(gx, gy)
    return np.linalg.solve(D * Lap @ Lap, Q * np.ones(Lap.shape[0]))


def a_norm(gx, gy, inner):
    Lap = lap2d(gx, gy)
    v = Lap @ inner.reshape(-1)
    return math.sqrt(D * gx.h * gy.h * float(v @ v))


def pair(a, b, n_x, n_y, N):
    first = coupled_first_halfstep(a, b, D, Q, n_x, n_y, N, transform="dense")
    second = coupled_second_halfstep(first, transform="dense")
    return first, second


def test_first_energy_is_the_two_dimensional_pi_h():
    first, _ = pair(1.0, 1.5, 10, 12, 2)
    gx, gy = first.grid_x, first.grid_y
    e = first_energy(gx, gy, first.x, first.y, D, Q)
    inner = field_from_sets(gx, gy, first.x, first.y)[1:-1, 1:-1]
    Lap = lap2d(gx, gy)
    v = Lap @ inner.reshape(-1)
    pi = 0.5 * D * gx.h * gy.h * float(v @ v) - Q * gx.h * gy.h * float(inner.sum())
    scale = 0.5 * D * gx.h * gy.h * float(np.abs(Lap) @ np.abs(inner.reshape(-1)) @ (np.abs(Lap) @ np.abs(inner.reshape(-1)))) \
        + Q * gx.h * gy.h * float(np.abs(inner).sum())
    assert abs(e.P - pi) <= 8 * gamma(inner.size + 16) * scale and e.P < 0
    assert e.status == NOT_APPLICABLE and e.tau is None


@pytest.mark.parametrize("a, b, n_x, n_y, N", [(1.0, 1.5, 10, 12, 2), (1.0, 0.5, 12, 8, 3), (2.0, 3.0, 8, 12, 3)])
def test_omega_bounds_bound_the_true_distance(a, b, n_x, n_y, N):
    first, second = pair(a, b, n_x, n_y, N)
    gx, gy = first.grid_x, first.grid_y
    omega = omega_h(gx, gy).reshape(gx.interior_size, gy.interior_size)
    w = field_from_sets(gx, gy, second.x, second.y)
    bw, bw_a = omega_bounds(gx, gy, w, D, Q)
    assert a_norm(gx, gy, w[1:-1, 1:-1] - omega) <= bw * (1 + 1e-6)
    assert a_norm(gx, gy, omega) <= bw_a * (1 + 1e-6)  # a priori bound of |omega_h|_{a_h} (6.3, 6(a))


def test_eps_a_formula():
    first, _ = pair(1.0, 2.0, 20, 40, 3)
    gx, gy = first.grid_x, first.grid_y
    S = np.abs(first.x) @ np.abs(first.y).T
    expected = math.sqrt(D) * (4 / gx.h**2 + 4 / gy.h**2) * gamma(3) * math.sqrt(gx.h * gy.h * float(np.sum(S * S)))
    assert math.isclose(eps_a(gx, gy, first.x, first.y, D), expected, rel_tol=8 * gamma(S.size))


def test_gamma_accounts_for_an_inexact_solution():
    """ChV4-14, item 3: Delta P = Q - gamma holds for any computed set, also a perturbed one."""
    first, second = pair(1.0, 2.0, 20, 40, 2)
    gx, gy = first.grid_x, first.grid_y
    rng = np.random.default_rng(1)
    received = second.x * (1 + 1e-6 * rng.normal(size=second.x.shape))
    e = pair_energy(gx, gy, first.x, first.y, received, first.y, D, Q, solve_grid=gx,
                    coefficients=second.step.coefficients, received=received, companion_prev=first.x)
    model = 10 * (e.R_prev + e.R + e.R_f)
    assert abs(e.delta_P - (e.Q - e.gamma)) <= model
    assert abs(e.gamma) > 10 * model  # the perturbation is visible only through gamma


def test_exact_half_step_energy_and_statuses():
    first, second = pair(1.0, 2.0, 20, 40, 3)
    gx, gy = first.grid_x, first.grid_y
    e = pair_energy(gx, gy, first.x, first.y, second.x, second.y, D, Q, solve_grid=gx,
                    coefficients=second.step.coefficients, received=second.x, companion_prev=first.x)
    assert e.delta_P > 0 and e.P < e.P_prev
    assert (e.status == UNRESOLVED) == (e.Q < 10 * e.tau)
    assert e.status in (UNRESOLVED, E1_E2_HOLD) and "PASS" not in e.status
    assert e.tau == abs(e.gamma) + 10 * (e.R_prev + e.R + e.R_f)
    assert e.R_f == e.eps_a * math.sqrt(2 * e.Q) + e.B_omega * (e.eps_a_prev + e.eps_a) or \
        math.isclose(e.R_f, e.eps_a * math.sqrt(2 * e.Q) + e.B_omega * (e.eps_a_prev + e.eps_a), rel_tol=16 * U)


def test_energy_increase_is_reported_as_e1_violation(monkeypatch):
    """Status logic: with gamma = 0 (residual replaced by zero) a resolved energy increase violates E1."""
    import kkvim.coupled_energy as ce

    first, second = pair(1.0, 2.0, 20, 40, 2)
    gx, gy = first.grid_x, first.grid_y
    monkeypatch.setattr(ce, "coupled_residual", lambda grid, c, Dv, V: (np.zeros_like(V), 0.0))
    # reversed order: the "new" field is the worse one, so Pi_h increases by Q
    e = pair_energy(gx, gy, second.x, second.y, first.x, first.y, D, Q, solve_grid=gx,
                    coefficients=second.step.coefficients, received=first.x, companion_prev=second.x)
    assert e.gamma == 0.0 and e.delta_P < 0 and e.Q >= 10 * e.tau
    assert e.status == E1_VIOLATED and e.e1 is False
