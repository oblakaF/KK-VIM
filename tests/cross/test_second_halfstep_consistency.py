"""QA consistency tests of the second half-step s = 2 (X2, field w2) of kkvim (R2, step 5).

QA tests (D-033, item 8). The discrete X-half-step system is assembled here
independently of kkvim (own coefficient sums, own dense L_x and K_x, NumPy dense
solve), following docs/NUMERICAL_SPEC.md 5.3, 7.1, 8, 9.1 and D-030. No energy test
(A-7) and no comparison with the continuous X2 (A-5a/b/v): the semi-analytic
recursion of the reference is not integrated yet.

Tolerances (fixed before the comparison):
- coefficients against own sums: relative 1e-13 (level of A-3, table 18.1);
- X2_raw against the dense solve: 64 cond_2(K_x) u (rounding of the dense LU, 3.5);
- residual of the stored X2_raw with the own K_x: eta <= 1e-12 (A-6, 8.2);
- A-5g at s = 2, n = 80: relative 1e-12 (table 18.1);
- normalization: exact product of the normalized factors within 2u of the raw one (9.1).
"""

import math
from fractions import Fraction
from functools import lru_cache

import numpy as np
import pytest

from kkvim.first_halfstep import first_halfstep, solve_halfstep
from kkvim.second_halfstep import second_halfstep

U = 2.0**-53
COEFF_RTOL = 1e-13
ETA_MAX = 1e-12
A5G_TOL = 1e-12
PAIRS = ((7, 7), (20, 20), (20, 40), (40, 20), (21, 20), (20, 21), (7, 9))
LOADS = (1.0, -2.3)


@lru_cache(maxsize=None)
def computed(n_x, n_y, q, a=1.0, D=1.0):
    first = first_halfstep(a, D, q, n_x, n_y)
    return first, second_halfstep(first)


def own_coefficients(f, h, q):
    full = np.concatenate(([0.0], np.asarray(f, dtype=float), [0.0]))
    n = full.size - 1
    A = h * math.fsum(v * v for v in full[1:-1])
    B = h * math.fsum(((full[i + 1] - full[i]) / h) ** 2 for i in range(n))  # all n intervals
    second = [(full[i - 1] - 2.0 * full[i] + full[i + 1]) / h**2 for i in range(1, n)]
    C = h * math.fsum(v * v for v in second)
    F = q * h * math.fsum(full[1:-1])
    return A, B, C, F


def dense_k(n, h, A, B, C, D):
    m = n - 1
    lap = (np.diag(-2.0 * np.ones(m)) + np.diag(np.ones(m - 1), 1) + np.diag(np.ones(m - 1), -1)) / h**2
    return D * (A * lap @ lap - 2.0 * B * lap + C * np.eye(m))


@pytest.mark.parametrize("q", LOADS)
@pytest.mark.parametrize("n_x, n_y", PAIRS)
def test_coefficients_of_normalized_y1_on_grid_y(n_x, n_y, q):
    first, r = computed(n_x, n_y, q)
    expected = own_coefficients(first.y1_normalized, first.grid_y.h, q)
    got = (r.coefficients.A, r.coefficients.B, r.coefficients.C, r.coefficients.F)
    for value, exp in zip(got, expected):
        assert abs(value - exp) <= COEFF_RTOL * abs(exp)


@pytest.mark.parametrize("q", LOADS)
@pytest.mark.parametrize("n_x, n_y", PAIRS)
def test_x2_against_independent_dense_system_on_grid_x(n_x, n_y, q):
    first, r = computed(n_x, n_y, q)
    A, B, C, F = own_coefficients(first.y1_normalized, first.grid_y.h, q)
    h_x = first.grid_x.h
    K = dense_k(n_x, h_x, A, B, C, first.D)
    rhs = F * np.ones(n_x - 1)
    x_dense = np.linalg.solve(K, rhs)
    assert r.x2_raw.shape == (n_x - 1,)
    tol = 64 * np.linalg.cond(K) * U
    assert np.max(np.abs(r.x2_raw - x_dense)) <= tol * np.max(np.abs(x_dense))
    residual = rhs - K @ r.x2_raw
    eta = np.max(np.abs(residual)) / (
        first.D * (16 * A / h_x**4 + 8 * B / h_x**2 + C) * np.max(np.abs(r.x2_raw)) + abs(F)
    )
    assert eta <= ETA_MAX
    # Field w2 = X2 (x) Y1_normalized (axis 0 = x) against the independent solution.
    w2_ind = np.outer(np.concatenate(([0.0], x_dense, [0.0])), np.concatenate(([0.0], first.y1_normalized, [0.0])))
    assert np.max(np.abs(r.w2 - w2_ind)) <= (tol + 4 * U) * np.max(np.abs(w2_ind))


@pytest.mark.parametrize("q", LOADS)
@pytest.mark.parametrize("n_x, n_y", PAIRS)
def test_shape_boundary_sign_counters_and_kept_w1(n_x, n_y, q):
    first, r = computed(n_x, n_y, q)
    w = r.w2
    assert w.shape == (n_x + 1, n_y + 1)
    assert not w[0].any() and not w[-1].any() and not w[:, 0].any() and not w[:, -1].any()
    assert np.all(np.sign(w[1:-1, 1:-1]) == math.copysign(1.0, q))
    assert (r.s, r.k) == (2, 1) and r.k == r.s // 2  # A-5d, D-030
    assert r.w1 is first.w1 and not np.array_equal(r.w1, r.w2)


@pytest.mark.parametrize("q", LOADS)
@pytest.mark.parametrize("n_x, n_y", ((20, 40), (21, 20), (7, 9)))
def test_normalization_i4_keeps_field(n_x, n_y, q):
    first, r = computed(n_x, n_y, q)
    assert abs(np.max(np.abs(r.x2_normalized)) - 1.0) <= U and np.sum(r.x2_normalized) > 0.0
    assert (r.c2 > 0.0) == (q > 0.0)
    for i in range(n_x - 1):
        for j in range(n_y - 1):
            raw = Fraction(r.x2_raw[i]) * Fraction(first.y1_normalized[j])
            normalized = Fraction(r.x2_normalized[i]) * Fraction(r.y2_scaled[j])
            assert abs(normalized - raw) <= 2 * Fraction(U) * abs(raw)


def test_a5g_partial_scale_invariance_s2():
    """A-5g, s = 2 only (n = 80): X2 from Y1_raw instead of Y1_normalized gives the same field."""
    first, r = computed(80, 80, 1.0)
    alt = solve_halfstep(first.grid_y, first.y1_raw, first.grid_x, first.q, first.D)  # QA-only path
    w2_alt = np.outer(first.grid_x.with_boundary(alt.raw), first.grid_y.with_boundary(first.y1_raw))
    w2_main = np.outer(first.grid_x.with_boundary(r.x2_raw), first.grid_y.with_boundary(first.y1_normalized))
    assert np.max(np.abs(w2_alt - w2_main)) / np.max(np.abs(w2_main)) <= A5G_TOL
    assert np.max(np.abs(w2_alt - r.w2)) / np.max(np.abs(r.w2)) <= A5G_TOL
