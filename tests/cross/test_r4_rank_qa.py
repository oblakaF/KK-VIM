"""QA-δ3 (R4_NUMERICAL_SPEC 5.2, 5.4.1; РВ4-3): rank diagnostics on synthetic factors with an exactly known rank.

The exact rank of the product of the stored double factors is confirmed by rational elimination. The assertions are
(a) exact facts (rank, invariance in exact arithmetic up to rounding), (b) the strict storage component of delta_f,
(c) regression guards of the recorded observations rho_hat ~ u kappa_rep. Item (c) is NOT a value of c_d or of any
normative threshold (c_d, c_rank, tau_set, delta_f remain NOT_QUALIFIED; no rank is certified here).
"""

import numpy as np
import pytest

from kkvim_reference.r4_rank_diagnostics import (
    cancellation_set,
    dense_transform,
    diagnostics,
    exact_rank,
    storage_delta_f,
    tiny_rank_set,
)

U = 2.0**-53


@pytest.mark.parametrize("N", (2, 3))
@pytest.mark.parametrize("K", (1.0, 1e4, 1e8, 1e12))
def test_exact_rank_deficient_fields_with_exact_factors(N, K):
    """delta_f = 0 (exact integer factors): rho_hat > 0 only from the diagnostics, of order u kappa_rep."""
    ratios = []
    for seed in range(10):
        Uf, Vf = cancellation_set(12, 10, K, seed, N=N)
        assert exact_rank(Uf, Vf) == N - 1
        d = diagnostics(Uf, Vf)
        ratios.append(d.rho_N / (U * d.kappa_rep))
        assert d.kappa_rep >= 0.5 * K  # strong cancellation: kappa_rep grows with K
    assert max(ratios) <= 4.0  # regression guard of the observation (report, section 5), not c_d


@pytest.mark.parametrize("e", (10, 30, 50))
def test_exact_full_rank_distinguishable_with_small_kappa(e):
    Uf, Vf = tiny_rank_set(12, 10, e)
    assert exact_rank(Uf, Vf) == 3
    d = diagnostics(Uf, Vf)
    assert d.kappa_rep < 2.0
    assert abs(d.rho_N - d.sigma_full[-1] / d.sigma_full[0]) <= 1e-3 * d.rho_N + 4 * U  # QR path = full SVD path


@pytest.mark.parametrize("e", (30, 40, 50))
def test_same_exact_full_rank_field_becomes_indistinguishable_with_large_kappa(e):
    """The field of tiny_rank_set rewritten with T (cond 1e8): factors rounded on storage, kappa_rep ~ 1e8."""
    Uf, Vf = tiny_rank_set(12, 10, e)
    rho_exact = diagnostics(Uf, Vf).rho_N
    T = dense_transform(3, 1e8, 1)
    d = diagnostics(Uf @ T, Vf @ np.linalg.inv(T).T)
    assert d.kappa_rep > 1e7
    floor = storage_delta_f(d.kappa_rep, 3)  # strict storage part only
    assert rho_exact < floor  # the exact sigma_3 is below the storage part alone: cannot be resolved
    assert abs(d.rho_N - rho_exact) > 0.1 * rho_exact  # and the computed value is not the exact one


def test_reviewer_type_one_sided_rank_deficiency():
    """Exact rank-2 field written with 3 terms through a dense ill-conditioned T, the deficiency in ONE factor only
    (R4_NUMERICAL_SPEC 5.2, item 8): rho_hat of order u kappa_rep (first order)."""
    for cond in (4e2, 2.8e5, 9e8):
        rng = np.random.default_rng(3)
        P, Q = rng.standard_normal((20, 3)), np.column_stack([rng.standard_normal((16, 2)), np.zeros(16)])
        T = dense_transform(3, cond, 3)
        d = diagnostics(P @ T, Q @ np.linalg.inv(T).T)
        assert d.rho_N <= 4.0 * U * d.kappa_rep + storage_delta_f(d.kappa_rep, 3)
        assert d.rho_N >= 1e-3 * U * d.kappa_rep  # first order, not O(u^2)


def test_gl_n_invariance_of_rho_up_to_rounding():
    rng = np.random.default_rng(7)
    P, Q = rng.standard_normal((30, 3)), rng.standard_normal((25, 3))
    base = diagnostics(P, Q)
    for cond in (1.0, 1e3, 1e6):
        T = dense_transform(3, cond, 2)
        d = diagnostics(P @ T, Q @ np.linalg.inv(T).T)
        assert abs(d.rho_N - base.rho_N) <= 8 * U * d.kappa_rep + storage_delta_f(d.kappa_rep, 3)


def test_column_scaling_keeps_the_field_and_changes_the_set_check():
    Uf, Vf = tiny_rank_set(12, 10, 10)
    s = np.array([1.0, 2.0**-30, 2.0**30])  # exact powers of two: the product is unchanged exactly
    d0, d1 = diagnostics(Uf, Vf), diagnostics(Uf * s, Vf / s)
    assert np.array_equal(Uf @ Vf.T, (Uf * s) @ (Vf / s).T)
    assert abs(d0.rho_N - d1.rho_N) <= 1e-12 * d0.rho_N + 8 * U * d1.kappa_rep
    assert abs(d1.set_sigma_min - d0.set_sigma_min) <= 1e-12  # D_c normalises the column scales of the set
