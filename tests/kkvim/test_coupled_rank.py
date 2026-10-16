"""Own tests of kkvim.coupled_rank: rho_N, kappa_rep, set diagnostics and the QR (R4, step 16).

R4_NUMERICAL_SPEC 5.2 (ChV4-12), 5.4 (ChV4-13), 5.4.1, 5.7. Bounds are rounding models
(gamma_n, kappa_rep, condition numbers); the constants c_d, c_rank, tau_set and delta_f
stay NOT_QUALIFIED and are not chosen here. Builder unit tests, not QA.
"""

import math

import numpy as np
import pytest

from kkvim.coupled import coupled_coefficients, odd_sine_set
from kkvim.coupled_energy import field_from_sets
from kkvim.coupled_rank import mode_conditions, orthogonalize_received, rank_diagnostics
from kkvim.energy import discrete_energy, rounding_threshold
from kkvim.grid import Grid1D

U = 2.0**-53
D, Q = 3.1, 2.3


def gamma(n):
    return n * U / (1 - n * U)


def smooth_set(grid, N, seed, scales=None):
    rng = np.random.default_rng(seed)
    x = grid.nodes[1:-1] / grid.length
    basis = np.column_stack([np.sin(m * math.pi * x) for m in range(1, 7)])
    S = basis @ rng.normal(size=(6, N))
    return S * (np.ones(N) if scales is None else np.asarray(scales))


@pytest.mark.parametrize("n_u, n_v, N, seed", [(20, 40, 2, 1), (33, 17, 3, 2), (64, 320, 3, 3)])
def test_singular_values_equal_those_of_the_weighted_field(n_u, n_v, N, seed):
    gu, gv = Grid1D(1.0, n_u), Grid1D(2.0, n_v)
    Uset, Vset = smooth_set(gu, N, seed), smooth_set(gv, N, seed + 10)
    r = rank_diagnostics(gu, Uset, gv, Vset)
    full = np.linalg.svd(math.sqrt(gu.h * gv.h) * (Uset @ Vset.T), compute_uv=False)[:N]
    scale = np.linalg.norm(math.sqrt(gu.h) * Uset, 2) * np.linalg.norm(math.sqrt(gv.h) * Vset, 2)
    assert np.all(np.abs(r.sigma - full) <= 64 * gamma(n_u + n_v + N) * scale)
    assert r.rho_N == r.sigma[-1] / r.sigma[0]
    assert 1.0 - 8 * gamma(n_u + n_v) <= r.kappa_rep <= min(r.cond_U, r.cond_V) * (1 + 8 * gamma(n_u + n_v))


def test_rho_is_gl_n_invariant_and_kappa_is_not():
    gu, gv = Grid1D(1.0, 30), Grid1D(1.5, 45)
    Uset, Vset = smooth_set(gu, 3, 4), smooth_set(gv, 3, 5)
    T = np.array([[1.0, 2.0, 0.5], [0.0, 1.0, 3.0], [0.2, 0.0, 1.0]])
    r0 = rank_diagnostics(gu, Uset, gv, Vset)
    r1 = rank_diagnostics(gu, Uset @ T, gv, Vset @ np.linalg.inv(T).T)
    kap = max(r0.kappa_rep, r1.kappa_rep) * np.linalg.cond(T)
    assert abs(r1.rho_N - r0.rho_N) <= 64 * gamma(30 + 45 + 3) * kap
    assert r1.kappa_rep != r0.kappa_rep  # kappa_rep depends on the representation (5.2, item 5(b))


@pytest.mark.parametrize("n_x, N", [(4, 2), (20, 3), (320, 3)])
def test_kappa_rep_is_one_for_the_odd_sine_start(n_x, N):
    gx, gy = Grid1D(1.0, n_x), Grid1D(2.0, 40)
    r = rank_diagnostics(gx, odd_sine_set(gx, N), gy, smooth_set(gy, N, 6))
    assert abs(r.kappa_rep - 1.0) <= 16 * gamma(n_x + 40)
    assert abs(r.cond_U - 1.0) <= 16 * gamma(n_x)


def test_exact_rank_deficient_field_stays_below_the_research_guard():
    """Field of exact rank N - 1 in N terms (5.2, item 8): rho_N is at the rounding level u kappa_rep."""
    gu, gv = Grid1D(1.0, 40), Grid1D(1.0, 40)
    base = smooth_set(gu, 2, 7)
    Uset = np.column_stack([base, base[:, 0] + base[:, 1]])
    Vset = smooth_set(gv, 3, 8)
    r = rank_diagnostics(gu, Uset, gv, Vset)
    assert r.rho_N <= 1e6 * U * r.kappa_rep  # the research guard K u kappa_rep would stop (K = 1e6)
    received = np.column_stack([Vset[:, :2], Vset[:, 0] - 2.0 * Vset[:, 1]])
    r2 = rank_diagnostics(gu, smooth_set(gu, 3, 9), gv, received)
    assert r2.set_sigma_min <= 1e6 * U


def test_badly_scaled_but_independent_set_has_a_normalized_set_check():
    gu, gv = Grid1D(1.0, 40), Grid1D(1.0, 40)
    Vset = smooth_set(gv, 3, 10, scales=[1.0, 1e-6, 1e-12])
    r = rank_diagnostics(gu, odd_sine_set(gu, 3), gv, Vset)
    assert r.cond_V > 1e10 and r.set_sigma_min > 1e-3  # D_c removes the column scales (5.4.1, (ii))


@pytest.mark.parametrize("n_r, n_c, N, seed", [(40, 20, 2, 11), (36, 48, 3, 12), (320, 64, 3, 13)])
def test_qr_keeps_field_span_and_energy(n_r, n_c, N, seed):
    g_rec, g_comp = Grid1D(2.0, n_r), Grid1D(1.0, n_c)
    received, companion = smooth_set(g_rec, N, seed, scales=[1.0, 1e-3, 1e-5][:N]), smooth_set(g_comp, N, seed + 1)
    r = rank_diagnostics(g_comp, companion, g_rec, received)
    o = orthogonalize_received(g_rec, received, companion)
    assert np.array_equal(o.companion, companion @ o.R.T)  # multiplication, no inverse of R
    gram = g_rec.h * o.received.T @ o.received
    assert np.max(np.abs(gram - np.eye(N))) <= 16 * gamma(n_r + N)
    assert o.field_error <= 64 * gamma(n_r + n_c + N) * r.kappa_rep
    # span: principal angles between span(received) and span(new) are at the rounding level times cond
    Qa, _ = np.linalg.qr(received)
    Qb, _ = np.linalg.qr(o.received)
    sin_max = np.linalg.norm(Qb - Qa @ (Qa.T @ Qb), 2)
    assert sin_max <= 64 * gamma(n_r + N) * r.cond_V
    # energy of the same field before and after (two-dimensional Pi_h of R2)
    gx, gy = g_comp, g_rec
    w0, w1 = field_from_sets(gx, gy, companion, received), field_from_sets(gx, gy, o.companion, o.received)
    e0, e1 = discrete_energy(gx, gy, w0, D, Q), discrete_energy(gx, gy, w1, D, Q)
    e_d = discrete_energy(gx, gy, w1 - w0, D, Q)
    bound = (e_d.norm_a * (e0.norm_a + 0.5 * e_d.norm_a) + abs(Q) * gx.h * gy.h * float(np.sum(np.abs(w1 - w0)))
             + rounding_threshold(gx, gy, w0, D, Q) + rounding_threshold(gx, gy, w1, D, Q))
    assert abs(e1.pi - e0.pi) <= bound


def test_qr_uses_no_explicit_inverse(monkeypatch):
    for name in ("inv", "solve", "pinv", "lstsq"):
        monkeypatch.setattr(np.linalg, name, lambda *a, **k: pytest.fail("explicit inverse used"))
    g = Grid1D(1.0, 20)
    orthogonalize_received(g, smooth_set(g, 2, 14), smooth_set(Grid1D(1.0, 30), 2, 15))


def test_mode_conditions_of_the_odd_sine_start():
    gx, gy = Grid1D(1.0, 20), Grid1D(2.0, 40)
    c = coupled_coefficients(gx, odd_sine_set(gx, 3), Q)
    cond_M, cond_A = mode_conditions(c, gy)
    assert abs(cond_A - 1.0) <= 16 * gamma(20) and 1.0 <= cond_M < math.inf
