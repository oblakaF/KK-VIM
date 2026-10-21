"""QA tests: stopping controller of kkvim and the official A-11 (R2, step 8).

QA tests (D-033, item 8). NUMERICAL_SPEC.md 11.1-11.4 is checked with own expectations and
own computations: the pure stopping rule on synthetic sequences; eps_inf, eps_E, eps_L2, theta
and T_s from the full fields with an own two-dimensional energy norm; the strict parameters;
A-11 on full fields.

A-11 (table 18.1, 11.4), fixed before the comparison:
S_stop(n) = max|w_strict - w_normal| / max|w_normal(n)|;
E_disc(n) = max over the common nodes |w_n - w_2n| / max|w_n|, normal mode, nested grids, the
pair (n, 2n) for n <= 320 and (320, 640) for the top level 640;
S_stop <= 0.01 E_disc for n = 20, 40, 80, 160, 320 and the additional 640.
The half-step numbers of the stops are not part of the criterion.
"""

import math
import statistics
from functools import lru_cache

import numpy as np
import pytest

from kkvim.controller import (
    CONVERGED,
    MAX_HALF_STEPS,
    STAGNATED,
    run_until_stopped,
    stopping_decision,
    strict_parameters,
)
from kkvim.iteration import run_halfsteps

U = 2.0**-53
TOL_INF, TOL_E = 1e-10, 1e-8  # defaults of 11.2
GRIDS = (20, 40, 80, 160, 320, 640)


# Pure rule (11.3) on synthetic sequences --------------------------------------------------


def decide(eps, s_min=3, s_max=100):
    rows = [(s, e_inf, e_E) for s, (e_inf, e_E) in zip(range(2, 2 + len(eps)), eps)]
    return stopping_decision(1 + len(rows), rows, TOL_INF, TOL_E, s_min, s_max)


BIG, OK = (1e-3, 1e-3), (1e-11, 1e-9)
STAG = [(1e-3, 1e-3 * 0.95**j) for j in range(12)]


def test_converged_needs_both_tolerances_on_two_consecutive_half_steps():
    assert decide([BIG, OK, OK]) == CONVERGED
    assert decide([BIG, BIG, OK]) is None and decide([BIG, OK, BIG]) is None
    assert decide([BIG, (1e-11, 2e-8), (1e-11, 2e-8)]) is None
    assert decide([BIG, (2e-10, 1e-9), (2e-10, 1e-9)]) is None


def test_s_min_is_respected():
    assert decide([OK, OK], s_min=4) is None
    assert decide([OK, OK, OK], s_min=4) == CONVERGED


def test_stagnation_after_ten_defined_theta_and_reset():
    assert decide(STAG[:10]) is None and decide(STAG[:11]) == STAGNATED
    reset = STAG[:6] + [(1e-3, STAG[5][1] * 0.5)] + [(1e-3, STAG[5][1] * 0.5 * 0.95**j) for j in range(1, 10)]
    assert decide(reset) is None
    assert decide(reset + [(1e-3, reset[-1][1] * 0.95)]) == STAGNATED


def test_undefined_theta_and_small_eps_e_do_not_stagnate():
    zero = STAG[:5] + [(1e-3, 0.0)] + [(1e-3, 1e-3 * 0.95**j) for j in range(9)]
    assert decide(zero) is None  # theta after the zero is undefined and restarts the series
    assert decide([(1e-3, 1e-9 * 0.95**j) for j in range(14)]) is None  # eps_E <= tol_E


def test_max_half_steps_and_priorities():
    assert decide([BIG] * 4, s_max=5) == MAX_HALF_STEPS and decide([BIG] * 3, s_max=5) is None
    assert decide([BIG, OK, OK], s_max=4) == CONVERGED
    assert decide(STAG[:11], s_max=12) == STAGNATED


# eps, theta, T_s from the full fields (11.1, 11.3) ------------------------------------------


def norm_a(w, hx, hy, D):
    c = w[1:-1, 1:-1]
    lap = (w[:-2, 1:-1] - 2 * c + w[2:, 1:-1]) / hx**2 + (w[1:-1, :-2] - 2 * c + w[1:-1, 2:]) / hy**2
    return math.sqrt(D * hx * hy * math.fsum((lap**2).ravel()))


@pytest.mark.parametrize("n_x, n_y, q", [(20, 20, 1.0), (20, 40, 1.0), (40, 20, -2.3), (21, 20, -0.7)])
def test_eps_theta_tail_from_full_fields(n_x, n_y, q):
    D = 1.3
    res = run_until_stopped(1.0, D, q, n_x, n_y)
    h, hx, hy = res.history, 1.0 / n_x, 1.0 / n_y
    eps_E = []
    for change in res.changes:
        w_prev, w_now = h.field(change.s - 1), h.field(change.s)
        d = w_now - w_prev
        e_inf = np.max(np.abs(d)) / np.max(np.abs(w_now))
        e_E = norm_a(d, hx, hy, D) / norm_a(w_now, hx, hy, D)  # norm of the difference field
        e_L2 = math.sqrt(math.fsum((d * d).ravel())) / math.sqrt(math.fsum((w_now * w_now).ravel()))
        assert abs(change.eps_inf - e_inf) <= 1e-12 * e_inf
        assert abs(change.eps_E - e_E) <= 1e-12 * e_E
        assert abs(change.eps_L2 - e_L2) <= 1e-12 * e_L2
        eps_E.append(e_E)
    theta = [None] + [b / a for a, b in zip(eps_E, eps_E[1:])]
    for j, change in enumerate(res.changes):
        assert (change.theta is None) == (theta[j] is None)
        if theta[j] is not None:
            assert abs(change.theta - theta[j]) <= 1e-10 * theta[j]
        last3 = theta[max(0, j - 2): j + 1]
        if len(last3) == 3 and None not in last3 and statistics.median(last3) < 1.0:
            m = statistics.median(last3)
            assert abs(change.tail_estimate - eps_E[j] * m / (1 - m)) <= 1e-9 * abs(change.tail_estimate)
        else:
            assert change.tail_estimate is None
    assert "не доказанная" in res.tail_estimate_note


# Strict mode and configuration (11.2) ---------------------------------------------------------


@pytest.mark.parametrize("n", GRIDS)
def test_strict_parameters(n):
    floor = 10 * U * (n / math.pi) ** 2
    assert strict_parameters(n, n) == (TOL_INF / 100, max(TOL_E / 100, 10 * floor))
    if n >= 320:
        assert strict_parameters(n, n)[1] == 10 * floor > TOL_E / 100  # lower bound active


def test_unattainable_tol_e_is_error_input_without_grid_limit():
    r = run_until_stopped(1.0, 1.0, 1.0, 320, 320, tol_E=1e-10)  # 1e-10 < 10 eps_floor(320)
    assert r.status == "ERROR_INPUT" and r.s == 0
    assert run_until_stopped(1.0, 1.0, 1.0, 2000, 7, s_max=3).status == MAX_HALF_STEPS


# A-11 (11.4) ----------------------------------------------------------------------------------


@lru_cache(maxsize=None)
def normal(n):
    return run_until_stopped(1.0, 1.0, 1.0, n, n)


@lru_cache(maxsize=None)
def strict(n):
    tol_inf, tol_E = strict_parameters(n, n)
    return run_until_stopped(1.0, 1.0, 1.0, n, n, tol_inf=tol_inf, tol_E=tol_E)


def e_disc(n):
    coarse, fine = (n, 2 * n) if n < 640 else (320, 640)
    w_c = normal(coarse).last_field()
    w_f_common = normal(fine).last_field()[::2, ::2]  # nodes of the coarse grid
    return np.max(np.abs(w_c - w_f_common)) / np.max(np.abs(w_c))


@pytest.mark.parametrize("n", GRIDS)
def test_a11_stopping_does_not_distort_grid_convergence(n):
    assert normal(n).status == CONVERGED and strict(n).status == CONVERGED
    w_normal, w_strict = normal(n).last_field(), strict(n).last_field()
    s_stop = np.max(np.abs(w_strict - w_normal)) / np.max(np.abs(w_normal))
    assert s_stop <= 0.01 * e_disc(n)


def test_controller_fields_equal_fixed_count_run():
    r = normal(40)
    fixed = run_halfsteps(1.0, 1.0, 1.0, 40, 40, r.s)
    for s in range(1, r.s + 1):
        np.testing.assert_array_equal(r.history.field(s), fixed.field(s))
