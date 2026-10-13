"""QA tests: discrete energy Pi_h and the official energy test A-7 of kkvim (R2, step 7).

QA tests (D-033, item 8). NUMERICAL_SPEC.md 4.1 and 10.2-10.4 are implemented here
independently of kkvim.energy (own 2D assembly of Pi_h, own S_hat, R(w), R_f, gamma with a
matrix-free residual, tau, E1, E2, resolution and status) and compared with
kkvim.energy.check_history on the same half-step histories (kkvim.iteration.run_halfsteps).

A-7 (table 18.1, 10.4), fixed before the comparison:
- s = 1: NOT_APPLICABLE;
- s = 2: resolved (Q >= 10 tau) and PASS on all verification grids n = 20..320;
- s = 3: resolved and PASS for n <= 160;
- everywhere else: PASS if resolved, otherwise UNRESOLVED; FAIL nowhere.
Tolerances of the comparison of own and kkvim quantities: Pi_h relative 1e-12; Q, R, R_f
relative 1e-9; tau relative 1e-6; statuses equal.
"""

import dataclasses
import math
from functools import lru_cache

import numpy as np
import pytest

from kkvim.coefficients import ERROR_DEGENERATE
from kkvim.energy import ERROR_ENERGY_TEST, EnergyError, check_history, check_pair
from kkvim.iteration import run_halfsteps

U = 2.0**-53
MAIN_GRIDS = (20, 40, 80, 160, 320)


def laplacian(w, hx, hy):
    c = w[1:-1, 1:-1]
    return (w[:-2, 1:-1] - 2 * c + w[2:, 1:-1]) / hx**2 + (w[1:-1, :-2] - 2 * c + w[1:-1, 2:]) / hy**2


def abs_laplacian(w, hx, hy):
    a = np.abs(w)
    c = a[1:-1, 1:-1]
    return (a[:-2, 1:-1] + 2 * c + a[2:, 1:-1]) / hx**2 + (a[1:-1, :-2] + 2 * c + a[1:-1, 2:]) / hy**2


def energy(w, hx, hy, D, q):
    """(Pi_h, a_h(w, w)) by NUMERICAL_SPEC.md 4.1."""
    a = D * hx * hy * math.fsum((laplacian(w, hx, hy) ** 2).ravel())
    return 0.5 * a - q * hx * hy * math.fsum(w[1:-1, 1:-1].ravel()), a


def rounding_model(w, hx, hy, D, q):
    """R(w) = u (ceil(log2 N) + 16) S_hat(w), 10.3."""
    n_nodes = (w.shape[0] - 2) * (w.shape[1] - 2)
    s_hat = D * hx * hy * math.fsum((np.abs(laplacian(w, hx, hy)) * abs_laplacian(w, hx, hy)).ravel())
    s_hat += abs(q) * hx * hy * math.fsum(np.abs(w[1:-1, 1:-1]).ravel())
    return U * (math.ceil(math.log2(n_nodes)) + 16) * s_hat


def residual(f, h, A, B, C, D, F):
    """rho = F 1 - D [A L(L f) - 2B L f + C f], L with zero end values (matrix-free)."""
    full = np.concatenate(([0.0], f, [0.0]))
    lf = np.concatenate(([0.0], (full[:-2] - 2 * full[1:-1] + full[2:]) / h**2, [0.0]))
    llf = (lf[:-2] - 2 * lf[1:-1] + lf[2:]) / h**2
    return F - D * (A * llf - 2 * B * lf[1:-1] + C * f)


def own_check(prev, cur):
    hx, hy, D, q = cur.grid_x.h, cur.grid_y.h, cur.D, cur.q
    w_prev = np.outer(np.concatenate(([0.0], prev.x, [0.0])), np.concatenate(([0.0], prev.y, [0.0])))
    w_now = np.outer(np.concatenate(([0.0], cur.x, [0.0])), np.concatenate(([0.0], cur.y, [0.0])))
    P_prev, _ = energy(w_prev, hx, hy, D, q)
    P_now, a_now = energy(w_now, hx, hy, D, q)
    d = w_prev - w_now
    _, a_d = energy(d, hx, hy, D, q)
    Q = 0.5 * a_d
    h, f_prev = (hy, prev.y) if cur.direction == "Y" else (hx, prev.x)
    c = cur.coefficients
    gamma = h * math.fsum(residual(cur.raw, h, c.A, c.B, c.C, D, c.F) * (f_prev - cur.raw))
    R_prev, R_now = rounding_model(w_prev, hx, hy, D, q), rounding_model(w_now, hx, hy, D, q)
    R_f = 4 * U * (max(cur.grid_x.n, cur.grid_y.n) / math.pi) ** 2 * math.sqrt(a_now) * math.sqrt(a_d)
    tau = abs(gamma) + 10 * (R_prev + R_now + R_f)
    delta_P = P_prev - P_now
    e1, e2, resolved = delta_P >= -tau, abs(delta_P - Q) <= tau, Q >= 10 * tau
    status = "UNRESOLVED" if not resolved else ("PASS" if e1 and e2 else "FAIL")
    return dict(P_prev=P_prev, P_now=P_now, Q=Q, R_prev=R_prev, R_now=R_now, R_f=R_f, tau=tau, status=status, e1=e1, e2=e2)


@lru_cache(maxsize=None)
def history(n_x, n_y, q=1.0, count=4):
    return run_halfsteps(1.0, 1.0, q, n_x, n_y, count)


@lru_cache(maxsize=None)
def checks(n_x, n_y, q=1.0, count=4):
    return check_history(history(n_x, n_y, q, count))


def relative(a, b):
    return abs(a - b) / abs(b)


@pytest.mark.parametrize("n", MAIN_GRIDS)
@pytest.mark.parametrize("s", [2, 3, 4])
def test_own_recomputation_agrees_with_kkvim(n, s):
    b = checks(n, n)[s - 1]
    o = own_check(history(n, n).record(s - 1), history(n, n).record(s))
    assert b.status == o["status"]
    assert relative(b.P_now, o["P_now"]) <= 1e-12 and relative(b.P_prev, o["P_prev"]) <= 1e-12
    for name in ("Q", "R_prev", "R_now", "R_f"):
        assert relative(getattr(b, name), o[name]) <= 1e-9, name
    assert relative(b.tau, o["tau"]) <= 1e-6


@pytest.mark.parametrize("n", MAIN_GRIDS)
def test_a7_mandatory_conditions(n):
    cks = checks(n, n)
    assert cks[0].s == 1 and cks[0].status == "NOT_APPLICABLE"
    assert cks[1].s == 2 and cks[1].status == "PASS" and cks[1].Q >= 10 * cks[1].tau
    if n <= 160:
        assert cks[2].s == 3 and cks[2].status == "PASS" and cks[2].Q >= 10 * cks[2].tau
    for c in cks[1:]:
        assert c.status in ("PASS", "UNRESOLVED")
        assert (c.status == "PASS") == (c.Q >= 10 * c.tau)


def test_a7_diagnostic_grid_640():
    cks = checks(640, 640, 1.0, 3)
    assert cks[1].status == "PASS"
    assert all(c.status in ("PASS", "UNRESOLVED") for c in cks[1:])


@pytest.mark.parametrize("n_x, n_y, q", [(20, 40, 1.0), (40, 20, -2.3), (21, 20, 1.0), (7, 9, -0.7)])
def test_a7_unequal_grids_and_negative_load(n_x, n_y, q):
    cks = checks(n_x, n_y, q)
    assert cks[1].status == "PASS" and cks[2].status == "PASS"
    for s in (2, 3, 4):
        o = own_check(history(n_x, n_y, q).record(s - 1), history(n_x, n_y, q).record(s))
        assert cks[s - 1].status == o["status"] != "FAIL"


def test_error_energy_test_on_inconsistent_assembly():
    """Resolved E1 violation: the factor is moved away from the minimizer and F_h in gamma is moved with it."""
    r2, r3 = history(40, 40).record(2), history(40, 40).record(3)
    raw = 1.6 * r3.raw
    bad = dataclasses.replace(r3, raw=raw, y=raw / r3.c, coefficients=dataclasses.replace(r3.coefficients, F=1.6 * r3.coefficients.F))
    o = own_check(r2, bad)
    assert o["status"] == "FAIL" and not o["e1"]
    with pytest.raises(EnergyError) as info:
        check_pair(r2, bad)
    assert info.value.code == ERROR_ENERGY_TEST


def test_consistent_inexact_factor_gives_no_false_fail():
    r2, r3 = history(40, 40).record(2), history(40, 40).record(3)
    raw = 1.6 * r3.raw  # consistent coefficients: the defect is carried by gamma
    c = check_pair(r2, dataclasses.replace(r3, raw=raw, y=raw / r3.c))
    assert c.e1 and c.e2 and c.status in ("PASS", "UNRESOLVED")


@pytest.mark.parametrize("which", ["previous", "current"])
def test_error_degenerate_for_nonnegative_energy(which):
    r2, r3 = history(20, 20).record(2), history(20, 20).record(3)
    if which == "current":
        pair = (r2, dataclasses.replace(r3, raw=2.5 * r3.raw, y=2.5 * r3.raw / r3.c))  # Pi = (3.125 - 2.5) l > 0
    else:
        pair = (dataclasses.replace(r2, y=2.5 * r2.y), r3)  # y is not the fixed factor of s = 3
    assert own_check(*pair)["P_now" if which == "current" else "P_prev"] >= 0.0
    with pytest.raises(EnergyError) as info:
        check_pair(*pair)
    assert info.value.code == ERROR_DEGENERATE
