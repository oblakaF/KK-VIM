"""Program checks of the QA reference B (kkvim_reference.r5_cccc_ritz): Ritz method of L-120 (Taylor & Govindjee).

Checks of the implementation; the convergence of B to omega_0 is an OBSERVATION (no strict bound for the centre).
"""

import math

import numpy as np
import pytest

from kkvim_reference import r5_cccc_ritz as rz
from kkvim_reference.r5_cccc_tables import L120_TABLE_I, L120_TABLE_II_WORK

U = 2.0**-53


@pytest.mark.parametrize("a, b, M, N", [(1.0, 1.0, 5, 5), (1.0, 1.7, 6, 9), (2.0, 0.6, 7, 4)])
def test_p7_system_equals_the_own_gram_derivation(a, b, M, N):
    """K of P7 (11)-(12) = Ritz matrix of (7) from own 1D Gram integrals / (4 pi^4); rounding of a few operations."""
    p = rz.RitzProblem(a, b, M, N)
    K7, KG = rz.p7_matrix(p), rz.gram_matrix(p)
    assert np.max(np.abs(K7 - KG)) <= 8 * U * np.max(np.abs(K7))
    assert np.array_equal(K7, K7.T)


@pytest.mark.parametrize("a, b, M, N", [(1.0, 1.0, 8, 8), (1.0, 1.7, 6, 9), (2.0, 0.6, 12, 5)])
def test_smw_equals_the_direct_solution(a, b, M, N):
    """(14)-(17) is an exact transformation of the same system: both solutions agree within cond(K) u (backward stable)."""
    p = rz.RitzProblem(a, b, M, N)
    direct, smw = rz.solve_direct(p), rz.solve_smw(p)
    cond = np.linalg.cond(rz.p7_matrix(p))
    assert np.max(np.abs(direct.w - smw.w)) <= 16 * cond * U * np.max(np.abs(direct.w))
    assert rz.residual(smw) <= 16 * cond * U


@pytest.mark.parametrize("M", [200, 400, 600, 800, 1000])
def test_reproduces_l120_table_I_at_the_same_M(M):
    """Program check (task T2): the same procedure as L-120 gives the printed digits of table I (report p. 6) at the same M:
    w D/(q a^4) 1e3 and E Work 1e4 to 9 decimals, the condition number of the scaled capacity matrix to 4 decimals."""
    s = rz.solve_smw(rz.RitzProblem(1.0, 1.0, M, M))
    w, E, kappa = L120_TABLE_I[M]
    assert round(s.alpha() * 1e3, 9) == w
    assert round(s.work_normalized() * 1e4, 9) == E
    assert round(s.capacity_cond, 4) == kappa


@pytest.mark.parametrize("r", [1.2, 2.0])
def test_reproduces_l120_table_II_work(r):
    """Table II, E Work 1e4 to 8 decimals at 2000 x 2000 (as printed: 'solving 2000 x 2000 equations')."""
    s = rz.solve_smw(rz.RitzProblem(1.0, r, 2000, 2000))
    assert round(s.work_normalized() * 1e4, 8) == L120_TABLE_II_WORK[r]


def test_ritz_work_is_a_monotone_lower_bound_for_nested_bases():
    """Ritz property: the work does not decrease when the basis grows (exact arithmetic); rounding 4u W per value."""
    works = [rz.solve_smw(rz.RitzProblem(1.0, 1.5, M, M)).work() for M in range(4, 64, 4)]
    for w1, w2 in zip(works, works[1:]):
        assert w2 >= w1 - 4 * U * w2


def test_centre_value_symmetry_and_boundary():
    s = rz.solve_smw(rz.RitzProblem(1.0, 1.0, 20, 20))
    c = s.coefficients()
    assert np.max(np.abs(c - c.T)) <= 16 * U * np.max(np.abs(c))  # square: w_mn = w_nm
    assert s.value(np.array([0.5]), np.array([0.5]))[0] == pytest.approx(s.centre(), rel=64 * U)
    xs = np.linspace(0.0, 1.0, 9)
    edge = s.value(xs, np.ones_like(xs))
    assert np.max(np.abs(edge)) <= 64 * 400 * U * s.centre()  # 1 - cos(2 pi n) is zero up to the rounding of cos
    q = rz.solve_smw(rz.RitzProblem(2.0, 1.0, 20, 20, D=3.0, q=5.0))
    assert q.alpha() == pytest.approx(rz.solve_smw(rz.RitzProblem(1.0, 0.5, 20, 20)).alpha(), rel=64 * U)
