"""Program checks of the QA reference A (kkvim_reference.r5_cccc_hencky): Hencky-Inglis finite sections.

Checks of the implementation only. The accuracy of A relative to omega_0 has no strict bound (OV-M1...M3): every value of A
is an OBSERVATION (D-049.8, item 2). Error models: kappa_inf <= (1 + rho)/(1 - rho) with rho = max row sum of the section.
"""

import math

import numpy as np
import pytest
import scipy.linalg

from kkvim_reference import r5_cccc_hencky as hk

U = 2.0**-53


def kappa(solution):
    rho = solution.row_sum_max
    return (1 + rho) / (1 - rho)


@pytest.mark.parametrize("lx, ly, NX, NY", [(1.0, 1.0, 30, 30), (2.0, 1.0, 40, 20), (1.0, 3.0, 20, 60)])
def test_section_is_nonnegative_with_row_sums_below_p5_bound(lx, ly, NX, NY):
    """K >= 0 elementwise and the row sums of the section below 0.77 (P5 (3.7), the authors' bound; a check of the assembly)."""
    K, f, _, _ = hk.section(hk.HenckyProblem(lx, ly, NX, NY))
    assert np.all(K >= 0) and np.max(K.sum(axis=1)) < 0.77
    assert np.all(f[:NX] < 0) and np.all(f[NX:] > 0)  # F_m = -P b/(a alpha_m^2) < 0, H_l > 0


def test_system_equals_batista_26_27_after_the_change_of_variables():
    """L-058 (26)-(27) assembled independently with B~_n, D~_n (RM 2.4, T-7): B~_n = -b^2 X_{n+1}/alpha_{n+1},
    D~_n = a^2 Y_{n+1}/beta_{n+1} (Reviewer R-9)."""
    p = hk.HenckyProblem(1.4, 1.0, 28, 20, D=2.0, q=3.0)
    a, b, qD = p.a, p.b, p.q / p.D
    al, be = p.alphas(), p.betas()  # alpha^Bat_n = alpha^MG_{n+1}
    den = (al[:, None] ** 2 + be[None, :] ** 2) ** 2
    n1, n2 = p.NX, p.NY
    S = np.zeros((n1 + n2, n1 + n2))
    rhs = np.zeros(n1 + n2)
    # (26): (a/b) B_n (tanh + t/cosh^2)(alpha_n b) = -(4/a) sum_m D_m alpha_n beta_m^2/den + (q b/D)/alpha_n^3
    S[:n1, :n1] = np.diag(a / b * hk._delta(al * b))
    S[:n1, n1:] = 4.0 / a * al[:, None] * be[None, :] ** 2 / den
    rhs[:n1] = qD * b / al**3
    # (27): (b/a) D_n (...)(beta_n a) = -(4/b) sum_m B_m alpha_m^2 beta_n/den + (q a/D)/beta_n^3
    S[n1:, n1:] = np.diag(b / a * hk._delta(be * a))
    S[n1:, :n1] = 4.0 / b * (al[:, None] ** 2 * be[None, :] / den).T
    rhs[n1:] = qD * a / be**3
    z = scipy.linalg.solve(S, rhs)
    X = -z[:n1] * al / b**2
    Y = z[n1:] * be / a**2
    s = hk.solve(p)
    scale = max(np.max(np.abs(s.X)), np.max(np.abs(s.Y)))
    bound = 64 * kappa(s) * np.linalg.cond(S, np.inf) * U * scale
    assert np.max(np.abs(X - s.X)) <= bound and np.max(np.abs(Y - s.Y)) <= bound


def test_square_antisymmetry_and_series_consistency():
    s = hk.solve(hk.HenckyProblem(1.0, 1.0, 60, 60))
    assert np.max(np.abs(s.X + s.Y)) <= 64 * kappa(s) * U * np.max(np.abs(s.X))  # Y = -X for a = b (RM 2.5, item 4)
    assert s.residual <= 64 * kappa(s) * U
    c = s.value(np.array([0.5]), np.array([0.5]))[0]
    assert c == pytest.approx(s.centre(), rel=256 * U)  # field series (A-1) at the centre = centre formula (RM 4.2)
    xs = np.linspace(0.0, 1.0, 21)
    for edge in (s.value(xs, np.zeros_like(xs)), s.value(np.ones_like(xs), xs)):
        assert np.max(np.abs(edge)) <= 256 * (s.problem.NX + s.problem.NY) * U * s.centre()  # w = 0 on the contour


def test_load_and_size_scaling():
    """alpha depends only on the side ratio; work scales with q l^6/D."""
    s1 = hk.solve(hk.HenckyProblem(1.0, 2.0, 20, 40))
    s2 = hk.solve(hk.HenckyProblem(3.0, 6.0, 20, 40, D=2.5, q=0.7))
    s3 = hk.solve(hk.HenckyProblem(2.0, 1.0, 40, 20))  # sides exchanged: the same plate
    for other in (s2, s3):
        assert other.alpha() == pytest.approx(s1.alpha(), rel=256 * kappa(s1) * U)
        assert other.work_normalized() == pytest.approx(s1.work_normalized(), rel=256 * kappa(s1) * U)


def test_no_overflow_for_long_plates_and_many_terms():
    s = hk.solve(hk.default_section(20.0, 1.0, 200))
    assert all(math.isfinite(v) for v in (s.alpha(), s.work(), s.residual))
    assert round(s.alpha() * 1e3, 9) == 2.604166667  # L-120 table II, r = 20, Hencky: printed digits (program check)
