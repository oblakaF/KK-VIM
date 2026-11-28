"""R6 moment references (kkvim_reference.r6_moments): implementation checks (E_impl) and reviewed regression values.

The reviewed values (R6 review 4.2, 5.5, 7) are regression values of the reviewed derivations, not new criteria.
The Navier-Levy check uses the strict tail bounds plus the rounding models as the allowance of two independent
implementations (a test of the implementation); it does not decide RM-3 or qualify the reference (OD-4 is the owner's).
"""

import math

import numpy as np
import pytest

from kkvim_reference import r5_cccc_ritz as ritz
from kkvim_reference import r6_moments as rm


# ------------------------------------------------------------------------------------------------ stable forms


def test_stable_k_forms_equal_the_direct_formulas_at_moderate_alpha():
    rng = np.random.default_rng(5)
    worst = 0.0
    for _ in range(400):
        alpha = rng.uniform(math.pi / 2, 25.0)
        u = rng.uniform(-alpha, alpha)
        p = abs(u)
        z = alpha - p
        _, Ku, Kuu = rm.levy_k_direct(alpha, u)
        a = np.array([[alpha]])
        got_uu = float(rm._k_uu(a, np.array([[z]]), np.array([[p]]))[0, 0])
        got_u = math.copysign(1.0, u) * float(rm._k_u(a, np.array([[z]]), np.array([[p]]))[0, 0])
        worst = max(worst, abs(got_uu - Kuu), abs(got_u - Ku))
    assert worst <= 1e-12


def test_stable_forms_have_exact_edge_and_centre_line_values():
    alpha = np.array([[7.3]])
    assert float(rm._k_uu(alpha, np.array([[0.0]]), np.array([[7.3]]))[0, 0]) == 0.0
    assert float(rm._k_u(alpha, np.array([[7.3]]), np.array([[0.0]]))[0, 0]) == 0.0


# ------------------------------------------------------------------------------------------------ Navier: reviewed values


@pytest.mark.parametrize("r,mx,my", [(0.2, 0.0015098, 0.0049850), (0.5, 0.0115876, 0.0254208), (1.0, 0.0478864, 0.0478864),
                                     (2.0, 0.1016831, 0.0463503), (5.0, 0.1246245, 0.0377453)])
def test_navier_centre_moments_match_the_reviewed_table(r, mx, my):
    c = rm.navier_moments_points(1.0, r, 1.0, 0.3, [0.5], [0.5 * r])
    assert float(c["Mx"][0]) == pytest.approx(mx, abs=6e-8)
    assert float(c["My"][0]) == pytest.approx(my, abs=6e-8)
    assert abs(float(c["Mxy"][0])) <= 1e-15


@pytest.mark.parametrize("r,n1,n2", [(0.2, 2.89e-8, 1.80e-9), (1.0, 6.49e-8, 3.45e-9), (5.0, 7.22e-7, 4.50e-8)])
def test_tail_bounds_at_m_2559_match_the_review(r, n1, n2):
    b1, b2 = rm.navier_tail_bounds(2559, r, 0.3)
    assert b1 == pytest.approx(n1, rel=5e-3) and b2 == pytest.approx(n2, rel=5e-3)


@pytest.mark.parametrize("r", [0.2, 5.0])
def test_reference_margin_is_5_22_not_5_3(r):
    n1, _ = rm.navier_tail_bounds(2559, r, 0.3)
    assert rm.delta_m(r) / n1 == pytest.approx(5.22, abs=0.01)


@pytest.mark.parametrize("r,expected", [(1.0, 0.064965), (2.0, 0.092534), (3.0, 0.094830), (5.0, 0.094982)])
def test_corner_force_2mxy_equals_the_reviewed_r_values(r, expected):
    c = rm.navier_moments_points(1.0, r, 1.0, 0.3, [1.0], [r])
    assert 2 * float(c["Mxy"][0]) == pytest.approx(expected, abs=1.5e-6)


def test_navier_nodes_have_zero_edge_moments_and_the_corner_twist_sign():
    out = rm.navier_moments_nodes(2.0, 1.0, 1000.0, 0.3, 16, 8, M=255)
    for key in ("Mx", "My"):
        F = out[key]
        assert np.all(F[[0, -1], :] == 0.0) and np.all(F[:, [0, -1]] == 0.0)
    assert out["Mxy"][-1, -1] > 0 and out["Mxy"][0, 0] > 0 and out["Mxy"][-1, 0] < 0
    assert np.all(out["Mxy"][8, :] == 0.0)  # cos(m pi / 2) = 0 exactly on x = a/2
    pts = rm.navier_moments_points(2.0, 1.0, 1000.0, 0.3, [0.375, 1.25], [0.25, 0.5], M=255)
    assert pts["Mx"][0] == pytest.approx(out["Mx"][3, 2], rel=1e-13)
    assert pts["Mxy"][1] == pytest.approx(out["Mxy"][10, 4], abs=1e-12 * abs(out["Mxy"]).max())


# ------------------------------------------------------------------------------------------------ Navier against Levy


@pytest.mark.parametrize("a,b", [(1.0, 1.0), (1.0, 2.0), (2.0, 1.0), (1.0, 5.0), (3.0, 0.75)])
def test_navier_and_levy_agree_within_their_bounds(a, b):
    xs = np.array([0.5 * a, 0.3 * a, 0.9 * a, a, 0.05 * a, 0.7 * a])
    ys = np.array([0.5 * b, 0.2 * b, 0.6 * b, b, 0.5 * b, 0.97 * b])
    q, nu = 2500.0, 0.3
    N = rm.navier_moments_points(a, b, q, nu, xs, ys, M=2559)
    L = rm.levy_moments_points(a, b, q, nu, xs, ys, M=6117)
    for key in ("Mx", "My", "Mxy"):
        allowance = N["tail_bound"][key] + L["tail_bound"][key] + N["rounding_model"][key] + 1e3 * 2.0**-53 * q * min(a, b) ** 2
        assert np.max(np.abs(N[key] - L[key])) <= allowance, key


def test_levy_edges_and_swapped_orientation():
    a, b = 2.0, 1.0  # b < a: the Levy frame runs along y
    L = rm.levy_moments_points(a, b, 1.0, 0.3, [0.0, 2.0, 0.7, 1.0], [0.4, 0.3, 0.0, 1.0], M=301)
    # at x_L = a_L the factor sin(m pi) is a rounded value (~1e-16 m), not an exact zero: rounding only
    assert L["swapped"] and np.all(np.abs(L["Mx"]) <= 1e-14) and np.all(np.abs(L["My"]) <= 1e-14)
    assert L["Mx"][0] == 0.0 and L["My"][2] == 0.0  # the edges y_L = 0 (s = 0: T = K_uu = 0 exactly) and x_L = 0


def test_levy_bounds():
    b = rm.levy_uniform_bounds(6117, 0.3)
    assert b["Mx"] <= 1e-9 and b["My"] <= 1e-9 and b["Mxy"] <= 1e-9
    inner = rm.levy_interior_bounds(9, 0.3, 1.0, [0.0, 0.5, 1.0])
    assert inner["Mx"][0] is None and inner["Mx"][2] < inner["Mx"][1]


# ------------------------------------------------------------------------------------------------ CCCC: Ritz, D2


def test_ritz_moments_reviewed_values_and_edge_identities():
    sol = ritz.solve_smw(ritz.RitzProblem(1.0, 1.0, 160, 160))
    out = rm.ritz_moments(sol, [0.5, 0.0, 0.0], [0.5, 0.5, 0.0], 0.3)
    assert out["Mx"][0, 0] == pytest.approx(0.0229040, abs=2e-7)  # review 7, M = N = 160
    assert out["Mx"][1, 1] == pytest.approx(-0.0510552, abs=2e-7)  # slow (~1/M) at the edge: OBSERVATION only
    assert out["My"][1, 1] == pytest.approx(0.3 * out["Mx"][1, 1], rel=1e-12)  # w_yy = 0 along x = 0
    assert abs(out["Mxy"][1, 1]) <= 1e-12 and abs(out["Mxy"][2, 2]) <= 1e-12
