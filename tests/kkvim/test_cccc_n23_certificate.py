"""The RN-5 certificate and the conditions RT-7 (``kkvim.cccc_n23_certificate``; D-065, 5 and 6) on the actual saved factors of a run.

Unit-level fixtures only: the negative controls outside the domain of the production command (a nearly dependent V, a basis of another span,
non-finite input) call the certificate directly and never pass through the input checks of the command. No tolerance is used: every
assertion is exact or a strict inequality of the certificate itself.
"""

import math
from fractions import Fraction

import numpy as np
import pytest

import kkvim.cccc_n23 as cn
import kkvim.cccc_n23_certificate as cc

CASES = [(1.0, 1.3, 8, 7), (1.0, 0.6, 10, 9)]  # C-2, C-1 of Policy-23, 8 (cases of the QA certificate RN5_CERTIFIED_LIMITED)


def factors(a, b, n_x, n_y, N):
    step = cn.run_cccc_n23(a, b, 1.0, 1.0, n_x, n_y, N, 1).steps[0]
    return step.G, step.V


# --- outward square roots (the hang of the QA routine at a tiny rational is gone) ---------------------------------------------------

@pytest.mark.parametrize("x", [Fraction(2), Fraction(1, 3), Fraction(10**40 + 7, 3), Fraction(4), Fraction(9, 16), 2.0, 7])
def test_sqrt_bounds_enclose_the_root_tightly(x):
    x = Fraction(x)
    up, dn = cc.sqrt_up(x), cc.sqrt_dn(x)
    assert dn * dn <= x <= up * up and dn <= up
    assert (up - dn) / up <= Fraction(1, 2**79)  # the bracket of the default precision; a property of the construction, not a tolerance


def test_sqrt_of_zero_is_zero():
    assert cc.sqrt_up(0) == 0 and cc.sqrt_dn(Fraction(0)) == 0


@pytest.mark.parametrize("x", [Fraction(2**-1074), Fraction(1, 2**1100), Fraction(3, 2**5000), Fraction(1, 3 * 10**900), Fraction(2**5000 + 1, 7)])
def test_sqrt_of_subnormal_sized_and_huge_rationals_terminates_with_an_outward_bound(x):
    up, dn = cc.sqrt_up(x), cc.sqrt_dn(x)
    assert 0 < dn and dn * dn <= x <= up * up
    assert x > 2**1000 or math.ldexp(float(x), 0) < 2.0**-1021  # the inputs that made float(x) zero or subnormal in the QA routine


@pytest.mark.parametrize("x, error", [(-1, ValueError), (Fraction(-1, 3), ValueError), (math.nan, ValueError), (math.inf, OverflowError),
                                      (None, TypeError), ("2", TypeError), (True, TypeError), (1j, TypeError)])
def test_sqrt_refuses_invalid_input(x, error):
    with pytest.raises(error):
        cc.sqrt_up(x)
    with pytest.raises(error):
        cc.sqrt_dn(x)


# --- two exact rank procedures (R7-1) ------------------------------------------------------------------------------------------------

def test_the_two_rank_procedures_agree_on_known_ranks():
    rng = np.random.default_rng(3)
    M = rng.standard_normal((9, 3))
    dependent = np.column_stack([M[:, 0], M[:, 1], M[:, 0]])
    for X, rank in ((M, 3), (dependent, 2), (np.zeros((5, 2)), 0)):
        exact = cc.exact_matrix(X)
        assert cc.rank_fraction(exact) == rank and cc.rank_modp(exact) == rank
    with pytest.raises(cc.CertificateInputError):
        cc.rank_modp([[Fraction(1, 3)]])  # not dyadic: no stored double


@pytest.mark.parametrize("X", [np.array([[1.0, math.nan]]), np.array([[math.inf, 1.0]]), np.ones(3), "x"])
def test_an_exact_matrix_is_a_finite_real_matrix(X):
    with pytest.raises(cc.CertificateInputError):
        cc.exact_matrix(X)


# --- RN-5 ------------------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("N", [2, 3])
@pytest.mark.parametrize("a, b, n_x, n_y", CASES)
def test_rn5_certifies_the_actual_factors(a, b, n_x, n_y, N):
    G, V = factors(a, b, n_x, n_y, N)
    out = cc.rn5_certificate(a, b, n_x, n_y, N, G, V, 1.0, 1.0)
    assert out["certified"] and out["verdict"] == "RN5_CERTIFIED" and all(out["hypotheses"].values())
    values = out["values"]
    assert values["lambda_low"] > values["norm_dK"] > 0 and values["eps"] == values["eps1"] + values["eps2"] > 0
    assert out["sigma_low"] > out["eps"] and out["sigma_low"] ** 2 > out["eps"] ** 2 and values["margin_sigma_over_eps"] > 1
    assert all(isinstance(value, Fraction) for value in values.values())  # every bound is rational, none a float estimate


def test_rn5_with_the_physical_d_and_q_of_the_run_certifies():
    step = cn.run_cccc_n23_physical(1.0, 1.3, 0.01, 2.1e11, 0.3, 1000.0, 8, 7, 3, 1)
    first = step.steps[0]
    out = cc.rn5_certificate(1.0, 1.3, 8, 7, 3, first.G, first.V, step.D, 1000.0)
    assert out["certified"]


# --- P0-A: the sign of the load (the closed-form check of F is within |q| h m rho) ---------------------------------------------------

@pytest.mark.parametrize("q", [1, -1, 1000, -1000])
@pytest.mark.parametrize("N", [2, 3])
@pytest.mark.parametrize("a, b, n_x, n_y", CASES)
def test_the_closed_form_of_f_is_checked_within_the_magnitude_of_q(a, b, n_x, n_y, N, q):
    """F_c - F* = q h sum_l (Tc - T*)_li with |Tc - T*| <= rho at the m nodes: the admissible radius |q| h m rho is >= 0 for either sign."""
    la, q = Fraction(a), Fraction(q)
    hx, m = la / n_x, n_x - 1
    Tc, rho = cc.tstar(n_x, N)
    F_c = cc.coefficients(Tc, la, n_x, q)[3]
    assert all(abs(F_c[i] - q * la / 2) <= abs(q) * hx * m * rho for i in range(N)) and rho > 0
    assert F_c == [-f for f in cc.coefficients(Tc, la, n_x, -q)[3]]  # the deviation of F_c from F* = q a / 2 is odd in q


@pytest.mark.parametrize("q", [1.0, 1000.0])
@pytest.mark.parametrize("N", [2, 3])
@pytest.mark.parametrize("a, b, n_x, n_y", CASES)
def test_rn5_is_the_same_for_opposite_loads(a, b, n_x, n_y, N, q):
    """The field of s = 1 is odd in q, the certificate even: G is the start set, V(-q) = -V(q) bit for bit, every bound of RN-5 is equal."""
    up = cn.run_cccc_n23(a, b, 1.0, q, n_x, n_y, N, 1).steps[0]
    down = cn.run_cccc_n23(a, b, 1.0, -q, n_x, n_y, N, 1).steps[0]
    assert np.array_equal(down.G, up.G) and np.array_equal(down.V, -up.V)
    plus = cc.rn5_certificate(a, b, n_x, n_y, N, up.G, up.V, 1.0, q)
    minus = cc.rn5_certificate(a, b, n_x, n_y, N, down.G, down.V, 1.0, -q)
    assert plus["certified"] and minus["certified"] and minus["verdict"] == "RN5_CERTIFIED" and all(minus["hypotheses"].values())
    assert minus["hypotheses"] == plus["hypotheses"] and minus["values"] == plus["values"]
    assert (minus["eps"], minus["sigma_low"]) == (plus["eps"], plus["sigma_low"])


def test_rn5_for_a_negative_load_does_not_certify_the_factors_of_the_opposite_load():
    G, V = factors(1.0, 1.3, 8, 7, 2)  # q = +1
    out = cc.rn5_certificate(1.0, 1.3, 8, 7, 2, G, V, 1.0, -1.0)
    assert out["hypotheses"]["closed forms consistent with the enclosure"]
    assert not out["certified"] and out["verdict"] == "RN5_UNRESOLVED" and out["eps"] is None and out["sigma_low"] is None


def test_rn5_for_a_negative_load_does_not_certify_a_nearly_dependent_v():
    step = cn.run_cccc_n23(1.0, 1.3, 1.0, -1.0, 8, 7, 2, 1).steps[0]
    other = step.V[:, 0].copy()
    other[2] = np.nextafter(other[2], -np.inf)
    out = cc.rn5_certificate(1.0, 1.3, 8, 7, 2, step.G, np.column_stack([step.V[:, 0], other]), 1.0, -1.0)
    assert not out["certified"] and out["verdict"] == "RN5_UNRESOLVED" and out["eps"] is None and out["sigma_low"] is None
    assert out["hypotheses"]["closed forms consistent with the enclosure"] and not out["hypotheses"]["sigma_N(w_hat_w)^2 > eps^2 (Weyl)"]


def test_rn5_does_not_certify_a_nearly_dependent_v():
    G, V = factors(1.0, 1.3, 8, 7, 2)
    other = V[:, 0].copy()
    other[2] = np.nextafter(other[2], np.inf)  # exact rank 2, sigma_2 far below the propagated error
    out = cc.rn5_certificate(1.0, 1.3, 8, 7, 2, G, np.column_stack([V[:, 0], other]), 1.0, 1.0)
    assert not out["certified"] and out["verdict"] == "RN5_UNRESOLVED" and out["eps"] is None and out["sigma_low"] is None
    assert out["hypotheses"]["lambda_low(K(U~)) > ||dK||_2"] and not out["hypotheses"]["sigma_N(w_hat_w)^2 > eps^2 (Weyl)"]


def test_rn5_does_not_certify_factors_that_are_not_those_of_the_system():
    G, V = factors(1.0, 1.3, 8, 7, 2)
    out = cc.rn5_certificate(1.0, 1.3, 8, 7, 2, G, 2.0 * V, 1.0, 1.0)  # another V: the exact error of V^ is as large as V itself
    assert not out["certified"]


@pytest.mark.parametrize("n_x, n_y, N", [(12, 4, 3), (10, 4, 3), (6, 9, 3)])
def test_rn5_refuses_outside_its_hypotheses(n_x, n_y, N):
    out = cc.rn5_certificate(1.0, 1.0, n_x, n_y, N, np.ones((n_x - 1, N)), np.ones((n_y - 1, N)), 1.0, 1.0)
    assert not out["certified"] and out["verdict"] == "RN5_HYPOTHESIS_NOT_ESTABLISHED" and not all(out["hypotheses"].values())


@pytest.mark.parametrize("damage", ["nan", "inf", "shape"])
def test_rn5_refuses_invalid_input(damage):
    G, V = factors(1.0, 1.3, 8, 7, 2)
    V = V.copy()
    if damage == "shape":
        V = V[:-1]
    else:
        V[1, 1] = math.nan if damage == "nan" else math.inf
    out = cc.rn5_certificate(1.0, 1.3, 8, 7, 2, G, V, 1.0, 1.0)
    assert not out["certified"] and out["verdict"] == "RN5_INVALID_INPUT" and out["eps"] is None


def test_rn5_that_cannot_be_completed_is_not_certified(monkeypatch):
    G, V = factors(1.0, 1.3, 8, 7, 2)
    monkeypatch.setattr(cc, "band_ldl", lambda *a, **k: (_ for _ in ()).throw(ZeroDivisionError("injected")))
    out = cc.rn5_certificate(1.0, 1.3, 8, 7, 2, G, V, 1.0, 1.0)
    assert not out["certified"] and out["verdict"] == "RN5_NOT_COMPLETED" and "injected" in out["reason"]


def test_rn5_without_a_proved_lower_eigenvalue_is_not_certified(monkeypatch):
    G, V = factors(1.0, 1.3, 8, 7, 2)
    monkeypatch.setattr(cc, "_lambda_low", lambda *a: None)
    out = cc.rn5_certificate(1.0, 1.3, 8, 7, 2, G, V, 1.0, 1.0)
    assert not out["certified"] and out["verdict"] == "RN5_UNRESOLVED"


def test_rn5_with_a_lower_eigenvalue_not_above_the_perturbation_is_not_certified(monkeypatch):
    G, V = factors(1.0, 1.3, 8, 7, 2)
    monkeypatch.setattr(cc, "_lambda_low", lambda *a: Fraction(0))  # K > 0 proved, but lambda_low = 0 <= ||dK||_2
    out = cc.rn5_certificate(1.0, 1.3, 8, 7, 2, G, V, 1.0, 1.0)
    assert not out["certified"] and out["verdict"] == "RN5_UNRESOLVED" and not out["hypotheses"]["lambda_low(K(U~)) > ||dK||_2"]


def test_rn5_with_a_nonfinite_float_estimate_falls_back_to_exact_checks(monkeypatch):
    G, V = factors(1.0, 1.3, 8, 7, 2)
    monkeypatch.setattr(cc.np.linalg, "eigvalsh", lambda *a: np.array([math.nan]))
    out = cc.rn5_certificate(1.0, 1.3, 8, 7, 2, G, V, 1.0, 1.0)
    assert not out["certified"]  # no trial shift: lambda_low = 0 at best, which never exceeds ||dK||


# --- RT-7 ------------------------------------------------------------------------------------------------------------------------------

def certified(a=1.0, b=1.3, n_x=8, n_y=7, N=2):
    G, V = factors(a, b, n_x, n_y, N)
    return V, cn.orthonormalise(V, b / n_y)[0], cc.rn5_certificate(a, b, n_x, n_y, N, G, V, 1.0, 1.0)


@pytest.mark.parametrize("N", [2, 3])
def test_rt7_passes_on_the_actual_qr(N):
    V, G2, rn5 = certified(N=N)
    out = cc.rt7_conditions(V, G2, rn5)
    assert out["passed"] and out["R7-1"] and out["R7-2"] and out["R7-3"] and out["ranks"] == {"V": [N, N], "G2": [N, N]}
    assert out["t"] == rn5["eps"] / rn5["sigma_low"] and out["tau"] < (1 - out["t"]) ** 2 and out["sin_theta_upper"] < 1
    assert out["tau"] >= 0 and out["sin_theta_upper"] ** 2 >= out["tau"]


def test_rt7_fails_r7_1_for_a_dependent_g2():
    V, G2, rn5 = certified()
    out = cc.rt7_conditions(V, np.column_stack([G2[:, 0], G2[:, 0]]), rn5)
    assert not out["passed"] and not out["R7-1"] and out["ranks"]["G2"] == [1, 1] and "R7-1" in out["reason"]


def test_rt7_fails_r7_2_without_rn5():
    V, G2, rn5 = certified()
    out = cc.rt7_conditions(V, G2, {**rn5, "certified": False, "eps": None, "sigma_low": None})
    assert not out["passed"] and out["R7-1"] and not out["R7-2"] and not out["R7-3"]


def test_rt7_fails_r7_2_for_t_not_below_one():
    V, G2, rn5 = certified()
    out = cc.rt7_conditions(V, G2, {**rn5, "eps": rn5["sigma_low"]})
    assert not out["passed"] and out["t"] == 1 and not out["R7-2"]


def test_rt7_fails_r7_3_for_a_basis_of_another_span():
    V, G2, rn5 = certified()
    other = cn.orthonormalise(V + np.random.default_rng(0).standard_normal(V.shape) * np.max(np.abs(V)), 1.3 / 7)[0]
    out = cc.rt7_conditions(V, other, rn5)
    assert not out["passed"] and out["R7-1"] and out["R7-2"] and not out["R7-3"] and out["tau"] >= (1 - out["t"]) ** 2


def test_rt7_inequality_is_strict():
    V, G2 = np.array([[1.0], [0.0]]), np.array([[3.0], [4.0]])  # cos^2 = 9/25: tau = 16/25 exactly
    equal = cc.rt7_conditions(V, G2, {"certified": True, "eps": Fraction(1), "sigma_low": Fraction(5)})  # t = 1/5: (1 - t)^2 = 16/25
    assert equal["tau"] == Fraction(16, 25) and equal["t"] == Fraction(1, 5) and equal["R7-2"] and not equal["R7-3"] and not equal["passed"]
    below = cc.rt7_conditions(V, G2, {"certified": True, "eps": Fraction(1), "sigma_low": Fraction(5) + Fraction(1, 2**60)})
    assert below["passed"]


@pytest.mark.parametrize("G2", [np.array([[math.nan, 0.0]] * 7), np.zeros((5, 2))])
def test_rt7_refuses_invalid_input(G2):
    V, _, rn5 = certified()
    out = cc.rt7_conditions(V, G2, rn5)
    assert not out["passed"] and out["reason"]
