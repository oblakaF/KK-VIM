"""Instrument checks of the pre-Q3 E-4 / E-6 tool (kkvim_reference.r5_cccc_q3_e4e6). Not a Q3 campaign: small grids only.

Criteria, no thresholds: exact identities (Woodbury on the exact system, x* exact), the theorem VOK-14 (a) checked exactly against
the exact or certified reference, the budget rule 1/100 of D-049.6 as an exact comparison, the proven inequalities VOK-10...VOK-12
in double evaluation, statuses UNRESOLVED / RESEARCH_ONLY / PROVISIONAL where required. Measured values are OBSERVATION.
"""

import subprocess
import sys
from fractions import Fraction

import numpy as np
import pytest

from kkvim_reference import r5_cccc_halfstep as hs
from kkvim_reference import r5_cccc_q3_e4e6 as q

SMALL = [(1.0, 8, 1.3, 7, d, N) for d in hs.DIRECTIONS for N in (1, 2, 3)] + [(1.0, 10, 0.6, 9, "Y", 3), (0.7, 9, 1.0, 12, "X", 2)]


def problem(lx, nx, ly, ny, d, N, **kw):
    n_f = nx if d == "Y" else ny
    return hs.HalfStep(lx, nx, ly, ny, d, kw.pop("G", hs.start_set(n_f, N)), **kw)


def test_instrument_does_not_import_the_main_solver():
    code = ("import sys, kkvim_reference.r5_cccc_q3_e4e6 as m; "
            "print(any(k == 'kkvim' or k.startswith('kkvim.') for k in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    assert out == "False"


@pytest.mark.parametrize("case", SMALL)
@pytest.mark.parametrize("qr", [False, True])
def test_e4_exact_tier(case, qr):
    """Exact reference: Woodbury on the exact system equals x*; VOK-14 (a) holds exactly for the set and the field; the budget is
    satisfied (delta_ref = 0); delta S of both paths measured; components needing B-C internals are UNRESOLVED with the reason."""
    p = problem(*case)
    e = q.e4_case(p, qr=qr)
    assert e["reference"].startswith("EXACT")
    assert e["exact_woodbury_equals_solution"] is True
    assert e["vok14_status"].startswith("CERTIFICATE") and e["vok14_consistent"] is True
    assert e["budget_set"] == e["budget_field"] == "SATISFIED (exact reference)"
    assert e["set_error_2_down"] <= e["set_error_2_up"] <= e["vok14_bound_2_up"]
    assert e["field_error_a_up"] <= e["vok14_field_a_bound_up"]
    assert isinstance(e["eps_dS_modal"], float) and isinstance(e["eps_dS_rows"], float)
    for key in ("eps_0", "eps_Y", "eps_S", "T1_T4"):
        assert e[key] == q.MISSING_COMPONENTS
    assert e["eps_k"] == q.MISSING_EPS_K


def test_e4_certified_tier_and_budget_rule():
    """N m > EXACT_LIMIT: certified R-AC reference; the 1/100 budget is an exact comparison and may fail (then UNRESOLVED);
    VOK-14 (a) stays consistent; raised-precision components UNRESOLVED."""
    e = q.e4_case(problem(1.0, 24, 1.2, 20, "Y", 3), qr=True)
    assert e["reference"].startswith("CERTIFIED")
    allowed = {"SATISFIED", "NOT_MET: comparison UNRESOLVED"}
    assert e["budget_set"] in allowed and e["budget_field"] in allowed
    assert e["vok14_consistent"] is True
    assert e["set_error_2_down"] <= e["set_error_2_up"]
    assert e["eps_0"] == q.MISSING_RAISED


def test_budget_rule_is_exact():
    assert q._budget(Fraction(1, 10**6), Fraction(1, 100)) == "SATISFIED"  # (1e-3)^2 * 1e4 = 1e-2 <= 1e-2
    assert q._budget(Fraction(1, 10**6) + Fraction(1, 10**30), Fraction(1, 100)).startswith("NOT_MET")
    assert q._budget(Fraction(0), Fraction(0)).startswith("UNRESOLVED")


@pytest.mark.parametrize("N", [1, 2, 3])
def test_q_zero(N):
    """q = 0: x* = 0, the B-C set is 0; errors exactly 0; relative quantities and the budget rule are UNRESOLVED (undefined)."""
    p = problem(1.0, 8, 1.3, 7, "Y", N, q=0.0)
    e = q.e4_case(p)
    assert e["set_error_2_up"] == 0.0 and e["field_error_a_up"] == 0.0 and e["vok14_bound_2_up"] == 0.0
    assert e["set_error_rel_up"] == q.UNRESOLVED and e["field_error_a_rel_up"] == q.UNRESOLVED
    assert e["budget_set"].startswith("UNRESOLVED")
    f = q.e6_case(p)
    assert f["rho_0"] == f["kappa_bar_c"] == q.UNRESOLVED


@pytest.mark.parametrize("eps", [1e-2, 1e-6])
@pytest.mark.parametrize("qr", [False, True])
def test_near_dependent_sets_are_research_only(eps, qr):
    """Near-dependent given set (G_2 = G_1 + eps G_2): RESEARCH_ONLY, PROVISIONAL; certificates refer to the solved set."""
    G = hs.start_set(8, 2)
    G[:, 1] = G[:, 0] + eps * G[:, 1]
    p = hs.HalfStep(1.0, 8, 1.0, 7, "Y", G)
    e, f = q.e4_case(p, qr=qr), q.e6_case(p, qr=qr)
    assert e["run_class"].startswith("RESEARCH_ONLY") and f["run_class"].startswith("RESEARCH_ONLY")
    assert e["marking"].startswith("PROVISIONAL")
    assert e["certificates_refer_to"] == ("solved set (span U~ after QR)" if qr else "given set")
    if "vok14_consistent" in e:
        assert e["vok14_consistent"] is True


@pytest.mark.parametrize("case", SMALL + [(1.0, 24, 1.2, 20, "Y", 3), (1.2, 20, 1.0, 24, "X", 2)])
@pytest.mark.parametrize("qr", [False, True])
def test_e6_factors_and_proven_bounds(case, qr):
    """E-6: factors of table 4.3a logged; VOK-10, VOK-11, VOK-12 (item 3) hold in double evaluation; VOK-13 is a MODEL."""
    f = q.e6_case(problem(*case), qr=qr)
    for key in ("rho_0", "kappa_bar_c", "cond_S", "M_norm_2", "varrho", "beta_U", "chi_U", "max_cond_Mk"):
        assert isinstance(f[key], float)
    assert 0 < f["varrho"] <= 1
    assert f["vok10_check"].startswith("HOLDS") and f["vok11_check"].startswith("HOLDS") and f["vok12_check"].startswith("HOLDS")
    assert f["vok13_status"].startswith("MODEL") and "cond_A_before_QR" in f and "cond_A_after_QR" in f


@pytest.mark.parametrize("N, ny, method", [(1, 3, "dense"), (3, 5, "dense"), (1, 6, "subspace"), (3, 9, "subspace")])
def test_m_norm_small_dimension_rule(N, ny, method):
    """||M||_2: dense for N m <= 4N (no branch), subspace otherwise; n_s <= 3: the lower bound 1 is not applicable (N = 1, n_s = 3:
    ||M||_2 < 1, VOK-12 item 3)."""
    f = q.e6_case(hs.HalfStep(1.0, 10, 1.0, ny, "Y", hs.start_set(10, N)))
    assert f["M_norm_method"].startswith(method)
    if ny == 3:
        assert "not applicable" in f["vok12_check"] and (N != 1 or f["M_norm_2"] < 1)


def test_certified_lambda_min_of_A():
    p = problem(1.0, 10, 1.0, 9, "Y", 3)
    A = hs.exact_coefficients(p)[0]
    lam = q.certified_lambda_min_A(A)
    assert lam is not None and 0 < lam <= float(np.linalg.eigvalsh(np.array([[float(v) for v in r] for r in A]))[0])
    assert q._spd_exact([[A[i][j] - (Fraction(lam) if i == j else 0) for j in range(3)] for i in range(3)])


def test_exact_reference_factor_values_are_reported():
    """Exact tier: rho_0, kappa_bar_c, cond(S), ||M||_2 of the exact components are reported next to the logged double values."""
    e = q.e4_case(problem(1.0, 10, 0.6, 9, "Y", 3))
    for key in ("rho_0_exactref", "kappa_bar_c_exactref", "cond_S_exactref", "M_norm_2_exactref"):
        assert isinstance(e[key], float) and e[key] > 0
