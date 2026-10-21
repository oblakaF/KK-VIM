"""Own tests of kkvim.physical: physical input and scaling (R2, step 9).

Checks follow docs/NUMERICAL_SPEC.md 9.2 (I-5), 9.3 (ChV-14) and table 18.1 (A-9:
w_bar in the same nodes, n = 320, run to the stop, <= 1e-11 relative to max |w_bar|).
nu enters only through D (MATH_SPEC.md 2.2). QA qualifies A-9 independently.
"""

import math

import numpy as np
import pytest

from kkvim.coefficients import ERROR_INPUT
from kkvim.controller import CONVERGED, run_until_stopped
from kkvim.physical import PhysicalInputError, bending_stiffness, run_physical

A9_TOL = 1e-11  # table 18.1, A-9
BASE = dict(a=1.0, t=0.01, E=2.1e11, nu=0.3, q=1e3)


def w_bar_deviation(w, w_ref):
    return float(np.max(np.abs(w - w_ref)) / np.max(np.abs(w_ref)))


def test_bending_stiffness_formula():
    D = bending_stiffness(2.1e11, 0.01, 0.3)
    assert D == 2.1e11 * 0.01**3 / (12 * (1 - 0.3**2))
    r = run_physical(**BASE, n_x=20, n_y=20)
    assert r.D == D and r.status == CONVERGED and r.input.nu == 0.3


def test_physical_api_matches_direct_controller():
    r = run_physical(**BASE, n_x=20, n_y=30)
    direct = run_until_stopped(BASE["a"], r.D, BASE["q"], 20, 30)
    assert r.status == direct.status and r.run.s == direct.s
    np.testing.assert_array_equal(r.run.last_field(), direct.last_field())
    w = direct.last_field()
    np.testing.assert_array_equal(r.dimensionless_field(), w * r.D / (BASE["q"] * BASE["a"] ** 4))


def test_stopping_parameters_are_passed_through():
    r = run_physical(**BASE, n_x=20, n_y=20, tol_inf=1e-12, tol_E=1e-10, s_min=4, s_max=50)
    p = r.run.parameters
    assert (p.tol_inf, p.tol_E, p.s_min, p.s_max) == (1e-12, 1e-10, 4, 50)


def test_dimensionless_field_uses_last_completed_halfstep():
    r = run_physical(**BASE, n_x=20, n_y=20, s_max=4, tol_inf=1e-20)
    assert r.run.s == 4
    np.testing.assert_array_equal(r.dimensionless_field(), r.run.history.field(4) * r.D / BASE["q"])


@pytest.mark.parametrize(
    "change",
    [
        {"a": 0.0}, {"a": -1.0}, {"a": math.inf}, {"t": 0.0}, {"t": math.nan}, {"E": -2.1e11}, {"E": math.inf},
        {"nu": 1.0}, {"nu": -1.0}, {"nu": 1.5}, {"nu": math.nan}, {"q": 0.0}, {"q": math.nan}, {"q": -math.inf},
        {"a": True}, {"E": "2.1e11"}, {"q": None},
    ],
)
def test_invalid_physical_input_is_error_input(change):
    r = run_physical(**{**BASE, **change}, n_x=20, n_y=20)
    assert r.status == ERROR_INPUT and isinstance(r.error, PhysicalInputError)
    assert r.D is None and r.run is None
    assert r.dimensionless_field() is None  # no fictitious zero deflection


@pytest.mark.parametrize("n_x, n_y", [(3, 20), (20, 2), (20.0, 20), (True, 20)])
def test_invalid_grid_is_error_input(n_x, n_y):
    r = run_physical(**BASE, n_x=n_x, n_y=n_y)
    assert r.status == ERROR_INPUT and r.run is None


@pytest.mark.parametrize("change", [{"E": 1e300, "t": 1e200}, {"E": 1e-300, "t": 1e-120}, {"E": 1e300, "t": 1e3}])
def test_overflowing_or_zero_stiffness_is_rejected(change):
    r = run_physical(**{**BASE, **change}, n_x=20, n_y=20)
    assert r.status == ERROR_INPUT and r.D is None


def test_no_restriction_on_thickness_ratio():
    for t in (1e-5, 0.5, 3.0):  # including t > a
        r = run_physical(**{**BASE, "t": t}, n_x=20, n_y=20)
        assert r.status == CONVERGED


def test_negative_load_and_sign():
    up = run_physical(**BASE, n_x=20, n_y=30)
    down = run_physical(**{**BASE, "q": -BASE["q"]}, n_x=20, n_y=30)
    assert np.all(down.run.last_field()[1:-1, 1:-1] < 0.0)
    assert w_bar_deviation(down.dimensionless_field(), up.dimensionless_field()) <= A9_TOL


# A-9 on n = 320, to the stop -------------------------------------------------------------

N = 320
E_NU = BASE["E"] * (1 - 0.1**2) / (1 - 0.3**2)  # keeps D when nu: 0.3 -> 0.1
VARIANTS = {
    "q x 3.7": {"q": 3.7 * BASE["q"]},
    "E x 0.013": {"E": 0.013 * BASE["E"]},
    "t x 1.9": {"t": 1.9 * BASE["t"]},
    "nu 0.3 -> 0.1, same D": {"nu": 0.1, "E": E_NU},
    "a x 2.9": {"a": 2.9 * BASE["a"]},
}


@pytest.fixture(scope="module")
def base_run():
    return run_physical(**BASE, n_x=N, n_y=N)


@pytest.mark.parametrize("name", VARIANTS)
def test_a9_scaling(base_run, name):
    r = run_physical(**{**BASE, **VARIANTS[name]}, n_x=N, n_y=N)
    assert base_run.status == CONVERGED and r.status == CONVERGED
    assert w_bar_deviation(r.dimensionless_field(), base_run.dimensionless_field()) <= A9_TOL


def test_a9_nu_changes_only_through_d(base_run):
    r = run_physical(**{**BASE, **VARIANTS["nu 0.3 -> 0.1, same D"]}, n_x=N, n_y=N)
    assert r.input.nu == 0.1
    assert abs(r.D - base_run.D) <= 8 * np.finfo(float).eps * base_run.D  # a few ulp, not bitwise
    assert w_bar_deviation(r.dimensionless_field(), base_run.dimensionless_field()) <= A9_TOL
