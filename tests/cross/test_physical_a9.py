"""QA tests: physical input of kkvim and the official A-9 (R2, step 9).

QA tests (D-033, item 8). docs/NUMERICAL_SPEC.md 9.2-9.3 (I-5, ChV-14), table 18.1 (A-9).

- Input validation of I-5 (a, t, E > 0 and finite; q finite and != 0; |nu| < 1; no t/a limit),
  ERROR_INPUT without a field; D = E t^3 / (12 (1 - nu^2)) checked in exact rational arithmetic.
- A-9 at n = 320 until automatic stopping: q x 3.7, E x 0.013, t x 1.9, nu 0.3 -> 0.1 at the same D,
  a x 2.9; max|w_bar - w_bar_base| / max|w_bar_base| <= 1e-11 on the same nodes, w_bar = w D/(q a^4)
  with D computed here.
- Additional integration check of the physical API (not part of the A-9 criterion): A-9 cannot see an
  error of the D formula, because w_bar removes D. The dimensional field is compared with
  w_bar_ref q a^4 / D_independent, w_bar_ref from a run at a = D = q = 1 on the same grid and with the
  same stopping parameters; relative tolerance 1e-11 for the full field and the centre. A wrong D
  formula must fail this check.
"""

import math
from fractions import Fraction
from functools import lru_cache

import numpy as np
import pytest

import kkvim.controller as controller
import kkvim.physical as physical
from kkvim.controller import run_until_stopped
from kkvim.physical import run_physical

U = 2.0**-53
A9_TOL = 1e-11
DIMENSIONAL_TOL = 1e-11
BASE = dict(a=1.0, t=0.01, E=2.1e11, nu=0.3, q=1000.0, n_x=320, n_y=320)


def exact_stiffness(E, t, nu):
    E, t, nu = Fraction(E), Fraction(t), Fraction(nu)
    return float(E * t**3 / (12 * (1 - nu * nu)))


@lru_cache(maxsize=None)
def run(**kwargs):
    return run_physical(**kwargs)


def run_args(args):
    return run(**dict(sorted(args.items())))


def w_bar(args):
    res = run_args(args)
    assert res.status == "CONVERGED"
    return res.run.last_field() * exact_stiffness(args["E"], args["t"], args["nu"]) / (args["q"] * args["a"] ** 4)


# Input (I-5) and D ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "change",
    [
        {"a": 0.0}, {"t": -0.01}, {"E": math.inf}, {"a": math.nan}, {"nu": 1.0}, {"nu": -1.0}, {"q": 0.0},
        {"n_x": 3}, {"E": "2e11"}, {"E": 1e300, "t": 1e5}, {"E": 1e-300, "t": 1e-10},
    ],
)
def test_invalid_input_is_error_input_without_field(change):
    r = run_physical(**{**BASE, "n_x": 20, "n_y": 20, **change})
    assert r.status == "ERROR_INPUT" and r.run is None and r.D is None
    assert r.dimensionless_field() is None


@pytest.mark.parametrize("change", [{"nu": -0.5}, {"t": 10.0}, {"q": -500.0}])
def test_valid_input_without_extra_limits(change):
    r = run_physical(**{**BASE, "n_x": 20, "n_y": 20, **change})
    assert r.status == "CONVERGED" and r.dimensionless_field() is not None


@pytest.mark.parametrize("E, t, nu", [(2.1e11, 0.01, 0.3), (7e10, 0.003, 0.33), (3.3e9, 0.2, -0.4)])
def test_stiffness_formula_and_transfer_to_controller(E, t, nu):
    r = run_physical(**{**BASE, "E": E, "t": t, "nu": nu, "n_x": 20, "n_y": 20})
    expected = exact_stiffness(E, t, nu)
    assert abs(r.D - expected) <= 4 * U * expected
    assert all(record.D == r.D for record in r.run.history.records)
    direct = run_until_stopped(BASE["a"], r.D, BASE["q"], 20, 20)  # existing API with a ready D
    np.testing.assert_array_equal(direct.last_field(), r.run.last_field())


# A-9 ------------------------------------------------------------------------------------------


_NU2 = 0.1
VARIANTS = {
    "q x 3.7": {"q": BASE["q"] * 3.7},
    "E x 0.013": {"E": BASE["E"] * 0.013},
    "t x 1.9": {"t": BASE["t"] * 1.9},
    "nu 0.3 -> 0.1 at the same D": {"nu": _NU2, "E": BASE["E"] * (1 - _NU2**2) / (1 - BASE["nu"] ** 2)},
    "a x 2.9": {"a": BASE["a"] * 2.9},
}


@pytest.mark.parametrize("name", VARIANTS)
def test_a9_dimensionless_field_is_invariant(name):
    base = w_bar(BASE)
    varied = w_bar({**BASE, **VARIANTS[name]})
    assert run_args({**BASE, **VARIANTS[name]}).run.s == run_args(BASE).run.s
    assert np.max(np.abs(varied - base)) <= A9_TOL * np.max(np.abs(base))


# Additional dimensional check of the physical API ---------------------------------------------------


@lru_cache(maxsize=None)
def expected_dimensional_field():
    reference = run_until_stopped(1.0, 1.0, 1.0, 320, 320)  # w_bar_ref, same grid and stopping parameters
    D = exact_stiffness(BASE["E"], BASE["t"], BASE["nu"])
    return reference.last_field() * BASE["q"] * BASE["a"] ** 4 / D


def dimensional_deviation(result):
    w, expected = result.run.last_field(), expected_dimensional_field()
    field = np.max(np.abs(w - expected)) / np.max(np.abs(expected))
    centre = abs(w[160, 160] - expected[160, 160]) / abs(expected[160, 160])
    return field, centre


def test_dimensional_deflection_against_independent_stiffness():
    field, centre = dimensional_deviation(run_args(BASE))
    assert field <= DIMENSIONAL_TOL and centre <= DIMENSIONAL_TOL


def test_wrong_stiffness_formula_is_detected_only_by_dimensional_check(monkeypatch):
    monkeypatch.setattr(physical, "bending_stiffness", lambda E, t, nu: E * t**3 / (12.0 * (1.0 - nu)))
    wrong = run_physical(**BASE)
    field, centre = dimensional_deviation(wrong)
    assert field > DIMENSIONAL_TOL and centre > DIMENSIONAL_TOL
    # w_bar of the same run uses the same wrong D, so the A-9 type comparison cannot see the error
    base_bar = run_args(BASE).dimensionless_field()
    assert np.max(np.abs(wrong.dimensionless_field() - base_bar)) <= A9_TOL * np.max(np.abs(base_bar))


# Controller guard (step 9 fix of the step-8 remark) ---------------------------------------------------


@pytest.mark.parametrize("bad", [math.nan, math.inf])
def test_non_finite_stopping_diagnostics_give_error_nonfinite(monkeypatch, bad):
    original = controller._field_changes

    def patched(previous, current):
        values = original(previous, current)
        return (bad, values[1], values[2]) if current.s == 4 else values

    monkeypatch.setattr(controller, "_field_changes", patched)
    r = run_until_stopped(1.0, 1.0, 1.0, 20, 20)
    assert r.status == "ERROR_NONFINITE" and r.s == 4 and len(r.energy_checks) == 4
    with pytest.raises(ValueError):
        controller.stopping_decision(3, [(2, 1e-3, 1e-3), (3, bad, 1e-3)], 1e-10, 1e-8, 3, 100)
