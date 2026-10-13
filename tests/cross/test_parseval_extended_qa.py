"""QA tests: independent qualification of the extended Parseval recursion (kkvim_reference, a50f332).

Checks written by QA (NUMERICAL_SPEC.md 15.1-15.5, 15.7; MATH_SPEC.md В-23, В-24). The cross-check recursion
``own_recursion`` of scripts/r2_accuracy_study.py is written from the formulas ChV-17 .. ChV-19 with a fixed
M and a different organisation; no Reference value is used as an expected answer. Tolerances are the
Reference's own error budgets (or rounding for exact identities); no tolerance is tuned to the results.
"""

import importlib.util
import math
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from kkvim_reference.parseval_recursion import ParsevalRecursion

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "r2_accuracy_study.py"
_spec = importlib.util.spec_from_file_location("r2_accuracy_study", SCRIPT)
study = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(study)

U = 2.0**-53


@lru_cache(maxsize=None)
def recursion(s_max=10, **kwargs):
    return ParsevalRecursion(a=kwargs.get("a", 1.0), q=kwargs.get("q", 1.0), D=kwargs.get("D", 1.0), s_max=s_max,
                             x0_scale=kwargs.get("x0_scale", 1.0), m_scale=kwargs.get("m_scale", 1))


@lru_cache(maxsize=None)
def own():
    return study.own_recursion(10)


def test_alternation_and_counters_up_to_s10():
    r = recursion()
    assert [(r.halfstep(s).k, r.halfstep(s).variable) for s in range(1, 11)] == [(s // 2, "Y" if s % 2 else "X") for s in range(1, 11)]


def test_default_and_extended_runs_agree_for_s_le_3():
    short, long = ParsevalRecursion(a=1.0, q=1.0, D=1.0), recursion()
    for s in (1, 2, 3):
        assert short.halfstep(s).coefficients == long.halfstep(s).coefficients
        assert short.halfstep(s).step.coefficients() == long.halfstep(s).step.coefficients()


@pytest.mark.parametrize("s", [4, 7, 10])
def test_late_half_steps_against_own_recursion(s):
    r, o = recursion(), own()[s]
    budget, energy = r.halfstep(s).budget, r.energy(s)
    assert budget.valid
    for x, y in zip(r.coefficients(s).invariants(), o["invariants"]):
        assert abs(x - y) <= budget.invariant_error * abs(y)
    assert abs(r.centre_value(s) - o["centre"]) <= budget.field_error * abs(o["centre"])
    assert abs(energy.energy - o["energy"]) <= energy.error * abs(o["energy"])


def test_m_scale_recomputes_the_whole_recursion():
    base, doubled = recursion(s_max=7), recursion(s_max=7, m_scale=2)
    for s in range(1, 8):
        b = doubled.halfstep(s).budget
        assert b.M == 2 * (b.M_rule + 1) - 1
    assert doubled.halfstep(1).step.n_terms == 2 * base.halfstep(1).step.n_terms  # from the first half-step on
    budget = base.halfstep(7).budget
    for s in range(1, 8):  # diagnostic: doubling M changes the centres only within the budget (not a proof)
        assert abs(doubled.centre_value(s) - base.centre_value(s)) <= budget.field_error * abs(base.centre_value(s))


@pytest.mark.parametrize("scale", [3.7, -2.5, 1e-3])
def test_invariants_and_fields_do_not_depend_on_normalisation(scale):
    base, scaled = recursion(s_max=7), recursion(s_max=7, x0_scale=scale)
    budget = base.halfstep(7).budget
    for s in range(1, 8):
        for x, y in zip(scaled.coefficients(s).invariants(), base.coefficients(s).invariants()):
            assert abs(x - y) <= budget.invariant_error * abs(y)
        w, w0 = scaled.field_on_grid(s, 12, 12), base.field_on_grid(s, 12, 12)
        assert np.max(np.abs(w - w0)) <= budget.field_error * np.max(np.abs(w0))


@pytest.mark.parametrize("a, q, D", [(2.3, -4.1, 0.7), (0.6, 13.0, 2.0)])
def test_scaling_and_negative_load(a, q, D):
    base, other = recursion(s_max=7), recursion(s_max=7, a=a, q=q, D=D)
    budget = base.halfstep(7).budget
    for s in (1, 4, 7):
        w = other.field_on_grid(s, 12, 12) / (q * a**4 / D)
        w0 = base.field_on_grid(s, 12, 12)
        assert np.max(np.abs(w - w0)) <= budget.field_error * np.max(np.abs(w0))
        e = other.energy(s).energy / (q * q * a**6 / D)
        assert abs(e - base.energy(s).energy) <= base.energy(s).error * abs(base.energy(s).energy)
        assert math.copysign(1.0, other.centre_value(s)) == math.copysign(1.0, q)


def test_two_energy_forms_and_identity_b24():
    r = recursion()
    for s in range(1, 11):
        en = r.energy(s)
        assert abs(en.energy_full - en.energy) <= en.rounding * abs(en.energy)  # exact Galerkin: a(w, w) = l(w) (В-23)
        assert en.energy < 0
    for s in (2, 3, 4):  # changes resolved by subtraction: Pi_{s-1} - Pi_s = 1/2 ||w_s - w_{s-1}||^2 (В-24)
        en = r.energy(s)
        # subtracting two energies of size |Pi| carries an absolute rounding of a few u |Pi|; the norm itself
        # carries its relative error bound
        tol = 8 * U * abs(en.energy) + en.change_rel_error * en.change_norm2
        assert abs(-en.delta - 0.5 * en.change_norm2) <= tol
        assert 0.5 * en.change_norm2 > 100 * 8 * U * abs(en.energy)  # the change is resolved by subtraction here


def test_late_changes_below_their_error_bound_are_not_resolved():
    r = recursion()
    resolved = {s: r.energy(s).change_rel > r.energy(s).change_rel_error for s in range(2, 11)}
    assert all(resolved[s] for s in range(2, 7))
    assert not any(resolved[s] for s in range(7, 11))  # documented: changes smaller than the numerical error
    budgets = [r.halfstep(s).budget for s in range(1, 11)]
    assert all(b.valid for b in budgets)
    assert all(b2.input_error > b1.input_error for b1, b2 in zip(budgets, budgets[1:]))  # the transfer grows with s
