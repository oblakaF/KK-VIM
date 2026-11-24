"""``kkvim.cccc_n1``: the limited research mode of CCCC, N = 1 (D-061; checks B-1...B-9 of Policy 10.1).

Exact checks use Fraction. The solver is compared bit for bit with itself. Values of Policy 14 (SCRATCH) are compared in their printed digits, as
a record of the arithmetic, not as a tolerance of the method. No tolerance is chosen here. QA reference code is not imported: K-1...K-6 are QA.
"""

import ast
import math
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

import kkvim.cccc_n1 as cn
from kkvim.coupled import N_SOLVE
from kkvim.physical import bending_stiffness

SOURCE = Path(cn.__file__)


def fractions(values):
    return [Fraction(float(value)) for value in values]


def exact_d4(m, h):
    """D4 from its definition: Ł^T W Ł / h, Ł of the even reflection, W = diag(h/2, h, ..., h, h/2)."""
    n = m + 1
    h = Fraction(h)
    rows = [[Fraction(0)] * m for _ in range(n + 1)]
    for i in range(1, n):  # (Lf)_i = (f_{i-1} - 2 f_i + f_{i+1}) / h^2; column j is the inner node j + 1
        rows[i][i - 1] = -2 / h**2
        if i >= 2:
            rows[i][i - 2] = 1 / h**2
        if i <= n - 2:
            rows[i][i] = 1 / h**2
    rows[0][0] = 2 / h**2  # (Lf)_0 = 2 f_1 / h^2
    rows[n][m - 1] = 2 / h**2
    weights = [h / 2, *[h] * (n - 1), h / 2]
    return [[sum(weights[k] * rows[k][i] * rows[k][j] for k in range(n + 1)) / h for j in range(m)] for i in range(m)]


def exact_second_difference(v, h):
    padded = [Fraction(0), *v, Fraction(0)]
    return [(padded[i - 1] - 2 * padded[i] + padded[i + 1]) / Fraction(h) ** 2 for i in range(1, len(padded) - 1)]


def dot(u, v):
    return sum(a * b for a, b in zip(u, v, strict=True))


def matrix_times(matrix, v):
    return [dot(row, v) for row in matrix]


# --- B-1: the operators and the energy ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("m", [6, 7, 12])
def test_d4_is_the_definition_with_the_clamped_rows(m):
    h = Fraction(1, m + 1)
    d4 = exact_d4(m, h)
    assert d4[0][:3] == [7 / h**4, -4 / h**4, 1 / h**4] and d4[-1][-3:] == [1 / h**4, -4 / h**4, 7 / h**4]
    assert d4[2][:5] == [1 / h**4, -4 / h**4, 6 / h**4, -4 / h**4, 1 / h**4]
    assert all(d4[i][j] == d4[j][i] for i in range(m) for j in range(m))
    assert all(d4[i][j] == 0 for i in range(m) for j in range(m) if abs(i - j) > 2)


@pytest.mark.parametrize("m", [6, 7, 12])
def test_the_applied_operators_equal_the_definition(m):
    """Integers and h = 1/8 or a power of two: the arithmetic of the double code is exact here."""
    h = 0.125
    v = np.array([(-1) ** i * (i * i + 1) for i in range(m)], dtype=float)
    exact = matrix_times(exact_d4(m, h), fractions(v))
    assert [Fraction(float(value)) for value in cn.d4_apply(v, h)] == exact
    assert [Fraction(float(value)) for value in cn.second_difference(v, h)] == exact_second_difference(fractions(v), h)


def test_the_energy_formula_is_the_discrete_energy_at_the_product():
    """Dyadic factors, exact in double. Pi_h^C of Policy 2 from Ł and W against the moments of the factors (Policy 6), both exactly."""
    n_x, n_y = 8, 8
    a, b, D, q = 1.0, 2.0, 3.0, 5.0
    X, Y = np.arange(1.0, n_x), np.array([1.0, -2.0, 3.0, 1.0, 1.0, -1.0, 2.0])
    h_x, h_y = Fraction(a) / n_x, Fraction(b) / n_y

    def laplacian_rows(v, h):
        full = [Fraction(0), *fractions(v), Fraction(0)]
        n = len(full) - 1
        return [2 * full[1] / h**2, *[(full[i - 1] - 2 * full[i] + full[i + 1]) / h**2 for i in range(1, n)], 2 * full[n - 1] / h**2]

    def weights(h, n):
        return [h / 2, *[h] * (n - 1), h / 2]

    full_x, full_y = [Fraction(0), *fractions(X), Fraction(0)], [Fraction(0), *fractions(Y), Fraction(0)]
    lx, ly = laplacian_rows(X, h_x), laplacian_rows(Y, h_y)
    wx, wy = weights(h_x, n_x), weights(h_y, n_y)
    bending = sum(wx[i] * wy[j] * (lx[i] * full_y[j] + full_x[i] * ly[j]) ** 2 for i in range(n_x + 1) for j in range(n_y + 1))
    energy = Fraction(D) / 2 * bending - Fraction(q) * h_x * h_y * sum(fractions(X)) * sum(fractions(Y))
    assert Fraction(cn.discrete_energy(X, Y, a, b, D, q, n_x, n_y)) == energy


# --- B-2: C-3+ ---------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("a, b, n_x, n_y", [(1.0, 0.6, 10, 9), (1.0, 1.3, 8, 7), (1.0, 1.0, 12, 13)])
def test_c3plus_identity_and_bound_exactly_on_the_factors_of_a_run(a, b, n_x, n_y):
    result = cn.run_cccc_n1(a, b, 1.0, 1.0, n_x, n_y, 8)
    assert result.status == cn.COMPLETED_FIXED_HALF_STEPS
    for step in result.steps:
        l_f, n_f = (a, n_x) if step.s % 2 == 1 else (b, n_y)
        h = l_f / n_f
        g = fractions(step.given)
        m = len(g)
        assert step.premises_status == cn.PREMISES_ESTABLISHED and not step.premises_reasons
        exact = Fraction(h) * dot(g, matrix_times(exact_d4(m, h), g))
        second = exact_second_difference(g, h)
        assert Fraction(h) * dot(second, second) + 2 / Fraction(h) ** 3 * (g[0] ** 2 + g[-1] ** 2) == exact
        k = m + cn.K_EXTRA
        gamma = k * Fraction(cn.U) / (1 - k * Fraction(cn.U))
        assert abs(Fraction(step.C) - exact) <= gamma * exact


def test_c_is_c_l_plus_e_of_the_correctly_rounded_sums():
    g = cn.start_factor(10)
    parts = cn.assemble(g, 1.0, 10, 1.0)
    assert parts["C"] == parts["C_L"] + parts["E"]
    values = [Fraction(0), *fractions(g), Fraction(0)]
    assert parts["Y3"].tolist() == [float(values[i - 1] - 2 * values[i] + values[i + 1]) for i in range(1, 10)]


# --- B-3, the start ----------------------------------------------------------------------------------------------------------------

def test_the_coefficients_of_the_start_in_their_printed_digits():
    parts = cn.assemble(cn.start_factor(10), 1.0, 10, 1.0)
    assert f"{parts['A']:.4f}" == "0.3750" and f"{parts['B']:.4f}" == "4.7746" and f"{parts['C']:.2f}" == "182.37" and f"{parts['F']:.4f}" == "0.5000"


def test_the_start_and_the_first_half_step():
    result = cn.run_cccc_n1(1.0, 2.0, 1.0, 1.0, 12, 9, 1)
    step = result.steps[0]
    assert np.array_equal(step.given, np.sin(math.pi * np.arange(1, 12) / 12) ** 2)
    assert step.direction == "Y" and step.n_s == 9 and step.h_s == 2.0 / 9 and step.raw.size == 8 and step.given.size == 11
    assert step.k == 0 and step.status == cn.COMPLETED_FIXED_HALF_STEPS


# --- B-4: the sequence -------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("a, b, n_x, n_y", [(1.0, 1.0, 8, 8), (1.0, 2.0, 16, 9), (1.0, 0.5, 7, 12)])
def test_the_sequence_and_the_transfer_of_the_factors(a, b, n_x, n_y):
    result = cn.run_cccc_n1(a, b, 1.0, 1.0, n_x, n_y, 8)
    assert [step.direction for step in result.steps] == list("YXYXYXYX")
    assert [(step.s, step.k) for step in result.steps] == [(s, s // 2) for s in range(1, 9)]
    assert [step.n_s for step in result.steps] == [n_y, n_x] * 4
    for previous, step in zip(result.steps, result.steps[1:], strict=False):
        assert np.array_equal(step.given, previous.next_given)
    for step in result.steps:
        assert float(np.max(np.abs(step.next_given))) == 1.0 and math.fsum(step.next_given.tolist()) > 0
        assert np.array_equal(step.next_given, step.raw / step.c) and np.array_equal(step.next_conjugate, step.given * step.c)
        assert step.c == step.sigma * float(np.max(np.abs(step.raw)))
        # NS 4.4: the exact product of the factors at a node moves by at most 2u, the rounding of c * g and y / c. A product rounded to double
        # rounds once more, so two doubles of a node can differ by a little more; that is not this statement.
        for i in range(step.given.size):
            for j in range(step.raw.size):
                before = Fraction(float(step.given[i])) * Fraction(float(step.raw[j]))
                after = Fraction(float(step.next_conjugate[i])) * Fraction(float(step.next_given[j]))
                assert abs(after - before) <= 2 * Fraction(cn.U) * abs(before)


def test_the_normalization_keeps_the_solved_factor_positive_under_a_negative_load():
    """I-4: sigma is the sign of the exact sum, so the sign of q goes to the conjugate factor."""
    first = cn.run_cccc_n1(1.0, 1.0, 1.0, -1.0, 8, 8, 2).steps[0]
    assert first.sigma == -1.0 and first.c < 0.0 and math.fsum(first.next_given.tolist()) > 0.0 and np.all(first.next_conjugate < 0.0)
    assert float(np.max(np.abs(first.next_given))) == 1.0


def test_the_field_uses_the_factors_of_the_right_directions():
    for half_steps in (3, 4):
        result = cn.run_cccc_n1(1.0, 1.5, 1.0, 1.0, 10, 14, half_steps)
        X, Y = result.factors()
        assert X.size == 9 and Y.size == 13
        assert np.array_equal(result.field()[1:-1, 1:-1], np.outer(X, Y))


def test_exactly_the_given_number_of_half_steps_and_a_shorter_run_is_a_prefix():
    long = cn.run_cccc_n1(1.0, 1.0, 1.0, 1.0, 12, 12, 8)
    for half_steps in range(1, 8):
        short = cn.run_cccc_n1(1.0, 1.0, 1.0, 1.0, 12, 12, half_steps)
        assert len(short.steps) == short.half_steps_completed == half_steps and short.status == cn.COMPLETED_FIXED_HALF_STEPS
        for mine, other in zip(short.steps, long.steps, strict=False):
            assert mine.A == other.A and mine.C == other.C and np.array_equal(mine.raw, other.raw) and mine.eta == other.eta


def test_the_diagnostics_of_a_half_step():
    result = cn.run_cccc_n1(1.0, 1.0, 1.0, 1.0, 32, 32, 8)
    assert result.steps[0].field_change_inf == 1.0 and result.steps[0].delta_P_hat == -result.steps[0].P_hat
    for previous, step in zip(result.steps, result.steps[1:], strict=False):
        assert step.delta_P_hat == previous.P_hat - step.P_hat
    assert all(step.eta <= cn.ETA_MAX_C and step.cond_S >= 1.0 and step.min_DM > 0.0 for step in result.steps)
    assert all(step.alpha_centre > 0.0 for step in result.steps)
    assert cn.run_cccc_n1(1.0, 1.0, 1.0, 1.0, 33, 32, 2).steps[-1].alpha_centre is None


# --- the clamped plate -------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("a, b, n_x, n_y", [(1.0, 1.0, 32, 32), (1.0, 2.0, 32, 64), (1.0, 1.0, 47, 53)])
def test_the_field_is_clamped_and_loaded_downwards_by_the_sign_of_q(a, b, n_x, n_y):
    result = cn.run_cccc_n1(a, b, 1.0, 1.0, n_x, n_y, 8)
    w = result.field()
    assert w.shape == (n_x + 1, n_y + 1)
    assert not np.any(w[0, :]) and not np.any(w[-1, :]) and not np.any(w[:, 0]) and not np.any(w[:, -1])
    assert np.all(w[1:-1, 1:-1] > 0.0)
    down = cn.run_cccc_n1(a, b, 1.0, -1.0, n_x, n_y, 8)
    assert np.all(down.field()[1:-1, 1:-1] < 0.0) and np.array_equal(down.field(), -w)


def test_the_slope_at_the_wall_vanishes_with_the_grid():
    """w_1 / h, the one-sided difference at the wall with w_-1 = w_1: a relation of the sequence of grids; no value is a criterion."""
    slopes = []
    for n in (16, 32, 64, 128):
        w = cn.run_cccc_n1(1.0, 1.0, 1.0, 1.0, n, n, 4).field()
        slopes.append(float(np.max(np.abs(w[1, :]))) * n)
    assert slopes[0] > slopes[1] > slopes[2] > slopes[3]


@pytest.mark.parametrize("n_x, n_y, printed", [(32, 32, "1.27413e-03"), (40, 64, "1.26824e-03")])
def test_alpha_in_the_printed_digits_of_policy_8(n_x, n_y, printed):
    """Records of the arithmetic of Policy 8 (SCRATCH): not a closeness to a solution."""
    assert f"{cn.run_cccc_n1(1.0, 1.0, 1.0, 1.0, n_x, n_y, 8).steps[-1].alpha_centre:.5e}" == printed


def test_exchanging_the_sides_does_not_make_the_same_sequence():
    """The first half-step is always Y: the plate b x a with the grids exchanged is a different start (D-049.3), so no symmetry is assumed."""
    one = cn.run_cccc_n1(1.0, 2.0, 1.0, 1.0, 32, 64, 1)
    other = cn.run_cccc_n1(2.0, 1.0, 1.0, 1.0, 64, 32, 1)
    assert one.status == other.status == cn.COMPLETED_FIXED_HALF_STEPS
    assert one.steps[0].A != other.steps[0].A


# --- B-5: every row of the table of Policy 6 -----------------------------------------------------------------------------------------

def run_small(half_steps=3, **changes):
    values = {"a": 1.0, "b": 1.0, "D": 1.0, "q": 1.0, "n_x": 8, "n_y": 8, **changes}
    return cn.run_cccc_n1(**values, half_steps=half_steps)


def stop_at(monkeypatch, name, s, replacement):
    """``name`` of the module replaced at its call of half-step s (counted for this name: one call per half-step)."""
    real, count = getattr(cn, name), [0]

    def call(*args, **kwargs):
        count[0] += 1
        return replacement(*args, **kwargs) if count[0] == s else real(*args, **kwargs)

    monkeypatch.setattr(cn, name, call)
    return count


def assert_stopped(result, status, s):
    """The status, in the last record; the field is that of the last completed half-step; nothing is solved or tried after the stop."""
    assert result.status == status and result.status != cn.COMPLETED_FIXED_HALF_STEPS and result.reason
    assert len(result.steps) == s and result.steps[-1].status == status and result.steps[-1].reason == result.reason
    assert result.half_steps_completed == s - 1 and not result.steps[-1].done
    assert result.application_status == "RESEARCH_ONLY / PROVISIONAL / NOT_QUALIFIED" and result.mode == "FIXED_HALF_STEPS_RESEARCH"
    if s == 1:
        assert result.field() is None
    else:
        assert result.field() is not None


@pytest.mark.parametrize("changes, half_steps", [
    ({"n_x": 6}, 2), ({"n_y": 6}, 2), ({"n_x": 1281}, 2), ({"n_y": 1281}, 2), ({}, 0), ({}, 9), ({}, -1), ({}, 2.0), ({}, True), ({}, None),
    ({"b": 5.1}, 2), ({"a": 5.1}, 2),
])
def test_outside_the_domain_nothing_is_computed(monkeypatch, changes, half_steps):
    monkeypatch.setattr(cn, "assemble", lambda *a: pytest.fail("computed"))
    result = run_small(half_steps, **changes)
    assert result.status == cn.NOT_EXECUTED_OUTSIDE_DOMAIN and result.reason and result.steps == () and result.field() is None


@pytest.mark.parametrize("changes", [
    {"a": 0.0}, {"a": -1.0}, {"b": float("nan")}, {"D": 0.0}, {"D": -1.0}, {"D": float("inf")}, {"D": "1"}, {"q": 0.0}, {"q": float("nan")},
    {"q": float("inf")}, {"q": True}, {"n_x": 8.0}, {"n_y": "8"}, {"n_x": True},
])
def test_invalid_input_is_error_input(monkeypatch, changes):
    monkeypatch.setattr(cn, "assemble", lambda *a: pytest.fail("computed"))
    result = run_small(**changes)
    assert result.status == cn.ERROR_INPUT and result.reason and result.steps == ()


@pytest.mark.parametrize("name, value", [("t", 0.0), ("E", -1.0), ("nu", 1.0), ("nu", float("nan")), ("q", 0.0), ("a", float("inf"))])
def test_invalid_physical_input_is_error_input(name, value):
    args = {"a": 1.0, "b": 1.0, "t": 0.01, "E": 2.1e11, "nu": 0.3, "q": 1000.0, "n_x": 8, "n_y": 8, "half_steps": 2, name: value}
    result = cn.run_cccc_n1_physical(**args)
    assert result.status == cn.ERROR_INPUT and result.D is None and result.reason


def test_the_physical_entry_refuses_the_domain_first():
    args = {"a": 1.0, "b": 1.0, "t": 0.01, "E": 2.1e11, "nu": 0.3, "q": 1000.0, "n_x": 4, "n_y": 8, "half_steps": 2}
    assert cn.run_cccc_n1_physical(**args).status == cn.NOT_EXECUTED_OUTSIDE_DOMAIN  # 4 is valid for R2, not here


@pytest.mark.parametrize("values", [{"n_x": 7, "n_y": 7}, {"n_x": 1280, "n_y": 7}, {"b": 5.0}, {"a": 5.0}])
def test_the_bounds_of_the_domain_run(values):
    assert run_small(2, **values).status == cn.COMPLETED_FIXED_HALF_STEPS


@pytest.mark.parametrize("s", [1, 2])
@pytest.mark.parametrize("name, status, replacement", [
    ("solve_bc", cn.ERROR_NONFINITE, lambda *a: np.full(a[-1] - 1, math.nan)),
    ("solve_bc", cn.STOP_ZERO_FIELD, lambda *a: np.zeros(a[-1] - 1)),
    ("residual_eta", cn.STOPPED_ETA_GUARD, lambda *a: math.nan),  # a NaN does not pass the guard
    ("residual_eta", cn.STOPPED_ETA_GUARD, lambda *a: 2.0 * cn.ETA_MAX_C),
])
def test_stops_of_the_half_step(monkeypatch, s, name, status, replacement):
    stop_at(monkeypatch, name, s, replacement)
    assert_stopped(run_small(), status, s)


def test_the_value_of_the_guard_and_its_boundary(monkeypatch):
    assert cn.ETA_MAX_C == 100 * 2.0**-53
    monkeypatch.setattr(cn, "residual_eta", lambda *a: cn.ETA_MAX_C)
    assert run_small().status == cn.COMPLETED_FIXED_HALF_STEPS  # the condition is eta > eta_max


def test_the_guard_stops_a_real_residual_and_tries_nothing_else(monkeypatch):
    """The real eta of the first half-step against a lowered limit: a real path through the guard, no replaced value. No refinement or fallback."""
    eta = run_small(1).steps[0].eta
    assert eta > 0.0
    monkeypatch.setattr(cn, "ETA_MAX_C", 0.5 * eta)
    calls = stop_at(monkeypatch, "solve_bc", 99, None)
    result = run_small()
    assert_stopped(result, cn.STOPPED_ETA_GUARD, 1)
    assert calls == [1] and result.steps[0].eta == eta and result.steps[0].raw is not None


def test_nonfinite_coefficients(monkeypatch):
    real = cn.assemble
    monkeypatch.setattr(cn, "assemble", lambda *a: {**real(*a), "F": math.inf})
    assert_stopped(run_small(), cn.ERROR_NONFINITE, 1)


def test_nonfinite_given_factor(monkeypatch):
    monkeypatch.setattr(cn, "start_factor", lambda n: np.full(n - 1, math.nan))
    assert_stopped(run_small(), cn.ERROR_NONFINITE, 1)


def test_a_zero_given_factor_is_a_protective_stop(monkeypatch):
    monkeypatch.setattr(cn, "start_factor", lambda n: np.zeros(n - 1))
    assert_stopped(run_small(), cn.PROTECTIVE_STOP_FACTORIZATION, 1)


@pytest.mark.parametrize("failing", ["cholesky", "solve"])
def test_a_failed_capacitance_matrix_is_a_protective_stop(monkeypatch, failing):
    def fail(*args):
        raise np.linalg.LinAlgError("not positive definite")

    monkeypatch.setattr(np.linalg, failing, fail)
    assert_stopped(run_small(), cn.PROTECTIVE_STOP_FACTORIZATION, 1)


def test_a_nonpositive_modal_value_is_a_protective_stop(monkeypatch):
    real = cn.assemble
    monkeypatch.setattr(cn, "assemble", lambda *a: {**real(*a), "B": -1e30})
    assert_stopped(run_small(), cn.PROTECTIVE_STOP_FACTORIZATION, 1)


@pytest.mark.parametrize("calls_before_failure", [0, 1])
def test_a_rounding_probe_before_and_after_the_solve(monkeypatch, calls_before_failure):
    count = [0]

    def probe():
        count[0] += 1
        return ["P-1: round to nearest, Python"] if count[0] > calls_before_failure else []

    monkeypatch.setattr(cn, "rounding_probe_failures", probe)
    result = run_small()
    assert_stopped(result, cn.UNRESOLVED_C3PLUS_PREMISE, 1)
    assert result.steps[0].premises_status == cn.PREMISES_UNRESOLVED and "P-1" in result.reason
    assert (result.steps[0].raw is None) and (result.steps[0].min_DM is None) == (calls_before_failure == 0)


def test_the_rounding_probes_pass_here():
    assert cn.rounding_probe_failures() == []


@pytest.mark.parametrize("changes, fragment", [
    ({"C": math.inf}, "P-2: C"), ({"C_L": math.nan}, "P-2: C_L"), ({"E": 1e-310}, "P-3: E"), ({"g1": 1e-310}, "P-3: g1"), ({"h3": 1e-310}, "P-3: h3"),
    ({"h": 0.0}, "P-3: h"), ({"g1": 1e-160}, "P-3: квадрат"), ({"gm": 1e-160}, "P-3: квадрат"), ({"s": 1e308, "g1": 1e10}, "P-2: 2 max"),
])
def test_each_premise_is_checked(changes, fragment):
    parts = {**cn.assemble(cn.start_factor(10), 1.0, 10, 1.0), **changes}
    assert any(fragment in failure for failure in cn.premise_failures(parts, 10))


def test_a_premise_of_the_gram_sum_and_of_x_and_of_k():
    parts = cn.assemble(cn.start_factor(10), 1.0, 10, 1.0)
    assert cn.premise_failures(parts, 10) == []
    large = {**parts, "X": np.full(9, 1e154)}
    assert any("m max|X|^2" in failure for failure in cn.premise_failures(large, 10))
    small = {**parts, "X": np.full(9, 1e-160)}
    assert any("|X|" in failure for failure in cn.premise_failures(small, 10))
    assert any("P-5" in failure for failure in cn.premise_failures(parts, 2**53))


def test_a_failed_premise_stops_before_the_solve(monkeypatch):
    monkeypatch.setattr(cn, "premise_failures", lambda *a: ["P-3: x"])
    monkeypatch.setattr(cn, "solve_bc", lambda *a: pytest.fail("solved"))
    assert_stopped(run_small(), cn.UNRESOLVED_C3PLUS_PREMISE, 1)


# --- B-6, B-8: scaling and repetition ------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("a, b, D, q, n_x, n_y", [(1.0, 1.5, 1.0, 1.0, 16, 24), (0.75, 0.5, 3.0, -7.0, 13, 8)])
def test_the_scale_gives_the_same_w_bar_bit_for_bit(a, b, D, q, n_x, n_y):
    one = cn.run_cccc_n1(a, b, D, q, n_x, n_y, 8)
    other = cn.run_cccc_n1(2 * a, 2 * b, 4 * D, 8 * q, n_x, n_y, 8)
    assert np.array_equal(one.dimensionless_field(), other.dimensionless_field())
    assert np.array_equal(32.0 * one.field(), other.field())  # q a^4 / D times 32


def test_the_physical_entry_is_the_entry_with_d():
    a, b, t, E, nu, q = 1.0, 1.25, 0.01, 2.1e11, 0.3, 1000.0
    physical = cn.run_cccc_n1_physical(a, b, t, E, nu, q, 16, 20, 4)
    D = bending_stiffness(E, t, nu)
    direct = cn.run_cccc_n1(a, b, D, q, 16, 20, 4)
    assert physical.D == D and physical.physical == {"t": t, "E": E, "nu": nu}
    assert np.array_equal(physical.field(), direct.field())
    assert np.array_equal(physical.dimensionless_field(), physical.field() * D / (q * a**4))
    centre = physical.field()[8, 10]
    assert physical.steps[-1].alpha_centre == float(centre) * D / (q * a**4)  # min(a, b) = a here


def test_the_units_of_the_load_and_the_stiffness():
    """w is linear in q and in 1/D through powers of two, exactly."""
    one = cn.run_cccc_n1(1.0, 1.0, 1.0, 1.0, 12, 12, 4).field()
    assert np.array_equal(cn.run_cccc_n1(1.0, 1.0, 1.0, 4.0, 12, 12, 4).field(), 4.0 * one)
    assert np.array_equal(cn.run_cccc_n1(1.0, 1.0, 4.0, 1.0, 12, 12, 4).field(), 0.25 * one)


def test_a_repeated_run_in_one_process_is_equal_bit_for_bit():
    one = cn.run_cccc_n1(1.0, 1.3, 1.0, 1.0, 16, 21, 8)
    other = cn.run_cccc_n1(1.0, 1.3, 1.0, 1.0, 16, 21, 8)
    assert np.array_equal(one.field(), other.field())
    assert [step.eta for step in one.steps] == [step.eta for step in other.steps]


# --- B-9: boundaries of the module ---------------------------------------------------------------------------------------------------

def test_the_module_imports_nothing_of_the_ssss_scheme_or_of_the_references():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    imported = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)} | \
               {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    assert imported <= {"math", "dataclasses", "fractions", "numpy", "scipy.fft", "kkvim.physical", "kkvim.r3_input", "kkvim.rectangle"}


def test_no_reference_module_is_loaded_by_the_command():
    code = "import sys, kkvim.cli, kkvim.cccc_n1, kkvim.cli_cccc; print([m for m in sys.modules if m.startswith('kkvim_reference')])"
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert done.stdout.strip() == "[]"


def test_the_forbidden_words_are_not_statuses_of_this_module():
    names = [value for name, value in vars(cn).items() if name.isupper() and isinstance(value, str)]
    assert "CONVERGED" not in names and not any(name in {"converged", "qualified", "PASS"} for name in names)
    assert not hasattr(cn.CCCCN1Result, "converged")


def test_the_normative_range_of_the_coupled_ssss_solver_is_as_it_was():
    assert N_SOLVE == (2, 3)
