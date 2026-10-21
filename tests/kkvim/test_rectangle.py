"""Own tests of kkvim.rectangle: half-steps on the rectangle a x b (R3, stage A).

Checks follow docs/R3_NUMERICAL_SPEC.md 1.2, 2, 3 and table 9.1 with the tolerances of
NUMERICAL_SPEC.md 18.1 (D-032, D-039): A-3 <= 1e-13 rel., A-6 eta <= 1e-12,
A-8 <= 1e-12, A-9 <= 1e-11 relative to max |w_bar|. r = 1 must reproduce R2 exactly
(D-041: bitwise regression). The second start (ChV-R1) is built here directly from the
R2 building blocks on the grids of a x b. Comparisons with references are QA's.
"""

import math

import numpy as np
import pytest

from kkvim.coefficients import ERROR_INPUT, compute_coefficients
from kkvim.first_halfstep import HalfStepError, first_halfstep, node_value, normalization_factor, solve_halfstep
from kkvim.grid import Grid1D
from kkvim.iteration import run_halfsteps
from kkvim.linear_solver import backward_error
from kkvim.physical import PhysicalInputError, run_physical
from kkvim.rectangle import rectangle_first_halfstep, rectangle_halfsteps, run_r3_halfsteps
from kkvim.second_halfstep import second_halfstep

A3_TOL = 1e-13
ETA_MAX = 1e-12
A8_TOL = 1e-12
A9_TOL = 1e-11
D, Q = 3.1, 2.3

CASES = [(1.0, 0.5, 40, 20), (2.0, 3.0, 20, 36), (1.0, 5.0, 20, 100), (3.0, 1.0, 48, 16), (1.0, 0.2, 100, 20),
         (1.0, 1.4, 20, 28), (2.5, 1.0, 24, 40)]


def records_equal(r, s):
    assert (r.s, r.k, r.direction, r.grid_x, r.grid_y, r.D, r.q, r.c) == (s.s, s.k, s.direction, s.grid_x, s.grid_y,
                                                                          s.D, s.q, s.c)
    assert r.coefficients == s.coefficients and r.solver_diagnostics == s.solver_diagnostics
    for name in ("x", "y", "raw"):
        np.testing.assert_array_equal(getattr(r, name), getattr(s, name))
    np.testing.assert_array_equal(r.field(), s.field())


# r = 1: the R2 path is reproduced exactly ------------------------------------------------

@pytest.mark.parametrize("n_x, n_y", [(20, 20), (20, 32), (36, 24), (4, 4)])
def test_square_reproduces_r2_bitwise(n_x, n_y):
    a = 1.7
    old = run_halfsteps(a, D, Q, n_x, n_y, 6)
    new = rectangle_halfsteps(a, a, D, Q, n_x, n_y, 6)
    for r, s in zip(old.records, new.records, strict=True):
        records_equal(r, s)
    f_old, f_new = first_halfstep(a, D, Q, n_x, n_y), rectangle_first_halfstep(a, a, D, Q, n_x, n_y)
    np.testing.assert_array_equal(second_halfstep(f_old).w2, second_halfstep(f_new).w2)


def test_square_physical_input_reproduces_r2_physical():
    base = dict(a=1.0, t=0.01, E=2.1e11, nu=0.3, q=1e3)
    r2 = run_physical(**base, n_x=40, n_y=40)
    r3 = run_r3_halfsteps(b=1.0, **base, n_x=40, n_y=40, count=r2.run.s)
    assert r3.D == r2.D
    for r, s in zip(r2.run.history.records, r3.history.records, strict=True):
        records_equal(r, s)
    np.testing.assert_array_equal(r3.dimensionless_field(), r2.dimensionless_field())


# rectangles: grids, lengths of the solved half-steps, coefficients ----------------------

@pytest.mark.parametrize("a, b, n_x, n_y", CASES)
def test_grids_counters_and_field_shape(a, b, n_x, n_y):
    history = rectangle_halfsteps(a, b, D, Q, n_x, n_y, 5)
    for s, record in enumerate(history.records, start=1):
        assert (record.s, record.k, record.direction) == (s, s // 2, "Y" if s % 2 else "X")
        assert record.grid_x == Grid1D(a, n_x) and record.grid_y == Grid1D(b, n_y)
        w = record.field()
        assert w.shape == (n_x + 1, n_y + 1)
        assert not w[0].any() and not w[-1].any() and not w[:, 0].any() and not w[:, -1].any()
        assert np.all(w[1:-1, 1:-1] > 0.0)
        assert np.max(np.abs(record.computed)) == 1.0 and np.sum(record.computed) > 0.0


@pytest.mark.parametrize("a, b, n_x, n_y", CASES)
def test_first_coefficients_are_those_of_x0_on_g_a(a, b, n_x, n_y):
    c = rectangle_first_halfstep(a, b, D, Q, n_x, n_y).coefficients
    h = a / n_x
    mu = 4.0 / h**2 * math.sin(math.pi / (2 * n_x)) ** 2
    closed = (a / 2, a / 2 * mu, a / 2 * mu**2, Q * h / math.tan(math.pi / (2 * n_x)))  # ChV-6
    for value, exact in zip((c.A, c.B, c.C, c.F), closed, strict=True):
        assert abs(value - exact) <= A3_TOL * abs(exact)


@pytest.mark.parametrize("a, b, n_x, n_y", CASES)
def test_each_halfstep_is_solved_on_the_grid_of_its_own_side(a, b, n_x, n_y):
    gx, gy = Grid1D(a, n_x), Grid1D(b, n_y)
    history = rectangle_halfsteps(a, b, D, Q, n_x, n_y, 4)
    fixed = np.sin(math.pi * np.arange(1, n_x) / n_x)
    for record in history.records:
        fixed_grid, solve_grid, wrong = (gx, gy, Grid1D(a, n_y)) if record.direction == "Y" else (gy, gx, Grid1D(b, n_x))
        c = compute_coefficients(fixed_grid, fixed, Q)
        assert record.coefficients == c
        rhs = c.F * np.ones(solve_grid.interior_size)
        eta, _ = backward_error(solve_grid, c.A, c.B, c.C, D, record.raw, rhs)
        assert eta <= ETA_MAX and record.solver_diagnostics.eta <= ETA_MAX
        eta_wrong, _ = backward_error(wrong, c.A, c.B, c.C, D, record.raw, rhs)
        assert eta_wrong > 1e-6  # the length of the other side would not solve this system
        fixed = record.computed


@pytest.mark.parametrize("a, b, n_x, n_y", CASES)
def test_mirror_symmetry(a, b, n_x, n_y):
    for record in rectangle_halfsteps(a, b, D, Q, n_x, n_y, 4).records:
        w = record.field()
        peak = np.max(np.abs(w))
        assert np.max(np.abs(w - w[::-1, :])) <= A8_TOL * peak
        assert np.max(np.abs(w - w[:, ::-1])) <= A8_TOL * peak


def test_node_value_uses_fractions_of_each_side():
    record = rectangle_halfsteps(2.0, 3.0, D, Q, 20, 36, 3).records[-1]
    assert node_value(record, "1/2", "1/2") == record.field()[10, 18]
    assert node_value(record, "1/4", "1/3") == record.field()[5, 12]


# second start (ChV-R1): start along y on a x b == start along x on b x a, transposed ------

def start_y_fields(a, b, n_x, n_y, count):
    """Start Y0_j = sin(pi j / n_y) on G(b, n_y); odd s solve X on G(a, n_x), even s solve Y."""
    gx, gy = Grid1D(a, n_x), Grid1D(b, n_y)
    fixed = np.sin(math.pi * np.arange(1, n_y) / n_y)
    fields = []
    for s in range(1, count + 1):
        fixed_grid, solve_grid = (gy, gx) if s % 2 else (gx, gy)
        step = solve_halfstep(fixed_grid, fixed, solve_grid, Q, D)
        c = normalization_factor(step.raw)
        computed, scaled = step.raw / c, fixed * c
        x, y = (computed, scaled) if s % 2 else (scaled, computed)
        fields.append(np.outer(gx.with_boundary(x), gy.with_boundary(y)))
        fixed = computed
    return fields


@pytest.mark.parametrize("a, b, n_x, n_y", CASES)
def test_swapped_sides_equal_second_start_transposed(a, b, n_x, n_y):
    swapped = rectangle_halfsteps(b, a, D, Q, n_y, n_x, 5)  # b x a, start along x' = y
    for s, (record, w) in enumerate(zip(swapped.records, start_y_fields(a, b, n_x, n_y, 5), strict=True), start=1):
        assert (record.s, record.k) == (s, s // 2)  # counters kept, Y/X labels exchanged
        assert record.grid_x == Grid1D(b, n_y) and record.grid_y == Grid1D(a, n_x)
        np.testing.assert_array_equal(record.field().T, w)


def test_start_direction_matters_for_a_rectangle():
    w_x = rectangle_halfsteps(1.0, 2.0, D, Q, 20, 40, 1).records[0].field()
    w_y = start_y_fields(1.0, 2.0, 20, 40, 1)[0]
    assert np.max(np.abs(w_x - w_y)) > 1e-3 * np.max(np.abs(w_x))


# scaling of w_bar at fixed r (A-9, ChV-14) ------------------------------------------------

BASE = dict(a=1.0, t=0.01, E=2.1e11, nu=0.3, q=1e3)
E_NU = BASE["E"] * (1 - 0.1**2) / (1 - 0.3**2)  # keeps D when nu: 0.3 -> 0.1
VARIANTS = {
    "q x 3.7": {"q": 3.7 * BASE["q"]},
    "E x 0.013": {"E": 0.013 * BASE["E"]},
    "t x 1.9": {"t": 1.9 * BASE["t"]},
    "nu 0.3 -> 0.1, same D": {"nu": 0.1, "E": E_NU},
    "a and b x 2.9": {"a": 2.9 * BASE["a"], "b_factor": 2.9},
}
# table 9.1, A-9: family I (320, 320) and family II for each r
A9_GRIDS = {1 / 5: (320, 64), 1 / 3: (288, 96), 1 / 2: (320, 160), 2 / 3: (240, 160), 1.0: (320, 320),
            7 / 5: (160, 224), 3 / 2: (160, 240), 2.0: (160, 320), 3.0: (96, 288), 5.0: (64, 320)}
A9_COUNT = 7


def a9_run(r, n_x, n_y, change):
    change = dict(change)
    b = r * change.pop("b_factor", 1.0)
    return run_r3_halfsteps(**{**BASE, **change, "b": b}, n_x=n_x, n_y=n_y, count=A9_COUNT)


@pytest.mark.parametrize("r", list(A9_GRIDS), ids=lambda r: f"r={r:.4g}")
@pytest.mark.parametrize("family", ["I", "II"])
def test_a9_scaling_at_fixed_r(r, family):
    n_x, n_y = (320, 320) if family == "I" else A9_GRIDS[r]
    base = a9_run(r, n_x, n_y, {})
    for name, change in VARIANTS.items():
        run = a9_run(r, n_x, n_y, change)
        for s in range(1, A9_COUNT + 1):
            w_ref = base.dimensionless_field(s)
            deviation = np.max(np.abs(run.dimensionless_field(s) - w_ref)) / np.max(np.abs(w_ref))
            assert deviation <= A9_TOL, (name, s, deviation)


def test_dimensionless_field_uses_reference_side_a():
    run = run_r3_halfsteps(**{**BASE, "a": 2.0}, b=3.0, n_x=20, n_y=30, count=3)
    for s in (1, 3):
        expected = run.history.field(s) * run.D / (BASE["q"] * 2.0**4)  # w D / (q a^4), R3 1.2
        np.testing.assert_array_equal(run.dimensionless_field(s), expected)
    np.testing.assert_array_equal(run.dimensionless_field(), run.dimensionless_field(3))


# input ---------------------------------------------------------------------------------

@pytest.mark.parametrize(
    "change",
    [{"b": 0.0}, {"b": -1.0}, {"b": math.nan}, {"b": math.inf}, {"b": True}, {"b": "1"}, {"b": None},
     {"b": 5.1}, {"b": 0.19}, {"a": 0.0}, {"t": 0.0}, {"E": math.nan}, {"nu": 1.0}, {"q": 0.0},
     {"n_x": 3}, {"n_y": 20.0}, {"E": 1e300, "t": 1e200}],
)
def test_invalid_input_is_error_input(change):
    args = {**BASE, "b": 2.0, "n_x": 20, "n_y": 40, **change}
    with pytest.raises(PhysicalInputError) as info:
        run_r3_halfsteps(**args, count=2)
    assert info.value.code == ERROR_INPUT


def test_low_level_rejects_sides_and_d():
    with pytest.raises(PhysicalInputError):
        rectangle_first_halfstep(1.0, 6.0, D, Q, 20, 20)
    with pytest.raises(HalfStepError) as info:
        rectangle_first_halfstep(1.0, 2.0, -1.0, Q, 20, 20)
    assert info.value.code == ERROR_INPUT
    with pytest.raises(ValueError):
        rectangle_halfsteps(1.0, 2.0, D, Q, 20, 20, 0)
