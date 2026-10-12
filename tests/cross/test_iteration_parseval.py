"""QA cross tests: KK-VIM half-steps of kkvim against the semi-analytic Parseval recursion (R2, step 6).

QA tests (D-033, item 8): only QA cross tests import both packages; the packages never
import each other (D-028). Implementation under test: kkvim.iteration.run_halfsteps.
Independent reference: kkvim_reference.parseval_recursion.ParsevalRecursion (continuous
half-steps s = 1, 2, 3, NUMERICAL_SPEC.md 15). Comparisons only at the same s (15.7).

Criteria and tolerances of docs/NUMERICAL_SPEC.md, table 18.1, fixed before the comparison:
- A-5a: p = log2(E(n)/E(2n)) in [1.9, 2.1] on (80, 160), (160, 320) for the centre error and
  E_inf over all nodes (relative to max|w_SA|), s = 1, 2, 3; centre error decreasing from n = 20;
- A-5b: Richardson at the centre, w_RE = (4 w_2n[::2, ::2] - w_n)/3 on the nodes of grid n,
  |w_RE - w_SA| <= 0.05 |w_2n - w_SA| (empirical criterion, D-032);
- A-5v: invariants beta = B/A, gamma = C/A, chi = F^2/A, relative errors, p in [1.9, 2.1];
- A-5g: field and invariants with I-4 against a QA-only path without normalization,
  s = 1..6, n = 80, relative 1e-12;
- A-5d: s, k and direction equal for both, k = s // 2.
"""

import math
from functools import lru_cache

import numpy as np
import pytest

import kkvim.iteration as iteration
from kkvim.first_halfstep import first_halfstep, node_value, solve_halfstep
from kkvim.iteration import run_halfsteps
from kkvim.second_halfstep import second_halfstep
from kkvim_reference.parseval_recursion import ParsevalRecursion

P_WINDOW = (1.9, 2.1)
PAIRS = ((80, 160), (160, 320))
RICHARDSON_RATIO = 0.05
A5G_TOL = 1e-12
GRIDS = (20, 40, 80, 160, 320)
S_VALUES = (1, 2, 3)


@lru_cache(maxsize=None)
def reference():
    return ParsevalRecursion(a=1.0, q=1.0, D=1.0)


@lru_cache(maxsize=None)
def history(n):
    return run_halfsteps(1.0, 1.0, 1.0, n, n, 3)


@lru_cache(maxsize=None)
def reference_field(s, n):
    return reference().field_on_grid(s, n, n)


def errors(s, n):
    w, w_sa = history(n).field(s), reference_field(s, n)
    c = n // 2
    return abs(w[c, c] - w_sa[c, c]) / abs(w_sa[c, c]), np.max(np.abs(w - w_sa)) / np.max(np.abs(w_sa))


def invariants(G):
    return (G.B / G.A, G.C / G.A, G.F * G.F / G.A)


def invariant_errors(s, n):
    fd = invariants(history(n).record(s).coefficients)
    sa = reference().coefficients(s).invariants()
    return [abs(x - y) / abs(y) for x, y in zip(fd, sa)]


# A-5a -----------------------------------------------------------------------------


@pytest.mark.parametrize("s", S_VALUES)
@pytest.mark.parametrize("pair", PAIRS, ids=lambda p: f"{p[0]}-{p[1]}")
@pytest.mark.parametrize("which", ["centre", "field"])
def test_a5a_second_order(s, pair, which):
    n, n2 = pair
    k = 0 if which == "centre" else 1
    p = math.log2(errors(s, n)[k] / errors(s, n2)[k])
    assert P_WINDOW[0] <= p <= P_WINDOW[1], (s, pair, which, p)


@pytest.mark.parametrize("s", S_VALUES)
def test_a5a_centre_error_decreases_from_20(s):
    e_c = [errors(s, n)[0] for n in GRIDS]
    assert all(e1 > e2 for e1, e2 in zip(e_c, e_c[1:])), e_c


# A-5b -----------------------------------------------------------------------------


@pytest.mark.parametrize("s", S_VALUES)
@pytest.mark.parametrize("pair", PAIRS, ids=lambda p: f"{p[0]}-{p[1]}")
def test_a5b_richardson_centre(s, pair):
    n, n2 = pair
    coarse = history(n).field(s)
    fine_common = history(n2).field(s)[::2, ::2]  # grid-2n values at the nodes of grid n
    exact = reference_field(s, n)
    c = n // 2
    w_re = (4.0 * fine_common[c, c] - coarse[c, c]) / 3.0
    assert abs(w_re - exact[c, c]) <= RICHARDSON_RATIO * abs(fine_common[c, c] - exact[c, c])


# A-5v -----------------------------------------------------------------------------


@pytest.mark.parametrize("s", S_VALUES)
@pytest.mark.parametrize("pair", PAIRS, ids=lambda p: f"{p[0]}-{p[1]}")
@pytest.mark.parametrize("k, name", [(0, "beta"), (1, "gamma"), (2, "chi")])
def test_a5v_invariants_second_order(s, pair, k, name):
    n, n2 = pair
    p = math.log2(invariant_errors(s, n)[k] / invariant_errors(s, n2)[k])
    assert P_WINDOW[0] <= p <= P_WINDOW[1], (s, pair, name, p)


# A-5g -----------------------------------------------------------------------------


@lru_cache(maxsize=None)
def unnormalized_path(q, n, count):
    """QA-only control path: X0 and then every raw factor of the previous half-step, no I-4."""
    grid = run_halfsteps(1.0, 1.0, q, n, n, 1).record(1).grid_x
    x = np.sin(math.pi * np.arange(1, n) / n)
    y = None
    out = []
    for s in range(1, count + 1):
        if s % 2:
            step = solve_halfstep(grid, x, grid, q, 1.0)
            y = step.raw
        else:
            step = solve_halfstep(grid, y, grid, q, 1.0)
            x = step.raw
        out.append((np.outer(grid.with_boundary(x), grid.with_boundary(y)), step.coefficients))
    return out


@pytest.mark.parametrize("q", [1.0, -2.3])
@pytest.mark.parametrize("s", range(1, 7))
def test_a5g_normalization_invariance_up_to_s6(q, s):
    main = run_halfsteps(1.0, 1.0, q, 80, 80, 6)
    w_alt, coefficients_alt = unnormalized_path(q, 80, 6)[s - 1]
    w_main = main.field(s)
    assert np.max(np.abs(w_alt - w_main)) / np.max(np.abs(w_main)) <= A5G_TOL
    for x, y in zip(invariants(coefficients_alt), invariants(main.record(s).coefficients)):
        assert abs(x - y) <= A5G_TOL * abs(y)
    assert np.all(np.sign(w_main[1:-1, 1:-1]) == math.copysign(1.0, q))


# A-5d and structure ---------------------------------------------------------------------


def test_a5d_counters_and_directions():
    ref, h = reference(), run_halfsteps(1.0, 1.0, 1.0, 20, 20, 6)
    for s in S_VALUES:
        r = h.record(s)
        assert (r.s, r.k, r.direction) == (ref.halfstep(s).s, ref.halfstep(s).k, ref.halfstep(s).variable)
    assert [(r.s, r.k, r.direction) for r in h.records] == [(s, s // 2, "Y" if s % 2 else "X") for s in range(1, 7)]


def test_general_step_2_reproduces_qualified_second_halfstep():
    h = run_halfsteps(1.0, 1.0, -1.7, 20, 40, 2)
    w2 = second_halfstep(first_halfstep(1.0, 1.0, -1.7, 20, 40)).w2
    np.testing.assert_array_equal(h.field(2), w2)


def test_previous_field_lies_in_subspace_of_next_halfstep(monkeypatch):
    """ChV-11: the fixed factor of s is exactly the stored computed factor of s - 1, on the right grids."""
    seen = []
    original = iteration.solve_halfstep

    def spy(fixed_grid, fixed_factor, solve_grid, q, D):
        seen.append((fixed_grid, np.array(fixed_factor, copy=True), solve_grid))
        return original(fixed_grid, fixed_factor, solve_grid, q, D)

    monkeypatch.setattr(iteration, "solve_halfstep", spy)
    h = run_halfsteps(1.0, 1.0, 1.0, 20, 30, 6)
    for s in range(2, 7):
        fixed_grid, fixed, solve_grid = seen[s - 2]
        prev = h.record(s - 1)
        np.testing.assert_array_equal(fixed, prev.computed)
        if s % 2:  # Y-half-step: X fixed on grid_x
            assert fixed_grid is prev.grid_x and solve_grid is prev.grid_y
            np.testing.assert_array_equal(prev.x, fixed)
        else:
            assert fixed_grid is prev.grid_y and solve_grid is prev.grid_x
            np.testing.assert_array_equal(prev.y, fixed)


def test_node_value_returns_own_field_of_every_step():
    h = run_halfsteps(1.0, 1.0, 1.0, 20, 20, 6)
    values = [node_value(h.record(s), "1/2", "1/2") for s in range(1, 7)]
    assert values == [h.field(s)[10, 10] for s in range(1, 7)]
    assert len(set(values)) == 6
    first = first_halfstep(1.0, 1.0, 1.0, 20, 20)
    assert node_value(second_halfstep(first), "1/2", "1/2") == h.field(2)[10, 10]
    with pytest.raises(TypeError):
        node_value(object(), "1/2", "1/2")
