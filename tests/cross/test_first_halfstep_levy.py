"""QA cross tests: first half-step s = 1 of kkvim against the analytic Levy reference (R2, step 4).

Only QA cross tests may import both packages (D-028, D-033, item 8); the packages
themselves never import each other. Implementation under test: kkvim.first_halfstep.
Independent reference: kkvim_reference.levy (MATH_SPEC.md, В-29, В-30) - the first
half-step, not the Navier solution.

Criteria and tolerances are those of docs/NUMERICAL_SPEC.md, section 14 and table 18.1,
fixed before the comparison and not tuned to results:
- A-4a: observed order p = log2(E(n)/E(2n)) in [1.9, 2.1] on (80, 160), (160, 320),
  separately for the centre error e_c and the field error e_inf; e_c decreases from n = 20;
- A-4b: Richardson w_RE = (4 w_2n[::2, ::2] - w_n)/3 on the nodes of grid n;
  |w_RE - w_ex| <= 0.05 |w_2n[::2, ::2] - w_ex| at the centre and in the max norm over the
  nodes of grid n (empirical criterion, D-032; not a proof of an error expansion);
- A-4v: Levy centre against 0.004109346839, <= 5e-13 absolute (check of the reference);
- A-6: eta <= 1e-12, Im <= 1e-12 after <= 1 refinement;
- dimensionless centre for other data a, D, q: A-9 tolerance 1e-11 relative.
Grids n_x != n_y are checked for the axes and grids used (D-032, item 10); the rounding
tolerance there is 64 u, not an accuracy criterion.
"""

import math
from functools import lru_cache

import numpy as np
import pytest

from kkvim.first_halfstep import NotANodeError, first_halfstep, node_value
from kkvim.linear_solver import ETA_MAX, IMAG_MAX
from kkvim_reference.levy import levy_w1

U = 2.0**-53
A4A_WINDOW = (1.9, 2.1)
A4A_PAIRS = ((80, 160), (160, 320))
A4B_RATIO = 0.05
A4V_CENTRE = 0.004109346839
A4V_TOL = 5e-13
A9_TOL = 1e-11  # table 18.1, A-9: scaling, relative
GRIDS = (20, 40, 80, 160, 320)
CASES = {
    "unit": (1.0, 1.0, 1.0),  # a, D, q (NUMERICAL_SPEC.md, 14)
    "general": (2.5, 0.8, -3.7),  # other data: w_bar does not depend on a, D, q (ChV-14)
}


@lru_cache(maxsize=None)
def computed(case, n_x, n_y):
    a, D, q = CASES[case]
    return first_halfstep(a, D, q, n_x, n_y)


@lru_cache(maxsize=None)
def levy_field(case, n_x, n_y):
    a, D, q = CASES[case]
    x = np.arange(n_x + 1) * a / n_x
    y = np.arange(n_y + 1) * a / n_y
    X, Y = np.meshgrid(x, y, indexing="ij")  # axis 0 = x, axis 1 = y
    return levy_w1(X, Y, a=a, q=q, D=D)


def errors(case, n):
    w, w_ex = computed(case, n, n).w1, levy_field(case, n, n)
    c = n // 2
    e_c = abs(w[c, c] - w_ex[c, c]) / abs(w_ex[c, c])
    e_inf = np.max(np.abs(w - w_ex)) / np.max(np.abs(w_ex))
    return e_c, e_inf


# A-4a ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("pair", A4A_PAIRS, ids=lambda p: f"{p[0]}-{p[1]}")
@pytest.mark.parametrize("which", ["centre", "field"])
def test_a4a_second_order(case, pair, which):
    n, n2 = pair
    k = 0 if which == "centre" else 1
    p = math.log2(errors(case, n)[k] / errors(case, n2)[k])
    assert A4A_WINDOW[0] <= p <= A4A_WINDOW[1], (case, pair, which, p)


@pytest.mark.parametrize("case", CASES)
def test_a4a_centre_error_decreases_from_20(case):
    e_c = [errors(case, n)[0] for n in GRIDS]
    assert all(e1 > e2 for e1, e2 in zip(e_c, e_c[1:])), e_c


# A-4b ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("pair", A4A_PAIRS, ids=lambda p: f"{p[0]}-{p[1]}")
def test_a4b_richardson_centre_and_field(case, pair):
    n, n2 = pair
    coarse = computed(case, n, n).w1
    fine_common = computed(case, n2, n2).w1[::2, ::2]  # grid-2n values at the nodes of grid n
    exact = levy_field(case, n, n)
    np.testing.assert_array_equal(levy_field(case, n2, n2)[::2, ::2], exact)  # same physical nodes
    w_re = (4.0 * fine_common - coarse) / 3.0
    c = n // 2
    assert abs(w_re[c, c] - exact[c, c]) <= A4B_RATIO * abs(fine_common[c, c] - exact[c, c])
    assert np.max(np.abs(w_re - exact)) <= A4B_RATIO * np.max(np.abs(fine_common - exact))


# A-4v ---------------------------------------------------------------------------


def test_a4v_levy_centre_value():
    w_bar = float(levy_w1(0.5, 0.5, a=1.0, q=1.0, D=1.0))
    assert abs(w_bar - A4V_CENTRE) <= A4V_TOL


@pytest.mark.parametrize("case", CASES)
def test_dimensionless_centre_is_a_node_value(case):
    a, D, q = CASES[case]
    r = computed(case, 80, 80)
    w_bar = node_value(r, "1/2", "1/2") * D / (q * a**4)
    assert w_bar == r.w1[40, 40] * D / (q * a**4)
    # Same dimensionless centre for both data sets (scheme scaling ChV-14; A-9 tolerance 1e-11).
    unit = computed("unit", 80, 80).w1[40, 40]
    assert abs(w_bar - unit) <= A9_TOL * abs(unit)


# A-6 ----------------------------------------------------------------------------


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("n", GRIDS)
def test_a6_solver_control(case, n):
    d = computed(case, n, n).solver_diagnostics
    assert d.eta <= ETA_MAX and d.imag_ratio <= IMAG_MAX and d.refinements <= 1


# Grids n_x != n_y ------------------------------------------------------------------


UNEQUAL = ((20, 40), (40, 20), (21, 20), (20, 21))


@pytest.mark.parametrize("n_x, n_y", UNEQUAL)
def test_unequal_grids_axes_shape_and_boundary(n_x, n_y):
    r = computed("unit", n_x, n_y)
    w = r.w1
    assert w.shape == (n_x + 1, n_y + 1)
    assert r.grid_x.n == n_x and r.grid_y.n == n_y
    assert not w[0].any() and not w[-1].any() and not w[:, 0].any() and not w[:, -1].any()


@pytest.mark.parametrize("n_x, n_y", UNEQUAL)
def test_unequal_grids_coefficients_on_x_grid(n_x, n_y):
    # ChV-6 closed forms for X0 on G(a, n_x); A-3 tolerance 1e-13.
    a, _, q = CASES["unit"]
    h = a / n_x
    mu1 = 4.0 / h**2 * math.sin(math.pi / (2 * n_x)) ** 2
    expected = (a / 2, a / 2 * mu1, a / 2 * mu1**2, q * h / math.tan(math.pi / (2 * n_x)))
    cf = computed("unit", n_x, n_y).coefficients
    for value, exp in zip((cf.A, cf.B, cf.C, cf.F), expected):
        assert abs(value - exp) <= 1e-13 * abs(exp)


@pytest.mark.parametrize("n_x, n_y", UNEQUAL)
def test_unequal_grids_field_matches_levy_structure_on_correct_nodes(n_x, n_y):
    """w1 = X0 (x) Y and Levy = sin(pi x/a) (x) Y1 on the same nodes: their ratio is constant along x.

    Holds to rounding only if axis 0 is x on G(a, n_x) and the reference is evaluated at
    x_i = i a/n_x, y_j = j a/n_y. The ratio along y is the half-step error Y_h/Y1, not tested here.
    """
    w, w_ex = computed("unit", n_x, n_y).w1, levy_field("unit", n_x, n_y)
    ratio = w[1:-1, 1:-1] / w_ex[1:-1, 1:-1]
    spread = np.max(np.abs(ratio - ratio[n_x // 2 - 1 : n_x // 2, :]), axis=0)
    assert np.all(spread <= 64 * U)


@pytest.mark.parametrize("n_x, n_y", [(21, 20), (20, 21), (21, 21)])
def test_centre_is_not_a_node_for_odd_n(n_x, n_y):
    with pytest.raises(NotANodeError):
        node_value(computed("unit", n_x, n_y), "1/2", "1/2")
