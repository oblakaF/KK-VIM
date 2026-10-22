"""Self-checks of the E-2 instrument (kkvim_reference.r5_cccc_e2): formation error of the field W = U V^T.

Criteria, no tolerance and no threshold of E-2 (eps_floor^C, K_E are not assigned):
- exact rational identities (the norm ||.||_a by the definition of Pi_h^C equals D h_x h_y v^T A v with the 13-point matrix,
  Proposition D; the error e = w^ - w~ is exact);
- the proven inequalities VOK-1 (item 4), VOK-4 (elementwise, under P-1...P-5) and VOK-5 checked exactly;
- directed intervals: the exact value lies inside [down, up];
- structural facts of the method N-1: the swap of columns gives a false zero for N = 1 and for N = 2 with sequential summation,
  and ``measure`` does not use it.
Measured values (ratios to the bounds, errors of the double norm paths) are OBSERVATION and are not asserted.
"""

import subprocess
import sys
from fractions import Fraction

import numpy as np
import pytest

from kkvim_reference import r5_cccc_bc as bc
from kkvim_reference import r5_cccc_e2 as e2
from kkvim_reference import r5_cccc_halfstep as hs


def halfstep_inputs(N, direction, nx=12, ny=10, lx=1.0, ly=1.3):
    """Fixed arrays of a real half-step: given set after QR and the B-C solution (Y: X = given, Y = computed; X: reversed)."""
    n_f = nx if direction == "Y" else ny
    rec = bc.run_halfstep(hs.HalfStep(lx, nx, ly, ny, direction, hs.start_set(n_f, N)))
    S = rec.bc.s.reshape(N, -1).T
    G = rec.solved.given
    X, Y = (G, S) if direction == "Y" else (S, G)
    return e2.FieldInputs(X, Y, lx, nx, ly, ny, origin=f"B-C {direction} N={N}")


# --- own exact 13-point matrix (independent of the module) -----------------------------------------------------------

def _second(n, h):
    return [[Fraction(-2 if i == j else 1 if abs(i - j) == 1 else 0) / h**2 for j in range(n - 1)] for i in range(n - 1)]


def _d4(n, h):
    L = _second(n, h)
    m = n - 1
    D4 = [[sum(L[i][k] * L[k][j] for k in range(m)) for j in range(m)] for i in range(m)]
    D4[0][0] += 2 / h**4
    D4[m - 1][m - 1] += 2 / h**4
    return D4


def quad_13_point(inp, v):
    """D h_x h_y v^T A v, A = D4_x (x) I + 2 L_x (x) L_y + I (x) D4_y (NS 3.3), exactly; v[i][j], i - x, j - y."""
    hx, hy = Fraction(inp.lx) / inp.nx, Fraction(inp.ly) / inp.ny
    mx, my = inp.nx - 1, inp.ny - 1
    V = [[Fraction(x) for x in row] for row in v]
    D4x, D4y, Lx, Ly = _d4(inp.nx, hx), _d4(inp.ny, hy), _second(inp.nx, hx), _second(inp.ny, hy)
    AV = [[sum(D4x[i][k] * V[k][j] for k in range(mx)) + sum(V[i][k] * D4y[k][j] for k in range(my))
           + 2 * sum(Lx[i][k] * sum(V[k][l] * Ly[l][j] for l in range(my)) for k in range(mx)) for j in range(my)]
          for i in range(mx)]
    return Fraction(inp.D) * hx * hy * sum(V[i][j] * AV[i][j] for i in range(mx) for j in range(my))


# --- tests ------------------------------------------------------------------------------------------------------------

def test_instrument_does_not_import_the_main_solver():
    code = ("import sys, kkvim_reference.r5_cccc_e2 as m; "
            "print(any(k == 'kkvim' or k.startswith('kkvim.') for k in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    assert out == "False"


def test_inputs_are_fixed_and_identified():
    """Item 1: read-only copies; a later change of the source does not reach the inputs; the digest identifies the data."""
    X, Y = np.arange(1.0, 12.0).reshape(11, 1), np.linspace(-1, 1, 9).reshape(9, 1)
    inp = e2.FieldInputs(X, Y, 1.0, 12, 1.3, 10)
    d0 = inp.digest()
    X[0, 0] = 99.0
    assert inp.X[0, 0] == 1.0 and inp.digest() == d0
    with pytest.raises(ValueError):
        inp.X[0, 0] = 5.0
    assert e2.FieldInputs(X, Y, 1.0, 12, 1.3, 10).digest() != d0
    for bad in ((np.ones((10, 1)), Y), (X, np.ones((9, 2))), (np.full((11, 1), np.nan), Y), (np.ones((11, 4)), np.ones((9, 4)))):
        with pytest.raises(ValueError):
            e2.FieldInputs(*bad, 1.0, 12, 1.3, 10)


@pytest.mark.parametrize("nx, ny, lx, ly, D", [(3, 3, 1.0, 1.0, 1.0), (6, 4, 1.0, 1.7, 2.5), (5, 9, 0.7, 1.1, 0.3), (8, 8, 1.3, 1.3, 1.0)])
def test_energy_norm_by_the_definition_equals_the_13_point_form(nx, ny, lx, ly, D):
    """Item 3: ||v||_a^2 computed by the definition of Pi_h^C (NS 2.2) equals D h_x h_y v^T A v (Proposition D) EXACTLY."""
    rng = np.random.default_rng(nx * 10 + ny)
    inp = e2.FieldInputs(np.ones((nx - 1, 1)), np.ones((ny - 1, 1)), lx, nx, ly, ny, D=D)
    for v in (rng.standard_normal((nx - 1, ny - 1)), np.ones((nx - 1, ny - 1)), np.zeros((nx - 1, ny - 1))):
        assert e2.energy_sq(inp, v) == quad_13_point(inp, v)


@pytest.mark.parametrize("N", [1, 2, 3])
@pytest.mark.parametrize("direction", hs.DIRECTIONS)
def test_half_step_fields_both_directions(N, direction):
    """Item 5: N = 1-3, X and Y half-steps (B-C data): VOK-4 without violation, VOK-1 and VOK-5 hold exactly; intervals ordered;
    e is the exact difference of the standard field and the exact product."""
    inp = halfstep_inputs(N, direction)
    m = e2.measure(inp)
    v = m.values
    assert v["vok4_violations"] == 0 and v["vok1_upper_holds"] is True and v["vok5_holds"] is True
    for name in ("e_h", "e_a", "w_h", "w_a", "S_h", "eps_a_C", "rel_e_a", "kappa_W", "e_a_over_eps_a_C"):
        assert v[f"{name}_down"] <= v[f"{name}_up"]
    w_hat = inp.X @ inp.Y.T
    exact = e2.field_exact(inp)
    worst = max(abs(Fraction(float(w_hat[i, j])) - exact[i][j]) for i in range(inp.nx - 1) for j in range(inp.ny - 1))
    assert v["max_abs_e_up"] == hs._float_up(worst)
    assert set(m.provenance) >= {"python", "numpy", "scipy", "blas", "platform", "standard_call", "inputs_sha256"}
    assert v["w_a_sq_double_bound"] == e2.UNRESOLVED  # component (iv): no rigorous bound for the double paths


def test_zero_field():
    """Item 5: zero field: e = 0, norms 0, relative quantities and kappa_W UNRESOLVED (0/0)."""
    inp = e2.FieldInputs(np.zeros((9, 2)), np.ones((7, 2)), 1.0, 10, 1.0, 8)
    v = e2.measure(inp).values
    assert v["e_is_zero"] and v["max_abs_e_up"] == 0.0 and v["e_a_up"] == 0.0 and v["w_a_up"] == 0.0
    for key in ("max_rel_e_up", "rel_e_a_up", "kappa_W_up", "eps_a_C_rel_up", "e_a_over_eps_a_C_up"):
        assert v[key] == e2.UNRESOLVED
    assert v["vok4_violations"] == 0 and v["vok5_holds"] is True


@pytest.mark.parametrize("N", [1, 2, 3])
def test_negative_values(N):
    """Item 5: arrays with negative entries (random signs and magnitudes over 6 decades)."""
    rng = np.random.default_rng(N)
    X = rng.standard_normal((13, N)) * 10.0 ** rng.integers(-3, 3, (13, N))
    Y = -np.abs(rng.standard_normal((8, N))) * 10.0 ** rng.integers(-3, 3, (8, N))
    v = e2.measure(e2.FieldInputs(X, Y, 1.7, 14, 1.0, 9)).values
    assert v["vok4_violations"] == 0 and v["vok1_upper_holds"] and v["vok5_holds"]


def test_near_cancellation():
    """Item 5: w = x y^T - x~ y^T with x~ = fl(x (1 + 2^-30)): the terms almost cancel. kappa_W = ||S||_h/||w^||_h is about
    2/2^-30 = 2^31 by construction (relative perturbations of order 2^-23), so it lies in (2^30, 2^32); VOK-4, VOK-5 hold."""
    rng = np.random.default_rng(5)
    x, y = rng.uniform(0.5, 2.0, 11), rng.uniform(0.5, 2.0, 9)
    X = np.column_stack([x, x * (1 + 2.0**-30)])
    Y = np.column_stack([y, -y])
    v = e2.measure(e2.FieldInputs(X, Y, 1.0, 12, 1.0, 10)).values
    assert v["vok4_violations"] == 0 and v["vok5_holds"]
    assert 2.0**30 < v["kappa_W_down"] <= v["kappa_W_up"] < 2.0**32


def test_swap_of_columns_is_not_the_measure():
    """Item 6 (N-1): for N = 1 and for N = 2 with sequential summation the swap gives exactly 0 although the direct error e is not
    zero; ``measure`` works without the swap (it is replaced here by a function that fails) and reports no swap quantity."""
    one = halfstep_inputs(1, "Y")
    two = halfstep_inputs(2, "Y")
    assert e2.swap_difference(one) == 0.0 and e2.swap_difference(one, sequential=False) == 0.0
    assert e2.swap_difference(two, sequential=True) == 0.0  # false zero
    assert e2.measure(two).values["max_abs_e_up"] > 0  # the direct measure is not zero

    def fail(*args, **kwargs):
        raise AssertionError("swap used by measure")

    original = e2.swap_difference
    e2.swap_difference = fail
    try:
        v = e2.measure(two).values
    finally:
        e2.swap_difference = original
    assert not any("swap" in key or "perm" in key for key in v)


def test_sqrt_interval_contains_the_root():
    for x in (Fraction(0), Fraction(1, 3), Fraction(2), Fraction(10**40, 7), Fraction(1, 10**300), Fraction(49, 4)):
        lo, hi = e2.sqrt_interval(x)
        assert lo * lo <= x <= hi * hi and lo <= hi


def test_double_norm_paths_are_observations():
    """Item 4: the double evaluations of ||w^||_a^2 are reported with their exact relative error; no bound is claimed."""
    v = e2.measure(halfstep_inputs(2, "X")).values
    for path in ("definition", "hessian"):
        assert isinstance(v[f"w_a_sq_double_{path}"], float) and isinstance(v[f"w_a_sq_double_{path}_rel_error"], float)
    assert v["w_a_sq_double_bound"] == e2.UNRESOLVED
