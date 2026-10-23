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


# --- review 907de22: IE-1 (P-3), IE-2 (P-2, double range), IE-3 (input domain), N-1, N-2, N-4 ----------------------------

import platform  # noqa: E402
import warnings  # noqa: E402


def _measure_quiet(*args, **kwargs):
    """measure with every warning turned into an error: the instrument must not rely on warnings or raise them."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return e2.measure(e2.FieldInputs(*args, **kwargs)).values


@pytest.mark.parametrize("X, Y", [
    (np.full((2, 1), 2.0**-540), np.full((2, 1), 3 * 2.0**-540)),  # product 3 2^-1080 -> 0 (Reviewer, IE-1)
    (np.full((2, 1), 2.0**-530), np.full((2, 1), 2.0**-530 * (1 + 2.0**-40))),  # subnormal result (Reviewer, second example)
])
def test_ie1_reviewer_counterexamples_are_outside_p3(X, Y):
    """IE-1: outside P-3 the exact comparisons still show 'violations' (OBSERVATION), but VOK-4/VOK-5 get the status OUTSIDE_P3,
    not VIOLATED."""
    v = _measure_quiet(X, Y, 1.0, 3, 1.0, 3)
    assert v["premises"] == e2.OUTSIDE_P3
    assert v["vok4_status"] == v["vok5_status"] == e2.OUTSIDE_P3
    assert v["vok4_violations"] == 4 and v["vok5_holds"] is False  # observation kept, not a refutation


def test_ie1_subnormal_input_is_outside_p3():
    for X, Y in ((np.full((2, 1), 5e-324), np.ones((2, 1))), (np.ones((2, 2)), np.array([[1.0, 2.0**-1030], [1.0, 1.0]]))):
        v = _measure_quiet(X, Y, 1.0, 3, 1.0, 3)
        assert v["premises"] == e2.OUTSIDE_P3 and "nonzero subnormal input" in v["premise_reasons"]


def test_ie1_unprovable_absence_is_unresolved_not_violation():
    """Products 2^-1000 are normal, but e(X) + e(Y) < -918: absence of subnormal intermediates is not provable ->
    UNRESOLVED_PREMISES (the conservative criterion does not claim a violation)."""
    v = _measure_quiet(np.full((2, 2), 2.0**-500), np.full((2, 2), 2.0**-500), 1.0, 3, 1.0, 3)
    assert v["premises"] == e2.UNRESOLVED_PREMISES
    assert v["vok4_status"] == v["vok5_status"] == e2.UNRESOLVED_PREMISES
    assert v["vok4_violations"] == 0


@pytest.mark.parametrize("ey, expected", [(-459, e2.P3_CONFIRMED), (-460, e2.UNRESOLVED_PREMISES)])
def test_ie1_threshold_of_the_granularity_criterion(ey, expected):
    """e(X) + e(Y) = -918 is the edge of the proof (all values multiples of 2^-1022); -919 is not provable."""
    v = _measure_quiet(np.full((2, 1), 2.0**-459), np.full((2, 1), 2.0**ey), 1.0, 3, 1.0, 3)
    assert v["premises"] == expected


@pytest.mark.parametrize("N", [1, 2, 3])
def test_ie1_normal_half_step_data_are_confirmed(N):
    v = e2.measure(halfstep_inputs(N, "Y")).values
    assert v["premises"] == e2.P3_CONFIRMED and v["vok4_status"] == v["vok5_status"] == "HOLDS"


@pytest.mark.parametrize("X, Y", [
    (np.full((2, 1), 1e200), np.full((2, 1), 1e200)),  # (a) w^ overflows
    (np.array([[1e308, 1e308, -1e308]] * 2), np.ones((2, 3))),  # (b) a partial sum overflows, the exact sum 1e308 is finite
])
def test_ie2_overflow_of_the_field_is_outside_p2(X, Y):
    v = _measure_quiet(X, Y, 1.0, 3, 1.0, 3)
    assert v["premises"] == e2.OUTSIDE_P2 and v["vok4_status"] == v["vok5_status"] == e2.OUTSIDE_P2
    assert "max_abs_e_up" not in v and v["nonfinite_entries"] == 4


def test_ie2_overflow_only_in_component_iv_keeps_the_exact_measurement():
    """(c) field ~ 1e150 per node: the double a_h(w^, w^) overflows; (i) is measured exactly, (iv) is an OBSERVATION 'inf'."""
    v = _measure_quiet(np.full((3, 1), 1e150), np.full((3, 1), 1e150), 1.0, 4, 1.0, 4)
    assert v["premises"] == e2.P3_CONFIRMED and v["vok4_status"] == "HOLDS"
    assert v["w_a_sq_double_definition"] == "inf" and v["w_a_sq_double_definition_rel_error"] == e2.UNRESOLVED
    assert v["w_a_down"] < v["w_a_up"] < float("inf")


def test_ie2_exact_norm_above_dbl_max():
    """(d) finite field, ||w^||_a above DBL_MAX: the interval is [DBL_MAX, inf] (valid bounds), no exception."""
    v = _measure_quiet(np.full((3, 1), 1e10), np.ones((3, 1)), 1e-300, 4, 1e-300, 4)
    assert v["w_a_down"] == sys.float_info.max and v["w_a_up"] == float("inf")
    assert v["vok5_status"] == "HOLDS"


def test_ie2_grid_step_zero_in_double_skips_component_iv():
    v = _measure_quiet(np.ones((3, 1)), np.ones((3, 1)), 5e-324, 4, 1.0, 4)  # l_x/n_x = 0 in double
    assert v["component_iv"].startswith("not evaluated") and v["w_a_sq_double_definition"] == e2.UNRESOLVED
    assert v["vok4_status"] == "HOLDS"  # the exact measurement uses h = l/n exactly


@pytest.mark.parametrize("field, value", [("lx", 0.0), ("lx", -1.0), ("ly", float("inf")), ("ly", float("nan")), ("D", 0.0),
                                          ("D", -2.0), ("D", float("nan")), ("lx", True), ("lx", "1.0"), ("lx", 1 + 0j),
                                          ("lx", 10**400), ("lx", 2**53 + 1), ("nx", 2), ("nx", 5.0), ("nx", True),
                                          ("ny", np.float64(4))])
def test_ie3_parameter_domain(field, value):
    args = {"X": np.ones((3, 1)), "Y": np.ones((3, 1)), "lx": 1.0, "nx": 4, "ly": 1.0, "ny": 4, "D": 1.0}
    args[field] = value
    if field == "nx" and isinstance(value, int) and not isinstance(value, bool):
        args["X"] = np.ones((value - 1, 1))
    with pytest.raises(ValueError):
        e2.FieldInputs(**args)


def test_ie3_admissible_parameter_types():
    inp = e2.FieldInputs(np.ones((3, 1), dtype=np.int64), np.ones((3, 1), dtype=np.float32), 1, np.int64(4), np.float32(0.5),
                         4, D=np.float64(2.0))
    assert inp.lx == 1.0 and inp.nx == 4 and inp.ly == 0.5 and inp.D == 2.0 and inp.X.dtype == np.float64
    assert e2.measure(inp).values["premises"] == e2.P3_CONFIRMED


@pytest.mark.parametrize("bad", [np.ones((3, 1), dtype=complex), np.ones((3, 1), dtype=bool), np.array([["a"], ["b"], ["c"]]),
                                 np.array([[Fraction(1)], [Fraction(1)], [Fraction(1)]], dtype=object),
                                 np.full((3, 1), 2**60 + 1, dtype=np.int64)])
def test_ie3_arrays_must_be_real_and_exact(bad):
    with pytest.raises(ValueError):
        e2.FieldInputs(bad, np.ones((3, 1)), 1.0, 4, 1.0, 4)


def test_n1_runtime_provenance_without_host_name():
    prov = e2.measure(halfstep_inputs(1, "X")).provenance
    assert {"ftz_daz_selftest", "blas_runtime", "threads_env", "cpu_count", "cpu_features_used_by_numpy"} <= set(prov)
    assert "not a proof" in prov["ftz_daz_selftest"]
    assert platform.node() not in repr(prov)


def test_n2_digest_is_checked_at_start_and_end():
    inp = halfstep_inputs(2, "Y")
    v = e2.measure(inp).values
    assert v["digest"] == inp.digest()
    original = e2.field_standard

    def tamper(i):
        w = original(i)
        i.X.flags.writeable = True
        i.X[0, 0] += 1.0
        return w

    e2.field_standard = tamper
    try:
        with pytest.raises(RuntimeError):
            e2.measure(inp)
    finally:
        e2.field_standard = original


def test_n4_sqrt_interval_subnormal_and_large():
    tiny, big = Fraction(5e-324), Fraction(sys.float_info.max)
    for x in (tiny, tiny * tiny, Fraction(1, 2**2148), Fraction(2.0**-1022) ** 2, big, big * big, big * big * 4, 3 * tiny):
        lo, hi = e2.sqrt_interval(x)
        assert lo * lo <= x <= hi * hi and 0 < lo < hi
        assert (hi - lo) / lo <= Fraction(1, 2**63)


# --- review f5f85e1: IE-3a (exact scalar parameters), N-c (masked arrays) ----------------------------------------------------

EXTENDED_LONGDOUBLE = np.finfo(np.longdouble).nmant > 52


def _inputs_with(**kwargs):
    args = {"X": np.ones((3, 1)), "Y": np.ones((3, 1)), "lx": 1.0, "nx": 4, "ly": 1.0, "ny": 4, "D": 1.0}
    args.update(kwargs)
    return e2.FieldInputs(**args)


@pytest.mark.skipif(not EXTENDED_LONGDOUBLE, reason="numpy.longdouble is binary64 on this platform (nmant = 52): no value of it "
                                                    "is wider than a double, the IE-3a counterexample cannot be formed")
@pytest.mark.parametrize("field", ["lx", "ly", "D"])
def test_ie3a_extended_longdouble_is_refused(field):
    """IE-3a: 1 + 2^-60 in an extended long double would be rounded silently to 1.0; it is refused."""
    value = np.longdouble(1) + np.longdouble(2) ** -60
    assert value != np.longdouble(1.0)  # the value really is wider than binary64 here
    with pytest.raises(ValueError):
        _inputs_with(**{field: value})


@pytest.mark.parametrize("value", [1.5, np.float16(0.5), np.float32(0.25), np.float64(1.3), np.longdouble(0.75), np.longdouble(1.3),
                                   3, np.int64(7), np.uint16(9), 2**53])
@pytest.mark.parametrize("field", ["lx", "ly", "D"])
def test_ie3a_exactly_representable_values_are_accepted(field, value):
    """float, numpy float16/32/64, longdouble holding a binary64 value, and integers: accepted, stored as the same exact value."""
    inp = _inputs_with(**{field: value})
    stored = getattr(inp, field)
    assert isinstance(stored, float)
    exact = Fraction(int(value)) if isinstance(value, (int, np.integer)) else Fraction(*value.as_integer_ratio())
    assert Fraction(stored) == exact


@pytest.mark.parametrize("value", [2**53 + 1, 10**400, np.uint64(2**63 + 1), np.int64(2**62 + 1)])
@pytest.mark.parametrize("field", ["lx", "D"])
def test_ie3a_inexact_integers_are_refused(field, value):
    with pytest.raises(ValueError):
        _inputs_with(**{field: value})


@pytest.mark.skipif(not EXTENDED_LONGDOUBLE, reason="numpy.longdouble is binary64 on this platform: no extended value exists")
def test_ie3a_extended_longdouble_out_of_range_or_inexact_is_refused():
    for value in (np.longdouble(2) ** 1100, np.longdouble(2) ** -1100, np.longdouble(1) / np.longdouble(3)):
        with pytest.raises(ValueError):
            _inputs_with(lx=value)


def test_ie3a_domain_checks_still_hold():
    """Geometry and stiffness: zero, negative, infinite, NaN, bool, str, complex are refused as before; valid input is accepted."""
    for field in ("lx", "ly", "D"):
        for bad in (0.0, -1.0, float("inf"), float("nan"), True, "1.0", 1 + 0j, np.float32("nan")):
            with pytest.raises(ValueError):
                _inputs_with(**{field: bad})
    assert _inputs_with().lx == 1.0


def test_nc_masked_arrays_are_refused():
    """N-c: a masked array would lose its mask silently in the conversion; it is refused for X and Y."""
    masked = np.ma.masked_array(np.array([[1.0], [1e300], [1.0]]), mask=[[False], [True], [False]])
    with pytest.raises(ValueError):
        e2.FieldInputs(masked, np.ones((3, 1)), 1.0, 4, 1.0, 4)
    with pytest.raises(ValueError):
        e2.FieldInputs(np.ones((3, 1)), masked, 1.0, 4, 1.0, 4)
    with pytest.raises(ValueError):
        e2.FieldInputs(np.ma.masked_array(np.ones((3, 1))), np.ones((3, 1)), 1.0, 4, 1.0, 4)  # even without masked entries


class _WiderThanDouble(float):
    """A float whose reported exact value is wider than its double (as an extended long double would be): platform-independent
    check of the IE-3a logic, which compares the exact value of the argument with the double."""

    def as_integer_ratio(self):
        return (2**60 + 1, 2**60)


class _UnknownExactValue(float):
    def as_integer_ratio(self):
        raise TypeError("no exact value")


def test_ie3a_logic_on_every_platform():
    with pytest.raises(ValueError, match="not exactly representable"):
        _inputs_with(lx=_WiderThanDouble(1.0))
    with pytest.raises(ValueError, match="cannot be determined"):
        _inputs_with(D=_UnknownExactValue(1.0))
