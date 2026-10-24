"""Tests of the certified DST-I reference (kkvim_reference.r5_cccc_dst_cert; G-3 / OK-7).

Criteria: exact containment (rational / integer checks), independent second routes (Euler pi, cos Taylor, closed algebraic values of
special angles), interval identities (symmetry, orthogonality, sin^2 + cos^2 = 1, double angle), refusal of a certificate when the
reference is too wide, resource limits, platform independence (digest of the integer tables). No tolerance taken from observations;
measured errors of SciPy are OBSERVATION.
"""

import hashlib
import subprocess
import sys
from fractions import Fraction

import numpy as np
import pytest

from kkvim_reference import r5_cccc_bc as bc
from kkvim_reference import r5_cccc_dst_cert as dc

BITS = 160
ONE = 1 << BITS
# pi to 60 decimals (test data only, published constant): 3.14159265358979323846264338327950288419716939937510582097494459...
PI_60 = Fraction("3.141592653589793238462643383279502884197169399375105820974944")
PI_60_ERR = Fraction(1, 10**60)


def frac(v):
    return Fraction(v, ONE)


def test_module_does_not_import_the_main_solver():
    code = "import sys, kkvim_reference.r5_cccc_dst_cert; print(any(k == 'kkvim' or k.startswith('kkvim.') for k in sys.modules))"
    assert subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip() == "False"


def test_reference_does_not_use_floating_sin_or_pi():
    """The reference functions do not call math.sin, math.pi, numpy or scipy (they appear only in the diagnostics of the double
    path, which are not part of the reference)."""
    import inspect
    for fn in (dc._arctan_inv, dc.pi_bounds, dc._series_at, dc._sin_interval, dc._reduce, dc.sin_pi_rational, dc.sqrt2n_bounds,
               dc.psi_table, dc.certified_dst, dc.boundary_psi, dc.mu_bounds, dc.orthogonality_check):
        src = inspect.getsource(fn)
        for banned in ("math.sin", "math.pi", "np.sin", "np.pi", "scipy", "mpmath"):
            assert banned not in src, (fn.__name__, banned)


@pytest.mark.parametrize("route", ["machin", "euler"])
def test_pi_bounds_contain_pi(route):
    lo, hi = dc.pi_bounds(BITS, route)
    assert lo < hi and frac(lo) <= PI_60 + PI_60_ERR and frac(hi) >= PI_60 - PI_60_ERR
    assert frac(hi - lo) < Fraction(1, 2**(BITS - 4))


def test_special_angles_closed_algebraic_values():
    """sin(pi/6) = 1/2, sin(pi/4)^2 = 1/2, sin(pi/3)^2 = 3/4, sin(pi/2) = 1, sin(0) = sin(pi) = 0, sin^2(pi/5) = (5 - sqrt5)/8,
    sin(pi/10) = (sqrt5 - 1)/4 (closed forms; the square-root comparisons are done by squaring exact rationals). Both routes."""
    for route in ("A", "B"):
        s, lo, hi = dc.sin_pi_rational(1, 6, BITS, route)
        assert s == 1 and frac(lo) <= Fraction(1, 2) <= frac(hi)
        s, lo, hi = dc.sin_pi_rational(1, 4, BITS, route)
        assert frac(lo) ** 2 <= Fraction(1, 2) <= frac(hi) ** 2
        s, lo, hi = dc.sin_pi_rational(1, 3, BITS, route)
        assert frac(lo) ** 2 <= Fraction(3, 4) <= frac(hi) ** 2
        assert dc.sin_pi_rational(1, 2, BITS, route) == (1, ONE, ONE)
        assert dc.sin_pi_rational(0, 7, BITS, route)[1:] == (0, 0) and dc.sin_pi_rational(7, 7, BITS, route)[1:] == (0, 0)
        assert dc.sin_pi_rational(3, 2, BITS, route) == (-1, ONE, ONE)
        _, lo, hi = dc.sin_pi_rational(1, 5, BITS, route)  # 5 - 8 s^2 = sqrt5 in [5 - 8 hi^2, 5 - 8 lo^2]
        a, b = 5 - 8 * frac(hi) ** 2, 5 - 8 * frac(lo) ** 2
        assert 0 < a and a * a <= 5 <= b * b
        _, lo, hi = dc.sin_pi_rational(1, 10, BITS, route)  # 4 s + 1 = sqrt5
        assert (4 * frac(lo) + 1) ** 2 <= 5 <= (4 * frac(hi) + 1) ** 2


@pytest.mark.parametrize("n", [3, 4, 7, 16, 32, 64])
def test_two_independent_routes_overlap(n):
    """Route A (Machin pi, sin Taylor) and route B (Euler pi, cos Taylor of the complementary angle) give intervals that both contain
    the true value, hence overlap, for every angle pi j/n and pi j/(2n); widths are tiny."""
    for q in (n, 2 * n):
        for j in range(0, q + 1):
            sa, la, ha = dc.sin_pi_rational(j, q, BITS, "A")
            sb, lb, hb = dc.sin_pi_rational(j, q, BITS, "B")
            assert sa == sb and max(la, lb) <= min(ha, hb)
            assert ha - la < 2**16 and hb - lb < 2**16


@pytest.mark.parametrize("p, q", [(1, 7), (2, 9), (3, 11), (5, 16), (1, 64), (31, 64), (17, 40)])
def test_trigonometric_identities_in_interval_arithmetic(p, q):
    """sin^2 t + cos^2 t = 1 with cos t = sin(pi/2 - t) = sin(pi (q - 2p)/(2q)); sin 2t = 2 sin t cos t."""
    _, sl, sh = dc.sin_pi_rational(p, q, BITS)
    sc, cl, ch = dc.sin_pi_rational(q - 2 * p, 2 * q, BITS)
    c_lo, c_hi = (frac(cl), frac(ch)) if sc > 0 else (-frac(ch), -frac(cl))
    sq = lambda lo, hi: (0 if lo <= 0 <= hi else min(lo * lo, hi * hi), max(lo * lo, hi * hi))  # noqa: E731
    s2, c2 = sq(frac(sl), frac(sh)), sq(c_lo, c_hi)
    assert s2[0] + c2[0] <= 1 <= s2[1] + c2[1]
    sd, dl, dh = dc.sin_pi_rational(2 * p, q, BITS)
    d_lo, d_hi = (frac(dl), frac(dh)) if sd > 0 else (-frac(dh), -frac(dl))
    prods = [2 * a * b for a in (frac(sl), frac(sh)) for b in (c_lo, c_hi)]
    assert max(d_lo, min(prods)) <= min(d_hi, max(prods))


@pytest.mark.parametrize("n", [2, 3, 4, 7, 16, 18, 32, 64])
def test_sqrt_bounds(n):
    lo, hi = dc.sqrt2n_bounds(n, BITS)
    assert frac(lo) ** 2 * n <= 2 <= frac(hi) ** 2 * n
    if n in (2, 8, 18, 32, 50):  # sqrt(2/n) = sqrt(2n)/n rational
        assert hi - lo <= 1


@pytest.mark.parametrize("n", [3, 4, 7, 16])
def test_symmetry_boundary_signs_and_orthogonality(n):
    """Psi_ik = Psi_ki (same intervals), psi_k(m) = (-1)^(k+1) psi_k(1) exactly, and Psi Psi = I in interval arithmetic."""
    tab = dc.psi_table(n, BITS)
    for i in range(1, n):
        for k in range(1, n):
            assert tab.entry(i, k) == tab.entry(k, i)
    for z in dc.boundary_psi(n, BITS):
        a, b = z["psi_1"], z["psi_m"]
        assert b == (a if z["k"] % 2 == 1 else (-a[1], -a[0]))
    o = dc.orthogonality_check(n, BITS)
    assert o["contains_identity"] and o["max_width"] < Fraction(1, 2**(BITS - 16))


def test_n4_exact_values():
    """n = 4: Psi e_1 = (1/2, sqrt2/2, 1/2): the certified intervals contain the exact values."""
    r = dc.certified_dst(np.array([1.0, 0.0, 0.0]), BITS)
    (a0, b0), (a1, b1), (a2, b2) = (r.interval(i) for i in range(3))
    assert a0 <= Fraction(1, 2) <= b0 and a2 <= Fraction(1, 2) <= b2 and a1 * a1 <= Fraction(1, 2) <= b1 * b1


def test_zero_vector_and_signs_and_cancellation():
    """Zero vector: exact zero intervals, relative error UNRESOLVED. Negation is exact (lo(-x) = -hi(x)). Cancellation: the input
    Psi e_1 rounded (y close to e_1, the other components close to 0) is handled; the intervals of the near-zero components contain
    the value exactly and are tiny."""
    z = dc.certified_dst(np.zeros(15), BITS)
    assert all(v == 0 for v in z.lo + z.hi)
    c = dc.compare(bc.dst(np.zeros(15)), z, np.zeros(15))
    assert c["abs_err_2_up"] == 0.0 and c["rel_err_2_up"].startswith("UNRESOLVED") and c["resolution"].startswith("UNRESOLVED")
    assert c["c_dst_ratio_up"].startswith("UNRESOLVED")
    x = np.random.default_rng(1).standard_normal(15) * 10.0 ** np.random.default_rng(2).integers(-8, 8, 15)
    p, m = dc.certified_dst(x, BITS), dc.certified_dst(-x, BITS)
    assert p.exp == m.exp and m.lo == [-v for v in p.hi] and m.hi == [-v for v in p.lo]
    psi1 = bc.dst(np.eye(15)[0])  # double approximation of psi_1: y = Psi psi1 ~ e_1
    r = dc.certified_dst(psi1, BITS)
    widths = [b - a for a, b in (r.interval(i) for i in range(15))]
    assert r.interval(0)[0] > Fraction(9, 10) and max(widths) < Fraction(1, 2**(BITS - 16))
    assert all(abs(r.interval(i)[1]) < Fraction(1, 10**14) for i in range(1, 15))


@pytest.mark.parametrize("n", [3, 4, 7, 16, 32, 64])
def test_scipy_comparison_is_certified_and_resolved(n):
    """SciPy DST-I against the reference: the error interval is certified, and resolved (reference width <= 1/100 of the measured
    error) for random vectors; the comparison is consistent with route B (dense intervals from route B contain the same truth)."""
    rng = np.random.default_rng(n)
    for x in (rng.standard_normal(n - 1), np.ones(n - 1), -np.abs(rng.standard_normal(n - 1)) * 1e-200):
        ref = dc.certified_dst(x, BITS)
        c = dc.compare(bc.dst(x), ref, x)
        assert c["status"] == "CERTIFIED" and c["abs_err_2_down"] <= c["abs_err_2_up"]
        if c["abs_err_2_down"] > 0:
            assert c["resolution"].startswith("RESOLVED")
        # route B: own dense intervals; the true y_0 is in both
        lo = hi = Fraction(0)
        for k in range(1, n):
            s, a, b = dc.sin_pi_rational(k, n, BITS, "B")
            c_lo, c_hi = (frac(v) for v in dc.sqrt2n_bounds(n, BITS))
            e = sorted([s * c_lo * frac(a), s * c_hi * frac(b), s * c_lo * frac(b), s * c_hi * frac(a)])
            xv = Fraction(float(x[k - 1]))
            t = sorted([e[0] * xv, e[-1] * xv])
            lo, hi = lo + t[0], hi + t[1]
        a0, b0 = ref.interval(0)
        assert max(a0, lo) <= min(b0, hi)


def test_refusal_of_a_false_certificate_at_low_precision():
    """With too few bits the reference is wider than 1/100 of the SciPy error: the comparison is UNRESOLVED (never RESOLVED); the
    low-precision error interval still contains the high-precision one (containment does not depend on the precision)."""
    x = np.random.default_rng(7).standard_normal(31)
    y = bc.dst(x)
    hi_ref, lo_ref = dc.compare(y, dc.certified_dst(x, BITS), x), dc.compare(y, dc.certified_dst(x, 40), x)
    assert hi_ref["resolution"].startswith("RESOLVED")
    assert lo_ref["resolution"].startswith("UNRESOLVED")
    assert lo_ref["abs_err_2_down"] <= hi_ref["abs_err_2_down"] <= hi_ref["abs_err_2_up"] <= lo_ref["abs_err_2_up"]


def test_resource_limits_give_unresolved_resources():
    x = np.ones(63)
    assert dc.certified_dst(x, BITS, max_entries=100).status == dc.UNRESOLVED_RESOURCES
    assert dc.certified_dst(x, BITS, time_limit=0.0).status == dc.UNRESOLVED_RESOURCES
    assert dc.compare(bc.dst(x), dc.certified_dst(x, BITS, max_entries=100))["status"] == dc.UNRESOLVED_RESOURCES


@pytest.mark.parametrize("bad", [np.ones(5, dtype=np.float32), np.ones((2, 3)), np.array([1.0, np.nan]), np.array([np.inf]),
                                 np.array([], dtype=float), np.array([1, 2], dtype=np.int64)])
def test_inputs_must_be_finite_binary64_vectors(bad):
    with pytest.raises(ValueError):
        dc.certified_dst(bad, BITS)


def test_platform_independent_digest():
    """The certified integer tables are pure integer arithmetic: their digest is the same on every platform (Windows, Linux)."""
    h = hashlib.sha256()
    for n in (7, 16):
        t = dc.psi_table(n, BITS)
        for j in sorted(t.mag):
            h.update(f"{n}:{j}:{t.mag[j][0]}:{t.mag[j][1]};".encode())
    h.update(repr(dc.pi_bounds(BITS)).encode())
    assert h.hexdigest() == "d32fece813ab7cfdcead60939c7ffbb7bb545256db74b2b1624203b8422b87e5"


@pytest.mark.parametrize("n", [3, 16, 64])
def test_boundary_psi_mu_and_double_diagnostics(n):
    """E-4 quantities: psi_k(1), psi_k(m), products z z^T and mu_k intervals contain the double values up to a few u; the exact
    split of the double psi_k(1) error (normalisation, argument, sin evaluation, product rounding) sums to an interval containing the
    total error (identity of the split)."""
    z = dc.boundary_psi(n, BITS)
    for row in z:
        for key in ("psi_1_sq", "psi_m_sq"):
            assert row[key][0] >= 0
    mus = dc.mu_bounds(n, 1.3, BITS)
    md = bc.modal_eigenvalues(n, 1.3 / n)
    for (lo, hi), v in zip(mus, md):
        assert lo <= hi and abs(Fraction(float(v)) - lo) <= 16 * dc.U * hi + (hi - lo)
    for d in dc.double_boundary_diagnostics(n, BITS):
        parts = [d["norm_part"], d["arg_part"], d["sin_part"], (d["product_part"], d["product_part"])]
        s_lo, s_hi = sum(p[0] for p in parts), sum(p[1] for p in parts)
        assert max(s_lo, d["total"][0]) <= min(s_hi, d["total"][1])
        assert abs(d["total"][1]) <= 8 * dc.U
