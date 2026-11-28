"""R6 postprocessing kkvim.moments: the implementation tests PM-1...PM-6, PM-8 of the R6 review contract (section 8.7).

The expectations and tolerances are those of the contract, fixed before the implementation (D-025); they test the
implementation (E_impl), not the accuracy of the moments of a computed field (which is not qualified: RESEARCH / NOT_QUALIFIED).
PM-7 (E_der on exact fields) belongs to QA; ``test_pm7_rule_a_first_order_on_exact_data`` only documents the expected
first order of rule A on EXACT data (review 6.1) so that a later reader does not mistake it for a defect.
"""

import math

import numpy as np
import pytest

from kkvim import moments as mo
from kkvim.cccc_n1 import run_cccc_n1

U = 2.0**-53


def nodes(n, length):
    return np.linspace(0.0, length, n + 1)


# ------------------------------------------------------------------------------------------------ PM-1: operator exactness


@pytest.mark.parametrize("degree", [0, 1, 2, 3])
def test_pm1_interior_second_difference_is_exact_up_to_cubics(degree):
    x = nodes(20, 1.7)
    h = x[1]
    coefs = np.arange(1, degree + 2, dtype=float) * (-1) ** np.arange(degree + 1)
    f = np.polyval(coefs, x)
    exact = np.polyval(np.polyder(coefs, 2), x) if degree >= 2 else np.zeros_like(x)
    got = mo.second_derivative_nodes(f, h, "SS")
    scale = np.max(np.abs(f)) / h**2
    assert np.max(np.abs(got[1:-1] - exact[1:-1])) <= 100 * U * scale


@pytest.mark.parametrize("degree", [0, 1, 2])
def test_pm1_interior_first_difference_is_exact_up_to_quadratics(degree):
    x = nodes(16, 2.3)
    h = x[1]
    coefs = np.arange(2, degree + 3, dtype=float)
    f = np.polyval(coefs, x)
    exact = np.polyval(np.polyder(coefs, 1), x) if degree >= 1 else np.zeros_like(x)
    got = mo.first_derivative_nodes(f, h, "CC")
    assert np.max(np.abs(got[1:-1] - exact[1:-1])) <= 100 * U * np.max(np.abs(f)) / h


def test_pm1_cc_edge_rule_a_is_exact_for_c_x2():
    x = nodes(12, 1.0)
    h = x[1]
    c = 3.25
    f = c * x**2 * (1.0 - x) ** 0  # f0 = 0, f'(0) = 0
    f2 = c * (x[-1] - x) ** 2  # the mirrored case at the far edge
    assert mo.second_derivative_nodes(f, h, "CC")[0] == pytest.approx(2 * c, rel=100 * U)
    assert mo.second_derivative_nodes(f2, h, "CC")[-1] == pytest.approx(2 * c, rel=100 * U)


@pytest.mark.parametrize("degree", [1, 2])
def test_pm1_ss_edge_first_difference_is_exact_up_to_quadratics_with_f0_zero(degree):
    x = nodes(10, 1.3)
    h = x[1]
    a1, a2 = 0.7, (-1.9 if degree == 2 else 0.0)
    f = a1 * x + a2 * x**2  # f(0) = 0
    g = a1 * (x[-1] - x) + a2 * (x[-1] - x) ** 2  # zero at the far edge
    d = mo.first_derivative_nodes(f, h, "SS")
    e = mo.first_derivative_nodes(g, h, "SS")
    scale = 100 * U * np.max(np.abs(f)) / h
    assert abs(d[0] - a1) <= scale
    assert abs(e[-1] - (-a1)) <= scale


def test_operators_set_the_exact_edge_values():
    f = np.sin(np.pi * nodes(8, 1.0))
    assert mo.second_derivative_nodes(f, 0.125, "SS")[[0, -1]].tolist() == [0.0, 0.0]
    assert mo.first_derivative_nodes(f, 0.125, "CC")[[0, -1]].tolist() == [0.0, 0.0]


# ------------------------------------------------------------------------------------------------ manufactured fields


def ssss_field(nx, ny, a, b):
    x, y = nodes(nx, a), nodes(ny, b)
    return x, y, np.outer(np.sin(np.pi * x / a), np.sin(np.pi * y / b))


def test_manufactured_ssss_moments_converge_with_second_order_and_have_the_exact_signs():
    a, b, D, nu = 2.0, 1.0, 19230.0, 0.3
    errors = []
    for n in (16, 32, 64):
        x, y, W = ssss_field(2 * n, n, a, b)
        W = W.copy()
        W[[0, -1], :] = 0.0
        W[:, [0, -1]] = 0.0
        M = mo.plate_moments(W, a, b, D, nu, "SS")
        kx, ky = math.pi / a, math.pi / b
        sx, sy = np.sin(kx * x), np.sin(ky * y)
        cx, cy = np.cos(kx * x), np.cos(ky * y)
        Mx = D * (kx**2 + nu * ky**2) * np.outer(sx, sy)
        My = D * (ky**2 + nu * kx**2) * np.outer(sx, sy)
        Mxy = D * (1 - nu) * kx * ky * np.outer(cx, cy)
        errors.append(max(np.max(np.abs(M.Mx - Mx)), np.max(np.abs(M.My - My)), np.max(np.abs(M.Mxy - Mxy))))
        assert M.Mx[n, n // 2] > 0 and M.Mxy[0, 0] > 0 and M.Mxy[-1, 0] < 0  # signs of (101), (102)
    orders = [math.log2(errors[k] / errors[k + 1]) for k in range(2)]
    assert all(1.9 <= p <= 2.1 for p in orders), orders


# ------------------------------------------------------------------------------------------------ PM-2: field vs factors


def padded(v):
    out = np.zeros(v.shape[0] + 2) if v.ndim == 1 else np.zeros((v.shape[0] + 2, v.shape[1]))
    out[1:-1] = v
    return out


@pytest.mark.parametrize("bc", ["SS", "CC"])
def test_pm2_field_path_equals_the_factor_path(bc):
    rng = np.random.default_rng(7)
    a, b, D, nu = 1.5, 1.0, 1000.0, 0.3
    X = padded(rng.standard_normal((23, 3)))
    Y = padded(rng.standard_normal((17, 3)))
    M = mo.plate_moments(X @ Y.T, a, b, D, nu, bc)
    Fx, Fy, Fxy = mo.moments_from_factors(X, Y, a, b, D, nu, bc)
    hmin = min(a / 24, b / 18)
    tol = 100 * U * D * np.sum(np.abs(X).sum(axis=0) * np.abs(Y).sum(axis=0)) * hmin**-2
    for p, q in ((M.Mx, Fx), (M.My, Fy), (M.Mxy, Fxy)):
        assert np.max(np.abs(p - q)) <= tol


def test_pm2_on_a_computed_cccc_n1_field():
    result = run_cccc_n1(1.0, 1.0, 1000.0, 1.0, 16, 16, 4)
    X, Y = result.factors()
    X, Y = padded(np.asarray(X)), padded(np.asarray(Y))
    W = result.field()
    M = mo.plate_moments(W, 1.0, 1.0, 1000.0, 0.3, "CC")
    Fx, Fy, Fxy = mo.moments_from_factors(X, Y, 1.0, 1.0, 1000.0, 0.3, "CC")
    tol = 100 * U * 1000.0 * np.sum(np.abs(X)) * np.sum(np.abs(Y)) * 16**2
    assert max(np.max(np.abs(M.Mx - Fx)), np.max(np.abs(M.My - Fy)), np.max(np.abs(M.Mxy - Fxy))) <= tol


# ------------------------------------------------------------------------------------------------ PM-3: boundary identities


def cccc_field(nx, ny, a, b):
    x, y = nodes(nx, a), nodes(ny, b)
    W = np.outer(np.sin(np.pi * x / a) ** 2 * (1 + 0.3 * x), np.sin(np.pi * y / b) ** 2)
    W[[0, -1], :] = 0.0
    W[:, [0, -1]] = 0.0
    return W


def test_pm3_ssss_edges_have_zero_mx_my_and_nonzero_twist():
    _, _, W = ssss_field(20, 12, 2.0, 1.2)
    W = W.copy()
    W[[0, -1], :] = 0.0
    W[:, [0, -1]] = 0.0
    M = mo.plate_moments(W, 2.0, 1.2, 50.0, 0.25, "SS")
    for edge in (M.Mx[0], M.Mx[-1], M.Mx[:, 0], M.Mx[:, -1], M.My[0], M.My[-1], M.My[:, 0], M.My[:, -1]):
        assert np.all(edge == 0.0)
    assert M.Mxy[0, 0] != 0.0


@pytest.mark.parametrize("field_kind", ["manufactured", "solver"])
def test_pm3_cccc_edge_identities(field_kind):
    a, b, D, nu = 1.4, 1.0, 1000.0, 0.3
    if field_kind == "solver":
        W = run_cccc_n1(a, b, D, 1.0, 14, 10, 3).field()
    else:
        W = cccc_field(14, 10, a, b)
    M = mo.plate_moments(W, a, b, D, nu, "CC")
    big = np.max(np.abs([M.Mx, M.My]))
    for edge in (M.Mxy[0], M.Mxy[-1], M.Mxy[:, 0], M.Mxy[:, -1]):
        assert np.all(edge == 0.0)
    assert np.max(np.abs(M.My[[0, -1], :] - nu * M.Mx[[0, -1], :])) <= 4 * U * big
    assert np.max(np.abs(M.Mx[:, [0, -1]] - nu * M.My[:, [0, -1]])) <= 4 * U * big
    for i, j in ((0, 0), (0, -1), (-1, 0), (-1, -1)):
        assert M.Mx[i, j] == 0.0 and M.My[i, j] == 0.0 and M.Mxy[i, j] == 0.0


# ------------------------------------------------------------------------------------------------ PM-4: symmetry and a <-> b


@pytest.mark.parametrize("bc", ["SS", "CC"])
def test_pm4_swapping_the_sides_swaps_the_moments(bc):
    a, b, D, nu = 1.6, 1.0, 700.0, 0.2
    W = cccc_field(16, 10, a, b) if bc == "CC" else ssss_field(16, 10, a, b)[2].copy()
    W[[0, -1], :] = 0.0
    W[:, [0, -1]] = 0.0
    M = mo.plate_moments(W, a, b, D, nu, bc)
    S = mo.plate_moments(W.T.copy(), b, a, D, nu, bc)
    scale = np.max(np.abs([M.Mx, M.My, M.Mxy]))
    assert np.max(np.abs(M.Mx - S.My.T)) <= 4 * U * scale
    assert np.max(np.abs(M.My - S.Mx.T)) <= 4 * U * scale
    assert np.max(np.abs(M.Mxy - S.Mxy.T)) <= 4 * U * scale


def test_pm4_symmetric_field_gives_symmetric_moments():
    W = ssss_field(16, 12, 1.0, 1.0)[2].copy()
    W[[0, -1], :] = 0.0
    W[:, [0, -1]] = 0.0
    W = 0.5 * (W + W[::-1, :])
    M = mo.plate_moments(W, 1.0, 1.0, 1.0, 0.3, "SS")
    # "scale" of PM-4 for a mirrored stencil: the size of its terms, i.e. the rounding model 4u max|W| / h^2 D (1 + |nu|)
    # of review 8.6 (the order of the additions is mirrored, not the values); docs/R6_MOMENTS_IMPLEMENTATION.md, 6
    assert np.max(np.abs(M.Mx - M.Mx[::-1, :])) <= M.rounding_model
    assert np.max(np.abs(M.Mxy + M.Mxy[::-1, :])) <= M.rounding_model


# ------------------------------------------------------------------------------------------------ PM-5: thickness and rotation


def gauss_integral(fun, t, points=8):
    z, wts = np.polynomial.legendre.leggauss(points)
    return sum(w * fun(0.5 * t * zz) for zz, w in zip(z, wts)) * 0.5 * t


def test_pm5_stresses_integrate_to_the_moments_and_vanish_at_mid_plane():
    W = cccc_field(12, 12, 1.0, 1.0)
    t = 0.012
    M = mo.plate_moments(W, 1.0, 1.0, 1500.0, 0.3, "CC")
    sx = gauss_integral(lambda z: mo.stresses_at(M, t, z)[0] * z, t)
    sy = gauss_integral(lambda z: mo.stresses_at(M, t, z)[1] * z, t)
    txy = gauss_integral(lambda z: mo.stresses_at(M, t, z)[2] * z, t)
    scale = np.max(np.abs([M.Mx, M.My, M.Mxy]))
    assert np.max(np.abs(sx - M.Mx)) <= 1e-12 * scale
    assert np.max(np.abs(sy - M.My)) <= 1e-12 * scale
    assert np.max(np.abs(txy + M.Mxy)) <= 1e-12 * scale  # int tau_xy z dz = -M_xy (vector convention, review 3.1)
    mid = mo.stresses_at(M, t, 0.0)
    assert all(np.all(s == 0.0) for s in mid)
    surf = mo.surface_stresses(M, t)
    for top, bottom in zip(surf["top"], surf["bottom"]):
        assert np.array_equal(top, -bottom)
    assert np.allclose(surf["bottom"][0], 6 * M.Mx / t**2, rtol=1e-15, atol=0)


def test_pm5_rotation_gives_the_principal_moments():
    rng = np.random.default_rng(3)
    Mx, My, Mxy = rng.standard_normal((3, 200))
    M1, M2, alpha = mo.principal(Mx, My, Mxy)
    for k in range(200):
        Mn, Mnt = mo.moment_on_normal(Mx[k], My[k], Mxy[k], alpha[k])
        assert Mn == pytest.approx(M1[k], rel=1e-12, abs=1e-12)
        assert Mnt == pytest.approx(0.0, abs=1e-12 * (abs(M1[k]) + abs(M2[k])))
        assert -math.pi / 2 < alpha[k] <= math.pi / 2 and M1[k] >= M2[k]


def test_principal_direction_is_undefined_only_where_rho_is_zero():
    M1, M2, alpha = mo.principal([1.0, 1.0, 0.0], [1.0, 2.0, 0.0], [0.0, 0.0, 0.0])
    assert math.isnan(alpha[0]) and math.isnan(alpha[2]) and not math.isnan(alpha[1])
    assert M1[0] == M2[0] == 1.0


def test_z_outside_the_thickness_is_refused():
    M = mo.plate_moments(cccc_field(8, 8, 1.0, 1.0), 1.0, 1.0, 1.0, 0.3, "CC")
    with pytest.raises(mo.MomentsInputError):
        mo.stresses_at(M, 0.01, 0.0051)


# ------------------------------------------------------------------------------------------------ PM-6: invariance of the set


def test_pm6_change_of_the_factor_basis_leaves_the_moments():
    rng = np.random.default_rng(11)
    X = padded(rng.standard_normal((19, 3)))
    Y = padded(rng.standard_normal((15, 3)))
    G = rng.standard_normal((3, 3)) + 3 * np.eye(3)
    a, b, D, nu = 1.2, 1.0, 10.0, 0.3
    M = mo.plate_moments(X @ Y.T, a, b, D, nu, "CC")
    P = mo.moments_from_factors(X @ G, Y @ np.linalg.inv(G).T, a, b, D, nu, "CC")
    model = 100 * M.rounding_model * np.linalg.cond(G)
    for p, q in zip((M.Mx, M.My, M.Mxy), P):
        assert np.max(np.abs(p - q)) <= model


# ------------------------------------------------------------------------------------------------ PM-8: rule A on a discrete solution


def clamped_beam(n):
    """Discrete beam w'''' = 1 on [0, 1], w = 0 at both ends, reflection w[-1] = w[1] (the solver's D4, first row 7, -4, 1)."""
    h = 1.0 / n
    m = n - 1
    A = np.zeros((m, m))
    for i in range(m):
        for d, c in ((-2, 1.0), (-1, -4.0), (0, 6.0), (1, -4.0), (2, 1.0)):
            if 0 <= i + d < m:
                A[i, i + d] += c
    A[0, 0] += 1.0  # reflection w[-1] = w[1]
    A[-1, -1] += 1.0
    w = np.zeros(n + 1)
    w[1:-1] = np.linalg.solve(A / h**4, np.ones(m))
    return w, h


def test_pm8_rule_a_is_second_order_on_the_discrete_solution():
    exact = 1.0 / 12.0  # w = x^2 (1 - x)^2 / 24, w''(0) = 1/12
    errors_a, errors_b = [], []
    for n in (40, 80, 160, 320):
        w, h = clamped_beam(n)
        errors_a.append(abs(mo.second_derivative_nodes(w, h, "CC")[0] - exact))
        errors_b.append(abs((8 * w[1] - w[2]) / (2 * h * h) - exact))  # rule B: only here, never a postprocessing option
    orders_a = [math.log2(errors_a[k] / errors_a[k + 1]) for k in range(3)]
    orders_b = [math.log2(errors_b[k] / errors_b[k + 1]) for k in range(3)]
    assert all(abs(p - 2.0) <= 0.1 for p in orders_a), orders_a
    assert all(abs(p - 1.0) <= 0.1 for p in orders_b), orders_b  # B is first order on the computed field (review 6.1)


def test_pm7_rule_a_first_order_on_exact_data():
    errors = []
    for n in (40, 80, 160):
        x = nodes(n, 1.0)
        w = x**2 * (1 - x) ** 2 / 24
        errors.append(abs(mo.second_derivative_nodes(w, x[1], "CC")[0] - 1.0 / 12.0))
    orders = [math.log2(errors[k] / errors[k + 1]) for k in range(2)]
    assert all(abs(p - 1.0) <= 0.1 for p in orders), orders


# ------------------------------------------------------------------------------------------------ input checks, read-only input


@pytest.mark.parametrize("change", ["edge", "nan", "nu", "D", "small", "bc"])
def test_input_errors_are_refused(change):
    W = cccc_field(8, 8, 1.0, 1.0)
    args = dict(W=W, a=1.0, b=1.0, D=1.0, nu=0.3, bc="CC")
    if change == "edge":
        W[0, 3] = 1e-9
    elif change == "nan":
        W[3, 3] = np.nan
    elif change == "nu":
        args["nu"] = 1.0
    elif change == "D":
        args["D"] = 0.0
    elif change == "small":
        args["W"] = np.zeros((4, 9))
    else:
        args["bc"] = "CS"
    with pytest.raises(mo.MomentsInputError, match="ERROR_INPUT"):
        mo.plate_moments(**args)


def test_input_field_is_never_changed_and_outputs_are_read_only():
    W = cccc_field(10, 8, 1.0, 1.0)
    before = W.copy()
    W.setflags(write=False)
    M = mo.plate_moments(W, 1.0, 1.0, 1.0, 0.3, "CC")
    assert np.array_equal(W, before)
    with pytest.raises(ValueError):
        M.Mx[1, 1] = 0.0


def test_summary_centre_and_extrema():
    W = cccc_field(10, 8, 1.0, 1.0)
    x, y = nodes(10, 1.0), nodes(8, 1.0)
    M = mo.plate_moments(W, 1.0, 1.0, 1.0, 0.3, "CC")
    s = mo.field_summary(M.Mx, x, y)
    assert s["centre"]["i"] == 5 and s["centre"]["j"] == 4 and s["centre"]["value"] == M.Mx[5, 4]
    assert s["max"]["value"] == np.max(M.Mx) and s["min"]["value"] == np.min(M.Mx)
    assert mo.centre_index((10, 9)) is None
    assert M.bar("Mx", 0.0) is None
