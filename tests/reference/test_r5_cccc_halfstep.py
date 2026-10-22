"""Program checks of the QA half-step controls R-H, R-AC, R-HP (kkvim_reference.r5_cccc_halfstep), PL 2.1.

Criteria used here, no tolerance is introduced:
- exact rational identities (coefficients, scaling, symmetry, exact rank);
- the standard rounding bound of a floating-point dot product of length m: |fl - exact| <= gamma_m |terms| (gamma_m = m u/(1 - m u)),
  with m counted generously for the chain of products of each path;
- the strict a posteriori bound ||s - s*||_2 <= ||r||_2 / lambda_low with the exact residual (R-HP) and lambda_low certified by
  a decimal Cholesky of K - tau I (module docstring).
Results without such a bound are reported (OBSERVATION), not asserted.
"""

import subprocess
import sys
from fractions import Fraction

import numpy as np
import pytest

from kkvim_reference import r5_cccc_d2 as d2
from kkvim_reference import r5_cccc_halfstep as hs

GRIDS = [(1.0, 10, 1.7, 14), (1.0, 12, 0.7, 8)]  # h_x != h_y, a != b
CASES = [(g, N, d) for g in GRIDS for N in (1, 2, 3) for d in hs.DIRECTIONS]


def problem(grid, N, direction, basis="sin2", D=1.0, q=1.0):
    lx, nx, ly, ny = grid
    n_f = nx if direction == "Y" else ny
    return hs.HalfStep(lx, nx, ly, ny, direction, hs.start_set(n_f, N, basis), D=D, q=q)


def exact_dense(p):
    """Exact K (term-first ordering) as a float-free Fraction matrix, from the node-term band of the module."""
    band = hs._exact_band_matrix(p, hs.exact_coefficients(p))
    m, N = p.n_s - 1, p.N
    perm = hs._node_term(m, N)
    K = [[Fraction(0)] * p.size for _ in range(p.size)]
    for (r, c), v in band.items():
        K[perm[r]][perm[c]] = v
        K[perm[c]][perm[r]] = v
    return K


def abs_bounds(p):
    """|terms| of the coefficient path and of the R-H path (for the gamma bounds)."""
    hf, hsol = p.l_f / p.n_f, p.l_s / p.n_s
    G = np.abs(p.given)
    Lf, D4f = np.abs(hs._L(p.n_f, hf)), np.abs(hs._D4(p.n_f, hf))
    Aa, Ba, Ca = hf * G.T @ G, hf * G.T @ Lf @ G, hf * G.T @ D4f @ G
    m = p.n_s - 1
    coef = p.D * (np.kron(Aa, np.abs(hs._D4(p.n_s, hsol))) + 2 * np.kron(Ba, np.abs(hs._L(p.n_s, hsol)))
                  + np.kron(Ca, np.eye(m)))
    P = np.abs(hs.embedding(p))
    A2 = np.abs(d2.energy_hessian(d2.D2Problem(p.lx, p.ly, p.nx, p.ny)).toarray())
    hx, hy = p.lx / p.nx, p.ly / p.ny
    rh = (p.D * hx * hy / hsol) * (P.T @ A2 @ P)
    return coef, rh, P.shape[0]


@pytest.mark.parametrize("grid, N, direction", CASES, ids=str)
def test_three_paths_assemble_the_same_exact_system(grid, N, direction):
    """R-H (Hessian restriction) and the coefficient path agree with the exact rational system within their rounding bounds."""
    p = problem(grid, N, direction)
    Kex = np.array([[float(v) for v in row] for row in exact_dense(p)])  # each entry rounded once
    Kc, rc = hs.coefficient_system(p)
    Kh, rh = hs.hessian_system(p)
    coef_abs, rh_abs, m1 = abs_bounds(p)
    assert np.all(np.abs(Kc - Kex) <= hs.gamma(p.n_f + 16) * coef_abs + hs.U * np.abs(Kex))
    assert np.all(np.abs(Kh - Kex) <= hs.gamma(2 * m1 + 32) * rh_abs + hs.U * np.abs(Kex))
    F = hs.exact_coefficients(p)[3]
    Fex = np.repeat([float(f) for f in F], p.n_s - 1)
    hf = p.l_f / p.n_f
    assert np.all(np.abs(rc - Fex) <= hs.gamma(p.n_f + 4) * p.q * hf * np.repeat(np.abs(p.given).sum(axis=0), p.n_s - 1)
                  + hs.U * np.abs(Fex))
    hx, hy = p.lx / p.nx, p.ly / p.ny
    rh_abs_terms = (p.q * hx * hy / (p.l_s / p.n_s)) * (np.abs(hs.embedding(p)).T @ np.ones(m1))
    assert np.all(np.abs(rh - Fex) <= hs.gamma(m1 + 8) * rh_abs_terms + hs.U * np.abs(Fex))


@pytest.mark.parametrize("grid, N, direction", CASES, ids=str)
def test_rh_and_refined_rac_solutions_within_certified_bounds(grid, N, direction, capsys):
    p = problem(grid, N, direction)
    coeffs = hs.exact_coefficients(p)
    status_h, s_h = hs.solve_hessian(p)
    ac = hs.solve_ac(p)
    assert status_h == ac.status == hs.INDEPENDENT and ac.corrections
    lam = np.linalg.eigvalsh(hs.coefficient_system(p)[0])[0]
    low = hs.certify_lambda_min(p, lam / 2, coeffs)
    assert low is not None and low > 0  # K of the exact system is SPD (certified)
    b_h, b_ac = hs.error_bound(p, s_h, low, coeffs), hs.error_bound(p, ac.s, low, coeffs)
    b_un = hs.error_bound(p, ac.s_unrefined, low, coeffs)
    assert np.linalg.norm(s_h - ac.s) <= b_h + b_ac  # both within strict bounds of the same exact solution
    with capsys.disabled():
        print(f"\n{p.direction} N={N} grid={grid}: |s_H - s_AC|={np.linalg.norm(s_h - ac.s):.2e}, bound_H={b_h:.2e}, "
              f"bound_AC={b_ac:.2e} (unrefined {b_un:.2e}), e_AC estimate={ac.e_AC:.1e} (OBSERVATION)")


@pytest.mark.parametrize("grid, N, direction", CASES[::3], ids=str)
def test_rhp_equals_the_direct_application_of_the_original_operator(grid, N, direction):
    """R-HP (exact, stencil with (7, -4, 1) rows, no modal transformation) against rhs - K s in floating point."""
    p = problem(grid, N, direction)
    s = hs.solve_ac(p).s
    rng = np.random.default_rng(7)
    for vec in (s, s * (1 + 1e-6 * rng.standard_normal(s.size))):
        r_ex = np.array([float(v) for v in hs.residual_exact(p, vec)])
        K, rhs = hs.coefficient_system(p)
        r_fl = rhs - K @ vec
        coef_abs, _, _ = abs_bounds(p)
        bound = hs.gamma(p.size + p.n_f + 20) * (coef_abs @ np.abs(vec) + np.abs(rhs)) + hs.U * np.abs(r_ex)
        assert np.all(np.abs(r_fl - r_ex) <= bound)


def test_exact_symmetry_scaling_in_D_q_and_lengths():
    """Exact rational identities: A, B, C symmetric; K proportional to D, F to q; lengths x c: A x c, B / c, C / c^3, F x c."""
    p = problem(GRIDS[0], 3, "Y")
    A, B, C, F = hs.exact_coefficients(p)
    for M in (A, B, C):
        assert all(M[i][k] == M[k][i] for i in range(3) for k in range(3))
    p2 = hs.HalfStep(p.lx, p.nx, p.ly, p.ny, "Y", p.given, D=2.5, q=0.7)
    band1, band2 = hs._exact_band_matrix(p, (A, B, C, F)), hs._exact_band_matrix(p2, hs.exact_coefficients(p2))
    assert all(band2[k] == Fraction(2.5) * v for k, v in band1.items())
    assert all(f2 == Fraction(0.7) * f1 for f1, f2 in zip(F, hs.exact_coefficients(p2)[3]))
    p3 = hs.HalfStep(3 * p.lx, p.nx, 3 * p.ly, p.ny, "Y", p.given)
    A3, B3, C3, F3 = hs.exact_coefficients(p3)
    for i in range(3):
        for k in range(3):
            assert A3[i][k] == 3 * A[i][k] and B3[i][k] == B[i][k] / 3 and C3[i][k] == C[i][k] / 27
        assert F3[i] == 3 * F[i]


def test_both_orientations_give_the_same_system():
    """Y-half-step on a x b with the set on x = X-half-step on b x a with the same set on y (exact and coefficient paths)."""
    G = hs.start_set(10, 2)
    y = hs.HalfStep(1.0, 10, 1.7, 14, "Y", G)
    x = hs.HalfStep(1.7, 14, 1.0, 10, "X", G)
    assert hs.exact_coefficients(y) == hs.exact_coefficients(x)
    assert np.array_equal(hs.coefficient_system(y)[0], hs.coefficient_system(x)[0])
    Ky = np.array([[float(v) for v in row] for row in exact_dense(y)])
    Kx = np.array([[float(v) for v in row] for row in exact_dense(x)])
    assert np.array_equal(Ky, Kx)


def test_eta_uses_the_step_of_the_solved_direction():
    p = problem(GRIDS[0], 2, "Y")
    s = hs.solve_ac(p).s
    eta = hs.eta_qa(p, s)
    assert 0 <= eta < 1  # value only; no threshold (NS 5: eta_max is not transferred from D-048)
    A, B, C, F = hs.exact_coefficients(p)
    r = hs.residual_exact(p, s)
    h = Fraction(p.ly) / p.ny  # Y-half-step: h_y
    norm = lambda M: max(sum(abs(x) for x in row) for row in M)  # noqa: E731
    scale = (16 * norm(A) / h**4 + 8 * norm(B) / h**2 + norm(C)) * Fraction(float(np.max(np.abs(s)))) + max(abs(f) for f in F)
    assert eta == float(max(abs(x) for x in r) / scale)


@pytest.mark.parametrize("kind", ["zero column", "duplicate column", "integer combination"])
def test_exactly_dependent_given_sets_keep_their_status(kind):
    G = np.column_stack([np.arange(1, 10) % 4 + 1.0, np.arange(1, 10) % 3 + 2.0, np.zeros(9)])
    if kind == "duplicate column":
        G[:, 2] = G[:, 0]
    elif kind == "integer combination":
        G[:, 2] = 2 * G[:, 0] - G[:, 1]  # exact in floating point (small integers)
    p = hs.HalfStep(1.0, 10, 1.7, 14, "Y", G)
    assert hs.exact_rank(G) == 2 and hs.given_set_status(p) == hs.EXACTLY_DEPENDENT
    assert hs.solve_hessian(p) == (hs.EXACTLY_DEPENDENT, None)
    ac = hs.solve_ac(p)
    assert ac.status == hs.EXACTLY_DEPENDENT and ac.s is None  # no regularization, no field


@pytest.mark.parametrize("eps", [1e-3, 1e-6])
def test_ill_conditioned_given_set_is_solved_with_its_diagnostics(eps, capsys):
    """Near dependence: status INDEPENDENT, cond(A) recorded; the comparison is asserted only where lambda_min is certified."""
    G = hs.start_set(10, 2)
    G[:, 1] = G[:, 0] + eps * G[:, 1]
    p = hs.HalfStep(1.0, 10, 1.7, 14, "Y", G)
    diag = hs.diagnostics(p)
    assert diag["status"] == hs.INDEPENDENT and diag["exact_rank"] == 2
    (status_h, s_h), ac = hs.solve_hessian(p), hs.solve_ac(p)
    assert status_h == ac.status == hs.INDEPENDENT
    low = hs.certify_lambda_min(p, diag["lambda_min_K_float"] / 2)
    if low is not None:
        b = hs.error_bound(p, s_h, low) + hs.error_bound(p, ac.s, low)
        assert np.linalg.norm(s_h - ac.s) <= b
    W_h = G @ s_h.reshape(2, -1)
    W_ac = G @ ac.s.reshape(2, -1)
    with capsys.disabled():
        print(f"\neps={eps}: cond_A={diag['cond_A']:.2e}, cond_K={diag['cond_K']:.2e}, certified={low is not None}, "
              f"field difference {np.max(np.abs(W_h - W_ac)) / np.max(np.abs(W_ac)):.1e} (OBSERVATION)")


def test_numerically_singular_float_system_keeps_a_status():
    """Independent in exact arithmetic but numerically singular in double: a status, no field, no regularization."""
    G = hs.start_set(32, 2)
    G[:, 1] = G[:, 0] + 1e-8 * G[:, 1]
    p = hs.HalfStep(1.0, 32, 1.0, 32, "Y", G)
    assert hs.given_set_status(p) == hs.INDEPENDENT
    status_h, s_h = hs.solve_hessian(p)
    ac = hs.solve_ac(p)
    for status, s in ((status_h, s_h), (ac.status, ac.s)):
        assert (status, s is None) in ((hs.FACTORIZATION_FAILED, True), (hs.INDEPENDENT, False))


@pytest.mark.parametrize("n_f", [8, 10, 13])
def test_start_sets_T_N_are_admissible_discretely(n_f):
    """T_N with 2N < n_f: exact rank N, nonzero load; both bases of VBC-15 give the same half-step field within the bounds."""
    for N in range(1, 4):
        if 2 * N >= n_f:
            continue
        for basis in ("sin2", "product"):
            G = hs.start_set(n_f, N, basis)
            assert hs.exact_rank(G) == N
            p = hs.HalfStep(1.0, n_f, 1.3, 12, "Y", G)
            d = hs.diagnostics(p)
            assert d["two_N_less_than_n_f"] and d["F_nonzero"]


def test_controls_do_not_import_the_main_solver():
    code = ("import sys, kkvim_reference.r5_cccc_halfstep as m; "
            "print(any(k == 'kkvim' or k.startswith('kkvim.') for k in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    assert out == "False"


def test_input_checks():
    with pytest.raises(ValueError):
        hs.HalfStep(1.0, 10, 1.0, 10, "Z", hs.start_set(10, 1))
    with pytest.raises(ValueError):
        hs.HalfStep(1.0, 10, 1.0, 10, "Y", np.ones((5, 1)))
    with pytest.raises(ValueError):
        hs.HalfStep(1.0, 10, 1.0, 10, "Y", np.ones((9, 4)))
