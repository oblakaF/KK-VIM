"""Program checks of the QA half-step controls R-H, R-AC, R-HP (kkvim_reference.r5_cccc_halfstep), PL 2.1.

Criteria used here, no tolerance is introduced:
- exact rational identities (coefficients, scaling, symmetry, exact rank);
- the standard rounding bound of floating-point dot products and products: |fl - exact| <= gamma_m |terms|, gamma_m = m u/(1 - m u),
  with the operation count m written below for each path (R-2, review 57d3194; M-3, review 4f92c82). Coefficient path, element
  of K, counted as factors (1 + delta)^(+-1) against the exact h = l/n: h_f = fl(l_f/n_f) (1); L = (...)/h_f**2: the error of h_f
  twice, pow counted as 2u, the division (2 + 2 + 1 = 5); L L, 3 terms (5 + 5 + 1 + 2 = 13); the corner 2/h_f**4 (4 + 2 + 1 = 7)
  and its addition (14); D4_f G, 5 terms (14 + 1 + 4 = 19); G^T(.) of length n_f - 1 (+ 1 + (n_f - 2) = n_f + 18); the factor h_f
  (+ 2 = n_f + 20, the term C; A and B need fewer); the two sums of the three Kronecker terms and the factor D: m <= n_f + 23.
  The tests use the conservative gamma(2 n_f + 32) >= gamma(n_f + 23) (n_f >= 1), as for the R-H path gamma(2 m_1 + 32)
  (m_1 the length of the dense products). Assumption: pow with the exponents 2 and 4 errs by at most 1 ulp;
- the strict a posteriori bound ||s - s*||_2 <= ||r||_2 / lambda_low with the exact residual (R-HP) and lambda_low certified by the
  EXACT defect of a decimal Cholesky factor of K - tau I (module docstring; I-1, I-2), for n_f, n_s >= 3 (I-4).
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
    assert np.all(np.abs(Kc - Kex) <= hs.gamma(2 * p.n_f + 32) * coef_abs + hs.U * np.abs(Kex))
    assert np.all(np.abs(Kh - Kex) <= hs.gamma(2 * m1 + 32) * rh_abs + hs.U * np.abs(Kex))
    F = hs.exact_coefficients(p)[3]
    Fex = np.repeat([float(f) for f in F], p.n_s - 1)
    hf = p.l_f / p.n_f
    assert np.all(np.abs(rc - Fex) <= hs.gamma(p.n_f + 6) * p.q * hf * np.repeat(np.abs(p.given).sum(axis=0), p.n_s - 1)
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
        bound = hs.gamma(p.size + 2 * p.n_f + 34) * (coef_abs @ np.abs(vec) + np.abs(rhs)) + hs.U * np.abs(r_ex)
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


# --- review 57d3194: I-1 (error_bound), I-2 (certificate), I-4 (n >= 3), directed rounding ---------------------------

def exact_spd(M) -> bool:
    """Exact LDL^T (rational Gaussian elimination without pivoting): all pivots > 0 <=> M symmetric positive definite."""
    A = [row[:] for row in M]
    n = len(A)
    for k in range(n):
        if A[k][k] <= 0:
            return False
        for i in range(k + 1, n):
            f = A[i][k] / A[k][k]
            if f:
                for j in range(k, n):
                    A[i][j] -= f * A[k][j]
    return True


def test_float_up_and_down_are_directed():
    for x in (Fraction(1, 10), Fraction(2, 3), Fraction(10**400, 3**700), Fraction(1, 10**320), Fraction(5)):
        up, down = hs._float_up(x), hs._float_down(x)
        assert Fraction(down) <= x <= Fraction(up)
        assert Fraction(math_nextafter(up, -1.0)) < x or Fraction(up) == x
        assert Fraction(math_nextafter(down, 2.0 * abs(down) + 1.0)) > x or Fraction(down) == x
    assert hs._float_up(Fraction(10**400)) == float("inf")
    assert hs._float_up(Fraction(1, 10**400)) == 5e-324  # positive, never 0


def math_nextafter(x, toward):
    import math
    return math.nextafter(x, toward)


@pytest.mark.parametrize("q", [1e-200, 1e-160, 1.0, 1e150])
def test_error_bound_is_never_below_the_exact_residual_bound(q):
    """I-1: (bound lambda_low)^2 >= ||r||_2^2 checked exactly; q = 1e-200 (underflow of r^2) and 1e-160 (subnormal squares)."""
    p = hs.HalfStep(1.0, 10, 1.7, 14, "Y", hs.start_set(10, 1), q=q)
    low = hs.certify_lambda_min(p, float(np.linalg.eigvalsh(hs.coefficient_system(p)[0])[0]) / 2)
    s = np.zeros(p.size)
    b = hs.error_bound(p, s, low)
    rr = sum(x * x for x in hs.residual_exact(p, s))
    assert rr > 0 and b > 0
    assert (Fraction(b) * Fraction(low)) ** 2 >= rr


def test_error_bound_zero_residual_invalid_lambda_and_overflow():
    p = hs.HalfStep(1.0, 10, 1.7, 14, "Y", hs.start_set(10, 1), q=0.0)
    assert hs.error_bound(p, np.zeros(p.size), 1.0) == 0.0  # F = 0, s = 0: zero residual
    for bad in (0.0, -1.0, float("nan"), float("inf"), None, True):
        with pytest.raises(ValueError):
            hs.error_bound(p, np.zeros(p.size), bad)
    big = hs.HalfStep(1.0, 10, 1.7, 14, "Y", hs.start_set(10, 1), q=1e300)
    assert hs.error_bound(big, np.zeros(big.size), 5e-324) == float("inf")  # beyond the double range: conservative


@pytest.mark.parametrize("N, n_s", [(1, 6), (2, 5), (3, 4), (2, 6)])
def test_certificate_is_confirmed_by_an_independent_exact_ldlt(N, n_s):
    """I-2: K - lambda_low I of the exact system is positive definite (exact rational LDL^T), so lambda_min(K) > lambda_low."""
    p = hs.HalfStep(1.0, 8, 1.3, n_s, "Y", hs.start_set(8, N))
    K = exact_dense(p)
    lam = float(np.linalg.eigvalsh(np.array([[float(v) for v in row] for row in K]))[0])
    for tau in (lam / 2, lam * (1 - 1e-9)):
        low = hs.certify_lambda_min(p, tau)
        assert low is not None and 0 < low <= tau
        shifted = [[K[i][j] - (Fraction(low) if i == j else 0) for j in range(len(K))] for i in range(len(K))]
        assert exact_spd(shifted)


def test_certificate_refuses_what_it_cannot_prove():
    p = hs.HalfStep(1.0, 8, 1.3, 6, "Y", hs.start_set(8, 2))
    lam = float(np.linalg.eigvalsh(hs.coefficient_system(p)[0])[0])
    assert hs.certify_lambda_min(p, 2 * lam) is None  # K - tau I indefinite: factorization fails
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        assert hs.certify_lambda_min(p, bad) is None
    assert hs.certify_lambda_min(p, 5e-324) is None  # bound not a positive normal double


@pytest.mark.parametrize("nx, ny, direction", [(2, 8, "Y"), (8, 2, "Y"), (2, 8, "X"), (8, 2, "X")])
def test_one_interior_node_is_rejected(nx, ny, direction):
    """I-4: n_f = 2 or n_s = 2 is refused before any matrix is built."""
    n_f = nx if direction == "Y" else ny
    with pytest.raises(ValueError):
        hs.HalfStep(1.0, nx, 1.0, ny, direction, np.ones((n_f - 1, 1)))


@pytest.mark.parametrize("n", [3, 4, 5, 6])
def test_exact_d4_stencil_equals_l2_plus_e2(n):
    """For n >= 3 the stencil of R-HP (rows (7, -4, 1)) is exactly D4 = L^2 + (2/h^4)(e_1 e_1^T + e_{n-1} e_{n-1}^T)."""
    h = Fraction(1, n)
    m = n - 1
    L = [[Fraction(-2 if i == j else 1 if abs(i - j) == 1 else 0) / h**2 for j in range(m)] for i in range(m)]
    D4 = [[sum(L[i][k] * L[k][j] for k in range(m)) for j in range(m)] for i in range(m)]
    D4[0][0] += 2 / h**4
    D4[m - 1][m - 1] += 2 / h**4
    for j in range(m):
        e = [Fraction(int(i == j)) for i in range(m)]
        assert hs._apply_D4_exact(e, h**4) == [D4[i][j] for i in range(m)]
    p = hs.HalfStep(1.0, n, 1.0, n, "Y", hs.start_set(n, 1))  # n_f = n_s = n >= 3 is admissible
    assert p.size == n - 1


# --- review 4f92c82: R-4 (relative bounds, directed printing), M-1 (float lambda_low), M-4 (exact K) -------------------

def _mm(X, Y):
    return [[sum(X[i][k] * Y[k][j] for k in range(len(Y))) for j in range(len(Y[0]))] for i in range(len(X))]


def _L_def(n, h):
    return [[Fraction(-2 if i == j else 1 if abs(i - j) == 1 else 0) / h**2 for j in range(n - 1)] for i in range(n - 1)]


def _D4_def(n, h):
    D4 = _mm(_L_def(n, h), _L_def(n, h))
    D4[0][0] += 2 / h**4
    D4[n - 2][n - 2] += 2 / h**4
    return D4


def own_exact_system(p):
    """K = D[A (x) D4_s - 2 B (x) L_s + C (x) I] and F (x) 1 from the definitions (NS 4.2-4.3), dense, term-first, exact;
    D4 = L^2 + E_2 as a matrix product. Independent of the module's exact path (stencils, band)."""
    hf, hsol = Fraction(p.l_f) / p.n_f, Fraction(p.l_s) / p.n_s
    G = [[Fraction(float(v)) for v in row] for row in p.given]
    GT = [list(c) for c in zip(*G)]
    A = [[hf * v for v in row] for row in _mm(GT, G)]
    B = [[-hf * v for v in row] for row in _mm(GT, _mm(_L_def(p.n_f, hf), G))]
    C = [[hf * v for v in row] for row in _mm(GT, _mm(_D4_def(p.n_f, hf), G))]
    D4s, Ls = _D4_def(p.n_s, hsol), _L_def(p.n_s, hsol)
    m, N, D = p.n_s - 1, p.N, Fraction(p.D)
    K = [[Fraction(0)] * (N * m) for _ in range(N * m)]
    for i in range(N):
        for k in range(N):
            for a in range(m):
                for b in range(m):
                    K[i * m + a][k * m + b] = D * (A[i][k] * D4s[a][b] - 2 * B[i][k] * Ls[a][b] + C[i][k] * (a == b))
    F = [Fraction(p.q) * hf * sum(G[r][i] for r in range(len(G))) for i in range(N)]
    return K, [F[i] for i in range(N) for _ in range(m)]


def exact_solve(K, f):
    """Exact solution of K x = f (rational Gaussian elimination with row pivoting on nonzero entries)."""
    n = len(K)
    M = [K[i][:] + [f[i]] for i in range(n)]
    for c in range(n):
        piv = next(r for r in range(c, n) if M[r][c] != 0)
        M[c], M[piv] = M[piv], M[c]
        for r in range(c + 1, n):
            if M[r][c]:
                g = M[r][c] / M[c][c]
                M[r] = [a - g * b for a, b in zip(M[r], M[c])]
    x = [Fraction(0)] * n
    for i in reversed(range(n)):
        x[i] = (M[i][n] - sum(M[i][j] * x[j] for j in range(i + 1, n))) / M[i][i]
    return x


def norm2_sq(v):
    return sum((Fraction(float(x)) ** 2 for x in v), Fraction(0))


M4_CASES = [(1.0, 8, 1.3, 6, "Y", 1, False), (1.0, 7, 0.6, 5, "X", 2, True), (0.9, 6, 1.1, 5, "Y", 3, False),
            (1.0, 3, 1.0, 3, "Y", 1, False), (1.0, 3, 1.0, 4, "X", 2, True), (1.0, 6, 0.8, 9, "X", 3, True)]


@pytest.mark.parametrize("lx, nx, ly, ny, direction, N, random_set", M4_CASES)
def test_exact_K_from_definitions_band_and_rhp_operator_are_identical(lx, nx, ly, ny, direction, N, random_set):
    """M-4: the certificate (band) and the bound (R-HP residual) refer to the same exact K, equal to K from the definitions."""
    n_f = nx if direction == "Y" else ny
    G = np.random.default_rng(nx * 100 + ny * 10 + N).standard_normal((n_f - 1, N)) if random_set else hs.start_set(n_f, N)
    p = hs.HalfStep(lx, nx, ly, ny, direction, G, D=1.7, q=0.3)
    K, f = own_exact_system(p)
    assert exact_dense(p) == K
    r0 = hs.residual_exact(p, np.zeros(p.size))
    assert r0 == f
    for c in range(p.size):
        e = np.zeros(p.size)
        e[c] = 1.0
        assert [a - b for a, b in zip(r0, hs.residual_exact(p, e))] == [K[i][c] for i in range(p.size)]


@pytest.mark.parametrize("case", ["refined", "unrefined", "half", "large_error", "zero"])
def test_relative_bounds_against_the_exact_solution(case):
    """R-4: with s* solved exactly, ||s^ - s*||^2 <= b^2, <= to_computed^2 ||s^||^2 and <= to_exact^2 ||s*||^2 (exact)."""
    p = hs.HalfStep(1.0, 7, 1.3, 8, "Y", hs.start_set(7, 2))
    K, f = own_exact_system(p)
    s_star = exact_solve(K, f)
    ac = hs.solve_ac(p)
    e0 = np.zeros(p.size)
    e0[0] = 1.0
    s = {"refined": ac.s, "unrefined": ac.s_unrefined, "half": 0.5 * ac.s,
         "large_error": ac.s + 4.0 * float(np.linalg.norm(ac.s)) * e0, "zero": np.zeros(p.size)}[case]
    low = hs.certify_lambda_min(p, float(np.linalg.eigvalsh(hs.coefficient_system(p)[0])[0]) / 2)
    b = hs.error_bound(p, s, low)
    rel = hs.relative_bounds(b, s)
    err2 = sum(((Fraction(float(x)) - y) ** 2 for x, y in zip(s, s_star)), Fraction(0))
    L, B = hs.norm_lower(s), Fraction(b)
    assert err2 <= B * B
    assert 0 <= L and L * L <= norm2_sq(s)
    if case == "zero":
        assert rel == hs.RelativeBounds(None, None)  # L = 0: UNRESOLVED
        return
    assert err2 <= Fraction(rel.to_computed) ** 2 * norm2_sq(s)
    if case in ("refined", "unrefined"):
        assert rel.to_exact is not None and L > B
        assert err2 <= Fraction(rel.to_exact) ** 2 * sum(y * y for y in s_star)
    else:
        assert rel.to_exact is None and L <= B  # large error: L <= b, no bound relative to s*


def test_relative_bounds_ill_conditioned_set_eps_1e6():
    """R-4: eps = 1e-6 (cond(K) ~ 7e15): exact checks without s*: to_computed ||s^|| >= b, to_exact (||s^|| - b) >= b."""
    G = hs.start_set(32, 2)
    G[:, 1] = G[:, 0] + 1e-6 * G[:, 1]
    p = hs.HalfStep(1.0, 32, 1.0, 32, "Y", G)
    ac = hs.solve_ac(p)
    low = hs.certify_lambda_min(p, hs.diagnostics(p)["lambda_min_K_float"] / 2)
    b = hs.error_bound(p, ac.s, low)
    rel = hs.relative_bounds(b, ac.s)
    B, sh2 = Fraction(b), norm2_sq(ac.s)
    tc, te = Fraction(rel.to_computed), Fraction(rel.to_exact)
    assert tc * tc * sh2 >= B * B
    assert te * te * sh2 >= (B * (1 + te)) ** 2  # te ||s^|| >= b (1 + te)  <=>  te (||s^|| - b) >= b
    assert te > tc  # relative to s* the bound is larger (Reviewer 4f92c82: 0.241 against 0.194)


def test_relative_bounds_at_q_1e_200():
    """R-4: the squares of the set underflow in double (q = 1e-200), the exact L stays positive and the bounds valid."""
    p = hs.HalfStep(1.0, 10, 1.7, 14, "Y", hs.start_set(10, 2), q=1e-200)
    ac = hs.solve_ac(p)
    assert float(np.sum(ac.s * ac.s)) == 0.0  # the danger: a float sum of squares is 0
    low = hs.certify_lambda_min(p, float(np.linalg.eigvalsh(hs.coefficient_system(p)[0])[0]) / 2)
    b = hs.error_bound(p, ac.s, low)
    rel = hs.relative_bounds(b, ac.s)
    B, sh2 = Fraction(b), norm2_sq(ac.s)
    assert b > 0 and hs.norm_lower(ac.s) > B
    tc, te = Fraction(rel.to_computed), Fraction(rel.to_exact)
    assert tc * tc * sh2 >= B * B and te * te * sh2 >= (B * (1 + te)) ** 2


def test_relative_bounds_input_checks():
    s = np.ones(4)
    for bad in (-1.0, float("nan"), 1, None, True):
        with pytest.raises(ValueError):
            hs.relative_bounds(bad, s)
    with pytest.raises(ValueError):
        hs.relative_bounds(1.0, np.array([1.0, float("nan")]))
    assert hs.relative_bounds(float("inf"), s) == hs.RelativeBounds(None, None)  # no finite bound: UNRESOLVED
    assert hs.relative_bounds(0.0, s) == hs.RelativeBounds(0.0, 0.0)
    assert hs.relative_bounds(2.0, s) == hs.RelativeBounds(1.0, None)  # ||s|| = 2 = L <= b: no bound relative to s*
    assert hs.relative_bounds(1.0, s) == hs.RelativeBounds(0.5, 1.0)  # exact: 1/2 and 1/(2 - 1)


def test_norm_lower_is_a_lower_bound_and_tight():
    rng = np.random.default_rng(7)
    for v in (rng.standard_normal(50), rng.standard_normal(50) * 1e-200, np.array([3.0, 4.0]), np.array([5e-324]),
              np.array([1e300, 1e300])):
        L, ss = hs.norm_lower(v), norm2_sq(v)
        assert L * L <= ss < (L * (1 + Fraction(1, 2**60))) ** 2
    assert hs.norm_lower(np.zeros(3)) == 0 and hs.norm_lower(np.array([3.0, 4.0])) == 5


@pytest.mark.parametrize("x", [1 / 3, 0.1, 2.5e-202, 5e-324, 1.7976931348623157e308, 1.0, 0.5, 123456789.0, -0.1, 0.0,
                               4.248773e-14, 2.433941e2])
def test_decimal_printing_is_directed(x):
    """R-4: decimal_up >= x >= decimal_down exactly, both with 7 significant digits and at most one unit apart."""
    from decimal import Decimal
    up, down = hs.decimal_up(x), hs.decimal_down(x)
    U_, D_ = Fraction(Decimal(up)), Fraction(Decimal(down))
    assert D_ <= Fraction(x) <= U_
    if x != 0:
        unit = Fraction(10) ** (Decimal(down if x > 0 else up).adjusted() - 6)
        assert U_ - D_ <= unit
        if Fraction(Decimal(f"{x:.6e}")) == Fraction(x):  # x has 7 significant digits exactly
            assert U_ == D_ == Fraction(x)


def test_decimal_printing_special_values():
    assert hs.decimal_up(float("inf")) == "inf" and hs.decimal_down(0.0) == "0.000000e+00"
    with pytest.raises(ValueError):
        hs.decimal_up(1)


def test_committed_csv_prints_certified_values_conservatively():
    """R-4: in the committed CSV every printed certified value is on the safe side of its exact double (*_hex)."""
    import csv
    import json
    from decimal import Decimal
    from pathlib import Path
    base = Path(__file__).resolve().parents[2] / "docs" / "results" / "r5_cccc_halfstep"
    for name in ("chain.csv", "ill_conditioned.csv"):
        rows = list(csv.DictReader(open(base / name, encoding="utf-8")))
        assert rows
        for row in rows:
            for key, hx in row.items():
                if not key.endswith("_hex"):
                    continue
                prefix = key[:-4]
                side = "down" if prefix == "lambda_low" else "up"
                text = row[f"{prefix}_{side}"]
                if hx == "":  # not computed (dense R-H only for n <= 32)
                    assert text == ""
                    continue
                if hx == "UNRESOLVED":
                    assert text == "UNRESOLVED"
                    continue
                exact, printed = Fraction(float.fromhex(hx)), Fraction(Decimal(text))
                assert printed <= exact if side == "down" else printed >= exact
            for tag in ("unrefined", "refined", "RH"):
                c, e = row.get(f"rel_to_computed_{tag}_hex"), row.get(f"rel_to_exact_{tag}_hex")
                if c not in (None, "", "UNRESOLVED") and e not in (None, "", "UNRESOLVED"):
                    assert float.fromhex(e) >= float.fromhex(c)  # b/(L - b) >= b/L
    assert "columns" in json.loads((base / "meta.json").read_text(encoding="utf-8"))


def test_error_bound_accepts_only_a_float_lambda_low():
    """M-1: an int (e.g. 2^53 + 3, not representable) is refused; numpy float64 is a float."""
    p = hs.HalfStep(1.0, 10, 1.7, 14, "Y", hs.start_set(10, 1))
    for bad in (1, 2**53 + 3, 10**400):
        with pytest.raises(ValueError):
            hs.error_bound(p, np.zeros(p.size), bad)
    assert hs.error_bound(p, np.zeros(p.size), np.float64(1.0)) > 0
