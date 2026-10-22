"""Self-checks of the QA research prototype B-C (kkvim_reference.r5_cccc_bc); NS 5, 5.1; D-049.2; PL BC-3...BC-9.

Criteria, no tolerance taken from observed discrepancies:
- exact rational identities (decomposition K^C = K0 + Z G Z^T, Woodbury formula, S > 0, D/q scaling of the exact solution);
- bitwise identities where the order of operations is the same (orientation x <-> y, scaling by powers of two);
- the accepted strict certificates of r5_cccc_halfstep (reviews 57d3194, 4f92c82) in their proven domain: two certified
  approximations of the same exact solution satisfy ||s1 - s2|| <= b1 + b2 (checked exactly), and B-C is certified informative,
  i.e. ||s_BC - s*|| < ||s_BC|| (rel_to_exact exists: L > b, a qualitative property, not a tolerance);
- the standard rounding bound gamma_m for the coefficient path (operation count in ``test_coefficients_stencil_path``);
- identification checks for the DST normalisation, the sign of mu_k and the grid weight of the QR: they separate the correct
  constant from the plausible wrong ones (relative gaps >= 1/(2n), or a factor h^{+-1}) by a margin that is stated in each test; the
  margin is not a tolerance of the method.
Results without such a criterion (cond(S), kappa_c, differences) are OBSERVATION and are not asserted.
"""

import math
import subprocess
import sys
from fractions import Fraction

import numpy as np
import pytest

from kkvim_reference import r5_cccc_bc as bc
from kkvim_reference import r5_cccc_halfstep as hs

ALLOWED = {hs.INDEPENDENT, hs.EXACTLY_DEPENDENT, hs.FACTORIZATION_FAILED}


def problem(lx, nx, ly, ny, direction, N, D=1.0, q=1.0, G=None):
    n_f = nx if direction == "Y" else ny
    return hs.HalfStep(lx, nx, ly, ny, direction, hs.start_set(n_f, N) if G is None else G, D=D, q=q)


# --- exact definitions (independent of both modules) ----------------------------------------------------------------

def _mm(X, Y):
    return [[sum(X[i][k] * Y[k][j] for k in range(len(Y))) for j in range(len(Y[0]))] for i in range(len(X))]


def _L(n, h):
    return [[Fraction(-2 if i == j else 1 if abs(i - j) == 1 else 0) / h**2 for j in range(n - 1)] for i in range(n - 1)]


def _exact_parts(p):
    """Exact A, B, C, F (NS 4.2, D4 = L L + E_2 as matrices), L_s, L_s^2, E_2 of the solved grid, h_s."""
    hf, hsol = Fraction(p.l_f) / p.n_f, Fraction(p.l_s) / p.n_s
    G = [[Fraction(float(v)) for v in row] for row in p.given]
    GT = [list(c) for c in zip(*G)]
    Lf = _L(p.n_f, hf)
    D4f = _mm(Lf, Lf)
    D4f[0][0] += 2 / hf**4
    D4f[-1][-1] += 2 / hf**4
    A = [[hf * v for v in row] for row in _mm(GT, G)]
    B = [[-hf * v for v in row] for row in _mm(GT, _mm(Lf, G))]
    C = [[hf * v for v in row] for row in _mm(GT, _mm(D4f, G))]
    F = [Fraction(p.q) * hf * sum(G[r][i] for r in range(len(G))) for i in range(p.N)]
    Ls = _L(p.n_s, hsol)
    m = p.n_s - 1
    E2 = [[(2 / hsol**4) * (1 if i == j and i in (0, m - 1) else 0) for j in range(m)] for i in range(m)]
    return A, B, C, F, Ls, _mm(Ls, Ls), E2, hsol


def _kron_terms(p, A, B, C, Lop, L2):
    m, N, D = p.n_s - 1, p.N, Fraction(p.D)
    K = [[Fraction(0)] * (N * m) for _ in range(N * m)]
    for i in range(N):
        for k in range(N):
            for a in range(m):
                for b in range(m):
                    K[i * m + a][k * m + b] = D * (A[i][k] * L2[a][b] - 2 * B[i][k] * Lop[a][b] + C[i][k] * (a == b))
    return K


def _solve(K, f):
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


def _spd(M):
    A = [row[:] for row in M]
    for k in range(len(A)):
        if A[k][k] <= 0:
            return False
        for i in range(k + 1, len(A)):
            f = A[i][k] / A[k][k]
            if f:
                for j in range(k, len(A)):
                    A[i][j] -= f * A[k][j]
    return True


SMALL = [(1.0, 7, 1.3, 6, "Y", 1), (1.0, 6, 0.8, 5, "X", 2), (0.9, 7, 1.1, 5, "Y", 3), (1.0, 3, 1.0, 3, "Y", 1),
         (1.0, 5, 2.0, 4, "X", 2)]


# --- independence ---------------------------------------------------------------------------------------------------

def test_prototype_does_not_import_the_main_solver():
    code = ("import sys, kkvim_reference.r5_cccc_bc as m; "
            "print(any(k == 'kkvim' or k.startswith('kkvim.') for k in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    assert out == "False"


# --- 1. coefficients --------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("case", SMALL + [(1.0, 32, 1.7, 20, "Y", 3), (1.3, 40, 1.0, 24, "X", 2)])
def test_coefficients_stencil_path(case):
    """A, B, C, F of the stencil path against the exact values: |fl - exact| <= gamma_m |terms|. Count (factors (1 + delta)^(+-1)
    against the exact h): h_f (1); L stencil: 2 additions, /h_f**2 (error of h_f twice, pow 2u, division) -> 7; D4 stencil: 6 g or
    7 g (1), 4 additions, /h_f**4 (4 + 2 + 1) -> 12; G^T(.) of length n_f - 1 (+ 1 + n_f - 2) and the factor h_f (+ 2): C n_f + 13,
    B n_f + 8, A n_f + 1; F = (q h_f) sum: n_f + 1. Used: gamma(2 n_f + 32) >= gamma(n_f + 13)."""
    p = problem(*case)
    A, B, C, F = bc.bc_coefficients(p)
    Ae, Be, Ce, Fe = hs.exact_coefficients(p)
    hf = p.l_f / p.n_f
    G = np.abs(p.given)
    absL = (np.pad(G, ((1, 0), (0, 0)))[:-1] + 2 * G + np.pad(G, ((0, 1), (0, 0)))[1:]) / hf**2
    P = np.pad(G, ((2, 2), (0, 0)))
    w = np.full((G.shape[0], 1), 6.0)
    w[0] = w[-1] = 7.0
    absD4 = (P[:-4] + 4 * P[1:-3] + w * P[2:-2] + 4 * P[3:-1] + P[4:]) / hf**4
    g = hs.gamma(2 * p.n_f + 32)
    for X, Xe, T in ((A, Ae, hf * G.T @ G), (B, Be, hf * G.T @ absL), (C, Ce, hf * G.T @ absD4)):
        for i in range(p.N):
            for k in range(p.N):
                assert abs(Fraction(X[i, k]) - Xe[i][k]) <= Fraction(g * T[i, k])
    for i in range(p.N):
        assert abs(Fraction(F[i]) - Fe[i]) <= Fraction(g * p.q * hf * G[:, i].sum())


# --- 2. algebra of B-C (exact) ----------------------------------------------------------------------------------------

@pytest.mark.parametrize("case", SMALL)
def test_capacitance_decomposition_and_woodbury_are_exact(case):
    """BC-3 and NS 5: K^C = K0 + Z G Z^T exactly (Z by ``boundary_rows``, G = D A (x) (2/h^4) I_2), S = G^-1 + Z^T K0^-1 Z is
    symmetric positive definite, and x0 - Y0 S^-1 Z^T x0 is the exact solution of K^C x = F (x) 1 (rational arithmetic)."""
    p = problem(*case, D=1.7, q=0.3)
    A, B, C, F, Ls, L2, E2, h = _exact_parts(p)
    m, N, D = p.n_s - 1, p.N, Fraction(p.D)
    D4 = [[L2[a][b] + E2[a][b] for b in range(m)] for a in range(m)]
    KC = _kron_terms(p, A, B, C, Ls, D4)
    K0 = _kron_terms(p, A, B, C, Ls, L2)
    rows = bc.boundary_rows(N, m)
    n = N * m
    Gm = [[D * A[i][k] * (2 / h**4) * (s1 == s2) for k in range(N) for s2 in (0, 1)] for i in range(N) for s1 in (0, 1)]
    ZGZ = [[Fraction(0)] * n for _ in range(n)]
    for a, ra in enumerate(rows):
        for b, rb in enumerate(rows):
            ZGZ[ra][rb] += Gm[a][b]
    assert KC == [[K0[i][j] + ZGZ[i][j] for j in range(n)] for i in range(n)]
    b = [F[i] for i in range(N) for _ in range(m)]
    x0 = _solve(K0, b)
    Y0 = [_solve(K0, [Fraction(int(r == rows[c])) for r in range(n)]) for c in range(2 * N)]  # columns
    Ainv_rows = [_solve(A, [Fraction(int(i == j)) for i in range(N)]) for j in range(N)]  # columns of A^-1
    Ginv = [[(h**4 / (2 * D)) * Ainv_rows[k][i] * (s1 == s2) for k in range(N) for s2 in (0, 1)] for i in range(N) for s1 in (0, 1)]
    S = [[Ginv[a][c] + Y0[c][rows[a]] for c in range(2 * N)] for a in range(2 * N)]
    assert len(S) == 2 * N and all(S[a][c] == S[c][a] for a in range(2 * N) for c in range(2 * N))
    assert _spd(S)
    cvec = _solve(S, [x0[r] for r in rows])
    x = [x0[r] - sum(Y0[c][r] * cvec[c] for c in range(2 * N)) for r in range(n)]
    assert x == _solve(KC, b)
    # the floating-point S of the prototype is a positive definite matrix of the same size (values: OBSERVATION)
    sol = bc.solve_bc(p)
    assert sol.diagnostics["size_S"] == 2 * N and sol.diagnostics["min_eig_sym_S"] > 0


# --- 3. modal part: DST normalisation and signs -----------------------------------------------------------------------

@pytest.mark.parametrize("n", [3, 4, 8, 17, 64, 257])
def test_dst_normalisation_is_identified(n):
    """Psi = dst(I) equals sqrt(2/n) sin(k pi i/n) and is a symmetric involution. Identification margin: the plausible wrong
    constants sqrt(2/(n+1)), sqrt(1/n), sqrt(2/(n-1)), 1, 2 differ from sqrt(2/n) by a relative >= 1/(2n) >= 2^-10 here; the
    comparison bound 2^-30 lies 2^20 below that gap."""
    m = n - 1
    Psi = bc.dst(np.eye(m), axis=0)
    i = np.arange(1, n)
    sine = np.sin(np.outer(i, i) * math.pi / n)
    explicit = math.sqrt(2 / n) * sine
    assert np.max(np.abs(Psi - explicit)) < 2.0**-30
    for wrong in (math.sqrt(2 / (n + 1)), math.sqrt(1 / n), math.sqrt(2 / (n - 1)) if n > 2 else 0.0, 1.0, 2.0):
        assert np.max(np.abs(wrong * sine - explicit)) > 2.0**-20
    x = np.random.default_rng(n).standard_normal(m)
    assert np.max(np.abs(bc.dst(bc.dst(x)) - x)) < 2.0**-30 * np.max(np.abs(x))
    assert np.max(np.abs(Psi - Psi.T)) < 2.0**-30


@pytest.mark.parametrize("n, length", [(3, 1.0), (8, 0.7), (33, 2.0), (128, 1.3)])
def test_modal_eigenvalues_have_the_right_sign(n, length):
    """L psi_k = -mu_k psi_k with mu_k > 0 increasing (R4 3.2). The wrong sign would leave a residual 2 mu_k ||psi_k||_inf >=
    2 mu_1 ||psi_1||_inf; the bound used is 2^-30 mu_max, far below (identification)."""
    h = length / n
    mu = bc.modal_eigenvalues(n, h)
    assert np.all(mu > 0) and np.all(np.diff(mu) > 0)
    Psi = bc.dst(np.eye(n - 1), axis=0)
    L = hs._L(n, h)
    assert np.max(np.abs(L @ Psi + Psi * mu[None, :])) < 2.0**-30 * mu[-1]
    assert np.max(np.abs(L @ Psi - Psi * mu[None, :])) >= mu[0] * np.max(np.abs(Psi[:, 0]))  # half of the gap 2 mu_1 |psi_1|


@pytest.mark.parametrize("case", SMALL)
def test_modal_blocks_are_positive_and_carry_plus_two_mu_B(case):
    """M_k = mu_k^2 A + 2 mu_k B + C (the sign from -2B(-mu_k)): K0 (a (x) psi_k) = D (M_k a) (x) psi_k. The wrong sign changes the
    image by 4 mu_k D |B a|; the bound used is 2^-30 of the image scale (identification). All M_k > 0 (Cholesky in the solver)."""
    p = problem(*case)
    A, B, C, _ = bc.bc_coefficients(p)
    m, N = p.n_s - 1, p.N
    h = p.l_s / p.n_s
    mu = bc.modal_eigenvalues(p.n_s, h)
    Ls = hs._L(p.n_s, h)
    K0 = p.D * (np.kron(A, Ls @ Ls) - 2 * np.kron(B, Ls) + np.kron(C, np.eye(m)))
    Psi = bc.dst(np.eye(m), axis=0)
    a = np.arange(1, N + 1, dtype=float)
    for k in range(m):
        Mk = mu[k] ** 2 * A + 2 * mu[k] * B + C
        img = K0 @ np.kron(a, Psi[:, k])
        scale = np.max(np.abs(img))
        assert np.max(np.abs(img - p.D * np.kron(Mk @ a, Psi[:, k]))) < 2.0**-30 * scale
        assert np.linalg.eigvalsh(Mk)[0] > 0
    assert bc.solve_bc(p).diagnostics["min_eig_Mk"] > 0


# --- 4. QR with the grid weight ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("n_f, length, N", [(8, 1.0, 2), (16, 0.7, 3), (64, 1.3, 3), (33, 2.0, 2)])
def test_qr_normalisation_uses_the_grid_step(n_f, length, N):
    """U~ = Q/sqrt(h_f): the exact defect max|h_f U~^T U~ - I| is < 1/2, whereas U~ = Q or Q/h_f gives a defect >= |h_f^{+-1} - 1|
    >= 1/2 for h_f <= 1/2 (identification of the weight; the value of the defect is recorded as OBSERVATION). cond(A) before QR is
    cond(R)^2 by construction."""
    p = hs.HalfStep(length, n_f, 1.0, 10, "Y", hs.start_set(n_f, N))
    h = length / n_f
    assert h <= 0.5
    rec = bc.run_halfstep(p, qr=True)
    assert rec.log["qr"] and rec.log["orthonormality_defect_up"] < 0.5
    Q = math.sqrt(h) * rec.solved.given
    for wrong in (Q, Q / h):
        bad = hs.HalfStep(length, n_f, 1.0, 10, "Y", wrong)
        assert bc.orthonormality_defect(bad) >= 0.5
    sv = np.linalg.svd(rec.R, compute_uv=False)
    assert rec.log["cond_A_before_QR"] == (sv[0] / sv[-1]) ** 2
    assert "cond_A_after_QR" in rec.log and "cond_S" in rec.log


def test_n1_has_no_qr():
    rec = bc.run_halfstep(problem(1.0, 10, 1.7, 14, "Y", 1), qr=True)
    assert rec.log["qr"] is False and rec.R is None and rec.log["cond_A_before_QR"] == 1.0 and rec.log["size_S"] == 2


# --- 5. comparisons with R-AC and R-H under the certificates ----------------------------------------------------------

RECT = [(1.0, 12, r, n_s, d, N) for r, n_s in ((0.5, 8), (2.0, 24)) for d in hs.DIRECTIONS for N in (1, 2, 3)]


def _assert_certified(c):
    assert c["status"] == hs.INDEPENDENT and c["lambda_low"] != bc.UNRESOLVED
    assert c["rel_to_exact_bc"] != bc.UNRESOLVED  # certified informative: ||s_BC - s*|| < ||s_BC||
    assert c["bc_ac_certified_consistent"] is True
    if "bc_rh_certified_consistent" in c:
        assert c["bc_rh_certified_consistent"] is True


@pytest.mark.parametrize("lx, nx, ly, ny, direction, N", RECT)
@pytest.mark.parametrize("qr", [False, True])
def test_rectangles_r_and_1_over_r_both_directions(lx, nx, ly, ny, direction, N, qr):
    """b/a = 1/2 and 2, X and Y, N = 1-3, with and without QR: B-C certified, consistent with R-AC and R-H (n <= 32)."""
    n_f = nx if direction == "Y" else ny
    if direction == "X":
        lx, nx, ly, ny = ly, ny, lx, nx  # the same fixed/solved grids in the other orientation
        n_f = ny
    p = hs.HalfStep(lx, nx, ly, ny, direction, hs.start_set(n_f, N))
    rec = bc.run_halfstep(p, qr=qr)
    c = bc.compare(rec)
    _assert_certified(c)
    assert c["eta_bc"] == hs.eta_qa(rec.solved, rec.bc.s)  # eta of the original operator with D4 (R-HP)


@pytest.mark.parametrize("N", [1, 2, 3])
@pytest.mark.parametrize("qr", [False, True])
def test_orientation_swap_is_bitwise_identical(N, qr):
    """BC-9: the X-half-step on (a, b) and the Y-half-step on (b, a) are the same computation (same order of operations)."""
    a, b, n_a, n_b = 1.0, 1.7, 10, 14
    y = bc.run_halfstep(hs.HalfStep(b, n_b, a, n_a, "Y", hs.start_set(n_b, N)), qr=qr)
    x = bc.run_halfstep(hs.HalfStep(a, n_a, b, n_b, "X", hs.start_set(n_b, N)), qr=qr)
    assert np.array_equal(x.bc.s, y.bc.s)
    assert {k: v for k, v in x.log.items() if k != "direction"} == {k: v for k, v in y.log.items() if k != "direction"}


@pytest.mark.parametrize("D, q", [(4.0, 0.5), (0.25, 8.0)])
def test_scaling_by_powers_of_two_is_exact(D, q):
    """s(D, q) = (q/D) s(1, 1) bitwise: every operation scales by a power of two (no under- or overflow)."""
    for N in (1, 2, 3):
        base = bc.solve_bc(problem(1.0, 12, 1.3, 9, "Y", N)).s
        scaled = bc.solve_bc(problem(1.0, 12, 1.3, 9, "Y", N, D=D, q=q)).s
        assert np.array_equal(scaled, (q / D) * base)


@pytest.mark.parametrize("D, q", [(2.7, 0.3), (0.013, 41.0)])
def test_scaling_by_general_D_q_within_certificates(D, q):
    """The exact solution scales exactly: s*(D, q) = (q/D) s*(1, 1) (K = D K_1, F = q F_1 in exact arithmetic). Hence
    ||s(D, q) - (q/D) s(1, 1)|| <= b(D, q) + (q/D) b(1, 1), checked exactly."""
    for N in (1, 2, 3):
        p1, pd = problem(1.0, 12, 1.3, 9, "X", N), problem(1.0, 12, 1.3, 9, "X", N, D=D, q=q)
        s1, sd = bc.solve_bc(p1).s, bc.solve_bc(pd).s
        c1, cd = bc.compare(bc.run_halfstep(p1, qr=False)), bc.compare(bc.run_halfstep(pd, qr=False))
        ratio = Fraction(q) / Fraction(D)
        d2 = sum(((Fraction(float(a)) - ratio * Fraction(float(b))) ** 2 for a, b in zip(sd, s1)), Fraction(0))
        assert d2 <= (Fraction(cd["bound_abs_bc"]) + ratio * Fraction(c1["bound_abs_bc"])) ** 2


# --- 6. dependent and ill-conditioned sets ----------------------------------------------------------------------------

def test_exactly_dependent_set_stops_before_qr_and_solve():
    G = hs.start_set(12, 1)
    for given in (np.column_stack([G, 2 * G]), np.column_stack([G, hs.start_set(12, 2)[:, 1], 4 * G])):
        rec = bc.run_halfstep(hs.HalfStep(1.0, 12, 1.0, 10, "Y", given), qr=True)
        assert rec.status == hs.EXACTLY_DEPENDENT and rec.R is None and rec.bc is None
        assert bc.compare(rec) == {"status": hs.EXACTLY_DEPENDENT}
        assert bc.solve_bc(hs.HalfStep(1.0, 12, 1.0, 10, "Y", given)).status == hs.EXACTLY_DEPENDENT


@pytest.mark.parametrize("eps", [1e-2, 1e-4, 1e-6, 1e-8])
@pytest.mark.parametrize("qr", [False, True])
def test_ill_conditioned_sets_keep_a_status_and_certificates(eps, qr):
    """G_2 = G_1 + eps G_2 (PL BC-6 / BC-7 analogue): a status from the closed list, the three logged values (NS 5.1), and where a
    certificate exists the comparison with R-AC is consistent. No thresholds: cond values are OBSERVATION."""
    G = hs.start_set(24, 2)
    G[:, 1] = G[:, 0] + eps * G[:, 1]
    rec = bc.run_halfstep(hs.HalfStep(1.0, 24, 1.0, 20, "Y", G), qr=qr)
    assert rec.status in ALLOWED
    assert "cond_A_before_QR" in rec.log
    if qr:
        assert "cond_A_after_QR" in rec.log
    if rec.bc is not None and rec.bc.s is not None:
        assert "cond_S" in rec.log
        c = bc.compare(rec)
        if c["lambda_low"] != bc.UNRESOLVED and c.get("bound_abs_ac", bc.UNRESOLVED) != bc.UNRESOLVED:
            assert c["bc_ac_certified_consistent"] is True
        else:
            assert c.get("bc_ac_certified_consistent", bc.UNRESOLVED) == bc.UNRESOLVED
