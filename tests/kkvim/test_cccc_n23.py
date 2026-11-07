"""CCCC, N = 2-3 (``kkvim.cccc_n23``): the half-step s = 1 (D-063: B-1...B-6 and B-8 of docs/R5_CCCC_N23_LIMITED_POLICY.proposal.md, 9.1,
B-6 as amended by D-064) and the half-step s = 2 of D-065 (the order C-G of D-065, 7; the certificates themselves in test_cccc_n23_certificate.py).

The exact checks rebuild Pi_h^C, L, D4 = Ł^T W Ł / h and the system from their definitions (Policy 2; MATH_SPEC ВБC-12, ВБC-13) in Fraction,
here, without the code under test. Tolerances: |eta - eta_exact| <= 16u and eta <= 100u are the criteria of D-063.3 (variant A); the relative
1e-13 of the formulas of ВБC-13, item 3 for the double start set is the choice of this test, fixed before the first comparison (MATH_SPEC
records 1.8e-14); it is no criterion of the method. At s = 2 no tolerance is used (D-065): the checks are exact (C-3+ bound of the theorem,
bitwise identities, strict energy inequalities in Fraction) or count calls.
"""

import math
from fractions import Fraction

import numpy as np
import pytest

import kkvim.cccc_n1 as n1
import kkvim.cccc_n23 as cn

U = Fraction(1, 2**53)
SMALL = [(1.0, 0.6, 10, 9), (1.0, 1.3, 8, 7), (1.0, 1.0, 12, 13), (1.0, 1.25, 8, 10)]  # C-1...C-4 of Policy 8


# --- definitions of Policy 2 in Fraction ---------------------------------------------------------------------------------------

def extension(m):
    """E: inner values -> all nodes 0..n, zero on the boundary."""
    return [[Fraction(int(i == j + 1)) for j in range(m)] for i in range(m + 2)]


def lap_full(m, h):
    """Ł: (Łf)_0 = 2 f_1 / h^2, (Łf)_n = 2 f_(n-1) / h^2, inside (f_(i-1) - 2 f_i + f_(i+1)) / h^2 (the reflection w_-1 = w_1)."""
    rows = []
    for i in range(m + 2):
        row = [Fraction(0)] * m
        if i == 0:
            row[0] = 2 / h**2
        elif i == m + 1:
            row[m - 1] = 2 / h**2
        else:
            for j, c in ((i - 2, 1), (i - 1, -2), (i, 1)):
                if 0 <= j < m:
                    row[j] += c / h**2
        rows.append(row)
    return rows


def weights(m, h):
    return [h / 2] + [h] * m + [h / 2]


def matmul(P, Q):
    return [[sum(P[i][k] * Q[k][j] for k in range(len(Q))) for j in range(len(Q[0]))] for i in range(len(P))]


def transpose(P):
    return [list(row) for row in zip(*P, strict=True)]


def d4_matrix(m, h):
    """D4 := Ł^T W Ł / h (Policy 2)."""
    Lf, W = lap_full(m, h), weights(m, h)
    return [[sum(Lf[k][i] * W[k] * Lf[k][j] for k in range(m + 2)) / h for j in range(m)] for i in range(m)]


def l_matrix(m, h):
    return [[(Fraction(-2) if i == j else Fraction(1) if abs(i - j) == 1 else Fraction(0)) / h**2 for j in range(m)] for i in range(m)]


def frac(M):
    return [[Fraction(float(v)) for v in row] for row in np.atleast_2d(M)]


def coefficients(G, length, n, q=1):
    """A, B, C^C, F of the set G (columns) by the definitions of ВБC-13, item 1, with h = length / n exactly."""
    h = Fraction(length) / n
    m = n - 1
    Gq = frac(G)
    Gt = transpose(Gq)
    A = [[h * x for x in row] for row in matmul(Gt, Gq)]
    B = [[-h * x for x in row] for row in matmul(Gt, matmul(l_matrix(m, h), Gq))]
    C = [[h * x for x in row] for row in matmul(Gt, matmul(d4_matrix(m, h), Gq))]
    F = [Fraction(q) * h * sum(column) for column in Gt]
    return A, B, C, F


def energy_by_definition(G, V, a, b, n_x, n_y, D, q):
    """Pi_h^C(w) = D/2 (Δ w)^T (W_x ⊗ W_y)(Δ w) - q h_x h_y Σ w, Δ = Ł_x ⊗ E_y + E_x ⊗ Ł_y, at w = G V^T (Policy 2)."""
    hx, hy = Fraction(a) / n_x, Fraction(b) / n_y
    w = matmul(frac(G), transpose(frac(V)))
    lap = [[x + y for x, y in zip(r1, r2, strict=True)] for r1, r2 in
           zip(matmul(matmul(lap_full(n_x - 1, hx), w), transpose(extension(n_y - 1))),
               matmul(matmul(extension(n_x - 1), w), transpose(lap_full(n_y - 1, hy))), strict=True)]
    Wx, Wy = weights(n_x - 1, hx), weights(n_y - 1, hy)
    bending = sum(Wx[i] * Wy[j] * lap[i][j] ** 2 for i in range(n_x + 1) for j in range(n_y + 1))
    return Fraction(D) / 2 * bending - Fraction(q) * hx * hy * sum(sum(row) for row in w)


def integer_set(m, N, seed):
    return np.random.default_rng(seed).integers(-4, 5, size=(m, N)).astype(float)


# --- B-1: the coefficients and P_hat against the definition of Pi_h^C -----------------------------------------------------------

def test_the_stencils_are_the_operators_of_policy_2():
    for m in (6, 7, 11):
        h = Fraction(1, 8)
        D4 = d4_matrix(m, h)
        expected = matmul(l_matrix(m, h), l_matrix(m, h))
        expected[0][0] += 2 / h**4
        expected[m - 1][m - 1] += 2 / h**4
        assert D4 == expected  # D4 = L^2 + (2/h^4)(e_1 e_1^T + e_m e_m^T)
        V = integer_set(m, 3, m)  # dyadic data: the double stencils are exact
        assert frac(cn.d4_apply(V, 0.125)) == matmul(D4, frac(V))
        assert frac(cn.second_difference(V, 0.125)) == matmul(l_matrix(m, h), frac(V))


@pytest.mark.parametrize("N", [2, 3])
def test_the_coefficients_are_those_of_the_definition_exactly(N):
    G = integer_set(7, N, N)  # l = 1, n = 8: h = 1/8, every operation exact in binary64
    parts = cn.assemble(G, 1.0, 8, 3.0)
    A, B, C, F = coefficients(G, 1, 8, 3)
    assert frac(parts["A"]) == A and frac(parts["B"]) == B and frac(parts["C"]) == C and frac(parts["F"])[0] == F
    h = Fraction(1, 8)
    corner = [[2 / h**3 * (Fraction(G[0, i]) * Fraction(G[0, j]) + Fraction(G[-1, i]) * Fraction(G[-1, j])) for j in range(N)] for i in range(N)]
    LG = matmul(l_matrix(7, h), frac(G))
    C_L = [[h * x for x in row] for row in matmul(transpose(LG), LG)]
    assert C == [[x + y for x, y in zip(r1, r2, strict=True)] for r1, r2 in zip(C_L, corner, strict=True)]  # C^C = C_L + (2/h^3)(u1 u1^T + um um^T)
    assert frac(parts["C_L"]) == C_L and frac(parts["E"]) == corner


@pytest.mark.parametrize("N", [2, 3])
def test_p_hat_is_pi_h_at_w_equal_g_v_transposed(N):
    G, V = integer_set(7, N, 10 + N), integer_set(7, N, 20 + N)  # a = 1, b = 2 on 8 x 8: h_x = 1/8, h_y = 1/4
    D, q = 2.0, 3.0
    exact = energy_by_definition(G, V, 1, 2, 8, 8, D, q)
    Ax, Bx, Cx, Fx = coefficients(G, 1, 8)
    Ay, By, Cy, Fy = coefficients(V, 2, 8)
    formula = Fraction(D) / 2 * sum(Cx[i][j] * Ay[i][j] + 2 * Bx[i][j] * By[i][j] + Ax[i][j] * Cy[i][j] for i in range(N) for j in range(N)) \
        - Fraction(q) * sum(Fx[i] * Fy[i] for i in range(N))
    assert formula == exact  # the derivation of Policy 5, exactly
    assert Fraction(cn.discrete_energy(G, V, 1.0, 2.0, D, q, 8, 8)) == exact


# --- B-2: C-3+ --------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("N", [2, 3])
@pytest.mark.parametrize("a, b, n_x, n_y", SMALL)
def test_c3plus_bound_on_the_solved_set(a, b, n_x, n_y, N):
    step = cn.run_cccc_n23(a, b, 1.0, 1.0, n_x, n_y, N, 1).steps[0]
    assert step.premises_status == cn.PREMISES_ESTABLISHED
    _, _, C, _ = coefficients(step.G, a, n_x)
    k = n_x - 1 + cn.K_EXTRA
    gamma = k * U / (1 - k * U)
    for i in range(N):
        for j in range(N):
            d = Fraction(float(step.C[i, j])) - C[i][j]
            assert d * d <= gamma * gamma * C[i][i] * C[j][j], (i, j)
    assert np.array_equal(step.C, step.C_L + step.E)


# --- B-3: the start ----------------------------------------------------------------------------------------------------------

def test_the_start_set_is_independent_and_the_conditions_hold_on_the_whole_domain():
    for N in cn.N_VALUES:
        for n in range(cn.N_MIN, cn.N_MAX + 1):
            assert all(cn.start_conditions(N, n, n).values())
            assert cn.exact_rank(cn.start_set(n, N)) == N, (n, N)


def test_the_start_coefficients_at_n_x_10_are_those_of_vbc_13():
    a, n_x, q = 1.0, 10, 1.0
    for N in cn.N_VALUES:
        parts = cn.assemble(cn.start_set(n_x, N), a, n_x, q)
        mu = 4.0 / (a / n_x) ** 2 * np.sin(np.arange(1, N + 1) * math.pi / n_x) ** 2
        np.testing.assert_allclose(parts["A"], a * (np.eye(N) / 8 + np.ones((N, N)) / 4), rtol=1e-13, atol=0)
        np.testing.assert_allclose(np.diag(parts["B"]), a / 8 * mu, rtol=1e-13)
        np.testing.assert_allclose(np.diag(parts["C"]), a / 8 * mu**2, rtol=1e-13)
        off = ~np.eye(N, dtype=bool)
        assert np.max(np.abs(parts["B"][off])) <= 1e-13 * np.max(np.abs(parts["B"]))
        assert np.max(np.abs(parts["C"][off])) <= 1e-13 * np.max(np.abs(parts["C"]))
        np.testing.assert_allclose(parts["F"], q * a / 2, rtol=1e-13)


def test_the_condition_number_of_a_before_qr_is_one_plus_2n():
    for N in cn.N_VALUES:
        step = cn.run_cccc_n23(1.0, 1.0, 1.0, 1.0, 32, 32, N, 1).steps[0]
        assert step.cond_A_before_QR == pytest.approx(1 + 2 * N, rel=1e-12)  # an observation of ВБC-17, not a criterion


def test_the_exact_rank_decides_without_a_threshold():
    rng = np.random.default_rng(1)
    M = rng.standard_normal((9, 3))
    assert cn.exact_rank(M) == 3
    dependent = np.column_stack([M[:, 0], 2.0 * M[:, 0], M[:, 1]])  # exactly dependent in binary64
    assert cn.exact_rank(dependent) == 2
    assert cn.exact_rank(np.zeros((5, 2))) == 0
    assert cn.exact_rank(cn.start_set(6, 4)) == 4  # dependent only exactly (ВБC-13, item 4): the rounded doubles are independent
    other = M[:, 0].copy()
    other[3] = np.nextafter(other[3], np.inf)  # one ulp: independent only by a perturbation far below any tolerance
    tiny = np.column_stack([M[:, 0], other])
    assert cn.exact_rank(tiny) == 2


# --- B-5: the statuses of Policy 5 and the guard of s >= 2 ---------------------------------------------------------------------

def run(N=2, half_steps=1, **changes):
    plate = {"a": 1.0, "b": 0.6, "D": 1.0, "q": 1.0, "n_x": 10, "n_y": 9, **changes}
    return cn.run_cccc_n23(plate["a"], plate["b"], plate["D"], plate["q"], plate["n_x"], plate["n_y"], N, half_steps)


@pytest.mark.parametrize("N", [2, 3])
def test_one_half_step_completes(N):
    result = run(N)
    assert result.status == cn.COMPLETED_FIXED_HALF_STEPS and result.half_steps_completed == 1 and len(result.steps) == 1
    step = result.steps[0]
    assert step.eta <= cn.ETA_MAX_C and step.V.shape == (8, N) and step.G.shape == (9, N) and step.rank_decision == cn.RANK_DECISION
    w = result.field()
    assert w.shape == (11, 10) and not np.any(w[0]) and not np.any(w[-1]) and not np.any(w[:, 0]) and not np.any(w[:, -1])
    assert np.array_equal(w[1:-1, 1:-1], step.G @ step.V.T)


def _uncertified(*args, **kwargs):
    return {"certified": False, "verdict": "RN5_UNRESOLVED", "reason": "injected", "hypotheses": {}, "values": {}, "eps": None, "sigma_low": None}


def _counting(monkeypatch, names=("orthonormalise", "assemble", "solve_bc", "residual_eta", "rank_diagnostic", "start_set")):
    counts = {}
    for name in names:
        real = getattr(cn, name)

        def counted(*args, _real=real, _name=name, **kwargs):
            counts[_name] = counts.get(_name, 0) + 1
            return _real(*args, **kwargs)

        monkeypatch.setattr(cn, name, counted)
    qr_calls = []
    real_qr = np.linalg.qr
    monkeypatch.setattr(np.linalg, "qr", lambda *a, **k: qr_calls.append(1) or real_qr(*a, **k))
    return counts, qr_calls


@pytest.mark.parametrize("why", ["grid outside D-065.5", "RN-5 not certified"])
@pytest.mark.parametrize("N", [2, 3])
@pytest.mark.parametrize("half_steps", range(2, 9))
def test_s2_stops_before_qr_without_assembly_or_solve(monkeypatch, why, N, half_steps):
    changes = {"n_x": 33} if why == "grid outside D-065.5" else {}
    certificates = []
    real_rn5 = cn.certificate.rn5_certificate
    monkeypatch.setattr(cn.certificate, "rn5_certificate",
                        lambda *a: certificates.append(1) or (_uncertified() if why == "RN-5 not certified" else real_rn5(*a)))
    monkeypatch.setattr(cn.certificate, "rt7_conditions", lambda *a: pytest.fail("RT-7 without QR"))
    counts, qr_calls = _counting(monkeypatch)
    one = run(N, 1, **changes)
    once, qr_once = dict(counts), len(qr_calls)
    counts.clear()
    qr_calls.clear()
    result = run(N, half_steps, **changes)
    assert counts == once and len(qr_calls) == qr_once  # nothing of the half-step s = 2 is computed, no QR of V
    assert len(certificates) == (0 if why == "grid outside D-065.5" else 1)
    assert result.status == cn.STOPPED_RANK_GUARD_UNCERTIFIED and result.half_steps_completed == 1 and len(result.steps) == 2
    second = result.steps[1]
    assert second.s == 2 and second.direction == "X" and second.qr_executed is False and second.status == result.status
    assert second.A is None and second.G is None and second.eta is None and second.V is None and second.rt7 is None
    assert np.array_equal(result.field(), one.field())


def test_a_proven_dependence_of_v_stops_s2_before_qr(monkeypatch):
    real = cn.exact_rank
    monkeypatch.setattr(cn, "exact_rank", lambda M: real(M) - 1 if np.shape(M) == (8, 2) else real(M))  # only V of 10 x 9
    monkeypatch.setattr(cn.certificate, "rn5_certificate", lambda *a: pytest.fail("a certificate of a dependent set"))
    result = run(2, 2)
    assert result.status == cn.STOP_BEFORE_QR_S1 and result.half_steps_completed == 1 and result.steps[1].status == cn.STOP_BEFORE_QR_S1


def test_a_dependent_start_set_stops_before_qr(monkeypatch):
    monkeypatch.setattr(cn, "start_set", lambda n_x, N: np.column_stack([n1.start_factor(n_x)] * N))
    monkeypatch.setattr(cn, "orthonormalise", lambda *a: pytest.fail("QR of a dependent set"))
    result = run(3, 1)
    assert result.status == cn.STOP_BEFORE_QR_S1 and result.half_steps_completed == 0 and result.field() is None


def _negated_a(real):
    def assemble(*args):
        parts = real(*args)
        if args[1] == 1.0 and args[2] == 10:  # the given set only
            parts["A"] = -parts["A"]
        return parts
    return assemble


@pytest.mark.parametrize("name, replacement, status", [
    ("orthonormalise", lambda real: lambda T, h: (real(T, h)[0], np.diag([1.0, 0.0])), cn.PROTECTIVE_STOP_FACTORIZATION),
    ("assemble", _negated_a, cn.PROTECTIVE_STOP_FACTORIZATION),
    ("premise_failures", lambda real: lambda *a: ["P-2: injected"], cn.UNRESOLVED_C3PLUS_PREMISE),
    ("solve_bc", lambda real: lambda step, *a: np.full((8, 2), np.nan), cn.ERROR_NONFINITE),
    ("solve_bc", lambda real: lambda step, *a: np.zeros((8, 2)), cn.STOP_ZERO_FIELD),
    ("residual_eta", lambda real: lambda *a: 101 * cn.U, cn.STOPPED_ETA_GUARD),
    ("residual_eta", lambda real: lambda *a: math.nan, cn.ERROR_NONFINITE),
])
@pytest.mark.parametrize("half_steps", [1, 2])
def test_every_protective_row_is_reached_and_never_completes(monkeypatch, name, replacement, status, half_steps):
    monkeypatch.setattr(cn, name, replacement(getattr(cn, name)))
    result = run(2, half_steps)
    assert result.status == status and result.steps[0].status == status and result.half_steps_completed == 0
    assert result.field() is None and len(result.steps) == 1


@pytest.mark.parametrize("when", ["before", "after"])
def test_a_failed_rounding_probe_is_an_unresolved_premise(monkeypatch, when):
    calls = []
    real = n1.rounding_probe_failures

    def probe():
        calls.append(1)
        return ["P-1: injected"] if (len(calls) == 1) == (when == "before") else real()

    monkeypatch.setattr(n1, "rounding_probe_failures", probe)
    result = run(2, 1)
    assert result.status == cn.UNRESOLVED_C3PLUS_PREMISE and result.steps[0].premises_status == cn.PREMISES_UNRESOLVED


def test_the_guard_at_exactly_100u_passes(monkeypatch):
    monkeypatch.setattr(cn, "residual_eta", lambda *a: 100 * cn.U)
    assert run(2, 1).status == cn.COMPLETED_FIXED_HALF_STEPS


@pytest.mark.parametrize("changes, N, half_steps", [
    ({}, 1, 1), ({}, 4, 1), ({}, True, 1), ({}, 2.0, 1), ({}, 2, 0), ({}, 2, 9), ({}, 2, 1.0),
    ({"n_x": 6}, 2, 1), ({"n_y": 1281}, 2, 1), ({"b": 5.5}, 2, 1), ({"b": 0.19}, 3, 1),
])
def test_outside_the_domain_nothing_is_computed(monkeypatch, changes, N, half_steps):
    monkeypatch.setattr(cn, "start_set", lambda *a: pytest.fail("computed"))
    result = run(N, half_steps, **changes)
    assert result.status == cn.NOT_EXECUTED_OUTSIDE_DOMAIN and result.steps == () and result.reason


@pytest.mark.parametrize("changes", [{"q": 0.0}, {"q": math.nan}, {"D": -1.0}, {"a": math.inf}, {"n_x": 10.0}])
def test_invalid_input_is_error_input(monkeypatch, changes):
    monkeypatch.setattr(cn, "start_set", lambda *a: pytest.fail("computed"))
    result = run(2, 1, **changes)
    assert result.status == cn.ERROR_INPUT and result.steps == ()


def test_physical_input_is_checked_like_the_other_commands():
    assert cn.run_cccc_n23_physical(1, 1, 0.0, 2.1e11, 0.3, 1000, 32, 32, 2, 1).status == cn.ERROR_INPUT
    result = cn.run_cccc_n23_physical(1, 1, 0.01, 2.1e11, 0.3, 1000, 32, 32, 2, 1)
    assert result.status == cn.COMPLETED_FIXED_HALF_STEPS and result.D == 2.1e11 * 0.01**3 / (12 * (1 - 0.3**2))


def test_a_negative_load_gives_the_negative_field():
    up, down = run(3, 1, q=1.0), run(3, 1, q=-1.0)
    assert np.array_equal(down.field(), -up.field())


# --- B-6 and B-8 -------------------------------------------------------------------------------------------------------------

SCALED = [(1.0, 0.6, 10, 9), (1.0, 1.0, 32, 32), (1.0, 2.0, 32, 64)]


@pytest.mark.parametrize("N", [2, 3])
@pytest.mark.parametrize("a, b, n_x, n_y", SCALED)
def test_scaling_by_four_keeps_w_bar_bit_for_bit(a, b, n_x, n_y, N):
    """B-6 as amended by D-064 (Q-111): with sqrt(4) = 2 every input of QR scales exactly, and w_bar is equal bit for bit.

    A regression on these cases under the order of operations of Policy 3, step 2; not a theorem and not a cross-platform requirement.
    """
    one = cn.run_cccc_n23(a, b, 1.0, 1.0, n_x, n_y, N, 1)
    scaled = cn.run_cccc_n23(4 * a, 4 * b, 16.0, 64.0, n_x, n_y, N, 1)
    assert np.array_equal(one.dimensionless_field(), scaled.dimensionless_field())


@pytest.mark.parametrize("N", [2, 3])
def test_a_repeated_run_is_equal_bit_for_bit(N):
    first, second = run(N, 2, n_x=47, n_y=53, b=1.0), run(N, 2, n_x=47, n_y=53, b=1.0)
    for one, two in zip(first.steps, second.steps, strict=True):
        for name in ("T", "R", "G", "A", "B", "C", "F", "V"):
            x, y = getattr(one, name), getattr(two, name)
            assert (x is None and y is None) or np.array_equal(x, y), name
        assert (one.eta, one.cond_S, one.P_hat, one.sigma, one.status) == (two.eta, two.cond_S, two.P_hat, two.sigma, two.status)


# --- B-4: the solution satisfies the system of Policy 2 ------------------------------------------------------------------------

@pytest.mark.parametrize("N", [2, 3])
@pytest.mark.parametrize("a, b, n_x, n_y", SMALL)
def test_the_solution_solves_the_dense_system(a, b, n_x, n_y, N):
    step = cn.run_cccc_n23(a, b, 1.0, 1.0, n_x, n_y, N, 1).steps[0]
    A, B, C, F = coefficients(step.G, a, n_x)
    m = n_y - 1
    h = Fraction(b) / n_y
    D4, L = d4_matrix(m, h), l_matrix(m, h)
    K = [[A[i][j] * D4[p][r] - 2 * B[i][j] * L[p][r] + C[i][j] * (p == r) for j in range(N) for r in range(m)] for i in range(N) for p in range(m)]
    v = [Fraction(float(step.V[p, i])) for i in range(N) for p in range(m)]  # vec(V) = [V_1; ...; V_N]
    residual = [F[i] - sum(K[i * m + p][c] * v[c] for c in range(N * m)) for i in range(N) for p in range(m)]
    norm = lambda M: max(sum(abs(x) for x in row) for row in M)  # noqa: E731
    eta = max(abs(x) for x in residual) / ((16 * norm(A) / h**4 + 8 * norm(B) / h**2 + norm(C)) * max(abs(x) for x in v) + max(abs(x) for x in F))
    assert eta <= 100 * U
    assert abs(Fraction(step.eta) - eta) <= 16 * U


# --- D-065: the half-step s = 2 -------------------------------------------------------------------------------------------------

S2_CASES = SMALL[:3]  # C-1...C-3: the cases of the QA certificate RN5_CERTIFIED_LIMITED; no grid up to 32 x 32 is assumed certified


@pytest.mark.parametrize("N", [2, 3])
@pytest.mark.parametrize("a, b, n_x, n_y", S2_CASES)
def test_two_half_steps_complete_on_certified_cases(a, b, n_x, n_y, N):
    one = cn.run_cccc_n23(a, b, 1.0, 1.0, n_x, n_y, N, 1)
    result = cn.run_cccc_n23(a, b, 1.0, 1.0, n_x, n_y, N, 2)
    assert result.status == cn.COMPLETED_FIXED_HALF_STEPS and result.half_steps_completed == 2 and len(result.steps) == 2
    first, second = result.steps
    assert first.status == cn.HALF_STEP_DONE and second.status == cn.COMPLETED_FIXED_HALF_STEPS and second.done
    for name in ("T", "R", "G", "A", "B", "C", "C_L", "E", "F", "V"):  # s = 1 is not changed by what follows it
        assert np.array_equal(getattr(first, name), getattr(one.steps[0], name)), name
    assert (first.eta, first.P_hat, first.sigma) == (one.steps[0].eta, one.steps[0].P_hat, one.steps[0].sigma)
    rn5, rt7 = second.rn5, second.rt7
    assert rn5["certified"] and rn5["verdict"] == "RN5_CERTIFIED" and all(rn5["hypotheses"].values())
    assert rn5["sigma_low"] ** 2 > rn5["eps"] ** 2 and rn5["values"]["margin_sigma_over_eps"] > 1
    assert rt7["passed"] and rt7["R7-1"] and rt7["R7-2"] and rt7["R7-3"] and rt7["ranks"] == {"V": [N, N], "G2": [N, N]}
    assert rt7["t"] == rn5["eps"] / rn5["sigma_low"] and 0 <= rt7["t"] < 1 and rt7["tau"] < (1 - rt7["t"]) ** 2
    G2, R2 = cn.orthonormalise(first.V, b / n_y)  # the normalisation of Policy 3, step 2 over y
    assert np.array_equal(second.G, G2) and np.array_equal(second.R, R2) and second.given is first.V and second.qr_executed is True
    assert np.array_equal(second.companion, first.G @ R2.T) and second.exact_rank_G == N
    assert second.V.shape == (n_x - 1, N) and second.G.shape == (n_y - 1, N) and second.V_exact_rank == N
    w = result.field()
    assert np.array_equal(w[1:-1, 1:-1], second.V @ second.G.T) and not np.any(w[[0, -1], :]) and not np.any(w[:, [0, -1]])
    assert result.factors()[0] is second.V and result.factors()[1] is second.G


@pytest.mark.parametrize("N", [2, 3])
@pytest.mark.parametrize("a, b, n_x, n_y", S2_CASES)
def test_the_second_half_step_is_the_x_half_step_on_g2(a, b, n_x, n_y, N):
    """Roles of x and y exchanged (ВБC-13 p. 1): coefficients of G2 over y, C-3+ with the bound of its theorem, the solve over x."""
    second = cn.run_cccc_n23(a, b, 1.0, 1.0, n_x, n_y, N, 2).steps[1]
    parts = cn.assemble(second.G, b, n_y, 1.0)
    for name in ("A", "B", "C", "C_L", "E", "F"):
        assert np.array_equal(getattr(second, name), parts[name]), name
    assert second.premises_status == cn.PREMISES_ESTABLISHED and np.array_equal(second.C, second.C_L + second.E)
    A, B, C, F = coefficients(second.G, b, n_y)
    k = n_y - 1 + cn.K_EXTRA
    gamma = k * U / (1 - k * U)
    for i in range(N):
        for j in range(N):
            d = Fraction(float(second.C[i, j])) - C[i][j]
            assert d * d <= gamma * gamma * C[i][i] * C[j][j], (i, j)
    assert second.direction == "X" and second.n_s == n_x and second.h_s == a / n_x and second.k == 1


def test_the_solve_of_s2_is_over_x_with_the_coefficients_over_y(monkeypatch):
    calls = []
    real_assemble, real_solve = cn.assemble, cn.solve_bc
    monkeypatch.setattr(cn, "assemble", lambda G, l_f, n_f, q: calls.append(("assemble", G.shape, l_f, n_f, q)) or real_assemble(G, l_f, n_f, q))
    monkeypatch.setattr(cn, "solve_bc", lambda step, A, B, C, F, D, l_s, n_s: calls.append(("solve", l_s, n_s)) or real_solve(step, A, B, C, F, D, l_s, n_s))
    result = cn.run_cccc_n23(1.0, 1.3, 1.0, 3.0, 8, 7, 2, 2)  # q = 3: the energy assembles with q = 1, the half-steps with q
    assert result.status == cn.COMPLETED_FIXED_HALF_STEPS
    solves = [c for c in calls if c[0] == "solve"]
    given = [c for c in calls if c[0] == "assemble" and c[4] == 3.0]
    assert solves == [("solve", 1.3, 7), ("solve", 1.0, 8)]  # s = 1 solves Y over b, s = 2 solves X over a
    assert given == [("assemble", (7, 2), 1.0, 8, 3.0), ("assemble", (6, 2), 1.3, 7, 3.0)]  # given sets: x at s = 1, y at s = 2


@pytest.mark.parametrize("N", [2, 3])
@pytest.mark.parametrize("a, b, n_x, n_y", S2_CASES)
def test_the_energy_of_s2_is_below_that_of_s1_exactly(a, b, n_x, n_y, N):
    """Pi_h^C of the stored fields by its definition in Fraction: w(2) lowers it below w(1) in both bases (an observation made exact)."""
    first, second = cn.run_cccc_n23(a, b, 1.0, 1.0, n_x, n_y, N, 2).steps
    e1 = energy_by_definition(first.G, first.V, a, b, n_x, n_y, 1.0, 1.0)
    e1_qr = energy_by_definition(second.companion, second.G, a, b, n_x, n_y, 1.0, 1.0)
    e2 = energy_by_definition(second.V, second.G, a, b, n_x, n_y, 1.0, 1.0)
    assert e2 < e1 and e2 < e1_qr
    assert second.P_hat < second.P_hat_previous_QR and second.field_preservation is not None and math.isfinite(second.field_preservation)


def test_the_thresholds_of_s1_are_not_applied_at_s2(monkeypatch):
    real = cn.residual_eta
    monkeypatch.setattr(cn, "residual_eta", lambda V, A, B, C, F, D, h: 10**6 * cn.U if V.shape == (7, 2) else real(V, A, B, C, F, D, h))
    result = cn.run_cccc_n23(1.0, 1.3, 1.0, 1.0, 8, 7, 2, 2)  # V of s = 2 is (n_x - 1, N) = (7, 2)
    assert result.status == cn.COMPLETED_FIXED_HALF_STEPS and result.steps[1].eta == 10**6 * cn.U


def test_a_failed_rt7_stops_after_qr_before_assembly_and_solve(monkeypatch):
    calls = []
    real_assemble, real_solve, real_qr = cn.assemble, cn.solve_bc, cn.orthonormalise
    monkeypatch.setattr(cn, "assemble", lambda *a: calls.append("assemble") or real_assemble(*a))
    monkeypatch.setattr(cn, "solve_bc", lambda *a: calls.append("solve") or real_solve(*a))
    monkeypatch.setattr(cn, "discrete_energy", lambda *a: 0.0)

    def wrong_basis(T, h):  # the QR before s = 2 returns a basis of another span: R7-3 fails on the real arrays
        G, R = real_qr(T, h)
        if T.shape == (6, 2):
            G = real_qr(T + np.random.default_rng(0).standard_normal(T.shape) * np.max(np.abs(T)), h)[0]
        return G, R

    monkeypatch.setattr(cn, "orthonormalise", wrong_basis)
    result = cn.run_cccc_n23(1.0, 1.3, 1.0, 1.0, 8, 7, 2, 2)
    second = result.steps[1]
    assert result.status == cn.STOPPED_RANK_GUARD_UNCERTIFIED and result.half_steps_completed == 1
    assert second.qr_executed is True and second.rn5["certified"] and not second.rt7["passed"] and second.rt7["R7-1"] and not second.rt7["R7-3"]
    assert calls == ["assemble", "solve"]  # only s = 1; G2 is used for nothing
    assert second.A is None and second.V is None and second.eta is None and "RT-7" in result.reason
    assert np.array_equal(result.field()[1:-1, 1:-1], result.steps[0].G @ result.steps[0].V.T)


def test_an_rt7_failure_of_any_condition_stops(monkeypatch):
    monkeypatch.setattr(cn.certificate, "rt7_conditions", lambda *a: {"passed": False, "R7-1": True, "R7-2": True, "R7-3": False,
                                                                       "reason": "injected", "ranks": None, "t": None, "tau": None,
                                                                       "sin_theta_upper": None})
    monkeypatch.setattr(cn, "solve_bc", lambda *a, _real=cn.solve_bc: _real(*a) if a[-1] == 7 else pytest.fail("s = 2 solved"))
    result = cn.run_cccc_n23(1.0, 1.3, 1.0, 1.0, 8, 7, 3, 4)
    assert result.status == cn.STOPPED_RANK_GUARD_UNCERTIFIED and result.half_steps_completed == 1 and len(result.steps) == 2


def test_an_unestablished_premise_at_s2_is_refused(monkeypatch):
    real = cn.premise_failures
    monkeypatch.setattr(cn, "premise_failures", lambda parts, n_f: ["P-3: injected"] if n_f == 7 else real(parts, n_f))  # the given set of s = 2
    result = cn.run_cccc_n23(1.0, 1.3, 1.0, 1.0, 8, 7, 2, 2)
    second = result.steps[1]
    assert result.status == cn.UNRESOLVED_C3PLUS_PREMISE and second.premises_status == cn.PREMISES_UNRESOLVED and second.V is None
    assert result.half_steps_completed == 1 and result.steps[0].premises_status == cn.PREMISES_ESTABLISHED


@pytest.mark.parametrize("N", [2, 3])
@pytest.mark.parametrize("half_steps", range(3, 9))
def test_s3_stops_before_qr(monkeypatch, N, half_steps):
    counts, qr_calls = _counting(monkeypatch)
    two = run(N, 2)
    twice, qr_twice = dict(counts), len(qr_calls)
    counts.clear()
    qr_calls.clear()
    result = run(N, half_steps)
    assert counts == twice and len(qr_calls) == qr_twice  # no QR of X, no assembly, no solve of s = 3
    assert result.status == cn.STOPPED_RANK_GUARD_UNCERTIFIED and result.half_steps_completed == 2 and len(result.steps) == 3
    third = result.steps[2]
    assert third.s == 3 and third.direction == "Y" and third.qr_executed is False and third.G is None and third.A is None
    assert third.given is result.steps[1].V and result.steps[1].status == cn.HALF_STEP_DONE and "s ≥ 3" in result.reason
    assert np.array_equal(result.field(), two.field())


def test_a_proven_dependence_of_x_stops_s3_before_qr(monkeypatch):
    real, seen = cn.exact_rank, []

    def rank(M):  # (9, 2) are T, G and then X of 10 x 9: only X is lowered
        seen.extend([1] if np.shape(M) == (9, 2) else [])
        return real(M) - 1 if np.shape(M) == (9, 2) and len(seen) == 3 else real(M)

    monkeypatch.setattr(cn, "exact_rank", rank)
    result = run(2, 3)
    assert result.status == cn.STOP_BEFORE_QR_S1 and result.half_steps_completed == 2 and result.steps[2].status == cn.STOP_BEFORE_QR_S1


@pytest.mark.parametrize("N", [2, 3])
def test_two_half_steps_repeat_bit_for_bit(N):
    first, second = run(N, 2), run(N, 2)
    for one, two in zip(first.steps, second.steps, strict=True):
        for name in ("G", "R", "A", "B", "C", "F", "V", "companion"):
            x, y = getattr(one, name), getattr(two, name)
            assert (x is None and y is None) or np.array_equal(x, y), name
        assert (one.eta, one.P_hat, one.sigma, one.status) == (two.eta, two.P_hat, two.sigma, two.status)
    assert first.steps[1].rn5["eps"] == second.steps[1].rn5["eps"] and first.steps[1].rt7["tau"] == second.steps[1].rt7["tau"]
