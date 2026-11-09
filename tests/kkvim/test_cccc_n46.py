"""``kkvim.cccc_n46`` (D-067): CCCC, N = 4, 5, 6, the first half-step s = 1 only. Builder tests, not the independent QA of D-067.

Exact checks in Fraction where the operation is exact in binary64 (integer sets, h = 1/8); otherwise the dense system of (ЧФ-C1) is solved
separately and compared with a sanity tolerance of the test, which is not a criterion of the method. No threshold of eta is applied by the
solver for N >= 4 (D-067); the tests check that none exists.
"""

import math
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

import kkvim.cccc_n1 as n1
import kkvim.cccc_n23 as n23
import kkvim.cccc_n46 as cn

U = Fraction(1, 2**53)
PHYSICAL = (1.0, 1.0, 0.01, 2.1e11, 0.3, 1000.0, 32, 32)  # a, b, t, E, nu, q, n_x, n_y of the task


def small(N):
    """Small cases of every N, the lower grid bound 2N + 1 included."""
    return [(1.0, 0.6, 2 * N + 1, 2 * N + 2), (1.0, 1.3, 2 * N + 2, 2 * N + 1), (1.0, 1.0, 16, 16)]


CASES = [(N, *case) for N in cn.N_VALUES for case in small(N)]


def run(N=4, half_steps=1, **changes):
    plate = {"a": 1.0, "b": 0.6, "D": 1.0, "q": 1.0, "n_x": 2 * N + 1 if isinstance(N, int) and N >= 1 else 13, "n_y": 14, **changes}
    return cn.run_cccc_n46(plate["a"], plate["b"], plate["D"], plate["q"], plate["n_x"], plate["n_y"], N, half_steps)


# --- definitions of Policy 2 in Fraction ---------------------------------------------------------------------------------------

def lap_full(m, h):
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


def frac(M):
    return [[Fraction(float(v)) for v in row] for row in np.atleast_2d(M)]


def d4_matrix(m, h):
    Lf, W = lap_full(m, h), weights(m, h)
    return [[sum(Lf[k][i] * W[k] * Lf[k][j] for k in range(m + 2)) / h for j in range(m)] for i in range(m)]


def l_matrix(m, h):
    return [[(Fraction(-2) if i == j else Fraction(1) if abs(i - j) == 1 else Fraction(0)) / h**2 for j in range(m)] for i in range(m)]


def coefficients(G, length, n, q=1):
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
    """Pi_h^C(w) = D/2 (Δ w)^T (W_x ⊗ W_y)(Δ w) - q h_x h_y Σ w at w = G V^T (Policy 2)."""
    hx, hy = Fraction(a) / n_x, Fraction(b) / n_y
    w = matmul(frac(G), transpose(frac(V)))
    ext = lambda m: [[Fraction(int(i == j + 1)) for j in range(m)] for i in range(m + 2)]
    lap = [[x + y for x, y in zip(r1, r2, strict=True)] for r1, r2 in
           zip(matmul(matmul(lap_full(n_x - 1, hx), w), transpose(ext(n_y - 1))),
               matmul(matmul(ext(n_x - 1), w), transpose(lap_full(n_y - 1, hy))), strict=True)]
    Wx, Wy = weights(n_x - 1, hx), weights(n_y - 1, hy)
    bending = sum(Wx[i] * Wy[j] * lap[i][j] ** 2 for i in range(n_x + 1) for j in range(n_y + 1))
    return Fraction(D) / 2 * bending - Fraction(q) * hx * hy * sum(sum(row) for row in w)


def integer_set(m, N, seed):
    return np.random.default_rng(seed).integers(-4, 5, size=(m, N)).astype(float)


# --- 1. N = 4, 5, 6: Y first, dimensions, N fixed -----------------------------------------------------------------------------

@pytest.mark.parametrize("N", cn.N_VALUES)
def test_the_task_case_completes_one_y_half_step_with_fixed_n(N):
    result = cn.run_cccc_n46_physical(*PHYSICAL, N, 1)
    assert result.status == cn.COMPLETED_FIXED_HALF_STEPS and result.reason is None and result.half_steps_completed == 1 and len(result.steps) == 1
    step = result.steps[0]
    assert (step.s, step.k, step.direction, step.n_s, step.N, step.qr_executed) == (1, 0, "Y", 32, N, True)
    assert step.T.shape == step.G.shape == (31, N) and step.R.shape == (N, N) and step.V.shape == (31, N)
    assert step.A.shape == step.B.shape == step.C.shape == (N, N) and step.F.shape == (N,) and len(step.sigma) == N
    w = result.field()
    assert w.shape == (33, 33) and np.all(np.isfinite(w)) and np.all(np.isfinite(step.V)) and np.all(np.isfinite(step.G))
    assert not np.any(w[0]) and not np.any(w[-1]) and not np.any(w[:, 0]) and not np.any(w[:, -1])
    assert np.array_equal(w[1:-1, 1:-1], step.G @ step.V.T) and np.max(np.abs(w)) > 0
    assert result.D == 2.1e11 * 0.01**3 / (12 * (1 - 0.3**2)) and result.physical == {"t": 0.01, "E": 2.1e11, "nu": 0.3}
    assert result.mode == "FIXED_HALF_STEPS_RESEARCH" and result.application_status == "RESEARCH_ONLY / PROVISIONAL / NOT_QUALIFIED"
    assert result.rank_guard_rule == cn.RANK_GUARD_RULE and "D-067" in result.rank_guard_rule


@pytest.mark.parametrize("N, a, b, n_x, n_y", CASES)
def test_small_cases_complete_with_the_shapes_of_the_grid(N, a, b, n_x, n_y):
    result = cn.run_cccc_n46(a, b, 1.0, 1.0, n_x, n_y, N, 1)
    assert result.status == cn.COMPLETED_FIXED_HALF_STEPS, result.reason
    step = result.steps[0]
    assert step.G.shape == (n_x - 1, N) and step.V.shape == (n_y - 1, N) and result.field().shape == (n_x + 1, n_y + 1)
    assert step.V_exact_rank == N and step.premises_status == cn.PREMISES_ESTABLISHED


def test_the_operations_are_those_of_the_accepted_s1_path_bit_for_bit():
    """The execution path of D-067, entered below its domain at N = 3, gives the half-step s = 1 of D-063 bit for bit."""
    for a, b, n_x, n_y in [(1.0, 0.6, 10, 9), (1.0, 1.0, 32, 32)]:
        mine = cn._execute(a, b, 1.0, 1.0, n_x, n_y, 3, 1).steps[0]
        accepted = n23.run_cccc_n23(a, b, 1.0, 1.0, n_x, n_y, 3, 1).steps[0]
        for name in ("T", "R", "G", "A", "B", "C", "F", "V"):
            assert np.array_equal(getattr(mine, name), getattr(accepted, name)), name
        assert (mine.eta, mine.P_hat, mine.cond_S, mine.sigma) == (accepted.eta, accepted.P_hat, accepted.cond_S, accepted.sigma)


# --- 2. the prescribed start set and its discrete independence ------------------------------------------------------------------

@pytest.mark.parametrize("N", cn.N_VALUES)
def test_the_start_set_is_sin_squared_of_t_n(N):
    step = run(N).steps[0]
    n_x = 2 * N + 1
    l = np.arange(1, n_x)
    assert np.array_equal(step.T, np.column_stack([np.sin(i * math.pi * l / n_x) ** 2 for i in range(1, N + 1)]))
    assert step.exact_rank_T == N and step.exact_rank_G == N and step.R_diag_min_abs > 0
    assert all(step.conditions.values())


def test_the_start_set_is_independent_and_the_conditions_hold_on_the_whole_domain():
    for N in cn.N_VALUES:
        for n in range(cn.grid_min(N), cn.N_MAX + 1):
            assert all(n23.start_conditions(N, n, n).values())
            assert n23.exact_rank(n23.start_set(n, N)) == N, (n, N)


def test_the_start_coefficients_are_those_of_vbc_13():
    a, q = 1.0, 1.0
    for N in cn.N_VALUES:
        n_x = cn.grid_min(N)
        parts = n23.assemble(n23.start_set(n_x, N), a, n_x, q)
        mu = 4.0 / (a / n_x) ** 2 * np.sin(np.arange(1, N + 1) * math.pi / n_x) ** 2
        np.testing.assert_allclose(parts["A"], a * (np.eye(N) / 8 + np.ones((N, N)) / 4), rtol=1e-13, atol=0)
        np.testing.assert_allclose(np.diag(parts["B"]), a / 8 * mu, rtol=1e-13)
        np.testing.assert_allclose(parts["F"], q * a / 2, rtol=1e-13)


@pytest.mark.parametrize("N", cn.N_VALUES)
def test_the_condition_number_of_a_before_qr_is_one_plus_2n(N):
    step = cn.run_cccc_n46_physical(*PHYSICAL, N, 1).steps[0]
    assert step.cond_A_before_QR == pytest.approx(1 + 2 * N, rel=1e-12)  # an observation of ВБC-17, not a criterion


def test_a_dependent_start_set_stops_before_qr(monkeypatch):
    monkeypatch.setattr(n23, "start_set", lambda n_x, N: np.column_stack([n1.start_factor(n_x)] * N))
    monkeypatch.setattr(n23, "orthonormalise", lambda *a: pytest.fail("QR of a dependent set"))
    result = run(5, 1)
    assert result.status == cn.STOP_BEFORE_QR_S1 and result.half_steps_completed == 0 and result.field() is None


# --- 3. C-3+ assembly and premises --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("N", cn.N_VALUES)
def test_the_coefficients_are_those_of_the_definition_exactly(N):
    G = integer_set(7, N, N)  # l = 1, n = 8: h = 1/8, every operation exact in binary64
    parts = n23.assemble(G, 1.0, 8, 3.0)
    A, B, C, F = coefficients(G, 1, 8, 3)
    assert frac(parts["A"]) == A and frac(parts["B"]) == B and frac(parts["C"]) == C and frac(parts["F"])[0] == F


@pytest.mark.parametrize("N, a, b, n_x, n_y", CASES)
def test_c3plus_bound_and_premises_on_the_run(N, a, b, n_x, n_y):
    step = cn.run_cccc_n46(a, b, 1.0, 1.0, n_x, n_y, N, 1).steps[0]
    assert step.premises_status == cn.PREMISES_ESTABLISHED and step.premises_reasons == ()
    _, _, C, _ = coefficients(step.G, a, n_x)
    k = n_x - 1 + n23.K_EXTRA
    gamma = k * U / (1 - k * U)  # the proven bound of C-3+ (D-054.3), not a new tolerance
    for i in range(N):
        for j in range(N):
            d = Fraction(float(step.C[i, j])) - C[i][j]
            assert d * d <= gamma * gamma * C[i][i] * C[j][j], (i, j)
    assert np.array_equal(step.C, step.C_L + step.E)


def test_an_unestablished_premise_is_a_protective_stop(monkeypatch):
    monkeypatch.setattr(n23, "premise_failures", lambda *a: ["P-2: injected"])
    result = run(6, 1)
    assert result.status == cn.UNRESOLVED_C3PLUS_PREMISE and result.steps[0].premises_status == cn.PREMISES_UNRESOLVED
    assert result.half_steps_completed == 0 and result.field() is None and result.steps[0].V is None


@pytest.mark.parametrize("when", ["before", "after"])
def test_a_failed_rounding_probe_is_an_unresolved_premise(monkeypatch, when):
    calls = []
    real = n1.rounding_probe_failures

    def probe():
        calls.append(1)
        return ["P-1: injected"] if (len(calls) == 1) == (when == "before") else real()

    monkeypatch.setattr(n1, "rounding_probe_failures", probe)
    assert run(4, 1).status == cn.UNRESOLVED_C3PLUS_PREMISE


# --- 4. a real B-C solve, no eta threshold -------------------------------------------------------------------------------------

@pytest.mark.parametrize("N, a, b, n_x, n_y", CASES)
def test_the_solution_solves_the_dense_system(N, a, b, n_x, n_y):
    D, q = 2.0, 3.0
    step = cn.run_cccc_n46(a, b, D, q, n_x, n_y, N, 1).steps[0]
    m, h = n_y - 1, b / n_y
    D4 = np.array(d4_matrix(m, Fraction(h)), dtype=float)
    L = np.array(l_matrix(m, Fraction(h)), dtype=float)
    K = D * (np.kron(step.A, D4) - 2.0 * np.kron(step.B, L) + np.kron(step.C, np.eye(m)))  # (ЧФ-C1), vec = [V_1; ...; V_N]
    dense = np.linalg.solve(K, np.kron(step.F, np.ones(m))).reshape(N, m).T
    scale = np.max(np.abs(dense))
    assert np.max(np.abs(step.V - dense)) <= 1e-8 * scale  # sanity of the test (two different solvers), not a criterion of the method
    assert step.eta == n23.residual_eta(step.V, step.A, step.B, step.C, step.F, D, h)  # the recorded eta is the residual of this V


def test_no_eta_guard_exists_for_n_4_to_6(monkeypatch):
    assert not hasattr(cn, "ETA_MAX_C") and "STOPPED_ETA_GUARD" not in cn.STATUSES
    monkeypatch.setattr(n23, "residual_eta", lambda *a: 1e-3)  # far above 100u of N = 2, 3
    result = run(5, 1)
    assert result.status == cn.COMPLETED_FIXED_HALF_STEPS and result.steps[0].eta == 1e-3


@pytest.mark.parametrize("name, replacement, status", [
    ("orthonormalise", lambda real: lambda T, h: (real(T, h)[0], np.diag([1.0, 1.0, 1.0, 0.0])), cn.PROTECTIVE_STOP_FACTORIZATION),
    ("solve_bc", lambda real: lambda step, *a: np.full((13, 4), np.nan), cn.ERROR_NONFINITE),
    ("solve_bc", lambda real: lambda step, *a: np.zeros((13, 4)), cn.STOP_ZERO_FIELD),
    ("residual_eta", lambda real: lambda *a: math.nan, cn.ERROR_NONFINITE),
])
@pytest.mark.parametrize("half_steps", [1, 2])
def test_every_protective_row_is_reached_and_never_completes(monkeypatch, name, replacement, status, half_steps):
    monkeypatch.setattr(n23, name, replacement(getattr(n23, name)))
    result = run(4, half_steps)
    assert result.status == status and result.steps[0].status == status and result.half_steps_completed == 0
    assert result.field() is None and len(result.steps) == 1


def test_a_failed_factorization_of_the_half_step_is_a_protective_stop(monkeypatch):
    real = n23.assemble

    def negated(*args):
        parts = real(*args)
        if args[2] == 9:  # the given set over x only
            parts["A"] = -parts["A"]
        return parts

    monkeypatch.setattr(n23, "assemble", negated)
    assert run(4, 1).status == cn.PROTECTIVE_STOP_FACTORIZATION


# --- 5. field, energy ------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("N", cn.N_VALUES)
def test_p_hat_is_pi_h_at_w_equal_g_v_transposed(N):
    G, V = integer_set(7, N, 10 + N), integer_set(7, N, 20 + N)
    assert Fraction(n23.discrete_energy(G, V, 1.0, 2.0, 2.0, 3.0, 8, 8)) == energy_by_definition(G, V, 1, 2, 8, 8, 2.0, 3.0)


@pytest.mark.parametrize("N, a, b, n_x, n_y", CASES)
def test_the_energy_of_the_half_step_solution_is_minus_half_the_work(N, a, b, n_x, n_y):
    step = cn.run_cccc_n46(a, b, 1.0, 1.0, n_x, n_y, N, 1).steps[0]
    work = (b / n_y) * float(np.sum(step.F * step.V.sum(axis=0)))  # Pi = h_y (1/2 v^T K v - (F ⊗ 1)^T v) = -1/2 work at K v = F ⊗ 1
    assert step.P_hat < 0 and step.P_hat == pytest.approx(-0.5 * work, rel=1e-9)  # sanity of the test, not a criterion


@pytest.mark.parametrize("N", cn.N_VALUES)
def test_the_centre_value_and_the_symmetry_of_the_square_plate(N):
    result = cn.run_cccc_n46_physical(*PHYSICAL, N, 1)
    w, step = result.field(), result.steps[0]
    assert step.alpha_centre == float(w[16, 16]) * result.D / (1000.0 * 1.0**4)
    assert np.unravel_index(int(np.argmax(np.abs(w))), w.shape) == (16, 16) and np.all(w[1:-1, 1:-1] > 0)
    np.testing.assert_allclose(w, w[::-1, :], rtol=0, atol=1e-12 * np.max(w))  # symmetry about x = a/2 of the field of span(G): observation


def test_a_negative_load_gives_the_negative_field():
    plus, minus = run(5, 1, q=2.0), run(5, 1, q=-2.0)
    assert np.array_equal(minus.field(), -plus.field())


@pytest.mark.parametrize("N", cn.N_VALUES)
def test_a_repeated_run_is_equal_bit_for_bit(N):
    one, two = cn.run_cccc_n46_physical(*PHYSICAL, N, 1), cn.run_cccc_n46_physical(*PHYSICAL, N, 1)
    assert np.array_equal(one.field(), two.field()) and one.steps[0].eta == two.steps[0].eta


# --- 6. the domain ---------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("changes, N, half_steps", [
    ({}, 1, 1), ({}, 2, 1), ({}, 3, 1), ({}, 7, 1), ({}, 0, 1), ({}, True, 1), ({}, 4.0, 1), ({}, 4, 0), ({}, 4, 9), ({}, 4, 1.0),
    ({"n_x": 8}, 4, 1), ({"n_y": 10}, 5, 1), ({"n_x": 12}, 6, 1), ({"n_y": 1281}, 4, 1), ({"n_x": 1281}, 6, 1),
    ({"b": 5.5}, 4, 1), ({"b": 0.19}, 6, 1),
])
def test_outside_the_domain_nothing_is_computed(monkeypatch, changes, N, half_steps):
    monkeypatch.setattr(n23, "start_set", lambda *a: pytest.fail("computed"))
    result = run(N, half_steps, **changes)
    assert result.status == cn.NOT_EXECUTED_OUTSIDE_DOMAIN and result.steps == () and result.reason


@pytest.mark.parametrize("N", cn.N_VALUES)
def test_the_bounds_of_the_domain_are_inside(N):
    low = cn.grid_min(N)
    assert run(N, 1, n_x=low, n_y=low, b=1.0).status == cn.COMPLETED_FIXED_HALF_STEPS
    assert run(N, 1, n_x=low, n_y=low, b=0.2).status == cn.COMPLETED_FIXED_HALF_STEPS
    assert run(N, 1, n_x=low, n_y=low, b=5.0).status == cn.COMPLETED_FIXED_HALF_STEPS
    assert cn.domain_refusal(1.0, 1.0, 1280, 1280, N, 8) is None and cn.domain_refusal(1.0, 1.0, low - 1, low, N, 1)


@pytest.mark.parametrize("changes", [{"q": 0.0}, {"q": math.nan}, {"D": -1.0}, {"a": math.inf}, {"n_x": 10.0}])
def test_invalid_input_is_error_input(monkeypatch, changes):
    monkeypatch.setattr(n23, "start_set", lambda *a: pytest.fail("computed"))
    result = run(4, 1, **changes)
    assert result.status == cn.ERROR_INPUT and result.steps == ()


def test_physical_input_is_checked_like_the_other_commands():
    result = cn.run_cccc_n46_physical(1.0, 1.0, -0.01, 2.1e11, 0.3, 1000.0, 32, 32, 4, 1)
    assert result.status == cn.ERROR_INPUT and result.D is None and result.steps == ()


# --- 7. s >= 2 stops before the QR of the solved set ---------------------------------------------------------------------------

def _counting(monkeypatch, names=("orthonormalise", "assemble", "solve_bc", "residual_eta", "rank_diagnostic", "start_set")):
    counts = {}
    for name in names:
        real = getattr(n23, name)

        def counted(*args, _real=real, _name=name, **kwargs):
            counts[_name] = counts.get(_name, 0) + 1
            return _real(*args, **kwargs)

        monkeypatch.setattr(n23, name, counted)
    qr_calls = []
    real_qr = np.linalg.qr
    monkeypatch.setattr(np.linalg, "qr", lambda *a, **k: qr_calls.append(1) or real_qr(*a, **k))
    return counts, qr_calls


@pytest.mark.parametrize("N", cn.N_VALUES)
@pytest.mark.parametrize("half_steps", range(2, 9))
def test_s2_stops_before_qr_without_assembly_or_solve(monkeypatch, N, half_steps):
    monkeypatch.setattr(n23.certificate, "rn5_certificate", lambda *a: pytest.fail("RN-5 is not extended to N >= 4"))
    monkeypatch.setattr(n23.certificate, "rt7_conditions", lambda *a: pytest.fail("RT-7 is not extended to N >= 4"))
    counts, qr_calls = _counting(monkeypatch)
    one = run(N, 1)
    once, qr_once = dict(counts), len(qr_calls)
    counts.clear()
    qr_calls.clear()
    result = run(N, half_steps)
    assert counts == once and len(qr_calls) == qr_once  # nothing of s = 2 is computed, no QR of V
    assert once["orthonormalise"] == 1 and once["solve_bc"] == 1
    assert result.status == cn.STOPPED_RANK_GUARD_UNCERTIFIED and result.half_steps_completed == 1 and len(result.steps) == 2
    second = result.steps[1]
    assert (second.s, second.direction, second.qr_executed, second.status) == (2, "X", False, result.status)
    assert second.G is None and second.R is None and second.A is None and second.V is None and second.eta is None and second.rn5 is None
    assert second.conditions == {"S-1 (RN-3): N > floor(n_x/2)": False} and "D-067" in second.reason
    assert np.array_equal(result.field(), one.field()) and np.array_equal(second.given, result.steps[0].V)


def test_a_proven_dependence_of_v_stops_s2_before_qr(monkeypatch):
    real = n23.exact_rank
    monkeypatch.setattr(n23, "exact_rank", lambda M: real(M) - 1 if np.shape(M) == (13, 4) else real(M))  # only V of 9 x 14
    result = run(4, 2)
    assert result.status == cn.STOP_BEFORE_QR_S1 and result.half_steps_completed == 1 and result.steps[1].status == cn.STOP_BEFORE_QR_S1


# --- 8. no convergence, no accepted field ----------------------------------------------------------------------------------------

def test_no_status_says_converged_or_qualified():
    for status in cn.STATUSES:
        assert "CONVERG" not in status and "QUALIF" not in status and "PASS" not in status and "ACCEPT" not in status
    text = Path(cn.__file__).read_text(encoding="utf-8")
    assert '"CONVERGED"' not in text and "field.csv" not in text


# --- 10. N = 1 and N = 2-3 are unchanged ------------------------------------------------------------------------------------------

def test_the_other_cccc_modules_keep_their_domains():
    assert n23.N_VALUES == (2, 3) and n23.ETA_MAX_C == 100 * 2.0**-53
    for N in cn.N_VALUES:
        assert n23.run_cccc_n23(1.0, 1.0, 1.0, 1.0, 32, 32, N, 1).status == n23.NOT_EXECUTED_OUTSIDE_DOMAIN
    for N in (1, 2, 3):
        assert cn.run_cccc_n46(1.0, 1.0, 1.0, 1.0, 32, 32, N, 1).status == cn.NOT_EXECUTED_OUTSIDE_DOMAIN
    assert n23.run_cccc_n23(1.0, 0.6, 1.0, 1.0, 10, 9, 3, 1).status == n23.COMPLETED_FIXED_HALF_STEPS
    assert n1.run_cccc_n1(1.0, 0.6, 1.0, 1.0, 10, 9, 2).status == n1.COMPLETED_FIXED_HALF_STEPS


def test_the_module_imports_no_reference_and_no_ssss_scheme():
    text = Path(cn.__file__).read_text(encoding="utf-8")
    assert "kkvim_reference" not in text and "import scripts" not in text
    for forbidden in ("kkvim.coefficients", "kkvim.linear_solver", "kkvim.coupled", "kkvim.controller", "cccc_n23_certificate"):
        assert forbidden not in text, forbidden
