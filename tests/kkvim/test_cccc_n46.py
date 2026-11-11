"""``kkvim.cccc_n46`` (D-067, D-068): CCCC, N = 4, 5, 6, the half-step s = 1 and the guarded s = 2. Builder tests, not the independent QA.

Exact checks in Fraction where the operation is exact in binary64 (integer sets, h = 1/8); otherwise the dense system of (ЧФ-C1) is solved
separately and compared with a sanity tolerance of the test, which is not a criterion of the method. No threshold of eta is applied by the
solver for N >= 4 (D-067) or at s = 2 (D-068); the tests check that none exists. The s = 2 path runs RN-5 and RT-7 of
``kkvim.cccc_n23_certificate`` in exact arithmetic; the cases with the whole path use the smallest grids (2N + 1) to stay short.
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
    assert result.rank_guard_rule == cn.RANK_GUARD_RULE and "D-067" in result.rank_guard_rule and "D-068" in result.rank_guard_rule


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


# --- 7. s = 2 under RN-5 and RT-7 (D-068); s >= 3 stops before the QR of X ---------------------------------------------------

def run2(N=4, half_steps=2, **changes):
    """The smallest square grid 2N + 1 of N, where the whole s = 2 path is short."""
    low = cn.grid_min(N)
    plate = {"a": 1.0, "b": 1.0, "D": 1.0, "q": 1.0, "n_x": low, "n_y": low, **changes}
    return cn.run_cccc_n46(plate["a"], plate["b"], plate["D"], plate["q"], plate["n_x"], plate["n_y"], N, half_steps)


def _counting(monkeypatch, names=("orthonormalise", "_solve_given", "solve_bc", "residual_eta", "rank_diagnostic", "start_set")):
    counts = {}
    for name in names:
        real = getattr(n23, name)

        def counted(*args, _real=real, _name=name, **kwargs):
            counts[_name] = counts.get(_name, 0) + 1
            return _real(*args, **kwargs)

        monkeypatch.setattr(n23, name, counted)
    return counts


def _log_calls(monkeypatch, log):
    """The order of RN-5, QR, RT-7 and the half-step (assembly C-3+ and B-C in _solve_given); the real functions are called."""
    for module, name in ((n23.certificate, "rn5_certificate"), (n23.certificate, "rt7_conditions"), (n23, "orthonormalise"),
                         (n23, "_solve_given"), (n23, "solve_bc")):
        real = getattr(module, name)
        monkeypatch.setattr(module, name, lambda *a, _real=real, _name=name, **k: log.append(_name) or _real(*a, **k))


@pytest.mark.parametrize("N", cn.N_VALUES)
def test_s2_completes_with_rn5_qr_rt7_and_the_x_half_step(monkeypatch, N):
    log = []
    _log_calls(monkeypatch, log)
    result = run2(N)
    assert result.status == cn.COMPLETED_FIXED_HALF_STEPS and result.reason is None and result.half_steps_completed == 2, result.reason
    assert log == ["orthonormalise", "_solve_given", "solve_bc", "rn5_certificate", "orthonormalise", "rt7_conditions", "_solve_given", "solve_bc"]
    first, second = result.steps
    low = cn.grid_min(N)
    assert first.status == cn.HALF_STEP_DONE and second.status == cn.COMPLETED_FIXED_HALF_STEPS and first.done and second.done
    assert (second.s, second.k, second.direction, second.n_s, second.N, second.qr_executed) == (2, 1, "X", low, N, True)
    assert second.rn5["certified"] and second.rn5["verdict"] == "RN5_CERTIFIED" and all(second.rn5["hypotheses"].values())
    assert second.rt7["passed"] and second.rt7["R7-1"] and second.rt7["R7-2"] and second.rt7["R7-3"]
    assert second.rt7["tau"] < (1 - second.rt7["t"]) ** 2 and 0 <= second.rt7["t"] < 1  # the strict conditions, exact
    assert np.array_equal(second.given, first.V) and second.G.shape == (low - 1, N) and second.V.shape == (low - 1, N)
    G2, R = n23.orthonormalise(first.V, 1.0 / low)  # G2 is the QR of the saved V with the normalisation over y
    assert np.array_equal(second.G, G2) and np.array_equal(second.R, R) and np.array_equal(second.companion, first.G @ R.T)
    x, y = result.factors()
    assert x is second.V and y is second.G  # axis 0 = x: X solved at s = 2, G2 over y
    w = result.field()
    assert np.array_equal(w[1:-1, 1:-1], second.V @ second.G.T) and not np.any(w[0]) and not np.any(w[:, -1])
    assert second.V_exact_rank == N and second.premises_status == cn.PREMISES_ESTABLISHED and math.isfinite(second.eta)
    assert second.P_hat < 0 and second.P_hat <= first.P_hat * (1 - 1e-12)  # an observation of the minimisation (not a criterion)
    assert second.conditions == {"S-1 (RN-3): N > floor(n_x/2)": False, cn.GRID_KEY: True}


def test_s1_is_unchanged_when_s2_is_asked_for():
    one, two = run2(4, 1), run2(4, 2)
    a, b = one.steps[0], two.steps[0]
    for name in ("T", "G", "R", "A", "B", "C", "F", "V"):
        assert np.array_equal(getattr(a, name), getattr(b, name)), name
    assert (a.eta, a.P_hat, a.V_exact_rank, a.status) == (b.eta, b.P_hat, b.V_exact_rank, cn.COMPLETED_FIXED_HALF_STEPS)


def test_the_x_half_step_solves_the_exchanged_dense_system():
    result = run2(4)
    second = result.steps[1]
    low = cn.grid_min(4)
    A, B, C, F = coefficients(second.G, 1.0, low)  # the given set G2 over y (b = 1, n_y = low), the solved direction x
    m, h = low - 1, Fraction(1, low)
    K = np.kron(np.array(A, dtype=float), np.array(d4_matrix(m, h), dtype=float)) - 2 * np.kron(
        np.array(B, dtype=float), np.array(l_matrix(m, h), dtype=float)) + np.kron(np.array(C, dtype=float), np.eye(m))
    x = np.linalg.solve(K, np.kron(np.array(F, dtype=float), np.ones(m)))
    dense = x.reshape(4, m).T
    assert np.max(np.abs(second.V - dense)) <= 1e-8 * np.max(np.abs(dense))  # sanity of the test (two solvers), not a criterion


def test_no_eta_guard_at_s2(monkeypatch):
    real = n23.residual_eta
    monkeypatch.setattr(n23, "residual_eta", lambda V, *a: 1e-3 if V.shape == (8, 4) and len(calls) else calls.append(1) or real(V, *a))
    calls = []
    result = run2(4)
    assert result.status == cn.COMPLETED_FIXED_HALF_STEPS and result.steps[1].eta == 1e-3


def test_rn5_not_certified_stops_before_qr(monkeypatch):
    counts = _counting(monkeypatch)
    monkeypatch.setattr(n23.certificate, "rn5_certificate", lambda *a: {"certified": False, "verdict": "RN5_UNRESOLVED", "reason": "injected",
                                                                         "hypotheses": {}, "values": {}, "eps": None, "sigma_low": None})
    monkeypatch.setattr(n23.certificate, "rt7_conditions", lambda *a: pytest.fail("RT-7 without RN-5"))
    result = run2(4, 3)
    assert result.status == cn.STOPPED_RANK_GUARD_UNCERTIFIED and result.half_steps_completed == 1 and len(result.steps) == 2
    assert counts["orthonormalise"] == 1 and counts["_solve_given"] == 1 and counts["solve_bc"] == 1  # no QR of V, nothing of s = 2
    second = result.steps[1]
    assert second.qr_executed is False and second.G is None and second.R is None and second.A is None and second.V is None
    assert second.rn5["verdict"] == "RN5_UNRESOLVED" and second.rt7 is None and "RN-5" in result.reason and "D-068" in result.reason
    assert np.array_equal(result.field(), run2(4, 1).field())  # the field of s = 1, diagnostic


@pytest.mark.parametrize("N, n_x, n_y", [(4, 33, 9), (5, 11, 33), (6, 40, 1280)])
@pytest.mark.parametrize("half_steps", [2, 8])
def test_a_grid_beyond_32_keeps_s1_and_stops_before_rn5_and_qr(monkeypatch, N, n_x, n_y, half_steps):
    monkeypatch.setattr(n23.certificate, "rn5_certificate", lambda *a: pytest.fail("RN-5 beyond 32"))
    monkeypatch.setattr(n23.certificate, "rt7_conditions", lambda *a: pytest.fail("RT-7 beyond 32"))
    counts = _counting(monkeypatch)
    result = cn.run_cccc_n46(1.0, 1.0, 1.0, 1.0, n_x, n_y, N, half_steps)
    assert result.status == cn.STOPPED_RANK_GUARD_UNCERTIFIED and result.half_steps_completed == 1 and len(result.steps) == 2
    assert counts["orthonormalise"] == 1 and counts["solve_bc"] == 1
    second = result.steps[1]
    assert second.qr_executed is False and second.rn5 is None and second.G is None and second.V is None
    assert second.conditions[cn.GRID_KEY] is False and "D-068" in result.reason
    assert result.factors()[1] is result.steps[0].V  # s = 1 is the last completed half-step


def test_a_proven_dependence_of_v_stops_s2_before_qr(monkeypatch):
    real = n23.exact_rank
    monkeypatch.setattr(n23, "exact_rank", lambda M: real(M) - 1 if np.shape(M) == (13, 4) else real(M))  # only V of 9 x 14
    monkeypatch.setattr(n23.certificate, "rn5_certificate", lambda *a: pytest.fail("RN-5 at a proven dependence"))
    counts = _counting(monkeypatch)
    result = run(4, 2)
    assert result.status == cn.STOP_BEFORE_QR_S1 and result.half_steps_completed == 1 and result.steps[1].status == cn.STOP_BEFORE_QR_S1
    assert result.steps[1].qr_executed is False and counts["orthonormalise"] == 1


@pytest.mark.parametrize("rt7", [
    {"passed": False, "R7-1": True, "R7-2": True, "R7-3": False, "reason": "R7-3: injected", "ranks": None, "t": None, "tau": None,
     "sin_theta_upper": None},
    {"passed": False, "R7-1": False, "R7-2": False, "R7-3": False, "reason": "R7-1: injected", "ranks": None, "t": None, "tau": None,
     "sin_theta_upper": None},
])
def test_rt7_failure_prevents_assembly_and_solve(monkeypatch, rt7):
    monkeypatch.setattr(n23.certificate, "rt7_conditions", lambda *a: rt7)
    counts = _counting(monkeypatch)
    result = run2(4, 2)
    assert result.status == cn.STOPPED_RANK_GUARD_UNCERTIFIED and result.half_steps_completed == 1
    assert counts["orthonormalise"] == 2 and counts["_solve_given"] == 1 and counts["solve_bc"] == 1  # QR of V done; G2 never assembled
    second = result.steps[1]
    assert second.qr_executed is True and second.rn5["certified"] and second.rt7 is rt7
    assert second.A is None and second.V is None and second.eta is None and "RT-7" in result.reason and "G2" in result.reason
    assert np.array_equal(result.field(), run2(4, 1).field())


@pytest.mark.parametrize("fake", [
    {"certified": True, "verdict": "RN5_CERTIFIED", "eps": None, "sigma_low": None},  # missing quantities
    {"certified": True, "verdict": "RN5_CERTIFIED", "eps": Fraction(1), "sigma_low": Fraction(1, 2)},  # t = 2: not this run's certificate
    {"certified": True, "verdict": "RN5_CERTIFIED", "eps": 1e-20, "sigma_low": 1.0},  # floats, not exact certificate values
])
def test_missing_or_mismatched_certificate_data_cannot_authorize_s2(monkeypatch, fake):
    monkeypatch.setattr(n23.certificate, "rn5_certificate", lambda *a: {"reason": "injected", "hypotheses": {}, "values": {}, **fake})
    counts = _counting(monkeypatch)
    result = run2(4, 2)
    second = result.steps[1]
    assert result.status == cn.STOPPED_RANK_GUARD_UNCERTIFIED and result.half_steps_completed == 1
    assert second.rt7["passed"] is False and second.rt7["R7-2"] is False and counts["_solve_given"] == 1 and second.V is None


def test_an_unestablished_premise_at_s2_stops_safely(monkeypatch):
    real = n23.premise_failures
    monkeypatch.setattr(n23, "premise_failures", lambda parts, n: ["P-2: injected at s = 2"] if calls.append(1) or len(calls) > 1 else real(parts, n))
    calls = []
    result = run2(4)
    assert result.status == cn.UNRESOLVED_C3PLUS_PREMISE and result.half_steps_completed == 1 and result.steps[1].qr_executed is True
    assert result.steps[1].premises_status == cn.PREMISES_UNRESOLVED and result.steps[1].V is None


@pytest.mark.parametrize("replacement, status", [
    (lambda step, *a: np.full((8, 4), np.nan), cn.ERROR_NONFINITE), (lambda step, *a: np.zeros((8, 4)), cn.STOP_ZERO_FIELD),
])
def test_nonfinite_and_zero_at_s2_are_protected(monkeypatch, replacement, status):
    real = n23.solve_bc
    monkeypatch.setattr(n23, "solve_bc", lambda step, *a: replacement(step, *a) if step.s == 2 else real(step, *a))
    result = run2(4)
    assert result.status == status and result.half_steps_completed == 1 and result.steps[1].status == status


def test_a_failed_factorization_at_s2_is_a_protective_stop(monkeypatch):
    real = n23.assemble

    def negated(G, length, n, q):
        parts = real(G, length, n, q)
        if q == 2.0:  # the half-step assemblies (the energy assembles with q = 1)
            calls.append(1)
            if len(calls) == 2:  # the given set G2 of s = 2 only
                parts["A"] = -parts["A"]
        return parts

    calls = []
    monkeypatch.setattr(n23, "assemble", negated)
    result = run2(4, q=2.0)
    assert result.status == cn.PROTECTIVE_STOP_FACTORIZATION and result.half_steps_completed == 1 and result.steps[1].V is None


@pytest.mark.parametrize("half_steps", [3, 8])
def test_s3_stops_before_qr_assembly_and_solve(monkeypatch, half_steps):
    counts = _counting(monkeypatch)
    result = run2(4, half_steps)
    assert result.status == cn.STOPPED_RANK_GUARD_UNCERTIFIED and result.half_steps_completed == 2 and len(result.steps) == 3
    assert counts["orthonormalise"] == 2 and counts["_solve_given"] == 2 and counts["solve_bc"] == 2  # nothing of s = 3
    third = result.steps[2]
    assert (third.s, third.direction, third.qr_executed, third.status) == (3, "Y", False, result.status)
    assert third.G is None and third.A is None and third.V is None and "s = 3" in third.reason and "D-068" in third.reason
    assert np.array_equal(result.field(), run2(4, 2).field())  # the field of s = 2, diagnostic


def test_a_changed_grid_bound_of_cccc_n23_does_not_authorize_s2(monkeypatch):
    monkeypatch.setattr(n23, "S2_GRID_MAX", 40)
    monkeypatch.setattr(n23.certificate, "rn5_certificate", lambda *a: pytest.fail("RN-5"))
    result = run2(4)
    assert result.status == cn.STOPPED_RANK_GUARD_UNCERTIFIED and result.steps[1].qr_executed is False and "D-068" in result.reason


def test_no_rank_reduction_regularization_or_replacement_of_directions():
    result = run2(5, 2)
    assert all(step.N == 5 for step in result.steps) and result.steps[1].V.shape[1] == 5 and result.steps[1].G.shape[1] == 5
    text = Path(cn.__file__).read_text(encoding="utf-8")
    for forbidden in ("lstsq", "pinv", "regulari", "truncat", "svd(first", "rank_tol"):
        assert forbidden not in text, forbidden


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
    for forbidden in ("kkvim.coefficients", "kkvim.linear_solver", "kkvim.coupled", "kkvim.controller", "import kkvim.cccc_n23_certificate"):
        assert forbidden not in text, forbidden
    for own in ("def rn5", "def rt7", "def solve_bc", "def assemble", "def orthonormalise", "Fraction("):  # RN-5, RT-7, B-C, C-3+ not duplicated
        assert own not in text, own
