"""QA comparison of the Builder research controller N = 2-3 (R4 step 16) with the independent QA references.

Versions: Builder `builder/r4-coupled-modal-foundation` @ 4c86c1f099cafad0a6b2c9368582ca9af2dfee81, QA
`qa/r4-prequalification-address-review` @ 751199d0df345961c2dfb2cd949af584b35a609b (Reviewer 3aaca8e).
The QA references (kkvim_reference.r4_*) never import the Builder kernel; only this test imports both packages.

Combined environment (reproducible; docs/R4_CONTROLLER_COMPARISON.qa-report.md, section 1):
    git archive 4c86c1f src/kkvim tests/kkvim | tar -x -C scratch/r4cmp/builder
    git archive <this QA commit> src/kkvim_reference tests scripts docs pyproject.toml | tar -x -C scratch/r4cmp/qa
    KKVIM_R4_COMBINED=1 PYTHONPATH="scratch/r4cmp/builder/src;scratch/r4cmp/qa/src" \
        uv run pytest scratch/r4cmp/qa/tests/cross/test_r4_controller_comparison.py
In the ordinary CI of the QA branch the Builder modules are absent and the module is SKIPPED (not passed); with
KKVIM_R4_COMBINED=1 a missing Builder module is an error, never a skip.

All Builder results of the research mode are PROVISIONAL / NOT_QUALIFIED (D-047 addendum 1). The tolerances below are
first-order MODELS fixed before the comparison from conditioning and rounding quantities (unit constants, not fitted);
"consistent" means "within the model", not a proven bound and not a normative PASS (A4-1..A4-14 stay NOT_QUALIFIED).
"""

import math
import os

import numpy as np
import pytest

COMBINED = os.environ.get("KKVIM_R4_COMBINED") == "1"
try:
    import kkvim.coupled_run as cr
    from kkvim.coupled import odd_sine_set, solve_coupled_halfstep
    from kkvim.coupled_energy import field_from_sets
    from kkvim.coupled_rank import orthogonalize_received
    from kkvim.grid import Grid1D
except ImportError:
    if COMBINED:
        raise
    pytest.skip("Builder R4 controller (kkvim.coupled_run) not present: combined QA run only", allow_module_level=True)

from kkvim_reference.r4_spectral_control import SpectralGrid, nodal_energy, nodal_laplacian, run_control

U = 2.0**-53
S_LIST = (1, 2, 3, 4, 10, 20, 40)
RATIOS = {"0,2": 0.2, "1/3": 1 / 3, "1/2": 0.5, "1": 1.0, "2": 2.0, "3": 3.0, "5": 5.0}
RECT = {"0,2": (100, 20), "1/3": (60, 20), "1/2": (40, 20), "1": (40, 60), "2": (20, 40), "3": (20, 60), "5": (20, 100)}


# --- harness: Builder's own functions in the order of run_coupled, keeping the sets -------------------------------

def harness(r, N, nx, ny, transform, s_max):
    """Half-steps with Builder's solve and QR (as in run_coupled); records the arrays used by each half-step."""
    gx, gy = Grid1D(1.0, nx), Grid1D(r, ny)
    X, Y = odd_sine_set(gx, N), None
    out = []
    for s in range(1, s_max + 1):
        direction = "Y" if s % 2 else "X"
        given_grid, solve_grid = (gx, gy) if direction == "Y" else (gy, gx)
        given, companion = (X, Y) if direction == "Y" else (Y, X)
        step = solve_coupled_halfstep(given_grid, given, solve_grid, 1.0, 1.0, direction=direction, transform=transform)
        new = (X, step.solved) if direction == "Y" else (step.solved, Y)
        out.append({"s": s, "prev": (X, Y), "given": given, "companion": companion, "step": step, "new": new,
                    "given_grid": given_grid, "solve_grid": solve_grid, "gx": gx, "gy": gy})
        if s < s_max:
            o = orthogonalize_received(solve_grid, step.solved, given)
            X, Y = (o.companion, o.received) if direction == "Y" else (o.received, o.companion)
        else:
            X, Y = new
    return out


def field(X, Y):
    return np.asarray(X) @ np.asarray(Y).T


# --- independent QA energy quantities (R4_NUMERICAL_SPEC 6.2, 6.3; own implementation) -------------------------

def qa_coefficients(U_, h, q=1.0):
    P = np.zeros((U_.shape[0] + 2, U_.shape[1]))
    P[1:-1] = U_
    dU = (P[1:] - P[:-1]) / h
    lU = (P[:-2] - 2.0 * P[1:-1] + P[2:]) / h**2
    fs = lambda v: math.fsum(np.ravel(v).tolist())  # noqa: E731
    N = U_.shape[1]
    A = np.array([[h * fs(U_[:, i] * U_[:, j]) for j in range(N)] for i in range(N)])
    B = np.array([[h * fs(dU[:, i] * dU[:, j]) for j in range(N)] for i in range(N)])
    C = np.array([[h * fs(lU[:, i] * lU[:, j]) for j in range(N)] for i in range(N)])
    F = np.array([q * h * fs(U_[:, i]) for i in range(N)])
    return A, B, C, F


def qa_apply_K(V, h, A, B, C, D=1.0):
    P = np.zeros((V.shape[0] + 2, V.shape[1]))
    P[1:-1] = V
    LV = (P[:-2] - 2.0 * P[1:-1] + P[2:]) / h**2
    Q = np.zeros_like(P)
    Q[1:-1] = LV
    L2V = (Q[:-2] - 2.0 * Q[1:-1] + Q[2:]) / h**2
    return D * (L2V @ A.T - 2.0 * (LV @ B.T) + V @ C.T)


def a_form(W, hx, hy, D=1.0):
    lap = nodal_laplacian(W, hx, hy)
    return D * hx * hy * math.fsum((lap * lap).ravel().tolist())


def abs_lap(W, hx, hy):
    P = np.zeros((W.shape[0] + 2, W.shape[1] + 2))
    P[1:-1, 1:-1] = np.abs(W)
    return (P[:-2, 1:-1] + 2 * P[1:-1, 1:-1] + P[2:, 1:-1]) / hx**2 + (P[1:-1, :-2] + 2 * P[1:-1, 1:-1] + P[1:-1, 2:]) / hy**2


def rounding_R(W, hx, hy, D=1.0, q=1.0):
    """R(w) = u (ceil(log2 N) + 16) S_hat(w) (NUMERICAL_SPEC 10.3), own implementation."""
    n_nodes = W.size
    s_hat = D * hx * hy * math.fsum((np.abs(nodal_laplacian(W, hx, hy)) * abs_lap(W, hx, hy)).ravel().tolist())
    s_hat += abs(q) * hx * hy * math.fsum(np.abs(W).ravel().tolist())
    return U * (math.ceil(math.log2(n_nodes)) + 16) * s_hat


def eps_a(X, Y, hx, hy, D=1.0):
    N = X.shape[1]
    S = np.abs(X) @ np.abs(Y).T
    gamma_N = N * U / (1 - N * U)
    return math.sqrt(D) * (4 / hx**2 + 4 / hy**2) * gamma_N * math.sqrt(hx * hy * math.fsum((S * S).ravel().tolist()))


def b_omega(W, hx, hy, nx, ny, a, b, D=1.0, q=1.0):
    mu = 4 / hx**2 * math.sin(math.pi / (2 * nx)) ** 2 + 4 / hy**2 * math.sin(math.pi / (2 * ny)) ** 2
    r = D * nodal_laplacian(nodal_laplacian(W, hx, hy), hx, hy) - q
    return math.sqrt(hx * hy * math.fsum((r * r).ravel().tolist())) / (math.sqrt(D) * mu)


def qa_energy(rec, a, b, nx, ny):
    """Independent P, delta_P, Q, gamma, R, R_f, tau, E1, E2 for the arrays of one recorded half-step s >= 2."""
    hx, hy = a / nx, b / ny
    Xp, Yp = rec["prev"]
    Xn, Yn = rec["new"]
    Wp, Wn = field(Xp, Yp), field(Xn, Yn)
    d = Wp - Wn
    Pp, Pn = nodal_energy(Wp, hx, hy), nodal_energy(Wn, hx, hy)
    Q = 0.5 * a_form(d, hx, hy)
    h_fix, h_sol = rec["given_grid"].h, rec["solve_grid"].h
    A, B, C, F = qa_coefficients(np.asarray(rec["given"]), h_fix)
    V = np.asarray(rec["step"].solved)
    rho = F[None, :] - qa_apply_K(V, h_sol, A, B, C)
    gamma = h_sol * math.fsum((rho * (np.asarray(rec["companion"]) - V)).ravel().tolist())
    Rp, Rn = rounding_R(Wp, hx, hy), rounding_R(Wn, hx, hy)
    ea_p, ea_n = eps_a(Xp, Yp, hx, hy), eps_a(Xn, Yn, hx, hy)
    bw = b_omega(Wn, hx, hy, nx, ny, a, b)
    Rf = ea_n * math.sqrt(2 * Q) + bw * (ea_p + ea_n)
    tau = abs(gamma) + 10 * (Rp + Rn + Rf)
    dP = Pp - Pn
    return {"P_prev": Pp, "P": Pn, "delta_P": dP, "Q": Q, "gamma": gamma, "R_prev": Rp, "R": Rn, "R_f": Rf, "tau": tau,
            "B_omega": bw, "eps_a": ea_n, "e1": dP >= -tau, "e2": abs(dP - Q) <= tau, "resolved": Q >= 10 * tau}


# --- tests -----------------------------------------------------------------------------------------------------------

def test_harness_reproduces_run_coupled_bitwise():
    for N, r, g, tr in ((2, 1.0, (40, 40), "dense"), (3, 0.2, (100, 20), "fft"), (3, 5.0, (20, 100), "dense")):
        run = cr.run_coupled(1.0, r, 1.0, 1.0, *g, N, transform=tr, half_steps=12, research_mode=True)
        rec = harness(r, N, *g, tr, 12)
        assert run.status == cr.RESEARCH_LIMIT_REACHED and len(run.steps) == 12
        for log, h in zip(run.steps, rec):
            w = field_from_sets(h["gx"], h["gy"], *h["new"])
            assert log.w_max == float(np.max(np.abs(w)))
        np.testing.assert_array_equal(run.x, rec[-1]["new"][0])
        np.testing.assert_array_equal(run.y, rec[-1]["new"][1])


def _model(log_steps, qa_steps, j_max, nx, ny):
    """First-order model (unit constants) of the difference of two independent evaluations of the field at s = j_max:
    sum over half-steps of u [cond(M)(1 + (2 n_fix/pi)^2) + cond_LS^2 + log2(n_solve) + kappa_rep sqrt(N)]."""
    total = 0.0
    for j in range(j_max):
        lg, qa = log_steps[j], qa_steps[j]
        n_fix, n_sol = (nx, ny) if lg.direction == "Y" else (ny, nx)
        N = lg.rank.sigma.size
        total += U * (lg.cond_M_max * (1 + (2 * n_fix / math.pi) ** 2) + qa.ls_cond_max**2 + math.log2(n_sol)
                      + lg.rank.kappa_rep * math.sqrt(N))
    return total


CASES = [(N, lab, g, tr) for N in (2, 3) for lab in RATIOS for g in ((40, 40), RECT[lab]) for tr in ("dense", "fft")]


@pytest.mark.parametrize("N, lab, grid, transform", CASES, ids=lambda v: str(v))
def test_fields_energies_and_singular_values_against_the_qa_control(N, lab, grid, transform):
    r = RATIOS[lab]
    nx, ny = grid
    hx, hy = 1.0 / nx, r / ny
    run = cr.run_coupled(1.0, r, 1.0, 1.0, nx, ny, N, transform=transform, half_steps=max(S_LIST), research_mode=True)
    assert run.status == cr.RESEARCH_LIMIT_REACHED, run.reason
    rec = harness(r, N, nx, ny, transform, max(S_LIST))
    g = SpectralGrid(1.0, r, nx, ny)
    qa = run_control(g, N, max(S_LIST))
    omega_a = math.sqrt(-2.0 * g.omega_energy())
    for s in S_LIST:
        Wb = field(*rec[s - 1]["new"])
        Wq = qa[s - 1].W
        eps = _model(run.steps, qa, s, nx, ny)
        assert np.max(np.abs(Wb - Wq)) <= eps * np.max(np.abs(Wq)), (s, eps)
        # Weyl: singular values move by at most the 2-norm of the field difference (+ diagnostic rounding)
        dW2 = np.linalg.norm(Wb - Wq, 2)
        sb, sq = run.steps[s - 1].rank.sigma / math.sqrt(hx * hy), qa[s - 1].sigma
        assert np.all(np.abs(sb - sq) <= dW2 + 8 * U * sq[0] * max(1.0, run.steps[s - 1].rank.kappa_rep))
        # energy: |Pi(w1) - Pi(w2)| <= (|w|_a + |omega_h|_a) |w1 - w2|_a + rounding of both evaluations
        da = math.sqrt(a_form(Wb - Wq, hx, hy))
        Pb = run.steps[s - 1].energy.P
        assert abs(Pb - qa[s - 1].energy_spectral) <= (math.sqrt(a_form(Wq, hx, hy)) + omega_a) * da \
            + rounding_R(Wb, hx, hy) + rounding_R(Wq, hx, hy)
        # direction, counters and N are kept
        lg = run.steps[s - 1]
        assert lg.direction == ("Y" if s % 2 else "X") and lg.k == s // 2 and lg.rank.sigma.size == N


@pytest.mark.parametrize("N, lab, grid, transform", [(2, "1", (40, 40), "fft"), (3, "0,2", (100, 20), "dense"),
                                                      (3, "5", (20, 100), "fft"), (2, "3", (20, 60), "dense")])
def test_independent_energy_quantities_against_the_builder_log(N, lab, grid, transform):
    r = RATIOS[lab]
    nx, ny = grid
    run = cr.run_coupled(1.0, r, 1.0, 1.0, nx, ny, N, transform=transform, half_steps=20, research_mode=True)
    rec = harness(r, N, nx, ny, transform, 20)
    for log, h in zip(run.steps[1:], rec[1:]):
        e, b = qa_energy(h, 1.0, r, nx, ny), log.energy
        tau = e["tau"]
        for key in ("P_prev", "P"):  # same quantity, different summation: within the rounding model R
            assert abs(getattr(b, key) - e[key]) <= e["R"] + e["R_prev"]
        assert abs(b.delta_P - e["delta_P"]) <= tau and abs(b.Q - e["Q"]) <= tau and abs(b.gamma - e["gamma"]) <= tau
        for key in ("R_prev", "R"):  # positive sums: relative 2 n u
            assert abs(getattr(b, key) - e[key]) <= 2 * (nx * ny) * U * e[key]
        assert abs(b.eps_a - e["eps_a"]) <= 2 * (nx * ny) * U * e["eps_a"]
        assert abs(b.tau - tau) <= 0.01 * tau  # tau differs only through gamma, B_omega and summation order
        assert (b.e1, b.e2) == (e["e1"], e["e2"]) and ((b.Q >= 10 * b.tau) == e["resolved"] or abs(b.Q / (10 * b.tau) - 1) < 0.01)


def test_eta_n_recomputed_independently():
    for N, r, g, tr in ((2, 1.0, (40, 40), "fft"), (3, 0.2, (100, 20), "dense"), (3, 5.0, (20, 100), "fft")):
        rec = harness(r, N, *g, tr, 6)
        run = cr.run_coupled(1.0, r, 1.0, 1.0, *g, N, transform=tr, half_steps=6, research_mode=True)
        for log, h in zip(run.steps, rec):
            A, B, C, F = qa_coefficients(np.asarray(h["given"]), h["given_grid"].h)
            hs = h["solve_grid"].h
            V = np.asarray(h["step"].solved)
            rho = F[None, :] - qa_apply_K(V, hs, A, B, C)
            norm = lambda M: float(np.max(np.sum(np.abs(M), axis=1)))  # noqa: E731
            scale = (16 * norm(A) / hs**4 + 8 * norm(B) / hs**2 + norm(C)) * float(np.max(np.abs(V))) + float(np.max(np.abs(F)))
            eta = float(np.max(np.abs(rho))) / scale
            # both are evaluations of a residual at the rounding level: agree within a few u (eta itself ~ u)
            assert abs(eta - log.eta_N) <= 16 * U and log.eta_N <= 16 * U


@pytest.mark.parametrize("cond", (1.0, 1e3, 1e6))
def test_joint_halfstep_with_non_orthogonal_given_set(cond):
    """One coupled half-step with a given set X0 T (cross terms, MATH_SPEC ВБ-24; no QR before it): Builder field against
    the QA least-squares field and the field of X0 (equal in exact arithmetic, span invariance). Tolerance: the same
    first-order model as above for one half-step; the modal normal equations carry cond(A) = cond(T)^2, so the Builder
    error grows quadratically with cond(T) (report, 3.3) - the working cycle avoids it by the QR of D-046 RV4-4."""
    from kkvim.coupled_rank import mode_conditions, rank_diagnostics
    rng = np.random.default_rng(5)
    Q1, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    Q2, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    T = Q1 @ np.diag(np.geomspace(1, 1 / cond, 3)) @ Q2
    nx, ny, r = 36, 24, 1 / 3
    gx, gy = Grid1D(1.0, nx), Grid1D(r, ny)
    G = odd_sine_set(gx, 3) @ T
    step = solve_coupled_halfstep(gx, G, gy, 1.0, 1.0, direction="Y", transform="fft")
    Wb = field(G, step.solved)
    qa = run_control(SpectralGrid(1.0, r, nx, ny), 3, 1, start=G)[0]
    W0 = run_control(SpectralGrid(1.0, r, nx, ny), 3, 1)[0].W
    cond_M, _ = mode_conditions(step.coefficients, gy)
    kappa = rank_diagnostics(gx, G, gy, step.solved).kappa_rep
    eps = U * (cond_M * (1 + (2 * nx / math.pi) ** 2) + qa.ls_cond_max**2 + math.log2(ny) + kappa * math.sqrt(3))
    assert np.max(np.abs(Wb - qa.W)) <= eps * np.max(np.abs(qa.W))
    assert np.max(np.abs(Wb - W0)) <= eps * np.max(np.abs(W0))


def test_qr_keeps_field_and_energy():
    for N, r, g, tr in ((2, 1.0, (40, 40), "fft"), (3, 0.2, (100, 20), "dense"), (3, 5.0, (20, 100), "fft")):
        run = cr.run_coupled(1.0, r, 1.0, 1.0, *g, N, transform=tr, half_steps=10, research_mode=True)
        hx, hy = 1.0 / g[0], r / g[1]
        for log in run.steps[:-1]:
            assert log.qr_done and log.qr_field_error <= 8 * U * math.sqrt(N) * log.rank.kappa_rep * N
            # |Pi(w') - Pi(w)| <= (|w|_a + |omega_h|_a) |w' - w|_a, |w' - w|_a <= |Delta_h|_2 sqrt(ab) max|w' - w|
            lap = 4 / hx**2 + 4 / hy**2
            dmax = log.qr_field_error * log.w_max
            bound = 2 * log.energy.norm_a * lap * math.sqrt(r) * dmax + 2 * log.energy.R
            assert abs(log.P_after_qr - log.energy.P) <= bound
        assert not run.steps[-1].qr_done


def test_controller_semantics_and_statuses():
    # S-1 before the first half-step (ChV4-8, ChV4-9); N, n not changed
    run = cr.run_coupled(1.0, 1.0, 1.0, 1.0, 5, 40, 3, transform="fft", half_steps=4, research_mode=True)
    assert run.status == cr.STOPPED_RANK_LOSS_PROVEN and run.s == 0
    # normative mode: no QR, NOT_QUALIFIED after the diagnostics of s = 1; one half-step: HALF_STEP_LIMIT_REACHED
    run = cr.run_coupled(1.0, 1.0, 1.0, 1.0, 40, 40, 2, transform="dense", half_steps=5)
    assert run.status == cr.NOT_QUALIFIED and run.s == 1 and not run.steps[0].qr_done
    run = cr.run_coupled(1.0, 1.0, 1.0, 1.0, 40, 40, 2, transform="dense", half_steps=1)
    assert run.status == cr.HALF_STEP_LIMIT_REACHED and run.s == 1
    # N = 1 stays on the R2/R3 path; N = 4 rejected (D-047)
    for N in (1, 4):
        assert cr.run_coupled(1.0, 1.0, 1.0, 1.0, 40, 40, N, transform="fft", half_steps=2).status == "ERROR_INPUT"
    # S-2 is never assigned; every log record is PROVISIONAL / NOT_QUALIFIED; never CONVERGED
    run = cr.run_coupled(1.0, 2.0, 1.0, 1.0, 20, 40, 3, transform="fft", half_steps=8, research_mode=True)
    assert run.status != cr.STOPPED_RANK_NUMERICALLY_UNRESOLVED and not run.converged
    assert all(log.mark == cr.MARK for log in run.steps) and run.x.shape[1] == run.y.shape[1] == 3


def test_dependent_given_set_is_an_explicit_error_not_a_repair():
    gx, gy = Grid1D(1.0, 20), Grid1D(1.0, 20)
    X = odd_sine_set(gx, 2)
    G = np.column_stack([X[:, 0], 2.0 * X[:, 0]])  # exactly dependent (not equal up to sign)
    try:
        step = solve_coupled_halfstep(gx, G, gy, 1.0, 1.0, direction="Y", transform="fft")
    except Exception as exc:  # explicit error: the set is not repaired, N is not reduced
        assert getattr(exc, "code", None) in ("ERROR_LINEAR_SOLVE", "ERROR_NONFINITE", "ERROR_DEGENERATE")
    else:
        assert step.solved.shape[1] == 2  # no column dropped; the result is then outside the model (no regularization)


def test_guard_failure_on_the_last_half_step_is_reported(monkeypatch):
    """No natural guard failure exists for N = 2-3 on the scanned grids (report, 4.2); a stricter K is substituted IN
    THE TEST ONLY to make rho_N <= K u kappa_rep on the last half-step. Required semantics: a failed guard must not be
    reported as RESEARCH_LIMIT_REACHED (it is not a passed guard)."""
    monkeypatch.setattr(cr, "K_RESEARCH", 10**13)
    run = cr.run_coupled(1.0, 1.0, 1.0, 1.0, 40, 40, 3, transform="fft", half_steps=1, research_mode=True)
    log = run.steps[-1]
    assert log.guards is not None and not log.guards.rank_ok  # the guard fails on the last half-step
    if run.status == cr.RESEARCH_LIMIT_REACHED:
        pytest.xfail("DEFECT C-1 (report 4.2): s == half_steps is checked before the guards; RESEARCH_LIMIT_REACHED "
                     "is returned although rank_ok is False")
    assert run.status in (cr.PROVISIONAL_RANK_GUARD_STOP, cr.PROVISIONAL_SET_GUARD_STOP)


def test_negative_energy_changes_are_unresolved_and_within_the_rounding_model():
    """Delta P_s < 0 (QA-delta4, report 5.2): only UNRESOLVED steps, |Delta P_s| <= R(w_{s-1}) + R(w_s) (the R2 model of
    the evaluation of the two energies) - i.e. consistent with rounding, not only labelled UNRESOLVED."""
    seen = 0
    for N, r, g, tr in ((2, 0.2, (40, 40), "fft"), (3, 1.0, (40, 40), "dense"), (3, 5.0, (20, 100), "fft"),
                        (2, 1 / 3, (60, 20), "dense")):
        run = cr.run_coupled(1.0, r, 1.0, 1.0, *g, N, transform=tr, half_steps=40, research_mode=True)
        for log in run.steps[1:]:
            e = log.energy
            if e.delta_P < 0:
                seen += 1
                assert e.status == "UNRESOLVED" and e.e1
                assert -e.delta_P <= e.R_prev + e.R
    assert seen > 0


def test_operational_statuses_are_not_qualification(monkeypatch):
    """PROVISIONAL_ENERGY_E1_STOP: no QR, N kept, the step logged; HALF_STEP_LIMIT_REACHED: normative mode, one half-step.
    Neither is a scientific PASS (every record PROVISIONAL / NOT_QUALIFIED)."""
    real = cr.pair_energy

    def violating(*args, **kwargs):
        e = real(*args, **kwargs)
        return type(e)(**{**e.__dict__, "status": cr.E1_VIOLATED, "e1": False})

    monkeypatch.setattr(cr, "pair_energy", violating)
    run = cr.run_coupled(1.0, 1.0, 1.0, 1.0, 40, 40, 2, transform="fft", half_steps=5, research_mode=True)
    assert run.status == cr.PROVISIONAL_ENERGY_E1_STOP and run.s == 2
    assert not run.steps[-1].qr_done and run.x.shape[1] == run.y.shape[1] == 2 and run.steps[-1].mark == cr.MARK
    monkeypatch.setattr(cr, "pair_energy", real)
    run = cr.run_coupled(1.0, 1.0, 1.0, 1.0, 40, 40, 2, transform="fft", half_steps=1)
    assert run.status == cr.HALF_STEP_LIMIT_REACHED and not run.converged and run.steps[0].mark == cr.MARK


@pytest.mark.parametrize("N", (2, 3))
def test_high_precision_accumulated_field_error(N):
    """QA-delta1 (temporary mpmath): Builder fields s = 1..6 against the exact discrete sequence (60 digits, normal
    equations of every mode, raw representation) on a small grid; within the first-order model above."""
    mp = pytest.importorskip("mpmath")
    mp.mp.dps = 60
    nx = ny = 12
    run = cr.run_coupled(1.0, 1.0, 1.0, 1.0, nx, ny, N, transform="fft", half_steps=6, research_mode=True)
    rec = harness(1.0, N, nx, ny, "fft", 6)
    qa = run_control(SpectralGrid(1.0, 1.0, nx, ny), N, 6)
    P = mp.matrix([[mp.sqrt(mp.mpf(2) / nx) * (0 if (i * k) % nx == 0 else mp.sin(mp.pi * i * k / nx)) for k in range(1, nx)]
                   for i in range(1, nx)])
    mu = [4 * nx**2 * mp.sin(k * mp.pi / (2 * nx)) ** 2 for k in range(1, nx)]
    c = [mp.fsum(P[i, k] for i in range(nx - 1)) for k in range(nx - 1)]
    lam = [[(mu[k] + mu[l]) ** 2 for l in range(nx - 1)] for k in range(nx - 1)]
    ws = [[c[k] * c[l] / lam[k][l] for l in range(nx - 1)] for k in range(nx - 1)]

    def step(G, transpose):
        Gh = P * G
        V = mp.matrix(nx - 1, N)
        for m in range(nx - 1):
            A, rhs = mp.matrix(N, N), mp.matrix(N, 1)
            for t in range(nx - 1):
                w = lam[m][t] if transpose else lam[t][m]
                tg = ws[m][t] if transpose else ws[t][m]
                for i in range(N):
                    rhs[i] += w * tg * Gh[t, i]
                    for j in range(N):
                        A[i, j] += w * Gh[t, i] * Gh[t, j]
            v = mp.lu_solve(A, rhs)
            for i in range(N):
                V[m, i] = v[i]
        return P * V

    X = mp.matrix([[mp.sin((2 * j + 1) * mp.pi * i / nx) for j in range(N)] for i in range(1, nx)])
    Y = None
    for s in range(1, 7):
        if s % 2:
            Y = step(X, False)
        else:
            X = step(Y, True)
        W = X * Y.T
        We = np.array([[float(W[i, j]) for j in range(nx - 1)] for i in range(nx - 1)])
        Wb = field(*rec[s - 1]["new"])
        eps = _model(run.steps, qa, s, nx, ny)
        assert np.max(np.abs(Wb - We)) <= eps * np.max(np.abs(We)), s


# --- direct checks of the A4 candidates (preliminary matrix, report section 6) ---------------------------------------

def test_a4_2a_symmetric_assembly_and_a4_3_closed_forms_at_s1():
    for N, r, (nx, ny) in ((2, 1.0, (40, 40)), (3, 0.2, (100, 20)), (3, 5.0, (20, 100))):
        rec = harness(r, N, nx, ny, "fft", 4)
        for h in rec:
            co = h["step"].coefficients
            for M in (co.A, co.B, co.C):
                assert np.array_equal(M, M.T)  # A4-2(a): exact symmetry
        co = rec[0]["step"].coefficients  # s = 1, odd sines on G(1, nx) (ChV4-8)
        h = 1.0 / nx
        m = np.arange(1, 2 * N, 2)
        mu = 4 / h**2 * np.sin(m * math.pi / (2 * nx)) ** 2
        exact = (0.5 * np.eye(N), 0.5 * np.diag(mu), 0.5 * np.diag(mu**2), h / np.tan(m * math.pi / (2 * nx)))
        for Mb, Me in zip((co.A, co.B, co.C, co.F), exact):
            assert np.max(np.abs(Mb - Me)) <= 1e-13 * np.max(np.abs(Me))  # candidate of A4-3 (NOT_QUALIFIED)


@pytest.mark.parametrize("N, r, grid", [(2, 1.0, (16, 16)), (3, 0.2, (40, 8)), (3, 5.0, (12, 40))])
def test_a4_5_modal_solution_equals_the_assembled_block_system(N, r, grid):
    """Variant B (modal, ChF-3) against variant A (block matrix ChF-1 assembled and solved directly) on small grids;
    tolerance u cond(K) (n sizes) - a first-order model."""
    nx, ny = grid
    rec = harness(r, N, nx, ny, "fft", 3)
    for h in rec:
        co, n, hs = h["step"].coefficients, h["solve_grid"].n, h["solve_grid"].h
        L = (np.diag(np.full(n - 2, 1.0), -1) - 2 * np.eye(n - 1) + np.diag(np.full(n - 2, 1.0), 1)) / hs**2
        K = np.kron(co.A, L @ L) - 2 * np.kron(co.B, L) + np.kron(co.C, np.eye(n - 1))
        V = np.linalg.solve(K, np.kron(co.F, np.ones(n - 1))).reshape(N, n - 1).T
        cond = np.linalg.cond(K)
        assert np.max(np.abs(V - h["step"].solved)) <= 4 * n * U * cond * np.max(np.abs(V))


def test_a4_6_cross_terms_without_qr():
    """U = (s1, s3) against U' = (s1, s1 + s3) with non-diagonal coefficients, three half-steps WITHOUT QR (NB-6):
    same fields, energies and rho_N within the first-order model of one step per half-step (cond(U') is small)."""
    nx, ny, r = 24, 48, 2.0
    gx, gy = Grid1D(1.0, nx), Grid1D(r, ny)
    X = odd_sine_set(gx, 2)
    out = []
    for X0 in (X, np.column_stack([X[:, 0], X[:, 0] + X[:, 1]])):
        Xc, Yc, fields = X0, None, []
        for s in range(1, 4):
            if s % 2:
                Yc = solve_coupled_halfstep(gx, Xc, gy, 1.0, 1.0, direction="Y", transform="fft").solved
            else:
                Xc = solve_coupled_halfstep(gy, Yc, gx, 1.0, 1.0, direction="X", transform="fft").solved
            fields.append(field(Xc, Yc))
        out.append(fields)
    for Wa, Wb in zip(*out):
        assert np.max(np.abs(Wa - Wb)) <= 1e-12 * np.max(np.abs(Wa))
        assert abs(nodal_energy(Wa, 1 / nx, r / ny) - nodal_energy(Wb, 1 / nx, r / ny)) <= 1e-13 * abs(nodal_energy(Wa, 1 / nx, r / ny))


@pytest.mark.parametrize("N, r", [(2, 0.2), (2, 1.0), (3, 1.0), (3, 5.0)])
def test_a4_7_first_halfstep_against_the_navier_m_partial_sum(N, r):
    """s = 1 of the Builder kernel against the continuous x-oriented partial sum (Navier-m, odd M, strict tail; QA
    reference NOT_QUALIFIED as normative): observed order p on the pairs 80-160-320 (candidate window [1.9, 2.1])."""
    from kkvim_reference.r4_levy_partial import navier_m_truncated_at_nodes
    modes = list(range(1, 2 * N, 2))
    errs = []
    for n in (80, 160, 320):
        W = field(*harness(r, N, n, n, "fft", 1)[0]["new"])
        ref = navier_m_truncated_at_nodes(1.0, r, 1.0, 1.0, modes, n, n, 20001)[1:-1, 1:-1]
        errs.append(np.max(np.abs(W - ref)) / np.max(np.abs(ref)))
    for e1, e2 in zip(errs, errs[1:]):
        assert 1.9 <= math.log2(e1 / e2) <= 2.1, errs


def test_a4_12_reflections_and_scaling():
    run = cr.run_coupled(1.0, 3.0, 1.0, 1.0, 40, 120, 3, transform="fft", half_steps=10, research_mode=True)
    W = run.field()
    m = np.max(np.abs(W))
    assert np.max(np.abs(W - W[::-1, :])) <= 1e-12 * m and np.max(np.abs(W - W[:, ::-1])) <= 1e-12 * m
    # w_bar = w D / (q a^4) does not depend on q, D and on a common scale of a, b (ChV-14)
    base = run.field() * 1.0 / (1.0 * 1.0**4)
    for a, q, D in ((2.9, 1.0, 1.0), (1.0, 3.7, 1.0), (1.0, 1.0, 0.013), (2.9, -3.7, 0.013)):
        other = cr.run_coupled(a, 3.0 * a, D, q, 40, 120, 3, transform="fft", half_steps=10, research_mode=True)
        assert np.max(np.abs(other.field() * D / (q * a**4) - base)) <= 1e-11 * m
