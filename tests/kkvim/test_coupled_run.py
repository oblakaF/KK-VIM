"""Own tests of kkvim.coupled_run: research controller of coupled half-steps (R4, step 16).

R4_NUMERICAL_SPEC 5.4.1 (order), 5.5 (stops, variant A), 6 (energy), 7 (no convergence
status); D-047 addendum 1 (research mode, K_RESEARCH = 1e6, PROVISIONAL / NOT_QUALIFIED).
Every number here is PROVISIONAL / NOT_QUALIFIED. Builder unit tests, not QA.
"""

import dataclasses
import math

import numpy as np
import pytest

import kkvim.coupled_run as cr
from kkvim.coefficients import ERROR_DEGENERATE, ERROR_INPUT
from kkvim.coupled import CoupledHalfStepError, coupled_first_halfstep, coupled_second_halfstep
from kkvim.coupled_energy import E1_VIOLATED
from kkvim.coupled_run import (
    HALF_STEP_LIMIT_REACHED,
    K_RESEARCH,
    MARK,
    NOT_QUALIFIED,
    PROVISIONAL_ENERGY_E1_STOP,
    PROVISIONAL_RANK_GUARD_STOP,
    PROVISIONAL_SET_GUARD_STOP,
    MAX_HALF_STEPS,
    PROVISIONAL_FIELD_STABILIZED,
    RESEARCH_LIMIT_REACHED,
    STOPPED_RANK_LOSS_PROVEN,
    STOPPED_RANK_NUMERICALLY_UNRESOLVED,
    run_coupled,
)
from kkvim.linear_solver import ERROR_LINEAR_SOLVE

U = 2.0**-53
D, Q = 3.1, 2.3


def gamma(n):
    return n * U / (1 - n * U)


def check_run(run, half_steps, research_mode=True):
    assert run.mark == MARK and all(step.mark == MARK for step in run.steps)
    assert run.converged is False and run.status != "CONVERGED"
    assert run.status != STOPPED_RANK_NUMERICALLY_UNRESOLVED  # S-2 is never assigned from the research K
    for i, step in enumerate(run.steps, start=1):
        assert (step.s, step.k, step.direction) == (i, i // 2, "Y" if i % 2 else "X")
    if run.status in (RESEARCH_LIMIT_REACHED, PROVISIONAL_FIELD_STABILIZED, MAX_HALF_STEPS):
        if run.status == RESEARCH_LIMIT_REACHED:
            assert run.s == half_steps < cr.S_MAX
        elif run.status == MAX_HALF_STEPS:
            assert run.s == cr.S_MAX <= half_steps
        else:  # D-048.3/D-048.4: criterion at s - 1 and s, s >= s_min, guards passed, first such s
            assert cr.S_MIN <= run.s <= half_steps
            assert run.steps[-1].stabilization_ok and run.steps[-2].stabilization_ok
        if run.parameters.research_mode and run.status != PROVISIONAL_FIELD_STABILIZED:
            oks = [st.stabilization_ok for st in run.steps]
            assert not any(a and b for a, b in zip(oks[cr.S_MIN - 2:], oks[cr.S_MIN - 1:]))
        if run.status == PROVISIONAL_FIELD_STABILIZED:
            oks = [st.stabilization_ok for st in run.steps[:-1]]
            assert not any(a and b for a, b in zip(oks[cr.S_MIN - 2:], oks[cr.S_MIN - 1:]))  # not earlier
        assert all(step.guards.passed for step in run.steps)  # also on the last half-step (C-1, D-048.4)
        assert all(step.qr_done for step in run.steps[:-1]) and not run.steps[-1].qr_done
        for step in run.steps[:-1]:
            assert step.guards.passed
            assert step.rank.rho_N > K_RESEARCH * U * step.rank.kappa_rep
            assert step.rank.set_sigma_min > K_RESEARCH * U
        p = run.parameters
        for step in run.steps[:-1]:
            # Pi_h of w_s from the transformed arrays (after QR) against w_s before QR: the same field up to
            # |P(w + e) - P(w)| <= B_omega |e|_a + |e|_a^2 / 2 + R-terms, |e|_a <= D^1/2 (4/h_x^2 + 4/h_y^2) sqrt(ab) max|e|
            hx, hy = p.a / p.n_x, p.b / p.n_y
            e_max = 64 * gamma(p.n_x + p.n_y + p.N) * step.rank.kappa_rep * step.w_max  # as in test_qr_keeps_field
            e_a = math.sqrt(p.D) * (4 / hx**2 + 4 / hy**2) * math.sqrt(p.a * p.b) * e_max
            assert abs(step.P_after_qr - step.energy.P) <= step.energy.B_omega * e_a + 0.5 * e_a**2 + 2 * step.energy.R
        for prev, step in zip(run.steps, run.steps[1:]):
            assert step.energy.P_prev == prev.P_after_qr  # P_{s-1}^QR from the same transformed arrays (6.2)
            assert step.energy.status != E1_VIOLATED
            assert step.energy.P <= step.energy.P_prev + step.energy.tau  # energy does not increase beyond the model


def test_constants_of_the_research_mode():
    assert K_RESEARCH == 1_000_000 and MARK == "PROVISIONAL / NOT_QUALIFIED"


def test_normative_mode_stops_before_any_qr(monkeypatch):
    monkeypatch.setattr(cr, "orthogonalize_received", lambda *a, **k: pytest.fail("QR in the normative mode"))
    run = run_coupled(1.0, 2.0, D, Q, 20, 40, 2, transform="dense", half_steps=5)
    assert run.status == NOT_QUALIFIED and run.s == 1 and run.steps[0].guards is None and not run.steps[0].qr_done
    one = run_coupled(1.0, 2.0, D, Q, 20, 40, 2, transform="dense", half_steps=1)
    assert one.status == HALF_STEP_LIMIT_REACHED and one.s == 1
    check_run(run, 5, research_mode=False)


@pytest.mark.parametrize("N, n_x, n_y", [(3, 5, 40), (3, 20, 5), (3, 4, 4)])
def test_start_conditions_give_s1_before_the_first_half_step(N, n_x, n_y, monkeypatch):
    monkeypatch.setattr(cr, "solve_coupled_halfstep", lambda *a, **k: pytest.fail("solved"))
    run = run_coupled(1.0, 2.0, D, Q, n_x, n_y, N, transform="dense", half_steps=4, research_mode=True)
    assert run.status == STOPPED_RANK_LOSS_PROVEN and run.s == 0 and run.field() is None


@pytest.mark.parametrize("bad", [dict(N=1), dict(N=4), dict(b=6.0), dict(D=-1.0)])
def test_invalid_configuration_is_error_input(bad):
    args = dict(a=1.0, b=2.0, D=D, q=Q, n_x=20, n_y=40, N=2) | bad
    run = run_coupled(**args, transform="dense", half_steps=3, research_mode=True)
    assert run.status == ERROR_INPUT and run.s == 0  # N = 1 stays on the R2/R3 path


RESEARCH_CASES = [
    (1.0, 2.0, 20, 40, 2, 40), (1.0, 2.0, 20, 40, 3, 40), (1.0, 1.0, 32, 32, 3, 20), (1.0, 0.2, 100, 20, 3, 20),
    (1.0, 5.0, 16, 80, 2, 10), (1.0, 1 / 3, 96, 32, 3, 4), (2.0, 3.0, 24, 36, 2, 3), (1.0, 1.5, 40, 60, 3, 10),
]


@pytest.mark.parametrize("transform", ["dense", "fft"])
@pytest.mark.parametrize("a, b, n_x, n_y, N, s", RESEARCH_CASES)
def test_research_runs_reach_the_limit_or_stop_honestly(a, b, n_x, n_y, N, s, transform):
    run = run_coupled(a, b, D, Q, n_x, n_y, N, transform=transform, half_steps=s, research_mode=True)
    check_run(run, s)
    assert run.status in (RESEARCH_LIMIT_REACHED, PROVISIONAL_FIELD_STABILIZED, PROVISIONAL_RANK_GUARD_STOP,
                          PROVISIONAL_SET_GUARD_STOP)
    for step in run.steps:
        if step.qr_done:
            assert step.qr_field_error <= 64 * gamma(n_x + n_y + N) * step.rank.kappa_rep


@pytest.mark.parametrize("a, b, n_x, n_y, N, s", [(1.0, 2.0, 20, 40, 3, 10), (1.0, 0.2, 100, 20, 2, 6)])
def test_dense_and_fft_give_the_same_sequence(a, b, n_x, n_y, N, s):
    dense = run_coupled(a, b, D, Q, n_x, n_y, N, transform="dense", half_steps=s, research_mode=True)
    fast = run_coupled(a, b, D, Q, n_x, n_y, N, transform="fft", half_steps=s, research_mode=True)
    assert dense.status == fast.status and dense.s == fast.s
    assert dense.status in (RESEARCH_LIMIT_REACHED, PROVISIONAL_FIELD_STABILIZED)
    w_d, w_f = dense.field(), fast.field()
    cond = max(step.cond_M_max for step in dense.steps)
    assert np.max(np.abs(w_d - w_f)) <= 64 * s * gamma(n_x + n_y + N) * cond * np.max(np.abs(w_d))


def test_qr_does_not_change_the_field_of_the_second_half_step():
    """Fixed-count kernel path (no QR) against the research controller (QR after s = 1): same field w2."""
    run = run_coupled(1.0, 2.0, D, Q, 20, 40, 3, transform="dense", half_steps=2, research_mode=True)
    first = coupled_first_halfstep(1.0, 2.0, D, Q, 20, 40, 3, transform="dense")
    second = coupled_second_halfstep(first, transform="dense")
    plain = second.field()
    cond_raw = np.linalg.cond(second.step.coefficients.A)  # unnormalized Y set: poor representation
    assert np.max(np.abs(run.field() - plain)) <= 64 * gamma(20 + 40 + 3) * cond_raw * np.max(np.abs(plain))


def test_order_diagnostics_guards_then_qr(monkeypatch):
    calls = []
    real_rank, real_qr = cr.rank_diagnostics, cr.orthogonalize_received
    monkeypatch.setattr(cr, "rank_diagnostics", lambda *a: calls.append("diag") or real_rank(*a))
    monkeypatch.setattr(cr, "orthogonalize_received", lambda *a: calls.append("qr") or real_qr(*a))
    real_solve = cr.solve_coupled_halfstep
    monkeypatch.setattr(cr, "solve_coupled_halfstep", lambda *a, **k: calls.append("solve") or real_solve(*a, **k))
    run = run_coupled(1.0, 2.0, D, Q, 20, 40, 2, transform="dense", half_steps=3, research_mode=True)
    assert run.status == RESEARCH_LIMIT_REACHED
    assert calls == ["solve", "diag", "qr", "solve", "diag", "qr", "solve", "diag"]


@pytest.mark.parametrize("field, value, status", [
    ("rho_N", 1e-12, PROVISIONAL_RANK_GUARD_STOP),
    ("kappa_rep", 1e11, PROVISIONAL_RANK_GUARD_STOP),  # K u kappa_rep >= 1: guard not applicable
    ("set_sigma", np.array([1.0, 1e-12]), PROVISIONAL_SET_GUARD_STOP),
])
def test_guard_failures_stop_before_qr_and_keep_n(field, value, status, monkeypatch):
    real = cr.rank_diagnostics

    def forced(*args):
        r = real(*args)
        return dataclasses.replace(r, **{field: value}) if len(calls) == 1 else r

    calls = []
    monkeypatch.setattr(cr, "rank_diagnostics", lambda *a: calls.append(1) or forced(*a))
    monkeypatch.setattr(cr, "orthogonalize_received", lambda *a: pytest.fail("QR after a failed guard"))
    run = run_coupled(1.0, 2.0, D, Q, 20, 40, 2, transform="fft", half_steps=10, research_mode=True)
    assert run.status == status and run.s == 1 and not run.steps[0].qr_done
    if field == "kappa_rep":
        assert run.steps[0].guards.applicable is False and "K u kappa_rep >= 1" in run.reason
    assert run.x.shape[1] == 2 and run.y.shape[1] == 2  # N kept, no restored direction
    check_run(run, 10)


def test_energy_e1_violation_stops_and_is_not_hidden(monkeypatch):
    real = cr.pair_energy
    monkeypatch.setattr(cr, "pair_energy", lambda *a, **k: dataclasses.replace(real(*a, **k), status=E1_VIOLATED, e1=False))
    run = run_coupled(1.0, 2.0, D, Q, 20, 40, 2, transform="dense", half_steps=5, research_mode=True)
    assert run.status == PROVISIONAL_ENERGY_E1_STOP and run.s == 2 and run.steps[-1].energy.status == E1_VIOLATED


def test_nonnegative_energy_is_error_degenerate(monkeypatch):
    real = cr.first_energy
    monkeypatch.setattr(cr, "first_energy", lambda *a: dataclasses.replace(real(*a), P=0.0))
    run = run_coupled(1.0, 2.0, D, Q, 20, 40, 2, transform="dense", half_steps=3, research_mode=True)
    assert run.status == ERROR_DEGENERATE and run.s == 1


@pytest.mark.parametrize("code, expected", [(ERROR_LINEAR_SOLVE, ERROR_LINEAR_SOLVE), (ERROR_DEGENERATE, STOPPED_RANK_LOSS_PROVEN)])
def test_solver_errors_keep_completed_half_steps(code, expected, monkeypatch):
    real = cr.solve_coupled_halfstep

    def failing(*args, **kwargs):
        if kwargs["direction"] == "X":
            raise CoupledHalfStepError(code, "forced", direction="X", mode=1, N=2, n=20)
        return real(*args, **kwargs)

    monkeypatch.setattr(cr, "solve_coupled_halfstep", failing)
    run = run_coupled(1.0, 2.0, D, Q, 20, 40, 2, transform="dense", half_steps=5, research_mode=True)
    assert run.status == expected and run.s == 1 and run.steps[0].qr_done and run.field() is not None


def test_field_stabilizes_but_no_convergence_status():
    run = run_coupled(1.0, 2.0, D, Q, 20, 40, 2, transform="fft", half_steps=20, research_mode=True)
    assert run.status == PROVISIONAL_FIELD_STABILIZED and not run.converged and run.mark == MARK
    assert run.steps[-1].eps_inf <= cr.TOL_INF and run.steps[-1].eps_E <= cr.TOL_E and not run.steps[-1].qr_done
    assert all(math.isfinite(step.eta_N) for step in run.steps)
    check_run(run, 20)


# QA C-1 (f3f15bf): the research guards decide the status also on the last half-step --------

def forced_on_call(monkeypatch, call, **change):
    """Replace fields of the rank diagnostics of the given call (1-based) only."""
    real, calls = cr.rank_diagnostics, []

    def forced(*args):
        calls.append(1)
        r = real(*args)
        return dataclasses.replace(r, **change) if len(calls) == call else r

    monkeypatch.setattr(cr, "rank_diagnostics", forced)


def qr_counter(monkeypatch):
    real, count = cr.orthogonalize_received, []
    monkeypatch.setattr(cr, "orthogonalize_received", lambda *a: count.append(1) or real(*a))
    return count


def test_c1_rank_guard_failure_on_the_only_half_step(monkeypatch):
    """As the QA reproduction: N = 3, (40, 40), half_steps = 1, temporary K (production K is unchanged)."""
    monkeypatch.setattr(cr, "K_RESEARCH", 10**13)
    count = qr_counter(monkeypatch)
    run = run_coupled(1.0, 1.0, D, Q, 40, 40, 3, transform="dense", half_steps=1, research_mode=True)
    assert run.status == PROVISIONAL_RANK_GUARD_STOP and run.s == 1 and count == []
    assert run.steps[0].guards.rank_ok is False and run.steps[0].stop_reason == PROVISIONAL_RANK_GUARD_STOP
    assert run.status != STOPPED_RANK_NUMERICALLY_UNRESOLVED


@pytest.mark.parametrize("change, status", [
    (dict(rho_N=1e-12), PROVISIONAL_RANK_GUARD_STOP),
    (dict(set_sigma=np.array([1.0, 1e-12])), PROVISIONAL_SET_GUARD_STOP),
])
def test_c1_guard_failure_on_the_last_half_step_after_qr_steps(change, status, monkeypatch):
    forced_on_call(monkeypatch, 4, **change)
    count = qr_counter(monkeypatch)
    run = run_coupled(1.0, 2.0, D, Q, 20, 40, 2, transform="fft", half_steps=4, research_mode=True)
    assert run.status == status and run.s == 4 and len(count) == 3  # QR after s = 1, 2, 3 only
    last = run.steps[-1]
    assert last.stop_reason == status and not last.qr_done and not last.guards.passed and last.mark == MARK
    # the field and the log of the stop are those of the completed half-step 4, as in an unforced run
    monkeypatch.undo()
    plain = run_coupled(1.0, 2.0, D, Q, 20, 40, 2, transform="fft", half_steps=4, research_mode=True)
    assert plain.status == RESEARCH_LIMIT_REACHED
    assert np.array_equal(run.field(), plain.field())
    assert [s.energy.P for s in run.steps] == [s.energy.P for s in plain.steps]


def test_c1_limit_is_reached_only_with_passed_guards_and_without_a_last_qr(monkeypatch):
    count = qr_counter(monkeypatch)
    run = run_coupled(1.0, 2.0, D, Q, 20, 40, 3, transform="dense", half_steps=5, research_mode=True)
    assert run.status == RESEARCH_LIMIT_REACHED and len(count) == 4
    assert all(step.guards.passed for step in run.steps) and not run.steps[-1].qr_done
    assert run.steps[-1].stop_reason == RESEARCH_LIMIT_REACHED


def test_c1_normative_mode_is_unchanged(monkeypatch):
    monkeypatch.setattr(cr, "K_RESEARCH", 10**13)  # no effect: the normative mode applies no research guard
    monkeypatch.setattr(cr, "orthogonalize_received", lambda *a: pytest.fail("QR in the normative mode"))
    one = run_coupled(1.0, 1.0, D, Q, 40, 40, 3, transform="dense", half_steps=1)
    more = run_coupled(1.0, 1.0, D, Q, 40, 40, 3, transform="dense", half_steps=3)
    assert (one.status, one.s, one.steps[0].guards) == (HALF_STEP_LIMIT_REACHED, 1, None)
    assert (more.status, more.s, more.steps[0].guards) == (NOT_QUALIFIED, 1, None)
