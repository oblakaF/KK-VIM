"""Independent QA qualification of the D-048 implementation (Builder 3da624a) - CI part of K-1...K-9.

Plan fixed before the measurements: docs/R4_D048_QUALIFICATION.plan.md (e4e1109). Builder is only the object under test;
reference values come from kkvim_reference, the QA code of tests/cross/test_r4_controller_comparison.py, stored QA values
or decimal arithmetic (no mpmath here). The full samples and the high-precision parts are produced by
scripts/r4_d048_qualification.py (results in docs/results/r4_d048_qualification/). No tolerance is taken from observations.
"""

import dataclasses
import hashlib
import importlib.util
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

import kkvim.coupled as kc
import kkvim.coupled_run as cr
from kkvim.grid import Grid1D
from kkvim_reference.r4_spectral_control import SpectralGrid, run_control

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "docs" / "results" / "r4_d048_qualification"
U = 2.0**-53


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


q = _load("r4_d048_qualification", ROOT / "scripts" / "r4_d048_qualification.py")
cmp = _load("r4cmp_for_d048", Path(__file__).with_name("test_r4_controller_comparison.py"))
pq = _load("r4pq_for_d048", ROOT / "scripts" / "r4_numerical_prequal.py")


# --- K-1: across platforms, against stored QA values, in the doubled first-order field model ---------------------

def test_k1_fields_across_platforms_within_the_field_model():
    ref = np.load(RESULTS / "k1_reference_fields_n40.npy")
    assert ref.shape[0] == len(q.K1_REFERENCE_KEYS)
    controls = {}
    for (N, lab, s), stored in zip(q.K1_REFERENCE_KEYS, ref):
        r = q.R3[lab]
        qa = controls.setdefault((N, lab), run_control(SpectralGrid(1.0, r, 40, 40), N, 10))
        run = cr.run_coupled(1.0, r, 1.0, 1.0, 40, 40, N, transform="fft", half_steps=s, research_mode=True,
                             stop_on_stabilization=False)
        assert len(run.steps) == s
        W = run.x @ run.y.T
        eps = 2 * cmp._model(run.steps, qa, s, 40, 40)
        assert np.max(np.abs(W - stored)) <= eps * np.max(np.abs(stored)), (N, lab, s)


# --- K-2: stopping rule details (the sample itself: k2_stopping.csv) ----------------------------------------------

def test_k2_floor_matches_the_formula_and_error_input_on_both_sides_of_the_boundary():
    for N in (2, 3):
        for n in (40, 320, 1280):
            assert abs(cr.eps_floor(n, n, N) - q.eps_floor_formula(n, n, N)) <= 4 * U * q.eps_floor_formula(n, n, N)
    N = 3
    ok = [n for n in range(2200, 2340, 2) if cr.TOL_E >= 10 * q.eps_floor_formula(40, n, N)]
    n_lo, n_hi = max(ok), max(ok) + 2  # largest admissible and smallest inadmissible even n (n* about 2265)
    assert cr.run_coupled(1.0, 1.0, 1.0, 1.0, 40, n_hi, N, transform="fft", half_steps=1, research_mode=True).status \
        == "ERROR_INPUT"
    run = cr.run_coupled(1.0, 1.0, 1.0, 1.0, 40, n_lo, N, transform="fft", half_steps=1, research_mode=True)
    assert run.status == cr.RESEARCH_LIMIT_REACHED and len(run.steps) == 1


def test_k2_s_min_and_s_max_and_the_normative_mode(monkeypatch):
    with monkeypatch.context() as m:  # criterion satisfied already at s = 2: the stop must come exactly at s = 3
        m.setattr(cr, "TOL_INF", 1.0)
        m.setattr(cr, "TOL_E", 1.0)
        run = cr.run_coupled(1.0, 1.0, 1.0, 1.0, 40, 40, 2, transform="fft", half_steps=40, research_mode=True)
        assert run.status == cr.PROVISIONAL_FIELD_STABILIZED and len(run.steps) == 3
    with monkeypatch.context() as m:  # no stabilization before s_max = 5 (test only)
        m.setattr(cr, "S_MAX", 5)
        run = cr.run_coupled(1.0, 1.0, 1.0, 1.0, 40, 40, 2, transform="fft", half_steps=100, research_mode=True)
        assert run.status == cr.MAX_HALF_STEPS and len(run.steps) == 5 and not run.steps[-1].qr_done
    run = cr.run_coupled(1.0, 1.0, 1.0, 1.0, 40, 40, 2, transform="fft", half_steps=5)
    assert run.status == cr.NOT_QUALIFIED and len(run.steps) == 1 and not run.steps[0].qr_done
    assert run.steps[0].stabilization_ok is None and run.parameters.tol_E is None and run.first_stabilization_s is None
    run = cr.run_coupled(1.0, 1.0, 1.0, 1.0, 40, 40, 2, transform="fft", half_steps=1)
    assert run.status == cr.HALF_STEP_LIMIT_REACHED and len(run.steps) == 1 and not run.steps[0].qr_done
    assert cr.run_coupled(1.0, 1.0, 1.0, 1.0, 40, 40, 2, transform="fft", half_steps=5,
                          stop_on_stabilization=False).status == "ERROR_INPUT"


def test_k2_stabilized_runs_and_both_research_modes_on_a_small_configuration():
    for N in (2, 3):
        auto = cr.run_coupled(1.0, 1.0, 1.0, 1.0, 40, 40, N, transform="fft", half_steps=100, research_mode=True)
        fixed = cr.run_coupled(1.0, 1.0, 1.0, 1.0, 40, 40, N, transform="fft", half_steps=40, research_mode=True,
                               stop_on_stabilization=False)
        s = len(auto.steps)
        assert auto.status == cr.PROVISIONAL_FIELD_STABILIZED and s < cr.S_MAX
        assert all(st.guards.passed for st in auto.steps)
        assert fixed.status == cr.RESEARCH_LIMIT_REACHED and len(fixed.steps) == 40 and fixed.first_stabilization_s == s
        assert all(a.energy.P == b.energy.P and a.eta_N == b.eta_N and a.rank.rho_N == b.rank.rho_N
                   for a, b in zip(auto.steps, fixed.steps[:s]))
        assert all(st.guards.passed for st in fixed.steps[s:])
        assert not any(st.stop_reason == cr.PROVISIONAL_FIELD_STABILIZED for st in fixed.steps)


# --- K-3: mandatory grids, UNRESOLVED behaviour, E1 ---------------------------------------------------------------

def test_k3_mandatory_grid_flags_follow_d048_1_only():
    for r in (0.2, 1 / 3, 0.5, 2 / 3, 1.0, 1.4, 1.5, 2.0, 3.0, 5.0):
        for n in (160, 640, 1280):
            assert cr.is_mandatory_s2_grid(1.0, r, n, n)
        for n in (40, 80, 320):
            assert not cr.is_mandatory_s2_grid(1.0, r, n, n)
    for lab, (nx, ny) in q.FAMILY_II.items():
        assert cr.is_mandatory_s2_grid(1.0, q.R10[lab], nx, ny)
        assert not cr.is_mandatory_s2_grid(1.0, q.R10[lab] * 1.01, nx, ny)


def test_k3_unresolved_mandatory_s2_keeps_running_and_is_recorded(monkeypatch):
    real = cr.pair_energy

    def unresolved(*args, **kwargs):
        e = real(*args, **kwargs)
        return dataclasses.replace(e, status=cr.UNRESOLVED, tau=e.Q) if e.tau is not None else e

    monkeypatch.setattr(cr, "pair_energy", unresolved)
    run = cr.run_coupled(1.0, 1.0, 1.0, 1.0, 160, 160, 2, transform="fft", half_steps=6, research_mode=True,
                         stop_on_stabilization=False)
    assert run.status == cr.RESEARCH_LIMIT_REACHED and len(run.steps) == 6  # not stopped, no artificial status
    assert run.energy_s2_mandatory and run.energy_s2_mandatory_unresolved is True
    assert run.energy_s2_status == cr.UNRESOLVED and math.isclose(run.energy_s2_m, 0.1, rel_tol=1e-12)
    assert "m_2" in run.energy_s2_reason and run.steps[1].m_s == run.energy_s2_m
    assert run.steps[1].energy.status == cr.UNRESOLVED  # not PASS


def test_k3_unresolved_mandatory_s2_flag_in_the_half_step_log(monkeypatch):
    """K-3.3, sub-point 2 (FAIL on 3da624a, QA 7bb60e7; Builder fix 00cf453): the flag is in the log of s = 2 too."""
    real = cr.pair_energy

    def unresolved(*args, **kwargs):
        e = real(*args, **kwargs)
        return dataclasses.replace(e, status=cr.UNRESOLVED, tau=e.Q) if e.tau is not None else e

    with monkeypatch.context() as m:
        m.setattr(cr, "pair_energy", unresolved)
        run = cr.run_coupled(1.0, 1.0, 1.0, 1.0, 160, 160, 2, transform="fft", half_steps=3, research_mode=True,
                             stop_on_stabilization=False)
        off = cr.run_coupled(1.0, 1.0, 1.0, 1.0, 40, 40, 2, transform="fft", half_steps=3, research_mode=True,
                             stop_on_stabilization=False)  # UNRESOLVED on a grid outside D-048.1
    # mandatory grid, s = 2 UNRESOLVED: flag in the result and in the log of s = 2, same reason, same m_2
    assert run.status == cr.RESEARCH_LIMIT_REACHED and len(run.steps) == 3  # not stopped by the UNRESOLVED test
    st2 = run.steps[1]
    assert st2.s == 2 and run.energy_s2_mandatory is True
    assert run.energy_s2_mandatory_unresolved is True and st2.energy_s2_mandatory_unresolved is True
    assert run.energy_s2_reason and st2.energy_s2_reason == run.energy_s2_reason
    assert st2.m_s == run.energy_s2_m and math.isclose(run.energy_s2_m, 0.1, rel_tol=1e-12)
    assert st2.energy.status == run.energy_s2_status == cr.UNRESOLVED  # not PASS
    for st in (run.steps[0], run.steps[2]):  # s != 2
        assert st.energy_s2_mandatory_unresolved is None and st.energy_s2_reason is None
    # not mandatory: UNRESOLVED at s = 2 gives False, without a reason
    assert off.energy_s2_mandatory is False and off.energy_s2_status == cr.UNRESOLVED
    assert off.energy_s2_mandatory_unresolved is False and off.steps[1].energy_s2_mandatory_unresolved is False
    assert off.energy_s2_reason is None and off.steps[1].energy_s2_reason is None
    # resolved s = 2 on a mandatory grid: False in both, no reason
    ok = cr.run_coupled(1.0, 1.0, 1.0, 1.0, 160, 160, 2, transform="fft", half_steps=3, research_mode=True,
                        stop_on_stabilization=False)
    assert ok.energy_s2_mandatory is True and ok.energy_s2_status != cr.UNRESOLVED
    assert ok.energy_s2_mandatory_unresolved is False and ok.steps[1].energy_s2_mandatory_unresolved is False
    assert ok.energy_s2_reason is None and ok.steps[1].energy_s2_reason is None
    assert all(st.energy_s2_mandatory_unresolved is None and st.energy_s2_reason is None
               for st in ok.steps if st.s != 2)
    # s = 2 not reached: no value anywhere
    one = cr.run_coupled(1.0, 1.0, 1.0, 1.0, 160, 160, 2, transform="fft", half_steps=1, research_mode=True)
    assert len(one.steps) == 1 and not one.energy_s2_checked
    assert one.energy_s2_mandatory_unresolved is None and one.energy_s2_reason is None and one.energy_s2_m is None
    assert one.steps[0].energy_s2_mandatory_unresolved is None and one.steps[0].energy_s2_reason is None


def test_k3_resolved_e1_violation_stops_before_the_qr(monkeypatch):
    real = cr.pair_energy
    monkeypatch.setattr(cr, "pair_energy",
                        lambda *a, **k: dataclasses.replace(real(*a, **k), status=cr.E1_VIOLATED, e1=False))
    run = cr.run_coupled(1.0, 1.0, 1.0, 1.0, 160, 160, 3, transform="fft", half_steps=10, research_mode=True)
    assert run.status == cr.PROVISIONAL_ENERGY_E1_STOP and len(run.steps) == 2 and not run.steps[-1].qr_done
    assert run.x.shape[1] == run.y.shape[1] == 3


# --- D-1 (review 782e4ab): integer types of the grid sizes in the s = 2 classifier (address re-check of 00cf453) ------

def _qa_mandatory(a, b, nx, ny):
    """QA's own statement of D-048.1 (list of grids) with the 4u comparison of r = b/a (D-038), exact rationals."""
    fa, fb, tol = Fraction(a), Fraction(b), 4 * Fraction(U)
    close = lambda r: abs(fb - r * fa) <= tol * max(fb, r * fa)  # noqa: E731
    if nx == ny and nx in (160, 640, 1280):
        return any(close(Fraction(lab)) for lab in q.R10)
    if (nx, ny) in list(q.FAMILY_II.values()):
        return close(Fraction(ny, nx))
    return False


INT_TYPES = (int, np.int64, np.int32)


def test_d1_reviewer_case_returns_a_status_and_agrees_with_int():
    runs = {t.__name__: cr.run_coupled(1.0, 0.2, 1.0, 1.0, t(1600), t(320), 2, transform="fft", half_steps=2,
                                       research_mode=True) for t in INT_TYPES}  # no OverflowError escapes
    base = runs["int"]
    assert base.status == cr.RESEARCH_LIMIT_REACHED and len(base.steps) == 2
    assert base.energy_s2_mandatory is True and base.energy_s2_mandatory_unresolved is False
    for name, run in runs.items():  # same process, same path: bitwise
        assert run.status == base.status and run.reason == base.reason, name
        assert np.array_equal(run.x, base.x) and np.array_equal(run.y, base.y), name
        assert (run.energy_s2_checked, run.energy_s2_mandatory, run.energy_s2_status, run.energy_s2_m,
                run.energy_s2_mandatory_unresolved, run.energy_s2_reason) == \
               (base.energy_s2_checked, base.energy_s2_mandatory, base.energy_s2_status, base.energy_s2_m,
                base.energy_s2_mandatory_unresolved, base.energy_s2_reason), name
        log = [(st.energy.P, st.energy.status, st.m_s, st.eta_N, st.energy_s2_mandatory_unresolved, st.energy_s2_reason)
               for st in run.steps]
        assert log == [(st.energy.P, st.energy.status, st.m_s, st.eta_N, st.energy_s2_mandatory_unresolved,
                        st.energy_s2_reason) for st in base.steps], name
        assert type(run.parameters.n_x) is int and (run.parameters.n_x, run.parameters.n_y) == (1600, 320)
        assert abs(run.parameters.eps_floor - base.parameters.eps_floor) <= 4 * U * base.parameters.eps_floor


def test_d1_classifier_same_for_int_types_and_equal_to_the_qa_rule_around_the_4u_boundary():
    grids = [(n, n) for n in (160, 640, 1280)] + list(q.FAMILY_II.values())
    neighbours = [(nx + d, ny) for nx, ny in grids for d in (-2, 2)] + [(nx, ny + 2) for nx, ny in grids] \
        + [(ny, nx) for nx, ny in q.FAMILY_II.values()] + [(n, n) for n in (40, 80, 320, 2560)]
    crossed = 0
    for nx, ny in grids + neighbours:
        for lab in q.R10:
            b0 = float(Fraction(lab))
            seen = set()
            for k in range(-6, 7):  # b0 moved by k ulps: crosses the 4u band (about 3 ulps at r = 1/5)
                b = b0
                for _ in range(abs(k)):
                    b = math.nextafter(b, math.inf if k > 0 else -math.inf)
                want = _qa_mandatory(1.0, b, nx, ny)
                seen.add(want)
                got = {cr.is_mandatory_s2_grid(1.0, b, t(nx), t(ny)) for t in INT_TYPES}
                assert got == {want}, (nx, ny, lab, k)
            crossed += seen == {True, False}
    assert crossed > 0  # the sweep reaches both sides of the 4u boundary
    # the six family II grids are mandatory exactly at h_x = h_y, for every integer type
    for lab, (nx, ny) in q.FAMILY_II.items():
        for t in INT_TYPES:
            assert cr.is_mandatory_s2_grid(1.0, q.R10[lab], t(nx), t(ny)) is True
            assert cr.is_mandatory_s2_grid(1.0, q.R10[lab] * (1 + 1e-3), t(nx), t(ny)) is False
    assert cr.MANDATORY_S2_FAMILY_I == (160, 640, 1280)
    assert set(cr.MANDATORY_S2_FAMILY_II) == set(q.FAMILY_II.values()) and len(cr.MANDATORY_S2_FAMILY_II) == 6
    assert set(cr.CONTROL_R) == {Fraction(lab) for lab in q.R10}


@pytest.mark.parametrize("bad", [1600.0, np.float64(1600.0), "1600", True, np.bool_(True), Fraction(1600), None])
def test_d1_invalid_size_types_are_not_converted(bad):
    with pytest.raises(TypeError):
        cr.is_mandatory_s2_grid(1.0, 0.2, bad, 320)
    with pytest.raises(TypeError):
        cr.is_mandatory_s2_grid(1.0, 0.2, 1600, bad)
    for run in (cr.run_coupled(1.0, 0.2, 1.0, 1.0, bad, 320, 2, transform="fft", half_steps=2, research_mode=True),
                cr.run_coupled(1.0, 0.2, 1.0, 1.0, 1600, bad, 2, transform="fft", half_steps=2, research_mode=True)):
        assert run.status == "ERROR_INPUT" and run.steps == () and run.parameters is None


# --- K-4: residual, one refinement, ERROR_LINEAR_SOLVE (monkeypatch of the internal mode solvers) -----------------

def _eta_qa(given, given_grid, solved, solve_grid):
    A, B, C, F = cmp.qa_coefficients(np.asarray(given), given_grid.h)
    hs = solve_grid.h
    rho = F[None, :] - cmp.qa_apply_K(np.asarray(solved), hs, A, B, C)
    norm = lambda M: float(np.max(np.sum(np.abs(M), axis=1)))  # noqa: E731
    scale = (16 * norm(A) / hs**4 + 8 * norm(B) / hs**2 + norm(C)) * float(np.max(np.abs(solved))) + float(np.max(np.abs(F)))
    return float(np.max(np.abs(rho))) / scale


def test_k4_perturbation_fixed_by_exactly_one_refinement(monkeypatch):
    real, calls = kc._solve_modes, {"refine": 0}

    def perturbed(*a, **k):
        v = real(*a, **k)
        return v + 1e-8 * float(np.max(np.abs(v)))  # relative 1e-8 in every mode coefficient

    real_refine = cr.refine_once

    def counting(step):
        calls["refine"] += 1
        return real_refine(step)

    monkeypatch.setattr(kc, "_solve_modes", perturbed)
    monkeypatch.setattr(cr, "refine_once", counting)
    run = cr.run_coupled(1.0, 1.0, 1.0, 1.0, 40, 40, 2, transform="fft", half_steps=1, research_mode=True)
    st = run.steps[0]
    assert run.status == cr.RESEARCH_LIMIT_REACHED and calls["refine"] == 1 and st.refined
    assert st.eta_initial > 1e-12 and st.eta_N <= 1e-12
    gx, gy = Grid1D(1.0, 40), Grid1D(1.0, 40)
    assert _eta_qa(kc.odd_sine_set(gx, 2), gx, run.y, gy) <= 1e-12  # independent of coupled_residual


def test_k4_persistent_error_gives_error_linear_solve_after_one_refinement(monkeypatch):
    real_s, real_c, calls, shift = kc._solve_modes, kc._correction_modes, {"refine": 0}, {}

    def solve(*a, **k):  # constant additive shift 1e-6 max|v| of the solution modes v ...
        v = real_s(*a, **k)
        shift["c"] = 1e-6 * float(np.max(np.abs(v)))
        return v + shift["c"]

    def correction(*a, **k):  # ... and the same shift on every further call (the refinement)
        return real_c(*a, **k) + shift["c"]

    monkeypatch.setattr(kc, "_solve_modes", solve)
    monkeypatch.setattr(kc, "_correction_modes", correction)
    real_refine = cr.refine_once

    def counting(step):
        calls["refine"] += 1
        return real_refine(step)

    monkeypatch.setattr(cr, "refine_once", counting)
    run = cr.run_coupled(1.0, 1.0, 1.0, 1.0, 40, 40, 2, transform="fft", half_steps=3, research_mode=True)
    assert run.status == "ERROR_LINEAR_SOLVE" and calls["refine"] == 1 and len(run.steps) == 0


def test_k4_no_unnecessary_refinement(monkeypatch):
    real, calls = kc._solve_modes, {"solve": 0, "refine": 0}

    def counting_solve(*a, **k):
        calls["solve"] += 1
        return real(*a, **k)

    monkeypatch.setattr(kc, "_solve_modes", counting_solve)
    monkeypatch.setattr(cr, "refine_once", lambda step: calls.__setitem__("refine", calls["refine"] + 1) or step)
    run = cr.run_coupled(1.0, 1.0, 1.0, 1.0, 40, 40, 3, transform="fft", half_steps=5, research_mode=True,
                         stop_on_stabilization=False)
    assert len(run.steps) == 5 and calls == {"solve": 5, "refine": 0}
    assert all(st.eta_N <= 1e-12 and not st.refined for st in run.steps)


# --- K-5: rank guards and their priority ---------------------------------------------------------------------------

def test_k5_no_rank_deficient_construction_passes_the_guards():
    for row in pq.b4_rank():
        assert row["criterion_i_rank_guard_fails"], row


@pytest.mark.parametrize("fail_at", ["1", "3", "last"])
def test_k5_guard_failure_stops_before_the_qr_and_before_any_stop_status(monkeypatch, fail_at):
    base = cr.run_coupled(1.0, 1.0, 1.0, 1.0, 40, 40, 3, transform="fft", half_steps=100, research_mode=True)
    s_fail = len(base.steps) if fail_at == "last" else int(fail_at)  # "last": the half-step of the stabilization
    real, count, K = cr._guards, {"n": 0}, cr.K_RESEARCH

    def substituted(rank):  # test substitution of K_RESEARCH on half-step s_fail only (K u kappa_rep >= 1 there)
        count["n"] += 1
        cr.K_RESEARCH = 1e30 if count["n"] == s_fail else K
        try:
            return real(rank)
        finally:
            cr.K_RESEARCH = K

    monkeypatch.setattr(cr, "_guards", substituted)
    run = cr.run_coupled(1.0, 1.0, 1.0, 1.0, 40, 40, 3, transform="fft", half_steps=100, research_mode=True)
    assert cr.K_RESEARCH == K
    assert run.status == cr.PROVISIONAL_RANK_GUARD_STOP and len(run.steps) == s_fail
    assert not run.steps[-1].qr_done and not run.steps[-1].guards.applicable
    assert run.x.shape[1] == run.y.shape[1] == 3 and run.first_stabilization_s is None
    assert not any(st.stop_reason == cr.PROVISIONAL_FIELD_STABILIZED for st in run.steps)
    for a, b in zip(base.steps[:s_fail - 1], run.steps[:s_fail - 1]):  # earlier half-steps unchanged
        assert a.energy.P == b.energy.P and a.rank.rho_N == b.rank.rho_N and a.qr_done == b.qr_done


# --- K-6: dense DST (exact checks and the np.sin premise on this platform; high precision in the script) -----------

def test_k6_exact_reduction_zeros_symmetry_and_sign():
    for row in q.k6_exact():
        assert row["verdict"] == "PASS", row


def test_k6_np_sin_premise_on_this_platform(capsys):
    """Measures np.sin on the actual Builder arguments against decimal (50 digits) and prints the record: on a
    platform where the premise <= 1 ulp fails, the QA model 7u is UNRESOLVED there (plan K-6.0) - the assertion makes
    such a platform visible in CI; it is a check of a premise, not of the Builder algorithm."""
    rows = q.sin_premise()
    with capsys.disabled():
        print("\nK-6.0 platform:", q.platform_record())
        for row in rows:
            print("K-6.0", row)
    for row in rows:
        assert row["arguments_rebuilt_bitwise"] and row["same_values_1d_2d"]
        assert row["premise_le_1ulp"], row


# --- K-7: A4-3 (n <= 640, per coefficient) and A4-12 (area of D-048.7) ----------------------------------------------

def test_k7_a4_3_per_coefficient_up_to_640():
    for row in q.k7_a43(n_list=(40, 160, 320, 640)):
        assert row["max_rel_error_per_coefficient"] <= 1e-13, row


def test_k7_a4_12_in_the_area_of_d048_7():
    for row in q.k7_a412():
        assert row["verdict"] == "PASS", row


# --- K-8: statuses ------------------------------------------------------------------------------------------------

GROUP_B_SHA256 = {  # docs/results/r4_group_b at main 8c4dbe1 (LF line endings); continuous.csv, master.csv,
    # tables.md - statuses of D-056.4 only (scripts/r4_group_b_study.py --apply-d056-4)
    "centre_decomposition.csv": "bad0716ec98787d91eecb2f8c1129eb9fc7ceeeac71c7c1ebfaeda4b381d5dce",
    "continuous.csv": "4492ccc853fe9b689312da25660ca230003dccec74dedba0ea58e747d04c94e1",
    "full_grid_r1.csv": "bf54f6495f17f32a261c3cbdd0c5552e910e35021354abe7e55fe5cde882e0d8",
    "grid.csv": "d3f55c9e7f58e7f81fd85bf05cbec1001bd4c05e3d3f8a9d9d2403e4f89c235e",
    "high_precision.csv": "84918a7a84673939ec3b4de2be9b277a0cb86aa08a033dd4434e446b1b5dd586",
    "master.csv": "2246585a36df42d6e7e005d12c369e3f31bf719092eb662f3f9e749e5d8c2c6c",
    "meta.json": "535e6c7ce5cc9716b6febd09e4aa23a519f472280f6b8abb8e267fc4ff1237ff",
    "richardson.csv": "9e883e3a3a1c138a06bda02a66f205b0217f8bb048661334479f35cb2f86f106",
    "s1_navier.csv": "9e74075a42d3873a0b5322391b2bf1396a5abae2a2e6bdd69eecad98fcbbbda0",
    "scale_invariance.csv": "289234d52e10c6cc08243f1f92a252bbcc55e9747accc70af4b03e3d17bc9e07",
    "tables.md": "0ac1139b1d9c0388a04d310cc483be7cec1b8905ccee54b2a671a3f583dfbdd3",
}


def test_k8_statuses_and_unchanged_scientific_records():
    assert not hasattr(cr, "CONVERGED")
    run = cr.run_coupled(1.0, 1.0, 1.0, 1.0, 40, 40, 2, transform="fft", half_steps=100, research_mode=True)
    assert run.status == cr.PROVISIONAL_FIELD_STABILIZED and not run.converged
    assert run.mark == cr.MARK == "PROVISIONAL / NOT_QUALIFIED" and all(st.mark == cr.MARK for st in run.steps)
    assert "not convergence" in run.reason and "not a rank certificate" in run.reason
    for name, digest in GROUP_B_SHA256.items():
        data = (ROOT / "docs" / "results" / "r4_group_b" / name).read_bytes().replace(b"\r\n", b"\n")
        assert hashlib.sha256(data).hexdigest() == digest, name
    q022 = [line for line in (ROOT / "docs" / "OPEN_QUESTIONS.md").read_text(encoding="utf-8").splitlines()
            if line.startswith("| Q-022 |")]
    assert len(q022) == 1 and "| OPEN." in q022[0]
