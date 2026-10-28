"""Tests of the QA tool of own B-C trajectories with the C-3+ assembly (scripts/r5_cccc_q4_d1_c3plus_trajectories.py; OP-8): own
sequences for N = 1, 2, 3 (orientation, I-4, QR of the prototype, transfer of the factors), the input of s + 1 is the B-C result of s
(no R-AC), C-3+ of the solved set after QR, only C replaced, no fallback to OLD on a failed premise, exact dependence stopped before QR,
zero load and zero field, the small control (14, 6), N = 3 (test configuration only), the independent exact / R-H check of small
half-steps, the journal and the restoration of the next input, the wrapper restoration, no import of the working kkvim and no change of
the used modules. Small deterministic configurations only; no D1 run."""

import ast
import importlib.util
import json
import subprocess
import sys
import threading
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from kkvim_reference import r5_cccc_bc as bc
from kkvim_reference import r5_cccc_halfstep as hs
from kkvim_reference import r5_cccc_sequence as seq

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "r5_cccc_q4_d1_c3plus_trajectories.py"
spec = importlib.util.spec_from_file_location("r5_cccc_q4_d1_c3plus_trajectories", SCRIPT)
tj = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tj)
ca = tj._ca()

# small test configurations (none of them is a D1 grid except the explicitly requested control (14, 6), N = 3)
CFG1 = tj.TrajectoryConfig(1.0, 8, 1.25, 10, 1, s_max=4)
CFG2 = tj.TrajectoryConfig(1.0, 8, 1.25, 10, 2, s_max=4)
CFG3 = tj.TrajectoryConfig(1.0, 8, 1.25, 10, 3, s_max=4)
CTRL = tj.TrajectoryConfig(1.0, 14, 1.0, 6, 3, s_max=8)


@pytest.fixture(scope="module")
def runs():
    return {c.N: tj.run_trajectory(c) for c in (CFG1, CFG2, CFG3)}


def _halfstep(cfg, s, given):
    return hs.HalfStep(cfg.lx, cfg.nx, cfg.ly, cfg.ny, tj.direction_of(s), given, D=cfg.D, q=cfg.q)


def test_config_is_validated_and_the_first_half_step_is_by_y():
    for bad in ({"N": 4}, {"s_max": 9}, {"s_max": 0}, {"nx": 2}, {"lx": 0.0}, {"q": float("nan")}):
        with pytest.raises(ValueError):
            tj.TrajectoryConfig(**{**{"lx": 1.0, "nx": 8, "ly": 1.0, "ny": 8, "N": 1}, **bad})
    assert [tj.direction_of(s) for s in range(1, 9)] == ["Y", "X"] * 4
    assert not hasattr(CFG1, "first")  # D-049.3, item 2 is fixed, not an option


# --- A. N = 1: own sequence of at least three half-steps -----------------------------------------------------------------------

def test_n1_own_sequence_with_I4(runs):
    tr = runs[1]
    assert tr.status.startswith("COMPLETED") and len(tr.records) == 4
    assert np.array_equal(tr.arrays[0]["given_in"], hs.start_set(8, 1)) and not np.any(tr.arrays[0]["companion_in"])
    assert tr.records[0]["prev_field"].startswith("ZERO")
    for rec, arr in zip(tr.records, tr.arrays):
        assert rec["status"] == tj.OK and rec["qr"] is False and "R" not in arr
        assert np.array_equal(arr["solved_set"], arr["given_in"]) and np.array_equal(arr["companion_solved"], arr["companion_in"])
        f, c = arr["V_bc"][:, 0], rec["normalisation_c"]
        exact_sum = sum((Fraction(float(v)) for v in f), Fraction(0))
        assert rec["sigma"] == (1.0 if exact_sum >= 0 else -1.0) and c == rec["sigma"] * float(np.max(np.abs(f)))
        assert np.array_equal(arr["next_given"], (f / c)[:, None])  # f <- f/c
        assert np.array_equal(arr["next_companion"], arr["solved_set"] * c)  # g <- g c
        assert rec["f_after_max_abs"] == 1.0 and sum(Fraction(float(v)) for v in arr["next_given"][:, 0]) >= 0
        # fields before and after I-4 are recorded separately
        assert rec["field_s_sha"] == tj.array_sha(arr["solved_set"] @ arr["V_bc"].T if rec["direction"] == "Y"
                                                  else arr["V_bc"] @ arr["solved_set"].T)
        assert "field_s_after_normalisation_sha" in rec and "field_s_normalisation_max_abs_change" in rec
    for k in range(1, 4):
        assert tr.records[k]["input_link_bitwise"] is True
        assert np.array_equal(tr.arrays[k]["given_in"], tr.arrays[k - 1]["next_given"])
        assert np.array_equal(tr.arrays[k]["companion_in"], tr.arrays[k - 1]["next_companion"])


# --- B. N = 2, 3: QR, transfer of the factors, orientation ------------------------------------------------------------------------

@pytest.mark.parametrize("N", [2, 3])
def test_n23_own_sequences_qr_transfer_orientation(runs, N):
    tr = runs[N]
    cfg = tr.config
    assert tr.status.startswith("COMPLETED") and len(tr.records) == 4
    for k, (rec, arr) in enumerate(zip(tr.records, tr.arrays)):
        s = k + 1
        n_f, n_s = (cfg.nx, cfg.ny) if s % 2 else (cfg.ny, cfg.nx)
        assert (rec["direction"], rec["n_f"], rec["n_s"]) == (tj.direction_of(s), n_f, n_s)
        assert arr["given_in"].shape == (n_f - 1, N) and arr["V_bc"].shape == (n_s - 1, N)
        assert rec["marking"].startswith("RESEARCH_ONLY") and rec["rank_decision"].startswith("UNRESOLVED")
        p = _halfstep(cfg, s, arr["given_in"])
        Ut, R = bc.orthonormalise(p)  # the prototype's QR (NS 5.1, step 2), also at s = 1
        assert np.array_equal(arr["solved_set"], Ut) and np.array_equal(arr["R"], R)
        proto = bc.run_halfstep(p, qr=True)  # same solved set and R as the prototype's own order
        assert np.array_equal(proto.solved.given, arr["solved_set"]) and np.array_equal(proto.R, arr["R"])
        assert np.array_equal(arr["companion_solved"], arr["companion_in"] @ R.T)  # C' = fl(C R^T)
        assert np.array_equal(arr["next_given"], arr["V_bc"]) and np.array_equal(arr["next_companion"], arr["solved_set"])
        assert rec["V_exact_rank"] == N and "field_s_rho_N" in rec
        if s == 1:
            assert rec["cond_A_before_QR"] == pytest.approx(1 + 2 * N)  # NS 5.1: T_N, 2N < n_x (OBSERVATION of the log)
            assert rec["start_independence_condition"].endswith("True")
        else:
            assert np.array_equal(arr["given_in"], tr.arrays[k - 1]["V_bc"])
            assert np.array_equal(arr["companion_in"], tr.arrays[k - 1]["solved_set"])
            assert "prev_field_rho_N" in rec and rec["input_link_bitwise"] is True


# --- C. the next half-step uses the previous B-C result, not R-AC -------------------------------------------------------------

@pytest.mark.parametrize("cfg", [CFG1, CFG2])
def test_input_of_next_step_is_the_bc_result_not_rac(runs, cfg, monkeypatch):
    ref = runs[cfg.N]
    for k, arr in enumerate(ref.arrays):
        s_bc = arr["V_bc"].T.reshape(-1)
        ac = hs.solve_ac(_halfstep(cfg, k + 1, arr["solved_set"]))  # R-AC on the same solved system
        assert ac.s is not None and not np.array_equal(ac.s, s_bc)  # the trajectory value is not the R-AC value
    for k in range(1, len(ref.arrays)):
        prev, rec = ref.arrays[k - 1], ref.records[k - 1]
        from_bc = prev["V_bc"] if cfg.N >= 2 else prev["V_bc"] / rec["normalisation_c"]  # the B-C factor of s - 1 (after I-4, N = 1)
        assert np.array_equal(ref.arrays[k]["given_in"], from_bc)

    def boom(*a, **k):
        raise AssertionError("an R-AC / R-H / Q3-controller / OLD-run path was called by the trajectory")

    for mod, name in ((hs, "solve_ac"), (hs, "solve_hessian"), (hs, "hessian_system"), (seq, "run_sequence"), (bc, "run_halfstep"),
                      (bc, "compare")):
        monkeypatch.setattr(mod, name, boom)
    tr = tj.run_trajectory(cfg)
    assert tr.status == ref.status
    for a, b in zip(tr.arrays, ref.arrays):
        assert a.keys() == b.keys() and all(np.array_equal(a[x], b[x]) for x in a)


# --- D. C-3+ of the solved set after QR; E. only C replaced ----------------------------------------------------------------------

@pytest.mark.parametrize("cfg", [CFG1, CFG2, CFG3])
def test_c3plus_is_assembled_on_the_solved_set_after_qr(cfg, monkeypatch):
    seen = []
    orig = ca.c3plus

    def spy(p):
        seen.append(np.array(p.given, copy=True))
        return orig(p)

    monkeypatch.setattr(ca, "c3plus", spy)
    tr = tj.run_trajectory(cfg)
    assert len(seen) == len(tr.records)
    for k, (rec, arr) in enumerate(zip(tr.records, tr.arrays)):
        assert np.array_equal(seen[k], arr["solved_set"])
        if cfg.N >= 2:
            assert not np.array_equal(seen[k], arr["given_in"])  # after QR, not the set before QR
        assert np.array_equal(arr["C3plus"], orig(_halfstep(cfg, k + 1, arr["solved_set"]))["C"])
        assert rec["c3plus_bound_check_exact"] == "PASS" and rec["c3plus_C_spd_exact"] is True
        assert rec["c3plus_bound_check_scope"] == tj.BOUND_CHECK_SCOPE and "not a proof or a refutation" in tj.BOUND_CHECK_SCOPE
        assert rec["c3plus_premises_status"] == tj.PREMISES_OK == "ESTABLISHED_CONDITIONAL_ON_P4"
        assert rec["c3plus_premises_summary"] == tj.PREMISES_OK_SUMMARY
        assert rec["c3plus_premises"]["classes"] == {"P-1": "ESTABLISHED", "P-2": "ESTABLISHED", "P-3": "ESTABLISHED",
                                                     "P-5": "ESTABLISHED", "P-4": "ASSUMED"}
        assert rec["c3plus_premises"]["P-4"].startswith("ASSUMED") and "not proven" in rec["c3plus_premises"]["P-4"]
        assert rec["C_assembly_applied"].startswith("C-3+")
        # the B-C solution is the prototype's solve_bc with only C replaced (recomputed independently here, single thread, by the
        # global replacement replaced_c of PR #40 - used in this test only, not by the tool)
        solved = _halfstep(cfg, k + 1, arr["solved_set"])
        with ca.replaced_c(arr["C3plus"]):
            sol = bc.solve_bc(solved)
        assert np.array_equal(sol.s, arr["V_bc"].T.reshape(-1))


@pytest.mark.parametrize("cfg", [CFG1, CFG2, CFG3])
def test_only_c_is_replaced(cfg):
    """E (Б-1): A, B, F bitwise those of the prototype's own path; only C replaced; the code of solve_bc is the prototype's."""
    tr = tj.run_trajectory(cfg)
    for k, (rec, arr) in enumerate(zip(tr.records, tr.arrays)):
        solved = _halfstep(cfg, k + 1, arr["solved_set"])
        A, B, C_old, F = bc.bc_coefficients(solved)
        assert rec["only_C_replaced"] is True and rec["bc_coefficients_calls"] == 1 and rec["solve_isolation"] == tj.ISOLATION
        assert rec["solve_bc_code_is_prototype"] is True and rec["private_globals_differ_only_in_bc_coefficients"] is True
        assert (rec["A_sha"], rec["B_sha"], rec["F_sha"], rec["C_old_sha"]) == tuple(tj.array_sha(x) for x in (A, B, F, C_old))
        assert rec["C_old_equals_C3plus_bitwise"] is False and tj.array_sha(arr["C3plus"]) != rec["C_old_sha"]
    assert bc.bc_coefficients is tj._BC_COEFFICIENTS and bc.solve_bc is tj._BC_SOLVE


def test_a_drift_of_a_inside_the_call_is_detected(monkeypatch):
    calls = {"n": 0}
    orig = tj._BC_COEFFICIENTS

    def drifting(p):  # the second call (the one inside the isolated solve_bc) returns another A
        calls["n"] += 1
        A, B, C, F = orig(p)
        return (2.0 * A if calls["n"] == 2 else A), B, C, F

    monkeypatch.setattr(tj, "_BC_COEFFICIENTS", drifting)
    tr = tj.run_trajectory(CFG1)
    assert tr.records[0]["status"].startswith(tj.FAIL_TOOL) and tr.records[0]["only_C_replaced"] is False
    assert len(tr.records) == 1 and tr.status.startswith("STOPPED at s = 1")


# --- Б-1. isolated C-3+ solve: no global replacement; external B-C calls keep the prototype's C ----------------------------------

TIMEOUT = 120  # seconds; only against a hang of the test, not a numerical tolerance


def _solved_c3(cfg=CFG2):
    p = _halfstep(cfg, 1, hs.start_set(cfg.nx, cfg.N))
    solved = _halfstep(cfg, 1, bc.orthonormalise(p)[0]) if cfg.N >= 2 else p
    return solved, ca.c3plus(solved)["C"]


def _c3plus_reference(solved, C):
    """Single-thread reference: solve_bc with only C replaced by replaced_c of PR #40 (used in the tests only, not by the tool)."""
    with ca.replaced_c(C):
        return bc.solve_bc(solved).s.copy()


def _bc_namespace_identity():
    return {k: id(v) for k, v in vars(bc).items()}


def _inside_isolated_call():
    return sys._getframe(2).f_code.co_name == "bc_coefficients_c3plus"  # caller of the hook is the tool's local function


def test_prototype_functions_keep_their_identity_during_the_call():
    """Б-1 A: bc.bc_coefficients, bc.solve_bc and the module namespace never change during the tool's calls (checked at every Python
    call event of this thread by a profile function) and after them."""
    solved, C = _solved_c3()
    before = _bc_namespace_identity()
    events, bad = [0], []

    def prof(frame, event, arg):
        if event == "call":
            events[0] += 1
            if bc.bc_coefficients is not tj._BC_COEFFICIENTS or bc.solve_bc is not tj._BC_SOLVE:
                bad.append(frame.f_code.co_name)

    sys.setprofile(prof)
    try:
        sol, info = tj.solve_with_c(solved, C)
        res = tj.run_step(CFG2, 1, hs.start_set(8, 2), np.zeros((9, 2)))
    finally:
        sys.setprofile(None)
    assert events[0] > 50 and not bad
    assert info["only_C_replaced"] is True and res.record["status"] == tj.OK
    assert _bc_namespace_identity() == before


def test_tool_source_has_no_global_replacement():
    """Б-1 A, N-4: the tool assigns into no module namespace and no environment variable, calls no replaced_c, uses no lock."""
    src = SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(src)
    modules = {"bc", "hs", "seq", "np", "scipy", "os", "sys", "ca", "types"}
    stores = []
    for n in ast.walk(tree):
        targets = n.targets if isinstance(n, ast.Assign) else [n.target] if isinstance(n, (ast.AugAssign, ast.AnnAssign)) else []
        for t in targets:
            base = t
            while isinstance(base, (ast.Attribute, ast.Subscript)):
                base = base.value
            if t is not base and isinstance(base, ast.Name) and base.id in modules:
                stores.append(ast.unparse(t))
    calls = [ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)]
    assert not stores
    env_mutators = ("os.environ.setdefault", "os.environ.update", "os.environ.pop", "os.environ.clear", "os.putenv", "os.unsetenv")
    assert not [c for c in calls if c in ("setattr", "delattr") or c.endswith("replaced_c") or c in env_mutators]
    assert not [n for n in ast.walk(tree) if isinstance(n, ast.Global)]
    assert "import threading" not in src and "Lock(" not in src


def test_concurrent_external_bc_calls_get_the_prototype_c(monkeypatch):
    """Б-1 Б, В, Ж: an external thread calls the ordinary bc.solve_bc WHILE the tool is inside its isolated solve (the overlap is
    forced by barriers, not left to the scheduler). The external results are bitwise the prototype's (its own C), the tool's results
    are the C-3+ ones; the tool's solution is confirmed by the certified small reference (independent_check)."""
    solved, C = _solved_c3()
    other = hs.HalfStep(1.0, 10, 1.0, 12, "Y", hs.start_set(10, 2))
    ref_same, ref_other = bc.solve_bc(solved).s.copy(), bc.solve_bc(other).s.copy()
    ref_c3 = _c3plus_reference(solved, C)
    assert not np.array_equal(ref_same, ref_c3)
    orig, proto = tj._BC_COEFFICIENTS, bc.bc_coefficients
    start, end = threading.Barrier(2, timeout=TIMEOUT), threading.Barrier(2, timeout=TIMEOUT)

    def pausing(p):  # inside the tool's isolated call: the external thread runs all its calls of this round, then the tool goes on
        res = orig(p)
        if _inside_isolated_call():
            start.wait()
            end.wait()
        return res

    monkeypatch.setattr(tj, "_BC_COEFFICIENTS", pausing)
    rounds, per_round = 5, 10
    ext = {"calls": 0, "wrong": 0, "identity": True, "errors": []}

    def external():
        try:
            for _ in range(rounds):
                start.wait()
                for _ in range(per_round):
                    for p, ref in ((solved, ref_same), (other, ref_other)):
                        ext["identity"] &= bc.bc_coefficients is proto and bc.solve_bc is tj._BC_SOLVE
                        s = bc.solve_bc(p).s
                        ext["calls"] += 1
                        ext["wrong"] += not np.array_equal(s, ref)
                end.wait()
        except Exception as exc:  # reported below
            ext["errors"].append(repr(exc))

    th = threading.Thread(target=external, daemon=True)
    th.start()
    results = [tj.solve_with_c(solved, C) for _ in range(rounds)]
    th.join(TIMEOUT)
    assert not th.is_alive() and not ext["errors"]
    assert ext["calls"] == rounds * per_round * 2 and ext["wrong"] == 0 and ext["identity"]
    for sol, info in results:
        assert info["only_C_replaced"] is True and np.array_equal(sol.s, ref_c3)
    m = solved.n_s - 1
    ic = tj.independent_check(CFG2, 1, {"solved_set": solved.given, "V_bc": results[0][0].s.reshape(2, m).T})
    assert ic["status"] == "CONSISTENT"
    print(f"\nconcurrency: {rounds} forced overlaps, {ext['calls']} external solve_bc calls, wrong {ext['wrong']}; "
          f"tool calls with C-3+ {sum(np.array_equal(s.s, ref_c3) for s, _ in results)}/{rounds} (OBSERVATION)")


def test_tool_unaffected_while_an_external_call_is_inside_bc_solve(monkeypatch):
    """Б-1 Б: the reverse overlap - the external thread is paused INSIDE bc.solve_bc (in its bc_coefficients) while the tool solves;
    a foreign global replacement installed by this test is not used by the tool, and the external call gets the prototype's C."""
    solved, C = _solved_c3()
    ref_same, ref_c3 = bc.solve_bc(solved).s.copy(), _c3plus_reference(solved, C)
    proto = bc.bc_coefficients
    start, end = threading.Barrier(2, timeout=TIMEOUT), threading.Barrier(2, timeout=TIMEOUT)
    main_calls = []

    def external_pause(p):  # a foreign global replacement (this test's), returns the prototype's values
        if threading.current_thread() is threading.main_thread():
            main_calls.append(p)
            return proto(p)
        res = proto(p)
        start.wait()
        end.wait()
        return res

    monkeypatch.setattr(bc, "bc_coefficients", external_pause)
    out = {}
    th = threading.Thread(target=lambda: out.setdefault("s", bc.solve_bc(solved).s.copy()), daemon=True)
    th.start()
    start.wait()
    sol, info = tj.solve_with_c(solved, C)  # the external call is now inside bc.solve_bc
    end.wait()
    th.join(TIMEOUT)
    assert not th.is_alive() and np.array_equal(out["s"], ref_same)
    assert info["only_C_replaced"] is True and np.array_equal(sol.s, ref_c3) and not main_calls


def test_overlap_test_detects_a_global_replacement():
    """Control of the test design (not of the tool): with the same forced overlap, a GLOBAL replacement (replaced_c of PR #40, the
    mechanism of 70f3407) gives the external call the C-3+ result - the overlap test above would detect the old defect."""
    solved, C = _solved_c3()
    ref_same, ref_c3 = bc.solve_bc(solved).s.copy(), _c3plus_reference(solved, C)
    start, end = threading.Barrier(2, timeout=TIMEOUT), threading.Barrier(2, timeout=TIMEOUT)
    out = {}

    def external():
        start.wait()
        out["s"] = bc.solve_bc(solved).s.copy()
        end.wait()

    th = threading.Thread(target=external, daemon=True)
    th.start()
    with ca.replaced_c(C):
        start.wait()
        end.wait()
    th.join(TIMEOUT)
    assert np.array_equal(out["s"], ref_c3) and not np.array_equal(out["s"], ref_same)
    assert bc.bc_coefficients is tj._BC_COEFFICIENTS


def test_exception_inside_the_isolated_call_changes_no_global_state(monkeypatch):
    """Б-1 Г: an exception inside the isolated call (injected, and a natural one from the code of solve_bc) leaves the prototype's
    namespace unchanged; the next calls are correct."""
    solved, C = _solved_c3()
    ref_same, ref_c3 = bc.solve_bc(solved).s.copy(), _c3plus_reference(solved, C)
    before = _bc_namespace_identity()
    orig = tj._BC_COEFFICIENTS

    def raising(p):
        if _inside_isolated_call():
            raise RuntimeError("injected inside the isolated call")
        return orig(p)

    monkeypatch.setattr(tj, "_BC_COEFFICIENTS", raising)
    with pytest.raises(RuntimeError, match="injected"):
        tj.solve_with_c(solved, C)
    tr = tj.run_trajectory(CFG2)
    assert tr.records[0]["status"].startswith(tj.FAIL_TOOL + ": exception RuntimeError") and len(tr.records) == 1
    assert _bc_namespace_identity() == before and bc.bc_coefficients is orig
    monkeypatch.undo()
    with pytest.raises(ValueError):  # a natural exception in the code of solve_bc: C of a wrong shape
        tj.solve_with_c(solved, np.ones((3, 3)))
    assert _bc_namespace_identity() == before
    sol, info = tj.solve_with_c(solved, C)
    assert info["only_C_replaced"] is True and np.array_equal(sol.s, ref_c3) and np.array_equal(bc.solve_bc(solved).s, ref_same)


def test_nested_and_repeated_calls(monkeypatch):
    """Б-1 Д: a call nested inside the isolated call (same thread) completes without a hang, both results are correct; repeated
    calls give bitwise the same result and record."""
    solved, C = _solved_c3()
    solved1, C1 = _solved_c3(CFG1)
    ref, ref1 = _c3plus_reference(solved, C), _c3plus_reference(solved1, C1)
    orig = tj._BC_COEFFICIENTS
    state, nested = {"armed": True}, []

    def nesting(p):
        if state["armed"] and _inside_isolated_call():
            state["armed"] = False
            nested.append(tj.solve_with_c(solved1, C1))
        return orig(p)

    monkeypatch.setattr(tj, "_BC_COEFFICIENTS", nesting)
    box = {}
    th = threading.Thread(target=lambda: box.setdefault("r", tj.solve_with_c(solved, C)), daemon=True)
    th.start()
    th.join(TIMEOUT)
    assert not th.is_alive()  # no deadlock
    sol, info = box["r"]
    assert info["only_C_replaced"] is True and np.array_equal(sol.s, ref)
    ((sol1, info1),) = nested
    assert info1["only_C_replaced"] is True and np.array_equal(sol1.s, ref1)
    monkeypatch.undo()
    reps = [tj.solve_with_c(solved, C) for _ in range(3)]
    assert all(np.array_equal(s.s, ref) and i == reps[0][1] for s, i in reps)
    assert bc.bc_coefficients is tj._BC_COEFFICIENTS and bc.solve_bc is tj._BC_SOLVE


# --- E. a failed premise of C-3+ gives UNRESOLVED, never OLD ------------------------------------------------------------------

def _no_solve(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("B-C solve called although a premise is not established")

    monkeypatch.setattr(tj, "solve_with_c", boom)
    monkeypatch.setattr(bc, "solve_bc", boom)


def test_injected_premise_failure_stops_without_old(monkeypatch):
    _no_solve(monkeypatch)
    monkeypatch.setattr(ca, "premises_c3plus", lambda parts: "P-3: injected")
    tr = tj.run_trajectory(CFG2)
    rec, arr = tr.records[0], tr.arrays[0]
    assert rec["status"].startswith(tj.UNRES_PREMISE) and "P-3: injected" in rec["status"]
    assert "V_bc" not in arr and "next_given" not in arr and len(tr.records) == 1
    assert "bc_status" not in rec and "C_assembly_applied" not in rec
    assert rec["c3plus_premises_status"] == tj.UNRESOLVED and rec["c3plus_premises"]["classes"]["P-3"] == "UNRESOLVED"
    assert rec["c3plus_premises"]["classes"]["P-4"] == "ASSUMED" and rec["c3plus_premises"]["P-3"].startswith("UNRESOLVED")


def _status_of_premises(rec):
    return rec["c3plus_premises_status"], rec["c3plus_premises"]["classes"]


def test_natural_subnormal_premise_failure(monkeypatch):
    _no_solve(monkeypatch)
    given = hs.start_set(10, 1) * 2.0**-1000  # s = 2 (X): X^2 below 2^-1022, Gram products may underflow
    res = tj.run_step(CFG1, 2, given, np.ones((7, 1)))
    assert res.record["status"].startswith(tj.UNRES_PREMISE) and "P-3" in res.record["status"] and res.next_given is None
    status, classes = _status_of_premises(res.record)
    assert status == tj.UNRESOLVED and classes["P-3"] == "UNRESOLVED" and classes["P-4"] == "ASSUMED"


def test_natural_overflow_premise_failure(monkeypatch):
    """P-2 failing naturally (N-6): a given set of size 1e300 makes the Gram sum overflow; no solve, P-2 UNRESOLVED."""
    _no_solve(monkeypatch)
    res = tj.run_step(CFG1, 2, hs.start_set(10, 1) * 1e300, np.ones((7, 1)))
    assert res.record["status"].startswith(tj.UNRES_PREMISE) and "P-2" in res.record["status"]
    status, classes = _status_of_premises(res.record)
    assert status == tj.UNRESOLVED and classes["P-2"] == "UNRESOLVED" and classes["P-4"] == "ASSUMED"


def test_rounding_probe_failure(monkeypatch):
    _no_solve(monkeypatch)
    good = tj.fp_environment_probe()
    assert all(good.values())
    monkeypatch.setattr(tj, "fp_environment_probe", lambda: {**good, "nearest_numpy": False})
    tr = tj.run_trajectory(CFG1)
    assert tr.records[0]["status"].startswith(tj.UNRES_PREMISE) and "P-1" in tr.records[0]["status"]
    assert _status_of_premises(tr.records[0])[1]["P-1"] == "UNRESOLVED"


def test_probe_failure_after_the_solve(monkeypatch):
    """N-6: the P-1/P-3 probe after the solve fails -> UNRESOLVED, the trajectory stops, no next input."""
    good = tj.fp_environment_probe()
    seen = {"n": 0}

    def second_fails():
        seen["n"] += 1
        return good if seen["n"] == 1 else {**good, "no_daz": False}

    monkeypatch.setattr(tj, "fp_environment_probe", second_fails)
    tr = tj.run_trajectory(CFG1)
    assert tr.records[0]["status"] == f"{tj.UNRES_PREMISE}: P-1/P-3 probe after the solve failed" and len(tr.records) == 1
    assert "next_given" not in tr.arrays[0]


def test_bound_violation_is_a_conditional_fail(monkeypatch):
    """Б-2: a violated inequality is FAIL with P-1, P-2, P-3, P-5 established and P-4 ASSUMED - not an unconditional refutation."""
    _no_solve(monkeypatch)
    monkeypatch.setattr(ca, "check_c3plus", lambda C, Cx, k: {"elementwise": False, "frobenius": True, "max_ratio_sq": Fraction(4)})
    tr = tj.run_trajectory(CFG1)
    rec = tr.records[0]
    assert rec["status"] == tj.FAIL_THEOREM and len(tr.records) == 1 and rec["c3plus_bound_check_exact"] == "FAIL"
    assert "P-4 assumed" in tj.FAIL_THEOREM and "not an unconditional refutation" in tj.FAIL_THEOREM
    assert "with established premises" not in tj.FAIL_THEOREM
    assert rec["c3plus_premises_status"] == tj.PREMISES_OK and rec["c3plus_premises"]["classes"]["P-4"] == "ASSUMED"


def test_p4_is_never_declared_established(tmp_path, runs):
    """Б-2: in the code constants and in the JSON journal P-4 is ASSUMED only; the summary is conditional on P-4."""
    tj.write_journal(runs[2], tmp_path)
    data = json.loads((tmp_path / "journal.json").read_text(encoding="utf-8"))
    for rec in data["steps"]:
        prem = rec["c3plus_premises"]
        assert prem["classes"]["P-4"] == "ASSUMED" and prem["P-4"].startswith("ASSUMED")
        assert rec["c3plus_premises_status"] == "ESTABLISHED_CONDITIONAL_ON_P4" and "P-4 assumed" in rec["c3plus_premises_summary"]
    text = (tmp_path / "journal.json").read_text(encoding="utf-8")
    assert '"P-4": "ESTABLISHED' not in text and "P-4 established" not in text
    src = SCRIPT.read_text(encoding="utf-8")
    assert "with established premises" not in src and '"ESTABLISHED (P-1, P-2, P-3, P-5 checked' not in src


# --- F. exact dependence before QR; zero load and zero field ------------------------------------------------------------------

def test_exact_dependence_stops_before_qr(monkeypatch):
    calls = []
    monkeypatch.setattr(bc, "orthonormalise", lambda p: calls.append(p) or (_ for _ in ()).throw(AssertionError("QR")))
    monkeypatch.setattr(ca, "c3plus", lambda p: (_ for _ in ()).throw(AssertionError("C-3+")))
    v = np.linspace(0.1, 0.9, 9)
    dep = np.column_stack([v, 2.0 * v])  # exactly dependent given set (s = 2: X-half-step, fixed y, n_y - 1 = 9 rows)
    res = tj.run_step(CFG2, 2, dep, np.ones((7, 2)))
    assert res.record["status"] == tj.STOP_DEPENDENT and res.record["given_exact_rank"] == 1 and res.next_given is None
    assert not calls and set(res.arrays) == {"given_in", "companion_in"}
    start = tj.run_trajectory(tj.TrajectoryConfig(1.0, 5, 1.0, 7, 3))  # N = 3 > floor(5/2): proven dependence of T_N (D-049.3)
    assert start.records[0]["status"] == tj.STOP_START and not calls
    assert "proven for the exact T_N" in tj.STOP_START and "not a statement about the stored doubles" in tj.STOP_START


def test_start_theorem_is_applied_to_t_n_only():
    """N-2: run_step(s = 1) accepts only T_N = hs.start_set(n_x, N) with a zero companion; any other input is refused."""
    t = hs.start_set(8, 2)
    for given, comp in ((t * 2.0, np.zeros((9, 2))), (t[:, ::-1], np.zeros((9, 2))), (t, np.ones((9, 2)))):
        with pytest.raises(ValueError, match="T_N"):
            tj.run_step(CFG2, 1, given, comp)
    with pytest.raises(ValueError, match="T_N"):  # the start theorem (N > floor(n_x/2)) is not applied to another set
        tj.run_step(tj.TrajectoryConfig(1.0, 5, 1.0, 7, 3), 1, hs.start_set(5, 3) * 3.0, np.zeros((6, 3)))
    res = tj.run_step(CFG2, 1, t, np.zeros((9, 2)))
    assert res.record["status"] == tj.OK and res.record["start_set"].endswith("companion 0")


def test_protective_stop_after_qr(monkeypatch):
    """N-6: an exactly singular R after QR gives PROTECTIVE_STOP (a protection after QR, not the stop before QR of D-049.5)."""
    orig = bc.orthonormalise

    def singular_r(p):
        Ut, R = orig(p)
        R = R.copy()
        R[-1, -1] = 0.0
        return Ut, R

    monkeypatch.setattr(bc, "orthonormalise", singular_r)
    tr = tj.run_trajectory(CFG2)
    assert tr.records[0]["status"].startswith(tj.PROTECTIVE) and len(tr.records) == 1 and "C3plus" not in tr.arrays[0]


def test_protective_stop_on_a_failed_bc_factorization(monkeypatch):
    """N-6: B-C factorization failure (no solution) gives PROTECTIVE_STOP; nothing is passed on."""
    real = tj.solve_with_c

    def failed(solved, C):
        _, info = real(solved, C)
        return bc.BCSolution(hs.FACTORIZATION_FAILED), info

    monkeypatch.setattr(tj, "solve_with_c", failed)
    tr = tj.run_trajectory(CFG2)
    assert tr.records[0]["status"] == f"{tj.PROTECTIVE}: B-C status {hs.FACTORIZATION_FAILED}" and len(tr.records) == 1


def test_broken_input_link_is_fail_tool(monkeypatch):
    """N-6: if the input of s is not bitwise the next input of s - 1, the trajectory stops with FAIL_TOOL."""
    real = tj.run_step

    def tampered(cfg, s, given, companion):
        res = real(cfg, s, given, companion)
        if s == 2:
            res.arrays["given_in"] = res.arrays["given_in"] * (1.0 + 2.0**-52)
        return res

    monkeypatch.setattr(tj, "run_step", tampered)
    tr = tj.run_trajectory(CFG2)
    assert len(tr.records) == 2 and tr.records[1]["input_link_bitwise"] is False
    assert tr.records[1]["status"] == f"{tj.FAIL_TOOL}: input of s is not the B-C result of s - 1"


def test_zero_load_and_zero_field():
    tr1 = tj.run_trajectory(tj.TrajectoryConfig(1.0, 8, 1.0, 10, 1, q=0.0))
    r = tr1.records[0]
    assert r["status"] == tj.STOP_ZERO and r["eta_qa"] == tj.UNRESOLVED and r["V_exact_rank"] == 0 and len(tr1.records) == 1
    assert r["prev_field"].startswith("ZERO") and "normalisation_c" not in r and not np.any(tr1.arrays[0]["V_bc"])
    tr2 = tj.run_trajectory(tj.TrajectoryConfig(1.0, 8, 1.0, 10, 2, q=0.0, s_max=3))
    assert tr2.records[0]["status"] == tj.OK and tr2.records[0]["eta_qa"] == tj.UNRESOLVED
    assert tr2.records[0]["next_step_note"].startswith("V exactly dependent")
    assert tr2.records[1]["status"] == tj.STOP_DEPENDENT and tr2.records[1]["given_exact_rank"] == 0 and len(tr2.records) == 2


# --- G. small control (14, 6), N = 3 (test configuration, not block D) ---------------------------------------------------------

def test_small_control_14x6_n3_with_independent_check():
    tr = tj.run_trajectory(CTRL, independent=True)
    assert tr.status.startswith("COMPLETED s = 1..8") and len(tr.records) == 8
    r1 = tr.records[0]
    assert r1["start_independence_condition"] == "N <= floor(n_x/2): 3 <= 7 -> True"
    assert r1["start_exact_formulas_condition"] == "2N < n_x: 6 < 14 -> True"
    for rec in tr.records:
        assert rec["status"] == tj.OK and "S-1" not in rec["status"]  # no S-1 without a proven dependence (n_y = 6 is no reason)
        assert rec["given_exact_rank"] == 3 and rec["V_exact_rank"] == 3 and rec["rank_decision"].startswith("UNRESOLVED")
        assert rec["independent_check"]["status"] == "CONSISTENT"
    assert [r["n_s"] for r in tr.records] == [6, 14] * 4


# --- H. independent check of small half-steps (exact system, R-H) -----------------------------------------------------------------

@pytest.mark.parametrize("cfg", [CFG1, CFG2, CFG3])
def test_independent_check_of_small_half_steps(cfg):
    tr = tj.run_trajectory(cfg, independent=True)
    for rec in tr.records:
        ic = rec["independent_check"]
        assert ic["status"] == "CONSISTENT" and ic["rh_status"] == hs.INDEPENDENT
        assert ic["exact_solution_identity_Kx_eq_b"] and ic["bc_within_certified_bound"] and ic["rh_within_certified_bound"]
        assert ic["bc_rh_certified_consistent"] and 0.0 < ic["bc_bound_2_up"] < float("inf")


def test_independent_check_detects_an_invalid_certificate(monkeypatch):
    tr = tj.run_trajectory(CFG2)
    monkeypatch.setattr(hs, "certify_lambda_min", lambda p, tau, coeffs=None, digits=50: 1e30)  # not a valid lower bound
    ic = tj.independent_check(CFG2, 1, tr.arrays[0])
    assert ic["status"] == "CONTRADICTION" and ic["bc_within_certified_bound"] is False


# --- journal, provenance, restoration of the next input ----------------------------------------------------------------------

def test_journal_restores_the_next_input(tmp_path, runs):
    tr = runs[2]
    meta = tj.write_journal(tr, tmp_path)
    data = json.loads((tmp_path / "journal.json").read_text(encoding="utf-8"))
    z = tj.load_arrays(tmp_path)
    assert data["meta"]["data_status"].startswith("QA tool data") and "NOT D1 data" in data["meta"]["data_status"]
    prov = data["meta"]["provenance"]
    for k in ("git_commit", "tool_blob", "numpy", "scipy", "platform", "blas", "blas_thread_env_vars_as_found"):
        assert k in prov
    assert prov["blas_threads_effective"].startswith("NOT_VERIFIED") and "threads_env" not in prov
    assert "not across BLAS platforms" in prov["reproducibility"] and prov["solve_isolation"] == tj.ISOLATION
    assert all(v["unchanged"] for v in prov["used_modules"].values())
    assert "src/kkvim_reference/r5_cccc_energy.py" in prov["used_modules"]  # N-5: imported through r5_cccc_sequence
    assert prov["used_modules_status"].startswith("listed modules: blob = base") and "not checked" in prov["used_modules_status"]
    assert all(v.startswith("NOT_CHECKED") for v in prov["loaded_project_modules_not_checked"].values())
    assert not set(prov["loaded_project_modules_not_checked"]) & set(tj.USED_MODULES)
    assert "not a mechanism that prevents a D1 run" in data["meta"]["d1_guard"] == prov["d1_guard"]
    assert len(data["steps"]) == 4 and meta["arrays_sha256"] == {k: tj.array_sha(v) for k, v in z.items()}
    for s in range(1, 4):
        g, c = z[f"s{s}_next_given"], z[f"s{s}_next_companion"]
        assert np.array_equal(g, z[f"s{s + 1}_given_in"]) and np.array_equal(c, z[f"s{s + 1}_companion_in"])
        res = tj.run_step(tr.config, s + 1, g, c)  # independent restoration of half-step s + 1 from the stored arrays
        assert np.array_equal(res.arrays["V_bc"], z[f"s{s + 1}_V_bc"])


def test_trajectory_is_deterministic_in_one_process(runs):
    again = tj.run_trajectory(CFG3)
    for a, b in zip(again.arrays, runs[3].arrays):
        assert all(np.array_equal(a[x], b[x]) for x in a)


def test_command_line_runs_small_and_refuses_d1(tmp_path):
    out = tmp_path / "traj"
    res = subprocess.run([sys.executable, str(SCRIPT), "run", "--lx", "1", "--nx", "8", "--ly", "1.25", "--ny", "10", "--N", "2",
                          "--s-max", "2", "--out", str(out), "--independent-check"], cwd=ROOT, capture_output=True, text=True,
                         timeout=600)
    assert res.returncode == 0, res.stderr[-2000:]
    steps = json.loads((out / "journal.json").read_text(encoding="utf-8"))["steps"]
    assert [s["status"] for s in steps] == [tj.OK, tj.OK] and all(s["independent_check"]["status"] == "CONSISTENT" for s in steps)
    for nx, ny in ((14, 6), (50, 50), (320, 1280), (400, 1600)):
        ref = subprocess.run([sys.executable, str(SCRIPT), "run", "--lx", "1", "--nx", str(nx), "--ly", "1", "--ny", str(ny), "--N", "1",
                              "--out", str(tmp_path / "d1")], cwd=ROOT, capture_output=True, text=True, timeout=600)
        assert ref.returncode == 2 and "D1_START = BLOCKED" in ref.stderr
    assert not (tmp_path / "d1").exists()


def test_command_line_refuses_output_under_docs(capsys):
    """N-6: the command line does not write under docs/ (checked before any computation)."""
    target = ROOT / "docs" / "op8_tool_output_must_not_exist"
    assert tj.main(["run", "--lx", "1", "--nx", "8", "--ly", "1", "--ny", "10", "--N", "1", "--out", str(target)]) == 2
    assert "not written under docs/" in capsys.readouterr().err and not target.exists()


# --- protection: no import of the working kkvim, used modules unchanged ------------------------------------------------------

def test_no_import_of_the_working_kkvim():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    names = [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
    names += [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    assert not [m for m in names if m == "kkvim" or m.startswith("kkvim.")]
    code = ("import importlib.util, sys; s = importlib.util.spec_from_file_location('t', sys.argv[1]);"
            " m = importlib.util.module_from_spec(s); s.loader.exec_module(m); m._ca(); m.run_trajectory(m.TrajectoryConfig("
            "1.0, 8, 1.25, 10, 2, s_max=2), independent=True);"
            " print(sorted(k for k in sys.modules if k == 'kkvim' or k.startswith('kkvim.')));"
            " print(sorted(m.loaded_modules_check()))")
    res = subprocess.run([sys.executable, "-c", code, str(SCRIPT)], cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert res.returncode == 0 and res.stdout.split("\n")[:2] == ["[]", "[]"], (res.stdout, res.stderr[-2000:])  # N-5: all listed


def test_used_modules_and_data_unchanged():
    assert all(v["unchanged"] for v in tj.used_modules_check().values()), tj.used_modules_check()
    for rel in ("src/kkvim", "src/kkvim_reference", "docs/results", "scripts/r5_cccc_q4_c_assembly_e4_audit.py",
                "scripts/r5_cccc_q4_energy_bc_audit.py", "scripts/r5_cccc_q4_bc_reference_audit.py",
                "scripts/r5_cccc_q3_prequalification.py"):
        out = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", rel], cwd=ROOT)
        assert out.returncode == 0, rel
