"""Tests of the QA tool of own B-C trajectories with the C-3+ assembly (scripts/r5_cccc_q4_d1_c3plus_trajectories.py; OP-8): own
sequences for N = 1, 2, 3 (orientation, I-4, QR of the prototype, transfer of the factors), the input of s + 1 is the B-C result of s
(no R-AC), C-3+ of the solved set after QR, only C replaced, no fallback to OLD on a failed premise, exact dependence stopped before QR,
zero load and zero field, the small control (14, 6), N = 3 (test configuration only), the independent exact / R-H check of small
half-steps, the journal and the restoration of the next input, the wrapper restoration, no import of the working kkvim and no change of
the used modules. Small deterministic configurations only; no D1 run."""

import ast
import contextlib
import importlib.util
import json
import subprocess
import sys
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
        assert rec["c3plus_theorem_exact"] == "PASS" and rec["c3plus_C_spd_exact"] is True
        assert rec["c3plus_premises_status"].startswith("ESTABLISHED") and rec["c3plus_premises"]["P-4"].startswith("ASSUMED")
        assert rec["C_assembly_applied"].startswith("C-3+")
        # the B-C solution is the prototype's solve_bc with only C replaced (recomputed independently here)
        solved = _halfstep(cfg, k + 1, arr["solved_set"])
        with ca.replaced_c(arr["C3plus"]):
            sol = bc.solve_bc(solved)
        assert np.array_equal(sol.s, arr["V_bc"].T.reshape(-1))


@pytest.mark.parametrize("cfg", [CFG1, CFG3])
def test_only_c_is_replaced(cfg):
    tr = tj.run_trajectory(cfg)
    for k, (rec, arr) in enumerate(zip(tr.records, tr.arrays)):
        solved = _halfstep(cfg, k + 1, arr["solved_set"])
        A, B, C_old, F = bc.bc_coefficients(solved)
        assert rec["only_C_replaced"] is True and rec["bc_coefficients_calls"] == 1 and rec["wrapper_restored"] is True
        assert (rec["A_sha"], rec["B_sha"], rec["F_sha"], rec["C_old_sha"]) == tuple(tj.array_sha(x) for x in (A, B, F, C_old))
    assert bc.bc_coefficients is tj._BC_COEFFICIENTS


def test_a_replacement_that_changes_a_is_detected(monkeypatch):
    @contextlib.contextmanager
    def bad(C_new):
        orig = bc.bc_coefficients

        def w(p):
            A, B, _, F = orig(p)
            return 2.0 * A, B, C_new, F

        bc.bc_coefficients = w
        try:
            yield
        finally:
            bc.bc_coefficients = orig

    monkeypatch.setattr(ca, "replaced_c", bad)
    tr = tj.run_trajectory(CFG2)
    assert tr.records[0]["status"].startswith(tj.FAIL_TOOL) and tr.records[0]["only_C_replaced"] is False
    assert len(tr.records) == 1 and tr.status.startswith("STOPPED at s = 1")


def test_wrapper_is_restored_on_exception_and_not_nested(monkeypatch):
    solved = _halfstep(CFG2, 1, bc.orthonormalise(_halfstep(CFG2, 1, hs.start_set(8, 2)))[0])
    C = ca.c3plus(solved)["C"]

    def raising(p, capacitance="modal"):
        raise RuntimeError("injected")

    monkeypatch.setattr(bc, "solve_bc", raising)
    with pytest.raises(RuntimeError, match="injected"):
        tj.solve_with_c(solved, C)
    assert bc.bc_coefficients is tj._BC_COEFFICIENTS
    assert tj._BC_LOCK.acquire(blocking=False)
    tj._BC_LOCK.release()
    tr = tj.run_trajectory(CFG2)
    assert tr.records[0]["status"].startswith(tj.FAIL_TOOL + ": exception RuntimeError") and len(tr.records) == 1
    assert bc.bc_coefficients is tj._BC_COEFFICIENTS
    monkeypatch.undo()
    monkeypatch.setattr(bc, "bc_coefficients", lambda p: tj._BC_COEFFICIENTS(p))  # a foreign replacement is refused
    with pytest.raises(RuntimeError, match="already replaced"):
        tj.solve_with_c(solved, C)


# --- E. a failed premise of C-3+ gives UNRESOLVED, never OLD ------------------------------------------------------------------

def _no_solve(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("solve_bc called although a premise is not established")

    monkeypatch.setattr(bc, "solve_bc", boom)


def test_injected_premise_failure_stops_without_old(monkeypatch):
    _no_solve(monkeypatch)
    monkeypatch.setattr(ca, "premises_c3plus", lambda parts: "P-3: injected")
    tr = tj.run_trajectory(CFG2)
    rec, arr = tr.records[0], tr.arrays[0]
    assert rec["status"].startswith(tj.UNRES_PREMISE) and "P-3: injected" in rec["status"]
    assert "V_bc" not in arr and "next_given" not in arr and len(tr.records) == 1
    assert "bc_status" not in rec and "C_assembly_applied" not in rec


def test_natural_subnormal_premise_failure(monkeypatch):
    _no_solve(monkeypatch)
    given = hs.start_set(8, 1) * 2.0**-1000  # X^2 below 2^-1022: Gram products may underflow
    res = tj.run_step(CFG1, 1, given, np.zeros((9, 1)))
    assert res.record["status"].startswith(tj.UNRES_PREMISE) and "P-3" in res.record["status"] and res.next_given is None


def test_rounding_probe_failure(monkeypatch):
    _no_solve(monkeypatch)
    good = tj.fp_environment_probe()
    assert all(good.values())
    monkeypatch.setattr(tj, "fp_environment_probe", lambda: {**good, "nearest_numpy": False})
    tr = tj.run_trajectory(CFG1)
    assert tr.records[0]["status"].startswith(tj.UNRES_PREMISE) and "P-1" in tr.records[0]["status"]


def test_theorem_violation_with_premises_is_fail(monkeypatch):
    _no_solve(monkeypatch)
    monkeypatch.setattr(ca, "check_c3plus", lambda C, Cx, k: {"elementwise": False, "frobenius": True, "max_ratio_sq": Fraction(4)})
    tr = tj.run_trajectory(CFG1)
    assert tr.records[0]["status"] == tj.FAIL_THEOREM and len(tr.records) == 1


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
    for k in ("git_commit", "tool_blob", "numpy", "scipy", "platform", "blas", "threads_env"):
        assert k in prov
    assert "not across BLAS platforms" in prov["reproducibility"]
    assert all(v["unchanged"] for v in prov["used_modules"].values())
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


# --- protection: no import of the working kkvim, used modules unchanged ------------------------------------------------------

def test_no_import_of_the_working_kkvim():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    names = [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
    names += [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    assert not [m for m in names if m == "kkvim" or m.startswith("kkvim.")]
    code = ("import importlib.util, sys; s = importlib.util.spec_from_file_location('t', sys.argv[1]);"
            " m = importlib.util.module_from_spec(s); s.loader.exec_module(m); m._ca();"
            " print(sorted(k for k in sys.modules if k == 'kkvim' or k.startswith('kkvim.')))")
    res = subprocess.run([sys.executable, "-c", code, str(SCRIPT)], cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert res.returncode == 0 and res.stdout.strip() == "[]", res.stderr[-2000:]


def test_used_modules_and_data_unchanged():
    assert all(v["unchanged"] for v in tj.used_modules_check().values()), tj.used_modules_check()
    for rel in ("src/kkvim", "src/kkvim_reference", "docs/results", "scripts/r5_cccc_q4_c_assembly_e4_audit.py",
                "scripts/r5_cccc_q4_energy_bc_audit.py", "scripts/r5_cccc_q4_bc_reference_audit.py",
                "scripts/r5_cccc_q3_prequalification.py"):
        out = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", rel], cwd=ROOT)
        assert out.returncode == 0, rel
