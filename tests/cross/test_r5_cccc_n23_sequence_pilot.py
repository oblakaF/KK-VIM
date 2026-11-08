"""Targeted checks of the D-066 pilot scripts/r5_cccc_n23_sequence_pilot.py (task R5-CCCC-N23-FULL-SEQUENCE-PILOT).

They protect the working pilot only: the Y/X alternation, s = 1 and s = 2 identical to production, the dimensions, the paired QR
transformation, refusals of an exact dependence, a nonfinite set and a failed premise, reproducible output with provenance, the separation
of the two execution paths and the unchanged production guard of s >= 3. The orientation of a non-square plate is checked against the
energy of the independent D2 matrix (the test may import both packages; the pilot's paths may not).
"""

import ast
import copy
import importlib.util
import itertools
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import kkvim.cccc_n23 as cn
from kkvim_reference import r5_cccc_d2 as d2

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "r5_cccc_n23_sequence_pilot.py"
PLATE = {"a": 1.0, "b": 0.8, "t": 0.01, "E": 2.1e11, "nu": 0.3, "q": 1000.0, "n_x": 12, "n_y": 10}
U = 2.0**-53
ROUNDING = 1e-10  # rounding-level identities of sums over <= 200 terms (fixed before the runs)


def _load():
    spec = importlib.util.spec_from_file_location("r5_cccc_n23_sequence_pilot", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pilot = _load()


def _run(N=2, half_steps=5, **over):
    p = {**PLATE, **over}
    return pilot.run_pilot(p["a"], p["b"], p["t"], p["E"], p["nu"], p["q"], p["n_x"], p["n_y"], N, half_steps)


@pytest.fixture(scope="module")
def run2():
    return _run(N=2, half_steps=5)


def test_alternation_and_completion(run2):
    status, reason, _, rows = run2
    assert status == pilot.RESEARCH_LIMIT_REACHED and reason is None
    assert [r.s for r in rows] == [1, 2, 3, 4, 5]
    assert [r.direction for r in rows] == ["Y", "X", "Y", "X", "Y"]
    assert all(r.done and r.status == cn.HALF_STEP_DONE for r in rows)


def test_dimensions(run2):
    _, _, res, rows = run2
    for r in rows:
        x, y = (r.G, r.V) if r.direction == "Y" else (r.V, r.G)
        assert x.shape == (res.n_x - 1, r.N) and y.shape == (res.n_y - 1, r.N)
        assert r.n_s == (res.n_y if r.direction == "Y" else res.n_x)


def test_s1_s2_identical_to_production(run2):
    rows = run2[3]
    prod = cn.run_cccc_n23_physical(PLATE["a"], PLATE["b"], PLATE["t"], PLATE["E"], PLATE["nu"], PLATE["q"], PLATE["n_x"], PLATE["n_y"], 2, 2)
    assert prod.status == cn.COMPLETED_FIXED_HALF_STEPS
    for mine, theirs in zip(rows[:2], prod.steps, strict=True):
        assert np.array_equal(mine.G, theirs.G) and np.array_equal(mine.V, theirs.V)
        assert mine.P_hat == theirs.P_hat and mine.eta == theirs.eta


def test_generic_step_reproduces_production_s2_bitwise(run2):
    """``pilot_step`` at s = 2 (without the RN-5/RT-7 guards, which change no array) gives the production G2, companion and X."""
    _, _, res, rows = run2
    step = pilot.pilot_step(rows[0], 2, res.a, res.b, res.D, res.q, res.n_x, res.n_y)
    assert step.done
    for name in ("G", "R", "companion", "V", "A", "B", "C", "F"):
        assert np.array_equal(getattr(step, name), getattr(rows[1], name)), name
    assert step.P_hat == rows[1].P_hat and step.P_hat_previous_QR == rows[1].P_hat_previous_QR


def test_paired_qr_transformation(run2):
    _, _, res, rows = run2
    for prev, r in itertools.pairwise(rows[1:]):
        h_g = res.a / res.n_x if r.direction == "Y" else res.b / res.n_y
        assert np.allclose(h_g * r.G.T @ r.G, np.eye(r.N), rtol=0, atol=64 * r.N * U * (r.G.shape[0] + 2))
        assert np.array_equal(r.given, prev.V)
        assert np.array_equal(r.companion, prev.G @ r.R.T)
        assert r.field_preservation < ROUNDING
        assert abs(r.P_hat_previous_QR - prev.P_hat) <= ROUNDING * abs(prev.P_hat)


def test_orientation_by_the_independent_d2_energy(run2):
    """P_hat of every field = 1/2 D h_x h_y x^T A x - q h_x h_y sum x with the D2 nodewise matrix (Proposition D) on a 1 x 0.8, 12 x 10 plate."""
    _, _, res, rows = run2
    problem = d2.D2Problem(res.a, res.b, res.n_x, res.n_y, res.D, res.q)
    A = d2.assemble(problem)
    hx, hy = res.a / res.n_x, res.b / res.n_y
    for r in rows:
        x, y = (r.G, r.V) if r.direction == "Y" else (r.V, r.G)
        v = (x @ y.T).ravel()  # k = (i - 1)(n_y - 1) + (j - 1), x index slow, as D2
        energy = 0.5 * res.D * hx * hy * float(v @ (A @ v)) - res.q * hx * hy * float(v.sum())
        assert abs(energy - r.P_hat) <= ROUNDING * abs(r.P_hat), r.s


def _prev(run2):
    _, _, res, rows = run2
    return res, copy.deepcopy(rows[3])


def test_exact_dependence_stops_before_qr(run2):
    res, prev = _prev(run2)
    prev.V[:, 1] = 2.0 * prev.V[:, 0]
    prev.V_exact_rank = cn.exact_rank(prev.V)
    assert prev.V_exact_rank == 1
    step = pilot.pilot_step(prev, 5, res.a, res.b, res.D, res.q, res.n_x, res.n_y)
    assert not step.done and step.status == cn.STOP_BEFORE_QR_S1 and step.qr_executed is False and step.G is None


def test_nonfinite_set_stops_before_qr(run2):
    res, prev = _prev(run2)
    prev.V[3, 0] = np.nan
    step = pilot.pilot_step(prev, 5, res.a, res.b, res.D, res.q, res.n_x, res.n_y)
    assert not step.done and step.status == cn.ERROR_NONFINITE and step.qr_executed is False


def test_failed_premise_stops_without_solve(run2, monkeypatch):
    res, prev = _prev(run2)
    monkeypatch.setattr(cn, "premise_failures", lambda parts, n: ["P-2: injected"])
    monkeypatch.setattr(cn, "solve_bc", lambda *a, **k: pytest.fail("solve after a failed premise"))
    step = pilot.pilot_step(prev, 5, res.a, res.b, res.D, res.q, res.n_x, res.n_y)
    assert not step.done and step.status == cn.UNRESOLVED_C3PLUS_PREMISE and step.V is None


def test_stop_preserves_completed_rows(monkeypatch):
    real = cn.premise_failures
    calls = {"n": 0}

    def fourth_fails(parts, n):
        calls["n"] += 1
        return ["P-2: injected"] if calls["n"] == 4 else real(parts, n)

    monkeypatch.setattr(cn, "premise_failures", fourth_fails)
    status, reason, _, rows = _run(N=2, half_steps=6)
    assert status == pilot.PILOT_STOPPED and reason.startswith("s = 4: " + cn.UNRESOLVED_C3PLUS_PREMISE)
    assert [r.done for r in rows] == [True, True, True, False]


def test_reproducible_output_and_provenance(tmp_path):
    args = ["run", "--N", "2", "--nx", "12", "--ny", "12", "--half-steps", "3", "--no-compare"]
    assert pilot.main([*args, "--out", str(tmp_path / "a")]) == 0
    assert pilot.main([*args, "--out", str(tmp_path / "b")]) == 0
    ra, rb = (json.loads((tmp_path / d / "result.json").read_text(encoding="utf-8")) for d in "ab")
    assert ra["fields_sha256"] == rb["fields_sha256"]
    fa, fb = np.load(tmp_path / "a" / "fields.npz"), np.load(tmp_path / "b" / "fields.npz")
    assert fa.files == fb.files == ["w_s1", "w_s2", "w_s3"]
    assert all(np.array_equal(fa[k], fb[k]) for k in fa.files)
    assert "RESEARCH_ONLY / PROVISIONAL / NOT_QUALIFIED" in ra["marking"] and ra["decision"] == "D-066"
    assert len(ra["provenance"]["git_commit"]) == 40 and ra["history"][2]["rank_status"] == pilot.RANK_S3
    assert (tmp_path / "a" / "history.csv").read_text(encoding="utf-8").startswith("s,direction,status")


def test_compare_runs_in_a_process_without_kkvim(tmp_path):
    assert pilot.main(["run", "--N", "2", "--nx", "12", "--ny", "12", "--half-steps", "2", "--no-compare", "--out", str(tmp_path)]) == 0
    done = subprocess.run([sys.executable, str(SCRIPT), "compare", "--out", str(tmp_path)], cwd=ROOT, capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stderr
    cmp = json.loads((tmp_path / "comparison.json").read_text(encoding="utf-8"))
    assert cmp["reference"]["kkvim_imported_in_this_process"] is False and cmp["reference"]["D_bitwise_equal"] is True
    assert [r["s"] for r in cmp["rows"]] == [1, 2]


def test_paths_import_only_their_package():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    top = {alias.name.split(".")[0] for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))
           for alias in (node.names if isinstance(node, ast.Import) else [ast.alias(node.module)])}
    assert not top & {"kkvim", "kkvim_reference"}
    funcs = {node.name: ast.dump(node) for node in tree.body if isinstance(node, ast.FunctionDef)}
    for name, dump in funcs.items():
        if "kkvim_reference" in dump:
            assert name == "command_compare"
        if name == "command_compare":
            assert "'kkvim.cccc" not in dump and "'kkvim.c" not in dump


def test_production_guard_of_s3_unchanged():
    res = cn.run_cccc_n23_physical(1.0, 1.0, 0.01, 2.1e11, 0.3, 1000.0, 12, 12, 2, 8)
    assert res.status == cn.STOPPED_RANK_GUARD_UNCERTIFIED and len(res.steps) == 3
    assert res.steps[2].qr_executed is False and res.steps[2].G is None and res.steps[2].V is None
    for path in Path(cn.__file__).parent.glob("*.py"):
        assert "sequence_pilot" not in path.read_text(encoding="utf-8")


def test_run_is_the_default_sub_command(tmp_path):
    assert pilot.main(["--N", "2", "--nx", "12", "--ny", "12", "--half-steps", "1", "--no-compare", "--out", str(tmp_path)]) == 0
    assert json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))["half_steps_completed"] == 1


# --- iteration controller (task R5-CCCC-N23-ITERATION-CONTROLLER) ---------------------------------------------------------------------

@pytest.fixture(scope="module")
def run11():
    return _run(N=3, half_steps=11)


def test_alternation_continues_beyond_s8_and_budget_ends_the_run(run11):
    status, reason, _, rows = run11
    assert status == pilot.RESEARCH_LIMIT_REACHED and reason is None
    assert [r.s for r in rows] == list(range(1, 12))
    assert [r.direction for r in rows] == ["Y", "X"] * 5 + ["Y"]
    assert all(r.done and r.N == 3 and r.V.shape[1] == 3 for r in rows)


def test_first_eight_steps_do_not_depend_on_the_budget(run11):
    rows8 = _run(N=3, half_steps=8)[3]
    for short, long in zip(rows8, run11[3][:8], strict=True):
        for name in ("G", "V", "R", "companion"):
            a, b = getattr(short, name), getattr(long, name)
            assert (a is None and b is None) or np.array_equal(a, b), (short.s, name)


def test_continuation_state_beyond_s8(run11):
    rows = run11[3]
    for prev, r in itertools.pairwise(rows[7:]):
        assert np.array_equal(r.given, prev.V) and np.array_equal(r.companion, prev.G @ r.R.T)
        assert r.field_preservation < ROUNDING and r.repeated_state_of is None


def test_budget_bounds():
    for bad in (0, 33, True, 8.0):
        with pytest.raises(ValueError):
            _run(half_steps=bad)
    with pytest.raises(SystemExit):
        pilot.main(["--N", "2", "--nx", "12", "--ny", "12", "--half-steps", "33", "--out", "unused"])


def test_repeated_state_needs_bitwise_equality_of_the_same_phase(run11):
    rows = run11[3]
    twin = copy.deepcopy(rows[6])
    twin.s = 9
    assert pilot.repeated_state(rows[:8] + [twin], twin) == 7
    twin.V[0, 0] = np.nextafter(twin.V[0, 0], np.inf)
    assert pilot.repeated_state(rows[:8] + [twin], twin) is None
    other_phase = copy.deepcopy(rows[6])
    other_phase.direction, other_phase.s = "X", 9
    assert pilot.repeated_state(rows[:8] + [other_phase], other_phase) is None


def test_repeated_state_stops_the_run(monkeypatch):
    real = pilot.pilot_step
    seen = []

    def repeating(prev, s, *args):
        step = real(prev, s, *args)
        seen.append(step)
        if s == 6:  # the state of s = 4 (same phase X) once more
            step = copy.deepcopy(seen[-3])
            step.s, step.k = 6, 5
        return step

    monkeypatch.setattr(pilot, "pilot_step", repeating)
    status, reason, _, rows = _run(N=2, half_steps=10)
    assert status == pilot.STOPPED_BITWISE_REPEATED_STATE and "s = 6" in reason and "s = 4" in reason
    assert [r.s for r in rows] == [1, 2, 3, 4, 5, 6] and rows[-1].repeated_state_of == 4 and rows[-1].done


def test_no_convergence_label(tmp_path):
    assert pilot.main(["--N", "2", "--nx", "12", "--ny", "12", "--half-steps", "9", "--no-compare", "--out", str(tmp_path)]) == 0
    r = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert r["status"] == pilot.RESEARCH_LIMIT_REACHED and r["convergence"].startswith("NOT ASSESSED")
    assert {pilot.RESEARCH_LIMIT_REACHED, pilot.STOPPED_BITWISE_REPEATED_STATE, pilot.PILOT_STOPPED} == set(r["stops"])
    text = json.dumps(r)
    assert "CONVERGED" not in text.replace("no CONVERGED status", "")
    h = r["history"]
    assert [x["direction"] for x in h] == ["Y", "X"] * 4 + ["Y"]
    assert h[1]["same_phase_change_max"] is None and h[8]["same_phase_change_max"] > 0
    assert h[8]["rank_status"] == pilot.RANK_S3 and "C-3+ premises" in h[8]["guards"]
