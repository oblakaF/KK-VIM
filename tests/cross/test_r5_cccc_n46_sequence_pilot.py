"""Targeted checks of the N = 4-6 research pilot scripts/r5_cccc_n46_sequence_pilot.py (task R5-CCCC-N46-FULL-SEQUENCE-PILOT-01).

They protect the working pilot only (patterns of tests/cross/test_r5_cccc_n23_sequence_pilot.py; no qualification is reopened):
routing of N = 4, 5, 6 and the authorized domain, s = 1 and s = 2 identical to the unchanged production ``run_cccc_n46_physical``,
orientation, dimensions and continuation beyond s = 2, the budget prefix, the paired QR / companion transformation, stops on an exact
dependence, a nonfinite set and a failed premise (before the solve), no continuation after a failed RN-5 or RT-7, the separate D2 process,
the N = 4-6 metadata (not D-066) with the completed fields kept, the unchanged production guard of s >= 3 and no convergence claim.
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
import kkvim.cccc_n23_certificate as certificate
import kkvim.cccc_n46 as c46

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "r5_cccc_n46_sequence_pilot.py"
PLATE = {"a": 1.0, "b": 1.0, "t": 0.01, "E": 2.1e11, "nu": 0.3, "q": 1000.0}
GRID = {4: 10, 5: 12, 6: 14}  # the smallest even grids >= 2N + 1 (D-067), inside the D-068 domain n <= 32
U = 2.0**-53
ROUNDING = 1e-10  # rounding-level identities of sums over <= 200 terms (fixed before the runs, as for N = 2-3)


def _load():
    spec = importlib.util.spec_from_file_location("r5_cccc_n46_sequence_pilot", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pilot = _load()


def _run(N=4, half_steps=5, n=None, **over):
    p = {**PLATE, **over}
    n = GRID[N] if n is None else n
    return pilot.run_pilot(p["a"], p["b"], p["t"], p["E"], p["nu"], p["q"], n, n, N, half_steps)


def _cli(N, half_steps, out, *extra):
    n = str(GRID[N])
    return pilot.main(["run", "--N", str(N), "--nx", n, "--ny", n, "--half-steps", str(half_steps), *extra, "--out", str(out)])


@pytest.fixture(scope="module")
def runs():
    return {N: _run(N=N, half_steps=5) for N in (4, 5, 6)}


# 1, 3 ------------------------------------------------------------------------------------------------------------------------------

def test_routing_orientation_dimensions_and_continuation(runs):
    for N, (status, reason, res, rows) in runs.items():
        assert status == pilot.RESEARCH_LIMIT_REACHED and reason is None and res.N == N
        assert [r.s for r in rows] == [1, 2, 3, 4, 5] and [r.direction for r in rows] == ["Y", "X", "Y", "X", "Y"]
        assert all(r.done and r.N == N and r.V.shape[1] == N for r in rows)
        for r in rows:
            x, y = (r.G, r.V) if r.direction == "Y" else (r.V, r.G)
            assert x.shape == (res.n_x - 1, N) and y.shape == (res.n_y - 1, N)
            assert r.n_s == (res.n_y if r.direction == "Y" else res.n_x)


def test_refused_inputs():
    for N, n, k in ((3, 12, 4), (7, 16, 4), (4, 33, 4), (4, 12, 0), (4, 12, 33), (True, 12, 4), (4, 12, 3.0)):
        with pytest.raises(ValueError):
            pilot.run_pilot(1.0, 1.0, 0.01, 2.1e11, 0.3, 1000.0, n, n, N, k)
    with pytest.raises(SystemExit):
        pilot.main(["run", "--N", "3", "--nx", "12", "--ny", "12", "--half-steps", "4", "--out", "unused"])


# 2 ---------------------------------------------------------------------------------------------------------------------------------

def test_s1_s2_identical_to_unchanged_production(runs):
    for N, (_, _, _, rows) in runs.items():
        n = GRID[N]
        prod = c46.run_cccc_n46_physical(*PLATE.values(), n, n, N, 2)
        assert prod.status == c46.COMPLETED_FIXED_HALF_STEPS
        for mine, theirs in zip(rows[:2], prod.steps, strict=True):
            for name in ("G", "V", "R"):
                assert np.array_equal(getattr(mine, name), getattr(theirs, name)), (N, mine.s, name)
            assert mine.P_hat == theirs.P_hat and mine.eta == theirs.eta
        assert rows[1].rn5["certified"] and rows[1].rt7["passed"]


# 4 ---------------------------------------------------------------------------------------------------------------------------------

def test_budget_prefix(runs):
    short = _run(N=4, half_steps=3)[3]
    for a, b in zip(short, runs[4][3][:3], strict=True):
        assert np.array_equal(a.G, b.G) and np.array_equal(a.V, b.V)
    assert _run(N=4, half_steps=1)[0] == pilot.RESEARCH_LIMIT_REACHED and len(_run(N=4, half_steps=1)[3]) == 1


# 5 ---------------------------------------------------------------------------------------------------------------------------------

def test_paired_qr_and_companion(runs):
    for N, (_, _, res, rows) in runs.items():
        for prev, r in itertools.pairwise(rows[1:]):
            h_g = res.a / res.n_x if r.direction == "Y" else res.b / res.n_y
            assert np.allclose(h_g * r.G.T @ r.G, np.eye(N), rtol=0, atol=64 * N * U * (r.G.shape[0] + 2))
            assert np.array_equal(r.given, prev.V)
            assert np.array_equal(r.companion, prev.G @ r.R.T)
            assert r.field_preservation < ROUNDING
            assert abs(r.P_hat_previous_QR - prev.P_hat) <= ROUNDING * abs(prev.P_hat)


# 6, 7 ------------------------------------------------------------------------------------------------------------------------------

def _prev(runs, N=5):
    _, _, res, rows = runs[N]
    return res, copy.deepcopy(rows[3])


def test_exact_dependence_and_nonfinite_stop_before_qr(runs):
    res, prev = _prev(runs)
    prev.V[:, 2] = 2.0 * prev.V[:, 0]  # exact in binary64
    prev.V_exact_rank = cn.exact_rank(prev.V)
    assert prev.V_exact_rank == 4
    step = pilot.p23.pilot_step(prev, 5, res.a, res.b, res.D, res.q, res.n_x, res.n_y)
    assert not step.done and step.status == cn.STOP_BEFORE_QR_S1 and step.qr_executed is False and step.G is None
    res, prev = _prev(runs)
    prev.V[2, 1] = np.inf
    step = pilot.p23.pilot_step(prev, 5, res.a, res.b, res.D, res.q, res.n_x, res.n_y)
    assert not step.done and step.status == cn.ERROR_NONFINITE and step.qr_executed is False


def test_failed_premise_stops_before_the_solve(runs, monkeypatch):
    res, prev = _prev(runs, N=6)
    monkeypatch.setattr(cn, "premise_failures", lambda parts, n: ["P-3: injected"])
    monkeypatch.setattr(cn, "solve_bc", lambda *a, **k: pytest.fail("solve after a failed premise"))
    step = pilot.p23.pilot_step(prev, 5, res.a, res.b, res.D, res.q, res.n_x, res.n_y)
    assert not step.done and step.status == cn.UNRESOLVED_C3PLUS_PREMISE and step.V is None


# 8 ---------------------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("guard", ["rn5", "rt7"])
def test_no_continuation_after_failed_initialization(guard, monkeypatch, tmp_path):
    if guard == "rn5":
        real = certificate.rn5_certificate
        monkeypatch.setattr(certificate, "rn5_certificate",
                            lambda *a, **k: {**real(*a, **k), "certified": False, "verdict": "INJECTED", "reason": "injected"})
    else:
        real = certificate.rt7_conditions
        monkeypatch.setattr(certificate, "rt7_conditions", lambda *a, **k: {**real(*a, **k), "passed": False, "reason": "injected"})
    monkeypatch.setattr(pilot.p23, "pilot_step", lambda *a, **k: pytest.fail("continued after a failed initialization"))
    status, reason, res, rows = _run(N=4, half_steps=6)
    assert status == pilot.PILOT_STOPPED and res.status == c46.STOPPED_RANK_GUARD_UNCERTIFIED
    assert reason.startswith("s = 2: STOPPED_RANK_GUARD_UNCERTIFIED") and "D-068" in reason
    assert [r.s for r in rows] == [1, 2] and rows[0].done and not rows[1].done
    assert _cli(4, 6, tmp_path / "out", "--no-compare") == 6
    r = json.loads((tmp_path / "out" / "result.json").read_text(encoding="utf-8"))
    assert r["status"] == pilot.PILOT_STOPPED and r["half_steps_completed"] == 1
    assert np.load(tmp_path / "out" / "fields.npz").files == ["w_s1"]  # the completed field of s = 1 is kept


# 9 ---------------------------------------------------------------------------------------------------------------------------------

def test_compare_runs_in_a_process_without_kkvim(tmp_path):
    assert _cli(4, 3, tmp_path, "--no-compare") == 0
    done = subprocess.run([sys.executable, str(SCRIPT), "compare", "--out", str(tmp_path)], cwd=ROOT, capture_output=True, text=True,
                          check=False, env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"})
    assert done.returncode == 0, done.stderr
    cmp = json.loads((tmp_path / "comparison.json").read_text(encoding="utf-8"))
    assert cmp["kkvim_imported_in_this_process"] is False and cmp["reference"]["kkvim_imported_in_this_process"] is False
    assert cmp["reference"]["D_bitwise_equal"] is True and cmp["N"] == 4 and cmp["task"] == pilot.TASK
    assert cmp["marking"] == pilot.MARKING and [r["s"] for r in cmp["rows"]] == [1, 2, 3]
    assert set(json.loads((tmp_path / "artifacts.sha256.json").read_text(encoding="utf-8"))) == set(pilot.ARTIFACTS)


def test_paths_import_only_their_package():
    for path in (SCRIPT, pilot.N23_PILOT):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        top = {alias.name.split(".")[0] for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))
               for alias in (node.names if isinstance(node, ast.Import) else [ast.alias(node.module)])}
        assert not top & {"kkvim", "kkvim_reference"}, path
    imports = {}  # function -> top-level packages it imports
    for func in ast.parse(SCRIPT.read_text(encoding="utf-8")).body:
        if isinstance(func, ast.FunctionDef):
            imports[func.name] = {alias.name.split(".")[0] for node in ast.walk(func) if isinstance(node, (ast.Import, ast.ImportFrom))
                                  for alias in (node.names if isinstance(node, ast.Import) else [ast.alias(node.module or "")])}
    assert {name for name, packages in imports.items() if "kkvim" in packages} == {"run_pilot"}
    assert not any("kkvim_reference" in packages for packages in imports.values())  # the reference only via p23.command_compare


# 10, 12 ----------------------------------------------------------------------------------------------------------------------------

def test_metadata_and_no_convergence_claim(tmp_path):
    assert _cli(5, 4, tmp_path / "a", "--no-compare") == 0
    r = json.loads((tmp_path / "a" / "result.json").read_text(encoding="utf-8"))
    assert r["task"] == pilot.TASK and r["status"] == pilot.RESEARCH_LIMIT_REACHED and r["half_steps_completed"] == 4
    assert "RESEARCH_ONLY / PROVISIONAL / NOT_QUALIFIED" in r["marking"] and r["convergence"].startswith("NOT ASSESSED")
    assert "D-066" not in r["decision"] and "D-066" not in r["authorization"] and "D-068" in r["authorization"]
    assert [h["rank_status"] for h in r["history"]] == [pilot.RANK_STATUS[1], pilot.RANK_STATUS[2], pilot.RANK_S3, pilot.RANK_S3]
    assert "D-063" not in json.dumps(r["history"]) and "D-065)" not in json.dumps([h["rank_status"] for h in r["history"]])
    text = json.dumps(r).replace("no CONVERGED status", "")
    assert "CONVERGED" not in text and "QUALIFIED" not in text.replace("NOT_QUALIFIED", "")
    assert r["provenance"]["numpy_blas"] and len(r["provenance"]["git_commit"]) == 40
    assert _cli(5, 4, tmp_path / "a", "--no-compare") == 2  # a populated output directory is never overwritten
    assert _cli(5, 4, tmp_path / "b", "--no-compare") == 0
    fa, fb = np.load(tmp_path / "a" / "fields.npz"), np.load(tmp_path / "b" / "fields.npz")
    assert fa.files == ["w_s1", "w_s2", "w_s3", "w_s4"] and all(np.array_equal(fa[k], fb[k]) for k in fa.files)


def test_protective_stop_keeps_the_completed_fields(monkeypatch, tmp_path):
    real = cn.premise_failures
    calls = {"n": 0}

    def fourth_fails(parts, n):
        calls["n"] += 1
        return ["P-2: injected"] if calls["n"] == 4 else real(parts, n)

    monkeypatch.setattr(cn, "premise_failures", fourth_fails)
    assert _cli(6, 8, tmp_path, "--no-compare") == 6
    r = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert r["status"] == pilot.PILOT_STOPPED and r["reason"].startswith("s = 4: " + cn.UNRESOLVED_C3PLUS_PREMISE)
    assert r["half_steps_completed"] == 3 and r["last_completed_s"] == 3 and len(r["history"]) == 4
    assert np.load(tmp_path / "fields.npz").files == ["w_s1", "w_s2", "w_s3"]


def test_summary_marks_unavailable_values(tmp_path):
    assert _cli(4, 3, tmp_path / "run", "--no-compare") == 0
    assert pilot.main(["summary", "--out", str(tmp_path / "sum"), str(tmp_path / "run")]) == 0
    s = json.loads((tmp_path / "sum" / "summary.json").read_text(encoding="utf-8"))["runs"][0]
    assert s["w_centre_s2"] is not None and s["w_centre_s8"] is None and s["w_centre_s32"] is None
    assert s["completion_kind"] == "fixed-budget completion" and s["rel_max_vs_D2_s1"] is None  # no comparison was run
    assert (tmp_path / "sum" / "w_centre_10x10.svg").read_text(encoding="utf-8").startswith("<svg")


# 11 --------------------------------------------------------------------------------------------------------------------------------

def test_production_guard_of_s3_unchanged():
    res = c46.run_cccc_n46_physical(*PLATE.values(), 10, 10, 4, 8)
    assert res.status == c46.STOPPED_RANK_GUARD_UNCERTIFIED and len(res.steps) == 3
    assert res.steps[2].qr_executed is False and res.steps[2].G is None and res.steps[2].V is None
    for path in Path(c46.__file__).parent.glob("*.py"):
        assert "sequence_pilot" not in path.read_text(encoding="utf-8")
