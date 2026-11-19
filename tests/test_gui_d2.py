"""Independent CCCC D2 in the GUI (GUI-MULTIMETHOD-CCCC-D2-01), without Qt.

- the D2 worker (``kkvim_d2_worker``) against ``kkvim_reference.r5_cccc_d2.solve`` called directly: the same field and
  centre bitwise, D = ``navier.flexural_rigidity``; the retained controls of the saved data reproduced bitwise;
- normalization: beta_a = w_c D/(q a^4) (length a) next to the reference's alpha with the shorter side; for b < a they
  differ, and the transposed plate gives the same dimensional centre deflection within the D2-8 model;
- refusals: SSSS, too large a grid (engineering limit, no computation), invalid input, wrong contract; odd grid: no
  centre value (nothing interpolated);
- the job in a real separate process: answer, timings, an unavailable worker, a timeout, no process left behind;
- the comparison: the D2 row computed for this run next to the unchanged saved rows, or alone when no saved data apply;
- the package: d2_field.csv, d2_result.json with SHA256 in the manifest, comparison.csv per-row origin, the report sections;
- the load series: D2 per q, its curve only from computed points, the export columns;
- O-1: a computing process that dies without a message gives a CRASH outcome, never a BrokenPipeError.
"""

import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import kkvim_d2_worker
from kkvim_gui.backend import benchmarks, d2_job, kkvim_adapter, report, series_export
from kkvim_gui.backend.process import CRASH, SolverProcess
from kkvim_gui.backend.request import RunRequest
from kkvim_gui.model import series as series_model
from kkvim_reference import navier, r5_cccc_d2

U = 2.0 ** -53
E, T, NU, Q = 210e9, 0.01, 0.3, 1000.0


def request(a=1.0, b=1.0, n_x=32, n_y=32, q=Q, N=1, s=8, supports="CCCC"):
    return RunRequest(entered=(), a=a, b=b, t=T, E=E, nu=NU, q=q, n_x=n_x, n_y=n_y, N=N, supports=supports,
                      half_steps=s if supports == "CCCC" else None,
                      **({} if supports == "CCCC" else {"tol_inf": 1e-10, "tol_E": 1e-8, "s_min": 3, "s_max": 100}))


def worker(req):
    return kkvim_d2_worker.compute(d2_job.worker_input(req))


def record(req):
    """The D2 record of a run, from the worker contract evaluated in this process (the real process: below)."""
    return d2_job.d2_record(d2_job.parse_output(json.dumps(worker(req)), d2_job.worker_input(req)), req)


@pytest.fixture(scope="module")
def square():
    req = request()
    return req, kkvim_adapter.execute(req), record(req)


# --- the worker against the reference called directly -----------------------------------------------------------------

@pytest.mark.parametrize(("a", "b", "n_x", "n_y"), [(1.0, 1.0, 32, 32), (1.0, 1.4, 32, 32), (1.4, 1.0, 16, 12),
                                                    (0.37, 1.13, 6, 9)])
def test_worker_equals_the_reference_called_directly(a, b, n_x, n_y):
    out = worker(request(a, b, n_x, n_y))
    assert out["state"] == "ok", out["reason"]
    D = navier.flexural_rigidity(E, T, NU)
    sol = r5_cccc_d2.solve(r5_cccc_d2.D2Problem(a, b, n_x, n_y, D, Q))
    assert out["D"] == D
    assert np.array_equal(np.array(out["result"]["field_m"]), sol.field())
    assert out["result"]["w_centre_m"] == sol.centre()
    assert out["result"]["e_D2_m"] == sol.e_D2 and out["result"]["corrections_m"] == sol.corrections
    assert out["problem"] == {"lx": a, "ly": b, "nx": n_x, "ny": n_y, "D": D, "q": Q, "unknowns": (n_x - 1) * (n_y - 1),
                              "ordering": "MMD_AT_PLUS_A"}
    assert set(out["timing"]) == {"import_s", "solve_s", "post_s", "worker_s"}
    assert all(v >= 0 for v in out["timing"].values())


@pytest.mark.parametrize(("b", "beta", "w"), [(1.0, 0.0012759787033572092, 6.635089257457487e-05),
                                              (1.4, 0.002084859282139107, 0.00010841268267123354)])
def test_retained_controls_are_reproduced_bitwise(b, beta, w):
    """The saved D2 values of benchmarks_v1.json (32 x 32) recomputed by the user-facing worker."""
    res = worker(request(b=b))["result"]
    assert res["beta_a"] == beta and res["w_centre_m"] == w
    stored = next(g["D2"] for c in benchmarks.load()["cccc"] if c["b"] == b for g in c["grids"] if g["n_x"] == 32)
    assert (stored["beta"], stored["w_centre_m"]) == (beta, w)


def test_normalization_uses_the_length_a_and_never_the_shorter_side_alpha():
    """b < a: alpha (shorter side b) and beta_a differ by (a/b)^4; the dimensional deflection is what is compared."""
    res = worker(request(a=1.4, b=1.0, n_x=28, n_y=20))["result"]
    D = navier.flexural_rigidity(E, T, NU)
    w = res["w_centre_m"]
    assert res["beta_a"] == w * D / (Q * 1.4 ** 4)
    assert res["alpha_short_side"] == w * D / (Q * 1.0 ** 4) and res["short_side_m"] == 1.0
    assert res["alpha_short_side"] / res["beta_a"] == pytest.approx(1.4 ** 4, rel=4 * U)
    # the transposed plate (a <-> b, n_x <-> n_y): the same dimensional centre deflection (D2-8 model)
    t = worker(request(a=1.0, b=1.4, n_x=20, n_y=28))["result"]
    bound = 4 * U * max(abs(np.array(res["field_m"])).max(), abs(np.array(t["field_m"])).max()) + res["e_D2_m"] + t["e_D2_m"]
    assert abs(t["w_centre_m"] - w) <= bound
    assert t["beta_a"] != res["beta_a"]  # beta_a uses each plate's own a


def test_square_alpha_equals_beta():
    res = worker(request())["result"]
    assert res["alpha_short_side"] == res["beta_a"]


@pytest.mark.parametrize(("change", "state", "text"), [
    ({"supports": "SSSS"}, "rejected", "CCCC"),
    ({"n_x": 2}, "rejected", "nx"),
    ({"nu": 1.5}, "rejected", "nu"),
    ({"t": 0.0}, "rejected", "thickness"),
    ({"n_x": 1280, "n_y": 257}, "out_of_scope", "engineering limit"),
    ({"contract": "other/1"}, "rejected", "contract"),
    ({"n_x": 32.0}, "rejected", "integer"),
])
def test_refusals_are_explicit_and_carry_no_value(change, state, text):
    inputs = {**d2_job.worker_input(request()), **change}
    out = kkvim_d2_worker.compute(inputs)
    assert out["state"] == state and text in out["reason"]
    assert out["result"] is None


def test_out_of_scope_grid_is_refused_before_any_computation():
    out = kkvim_d2_worker.compute({**d2_job.worker_input(request()), "n_x": 1281, "n_y": 256})
    assert out["state"] == "out_of_scope" and "solve_s" not in out["timing"]
    assert kkvim_d2_worker.MAX_UNKNOWNS == 1279 * 255


def test_odd_grid_has_no_centre_value_and_nothing_is_interpolated():
    res = worker(request(n_x=15, n_y=16))["result"]
    assert res["w_centre_m"] is None and res["beta_a"] is None and res["centre"] is None
    assert "not a node" in res["centre_reason"]
    assert np.array(res["field_m"]).shape == (16, 17)


# --- the job in a separate process ------------------------------------------------------------------------------------

def wait(job, timeout=120.0):
    t0 = time.monotonic()
    while (rec := job.poll()) is None:
        assert time.monotonic() - t0 < timeout
        time.sleep(0.02)
    return rec


def test_job_runs_the_worker_in_a_separate_process():
    req = request(n_x=16, n_y=16)
    job = d2_job.D2Job(d2_job.worker_input(req))
    rec = wait(job)
    assert rec["state"] == "ok" and rec["inputs"] == d2_job.worker_input(req)
    assert job._process.returncode == 0
    t = rec["timing"]
    assert t["wall_s"] >= t["worker_s"] >= t["solve_s"] > 0 and t["parse_s"] >= 0
    d2 = d2_job.d2_record(rec, req)
    direct = r5_cccc_d2.solve(r5_cccc_d2.D2Problem(1.0, 1.0, 16, 16, navier.flexural_rigidity(E, T, NU), Q))
    assert np.array_equal(d2["field"], direct.field()) and not d2["field"].flags.writeable
    assert d2["centre"] == {"i": 8, "j": 8, "x": 0.5, "y": 0.5}
    assert "field" not in d2_job.plain(d2) and json.dumps(d2_job.plain(d2))


def test_unavailable_worker_is_reported_and_no_value_is_substituted():
    req = request(n_x=8, n_y=8)
    job = d2_job.D2Job(d2_job.worker_input(req), command=[sys.executable, "-I", "-m", "kkvim_d2_worker_unavailable"])
    rec = wait(job)
    assert rec["state"] == "failed" and "No module named" in rec["reason"]
    d2 = d2_job.d2_record(rec, req)
    assert d2["state"] == "failed" and "w_centre_m" not in d2 and "field" not in d2


def test_timeout_terminates_the_worker():
    req = request(n_x=8, n_y=8)
    job = d2_job.D2Job(d2_job.worker_input(req), command=[sys.executable, "-c", "import time; time.sleep(60)"], timeout=0.5)
    rec = wait(job, timeout=30)
    assert rec["state"] == "failed" and "не ответил" in rec["reason"]
    assert job._process.poll() is not None  # no process left behind


def test_terminate_leaves_no_process():
    job = d2_job.D2Job(d2_job.worker_input(request()), command=[sys.executable, "-c", "import time; time.sleep(60)"])
    assert job.terminate() and job._process.poll() is not None


@pytest.mark.parametrize("defect", ["contract", "inputs", "field"])
def test_a_defective_answer_is_a_failed_job(defect):
    req = request(n_x=8, n_y=8)
    sent = d2_job.worker_input(req)
    out = worker(req)
    if defect == "contract":
        out["contract"] = "other/1"
    elif defect == "inputs":
        out["inputs"] = {**sent, "q": 2000.0}
    else:
        out["result"]["field_m"] = out["result"]["field_m"][:-1]
    assert d2_job.parse_output(json.dumps(out), sent)["state"] == "failed"


def test_record_of_other_inputs_is_refused():
    rec = d2_job.parse_output(json.dumps(worker(request(n_x=8, n_y=8))), d2_job.worker_input(request(n_x=8, n_y=8)))
    assert d2_job.d2_record(rec, request(n_x=8, n_y=8, q=2000.0))["state"] == "failed"


# --- comparison ------------------------------------------------------------------------------------------------------

def test_comparison_adds_the_executed_row_next_to_the_unchanged_saved_rows(square):
    req, outcome, d2 = square
    saved = benchmarks.compare(outcome)
    c = benchmarks.compare(outcome, d2=d2)
    assert c.state == benchmarks.AVAILABLE and c.executed and c.saved and not c.live
    assert [r.reference.kind for r in c.rows] == ["D2_live", "published", "D2", "FEM"]
    assert c.rows[1:] == saved.rows  # the saved controls are not changed
    live, stored = c.rows[0], c.rows[2]
    assert live.reference.beta == stored.reference.beta == 0.0012759787033572092
    assert live.reference.w_centre_m == d2["w_centre_m"]
    # the KK-VIM value of this run, checked within the run (its last bits depend on the BLAS summation order: no frozen
    # literal across platforms, AGENTS.md section 4; D-025)
    w_c = float(outcome.w[c.centre["i"], c.centre["j"]])
    assert c.w_centre_m == w_c and c.beta == w_c * outcome.D / (req.q * req.a ** 4)
    assert live.delta_w_m == c.w_centre_m - d2["w_centre_m"] and live.beta_kkvim == c.beta
    assert all(r.beta_kkvim == c.beta for r in c.rows)
    assert live.delta == c.beta - live.reference.beta and live.rel_pct == live.delta / live.reference.beta * 100.0
    assert live.reference.title != stored.reference.title and "сохранённое" in stored.reference.title
    assert benchmarks.state_title(c) == benchmarks.MIXED_TITLE
    assert any("не является ошибкой континуальной модели" in n for n in c.notes)
    rec = benchmarks.as_record(c)
    assert rec["executed_reference_kinds"] == ["D2_live"]


def test_without_saved_data_only_the_executed_row_is_shown():
    req = request(q=2500.0, n_x=16, n_y=16)
    outcome = kkvim_adapter.execute(req)
    assert benchmarks.compare(outcome).state == benchmarks.UNAVAILABLE
    c = benchmarks.compare(outcome, d2=record(req))
    assert c.state == benchmarks.AVAILABLE and c.live and [r.reference.kind for r in c.rows] == ["D2_live"]
    assert c.missing[0][0] == "saved" and "q, Pa" in c.missing[0][1]
    assert benchmarks.state_title(c) == benchmarks.D2_LIVE_TITLE


def test_failed_pending_and_ssss_d2(square):
    req, outcome, _ = square
    failed = benchmarks.compare(outcome, d2={"state": "failed", "reason": "процесс D2 не дал ответа: x"})
    assert [r.reference.kind for r in failed.rows] == ["published", "D2", "FEM"]
    assert failed.missing[0] == ("D2_live", "процесс D2 не дал ответа: x")
    other = request(q=2500.0, n_x=16, n_y=16)
    pending = benchmarks.compare(kkvim_adapter.execute(other), d2=d2_job.pending_record())
    assert pending.state == benchmarks.PENDING and not pending.rows
    ssss = request(a=2.0, b=1.0, n_x=16, n_y=8, supports="SSSS")
    c = benchmarks.compare(kkvim_adapter.execute(ssss), d2=record(request()))
    assert not any(r.reference.kind == "D2_live" for r in c.rows)


def test_a_diagnostic_field_is_never_compared_with_d2():
    req = request(N=2, s=2, n_x=34, n_y=34)  # s = 2 above the grid limit 32: protective stop, diagnostic field
    outcome = kkvim_adapter.execute(req)
    assert outcome.diagnostic_only
    c = benchmarks.compare(outcome, d2=record(req))
    assert c.state == benchmarks.DIAGNOSTIC and not c.rows
    assert d2_job.field_difference(record(req), outcome) is None


def test_field_difference_is_signed_at_the_largest_node(square):
    _, outcome, d2 = square
    diff = d2_job.field_difference(d2, outcome)
    delta = outcome.w - d2["field"]
    assert diff["delta_w_m"] == float(delta[diff["i"], diff["j"]]) and abs(diff["delta_w_m"]) == float(np.abs(delta).max())


# --- package ---------------------------------------------------------------------------------------------------------

def test_package_keeps_d2_with_integrity_and_origin(square, tmp_path):
    req, outcome, d2 = square
    c = benchmarks.compare(outcome, d2=d2)
    timing = {"kkvim_import_s": 0.2, "kkvim_solver_s": 0.01, "kkvim_wall_s": 0.4}
    paths = report.write_package(outcome, tmp_path / "pkg", {"run_id": 1}, (), comparison=c, d2=d2, timing=timing)
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    ind = manifest["independent_d2"]
    assert ind["state"] == "ok" and ind["record"]["w_centre_m"] == d2["w_centre_m"]
    for name, digest in ind["files_sha256"].items():
        assert hashlib.sha256((tmp_path / "pkg" / name).read_bytes()).hexdigest() == digest
    assert set(ind["files_sha256"]) == {"d2_field.csv", "d2_result.json"}
    rows = [r for r in (tmp_path / "pkg" / "d2_field.csv").read_text(encoding="utf-8").splitlines() if not r.startswith("#")]
    table = list(csv.DictReader(rows))
    assert len(table) == 33 * 33
    assert np.array_equal(np.array([float(r["w_d2 [m]"]) for r in table]).reshape(33, 33), d2["field"])
    result = json.loads((tmp_path / "pkg" / "d2_result.json").read_text(encoding="utf-8"))
    assert result["beta_a"] == d2["beta_a"] and "field" not in result and "d2_solve_s" in result["timer_definitions"]
    comp = [r for r in paths["comparison"].read_text(encoding="utf-8").splitlines() if not r.startswith("#")]
    origin = {r["reference_kind"]: r["reference_origin"] for r in csv.DictReader(comp)}
    assert origin == {"D2_live": "executed", "published": "stored", "D2": "stored", "FEM": "stored"}
    html = paths["report"].read_text(encoding="utf-8")
    assert "6а. МКР — двумерный (D2)" in html and "6б. Время вычислений" in html and "d2_field.csv" in html
    assert html.count(outcome.science_text) == 2  # sections 5 and 6, as before; never in the D2 section
    timers = {t["timer"]: t for t in manifest["timing"]["timers"]}
    assert timers["kkvim_solver_s"]["seconds"] == 0.01 and "d2_solve_s" in timers and "d2_wall_s" in timers
    assert all(t["definition"] for t in timers.values())
    with pytest.raises(OSError):  # an existing package is never overwritten
        report.write_package(outcome, tmp_path / "pkg", {"run_id": 1}, (), comparison=c, d2=d2)


def test_package_states_a_missing_d2_without_files(square, tmp_path):
    _, outcome, _ = square
    d2 = d2_job.cancelled_record()
    paths = report.write_package(outcome, tmp_path / "pkg", {}, (), d2=d2)
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert manifest["independent_d2"]["state"] == "cancelled" and manifest["independent_d2"]["files_sha256"] == {}
    assert "d2_field" not in paths and "остановлен" in paths["report"].read_text(encoding="utf-8")


# --- load series -----------------------------------------------------------------------------------------------------

def fake_run(run_id, req, d2, status="finished", series_id=1, timing=None):
    outcome = kkvim_adapter.execute(req)
    return SimpleNamespace(run_id=run_id, status=status, outcome=outcome, d2=d2, timing=timing or {"kkvim_solver_s": 0.01},
                           navier=None, series_id=series_id)


def test_series_keeps_d2_per_point_and_only_computed_points(tmp_path):
    loads = (500.0, 1000.0, 2000.0)
    reqs = [request(q=q, n_x=16, n_y=16) for q in loads]
    runs = [fake_run(1, reqs[0], record(reqs[0])), fake_run(2, reqs[1], d2_job.cancelled_record()), None]
    manager = SimpleNamespace(get=lambda i: runs[i - 1])
    rec = SimpleNamespace(series_id=1, status="cancelled", started=__import__("datetime").datetime(2026, 11, 14),
                          snapshot=SimpleNamespace(with_d2=True), requests=tuple(reqs), run_ids=[1, 2], navier=None)
    data = series_model.series_data(rec, manager)
    assert data.d2 and [p.d2_state for p in data.points] == ["ok", "cancelled", "not_run"]
    assert [p.on_d2_curve for p in data.points] == [True, False, False]
    p = data.points[0]
    assert p.d2_delta_w_m == p.w_centre_m - p.d2_w_m and p.d2_rel_pct == p.d2_delta_w_m / p.d2_w_m * 100.0
    paths = series_export.write_series(data, tmp_path / "s")
    rows = list(csv.DictReader([r for r in paths["load_series"].read_text(encoding="utf-8").splitlines()
                                if not r.startswith("#")]))
    assert [r["d2_state"] for r in rows] == ["ok", "cancelled", "not_run"]
    assert float(rows[0]["w_centre_d2 [m]"]) == p.d2_w_m and rows[1]["w_centre_d2 [m]"] == ""
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert manifest["independent_d2"]["points_on_curve"] == 1
    for rec_ in manifest["files"].values():
        assert hashlib.sha256((tmp_path / "s" / rec_["file"]).read_bytes()).hexdigest() == rec_["sha256"]


def test_series_without_d2_keeps_the_previous_columns_empty(tmp_path):
    req = request(q=1000.0, n_x=16, n_y=16)
    runs = [fake_run(1, req, None)]
    rec = SimpleNamespace(series_id=1, status="finished", started=__import__("datetime").datetime(2026, 11, 14),
                          snapshot=SimpleNamespace(with_d2=False), requests=(req,), run_ids=[1], navier=None)
    data = series_model.series_data(rec, SimpleNamespace(get=lambda i: runs[i - 1]))
    assert not data.d2 and data.points[0].d2_state is None
    row = series_export.rows(data)[0]
    assert row[len(series_export.COLUMNS) - 12] == "not applicable"


# --- O-1: a computing process that dies without a message -------------------------------------------------------------

@pytest.fixture
def targets(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parent))  # the spawned child imports the target by name
    import process_targets

    return process_targets


def poll_until(process, timeout=60.0):
    t0 = time.monotonic()
    while (message := process.poll()) is None:
        assert time.monotonic() - t0 < timeout
        time.sleep(0.02)
    return message


@pytest.mark.parametrize(("name", "code"), [("die_at_once", 3), ("die_later", 7)])
def test_dead_process_without_a_message_is_a_crash_not_an_exception(targets, name, code):
    p = SolverProcess(None, target=getattr(targets, name))
    kind, text = poll_until(p)
    assert kind == CRASH and f"(код {code})" in text
    assert not p._process.is_alive() and p.exitcode == code
    assert p.poll() == (kind, text) and p.received is not None and p.received >= p.started


def test_killed_process_is_a_crash_and_leaves_nothing(targets):
    p = SolverProcess(None, target=targets.hang)
    time.sleep(0.5)
    assert p.poll() is None
    p._process.kill()  # an abnormal termination from outside, as an out-of-memory kill would be
    kind, text = poll_until(p)
    assert kind == CRASH and "без результата" in text and not p._process.is_alive()
    assert p.terminate()


def test_solver_timing_is_measured_inside_the_process():
    p = SolverProcess(request(n_x=16, n_y=16))
    kind, outcome = poll_until(p)
    assert kind == "ok" and set(outcome.timing) == {"import_s", "solver_s"}
    assert 0 < outcome.timing["solver_s"] and p.received - p.started > outcome.timing["solver_s"]
    assert math.isfinite(outcome.timing["import_s"])
