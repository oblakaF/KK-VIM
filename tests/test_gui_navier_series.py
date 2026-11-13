"""Navier reference worker, live SSSS comparison, load series and their exports (GUI-PLATE-WORKFLOW-UX-01). Qt-free.

- the worker evaluates exactly the qualified reference (kkvim_reference.navier.NavierRectangle) and reproduces the
  saved square value bit for bit; it receives the plate input only and answers outside its scope with a reason;
- the GUI starts it as a separate process and never imports it or kkvim_reference;
- the live comparison uses the centre node of the KK-VIM field (no interpolation), signed differences, no threshold;
  diagnostic fields, pending, failed or out-of-scope references give no numbers;
- report.html and comparison.csv carry the same values; the scientific status is written once;
- a load series is one real run per q (only q differs), every point is checked first, diagnostic runs are not points
  of the curve, the export never overwrites a package and records SHA256 of its files.
"""

import csv
import dataclasses
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from kkvim_gui.backend import benchmarks as bm
from kkvim_gui.backend import kkvim_adapter, navier_job, report, series_export
from kkvim_gui.backend.outcome import ROLE_ACCEPTED, ROLE_DIAGNOSTIC
from kkvim_gui.backend.request import RunRequest
from kkvim_gui.model import series as series_model
from kkvim_gui.model.setup_state import SetupState, parse_series, to_request, to_series_requests, validate

OWNER = dict(a=2.0, b=1.0, t=0.01, E=2.1e11, nu=0.3, q=5000.0)


def request(**kw):
    base = dict(entered=(), **OWNER, n_x=16, n_y=8, N=1, supports="SSSS", tol_inf=kkvim_adapter.DEFAULTS["tol_inf"],
                tol_E=kkvim_adapter.DEFAULTS["tol_E"], s_min=kkvim_adapter.DEFAULTS["s_min"],
                s_max=kkvim_adapter.DEFAULTS["s_max"])
    return RunRequest(**{**base, **kw})


@pytest.fixture(scope="module")
def r3():
    """A real accepted R3 run of the owner's plate on a small grid (centre node exists)."""
    return kkvim_adapter.execute(request())


@pytest.fixture(scope="module")
def navier_record():
    """The job record of the real worker process for the owner's plate at two loads."""
    job = navier_job.NavierJob(navier_job.worker_input(request(), [1000.0, 5000.0]))
    while job.poll() is None:
        pass
    return job.record


# --- worker ------------------------------------------------------------------------------------------------------------

def test_worker_evaluates_exactly_the_qualified_reference():
    from kkvim_navier_worker import CONTRACT, compute
    from kkvim_reference.navier import M_R3, NavierRectangle

    out = compute({"contract": CONTRACT, **{k: OWNER[k] for k in ("a", "b", "t", "E", "nu")}, "q": [5000.0]})
    plate = NavierRectangle.from_material(a=2.0, b=1.0, q=5000.0, E=2.1e11, thickness=0.01, nu=0.3)
    ref = plate.at_points([1.0], [0.5], M=M_R3, diagnostics=False)
    assert out["state"] == "ok" and out["M"] == 2559 and (out["x"], out["y"]) == (1.0, 0.5)
    p = out["points"][0]
    assert (p["w_m"], p["w_coef"], p["error_bound_coef"]) == (float(ref.w[0]), float(ref.w_coef[0]), ref.w_error_bound_coef)
    assert out["D"] == plate.D


def test_worker_reproduces_the_saved_square_reference_bitwise():
    from kkvim_navier_worker import CONTRACT, compute

    saved = bm.load()["ssss"][0]["navier"]
    out = compute({"contract": CONTRACT, "a": 1.0, "b": 1.0, "t": 0.01, "E": 2.1e11, "nu": 0.3, "q": [1000.0]})
    p = out["points"][0]
    assert (p["w_coef"], p["w_m"], p["error_bound_coef"]) == (saved["beta"], saved["w_centre_m"], saved["error_bound_beta"])


@pytest.mark.parametrize("change, state", [(dict(b=0.39), "out_of_scope"), (dict(a=6.0), "out_of_scope"),
                                           (dict(contract="other"), "rejected"), (dict(q="1000"), "rejected"),
                                           (dict(nu=1.2), "rejected")])
def test_worker_refuses_outside_its_scope_with_a_reason(change, state):
    from kkvim_navier_worker import CONTRACT, compute

    req = {"contract": CONTRACT, "a": 2.0, "b": 1.0, "t": 0.01, "E": 2.1e11, "nu": 0.3, "q": [5000.0], **change}
    if isinstance(req["q"], str):
        req["q"] = [req["q"]]
    out = compute(req)
    assert out["state"] == state and out["reason"] and out["points"] == []


def test_worker_imports_only_the_reference(tmp_path):
    """The worker package imports kkvim_reference and the standard library, never kkvim or kkvim_gui."""
    import ast
    from pathlib import Path

    folder = Path(__file__).resolve().parents[1] / "src" / "kkvim_navier_worker"
    modules = set()
    for path in folder.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                modules |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                modules.add(node.module.split(".")[0])
    assert modules <= {"kkvim_reference", "kkvim_navier_worker", "json", "sys", "platform", "importlib"}


def test_job_runs_the_worker_in_its_own_process_with_the_plate_input_only(navier_record):
    sent = navier_record["inputs"]
    assert set(sent) == {"contract", "a", "b", "t", "E", "nu", "q"}  # no KK-VIM field or result is ever sent
    assert navier_record["state"] == "ok" and [p["q"] for p in navier_record["points"]] == [1000.0, 5000.0]
    # every load is evaluated by the reference itself: w = (q a^4 / D) w_coef with the same coefficient
    w1, w5 = (p["w_m"] for p in navier_record["points"])
    c1, c5 = (p["w_coef"] for p in navier_record["points"])
    assert c1 == c5 and w5 == pytest.approx(5 * w1, rel=1e-15)


def test_point_record_is_for_the_exact_request(navier_record):
    rec = navier_job.point_record(navier_record, request())
    assert rec["state"] == "ok" and rec["q"] == 5000.0 and rec["M"] == 2559
    assert navier_job.point_record(navier_record, request(q=4999.0))["state"] == navier_job.FAILED
    assert navier_job.point_record(navier_record, request(E=2.0e11))["state"] == navier_job.FAILED


@pytest.mark.parametrize("text, sent_change", [("{", {}), ('{"contract": "x"}', {}), (None, {"a": 3.0}), (None, {"q": [1.0]})])
def test_defective_answers_are_failed_never_values(text, sent_change):
    from kkvim_navier_worker import compute

    sent = navier_job.worker_input(request(), [5000.0])
    answer = text if text is not None else json.dumps(compute({**sent, **sent_change}))
    assert navier_job.parse_output(answer, sent)["state"] == navier_job.FAILED


def test_failed_process_and_timeout_are_reported(tmp_path):
    import sys

    job = navier_job.NavierJob({"contract": "x"}, command=[sys.executable, "-c", "import sys; sys.exit(3)"])
    while job.poll() is None:
        pass
    assert job.record["state"] == navier_job.FAILED
    job = navier_job.NavierJob({}, command=[sys.executable, "-c", "import time; time.sleep(60)"], timeout=0.5)
    while job.poll() is None:
        pass
    assert job.record["state"] == navier_job.FAILED and "остановлен" in job.record["reason"]
    assert job._process.poll() is not None  # no orphan process


# --- live comparison -----------------------------------------------------------------------------------------------------

def live(r3, navier_record):
    return bm.compare(r3, navier=navier_job.point_record(navier_record, r3.request))


def test_live_comparison_uses_the_centre_node_and_signed_differences(r3, navier_record):
    c = live(r3, navier_record)
    nav = navier_job.point_record(navier_record, r3.request)
    assert c.state == bm.AVAILABLE and c.live and len(c.rows) == 1
    ce = c.centre
    assert (ce["x"], ce["y"]) == (1.0, 0.5) and c.w_centre_m == float(r3.w[ce["i"], ce["j"]])
    row = c.rows[0]
    assert c.beta == c.w_centre_m * r3.D / (5000.0 * 2.0**4)
    assert row.reference.kind == "navier_live" and row.reference.beta == nav["w_coef"]
    assert row.delta == c.beta - nav["w_coef"] and row.rel_pct == row.delta / nav["w_coef"] * 100.0
    assert row.delta_w_m == c.w_centre_m - nav["w_m"]
    assert bm.cells(row)[5] == repr(row.delta_w_m)
    assert any("порога согласия нет" in n for n in c.notes) and not any("PROVISIONAL" in n for n in c.notes)


@pytest.mark.parametrize("nav, state", [({"state": "pending", "reason": ""}, bm.PENDING),
                                        ({"state": "out_of_scope", "reason": "вне области"}, bm.UNAVAILABLE),
                                        ({"state": "failed", "reason": "нет ответа"}, bm.UNAVAILABLE)])
def test_no_reference_value_no_numbers(r3, nav, state):
    c = bm.compare(r3, navier=nav)
    assert c.state == state and not c.has_numbers and c.reason


def test_odd_grid_has_no_centre_node_and_is_not_interpolated(navier_record):
    o = kkvim_adapter.execute(request(n_x=15, n_y=8))
    c = bm.compare(o, navier=navier_job.point_record(navier_record, o.request))
    assert c.state == bm.UNAVAILABLE and "не интерполируется" in c.reason


def test_diagnostic_field_is_never_compared(r3, navier_record):
    diag = dataclasses.replace(r3, field_role=ROLE_DIAGNOSTIC, requested_completed=False)
    assert bm.compare(diag, navier=navier_job.point_record(navier_record, r3.request)).state == bm.DIAGNOSTIC


def test_cccc_never_uses_navier(navier_record):
    from tests.test_gui_benchmarks import outcome as cccc_outcome  # noqa: PLC0415

    o = cccc_outcome()
    c = bm.compare(o, navier=navier_job.point_record(navier_record, request()))
    assert c.state == bm.AVAILABLE and {r.reference.kind for r in c.rows} == {"published", "D2", "FEM"}


def test_saved_square_and_live_navier_give_the_same_numbers():
    o = kkvim_adapter.execute(request(a=1.0, b=1.0, q=1000.0, n_x=16, n_y=16))
    job = navier_job.NavierJob(navier_job.worker_input(o.request, [1000.0]))
    while job.poll() is None:
        pass
    saved, fresh = bm.compare(o), bm.compare(o, navier=navier_job.point_record(job.record, o.request))
    assert (saved.rows[0].reference.beta, saved.rows[0].delta, saved.rows[0].rel_pct) == \
        (fresh.rows[0].reference.beta, fresh.rows[0].delta, fresh.rows[0].rel_pct)


def test_report_and_csv_reproduce_the_live_comparison(tmp_path, r3, navier_record):
    c = live(r3, navier_record)
    paths = report.write_package(r3, tmp_path / "pkg", {"run_id": 1}, (), comparison=c)
    rows = [r for r in csv.reader(line for line in paths["comparison"].read_text(encoding="utf-8").splitlines()
                                   if not line.startswith("#"))]
    head, row = rows[0], dict(zip(rows[0], rows[1], strict=True))
    assert "delta_w_centre [m]" in head and float(row["delta_w_centre [m]"]) == c.rows[0].delta_w_m
    assert float(row["beta_reference [1]"]) == c.rows[0].reference.beta and row["reference_level_M"] == "2559"
    assert row["reference_data_payload_sha256"] == "" and "kkvim_reference.navier" in row["reference_data"]
    html = paths["report"].read_text(encoding="utf-8")
    for text in bm.cells(c.rows[0])[1:6]:
        assert text in html
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert manifest["files"]["comparison"] == "comparison.csv"
    assert r3.request.q == 5000.0 and np.array_equal(r3.w, kkvim_adapter.execute(request()).w)  # solver result unchanged


def test_report_states_the_scientific_status_once(tmp_path):
    from tests.test_gui_benchmarks import outcome as cccc_outcome  # noqa: PLC0415

    o = dataclasses.replace(cccc_outcome(), science_text="RESEARCH_ONLY / PROVISIONAL / NOT_QUALIFIED: CCCC, N = 1")
    html = report.report_html(o, {}, comparison=bm.compare(o))
    section = html[html.index("6. Сравнение"):html.index("7. История")]
    assert section.count("RESEARCH_ONLY / PROVISIONAL / NOT_QUALIFIED") == 1


# --- load series: form, requests, data and export ---------------------------------------------------------------------

def test_series_field_parsing():
    assert parse_series("1; 2,5 5\t7.5") == ([1.0, 2.5, 5.0, 7.5], None)
    for bad in ("5", "1; 1", "1; x", "; ".join(str(k) for k in range(1, 23))):
        values, error = parse_series(bad)
        assert values is None and error


def test_series_requests_differ_only_in_q():
    st = SetupState().with_choice(load_mode="series").with_text("q_series", "1; 2,5; 5")
    assert not validate(st) and to_request(st) is None
    reqs = to_series_requests(st)
    single = to_request(dataclasses.replace(st, load_mode="single").with_text("q", "5"))
    assert [r.q for r in reqs] == [1000.0, 2500.0, 5000.0]
    for r in reqs:
        assert dataclasses.replace(r, q=single.q, entered=single.entered) == single
        assert dict((k, (t, u)) for k, t, u in r.entered)["q"][1] == "kPa"
    assert reqs[-1] == single  # the 5 kPa point is exactly the single run of 5 kPa


def fake_run(run_id, o, status="finished"):
    return SimpleNamespace(run_id=run_id, outcome=o, status=status)


def series_record(requests, runs, navier):
    return SimpleNamespace(series_id=1, status="finished", started=__import__("datetime").datetime(2026, 11, 13),
                           requests=tuple(requests), run_ids=[r.run_id for r in runs], navier=navier)


def test_series_points_use_real_runs_and_exclude_diagnostics(navier_record, r3):
    o1 = kkvim_adapter.execute(request(q=1000.0))
    diag = dataclasses.replace(o1, field_role=ROLE_DIAGNOSTIC, requested_completed=False, status="MAX_HALF_STEPS")
    runs = [fake_run(1, o1), fake_run(2, r3), fake_run(3, diag, status="finished")]
    reqs = [request(q=1000.0), request(q=5000.0), request(q=1000.0)]
    manager = SimpleNamespace(get=lambda i: runs[i - 1])
    data = series_model.series_data(series_record(reqs, runs, navier_record), manager)
    p1, p5, pd = data.points
    assert p1.included and p5.included and not pd.included and "диагност" in pd.reason
    assert p5.w_centre_m == float(r3.w[8, 4]) and p5.navier_w_m == navier_record["points"][1]["w_m"]
    assert p5.delta_w_m == p5.w_centre_m - p5.navier_w_m and p5.rel_pct == p5.delta_w_m / p5.navier_w_m * 100.0
    assert p5.w_at_max_abs_m == r3.extremum()["w"] and p5.field_role == ROLE_ACCEPTED


def test_series_export_round_trips_and_never_overwrites(tmp_path, navier_record, r3):
    o1 = kkvim_adapter.execute(request(q=1000.0))
    runs = [fake_run(1, o1), fake_run(2, r3)]
    data = series_model.series_data(series_record([request(q=1000.0), request(q=5000.0)], runs, navier_record),
                                    SimpleNamespace(get=lambda i: runs[i - 1]))
    paths = series_export.write_series(data, tmp_path / "s")
    lines = [line for line in paths["load_series"].read_text(encoding="utf-8").splitlines() if not line.startswith("#")]
    rows = list(csv.DictReader(lines))
    assert [float(r["q [Pa]"]) for r in rows] == [1000.0, 5000.0]
    assert [float(r["w_centre_kkvim [m]"]) for r in rows] == [p.w_centre_m for p in data.points]
    assert [float(r["w_centre_navier [m]"]) for r in rows] == [p.navier_w_m for p in data.points]
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert manifest["kind"] == "load_series" and manifest["schema"] == series_export.SCHEMA
    assert manifest["files"]["load_series"]["sha256"] == hashlib.sha256(paths["load_series"].read_bytes()).hexdigest()
    assert manifest["reference"]["M"] == 2559
    with pytest.raises(FileExistsError):
        series_export.write_series(data, tmp_path / "s")


def test_cccc_series_has_no_navier_curve():
    from tests.test_gui_benchmarks import outcome as cccc_outcome  # noqa: PLC0415

    o = cccc_outcome()
    runs = [fake_run(1, o), fake_run(2, o)]
    data = series_model.series_data(series_record([o.request, dataclasses.replace(o.request, q=2000.0)], runs, None),
                                    SimpleNamespace(get=lambda i: runs[i - 1]))
    assert data.navier is None and all(p.navier_state is None and p.navier_w_m is None for p in data.points)
