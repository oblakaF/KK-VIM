"""VIM-1, VIM-2, VIM-3 of one KK-VIM run (GUI-VIM-CYCLES-123-01), without Qt.

- VIM-k is the state after the completed full cycle k, s = 2k (D-030), of ONE run; s = 1 (k = 0) is never VIM-1;
- values come from the stored half-step history of that run (R3: w_bar_centre; CCCC: alpha_centre), never from the final
  field, another load or a new computation; w_c = value * q * L^4 / D;
- the conversion is checked against the solver's own half-step fields (direct ``run_rectangle_physical``) with a bound
  fixed before the comparison: 16 u relative (the forward and backward normalisation are at most 8 roundings; D-025);
- signed differences with the reference of the run; unavailable and diagnostic states are stated, never substituted;
- CCCC N >= 2 is limited to s <= 2 (RN-5/RT-7): VIM-2 and VIM-3 never exist there;
- export: the CSV rows are the records of the table; the package and the series package record SHA256 of the new files.
"""

import csv
import dataclasses
import hashlib
import json
from datetime import datetime
from types import SimpleNamespace

import pytest

from kkvim.rectangle_run import run_rectangle_physical
from kkvim_gui.backend import kkvim_adapter, navier_job, report, series_export, vim_cycles
from kkvim_gui.model.series import series_data
from kkvim_gui.model.setup_state import SetupState, to_request

U = 2.0 ** -53
BOUND = 16 * U  # fixed before the comparison (module text)


def navier_record(request, loads=None):
    """The job record of the real Navier worker contract, evaluated in this process."""
    from kkvim_navier_worker import compute

    inputs = navier_job.worker_input(request, loads or [request.q])
    return navier_job.parse_output(json.dumps(compute(inputs)), inputs)


@pytest.fixture(scope="module")
def case_a():
    request = to_request(SetupState())  # the owner's plate: SSSS, N = 1, 2 x 1 m, 10 mm, 210 GPa, 0.3, 5 kPa, 80 x 40
    assert (request.supports, request.N, request.a, request.b, request.q, request.n_x, request.n_y) == (
        "SSSS", 1, 2.0, 1.0, 5000.0, 80, 40)
    outcome = kkvim_adapter.execute(request)
    navier = navier_job.point_record(navier_record(request), request)
    return request, outcome, navier


def test_labels_map_to_s_equal_2k_and_never_to_n():
    assert vim_cycles.K_MAX == 3
    assert [vim_cycles.label(k) for k in (1, 2, 3)] == ["VIM-1", "VIM-2", "VIM-3"]
    assert vim_cycles.cycle_text(1) == "VIM-1 (k = 1, s = 2: Y¹, X²)"
    assert vim_cycles.cycle_text(3) == "VIM-3 (k = 3, s = 6: Y⁵, X⁶)"
    assert "не N = 1, 2, 3" in vim_cycles.OWNER_NOTE and "не обозначения авторов" in vim_cycles.OWNER_NOTE


@pytest.mark.parametrize(("s", "text"), [(1, "s = 1: только Y¹, полного цикла нет (k = 0)"),
                                         (6, "s = 6: k = 3 полных цикла, последний полушаг — X⁶"),
                                         (7, "s = 7: k = 3 полных цикла и ещё один Y-полушаг (Y⁷)"),
                                         (8, "s = 8: k = 4 полных цикла, последний полушаг — X⁸"),
                                         (11, "s = 11: k = 5 полных циклов и ещё один Y-полушаг (Y¹¹)")])
def test_termination_counts_full_cycles_and_the_extra_half_step(s, text):
    assert vim_cycles.termination_text(s) == text


def test_case_a_values_are_the_half_steps_2_4_6_of_the_run(case_a):
    request, o, navier = case_a
    t = vim_cycles.cycle_table(o, navier=navier)
    assert t.supported and t.key == "w_bar_centre" and t.norm == "a" and t.reference == vim_cycles.NAVIER
    assert (t.solver_status, t.s, t.k, t.criterion_met) == ("CONVERGED", 8, 4, True)
    assert t.partial.label == "s = 1 (k = 0)" and t.partial.k == 0 and t.partial.s == 1 and not t.partial.on_graph
    rows = {row["s"]: row for row in o.history}
    for k, c in enumerate(t.cycles, start=1):
        assert (c.label, c.k, c.s, c.direction) == (f"VIM-{k}", k, 2 * k, "X")
        assert c.available and c.on_graph and c.reason == ""
        assert c.w_norm == rows[2 * k]["w_bar_centre"]  # the stored value of the row s = 2k, unchanged
        assert c.w_m == rows[2 * k]["w_bar_centre"] * request.q * request.a ** 4 / o.D  # MATH_SPEC 2.5, as specified
    # against the solver's own half-step fields (direct call, not the GUI path)
    direct = run_rectangle_physical(request.a, request.b, request.t, request.E, request.nu, request.q, request.n_x, request.n_y)
    records = direct.run.history.records
    for c in (t.partial, *t.cycles):
        rec = records[c.s - 1]
        i, j = list(rec.grid_x.nodes).index(request.a / 2), list(rec.grid_y.nodes).index(request.b / 2)
        w = float(rec.field()[i, j])
        assert abs(c.w_m - w) <= BOUND * abs(w), (c.s, c.w_m, w)
    final_centre = float(o.w[40, 20])
    assert t.cycles[2].w_m != final_centre  # VIM-3 is s = 6, not the final field (s = 8)


def test_case_a_signed_difference_with_navier(case_a):
    _, o, navier = case_a
    t = vim_cycles.cycle_table(o, navier=navier)
    assert navier["state"] == "ok" and t.ref_w_m == navier["w_m"]
    for c in t.cycles:
        assert c.ref_w_m == navier["w_m"] and c.delta_w_m == c.w_m - navier["w_m"]
        assert c.delta_pct == (c.w_m - navier["w_m"]) / navier["w_m"] * 100.0
    assert t.partial.delta_w_m == t.partial.w_m - navier["w_m"]


def test_reference_not_available_gives_no_difference(case_a):
    _, o, _ = case_a
    t = vim_cycles.cycle_table(o, navier={"state": "failed", "reason": "процесс эталона не запущен"})
    assert t.ref_w_m is None and "не запущен" in t.reference_reason
    assert all(c.on_graph and c.delta_w_m is None and c.delta_pct is None for c in t.cycles)
    assert vim_cycles.cycle_table(o).reference_reason.startswith("эталон Навье для этого расчёта не рассчитывался")


def r3(request, **changes):
    return kkvim_adapter.execute(dataclasses.replace(request, **changes))


def test_run_converged_before_s_6_has_no_vim_3():
    request = dataclasses.replace(to_request(SetupState()), n_x=16, n_y=8)
    o = r3(request, s_min=2, tol_inf=1e-2, tol_E=1e-2)  # non-default stopping parameters (stated in the run)
    assert o.status == "CONVERGED" and o.s < 6
    t = vim_cycles.cycle_table(o)
    drawn = [c.k for c in t.cycles if c.on_graph]
    assert drawn == list(range(1, o.s // 2 + 1))
    missing = t.cycles[2]
    assert not missing.available and missing.w_m is None and f"полушаг s = 6 не выполнялся" in missing.reason
    assert f"s = {o.s}" in missing.reason and "более позднее поле не подставляется" in missing.reason


def test_run_stopped_by_s_max_is_diagnostic_only():
    request = dataclasses.replace(to_request(SetupState()), n_x=16, n_y=8)
    o = r3(request, s_max=3)
    assert not o.requested_completed and o.s == 3
    t = vim_cycles.cycle_table(o)
    assert not t.criterion_met and t.termination == "s = 3: k = 1 полный цикл и ещё один Y-полушаг (Y³)"
    vim1 = t.cycles[0]
    assert vim1.available and not vim1.on_graph and "только диагностика" in vim1.reason
    assert not t.cycles[1].available and not t.cycles[2].available and not t.drawn


def test_odd_grid_has_no_centre_value():
    request = dataclasses.replace(to_request(SetupState()), n_x=17, n_y=8)
    t = vim_cycles.cycle_table(kkvim_adapter.execute(request))
    assert not t.drawn and all("не является узлом" in c.reason and c.w_m is None for c in t.cycles)


def test_route_without_centre_history_is_unavailable():
    st = SetupState().with_choice(N=2).with_text("n_x", "16").with_text("n_y", "8")
    o = kkvim_adapter.execute(to_request(st))
    assert o.solver == "R4"
    t = vim_cycles.cycle_table(o)
    assert not t.supported and "max|w|" in t.route_reason and not t.drawn
    assert all(not c.available and c.reason == t.route_reason for c in (t.partial, *t.cycles))


def cccc_request(N, half_steps, b="1"):
    st = (SetupState().with_choice(supports="CCCC", N=N).with_choice(cccc_half_steps=half_steps).with_text("a", "1")
          .with_text("b", b).with_text("q", "1").with_text("n_x", "32").with_text("n_y", "32"))
    request = to_request(st)
    assert request is not None and request.supports == "CCCC" and request.half_steps == half_steps
    return request


def test_cccc_n1_uses_the_stored_alpha_and_the_d2_reference():
    request = cccc_request(1, 8)
    o = kkvim_adapter.execute(request)
    d2 = {"state": "ok", "w_centre_m": 6.635089257457487e-05}
    t = vim_cycles.cycle_table(o, d2=d2)
    assert t.supported and t.key == "alpha_centre" and t.norm == "min(a, b)" and t.reference == vim_cycles.D2
    rows = {row["s"]: row for row in o.history}
    for k, c in enumerate(t.cycles, start=1):
        assert c.on_graph and c.w_norm == rows[2 * k]["alpha_centre"]
        assert c.w_m == rows[2 * k]["alpha_centre"] * request.q * min(request.a, request.b) ** 4 / o.D
        assert c.delta_w_m == c.w_m - d2["w_centre_m"]
    last = rows[8]["alpha_centre"] * request.q * min(request.a, request.b) ** 4 / o.D  # s = 8 against the stored field
    assert abs(last - float(o.w[16, 16])) <= BOUND * abs(float(o.w[16, 16]))
    assert not t.criterion_met and o.mark.startswith("RESEARCH_ONLY")  # fixed half-steps: no stopping criterion
    assert vim_cycles.cycle_table(o).reference_reason.startswith("МКР D2 для этого расчёта не выбран")


def test_cccc_n2_has_at_most_vim_1():
    request = cccc_request(2, 2, b="1,4")
    o = kkvim_adapter.execute(request)
    assert (o.solver, o.status, o.s) == ("CCCC-N23", "COMPLETED_FIXED_HALF_STEPS", 2)  # RN-5/RT-7 passed (32 x 32)
    t = vim_cycles.cycle_table(o)
    assert [c.k for c in t.cycles if c.available] == [1] and t.cycles[0].on_graph
    assert not t.criterion_met and o.mark.startswith("RESEARCH_ONLY")
    for c in t.cycles[1:]:
        assert not c.available and c.w_m is None and "допускает не более s = 2" in c.reason and "RN-5/RT-7" in c.reason


def test_csv_rows_are_the_table_records(case_a, tmp_path):
    _, o, navier = case_a
    t = vim_cycles.cycle_table(o, navier=navier)
    path = tmp_path / "c.csv"
    vim_cycles.write_csv(path, [(7, t)])
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if not line.startswith("#")]
    rows = list(csv.DictReader(lines))
    assert [r["label"] for r in rows] == ["s = 1 (k = 0)", "VIM-1", "VIM-2", "VIM-3"]
    for r, c in zip(rows, (t.partial, *t.cycles), strict=True):
        assert float(r["w_KKVIM_m [m]"]) == c.w_m and float(r["w_Navier_m [m]"]) == navier["w_m"]
        assert float(r["delta_w_m [m]"]) == c.delta_w_m and float(r["delta_percent [%]"]) == c.delta_pct
        assert (int(r["k"]), int(r["s"]), r["run_id"], float(r["q_Pa [Pa]"]), r["N"]) == (c.k, c.s, "7", 5000.0, "1")
        assert r["solver_status"] == "CONVERGED" and r["scientific_status"] and r["provenance"].startswith("история")
    for name in ("run_id", "q_Pa [Pa]", "N", "k", "s", "w_KKVIM_m [m]", "w_Navier_m [m]", "delta_w_m [m]",
                 "delta_percent [%]", "solver_status", "scientific_status", "provenance"):
        assert name in rows[0]


def test_package_records_the_cycles_and_their_sha256(case_a, tmp_path):
    _, o, navier = case_a
    t = vim_cycles.cycle_table(o, navier=navier)
    paths = report.write_package(o, tmp_path / "pkg", {"run_id": 3}, cycles=t)
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    rec = manifest["vim_cycles"]
    assert rec["files_sha256"] == {"vim_cycles.csv": hashlib.sha256(paths["vim_cycles"].read_bytes()).hexdigest()}
    assert [s["w_KKVIM_m"] for s in rec["states"]] == [c.w_m for c in (t.partial, *t.cycles)]
    assert rec["s"] == 8 and rec["k"] == 4 and rec["criterion_met"] is True and rec["run_id"] == 3
    html = paths["report"].read_text(encoding="utf-8")
    assert "6в. Полные циклы одного расчёта КК-ВИМ: VIM-1, VIM-2, VIM-3" in html and "vim_cycles.csv" in html
    with pytest.raises(FileExistsError):  # an existing package is never overwritten
        report.write_package(o, tmp_path / "pkg", {"run_id": 3}, cycles=t)


def test_series_points_use_their_own_runs_and_the_series_export(tmp_path):
    base = dataclasses.replace(to_request(SetupState()), n_x=16, n_y=8)
    requests = [dataclasses.replace(base, q=q) for q in (1000.0, 5000.0, 2500.0)]
    runs = [SimpleNamespace(run_id=i + 1, status="finished", outcome=kkvim_adapter.execute(r), d2=None, timing={})
            for i, r in enumerate(requests)]
    runs[2].outcome = r3(requests[2], s_max=3)  # this point did not end as requested
    runs[2].status = "finished"
    series = SimpleNamespace(series_id=1, status="finished", started=datetime(2026, 11, 16), requests=tuple(requests),
                             run_ids=[1, 2, 3], navier=navier_record(base, [r.q for r in requests]),
                             snapshot=SetupState())
    manager = SimpleNamespace(get=lambda i: runs[i - 1])
    data = series_data(series, manager)
    assert data.cycles_supported
    for p, run in zip(data.points, runs, strict=True):
        own = vim_cycles.cycle_table(run.outcome, navier=navier_job.point_record(series.navier, run.outcome.request))
        assert p.cycles == own and p.q == run.outcome.request.q
    assert [p.q for p in data.cycle_included(1)] == [1000.0, 5000.0]  # the third run is diagnostic only
    assert data.points[2].cycle(1).available and not data.points[2].cycle(1).on_graph
    paths = series_export.write_series(data, tmp_path / "series")
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    entry = manifest["files"]["load_series_vim_cycles"]
    assert entry["sha256"] == hashlib.sha256(paths["load_series_vim_cycles"].read_bytes()).hexdigest()
    assert manifest["vim_cycles"]["points_on_curve"] == {"VIM-1": 2, "VIM-2": 2, "VIM-3": 2}
    lines = [x for x in paths["load_series_vim_cycles"].read_text(encoding="utf-8").splitlines() if not x.startswith("#")]
    rows = list(csv.DictReader(lines))
    assert len(rows) == 3 * 4 and {r["run_id"] for r in rows} == {"1", "2", "3"}
    for r in rows:
        p = data.points[int(r["run_id"]) - 1]
        c = p.cycles.partial if r["k"] == "0" else p.cycle(int(r["k"]))
        assert float(r["q_Pa [Pa]"]) == p.q
        assert r["w_KKVIM_m [m]"] == "" if c.w_m is None else float(r["w_KKVIM_m [m]"]) == c.w_m
