"""GUI-CORE-N16-P1: the GUI routes SSSS and CCCC, N = 1-6, to the existing solvers and states the role of every field.

Qt-free checks of ``kkvim_gui.backend``: the route of (supports, N), the solver called and its arguments, the copy of the
solver arrays (compared bit for bit with a direct call of the same kkvim entry; nothing is recomputed), the one mapping
status -> field role, the file name of the field, the manifest (schema 2) and the report, the protective stops of the
solvers (RN-5, RT-7, the grid bound of s = 2), the refusals before a run and the reading of a schema 1 manifest. The control
cases are the small CCCC cases of D-063, D-065 and D-068 and the studied cell of SSSS N = 4-6; no tolerance is used.
"""

import csv
import json

import numpy as np
import pytest

import kkvim.cccc_n1 as n1
import kkvim.cccc_n23 as n23
import kkvim.cccc_n46 as n46
import kkvim.coupled_run as cr
import kkvim.research_ssss as rs
from kkvim.rectangle import validated_rectangle_input
from kkvim.rectangle_run import run_rectangle_physical
from kkvim_gui.backend import export, kkvim_adapter, report
from kkvim_gui.backend.outcome import RunOutcome
from kkvim_gui.backend.request import (
    CCCC_HALF_STEPS,
    R4_NORMATIVE,
    R4_RESEARCH_STOP,
    ROUTE_CCCC_N1,
    ROUTE_CCCC_N23,
    ROUTE_CCCC_N46,
    ROUTE_R3,
    ROUTE_R4,
    ROUTE_SSSS_N46,
    RunRequest,
    route_of,
)
from kkvim_gui.model.setup_state import SetupState, to_request, validate

PLATE = {"t": 0.01, "E": 2.1e11, "nu": 0.3}
RESEARCH = "RESEARCH_ONLY / PROVISIONAL / NOT_QUALIFIED"


def req(supports, N, a=1.0, b=1.0, n_x=32, n_y=32, q=1000.0, half_steps=None, **changes):
    if supports == "SSSS" and N == 1:
        changes = {"tol_inf": 1e-10, "tol_E": 1e-8, "s_min": 3, "s_max": 100, **changes}
    return RunRequest(entered=(), a=a, b=b, q=q, n_x=n_x, n_y=n_y, N=N, supports=supports, half_steps=half_steps,
                      **PLATE, **changes)


def physical(r):
    return (r.a, r.b, r.t, r.E, r.nu, r.q, r.n_x, r.n_y)


_CACHE = {}


def run(name, r):
    """One adapter call per control case of this module."""
    if name not in _CACHE:
        assert kkvim_adapter.check_request(r) == []
        _CACHE[name] = kkvim_adapter.execute(r)
    return _CACHE[name]


CCCC_N23_S2 = {"N2_C1": req("CCCC", 2, b=0.6, n_x=10, n_y=9, half_steps=2), "N3_C2": req("CCCC", 3, b=1.3, n_x=8, n_y=7, half_steps=2)}
CCCC_N46_S2 = {"N5_C7_20x13": req("CCCC", 5, b=0.6, n_x=20, n_y=13, half_steps=2)}  # a grid above the minimum 2N + 1
N6_RN5_REFUSAL = req("CCCC", 6, a=5.0, b=1.0, n_x=32, n_y=13, half_steps=2)


# --- routes ---------------------------------------------------------------------------------------------------------------

def test_every_supports_and_n_has_exactly_its_route():
    assert [route_of("SSSS", N) for N in range(1, 7)] == [ROUTE_R3, ROUTE_R4, ROUTE_R4, ROUTE_SSSS_N46, ROUTE_SSSS_N46,
                                                          ROUTE_SSSS_N46]
    assert [route_of("CCCC", N) for N in range(1, 7)] == [ROUTE_CCCC_N1, ROUTE_CCCC_N23, ROUTE_CCCC_N23, ROUTE_CCCC_N46,
                                                          ROUTE_CCCC_N46, ROUTE_CCCC_N46]
    assert route_of("SSSS", 7) is None and route_of("SCSC", 1) is None
    assert CCCC_HALF_STEPS == {ROUTE_CCCC_N1: tuple(range(1, 9)), ROUTE_CCCC_N23: (1, 2), ROUTE_CCCC_N46: (1, 2)}
    assert kkvim_adapter.ROUTES[ROUTE_CCCC_N46]["decisions"] == ("D-067", "D-068")


@pytest.mark.parametrize("name, r, fragment", [
    ("SSSS N = 4 without the research choice", req("SSSS", 4, n_x=40, n_y=40, half_steps=4), "D-056.7"),
    ("SSSS N = 5 outside the studied grids", req("SSSS", 5, n_x=41, n_y=41, half_steps=4, ssss_research=True),
     "NOT_EXECUTED_OUTSIDE_DOMAIN"),
    ("SSSS N = 6 beyond s = 40", req("SSSS", 6, n_x=40, n_y=40, half_steps=41, ssss_research=True), "NOT_EXECUTED_OUTSIDE_DOMAIN"),
    ("CCCC N = 2, s = 3", req("CCCC", 2, n_x=16, n_y=16, half_steps=3), "s ≥ 3"),
    ("CCCC N = 5, s = 3", req("CCCC", 5, n_x=16, n_y=16, half_steps=3), "s ≥ 3"),
    ("CCCC N = 1, s = 9", req("CCCC", 1, n_x=16, n_y=16, half_steps=9), "1…8"),
    ("CCCC N = 4, grid below 2N + 1", req("CCCC", 4, n_x=8, n_y=16, half_steps=1), "NOT_EXECUTED_OUTSIDE_DOMAIN"),
    ("CCCC N = 1, b/a = 6", req("CCCC", 1, b=6.0, n_x=16, n_y=16, half_steps=1), "ERROR_INPUT"),
])
def test_refused_input_never_reaches_a_solver(monkeypatch, name, r, fragment):
    calls = []
    for module, entry in ((rs, "run_research"), (n1, "run_cccc_n1_physical"), (n23, "run_cccc_n23_physical"),
                          (n46, "run_cccc_n46_physical")):
        monkeypatch.setattr(module, entry, lambda *a, _e=entry, **k: calls.append(_e))
    problems = kkvim_adapter.check_request(r)
    assert any(fragment in p for p in problems), problems
    with pytest.raises(kkvim_adapter.AdapterInputError):
        kkvim_adapter.execute(r)
    assert calls == []


# --- SSSS -------------------------------------------------------------------------------------------------------------------

def test_ssss_n1_is_the_accepted_r3_path_unchanged():
    r = req("SSSS", 1)
    out = run("ssss_n1", r)
    direct = run_rectangle_physical(*physical(r), tol_inf=1e-10, tol_E=1e-8, s_min=3, s_max=100)
    assert out.route == ROUTE_R3 and out.solver == "R3" and out.status == "CONVERGED" and out.mark is None
    assert np.array_equal(out.w, direct.run.last_field()) and np.array_equal(out.w_bar, direct.dimensionless_field())
    assert out.field_role == "accepted" and out.requested_completed and out.field_file == "field"


@pytest.mark.parametrize("mode, research_mode", [(R4_NORMATIVE, False), (R4_RESEARCH_STOP, True)])
def test_ssss_n2_3_route_to_run_coupled_with_the_mode_unchanged(monkeypatch, mode, research_mode):
    seen = []
    real = cr.run_coupled
    monkeypatch.setattr(cr, "run_coupled", lambda *a, **k: seen.append((a, k)) or real(*a, **k))
    r = req("SSSS", 3, n_x=40, n_y=40, transform="fft", half_steps=100, r4_mode=mode)
    out = kkvim_adapter.execute(r)
    (args, kwargs), = seen
    assert kwargs["research_mode"] is research_mode and args[6] == 3 and out.route == ROUTE_R4 and out.mark == cr.MARK
    _, D = validated_rectangle_input(*physical(r))
    direct = real(r.a, r.b, D, r.q, r.n_x, r.n_y, 3, transform="fft", half_steps=100, research_mode=research_mode,
                  stop_on_stabilization=True)
    assert out.status == direct.status and np.array_equal(out.w, direct.field())
    assert out.field_role == "research" and out.requested_completed  # NOT_QUALIFIED / PROVISIONAL_FIELD_STABILIZED: the mode's end


def test_ssss_n4_6_research_runs_the_studied_cell_scaled_to_metres():
    r = req("SSSS", 4, n_x=40, n_y=40, half_steps=4, ssss_research=True)
    out = run("ssss_n4", r)
    _, D = validated_rectangle_input(*physical(r))
    cell = rs.run_research(1.0, 40, 40, 4, half_steps=4)
    g = r.q * r.a**4 / D
    assert out.route == ROUTE_SSSS_N46 and out.status == cr.RESEARCH_LIMIT_REACHED and out.s == out.s_requested == 4
    assert np.array_equal(out.w_bar, cell.field()) and np.array_equal(out.w, cell.field() * g)  # w = w_bar q a^4 / D
    assert out.mark == RESEARCH and out.field_role == "research" and out.requested_completed
    assert next(c for c in out.columns if c.key == "P").unit == "q²a⁶/D"  # energies stay in cell units, labelled so


# --- CCCC -------------------------------------------------------------------------------------------------------------------

def test_cccc_n1_fixed_half_steps_give_a_research_field():
    r = req("CCCC", 1, n_x=16, n_y=16, half_steps=4)
    out = run("cccc_n1", r)
    direct = n1.run_cccc_n1_physical(*physical(r), 4)
    assert out.route == ROUTE_CCCC_N1 and out.status == n1.COMPLETED_FIXED_HALF_STEPS and out.s == 4
    assert np.array_equal(out.w, direct.field()) and np.array_equal(out.w_bar, direct.dimensionless_field())
    assert out.field_role == "research" and out.mark == RESEARCH and out.stop_kind == "none"
    assert np.array_equal(out.x, [i * (1.0 / 16) for i in range(17)])


@pytest.mark.parametrize("name", list(CCCC_N23_S2))
def test_cccc_n2_3_s2_runs_under_rn5_and_rt7_of_the_solver(name):
    r = CCCC_N23_S2[name]
    out = run(name, r)
    direct = n23.run_cccc_n23_physical(*physical(r), r.N, 2)
    assert out.status == n23.COMPLETED_FIXED_HALF_STEPS and out.s == out.s_requested == 2 and out.requested_completed
    assert np.array_equal(out.w, direct.field()) and np.array_equal(out.w_bar, direct.dimensionless_field())
    s2 = out.history[1]
    assert s2["rn5_verdict"] == "RN5_CERTIFIED" and s2["rt7_passed"] is True and s2["qr_executed"] is True
    assert s2["direction"] == "X" and out.field_role == "research"


@pytest.mark.parametrize("name", list(CCCC_N46_S2))
def test_cccc_n4_6_s2_on_a_nonminimal_grid(name):
    r = CCCC_N46_S2[name]
    out = run(name, r)
    direct = n46.run_cccc_n46_physical(*physical(r), r.N, 2)
    assert out.route == ROUTE_CCCC_N46 and out.status == n46.COMPLETED_FIXED_HALF_STEPS and out.s == 2
    assert np.array_equal(out.w, direct.field()) and out.history[1]["rn5_verdict"] == "RN5_CERTIFIED"
    assert out.history[1]["rt7_passed"] is True and out.field_role == "research" and out.decisions == ("D-067", "D-068")
    e = out.extremum()
    assert (e["i"], e["j"]) == (10, 6) and e["w"] == float(np.max(np.abs(direct.field())))


def test_cccc_n6_rn5_refusal_keeps_only_the_diagnostic_field_of_s1():
    out = run("n6_rn5", N6_RN5_REFUSAL)
    one = n46.run_cccc_n46_physical(*physical(N6_RN5_REFUSAL), 6, 1)
    assert out.status == n46.STOPPED_RANK_GUARD_UNCERTIFIED and out.s == 1 and out.s_requested == 2
    assert not out.requested_completed and out.field_role == "diagnostic" and out.diagnostic_only and not out.is_error
    assert out.stop_kind == "not_met" and "Защитная остановка" in out.stop_text
    s2 = out.history[1]
    assert s2["rn5_verdict"] != "RN5_CERTIFIED" and s2["qr_executed"] is False and s2["rt7_passed"] is None
    assert np.array_equal(out.w, one.field())  # the field of s = 1, unchanged


def test_cccc_s2_above_the_grid_bound_stops_before_rn5_qr_and_the_x_solve(monkeypatch):
    calls = []
    real_solve, real_qr = n23.solve_bc, n23.orthonormalise
    monkeypatch.setattr(n23, "solve_bc", lambda *a: calls.append("solve") or real_solve(*a))
    monkeypatch.setattr(n23, "orthonormalise", lambda *a: calls.append("qr") or real_qr(*a))
    monkeypatch.setattr(n23.certificate, "rn5_certificate", lambda *a: pytest.fail("RN-5 above the grid bound"))
    out = kkvim_adapter.execute(req("CCCC", 3, n_x=33, n_y=33, half_steps=2))
    assert out.status == n23.STOPPED_RANK_GUARD_UNCERTIFIED and out.s == 1 and out.field_role == "diagnostic"
    assert calls == ["qr", "solve"]  # only s = 1: QR of the start set and the Y-solve; no QR of V and no X-solve
    assert any("до RN-5 и QR" in n for n in out.notes)


def test_negative_q_keeps_the_sign_of_w_and_the_normalized_field(tmp_path):
    up = run("N2_C1", CCCC_N23_S2["N2_C1"])
    down = kkvim_adapter.execute(req("CCCC", 2, b=0.6, n_x=10, n_y=9, half_steps=2, q=-1000.0))
    assert down.status == up.status == n23.COMPLETED_FIXED_HALF_STEPS
    assert np.array_equal(down.w, -up.w) and np.array_equal(down.w_bar, up.w_bar)
    assert np.sign(down.extremum()["w"]) == -1 and down.extremum()["w_bar"] == up.extremum()["w_bar"]
    paths = export.write_result(down, tmp_path / "down")
    rows = _rows(paths["field"])
    assert [float(r["w [m]"]) for r in rows] == [float(v) for v in down.w.ravel()]


# --- field role, files, manifest and report ------------------------------------------------------------------------------------

def _rows(path):
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if not line.startswith("#")]
    return list(csv.DictReader(lines))


@pytest.mark.parametrize("name, r, role, file", [
    ("ssss_n1", req("SSSS", 1), "accepted", "field.csv"),
    ("ssss_n4", req("SSSS", 4, n_x=40, n_y=40, half_steps=4, ssss_research=True), "research", "field_research.csv"),
    ("cccc_n1", req("CCCC", 1, n_x=16, n_y=16, half_steps=4), "research", "field_research.csv"),
    ("n6_rn5", N6_RN5_REFUSAL, "diagnostic", "field_diagnostic.csv"),
])
def test_the_field_file_manifest_and_report_agree(tmp_path, name, r, role, file):
    out = run(name, r)
    paths = report.write_package(out, tmp_path / name, {"run_id": 1})
    data = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    names = sorted(p.name for p in (tmp_path / name).iterdir())
    assert paths["field"].name == file and file in names and [n for n in names if n.startswith("field")] == [file]
    st, res = data["statuses"], data["results"]
    assert data["schema_version"] == 2 and st["field_role"] == res["field_role_code"] == role and st["field_file"] == file
    assert data["files"]["field"] == file and res["field_file"] == file
    assert st["solver_status"] == out.status and st["solver_reason"] == out.reason and st["mark"] == out.mark
    assert st["requested_run_completed"] is out.requested_completed is (role != "diagnostic")
    assert res["half_steps"] == {"completed": out.s, "requested": out.s_requested}
    assert data["model"]["supports"] == r.supports and data["model"]["N"] == r.N and data["model"]["route"] == out.route
    assert data["input"]["si"]["q"] == {"value": r.q, "unit": "Pa"} and data["limitations"] == list(out.limitations)
    assert data["program"]["versions"]["kkvim"] and data["solver"]["call"] == out.call
    rows = _rows(paths["field"])  # copied from the solver arrays, repr round trip
    assert [float(x["w [m]"]) for x in rows] == [float(v) for v in out.w.ravel()]
    assert [float(x["w_bar [1]"]) for x in rows] == [float(v) for v in out.w_bar.ravel()]
    first = paths["field"].read_text(encoding="utf-8").splitlines()[0]
    html = paths["report"].read_text(encoding="utf-8")
    assert file in html and out.status in html and report.ROLE_TITLE[role] in html
    if role == "diagnostic":
        assert first.startswith("# RESEARCH_ONLY") and any(m.startswith("DIAGNOSTIC") for m in data["marks"])
        assert "ДИАГНОСТИКА" in html and "CONVERGED" not in html and "CONVERGED" not in json.dumps(data)
    if out.mark:
        assert first == f"# {out.mark}" and all(t in html for t in out.limitations[:1])


def test_a_package_is_never_written_over_an_existing_one(tmp_path):
    out = run("cccc_n1", req("CCCC", 1, n_x=16, n_y=16, half_steps=4))
    folder = tmp_path / "package"
    first = export.write_result(out, folder)
    before = {p.name: p.read_bytes() for p in folder.iterdir()}
    with pytest.raises(FileExistsError):
        export.write_result(out, folder)
    with pytest.raises(FileExistsError):
        report.write_package(out, folder)
    assert {p.name: p.read_bytes() for p in folder.iterdir()} == before and first["field"].exists()


def test_the_outcome_refuses_a_field_without_a_role_and_a_role_without_a_field():
    out = run("cccc_n1", req("CCCC", 1, n_x=16, n_y=16, half_steps=4))
    import dataclasses

    with pytest.raises(ValueError):
        dataclasses.replace(out, field_role=None)
    with pytest.raises(ValueError):
        dataclasses.replace(out, field_role="converged")
    assert isinstance(out, RunOutcome)


def test_a_schema_1_manifest_is_read_with_the_roles_of_schema_2(tmp_path):
    legacy = {
        "schema_version": 1, "data_origin": "solver", "marks": ["DIAGNOSTIC: the run ended with a solver error; not a result"],
        "input": {"supports": "SSSS", "N": 1}, "solver": {"stage": "R3"},
        "statuses": {"solver_status": "ERROR_LINEAR_SOLVE", "solver_reason": "", "mark": None, "stop": {"kind": "error"}},
        "results": {"s": 2, "field_available": True,
                    "field_role": "diagnostic only: field of the last completed half-step of a run that ended with an error"},
        "files": {"history": "history.csv", "field": "field.csv"},
    }
    (tmp_path / "m1.json").write_text(json.dumps(legacy), encoding="utf-8")
    read = export.read_manifest(tmp_path / "m1.json")
    assert read["legacy"] and read["field_role"] == "diagnostic" and read["field_file"] == "field.csv"
    converged = {**legacy, "statuses": {**legacy["statuses"], "solver_status": "CONVERGED"},
                 "results": {**legacy["results"], "field_role": "field of the last completed half-step"}}
    (tmp_path / "m2.json").write_text(json.dumps(converged), encoding="utf-8")
    assert export.read_manifest(tmp_path / "m2.json")["field_role"] == "accepted"
    (tmp_path / "m9.json").write_text(json.dumps({**legacy, "schema_version": 9}), encoding="utf-8")
    with pytest.raises(ValueError):
        export.read_manifest(tmp_path / "m9.json")
    out = run("n6_rn5", N6_RN5_REFUSAL)
    paths = export.write_result(out, tmp_path / "v2")
    read2 = export.read_manifest(paths["manifest"])
    assert not read2["legacy"] and read2["field_role"] == "diagnostic" and read2["field_file"] == "field_diagnostic.csv"
    assert read2["s_completed"] == 1 and read2["s_requested"] == 2 and read2["requested_run_completed"] is False


# --- the form -----------------------------------------------------------------------------------------------------------------

def test_the_form_never_starts_a_research_continuation_implicitly():
    state = SetupState().with_choice(supports="CCCC", N=3)
    assert state.cccc_half_steps == 1 and to_request(state).half_steps == 1 and to_request(state).route == ROUTE_CCCC_N23
    state = state.with_choice(cccc_half_steps=2)
    assert state.with_choice(N=4).cccc_half_steps == 1 and state.with_choice(supports="SSSS").cccc_half_steps == 1
    with pytest.raises(ValueError):
        state.with_choice(cccc_half_steps=3)
    ssss = SetupState().with_choice(N=5)
    assert "research" in validate(ssss) and not ssss.ssss_research
    chosen = ssss.with_choice(ssss_research=True)
    assert "research" not in validate(chosen) and to_request(chosen).ssss_research is True
    assert chosen.with_choice(N=6).ssss_research is False and chosen.with_choice(supports="CCCC").ssss_research is False
    assert SetupState().with_choice(N=2, r4_mode=R4_RESEARCH_STOP).with_choice(supports="CCCC").r4_mode == R4_NORMATIVE
