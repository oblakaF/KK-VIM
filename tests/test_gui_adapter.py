"""GUI backend adapter: the GUI calls kkvim unchanged and copies its results (GUI_DESIGN.md, 3.3-3.6). Qt-free.

- N = 1 is ``run_rectangle_physical``; N = 2-3 is ``validated_rectangle_input`` then ``run_coupled`` with the
  research mode only when it was chosen explicitly: the default is ``research_mode=False`` (D-048.3, D-048.5),
  and a change of N resets an explicit choice;
- dense only while max(n_x, n_y) <= 40 (D-048.6), checked by one rule in the form, the adapter and the process;
- the outcome equals the direct API call of the same process bitwise (status, D, history, field);
- the GUI never changes the solver arrays, and its copies are read-only;
- statuses and reasons of the library are kept unchanged; every R4 result carries PROVISIONAL / NOT_QUALIFIED;
- input rejected by the library checks is never computed (E < 0 and t < 0 for N = 2-3 included);
- export: JSON and CSV with units and statuses, values bit-identical to the outcome (repr round trip).
"""

import csv
import json
import multiprocessing
import time

import numpy as np
import pytest

import kkvim.coupled_run as cr
import kkvim.rectangle_run as rr
from kkvim.linear_solver import ERROR_LINEAR_SOLVE, LinearSolveError
from kkvim.rectangle import validated_rectangle_input

from kkvim_gui.backend import export, kkvim_adapter, units
from kkvim_gui.backend.process import CRASH, INPUT, OK, SolverProcess, compute
from kkvim_gui.backend.request import R4_NORMATIVE, R4_RESEARCH_FIXED, R4_RESEARCH_STOP, RunRequest
from kkvim_gui.model.setup_state import SetupState, to_request, validate

STEEL = dict(entered=(), a=1.0, b=1.5, t=0.01, E=2.1e11, nu=0.3, q=1000.0, n_x=24, n_y=36)
R3_DEFAULTS = dict(tol_inf=cr.TOL_INF, tol_E=1e-8, s_min=3, s_max=100)


def r3(**changes):
    return RunRequest(**{**STEEL, "N": 1, **R3_DEFAULTS, **changes})


def r4(N=2, mode=R4_RESEARCH_STOP, transform="fft", half_steps=100, **changes):
    return RunRequest(**{**STEEL, "N": N, "r4_mode": mode, "transform": transform, "half_steps": half_steps,
                         **changes})


def same(a, b):
    """Bitwise equality of numbers, arrays, None and strings."""
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.array_equal(np.asarray(a), np.asarray(b)) and np.asarray(a).dtype == np.asarray(b).dtype
    if isinstance(a, float) and isinstance(b, float):
        return a == b or (a != a and b != b)
    return a == b


# --- units and request ----------------------------------------------------------------------------------

def test_units_are_exact_si_prefixes():
    assert units.to_base(10.0, "mm") == 10.0 * 1e-3 and units.to_base(2.0, "m") == 2.0
    assert units.to_base(210.0, "GPa") == 210.0 * 1e9 and units.to_base(5.0, "kPa") == 5000.0
    assert units.base_unit("cm") == "m" and units.base_unit("MPa") == "Pa"
    with pytest.raises(ValueError):
        units.to_base(1.0, "inch")


def test_form_values_reach_the_solver_in_si_with_the_entered_text_kept():
    state = (SetupState().with_text("a", "2000").with_unit("a", "mm").with_text("h", "1,0").with_unit("h", "cm")
             .with_text("E", "210000").with_unit("E", "MPa").with_text("q", "5").with_unit("q", "kPa"))
    req = to_request(state)
    assert (req.a, req.t, req.E, req.q) == (2.0, 0.01, 2.1e11, 5000.0)
    assert ("a", "2000", "mm") in req.entered and ("h", "1,0", "cm") in req.entered
    assert req.N == 1 and (req.tol_inf, req.tol_E, req.s_min, req.s_max) == (1e-10, 1e-8, 3, 100)
    r4_req = to_request(state.with_choice(N=3, r4_mode=R4_NORMATIVE).with_text("half_steps", "5"))
    assert (r4_req.N, r4_req.r4_mode, r4_req.transform, r4_req.half_steps) == (3, R4_NORMATIVE, "fft", 5)
    assert r4_req.tol_inf is None  # R4 tolerances are those of D-048, not form fields


def test_defaults_come_from_kkvim():
    assert kkvim_adapter.DEFAULTS["tol_inf"] == rr.TOL_INF and kkvim_adapter.DEFAULTS["tol_E"] == rr.TOL_E
    assert kkvim_adapter.DEFAULTS["s_min"] == rr.S_MIN and kkvim_adapter.DEFAULTS["s_max"] == rr.S_MAX
    assert kkvim_adapter.R4_FIXED["s_max"] == cr.S_MAX and kkvim_adapter.R4_FIXED["eta_max"] == cr.ETA_MAX
    assert kkvim_adapter.N_AVAILABLE == (1, 2, 3, 4, 5, 6) and kkvim_adapter.MARK == cr.MARK  # every N has a route (P1)


# --- input checks ----------------------------------------------------------------------------------------

@pytest.mark.parametrize(("request_", "fragment"), [
    (r3(supports="CCCC"), "CCCC"),
    (r3(N=4), "N = 4"),
    (r3(nu=1.0), "ERROR_INPUT"),
    (r3(b=6.0), "ERROR_INPUT"),
    (r3(n_x=3), "ERROR_INPUT"),
    (r4(E=-2.1e11, t=-0.01), "ERROR_INPUT"),   # D > 0, but E and t are rejected like R3 (QA, observation 1)
    (r4(transform="dense", n_x=48, n_y=40), "D-048.6"),
])
def test_rejected_input_is_never_computed(monkeypatch, request_, fragment):
    calls = []
    monkeypatch.setattr(cr, "run_coupled", lambda *a, **k: calls.append(a))
    monkeypatch.setattr(kkvim_adapter, "run_rectangle_physical", lambda *a, **k: calls.append(a))
    problems = kkvim_adapter.check_request(request_)
    assert any(fragment in p for p in problems), problems
    with pytest.raises(kkvim_adapter.AdapterInputError):
        kkvim_adapter.execute(request_)
    assert calls == []


def test_negative_E_and_t_give_a_positive_D_but_are_rejected():
    assert (-2.1e11) * (-0.01) ** 3 > 0  # the reason for the explicit check
    assert kkvim_adapter.check_request(r4(E=-2.1e11, t=-0.01))
    assert kkvim_adapter.check_request(r4(transform="dense", n_x=40, n_y=40)) == []


# --- N = 2-3: research continuation only on explicit choice (D-048.3, D-048.5) ---------------------------

def spy_run_coupled(monkeypatch):
    seen, original = [], cr.run_coupled

    def spy(*args, **kwargs):
        seen.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(cr, "run_coupled", spy)
    return seen


def test_the_default_of_n_2_3_is_research_mode_false(monkeypatch):
    assert kkvim_adapter.DEFAULTS["r4_mode"] == R4_NORMATIVE
    req = RunRequest(**STEEL, N=2, transform="fft", half_steps=100)        # no mode given
    assert req.r4_mode == R4_NORMATIVE and req.research_mode is False
    seen = spy_run_coupled(monkeypatch)
    out = kkvim_adapter.execute(req)
    assert seen[0]["research_mode"] is False and seen[0]["stop_on_stabilization"] is True
    assert out.status == cr.NOT_QUALIFIED and out.s == 1 and out.mark == cr.MARK    # D-047 after s = 1
    assert dict(out.parameters)["research_mode"] is False


@pytest.mark.parametrize("N", [2, 3])
def test_choosing_n_2_3_in_the_form_does_not_enable_research(monkeypatch, N):
    state = SetupState().with_text("n_x", "24").with_text("n_y", "36").with_choice(N=N)
    assert state.r4_mode == R4_NORMATIVE and not state.research
    req = to_request(state)
    assert req.r4_mode == R4_NORMATIVE and req.research_mode is False
    seen = spy_run_coupled(monkeypatch)
    out = kkvim_adapter.execute(req)
    assert seen[0]["research_mode"] is False and out.status == cr.NOT_QUALIFIED and out.s == 1


@pytest.mark.parametrize("mode", [R4_RESEARCH_STOP, R4_RESEARCH_FIXED])
def test_a_change_of_n_resets_the_research_choice(mode):
    chosen = SetupState().with_choice(N=2).with_choice(r4_mode=mode)
    assert chosen.research and to_request(chosen).research_mode is True     # an explicit choice is kept ...
    assert chosen.with_text("n_x", "48").with_choice(transform="fft", N=2).r4_mode == mode  # ... while N stays
    for N in (3, 1):
        reset = chosen.with_choice(N=N)
        assert reset.r4_mode == R4_NORMATIVE and not reset.research          # ... and is reset by a change of N
        assert to_request(reset.with_choice(N=2)).research_mode is False
    assert SetupState().with_choice(N=3, r4_mode=mode).r4_mode == mode       # both in one explicit choice


# --- dense only while max(n_x, n_y) <= 40 (D-048.6) ------------------------------------------------------

DENSE_CASES = [((40, 40), True), ((40, 41), False), ((41, 40), False), ((40, 160), False), ((160, 40), False)]


@pytest.mark.parametrize(("size", "allowed"), DENSE_CASES)
def test_dense_scope_is_checked_on_both_sizes(monkeypatch, size, allowed):
    n_x, n_y = size
    req = r4(mode=R4_NORMATIVE, transform="dense", n_x=n_x, n_y=n_y, half_steps=3)
    message = kkvim_adapter.dense_problem("dense", n_x, n_y)
    assert (message is None) == allowed
    assert kkvim_adapter.dense_problem("fft", n_x, n_y) is None                # fft is not limited
    state = (SetupState().with_text("n_x", str(n_x)).with_text("n_y", str(n_y)).with_choice(N=2)
             .with_choice(transform="dense"))
    assert validate(state).get("transform") == message                         # the form, before Run
    assert kkvim_adapter.check_request(req) == ([] if allowed else [message])
    if allowed:
        seen = spy_run_coupled(monkeypatch)
        out = kkvim_adapter.execute(req)
        assert seen[0]["transform"] == "dense" and out.status == cr.NOT_QUALIFIED
        return
    assert "max(n_x, n_y) ≤ 40" in message and f"n_x = {n_x}, n_y = {n_y}" in message and "fft" in message
    monkeypatch.setattr(cr, "run_coupled", lambda *a, **k: pytest.fail("dense outside D-048.6 was computed"))
    with pytest.raises(kkvim_adapter.AdapterInputError, match="D-048.6"):
        kkvim_adapter.execute(req)
    parent, child = multiprocessing.Pipe(duplex=False)
    compute(req, child)                                                         # the body of the computing process
    kind, payload = parent.recv()
    assert kind == INPUT and "D-048.6" in payload


def test_dense_does_not_concern_n_1():
    state = SetupState().with_text("n_x", "160").with_text("n_y", "80").with_choice(transform="dense")
    assert "transform" not in validate(state) and to_request(state).transform is None


# --- results equal the direct API ------------------------------------------------------------------------

def test_n1_is_run_rectangle_physical_bitwise():
    req = r3()
    out = kkvim_adapter.execute(req)
    direct = rr.run_rectangle_physical(1.0, 1.5, 0.01, 2.1e11, 0.3, 1000.0, 24, 36)
    report = rr.halfstep_report(direct)
    assert out.call == "kkvim.rectangle_run.run_rectangle_physical" and out.solver == "R3"
    assert out.status == direct.status == "CONVERGED" and out.D == direct.D and out.s == direct.run.s
    assert same(out.w, direct.run.last_field()) and same(out.w_bar, direct.dimensionless_field())
    assert len(out.history) == len(report.halfsteps)
    for row, h in zip(out.history, report.halfsteps, strict=True):
        assert row["s"] == h.s and same(row["eta"], h.eta) and same(row["P"], h.P)
        if h.changes is not None:
            assert same(row["eps_inf"], h.changes.eps_inf) and same(row["eps_E"], h.changes.eps_E)
        if h.energy is not None:
            assert row["energy_status"] == h.energy.status
    assert out.mark is None and out.stop_kind == "met" and "CONVERGED" in out.stop_text


@pytest.mark.parametrize(("N", "mode", "half_steps", "research", "stop"), [
    (2, R4_RESEARCH_STOP, 100, True, True),
    (3, R4_RESEARCH_FIXED, 12, True, False),
    (2, R4_NORMATIVE, 5, False, True),
])
def test_n2_3_is_run_coupled_bitwise_with_the_explicit_mode(monkeypatch, N, mode, half_steps, research, stop):
    seen = []
    original = cr.run_coupled

    def spy(*args, **kwargs):
        seen.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(cr, "run_coupled", spy)
    out = kkvim_adapter.execute(r4(N=N, mode=mode, half_steps=half_steps))
    assert seen == [dict(transform="fft", half_steps=half_steps, research_mode=research, stop_on_stabilization=stop)]
    _, D = validated_rectangle_input(1.0, 1.5, 0.01, 2.1e11, 0.3, 1000.0, 24, 36)
    direct = original(1.0, 1.5, D, 1000.0, 24, 36, N, transform="fft", half_steps=half_steps,
                      research_mode=research, stop_on_stabilization=stop)
    assert out.status == direct.status and out.reason == direct.reason and out.s == direct.s and out.D == D
    assert same(out.w, direct.field()) and same(out.w_bar, direct.field() * D / (1000.0 * 1.0**4))
    for row, step in zip(out.history, direct.steps, strict=True):
        assert same(row["eps_inf"], step.eps_inf) and same(row["P"], step.energy.P) and row["mark"] == cr.MARK
        assert row["stop_reason"] == step.stop_reason and same(row["eta"], step.eta_N)
    assert out.mark == cr.MARK and out.science_kind == "warning" and cr.MARK in out.science_text
    if mode == R4_NORMATIVE:
        assert out.status == cr.NOT_QUALIFIED and out.s == 1 and out.stop_kind == "none"


def test_statuses_and_reasons_are_the_library_texts():
    for req in (r3(), r4(), r4(N=3, n_x=4)):
        out = kkvim_adapter.execute(req)
        assert out.status in out.stop_text  # the code of the library is shown unchanged
    rank = kkvim_adapter.execute(r4(N=3, n_x=4))
    assert rank.status == cr.STOPPED_RANK_LOSS_PROVEN and not rank.has_field and rank.s == 0
    assert rank.reason.startswith("n_x = 4 < 2N = 6") and not rank.is_error


# --- the GUI does not change the solver arrays -----------------------------------------------------------

def test_outcome_arrays_are_read_only_copies_and_the_solver_arrays_are_unchanged():
    req = r4(N=3)
    _, D = validated_rectangle_input(req.a, req.b, req.t, req.E, req.nu, req.q, req.n_x, req.n_y)
    run = kkvim_adapter.run_coupled_for(req, D)
    before = (run.x.copy(), run.y.copy(), run.field().copy())
    out = kkvim_adapter.outcome_from_r4(req, D, run)
    assert np.array_equal(run.x, before[0]) and np.array_equal(run.y, before[1])
    for array in (out.w, out.w_bar, out.x, out.y):
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array[0] = 1.0
    assert not np.shares_memory(out.w, before[2])
    run.x[:] = 0.0  # a later change of the solver object does not reach the GUI copy
    assert np.array_equal(out.w, before[2])


def test_r3_result_object_is_unchanged_by_the_adapter():
    result = rr.run_rectangle_physical(1.0, 1.5, 0.01, 2.1e11, 0.3, 1000.0, 24, 36)
    fields = [r.field().copy() for r in result.run.history.records]
    out = kkvim_adapter.outcome_from_r3(r3(), result)
    assert all(np.array_equal(r.field(), f) for r, f in zip(result.run.history.records, fields, strict=True))
    assert same(out.w, fields[-1])


# --- errors are not results ------------------------------------------------------------------------------

def test_r3_error_with_partial_history_is_an_error_with_a_diagnostic_field(monkeypatch):
    import kkvim.rectangle_run as module

    original = module.next_halfstep

    def failing(record):
        if record.s == 3:
            raise LinearSolveError(ERROR_LINEAR_SOLVE, "injected")
        return original(record)

    monkeypatch.setattr(module, "next_halfstep", failing)
    out = kkvim_adapter.execute(r3())
    assert out.status == ERROR_LINEAR_SOLVE and out.is_error and out.stop_kind == "error"
    assert out.s == 3 and out.has_field and out.diagnostic_only
    data = export.manifest(out)
    assert data["results"]["completed_without_error"] is False
    assert data["results"]["field_role"].startswith("diagnostic only")
    assert any(m.startswith("DIAGNOSTIC") for m in data["marks"])


# --- export ----------------------------------------------------------------------------------------------

def _read_csv(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    comments = [line for line in lines if line.startswith("#")]
    rows = list(csv.reader(line for line in lines if not line.startswith("#")))
    return comments, rows[0], rows[1:]


def _value(text):
    if text == "":
        return None
    if text in ("true", "false"):
        return text == "true"
    try:
        return int(text)
    except ValueError:
        try:
            return float(text)
        except ValueError:
            return text


@pytest.mark.parametrize("make", [r3, lambda: r4(N=3)])
def test_export_is_bit_identical_with_units_and_statuses(tmp_path, make):
    out = kkvim_adapter.execute(make())
    paths = export.write_result(out, tmp_path / "res", {"run_id": 1}, [("10:00:00", "ИНФО", "x")])
    data = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert data["data_origin"] == "solver" and data["statuses"]["solver_status"] == out.status
    assert data["statuses"]["mark"] == out.mark and data["statuses"]["stop"]["text"] == out.stop_text
    assert data["input"]["si"]["E"] == {"value": 2.1e11, "unit": "Pa"} and data["results"]["D"]["unit"] == "N·m"
    assert data["results"]["extremum_by_nodes"]["w"]["unit"] == "m"
    assert data["results"]["D"]["value"] == out.D and data["solver"]["call"] == out.call
    comments, header, rows = _read_csv(paths["history"])
    if out.mark:
        assert comments[0] == "# PROVISIONAL / NOT_QUALIFIED" and data["marks"] == [out.mark]
    else:
        assert data["marks"] == []
    for c, h in zip(out.columns, header, strict=True):  # every number column carries its unit, text columns none
        assert ("[" in h) == (c.kind == "number" and c.key not in ("s", "k")), h
    for row, stored in zip(rows, out.history, strict=True):
        for c, text in zip(out.columns, row, strict=True):
            expected = stored[c.key]
            got = _value(text)
            assert same(got, expected) or (expected is None and got is None), (c.key, text, expected)
    comments, header, rows = _read_csv(paths["field"])
    assert header == ["i", "j", "x [m]", "y [m]", "w [m]", "w_bar [1]"]
    w = np.array([float(r[4]) for r in rows]).reshape(out.w.shape)
    w_bar = np.array([float(r[5]) for r in rows]).reshape(out.w.shape)
    assert np.array_equal(w, out.w) and np.array_equal(w_bar, out.w_bar)


def test_a_run_without_a_field_exports_no_field(tmp_path):
    out = kkvim_adapter.execute(r4(N=3, n_x=4))
    paths = export.write_result(out, tmp_path / "res")
    assert "field" not in paths and not (tmp_path / "res" / "field.csv").exists()
    data = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert data["results"]["field_available"] is False and data["results"]["extremum_by_nodes"] is None


# --- the separate computing process ----------------------------------------------------------------------

def _wait(process, timeout=120.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        message = process.poll()
        if message is not None:
            return message
        time.sleep(0.05)
    raise AssertionError("no message from the computing process")


def test_the_process_returns_the_same_outcome_as_the_library_in_process():
    kind, out = _wait(SolverProcess(r4(N=3)))
    assert kind == OK
    local = kkvim_adapter.execute(r4(N=3))
    assert out.status == local.status and out.D == local.D and np.array_equal(out.w, local.w)
    assert [r["eps_E"] for r in out.history] == [r["eps_E"] for r in local.history]


def test_the_process_reports_rejected_input_and_crashes_without_stopping_the_gui():
    kind, text = _wait(SolverProcess(r3(nu=1.0)))
    assert kind == INPUT and "ERROR_INPUT" in text
    kind, text = _wait(SolverProcess(None))  # not a request: the computation fails inside the process
    assert kind == CRASH and "Traceback" in text


def test_stop_terminates_a_long_computation():
    long = RunRequest(**{**STEEL, "a": 1.0, "b": 1.0, "n_x": 1280, "n_y": 1280, "N": 3,
                         "r4_mode": R4_RESEARCH_FIXED, "transform": "fft", "half_steps": 100})
    process = SolverProcess(long)
    time.sleep(1.0)
    assert process.poll() is None  # still computing
    start = time.monotonic()
    assert process.terminate() is True
    assert time.monotonic() - start < 10.0
