"""``python -m kkvim run-ssss``: the user-facing command for SSSS, explicit N = 1..6 (``kkvim.cli_ssss``; D-060).

The tests run the command itself and compare what it wrote with the direct call of the route:
- N = 1: the files of ``python -m kkvim run`` and of the R3 API;
- N = 2-3: ``run_coupled`` bit for bit, its statuses unchanged;
- N = 4-6: labels, refusals and safeguards here; agreement with the study and with the independent control in
  tests/cross/test_cli_ssss_research.py.
Values of the routes are compared bit for bit or exactly. No numerical tolerance of the method is chosen here; the only numbers
of this kind are the local structural check of the singular values in ``test_the_field_keeps_n_and_the_half_steps_alternate``, which is
a smoke check of this test and neither a criterion of the method nor a rank certificate.
"""

import csv
import dataclasses
import json
import os
import re
import shutil
import subprocess
import sys
import time

import numpy as np
import pytest

import kkvim.coupled as kc
import kkvim.coupled_run as cr
import kkvim.rectangle_run as rr
from kkvim import cli, cli_ssss
from kkvim.coefficients import ERROR_INPUT
from kkvim.controller import CONVERGED, MAX_HALF_STEPS
from kkvim.physical import bending_stiffness
from kkvim.research_ssss import S_STUDY, control_ratio, domain_refusal

PLATE = dict(a=1.0, b=1.0, t=0.01, E=2.1e11, nu=0.3, q=1000.0)
D = bending_stiffness(PLATE["E"], PLATE["t"], PLATE["nu"])


def plate(**changes):
    values = {**PLATE, "nx": 40, "ny": 40, **changes}
    args = []
    for name, value in values.items():
        args += [f"--{name}", repr(value) if isinstance(value, float) else str(value)]
    return args


def command(out, N, *extra, half_steps=None, **changes):
    args = ["run-ssss", "--N", str(N), *plate(**changes), "--out", str(out), *extra]
    if half_steps is not None:
        args += ["--half-steps", str(half_steps)]
    if N >= 4:
        args.append("--research-only")
    return args


def document(out):
    return json.loads((out / "result.json").read_text(encoding="utf-8"))


def table(path):
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if not line.startswith("#")]
    return list(csv.DictReader(lines))


def field_arrays(path, shape):
    w, w_bar = np.empty(shape), np.empty(shape)
    rows = table(path)
    assert len(rows) == shape[0] * shape[1]
    for row in rows:
        w[int(row["i"]), int(row["j"])] = float(row["w [m]"])
        w_bar[int(row["i"]), int(row["j"])] = float(row["w_bar [1]"])
    return w, w_bar


def names(out):
    return sorted(path.name for path in out.iterdir())


# --- N = 1: the accepted R3 path ------------------------------------------------------------------------------------------

def test_n1_equals_the_run_command_and_the_r3_api(tmp_path):
    old, new = tmp_path / "run", tmp_path / "ssss"
    assert cli.main(["run", *plate(nx=24, ny=24), "--out", str(old)]) == cli.EXIT_CONVERGED
    assert cli.main(command(new, 1, nx=24, ny=24)) == cli.EXIT_CONVERGED
    assert names(old) == names(new) == ["field.csv", "history.csv", "result.json"]
    assert (old / "field.csv").read_bytes() == (new / "field.csv").read_bytes()
    assert (old / "history.csv").read_bytes() == (new / "history.csv").read_bytes()
    direct = rr.run_rectangle_physical(**PLATE, n_x=24, n_y=24)
    w, w_bar = field_arrays(new / "field.csv", (25, 25))
    assert np.array_equal(w, direct.run.last_field()) and np.array_equal(w_bar, direct.dimensionless_field())
    result = document(new)
    assert direct.status == result["status"] == CONVERGED and result["half_steps"] == result["half_steps_completed"] == 7
    r3 = document(old)  # the keys of the R3 document have the same values, apart from those that name this command
    for key in set(r3) - {"schema", "program"}:
        assert result[key] == r3[key]
    assert result["solver_path"]["mode"] == "R3_ACCEPTED_SOLVER"
    assert result["scientific_applicability"]["accepted_by"] == "D-043"
    assert result["N"] == 1 and result["boundary_conditions"] == "SSSS"


def test_n1_keeps_the_stopping_options_and_statuses_of_r3(tmp_path):
    out = tmp_path / "limit"
    assert cli.main(command(out, 1, "--s-max", "3", nx=24, ny=24)) == cli.EXIT_NOT_CONVERGED
    assert names(out) == ["field_diagnostic.csv", "history.csv", "result.json"]
    assert document(out)["status"] == MAX_HALF_STEPS and document(out)["execution"] == "STOPPED"


GRID_WARNING = "не удовлетворяет необходимым условиям квалификационной сетки"


def grid_warnings(text):
    return [line for line in text.splitlines() if GRID_WARNING in line]


def both_commands(tmp_path, capsys, extra=(), **changes):
    """The run command and run-ssss --N 1 on one plate: exit code and printed lines of each."""
    old, new = tmp_path / "run", tmp_path / "ssss"
    code_run = cli.main(["run", *plate(**changes), "--out", str(old), *extra])
    printed_run = capsys.readouterr().out
    code_ssss = cli.main(command(new, 1, *extra, **changes))
    printed_ssss = capsys.readouterr().out
    return old, new, (code_run, printed_run), (code_ssss, printed_ssss)


@pytest.mark.parametrize("nx, ny, status, extra", [(30, 30, CONVERGED, ()), (16, 16, CONVERGED, ()), (24, 30, CONVERGED, ()),
                                                   (30, 30, MAX_HALF_STEPS, ("--s-max", "3"))])
def test_n1_prints_the_grid_warning_of_the_run_command(tmp_path, capsys, nx, ny, status, extra):
    old, new, (code_run, printed_run), (code_ssss, printed_ssss) = both_commands(tmp_path, capsys, extra, nx=nx, ny=ny)
    assert code_ssss == code_run
    assert grid_warnings(printed_ssss) == grid_warnings(printed_run)
    assert grid_warnings(printed_ssss) == [
        f"Сетка {nx}×{ny} не удовлетворяет необходимым условиям квалификационной сетки "
        "R3_NUMERICAL_SPEC.md 6.1 (min ≥ 20, max ≤ 640, кратность 4): расчёт диагностический."]
    assert document(new)["status"] == document(old)["status"] == status  # the warning does not change the status
    assert f"Статус: {status}" in printed_ssss


def test_n1_prints_the_grid_warning_of_the_run_command_after_an_input_error(tmp_path, capsys):
    _, new, (code_run, printed_run), (code_ssss, printed_ssss) = both_commands(tmp_path, capsys, nu=1.5, nx=30, ny=30)
    assert code_ssss == code_run == cli.EXIT_SOLVER_ERROR
    assert document(new)["status"] == ERROR_INPUT
    assert grid_warnings(printed_ssss) == grid_warnings(printed_run) != []


def test_n1_on_a_grid_with_the_necessary_conditions_prints_no_grid_warning(tmp_path, capsys):
    _, new, (code_run, printed_run), (code_ssss, printed_ssss) = both_commands(tmp_path, capsys, nx=24, ny=24)
    assert code_ssss == code_run == cli.EXIT_CONVERGED
    assert grid_warnings(printed_run) == grid_warnings(printed_ssss) == []
    assert not [value for key, value in document(new)["applicability"]["necessary_grid_conditions_6_1"].items()
                if key != "note" and value is not True]


@pytest.mark.parametrize("nx, ny", [(30, 30), (24, 24)])
def test_the_grid_warning_changes_no_saved_value(tmp_path, capsys, nx, ny):
    old, new, *_ = both_commands(tmp_path, capsys, nx=nx, ny=ny)
    assert names(new) == ["field.csv", "history.csv", "result.json"]
    assert (new / "field.csv").read_bytes() == (old / "field.csv").read_bytes()
    assert (new / "history.csv").read_bytes() == (old / "history.csv").read_bytes()
    r3 = document(old)
    result = document(new)
    for key in set(r3) - {"schema", "program"}:
        assert result[key] == r3[key]
    assert result["status"] == CONVERGED and result["converged"] is True
    assert result["applicability"]["necessary_grid_conditions_6_1"] == cli.grid_conditions(nx, ny)


def test_summary_writes_nothing_and_reads_the_grid_conditions_from_the_document(tmp_path, capsys):
    out = tmp_path / "out"
    assert cli.main(command(out, 1, nx=30, ny=30)) == cli.EXIT_CONVERGED
    capsys.readouterr()
    before = {path.name: path.read_bytes() for path in out.iterdir()}
    args = cli.build_parser().parse_args(command(out, 1, "--overwrite", nx=30, ny=30))
    texts = {cli.RESULT: before[cli.RESULT].decode("utf-8")}
    cli_ssss.summary(args, CONVERGED, texts, sorted(before))
    assert grid_warnings(capsys.readouterr().out) != []
    conditions = json.loads(texts[cli.RESULT])["applicability"]["necessary_grid_conditions_6_1"]
    texts[cli.RESULT] = json.dumps({**json.loads(texts[cli.RESULT]), "applicability": {
        "necessary_grid_conditions_6_1": {**{key: True for key in conditions}, "note": conditions["note"]}}})
    cli_ssss.summary(args, CONVERGED, texts, sorted(before))
    assert grid_warnings(capsys.readouterr().out) == []
    assert {path.name: path.read_bytes() for path in out.iterdir()} == before


@pytest.mark.parametrize("N, half_steps, nx", [(2, 2, 30), (3, 2, 30), (4, 2, 40), (6, 2, 40)])
def test_n2_to_n6_have_no_grid_warning_of_r3(tmp_path, capsys, N, half_steps, nx):
    assert cli.main(command(tmp_path / "out", N, half_steps=half_steps, nx=nx, ny=nx)) == cli.EXIT_CONVERGED
    assert grid_warnings(capsys.readouterr().out) == []


# --- N = 2-3: run_coupled -------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("N, changes", [(2, {}), (3, {}), (2, {"b": 2.0, "nx": 40, "ny": 80}),
                                        (3, {"a": 2.0, "b": 0.4, "nx": 80, "ny": 40, "q": -500.0})])
def test_n2_n3_equal_run_coupled_bitwise(tmp_path, N, changes):
    out = tmp_path / "out"
    assert cli.main(command(out, N, half_steps=8, **changes)) == cli.EXIT_CONVERGED
    values = {**PLATE, "nx": 40, "ny": 40, **changes}
    direct = cr.run_coupled(values["a"], values["b"], D, values["q"], values["nx"], values["ny"], N, transform="fft",
                            half_steps=8, research_mode=True, stop_on_stabilization=True)
    assert direct.status == cr.RESEARCH_LIMIT_REACHED and document(out)["status"] == direct.status
    assert names(out) == ["field_research.csv", "history.csv", "result.json"]
    shape = (values["nx"] + 1, values["ny"] + 1)
    w, w_bar = field_arrays(out / "field_research.csv", shape)
    assert np.array_equal(w, direct.field())
    assert np.array_equal(w_bar, direct.field() / (values["q"] * values["a"]**4 / D))
    rows = table(out / "history.csv")
    assert [int(row["s"]) for row in rows] == [log.s for log in direct.steps]
    for row, log in zip(rows, direct.steps):
        assert float(row["rho_N"]) == log.rank.rho_N and float(row["kappa_rep"]) == log.rank.kappa_rep
        assert float(row["w_max"]) == log.w_max and float(row["P"]) == log.energy.P
        assert float(row["eta_N"]) == log.eta_N and row["energy_status"] == log.energy.status
        assert (row["qr_done"] == "true") == log.qr_done and row["stop_reason"] == (log.stop_reason or "")
    result = document(out)
    assert result["numerical_parameters"]["K_research"] == cr.K_RESEARCH and result["numerical_parameters"]["N"] == N


def test_n2_stops_by_the_stabilization_of_d048_and_fixed_half_steps_only_record_it(tmp_path):
    stopped, fixed = tmp_path / "stopped", tmp_path / "fixed"
    assert cli.main(command(stopped, 2)) == cli.EXIT_CONVERGED
    direct = cr.run_coupled(1.0, 1.0, D, 1000.0, 40, 40, 2, transform="fft", half_steps=cr.S_MAX, research_mode=True)
    result = document(stopped)
    assert result["status"] == direct.status == cr.PROVISIONAL_FIELD_STABILIZED and result["half_steps_completed"] == direct.s
    assert result["first_stabilization_s"] == direct.first_stabilization_s
    half_steps = direct.s + 2
    assert cli.main(command(fixed, 2, "--fixed-half-steps", half_steps=half_steps)) == cli.EXIT_CONVERGED
    result = document(fixed)
    assert result["status"] == cr.RESEARCH_LIMIT_REACHED and result["half_steps_completed"] == half_steps
    assert result["first_stabilization_s"] == direct.first_stabilization_s
    assert result["input"]["fixed_half_steps"] is True


def test_n3_outside_the_area_of_d048_is_computed_and_marked(tmp_path, capsys):
    inside, outside = tmp_path / "inside", tmp_path / "outside"
    assert cli.main(command(inside, 3, half_steps=2)) == cli.EXIT_CONVERGED
    assert document(inside)["scientific_applicability"]["domain"]["inside_area"] is True
    capsys.readouterr()
    assert cli.main(command(outside, 3, half_steps=2, b=1.2, ny=48)) == cli.EXIT_CONVERGED
    area = document(outside)["scientific_applicability"]["domain"]
    assert area["inside_area"] is False and area["control_r"] is None
    assert "Вне области D-048" in capsys.readouterr().out


# --- routing, N is never reduced --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("N, mode", [(2, "R4_LIMITED_RESEARCH_MODE"), (3, "R4_LIMITED_RESEARCH_MODE"), (4, "RESEARCH_ONLY_N4_N6"),
                                     (5, "RESEARCH_ONLY_N4_N6"), (6, "RESEARCH_ONLY_N4_N6")])
def test_the_field_keeps_n_and_the_half_steps_alternate(tmp_path, N, mode):
    out = tmp_path / "out"
    assert cli.main(command(out, N, half_steps=5)) == cli.EXIT_CONVERGED
    result = document(out)
    assert result["N"] == N and result["numerical_parameters"]["N"] == N and result["solver_path"]["mode"] == mode
    rows = table(out / "history.csv")
    assert [row["direction"] for row in rows] == ["Y", "X", "Y", "X", "Y"]
    assert [int(row["s"]) for row in rows] == [1, 2, 3, 4, 5] and [int(row["k"]) for row in rows] == [0, 1, 1, 2, 2]
    assert [row["qr_done"] for row in rows] == ["true"] * 4 + ["false"]  # no QR after the last half-step
    assert all(row["mark"] == cr.MARK for row in rows)
    assert rows[-1]["stop_reason"] == cr.RESEARCH_LIMIT_REACHED
    # smoke check of this test, its numbers are not criteria of the method: the written field is a sum of N products
    _, w_bar = field_arrays(out / "field_research.csv", (41, 41))
    sigma = np.linalg.svd(w_bar, compute_uv=False)
    assert sigma[N - 1] > 1e3 * sigma[N] and sigma[N] < 1e-12 * sigma[0]


def test_the_production_range_and_its_guard_stay(tmp_path):
    assert cli.main(command(tmp_path / "out", 4, half_steps=2)) == cli.EXIT_CONVERGED
    assert kc.N_SOLVE == cr.N_SOLVE == (2, 3)
    assert cr.run_coupled(1.0, 1.0, D, 1.0, 40, 40, 4, transform="fft", half_steps=2, research_mode=True).status == ERROR_INPUT


def test_the_package_path_does_not_use_the_references():
    code = "import sys, kkvim.cli_ssss, kkvim.research_ssss; sys.exit('kkvim_reference' in sys.modules)"
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0


# --- labels -----------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("N, changes, half_steps", [(2, {}, 3), (3, {}, None), (4, {}, 3), (5, {}, 3), (6, {}, 3),
                                                    (6, {"b": 0.2}, 3), (3, {"nx": 4}, 2), (5, {"b": 1.2}, 3), (4, {"nu": 1.5}, 3)])
def test_no_research_output_claims_convergence_or_qualification(tmp_path, capsys, N, changes, half_steps):
    out = tmp_path / "out"
    cli.main(command(out, N, half_steps=half_steps, **changes))
    text = capsys.readouterr().out + "".join(path.read_text(encoding="utf-8") for path in out.iterdir())
    assert "CONVERGED" not in text and not re.search(r"(?<!NOT_)QUALIFIED", text) and "PASS" not in text
    if N >= 4:
        assert "ACCEPTED" not in text and "RESEARCH_ONLY" in text
    result = document(out)
    assert not (out / "field.csv").exists()
    assert result.get("converged", False) is False
    scientific = result["scientific_applicability"]
    assert scientific["mark"] == cr.MARK and scientific["accepted_solver"] is False and scientific["qualified"] is False
    assert scientific["research_only"] is (N >= 4)


def test_research_fields_are_marked_in_the_file(tmp_path):
    out = tmp_path / "out"
    assert cli.main(command(out, 5, half_steps=3)) == cli.EXIT_CONVERGED
    assert (out / "field_research.csv").read_text(encoding="utf-8").startswith(f"# RESEARCH: {cr.MARK}")
    assert (out / "history.csv").read_text(encoding="utf-8").startswith(f"# {cr.MARK}")
    field = document(out)["field"]
    assert field["role"] == "research" and field["file"] == "field_research.csv"
    assert field["w_max"]["value"] == float(np.max(np.abs(field_arrays(out / "field_research.csv", (41, 41))[0])))


# --- guards -----------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("b", [0.2, 1.0 / 3.0])
def test_the_rank_guard_of_the_study_stops_n6_before_the_qr(tmp_path, b):
    """N = 6, r = 1/5, 1/3: PROVISIONAL_RANK_GUARD_STOP at s = 1 in all 14 cells of the study (D-056.7, item 2)."""
    out = tmp_path / "out"
    assert cli.main(command(out, 6, half_steps=8, b=b)) == cli_ssss.EXIT_GUARD_STOP
    result = document(out)
    assert result["status"] == cr.PROVISIONAL_RANK_GUARD_STOP and result["half_steps_completed"] == 1
    assert result["N"] == 6 and result["execution"] == "STOPPED" and result["field"]["role"] == "diagnostic"
    assert names(out) == ["field_diagnostic.csv", "history.csv", "result.json"]
    rows = table(out / "history.csv")
    assert len(rows) == 1 and rows[0]["qr_done"] == "false" and rows[0]["guard_rank_ok"] == "false"
    assert (out / "field_diagnostic.csv").read_text(encoding="utf-8").startswith("# DIAGNOSTIC")
    assert "не доказанная потеря ранга" in result["status_note"]


@pytest.mark.parametrize("N", [3, 4])
@pytest.mark.parametrize("failed, status", [("rank", cr.PROVISIONAL_RANK_GUARD_STOP), ("set", cr.PROVISIONAL_SET_GUARD_STOP)])
def test_a_guard_failing_later_stops_both_routes_before_the_qr(tmp_path, monkeypatch, N, failed, status):
    real, calls = cr._guards, []

    def guards(rank):
        calls.append(rank)
        result = real(rank)
        if len(calls) < 3:
            return result
        return cr.GuardResult(finite=True, applicable=True, rank_ok=failed != "rank", set_ok=failed != "set",
                              rank_threshold=result.rank_threshold, set_threshold=result.set_threshold)

    monkeypatch.setattr(cr, "_guards", guards)
    out = tmp_path / "out"
    assert cli.main(command(out, N, half_steps=6)) == cli_ssss.EXIT_GUARD_STOP
    result = document(out)
    assert result["status"] == status and result["half_steps_completed"] == 3 and result["N"] == N
    assert [row["qr_done"] for row in table(out / "history.csv")] == ["true", "true", "false"]
    assert names(out) == ["field_diagnostic.csv", "history.csv", "result.json"]


@pytest.mark.parametrize("N, nx", [(3, 4), (6, 10)])
def test_a_proven_rank_loss_ends_before_the_first_half_step(tmp_path, N, nx):
    out = tmp_path / "out"
    assert cli.main(command(out, N, half_steps=2, nx=nx)) == cli_ssss.EXIT_GUARD_STOP
    result = document(out)
    assert result["status"] == cr.STOPPED_RANK_LOSS_PROVEN and f"< 2N = {2 * N}" in result["reason"]
    assert result["N"] == N and result["half_steps_completed"] == 0 and result["field"] is None
    assert names(out) == ["history.csv", "result.json"] and table(out / "history.csv") == []


# --- refusals ---------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("N", [1, 2, 3, 4, 6])
@pytest.mark.parametrize("name, value", [("nu", 1.0), ("t", -0.01), ("E", 0.0), ("a", 0.0), ("q", 0.0), ("b", 6.0), ("nx", 1)])
def test_invalid_physical_input_is_error_input_without_values(tmp_path, N, name, value):
    out = tmp_path / "out"
    assert cli.main(command(out, N, half_steps=None if N == 1 else 2, **{name: value})) == cli.EXIT_SOLVER_ERROR
    result = document(out)
    assert result["status"] == ERROR_INPUT and result["half_steps_completed"] == 0 and result["reason"]
    assert result["field"] is None and result["D"] is None
    assert names(out) == ["history.csv", "result.json"]


@pytest.mark.parametrize("args, fragment", [
    (dict(N=5, b=1.2, half_steps=4), "ten r"), (dict(N=4, nx=48, ny=48, half_steps=4), "was not computed"),
    (dict(N=4, nx=40, ny=80, half_steps=4), "was not computed"), (dict(N=6, half_steps=S_STUDY + 1), "s <= 40"),
    (dict(N=5, b=5.0, nx=1600, ny=320, half_steps=4), "was not computed"),
])
def test_input_outside_the_study_is_refused_with_its_reason(tmp_path, monkeypatch, args, fragment):
    monkeypatch.setattr(cli_ssss, "run_research", lambda *a, **k: pytest.fail("the refused input was computed"))
    out = tmp_path / "out"
    assert cli.main(command(out, args.pop("N"), half_steps=args.pop("half_steps"), **args)) == cli.EXIT_USAGE
    result = document(out)
    assert result["status"] == cli_ssss.NOT_EXECUTED and result["execution"] == "NOT_EXECUTED"
    assert fragment in result["reason"] and result["field"] is None and result["half_steps_completed"] == 0
    assert names(out) == ["history.csv", "result.json"] and table(out / "history.csv") == []


@pytest.mark.parametrize("a, b, nx, ny", [(1.0, 1.0, 1280, 1280), (0.1, 0.3, 40, 40), (2.0, 10.0, 320, 1600), (5.0, 1.0, 1600, 320),
                                          (0.7, 0.98, 160, 160)])
def test_the_study_matrix_is_inside(a, b, nx, ny):
    assert domain_refusal(a, b, nx, ny, 4, S_STUDY) is None


@pytest.mark.parametrize("bc", ["CCCC", "cccc", "SSCC", "ssss"])
def test_other_supports_are_refused_and_nothing_is_created(tmp_path, monkeypatch, capsys, bc):
    monkeypatch.setattr(cli_ssss, "execute_research", lambda *a: pytest.fail("computed"))
    out = tmp_path / "out"
    assert cli.main(command(out, 3, "--bc", bc, half_steps=2)) == cli.EXIT_USAGE
    err = capsys.readouterr().err
    assert not out.exists() and "run-cccc-n1" in err and "run-cccc-n23" in err and "RESEARCH_ONLY / PROVISIONAL / NOT_QUALIFIED" in err and "N ≥ 4" in err


@pytest.mark.parametrize("N, extra, half_steps, fragment", [
    (1, ("--fixed-half-steps",), None, "не применяется"), (1, ("--research-only",), None, "не применяется"),
    (1, (), 5, "не применяется"), (2, ("--tol-E", "1e-6"), 5, "только для N = 1"), (4, ("--s-max", "5"), 5, "только для N = 1"),
    (3, ("--research-only",), 5, "N = 4–6"), (3, ("--fixed-half-steps",), None, "--half-steps"), (3, (), 0, "положительным"),
    (3, (), cr.S_MAX + 1, "s_max"), (5, (), None, "--half-steps"), (5, (), -1, "положительным"),
])
def test_options_that_do_not_apply_are_refused(tmp_path, capsys, N, extra, half_steps, fragment):
    out = tmp_path / "out"
    assert cli.main(command(out, N, *extra, half_steps=half_steps)) == cli.EXIT_USAGE
    assert not out.exists() and fragment in capsys.readouterr().err


@pytest.mark.parametrize("N", [4, 5, 6])
def test_n4_n6_need_the_explicit_research_option(tmp_path, capsys, N):
    out = tmp_path / "out"
    assert cli.main(["run-ssss", "--N", str(N), *plate(), "--out", str(out), "--half-steps", "2"]) == cli.EXIT_USAGE
    assert not out.exists() and "--research-only" in capsys.readouterr().err


@pytest.mark.parametrize("N", ["0", "7", "-1", "2.5", "x"])
def test_n_outside_1_to_6_is_a_command_line_error(tmp_path, N):
    out = tmp_path / "out"
    with pytest.raises(SystemExit) as exit_:
        cli.main(["run-ssss", "--N", N, *plate(), "--out", str(out)])
    assert exit_.value.code == cli.EXIT_USAGE and not out.exists()


def test_n_must_be_given(tmp_path):
    with pytest.raises(SystemExit) as exit_:
        cli.main(["run-ssss", *plate(), "--out", str(tmp_path / "out")])
    assert exit_.value.code == cli.EXIT_USAGE


def test_the_run_command_has_no_n_option(tmp_path):
    with pytest.raises(SystemExit) as exit_:
        cli.main(["run", "--N", "3", *plate(), "--out", str(tmp_path / "out")])
    assert exit_.value.code == cli.EXIT_USAGE


# --- output -----------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("N", [1, 3, 5])
def test_the_recorded_input_reproduces_the_run(tmp_path, N):
    first, second = tmp_path / "first", tmp_path / "second"
    assert cli.main(command(first, N, half_steps=None if N == 1 else 4)) == cli.EXIT_CONVERGED
    result = document(first)
    argv = list(result["program"]["argv"])
    argv[argv.index("--out") + 1] = str(second)
    assert cli.main(argv) == cli.EXIT_CONVERGED
    assert names(first) == names(second)
    for name in names(first):
        if name != "result.json":
            assert (first / name).read_bytes() == (second / name).read_bytes()
    repeated = document(second)
    for key in ("N", "boundary_conditions", "input", "status", "half_steps_completed", "solver_path", "D", "provenance"):
        assert repeated[key] == result[key]
    assert result["input"]["a"] == {"value": 1.0, "unit": "m"} and result["input"]["q"] == {"value": 1000.0, "unit": "Pa"}
    assert result["input"]["n_x"] == 40 and result["provenance"]["versions"]["numpy"] == np.__version__


@pytest.mark.parametrize("N", [1, 3, 6])
def test_existing_results_are_not_overwritten(tmp_path, monkeypatch, capsys, N):
    out = tmp_path / "out"
    half_steps = None if N == 1 else 2
    assert cli.main(command(out, N, half_steps=half_steps)) == cli.EXIT_CONVERGED
    before = {name: (out / name).read_bytes() for name in names(out)}
    capsys.readouterr()
    monkeypatch.setattr(cli_ssss, "execute_r3", lambda *a: pytest.fail("computed"))
    monkeypatch.setattr(cli_ssss, "execute_research", lambda *a: pytest.fail("computed"))
    assert cli.main(command(out, N, half_steps=half_steps)) == cli.EXIT_USAGE
    assert "--overwrite" in capsys.readouterr().err
    assert {name: (out / name).read_bytes() for name in names(out)} == before


def test_a_valid_overwrite_leaves_only_the_files_of_the_new_run(tmp_path):
    out = tmp_path / "out"
    assert cli.main(command(out, 1)) == cli.EXIT_CONVERGED
    assert (out / "field.csv").exists()
    (out / "other.txt").write_text("kept", encoding="utf-8")
    assert cli.main(command(out, 5, "--overwrite", half_steps=2)) == cli.EXIT_CONVERGED
    assert names(out) == ["field_research.csv", "history.csv", "other.txt", "result.json"]
    assert cli.main(command(out, 6, "--overwrite", half_steps=2, b=0.2)) == cli_ssss.EXIT_GUARD_STOP
    assert names(out) == ["field_diagnostic.csv", "history.csv", "other.txt", "result.json"]
    assert cli.main(command(out, 5, "--overwrite", half_steps=2)) == cli.EXIT_CONVERGED
    assert cli.main(["run", *plate(), "--out", str(out), "--overwrite"]) == cli.EXIT_CONVERGED  # a stale research field goes too
    assert names(out) == ["field.csv", "history.csv", "other.txt", "result.json"]


def test_a_write_error_is_exit_4_and_leaves_no_partial_file(tmp_path, monkeypatch, capsys):
    out = tmp_path / "out"

    def fail(*args):
        raise PermissionError("locked")

    monkeypatch.setattr(os, "replace", fail)
    assert cli.main(command(out, 4, half_steps=2)) == cli.EXIT_OUTPUT_ERROR
    assert names(out) == [] and "locked" in capsys.readouterr().err


def test_an_unusable_folder_costs_no_run(tmp_path, monkeypatch):
    monkeypatch.setattr(cli_ssss, "execute_research", lambda *a: pytest.fail("computed"))
    (tmp_path / "file").write_text("x", encoding="utf-8")
    assert cli.main(command(tmp_path / "file", 3, half_steps=2)) == cli.EXIT_USAGE
    assert cli.main(command(tmp_path / "file" / "under", 3, half_steps=2)) == cli.EXIT_OUTPUT_ERROR


def test_an_internal_error_is_exit_5(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cr, "run_coupled", lambda *a, **k: 1 / 0)
    assert cli.main(command(tmp_path / "out", 2, half_steps=2)) == cli.EXIT_INTERNAL_ERROR
    assert "не статус решателя" in capsys.readouterr().err
    assert names(tmp_path / "out") == []


# --- earlier results survive everything but a valid replacement --------------------------------------------------------------

def snapshot(out):
    return {path.name: path.read_bytes() for path in out.iterdir()}


@pytest.fixture
def earlier(tmp_path):
    """A folder with the results of a completed N = 3 run and a file of the user."""
    out = tmp_path / "out"
    assert cli.main(command(out, 3, half_steps=2)) == cli.EXIT_CONVERGED
    (out / "other.txt").write_text("kept", encoding="utf-8")
    return out


@pytest.mark.parametrize("N, changes, half_steps, code, fragment", [
    (1, {"nu": 1.5}, None, cli.EXIT_SOLVER_ERROR, "ERROR_INPUT"), (3, {"nu": 1.5}, 2, cli.EXIT_SOLVER_ERROR, "ERROR_INPUT"),
    (6, {"nu": 1.5}, 2, cli.EXIT_SOLVER_ERROR, "ERROR_INPUT"), (5, {"b": 1.2}, 2, cli.EXIT_USAGE, cli_ssss.NOT_EXECUTED),
    (6, {}, S_STUDY + 1, cli.EXIT_USAGE, cli_ssss.NOT_EXECUTED), (3, {"nx": 4}, 2, cli_ssss.EXIT_GUARD_STOP, cr.STOPPED_RANK_LOSS_PROVEN),
])
def test_a_refused_input_changes_nothing(earlier, capsys, N, changes, half_steps, code, fragment):
    before = snapshot(earlier)
    capsys.readouterr()
    assert cli.main(command(earlier, N, "--overwrite", half_steps=half_steps, **changes)) == code
    assert snapshot(earlier) == before
    err = capsys.readouterr().err
    assert fragment in err and "Причина:" in err and "не изменены" in err and "field_research.csv" in err


@pytest.mark.parametrize("extra, N, half_steps", [(("--bc", "CCCC"), 3, 2), (("--tol-E", "1e-6"), 3, 2), (("--research-only",), 3, 2)])
def test_a_refused_command_line_changes_nothing(earlier, extra, N, half_steps):
    before = snapshot(earlier)
    assert cli.main(command(earlier, N, "--overwrite", *extra, half_steps=half_steps)) == cli.EXIT_USAGE
    assert snapshot(earlier) == before


@pytest.mark.parametrize("N, attribute", [(3, "run_coupled"), (6, "run_research")])
def test_an_error_of_the_solver_changes_nothing(earlier, monkeypatch, capsys, N, attribute):
    owner = cr if attribute == "run_coupled" else cli_ssss
    real = getattr(owner, attribute)
    monkeypatch.setattr(owner, attribute, lambda *a, **k: dataclasses.replace(real(*a, **k), status="ERROR_DEGENERATE", reason="broken"))
    before = snapshot(earlier)
    capsys.readouterr()
    assert cli.main(command(earlier, N, "--overwrite", half_steps=3)) == cli.EXIT_SOLVER_ERROR
    assert snapshot(earlier) == before
    err = capsys.readouterr().err
    assert "ERROR_DEGENERATE" in err and "broken" in err and "новую папку" in err


def test_the_same_failed_run_without_earlier_results_still_writes_its_record(tmp_path, monkeypatch):
    real = cr.run_coupled
    monkeypatch.setattr(cr, "run_coupled", lambda *a, **k: dataclasses.replace(real(*a, **k), status="ERROR_DEGENERATE", reason="broken"))
    out = tmp_path / "new"
    assert cli.main(command(out, 3, "--overwrite", half_steps=3)) == cli.EXIT_SOLVER_ERROR
    assert names(out) == ["field_diagnostic.csv", "history.csv", "result.json"] and document(out)["execution"] == "ERROR"


@pytest.mark.parametrize("N, half_steps, owner, attribute", [(1, None, cli, "solve"), (3, 2, cr, "run_coupled"),
                                                             (6, 2, cli_ssss, "run_research")])
def test_an_interrupt_of_the_calculation_changes_nothing(earlier, monkeypatch, capsys, N, half_steps, owner, attribute):
    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(owner, attribute, interrupt)
    before = snapshot(earlier)
    capsys.readouterr()
    assert cli.main(command(earlier, N, "--overwrite", half_steps=half_steps)) == cli_ssss.EXIT_INTERRUPTED
    assert snapshot(earlier) == before and "прерван" in capsys.readouterr().err


def test_an_internal_error_changes_nothing(earlier, monkeypatch, capsys):
    monkeypatch.setattr(cr, "run_coupled", lambda *a, **k: 1 / 0)
    before = snapshot(earlier)
    capsys.readouterr()
    assert cli.main(command(earlier, 3, "--overwrite", half_steps=2)) == cli.EXIT_INTERNAL_ERROR
    assert snapshot(earlier) == before and "не изменены" in capsys.readouterr().err


def failing_at(call, error, always=False):
    """os.replace raising ``error`` at its ``call``-th use, and after it if ``always``; the real one otherwise."""
    real, count = os.replace, [0]

    def replace(*args, **kwargs):
        count[0] += 1
        if count[0] == call or (always and count[0] > call):
            raise error
        return real(*args, **kwargs)

    return replace


@pytest.mark.parametrize("call", [1, 2, 3])
@pytest.mark.parametrize("error, code", [(PermissionError("locked"), cli.EXIT_OUTPUT_ERROR), (KeyboardInterrupt(), cli_ssss.EXIT_INTERRUPTED)])
def test_a_failure_during_the_replacement_restores_everything(earlier, monkeypatch, capsys, call, error, code):
    """N = 1 over N = 3: field.csv is new and field_research.csv is stale, so all kinds of change are undone."""
    before = snapshot(earlier)
    monkeypatch.setattr(os, "replace", failing_at(call, error))
    capsys.readouterr()
    assert cli.main(command(earlier, 1, "--overwrite")) == code
    assert snapshot(earlier) == before
    assert "не изменены" in capsys.readouterr().err


@pytest.mark.parametrize("error, code", [(PermissionError("locked"), cli.EXIT_OUTPUT_ERROR), (KeyboardInterrupt(), cli_ssss.EXIT_INTERRUPTED)])
def test_a_failure_removing_a_stale_file_restores_everything(earlier, monkeypatch, error, code):
    real, count = os.remove, [0]

    def remove(path):
        count[0] += 1
        if count[0] == 2:  # the first one takes away the earlier result.json
            raise error
        real(path)

    before = snapshot(earlier)
    monkeypatch.setattr(os, "remove", remove)
    assert cli.main(command(earlier, 1, "--overwrite")) == code
    assert snapshot(earlier) == before


@pytest.mark.parametrize("error, code", [(PermissionError("locked"), cli.EXIT_OUTPUT_ERROR), (KeyboardInterrupt(), cli_ssss.EXIT_INTERRUPTED)])
def test_a_failure_taking_away_the_earlier_result_changes_nothing(earlier, monkeypatch, error, code):
    def remove(path):
        raise error

    before = snapshot(earlier)
    monkeypatch.setattr(os, "remove", remove)
    assert cli.main(command(earlier, 3, "--overwrite", half_steps=3)) == code
    assert snapshot(earlier) == before


def test_a_failure_writing_the_new_files_changes_nothing(earlier, monkeypatch):
    real, count = cli_ssss._write_file, [0]

    def write(path, text):
        count[0] += 1
        if count[0] == 2:
            raise OSError("disk full")
        real(path, text)

    before = snapshot(earlier)
    monkeypatch.setattr(cli_ssss, "_write_file", write)
    assert cli.main(command(earlier, 1, "--overwrite")) == cli.EXIT_OUTPUT_ERROR
    assert snapshot(earlier) == before


def test_a_failed_restoration_is_reported_and_keeps_the_originals(earlier, monkeypatch, capsys):
    before = snapshot(earlier)
    monkeypatch.setattr(os, "replace", failing_at(3, PermissionError("locked"), always=True))  # the install and all putting back fail
    capsys.readouterr()
    assert cli.main(command(earlier, 1, "--overwrite")) == cli.EXIT_OUTPUT_ERROR
    err = capsys.readouterr().err
    assert "ВНИМАНИЕ" in err and "прежние файлы результатов" in err
    monkeypatch.undo()
    assert not (earlier / "result.json").exists()  # never beside data of another run
    stage = next(earlier.glob(cli_ssss.STAGE_PREFIX + "*"))
    assert {name: (stage / "old" / name).read_bytes() for name in ("field_research.csv", "history.csv", "result.json")} == \
           {name: before[name] for name in ("field_research.csv", "history.csv", "result.json")}
    assert cli.main(command(earlier, 3, "--overwrite", half_steps=2)) == cli.EXIT_USAGE  # refused until the save is dealt with


def test_result_json_stays_away_while_a_data_file_cannot_be_put_back(earlier, monkeypatch, capsys):
    """Only history.csv fails: the earlier result.json could come back, but it must not vouch for the new history.csv."""
    before, real = snapshot(earlier), os.replace

    def replace(source, target):
        name, folder = os.path.basename(target), os.path.basename(os.path.dirname(source))
        if (name == "result.json" and folder == "new") or (name == "history.csv" and folder != "new"):
            raise PermissionError("locked")
        real(source, target)

    monkeypatch.setattr(os, "replace", replace)
    capsys.readouterr()
    assert cli.main(command(earlier, 3, "--overwrite", half_steps=3)) == cli.EXIT_OUTPUT_ERROR
    assert "ВНИМАНИЕ" in capsys.readouterr().err
    monkeypatch.undo()
    assert not (earlier / "result.json").exists()
    assert (earlier / "field_research.csv").read_bytes() == before["field_research.csv"]
    assert (earlier / "history.csv").read_bytes() != before["history.csv"]
    assert (next(earlier.glob(cli_ssss.STAGE_PREFIX + "*")) / cli_ssss.REPLACING_MARK).exists()
    assert cli.main(command(earlier, 3, "--overwrite", half_steps=2)) == cli.EXIT_USAGE


def test_the_order_of_the_replacement(tmp_path, monkeypatch):
    """Stale files too: N = 1 over N = 3. The mark is on disk before the first change, result.json goes first and comes last."""
    out = tmp_path / "out"
    assert cli.main(command(out, 3, half_steps=2)) == cli.EXIT_CONVERGED
    log, real_remove, real_replace = [], os.remove, os.replace

    def marked():
        return [(stage / cli_ssss.REPLACING_MARK).exists() for stage in out.glob(cli_ssss.STAGE_PREFIX + "*")] == [True]

    def remove(path):
        log.append(("remove", os.path.basename(path), marked()))
        real_remove(path)

    def replace(source, target):
        log.append(("replace", os.path.basename(target), marked()))
        real_replace(source, target)

    monkeypatch.setattr(os, "remove", remove)
    monkeypatch.setattr(os, "replace", replace)
    assert cli.main(command(out, 1, "--overwrite")) == cli.EXIT_CONVERGED
    assert log[0] == ("remove", "result.json", True) and log[-1] == ("replace", "result.json", True)
    assert sorted(log[1:-1]) == [("remove", "field_research.csv", True), ("replace", "field.csv", True), ("replace", "history.csv", True)]


def test_a_failed_install_of_result_json_keeps_one_record_of_it_and_restores_it(earlier, monkeypatch):
    """A second record of result.json would undo it twice: the second time, with nothing left in old, it would delete the restored file."""
    before = snapshot(earlier)
    records, real = [], cli_ssss._restore

    def restore(out, old, changed):
        records.extend(changed)
        return real(out, old, changed)

    monkeypatch.setattr(cli_ssss, "_restore", restore)
    monkeypatch.setattr(os, "replace", failing_at(3, PermissionError("locked")))
    assert cli.main(command(earlier, 3, "--overwrite", half_steps=3)) == cli.EXIT_OUTPUT_ERROR
    assert sorted(records) == ["field_research.csv", "history.csv", "result.json"]
    assert snapshot(earlier) == before


def test_a_failed_change_is_not_recorded_but_an_interrupted_one_is():
    changed = []

    def fail():
        raise PermissionError("locked")

    def interrupt():
        raise KeyboardInterrupt

    with pytest.raises(PermissionError):
        cli_ssss._change(changed, "a", fail)
    assert changed == []
    with pytest.raises(KeyboardInterrupt):
        cli_ssss._change(changed, "a", interrupt)
    cli_ssss._change(changed, "a", lambda: None)
    with pytest.raises(PermissionError):
        cli_ssss._change(changed, "a", fail)
    assert changed == ["a"]


def test_originals_without_hard_links_are_copied_and_restored(earlier, monkeypatch):
    def no_link(*args):
        raise OSError("no hard links here")

    before = snapshot(earlier)
    monkeypatch.setattr(os, "link", no_link)
    with monkeypatch.context() as patched:
        patched.setattr(os, "replace", failing_at(3, PermissionError("locked")))
        assert cli.main(command(earlier, 3, "--overwrite", half_steps=3)) == cli.EXIT_OUTPUT_ERROR
    assert snapshot(earlier) == before
    assert cli.main(command(earlier, 3, "--overwrite", half_steps=3)) == cli.EXIT_CONVERGED
    assert document(earlier)["half_steps_completed"] == 3


# --- a process killed for real at a known point of the save ----------------------------------------------------------------------

# argv: the kind of point, its number, before or after, then the command. The process stops there and the test kills it (TerminateProcess
# or SIGKILL), so no handler, finally or rollback of the program runs. "change" counts os.remove and os.replace, "write" the synced files.
# "rollback": the change of that number fails, as a locked file does, and the process stops at the first putting back through os.replace,
# before or after it.
STOPPED_AT = """
import os, sys, time
from kkvim import cli, cli_ssss
kind, number, moment = sys.argv[1], int(sys.argv[2]), sys.argv[3]
def stop():
    print("STOPPED", flush=True)
    time.sleep(3600)
count = {"change": 0, "write": 0}
failed = []
def hooked(real, group):
    def call(*args, **kwargs):
        if failed:
            if moment == "after":
                real(*args, **kwargs)
            stop()
        count[group] += 1
        here = kind == group and count[group] == number
        if kind == "rollback" and group == "change" and count[group] == number:
            failed.append(True)
            raise PermissionError("locked")
        if here and moment == "before":
            stop()
        done = real(*args, **kwargs)
        if here and moment == "after":
            stop()
        return done
    return call
os.remove, os.replace = hooked(os.remove, "change"), hooked(os.replace, "change")
cli_ssss._write_file = hooked(cli_ssss._write_file, "write")
sys.exit(cli.main(sys.argv[4:]))
"""

OLD, ABSENT, NEW = "complete earlier run", "no result.json", "complete new run"
A1 = (5, ("field_research.csv", "history.csv", "result.json"))  # N = 5 over N = 4: the finding of the review
STALE = (1, ("field.csv", "history.csv", "result.json"))  # N = 1 over N = 4: a new name and a stale name


def stop_and_kill(out, N, point):
    kind, number, moment = point
    half_steps = None if N == 1 else 3
    process = subprocess.Popen([sys.executable, "-c", STOPPED_AT, kind, str(number), moment, *command(out, N, "--overwrite", half_steps=half_steps)],
                               stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    line = process.stdout.readline()
    process.kill()
    process.communicate()
    assert line.strip() == "STOPPED", line


def snapshot_all(out):
    return {str(path.relative_to(out)): path.read_bytes() for path in out.rglob("*") if path.is_file()}


def assert_never_mixed(out, before, new_names, state):
    """What a reader of the folder sees: the earlier set, the new set or no result.json, as this point requires."""
    earlier_names = [name for name in before if name in cli.OUTPUTS]
    present = [name for name in cli.OUTPUTS if (out / name).exists()]
    if state == OLD:
        assert sorted(present) == sorted(earlier_names)
        assert all((out / name).read_bytes() == before[name] for name in present)
    elif state == ABSENT:
        assert "result.json" not in present
    else:
        assert sorted(present) == sorted(new_names)
        assert all((out / name).read_bytes() != before.get(name) for name in present)
        assert document(out)["half_steps_completed"] > 0
    assert (out / "other.txt").read_bytes() == before["other.txt"]


def assert_refused_and_recoverable(out, before, capsys):
    """Detected on restart, with and without --overwrite, nothing computed or changed; the instruction printed gives back the earlier set."""
    stage = next(out.glob(cli_ssss.STAGE_PREFIX + "*"))
    assert (stage / cli_ssss.REPLACING_MARK).exists()
    for name in before:
        if name in cli.OUTPUTS:
            assert (stage / "old" / name).read_bytes() == before[name]
    left = snapshot_all(out)
    for extra in (("--overwrite",), ()):
        capsys.readouterr()
        assert cli.main(command(out, 3, *extra, half_steps=2)) == cli.EXIT_USAGE
        err = capsys.readouterr().err
        assert stage.name in err and "незавершённая замена" in err and "перенесите" in err
        assert snapshot_all(out) == left
    for name in cli.OUTPUTS:
        (out / name).unlink(missing_ok=True)
    for path in (stage / "old").iterdir():
        os.replace(path, out / path.name)
    shutil.rmtree(stage)
    assert snapshot(out) == before


@pytest.fixture
def earlier_n4(tmp_path):
    out = tmp_path / "out"
    assert cli.main(command(out, 4, half_steps=2)) == cli.EXIT_CONVERGED
    (out / "other.txt").write_text("kept", encoding="utf-8")
    return out


# Changes of N = 5 over N = 4: 1 remove result.json, 2-3 data, 4 install. Of N = 1 over N = 4: 1 remove result.json, 2-3 data, 4 stale, 5 install.
@pytest.mark.parametrize("new, point, state", [
    (A1, ("change", 1, "before"), OLD), (A1, ("change", 1, "after"), ABSENT),
    (A1, ("change", 2, "before"), ABSENT), (A1, ("change", 2, "after"), ABSENT),  # the state of the review: one data file of N = 5
    (A1, ("change", 3, "before"), ABSENT), (A1, ("change", 3, "after"), ABSENT),  # all data of N = 5, result.json not yet
    (A1, ("change", 4, "before"), ABSENT), (A1, ("change", 4, "after"), NEW),
    (STALE, ("change", 1, "after"), ABSENT), (STALE, ("change", 2, "after"), ABSENT), (STALE, ("change", 3, "after"), ABSENT),
    (STALE, ("change", 4, "after"), ABSENT), (STALE, ("change", 5, "before"), ABSENT), (STALE, ("change", 5, "after"), NEW),
    (A1, ("write", 4, "after"), OLD),  # the mark is on disk, nothing has changed
    (A1, ("rollback", 3, "before"), ABSENT), (A1, ("rollback", 3, "after"), ABSENT),  # killed while undoing a failed replacement
    (A1, ("rollback", 4, "after"), ABSENT), (STALE, ("rollback", 4, "before"), ABSENT), (STALE, ("rollback", 5, "after"), ABSENT),
])
def test_a_killed_replacement_never_shows_a_mixed_run_and_is_refused(earlier_n4, capsys, new, point, state):
    N, new_names = new
    before = snapshot(earlier_n4)
    stop_and_kill(earlier_n4, N, point)
    assert_never_mixed(earlier_n4, before, new_names, state)
    assert_refused_and_recoverable(earlier_n4, before, capsys)
    assert cli.main(command(earlier_n4, 3, "--overwrite", half_steps=2)) == cli.EXIT_CONVERGED  # the folder works again


@pytest.mark.parametrize("point", [("write", 1, "before"), ("write", 2, "after"), ("write", 4, "before")])
def test_a_process_killed_before_the_mark_does_not_block_the_next_run(earlier_n4, point):
    """The new files are not complete or the mark is not yet written: no result file has changed and the save is simply dropped."""
    before = snapshot(earlier_n4)
    stop_and_kill(earlier_n4, 5, point)
    stage = next(earlier_n4.glob(cli_ssss.STAGE_PREFIX + "*"))
    assert not (stage / cli_ssss.REPLACING_MARK).exists()
    assert_never_mixed(earlier_n4, before, A1[1], OLD)
    assert cli.main(command(earlier_n4, 5, "--overwrite", half_steps=3)) == cli.EXIT_CONVERGED
    assert not list(earlier_n4.glob(cli_ssss.STAGE_PREFIX + "*")) and document(earlier_n4)["half_steps_completed"] == 3


def test_a_save_killed_during_a_longer_run_stops_that_run_from_saving(earlier_n4, monkeypatch, capsys):
    real = cli_ssss.run_research

    def run(*args, **kwargs):
        stop_and_kill(earlier_n4, 5, ("change", 2, "after"))
        return real(*args, **kwargs)

    monkeypatch.setattr(cli_ssss, "run_research", run)
    capsys.readouterr()
    assert cli.main(command(earlier_n4, 6, "--overwrite", half_steps=2)) == cli.EXIT_USAGE
    assert "незавершённая замена" in capsys.readouterr().err and not (earlier_n4 / "result.json").exists()


@pytest.mark.skipif(sys.platform != "win32", reason="an open file blocks its replacement only on Windows")
def test_a_result_file_held_open_by_another_program(earlier, capsys):
    before = snapshot(earlier)
    capsys.readouterr()
    with open(earlier / "history.csv", "rb"):
        assert cli.main(command(earlier, 1, "--overwrite")) == cli.EXIT_OUTPUT_ERROR
        assert "не изменены" in capsys.readouterr().err
    assert {name: (earlier / name).read_bytes() for name in before} == before
    assert cli.main(command(earlier, 1, "--overwrite")) == cli.EXIT_CONVERGED  # closed: the run works, nothing is left of the failed save
    assert names(earlier) == ["field.csv", "history.csv", "other.txt", "result.json"]


@pytest.mark.skipif(sys.platform != "win32", reason="an open file blocks its removal only on Windows")
def test_the_earlier_result_json_held_open_by_another_program(earlier, capsys):
    before = snapshot(earlier)
    capsys.readouterr()
    with open(earlier / "result.json", "rb"):
        assert cli.main(command(earlier, 3, "--overwrite", half_steps=3)) == cli.EXIT_OUTPUT_ERROR
        assert "не изменены" in capsys.readouterr().err
    assert {name: (earlier / name).read_bytes() for name in before} == before
    assert cli.main(command(earlier, 3, "--overwrite", half_steps=3)) == cli.EXIT_CONVERGED


def test_a_process_killed_during_the_calculation_changes_nothing(tmp_path):
    out = tmp_path / "out"
    assert cli.main(command(out, 6, half_steps=2)) == cli.EXIT_CONVERGED
    before = snapshot(out)
    process = subprocess.Popen([sys.executable, "-m", "kkvim", *command(out, 6, "--overwrite", half_steps=S_STUDY, nx=1280, ny=1280)],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(3.0)
    assert process.poll() is None
    process.kill()
    process.wait()
    assert snapshot(out) == before


def test_files_appearing_during_a_run_without_overwrite_are_kept(tmp_path, monkeypatch, capsys):
    out = tmp_path / "out"
    real = cr.run_coupled

    def run(*args, **kwargs):
        (out / "result.json").write_text("of another run", encoding="utf-8")
        return real(*args, **kwargs)

    monkeypatch.setattr(cr, "run_coupled", run)
    assert cli.main(command(out, 3, half_steps=2)) == cli.EXIT_USAGE
    assert snapshot(out) == {"result.json": b"of another run"} and "не сохранены" in capsys.readouterr().err


def test_a_successful_replacement_leaves_no_working_files_and_no_shared_contents(earlier):
    outdated = (earlier / "history.csv").read_bytes()
    assert cli.main(command(earlier, 3, "--overwrite", half_steps=4)) == cli.EXIT_CONVERGED
    assert names(earlier) == ["field_research.csv", "history.csv", "other.txt", "result.json"]
    assert (earlier / "history.csv").read_bytes() != outdated and (earlier / "history.csv").stat().st_nlink == 1


def test_the_run_command_keeps_its_own_saving(tmp_path):
    """``run`` is backward compatible: its refused overwrite still replaces the set, as its tests and documents say."""
    out = tmp_path / "out"
    assert cli.main(["run", *plate(), "--out", str(out)]) == cli.EXIT_CONVERGED
    assert cli.main(["run", *plate(nu=1.5), "--out", str(out), "--overwrite"]) == cli.EXIT_SOLVER_ERROR
    assert names(out) == ["history.csv", "result.json"] and document(out)["status"] == ERROR_INPUT


# --- the version of the program ----------------------------------------------------------------------------------------------

def fake_git(monkeypatch, **answers):
    """subprocess.run of cli_ssss for git: the bytes or the exception of each subcommand; decoding is not asked for."""

    def run(command_line, **kwargs):
        assert "text" not in kwargs and "encoding" not in kwargs
        answer = answers[command_line[1].replace("-", "_")]
        if isinstance(answer, Exception):
            raise answer
        return subprocess.CompletedProcess(command_line, 0, stdout=answer, stderr=b"")

    monkeypatch.setattr(cli_ssss.subprocess, "run", run)


SHA = b"18ace0a5666f946077fed1871f76980ea8caf091\n"
FAILED = subprocess.CalledProcessError(128, "git")


@pytest.mark.parametrize("N", [1, 3, 6])
def test_invalid_git_output_does_not_cost_the_calculation(tmp_path, monkeypatch, capsys, N):
    reference = tmp_path / "reference"
    half_steps = None if N == 1 else 3
    assert cli.main(command(reference, N, half_steps=half_steps)) == cli.EXIT_CONVERGED
    fake_git(monkeypatch, rev_parse=b"\xff\xfe\x80\n", status=b"")
    out = tmp_path / "out"
    capsys.readouterr()
    assert cli.main(command(out, N, half_steps=half_steps)) == cli.EXIT_CONVERGED
    provenance = document(out)["provenance"]
    assert provenance["git_commit"] is None and provenance["git_worktree_modified"] is None
    assert "версия Git недоступна" in provenance["git_diagnostic"] and "UnicodeDecodeError" in provenance["git_diagnostic"]
    assert "версия Git недоступна" in capsys.readouterr().out
    for name in names(reference):
        if name != "result.json":
            assert (out / name).read_bytes() == (reference / name).read_bytes()


def test_the_commit_and_the_state_of_the_checkout_are_recorded(monkeypatch):
    fake_git(monkeypatch, rev_parse=SHA, status=b"")
    provenance = cli_ssss.software_provenance()
    assert provenance["git_commit"] == SHA.decode().strip() and provenance["git_worktree_modified"] is False
    assert provenance["git_diagnostic"] is None
    fake_git(monkeypatch, rev_parse=SHA, status=b" M src/kkvim/cli.py\n")
    assert cli_ssss.software_provenance()["git_worktree_modified"] is True


def test_file_names_in_any_encoding_are_only_a_modified_worktree(monkeypatch):
    fake_git(monkeypatch, rev_parse=SHA, status=b"?? caf\xe9-\xff.txt\n")
    provenance = cli_ssss.software_provenance()
    assert provenance["git_commit"] == SHA.decode().strip() and provenance["git_worktree_modified"] is True
    assert provenance["git_diagnostic"] is None


@pytest.mark.parametrize("answers, commit, fragment", [
    (dict(rev_parse=b"not a sha\n", status=b""), None, "не является SHA"),
    (dict(rev_parse=FileNotFoundError("no git"), status=b""), None, "FileNotFoundError"),
    (dict(rev_parse=FAILED, status=b""), None, "CalledProcessError"),
    (dict(rev_parse=subprocess.TimeoutExpired("git", 30), status=b""), None, "TimeoutExpired"),
    (dict(rev_parse=SHA, status=FAILED), SHA.decode().strip(), "состояние рабочей копии неизвестно"),
    (dict(rev_parse=OSError("nam\udcffe"), status=b""), None, "OSError"),
])
def test_an_unavailable_version_is_stated(monkeypatch, answers, commit, fragment):
    fake_git(monkeypatch, **answers)
    provenance = cli_ssss.software_provenance()
    assert provenance["git_commit"] == commit and provenance["git_worktree_modified"] is None
    assert fragment in provenance["git_diagnostic"]
    json.dumps(provenance, ensure_ascii=False).encode("utf-8")  # always writable


def test_a_copy_without_a_source_checkout_says_so(monkeypatch, tmp_path):
    file = tmp_path / "site-packages" / "kkvim" / "cli_ssss.py"
    file.parent.mkdir(parents=True)
    file.touch()
    monkeypatch.setattr(cli_ssss, "__file__", str(file))
    monkeypatch.setattr(cli_ssss.subprocess, "run", lambda *a, **k: pytest.fail("git was called"))
    provenance = cli_ssss.software_provenance()
    assert provenance["git_commit"] is None and "не из рабочей копии" in provenance["git_diagnostic"]


# --- units of the energies -----------------------------------------------------------------------------------------------------

ENERGY_COLUMNS = {"P", "delta_P", "Q", "gamma_s", "tau", "P_after_qr"}


def header_of(path):
    return next(line for line in path.read_text(encoding="utf-8").splitlines() if not line.startswith("#"))


@pytest.mark.parametrize("N, changes, half_steps, kind, unit, word", [
    (2, {}, 3, "physical", "N·m", "ФИЗИЧЕСКИЕ"), (3, {}, 3, "physical", "N·m", "ФИЗИЧЕСКИЕ"),
    (4, {}, 3, "normalized", "q^2 a^6 / D", "НОРМИРОВАННЫЕ"), (6, {}, 3, "normalized", "q^2 a^6 / D", "НОРМИРОВАННЫЕ"),
    (6, {"b": 0.2}, 3, "normalized", "q^2 a^6 / D", "НОРМИРОВАННЫЕ"), (5, {"b": 1.2}, 3, "normalized", "q^2 a^6 / D", "НОРМИРОВАННЫЕ"),
    (3, {"nx": 4}, 2, "physical", "N·m", "ФИЗИЧЕСКИЕ"),
])
def test_the_unit_of_the_energies_is_stated_in_every_file(tmp_path, N, changes, half_steps, kind, unit, word):
    out = tmp_path / "out"
    cli.main(command(out, N, half_steps=half_steps, **changes))
    history = document(out)["history"]
    assert history["energy_kind"] == kind and history["energy_unit"] == unit and word in history["energy_note"]
    columns = {column["name"]: column["unit"] for column in history["columns"]}
    assert {name for name, value in columns.items() if value == unit} == ENERGY_COLUMNS
    assert "energy" not in columns.values()
    comments = [line for line in (out / "history.csv").read_text(encoding="utf-8").splitlines() if line.startswith("#")]
    assert any(word in line and unit in line for line in comments) and history["comments"] == [line[2:] for line in comments]


def test_the_csv_columns_are_the_same_for_physical_and_normalized_energies(tmp_path):
    expected = ",".join(name for name, _, _ in cli_ssss.COUPLED_COLUMNS)
    for N in (2, 3, 4, 5, 6):
        out = tmp_path / str(N)
        assert cli.main(command(out, N, half_steps=2)) == cli.EXIT_CONVERGED
        assert header_of(out / "history.csv") == expected
        rows = list(csv.DictReader(line for line in (out / "history.csv").read_text(encoding="utf-8").splitlines() if line[0] != "#"))
        assert float(rows[0]["P"]) < 0.0


def test_the_factor_of_the_normalized_energies_is_q2_a6_over_d(tmp_path, capsys):
    out = tmp_path / "out"
    capsys.readouterr()
    assert cli.main(command(out, 4, half_steps=2, a=2.0, b=2.0, q=D)) == cli.EXIT_CONVERGED  # exact: q a^4 / D = 16
    assert document(out)["energy_scale_q2_a6_over_D"] == {"value": 64.0 * D, "unit": "N·m",
                                                          "use": "energy [N·m] = energy of history.csv * value"}
    assert "НОРМИРОВАННЫЕ" in capsys.readouterr().out
    physical = tmp_path / "physical"
    assert cli.main(command(physical, 3, half_steps=2)) == cli.EXIT_CONVERGED
    assert "energy_scale_q2_a6_over_D" not in document(physical)
    assert "ФИЗИЧЕСКИЕ" in capsys.readouterr().out


# --- the mandatory energy check of s = 2 ----------------------------------------------------------------------------------------

def unresolved_energy(monkeypatch):
    real = cr.pair_energy
    monkeypatch.setattr(cr, "pair_energy", lambda *a, **k: dataclasses.replace(real(*a, **k), status=cr.UNRESOLVED))


def test_an_unresolved_mandatory_check_is_shown_and_changes_no_status(tmp_path, monkeypatch, capsys):
    """The energy test is forced to UNRESOLVED; the flag and the reason then come from run_coupled itself (D-048.1)."""
    plain, flagged = tmp_path / "plain", tmp_path / "flagged"
    grid = dict(nx=160, ny=160)
    assert cli.main(command(plain, 3, half_steps=3, **grid)) == cli.EXIT_CONVERGED
    assert "обязательная проверка" not in capsys.readouterr().out
    unresolved_energy(monkeypatch)
    assert cli.main(command(flagged, 3, half_steps=3, **grid)) == cli.EXIT_CONVERGED
    out = capsys.readouterr().out
    result, reference = document(flagged), document(plain)
    energy = result["energy_s2"]
    assert energy["mandatory_d048"] is True and energy["status"] == cr.UNRESOLVED and energy["mandatory_unresolved"] is True
    assert "обязательная проверка энергии при s = 2 не разрешена" in out and energy["reason"] in out
    assert "Статус расчёта от этого не меняется" in out
    for key in ("status", "execution", "converged", "half_steps_completed", "scientific_applicability", "status_note"):
        assert result[key] == reference[key]


def test_an_unresolved_check_on_a_grid_where_it_is_not_mandatory_is_not_announced(tmp_path, monkeypatch, capsys):
    unresolved_energy(monkeypatch)
    out = tmp_path / "out"
    assert cli.main(command(out, 3, half_steps=3)) == cli.EXIT_CONVERGED
    energy = document(out)["energy_s2"]
    assert energy["mandatory_d048"] is False and energy["mandatory_unresolved"] is False and energy["status"] == cr.UNRESOLVED
    assert "обязательная проверка" not in capsys.readouterr().out


def test_the_flag_of_d048_is_not_applied_to_n4_n6(tmp_path, capsys):
    out = tmp_path / "out"
    assert cli.main(command(out, 4, half_steps=3, nx=160, ny=160)) == cli.EXIT_CONVERGED
    energy = document(out)["energy_s2"]
    assert energy["mandatory_d048"] is None and energy["mandatory_unresolved"] is None and energy["checked"] is True
    assert "обязательная проверка" not in capsys.readouterr().out


def test_a_run_ended_before_s2_has_no_check(tmp_path, capsys):
    out = tmp_path / "out"
    assert cli.main(command(out, 3, half_steps=1, nx=160, ny=160)) == cli.EXIT_CONVERGED
    energy = document(out)["energy_s2"]
    assert energy["checked"] is False and energy["status"] is None and energy["mandatory_unresolved"] is None
    assert "обязательная проверка" not in capsys.readouterr().out


# --- metadata of result.json --------------------------------------------------------------------------------------------------

COMMON_KEYS = {"schema", "N", "boundary_conditions", "theory", "method", "solver_path", "provenance", "execution", "status",
               "status_note", "reason", "half_steps_completed", "program", "input", "D", "scientific_applicability", "field", "history"}


@pytest.mark.parametrize("N, mode, route, decisions, half_steps", [
    (1, "R3_ACCEPTED_SOLVER", "kkvim.rectangle_run.run_rectangle_physical", ["D-043"], None),
    (3, "R4_LIMITED_RESEARCH_MODE", "kkvim.coupled_run.run_coupled(research_mode=True, transform='fft')",
     ["D-047", "D-048", "D-049.7", "D-056.8", "D-060"], 3),
    (6, "RESEARCH_ONLY_N4_N6", "kkvim.research_ssss.run_research", ["D-056.7", "D-060"], 3),
])
def test_the_common_metadata_of_each_route(tmp_path, N, mode, route, decisions, half_steps):
    out = tmp_path / "out"
    assert cli.main(command(out, N, half_steps=half_steps)) == cli.EXIT_CONVERGED
    result = document(out)
    assert COMMON_KEYS <= set(result)
    assert result["schema"] == cli_ssss.SCHEMA and result["N"] == N and result["boundary_conditions"] == "SSSS"
    assert result["theory"] == cli_ssss.THEORY and result["method"] == cli_ssss.METHOD
    path = result["solver_path"]
    assert path["mode"] == mode and path["route"] == route and path["decisions"] == decisions and path["research"] is (N != 1)
    assert result["execution"] == "COMPLETED" and (result["reason"] is None) is (N == 1)  # N >= 2: why the run ended
    provenance = result["provenance"]
    assert set(provenance) == {"versions", "git_commit", "git_worktree_modified", "git_diagnostic", "module"}
    assert provenance["module"] == "kkvim.cli_ssss" and {"python", "kkvim", "numpy", "scipy"} <= set(provenance["versions"])
    assert result["program"]["argv"][:3] == ["run-ssss", "--N", str(N)]
    assert result["program"]["command"] == "python -m kkvim " + " ".join(result["program"]["argv"])
    assert result["D"] == {"value": D, "unit": "N·m"}
    for name, unit in cli.INPUT_UNITS:
        assert result["input"][name] == {"value": PLATE[name], "unit": unit}
    assert result["input"]["n_x"] == result["input"]["n_y"] == 40


@pytest.mark.parametrize("N, name, role, half_steps", [(1, "field.csv", "converged", None), (3, "field_research.csv", "research", 3),
                                                       (6, "field_research.csv", "research", 3)])
def test_the_field_record_describes_the_field_file(tmp_path, N, name, role, half_steps):
    out = tmp_path / "out"
    assert cli.main(command(out, N, half_steps=half_steps)) == cli.EXIT_CONVERGED
    result = document(out)
    field = result["field"]
    assert field["file"] == name and field["role"] == role and field["shape"] == [41, 41]
    assert field["half_step"] == result["half_steps_completed"]
    assert field["units"] == {"x": "m", "y": "m", "w": "m", "w_bar": "1"} and field["w_bar_definition"] == cli.W_BAR
    _, w_bar = field_arrays(out / name, (41, 41))
    assert field["w_bar_centre"] == float(w_bar[20, 20])
    assert result["history"]["rows"] == len(table(out / "history.csv")) == result["half_steps_completed"]


@pytest.mark.parametrize("N, q", [(2, 1000.0), (3, -500.0), (4, 1000.0), (6, -500.0)])
def test_the_history_columns_of_w_agree_with_the_written_field(tmp_path, N, q):
    """w_max in metres, w_bar_max dimensionless, at a != 1, for both routes and both signs of q, exactly."""
    out = tmp_path / "out"
    assert cli.main(command(out, N, half_steps=3, a=2.0, b=2.0, q=q)) == cli.EXIT_CONVERGED
    last = table(out / "history.csv")[-1]
    w, w_bar = field_arrays(out / "field_research.csv", (41, 41))
    assert float(last["w_max"]) == float(np.max(np.abs(w)))
    assert float(last["w_bar_max"]) == float(np.max(np.abs(w_bar)))
    assert float(last["w_max"]) != float(last["w_bar_max"])
    columns = {column["name"]: column["unit"] for column in document(out)["history"]["columns"]}
    assert columns["w_max"] == "m" and columns["w_bar_max"] == ""


def test_a_grid_without_a_centre_node_has_no_centre_value(tmp_path):
    out = tmp_path / "out"
    assert cli.main(command(out, 3, half_steps=2, nx=41, ny=41)) == cli.EXIT_CONVERGED
    assert document(out)["field"]["w_bar_centre"] is None


def test_n1_is_labelled_as_the_accepted_solver_and_nothing_more(tmp_path):
    out = tmp_path / "out"
    assert cli.main(command(out, 1)) == cli.EXIT_CONVERGED
    scientific = document(out)["scientific_applicability"]
    assert scientific["accepted_solver"] is True and scientific["accepted_by"] == "D-043"
    assert scientific["research_only"] is False and scientific["mark"] is None
    assert "не научная приёмка" in scientific["domain"] and scientific["limits"] == list(cli.APPLICABILITY)


def test_the_tolerance_of_a_control_ratio_is_the_representation_of_the_input():
    assert control_ratio(1.0, 1.0 + 2e-16) == 1
    assert control_ratio(1.0, 1.0 + 1e-15) is None
    assert domain_refusal(1.0, 1.0 + 2e-16, 40, 40, 4, 2) is None
    assert "ten r" in domain_refusal(1.0, 1.0 + 1e-15, 40, 40, 4, 2)


@pytest.mark.parametrize("a, b, n_x, n_y, inside, mandatory", [
    (1.0, 1.0, 40, 40, True, False), (1.0, 1.0, 160, 160, True, True), (1.0, 1.0, 640, 640, True, True), (1.0, 1.0, 1280, 1280, True, True),
    (1.0, 1.0, 80, 80, False, False), (1.0, 1.0, 320, 320, False, False), (1.0, 1.0, 160, 320, False, False),
    (1.0, 5.0, 320, 1600, True, True), (5.0, 1.0, 1600, 320, True, True), (1.0, 1.0, 320, 1600, False, False), (1.0, 1.2, 160, 160, False, False),
])
def test_the_area_of_d048(a, b, n_x, n_y, inside, mandatory):
    area = cli_ssss.d048_area(a, b, n_x, n_y)
    assert area["inside_area"] is inside and area["mandatory_s2_grid"] is mandatory


def test_the_research_records_of_n2_n3_and_n4_n6(tmp_path):
    three, six = tmp_path / "three", tmp_path / "six"
    assert cli.main(command(three, 3, half_steps=3, a=2.0, b=2.0)) == cli.EXIT_CONVERGED
    assert cli.main(command(six, 6, half_steps=3, a=2.0, b=2.0)) == cli.EXIT_CONVERGED
    n3, n6 = document(three), document(six)
    scale = 1000.0 * 2.0**4 / D
    for result in (n3, n6):
        assert result["w_scale_q_a4_over_D"]["value"] == scale and result["w_scale_q_a4_over_D"]["unit"] == "m"
        assert result["converged"] is False and result["cycles_k"] == 1 and result["input"]["half_steps_requested"] == 3
    # N = 2-3 keep the physical parameters of the run; N = 4-6 say that theirs are of the studied cell
    assert n3["numerical_parameters"]["a"] == 2.0 and n3["numerical_parameters"]["D"] == D and "cell" not in n3["numerical_parameters"]
    assert n3["numerical_parameters"]["K_research"] == cr.K_RESEARCH
    assert n6["numerical_parameters"]["a"] == 1.0 and n6["numerical_parameters"]["q"] == 1.0 and "ячейки" in n6["numerical_parameters"]["cell"]
    assert n3["input"]["fixed_half_steps"] is False and n6["input"]["fixed_half_steps"] is True
    assert n3["scientific_applicability"]["implementation"].startswith("ACCEPTED_LIMITED_ENGINEERING")
    assert n3["scientific_applicability"]["domain"]["inside_area"] is True
    assert n6["scientific_applicability"]["status"] == "RESEARCH_ONLY (D-056.7)" and "d048" not in json.dumps(n6["scientific_applicability"])
    assert n6["scientific_applicability"]["domain"]["control_r"] == "1"


def test_a_field_record_gives_the_node_of_max_w_in_metres(tmp_path):
    out = tmp_path / "out"
    assert cli.main(command(out, 3, half_steps=3, a=2.0, b=2.0)) == cli.EXIT_CONVERGED
    w, _ = field_arrays(out / "field_research.csv", (41, 41))
    w_max = document(out)["field"]["w_max"]
    assert w_max["unit"] == "m" and w_max["value"] == abs(w[w_max["i"], w_max["j"]]) == float(np.max(np.abs(w)))
    assert w_max["x"] == w_max["i"] * 2.0 / 40 and w_max["y"] == w_max["j"] * 2.0 / 40


def test_a_stopped_run_records_its_status_and_a_diagnostic_field(tmp_path):
    out = tmp_path / "out"
    assert cli.main(command(out, 6, half_steps=3, b=0.2)) == cli_ssss.EXIT_GUARD_STOP
    result = document(out)
    assert result["execution"] == "STOPPED" and result["reason"] and result["field"]["file"] == "field_diagnostic.csv"
    assert result["field"]["role"] == "diagnostic" and "не решение" in result["field"]["note"]
    assert result["history"]["energy_kind"] == "normalized"


@pytest.mark.parametrize("N, half_steps", [(1, None), (3, 8), (6, 8)])
def test_python_m_kkvim_runs_the_documented_examples(tmp_path, N, half_steps):
    out = tmp_path / "out"
    done = subprocess.run([sys.executable, "-m", "kkvim", *command(out, N, half_steps=half_steps)], capture_output=True,
                          encoding="utf-8", check=False)
    assert done.returncode == cli.EXIT_CONVERGED, done.stderr
    assert f"SSSS, N = {N}" in done.stdout and document(out)["N"] == N
