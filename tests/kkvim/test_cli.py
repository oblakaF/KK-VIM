"""Command line run of R3 (``python -m kkvim run``; ``kkvim.cli``).

- the command calls run_rectangle_physical and halfstep_report; field.csv and history.csv equal the arrays and
  quantities of the direct API call in the same process bit for bit;
- the control case a = b = 1, n_x = n_y = 24 (QA control case r3_square, PR #24) is CONVERGED with s = 7;
- statuses of the solver are written unchanged; a field of a run that did not converge or ended with an error
  is written only as field_diagnostic.csv, never as field.csv;
- invalid input is ERROR_INPUT of the solver (or a command line error); nothing is overwritten without --overwrite.
"""

import csv
import json
import os
import subprocess
import sys

import numpy as np
import pytest

import kkvim.rectangle_run as rr
from kkvim import cli
from kkvim.coefficients import ERROR_INPUT
from kkvim.controller import CONVERGED, MAX_HALF_STEPS, STAGNATED
from kkvim.linear_solver import ERROR_LINEAR_SOLVE, LinearSolveError

PLATE = dict(a=1.0, b=1.0, t=0.01, E=2.1e11, nu=0.3, q=1000.0)
EXAMPLE = ["--a", "1", "--b", "1", "--t", "0.01", "--E", "2.1e11", "--nu", "0.3", "--q", "1000", "--nx", "24",
           "--ny", "24"]


def command(out, *extra, base=EXAMPLE):
    return ["run", *base, "--out", str(out), *extra]


def with_option(name, value):
    args = list(EXAMPLE)
    args[args.index(name) + 1] = value
    return args


def read_csv(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    comments = [line for line in lines if line.startswith("#")]
    rows = list(csv.DictReader(line for line in lines if not line.startswith("#")))
    return comments, rows


def read_field(path, shape):
    comments, rows = read_csv(path)
    grid = {key: np.empty(shape) for key in ("x [m]", "y [m]", "w [m]", "w_bar [1]")}
    for row in rows:
        i, j = int(row["i"]), int(row["j"])
        for key, values in grid.items():
            values[i, j] = float(row[key])
    assert len(rows) == shape[0] * shape[1]
    return comments, grid


def number(text):
    return None if text == "" else float(text)


# --- the control case and the direct API -------------------------------------------------------------------

def test_example_writes_the_result_of_the_direct_api_call_bitwise(tmp_path, capsys):
    out = tmp_path / "demo"
    assert cli.main(command(out)) == cli.EXIT_CONVERGED
    direct = rr.run_rectangle_physical(**PLATE, n_x=24, n_y=24)
    report = rr.halfstep_report(direct)
    assert sorted(p.name for p in out.iterdir()) == ["field.csv", "history.csv", "result.json"]

    doc = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert doc["status"] == direct.status == CONVERGED and doc["converged"] is True and doc["error"] is None
    assert doc["half_steps"] == direct.run.s == 7 and doc["cycles_k"] == 3    # QA control case r3_square: s = 7
    assert doc["D"] == {"value": direct.D, "unit": "N·m"} and direct.D == 19230.769230769234
    assert doc["input"]["E"] == {"value": 2.1e11, "unit": "Pa"} and (doc["input"]["n_x"], doc["input"]["n_y"]) == (24, 24)
    params = direct.run.parameters
    assert doc["stopping_parameters"] == {"tol_inf": params.tol_inf, "tol_E": params.tol_E, "s_min": params.s_min,
                                          "s_max": params.s_max, "eps_floor": params.eps_floor}
    assert doc["program"]["calls"] == list(cli.CALLS) and doc["program"]["versions"] == report.versions
    assert doc["field"]["file"] == "field.csv" and doc["field"]["role"] == "converged"
    assert doc["field"]["w_bar_centre"] == direct.dimensionless_field()[12, 12]
    assert any("D-043" in text for text in doc["applicability"]["limits"])
    assert doc["applicability"]["max_n_within_qualified_640"] is True

    comments, field = read_field(out / "field.csv", (25, 25))
    assert comments == []
    last = direct.run.history.records[-1]
    np.testing.assert_array_equal(field["w [m]"], direct.run.last_field())
    np.testing.assert_array_equal(field["w_bar [1]"], direct.dimensionless_field())
    np.testing.assert_array_equal(field["x [m]"][:, 0], last.grid_x.nodes)
    np.testing.assert_array_equal(field["y [m]"][0, :], last.grid_y.nodes)

    _, history = read_csv(out / "history.csv")
    assert [int(r["s"]) for r in history] == [h.s for h in report.halfsteps] == list(range(1, 8))
    for row, h in zip(history, report.halfsteps, strict=True):
        assert (int(row["k"]), row["direction"], int(row["solve_n"])) == (h.k, h.direction, h.solve_grid[1])
        for key, value in (("c", h.c), ("beta", h.beta), ("eta", h.eta), ("P", h.P), ("norm_a", h.norm_a),
                           ("residual_norm", h.residual_norm)):
            assert number(row[key]) == value, key
        assert row["energy_status"] == h.energy.status and number(row["Q"]) == h.energy.Q
        assert number(row["w_bar_centre"]) == dict(h.control_values)[rr.CENTRE]
        if h.changes is None:
            assert row["eps_inf"] == row["eps_E"] == "" and h.s == 1
        else:
            assert number(row["eps_inf"]) == h.changes.eps_inf and number(row["eps_E"]) == h.changes.eps_E
            assert row["satisfied"] == ("true" if h.changes.satisfied else "false")
    printed = capsys.readouterr().out
    assert "CONVERGED" in printed and "s = 7" in printed and str(out.resolve()) in printed


def test_python_m_kkvim_runs_the_readme_example(tmp_path):
    env = {k: v for k, v in os.environ.items() if k != "PYTHONIOENCODING"}
    done = subprocess.run([sys.executable, "-m", "kkvim", *command("scratch/demo")], cwd=tmp_path, env=env,
                          capture_output=True, timeout=300)
    text = done.stdout.decode("utf-8")                     # messages are UTF-8 even through a pipe
    assert done.returncode == 0, done.stderr.decode("utf-8", "replace")
    assert "Статус решателя R3: CONVERGED" in text and "result.json" in text
    assert sorted(p.name for p in (tmp_path / "scratch" / "demo").iterdir()) == ["field.csv", "history.csv",
                                                                                 "result.json"]


# --- input errors --------------------------------------------------------------------------------------------

@pytest.mark.parametrize(("name", "value"), [("--nu", "1"), ("--b", "6"), ("--t", "-0.01"), ("--nx", "3"),
                                             ("--q", "0"), ("--E", "nan")])
def test_invalid_input_is_error_input_of_the_solver(tmp_path, name, value):
    out = tmp_path / "bad"
    assert cli.main(command(out, base=with_option(name, value))) == cli.EXIT_SOLVER_ERROR
    assert sorted(p.name for p in out.iterdir()) == ["history.csv", "result.json"]   # no field at all
    doc = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert doc["status"] == ERROR_INPUT and doc["converged"] is False and doc["error"]
    assert doc["D"] is None and doc["field"] is None and doc["half_steps"] == 0
    assert doc["stopping_parameters"] is None
    _, rows = read_csv(out / "history.csv")
    assert rows == []


def test_a_tolerance_below_the_rounding_floor_is_error_input(tmp_path):
    out = tmp_path / "tol"
    assert cli.main(command(out, "--tol-E", "1e-16")) == cli.EXIT_SOLVER_ERROR
    assert json.loads((out / "result.json").read_text(encoding="utf-8"))["status"] == ERROR_INPUT


@pytest.mark.parametrize("args", [
    ["run", "--a", "1"],                                               # required options missing
    command("x", base=with_option("--nx", "2.5")),                     # not an integer
    command("x", base=with_option("--q", "abc")),                      # not a number
    ["solve"],                                                         # unknown command
])
def test_command_line_errors_compute_and_write_nothing(tmp_path, monkeypatch, args):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "run_rectangle_physical", lambda *a, **k: pytest.fail("the solver was called"))
    with pytest.raises(SystemExit) as stop:
        cli.main(args)
    assert stop.value.code == cli.EXIT_USAGE and list(tmp_path.iterdir()) == []


# --- runs without convergence and solver errors --------------------------------------------------------------

def test_max_half_steps_writes_the_field_only_as_diagnostic(tmp_path, capsys):
    out = tmp_path / "short"
    assert cli.main(command(out, "--s-max", "4")) == cli.EXIT_NOT_CONVERGED
    direct = rr.run_rectangle_physical(**PLATE, n_x=24, n_y=24, s_max=4)
    assert direct.status == MAX_HALF_STEPS
    assert sorted(p.name for p in out.iterdir()) == ["field_diagnostic.csv", "history.csv", "result.json"]
    doc = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert doc["status"] == MAX_HALF_STEPS and doc["converged"] is False and doc["half_steps"] == 4
    assert doc["field"]["file"] == "field_diagnostic.csv" and doc["field"]["role"] == "diagnostic"
    assert doc["stopping_parameters"]["s_max"] == 4
    comments, field = read_field(out / "field_diagnostic.csv", (25, 25))
    assert len(comments) == 1 and "DIAGNOSTIC" in comments[0] and MAX_HALF_STEPS in comments[0]
    np.testing.assert_array_equal(field["w [m]"], direct.run.last_field())
    assert "field_diagnostic.csv (ДИАГНОСТИКА, не решение)" in capsys.readouterr().out


def test_solver_error_keeps_its_status_and_no_solution_field(tmp_path, monkeypatch):
    original = rr.next_halfstep

    def failing(record):
        if record.s == 2:                                 # the half-step s = 3 fails
            raise LinearSolveError(ERROR_LINEAR_SOLVE, "injected")
        return original(record)

    monkeypatch.setattr(rr, "next_halfstep", failing)
    out = tmp_path / "error"
    assert cli.main(command(out)) == cli.EXIT_SOLVER_ERROR
    doc = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert doc["status"] == ERROR_LINEAR_SOLVE and "injected" in doc["error"] and doc["half_steps"] == 2
    assert doc["converged"] is False and doc["field"]["role"] == "diagnostic"
    assert not (out / "field.csv").exists() and (out / "field_diagnostic.csv").exists()
    comments, _ = read_csv(out / "field_diagnostic.csv")
    assert ERROR_LINEAR_SOLVE in comments[0] and "не рассчитанное решение" in comments[0]


@pytest.mark.parametrize(("status", "code"), [(CONVERGED, 0), (STAGNATED, 1), (MAX_HALF_STEPS, 1),
                                              (ERROR_INPUT, 3), (ERROR_LINEAR_SOLVE, 3), ("ERROR_ENERGY_TEST", 3)])
def test_exit_code_and_field_file_follow_the_solver_status(status, code):
    assert cli.exit_code(status) == code
    assert cli.field_file(status, 5) == ("field.csv" if status == CONVERGED else "field_diagnostic.csv")
    assert cli.field_file(status, 0) is None


# --- protection of existing results --------------------------------------------------------------------------

def test_existing_results_are_not_overwritten(tmp_path, monkeypatch, capsys):
    out = tmp_path / "demo"
    assert cli.main(command(out)) == 0
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    monkeypatch.setattr(cli, "run_rectangle_physical", lambda *a, **k: pytest.fail("the solver was called"))
    assert cli.main(command(out)) == cli.EXIT_USAGE
    assert "--overwrite" in capsys.readouterr().err
    assert {p.name: p.read_bytes() for p in out.iterdir()} == before


def test_overwrite_replaces_and_removes_the_files_of_the_earlier_run(tmp_path):
    out = tmp_path / "demo"
    (out).mkdir()
    (out / "notes.txt").write_text("user file", encoding="utf-8")
    assert cli.main(command(out)) == 0
    assert cli.main(command(out, "--overwrite", base=with_option("--nu", "1"))) == cli.EXIT_SOLVER_ERROR
    assert sorted(p.name for p in out.iterdir()) == ["history.csv", "notes.txt", "result.json"]  # no stale field
    assert (out / "notes.txt").read_text(encoding="utf-8") == "user file"                     # other files kept
    assert json.loads((out / "result.json").read_text(encoding="utf-8"))["status"] == ERROR_INPUT


def test_an_output_path_that_is_a_file_is_refused(tmp_path):
    target = tmp_path / "file.txt"
    target.write_text("keep", encoding="utf-8")
    assert cli.main(command(target, "--overwrite")) == cli.EXIT_USAGE
    assert target.read_text(encoding="utf-8") == "keep"
