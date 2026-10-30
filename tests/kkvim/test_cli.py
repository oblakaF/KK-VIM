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
    conditions = doc["applicability"]["necessary_grid_conditions_6_1"]
    assert (conditions["min_n_at_least_20"], conditions["max_n_at_most_640"],
            conditions["n_x_and_n_y_multiples_of_4"]) == (True, True, True)
    assert "max_n_within_qualified_640" not in doc["applicability"]
    assert_no_qualification_verdict(doc)

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


# --- B-1: errors of saving and internal errors are not statuses of the solver ------------------------------

def test_exit_codes_are_distinct():
    codes = (cli.EXIT_CONVERGED, cli.EXIT_NOT_CONVERGED, cli.EXIT_USAGE, cli.EXIT_SOLVER_ERROR,
             cli.EXIT_OUTPUT_ERROR, cli.EXIT_INTERNAL_ERROR)
    assert codes == (0, 1, 2, 3, 4, 5)


def test_a_folder_under_a_regular_file_is_an_output_error(tmp_path, monkeypatch, capsys):
    blocker = tmp_path / "file.txt"
    blocker.write_text("keep", encoding="utf-8")
    monkeypatch.setattr(cli, "run_rectangle_physical", lambda *a, **k: pytest.fail("the solver was called"))
    assert cli.main(command(blocker / "sub")) == cli.EXIT_OUTPUT_ERROR
    captured = capsys.readouterr()
    assert "не удалось создать папку результатов" in captured.err and "Traceback" not in captured.err
    assert "Результаты:" not in captured.out
    assert blocker.read_text(encoding="utf-8") == "keep" and sorted(p.name for p in tmp_path.iterdir()) == ["file.txt"]


def test_python_m_kkvim_reports_an_output_error_without_a_traceback(tmp_path):
    (tmp_path / "file.txt").write_text("keep", encoding="utf-8")
    done = subprocess.run([sys.executable, "-m", "kkvim", *command("file.txt/sub")], cwd=tmp_path,
                          capture_output=True, timeout=300)
    err = done.stderr.decode("utf-8")
    assert done.returncode == cli.EXIT_OUTPUT_ERROR and "Traceback" not in err and "file.txt" in err
    assert (tmp_path / "file.txt").read_text(encoding="utf-8") == "keep"


def test_a_write_error_after_the_run_leaves_the_folder_as_it_was(tmp_path, monkeypatch, capsys):
    out = tmp_path / "demo"
    assert cli.main(command(out)) == 0
    (out / "result.json.tmp").write_text("user file", encoding="utf-8")       # a name a naive writer would use
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    capsys.readouterr()

    def failing_replace(source, target):
        raise PermissionError(13, "access denied", str(target))

    monkeypatch.setattr(cli.os, "replace", failing_replace)
    code = cli.main(command(out, "--overwrite", base=with_option("--nu", "1")))   # ERROR_INPUT would drop field.csv
    assert code == cli.EXIT_OUTPUT_ERROR
    captured = capsys.readouterr()
    assert "Сохранение результатов завершилось ошибкой: замена " in captured.err and ERROR_INPUT in captured.err
    assert "Прежние файлы результатов" in captured.err and "не изменены" in captured.err
    assert "ВНИМАНИЕ" not in captured.err and "Уже " not in captured.err      # nothing changed: no false alarm
    assert "Результаты:" not in captured.out and "Traceback" not in captured.err
    assert {p.name: p.read_bytes() for p in out.iterdir()} == before          # nothing removed, replaced or left


# --- B-1a: a partial update is reported exactly ---------------------------------------------------------------

def partial_setup(tmp_path, *earlier):
    """An earlier run in ``out`` with files of other names; returns (out, bytes of every file before)."""
    out = tmp_path / "demo"
    assert cli.main(command(out, *earlier)) in (0, 1)
    (out / "notes.txt").write_text("user file", encoding="utf-8")
    (out / "result.json.tmp").write_text("user tmp", encoding="utf-8")
    return out, {p.name: p.read_bytes() for p in out.iterdir()}


def fail_on_call(monkeypatch, name, number):
    """``cli.os.<name>`` fails on its ``number``-th call (a file locked by another program)."""
    original, calls = getattr(cli.os, name), []

    def operation(*args):
        calls.append(args)
        if len(calls) == number:
            raise PermissionError(13, "file is locked by another process", str(args[-1]))
        return original(*args)

    monkeypatch.setattr(cli.os, name, operation)
    return calls


def check_partial(out, before, captured, *, replaced, failed, kept_old, extra=()):
    files = {p.name: p.read_bytes() for p in out.iterdir()}
    assert not any(name.startswith(".kkvim-") for name in files)                      # no own temporary file left
    assert files["notes.txt"] == before["notes.txt"] and files["result.json.tmp"] == before["result.json.tmp"]
    for name in replaced:
        assert files[name] != before[name], name                                       # really the new run
    for name in kept_old:
        assert files[name] == before[name], name                                       # really the earlier run
    err = captured.err
    assert "Сохранение результатов завершилось ошибкой" in err and failed in err
    assert ("Уже заменены новыми: " + ", ".join(replaced)) in err if replaced else "Уже заменены" not in err
    for line in extra:
        assert line in err, line
    assert "ВНИМАНИЕ: каталог может содержать результаты разных расчётов" in err
    assert "Использовать его как завершённый расчёт нельзя" in err and "не изменены" not in err
    assert "Результаты:" not in captured.out and "Traceback" not in err


@pytest.mark.parametrize(("number", "replaced", "failed", "kept_old"), [
    (2, ["field.csv"], "Не выполнено: history.csv (замена), result.json (замена)", ["history.csv", "result.json"]),
    (3, ["field.csv", "history.csv"], "Не выполнено: result.json (замена)", ["result.json"]),
])
def test_a_failed_second_or_third_replacement_is_reported_as_partial(tmp_path, monkeypatch, capsys, number,
                                                                    replaced, failed, kept_old):
    out, before = partial_setup(tmp_path)
    capsys.readouterr()
    calls = fail_on_call(monkeypatch, "replace", number)
    assert cli.main(command(out, "--overwrite", base=with_option("--q", "2000"))) == cli.EXIT_OUTPUT_ERROR
    assert len(calls) == number                                                        # stopped at the failure
    check_partial(out, before, capsys.readouterr(), replaced=replaced, failed=failed, kept_old=kept_old)


def test_a_stale_diagnostic_field_that_cannot_be_removed_is_reported(tmp_path, monkeypatch, capsys):
    out, before = partial_setup(tmp_path, "--s-max", "4")                             # MAX_HALF_STEPS
    assert "field_diagnostic.csv" in before and "field.csv" not in before
    capsys.readouterr()
    fail_on_call(monkeypatch, "remove", 1)
    assert cli.main(command(out, "--overwrite")) == cli.EXIT_OUTPUT_ERROR             # CONVERGED: field.csv
    captured = capsys.readouterr()
    check_partial(out, before, captured, replaced=["history.csv", "result.json"],
                  failed="Не выполнено: field_diagnostic.csv (удаление устаревшего файла)",
                  kept_old=["field_diagnostic.csv"], extra=("Уже записаны (новые): field.csv",))
    assert json.loads((out / "result.json").read_text(encoding="utf-8"))["status"] == CONVERGED
    assert (out / "field.csv").exists() and (out / "field_diagnostic.csv").exists()   # two runs side by side


def test_a_failed_temporary_write_changes_no_result(tmp_path, monkeypatch, capsys):
    out, before = partial_setup(tmp_path)
    capsys.readouterr()
    written = []
    original = cli._write_temporary

    def failing(folder, text):
        if written:
            raise OSError(28, "No space left on device")
        written.append(original(folder, text))
        return written[-1]

    monkeypatch.setattr(cli, "_write_temporary", failing)
    assert cli.main(command(out, "--overwrite", base=with_option("--q", "2000"))) == cli.EXIT_OUTPUT_ERROR
    captured = capsys.readouterr()
    assert "запись временного файла для history.csv" in captured.err and "не изменены" in captured.err
    assert "ВНИМАНИЕ" not in captured.err and "Результаты:" not in captured.out
    assert {p.name: p.read_bytes() for p in out.iterdir()} == before and not written[0].exists()


def test_an_unexpected_error_of_the_computation_is_exit_5(tmp_path, monkeypatch, capsys):
    def broken(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(cli, "run_rectangle_physical", broken)
    out = tmp_path / "demo"
    assert cli.main(command(out)) == cli.EXIT_INTERNAL_ERROR
    captured = capsys.readouterr()
    assert "Внутренняя ошибка программы" in captured.err and "не статус решателя" in captured.err
    assert "RuntimeError: boom" in captured.err and "Результаты:" not in captured.out
    assert list(out.iterdir()) == []


def test_an_unexpected_error_while_preparing_the_files_is_exit_5(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "result_document", lambda *a, **k: 1 / 0)
    out = tmp_path / "demo"
    assert cli.main(command(out)) == cli.EXIT_INTERNAL_ERROR
    assert "ZeroDivisionError" in capsys.readouterr().err and list(out.iterdir()) == []


# --- B-2: necessary grid conditions are facts, never a qualification verdict ---------------------------------

def assert_no_qualification_verdict(doc):
    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                assert not ("qualif" in key.lower() and value is True), key
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)
        else:
            assert node not in ("QUALIFIED", "PASS", "ACCEPTED"), node

    walk({k: v for k, v in doc.items() if k != "model"})   # "model" names the accepted solver R3 (D-043)


@pytest.mark.parametrize(("sides", "grid", "expected", "diagnostic"), [
    (("1", "1"), (8, 8), (False, True, True), True),            # min(n) < 20
    (("1", "1"), (22, 22), (True, True, False), True),          # not multiples of 4
    (("1", "1.7"), (40, 68), (True, True, True), False),        # r = 1.7 is not one of the ten approved ratios
])
def test_grids_outside_the_qualification_set_are_not_presented_as_qualified(tmp_path, capsys, sides, grid,
                                                                           expected, diagnostic):
    base = with_option("--b", sides[1])
    base[base.index("--nx") + 1], base[base.index("--ny") + 1] = str(grid[0]), str(grid[1])
    out = tmp_path / "grid"
    code = cli.main(command(out, base=base))
    direct = rr.run_rectangle_physical(**{**PLATE, "b": float(sides[1])}, n_x=grid[0], n_y=grid[1])
    assert code == cli.exit_code(direct.status)               # the run itself is allowed (diagnostic)
    doc = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert doc["status"] == direct.status and doc["half_steps"] == direct.run.s
    conditions = doc["applicability"]["necessary_grid_conditions_6_1"]
    assert (conditions["min_n_at_least_20"], conditions["max_n_at_most_640"],
            conditions["n_x_and_n_y_multiples_of_4"]) == expected
    note = conditions["note"]
    for text in ("min(n_x, n_y) ≥ 20", "max(n_x, n_y) ≤ 640", "кратны 4", "десять утверждённых",
                 "не означает научной приёмки", "отношение сторон и семейство командой не проверяются"):
        assert text in note, text
    assert set(conditions) == {"min_n_at_least_20", "max_n_at_most_640", "n_x_and_n_y_multiples_of_4", "note"}
    assert_no_qualification_verdict(doc)
    assert ("расчёт диагностический" in capsys.readouterr().out) == diagnostic
