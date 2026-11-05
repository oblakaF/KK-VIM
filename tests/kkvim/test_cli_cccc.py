"""``python -m kkvim run-cccc-n1`` (``kkvim.cli_cccc``; D-061; B-5, B-7, B-9 of Policy 10.1): what it writes against the direct call, bit for bit."""

import csv
import json
import os
import re
import shutil
import subprocess
import sys

import numpy as np
import pytest

import kkvim.cccc_n1 as cn
from kkvim import cli, cli_cccc, cli_ssss

PLATE = {"a": 1.0, "b": 1.25, "t": 0.01, "E": 2.1e11, "nu": 0.3, "q": 1000.0, "nx": 16, "ny": 20}
SHA = re.compile(r"[0-9a-f]{40}")


def command(out, *extra, half_steps=4, **changes):
    args = ["run-cccc-n1"]
    for name, value in {**PLATE, **changes}.items():
        args += [f"--{name}", repr(value) if isinstance(value, float) else str(value)]
    return [*args, "--half-steps", str(half_steps), "--out", str(out), *extra]


def direct(half_steps=4, **changes):
    plate = {**PLATE, **changes}
    return cn.run_cccc_n1_physical(plate["a"], plate["b"], plate["t"], plate["E"], plate["nu"], plate["q"], plate["nx"], plate["ny"], half_steps)


def document(out):
    return json.loads((out / "result.json").read_text(encoding="utf-8"))


def table(path):
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if not line.startswith("#")]
    return list(csv.DictReader(lines))


def names(out):
    return sorted(path.name for path in out.iterdir())


def snapshot(out):
    return {path.name: path.read_bytes() for path in out.iterdir()}


def text_of_all(out):
    return "\n".join(path.read_text(encoding="utf-8") for path in out.iterdir())


@pytest.fixture
def earlier(tmp_path):
    out = tmp_path / "out"
    assert cli.main(command(out)) == cli.EXIT_CONVERGED
    (out / "other.txt").write_text("kept", encoding="utf-8")
    return out


def test_a_real_run_of_the_command_end_to_end(tmp_path):
    out = tmp_path / "out"
    done = subprocess.run([sys.executable, "-m", "kkvim", *command(out)], capture_output=True, encoding="utf-8", errors="replace", check=False)
    assert done.returncode == 0, done.stderr
    assert "COMPLETED_FIXED_HALF_STEPS" in done.stdout and "RESEARCH_ONLY / PROVISIONAL / NOT_QUALIFIED" in done.stdout and "max|w|" in done.stdout
    assert names(out) == ["factors.csv", "field_research.csv", "history.csv", "result.json"]


def test_the_field_is_the_field_of_the_solver_bit_for_bit(earlier):
    result = direct()
    rows = table(earlier / "field_research.csv")
    w, w_bar = result.field(), result.dimensionless_field()
    assert len(rows) == w.size
    for row in rows:
        i, j = int(row["i"]), int(row["j"])
        assert float(row["w [m]"]) == float(w[i, j]) and float(row["w_bar [1]"]) == float(w_bar[i, j])
        assert float(row["x [m]"]) == i * (PLATE["a"] / PLATE["nx"]) and float(row["y [m]"]) == j * (PLATE["b"] / PLATE["ny"])
    boundary = [row for row in rows if int(row["i"]) in (0, PLATE["nx"]) or int(row["j"]) in (0, PLATE["ny"])]
    assert boundary and all(float(row["w [m]"]) == 0.0 for row in boundary)


def test_the_history_is_the_record_of_the_solver_bit_for_bit(earlier):
    result = direct()
    rows = table(earlier / "history.csv")
    assert [row["direction"] for row in rows] == list("YXYX") and [int(row["k"]) for row in rows] == [0, 1, 1, 2]
    assert [row["status"] for row in rows] == [cn.HALF_STEP_DONE] * 3 + [cn.COMPLETED_FIXED_HALF_STEPS]
    for row, step in zip(rows, result.steps, strict=True):
        for name in ("h_s", "A", "B", "C", "F", "C_L", "E", "cond_S", "min_DM", "eta", "c", "sigma", "P_hat", "delta_P_hat", "field_change_inf",
                     "alpha_centre"):
            assert float(row[name]) == getattr(step, name), name
        assert float(row["B_over_A"]) == step.B / step.A and float(row["C_over_A"]) == step.C / step.A
        assert float(row["F2_over_A"]) == step.F * step.F / step.A
        assert row["premises_status"] == cn.PREMISES_ESTABLISHED and int(row["n_s"]) == step.n_s


def test_the_factors_restore_every_half_step_bit_for_bit(earlier):
    result = direct()
    rows = table(earlier / "factors.csv")
    for step in result.steps:
        for role, values in (("given", step.given), ("raw", step.raw), ("next_given", step.next_given), ("next_conjugate", step.next_conjugate)):
            stored = [row for row in rows if int(row["s"]) == step.s and row["role"] == role]
            assert [int(row["index"]) for row in stored] == list(range(1, values.size + 1))
            assert np.array_equal(np.array([float(row["value"]) for row in stored]), values)
            assert {row["direction"] for row in stored} == {step.direction}


def test_the_record_of_the_run(earlier):
    d = document(earlier)
    assert d["schema"] == "kkvim.cli.cccc_n1/1" and d["mode"] == "FIXED_HALF_STEPS_RESEARCH"
    assert d["application_status"] == "RESEARCH_ONLY / PROVISIONAL / NOT_QUALIFIED" and d["status"] == cn.COMPLETED_FIXED_HALF_STEPS
    assert d["half_steps_requested"] == d["half_steps_completed"] == 4 and d["cycles_k"] == 2 and d["reason"] is None
    assert d["model"]["boundary"] == "CCCC" and d["model"]["N"] == 1 and d["model"]["decision"] == "D-061"
    assert {name: d["input"][name]["value"] for name in ("a", "b", "t", "E", "nu", "q")} == {name: PLATE[name] for name in ("a", "b", "t", "E", "nu", "q")}
    assert d["input"]["n_x"] == 16 and d["input"]["n_y"] == 20 and d["input"]["half_steps"] == 4
    assert d["D"] == {"value": direct().D, "unit": "N·m"} and d["input"]["a"]["unit"] == "m" and d["input"]["q"]["unit"] == "Pa"
    assert d["field"]["role"] == "research" and d["field"]["units"]["w"] == "m" and d["field"]["alpha_centre"] == direct().steps[-1].alpha_centre
    assert d["numerical_policy"]["eta_max_C"]["value"] == 100 * 2.0**-53 and d["premises_c3plus"]["status_by_half_steps"] == [cn.PREMISES_ESTABLISHED]
    assert "ASSUMED" in d["premises_c3plus"]["P-4"] and d["evidence_scope"] and d["limits"]
    assert d["program"]["argv"][0] == "run-cccc-n1"
    provenance = d["provenance"]
    assert provenance["module"] == "kkvim.cli_cccc" and "numpy" in provenance["versions"]
    assert provenance["git_commit"] is None or SHA.fullmatch(provenance["git_commit"])


def test_nothing_says_that_the_run_converged_or_is_qualified(earlier):
    text = text_of_all(earlier)
    assert not re.search(r"\bCONVERGED\b|\bconverged\b|\bqualified\b|\bACCEPTED\b", text)
    assert not (earlier / "field.csv").exists()
    assert not re.search(r"\bPASS\b", json.dumps({k: document(earlier)[k] for k in ("status", "application_status", "mode")}))
    for name in ("history.csv", "factors.csv", "field_research.csv"):
        assert "RESEARCH_ONLY / PROVISIONAL / NOT_QUALIFIED" in (earlier / name).read_text(encoding="utf-8").splitlines()[0]


def test_a_grid_without_a_centre_node_says_so(tmp_path):
    out = tmp_path / "out"
    assert cli.main(command(out, nx=15, half_steps=2)) == cli.EXIT_CONVERGED
    assert document(out)["field"]["alpha_centre"] is None and document(out)["field"]["alpha_note"] == "центр не узел"
    rows = table(out / "history.csv")
    assert rows[-1]["alpha_centre"] == "" and rows[-1]["alpha_note"] == "центр не узел"


# --- refusals --------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("changes, half_steps, code, status", [
    ({"nx": 6}, 2, cli.EXIT_USAGE, cn.NOT_EXECUTED_OUTSIDE_DOMAIN), ({"ny": 1281}, 2, cli.EXIT_USAGE, cn.NOT_EXECUTED_OUTSIDE_DOMAIN),
    ({}, 9, cli.EXIT_USAGE, cn.NOT_EXECUTED_OUTSIDE_DOMAIN), ({}, 0, cli.EXIT_USAGE, cn.NOT_EXECUTED_OUTSIDE_DOMAIN),
    ({"b": 5.5}, 2, cli.EXIT_USAGE, cn.NOT_EXECUTED_OUTSIDE_DOMAIN), ({"nu": 1.5}, 2, cli.EXIT_SOLVER_ERROR, cn.ERROR_INPUT),
    ({"q": 0.0}, 2, cli.EXIT_SOLVER_ERROR, cn.ERROR_INPUT), ({"t": -1.0}, 2, cli.EXIT_SOLVER_ERROR, cn.ERROR_INPUT),
    ({"q": float("nan")}, 2, cli.EXIT_SOLVER_ERROR, cn.ERROR_INPUT),
])
def test_a_refused_input_in_an_empty_folder_writes_only_its_record(tmp_path, changes, half_steps, code, status):
    out = tmp_path / "out"
    assert cli.main(command(out, half_steps=half_steps, **changes)) == code
    assert names(out) == ["result.json"]
    d = document(out)
    assert d["status"] == status and d["reason"] and d["field"] is None and d["history"] is None and d["half_steps_completed"] == 0
    assert d["application_status"] == "RESEARCH_ONLY / PROVISIONAL / NOT_QUALIFIED"


@pytest.mark.parametrize("changes, half_steps, code", [({"nx": 6}, 2, cli.EXIT_USAGE), ({"nu": 1.5}, 2, cli.EXIT_SOLVER_ERROR), ({}, 9, cli.EXIT_USAGE)])
def test_a_refused_input_keeps_earlier_results(earlier, capsys, changes, half_steps, code):
    before = snapshot(earlier)
    capsys.readouterr()
    assert cli.main(command(earlier, "--overwrite", half_steps=half_steps, **changes)) == code
    assert snapshot(earlier) == before
    err = capsys.readouterr().err
    assert "Причина:" in err and "не изменены" in err and "factors.csv" in err


@pytest.mark.parametrize("extra", [("--N", "2"), ("--N", "1"), ("--bc", "SSSS"), ("--tol-E", "1e-6")])
def test_options_of_other_modes_do_not_exist(tmp_path, extra, capsys):
    with pytest.raises(SystemExit) as stop:
        cli.main(command(tmp_path / "out", *extra))
    assert stop.value.code == cli.EXIT_USAGE and not (tmp_path / "out").exists()
    capsys.readouterr()


def test_the_number_of_half_steps_has_no_default(tmp_path, capsys):
    args = command(tmp_path / "out")
    position = args.index("--half-steps")
    with pytest.raises(SystemExit) as stop:
        cli.main(args[:position] + args[position + 2:])
    assert stop.value.code == cli.EXIT_USAGE and not (tmp_path / "out").exists()
    capsys.readouterr()


def test_overwriting_only_with_the_option(earlier, monkeypatch, capsys):
    monkeypatch.setattr(cn, "run_cccc_n1_physical", lambda *a: pytest.fail("computed"))
    before = snapshot(earlier)
    capsys.readouterr()
    assert cli.main(command(earlier, half_steps=2)) == cli.EXIT_USAGE
    assert snapshot(earlier) == before and "--overwrite" in capsys.readouterr().err


def test_factors_alone_are_a_result_file_of_this_command(tmp_path, monkeypatch):
    monkeypatch.setattr(cn, "run_cccc_n1_physical", lambda *a: pytest.fail("computed"))
    (tmp_path / "factors.csv").write_text("x", encoding="utf-8")
    assert cli.main(command(tmp_path)) == cli.EXIT_USAGE


def test_a_valid_overwrite_replaces_the_whole_set(earlier):
    before = snapshot(earlier)
    assert cli.main(command(earlier, "--overwrite", half_steps=3)) == cli.EXIT_CONVERGED
    assert names(earlier) == ["factors.csv", "field_research.csv", "history.csv", "other.txt", "result.json"]
    assert (earlier / "history.csv").read_bytes() != before["history.csv"] and document(earlier)["half_steps_completed"] == 3
    assert (earlier / "other.txt").read_bytes() == before["other.txt"]


def test_the_folder_is_checked_before_the_run(tmp_path, monkeypatch):
    monkeypatch.setattr(cn, "run_cccc_n1_physical", lambda *a: pytest.fail("computed"))
    (tmp_path / "file").write_text("x", encoding="utf-8")
    assert cli.main(command(tmp_path / "file")) == cli.EXIT_USAGE
    assert cli.main(command(tmp_path / "file" / "under")) == cli.EXIT_OUTPUT_ERROR


# --- stops and failures ------------------------------------------------------------------------------------------------------------

def stop_second_half_step(monkeypatch, name, replacement):
    real, count = getattr(cn, name), [0]

    def call(*args, **kwargs):
        count[0] += 1
        return replacement(*args, **kwargs) if count[0] == 2 else real(*args, **kwargs)

    monkeypatch.setattr(cn, name, call)


@pytest.mark.parametrize("name, replacement, status, code", [
    ("residual_eta", lambda *a: 1.0, cn.STOPPED_ETA_GUARD, cli_cccc.EXIT_PROTECTIVE_STOP),
    ("solve_bc", lambda *a: np.zeros(a[-1] - 1), cn.STOP_ZERO_FIELD, cli_cccc.EXIT_PROTECTIVE_STOP),
    ("solve_bc", lambda *a: np.full(a[-1] - 1, np.nan), cn.ERROR_NONFINITE, cli.EXIT_SOLVER_ERROR),
    ("premise_failures", lambda *a: ["P-3: x"], cn.UNRESOLVED_C3PLUS_PREMISE, cli_cccc.EXIT_PROTECTIVE_STOP),
])
def test_a_stop_saves_a_diagnostic_field_and_never_a_research_one(tmp_path, monkeypatch, capsys, name, replacement, status, code):
    out = tmp_path / "out"
    stop_second_half_step(monkeypatch, name, replacement)
    capsys.readouterr()
    assert cli.main(command(out)) == code
    assert names(out) == ["factors.csv", "field_diagnostic.csv", "history.csv", "result.json"]
    d = document(out)
    assert d["status"] == status and d["reason"] and d["half_steps_completed"] == 1 and d["field"]["role"] == "diagnostic"
    rows = table(out / "history.csv")
    assert [row["status"] for row in rows] == [cn.HALF_STEP_DONE, status] and rows[-1]["reason"] == d["reason"]
    first = (out / "field_diagnostic.csv").read_text(encoding="utf-8").splitlines()[0]
    assert first.startswith("# DIAGNOSTIC") and status in first and "Причина" in capsys.readouterr().err


def test_every_status_has_its_exit_code_and_note():
    statuses = {value for name, value in vars(cn).items() if isinstance(value, str) and name.isupper()
                and name in ("COMPLETED_FIXED_HALF_STEPS", "NOT_EXECUTED_OUTSIDE_DOMAIN", "ERROR_INPUT", "ERROR_NONFINITE", "UNRESOLVED_C3PLUS_PREMISE",
                             "PROTECTIVE_STOP_FACTORIZATION", "STOPPED_ETA_GUARD", "STOP_ZERO_FIELD")}
    assert len(statuses) == 8 and set(cli_cccc.EXIT_CODES) == statuses and set(cli_cccc.STATUS_NOTES) == statuses
    assert [status for status, code in cli_cccc.EXIT_CODES.items() if code == cli.EXIT_CONVERGED] == [cn.COMPLETED_FIXED_HALF_STEPS]


def test_a_guard_stop_at_the_first_half_step_keeps_earlier_results(earlier, monkeypatch):
    monkeypatch.setattr(cn, "residual_eta", lambda *a: 1.0)
    before = snapshot(earlier)
    assert cli.main(command(earlier, "--overwrite")) == cli_cccc.EXIT_PROTECTIVE_STOP
    assert snapshot(earlier) == before


def test_an_error_after_a_half_step_keeps_earlier_results(earlier, monkeypatch):
    stop_second_half_step(monkeypatch, "solve_bc", lambda *a: np.full(a[-1] - 1, np.nan))
    before = snapshot(earlier)
    assert cli.main(command(earlier, "--overwrite")) == cli.EXIT_SOLVER_ERROR
    assert snapshot(earlier) == before


def test_a_write_failure_is_exit_4(earlier, monkeypatch, capsys):
    def fail(*args):
        raise PermissionError("locked")

    monkeypatch.setattr(os, "replace", fail)
    capsys.readouterr()
    assert cli.main(command(earlier, "--overwrite", half_steps=3)) == cli.EXIT_OUTPUT_ERROR
    assert "locked" in capsys.readouterr().err


def test_an_internal_error_is_exit_5(earlier, monkeypatch, capsys):
    monkeypatch.setattr(cn, "run_cccc_n1_physical", lambda *a: 1 / 0)
    before = snapshot(earlier)
    capsys.readouterr()
    assert cli.main(command(earlier, "--overwrite")) == cli.EXIT_INTERNAL_ERROR
    assert snapshot(earlier) == before and "не статус решателя" in capsys.readouterr().err


def test_an_interrupt_keeps_earlier_results(earlier, monkeypatch, capsys):
    def interrupt(*args):
        raise KeyboardInterrupt

    monkeypatch.setattr(cn, "run_cccc_n1_physical", interrupt)
    before = snapshot(earlier)
    capsys.readouterr()
    assert cli.main(command(earlier, "--overwrite")) == 130
    assert snapshot(earlier) == before and "прерван" in capsys.readouterr().err


# --- saving through the transaction of run-ssss (D-062) ---------------------------------------------------------------------------

STAGE = cli_ssss.STAGE_PREFIX + "*"
SSSS_COMMANDS = [["run"], ["run-ssss", "--N", "1"], ["run-ssss", "--N", "3", "--half-steps", "2"]]
KILLED_AT = """
import os, sys, time
from kkvim import cli
real, count = os.replace, [0]
def replace(*args, **kwargs):
    count[0] += 1
    if count[0] == int(sys.argv[1]):
        print("STOPPED", flush=True)
        time.sleep(3600)
    return real(*args, **kwargs)
os.replace = replace
sys.exit(cli.main(sys.argv[2:]))
"""


def plate_args(**changes):
    args = []
    for name, value in {**PLATE, **changes}.items():
        args += [f"--{name}", repr(value) if isinstance(value, float) else str(value)]
    return args


def failing_at(call, error):
    """os.replace raising ``error`` at its ``call``-th use; the real one otherwise."""
    real, count = os.replace, [0]

    def replace(*args, **kwargs):
        count[0] += 1
        if count[0] == call:
            raise error
        return real(*args, **kwargs)

    return replace


def test_a_save_replaces_the_whole_set_through_the_transaction(earlier, monkeypatch):
    monkeypatch.setattr(cli, "write_results", lambda *a: pytest.fail("the non-atomic save was used"))
    assert cli.main(command(earlier, "--overwrite", half_steps=3)) == cli.EXIT_CONVERGED
    assert names(earlier) == ["factors.csv", "field_research.csv", "history.csv", "other.txt", "result.json"]
    assert document(earlier)["half_steps_completed"] == 3 and len(table(earlier / "history.csv")) == 3
    assert (earlier / "other.txt").read_text(encoding="utf-8") == "kept"


@pytest.mark.parametrize("call", [1, 2, 3, 4])
@pytest.mark.parametrize("error, code, said", [(PermissionError("locked"), cli.EXIT_OUTPUT_ERROR, "locked"), (KeyboardInterrupt(), 130, "прерван")])
def test_an_interrupted_replacement_restores_the_earlier_set(earlier, monkeypatch, capsys, call, error, code, said):
    before = snapshot(earlier)
    with monkeypatch.context() as patched:
        patched.setattr(os, "replace", failing_at(call, error))
        capsys.readouterr()
        assert cli.main(command(earlier, "--overwrite", half_steps=3)) == code
    err = capsys.readouterr().err
    assert snapshot(earlier) == before and said in err and "не изменены" in err and "Traceback" not in err
    assert cli.main(command(earlier, "--overwrite", half_steps=3)) == cli.EXIT_CONVERGED


def test_ctrl_c_while_the_new_files_are_written_is_exit_130(earlier, monkeypatch, capsys):
    def interrupt(*args):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli_ssss, "_write_file", interrupt)
    before = snapshot(earlier)
    assert cli.main(command(earlier, "--overwrite", half_steps=3)) == 130
    assert snapshot(earlier) == before and "прерван" in capsys.readouterr().err


def test_a_killed_replacement_is_refused_until_dealt_with_and_the_earlier_set_comes_back(earlier, monkeypatch, capsys):
    before = snapshot(earlier)
    process = subprocess.Popen([sys.executable, "-c", KILLED_AT, "2", *command(earlier, "--overwrite", half_steps=3)],
                               stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    line = process.stdout.readline()
    process.kill()
    process.communicate()
    assert line.strip() == "STOPPED", line
    assert not (earlier / "result.json").exists()  # a mixed set never looks like a result
    stage = next(earlier.glob(STAGE))
    assert (stage / cli_ssss.REPLACING_MARK).exists() and (stage / "old" / "factors.csv").read_bytes() == before["factors.csv"]
    monkeypatch.setattr(cn, "run_cccc_n1_physical", lambda *a: pytest.fail("computed"))
    for extra in (("--overwrite",), ()):
        capsys.readouterr()
        assert cli.main(command(earlier, *extra, half_steps=2)) == cli.EXIT_USAGE
        err = capsys.readouterr().err
        assert stage.name in err and "незавершённая замена" in err and "factors.csv" in err
    for name in cli.OUTPUTS:  # the instruction printed
        (earlier / name).unlink(missing_ok=True)
    for path in (stage / "old").iterdir():
        os.replace(path, earlier / path.name)
    shutil.rmtree(stage)
    assert snapshot(earlier) == before


@pytest.mark.parametrize("new", SSSS_COMMANDS)
def test_stale_factors_do_not_stay_next_to_new_ssss_results(earlier, new):
    assert cli.main([*new, *plate_args(), "--out", str(earlier), "--overwrite"]) == cli.EXIT_CONVERGED
    assert not (earlier / "factors.csv").exists() and (earlier / "other.txt").exists()
    assert document(earlier).get("schema") != cli_cccc.SCHEMA


@pytest.mark.parametrize("new", SSSS_COMMANDS)
def test_an_existing_factors_file_protects_the_folder_from_every_command(tmp_path, capsys, new):
    (tmp_path / "factors.csv").write_text("x", encoding="utf-8")
    assert cli.main([*new, *plate_args(), "--out", str(tmp_path)]) == cli.EXIT_USAGE
    assert snapshot(tmp_path) == {"factors.csv": b"x"} and "factors.csv" in capsys.readouterr().err


def test_a_failed_ssss_replacement_gives_the_factors_back(earlier, monkeypatch):
    before = snapshot(earlier)
    monkeypatch.setattr(os, "replace", failing_at(1, KeyboardInterrupt()))
    assert cli.main(["run-ssss", "--N", "1", *plate_args(), "--out", str(earlier), "--overwrite"]) == 130
    assert snapshot(earlier) == before


def test_new_results_replace_the_field_of_an_ssss_run(tmp_path):
    assert cli.main(["run-ssss", "--N", "1", *plate_args(), "--out", str(tmp_path)]) == cli.EXIT_CONVERGED
    assert cli.main(command(tmp_path, "--overwrite")) == cli.EXIT_CONVERGED
    assert names(tmp_path) == ["factors.csv", "field_research.csv", "history.csv", "result.json"]


def test_the_numbers_and_statuses_of_the_first_implementation_are_kept(tmp_path):
    """The run reported at d68cb8d. The tolerance only absorbs the summation order of another BLAS (D-025); it is no criterion of the method."""
    assert cli.main(command(tmp_path, half_steps=8, b=1.0, nx=32, ny=32)) == cli.EXIT_CONVERGED
    d = document(tmp_path)
    assert d["status"] == cn.COMPLETED_FIXED_HALF_STEPS and d["mode"] == "FIXED_HALF_STEPS_RESEARCH"
    assert d["application_status"] == "RESEARCH_ONLY / PROVISIONAL / NOT_QUALIFIED"
    assert d["premises_c3plus"]["status_by_half_steps"] == [cn.PREMISES_ESTABLISHED]
    assert d["numerical_policy"]["eta_max_C"]["value"] == 100 * 2.0**-53
    assert d["field"]["alpha_centre"] == pytest.approx(0.0012741269462195456, rel=1e-12)
    w = direct(half_steps=8, b=1.0, nx=32, ny=32).field()
    assert w[16, 16] == float(next(row for row in table(tmp_path / "field_research.csv") if (row["i"], row["j"]) == ("16", "16"))["w [m]"])
    assert abs(w[16, 16]) == np.abs(w).max() and abs(w[16, 16]) == pytest.approx(6.625460120341636e-05, rel=1e-12)


# --- routing -----------------------------------------------------------------------------------------------------------------------

def test_the_command_is_registered_next_to_the_others(capsys):
    with pytest.raises(SystemExit) as stop:
        cli.main(["--help"])
    assert stop.value.code == 0
    text = capsys.readouterr().out
    assert "run-cccc-n1" in text and "run-ssss" in text and "run " in text


def test_ssss_routes_are_not_affected(tmp_path, monkeypatch):
    monkeypatch.setattr(cn, "run_cccc_n1_physical", lambda *a: pytest.fail("the CCCC route"))
    plate = ["--a", "1", "--b", "1", "--t", "0.01", "--E", "2.1e11", "--nu", "0.3", "--q", "1000", "--nx", "24", "--ny", "24"]
    assert cli.main(["run", *plate, "--out", str(tmp_path / "run")]) == cli.EXIT_CONVERGED
    assert cli.main(["run-ssss", "--N", "1", *plate, "--out", str(tmp_path / "ssss")]) == cli.EXIT_CONVERGED
    assert cli.main(["run-ssss", "--N", "3", *plate, "--half-steps", "2", "--out", str(tmp_path / "n3")]) == cli.EXIT_CONVERGED
    assert cli.main(["run-ssss", "--N", "3", "--bc", "CCCC", *plate, "--out", str(tmp_path / "refused")]) == cli.EXIT_USAGE
    assert (tmp_path / "ssss" / "field.csv").exists() and json.loads((tmp_path / "n3" / "result.json").read_text(encoding="utf-8"))["solver_path"]
