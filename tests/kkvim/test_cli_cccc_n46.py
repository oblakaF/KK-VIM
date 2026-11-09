"""``python -m kkvim run-cccc-n46`` (``kkvim.cli_cccc_n46``; D-067): what it writes against the direct call, bit for bit; statuses, exit codes and
the transactional saving of D-062. Builder tests, not the independent QA of D-067."""

import csv
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import kkvim.cccc_n23 as n23
import kkvim.cccc_n46 as cn
from kkvim import cli, cli_cccc_n46, cli_ssss

PLATE = {"a": 1.0, "b": 1.25, "t": 0.01, "E": 2.1e11, "nu": 0.3, "q": 1000.0, "nx": 16, "ny": 20}
SHA = re.compile(r"[0-9a-f]{40}")
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


def command(out, *extra, N=5, half_steps=1, **changes):
    args = ["run-cccc-n46", "--N", str(N)]
    for name, value in {**PLATE, **changes}.items():
        args += [f"--{name}", repr(value) if isinstance(value, float) else str(value)]
    return [*args, "--half-steps", str(half_steps), "--out", str(out), *extra]


def direct(N=5, half_steps=1, **changes):
    p = {**PLATE, **changes}
    return cn.run_cccc_n46_physical(p["a"], p["b"], p["t"], p["E"], p["nu"], p["q"], p["nx"], p["ny"], N, half_steps)


def document(out):
    return json.loads((out / "result.json").read_text(encoding="utf-8"))


def table(path):
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if not line.startswith("#")]
    return list(csv.DictReader(lines))


def names(out):
    return sorted(path.name for path in out.iterdir())


def snapshot(out):
    return {path.name: path.read_bytes() for path in out.iterdir()}


def failing_at(call, error):
    real, count = os.replace, [0]

    def replace(*args, **kwargs):
        count[0] += 1
        if count[0] == call:
            raise error
        return real(*args, **kwargs)

    return replace


@pytest.fixture
def earlier(tmp_path):
    out = tmp_path / "out"
    assert cli.main(command(out)) == cli.EXIT_CONVERGED
    (out / "other.txt").write_text("kept", encoding="utf-8")
    return out


# --- a working command for N = 4, 5, 6 -----------------------------------------------------------------------------------------

@pytest.mark.parametrize("N", cn.N_VALUES)
def test_a_real_run_of_the_command_end_to_end(tmp_path, N):
    out = tmp_path / "out"
    done = subprocess.run([sys.executable, "-m", "kkvim", *command(out, N=N)], capture_output=True, encoding="utf-8", errors="replace", check=False)
    assert done.returncode == 0, done.stderr
    assert "COMPLETED_FIXED_HALF_STEPS" in done.stdout and "RESEARCH_ONLY / PROVISIONAL / NOT_QUALIFIED" in done.stdout and "max|w|" in done.stdout
    assert names(out) == ["factors.csv", "field_research.csv", "history.csv", "result.json"]
    assert document(out)["input"]["N"] == N and document(out)["model"]["N"] == N


@pytest.mark.parametrize("N", cn.N_VALUES)
def test_the_field_is_the_field_of_the_solver_bit_for_bit(tmp_path, N):
    assert cli.main(command(tmp_path, N=N)) == cli.EXIT_CONVERGED
    result = direct(N)
    w, w_bar = result.field(), result.dimensionless_field()
    rows = table(tmp_path / "field_research.csv")
    assert len(rows) == w.size
    for row in rows:
        i, j = int(row["i"]), int(row["j"])
        assert float(row["w [m]"]) == float(w[i, j]) and float(row["w_bar [1]"]) == float(w_bar[i, j])
    assert document(tmp_path)["field"]["alpha_centre"] == result.steps[0].alpha_centre
    assert document(tmp_path)["field"]["w_max"]["value"] == float(np.max(np.abs(w)))


@pytest.mark.parametrize("N", cn.N_VALUES)
def test_the_factors_restore_the_half_step_bit_for_bit(tmp_path, N):
    assert cli.main(command(tmp_path, N=N)) == cli.EXIT_CONVERGED
    step = direct(N).steps[0]
    rows = table(tmp_path / "factors.csv")
    assert {row["s"] for row in rows} == {"1"} and {row["role"] for row in rows} == set(cli_cccc_n46.FACTOR_ROLES) - {"companion"}
    for role in cli_cccc_n46.FACTOR_ROLES:
        values = getattr(step, role)
        if values is None:
            continue
        values = np.asarray(values, dtype=float).reshape(len(values), -1)
        stored = [row for row in rows if row["role"] == role]
        restored = np.zeros(values.shape)
        for row in stored:
            restored[int(row["i"]) - 1, int(row["j"]) - 1] = float(row["value"])
        assert len(stored) == values.size and np.array_equal(restored, values), role
    G = np.zeros(step.G.shape)
    V = np.zeros(step.V.shape)
    for row in rows:
        if row["role"] in ("G", "V"):
            (G if row["role"] == "G" else V)[int(row["i"]) - 1, int(row["j"]) - 1] = float(row["value"])
    assert np.array_equal(G @ V.T, direct(N).field()[1:-1, 1:-1])  # the field from the stored factors


@pytest.mark.parametrize("N", cn.N_VALUES)
def test_the_history_is_the_record_of_the_solver_bit_for_bit(tmp_path, N):
    assert cli.main(command(tmp_path, N=N, half_steps=3)) == 6
    result = direct(N, half_steps=3)
    rows = table(tmp_path / "history.csv")
    assert [row["s"] for row in rows] == ["1", "2"] and [row["direction"] for row in rows] == ["Y", "X"]
    assert [row["status"] for row in rows] == [cn.HALF_STEP_DONE, cn.STOPPED_RANK_GUARD_UNCERTIFIED]
    first = result.steps[0]
    for name in ("cond_A_before_QR", "cond_A_after_QR", "R_diag_min_abs", "cond_S", "min_eig_DM", "eta", "rho_N", "kappa_rep", "P_hat"):
        assert float(rows[0][name]) == getattr(first, name), name
    assert [float(x) for x in rows[0]["A_diag"].split(";")] == np.diag(first.A).tolist()
    assert [float(rows[0][f"sigma_{i}"]) for i in range(1, N + 1)] == list(first.sigma)
    assert all(rows[0][f"sigma_{i}"] == "" for i in range(N + 1, 7))
    assert rows[0]["V_exact_rank"] == str(N) and rows[0]["exact_rank_T"] == str(N) and rows[0]["premises_status"] == cn.PREMISES_ESTABLISHED
    assert rows[0]["N"] == rows[1]["N"] == str(N) and rows[0]["orientation"] == cn.ORIENTATION
    assert rows[1]["qr_executed"] == "false" and rows[1]["eta"] == "" and rows[1]["A_diag"] == "" and "D-067" in rows[1]["reason"]


def test_the_record_of_the_run(earlier):
    d = document(earlier)
    result = direct()
    assert d["schema"] == "kkvim.cli.cccc_n46/1" and d["mode"] == "FIXED_HALF_STEPS_RESEARCH"
    assert d["application_status"] == "RESEARCH_ONLY / PROVISIONAL / NOT_QUALIFIED" and d["status"] == cn.COMPLETED_FIXED_HALF_STEPS
    assert "RESEARCH_ONLY / PROVISIONAL / NOT_QUALIFIED" in d["scientific_status"] and "D1_START = BLOCKED" in d["scientific_status"]
    assert d["rank_guard_rule"] == cn.RANK_GUARD_RULE and d["half_steps_requested"] == 1 and d["half_steps_completed"] == 1
    assert d["model"]["boundary"] == "CCCC" and d["model"]["N"] == 5 and d["model"]["decision"] == "D-067" and d["model"]["orientation"] == cn.ORIENTATION
    assert d["input"]["N"] == 5 and d["input"]["n_x"] == 16 and d["input"]["n_y"] == 20 and d["input"]["half_steps"] == 1
    assert d["input"]["E"] == {"value": 2.1e11, "unit": "Pa"} and d["D"]["value"] == result.D
    policy = d["numerical_policy"]
    assert policy["eta"]["value"] == result.steps[0].eta and "OBSERVATION" in policy["eta"]["role"] and "eta_max_C" not in policy
    assert "100u" in policy["not_transferred"] and "16u" in policy["not_transferred"] and "c_f = 10" in policy["not_transferred"]
    rank = d["rank"]
    assert rank["V_exact_rank"] == 5 and rank["full_rank_of_exact_discrete_sequence"] == "UNRESOLVED" and rank["half_step_2"] is None
    assert rank["sigma"] == list(result.steps[0].sigma) and rank["cond_S"] == result.steps[0].cond_S
    assert d["field"]["half_step"] == 1 and d["field"]["definition"] == "w(1) = G V^T" and d["field"]["role"] == "research"
    assert "ASSUMED" in d["premises_c3plus"]["P-4"] and d["premises_c3plus"]["status_by_half_steps"] == [cn.PREMISES_ESTABLISHED]
    assert any("2N + 1" in limit for limit in d["limits"]) and any("UNRESOLVED" in limit for limit in d["limits"])
    provenance = d["provenance"]
    assert SHA.fullmatch(provenance["git_commit"] or "0" * 40) and provenance["module"] == "kkvim.cli_cccc_n46"
    assert d["program"]["calls"] == ["kkvim.cccc_n46.run_cccc_n46_physical"] and d["program"]["argv"][:3] == ["run-cccc-n46", "--N", "5"]


def test_nothing_says_that_the_run_converged_or_is_qualified(tmp_path):
    for half_steps in (1, 2):
        out = tmp_path / str(half_steps)
        cli.main(command(out, half_steps=half_steps))
        text = "\n".join(path.read_text(encoding="utf-8") for path in out.iterdir())
        assert "CONVERGED" not in text and '"converged"' not in text and not (out / "field.csv").exists()
        assert not re.search(r'"status": "(PASS|QUALIFIED|ACCEPTED)"', text)


@pytest.mark.parametrize("N", cn.N_VALUES)
@pytest.mark.parametrize("half_steps", [2, 5, 8])
def test_s2_is_a_protective_stop_with_the_field_of_s1_as_diagnostic(tmp_path, N, half_steps):
    assert cli.main(command(tmp_path, N=N, half_steps=half_steps)) == 6
    assert names(tmp_path) == ["factors.csv", "field_diagnostic.csv", "history.csv", "result.json"]
    d = document(tmp_path)
    assert d["status"] == cn.STOPPED_RANK_GUARD_UNCERTIFIED and d["half_steps_completed"] == 1 and d["half_steps_requested"] == half_steps
    assert d["field"]["role"] == "diagnostic" and d["rank"]["half_step_2"]["qr_executed"] is False
    first = (tmp_path / "field_diagnostic.csv").read_text(encoding="utf-8").splitlines()[0]
    assert first.startswith("# DIAGNOSTIC") and cn.STOPPED_RANK_GUARD_UNCERTIFIED in first
    w = direct(N, half_steps=1).field()
    assert [float(row["w [m]"]) for row in table(tmp_path / "field_diagnostic.csv")] == [float(v) for v in w.reshape(-1)]


def test_every_status_has_its_exit_code_and_note():
    assert set(cli_cccc_n46.EXIT_CODES) == set(cn.STATUSES) == set(cli_cccc_n46.STATUS_NOTES)
    assert n23.STOPPED_ETA_GUARD not in cli_cccc_n46.EXIT_CODES
    assert {cli_cccc_n46.EXIT_CODES[s] for s in set(cn.STATUSES) - {cn.COMPLETED_FIXED_HALF_STEPS, cn.NOT_EXECUTED_OUTSIDE_DOMAIN, cn.ERROR_INPUT,
                                                                    cn.ERROR_NONFINITE}} == {6}
    assert cli_cccc_n46.EXIT_CODES[cn.NOT_EXECUTED_OUTSIDE_DOMAIN] == 2 and cli_cccc_n46.EXIT_CODES[cn.ERROR_INPUT] == 3
    assert cli_cccc_n46.EXIT_CODES[cn.COMPLETED_FIXED_HALF_STEPS] == 0


@pytest.mark.parametrize("changes, N, code, status", [
    ({}, 3, 2, cn.NOT_EXECUTED_OUTSIDE_DOMAIN), ({}, 7, 2, cn.NOT_EXECUTED_OUTSIDE_DOMAIN), ({"nx": 10}, 5, 2, cn.NOT_EXECUTED_OUTSIDE_DOMAIN),
    ({"ny": 12}, 6, 2, cn.NOT_EXECUTED_OUTSIDE_DOMAIN), ({"b": 6.0}, 4, 2, cn.NOT_EXECUTED_OUTSIDE_DOMAIN), ({"q": 0.0}, 4, 3, cn.ERROR_INPUT),
    ({"t": -0.01}, 5, 3, cn.ERROR_INPUT),
])
def test_a_refused_input_in_an_empty_folder_writes_only_its_record(tmp_path, changes, N, code, status):
    assert cli.main(command(tmp_path, N=N, **changes)) == code
    assert names(tmp_path) == ["result.json"] and document(tmp_path)["status"] == status and document(tmp_path)["field"] is None


def test_half_steps_outside_one_to_eight_are_refused(tmp_path):
    assert cli.main(command(tmp_path / "a", half_steps=9)) == 2 and document(tmp_path / "a")["status"] == cn.NOT_EXECUTED_OUTSIDE_DOMAIN
    assert cli.main(command(tmp_path / "b", half_steps=0)) == 2


@pytest.mark.parametrize("changes, N, code", [({}, 3, 2), ({"q": 0.0}, 5, 3)])
def test_a_refused_input_keeps_earlier_results(earlier, capsys, changes, N, code):
    before = snapshot(earlier)
    capsys.readouterr()
    assert cli.main(command(earlier, "--overwrite", N=N, **changes)) == code
    assert snapshot(earlier) == before and "не заменён" in capsys.readouterr().err


@pytest.mark.parametrize("name, value, status", [
    ("premise_failures", lambda *a: ["P-2: injected"], cn.UNRESOLVED_C3PLUS_PREMISE),
    ("residual_eta", lambda *a: float("nan"), cn.ERROR_NONFINITE),
])
def test_a_stop_at_s1_keeps_earlier_results_and_writes_no_field(earlier, tmp_path, monkeypatch, name, value, status):
    monkeypatch.setattr(n23, name, value)
    before = snapshot(earlier)
    assert cli.main(command(earlier, "--overwrite")) == cli_cccc_n46.EXIT_CODES[status] and snapshot(earlier) == before
    empty = tmp_path / "empty"
    assert cli.main(command(empty)) == cli_cccc_n46.EXIT_CODES[status]
    assert names(empty) == ["factors.csv", "history.csv", "result.json"] and document(empty)["status"] == status


def test_a_large_eta_is_written_and_does_not_stop(tmp_path, monkeypatch):
    monkeypatch.setattr(n23, "residual_eta", lambda *a: 1e-6)
    assert cli.main(command(tmp_path)) == cli.EXIT_CONVERGED
    assert document(tmp_path)["numerical_policy"]["eta"]["value"] == 1e-6 and float(table(tmp_path / "history.csv")[0]["eta"]) == 1e-6


def test_overwriting_only_with_the_option(earlier, monkeypatch, capsys):
    monkeypatch.setattr(cn, "run_cccc_n46_physical", lambda *a: pytest.fail("computed"))
    before = snapshot(earlier)
    assert cli.main(command(earlier)) == cli.EXIT_USAGE
    assert snapshot(earlier) == before and "--overwrite" in capsys.readouterr().err


def test_a_stop_at_s2_replaces_a_research_field_by_a_diagnostic_one(earlier):
    assert cli.main(command(earlier, "--overwrite", half_steps=2)) == 6
    assert names(earlier) == ["factors.csv", "field_diagnostic.csv", "history.csv", "other.txt", "result.json"]


# --- saving through the transaction of D-062 -----------------------------------------------------------------------------------

def test_a_save_goes_through_the_transaction(earlier, monkeypatch):
    monkeypatch.setattr(cli, "write_results", lambda *a: pytest.fail("the non-atomic save was used"))
    assert cli.main(command(earlier, "--overwrite", N=4)) == cli.EXIT_CONVERGED
    assert document(earlier)["model"]["N"] == 4 and (earlier / "other.txt").read_text(encoding="utf-8") == "kept"


@pytest.mark.parametrize("call", [1, 2, 3, 4])
@pytest.mark.parametrize("error, code, said", [(PermissionError("locked"), cli.EXIT_OUTPUT_ERROR, "locked"), (KeyboardInterrupt(), 130, "прерван")])
def test_an_interrupted_replacement_restores_the_earlier_set(earlier, monkeypatch, capsys, call, error, code, said):
    before = snapshot(earlier)
    with monkeypatch.context() as patched:
        patched.setattr(os, "replace", failing_at(call, error))
        capsys.readouterr()
        assert cli.main(command(earlier, "--overwrite", N=4)) == code
    err = capsys.readouterr().err
    assert snapshot(earlier) == before and said in err and "Traceback" not in err


def test_ctrl_c_during_the_calculation_is_exit_130(earlier, monkeypatch, capsys):
    def interrupt(*args):
        raise KeyboardInterrupt

    monkeypatch.setattr(cn, "run_cccc_n46_physical", interrupt)
    before = snapshot(earlier)
    assert cli.main(command(earlier, "--overwrite")) == 130 and snapshot(earlier) == before and "прерван" in capsys.readouterr().err


def test_an_internal_error_is_exit_5_and_keeps_earlier_results(earlier, monkeypatch):
    monkeypatch.setattr(cli_cccc_n46, "result_texts", lambda *a: 1 / 0)
    before = snapshot(earlier)
    assert cli.main(command(earlier, "--overwrite")) == cli.EXIT_INTERNAL_ERROR and snapshot(earlier) == before


def test_a_killed_replacement_is_refused_until_dealt_with(earlier, monkeypatch, capsys):
    before = snapshot(earlier)
    process = subprocess.Popen([sys.executable, "-c", KILLED_AT, "2", *command(earlier, "--overwrite", N=4)],
                               stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    line = process.stdout.readline()
    process.kill()
    process.communicate()
    assert line.strip() == "STOPPED", line
    assert not (earlier / "result.json").exists()
    stage = next(earlier.glob(cli_ssss.STAGE_PREFIX + "*"))
    monkeypatch.setattr(cn, "run_cccc_n46_physical", lambda *a: pytest.fail("computed"))
    for extra in (("--overwrite",), ()):
        capsys.readouterr()
        assert cli.main(command(earlier, *extra)) == cli.EXIT_USAGE
        assert "незавершённая замена" in capsys.readouterr().err
    for name in cli.OUTPUTS:
        (earlier / name).unlink(missing_ok=True)
    for path in (stage / "old").iterdir():
        os.replace(path, earlier / path.name)
    shutil.rmtree(stage)
    assert snapshot(earlier) == before


# --- routing; the other commands ----------------------------------------------------------------------------------------------

def test_the_command_is_registered_next_to_the_others(capsys):
    with pytest.raises(SystemExit):
        cli.main(["--help"])
    text = capsys.readouterr().out
    assert "run-cccc-n46" in text and "run-cccc-n23" in text and "run-cccc-n1" in text and "run-ssss" in text


def test_the_arguments_are_required(capsys):
    for missing in ("--N", "--nx", "--half-steps", "--out", "--E"):
        args = command("unused")
        index = args.index(missing)
        with pytest.raises(SystemExit):
            cli.main(args[:index] + args[index + 2:])
        assert missing in capsys.readouterr().err


def test_the_other_cccc_commands_still_run(tmp_path):
    plate = [f"--{name}={value!r}" if isinstance(value, float) else f"--{name}={value}" for name, value in PLATE.items()]
    assert cli.main(["run-cccc-n1", *plate, "--half-steps", "2", "--out", str(tmp_path / "n1")]) == cli.EXIT_CONVERGED
    assert document(tmp_path / "n1")["schema"] == "kkvim.cli.cccc_n1/1"
    assert cli.main(["run-cccc-n23", "--N", "3", *plate, "--half-steps", "1", "--out", str(tmp_path / "n23")]) == cli.EXIT_CONVERGED
    assert document(tmp_path / "n23")["schema"] == "kkvim.cli.cccc_n23/2"
    assert cli.main(["run-cccc-n23", "--N", "4", *plate, "--half-steps", "1", "--out", str(tmp_path / "n23x")]) == cli.EXIT_USAGE


def test_the_cli_module_imports_no_reference():
    text = Path(cli_cccc_n46.__file__).read_text(encoding="utf-8")
    assert "kkvim_reference" not in text and "import scripts" not in text and "cccc_n23_certificate" not in text
