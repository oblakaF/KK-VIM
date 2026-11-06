"""``python -m kkvim run-cccc-n23`` (``kkvim.cli_cccc_n23``; D-063; B-5, B-7, B-9 of Policy 9.1): what it writes against the direct call, bit for bit."""

import csv
import json
import os
import re
import shutil
import subprocess
import sys

import numpy as np
import pytest

import kkvim.cccc_n23 as cn
from kkvim import cli, cli_cccc_n23, cli_ssss

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


def command(out, *extra, N=3, half_steps=1, **changes):
    args = ["run-cccc-n23", "--N", str(N)]
    for name, value in {**PLATE, **changes}.items():
        args += [f"--{name}", repr(value) if isinstance(value, float) else str(value)]
    return [*args, "--half-steps", str(half_steps), "--out", str(out), *extra]


def direct(N=3, half_steps=1, **changes):
    p = {**PLATE, **changes}
    return cn.run_cccc_n23_physical(p["a"], p["b"], p["t"], p["E"], p["nu"], p["q"], p["nx"], p["ny"], N, half_steps)


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


def test_a_real_run_of_the_command_end_to_end(tmp_path):
    out = tmp_path / "out"
    done = subprocess.run([sys.executable, "-m", "kkvim", *command(out)], capture_output=True, encoding="utf-8", errors="replace", check=False)
    assert done.returncode == 0, done.stderr
    assert "COMPLETED_FIXED_HALF_STEPS" in done.stdout and "RESEARCH_ONLY / PROVISIONAL / NOT_QUALIFIED" in done.stdout and "max|w|" in done.stdout
    assert names(out) == ["factors.csv", "field_research.csv", "history.csv", "result.json"]


@pytest.mark.parametrize("N", [2, 3])
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


@pytest.mark.parametrize("N", [2, 3])
def test_the_factors_restore_the_half_step_bit_for_bit(tmp_path, N):
    assert cli.main(command(tmp_path, N=N)) == cli.EXIT_CONVERGED
    step = direct(N).steps[0]
    rows = table(tmp_path / "factors.csv")
    for role in cli_cccc_n23.FACTOR_ROLES:
        values = np.asarray(getattr(step, role), dtype=float).reshape(len(getattr(step, role)), -1)
        stored = [row for row in rows if row["role"] == role]
        restored = np.zeros(values.shape)
        for row in stored:
            restored[int(row["i"]) - 1, int(row["j"]) - 1] = float(row["value"])
        assert len(stored) == values.size and np.array_equal(restored, values), role
    restored_V = np.zeros(step.V.shape)
    for row in rows:
        if row["role"] == "V":
            restored_V[int(row["i"]) - 1, int(row["j"]) - 1] = float(row["value"])
    G = np.array([[float(r["value"]) for r in rows if r["role"] == "G" and int(r["i"]) == i] for i in range(1, step.G.shape[0] + 1)])
    assert np.array_equal(G @ restored_V.T, direct(N).field()[1:-1, 1:-1])  # the field from the stored factors


def test_the_history_is_the_record_of_the_solver_bit_for_bit(tmp_path):
    assert cli.main(command(tmp_path, half_steps=2)) == 6
    result = direct(half_steps=2)
    rows = table(tmp_path / "history.csv")
    assert [row["s"] for row in rows] == ["1", "2"] and [row["direction"] for row in rows] == ["Y", "X"]
    assert [row["status"] for row in rows] == [cn.HALF_STEP_DONE, cn.STOPPED_RANK_GUARD_UNCERTIFIED]
    first = result.steps[0]
    for name in ("cond_A_before_QR", "cond_A_after_QR", "R_diag_min_abs", "cond_S", "min_eig_DM", "eta", "rho_N", "kappa_rep", "P_hat", "alpha_centre"):
        assert float(rows[0][name]) == getattr(first, name), name
    assert [float(x) for x in rows[0]["A_diag"].split(";")] == np.diag(first.A).tolist()
    assert [float(rows[0][f"sigma_{i}"]) for i in (1, 2, 3)] == list(first.sigma)
    assert rows[0]["V_exact_rank"] == "3" and rows[0]["exact_rank_T"] == "3" and rows[0]["premises_status"] == cn.PREMISES_ESTABLISHED
    assert rows[1]["qr_executed"] == "false" and rows[1]["eta"] == "" and rows[1]["A_diag"] == ""


def test_the_record_of_the_run(earlier):
    d = document(earlier)
    assert d["schema"] == "kkvim.cli.cccc_n23/1" and d["mode"] == "FIXED_HALF_STEPS_RESEARCH"
    assert d["application_status"] == "RESEARCH_ONLY / PROVISIONAL / NOT_QUALIFIED" and d["status"] == cn.COMPLETED_FIXED_HALF_STEPS
    assert d["rank_guard_rule"] == "D-057.1 (вариант В): N = 2, 3 — только s = 1" and d["half_steps_completed"] == 1
    assert d["model"]["boundary"] == "CCCC" and d["model"]["N"] == 3 and d["model"]["decision"] == "D-063" and d["model"]["orientation"] == cn.ORIENTATION
    assert d["input"]["N"] == 3 and d["input"]["n_x"] == 16 and d["D"]["value"] == direct().D
    assert d["numerical_policy"]["eta_max_C"]["value"] == 100 * 2.0**-53
    assert d["rank"]["decision"] == cn.RANK_DECISION and d["rank"]["V_exact_rank"] == 3 and d["rank"]["half_step_2"] is None
    assert "ASSUMED" in d["premises_c3plus"]["P-4"] and d["evidence_scope"] and d["limits"] and d["field"]["role"] == "research"
    provenance = d["provenance"]
    assert SHA.fullmatch(provenance["git_commit"] or "0" * 40) and provenance["module"] == "kkvim.cli_cccc_n23"


def test_nothing_says_that_the_run_converged_or_is_qualified(tmp_path):
    for half_steps in (1, 2):
        out = tmp_path / str(half_steps)
        cli.main(command(out, half_steps=half_steps))
        text = "\n".join(path.read_text(encoding="utf-8") for path in out.iterdir())
        assert "CONVERGED" not in text and '"converged"' not in text and not (out / "field.csv").exists()
        assert not re.search(r'"status": "(PASS|QUALIFIED|ACCEPTED)"', text)


@pytest.mark.parametrize("N", [2, 3])
@pytest.mark.parametrize("half_steps", [2, 5, 8])
def test_s2_is_a_protective_stop_with_the_field_of_s1_as_diagnostic(tmp_path, N, half_steps):
    assert cli.main(command(tmp_path, N=N, half_steps=half_steps)) == 6
    assert names(tmp_path) == ["factors.csv", "field_diagnostic.csv", "history.csv", "result.json"]
    d = document(tmp_path)
    assert d["status"] == cn.STOPPED_RANK_GUARD_UNCERTIFIED and d["field"]["role"] == "diagnostic" and d["rank"]["half_step_2"]["qr_executed"] is False
    rows = table(tmp_path / "field_diagnostic.csv")
    w = direct(N, half_steps).field()
    assert all(float(row["w [m]"]) == float(w[int(row["i"]), int(row["j"])]) for row in rows)
    assert "DIAGNOSTIC" in (tmp_path / "field_diagnostic.csv").read_text(encoding="utf-8").splitlines()[0]


def test_every_status_has_its_exit_code_and_note():
    statuses = {cn.COMPLETED_FIXED_HALF_STEPS, cn.STOPPED_RANK_GUARD_UNCERTIFIED, cn.STOP_BEFORE_QR_S1, cn.NOT_EXECUTED_OUTSIDE_DOMAIN, cn.ERROR_INPUT,
                cn.ERROR_NONFINITE, cn.UNRESOLVED_C3PLUS_PREMISE, cn.PROTECTIVE_STOP_FACTORIZATION, cn.STOPPED_ETA_GUARD, cn.STOP_ZERO_FIELD}
    assert set(cli_cccc_n23.EXIT_CODES) == statuses == set(cli_cccc_n23.STATUS_NOTES)
    assert {cli_cccc_n23.EXIT_CODES[s] for s in statuses - {cn.COMPLETED_FIXED_HALF_STEPS, cn.NOT_EXECUTED_OUTSIDE_DOMAIN, cn.ERROR_INPUT,
                                                              cn.ERROR_NONFINITE}} == {6}
    assert cli_cccc_n23.EXIT_CODES[cn.NOT_EXECUTED_OUTSIDE_DOMAIN] == 2 and cli_cccc_n23.EXIT_CODES[cn.ERROR_INPUT] == 3


@pytest.mark.parametrize("changes, N, code, status", [
    ({}, 1, 2, cn.NOT_EXECUTED_OUTSIDE_DOMAIN), ({}, 4, 2, cn.NOT_EXECUTED_OUTSIDE_DOMAIN), ({"nx": 6}, 2, 2, cn.NOT_EXECUTED_OUTSIDE_DOMAIN),
    ({"b": 6.0}, 3, 2, cn.NOT_EXECUTED_OUTSIDE_DOMAIN), ({"q": 0.0}, 2, 3, cn.ERROR_INPUT), ({"t": -0.01}, 3, 3, cn.ERROR_INPUT),
])
def test_a_refused_input_in_an_empty_folder_writes_only_its_record(tmp_path, changes, N, code, status):
    assert cli.main(command(tmp_path, N=N, **changes)) == code
    assert names(tmp_path) == ["result.json"] and document(tmp_path)["status"] == status and document(tmp_path)["field"] is None


@pytest.mark.parametrize("changes, N, code", [({}, 4, 2), ({"q": 0.0}, 3, 3)])
def test_a_refused_input_keeps_earlier_results(earlier, capsys, changes, N, code):
    before = snapshot(earlier)
    capsys.readouterr()
    assert cli.main(command(earlier, "--overwrite", N=N, **changes)) == code
    assert snapshot(earlier) == before and "не заменён" in capsys.readouterr().err


@pytest.mark.parametrize("name, value, status", [
    ("residual_eta", lambda *a: 101 * cn.U, cn.STOPPED_ETA_GUARD), ("premise_failures", lambda *a: ["P-2: injected"], cn.UNRESOLVED_C3PLUS_PREMISE),
])
def test_a_stop_at_s1_keeps_earlier_results_and_writes_no_field(earlier, tmp_path, monkeypatch, name, value, status):
    monkeypatch.setattr(cn, name, value)
    before = snapshot(earlier)
    assert cli.main(command(earlier, "--overwrite")) == 6 and snapshot(earlier) == before
    empty = tmp_path / "empty"
    assert cli.main(command(empty)) == 6
    assert names(empty) == ["factors.csv", "history.csv", "result.json"] and document(empty)["status"] == status


def test_overwriting_only_with_the_option(earlier, monkeypatch, capsys):
    monkeypatch.setattr(cn, "run_cccc_n23_physical", lambda *a: pytest.fail("computed"))
    before = snapshot(earlier)
    assert cli.main(command(earlier)) == cli.EXIT_USAGE
    assert snapshot(earlier) == before and "--overwrite" in capsys.readouterr().err


def test_a_stop_at_s2_replaces_a_research_field_by_a_diagnostic_one(earlier):
    assert cli.main(command(earlier, "--overwrite", half_steps=2)) == 6
    assert names(earlier) == ["factors.csv", "field_diagnostic.csv", "history.csv", "other.txt", "result.json"]


# --- saving through the transaction of D-062 -----------------------------------------------------------------------------------

def test_a_save_goes_through_the_transaction(earlier, monkeypatch):
    monkeypatch.setattr(cli, "write_results", lambda *a: pytest.fail("the non-atomic save was used"))
    assert cli.main(command(earlier, "--overwrite", N=2)) == cli.EXIT_CONVERGED
    assert document(earlier)["model"]["N"] == 2 and (earlier / "other.txt").read_text(encoding="utf-8") == "kept"


@pytest.mark.parametrize("call", [1, 2, 3, 4])
@pytest.mark.parametrize("error, code, said", [(PermissionError("locked"), cli.EXIT_OUTPUT_ERROR, "locked"), (KeyboardInterrupt(), 130, "прерван")])
def test_an_interrupted_replacement_restores_the_earlier_set(earlier, monkeypatch, capsys, call, error, code, said):
    before = snapshot(earlier)
    with monkeypatch.context() as patched:
        patched.setattr(os, "replace", failing_at(call, error))
        capsys.readouterr()
        assert cli.main(command(earlier, "--overwrite", N=2)) == code
    err = capsys.readouterr().err
    assert snapshot(earlier) == before and said in err and "Traceback" not in err


def test_ctrl_c_during_the_calculation_is_exit_130(earlier, monkeypatch, capsys):
    def interrupt(*args):
        raise KeyboardInterrupt

    monkeypatch.setattr(cn, "run_cccc_n23_physical", interrupt)
    before = snapshot(earlier)
    assert cli.main(command(earlier, "--overwrite")) == 130 and snapshot(earlier) == before and "прерван" in capsys.readouterr().err


def test_a_killed_replacement_is_refused_until_dealt_with(earlier, monkeypatch, capsys):
    before = snapshot(earlier)
    process = subprocess.Popen([sys.executable, "-c", KILLED_AT, "2", *command(earlier, "--overwrite", N=2)],
                               stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    line = process.stdout.readline()
    process.kill()
    process.communicate()
    assert line.strip() == "STOPPED", line
    assert not (earlier / "result.json").exists()
    stage = next(earlier.glob(cli_ssss.STAGE_PREFIX + "*"))
    monkeypatch.setattr(cn, "run_cccc_n23_physical", lambda *a: pytest.fail("computed"))
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


# --- routing (B-9) --------------------------------------------------------------------------------------------------------------

def test_the_command_is_registered_next_to_the_others(capsys):
    with pytest.raises(SystemExit):
        cli.main(["--help"])
    text = capsys.readouterr().out
    assert "run-cccc-n23" in text and "run-cccc-n1" in text and "run-ssss" in text


def test_the_other_cccc_and_ssss_commands_still_run(tmp_path):
    plate = [f"--{name}={value!r}" if isinstance(value, float) else f"--{name}={value}" for name, value in PLATE.items()]
    assert cli.main(["run-cccc-n1", *plate, "--half-steps", "2", "--out", str(tmp_path / "n1")]) == cli.EXIT_CONVERGED
    assert document(tmp_path / "n1")["schema"] == "kkvim.cli.cccc_n1/1"
    assert cli.main(["run-ssss", "--N", "2", *plate, "--half-steps", "2", "--out", str(tmp_path / "ssss")]) in (0, 1, 6)
    assert document(tmp_path / "ssss")["schema"] == "kkvim.cli.ssss/1"


def test_run_ssss_points_to_the_cccc_commands(tmp_path, capsys):
    plate = [f"--{name}={value!r}" if isinstance(value, float) else f"--{name}={value}" for name, value in PLATE.items()]
    assert cli.main(["run-ssss", "--N", "2", "--bc", "CCCC", *plate, "--half-steps", "2", "--out", str(tmp_path / "x")]) == cli.EXIT_USAGE
    err = capsys.readouterr().err
    assert "run-cccc-n1" in err and "run-cccc-n23" in err and "N ≥ 4" in err and not (tmp_path / "x").exists()


def test_the_n1_modules_and_the_reference_are_untouched_by_this_mode():
    source = (cli_cccc_n23.__file__, cn.__file__)
    for path in source:
        text = open(path, encoding="utf-8").read()
        assert "kkvim_reference" not in text and "import scripts" not in text
        for forbidden in ("kkvim.coefficients", "kkvim.linear_solver", "kkvim.coupled", "kkvim.controller", "fourth_difference"):
            assert forbidden not in text, (path, forbidden)
    import kkvim.coupled as kc
    assert kc.N_SOLVE == (2, 3)
