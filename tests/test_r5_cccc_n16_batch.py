"""scripts/r5_cccc_n16_batch.py (R5-CCCC-N16-BATCH-01): dispatch of N = 1...6 to the existing commands, the saved files and the summary,
the statuses of protective stops kept as stops, a populated folder refused. Builder test of the engineering command, not a QA of the
solvers. One real batch on 34 x 34: N = 1 does s = 1, 2; N = 2...6 stop before s = 2 (the grid is outside n_x, n_y <= 32, D-065.5, D-068)."""

import csv
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "r5_cccc_n16_batch.py"
_spec = importlib.util.spec_from_file_location("r5_cccc_n16_batch", SCRIPT)
batch = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(batch)

PLATE = {"a": "1", "b": "1", "t": "0.01", "E": "2.1e11", "nu": "0.3", "q": "1000", "nx": "34", "ny": "34"}


def test_every_N_goes_to_its_own_command_with_the_same_input_and_two_half_steps(tmp_path):
    assert batch.route_of(1) == ["run-cccc-n1"]
    assert [batch.route_of(N) for N in (2, 3)] == [["run-cccc-n23", "--N", "2"], ["run-cccc-n23", "--N", "3"]]
    assert [batch.route_of(N) for N in (4, 5, 6)] == [["run-cccc-n46", "--N", str(N)] for N in (4, 5, 6)]
    for bad in (0, 7):
        with pytest.raises(ValueError):
            batch.route_of(bad)
    tails = {tuple(batch.command_of(N, PLATE, tmp_path / f"N{N}")[len(batch.route_of(N)):-2]) for N in batch.N_VALUES}
    assert tails == {("--a", "1", "--b", "1", "--t", "0.01", "--E", "2.1e11", "--nu", "0.3", "--q", "1000", "--nx", "34", "--ny", "34",
                      "--half-steps", "2")}


def test_a_populated_output_folder_is_refused_and_left_unchanged(tmp_path, capsys):
    (tmp_path / "old.txt").write_text("earlier calculation", encoding="utf-8")
    args = [x for name, value in PLATE.items() for x in (f"--{name}", value)]
    assert batch.main([*args, "--out", str(tmp_path)]) == batch.EXIT_USAGE
    assert sorted(p.name for p in tmp_path.iterdir()) == ["old.txt"]
    assert "not overwritten" in capsys.readouterr().err


@pytest.fixture(scope="module")
def batch_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("n16") / "batch"
    args = [x for name, value in PLATE.items() for x in (f"--{name}", value)]
    done = subprocess.run([sys.executable, str(SCRIPT), *args, "--out", str(out)], cwd=ROOT, capture_output=True, text=True,
                          encoding="utf-8", check=False)
    return out, done, json.loads((out / "summary.json").read_text(encoding="utf-8"))


def test_the_six_results_are_saved_unchanged_and_summarised_from_their_own_result_json(batch_run):
    out, done, summary = batch_run
    assert done.returncode == batch.EXIT_PROTECTIVE_STOP, done.stderr
    assert [row["N"] for row in summary["rows"]] == [1, 2, 3, 4, 5, 6]
    for row in summary["rows"]:
        folder = out / f"N{row['N']}"
        document = json.loads((folder / "result.json").read_text(encoding="utf-8"))
        assert document["model"]["N"] == row["N"] and document["program"]["argv"][0] == row["route"] == batch.route_of(row["N"])[0]
        assert row["files"] == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.iterdir())}
        field = document["field"]
        assert (row["status"], row["reason"], row["half_steps_requested"], row["half_steps_completed"]) == (
            document["status"], document["reason"], document["half_steps_requested"], document["half_steps_completed"])
        assert (row["field_role"], row["field_file"], row["alpha_centre"], row["w_max_abs"], row["w_max_unit"]) == (
            field["role"], field["file"], field["alpha_centre"], field["w_max"]["value"], "m")
        assert row["application_status"] == document["application_status"] == batch.MARKING and row["problem"] is None
        assert row["decision"] == document["model"]["decision"] and row["provenance"] == document["provenance"]


def test_protective_stops_stay_stops_with_their_diagnostic_field(batch_run):
    out, done, summary = batch_run
    rows = {row["N"]: row for row in summary["rows"]}
    assert (rows[1]["status"], rows[1]["exit_code"], rows[1]["half_steps_completed"], rows[1]["field_role"]) == (
        "COMPLETED_FIXED_HALF_STEPS", 0, 2, "research")
    for N in (2, 3, 4, 5, 6):
        row = rows[N]
        assert (row["status"], row["exit_code"], row["half_steps_completed"], row["field_role"], row["field_file"]) == (
            "STOPPED_RANK_GUARD_UNCERTIFIED", 6, 1, "diagnostic", "field_diagnostic.csv"), N
        assert "34 × 34" in row["reason"] and "field_research.csv" not in row["files"]
    assert "protective stops for N = [2, 3, 4, 5, 6]" in summary["batch_outcome"]
    assert all(row["status"] != "CONVERGED" for row in summary["rows"])
    assert summary["convergence"] == "NOT ASSESSED (no CONVERGED status)" and summary["D1_START"] == "BLOCKED"


def test_the_csv_table_repeats_the_json_rows_exactly(batch_run):
    out, _, summary = batch_run
    lines = [line for line in (out / "summary.csv").read_text(encoding="utf-8").splitlines() if not line.startswith("#")]
    table = list(csv.DictReader(lines))
    assert [int(r["N"]) for r in table] == [1, 2, 3, 4, 5, 6]
    for r, row in zip(table, summary["rows"], strict=True):
        assert r["status"] == row["status"] and r["field_role"] == row["field_role"]
        assert float(r["alpha_centre"]) == row["alpha_centre"] and float(r["w_max_abs"]) == row["w_max_abs"]
        assert r["result_json_sha256"] == row["files"]["result.json"] and r["field_sha256"] == row["files"][row["field_file"]]


def test_a_batch_row_is_the_same_as_a_direct_run_of_the_command(batch_run, tmp_path):
    out, _, summary = batch_run
    direct = tmp_path / "direct"
    argv = batch.command_of(3, PLATE, direct)
    subprocess.run([sys.executable, "-m", "kkvim", *argv], cwd=ROOT, capture_output=True, check=False)
    for name in ("history.csv", "factors.csv", "field_diagnostic.csv"):
        assert (direct / name).read_bytes() == (out / "N3" / name).read_bytes(), name
