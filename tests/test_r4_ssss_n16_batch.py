"""scripts/r4_ssss_n16_batch.py (R4-SSSS-N16-BATCH-01): the mode of every N, the saved run-ssss files and the summary copied from them,
field roles, refusals kept as refusals, a populated folder refused. Builder test of the engineering command, not a QA of the solvers.

Two real batches: 40 x 40 (inside every route; N = 1 CONVERGED by R3, N >= 2 the 8 fixed half-steps) and 24 x 24 (N = 4-6 outside the
studied grids of D-056.7: NOT_EXECUTED_OUTSIDE_DOMAIN, nothing computed, no field)."""

import csv
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "r4_ssss_n16_batch.py"
_spec = importlib.util.spec_from_file_location("r4_ssss_n16_batch", SCRIPT)
batch = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(batch)

PLATE = {"a": "1", "b": "1", "t": "0.01", "E": "2.1e11", "nu": "0.3", "q": "1000", "nx": "40", "ny": "40"}


def _args(plate):
    return [x for name, value in plate.items() for x in (f"--{name}", value)]


def _run(out, plate):
    done = subprocess.run([sys.executable, str(SCRIPT), *_args(plate), "--out", str(out)], cwd=ROOT, capture_output=True, text=True,
                          encoding="utf-8", check=False)
    return done, json.loads((out / "summary.json").read_text(encoding="utf-8"))


def test_every_N_gets_its_own_mode_and_the_same_input(tmp_path):
    assert batch.options_of(1) == []
    assert [batch.options_of(N) for N in (2, 3)] == [["--half-steps", "8", "--fixed-half-steps"]] * 2
    assert [batch.options_of(N) for N in (4, 5, 6)] == [["--half-steps", "8", "--research-only"]] * 3
    for bad in (0, 7):
        with pytest.raises(ValueError):
            batch.options_of(bad)
    for N in batch.N_VALUES:
        argv = batch.command_of(N, PLATE, tmp_path / f"N{N}")
        assert argv[:3] == ["run-ssss", "--N", str(N)] and argv[3:19] == _args(PLATE)
        assert argv[19:-2] == batch.options_of(N) and argv[-2:] == ["--out", str(tmp_path / f"N{N}")]


def test_a_populated_output_folder_is_refused_and_left_unchanged(tmp_path, capsys):
    (tmp_path / "old.txt").write_text("earlier calculation", encoding="utf-8")
    assert batch.main([*_args(PLATE), "--out", str(tmp_path)]) == batch.EXIT_USAGE
    assert sorted(p.name for p in tmp_path.iterdir()) == ["old.txt"]
    assert "not overwritten" in capsys.readouterr().err


@pytest.fixture(scope="module")
def inside(tmp_path_factory):
    out = tmp_path_factory.mktemp("ssss16") / "batch"
    return (out, *_run(out, PLATE))


def test_the_six_runs_are_saved_unchanged_and_summarised_from_their_own_result_json(inside):
    out, done, summary = inside
    assert done.returncode == batch.EXIT_DONE, done.stdout + done.stderr
    for row in summary["rows"]:
        folder = out / f"N{row['N']}"
        document = json.loads((folder / "result.json").read_text(encoding="utf-8"))
        assert document["N"] == row["N"] and document["program"]["argv"][:3] == ["run-ssss", "--N", str(row["N"])]
        assert row["mode"] == document["solver_path"]["mode"] == batch.MODES[row["N"]] and row["problem"] is None
        assert row["files"] == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.iterdir())}
        field, applicability = document["field"], document["scientific_applicability"]
        assert (row["status"], row["reason"], row["half_steps_completed"], row["field_file"], row["field_role"]) == (
            document["status"], document["reason"], document["half_steps_completed"], field["file"], field["role"])
        assert (row["w_bar_centre"], row["mark"], row["accepted_solver"], row["research_only"]) == (
            field["w_bar_centre"], applicability["mark"], applicability["accepted_solver"], applicability["research_only"])
        assert row["w_max_m"] == (field.get("w_max") or {}).get("value") and row["provenance"] == document["provenance"]


def test_N1_is_R3_with_its_rule_and_N_ge_2_is_research_never_converged(inside):
    _, _, summary = inside
    rows = {row["N"]: row for row in summary["rows"]}
    r1 = rows[1]
    assert (r1["status"], r1["field_role"], r1["field_file"], r1["accepted_solver"], r1["accepted_by"]) == (
        "CONVERGED", "converged", "field.csv", True, "D-043")
    assert r1["half_steps_requested"] is None and r1["stopping_parameters"]["s_max"] == 100
    assert r1["w_max_m"] is None and r1["w_max_note"].startswith("not stated")  # R3 writes no max|w|: not recomputed
    for N in (2, 3, 4, 5, 6):
        row = rows[N]
        assert (row["status"], row["half_steps_requested"], row["fixed_half_steps"], row["half_steps_completed"]) == (
            "RESEARCH_LIMIT_REACHED", 8, True, 8)
        assert (row["field_role"], row["field_file"], row["converged"], row["accepted_solver"]) == (
            "research", "field_research.csv", False, False)
        assert row["mark"] == "PROVISIONAL / NOT_QUALIFIED" and row["research_only"] is (N >= 4)
        assert "field.csv" not in row["files"] and row["w_max_m"] > 0


def test_the_csv_table_repeats_the_json_rows(inside):
    out, _, summary = inside
    lines = [line for line in (out / "summary.csv").read_text(encoding="utf-8").splitlines() if not line.startswith("#")]
    table = list(csv.DictReader(lines))
    for r, row in zip(table, summary["rows"], strict=True):
        assert (int(r["N"]), r["status"], r["field_role"], r["mode"]) == (row["N"], row["status"], row["field_role"], row["mode"])
        assert float(r["w_bar_centre"]) == row["w_bar_centre"] and r["result_json_sha256"] == row["files"]["result.json"]
        assert (r["w_max_m"] == "") if row["w_max_m"] is None else float(r["w_max_m"]) == row["w_max_m"]


def test_a_batch_run_is_the_same_as_a_direct_run_of_run_ssss(inside, tmp_path):
    out, _, _ = inside
    direct = tmp_path / "direct"
    subprocess.run([sys.executable, "-m", "kkvim", *batch.command_of(5, PLATE, direct)], cwd=ROOT, capture_output=True, check=False)
    for name in ("history.csv", "field_research.csv"):
        assert (direct / name).read_bytes() == (out / "N5" / name).read_bytes(), name


def test_outside_the_studied_domain_N4_N6_are_refusals_without_a_field(tmp_path):
    done, summary = _run(tmp_path / "batch", {**PLATE, "nx": "24", "ny": "24"})
    assert done.returncode == batch.EXIT_STOPPED, done.stdout + done.stderr
    rows = {row["N"]: row for row in summary["rows"]}
    for N in (4, 5, 6):
        row = rows[N]
        assert (row["status"], row["exit_code"], row["field_file"], row["field_available"]) == (
            "NOT_EXECUTED_OUTSIDE_DOMAIN", 2, None, False), N
        assert row["w_bar_centre"] is None and row["w_max_m"] is None and row["problem"] is None
    assert "stopped or refused for N = [4, 5, 6]" in summary["batch_outcome"]
