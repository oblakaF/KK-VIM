"""R6 CLI: ``python -m kkvim moments DIR`` and ``--moments`` of the run commands (cli_moments; review 8.4).

Checked: the files and columns of the contract; the role rule (research / diagnostic); the original files are never
changed; existing moment files are not replaced without --overwrite; a run without a field gets no file and exit 7;
the exit code of the run is kept (protective stop 6); the moments of the CSV equal kkvim.moments of the saved field
(the CSV holds repr of the solver values, review U-3); SSSS edges M_x = M_y = 0 on a real field; the separate status.
"""

import csv
import hashlib
import json

import numpy as np
import pytest

from kkvim import cli, cli_moments
from kkvim import moments as mo

MAT = ["--t", "0.01", "--E", "2.1e11", "--nu", "0.3"]


def digest(folder):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.iterdir())}


def rows(path):
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if not line.startswith("#")]
    return list(csv.DictReader(lines))


def run(argv):
    return cli.main([str(a) for a in argv])


@pytest.fixture(scope="module")
def ssss_folder(tmp_path_factory):
    out = tmp_path_factory.mktemp("r3") / "run"
    code = run(["run", "--a", "2.0", "--b", "1.0", *MAT, "--q", "5000", "--nx", "16", "--ny", "8", "--out", out, "--moments"])
    return out, code


def test_run_with_moments_writes_the_contract_files(ssss_folder):
    out, code = ssss_folder
    assert code == 0
    names = {p.name for p in out.iterdir()}
    assert {"result.json", "field.csv", "history.csv", "moments_research.csv", "moments.json"} <= names
    table = rows(out / "moments_research.csv")
    assert list(table[0].keys()) == list(cli_moments.COLUMNS)
    assert len(table) == 17 * 9
    doc = json.loads((out / "moments.json").read_text(encoding="utf-8"))
    assert doc["status"] == "RESEARCH / NOT_QUALIFIED" and doc["role"] == "research"
    assert doc["source_field"]["displacement_application_status"] == "ACCEPTED (D-043)"  # kept, not transferred
    assert doc["operators"]["cc_edge_rule"].startswith("A (2 f1 / h^2)")
    assert "rule B" in doc["not_computed"] and not any("B" in c.split()[0] for c in cli_moments.COLUMNS if "_B" in c)


def test_csv_moments_equal_kkvim_moments_of_the_saved_field(ssss_folder):
    out, _ = ssss_folder
    field = rows(out / "field.csv")
    W = np.zeros((17, 9))
    for r in field:
        W[int(r["i"]), int(r["j"])] = float(r["w [m]"])
    result = json.loads((out / "result.json").read_text(encoding="utf-8"))
    D = result["D"]["value"]
    M = mo.plate_moments(W, 2.0, 1.0, D, 0.3, "SS")
    table = rows(out / "moments_research.csv")
    for r in table:
        i, j = int(r["i"]), int(r["j"])
        assert float(r["Mx [N]"]) == M.Mx[i, j] and float(r["Mxy [N]"]) == M.Mxy[i, j]  # bit for bit (repr)
        assert float(r["sigma_x_bottom [Pa]"]) == pytest.approx(6 * M.Mx[i, j] / 0.01**2, rel=1e-15)
        assert float(r["sigma_x_top [Pa]"]) == -float(r["sigma_x_bottom [Pa]"])
        if i in (0, 16) or j in (0, 8):
            assert float(r["Mx [N]"]) == 0.0 and float(r["My [N]"]) == 0.0  # PM-3 on the solver field (SSSS)


def test_moments_command_never_changes_the_run_and_refuses_to_replace(ssss_folder, tmp_path):
    out, _ = ssss_folder
    before = {k: v for k, v in digest(out).items() if not k.startswith("moments")}
    assert run(["moments", out]) == cli_moments.EXIT_MOMENTS_ERROR  # files exist: no --overwrite, nothing replaced
    assert run(["moments", out, "--overwrite", "--z", "0.0025"]) == 0
    after = {k: v for k, v in digest(out).items() if not k.startswith("moments")}
    assert after == before
    table = rows(out / "moments_research.csv")
    assert "sigma_x_z [Pa]" in table[0]
    k = next(r for r in table if r["i"] == "8" and r["j"] == "4")
    assert float(k["sigma_x_z [Pa]"]) == pytest.approx(0.5 * float(k["sigma_x_bottom [Pa]"]), rel=1e-15)


def test_protective_stop_keeps_its_exit_code_and_gets_diagnostic_moments(tmp_path):
    out = tmp_path / "c46"
    code = run(["run-cccc-n46", "--N", "6", "--a", "5.0", "--b", "1.0", *MAT, "--q", "1000", "--nx", "32", "--ny", "13",
                "--half-steps", "2", "--out", out, "--moments"])
    assert code == 6
    assert (out / "moments_diagnostic.csv").is_file() and not (out / "moments_research.csv").exists()
    doc = json.loads((out / "moments.json").read_text(encoding="utf-8"))
    assert doc["role"] == "diagnostic" and doc["source_field"]["solver_status"].startswith("STOPPED")
    assert (out / "moments_diagnostic.csv").read_text(encoding="utf-8").splitlines()[1].startswith("# DIAGNOSTIC")


def test_cccc_research_moments_satisfy_the_clamped_edge_identities(tmp_path):
    out = tmp_path / "c1"
    assert run(["run-cccc-n1", "--a", "1.2", "--b", "1.0", *MAT, "--q", "1000", "--nx", "12", "--ny", "10",
                "--half-steps", "4", "--out", out, "--moments"]) == 0
    table = rows(out / "moments_research.csv")
    for r in table:
        i, j = int(r["i"]), int(r["j"])
        if i in (0, 12) or j in (0, 10):
            assert float(r["Mxy [N]"]) == 0.0
        if (i in (0, 12)) and (j in (0, 10)):
            assert float(r["Mx [N]"]) == 0.0 and float(r["My [N]"]) == 0.0
    doc = json.loads((out / "moments.json").read_text(encoding="utf-8"))
    assert doc["bc"] == "CC" and doc["source_field"]["displacement_application_status"].startswith("RESEARCH_ONLY")


def test_folder_without_a_field_gets_no_moment_file(tmp_path):
    folder = tmp_path / "nofield"
    folder.mkdir()
    (folder / "result.json").write_text(json.dumps({"model": {"supports": "SSSS"}, "status": "ERROR_INPUT", "field": None,
                                                    "input": {}, "D": None}), encoding="utf-8")
    assert run(["moments", folder]) == cli_moments.EXIT_MOMENTS_ERROR
    assert sorted(p.name for p in folder.iterdir()) == ["result.json"]


def test_new_run_with_overwrite_removes_stale_moment_files(tmp_path):
    out = tmp_path / "again"
    assert run(["run-cccc-n1", "--a", "1.0", "--b", "1.0", *MAT, "--q", "1000", "--nx", "8", "--ny", "8",
                "--half-steps", "2", "--out", out, "--moments"]) == 0
    assert run(["run-cccc-n1", "--a", "1.0", "--b", "1.0", *MAT, "--q", "1000", "--nx", "8", "--ny", "8",
                "--half-steps", "2", "--out", out]) == cli.EXIT_USAGE  # moments files are results of the folder
    assert run(["run-cccc-n1", "--a", "1.0", "--b", "1.0", *MAT, "--q", "1000", "--nx", "8", "--ny", "8",
                "--half-steps", "3", "--out", out, "--overwrite"]) == 0
    assert not (out / "moments.json").exists() and not (out / "moments_research.csv").exists()
