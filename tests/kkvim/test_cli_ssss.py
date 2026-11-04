"""``python -m kkvim run-ssss``: the user-facing command for SSSS, explicit N = 1..6 (``kkvim.cli_ssss``; D-060).

The tests run the command itself and compare what it wrote with the direct call of the route:
- N = 1: the files of ``python -m kkvim run`` and of the R3 API;
- N = 2-3: ``run_coupled`` bit for bit, its statuses unchanged;
- N = 4-6: labels, refusals and safeguards here; agreement with the study and with the independent control in
  tests/cross/test_cli_ssss_research.py.
Values are compared bit for bit or exactly: this file introduces no numerical tolerance of the method.
"""

import csv
import json
import os
import re
import subprocess
import sys

import numpy as np
import pytest

import kkvim.coupled as kc
import kkvim.coupled_run as cr
import kkvim.rectangle_run as rr
from kkvim import cli, cli_ssss
from kkvim.coefficients import ERROR_INPUT
from kkvim.controller import CONVERGED, MAX_HALF_STEPS
from kkvim.physical import bending_stiffness
from kkvim.research_ssss import S_STUDY, domain_refusal

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
    r3 = document(old)  # every key of the R3 document is kept, apart from what names this command
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
    # the written field has N terms: a rounding-level gap right after sigma_N of the N-term field (a smoke check)
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
    assert not out.exists() and "D-049.7" in capsys.readouterr().err


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


def test_overwrite_leaves_only_the_files_of_the_new_run(tmp_path):
    out = tmp_path / "out"
    assert cli.main(command(out, 1)) == cli.EXIT_CONVERGED
    assert (out / "field.csv").exists()
    (out / "other.txt").write_text("kept", encoding="utf-8")
    assert cli.main(command(out, 5, "--overwrite", half_steps=2)) == cli.EXIT_CONVERGED
    assert names(out) == ["field_research.csv", "history.csv", "other.txt", "result.json"]
    assert cli.main(command(out, 6, "--overwrite", half_steps=2, b=0.2)) == cli_ssss.EXIT_GUARD_STOP
    assert names(out) == ["field_diagnostic.csv", "history.csv", "other.txt", "result.json"]
    assert cli.main(command(out, 5, "--overwrite", half_steps=2, b=1.2)) == cli.EXIT_USAGE
    assert names(out) == ["history.csv", "other.txt", "result.json"]
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


@pytest.mark.parametrize("N, half_steps", [(1, None), (3, 8), (6, 8)])
def test_python_m_kkvim_runs_the_documented_examples(tmp_path, N, half_steps):
    out = tmp_path / "out"
    done = subprocess.run([sys.executable, "-m", "kkvim", *command(out, N, half_steps=half_steps)], capture_output=True,
                          encoding="utf-8", check=False)
    assert done.returncode == cli.EXIT_CONVERGED, done.stderr
    assert f"SSSS, N = {N}" in done.stdout and document(out)["N"] == N
