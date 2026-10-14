"""QA tests of P-12 (R3, Navier vs full Levy series, deflection; D-038) - scripts/r3_p12_study.py.

Checked: completeness of the fixed control set of D-038 (10 plates, 41 x 41 grid with edges, 36 near-edge points and
the centre); the normalisation q a^4 / D of the original side a for values and bounds (B_N, B_L by their full
formulas); internal consistency of points.csv, plates.csv and metadata.json; the report blocks against the CSV; the
verdict rule of D-038, item 4; a fresh recomputation against the stored data within the rounding models (BLAS
summation order depends on the CPU, so not bitwise). No threshold is derived from the measured differences.
"""

import csv
import importlib.util
import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "r3_p12_study.py"
_spec = importlib.util.spec_from_file_location("r3_p12_study", SCRIPT)
study = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(study)

OUT = ROOT / "docs" / "results" / "r3" / "p12"
U = 2.0**-53
ZETA3 = 1.2020569031595942853997381  # [СТАНД.]
D038_PLATES = [(1.0, 0.2), (3.0, 1.0), (2.0, 1.0), (3.0, 2.0), (1.0, 1.0), (1.0, 1.4), (2.0, 3.0), (1.0, 2.0),
               (1.0, 3.0), (1.0, 5.0)]


def read(name):
    with open(OUT / name, encoding="utf-8") as f:
        return list(csv.DictReader(f))


@pytest.fixture(scope="module")
def points():
    return read("points.csv")


@pytest.fixture(scope="module")
def plates():
    return read("plates.csv")


@pytest.fixture(scope="module")
def meta():
    return json.loads((OUT / "metadata.json").read_text(encoding="utf-8"))


def f(v):
    return math.inf if v == "inf" else float(v)


def test_parameters_are_those_of_d038(meta):
    assert [(a, b) for _, a, b in study.PLATES] == D038_PLATES
    assert (study.M_N, study.M_L, study.Q, study.D, study.NU, study.N_GRID) == (2559, 1131, 1.0, 1.0, 0.3, 40)
    assert study.EDGE_DISTANCES == (1e-3, 1e-6, 1e-9) and study.TRANSVERSE == (0.25, 0.5, 0.75)
    assert (meta["M_N"], meta["M_L"], meta["grid"], meta["nu"]) == (2559, 1131, "41x41", 0.3)
    assert [tuple(p[1:]) for p in meta["plates"]] == D038_PLATES


def test_control_set_is_complete(points):
    assert len(points) == 10 * (41 * 41 + 36 + 1) == 17180
    for name, a, b in study.PLATES:
        rows = [r for r in points if r["plate"] == name]
        grid = [r for r in rows if r["kind"] == "grid"]
        assert {(int(r["i"]), int(r["j"])) for r in grid} == {(i, j) for i in range(41) for j in range(41)}
        for r in grid:  # x_i = i a / 40, y_j = j b / 40 (as computed by the Navier grid)
            assert float(r["x"]) == a * int(r["i"]) / 40 and float(r["y"]) == b * int(r["j"]) / 40
        edge = [r for r in rows if r["kind"] == "edge"]
        expected = study.edge_points(a, b)[:-1]
        assert len(edge) == 36 == len({r["label"] for r in edge})
        assert [(r["label"], float(r["x"]), float(r["y"])) for r in edge] == expected
        x_near = {d * a for d in (1e-3, 1e-6, 1e-9)} | {a - d * a for d in (1e-3, 1e-6, 1e-9)}
        y_near = {d * b for d in (1e-3, 1e-6, 1e-9)} | {b - d * b for d in (1e-3, 1e-6, 1e-9)}
        assert sum(float(r["x"]) in x_near for r in edge) == 18 and sum(float(r["y"]) in y_near for r in edge) == 18
        centre = [r for r in rows if r["kind"] == "centre"]
        assert len(centre) == 1 and (float(centre[0]["x"]), float(centre[0]["y"])) == (a / 2, b / 2)


def test_bounds_are_in_the_common_normalisation(points):
    """B_N = (16/pi^6)(7 zeta(3)/8) r^2/(8 M_N^2), B_L = (4/pi^5)/(8 M_L^4) (a_L/a)^4, both in q a^4 / D."""
    for name, a, b in study.PLATES:
        r = next(p for p in points if p["plate"] == name)
        BN = 16 / math.pi**6 * (7 * ZETA3 / 8) * (b / a) ** 2 / (8 * 2559**2)
        BL = 4 / math.pi**5 / (8 * 1131**4) * (min(a, b) / a) ** 4
        assert abs(float(r["B_N"]) - BN) <= 32 * U * BN
        assert abs(float(r["B_L"]) - BL) <= 32 * U * BL
        assert float(r["R_L"]) > 0 and float(r["R_N"]) > 0


def test_values_are_in_the_common_normalisation(points):
    """Centre values are w_bar in q a^4 / D of the original a: turned plates give w(1/r) = r^-4 w(r) for both series."""
    centre = {r["plate"]: r for r in points if r["kind"] == "centre"}
    for small, large, k in (("1/3", "3", 3.0), ("1/2", "2", 2.0), ("2/3", "1,5", 1.5)):
        for col, model in (("w_N", "R_N"), ("w_L", "R_L")):
            lhs, rhs = float(centre[small][col]), float(centre[large][col]) / k**4
            assert abs(lhs - rhs) <= float(centre[small][model]) + float(centre[large][model]) / k**4
    assert abs(float(centre["1"]["w_N"]) - 0.0040623527) <= 5e-11  # square, L-107 (131)/(140) scale check


def test_point_rows_are_consistent(points):
    for r in points:
        wn, wl, diff = float(r["w_N"]), float(r["w_L"]), float(r["diff"])
        bound = float(r["B_N"]) + float(r["B_L"]) + float(r["R_N"]) + float(r["R_L"])
        assert diff == wn - wl and float(r["abs_diff"]) == abs(diff)
        assert float(r["bound"]) == bound
        assert f(r["margin"]) == (math.inf if diff == 0.0 else bound / abs(diff))
        assert r["ok"] == str(abs(diff) <= bound)


def test_plate_rows_and_metadata_are_consistent(points, plates, meta):
    assert [p["plate"] for p in plates] == [n for n, _, _ in study.PLATES]
    for p in plates:
        rows = [r for r in points if r["plate"] == p["plate"]]
        assert int(p["n_points"]) == len(rows) == 1718
        assert (int(p["n_grid"]), int(p["n_edge"]), int(p["n_centre"])) == (1681, 36, 1)
        assert int(p["n_ok"]) == sum(r["ok"] == "True" for r in rows)
        assert int(p["n_exceed"]) == len(rows) - int(p["n_ok"])
        worst = max(rows, key=lambda r: float(r["abs_diff"]))
        tight = min(rows, key=lambda r: f(r["margin"]))
        assert float(p["max_abs_diff"]) == float(worst["abs_diff"])
        assert (float(p["x_max_diff"]), float(p["y_max_diff"])) == (float(worst["x"]), float(worst["y"]))
        assert f(p["min_margin"]) == f(tight["margin"])
        assert (float(p["x_min_margin"]), float(p["y_min_margin"])) == (float(tight["x"]), float(tight["y"]))
        assert p["verdict"] == ("PASS" if int(p["n_exceed"]) == 0 else p["verdict"])
    assert meta["n_comparisons"] == len(points)
    assert meta["n_exceed"] == sum(r["ok"] == "False" for r in points)
    assert meta["global_max_abs_diff"] == max(float(r["abs_diff"]) for r in points)
    assert meta["global_min_margin"] == min(f(r["margin"]) for r in points)
    verdicts = [p["verdict"] for p in plates]
    assert meta["verdict"] == ("FAIL" if "FAIL" in verdicts else "UNRESOLVED" if "UNRESOLVED" in verdicts else "PASS")


def test_every_exceedance_has_a_high_precision_recomputation(points):
    exceed = [r for r in points if r["ok"] == "False"]
    if not exceed:
        return
    hp = read("hp_rechecks.csv")
    done = {(h["plate"], h["label"]) for h in hp if h["role"] == "exceedance"}
    assert {(r["plate"], r["label"]) for r in exceed} <= done
    assert all(int(h["dps"]) >= 40 for h in hp)


def test_hp_rechecks_if_present_are_consistent(points):
    path = OUT / "hp_rechecks.csv"
    if not path.exists():
        pytest.skip("optional diagnostic not stored")
    rows = {(r["plate"], r["label"]): r for r in points}
    for h in read("hp_rechecks.csv"):
        r = rows[(h["plate"], h["label"])]
        assert int(h["dps"]) >= 40
        assert h["truncation_ok"] == str(abs(float(h["diff_hp"])) <= float(h["B_N_plus_B_L"]))
        # the double values agree with their high-precision recomputation within the rounding models
        assert float(h["w_N_double_err"]) <= float(r["R_N"]) and float(h["w_L_double_err"]) <= float(r["R_L"])


def test_verdict_rule_of_d038():
    ok_hp = {"truncation_ok": True}
    bad_hp = {"truncation_ok": False}
    assert study.plate_verdict([], [])[0] == "PASS"
    assert study.plate_verdict([{}], [None])[0] == "UNRESOLVED"  # no recomputation - never explained by rounding
    assert study.plate_verdict([{}, {}], [ok_hp, bad_hp])[0] == "FAIL"
    assert study.plate_verdict([{}], [ok_hp])[0] == "UNRESOLVED"


def test_report_blocks_are_generated_from_the_stored_results():
    generated = study.report_blocks(OUT)
    in_report = study.report_block_contents(study.REPORT)
    assert set(generated) == set(in_report) == {"plates", "summary"}
    for name, block in generated.items():
        assert in_report[name] == block, name


def test_fresh_recomputation_agrees_with_the_stored_data(points):
    """Recompute every plate: values within the rounding models of the stored ones, same ok flags."""
    stored = {(r["plate"], r["label"]): r for r in points}
    for name, a, b in study.PLATES:
        rows, _ = study.compute_plate(name, a, b)
        for r in rows:
            s = stored[(name, r["label"])]
            assert abs(r["w_N"] - float(s["w_N"])) <= 2 * r["R_N"]
            assert abs(r["w_L"] - float(s["w_L"])) <= 2 * r["R_L"]
            assert str(r["ok"]) == s["ok"]
