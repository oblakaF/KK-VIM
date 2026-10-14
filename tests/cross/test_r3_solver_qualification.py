"""QA tests of steps 1-2 of the R3 solver qualification (R3_NUMERICAL_SPEC 12) - scripts/r3_solver_qualification.py,
scripts/r3_a10b_hp.py.

Checked: the fixed pairs of table 7.2 and the admitted levels of table 7.3 follow the rules of 6.3 and 7.1 (kappa_s of the
qualified Parseval recursion, zeta_A4, zeta_A5, classes Q/A/G, P-C-min, Q-Q for Richardson); the stored tables are
complete and their verdicts follow the fixed criteria (NUMERICAL_SPEC 18.1, D-039, RV-3); a fresh recomputation of small
cases; the report blocks against the CSV. Thresholds are those of the specification only.
"""

import csv
import importlib.util
import json
import math
from pathlib import Path

import pytest

from kkvim_reference.parseval_recursion import ParsevalRecursion

ROOT = Path(__file__).resolve().parents[2]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


qual = _load("r3_solver_qualification")
OUT = qual.OUT
A10B = ROOT / "docs" / "results" / "r3" / "solver" / "a10b"


def read(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


# --- tables 7.2 and 7.3 against the rules of 6.3 and 7.1 ---------------------------------------------------------------

def kappas(b):
    P = ParsevalRecursion(a=1.0, b=b, q=1.0, D=1.0, s_max=3)
    return [(P.coefficients(s).C / P.coefficients(s).A) ** 0.25 for s in (1, 2, 3)]


SQUARE = kappas(1.0)
ZETA_R2 = {"A4": (math.pi / 80, math.pi / 20),
           "A5": (max(math.pi, *SQUARE) / 80, max(math.pi, *SQUARE) / 20)}


def zeta(check, b, grid, k):
    hx, hy = 1.0 / grid[0], b / grid[1]
    if check == "A4":
        return math.pi * max(hx, hy)
    return max(math.pi * hx, k[0] * hy, k[1] * hx, k[2] * hy)


def level_class(check, b, grid, k):
    z = zeta(check, b, grid, k)
    q80, q20 = ZETA_R2[check]
    if min(grid) < 20:
        return "G"
    if z <= q80 * (1 + 1e-9):
        return "Q"
    return "A" if z <= q20 * (1 + 1e-9) else "G"


def test_kappa_and_reference_zeta_values_of_6_3():
    assert abs(ZETA_R2["A5"][0] - 0.039420) < 5e-7 and abs(ZETA_R2["A5"][1] - 0.157681) < 5e-7
    assert abs(kappas(0.2)[1] - 15.7144) < 5e-4 and abs(kappas(5.0)[1] - 1.0182) < 5e-4


@pytest.mark.parametrize("key", list(qual.PAIRS), ids=lambda k: f"{k[0]}-{k[1]}")
def test_pairs_and_levels_follow_the_rules_of_7_1(key):
    label, family = key
    if label == "1":  # r = 1: the pairs of R2 (D-032, item 7), all levels admitted (7.3)
        assert all(p == qual.I_LOW for p in qual.PAIRS[key])
        return
    b = float(qual.RATIO[label])
    k = kappas(b)
    chain = qual.CHAINS[key]
    expected = []
    for check in ("A4", "A5"):
        cls = [level_class(check, b, g, k) for g in chain]
        admitted = [g for g, c in zip(chain, cls) if c in "AQ"]
        qq = [(g, g2) for (g, c), (g2, c2) in zip(zip(chain, cls), list(zip(chain, cls))[1:]) if c == c2 == "Q"]
        if len(qq) >= 2:
            p_pairs = tuple(qq[:2])
        else:  # the finest admitted triple of consecutive levels
            idx = [i for i in range(len(chain) - 2) if all(c in "AQ" for c in cls[i:i + 3])]
            i = idx[-1]
            p_pairs = ((chain[i], chain[i + 1]), (chain[i + 1], chain[i + 2]))
        rich = tuple(qq[:2])
        expected.append((p_pairs, rich, tuple(admitted)))
    (p4, r4, adm4), (p5, r5, adm5) = expected
    assert qual.PAIRS[key] == (p4, p5, r4, r5), key
    assert qual.MONOTONE[key] == (adm4, adm5), key


# --- stored results --------------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def meta():
    return json.loads((OUT / "metadata.json").read_text(encoding="utf-8"))


def test_metadata(meta):
    assert len(meta["commit"]) == 40 and meta["dirty"] is False
    assert {"numpy", "scipy", "kkvim"} <= set(meta["versions"])
    assert meta["tolerances"] == qual.TOL


def test_every_mandatory_pair_of_table_7_2_is_evaluated():
    rows = read(OUT / "pairs.csv")
    for key, (p4, p5, r4, r5) in qual.PAIRS.items():
        label, family = key
        got = {(r["check"], r["pair"]) for r in rows if (r["r"], r["family"]) == key and r["mandatory"] == "True"}
        for check, pairs in (("A-4a p", p4), ("A-5a p", p5), ("A-5v p", p5), ("A-4b Richardson", r4), ("A-5b Richardson", r5)):
            want = {(check, f"{g}->{g2}") for g, g2 in pairs} if pairs else {(check, "—")} if "Richardson" in check else set()
            assert want <= got, (key, check)


def test_pair_verdicts_follow_the_criteria():
    for r in read(OUT / "pairs.csv"):
        if r["pair"] == "—":
            assert r["verdict"].startswith("UNRESOLVED")
            continue
        verdict = r["verdict"].removeprefix("DIAGNOSTIC (").rstrip(")")
        p = float(r["p"])
        rv3 = float(r["rv3_ratio"]) if r["rv3_ratio"] else None
        if r["check"].endswith(" p"):
            want = "UNRESOLVED (RV-3" if rv3 is not None and rv3 > 0.01 else ("PASS" if 1.9 <= p <= 2.1 else "FAIL")
        else:
            ratio = float(r["ratio"])
            if not 1.9 <= p <= 2.1:
                want = "NOT APPLIED"
            elif rv3 is not None and rv3 > 0.01:
                want = "UNRESOLVED (RV-3"
            else:
                want = "PASS" if ratio <= 0.05 else "FAIL"
        assert verdict.startswith(want), r


def test_one_dimensional_and_coefficient_checks():
    for r in read(OUT / "a1_a2.csv"):
        if r["check"] == "A-1a":
            assert (r["verdict"] == "PASS") is (float(r["value"]) <= 1e-11)
    assert {r["case"] for r in read(OUT / "a1_a2.csv") if r["check"] == "A-1a"} >= {"RV-1 kappa_c=16/l"}
    for r in read(OUT / "a3.csv"):
        assert (r["verdict"] == "PASS") is (float(r["max_rel_dev"]) <= 1e-13)
    assert {int(r["nx"]) for r in read(OUT / "a3.csv")} >= {4, 7, 20, 320, 576, 640}


def test_a11_rule_and_completeness():
    rows = read(OUT / "a11.csv")
    quals = sum(qual.step0.qualification(g) for chain in qual.CHAINS.values() for g in chain)
    assert len(rows) == quals
    for r in rows:
        assert (r["verdict"] == "PASS") is (float(r["S_stop"]) <= 0.01 * float(r["E_disc"]))


def test_a10b_results():
    meta = json.loads((A10B / "metadata.json").read_text(encoding="utf-8"))
    assert meta["dps"] >= 50 and meta["dirty"] is False and meta["all_pass"] is True
    rows = read(A10B / "a10b_hp.csv")
    expected = {(label, family, g) for (label, family), chain in qual.CHAINS.items() if label != "1"
                for g in chain if qual.step0.qualification(g)}
    assert {(r["r"], r["family"], (int(r["nx"]), int(r["ny"]))) for r in rows} == expected
    assert all(float(r["max_rel_dev"]) <= 1e-13 for r in rows)
    assert all(r["agree"] == "True" for r in read(A10B / "a10b_method_check.csv"))


@pytest.mark.parametrize("label, family, grid", [("1/5", "II", (160, 32)), ("7/5", "II", (20, 28)), ("3", "I", (20, 20))])
def test_fresh_recomputation_of_level_checks(label, family, grid):
    stored = next(r for r in read(OUT / "levels.csv")
                  if (r["r"], r["family"], int(r["nx"]), int(r["ny"])) == (label, family, *grid))
    fresh = qual.level_checks(label, family, grid)
    assert fresh["status"] == stored["status"] and fresh["s_stop"] == int(stored["s_stop"])
    assert str(fresh["controller_match"]) == stored["controller_match"] == "True"
    assert fresh["A6_eta_max"] <= 1e-12 and fresh["A8_reflection_max"] <= 1e-12 and fresh["A12_boundary_zero"]
    assert fresh["report_before_norm_ok"] and fresh["report_control_points_ok"]


def test_report_blocks_are_generated_from_the_stored_results():
    generated = qual.report_blocks(OUT)
    in_report = qual.step0.report_block_contents(qual.REPORT)
    for name, block in generated.items():
        assert in_report[name] == block, name


# --- stopping controller on the rectangle (12, item 12; NUMERICAL_SPEC 11.2, 11.3): QA checks of the exits -------------

def test_controller_exits_on_the_rectangle(monkeypatch):
    import kkvim.rectangle_run as rr
    from kkvim.controller import eps_floor, strict_parameters
    from kkvim.rectangle import rectangle_halfsteps

    run = rr.run_rectangle_until_stopped(1.0, 1.5, 1.0, 1.0, 40, 60, s_max=4)
    assert run.status == "MAX_HALF_STEPS" and len(run.history) == 4
    fixed = rectangle_halfsteps(1.0, 1.5, 1.0, 1.0, 40, 60, 4)
    for s in range(1, 5):  # the controller does not change the half-steps
        assert (run.history.field(s) == fixed.field(s)).all()
    floor = 10 * 2.0**-53 * (60 / math.pi) ** 2  # eps_floor with max(n_x, n_y) (11.2, ChV-R3)
    assert math.isclose(eps_floor(40, 60), floor, rel_tol=1e-15)
    assert strict_parameters(40, 60) == (1e-12, max(1e-10, 10 * floor))
    bad = rr.run_rectangle_until_stopped(1.0, 1.5, 1.0, 1.0, 40, 60, tol_E=9.9 * floor)
    assert bad.status == "ERROR_INPUT" and len(bad.history) == 0
    assert rr.run_rectangle_until_stopped(1.0, 5.000001, 1.0, 1.0, 40, 40).status == "ERROR_INPUT"  # D-038
    # STAGNATED: ten consecutive theta >= 0.9 with eps_E > tol_E (constant changes give theta = 1 from s = 3)
    monkeypatch.setattr(rr, "_field_changes", lambda prev, cur: (1e-3, 1e-3, 1e-3))
    stag = rr.run_rectangle_until_stopped(1.0, 1.5, 1.0, 1.0, 20, 32)
    assert stag.status == "STAGNATED" and len(stag.history) == 12
