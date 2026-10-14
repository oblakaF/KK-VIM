"""QA tests of step 0 of the R3 solver qualification (R3_NUMERICAL_SPEC 12; D-039) - scripts/r3_solver_step0.py.

Checked: the selected chains and the mandatory region against the specification (6.3, 8.2, 8.3) and the categories
of the R-7 surrogate (margins.csv); completeness of the stored table; the classification rule; the independent QA
energy formulas against the published discrete R2 values at r = 1 (calibration_r1.csv); a fresh recomputation of
small grids; the report blocks against the CSV. No criterion or grid is derived from the measured values.
"""

import csv
import importlib.util
import json
import math
from fractions import Fraction
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("r3_solver_step0", ROOT / "scripts" / "r3_solver_step0.py")
step0 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(step0)
OUT = step0.OUT
MARGIN_LABEL = {label: m for label, _, m in step0.RATIOS}


def read(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


@pytest.fixture(scope="module")
def rows():
    return read(OUT / "step0_energy.csv")


@pytest.fixture(scope="module")
def runs():
    return read(OUT / "step0_runs.csv")


@pytest.fixture(scope="module")
def meta():
    return json.loads((OUT / "metadata.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def margins():
    return read(step0.MARGINS)


def test_chains_follow_the_family_ii_formula_and_doubling():
    """6.1, 6.3: family II (n_x, n_y) = (4qk, 4pk) for r = p/q; every chain doubles both components."""
    for label, r, family, chain in step0.chains():
        assert all((2 * a, 2 * b) == c for (a, b), c in zip(chain, chain[1:]))
        for nx, ny in chain:
            assert Fraction(ny, nx) == r if family == "II" else nx == ny
            if family == "II":
                k = Fraction(nx, 4 * r.denominator)
                assert k.denominator == 1 and (nx, ny) == (4 * r.denominator * k, 4 * r.numerator * k)
        top = chain[-1]
        assert max(top) <= 640 and max(2 * top[0], 2 * top[1]) > 640  # the chain ends at the largest level <= 640
    assert [x[0] for x in step0.RATIOS] == ["1/5", "1/3", "1/2", "2/3", "1", "7/5", "3/2", "2", "3", "5"]


def test_mandatory_s3_region_is_the_mandatory_candidate_category_of_r7(margins):
    """8.2, item 2: s = 3 mandatory where the R-7 category is MANDATORY_CANDIDATE (m_low >= 4); r = 1: n <= 160."""
    category = {(m["r"], m["family"], int(m["nx"]), int(m["ny"]), int(m["s"])): m["category"] for m in margins}
    for label, _, family, chain in step0.chains():
        expected = tuple(g for g in chain if step0.qualification(g) and (
            max(g) <= 160 if label == "1" else category[(MARGIN_LABEL[label], family, *g, 3)] == "MANDATORY_CANDIDATE"))
        assert step0.MANDATORY_S3[(label, family)] == expected, (label, family)
        for g in chain:  # s = 2: every qualification grid is MANDATORY_CANDIDATE in R-7
            if step0.qualification(g):
                assert category[(MARGIN_LABEL[label], family, *g, 2)] == "MANDATORY_CANDIDATE"


def test_table_is_complete(rows, runs):
    seen = {(r["r"], r["family"], int(r["nx"]), int(r["ny"]), int(r["s"])) for r in rows}
    for label, _, family, chain in step0.chains():
        for g in chain:
            run = [x for x in runs if (x["r"], x["family"], int(x["nx"]), int(x["ny"])) == (label, family, *g)]
            assert len(run) == 1
            for s in range(2, int(run[0]["s_stop"]) + 1):
                assert (label, family, *g, s) in seen
            assert int(run[0]["s_stop"]) >= 3
    assert len(seen) == len(rows)


def test_mandatory_flags_and_verdicts_follow_the_rule(rows):
    for r in rows:
        grid = (int(r["nx"]), int(r["ny"]))
        s = int(r["s"])
        qual = step0.qualification(grid)
        mandatory = qual and (s == 2 or (s == 3 and grid in step0.MANDATORY_S3.get((r["r"], r["family"]), ())))
        assert (r["qualification"] == "True") is qual and (r["mandatory"] == "True") is mandatory
        Q, tau, tau_R = float(r["Q"]), float(r["tau"]), float(r["tau_R"])
        assert (r["resolved"] == "True") is (Q >= 10 * tau)
        status = "UNRESOLVED" if Q < 10 * tau else ("PASS" if r["e1"] == r["e2"] == "True" else "FAIL")
        assert r["status_qa"] == status
        assert (r["abs_gamma_le_g_allow"] == "True") is (abs(float(r["gamma"])) <= float(r["g_allow"]))
        assert math.isclose(tau, abs(float(r["gamma"])) + tau_R, rel_tol=1e-9)
        if mandatory:
            ok = (status == "PASS" and r["abs_gamma_le_g_allow"] == "True" and r["within_e_sur"] == "True")
            assert r["verdict"] == ("PASS" if ok else "BLOCKER")
        else:
            assert r["verdict"] == status


def test_metadata_is_consistent(rows, runs, meta):
    assert meta["n_energy_rows"] == len(rows) and meta["n_runs"] == len(runs)
    assert meta["n_mandatory_rows"] == sum(r["mandatory"] == "True" for r in rows)
    assert meta["n_blockers"] == sum(r["verdict"] in ("BLOCKER", "FAIL") for r in rows)
    assert meta["surrogate_margins_sha256"] == step0.sha256(step0.MARGINS)
    assert len(meta["commit"]) == 40 and meta["dirty"] is False
    assert {"numpy", "scipy", "kkvim"} <= set(meta["versions"])


def test_qa_energy_formulas_reproduce_the_published_r2_values():
    """r = 1, n = 20, 40: Q_s, R(w(s-1)), R(w(s)), R_f of the QA formulas against the discrete R2 values of R-7."""
    published = {(int(r["n"]), int(r["s"])): r for r in read(step0.CALIBRATION)}
    for n in (20, 40):
        res = step0.run_rectangle_until_stopped(1.0, 1.0, 1.0, 1.0, n, n)
        recs = res.history.records
        for s in (2, 3, 4):
            e = step0.qa_energy(recs[s - 2], recs[s - 1], 1.0 / n, 1.0 / n, n, n)
            p = published[(n, s)]
            assert abs(e["Q"] - float(p["Q_disc"])) <= e["tau"]  # same quantity, rounding level
            for k, col in (("R_prev", "R_prev_disc"), ("R_now", "R_now_disc"), ("R_f", "R_f_disc")):
                assert math.isclose(e[k], float(p[col]), rel_tol=1e-8), (n, s, k)  # 10 printed digits


@pytest.mark.parametrize("label, family, grid", [("1/5", "I", (20, 20)), ("7/5", "II", (20, 28)), ("3", "II", (24, 72)),
                                                 ("2/3", "II", (60, 40))])
def test_fresh_recomputation_agrees_with_the_stored_rows(rows, label, family, grid):
    r = dict((x[0], x[1]) for x in step0.RATIOS)[label]
    nx, ny = grid
    res = step0.run_rectangle_until_stopped(1.0, float(r), 1.0, 1.0, nx, ny)
    recs = res.history.records
    stored = {int(x["s"]): x for x in rows if (x["r"], x["family"], int(x["nx"]), int(x["ny"])) == (label, family, nx, ny)}
    assert sorted(stored) == list(range(2, len(recs) + 1))
    for prev, cur in zip(recs, recs[1:]):
        e = step0.qa_energy(prev, cur, 1.0 / nx, float(r) / ny, nx, ny)
        x = stored[cur.s]
        assert abs(e["Q"] - float(x["Q"])) <= e["tau"] and math.isclose(e["tau_R"], float(x["tau_R"]), rel_tol=1e-8)
        status = "UNRESOLVED" if not e["resolved"] else ("PASS" if e["e1"] and e["e2"] else "FAIL")
        assert status == x["status_qa"]


def test_report_blocks_are_generated_from_the_stored_results():
    generated = step0.report_blocks(OUT)
    in_report = step0.report_block_contents(step0.REPORT)
    assert set(generated) <= set(in_report)
    for name, block in generated.items():
        assert in_report[name] == block, name
