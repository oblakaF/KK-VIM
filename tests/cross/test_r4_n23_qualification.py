"""QA checks of the R4 N23 qualification (docs/R4_N23_QUALIFICATION.plan.md, docs/R4_N23_QUALIFICATION.qa-report.md).

Fast checks run in CI: independence of the QA script, consistency of the committed result tables with the rule of the plan,
unchanged group B statuses. The checks that need python-flint or mpmath (temporary tools, not project dependencies) are
skipped when they are absent. The full campaign (minutes) is re-run only with KKVIM_N23_HEAVY=1.
"""

import ast
import csv
import decimal
import importlib.util
import json
import math
import os
from decimal import Decimal
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "r4_n23_qualification.py"
RESULTS = ROOT / "docs" / "results" / "r4_n23_qualification"
RULE = 0.01
decimal.getcontext().prec = 120  # the 45-digit control values must not be rounded to the default 28 digits
MEASURES = ("err_abs", "e_rel", "delta_c", "E_inf")


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _rows(name):
    with open(RESULTS / name, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_qa_script_does_not_import_the_solver_or_the_references():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    assert not names & {"kkvim", "kkvim_reference"}


def test_matrix_rows_follow_the_rule_of_the_plan():
    rows = _rows("matrix.csv")
    assert len(rows) == 800  # 10 r x N = 2, 3 x s = 1..40
    assert {(r["r_label"], r["N"]) for r in rows} == {(lab, str(N)) for lab in
                                                      ("1/5", "1/3", "1/2", "2/3", "1", "7/5", "3/2", "2", "3", "5")
                                                      for N in (2, 3)}
    for r in rows:
        assert r["same_data_as_group_B"] == "True"
        # the transfer budget used is the larger of the double value and the rigorous upper bound of the arb path
        assert float(r["B_tr"]) >= float(r["E_dbl"])
        assert Decimal(float(r["B_tr"])) >= Decimal(r["E_arb_up"])  # exact binary value of the float
        for k in MEASURES:
            R = float(r[f"R_{k}"])
            if r[f"rule_{k}"] == "PASS":
                assert R <= RULE and not r[f"reason_{k}"].startswith("R-")
            else:
                assert r[f"rule_{k}"] == "UNRESOLVED" and R > RULE and r[f"reason_{k}"].startswith("R-")
        for k in ("err_abs", "e_rel"):
            assert (r[f"upper_{k}"] == "PASS") == (float(r[f"RU_{k}"]) <= RULE)
        assert r["L2"].startswith("UNRESOLVED")


def test_two_sided_pass_implies_one_sided_pass_and_monotone_transfer():
    rows = _rows("matrix.csv")
    cells = {}
    for r in rows:
        cells.setdefault((r["r_label"], r["N"]), []).append(r)
    for cell in cells.values():
        cell.sort(key=lambda r: int(r["s"]))
        E = [float(r["E_arb_up"]) for r in cell]
        assert all(b > a for a, b in zip(E, E[1:]))  # Q22-6, item 3: the bound grows strictly
        U = [float(r["U_arb_up"]) for r in cell]
        assert all(b <= a for a, b in zip(U, U[1:]))  # N23-7: U_s does not increase
        for r in cell:
            if int(r["s"]) >= 2 and r["rule_err_abs"] == "PASS":
                assert r["upper_err_abs"] == "PASS"


def test_high_precision_requirements_of_the_plan():
    s = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
    assert s["cells"] == 800 and s["same_data_as_group_B"] == 800
    for key in ("max_relrad_errM", "max_relrad_wc", "max_relrad_d", "max_sigma_relwidth",
                "max_mp_vs_arb_errM", "max_mp_vs_arb_wc", "max_mp_vs_arb_d", "max_E_mid_vs_mp"):
        assert s[key] <= 1e-40, key
    assert s["sigma_mp_in_ball"] is True


def test_group_b_statuses_unchanged():
    with open(ROOT / "docs" / "results" / "r4_group_b" / "continuous.csv", newline="", encoding="utf-8") as f:
        cont = list(csv.DictReader(f))
    assert len(cont) == 800
    for r in cont:
        for k in ("e_rel", "e_rel_E", "delta_c", "E_inf"):
            assert r[f"status_cont_{k}"] == ("RESOLVED" if r["s"] == "1" else "PROVISIONAL")


def test_rank_checks_and_printed_bounds_tables():
    rank = _rows("rank_checks.csv")
    assert rank and all(r["certified"] == "True" for r in rank if r["grid_or_M"] != "M = 21")
    for r in rank:
        if r["certified"] == "True":
            assert Decimal(r["mu_up"]) < 1 and Decimal(r["enc_lo"]) > 0
    printed = _rows("printed_bounds.csv")
    assert len(printed) == 20 + 132
    for r in printed:
        assert (Decimal(r["mu_printed"]) >= Decimal(r["mu_author"])) == (r["mu_printed_is_upper_bound"] == "True")


def test_group_b_diagnostic_on_a_small_problem():
    pytest.importorskip("kkvim_reference")
    study = _load(ROOT / "scripts" / "r4_group_b_study.py", "r4_group_b_study_n23")
    rows = study.n23_transfer_case(1.0, 0.2, 1.0, 1.0, 3, M=41, s_max=6)
    assert [r["s"] for r in rows] == list(range(1, 7))
    assert rows[0]["E_transfer"] == rows[0]["T"]
    for a, b in zip(rows, rows[1:]):
        assert b["E_transfer"] > a["E_transfer"] and b["U_upper"] <= a["U_upper"]
        assert 0 < b["theta"] <= 1
    for r in rows:
        assert math.isclose(r["err2_direct"], r["err_M"] ** 2 + 0.5 * r["T"] ** 2, rel_tol=1e-14)
    # the former CSV format of group B is the default of write_csv
    assert study.write_csv.__defaults__ == (".6e",)


def test_arb_and_mpmath_paths_agree_on_a_small_problem():
    pytest.importorskip("flint")
    pytest.importorskip("mpmath")
    q = _load(SCRIPT, "r4_n23_qualification_small")
    P = q.ArbProblem("1/5", 41, 512)
    arb_rows = q.arb_case(P, 3, 6)
    mp_rows = q.mp_case("1/5", 3, 41, 6, 80, {})
    for a, m in zip(arb_rows, mp_rows):
        assert Decimal(a["sigma_lo"]) <= Decimal(m["sigma"]) * (1 + Decimal("1e-40"))
        assert Decimal(m["sigma"]) <= Decimal(a["sigma_hi"]) * (1 + Decimal("1e-40"))
        for ka, km in (("errM_mid", "errM"), ("wc_mid", "wc"), ("E_mid", "E")):
            assert abs(Decimal(a[ka]) - Decimal(m[km])) <= Decimal("1e-40") * abs(Decimal(m[km]))
        assert Decimal(a["E_up"]) >= Decimal(m["E"])


def test_arb_upper_bound_dominates_group_b_double_on_a_small_problem():
    pytest.importorskip("flint")
    pytest.importorskip("mpmath")
    pytest.importorskip("kkvim_reference")
    q = _load(SCRIPT, "r4_n23_qualification_small2")
    study = _load(ROOT / "scripts" / "r4_group_b_study.py", "r4_group_b_study_n23b")
    dbl = study.n23_transfer_case(1.0, 1.0, 1.0, 1.0, 2, M=41, s_max=6)
    hp = q.arb_case(q.ArbProblem("1", 41, 512), 2, 6)
    for d, h in zip(dbl, hp):
        assert math.isclose(d["err_M"], float(h["errM_mid"]), rel_tol=1e-12)
        assert math.isclose(d["sigma_N_L2"], float(Decimal(h["sigma_lo"])), rel_tol=1e-10)
        assert math.isclose(d["E_transfer"], float(h["E_up"]), rel_tol=1e-10)


def test_rank_certificate_at_the_boundary_m23():
    pytest.importorskip("flint")
    pytest.importorskip("mpmath")
    q = _load(SCRIPT, "r4_n23_qualification_rank")
    res = q.rank_continuous("1", 2, 23)
    assert res["certified"] and Decimal(res["mu_up"]) < Decimal("0.1")


def _text(data):
    """Bytes of a text result with CRLF line ends normalised to LF: a checkout of Git for Windows (core.autocrlf=true) has CRLF,
    the script writes LF (csv) or the platform line end (json). Nothing else is changed; a lone CR is content."""
    text = data.replace(b"\r\n", b"\n")
    assert b"\r" not in text
    return text


def test_text_comparison_ignores_only_line_ends():
    lf = b"r_label,N,s,R\n1/5,2,1,1.25e-03\n1/5,2,2,3.5e-03\n"
    assert _text(lf.replace(b"\n", b"\r\n")) == _text(lf)
    for other in (b"r_label,N,s,R\n1/5,2,1,1.26e-03\n1/5,2,2,3.5e-03\n",     # one digit
                  b"r_label,N,s,R\n1/5,2,2,3.5e-03\n1/5,2,1,1.25e-03\n",     # order of the rows
                  b"r_label,N,s,R\n1/5,2,1,1.250e-03\n1/5,2,2,3.5e-03\n",    # printed precision
                  b"r_label,N,s,R\n1/5,2,1,1.25e-03\n",                      # a missing row
                  b"r_label,N,s,R\n1/5,2,1,1.25e-03\n1/5,2,2,3.5e-03"):      # a missing final line end
        assert _text(other) != _text(lf) and _text(other.replace(b"\n", b"\r\n")) != _text(lf)
    with pytest.raises(AssertionError):
        _text(b"1/5,2,1\r1.25e-03\n")


@pytest.mark.skipif(os.environ.get("KKVIM_N23_HEAVY") != "1", reason="full combine only with KKVIM_N23_HEAVY=1")
def test_combine_reproduces_the_committed_tables(tmp_path):
    import shutil

    for name in ("group_b_transfer.csv", "hp_arb.csv", "hp_mpmath.csv"):
        shutil.copy(RESULTS / name, tmp_path / name)
    q = _load(SCRIPT, "r4_n23_qualification_combine")
    q.main(["combine", "--out", str(tmp_path)])
    for name in ("matrix.csv", "smax.csv", "summary.json"):
        assert _text((tmp_path / name).read_bytes()) == _text((RESULTS / name).read_bytes()), name
