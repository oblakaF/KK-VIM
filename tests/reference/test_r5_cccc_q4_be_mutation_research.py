"""Tests of the mutation research of B_E (scripts/r5_cccc_q4_be_mutation_research.py): every mutation of the catalogue reaches the
verdict fixed for it before the run; the G-5 analogue (a dropped term) is found by the comparison with the independent full bound
while |E_s| <= the reduced bound still holds; an equivalent rewrite is INAPPLICABLE (no channel reads a mutation flag); the files
under test are not changed."""

import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "scripts" / "r5_cccc_q4_be_mutation_research.py"
FILES = ("scripts/r5_cccc_q4_energy_bound_premises_audit.py", "scripts/r5_cccc_q4_be_second_path_research.py")


def _load():
    spec = importlib.util.spec_from_file_location("_be2_mutations", TOOL)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


mu = sys.modules.get("_be2_mutations") or _load()


def _digest():
    return {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in FILES}


@pytest.fixture(scope="module")
def result():
    before = _digest()
    res = mu.run_all()
    assert _digest() == before  # the mutations are executed as separate module objects; no file is changed
    return res


def test_catalogue_covers_the_required_categories():
    cats = " | ".join(m[2] for m in mu.MUTATIONS)
    for c in ("omitted", "sign", "upper bound replaced by the lower", "rounded down", "grid-step", "normalisation", "swapped",
              "substituted", "rounded prematurely", "number of terms", "second-order"):
        assert sum(c in m[2] for m in mu.MUTATIONS if m[1] == "primary") == 1, c
        assert sum(c in m[2] for m in mu.MUTATIONS if m[1] == "second") == 1, c


def test_every_mutation_reaches_its_declared_verdict(result):
    assert result["all_as_expected"], [(r["id"], {k: v["verdict"] for k, v in r["inputs"].items()}) for r in result["results"]
                                       if not r["as_expected"]]
    assert result["undetected"] == []
    for r in result["results"]:
        for x in r["inputs"].values():
            if x["verdict"] == "DETECTED":
                assert x["effective"] and x["channels_fired"]
            if x["verdict"] == "INAPPLICABLE":
                assert not x["effective"] and not x["channels_fired"]


def test_g5_analogue_dropped_term_found_by_the_full_independent_bound(result):
    p01 = next(r for r in result["results"] if r["id"] == "P-01")
    for x in p01["inputs"].values():
        assert x["verdict"] == "DETECTED" and x["inequality_still_holds"] is True  # the reduced bound still covers |E_s|
        assert set(x["channels_fired"]) == {"comparison"} and x["first_divergence"]["component"] == "BE"
        assert x["first_divergence"]["status"] == "ERROR_UNSAFE"


def test_unsafe_primary_mutations_are_classified_unsafe(result):
    by = {r["id"]: r for r in result["results"]}
    for mid, inp, comp in (("P-02", "rect", "B3"), ("P-03", "rect_D2", "B_omega"), ("P-04", "rect", "d_a"), ("P-11", "rect", "B2")):
        fd = by[mid]["inputs"][inp]["first_divergence"]
        assert fd["component"] == comp and fd["status"] == "ERROR_UNSAFE", (mid, fd)
    for mid, comp in (("P-06", "p.Sq"), ("P-08", "res_h2"), ("P-10", "grid.L")):
        fd = by[mid]["inputs"]["rect"]["first_divergence"]
        assert fd["component"] == comp and fd["status"] == "ERROR_VALUE", (mid, fd)


def test_inapplicable_inputs_and_the_null_mutation(result):
    by = {r["id"]: r for r in result["results"]}
    assert by["P-03"]["inputs"]["rect"]["verdict"] == "INAPPLICABLE"  # D = 1: exact root of D
    assert by["P-05"]["inputs"]["square"]["verdict"] == "INAPPLICABLE"  # h_x = h_y
    assert by["N-01"]["inputs"]["rect"]["verdict"] == "INAPPLICABLE"  # equivalent rewrite: nothing fires


def test_second_path_mutations_found_by_independent_channels(result):
    by = {r["id"]: r for r in result["results"]}
    assert "self_check" in by["S-04"]["inputs"]["rect"]["channels_fired"]  # root not certified by squaring
    assert "reference" in by["S-06"]["inputs"]["rect"]["channels_fired"]  # analytic S_q of the unit perturbation
    assert all("comparison" in by[m]["inputs"]["rect"]["channels_fired"] for m in ("S-01", "S-02", "S-07", "S-08", "S-09",
                                                                                    "S-10", "S-11"))


def test_references_hold_for_the_unmutated_path(result):
    assert all(result["references_unmutated"].values())


def test_a_mutation_text_must_occur_once():
    bad = ("X-01", "primary", "test", "this text does not occur", "x", {"rect": "DETECTED"}, "", "")
    with pytest.raises(RuntimeError, match="occurs 0 times"):
        mu.run_mutation(bad)
