"""Saved benchmark data of the "Сравнение" tab (kkvim_gui.backend.benchmarks; GUI-BENCHMARK-COMPARISON-01). Qt-free.

- the packaged data are verified by their payload SHA256 and schema; damaged or missing data are refused;
- values are the full-precision numbers of the completed experiments (checked against the committed D2 CSV and, where
  the local scratch evidence exists, against its source CSV files);
- a comparison is made only for exactly equal inputs, an exact centre node and a run that ended as requested;
- differences are beta_KK-VIM - beta_ref and (beta_KK-VIM - beta_ref) / beta_ref * 100, beta = w_c D / (q a^4);
- the solver arrays are not changed; kkvim_reference is not imported by the GUI.
"""

import csv
import dataclasses
import hashlib
import json
from importlib import resources
from pathlib import Path

import numpy as np
import pytest

from kkvim_gui.backend import benchmarks as bm
from kkvim_gui.backend.outcome import ROLE_DIAGNOSTIC, ROLE_RESEARCH, RunOutcome, frozen_copy
from kkvim_gui.backend.request import RunRequest

ROOT = Path(__file__).resolve().parents[1]
D = 2.1e11 * 0.01**3 / (12 * (1 - 0.3**2))
PUBLISHED = {1.0: "0.001265319087", 1.2: "0.001724870503", 1.4: "0.002068143209", 2.0: "0.002532955769"}


def packaged_text():
    return resources.files("kkvim_gui.backend").joinpath(bm.DATA_FILE).read_text(encoding="utf-8")


def request(**kw):
    base = dict(entered=(), a=1.0, b=1.0, t=0.01, E=2.1e11, nu=0.3, q=1000.0, n_x=32, n_y=32, N=1, supports="CCCC",
                half_steps=8)
    return RunRequest(**{**base, **kw})


def outcome(w_centre=6.5e-5, role=ROLE_RESEARCH, completed=True, origin="solver", **kw):
    """A synthetic solver outcome: a field with the given value at the centre node (no solver involved)."""
    r = request(**kw)
    x, y = r.a * np.arange(r.n_x + 1) / r.n_x, r.b * np.arange(r.n_y + 1) / r.n_y
    w = np.full((r.n_x + 1, r.n_y + 1), 1e-6)
    if r.n_x % 2 == 0 and r.n_y % 2 == 0:
        w[r.n_x // 2, r.n_y // 2] = w_centre
    Dr = r.E * r.t**3 / (12 * (1 - r.nu**2))
    return RunOutcome(origin=origin, request=r, solver="CCCC-N1", call="test", status="COMPLETED_FIXED_HALF_STEPS",
                      reason="", error=None, stop_kind="met" if completed else "not_met", stop_text="", science_kind="warning",
                      science_text="", mark="RESEARCH_ONLY / PROVISIONAL / NOT_QUALIFIED", D=Dr, x=frozen_copy(x),
                      y=frozen_copy(y), w=frozen_copy(w), w_bar=frozen_copy(w * Dr / (r.q * r.a**4)), s=8,
                      field_role=role, requested_completed=completed)


# --- data -------------------------------------------------------------------------------------------------------------

def test_packaged_data_are_verified_and_complete():
    p = bm.load()
    assert p["schema"] == bm.SCHEMA
    assert hashlib.sha256(bm.canonical(p)).hexdigest() == bm.PAYLOAD_SHA256
    assert p["inputs"] == {"a": 1.0, "t": 0.01, "E": 2.1e11, "nu": 0.3, "q": 1000.0, "D": D}
    assert [c["b"] for c in p["cccc"]] == [1.0, 1.2, 1.4, 2.0]
    for c in p["cccc"]:
        assert c["published"]["beta"] == float(PUBLISHED[c["b"]]) and c["published"]["printed"] == PUBLISHED[c["b"]]
        assert [(g["n_x"], g["n_y"]) for g in c["grids"]] == [(16, 16), (32, 32), (64, 64)]
        for g in c["grids"]:
            assert g["FEM"]["triangles"] == 2 * g["n_x"] ** 2 and g["FEM"]["solver_status"] == "SOLVED"
            for kind in ("D2", "FEM"):  # beta and the stored centre deflection use the same normalization
                assert g[kind]["beta"] == pytest.approx(g[kind]["w_centre_m"] * D / 1000.0, rel=1e-15)
    assert p["ssss"] == [{"b": 1.0, "navier": {"beta": 0.0040623526606750485, "w_centre_m": p["ssss"][0]["navier"]["w_centre_m"],
                                                "M": 2559, "error_bound_beta": p["ssss"][0]["navier"]["error_bound_beta"]}}]


def test_d2_square_equals_the_committed_qa_value_bitwise():
    rows = {r["nx"]: r for r in csv.DictReader(open(ROOT / "docs/results/r5_cccc_references/d2_external.csv",
                                                    encoding="utf-8")) if r["r"] == "1"}
    square = bm.load()["cccc"][0]["grids"]
    assert square[1]["D2"]["beta"] == float(rows["32"]["alpha_ref"])  # committed by QA 3e6fe2b


@pytest.mark.skipif(not (ROOT / "scratch/published_plate_fem_comparison_01/results.csv").exists(),
                    reason="local evidence of the experiments (scratch/, not in git)")
def test_values_are_the_full_precision_source_values():
    p = bm.load()
    src02 = list(csv.DictReader(open(ROOT / "scratch/published_plate_comparison_02/results.csv", encoding="utf-8")))
    fem = list(csv.DictReader(open(ROOT / "scratch/published_plate_fem_comparison_01/results.csv", encoding="utf-8")))
    for c in p["cccc"]:
        for g in c["grids"]:
            r = next(x for x in src02 if float(x["b_over_a"]) == c["b"] and x["N"] == "1" and x["grid"] == f"{g['n_x']}x{g['n_y']}")
            f = next(x for x in fem if float(x["b_over_a"]) == c["b"] and int(x["n"]) == g["n_x"])
            assert repr(g["D2"]["beta"]) == r["beta_D2"] and repr(g["D2"]["w_centre_m"]) == r["w_centre_D2_m"]
            assert repr(g["FEM"]["beta"]) == f["beta_FEM"] and repr(g["FEM"]["w_centre_m"]) == f["w_centre_m"]
    files = p["provenance"]["files_sha256"]
    for name, digest in files.items():
        if name.startswith(("scratch/", "docs/")):
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name


@pytest.mark.parametrize("damage", ["", "{", "{}", '{"payload": {}}', "value", "hash", "schema"])
def test_missing_or_damaged_data_are_refused(damage):
    doc = json.loads(packaged_text())
    if damage == "value":
        doc["payload"]["cccc"][0]["published"]["beta"] = 0.0012653
        text = json.dumps(doc)
    elif damage == "hash":  # consistent with itself, but not the released data
        doc["payload"]["cccc"][0]["published"]["beta"] = 0.0012653
        doc["payload_sha256"] = hashlib.sha256(bm.canonical(doc["payload"])).hexdigest()
        text = json.dumps(doc)
    elif damage == "schema":
        doc["payload"]["schema"] = "other"
        text = json.dumps(doc)
    else:
        text = damage
    with pytest.raises(bm.BenchmarkDataError):
        bm.load(text)
    c = bm.compare(outcome(), text)
    assert c.state == bm.DATA_ERROR and not c.rows and c.beta is None


def test_line_endings_of_the_checkout_do_not_matter():
    assert bm.load(packaged_text().replace("\n", "\r\n"))["schema"] == bm.SCHEMA


# --- matching and arithmetic -------------------------------------------------------------------------------------------

def test_cccc_exact_case_signs_units_and_denominators():
    o = outcome(w_centre=6.6e-5, b=1.4)
    c = bm.compare(o)
    assert c.state == bm.AVAILABLE and [r.reference.kind for r in c.rows] == ["published", "D2", "FEM"]
    assert c.centre == {"i": 16, "j": 16, "x": 0.5, "y": 0.7}
    assert c.w_centre_m == 6.6e-5 and c.beta == 6.6e-5 * D / (1000.0 * 1.0**4) and c.beta == c.w_bar_solver
    for r in c.rows:
        assert r.beta_kkvim == c.beta and r.delta == c.beta - r.reference.beta
        assert r.rel_pct == (c.beta - r.reference.beta) / r.reference.beta * 100.0
    pub, d2, fem = (r.reference.beta for r in c.rows)
    assert pub == 0.002068143209 and d2 == 0.002084859282139107 and fem == 0.002122772721266153
    assert c.rows[0].delta < 0 and bm.compare(outcome(w_centre=1.2e-4, b=1.4)).rows[0].delta > 0


@pytest.mark.parametrize("change", [dict(q=999.9999999), dict(q=-1000.0), dict(E=2.0e11), dict(t=0.0100001), dict(nu=0.25),
                                    dict(a=2.0, b=2.0), dict(b=1.5), dict(b=0.5)])
def test_other_physical_inputs_are_never_compared(change):
    c = bm.compare(outcome(**change))
    assert c.state == bm.UNAVAILABLE and not c.rows and c.beta is None and "не подставляются" in c.reason


def test_other_grid_keeps_only_the_continuum_reference():
    c = bm.compare(outcome(n_x=48, n_y=48))
    assert [r.reference.kind for r in c.rows] == ["published"]
    assert [m[0] for m in c.missing] == ["D2", "FEM"] and "48×48" in c.missing[0][1]
    c = bm.compare(outcome(n_x=32, n_y=64))
    assert [r.reference.kind for r in c.rows] == ["published"] and len(c.missing) == 2


def test_no_centre_node_no_interpolation():
    c = bm.compare(outcome(n_x=33, n_y=33))
    assert c.state == bm.UNAVAILABLE and not c.rows and "не интерполируется" in c.reason


def test_ssss_square_against_navier_only():
    c = bm.compare(outcome(supports="SSSS", w_centre=2.1e-4))
    assert c.state == bm.AVAILABLE and [r.reference.kind for r in c.rows] == ["navier"]
    assert c.rows[0].reference.beta == 0.0040623526606750485 and not c.missing
    for change in (dict(b=1.2), dict(b=2.0), dict(q=2000.0)):
        assert bm.compare(outcome(supports="SSSS", **change)).state == bm.UNAVAILABLE


@pytest.mark.parametrize("kw,state", [(dict(role=ROLE_DIAGNOSTIC, completed=False), bm.DIAGNOSTIC),
                                      (dict(completed=False), bm.DIAGNOSTIC), (dict(origin="demo"), bm.NO_RUN)])
def test_diagnostic_or_demo_fields_are_never_compared(kw, state):
    c = bm.compare(outcome(**kw))
    assert c.state == state and not c.rows and c.beta is None and c.w_centre_m is None
    assert bm.compare(None).state == bm.NO_RUN


def test_solver_arrays_are_not_changed():
    o = outcome()
    before = (o.w.copy(), o.w_bar.copy(), o.x.copy(), o.y.copy())
    bm.compare(o)
    for a, b in zip((o.w, o.w_bar, o.x, o.y), before, strict=True):
        np.testing.assert_array_equal(a, b)
        assert not a.flags.writeable


def test_gui_does_not_import_kkvim_reference():
    for path in (ROOT / "src/kkvim_gui").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "import kkvim_reference" not in text and "from kkvim_reference" not in text, path


def test_real_cccc_run_is_compared_at_full_precision():
    """A real CCCC N = 1 run (16 x 16, s = 1): beta from the solver field; the stored references unchanged."""
    from kkvim_gui.backend import kkvim_adapter

    o = kkvim_adapter.execute(request(n_x=16, n_y=16, half_steps=1))
    assert o.field_role == ROLE_RESEARCH and o.requested_completed
    c = bm.compare(o)
    i = j = 8
    assert c.centre == {"i": i, "j": j, "x": 0.5, "y": 0.5}
    assert c.w_centre_m == float(o.w[i, j]) and c.beta == float(o.w[i, j]) * o.D / (o.request.q * o.request.a**4)
    assert c.beta == pytest.approx(float(o.w_bar[i, j]), rel=1e-15)
    assert [r.reference.beta for r in c.rows] == [0.001265319087, 0.0013073014529768486, 0.0013747615249065552]


def test_comparison_record_is_json_with_exact_floats():
    c = bm.compare(outcome())
    back = json.loads(json.dumps(bm.as_record(c)))
    assert back["beta"] == c.beta and [r["delta"] for r in back["rows"]] == [r.delta for r in c.rows]
    assert dataclasses.is_dataclass(c)
