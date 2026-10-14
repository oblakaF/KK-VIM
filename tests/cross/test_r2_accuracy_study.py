"""Technical checks of the R2 group-B research script scripts/r2_accuracy_study.py (QA).

Only the metric formulas, the figure writer and the pipeline are checked here, on small independent
arrays and small grids. There is deliberately no threshold on the difference from Navier: group B
measures that difference (D-025; NUMERICAL_SPEC.md 17, 18.2).
"""

import csv
import importlib.util
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "r2_accuracy_study.py"
spec = importlib.util.spec_from_file_location("r2_accuracy_study", SCRIPT)
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


def grid_values(f, n):
    x = np.arange(n + 1) / n
    X, Y = np.meshgrid(x, x, indexing="ij")
    return f(X, Y)


def test_trapezoid_weights_integrate_bilinear_exactly():
    for n_x, n_y, a, b in ((4, 6, 1.0, 1.0), (10, 3, 2.0, 0.5)):
        w = study.trapezoid_weights(n_x, n_y, a, b)
        x = np.arange(n_x + 1) * a / n_x
        y = np.arange(n_y + 1) * b / n_y
        X, Y = np.meshgrid(x, y, indexing="ij")
        assert math.isclose(w.sum(), a * b, rel_tol=1e-14)
        assert math.isclose(float(np.sum(w * X * Y)), a * a * b * b / 4, rel_tol=1e-14)


def test_relative_errors_on_scaled_field():
    ref = grid_values(lambda x, y: np.sin(np.pi * x) * np.sin(np.pi * y), 8)
    w = 1.01 * ref
    assert math.isclose(study.error_inf(w, ref), 0.01, rel_tol=1e-12)
    assert math.isclose(study.error_l2(w, ref, study.trapezoid_weights(8, 8)), 0.01, rel_tol=1e-12)
    assert math.isclose(study.delta_centre(w, ref), 0.01, rel_tol=1e-12)
    assert study.delta_centre(0.98 * ref, ref) < 0.0


def test_max_abs_difference_location():
    ref = np.zeros((5, 5))
    w = ref.copy()
    w[1, 3] = -2.0
    w[3, 1] = 1.5
    value, x, y = study.max_abs_difference(w, ref)
    assert value == 2.0 and (x, y) == (0.25, 0.75)


def test_grid_difference_order_and_richardson_on_synthetic_h2_fields():
    exact = lambda x, y: np.sin(np.pi * x) * np.sin(np.pi * y)
    error = lambda x, y: x * (1 - x) * y * (1 - y)
    fields = {n: grid_values(lambda x, y: exact(x, y) + 3.0 * error(x, y) / n**2, n) for n in (8, 16, 32)}
    p = study.self_convergence_order(fields[8], fields[16], fields[32])
    assert math.isclose(p, 2.0, rel_tol=1e-12)
    w_re = study.richardson(fields[8], fields[16])
    assert np.max(np.abs(w_re - grid_values(exact, 8))) <= 1e-15
    expected = np.max(np.abs(fields[8] - study.restrict(fields[16]))) / np.max(np.abs(fields[8]))
    assert study.grid_difference(fields[8], fields[16]) == expected
    assert math.isclose(study.observed_order(4.0e-3, 1.0e-3), 2.0)
    assert study.in_window(2.0) and not study.in_window(1.8) and not study.in_window(None) and not study.in_window(float("nan"))


def test_asymmetry_and_energy_error():
    sym = grid_values(lambda x, y: np.sin(np.pi * x) * np.sin(np.pi * y), 6)
    assert study.asymmetry(sym) <= 1e-16
    asym = sym.copy()
    asym[2, 4] += 0.1
    assert math.isclose(study.asymmetry(asym), 0.1 / np.max(np.abs(asym)), rel_tol=1e-12)
    assert study.energy_error(-0.9, -1.0) == pytest.approx(0.1, rel=1e-15, abs=0)
    assert study.energy_error(-1.1, -1.0) < 0.0


def test_svg_figures_are_valid_xml(tmp_path):
    study.svg_line_chart(tmp_path / "a.svg", [("s1", [1, 2, 3], [1.0, 0.5, 0.25], False), ("s2", [1, 2, 3], [2.0, 1.0, 0.5], True)],
                         title="t", xlabel="x", ylabel="y", ylog=True, hlines=[("ref", 0.4)])
    study.svg_heatmap(tmp_path / "b.svg", np.arange(25.0).reshape(5, 5) - 12.0, title="h", stride=1)
    for name in ("a.svg", "b.svg"):
        assert ET.parse(tmp_path / name).getroot().tag.endswith("svg")


def test_own_recursion_matches_reference_within_budgets():
    from kkvim_reference.parseval_recursion import ParsevalRecursion

    rec = ParsevalRecursion(a=1.0, q=1.0, D=1.0, s_max=7)
    own = study.own_recursion(7)
    for s in range(1, 8):
        budget, energy = rec.halfstep(s).budget, rec.energy(s)
        for x, y in zip(rec.coefficients(s).invariants(), own[s]["invariants"]):
            assert abs(x - y) <= budget.invariant_error * abs(y)
        assert abs(rec.centre_value(s) - own[s]["centre"]) <= budget.field_error * abs(own[s]["centre"])
        assert abs(energy.energy - own[s]["energy"]) <= energy.error * abs(own[s]["energy"])


def test_russian_number_format():
    assert study.ru(0.0040623526607, 11) == "0,0040623526607"
    assert study.ru(-1.0281e-3, 5) == "−0,0010281"
    assert study.ru(5.301e-6) == "5,301·10⁻⁶"
    assert study.ru(1350.0, 3) == "1,35·10³"
    assert study.signed(2.4e-10, 2) == "+2,4·10⁻¹⁰" and study.ru(None) == "—" and study.ru("") == "—"
    assert study.md_table(["max|x|"], [("a|b",)]).splitlines()[0] == r"| max\|x\| |"


def test_report_tables_are_generated_from_the_committed_csv():
    """The AUTO blocks of docs/R2_ACCURACY_STUDY.md equal the tables built from docs/results/r2/*.csv."""
    root = SCRIPT.parents[1]
    generated = study.report_blocks(root / "docs" / "results" / "r2")
    in_report = study.report_block_contents(root / "docs" / "R2_ACCURACY_STUDY.md")
    assert set(generated) == set(in_report)
    for name, block in generated.items():
        assert in_report[name] == block, name


def test_pipeline_on_small_grids(tmp_path):
    study.run_study(tmp_path, grids=(20, 40), diagnostic_grid=None, report=None)
    expected = {"deflection_errors.csv", "quarters.csv", "halfsteps_n160_n320.csv", "grid_convergence.csv", "richardson.csv",
                "energy_discrete.csv", "sa_states.csv", "parseval_extended.csv", "energy_extrapolation.csv", "error_decomposition.csv",
                "asymmetry.csv", "navier_reference.csv", "stopping.csv", "metadata.json", "tables.md", "centre_vs_s.svg",
                "delta_c_vs_s.svg", "errors_vs_h.svg", "energy_vs_h.svg", "difference_field_stop.svg"}
    assert expected <= {p.name for p in tmp_path.iterdir()}
    with open(tmp_path / "stopping.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [r["status"] for r in rows] == ["CONVERGED", "CONVERGED"]
    with open(tmp_path / "deflection_errors.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert {r["s"] for r in rows} == {"1", "2", "3", "4", "stop"}
    for r in rows:  # stored metrics are consistent with the stored centre values
        dc = (float(r["w_h_centre"]) - float(r["w_N_centre"])) / float(r["w_N_centre"])
        # CSV keeps 12 significant digits: the recomputed delta_c (~1e-3) carries ~1e-9 relative rounding
        assert abs(dc - float(r["delta_c"])) <= 1e-8 * max(abs(dc), 1e-12)
    blocks = study.report_blocks(tmp_path)  # report tables can be built from the CSV of any run
    assert {"key_results", "decomposition", "sa_states", "parseval_extended"} <= set(blocks)
    with open(tmp_path / "sa_states.csv", encoding="utf-8") as f:
        sa = list(csv.DictReader(f))
    assert [r["s"] for r in sa] == ["1", "2", "3", "4", "stop"] and sa[-1]["s_actual"] == "7"
