"""Technical checks of the R4 group-B research script scripts/r4_group_b_study.py (QA, B-PREQUALIFICATION).

Group B measures the error of KK-VIM; there is deliberately no threshold on that error (D-025, D-046). The tests check
the metric formulas, the two evaluations of the energy error, the exact identities of the projections, the equality of
the spectral discrete path with run_control, and the reproduction of the preliminary centre values of the task at the
precision they were given. The full pipeline (minutes) runs only with KKVIM_GROUP_B=1 - not in the mandatory CI.
"""

import csv
import importlib.util
import math
import os
from pathlib import Path

import numpy as np
import pytest

from kkvim_reference.r4_continuous_sequence import continuous_sequence, field_at_nodes, omega0_coefficients
from kkvim_reference.r4_energy_reference import omega0_energy
from kkvim_reference.r4_spectral_control import SpectralGrid, run_control

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "r4_group_b_study.py"
RESULTS = ROOT / "docs" / "results" / "r4_group_b"
spec = importlib.util.spec_from_file_location("r4_group_b_study", SCRIPT)
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


def test_ht_norm_of_one_mode_and_of_omega0():
    a, b, D, M = 1.3, 0.7, 2.0, 9
    V = np.zeros((5, 5))
    V[1, 2] = 0.25  # m = 3, n = 5
    expected = D * a * b / 4 * math.pi**4 * (9 / a**2 + 25 / b**2) ** 2 * 0.25**2
    assert math.isclose(study.ht_norm2(V, a, b, D, M), expected, rel_tol=1e-14)
    # |omega_0|^2 = -2 Pi(omega_0): block sum + strict outside bound encloses it
    for r in (0.2, 1.0, 5.0):
        M = 401
        w0 = omega0_coefficients(1.0, r, 1.0, 1.0, M)
        block = study.ht_norm2(w0, 1.0, r, 1.0, M)
        pi0, budget = omega0_energy(1.0, r, M=4001)
        out = study.omega0_ht_outside(1.0, r, 1.0, 1.0, M)
        assert block - 1e-14 * block <= -2 * pi0 + 2 * budget
        assert -2 * pi0 - 2 * budget <= block + out + 1e-14 * block


def test_outside_bounds_are_bounds():
    for r in (0.2, 1.0, 5.0):
        big = omega0_coefficients(1.0, r, 1.0, 1.0, 801)
        for M in (101, 201):
            K = (M + 1) // 2
            inside = np.zeros_like(big)
            inside[:K, :K] = big[:K, :K]
            outside_ht = study.ht_norm2(big - inside, 1.0, r, 1.0, 801)
            outside_l2 = study.l2_norm2(big - inside, 1.0, r)
            assert outside_ht <= study.omega0_ht_outside(1.0, r, 1.0, 1.0, M)
            assert outside_l2 <= study.omega0_l2_outside(1.0, r, 1.0, 1.0, M)


def test_centre_value_equals_the_node_value():
    st = continuous_sequence(1.0, 2.0, 3, 3, 61)[-1]
    W = field_at_nodes(st, 8, 8)
    assert math.isclose(study.centre_value(st.x, st.y), W[4, 4], rel_tol=1e-13)


def test_energy_routes_agree_and_projection_identities_hold():
    """Route E (difference of energies) and route D (positive Parseval sum) give the same err^2 within the budgets of
    both routes; the truncated recursion is an exact KK-VIM sequence of the restricted problem, so the Pythagoras
    identity err2(s-1) = err2(s) + |w_s - w_{s-1}|^2 (MATH_SPEC ВБ-17) holds to rounding and err_abs does not grow."""
    rows = study.continuous_case(1.0, 1.0, 1.0, 1.0, 2, M=201, M_check=401, s_max=6)
    for row in rows:
        budget_E = (row["E_strict_rel"] + row["E_model_rel"] + row["E_diag_rel"]) * row["dPi_energy_route"]
        budget_D = (row["D_strict_rel"] + row["D_model_rel"] + row["D_diag_rel"]) * row["err2_direct"]
        # the two routes share the coefficient arrays: this checks the evaluation of the norm, not the recursion
        assert abs(2 * row["dPi_energy_route"] - row["err2_direct"]) <= 2 * budget_E + budget_D
    for prev, row in zip(rows, rows[1:]):
        assert abs(row["pythagoras_defect_rel"]) <= 1e-12
        assert row["err2_direct"] <= prev["err2_direct"] * (1 + 1e-12)


def test_status_rule():
    assert study.status(1e-3, 0.2) == "RESOLVED"
    assert study.status(1e-3, -0.2) == "RESOLVED"
    assert study.status(3e-3, 0.2) == "UNRESOLVED"
    assert study.status(math.nan, 0.2) == "UNRESOLVED"


def test_mode_count_is_not_the_largest_index():
    """M is the largest kept odd index; the number of odd modes per axis is (M + 1)/2 (review b0c208d, И-2)."""
    for M, count in ((1, 1), (41, 21), (81, 41), (4001, 2001), (8001, 4001)):
        assert study.mode_count(M) == count == study.odd_modes(M).size
        assert study.odd_modes(M)[-1] == M
    for bad in (0, 80, 4000, 2.0, True):
        with pytest.raises(ValueError):
            study.mode_count(bad)


def test_measurement_status_and_continuous_method_status_are_separate():
    """И-1: RESOLVED for the finite spectral problem; for the untruncated continuous KK-VIM RESOLVED only at s = 1 (strict
    Navier-m tail), at s >= 2 at best PROVISIONAL (Q-022) - never RESOLVED, never automatically UNRESOLVED."""
    cs = study.continuous_status
    assert cs(1, "RESOLVED", 1e-6, 1.0, 1.0) == "RESOLVED"  # s = 1: decided by the strict budget, not by diagnostics
    assert cs(1, "RESOLVED", 0.5, 0.0, 1.0) == "UNRESOLVED"
    assert cs(2, "RESOLVED", 0.0, 1e-6, 1.0) == "PROVISIONAL"
    assert cs(2, "RESOLVED", 0.0, 0.5, 1.0) == "UNRESOLVED"  # diagnostics do not support the transfer
    assert cs(2, "UNRESOLVED", 0.0, 1e-6, 1.0) == "UNRESOLVED"
    rows = study.continuous_case(1.0, 1.0, 1.0, 1.0, 3, M=401, M_check=801, s_max=4)
    for row in rows:
        for key in ("e_rel", "e_rel_E", "delta_c", "E_inf"):
            assert row[f"status_meas_{key}"] == "RESOLVED"
            assert row[f"status_cont_{key}"] == ("RESOLVED" if row["s"] == 1 else "PROVISIONAL")
        assert row["sqrt_e_rel"] == math.sqrt(row["e_rel"]) and row["modes_per_axis"] == 201


def _read(name):
    with open(RESULTS / name, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_committed_tables_keep_the_scientific_numbers():
    """The committed CSV values are the output of the generator: recomputation of N = 3, r = 1 (M = 4001) reproduces
    e_rel, err_abs, sqrt(e_rel), delta_c, Pi(w), Pi(omega_0) to the 7 significant digits written (И-1...И-4 changed only
    statuses, budget columns and names, not the measured numbers)."""
    committed = {(int(t["N"]), t["r_label"], int(t["s"])): t for t in _read("continuous.csv")}
    rows = study.continuous_case(1.0, 1.0, 1.0, 1.0, 3, M=4001, M_check=4001, s_max=10)
    for row in rows:
        row["r_label"] = "1"
    study.apply_d056_4(rows)  # D-056.4: N = 3, r = 1, s = 2 - e_rel RESOLVED at the measurement level
    for row in rows:
        if row["s"] not in (1, 2, 3, 10):
            continue
        t = committed[(3, "1", row["s"])]
        for key in ("e_rel", "err_abs", "sqrt_e_rel", "delta_c", "Pi_w", "Pi_0", "E_inf_sampled", "L2_rel"):
            assert math.isclose(float(t[key]), row[key], rel_tol=1e-6), (key, row["s"])
        for key in ("status_meas_e_rel", "status_cont_e_rel", "status_meas_delta_c", "status_cont_delta_c"):
            assert t[key] == row[key]


def test_master_and_tables_are_consistent_with_the_sources():
    cont = {(t["N"], t["r_label"], t["s"]): t for t in _read("continuous.csv")}
    grid = {(t["N"], t["r_label"], t["s"], t["nx"], t["ny"], t["family"]): t for t in _read("grid.csv")}
    master = _read("master.csv")
    assert len(master) == len(grid)
    area = study.d056_4_area()
    for m in master:
        c = cont[(m["N"], m["r"], m["s"])]
        g = grid[(m["N"], m["r"], m["s"], m["n_x"], m["n_y"], m["family"])]
        for key in ("e_rel", "sqrt_e_rel", "err_abs", "status_meas_e_rel", "status_cont_e_rel", "status_meas_delta_c",
                    "status_cont_delta_c"):
            assert m[key] == c[key]
        assert m["delta_c_method"] == c["delta_c"] and m["delta_c_disc"] == g.get("d_disc_c", "")
        assert m["E_inf_disc"] == g["E_inf_disc"] and m["e_rel_h"] == g["e_rel_h"]
        assert math.isclose(float(m["sqrt_e_rel"]) ** 2, float(m["e_rel"]), rel_tol=2e-6)
        if int(m["s"]) >= 2:  # RESOLVED (measurement level) only inside the area of D-056.4
            for k in ("e_rel", "delta_c"):
                assert (m[f"status_cont_{k}"] == "RESOLVED") == ((int(m["N"]), m["r"], int(m["s"])) in area[k])
    tables = (RESULTS / "tables.md").read_text(encoding="utf-8")
    assert "√e_rel" in tables and "согласие e_rel,h с e_rel (1 %)" in tables
    assert "1/100 (энергия)" not in tables  # И-4: the agreement measure is not an energy distinguishability


def test_spectral_discrete_path_equals_run_control_and_discrete_identity():
    g = SpectralGrid(1.0, 0.5, 24, 12)
    ctrl = run_control(g, 3, 4)
    light = study.discrete_sequence(g, 3, (1, 2, 3, 4))
    for s in (1, 2, 3, 4):
        W = g.to_nodal(light[s][0])
        assert np.max(np.abs(W - ctrl[s - 1].W)) <= 1e-14 * np.max(np.abs(W))
        # Pi_h(w) - Pi_h(omega_h) = 1/2 |w - omega_h|_h^2
        d = g.spectral_energy(light[s][0]) - g.omega_energy()
        assert math.isclose(2 * d, study.discrete_err2(g, light[s][0]), rel_tol=1e-6)


def test_preliminary_centre_values_are_reproduced_at_their_printed_precision():
    """Task B-2 (r = 1): the preliminary signed centre errors are reproduced independently. Tolerance = one unit of the
    last printed digit of the given value (its own precision; two given values are consistent with truncation or a
    double rounding, not with a single rounding - report, B-2), not a new threshold. Signs must agree."""
    for N in (2, 3):
        rows = {row["s"]: row for row in study.continuous_case(1.0, 1.0, 1.0, 1.0, N, M=2001, M_check=2001, s_max=10)}
        for (n, s), ref in study.REFERENCE_VALUES.items():
            if n != N:
                continue
            digits = 3 if s == 10 else 2
            unit = 10.0 ** (math.floor(math.log10(abs(ref))) - digits + 1)
            assert math.copysign(1, rows[s]["delta_c"]) == math.copysign(1, ref)
            assert abs(rows[s]["delta_c"] - ref) <= unit, (N, s, rows[s]["delta_c"], ref)


@pytest.mark.skipif(os.environ.get("KKVIM_GROUP_B") != "1", reason="full group-B pipeline (minutes): KKVIM_GROUP_B=1")
def test_quick_pipeline_writes_tables(tmp_path):
    study.main(["--quick", "--out", str(tmp_path)])
    for name in ("continuous.csv", "grid.csv", "richardson.csv", "s1_navier.csv", "meta.json"):
        assert (tmp_path / name).exists()
