"""Technical checks of scripts/r4_numerical_prequal.py (QA, R4 numerical prequalification N = 2-3).

The criteria were fixed before the computations (docs/R4_NUMERICAL_PREQUALIFICATION.plan.md, commit 7748877). These
tests check the harness, the definitions (zeta^(N) invariance, the rank guard on exact rank-deficient fields), the
strict modal bound on small grids (temporary mpmath, skipped without it) and the consistency of the committed tables
with the pre-fixed decision rules. No threshold is chosen here.
"""

import csv
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

from kkvim.coupled import coupled_coefficients
from kkvim.grid import Grid1D

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "r4_numerical_prequal.py"
RESULTS = ROOT / "docs" / "results" / "r4_numerical_prequal"
spec = importlib.util.spec_from_file_location("r4_numerical_prequal", SCRIPT)
pq = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pq)


def _rows(name):
    with open(RESULTS / name, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_harness_reproduces_run_coupled():
    for r, N, n in ((1.0, 2, 24), (0.2, 3, 30), (5.0, 3, 24)):
        assert pq.harness_matches_run_coupled(r, N, n, n, 6)


def test_zeta_measure_is_gl_invariant_and_grows_with_the_highest_start_mode():
    """kappa^(N) = lambda_max(A^-1 C)^(1/4) does not depend on the representation of the given set (R4_NUMERICAL_SPEC 4.4)."""
    g = Grid1D(1.0, 40)
    rec = pq.harness(1.0, 3, 40, 40, 1)
    G = rec[0]["given"]
    T = np.array([[1.0, 0.3, -2.0], [0.0, 2.0, 0.5], [0.4, 0.0, 1.0]])
    lam = []
    for M in (G, G @ T):
        co = coupled_coefficients(g, M, 1.0)
        lam.append(float(np.max(np.linalg.eigvals(np.linalg.solve(co.A, co.C)).real)))
    assert math.isclose(lam[0], lam[1], rel_tol=1e-10)
    assert pq.zeta_of(1.0, 3, 40, 40) > pq.zeta_of(1.0, 2, 40, 40)


def test_rank_guard_rejects_exact_rank_deficient_constructions():
    """Criterion B4 (i) on a subset: no exact rank-(N-1) field passes the research rank guard."""
    for row in pq.b4_rank():
        if row["guard_K_u_kappa_lt_1"]:
            assert row["criterion_i_rank_guard_fails"], row


def test_strict_modal_bound_on_a_small_grid():
    mp = pytest.importorskip("mpmath")
    mp.mp.dps = 40
    for h in pq.harness(1.0, 3, 12, 12, 3):
        row = pq.b2_modes(mp, h["step"], 3, "1", 12, h["s"])[0]
        assert row["violations"] == 0 and row["min_bound_over_actual"] >= 1.0


def test_committed_tables_follow_the_pre_fixed_rules():
    b5 = [t for t in _rows("b5_energy.csv") if t.get("m")]
    for t in b5:
        assert math.isclose(float(t["m"]), float(t["Q"]) / (10 * float(t["tau"])), rel_tol=1e-5)
        if float(t["m"]) >= 1:  # resolved test: E1 and E2 must hold (criterion B5 (i))
            assert t["e1"] == "True" and t["e2"] == "True", t
    for t in _rows("b6_summary.csv"):
        assert float(t["tolE_over_10floor"]) == pytest.approx(pq.TOL_E / (10 * float(t["eps_floor_N"])), rel=1e-5)
    for t in _rows("b2_modal.csv"):
        assert int(t["violations"]) == 0
    for t in _rows("b4_rank.csv"):
        if t["guard_K_u_kappa_lt_1"] == "True":
            assert t["criterion_i_rank_guard_fails"] == "True"
    for t in _rows("b3_dense_elements.csv"):
        assert int(t["model_violations"]) == 0
    for t in _rows("b8_zeta.csv"):
        z = float(t["zeta_N"])
        assert t["class"] == ("Q" if z <= pq.ZETA_Q else ("A" if z <= pq.ZETA_A else "G"))
