"""Tests of the R5-V Q4 gamma audit tool (scripts/r5_cccc_q4_gamma_audit.py): exact identities, the two independent exact paths, the
reproduction of the Q3 double path, the bounds (including small decimal precisions), negative and not-applicable cases, data
provenance and the deterministic CI profile. QA research tool; no import of the working ``kkvim``."""

import ast
import csv
import importlib.util
import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from kkvim_reference import r5_cccc_halfstep as hs

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "r5_cccc_q4_gamma_audit.py"
spec = importlib.util.spec_from_file_location("r5_cccc_q4_gamma_audit", SCRIPT)
ga = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ga)


def random_halfstep(rng, nx, ny, N, direction):
    """Given set, companion and solved set with values of different binary exponents (not a solution: gamma is then large)."""
    n_f, n_s = (nx, ny) if direction == "Y" else (ny, nx)
    scale = lambda: 2.0 ** rng.integers(-30, 4)  # noqa: E731
    G = rng.standard_normal((n_f - 1, N)) * scale()
    C = rng.standard_normal((n_s - 1, N)) * scale()
    V = rng.standard_normal((n_s - 1, N)) * scale()
    return G, C, V


def exact_gamma_A(lx, nx, ly, ny, direction, G, C, V, D=1.0, q=1.0):
    lf, nf, ls, ns = (lx, nx, ly, ny) if direction == "Y" else (ly, ny, lx, nx)
    (Gi,), EG = ga.dyadic(G)
    (Vi, Ci), EV = ga.dyadic(V, C)
    coef = ga.coefficients_exact(Gi, EG, Fraction(lf) / nf, Fraction(q))
    return ga.gamma_from_coefficients(coef, Vi, Ci, EV, Fraction(ls) / ns, Fraction(D))


def oriented(direction, given, other):
    return (given, other) if direction == "Y" else (other, given)


CASES = [(1.0, 5, 1.3, 6, 1, "Y"), (1.0, 7, 0.6, 5, 2, "X"), (2.0, 8, 1.0, 9, 3, "Y"), (1.0, 6, 1.0, 6, 2, "Y"),
         (0.5, 9, 1.5, 7, 3, "X"), (1.0 / 3, 6, 1.0, 8, 1, "X")]


@pytest.mark.parametrize("lx, nx, ly, ny, N, direction", CASES)
def test_paths_A_and_B_agree_exactly(lx, nx, ly, ny, N, direction):
    """VOK-2 by two different implementations: residual path A and two-dimensional energy path B give the same rational gamma."""
    rng = np.random.default_rng(nx * 100 + ny * 10 + N)
    G, C, V = random_halfstep(rng, nx, ny, N, direction)
    gA = exact_gamma_A(lx, nx, ly, ny, direction, G, C, V, D=1.7, q=-0.3)
    Pp, Ps, Q = ga.energies_path_b(lx, nx, ly, ny, 1.7, -0.3, *oriented(direction, G, C), *oriented(direction, G, V))
    assert gA == Q - (Pp - Ps)
    assert gA != 0


@pytest.mark.parametrize("lx, nx, ly, ny, N, direction", CASES[:4])
def test_path_A_equals_the_accepted_exact_residual(lx, nx, ly, ny, N, direction):
    """Comparison with the accepted R-HP reference (r5_cccc_halfstep.residual_exact): same exact gamma."""
    rng = np.random.default_rng(7 + nx + N)
    G, C, V = random_halfstep(rng, nx, ny, N, direction)
    p = hs.HalfStep(lx, nx, ly, ny, direction, G)
    r = hs.residual_exact(p, V.T.reshape(-1))
    Delta = [Fraction(float(C[j, i])) - Fraction(float(V[j, i])) for i in range(N) for j in range(V.shape[0])]  # exact C - V
    expected = Fraction(p.l_s) / p.n_s * sum((a * b for a, b in zip(r, Delta)), Fraction(0))
    assert exact_gamma_A(lx, nx, ly, ny, direction, G, C, V) == expected


def test_path_B_energy_equals_the_accepted_exact_energy():
    """Pi of path B against r5_cccc_energy.pi_exact (accepted G-2 reference) on one field."""
    from kkvim_reference import r5_cccc_energy as en

    rng = np.random.default_rng(3)
    X, Y = rng.standard_normal((6, 2)), rng.standard_normal((8, 2))
    Z = (np.zeros((6, 2)), np.zeros((8, 2)))
    _, Ps, _ = ga.energies_path_b(1.0, 7, 1.4, 9, 1.0, 1.0, *Z, X, Y)
    w = [[sum(Fraction(float(X[i, k])) * Fraction(float(Y[j, k])) for k in range(2)) for j in range(8)] for i in range(6)]
    assert Ps == en.pi_exact(en.Grid(1.0, 7, 1.4, 9), w)


def solved_halfstep(lx, nx, ly, ny, N, direction, seed):
    """A refined R-AC solution (as in Q3) with a random companion: gamma is then small and gamma^ is noise."""
    rng = np.random.default_rng(seed)
    n_f = nx if direction == "Y" else ny
    G = hs.start_set(n_f, N)
    p = hs.HalfStep(lx, nx, ly, ny, direction, G)
    V = hs.solve_ac(p).s.reshape(N, p.n_s - 1).T
    C = V + 1e-3 * rng.standard_normal(V.shape)
    return G, C, V


@pytest.mark.parametrize("lx, nx, ly, ny, N, direction", [(1.0, 12, 1.3, 10, 1, "Y"), (1.0, 16, 0.5, 9, 2, "X"),
                                                          (1.0, 14, 1.0, 14, 3, "Y")])
def test_stage_decomposition_telescopes_and_bounds_hold(lx, nx, ly, ny, N, direction):
    G, C, V = solved_halfstep(lx, nx, ly, ny, N, direction, nx + N)
    dp = ga.double_path(lx, nx, ly, ny, direction, G, C, V)
    lf, nf, ls, ns = (lx, nx, ly, ny) if direction == "Y" else (ly, ny, lx, nx)
    h_f, h_s = Fraction(lf) / nf, Fraction(ls) / ns
    (Gi,), EG = ga.dyadic(G)
    (Vi, Ci), EV = ga.dyadic(V, C)
    gA = ga.gamma_from_coefficients(ga.coefficients_exact(Gi, EG, h_f, Fraction(1)), Vi, Ci, EV, h_s, Fraction(1))
    Ad, Bd, Cd, Fd = dp["coef"]
    cd = ([[Fraction(float(Ad[i, k])) for k in range(N)] for i in range(N)],
          [[Fraction(float(Bd[i, k])) for k in range(N)] for i in range(N)],
          [[Fraction(float(Cd[i, k])) for k in range(N)] for i in range(N)], [Fraction(float(x)) for x in Fd])
    gdc = ga.gamma_from_coefficients(cd, Vi, Ci, EV, h_s, Fraction(1))
    Dl = [[a - b for a, b in zip(Ci[i], Vi[i])] for i in range(N)]
    parts = ga.stages(dp, gA, gdc, Dl, EV, h_s)  # asserts the exact telescoping
    assert sum(parts.values(), Fraction(0)) == Fraction(dp["gamma_hat"]) - gA
    ms = ga.majorant_sums(Gi, EG, Vi, Ci, EV, h_f, h_s, Fraction(1), Fraction(1))
    assert ga.premises_double(dp) is None
    rd = abs(ga._frac_vec_dot(np.abs(dp["rho"]), np.abs(dp["Delta"])))
    B = ga.bound(ms, ga.double_counts(nf, ns, N), ga.U, h_s, abs(Fraction(dp["g"])), rd)
    assert abs(Fraction(dp["gamma_hat"]) - gA) <= B
    assert abs(gdc + parts["operator_assembly"] - gA) <= ga.coefficient_bound(ms, ga.double_counts(nf, ns, N), h_s)
    for prec in (5, 6, 8, 12, 34):
        gD, gabs, rdD = ga.decimal_gamma(lf, nf, ls, ns, G, C, V, 1.0, 1.0, prec)
        uP = Fraction(1, 2 * 10 ** (prec - 1))
        BD = ga.bound(ms, ga.decimal_counts(nf, ns, N), uP, h_s, gabs, rdD)
        assert abs(gD - gA) <= BD, prec


def test_double_path_is_the_q3_formula():
    """gamma^ of double_path equals the expression of r5_cccc_sequence._energy (same code path, bitwise)."""
    G, C, V = solved_halfstep(1.0, 10, 1.3, 9, 2, "Y", 1)
    dp = ga.double_path(1.0, 10, 1.3, 9, "Y", G, C, V)
    p = hs.HalfStep(1.0, 10, 1.3, 9, "Y", G)
    K, rhs = hs.coefficient_system(p)
    rho = rhs - K @ V.T.reshape(-1)
    expected = float(p.l_s / p.n_s) * float(rho @ (C - V).T.reshape(-1))
    assert dp["gamma_hat"] == expected and dp["rho_same"]


# --- negative and not-applicable cases --------------------------------------------------------------------------------------------

def test_non_finite_input_is_rejected():
    with pytest.raises(ValueError):
        ga.dyadic(np.array([[1.0], [np.nan]]))
    with pytest.raises(ValueError):
        ga.dyadic(np.array([[np.inf]]))


def test_gamma_k_is_undefined_for_k_u_at_least_one():
    with pytest.raises(ValueError):
        ga.gam(10, Fraction(1, 10))
    with pytest.raises(ValueError):  # a decimal precision too small for the counts: the bound is not applicable
        ga.bound({"SF": 1, "SA": 1, "SB": 1, "SC": 1}, ga.decimal_counts(40, 40, 3), Fraction(1, 2 * 10), Fraction(1), Fraction(1),
                 Fraction(1))


def test_premises_reject_subnormal_and_non_finite_values():
    G, C, V = solved_halfstep(1.0, 10, 1.3, 9, 1, "Y", 2)
    dp = ga.double_path(1.0, 10, 1.3, 9, "Y", G, C, V)
    assert ga.premises_double(dp) is None
    bad = dict(dp, rho=dp["rho"].copy())
    bad["rho"][0] = 2.0**-1070
    assert ga.premises_double(bad).startswith("P-3")
    bad = dict(dp, Ks=dp["Ks"].copy())
    bad["Ks"][1] = np.inf
    assert ga.premises_double(bad).startswith("P-2")


def test_reading_rules():
    assert ga.read_double(repr(0.1)) == 0.1
    with pytest.raises(ValueError):
        ga.read_double("0.10000000000000001")  # not the repr of the double
    up = hs.decimal_up(0.1, 17)  # a directed 17-digit decimal above the double 0.1
    assert ga.read_bound(up) >= Fraction(0.1) and ga.read_bound(up) == Fraction(up)


def test_test_form_intervals():
    k = ga.k_test_form(Fraction(3), Fraction(2), Fraction(0), Fraction(1, 10))  # |dP - Q| = 1, R = 0.1 -> K_low = 10
    assert k["K_low"] == 10
    assert k["K_max"] == pytest.approx(((0 + (0 + 4 * 0.1 * 2) ** 0.5) / (2 * 0.1)))
    k = ga.k_test_form(Fraction(-5), Fraction(1), Fraction(1), Fraction(1))  # E1: -dP - a = 4, E2: |dP - Q| - a = 5
    assert (k["K_E1"], k["K_E2"], k["K_low"]) == (4, 5, 5)


# --- independence and provenance ---------------------------------------------------------------------------------------------------

def test_no_import_of_the_working_kkvim():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    names = [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
    names += [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    assert not [m for m in names if m == "kkvim" or m.startswith("kkvim.")]


def test_paths_A_and_B_do_not_share_the_operator():
    """Path B uses neither the coefficients nor the 1-D stencils of path A (source inspection)."""
    src = ast.get_source_segment(SCRIPT.read_text(encoding="utf-8"),
                                 next(n for n in ast.parse(SCRIPT.read_text(encoding="utf-8")).body
                                      if isinstance(n, ast.FunctionDef) and n.name == "energies_path_b"))
    for name in ("coefficients_exact", "gamma_from_coefficients", "st_L", "st_D4", "hs.", "en."):
        assert name not in src


def test_q3_data_unchanged_since_the_campaign_commit():
    out = subprocess.run(["git", "cat-file", "-t", ga.Q3_DATA_COMMIT], cwd=ROOT, capture_output=True, text=True)
    if out.stdout.strip() != "commit":
        pytest.skip("Q3 campaign commit not in this clone (shallow checkout)")
    assert ga.provenance(ga.Q3_CSV)["unchanged_since_q3_campaign"]


def test_q3_arithmetic_of_all_rows():
    rows = ga.read_q3_rows(ga.Q3_CSV)
    summary = json.loads((ga.Q3_DIR / "summary.json").read_text(encoding="utf-8"))
    res = ga.q3_arithmetic(rows, summary)
    c = res["checks"]
    assert res["rows"] == 984
    assert c["repr"] == c["defect"] == c["m_num"] == c["part_solution"] == c["gamma_err"] == c["m_s_cR22_rel4u"] == 984
    assert all(res["stats"][f"m_s_cR22_N{N}_equals_summary"] for N in (1, 2, 3))
    assert res["stats"]["m_s_gt_1_by_s"] == {1: 91, 2: 25}
    assert res["stats"]["gamma_le_gamma_hat"] == 984


def test_control_rule_is_deterministic_and_covers_the_required_cases():
    rows = ga.read_q3_rows(ga.Q3_CSV)
    ids = ga.control_rule(rows)
    assert ids == ga.control_rule(list(reversed(rows)))
    assert {i.split("|")[2] for i in ids} == {"N1", "N2", "N3"}
    assert {"s1", "s2", "s8"} <= {i.split("|")[4] for i in ids}
    top = max(rows, key=lambda r: float(r["E1_m_s_cR22"]))
    assert ga.row_id(top) in ids


def test_ci_profile_is_deterministic(tmp_path):
    outs = []
    for k in range(2):
        out = tmp_path / f"run{k}"
        res = subprocess.run([sys.executable, str(SCRIPT), "run", "--profile", "ci", "--out", str(out)], cwd=ROOT,
                             capture_output=True, text=True, timeout=600)
        assert res.returncode == 0, res.stderr[-2000:]
        outs.append((out / "gamma_rows.csv").read_bytes())
        meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
        assert meta["threads"] == {k: "1" for k in ga.THREAD_VARS} and meta["python"] and meta["numpy"]
    assert outs[0] == outs[1]
    rows = list(csv.DictReader(outs[0].decode("utf-8").splitlines()))
    assert len(rows) == 16 and {r["N"] for r in rows} == {"1", "2", "3"} and {r["first"] for r in rows} == {"Y", "X"}
    for r in rows:
        assert r["gamma_hat_bitwise"] == r["gamma_A_equals_q3_exact"] == r["A_equals_B"] == r["dec34_holds"] == "True"
        assert r["B_holds"] == "True" and r["given_consistent"] == "True"
