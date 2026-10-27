"""Tests of the numerical-rank audit N = 2, 3 (scripts/r5_cccc_q4_rank_n23_audit.py): exact dyadic integers, exact rank of stored
factors with certificates (minor witness, null vector) against an independent brute-force and Fraction elimination on small
matrices, matrices of known full and deficient rank, rounding of a deficient construction to an exactly full-rank stored matrix,
certified eigenvalue enclosures (Sturm), exact rho_N / kappa_rep against SVD, the inequality R4 5.4.1 item 1, the fixed control
constructions, the paired AC/BC rows of the CI profile (bitwise Q3 replica, reproduced Q3 state), no change of the used modules and
data, no import of the working kkvim, deterministic CI profile. QA research tool."""

import ast
import csv
import importlib.util
import itertools
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from kkvim_reference import r5_cccc_halfstep as hs

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "r5_cccc_q4_rank_n23_audit.py"
spec = importlib.util.spec_from_file_location("r5_cccc_q4_rank_n23_audit", SCRIPT)
rk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rk)


def _rank_of_product(rng, m, n, r):
    """Integer m x n matrix of exact rank r (product of random integer factors; rank checked by Fraction elimination)."""
    while True:
        A = rng.integers(-5, 6, size=(m, r)) @ rng.integers(-5, 6, size=(r, n))
        if rk.rank_fraction([[Fraction(int(v)) for v in row] for row in A]) == r:
            return A.astype(float)


def test_dyadic_integers_are_exact():
    rng = np.random.default_rng(1)
    M = rng.standard_normal((9, 3)) * np.array([1e-12, 1.0, 3e7])
    M[2, 1] = 0.0
    M[4, 0] = 5e-324  # subnormal
    Mi, E = rk.dyadic_int(M)
    for i in range(9):
        for j in range(3):
            assert Fraction(Mi[i][j]) * Fraction(2) ** E == Fraction(float(M[i, j]))


@pytest.mark.parametrize("seed", range(12))
def test_exact_rank_against_brute_force_small(seed):
    """Independent control on small matrices: own certified rank == brute force over all minors == Fraction Gauss-Jordan."""
    rng = np.random.default_rng(seed)
    N = 2 + seed % 2
    r = seed % (N + 1)  # ranks 0..N
    M = _rank_of_product(rng, 7, N, r) if r > 0 else np.zeros((7, N))
    M = M * 2.0 ** rng.integers(-30, 30)  # exact scaling
    c = rk.exact_rank_cert(M)
    assert c["ok"] and c["rank"] == r == rk.rank_brute_force(M) == hs.exact_rank(M)
    if r == N:
        assert c["det_two_paths_equal"] and c["det_nonzero"] and c["certificate"].startswith("MINOR")
    else:
        z = c["null"]
        assert any(z) and np.all(M @ np.array(z, dtype=float) == 0)


def test_known_deficient_constructions_are_detected_exactly():
    rng = np.random.default_rng(7)
    v1, v2 = rng.integers(-9, 10, 12).astype(float), rng.integers(-9, 10, 12).astype(float)
    for M in (np.column_stack([v1, v2, v1 + v2]),  # exact combination (integers)
              np.column_stack([v1, v2, v1]),  # repeated column
              np.column_stack([v1, v2, np.zeros(12)]),  # zero column
              np.column_stack([v1, 2.0 * v1])):  # N = 2, proportional
        c = rk.exact_rank_cert(M)
        assert c["ok"] and c["rank"] == M.shape[1] - 1 and c["certificate"].startswith("NULL")


def test_rounding_makes_a_deficient_construction_exactly_full_rank():
    """v3 = v1/3 + v2 (exact construction of rank 2) stored after correct rounding has EXACT rank 3."""
    v1 = np.array([1.0, 2.0, 5.0, 7.0, 11.0, 13.0, 4.0])
    v2 = np.array([3.0, -1.0, 2.0, 0.5, 1.0, -2.0, 6.0])
    src = [[Fraction(a), Fraction(b), Fraction(a) / 3 + Fraction(b)] for a, b in zip(v1, v2)]
    assert rk.rank_fraction(src) == 2
    stored = np.array([[float(x) for x in row] for row in src])
    c = rk.exact_rank_cert(stored)
    assert c["ok"] and c["rank"] == 3 == rk.rank_brute_force(stored)


@pytest.mark.parametrize("seed", range(6))
def test_bareiss_det_and_mod_p(seed):
    rng = np.random.default_rng(100 + seed)
    for r in range(1, 6):
        A = _rank_of_product(rng, 6, 5, r)
        Ai, _ = rk.dyadic_int(A)
        assert rk.bareiss(Ai)[0] == r
        assert rk.rank_mod_p(Ai)[0] <= r
    S = rng.integers(-20, 21, size=(4, 4))
    assert rk.det_int(S.tolist()) == rk.det_leibniz(S.tolist())


def test_certified_roots_contain_the_exact_eigenvalues():
    """M = P D P^-1 with a unimodular integer P: eigenvalues are the integers of D (incl. 0, repeated and widely spread)."""
    P = np.array([[1, 2, 0], [0, 1, 3], [0, 0, 1]])
    Pinv = np.round(np.linalg.inv(P)).astype(int)
    for d in ((5, 3, 1), (7, 7, 2), (4, 0, 9), (2**200, 3, 1)):
        D = np.diag(np.array(d, dtype=object))
        M = (P.astype(object) @ D @ Pinv.astype(object)).tolist()
        res = rk.roots_certified(M)
        assert res["ok"]
        pos = sorted({x for x in d if x > 0})
        assert res["zero_root"] == (0 in d)
        assert len(res["roots"]) == len(pos)
        for (lo, hi), x in zip(res["roots"], pos):
            assert lo <= x <= hi and (hi - lo) <= lo / 2**rk.ROOT_BITS


def test_sqrt_bounds():
    for x in (Fraction(2), Fraction(1, 3), Fraction(10**40 + 1, 7), Fraction(1, 2**300)):
        lo, hi = rk.sqrt_bounds(x)
        assert lo * lo <= x <= hi * hi and hi - lo <= lo / 2**80


@pytest.mark.parametrize("N", (2, 3))
def test_exact_rho_kappa_against_svd_and_R4_inequality(N):
    rng = np.random.default_rng(N)
    U = rng.standard_normal((20, N))
    V = rng.standard_normal((15, N)) @ np.diag([1.0, 1e-3, 1e-5][:N])
    ex = rk.field_exact(U, V, Fraction(1, 7), Fraction(1, 5))
    assert not ex["_fails"]
    assert ex["U_rank_exact"] == ex["V_rank_exact"] == ex["field_rank_exact"] == ex["field_rank_gram_path"] == N
    assert ex["field_minor_equals_product"] and ex["det_gram_product_positive"]
    s = np.linalg.svd(U @ V.T, compute_uv=False)
    assert abs(s[N - 1] / s[0] - ex["rho_exact_down"]) <= 1e-9 * ex["rho_exact_down"]
    kap = np.linalg.norm(U, 2) * np.linalg.norm(V, 2) / s[0]
    assert abs(kap - ex["kappa_exact_up"]) <= 1e-12 * kap
    assert ex["R4_541_inequality"] == "HOLDS" and ex["E7_exact_frame_holds"]
    # deficient V: exact rank N - 1 of the field and rho = 0 exactly
    Vd = V.copy()
    Vd[:, -1] = Vd[:, 0]
    ex = rk.field_exact(U, Vd, Fraction(1), Fraction(1))
    assert not ex["_fails"] and ex["field_rank_exact"] == N - 1 and ex["rho_exact_up"] == 0.0


def test_control_constructions_fixed_and_of_designed_rank():
    a, b = rk.control_cases(), rk.control_cases()
    assert len(a) == 2 * (4 * len(rk.A_PARAMS) + len(rk.K_PARAMS) + 3) == 94
    for x, y in zip(a, b):
        assert x["family"] == y["family"] and np.array_equal(x["U"], y["U"]) and np.array_equal(x["V"], y["V"])
    for c in a:
        assert rk.rank_fraction(rk._mm(c["U_src"], rk._tr(c["V_src"]))) == c["math_rank"]
        if c["storage"].startswith("exact"):  # exactly stored: the doubles are the exact source
            assert all(Fraction(float(c["U"][i, j])) == c["U_src"][i][j] for i, j in itertools.product(range(c["U"].shape[0]),
                                                                                                         range(c["N"])))
            assert all(Fraction(float(c["V"][i, j])) == c["V_src"][i][j] for i, j in itertools.product(range(c["V"].shape[0]),
                                                                                                         range(c["N"])))


def test_controls_exact_ranks_and_rounding_effect():
    res = {r["id"]: r for r in rk.control_job(rk.control_cases())}
    assert all(r["status"] == "COMPLETE" for r in res.values())
    for r in res.values():
        if r["family"] in ("F2_DEFICIENT_EXACT_UNIMOD", "F8_DEFICIENT_REPEAT_SIN2"):
            assert r["field_rank_exact"] == r["N"] - 1 and r["rho_exact_up"] == 0.0  # exactly deficient stored arrays
        if r["family"] in ("F1_FULL_EXACT_UNIMOD", "F3_FULL_TINY_SIGMA", "F5_FULL_ROUNDED_T"):
            assert r["field_rank_exact"] == r["N"]
        if r["family"] in ("F4_DEFICIENT_ROUNDED_T", "F6_DEFICIENT_ROUNDED_COMB", "F7_DEFICIENT_SIN2_SUM"):
            assert r["math_rank_exact_source"] == r["N"] - 1  # mathematically deficient; the stored rank is recorded, not assumed
        assert r["U_rank_brute_force"] == r["U_rank_exact"] and r["V_rank_brute_force"] == r["V_rank_exact"]
    # a positive computed sigma_N is not a proof of rank: exactly deficient stored factors give rho^ > 0
    assert all(res[k]["p1_rho_N"] > 0 for k in res if res[k]["family"] == "F2_DEFICIENT_EXACT_UNIMOD")


@pytest.fixture(scope="module")
def ci_rows(tmp_path_factory):
    out = tmp_path_factory.mktemp("rank_ci")
    res = subprocess.run([sys.executable, str(SCRIPT), "run", "--profile", "ci", "--out", str(out)], cwd=ROOT, capture_output=True,
                         text=True, timeout=900, check=False)
    assert res.returncode == 0, res.stderr[-2000:]
    return (out / "rows.csv").read_bytes()


def test_ci_profile_rows(ci_rows):
    rows = list(csv.DictReader(ci_rows.decode("utf-8").splitlines()))
    fields = [r for r in rows if r["row_type"] in ("AC", "BC")]
    assert len(fields) == 24 and len(rows) == 24 + 94 and all(r["status"] == "COMPLETE" for r in rows)
    for r in fields:
        N = int(r["N"])
        assert int(r["U_rank_exact"]) == int(r["V_rank_exact"]) == int(r["field_rank_exact"]) == N
        assert r["U_rank_certificate"].startswith("MINOR") and r["field_minor_equals_product"] == "True"
        assert r["R4_541_inequality"] == "HOLDS" and int(r["flW_rank_exact"]) > N
        assert float(r["rho_exact_down"]) <= float(r["rho_exact_up"])
        if r["row_type"] == "AC":
            assert r["q3_p1_bitwise"] == "True" and r["halfstep_rank_certificate"].startswith("CERTIFIED")
        else:
            assert r["only_C_replaced"] == "True" and float(r["cond_S"]) > 0
            assert r["q3_state_reproduced_bitwise"] in ("True", "NOT_APPLICABLE (s = 1: U~ = QR of T_N)")


def test_ci_profile_is_deterministic(ci_rows, tmp_path):
    res = subprocess.run([sys.executable, str(SCRIPT), "run", "--profile", "ci", "--out", str(tmp_path)], cwd=ROOT,
                         capture_output=True, text=True, timeout=900, check=False)
    assert res.returncode == 0, res.stderr[-2000:]
    assert (tmp_path / "rows.csv").read_bytes() == ci_rows


def test_used_modules_and_data_unchanged():
    for rel in ("src/kkvim_reference", "src/kkvim", "docs/results/r5_cccc_q3_prequalification", "docs/results/r5_cccc_q4_gamma_audit",
                "docs/results/r5_cccc_q4_energy_bc", "docs/results/r5_cccc_q4_c_assembly_e4",
                "scripts/r5_cccc_q4_gamma_audit.py", "scripts/r5_cccc_q4_energy_bc_audit.py",
                "scripts/r5_cccc_q4_c_assembly_e4_audit.py", "scripts/r5_cccc_q3_prequalification.py"):
        out = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", rel], cwd=ROOT, check=False)
        assert out.returncode == 0, rel


def test_no_import_of_the_working_kkvim_and_no_library_rank():
    text = SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(text)
    names = [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
    names += [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    assert not [m for m in names if m == "kkvim" or m.startswith("kkvim.")]
    # numpy.linalg.matrix_rank appears only as the selection aid of the control generator, never for an exact claim
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Attribute) and n.attr == "matrix_rank"]
    assert len(calls) == 1
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_int_matrix")
    assert any(isinstance(n, ast.Attribute) and n.attr == "matrix_rank" for n in ast.walk(fn))
