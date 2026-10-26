"""Tests of the R5-V Q4 B-C reference audit tool (scripts/r5_cccc_q4_bc_reference_audit.py): exact assembly of the half-step system
and the D4 boundary rows, exactness of the residual, the raised-precision solution against the exact rational one, the lambda_min
certificate, directed intervals, the budget rule, the small exact cases of Q3, certificate failure, no import of the working kkvim,
reproducibility. QA research tool."""

import ast
import csv
import ctypes
import importlib.util
import json
import platform
import subprocess
import sys
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
import scipy

from kkvim_reference import r5_cccc_halfstep as hs
from kkvim_reference import r5_cccc_q3_e4e6 as e46

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "r5_cccc_q4_bc_reference_audit.py"
spec = importlib.util.spec_from_file_location("r5_cccc_q4_bc_reference_audit", SCRIPT)
ra = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ra)

SMALL = [(1.0, 8, 1.3, 7, "Y", 1), (1.0, 8, 1.3, 7, "X", 2), (1.0, 10, 0.6, 9, "Y", 3), (2.0, 6, 1.0, 9, "X", 1)]


def halfstep(lx, nx, ly, ny, direction, N, seed=0, start=True):
    n_f = nx if direction == "Y" else ny
    G = hs.start_set(n_f, N) if start else np.random.default_rng(seed).standard_normal((n_f - 1, N))
    return hs.HalfStep(lx, nx, ly, ny, direction, G)


# --- exact system --------------------------------------------------------------------------------------------------------------------

def test_d4_stencil_boundary_rows():
    m = 7
    cols = [ra._st_D4([int(i == k) for i in range(m)]) for k in range(m)]  # columns of D4 (symmetric)
    D4 = [[cols[k][i] for k in range(m)] for i in range(m)]
    assert D4[0][:3] == [7, -4, 1] and D4[-1][-3:] == [1, -4, 7]
    assert D4[3][1:6] == [1, -4, 6, -4, 1]
    L = [[ra._st_L([int(i == k) for i in range(m)])[j] for k in range(m)] for j in range(m)]
    L2 = [[sum(L[i][t] * L[t][j] for t in range(m)) for j in range(m)] for i in range(m)]
    E = [[2 * (i == j and i in (0, m - 1)) for j in range(m)] for i in range(m)]
    assert D4 == [[L2[i][j] + E[i][j] for j in range(m)] for i in range(m)]  # D4 = L^2 + 2 (e1 e1^T + em em^T)


@pytest.mark.parametrize("args", SMALL)
@pytest.mark.parametrize("start", [True, False])
def test_exact_assembly_equals_the_accepted_band_and_coefficients(args, start):
    p = halfstep(*args, start=start)
    coef = ra.exact_coefficients(p)
    q3 = hs.exact_coefficients(p)
    assert all(coef[t] == q3[t] for t in range(4))
    assert ra.exact_band(p, coef) == hs._exact_band_matrix(p, q3)


@pytest.mark.parametrize("args", SMALL)
def test_exact_residual_equals_rhp(args):
    p = halfstep(*args)
    coef = ra.exact_coefficients(p)
    N, m = p.N, p.n_s - 1
    rows = ra.band_rows(ra.exact_band(p, coef), N * m)
    b = ra.rhs_exact(p, coef)
    x_tf = np.random.default_rng(1).standard_normal(N * m)
    r = ra.residual(rows, b, ra.to_node_term([Fraction(float(v)) for v in x_tf], N, m))
    assert ra.to_term_first_frac(r, N, m) == hs.residual_exact(p, x_tf)


# --- raised-precision reference and certificate ----------------------------------------------------------------------------------

@pytest.mark.parametrize("args", SMALL)
def test_reference_is_within_its_certified_bound_of_the_exact_solution(args):
    p = halfstep(*args)
    coef = ra.exact_coefficients(p)
    N, m = p.N, p.n_s - 1
    n = N * m
    band = ra.exact_band(p, coef)
    rows, b = ra.band_rows(band, n), ra.rhs_exact(p, coef)
    for prec in (12, 20, 40):
        x, r, log = ra.refine(rows, b, ra.dec_cholesky(band, n, 3 * N - 1, prec))
        lam = hs.certify_lambda_min(p, ra.tau_trial(p), hs.exact_coefficients(p))
        b2 = ra.sqrt_up(ra.norm2_sq(r)) / Fraction(lam)
        K, be = e46.exact_system(p)
        x_star = ra.to_node_term(e46._solve_many(K, [be])[0], N, m)
        err2 = ra.norm2_sq([a - c for a, c in zip(x, x_star, strict=True)])
        assert err2 <= b2 * b2, prec
        assert log[-1] <= log[0]


@pytest.mark.parametrize("args", SMALL)
def test_lambda_certificate_is_a_true_lower_bound(args):
    p = halfstep(*args)
    lam = hs.certify_lambda_min(p, ra.tau_trial(p), hs.exact_coefficients(p))
    K, _ = e46.exact_system(p)
    shifted = [[K[i][j] - (Fraction(lam) if i == j else 0) for j in range(len(K))] for i in range(len(K))]
    assert e46._spd_exact(shifted)  # K - lambda_low I is positive definite (exact LDL^T)
    assert ra.gershgorin_up(ra.band_rows(ra.exact_band(p, ra.exact_coefficients(p)), p.size)) >= Fraction(lam)


def test_directed_intervals():
    rng = np.random.default_rng(5)
    for _ in range(200):
        x = Fraction(int(rng.integers(1, 10**12)), int(rng.integers(1, 10**12))) * Fraction(10) ** int(rng.integers(-40, 40))
        assert ra.sqrt_down(x) ** 2 <= x <= ra.sqrt_up(x) ** 2
        assert Fraction(Decimal(ra.dec_down(x))) <= x <= Fraction(Decimal(ra.dec_up(x)))
        assert Fraction(ra.down(x)) <= x <= Fraction(ra.up(x))


def test_budget_rule():
    assert ra.budget_ok(Fraction(1), Fraction(100))
    assert not ra.budget_ok(Fraction(1), Fraction(99))
    assert not ra.budget_ok(Fraction(0), Fraction(0))  # an error not separated from 0 is never resolved
    assert ra.budget_ok(Fraction(0), Fraction(1, 10**30))


# --- whole points ------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("lx, nx, ly, ny", [(1.0, 8, 1.3, 7), (1.0, 10, 0.6, 9), (1.0, 12, 1.0, 13)])
@pytest.mark.parametrize("direction, N, qr", [("Y", 1, False), ("X", 2, True), ("Y", 3, False)])
def test_small_exact_cases_of_q3(lx, nx, ly, ny, direction, N, qr):
    p = halfstep(lx, nx, ly, ny, direction, N)
    out = ra.analyse_point(p, qr, None, True)
    if N * (out["n_s"] - 1) <= ra.EXACT_LIMIT:
        assert out["exact_ref_within_bound"] is True and out["exact_bc_error_in_interval"] is True
    assert out["rh_consistent"] is True and out["band_equal_q1b"] and out["coefficients_equal_q3"]
    assert out["result"] == "RESOLVED (both norms)"


def test_certificate_failure_gives_unresolved(monkeypatch):
    p = halfstep(1.0, 8, 1.3, 7, "Y", 2)
    monkeypatch.setattr(ra, "tau_trial", lambda ps: 1e30)  # trial value far above lambda_min: K - t I not positive definite
    out = ra.analyse_point(p, True, None, False)
    assert out["lambda_low"] is None and out["result"].startswith("UNRESOLVED: no certificate")
    assert "budget_set" not in out


def test_measured_error_zero_is_unresolved(monkeypatch):
    """B-C replaced by the reference itself rounded to double: the measured error is not separated from the reference error."""
    p = halfstep(1.0, 8, 1.3, 7, "Y", 1)
    real = ra.bc.run_halfstep

    def fake(ps, qr=True, capacitance="modal"):
        rec = real(ps, qr=qr, capacitance=capacitance)
        K, b = e46.exact_system(rec.solved)
        x_star = e46._solve_many(K, [b])[0]
        rec.bc.s = np.array([float(v) for v in x_star])
        return rec

    monkeypatch.setattr(ra.bc, "run_halfstep", fake)
    out = ra.analyse_point(p, False, None, False)
    # the double rounding of x* is a genuine (tiny) error: it is either resolved with the certified bound or UNRESOLVED, never a
    # false bound; the exact error lies in the interval in every case
    assert out["exact_bc_error_in_interval"] is True


# --- independence, provenance, reproducibility -------------------------------------------------------------------------------------

def test_no_import_of_the_working_kkvim():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    names = [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
    names += [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    assert not [m for m in names if m == "kkvim" or m.startswith("kkvim.")]


def test_reference_does_not_call_bc():
    """The reference functions use neither r5_cccc_bc nor a DST (source inspection)."""
    src = SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(src)
    for name in ("exact_coefficients", "exact_band", "dec_cholesky", "dec_solve", "refine", "residual", "tau_trial"):
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        seg = ast.get_source_segment(src, fn)
        assert "bc." not in seg and "dst" not in seg.lower(), name


def test_e4_sample_is_the_q3_sample(tmp_path):
    arrays = ra.Q3_ARRAYS_DEFAULT
    pts = ra.e4_points("full", arrays)
    assert len(pts) == 476
    with open(ra.Q3_E4, encoding="utf-8") as f:
        assert sum(1 for _ in csv.DictReader(f)) == 476


def test_q3_e4_unchanged_since_the_campaign():
    out = subprocess.run(["git", "cat-file", "-t", ra.Q3_DATA_COMMIT], cwd=ROOT, capture_output=True, text=True)
    if out.stdout.strip() != "commit":
        pytest.skip("Q3 campaign commit not in this clone")
    rel = "docs/results/r5_cccc_q3_prequalification/e4.csv"
    now = subprocess.run(["git", "hash-object", rel], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    then = subprocess.run(["git", "rev-parse", f"{ra.Q3_DATA_COMMIT}:{rel}"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    assert now == then


Q3_CORE = "Haswell"  # runtime OpenBLAS kernel on the Q3 processor (Intel64 Family 6 Model 183); not recorded in the Q3 meta.json


def _openblas_core() -> str | None:
    """Runtime kernel of the NumPy OpenBLAS (DYNAMIC_ARCH: chosen by the CPU or OPENBLAS_CORETYPE); None if not determinable."""
    try:
        lib = ctypes.CDLL(str(min((Path(np.__file__).resolve().parents[1] / "numpy.libs").glob("libscipy_openblas64_*"))))
        f = lib.scipy_openblas_get_corename64_
        f.restype = ctypes.c_char_p
        return f().decode()
    except (OSError, AttributeError, ValueError):
        return None


def _in_q3_environment() -> bool:
    """The double B-C solution (numpy LAPACK QR/solve, BLAS products) depends on the BLAS kernel and build: the Q3 row is reproduced
    bitwise only in the environment of the Q3 campaign (its meta.json: OS, machine, processor, NumPy, SciPy, BLAS) with its kernel."""
    meta = json.loads((ra.Q3_DIR / "meta.json").read_text(encoding="utf-8"))
    blas = np.show_config(mode="dicts")["Build Dependencies"]["blas"]
    return (platform.system() == meta["platform"].split("-")[0] and platform.machine() == meta["machine"]
            and platform.processor() == meta["processor"] and np.__version__ == meta["numpy"] and scipy.__version__ == meta["scipy"]
            and f"{blas['name']} {blas['version']}" == meta["blas"] and _openblas_core() == Q3_CORE)


def _exact_band_ldl(band: dict, n: int, pb: int, shift=Fraction(0), rhs=None):
    """Exact (rational) symmetric Gaussian elimination of the band of K - shift I without pivoting: the pivots and, with ``rhs``, the
    exact solution. All pivots > 0 <=> K - shift I is positive definite. Test path, independent of the decimal reference and of B-C."""
    U = dict(band)
    for i in range(n):
        U[(i, i)] = U.get((i, i), Fraction(0)) - shift
    y = None if rhs is None else list(rhs)
    piv = []
    for k in range(n):
        piv.append(U[(k, k)])
        if piv[-1] <= 0:
            return piv, None
        for i in range(k + 1, min(n, k + pb + 1)):
            f = U.get((k, i), 0) / piv[-1]
            if f:
                for j in range(i, min(n, k + pb + 1)):
                    U[(i, j)] = U.get((i, j), Fraction(0)) - f * U.get((k, j), 0)
                if y is not None:
                    y[i] -= f * y[k]
    if y is None:
        return piv, None
    x = [Fraction(0)] * n
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - sum(U.get((i, j), 0) * x[j] for j in range(i + 1, min(n, i + pb + 1)))) / U[(i, i)]
    return piv, x


def test_one_q3_certified_point_is_reproduced_and_resolved(monkeypatch):
    """The pair 47 x 53, N = 2, X, with QR (s = 1), Q3 budget NOT_MET.

    Every environment: the exact system and the lambda_low certificate are checked independently (exact LDL^T of K - lambda_low I),
    x* is solved exactly (residual 0), the reference lies within its bound of x*, the exact error of THIS B-C solution lies in both new
    intervals, the budget 1/100 holds with the printed directed values. Only in the Q3 environment: B-C reproduces the Q3 row bitwise
    and the new intervals intersect the Q3 intervals (elsewhere B-C gives another double solution, whose error the Q3 interval does not
    enclose by construction)."""
    with open(ra.Q3_E4, encoding="utf-8") as f:
        row = next(r for r in csv.DictReader(f) if r["grid"] == "1x1:47x53" and r["s"] == "1" and r["N"] == "2"
                   and r["direction"] == "X" and r["qr"] == "True")
    assert row["budget_set"].startswith("NOT_MET") and row["budget_field"].startswith("NOT_MET")
    recs, refs = [], []
    real_run, real_refine = ra.bc.run_halfstep, ra.refine
    monkeypatch.setattr(ra.bc, "run_halfstep", lambda *a, **k: recs.append(real_run(*a, **k)) or recs[-1])
    monkeypatch.setattr(ra, "refine", lambda *a, **k: refs.append(real_refine(*a, **k)) or refs[-1])
    p = halfstep(1.0, 47, 1.0, 53, "X", 2)
    out = ra.analyse_point(p, True, row, False)
    assert out["result"] == "RESOLVED (both norms)" and out["budget_set"] == "SATISFIED" and out["budget_field"] == "SATISFIED"
    assert out["coefficients_equal_q3"] and out["band_equal_q1b"] and out["refine_converged"]
    # independent exact path on the solved set of this run
    ps = recs[-1].solved
    N, m = ps.N, ps.n_s - 1
    n, pb = N * m, 3 * N - 1
    coef = ra.exact_coefficients(ps)
    band = ra.exact_band(ps, coef)
    rows, b = ra.band_rows(band, n), ra.rhs_exact(ps, coef)
    lam = Fraction(out["lambda_low"])
    assert 0 < lam <= ra.gershgorin_up(rows) and all(d > 0 for d in _exact_band_ldl(band, n, pb, shift=lam)[0])
    _, x_star = _exact_band_ldl(band, n, pb, rhs=b)
    assert not any(ra.residual(rows, b, x_star))
    D = lambda key: Fraction(Decimal(out[key]))  # noqa: E731
    assert ra.norm2_sq([a - c for a, c in zip(refs[-1][0], x_star, strict=True)]) <= D("ref_bound_2_up") ** 2
    s_bc = ra.to_node_term([Fraction(float(v)) for v in recs[-1].bc.s], N, m)
    e = [a - c for a, c in zip(s_bc, x_star, strict=True)]
    eK2 = -sum(a * r for a, r in zip(e, ra.residual(rows, b, s_bc), strict=True))  # e^T K e = -e^T r, since K e = -r
    assert D("set_error_2_down") ** 2 <= ra.norm2_sq(e) <= D("set_error_2_up") ** 2
    assert D("field_error_a_down") ** 2 <= Fraction(ps.l_s) / ps.n_s * eK2 <= D("field_error_a_up") ** 2
    assert 100 * D("ref_bound_2_up") <= D("set_error_2_down") and 100 * D("ref_bound_field_a_up") <= D("field_error_a_down")
    if _in_q3_environment():
        assert out["q3_reproduced"] and out["q3_set_interval_intersects"] and out["q3_field_interval_intersects"]


def test_ci_profile_is_deterministic(tmp_path):
    outs = []
    for k in range(2):
        out = tmp_path / f"run{k}"
        res = subprocess.run([sys.executable, str(SCRIPT), "run", "--profile", "ci", "--out", str(out)], cwd=ROOT,
                             capture_output=True, text=True, timeout=900)
        assert res.returncode == 0, res.stderr[-2000:]
        outs.append((out / "e4_reference.csv").read_bytes())
        meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
        assert meta["threads"] == {k: "1" for k in ra.THREAD_VARS} and meta["precision_ladder"] == list(ra.P_LADDER)
    assert outs[0] == outs[1]
    rows = list(csv.DictReader(outs[0].decode("utf-8").splitlines()))
    assert len(rows) == 6 and all(r["result"] == "RESOLVED (both norms)" for r in rows)
