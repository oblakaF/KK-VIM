"""Tests of the C-3+ E-4 audit (scripts/r5_cccc_q4_c_assembly_e4_audit.py): the exact identity of the symmetric form, one correct
rounding of the exact stencils, the theorem C-3+ with k = m + 13 and its exact check (incl. a detected violation), premises
(subnormal, overflow: NOT_APPLICABLE), positive definiteness of the computed C, the replacement of C only, whole points against the
exact rational solution, no change of the prototype, no import of the working kkvim, reproducibility. QA research tool."""

import ast
import csv
import importlib.util
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from kkvim_reference import r5_cccc_bc as bc
from kkvim_reference import r5_cccc_halfstep as hs
from kkvim_reference import r5_cccc_q3_e4e6 as e46

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "r5_cccc_q4_c_assembly_e4_audit.py"
spec = importlib.util.spec_from_file_location("r5_cccc_q4_c_assembly_e4_audit", SCRIPT)
ca = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ca)

U = Fraction(1, 2**53)


def halfstep(G, lx=1.0, nx=None, ly=1.3, ny=9, direction="Y"):
    nx = nx or G.shape[0] + 1
    return hs.HalfStep(lx, nx, ly, ny, direction, G)


SETS = {"sin2": lambda m, N, rng: hs.start_set(m + 1, N),
        "random": lambda m, N, rng: rng.standard_normal((m, N)),
        "rough": lambda m, N, rng: rng.standard_normal((m, N)) * 2.0 ** rng.integers(-20, 20, size=(m, N))}


@pytest.mark.parametrize("kind", SETS)
@pytest.mark.parametrize("m, N", [(2, 1), (7, 2), (30, 3), (127, 2)])
def test_symmetric_form_is_exactly_C(kind, m, N):
    """h (LG)^T (LG) + (2/h^3)(g1 g1^T + gm gm^T) = h G^T D4 G exactly (NS 3.1, 4.2)."""
    rng = np.random.default_rng(m * 10 + N)
    G = SETS[kind](m, N, rng)
    p = halfstep(G, lx=0.7)
    h = Fraction(p.l_f) / p.n_f
    Gf = [[Fraction(float(G[i, j])) for j in range(N)] for i in range(m)]
    Z3 = [[(Gf[i - 1][j] if i > 0 else 0) - 2 * Gf[i][j] + (Gf[i + 1][j] if i < m - 1 else 0) for j in range(N)] for i in range(m)]
    X = [[z / h**2 for z in row] for row in Z3]
    C = [[h * sum(X[t][i] * X[t][j] for t in range(m)) + 2 / h**3 * (Gf[0][i] * Gf[0][j] + Gf[-1][i] * Gf[-1][j])
          for j in range(N)] for i in range(N)]
    assert C == hs.exact_coefficients(p)[2]


@pytest.mark.parametrize("stencil", ["Z3", "Z5"])
def test_exact_stencil_is_rounded_once(stencil):
    rng = np.random.default_rng(4)
    G = rng.standard_normal((40, 3)) * 2.0 ** rng.integers(-30, 30, size=(40, 3))
    Y = ca._exact_stencil_rounded(G, getattr(ca, stencil))
    m = G.shape[0]
    for j in range(3):
        g = [Fraction(float(v)) for v in G[:, j]]
        if stencil == "Z3":
            exact = [(g[i - 1] if i else 0) - 2 * g[i] + (g[i + 1] if i < m - 1 else 0) for i in range(m)]
        else:
            exact = [v * 1 for v in hs._apply_D4_exact(g, Fraction(1))]  # (7, -4, 1) rows, interior (1, -4, 6, -4, 1)
        for i in range(m):
            assert abs(Fraction(float(Y[i, j])) - exact[i]) <= U * abs(exact[i])  # one correct rounding
            assert float(Y[i, j]) == float(exact[i])


@pytest.mark.parametrize("kind", SETS)
@pytest.mark.parametrize("m, N", [(5, 1), (31, 2), (255, 3), (1023, 2)])
def test_c3plus_theorem_holds_and_the_checker_detects_violations(kind, m, N):
    rng = np.random.default_rng(m + N)
    G = SETS[kind](m, N, rng)
    p = halfstep(G, lx=1.0 / 3)
    parts = ca.c3plus(p)
    assert ca.premises_c3plus(parts) is None
    Cx = hs.exact_coefficients(p)[2]
    chk = ca.check_c3plus(parts["C"], Cx, m + ca.K_EXTRA)
    assert chk["elementwise"] and chk["frobenius"]
    bad = ca.check_c3plus(parts["C"] * (1 + 2.0**-20), Cx, m + ca.K_EXTRA)  # a perturbation far above gamma_k
    assert not bad["elementwise"]


def test_premises_subnormal_and_overflow_are_not_applicable():
    G = hs.start_set(20, 2)
    assert ca.premises_c3plus(ca.c3plus(halfstep(G * 2.0**-1060))) is not None  # subnormal stencil values / products
    assert ca.premises_c3plus(ca.c3plus(halfstep(G * 2.0**600))) is not None  # Gram sums overflow
    assert ca.premises_c3plus(ca.c3plus(halfstep(G))) is None


def test_positive_definiteness_check():
    assert ca.spd_exact(np.array([[2.0, 1.0], [1.0, 2.0]]))
    assert not ca.spd_exact(np.array([[1.0, 2.0], [2.0, 1.0]]))
    assert not ca.spd_exact(np.array([[1.0, 1.0], [1.0, 1.0]]))  # semidefinite is not positive definite
    G = hs.start_set(30, 3)
    assert ca.spd_exact(ca.c3plus(halfstep(G))["C"])


def test_only_C_is_replaced():
    p = halfstep(hs.start_set(16, 2), nx=16, ny=12)
    A, B, C, F = bc.bc_coefficients(p)
    old = bc.solve_bc(p)
    with ca.replaced_c(C.copy()):
        same = bc.solve_bc(p)
        A2, B2, C2, F2 = bc.bc_coefficients(p)
    assert np.array_equal(old.s, same.s)
    assert np.array_equal(A2, A) and np.array_equal(B2, B) and np.array_equal(F2, F) and np.array_equal(C2, C)
    assert bc.bc_coefficients is not None and bc.bc_coefficients.__name__ == "bc_coefficients"  # restored


@pytest.mark.parametrize("lx, nx, ly, ny, direction, N, qr", [(1.0, 8, 1.3, 7, "Y", 1, False), (1.0, 10, 0.6, 9, "X", 2, True),
                                                               (1.0, 12, 1.0, 13, "Y", 3, False)])
def test_whole_point_against_the_exact_rational_solution(lx, nx, ly, ny, direction, N, qr):
    n_f = nx if direction == "Y" else ny
    p = hs.HalfStep(lx, nx, ly, ny, direction, hs.start_set(n_f, N))
    out = ca.analyse_point(p, qr, None)
    assert out["result"] == "COMPLETE" and out["c3plus_inequality"] == "PASS"
    rec = bc.run_halfstep(p, qr=qr)
    ps = rec.solved
    if ps.size <= e46.EXACT_LIMIT:
        K, b = e46.exact_system(ps)
        xs = e46._solve_many(K, [b])[0]
        A, B, C, F = bc.bc_coefficients(ps)
        with ca.replaced_c(ca.c3plus(ps)["C"]):
            s_new = bc.solve_bc(ps).s
        err = sum((Fraction(float(a)) - c) ** 2 for a, c in zip(s_new, xs))
        from decimal import Decimal

        lo, hi = Fraction(Decimal(out["NEW_set_down"])), Fraction(Decimal(out["NEW_set_up"]))
        assert lo * lo <= err <= hi * hi


def test_prototype_and_inputs_unchanged():
    for rel in ("src/kkvim_reference/r5_cccc_bc.py", "docs/results/r5_cccc_q4_bc_reference_audit/e4_reference.csv",
                "scripts/r5_cccc_q4_bc_reference_audit.py"):
        out = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", rel], cwd=ROOT)
        assert out.returncode == 0, rel


def test_no_import_of_the_working_kkvim():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    names = [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
    names += [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    assert not [m for m in names if m == "kkvim" or m.startswith("kkvim.")]


def test_ci_profile_is_deterministic(tmp_path):
    outs = []
    for k in range(2):
        out = tmp_path / f"run{k}"
        res = subprocess.run([sys.executable, str(SCRIPT), "run", "--profile", "ci", "--out", str(out)], cwd=ROOT,
                             capture_output=True, text=True, timeout=900)
        assert res.returncode == 0, res.stderr[-2000:]
        outs.append((out / "rows.csv").read_bytes())
    assert outs[0] == outs[1]
    rows = list(csv.DictReader(outs[0].decode("utf-8").splitlines()))
    assert len(rows) == 6 and all(r["c3plus_inequality"] == "PASS" and r["result"] == "COMPLETE" for r in rows)
