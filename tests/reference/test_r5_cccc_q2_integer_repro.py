"""R5-V Q2 (QA): checks of the independent integer D2 reproduction (scripts/r5_cccc_q2_integer_repro.py).

Exact checks (no tolerances): the 13-point stencil and the CCCC ghost rule against a third, apply-based formulation; integer
assemblies; symmetry; scaling A = n^4 Z and alpha = y_c / n^4; the exact residual against fractions; exact power-of-two
scaling at dyadic n (V1 = V3 and V2 = V4 bitwise, same code path); the resolvent identity behind eps_equiv. The comparison
with kkvim_reference is a separate test at the end (the script itself does not import it).
"""

import ast
import importlib.util
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "r5_cccc_q2_integer_repro.py"
_spec = importlib.util.spec_from_file_location("r5_cccc_q2_integer_repro", SCRIPT)
q = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(q)


# --- helpers: a third formulation of Z and exact rational solves --------------------------------------------------

def z_by_application(nx, ny):
    """Column k of Z = the 13-point formula applied to the unit field e_k on the grid padded with ghost nodes,
    w = 0 on the boundary and w_-1 = w_1, w_{n+1} = w_{n-1} (no stencil table, no index mirroring)."""
    size = (nx - 1) * (ny - 1)
    out = np.zeros((size, size), dtype=np.int64)
    for k in range(size):
        w = np.zeros((nx + 3, ny + 3), dtype=np.int64)  # node i at row i + 1, i = -1..n+1
        i0, j0 = divmod(k, ny - 1)
        w[i0 + 2, j0 + 2] = 1
        w[0, :] = w[2, :]
        w[nx + 2, :] = w[nx, :]
        w[:, 0] = w[:, 2]
        w[:, ny + 2] = w[:, ny]
        for i in range(1, nx):
            for j in range(1, ny):
                a, b = i + 1, j + 1
                dx4 = w[a - 2, b] - 4 * w[a - 1, b] + 6 * w[a, b] - 4 * w[a + 1, b] + w[a + 2, b]
                dy4 = w[a, b - 2] - 4 * w[a, b - 1] + 6 * w[a, b] - 4 * w[a, b + 1] + w[a, b + 2]
                mixed = sum((1, -2, 1)[p + 1] * (1, -2, 1)[s + 1] * w[a + p, b + s] for p in (-1, 0, 1) for s in (-1, 0, 1))
                out[(i - 1) * (ny - 1) + (j - 1), k] = dx4 + 2 * mixed + dy4
    return out


def solve_exact(matrix, rhs):
    """Gaussian elimination in fractions (small dense systems)."""
    m = [[Fraction(int(v)) if not isinstance(v, Fraction) else v for v in row] + [Fraction(b)] for row, b in zip(matrix, rhs)]
    size = len(m)
    for col in range(size):
        piv = next(r for r in range(col, size) if m[r][col] != 0)
        m[col], m[piv] = m[piv], m[col]
        for r in range(size):
            if r != col and m[r][col] != 0:
                f = m[r][col] / m[col][col]
                m[r] = [a - f * b for a, b in zip(m[r], m[col])]
    return [m[r][size] / m[r][r] for r in range(size)]


# --- stencil and assembly (PROVEN by exact integer comparison) ---------------------------------------------------

def test_stencil13_coefficients():
    st = q.stencil13()
    expected = {(0, 0): 20, (1, 0): -8, (-1, 0): -8, (0, 1): -8, (0, -1): -8, (1, 1): 2, (1, -1): 2, (-1, 1): 2,
                (-1, -1): 2, (2, 0): 1, (-2, 0): 1, (0, 2): 1, (0, -2): 1}
    assert st == expected
    assert sum(st.values()) == 0


def test_mirror_rule():
    n = 6
    k = np.arange(-1, n + 2)
    assert q.mirror(k, n).tolist() == [1, 0, 1, 2, 3, 4, 5, 0, 5]


@pytest.mark.parametrize("nx, ny", [(3, 3), (4, 4), (4, 6), (6, 4), (5, 7), (8, 4)])
def test_three_integer_formulations_agree(nx, ny):
    z_node = q.z_nodewise(nx, ny).toarray()
    assert np.array_equal(z_node, q.z_kron(nx, ny).toarray())
    assert np.array_equal(z_node, z_by_application(nx, ny))
    assert np.array_equal(z_node, z_node.T)


def test_boundary_rows_cccc():
    """Ghost w_-1 = w_1: diagonal 20 inside, 21 next to one edge, 22 in the corner node; distance-2 neighbour inside."""
    nx = ny = 6
    z = q.z_nodewise(nx, ny).toarray()

    def idx(i, j):
        return (i - 1) * (ny - 1) + (j - 1)

    assert z[idx(3, 3), idx(3, 3)] == 20
    assert z[idx(1, 3), idx(1, 3)] == 21
    assert z[idx(3, 1), idx(3, 1)] == 21
    assert z[idx(1, 1), idx(1, 1)] == 22
    assert z[idx(1, 1), idx(2, 1)] == -8 and z[idx(1, 1), idx(3, 1)] == 1 and z[idx(1, 1), idx(2, 2)] == 2
    assert np.linalg.eigvalsh(z.astype(float)).min() > 0  # SPD (dense check on a small grid)


@pytest.mark.parametrize("kind", q.ASSEMBLIES)
@pytest.mark.parametrize("n", [16, 32])
def test_f_assemblies_exact_at_dyadic_n(kind, n):
    """h = 1/n exact: every F-assembly equals n^4 Z bitwise (PROVEN: all operations exact)."""
    r = 2
    nx, ny, lx, ly, _ = q.geometry(r, n, "x")
    z = q.z_nodewise(nx, ny)
    a = q.assemble_f(kind, z, nx, ny, lx, ly, n)
    assert q.matrix_difference(a, z, n) == (0.0, 0)


@pytest.mark.parametrize("kind", q.ASSEMBLIES)
def test_f_assemblies_are_the_same_operator_at_non_dyadic_n(kind):
    """Non-dyadic h: the F-assembly is n^4 Z up to rounding of its entries, not a different scaling or stencil:
    the entries divided by n^4 round to the integers of Z (exact check, pattern identical)."""
    n, r = 12, 2
    nx, ny, lx, ly, _ = q.geometry(r, n, "x")
    z = q.z_nodewise(nx, ny)
    a = q.assemble_f(kind, z, nx, ny, lx, ly, n).tocsr()
    a.sort_indices()
    zz = z.tocsr()
    zz.sort_indices()
    assert np.array_equal(a.indptr, zz.indptr) and np.array_equal(a.indices, zz.indices)
    assert np.array_equal(np.rint(a.data / n**4).astype(np.int64), zz.data)


# --- scaling and exact residual ----------------------------------------------------------------------------------

def test_scaling_alpha_is_y_centre_over_n4():
    """A = Z/h^4 with h = l/n and alpha = w_c D/(q l^4): for l = 2 (dyadic) the physical F-assembly equals (n/l)^4 Z
    bitwise and its exact solution gives the same alpha as y_c / n^4 (exact rational solve, small grid)."""
    n, r, l = 4, 2, 2.0
    nx, ny = r * n, n
    z = q.z_nodewise(nx, ny)
    a = q.f_kron(nx, ny, r * l, l)
    assert np.array_equal(a.toarray(), z.toarray() * (n / l) ** 4)
    y = solve_exact(z.toarray().tolist(), [1] * z.shape[0])
    w = solve_exact([[Fraction(v) for v in row] for row in a.toarray().tolist()], [1] * z.shape[0])
    c = (nx // 2 - 1) * (ny - 1) + (ny // 2 - 1)
    assert w[c] / Fraction(l) ** 4 == y[c] / n**4


def test_exact_residual_matches_fractions():
    rng = np.random.default_rng(7)
    nx, ny, n = 6, 4, 4
    z = q.z_nodewise(nx, ny)
    for a in (z, q.f_div(z, 3) * 1.0):  # integer entries and non-integer double entries
        a = a.tocsr()
        x = rng.normal(size=a.shape[0]) * 10.0 ** rng.integers(-12, 3, size=a.shape[0])
        for factor in (1, n**4):
            got = q.ExactOperator(a).residual(x, factor)
            dense = a.toarray()
            exact = [1 - factor * sum(Fraction(float(dense[i, j])) * Fraction(float(x[j])) for j in range(a.shape[1]))
                     for i in range(a.shape[0])]
            assert got.tolist() == [float(v) for v in exact]


def test_refined_alpha_equals_exact_rational_on_small_grid():
    """V4 against the exact rational solution: |alpha_V4 - alpha_exact| within one rounding of alpha (the refined iterate
    is the double nearest to the exact solution or its neighbour; the comparison is in ulps, not a tolerance)."""
    for r, n in [(1, 4), (2, 4), (1, 6)]:
        nx, ny, _, _, c = q.geometry(r, n, "x")
        z = q.z_nodewise(nx, ny)
        y_exact = solve_exact(z.toarray().tolist(), [1] * z.shape[0])
        _, y, corr, _ = q.refine(z.astype(float).tocsc(), q.ExactOperator(z))
        assert corr[-1] <= corr[0]
        alpha_exact = y_exact[c] / n**4
        alpha = y[c] / n**4
        assert abs(Fraction(alpha) - alpha_exact) <= Fraction(math.ulp(alpha))


# --- dyadic controls: exact power-of-two scaling -----------------------------------------------------------------

def test_dyadic_v1_equals_v3_and_v2_equals_v4_bitwise():
    """At dyadic n the F-operator is n^4 Z = 2^(4k) Z: the same LU code path on exactly scaled data gives exactly scaled
    results (PROVEN, IEEE scaling by a power of two, no under/overflow). Small grid, both orientations."""
    rows = q.run_point(2, 8, "TEST", sensitivity=False)
    by = {}
    for row in rows:
        by.setdefault((row["orientation"], row["variant"], row["assembly"]), row["alpha"])
    for o in ("x", "y"):
        for kind in q.ASSEMBLIES:
            assert by[(o, "V1", kind)] == by[(o, "V3", "INT")]
            assert by[(o, "V2", kind)] == by[(o, "V4", "INT")]
            assert by[(o, "V2X", kind)] == by[(o, "V4", "INT")]
    assert all(row["matrix_entries_differing"] == 0 for row in rows)


def test_run_point_records_every_variant():
    rows = q.run_point(1, 6, "TEST", sensitivity=True)
    variants = {(row["variant"], row["assembly"], row["ordering"]) for row in rows if row["orientation"] == "x"}
    assert ("V3", "INT", "COLAMD") in variants and ("V3", "INT", "MMD_AT_PLUS_A") in variants
    assert ("V4", "INT", "COLAMD") in variants
    for kind in q.ASSEMBLIES:
        assert {("V1", kind, "COLAMD"), ("V1", kind, "MMD_AT_PLUS_A"), ("V2", kind, "COLAMD"), ("V2X", kind, "COLAMD")} <= variants
    for row in rows:
        assert row["backward_error_Z"] >= 0 and row["refined"] == (row["variant"] in ("V2", "V2X", "V4"))


# --- eps_equiv: the resolvent identity (exact) -------------------------------------------------------------------

def test_resolvent_identity_behind_eps_equiv():
    """(Z + eps I)^-1 1 - Z^-1 1 = -eps (Z + eps I)^-1 Z^-1 1 exactly; its eps -> 0 limit is the first-order coefficient
    -(Z^-2 1)_c used for eps_equiv."""
    nx, ny, _, _, c = q.geometry(1, 4, "x")
    z = q.z_nodewise(nx, ny).toarray().tolist()
    eps = Fraction(1, 2**40)
    zp = [[Fraction(v) + (eps if i == j else 0) for j, v in enumerate(row)] for i, row in enumerate(z)]
    y = solve_exact(z, [1] * len(z))
    yp = solve_exact(zp, [1] * len(z))
    t = solve_exact(zp, y)
    assert yp[c] - y[c] == -eps * t[c]
    # and t = (Z + eps I)^-1 Z^-1 1 differs from g = Z^-2 1 by exactly eps (Z + eps I)^-1 g (the same identity, second order)
    g = solve_exact(z, y)
    assert g[c] - t[c] == eps * solve_exact(zp, g)[c]


# --- printed-digit rule and independence -------------------------------------------------------------------------

def test_reproduced_to_printed_digits_rule():
    assert q.reproduced(1.2759787033576e-3, "1.2759787033576e-3")
    assert q.reproduced(1.27597870335764e-3, "1.2759787033576e-3")
    assert not q.reproduced(1.2759787033577e-3, "1.2759787033576e-3")
    assert q.printed(0.0012759787033572092) == "1.2759787033572e-03"


def test_external_values_are_rm98_and_csv():
    """27 strings as in RM 9.8 and in d2_external.csv (column external)."""
    import csv

    assert len(q.EXTERNAL) == 27
    with (ROOT / "docs" / "results" / "r5_cccc_references" / "d2_external.csv").open(newline="") as f:
        stored = {(int(r["r"]), int(r["n"])): r["external"] for r in csv.DictReader(f) if r["external"][:1].isdigit()}
    assert stored == q.EXTERNAL
    assert set(q.TARGET) <= set(q.EXTERNAL) and set(q.DYADIC) <= set(q.EXTERNAL)


def test_script_does_not_import_kkvim_or_kkvim_reference():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            names.add((node.module or "").split(".")[0])
    assert not names & {"kkvim", "kkvim_reference"}


# --- separate comparison with kkvim_reference (the project D2) ---------------------------------------------------

@pytest.mark.parametrize("nx, ny", [(4, 4), (6, 9), (12, 6), (24, 8)])
def test_integer_operator_equals_kkvim_reference_parts(nx, ny):
    from kkvim_reference import r5_cccc_d2 as d2

    zx, zxy, zy = d2.nodewise_parts(nx, ny)
    assert np.array_equal((zx + zxy + zy).toarray(), q.z_nodewise(nx, ny).toarray())


def test_refined_alpha_agrees_with_kkvim_reference_solve():
    """Two refined solutions of the same discrete problem, PL 1.4 item 4 rule fixed before comparison:
    |a - b| <= 4 u a + 4 (e_a + e_b)."""
    from kkvim_reference import r5_cccc_d2 as d2

    for r, n in [(1, 32), (3, 16)]:
        nx, ny, lx, ly, c = q.geometry(r, n, "x")
        z = q.z_nodewise(nx, ny)
        _, y, corr, _ = q.refine(z.astype(float).tocsc(), q.ExactOperator(z))
        a, e_a = y[c] / n**4, corr[-1] / n**4
        sol = d2.solve(d2.D2Problem(lx, ly, nx, ny))
        b = sol.alpha()
        assert abs(a - b) <= 4 * 2.0**-53 * a + 4 * (e_a + sol.e_D2)
