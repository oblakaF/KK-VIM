"""Tests of the second computational path of B_E (scripts/r5_cccc_q4_be_second_path_research.py; ST = docs/R5_CCCC_Q4_ENERGY_ERROR_
BOUND.study.md). Expectations are stated before the comparisons: exact arithmetic facts, closed forms of reference constructions
derived by hand (the derivations are in the QA report, sect. 5), the inequalities of E-1, E-2, L-6, E-3 and the refusals. A numerical
agreement of the two implementations is NOT taken as a proof of their common formula."""

import ast
import importlib.util
import math
import os
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "scripts" / "r5_cccc_q4_be_second_path_research.py"


def _load():
    spec = importlib.util.spec_from_file_location("_be2_second", TOOL)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


be = sys.modules.get("_be2_second") or _load()
F = Fraction


def _vals(m, w):
    return be.field_quantities(m, [F(x) for x in w])


# --- exact arithmetic ------------------------------------------------------------------------------------------------------------

def test_exact_conversion_of_binary64():
    for x in (0.1, -3.5, 2.0**-1074, -(2.0**-1060), 1.7976931348623157e308, 1.0, 2.0**-1022):
        assert be.exact(x) == F(x)
    assert be.exact(-0.0) == 0 and be.exact(np.float64(0.25)) == F(1, 4)
    for x in (math.inf, -math.inf, math.nan):
        with pytest.raises(be.SecondPathError):
            be.exact(x)


@pytest.mark.parametrize("x", [F(0), F(4), F(9, 16), F(2), F(1, 3), F(10**40 + 1, 7), F(1, 2**2000), F(2**2000 + 1)])
def test_directed_roots_are_certified(x):
    lo, hi = be.sqrt_bounds(x)
    assert lo * lo <= x <= hi * hi and lo <= hi
    if x in (0, 4, F(9, 16)):
        assert lo == hi and lo * lo == x  # exact squares: exact root
    else:
        assert hi - lo <= hi * F(1, 2**90)  # relative width of the own rule (96 bits), not a tolerance of any comparison
    qlo, qhi = be.sqrt_bounds_qaeb(x)
    assert qlo * qlo <= x <= qhi * qhi
    with pytest.raises(be.SecondPathError):
        be.sqrt_bounds(F(-1, 3))


@pytest.mark.parametrize("x", [F(1, 10), F(-1, 10), F(2**-1074), F(3, 2**1076), F(1) + F(1, 2**60), F(2**1024),
                               F(2**1024 - 2**970), F(5)])
def test_directed_rounding_to_binary64(x):
    up, dn = be.round_up(x), be.round_down(x)
    assert F(dn) <= x if math.isfinite(dn) else True
    if math.isfinite(up):
        assert F(up) >= x and (F(math.nextafter(up, -math.inf)) < x)
    else:
        assert x > F(1.7976931348623157e308)
    try:
        representable = F(float(x)) == x
    except OverflowError:
        representable = False
    if representable:  # both directions give the value itself
        assert up == dn == float(x)


# --- the operator and the reference constructions (closed forms derived by hand) -------------------------------------------------

def test_sparse_operator_on_the_smallest_mesh():
    m = be.Mesh(1.0, 3, 1.0, 3)  # interior 2 x 2, nodes 4 x 4
    h2 = F(1, 9) ** -1  # 1/h^2 = 9
    t = m.apply([F(1), F(0), F(0), F(0)])  # w = 1 at interior (1, 1)
    node = {(I, J): t[I * 4 + J] for I in range(4) for J in range(4)}
    assert node[(1, 1)] == -4 * h2 and node[(2, 1)] == h2 and node[(1, 2)] == h2
    assert node[(0, 1)] == 2 * h2 and node[(1, 0)] == 2 * h2 and node[(0, 0)] == 0 and node[(2, 2)] == 0
    A = m.apply([F(-1), F(0), F(0), F(0)], absolute=True)
    assert A[1 * 4 + 1] == 4 * h2 and A[0 * 4 + 1] == 2 * h2


def test_reference_zero_field():
    row = be.synthetic_row(1.0, 8, 1.0, 10, N=1, seed=1)
    z = {k: np.zeros_like(row[k]) for k in ("Xp", "Yp", "Xs", "Ys", "wh_p", "wh_s", "dh")}
    g = be.en.Grid(1.0, 8, 1.0, 10)
    res = be.second_path(row["mesh"], z["Xp"], z["Yp"], z["Xs"], z["Ys"], z["wh_p"], z["wh_s"], z["dh"],
                         {"P_p": be.en.pi_double(g, z["wh_p"]), "P_s": 0.0, "Q": 0.0}, parts=True)
    assert res["status"] == be.CERTIFIED and res["BE"] == 0 and res["E_s"] == 0 and res["BE_up"] == 0.0
    assert all(res["assembly"][k] == 0 for k in ("B1", "B2", "B3", "B4", "eps_s", "eps_p"))
    assert res["parts"]["ok"]


def test_reference_unit_perturbation():
    m = be.Mesh(1.0, 8, 1.0, 10)
    hx, hy = F(1, 8), F(1, 10)
    w = [0] * m.n_int
    w[m.col(4, 5)] = 1
    q = _vals(m, w)
    a2 = hx * hy * ((2 / hx**2 + 2 / hy**2) ** 2 + 2 / hx**4 + 2 / hy**4)  # 5-point stencil, all weights h_x h_y
    assert q["a2"] == a2 and q["Sq"] == a2 / 2 and q["S2"] == a2 / 2  # A = |t| for a single positive node
    assert q["Sl"] == hx * hy and q["h2"] == hx * hy and q["Pi"] == a2 / 2 - hx * hy
    w[m.col(4, 5)] = -2  # sign: a2 scales with the square, S_l with the modulus, the load changes sign
    q2 = _vals(m, w)
    assert q2["a2"] == 4 * a2 and q2["Sl"] == 2 * hx * hy and q2["load"] == -2 * hx * hy


def test_reference_constant_field():
    for nx, ny, lx, ly in ((8, 10, 1.0, 1.0), (3, 3, 1.0, 2.0), (5, 7, 0.75, 1.5)):
        m = be.Mesh(lx, nx, ly, ny)
        hx, hy, c = F(lx) / nx, F(ly) / ny, F(3, 4)
        q = _vals(m, [c] * m.n_int)
        a2 = 6 * c**2 * (ny - 1) * hy / hx**3 + 6 * c**2 * (nx - 1) * hx / hy**3 + 8 * c**2 / (hx * hy)
        assert q["a2"] == a2, (nx, ny)
        assert q["Sl"] == hx * hy * (nx - 1) * (ny - 1) * c and q["h2"] == hx * hy * (nx - 1) * (ny - 1) * c**2


def test_reference_symmetric_and_antisymmetric_fields():
    m = be.Mesh(1.0, 8, 1.0, 10)
    rng = np.random.default_rng(5)
    base = np.round(rng.standard_normal((7, 9)) * 64) / 64
    sym = base + base[::-1, ::-1]  # w(i, j) = w(n_x - i, n_y - j)
    q1, q2 = _vals(m, sym.ravel()), _vals(m, sym[::-1, ::-1].ravel())
    assert all(q1[k] == q2[k] for k in ("a2", "Sq", "S2", "Sl", "h2", "Pi"))
    anti = base - base[::-1, :]  # antisymmetric in x: zero load exactly, Pi = ||w||_a^2/2
    qa = _vals(m, anti.ravel())
    assert qa["load"] == 0 and qa["Pi"] == qa["a2"] / 2 and qa["Sl"] > 0
    # the residual route Delta^T W Delta equals the bilinear form: v^T (D A w) h_x h_y = a_h(v, w)
    v = [F(x) for x in base.ravel()]
    w = [F(x) for x in sym.ravel()]
    t = m.apply(w)
    Aw = m.apply_T([m.W[n] * t[n] for n in range(m.n_nodes)])
    assert sum((a * b for a, b in zip(v, Aw)), F(0)) == be.bilinear(m, v, w)


def test_lemma6_and_norm_relations_on_reference_fields():
    m = be.Mesh(1.0, 8, 1.0, 10)
    rng = np.random.default_rng(7)
    for w in (rng.standard_normal(m.n_int), np.ones(m.n_int), (-1.0) ** np.arange(m.n_int)):
        q = _vals(m, np.round(w * 128) / 128)
        assert q["S2"] <= m.D / 2 * m.Lam**2 * q["h2"]  # L-6
        assert q["a2"] >= m.D * m.mu_low**2 * q["h2"]  # ||w||_a >= D^(1/2) mu_low ||w||_h (VOK-1 item 4)
        assert q["a2"] / 2 <= q["Sq"]  # S_q >= (D/2) sum W t^2


# --- grid constants ---------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("grid", [(1.0, 8, 1.0, 10), (1.0, 12, 0.2, 9), (1.0, 16, 1 / 3, 16), (1.0, 23, 3.0, 19)])
def test_grid_constants_closed_forms_equal_corner_enumeration(grid):
    kc = be.grid_constants(be.Mesh(*grid), N=3)
    assert all(kc["self_checks"].values()), kc["self_checks"]
    assert kc["eta_t"] == kc["eta"] * (1 + be.U) + be.U and kc["K_star"] >= 1
    assert kc["L"] == be.pairwise_depth(kc["M"]) and kc["M"] == (grid[1] + 1) * (grid[3] + 1) + (grid[1] - 1) * (grid[3] - 1)


# --- the bound on synthetic rows: components, inequalities, parts -----------------------------------------------------------------

@pytest.fixture(scope="module")
def rows():
    specs = ((1.0, 8, 1.0, 10, 1, 1, False), (1.0, 12, 0.75, 9, 2, 2, False), (1.0, 10, 1.0, 10, 3, 3, True),
             (2.0, 9, 1.0, 7, 1, 4, False))
    out = []
    for s in specs:
        row = be.synthetic_row(s[0], s[1], s[2], s[3], N=s[4], seed=s[5], zero_prev=s[6])
        out.append((row, be.run_row(row, parts=True)))
    return out


def test_every_component_and_the_theorems_on_synthetic_rows(rows):
    for row, res in rows:
        assert res["status"] == be.CERTIFIED and res["contradictions"] == [], res["reasons"]
        p = res["parts"]
        assert p["checks"]["identity"]  # E^(1) + E^(2) + E^(3) + E^(4) = E_s exactly (E-1)
        assert all(p["checks"].values()), p["checks"]  # each part below its own bound (L-3...L-8), L-6, ||e||_a <= eps
        assert res["E_s_le_B_E"] and res["B_E_le_Kstar_R"]  # E-2, E-3
        a, lo = res["assembly"], res["assembly_lower"]
        assert a["BE"] == a["B1"] + a["B2"] + a["B3"] + a["B4"] and lo["BE"] <= a["BE"]
        assert a["B3"] == a["beta"] * a["d_a"] + a["beta"] ** 2 / 2 and a["B3_second_order"] == a["beta"] ** 2 / 2
        assert a["B_nu_second_order"] > 0  # the nu S_2 remainders are present (ST 5.3)
        assert F(res["BE_up"]) >= a["BE"] > 0


def test_zero_previous_field_gives_an_exactly_zero_formation_bound(rows):
    row, res = rows[2]
    assert res["formation"]["p"]["exact_zero"] and res["formation"]["p"]["factor_zero"] and res["assembly"]["eps_p"] == 0


def test_comparison_with_the_primary_qaeb(rows):
    for row, res in rows:
        prim = be.primary_qaeb(be.load_qaeb(), row["mesh"], row["Xp"], row["Yp"], row["Xs"], row["Ys"], row["dh"], row["wh_p"],
                               row["wh_s"])
        c = be.compare(res, prim)
        assert c["all_match"] and c["first_divergence"] is None
        rel = c["BE_own_vs_primary"]
        assert rel["both_upper_bounds_of_the_exact_formula"] and rel["primary_is_documented_representation"]


def test_first_divergence_is_reported_with_its_conclusion(rows):
    row, res = rows[0]
    prim = be.primary_qaeb(be.load_qaeb(), row["mesh"], row["Xp"], row["Yp"], row["Xs"], row["Ys"], row["dh"], row["wh_p"],
                           row["wh_s"])
    bad = dict(prim)
    bad["B2"] = prim["B2"] - F(1, 2**200)  # below the exact formula
    bad["BE"] = prim["BE"] - F(1, 2**200)
    c = be.compare(res, bad)
    fd = c["first_divergence"]
    assert fd["component"] == "B2" and fd["status"] == "ERROR_UNSAFE" and "unsafe" in fd["conclusion"]
    assert fd["inputs"]["mesh"] == row["mesh"] and fd["definition"] and fd["source"].startswith("ST")
    bad = dict(prim)
    bad["p.Sq"] = prim["p.Sq"] * 2
    assert be.compare(res, bad)["first_divergence"]["status"] == "ERROR_VALUE"
    bad = dict(prim)
    bad["B4"] = prim["B4"] * 2
    assert be.compare(res, bad)["first_divergence"]["status"] == "ERROR_NOT_DOCUMENTED"


# --- special inputs and refusals: never a false certificate ----------------------------------------------------------------------

def test_nearly_cancelling_difference():
    row = be.synthetic_row(1.0, 8, 1.0, 10, N=1, seed=8, near=1)
    res = be.run_row(row, parts=True)
    assert res["status"] == be.CERTIFIED and res["parts"]["ok"] and res["E_s_le_B_E"]
    assert 0 < res["radicands"]["d"]["h2"] < res["radicands"]["s"]["h2"] * F(1, 2**90)


def test_values_near_the_normal_range_are_refused():
    row = be.synthetic_row(1.0, 8, 1.0, 10, N=1, seed=9, scale=2.0**-500)  # products and squares fall below 2^-1022
    res = be.run_row(row)
    assert res["status"] == be.UNRESOLVED
    assert any(r["premise"] == "P-3'(c), P-2" for r in res["reasons"]), res["reasons"]


@pytest.mark.parametrize("D, reason", [(0.0, "root"), (-1.0, "root")])
def test_domain_of_the_bound(D, reason):
    row = be.synthetic_row(1.0, 8, 1.0, 10, N=1, seed=10)
    res = be.second_path({**row["mesh"], "D": D}, row["Xp"], row["Yp"], row["Xs"], row["Ys"], row["wh_p"], row["wh_s"], row["dh"])
    assert res["status"] == be.UNRESOLVED and res["reasons"][0]["component"] == reason


def test_not_finite_input_and_violated_premises_are_refused():
    row = be.synthetic_row(1.0, 8, 1.0, 10, N=1, seed=10)
    bad = row["wh_s"].copy()
    bad[2, 3] = math.nan
    res = be.second_path(row["mesh"], row["Xp"], row["Yp"], row["Xs"], row["Ys"], row["wh_p"], bad, row["dh"])
    assert res["status"] == be.UNRESOLVED and res["reasons"][0]["component"] == "input"
    # U-1: a field that is not fl(X Y^T) (one entry moved by 2^10 ulps)
    w = row["wh_s"].copy()
    w[3, 4] = w[3, 4] * (1 + 2.0**-42)
    res = be.second_path(row["mesh"], row["Xp"], row["Yp"], row["Xs"], row["Ys"], row["wh_p"], w, row["wh_p"] - w, None)
    assert res["status"] == be.UNRESOLVED and any(r["premise"] == "U-1" for r in res["reasons"])
    # I-1: given P^ not the values of the path
    P = dict(row["P_hat"])
    P["Q"] = math.nextafter(P["Q"], 1.0)
    res = be.run_row({**row, "P_hat": P})
    assert res["status"] == be.UNRESOLVED and any(r["premise"] == "I-1" for r in res["reasons"])
    # L-2: d^ that is not fl(w^_p - w^_s)
    d = row["dh"].copy()
    d[0, 0] = math.nextafter(d[0, 0], 1.0)
    res = be.run_row({**row, "dh": d, "P_hat": None})
    assert res["status"] == be.UNRESOLVED and any(r["premise"] == "L-2 (d^)" for r in res["reasons"])


def test_operation_checks_see_an_underflow():
    m = be.Mesh(1.0, 8, 1.0, 10)
    ops = be.OpChecks()
    w = np.zeros((7, 9))
    w[3, 4] = 2.0**-1070  # t ~ 2^-1063, t*t underflows to 0
    be.replicate_g4(m, w, True, ops, "x")
    assert any("P-3'(c)" in v for v in ops.violations)
    ops = be.OpChecks()
    be.replicate_g4(m, np.full((7, 9), 0.5), True, ops, "x")
    assert ops.violations == []


def test_operation_checks_on_constructed_operands():
    """The classification itself (it must work under any FTZ/DAZ mode, so it is tested on constructed bit patterns)."""
    ops = be.OpChecks()
    t = 2.0**-511
    ops.mul_div("exactly 2^-1022", np.array([t]), np.array([t]), np.array([t * t]))  # |result| = 2^-1022: not strictly above
    assert any("<= 2^-1022" in v for v in ops.violations)
    ops = be.OpChecks()
    ops.mul_div("normal", np.array([2.0**-500]), np.array([2.0**-500]), np.array([2.0**-1000]))
    ops.mul_div("zero operand", np.array([0.0]), np.array([3.0]), np.array([0.0]))
    assert ops.violations == []
    ops = be.OpChecks()
    tiny = 2.0**-1074
    ops.add_sub("FTZ-like", np.array([tiny]), np.array([tiny]), np.array([0.0]), sub=False)  # nonzero sum flushed to 0
    assert any("without exact cancellation" in v for v in ops.violations)
    ops = be.OpChecks()
    ops.add_sub("cancel +", np.array([1.5, -0.0]), np.array([-1.5, 0.0]), np.array([0.0, 0.0]), sub=False)
    ops.add_sub("cancel -", np.array([2.5]), np.array([2.5]), np.array([0.0]), sub=True)
    assert ops.violations == []


def test_guards_of_the_own_routines(monkeypatch):
    """Self-certifications that cannot fire on correct routines: a defective integer root and corner enumeration that disagrees
    with the closed form must be refused."""
    monkeypatch.setattr(be, "_isqrt_up", math.isqrt)  # upper integer root rounded down
    with pytest.raises(be.SecondPathError, match="not certified"):
        be.sqrt_bounds(F(2))
    monkeypatch.undo()
    real = be._corners
    monkeypatch.setattr(be, "_corners", lambda base, k, divisor=F(1): real(base, k, divisor) / 2)
    kc = be.grid_constants(be.Mesh(1.0, 8, 1.0, 10), N=1)
    assert not kc["self_checks"]["eta_closed_equals_corners"]
    row = be.synthetic_row(1.0, 8, 1.0, 10, N=1, seed=3)
    res = be.run_row(row)
    assert res["status"] == be.UNRESOLVED and any(r["premise"] == "I-2" for r in res["reasons"])


def test_probes_in_this_thread():
    p = be.probes()
    assert p["ok"], p["fails"]


# --- published Q3 rows (inputs only if the arrays are available and their sha256 match) ------------------------------------------

def _arrays():
    d = Path(os.environ.get("KKVIM_Q3_ARRAYS", r"D:\git\Test_Versions\KK-VIM\KK-VIM-Q3-QA\scratch\r5_cccc_q3_prequalification\arrays"))
    return d if d.is_dir() and any(d.glob("*.npz")) else None


@pytest.mark.skipif(_arrays() is None, reason="Q3 NPZ arrays not available here (outside git); published-row part NOT performed")
def test_published_q3_rows_reproduced():
    got = [r for r in be.q3_rows(_arrays(), max_nodes=1024, limit=2) if "unusable" not in r]
    assert len(got) == 2
    for row in got:
        assert row["bound_to_published"] is True and row["published_status"] == "VERIFIED"
        res = be.run_row(row, parts=True)
        assert res["status"] == be.CERTIFIED and res["parts"]["ok"] and res["E_s_le_B_E"] and res["B_E_le_Kstar_R"]
        assert be.round_up(res["assembly_qaeb80"]["BE"]).hex() == float(row["published_B_E_up"]).hex()


# --- independence and guards --------------------------------------------------------------------------------------------------------

def test_the_b_e_functions_do_not_call_the_primary_or_the_energy_module():
    """Independence of the implementation: r5_cccc_energy (the object) only binds the replication and builds inputs; QA-EB is
    loaded only for the comparison; no import of r5_cccc_d2, of the gamma tools or of the working kkvim."""
    tree = ast.parse(TOOL.read_text(encoding="utf-8"))
    allowed_en = {"bind_double_path", "synthetic_row", "q3_rows"}
    allowed_qaeb = {"load_qaeb", "resources", "main", "provenance"}
    for fn in [n for n in tree.body if isinstance(n, ast.FunctionDef)]:
        names = {n.value.id for n in ast.walk(fn) if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)}
        called = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)}
        assert "en" not in names or fn.name in allowed_en, fn.name
        assert not ({"load_qaeb", "QAEB_TOOL"} & called) or fn.name in allowed_qaeb, fn.name
        for n in ast.walk(fn):
            if isinstance(n, (ast.Import, ast.ImportFrom)):
                assert fn.name == "q3_rows", fn.name  # input reconstruction of the published rows only (r5_cccc_bc, halfstep)
    top = [ast.unparse(n) for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert [x for x in top if "kkvim" in x] == ["from kkvim_reference import r5_cccc_energy as en"]
    text = TOOL.read_text(encoding="utf-8")
    for forbidden in ("r5_cccc_d2 import", "import r5_cccc_d2", "gamma_interval_audit", "from kkvim import", "import kkvim\n"):
        assert forbidden not in text, forbidden


def test_primary_files_unchanged_and_output_guard():
    out = subprocess.run(["git", "status", "--porcelain", "--", "scripts/r5_cccc_q4_energy_bound_premises_audit.py",
                          "src/kkvim_reference", "docs/results"], cwd=ROOT, capture_output=True, text=True)
    assert out.returncode != 0 or out.stdout.strip() == ""
    p = subprocess.run([sys.executable, str(TOOL), "run", "--profile", "synthetic", "--out", "docs/results/x"], cwd=ROOT,
                       capture_output=True, text=True)
    assert p.returncode != 0 and "refused" in (p.stderr + p.stdout) and not (ROOT / "docs" / "results" / "x").exists()
    assert "NOT D1 data" in be.DATA_STATUS and "OD-11" in be.DATA_STATUS
