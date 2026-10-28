"""Tests of the QA check of the arithmetic premises of |E_s| <= B_E,s <= K* R_Sigma (scripts/r5_cccc_q4_energy_bound_premises_audit.py):
bit-level classification and exact dyadic decoding, the FTZ / DAZ / rounding probe, detection of violations of P-3' (incl. the
boundary example of the review), the replication of the double path G-4 (bitwise the module), the exact formation check U-1 (incl. an
underflowing product: N-7), own exact energy quantities against r5_cccc_energy / r5_cccc_d2, directed roots, the grid constants U-3
against the review, B_E and the separate mathematical path on synthetic hard cases, the binding and the deterministic CI profile,
unchanged modules and data, no import of the working kkvim. QA research tool."""

import ast
import csv
import importlib.util
import json
import math
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from kkvim_reference import r5_cccc_d2 as d2
from kkvim_reference import r5_cccc_energy as en

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "r5_cccc_q4_energy_bound_premises_audit.py"
spec = importlib.util.spec_from_file_location("r5_cccc_q4_energy_bound_premises_audit", SCRIPT)
ep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ep)
U = Fraction(1, 2**53)


# --- bit level -----------------------------------------------------------------------------------------------------------------

def test_bit_classification_and_negation():
    v = np.array([0.0, -0.0, 5e-324, 2.0**-1022, np.nextafter(2.0**-1022, 1.0), 1.0, -np.inf, np.nan, -2.0**-1023])
    assert ep.is_zero(v).tolist() == [True, True, False, False, False, False, False, False, False]
    assert ep.is_subnormal(v).tolist() == [False, False, True, False, False, False, False, False, True]
    assert ep.gt_tiny(v[:6]).tolist() == [False, False, False, False, True, True]
    assert ep.is_finite(v).tolist() == [True] * 6 + [False, False, True]
    n = ep.negate(v[:6])
    assert (ep._u64(n) ^ ep._u64(v[:6])).tolist() == [int(ep._SIGN)] * 6
    assert ep.same_value(0.0, -0.0).all() and not ep.same_value(5e-324, -5e-324).any()


def test_dyadic_is_exact_including_subnormals():
    rng = np.random.default_rng(7)
    a = rng.standard_normal((9, 4)) * 2.0 ** rng.integers(-1100, 300, size=(9, 4))
    a[0, 0], a[1, 1], a[2, 2] = 0.0, -0.0, -5e-324
    I, E = ep.dyadic(a)
    for x, k in zip(a.ravel(), I.ravel()):
        assert Fraction(int(k)) * ep.p2(E) == Fraction(float(x))
    assert ep.frac_of(-2.0**-1074) == -Fraction(1, 2**1074)
    with pytest.raises(ValueError):
        ep.dyadic(np.array([1.0, np.inf]))


# --- U-2 (a): probe ------------------------------------------------------------------------------------------------------------

def test_probe_expectations_are_the_round_to_nearest_results():
    ops = {"add": lambda a, b: a + b, "sub": lambda a, b: a - b, "mul": lambda a, b: a * b, "div": lambda a, b: a / b}
    for name, op, a, b, e in ep._PROBES:
        exact = ops[op](Fraction(float.fromhex(a)), Fraction(float.fromhex(b)))
        assert Fraction(float(exact)) == Fraction(float.fromhex(e)), name  # float(Fraction): correctly rounded, integer arithmetic
    subnormal = [p for p in ep._PROBES if p[0].startswith(("FTZ", "DAZ"))]
    assert all(ep.is_subnormal(np.array([float.fromhex(x) for x in p[2:]])).any() for p in subnormal)


def test_probe_in_this_thread_is_ok():
    p = ep.probe_mode()
    assert p["ok"] and not p["fails"] and p["main_thread"]


# --- U-2 (c): detection of violations --------------------------------------------------------------------------------------------

def test_boundary_example_of_the_review_is_refused():
    a = float.fromhex("0x1p-540") * (6361 * 69431)
    b = float.fromhex("0x1p-535") * 20394401
    assert ep.gt_tiny(np.array([a, b])).all()
    r = a * b
    assert Fraction(a) * Fraction(b) == Fraction(2**53 - 1, 2**1075) and r == 2.0**-1022  # |delta| = u/(1 - u) > u
    log = ep.OpLog()
    log.mul("boundary", a, b, r)
    assert log.violations and "<= 2^-1022" in log.violations[0]


def test_oplog_flags_flush_zero_division_and_non_finite():
    log = ep.OpLog()
    a, b = np.array([1.0, 2.0**-1022 * 1.5, 3.0]), np.array([1.0, 2.0**-1022, -3.0])
    log.add("sub ok", a, b, a - b, sub=True)
    assert not log.violations and log.subnormal_results == 1 and log.cancellations == 1
    log.add("flushed", a, b, np.array([0.0, 0.0, 6.0]), sub=True)  # what FTZ would return
    assert any("flush" in v for v in log.violations)
    log2 = ep.OpLog()
    log2.div("by zero", np.array([1.0]), np.array([0.0]), np.array([np.inf]))
    log2.exact2("x2", np.array([5e-324]), np.array([0.0]))
    log2.mul("ok", np.array([0.0, 2.0]), np.array([np.pi, 3.0]), np.array([0.0, 6.0]))
    assert len(log2.violations) == 3  # division by zero, non-finite, x2 zero pattern


@pytest.mark.parametrize("nx, ny, lx, ly", [(10, 9, 1.0, 1.0), (12, 9, 1.0, 0.2), (16, 16, 1.0, 1 / 3), (11, 13, 1.3, 0.7)])
def test_replication_is_bitwise_the_module(nx, ny, lx, ly):
    rng = np.random.default_rng(nx + ny)
    g = en.Grid(lx, nx, ly, ny)
    w = rng.standard_normal((nx - 1, ny - 1))
    w[2, :] = 0.0
    log = ep.OpLog()
    for with_load, d in ((True, w), (False, w - w * (1.0 + 2.0**-30))):
        r = ep.replicate_energy(g, d, log, "w", with_load)
        assert r["t_bitwise_module"] and r["quad_bitwise_module"] and r["value_bitwise_module"]
        assert r["levels"] == (ep.ceil_log2(g.M) if with_load else ep.ceil_log2((nx + 1) * (ny + 1)))
    assert not log.violations


def test_replication_refuses_underflowing_fields():
    g = en.Grid(1.0, 6, 1.0, 5)
    w = np.full((5, 4), 1e-310)
    log = ep.OpLog()
    r = ep.replicate_energy(g, w, log, "tiny", True)
    assert r["value_bitwise_module"] and log.violations and log.subnormal_operands > 0


# --- U-1 and N-7 ------------------------------------------------------------------------------------------------------------------

def test_formation_check_holds_and_detects_violations():
    rng = np.random.default_rng(3)
    X, Y = rng.standard_normal((7, 3)), rng.standard_normal((6, 3))
    W = X @ Y.T
    f = ep.formation_check(X, Y, W)
    assert f["holds"] and f["max_ratio"] <= 1 and f["elements"] == 42
    S = np.abs(X) @ np.abs(Y).T
    W2 = W.copy()
    W2[1, 2] += S[1, 2] * 2.0**-49
    assert not ep.formation_check(X, Y, W2)["holds"]


def test_formation_compensation_and_zero_factors():
    X = np.array([[1.0, 1.0], [0.0, 0.0], [2.0, -2.0]])
    Y = np.array([[1.0, -1.0 + 2.0**-30], [3.0, 0.5]])
    f = ep.formation_check(X, Y, X @ Y.T)
    assert f["holds"] and f["compensation"] >= 1 and f["zero_factor_entries"] == 2 and f["zero_S"] == 2


def test_underflowing_product_is_not_an_exact_zero():
    X, Y = np.array([[2.0**-600]]), np.array([[2.0**-600]])
    W = X @ Y.T
    assert W[0, 0] == 0.0  # the computed field is zero ...
    f = ep.formation_check(X, Y, W)
    assert f["computed_field_zero"] and not f["exact_product_zero"] and not f["holds"]  # ... the exact product is not (N-7)
    assert not f["exact_products_normal"]
    z = ep.formation_check(np.ones((4, 2)), np.zeros((3, 2)), np.ones((4, 2)) @ np.zeros((3, 2)).T)
    assert z["exact_product_zero"] and z["computed_field_zero"] and z["factor_all_zero"] and z["holds"]


# --- own exact quantities -----------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("nx, ny, lx, ly", [(5, 4, 1.0, 1.3), (7, 9, 1.0, 0.6), (6, 6, 1.0, 1 / 3), (8, 5, 1.3, 0.7)])
def test_own_exact_quantities_equal_the_module(nx, ny, lx, ly):
    rng = np.random.default_rng(nx * ny)
    g = en.Grid(lx, nx, ly, ny)
    gq = ep.GridQ(lx, nx, ly, ny)
    w = rng.standard_normal((nx - 1, ny - 1)) * 2.0 ** rng.integers(-6, 6, size=(nx - 1, ny - 1))
    I, E = ep.dyadic(w)
    s = ep.field_sums(gq, I, E)
    assert (s["Sq"], s["Sl"]) == en.majorant_S(g, w)
    assert s["a2"] == en.norm_a_sq_exact(g, w) == ep.a13(gq, (I, E), (I, E))  # Prop. D, two routes
    assert s["h2"] == en.norm_h_sq_exact(g, w)
    r = d2.residual_exact(d2.D2Problem(lx, ly, nx, ny, D=1.0, q=1.0), w.reshape(-1))
    assert ep.residual_h2(gq, I, E) == gq.hx * gq.hy * sum((x * x for x in r), Fraction(0))
    assert s["S2"] <= Fraction(1, 2) * gq.Lam**2 * s["h2"]  # L-6


def test_sqrt_bounds_directions():
    rng = np.random.default_rng(11)
    for _ in range(300):
        x = Fraction(int(rng.integers(1, 10**12)), int(rng.integers(1, 10**9))) * Fraction(2) ** int(rng.integers(-300, 300))
        lo, hi = ep.sqrt_bounds(x)
        assert lo * lo <= x <= hi * hi and hi - lo <= Fraction(1, x.denominator << ep.SQRT_BITS)
    assert ep.sqrt_bounds(Fraction(9, 4)) == (Fraction(3, 2), Fraction(3, 2))
    assert ep.sqrt_bounds(Fraction(0)) == (0, 0)
    with pytest.raises(ValueError):
        ep.sqrt_bounds(Fraction(-1))


def test_lin_le():
    X = np.array([3, -5, 0], dtype=object)
    assert ep.lin_le([(Fraction(1, 2), X)], [(Fraction(1), np.array([2, 3, 0], dtype=object))]) == (True, 5 / 6)
    assert ep.lin_le([(1, X)], [(1, np.array([2, 9, 0], dtype=object))])[0] is False


# --- U-3 -------------------------------------------------------------------------------------------------------------------------

def test_grid_constants_against_the_review():
    a = ep.grid_constants(1.0, 1024, 0.5, 512)
    b = ep.grid_constants(1.0, 512, 2.0, 1024)
    assert f"{float(a['K_star'] - 1):.4e}" == ep.REVIEWER_KSTAR_MINUS_1 == f"{float(b['K_star'] - 1):.4e}"
    assert a["K_star"] == a["K_Q"] and ep.grid_ok(a)
    c = ep.grid_constants(1.0, 32, 1.0, 32)  # exact h and h^2: c_q = 12 u + second order
    assert c["delta_hx"] == c["delta_cx"] == 0 and 12 <= float(c["c_q"] / U) < 12.0001 and c["pow_equals_mul"]
    assert c["L_equals_pairwise_depth"] and c["L_equals_EN_log2"] and float(c["K_P"]) < 1
    t = ep.grid_constants(1.0, 160, 0.2, 32)  # h_x = 1/160 not a double
    assert t["delta_hx"] != 0 and abs(t["delta_hx"]) <= U and abs(t["delta_cx"]) <= 2 * U
    third = ep.grid_constants(1.0, 96, 1 / 3, 32)  # h_y exact (division by 2^5), h_y^2 not
    assert third["delta_hy"] == 0 and third["delta_cy"] != 0


def test_pairwise_levels_match_the_module_tree():
    for m in (1, 2, 3, 5, 8, 9, 1000, 1025):
        lvl, v = 0, m
        while v > 1:
            v = (v + 1) // 2
            lvl += 1
        assert ep.pairwise_levels(m) == lvl == (ep.ceil_log2(m) if m > 1 else 0)


# --- B_E and the separate mathematical path on hard synthetic cases ---------------------------------------------------------------

def _case(kind, nx, ny, N, rng):
    X = rng.standard_normal((nx - 1, N))
    Yp = rng.standard_normal((ny - 1, N))
    if kind == "zero_prev":
        return X, np.zeros_like(Yp), X, Yp
    if kind == "stagnation":
        return X, Yp, X, np.nextafter(np.nextafter(Yp, np.inf), np.inf)
    if kind == "compensation":
        X[:, -1] = X[:, 0] * (1 + 2.0**-30)
        Yp[:, -1] = -Yp[:, 0]
        return X, Yp, X, Yp + 1e-3 * rng.standard_normal(Yp.shape)
    return X, Yp, X, Yp + 1e-3 * rng.standard_normal(Yp.shape)


_GRIDS = [(10, 9, 1.0, 1.0, 1), (12, 9, 1.0, 0.2, 2), (16, 16, 1.0, 1 / 3, 3), (11, 13, 1.3, 0.7, 3)]
_CASES = [(k, *gr) for k in ("early", "zero_prev", "stagnation", "compensation") for gr in _GRIDS
          if not (k == "compensation" and gr[-1] == 1)]  # compensation of two columns needs N >= 2


@pytest.mark.parametrize("kind, nx, ny, lx, ly, N", _CASES)
def test_b_e_and_control_path_on_synthetic_cases(kind, nx, ny, lx, ly, N):
    rng = np.random.default_rng(nx * 100 + ny + N)
    Xp, Yp, Xs, Ys = _case(kind, nx, ny, N, rng)
    g, gq = en.Grid(lx, nx, ly, ny), ep.GridQ(lx, nx, ly, ny)
    kc = ep.grid_constants(lx, nx, ly, ny, N=N)
    wp, ws = Xp @ Yp.T, Xs @ Ys.T
    dh = wp - ws
    fp, fs = ep.formation_check(Xp, Yp, wp), ep.formation_check(Xs, Ys, ws)
    assert fp["holds"] and fs["holds"]
    be = ep.b_e(gq, kc, N, fp, fs, dh, keep=True)
    Pp, Ps, Qd = en.pi_double(g, wp), en.pi_double(g, ws), en.q_double(g, dh)
    tp, ts = (fp["_Wt"], fp["_EWt"]), (fs["_Wt"], fs["_EWt"])
    dt = ep._sub_fields(tp, ts)
    pi = lambda x: ep.a13(gq, x, x) / 2 - gq.q * gq.hx * gq.hy * ep.p2(x[1]) * ep.isum(x[0])  # noqa: E731
    gamma = -(pi(tp) - pi(ts) - ep.a13(gq, dt, dt) / 2)  # VOK-2: f(w~_p, w~_s) = -gamma, exact
    E_s = ep.frac_of(Pp) - ep.frac_of(Ps) - ep.frac_of(Qd) + gamma
    mt = ep._eq().model_terms(g, Xp, Yp, Xs, Ys)
    R = mt["_R_Sigma"]
    assert be["PropD"] and abs(E_s) <= be["BE"] <= kc["K_star"] * R
    log = ep.OpLog()
    reps = {k: ep.replicate_energy(g, w, log, k, k != "d", keep=True) for k, w in (("p", wp), ("s", ws), ("d", dh))}
    ints = {"wp": be["wp"], "ws": be["ws"], "dh": be["dd"], "tp": tp, "ts": ts}
    cp = ep.control_path(gq, kc, g, {"wh_prev": wp, "wh_s": ws, "dh": dh}, reps, be["sums"], ints, be["bnd"], E_s, Pp, Ps, Qd)
    assert cp["ok"], {k: v for k, v in cp.items() if v is False}
    eps_s, eps_p, da, Bw = mt["_Rf_parts"]
    assert be["eps_s"] <= eps_s and be["eps_p"] <= eps_p and be["dh_a"] <= da and be["Bw"] <= Bw
    if kind == "zero_prev":
        assert be["eps_p"] == 0 and fp["exact_product_zero"]


# --- campaign: binding, CI determinism, sample, control rule --------------------------------------------------------------------

@pytest.fixture(scope="module")
def ci_runs(tmp_path_factory):
    outs = []
    for k in range(2):
        out = tmp_path_factory.mktemp(f"ci{k}")
        assert ep.main(["run", "--profile", "ci", "--out", str(out)]) == 0
        outs.append(out)
    return outs


def test_ci_profile_all_verified_and_deterministic(ci_runs):
    a, b = ((o / "rows.csv").read_bytes() for o in ci_runs)
    assert a == b
    s = json.loads((ci_runs[0] / "summary.json").read_text(encoding="utf-8"))
    assert s["status"] == {"VERIFIED": 16, "UNRESOLVED": 0, "FAIL": 0} and s["verdict_by_protocol"] == "QA_READY_FOR_REVIEW"
    for k in ("binding_required_4", "U1_prev_holds", "U1_s_holds", "U2_replication_bitwise", "E_s_le_B_E", "B_E_le_Kstar_R", "ctl_ok"):
        assert s[k] == "16/16", k
    assert s["N7_ok"] == "4/4" and s["U2"]["rows_with_violations"] == 0


def test_binding_mismatch_is_unresolved(tmp_path):
    eq = ep._eq()
    ga = eq._ga()
    rows, arrays, _ = eq.ci_specs_rows(tmp_path)
    rs = [r for r in rows if r["N"] == "1"]
    sp = ga._ci_spec(rs[0])
    npz = str(arrays / ga._q3().seq_file(sp))
    ids = [ga.row_id(r) for r in rs]
    hist = {h["id"]: h for h in eq.seq_job(sp, rs, npz, ids, {}, False)["rows"]}
    first = ids[1]
    hist[first] = dict(hist[first], P_hat_prev=np.nextafter(hist[first]["P_hat_prev"], np.inf))
    res = ep.seq_job(sp, rs, npz, hist, ids, {})
    st = {r["id"]: r["status"] for r in res["rows"]}
    assert st[first].startswith("UNRESOLVED") and "binding" in st[first]
    assert all(v == "VERIFIED" for k, v in st.items() if k != first)


def test_sample_hashes_and_control_rule_of_the_full_campaign():
    meta_ga = json.loads((ROOT / "docs/results/r5_cccc_q4_gamma_audit/meta.json").read_text(encoding="utf-8"))
    meta_eq = json.loads((ROOT / "docs/results/r5_cccc_q4_energy_bc/meta.json").read_text(encoding="utf-8"))
    assert {e["file"]: e["sha256"] for e in meta_ga["npz"]} == {e["file"]: e["sha256"] for e in meta_eq["npz"]}
    assert len(meta_eq["npz"]) == 123
    ga = ep._eq()._ga()
    q3 = ga.read_q3_rows(ep.Q3_CSV)
    with open(ep.EQ_ROWS, encoding="utf-8") as f:
        eqr = list(csv.DictReader(f))
    assert {ga.row_id(r) for r in q3} == {r["id"] for r in eqr} and len(eqr) == 984
    ctl = [r for r in q3 if int(r["nx"]) * int(r["ny"]) <= ep.CONTROL_MAX_NODES]
    assert len(ctl) == 288 and {int(r["N"]) for r in ctl} == {1, 2, 3} and {int(r["j_s"]) for r in ctl} == set(range(1, 9))
    zero = [r for r in eqr if float(r["eps_a_C_prev_up"]) == 0.0]
    assert len(zero) == 123 and {r["s"] for r in zero} == {"1"}


def test_used_modules_and_data_unchanged():
    for rel in ("src/kkvim_reference", "src/kkvim", "docs/results/r5_cccc_q3_prequalification", "docs/results/r5_cccc_q4_gamma_audit",
                "docs/results/r5_cccc_q4_gamma_intervals", "docs/results/r5_cccc_q4_c_assembly_e4", "docs/results/r5_cccc_q4_energy_bc",
                "scripts/r5_cccc_q4_energy_bc_audit.py", "scripts/r5_cccc_q4_gamma_audit.py", "scripts/r5_cccc_q4_gamma_interval_audit.py",
                "scripts/r5_cccc_q4_c_assembly_e4_audit.py", "scripts/r5_cccc_q3_prequalification.py",
                "docs/R5_CCCC_Q4_ENERGY_ERROR_BOUND.study.md", "docs/R5_CCCC_Q4_ENERGY_ERROR_BOUND.independent-review.md"):
        out = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", rel], cwd=ROOT, check=False)
        assert out.returncode == 0, rel


def test_no_import_of_the_working_kkvim():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    names = [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
    names += [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    assert not [m for m in names if m == "kkvim" or m.startswith("kkvim.")]
    assert math.isfinite(float(ep.REVIEWER_KSTAR_MINUS_1)) and sys.version_info >= (3, 13)
