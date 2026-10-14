"""QA, R3 solver qualification, steps 1-2 of docs/R3_NUMERICAL_SPEC.md 12 and group B (section 10), after step 0.

Group A (9.1; criteria of NUMERICAL_SPEC 18.1, D-039, D-040 / RV-1, RV-3; nothing new):
- A-1a (T-1D-1, R2 sets and the RV-1 set kappa_c = 16/l), A-1b, A-2a-c (T-1D-2): the one-dimensional solver of kkvim;
- A-3: coefficients of X0 against ChV-6 on n_x = 4, 7, 20, 320 and every n_x of the chains (up to 640), a = 1 and 2.9;
- A-4a/b: s = 1 against ParsevalRecursion(a, b, s_max).field_on_grid (r != 1; table 9.2, variant (1)) and against
  levy_w1 (r = 1, R2); p on the fixed P-C-min pairs (table 7.2), monotonicity of e_c on the admitted levels (7.3),
  Richardson only on the admitted Q-Q pairs; RV-3: budget / min(E(L), E(2L)) > 1/100 -> UNRESOLVED;
- A-5a-d: s = 1, 2, 3 against the Parseval recursion (field, centre, invariants beta, gamma, chi; RV-3 with the
  field_error / invariant_error budgets), normalization (n = 80, s <= 6, QA path without I-4), counters;
- A-6, A-8 (reflections only), A-9 (both families, grids of 9.1), A-11 (strict mode), A-12 (nodes, exact zeros);
- stopping controller: QA re-implementation of 11.1/11.3 on the fields of every run; report of 11.5 (coefficients
  before normalization, control points incl. quarters);
- unequal grids (D-032 item 10; one-axis refinement sequences of 9.1): observation without a tolerance.
Group B (10; D-025): against NavierRectangle (M = 2559), continuous Parseval sequence, two starts (plate b x a),
energies; no PASS/FAIL.

Geometry is passed to NavierRectangle and ParsevalRecursion as Python float (D-042 note in STATUS).

    uv run python scripts/r3_solver_qualification.py     # writes docs/results/r3/solver/qual/ and the report blocks
"""

import argparse
import csv
import datetime
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from fractions import Fraction
from importlib import metadata
from pathlib import Path

import numpy as np

from kkvim.controller import strict_parameters
from kkvim.first_halfstep import NOT_A_NODE_MESSAGE, solve_halfstep
from kkvim.grid import Grid1D
from kkvim.linear_solver import apply_k, solve_k
from kkvim.rectangle import rectangle_first_halfstep
from kkvim.rectangle_run import halfstep_report, run_rectangle_physical, run_rectangle_until_stopped
from kkvim_reference.levy import levy_series_w, levy_w1
from kkvim_reference.navier import NavierRectangle
from kkvim_reference.parseval_recursion import ParsevalRecursion

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "results" / "r3" / "solver" / "qual"
REPORT = ROOT / "docs" / "R3_SOLVER_QUALIFICATION.qa-report.md"
sys.path.insert(0, str(ROOT / "scripts"))
import r3_solver_step0 as step0  # noqa: E402  (chains, ratios and the independent energy formulas of step 0)

U = 2.0**-53
P_WINDOW = (1.9, 2.1)
P_WINDOW_1D = (1.95, 2.05)
RE_RATIO = 0.05
RV3 = 1.0 / 100.0
TOL = {"A-1a": 1e-11, "A-2v": 1e-14, "A-3": 1e-13, "A-4v": 5e-13, "A-5g": 1e-12, "A-6": 1e-12, "A-8": 1e-12,
       "A-9": 1e-11, "A-10b_tail": 1e-14, "A-10b_transfer": 1e-12, "A-11": 0.01}
A4V_CENTRE = 0.004109346839
RATIO = {label: r for label, r, _ in step0.RATIOS}
CHAINS = {(label, family): chain for label, _, family, chain in step0.chains()}


def _p(*pairs):
    return tuple(((a, a), (b, b)) if isinstance(a, int) else (a, b) for a, b in pairs)


I_LOW, I_HIGH = _p((80, 160), (160, 320)), _p((160, 320), (320, 640))
# table 7.2: A-4 p, A-5 p (also A-5v), A-4 Richardson, A-5 Richardson (empty: UNRESOLVED; one pair: limited scope)
PAIRS = {
    ("1/5", "I"): (I_LOW, I_HIGH, I_LOW, ()),
    ("1/5", "II"): (_p(((160, 32), (320, 64)), ((320, 64), (640, 128))),) * 3 + ((),),
    ("1/3", "I"): (I_LOW, I_HIGH, I_LOW, _p((320, 640))),
    ("1/3", "II"): (_p(((144, 48), (288, 96)), ((288, 96), (576, 192))),) * 3 + (_p(((288, 96), (576, 192))),),
    ("1/2", "I"): (I_LOW, I_HIGH, I_LOW, I_HIGH),
    ("1/2", "II"): (_p(((80, 40), (160, 80)), ((160, 80), (320, 160))), _p(((160, 80), (320, 160)), ((320, 160), (640, 320))),
                    _p(((80, 40), (160, 80)), ((160, 80), (320, 160))), _p(((160, 80), (320, 160)), ((320, 160), (640, 320)))),
    ("2/3", "I"): (I_LOW, I_HIGH, I_LOW, I_HIGH),
    ("2/3", "II"): (_p(((120, 80), (240, 160)), ((240, 160), (480, 320))),) * 4,
    ("1", "I"): (I_LOW,) * 4,
    ("7/5", "I"): (I_HIGH,) * 4,
    ("7/5", "II"): (_p(((80, 112), (160, 224)), ((160, 224), (320, 448))),) * 4,
    ("3/2", "I"): (I_HIGH,) * 4,
    ("3/2", "II"): (_p(((80, 120), (160, 240)), ((160, 240), (320, 480))),) * 4,
    ("2", "I"): (I_HIGH,) * 4,
    ("2", "II"): (_p(((80, 160), (160, 320)), ((160, 320), (320, 640))),) * 4,
    ("3", "I"): (I_HIGH, I_HIGH, _p((320, 640)), _p((320, 640))),
    ("3", "II"): (_p(((48, 144), (96, 288)), ((96, 288), (192, 576))),) * 2 + (_p(((96, 288), (192, 576))),) * 2,
    ("5", "I"): (I_HIGH, I_HIGH, (), ()),
    ("5", "II"): (_p(((32, 160), (64, 320)), ((64, 320), (128, 640))),) * 2 + ((), ()),
}


def _lv(*levels):
    return tuple((n, n) if isinstance(n, int) else n for n in levels)


ALL_I = _lv(20, 40, 80, 160, 320, 640)
# table 7.3: admitted levels for the monotonicity of A-4 and A-5
MONOTONE = {
    ("1/5", "I"): (ALL_I, _lv(160, 320, 640)), ("1/5", "II"): (_lv((160, 32), (320, 64), (640, 128)),) * 2,
    ("1/3", "I"): (ALL_I, _lv(80, 160, 320, 640)), ("1/3", "II"): (_lv((72, 24), (144, 48), (288, 96), (576, 192)),) * 2,
    ("1/2", "I"): (ALL_I, _lv(40, 80, 160, 320, 640)), ("1/2", "II"): (CHAINS[("1/2", "II")],) * 2,
    ("2/3", "I"): (ALL_I, _lv(40, 80, 160, 320, 640)), ("2/3", "II"): (CHAINS[("2/3", "II")],) * 2,
    ("1", "I"): (ALL_I, ALL_I),
    ("7/5", "I"): (_lv(40, 80, 160, 320, 640),) * 2, ("7/5", "II"): (CHAINS[("7/5", "II")],) * 2,
    ("3/2", "I"): (_lv(40, 80, 160, 320, 640),) * 2, ("3/2", "II"): (CHAINS[("3/2", "II")],) * 2,
    ("2", "I"): (_lv(40, 80, 160, 320, 640),) * 2, ("2", "II"): (CHAINS[("2", "II")],) * 2,
    ("3", "I"): (_lv(80, 160, 320, 640),) * 2, ("3", "II"): (_lv((24, 72), (48, 144), (96, 288), (192, 576)),) * 2,
    ("5", "I"): (_lv(160, 320, 640),) * 2, ("5", "II"): (_lv((32, 160), (64, 320), (128, 640)),) * 2,
}
# A-9 grids (9.1)
A9_GRIDS = {"1/5": (320, 64), "1/3": (288, 96), "1/2": (320, 160), "2/3": (240, 160), "7/5": (160, 224),
            "3/2": (160, 240), "2": (160, 320), "3": (96, 288), "5": (64, 320)}
# one-axis refinement (9.1): top level of the family II chain (r = 1: (640, 640))
TOP_II = {label: (CHAINS[(label, "II")][-1] if label != "1" else (640, 640)) for label, _, _ in step0.RATIOS}
QUARTERS = (("1/4", "1/4"), ("1/4", "1/2"), ("1/2", "1/4"), ("3/4", "1/4"), ("1/4", "3/4"))


# --- cached computations ---------------------------------------------------------------------------------------------

_runs, _refs, _navier = {}, {}, {}


def run(a, b, nx, ny, strict=False):
    key = (a, b, nx, ny, strict)
    if key not in _runs:
        kw = {}
        if strict:
            kw["tol_inf"], kw["tol_E"] = strict_parameters(nx, ny)
        _runs[key] = run_rectangle_until_stopped(a, b, 1.0, 1.0, nx, ny, **kw)
    return _runs[key]


def parseval(label):
    if label not in _refs:
        _refs[label] = ParsevalRecursion(a=1.0, b=float(RATIO[label]), q=1.0, D=1.0, s_max=4)
    return _refs[label]


def field(res, s):
    return res.history.field(s)


def centre(w):
    return w[(w.shape[0] - 1) // 2, (w.shape[1] - 1) // 2]


def node(w, fx, fy):
    i, j = Fraction(fx) * (w.shape[0] - 1), Fraction(fy) * (w.shape[1] - 1)
    assert i.denominator == 1 and j.denominator == 1
    return w[int(i), int(j)]


def sci(v, d=6):
    return "" if v is None else f"{v:.{d}e}"


def window(p, w=P_WINDOW):
    return w[0] <= p <= w[1]


# --- A-1a, A-1b, A-2 --------------------------------------------------------------------------------------------------

def one_dimensional():
    rows = []
    sets = [("R2 (1,0,0)", lambda l: (1.0, 0.0, 0.0)), ("R2 (1,3,5)", lambda l: (1.0, 3.0, 5.0)),
            ("R2 (0.5,pi^2/2,pi^4/2)", lambda l: (0.5, math.pi**2 / 2, math.pi**4 / 2)),
            ("RV-1 kappa_c=16/l", lambda l: (1.0, (16 / l) ** 2, (16 / l) ** 4))]
    for name, coeff in sets:
        for ell in (1.0, 2.5):
            A_, B_, C_ = coeff(ell)
            kf = math.pi / ell
            errs = {}
            for n in (20, 40, 80, 160, 320, 640):
                grid = Grid1D(ell, n)
                f = np.sin(math.pi * np.arange(1, n) / n)  # sin(kappa_f xi_i), xi_i = i l / n
                g = (A_ * kf**4 + 2 * B_ * kf**2 + C_) * f
                fh = solve_k(grid, A_, B_, C_, 1.0, g).y
                mu1 = 4.0 / grid.h**2 * math.sin(math.pi / (2 * n)) ** 2
                rho = (A_ * kf**4 + 2 * B_ * kf**2 + C_) / (A_ * mu1**2 + 2 * B_ * mu1 + C_)
                e_a = float(np.max(np.abs(fh - rho * f)) / np.max(np.abs(f)))
                e_inf = float(np.max(np.abs(fh - f)) / np.max(np.abs(f)))
                errs[n] = e_inf
                rows.append({"check": "A-1a", "case": name, "ell": ell, "n": n, "value": sci(e_a), "tolerance": TOL["A-1a"],
                             "verdict": "PASS" if e_a <= TOL["A-1a"] else "FAIL"})
            for n in (20, 40, 80, 160):
                p = math.log2(errs[n] / errs[2 * n])
                mandatory = name.startswith("R2")
                rows.append({"check": "A-1b" if mandatory else "A-1b (diag., RV-1)", "case": name, "ell": ell,
                             "n": f"{n}-{2 * n}", "value": sci(p), "tolerance": "[1.95, 2.05]",
                             "verdict": ("PASS" if window(p, P_WINDOW_1D) else "FAIL") if mandatory else "DIAGNOSTIC"})
    # T-1D-2: f = (y^4 - 2 l y^3 + l^3 y)/24, l = 1
    for A_, B_, C_ in ((1.0, 0.0, 0.0), (1.0, 3.0, 5.0)):
        name = f"T-1D-2 ({A_:g},{B_:g},{C_:g})"
        errs, sol = {}, {}
        for n in (20, 40, 80, 160, 320, 640):
            y = np.arange(1, n) / n
            f = (y**4 - 2 * y**3 + y) / 24
            g = A_ - 2 * B_ * (y**2 - y) / 2 + C_ * f
            fh = solve_k(Grid1D(1.0, n), A_, B_, C_, 1.0, g).y
            errs[n], sol[n] = float(np.max(np.abs(fh - f)) / np.max(np.abs(f))), (fh, f)
        mono = all(errs[n] > errs[2 * n] for n in (20, 40, 80, 160, 320))
        rows.append({"check": "A-2a monotone", "case": name, "ell": 1.0, "n": "20..640", "value": str(mono),
                     "tolerance": "decreasing", "verdict": "PASS" if mono else "FAIL"})
        for n in (80, 160):
            p = math.log2(errs[n] / errs[2 * n])
            rows.append({"check": "A-2a", "case": name, "ell": 1.0, "n": f"{n}-{2 * n}", "value": sci(p),
                         "tolerance": "[1.9, 2.1]", "verdict": "PASS" if window(p) else "FAIL"})
            (fh, f), (fh2, _) = sol[n], sol[2 * n]
            re = (4 * fh2[1::2] - fh) / 3
            ratio = float(np.max(np.abs(re - f)) / np.max(np.abs(fh2[1::2] - f)))
            rows.append({"check": "A-2b", "case": name, "ell": 1.0, "n": f"{n}-{2 * n}", "value": sci(ratio),
                         "tolerance": RE_RATIO, "verdict": "PASS" if ratio <= RE_RATIO else "FAIL"})
        for n in (4, 20):
            grid = Grid1D(1.0, n)
            e1 = np.zeros(n - 1)
            e1[0] = 1.0
            k = apply_k(grid, A_, B_, C_, 1.0, e1)
            h = grid.h
            expect = np.zeros(n - 1)
            expect[:3] += A_ * np.array([5.0, -4.0, 1.0]) / h**4
            expect[:2] += 2 * B_ * np.array([2.0, -1.0]) / h**2
            expect[0] += C_
            dev = float(np.max(np.abs(k - expect)) / np.max(np.abs(expect)))
            rows.append({"check": "A-2v", "case": name, "ell": 1.0, "n": n, "value": sci(dev), "tolerance": TOL["A-2v"],
                         "verdict": "PASS" if dev <= TOL["A-2v"] else "FAIL"})
    return rows


# --- A-3 --------------------------------------------------------------------------------------------------------------

def a3():
    rows = []
    nxs = sorted({4, 7, 20, 320} | {g[0] for chain in CHAINS.values() for g in chain if g[0] >= 4})
    for a in (1.0, 2.9):
        for nx in nxs:
            res = rectangle_first_halfstep(a, a, 1.0, 1.0, nx, 20)  # X0 coefficients depend only on a, n_x
            c = res.coefficients
            h = a / nx
            mu1 = 4.0 / h**2 * math.sin(math.pi / (2 * nx)) ** 2
            exact = (a / 2, a / 2 * mu1, a / 2 * mu1**2, h / math.tan(math.pi / (2 * nx)))
            dev = max(abs(x - y) / abs(y) for x, y in zip((c.A, c.B, c.C, c.F), exact))
            rows.append({"a": a, "nx": nx, "max_rel_dev": sci(dev), "verdict": "PASS" if dev <= TOL["A-3"] else "FAIL"})
    return rows


# --- per level: A-4/A-5 errors, A-6, A-8, A-11, A-12, controller, report -----------------------------------------------

def a5_errors(res, label, grid, s):
    P = parseval(label)
    w, ref = field(res, s), P.field_on_grid(s, *grid)
    e_inf = float(np.max(np.abs(w - ref)) / np.max(np.abs(ref)))
    e_c = float(abs(centre(w) - centre(ref)) / abs(centre(ref)))
    G = res.history.record(s).coefficients
    inv = (G.B / G.A, G.C / G.A, G.F * G.F / G.A)
    inv_ref = P.coefficients(s).invariants()
    inv_err = [abs(x - y) / abs(y) for x, y in zip(inv, inv_ref)]
    budget = P.halfstep(s).budget
    return {"E_inf": e_inf, "e_c": e_c, "beta": inv_err[0], "gamma": inv_err[1], "chi": inv_err[2],
            "budget_field": budget.field_error,
            "budget_centre": budget.field_error * float(np.max(np.abs(ref))) / abs(centre(ref)),
            "budget_invariant": budget.invariant_error}


def a4_reference(label, grid):
    if label == "1":
        n = grid[0]
        X, Y = np.meshgrid(np.arange(n + 1) / n, np.arange(n + 1) / n, indexing="ij")
        return levy_w1(X, Y, a=1.0, q=1.0, D=1.0), None
    P = parseval(label)
    ref = P.field_on_grid(1, *grid)
    return ref, P.halfstep(1).budget.field_error


def qa_decision(eps_rows, s, tol_inf=1e-10, tol_E=1e-8, s_min=3, s_max=100):
    """Independent implementation of 11.3 on (s, eps_inf, eps_E)."""
    ok = [ei <= tol_inf and ee <= tol_E for _, ei, ee in eps_rows]
    if s >= s_min and len(ok) >= 2 and ok[-1] and ok[-2]:
        return "CONVERGED"
    count = 0
    for i, (_, _, ee) in enumerate(eps_rows):
        th = None if i == 0 or eps_rows[i - 1][2] == 0 else ee / eps_rows[i - 1][2]
        count = count + 1 if th is not None and th >= 0.9 and ee > tol_E else 0
    if count >= 10:
        return "STAGNATED"
    return "MAX_HALF_STEPS" if s >= s_max else None


def level_checks(label, family, grid):
    nx, ny = grid
    b = float(RATIO[label])
    res = run(1.0, b, nx, ny)
    recs = res.history.records
    hx, hy = 1.0 / nx, b / ny
    out = {"r": label, "family": family, "nx": nx, "ny": ny, "qualification": step0.qualification(grid),
           "status": res.status, "s_stop": len(recs)}
    diag = [r.solver_diagnostics for r in recs]
    out["A6_eta_max"], out["A6_imag_max"] = max(d.eta for d in diag), max(d.imag_ratio for d in diag)
    out["A6_refined"] = sum(bool(d.refined) for d in diag)
    refl, zero = 0.0, True
    for s in range(1, len(recs) + 1):
        w = field(res, s)
        m = float(np.max(np.abs(w)))
        refl = max(refl, float(np.max(np.abs(w - w[::-1, :]))) / m, float(np.max(np.abs(w - w[:, ::-1]))) / m)
        zero = zero and not (np.any(w[0, :]) or np.any(w[-1, :]) or np.any(w[:, 0]) or np.any(w[:, -1]))
    out["A8_reflection_max"], out["A12_boundary_zero"] = refl, zero
    # controller: QA re-implementation of 11.1 and 11.3 on the fields of the run
    eps_rows, dev_eps, decision_s = [], 0.0, None
    for prev, cur in zip(recs, recs[1:]):
        w0, w1 = step0.field(prev), step0.field(cur)
        d = w1 - w0
        ei = float(np.max(np.abs(d)) / np.max(np.abs(w1)))
        ee = math.sqrt(step0.a_form(d, hx, hy) / step0.a_form(w1, hx, hy))
        eps_rows.append((cur.s, ei, ee))
        ch = res.changes[cur.s - 2]
        dev_eps = max(dev_eps, abs(ch.eps_inf - ei) / ei if ei else abs(ch.eps_inf), abs(ch.eps_E - ee) / ee if ee else abs(ch.eps_E))
        if decision_s is None and qa_decision(eps_rows, cur.s):
            decision_s = (cur.s, qa_decision(eps_rows, cur.s))
    out["controller_qa"] = f"{decision_s[1]}@{decision_s[0]}" if decision_s else "none"
    out["controller_match"] = decision_s == (len(recs), res.status)
    out["eps_rel_dev_max"] = dev_eps
    # A-11
    st = run(1.0, b, nx, ny, strict=True)
    s_stop = float(np.max(np.abs(st.last_field() - res.last_field())) / np.max(np.abs(res.last_field())))
    out["A11_S_stop"], out["A11_status_strict"], out["A11_s_strict"] = s_stop, st.status, len(st.history)
    # report of 11.5: before-normalization coefficients and control points incl. quarters
    rep = halfstep_report(_result_of(res, b, nx, ny), control_points=(step0_centre(),) + QUARTERS)
    # "before normalization" (NUMERICAL_SPEC 11.5) = coefficients of the raw computed factor of s - 1, before I-4:
    # compared with the independent QA coefficients of that raw factor; the scaling relation (c^2 for A, B, C and c for
    # F) is recorded as a diagnostic (C contains second differences, whose cancellation is of order u n^2)
    rel_ok, ctrl_ok, dev_raw, dev_scale = True, True, 0.0, 0.0
    for i, h in enumerate(rep.halfsteps):
        w_bar = field(res, h.s)  # q = D = a = 1: w_bar = w
        for point, value in h.control_values:
            ctrl_ok = ctrl_ok and value == node(w_bar, *point)
        rel_ok = rel_ok and h.k == h.s // 2 and h.direction == ("Y" if h.s % 2 else "X")
        if h.s == 1:
            rel_ok = rel_ok and h.coefficients_before_normalization is None
            continue
        prev = recs[i - 1]
        bef, aft = h.coefficients_before_normalization, h.coefficients
        qa = step0.coefficients(np.asarray(prev.raw), h.fixed_grid[0] / h.fixed_grid[1])
        dev_raw = max(dev_raw, max(abs(x - y) / abs(y) for x, y in zip((bef.A, bef.B, bef.C, bef.F), qa)))
        expect = (aft.A * prev.c**2, aft.B * prev.c**2, aft.C * prev.c**2, aft.F * prev.c)
        dev_scale = max(dev_scale, max(abs(x - y) / abs(y) for x, y in zip((bef.A, bef.B, bef.C, bef.F), expect)))
    rel_ok = rel_ok and dev_raw <= 1e-12
    out["report_before_norm_ok"], out["report_control_points_ok"] = rel_ok, ctrl_ok
    out["report_before_vs_qa_raw"], out["report_before_vs_scaled_after"] = dev_raw, dev_scale
    out["report_rows"] = len(rep.halfsteps)
    return out


def step0_centre():
    return ("1/2", "1/2")


def _result_of(res, b, nx, ny):
    from kkvim.rectangle_run import RectangleInput, RectangleResult
    return RectangleResult(res.status, RectangleInput(a=1.0, b=b, t=1.0, E=1.0, nu=0.3, q=1.0, n_x=nx, n_y=ny), 1.0, res,
                           res.error)


# --- A-4, A-5 levels and pairs -----------------------------------------------------------------------------------------

def accuracy_levels():
    a4, a5 = [], []
    for (label, family), chain in CHAINS.items():
        b = float(RATIO[label])
        for grid in chain:
            res = run(1.0, b, *grid)
            w1 = field(res, 1)
            ref, budget = a4_reference(label, grid)
            e_c = float(abs(centre(w1) - centre(ref)) / abs(centre(ref)))
            e_inf = float(np.max(np.abs(w1 - ref)) / np.max(np.abs(ref)))
            row = {"r": label, "family": family, "nx": grid[0], "ny": grid[1], "e_c": sci(e_c), "e_inf": sci(e_inf),
                   "reference": "levy_w1 (R2)" if label == "1" else "Parseval s=1",
                   "budget_field": sci(budget), "budget_centre": sci(None if budget is None else
                                                                    budget * float(np.max(np.abs(ref))) / abs(centre(ref)))}
            if RATIO[label] >= 1 and label != "1":  # independent cross-check of the reference: levy_series_w(M = 1)
                X, Y = np.meshgrid(np.arange(grid[0] + 1) / grid[0], b * np.arange(grid[1] + 1) / grid[1], indexing="ij")
                lv = levy_series_w(X, Y, a=1.0, b=b, q=1.0, D=1.0, M=1)
                row["levy_M1_vs_parseval"] = sci(float(np.max(np.abs(lv - ref)) / np.max(np.abs(ref))))
            else:
                row["levy_M1_vs_parseval"] = ""
            a4.append(row)
            for s in (1, 2, 3):
                e = a5_errors(res, label, grid, s)
                rec, href = res.history.record(s), parseval(label).halfstep(s)
                a5.append({"r": label, "family": family, "nx": grid[0], "ny": grid[1], "s": s,
                           **{k: sci(v) for k, v in e.items()},
                           "A5d_counters": rec.s == s and rec.k == s // 2 == href.k and rec.direction == href.variable})
    return a4, a5


def pair_rows(a4, a5):
    idx4 = {(r["r"], r["family"], int(r["nx"]), int(r["ny"])): r for r in a4}
    idx5 = {(r["r"], r["family"], int(r["nx"]), int(r["ny"]), int(r["s"])): r for r in a5}
    rows = []

    def add(check, label, family, pair, quantity, eL, e2L, budget, mandatory, kind="p", extra=None):
        p = math.log2(eL / e2L)
        rv3 = None if budget is None else budget / min(eL, e2L)
        if kind == "p":
            ok = window(p)
            verdict = "UNRESOLVED (RV-3)" if rv3 is not None and rv3 > RV3 else ("PASS" if ok else "FAIL")
        else:
            verdict = extra
        rows.append({"check": check, "r": label, "family": family, "pair": f"{pair[0]}->{pair[1]}", "quantity": quantity,
                     "E_L": sci(eL), "E_2L": sci(e2L), "p": sci(p, 4), "rv3_ratio": sci(rv3, 3),
                     "mandatory": mandatory, "verdict": verdict if mandatory else f"DIAGNOSTIC ({verdict})"})

    for (label, family), chain in CHAINS.items():
        p4, p5, r4, r5 = PAIRS[(label, family)]
        b = float(RATIO[label])
        adjacent = list(zip(chain, chain[1:]))
        for pair in adjacent:
            q = all(min(g) >= 20 for g in pair)
            L, L2 = idx4[(label, family, *pair[0])], idx4[(label, family, *pair[1])]
            bud = lambda key: None if L[key] == "" else float(L[key])  # noqa: E731
            for quantity, key, bkey in (("e_c", "e_c", "budget_centre"), ("e_inf", "e_inf", "budget_field")):
                add("A-4a p", label, family, pair, quantity, float(L[key]), float(L2[key]), bud(bkey), pair in p4)
            if pair in r4 or q:
                rows.extend(richardson4(label, family, pair, b, pair in r4, rows, len(r4) == 1))
            for s in (1, 2, 3):
                M, M2 = idx5[(label, family, *pair[0], s)], idx5[(label, family, *pair[1], s)]
                for quantity, key, bkey in (("E_inf", "E_inf", "budget_field"), ("e_c", "e_c", "budget_centre")):
                    add("A-5a p", label, family, pair, f"s={s} {quantity}", float(M[key]), float(M2[key]),
                        float(M[bkey]), pair in p5)
                for inv in ("beta", "gamma", "chi"):
                    add("A-5v p", label, family, pair, f"s={s} {inv}", float(M[inv]), float(M2[inv]),
                        float(M["budget_invariant"]), pair in p5)
                if pair in r5 or q:
                    rows.append(richardson5(label, family, pair, b, s, pair in r5, len(r5) == 1, rows))
        for check, table in (("A-4b Richardson", r4), ("A-5b Richardson", r5)):
            if not table:
                rows.append({"check": check, "r": label, "family": family, "pair": "—", "quantity": "—", "E_L": "",
                             "E_2L": "", "p": "", "rv3_ratio": "", "mandatory": True,
                             "verdict": "UNRESOLVED (нет пар Q–Q, табл. 7.2)"})
    return rows


def _p_of(rows, check, label, family, pair, quantity):
    for r in rows:
        if r["check"] == check and (r["r"], r["family"], r["pair"], r["quantity"]) == (label, family, f"{pair[0]}->{pair[1]}", quantity):
            return float(r["p"])
    raise KeyError((check, label, family, pair, quantity))


def richardson4(label, family, pair, b, mandatory, rows, one_pair):
    """A-4b: w_RE = (4 w_2L - w_L)/3 on the nodes of L; centre and max norm; only if p is in [1.9, 2.1] on the pair."""
    (g, g2) = pair
    coarse, fine = field(run(1.0, b, *g), 1), field(run(1.0, b, *g2), 1)[::2, ::2]
    ref, budget = a4_reference(label, g)
    abs_budget = None if budget is None else budget * float(np.max(np.abs(ref)))  # absolute error of the reference
    out = []
    for quantity, num, den, pkey in (
            ("centre", abs((4 * centre(fine) - centre(coarse)) / 3 - centre(ref)), abs(centre(fine) - centre(ref)), "e_c"),
            ("field", float(np.max(np.abs((4 * fine - coarse) / 3 - ref))), float(np.max(np.abs(fine - ref))), "e_inf")):
        p = _p_of(rows, "A-4a p", label, family, pair, pkey)
        ratio = num / den
        rv3 = None if abs_budget is None else abs_budget / den
        if not window(p):
            verdict = "NOT APPLIED (p outside [1.9, 2.1])"
        elif rv3 is not None and rv3 > RV3:
            verdict = "UNRESOLVED (RV-3)"
        else:
            verdict = ("PASS" if ratio <= RE_RATIO else "FAIL") + (" (ограниченный объём: одна пара)" if one_pair else "")
        out.append({"check": "A-4b Richardson", "r": label, "family": family, "pair": f"{g}->{g2}", "quantity": quantity,
                    "E_L": sci(num), "E_2L": sci(den), "p": sci(p, 4), "rv3_ratio": sci(rv3, 3), "mandatory": mandatory,
                    "verdict": verdict if mandatory else f"DIAGNOSTIC ({verdict})", "ratio": sci(ratio, 4)})
    return out


def richardson5(label, family, pair, b, s, mandatory, one_pair, rows):
    (g, g2) = pair
    coarse, fine = field(run(1.0, b, *g), s), field(run(1.0, b, *g2), s)[::2, ::2]
    P = parseval(label)
    ref = P.field_on_grid(s, *g)
    num = abs((4 * centre(fine) - centre(coarse)) / 3 - centre(ref))
    den = abs(centre(fine) - centre(ref))
    p = _p_of(rows, "A-5a p", label, family, pair, f"s={s} e_c")
    rv3 = P.halfstep(s).budget.field_error * float(np.max(np.abs(ref))) / den
    ratio = num / den
    if not window(p):
        verdict = "NOT APPLIED (p outside [1.9, 2.1])"
    elif rv3 > RV3:
        verdict = "UNRESOLVED (RV-3)"
    else:
        verdict = ("PASS" if ratio <= RE_RATIO else "FAIL") + (" (ограниченный объём: одна пара)" if one_pair else "")
    return {"check": "A-5b Richardson", "r": label, "family": family, "pair": f"{g}->{g2}", "quantity": f"s={s} centre",
            "E_L": sci(num), "E_2L": sci(den), "p": sci(p, 4), "rv3_ratio": sci(rv3, 3), "mandatory": mandatory,
            "verdict": verdict if mandatory else f"DIAGNOSTIC ({verdict})", "ratio": sci(ratio, 4)}


def monotone_rows(a4, a5):
    idx4 = {(r["r"], r["family"], int(r["nx"]), int(r["ny"])): r for r in a4}
    idx5 = {(r["r"], r["family"], int(r["nx"]), int(r["ny"]), int(r["s"])): r for r in a5}
    rows = []
    for (label, family), (lv4, lv5) in MONOTONE.items():
        e = [float(idx4[(label, family, *g)]["e_c"]) for g in lv4]
        rows.append({"check": "A-4a monotone e_c", "r": label, "family": family, "levels": " ".join(map(str, lv4)),
                     "values": " ".join(f"{v:.3e}" for v in e),
                     "verdict": "PASS" if all(x > y for x, y in zip(e, e[1:])) else "FAIL"})
        for s in (1, 2, 3):
            e = [float(idx5[(label, family, *g, s)]["e_c"]) for g in lv5]
            rows.append({"check": f"A-5a monotone e_c s={s}", "r": label, "family": family,
                         "levels": " ".join(map(str, lv5)), "values": " ".join(f"{v:.3e}" for v in e),
                         "verdict": "PASS" if all(x > y for x, y in zip(e, e[1:])) else "FAIL"})
            e = [float(idx5[(label, family, *g, s)]["E_inf"]) for g in lv5]
            rows.append({"check": f"A-5a monotone E_inf s={s} (diag.)", "r": label, "family": family,
                         "levels": " ".join(map(str, lv5)), "values": " ".join(f"{v:.3e}" for v in e),
                         "verdict": "DIAGNOSTIC (" + ("decreasing" if all(x > y for x, y in zip(e, e[1:])) else "not decreasing") + ")"})
    return rows


# --- A-4v, A-5g, A-9, A-11 (E_disc), A-12 nodes --------------------------------------------------------------------

def a4v():
    X = np.array([[0.5]])
    c = float(levy_w1(X, X, a=1.0, q=1.0, D=1.0)[0, 0])
    return {"check": "A-4v", "value": sci(c, 15), "deviation": sci(abs(c - A4V_CENTRE)),
            "verdict": "PASS" if abs(c - A4V_CENTRE) <= TOL["A-4v"] else "FAIL"}


def a5g():
    rows = []
    for label, r, _ in step0.RATIOS:
        b = float(r)
        nx = ny = 80
        gx, gy = Grid1D(1.0, nx), Grid1D(b, ny)
        X = np.sin(math.pi * np.arange(1, nx) / nx)
        Y = None
        res = run_rectangle_until_stopped(1.0, b, 1.0, 1.0, nx, ny, s_min=6, s_max=6)
        worst_f = worst_i = 0.0
        for s in range(1, 7):
            if s % 2:
                st = solve_halfstep(gx, X, gy, 1.0, 1.0)
                Y = st.raw
            else:
                st = solve_halfstep(gy, Y, gx, 1.0, 1.0)
                X = st.raw
            w_raw = np.outer(gx.with_boundary(X), gy.with_boundary(Y))
            w = field(res, s)
            worst_f = max(worst_f, float(np.max(np.abs(w - w_raw)) / np.max(np.abs(w_raw))))
            G, Gr = res.history.record(s).coefficients, st.coefficients
            inv = (G.B / G.A, G.C / G.A, G.F**2 / G.A)
            inv_r = (Gr.B / Gr.A, Gr.C / Gr.A, Gr.F**2 / Gr.A)
            worst_i = max(worst_i, max(abs(x - y) / abs(y) for x, y in zip(inv, inv_r)))
        rows.append({"r": label, "n": 80, "s_max": 6, "field_rel": sci(worst_f), "invariant_rel": sci(worst_i),
                     "verdict": "PASS" if max(worst_f, worst_i) <= TOL["A-5g"] else "FAIL"})
    return rows


def a9():
    rows = []
    base = {"a": 1.0, "t": 0.05, "E": 2.0e5, "nu": 0.3, "q": 1.0}
    nu2 = 0.1
    variants = {"base": {}, "q x 3.7": {"q": 3.7}, "E x 0.013": {"E": 2.0e5 * 0.013}, "t x 1.9": {"t": 0.05 * 1.9},
                "nu 0.3 -> 0.1, same D": {"nu": nu2, "E": 2.0e5 * (1 - nu2**2) / (1 - 0.3**2)}, "a, b x 2.9": {"a": 2.9}}
    for label, r, _ in step0.RATIOS:
        grids = [("I", (320, 320))] + ([] if label == "1" else [("II", A9_GRIDS[label])])
        for family, (nx, ny) in grids:
            fields = {}
            for name, change in variants.items():
                p = dict(base, **change)
                a = p["a"]
                res = run_rectangle_physical(a, a * float(r), p["t"], p["E"], p["nu"], p["q"], nx, ny)
                fields[name] = (res.dimensionless_field(), res.status, res.run.s if res.run else 0)
            w0 = fields["base"][0]
            for name, (w, status, s) in fields.items():
                if name == "base":
                    continue
                dev = float(np.max(np.abs(w - w0)) / np.max(np.abs(w0)))
                rows.append({"r": label, "family": family, "nx": nx, "ny": ny, "variant": name, "status": status, "s": s,
                             "s_base": fields["base"][2], "rel_dev": sci(dev),
                             "verdict": "PASS" if dev <= TOL["A-9"] else "FAIL"})
    return rows


def a11(levels):
    idx = {(r["r"], r["family"], r["nx"], r["ny"]): r for r in levels}
    rows = []
    for (label, family), chain in CHAINS.items():
        b = float(RATIO[label])
        for i, g in enumerate(chain):
            if not step0.qualification(g):
                continue
            pair = (g, chain[i + 1]) if i + 1 < len(chain) else (chain[i - 1], g)
            wL, w2L = run(1.0, b, *pair[0]).last_field(), run(1.0, b, *pair[1]).last_field()
            e_disc = float(np.max(np.abs(wL - w2L[::2, ::2])) / np.max(np.abs(w2L)))
            s_stop = idx[(label, family, *g)]["A11_S_stop"]
            rows.append({"r": label, "family": family, "nx": g[0], "ny": g[1], "pair": f"{pair[0]}->{pair[1]}",
                         "S_stop": sci(s_stop), "E_disc": sci(e_disc), "ratio": sci(s_stop / e_disc),
                         "status_strict": idx[(label, family, *g)]["A11_status_strict"],
                         "verdict": "PASS" if s_stop <= TOL["A-11"] * e_disc else "FAIL"})
    return rows


def a12_nodes():
    rows = []
    for label, r, _ in step0.RATIOS:
        b = float(r)
        for nx, ny in ((20, 20), (21, 21), (20, 21), (21, 20)):
            res = run_rectangle_physical(1.0, b, 0.05, 2.0e5, 0.3, 1.0, nx, ny, s_min=3, s_max=3)
            rep = halfstep_report(res, control_points=(("1/2", "1/2"),) + QUARTERS)
            ok = True
            for h in rep.halfsteps:
                w = res.run.history.field(h.s) * res.D / (res.input.q * res.input.a**4)
                for (fx, fy), value in h.control_values:
                    is_node = (Fraction(fx) * nx).denominator == 1 and (Fraction(fy) * ny).denominator == 1
                    ok = ok and ((value == node(w, fx, fy)) if is_node else value == NOT_A_NODE_MESSAGE)
                ok = ok and not (np.any(w[0, :]) or np.any(w[-1, :]) or np.any(w[:, 0]) or np.any(w[:, -1]))
            rows.append({"r": label, "nx": nx, "ny": ny, "status": res.status, "verdict": "PASS" if ok else "FAIL"})
    return rows


# --- unequal grids (D-032, item 10): observation ------------------------------------------------------------------

def unequal():
    rows = []
    for label, r, _ in step0.RATIOS:
        b = float(r)
        top = TOP_II[label]
        for axis in ("x", "y"):
            prev = None
            for n in (20, 40, 80, 160, 320, 640):
                grid = (n, top[1]) if axis == "x" else (top[0], n)
                res = run(1.0, b, *grid)
                errs = [a5_errors(res, label, grid, s)["E_inf"] for s in (1, 2, 3)]
                st = {c.s: c.status for c in res.energy_checks}
                diag = [x.solver_diagnostics for x in res.history.records]
                row = {"r": label, "refined_axis": axis, "nx": grid[0], "ny": grid[1], "status": res.status,
                       **{f"E_inf_s{s}": sci(e) for s, e in zip((1, 2, 3), errs)},
                       **{f"p_s{s}": "" if prev is None else sci(math.log2(prev[s - 1] / errs[s - 1]), 4) for s in (1, 2, 3)},
                       "A7_s2": st.get(2), "A7_s3": st.get(3),
                       "A6_ok": max(d.eta for d in diag) <= TOL["A-6"] and max(d.imag_ratio for d in diag) <= TOL["A-6"]}
                rows.append(row)
                prev = errs
    return rows


# --- group B (no PASS/FAIL) ---------------------------------------------------------------------------------------

def navier(label, grid):
    key = (label, grid)
    if key not in _navier:
        nav = NavierRectangle(a=1.0, b=float(RATIO[label]), q=1.0, D=1.0, nu=0.3)
        _navier[key] = nav.at_nodes(*grid, diagnostics=False).w_coef
    return _navier[key]


def trapezoid_norm(v, hx, hy):
    wx = np.full(v.shape[0], hx)
    wx[[0, -1]] *= 0.5
    wy = np.full(v.shape[1], hy)
    wy[[0, -1]] *= 0.5
    return math.sqrt(float(np.sum(np.outer(wx, wy) * v * v)))


def group_b():
    levels, starts = [], []
    for (label, family), chain in CHAINS.items():
        b = float(RATIO[label])
        nav_plate = NavierRectangle(a=1.0, b=b, q=1.0, D=1.0, nu=0.3)
        pi0 = nav_plate.deflection_integral().Pi_M
        P = parseval(label)
        for grid in chain:
            nx, ny = grid
            hx, hy = 1.0 / nx, b / ny
            res = run(1.0, b, nx, ny)
            wn = navier(label, grid)
            swapped = run(b, 1.0, ny, nx)  # second start: plate b x a started along its x = y of a x b (ChV-R1)
            s_stop = len(res.history)
            for s in (1, 2, 3, 4, "stop"):
                ss = s_stop if s == "stop" else s
                w = field(res, ss)
                row = {"r": label, "family": family, "nx": nx, "ny": ny, "s": s, "s_value": ss,
                       "delta_c": sci((centre(w) - centre(wn)) / centre(wn)),
                       "E_inf_N": sci(float(np.max(np.abs(w - wn)) / np.max(np.abs(wn)))),
                       "E_2_N": sci(trapezoid_norm(w - wn, hx, hy) / trapezoid_norm(wn, hx, hy)),
                       **{f"q_{fx}_{fy}".replace("/", ""): sci((node(w, fx, fy) - node(wn, fx, fy)) / node(wn, fx, fy))
                          for fx, fy in QUARTERS},
                       "e_h": sci((step0.energy(w, hx, hy) - pi0) / abs(pi0))}
                if ss <= 4:
                    wsa = P.field_on_grid(ss, nx, ny)
                    row.update({"delta_c_SA": sci((centre(wsa) - centre(wn)) / centre(wn)),
                                "E_inf_SA_N": sci(float(np.max(np.abs(wsa - wn)) / np.max(np.abs(wn)))),
                                "split_h_SA_centre": sci((centre(w) - centre(wsa)) / centre(wn)),
                                "e_SA": sci((P.energy(ss).energy - pi0) / abs(pi0))})
                levels.append(row)
                ss_ba = len(swapped.history) if s == "stop" else ss  # each plate at its own stop
                if ss_ba <= len(swapped.history):
                    w2 = field(swapped, ss_ba).T  # back to the nodes of a x b
                    starts.append({"r": label, "family": family, "nx": nx, "ny": ny, "s": s, "s_value": ss,
                                   "s_value_ba": ss_ba,
                                   "s_stop_ab": s_stop, "s_stop_ba": len(swapped.history),
                                   "status_ba": swapped.status,
                                   "field_diff": sci(float(np.max(np.abs(w - w2)) / np.max(np.abs(w)))),
                                   "centre_ab": sci(centre(w), 12), "centre_ba": sci(centre(w2), 12),
                                   "energy_ab": sci(step0.energy(w, hx, hy), 12), "energy_ba": sci(step0.energy(w2, hx, hy), 12),
                                   "A7_s2_ba": next((c.status for c in swapped.energy_checks if c.s == 2), ""),
                                   "A6_ba_ok": all(x.solver_diagnostics.eta <= TOL["A-6"] and x.solver_diagnostics.imag_ratio
                                                   <= TOL["A-6"] for x in swapped.history.records)})
    return levels, starts


# --- driver ------------------------------------------------------------------------------------------------------

def write_csv(path, rows):
    fields = []
    for r in rows:
        fields += [k for k in r if k not in fields]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: (sci(v) if isinstance(v, float) else v) for k, v in r.items()})


def _git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def _version(p):
    try:
        return metadata.version(p)
    except metadata.PackageNotFoundError:
        return "unknown"


def main(out_dir=OUT, report=REPORT):
    commit, dirty = _git("rev-parse", "HEAD"), bool(_git("status", "--porcelain"))
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tables = {}
    tables["a1_a2"] = one_dimensional()
    tables["a3"] = a3()
    tables["a4v"] = [a4v()]
    levels = [level_checks(label, family, g) for (label, family), chain in CHAINS.items() for g in chain]
    tables["levels"] = levels
    print("levels", round(time.time() - t0), flush=True)
    a4l, a5l = accuracy_levels()
    tables["a4_levels"], tables["a5_levels"] = a4l, a5l
    tables["pairs"] = pair_rows(a4l, a5l)
    tables["monotone"] = monotone_rows(a4l, a5l)
    print("accuracy", round(time.time() - t0), flush=True)
    tables["a5g"] = a5g()
    tables["a9"] = a9()
    tables["a11"] = a11(levels)
    tables["a12_nodes"] = a12_nodes()
    print("a5g-a12", round(time.time() - t0), flush=True)
    tables["unequal"] = unequal()
    print("unequal", round(time.time() - t0), flush=True)
    tables["groupB_levels"], tables["groupB_starts"] = group_b()
    print("group B", round(time.time() - t0), flush=True)
    for name, rows in tables.items():
        write_csv(out_dir / f"{name}.csv", rows)
    meta = {"study": "R3 solver qualification, steps 1-2 of R3_NUMERICAL_SPEC 12 and group B", "commit": commit,
            "dirty": dirty, "date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "python": platform.python_version(), "platform": platform.platform(),
            "versions": {p: _version(p) for p in ("numpy", "scipy", "kkvim")},
            "plate": "a = 1, b = float(r), q = D = 1 (A-9, A-12: physical input)", "stopping": "defaults; A-11 strict_parameters",
            "tolerances": TOL, "p_window": P_WINDOW, "p_window_1d": P_WINDOW_1D, "richardson_ratio": RE_RATIO, "rv3": RV3,
            "pairs_table_7_2": {f"{k[0]}|{k[1]}": v for k, v in PAIRS.items()},
            "monotone_table_7_3": {f"{k[0]}|{k[1]}": v for k, v in MONOTONE.items()},
            "a9_grids": A9_GRIDS, "one_axis_top": TOP_II, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "seconds": round(time.time() - t0), "rows": {k: len(v) for k, v in tables.items()}}
    meta["summary"] = summary(tables)
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    write_csv(out_dir / "summary.csv", meta["summary"])
    if report is not None:
        step0.inject_report(report, report_blocks(out_dir))
    print(json.dumps(meta["summary"], ensure_ascii=False, indent=0))


def _status(v):
    v = str(v)
    for key in ("FAIL", "UNRESOLVED", "NOT APPLIED", "DIAGNOSTIC", "PASS"):
        if v.startswith(key):
            return key
    return v


def summary(tables):
    out = []

    def count(check, verdicts):
        c = {}
        for v in verdicts:
            k = _status(v)
            c[k] = c.get(k, 0) + 1
        out.append({"check": check, **{k: c.get(k, 0) for k in ("PASS", "FAIL", "UNRESOLVED", "NOT APPLIED", "DIAGNOSTIC")},
                    "limited_scope": sum("ограниченный" in str(v) for v in verdicts)})

    one = tables["a1_a2"]
    for check in ("A-1a", "A-1b", "A-2a", "A-2a monotone", "A-2b", "A-2v"):
        count(check, [r["verdict"] for r in one if r["check"] == check])
    count("A-3", [r["verdict"] for r in tables["a3"]])
    count("A-4v", [r["verdict"] for r in tables["a4v"]])
    pairs = tables["pairs"]
    for check in ("A-4a p", "A-4b Richardson", "A-5a p", "A-5b Richardson", "A-5v p"):
        count(check, [r["verdict"] for r in pairs if r["check"] == check and r["mandatory"] is True])
        count(check + " (diag.)", [r["verdict"] for r in pairs if r["check"] == check and r["mandatory"] is not True])
    mono = tables["monotone"]
    count("A-4a/A-5a monotone", [r["verdict"] for r in mono if not r["check"].endswith("(diag.)")])
    count("A-5d counters", ["PASS" if r["A5d_counters"] else "FAIL" for r in tables["a5_levels"]])
    count("A-5g", [r["verdict"] for r in tables["a5g"]])
    lv = [r for r in tables["levels"] if r["qualification"]]
    count("A-6", ["PASS" if r["A6_eta_max"] <= TOL["A-6"] and r["A6_imag_max"] <= TOL["A-6"] else "FAIL" for r in lv])
    count("A-8", ["PASS" if r["A8_reflection_max"] <= TOL["A-8"] else "FAIL" for r in lv])
    count("A-9", [r["verdict"] for r in tables["a9"]])
    count("A-11", [r["verdict"] for r in tables["a11"]])
    count("A-12 boundary", ["PASS" if r["A12_boundary_zero"] else "FAIL" for r in lv])
    count("A-12 nodes", [r["verdict"] for r in tables["a12_nodes"]])
    count("controller (11.1, 11.3)", ["PASS" if r["controller_match"] and r["eps_rel_dev_max"] <= 1e-6 else "FAIL"
                                      for r in tables["levels"]])
    count("report 11.5", ["PASS" if r["report_before_norm_ok"] and r["report_control_points_ok"] else "FAIL"
                          for r in tables["levels"]])
    return out


# --- report blocks from the CSV ----------------------------------------------------------------------------------

def read(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def g3(v):
    return "—" if v in ("", None) else f"{float(v):.3g}"


def md(rows, cols, names=None):
    names = names or cols
    out = ["| " + " | ".join(names) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(str(r[c]).replace("|", "\\|") for c in cols) + " |")
    return "\n".join(out)


def report_blocks(out_dir=OUT):
    b = {}
    summ = read(out_dir / "summary.csv")
    b["qual_summary"] = md(summ, ["check", "PASS", "FAIL", "UNRESOLVED", "NOT APPLIED", "DIAGNOSTIC", "limited_scope"],
                           ["проверка", "PASS", "FAIL", "UNRESOLVED", "не применён", "диагностика", "огр. объём"])
    pairs = read(out_dir / "pairs.csv")
    man = [r for r in pairs if r["mandatory"] == "True"]
    for check, key in (("A-4a p", "qual_a4p"), ("A-5a p", "qual_a5p"), ("A-5v p", "qual_a5v")):
        sel = [r for r in man if r["check"] == check]
        b[key] = md(sel, ["r", "family", "pair", "quantity", "E_L", "E_2L", "p", "rv3_ratio", "verdict"],
                    ["r", "сем.", "пара", "величина", "E(L)", "E(2L)", "p", "бюджет/min E", "вердикт"])
    rich = [r for r in man if "Richardson" in r["check"]]
    b["qual_richardson"] = md(rich, ["check", "r", "family", "pair", "quantity", "p", "ratio", "rv3_ratio", "verdict"],
                              ["проверка", "r", "сем.", "пара", "величина", "p", "\\|w_RE−w_ex\\|/\\|w_2L−w_ex\\|",
                               "бюджет/\\|w_2L−w_ex\\|", "вердикт"])
    bad = [r for r in pairs + read(out_dir / "monotone.csv") if _status(r["verdict"]) in ("FAIL", "UNRESOLVED", "NOT APPLIED")]
    b["qual_problems"] = md(bad, ["check", "r", "family", "verdict"], ["проверка", "r", "сем.", "вердикт"]) if bad else \
        "Нет FAIL, UNRESOLVED и неприменённых сравнений в обязательных и диагностических записях."
    lv = read(out_dir / "levels.csv")
    agg = []
    for (label, family) in CHAINS:
        sel = [r for r in lv if r["r"] == label and r["family"] == family and r["qualification"] == "True"]
        agg.append({"r": label, "family": family, "grids": len(sel), "eta": g3(max(float(r["A6_eta_max"]) for r in sel)),
                    "im": g3(max(float(r["A6_imag_max"]) for r in sel)),
                    "refl": g3(max(float(r["A8_reflection_max"]) for r in sel)),
                    "zero": all(r["A12_boundary_zero"] == "True" for r in sel),
                    "ctrl": all(r["controller_match"] == "True" for r in sel),
                    "rep": all(r["report_before_norm_ok"] == r["report_control_points_ok"] == "True" for r in sel)})
    b["qual_levels"] = md(agg, ["r", "family", "grids", "eta", "im", "refl", "zero", "ctrl", "rep"],
                          ["r", "сем.", "квал. сеток", "max η (А-6)", "max Im (А-6)", "max отражение (А-8)",
                           "края = 0 (А-12)", "контроллер = QA", "отчёт 11.5"])
    a11 = read(out_dir / "a11.csv")
    worst = sorted(a11, key=lambda r: -float(r["ratio"]))[:8]
    b["qual_a11"] = md(worst, ["r", "family", "nx", "ny", "pair", "S_stop", "E_disc", "ratio", "verdict"],
                       ["r", "сем.", "n_x", "n_y", "пара E_disc", "S_stop", "E_disc", "S_stop/E_disc", "вердикт"])
    a9 = read(out_dir / "a9.csv")
    b["qual_a9"] = f"Наибольшее отклонение w̄: {g3(max(float(r['rel_dev']) for r in a9))} (допуск 10⁻¹¹), строк {len(a9)}."
    one = read(out_dir / "a1_a2.csv")
    rv1 = [r for r in one if r["check"] == "A-1a" and r["case"].startswith("RV-1")]
    b["qual_a1"] = md(rv1, ["case", "ell", "n", "value", "verdict"], ["набор", "ℓ", "n", "max\\|f_h − ρ_h f\\|/max\\|f\\|", "вердикт"])
    a3 = read(out_dir / "a3.csv")
    b["qual_a3"] = f"n_x: {len(a3) // 2} значений (4, 7, 20, 320 и все n_x цепочек до 640), a = 1 и 2,9; " \
                   f"наибольшее отн. отклонение {g3(max(float(r['max_rel_dev']) for r in a3))} (допуск 10⁻¹³)."
    a10 = out_dir.parent / "a10b"
    if (a10 / "metadata.json").exists():
        m10 = json.loads((a10 / "metadata.json").read_text(encoding="utf-8"))
        hp = read(a10 / "a10b_hp.csv")
        bud = read(a10 / "a10b_budgets.csv")
        chk = read(a10 / "a10b_method_check.csv")
        b["a10b_summary"] = (
            f"А-10б, s = 1 против В-29 в {m10['dps']} знаках (mpmath {m10['mpmath']}, коммит `{m10['commit'][:7]}`, dirty = "
            f"{m10['dirty']}): {len(hp)} квалификационных сеток А-4 (r ≠ 1, оба семейства), все узлы; наибольшее "
            f"max\\|w_Парсеваль − w_В-29\\|/max\\|w¹\\| = {g3(max(float(r['max_rel_dev']) for r in hp))} (допуск 10⁻¹³); "
            f"PASS {sum(r['verdict'] == 'PASS' for r in hp)} из {len(hp)}. Проверка метода разности против полного произведения "
            f"в {m10['dps']} знаках: совпадение {sum(r['agree'] == 'True' for r in chk)} из {len(chk)}.\n\n"
            + md(bud, ["r", "s", "M", "max_tail_rel", "field_error", "invariant_error", "verdict_tail", "verdict_transfer"],
                 ["r", "s", "M", "хвосты И-12 (max отн.)", "field_error", "invariant_error", "хвосты ≤ 10⁻¹⁴", "перенос ≤ 10⁻¹²"]))
    a4 = read(out_dir / "a4_levels.csv")
    lm = [float(r["levy_M1_vs_parseval"]) for r in a4 if r["levy_M1_vs_parseval"]]
    b["qual_a4_ref"] = f"Перекрёстная проверка эталона А-4 при r ≥ 1 (r ≠ 1): max\\|levy_series_w(M = 1) − Parseval s = 1\\|/max\\|w¹\\| = " \
                       f"{g3(max(lm))} по {len(lm)} сеткам (диагностика эталона, не критерий решателя)."
    un = read(out_dir / "unequal.csv")
    compact = []
    for label, _, _ in step0.RATIOS:
        for axis in ("x", "y"):
            sel = [r for r in un if r["r"] == label and r["refined_axis"] == axis]
            fixed = sel[0]["ny"] if axis == "x" else sel[0]["nx"]
            compact.append({"r": label, "axis": axis, "fixed": fixed,
                            "p1": " ".join(f"{float(r['p_s1']):.2f}" for r in sel if r["p_s1"]),
                            "p3": " ".join(f"{float(r['p_s3']):.2f}" for r in sel if r["p_s3"]),
                            "e3": " ".join(f"{float(r['E_inf_s3']):.2g}" for r in sel),
                            "a7": " ".join(sorted({f"{r['A7_s2']}/{r['A7_s3']}" for r in sel})),
                            "a6": all(r["A6_ok"] == "True" for r in sel)})
    b["qual_unequal"] = md(compact, ["r", "axis", "fixed", "p1", "p3", "e3", "a7", "a6"],
                           ["r", "сгущаемая ось", "вторая компонента", "p, s = 1 (20→…→640)", "p, s = 3", "E∞, s = 3 (20…640)",
                            "А-7 s = 2/3", "А-6"])
    gb = read(out_dir / "groupB_levels.csv")
    fine = []
    for (label, family), chain in CHAINS.items():
        g = chain[-1]
        r = next(x for x in gb if (x["r"], x["family"], int(x["nx"]), int(x["ny"]), x["s"]) == (label, family, *g, "stop"))
        fine.append(r)
    b["groupb_stop"] = md(fine, ["r", "family", "nx", "ny", "s_value", "delta_c", "E_inf_N", "E_2_N", "q_14_14", "q_34_14", "q_14_34", "e_h"],
                          ["r", "сем.", "n_x", "n_y", "s", "δ_c", "E∞^N", "E₂^N", "(¼,¼)", "(¾,¼)", "(¼,¾)", "e^(h)"])
    sa = [x for x in gb if x["s"] in ("1", "2", "3", "4") and (x["r"], x["family"]) in CHAINS
          and (int(x["nx"]), int(x["ny"])) == CHAINS[(x["r"], x["family"])][-1] and x["family"] == "I"]
    b["groupb_sa"] = md(sa, ["r", "s", "delta_c_SA", "E_inf_SA_N", "split_h_SA_centre", "e_SA", "delta_c"],
                        ["r", "s", "δ_c^SA", "E∞(SA−N)", "(w_h−w_SA)/w_N центр", "e_s (непр.)", "δ_c (сетка)"])
    st = read(out_dir / "groupB_starts.csv")
    sel = [x for x in st if x["family"] == "I" and int(x["nx"]) == 160]
    b["groupb_starts"] = md(sel, ["r", "s", "s_value", "s_value_ba", "field_diff", "centre_ab", "centre_ba", "A7_s2_ba"],
                            ["r", "s", "s a×b", "s b×a", "max\\|w_ab − w_ba\\|/max\\|w\\|", "центр a×b", "центр b×a",
                             "А-7 s=2 b×a"])
    return b


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    if args.report_only:
        step0.inject_report(REPORT, report_blocks(OUT))
    else:
        main()
