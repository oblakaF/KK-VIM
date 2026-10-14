"""QA, R3 solver qualification, step 0 (docs/R3_NUMERICAL_SPEC.md 12, D-039): energy measurements before any
comparison with the references.

For the ten control ratios r = b/a and every level of the selected chains of families I and II (6.3; qualification
grids min(n_x, n_y) >= 20, max <= 640, and the coarser chain levels as diagnostics) the solver kkvim.rectangle_run
runs on the plate 1 x r, q = D = 1 (the plate of the R-7 surrogate, docs/results/r3/r7/margins.csv), with the default
stopping parameters. For every half-step s >= 2 QA computes INDEPENDENTLY of kkvim.energy, from the stored factors
X, Y, the computed factor before normalization and the grids only (R3_NUMERICAL_SPEC 8.1 = NUMERICAL_SPEC 10.2-10.4):

    P_s = Pi_h(w(s)) = a_h(w, w)/2 - l_h(w), two-dimensional assembly of Delta_h = L_x + L_y;
    Q_s = a_h(d, d)/2, d = w(s-1) - w(s);  Delta P_s = P_{s-1} - P_s;
    gamma_s = h rho^T d_f, rho = F_h 1 - K Y_raw, d_f = f_{s-2} - Y_raw on the recomputed axis, K and the coefficients
              A_h, B_h, C_h, F_h recomputed by QA from the fixed factor (3.3), h the step of the solved grid;
    R(w) = u (ceil(log2 N) + 16) S_hat(w), N = (n_x - 1)(n_y - 1);  R_f = 4u (max(n_x, n_y)/pi)^2 |w(s)|_a |d|_a;
    tau_s = |gamma_s| + 10 [R(w(s-1)) + R(w(s)) + R_f];  tau_R = tau_s - |gamma_s|;
    E1: Delta P_s >= -tau_s;  E2: |Delta P_s - Q_s| <= tau_s;  resolved: Q_s >= 10 tau_s;
    m_s = Q_s/(10 tau_s);  m_up = Q_s/(10 tau_R);  g_allow = tau_R (m_up - 1).

Mandatory region (8.2, 8.3; D-039; D-032 item 8 at r = 1): s = 2 on every qualification grid; s = 3 on the levels of
table 8.3. A mandatory row passes if the test is resolved, E1 and E2 hold, |gamma_s| <= g_allow and the surrogate
model holds, |Q_s - Q_sur|/Q_sur <= e_sur (R3_R7_ENERGY_SOLVABILITY_DRAFT 7, item 7). Outside it the status is
PASS/FAIL if resolved, otherwise UNRESOLVED (not FAIL). Nothing is tuned: grids, criteria and the surrogate data are
those fixed before this run. The solver's own EnergyCheck values are compared with the QA values afterwards only.

r = 1 regression (12, item 5): the R3 path on a x a against the R2 path kkvim.controller.run_until_stopped, bitwise,
on n = 20..640; m_2, m_3 against the published discrete R2 values (docs/results/r3/r7/calibration_r1.csv).

    uv run python scripts/r3_solver_step0.py            # writes docs/results/r3/solver/step0/ and the report blocks
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
from fractions import Fraction
from importlib import metadata
from pathlib import Path

import numpy as np

from kkvim.controller import run_until_stopped
from kkvim.rectangle_run import run_rectangle_until_stopped

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "results" / "r3" / "solver" / "step0"
MARGINS = ROOT / "docs" / "results" / "r3" / "r7" / "margins.csv"
CALIBRATION = ROOT / "docs" / "results" / "r3" / "r7" / "calibration_r1.csv"
REPORT = ROOT / "docs" / "R3_SOLVER_QUALIFICATION.qa-report.md"
U = 2.0**-53
A, Q, D = 1.0, 1.0, 1.0  # plate 1 x r, q = D = 1 (the plate of the R-7 surrogate)

# label, r = b/a exactly, label in margins.csv
RATIOS = [("1/5", Fraction(1, 5), "0,2"), ("1/3", Fraction(1, 3), "1/3"), ("1/2", Fraction(1, 2), "1/2"),
          ("2/3", Fraction(2, 3), "2/3"), ("1", Fraction(1), "1"), ("7/5", Fraction(7, 5), "1,4"),
          ("3/2", Fraction(3, 2), "1,5"), ("2", Fraction(2), "2"), ("3", Fraction(3), "3"), ("5", Fraction(5), "5")]

# selected nested chains, R3_NUMERICAL_SPEC 6.3 (family I for every r; family II: chain k0, 2 k0, ...; r = 1: I = II)
CHAIN_I = tuple((n, n) for n in (20, 40, 80, 160, 320, 640))
CHAIN_II = {
    "1/5": ((20, 4), (40, 8), (80, 16), (160, 32), (320, 64), (640, 128)),
    "1/3": ((36, 12), (72, 24), (144, 48), (288, 96), (576, 192)),
    "1/2": ((40, 20), (80, 40), (160, 80), (320, 160), (640, 320)),
    "2/3": ((60, 40), (120, 80), (240, 160), (480, 320)),
    "7/5": ((20, 28), (40, 56), (80, 112), (160, 224), (320, 448)),
    "3/2": ((40, 60), (80, 120), (160, 240), (320, 480)),
    "2": ((20, 40), (40, 80), (80, 160), (160, 320), (320, 640)),
    "3": ((12, 36), (24, 72), (48, 144), (96, 288), (192, 576)),
    "5": ((4, 20), (8, 40), (16, 80), (32, 160), (64, 320), (128, 640)),
}


def _n(*levels):
    return tuple((n, n) for n in levels)


# s = 3 mandatory levels, table 8.3 ("обязательно при s = 3"); r = 1: n <= 160 (D-032, item 8)
MANDATORY_S3 = {
    ("1/5", "I"): _n(20, 40, 80, 160, 320, 640), ("1/5", "II"): ((160, 32), (320, 64), (640, 128)),
    ("1/3", "I"): _n(20, 40, 80, 160, 320, 640), ("1/3", "II"): ((72, 24), (144, 48), (288, 96), (576, 192)),
    ("1/2", "I"): _n(20, 40, 80, 160, 320), ("1/2", "II"): ((40, 20), (80, 40), (160, 80), (320, 160), (640, 320)),
    ("2/3", "I"): _n(20, 40, 80, 160), ("2/3", "II"): ((60, 40), (120, 80), (240, 160)),
    ("1", "I"): _n(20, 40, 80, 160),
    ("7/5", "I"): _n(20, 40, 80, 160), ("7/5", "II"): ((20, 28), (40, 56), (80, 112)),
    ("3/2", "I"): _n(20, 40, 80), ("3/2", "II"): ((40, 60), (80, 120)),
    ("2", "I"): _n(20, 40, 80), ("2", "II"): ((20, 40), (40, 80), (80, 160)),
    ("3", "I"): _n(20, 40, 80), ("3", "II"): ((24, 72), (48, 144), (96, 288)),
    ("5", "I"): _n(20, 40, 80), ("5", "II"): ((32, 160), (64, 320)),
}


def priority(label, family, grid):
    """Priority cases of 12, item 4: r = 1/5 and 5; r = 7/5, 3/2 at n = 160; r = 1/3 at n = 640."""
    if label in ("1/5", "5"):
        return True
    return (label in ("7/5", "3/2") and grid == (160, 160)) or (label == "1/3" and grid == (640, 640))


def chains():
    for label, r, _ in RATIOS:
        yield label, r, "I", CHAIN_I
        if label != "1":
            yield label, r, "II", CHAIN_II[label]


def qualification(grid):
    return min(grid) >= 20 and max(grid) <= 640 and grid[0] % 4 == 0 and grid[1] % 4 == 0


# --- independent energy quantities (QA formulas) -------------------------------------------------------------------

def full(v):
    out = np.zeros(len(v) + 2)
    out[1:-1] = v
    return out


def fsum(a):
    return math.fsum(np.ravel(a).tolist())


def lap(w, hx, hy):
    """(Delta_h w) at the interior nodes: L_x w + L_y w."""
    c = w[1:-1, 1:-1]
    return (w[:-2, 1:-1] - 2.0 * c + w[2:, 1:-1]) / hx**2 + (w[1:-1, :-2] - 2.0 * c + w[1:-1, 2:]) / hy**2


def abs_lap(w, hx, hy):
    a = np.abs(w)
    c = a[1:-1, 1:-1]
    return (a[:-2, 1:-1] + 2.0 * c + a[2:, 1:-1]) / hx**2 + (a[1:-1, :-2] + 2.0 * c + a[1:-1, 2:]) / hy**2


def a_form(w, hx, hy):
    return D * hx * hy * fsum(lap(w, hx, hy) ** 2)


def energy(w, hx, hy):
    return 0.5 * a_form(w, hx, hy) - Q * hx * hy * fsum(w[1:-1, 1:-1])


def rounding_R(w, hx, hy):
    n_nodes = (w.shape[0] - 2) * (w.shape[1] - 2)
    s_hat = D * hx * hy * fsum(np.abs(lap(w, hx, hy)) * abs_lap(w, hx, hy)) + abs(Q) * hx * hy * fsum(np.abs(w[1:-1, 1:-1]))
    return U * (math.ceil(math.log2(n_nodes)) + 16) * s_hat


def second_diff(f_full, h):
    return (f_full[:-2] - 2.0 * f_full[1:-1] + f_full[2:]) / h**2


def coefficients(f, h):
    """A_h = <f, f>, B_h = sum h (D+ f)^2 (midpoints), C_h = <L f, L f>, F_h = q <f, 1> (3.3)."""
    ff = full(f)
    return (h * fsum(f * f), h * fsum(((ff[1:] - ff[:-1]) / h) ** 2), h * fsum(second_diff(ff, h) ** 2), Q * h * fsum(f))


def apply_K(y, h, A_, B_, C_):
    """K y = D [A L^2 y - 2 B L y + C y], L with y_0 = y_n = 0 and (L y)_0 = (L y)_n = 0."""
    Ly = second_diff(full(y), h)
    return D * (A_ * second_diff(full(Ly), h) - 2.0 * B_ * Ly + C_ * y)


def field(rec):
    return np.outer(full(np.asarray(rec.x)), full(np.asarray(rec.y)))


def qa_energy(prev, cur, hx, hy, nx, ny):
    """Independent quantities of the energy test of half-step s = cur.s >= 2."""
    if cur.direction == "Y":  # Y solved on G(b, n_y); fixed X = normalized computed factor of s - 1 on G(a, n_x)
        fixed, h_fixed, h_solve, f_prev = prev.x, hx, hy, prev.y
    else:
        fixed, h_fixed, h_solve, f_prev = prev.y, hy, hx, prev.x
    A_, B_, C_, F_ = coefficients(np.asarray(fixed), h_fixed)
    y_raw = np.asarray(cur.raw)
    rho = F_ - apply_K(y_raw, h_solve, A_, B_, C_)
    gamma = h_solve * fsum(rho * (np.asarray(f_prev) - y_raw))
    w_prev, w_now = field(prev), field(cur)
    d = w_prev - w_now
    P_prev, P_now = energy(w_prev, hx, hy), energy(w_now, hx, hy)
    a_d, a_now = a_form(d, hx, hy), a_form(w_now, hx, hy)
    Qs = 0.5 * a_d
    R_prev, R_now = rounding_R(w_prev, hx, hy), rounding_R(w_now, hx, hy)
    R_f = 4.0 * U * (max(nx, ny) / math.pi) ** 2 * math.sqrt(a_now) * math.sqrt(a_d)
    tau_R = 10.0 * (R_prev + R_now + R_f)
    tau = abs(gamma) + tau_R
    dP = P_prev - P_now
    m_up = Qs / (10.0 * tau_R)
    return {
        "Q": Qs, "P_prev": P_prev, "P_now": P_now, "delta_P": dP, "gamma": gamma, "R_prev": R_prev, "R_now": R_now,
        "R_f": R_f, "tau_R": tau_R, "tau": tau, "m_s": Qs / (10.0 * tau), "m_up": m_up, "g_allow": tau_R * (m_up - 1.0),
        "e1": dP >= -tau, "e2": abs(dP - Qs) <= tau, "resolved": Qs >= 10.0 * tau,
        "coef_A": A_, "coef_B": B_, "coef_C": C_, "coef_F": F_,
    }


# --- surrogate data ---------------------------------------------------------------------------------------------------

def surrogate():
    table = {}
    with open(MARGINS, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            table[(row["r"], row["family"], int(row["nx"]), int(row["ny"]), int(row["s"]))] = row
    return table


# --- run --------------------------------------------------------------------------------------------------------------

def _git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sci(v, digits=10):
    return "" if v is None else f"{v:.{digits}e}"


ENERGY_FIELDS = ["r", "family", "nx", "ny", "qualification", "priority", "s", "k", "mandatory", "Q", "delta_P", "gamma",
                 "R_prev", "R_now", "R_f", "tau_R", "tau", "m_s", "m_up", "g_allow", "abs_gamma_le_g_allow", "e1", "e2",
                 "resolved", "status_qa", "Q_sur", "m_up_sur", "e_sur", "category_sur", "dev_Q_sur", "dev_over_e_sur",
                 "within_e_sur", "dm_over_m", "dm_over_m_n2", "status_solver", "status_match", "diff_Q_over_tau",
                 "rel_diff_tau", "diff_delta_P_over_tau", "abs_diff_gamma_over_tau", "rel_diff_coefficients", "verdict",
                 "reason"]
RUN_FIELDS = ["r", "family", "nx", "ny", "qualification", "status", "s_stop", "eta_max", "imag_max", "refined_steps"]


def classify(row, mandatory):
    reasons = []
    if not row["resolved"]:
        reasons.append("not resolved (m_s < 1)")
    elif not (row["e1"] and row["e2"]):
        reasons.append("FAIL " + ("E1 " if not row["e1"] else "") + ("E2" if not row["e2"] else ""))
    if not row["abs_gamma_le_g_allow"]:
        reasons.append("|gamma| > g_allow")
    if row["within_e_sur"] is False:
        reasons.append("|Q - Q_sur|/Q_sur > e_sur")
    if mandatory:
        return ("PASS" if not reasons else "BLOCKER"), "; ".join(reasons)
    # outside the mandatory region: PASS/FAIL if resolved, otherwise UNRESOLVED; the model is only recorded there
    if not row["resolved"]:
        return "UNRESOLVED", "not resolved outside the mandatory region (8.2, item 4)"
    return ("PASS" if row["e1"] and row["e2"] else "FAIL"), "; ".join(r for r in reasons if r.startswith("FAIL"))


def run(out_dir=OUT, report=REPORT):
    source_commit = _git("rev-parse", "HEAD")  # captured before any output is written
    source_dirty = bool(_git("status", "--porcelain"))
    out_dir.mkdir(parents=True, exist_ok=True)
    sur = surrogate()
    energy_rows, run_rows = [], []
    for label, r, family, chain in chains():
        b = float(r)
        for nx, ny in chain:
            grid = (nx, ny)
            hx, hy = A / nx, b / ny  # the steps of Grid1D(a, n_x), Grid1D(b, n_y)
            res = run_rectangle_until_stopped(A, b, D, Q, nx, ny)
            recs = res.history.records
            diag = [rec.solver_diagnostics for rec in recs]
            run_rows.append({"r": label, "family": family, "nx": nx, "ny": ny, "qualification": qualification(grid),
                             "status": res.status, "s_stop": len(recs), "eta_max": sci(max(d.eta for d in diag)),
                             "imag_max": sci(max(d.imag_ratio for d in diag)),
                             "refined_steps": sum(bool(d.refined) for d in diag)})
            checks = {c.s: c for c in res.energy_checks}
            for prev, cur in zip(recs, recs[1:]):
                s = cur.s
                e = qa_energy(prev, cur, hx, hy, nx, ny)
                mandatory = qualification(grid) and (s == 2 or (s == 3 and grid in MANDATORY_S3.get((label, family), ())))
                e["abs_gamma_le_g_allow"] = abs(e["gamma"]) <= e["g_allow"]
                e["status_qa"] = "UNRESOLVED" if not e["resolved"] else ("PASS" if e["e1"] and e["e2"] else "FAIL")
                srow = sur.get((dict((x[0], x[2]) for x in RATIOS)[label], family, nx, ny, s))
                if srow is not None:
                    Q_sur, e_sur, m_sur = float(srow["Q_sur"]), float(srow["e_sur"]), float(srow["m_up"])
                    dev = abs(e["Q"] - Q_sur) / Q_sur
                    e.update(Q_sur=Q_sur, e_sur=e_sur, m_up_sur=m_sur, category_sur=srow["category"], dev_Q_sur=dev,
                             dev_over_e_sur=dev / e_sur, within_e_sur=dev <= e_sur,
                             dm_over_m=(e["m_up"] - m_sur) / m_sur,
                             dm_over_m_n2=abs(e["m_up"] - m_sur) / m_sur * max(nx, ny) ** 2)
                else:
                    e.update(Q_sur=None, e_sur=None, m_up_sur=None, category_sur="", dev_Q_sur=None,
                             dev_over_e_sur=None, within_e_sur=None, dm_over_m=None, dm_over_m_n2=None)
                verdict, reason = classify(e, mandatory)
                c = checks.get(s)
                cf = cur.coefficients
                rel_coef = max(abs(x - y) / abs(y) for x, y in zip((e["coef_A"], e["coef_B"], e["coef_C"], e["coef_F"]),
                                                                    (cf.A, cf.B, cf.C, cf.F)))
                energy_rows.append({
                    "r": label, "family": family, "nx": nx, "ny": ny, "qualification": qualification(grid),
                    "priority": priority(label, family, grid), "s": s, "k": cur.k, "mandatory": mandatory,
                    **{k: sci(e[k]) for k in ("Q", "delta_P", "gamma", "R_prev", "R_now", "R_f", "tau_R", "tau",
                                              "m_s", "m_up", "g_allow")},
                    "abs_gamma_le_g_allow": e["abs_gamma_le_g_allow"], "e1": e["e1"], "e2": e["e2"],
                    "resolved": e["resolved"], "status_qa": e["status_qa"],
                    **{k: sci(e[k]) for k in ("Q_sur", "m_up_sur", "e_sur", "dev_Q_sur", "dev_over_e_sur", "dm_over_m",
                                              "dm_over_m_n2")},
                    "category_sur": e["category_sur"],
                    "within_e_sur": "" if e["within_e_sur"] is None else e["within_e_sur"],
                    "status_solver": None if c is None else c.status,
                    "status_match": c is not None and c.status == e["status_qa"],
                    "diff_Q_over_tau": sci(abs(c.Q - e["Q"]) / e["tau"], 3) if c else "",
                    "rel_diff_tau": sci(abs(c.tau - e["tau"]) / e["tau"], 3) if c else "",
                    "diff_delta_P_over_tau": sci(abs(c.delta_P - e["delta_P"]) / e["tau"], 3) if c else "",
                    "abs_diff_gamma_over_tau": sci(abs(c.gamma - e["gamma"]) / e["tau"], 3) if c else "",
                    "rel_diff_coefficients": sci(rel_coef, 3), "verdict": verdict, "reason": reason,
                })
            print(f"{label:>4} {family:>2} ({nx},{ny}) {res.status} s={len(recs)}", flush=True)

    r1 = r1_regression()
    write_csv(out_dir / "step0_energy.csv", ENERGY_FIELDS, energy_rows)
    write_csv(out_dir / "step0_runs.csv", RUN_FIELDS, run_rows)
    write_csv(out_dir / "step0_r1_regression.csv", list(r1[0]), r1)
    mandatory = [r for r in energy_rows if r["mandatory"]]
    blockers = [r for r in energy_rows if r["verdict"] in ("BLOCKER", "FAIL")]
    meta = {
        "study": "R3 solver qualification, step 0 (R3_NUMERICAL_SPEC 12; D-039)",
        "commit": source_commit, "dirty": source_dirty,
        "date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(), "platform": platform.platform(),
        "versions": {p: _version(p) for p in ("numpy", "scipy", "kkvim")},
        "plate": {"a": A, "b": "float(r)", "q": Q, "D": D}, "stopping": "defaults (tol_inf 1e-10, tol_E 1e-8, s_min 3, s_max 100)",
        "ratios": [x[0] for x in RATIOS], "chain_I": CHAIN_I, "chain_II": CHAIN_II,
        "mandatory_s3": {f"{k[0]}|{k[1]}": v for k, v in MANDATORY_S3.items()},
        "surrogate_margins_sha256": sha256(MARGINS), "calibration_r1_sha256": sha256(CALIBRATION),
        "script_sha256": sha256(__file__),
        "n_runs": len(run_rows), "n_energy_rows": len(energy_rows), "n_mandatory_rows": len(mandatory),
        "n_mandatory_pass": sum(r["verdict"] == "PASS" for r in mandatory),
        "n_blockers": len(blockers), "n_status_mismatch": sum(not r["status_match"] for r in energy_rows),
        "r1_bitwise_all": all(row["bitwise_equal"] for row in r1),
        "verdict": "PASS" if not blockers and all(row["bitwise_equal"] for row in r1) else "BLOCKER",
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    if report is not None:
        inject_report(report, report_blocks(out_dir))
    print(json.dumps({k: meta[k] for k in ("n_runs", "n_energy_rows", "n_mandatory_rows", "n_mandatory_pass",
                                           "n_blockers", "n_status_mismatch", "r1_bitwise_all", "verdict")}))
    return meta


def _version(package):
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "unknown"


def _same(x, y):
    if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
        return np.array_equal(np.asarray(x), np.asarray(y))
    return x == y or (isinstance(x, float) and isinstance(y, float) and math.isnan(x) and math.isnan(y))


def r1_regression():
    """R3 path on a x a against the R2 path, bitwise, n = 20..640; m_2, m_3 against calibration_r1.csv."""
    published = {}
    with open(CALIBRATION, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            published[(int(row["n"]), int(row["s"]))] = float(row["margin_disc"])
    rows = []
    for n in (20, 40, 80, 160, 320, 640):
        r3 = run_rectangle_until_stopped(A, A, D, Q, n, n)
        r2 = run_until_stopped(A, D, Q, n, n)
        same = r3.status == r2.status and len(r3.history) == len(r2.history)
        for x, y in zip(r3.history.records, r2.history.records):
            for name in ("s", "k", "direction", "grid_x", "grid_y", "D", "q", "x", "y", "raw", "c", "coefficients",
                         "solver_diagnostics"):
                same = same and _same(getattr(x, name), getattr(y, name))
        same = same and r3.energy_checks == r2.energy_checks and r3.changes == r2.changes and r3.parameters == r2.parameters
        recs = r3.history.records
        hx = hy = A / n
        m = {s: qa_energy(recs[s - 2], recs[s - 1], hx, hy, n, n)["m_s"] for s in (2, 3)}
        rows.append({"n": n, "status_r3": r3.status, "status_r2": r2.status, "s_stop": len(recs), "bitwise_equal": same,
                     "m2_qa": sci(m[2]), "m2_r2_published": sci(published[(n, 2)]),
                     "m2_rel_diff": sci(abs(m[2] - published[(n, 2)]) / published[(n, 2)], 3),
                     "m3_qa": sci(m[3]), "m3_r2_published": sci(published[(n, 3)]),
                     "m3_rel_diff": sci(abs(m[3] - published[(n, 3)]) / published[(n, 3)], 3)})
    return rows


def write_csv(path, fields, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


# --- report blocks, generated from the CSV only ----------------------------------------------------------------------

def read(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def g3(v):
    return "—" if v in ("", None) else f"{float(v):.3g}"


def report_blocks(out_dir=OUT):
    rows = read(out_dir / "step0_energy.csv")
    runs = read(out_dir / "step0_runs.csv")
    r1 = read(out_dir / "step0_r1_regression.csv")
    meta = json.loads((out_dir / "metadata.json").read_text(encoding="utf-8"))
    blocks = {}

    lines = ["| r | сем. | сеток (квал.) | статусы выхода | обяз. строк | PASS | блокеры | min m₂ (обяз.) | min m₃ (обяз.) | "
             "max \\|Q−Q_sur\\|/(Q_sur·e_sur) (обяз.) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for label, _, family, chain in chains():
        sel = [r for r in rows if r["r"] == label and r["family"] == family]
        man = [r for r in sel if r["mandatory"] == "True"]
        rr = [r for r in runs if r["r"] == label and r["family"] == family]
        statuses = sorted({r["status"] for r in rr})
        m2 = [float(r["m_s"]) for r in man if r["s"] == "2"]
        m3 = [float(r["m_s"]) for r in man if r["s"] == "3"]
        dev = [float(r["dev_over_e_sur"]) for r in man if r["dev_over_e_sur"]]
        lines.append(f"| {label} | {family} | {len(rr)} ({sum(r['qualification'] == 'True' for r in rr)}) | "
                     f"{', '.join(statuses)} | {len(man)} | {sum(r['verdict'] == 'PASS' for r in man)} | "
                     f"{sum(r['verdict'] in ('BLOCKER', 'FAIL') for r in sel)} | {g3(min(m2)) if m2 else '—'} | "
                     f"{g3(min(m3)) if m3 else '—'} | {g3(max(dev)) if dev else '—'} |")
    lines.append("")
    lines.append(f"Всего: прогонов {meta['n_runs']}, строк s ≥ 2 — {meta['n_energy_rows']}, обязательных — "
                 f"{meta['n_mandatory_rows']}, из них PASS — {meta['n_mandatory_pass']}; блокеров и FAIL — "
                 f"{meta['n_blockers']}; расхождений статуса QA/решатель — {meta['n_status_mismatch']}; "
                 f"побитовая регрессия r = 1 — {'да' if meta['r1_bitwise_all'] else 'НЕТ'}. Вердикт шага 0: **{meta['verdict']}**.")
    blocks["step0_summary"] = "\n".join(lines)

    def table(selection, title_cols=True):
        out = ["| r | сем. | (n_x, n_y) | s | m_s | m_up (решатель) | m_up (зам.) | Δm/m | \\|γ_s\\|/g_allow | E1 | E2 | "
               "\\|Q−Q_sur\\|/Q_sur | e_sur | статус | вердикт |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in selection:
            ga = abs(float(r["gamma"])) / float(r["g_allow"]) if float(r["g_allow"]) > 0 else float("inf")
            out.append(f"| {r['r']} | {r['family']} | ({r['nx']},{r['ny']}) | {r['s']} | {g3(r['m_s'])} | {g3(r['m_up'])} | "
                       f"{g3(r['m_up_sur'])} | {g3(r['dm_over_m'])} | {ga:.2g} | {r['e1']} | {r['e2']} | "
                       f"{g3(r['dev_Q_sur'])} | {g3(r['e_sur'])} | {r['status_qa']} | {r['verdict']} |")
        return "\n".join(out)

    blocks["step0_mandatory_s3"] = table([r for r in rows if r["mandatory"] == "True" and r["s"] == "3"])
    blocks["step0_outside_s3"] = table([r for r in rows if r["mandatory"] != "True" and r["s"] == "3"
                                        and r["qualification"] == "True"])
    blocks["step0_priority"] = table([r for r in rows if r["priority"] == "True" and r["s"] in ("2", "3")
                                      and r["qualification"] == "True"])
    worst = sorted((r for r in rows if r["mandatory"] == "True" and r["s"] == "2"), key=lambda r: float(r["m_s"]))[:5]
    blocks["step0_s2_lowest"] = table(worst)
    blk = [r for r in rows if r["verdict"] in ("BLOCKER", "FAIL")]
    blocks["step0_blockers"] = table(blk) if blk else "Блокеров и FAIL нет."

    agree = ["| величина | max по всем строкам s ≥ 2 |", "|---|---|"]
    for k, name in (("diff_Q_over_tau", "\\|Q_реш − Q_QA\\|/τ_QA"), ("diff_delta_P_over_tau", "\\|ΔP_реш − ΔP_QA\\|/τ_QA"),
                    ("rel_diff_tau", "\\|τ_реш − τ_QA\\|/τ_QA"), ("abs_diff_gamma_over_tau", "\\|γ_реш − γ_QA\\|/τ_QA"),
                    ("rel_diff_coefficients", "коэффициенты A…F: max отн. разность")):
        agree.append(f"| {name} | {g3(max(float(r[k]) for r in rows if r[k]))} |")
    agree.append(f"| совпадение статусов QA и решателя | {sum(r['status_match'] == 'True' for r in rows)} из {len(rows)} |")
    blocks["step0_agreement"] = "\n".join(agree)

    reg = ["| n | статус R3 / R2 | s | побитово | m₂ QA | m₂ R2 (опубл.) | отн. | m₃ QA | m₃ R2 (опубл.) | отн. |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for r in r1:
        reg.append(f"| {r['n']} | {r['status_r3']} / {r['status_r2']} | {r['s_stop']} | {r['bitwise_equal']} | "
                   f"{g3(r['m2_qa'])} | {g3(r['m2_r2_published'])} | {g3(r['m2_rel_diff'])} | {g3(r['m3_qa'])} | "
                   f"{g3(r['m3_r2_published'])} | {g3(r['m3_rel_diff'])} |")
    blocks["step0_r1"] = "\n".join(reg)
    return blocks


START, END = "<!-- AUTO:{} -->", "<!-- /AUTO:{} -->"


def report_block_contents(path):
    text = Path(path).read_text(encoding="utf-8")
    out = {}
    pos = 0
    while (i := text.find("<!-- AUTO:", pos)) >= 0:
        name = text[i + 10:text.index(" -->", i)]
        j = text.index(END.format(name), i)
        out[name] = text[text.index("-->", i) + 3:j].strip("\n")
        pos = j
    return out


def inject_report(path, blocks):
    if not Path(path).exists():
        return
    text = Path(path).read_text(encoding="utf-8")
    for name, block in blocks.items():
        start, end = START.format(name), END.format(name)
        if start not in text:
            continue
        i, j = text.index(start) + len(start), text.index(end)
        text = text[:i] + "\n" + block + "\n" + text[j:]
    Path(path).write_text(text, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--report-only", action="store_true", help="regenerate the report blocks from the stored CSV")
    args = parser.parse_args()
    if args.report_only:
        inject_report(REPORT, report_blocks(OUT))
    else:
        run()
    sys.exit(0)
