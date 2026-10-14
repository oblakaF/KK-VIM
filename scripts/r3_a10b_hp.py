"""QA, A-10b of the R3 solver qualification (R3_NUMERICAL_SPEC 9.1, D-040 O-2, RV-4): the Parseval reference at s = 1
against the closed form В-29 computed with 50 decimal digits, at ALL nodes of every qualification grid of the chains on
which A-4 is compared (6.3, both families, every r != 1), relative to max|w1|, tolerance 1e-13; and the budgets of the
reference for s <= 3 (tails of И-12 <= 1e-14, transfer field_error <= 1e-12, NUMERICAL_SPEC 15.4, 15.5, 18.1).

В-29 (MATH_SPEC 9.1): w1 = sin(pi x/a) Y1(y), Y1 = C0 [1 - (alpha th alpha + 2)/(2 ch alpha) ch(pi y_c/a)
+ (pi y_c/a) sh(pi y_c/a)/(2 ch alpha)], C0 = 4 q a^4/(pi^5 D), alpha = pi b/(2a), y_c = y - b/2. The first half-step is
separable (9.1): Y1 at the n_y + 1 nodes and the sine at the n_x + 1 nodes are computed in 50 digits; the difference with
the double field of the reference, fl(x_i y_j), is formed exactly as e_ij + x_i dY_j + dX_i y_j + dX_i dY_j (e_ij the exact
error of the double product by Dekker's two-product, dX, dY the high-precision corrections of the factors), whose
evaluation error is of order u times 1e-16 max|w1|, far below the tolerance; on small grids it is checked against the
full product in 50 digits. mpmath is a temporary QA dependency only:

    uv run --with mpmath python scripts/r3_a10b_hp.py
"""

import csv
import datetime
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import mpmath as mp
import numpy as np

from kkvim_reference.parseval_recursion import ParsevalRecursion

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "results" / "r3" / "solver" / "a10b"
sys.path.insert(0, str(ROOT / "scripts"))
import r3_solver_step0 as step0  # noqa: E402

DPS = 50
TOL_HP, TOL_TAIL, TOL_TRANSFER = 1e-13, 1e-14, 1e-12


def split(a):
    c = 134217729.0 * a
    hi = c - (c - a)
    return hi, a - hi


def two_product(a, b):
    p = a * b
    ah, al = split(a)
    bh, bl = split(b)
    return p, ((ah * bh - p) + ah * bl + al * bh) + al * bl


def v29_factors(b, nx, ny):
    """sin(pi i/nx), i = 0..nx, and Y1(y_j), y_j = j b/ny, in DPS digits (a = q = D = 1, b the exact float)."""
    B = mp.mpf(b)
    alpha = mp.pi * B / 2
    C0 = 4 / mp.pi**5
    ch = mp.cosh(alpha)
    P = (alpha * mp.tanh(alpha) + 2) / (2 * ch)
    X = [mp.sin(mp.pi * i / nx) for i in range(nx + 1)]
    Y = []
    for j in range(ny + 1):
        t = mp.pi * (B * j / ny - B / 2)
        Y.append(C0 * (1 - P * mp.cosh(t) + t * mp.sinh(t) / (2 * ch)))
    return X, Y


def compare(P, b, nx, ny, full_check=False):
    x = np.asarray(P._multiplier_nodal(0, nx))  # X0 of the reference (s = 1: x factor index 0)
    y = np.asarray(P._multiplier_nodal(1, ny))  # Y1 of the reference
    w = P.field_on_grid(1, nx, ny)
    assert np.array_equal(np.outer(x, y), w)  # the reference field is the double product of its factors
    X, Y = v29_factors(b, nx, ny)
    dX = np.array([float(Xi - mp.mpf(float(xi))) for Xi, xi in zip(X, x)])
    dY = np.array([float(Yj - mp.mpf(float(yj))) for Yj, yj in zip(Y, y)])
    p, e = two_product(x[:, None], y[None, :])
    assert np.array_equal(p, w)
    # fl(xy) - X Y = p - (x + dX)(y + dY) = -(e + x dY + dX y + dX dY), e = x y - p exactly
    diff = -(e + (x[:, None] * dY[None, :] + dX[:, None] * y[None, :] + dX[:, None] * dY[None, :]))
    w_max = float(max(abs(v) for v in X) * max(abs(v) for v in Y))
    rel = float(np.max(np.abs(diff))) / w_max
    check = None
    if full_check:
        exact = max(abs(mp.mpf(float(w[i, j])) - X[i] * Y[j]) for i in range(nx + 1) for j in range(ny + 1))
        check = float(exact) / w_max
    return rel, check


def _git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def main():
    commit, dirty = _git("rev-parse", "HEAD"), bool(_git("status", "--porcelain"))
    mp.mp.dps = DPS
    OUT.mkdir(parents=True, exist_ok=True)
    rows, budgets, checks = [], [], []
    for label, r, _ in step0.RATIOS:
        b = float(r)
        P = ParsevalRecursion(a=1.0, b=b, q=1.0, D=1.0, s_max=3)
        for s in (1, 2, 3):
            bd = P.halfstep(s).budget
            tails = max(bd.tail_function_rel, bd.tail_A_rel, bd.tail_B_rel, bd.tail_C_rel, bd.tail_F_rel)
            budgets.append({"r": label, "s": s, "M": bd.M, "max_tail_rel": f"{tails:.3e}", "field_error": f"{bd.field_error:.3e}",
                            "invariant_error": f"{bd.invariant_error:.3e}", "valid": bd.valid,
                            "verdict_tail": "PASS" if tails <= TOL_TAIL else "FAIL",
                            "verdict_transfer": "PASS" if bd.valid and bd.field_error <= TOL_TRANSFER else "FAIL"})
        if label == "1":
            continue  # r = 1: A-4 reference is levy_w1 of R2 (A-4v, A-10b of R2)
        for family in ("I", "II"):
            for grid in chain_levels(label, family):
                rel, _ = compare(P, b, *grid)
                rows.append({"r": label, "family": family, "nx": grid[0], "ny": grid[1], "max_rel_dev": f"{rel:.3e}",
                             "verdict": "PASS" if rel <= TOL_HP else "FAIL"})
                print(label, family, grid, f"{rel:.2e}", flush=True)
        for grid in ((20, 20), (12, 20)):  # validation of the exact difference against the full 50-digit product
            rel, full = compare(P, b, *grid, full_check=True)
            checks.append({"r": label, "nx": grid[0], "ny": grid[1], "decomposed": f"{rel:.6e}", "full_product": f"{full:.6e}",
                           "agree": abs(rel - full) <= 1e-3 * max(full, 1e-30)})
    for name, data in (("a10b_hp.csv", rows), ("a10b_budgets.csv", budgets), ("a10b_method_check.csv", checks)):
        with open(OUT / name, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]), lineterminator="\n")
            w.writeheader()
            w.writerows(data)
    meta = {"study": "A-10b, R3 solver qualification", "commit": commit, "dirty": dirty,
            "date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "python": platform.python_version(), "numpy": np.__version__, "mpmath": mp.__version__, "dps": DPS,
            "tolerances": {"hp": TOL_HP, "tail": TOL_TAIL, "transfer": TOL_TRANSFER},
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "n_grids": len(rows), "max_rel_dev": max(float(r["max_rel_dev"]) for r in rows),
            "all_pass": all(r["verdict"] == "PASS" for r in rows) and all(
                b["verdict_tail"] == b["verdict_transfer"] == "PASS" for b in budgets) and all(c["agree"] for c in checks)}
    (OUT / "metadata.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=1))


def chain_levels(label, family):
    for lab, _, fam, chain in step0.chains():
        if (lab, fam) == (label, family):
            return [g for g in chain if step0.qualification(g)]
    return []


if __name__ == "__main__":
    main()
