"""R5-V CCCC: observations of the QA half-step controls R-H, R-AC, R-HP on a chain of grids (no thresholds).

Runs only kkvim_reference (no kkvim). Writes docs/results/r5_cccc_halfstep/ (or --out):
- chain.csv        N = 1, 2, 3, both directions, h_x = h_y, n_f x n_s up to 256 x 512: certified lambda_low, certified bounds of the
                   SET S (absolute; relative to the computed set s^ and to the exact solution s*) of the unrefined and the refined
                   R-AC solution and of R-H (n <= 32); cond(A), cond(K), bound / (cond(K) u), the last refinement correction
                   (estimate), eta_QA (NS 5), R-H against R-AC (OBSERVATION);
- ill_conditioned.csv  near-dependent given sets G_2 = G_1 + eps G_2 (no QR): certificate, certified bounds, field and set differences;
- meta.json        environment and the meaning of every column (``columns``).
Columns (R-4, review 4f92c82):
- ``*_hex``  exact double (float.hex) of a certified quantity;
- ``*_down`` certified LOWER bound printed with rounding towards -inf only (7 significant digits);
- ``*_up``   certified UPPER bound printed with rounding towards +inf only (7 significant digits);
- ``UNRESOLVED``  no certificate or the condition of the relative bound is not met (no number is substituted);
- all other numeric columns are OBSERVATION (rounded to nearest), among them cond_A, cond_K (eigvalsh) and bound / (cond(K) u).
The bounds are bounds of the SET S, not of the field W = G S^T. Earlier editions: history_b72515b/, history_aed5ba4/.

    uv run python scripts/r5_cccc_halfstep_observations.py          # about 10 s on the QA machine
"""

import argparse
import csv
import json
import platform
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

import numpy as np

from kkvim_reference import r5_cccc_halfstep as hs

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "results" / "r5_cccc_halfstep"
CHAIN = [(16, 16), (32, 32), (32, 64), (64, 128), (128, 256), (256, 512)]  # (n_f, n_s), h_x = h_y = 1/n_f
UNRESOLVED = "UNRESOLVED"

COLUMNS = {
    "lambda_low_hex": "certified lower bound of lambda_min(K), exact double (certify_lambda_min, tau = lambda_min_float/2)",
    "lambda_low_down": "the same, decimal rounded down",
    "bound_abs_{x}_hex": "certified upper bound of ||s^ - s*||_2 (error_bound), exact double",
    "bound_abs_{x}_up": "the same, decimal rounded up",
    "rel_to_computed_{x}_hex": "certified upper bound of ||s^ - s*||_2 / ||s^||_2 = b/L, L = norm_lower(s^) > 0, exact double",
    "rel_to_computed_{x}_up": "the same, decimal rounded up",
    "rel_to_exact_{x}_hex": "certified upper bound of ||s^ - s*||_2 / ||s*||_2 = b/(L - b), requires L > b, exact double",
    "rel_to_exact_{x}_up": "the same, decimal rounded up",
    "{x}": "unrefined / refined: R-AC before / after the refinement; RH: dense R-H solution (n <= 32)",
    "cond_A, cond_K, bound_refined_over_condK_u, e_AC_rel_estimate, eta_*, RH_minus_RAC_rel, *_RH_minus_RAC_rel, seconds":
        "OBSERVATION (floating point, rounded to nearest; cond from eigvalsh; bound_refined_over_condK_u = b/||s^||/(cond(K) u))",
    "UNRESOLVED": "no certificate, b = inf, or the condition L > 0 / L > b not met; no number is substituted",
}


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0])
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, lineterminator="\n")
        w.writeheader()
        for row in rows:
            w.writerow({k: (f"{v:.6e}" if isinstance(v, float) else v) for k, v in row.items()})


def certified(prefix, value, side):
    """{prefix}_hex and {prefix}_up / _down of a certified double; UNRESOLVED if None."""
    if value is None:
        return {f"{prefix}_hex": UNRESOLVED, f"{prefix}_{side}": UNRESOLVED}
    text = hs.decimal_up(value) if side == "up" else hs.decimal_down(value)
    return {f"{prefix}_hex": value.hex(), f"{prefix}_{side}": text}


def bounds(p, s, low, coeffs, tag):
    """Certified absolute and relative bounds of the set S for the computed set s (R-4); also returns b."""
    if low is None or s is None:
        b, rel = None, hs.RelativeBounds(None, None)
    else:
        b = hs.error_bound(p, s, low, coeffs)
        rel = hs.relative_bounds(b, s)
    return {**certified(f"bound_abs_{tag}", b, "up"), **certified(f"rel_to_computed_{tag}", rel.to_computed, "up"),
            **certified(f"rel_to_exact_{tag}", rel.to_exact, "up")}, b


def chain_rows():
    rows = []
    for N in (1, 2, 3):
        for direction in hs.DIRECTIONS:
            for n_f, n_s in CHAIN:
                lf, ls = 1.0, n_s / n_f
                lx, nx, ly, ny = (lf, n_f, ls, n_s) if direction == "Y" else (ls, n_s, lf, n_f)
                p = hs.HalfStep(lx, nx, ly, ny, direction, hs.start_set(n_f, N))
                t0 = time.time()
                coeffs = hs.exact_coefficients(p)
                d = hs.diagnostics(p)
                ac = hs.solve_ac(p)
                low = hs.certify_lambda_min(p, d["lambda_min_K_float"] / 2, coeffs)
                norm_s = float(np.linalg.norm(ac.s))
                row = {"N": N, "direction": direction, "n_f": n_f, "n_s": n_s, "unknowns": p.size,
                       **certified("lambda_low", low, "down")}
                cols_un, _ = bounds(p, ac.s_unrefined, low, coeffs, "unrefined")
                cols_ref, b = bounds(p, ac.s, low, coeffs, "refined")
                row.update(cols_un)
                row.update(cols_ref)
                row.update({"cond_A": d["cond_A"], "cond_K": d["cond_K"],
                            "bound_refined_over_condK_u": None if b is None else b / norm_s / (d["cond_K"] * hs.U),
                            "e_AC_rel_estimate": ac.e_AC / float(np.max(np.abs(ac.s))), "refinement_steps": len(ac.corrections),
                            "eta_unrefined": hs.eta_qa(p, ac.s_unrefined, coeffs), "eta_refined": hs.eta_qa(p, ac.s, coeffs)})
                if n_f <= 32 and n_s <= 32:
                    _, s_h = hs.solve_hessian(p)
                    row["RH_minus_RAC_rel"] = float(np.linalg.norm(s_h - ac.s)) / norm_s
                    row.update(bounds(p, s_h, low, coeffs, "RH")[0])
                row["seconds"] = round(time.time() - t0, 2)
                rows.append(row)
                print(N, direction, n_f, n_s, flush=True)
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    return [{k: r.get(k) for k in keys} for r in rows]


def ill_rows():
    rows = []
    for eps in (1e-2, 1e-4, 1e-6, 1e-8):
        G = hs.start_set(32, 2)
        G[:, 1] = G[:, 0] + eps * G[:, 1]
        p = hs.HalfStep(1.0, 32, 1.0, 32, "Y", G)
        d = hs.diagnostics(p)
        (status_h, s_h), ac = hs.solve_hessian(p), hs.solve_ac(p)
        both = s_h is not None and ac.s is not None
        run_cert = d["lambda_min_K_float"] > 0
        low = hs.certify_lambda_min(p, d["lambda_min_K_float"] / 2) if run_cert else None
        cert = ("certified" if low is not None else "not certified (factorization of K - tau I failed)") if run_cert else \
            "not run: computed lambda_min(K) in double <= 0 (the exact K is SPD: independent set)"
        row = {"eps": eps, "set_status": d["status"], "RH_status": status_h, "RAC_status": ac.status, "cond_A": d["cond_A"],
               "cond_K": d["cond_K"], "certificate": cert, **certified("lambda_low", low, "down")}
        row.update(bounds(p, ac.s, low, None, "refined")[0])
        row.update({"field_RH_minus_RAC_rel": None, "set_RH_minus_RAC_rel": None,
                    "note": "given set without QR (OBSERVATION; PL BC-6 analogue for the QA controls)"})
        if both:
            W_h, W_ac = G @ s_h.reshape(2, -1), G @ ac.s.reshape(2, -1)
            row["field_RH_minus_RAC_rel"] = float(np.max(np.abs(W_h - W_ac)) / np.max(np.abs(W_ac)))
            row["set_RH_minus_RAC_rel"] = float(np.linalg.norm(s_h - ac.s) / np.linalg.norm(ac.s))
        rows.append(row)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default=str(OUT))
    out = Path(parser.parse_args(argv).out)
    t0 = time.time()
    write(out / "chain.csv", chain_rows())
    write(out / "ill_conditioned.csv", ill_rows())
    meta = {"study": "R5-V CCCC half-step controls R-H, R-AC, R-HP (Q1b)", "python": platform.python_version(),
            "platform": platform.platform(), "numpy": metadata.version("numpy"), "scipy": metadata.version("scipy"),
            "seconds": round(time.time() - t0), "columns": COLUMNS,
            "note": "bounds of the SET S, not of the field W = G S^T; earlier editions in history_b72515b/, history_aed5ba4/"}
    try:
        meta["git_commit"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                                            check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        meta["git_commit"] = "unknown"
    (out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    print("done", meta["seconds"], "s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
