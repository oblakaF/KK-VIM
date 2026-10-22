"""R5-V CCCC: observations of the QA half-step controls R-H, R-AC, R-HP on a chain of grids (OBSERVATION, no thresholds).

Runs only kkvim_reference (no kkvim). Writes docs/results/r5_cccc_halfstep/ (or --out):
- chain.csv        N = 1, 2, 3, both directions, h_x = h_y, n_f x n_s up to 256 x 512: cond(A), cond(K), certified lambda_low,
                   a posteriori bounds of the SET S, ||s - s*||_2 / ||s||_2 (certificate by the exact defect, review 57d3194), of
                   the unrefined and the refined R-AC solution, the ratio bound / (cond(K) u), the last refinement
                   correction (estimate), eta_QA (NS 5) of both, R-H against R-AC where the dense R-H is affordable (n <= 32);
- ill_conditioned.csv  near-dependent given sets G_2 = G_1 + eps G_2 (no QR): cond(A), certification, field difference R-H / R-AC;
- meta.json; the first edition is kept in history_b72515b/ (review 57d3194).

    uv run python scripts/r5_cccc_halfstep_observations.py          # about 1 min on the QA machine
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


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0])
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, lineterminator="\n")
        w.writeheader()
        for row in rows:
            w.writerow({k: (f"{v:.6e}" if isinstance(v, float) else v) for k, v in row.items()})


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
                       "cond_A": d["cond_A"], "cond_K": d["cond_K"], "lambda_low_certified": low,
                       "bound_unrefined_rel": None if low is None else hs.error_bound(p, ac.s_unrefined, low, coeffs) / norm_s,
                       "bound_refined_rel": None if low is None else hs.error_bound(p, ac.s, low, coeffs) / norm_s,
                       "bound_refined_over_condK_u": None if low is None else
                       hs.error_bound(p, ac.s, low, coeffs) / norm_s / (d["cond_K"] * hs.U),
                       "e_AC_rel_estimate": ac.e_AC / float(np.max(np.abs(ac.s))), "refinement_steps": len(ac.corrections),
                       "eta_unrefined": hs.eta_qa(p, ac.s_unrefined, coeffs), "eta_refined": hs.eta_qa(p, ac.s, coeffs)}
                if n_f <= 32 and n_s <= 32:
                    _, s_h = hs.solve_hessian(p)
                    row["RH_minus_RAC_rel"] = float(np.linalg.norm(s_h - ac.s)) / norm_s
                    row["RH_bound_rel"] = None if low is None else hs.error_bound(p, s_h, low, coeffs) / norm_s
                row["seconds"] = round(time.time() - t0, 2)
                row["status"] = "OBSERVATION"
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
        cert = ("certified" if low is not None else "not certified (factorization of K - tau I failed)") if run_cert else             "not run: computed lambda_min(K) in double <= 0 (the exact K is SPD: independent set)"
        row = {"eps": eps, "set_status": d["status"], "RH_status": status_h, "RAC_status": ac.status, "cond_A": d["cond_A"],
               "cond_K": d["cond_K"], "certificate": cert,
               "bound_refined_rel": None if (low is None or ac.s is None) else
               hs.error_bound(p, ac.s, low) / float(np.linalg.norm(ac.s)),
               "field_RH_minus_RAC_rel": None, "set_RH_minus_RAC_rel": None,
               "note": "given set without QR (OBSERVATION; PL BC-6 analogue for the QA controls)"}
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
            "seconds": round(time.time() - t0)}
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
