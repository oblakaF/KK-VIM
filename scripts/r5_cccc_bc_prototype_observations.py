"""R5-V CCCC: observations of the QA research prototype B-C (no thresholds; not the Q3 calibration of E-4, E-6).

Runs only kkvim_reference (no kkvim). Writes docs/results/r5_cccc_bc/ (or --out):
- chain.csv  N = 1, 2, 3, both directions, h_x = h_y, n_f x n_s from 16 x 16 to 256 x 512, plus b/a = 1/2 and 2 at 32 x 32
             (with QR for N >= 2, order NS 5.1);
- ill_conditioned.csv  G_2 = G_1 + eps G_2 (32 x 32, N = 2), without and with QR;
- meta.json  environment and the meaning of the columns.
Columns: *_hex exact double of a certified value, *_up decimal rounded up, *_down rounded down (R-4 conventions of
r5_cccc_halfstep_observations); UNRESOLVED - no certificate or condition not met; all other numbers OBSERVATION.
The certified bounds are bounds of the SET S of the system actually solved (after QR if applied), not of the field W.

    uv run python scripts/r5_cccc_bc_prototype_observations.py        # about 1 min
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

from kkvim_reference import r5_cccc_bc as bc
from kkvim_reference import r5_cccc_halfstep as hs

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "results" / "r5_cccc_bc"
CHAIN = [(16, 16), (32, 32), (64, 128), (128, 256), (256, 512)]
LOG = ["cond_A_before_QR", "cond_A_after_QR", "orthonormality_defect_up", "cond_S", "min_eig_sym_S", "max_cond_Mk", "kappa_c",
       "eta_bc"]
COLUMNS = {
    "lambda_low_hex / _down": "certified lower bound of lambda_min(K) of the system solved (certify_lambda_min)",
    "bound_abs_bc_hex / _up": "certified upper bound of ||s_BC - s*||_2 (error_bound)",
    "rel_to_exact_bc_hex / _up": "certified upper bound of ||s_BC - s*||_2/||s*||_2 = b/(L - b), L > b",
    "rel_to_exact_ac_hex / _up": "the same for R-AC with refinement",
    "bc_ac_certified_consistent, bc_rh_certified_consistent": "exact check ||s_BC - s_ref|| <= b_BC + b_ref (must be True)",
    "cond_A_before_QR, cond_A_after_QR, cond_S, min_eig_sym_S, max_cond_Mk, kappa_c": "OBSERVATION (NS 5.1 log; E-6 preparation)",
    "orthonormality_defect_up": "exact max|h_f U~^T U~ - I| rounded up (QR control), value only",
    "eta_bc, bc_minus_ac_rel, bc_minus_rh_rel, seconds": "OBSERVATION",
}


def certified(name, value, side="up"):
    if value is None or value == bc.UNRESOLVED:
        return {f"{name}_hex": bc.UNRESOLVED, f"{name}_{side}": bc.UNRESOLVED}
    text = hs.decimal_up(value) if side == "up" else hs.decimal_down(value)
    return {f"{name}_hex": value.hex(), f"{name}_{side}": text}


def row_of(p, qr, extra):
    t0 = time.time()
    rec = bc.run_halfstep(p, qr=qr)
    c = bc.compare(rec)
    row = {**extra, "N": p.N, "direction": p.direction, "n_f": p.n_f, "n_s": p.n_s, "qr": rec.log["qr"], "status": rec.status}
    row.update({k: rec.log.get(k) for k in LOG})
    row.update(certified("lambda_low", c.get("lambda_low"), "down"))
    for name in ("bound_abs_bc", "rel_to_exact_bc", "rel_to_exact_ac"):
        row.update(certified(name, c.get(name)))
    for name in ("bc_ac_certified_consistent", "bc_rh_certified_consistent", "bc_minus_ac_rel", "bc_minus_rh_rel"):
        row[name] = c.get(name)
    row["seconds"] = round(time.time() - t0, 2)
    print(p.direction, p.N, p.n_f, p.n_s, qr, flush=True)
    return row


def write(path, rows):
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6e}" if isinstance(v, float) else v) for k, v in ((k, r.get(k)) for k in keys)})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default=str(OUT))
    out = Path(parser.parse_args(argv).out)
    t0 = time.time()
    chain = []
    for N in (1, 2, 3):
        for d in hs.DIRECTIONS:
            for n_f, n_s in CHAIN:
                lf, ls = 1.0, n_s / n_f
                lx, nx, ly, ny = (lf, n_f, ls, n_s) if d == "Y" else (ls, n_s, lf, n_f)
                chain.append(row_of(hs.HalfStep(lx, nx, ly, ny, d, hs.start_set(n_f, N)), True, {"b_over_a": ls}))
            for r in (0.5, 2.0):
                lx, nx, ly, ny = (1.0, 32, r, 32) if d == "Y" else (r, 32, 1.0, 32)
                chain.append(row_of(hs.HalfStep(lx, nx, ly, ny, d, hs.start_set(32, N)), True, {"b_over_a": r}))
    write(out / "chain.csv", chain)
    ill = []
    for eps in (1e-2, 1e-4, 1e-6, 1e-8):
        G = hs.start_set(32, 2)
        G[:, 1] = G[:, 0] + eps * G[:, 1]
        for qr in (False, True):
            ill.append(row_of(hs.HalfStep(1.0, 32, 1.0, 32, "Y", G), qr, {"eps": eps}))
    write(out / "ill_conditioned.csv", ill)
    meta = {"study": "R5-V CCCC QA research prototype B-C (self-checks; no thresholds, no Q3 calibration)",
            "python": platform.python_version(), "platform": platform.platform(), "numpy": metadata.version("numpy"),
            "scipy": metadata.version("scipy"), "seconds": round(time.time() - t0), "columns": COLUMNS}
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
