"""R5-V CCCC: pilot of the certified DST-I reference (G-3 / OK-7). NOT the E-5 campaign: a few vectors per n, no thresholds, no c_DST.

Runs only kkvim_reference (no kkvim). Writes docs/results/r5_cccc_q3_dst_pilot/ (or --out): dst.csv, boundary.csv, meta.json.
n = 3, 4, 7, 16, 32, 64, then 128, 256, 512 staged with a time limit per case (UNRESOLVED_RESOURCES instead of lower precision).
Vectors: e_1, e_m, ones (the load F (x) 1), random normal, random normal times 10^U(-8, 8) (mixed scales), psi_1 in double (strong
cancellation: y close to e_1). Printing: *_up rounded up, *_down rounded down (decimal, directed); other numbers OBSERVATION.

    uv run python scripts/r5_cccc_q3_dst_pilot.py        # well under a minute
"""

import argparse
import csv
import json
import platform
import subprocess
import sys
import time
from fractions import Fraction
from importlib import metadata
from pathlib import Path

import numpy as np

from kkvim_reference import r5_cccc_bc as bc
from kkvim_reference import r5_cccc_dst_cert as dc
from kkvim_reference import r5_cccc_halfstep as hs

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "results" / "r5_cccc_q3_dst_pilot"
SMALL = [3, 4, 7, 16, 32, 64]
STAGED = [128, 256, 512]
TIME_LIMIT = 120.0  # seconds per certified DST (resource limit of the pilot)


def vectors(n):
    m = n - 1
    rng = np.random.default_rng(1000 + n)
    yield "e_1", np.eye(m)[0]
    yield "e_m", np.eye(m)[-1]
    yield "ones", np.ones(m)
    yield "random", rng.standard_normal(m)
    yield "mixed_scales", rng.standard_normal(m) * 10.0 ** rng.uniform(-8, 8, m)
    yield "psi_1_double", bc.dst(np.eye(m)[0])


def fmt(key, v):
    if isinstance(v, bool) or not isinstance(v, float):
        return v
    if key.endswith("_up"):
        return hs.decimal_up(v)
    if key.endswith("_down"):
        return hs.decimal_down(v)
    return f"{v:.6e}"


def write(path, rows):
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: fmt(k, r.get(k)) for k in keys})


def up(x: Fraction) -> float:
    return dc._up(x)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default=str(OUT))
    out = Path(parser.parse_args(argv).out)
    t0 = time.time()
    rows, brows = [], []
    for n in SMALL + STAGED:
        t1 = time.perf_counter()
        tab = dc.psi_table(n, dc.DEFAULT_BITS)
        t_tab = time.perf_counter() - t1
        c_lo, c_hi = tab.sqrt_bounds
        for name, x in vectors(n):
            ref = dc.certified_dst(x, dc.DEFAULT_BITS, time_limit=TIME_LIMIT)
            row = {"n": n, "vector": name, "bits": dc.DEFAULT_BITS, "stage": "small" if n in SMALL else "staged",
                   "status": ref.status, "table_seconds": round(t_tab, 4), "dst_seconds": round(ref.seconds, 4),
                   "sin_width_max_up": up(tab.sin_width_max), "sqrt_width_up": up(Fraction(c_hi - c_lo, 1 << dc.DEFAULT_BITS))}
            row.update({k: v for k, v in dc.compare(bc.dst(x), ref, x).items() if k not in ("status", "n", "bits")})
            rows.append(row)
        u = dc.U
        diag = dc.double_boundary_diagnostics(n, dc.DEFAULT_BITS)
        mx = lambda key: max(max(abs(d[key][0]), abs(d[key][1])) for d in diag)  # noqa: E731
        brow = {"n": n, "formula": "math.sqrt(2.0/n) * np.sin(k * math.pi / n) (r5_cccc_bc.solve_bc, psi_k(1))",
                "max_abs_arg_error_up": up(mx("arg_error")), "max_norm_part_over_u_up": up(mx("norm_part") / u),
                "max_arg_part_over_u_up": up(mx("arg_part") / u), "max_sin_part_over_u_up": up(mx("sin_part") / u),
                "max_product_part_over_u_up": up(max(abs(d["product_part"]) for d in diag) / u),
                "max_total_over_u_up": up(mx("total") / u)}
        if n <= 64:
            o = dc.orthogonality_check(n, dc.DEFAULT_BITS)
            brow["orthogonality_contains_identity"] = o["contains_identity"]
            brow["orthogonality_max_width_up"] = up(o["max_width"])
        brows.append(brow)
        print(n, round(t_tab, 3), flush=True)
    write(out / "dst.csv", rows)
    write(out / "boundary.csv", brows)
    meta = {"study": "pilot of the certified DST-I reference (G-3 / OK-7); not the E-5 campaign", "python": platform.python_version(),
            "platform": platform.platform(), "numpy": metadata.version("numpy"), "scipy": metadata.version("scipy"),
            "bits": dc.DEFAULT_BITS, "time_limit_per_case_s": TIME_LIMIT, "seconds": round(time.time() - t0),
            "path_under_test": "r5_cccc_bc.dst = scipy.fft.dst(type=1, norm='ortho')",
            "printing": "*_up rounded up, *_down rounded down (decimal, directed); other numbers OBSERVATION"}
    try:
        meta["git_commit_base"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                                                 check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        meta["git_commit_base"] = "unknown"
    (out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    print("done", meta["seconds"], "s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
