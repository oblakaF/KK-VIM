"""R5-V CCCC: pilot journal of the pre-Q3 E-4 / E-6 instrument (small grids; NOT the Q3 campaign, no thresholds).

Runs only kkvim_reference (no kkvim). Writes docs/results/r5_cccc_q3_e4e6_pilot/ (or --out): e4.csv, e6.csv, meta.json.
Cases: grids 8 x 7 and 10 x 9 (exact tier) and 24 x 20 (certified R-AC tier); N = 1, 2, 3; both directions; without and with QR for
N >= 2; a near-dependent set (eps = 1e-6, N = 2); q = 0 (N = 2).
Printing: *_up rounded up, *_down rounded down (certified bounds and intervals); other numbers OBSERVATION (nearest); statuses as text
(UNRESOLVED, RESEARCH_ONLY, PROVISIONAL / NOT_QUALIFIED).

    uv run python scripts/r5_cccc_q3_e4e6_pilot.py        # about 1 min
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

from kkvim_reference import r5_cccc_halfstep as hs
from kkvim_reference import r5_cccc_q3_e4e6 as q

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "results" / "r5_cccc_q3_e4e6_pilot"
GRIDS = [(1.0, 8, 1.3, 7), (1.0, 10, 0.6, 9), (1.0, 24, 1.2, 20)]


def cases():
    for lx, nx, ly, ny in GRIDS:
        for d in hs.DIRECTIONS:
            for N in (1, 2, 3):
                n_f = nx if d == "Y" else ny
                for qr in ((False,) if N == 1 else (False, True)):
                    yield "T_N", hs.HalfStep(lx, nx, ly, ny, d, hs.start_set(n_f, N)), qr
    for lx, nx, ly, ny in (GRIDS[0], GRIDS[2]):
        G = hs.start_set(nx, 2)
        G[:, 1] = G[:, 0] + 1e-6 * G[:, 1]
        for qr in (False, True):
            yield "near-dependent eps=1e-6", hs.HalfStep(lx, nx, ly, ny, "Y", G), qr
    yield "q = 0", hs.HalfStep(1.0, 8, 1.3, 7, "Y", hs.start_set(8, 2), q=0.0), True


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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default=str(OUT))
    out = Path(parser.parse_args(argv).out)
    t0 = time.time()
    e4, e6 = [], []
    for label, p, qr in cases():
        e4.append({"set": label, **q.e4_case(p, qr=qr)})
        e6.append({"set": label, **q.e6_case(p, qr=qr)})
        print(label, p.direction, p.N, p.n_f, p.n_s, qr, flush=True)
    write(out / "e4.csv", e4)
    write(out / "e6.csv", e6)
    meta = {"study": "pilot journal of the pre-Q3 E-4 / E-6 instrument (not Q3)", "python": platform.python_version(),
            "platform": platform.platform(), "numpy": metadata.version("numpy"), "scipy": metadata.version("scipy"),
            "exact_limit": q.EXACT_LIMIT, "seconds": round(time.time() - t0),
            "printing": "*_up rounded up, *_down rounded down; other numbers OBSERVATION (nearest)"}
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
