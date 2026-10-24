"""R5-V CCCC: pilot journal of the research QA half-step sequences with exact energy (G-1, G-2, G-4). Small grids only; NOT the Q3
campaign, no thresholds, no c_R, K_E, eps_floor^C.

Runs only kkvim_reference (no kkvim). Writes docs/results/r5_cccc_q3_sequence_pilot/ (or --out): sequence.csv, meta.json.
Cases: 10 x 9 (a = 1, b = 1.3) and 12 x 12 (square); N = 1, 2, 3; first half-step Y (D-049.3) and X; s = 1..8; QR for N >= 2
(RESEARCH_ONLY); a near-dependent start (eps = 1e-6, N = 2); q = 0 (N = 1).
Printing: *_up rounded up, *_down rounded down; other numbers OBSERVATION (nearest).

    uv run python scripts/r5_cccc_q3_sequence_pilot.py        # about 30 s
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

from kkvim_reference import r5_cccc_energy as en
from kkvim_reference import r5_cccc_halfstep as hs
from kkvim_reference import r5_cccc_sequence as sq

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "results" / "r5_cccc_q3_sequence_pilot"


def configs():
    for lx, nx, ly, ny in ((1.0, 10, 1.3, 9), (1.0, 12, 1.0, 12)):
        for N in (1, 2, 3):
            for first in ("Y", "X"):
                yield f"T_N {first}", sq.SequenceConfig(lx, nx, ly, ny, N, first=first)
    start = hs.start_set(10, 2)
    start[:, 1] = start[:, 0] + 1e-6 * start[:, 1]
    yield "near-dependent eps=1e-6", sq.SequenceConfig(1.0, 10, 1.3, 9, 2, start=start)
    yield "q = 0", sq.SequenceConfig(1.0, 10, 1.3, 9, 1, q=0.0)


def fmt(key, v):
    if isinstance(v, bool) or not isinstance(v, float):
        return v
    if key.endswith("_up"):
        return hs.decimal_up(v)
    if key.endswith("_down"):
        return hs.decimal_down(v)
    return f"{v:.6e}"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default=str(OUT))
    out = Path(parser.parse_args(argv).out)
    t0 = time.time()
    rows = []
    for label, cfg in configs():
        res = sq.run_sequence(cfg)
        for row in sq.journal(res):
            rows.append({"case": label, "grid": f"{cfg.nx}x{cfg.ny}", "sequence_status": res.status, **row})
        print(label, cfg.N, cfg.nx, cfg.ny, res.status, flush=True)
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "sequence.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: fmt(k, r.get(k)) for k in keys})
    meta = {"study": "pilot journal of research CCCC half-step sequences (G-1, G-2, G-4); not Q3",
            "python": platform.python_version(), "platform": platform.platform(), "numpy": metadata.version("numpy"),
            "scipy": metadata.version("scipy"), "c_R_count_of_the_double_path": en.C_R_COUNT, "seconds": round(time.time() - t0),
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
