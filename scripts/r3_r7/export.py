"""Export compact CSV and metadata of the R-7 surrogate study into docs/results/r3/r7/. Scratch."""

import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np

OUT = "docs/results/r3/r7"
os.makedirs(OUT, exist_ok=True)
R2_LEVELS = {20, 40, 80, 160, 320, 640}

calib = list(csv.DictReader(open("scratch/r7/calibration_r1.csv")))
disc = {(int(r["n"]), int(r["s"])): r for r in calib}

cols = ["r", "a", "b", "family", "nx", "ny", "hx", "hy", "s", "Q_cont", "Q_sur", "R_prev", "R_now", "R_f", "tau_R",
        "m_up", "e_sur", "m_low", "m_high", "gamma_allow_over_tauR", "category", "coarse_lt20", "evidence", "m_disc_R2"]
with open(f"{OUT}/margins.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(cols)
    for r in csv.DictReader(open("scratch/r7/r7_margins.csv")):
        nx, ny, s = int(r["nx"]), int(r["ny"]), int(r["s"])
        measured = r["r"] == "1" and r["family"] == "I" and nx in R2_LEVELS
        ev = "R2_DISCRETE_AND_SURROGATE" if measured else "SURROGATE"
        mdisc = f"{float(disc[(nx, s)]['margin_disc']):.6e}" if measured else ""
        cat = r["category"] if s in (2, 3) else "DIAGNOSTIC_S4"
        row = [r["r"], f"{float(r['a']):.17g}", f"{float(r['b']):.17g}", r["family"], nx, ny,
               f"{float(r['hx']):.6e}", f"{float(r['hy']):.6e}", s]
        row += [f"{float(r[k]):.6e}" for k in ("Q_cont", "Q_sur", "R_prev", "R_now", "R_f", "tau_R", "m_up", "e_sur", "m_low",
                                               "m_high", "gamma_allow_over_tauR")]
        row += [cat, r["coarse_lt20"], ev, mdisc]
        w.writerow(row)

ccols = ["n", "s", "Q_disc", "Q_sur", "Q_cont", "dQ_rel", "dQcont_rel", "R_prev_disc", "R_prev_sur", "R_now_disc", "R_now_sur",
         "R_f_disc", "R_f_sur", "gamma_disc", "tau_disc", "margin_disc", "margin_upper_disc", "margin_upper_sur", "dmargin_rel",
         "status_disc"]
with open(f"{OUT}/calibration_r1.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(ccols)
    for r in calib:
        w.writerow([r["n"], r["s"]] + [f"{float(r[k]):.9e}" for k in ccols[2:-1]] + [r["status_disc"]])


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


import kkvim_reference  # noqa: E402

commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
meta = {
    "study": "R-7 (Q-029): surrogate margins Q_s/(10 tau_s) of the energy test A-7 for R3 rectangles, preliminary",
    "base_commit": commit,
    "date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform(),
    "plate": "a = 1, b = r, q = D = 1 (margins invariant under a, q, D at fixed r, nx, ny)",
    "surrogate": "kkvim_reference.parseval_recursion (b = r, s_max = 4) sampled at nodes; discrete a_h, R, R_f of NUMERICAL_SPEC 4.1, 10.3",
    "calibration": "kkvim.iteration.run_halfsteps + kkvim.energy.check_history (accepted R2 code, run only), r = 1, n = 20..640",
    "C_s": {"2": 12.596892390516768, "3": 7.152240717468618, "4": 7.152240717468618},
    "rules_sha256": sha("scratch/r7/RULES_fixed_before_R3.md"),
    "scripts_sha256": {p: sha(f"scratch/r7/{p}") for p in ("surrogate.py", "calibrate.py", "margins.py", "summarize.py",
                                                         "extra.py", "selfcheck.py", "export.py")},
    "scripts_location": "scratch/r7 (outside git, task rule)",
    "rows": {"margins": 6534, "grid_pairs": 2178},
}
json.dump(meta, open(f"{OUT}/metadata.json", "w"), indent=2)
print(json.dumps(meta, indent=1)[:900])
