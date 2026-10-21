"""Calibration of the surrogate on accepted R2 (r = 1, a = q = D = 1): discrete kkvim vs surrogate. Scratch."""

import csv
import math
import sys
import time

sys.path.insert(0, "scratch/r7")
from surrogate import Surrogate, test_quantities  # noqa: E402

from kkvim.energy import check_history  # accepted R2 code, run only
from kkvim.iteration import run_halfsteps

GRIDS = (20, 40, 80, 160, 320, 640)
sur = Surrogate(1.0, 1.0, s_max=4)
rows = []
for n in GRIDS:
    t0 = time.time()
    checks = check_history(run_halfsteps(1.0, 1.0, 1.0, n, n, 4))
    for s in (2, 3, 4):
        c = checks[s - 1]
        g = test_quantities(sur, s, n, n)
        disc_upper = c.Q / (10 * 10 * (c.R_prev + c.R_now + c.R_f))
        rows.append({
            "n": n, "s": s,
            "Q_disc": c.Q, "Q_sur": g["Q"], "Q_cont": sur.continuous_Q(s),
            "dQ_rel": (g["Q"] - c.Q) / c.Q, "dQcont_rel": (sur.continuous_Q(s) - c.Q) / c.Q,
            "R_prev_disc": c.R_prev, "R_prev_sur": g["R_prev"], "R_now_disc": c.R_now, "R_now_sur": g["R_now"],
            "R_f_disc": c.R_f, "R_f_sur": g["R_f"], "gamma_disc": c.gamma, "tau_disc": c.tau,
            "margin_disc": c.resolution_ratio, "margin_upper_disc": disc_upper, "margin_upper_sur": g["margin_upper"],
            "dmargin_rel": (g["margin_upper"] - c.resolution_ratio) / c.resolution_ratio, "status_disc": c.status,
        })
    print(f"n={n} done {time.time() - t0:.1f}s", flush=True)
with open("scratch/r7/calibration_r1.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
for r in rows:
    print(f"n={r['n']:4d} s={r['s']} Qd={r['Q_disc']:.6e} Qs={r['Q_sur']:.6e} Qc={r['Q_cont']:.6e} dQ={r['dQ_rel']:+.2e} "
          f"dQc={r['dQcont_rel']:+.2e} | R {r['R_now_disc']:.3e}/{r['R_now_sur']:.3e} Rf {r['R_f_disc']:.3e}/{r['R_f_sur']:.3e} "
          f"g={r['gamma_disc']:+.1e} tau={r['tau_disc']:.3e} | m_disc={r['margin_disc']:.4g} m_up_disc={r['margin_upper_disc']:.4g} "
          f"m_sur={r['margin_upper_sur']:.4g} dm={r['dmargin_rel']:+.2e} {r['status_disc']}")
