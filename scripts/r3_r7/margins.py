"""R3 margins Q_s/(10 tau_s) by the surrogate for all control r, families I and II, s = 2, 3 (4 diagnostic). Scratch.

Rules: scratch/r7/RULES_fixed_before_R3.md (fixed before this script was run).
Output: scratch/r7/r7_margins.csv
"""

import csv
import math
import sys
import time

sys.path.insert(0, "scratch/r7")
import surrogate as S  # noqa: E402

C_S = {2: 12.596892390516768, 3: 7.152240717468618, 4: 7.152240717468618}  # rules, item 2


class Cached(S.Surrogate):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self._vals = {}

    def values(self, index, n, coeffs=None):
        if coeffs is None and index is not None:
            key = (index, n)
            if key not in self._vals:
                self._vals[key] = super().values(index, n)
            return self._vals[key]
        return super().values(index, n, coeffs)


def category(m_up, e, small):
    low, high = m_up * (1 - e), m_up * (1 + e)
    if low >= 4:
        return "RESOLVABLE_NO_HEADROOM" if small else "MANDATORY_CANDIDATE"
    if low >= 1:
        return "RESOLVABLE_NO_HEADROOM"
    if high >= 1:
        return "BORDERLINE"
    return "EXPECTED_UNRESOLVED"


rows = []
t_start = time.time()
for label, r in S.RATIOS.items():
    a, b = 1.0, float(r)
    sur = Cached(a, b, s_max=4)
    Qc = {s: sur.continuous_Q(s) for s in (2, 3, 4)}
    grids = [("I", nx, ny) for nx, ny in S.family_I()] + [("II", nx, ny) for nx, ny in S.family_II(r)]
    for fam, nx, ny in grids:
        hx, hy = a / nx, b / ny
        l_min = min(a, b)
        small = min(nx, ny) < 20
        for s in (2, 3, 4):
            g = S.test_quantities(sur, s, nx, ny)
            e = 2 * C_S[s] * ((hx / l_min) ** 2 + (hy / l_min) ** 2) / 2
            cat = category(g["margin_upper"], e, small) if s in (2, 3) else "DIAGNOSTIC_S4"
            rows.append({
                "r": label, "r_value": b, "a": a, "b": b, "family": fam, "nx": nx, "ny": ny, "hx": hx, "hy": hy, "s": s,
                "Q_sur": g["Q"], "Q_cont": Qc[s], "R_prev": g["R_prev"], "R_now": g["R_now"], "R_f": g["R_f"],
                "tau_R": g["tau_R"], "m_up": g["margin_upper"], "gamma_allow": g["gamma_allowance"],
                "gamma_allow_over_tauR": g["gamma_allowance"] / g["tau_R"], "e_sur": e,
                "m_low": g["margin_upper"] * (1 - e), "m_high": g["margin_upper"] * (1 + e),
                "coarse_lt20": small, "category": cat,
            })
    print(f"r={label}: {len(grids)} grids, t={time.time() - t_start:.0f}s", flush=True)

with open("scratch/r7/r7_margins.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print("rows", len(rows))
