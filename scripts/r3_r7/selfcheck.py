"""Self-check of the surrogate formulas against kkvim.energy functions on rectangular grids (arrays only). Scratch."""

import math
import sys

sys.path.insert(0, "scratch/r7")
import surrogate as S  # noqa: E402

from kkvim.energy import discrete_energy, rounding_threshold
from kkvim.grid import Grid1D

worst = 0.0
for label in ("0,2", "1,4", "5"):
    r = S.RATIOS[label]
    sur = S.Surrogate(1.0, float(r), s_max=3)
    pairs = [(40, 40), (60, 28)] + S.family_II(r)[:2]
    for nx, ny in pairs:
        gx, gy = Grid1D(1.0, nx), Grid1D(float(r), ny)
        for s in (2, 3):
            mine = S.test_quantities(sur, s, nx, ny)
            d = sur.field(s - 1, nx, ny) - sur.field(s, nx, ny)
            d_factored = sur.change(s, nx, ny)
            ref_Q = 0.5 * discrete_energy(gx, gy, d, 1.0, 1.0).a
            ref_R = rounding_threshold(gx, gy, sur.field(s, nx, ny), 1.0, 1.0)
            e1 = abs(mine["Q"] - ref_Q) / ref_Q
            e2 = abs(mine["R_now"] - ref_R) / ref_R
            e3 = float(abs(d - d_factored).max() / abs(d).max())
            worst = max(worst, e1, e2)
            print(f"r={label} ({nx},{ny}) s={s}: Q rel {e1:.1e}  R rel {e2:.1e}  d(fields)-d(factored) {e3:.1e}")
print("worst Q/R relative difference:", worst)
