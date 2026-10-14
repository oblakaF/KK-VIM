"""Extra summaries for the R-7 draft: s = 4 diagnostics, calibration sensitivity at r = 1, Q_sur/Q_cont. Scratch."""

import csv

rows = list(csv.DictReader(open("scratch/r7/r7_margins.csv")))
for r in rows:
    for k in ("nx", "ny", "s"):
        r[k] = int(r[k])
    for k in ("m_up", "e_sur", "Q_sur", "Q_cont"):
        r[k] = float(r[k])
ORDER = ["0,2", "1/3", "1/2", "2/3", "1", "1,4", "1,5", "2", "3", "5"]
s4 = [r for r in rows if r["s"] == 4]
best = max(s4, key=lambda r: r["m_up"])
print("s=4: max m_up", best["m_up"], best["r"], best["family"], best["nx"], best["ny"])
print("s=4 cases with m_up >= 1:", sum(1 for r in s4 if r["m_up"] >= 1), "of", len(s4))
for lab in ORDER:
    sel = [r for r in s4 if r["r"] == lab and min(r["nx"], r["ny"]) >= 20]
    m = max(sel, key=lambda r: r["m_up"])
    print(f"  r={lab}: s=4 max m_up (min n >= 20) {m['m_up']:.3g} at {m['family']} ({m['nx']},{m['ny']})")

# measured calibration error law at r = 1 (no factor 2): |dm| ~ C_s / n^2
C = {2: 12.596892390516768, 3: 7.152240717468618}
flips = []
for r in rows:
    if r["r"] == "1" and r["s"] in (2, 3) and min(r["nx"], r["ny"]) >= 20:
        e = C[r["s"]] / (r["nx"] * r["ny"])
        for t in (1.0, 4.0):
            if r["m_up"] * (1 - e) < t <= r["m_up"] * (1 + e):
                flips.append((r["family"], r["nx"], r["ny"], r["s"], r["m_up"], e, t))
print("r=1 cases whose class could change within the measured calibration error:", flips)

print("Q_sur / Q_cont (family I):")
for lab in ORDER:
    out = []
    for n in (20, 160, 640):
        for s in (2, 3):
            x = next(r for r in rows if r["r"] == lab and r["family"] == "I" and r["nx"] == n and r["s"] == s)
            out.append(f"n={n},s={s}: {x['Q_sur'] / x['Q_cont'] - 1:+.2e}")
    q2 = next(r for r in rows if r["r"] == lab and r["s"] == 2)["Q_cont"]
    q3 = next(r for r in rows if r["r"] == lab and r["s"] == 3)["Q_cont"]
    print(f"  r={lab}: Q2_cont={q2:.5e} Q3_cont={q3:.5e}; " + "; ".join(out))

# family I boundaries at the nested levels for s = 3
print("s=3 family I, nested levels 20..640, category:")
for lab in ORDER:
    out = []
    for n in (20, 40, 80, 160, 320, 640):
        x = next(r for r in rows if r["r"] == lab and r["family"] == "I" and r["nx"] == n and r["s"] == 3)
        out.append(f"{n}:{x['category'][:4]}")
    print(f"  r={lab}: " + " ".join(out))
