"""Summary of scratch/r7/r7_margins.csv for the R-7 draft. Scratch."""

import csv
from collections import defaultdict

rows = list(csv.DictReader(open("scratch/r7/r7_margins.csv")))
for r in rows:
    for k in ("nx", "ny", "s"):
        r[k] = int(r[k])
    for k in ("m_up", "m_low", "m_high", "e_sur", "Q_sur", "Q_cont", "tau_R", "gamma_allow_over_tauR", "R_f", "R_now"):
        r[k] = float(r[k])
    r["coarse_lt20"] = r["coarse_lt20"] == "True"

ORDER = ["0,2", "1/3", "1/2", "2/3", "1", "1,4", "1,5", "2", "3", "5"]
print("grid counts:")
total = 0
for lab in ORDER:
    fam = defaultdict(set)
    for r in rows:
        if r["r"] == lab:
            fam[r["family"]].add((r["nx"], r["ny"]))
    coarse = sum(1 for p in fam["II"] if min(p) < 20)
    total += len(fam["I"]) + len(fam["II"])
    print(f"  r={lab}: I {len(fam['I'])}, II {len(fam['II'])} (coarse <20: {coarse}); II range {min(fam['II'])}..{max(fam['II'])}")
print("  total grid pairs:", total, " total case rows (s=2,3,4):", len(rows))

print("\ncategory boundaries (by max(nx,ny)); contiguity check:")
for lab in ORDER:
    for fam in ("I", "II"):
        for s in (2, 3):
            sel = sorted((r for r in rows if r["r"] == lab and r["family"] == fam and r["s"] == s), key=lambda r: max(r["nx"], r["ny"]))
            cats = [r["category"] for r in sel]
            # contiguity: categories should change at most along MANDATORY -> NO_HEADROOM -> BORDERLINE -> UNRESOLVED (ignoring coarse)
            rank = {"MANDATORY_CANDIDATE": 0, "RESOLVABLE_NO_HEADROOM": 1, "BORDERLINE": 2, "EXPECTED_UNRESOLVED": 3}
            fine = [rank[r["category"]] for r in sel if not r["coarse_lt20"]]
            monotone = all(x <= y for x, y in zip(fine, fine[1:]))
            mono_m = all(a["m_up"] >= b["m_up"] for a, b in zip(sel, sel[1:]))
            def last(cat):
                c = [max(r["nx"], r["ny"]) for r in sel if r["category"] == cat]
                return (min(c), max(c)) if c else None
            print(f"  r={lab:>3} fam {fam} s={s}: MAND {last('MANDATORY_CANDIDATE')}, NOHEAD {last('RESOLVABLE_NO_HEADROOM')}, "
                  f"BORDER {last('BORDERLINE')}, UNRES {last('EXPECTED_UNRESOLVED')}; categories monotone {monotone}, m_up monotone {mono_m}")

print("\nfamily I margins m_up at key levels (s=2 | s=3), e_sur at s=3:")
for lab in ORDER:
    line = []
    for n in (20, 40, 80, 160, 320, 640):
        m2 = next(r for r in rows if r["r"] == lab and r["family"] == "I" and r["nx"] == n and r["s"] == 2)
        m3 = next(r for r in rows if r["r"] == lab and r["family"] == "I" and r["nx"] == n and r["s"] == 3)
        line.append(f"{n}: {m2['m_up']:.3g} | {m3['m_up']:.3g} ({m3['e_sur']:.1e})")
    print(f"  r={lab}: " + "; ".join(line))

print("\nfamily II margins s=3 at selected pairs:")
for lab in ORDER:
    sel = sorted((r for r in rows if r["r"] == lab and r["family"] == "II" and r["s"] == 3), key=lambda r: max(r["nx"], r["ny"]))
    pick = [sel[0], sel[len(sel) // 4], sel[len(sel) // 2], sel[3 * len(sel) // 4], sel[-1]]
    print(f"  r={lab}: " + "; ".join(f"({p['nx']},{p['ny']}): {p['m_up']:.3g}" for p in pick))

print("\nminimum s=2 margin over all non-coarse grids:", min(r["m_up"] for r in rows if r["s"] == 2 and not r["coarse_lt20"]))
print("minimum s=2 margin over all grids:", min(r["m_up"] for r in rows if r["s"] == 2))
print("max e_sur over non-coarse grids, s=2:", max(r["e_sur"] for r in rows if r["s"] == 2 and not r["coarse_lt20"]))
print("max e_sur over non-coarse grids, s=3:", max(r["e_sur"] for r in rows if r["s"] == 3 and not r["coarse_lt20"]))
print("min gamma_allow/tau_R, s=2 non-coarse:", min(r["gamma_allow_over_tauR"] for r in rows if r["s"] == 2 and not r["coarse_lt20"]))
sens = [r for r in rows if r["s"] in (2, 3) and r["category"] == "BORDERLINE"]
print("BORDERLINE cases:", len(sens))
for r in sens:
    print(f"   r={r['r']} {r['family']} ({r['nx']},{r['ny']}) s={r['s']} m_up={r['m_up']:.4f} e={r['e_sur']:.2e}")
near4 = [r for r in rows if r["s"] in (2, 3) and not r["coarse_lt20"] and r["m_up"] * (1 - r["e_sur"]) < 4 <= r["m_up"] * (1 + r["e_sur"])]
print("cases where e_sur straddles the factor 4:", len(near4))
for r in near4:
    print(f"   r={r['r']} {r['family']} ({r['nx']},{r['ny']}) s={r['s']} m_up={r['m_up']:.4f} e={r['e_sur']:.2e}")
