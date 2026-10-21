"""P-12 of R3: Navier against the full Levy series, deflection only, by the rule of D-038 (QA).

For every control point P of the fixed set of D-038, item 3,

    |w_N(P) - w_L(P)| <= B_N + B_L + R_N(P) + R_L,

all in q a^4 / D with the original side a (Р-3):
- w_N: NavierRectangle(a, b, q=1, D=1, nu=0.3), M_N = 2559 (D-037); grid nodes by at_nodes(40, 40), other points
  by at_points; w_N = w_coef;
- w_L: levy_series_w_grid (nodes) / levy_series_w (points), M_L = 1131 (D-038), divided by q a^4 / D;
- B_N: strict ЧВ-R2 0.0021881 r^2 / M_N^2 (w_tail_bound_coef); B_L: strict ЧВ-Л4 (4/pi^5)/(8 M_L^4) (a_L/a)^4
  (levy_field_bounds(...).truncation / (q a^4 / D));
- R_N: rounding model of Navier for the method used (rounding_w of at_nodes or of at_points);
  R_L: rounding model of Levy (levy_field_bounds(...).rounding_model / (q a^4 / D)). Models, not strict bounds.

Any exceedance triggers an independent recomputation of BOTH truncated series at the same M and point with
mpmath (HP_DPS >= 40 digits; D-038, item 4): FAIL if |Delta_hp| > B_N + B_L, otherwise UNRESOLVED; without the
recomputation (mpmath unavailable) - UNRESOLVED. The maximal |w_N - w_L| is a diagnostic only (D-038, item 5).

Reproduction (from the repository root):
    uv run python scripts/r3_p12_study.py                       # writes docs/results/r3/p12 and the report blocks
    uv run --with mpmath python scripts/r3_p12_study.py --hp-min-margin   # also the optional HP diagnostic
"""

import argparse
import csv
import json
import math
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from kkvim_reference.levy import levy_field_bounds, levy_series_w, levy_series_w_grid, levy_orientation
from kkvim_reference.navier import NavierRectangle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "results" / "r3" / "p12"
REPORT = ROOT / "docs" / "R3_P12_REPORT.md"

# Fixed by D-038, item 3 (not chosen from data).
PLATES = (("0,2", 1.0, 0.2), ("1/3", 3.0, 1.0), ("1/2", 2.0, 1.0), ("2/3", 3.0, 2.0), ("1", 1.0, 1.0),
          ("1,4", 1.0, 1.4), ("1,5", 2.0, 3.0), ("2", 1.0, 2.0), ("3", 1.0, 3.0), ("5", 1.0, 5.0))
M_N = 2559
M_L = 1131
Q = 1.0
D = 1.0
NU = 0.3
N_GRID = 40
EDGE_DISTANCES = (1e-3, 1e-6, 1e-9)
TRANSVERSE = (0.25, 0.5, 0.75)
HP_DPS = 45

POINT_FIELDS = ("plate", "a", "b", "kind", "label", "i", "j", "x", "y", "w_N", "w_L", "diff", "abs_diff",
                "B_N", "B_L", "R_N", "R_L", "bound", "margin", "ok")
PLATE_FIELDS = ("plate", "a", "b", "r", "a_L", "n_points", "n_grid", "n_edge", "n_centre", "n_ok", "n_exceed",
                "max_abs_diff", "x_max_diff", "y_max_diff", "min_margin", "x_min_margin", "y_min_margin",
                "kind_min_margin", "B_N", "B_L", "R_N_nodes", "R_N_points", "R_L", "verdict")


def edge_points(a, b):
    """36 points near the edges and the centre (D-038, item 3), as (label, x, y)."""
    pts = []
    for d in EDGE_DISTANCES:
        for name, x in ((f"x={d:g}a", d * a), (f"x=a-{d:g}a", a - d * a)):
            for t in TRANSVERSE:
                pts.append((f"{name},y={t:g}b", x, t * b))
    for d in EDGE_DISTANCES:
        for name, y in ((f"y={d:g}b", d * b), (f"y=b-{d:g}b", b - d * b)):
            for t in TRANSVERSE:
                pts.append((f"{name},x={t:g}a", t * a, y))
    pts.append(("centre", a / 2, b / 2))
    return pts


def compute_plate(name, a, b):
    """All comparisons of one plate: list of point rows (dicts) and the plate scalars."""
    scale = Q * a**4 / D  # q a^4 / D of the original side a
    nav = NavierRectangle(a=a, b=b, q=Q, D=D, nu=NU)
    lb = levy_field_bounds(a=a, b=b, q=Q, D=D, M=M_L)
    B_L = lb.truncation / scale
    R_L = lb.rounding_model / scale

    rows = []
    grid = nav.at_nodes(N_GRID, N_GRID, M=M_N, diagnostics=False)
    wl_grid = levy_series_w_grid(grid.x, grid.y, a=a, b=b, q=Q, D=D, M=M_L) / scale
    B_N = grid.w_tail_bound_coef
    for i in range(N_GRID + 1):
        for j in range(N_GRID + 1):
            rows.append(_row(name, a, b, "grid", f"i={i},j={j}", i, j, grid.x[i], grid.y[j], grid.w_coef[i, j],
                             wl_grid[i, j], B_N, B_L, grid.rounding_w, R_L))
    pts = edge_points(a, b)
    xs = np.array([p[1] for p in pts])
    ys = np.array([p[2] for p in pts])
    res = nav.at_points(xs, ys, M=M_N, diagnostics=False)
    wl_pts = levy_series_w(xs, ys, a=a, b=b, q=Q, D=D, M=M_L) / scale
    for k, (label, x, y) in enumerate(pts):
        kind = "centre" if label == "centre" else "edge"
        rows.append(_row(name, a, b, kind, label, "", "", x, y, res.w_coef[k], wl_pts[k], res.w_tail_bound_coef, B_L,
                         res.rounding_w, R_L))
    scalars = {"B_N": B_N, "B_L": B_L, "R_N_nodes": grid.rounding_w, "R_N_points": res.rounding_w, "R_L": R_L,
               "a_L": levy_orientation(a, b).a_L, "r": nav.r}
    return rows, scalars


def _row(name, a, b, kind, label, i, j, x, y, wn, wl, B_N, B_L, R_N, R_L):
    wn, wl = float(wn), float(wl)
    diff = wn - wl
    bound = B_N + B_L + R_N + R_L
    margin = math.inf if diff == 0.0 else bound / abs(diff)
    return {"plate": name, "a": a, "b": b, "kind": kind, "label": label, "i": i, "j": j, "x": float(x), "y": float(y),
            "w_N": wn, "w_L": wl, "diff": diff, "abs_diff": abs(diff), "B_N": B_N, "B_L": B_L, "R_N": R_N, "R_L": R_L,
            "bound": bound, "margin": margin, "ok": abs(diff) <= bound}


# --- independent high-precision recomputation (D-038, item 4) ----------------------------------------------------

def hp_recompute(a, b, x, y):
    """Both truncated series at the same M and point with mpmath (HP_DPS digits), in q a^4 / D.

    Navier: (16/pi^6) sum_{m,n odd <= M_N} sin(m pi x/a) sin(n pi y/b) / (m n (m^2 + n^2/r^2)^2), r = b/a exactly
    from the given floats. Levy: (139) of L-107 in its original form (ch, sh) along the shorter side, M_L terms,
    times (a_L/a)^4. Returns (w_N_hp, w_L_hp) as mpmath numbers.
    """
    import mpmath as mp  # optional dependency of this check only

    mp.mp.dps = HP_DPS
    A, B, X, Y = (mp.mpf(v) for v in (a, b, x, y))
    r2 = (B / A) ** 2
    sm = [mp.sin(m * mp.pi * X / A) / m for m in range(1, M_N + 1, 2)]
    sn = [mp.sin(n * mp.pi * Y / B) / n for n in range(1, M_N + 1, 2)]
    total = mp.mpf(0)
    for jm, m in enumerate(range(1, M_N + 1, 2)):
        m2 = m * m
        inner = mp.fsum(sn[jn] / (m2 + n * n / r2) ** 2 for jn, n in enumerate(range(1, M_N + 1, 2)))
        total += sm[jm] * inner
    w_n = 16 / mp.pi**6 * total
    if B < A:
        aL, bL, xL, eta = B, A, Y, X - A / 2
    else:
        aL, bL, xL, eta = A, B, X, Y - B / 2
    s = mp.mpf(0)
    for m in range(1, M_L + 1, 2):
        al = m * mp.pi * bL / (2 * aL)
        ch = mp.cosh(al)
        P = (al * mp.tanh(al) + 2) / (2 * ch)
        u = 2 * al * eta / bL
        T = 1 - P * mp.cosh(u) + al / (2 * ch) * (2 * eta / bL) * mp.sinh(u)
        s += T * mp.sin(m * mp.pi * xL / aL) / mp.mpf(m) ** 5
    w_l = 4 / mp.pi**5 * (aL / A) ** 4 * s
    return w_n, w_l


def hp_check(row):
    """HP recomputation of one row; returns a dict for hp_rechecks.csv (Delta_hp against B_N + B_L)."""
    try:
        w_n, w_l = hp_recompute(row["a"], row["b"], row["x"], row["y"])
    except ImportError:
        return None
    import mpmath as mp

    d = w_n - w_l
    return {"plate": row["plate"], "kind": row["kind"], "label": row["label"], "x": row["x"], "y": row["y"],
            "dps": HP_DPS, "w_N_hp": mp.nstr(w_n, 30), "w_L_hp": mp.nstr(w_l, 30), "diff_hp": mp.nstr(d, 20),
            "w_N_double_err": float(abs(mp.mpf(row["w_N"]) - w_n)), "w_L_double_err": float(abs(mp.mpf(row["w_L"]) - w_l)),
            "abs_diff_double": row["abs_diff"], "B_N_plus_B_L": row["B_N"] + row["B_L"],
            "truncation_ok": float(abs(d)) <= row["B_N"] + row["B_L"]}


def plate_verdict(exceed, hp_exceed):
    """Verdict of one plate by D-038, item 4: exceedances and their HP recomputations (None - not possible)."""
    if not exceed:
        return "PASS", ""
    if any(h is None for h in hp_exceed):
        return "UNRESOLVED", "exceedance without high-precision recomputation (mpmath unavailable)"
    if any(not h["truncation_ok"] for h in hp_exceed):
        return "FAIL", "high precision confirms |Delta_hp| > B_N + B_L"
    return "UNRESOLVED", "exceedance in double only; |Delta_hp| <= B_N + B_L (rounding models insufficient)"


# --- run, files, report ----------------------------------------------------------------------------------------

def _fmt(v):
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        return "inf" if math.isinf(v) else repr(v)
    return str(v)


def run(out_dir=OUT, report=REPORT, hp_min_margin=False):
    # Provenance belongs to the CLEAN INPUT tree, not to the output files being
    # written. Checking git status after producing CSV/JSON would itself turn
    # a clean reproduction into dirty=true (D-038, P-12).
    source_commit = _git("rev-parse", "HEAD")
    source_dirty = bool(_git("status", "--porcelain"))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows, plates, hp_rows = [], [], []
    for name, a, b in PLATES:
        rows, sc = compute_plate(name, a, b)
        all_rows += rows
        exceed = [r for r in rows if not r["ok"]]
        hp_exceed = [hp_check(r) for r in exceed]
        verdict, reason = plate_verdict(exceed, hp_exceed)
        for h in hp_exceed:
            if h is not None:
                h["role"] = "exceedance"
                hp_rows.append(h)
        worst = max(rows, key=lambda r: r["abs_diff"])
        tight = min(rows, key=lambda r: r["margin"])
        if hp_min_margin:
            h = hp_check(tight)
            if h is not None:
                h["role"] = "diagnostic: point of minimal margin"
                hp_rows.append(h)
        plates.append({"plate": name, "a": a, "b": b, "r": sc["r"], "a_L": sc["a_L"], "n_points": len(rows),
                       "n_grid": sum(r["kind"] == "grid" for r in rows), "n_edge": sum(r["kind"] == "edge" for r in rows),
                       "n_centre": sum(r["kind"] == "centre" for r in rows), "n_ok": sum(r["ok"] for r in rows),
                       "n_exceed": len(exceed), "max_abs_diff": worst["abs_diff"], "x_max_diff": worst["x"],
                       "y_max_diff": worst["y"], "min_margin": tight["margin"], "x_min_margin": tight["x"],
                       "y_min_margin": tight["y"], "kind_min_margin": f"{tight['kind']}:{tight['label']}",
                       "B_N": sc["B_N"], "B_L": sc["B_L"], "R_N_nodes": sc["R_N_nodes"], "R_N_points": sc["R_N_points"],
                       "R_L": sc["R_L"], "verdict": verdict, "_reason": reason})
    verdicts = [p["verdict"] for p in plates]
    overall = "FAIL" if "FAIL" in verdicts else "UNRESOLVED" if "UNRESOLVED" in verdicts else "PASS"
    _write_csv(out_dir / "points.csv", POINT_FIELDS, all_rows)
    _write_csv(out_dir / "plates.csv", PLATE_FIELDS, plates)
    if hp_rows:
        _write_csv(out_dir / "hp_rechecks.csv", tuple(hp_rows[0].keys()), hp_rows)
    elif (out_dir / "hp_rechecks.csv").exists():  # no stale recomputations of an earlier run
        (out_dir / "hp_rechecks.csv").unlink()
    gmax = max(all_rows, key=lambda r: r["abs_diff"])
    gmin = min(all_rows, key=lambda r: r["margin"])
    meta = {
        "study": "P-12 of R3, Navier vs full Levy series, deflection (D-038)",
        "decision": "D-038 (accepted 2026-10-14); B_L confirmed by the Auditor (9919f32, VERIFIED_TRUNCATION)",
        "commit": source_commit, "dirty": source_dirty,
        "date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0], "numpy": np.__version__, "platform": platform.platform(),
        "M_N": M_N, "M_L": M_L, "q": Q, "D": D, "nu": NU, "grid": f"{N_GRID + 1}x{N_GRID + 1}",
        "edge_distances": list(EDGE_DISTANCES), "transverse": list(TRANSVERSE), "plates": [[n, a, b] for n, a, b in PLATES],
        "n_comparisons": len(all_rows), "n_exceed": sum(not r["ok"] for r in all_rows),
        "global_max_abs_diff": gmax["abs_diff"], "global_max_at": [gmax["plate"], gmax["kind"], gmax["label"]],
        "global_min_margin": gmin["margin"], "global_min_margin_at": [gmin["plate"], gmin["kind"], gmin["label"]],
        "hp_rechecks": len(hp_rows), "hp_dps": HP_DPS, "verdict": overall,
        "plate_verdicts": {p["plate"]: p["verdict"] for p in plates},
        "unresolved_or_fail_reasons": {p["plate"]: p["_reason"] for p in plates if p["_reason"]},
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if report is not None and Path(report).exists():
        inject_report(report, report_blocks(out_dir))
    return meta


def _write_csv(path, fields, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(fields)
        for r in rows:
            w.writerow([_fmt(r[k]) for k in fields])


def _git(*args):
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def sci(v, digits=3):
    """Russian scientific notation for the report tables (∞ for inf)."""
    v = float(v)
    if math.isinf(v):
        return "∞"
    if v == 0:
        return "0"
    e = math.floor(math.log10(abs(v)))
    m = f"{v / 10**e:.{digits - 1}f}".replace(".", ",")
    sup = str(e).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))
    return f"{m}·10{sup}"


def report_blocks(out_dir):
    """Markdown blocks of the report, built only from the CSV/JSON files of out_dir."""
    out_dir = Path(out_dir)
    with open(out_dir / "plates.csv", encoding="utf-8") as f:
        plates = list(csv.DictReader(f))
    meta = json.loads((out_dir / "metadata.json").read_text(encoding="utf-8"))
    head = ("| r = b/a | (a; b) | точек | превышений | max\\|w_N − w_L\\|, qa⁴/D | (x; y) max | мин. запас | точка мин. запаса "
            "| B_N | B_L | R_N узлы / точки | R_L | вердикт |")
    lines = [head, "|" + "---|" * 13]
    for p in plates:
        lines.append(
            f"| {p['plate']} | ({p['a']}; {p['b']}) | {p['n_points']} | {p['n_exceed']} | {sci(p['max_abs_diff'])} "
            f"| ({sci(p['x_max_diff'], 4)}; {sci(p['y_max_diff'], 4)}) | {sci(p['min_margin'], 4)} | {p['kind_min_margin']} "
            f"| {sci(p['B_N'])} | {sci(p['B_L'])} | {sci(p['R_N_nodes'], 2)} / {sci(p['R_N_points'], 2)} | {sci(p['R_L'], 2)} "
            f"| **{p['verdict']}** |")
    summary = [
        f"- сравнений: **{meta['n_comparisons']}** (10 пластин × (41·41 узлов + 36 приграничных точек + центр));",
        f"- превышений правила D-038: **{meta['n_exceed']}**; высокоточных пересчётов: {meta['hp_rechecks']};",
        f"- максимальная \\|w_N − w_L\\| (диагностика, не порог): **{sci(meta['global_max_abs_diff'])}** qa⁴/D — "
        f"пластина {meta['global_max_at'][0]}, {meta['global_max_at'][1]}:{meta['global_max_at'][2]};",
        f"- минимальный запас (B_N + B_L + R_N + R_L)/\\|w_N − w_L\\|: **{sci(meta['global_min_margin'], 4)}** — "
        f"пластина {meta['global_min_margin_at'][0]}, {meta['global_min_margin_at'][1]}:{meta['global_min_margin_at'][2]};",
        f"- **сводный вердикт П-12: {meta['verdict']}**.",
    ]
    return {"plates": "\n".join(lines), "summary": "\n".join(summary)}


_BLOCK = re.compile(r"<!-- AUTO:(\w+) -->\n(.*?)\n<!-- /AUTO:\1 -->", re.S)


def report_block_contents(path):
    return {m.group(1): m.group(2) for m in _BLOCK.finditer(Path(path).read_text(encoding="utf-8"))}


def inject_report(path, blocks):
    text = Path(path).read_text(encoding="utf-8")
    for name, body in blocks.items():
        text, n = re.subn(rf"(<!-- AUTO:{name} -->\n).*?(\n<!-- /AUTO:{name} -->)", lambda m: m.group(1) + body + m.group(2),
                          text, flags=re.S)
        if n != 1:
            raise ValueError(f"report block {name} not found exactly once")
    Path(path).write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--report", default=str(REPORT), help="report to update ('none' to skip)")
    parser.add_argument("--hp-min-margin", action="store_true", help="optional HP diagnostic at the minimal margin per plate")
    args = parser.parse_args()
    meta = run(args.out, None if args.report == "none" else args.report, args.hp_min_margin)
    print(json.dumps({k: meta[k] for k in ("n_comparisons", "n_exceed", "global_max_abs_diff", "global_min_margin",
                                           "verdict", "plate_verdicts")}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
