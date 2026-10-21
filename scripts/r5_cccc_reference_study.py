"""R5-V CCCC: study of the QA references A (Hencky-Inglis), B (Ritz, L-120), D2 and the published data C (tables of the report).

Runs only kkvim_reference (no kkvim). Writes docs/results/r5_cccc_references/ (or --out):
- d2_external.csv   task T4: the 27 external raw values of RM 9.8 (UNVERIFIED) and r = 5, n = 256; both orientations, two
                    orderings; status by plan 1.5 (REPRODUCED_TO_PRINTED_DIGITS; within the ENGINEERING CANDIDATE B_ext, not
                    approved; NOT_REPRODUCED; UNRESOLVED if e_D2/alpha > 1e-13, candidate);
- d2_sequence.csv   D2 alpha on n = 32...256 (and 384, 512 for r <= 2), Richardson by triples with observed exponents and the
                    difference to A (grid error of D1 = D2): OBSERVATION (group B; H-D2-2, OV-M9);
- hencky.csv        A by N_Y = 50...800 for r = 1...20 and the H-osc points: OBSERVATION;
- ritz.csv          B by M, N; L-120 tables I, II at the same M: program check; centre OBSERVATION; work: Ritz lower bound;
- compare.csv       alpha and work of A, B, C, D2 per r: OBSERVATION (no tolerance from C);
- budgets.csv       own error estimates of the references and the rule 1/100 (D-049.6, item 4) in the same relative norm;
- meta.json         versions, platform, git commit, timing.

    uv run python scripts/r5_cccc_reference_study.py all          # about 10 min on the QA machine
    uv run python scripts/r5_cccc_reference_study.py d2-external hencky ritz compare budgets
"""

import argparse
import csv
import json
import math
import platform
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

from kkvim_reference import r5_cccc_d2 as d2
from kkvim_reference import r5_cccc_hencky as hk
from kkvim_reference import r5_cccc_ritz as rz
from kkvim_reference import r5_cccc_tables as tb

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "results" / "r5_cccc_references"
U = 2.0**-53
R_MAIN = (1, 2, 3, 5)

# RM 9.8: the 27 external raw values of the consultant (UNVERIFIED_EXTERNAL_NUMERICAL_EVIDENCE), printed strings
EXTERNAL = {
    1: {32: "1.2759787033576e-3", 48: "1.2700711562290e-3", 64: "1.2679950248813e-3", 96: "1.2665093207759e-3",
        128: "1.2659887771808e-3", 192: "1.2656167846025e-3", 256: "1.2654865531591e-3"},
    2: {32: "2.5518770709064e-3", 48: "2.5413696856572e-3", 64: "2.5376894721877e-3", 96: "2.5350599141738e-3",
        128: "2.5341394052183e-3", 192: "2.5334818417406e-3", 256: "2.5332516902792e-3"},
    3: {32: "2.6376772204295e-3", 48: "2.6263173160792e-3", 64: "2.6223412711801e-3", 96: "2.6195012156523e-3",
        128: "2.6185071923402e-3", 192: "2.6177971682155e-3", 256: "2.6175486667662e-3"},
    5: {32: "2.6245775748442e-3", 48: "2.6132718443996e-3", 64: "2.6093148563303e-3", 96: "2.6064884413949e-3",
        128: "2.6054991977135e-3", 192: "2.6047925893927e-3"},
}
EXTERNAL_EXTRAPOLATED = {1: 0.0012653190863, 2: 0.0025329557677, 3: 0.00261722915, 5: 0.0026042273}  # UNVERIFIED
E_D2_CANDIDATE = 1e-13  # plan 1.4, engineering candidate (D-049.6, item 2)
B_EXT_FACTOR = 10  # plan 1.5, engineering candidate (D-049.6, item 2)


def write(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0])
    for row in rows[1:]:
        keys += [k for k in row if k not in keys]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, lineterminator="\n")
        w.writeheader()
        for row in rows:
            w.writerow({k: (f"{v:.17g}" if isinstance(v, float) else v) for k, v in row.items()})


def printed(value: float, template: str) -> str:
    """value with the number of significant digits of the printed template ('1.2759787033576e-3' -> 14)."""
    mant = template.split("e")[0].replace(".", "").lstrip("0")
    return f"{value:.{len(mant) - 1}e}"


def same_printed(value: float, template: str) -> bool:
    return float(printed(value, template)) == float(template)


# --- D2 --------------------------------------------------------------------------------------------------------

def d2_point(r: int, n: int) -> dict:
    p = d2.D2Problem(float(r), 1.0, r * n, n)  # long side along x (RM 9.2)
    t0 = time.time()
    s_mmd = d2.solve(p, "MMD_AT_PLUS_A")
    s_col = d2.solve(p, "COLAMD")
    pt = d2.transpose_problem(p)
    s_t = d2.solve(pt, "MMD_AT_PLUS_A")
    s_tc = d2.solve(pt, "COLAMD")
    a_ref = s_mmd.alpha()
    return {"r": r, "n": n, "nx": p.nx, "ny": p.ny, "unknowns": p.size, "alpha_ref": a_ref,
            "alpha_ref_colamd": s_col.alpha(), "alpha_ref_transposed": s_t.alpha(),
            "alpha_unref_colamd": s_col.alpha(s_col.x_unrefined),
            "alpha_unref_colamd_transposed": s_tc.alpha(s_tc.x_unrefined),
            "e_D2": max(s_mmd.e_D2, s_col.e_D2), "e_D2_transposed": max(s_t.e_D2, s_tc.e_D2),
            "e_D2_rel": max(s_mmd.e_D2, s_col.e_D2) / a_ref,
            "orderings_diff": abs(s_mmd.alpha() - s_col.alpha()),
            "orderings_ok": abs(s_mmd.alpha() - s_col.alpha()) <= 4 * (s_mmd.e_D2 + s_col.e_D2) + 4 * U * a_ref,
            "orientation_diff": abs(a_ref - s_t.alpha()),
            "orientation_ok": abs(a_ref - s_t.alpha()) <= 4 * U * a_ref + s_mmd.e_D2 + s_t.e_D2,
            "refinement_steps": len(s_mmd.corrections), "work_h": s_mmd.work(), "seconds": round(time.time() - t0, 1)}


def d2_external() -> list[dict]:
    rows = []
    points = [(r, n) for r in EXTERNAL for n in EXTERNAL[r]] + [(5, 256)]
    for r, n in points:
        row = d2_point(r, n)
        ext = EXTERNAL.get(r, {}).get(n)
        if ext is None:
            row.update(external="not computed (new point)", status="NEW_VALUE (no external value)")
        else:
            a_ref, ext_v = row["alpha_ref"], float(ext)
            b_ext = 0.5e-13 * abs(a_ref) + B_EXT_FACTOR * abs(row["alpha_unref_colamd"] - a_ref) + row["e_D2"]
            diff = abs(a_ref - ext_v)
            if row["e_D2_rel"] > E_D2_CANDIDATE:
                status = "UNRESOLVED (e_D2/alpha > 1e-13, candidate)"
            elif same_printed(a_ref, ext):
                status = "REPRODUCED_TO_PRINTED_DIGITS"
            elif diff <= b_ext:
                status = "CONSISTENT_WITHIN_CANDIDATE_B_EXT (engineering candidate, not approved)"
            else:
                status = "NOT_REPRODUCED"
            row.update(external=ext, alpha_ref_printed=printed(a_ref, ext), abs_diff=diff, rel_diff=diff / a_ref,
                       B_ext=b_ext, diff_over_B_ext=diff / b_ext,
                       unref_vs_external=abs(row["alpha_unref_colamd"] - ext_v),
                       unref_transposed_vs_external=abs(row["alpha_unref_colamd_transposed"] - ext_v),
                       status=status)
        rows.append(row)
        print("D2", r, n, row["status"], flush=True)
    return rows


def richardson(a1, a2, a4):
    """Observed exponent and the h^2 and h^4 eliminations for a doubling triple (RM 9.8); OBSERVATION."""
    p = math.log2((a1 - a2) / (a2 - a4)) if (a1 - a2) * (a2 - a4) > 0 else float("nan")
    r1a = a2 + (a2 - a1) / 3
    r1b = a4 + (a4 - a2) / 3
    return p, r1b, r1b + (r1b - r1a) / 15


def d2_sequence(alpha_a: dict) -> list[dict]:
    rows = []
    for r in R_MAIN:
        ns = [32, 48, 64, 96, 128, 192, 256] + ([384, 512] if r <= 2 else [])
        values = {}
        for n in ns:
            s = d2.solve(d2.D2Problem(float(r), 1.0, r * n, n))
            values[n] = (s.alpha(), s.e_D2 / s.alpha())
            print("D2 seq", r, n, flush=True)
        for n in ns:
            a, e_rel = values[n]
            row = {"r": r, "n": n, "alpha": a, "e_D2_rel": e_rel, "alpha_minus_A": a - alpha_a[r],
                   "rel_grid_error_vs_A": (a - alpha_a[r]) / alpha_a[r], "status": "OBSERVATION (group B)"}
            if 2 * n in values:
                row["observed_order_vs_A_n_to_2n"] = math.log2((a - alpha_a[r]) / (values[2 * n][0] - alpha_a[r]))
            if 2 * n in values and 4 * n in values:
                p, r1, r2 = richardson(a, values[2 * n][0], values[4 * n][0])
                row.update(triple=f"({n}, {2 * n}, {4 * n})", observed_p=p, R1_h2=r1, R2_h2_h4=r2,
                           R2_minus_A=r2 - alpha_a[r])
            rows.append(row)
    return rows


# --- A, B --------------------------------------------------------------------------------------------------------

R_HENCKY = (1.0, 1.2, 1.4, 1.5, 1.6, 1.8, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0, 2.35, 2.40, 5.15, 5.20, 7.9, 8.0)


def hencky() -> list[dict]:
    rows = []
    for r in R_HENCKY:
        prev = None
        for NY in (50, 100, 200, 400, 800):
            s = hk.solve(hk.default_section(r, 1.0, NY))
            xs = [i / 20 * r for i in range(21)]
            import numpy as np  # noqa: PLC0415
            slope = float(np.max(np.abs(s.boundary_slope(np.array(xs)))))
            a = s.alpha()
            rows.append({"r": r, "NY": NY, "NX": s.problem.NX, "alpha": a, "work_norm": s.work_normalized(),
                         "delta_strip": a - tb.STRIP_LIMIT, "change_from_previous_N": None if prev is None else a - prev,
                         "row_sum_max": s.row_sum_max, "residual": s.residual, "edge_slope_max": slope,
                         "status": "OBSERVATION (no strict error bound, OV-M1...M3)"})
            prev = a
        print("A", r, flush=True)
    return rows


def ritz() -> list[dict]:
    rows = []
    for M in sorted(tb.L120_TABLE_I):
        s = rz.solve_smw(rz.RitzProblem(1.0, 1.0, M, M))
        w, E, kappa = tb.L120_TABLE_I[M]
        rows.append({"r": 1.0, "M": M, "N": M, "alpha": s.alpha(), "work_norm": s.work_normalized(),
                     "capacity_cond": s.capacity_cond, "L120_w_1e3": w, "L120_E_1e4": E, "L120_kappa": kappa,
                     "reproduces_L120": round(s.alpha() * 1e3, 9) == w and round(s.work_normalized() * 1e4, 9) == E
                     and (kappa is None or round(s.capacity_cond, 4) == kappa),
                     "status": "program check (same procedure, same M)"})
        print("B table I", M, flush=True)
    for r, E in tb.L120_TABLE_II_WORK.items():
        s = rz.solve_smw(rz.RitzProblem(1.0, r, 2000, 2000))
        pub = next(p.alpha for p in tb.ALPHA if p.source == tb.L120 and p.r == r)
        rows.append({"r": r, "M": 2000, "N": 2000, "alpha": s.alpha(), "work_norm": s.work_normalized(),
                     "capacity_cond": s.capacity_cond, "L120_w_1e3": pub * 1e3, "L120_E_1e4": E,
                     "reproduces_L120": round(s.alpha() * 1e3, 9) == round(pub * 1e3, 9)
                     and round(s.work_normalized() * 1e4, 8) == E,
                     "status": "program check (2000 x 2000 as printed)"})
        print("B table II", r, flush=True)
    for r in (2.0, 3.0, 5.0):  # long side along x: M ~ r N
        prev = None
        for N in (100, 200, 400, 800):
            M = round(r * N)
            if M * N > 4_000_000:
                break
            s = rz.solve_smw(rz.RitzProblem(r, 1.0, M, N))
            rows.append({"r": r, "M": M, "N": N, "alpha": s.alpha(), "work_norm": s.work_normalized(),
                         "capacity_cond": s.capacity_cond, "change_from_previous": None if prev is None else s.alpha() - prev,
                         "status": "OBSERVATION (centre); work is a Ritz lower bound"})
            prev = s.alpha()
        print("B", r, flush=True)
    return rows


# --- comparison and budgets ----------------------------------------------------------------------------------------

def compare(hencky_rows, ritz_rows, seq_rows) -> list[dict]:
    rows = []
    for r in sorted({p.r for p in tb.ALPHA} | set(map(float, R_MAIN))):
        a_rows = [x for x in hencky_rows if x["r"] == r]
        if not a_rows:
            continue
        a_best = a_rows[-1]
        b_rows = [x for x in ritz_rows if x["r"] == r]
        b_best = b_rows[-1] if b_rows else None
        d_rows = [x for x in seq_rows if float(x["r"]) == r]
        d_last = d_rows[-1] if d_rows else None
        d_r2 = [x for x in d_rows if "R2_h2_h4" in x]
        row = {"r": r, "alpha_A": a_best["alpha"], "A_NY": a_best["NY"], "A_last_change": a_best["change_from_previous_N"],
               "work_A": a_best["work_norm"]}
        if b_best:
            row.update(alpha_B=b_best["alpha"], B_MN=f"{b_best['M']}x{b_best['N']}", work_B=b_best["work_norm"],
                       A_minus_B=a_best["alpha"] - b_best["alpha"],
                       work_A_not_below_B=a_best["work_norm"] >= b_best["work_norm"] * (1 - 64 * U))
        for p in tb.alpha_values(r):
            key = p.source.split()[0]
            row[f"C_{key}"] = p.alpha
            row[f"A_minus_C_{key}_in_printed_steps"] = (a_best["alpha"] - p.alpha) / p.step
        if d_last:
            row.update(alpha_D2_finest=d_last["alpha"], D2_n=d_last["n"], D2_minus_A=d_last["alpha_minus_A"])
        if d_r2:
            row.update(D2_R2_last_triple=d_r2[-1]["R2_h2_h4"], D2_R2_minus_A=d_r2[-1]["R2_minus_A"])
        if int(r) == r and int(r) in EXTERNAL_EXTRAPOLATED:
            row.update(external_extrapolated=EXTERNAL_EXTRAPOLATED[int(r)],
                       external_extrapolated_minus_A=EXTERNAL_EXTRAPOLATED[int(r)] - a_best["alpha"])
        row["status"] = "OBSERVATION (group B); C is data, not a tolerance"
        rows.append(row)
    return rows


def budgets(ext_rows, seq_rows, hencky_rows, ritz_rows) -> list[dict]:
    rows = []
    e_rel = max([x["e_D2_rel"] for x in ext_rows] + [x["e_D2_rel"] for x in seq_rows])
    rows.append({"reference": "D2 (discrete problem D1 = D2)", "quantity": "alpha, field",
                 "own_error_estimate_rel": e_rel, "kind": "last refinement correction (estimate, not a strict bound)",
                 "smallest_measurable_rel_error_by_1_100": 100 * e_rel,
                 "use": "discrete errors (KK-VIM - D1 on the same grid) above this level; the grid error relative to omega_0 "
                        "is not measured by D2", "status": "estimate; plan candidate e_D2/alpha <= 1e-13 met"
                 if e_rel <= E_D2_CANDIDATE else "UNRESOLVED"})
    a_changes = [abs(x["change_from_previous_N"]) / abs(x["alpha"]) for x in hencky_rows
                 if x["NY"] == 800 and x["change_from_previous_N"] is not None]
    rows.append({"reference": "A (Hencky-Inglis)", "quantity": "alpha, work, field",
                 "own_error_estimate_rel": max(a_changes), "kind": "change N_Y 400 -> 800 (OBSERVATION; no strict bound)",
                 "smallest_measurable_rel_error_by_1_100": 100 * max(a_changes),
                 "use": "comparisons with A as a reference", "status": "UNRESOLVED (OV-M1...M3: no strict error bound)"})
    last_b = {}
    for x in ritz_rows:  # the change at the LAST refinement of every M, N sequence (r = 2, 3, 5)
        if x.get("change_from_previous") is not None:
            last_b[x["r"]] = abs(x["change_from_previous"]) / abs(x["alpha"])
    b_changes = list(last_b.values())
    rows.append({"reference": "B (Ritz)", "quantity": "alpha (centre)",
                 "own_error_estimate_rel": max(b_changes) if b_changes else None,
                 "kind": "change at the last refinement of M, N (OBSERVATION)",
                 "smallest_measurable_rel_error_by_1_100": 100 * max(b_changes) if b_changes else None,
                 "use": "comparisons with B as a reference", "status": "UNRESOLVED for the centre (no strict bound)"})
    rows.append({"reference": "B (Ritz)", "quantity": "load work", "own_error_estimate_rel": None,
                 "kind": "strict ONE-SIDED bound: Ritz work <= exact work", "smallest_measurable_rel_error_by_1_100": None,
                 "use": "detects a work below the bound", "status": "one-sided only; no two-sided budget"})
    for p in tb.ALPHA:
        if p.role == tb.CONTROL:
            rows.append({"reference": f"C {p.source.split()[0]} r={p.r}", "quantity": "alpha",
                         "own_error_estimate_rel": 0.5 * p.step / p.alpha, "kind": "half printed step",
                         "smallest_measurable_rel_error_by_1_100": 50 * p.step / p.alpha,
                         "use": "control of A, B, final numbers", "status": "data, not a tolerance"})
    return rows


def source_info() -> dict:
    info = {"python": platform.python_version(), "platform": platform.platform(), "processor": platform.processor()}
    for pkg in ("numpy", "scipy"):
        info[pkg] = metadata.version(pkg)
    try:
        info["git_commit"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                                            check=True).stdout.strip()
        info["git_dirty"] = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                                                capture_output=True, text=True, check=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        info["git_commit"], info["git_dirty"] = "unknown", None
    return info


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("parts", nargs="+", choices=("all", "d2-external", "d2-sequence", "hencky", "ritz", "compare",
                                                     "budgets"))
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args(argv)
    out = Path(args.out)
    parts = set(args.parts)
    if "all" in parts:
        parts = {"d2-external", "d2-sequence", "hencky", "ritz", "compare", "budgets"}
    t0 = time.time()
    meta = {"study": "R5-V CCCC references A, B, D2, C", "decision": "D-049 (2e4ffe1)", **source_info()}
    cache = {}

    def get(name, fn):
        if name not in cache:
            cache[name] = fn()
            write(out / f"{name}.csv", cache[name])
        return cache[name]

    need_h = parts & {"hencky", "d2-sequence", "compare", "budgets"}
    h_rows = get("hencky", hencky) if need_h else None
    alpha_a = {r: next(x["alpha"] for x in h_rows if x["r"] == float(r) and x["NY"] == 800) for r in R_MAIN} if h_rows else {}
    if parts & {"d2-external", "budgets"}:
        get("d2_external", d2_external)
    if parts & {"d2-sequence", "compare", "budgets"}:
        get("d2_sequence", lambda: d2_sequence(alpha_a))
    if parts & {"ritz", "compare", "budgets"}:
        get("ritz", ritz)
    if "compare" in parts:
        get("compare", lambda: compare(cache["hencky"], cache["ritz"], cache["d2_sequence"]))
    if "budgets" in parts:
        get("budgets", lambda: budgets(cache["d2_external"], cache["d2_sequence"], cache["hencky"], cache["ritz"]))
    meta["seconds"] = round(time.time() - t0)
    meta["parts"] = sorted(parts)
    (out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    print("done", meta["seconds"], "s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
