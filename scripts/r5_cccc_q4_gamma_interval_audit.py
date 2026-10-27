"""QA-gamma3 (R5-V Q4, priority 1 of D-053.4; docs/R5_CCCC_Q4_GAMMA_PROPOSAL.md 4.5, 5.3, 7): strict directed binary64 intervals of the
exact gamma_s of path B1 for the 984 half-steps of Q3, and Gamma_up = max(|gamma_down|, |gamma_up|) >= |gamma_s|. QA research
control tool; not a K_E calibration. No import of the working ``kkvim``; existing modules, the Q4 gamma audit and the Q3 data are used
read-only.

Path B1 (proposal 4.5; NOT a new definition): the stored binary64 factors of each half-step (given set U, companion V_prev, solved set
V) are taken exactly; A, B, C, F exactly (r5_cccc_halfstep.exact_coefficients); rho = F (x) 1 - K vec(V) exactly
(r5_cccc_halfstep.residual_exact, R-HP); Delta = V_prev - V exactly; gamma_s = h_s rho^T Delta with h_s = l_s/n_s exactly - the same
computation as r5_cccc_sequence._energy (column j_gamma_exact of Q3). The half-step arrays are reconstructed from the Q3 NPZ files by
the reader of the Q4 gamma audit (bitwise consistency checks kept).

Directed rounding (``directed``), for an exact rational x:
- |x| > DBL_MAX: x > 0 gives [DBL_MAX, +inf], x < 0 gives [-inf, -DBL_MAX];
- otherwise f = float(x) (CPython: correctly rounded quotient of integers, also into the subnormal range); if Fraction(f) == x the
  interval is [f, f]; if Fraction(f) < x it is [f, nextafter(f, +inf)], else [nextafter(f, -inf), f];
- EVERY result is checked by exact rational comparisons: Fraction(down) <= x <= Fraction(up), and, if x is not a double, down and up
  are ADJACENT doubles (nextafter(down, +inf) == up), so the interval is the tightest binary64 interval; the check does not rely on the
  correctness of float() (a wrong rounding would fail the check, never pass silently).
Gamma_up = max(|down|, |up|): from down <= x <= up, |x| <= max(|down|, |up|) (checked exactly as well).

Inputs that are not finite doubles are refused (no interval: UNRESOLVED); x = 0 gives [0.0, 0.0].

Usage (repository root):
    uv run python scripts/r5_cccc_q4_gamma_interval_audit.py run --profile full [--workers 8]
    uv run python scripts/r5_cccc_q4_gamma_interval_audit.py run --profile ci --out scratch/r5_cccc_q4_gamma_intervals_ci
"""

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import multiprocessing
import os
import platform
import subprocess
import sys
import tempfile
import time
from fractions import Fraction
from pathlib import Path

THREAD_VARS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
for _k in THREAD_VARS:
    os.environ[_k] = "1"

import numpy as np  # noqa: E402
import scipy  # noqa: E402

from kkvim_reference import r5_cccc_halfstep as hs  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "results" / "r5_cccc_q4_gamma_intervals"
GAMMA_AUDIT = ROOT / "scripts" / "r5_cccc_q4_gamma_audit.py"
GAMMA_AUDIT_META = ROOT / "docs" / "results" / "r5_cccc_q4_gamma_audit" / "meta.json"
GAMMA_AUDIT_ROWS = ROOT / "docs" / "results" / "r5_cccc_q4_gamma_audit" / "gamma_rows.csv"
DBL_MAX = 1.7976931348623157e308
MIN_NORMAL = 2.2250738585072014e-308
UNRESOLVED = "UNRESOLVED"


def _ga():
    """The Q4 gamma audit as a module (Q3 reader, control rule, energy path); read-only use."""
    if not hasattr(_ga, "mod"):
        spec = importlib.util.spec_from_file_location("r5_cccc_q4_gamma_audit", GAMMA_AUDIT)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _ga.mod = mod
    return _ga.mod


# ---------------------------------------------------------------------------------------------------------------------------------
# directed rounding of an exact rational to binary64
# ---------------------------------------------------------------------------------------------------------------------------------

def directed(x: Fraction) -> tuple[float, float]:
    """(down, up): the tightest binary64 interval with down <= x <= up (module docstring); verified exactly before it is returned."""
    if not isinstance(x, Fraction):
        raise TypeError("x must be an exact Fraction")
    if x > Fraction(DBL_MAX):
        down, up = DBL_MAX, math.inf
    elif x < -Fraction(DBL_MAX):
        down, up = -math.inf, -DBL_MAX
    else:
        f = float(x)
        if Fraction(f) == x:
            down = up = f
        elif Fraction(f) < x:
            down, up = f, math.nextafter(f, math.inf)
        else:
            down, up = math.nextafter(f, -math.inf), f
    check = verify(x, down, up)
    if check is not True:
        raise ArithmeticError(f"directed rounding failed the exact check: {check}")
    return down, up


def _as_fraction_bound(v: float) -> Fraction | None:
    return None if math.isinf(v) else Fraction(v)


def verify(x: Fraction, down: float, up: float):
    """Exact checks: down <= x <= up, and tightness (equal or adjacent doubles). True or a reason string."""
    if math.isnan(down) or math.isnan(up):
        return "NaN bound"
    lo, hi = _as_fraction_bound(down), _as_fraction_bound(up)
    if not ((lo is None and down < 0) or (lo is not None and lo <= x)):
        return "down > x"
    if not ((hi is None and up > 0) or (hi is not None and x <= hi)):
        return "x > up"
    if lo is not None and hi is not None:
        if lo == x == hi:
            return True
        if lo == x or hi == x:
            return "x is a double but the interval is not degenerate"
        if math.nextafter(down, math.inf) != up:
            return "down and up are not adjacent doubles"
    return True


def gamma_up(down: float, up: float) -> float:
    return max(abs(down), abs(up))


def gamma_up_holds(x: Fraction, g: float) -> bool:
    return math.isinf(g) or abs(x) <= Fraction(g)


# ---------------------------------------------------------------------------------------------------------------------------------
# path B1 on the stored arrays
# ---------------------------------------------------------------------------------------------------------------------------------

def gamma_b1(lx, nx, ly, ny, direction, given, companion, V, D=1.0, q=1.0) -> Fraction:
    """gamma_s = h_s rho^T Delta exactly: exact coefficients, R-HP residual, exact Delta (proposal 4.5, path B1)."""
    for a in (given, companion, V):
        if not np.all(np.isfinite(np.asarray(a, dtype=float))):
            raise ValueError("non-finite input: no exact gamma")
    p = hs.HalfStep(lx, nx, ly, ny, direction, given, D=D, q=q)
    coeffs = hs.exact_coefficients(p)
    r = hs.residual_exact(p, np.asarray(V, dtype=float).T.reshape(-1), coeffs)
    N, m = p.N, p.n_s - 1
    Delta = [Fraction(float(companion[j, i])) - Fraction(float(V[j, i])) for i in range(N) for j in range(m)]
    return Fraction(p.l_s) / p.n_s * sum((a * b for a, b in zip(r, Delta, strict=True)), Fraction(0))


def exact_record(x: Fraction) -> dict:
    """Verifiable representation of the exact rational: numerator and denominator (decimal), their bit lengths and a sha256."""
    text = f"{x.numerator}/{x.denominator}"
    return {"gamma_exact": text, "gamma_exact_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "gamma_exact_num_bits": x.numerator.bit_length(), "gamma_exact_den_bits": x.denominator.bit_length()}


def classify(x: Fraction) -> str:
    if x == 0:
        return "zero"
    a = abs(x)
    if a < Fraction(MIN_NORMAL):
        return "subnormal range"
    return "positive" if x > 0 else "negative"


def seq_job(spec: dict, rows: list[dict], npz_path: str, control_ids: list[str]) -> dict:
    ga = _ga()
    t0 = time.perf_counter()
    z = np.load(npz_path)
    steps = ga._steps_from_npz(z, spec)
    by_s = {int(r["j_s"]): r for r in rows}
    out = []
    for st in steps:
        row = by_s[st["s"]]
        rid = ga.row_id(row)
        rec = {"id": rid, "group": spec["group"], "grid": row["grid"], "N": spec["N"], "first": spec["first"], "s": st["s"],
               "direction": st["direction"], "given_consistent": st["given_consistent"]}
        try:
            g = gamma_b1(spec["lx"], spec["nx"], spec["ly"], spec["ny"], st["direction"], st["given"], st["companion"], st["V"])
        except ValueError as exc:
            rec.update({"status": f"{UNRESOLVED}: {exc}"})
            out.append(rec)
            continue
        down, up = directed(g)
        G = gamma_up(down, up)
        stored = ga.read_double(row["j_gamma_exact"])
        rec.update(exact_record(g))
        rec.update({"sign": classify(g), "gamma_down": down, "gamma_up": up, "Gamma_up": G,
                    "inclusion_exact": verify(g, down, up) is True, "Gamma_up_holds": gamma_up_holds(g, G),
                    "interval_degenerate": down == up,
                    "q3_gamma_exact_is_an_endpoint": stored in (down, up),
                    "q3_gamma_exact_is_nearest": stored == float(g)})
        # existing exact gamma of the Q4 gamma audit (path A, own implementation): exact equality
        (Gi,), EG = ga.dyadic(st["given"])
        (Vi, Ci), EV = ga.dyadic(st["V"], st["companion"])
        lf, nf, ls, ns = (spec["lx"], spec["nx"], spec["ly"], spec["ny"]) if st["direction"] == "Y" else \
            (spec["ly"], spec["ny"], spec["lx"], spec["nx"])
        gA = ga.gamma_from_coefficients(ga.coefficients_exact(Gi, EG, Fraction(lf) / nf, Fraction(1)), Vi, Ci, EV,
                                        Fraction(ls) / ns, Fraction(1))
        rec["equals_audit_path_A"] = gA == g
        if rid in control_ids:  # independent energy path (two-dimensional Pi_h^C, gamma = Q - dP) on the control subsample
            Pp, Ps, Qe = ga.energies_path_b(spec["lx"], spec["nx"], spec["ly"], spec["ny"], 1.0, 1.0, *st["prev"], *st["now"])
            rec["equals_energy_path"] = Qe - (Pp - Ps) == g
        rec["status"] = "PASS" if (rec["inclusion_exact"] and rec["Gamma_up_holds"] and rec["equals_audit_path_A"]
                                   and rec["q3_gamma_exact_is_an_endpoint"] and rec.get("equals_energy_path", True)
                                   and st["given_consistent"]) else "FAIL"
        out.append(rec)
    return {"rows": out, "npz": Path(npz_path).name, "npz_sha256": _sha(Path(npz_path)),
            "seconds": round(time.perf_counter() - t0, 2), "peak_mb": _ga()._q3().peak_mb()}


# ---------------------------------------------------------------------------------------------------------------------------------
# campaign
# ---------------------------------------------------------------------------------------------------------------------------------

def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=False).stdout.strip()


def fmt(v):
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, (float, np.floating)):
        return repr(float(v))
    return "" if v is None else v


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: fmt(r.get(k)) for k in keys})


def summarize(rows: list[dict]) -> dict:
    def cnt(k):
        return f"{sum(1 for r in rows if r.get(k) is True)}/{sum(1 for r in rows if k in r)}"

    out = {"rows": len(rows), "pass": sum(1 for r in rows if r.get("status") == "PASS"),
           "fail_ids": [r["id"] for r in rows if r.get("status") != "PASS"]}
    for k in ("inclusion_exact", "Gamma_up_holds", "equals_audit_path_A", "q3_gamma_exact_is_an_endpoint",
              "q3_gamma_exact_is_nearest", "equals_energy_path", "given_consistent", "interval_degenerate"):
        out[k] = cnt(k)
    out["by_sign"] = {s: sum(1 for r in rows if r.get("sign") == s) for s in ("positive", "negative", "zero", "subnormal range")}
    vals = [abs(Fraction(r["gamma_exact"].split("/")[0])) / Fraction(r["gamma_exact"].split("/")[1]) for r in rows if "gamma_exact" in r]
    out["min_abs_gamma"] = float(min(vals)) if vals else None
    out["max_abs_gamma"] = float(max(vals)) if vals else None
    rel = [(Fraction(r["Gamma_up"]) - v) / v for r, v in zip((r for r in rows if "gamma_exact" in r), vals, strict=True) if v]
    out["max_rel_excess_Gamma_up"] = float(max(rel)) if rel else None
    out["max_exact_digits"] = max((len(r["gamma_exact"]) for r in rows if "gamma_exact" in r), default=0)
    return out


def run_campaign(profile: str, out: Path, workers: int, arrays: Path) -> dict:
    ga = _ga()
    t0 = time.time()
    out.mkdir(parents=True, exist_ok=True)
    if profile == "ci":
        tmp = Path(tempfile.mkdtemp(prefix="r5q4gi_"))
        csv_path, arrays = ga.ci_data(tmp)
        rows = ga.read_q3_rows(csv_path)
        control = {ga.row_id(r) for r in rows}
        spec_of = ga._ci_spec
        prov = {"csv": "generated by the Q3 seq_job (CI profile)", "sha256": _sha(csv_path)}
    else:
        rows = ga.read_q3_rows(ga.Q3_CSV)
        control = ga.control_rule(rows)  # the fixed control subsample of the Q4 gamma audit (57 half-steps)
        spec_of = ga.spec_of
        prov = ga.provenance(ga.Q3_CSV)
        recorded = {e["file"]: e["sha256"] for e in json.loads(GAMMA_AUDIT_META.read_text(encoding="utf-8"))["npz"]}
        prov["npz_recorded_in_gamma_audit"] = len(recorded)
    groups = {}
    for r in rows:
        groups.setdefault(ga.seq_key(r), []).append(r)
    jobs = []
    q3 = ga._q3()
    for rs in groups.values():
        spec = spec_of(rs[0])
        npz = arrays / q3.seq_file(spec)
        if not npz.exists():
            raise FileNotFoundError(f"STOP: Q3 array missing: {npz} (inputs are not substituted)")
        jobs.append((spec, rs, str(npz), sorted(control)))
    jobs.sort(key=lambda j: -(j[0]["nx"] * j[0]["ny"] * j[0]["N"]))
    if workers > 1:
        with multiprocessing.get_context("spawn").Pool(workers) as pool:
            results = pool.starmap(seq_job, jobs, chunksize=1)
    else:
        results = [seq_job(*j) for j in jobs]
    res_rows = sorted((r for res in results for r in res["rows"]), key=lambda r: (r["group"], r["grid"], r["N"], r["first"], r["s"]))
    npz_ok = None
    if profile == "full":
        npz_ok = all(recorded.get(res["npz"]) == res["npz_sha256"] for res in results)
    write_csv(out / "gamma_intervals.csv", res_rows)
    summ = summarize(res_rows)
    summ["npz_sha256_equal_to_gamma_audit"] = npz_ok
    summ["verdict"] = "PASS" if summ["pass"] == summ["rows"] and (npz_ok is not False) else "FAIL/UNRESOLVED"
    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summ, f, indent=1, ensure_ascii=False)
        f.write("\n")
    meta = {"study": "QA-gamma3: directed binary64 intervals of the exact gamma_s (path B1) and Gamma_up for the Q3 half-steps; "
                     "research control, not a K_E calibration",
            "profile": profile, "git_commit": git("rev-parse", "HEAD"), "git_status_porcelain": git("status", "--porcelain").splitlines(),
            "python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__, "platform": platform.platform(),
            "machine": platform.machine(), "processor": platform.processor(), "threads": {k: os.environ.get(k) for k in THREAD_VARS},
            "workers": workers, "data": prov, "q3_arrays_dir": str(arrays) if profile == "full" else "temporary (CI)",
            "q3_arrays_used_read_only": True, "control_subsample": len(control),
            "npz": [{"file": r["npz"], "sha256": r["npz_sha256"], "seconds": r["seconds"], "peak_mb": r["peak_mb"]} for r in results],
            "seconds_total": round(time.time() - t0, 1)}
    with open(out / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1, ensure_ascii=False)
        f.write("\n")
    return {"summary": summ, "meta": meta}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--profile", choices=("full", "ci"), default="full")
    r.add_argument("--out", type=Path, default=None)
    r.add_argument("--workers", type=int, default=8)
    r.add_argument("--arrays", type=Path, default=None)
    a = ap.parse_args(argv)
    arrays = a.arrays or _ga().Q3_ARRAYS_DEFAULT
    out = a.out or (OUT if a.profile == "full" else ROOT / "scratch" / "r5_cccc_q4_gamma_intervals_ci")
    res = run_campaign(a.profile, out, a.workers if a.profile == "full" else 1, arrays)
    s = res["summary"]
    print(json.dumps({k: s[k] for k in ("rows", "pass", "verdict", "inclusion_exact", "Gamma_up_holds", "equals_audit_path_A",
                                         "q3_gamma_exact_is_an_endpoint", "equals_energy_path", "by_sign", "fail_ids")}))
    return 0 if s["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
