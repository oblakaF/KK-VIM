"""QA scratch driver (R5-RN5-NEGATIVE-Q-FIX-P0A-INDEPENDENT): runs the REAL kkvim of a chosen tree and saves everything.

usage: python runner_negq.py <src-tree> <out-dir> <label>
  <src-tree>  the src directory to import kkvim from (the hotfix worktree src, or the git-archive of the parent)
  <out-dir>   where <label>_<case>.npz and <label>_runs.json are written

Production code is not edited; only module attributes of this process are wrapped to log the call order.
Cases: N in (2, 3) x geometry (a=1,b=0.6,10x9), (a=1,b=1.3,8x7) x q in (+1,-1,+1000,-1000); D = 1, half_steps = 2.
"""
import hashlib
import json
import sys

sys.set_int_max_str_digits(0)
from fractions import Fraction
from pathlib import Path

import numpy as np

tree, out_dir, label = Path(sys.argv[1]).resolve(), Path(sys.argv[2]), sys.argv[3]
sys.path.insert(0, str(tree))
import kkvim.cccc_n23 as cn  # noqa: E402
import kkvim.cccc_n23_certificate as cc  # noqa: E402

assert Path(cn.__file__).resolve().is_relative_to(tree), (cn.__file__, tree)
out_dir.mkdir(parents=True, exist_ok=True)

GEOM = [("G1", 1.0, 0.6, 10, 9), ("G2", 1.0, 1.3, 8, 7)]
QS = [1.0, -1.0, 1000.0, -1000.0]
CASES = [(f"N{N}_{g}_q{q:+g}", N, a, b, nx, ny, q) for N in (2, 3) for (g, a, b, nx, ny) in GEOM for q in QS]

calls = []
_real = {"orthonormalise": cn.orthonormalise, "solve_bc": cn.solve_bc,
         "rn5_certificate": cc.rn5_certificate, "rt7_conditions": cc.rt7_conditions}


def _wrap(name):
    def f(*a, **k):
        calls.append(name)
        return _real[name](*a, **k)
    return f


cn.orthonormalise, cn.solve_bc = _wrap("orthonormalise"), _wrap("solve_bc")
cn.certificate.rn5_certificate, cn.certificate.rt7_conditions = _wrap("rn5_certificate"), _wrap("rt7_conditions")


def js(v):
    if isinstance(v, Fraction):  # exact value by its SHA256 (canonical num/den), plus a float for reading
        return {"frac_sha256": hashlib.sha256(f"{v.numerator}/{v.denominator}".encode()).hexdigest(), "approx": float(v)}
    if isinstance(v, dict):
        return {str(k): js(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [js(x) for x in v]
    if isinstance(v, (np.floating, float)):
        return float(v).hex()
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, np.ndarray):
        return None
    return v


ARR = ("T", "G", "R", "A", "B", "C", "F", "V", "companion", "given")
SCAL = ("status", "reason", "done", "qr_executed", "exact_rank_T", "exact_rank_G", "V_exact_rank", "eta", "P_hat", "P_hat_previous_QR",
        "alpha_centre", "field_preservation", "premises_status", "rank_decision", "direction", "cond_S", "min_eig_DM", "sigma", "rho_N",
        "kappa_rep")
summary = {"tree": str(tree), "kkvim_file": cn.__file__, "numpy": np.__version__, "python": sys.version, "cases": {}}
for name, N, a, b, nx, ny, q in CASES:
    calls.clear()
    r = cn.run_cccc_n23(a, b, 1.0, q, nx, ny, N, 2)
    arrays = {}
    rec = {"N": N, "a": a, "b": b, "nx": nx, "ny": ny, "q": q, "status": r.status, "reason": r.reason,
           "completed": r.half_steps_completed, "mode": r.mode, "application_status": r.application_status,
           "rank_guard_rule": r.rank_guard_rule, "calls": list(calls), "steps": []}
    for st in r.steps:
        srec = {k: js(getattr(st, k)) for k in SCAL}
        srec["s"] = st.s
        srec["rn5"] = js(st.rn5)
        srec["rt7"] = js(st.rt7)
        rec["steps"].append(srec)
        for k in ARR:
            v = getattr(st, k)
            if v is not None:
                arrays[f"s{st.s}_{k}"] = np.asarray(v)
    w = r.field()
    if w is not None:
        arrays["field"] = w
        arrays["wbar"] = r.dimensionless_field()
    # the closed-form check of F on the enclosure used by RN-5 (kkvim's own tstar; observation, the independent path is separate)
    Tc, rho = cc.tstar(nx, N)
    la, qq = Fraction(a), Fraction(q)
    F_c = cc.coefficients(Tc, la, nx, qq)[3]
    dev = [abs(F_c[i] - qq * la / 2) for i in range(N)]
    hx, m = la / nx, nx - 1
    rec["F_check"] = {"max_dev": js(max(dev)), "bound_abs_q": js(abs(qq) * hx * m * rho), "bound_signed_q": js(qq * hx * m * rho),
                      "rho": js(rho), "new_ok": all(d <= abs(qq) * hx * m * rho for d in dev),
                      "old_ok": all(d <= qq * hx * m * rho for d in dev)}
    np.savez(out_dir / f"{label}_{name}.npz", **arrays)
    summary["cases"][name] = rec
    print(name, r.status, r.half_steps_completed, rec["calls"], flush=True)

(out_dir / f"{label}_runs.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
