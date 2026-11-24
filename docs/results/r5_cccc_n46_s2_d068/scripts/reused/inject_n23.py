"""QA scratch driver: the real kkvim (CCCC N = 2-3) with ONE fault injection and a call log (env QA_INJECT, QA_EVENTS, QA_MODE=api|cli).
Production code is not edited; patches are module attributes set in this process only.

QA_MODE=api: argv = a b nx ny N half_steps q   (D = 1); writes a JSON summary of run_cccc_n23 to QA_EVENTS.
QA_MODE=cli: argv = the CLI arguments; runs kkvim.cli.main and writes the call log to QA_EVENTS.
"""
import json
import math
import os
import sys

import numpy as np

import kkvim.cccc_n23 as cn
import kkvim.cccc_n23_certificate as cert

mode, inject, events_path = os.environ.get("QA_MODE", "api"), os.environ.get("QA_INJECT", ""), os.environ["QA_EVENTS"]
events, count = [], {}


def tick(name):
    count[name] = count.get(name, 0) + 1
    return count[name]


real = {"qr": cn.orthonormalise, "rn5": cert.rn5_certificate, "rt7": cert.rt7_conditions, "bc": cn.solve_bc, "rank": cn.exact_rank}


def qr(T, h):
    events.append(["qr", list(T.shape)])
    return real["qr"](T, h)


def rn5(*a, **k):
    events.append(["rn5", float(a[-1])])  # the q passed to RN-5
    out = real["rn5"](*a, **k)
    if inject == "rn5_refuse":
        out = {**out, "certified": False, "verdict": "RN5_UNRESOLVED", "reason": "QA injected refusal", "eps": None, "sigma_low": None}
    elif inject == "rn5_float_eps":
        out = {**out, "eps": float(out["eps"]), "sigma_low": float(out["sigma_low"])}
    elif inject == "rn5_missing_eps":
        out = {**out, "eps": None, "sigma_low": None}
    events.append(["rn5_result", out.get("verdict"), out.get("certified")])
    return out


def rt7(*a, **k):
    events.append(["rt7"])
    out = real["rt7"](*a, **k)
    if inject == "rt7_refuse":
        out = {**out, "passed": False, "R7-3": False, "reason": "QA injected RT-7 refusal"}
    events.append(["rt7_result", out.get("passed"), out.get("reason")])
    return out


def bc(step, A, B, C, F, D, l_s, n_s):
    n = tick("bc")
    events.append(["solve_bc", step.s])
    if n == 2 and inject == "nonfinite_s2":
        return np.full((n_s - 1, A.shape[0]), math.nan)
    return real["bc"](step, A, B, C, F, D, l_s, n_s)


def rank(M):
    n = tick("rank")
    if inject == "dep" and n == 3:  # T, G, then V of s = 1
        events.append(["rank_injected", list(np.shape(M))])
        return M.shape[1] - 1
    return real["rank"](M)


cn.orthonormalise, cn.solve_bc, cn.exact_rank = qr, bc, rank
cert.rn5_certificate, cert.rt7_conditions = rn5, rt7


def dump(extra):
    with open(events_path, "w", encoding="utf-8") as f:
        json.dump({"inject": inject, "events": events, **extra}, f, ensure_ascii=False, indent=1, default=str)


if mode == "api":
    a, b, nx, ny, N, hs, q = (float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]),
                              int(sys.argv[6]), float(sys.argv[7]))
    r = cn.run_cccc_n23(a, b, 1.0, q, nx, ny, N, hs)
    steps = [{"s": s.s, "status": s.status, "done": s.done, "qr_executed": s.qr_executed, "G_none": s.G is None, "R_none": s.R is None,
              "A_none": s.A is None, "V_none": s.V is None, "eta_none": s.eta is None,
              "rn5": None if s.rn5 is None else s.rn5.get("verdict"), "rt7": None if s.rt7 is None else s.rt7.get("passed"),
              "reason": s.reason} for s in r.steps]
    w = r.field()
    first = r.steps[0] if r.steps else None
    field_from = None
    if w is not None and first is not None and first.done:
        if np.array_equal(w[1:-1, 1:-1], first.G @ first.V.T):
            field_from = "s1"
        elif len(r.steps) > 1 and r.steps[1].done and np.array_equal(w[1:-1, 1:-1], r.steps[1].V @ r.steps[1].G.T):
            field_from = "s2"
        else:
            field_from = "other"
    rejected_used = None
    if len(r.steps) > 1 and r.steps[1].G is not None and not r.steps[1].done and w is not None:
        # the rejected G2 must not be a factor of the field: the field equals s1, and the factors are (G1, V1)
        fac = r.factors()
        rejected_used = not (np.array_equal(fac[0], first.G) and np.array_equal(fac[1], first.V))
    dump({"status": r.status, "reason": r.reason, "completed": r.half_steps_completed, "steps": steps, "field_from": field_from,
          "rejected_G2_used": rejected_used, "field_sum": None if w is None else float(np.sum(w))})
else:
    from kkvim.cli import main
    code = main(sys.argv[1:])
    dump({"exit": code})
    sys.exit(code)
