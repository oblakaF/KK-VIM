"""QA scratch driver: the real kkvim code with ONE fault injection and a call log (env QA_INJECT, QA_EVENTS, QA_MODE=api|cli).
Production code is not edited; patches are module attributes set in this process only.

QA_MODE=api: argv = a b nx ny N half_steps; writes a JSON summary of run_cccc_n46_physical to QA_EVENTS.
QA_MODE=cli: argv = the CLI arguments; runs kkvim.cli.main and writes the call log to QA_EVENTS.
"""
import json
import math
import os
import sys

import numpy as np

import kkvim.cccc_n23 as n23
import kkvim.cccc_n23_certificate as cert
import kkvim.cccc_n46 as n46

mode, inject, events_path = os.environ.get("QA_MODE", "api"), os.environ.get("QA_INJECT", ""), os.environ["QA_EVENTS"]
events = []
count = {}


def tick(name):
    count[name] = count.get(name, 0) + 1
    return count[name]


real = {"qr": n23.orthonormalise, "rn5": cert.rn5_certificate, "rt7": cert.rt7_conditions, "solve": n23._solve_given,
        "bc": n23.solve_bc, "rank": n23.exact_rank, "prem": n23.premise_failures}


def qr(T, h):
    events.append(["qr", list(T.shape)])
    return real["qr"](T, h)


def rn5(*a, **k):
    events.append(["rn5"])
    out = real["rn5"](*a, **k)
    if inject == "rn5_refuse":
        out = {**out, "certified": False, "verdict": "RN5_UNRESOLVED", "reason": "QA injected refusal", "eps": None, "sigma_low": None}
    elif inject == "rn5_missing_eps":
        out = {**out, "eps": None, "sigma_low": None}
    elif inject == "rn5_float_eps":
        out = {**out, "eps": float(out["eps"]), "sigma_low": float(out["sigma_low"])}
    elif inject == "rn5_empty":
        out = {}
    return out


def rt7(*a, **k):
    events.append(["rt7"])
    out = real["rt7"](*a, **k)
    if inject == "rt7_refuse":
        out = {**out, "passed": False, "R7-3": False, "reason": "QA injected RT-7 refusal"}
    return out


def solve(step, *a, **k):
    events.append(["solve_given", step.s])
    return real["solve"](step, *a, **k)


def bc(step, A, B, C, F, D, l_s, n_s):
    n = tick("bc")
    events.append(["solve_bc", step.s])
    if n == 2 and inject == "nonfinite_s2":
        return np.full((n_s - 1, A.shape[0]), math.nan)
    if n == 2 and inject == "factor_s2":
        return real["bc"](step, -A, B, C, F, D, l_s, n_s)
    return real["bc"](step, A, B, C, F, D, l_s, n_s)


def rank(M):
    n = tick("rank")
    if inject == "dep" and n == 3:  # T, G, then V of s = 1
        return M.shape[1] - 1
    return real["rank"](M)


def prem(parts, n_f):
    n = tick("prem")
    if inject == "premise_s2" and n == 2:
        return ["QA injected: P-2 at s = 2"]
    return real["prem"](parts, n_f)


n23.orthonormalise, cert.rn5_certificate, cert.rt7_conditions = qr, rn5, rt7
n23._solve_given, n23.solve_bc, n23.exact_rank, n23.premise_failures = solve, bc, rank, prem
if inject == "interrupt":
    def boom(*a, **k):
        raise KeyboardInterrupt
    n46.run_cccc_n46_physical = boom


def dump(extra):
    with open(events_path, "w", encoding="utf-8") as f:
        json.dump({"inject": inject, "events": events, **extra}, f, ensure_ascii=False, indent=1, default=str)


if mode == "api":
    a, b, nx, ny, N, hs = (float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]))
    try:
        r = n46.run_cccc_n46_physical(a, b, 0.01, 2.1e11, 0.3, 1000.0, nx, ny, N, hs)
    except Exception as exc:  # noqa: BLE001
        dump({"exception": f"{type(exc).__name__}: {exc}"})
        sys.exit(0)
    steps = [{"s": s.s, "status": s.status, "done": s.done, "qr_executed": s.qr_executed, "G_none": s.G is None, "A_none": s.A is None,
              "V_none": s.V is None, "rn5": None if s.rn5 is None else s.rn5.get("verdict"),
              "rt7": None if s.rt7 is None else s.rt7.get("passed"), "reason": s.reason} for s in r.steps]
    fac = r.factors()
    last = r.completed[-1] if r.completed else None
    same_as = None
    if fac is not None and r.completed:
        first = r.steps[0]
        same_as = "s1" if (np.array_equal(fac[0], first.G) and np.array_equal(fac[1], first.V)) else \
            ("s2" if len(r.steps) > 1 and r.steps[1].done and np.array_equal(fac[0], r.steps[1].V) and np.array_equal(fac[1], r.steps[1].G) else "other")
    dump({"status": r.status, "reason": r.reason, "completed": r.half_steps_completed, "steps": steps, "field_from": same_as,
          "last_completed_s": None if last is None else last.s})
else:
    from kkvim.cli import main
    code = main(sys.argv[1:])
    dump({"exit": code})
    sys.exit(code)
