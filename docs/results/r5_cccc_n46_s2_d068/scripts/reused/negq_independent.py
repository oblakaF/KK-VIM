"""R5-RN5-NEGATIVE-Q-FIX-P0A-INDEPENDENT: independent verification (QA scratch, not in git). No import of kkvim / kkvim_reference.

Inputs: runs/hotfix_*.npz, runs/hotfix_runs.json, runs/parent_*.npz, runs/parent_runs.json (written by runner_negq.py from the real code).
Exact helpers (BBP pi, Taylor sin^2 enclosure, exact Gauss, characteristic polynomials) come from my earlier independent RN-5 module
scratch/rn5_n46/rn5_n46_independent.py (no kkvim import). That module carries the SAME sign assumption in its F check
(``q * hx * mx * rho``); here its RN-5 function is re-instantiated with ONE textual replacement of that check by a q-free form:
    F_c - F* = q (h 1^T Tc_i - a/2)  (exact factorisation, asserted), and  |h 1^T Tc_i - a/2| <= h m rho,
which is the derivation of section K1 below, not a copy of the hotfix.

Criteria (fixed before any comparison):
K1  For every (n_x, N, q) of the cases, with MY sin^2 enclosure: the exact identity h * sum_l T*_li = a/2 is consistent with the enclosure
    (|h 1^T Tc_i - a/2| <= h m rho), F_c - F* == q (h 1^T Tc_i - a/2) exactly, |F_c - F*| <= |q| h m rho; and for q < 0 the signed form
    q h m rho is < 0, so the old check rejects unconditionally.
K2  My RN-5 (q-free F check) certifies the saved s = 1 factors (G1, V1) of every one of the 16 hotfix runs; all its exact values
    (rho, norms, lambda_low, eps, sigma2_low) coincide between q and -q.
K3  Exact Galerkin solves with my operators (D4 = L^2 + 2/h^4 corner terms): s = 1 (given G1, unknown V1) and s = 2 (given G2, unknown X):
    ||x_hat - x*||_2 / ||x*||_2 <= 2 kappa_2(K) n u, kappa_2 from float64 eigenvalues of the exact K (estimate).
K4  q -> -q: even arrays (T, G1, R1, A1, B1, C1, G2, A2, B2, C2, wbar) bit-identical up to the sign of zero; odd arrays (V1, F1, R2,
    companion, given, F2, X, field) are exact negatives up to the sign of zero; every scalar of every step, the rn5 and rt7 records equal.
    A differing sign bit is counted only where both values are zero (IEEE-754 +0.0 / -0.0), never as a field discrepancy.
K5  P_hat(s) vs my exact Pi of the saved factors: |P_hat - Pi| <= 1e-12 |Pi| (observation threshold); Pi(X* G2^T) <= Pi(companion G2^T)
    exactly (Galerkin minimality in span G2); Pi < 0 at s = 1, 2.
K6  Positive q: hotfix arrays and records bit-identical to the parent 0551141 (all npz keys, all json fields except the tree path).
K7  No status CONVERGED / ACCEPTED / QUALIFIED (NOT_QUALIFIED is the required label) anywhere in the records.
"""
import hashlib
import json
import re
import sys
import time
from fractions import Fraction as Q
from pathlib import Path

import numpy as np

sys.set_int_max_str_digits(0)
HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
IND = HERE.parent / "rn5_n46" / "rn5_n46_independent.py"
IND_SHA = "b77a2f1e6024faee55aa54da1455b554883db5cc3d8bf77dba2faab7085b4603"
U = Q(1, 2**53)
assert hashlib.sha256(IND.read_bytes()).hexdigest() == IND_SHA, "my earlier independent module changed"
sys.path.insert(0, str(IND.parent))
import rn5_n46_independent as ind  # noqa: E402

assert not any(m == "kkvim" or m.startswith("kkvim") for m in sys.modules), "kkvim must not be imported"

# --- my RN-5 with the q-free F check (one textual replacement, asserted) ------------------------------------------------------------
_OLD = ("    for i in range(N):\n"
        "        if abs(Fc[i] - F_T[i]) > q * hx * mx * rho:\n"
        "            ok = False\n")
_NEW = ("    for i in range(N):\n"
        "        g = hx * sum(Tc[l][i] for l in range(mx)) - la / 2\n"
        "        assert Fc[i] - F_T[i] == q * g\n"
        "        if abs(g) > hx * mx * rho:\n"
        "            ok = False\n")
_src = IND.read_text(encoding="utf-8")
_start = _src.index("def rn5_independent(")
_end = _src.index("\ndef rayleigh_upper(")
_fn = _src[_start:_end]
assert _fn.count(_OLD) == 1
_ns = dict(vars(ind))
exec(_fn.replace(_OLD, _NEW), _ns)  # noqa: S102 - my own QA code, the replaced text is asserted above
rn5_signfree = _ns["rn5_independent"]


def fq(M):
    return [[Q(float(v)) for v in r] for r in np.atleast_2d(M)]


def frac_key(v):
    return hashlib.sha256(f"{v.numerator}/{v.denominator}".encode()).hexdigest() if isinstance(v, Q) else repr(v)


# --- K1 ------------------------------------------------------------------------------------------------------------------------------
def k1(a, nx, N, q):
    la, qq = Q(a), Q(q)
    hx, m = la / nx, nx - 1
    Tc, rho = ind.tstar(nx, N)
    out = {"rho": float(rho)}
    ok = True
    worst = Q(0)
    for i in range(N):
        g = hx * sum(Tc[l][i] for l in range(m)) - la / 2
        Fc = qq * hx * sum(Tc[l][i] for l in range(m))
        dev = Fc - qq * la / 2
        ok &= dev == qq * g and abs(g) <= hx * m * rho and abs(dev) <= abs(qq) * hx * m * rho
        worst = max(worst, abs(dev))
    out.update(identity_and_bound=ok, max_abs_dev=float(worst), bound_abs_q=float(abs(qq) * hx * m * rho),
               signed_bound=float(qq * hx * m * rho))
    out["old_form_rejects"] = bool(qq * hx * m * rho < 0)  # then |F_c - F*| >= 0 > q h m rho for every i
    return out


# --- K3 ------------------------------------------------------------------------------------------------------------------------------
def galerkin(given, l_g, n_g, l_s, n_s, D, q):
    """K, f of D[A (x) D4_s - 2 B (x) L_s + C (x) I] vec = F (x) 1 (index p N + i) from the given set, my operators."""
    hg, hs = Q(l_g) / n_g, Q(l_s) / n_s
    Lg, D4g = ind.ops(n_g, hg)
    Ls, D4s = ind.ops(n_s, hs)
    A, B, C, F = ind.coeffs(fq(given), hg, Q(q), Lg, D4g)
    N, ms = len(A), n_s - 1
    K = [[Q(0)] * (ms * N) for _ in range(ms * N)]
    for p in range(ms):
        for r in range(max(0, p - 2), min(ms, p + 3)):
            for i in range(N):
                for j in range(N):
                    K[p * N + i][r * N + j] = Q(D) * (A[i][j] * D4s[p][r] - 2 * B[i][j] * Ls[p][r] + (C[i][j] if p == r else 0))
    f = [F[i] for p in range(ms) for i in range(N)]
    return K, f, (Lg, D4g, Ls, D4s)


def exact_check(given, solved, l_g, n_g, l_s, n_s, D, q):
    K, f, _ = galerkin(given, l_g, n_g, l_s, n_s, D, q)
    xs, det = ind.solve(K, [[v] for v in f])
    assert det != 0
    xs = [r[0] for r in xs]
    N = solved.shape[1]
    xh = [Q(float(solved[p, i])) for p in range(solved.shape[0]) for i in range(N)]
    err2 = sum((a - b) ** 2 for a, b in zip(xh, xs, strict=True))
    nrm2 = sum(b * b for b in xs)
    rel = float(ind.sqrt_up(err2 / nrm2))
    ev = np.linalg.eigvalsh(np.array([[float(v) for v in r] for r in K]))
    kappa = float(ev[-1] / ev[0])
    crit = 2 * kappa * len(xs) * float(U)
    return {"rel_err": rel, "kappa": kappa, "n": len(xs), "criterion": crit, "ok": rel <= crit and ev[0] > 0}, xs


# --- K5: exact energy of a field W (x rows, y cols, inner nodes) ------------------------------------------------------------------
def pi_exact(W, a, b, nx, ny, D, q):
    hx, hy = Q(a) / nx, Q(b) / ny
    Lx, D4x = ind.ops(nx, hx)
    Ly, D4y = ind.ops(ny, hy)
    Wt = ind.tp(W)
    t1 = sum(v * w for r1, r2 in zip(ind.mm(D4x, W), W) for v, w in zip(r1, r2))
    t2 = sum(v * w for r1, r2 in zip(ind.mm(ind.mm(Lx, W), Ly), W) for v, w in zip(r1, r2))
    t3 = sum(v * w for r1, r2 in zip(ind.mm(D4y, Wt), Wt) for v, w in zip(r1, r2))
    return Q(D) / 2 * hx * hy * (t1 + 2 * t2 + t3) - Q(q) * hx * hy * sum(v for r in W for v in r)


def field_q(Xf, Yf):
    return ind.mm(fq(Xf), ind.tp(fq(Yf)))


# --- K4 ------------------------------------------------------------------------------------------------------------------------------
EVEN = ("s1_T", "s1_G", "s1_R", "s1_A", "s1_B", "s1_C", "s2_G", "s2_A", "s2_B", "s2_C", "wbar")
ODD = ("s1_V", "s1_F", "s2_R", "s2_companion", "s2_given", "s2_F", "s2_V", "field")


def sign_compare(x, y, odd):
    """(equal up to the sign of zero, count of zero positions whose sign bit differs, count of any other bit difference)."""
    y2 = -y if odd else y
    bx, by = x.view(np.uint64), y2.view(np.uint64)
    diff = bx != by
    zero_both = (x == 0) & (y2 == 0)
    return bool(np.array_equal(x, y2) and not np.any(diff & ~zero_both)), int(np.sum(diff & zero_both)), int(np.sum(diff & ~zero_both))


def strip(rec):
    return {k: v for k, v in rec.items() if k not in ("q",)}


def main():
    t0 = time.perf_counter()
    hot = json.loads((RUNS / "hotfix_runs.json").read_text(encoding="utf-8"))
    par = json.loads((RUNS / "parent_runs.json").read_text(encoding="utf-8"))
    report = {"K1": {}, "cases": {}, "pairs": {}, "parent_positive": {}, "parent_negative": {}, "forbidden": {}}
    names = list(hot["cases"])
    # K1 for each (nx, N, q)
    for name in names:
        c = hot["cases"][name]
        report["K1"][name] = k1(c["a"], c["nx"], c["N"], c["q"])
    # per case: K2, K3, K5
    rn5_vals = {}
    for name in names:
        c = hot["cases"][name]
        z = np.load(RUNS / f"hotfix_{name}.npz")
        a, b, nx, ny, N, q = c["a"], c["b"], c["nx"], c["ny"], c["N"], c["q"]
        s1, s2 = c["steps"]
        r = {"status": c["status"], "completed": c["completed"], "s1_status": s1["status"], "s2_status": s2["status"],
             "s1_done": s1["done"], "s2_done": s2["done"], "calls_ok": c["calls"] == ["orthonormalise", "solve_bc", "rn5_certificate",
                                                                                     "orthonormalise", "rt7_conditions", "solve_bc"],
             "rn5_builder": s2["rn5"]["verdict"], "rt7_builder": s2["rt7"]["passed"], "qr_s2": s2["qr_executed"]}
        # the RN-5 input is the saved s = 1 pair: s2.given (V before QR) is V1 bit for bit
        r["given_is_V1"] = bool(np.array_equal(z["s2_given"], z["s1_V"]) and z["s2_given"].tobytes() == z["s1_V"].tobytes())
        ri = rn5_signfree(a, b, nx, ny, N, z["s1_G"], z["s1_V"], 1.0, q)
        r["rn5_mine"] = {"certified": ri["certified"], "first_failed": ri["first_failed"],
                         "margin_sigma_over_eps_low": float(ri["values"].get("margin_sigma_over_eps_low", 0)),
                         "eps": float(ri["values"].get("eps", 0)), "sigma_low": float(ri["values"].get("sigma_low", 0))}
        rn5_vals[name] = {k: frac_key(v) for k, v in ri["values"].items()}
        # RT-7 independently (exact ranks, tau, t from MY eps / sigma_low)
        Vq, Gq = fq(z["s1_V"]), fq(z["s2_G"])
        rk = (ind.solve(ind.mm(ind.tp(Vq), Vq), ind.eye(N))[1] != 0, ind.solve(ind.mm(ind.tp(Gq), Gq), ind.eye(N))[1] != 0)
        Vt, Gt = ind.tp(Vq), ind.tp(Gq)
        M = ind.mm(ind.mm(ind.solve(ind.mm(Vt, Vq), ind.eye(N))[0], ind.mm(Vt, Gq)),
                   ind.mm(ind.solve(ind.mm(Gt, Gq), ind.eye(N))[0], ind.mm(Gt, Vq)))
        tau = sum(1 - M[i][i] for i in range(N))
        t = ri["values"]["eps"] / ri["values"]["sigma_low"] if ri["certified"] else None
        r["rt7_mine"] = {"ranks_full": all(rk), "tau": float(tau), "t": float(t) if t is not None else None,
                         "passed": all(rk) and t is not None and 0 <= t < 1 and tau < (1 - t) ** 2}
        rn5_vals[name]["tau"] = frac_key(tau)
        # K3 exact solves
        e1, _ = exact_check(z["s1_G"], z["s1_V"], a, nx, b, ny, 1.0, q)
        e2, x2 = exact_check(z["s2_G"], z["s2_V"], b, ny, a, nx, 1.0, q)
        r["K3_s1"], r["K3_s2"] = e1, e2
        # field = X G2^T with the zero boundary, axis 0 = x
        w = z["field"]
        r["field_is_X_G2T"] = bool(np.array_equal(w[1:-1, 1:-1], z["s2_V"] @ z["s2_G"].T) and not np.any(w[0]) and not np.any(w[-1])
                                   and not np.any(w[:, 0]) and not np.any(w[:, -1]) and w.shape == (nx + 1, ny + 1))
        r["wbar_is_w_over_q"] = bool(np.array_equal(z["wbar"], w * 1.0 / (q * a**4)))
        # K5 energies
        P1 = pi_exact(field_q(z["s1_G"], z["s1_V"]), a, b, nx, ny, 1.0, q)
        P2 = pi_exact(field_q(z["s2_V"], z["s2_G"]), a, b, nx, ny, 1.0, q)
        Pc = pi_exact(field_q(z["s2_companion"], z["s2_G"]), a, b, nx, ny, 1.0, q)
        N2 = N
        Xs = [[x2[p * N2 + i] for i in range(N2)] for p in range(nx - 1)]
        Pstar = pi_exact(ind.mm(Xs, ind.tp(fq(z["s2_G"]))), a, b, nx, ny, 1.0, q)
        ph1, ph2, phc = (float.fromhex(s1["P_hat"]), float.fromhex(s2["P_hat"]), float.fromhex(s2["P_hat_previous_QR"]))
        r["K5"] = {"Pi1": float(P1), "Pi2": float(P2), "P_hat1": ph1, "P_hat2": ph2, "P_hat_prevQR": phc,
                   "rel_P1": abs(ph1 - float(P1)) / abs(float(P1)), "rel_P2": abs(ph2 - float(P2)) / abs(float(P2)),
                   "rel_Pc": abs(phc - float(Pc)) / abs(float(Pc)), "galerkin_min": Pstar <= Pc, "negative": P1 < 0 and P2 < 0,
                   "Pi2_minus_Pi1_saved": float(P2 - P1)}
        r["K5"]["ok"] = max(r["K5"]["rel_P1"], r["K5"]["rel_P2"], r["K5"]["rel_Pc"]) <= 1e-12 and r["K5"]["galerkin_min"] and r["K5"]["negative"]
        r["ok"] = (r["status"] == "COMPLETED_FIXED_HALF_STEPS" and r["completed"] == 2 and r["s1_done"] and r["s2_done"] and r["calls_ok"]
                   and r["rn5_builder"] == "RN5_CERTIFIED" and r["rt7_builder"] and r["qr_s2"] and r["given_is_V1"]
                   and r["rn5_mine"]["certified"] and r["rt7_mine"]["passed"] and e1["ok"] and e2["ok"] and r["field_is_X_G2T"]
                   and r["wbar_is_w_over_q"] and r["K5"]["ok"])
        report["cases"][name] = r
        print(name, "ok" if r["ok"] else "FAIL", f"K3 s1 {e1['rel_err']:.2e}/{e1['criterion']:.1e} s2 {e2['rel_err']:.2e}/{e2['criterion']:.1e}",
              f"mine t={r['rt7_mine']['t']:.3e}", flush=True)
    # K4 pairs
    for name in names:
        c = hot["cases"][name]
        if c["q"] > 0:
            neg = name.replace("q+", "q-")
            zp, zn = np.load(RUNS / f"hotfix_{name}.npz"), np.load(RUNS / f"hotfix_{neg}.npz")
            assert set(zp.files) == set(zn.files)
            arrays, signed_zero, other = {}, 0, 0
            for k in zp.files:
                assert k in EVEN or k in ODD, k
                eq, sz, ot = sign_compare(zp[k], zn[k], k in ODD)
                arrays[k] = eq
                signed_zero += sz
                other += ot
            cp, cn_ = hot["cases"][name], hot["cases"][neg]
            steps_equal = cp["steps"] == cn_["steps"]
            rec_equal = {k: cp[k] == cn_[k] for k in cp if k not in ("q", "F_check")}
            report["pairs"][name] = {"arrays_ok": all(arrays.values()), "arrays": arrays, "signed_zero_positions": signed_zero,
                                     "other_bit_differences": other, "steps_records_equal": steps_equal,
                                     "run_records_equal": all(rec_equal.values()), "mine_rn5_values_equal": rn5_vals[name] == rn5_vals[neg]}
            print("pair", name, report["pairs"][name]["arrays_ok"], signed_zero, other, steps_equal, report["pairs"][name]["mine_rn5_values_equal"])
    # K6 parent vs hotfix
    for name in names:
        c, p = hot["cases"][name], par["cases"][name]
        zh, zp = np.load(RUNS / f"hotfix_{name}.npz"), np.load(RUNS / f"parent_{name}.npz")
        if c["q"] > 0:
            same_keys = set(zh.files) == set(zp.files)
            arr = {k: zh[k].dtype == zp[k].dtype and zh[k].shape == zp[k].shape and zh[k].tobytes() == zp[k].tobytes() for k in zh.files}
            report["parent_positive"][name] = {"keys": same_keys, "arrays_bitwise": all(arr.values()), "differing": [k for k, v in arr.items() if not v],
                                               "record_equal": c == p}
        else:
            s2 = p["steps"][1] if len(p["steps"]) > 1 else {}
            report["parent_negative"][name] = {"status": p["status"], "completed": p["completed"], "calls": p["calls"],
                                               "rn5_verdict": (s2.get("rn5") or {}).get("verdict"),
                                               "closed_forms": ((s2.get("rn5") or {}).get("hypotheses") or {}).get("closed forms consistent with the enclosure"),
                                               "F_old_ok": p["F_check"]["old_ok"], "F_new_ok": p["F_check"]["new_ok"],
                                               "s1_bitwise_as_hotfix": all(zh[k].tobytes() == zp[k].tobytes() for k in zp.files if k.startswith("s1_"))}
    # K7 forbidden statuses
    text = json.dumps(hot, ensure_ascii=False)
    report["forbidden"] = {w: len(re.findall(p, text)) for w, p in (("CONVERGED", r"CONVERGED"), ("ACCEPTED", r"ACCEPTED"),
                                                                     ("QUALIFIED_not_NOT", r"(?<!NOT_)QUALIFIED"))}
    report["application_status"] = sorted({c["application_status"] for c in hot["cases"].values()})
    report["kkvim_imported"] = any(m == "kkvim" or m.startswith("kkvim") for m in sys.modules)
    allk1 = all(v["identity_and_bound"] and (v["old_form_rejects"] == (hot["cases"][n]["q"] < 0)) for n, v in report["K1"].items())
    report["summary"] = {
        "K1": allk1,
        "K2_K3_K5_cases": all(r["ok"] for r in report["cases"].values()),
        "K4_pairs": all(v["arrays_ok"] and v["steps_records_equal"] and v["run_records_equal"] and v["mine_rn5_values_equal"]
                        for v in report["pairs"].values()),
        "K6_parent_positive": all(v["keys"] and v["arrays_bitwise"] and v["record_equal"] for v in report["parent_positive"].values()),
        "parent_negative_reproduces_defect": all(v["status"] == "STOPPED_RANK_GUARD_UNCERTIFIED" and v["closed_forms"] is False
                                                 and v["F_old_ok"] is False and v["s1_bitwise_as_hotfix"] for v in report["parent_negative"].values()),
        "K7": not any(report["forbidden"].values()),
        "no_kkvim": not report["kkvim_imported"],
        "seconds": time.perf_counter() - t0,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (HERE / "negq_independent.json").write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps(report["summary"], indent=1), report["application_status"], report["forbidden"])


if __name__ == "__main__":
    main()
