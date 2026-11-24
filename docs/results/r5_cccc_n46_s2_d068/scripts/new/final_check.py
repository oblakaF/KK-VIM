"""R5-CCCC-N46-S2-D068-FINAL: independent verification of the new and hotfix-affected cases (QA scratch, not in git). No import of kkvim.

Reused, unchanged (imported, SHA256 asserted):
  ../d068_s2/s2_independent_check.py  (operator from kkvim_reference.r5_cccc_d2.nodewise_parts, exact projected X system, bounds)
  ../rn5_negq/negq_independent.py     (my independent RN-5 with the q-free F check; helpers of ../rn5_n46/rn5_n46_independent.py)

Projected X system (s = 2, given G2 on y, unknown X on x, w = X G2^T, x index slow):
  K = P^T A P,  f = (q/D) P^T 1,  P = I_(n_x-1) (x) G2,  A = sx Z_x + sxy Z_xy + sy Z_y  (D2 nodewise parts, exact rational scales)

Criteria (fixed BEFORE any comparison; the same as the D-068 QA, whose task fixed "the error bound before comparison"):
  X1  certified forward bound  ||X_hat - X*||_F <= ||rho||_2 / lambda_low(K), rho = K x_hat - f EXACT (fractions),
      lambda_low(K) >= (mu_x^2 + mu_y^2) * Gershgorin_low(G2^T G2) (proved), kappa_up = Gershgorin_up(K) / lambda_low(K) (proved upper bound).
      PASS iff B_rel = bound/||X_hat|| <= 2 kappa_up n u (classical normwise forward bound of a backward-stable solve; a CERTIFIED
      kappa_up, not an estimate). Heuristic kappa (float eigvalsh) is reported separately and decides nothing.
  REF reference X_ref = X_hat - K_float^-1 rho (one refinement step with the EXACT residual); its certified uncertainty
      ||X_ref - X*||_F <= ||rho_ref|| / lambda_low. delta_abs = max|w_prod - X_ref G2^T|, delta_rel = delta_abs / max|X_ref G2^T|;
      certified field error of production: max|w_prod - w*| <= gamma_N-term + ||X_hat - X*||_F max_j ||G2_j||_2.
  F1/E1/E2/R7 as in the D-068 QA (field = X G2^T within gamma_N; P_hat within the stated estimate; Pi(w2) <= Pi(companion G2^T) +
      1/2 c ||rho||^2/lambda_low; RT-7 exact with t from MY RN-5).
  D2  (investigation, observation) full 2-D D2 solution u (kkvim_reference.solve, refined): delta_rel(w_prod, u); the exact identity
      Pi(w) - Pi(u) = 1/2 c (w-u)^T A (w-u) + c (w-u)^T r_u (r_u = A u - q/D 1 exact) splits the difference into the energy-norm
      distance of the rank-N field from the full-space solution and the (tiny) residual term of u.
  SIGN q -> -q on the CLI files: even factors equal, odd factors negated, field w negated, w_bar equal; differences of the sign bit only at
      exact zeros (IEEE-754 +0.0/-0.0) are counted, never a discrepancy.
  STOP refusal cases: exit 6, status, no QR of V, no s = 2 factors other than allowed, field_diagnostic.csv only, = G1 V1^T.
"""
import csv
import hashlib
import io
import json
import sys
import time
from fractions import Fraction as Q
from pathlib import Path

import numpy as np

sys.set_int_max_str_digits(0)
sys.stdout.reconfigure(encoding="utf-8")  # -I ignores PYTHONIOENCODING
HERE =Path(__file__).resolve().parent
SCR = HERE.parent
RUNS = HERE / "runs"
PREV_RUNS = SCR / "d068_s2" / "runs"
REUSE = {SCR / "d068_s2" / "s2_independent_check.py": "60948d8becd31f9472b4b3ee04287e6b88f4480a3b293a0725567f26d8ada5f1",
         SCR / "rn5_negq" / "negq_independent.py": "215edcbb224572f7d58fdbcc7b5110168720467648ddc32f1dcfa153443c3872"}
for p, h in REUSE.items():
    assert hashlib.sha256(p.read_bytes()).hexdigest() == h, p
sys.path.insert(0, str(SCR / "rn5_negq"))
sys.path.insert(0, str(SCR / "d068_s2"))
import negq_independent as ng  # noqa: E402,I001 - first: its guard rejects any module named kkvim* (also kkvim_reference)
import s2_independent_check as s2c  # noqa: E402
from kkvim_reference import r5_cccc_d2 as d2  # noqa: E402

assert not any(m == "kkvim" or m.startswith("kkvim.") for m in sys.modules), "kkvim must not be imported"
U = Q(1, 2**53)
T_, E_, NU_ = 0.01, 2.1e11, 0.3

S2_CASES = {"B_pos": (6, 1.0, 5.0, 24, 32, 1000.0), "B_neg": (6, 1.0, 5.0, 24, 32, -1000.0),
            "D_N4_neg": (4, 1.0, 1.0, 32, 32, -1000.0), "D_N5_neg": (5, 1.0, 0.6, 20, 13, -1000.0)}
STOP_CASES = {"C_pos": (6, 5.0, 1.0, 32, 13, 1000.0), "C_neg": (6, 5.0, 1.0, 32, 13, -1000.0)}
RERUNS = {"R_C1": "C1", "R_C2": "C2", "R_C3": "C3", "R_C4": "C4", "R_C5": "C5", "R_C6": "C6", "R_C7": "C7"}
PAIRS = {"B_neg": "B_pos", "D_N4_neg": "R_C1", "D_N5_neg": "R_C7", "C_neg": "C_pos"}
EVEN_ROLES = {"T", "G", "A", "B", "C", "C_L", "E"}
ODD_ROLES = {"V", "F", "companion"}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rows(path):
    lines = [ln for ln in Path(path).read_text(encoding="utf-8").splitlines() if not ln.startswith("#")]
    return list(csv.DictReader(io.StringIO("\n".join(lines))))


def gersh_lo(M):
    return min(M[i][i] - sum(abs(M[i][j]) for j in range(len(M)) if j != i) for i in range(len(M)))


def json_diff(a, b, path=""):
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            out += json_diff(a.get(k, "<missing>"), b.get(k, "<missing>"), f"{path}/{k}")
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)):
            out += json_diff(x, y, f"{path}[{i}]")
    elif a != b:
        out.append(path)
    return out


# ------------------------------------------------------------------------------------------------------------------------------ s = 2
def check_s2(tag, N, a, b, nx, ny, q):
    t0 = time.time()
    d = RUNS / tag
    res = {"tag": tag, "N": N, "a": a, "b": b, "nx": nx, "ny": ny, "q": q, "checks": {}, "values": {}}
    ck, val = res["checks"], res["values"]
    doc = json.loads((d / "result.json").read_text(encoding="utf-8"))
    hist = rows(d / "history.csv")
    fac = s2c.read_factors(d / "factors.csv")
    D = float(doc["D"]["value"]) if isinstance(doc["D"], dict) else float(doc["D"])
    val["D"] = D
    inp = doc["input"]
    ck["input"] = (inp["N"], inp["n_x"], inp["n_y"], inp["half_steps"]) == (N, nx, ny, 2) and inp["a"]["value"] == a and \
        inp["b"]["value"] == b and inp["q"]["value"] == q and inp["t"]["value"] == T_ and inp["E"]["value"] == E_ and inp["nu"]["value"] == NU_
    ck["schema"] = doc["schema"] == "kkvim.cli.cccc_n46/2"
    ck["status_completed_2"] = doc["status"] == "COMPLETED_FIXED_HALF_STEPS" and doc["half_steps_completed"] == 2 and doc["reason"] is None
    ck["research_file_only"] = (d / "field_research.csv").exists() and not (d / "field_diagnostic.csv").exists() and not (d / "field.csv").exists()
    ck["not_qualified_label"] = "NOT_QUALIFIED" in json.dumps(doc["application_status"], ensure_ascii=False)
    forb = {"CONVERGED", "ACCEPTED", "QUALIFIED", "Q5_ACCEPTED"}
    ck["no_forbidden_status"] = not any(s in forb for s in s2c.walk_strings(doc)) and \
        all(r["status"] in ("HALF_STEP_DONE", "COMPLETED_FIXED_HALF_STEPS") for r in hist)
    h2 = doc["rank"]["half_step_2"]
    ck["run_rn5_rt7_qr"] = h2["rn5"]["certified"] is True and h2["rn5"]["verdict"] == "RN5_CERTIFIED" and h2["qr_executed"] is True and \
        h2["rt7"]["passed"] is True and all(h2["rt7"][k] for k in ("R7-1", "R7-2", "R7-3"))
    G1, V1 = fac[(1, "G")], fac[(1, "V")]
    G2, R2, X, comp = fac[(2, "G")], fac[(2, "R")], fac[(2, "V")], fac[(2, "companion")]
    ck["shapes"] = G1.shape == (nx - 1, N) and V1.shape == (ny - 1, N) and G2.shape == (ny - 1, N) and X.shape == (nx - 1, N)
    ck["R_upper"] = bool(np.all(np.tril(R2, -1) == 0.0)) and bool(np.all(np.diag(R2) != 0.0))
    val["qr_relation_max_rel"] = float(np.max(np.abs(G2 @ R2 - V1)) / np.max(np.abs(V1)))
    # --- my RN-5 on the saved s = 1 factors of THIS run (q-free F check)
    ri = ng.rn5_signfree(a, b, nx, ny, N, G1, V1, D, q)
    ck["independent_rn5_certified"] = ri["certified"]
    val["rn5_mine"] = {"certified": ri["certified"], "first_failed": ri["first_failed"], "seconds": ri.get("seconds"),
                       "eps": float(ri["values"].get("eps", 0)), "sigma_low": float(ri["values"].get("sigma_low", 0)),
                       "margin": float(ri["values"].get("margin_sigma_over_eps_low", 0)),
                       "lambda_low": float(ri["values"].get("lambda_low", 0)), "norm_dK": float(ri["values"].get("norm_dK", 0))}
    val["rn5_run"] = {k: h2["rn5"]["values"].get(k) for k in ("lambda_low", "norm_dK", "eps", "sigma_low", "margin_sigma_over_eps")}
    # --- RT-7 exact, t from my RN-5
    V1f, G2f, G1f = s2c.fr(V1), s2c.fr(G2), s2c.fr(G1)
    rV, rG = s2c.rank_exact(V1f), s2c.rank_exact(G2f)
    GV, GG, VG = s2c.gram(V1f, V1f), s2c.gram(G2f, G2f), s2c.gram(V1f, G2f)
    M1 = s2c.solve_exact(GV, VG)
    M2 = s2c.solve_exact(GG, [list(r) for r in zip(*VG)])
    tau = N - sum(sum(M1[i][m] * M2[m][i] for m in range(N)) for i in range(N))
    t_up = ri["values"]["eps"] / s2c.sqrt_dn(ri["values"]["sigma2_low"]) if ri["certified"] else None
    ck["RT7_independent"] = rV == N and rG == N and t_up is not None and 0 <= t_up < 1 and tau < (1 - t_up) ** 2
    val["rt7"] = {"tau": float(tau), "t_qa": None if t_up is None else float(t_up), "t_run": h2["rt7"]["t"], "tau_run": h2["rt7"]["tau"]}
    # --- companion = G1 R^T within gamma_N
    R2f, compf = s2c.fr(R2), s2c.fr(comp)
    gN = s2c.gamma(N)
    ck["companion"] = all(abs(compf[i][k] - sum(G1f[i][m] * R2f[k][m] for m in range(N))) <= gN * sum(abs(G1f[i][m] * R2f[k][m]) for m in range(N))
                          for i in range(nx - 1) for k in range(N))
    # --- the projected X system, exact residual, certified bounds
    op = s2c.Operator(a, b, nx, ny)
    ck["operator_kron_exact"] = op.kron_structure_exact()
    Dq, qq = Q(D), Q(q)
    load = qq / Dq
    my = ny - 1
    gl = gersh_lo(GG)
    lam_low = op.lambda_low() * gl
    lam_up = op.gersh_up() * max(GG[i][i] + sum(abs(GG[i][j]) for j in range(N) if j != i) for i in range(N))
    n_dim = N * (nx - 1)

    def exact_rho(Xm):
        Xq = s2c.fr(Xm)
        w = s2c.field_exact(Xq, G2f)
        Pi, Aw = s2c.pi_exact(op, w, Dq, qq)
        r = [v - load for v in Aw]
        rho = [[sum(G2f[j][k] * r[i * my + j] for j in range(my)) for k in range(N)] for i in range(nx - 1)]
        return Xq, w, Pi, rho, sum(v * v for row in rho for v in row)

    Xf, w2, Pi2, rho, rho2 = exact_rho(X)
    x2 = sum(v * v for row in Xf for v in row)
    bound = s2c.sqrt_up(rho2) / lam_low
    B_rel = bound / s2c.sqrt_dn(x2)
    kappa_up = lam_up / lam_low
    crit = 2 * kappa_up * n_dim * U
    ck["lambda_low_positive"] = gl > 0 and lam_low > 0
    ck["X1_certified_bound_within_criterion"] = B_rel <= crit
    import scipy.sparse as sp
    Af = d2.assemble(op.problem)
    P = sp.kron(sp.identity(nx - 1), sp.csr_matrix(G2)).tocsc()
    Kf = (P.T @ (Af @ P)).toarray()
    ev = np.linalg.eigvalsh(Kf)
    ff = float(load) * (P.T @ np.ones(op.problem.size))
    x_float = np.linalg.solve(Kf, ff).reshape(nx - 1, N)
    rho_f = np.array([[float(v) for v in row] for row in rho]).reshape(-1)
    X_ref = X - np.linalg.solve(Kf, rho_f).reshape(nx - 1, N)
    _, w_ref_q, _, rho_ref, rho_ref2 = exact_rho(X_ref)
    ref_bound = s2c.sqrt_up(rho_ref2) / lam_low
    g2max = max(s2c.sqrt_up(sum(G2f[j][k] ** 2 for k in range(N))) for j in range(my))
    # production field from the file
    F = {(int(r["i"]), int(r["j"])): r for r in rows(d / "field_research.csv")}
    ck["field_grid"] = len(F) == (nx + 1) * (ny + 1)
    w_prod = np.zeros((nx + 1, ny + 1))
    okb = okw = okbar = okx = True
    hx_f, hy_f = a / nx, b / ny
    for (i, j), r in F.items():
        w = float(r["w [m]"])
        w_prod[i, j] = w
        if i in (0, nx) or j in (0, ny):
            okb &= w == 0.0
            continue
        ex = w2[(i - 1) * my + (j - 1)]
        okw &= abs(Q(w) - ex) <= gN * sum(abs(Xf[i - 1][k] * G2f[j - 1][k]) for k in range(N))
        okx &= abs(float(r["x [m]"]) - i * hx_f) <= 4e-16 * a and abs(float(r["y [m]"]) - j * hy_f) <= 4e-16 * b
        wb = w * D / (q * a**4)
        okbar &= abs(float(r["w_bar [1]"]) - wb) <= 4 * 2.0**-53 * abs(wb)
    ck["F1_field_eq_X_G2T"], ck["field_boundary_zero"], ck["field_axis0_x"], ck["field_w_bar"] = bool(okw), bool(okb), bool(okx), bool(okbar)
    w_ref = np.zeros_like(w_prod)
    w_ref[1:-1, 1:-1] = X_ref @ G2.T
    delta_abs = float(np.max(np.abs(w_prod - w_ref)))
    wmax = float(np.max(np.abs(w_ref)))
    gamma_term = max(float(gN * sum(abs(Xf[i][k] * G2f[j][k]) for k in range(N))) for i in range(nx - 1) for j in range(my))
    val["X"] = {"n": n_dim, "rho_prod": float(s2c.sqrt_up(rho2)), "rho_ref": float(s2c.sqrt_up(rho_ref2)),
                "lambda_low_K_certified": float(lam_low), "lambda_up_K_certified": float(lam_up), "kappa_up_certified": float(kappa_up),
                "kappa_heuristic_eigvalsh": float(ev[-1] / ev[0]), "lambda_min_heuristic": float(ev[0]),
                "abs_bound_X_prod": float(bound), "B_rel": float(B_rel), "criterion_2kappa_n_u": float(crit),
                "abs_bound_X_ref": float(ref_bound), "rel_bound_X_ref": float(ref_bound / s2c.sqrt_dn(x2)),
                "float_solve_rel_diff": float(np.linalg.norm(x_float - X) / np.linalg.norm(X)),
                "delta_abs_field": delta_abs, "delta_rel_field": delta_abs / wmax, "w_ref_max": wmax,
                "certified_field_err_prod_vs_exact": float(bound * g2max) + gamma_term,
                "certified_field_err_ref_vs_exact": float(ref_bound * g2max) + 1e-300}
    # --- energies
    w1 = s2c.field_exact(G1f, V1f)
    Pi1, _ = s2c.pi_exact(op, w1, Dq, qq)
    Pic, _ = s2c.pi_exact(op, s2c.field_exact(compf, G2f), Dq, qq)
    c = Dq * op.hx * op.hy
    gap = c / 2 * rho2 / lam_low
    ck["E2_monotone_proven"] = Pi2 <= Pic + gap
    p_hist = {int(r["s"]): float(r["P_hat"]) for r in hist}
    val["energy"] = {"Pi1": float(Pi1), "Pi2": float(Pi2), "Pic": float(Pic), "P_hat1": p_hist[1], "P_hat2": p_hist[2],
                     "rel_P_hat2": abs(p_hist[2] - float(Pi2)) / abs(float(Pi2)), "rel_P_hat1": abs(p_hist[1] - float(Pi1)) / abs(float(Pi1)),
                     "Pi2_minus_Pi1": float(Pi2 - Pi1), "solve_gap": float(gap)}
    # E1 (P_hat vs exact Pi) is reported as an observation only; no threshold is introduced here
    # --- D2 full 2-D solution (investigation of a ~1e-6 disagreement)
    sol = d2.solve(d2.D2Problem(a, b, nx, ny, D, q))
    u = sol.field()
    uq = [Q(float(v)) for v in sol.x]
    Pu, Au = s2c.pi_exact(op, uq, Dq, qq)
    ru = [v - load for v in Au]
    diff = [x - y for x, y in zip(w2, uq)]
    Adiff = op.apply_exact(diff)
    en = c / 2 * sum(x * y for x, y in zip(diff, Adiff))
    lin = c * sum(x * y for x, y in zip(diff, ru))
    val["D2"] = {"delta_abs_prod_vs_D2": float(np.max(np.abs(w_prod - u))), "delta_rel_prod_vs_D2": float(np.max(np.abs(w_prod - u)) / np.max(np.abs(u))),
                 "delta_rel_s1_vs_D2": float(np.max(np.abs(np.pad(G1 @ V1.T, 1) - u)) / np.max(np.abs(u))),
                 "D2_own_error_estimate_e_D2": sol.e_D2, "D2_residual_max": float(max(abs(v) for v in ru)),
                 "Pi_w2_minus_Pi_u": float(Pi2 - Pu), "half_energy_norm_sq": float(en), "residual_term": float(lin),
                 "identity_exact": Pi2 - Pu == en + lin, "energy_rel_gap": float((Pi2 - Pu) / abs(Pu))}
    ck["D2_identity_exact"] = val["D2"]["identity_exact"]
    res["seconds"] = time.time() - t0
    res["all_checks"] = all(ck.values())
    return res


# ------------------------------------------------------------------------------------------------------------------------------ stops
def check_stop(tag, N, a, b, nx, ny, q):
    d = RUNS / tag
    log = (RUNS / f"{tag}.log").read_text(encoding="utf-8")
    doc = json.loads((d / "result.json").read_text(encoding="utf-8"))
    fac = s2c.read_factors(d / "factors.csv")
    hist = rows(d / "history.csv")
    h2 = doc["rank"]["half_step_2"]
    G1, V1 = fac[(1, "G")], fac[(1, "V")]
    F = {(int(r["i"]), int(r["j"])): float(r["w [m]"]) for r in rows(d / "field_diagnostic.csv")}
    W = np.zeros((nx + 1, ny + 1))
    for (i, j), v in F.items():
        W[i, j] = v
    ri = ng.rn5_signfree(a, b, nx, ny, N, G1, V1, float(doc["D"]["value"] if isinstance(doc["D"], dict) else doc["D"]), q)
    out = {"status": doc["status"], "completed": doc["half_steps_completed"], "reason": doc["reason"],
           "rn5_run": {k: h2["rn5"].get(k) for k in ("certified", "verdict", "reason")}, "rn5_hypotheses": h2["rn5"]["hypotheses"],
           "rn5_values": {k: h2["rn5"]["values"].get(k) for k in ("lambda_low", "norm_dK", "eps", "sigma_low")},
           "qr_executed": h2["qr_executed"], "rt7": h2.get("rt7"), "s2_factor_roles": sorted(r for s, r in fac if s == 2),
           "files": sorted(p.name for p in d.iterdir()), "history_rows": [(r["s"], r["status"]) for r in hist],
           "diagnostic_eq_G1V1T": bool(np.array_equal(W[1:-1, 1:-1], G1 @ V1.T)) and not np.any(W[0]) and not np.any(W[-1]),
           "diag_header": "DIAGNOSTIC" in (d / "field_diagnostic.csv").read_text(encoding="utf-8"),
           "exit_in_log": "exit" in log, "rn5_mine": {"certified": ri["certified"], "first_failed": ri["first_failed"],
                                                    "values": {k: float(v) for k, v in ri["values"].items() if isinstance(v, Q)}}}
    out["ok"] = out["status"] == "STOPPED_RANK_GUARD_UNCERTIFIED" and out["completed"] == 1 and out["rn5_run"]["certified"] is False and \
        out["qr_executed"] is False and not out["s2_factor_roles"] and "field_research.csv" not in out["files"] and \
        "field_diagnostic.csv" in out["files"] and out["diagnostic_eq_G1V1T"] and out["diag_header"]
    return out


# ------------------------------------------------------------------------------------------------------------------------------ sign
def sign_pair(neg, pos):
    dn, dp = RUNS / neg, RUNS / pos
    fn, fp = rows(dn / "factors.csv"), rows(dp / "factors.csv")
    bad, sz = [], 0
    ok = len(fn) == len(fp)
    for a_, b_ in zip(fn, fp):
        key = (a_["s"], a_["role"], a_["i"], a_["j"])
        ok &= key == (b_["s"], b_["role"], b_["i"], b_["j"])
        role = a_["role"]
        odd = role in ODD_ROLES or (role == "R" and a_["s"] == "2")
        assert odd or role in EVEN_ROLES or role == "R", role
        x, y = float(a_["value"]), float(b_["value"])
        y = -y if odd else y
        if np.float64(x).view(np.uint64) != np.float64(y).view(np.uint64):
            if x == 0 and y == 0:
                sz += 1
            else:
                bad.append(key)
    field_name = "field_research.csv" if (dn / "field_research.csv").exists() else "field_diagnostic.csv"
    wn, wp = rows(dn / field_name), rows(dp / field_name)
    fbad, fsz = 0, 0
    for a_, b_ in zip(wn, wp):
        for col, odd in (("w [m]", True), ("w_bar [1]", False), ("x [m]", False), ("y [m]", False)):
            x, y = float(a_[col]), float(b_[col])
            y = -y if odd else y
            if np.float64(x).view(np.uint64) != np.float64(y).view(np.uint64):
                if x == 0 and y == 0:
                    fsz += 1
                else:
                    fbad += 1
    hn, hp = rows(dn / "history.csv"), rows(dp / "history.csv")
    hdiff = sorted({k for a_, b_ in zip(hn, hp) for k in a_ if a_[k] != b_[k]})
    jn = json.loads((dn / "result.json").read_text(encoding="utf-8"))
    jp = json.loads((dp / "result.json").read_text(encoding="utf-8"))
    return {"factors_ok": ok and not bad, "factor_bad": bad[:5], "factor_signed_zeros": sz, "field_file": field_name,
            "field_bad": fbad, "field_signed_zeros": fsz, "history_diff_cols": hdiff, "result_json_diff_paths": json_diff(jn, jp)}


def main():
    t0 = time.time()
    out = {"task": "R5-CCCC-N46-S2-D068-FINAL", "script_sha256": sha(__file__), "reused": {str(p.relative_to(SCR)): h for p, h in REUSE.items()},
           "kkvim_reference_file": d2.__file__, "kkvim_reference_sha256": sha(d2.__file__), "python": sys.version, "numpy": np.__version__,
           "criteria": __doc__[__doc__.index("Criteria"):], "reruns": {}, "s2": {}, "stops": {}, "sign": {}}
    for tag, prev in RERUNS.items():
        same = {n: sha(RUNS / tag / n) == sha(PREV_RUNS / prev / n) for n in ("history.csv", "factors.csv", "field_research.csv")}
        a_ = json.loads((RUNS / tag / "result.json").read_text(encoding="utf-8"))
        b_ = json.loads((PREV_RUNS / prev / "result.json").read_text(encoding="utf-8"))
        out["reruns"][tag] = {"files_bitwise": same, "result_json_diff_paths": json_diff(a_, b_)}
        print(tag, same, out["reruns"][tag]["result_json_diff_paths"], flush=True)
    for tag, args in S2_CASES.items():
        r = out["s2"][tag] = check_s2(tag, *args)
        bad = [k for k, v in r["checks"].items() if not v]
        X = r["values"]["X"]
        print(tag, "ALL_OK" if not bad else f"FAILED {bad}", f"B_rel={X['B_rel']:.3e} crit={X['criterion_2kappa_n_u']:.3e}",
              f"kup={X['kappa_up_certified']:.2e} kh={X['kappa_heuristic_eigvalsh']:.2e} dRel={X['delta_rel_field']:.2e}",
              f"D2rel={r['values']['D2']['delta_rel_prod_vs_D2']:.2e} {r['seconds']:.0f}s", flush=True)
    for tag, args in STOP_CASES.items():
        r = out["stops"][tag] = check_stop(tag, *args)
        print(tag, r["ok"], r["status"], r["rn5_run"], r["rn5_mine"]["certified"], r["rn5_mine"]["first_failed"], flush=True)
    for neg, pos in PAIRS.items():
        r = out["sign"][neg] = sign_pair(neg, pos)
        print("sign", neg, pos, r["factors_ok"], r["factor_signed_zeros"], r["field_bad"], r["field_signed_zeros"], r["history_diff_cols"],
              r["result_json_diff_paths"], flush=True)
    out["kkvim_imported_end"] = any(m == "kkvim" or m.startswith("kkvim.") for m in sys.modules)
    out["seconds"] = time.time() - t0
    (HERE / "final_check.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
