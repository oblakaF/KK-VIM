"""R5-V Q4 PRELAUNCH (protocol docs/R5_CCCC_Q4_D1_PROTOCOL.md 5 items 8-9, 5.1): the certified reference of BR (PR #39) applied to the
ACTUAL solution V_bc of an own OP-8 half-step with the C-3+ assembly (known limitation N-1 of the PRELAUNCH: BR was measured on the
OLD assembly of the prototype). N = 1, small non-D1 inputs only.

QA research adapter, not a qualification and not the working solver. The functions of scripts/r5_cccc_q4_bc_reference_audit.py (BR)
and of kkvim_reference are used read-only; nothing of them is changed. The adapter NEVER calls the B-C solver (bc.run_halfstep,
bc.solve_bc) nor the OP-8 solve: the approximate solution under test is V_bc read from the OP-8 journal (or given with its record), and a
B-C call is not an independent reference (protocol 5 item 9).

Reference for the FIXED input of the half-step (the solved set of the journal, l_x, l_y, D, q exactly, h = l/n exactly; NS 4.2-4.3):
1. K and b assembled EXACTLY (BR ``exact_coefficients``, ``exact_band``, ``rhs_exact``; equality with the Q1b band is recorded);
2. lambda_low < lambda_min(K) by the Q1b certificate (hs.certify_lambda_min, t = lambda_est/2 of BR ``tau_trial``): this certifies K
   SPD; without it the reference is UNRESOLVED;
3. decimal banded Cholesky with the BR precision ladder p = 40, 80, 160, refinement with the EXACT residual r = b - K x (BR
   ``refine``); ||x - x*||_2 <= ||r||_2/lambda_low, ||x - x*||_K <= ||r||_2/lambda_low^(1/2);
4. error of V_bc: d = ||x^ - x||_2, d_K, d_inf = ||x^ - x||_inf exactly; intervals [d - b, d + b] in each norm (the inf-norm radius
   b_inf <= ||r||_2/lambda_low since ||.||_inf <= ||.||_2 for the error of the reference itself; the inf-norm error of B-C is NOT
   derived from its 2-norm, protocol 5.1); budget 1/100 (D-049.6 item 4) in the same norm: 100 b <= max(0, d - b), else UNRESOLVED
   (budget); a refinement that does not converge -> UNRESOLVED;
5. observations (OBSERVATION, protocol 5.1): BC-1 form - relative inf-norm interval of the set; TS-1 - e_field (energy norm of the
   field, = the K-norm relative error, BR 3) against the candidate 10 u n_s^2 (c_f = 10; candidate, not a norm);
6. independent paths of BR: exact rational solution for N m <= 36 and R-H for (n_x - 1)(n_y - 1) <= 4096: the reference must lie
   within its bound of x*, the B-C error within its interval.

Input confirmation (refusal = UNRESOLVED, never a precision claim): with a journal record the half-step must be an OP-8 C-3+ half-step
with status OK, P-1, P-2, P-3, P-5 established (P-4 assumed), the only-C replacement confirmed and V_bc / solved set bitwise the
recorded hashes.

Usage: from Python (``reference_for_journal``); the controls harness and the tests call it.
"""

import importlib.util
import math
import os
import sys
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

THREAD_VARS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
for _k in THREAD_VARS:
    os.environ[_k] = "1"  # as in BR (effective only if NumPy is not loaded yet)

import numpy as np  # noqa: E402

from kkvim_reference import r5_cccc_halfstep as hs  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BR_TOOL = "scripts/r5_cccc_q4_bc_reference_audit.py"
PIPELINE_TOOL = "scripts/r5_cccc_q4_d1_row_pipeline_prelaunch.py"
UNRESOLVED = "UNRESOLVED"
U = Fraction(1, 2**53)
C_F = 10  # TS-1 candidate c_f = 10 (D-054.4 item 3): OBSERVATION only, not a norm


def _load(rel: str, name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def br():
    return _load(BR_TOOL, "_ad_br")


def pl():
    return _load(PIPELINE_TOOL, "_pl_pipeline")


def _dec(x: Fraction, up: bool) -> str:
    b = br()
    return b.dec_up(x) if up else b.dec_down(x)


def confirm_c3plus(rec: dict, solved: np.ndarray, V: np.ndarray) -> list[str]:
    """Reasons why the record is not an OP-8 C-3+ half-step with this solution (empty list: confirmed)."""
    op8 = pl().op8()
    r = []
    if rec.get("status") != op8.OK:
        r.append(f"OP-8 status {rec.get('status')!r} is not OK")
    if rec.get("c3plus_premises_status") != op8.PREMISES_OK:
        r.append("C-3+ premises not ESTABLISHED_CONDITIONAL_ON_P4")
    if rec.get("only_C_replaced") is not True or not str(rec.get("C_assembly_applied", "")).startswith("C-3+"):
        r.append("the C-3+ assembly of the solve is not confirmed (only_C_replaced / C_assembly_applied)")
    if rec.get("V_bc_sha") != op8.array_sha(V):
        r.append("V_bc is not bitwise the recorded solution (V_bc_sha)")
    if rec.get("solved_set_sha") != op8.array_sha(solved):
        r.append("the solved set is not bitwise the recorded one (solved_set_sha)")
    return r


def reference(cfg, s: int, solved: np.ndarray, V: np.ndarray, ladder=None, tau_override: float | None = None) -> dict:
    """Module docstring, steps 1-6, for the fixed input (solved set) and the approximate solution V (m x N) of half-step s."""
    b_ = br()
    pl().guard_config(cfg)
    direction = pl().op8().direction_of(s)
    ps = hs.HalfStep(float(cfg.lx), cfg.nx, float(cfg.ly), cfg.ny, direction, np.asarray(solved, dtype=float), D=float(cfg.D),
                     q=float(cfg.q))
    N, m = ps.N, ps.n_s - 1
    n, pb = N * m, 3 * N - 1
    V = np.asarray(V, dtype=float)
    out = {"s": s, "direction": direction, "N": N, "n_s": ps.n_s, "unknowns": n, "solution_under_test": "V_bc of the OP-8 C-3+ "
           "half-step (no B-C call here)", "V_sha": pl().op8().array_sha(V)}
    if V.shape != (m, N) or not np.all(np.isfinite(V)):
        out["result"] = f"{UNRESOLVED}: V is not a finite ({m}, {N}) array"
        return out
    coef = b_.exact_coefficients(ps)
    q1b = hs.exact_coefficients(ps)
    out["coefficients_equal_q1b"] = all(coef[t] == q1b[t] for t in range(4))
    band = b_.exact_band(ps, coef)
    out["band_equal_q1b"] = band == hs._exact_band_matrix(ps, q1b)
    rows = b_.band_rows(band, n)
    b = b_.rhs_exact(ps, coef)
    s_hat = b_.to_node_term([Fraction(float(v)) for v in V.T.reshape(-1)], N, m)
    r_hat = b_.residual(rows, b, s_hat)
    tau = tau_override if tau_override is not None else b_.tau_trial(ps)
    lam = None if tau is None else hs.certify_lambda_min(ps, tau, q1b)
    out["tau_trial"] = tau
    out["lambda_low"] = lam
    out["K_SPD_certified"] = lam is not None and lam > 0
    if not out["K_SPD_certified"]:
        out["result"] = f"{UNRESOLVED}: no certificate of lambda_min(K) > 0 (Q1b): K SPD not certified"
        return out
    L = Fraction(lam)
    attempts, used, ok_set = [], None, False
    for prec in (ladder or b_.P_LADDER):
        fac = b_.dec_cholesky(band, n, pb, prec)
        if fac is None:
            attempts.append(f"p={prec}: factorization failed")
            continue
        x, r, log = b_.refine(rows, b, fac)
        r2 = b_.norm2_sq(r)
        b2 = b_.sqrt_up(r2) / L
        bK = b_.sqrt_up(r2) / b_.sqrt_down(L)
        dv = [a - c for a, c in zip(s_hat, x, strict=True)]
        d2 = b_.norm2_sq(dv)
        dK2 = sum((a * (rr - rb) for a, rr, rb in zip(dv, r, r_hat, strict=True)), Fraction(0))
        dinf = max((abs(v) for v in dv), default=Fraction(0))
        set_lo, set_hi = max(Fraction(0), b_.sqrt_down(d2) - b2), b_.sqrt_up(d2) + b2
        K_lo, K_hi = max(Fraction(0), b_.sqrt_down(dK2) - bK), b_.sqrt_up(dK2) + bK
        inf_lo, inf_hi = max(Fraction(0), dinf - b2), dinf + b2
        ok_set, ok_field, ok_inf = b_.budget_ok(b2, set_lo), b_.budget_ok(bK, K_lo), b_.budget_ok(b2, inf_lo)
        used = prec
        attempts.append(f"p={prec}: refinement {len(log) - 1} steps, ||r||_inf {log[0]:.2e} -> {log[-1]:.2e}, budget set {ok_set}, "
                        f"field {ok_field}, inf {ok_inf}")
        if ok_set and ok_field and ok_inf or prec == (ladder or b_.P_LADDER)[-1]:
            break
    out["precision_log"] = " | ".join(attempts)
    if used is None:
        out["result"] = f"{UNRESOLVED}: no factorization at any precision of the ladder"
        return out
    out["precision"] = used
    out["refine_steps"] = len(log) - 1
    out["refine_converged"] = len(log) == 1 or log[-1] < log[0]
    h_s = Fraction(ps.l_s) / ps.n_s
    sh_up, sh_lo = b_.sqrt_up(h_s), b_.sqrt_down(h_s)
    xs2 = b_.norm2_sq(x)
    xK2 = sum((a * (bb - rr) for a, bb, rr in zip(x, b, r, strict=True)), Fraction(0))
    xs_lo = max(Fraction(0), b_.sqrt_down(xs2) - b2)
    xK_lo = max(Fraction(0), b_.sqrt_down(xK2) - bK)
    xinf_lo = max(Fraction(0), max(abs(v) for v in x) - b2)
    out.update({"ref_bound_2_up": _dec(b2, True), "ref_bound_K_up": _dec(bK, True), "ref_bound_inf_up": _dec(b2, True),
                "set_error_2_down": _dec(set_lo, False), "set_error_2_up": _dec(set_hi, True),
                "field_error_a_down": _dec(sh_lo * K_lo, False), "field_error_a_up": _dec(sh_up * K_hi, True),
                "set_error_inf_down": _dec(inf_lo, False), "set_error_inf_up": _dec(inf_hi, True),
                "budget_set": "SATISFIED" if ok_set else f"NOT_MET: comparison {UNRESOLVED}",
                "budget_field": "SATISFIED" if ok_field else f"NOT_MET: comparison {UNRESOLVED}",
                "budget_inf": "SATISFIED" if ok_inf else f"NOT_MET: comparison {UNRESOLVED}"})
    # observations (protocol 5.1): BC-1 form (inf-norm of the set) and TS-1 (energy norm of the field = K-norm relative)
    if xinf_lo > 0 and ok_inf:
        out["BC1_rel_inf_down"], out["BC1_rel_inf_up"] = _dec(inf_lo / (max(abs(v) for v in x) + b2), False), _dec(inf_hi / xinf_lo, True)
    else:
        out["BC1_rel_inf"] = f"{UNRESOLVED}: budget or certified lower bound of ||V*||_inf"
    if xK_lo > 0 and ok_field:
        e_lo, e_hi = K_lo / (b_.sqrt_up(xK2) + bK), K_hi / xK_lo
        cand = C_F * U * ps.n_s ** 2
        out["TS1_e_field_down"], out["TS1_e_field_up"] = _dec(e_lo, False), _dec(e_hi, True)
        out["TS1_candidate_10_u_ns2"] = _dec(cand, True)
        out["TS1_observation"] = ("ABOVE the candidate (OBSERVATION)" if e_lo > cand else
                                  "WITHIN the candidate (OBSERVATION)" if e_hi <= cand else "UNDECIDED (interval contains the candidate)")
    else:
        out["TS1_e_field"] = f"{UNRESOLVED}: budget or certified lower bound of ||U V*^T||_a"
    # independent paths of BR
    checks = []
    if n <= b_.EXACT_LIMIT:
        Ke, be = b_.e46.exact_system(ps, q1b)
        x_star = b_.to_node_term(b_.e46._solve_many(Ke, [be])[0], N, m)
        out["exact_ref_within_bound"] = b_.norm2_sq([a - c for a, c in zip(x, x_star, strict=True)]) <= b2 * b2
        e_true2 = b_.norm2_sq([a - c for a, c in zip(s_hat, x_star, strict=True)])
        out["exact_bc_error_in_interval"] = set_lo <= b_.sqrt_up(e_true2) and b_.sqrt_down(e_true2) <= set_hi
        checks += [out["exact_ref_within_bound"], out["exact_bc_error_in_interval"]]
    if (cfg.nx - 1) * (cfg.ny - 1) <= b_.RH_LIMIT:
        st, xh = hs.solve_hessian(ps)
        if xh is not None:
            xh_nt = b_.to_node_term([Fraction(float(v)) for v in xh], N, m)
            bh = b_.sqrt_up(b_.norm2_sq(b_.residual(rows, b, xh_nt))) / L
            out["rh_consistent"] = b_.norm2_sq([a - c for a, c in zip(xh_nt, x, strict=True)]) <= (bh + b2) ** 2
            checks.append(out["rh_consistent"])
        else:
            out["rh_consistent"] = f"{UNRESOLVED}: {st}"
    out["independent_paths_consistent"] = all(checks) if checks else None
    if not out["refine_converged"]:
        out["result"] = f"{UNRESOLVED}: refinement did not converge"
    elif checks and not all(checks):
        out["result"] = f"{UNRESOLVED}: CONTRADICTION of the independent paths"
    elif ok_set and ok_field and ok_inf:
        out["result"] = "RESOLVED (set 2-norm, set inf-norm, field energy norm)"
    else:
        out["result"] = f"{UNRESOLVED}: budget 1/100 not met (set {ok_set}, field {ok_field}, inf {ok_inf})"
    return out


def reference_for_journal(jdir: Path, s: int, ladder=None, tau_override: float | None = None) -> dict:
    """The reference for half-step s of an OP-8 journal: input confirmation, then ``reference`` on (solved set, V_bc)."""
    p = pl()
    j, arrays = p.load_journal(Path(jdir))
    cfg = p.config_of(j["meta"])
    p.guard_config(cfg)
    rec = next((r for r in j["steps"] if r["s"] == s), None)
    if rec is None or f"s{s}_V_bc" not in arrays:
        return {"s": s, "result": f"{UNRESOLVED}: no solved half-step {s} in the journal"}
    solved, V = arrays[f"s{s}_solved_set"], arrays[f"s{s}_V_bc"]
    bad = confirm_c3plus(rec, solved, V)
    binding = p.bind_journal(j, arrays)[s]
    if not binding["ok"]:
        bad += binding["reasons"]
    if bad:
        return {"s": s, "input_confirmed": False, "result": f"{UNRESOLVED}: input not confirmed as the OP-8 C-3+ output: " + "; ".join(bad)}
    out = reference(cfg, s, solved, V, ladder, tau_override)
    out["input_confirmed"] = True
    out["C3plus_sha"] = rec.get("C3plus_sha")
    out["C_old_equals_C3plus_bitwise"] = rec.get("C_old_equals_C3plus_bitwise")
    return out


def decimal_value(text: str) -> Fraction:
    return Fraction(Decimal(text))
