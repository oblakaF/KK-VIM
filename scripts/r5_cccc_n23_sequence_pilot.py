"""D-066: isolated numerical pilot of CCCC N = 2-3 beyond s = 2 (task R5-CCCC-N23-FULL-SEQUENCE-PILOT; goals G3.4, G4, G5; R5-V).

EXPERIMENTAL. Every result is RESEARCH_ONLY / PROVISIONAL / NOT_QUALIFIED. This is not a production command: ``run-cccc-n23`` still stops before
s = 3 (D-065); nothing in ``src/kkvim`` is changed or bypassed.

Two separate execution paths:

    run      (imports ``kkvim`` only)            the half-steps s = 1..K, K <= 32, Y, X, Y, X, ...
    compare  (imports ``kkvim_reference`` only)  the independent D2 reference on the same grid; run as a separate process

    uv run python scripts/r5_cccc_n23_sequence_pilot.py [run] --N 2 --nx 12 --ny 12 --half-steps 8 --out scratch/d066/n2
    uv run python scripts/r5_cccc_n23_sequence_pilot.py compare --out scratch/d066/n2     (``run`` starts it unless --no-compare)

s = 1 and s = 2 are those of ``kkvim.cccc_n23.run_cccc_n23_physical(..., half_steps=2)`` unchanged (D-063, D-065: RN-5 before the QR of V,
RT-7 after it, grid <= 32). Every half-step s >= 3 repeats the operations of s = 2 with the roles of x and y alternating:

    1  the solved set V of half-step s - 1 (finite; exact rank of the stored doubles; < N is a proven dependence -> stop)
    2  QR with the grid weight of its direction: sqrt(h) V = Q R, G_s = Q / sqrt(h); companion = G_{s-1} R^T, so w(s-1) = companion x G_s
    3  exact rank of G_s (< N or a zero diagonal of R -> stop)
    4  C-3+ assembly of A, B, C, F on G_s (``cccc_n23.assemble``), rounding probes P-1 and premises P-2, P-3, P-5 (failure -> stop)
    5  B-C solve (``cccc_n23.solve_bc``) of the other direction; nonfinite or zero -> stop
    6  eta, exact rank of the new set, rank diagnostic, P_hat = Pi_h^C of the field, alpha_centre: all written, none is a threshold

No rank cutoff, no convergence tolerance, no regularization, no lower N, no eta guard at s >= 2 (the thresholds of s = 1 are not
transferred, D-065). No RN-5 / RT-7 at s >= 3: the rank and the accuracy of the exact discrete sequence are UNRESOLVED (O4-h for s >= 2);
the exact rank of the stored binary64 factors is an algebraic diagnostic only. No CONVERGED status exists.

Iteration controller (task R5-CCCC-N23-ITERATION-CONTROLLER; s = 1..8 under D-066, s = 9..32 a research-only continuation by the owner's
instruction of that task, without a new decision). ``--half-steps`` is the budget K <= 32. The only stops:

    RESEARCH_LIMIT_REACHED          the budget K is reached
    STOPPED_BITWISE_REPEATED_STATE  the stored (G, V) of a done half-step equal bitwise those of an earlier done half-step of the same phase
                                    (Y or X); the next half-steps depend only on (G, V), so the floating-point computation would repeat.
                                    It proves no mathematical convergence.
    PILOT_STOPPED                   a protective stop of production s = 1, 2 or of ``pilot_step`` (proven dependence, nonfinite value,
                                    failed C-3+ premise, failed factorization or solve, zero field); the reason names it

No approximate stabilization test is applied; the changes below are only written. change = max|w_s - w_{s-1}| (w_0 = 0),
same_phase_change = max|w_s - w_{s-2}| (s >= 3), both relative to max|w_s| over all nodes (None if max|w_s| = 0).

The D2 comparison (``compare``) uses ``kkvim_reference.r5_cccc_d2`` (nodewise 13-point assembly, sparse LU, refinement against the
double-double residual of the exact operator); D is recomputed there from t, E, nu. D2 is an independent implementation of the accepted
discrete problem (D-049.1), not a measure of the grid error; every difference is an OBSERVATION.
"""

import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MARKING = "RESEARCH_ONLY / PROVISIONAL / NOT_QUALIFIED (D-066, experimental pilot; not a production computation)"
RANK_S3 = "UNRESOLVED: no RN-5/RT-7 at s >= 3; O4-h for s >= 2 UNRESOLVED; exact rank of stored binary64 = algebraic diagnostic only"
RESEARCH_LIMIT_REACHED = "RESEARCH_LIMIT_REACHED"
STOPPED_BITWISE_REPEATED_STATE = "STOPPED_BITWISE_REPEATED_STATE"
PILOT_STOPPED = "PILOT_STOPPED"
BUDGET_MAX = 32  # the owner's bound of the research continuation; not a tolerance
AUTHORIZATION = ("s = 1, 2: D-063, D-065; s = 3..8: D-066; s = 9..32: owner instruction R5-CCCC-N23-ITERATION-CONTROLLER "
                 "(research-only continuation, no new decision)")
HISTORY_COLUMNS = ("s", "direction", "status", "reason", "N", "n_s", "h_s", "given_exact_rank", "exact_rank_G", "R_diag_min_abs",
                   "cond_A_before_QR", "cond_A_after_QR", "premises_status", "cond_S", "min_eig_DM", "eta", "V_exact_rank", "sigma",
                   "rho_N", "kappa_rep", "field_preservation", "P_hat_previous_QR", "P_hat", "w_centre", "alpha_centre", "max_abs_w",
                   "change_vs_previous_max", "rel_change_vs_previous", "same_phase_change_max", "rel_same_phase_change",
                   "repeated_state_of", "guards", "rn5_verdict", "rn5_margin", "rt7_passed", "rank_status")


def _git(*args) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "UNAVAILABLE"


def provenance() -> dict:
    import scipy

    return {"git_commit": _git("rev-parse", "HEAD"), "git_branch": _git("branch", "--show-current"),
            "worktree_modified": bool(_git("status", "--porcelain", "--untracked-files=no")),
            "python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__, "platform": platform.platform(),
            "machine": platform.machine(), "script": "scripts/r5_cccc_n23_sequence_pilot.py"}


def _number(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, tuple):
        return list(value)
    return value


# --- the working path ----------------------------------------------------------------------------------------------------------------

def _pilot_classes():
    import kkvim.cccc_n23 as cn

    class PilotHalfStep(cn.HalfStep):
        """A row of the pilot; the field factors follow the solved direction, not the index s."""

        def field_factors(self):
            return (self.G, self.V) if self.direction == "Y" else (self.V, self.G)

        def previous_factors(self):
            return (self.G, self.companion) if self.direction == "Y" else (self.companion, self.G)

    return cn, PilotHalfStep


def pilot_step(prev, s: int, a: float, b: float, D: float, q: float, n_x: int, n_y: int):
    """Half-step s >= 2 from the done row ``prev`` (operations 1-6 of the module docstring); a stop is a row with its status."""
    import kkvim.cccc_n1 as n1

    cn, PilotHalfStep = _pilot_classes()
    direction = "Y" if s % 2 else "X"
    l_g, n_g, l_s, n_s = (a, n_x, b, n_y) if direction == "Y" else (b, n_y, a, n_x)
    N = prev.N
    step = PilotHalfStep(s=s, k=s - 1, direction=direction, n_s=n_s, h_s=l_s / n_s, N=N, qr_executed=False, given=prev.V)
    step.rank_decision = RANK_S3
    try:
        if not np.all(np.isfinite(prev.V)):
            raise cn._Stop(cn.ERROR_NONFINITE, f"s = {s}: решённый набор полушага {s - 1} не конечен")
        if prev.V_exact_rank < N:
            raise cn._Stop(cn.STOP_BEFORE_QR_S1, f"s = {s}: точный ранг решённого набора {prev.V_exact_rank} < N = {N} (доказанная зависимость)")
        step.G, step.R = cn.orthonormalise(prev.V, l_g / n_g)
        step.qr_executed = True
        step.R_diag_min_abs = float(np.min(np.abs(np.diag(step.R))))
        singular = np.linalg.svd(step.R, compute_uv=False)
        step.cond_A_before_QR = float((singular[0] / singular[-1]) ** 2) if singular[-1] > 0 else math.inf
        step.companion = prev.G @ step.R.T
        step.exact_rank_G = cn.exact_rank(step.G)
        if not np.all(np.diag(step.R) != 0.0) or step.exact_rank_G < N:
            raise cn._Stop(cn.PROTECTIVE_STOP_FACTORIZATION, f"s = {s}, после QR: min|diag R| = {step.R_diag_min_abs!r}, точный ранг G = {step.exact_rank_G}")
        with np.errstate(all="ignore"):
            w_prev = cn._field(*prev.field_factors())
            step.field_preservation = float(np.max(np.abs(cn._field(*step.previous_factors()) - w_prev)) / np.max(np.abs(w_prev)))
        step.P_hat_previous_QR = cn.discrete_energy(*step.previous_factors(), a, b, D, q, n_x, n_y)
        # operations 4-6: those of cccc_n23._solve_given with eta_guard=False, the field oriented by the solved direction
        failures = n1.rounding_probe_failures()
        parts = cn.assemble(step.G, l_g, n_g, q)
        step.A, step.B, step.C, step.F, step.C_L, step.E = (parts[name] for name in ("A", "B", "C", "F", "C_L", "E"))
        if not all(np.all(np.isfinite(value)) for value in (step.A, step.B, step.F)):
            raise cn._Stop(cn.ERROR_NONFINITE, "A, B или F не конечно")
        eigenvalues = np.linalg.eigvalsh(step.A)
        step.cond_A_after_QR = float(eigenvalues[-1] / eigenvalues[0]) if eigenvalues[0] > 0 else math.inf
        failures += cn.premise_failures(parts, n_g)
        step.premises_status = cn.PREMISES_UNRESOLVED if failures else cn.PREMISES_ESTABLISHED
        step.premises_reasons = tuple(failures)
        if failures:
            raise cn._Stop(cn.UNRESOLVED_C3PLUS_PREMISE, "; ".join(failures))
        V = cn.solve_bc(step, step.A, step.B, step.C, step.F, D, l_s, n_s)
        failures = n1.rounding_probe_failures()
        if failures:
            step.premises_status, step.premises_reasons = cn.PREMISES_UNRESOLVED, tuple(failures)
            raise cn._Stop(cn.UNRESOLVED_C3PLUS_PREMISE, "; ".join(failures))
        if not np.all(np.isfinite(V)):
            raise cn._Stop(cn.ERROR_NONFINITE, "V не конечно")
        step.V = V
        if not np.any(V):
            raise cn._Stop(cn.STOP_ZERO_FIELD, "V = 0 точно")
        step.eta = cn.residual_eta(V, step.A, step.B, step.C, step.F, D, step.h_s)
        if not math.isfinite(step.eta):
            raise cn._Stop(cn.ERROR_NONFINITE, f"η = {step.eta!r}")
        step.V_exact_rank = cn.exact_rank(V)
        step.sigma, step.rho_N, step.kappa_rep = cn.rank_diagnostic(step.G, l_g / n_g, V, l_s / n_s)
        x, y = step.field_factors()
        step.P_hat = cn.discrete_energy(x, y, a, b, D, q, n_x, n_y)
        if n_x % 2 == 0 and n_y % 2 == 0:
            step.alpha_centre = float(cn._field(x, y)[n_x // 2, n_y // 2]) * D / (q * min(a, b) ** 4)
        step.done = True
    except cn._Stop as stop:
        step.status, step.reason = stop.status, stop.reason
    return step


def repeated_state(rows, step):
    """s of an earlier done row of the same phase whose stored G and V equal those of ``step`` bitwise, or None."""
    for earlier in rows:
        if (earlier is not step and earlier.done and earlier.direction == step.direction and earlier.G.shape == step.G.shape
                and earlier.V.shape == step.V.shape and np.array_equal(earlier.G, step.G) and np.array_equal(earlier.V, step.V)):
            return earlier.s
    return None


def run_pilot(a, b, t, E, nu, q, n_x, n_y, N, half_steps):
    """(status, reason, production result of s = 1..2, all rows). s = 1, 2: production unchanged; s >= 3: ``pilot_step``."""
    import kkvim.cccc_n23 as cn

    if not (isinstance(half_steps, int) and not isinstance(half_steps, bool) and 1 <= half_steps <= BUDGET_MAX):
        raise ValueError(f"half_steps must be an integer 1..{BUDGET_MAX}")
    result = cn.run_cccc_n23_physical(a, b, t, E, nu, q, n_x, n_y, N, min(half_steps, 2))
    rows = list(result.steps)
    if result.status != cn.COMPLETED_FIXED_HALF_STEPS:
        return PILOT_STOPPED, f"s = {len(rows)}: {result.status}: {result.reason}", result, rows
    for step in rows:
        step.status = cn.HALF_STEP_DONE
    prev = rows[-1]
    for s in range(3, half_steps + 1):
        prev = pilot_step(prev, s, result.a, result.b, result.D, result.q, result.n_x, result.n_y)
        rows.append(prev)
        if not prev.done:
            return PILOT_STOPPED, f"s = {s}: {prev.status}: {prev.reason}", result, rows
        prev.repeated_state_of = repeated_state(rows, prev)
        if prev.repeated_state_of is not None:
            reason = (f"s = {s}: G и V побитово равны сохранённым на s = {prev.repeated_state_of} той же фазы "
                      "(повтор вычислительного состояния; не доказательство сходимости)")
            return STOPPED_BITWISE_REPEATED_STATE, reason, result, rows
    return RESEARCH_LIMIT_REACHED, None, result, rows


def _guards(step) -> str:
    """The guard outcomes of a row, in the order of the operations."""
    out = []
    if step.s >= 2 and step.given is not None and step.qr_executed:
        out.append("exact rank of the given solved set = N")
    if step.qr_executed:
        out.append(f"QR: min|diag R| = {step.R_diag_min_abs:.3e}, exact rank G = {step.exact_rank_G}")
    if step.rn5 is not None:
        out.append(f"RN-5 {step.rn5['verdict']}")
    if step.rt7 is not None:
        out.append(f"RT-7 {'passed' if step.rt7['passed'] else 'failed'}")
    if step.premises_status is not None:
        out.append(f"C-3+ premises {step.premises_status}")
    if step.V is not None:
        out.append("solve finite, V != 0")
    if not step.done:
        out.append(f"STOP {step.status}")
    return "; ".join(out)


def _relative(change, scale):
    return None if change is None or not scale else change / scale


def _row(step, D, q, a, b, n_x, n_y, previous_field, phase_field=None):
    import kkvim.cccc_n23 as cn

    row = {name: _number(getattr(step, name, None)) for name in HISTORY_COLUMNS if hasattr(step, name)}
    row.update(s=step.s, direction=step.direction, status=step.status, reason=step.reason)
    row.update(same_phase_change_max=None, rel_same_phase_change=None)  # s <= 2 or a stopped row: no same-phase field
    row["given_exact_rank"] = None if step.given is None else int(cn.exact_rank(step.given))
    field = None
    if step.done:
        x, y = (step.G, step.V) if step.direction == "Y" else (step.V, step.G)
        field = cn._field(x, y)
        row["max_abs_w"] = float(np.max(np.abs(field)))
        row["w_centre"] = float(field[n_x // 2, n_y // 2]) if n_x % 2 == 0 and n_y % 2 == 0 else None
        row["change_vs_previous_max"] = float(np.max(np.abs(field - previous_field)))
        row["rel_change_vs_previous"] = _relative(row["change_vs_previous_max"], row["max_abs_w"])
        if phase_field is not None:
            row["same_phase_change_max"] = float(np.max(np.abs(field - phase_field)))
            row["rel_same_phase_change"] = _relative(row["same_phase_change_max"], row["max_abs_w"])
    row["repeated_state_of"] = getattr(step, "repeated_state_of", None)
    row["guards"] = _guards(step)
    if step.rn5 is not None:
        row["rn5_verdict"] = step.rn5["verdict"]
        eps, low = step.rn5.get("eps"), step.rn5.get("sigma_low")
        row["rn5_margin"] = float(low / eps) if eps and low else None
    if step.rt7 is not None:
        row["rt7_passed"] = bool(step.rt7["passed"])
    row["rank_status"] = ("s = 1: exact rank of the stored factors; D-063" if step.s == 1 else
                          "s = 2: RN-5 + RT-7 (D-065)" if step.s == 2 else RANK_S3)
    return row, field


def command_run(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    status, reason, result, rows = run_pilot(args.a, args.b, args.t, args.E, args.nu, args.q, args.nx, args.ny, args.N, args.half_steps)
    D = result.D
    history, fields, factors = [], {}, {}
    previous = np.zeros((args.nx + 1, args.ny + 1))
    for step in rows:
        row, field = _row(step, D, args.q, args.a, args.b, args.nx, args.ny, previous, fields.get(f"w_s{step.s - 2}"))
        history.append(row)
        if field is not None:
            fields[f"w_s{step.s}"] = field
            previous = field
        for name in ("G", "V", "R", "companion", "given"):
            value = getattr(step, name, None)
            if value is not None:
                factors[f"{name}_s{step.s}"] = value
    np.savez(out / "fields.npz", **fields)
    np.savez(out / "factors.npz", **factors)
    with open(out / "history.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(history)
    digest = hashlib.sha256(b"".join(fields[k].tobytes() for k in sorted(fields))).hexdigest()
    summary = {"task": "R5-CCCC-N23-ITERATION-CONTROLLER (pilot R5-CCCC-N23-FULL-SEQUENCE-PILOT)", "decision": "D-066",
               "authorization": AUTHORIZATION, "marking": MARKING, "status": status, "reason": reason,
               "stops": [RESEARCH_LIMIT_REACHED, STOPPED_BITWISE_REPEATED_STATE, PILOT_STOPPED],
               "convergence": "NOT ASSESSED (no CONVERGED status)",
               "half_steps_requested": args.half_steps, "half_steps_completed": sum(1 for step in rows if step.done),
               "sequence": "Y, X, Y, X, ... (s = 1 solves Y on the start set T_N in x, D-049.3)",
               "input": {"a": args.a, "b": args.b, "t": args.t, "E": args.E, "nu": args.nu, "q": args.q, "n_x": args.nx, "n_y": args.ny,
                         "N": args.N, "D_solver": D},
               "s1_s2": "kkvim.cccc_n23.run_cccc_n23_physical(..., half_steps=min(K, 2)) unchanged",
               "s_ge_3": RANK_S3, "fields_sha256": digest, "provenance": provenance(), "history": history}
    (out / "result.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"{status}: {reason or ''}\n  completed {summary['half_steps_completed']}/{args.half_steps}, fields sha256 {digest[:16]}, out {out}", flush=True)
    if args.no_compare:
        return 6 if status == PILOT_STOPPED else 0
    done = subprocess.run([sys.executable, str(Path(__file__).resolve()), "compare", "--out", str(out)], cwd=ROOT, check=False)
    return done.returncode or (6 if status == PILOT_STOPPED else 0)


# --- the reference path (separate process; no kkvim import) ------------------------------------------------------------------------

def command_compare(args) -> int:
    from kkvim_reference import r5_cccc_d2 as d2

    out = Path(args.out)
    summary = json.loads((out / "result.json").read_text(encoding="utf-8"))
    p = summary["input"]
    D_ref = p["E"] * p["t"] ** 3 / (12.0 * (1.0 - p["nu"] ** 2))
    problem = d2.D2Problem(p["a"], p["b"], p["n_x"], p["n_y"], D_ref, p["q"])
    sol = d2.solve(problem)
    w_ref = sol.field()
    A = d2.assemble(problem)
    kron_gap = float(abs(A - d2.kron_matrix(problem)).max() / abs(A).max())
    x_ref = sol.x
    energy_ref = float(x_ref @ (A @ x_ref))
    hx, hy = p["a"] / p["n_x"], p["b"] / p["n_y"]
    Pi_ref = -p["q"] * sol.work() / 2.0
    fields = np.load(out / "fields.npz")
    history = {row["s"]: row for row in summary["history"]}
    rows = []
    previous = np.zeros_like(w_ref)
    for s in sorted(int(k[3:]) for k in fields.files):
        w = fields[f"w_s{s}"]
        e = w - w_ref
        ei = e[1:-1, 1:-1].ravel()
        h = history[s]
        rows.append({"s": s, "direction": h["direction"],
                     "w_centre": float(w[p["n_x"] // 2, p["n_y"] // 2]),
                     "alpha_centre": float(w[p["n_x"] // 2, p["n_y"] // 2]) * D_ref / (p["q"] * min(p["a"], p["b"]) ** 4),
                     "max_abs_diff_vs_D2": float(np.max(np.abs(e))),
                     "rel_max_diff_vs_D2": float(np.max(np.abs(e)) / np.max(np.abs(w_ref))),
                     "rel_l2_diff_vs_D2": float(np.linalg.norm(e) / np.linalg.norm(w_ref)),
                     "rel_energy_diff_vs_D2": float(math.sqrt(max(float(ei @ (A @ ei)), 0.0) / energy_ref)),
                     "change_vs_previous_max": float(np.max(np.abs(w - previous))),
                     "same_phase_change_max": h.get("same_phase_change_max"),
                     "P_hat": h["P_hat"], "P_hat_minus_Pi_D2": (h["P_hat"] - Pi_ref) if h["P_hat"] is not None else None,
                     "eta": h["eta"], "cond_S": h["cond_S"], "rho_N": h["rho_N"], "V_exact_rank": h["V_exact_rank"],
                     "rank_status": h["rank_status"]})
        previous = w
    reference = {"module": "kkvim_reference.r5_cccc_d2 (nodewise 13-point, SuperLU MMD_AT_PLUS_A, double-double refinement)",
                 "D_reference": D_ref, "D_solver": p["D_solver"], "D_bitwise_equal": D_ref == p["D_solver"],
                 "w_centre": sol.centre(), "alpha_centre": sol.alpha(), "max_abs_w": float(np.max(np.abs(w_ref))),
                 "Pi_h": Pi_ref, "e_D2_last_correction": sol.e_D2, "refinement_corrections": sol.corrections,
                 "residual_max_dd": sol.residual_max, "nodewise_vs_kron_rel_max": kron_gap,
                 "norms": "rel_max = max|w - w_D2| / max|w_D2|; rel_l2 = ||w - w_D2||_F / ||w_D2||_F over all nodes; "
                          "rel_energy = sqrt(e^T A e / x^T A x), A = the D2 nodewise matrix, inner nodes",
                 "meaning": "independent implementation of the same discrete problem (D-049.1); not a grid-error measure; OBSERVATION",
                 "kkvim_imported_in_this_process": "kkvim" in sys.modules, "h": [hx, hy]}
    with open(out / "comparison.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["s"])
        writer.writeheader()
        writer.writerows(rows)
    (out / "comparison.json").write_text(json.dumps({"marking": MARKING, "pilot_status": summary["status"], "pilot_reason": summary["reason"],
                                                     "reference": reference, "rows": rows}, indent=2, ensure_ascii=False, default=str),
                                         encoding="utf-8")
    print(f"D2: alpha {sol.alpha():.16g}, e_D2 {sol.e_D2:.2e}, D equal {D_ref == p['D_solver']}, kkvim imported {'kkvim' in sys.modules}")
    print(f"{'s':>2} {'dir':>3} {'alpha_centre':>22} {'rel_max vs D2':>13} {'rel_energy':>11} {'change':>10} {'P_hat - Pi_D2':>13} {'eta':>9} {'rank':>4}")
    for r in rows:
        gap = r["P_hat_minus_Pi_D2"]
        print(f"{r['s']:>2} {r['direction']:>3} {r['alpha_centre']:>22.16g} {r['rel_max_diff_vs_D2']:>13.3e} {r['rel_energy_diff_vs_D2']:>11.3e} "
              f"{r['change_vs_previous_max']:>10.3e} {gap if gap is None else format(gap, '13.3e'):>13} {r['eta']:>9.2e} {r['V_exact_rank']:>4}")
    print(f"pilot: {summary['status']} {summary['reason'] or ''}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--N", type=int, required=True, choices=(2, 3))
    run.add_argument("--nx", type=int, required=True)
    run.add_argument("--ny", type=int, required=True)
    run.add_argument("--half-steps", type=int, default=8, choices=range(1, BUDGET_MAX + 1), metavar="1..32")
    run.add_argument("--a", type=float, default=1.0)
    run.add_argument("--b", type=float, default=1.0)
    run.add_argument("--t", type=float, default=0.01)
    run.add_argument("--E", type=float, default=2.1e11)
    run.add_argument("--nu", type=float, default=0.3)
    run.add_argument("--q", type=float, default=1000.0)
    run.add_argument("--out", required=True)
    run.add_argument("--no-compare", action="store_true")
    compare = sub.add_parser("compare")
    compare.add_argument("--out", required=True)
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] not in ("run", "compare", "-h", "--help"):
        argv = ["run", *argv]  # the form of the task: the options of ``run`` without the sub-command
    args = parser.parse_args(argv)
    return command_run(args) if args.command == "run" else command_compare(args)


if __name__ == "__main__":
    sys.exit(main())
