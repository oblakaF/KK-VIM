"""R5-V Q4 PRELAUNCH: integrated controls A, B, V (В), G (Г) of the N = 1 row pipeline (protocol docs/R5_CCCC_Q4_D1_PROTOCOL.md 4, 6,
12 item 4) against the expectations fixed BEFORE the pipeline and this harness were written
(tests/reference/r5_cccc_q4_d1_controls_prelaunch_expected.json, its own commit; sha256 checked, the harness refuses a changed file).

Research harness on small non-D1 configurations (rows T1, T2 of the expectations), not the D1 campaign: no D1 row, no campaign verdict,
no D1_RESULT. The pipeline (scripts/r5_cccc_q4_d1_row_pipeline_prelaunch.py), the BR adapter
(scripts/r5_cccc_q4_d1_bc_reference_adapter.py) and the PR #48 classifier with its frozen controls (scripts/r5_cccc_q4_d1_preflight.py,
docs/results/r5_cccc_q4_d1_prelaunch/control_ab_expected.json) are used read-only; the frozen files of PR #48 are not rewritten.

- A: the PR #48 harness (213 row cases, 18 aggregation cases, detector functions) re-run unchanged, and the pipeline-boundary cases
  AP-01...AP-25 (exact dP and tau', Z-3 rejection, absent / not finite components) through ``assemble``.
- B: REAL detector calls on real OP-8 rows: each injection changes an input or an intermediate of the real computation (journal arrays,
  bound values, fields, grid check, previous factors, B_E inputs, floating-point mode of the separate process) and the detectors of
  the pipeline decide; expected UNRESOLVED_CERTIFICATE with the injected premise among the reasons; the clean twins (the same rows
  without injection) must have all premises established and E1/E2.
- V (В, positive): consistent perturbation of the factors (V, and the companion where it is nonzero), every dependent quantity
  recomputed by the pipeline; the change of the fields and of all dependent quantities is confirmed (values and hashes) before the
  control counts; with established premises E1/E2 must hold (E-2).
- G (Г, negative): mutations of the computation with the channel declared in the expectations; a mutation detected by no channel is
  a gap (UNRESOLVED of the control), never PASS; channels proposed in OD-4 keep the control UNRESOLVED until the owner's decision.
- BR adapter on the clean rows (OBSERVATION; budget 1/100, otherwise UNRESOLVED).

Usage (repository root): uv run python scripts/r5_cccc_q4_d1_controls_prelaunch.py run --out scratch/r5_q4_pipeline_controls
"""

import argparse
import hashlib
import importlib.util
import json
import math
import os
import shutil
import sys
import time
from fractions import Fraction
from pathlib import Path

THREAD_VARS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
for _k in THREAD_VARS:
    os.environ[_k] = "1"  # as in Q3, EQ, QA-EB, BR and the pipeline: one BLAS thread (set before NumPy is loaded; recorded)

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = ROOT / "tests" / "reference" / "r5_cccc_q4_d1_controls_prelaunch_expected.json"
EXPECTED_SHA256 = "939f3e339c4292e409bc1a6efd40afe2569b257842635c0b006a84f51da9f4aa"
PIPELINE_TOOL = "scripts/r5_cccc_q4_d1_row_pipeline_prelaunch.py"
ADAPTER_TOOL = "scripts/r5_cccc_q4_d1_bc_reference_adapter.py"
CERT = "UNRESOLVED_CERTIFICATE"
FAIL = "FAIL_COMPUTATION"


def _load(rel: str, name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def pl():
    return _load(PIPELINE_TOOL, "_pl_pipeline")


def ad():
    return _load(ADAPTER_TOOL, "_ad_adapter")


def load_expected(path: Path = EXPECTED) -> dict:
    raw = path.read_bytes().replace(b"\r\n", b"\n")
    if hashlib.sha256(raw).hexdigest() != EXPECTED_SHA256:
        raise RuntimeError("the expectations fixed before the code were modified (sha256 differs): controls are not run")
    return json.loads(raw)


def _cfg(spec: dict):
    return pl().op8().TrajectoryConfig(spec["lx"], spec["nx"], spec["ly"], spec["ny"], spec["N"], spec["D"], spec["q"], spec["s_max"])


# ---------------------------------------------------------------------------------------------------------------------------------
# A
# ---------------------------------------------------------------------------------------------------------------------------------

AP_KEYS = ("status", "e1", "e2", "e_diag", "reason_premises", "rejected", "source_undetermined")


def control_a(exp: dict) -> dict:
    p = pl()
    pr48 = p.pf().run_controls()
    cases = []
    for c in exp["pipeline_boundary_cases"]:
        prem = {k: {"state": "ESTABLISHED", "reason": "synthetic"} for k in p.pf().PREMISES}
        for k, v in c.get("premises", {}).items():
            prem[k] = {"state": v, "reason": "synthetic"}
        act = p.assemble(c["components"], prem, c.get("recorded"))
        got = {k: act.get(k) for k in c["expected"]}
        cases.append({"id": c["id"], "expected": c["expected"], "actual": got, "match": got == c["expected"],
                      "reasons": act["reasons"]})
    s = pr48["summary"]
    return {"pr48_summary": s, "pr48_all_match": bool(s["control_A_all_match"] and s["control_B_synthetic_all_match"]
                                                        and s["control_B_detector_functions_all_match"]),
            "boundary_cases": cases, "boundary_matching": sum(c["match"] for c in cases), "boundary_total": len(cases)}


# ---------------------------------------------------------------------------------------------------------------------------------
# rows and journals
# ---------------------------------------------------------------------------------------------------------------------------------

def make_journals(exp: dict, work: Path) -> dict:
    return {rid: pl().run_trajectory_journal(_cfg(spec), work / rid) for rid, spec in exp["rows"].items()}


def _spec(journals: dict, row: list, **kw) -> dict:
    return {"kind": "journal", "journal": str(journals[row[0]]), "s": row[1], **kw}


def _row(r: dict) -> dict:
    """The computed row, or an empty one for a row that was not executed (pipeline exception)."""
    return r.get("row") or {"diagnostics": {}}


def _premise_states(r: dict) -> dict:
    return {k: v["state"] for k, v in r["premises"].items()}


def _clean_ok(r: dict, ce: dict) -> tuple[bool, list[str]]:
    bad = []
    if ce["premises_all_established"] and any(v != "ESTABLISHED" for v in _premise_states(r).values()):
        bad.append(f"premises {_premise_states(r)}")
    c = r["classification"]
    if c["e1"] is not ce["e1"] or c["e2"] is not ce["e2"]:
        bad.append(f"E1/E2 {c['e1']}/{c['e2']}")
    if r["status"] not in ce["status_in"]:
        bad.append(f"status {r['status']}")
    if r["contradictions"] != ce["contradictions"]:
        bad.append(f"contradictions {r['contradictions']}")
    sp = _row(r)["diagnostics"].get("second_path_gamma", {}).get("consistent")
    if sp is not ce["second_path_gamma_consistent"]:
        bad.append(f"second path gamma {sp}")
    return not bad, bad


def _row_summary(r: dict) -> dict:
    c = r["classification"]
    return {"status": r["status"], "premises": _premise_states(r) if "premises" in r else None, "e1": c.get("e1"), "e2": c.get("e2"),
            "e_diag": c.get("e_diag"), "reason_premises": c.get("reason_premises"), "reasons": c.get("reasons"),
            "rejected": c.get("rejected"), "contradictions": r.get("contradictions"),
            "second_path_gamma": _row(r)["diagnostics"].get("second_path_gamma"),
            "reproduction": r.get("reproduction"), "values": {k: _row(r).get(k) for k in
                                                              ("P_hat_p", "P_hat_s", "Q_hat", "Gamma_up", "B_E_up")},
            "tau_exact": c.get("tau_exact"), "dP_exact": c.get("dP_exact")}


# ---------------------------------------------------------------------------------------------------------------------------------
# B
# ---------------------------------------------------------------------------------------------------------------------------------

def _tampered_journal(src: Path, dst: Path, key: str, how: str) -> Path:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    arrays = pl().op8().load_arrays(dst)
    a = arrays[key].copy()
    if how == "1ulp":
        a.flat[0] = math.nextafter(float(a.flat[0]), math.inf)
    elif how == "negzero":
        if a.flat[0] != 0:
            raise ValueError("negzero injection needs a zero entry")
        a.flat[0] = -0.0
    arrays[key] = a
    np.savez(dst / "arrays.npz", **arrays)
    return dst


def control_b(exp: dict, journals: dict, work: Path, clean: dict) -> dict:
    p = pl()
    out = []
    for c in exp["control_B"]:
        row = c["row"]
        twin = clean[(row[0], row[1])]
        if c["platform"] != "any" and sys.platform != c["platform"]:
            out.append({"id": c["id"], "premise": c["premise"], "result": "NOT_EXECUTED", "reason": f"needs {c['platform']}",
                        "match": None})
            continue
        inj = c["injection"]
        bound = None
        if inj.startswith("journal_array_"):
            how, key = inj[len("journal_array_"):].split(":")
            jd = _tampered_journal(journals[row[0]], work / f"{c['id']}_journal", key, how)
            spec = {"kind": "journal", "journal": str(jd), "s": row[1]}
        elif inj.startswith("recorded_value_1ulp:"):
            key = inj.split(":")[1]
            v = float.fromhex(twin["row"][key])
            bound = {key: math.nextafter(v, math.inf).hex()}
            spec = _spec(journals, row)
        elif inj.startswith("mode_"):
            spec = _spec(journals, row, mode=inj[len("mode_"):])
        else:
            spec = _spec(journals, row, injection=inj)
        r = p.analyse(spec, bound_values=bound)
        e = c["expected"]
        ok = (r["status"] == e["status"] and e["premise_in_reason_premises"] in r["classification"]["reason_premises"]
              and r["status"] not in e["not_status"])
        twin_ok, twin_bad = _clean_ok(twin, exp["clean_expected"])
        values_equal = {k: _row(r).get(k) == _row(twin).get(k) for k in ("P_hat_p", "P_hat_s", "Q_hat", "Gamma_up", "B_E_up")}
        out.append({"id": c["id"], "premise": c["premise"], "injection": inj, "actual": _row_summary(r), "match_injected": ok,
                    "clean_twin_match": twin_ok, "clean_twin_mismatch": twin_bad, "match": ok and twin_ok,
                    "values_equal_to_clean_twin": values_equal})
    return {"cases": out, "matching": sum(1 for x in out if x["match"]), "executed": sum(1 for x in out if x["match"] is not None),
            "not_executed": [x["id"] for x in out if x["match"] is None]}


# ---------------------------------------------------------------------------------------------------------------------------------
# V (В)
# ---------------------------------------------------------------------------------------------------------------------------------

def _perturbed(arrays: dict, s: int, perturb_companion: bool):
    given, comp, V = (arrays[f"s{s}_{k}"].copy() for k in ("solved_set", "companion_solved", "V_bc"))
    sign = np.where(np.arange(V.shape[0]) % 2 == 0, 1.0, -1.0)[:, None]
    V2 = V * (1.0 + 2.0**-20) + sign * (2.0**-30 * float(np.max(np.abs(V))))
    comp2 = comp * (1.0 + 2.0**-25) if perturb_companion else comp
    return given, comp2, V2


def control_v(exp: dict, journals: dict, work: Path, clean: dict) -> dict:
    p = pl()
    out = []
    for c in exp["control_V"]:
        row = c["row"]
        base = clean[(row[0], row[1])]
        _, arrays = p.load_journal(journals[row[0]])
        given, comp, V = _perturbed(arrays, row[1], "field_prev_sha" in c["expected"]["changed"])
        npz = work / f"{c['id']}_arrays.npz"
        sha = p.write_control_arrays(npz, given, comp, V)
        cfg = _cfg(exp["rows"][row[0]])
        spec = {"kind": "arrays", "arrays_npz": str(npz), "arrays_sha256": sha, "s": row[1],
                "config": {k: getattr(cfg, k) for k in ("lx", "nx", "ly", "ny", "N", "D", "q", "s_max")}}
        r = p.analyse(spec)
        e = c["expected"]

        def q(x, key):
            if key == "tau":
                return x["classification"]["tau_exact"]
            return _row(x).get("B_E_up" if key == "B_E" else key)

        changed = {k: q(r, k) != q(base, k) for k in e["changed"]}
        unchanged = {k: q(r, k) == q(base, k) for k in e.get("unchanged", [])}
        confirmed = all(changed.values()) and all(unchanged.values())
        prem_ok = all(v == "ESTABLISHED" for v in _premise_states(r).values())
        cl = r["classification"]
        if not confirmed:
            verdict = "UNRESOLVED (empty control: change not confirmed)"
        elif prem_ok and not (cl["e1"] and cl["e2"]):
            verdict = "FAIL (E1/E2 violated with established premises: BLOCKED_CAMPAIGN-equivalent finding)"
        elif not prem_ok:
            verdict = "UNRESOLVED (premises not established on the perturbed row)"
        elif r["status"] in e["status_in"] and r["contradictions"] == e["contradictions"]:
            verdict = "PASS"
        else:
            verdict = f"UNEXPECTED ({r['status']}, contradictions {r['contradictions']})"
        match = verdict == "PASS" and cl["e1"] is e["e1"] and cl["e2"] is e["e2"]
        out.append({"id": c["id"], "perturbation": c["perturbation"], "changed": changed, "unchanged": unchanged,
                    "change_confirmed": confirmed, "actual": _row_summary(r), "verdict": verdict, "match": match})
    return {"cases": out, "matching": sum(x["match"] for x in out)}


# ---------------------------------------------------------------------------------------------------------------------------------
# G (Г)
# ---------------------------------------------------------------------------------------------------------------------------------

def control_g(exp: dict, journals: dict, work: Path, clean: dict) -> dict:
    p = pl()
    out = []
    for c in exp["control_G"]:
        row = c["row"]
        base = clean[(row[0], row[1])]
        mut = c["mutation"]
        recorded, spec = None, None
        if mut.startswith("gamma_from_step:"):
            other = clean[(row[0], int(mut.split(":")[1]))]
            spec = _spec(journals, row, mutation="gamma_from_step", gamma_other=other["row"]["gamma_exact"])
        elif mut == "recorded_tau_double":
            spec = _spec(journals, row)
            recorded = {"tau": str(2 * Fraction(base["classification"]["tau_exact"]))}
        elif mut == "recorded_dP_offset":
            spec = _spec(journals, row)
            recorded = {"dP": str(Fraction(base["classification"]["dP_exact"]) + Fraction(1, 2**1080))}
        else:
            spec = _spec(journals, row, mutation=mut)
        r = p.analyse(spec, recorded=recorded)
        cl = r["classification"]
        prem_ok = all(v == "ESTABLISHED" for v in _premise_states(r).values())
        fired = {"E1/E2": bool(prem_ok and r["status"] == FAIL),
                 "second_path_gamma": _row(r)["diagnostics"].get("second_path_gamma", {}).get("consistent") is False,
                 "boundary_definition": bool(cl.get("rejected"))}
        e = c["expected"]
        if "detected_by" in e:
            detected = fired[e["detected_by"]]
            status_ok = e.get("status") is None or r["status"] == e["status"]
            rej_ok = e.get("rejected") is None or cl.get("rejected") is e["rejected"]
            match = detected and status_ok and rej_ok
            outcome = (f"DETECTED by {e['detected_by']}" if detected else "NOT DETECTED by the declared channel: gap (UNRESOLVED)")
        else:
            match = True  # G-5: no channel implemented; the expectation is the gap itself
            outcome = "GAP: no implemented channel (UNRESOLVED of control G)" + (
                f"; incidentally fired: {[k for k, v in fired.items() if v]}" if any(fired.values()) else "")
        out.append({"id": c["id"], "mutation": mut, "channel": c["channel"], "channel_status": c["channel_status"], "fired": fired,
                    "outcome": outcome, "actual": _row_summary(r), "match": bool(match)})
    gaps = [x["id"] for x in out if x["outcome"].startswith("GAP") or "gap" in x["outcome"]]
    proposed = [x["id"] for x in out if x["channel_status"].startswith("PROPOSED")]
    return {"cases": out, "matching": sum(x["match"] for x in out), "gaps": gaps, "channels_pending_OD4": proposed,
            "control_status": "UNRESOLVED (gaps and channels pending OD-4)" if gaps or proposed else "PASS"}


# ---------------------------------------------------------------------------------------------------------------------------------
# BR adapter on the clean rows
# ---------------------------------------------------------------------------------------------------------------------------------

def reference_rows(exp: dict, journals: dict) -> list[dict]:
    out = []
    for rid, s in exp["clean_rows"]:
        r = ad().reference_for_journal(journals[rid], s)
        out.append({"row": [rid, s], **{k: r.get(k) for k in ("result", "input_confirmed", "precision", "K_SPD_certified",
                                                             "lambda_low", "budget_set", "budget_field", "budget_inf",
                                                             "set_error_2_down", "set_error_2_up", "TS1_observation",
                                                             "exact_ref_within_bound", "exact_bc_error_in_interval", "rh_consistent",
                                                             "C_old_equals_C3plus_bitwise")}})
    return out


def run_all(work: Path) -> dict:
    exp = load_expected()
    p = pl()
    p.guard_out(work)
    work.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    journals = make_journals(exp, work)
    clean = {(rid, s): p.analyse(_spec(journals, [rid, s])) for rid, s in exp["clean_rows"]}
    clean_res = [{"row": [rid, s], "match": _clean_ok(r, exp["clean_expected"])[0], "mismatch": _clean_ok(r, exp["clean_expected"])[1],
                  "actual": _row_summary(r)} for (rid, s), r in clean.items()]
    a = control_a(exp)
    b = control_b(exp, journals, work, clean)
    v = control_v(exp, journals, work, clean)
    g = control_g(exp, journals, work, clean)
    refs = reference_rows(exp, journals)
    summary = {
        "control_A": {"pr48_all_match": a["pr48_all_match"], "boundary": f"{a['boundary_matching']}/{a['boundary_total']}",
                      "status": "PASS" if a["pr48_all_match"] and a["boundary_matching"] == a["boundary_total"] else "MISMATCH"},
        "clean_rows": f"{sum(x['match'] for x in clean_res)}/{len(clean_res)}",
        "control_B": {"matching": f"{b['matching']}/{b['executed']}", "not_executed": b["not_executed"],
                      "status": "PASS" if b["matching"] == b["executed"] and all(x["match"] for x in clean_res) else "MISMATCH"},
        "control_V": {"matching": f"{v['matching']}/{len(v['cases'])}", "status": "PASS" if v["matching"] == len(v["cases"]) else
                      "MISMATCH"},
        "control_G": {"matching": f"{g['matching']}/{len(g['cases'])}", "status": g["control_status"], "gaps": g["gaps"],
                      "pending_OD4": g["channels_pending_OD4"]},
        "reference_adapter": [x["result"] for x in refs],
    }
    return {"document": "R5-V Q4 PRELAUNCH: integrated controls of the N = 1 row pipeline", "data_status": p.DATA_STATUS,
            "expected_file": str(EXPECTED.relative_to(ROOT)).replace("\\", "/"), "expected_sha256": EXPECTED_SHA256,
            "provenance": p.provenance(), "seconds": round(time.perf_counter() - t0, 1), "summary": summary, "clean_rows": clean_res,
            "control_A": a, "control_B": b, "control_V": v, "control_G": g, "reference_adapter": refs}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    try:
        pl().guard_out(a.out)
    except pl().D1Refused as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    res = run_all(a.out)
    with open(a.out / "controls.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, ensure_ascii=False, default=pl()._json_default)
    print(json.dumps(res["summary"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
