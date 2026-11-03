"""R5-V Q4 QA-D1-READY: integrated controls A, B, V (В), G (Г), the Z-4 procedure, the rank protection of D-057.1 (RT-3, RT-15), the
forbidden configurations, the row plan, SP-OP, the block-C cross-check (OD-3), RT-17 and the platform rule, against the expectations
fixed BEFORE the code (tests/reference/r5_cccc_q4_d1_ready_controls_expected.json, its own commit; sha256 checked, a changed file
is refused).

Research harness on small non-D1 configurations: no D1 row, no campaign verdict, no D1_RESULT. Every case is executed on the
integrated layer (scripts/r5_cccc_q4_d1_ready.py) and the real tools; nothing is simulated except the synthetic rows of the
aggregation cases of control A (the classifier of PR #48 itself is re-run on its frozen harness unchanged). A mutation detected by
no channel is a gap of control G (UNRESOLVED of the control), never PASS.

Usage (repository root): uv run python scripts/r5_cccc_q4_d1_ready_controls.py run --out scratch/qa_d1_ready/controls
"""

import argparse
import hashlib
import importlib.util
import json
import math
import os
import shutil
import sys
from fractions import Fraction
from pathlib import Path

THREAD_VARS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
for _k in THREAD_VARS:
    os.environ[_k] = "1"

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = ROOT / "tests" / "reference" / "r5_cccc_q4_d1_ready_controls_expected.json"
EXPECTED_SHA256 = "b12f3157ad0f13d85dd46ef22d166fbacf981fef2de707215dbbda37d6eae89e"  # LF content, commit bbb90f6
READY_TOOL = "scripts/r5_cccc_q4_d1_ready.py"
OLD_CONTROLS = "scripts/r5_cccc_q4_d1_controls_prelaunch.py"
CERT, FAIL, UNRES = "UNRESOLVED_CERTIFICATE", "FAIL_COMPUTATION", "UNRESOLVED"


def _load(rel: str, name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def rd():
    return _load(READY_TOOL, "_rd_ready")


def pl():
    return rd().pl()


def load_expected(path: Path = EXPECTED) -> dict:
    raw = Path(path).read_bytes().replace(b"\r\n", b"\n")
    if hashlib.sha256(raw).hexdigest() != EXPECTED_SHA256:
        raise RuntimeError("the expectations fixed before the code were modified (sha256 differs): controls are not run")
    return json.loads(raw)


def _cfg(c: dict):
    return pl().op8().TrajectoryConfig(float(c["lx"]), int(c["nx"]), float(c["ly"]), int(c["ny"]), int(c["N"]), float(c["D"]),
                                       float(c["q"]), int(c["s_max"]))


def _states(r: dict) -> dict:
    return {k: v["state"] for k, v in (r.get("premises") or {}).items()}


def _summary(r: dict) -> dict:
    cl = r.get("classification") or {}
    row = r.get("row") or {}
    d = row.get("diagnostics") or {}
    return {"status": r.get("status"), "reason_premises": cl.get("reason_premises"), "e1": cl.get("e1"), "e2": cl.get("e2"),
            "premises": _states(r), "f2_defects": r.get("f2_defects"), "tool_defects": r.get("tool_defects"),
            "channels": r.get("channels"), "sp_gamma": {k: (d.get("sp_gamma") or {}).get(k) for k in ("inclusion", "tight")},
            "second_path_BE_match": (d.get("second_path_BE") or {}).get("match"),
            "contradictions": [x["source"] for x in r.get("contradictions") or []], "faults": [f["stage"] for f in r.get("faults") or []],
            "z4": (r.get("z4") or {}).get("outcome"), "not_executed_reason": r.get("not_executed_reason")}


# ---------------------------------------------------------------------------------------------------------------------------------
# rank protection (D-057.1), RT-3, RT-15, forbidden configurations, row plan
# ---------------------------------------------------------------------------------------------------------------------------------

def rank_guard(exp: dict) -> list[dict]:
    o, out = pl().op8(), []
    for c in exp["rank_guard"]:
        cfg = _cfg(exp["rows"][c["row"]])
        tr = o.run_trajectory(cfg)
        e, recs, arrs = c["expected"], tr.records, tr.arrays
        got = {"records": len(recs), "trajectory_status": tr.status}
        if "s2_status_prefix" in e:
            ok = (len(recs) == e["records"] and recs[0]["status"] == e["s1_status"] and recs[0].get("qr") is e["s1_qr"]
                  and str(recs[1]["status"]).startswith(e["s2_status_prefix"]) and recs[1].get("qr_executed") is False
                  and ("R" in arrs[1]) is e["s2_R_array"] and ("V_bc" in arrs[1]) is e["s2_solved"]
                  and tr.status.startswith(e["trajectory_status_prefix"]) and recs[1]["N"] == e["N_kept"]
                  and not str(recs[1]["status"]).startswith("STOP_BEFORE_QR: S-1") and "S-2" not in str(recs[1]["status"]).split(";")[0])
            got.update({"s1": recs[0]["status"], "s2": str(recs[1]["status"])[:80] if len(recs) > 1 else None})
        else:
            ok = (len(recs) == e["records"] and all(r["status"] == o.OK for r in recs) and tr.status.startswith(e["trajectory_status_prefix"])
                  and not any("R" in a for a in arrs))
        out.append({"id": c["id"], "actual": got, "match": bool(ok)})
    return out


def rt3(exp: dict) -> list[dict]:
    o, out = pl().op8(), []
    for c in exp["rt3"]:
        cfg = _cfg(c["config"])
        tr = o.run_trajectory(cfg)
        e, rec, arr = c["expected"], tr.records[0], tr.arrays[0]
        ok = (len(tr.records) == e["records"] and str(rec["status"]).startswith(e["s1_status_prefix"]) and "R" not in arr
              and "V_bc" not in arr and "C3plus" not in arr and rec.get("qr") is None)
        if "conditions_all_hold" in e:
            cond = rec.get("config_S1_RN3") or {}
            ok = ok and cond.get("all_hold") is True and cond.get("n_s") == e["n_s"] and cond.get("floor_n_s_half") == e["floor_n_s_half"] \
                and tr.status.startswith(e["trajectory_status_prefix"])
        out.append({"id": c["id"], "actual": {"status": str(rec["status"])[:90], "records": len(tr.records),
                                              "conditions": rec.get("config_S1_RN3")}, "match": bool(ok)})
    return out


def rt15(exp: dict, work: Path, journals: dict, paired: dict) -> list[dict]:
    R, out = rd(), []
    for c in exp["rt15_rows"]:
        e = c["expected"]
        if "row" in c:
            rows = [R.analyse_ready({"kind": "journal", "journal": str(journals[c["row"]]), "s": s}) for s in range(1, 9)]
            r1, rest = rows[0], rows[1:]
            ok = (r1["status"] in e["s1_status_in"] and all(v == "ESTABLISHED" for v in _states(r1).values()) and bool(_states(r1))
                  and all(r["status"] == e["s2_to_s8_status"] and e["s2_to_s8_reason_contains"] in (r.get("not_executed_reason") or "")
                          for r in rest) and not any(r["status"] in e["never"] for r in rest))
            actual = [r["status"] for r in rows]
        elif "config" in c:
            cfg = _cfg(next(x for x in exp["rt3"] if x["id"] == c["config"])["config"])
            jd = pl().run_trajectory_journal(cfg, work / f"rt15_{c['config']}")
            rows = [R.analyse_ready({"kind": "journal", "journal": str(jd), "s": s}) for s in range(1, 9)]
            ok = all(r["status"] == e["s1_to_s8_status"] for r in rows) and e["s1_reason_contains"] in (rows[0].get("not_executed_reason") or "")
            actual = [r["status"] for r in rows]
        else:
            doc = paired[c["paired"]]
            rows = []
            for x in doc["rows"]:
                rows.append(R.analyse_ready(x["spec"]) if x["executed"] else {"status": UNRES, "not_executed_reason": x["reason"]})
            r1, rest = rows[0], rows[1:]
            chain = json.loads((Path(doc["dir"]) / "chain.json").read_text(encoding="utf-8"))
            crec = {r["s"]: r for r in chain["records"]}
            ok = (r1["status"] in e["s1_status_in"] and all(v == "ESTABLISHED" for v in _states(r1).values())
                  and all(r["status"] == e["s2_to_s8_status"] and e["s2_to_s8_reason_contains"] in (r.get("not_executed_reason") or "")
                          for r in rest)
                  and crec[1]["status"] == "OK" and crec[1]["qr_transform"] is False
                  and all(str(crec[s]["status"]).startswith(e["rac_chain"]["s2_to_s8"]) for s in range(2, 9)))
            actual = [r["status"] for r in rows]
        out.append({"id": c["id"], "actual": actual, "match": bool(ok)})
    return out


def forbidden(exp: dict, work: Path, journals: dict) -> list[dict]:
    p, R, o = pl(), rd(), pl().op8()
    TC = o.TrajectoryConfig
    calls = []

    def sensor(*a, **k):
        calls.append("computation")
        raise AssertionError("computation reached")

    out = []

    def refused(fn) -> bool:
        before = len(calls)
        try:
            fn()
        except (p.D1Refused, R.D1Blocked):
            return len(calls) == before
        return False

    orig_core = p._compute_row_core
    p._compute_row_core = sensor
    try:
        z = np.zeros((9, 2))
        res = {
            "FB-1": refused(lambda: p.compute_row(TC(1.0, 8, 1.25, 10, 2, s_max=2), 2, np.zeros((7, 2)), z, z)),
            "FB-2": refused(lambda: p.compute_row(TC(1.0, 8, 1.25, 10, 3, s_max=3), 3, np.zeros((9, 3)), np.zeros((7, 3)), np.zeros((7, 3)))),
            "FB-3": refused(lambda: p.compute_row(TC(1.0, 50, 1.0, 50, 1, s_max=1), 1, np.zeros((49, 1)), np.zeros((49, 1)), np.zeros((49, 1)))),
            "FB-4": refused(lambda: R.analyse_ready({"kind": "paired", "paired_npz": str(work / "absent.npz"), "paired_sha256": {},
                                                     "chain_json": str(work / "absent.json"), "s": 2,
                                                     "config": {"lx": 1.0, "nx": 8, "ly": 1.25, "ny": 10, "N": 2, "D": 1.0, "q": 1.0, "s_max": 8}})),
            "FB-5": refused(lambda: R.bcd().write_paired(TC(1.0, 100, 1.0, 100, 1, s_max=8), work / "fb5")),
            "FB-6": refused(lambda: R.campaign("A")),
        }
    finally:
        p._compute_row_core = orig_core
    r = o.run_step(TC(1.0, 8, 1.25, 10, 2, s_max=2), 2, np.random.default_rng(1).standard_normal((9, 2)), np.ones((7, 2)))
    res["FB-7"] = (str(r.record["status"]).startswith(o.RANK_GUARD) and "R" not in r.arrays and "C3plus" not in r.arrays
                   and "V_bc" not in r.arrays and r.next_given is None)
    for c in exp["forbidden"]:
        out.append({"id": c["id"], "call": c["call"], "match": bool(res[c["id"]])})
    return out


# ---------------------------------------------------------------------------------------------------------------------------------
# control A: PR #48 harness unchanged + aggregation after Z-4
# ---------------------------------------------------------------------------------------------------------------------------------

def _syn(status: str) -> dict:
    est = {p: {"state": "ESTABLISHED"} for p in ("P-a", "P-b", "P-c", "P-d", "P-e", "P-f")}
    if status == "PASS":
        return {"executed": True, "premises": est, "dP": Fraction(1), "Q": 1.0, "tau": Fraction(1, 4)}
    if status == "UNRESOLVED_RESOLVABILITY":
        return {"executed": True, "premises": est, "dP": Fraction(1), "Q": 1.0, "tau": Fraction(2)}
    if status == "FAIL_COMPUTATION":
        return {"executed": True, "premises": est, "dP": Fraction(0), "Q": 1.0, "tau": Fraction(1, 2)}
    return {"executed": False, "not_executed": {"reason": status}}


def control_a(exp: dict) -> dict:
    R = rd()
    f = R.pf()
    pr48 = f.run_controls()
    mand_rows = f.mandatory_rows()
    mand = f.mandatory_index(mand_rows)
    keys = [r["key"] for r in mand_rows]
    k0 = keys[0]

    def rows(spec: dict, extra=()):
        out = []
        for k in keys:
            blk, N, s = mand[k]
            st, z4, td = spec.get(k, ("PASS", None, []))
            out.append({"key": k, "block": blk, "N": N, "s": s, "mutated": False, "input": _syn(st), "mandatory": True,
                        "z4": z4, "tool_defects": td})
        return out + list(extra)

    conf = {"outcome": "CONFIRMED"}
    b_fail = {"key": "D1-B|r=1|50x50|N2|s1", "block": "B", "N": 2, "s": 1, "mutated": False, "input": _syn("FAIL_COMPUTATION"),
              "mandatory": False, "z4": conf, "tool_defects": []}
    guarded = [{"key": f"D1-B|r=1|50x50|N2|s{s}", "block": "B", "N": 2, "s": s, "mutated": False, "mandatory": False,
                "input": _syn("not executed: STOPPED_RANK_GUARD_UNCERTIFIED"), "z4": None, "tool_defects": []} for s in range(2, 9)]
    cases = {"AZ-1": rows({}), "AZ-2": rows({k0: ("PASS", None, ["DEFECT_GAMMA_NARROWNESS"])}),
             "AZ-3": rows({k0: ("UNRESOLVED_RESOLVABILITY", conf, [])}), "AZ-4": rows({k0: ("UNRESOLVED_RESOLVABILITY", None, [])}),
             "AZ-5": rows({}, [b_fail]), "AZ-6": rows({k0: ("UNRESOLVED_RESOLVABILITY", {"outcome": "BINDING_MISMATCH"}, [])}),
             "AZ-7": rows({k0: ("UNRESOLVED_RESOLVABILITY", {"outcome": "DEFECT_GAMMA_INCLUSION"}, [])}), "AZ-8": rows({}, guarded)}
    out = []
    for c in exp["control_A"]["z4_aggregation"]:
        v = R.verdict(cases[c["id"]], mand)
        e = c["expected"]
        ok = v["verdict"] == e["verdict"]
        if "classifier_verdict" in e:
            ok = ok and v["classifier"]["campaign_verdict"] == e["classifier_verdict"]
        if "row_status_after_z4" in e:
            ok = ok and k0 in v["classifier"].get("unresolved_keys", []) and \
                any(k == k0 and st == e["row_status_after_z4"] for k, st, _ in v["classifier"].get("unresolved_all", []))
        if "denominator" in e:
            ok = ok and v["classifier"].get("denominator") == e["denominator"]
        out.append({"id": c["id"], "verdict": v["verdict"], "classifier": v["classifier"]["campaign_verdict"], "match": bool(ok)})
    sm = pr48["summary"]
    allm = bool(sm["control_A_all_match"] and sm["control_B_synthetic_all_match"] and sm["control_B_detector_functions_all_match"]
                and sm["row_cases_matching"] == sm["row_cases"] and sm["aggregation_matching"] == sm["aggregation_cases"])
    return {"pr48_harness": sm, "pr48_all_match": allm, "z4_aggregation": out}


# ---------------------------------------------------------------------------------------------------------------------------------
# rows: journals, paired chains, clean rows, B, V, G, Z-4
# ---------------------------------------------------------------------------------------------------------------------------------

def make_inputs(exp: dict, work: Path):
    journals = {rid: pl().run_trajectory_journal(_cfg(c), work / rid) for rid, c in exp["rows"].items()}
    paired = {}
    for pid, c in exp["paired"].items():
        d = work / pid
        doc = rd().bcd().write_paired(_cfg(c), d)
        doc["dir"] = str(d)
        paired[pid] = doc
    return journals, paired


def _spec(journals, paired, row, **kw):
    rid, s = row
    if rid in paired:
        x = next(x for x in paired[rid]["rows"] if x["s"] == s)
        return {**x["spec"], **kw}
    return {"kind": "journal", "journal": str(journals[rid]), "s": s, **kw}


def _clean_ok(r: dict, ce: dict) -> tuple[bool, list[str]]:
    bad = []
    st = _states(r)
    if not st or any(v != "ESTABLISHED" for v in st.values()):
        bad.append(f"premises {st}")
    cl = r.get("classification") or {}
    if cl.get("e1") is not True or cl.get("e2") is not True:
        bad.append("E1/E2")
    if r.get("status") not in ce["status_in"]:
        bad.append(f"status {r.get('status')}")
    if r.get("f2_defects"):
        bad.append(f"F-2 {r['f2_defects']}")
    d = (r.get("row") or {}).get("diagnostics") or {}
    if (d.get("sp_gamma") or {}).get("inclusion") is not True or (d.get("sp_gamma") or {}).get("tight") is not True:
        bad.append("SP-GAMMA")
    if (d.get("second_path_BE") or {}).get("match") is not True:
        bad.append("second path B_E")
    if r.get("contradictions") or r.get("faults"):
        bad.append(f"contradictions/faults {r.get('contradictions')} {r.get('faults')}")
    return not bad, bad


def clean_rows(exp, journals, paired) -> dict:
    out = {}
    for row in exp["clean_rows"]:
        r = rd().analyse_ready(_spec(journals, paired, row), sp_be=True)
        ok, bad = _clean_ok(r, exp["clean_expected"])
        out[tuple(row)] = {"result": r, "match": ok, "mismatch": bad}
    return out


def _tamper_npz(path: Path, dst: Path, key: str) -> Path:
    with np.load(path) as z:
        arr = {k: z[k] for k in z.files}
    a = arr[key].copy()
    a.flat[0] = math.nextafter(float(a.flat[0]), math.inf) if a.flat[0] != 0 else math.nextafter(0.0, 1.0)
    arr[key] = a
    np.savez(dst, **arr)
    return dst


def control_b(exp, journals, paired, work, clean) -> list[dict]:
    old = _load(OLD_CONTROLS, "_ct_old_controls")
    out = []
    for c in exp["control_B"]:
        row = c["row"]
        if c.get("platform") and sys.platform != c["platform"]:
            out.append({"id": c["id"], "match": None, "result": f"NOT_EXECUTED (needs {c['platform']})"})
            continue
        inj = c["injection"]
        if inj.startswith("journal_array_"):
            how, key = inj[len("journal_array_"):].split(":")
            jd = old._tampered_journal(journals[row[0]], work / f"{c['id']}_journal", key, how)
            spec = {"kind": "journal", "journal": str(jd), "s": row[1]}
        elif inj.startswith("paired_array_1ulp:"):
            key = inj.split(":")[1]
            base = _spec(journals, paired, row)
            npz = _tamper_npz(Path(base["paired_npz"]), work / f"{c['id']}.npz", key)
            spec = {**base, "paired_npz": str(npz)}
        elif inj.startswith("mode_"):
            spec = _spec(journals, paired, row, mode=inj[len("mode_"):])
        else:
            spec = _spec(journals, paired, row, injection=inj)
        r = rd().analyse_ready(spec)
        e = exp["control_B_expected"]
        ok = (r["status"] == e["status"] and c["premise"] in ((r.get("classification") or {}).get("reason_premises") or [])
              and r["status"] not in e["not_status"])
        twin = clean[tuple(row)]["match"]
        out.append({"id": c["id"], "injection": inj, "premise": c["premise"], "actual": _summary(r), "clean_twin_match": twin,
                    "match": bool(ok and twin)})
    return out


def _perturbed(given, comp, V, perturb_companion: bool):
    sign = np.where(np.arange(V.shape[0]) % 2 == 0, 1.0, -1.0)[:, None]
    V2 = V * (1.0 + 2.0**-20) + sign * (2.0**-30 * float(np.max(np.abs(V))))
    comp2 = comp * (1.0 + 2.0**-25) if perturb_companion else comp
    return given, comp2, V2


def _arrays_of(journals, paired, row):
    rid, s = row
    if rid in paired:
        x = next(x for x in paired[rid]["rows"] if x["s"] == s)
        with np.load(x["spec"]["paired_npz"]) as z:
            return z["solved_set"], z["companion_solved"], z["V"], x["spec"]["config"]
    j, arrays = pl().load_journal(journals[rid])
    return arrays[f"s{s}_solved_set"], arrays[f"s{s}_companion_solved"], arrays[f"s{s}_V_bc"], j["meta"]["config"]


def control_v(exp, journals, paired, work, clean) -> list[dict]:
    p, out = pl(), []
    E = exp["control_V_expected"]
    for c in exp["control_V"]:
        row = c["row"]
        base = clean[tuple(row)]["result"]
        given, comp, V, cfgd = _arrays_of(journals, paired, row)
        g2, c2, V2 = _perturbed(given, comp, V, "field_prev_sha" in c["changed"])
        npz = work / f"{c['id']}.npz"
        sha = p.write_control_arrays(npz, g2, c2, V2)
        spec = {"kind": "arrays", "arrays_npz": str(npz), "arrays_sha256": sha, "s": row[1], "config": cfgd}
        r = rd().analyse_ready(spec, sp_be=True)

        def q(x, k):
            return (x.get("classification") or {}).get("tau_exact") if k == "tau" else (x.get("row") or {}).get(k)

        changed = {k: q(r, k) != q(base, k) for k in c["changed"]}
        unchanged = {k: q(r, k) == q(base, k) for k in c.get("unchanged", [])}
        confirmed = all(changed.values()) and all(unchanged.values())
        ok, bad = _clean_ok(r, E)
        verdict = ("UNRESOLVED (empty control)" if not confirmed else "PASS" if ok else f"UNEXPECTED {bad}")
        out.append({"id": c["id"], "changed": changed, "unchanged": unchanged, "actual": _summary(r), "verdict": verdict,
                    "match": verdict == "PASS"})
    return out


def control_g(exp, journals, paired, work, clean) -> list[dict]:
    out = []
    for c in exp["control_G"]:
        row, mut = c["row"], c["mutation"]
        base = clean[tuple(row)]["result"]
        recorded = None
        if mut.startswith("gamma_from_step_arrays:"):
            spec = _spec(journals, paired, row, mutation="gamma_from_step_arrays", gamma_other_step=int(mut.split(":")[1]))
        elif mut == "recorded_tau_double":
            spec = _spec(journals, paired, row)
            recorded = {"tau": str(2 * Fraction(base["classification"]["tau_exact"]))}
        else:
            spec = _spec(journals, paired, row, mutation=mut)
        e = c["expected"]
        r = rd().analyse_ready(spec, recorded=recorded, sp_be=True, force_z4="z4" in e)
        detected = e["detected_by"] in (r.get("channels") or [])
        ok = detected
        if e.get("status") == "same as the clean twin":
            ok = ok and r["status"] == base["status"]
        elif e.get("status"):
            ok = ok and r["status"] == e["status"]
        if e.get("premise"):
            ok = ok and e["premise"] in ((r.get("classification") or {}).get("reason_premises") or [])
        if "rejected" in e:
            ok = ok and (r.get("classification") or {}).get("rejected") is e["rejected"]
        if "z4" in e:
            ok = ok and (r.get("z4") or {}).get("outcome") == e["z4"]
        out.append({"id": c["id"], "mutation": mut, "channel": c["channel"], "detected": detected,
                    "outcome": "DETECTED" if detected else "NOT DETECTED by the declared channel: gap (UNRESOLVED of control G)",
                    "actual": _summary(r), "match": bool(ok)})
    return out


def z4_procedure(exp, journals, paired, work, clean) -> list[dict]:
    R, old, out = rd(), _load(OLD_CONTROLS, "_ct_old_controls"), []
    muts = {"gamma_function_sign_flip": "gamma_function_sign_flip", "gamma_interval_widened": "gamma_interval_widened",
            "B_E_function_drop_B1": "B_E_function_drop_B1"}
    for c in exp["z4_procedure"]:
        row, e = c["row"], c["expected"]
        base = clean[tuple(row)]["result"]
        if c["case"] in muts:
            r = R.analyse_ready(_spec(journals, paired, row, mutation=muts[c["case"]]), force_z4=True)
            z, st = r["z4"], r["status"]
        elif c["case"].startswith("binding repeated"):
            src = work / f"{c['id']}_journal"
            if src.exists():
                shutil.rmtree(src)
            shutil.copytree(journals[row[0]], src)
            spec = {"kind": "journal", "journal": str(src), "s": row[1]}
            first = R.analyse_ready(spec)
            old._tampered_journal(src, work / f"{c['id']}_tmp", f"s{row[1]}_given_in", "1ulp")
            shutil.rmtree(src)
            shutil.copytree(work / f"{c['id']}_tmp", src)
            z = R.z4_confirm(spec, first)
            st = R._reassemble(first, [f"Z-4: {x}" for x in z["binding_differences"]], None)["status"] \
                if z["outcome"] == "BINDING_MISMATCH" else first["status"]
        else:
            r = R.analyse_ready(_spec(journals, paired, row), force_z4=True)
            z, st = r["z4"], r["status"]
        ok = z["outcome"] == e["outcome"]
        if e.get("status") == "same as the clean twin":
            ok = ok and st == base["status"]
        elif e.get("status"):
            ok = ok and st == e["status"]
        out.append({"id": c["id"], "outcome": z["outcome"], "status": st, "checks": z["checks"], "match": bool(ok)})
    return out


def sp_op(exp, journals) -> list[dict]:
    R, out = rd(), []
    from kkvim_reference import r5_cccc_bc as bc
    from kkvim_reference import r5_cccc_halfstep as hs

    for c in exp["sp_op"]:
        rid, s = c["row"]
        j, arrays = pl().load_journal(journals[rid])
        cfg = pl().config_of(j["meta"])
        U = arrays[f"s{s}_solved_set"]
        override = None
        if c.get("injection"):
            p = hs.HalfStep(float(cfg.lx), cfg.nx, float(cfg.ly), cfg.ny, pl().op8().direction_of(s), U, D=float(cfg.D), q=float(cfg.q))
            A, B, C, F = bc.bc_coefficients(p)
            fr = lambda M: [[Fraction(float(v)) for v in r] for r in np.atleast_2d(M)]  # noqa: E731
            override = (fr(A), fr(B), fr(C), [Fraction(float(v)) for v in np.atleast_1d(F)])
        r = R.sp_op_check(cfg, s, U, coeffs_override=override)
        want = c["expected"].split()[0]
        out.append({"id": c["id"], "status": r["status"], "entries_checked": r["entries_checked"], "match": r["status"] == want})
    return out


def block_c(exp) -> list[dict]:
    out = []
    for c in exp["block_C_crosscheck"]:
        r = rd().bcd().crosscheck(_cfg(exp["paired"][c["paired"]]))
        e = c["expected"]
        ok = [x["s"] for x in r["rows"]] == e["s"] and r["all_bitwise"] is e["EQ_equals_OP8_bitwise"] and \
            all(x["input_is_RAC_state"] for x in r["rows"])
        if "blas_threads_read" in e:
            ok = ok and r["blas_threads_read"] == e["blas_threads_read"]
        if "rac_s1_qr_transform" in e:
            chain = rd().bcd().rac_chain(_cfg(exp["paired"][c["paired"]]))
            ok = ok and chain["records"][0]["qr_transform"] is e["rac_s1_qr_transform"]
        out.append({"id": c["id"], "rows": [(x["s"], x["bitwise"]) for x in r["rows"]], "blas_threads_read": r["blas_threads_read"],
                    "match": bool(ok)})
    return out


def rt17(exp) -> dict:
    R = rd()
    plan = R.row_plan()
    geo_bad = [r["key"] for r in plan if R.geometry_check(r)]
    e = exp["rt17"]["set_inf_norm"]
    v = R.set_inf_norm(e["V"])
    x = np.array([[0.5], [-2.0], [1.0]])
    n1 = R.set_inf_norm(x) == float(np.linalg.norm(x[:, 0], np.inf))
    bc1 = R.bc1_record(np.array(e["V"]), np.zeros((2, 2)), np.eye(2))
    return {"geometry_violations": geo_bad, "set_inf_norm": v, "N1_equals_vector_norm": n1, "basis_recorded": len(bc1["basis_sha"]) == 64,
            "match": not geo_bad and v == e["expected"] and n1 is e["N1_equals_vector_norm"] and len(bc1["basis_sha"]) == 64}


def platform_rule(exp) -> dict:
    R = rd()
    rec = R.platform_record()
    fr = R.freeze_allowed(rec)
    a = exp["platform"]["approved"]
    should = rec["sys_platform"] == a["sys_platform"] and rec["python"] == a["python"] and rec["blas_threads_read_from_library"] == 1
    alien = R.freeze_allowed({**rec, "python": "3.14.8"})
    return {"platform": {k: rec[k] for k in ("sys_platform", "python", "numpy", "scipy", "blas_threads_read_from_library")},
            "freeze": fr, "match": fr["allowed"] is should and alien["allowed"] is False and alien["state"].startswith("UNRESOLVED")}


def run_all(work: Path) -> dict:
    exp = load_expected()
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    journals, paired = make_inputs(exp, work)
    clean = clean_rows(exp, journals, paired)
    res = {"data_status": rd().DATA_STATUS, "expected_sha256": EXPECTED_SHA256,
           "rank_guard": rank_guard(exp), "rt3": rt3(exp), "rt15": rt15(exp, work, journals, paired),
           "forbidden": forbidden(exp, work, journals), "row_plan": {"actual": rd().plan_counts(rd().row_plan())},
           "control_A": control_a(exp),
           "clean_rows": [{"row": list(k), "match": v["match"], "mismatch": v["mismatch"], "status": v["result"]["status"]}
                          for k, v in clean.items()],
           "control_B": control_b(exp, journals, paired, work, clean), "control_V": control_v(exp, journals, paired, work, clean),
           "control_G": control_g(exp, journals, paired, work, clean), "z4_procedure": z4_procedure(exp, journals, paired, work, clean),
           "sp_op": sp_op(exp, journals), "block_C_crosscheck": block_c(exp), "rt17": rt17(exp), "platform": platform_rule(exp),
           "provenance": rd().provenance()}
    res["row_plan"]["match"] = res["row_plan"]["actual"] == exp["row_plan"]
    sections = ("rank_guard", "rt3", "rt15", "forbidden", "clean_rows", "control_B", "control_V", "control_G", "z4_procedure", "sp_op",
                "block_C_crosscheck")
    summary = {}
    for k in sections:
        xs = res[k]
        summary[k] = {"cases": len(xs), "match": sum(x["match"] is True for x in xs), "not_executed": sum(x["match"] is None for x in xs),
                      "mismatch": [x.get("id", x.get("row")) for x in xs if x["match"] is False]}
    summary["control_A"] = {"pr48_harness_all_match": res["control_A"]["pr48_all_match"],
                            "z4_aggregation_match": sum(x["match"] for x in res["control_A"]["z4_aggregation"]),
                            "z4_aggregation_cases": len(res["control_A"]["z4_aggregation"]),
                            "mismatch": [x["id"] for x in res["control_A"]["z4_aggregation"] if not x["match"]]}
    for k in ("row_plan", "rt17", "platform"):
        summary[k] = {"match": res[k]["match"]}
    res["summary"] = summary
    res["all_match"] = all(not v.get("mismatch") for v in summary.values() if "mismatch" in v) and \
        summary["control_A"]["pr48_harness_all_match"] and all(summary[k]["match"] for k in ("row_plan", "rt17", "platform"))
    return res


def summary_doc(res: dict) -> dict:
    """The compact record of a controls run (published under docs/results/r5_cccc_q4_d1_ready/controls_summary.json)."""
    prov = res["provenance"]
    return {"document": "R5-V Q4 QA-D1-READY integrated controls - summary (protocol 6, 12 item 4; D-057)", "data_status": res["data_status"],
            "expected_sha256": res["expected_sha256"], "all_match": res["all_match"], "summary": res["summary"],
            "control_G": [{k: x[k] for k in ("id", "mutation", "channel", "detected", "match")} | {"status": x["actual"]["status"],
                                                                                            "channels": x["actual"]["channels"]}
                          for x in res["control_G"]],
            "z4_procedure": [{k: x[k] for k in ("id", "outcome", "status", "match")} for x in res["z4_procedure"]],
            "control_B": [{"id": x["id"], "premise": x.get("premise"), "match": x["match"],
                           "status": (x.get("actual") or {}).get("status")} for x in res["control_B"]],
            "row_plan": res["row_plan"], "platform": res["platform"]["platform"], "freeze": res["platform"]["freeze"],
            "hash_platform": rd().pf().platform_fingerprint(), "tool_blobs": prov["tool_blobs"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    pl().guard_out(a.out)
    res = run_all(a.out / "work")
    with open(a.out / "controls_results.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, ensure_ascii=False, default=str)
    with open(a.out / "controls_summary.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(summary_doc(res), f, indent=1, ensure_ascii=False, default=str)
        f.write("\n")
    print(json.dumps({"all_match": res["all_match"], "summary": res["summary"]}, default=str))
    return 0 if res["all_match"] else 1


if __name__ == "__main__":
    sys.exit(main())
