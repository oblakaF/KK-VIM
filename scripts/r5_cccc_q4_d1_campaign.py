"""R5-V Q4 D1 campaign runner (D-059, C3; protocol docs/R5_CCCC_Q4_D1_PROTOCOL.md 3-6, 8, 12, 13). QA tool: it connects the
approved components and adds no numerical algorithm, formula, threshold, tolerance or status. No import of the working ``kkvim``.

What it connects (all read-only):
- the row plan of D1 (``ready.row_plan``: 824 classified rows, 102 mandatory block-A keys; 168 R-AC half-steps of block C that are
  NOT campaign rows; F2: 497 positions N >= 2, s >= 2 = 399 classified rows + 98 R-AC half-steps, never computed);
- OP-8 journals (``pipeline.run_trajectory_journal``; the rank guard of D-057.1 stops N = 2, 3 before the QR of s = 2) and the
  block-C driver (``write_paired``: R-AC chain and paired half-steps);
- each executable row through ``ready.analyse_ready`` (binding, separate-process replay, F-2, F-3, SP-GAMMA, SP-BE, Z-4);
- observations outside the verdict: SP-OP (``ready.sp_op_check``), SP-BR (reference adapter, ``reference_for_journal``); SP-RH,
  SP-LARGE, SP-C have no row-level implementation in the frozen tools and are recorded UNRESOLVED (never PASS);
- the controls (PR #48 harness and the QA-D1-READY controls A/B/V/G against their frozen expectations, run in a child process
  WITHOUT the launch signal) and the s = 1 input hashes / start archive before the first row;
- the verdict: ``ready.verdict`` (unchanged Z-4 rule) with ``input_hash_mismatch`` and ``control_v_fail`` passed to the unchanged
  ``campaign_verdict`` of PR #48 (F4: BLOCKED_CAMPAIGN).

Execution conditions (as the accepted PL 6.3 probes): every job (journal, block-C configuration, row) runs in its own child process
with OPENBLAS/OMP/MKL_NUM_THREADS = 1 set by this launcher before NumPy is imported and the memory watchdog of the probes
(``preflight._watchdog``) at 50 % of the physical RAM (PL 6.3); an aborted job makes its rows UNRESOLVED_RESOURCES with the measured
values (classify_row of PR #48). The replay processes started by ``analyse`` are not under this watchdog (limitation, recorded).

Gate: a real run (``dry_run`` False) is refused before anything is computed unless ``pipeline.campaign_gate`` authorises it (D-059:
launch signal + owner's re-freeze act + addendum + frozen manifest + tool blobs), the platform is the approved one (Windows,
Python 3.13.15, one BLAS thread read from the libraries of NumPy and SciPy) and the plan equals the frozen manifest. Every child that
touches a D1 grid validates the gate again in its own process. ``dry_run`` accepts only non-D1 grids (tests: launch, interruption,
restart); its data is never D1 data. Output only under scratch/ (no published data during the run).

Usage (repository root; refused while D1_START = BLOCKED):
    KKVIM_D1_CAMPAIGN_AUTH=D-059:<addendum blob> OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
        python scripts/r5_cccc_q4_d1_campaign.py run --out scratch/r5_cccc_q4_d1_campaign
"""

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
from fractions import Fraction
from pathlib import Path

THREAD_VARS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
for _k in THREAD_VARS:
    os.environ[_k] = "1"  # OD-6: before NumPy is imported (the children get it from this launcher as well)

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SELF = "scripts/r5_cccc_q4_d1_campaign.py"
READY_TOOL = "scripts/r5_cccc_q4_d1_ready.py"
CONTROLS_TOOL = "scripts/r5_cccc_q4_d1_ready_controls.py"
ADAPTER_TOOL = "scripts/r5_cccc_q4_d1_bc_reference_adapter.py"
DATA_STATUS_CAMPAIGN = ("D1 CAMPAIGN DATA (D-059; protocol 12 item 6): valid only under the owner's re-freeze act recorded in "
                        "docs/DECISIONS.md and the addendum it names")
DATA_STATUS_DRY = "DRY RUN on non-D1 grids (D-059 tests): NOT D1 data, never D1_RESULT"
NOT_IMPLEMENTED_OBS = ("UNRESOLVED: no row-level implementation of this second path in the frozen tool set (D-059: no new numerical "
                       "algorithm); not performed")
LAUNCH_ENV = {k: "1" for k in THREAD_VARS}


class CampaignRefused(RuntimeError):
    """The campaign is refused before any computation (gate, platform, plan, output, resume provenance)."""


class CampaignInterrupted(RuntimeError):
    """Test hook: the run stopped after the requested number of jobs (state kept for the restart)."""


def _load(rel: str, name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def rd():
    return _load(READY_TOOL, "_d1_ready")


def pl():
    return rd().pl()


def pf():
    return rd().pf()


def bcd():
    return rd().bcd()


def adapter():
    return _load(ADAPTER_TOOL, "_d1_br_adapter")


def _dump(o) -> str:
    return json.dumps(o, default=pl()._json_default, ensure_ascii=False)


def _write_json(path: Path, o) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(o, default=pl()._json_default, ensure_ascii=False, indent=1) + "\n")
    os.replace(tmp, path)


# ---------------------------------------------------------------------------------------------------------------------------------
# the exact classifier input (C1 returns it; it is stored as text and read back without any reconstruction)
# ---------------------------------------------------------------------------------------------------------------------------------

def encode_input(inp: dict) -> dict:
    """JSON form of the exact classifier input: Fractions as 'p/q' (``pipeline.ftext``), binary64 values as JSON numbers (repr:
    exact round trip), nothing else touched."""
    return json.loads(_dump(inp))


def decode_input(d: dict) -> dict:
    """Inverse of ``encode_input``: the exact rationals dP and tau back as Fractions."""
    out = json.loads(json.dumps(d))
    for k in ("dP", "tau"):
        if isinstance(out.get(k), str):
            out[k] = Fraction(out[k])
    return out


# ---------------------------------------------------------------------------------------------------------------------------------
# plan and its accounting (F2)
# ---------------------------------------------------------------------------------------------------------------------------------

def plan_accounting(plan: list[dict]) -> dict:
    cls = [r for r in plan if r["kind"] == "row"]
    rac = [r for r in plan if r["kind"] == "rac"]

    def n23(xs):
        return [r for r in xs if r["N"] >= 2 and r["s"] >= 2]

    return {"classified_rows": len(cls), "mandatory_keys": sum(r["mandatory"] for r in cls),
            "classified_executable": sum(r["executable"] for r in cls), "classified_not_executed": sum(not r["executable"] for r in cls),
            "rac_half_steps_not_campaign_rows": len(rac), "rac_executable": sum(r["executable"] for r in rac),
            "n23_s2plus": {"classified_rows": len(n23(cls)), "rac_half_steps": len(n23(rac)), "total": len(n23(cls)) + len(n23(rac))}}


def frozen_manifest() -> dict:
    return json.loads((ROOT / pl().FROZEN_MANIFEST).read_text(encoding="utf-8"))


def plan_defects(plan: list[dict], frozen: dict) -> list[str]:
    """The plan of a real run must be the frozen one: counts of d057, the 102 mandatory keys in order, the F2 accounting."""
    bad = []
    if rd().plan_counts(plan) != frozen["d057"]["row_plan_counts"]:
        bad.append("row plan counts differ from the frozen manifest (d057.row_plan_counts)")
    keys = [r["key"] for r in plan if r["kind"] == "row" and r["mandatory"]]
    if keys != [r["key"] for r in frozen["mandatory_rows"]]:
        bad.append("mandatory keys differ from the frozen manifest")
    acc = plan_accounting(plan)
    want = {"classified_rows": 824, "mandatory_keys": 102, "n23_s2plus": {"classified_rows": 399, "rac_half_steps": 98, "total": 497}}
    for k, v in want.items():
        if acc[k] != v:
            bad.append(f"accounting {k} = {acc[k]!r}, frozen {v!r} (F2)")
    for r in plan:
        if r["N"] >= 2 and r["s"] >= 2 and (r["executable"] or (r["kind"] == "row" and r["planned_status"] != "UNRESOLVED")):
            bad.append(f"{r['key']} ({r['kind']}): N >= 2, s >= 2 planned as executable (D-057.1)")
    return bad


def units(plan: list[dict]) -> list[dict]:
    """Work units: one OP-8 trajectory (blocks A, B, D, S) or one block-C configuration (R-AC chain + paired rows), in plan order."""
    out, index = [], {}
    for r in plan:
        uid = f"{r['block']}|{r['family']}|{r['nx']}x{r['ny']}|N{r['N']}"
        if uid not in index:
            index[uid] = {"id": uid, "block": r["block"], "config": {"lx": float.fromhex(r["lx_hex"]), "nx": r["nx"],
                                                                    "ly": float.fromhex(r["ly_hex"]), "ny": r["ny"], "N": r["N"],
                                                                    "D": rd().D_VALUE, "q": rd().Q_VALUE, "s_max": rd().S_MAX},
                          "rows": [], "rac": []}
            out.append(index[uid])
        index[uid]["rac" if r["kind"] == "rac" else "rows"].append(r)
    return out


def _cfg(unit: dict):
    return pl().op8().TrajectoryConfig(**unit["config"])


def _safe(uid: str) -> str:
    return hashlib.sha256(uid.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------------------------------------------------------------
# child jobs (launch as the accepted probes: launcher environment, memory watchdog at 50 % RAM)
# ---------------------------------------------------------------------------------------------------------------------------------

def memory_limit() -> int:
    ram = pf().total_ram_bytes()
    return int(ram * rd().PL63_RAM_FRACTION) if ram else 0


def run_job(job: dict, work: Path, limit: int, signal: str | None) -> dict:
    """One job in a child process. Returns its result, or {"aborted": flag record} (watchdog: memory), or {"failed": ...}."""
    work.mkdir(parents=True, exist_ok=True)
    spec, flag, log = work / "job.json", work / "abort.json", work / "job.out"
    flag.unlink(missing_ok=True)
    spec.write_text(_dump(job), encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if k != pl().CAMPAIGN_AUTH_ENV}
    env.update(LAUNCH_ENV)
    if signal:
        env[pl().CAMPAIGN_AUTH_ENV] = signal
    t0 = time.perf_counter()
    with open(log, "w", encoding="utf-8") as f:
        p = subprocess.Popen([sys.executable, str(ROOT / SELF), "_job", str(spec), str(limit), str(flag)], cwd=ROOT, stdout=f,
                             stderr=subprocess.STDOUT, text=True, env=env)
        try:
            p.wait(timeout=rd().SAFETY_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()
    seconds = round(time.perf_counter() - t0, 2)
    text = log.read_text(encoding="utf-8", errors="replace")
    if flag.exists():
        return {"aborted": json.loads(flag.read_text(encoding="utf-8")), "seconds": seconds}
    if p.returncode != 0:
        return {"failed": f"child job exited with {p.returncode}", "tail": text[-1500:], "seconds": seconds}
    return {**json.loads(text.strip().splitlines()[-1]), "seconds": seconds}


def _job_child(spec_path: str, limit: int, flag: str) -> dict:
    job = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    stop = threading.Event()
    if limit > 0:
        threading.Thread(target=pf()._watchdog, args=(limit, Path(flag), stop), daemon=True).start()
    out ={"process": {"pid": os.getpid(), "blas_threads": pl().blas_threads_both(),
                       "launch_signal_present": pl().CAMPAIGN_AUTH_ENV in os.environ}}
    if job["kind"] == "journal":
        jdir = pl().run_trajectory_journal(pl().op8().TrajectoryConfig(**job["config"]), Path(job["out"]))
        out["journal"] = str(jdir)
    elif job["kind"] == "paired":
        doc = bcd().write_paired(pl().op8().TrajectoryConfig(**job["config"]), Path(job["out"]))
        out["paired"] = doc
        out["chain"] = json.loads((Path(job["out"]) / "chain.json").read_text(encoding="utf-8"))
    elif job["kind"] == "row":
        res = rd().analyse_ready(job["spec"], reproduce=True, mandatory=job["mandatory"], sp_be=job["sp_be"])
        out["analysis"] = json.loads(_dump(res))
        out["input"] = encode_input(res["classification"]["input"])
        out["observations"] = observations(job, res)
    else:
        raise ValueError(f"unknown job kind {job['kind']!r}")
    out["process"]["peak"] = pf()._mem()
    stop.set()
    return out


def observations(job: dict, res: dict) -> dict:
    """Second paths outside the verdict (protocol 5 item 9): only existing functions; exceptions recorded, never hidden."""
    paths, spec, obs = set(job.get("subset_paths") or []), job["spec"], {}
    if res.get("row") is None:
        return {p: "NOT_APPLICABLE: row not executed" for p in sorted(paths)}
    for name in ("SP-RH", "SP-LARGE", "SP-C"):
        if name in paths:
            obs[name] = NOT_IMPLEMENTED_OBS
    try:
        if "SP-OP" in paths and spec["kind"] == "journal":
            _, arrays = pl().load_journal(Path(spec["journal"]))
            cfg = pl().spec_config(spec)
            r = rd().sp_op_check(cfg, spec["s"], arrays[f"s{spec['s']}_solved_set"])
            obs["SP-OP"] = {k: r.get(k) for k in ("status", "entries_checked")}
        elif "SP-OP" in paths:
            obs["SP-OP"] = "UNRESOLVED: SP-OP is defined on journal rows of the subset"
    except Exception as exc:  # recorded, never hidden
        obs["SP-OP"] = f"UNRESOLVED: {type(exc).__name__}: {str(exc)[-300:]}"
    try:
        if "SP-BR" in paths and spec["kind"] == "journal":
            r = adapter().reference_for_journal(Path(spec["journal"]), spec["s"])
            obs["SP-BR"] = json.loads(_dump(r))
        elif "SP-BR" in paths:
            obs["SP-BR"] = "UNRESOLVED: SP-BR is defined on journal rows of the subset"
    except Exception as exc:  # recorded, never hidden
        obs["SP-BR"] = f"UNRESOLVED: {type(exc).__name__}: {str(exc)[-300:]}"
    return obs


# ---------------------------------------------------------------------------------------------------------------------------------
# rows of a unit
# ---------------------------------------------------------------------------------------------------------------------------------

def _not_executed_record(row: dict, reason: str, resources: dict | None = None) -> dict:
    ne = {"reason": reason}
    if resources is not None:
        ne.update(resources=True, measured=resources)
    inp = {"executed": False, "not_executed": ne}  # the input of this record is made here (no pipeline call): it is the source
    res = pf().classify_row(inp)
    return {"key": row["key"], "block": row["block"], "N": row["N"], "s": row["s"], "mandatory": row["mandatory"], "executed": False,
            "status": res["status"], "reasons": res["reasons"], "input": encode_input(inp), "z4": None, "tool_defects": []}


def _planned_reason(row: dict, cfg) -> str:
    if row["N"] >= 2 and row["s"] >= 2:
        return pl().rank_guard_reason(cfg, row["s"])  # D-057.1: trajectory STOPPED_RANK_GUARD_UNCERTIFIED, row UNRESOLVED
    return row["reason"]


def _job_fault(r: dict, what: str) -> tuple[str, dict | None]:
    if "aborted" in r:
        return f"PL 6.3: {what} job aborted by the memory watchdog", {**r["aborted"], "seconds": r.get("seconds")}
    return f"not executed: {what} job failed ({r.get('failed')}): {str(r.get('tail', ''))[-400:]}", None


def _row_job(row: dict, spec: dict, subsets: dict, work: Path, limit: int, signal, dry_run: bool) -> dict:
    paths = subsets.get(row["key"], [])
    job = {"kind": "row", "spec": spec, "mandatory": row["mandatory"], "sp_be": "SP-BE" in paths, "subset_paths": paths,
           "dry_run": dry_run}
    r = run_job(job, work, limit, signal)
    if "aborted" in r or "failed" in r:
        reason, measured = _job_fault(r, "row")
        return _not_executed_record(row, reason, measured)
    a = r["analysis"]
    a["tool_data_status"] = a.pop("data_status", None)  # the label of the row tool; the record is labelled by this run
    a["data_status"] = DATA_STATUS_DRY if dry_run else DATA_STATUS_CAMPAIGN
    return {"key": row["key"], "block": row["block"], "N": row["N"], "s": row["s"], "mandatory": row["mandatory"],
            "executed": a.get("row") is not None, "status": a["status"], "reasons": a["classification"].get("reasons"),
            "input": r["input"], "z4": a.get("z4"), "tool_defects": a.get("tool_defects") or [], "observations": r["observations"],
            "process": r["process"], "seconds": r["seconds"], "analysis": a}


def run_unit(unit: dict, work: Path, limit: int, signal, subsets: dict, dry_run: bool, jobs: list) -> dict:
    cfg = _cfg(unit)
    rows = [None] * len(unit["rows"])
    execs = [i for i, r in enumerate(unit["rows"]) if r["executable"]]
    for i, r in enumerate(unit["rows"]):
        if not r["executable"]:
            rows[i] = _not_executed_record(r, _planned_reason(r, cfg))  # never computed
    rac = [{"key": r["key"], "s": r["s"], "planned_executable": r["executable"],
            "trajectory_reason": None if r["executable"] else _planned_reason(r, cfg)} for r in unit["rac"]]
    if not execs:
        return {"unit": unit["id"], "rows": rows, "rac": rac, "prep": None}
    kind = "paired" if unit["block"] == "C" else "journal"
    prep = run_job({"kind": kind, "config": unit["config"], "out": str(work / "data"), "dry_run": dry_run}, work / "prep", limit,
                   signal)
    jobs.append(unit["id"] + "|prep")
    if "aborted" in prep or "failed" in prep:
        reason, measured = _job_fault(prep, kind)
        for i in execs:
            rows[i] = _not_executed_record(unit["rows"][i], reason, measured)
        return {"unit": unit["id"], "rows": rows, "rac": rac, "prep": {k: prep[k] for k in prep if k != "tail"}}
    by_s = {x["s"]: x for x in prep["paired"]["rows"]} if kind == "paired" else {}
    for i in execs:
        r = unit["rows"][i]
        if kind == "journal":
            spec = {"kind": "journal", "journal": prep["journal"], "s": r["s"]}
        else:
            x = by_s.get(r["s"])
            if x is None or not x.get("executed"):
                rows[i] = _not_executed_record(r, (x or {}).get("reason") or f"paired half-step s = {r['s']} absent")
                continue
            spec = x["spec"]
        rows[i] = _row_job(r, spec, subsets, work / f"s{r['s']}", limit, signal, dry_run)
        jobs.append(f"{unit['id']}|s{r['s']}")
    if kind == "paired":
        recs = {x.get("s"): x for x in (prep.get("chain") or {}).get("records", []) if isinstance(x, dict)}
        for x in rac:
            x["chain_record_status"] = (recs.get(x["s"]) or {}).get("status")
    return {"unit": unit["id"], "rows": rows, "rac": rac,
            "prep": {"process": prep.get("process"), "seconds": prep.get("seconds"),
                     "chain_status": (prep.get("paired") or {}).get("chain_status")}}


# ---------------------------------------------------------------------------------------------------------------------------------
# before the run, verdict
# ---------------------------------------------------------------------------------------------------------------------------------

def platform_defects() -> list[str]:
    ap = rd().APPROVED_PLATFORM
    bad = []
    if sys.platform != ap["sys_platform"] or platform.python_version() != ap["python"]:
        bad.append(f"platform {sys.platform} / Python {platform.python_version()} is not the approved one (OD-6)")
    bt = pl().blas_threads_both()
    if bt != {"numpy": 1, "scipy": 1}:
        bad.append(f"BLAS threads read from the libraries {bt!r}, not 1 / 1 (OD-6, D-058.3)")
    for k in THREAD_VARS:
        if os.environ.get(k) != "1":
            bad.append(f"{k} is not 1")
    return bad


def prerun_checks(plan: list[dict], frozen: dict, work: Path) -> dict:
    """Protocol 4.4, 6, 12.4: s = 1 input hashes and the start archive against the frozen manifest; controls of PR #48 and of
    QA-D1-READY (A/B/V/G) against their frozen expectations, in a child process without the launch signal."""
    want = {x["key"]: x for x in frozen["d057"]["input_hashes_s1"]}
    have = rd().input_hashes(plan)
    mism = [x["key"] for x in have if want.get(x["key"]) != x]
    mism += [f"missing s = 1 hash: {k}" for k in sorted(set(want) - {x["key"] for x in have})]
    arch = json.loads((ROOT / frozen["d057"]["start_archive"]["path"]).read_text(encoding="utf-8"))
    mism += [f"start archive: {x}" for x in rd().verify_archive(arch)]
    pr48 = pf().run_controls()
    s48 = pr48["summary"]
    pr48_ok = all(s48.get(k) for k in ("control_A_all_match", "control_B_synthetic_all_match", "control_B_detector_functions_all_match"))
    env = {k: v for k, v in os.environ.items() if k != pl().CAMPAIGN_AUTH_ENV}
    env.update(LAUNCH_ENV)
    cdir = work / "controls"
    p = subprocess.run([sys.executable, str(ROOT / CONTROLS_TOOL), "run", "--out", str(cdir)], cwd=ROOT, capture_output=True, text=True,
                       env=env)
    ready_ok, v_fail, detail = False, False, f"controls child exited with {p.returncode}: {p.stderr[-800:]}"
    if p.returncode == 0:
        res = json.loads((cdir / "controls_results.json").read_text(encoding="utf-8"))
        ready_ok = res["all_match"] is True
        v_fail = any(str(x.get("verdict", "")).startswith("UNEXPECTED") for x in res["control_V"])
        detail = res["summary"]
    return {"input_hash_mismatch": mism, "pr48_controls_all_match": pr48_ok, "ready_controls_all_match": ready_ok,
            "control_v_fail": v_fail, "controls_detail": detail}


def campaign_verdict(records: list[dict], input_hash_mismatch=(), control_v_fail: bool = False) -> dict:
    """``ready.verdict`` (unchanged) on the classified rows, the mandatory list of the frozen manifest (F4)."""
    mand = pf().mandatory_index(frozen_manifest()["mandatory_rows"])
    rows = [{"key": r["key"], "block": r["block"], "N": r["N"], "s": r["s"], "mutated": False, "input": decode_input(r["input"]),
             "z4": r.get("z4"), "tool_defects": r.get("tool_defects") or [], "mandatory": r["mandatory"]} for r in records]
    return rd().verdict(rows, mand, input_hash_mismatch=tuple(input_hash_mismatch), control_v_fail=control_v_fail)


# ---------------------------------------------------------------------------------------------------------------------------------
# the campaign
# ---------------------------------------------------------------------------------------------------------------------------------

def _provenance(dry_run: bool, signal) -> dict:
    tools = sorted({READY_TOOL, SELF, pl().SELF, CONTROLS_TOOL, ADAPTER_TOOL, *pl().USED_TOOLS})
    return {"dry_run": dry_run, "head": pl()._git_out(ROOT, "rev-parse", "HEAD")[1].strip(), "launch_signal": signal,
            "tool_blobs": {t: pl().git_blob_id(ROOT / t) for t in tools}, "python": platform.python_version(),
            "sys_platform": sys.platform, "numpy": np.__version__}


def _plan_sha(plan: list[dict]) -> str:
    return hashlib.sha256(pf().canonical(plan)).hexdigest()


def run_campaign(out: Path, plan: list[dict] | None = None, dry_run: bool = False, memory_limit_bytes: int | None = None,
                 stop_after_jobs: int | None = None, subsets: dict | None = None) -> dict:
    """The campaign (or a dry run on non-D1 grids). Refused before any computation when a condition fails; resumable: a unit whose
    result is stored is not run again, and a restart must have the same provenance."""
    out = Path(out).resolve()
    scratch = (ROOT / "scratch").resolve()
    if out != scratch and scratch not in out.parents:
        raise CampaignRefused(f"output must be under {scratch} (no published data during the run)")
    pl().guard_out(out)
    signal = os.environ.get(pl().CAMPAIGN_AUTH_ENV)
    if dry_run:
        if plan is None:
            raise CampaignRefused("a dry run needs its own plan of non-D1 grids")
        d1 = sorted({(r["nx"], r["ny"]) for r in plan if (r["nx"], r["ny"]) in pl().op8().D1_GRIDS})
        if d1:
            raise CampaignRefused(f"a dry run never touches the D1 grids: {d1}")
        signal = None
        frozen = None
    else:
        if plan is not None or memory_limit_bytes is not None or subsets is not None:
            raise CampaignRefused("a real run uses the frozen plan, subsets and the PL 6.3 memory limit only")
        gate = pl().campaign_gate()
        if not gate["authorized"]:
            raise CampaignRefused(f"D1_START = BLOCKED (D-059 gate {gate['state']}: {'; '.join(gate['reasons'])})")
        bad = platform_defects()
        frozen = frozen_manifest()
        plan = rd().row_plan()
        bad += plan_defects(plan, frozen)
        if bad:
            raise CampaignRefused("; ".join(bad))
        subsets = {x["key"]: x["paths"] for x in frozen["d057"]["subsets"]["rows"]}
        for r in plan:  # protocol 5 item 9: R-H on every block-D N = 1 row (recorded, even while not implemented)
            if r["kind"] == "row" and r["block"] == "D" and r["N"] == 1 and r["executable"]:
                subsets.setdefault(r["key"], []).append("SP-RH")
    subsets = subsets or {}
    limit = memory_limit_bytes if memory_limit_bytes is not None else memory_limit()
    prov = _provenance(dry_run, signal)
    state_path = out / "state.json"
    if state_path.exists():
        st = json.loads(state_path.read_text(encoding="utf-8"))
        if st["provenance"] != prov or st["plan_sha256"] != _plan_sha(plan):
            raise CampaignRefused("restart refused: provenance or plan differs from the interrupted run")
    else:
        _write_json(state_path, {"provenance": prov, "plan_sha256": _plan_sha(plan), "started": time.strftime("%Y-%m-%dT%H:%M:%S")})
    status = DATA_STATUS_DRY if dry_run else DATA_STATUS_CAMPAIGN
    pre = None
    if not dry_run:
        pre_path = out / "prerun.json"
        if pre_path.exists():
            pre = json.loads(pre_path.read_text(encoding="utf-8"))
        else:
            pre = prerun_checks(plan, frozen, out / "work_prerun")
            _write_json(pre_path, pre)
        if pre["input_hash_mismatch"] or pre["control_v_fail"]:
            recs = [_not_executed_record(r, "not executed: the campaign is blocked before the run (protocol 4.4)")
                    for r in plan if r["kind"] == "row"]
            doc = {"data_status": status, "state": "BLOCKED_BEFORE_THE_RUN", "prerun": pre, "accounting": plan_accounting(plan),
                   "verdict": campaign_verdict(recs, pre["input_hash_mismatch"], pre["control_v_fail"]), "provenance": prov}
            _write_json(out / "campaign_result.json", doc)
            return doc
        if not (pre["pr48_controls_all_match"] and pre["ready_controls_all_match"]):
            doc = {"data_status": status, "state": "NOT_STARTED_CONTROLS_MISMATCH (protocol 12 item 4)", "prerun": pre,
                   "provenance": prov}
            _write_json(out / "campaign_result.json", doc)
            return doc
    jobs = []
    results = []
    for u in units(plan):
        upath = out / "units" / f"{_safe(u['id'])}.json"
        if upath.exists():
            results.append(json.loads(upath.read_text(encoding="utf-8")))
            continue
        work = out / "work" / _safe(u["id"])
        if work.exists():
            shutil.rmtree(work)  # the partial work of an interrupted unit (always under out/)
        res = run_unit(u, work, limit, signal, subsets, dry_run, jobs)
        _write_json(upath, res)
        results.append(res)
        if stop_after_jobs is not None and len(jobs) >= stop_after_jobs:
            raise CampaignInterrupted(f"stopped after {len(jobs)} jobs (test hook)")
    records = [x for u in results for x in u["rows"]]
    doc = {"data_status": status, "state": "COMPLETED", "accounting": plan_accounting(plan), "prerun": pre, "provenance": prov,
           "memory_limit_bytes": limit, "limitations": ["replay processes of analyse are not under the memory watchdog of the job",
                                                         "SP-RH, SP-LARGE, SP-C: no row-level implementation in the frozen tools",
                                                         "BC-1 / TS-1 recorded on the rows of SP-BR only (reference adapter)"],
           "rows": [{k: x.get(k) for k in ("key", "block", "N", "s", "mandatory", "executed", "status", "z4", "tool_defects")}
                    for x in records],
           "rac": [x for u in results for x in u["rac"]]}
    if not dry_run:
        doc["verdict"] = campaign_verdict(records)
    _write_json(out / "campaign_result.json", doc)
    return doc


# ---------------------------------------------------------------------------------------------------------------------------------
# C4: the re-freeze addendum (written once, before the owner's act; its identity is the git blob named by that act)
# ---------------------------------------------------------------------------------------------------------------------------------

CHANGED = ("scripts/r5_cccc_q4_d1_row_pipeline_prelaunch.py", READY_TOOL)
NEW = (SELF,)
ADDED_UNCHANGED = (ADAPTER_TOOL,)


def build_addendum(controls_results: Path, base_commit: str) -> dict:
    p = pl()
    frozen = frozen_manifest()
    d058 = {**{c["path"]: c["blob"] for c in frozen["sha_inventory"]["components"] if c.get("path") and c.get("blob")},
            **{c["path"]: c["blob"] for c in frozen["sha_inventory"]["other_used_files"]}, **frozen["d057"]["tool_blobs"]}
    tools = {"changed": {t: {"d058_blob": d058[t], "blob": p.git_blob_id(ROOT / t)} for t in CHANGED},
             "new": {t: {"blob": p.git_blob_id(ROOT / t)} for t in NEW},
             "unchanged": {t: {"blob": b} for t, b in sorted(d058.items()) if t not in CHANGED}}
    for t in ADDED_UNCHANGED:
        tools["unchanged"][t] = {"blob": p.git_blob_id(ROOT / t), "note": "added to the frozen set unchanged (D-059, C3)"}
    raw = Path(controls_results).read_bytes()
    res = json.loads(raw)
    plan = rd().row_plan()
    return {
        "document": "R5-V Q4 D1 campaign - re-freeze addendum (D-059, C4)",
        "decision": p.CAMPAIGN_DECISION, "state": p.PENDING_REFREEZE, "d1_start": "BLOCKED",
        "activation": {
            "proposed_state": "AUTHORIZED only through pipeline.campaign_gate after the owner's re-freeze act, Reviewer PASS and "
                              "integration (D-059.4); until then every D1 grid is refused",
            "refreeze_act_line": "D1_CAMPAIGN_REFREEZE_ACT: decision=D-NNN; authorizes=D-059; addendum=" + p.CAMPAIGN_ADDENDUM +
                                 "; addendum_blob=<git blob of this file>; state=ACCEPTED",
            "launch_signal": p.CAMPAIGN_AUTH_ENV + "=D-059:<git blob of this file>",
            "identity": "the git blob of this file named by the owner's act (no self-referential field)"},
        "frozen_baseline": {
            "main_commit": base_commit,
            "d058_frozen_manifest": {"path": p.FROZEN_MANIFEST, "blob": p.FROZEN_MANIFEST_BLOB, "decision": "D-058",
                                     "state": "FROZEN, unchanged (byte-identical)"},
            "prelaunch_candidate_manifest": {"path": p.FROZEN_MANIFEST, "blob": "5a794e6e4c3e0031e0bbd496635b27d31bb02e44",
                                             "commit": "2e0aac636f73e1148f6c1db8201034299a035f7f",
                                             "state": "candidate before the D-058 freeze act; NOT the frozen manifest"},
            "mandatory_rows_csv": {"path": "docs/results/r5_cccc_q4_d1_prelaunch/mandatory_rows.csv",
                                   "blob": frozen["freeze_act"]["baseline"]["mandatory_rows_csv_blob"]}},
        "tools": tools,
        "plan": {"counts": rd().plan_counts(plan), "accounting": plan_accounting(plan),
                 "mandatory_keys_sha256": frozen["mandatory_keys_sha256"]},
        "platform": {**rd().platform_record(), "blas_threads_both_libraries": p.blas_threads_both()},
        "controls": {"results_sha256": hashlib.sha256(raw).hexdigest(), "all_match": res["all_match"], "summary": res["summary"],
                     "expected_sha256": res["expected_sha256"], "pr48": pf().run_controls()["summary"],
                     "run": "QA-D1-READY controls A/B/V/G and PR #48 harness on the changed tools, approved platform, no launch signal"},
        "probes": {"accepted": "16 PL 6.3 probes of D-058.2 (docs/results/r5_cccc_q4_d1_ready/resource_probes.json)",
                   "carry_forward": "applicable to the computation: _compute_row_core, OP-8, the block-C driver, the BR tool and the "
                                    "numerical environment are unchanged; the launch (child process, launcher environment, memory "
                                    "watchdog at 50 % RAM) is the one of the probes and is tested separately "
                                    "(tests/reference/test_r5_cccc_q4_d1_campaign.py); the probes do not establish new behaviour",
                   "not_established": ["replay processes of analyse are not under the job watchdog",
                                       "the 30 min factorization limit is not measured per stage during the run (probes: <= 4.62 s)"]},
        "limitations": ["SP-RH, SP-LARGE, SP-C: no row-level implementation in the frozen tools: recorded UNRESOLVED, never PASS",
                        "BC-1 / TS-1 are recorded on the rows of SP-BR only (reference adapter; journal rows)",
                        "D-058.5 retained: B_E coverage partial (N-4, N-6); SP-OP measured on one probe; limited reviewer independence",
                        "the gate trusts the local git repository (refs/remotes/origin/main, HEAD); not a security boundary"],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", type=Path, required=True)
    j = sub.add_parser("_job")
    j.add_argument("spec")
    j.add_argument("limit", type=int)
    j.add_argument("flag")
    a_ = sub.add_parser("addendum")
    a_.add_argument("--controls-results", type=Path, required=True)
    a_.add_argument("--base-commit", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "_job":
        try:
            res = _job_child(a.spec, a.limit, a.flag)
        except pl().D1Refused as exc:
            print(f"refused: {exc}", file=sys.stderr)
            return 2
        print(_dump(res))
        return 0
    if a.cmd == "addendum":
        _write_json(ROOT / pl().CAMPAIGN_ADDENDUM, build_addendum(a.controls_results, a.base_commit))
        return 0
    try:
        doc = run_campaign(a.out)
    except CampaignRefused as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"state": doc["state"], "verdict": (doc.get("verdict") or {}).get("verdict")}, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
