"""R5-V Q4 D1 campaign runner (D-059, C3; protocol docs/R5_CCCC_Q4_D1_PROTOCOL.md 3-6, 8, 12, 13). QA tool: it connects the
approved components and adds no numerical algorithm, formula, threshold, tolerance or status. No import of the working ``kkvim``.

What it connects (all read-only):
- the row plan of D1 (``ready.row_plan``: 824 classified rows, 102 mandatory block-A keys; 168 R-AC half-steps of block C that are
  NOT campaign rows; F2: 497 positions N >= 2, s >= 2 = 399 classified rows + 98 R-AC half-steps, never computed);
- OP-8 journals (``pipeline.run_trajectory_journal``; the rank guard of D-057.1 stops N = 2, 3 before the QR of s = 2) and the
  block-C driver (``write_paired``: R-AC chain and paired half-steps);
- each executable row through ``ready.analyse_ready`` (binding, separate-process replay, F-2, F-3, SP-GAMMA, SP-BE, Z-4);
- second paths (protocol 5 item 9; OD-11), in a job of their own after the row, bound by sha256 to the row's arrays: SP-RH (direct R-H,
  ``hs.solve_hessian``, with the exact-residual bound), SP-LARGE (exact A, B, C, F against the doubles, exact residual R-HP, certified
  lambda_low), SP-C (exact rational C, the C-3+ theorem bound, CONTROL as the double cross-check), SP-BR and SP-OP; a contradiction of
  certified values is a tool defect (the verdict is then not issued); PASS of a second path means "established, no contradiction",
  not accuracy (no acceptance tolerance is assigned; measured values are recorded);
- accuracy observations (protocol 5.1): BC-1 and TS-1 by the BR reference on every executed classified row, OBSERVATION only;
- the controls (PR #48 harness and the QA-D1-READY controls A/B/V/G against their frozen expectations, run in a child process
  WITHOUT the launch signal) and the s = 1 input hashes / start archive before the first row;
- the verdict: ``ready.verdict`` (unchanged Z-4 rule) with ``input_hash_mismatch`` and ``control_v_fail`` passed to the unchanged
  ``campaign_verdict`` of PR #48 (F4: BLOCKED_CAMPAIGN).

Execution conditions: every job (journal, block-C configuration, row, second paths) runs in its own child process with
OPENBLAS/OMP/MKL_NUM_THREADS = 1 set by this launcher before NumPy is imported, as the accepted PL 6.3 probes. PL 6.3 is enforced from
the parent (``monitor_job``; limits unchanged, OD-7): the peak memory of every process of the job tree, replay children included,
against 50 % of the physical RAM, and every factorization stage against 30 min (stage log of the pipeline, ``install_stage_timers``);
the tree is stopped on a violation and the rows of a row or preparation job become UNRESOLVED_RESOURCES with the measured values, the
responsible process and stage. A violation in the second-path job leaves the status of the row unchanged (OD-7). The convergence of the
refinement stays where the contract has it: a reference that does not converge is UNRESOLVED as an observation.

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
import time
from fractions import Fraction
from pathlib import Path

THREAD_VARS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
for _k in THREAD_VARS:
    os.environ[_k] = "1"  # OD-6: before NumPy is imported (the children get it from this launcher as well)

import numpy as np  # noqa: E402

from kkvim_reference import r5_cccc_bc as bc  # noqa: E402
from kkvim_reference import r5_cccc_halfstep as hs  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SELF = "scripts/r5_cccc_q4_d1_campaign.py"
READY_TOOL = "scripts/r5_cccc_q4_d1_ready.py"
CONTROLS_TOOL = "scripts/r5_cccc_q4_d1_ready_controls.py"
ADAPTER_TOOL = "scripts/r5_cccc_q4_d1_bc_reference_adapter.py"
DATA_STATUS_CAMPAIGN = ("D1 CAMPAIGN DATA (D-059; protocol 12 item 6): valid only under the owner's re-freeze act recorded in "
                        "docs/DECISIONS.md and the addendum it names")
DATA_STATUS_DRY = "DRY RUN on non-D1 grids (D-059 tests): NOT D1 data, never D1_RESULT"
PASS, UNRESOLVED, CONTRADICTION = "PASS", "UNRESOLVED", "CONTRADICTION"
UNRESOLVED_RESOURCES_OF_CHECK = "UNRESOLVED_RESOURCES_OF_CHECK"  # OD-7: of the check, not of the row
BLOCKED_FOR_OWNER = "BLOCKED_FOR_OWNER_DECISION"
PASS_MEANS = ("the second path was computed on the row's bound arrays and no certified value contradicts the row; NOT a statement of "
              "accuracy: no acceptance tolerance is assigned (D-054.4), measured values are recorded")
ACCURACY = "BC-1/TS-1"
EXEC_DONE, EXEC_PLAN, EXEC_GUARD, EXEC_RESOURCES, EXEC_FAILED = ("EXECUTED", "NOT_EXECUTED_PLAN", "NOT_EXECUTED_RANK_GUARD",
                                                                 "NOT_EXECUTED_RESOURCES", "NOT_EXECUTED_JOB_FAILED")
HS_MODULE, BC_MODULE = "src/kkvim_reference/r5_cccc_halfstep.py", "src/kkvim_reference/r5_cccc_bc.py"
CA_TOOL, BR_TOOL = "scripts/r5_cccc_q4_c_assembly_e4_audit.py", "scripts/r5_cccc_q4_bc_reference_audit.py"
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


def _process_table() -> dict[int, int]:
    """pid -> parent pid of the running processes ({} when not measurable here)."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class PE(ctypes.Structure):
            _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD), ("pid", wintypes.DWORD),
                        ("heap", ctypes.c_size_t), ("module", wintypes.DWORD), ("threads", wintypes.DWORD),
                        ("ppid", wintypes.DWORD), ("pri", ctypes.c_long), ("flags", wintypes.DWORD), ("exe", ctypes.c_wchar * 260)]

        k = ctypes.WinDLL("kernel32", use_last_error=True)
        k.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
        k.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
        k.Process32FirstW.argtypes = k.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(PE)]
        k.CloseHandle.argtypes = [wintypes.HANDLE]
        h = k.CreateToolhelp32Snapshot(2, 0)
        if h in (None, wintypes.HANDLE(-1).value):
            return {}
        e, out = PE(), {}
        e.dwSize = ctypes.sizeof(PE)
        ok = k.Process32FirstW(h, ctypes.byref(e))
        while ok:
            out[int(e.pid)] = int(e.ppid)
            ok = k.Process32NextW(h, ctypes.byref(e))
        k.CloseHandle(h)
        return out
    out = {}
    try:
        names = os.listdir("/proc")
    except OSError:
        return {}
    for n in names:
        if n.isdigit():
            try:
                with open(f"/proc/{n}/stat", encoding="utf-8", errors="replace") as f:
                    out[int(n)] = int(f.read().rsplit(")", 1)[1].split()[1])
            except (OSError, IndexError, ValueError):
                continue
    return out


def _peak_of(pid: int) -> int | None:
    """Peak working set (resident set) of a running process in bytes; None when it cannot be read."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class C(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t), ("a", ctypes.c_size_t), ("b", ctypes.c_size_t), ("c", ctypes.c_size_t),
                        ("d", ctypes.c_size_t), ("e", ctypes.c_size_t), ("f", ctypes.c_size_t)]

        k, ps = ctypes.WinDLL("kernel32"), ctypes.WinDLL("psapi")
        k.OpenProcess.restype = wintypes.HANDLE
        k.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        k.CloseHandle.argtypes = [wintypes.HANDLE]
        ps.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(C), wintypes.DWORD]
        h = k.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return None
        c = C()
        c.cb = ctypes.sizeof(c)
        ok = ps.GetProcessMemoryInfo(h, ctypes.byref(c), c.cb)
        k.CloseHandle(h)
        return int(c.PeakWorkingSetSize) if ok else None
    try:
        with open(f"/proc/{pid}/status", encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) * 1024
    except (OSError, IndexError, ValueError):
        return None
    return None


def memory_measurement_works() -> bool:
    """The PL 6.3 monitor can see this process and read its peak (checked before a real run)."""
    return os.getpid() in _process_table() and bool(_peak_of(os.getpid()))


def _stop_tree(root: int, pids) -> None:
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/PID", str(root), "/T", "/F"], capture_output=True)
    table = _process_table()
    for pid in pids:
        if pid in table:
            try:
                os.kill(pid, 9)
            except OSError:
                pass


def _read_stages(path: Path, pos: int) -> tuple[list[dict], int]:
    """Complete new lines of the stage log."""
    try:
        with open(path, "rb") as f:
            f.seek(pos)
            data = f.read()
    except OSError:
        return [], pos
    end = data.rfind(b"\n") + 1
    out = []
    for line in data[:end].splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out, pos + end


def monitor_job(proc, limit_bytes: int, factorization_limit_s: float, stage_log: Path, poll_s: float = 0.2) -> dict:
    """PL 6.3 from the parent (limits unchanged, D-057.4): the peak of EVERY process of the job tree (replay children included) against
    the memory limit, and every factorization stage of the stage log against the time limit, while it runs and when it ends. On a
    violation the tree is stopped; the record names the rule, the measured value, the process and its last stage. The sum over the tree
    is a measurement only (D-057.4 states the limit per process)."""
    t0 = time.perf_counter()
    root = proc.pid
    known, peaks, last_stage, active, stages = {root}, {}, {}, {}, {}
    pos, violation, sum_peak, safety = 0, None, 0, False

    def check_memory():
        table = _process_table()
        grew = True
        while grew:
            kids = {p for p, pp in table.items() if pp in known and p not in known}
            grew = bool(kids)
            known.update(kids)
        total = 0
        for pid in known:
            pk = _peak_of(pid) if pid in table else None
            if pk is not None:
                peaks[pid] = max(peaks.get(pid, 0), pk)
                total += pk
        worst = max(peaks, key=peaks.get, default=None)
        if limit_bytes > 0 and worst is not None and peaks[worst] > limit_bytes:
            return {"aborted": "memory", "working_set_bytes": peaks[worst], "limit_bytes": limit_bytes, "pid": worst,
                    "process_stage": last_stage.get(worst), "rule": "PL 6.3: peak of a process <= 50 % of the physical RAM (D-057.4)"}, total
        return None, total

    def check_stages():
        nonlocal pos
        events, pos = _read_stages(stage_log, pos)
        now, found = time.time(), None
        for e in events:
            last_stage[e["pid"]] = e["stage"]
            if not e.get("factorization"):
                continue
            key = (e["pid"], e["stage"])
            if e["event"] == "start":
                active.setdefault(key, []).append(e["t"])
            elif active.get(key):
                sec = e["t"] - active[key].pop()
                st = stages.setdefault(e["stage"], {"count": 0, "max_s": 0.0, "pid_of_max": None})
                st["count"] += 1
                if sec >= st["max_s"]:
                    st.update(max_s=round(sec, 3), pid_of_max=e["pid"])
                if sec > factorization_limit_s and found is None:
                    found = {"aborted": "factorization_time", "seconds": round(sec, 3), "limit_seconds": factorization_limit_s,
                             "stage": e["stage"], "pid": e["pid"], "completed": True}
        for (pid, name), starts in active.items():
            if starts and now - starts[0] > factorization_limit_s and found is None:
                found = {"aborted": "factorization_time", "seconds": round(now - starts[0], 3), "limit_seconds": factorization_limit_s,
                         "stage": name, "pid": pid, "completed": False}
        if found:
            found["rule"] = "PL 6.3: each factorization stage <= 30 min (D-057.4)"
        return found

    while True:
        running = proc.poll() is None
        violation, total = check_memory()
        sum_peak = max(sum_peak, total)
        violation = violation or check_stages()
        if violation or not running:
            break
        if time.perf_counter() - t0 > rd().SAFETY_TIMEOUT_S:  # operational safety of the runner, not a PL 6.3 limit
            safety = True
            break
        time.sleep(poll_s)
    if violation or safety:
        _stop_tree(root, set(known))
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    return {"violation": violation, "safety_timeout": safety, "wall_s": round(time.perf_counter() - t0, 2),
            "processes": {str(k): v for k, v in peaks.items()}, "max_process_peak_bytes": max(peaks.values(), default=None),
            "max_tree_sum_bytes": sum_peak, "limit_bytes": limit_bytes, "factorization_limit_s": factorization_limit_s,
            "factorization_stages": stages, "last_stage_by_process": {str(k): v for k, v in last_stage.items()}}


def run_job(job: dict, work: Path, limit: int, signal: str | None, factorization_limit_s: float | None = None) -> dict:
    """One job in a child process under ``monitor_job``. Returns its result, or {"aborted": record} (PL 6.3), or {"failed": ...}."""
    work.mkdir(parents=True, exist_ok=True)
    spec, log, stage_log = work / "job.json", work / "job.out", work / "stages.jsonl"
    stage_log.unlink(missing_ok=True)
    spec.write_text(_dump(job), encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if k != pl().CAMPAIGN_AUTH_ENV}
    env.update(LAUNCH_ENV)
    env[pl().STAGE_LOG_ENV] = str(stage_log)
    if signal:
        env[pl().CAMPAIGN_AUTH_ENV] = signal
    with open(log, "w", encoding="utf-8") as f:
        p = subprocess.Popen([sys.executable, str(ROOT / SELF), "_job", str(spec)], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, text=True,
                             env=env)
        mon = monitor_job(p, limit, rd().PL63_FACTORIZATION_S if factorization_limit_s is None else factorization_limit_s, stage_log)
    text = log.read_text(encoding="utf-8", errors="replace")
    if mon["violation"]:
        return {"aborted": mon["violation"], "resources": mon, "seconds": mon["wall_s"]}
    if mon["safety_timeout"]:
        return {"failed": f"operational safety timeout of the runner ({rd().SAFETY_TIMEOUT_S} s; not a PL 6.3 limit)",
                "tail": text[-1500:], "resources": mon, "seconds": mon["wall_s"]}
    if p.returncode != 0:
        return {"failed": f"child job exited with {p.returncode}", "tail": text[-1500:], "resources": mon, "seconds": mon["wall_s"]}
    return {**json.loads(text.strip().splitlines()[-1]), "resources": mon, "seconds": mon["wall_s"]}


def _job_child(spec_path: str) -> dict:
    job = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    pl().stage_event(f"job: {job['kind']}", "start")
    extra = [(adapter().br(), "dec_cholesky", "decimal Cholesky of BR")] if job["kind"] == "second_paths" else []
    pl().install_stage_timers(extra)
    out = {"process": {"pid": os.getpid(), "blas_threads": pl().blas_threads_both(),
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
    elif job["kind"] == "second_paths":
        out["checks"] = json.loads(_dump(second_paths(job)))
    else:
        raise ValueError(f"unknown job kind {job['kind']!r}")
    out["process"]["peak"] = pf()._mem()
    return out


# ---------------------------------------------------------------------------------------------------------------------------------
# second paths (protocol 5 item 9, OD-11) and accuracy observations (protocol 5.1): existing reference functions only, on the arrays of
# the row, bound by sha256. Never a call of the B-C solver.
# ---------------------------------------------------------------------------------------------------------------------------------

def row_arrays(spec: dict) -> dict:
    """The arrays of an executed row with the check of their binding (journal or paired row)."""
    cfg = pl().spec_config(spec)
    pl().guard_config(cfg)
    sha, s, bad = pl().op8().array_sha, spec["s"], []
    if spec["kind"] == "journal":
        j, arrays = pl().load_journal(Path(spec["journal"]))
        rec = next((r for r in j["steps"] if r.get("s") == s), None)
        if rec is None or f"s{s}_V_bc" not in arrays:
            raise ValueError(f"no solved half-step {s} in the journal")
        solved, V, C3 = arrays[f"s{s}_solved_set"], arrays[f"s{s}_V_bc"], arrays.get(f"s{s}_C3plus")
        binding = pl().bind_journal(j, arrays)[s]
        if not binding["ok"]:
            bad += list(binding["reasons"])
        for name, a, key in (("solved set", solved, "solved_set_sha"), ("V", V, "V_bc_sha"), ("C-3+", C3, "C3plus_sha")):
            if a is None or rec.get(key) != sha(a):
                bad.append(f"{name}: sha256 differs from the journal record ({key})")
    elif spec["kind"] == "paired":
        rec, C3 = None, None
        with np.load(spec["paired_npz"]) as z:
            solved, V = z["solved_set"], z["V"]
        for name, a in (("solved_set", solved), ("V", V)):
            if (spec.get("paired_sha256") or {}).get(name) != sha(a):
                bad.append(f"{name}: sha256 differs from the paired specification")
    else:
        raise ValueError(f"row kind {spec['kind']!r} has no second path")
    return {"cfg": cfg, "s": s, "solved": np.asarray(solved, dtype=float), "V": np.asarray(V, dtype=float),
            "C3plus": None if C3 is None else np.asarray(C3, dtype=float), "record": rec, "binding_defects": bad,
            "input_hashes": {"solved_set_sha": sha(solved), "V_sha": sha(V), "C3plus_sha": None if C3 is None else sha(C3)}}


def _half_step(inp: dict) -> hs.HalfStep:
    cfg = inp["cfg"]
    return hs.HalfStep(float(cfg.lx), cfg.nx, float(cfg.ly), cfg.ny, pl().op8().direction_of(inp["s"]), inp["solved"], D=float(cfg.D),
                       q=float(cfg.q))


def _lambda_low(ps: hs.HalfStep, coeffs) -> tuple[float | None, float | None]:
    tau = adapter().br().tau_trial(ps)
    return tau, None if tau is None else hs.certify_lambda_min(ps, tau, coeffs)


def _term_first(V: np.ndarray) -> np.ndarray:
    return np.asarray(V, dtype=float).T.reshape(-1)  # the ordering of the half-step unknowns (OP-8: V = s.reshape(N, m).T)


def _ca():
    return pl().op8()._ca()


def sp_rh(inp: dict) -> dict:
    """SP-RH (protocol 5 item 9, small grids): direct R-H with its own bound by the exact residual, as in BR 4. With x* the exact
    solution: ||x_H - x*||_2 <= b_H and ||V - x*||_2 <= b_V, both by ||r||_2 / lambda_low on the same certificate, so
    ||V - x_H||_2 <= b_H + b_V must hold exactly; otherwise a certified value is wrong (CONTRADICTION). The error of V is recorded as the
    interval [d - b_H, d + b_H], d = ||V - x_H||_2; its size is not judged (no tolerance assigned; budget 1/100 recorded)."""
    cfg = inp["cfg"]
    br = adapter().br()
    if (cfg.nx - 1) * (cfg.ny - 1) > br.RH_LIMIT:
        return {"status": UNRESOLVED, "reason": f"(n_x - 1)(n_y - 1) > {br.RH_LIMIT}: R-H is outside its rule (protocol 5 item 9)"}
    ps = _half_step(inp)
    st, xh = hs.solve_hessian(ps)
    if xh is None or not np.all(np.isfinite(xh)):
        return {"status": UNRESOLVED, "reason": f"R-H gave no finite solution: {st}"}
    coeffs = hs.exact_coefficients(ps)
    tau, lam = _lambda_low(ps, coeffs)
    if lam is None:
        return {"status": UNRESOLVED, "reason": "no certificate of lambda_min(K) > 0", "tau_trial": tau}
    v = _term_first(inp["V"])
    b_h, b_v = hs.error_bound(ps, xh, lam, coeffs), hs.error_bound(ps, v, lam, coeffs)
    if not (np.isfinite(b_h) and np.isfinite(b_v)):
        return {"status": UNRESOLVED, "reason": "a bound by the exact residual overflowed", "lambda_low": lam}
    d2 = sum(((Fraction(float(a)) - Fraction(float(c))) ** 2 for a, c in zip(v, xh, strict=True)), Fraction(0))
    bh, bv = Fraction(b_h), Fraction(b_v)
    lo, hi = max(Fraction(0), br.sqrt_down(d2) - bh), br.sqrt_up(d2) + bh
    holds = d2 <= (bh + bv) ** 2
    return {"status": PASS if holds else CONTRADICTION, "reason": None if holds else "||V - x_H||_2 > b_H + b_V: a certified bound is wrong",
            "lambda_low": lam, "tau_trial": tau, "R_H_bound_2_up": b_h, "V_bound_2_up": b_v, "V_error_2_down": br.dec_down(lo),
            "V_error_2_up": br.dec_up(hi), "budget_1_100": "SATISFIED" if br.budget_ok(bh, lo) else "NOT_MET (comparison UNRESOLVED; item 8)"}


def _measured(hat, exact) -> dict:
    """max |hat - exact| exactly, and its size against max |exact|, rounded upwards. A measurement: no tolerance is assigned."""
    hat = np.atleast_2d(np.asarray(hat, dtype=float))
    rows = [exact] if not isinstance(exact[0], list) else exact
    worst, scale = Fraction(0), Fraction(0)
    for i, row in enumerate(rows):
        for j, x in enumerate(row):
            worst = max(worst, abs(Fraction(float(hat[i, j])) - x))
            scale = max(scale, abs(x))
    return {"max_abs_diff_up": hs._float_up(worst), "relative_up": hs._float_up(worst / scale) if scale else None}


def _c3plus_k(inp: dict):
    rec = inp["record"] or {}
    return (rec.get("c3plus_premises") or {}).get("k")


def sp_large(inp: dict) -> dict:
    """SP-LARGE (protocol 5 item 9, large grids): exact A, B, C, F of the stored binary64 set against the doubles of the row (A, B, F of
    the prototype, C-3+ of the journal), the theorem bound of C-3+, the exact residual R-HP of the original system on V and lambda_low.
    Contradictions: the theorem bound of C-3+ violated; eta_QA of the record not reproduced by the exact residual."""
    ps = _half_step(inp)
    A, B, _, F = bc.bc_coefficients(ps)
    C = inp["C3plus"]
    if C is None or any(not np.all(np.isfinite(x)) for x in (A, B, C, F)):
        return {"status": UNRESOLVED, "reason": "a double coefficient is absent or not finite"}
    k = _c3plus_k(inp)
    if k is None:
        return {"status": UNRESOLVED, "reason": "the record has no k of the C-3+ theorem"}
    Ax, Bx, Cx, Fx = coeffs = hs.exact_coefficients(ps)
    out = {"A": _measured(A, Ax), "B": _measured(B, Bx), "C": _measured(C, Cx), "F": _measured(F[None, :], Fx)}
    chk = _ca().check_c3plus(C, Cx, k)
    out["C3plus_theorem_bound"] = "HOLDS" if chk["elementwise"] and chk["frobenius"] else "VIOLATED"
    v = _term_first(inp["V"])
    r = hs.residual_exact(ps, v, coeffs)
    out["residual_inf_up"] = hs._float_up(max((abs(x) for x in r), default=Fraction(0)))
    eta = bc._eta(ps, v, coeffs)
    out["eta_qa"], out["eta_qa_recorded"] = eta, (inp["record"] or {}).get("eta_qa")
    tau, lam = _lambda_low(ps, coeffs)
    out.update(tau_trial=tau, lambda_low=lam)
    if out["C3plus_theorem_bound"] != "HOLDS":
        return {**out, "status": CONTRADICTION, "reason": "the theorem bound of C-3+ is violated by the exact C"}
    if eta != out["eta_qa_recorded"]:
        return {**out, "status": CONTRADICTION, "reason": "eta_QA of the record is not reproduced by the exact residual on V"}
    if lam is None:
        return {**out, "status": UNRESOLVED, "reason": "no certificate of lambda_min(K) > 0"}
    b = hs.error_bound(ps, v, lam, coeffs)
    if not np.isfinite(b):
        return {**out, "status": UNRESOLVED, "reason": "the bound by the exact residual overflowed"}
    return {**out, "status": PASS, "reason": None, "V_bound_2_up": b}


def sp_c(inp: dict) -> dict:
    """SP-C (protocol 5 item 9; D-054.3 item 2): the exact rational C of the stored binary64 set is the reference; the C-3+ of the row is
    checked against it (its theorem bound) and is bitwise the C-3+ assembly of that set; CONTROL is the separate double cross-check,
    its error is a measurement."""
    ps = _half_step(inp)
    C = inp["C3plus"]
    if C is None or not np.all(np.isfinite(C)):
        return {"status": UNRESOLVED, "reason": "the C-3+ of the row is absent or not finite"}
    k = _c3plus_k(inp)
    if k is None:
        return {"status": UNRESOLVED, "reason": "the record has no k of the C-3+ theorem"}
    ca = _ca()
    Cx = hs.exact_coefficients(ps)[2]
    ctl = ca.c_control(ps)
    if not np.all(np.isfinite(ctl)):
        return {"status": UNRESOLVED, "reason": "CONTROL is not finite"}
    chk = ca.check_c3plus(C, Cx, k)
    assembled = bool(np.array_equal(ca.c3plus(ps)["C"], C))
    out = {"k": k, "C3plus_error_up": ca.c_error(C, Cx), "CONTROL_error_up": ca.c_error(ctl, Cx),
           "C3plus_vs_CONTROL_max_abs_up": hs._float_up(max(abs(Fraction(float(a)) - Fraction(float(b))) for a, b in
                                                            zip(C.ravel(), ctl.ravel()))),
           "C3plus_spd_exact": bool(ca.spd_exact(C)), "CONTROL_spd_exact": bool(ca.spd_exact(ctl)), "C3plus_bitwise_the_assembly": assembled,
           "C3plus_theorem_bound": "HOLDS" if chk["elementwise"] and chk["frobenius"] else "VIOLATED"}
    if out["C3plus_theorem_bound"] != "HOLDS":
        return {**out, "status": CONTRADICTION, "reason": "the theorem bound of C-3+ is violated by the exact C"}
    if not assembled:
        return {**out, "status": CONTRADICTION, "reason": "the C-3+ of the row is not the C-3+ assembly of its stored set"}
    return {**out, "status": PASS, "reason": None}


def sp_op(inp: dict) -> dict:
    r = rd().sp_op_check(inp["cfg"], inp["s"], inp["solved"])
    st = r.get("status")
    return {"status": PASS if st == "HOLDS" else CONTRADICTION, "reason": None if st == "HOLDS" else f"SP-OP: {st}", "sp_op_status": st,
            "entries_checked": r.get("entries_checked")}


def br_reference(inp: dict, spec: dict) -> dict:
    if spec["kind"] == "journal":
        return adapter().reference_for_journal(Path(spec["journal"]), inp["s"])
    return adapter().reference(inp["cfg"], inp["s"], inp["solved"], inp["V"])


ACCURACY_FIELDS = ("lambda_low", "K_SPD_certified", "precision", "refine_converged", "budget_set", "budget_field", "budget_inf",
                   "set_error_inf_down", "set_error_inf_up", "BC1_rel_inf_down", "BC1_rel_inf_up", "BC1_rel_inf", "TS1_e_field_down",
                   "TS1_e_field_up", "TS1_e_field", "TS1_candidate_10_u_ns2", "TS1_observation", "independent_paths_consistent")


def accuracy_record(ref: dict, selected_br: bool) -> dict:
    """BC-1 and TS-1 (protocol 5.1): OBSERVATION. An exceedance of the TS-1 candidate is an observation; a reference without its
    certificate, convergence or budget leaves the observation UNRESOLVED; neither touches the status of the row."""
    res = str(ref.get("result"))
    return {"semantics": "OBSERVATION (protocol 5, items 7-8; 5.1; D-054.4): never a status of the row, never FAIL_COMPUTATION",
            "basis": ("frozen subset SP-BR" if selected_br else "protocol 5.1 (reference applied to each new fixed input), D-057.3 item 1"),
            "norm": "OD-2: elementwise max of the set; basis: the solved set of input_hashes",
            "status": "RESOLVED" if res.startswith("RESOLVED") else UNRESOLVED, "reason": None if res.startswith("RESOLVED") else res,
            **{k: ref[k] for k in ACCURACY_FIELDS if k in ref}}


CHECK_METHODS = {
    "SP-RH": ("direct R-H (D2 Hessian restricted, dense Cholesky) with the bound by the exact residual; lambda_low", (HS_MODULE, SELF), sp_rh),
    "SP-LARGE": ("exact A, B, C, F against the doubles; exact residual R-HP on V; certified lambda_low", (HS_MODULE, BC_MODULE, CA_TOOL, SELF), sp_large),
    "SP-C": ("exact rational C of the stored set; C-3+ theorem bound; CONTROL (variant 1) in double", (HS_MODULE, CA_TOOL, SELF), sp_c),
    "SP-OP": ("exact node-wise operator D2 against the tensor K^C", (READY_TOOL,), sp_op),
}
SECOND_PATHS = ("SP-RH", "SP-LARGE", "SP-C", "SP-BR", "SP-OP")


def _blobs(paths) -> dict:
    return {t: pl().git_blob_id(ROOT / t) for t in paths}


def second_paths(job: dict) -> dict:
    """Every selected check of the row and the accuracy observation of BC-1 / TS-1, each with its inputs, method, component blobs,
    status and reason."""
    spec, selected = job["spec"], [p for p in SECOND_PATHS if p in (job.get("subset_paths") or [])]
    out = {}
    try:
        inp = row_arrays(spec)
    except pl().D1Refused:
        raise
    except Exception as exc:  # recorded, never hidden
        reason = f"the arrays of the row are not readable: {type(exc).__name__}: {str(exc)[-300:]}"
        return {name: {"check": name, "status": UNRESOLVED, "reason": reason} for name in [*selected, ACCURACY]}
    common = {"row_key": job.get("row_key"), "input_hashes": inp["input_hashes"]}
    for name in selected:
        if spec["kind"] != "journal":
            out[name] = {"check": name, **common, "status": BLOCKED_FOR_OWNER,
                         "reason": f"{name} is defined on an own trajectory (journal row); no approved path for a {spec['kind']} row"}
            continue
        if name == "SP-BR":
            continue
        method, tools, fn = CHECK_METHODS[name]
        rec = {"check": name, **common, "method": method, "implementation": _blobs(tools), "pass_means": PASS_MEANS}
        if inp["binding_defects"]:
            out[name] = {**rec, "status": UNRESOLVED, "reason": "input binding: " + "; ".join(inp["binding_defects"])}
            continue
        pl().stage_event(f"second path {name}", "start")
        try:
            out[name] = {**rec, **fn(inp)}
        except pl().D1Refused:
            raise
        except Exception as exc:  # recorded, never hidden
            out[name] = {**rec, "status": UNRESOLVED, "reason": f"{type(exc).__name__}: {str(exc)[-300:]}"}
    pl().stage_event("reference BR", "start")
    method = "BR: exact assembly, decimal band Cholesky, exact residual, lambda_low, budget 1/100 (reference adapter)"
    rec = {**common, "method": method, "implementation": _blobs((ADAPTER_TOOL, BR_TOOL, HS_MODULE))}
    selected_br = "SP-BR" in selected and spec["kind"] == "journal"
    try:
        ref = br_reference(inp, spec) if not inp["binding_defects"] else {"result": f"{UNRESOLVED}: input binding: " + "; ".join(inp["binding_defects"])}
    except pl().D1Refused:
        raise
    except Exception as exc:  # recorded, never hidden
        ref = {"result": f"{UNRESOLVED}: {type(exc).__name__}: {str(exc)[-300:]}"}
    out[ACCURACY] = {"check": ACCURACY, **rec, **accuracy_record(ref, selected_br)}
    if selected_br:
        res = str(ref.get("result"))
        st = PASS if res.startswith("RESOLVED") else CONTRADICTION if "CONTRADICTION" in res else UNRESOLVED
        out["SP-BR"] = {"check": "SP-BR", **rec, "pass_means": PASS_MEANS, "status": st, "reason": None if st == PASS else res,
                        "reference": json.loads(_dump(ref))}
    return out


def _not_executed_record(row: dict, reason: str, resources: dict | None = None, execution: str = EXEC_PLAN) -> dict:
    ne = {"reason": reason}
    if resources is not None:
        ne.update(resources=True, measured=resources)
    inp = {"executed": False, "not_executed": ne}  # the input of this record is made here (no pipeline call): it is the source
    res = pf().classify_row(inp)
    return {"key": row["key"], "block": row["block"], "N": row["N"], "s": row["s"], "mandatory": row["mandatory"], "executed": False,
            "execution_status": execution, "status": res["status"], "reasons": res["reasons"], "input": encode_input(inp), "z4": None,
            "tool_defects": []}


def _planned_reason(row: dict, cfg) -> str:
    if row["N"] >= 2 and row["s"] >= 2:
        return pl().rank_guard_reason(cfg, row["s"])  # D-057.1: trajectory STOPPED_RANK_GUARD_UNCERTIFIED, row UNRESOLVED
    return row["reason"]


def _job_fault(r: dict, what: str) -> tuple[str, dict | None, str]:
    if "aborted" in r:
        a = r["aborted"]
        who = f"stage {a.get('stage') or a.get('process_stage')!r}, process {a.get('pid')}"
        return f"PL 6.3: {what} job stopped ({a['aborted']}; {who})", {**a, "seconds": r.get("seconds")}, EXEC_RESOURCES
    return f"not executed: {what} job failed ({r.get('failed')}): {str(r.get('tail', ''))[-400:]}", None, EXEC_FAILED


def _row_job(row: dict, spec: dict, subsets: dict, work: Path, limit: int, signal, dry_run: bool, factorization_limit_s=None) -> dict:
    paths = list(subsets.get(row["key"], []))
    job = {"kind": "row", "spec": spec, "mandatory": row["mandatory"], "sp_be": "SP-BE" in paths, "dry_run": dry_run}
    r = run_job(job, work / "row", limit, signal, factorization_limit_s)
    if "aborted" in r or "failed" in r:
        reason, measured, ex = _job_fault(r, "row")
        rec = _not_executed_record(row, reason, measured, ex)
        rec["resources"] = r.get("resources")
        return rec
    a = r["analysis"]
    rec = {"key": row["key"], "block": row["block"], "N": row["N"], "s": row["s"], "mandatory": row["mandatory"],
           "executed": a.get("row") is not None, "execution_status": EXEC_DONE if a.get("row") is not None else EXEC_PLAN,
           "status": a["status"], "reasons": a["classification"].get("reasons"), "input": r["input"], "z4": a.get("z4"),
           "tool_defects": list(a.get("tool_defects") or []), "process": r["process"], "resources": r["resources"], "seconds": r["seconds"],
           "replay": a.get("reproduction"), "input_hashes": (a.get("row") or {}).get("inputs_sha"), "analysis": a}
    names = [*[p for p in SECOND_PATHS if p in paths], ACCURACY]
    if not rec["executed"]:
        rec["checks"] = {name: {"check": name, "status": UNRESOLVED, "reason": f"row not executed: {a.get('not_executed_reason')}"}
                         for name in names}
        return rec
    sjob = {"kind": "second_paths", "spec": spec, "row_key": row["key"], "subset_paths": paths, "dry_run": dry_run}
    sp = run_job(sjob, work / "checks", limit, signal, factorization_limit_s)
    if "aborted" in sp or "failed" in sp:  # OD-7: of the checks, not of the row
        st = UNRESOLVED_RESOURCES_OF_CHECK if "aborted" in sp else UNRESOLVED
        why = sp.get("aborted") or {"failed": sp.get("failed"), "tail": sp.get("tail", "")[-400:]}
        rec["checks"] = {name: {"check": name, "status": st, "reason": why} for name in names}
    else:
        rec["checks"] = sp["checks"]
    rec["checks_resources"] = sp.get("resources")
    for name, c_ in rec["checks"].items():
        if c_.get("status") == CONTRADICTION:  # a defect of a tool: the verdict is not issued until it is removed (D-057.3 item 4)
            rec["tool_defects"].append(f"SECOND_PATH_CONTRADICTION {name}: {c_.get('reason')}")
    return rec


def run_unit(unit: dict, work: Path, limit: int, signal, subsets: dict, dry_run: bool, jobs: list, factorization_limit_s=None) -> dict:
    cfg = _cfg(unit)
    rows = [None] * len(unit["rows"])
    execs = [i for i, r in enumerate(unit["rows"]) if r["executable"]]
    for i, r in enumerate(unit["rows"]):
        if not r["executable"]:  # never computed
            rows[i] = _not_executed_record(r, _planned_reason(r, cfg), execution=EXEC_GUARD if r["N"] >= 2 and r["s"] >= 2 else EXEC_PLAN)
    rac = [{"key": r["key"], "s": r["s"], "planned_executable": r["executable"],
            "trajectory_reason": None if r["executable"] else _planned_reason(r, cfg)} for r in unit["rac"]]
    if not execs:
        return {"unit": unit["id"], "rows": rows, "rac": rac, "prep": None}
    kind = "paired" if unit["block"] == "C" else "journal"
    prep = run_job({"kind": kind, "config": unit["config"], "out": str(work / "data"), "dry_run": dry_run}, work / "prep", limit,
                   signal, factorization_limit_s)
    jobs.append(unit["id"] + "|prep")
    if "aborted" in prep or "failed" in prep:
        reason, measured, ex = _job_fault(prep, kind)
        for i in execs:
            rows[i] = _not_executed_record(unit["rows"][i], reason, measured, ex)
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
        rows[i] = _row_job(r, spec, subsets, work / f"s{r['s']}", limit, signal, dry_run, factorization_limit_s)
        jobs.append(f"{unit['id']}|s{r['s']}")
    if kind == "paired":
        recs = {x.get("s"): x for x in (prep.get("chain") or {}).get("records", []) if isinstance(x, dict)}
        for x in rac:
            x["chain_record_status"] = (recs.get(x["s"]) or {}).get("status")
            x["chain_certificate"] = (recs.get(x["s"]) or {}).get("certificate")  # of the R-AC reference; OD-7: not of the row
    return {"unit": unit["id"], "rows": rows, "rac": rac,
            "prep": {"process": prep.get("process"), "resources": prep.get("resources"), "seconds": prep.get("seconds"),
                     "chain_status": (prep.get("paired") or {}).get("chain_status")}}


def second_path_coverage(records: list[dict], subsets: dict) -> dict:
    """For each check: the selected rows and what happened on each; a row that was not established is listed with its reason, so that
    missing coverage can never read as protocol coverage. OBSERVATION: it does not change a status of a row."""
    by = {r["key"]: r for r in records}
    out = {}
    for name in SECOND_PATHS:
        sel = sorted(k for k, v in subsets.items() if name in v and k in by)
        entry = {"selected": len(sel), "pass": 0, "not_established": [], "contradictions": []}
        for k in sel:
            c_ = (by[k].get("checks") or {}).get(name)
            st = (c_ or {}).get("status")
            if st == PASS:
                entry["pass"] += 1
            elif st == CONTRADICTION:
                entry["contradictions"].append({"key": k, "reason": c_.get("reason")})
            else:
                entry["not_established"].append({"key": k, "status": st or "NOT_RUN", "reason": (c_ or {}).get("reason") or
                                                 f"row {by[k].get('execution_status')}: {by[k].get('status')}"})
        entry["complete"] = entry["pass"] == entry["selected"]
        out[name] = entry
    acc = [r for r in records if r.get("execution_status") == EXEC_DONE]
    res = [r for r in acc if ((r.get("checks") or {}).get(ACCURACY) or {}).get("status") == "RESOLVED"]
    out[ACCURACY] = {"semantics": "OBSERVATION", "executed_rows": len(acc), "resolved": len(res),
                     "unresolved_keys": sorted(r["key"] for r in acc if r not in res)}
    return out


def relabel(o, status: str):
    """The data label of this run on every nested record; the label a reused tool wrote is kept as ``tool_data_status`` (the constants
    of the tools are not changed: the files the tools write under work/ keep their own labels, PROVENANCE_README.json says so)."""
    if isinstance(o, dict):
        out = {k: relabel(v, status) for k, v in o.items()}
        if "data_status" in out and out["data_status"] != status:
            out["tool_data_status"] = out["data_status"]
            out["data_status"] = status
        return out
    if isinstance(o, list):
        return [relabel(x, status) for x in o]
    return o


# ---------------------------------------------------------------------------------------------------------------------------------
# before the run, verdict
# ---------------------------------------------------------------------------------------------------------------------------------

def platform_defects() -> list[str]:
    ap = rd().APPROVED_PLATFORM
    bad = []
    if sys.platform != ap["sys_platform"] or platform.python_version() != ap["python"]:
        bad.append(f"platform {sys.platform} / Python {platform.python_version()} is not the approved one (OD-6)")
    if not memory_measurement_works() or not pf().total_ram_bytes():
        bad.append("the PL 6.3 monitor cannot measure the processes or the physical RAM here")
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
    tools = sorted({READY_TOOL, SELF, pl().SELF, CONTROLS_TOOL, ADAPTER_TOOL, BR_TOOL, HS_MODULE, BC_MODULE, *pl().USED_TOOLS})
    return {"dry_run": dry_run, "head": pl()._git_out(ROOT, "rev-parse", "HEAD")[1].strip(), "launch_signal": signal,
            "tool_blobs": {t: pl().git_blob_id(ROOT / t) for t in tools}, "python": platform.python_version(),
            "sys_platform": sys.platform, "numpy": np.__version__}


def _plan_sha(plan: list[dict]) -> str:
    return hashlib.sha256(pf().canonical(plan)).hexdigest()


PROVENANCE_README = ("campaign_result.json, units/ and ledger.json carry the label of this run (data_status); the label a reused tool wrote "
                     "is kept as tool_data_status. The files the frozen tools write under work/ (journals, paired arrays, chain.json) keep "
                     "their own PRELAUNCH labels and are data of this run only through this directory and state.json.")


def block_d_rh(plan: list[dict]) -> dict:
    """Protocol 5 item 9: R-H on all the grids of block D: every executable row of block D, N = 1 (frozen rule) and (14, 6), N = 3 at s = 1
    (the grid is named by the protocol; the frozen rule still says it was blocked before D-057.1: reported to the Reviewer)."""
    return {r["key"]: ["SP-RH"] for r in plan if r["kind"] == "row" and r["block"] == "D" and r["executable"]}


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_campaign(out: Path, plan: list[dict] | None = None, dry_run: bool = False, memory_limit_bytes: int | None = None,
                 stop_after_jobs: int | None = None, subsets: dict | None = None, factorization_limit_s: float | None = None) -> dict:
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
        if plan is not None or memory_limit_bytes is not None or subsets is not None or factorization_limit_s is not None:
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
        subsets.update(block_d_rh(plan))
    subsets = subsets or {}
    limit = memory_limit_bytes if memory_limit_bytes is not None else memory_limit()
    prov = _provenance(dry_run, signal)
    state_path, ledger_path = out / "state.json", out / "ledger.json"
    ledger = {}
    if state_path.exists():
        st = json.loads(state_path.read_text(encoding="utf-8"))
        if st["provenance"] != prov or st["plan_sha256"] != _plan_sha(plan):
            raise CampaignRefused("restart refused: provenance or plan differs from the interrupted run")
        ledger = json.loads(ledger_path.read_text(encoding="utf-8")) if ledger_path.exists() else {}
        stored = {p.stem: p for p in (out / "units").glob("*.json")} if (out / "units").exists() else {}
        changed = sorted(k for k, p in stored.items() if ledger.get(k) != _sha_file(p)) + sorted(set(ledger) - set(stored))
        if changed:
            raise CampaignRefused(f"restart refused: stored units differ from the ledger: {changed}")
        st["starts"].append(time.strftime("%Y-%m-%dT%H:%M:%S"))
    else:
        st = {"provenance": prov, "plan_sha256": _plan_sha(plan), "starts": [time.strftime("%Y-%m-%dT%H:%M:%S")]}
    _write_json(state_path, st)
    _write_json(out / "PROVENANCE_README.json", {"text": PROVENANCE_README})
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
            doc = {"data_status": status, "state": "BLOCKED_BEFORE_THE_RUN", "prerun": pre, "scope": "no row was computed", "accounting": plan_accounting(plan),
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
        res = relabel(run_unit(u, work, limit, signal, subsets, dry_run, jobs, factorization_limit_s), status)
        _write_json(upath, res)
        ledger[upath.stem] = _sha_file(upath)
        _write_json(ledger_path, ledger)
        results.append(res)
        if stop_after_jobs is not None and len(jobs) >= stop_after_jobs:
            raise CampaignInterrupted(f"stopped after {len(jobs)} jobs (test hook)")
    records = [x for u in results for x in u["rows"]]
    doc = {"data_status": status, "state": "COMPLETED", "accounting": plan_accounting(plan), "prerun": pre, "provenance": prov,
           "restarts": st["starts"], "memory_limit_bytes": limit, "second_path_coverage": second_path_coverage(records, subsets),
           "scope": "no scientific qualification and no owner acceptance is claimed; N = 2, 3 RESEARCH_ONLY / NOT_QUALIFIED; second paths and "
                    "BC-1 / TS-1 are observations",
           "limitations": ["PASS of a second path means established without contradiction, not accuracy (no tolerance assigned)",
                           "A, B, F of SP-LARGE: measured differences only (no approved bound)"],
           "rows": [{k: x.get(k) for k in ("key", "block", "N", "s", "mandatory", "executed", "execution_status", "status", "z4", "tool_defects")}
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


def build_addendum(controls_results: Path, base_commit: str, surrogate_costs: Path | None = None) -> dict:
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
                   "carry_forward": "applicable to the unchanged computation: _compute_row_core, OP-8, the block-C driver, the BR tool and the "
                                    "numerical environment; they do not establish the behaviour of the runtime monitor, tested separately "
                                    "(tests/reference/test_r5_cccc_q4_d1_campaign_gaps.py), nor the cost of SP-RH, SP-LARGE and SP-C, measured only on "
                                    "non-D1 grids of comparable size (second_path_costs_on_non_D1_grids)",
                   "second_path_costs_on_non_D1_grids": None if surrogate_costs is None else json.loads(Path(surrogate_costs).read_text(encoding="utf-8")),
                   "runtime_enforcement": "monitor_job: peak of every process of the job tree against 50 % RAM, every factorization stage "
                                          "against 30 min, from the parent; limits unchanged (D-057.4)"},
        "second_paths": {"SP-RH": "hs.solve_hessian, hs.error_bound, hs.certify_lambda_min", "SP-LARGE": "hs.exact_coefficients, "
                         "bc.bc_coefficients, hs.residual_exact, hs.certify_lambda_min, check_c3plus", "SP-C": "hs.exact_coefficients, "
                         "c3plus, c_control, check_c3plus", "SP-BR": "reference adapter (BR)", "SP-OP": "ready.sp_op_check",
                         "pass_means": PASS_MEANS, "contradiction": "tool defect: verdict not issued (D-057.3 item 4)"},
        "accuracy_observations": {"BC-1, TS-1": "BR reference on every executed classified row; OBSERVATION"},
        "limitations": ["A, B, F of SP-LARGE and the CONTROL error: measured values only (no approved bound)",
                        "BC-1 / TS-1 outside SP-BR: reading of protocol 5.1 and D-057.3 item 1, to be confirmed by the Reviewer",
                        "SP-RH on (14, 6), N = 3, s = 1: the protocol names the grid, the frozen rule predates D-057.1",
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
    a_ = sub.add_parser("addendum")
    a_.add_argument("--controls-results", type=Path, required=True)
    a_.add_argument("--base-commit", required=True)
    a_.add_argument("--surrogate-costs", type=Path, default=None)
    a = ap.parse_args(argv)
    if a.cmd == "_job":
        try:
            res = _job_child(a.spec)
        except pl().D1Refused as exc:
            print(f"refused: {exc}", file=sys.stderr)
            return 2
        print(_dump(res))
        return 0
    if a.cmd == "addendum":
        _write_json(ROOT / pl().CAMPAIGN_ADDENDUM, build_addendum(a.controls_results, a.base_commit, a.surrogate_costs))
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
