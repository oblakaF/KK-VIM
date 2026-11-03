"""R5-V Q4 QA-D1-READY (D-057.1...D-057.6): the integrated PRELAUNCH layer of the D1 campaign (protocol
docs/R5_CCCC_Q4_D1_PROTOCOL.md 3-6, 8, 9, 12, 13). QA research tool. THIS IS NOT THE D1 CAMPAIGN: D1_START = BLOCKED; the D1 grids are
refused before any computation of a row (``campaign`` refuses always); no candidate verdict is issued for D1. No import of the
working ``kkvim``; no new formula, threshold, tolerance or row status; the classifier of PR #48 (``classify_row``,
``campaign_verdict``) and its exact arithmetic are used unchanged.

Composition (existing components used read-only, none rebuilt):
- OP-8 own trajectories with the rank protection of D-057.1 (scripts/r5_cccc_q4_d1_c3plus_trajectories.py);
- the row pipeline of PR #50 (scripts/r5_cccc_q4_d1_row_pipeline_prelaunch.py), extended by QA-D1-READY with rows N = 2, 3 at s = 1,
  paired block-C rows and the records of F-2 / F-3 / SP-BE;
- the block-C driver (scripts/r5_cccc_q4_d1_block_c_driver.py; OD-3);
- the second path of B_E (PR #51 with N-1, N-2; scripts/r5_cccc_q4_be_second_path_research.py) and the second path of gamma
  (``energies_path_b`` of the gamma audit);
- the classifier and the manifest tool of PR #48 (scripts/r5_cccc_q4_d1_preflight.py).

What this layer adds (D-057.3):
1. F-2 - binding of the actual inputs: ``pl.actual_input_defects`` on every row; a defect makes П-а NOT_ESTABLISHED and the row is
   classified again by the unchanged ``classify_row`` (via the pipeline's ``assemble``): UNRESOLVED_CERTIFICATE, never PASS. The same
   for a process whose effective BLAS thread count (read from the library) is not 1 (OD-6: set by the launching process before
   NumPy is imported; the bitwise binding of OD-12 holds only for the fixed thread setting).
2. F-3 - on every executed row (SP-GAMMA: all 102 mandatory rows and every other row): inclusion of the exact independent gamma_B in
   [gamma_down, gamma_up] and narrowness gamma_down = RD(gamma_B), gamma_up = RU(gamma_B) bitwise. A violation is a defect of path
   B1 (tool defect): the row status stays the one of protocol 4.2, the candidate verdict is NOT issued until the cause is removed.
3. SP-BE - the second path of B_E on the rows of the SP-BE subset and in Z-4 (P^ mandatory, N-3; QA-EB representation, N-5); a
   mismatch is the channel of G-5 and a tool defect.
4. Z-4 - the confirmation procedure (``z4_confirm``) for every mandatory row with UNRESOLVED_RESOLVABILITY and every row with
   FAIL_COMPUTATION, before its final status: (1) gamma_B, inclusion and narrowness; (2) second path of B_E and |E_s| <= B_E2;
   (3) the binding of the actual inputs repeated in a fresh analysis (separate process included). Outcomes: CONFIRMED;
   DEFECT_GAMMA_INCLUSION; DEFECT_GAMMA_NARROWNESS; DEFECT_BE_SECOND_PATH; CONTRADICTION_E2; BINDING_MISMATCH (-> П-а not
   established, UNRESOLVED_CERTIFICATE). The row status is decided by the tree of protocol 4.2 only.
5. ``verdict``: the campaign verdict of PR #48 (unchanged) after Z-4; it is NOT issued (VERDICT_NOT_ISSUED) while a required Z-4 is
   missing or a defect of a tool was found (D-057.3 item 4).
6. Row plan of D1 (``row_plan``) under D-057.1: rows executed and rows UNRESOLVED with the precise reason; denominator 102 unchanged.
7. Platform record and freeze rule (OD-6, OD-12), archive of the start sets (``start_archive``), input hashes, subsets (OD-11:
   SP-GAMMA 102, SP-BE = PRE rule + (320, 1280) and (400, 1000) at s = 2, SP-OP), set infinity norm of OD-2 (``set_inf_norm``),
   geometry of block D (OD-1, 1 x 1), D = q = 1 (OD-5).
8. SP-OP (``sp_op_check``): exact node-wise application of the operator of the discretisation D2 (the mesh of the second B_E path)
   to fields U (x) e_k against the exact tensor K^C of the half-step: a_h(phi_p, phi_q) = h_s K^C[p, q] exactly (BR 3: ||U~ V^T||_a =
   h_s^(1/2) ||x||_K) for the checked columns q.
9. Resource probes (``run_probes``): PL 6.3 unchanged, resources and reference certificates recorded apart (OD-7), s = 2, 3 only on
   surrogate inputs fixed before the run, N = 3 at s = 1 (OD-9); no value of a row is recorded.

Data: PRELAUNCH research observation, NOT D1 data, never D1_RESULT.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import subprocess
import sys
import threading
import time
from fractions import Fraction
from pathlib import Path

THREAD_VARS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
for _k in THREAD_VARS:
    os.environ[_k] = "1"  # OD-6: single-threaded BLAS, set before NumPy is imported; the effective count is read from the library

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SELF = "scripts/r5_cccc_q4_d1_ready.py"
PIPELINE_TOOL = "scripts/r5_cccc_q4_d1_row_pipeline_prelaunch.py"
PREFLIGHT_TOOL = "scripts/r5_cccc_q4_d1_preflight.py"
BLOCK_C_TOOL = "scripts/r5_cccc_q4_d1_block_c_driver.py"
BR_TOOL = "scripts/r5_cccc_q4_bc_reference_audit.py"
D1_START = "BLOCKED"
DATA_STATUS = "PRELAUNCH research observation (QA-D1-READY); NOT D1 data, never D1_RESULT (D1_START = BLOCKED)"
D_VALUE = Q_VALUE = 1.0  # OD-5 (D-057.2 item 2)
BLOCK_D_GEOMETRY = (1.0, 1.0)  # OD-1 (D-057.2 item 1): l_x = l_y = 1
APPROVED_PLATFORM = {"sys_platform": "win32", "python": "3.13.15", "blas_threads": 1}  # OD-6 (D-057.2 item 3)
S_MAX = 8
CERT, UNRES, UNRES_RES, FAIL, PASS = "UNRESOLVED_CERTIFICATE", "UNRESOLVED", "UNRESOLVED_RESOLVABILITY", "FAIL_COMPUTATION", "PASS"
VERDICT_NOT_ISSUED = "VERDICT_NOT_ISSUED"
Z4_OUTCOMES = ("CONFIRMED", "DEFECT_GAMMA_INCLUSION", "DEFECT_GAMMA_NARROWNESS", "DEFECT_BE_SECOND_PATH", "CONTRADICTION_E2",
               "BINDING_MISMATCH")
SP_BE_EXTRA = ((320, 1280, 2), (400, 1000, 2))  # OD-11 (D-057.3 item 6): large configurations of SP-BE at s = 2


class D1Blocked(RuntimeError):
    """D1_START = BLOCKED: no campaign row is executed by this tool."""


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


def pf():
    return pl().pf()


def bcd():
    return pl().bcd()


def op8():
    return pl().op8()


def git_blob_id(path: Path) -> str:
    data = Path(path).read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def campaign(*_a, **_k):
    """The D1 campaign is not run by this tool (D1_START = BLOCKED; protocol 12, item 6: no owner permission)."""
    raise D1Blocked("refused: D1_START = BLOCKED (protocol 12; no owner permission for the campaign)")


# ---------------------------------------------------------------------------------------------------------------------------------
# row plan of D1 under D-057.1 (blocks A, B, C, D, S; nothing is computed)
# ---------------------------------------------------------------------------------------------------------------------------------

LEVEL2 = ((100, 100), (100, 150), (96, 64), (100, 250), (100, 40), (80, 320), (100, 25))  # protocol 3, block C


def _families():
    return {r: grids for r, grids in pf().FAMILIES}


def _row(block, family, r, level, nx, ny, lx, ly, N, s, kind="row"):
    d = "Y" if s % 2 == 1 else "X"
    n_s = ny if d == "Y" else nx
    row = {"key": pf().row_key(block if kind == "row" else f"{block}-RAC", family, nx, ny, N, s), "block": block, "kind": kind,
           "family": family, "r": r, "level": level, "nx": nx, "ny": ny, "lx_hex": float(lx).hex(), "ly_hex": float(ly).hex(), "N": N,
           "s": s, "direction": d, "n_s": n_s, "mandatory": block == "A" and kind == "row" and s in (1, 2, 3)}
    if N >= 2 and s >= 2:
        row.update({"executable": False, "planned_status": UNRES,
                    "reason": (f"not executed: N = {N}, s = {s} - trajectory {op8().RANK_GUARD} at s = 2 (D-057.1 variant V)")})
    elif N > nx // 2:
        row.update({"executable": False, "planned_status": UNRES, "reason": "S-1 start set (N > floor(n_x/2))"})
    elif s == 1 and N > ny // 2:
        row.update({"executable": False, "planned_status": UNRES, "reason": "S-1 configuration (RN-3): N > floor(n_y/2)"})
    else:
        row.update({"executable": True, "planned_status": None, "reason": None})
    return row


def row_plan() -> list[dict]:
    """Every classified row (824) and every R-AC half-step of block C (168) of protocol 3 with its executability under D-057.1."""
    rows = []
    fam = _families()
    for g in pf().block_a_grids():
        for s in range(1, S_MAX + 1):
            rows.append(_row("A", g["family"], g["r"], g["level"], g["nx"], g["ny"], g["lx"], g["ly"], 1, s))
    for r, grids in fam.items():
        for level, (nx, ny) in enumerate(grids[:3], start=1):
            for N in (2, 3):
                for s in range(1, S_MAX + 1):
                    rows.append(_row("B", f"r={r}", r, level, nx, ny, 1.0, float(Fraction(r)), N, s))
    for r, grids in fam.items():
        nx, ny = grids[1]
        assert (nx, ny) in LEVEL2
        for N in (1, 2, 3):
            for s in range(1, S_MAX + 1):
                for kind in ("row", "rac"):
                    rows.append(_row("C", f"r={r}", r, 2, nx, ny, 1.0, float(Fraction(r)), N, s, kind))
    for nx, ny, N in pf().BLOCK_D:
        for s in range(1, S_MAX + 1):
            rows.append(_row("D", "1x1", "OD-1: 1 x 1", "D", nx, ny, *BLOCK_D_GEOMETRY, N, s))
    for r, nx, ny in pf().BLOCK_S:
        for s in range(1, S_MAX + 1):
            rows.append(_row("S", f"r={r}", r, "S", nx, ny, 1.0, float(Fraction(r)), 1, s))
    return rows


def plan_counts(plan: list[dict]) -> dict:
    out = {}
    for b in ("A", "B", "D", "S"):
        sel = [r for r in plan if r["block"] == b]
        out[b] = {"rows": len(sel), "executable": sum(r["executable"] for r in sel),
                  "not_executed_rank_guard": sum((not r["executable"]) and "RANK_GUARD" in (r["reason"] or "") for r in sel)}
    c = [r for r in plan if r["block"] == "C" and r["kind"] == "row"]
    out["C_paired"] = {"rows": len(c), "executable": sum(r["executable"] for r in c),
                       "not_executed_rank_guard": sum((not r["executable"]) and "RANK_GUARD" in (r["reason"] or "") for r in c)}
    rac = [r for r in plan if r["kind"] == "rac"]
    out["C_RAC_reference_chain"] = {"half_steps": len(rac), "executable": sum(r["executable"] for r in rac),
                                    "stopped_rank_guard": sum(not r["executable"] for r in rac)}
    cls = [r for r in plan if r["kind"] == "row"]
    out["classified"] = len(cls)
    out["mandatory"] = sum(r["mandatory"] for r in cls)
    out["mandatory_executable"] = sum(r["mandatory"] and r["executable"] for r in cls)
    return out


# ---------------------------------------------------------------------------------------------------------------------------------
# RT-17: geometry and the set infinity norm (OD-1, OD-2)
# ---------------------------------------------------------------------------------------------------------------------------------

def geometry_check(row: dict) -> list[str]:
    """Exact geometry of a planned row (protocol 2.1; OD-1): l_x = 1; l_y = fl(r) for a family, 1 for the anisotropic grids and
    block D; n_s of the solved direction. Empty when it holds."""
    bad = []
    lx, ly = float.fromhex(row["lx_hex"]), float.fromhex(row["ly_hex"])
    if lx != 1.0:
        bad.append("l_x != 1")
    if row["family"] == "1x1":
        if ly != 1.0:
            bad.append("l_y != 1 on a 1 x 1 grid")
    elif ly != float(Fraction(row["r"])):
        bad.append("l_y != fl(r)")
    if row["n_s"] != (row["ny"] if row["direction"] == "Y" else row["nx"]):
        bad.append("n_s is not the number of intervals of the solved direction")
    return bad


def set_inf_norm(V) -> float:
    """OD-2 (D-057.3 item 1): ||V||_max = ||vec V||_inf, the largest modulus of the entries of the set (n_s - 1) x N, in the basis
    U~ that is recorded with it (its sha256). N = 1: the vector infinity norm. Exact (a maximum of moduli of binary64 values)."""
    a = np.asarray(V, dtype=float)
    if a.size == 0 or not np.all(np.isfinite(a)):
        raise ValueError("the set must be non-empty and finite")
    return float(np.max(np.abs(a)))


def bc1_record(V_hat, V_ref, basis) -> dict:
    """Form BC-1 data of OD-2 for one row: d_inf = ||V^ - V_ref||_max exactly (rational), the reference norm, the basis hash."""
    a, b = np.asarray(V_hat, dtype=float), np.asarray(V_ref, dtype=float)
    d = max((abs(Fraction(float(x)) - Fraction(float(y))) for x, y in zip(a.ravel(), b.ravel())), default=Fraction(0))
    return {"norm": "elementwise max of the set (OD-2 (i))", "d_inf_exact": f"{d.numerator}/{d.denominator}",
            "ref_norm": set_inf_norm(b), "basis_sha": op8().array_sha(np.asarray(basis, dtype=float))}


# ---------------------------------------------------------------------------------------------------------------------------------
# platform (OD-6, OD-12), start archive, input hashes
# ---------------------------------------------------------------------------------------------------------------------------------

def blas_threads_read() -> int | None:
    return pl().qa().blas_info().get("threads")


def platform_record() -> dict:
    import scipy

    try:
        cfg = np.show_config(mode="dicts")
        blas = {k: cfg["Build Dependencies"].get(k) for k in ("blas", "lapack")}
    except Exception as exc:  # recorded
        blas = {"error": f"{type(exc).__name__}: {exc}"}
    return {"sys_platform": sys.platform, "platform": platform.platform(), "machine": platform.machine(),
            "processor": platform.processor(), "python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__, "blas_lapack": blas, "blas_threads_read_from_library": blas_threads_read(),
            "thread_env_vars": {k: os.environ.get(k) for k in THREAD_VARS},
            "thread_env_set_before_numpy_import": "by this tool at import (OD-6: by the launching process); effective count read"}


def freeze_allowed(rec: dict | None = None) -> dict:
    """OD-6 / OD-12: platform-dependent values are frozen only on Windows, Python 3.13.15, BLAS threads = 1 (read from the library).
    Elsewhere: UNRESOLVED (never PASS)."""
    rec = rec or platform_record()
    ok = (rec.get("sys_platform") == APPROVED_PLATFORM["sys_platform"] and rec.get("python") == APPROVED_PLATFORM["python"]
          and rec.get("blas_threads_read_from_library") == APPROVED_PLATFORM["blas_threads"])
    return {"allowed": ok, "state": "APPROVED_PLATFORM" if ok else "UNRESOLVED (platform not the approved one; not frozen)",
            "approved": APPROVED_PLATFORM}


def start_archive(plan: list[dict] | None = None) -> dict:
    """OD-12 (b): the generated start sets T_N = hs.start_set(n_x, N) of every executable s = 1 row of D1, as float.hex text
    (column-major), with their OP-8 sha256. Platform dependent (numpy.sin): strict on the recorded platform only."""
    from kkvim_reference import r5_cccc_halfstep as hs

    plan = plan if plan is not None else row_plan()
    keys = sorted({(r["nx"], r["N"]) for r in plan if r["s"] == 1 and r["executable"]})
    sets = []
    for nx, N in keys:
        T = hs.start_set(nx, N)
        sets.append({"nx": nx, "N": N, "shape": list(T.shape), "sha": op8().array_sha(T),
                     "columns_hex": [[float(v).hex() for v in T[:, i]] for i in range(N)]})
    return {"document": "start sets T_N of the executable s = 1 rows of D1 (OD-12 (b))", "generator": "hs.start_set(n_x, N)",
            "platform_dependent": "numpy.sin: strict on the recorded platform only (OD-12)", "sets": sets}


def verify_archive(doc: dict) -> list[str]:
    """Generation against the archive, bitwise: text -> array equals hs.start_set(n_x, N) and its sha."""
    from kkvim_reference import r5_cccc_halfstep as hs

    bad = []
    for e in doc["sets"]:
        T = hs.start_set(e["nx"], e["N"])
        A = np.array([[float.fromhex(x) for x in col] for col in e["columns_hex"]], dtype=float).T
        if A.shape != T.shape or A.tobytes() != T.tobytes():
            bad.append(f"({e['nx']}, {e['N']}): archived set differs from the generation")
        if op8().array_sha(A) != e["sha"]:
            bad.append(f"({e['nx']}, {e['N']}): sha256 of the archived set differs")
    return bad


def input_hashes(plan: list[dict] | None = None) -> list[dict]:
    """sha256 of the inputs of every executable s = 1 row known before the run: start set and zero companion (paired rows of block
    C: the same start state of the R-AC chain); s >= 2 inputs are dynamic links (never computed here)."""
    from kkvim_reference import r5_cccc_halfstep as hs

    plan = plan if plan is not None else row_plan()
    out = []
    for r in plan:
        if r["s"] == 1 and r["executable"]:
            out.append({"key": r["key"], "given_in_sha": op8().array_sha(hs.start_set(r["nx"], r["N"])),
                        "companion_in_sha": op8().array_sha(np.zeros((r["ny"] - 1, r["N"])))})
    return out


def subsets(plan: list[dict] | None = None) -> dict:
    """OD-11 (D-057.3 item 6), by features of the rows only: SP-GAMMA all 102 mandatory rows; SP-BE the PRE rule plus (320, 1280) and
    (400, 1000) at s = 2; SP-OP the SP-LARGE rows of PRE (large grids); SP-RH, SP-LARGE, SP-BR, SP-C by the PRE rule unchanged."""
    plan = plan if plan is not None else row_plan()
    mand = [r for r in plan if r["mandatory"]]
    rows = []
    for r in mand:
        f = {k: r[k] for k in ("block", "family", "level", "nx", "ny", "s")}
        paths = set(pf().sample_paths(f))
        paths.add("SP-GAMMA")
        if (r["nx"], r["ny"], r["s"]) in SP_BE_EXTRA:
            paths.add("SP-BE")
        if "SP-LARGE" in paths:
            paths.add("SP-OP")
        rows.append({"key": r["key"], "paths": sorted(paths)})
    names = ("SP-RH", "SP-LARGE", "SP-GAMMA", "SP-BR", "SP-C", "SP-BE", "SP-OP")
    counts = {p: sum(p in x["paths"] for x in rows) for p in names}
    rule = {"SP-GAMMA": "all 102 mandatory rows (D-057.3 item 6)", "SP-BE": "PRE rule + (320, 1280), (400, 1000) at s = 2",
            "SP-OP": "exact node-wise D2 against the tensor K^C: the SP-LARGE rows", "others": "PRE rule (scripts/r5_cccc_q4_d1_preflight.py)"}
    return {"rule": rule, "rows": rows, "counts": counts, "sha256": hashlib.sha256(pf().canonical(rows)).hexdigest()}


# ---------------------------------------------------------------------------------------------------------------------------------
# the integrated row: F-2, F-3, SP-BE, Z-4
# ---------------------------------------------------------------------------------------------------------------------------------

def _tool_defects(row: dict) -> list[str]:
    d = (row or {}).get("diagnostics") or {}
    out = []
    spg = d.get("sp_gamma") or {}
    if spg.get("performed"):
        if spg.get("inclusion") is False:
            out.append("DEFECT_GAMMA_INCLUSION")
        elif spg.get("tight") is False:
            out.append("DEFECT_GAMMA_NARROWNESS")
    spb = d.get("second_path_BE") or {}
    if spb.get("match") is False:
        out.append("DEFECT_BE_SECOND_PATH")
    return out


def _channels(res: dict, f2: list[str]) -> list[str]:
    row = res.get("row") or {}
    d = row.get("diagnostics") or {}
    ch = []
    if f2:
        ch.append("binding_F2")
    spg = d.get("sp_gamma") or {}
    if spg.get("inclusion") is False:
        ch.append("sp_gamma_inclusion")
    if spg.get("inclusion") is not False and spg.get("tight") is False:
        ch.append("sp_gamma_narrowness")
    if (d.get("second_path_BE") or {}).get("match") is False:
        ch.append("second_path_BE")
    cl = res.get("classification") or {}
    if cl.get("rejected"):
        ch.append("boundary_definition")
    if cl.get("e1") is False or cl.get("e2") is False:
        ch.append("E1/E2")
    return ch


def _reassemble(res: dict, extra_pa: list[str], recorded: dict | None) -> dict:
    """Classify again with П-а not established by ``extra_pa`` (unchanged classifier through the pipeline's ``assemble``)."""
    p = pl()
    prem = dict(res["premises"])
    pa = prem.get("P-a") or {"state": p.NOT_CHECKABLE, "reason": "not provided"}
    reason = "; ".join(extra_pa)
    if pa.get("state") == p.ESTABLISHED:
        pa = {"state": p.NOT_ESTABLISHED, "reason": reason}
    else:
        pa = {"state": pa["state"], "reason": f"{pa.get('reason', '')}; {reason}"}
    prem["P-a"] = pa
    comps = {k: (float.fromhex(v) if isinstance(v, str) else v) for k, v in res["row"]["components"].items()}
    cl = p.assemble(comps, prem, recorded, faults=res["row"].get("faults"))
    return {**res, "premises": prem, "classification": cl, "status": cl["status"]}


def analyse_ready(spec: dict, reproduce: bool = True, recorded: dict | None = None, mandatory: bool = False,
                  force_z4: bool = False, sp_be: bool | None = None) -> dict:
    """One row through the integrated layer (module docstring 1-4). ``spec`` as for the pipeline (journal / paired / arrays);
    SP-GAMMA is run on every executed row; SP-BE when ``sp_be`` (default: in Z-4 only, plus the explicit request)."""
    p = pl()
    spec = {**spec, "sp_gamma": "all", "sp_be": bool(sp_be)}
    res = p.analyse(spec, reproduce=reproduce, recorded=recorded)
    out = {**res, "mandatory": mandatory, "f2_defects": [], "tool_defects": [], "channels": [], "z4": None}
    if res.get("row") is None:  # not executed (rank guard, stop, absent) or a pipeline exception: UNRESOLVED as classified
        return out
    cfg = p.spec_config(spec)
    f2 = p.actual_input_defects(res["row"], cfg)
    thr = blas_threads_read()
    out["blas_threads_read"] = thr
    od6 = [] if thr == 1 else [f"OD-6 (D-057.2 item 3): effective BLAS threads read from the library = {thr!r}, not 1 (the launching "
                               "process must set them before NumPy is imported); the bitwise binding of the row is not established"]
    if f2 or od6:
        out = {**out, **_reassemble(res, f2 + od6, recorded)}
    out["f2_defects"] = f2
    out["tool_defects"] = _tool_defects(res["row"])
    out["channels"] = _channels(out, f2)
    if force_z4 or z4_required(out):
        out["z4"] = z4_confirm(spec, out, reproduce=reproduce, recorded=recorded)
        if out["z4"]["outcome"] == "BINDING_MISMATCH":
            out = {**out, **_reassemble(out, [f"Z-4: {x}" for x in out["z4"]["binding_differences"]] or ["Z-4: binding mismatch"],
                                        recorded)}
    return out


def z4_required(res: dict) -> bool:
    """D-057.3 item 4: any mandatory row with UNRESOLVED_RESOLVABILITY and any row with FAIL_COMPUTATION, before its final status."""
    st = res.get("status")
    return (res.get("mandatory") and st == UNRES_RES) or st == FAIL


def _f2_view(res: dict) -> dict:
    row = res.get("row") or {}
    return {"inputs_sha": row.get("inputs_sha"), "actual_inputs": row.get("actual_inputs"),
            "P-a": (res.get("premises") or {}).get("P-a", {}).get("state")}


def z4_confirm(spec: dict, first: dict, reproduce: bool = True, recorded: dict | None = None) -> dict:
    """The confirmation procedure Z-4 (D-057.3 item 4; module docstring 4). Returns the outcome and every check."""
    p = pl()
    spec2 = {**spec, "sp_gamma": "all", "sp_be": True}
    repeat = p.analyse(spec2, reproduce=reproduce, recorded=recorded)
    checks = {}
    row2 = repeat.get("row") or {}
    d2 = row2.get("diagnostics") or {}
    spg, spb = d2.get("sp_gamma") or {}, d2.get("second_path_BE") or {}
    checks["gamma_inclusion"] = spg.get("inclusion")
    checks["gamma_narrowness"] = spg.get("tight")
    checks["BE_second_path_match"] = spb.get("match")
    checks["E_s_le_B_E2"] = spb.get("E_s_le_B_E2")
    established = all(v.get("state") == p.ESTABLISHED for v in (repeat.get("premises") or {}).values()) and bool(repeat.get("premises"))
    a, b = _f2_view(first), _f2_view(repeat)
    diffs = [] if a.get("inputs_sha") == b.get("inputs_sha") else ["inputs_sha"]
    ai_a, ai_b = a.get("actual_inputs") or {}, b.get("actual_inputs") or {}
    common = sorted(set(ai_a) & set(ai_b))  # the repeat adds the record of the second B_E path the first pass may not have run
    if not common or set(ai_a) - set(ai_b):
        diffs.append("actual_inputs: bound computations of the first pass missing in the repeat")
    diffs += [f"actual_inputs.{k}" for k in common if ai_a[k] != ai_b[k]]
    if b.get("P-a") != p.ESTABLISHED:
        diffs.append(f"repeated П-а {b.get('P-a')}: {(repeat.get('premises') or {}).get('P-a', {}).get('reason', '')[:300]}")
    if row2:
        f2b = p.actual_input_defects(row2, p.spec_config(spec2))
        diffs += [f"repeated F-2: {x}" for x in f2b]
    checks["binding_repeated_equal"] = not diffs
    if spg.get("performed") is not True or spg.get("inclusion") is False:
        outcome = "DEFECT_GAMMA_INCLUSION"
    elif spg.get("tight") is False:
        outcome = "DEFECT_GAMMA_NARROWNESS"
    elif spb.get("performed") is not True or spb.get("match") is not True:
        outcome = "DEFECT_BE_SECOND_PATH"
    elif established and spb.get("E_s_le_B_E2") is False:
        outcome = "CONTRADICTION_E2"
    elif diffs:
        outcome = "BINDING_MISMATCH"
    else:
        outcome = "CONFIRMED"
    return {"outcome": outcome, "checks": checks, "binding_differences": diffs, "repeat_status": repeat.get("status"),
            "rule": "D-057.3 item 4: status by the tree of protocol 4.2; a binding difference -> П-а not established; a defect -> "
                    "the candidate verdict is not issued until the cause is removed"}


def verdict(rows: list[dict], mandatory: dict) -> dict:
    """Candidate / campaign verdict after Z-4 (module docstring 5). ``rows``: dicts of PR #48 ``campaign_verdict`` (key, block, N, s,
    mutated, input) plus optional ``z4`` (outcome dict or None), ``tool_defects`` (list) and ``mandatory`` (bool). The classifier
    verdict is computed unchanged; it is NOT issued while a required Z-4 is missing or a defect of a tool was found."""
    f = pf()
    rows2, pending, defects = [], [], []
    for r in rows:
        r2 = dict(r)
        res = f.classify_row(r["input"])
        st = res["status"]
        z4 = r.get("z4")
        if (r.get("mandatory") and st == UNRES_RES) or st == FAIL:
            if not z4:
                pending.append(r["key"])
        if z4 and z4.get("outcome") == "BINDING_MISMATCH":
            inp = dict(r["input"])
            inp["premises"] = {**(inp.get("premises") or {}), "P-a": {"state": "NOT_ESTABLISHED", "reason": "Z-4 binding mismatch"}}
            r2["input"] = inp
        if z4 and z4.get("outcome") not in (None, "CONFIRMED", "BINDING_MISMATCH"):
            defects.append(f"{r['key']}: Z-4 {z4['outcome']}")
        defects += [f"{r['key']}: {d}" for d in r.get("tool_defects") or []]
        rows2.append({k: v for k, v in r2.items() if k not in ("z4", "tool_defects", "mandatory")})
    cv = f.campaign_verdict(rows2, mandatory)
    if pending or defects:
        return {"verdict": VERDICT_NOT_ISSUED, "classifier": cv, "z4_pending": sorted(pending), "defects": sorted(set(defects)),
                "rule": "D-057.3 item 4: the candidate verdict is not issued until Z-4 is done and the cause of a defect is removed"}
    return {"verdict": cv["campaign_verdict"], "classifier": cv, "z4_pending": [], "defects": []}


# ---------------------------------------------------------------------------------------------------------------------------------
# SP-OP: exact node-wise D2 against the exact tensor K^C (OD-11)
# ---------------------------------------------------------------------------------------------------------------------------------

def _exact_K_column(p, coeffs, j: int) -> list[Fraction]:
    from kkvim_reference import r5_cccc_halfstep as hs

    n = p.size
    e = np.zeros(n)
    e[j] = 1.0
    r0 = hs.residual_exact(p, np.zeros(n), coeffs)
    rj = hs.residual_exact(p, e, coeffs)
    return [a - b for a, b in zip(r0, rj)]


def sp_op_check(cfg, s: int, solved_set, columns=None, coeffs_override=None, mesh=None) -> dict:
    """SP-OP (module docstring 8). ``columns``: term-first indices q = k (n_s - 1) + b of the checked basis fields phi_q = U_k (x) e_b
    (default: b in {0, middle, last} for every k). ``coeffs_override``: replaces the exact tensor coefficients (negative control)."""
    from kkvim_reference import r5_cccc_halfstep as hs

    direction = op8().direction_of(s)
    U = np.asarray(solved_set, dtype=float)
    p = hs.HalfStep(float(cfg.lx), cfg.nx, float(cfg.ly), cfg.ny, direction, U, D=float(cfg.D), q=float(cfg.q))
    coeffs = coeffs_override if coeffs_override is not None else hs.exact_coefficients(p)
    N, m = p.N, p.n_s - 1
    if columns is None:
        columns = sorted({k * m + b for k in range(N) for b in (0, m // 2, m - 1)})
    be2 = pl().be2()
    M = mesh or be2.Mesh(float(cfg.lx), cfg.nx, float(cfg.ly), cfg.ny, float(cfg.D), float(cfg.q))
    Uf = [[Fraction(float(v)) for v in U[:, k]] for k in range(N)]
    h_s = Fraction(float(cfg.ly if direction == "Y" else cfg.lx)) / p.n_s
    nxm, nym = cfg.nx - 1, cfg.ny - 1
    bad, checked = [], 0
    for q in columns:
        k, b = divmod(q, m)
        w = [Fraction(0)] * M.n_int
        if direction == "Y":  # phi[ix, jy] = U[ix, k] delta(jy, b)
            for ix in range(nxm):
                w[ix * nym + b] = Uf[k][ix]
        else:  # phi[ix, jy] = delta(ix, b) U[jy, k]
            for jy in range(nym):
                w[b * nym + jy] = Uf[k][jy]
        t = M.apply(w)
        g = M.apply_T([M.W[n] * t[n] for n in range(M.n_nodes)])  # Delta^T W Delta phi_q; a_h(v, phi_q) = D v . g
        Kq = _exact_K_column(p, coeffs, q)
        for kk in range(N):
            for a in range(m):
                if direction == "Y":
                    ah = M.D * sum((Uf[kk][ix] * g[ix * nym + a] for ix in range(nxm)), Fraction(0))
                else:
                    ah = M.D * sum((Uf[kk][jy] * g[a * nym + jy] for jy in range(nym)), Fraction(0))
                checked += 1
                if ah != h_s * Kq[kk * m + a]:
                    bad.append((q, kk * m + a))
    return {"status": "HOLDS" if not bad else "VIOLATED", "columns": list(columns), "entries_checked": checked,
            "violations": len(bad), "first_violations": bad[:5],
            "identity": "a_h(phi_p, phi_q) = h_s K^C[p, q] exactly (BR 3), node-wise D2 operator of the second B_E path"}


# ---------------------------------------------------------------------------------------------------------------------------------
# resource probes (PL 6.3 unchanged; OD-7, OD-9)
# ---------------------------------------------------------------------------------------------------------------------------------

PROBE_LABEL = "PRELAUNCH_RESOURCE_PROBE (not D1_RESULT; no row value recorded)"
SURROGATE = ("s = 2 (half-step X): given = hs.start_set(n_y, N), companion = 2^-10 hs.start_set(n_x, N); s = 3 (half-step Y): given = "
             "hs.start_set(n_x, N), companion = 2^-10 hs.start_set(n_y, N); fixed before the run; not a D1 input (OD-9 (a))")
PROBES = {  # name -> (block, l_x, n_x, l_y, n_y, N, s)
    "A:320x1280:N1:s2": ("A", 1.0, 320, 4.0, 1280, 1, 2), "A:320x1280:N1:s3": ("A", 1.0, 320, 4.0, 1280, 1, 3),
    "A:400x1000:N1:s2": ("A", 1.0, 400, 2.5, 1000, 1, 2), "A:400x1000:N1:s3": ("A", 1.0, 400, 2.5, 1000, 1, 3),
    "A:400x600:N1:s2": ("A", 1.0, 400, 1.5, 600, 1, 2), "S:400x1600:N1:s2": ("S", 1.0, 400, 4.0, 1600, 1, 2),
    "S:1600x400:N1:s2": ("S", 1.0, 1600, 0.25, 400, 1, 2), "B:160x640:N3:s1": ("B", 1.0, 160, 4.0, 640, 3, 1),
    "B:200x500:N3:s1": ("B", 1.0, 200, 2.5, 500, 3, 1), "C:80x320:N3:s1": ("C", 1.0, 80, 4.0, 320, 3, 1),
    "C:100x250:N3:s1": ("C", 1.0, 100, 2.5, 250, 3, 1), "C:100x250:N1:s2": ("C", 1.0, 100, 2.5, 250, 1, 2),
    "D:9x11:N1:s1": ("D", 1.0, 9, 1.0, 11, 1, 1), "D:11x15:N1:s1": ("D", 1.0, 11, 1.0, 15, 1, 1),
    "D:14x6:N1:s1": ("D", 1.0, 14, 1.0, 6, 1, 1), "D:14x6:N3:s1": ("D", 1.0, 14, 1.0, 6, 3, 1),
}
PROBE_SP_BE = ("A:320x1280:N1:s2", "A:400x1000:N1:s2")
PROBE_SP_OP = ("A:320x1280:N1:s2",)
SMOKE = {"SMOKE:8x10:N1:s2": ("A", 1.0, 8, 1.25, 10, 1, 2), "SMOKE:8x10:N3:s1": ("C", 1.0, 8, 1.25, 10, 3, 1)}
PL63_RAM_FRACTION = Fraction(1, 2)
PL63_FACTORIZATION_S = 30 * 60
SAFETY_TIMEOUT_S = 6 * 3600  # operational safety of the probe only (not a PL 6.3 limit)
PROBE_ENV = {k: "1" for k in THREAD_VARS}


def surrogate_input(nx: int, ny: int, N: int, s: int):
    from kkvim_reference import r5_cccc_halfstep as hs

    if s == 1:
        return hs.start_set(nx, N), np.zeros((ny - 1, N))
    if s % 2 == 0:
        return hs.start_set(ny, N), 2.0**-10 * hs.start_set(nx, N)
    return hs.start_set(nx, N), 2.0**-10 * hs.start_set(ny, N)


def _timed(stages: dict, name: str, fn, factorization: bool):
    t = time.perf_counter()
    try:
        r = fn()
        stages[name] = {"seconds": round(time.perf_counter() - t, 2), "completed": True, "factorization": factorization}
        return r
    except Exception as exc:  # recorded, never hidden
        stages[name] = {"seconds": round(time.perf_counter() - t, 2), "completed": False, "factorization": factorization,
                        "exception": f"{type(exc).__name__}: {str(exc)[-300:]}"}
        return None


def _probe_child(name: str, limit: int, flag: str) -> dict:
    """One probe (module docstring 9) in this process; every value of the row is discarded."""
    from kkvim_reference import r5_cccc_halfstep as hs
    from kkvim_reference import r5_cccc_sequence as sq

    block, lx, nx, ly, ny, N, s = {**PROBES, **SMOKE}[name]
    stop = threading.Event()
    threading.Thread(target=pf()._watchdog, args=(limit, Path(flag), stop), daemon=True).start()
    t0 = time.perf_counter()
    p, o, b = pl(), op8(), bcd()
    cfg = o.TrajectoryConfig(lx, nx, ly, ny, N, D_VALUE, Q_VALUE, max(s, 1))
    if N >= 2 and s >= 2:
        raise p.RankGuardRefused("probe of N >= 2 at s >= 2 refused (D-057.1)")
    given, comp = surrogate_input(nx, ny, N, s)
    st = {}
    out = {"probe": name, "label": PROBE_LABEL, "block": block, "N": N, "s": s, "nx": nx, "ny": ny, "lx": repr(lx), "ly": repr(ly),
           "cells": nx * ny, "input": "start set T_N, zero companion (s = 1)" if s == 1 else "surrogate (OD-9 (a))",
           "blas_threads_read_from_library": blas_threads_read(), "stages": st}
    res = _timed(st, "op8_halfstep", lambda: o.run_step(cfg, s, given, comp), True)
    if res is None or res.record.get("status") != o.OK:
        out["op8_status_class"] = None if res is None else str(res.record.get("status")).split(":")[0]
    else:
        out["op8_status_class"] = "OK"
        a = res.arrays
        _timed(st, "row_G4_gamma_B1_premises_B_E", lambda: p._compute_row_core(cfg, s, a["solved_set"], a["companion_solved"],
                                                                              a["V_bc"]), False)
        Xp, Yp = p.eq().orient(o.direction_of(s), a["solved_set"], a["companion_solved"])
        Xs, Ys = p.eq().orient(o.direction_of(s), a["solved_set"], a["V_bc"])
        _timed(st, "SP_GAMMA_energies_path_b", lambda: p.ga().energies_path_b(lx, nx, ly, ny, D_VALUE, Q_VALUE, Xp, Yp, Xs, Ys), False)
        if name in PROBE_SP_BE:
            def be():
                wp, ws = Xp @ Yp.T, Xs @ Ys.T
                from kkvim_reference import r5_cccc_energy as en
                g = en.Grid(lx, nx, ly, ny, D=D_VALUE, q=Q_VALUE)
                dh = wp - ws
                return p.be2().second_path({"lx": lx, "nx": nx, "ly": ly, "ny": ny, "D": D_VALUE, "q": Q_VALUE}, Xp, Yp, Xs, Ys,
                                           wp, ws, dh, P_hat={"P_p": en.pi_double(g, wp), "P_s": en.pi_double(g, ws), "Q": en.q_double(g, dh)})
            r = _timed(st, "SP_BE_second_path", be, False)
            st["SP_BE_second_path"]["status_class"] = None if r is None else str(r.get("status"))
        if name in PROBE_SP_OP:
            r = _timed(st, "SP_OP", lambda: sp_op_check(cfg, s, a["solved_set"]), False)
            st["SP_OP"]["status_class"] = None if r is None else r["status"]
        br = _load(BR_TOOL, "_br_ready")
        ph = hs.HalfStep(lx, nx, ly, ny, o.direction_of(s), np.asarray(given, dtype=float), D=D_VALUE, q=Q_VALUE)
        rb = _timed(st, "BR_reference", lambda: br.analyse_point(ph, N >= 2, None, (nx - 1) * (ny - 1) <= br.RH_LIMIT), True)
        if rb is not None:
            st["BR_reference"].update({"certificate_lambda_low": rb.get("lambda_low") is not None,
                                       "refine_converged": rb.get("refine_converged"), "precision_used": rb.get("precision"),
                                       "kind": "reference certificate (observation; OD-7: apart from the resources of the row)"})
        if block == "C":
            if N == 1:
                _timed(st, "RAC_halfstep", lambda: hs.solve_ac(ph), True)
            else:
                _timed(st, "RAC_halfstep_s1_no_QR", lambda: sq.run_sequence(b._seq_config(cfg, 1, qr=False), energy=False), True)
            eqr = _timed(st, "EQ_paired_halfstep", lambda: b.eq_paired(cfg, s, given, comp) if (nx, ny) not in o.D1_GRIDS else
                         _eq_paired_unguarded(b, cfg, s, given, comp), True)
            if eqr is not None:
                out["EQ_equals_OP8_bitwise"] = b.compare_eq_op8(eqr, res)["bitwise"]
        if N >= 2:
            out["start_rank_diagnostic"] = {"exact_rank_of_start": int(hs.exact_rank(np.asarray(given, dtype=float))),
                                            "note": "s = 1: QR of the start set only (VBC-13 item 4: independent start)"}
    stop.set()
    out["peak_working_set_bytes"] = pf()._mem()["peak"]
    out["total_seconds"] = round(time.perf_counter() - t0, 2)
    return out


def _eq_paired_unguarded(b, cfg, s, given, comp):
    """EQ paired half-step for a probe on a D1-size grid (values discarded): the same functions as ``eq_paired`` without the D1-grid
    guard; the rank guard is kept."""
    from kkvim_reference import r5_cccc_bc as bc
    from kkvim_reference import r5_cccc_halfstep as hs

    if cfg.N >= 2 and s >= 2:
        raise pl().RankGuardRefused("rank guard")
    p_in = hs.HalfStep(float(cfg.lx), cfg.nx, float(cfg.ly), cfg.ny, op8().direction_of(s), np.asarray(given, dtype=float),
                       D=float(cfg.D), q=float(cfg.q))
    rec = bc.run_halfstep(p_in, qr=cfg.N >= 2)
    solved = rec.solved
    Cp = np.asarray(comp, dtype=float) @ rec.R.T if rec.R is not None else np.asarray(comp, dtype=float)
    r = pl().eq().solve_c3plus(solved, hs.exact_coefficients(solved))
    sol = r.pop("_sol")
    return {"status": "OK" if sol.s is not None else "STOP", "solved_set": np.array(solved.given), "companion_solved": Cp,
            "V": None if sol.s is None else sol.s.reshape(cfg.N, solved.n_s - 1).T}


def evaluate_probe(res: dict, ram: int | None) -> dict:
    """PL 6.3 unchanged (OD-7): peak working set <= 50 % of physical RAM; each stage containing a factorization <= 30 min. The BR
    certificate is a property of the reference observation, recorded apart (never a resource failure of the row)."""
    lim = None if ram is None else ram * PL63_RAM_FRACTION
    pk = res.get("peak_working_set_bytes")
    stages = res.get("stages", {})
    fact = [v["seconds"] for v in stages.values() if v.get("factorization")]
    ev = {"peak_le_50pct_ram": None if (lim is None or pk is None) else pk <= lim,
          "factorization_stages_le_30min": all(x <= PL63_FACTORIZATION_S for x in fact) if fact else None,
          "all_stages_completed": all(v.get("completed") for v in stages.values()) and bool(stages),
          "B_E_and_premises_included": bool(stages.get("row_G4_gamma_B1_premises_B_E", {}).get("completed"))}
    br = stages.get("BR_reference", {})
    ev["reference_certificate"] = ("NOT_RUN" if not br else "CERTIFIED" if br.get("certificate_lambda_low") and br.get("refine_converged")
                                   else "UNRESOLVED (budget of the reference; protocol 5 item 8: not a resource status of the row)")
    ev["all_within"] = all(ev[k] is True for k in ("peak_le_50pct_ram", "factorization_stages_le_30min", "all_stages_completed",
                                                    "B_E_and_premises_included"))
    return ev


def run_probes(names: list[str], out_dir: Path, workers: int = 1) -> dict:
    """Each probe in its own child process (launcher environment PROBE_ENV, OD-6); up to ``workers`` at a time (each single
    threaded). Returns the resource records only."""
    ram = pf().total_ram_bytes()
    limit = int(ram * PL63_RAM_FRACTION) if ram else 0
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    blob = git_blob_id(ROOT / SELF)
    procs, results = {}, {}
    pending = list(names)
    for n in names:
        if n not in PROBES and n not in SMOKE:
            raise ValueError(f"unknown probe {n}")

    def start(n):
        flag = out_dir / f"abort_{n.replace(':', '_')}.json"
        flag.unlink(missing_ok=True)
        f = open(out_dir / f"probe_{n.replace(':', '_')}.out", "w", encoding="utf-8")
        pr = subprocess.Popen([sys.executable, str(ROOT / SELF), "_probe_child", n, str(limit), str(flag)], cwd=ROOT, stdout=f,
                              stderr=subprocess.STDOUT, text=True, env={**os.environ, **PROBE_ENV})
        procs[n] = (pr, f, flag, time.perf_counter())

    while pending or procs:
        while pending and len(procs) < max(1, workers):
            start(pending.pop(0))
        for n, (pr, f, flag, t0) in list(procs.items()):
            if pr.poll() is None and time.perf_counter() - t0 < SAFETY_TIMEOUT_S:
                continue
            if pr.poll() is None:
                pr.kill()
            pr.wait()
            f.close()
            del procs[n]
            text = (out_dir / f"probe_{n.replace(':', '_')}.out").read_text(encoding="utf-8")
            if flag.exists():
                results[n] = {"probe": n, "result": "UNRESOLVED_RESOURCES", **json.loads(flag.read_text(encoding="utf-8"))}
            elif pr.returncode != 0:
                results[n] = {"probe": n, "result": "UNRESOLVED", "reason": f"child failed ({pr.returncode})", "tail": text[-1500:]}
            else:
                res = json.loads(text.strip().splitlines()[-1])
                res["pl63"] = evaluate_probe(res, ram)
                res["result"] = "WITHIN_PL63" if res["pl63"]["all_within"] else "UNRESOLVED_RESOURCES"
                results[n] = res
        time.sleep(0.5)
    return {"document": "R5-V Q4 QA-D1-READY resource probes (protocol 8, 12.5; PL 6.3; D-057.4)", "label": PROBE_LABEL,
            "data_status": DATA_STATUS, "total_ram_bytes": ram, "limit_bytes_50pct": limit, "platform": platform_record(),
            "freeze": freeze_allowed(), "probe_tool_blob": blob, "surrogate_definition": SURROGATE, "material_load": "D = q = 1 (OD-5)",
            "launcher_environment": PROBE_ENV, "results": [results[n] for n in names]}


def provenance() -> dict:
    tools = (SELF, PIPELINE_TOOL, PREFLIGHT_TOOL, BLOCK_C_TOOL, "scripts/r5_cccc_q4_d1_c3plus_trajectories.py",
             "scripts/r5_cccc_q4_be_second_path_research.py", "scripts/r5_cccc_q4_energy_bound_premises_audit.py",
             "scripts/r5_cccc_q4_energy_bc_audit.py", "scripts/r5_cccc_q4_gamma_interval_audit.py", "scripts/r5_cccc_q4_gamma_audit.py",
             "scripts/r5_cccc_q4_c_assembly_e4_audit.py", BR_TOOL)
    return {"tool_blobs": {t: git_blob_id(ROOT / t) for t in tools}, "platform": platform_record(), "freeze": freeze_allowed()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("_probe_child")
    c.add_argument("name")
    c.add_argument("limit", type=int)
    c.add_argument("flag")
    r = sub.add_parser("probes")
    r.add_argument("names", nargs="*")
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--workers", type=int, default=1)
    sub.add_parser("campaign")
    a = ap.parse_args(argv)
    if a.cmd == "_probe_child":
        print(json.dumps(_probe_child(a.name, a.limit, a.flag), default=str))
        return 0
    if a.cmd == "campaign":
        print("refused: D1_START = BLOCKED (protocol 12)", file=sys.stderr)
        return 2
    pl().guard_out(a.out)
    doc = run_probes(a.names or list(PROBES), a.out, a.workers)
    with open(Path(a.out) / "resource_probes.json", "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1, ensure_ascii=False, default=str)
    print(json.dumps({r["probe"]: r["result"] for r in doc["results"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
