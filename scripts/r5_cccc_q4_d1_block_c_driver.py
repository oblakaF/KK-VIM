"""R5-V Q4 QA-D1-READY (D-057.1, D-057.3 OD-3): driver of block C of the D1 protocol (docs/R5_CCCC_Q4_D1_PROTOCOL.md 3, 3.1) - the
R-AC reference chain and the paired B-C (C-3+) half-steps on its inputs. QA research tool: NOT the D1 campaign (D1_START = BLOCKED; the
D1 grids are refused before any computation), not the working CCCC solver (no import of ``kkvim``), no threshold, no new formula.

Paths (fixed by D-057.3, OD-3; used read-only, none of them is changed):
- R-AC chain: ``r5_cccc_sequence.run_sequence`` (SEQ, D-051.1) of src/kkvim_reference, orientation first = Y, D = q = 1 (OD-5). The
  input state (given, companion) of half-step s is the state of the chain after s - 1 half-steps: ``run_sequence`` with s_max = s - 1
  and the final factors (X, Y) of the result (the code of SEQ itself; no transformation is re-implemented here); s = 1: start set T_N
  of SEQ and a zero companion;
- paired B-C half-step: the functions of EQ (scripts/r5_cccc_q4_energy_bc_audit.py, PR #41): ``bc.run_halfstep`` of the prototype
  (QR for N >= 2 at s = 1: the QR of the start set), companion C' = C R^T, ``solve_c3plus`` (C replaced by C-3+). ``solve_c3plus``
  replaces ``bc.bc_coefficients`` globally while it runs: the driver is single-threaded (one process, BLAS threads 1 set before NumPy
  is imported, the effective number read from the library and recorded);
- before use (OD-3): the paired arrays of EQ are compared BITWISE with the isolated OP-8 call (``run_step`` of
  scripts/r5_cccc_q4_d1_c3plus_trajectories.py, no global replacement) on the same input, at every executed s (small grids: s = 1..8).

Rank protection (D-057.1, items 1-6): for N = 2, 3 the R-AC chain runs only its first half-step, WITHOUT the QR transformation of the
solved set (``SequenceConfig(qr=False)``; its rank diagnostic, which factorises the solved set for rho_N, is an OBSERVATION and changes
no set); half-steps s >= 2 of the chain are not run: ``STOPPED_RANK_GUARD_UNCERTIFIED``. Before the first half-step the configurational
S-1 of RN-3 and the start condition N <= floor(n_x/2) are checked on the configuration (OP-8 ``config_s1_conditions``); a proven
dependence stops the chain with S-1 before any solve. Paired rows of N >= 2 at s >= 2 are not executed: UNRESOLVED with the reason.

Data: PRELAUNCH research observation, NOT D1 data, never D1_RESULT. Output only outside docs/.
"""

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

THREAD_VARS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
for _k in THREAD_VARS:
    os.environ[_k] = "1"  # single-threaded BLAS (effective only before NumPy is loaded; the effective count is read and recorded)

import numpy as np  # noqa: E402

from kkvim_reference import r5_cccc_bc as bc  # noqa: E402
from kkvim_reference import r5_cccc_halfstep as hs  # noqa: E402
from kkvim_reference import r5_cccc_sequence as sq  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PIPELINE_TOOL = "scripts/r5_cccc_q4_d1_row_pipeline_prelaunch.py"
DATA_STATUS = "PRELAUNCH research observation (block C driver); NOT D1 data, never D1_RESULT (D1_START = BLOCKED)"
PAIRED_ARRAYS = ("given_in", "companion_in", "solved_set", "companion_solved", "V")
OK = "OK"


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


def op8():
    return pl().op8()


def eq():
    return pl().eq()


def threads_read() -> int | None:
    """The effective number of BLAS threads read from the library (QA-EB ``blas_info``), not the environment variables."""
    return pl().qa().blas_info().get("threads")


def _seq_config(cfg, s_max: int, qr: bool):
    return sq.SequenceConfig(float(cfg.lx), cfg.nx, float(cfg.ly), cfg.ny, cfg.N, D=float(cfg.D), q=float(cfg.q), s_max=s_max,
                             first="Y", qr=qr)


def rac_state(cfg, s: int):
    """(given, companion) entering half-step s of the R-AC chain, as SEQ forms them (module docstring). N >= 2: s = 1 only."""
    pl().guard_row(cfg, s)
    if s == 1:
        return hs.start_set(cfg.nx, cfg.N), np.zeros((cfg.ny - 1, cfg.N))
    res = sq.run_sequence(_seq_config(cfg, s - 1, qr=True), energy=False)
    if len(res.steps) != s - 1 or res.X is None or any(st.get("status") != "OK" for st in res.steps):
        raise RuntimeError(f"R-AC chain stopped before s = {s}: {res.status}")
    direction = op8().direction_of(s)
    return (res.X, res.Y) if direction == "Y" else (res.Y, res.X)


def _sha(a) -> str:
    return op8().array_sha(a)


def rac_chain(cfg) -> dict:
    """The R-AC reference chain of one block-C configuration with the rank protection (module docstring). Records per s: status,
    certificate status of SEQ, the input state hashes; N >= 2: only s = 1 (no QR transformation), s >= 2 STOPPED_RANK_GUARD."""
    pl().guard_config(cfg)
    o = op8()
    recs, states = [], {}
    start = hs.start_set(cfg.nx, cfg.N)
    rank = hs.exact_rank(start)
    if cfg.N > cfg.nx // 2:
        recs.append({"s": 1, "status": o.STOP_START})
        return {"config": _cfg_dict(cfg), "records": recs, "states": states, "status": f"STOPPED at s = 1: {o.STOP_START}"}
    cond = o.config_s1_conditions(cfg, cfg.N, cfg.ny, rank)
    if cond["all_hold"]:
        recs.append({"s": 1, "status": o.STOP_CONFIG, "config_S1_RN3": cond})
        return {"config": _cfg_dict(cfg), "records": recs, "states": states, "status": f"STOPPED at s = 1: {o.STOP_CONFIG}"}
    s_exec = cfg.s_max if cfg.N == 1 else 1
    full = sq.run_sequence(_seq_config(cfg, s_exec, qr=cfg.N == 1), energy=False)
    for st in full.steps:
        s = st["s"]
        g, c = rac_state(cfg, s)
        states[s] = (g, c)
        recs.append({"s": s, "direction": st["direction"], "status": st.get("status"), "certificate": st.get("certificate"),
                     "qr_transform": False, "run_class": st.get("run_class"), "config_S1_RN3": cond if s == 1 else None,
                     "given_in_sha": _sha(g), "companion_in_sha": _sha(c)})
    if cfg.N >= 2:
        for s in range(2, cfg.s_max + 1):
            recs.append({"s": s, "status": o.STOP_RANK_GUARD, "qr_transform": False,
                         "reason": "R-AC chain: no QR of the solved set of s - 1 (D-057.1 items 2, 6)"})
        status = f"{o.RANK_GUARD} at s = 2: {o.STOP_RANK_GUARD}" if cfg.s_max >= 2 else full.status
    else:
        status = full.status
    return {"config": _cfg_dict(cfg), "records": recs, "states": states, "status": status, "sequence_status": full.status}


def _cfg_dict(cfg) -> dict:
    return {k: getattr(cfg, k) for k in ("lx", "nx", "ly", "ny", "N", "D", "q", "s_max")}


def eq_paired(cfg, s: int, given_in, companion_in) -> dict:
    """The paired B-C half-step by the EQ functions (module docstring): arrays and the EQ flags."""
    pl().guard_row(cfg, s)
    direction = op8().direction_of(s)
    p_in = hs.HalfStep(float(cfg.lx), cfg.nx, float(cfg.ly), cfg.ny, direction, np.array(given_in, dtype=float), D=float(cfg.D),
                       q=float(cfg.q))
    rec = bc.run_halfstep(p_in, qr=cfg.N >= 2)
    if rec.solved is None:
        return {"status": f"STOP: prototype set status {rec.log.get('set_status')}"}
    solved = rec.solved
    Cp = np.asarray(companion_in, dtype=float) @ rec.R.T if rec.R is not None else np.array(companion_in, dtype=float)
    coeffs = hs.exact_coefficients(solved)
    res = eq().solve_c3plus(solved, coeffs)
    sol = res.pop("_sol")
    if sol.s is None:
        return {"status": f"STOP: B-C {sol.status}", **{k: res.get(k) for k in ("c3plus_premises", "c3plus_theorem")}}
    V = sol.s.reshape(cfg.N, solved.n_s - 1).T
    return {"status": OK, "solved_set": np.array(solved.given), "companion_solved": Cp, "V": V,
            "only_C_replaced": res.get("only_C_replaced"), "c3plus_premises": res.get("c3plus_premises"),
            "c3plus_theorem": res.get("c3plus_theorem")}


def op8_twin(cfg, s: int, given_in, companion_in):
    """The isolated OP-8 call on the same input (no global replacement)."""
    pl().guard_row(cfg, s)
    return op8().run_step(cfg, s, given_in, companion_in)


def compare_eq_op8(eqr: dict, twin) -> dict:
    eqv = {"solved_set": eqr.get("solved_set"), "companion_solved": eqr.get("companion_solved"), "V": eqr.get("V")}
    tw = {"solved_set": twin.arrays.get("solved_set"), "companion_solved": twin.arrays.get("companion_solved"), "V": twin.arrays.get("V_bc")}
    eqa = {k: bool(eqv[k] is not None and tw[k] is not None and eqv[k].shape == tw[k].shape and eqv[k].tobytes() == tw[k].tobytes())
           for k in eqv}
    return {"arrays_equal": eqa, "bitwise": bool(eqr.get("status") == OK and twin.record.get("status") == OK and all(eqa.values())),
            "op8_status": twin.record.get("status"), "eq_status": eqr.get("status")}


def write_paired(cfg, out: Path) -> dict:
    """Chain + paired half-steps of one configuration, written under ``out`` (NPZ per s with sha256, chain.json, paired.json)."""
    pl().guard_config(cfg)
    pl().guard_out(out)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    chain = rac_chain(cfg)
    chain_doc = {"data_status": DATA_STATUS, "config": chain["config"], "status": chain["status"], "records": chain["records"],
                 "blas_threads_read": threads_read()}
    with open(out / "chain.json", "w", encoding="utf-8") as f:
        json.dump(chain_doc, f, indent=1, default=str)
    rows = []
    for s in range(1, cfg.s_max + 1):
        if s not in chain["states"]:
            reason = (pl().rank_guard_reason(cfg, s) if cfg.N >= 2 and s >= 2 else
                      f"not executed: the R-AC chain has no state for s = {s} ({chain['status']})")
            rows.append({"s": s, "executed": False, "reason": reason})
            continue
        g, c = chain["states"][s]
        eqr = eq_paired(cfg, s, g, c)
        twin = op8_twin(cfg, s, g, c)
        cmpd = compare_eq_op8(eqr, twin)
        if eqr.get("status") != OK:
            rows.append({"s": s, "executed": False, "reason": f"paired half-step not solved: {eqr.get('status')}", "crosscheck": cmpd})
            continue
        arrays = {"given_in": np.asarray(g, dtype=float), "companion_in": np.asarray(c, dtype=float), "solved_set": eqr["solved_set"],
                  "companion_solved": eqr["companion_solved"], "V": eqr["V"]}
        npz = out / f"paired_s{s}.npz"
        np.savez(npz, **arrays)
        spec = {"kind": "paired", "paired_npz": str(npz), "paired_sha256": {k: _sha(v) for k, v in arrays.items()},
                "chain_json": str(out / "chain.json"), "config": _cfg_dict(cfg), "s": s}
        rows.append({"s": s, "executed": True, "spec": spec, "crosscheck": cmpd, "eq_only_C_replaced": eqr.get("only_C_replaced"),
                     "eq_c3plus_premises": eqr.get("c3plus_premises"), "eq_c3plus_theorem": eqr.get("c3plus_theorem")})
    doc = {"data_status": DATA_STATUS, "config": _cfg_dict(cfg), "chain_status": chain["status"], "rows": rows,
           "blas_threads_read": threads_read()}
    with open(out / "paired.json", "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1, default=str)
    return doc


def paired_chain_binding(spec: dict) -> list[str]:
    """П-а of a paired row against its chain: the state hashes of the chain record of s are the hashes of the paired input arrays."""
    bad = []
    try:
        with open(spec["chain_json"], encoding="utf-8") as f:
            chain = json.load(f)
        rec = next((r for r in chain["records"] if r.get("s") == spec["s"]), None)
        if rec is None or rec.get("status") != "OK":
            return [f"chain: half-step s = {spec['s']} is not an executed half-step of the R-AC chain"]
        if chain.get("config") != spec.get("config"):
            bad.append("chain: configuration differs from the row specification")
        sh = spec.get("paired_sha256") or {}
        if rec.get("given_in_sha") != sh.get("given_in") or rec.get("companion_in_sha") != sh.get("companion_in"):
            bad.append("chain: the paired input is not the recorded R-AC state of s")
    except Exception as exc:  # recorded, never hidden
        bad.append(f"chain binding not checkable: {type(exc).__name__}: {str(exc)[-200:]}")
    return bad


def paired_inputs(spec: dict, cfg, s: int):
    """(solved set, companion, V, binding reasons, rerun) of a paired row (pipeline ``_inputs_from_spec``): the stored arrays against
    the sha256 of the specification, the EQ paired half-step and its OP-8 twin re-run on the stored input, bitwise."""
    pl().guard_row(cfg, s)
    reasons = []
    with np.load(spec["paired_npz"]) as z:
        arr = {k: z[k] for k in PAIRED_ARRAYS}
    for k, a in arr.items():
        if (spec.get("paired_sha256") or {}).get(k) != _sha(a):
            reasons.append(f"paired input {k}: sha256 differs from the specification")
    eqr = eq_paired(cfg, s, arr["given_in"], arr["companion_in"])
    twin = op8_twin(cfg, s, arr["given_in"], arr["companion_in"])
    cmpd = compare_eq_op8(eqr, twin)
    stored = {"solved_set": arr["solved_set"], "companion_solved": arr["companion_solved"], "V": arr["V"]}
    eq_equal = {k: bool(eqr.get("status") == OK and eqr[k].shape == v.shape and eqr[k].tobytes() == v.tobytes()) for k, v in stored.items()}
    rec = twin.record
    rerun = {"status": eqr.get("status"), "op8_status": rec.get("status"), "arrays_equal": {**eq_equal,
             **{f"op8_{k}": v for k, v in cmpd["arrays_equal"].items()}}, "eq_equals_op8_bitwise": cmpd["bitwise"],
             "reproduced": bool(all(eq_equal.values()) and cmpd["bitwise"]), "op8_record": rec,
             "c3plus_premises_status": rec.get("c3plus_premises_status"),
             "c3plus_premise_classes": (rec.get("c3plus_premises") or {}).get("classes"),
             "c3plus_record": pl()._c3plus_of_record(rec), "V_bc_sha": rec.get("V_bc_sha"), "blas_threads_read": threads_read()}
    return arr["solved_set"], arr["companion_solved"], arr["V"], reasons, rerun


def crosscheck(cfg) -> dict:
    """OD-3 bitwise comparison of the EQ path with the isolated OP-8 call on the chain inputs, every executed s (no file written)."""
    pl().guard_config(cfg)
    chain = rac_chain(cfg)
    out = []
    for s in sorted(chain["states"]):
        g, c = chain["states"][s]
        cmpd = compare_eq_op8(eq_paired(cfg, s, g, c), op8_twin(cfg, s, g, c))
        out.append({"s": s, **cmpd, "input_is_RAC_state": True})
    return {"config": _cfg_dict(cfg), "chain_status": chain["status"], "rows": out, "blas_threads_read": threads_read(),
            "all_bitwise": bool(out) and all(r["bitwise"] for r in out)}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
