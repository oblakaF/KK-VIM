"""Tests of the PRELAUNCH N = 1 row pipeline scripts/r5_cccc_q4_d1_row_pipeline_prelaunch.py (protocol 4.1-4.2, 5, 12; task: Z-3).

Small non-D1 configurations only (T1 of the frozen control expectations: 8 x 10 on 1 x 1.25, N = 1). Checked: the full path from an
OP-8 journal to the classification; exact dP and tau' at the boundary with the Z-3 counterexamples; no false PASS with an incomplete
certificate; E1/E2, E_DIAG and the strict sign of resolvability; real detectors; binding, hash chain and reproduction in a separate
process; the D0 regression against the published EQ / QA-EB rows (portable checks on every platform, the bitwise
comparison where the computational path is shown identical to the recorded one); the guard against the D1 campaign.
"""

import ast
import contextlib
import copy
import csv
import ctypes
import importlib.util
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
import scipy

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "scripts" / "r5_cccc_q4_d1_row_pipeline_prelaunch.py"


def _load():
    spec = importlib.util.spec_from_file_location("_pl_pipeline", TOOL)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


pl = sys.modules.get("_pl_pipeline") or _load()
OK = {p: {"state": "ESTABLISHED", "reason": "synthetic"} for p in ("P-a", "P-b", "P-c", "P-d", "P-e", "P-f")}


@pytest.fixture(scope="module")
def journal(tmp_path_factory):
    cfg = pl.op8().TrajectoryConfig(1.0, 8, 1.25, 10, 1, 1.0, 1.0, 3)
    return pl.run_trajectory_journal(cfg, tmp_path_factory.mktemp("pl_T1"))


@pytest.fixture(scope="module")
def rows(journal):
    return pl.analyse_journal(journal)


def _states(r):
    return {k: v["state"] for k, v in r["premises"].items()}


# --- 1. full path ----------------------------------------------------------------------------------------------------------------

def test_full_path_from_the_op8_journal_to_the_classification(journal, rows):
    j, arrays = pl.load_journal(journal)
    assert [r["status"] for r in rows] == ["PASS"] * 3
    for s, r in enumerate(rows, start=1):
        rec = j["steps"][s - 1]
        assert set(_states(r).values()) == {"ESTABLISHED"}
        assert r["reproduction"]["bitwise_equal"] is True and r["op8_rerun"]["reproduced"] is True
        assert r["row"]["field_s_sha"] == rec["field_s_sha"] and r["row"]["field_prev_sha"] == rec["prev_field_after_QR_sha"]
        c = r["classification"]
        # dP and tau' are the exact rationals of the recorded binary64 components (protocol 4.1)
        assert Fraction(c["dP_exact"]) == Fraction(float.fromhex(r["row"]["P_hat_p"])) - Fraction(float.fromhex(r["row"]["P_hat_s"]))
        assert Fraction(c["tau_exact"]) == Fraction(float.fromhex(r["row"]["Gamma_up"])) + Fraction(float.fromhex(r["row"]["B_E_up"]))
        assert c["e1"] is True and c["e2"] is True and r["contradictions"] == []
        assert r["row"]["diagnostics"]["second_path_gamma"]["consistent"] is True
        assert r["data_status"].startswith("PRELAUNCH") and "never D1_RESULT" in r["data_status"]


# --- 2. Z-3 at the boundary ------------------------------------------------------------------------------------------------------

def test_z3_exact_dP_and_tau_and_rejection_of_recorded_values():
    big = "0x1.0000000000001p+53"  # 2^53 + 2 = fl((2^53 + 2) - 0.5)
    comp = {"P_p": big, "P_s": 0.5, "Q": big, "Gamma_up": 0.125, "B_E": 0.125}
    r = pl.assemble(comp, OK)
    assert r["dP_exact"] == f"{2**54 + 3}/2" and r["status"] == "FAIL_COMPUTATION"  # exact |dP - Q^| = 1/2 > 1/4
    q = 0.1 + 0.2
    r = pl.assemble({"P_p": q, "P_s": 0.0, "Q": q, "Gamma_up": 0.1, "B_E": 0.2}, OK)
    assert Fraction(r["tau_exact"]) == Fraction(0.1) + Fraction(0.2) < Fraction(q) and r["status"] == "PASS"
    # counterexamples of Z-3: a recorded value that is not the definition is never accepted, never PASS
    for rec in ({"dP": "hex:" + big}, {"dP": str(Fraction(2**54 + 3, 2) + Fraction(1, 2**1080))}):
        x = pl.assemble(comp, OK, rec)
        assert x["rejected"] and x["status"] == "UNRESOLVED_CERTIFICATE" and x["source_undetermined"] and x["dP_exact"] is None
    x = pl.assemble({"P_p": q, "P_s": 0.0, "Q": q, "Gamma_up": 0.1, "B_E": 0.2}, OK, {"tau": q})
    assert x["rejected"] and x["status"] == "UNRESOLVED_CERTIFICATE" and x["tau_exact"] is None
    x = pl.assemble({"P_p": 3.0, "P_s": 2.0, "Q": 0.25, "Gamma_up": 0.25, "B_E": 0.25}, OK, {"tau": "1"})  # would hide FAIL
    assert x["rejected"] and x["status"] not in ("PASS", "FAIL_COMPUTATION")
    x = pl.assemble({"P_p": 3.0, "P_s": 2.0, "Q": 1.0, "Gamma_up": 0.25}, OK, {"tau": "1/2"})  # recorded tau' with B_E absent
    assert x["rejected"] and x["status"] == "UNRESOLVED_CERTIFICATE"
    x = pl.assemble({"P_p": 3.0, "P_s": 2.0, "Q": 1.0, "Gamma_up": 0.25, "B_E": 0.25}, OK, {"dP": 1.0, "tau": "1/2"})
    assert not x["rejected"] and x["status"] == "PASS"
    x = pl.assemble({"P_p": 1.0, "P_s": 0.0, "Q": 1.0, "Gamma_up": 0.0, "B_E": 0.0}, OK, {"tau": float("inf")})
    assert x["rejected"] and x["status"] == "UNRESOLVED_CERTIFICATE" and x["tau_exact"] is None


# --- 3. no false PASS with an incomplete certificate ---------------------------------------------------------------------------

def test_no_false_pass_with_an_incomplete_certificate():
    good = {"P_p": 3.0, "P_s": 2.0, "Q": 1.0, "Gamma_up": 0.25, "B_E": 0.25}
    assert pl.assemble(good, OK)["status"] == "PASS"
    for p in OK:
        for st in ("NOT_ESTABLISHED", "NOT_CHECKABLE"):
            r = pl.assemble(good, {**OK, p: {"state": st, "reason": "x"}})
            assert r["status"] == "UNRESOLVED_CERTIFICATE" and r["reason_premises"] == [p]
    for k in good:
        for v in (None, "inf", "nan"):
            r = pl.assemble({**good, k: v}, OK)
            assert r["status"] == "UNRESOLVED_CERTIFICATE", (k, v)
    r = pl.assemble({k: v for k, v in good.items() if k != "B_E"}, OK)
    assert r["tau_exact"] is None and r["status"] == "UNRESOLVED_CERTIFICATE"  # no assumed component
    r = pl.not_executed({"status": "UNRESOLVED_C3PLUS_PREMISE: P-3: FTZ/DAZ probe"}, 2)
    assert r["status"] == "UNRESOLVED" and "OD-8" in r["classification"]["reasons"][0]
    for st in ("FAIL_C3PLUS_BOUND_CHECK: x", "PROTECTIVE_STOP: x", "STOP_ZERO_FIELD: x", "FAIL_TOOL: x"):
        assert pl.not_executed({"status": st}, 3)["status"] == "UNRESOLVED"
    assert pl.not_executed(None, 4)["status"] == "UNRESOLVED"


def test_reproduction_is_required_for_p_a(journal):
    r = pl.analyse({"kind": "journal", "journal": str(journal), "s": 2}, reproduce=False)
    assert r["premises"]["P-a"]["state"] == "NOT_CHECKABLE" and r["status"] == "UNRESOLVED_CERTIFICATE"
    v = float.fromhex(pl.analyse({"kind": "journal", "journal": str(journal), "s": 2}, reproduce=False)["row"]["Q_hat"])
    r = pl.analyse({"kind": "journal", "journal": str(journal), "s": 2}, bound_values={"Q_hat": math.nextafter(v, 0).hex()})
    assert r["premises"]["P-a"]["state"] == "NOT_ESTABLISHED" and r["status"] == "UNRESOLVED_CERTIFICATE"


def test_a_value_not_reproduced_in_the_separate_process_breaks_p_a(journal, monkeypatch):
    real = pl.run_child

    def differing(spec, timeout=1800):
        job = real(spec, timeout)
        v = math.nextafter(float.fromhex(job["row"]["Q_hat"]), 0.0)
        job["row"]["Q_hat"], job["row"]["components"]["Q"] = v.hex(), v
        return job

    monkeypatch.setattr(pl, "run_child", differing)
    r = pl.analyse({"kind": "journal", "journal": str(journal), "s": 2})
    assert r["reproduction"]["bitwise_equal"] is False and "Q_hat" in r["reproduction"]["differences"]
    assert r["premises"]["P-a"]["state"] == "NOT_ESTABLISHED" and r["status"] == "UNRESOLVED_CERTIFICATE"


# --- 4. E1/E2, E_DIAG, strict sign -----------------------------------------------------------------------------------------------

def test_e1_e2_e_diag_and_the_strict_sign():
    r = pl.assemble({"P_p": 2.5, "P_s": 2.0, "Q": 0.5, "Gamma_up": 0.25, "B_E": 0.25}, OK)
    assert r["status"] == "UNRESOLVED_RESOLVABILITY"  # Q^ = tau'
    r = pl.assemble({"P_p": "0x1.0000000000001p-1", "P_s": 0.0, "Q": "0x1.0000000000001p-1", "Gamma_up": 0.25, "B_E": 0.25}, OK)
    assert r["status"] == "PASS"  # Q^ = tau' + 2^-53
    r = pl.assemble({"P_p": 0.0, "P_s": 0.5, "Q": 0.0, "Gamma_up": 0.25, "B_E": 0.25}, OK)
    assert (r["e1"], r["e2"], r["status"]) == (True, True, "UNRESOLVED_RESOLVABILITY")  # E1, E2 non-strict boundaries
    r = pl.assemble({"P_p": 3.0, "P_s": 2.0, "Q": 0.25, "Gamma_up": 0.25, "B_E": 0.25}, {**OK, "P-d": {"state": "NOT_ESTABLISHED"}})
    assert (r["status"], r["e1"], r["e2"], r["e_diag"]) == ("UNRESOLVED_CERTIFICATE", True, False, True)


# --- 5. real detectors ------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("injection, premise", [("perturb_field_s_rel_2^-40", "P-b"), ("subnormal_field_s_node", "P-c"),
                                                ("grid_D_2^-1074", "P-d"), ("mu_low_zero", "P-f"), ("B_E_sqrt_negative_D", "P-f")])
def test_injections_reach_the_real_detectors(journal, injection, premise):
    r = pl.analyse({"kind": "journal", "journal": str(journal), "s": 2, "injection": injection})
    assert r["status"] == "UNRESOLVED_CERTIFICATE" and premise in r["classification"]["reason_premises"]


def test_n7_detector_and_source_of_the_fault(journal):
    r = pl.analyse({"kind": "journal", "journal": str(journal), "s": 1, "injection": "prev_factors_2^-600"})
    assert "P-e" in r["classification"]["reason_premises"] and r["row"]["N7_applicable"] is True
    r = pl.analyse({"kind": "journal", "journal": str(journal), "s": 2, "injection": "scale_field_s_2^1000"})
    assert {"P-c", "P-f"} <= set(r["classification"]["reason_premises"]) and r["status"] == "UNRESOLVED_CERTIFICATE"


# --- 9. binding, hash chain, binary representation -------------------------------------------------------------------------------

def test_hash_chain_and_binding(journal, tmp_path):
    j, arrays = pl.load_journal(journal)
    assert all(v["ok"] for v in pl.bind_journal(j, arrays).values())
    assert pl.bind_journal(j, arrays)[2]["ok"] and j["steps"][1]["given_in_sha"] == j["steps"][0]["next_given_sha"]
    # a significant input changed by 1 ulp is always detected
    a = dict(arrays)
    a["s1_next_given"] = a["s1_next_given"].copy()
    a["s1_next_given"].flat[0] = math.nextafter(float(a["s1_next_given"].flat[0]), math.inf)
    b = pl.bind_journal(j, a)
    assert not b[1]["ok"] and not b[2]["ok"] and any("chain" in x for x in b[2]["reasons"])
    # a hash binding is not a coincidence of values: -0.0 for +0.0 leaves the computed values but breaks the binding
    d = tmp_path / "negzero"
    shutil.copytree(journal, d)
    arr = pl.op8().load_arrays(d)
    arr["s1_companion_in"] = arr["s1_companion_in"].copy()
    arr["s1_companion_in"].flat[0] = -0.0
    np.savez(d / "arrays.npz", **arr)
    clean = pl.analyse({"kind": "journal", "journal": str(journal), "s": 1}, reproduce=False)
    neg = pl.analyse({"kind": "journal", "journal": str(d), "s": 1})
    assert all(neg["row"][k] == clean["row"][k] for k in ("P_hat_p", "P_hat_s", "Q_hat", "Gamma_up", "B_E_up"))
    assert neg["premises"]["P-a"]["state"] == "NOT_ESTABLISHED" and neg["status"] == "UNRESOLVED_CERTIFICATE"


def test_binary_representation_is_kept():
    for x in (0.1, -0.0, 2.0**-1074, 1.7976931348623157e308, -3.5):
        assert float.fromhex(pl.fhex(x)).hex() == x.hex()
    f = Fraction(3, 2**1080) + 1
    assert Fraction(pl.ftext(f)) == f and pl.parse_recorded("hex:" + (0.1).hex()) == Fraction(0.1)
    assert pl.parse_recorded(pl.ftext(f)) == f


# --- D0 regression (published EQ and QA-EB rows) ----------------------------------------------------------------------------------
# Protocol 7, area D0: a repeat gives the published values bitwise, otherwise UNRESOLVED. D-025 / AGENTS.md 4: bitwise equality is
# required only with a fixed order of the arithmetic operations. The regression is split in two checks:
#  - portable (every platform): composition and identifiers of the sample, the input of s = 1, premises P-a...P-f and the status of
#    the certificate of the recomputed rows (whole pipeline: OP-8 journal, binding, OP-8 re-run, reproduction in a separate process),
#    exact dP and tau', the directed bounds (gamma contract, B_E_up >= B_E), E1/E2 and the strict sign; the published rows on their
#    own values. Values of two machines are never compared with each other, and no tolerance is used;
#  - bitwise (recomputed against published values): a row equal bitwise is reproduced on any path; a difference is a FAIL only on a
#    computational path shown identical to the recorded path of the published runs (_d0_path_identity); otherwise that comparison is
#    NOT_VERIFIED (protocol 7: UNRESOLVED, not FAIL) and the check is skipped with the reason. sys.platform alone fixes no path.

D0_IDS = ("chain|1x1:64x64|N1|Y|s1", "chain|1x3:32x96|N1|Y|s1", "chain|1x0.333333:96x32|N1|Y|s1", "pairs|1x1:40x64|N1|Y|s1",
          "pairs|1x1:64x40|N1|Y|s1", "pairs|1x1:47x53|N1|Y|s1", "chain|1x2:32x64|N1|Y|s1", "chain|1x0.5:64x32|N1|Y|s1",
          "chain|1x1:32x32|N1|Y|s1")
D0_RESULTS = ROOT / "docs" / "results"
D0_META = {"Q3": D0_RESULTS / "r5_cccc_q3_prequalification" / "meta.json",
           "QA-EB": D0_RESULTS / "r5_cccc_q4_energy_error_bound" / "meta.json",
           "EQ": D0_RESULTS / "r5_cccc_q4_energy_bc" / "meta.json"}
# (component of the recomputed row, published run, column): the six components of the former test and gamma_up
D0_BITWISE = (("P_hat_p", "QA-EB", "P_hat_prev"), ("P_hat_s", "QA-EB", "P_hat_s"), ("Q_hat", "QA-EB", "Q_hat"),
              ("B_E_up", "QA-EB", "B_E_up"), ("Gamma_up", "EQ", "Gamma_up"), ("gamma_down", "EQ", "gamma_down"),
              ("gamma_up", "EQ", "gamma_up"))
D0_PUBLISHED_FLAGS = {"QA-EB": {"status": "VERIFIED", "q3_state_reproduced_bitwise": "True", "B_E_finite": "True", "E_s_le_B_E": "True"},
                      "EQ": {"status": "COMPLETE", "q3_state_reproduced_bitwise": "True", "gamma_inclusion_exact": "True",
                             "Gamma_up_holds": "True"}}
# parameters that fix the arithmetic path of a D0 row: OS build (C runtime), processor model (CPU dispatch of NumPy and the OpenBLAS
# core selection), NumPy and SciPy builds, NumPy's OpenBLAS build and file, the cores actually used by NumPy's and SciPy's OpenBLAS
# (read from the loaded libraries) and their thread counts, the NumPy variables that restrict its CPU dispatch
D0_PATH_KEYS = ("platform", "machine", "processor", "numpy", "scipy", "numpy_blas", "numpy_blas_library", "numpy_blas_core",
                "scipy_blas_core", "blas_threads", "numpy_cpu_env")
NPY_CPU_ENV = ("NPY_DISABLE_CPU_FEATURES", "NPY_ENABLE_CPU_FEATURES")
OPENBLAS = ((np, "64_"), (scipy, ""))  # NumPy's ILP64 and SciPy's LP64 scipy-openblas (symbol suffixes), as in QA-EB blas_info


def _openblas(pkg, suffix: str):
    """(ctypes library, file name) of the scipy-openblas bundled with ``pkg`` (wheel layout <pkg>.libs), or (None, reason)."""
    libs = Path(pkg.__file__).resolve().parent.parent / f"{pkg.__name__}.libs"
    found = sorted(libs.glob("*openblas*")) if libs.is_dir() else []
    if len(found) != 1:
        return None, f"{len(found)} OpenBLAS libraries in {libs.name}"
    try:
        lib = ctypes.CDLL(str(found[0]))
        for name in ("get_corename", "get_num_threads", "set_num_threads"):
            getattr(lib, f"scipy_openblas_{name}{suffix}")
    except (OSError, AttributeError) as exc:
        return None, f"{found[0].name}: {exc}"
    return lib, found[0].name


def _openblas_state(pkg, suffix: str) -> dict:
    lib, name = _openblas(pkg, suffix)
    if lib is None:
        return {"library": None, "core": None, "threads": None, "error": name}
    core = getattr(lib, f"scipy_openblas_get_corename{suffix}")
    core.restype = ctypes.c_char_p
    threads = getattr(lib, f"scipy_openblas_get_num_threads{suffix}")
    threads.restype = ctypes.c_int
    return {"library": name, "core": core().decode(), "threads": threads()}


@contextlib.contextmanager
def _one_blas_thread():
    """One OpenBLAS thread in NumPy's and SciPy's library of this process, as recorded for the published runs; the former counts are
    restored afterwards. A library that cannot be set keeps its count (the path is then not identical)."""
    saved = []
    for pkg, suffix in OPENBLAS:
        lib, _ = _openblas(pkg, suffix)
        if lib is None:
            continue
        get, put = getattr(lib, f"scipy_openblas_get_num_threads{suffix}"), getattr(lib, f"scipy_openblas_set_num_threads{suffix}")
        get.restype, put.argtypes = ctypes.c_int, [ctypes.c_int]
        saved.append((put, get()))
        put(1)
    try:
        yield [n for _, n in saved]
    finally:
        for put, n in saved:
            put(n)


def _d0_current_path() -> dict:
    """The path parameters of this process (D0_PATH_KEYS) and, under "info", further ones for the report only (not compared)."""
    nb, sb = _openblas_state(np, "64_"), _openblas_state(scipy, "")
    try:
        b = np.show_config(mode="dicts")["Build Dependencies"]["blas"]
        numpy_blas = f"{b['name']} {b['version']}"
    except Exception:  # noqa: BLE001 - not readable: the parameter is then missing, never guessed
        numpy_blas = None
    try:
        from numpy._core._multiarray_umath import __cpu_baseline__, __cpu_dispatch__
        dispatch = {"baseline": list(__cpu_baseline__), "dispatch": list(__cpu_dispatch__)}
    except ImportError:
        dispatch = None
    return {"platform": platform.platform(), "machine": platform.machine(), "processor": platform.processor() or None,
            "numpy": np.__version__, "scipy": scipy.__version__, "numpy_blas": numpy_blas, "numpy_blas_library": nb["library"],
            "numpy_blas_core": nb["core"], "scipy_blas_core": sb["core"],
            "blas_threads": None if None in (nb["threads"], sb["threads"]) else (nb["threads"], sb["threads"]),
            "numpy_cpu_env": {k: os.environ.get(k) for k in NPY_CPU_ENV},
            "info": {"python": platform.python_version(), "sys_platform": sys.platform, "scipy_blas_library": sb["library"],
                     "OPENBLAS_CORETYPE": os.environ.get("OPENBLAS_CORETYPE"), "numpy_cpu": dispatch}}


def _d0_recorded_path(meta: dict | None = None) -> tuple[dict, list[str]]:
    """The computational path recorded by the published D0 runs (Q3 fields, QA-EB energies and B_E, EQ gamma) in their meta.json. A
    parameter the runs record differently, or do not record, is None and listed: no identity can be shown for it."""
    m = {run: json.loads(Path(p).read_text(encoding="utf-8")) for run, p in (meta or D0_META).items()}
    rec, problems = {}, []
    for key in ("platform", "machine", "processor", "numpy", "scipy"):
        vals = {run: mm.get(key) for run, mm in m.items()}
        rec[key] = vals["Q3"] if None not in vals.values() and len(set(vals.values())) == 1 else None
        if rec[key] is None:
            problems.append(f"{key} recorded as {vals}")
    eb = m["QA-EB"].get("blas_main") or {}
    nb = eb.get("numpy_blas") or {}
    rec["numpy_blas"] = f"{nb['name']} {nb['version']}" if nb.get("name") and nb.get("version") else None
    if rec["numpy_blas"] is None or m["Q3"].get("blas") != rec["numpy_blas"]:
        problems.append(f"NumPy BLAS recorded as {m['Q3'].get('blas')!r} (Q3) and {rec['numpy_blas']!r} (QA-EB)")
        rec["numpy_blas"] = None
    rec["numpy_blas_library"] = eb.get("library")
    rec["numpy_blas_core"] = eb.get("core")  # read from the loaded library in the QA-EB run, which reproduced the Q3 state bitwise
    # SciPy's OpenBLAS core is not recorded. It is taken equal to the recorded NumPy core: both libraries select the core by the same
    # rule (OPENBLAS_CORETYPE, else the processor), and the processor is compared. An inference, not a record (QA report 12.8).
    rec["scipy_blas_core"] = eb.get("core")
    one = dict.fromkeys(("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"), "1")
    env = {"Q3": m["Q3"].get("threads"), "QA-EB": m["QA-EB"].get("threads_env"), "EQ": m["EQ"].get("threads")}
    rec["blas_threads"] = (1, 1) if all(v == one for v in env.values()) and eb.get("threads") == 1 else None
    if rec["blas_threads"] is None:
        problems.append(f"BLAS threads recorded as {env}, measured {eb.get('threads')!r}")
    # NumPy's CPU dispatch is not recorded: fixed by the processor model and the NumPy build unless an NPY_*_CPU_FEATURES variable
    # restricts it; these are taken as unset in the published runs (an inference, QA report 12.8)
    rec["numpy_cpu_env"] = dict.fromkeys(NPY_CPU_ENV)
    return rec, problems


def _d0_path_identity(now: dict, rec: dict, problems=()) -> tuple[bool, list[str]]:
    """Identical computational path only when every parameter of D0_PATH_KEYS is recorded, readable here and equal; otherwise
    (False, the differences). sys.platform is not a parameter: equal on every Windows machine, it fixes neither kernel nor order."""
    diff = [f"record: {p}" for p in problems]
    for k in D0_PATH_KEYS:
        if rec.get(k) is None:
            diff.append(f"{k}: not recorded")
        elif now.get(k) is None:
            diff.append(f"{k}: not readable here")
        elif now[k] != rec[k]:
            diff.append(f"{k}: {now[k]!r} here, {rec[k]!r} recorded")
    return not diff, diff


def _d0_identity(d0: dict) -> tuple[bool, list[str]]:
    """The path of the D0 computation, read before and after it, against the record."""
    diff = []
    for when in ("path", "path_after"):
        diff += [x for x in _d0_path_identity(d0[when], d0["recorded"], d0["record_problems"])[1] if x not in diff]
    return not diff, diff


def _d0_bitwise_mismatches(case: dict) -> list[str]:
    ga, row, out = pl.ga(), case["result"]["row"], []
    for comp, run, col in D0_BITWISE:
        if case.get(run) is None:
            out.append(f"{comp}: no published {run} row")
            continue
        pub = ga.read_double(case[run][col]).hex()
        if row.get(comp) != pub:
            out.append(f"{comp}: computed {row.get(comp)}, published {pub}")
    return out


def _d0_bitwise_verdict(identical: bool, mismatches: list[str]) -> str:
    """REPRODUCED: bitwise equal (on any path). FAIL: a difference on a path shown identical. NOT_VERIFIED: a difference on a path
    not shown identical - the published value is neither confirmed nor refuted there (protocol 7: UNRESOLVED)."""
    if not mismatches:
        return "REPRODUCED"
    return "FAIL" if identical else "NOT_VERIFIED"


def _exact_tests(P_p, P_s, Q, G, B_E):
    """dP, tau', E1, E2 and the strict sign of resolvability of binary64 components, in exact rationals (D-054.1 items 1, 3)."""
    dP, tau = Fraction(P_p) - Fraction(P_s), Fraction(G) + Fraction(B_E)
    return dP, tau, dP >= -tau, abs(dP - Fraction(Q)) <= tau, Fraction(Q) > tau


def _d0_composition_defects(cases: list[dict]) -> list[str]:
    bad = []
    ids = [c["id"] for c in cases]
    if ids != list(D0_IDS):
        bad.append(f"the D0 sample is {ids}, not the published one {list(D0_IDS)}")
    bad += [f"{c['id']}: no published {run} row" for c in cases for run in ("QA-EB", "EQ") if c.get(run) is None]
    return bad


def _d0_portable_defects(case: dict) -> list[str]:
    """The checks of a D0 row that need no equality of values across machines; empty when all hold."""
    ga, bad = pl.ga(), []
    res, q, h, e = case["result"], case["Q3"], case.get("QA-EB"), case.get("EQ")
    if h is None or e is None:
        return [f"{case['id']}: no published QA-EB / EQ row"]
    # identifiers; the input of s = 1 is T_1 with a zero companion on the grid of the Q3 row
    if not ga.row_id(q) == h["id"] == e["id"] == case["id"]:
        bad.append(f"identifiers differ: Q3 {ga.row_id(q)}, QA-EB {h['id']}, EQ {e['id']}, case {case['id']}")
    sp = ga.spec_of(q)
    j, arrays = pl.load_journal(case["journal"])
    cfg = pl.config_of(j["meta"])
    if (cfg.lx, cfg.nx, cfg.ly, cfg.ny, cfg.N) != (sp["lx"], sp["nx"], sp["ly"], sp["ny"], 1):
        bad.append(f"the journal configuration {cfg} is not the grid of the Q3 row")
    t1, zero = pl.hs.start_set(sp["nx"], 1), np.zeros((sp["ny"] - 1, 1))
    given, comp_in = arrays["s1_given_in"], arrays["s1_companion_in"]
    if given.shape != t1.shape or given.tobytes() != t1.tobytes() or comp_in.shape != zero.shape or comp_in.tobytes() != zero.tobytes():
        bad.append("the input of s = 1 is not T_1 with a zero companion")
    # the certificate of the recomputed row through the whole pipeline
    states = {k: v["state"] for k, v in res["premises"].items()}
    if states != dict.fromkeys(("P-a", "P-b", "P-c", "P-d", "P-e", "P-f"), "ESTABLISHED"):
        bad.append(f"premises {states}")
    if not (res.get("op8_rerun") or {}).get("reproduced") or res["reproduction"].get("bitwise_equal") is not True:
        bad.append("OP-8 re-run or the reproduction in a separate process differs")
    if res["contradictions"] or res["faults"] or res["row"]["contradictions"] or res["row"]["faults"]:
        bad.append("contradictions or faults recorded")
    bad += [f"row: {x}" for x in pl.row_self_consistency(res["row"])]
    if res["status"] != "PASS" or res["classification"]["status"] != "PASS":
        bad.append(f"status {res['status']} / {res['classification']['status']}, published rows PASS")
    row, c = res["row"], res["classification"]
    x = {k: float.fromhex(row[k]) for k in ("P_hat_p", "P_hat_s", "Q_hat", "Gamma_up", "B_E_up", "gamma_down", "gamma_up")}
    dP, tau, e1, e2, resolvable = _exact_tests(x["P_hat_p"], x["P_hat_s"], x["Q_hat"], x["Gamma_up"], x["B_E_up"])
    if Fraction(c["dP_exact"]) != dP or Fraction(c["tau_exact"]) != tau:
        bad.append("dP or tau' is not the exact rational of the recorded components")
    bad += [f"gamma contract: {v}" for v in pl.gamma_contract(Fraction(row["gamma_exact"]), x["gamma_down"], x["gamma_up"],
                                                              x["Gamma_up"])]
    if not Fraction(x["B_E_up"]) >= Fraction(row["B_E_exact"]):
        bad.append("B_E_up is below the exact B_E")
    if row["diagnostics"]["second_path_gamma"].get("consistent") is not True:
        bad.append("the exact gamma of the second path is not consistent")
    if not (e1 and e2 and c["e1"] is True and c["e2"] is True):
        bad.append("E1 / E2 violated with established premises")
    if not resolvable:
        bad.append("Q^ is not > tau'")
    # the published rows on their own values: flags, agreement of QA-EB and EQ, Gamma_up = max, E1/E2 and the strict sign
    for run, flags in D0_PUBLISHED_FLAGS.items():
        bad += [f"published {run} {k} = {case[run].get(k)!r}" for k, v in flags.items() if case[run].get(k) != v]
    p = {k: ga.read_double(h[col]) for k, col in (("P_p", "P_hat_prev"), ("P_s", "P_hat_s"), ("Q", "Q_hat"), ("B_E", "B_E_up"))}
    for k, col in (("P_p", "P_hat_prev"), ("P_s", "P_hat_s"), ("Q", "Q_hat")):
        if ga.read_double(e[col]).hex() != p[k].hex():
            bad.append(f"published {col}: QA-EB and EQ differ")
    gd, gu, G = (ga.read_double(e[k]) for k in ("gamma_down", "gamma_up", "Gamma_up"))
    if not (gd <= gu and G.hex() == max(abs(gd), abs(gu)).hex()):
        bad.append("published Gamma_up is not max(|gamma_down|, |gamma_up|) of an ordered interval")
    _, _, pe1, pe2, pres = _exact_tests(p["P_p"], p["P_s"], p["Q"], G, p["B_E"])
    if not (pe1 and pe2 and pres):
        bad.append("published row: E1 / E2 or the strict sign does not hold")
    return bad


@pytest.fixture(scope="module")
def d0(tmp_path_factory):
    """The nine published D0 rows recomputed through the pipeline (journal of s = 1, analyse) with one BLAS thread."""
    ga = pl.ga()
    q3 = ga.read_q3_rows(D0_RESULTS / "r5_cccc_q3_prequalification" / "sequences_e1_e7_e9.csv")
    pub = {}
    for run, name in (("QA-EB", "r5_cccc_q4_energy_error_bound"), ("EQ", "r5_cccc_q4_energy_bc")):
        with open(D0_RESULTS / name / "rows.csv", encoding="utf-8") as f:
            pub[run] = {r["id"]: r for r in csv.DictReader(f)}
    sel = [r for r in q3 if r["N"] == "1" and r["j_s"] == "1" and r["first"] == "Y" and int(r["nx"]) * int(r["ny"]) <= 4096]
    out, cases = tmp_path_factory.mktemp("d0"), []
    threads_before = [_openblas_state(pkg, suffix)["threads"] for pkg, suffix in OPENBLAS]
    with _one_blas_thread():
        path = _d0_current_path()
        for i, r in enumerate(sel):
            sp, rid = ga.spec_of(r), ga.row_id(r)
            cfg = pl.op8().TrajectoryConfig(sp["lx"], sp["nx"], sp["ly"], sp["ny"], 1, 1.0, 1.0, 1)
            jd = pl.run_trajectory_journal(cfg, out / f"row{i}")
            cases.append({"id": rid, "Q3": r, "QA-EB": pub["QA-EB"].get(rid), "EQ": pub["EQ"].get(rid), "journal": str(jd),
                          "result": pl.analyse_journal(jd)[0]})
        path_after = _d0_current_path()
    rec, problems = _d0_recorded_path()
    return {"cases": cases, "path": path, "path_after": path_after, "recorded": rec, "record_problems": problems,
            "threads_before": threads_before}


def _d0_case(d0, rid):
    return copy.deepcopy(next(c for c in d0["cases"] if c["id"] == rid))


def test_d0_regression_portable(d0):
    assert _d0_composition_defects(d0["cases"]) == []
    for case in d0["cases"]:
        assert _d0_portable_defects(case) == [], case["id"]


def test_d0_regression_bitwise(d0):
    identical, why = _d0_identity(d0)
    res = {c["id"]: (_d0_bitwise_verdict(identical, m), m) for c in d0["cases"] for m in [_d0_bitwise_mismatches(c)]}
    assert not {k: m for k, (v, m) in res.items() if v == "FAIL"}, "D0 not reproduced bitwise on the recorded computational path"
    open_ = {k: m for k, (v, m) in res.items() if v == "NOT_VERIFIED"}
    if open_:
        first = next(iter(open_.items()))
        pytest.skip(f"D0 bitwise regression NOT_VERIFIED (protocol 7: UNRESOLVED) for {len(open_)} of {len(res)} rows; reproduced "
                    f"bitwise: {len(res) - len(open_)}. Their values differ from the published ones on a computational path not shown "
                    f"identical to the recorded one: {'; '.join(why)}. First: {first[0]}: {first[1][0]}")
    assert all(v == "REPRODUCED" for v, _ in res.values()) and len(res) == len(D0_IDS)


# --- D0: the path condition and the portable checks (negative tests) --------------------------------------------------------------

def test_d0_recorded_path_is_usable(d0):
    rec, problems = d0["recorded"], d0["record_problems"]
    assert problems == [] and all(rec[k] is not None for k in D0_PATH_KEYS)
    assert rec["numpy_blas_core"] == "Haswell" and rec["blas_threads"] == (1, 1) and rec["numpy_blas"].startswith("scipy-openblas")


def test_d0_inconsistent_or_missing_record_gives_no_identity(d0, tmp_path):
    meta = {}
    for run, p in D0_META.items():
        m = json.loads(p.read_text(encoding="utf-8"))
        if run == "EQ":
            m["processor"] = m["processor"] + " (other run)"
        if run == "QA-EB":
            m["blas_main"].pop("core")
        meta[run] = tmp_path / f"{run}.json"
        meta[run].write_text(json.dumps(m), encoding="utf-8")
    rec, problems = _d0_recorded_path(meta)
    assert rec["processor"] is None and rec["numpy_blas_core"] is None and any("processor" in x for x in problems)
    ok, diff = _d0_path_identity(dict(d0["recorded"]), rec, problems)
    assert not ok and "processor: not recorded" in diff and "numpy_blas_core: not recorded" in diff


@pytest.mark.parametrize("key", D0_PATH_KEYS)
def test_d0_every_path_parameter_is_needed(d0, key):
    rec = d0["recorded"]
    other = {"blas_threads": (24, 1), "numpy_cpu_env": {"NPY_DISABLE_CPU_FEATURES": "X86_V3", "NPY_ENABLE_CPU_FEATURES": None}}
    assert _d0_path_identity(rec, rec) == (True, [])
    ok, diff = _d0_path_identity({**rec, key: other.get(key, "other")}, rec)
    assert not ok and len(diff) == 1 and diff[0].startswith(f"{key}:")
    assert _d0_path_identity({**rec, key: None}, rec) == (False, [f"{key}: not readable here"])


def test_d0_path_read_after_the_computation_counts(d0):
    rec = d0["recorded"]
    moved = {**d0, "path": dict(rec), "path_after": {**rec, "numpy_blas_core": "Sandybridge"}, "record_problems": []}
    ok, diff = _d0_identity(moved)
    assert not ok and diff == ["numpy_blas_core: 'Sandybridge' here, 'Haswell' recorded"]
    assert _d0_identity({**moved, "path_after": dict(rec)}) == (True, [])


def test_d0_win32_alone_is_not_an_identical_path(d0):
    rec = d0["recorded"]
    ok, diff = _d0_path_identity({"sys_platform": "win32"}, rec)
    assert not ok and len(diff) == len(D0_PATH_KEYS)
    # the fingerprint of the preflight (sys.platform, machine, processor, NumPy) equal, another OpenBLAS core: not identical
    assert {"sys_platform", "machine", "processor", "numpy"} <= set(pl.pf().platform_fingerprint())
    ok, diff = _d0_path_identity({**rec, "numpy_blas_core": "Sandybridge", "scipy_blas_core": "Sandybridge"}, rec)
    assert not ok and [x.split(":")[0] for x in diff] == ["numpy_blas_core", "scipy_blas_core"]
    # a Windows CI runner: another OS build and processor (its core is not established): not identical
    runner = {**rec, "platform": "Windows-10-10.0.20348-SP0", "processor": "AMD64 Family 25 Model 1 Stepping 1, AuthenticAMD"}
    assert not _d0_path_identity(runner, rec)[0]


def test_d0_platform_mismatch_is_not_a_false_fail(d0):
    case = _d0_case(d0, "chain|1x3:32x96|N1|Y|s1")
    # the value of GitHub Actions run 40946309468 (Windows, Python 3.13 and 3.14): 1 ULP from the published P_hat_s
    assert case["QA-EB"]["P_hat_s"] == repr(float.fromhex("-0x1.7e08f45e0914ap-10"))
    case["result"]["row"]["P_hat_s"] = "-0x1.7e08f45e09149p-10"
    mism = _d0_bitwise_mismatches(case)
    assert mism == ["P_hat_s: computed -0x1.7e08f45e09149p-10, published -0x1.7e08f45e0914ap-10"]
    rec = d0["recorded"]
    runner = {**rec, "platform": "Windows-10-10.0.20348-SP0", "processor": "AMD64 Family 25 Model 1 Stepping 1, AuthenticAMD"}
    assert _d0_bitwise_verdict(_d0_path_identity(runner, rec)[0], mism) == "NOT_VERIFIED"
    assert _d0_bitwise_verdict(_d0_path_identity(rec, rec)[0], mism) == "FAIL"
    # the portable checks do not depend on the path condition: the unchanged row keeps them all
    assert _d0_portable_defects(_d0_case(d0, case["id"])) == []


@pytest.mark.parametrize("comp, run, col", [("P_hat_p", "QA-EB", "P_hat_prev"), ("P_hat_s", "QA-EB", "P_hat_s"),
                                             ("Q_hat", "QA-EB", "Q_hat"), ("B_E_up", "QA-EB", "B_E_up"), ("Gamma_up", "EQ", "Gamma_up"),
                                             ("gamma_down", "EQ", "gamma_down"), ("gamma_up", "EQ", "gamma_up")])
def test_d0_bitwise_is_strict_on_an_identical_path(d0, comp, run, col):
    # the compared components listed here, not taken from D0_BITWISE: a component dropped from the comparison is then detected
    case = _d0_case(d0, "chain|1x1:64x64|N1|Y|s1")
    pub = pl.ga().read_double(case[run][col])
    for v in (math.nextafter(pub, math.inf), math.nextafter(pub, -math.inf)) + ((-pub,) if pub == 0.0 else ()):
        case["result"]["row"][comp] = v.hex()
        mism = _d0_bitwise_mismatches(case)
        assert len(mism) == 1 and mism[0].startswith(f"{comp}:")
        assert _d0_bitwise_verdict(True, mism) == "FAIL" and _d0_bitwise_verdict(False, mism) == "NOT_VERIFIED"
    case["result"]["row"][comp] = pub.hex()
    assert _d0_bitwise_mismatches(case) == [] and _d0_bitwise_verdict(False, []) == "REPRODUCED"


def _set_hex(field, f):
    def edit(case):
        row = case["result"]["row"]
        row[field] = f(float.fromhex(row[field]), row).hex()
    return edit


def _set(path, value):
    def edit(case):
        d = case
        for k in path[:-1]:
            d = d[k]
        d[path[-1]] = value
    return edit


def _input_not_t1(case):
    j = Path(case["journal"])
    d = j.parent / (j.name + "_input")
    shutil.copytree(j, d)
    arr = pl.op8().load_arrays(d)
    arr["s1_given_in"] = arr["s1_given_in"].copy()
    arr["s1_given_in"].flat[1] = math.nextafter(float(arr["s1_given_in"].flat[1]), math.inf)
    np.savez(d / "arrays.npz", **arr)
    case["journal"] = str(d)


D0_VIOLATIONS = {
    "Gamma_up below max(|gamma|)": (_set_hex("Gamma_up", lambda v, r: math.nextafter(v, 0.0)), "gamma contract"),
    "interval above the exact gamma": (_set_hex("gamma_down", lambda v, r: float.fromhex(r["gamma_up"])), "gamma contract"),
    "B_E_up below the exact B_E": (_set_hex("B_E_up", lambda v, r: float(Fraction(r["B_E_exact"])) / 2), "B_E_up is below"),
    "E2 violated": (_set_hex("P_hat_s", lambda v, r: 2 * v), "E1 / E2"),
    "dP not exact": (_set(["result", "classification", "dP_exact"], "1/3"), "not the exact rational"),
    "tau not exact": (_set(["result", "classification", "tau_exact"], "0"), "not the exact rational"),
    "premise not established": (_set(["result", "premises", "P-c"], {"state": "NOT_ESTABLISHED", "reason": "x"}), "premises"),
    "OP-8 re-run differs": (_set(["result", "op8_rerun", "reproduced"], False), "OP-8 re-run"),
    "reproduction differs": (_set(["result", "reproduction", "bitwise_equal"], False), "separate process"),
    "status not PASS": (_set(["result", "status"], "UNRESOLVED_RESOLVABILITY"), "status"),
    "component record inconsistent": (_set(["result", "row", "components", "Q"], 1.0), "row:"),
    "second path of gamma inconsistent": (_set(["result", "row", "diagnostics", "second_path_gamma", "consistent"], False),
                                          "second path"),
    "input of s = 1 not T_1": (_input_not_t1, "not T_1"),
    "published Gamma_up not max": (_set(["EQ", "Gamma_up"], "1.0"), "published Gamma_up"),
    "published sources disagree": (_set(["EQ", "P_hat_s"], "-1.0"), "QA-EB and EQ differ"),
    "published E2 violated": (lambda c: [_set([run, "Q_hat"], "1.0")(c) for run in ("QA-EB", "EQ")], "published row: E1 / E2"),
    "published status": (_set(["QA-EB", "status"], "UNRESOLVED"), "published QA-EB status"),
    "identifier": (_set(["id"], "chain|1x1:64x64|N1|Y|s2"), "identifiers differ"),
}


@pytest.mark.parametrize("name", list(D0_VIOLATIONS))
def test_d0_portable_violation_is_detected(d0, name):
    edit, expected = D0_VIOLATIONS[name]
    case = _d0_case(d0, "chain|1x1:64x64|N1|Y|s1")
    assert _d0_portable_defects(case) == []
    edit(case)
    bad = _d0_portable_defects(case)
    assert bad and any(expected in x for x in bad), bad


def test_d0_composition_is_checked(d0):
    cases = d0["cases"]
    assert _d0_composition_defects(cases) == []
    assert _d0_composition_defects(cases[:-1]) and _d0_composition_defects(cases[1:] + cases[:1])
    assert _d0_composition_defects([*cases[:-1], {**cases[-1], "EQ": None}]) != []


@pytest.mark.skipif(platform.machine().lower() not in ("amd64", "x86_64"), reason="OpenBLAS cores of x86-64 only")
@pytest.mark.parametrize("core", ["Haswell", "Sandybridge", "Prescott"])
def test_d0_real_openblas_cores_in_a_child_process(d0, core, tmp_path):
    """The two D0 tests with OPENBLAS_CORETYPE forced in a child process: another core never makes the bitwise check FAIL and leaves
    the portable checks intact; on the recorded path the recorded core keeps the bitwise check strict and passing."""
    xml = tmp_path / "d0.xml"
    p = subprocess.run([sys.executable, "-m", "pytest", str(Path(__file__)), "-q", "-p", "no:cacheprovider", f"--junitxml={xml}",
                        "-k", "test_d0_regression_portable or test_d0_regression_bitwise"], cwd=ROOT,
                       env={**os.environ, "OPENBLAS_CORETYPE": core}, capture_output=True, text=True, timeout=1800)
    if not xml.exists():
        pytest.skip(f"the core {core} does not run on this processor (exit {p.returncode}): {p.stderr[-300:]}")
    cases = {t.get("name"): t for t in ET.parse(xml).getroot().iter("testcase")}
    outcome = {n: ("failed" if t.find("failure") is not None or t.find("error") is not None else
                   "skipped" if t.find("skipped") is not None else "passed") for n, t in cases.items()}
    assert outcome["test_d0_regression_portable"] == "passed" and p.returncode == 0, p.stdout[-2000:]
    assert outcome["test_d0_regression_bitwise"] in ("passed", "skipped")
    if _d0_identity(d0)[0]:  # this machine is the recorded path: the forced core decides
        if core == d0["recorded"]["numpy_blas_core"]:
            assert outcome["test_d0_regression_bitwise"] == "passed"
        else:
            msg = cases["test_d0_regression_bitwise"].find("skipped").get("message")
            assert outcome["test_d0_regression_bitwise"] == "skipped" and "NOT_VERIFIED" in msg and "numpy_blas_core" in msg, msg


def test_d0_blas_threads_restored_after_the_d0_computation(d0):
    assert [_openblas_state(pkg, suffix)["threads"] for pkg, suffix in OPENBLAS] == d0["threads_before"]
    if d0["path"]["blas_threads"] is not None:
        assert d0["path"]["blas_threads"] == (1, 1) == d0["path_after"]["blas_threads"]


def test_d0_path_condition_is_local_to_the_d0_checks():
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    cond = {"_d0_path_identity", "_d0_identity", "_d0_bitwise_verdict", "_one_blas_thread"}
    users = {n.name for n in tree.body if isinstance(n, ast.FunctionDef) and n.name not in cond
             and {x.id for x in ast.walk(n) if isinstance(x, ast.Name)} & cond}
    assert users == {"d0", "test_d0_regression_bitwise", "test_d0_every_path_parameter_is_needed",
                     "test_d0_win32_alone_is_not_an_identical_path", "test_d0_platform_mismatch_is_not_a_false_fail",
                     "test_d0_bitwise_is_strict_on_an_identical_path", "test_d0_inconsistent_or_missing_record_gives_no_identity",
                     "test_d0_real_openblas_cores_in_a_child_process", "test_d0_path_read_after_the_computation_counts"}
    src = TOOL.read_text(encoding="utf-8")
    assert "OPENBLAS_CORETYPE" not in src and "_d0_" not in src


HISTORY_TESTS = tuple(f"tests/reference/{m}::{t}" for m, t in (
    ("test_r5_cccc_q4_d1_controls_prelaunch.py", "test_expectations_committed_before_the_code"),
    ("test_r5_cccc_q4_d1_controls_prelaunch.py", "test_expectations_history_check_fails_on_a_wrong_history"),
    ("test_r5_cccc_q4_d1_controls_prelaunch.py", "test_expectations_history_check_fails_on_a_changed_file"),
    ("test_r5_cccc_q4_d1_preflight.py", "test_manifest_tool_blob_is_the_historical_one"),
    ("test_r5_cccc_q4_d1_preflight.py", "test_sha_inventory_blobs_at_the_base_commit"),
    ("test_r5_cccc_q4_d1_preflight.py", "test_only_new_files_in_the_branch")))


HISTORY_CASES = 8  # the parametrized wrong-history test has three cases


def _history_verdict(shallow: bool, n: int, failed: list, skipped: list) -> str:
    """PASS: all historical cases executed and passed. SKIP: a shallow clone, where they skip themselves with their reason and none
    failed. FAIL otherwise - a missing case, a failure, or a skip in a full clone (there they must run)."""
    if n != HISTORY_CASES or failed:
        return "FAIL"
    if skipped:
        return "SKIP" if shallow else "FAIL"
    return "PASS"


def test_history_verdict():
    assert _history_verdict(False, 8, [], []) == "PASS" and _history_verdict(True, 8, [], ["a"]) == "SKIP"
    assert _history_verdict(False, 8, [], ["a"]) == "FAIL" and _history_verdict(True, 8, ["a"], []) == "FAIL"
    assert _history_verdict(False, 7, [], []) == "FAIL" and _history_verdict(True, 7, [], ["a"]) == "FAIL"


def test_historical_tests_of_a_and_b_still_run(tmp_path):
    shallow = subprocess.run(["git", "rev-parse", "--is-shallow-repository"], cwd=ROOT, capture_output=True, text=True)
    if shallow.returncode != 0:
        pytest.skip("no git repository: the historical tests skip themselves")
    xml = tmp_path / "h.xml"
    p = subprocess.run([sys.executable, "-m", "pytest", *HISTORY_TESTS, "-q", "-p", "no:cacheprovider", f"--junitxml={xml}"],
                       cwd=ROOT, capture_output=True, text=True, timeout=1800)
    t = list(ET.parse(xml).getroot().iter("testcase"))
    failed = [x.get("name") for x in t if x.find("failure") is not None or x.find("error") is not None]
    skipped = [x.get("name") for x in t if x.find("skipped") is not None]
    verdict = _history_verdict(shallow.stdout.strip() == "true", len(t), failed, skipped)
    assert verdict != "FAIL", (len(t), failed, skipped, p.stdout[-2000:])
    if verdict == "SKIP":
        pytest.skip(f"shallow clone: {len(skipped)} historical tests skipped with their reason, none failed")
    assert p.returncode == 0


# --- 10. guard against the D1 campaign -------------------------------------------------------------------------------------------

def test_guard_against_d1(tmp_path):
    TC = pl.op8().TrajectoryConfig
    for nx, ny in pl.op8().D1_GRIDS:
        if nx >= 3 and ny >= 3:
            with pytest.raises(pl.D1Refused):
                pl.guard_config(TC(1.0, nx, 1.0, ny, 1, s_max=1))
    with pytest.raises(pl.D1Refused):
        pl.guard_config(TC(1.0, 8, 1.25, 10, 2, s_max=1))
    with pytest.raises(pl.D1Refused):
        pl.guard_out(ROOT / "docs" / "results" / "x")
    with pytest.raises(pl.D1Refused):
        pl.compute_row(TC(1.0, 50, 1.0, 50, 1, s_max=1), 1, np.zeros((49, 1)), np.zeros((49, 1)), np.zeros((49, 1)))
    for args in (["--nx", "50", "--ny", "50", "--out", str(tmp_path / "x")], ["--nx", "8", "--ny", "10", "--out", "docs/x"]):
        p = subprocess.run([sys.executable, str(TOOL), "run", "--lx", "1", "--ly", "1", *args], cwd=ROOT, capture_output=True,
                           text=True)
        assert p.returncode == 2 and "refused" in p.stderr
    assert not (ROOT / "docs" / "x").exists()
    tree = ast.parse(TOOL.read_text(encoding="utf-8"))
    names = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)} | {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    assert not names & {"campaign_verdict", "mandatory_rows", "build_manifest", "run_probes", "write_manifest", "D1_RESULT"}
    assert "NOT D1 data" in pl.DATA_STATUS and pl.D1_START == "BLOCKED"


def test_tools_used_read_only():
    out = subprocess.run(["git", "status", "--porcelain", "--", *pl.USED_TOOLS, "src"], cwd=ROOT, capture_output=True, text=True)
    assert out.returncode != 0 or out.stdout.strip() == ""
    assert json.loads(json.dumps(pl.provenance(), default=pl._json_default))["tool_blob"] == pl.git_blob_id(TOOL)


# --- review of 16b5f69..472a262: I-2, Z-1...Z-6 -----------------------------------------------------------------------------------

def _row_arrays(journal, s):
    j, arrays = pl.load_journal(journal)
    return pl.config_of(j["meta"]), arrays[f"s{s}_solved_set"], arrays[f"s{s}_companion_solved"], arrays[f"s{s}_V_bc"]


def _edited_journal(journal, dst, edit):
    shutil.copytree(journal, dst)
    j = json.loads((dst / "journal.json").read_text(encoding="utf-8"))
    edit(j)
    (dst / "journal.json").write_text(json.dumps(j), encoding="utf-8")
    return dst


@pytest.mark.parametrize("bad", ["0.1", "10", "3", "0.25", "", " ", "0x", "0x1.8", "0x1.8p", "0xg", "1p3", "0x1p+99999", "hex",
                                 "1/2", "1e3", "inf1", True, Fraction(1, 3), 2**53 + 1, 10**400, [1.0], b"0x1p0"])
def test_z1_component_outside_the_contract_is_withheld_never_converted(bad):
    with pytest.raises(pl.MalformedValue):
        pl.parse_component(bad)
    good = {"P_p": 3.0, "P_s": 2.0, "Q": 1.0, "Gamma_up": 0.25, "B_E": 0.25}
    for k in good:
        r = pl.assemble({**good, k: bad}, OK)
        assert r["rejected"] and r["status"] == "UNRESOLVED_CERTIFICATE" and k in r["rejections"][0], (k, bad)


def test_z1_component_contract_accepted_forms():
    pc = pl.parse_component
    assert pc("0x1.8000000000000p+1") == 3.0 and pc("0x1.8p+1") == 3.0 and pc("0X1P-2") == 0.25 and pc(" 0x1p0 ") == 1.0
    z = pc("-0x0p+0")
    assert z == 0.0 and math.copysign(1.0, z) == -1.0 and pc("-0x0.0p+0").hex() == "-0x0.0p+0"
    assert pc("inf") == math.inf and pc("-inf") == -math.inf and math.isnan(pc("nan")) and math.isnan(pc(float("nan")))
    assert pc(3) == 3.0 and pc(2**53) == 2.0**53 and pc(2**1023) == 2.0**1023 and pc(0.1) == 0.1
    # the decimal text that the former float.fromhex read as hexadecimal digits: '0.1' -> 1/16, '10' -> 16
    assert float.fromhex("0.1") == 0.0625 and float.fromhex("10") == 16.0
    # not finite components: kept (not rejected) and handled by the classifier as faults of the certificate, never PASS
    for v in ("inf", "-inf", "nan", math.inf, math.nan):
        r = pl.assemble({"P_p": 3.0, "P_s": 2.0, "Q": 1.0, "Gamma_up": 0.25, "B_E": v}, OK)
        assert not r["rejected"] and r["status"] == "UNRESOLVED_CERTIFICATE" and r["value_faults"], v


def test_z1_recorded_value_contract():
    pr = pl.parse_recorded
    assert pr("0.1") == Fraction(1, 10) and pr("3/4") == Fraction(3, 4) and pr("hex:0x1.8p+1") == 3 and pr(2**60 + 1) == 2**60 + 1
    for bad in ("hex:0.1", "hex:10", "hex:", "nan", "inf", "", "1/0", "x", math.inf, math.nan, True, None, [1]):
        with pytest.raises(pl.MalformedValue):
            pr(bad)
    good = {"P_p": 3.0, "P_s": 2.0, "Q": 1.0, "Gamma_up": 0.25, "B_E": 0.25}
    for rec in ({"tau": "hex:0.8"}, {"dP": "nan"}, {"dP": "1/0"}, {"tau": True}):
        x = pl.assemble(good, OK, rec)
        assert x["rejected"] and x["status"] == "UNRESOLVED_CERTIFICATE", rec


def test_z2_fault_of_gamma_b1_is_recorded_without_an_assumed_premise(journal, monkeypatch):
    cfg, given, comp, V = _row_arrays(journal, 2)
    clean = pl.compute_row(cfg, 2, given, comp, V)
    assert clean["faults"] == [] and pl.assemble(clean["components"], OK, faults=clean["faults"])["status"] == "PASS"

    def broken(*a, **k):
        raise ValueError("synthetic failure of path B1")

    monkeypatch.setattr(pl.gi(), "gamma_b1", broken)
    row = pl.compute_row(cfg, 2, given, comp, V)
    assert row["Gamma_up"] is None and row["components"]["Gamma_up"] is None and "synthetic" in row["gamma_error"]
    assert [f["stage"] for f in row["faults"]] == ["gamma_B1"] and row["faults"][0]["premise"] is None
    r = pl.assemble(row["components"], OK, faults=row["faults"])
    assert r["status"] == "UNRESOLVED_CERTIFICATE" and r["tau_exact"] is None and r["source_undetermined"]
    assert r["reason_premises"] == []  # no premise of protocol 4.1 is attributed to the fault
    assert any("FAULT at stage gamma_B1" in x and "synthetic failure of path B1" in x for x in r["reasons"])
    assert r["faults"][0]["source"].startswith("Gamma_up")


def test_z2_fault_reaches_the_classification_of_analyse(journal, monkeypatch):
    def broken(*a, **k):
        raise ArithmeticError("synthetic failure of path B1")

    monkeypatch.setattr(pl.gi(), "gamma_b1", broken)
    r = pl.analyse({"kind": "journal", "journal": str(journal), "s": 2}, reproduce=False)  # in this process: the double is active
    assert r["status"] == "UNRESOLVED_CERTIFICATE" and [f["stage"] for f in r["faults"]] == ["gamma_B1"]
    assert r["classification"]["faults"] == r["faults"] and r["classification"]["reason_premises"] == ["P-a"]  # P-a: no reproduction
    assert any(x.startswith("FAULT at stage gamma_B1") for x in r["classification"]["reasons"])


def test_z2_fault_of_the_second_path_is_diagnostic_and_distinguished(journal, monkeypatch):
    cfg, given, comp, V = _row_arrays(journal, 2)

    def broken(*a, **k):
        raise ArithmeticError("synthetic failure of the second path")

    monkeypatch.setattr(pl.ga(), "energies_path_b", broken)
    row = pl.compute_row(cfg, 2, given, comp, V)
    assert [f["stage"] for f in row["faults"]] == ["gamma_second_path"] and row["Gamma_up"] is not None
    assert row["diagnostics"]["second_path_gamma"]["performed"] is False and "gamma_error" not in row
    r = pl.assemble(row["components"], OK, faults=row["faults"])
    assert r["status"] == "PASS"  # a diagnostic stage does not change the protocol status (OD-4), but it is listed
    assert any("FAULT at stage gamma_second_path" in x and "diagnostic only" in x for x in r["reasons"])


def test_z4_mode_change_between_the_probes_is_detected(journal, monkeypatch):
    """Survivor 1 of the review: the probe after the row. A mode that changes DURING the row (the probe before passes) is seen only
    by the probe after the row; the separate process does not see it when the change is deterministic. Portable: a probe double."""
    cfg, given, comp, V = _row_arrays(journal, 2)
    real, calls = pl.qa().probe_mode, []

    def probe():
        out = real()
        calls.append(out["ok"])
        if len(calls) == 2:
            out = {**out, "ok": False, "fails": ["synthetic: rounding mode changed during the row"]}
        return out

    monkeypatch.setattr(pl.qa(), "probe_mode", probe)
    row = pl.compute_row(cfg, 2, given, comp, V)
    assert calls == [True, True] and row["mode_before_ok"] is True and row["mode_after_ok"] is False
    assert row["premises"]["P-c"]["state"] == "NOT_ESTABLISHED" and "mode probe after the row" in row["premises"]["P-c"]["reason"]
    assert pl.assemble(row["components"], {**OK, "P-c": row["premises"]["P-c"]})["status"] == "UNRESOLVED_CERTIFICATE"


@pytest.mark.skipif(sys.platform != "win32", reason="the real FPU control word is set with _controlfp (Windows); NOT_VERIFIED elsewhere")
@pytest.mark.parametrize("mode", ["round_up", "ftz_daz"])
def test_z4_real_mode_change_during_the_row_is_detected(journal, monkeypatch, mode):
    import ctypes

    cfg, given, comp, V = _row_arrays(journal, 2)
    real = pl.gi().gamma_b1

    def switching(*a, **k):  # called between the two probes of compute_row; switches the mode of this thread
        out = real(*a, **k)
        pl._set_mode(mode)
        return out

    f = ctypes.CDLL("ucrtbase")._controlfp
    f.restype, f.argtypes = ctypes.c_uint, [ctypes.c_uint, ctypes.c_uint]
    monkeypatch.setattr(pl.gi(), "gamma_b1", switching)
    try:
        row = pl.compute_row(cfg, 2, given, comp, V)
    finally:
        f(0, 0x300 | 0x03000000)  # round to nearest, no flush of subnormals
    assert pl.qa().probe_mode()["ok"]
    assert row["mode_before_ok"] is True and row["mode_after_ok"] is False
    assert row["premises"]["P-c"]["state"] == "NOT_ESTABLISHED" and "mode probe after the row" in row["premises"]["P-c"]["reason"]


@pytest.mark.parametrize("key", ["field_s_sha", "prev_field_after_QR_sha"])
def test_z4_field_hash_of_the_record_is_checked_apart(journal, tmp_path, key):
    """Survivor 2 of the review: the arrays are bound by their hashes, but the field hash of the record is not part of that binding;
    a record whose field hash is not the field formed from its arrays (inconsistent journal) must break P-a."""
    d = _edited_journal(journal, tmp_path / "j", lambda j: j["steps"][1].__setitem__(key, "0" * 64))
    r = pl.analyse({"kind": "journal", "journal": str(d), "s": 2})
    assert r["reproduction"]["bitwise_equal"] is True  # the separate process does not see it
    assert r["premises"]["P-a"]["state"] == "NOT_ESTABLISHED" and key in r["premises"]["P-a"]["reason"]
    assert r["status"] == "UNRESOLVED_CERTIFICATE"


@pytest.mark.parametrize("mutation, sources, status", [("gamma_sign_flip", ["second_path_gamma"], None),
                                                       ("Q_from_field_s", ["theorem_E2"], "FAIL_COMPUTATION"),
                                                       ("P_prev_from_field_s", ["theorem_E2"], "FAIL_COMPUTATION")])
def test_z4_contradictions_are_registered_with_their_source(journal, mutation, sources, status):
    """Survivors 3 and 4 of the review: the diagnostic flag alone is not enough, the contradiction must be registered with its
    source (second path of gamma: G-1, G-2; theorem E-2: G-3, G-4)."""
    r = pl.analyse({"kind": "journal", "journal": str(journal), "s": 2, "mutation": mutation}, reproduce=False)
    got = r["contradictions"]
    assert got and [c["source"] for c in got] == sources and all(c["text"] for c in got)
    if "second_path_gamma" in sources:
        assert r["row"]["diagnostics"]["second_path_gamma"]["consistent"] is False
    if "theorem_E2" in sources:
        assert r["row"]["diagnostics"]["E_s_le_B_E"] is False
    if status:
        assert pl.assemble(r["row"]["components"], OK)["status"] == status


def test_z4_gamma_from_another_step_is_registered(journal):
    other = pl.analyse({"kind": "journal", "journal": str(journal), "s": 1}, reproduce=False)["row"]["gamma_exact"]
    r = pl.analyse({"kind": "journal", "journal": str(journal), "s": 2, "mutation": "gamma_from_step", "gamma_other": other},
                   reproduce=False)
    assert [c["source"] for c in r["contradictions"]] == ["second_path_gamma"]


def test_z5_portable_invariants_of_gamma_up_and_b_e_up(rows):
    """Portable (any platform, computed here, not the Windows D0 publication): Gamma_up = max(|gamma_down|, |gamma_up|) of the
    tightest directed interval of the exact gamma, |gamma| <= Gamma_up; B_E_up = the upward binary64 of the exact B_E."""
    inexact_gamma = inexact_be = 0
    for r in rows:
        x = r["row"]
        g, gd, gu = Fraction(x["gamma_exact"]), float.fromhex(x["gamma_down"]), float.fromhex(x["gamma_up"])
        G = float.fromhex(x["Gamma_up"])
        assert Fraction(gd) <= g <= Fraction(gu) and (gd == gu or math.nextafter(gd, math.inf) == gu)
        assert G == max(abs(gd), abs(gu)) and abs(g) <= Fraction(G)
        be, beu = Fraction(x["B_E_exact"]), float.fromhex(x["B_E_up"])
        assert Fraction(beu) >= be and (Fraction(beu) == be or Fraction(math.nextafter(beu, -math.inf)) < be)
        inexact_gamma += gd != gu
        inexact_be += Fraction(beu) != be
    assert inexact_gamma and inexact_be  # not vacuous: min instead of max, or rounding B_E down, would violate them here


def test_z6_c3plus_premises_carried_p4_assumed(rows, journal, tmp_path):
    for r in rows:
        c = r["c3plus"]
        assert c["c3plus_premises_status"] == pl.op8().PREMISES_OK and c["P-4"] == "ASSUMED"
        assert c["c3plus_premise_classes"]["P-4"] == "ASSUMED" and c["applicability"].startswith("CONDITIONAL_ON_P4")
        assert "not proven" in c["applicability"] and r["scope"]["model"] == "NOT_PROVEN by the row"
        assert "computational consistency" in r["scope"]["status"] and r["scope"]["bc_accuracy"].startswith("NOT_ESTABLISHED")
        back = json.loads(json.dumps(r, default=pl._json_default))
        assert back["c3plus"]["c3plus_premises_status"] == pl.op8().PREMISES_OK and back["c3plus"]["P-4"] == "ASSUMED"
    # a record raising P-4, or premises not ESTABLISHED_CONDITIONAL_ON_P4 on an executed half-step: not the OP-8 record, P-a broken
    for name, edit in (("p4", lambda j: j["steps"][1]["c3plus_premises"]["classes"].__setitem__("P-4", "ESTABLISHED")),
                       ("st", lambda j: j["steps"][1].__setitem__("c3plus_premises_status", "UNRESOLVED"))):
        d = _edited_journal(journal, tmp_path / name, edit)
        r = pl.analyse({"kind": "journal", "journal": str(d), "s": 2})
        assert r["premises"]["P-a"]["state"] == "NOT_ESTABLISHED" and "C-3+ premise" in r["premises"]["P-a"]["reason"]
        assert r["status"] == "UNRESOLVED_CERTIFICATE" and r["c3plus"]["P-4"] == "ASSUMED"
        assert r["c3plus"]["applicability"].startswith("NOT_ESTABLISHED")
    # without the OP-8 re-run (rerun_op8 False) the record check is the only one: it must see a raised P-4 by itself
    j, _ = pl.load_journal(journal)
    rec = json.loads(json.dumps(j["steps"][1]))
    assert pl.c3plus_status(rec, None)[1] == []
    rec["c3plus_premises"]["classes"]["P-4"] = "ESTABLISHED"
    c, bad = pl.c3plus_status(rec, None)
    assert c["P-4"] == "ASSUMED" and any("P-4 recorded as 'ESTABLISHED'" in x for x in bad)
    d = _edited_journal(journal, tmp_path / "p4_no_rerun",
                        lambda j: j["steps"][1]["c3plus_premises"]["classes"].__setitem__("P-4", "ESTABLISHED"))
    r = pl.analyse({"kind": "journal", "journal": str(d), "s": 2, "rerun_op8": False}, reproduce=False)
    assert "P-4 recorded as 'ESTABLISHED'" in r["premises"]["P-a"]["reason"] and r["status"] == "UNRESOLVED_CERTIFICATE"
    ne = pl.not_executed({"status": "UNRESOLVED_C3PLUS_PREMISE: P-3", "c3plus_premises_status": "UNRESOLVED",
                          "c3plus_premises": {"classes": {"P-1": "ESTABLISHED", "P-3": "UNRESOLVED", "P-4": "ASSUMED"}}}, 2)
    assert ne["status"] == "UNRESOLVED" and ne["c3plus"]["P-4"] == "ASSUMED" and ne["c3plus"]["applicability"].startswith("NOT_APPL")


# --- I-2: the guard before any binding, loading, re-run or child process -----------------------------------------------------------

@pytest.fixture
def d1_journal(journal, tmp_path):
    """A copy of the T1 journal relabelled as the D1 grid (9, 11): without the guard the half-step would be bound and re-run."""
    assert (9, 11) in pl.op8().D1_GRIDS
    return _edited_journal(journal, tmp_path / "d1", lambda j: j["meta"]["config"].update({"nx": 9, "ny": 11}))


@pytest.fixture
def sensors(monkeypatch):
    calls = []

    def sensor(name):
        def f(*a, **k):
            calls.append(name)
            raise AssertionError(f"{name} called before the guard")
        return f

    o = pl.op8()
    for name in ("run_step", "run_trajectory", "load_arrays", "write_journal"):
        monkeypatch.setattr(o, name, sensor(name))
    monkeypatch.setattr(pl, "run_child", sensor("run_child"))
    monkeypatch.setattr(pl, "compute_row", sensor("compute_row"))
    return calls


def test_i2_guard_precedes_every_half_step_on_a_d1_grid(d1_journal, tmp_path, sensors):
    spec = {"kind": "journal", "journal": str(d1_journal), "s": 1}
    j = json.loads((d1_journal / "journal.json").read_text(encoding="utf-8"))
    TC = pl.op8().TrajectoryConfig
    arrays_cfg = {"lx": 1.0, "nx": 50, "ly": 1.0, "ny": 50, "N": 1, "D": 1.0, "q": 1.0, "s_max": 1}
    calls = [lambda: pl.analyse(spec), lambda: pl.analyse({**spec, "mode": "round_up"}), lambda: pl.analyse(spec, reproduce=False),
             lambda: pl.analyse_journal(d1_journal), lambda: pl._row_job(spec), lambda: pl._inputs_from_spec(spec),
             lambda: pl.bind_journal(j, {}),
             lambda: pl.run_trajectory_journal(TC(1.0, 9, 1.0, 11, 1, s_max=1), tmp_path / "o"),
             lambda: pl.run_trajectory_journal(TC(1.0, 8, 1.25, 10, 2, s_max=1), tmp_path / "o"),
             lambda: pl.analyse({"kind": "arrays", "arrays_npz": str(tmp_path / "absent.npz"), "arrays_sha256": {}, "s": 1,
                                 "config": arrays_cfg}),
             lambda: pl.analyse({"kind": "arrays", "arrays_npz": str(tmp_path / "absent.npz"), "arrays_sha256": {}, "s": 1,
                                 "config": {**arrays_cfg, "nx": 8, "ny": 10, "N": 3}})]
    for i, call in enumerate(calls):
        with pytest.raises(pl.D1Refused):
            call()
        assert sensors == [], (i, sensors)


def test_i2_guard_of_the_output_paths(sensors):
    TC = pl.op8().TrajectoryConfig
    for out in (ROOT / "docs" / "x", ROOT / "docs", ROOT / "docs" / "results" / "r5" / "y"):
        with pytest.raises(pl.D1Refused):
            pl.run_trajectory_journal(TC(1.0, 8, 1.25, 10, 1, s_max=1), out)
        with pytest.raises(pl.D1Refused):
            pl.write_control_arrays(out / "a.npz", np.zeros((7, 1)), np.zeros((9, 1)), np.zeros((9, 1)))
    assert sensors == [] and not (ROOT / "docs" / "x").exists()


def test_i2_guard_of_the_internal_entry_and_the_command_line(d1_journal, tmp_path):
    (d1_journal / "arrays.npz").unlink()  # without the guard the entry would fail on the missing arrays (rc 1), not refuse (rc 2)
    sp = tmp_path / "spec.json"
    sp.write_text(json.dumps({"kind": "journal", "journal": str(d1_journal), "s": 1}), encoding="utf-8")
    p = subprocess.run([sys.executable, str(TOOL), "_row", str(sp)], cwd=ROOT, capture_output=True, text=True)
    assert p.returncode == 2 and "refused" in p.stderr and "D1" in p.stderr
    with pytest.raises(pl.D1Refused):
        pl.run_child({"kind": "journal", "journal": str(d1_journal), "s": 1})


# --- integration gate review (N-2...N-5): integrity of the journal, the sequence and the certificate components --------------------

def _rewrite(journal, dst, edit=None, arrays_edit=None, rehash=False):
    """A copy of the journal with edits of journal.json and of arrays.npz. ``rehash``: the meta and record hashes are rewritten to the
    edited arrays (a consistent rewrite: hashes alone no longer see it)."""
    shutil.copytree(journal, dst)
    j = json.loads((dst / "journal.json").read_text(encoding="utf-8"))
    if arrays_edit is not None:
        arr = pl.op8().load_arrays(dst)
        arrays_edit(arr)
        np.savez(dst / "arrays.npz", **arr)
        if rehash:
            sha = pl.op8().array_sha
            j["meta"]["arrays_sha256"] = {k: sha(v) for k, v in arr.items()}
            o = pl.op8()
            for rec in j["steps"]:
                s = rec["s"]
                for name, key in pl.STEP_ARRAYS:
                    if f"s{s}_{name}" in arr and key in rec:
                        rec[key] = sha(arr[f"s{s}_{name}"])
                g, c, V = arr[f"s{s}_solved_set"], arr[f"s{s}_companion_solved"], arr[f"s{s}_V_bc"]
                rec["field_s_sha"] = sha(o._field(o.direction_of(s), g, V))
                rec["prev_field_after_QR_sha"] = sha(o._field(o.direction_of(s), g, c))
    if edit is not None:
        edit(j)
    (dst / "journal.json").write_text(json.dumps(j), encoding="utf-8")
    return dst


def _ulp(arr, key):
    a = arr[key].copy()
    a.flat[0] = math.nextafter(float(a.flat[0]), math.inf)
    arr[key] = a


def _next_input_rewrite(arr):
    """The next input of s = 1 changed by 1 ulp and carried into s = 2 as OP-8 would (N = 1: solved set = given set, no QR)."""
    _ulp(arr, "s1_next_given")
    arr["s2_given_in"] = arr["s1_next_given"].copy()
    arr["s2_solved_set"] = arr["s1_next_given"].copy()


def _compute(journal, s, **kw):
    cfg, given, comp, V = _row_arrays(journal, s)
    return pl.compute_row(cfg, s, given, comp, V, **kw)


@pytest.fixture
def inproc_child(monkeypatch):
    """The separate process replaced by a fresh in-process computation of the same specification (same serialisation): the
    reproduction stays a real recomputation; used where the subject is the journal, not the process boundary (speed)."""
    def child(spec, timeout=1800):
        pl.guard_config(pl.spec_config(spec))
        return json.loads(json.dumps(pl._row_job(spec), default=pl._json_default))

    monkeypatch.setattr(pl, "run_child", child)


# N-2: canonical C-3+ record, with and without the OP-8 re-run ----------------------------------------------------------------------

def test_n2_the_op8_record_is_canonical(journal):
    j, _ = pl.load_journal(journal)
    for rec in j["steps"]:
        assert set(rec["c3plus_premises"]) == pl.C3PLUS_RECORD_KEYS and rec["c3plus_premises"]["classes"] == pl.C3PLUS_CLASSES
        assert pl.c3plus_record_defects(rec) == []
        c, bad = pl.c3plus_status(rec, None)
        assert bad == [] and c["P-4"] == c["P-4_recorded"] == "ASSUMED" and c["applicability"].startswith("CONDITIONAL_ON_P4")


def _classes(j, f):
    f(j["steps"][1]["c3plus_premises"]["classes"])


N2_EDITS = {
    "P-4 PROVEN": (lambda j: _classes(j, lambda c: c.__setitem__("P-4", "PROVEN")), "P-4 recorded as 'PROVEN'"),
    "P-1 UNRESOLVED": (lambda j: _classes(j, lambda c: c.__setitem__("P-1", "UNRESOLVED")), "P-1 recorded as 'UNRESOLVED'"),
    "P-2 missing": (lambda j: _classes(j, lambda c: c.pop("P-2")), "are not P-1...P-5"),
    "foreign class key": (lambda j: _classes(j, lambda c: c.__setitem__("P-6", "ESTABLISHED")), "are not P-1...P-5"),
    "foreign record key": (lambda j: j["steps"][1]["c3plus_premises"].__setitem__("P-4_proof", "x"), "foreign keys ['P-4_proof']"),
    "record key missing": (lambda j: j["steps"][1]["c3plus_premises"].pop("k"), "missing keys ['k']"),
    "status UNRESOLVED": (lambda j: j["steps"][1].__setitem__("c3plus_premises_status", "UNRESOLVED"), "status 'UNRESOLVED'"),
    "status ESTABLISHED": (lambda j: j["steps"][1].__setitem__("c3plus_premises_status", "ESTABLISHED"), "status 'ESTABLISHED'"),
    "summary": (lambda j: j["steps"][1].__setitem__("c3plus_premises_summary", "P-1...P-5 established"), "summary"),
    "text contradicts class": (lambda j: j["steps"][1]["c3plus_premises"].__setitem__("P-3", "UNRESOLVED: P-3: subnormal"),
                               "text of P-3"),
    "decomposition not seen": (lambda j: j["steps"][1]["c3plus_premises"].__setitem__("decomposition_C_eq_fl_CL_plus_E", False),
                               "decomposition"),
    "k": (lambda j: j["steps"][1]["c3plus_premises"].__setitem__("k", 3), "k = 3"),
    "C-3+ check FAIL": (lambda j: j["steps"][1].__setitem__("c3plus_bound_check_exact", "FAIL"), "exact C-3+ check"),
}


@pytest.mark.parametrize("name", sorted(N2_EDITS))
def test_n2_record_check_alone_without_the_rerun(journal, name):
    """The record check alone (no OP-8 re-run): every non-canonical record is a П-а defect with its own reason."""
    j, _ = pl.load_journal(journal)
    j = json.loads(json.dumps(j))
    edit, text = N2_EDITS[name]
    edit(j)
    c, bad = pl.c3plus_status(j["steps"][1], None)
    assert any(text in x for x in bad), (name, bad)
    assert c["P-4"] == "ASSUMED" and c["applicability"].startswith("NOT_ESTABLISHED")  # never raised, never established


@pytest.mark.parametrize("name", ["P-4 PROVEN", "P-1 UNRESOLVED", "P-2 missing", "foreign class key", "status UNRESOLVED"])
@pytest.mark.parametrize("rerun", [True, False])
def test_n2_corrupted_record_never_passes(journal, tmp_path, name, rerun):
    edit, text = N2_EDITS[name]
    d = _rewrite(journal, tmp_path / "j", edit)
    r = pl.analyse({"kind": "journal", "journal": str(d), "s": 2, "rerun_op8": rerun})
    assert r["status"] == "UNRESOLVED_CERTIFICATE" and r["classification"]["reason_premises"] == ["P-a"]
    assert r["premises"]["P-a"]["state"] == "NOT_ESTABLISHED" and text in r["premises"]["P-a"]["reason"]
    assert (r["op8_rerun"] is not None) is rerun and r["c3plus"]["P-4"] == "ASSUMED"
    if name == "P-4 PROVEN":
        assert r["c3plus"]["P-4_recorded"] == "PROVEN"  # the recorded value is kept apart, never adopted
    # the corrupted record of s = 2 also breaks s = 3 (predecessor), never s = 1
    r3 = pl.analyse({"kind": "journal", "journal": str(d), "s": 3, "rerun_op8": rerun}, reproduce=False)
    assert "predecessor s = 2: C-3+ premise record" in r3["premises"]["P-a"]["reason"]
    r1 = pl.analyse({"kind": "journal", "journal": str(d), "s": 1}, reproduce=False)
    assert "C-3+" not in r1["premises"]["P-a"]["reason"]


@pytest.mark.parametrize("field", ["classes", "k"])
def test_n2_recorded_and_reproduced_classes_differ(journal, monkeypatch, field):
    """A canonical record and an OP-8 re-run classifying differently (classes), or differing only in another field of the C-3+
    record (k): seen only by the comparison with the re-run (the record check alone passes). Without the re-run (rerun_op8 False)
    there is nothing to compare: П-а NOT_CHECKABLE, never PASS."""
    real = pl.op8().run_step

    def other(*a, **k):
        res = real(*a, **k)
        res.record["c3plus_premises"] = json.loads(json.dumps(res.record["c3plus_premises"]))
        if field == "classes":
            res.record["c3plus_premises"]["classes"]["P-3"] = "UNRESOLVED"
        else:
            res.record["c3plus_premises"]["k"] += 1
        return res

    monkeypatch.setattr(pl.op8(), "run_step", other)
    r = pl.analyse({"kind": "journal", "journal": str(journal), "s": 1}, reproduce=False)
    reason = r["premises"]["P-a"]["reason"]
    assert "C-3+ premises of the OP-8 re-run differ from the record" in reason and "C-3+ premise record:" not in reason
    assert r["status"] == "UNRESOLVED_CERTIFICATE" and r["c3plus"]["applicability"].startswith("NOT_ESTABLISHED")
    monkeypatch.setattr(pl.op8(), "run_step", real)
    r = pl.analyse({"kind": "journal", "journal": str(journal), "s": 1, "rerun_op8": False})
    assert r["premises"]["P-a"]["state"] == "NOT_CHECKABLE" and "rerun_op8 = False" in r["premises"]["P-a"]["reason"]
    assert r["status"] == "UNRESOLVED_CERTIFICATE" and r["op8_rerun"] is None and r["reproduction"]["bitwise_equal"] is True


# N-3: the sequence ---------------------------------------------------------------------------------------------------------------

def _stop_s2(j):
    j["steps"][1]["status"] = "STOP_ZERO_FIELD: synthetic stop"


def _drop_s2(j):
    del j["steps"][1]
    j["meta"]["half_steps"] = 2


def _swap(j):
    j["steps"][1], j["steps"][2] = j["steps"][2], j["steps"][1]


def _stopped_end_completed(j):
    del j["steps"][2]
    j["steps"][1]["status"] = "PROTECTIVE_STOP: synthetic"
    j["meta"]["half_steps"] = 2  # the meta status still says COMPLETED


def _dup(j):
    j["steps"].insert(2, json.loads(json.dumps(j["steps"][1])))
    j["meta"]["half_steps"] = 4


CERT, UNR = "UNRESOLVED_CERTIFICATE", "UNRESOLVED"
N3_JOURNALS = {  # edit, arrays edit, rehash, expected statuses of s = 1, 2, 3, text in the reasons of s = 3
    "stopped s = 2, record s = 3 kept": (_stop_s2, None, False, (CERT, UNR, UNR), "record after the stop"),
    "s = 2 absent (gap)": (_drop_s2, None, False, (CERT, UNR, UNR), "out of the order"),
    "order s = 1, 3, 2": (_swap, None, False, (CERT, UNR, UNR), "out of the order"),
    "duplicate s = 2": (_dup, None, False, (CERT, CERT, UNR), "after the break of the order"),
    "direction of s = 2": (lambda j: j["steps"][1].__setitem__("direction", "y"), None, False, (CERT,) * 3, "direction"),
    "meta stop without a stopped record": (lambda j: j["meta"].__setitem__("status", "STOPPED at s = 2: x"), None, False,
                                           (CERT,) * 3, "meta status"),
    "meta half_steps": (lambda j: j["meta"].__setitem__("half_steps", 2), None, False, (CERT,) * 3, "meta half_steps"),
    "meta COMPLETED with a stopped last record": (_stopped_end_completed, None, False, (CERT, UNR, UNR), "absent after the stop"),
    "record chain s = 2": (lambda j: j["steps"][1].__setitem__("given_in_sha", "0" * 64), None, False, ("PASS", CERT, CERT),
                           "record chain"),
    "s = 1 V_bc 1 ulp (hash)": (None, lambda a: _ulp(a, "s1_V_bc"), False, (CERT,) * 3, "predecessor s = 1"),
    "meta hash of s = 2 V_bc": (lambda j: j["meta"]["arrays_sha256"].__setitem__("s2_V_bc", "0" * 64), None, False,
                                ("PASS", CERT, CERT), "predecessor s = 2: s2_V_bc: sha256 differs from the meta"),
    "s = 1 C-3+ record": (lambda j: j["steps"][0]["c3plus_premises"]["classes"].__setitem__("P-4", "PROVEN"), None, False,
                          (CERT,) * 3, "predecessor s = 1: C-3+ premise record"),
    "s = 1 field hash of the record": (lambda j: j["steps"][0].__setitem__("field_s_sha", "0" * 64), None, False, (CERT,) * 3,
                                       "predecessor s = 1: field_s_sha"),
    "s = 1 next input consistently rewritten": (None, _next_input_rewrite, True, (CERT,) * 3,
                                                "predecessor s = 1: next_given is not bitwise the OP-8 re-run"),
}


@pytest.mark.parametrize("name", sorted(N3_JOURNALS))
def test_n3_a_row_is_never_pass_on_a_broken_sequence(journal, tmp_path, inproc_child, name):
    edit, aedit, rehash, want, text = N3_JOURNALS[name]
    d = _rewrite(journal, tmp_path / "j", edit, aedit, rehash)
    rows = pl.analyse_journal(d)
    assert tuple(r["status"] for r in rows) == want, name
    r3 = rows[2]
    reasons = r3["premises"]["P-a"]["reason"] if r3["status"] == CERT else r3["classification"]["reasons"][0]
    assert text in reasons, (name, reasons)
    for r in rows:  # not executed -> UNRESOLVED, never computed; a binding fault -> П-а; never FAIL_COMPUTATION
        assert r["status"] != "FAIL_COMPUTATION"
        if r["status"] == UNR:
            assert r.get("row") is None and r["classification"]["reason_premises"] == [] and r["not_executed_reason"]
        if r["status"] == CERT:
            assert r["classification"]["reason_premises"] == ["P-a"] and r["premises"]["P-a"]["state"] == "NOT_ESTABLISHED"


def test_n3_predecessor_hash_binding_alone(journal, tmp_path):
    """Without the OP-8 re-run the hash binding of the predecessor is the only channel for its corrupted array."""
    d = _rewrite(journal, tmp_path / "j", None, lambda a: _ulp(a, "s1_V_bc"))
    r = pl.analyse({"kind": "journal", "journal": str(d), "s": 2, "rerun_op8": False})
    assert r["premises"]["P-a"]["state"] == "NOT_ESTABLISHED" and r["status"] == CERT
    assert "predecessor s = 1: s1_V_bc: sha256 differs from the meta" in r["premises"]["P-a"]["reason"]


def test_n3_standalone_row_after_a_stop_is_not_computed(journal, tmp_path, monkeypatch):
    d = _rewrite(journal, tmp_path / "j", _stop_s2)

    def no(*a, **k):
        raise AssertionError("a row after the stop must not be computed")

    monkeypatch.setattr(pl, "compute_row", no)
    monkeypatch.setattr(pl, "run_child", no)
    r = pl.analyse({"kind": "journal", "journal": str(d), "s": 3})
    assert r["status"] == UNR and "record after the stop of the trajectory at s = 2" in r["classification"]["reasons"][0]
    assert "STOP_ZERO_FIELD: synthetic stop" in r["not_executed_reason"]
    r = pl.analyse({"kind": "journal", "journal": str(d), "s": 2})
    assert r["status"] == UNR and r["not_executed_reason"] == "OP-8 half-step s = 2: STOP_ZERO_FIELD: synthetic stop"


def test_n3_genuine_op8_stop_keeps_the_reason(tmp_path):
    """A real stop of OP-8 (q = 0: the computed factor is exactly 0 at s = 1, STOP_ZERO_FIELD): a consistent journal; the stopped row
    and the absent rows after it are not executed (UNRESOLVED), with the stop reason; nothing is attributed to a premise."""
    cfg = pl.op8().TrajectoryConfig(1.0, 8, 1.25, 10, 1, 1.0, 0.0, 3)
    jd = pl.run_trajectory_journal(cfg, tmp_path / "q0")
    j, _ = pl.load_journal(jd)
    st = pl.journal_structure(j)
    assert st["reasons"] == [] and st["executed"] == [] and st["stop"]["s"] == 1
    rows = pl.analyse_journal(jd)
    assert [r["status"] for r in rows] == [UNR] * 3
    assert rows[0]["not_executed_reason"].startswith("OP-8 half-step s = 1: STOP_ZERO_FIELD")
    assert all("absent after the stop of the trajectory at s = 1 (STOP_ZERO_FIELD" in r["not_executed_reason"] for r in rows[1:])
    assert rows[0]["c3plus"]["P-4"] == "ASSUMED" and rows[1]["c3plus"] is None  # P-4 as recorded by OP-8 at s = 1
    back = json.loads(json.dumps(rows, default=pl._json_default))
    assert back[0]["not_executed_reason"] == rows[0]["not_executed_reason"] and back[0]["record_status"].startswith("STOP_ZERO")


def test_n3_correct_journals_keep_their_results(journal, rows):
    """The new checks leave a correct journal as it was: the structure has no remark, the rows PASS with П-а established."""
    j, arrays = pl.load_journal(journal)
    assert pl.journal_structure(j) == {"executed": [1, 2, 3], "not_executed": {}, "stop": None, "reasons": []}
    for s in (1, 2, 3):
        assert pl.sequence_binding(j, arrays, s) == []
    assert [r["status"] for r in rows] == ["PASS"] * 3
    assert all(r["premises"]["P-a"]["state"] == "ESTABLISHED" and r["op8_rerun"]["reproduced"] for r in rows)
    assert all(all(r["op8_rerun"]["arrays_equal"].values()) for r in rows)
    assert all(r["row"]["gamma_contract"] == {"holds": True, "violations": []} for r in rows)
    assert all(r["row"]["diagnostics"]["gamma_interval_tightness"]["tight"] is True for r in rows)


def test_n3_consistent_rewrite_seen_by_the_rerun_or_by_external_hashes(journal, tmp_path, inproc_child):
    """A rewrite of the next input of s = 1 consistent in arrays, meta and records: the hashes do not see it; the OP-8 re-run of the
    predecessor does; without the re-run П-а is NOT_CHECKABLE (never PASS); a hash fixed outside the journal sees it too."""
    clean2 = pl.analyse({"kind": "journal", "journal": str(journal), "s": 2}, reproduce=False)["row"]
    d = _rewrite(journal, tmp_path / "j", None, _next_input_rewrite, rehash=True)
    j, arrays = pl.load_journal(d)
    assert all(v["ok"] for v in pl.bind_journal(j, arrays).values()) and pl.journal_structure(j)["reasons"] == []
    r = pl.analyse({"kind": "journal", "journal": str(d), "s": 2})
    assert "predecessor s = 1: next_given is not bitwise the OP-8 re-run" in r["premises"]["P-a"]["reason"]
    r = pl.analyse({"kind": "journal", "journal": str(d), "s": 2, "rerun_op8": False})
    assert r["premises"]["P-a"]["state"] == "NOT_CHECKABLE" and r["status"] == CERT
    r = pl.analyse({"kind": "journal", "journal": str(d), "s": 2}, bound_values={"field_prev_sha": clean2["field_prev_sha"]})
    assert "bound value field_prev_sha" in r["premises"]["P-a"]["reason"]


# N-4: the annotation of a row that is not executed -----------------------------------------------------------------------------------

NO_RECORD = "NOT_RECORDED (no C-3+ premise record in the record of the half-step)"


@pytest.mark.parametrize("rec, p4", [
    ({"status": "STOP_BEFORE_QR: S-1"}, NO_RECORD),
    ({"status": "UNRESOLVED_C3PLUS_PREMISE: P-3", "c3plus_premises": {"P-1": "x"}}, NO_RECORD),
    ({"status": "UNRESOLVED_C3PLUS_PREMISE: P-3", "c3plus_premises": {"classes": {"P-1": "ESTABLISHED", "P-3": "UNRESOLVED"}}},
     "NOT_RECORDED (C-3+ premise record without P-4)"),
    ({"status": "UNRESOLVED_C3PLUS_PREMISE: P-3", "c3plus_premises": {"classes": {"P-3": "UNRESOLVED", "P-4": "PROVEN"}}},
     "INVALID_RECORD (P-4 recorded as 'PROVEN'; OP-8 records ASSUMED only)"),
    ({"status": "UNRESOLVED_C3PLUS_PREMISE: P-3", "c3plus_premises_status": "UNRESOLVED",
      "c3plus_premises": {"classes": {"P-3": "UNRESOLVED", "P-4": "ASSUMED"}}}, "ASSUMED"),
])
def test_n4_not_executed_row_states_only_what_its_record_says(rec, p4):
    r = pl.not_executed(rec, 2)
    assert r["status"] == UNR and r["c3plus"]["P-4"] == p4
    assert r["c3plus"]["c3plus_premises_status"] == rec.get("c3plus_premises_status", "NOT_RECORDED")
    assert r["not_executed_reason"] == f"OP-8 half-step s = 2: {rec['status']}" and r["record_status"] == rec["status"]
    assert r["classification"]["reasons"][0].startswith(r["not_executed_reason"])
    back = json.loads(json.dumps(r, default=pl._json_default))
    assert back["c3plus"] == r["c3plus"] and back["not_executed_reason"] == r["not_executed_reason"]
    assert pl.not_executed(None, 4)["c3plus"] is None


# N-5: gamma, Gamma_up and their interval (accepted contract vs proposed narrowness) -----------------------------------------------

def test_n5_gamma_contract_and_narrowness_units():
    g = Fraction(1, 3)
    gd, gu = pl.gi().directed(g)
    G = max(abs(gd), abs(gu))
    assert pl.gamma_contract(g, gd, gu, G) == [] and pl.gamma_interval_tight(g, gd, gu)
    assert any("gamma_down" in x for x in pl.gamma_contract(g, gu, gu, G))  # gamma_down alone corrupted: above gamma
    assert any("gamma_up =" in x for x in pl.gamma_contract(g, gd, gd, G))  # gamma_up alone corrupted: below gamma
    assert any("Gamma_up" in x for x in pl.gamma_contract(g, gd, gu, min(abs(gd), abs(gu))))
    assert any("Gamma_up" in x for x in pl.gamma_contract(g, gd, gu, 0.5))  # Gamma_up of another half-step
    assert pl.gamma_contract(g, 0.0, 1.0, 1.0) == [] and not pl.gamma_interval_tight(g, 0.0, 1.0)  # widened: contract holds
    assert pl.gamma_contract(-g, -1.0, 0.0, 1.0) == [] and pl.gamma_contract(Fraction(0), -0.0, 0.0, 0.0) == []
    assert pl.gamma_interval_tight(Fraction(1, 4), 0.25, 0.25) and not pl.gamma_interval_tight(Fraction(1, 4), 0.25, 0.5)
    assert pl.gamma_contract(g, math.nan, gu, G) and pl.gamma_contract(g, gd, gu, Fraction(1))
    assert pl.gamma_contract(g, -math.inf, math.inf, math.inf) == []  # not finite: the classifier's fault of Gamma_up, not this


def _corrupt_interval(monkeypatch, how, call=0):
    """``directed`` corrupted on one call of compute_row only: call 0 is the interval of gamma, call 1 the rounding of B_E."""
    real, calls = pl.gi().directed, []

    def directed(x):
        calls.append(x)
        return how(*real(x)) if len(calls) - 1 == call else real(x)

    monkeypatch.setattr(pl.gi(), "directed", directed)


@pytest.mark.parametrize("name, how", [
    ("gamma_down alone above gamma", lambda gd, gu: (gu, gu) if gd != gu else (math.nextafter(gu, math.inf), gu)),
    ("gamma_up alone below gamma", lambda gd, gu: (gd, gd) if gd != gu else (gd, math.nextafter(gd, -math.inf))),
    ("interval not containing gamma", lambda gd, gu: (2 * abs(gu) + 1, 4 * abs(gu) + 2)),
])
def test_n5_corrupted_interval_withholds_gamma_up(journal, monkeypatch, name, how):
    clean = _compute(journal, 2)
    assert clean["gamma_contract"]["holds"]
    _corrupt_interval(monkeypatch, how)
    row = _compute(journal, 2)
    assert row["gamma_contract"]["holds"] is False and row["Gamma_up"] is None and row["components"]["Gamma_up"] is None
    assert row["Gamma_up_withheld"] is not None and [f["stage"] for f in row["faults"]] == ["gamma_contract"]
    # the second path (gamma_B = gamma) sees gamma_B outside the corrupted interval: the inclusion of its diagnostic is not redundant
    assert [c["source"] for c in row["contradictions"]] == ["second_path_gamma"]
    r = pl.assemble(row["components"], OK, faults=row["faults"])
    assert r["status"] == CERT and r["tau_exact"] is None and r["reason_premises"] == []
    assert r["source_undetermined"] and any("FAULT at stage gamma_contract" in x for x in r["reasons"])
    # every other value of the row is the clean one: only the certificate component is withheld
    assert all(row[k] == clean[k] for k in ("P_hat_p", "P_hat_s", "Q_hat", "B_E_up", "gamma_exact"))


def test_n5_b_e_up_rounded_down_is_withheld(journal, monkeypatch):
    """The other certificate component: B_E_up below the exact B_E (rounding down) is not an upper bound; withheld as a fault of B_E_up,
    no premise attributed. The portable invariant of Z-5 checks the same on clean rows; this is the detector on the row."""
    clean = _compute(journal, 2)
    _corrupt_interval(monkeypatch, lambda d, u: (d, d), call=1)
    row = _compute(journal, 2)
    assert Fraction(float.fromhex(row["B_E_up_withheld"])) < Fraction(clean["B_E_exact"]) and row["B_E_up"] is None
    assert [f["stage"] for f in row["faults"]] == ["B_E_up"] and row["gamma_contract"]["holds"]
    r = pl.assemble(row["components"], OK, faults=row["faults"])
    assert r["status"] == CERT and r["tau_exact"] is None and r["reason_premises"] == []


def test_n5_gamma_up_of_another_half_step(journal, monkeypatch):
    other = float.fromhex(_compute(journal, 1)["Gamma_up"])
    monkeypatch.setattr(pl.gi(), "gamma_up", lambda d, u: other)
    row = _compute(journal, 2)
    assert row["Gamma_up"] is None and "is not max(|gamma_down|, |gamma_up|)" in row["faults"][0]["error"]
    assert pl.assemble(row["components"], OK, faults=row["faults"])["status"] == CERT


def test_n5_gamma_down_of_another_half_step(journal, monkeypatch):
    """gamma_down of s = 1 with the gamma of s = 2: excluded from the contract when it lies above gamma_2; an interval that still
    encloses gamma_2 is a widening (contract holds; narrowness, not normative, records it)."""
    gd1 = float.fromhex(_compute(journal, 1)["gamma_down"])
    g2 = Fraction(_compute(journal, 2)["gamma_exact"])
    for down in (gd1, -abs(gd1) - 1.0, abs(float(g2)) + abs(gd1) + 1.0):
        with monkeypatch.context() as m:
            _corrupt_interval(m, lambda d, u, down=down: (down, u))
            row = _compute(journal, 2)
        if Fraction(down) > g2:
            assert row["gamma_contract"]["holds"] is False and row["Gamma_up"] is None
            assert pl.assemble(row["components"], OK, faults=row["faults"])["status"] == CERT
        else:
            assert row["gamma_contract"]["holds"] and row["diagnostics"]["gamma_interval_tightness"]["tight"] is False


def test_n5_widened_interval_is_a_diagnostic_limitation_not_a_fail(journal, monkeypatch):
    """The accepted contract (enclosure, max) holds for a widened interval: by the protocol it is no ground for FAIL or UNRESOLVED;
    the narrowness RD/RU (a proposal pending OD-4) only records it."""
    clean = _compute(journal, 2)
    tight = clean["diagnostics"]["gamma_interval_tightness"]
    assert tight["tight"] is True and tight["normative"] is False
    _corrupt_interval(monkeypatch, lambda gd, gu: (gd - abs(gd), gu + abs(gu)))
    row = _compute(journal, 2)
    assert row["gamma_contract"]["holds"] and row["faults"] == [] and row["contradictions"] == []
    assert row["diagnostics"]["gamma_interval_tightness"]["tight"] is False
    assert Fraction(float.fromhex(row["Gamma_up"])) == 2 * Fraction(float.fromhex(clean["Gamma_up"]))
    r = pl.assemble(row["components"], OK, faults=row["faults"])
    assert r["status"] == "PASS" and Fraction(r["tau_exact"]) > Fraction(pl.assemble(clean["components"], OK)["tau_exact"])


def test_n5_contradictory_gamma_hashes_break_p_a(journal, monkeypatch):
    """A correct gamma with a contradictory hash: seen by the reproduction, by the self-consistency of the row and by a hash fixed
    outside (bound value)."""
    real = pl.run_child

    def tampered(spec, timeout=1800):
        job = real(spec, timeout)
        job["row"]["gamma_exact_sha256"] = "0" * 64
        return job

    monkeypatch.setattr(pl, "run_child", tampered)
    r = pl.analyse({"kind": "journal", "journal": str(journal), "s": 2})
    assert "separate process: values differ: gamma_exact_sha256" in r["premises"]["P-a"]["reason"]
    assert pl.row_self_consistency(r["row"]) == []
    row = dict(r["row"], gamma_exact_sha256="0" * 64)
    assert pl.row_self_consistency(row) == ["row not self-consistent: gamma_exact_sha256 is not the sha256 of gamma_exact"]
    row = dict(r["row"], components={**r["row"]["components"], "Gamma_up": 1.0})
    assert pl.row_self_consistency(row) == ["row not self-consistent: component Gamma_up is not the recorded Gamma_up"]
    monkeypatch.setattr(pl, "run_child", real)
    r = pl.analyse({"kind": "journal", "journal": str(journal), "s": 2}, bound_values={"gamma_exact_sha256": "f" * 64})
    assert r["premises"]["P-a"]["state"] == "NOT_ESTABLISHED" and "bound value gamma_exact_sha256" in r["premises"]["P-a"]["reason"]


def test_n5_self_consistency_alone_when_both_processes_agree(journal, monkeypatch):
    """Both the primary and the separate process return the same contradictory row: the reproduction is equal, only the
    self-consistency check sees it."""
    def both(spec, timeout=1800):
        job = json.loads(json.dumps(pl._row_job({k: v for k, v in spec.items() if k != "mode"}), default=pl._json_default))
        job["row"]["gamma_exact_sha256"] = "0" * 64
        return job

    monkeypatch.setattr(pl, "run_child", both)
    r = pl.analyse({"kind": "journal", "journal": str(journal), "s": 2, "mode": "round_up"})  # the primary through run_child too
    assert r["reproduction"]["bitwise_equal"] is True
    assert "row not self-consistent: gamma_exact_sha256" in r["premises"]["P-a"]["reason"] and r["status"] == CERT


def test_n5_second_path_unavailable_is_a_diagnostic_limitation(journal, monkeypatch):
    """Without the second path of gamma (cost limit or failure) the protocol status is unchanged (the second path is a channel of
    OD-4 on a subsample of OD-11, not a premise); the limitation is recorded, never hidden."""
    monkeypatch.setattr(pl, "SECOND_PATH_MAX_NODES", 0)
    row = _compute(journal, 2)
    sp = row["diagnostics"]["second_path_gamma"]
    assert sp["performed"] is False and sp["consistent"] is None and sp["reason"].startswith("NOT_PERFORMED")
    assert pl.assemble(row["components"], OK, faults=row["faults"])["status"] == "PASS"
    # a wrong gamma with its own consistent interval is then seen by nothing: the accepted contract holds for it
    row = _compute(journal, 2, mutation="gamma_sign_flip")
    assert row["gamma_contract"]["holds"] and row["diagnostics"]["gamma_interval_tightness"]["tight"] and row["contradictions"] == []
    assert pl.assemble(row["components"], OK, faults=row["faults"])["status"] == "PASS"


# --- integrated: positive and negative end-to-end journals (task 6) -------------------------------------------------------------------

ARRAY_NAMES = [name for name, _ in pl.STEP_ARRAYS]


@pytest.mark.parametrize("rehash", [False, True])
@pytest.mark.parametrize("k", [1, 2, 3])
@pytest.mark.parametrize("name", ARRAY_NAMES)
def test_integrated_every_array_corrupted_never_passes_from_its_step_on(journal, tmp_path, inproc_child, name, k, rehash):
    """One array of half-step k changed by 1 ulp (a zero entry becomes the smallest subnormal), with the hashes left (seen by the hash
    binding) or rewritten consistently (seen by the OP-8 re-run, the start, the chain): rows before k keep PASS, rows from k on are
    never PASS (UNRESOLVED_CERTIFICATE through П-а only)."""
    d = _rewrite(journal, tmp_path / "j", None, lambda a: _ulp(a, f"s{k}_{name}"), rehash)
    rows = pl.analyse_journal(d)
    got = [r["status"] for r in rows]
    assert got == ["PASS"] * (k - 1) + ["UNRESOLVED_CERTIFICATE"] * (4 - k), (name, k, rehash, got)
    assert all(r["premises"]["P-a"]["state"] == "NOT_ESTABLISHED" for r in rows[k - 1:])


def test_integrated_second_configuration_end_to_end(tmp_path):
    """T2 of the frozen expectations (12 x 9 on 1 x 0.75, N = 1, s = 1, 2): structure, sequence, canonical C-3+, gamma contract,
    exact dP and tau', E1/E2 and the strict resolvability, П-а with the separate process; no contradiction, no fault."""
    cfg = pl.op8().TrajectoryConfig(1.0, 12, 0.75, 9, 1, 1.0, 1.0, 2)
    jd = pl.run_trajectory_journal(cfg, tmp_path / "t2")
    j, arrays = pl.load_journal(jd)
    assert pl.journal_structure(j)["reasons"] == [] and all(pl.c3plus_record_defects(r) == [] for r in j["steps"])
    rows = pl.analyse_journal(jd)
    assert [r["status"] for r in rows] == ["PASS", "PASS"]
    for r in rows:
        x, c = r["row"], r["classification"]
        assert set(v["state"] for v in r["premises"].values()) == {"ESTABLISHED"} and r["reproduction"]["bitwise_equal"]
        dP = Fraction(float.fromhex(x["P_hat_p"])) - Fraction(float.fromhex(x["P_hat_s"]))
        tau = Fraction(float.fromhex(x["Gamma_up"])) + Fraction(float.fromhex(x["B_E_up"]))
        Q = Fraction(float.fromhex(x["Q_hat"]))
        assert Fraction(c["dP_exact"]) == dP and Fraction(c["tau_exact"]) == tau  # exact, from the recorded components
        assert dP >= -tau and abs(dP - Q) <= tau and Q > tau and c["e1"] and c["e2"]
        assert x["gamma_contract"]["holds"] and pl.row_self_consistency(x) == [] and r["contradictions"] == r["faults"] == []
        assert Fraction(float.fromhex(x["B_E_up"])) >= Fraction(x["B_E_exact"])
