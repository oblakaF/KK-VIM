"""Control calculations of R3 (N = 1) and R4 (N = 2-3) through the real kkvim API, for a future GUI adapter (QA).

The script calls only library functions and serializes what they return; it has no solver of its own:
- N = 1: ``kkvim.rectangle_run.run_rectangle_physical`` and ``kkvim.rectangle_run.halfstep_report``;
- N = 2-3: ``kkvim.rectangle.validated_rectangle_input`` (the R3 physical checks and D, which is
  ``kkvim.physical.bending_stiffness``) and ``kkvim.coupled_run.run_coupled``. ``run_coupled`` itself takes D and
  checks only the sides, D and the grids, so E, t, nu are validated here by the library function of R3.
  The normative mode is the default of ``run_coupled`` (``research_mode=False``); the research mode is used only
  when a case asks for it.

The only arithmetic of the script is the normalization w_bar = w D / (q a^4) for N = 2-3 (MATH_SPEC 2.5; for
N = 1 it is taken from ``RectangleResult.dimensionless_field``), the maximum of |w| and its node.

Statuses are passed through unchanged. ``status_class`` only groups them; a run is reported as COMPLETED only for
the R3 status CONVERGED; every R4 result keeps PROVISIONAL / NOT_QUALIFIED; an unknown status is an error.

    uv run python scripts/gui_backend_cases.py list
    uv run python scripts/gui_backend_cases.py run r3_square r4_n2_research --out scratch/gui_demo --format csv
    uv run python scripts/gui_backend_cases.py run --all --out scratch/gui_demo --format npz
    uv run python scripts/gui_backend_cases.py custom --N 3 --mode research --a 1 --b 2 --t 0.01 --E 2.1e11 \\
        --nu 0.3 --q 1000 --nx 32 --ny 64 --out scratch/gui_demo
    uv run python scripts/gui_backend_cases.py golden          # docs/results/gui_backend/ (cases.json + fields)
"""

import argparse
import csv
import json
import math
import platform
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from importlib import metadata
from pathlib import Path

import numpy as np

import kkvim.coupled_run as cr
from kkvim.controller import CONVERGED, MAX_HALF_STEPS, STAGNATED
from kkvim.physical import PhysicalInputError, bending_stiffness
from kkvim.rectangle import validated_rectangle_input
from kkvim.rectangle_run import halfstep_report, run_rectangle_physical

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "docs" / "results" / "gui_backend"
SCHEMA = "kkvim-gui-backend-case/1"

STEEL = dict(t=0.01, E=2.1e11, nu=0.3, q=1000.0)  # m, Pa, -, Pa
UNIT = dict(t=1.0, E=12.0, nu=0.0, q=1.0)  # D = 12 * 1 / 12 = 1 exactly: the dimensionless plate of the QA tables

LIMIT_R3 = ("R3 is ACCEPTED (D-043) for the rectangle SSSS, N = 1, q = const, 0.2 <= b/a <= 5; qualification only on "
            "the checked grids; convergence in s and moments are not claimed")
LIMIT_R4 = ("R4 is IN_PROGRESS: every research result is PROVISIONAL / NOT_QUALIFIED (D-047, D-048); "
            "PROVISIONAL_FIELD_STABILIZED is a stabilization of the finite problem, not convergence to omega_0 and not a "
            "rank certificate; the normative mode stops after s = 1 with NOT_QUALIFIED")
LIMIT_GUI = "integration regression reference, not an independent scientific solution; the GUI itself is not verified"


def case(model, inputs, purpose, *, N=1, mode=None, transform="fft", half_steps=100):
    spec = {"model": model, "N": N, "inputs": dict(inputs), "purpose": purpose}
    if model == "R4":
        spec.update(mode=mode, transform=transform, half_steps=half_steps)
    return spec


def plate(a, b, n_x, n_y, material=STEEL, **change):
    return {"a": a, "b": b, **material, "n_x": n_x, "n_y": n_y, **change}


CASES = {
    # N = 1, R3 (also r = 1, the R2 square)
    "r3_square": case("R3", plate(1.0, 1.0, 24, 24), "square SSSS, steel, n = 24"),
    "r3_rect_b2a": case("R3", plate(1.0, 2.0, 24, 48), "rectangle b = 2a, h_x = h_y"),
    "r3_rect_a2b": case("R3", plate(2.0, 1.0, 48, 24), "rectangle a = 2b (sides swapped)"),
    "r3_rect_r5_unequal": case("R3", plate(0.5, 2.5, 16, 40), "b/a = 5 (range limit), h_x != h_y"),
    "r3_negative_q": case("R3", plate(1.0, 1.5, 24, 36, q=-500.0), "load q < 0: the field changes sign"),
    "r3_error_nu": case("R3", plate(1.0, 1.0, 24, 24, nu=1.0), "|nu| >= 1: ERROR_INPUT without a field"),
    "r3_error_ratio": case("R3", plate(1.0, 6.0, 24, 24), "b/a = 6 outside D-038: ERROR_INPUT"),
    "r3_error_grid": case("R3", plate(1.0, 1.0, 3, 24), "n_x = 3 < N_MIN: ERROR_INPUT"),
    # N = 2-3, R4
    "r4_n2_normative": case("R4", plate(1.0, 1.0, 24, 24), "normative mode (default): NOT_QUALIFIED after s = 1",
                            N=2, mode="normative", half_steps=5),
    "r4_n2_normative_one": case("R4", plate(1.0, 1.0, 24, 24), "normative mode, half_steps = 1",
                                N=2, mode="normative", half_steps=1),
    "r4_n2_research": case("R4", plate(1.0, 1.0, 24, 24), "research mode, D-048 stop", N=2, mode="research"),
    "r4_n3_research_rect": case("R4", plate(1.0, 2.0, 24, 48), "research mode, N = 3, b = 2a", N=3, mode="research"),
    "r4_n3_research_fixed": case("R4", plate(2.0, 1.0, 40, 20), "research mode, 12 fixed half-steps, a = 2b",
                                 N=3, mode="research_fixed", half_steps=12),
    "r4_n2_dense": case("R4", plate(1.0, 1.0, 24, 24), "research mode, dense DST (D-048.6: n <= 40)",
                        N=2, mode="research", transform="dense"),
    "r4_n3_rank_loss": case("R4", plate(1.0, 1.0, 4, 24), "n_x = 4 < 2N = 6: STOPPED_RANK_LOSS_PROVEN without a field",
                            N=3, mode="research"),
    "r4_error_E": case("R4", plate(1.0, 1.0, 24, 24, E=-2.1e11, t=-0.01), "E < 0, t < 0 (D > 0): ERROR_INPUT",
                       N=2, mode="research"),
}


# --- status classes ---------------------------------------------------------------------------------------------

R3_CLASSES = {CONVERGED: "COMPLETED", STAGNATED: "STOPPED_WITHOUT_CONVERGENCE", MAX_HALF_STEPS: "STOPPED_WITHOUT_CONVERGENCE"}
R4_CLASSES = {
    cr.PROVISIONAL_FIELD_STABILIZED: "PROVISIONAL", cr.RESEARCH_LIMIT_REACHED: "PROVISIONAL",
    cr.MAX_HALF_STEPS: "PROVISIONAL", cr.NOT_QUALIFIED: "NOT_QUALIFIED", cr.HALF_STEP_LIMIT_REACHED: "NOT_QUALIFIED",
    cr.PROVISIONAL_RANK_GUARD_STOP: "PROVISIONAL_STOPPED", cr.PROVISIONAL_SET_GUARD_STOP: "PROVISIONAL_STOPPED",
    cr.PROVISIONAL_ENERGY_E1_STOP: "PROVISIONAL_STOPPED", cr.STOPPED_RANK_LOSS_PROVEN: "STOPPED_RANK_LOSS",
    cr.STOPPED_RANK_NUMERICALLY_UNRESOLVED: "STOPPED_RANK_LOSS",
}


def status_class(model: str, status: str) -> str:
    """Group of a library status (no status is changed). Unknown statuses are UNKNOWN, which counts as an error."""
    if not isinstance(status, str):
        return "UNKNOWN"
    if status.startswith("ERROR_"):
        return "ERROR"
    return (R3_CLASSES if model == "R3" else R4_CLASSES if model == "R4" else {}).get(status, "UNKNOWN")


def scientific_status(model: str, status: str) -> str:
    if model == "R4":
        return cr.MARK
    if status_class(model, status) == "COMPLETED":
        return "R3 ACCEPTED (D-043) within its scope; this run: CONVERGED by the stopping rule of NUMERICAL_SPEC 11"
    return f"R3 run not completed ({status}); no result is claimed"


# --- runs -------------------------------------------------------------------------------------------------------

def run_case(spec: dict) -> dict:
    """Run one case through the library; returns the export record (JSON-ready) and the arrays."""
    if spec["model"] == "R3":
        return _run_r3(spec)
    if spec["model"] == "R4":
        return _run_r4(spec)
    raise ValueError(f"unknown model {spec['model']!r}")


def _run_r3(spec):
    p = spec["inputs"]
    result = run_rectangle_physical(p["a"], p["b"], p["t"], p["E"], p["nu"], p["q"], p["n_x"], p["n_y"])
    report = halfstep_report(result)
    call = ("kkvim.rectangle_run.run_rectangle_physical(a, b, t, E, nu, q, n_x, n_y); "
            "kkvim.rectangle_run.halfstep_report(result)")
    run = result.run
    w = None if run is None else run.last_field()
    w_bar = result.dimensionless_field()
    grid = None
    if w is not None:
        last = run.history.records[-1]
        grid = (last.grid_x, last.grid_y)
    history = [_r3_row(h) for h in report.halfsteps]
    params = None if run is None or run.parameters is None else _plain(run.parameters)
    return _record(spec, call, result.status, None if result.error is None else str(result.error),
                   result.D, w, w_bar, grid, history, params, mark=None, api_objects={"result": result, "report": report})


def _run_r4(spec):
    p = spec["inputs"]
    mode = spec["mode"]
    kwargs = {"transform": spec["transform"], "half_steps": spec["half_steps"]}
    if mode == "research":
        kwargs["research_mode"] = True
    elif mode == "research_fixed":
        kwargs.update(research_mode=True, stop_on_stabilization=False)
    elif mode != "normative":
        raise ValueError(f"unknown R4 mode {mode!r}")
    call = ("kkvim.rectangle.validated_rectangle_input(a, b, t, E, nu, q, n_x, n_y) -> D; "
            f"kkvim.coupled_run.run_coupled(a, b, D, q, n_x, n_y, N, {_kwargs_text(kwargs)})")
    try:
        rect, D = validated_rectangle_input(p["a"], p["b"], p["t"], p["E"], p["nu"], p["q"], p["n_x"], p["n_y"])
    except PhysicalInputError as exc:
        return _record(spec, call, "ERROR_INPUT", str(exc), None, None, None, None, [], None, mark=cr.MARK,
                       api_objects={"run": None})
    run = cr.run_coupled(rect.a, rect.b, D, rect.q, rect.n_x, rect.n_y, spec["N"], **kwargs)
    w = run.field()
    w_bar = None if w is None else w * D / (rect.q * rect.a**4)
    grid = None if w is None else (run.grid_x, run.grid_y)
    history = [_r4_row(st) for st in run.steps]
    params = None if run.parameters is None else _plain(run.parameters)
    extra = {"converged": run.converged, "first_stabilization_s": run.first_stabilization_s,
             "energy_s2": {k: _plain(getattr(run, k)) for k in ("energy_s2_checked", "energy_s2_mandatory",
                                                                "energy_s2_status", "energy_s2_m",
                                                                "energy_s2_mandatory_unresolved", "energy_s2_reason")}}
    return _record(spec, call, run.status, run.reason, D, w, w_bar, grid, history, params, mark=run.mark,
                   api_objects={"run": run}, extra=extra)


def _record(spec, call, status, reason, D, w, w_bar, grid, history, params, *, mark, api_objects, extra=None):
    model = spec["model"]
    cls = status_class(model, status)
    p = spec["inputs"]
    out = {
        "schema": SCHEMA,
        "model": model,
        "N": spec["N"],
        "purpose": spec.get("purpose", ""),
        "input": {**p, **({k: spec[k] for k in ("mode", "transform", "half_steps")} if model == "R4" else {})},
        "units": "consistent units of the input (e.g. m, Pa); no conversion; w in units of length, w_bar = w D/(q a^4)",
        "api_call": call,
        "source": source_info(),
        "status": status,
        "status_class": cls,
        "is_error": cls in ("ERROR", "UNKNOWN"),
        "completed": cls == "COMPLETED",
        "scientific_status": scientific_status(model, status),
        "mark": mark,
        "reason": reason,
        "D": D,
        "D_formula": "kkvim.physical.bending_stiffness: E t^3 / (12 (1 - nu^2))",
        "parameters": params,
        "half_steps_done": len(history),
        "history": history,
        "limitations": [LIMIT_R3 if model == "R3" else LIMIT_R4, LIMIT_GUI],
    }
    if extra:
        out.update(extra)
    out.update(_field_summary(w, w_bar, grid))
    arrays = {} if w is None else {"x": grid[0].nodes, "y": grid[1].nodes, "w": w, "w_bar": w_bar}
    return {"record": _plain(out), "arrays": arrays, "api": api_objects}


def _field_summary(w, w_bar, grid):
    if w is None:
        return {"field_available": False, "field": None}
    gx, gy = grid
    i, j = np.unravel_index(int(np.argmax(np.abs(w))), w.shape)
    centre = None
    if gx.n % 2 == 0 and gy.n % 2 == 0:
        ic, jc = gx.n // 2, gy.n // 2
        centre = {"i": ic, "j": jc, "x": float(gx.nodes[ic]), "y": float(gy.nodes[jc]),
                  "w": float(w[ic, jc]), "w_bar": float(w_bar[ic, jc])}
    boundary = np.concatenate([w[0, :], w[-1, :], w[:, 0], w[:, -1]])
    return {"field_available": True, "field": {
        "shape": list(w.shape),
        "axes": "axis 0 = x: n_x + 1 nodes on [0, a]; axis 1 = y: n_y + 1 nodes on [0, b]; boundary nodes included",
        "role": "field of the last completed half-step",
        "max_abs_w": float(np.max(np.abs(w))),
        "w_at_max_abs": float(w[i, j]),
        "argmax_abs": {"i": int(i), "j": int(j), "x": float(gx.nodes[i]), "y": float(gy.nodes[j])},
        "max_abs_w_bar": float(np.max(np.abs(w_bar))),
        "centre": centre,
        "boundary_max_abs": float(np.max(np.abs(boundary))),
        "w_bar_definition": "w D / (q a^4), a = length of the x side",
    }}


def _r3_row(h):
    ch, en = h.changes, h.energy
    return {
        "s": h.s, "k": h.k, "direction": h.direction, "solve_axis": h.solve_axis, "c": h.c,
        "A": h.coefficients.A, "B": h.coefficients.B, "C": h.coefficients.C, "F": h.coefficients.F,
        "eta": h.eta, "residual_norm": h.residual_norm, "refined": h.refined, "imag_ratio": h.imag_ratio,
        "P": h.P, "norm_a": h.norm_a,
        "eps_inf": None if ch is None else ch.eps_inf, "eps_E": None if ch is None else ch.eps_E,
        "eps_L2": None if ch is None else ch.eps_L2, "theta": None if ch is None else ch.theta,
        "satisfied": None if ch is None else ch.satisfied, "tail_estimate": None if ch is None else ch.tail_estimate,
        "energy_status": None if en is None else en.status, "delta_P": None if en is None else en.delta_P,
        "Q": None if en is None else en.Q, "tau": None if en is None else en.tau,
        "e1": None if en is None else en.e1, "e2": None if en is None else en.e2,
        "w_bar_centre": dict((f"{p[0]},{p[1]}", v) for p, v in h.control_values).get("1/2,1/2"),
    }


def _r4_row(st):
    e, g, rk = st.energy, st.guards, st.rank
    return {
        "s": st.s, "k": st.k, "direction": st.direction, "eta_N": st.eta_N, "eta_initial": st.eta_initial,
        "refined": st.refined, "rho_N": rk.rho_N, "kappa_rep": rk.kappa_rep, "cond_U": rk.cond_U, "cond_V": rk.cond_V,
        "set_sigma_min": rk.set_sigma_min, "cond_A_given": st.cond_A_given, "cond_M_max": st.cond_M_max,
        "P": None if e is None else e.P, "delta_P": None if e is None else e.delta_P,
        "Q": None if e is None else e.Q, "tau": None if e is None else e.tau,
        "e1": None if e is None else e.e1, "e2": None if e is None else e.e2,
        "energy_status": None if e is None else e.status, "m_s": st.m_s, "w_max": st.w_max,
        "eps_inf": st.eps_inf, "eps_E": st.eps_E, "stabilization_ok": st.stabilization_ok,
        "guards_passed": None if g is None else g.passed, "qr_done": st.qr_done, "qr_field_error": st.qr_field_error,
        "stop_reason": st.stop_reason, "mark": st.mark,
        "energy_s2_mandatory_unresolved": st.energy_s2_mandatory_unresolved, "energy_s2_reason": st.energy_s2_reason,
    }


def _plain(value):
    """JSON-ready copy: dataclasses -> dict, numpy scalars -> Python, non-finite floats -> strings."""
    if is_dataclass(value) and not isinstance(value, type):
        return _plain(asdict(value))
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return value if math.isfinite(value) else repr(value)
    return value


def _kwargs_text(kwargs):
    return ", ".join(f"{k}={v!r}" for k, v in kwargs.items())


def source_info() -> dict:
    info = {"python": platform.python_version(), "platform": platform.platform()}
    for package in ("kkvim", "numpy", "scipy"):
        try:
            info[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            info[package] = "unknown"
    try:
        info["git_commit"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                                            check=True).stdout.strip()
        info["git_dirty"] = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                                                capture_output=True, text=True, check=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        info["git_commit"], info["git_dirty"] = "unknown", None
    return info


# --- files ------------------------------------------------------------------------------------------------------

def export(name: str, result: dict, out: Path, fmt: str = "csv") -> dict:
    """Write <name>.json (+ field and history files). Returns the written file names."""
    out.mkdir(parents=True, exist_ok=True)
    record, arrays = dict(result["record"]), result["arrays"]
    files = {"json": f"{name}.json"}
    if record["history"]:
        files["history_csv"] = f"{name}_history.csv"
        write_rows(out / files["history_csv"], record["history"])
    if arrays:
        if fmt == "csv":
            files["w_csv"], files["w_bar_csv"] = f"{name}_w.csv", f"{name}_w_bar.csv"
            write_field_csv(out / files["w_csv"], arrays["x"], arrays["y"], arrays["w"])
            write_field_csv(out / files["w_bar_csv"], arrays["x"], arrays["y"], arrays["w_bar"])
        elif fmt == "npz":
            files["npz"] = f"{name}.npz"
            np.savez(out / files["npz"], **arrays)
        elif fmt == "npy":
            for key, value in arrays.items():
                files[f"{key}_npy"] = f"{name}_{key}.npy"
                np.save(out / files[f"{key}_npy"], value)
        else:
            raise ValueError(f"unknown format {fmt!r}")
    record["files"] = files
    (out / files["json"]).write_text(json.dumps(record, indent=1, ensure_ascii=False, allow_nan=False) + "\n",
                                     encoding="utf-8")
    return files


def write_field_csv(path: Path, x, y, values):
    """Wide CSV: first row 'x\\y' and the y nodes; every next row an x node and the values (repr, exact round trip)."""
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(["x\\y"] + [repr(float(v)) for v in y])
        for xi, row in zip(x, values):
            writer.writerow([repr(float(xi))] + [repr(float(v)) for v in row])


def read_field_csv(path: Path):
    with open(path, encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    y = np.array([float(v) for v in rows[0][1:]])
    x = np.array([float(r[0]) for r in rows[1:]])
    values = np.array([[float(v) for v in r[1:]] for r in rows[1:]])
    return x, y, values


def write_rows(path: Path, rows):
    keys = list(rows[0])
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if row[k] is None else repr(row[k]) if isinstance(row[k], float) else row[k])
                             for k in keys})


def golden(out: Path = GOLDEN) -> None:
    """Compact machine-readable set of all cases: cases.json + w fields (CSV) for the comparison with a GUI adapter."""
    out.mkdir(parents=True, exist_ok=True)
    summary = {"schema": SCHEMA + "/golden", "source": source_info(),
               "comparison": {
                   "same_environment": "a GUI adapter calling the library must reproduce the direct API call of the same "
                                       "environment bitwise (status, D, history, field)",
                   "other_platform_R4": "field within the doubled first-order field model _model of "
                                        "tests/cross/test_r4_controller_comparison.py (as K-1 of D-048)",
                   "other_platform_R3": "status, half-step count, D and shape exactly; field values informative only "
                                        "(no accepted cross-platform tolerance for R3 fields)"},
               "cases": {}}
    for name, spec in CASES.items():
        result = run_case(spec)
        files = export(name, result, out / "cases", fmt="csv")
        rec = result["record"]
        summary["cases"][name] = {"spec": spec, "input": rec["input"], "units": rec["units"],
                                  "api_call": rec["api_call"], "expected_status": rec["status"],
                                  "expected_status_class": rec["status_class"], "is_error": rec["is_error"],
                                  "scientific_status": rec["scientific_status"], "D": rec["D"],
                                  "half_steps_done": rec["half_steps_done"], "field_available": rec["field_available"],
                                  "field": rec["field"], "limitations": rec["limitations"],
                                  "files": {k: f"cases/{v}" for k, v in files.items()},
                                  "provenance": "kkvim library at source.git_commit, this script; not an independent "
                                                "reference"}
    (out / "cases.json").write_text(json.dumps(_plain(summary), indent=1, ensure_ascii=False, allow_nan=False) + "\n",
                                    encoding="utf-8")


# --- command line -----------------------------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    run = sub.add_parser("run")
    run.add_argument("names", nargs="*")
    run.add_argument("--all", action="store_true")
    run.add_argument("--out", default="scratch/gui_demo")
    run.add_argument("--format", choices=("csv", "npz", "npy"), default="csv")
    custom = sub.add_parser("custom")
    custom.add_argument("--name", default="custom")
    custom.add_argument("--N", type=int, default=1)
    custom.add_argument("--mode", choices=("normative", "research", "research_fixed"), default="normative")
    custom.add_argument("--transform", choices=("fft", "dense"), default="fft")
    custom.add_argument("--half-steps", type=int, default=100)
    for key in ("a", "b", "t", "E", "nu", "q"):
        custom.add_argument(f"--{key}", type=float, required=True)
    custom.add_argument("--nx", type=int, required=True)
    custom.add_argument("--ny", type=int, required=True)
    custom.add_argument("--out", default="scratch/gui_demo")
    custom.add_argument("--format", choices=("csv", "npz", "npy"), default="csv")
    gold = sub.add_parser("golden")
    gold.add_argument("--out", default=str(GOLDEN))
    args = parser.parse_args(argv)

    if args.cmd == "list":
        for name, spec in CASES.items():
            print(f"{name:24s} {spec['model']} N={spec['N']} {spec.get('mode') or ''} {spec['purpose']}")
        return 0
    if args.cmd == "golden":
        golden(Path(args.out))
        print("written", args.out)
        return 0
    if args.cmd == "run":
        names = list(CASES) if args.all else args.names
        unknown = [n for n in names if n not in CASES]
        if unknown or not names:
            parser.error(f"unknown or no cases: {unknown or names}")
        jobs = [(n, CASES[n]) for n in names]
    else:
        inputs = {"a": args.a, "b": args.b, "t": args.t, "E": args.E, "nu": args.nu, "q": args.q,
                  "n_x": args.nx, "n_y": args.ny}
        spec = (case("R3", inputs, "custom") if args.N == 1 else
                case("R4", inputs, "custom", N=args.N, mode=args.mode, transform=args.transform,
                     half_steps=args.half_steps))
        jobs = [(args.name, spec)]
    for name, spec in jobs:
        result = run_case(spec)
        files = export(name, result, Path(args.out), args.format)
        rec = result["record"]
        field = rec["field"]
        print(f"{name}: {rec['status']} [{rec['status_class']}] s={rec['half_steps_done']} "
              f"D={rec['D']!r} max|w|={None if field is None else field['max_abs_w']!r} -> "
              f"{', '.join(files.values())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
