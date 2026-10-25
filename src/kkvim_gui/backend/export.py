"""Machine-readable files of a result: manifest.json, history.csv, field.csv (GUI_DESIGN.md, 3.3, 8.4). Qt-free.

- Units are explicit: every dimensional value in JSON is {"value", "unit"}; every CSV column header
  carries its unit in brackets ("[1]" for dimensionless values).
- Statuses are explicit: data origin, solver status, stop, scientific status and, for R4, the mark
  PROVISIONAL / NOT_QUALIFIED in the manifest, in the first CSV line and in the R4 history column "mark".
- Demonstration data: first CSV line "# TEST DATA", ``data_origin: "demo"`` (GUI_DESIGN.md, 9).
- Numbers use the decimal point and Python's shortest round-trip representation (``repr``), so the
  exported values are bit-identical to the solver values; NaN and infinities are written as null/empty.
"""

import csv
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np

from kkvim_gui import program_version
from kkvim_gui.backend.outcome import RunOutcome

SCHEMA_VERSION = 1
SI_INPUT = {"a": "m", "b": "m", "t": "m", "E": "Pa", "nu": "1", "q": "Pa"}
FIELD_COLUMNS = (("i", ""), ("j", ""), ("x", "m"), ("y", "m"), ("w", "m"), ("w_bar", "1"))


def _plain(value):
    """JSON-compatible value: numpy scalars to Python, non-finite floats to None."""
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return [_plain(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _cell(value):
    """CSV text of a value: repr for floats (round-trip), true/false, empty for None or non-finite."""
    if value is None:
        return ""
    if isinstance(value, (bool, np.bool_)):
        return "true" if value else "false"
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return repr(value) if math.isfinite(value) else ""
    return str(value)


def _quantity(value, unit):
    return {"value": _plain(value), "unit": unit}


def marks(outcome: RunOutcome) -> list[str]:
    """Marks that every exported file of this result must carry, most important first."""
    out = []
    if outcome.is_demo:
        out.append("TEST DATA")
    if outcome.mark:
        out.append(outcome.mark)
    if outcome.is_error:
        out.append("DIAGNOSTIC: the run ended with a solver error; not a result")
    return out


def manifest(outcome: RunOutcome, run_meta: dict | None = None, log=(), files=None) -> dict:
    r = outcome.request
    extremum = outcome.extremum()
    data = {
        "schema_version": SCHEMA_VERSION,
        "data_origin": outcome.origin,
        "marks": marks(outcome),
        "created": datetime.now().isoformat(timespec="seconds"),
        "program": {"kkvim_gui": program_version(), "versions": _plain(outcome.versions)},
        "run": _plain(run_meta or {}),
        "input": {
            "entered": [{"key": k, "text": text, "unit": unit} for k, text, unit in r.entered],
            "si": {name: _quantity(getattr(r, name), unit) for name, unit in SI_INPUT.items()},
            "supports": r.supports,
            "load": "uniform",
            "theory": "Kirchhoff-Love",
            "N": r.N,
            "grid": {"n_x": r.n_x, "n_y": r.n_y},
            "method": _plain({key: getattr(r, key) for key in (
                ("tol_inf", "tol_E", "s_min", "s_max", "strict") if r.N == 1 else
                ("r4_mode", "research_mode", "stop_on_stabilization", "transform", "half_steps"))}),
        },
        "solver": {"stage": outcome.solver, "call": outcome.call, "parameters": _plain(dict(outcome.parameters))},
        "statuses": {
            "solver_status": outcome.status,
            "solver_reason": outcome.reason,
            "error": outcome.error,
            "stop": {"kind": outcome.stop_kind, "text": outcome.stop_text},
            "scientific": {"kind": outcome.science_kind, "text": outcome.science_text},
            "mark": outcome.mark,
        },
        "results": {
            "D": _quantity(outcome.D, "N·m"),
            "s": outcome.s,
            "k": outcome.k,
            "field_available": outcome.has_field,
            "field_role": None if not outcome.has_field else (
                "diagnostic only: field of the last completed half-step of a run that ended with an error"
                if outcome.is_error else "field of the last completed half-step"),
            "completed_without_error": not outcome.is_error,
            "extremum_by_nodes": None if extremum is None else {
                "i": extremum["i"], "j": extremum["j"], "x": _quantity(extremum["x"], "m"),
                "y": _quantity(extremum["y"], "m"), "w": _quantity(extremum["w"], "m"),
                "w_bar": _quantity(extremum["w_bar"], "1"),
            },
            "w_bar_definition": "w_bar = w D / (q a^4), MATH_SPEC 2.5",
        },
        "history_columns": [{"key": c.key, "title": c.title, "kind": c.kind,
                             "unit": (c.unit or "1") if c.kind == "number" else None} for c in outcome.columns],
        "diagnostics": _plain(dict(outcome.diagnostics)),
        "notes": list(outcome.notes),
        "log": [{"time": t, "level": level, "text": text} for t, level, text in log],
        "files": files or {},
    }
    return data


def _header_lines(outcome: RunOutcome) -> list[str]:
    lines = [f"# {m}" for m in marks(outcome)]
    lines.append(f"# solver {outcome.solver} ({outcome.call}); status {outcome.status}; N = {outcome.request.N}; "
                 f"grid {outcome.request.n_x} x {outcome.request.n_y}")
    return lines


def _write_csv(path: Path, comments, header, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        for line in comments:
            f.write(line + "\n")
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(header)
        for row in rows:
            writer.writerow([_cell(v) for v in row])


def write_history_csv(outcome: RunOutcome, path) -> Path:
    path = Path(path)
    header = [f"{c.key} [{c.unit or '1'}]" if c.kind == "number" and c.key not in ("s", "k") else c.key
              for c in outcome.columns]
    rows = ([row.get(c.key) for c in outcome.columns] for row in outcome.history)
    _write_csv(path, _header_lines(outcome), header, rows)
    return path


def write_field_csv(outcome: RunOutcome, path) -> Path:
    if not outcome.has_field:
        raise ValueError("the result has no field")
    path = Path(path)
    header = [name if not unit else f"{name} [{unit}]" for name, unit in FIELD_COLUMNS]
    x, y, w, w_bar = outcome.x, outcome.y, outcome.w, outcome.w_bar

    def rows():
        for i in range(len(x)):
            for j in range(len(y)):
                yield (i, j, x[i], y[j], w[i, j], w_bar[i, j])

    _write_csv(path, _header_lines(outcome), header, rows())
    return path


def write_result(outcome: RunOutcome, folder, run_meta=None, log=(), include_field=True) -> dict:
    """manifest.json, history.csv and (if requested and available) field.csv in ``folder``; {name: path}."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    suffix = "_TEST-DATA" if outcome.is_demo else ""
    paths = {"history": write_history_csv(outcome, folder / f"history{suffix}.csv")}
    if include_field and outcome.has_field:
        paths["field"] = write_field_csv(outcome, folder / f"field{suffix}.csv")
    files = {name: path.name for name, path in paths.items()}
    manifest_path = folder / f"manifest{suffix}.json"
    manifest_path.write_text(json.dumps(manifest(outcome, run_meta, log, files), ensure_ascii=False, indent=2),
                             encoding="utf-8")
    paths["manifest"] = manifest_path
    return paths
