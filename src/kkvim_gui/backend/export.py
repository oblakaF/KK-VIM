"""Machine-readable files of a result: manifest.json, history.csv and the field file (GUI_DESIGN.md, 3.3, 8.4). Qt-free.

- The field file is named by the field role of the outcome (``RunOutcome.field_role``, one rule of the adapter):
  field.csv — accepted R3 CONVERGED; field_research.csv — a research or PROVISIONAL run that ended as requested;
  field_diagnostic.csv — the field of the last completed half-step of a run that did not; no field — no file.
  The manifest (schema 2) states the role (code in ``statuses.field_role`` and ``results.field_role_code``; the text in
  ``results.field_role`` as in schema 1), the actual file name, the original solver status and reason, the
  scientific status, whether the requested run completed, the completed and the requested s, the route and its
  decisions and limits. ``read_manifest`` also reads schema 1 (written before the field roles) for the same keys.
- A result is written only into a new or empty folder: an existing package is never overwritten.

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

SCHEMA_VERSION = 2
SCHEMA_VERSIONS_READ = (1, 2)
ROLE_TEXT = {
    "accepted": "accepted solver result: field of the last completed half-step of an R3 run with CONVERGED",
    "research": "research field: the run ended as requested; RESEARCH / PROVISIONAL / NOT_QUALIFIED, never a qualified result",
    "diagnostic": "diagnostic only: field of the last completed half-step of a run that did not end as requested; not a result",
}
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
    if outcome.diagnostic_only:
        out.append("DIAGNOSTIC: the run ended with a solver error; not a result" if outcome.is_error else
                   f"DIAGNOSTIC: the run stopped with {outcome.status}; field of the last completed half-step, not a result")
    elif outcome.is_error:
        out.append("DIAGNOSTIC: the run ended with a solver error; not a result")
    return out


def field_file_name(outcome: RunOutcome) -> str | None:
    """The file name of the field of ``outcome`` (role, TEST DATA suffix), or None without a field."""
    if outcome.field_file is None:
        return None
    return f"{outcome.field_file}{'_TEST-DATA' if outcome.is_demo else ''}.csv"


def manifest(outcome: RunOutcome, run_meta: dict | None = None, log=(), files=None) -> dict:
    r = outcome.request
    extremum = outcome.extremum()
    field_name = (files or {}).get("field")
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
            "method": _plain(_method(r)),
        },
        "model": {"theory": "Kirchhoff-Love", "supports": r.supports, "N": r.N, "route": outcome.route,
                  "decisions": list(outcome.decisions)},
        "solver": {"stage": outcome.solver, "call": outcome.call, "parameters": _plain(dict(outcome.parameters))},
        "statuses": {
            "solver_status": outcome.status,
            "solver_reason": outcome.reason,
            "error": outcome.error,
            "stop": {"kind": outcome.stop_kind, "text": outcome.stop_text},
            "scientific": {"kind": outcome.science_kind, "text": outcome.science_text},
            "scientific_status": outcome.mark or "R3 ACCEPTED solver (D-043); a single run is not a scientific acceptance",
            "mark": outcome.mark,
            "requested_run_completed": outcome.requested_completed,
            "field_role": outcome.field_role,
            "field_file": field_name,
        },
        "results": {
            "D": _quantity(outcome.D, "N·m"),
            "s": outcome.s,
            "k": outcome.k,
            "half_steps": {"completed": outcome.s, "requested": outcome.s_requested},
            "field_available": outcome.has_field,
            "field_role": ROLE_TEXT.get(outcome.field_role),  # text, as in schema 1 ("diagnostic only: ..." for diagnostics)
            "field_role_code": outcome.field_role,
            "field_file": field_name,
            "requested_run_completed": outcome.requested_completed,
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
        "limitations": list(outcome.limitations),
        "log": [{"time": t, "level": level, "text": text} for t, level, text in log],
        "files": files or {},
    }
    return data


def _method(r) -> dict:
    """The method parameters of the request that its route uses."""
    from kkvim_gui.backend.request import CCCC_ROUTES, ROUTE_R3, ROUTE_SSSS_N46

    if r.route == ROUTE_R3:
        keys = ("tol_inf", "tol_E", "s_min", "s_max", "strict")
    elif r.route == ROUTE_SSSS_N46:
        keys = ("ssss_research", "half_steps")
    elif r.route in CCCC_ROUTES:
        keys = ("half_steps",)
    else:
        keys = ("r4_mode", "research_mode", "stop_on_stabilization", "transform", "half_steps")
    return {key: getattr(r, key) for key in keys}


def _header_lines(outcome: RunOutcome) -> list[str]:
    lines = [f"# {m}" for m in marks(outcome)]
    lines.append(f"# solver {outcome.solver} ({outcome.call}); status {outcome.status}; {outcome.request.supports}; "
                 f"N = {outcome.request.N}; grid {outcome.request.n_x} x {outcome.request.n_y}; "
                 f"s = {outcome.s} of {outcome.s_requested if outcome.s_requested is not None else '—'}; "
                 f"field role {outcome.field_role or 'none'}")
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


def new_folder(folder) -> Path:
    """Create ``folder``; an existing non-empty folder is refused (FileExistsError): a package is never overwritten."""
    folder = Path(folder)
    if folder.exists() and (not folder.is_dir() or any(folder.iterdir())):
        raise FileExistsError(f"папка уже существует и не пуста, расчёт в ней не перезаписывается: {folder}")
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def write_result(outcome: RunOutcome, folder, run_meta=None, log=(), include_field=True) -> dict:
    """manifest.json, history.csv and (if requested and available) the field file of its role in a new ``folder``."""
    folder = new_folder(folder)
    suffix = "_TEST-DATA" if outcome.is_demo else ""
    paths = {"history": write_history_csv(outcome, folder / f"history{suffix}.csv")}
    if include_field and outcome.has_field:
        paths["field"] = write_field_csv(outcome, folder / field_file_name(outcome))
    files = {name: path.name for name, path in paths.items()}
    manifest_path = folder / f"manifest{suffix}.json"
    manifest_path.write_text(json.dumps(manifest(outcome, run_meta, log, files), ensure_ascii=False, indent=2),
                             encoding="utf-8")
    paths["manifest"] = manifest_path
    return paths


def _legacy_role(data: dict) -> str | None:
    """The field role of a schema 1 manifest by the rule of schema 2 (its file was always field.csv)."""
    results, statuses = data.get("results", {}), data.get("statuses", {})
    if not results.get("field_available"):
        return None
    if str(results.get("field_role") or "").startswith("diagnostic"):
        return "diagnostic"
    if statuses.get("solver_status") == "CONVERGED" and data.get("solver", {}).get("stage") == "R3":
        return "accepted"
    if statuses.get("mark"):  # R4: the files of schema 1 do not record the mode's end
        return "research" if statuses.get("stop", {}).get("kind") in ("met", "none") else "diagnostic"
    return "diagnostic"


def read_manifest(path) -> dict:
    """A manifest of schema 1 or 2 with the keys of schema 2 that every reader needs (field role and file, statuses).

    Schema 1 did not record the role: it is derived from its statuses by the rule of schema 2, its field file keeps
    its actual name (field.csv) and ``legacy`` is True. Another schema is refused (ValueError).
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    version = data.get("schema_version")
    if version not in SCHEMA_VERSIONS_READ:
        raise ValueError(f"manifest schema {version!r} is not supported; supported: {SCHEMA_VERSIONS_READ}")
    statuses, results, files = data.get("statuses", {}), data.get("results", {}), data.get("files", {})
    role = statuses.get("field_role") if version == 2 else _legacy_role(data)
    return {
        "schema_version": version, "legacy": version == 1, "data_origin": data.get("data_origin"),
        "marks": data.get("marks", []), "supports": data.get("input", {}).get("supports"), "N": data.get("input", {}).get("N"),
        "solver_status": statuses.get("solver_status"), "solver_reason": statuses.get("solver_reason"),
        "scientific": statuses.get("scientific"), "mark": statuses.get("mark"),
        "requested_run_completed": results.get("requested_run_completed") if version == 2 else None,
        "field_role": role, "field_file": files.get("field"),
        "s_completed": results.get("s"), "s_requested": results.get("half_steps", {}).get("requested") if version == 2 else None,
    }
