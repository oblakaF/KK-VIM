"""Export of a load series: load_series.csv, series_manifest.json and the graph (GUI-PLATE-WORKFLOW-UX-01). Qt-free.

A load series is several real runs, so it is saved as its own package, never disguised as a single-run result:

- ``load_series.csv`` — one row per load q in the order entered: the run, its execution, solver and scientific
  statuses, field role, whether it is on the KK-VIM curve (and why not), the KK-VIM centre deflection, the signed w at
  the node of the largest |w|, the Navier reference and the signed differences (SSSS only), the plate, N and grid.
  Units in brackets; numbers in Python's round-trip form (``export._cell``), so the file holds the binary64 values;
- with D2 (CCCC, «МКР — двумерный (D2)»): per row the state of the D2 run at that q, its centre deflection, beta_a,
  e_D2, the signed difference KK-VIM − D2, and the timings of both methods (one execution each: observations);
- ``series_manifest.json`` — kind "load_series", schema ``kkvim-gui-load-series/1``, the same rows, provenance of the
  reference and the SHA256 of every file of the package;
- the graph (PNG), drawn by the interface (``render``);
- ``load_series_vim_cycles.csv`` (GUI-VIM-CYCLES-123-01, routes that record the centre per half-step): for every run
  of the series the states s = 1 and VIM-1 … VIM-3 (s = 2, 4, 6) from that run's stored history, with the reference of
  its q (``backend.vim_cycles``); its SHA256 is in the manifest like every file of the package.

The folder must be new or empty (``export.new_folder``): an existing package is never overwritten.
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path

from kkvim_gui import program_version
from kkvim_gui.backend import export, vim_cycles

SCHEMA = "kkvim-gui-load-series/1"
CSV_FILE = "load_series.csv"
MANIFEST_FILE = "series_manifest.json"
COLUMNS = (  # (name, unit): "1" dimensionless, "" text, flag or count
    ("point", ""), ("run_id", ""), ("q", "Pa"), ("run_status", ""), ("solver_status", ""), ("field_role", ""),
    ("requested_run_completed", ""), ("scientific_status", ""), ("mark", ""), ("on_kkvim_curve", ""), ("excluded_reason", ""),
    ("w_centre_kkvim", "m"), ("centre_i", ""), ("centre_j", ""), ("w_at_max_abs_node_kkvim", "m"),
    ("navier_state", ""), ("w_centre_navier", "m"), ("delta_w_centre", "m"), ("delta_w_centre_relative", "%"),
    ("supports", ""), ("N", ""), ("route", ""), ("a", "m"), ("b", "m"), ("h", "m"), ("E", "Pa"), ("nu", "1"),
    ("n_x", ""), ("n_y", ""),
    ("d2_state", ""), ("d2_reason", ""), ("on_d2_curve", ""), ("w_centre_d2", "m"), ("beta_a_d2", "1"), ("e_D2", "m"),
    ("delta_w_centre_kkvim_minus_d2", "m"), ("delta_w_centre_kkvim_minus_d2_relative", "%"),
    ("kkvim_solver_time", "s"), ("kkvim_process_wall_time", "s"), ("d2_solve_time", "s"), ("d2_process_wall_time", "s"),
)
DEFINITIONS = (
    "# load series: every row is a separate KK-VIM run at its own uniform load q (no scaling, no interpolation)",
    "# w_centre_kkvim: stored field at the node (a/2, b/2), only for runs that ended as requested with a result field; "
    "w_at_max_abs_node_kkvim: signed w at the node of the largest |w|",
    "# w_centre_navier: kkvim_reference.navier at (a/2, b/2), M = 2559, computed in a separate process for every q (SSSS "
    "only); delta_w_centre = w_centre_kkvim - w_centre_navier; delta_w_centre_relative = delta / w_centre_navier * 100 %",
    "# w_centre_d2: D2 (kkvim_reference.r5_cccc_d2, the same discrete problem as KK-VIM CCCC, D-049.1) computed in its own "
    "process at every q for the same plate and grid (CCCC with D2 chosen only); beta_a_d2 = w_centre_d2 * D / (q * a^4); "
    "delta_w_centre_kkvim_minus_d2 = w_centre_kkvim - w_centre_d2 (not an error of the continuous plate model)",
    "# times, s: kkvim_solver_time = kkvim_adapter.execute in the computing process; kkvim_process_wall_time = process "
    "start to the message read by the GUI (polling 0.2 s); d2_solve_time = r5_cccc_d2.solve in the D2 process; "
    "d2_process_wall_time = D2 process start to its complete answer; one execution each: observations, not a ranking",
    "# signed differences only: no agreement threshold is approved; the series is not a scientific acceptance",
)


def rows(data) -> list:
    """Rows of load_series.csv in the order of COLUMNS."""
    plate, (n_x, n_y) = data.plate, data.grid
    out = []
    for p in data.points:
        c = p.centre or {}
        out.append((p.index, p.run_id, p.q, p.run_status, p.solver_status, p.field_role,
                    p.requested_completed, p.scientific_status, p.mark or "", p.included, p.reason,
                    p.w_centre_m, c.get("i"), c.get("j"), p.w_at_max_abs_m,
                    p.navier_state or "not applicable", p.navier_w_m, p.delta_w_m, p.rel_pct,
                    data.supports, data.N, data.route, plate["a"], plate["b"], plate["t"], plate["E"], plate["nu"], n_x, n_y,
                    p.d2_state or "not applicable", p.d2_reason, p.on_d2_curve, p.d2_w_m, p.d2_beta_a, p.d2_e_m,
                    p.d2_delta_w_m, p.d2_rel_pct, p.kkvim_solver_s, p.kkvim_wall_s, p.d2_solve_s, p.d2_wall_s))
    return out


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_series(data, folder, render=None) -> dict:
    """Write the series package into a new ``folder``; ``render(folder) -> {"graph_png": path}`` draws the graph."""
    if not data.complete:
        raise ValueError("the series is still running: nothing is saved")
    folder = export.new_folder(folder)
    paths = {"load_series": folder / CSV_FILE}
    header = [f"{name} [{unit}]" if unit else name for name, unit in COLUMNS]
    marks = sorted({p.mark for p in data.points if p.mark})
    comments = [f"# {m}" for m in marks] + [f"# load series {data.series_id}: {data.supports}, N = {data.N}, route "
                                             f"{data.route}, grid {data.grid[0]} x {data.grid[1]}; status {data.status}"]
    export._write_csv(paths["load_series"], [*comments, *DEFINITIONS], header, rows(data))
    if data.cycles_supported:
        paths["load_series_vim_cycles"] = folder / vim_cycles.SERIES_CSV_FILE
        vim_cycles.write_csv(paths["load_series_vim_cycles"], [(p.run_id, p.cycles) for p in data.points
                                                               if p.cycles is not None], comments)
    if render is not None:
        paths.update({k: Path(v) for k, v in render(folder).items()})
    nav = data.navier if data.navier_ok else None
    manifest = {
        "schema": SCHEMA,
        "kind": "load_series",
        "created": datetime.now().isoformat(timespec="seconds"),
        "program": {"kkvim_gui": program_version()},
        "series": {"series_id": data.series_id, "status": data.status, "started": data.started, "supports": data.supports,
                   "N": data.N, "route": data.route, "grid": {"n_x": data.grid[0], "n_y": data.grid[1]},
                   "plate_si": {k: {"value": v, "unit": u} for (k, u), v in zip((("a", "m"), ("b", "m"), ("t", "m"),
                                                                             ("E", "Pa"), ("nu", "1")),
                                                                            (data.plate[k] for k in ("a", "b", "t", "E", "nu")),
                                                                            strict=True)},
                   "entered_without_q": [{"key": k, "text": t, "unit": u} for k, t, u in data.entered],
                   "loads_Pa": [p.q for p in data.points], "marks": marks},
        "columns": [{"name": n, "unit": u or None} for n, u in COLUMNS],
        "points": [dict(zip((n for n, _ in COLUMNS), (export._plain(v) for v in row), strict=True)) for row in rows(data)],
        "reference": None if nav is None else {"method": nav.get("method"), "source": nav.get("source"), "M": nav.get("M"),
                                               "x": nav.get("x"), "y": nav.get("y"), "D": nav.get("D"),
                                               "versions": nav.get("versions")},
        "reference_state": None if data.navier is None else data.navier.get("state"),
        "independent_d2": None if not data.d2 else {
            "title": "МКР — двумерный (D2)", "method": data.d2_method,
            "role": "executed in a separate process at every q for the same plate and grid; the same discrete problem as "
                    "KK-VIM CCCC (D-049.1, D-049.8); not stored benchmark data; not an error estimate of the plate model",
            "points_on_curve": len(data.d2_included)},
        "vim_cycles": None if not data.cycles_supported else {
            "labels": vim_cycles.OWNER_NOTE, "file": vim_cycles.SERIES_CSV_FILE,
            "points_on_curve": {vim_cycles.label(k): len(data.cycle_included(k)) for k in range(1, vim_cycles.K_MAX + 1)}},
        "limits": ["signed differences only; no approved agreement threshold",
                   "every point is a separate KK-VIM run; runs that did not end as requested are kept in the table and are "
                   "not points of the curve", "the series is not a scientific acceptance and does not change any status"],
        "files": {name: {"file": path.name, "sha256": _sha256(path)} for name, path in paths.items()},
    }
    paths["manifest"] = folder / MANIFEST_FILE
    paths["manifest"].write_text(json.dumps(export._plain(manifest), ensure_ascii=False, indent=2), encoding="utf-8")
    return paths
