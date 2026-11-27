"""``python -m kkvim moments DIR`` and ``--moments`` of every run command: R6 research moments of a saved result.

Contract: R6 review, section 8.4 (docs/R6_MOMENTS_IMPLEMENTATION.md). The folder of a run (``run``, ``run-ssss``,
``run-cccc-n1``, ``run-cccc-n23``, ``run-cccc-n46``) is read: ``result.json`` (input, D, supports, status, the field
block) and the field file it names. Two NEW files are written next to them:

- ``moments_research.csv`` - for a field of role "converged"/"accepted" or "research"; ``moments_diagnostic.csv`` - for
  a diagnostic field (the field of the last completed half-step of a run that did not end as requested). The moments
  of a diagnostic field are diagnostic too: never a result;
- ``moments.json`` - the separate R6 status, conventions, operators, steps, rounding model, the source field with the
  unchanged displacement status, the centre values and the sampled extrema.

The original files (result.json, the field, history, factors) are never changed or rewritten (the contract's
``result.json -> moments`` block is therefore kept in moments.json). Existing moment files are not replaced without
``--overwrite``. No field (no completed half-step) -> no file and exit code 7 with the reason. The moments are
RESEARCH / NOT_QUALIFIED whatever the status of the displacement result: an accepted deflection does not qualify
its derived moments or stresses.
"""

import csv
import io
import json
import math
import os
from pathlib import Path

import numpy as np

from kkvim import moments as mo

SCHEMA = "kkvim.cli.r6-moments/1"
MOMENTS_JSON = "moments.json"
MOMENTS_RESEARCH = "moments_research.csv"
MOMENTS_DIAGNOSTIC = "moments_diagnostic.csv"
MOMENT_OUTPUTS = (MOMENTS_JSON, MOMENTS_RESEARCH, MOMENTS_DIAGNOSTIC)  # = cli.MOMENT_FILES
EXIT_OK = 0
EXIT_MOMENTS_ERROR = 7  # the moments could not be computed or written (the reason is printed); the run is unchanged
COLUMNS = ("i", "j", "x [m]", "y [m]", "Mx [N]", "My [N]", "Mxy [N]", "Mx_bar [1]", "My_bar [1]", "Mxy_bar [1]",
           "M1 [N]", "M2 [N]", "alpha1 [rad]", "sigma_x_bottom [Pa]", "sigma_y_bottom [Pa]", "tau_xy_bottom [Pa]",
           "sigma_x_top [Pa]", "sigma_y_top [Pa]", "tau_xy_top [Pa]")
Z_COLUMNS = ("sigma_x_z [Pa]", "sigma_y_z [Pa]", "tau_xy_z [Pa]")
LIMITATIONS = (
    "RESEARCH / NOT_QUALIFIED: the accuracy of the moments and stresses is not qualified (R6 not accepted; R5 not accepted).",
    "Kirchhoff-Love plate theory: no transverse shear stresses tau_xz, tau_yz, no Q_x, Q_y, no sigma_z (BL-5); no "
    "three-dimensional accuracy (BL-4).",
    "CCCC edge curvature: rule A (2 f1 / h^2). Its second order on the computed field is observed for CCCC N = 1 "
    "(review 6.2), not proved for every N, grid or computed field.",
    "Agreement of deflections does not carry over to moments, nor moments to stresses (review 8.9).",
    "The rounding model is a model, not a bound (BL-6).",
)


class MomentsError(Exception):
    """The moments of a folder cannot be produced (the message is for the user)."""


def _value(entry):
    return entry["value"] if isinstance(entry, dict) else entry


def read_run(folder: Path) -> dict:
    """The inputs, supports, statuses and field of a saved run folder (nothing is changed)."""
    path = folder / "result.json"
    if not path.is_file():
        raise MomentsError(f"нет файла result.json в папке {folder}")
    doc = json.loads(path.read_text(encoding="utf-8"))
    model = doc.get("model") or {}
    supports = model.get("supports") or model.get("boundary") or doc.get("boundary_conditions")
    if isinstance(supports, dict):
        supports = supports.get("code") or supports.get("supports")
    if supports not in mo.BC_OF_SUPPORTS:
        raise MomentsError(f"закрепление {supports!r} не поддерживается (только SSSS и CCCC)")
    field = doc.get("field") or {}
    name = field.get("file") if isinstance(field, dict) else None
    if not name:
        raise MomentsError(f"у расчёта нет поля прогиба (статус {doc.get('status')}): моменты не вычисляются, "
                           "файл не создаётся")
    inputs = doc["input"]
    run = {key: float(_value(inputs[key])) for key in ("a", "b", "t", "E", "nu", "q")}
    run.update(n_x=int(_value(inputs["n_x"])), n_y=int(_value(inputs["n_y"])), supports=supports,
               D=float(_value(doc["D"])), status=doc.get("status"), field_file=name, field_role=field.get("role"),
               N=doc.get("N") or model.get("N"), schema=doc.get("schema"),
               application_status=doc.get("application_status") or doc.get("scientific_applicability")
               or (model.get("acceptance") if isinstance(model, dict) else None))
    run["W"], run["x"], run["y"] = read_field(folder / name, run["n_x"], run["n_y"])
    return run


def read_field(path: Path, n_x: int, n_y: int):
    """w [m] on all nodes from a field CSV (comment lines '#' skipped); the values are the repr of the solver values."""
    if not path.is_file():
        raise MomentsError(f"нет файла поля {path.name}")
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line and not line.startswith("#")]
    rows = list(csv.reader(lines))
    header = rows[0]
    try:
        ci, cj, cx, cy, cw = (header.index(k) for k in ("i", "j", "x [m]", "y [m]", "w [m]"))
    except ValueError as exc:
        raise MomentsError(f"неизвестные столбцы файла поля {path.name}: {header}") from exc
    W = np.full((n_x + 1, n_y + 1), np.nan)
    x, y = np.full(n_x + 1, np.nan), np.full(n_y + 1, np.nan)
    for row in rows[1:]:
        i, j = int(row[ci]), int(row[cj])
        W[i, j] = float(row[cw])
        x[i], y[j] = float(row[cx]), float(row[cy])
    if np.isnan(W).any():
        raise MomentsError(f"файл поля {path.name} не содержит всех узлов сетки {n_x} × {n_y}")
    return W, x, y


def _cell(value) -> str:
    if value is None:
        return ""
    value = float(value)
    return repr(value) if math.isfinite(value) else ""


def compute(run: dict, z=None) -> dict:
    """Moments, stresses and summaries of a read run (see ``read_run``)."""
    bc = mo.BC_OF_SUPPORTS[run["supports"]]
    M = mo.plate_moments(run["W"], run["a"], run["b"], run["D"], run["nu"], bc)
    surf = mo.surface_stresses(M, run["t"])
    stress_z = None if z is None else mo.stresses_at(M, run["t"], z)
    return {"M": M, "surfaces": surf, "z": z, "stress_z": stress_z}


def moments_csv(run: dict, res: dict) -> str:
    M, surf, q = res["M"], res["surfaces"], run["q"]
    bars = [M.bar(name, q) for name in ("Mx", "My", "Mxy")]
    diagnostic = run["field_role"] == "diagnostic"
    lines = [f"# R6 moments and stresses: {mo.STATUS}. Source field {run['field_file']} (role {run['field_role']}, "
             f"solver status {run['status']})."]
    if diagnostic:
        lines.append("# DIAGNOSTIC: moments of the field of the last completed half-step of a run that did not end as "
                     "requested. Not a result.")
    lines.append(f"# Conventions: {mo.CONVENTIONS}. CC edge rule {mo.CC_EDGE_RULE}.")
    header = list(COLUMNS) + (list(Z_COLUMNS) if res["stress_z"] is not None else [])
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)
    (bx, by, bt), (tx, ty, tt) = surf["bottom"], surf["top"]
    for i in range(M.Mx.shape[0]):
        for j in range(M.Mx.shape[1]):
            row = [i, j, _cell(run["x"][i]), _cell(run["y"][j]), _cell(M.Mx[i, j]), _cell(M.My[i, j]), _cell(M.Mxy[i, j])]
            row += [_cell(None if b is None else b[i, j]) for b in bars]
            row += [_cell(M.M1[i, j]), _cell(M.M2[i, j]), _cell(M.alpha1[i, j]), _cell(bx[i, j]), _cell(by[i, j]),
                    _cell(bt[i, j]), _cell(tx[i, j]), _cell(ty[i, j]), _cell(tt[i, j])]
            if res["stress_z"] is not None:
                row += [_cell(s[i, j]) for s in res["stress_z"]]
            writer.writerow(row)
    return "".join(f"{line}\n" for line in lines) + buffer.getvalue()


def moments_document(run: dict, res: dict, csv_name: str, argv=None) -> dict:
    M, surf = res["M"], res["surfaces"]
    x, y = run["x"], run["y"]
    fields = {"Mx": M.Mx, "My": M.My, "Mxy": M.Mxy, "M1": M.M1, "M2": M.M2,
              "sigma_x_bottom": surf["bottom"][0], "sigma_y_bottom": surf["bottom"][1], "tau_xy_bottom": surf["bottom"][2],
              "sigma_x_top": surf["top"][0], "sigma_y_top": surf["top"][1], "tau_xy_top": surf["top"][2]}
    units = {"Mx": "N", "My": "N", "Mxy": "N", "M1": "N", "M2": "N", "alpha1": "rad", "sigma": "Pa", "tau": "Pa",
             "x": "m", "y": "m", "z": "m", "M_bar": "1"}
    return {
        "schema": SCHEMA,
        "command": argv,
        "status": mo.STATUS,
        "role": "diagnostic" if run["field_role"] == "diagnostic" else "research",
        "file": csv_name,
        "conventions": mo.CONVENTIONS,
        "operators": mo.OPERATORS,
        "supports": run["supports"],
        "bc": mo.BC_OF_SUPPORTS[run["supports"]],
        "h_x": {"value": M.h_x, "unit": "m"},
        "h_y": {"value": M.h_y, "unit": "m"},
        "rounding_model": {"value": M.rounding_model, "unit": "N", "note": "a model, not a bound (review 8.6, BL-6)"},
        "inputs": {k: run[k] for k in ("a", "b", "t", "E", "nu", "q", "D", "n_x", "n_y")},
        "source_field": {"file": run["field_file"], "role": run["field_role"], "solver_status": run["status"],
                         "displacement_application_status": run["application_status"], "N": run["N"],
                         "schema": run["schema"],
                         "note": "the status of the displacement result is unchanged and is not transferred to the moments"},
        "stresses": {"surfaces": {"bottom": "z = +t/2", "top": "z = -t/2"}, "t": run["t"],
                     "z": res["z"], "z_columns": list(Z_COLUMNS) if res["z"] is not None else None},
        "dimensionless": "M_bar = M / (q a^2), a = side along x" if run["q"] != 0 else None,
        "units": units,
        "summary": {name: mo.field_summary(values, x, y) for name, values in fields.items()},
        "not_computed": {"tau_xz, tau_yz, Q_x, Q_y": "not specified (BL-5)", "equivalent stress": "not specified (OD-2)",
                         "rule B": "not a postprocessing option (OD-3)"},
        "limitations": list(LIMITATIONS),
        "authorization": "owner's restricted R6 research authorization (OD-7 of R6-MOMENTS-STRESSES-COMPLETE-SCIENTIFIC-REVIEW-01)",
    }


def postprocess(folder, z=None, overwrite=False, argv=None) -> dict:
    """Compute and write the moment files of the run in ``folder``; returns {name: path}. Raises MomentsError."""
    folder = Path(folder)
    run = read_run(folder)
    try:
        res = compute(run, z)
    except mo.MomentsInputError as exc:
        raise MomentsError(f"поле не принято постобработкой: {exc}") from exc
    name = MOMENTS_DIAGNOSTIC if run["field_role"] == "diagnostic" else MOMENTS_RESEARCH
    texts = {name: moments_csv(run, res),
             MOMENTS_JSON: json.dumps(moments_document(run, res, name, argv), ensure_ascii=False, indent=2,
                                      allow_nan=False) + "\n"}
    existing = [n for n in MOMENT_OUTPUTS if (folder / n).exists()]
    if existing and not overwrite:
        raise MomentsError(f"файлы моментов уже есть ({', '.join(existing)}); замена — только с --overwrite")
    from kkvim.cli import _write_temporary  # a new uniquely named temporary file (never an existing one)

    # Only the moment files are touched (cli.write_results would remove the other run outputs with overwrite).
    temporary = {}
    try:
        for n, text in texts.items():
            temporary[n] = _write_temporary(folder, text)
        for n, path in list(temporary.items()):
            os.replace(path, folder / n)
            del temporary[n]
    finally:
        for path in temporary.values():
            path.unlink(missing_ok=True)
    for stale in existing:
        if stale not in texts and stale in MOMENT_OUTPUTS:
            (folder / stale).unlink()
    return {n: folder / n for n in texts}


def add_parser(commands) -> None:
    p = commands.add_parser(
        "moments", help="R6: моменты и напряжения сохранённого расчёта (RESEARCH / NOT_QUALIFIED)",
        description="Вычислить изгибающие и крутящий моменты и напряжения в плоскости по полю прогиба сохранённого "
                    "расчёта (SSSS, CCCC). Исходные файлы не меняются; пишутся moments_*.csv и moments.json.")
    p.add_argument("folder", type=Path, help="папка результатов расчёта (result.json и файл поля)")
    p.add_argument("--z", type=float, metavar="VALUE", help="координата по толщине, m (вниз; −t/2 … t/2)")
    p.add_argument("--overwrite", action="store_true", help="заменить ранее вычисленные файлы моментов")


def run_command(folder, z, overwrite, argv, say) -> int:
    try:
        paths = postprocess(folder, z=z, overwrite=overwrite, argv=argv)
    except (MomentsError, OSError, ValueError, KeyError) as exc:
        say(f"Моменты не вычислены: {exc}")
        return EXIT_MOMENTS_ERROR
    say(f"R6: моменты и напряжения ({mo.STATUS}): {Path(folder).resolve()}")
    for name in paths:
        say(f"  {name}" + (" (ДИАГНОСТИКА, не результат)" if name == MOMENTS_DIAGNOSTIC else ""))
    return EXIT_OK
