"""Reopening a saved calculation (R6 workflow): a RunOutcome read from a result folder, never recomputed. Qt-free.

Two folder kinds are read (the files are only read, never changed):

- a package of the GUI (``backend.export.write_result``: manifest.json of schema 1 or 2, history.csv and the field file
  of its role; the «Сохранить расчёт и отчёт…» package and the report's data/ folder are of this kind);
- a folder of the CLI (``python -m kkvim run``, ``run-ssss``, ``run-cccc-n1``, ``run-cccc-n23``, ``run-cccc-n46``:
  result.json and the field file it names).

The outcome keeps the saved statuses, field role and marks unchanged (a diagnostic field stays diagnostic, an accepted
R3 field stays accepted); the field and w_bar are the saved values (the CSV holds repr of the solver values). The half-
step history is read as saved (numbers; no recomputation). Moments are then derived by kkvim.moments like for a fresh
run (``backend.moments_view``). A folder without a field is refused: there is nothing to analyse.
"""

import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from kkvim_gui.backend import export
from kkvim_gui.backend.outcome import FIELD_ROLES, ORIGIN_SOLVER, Column, RunOutcome, frozen_copy
from kkvim_gui.backend.request import RunRequest

ROLE_OF_CLI = {"converged": "accepted", "accepted": "accepted", "research": "research", "diagnostic": "diagnostic"}


class SavedRunError(ValueError):
    """The folder is not a readable saved calculation (the message is for the user)."""


def _value(entry):
    return entry["value"] if isinstance(entry, dict) else entry


def _rows(path: Path):
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line and not line.startswith("#")]
    return list(csv.reader(lines))


def _number(text):
    if text in ("", None):
        return None
    if text in ("true", "false"):
        return text == "true"
    try:
        return int(text)
    except ValueError:
        try:
            return float(text)
        except ValueError:
            return text


def read_field(path: Path, n_x: int, n_y: int):
    rows = _rows(path)
    header = rows[0]
    try:
        ci, cj, cx, cy, cw = (header.index(k) for k in ("i", "j", "x [m]", "y [m]", "w [m]"))
    except ValueError as exc:
        raise SavedRunError(f"неизвестные столбцы файла поля {path.name}") from exc
    cb = header.index("w_bar [1]") if "w_bar [1]" in header else None
    W, B = np.full((n_x + 1, n_y + 1), np.nan), np.full((n_x + 1, n_y + 1), np.nan)
    x, y = np.full(n_x + 1, np.nan), np.full(n_y + 1, np.nan)
    for row in rows[1:]:
        i, j = int(row[ci]), int(row[cj])
        W[i, j] = float(row[cw])
        if cb is not None:
            B[i, j] = float(row[cb])
        x[i], y[j] = float(row[cx]), float(row[cy])
    if np.isnan(W).any() or np.isnan(x).any() or np.isnan(y).any():
        raise SavedRunError(f"файл поля {path.name} не содержит всех узлов сетки {n_x} × {n_y}")
    return x, y, W, (None if cb is None else B)


def read_history(path: Path):
    """Columns and rows of a saved history.csv (numbers as saved)."""
    if path is None or not path.is_file():
        return (), ()
    rows = _rows(path)
    if not rows:
        return (), ()
    columns = []
    for name in rows[0]:
        key, unit = (name.split(" [", 1) + [""])[:2]
        unit = unit.rstrip("]")
        columns.append(Column(key, key, "" if unit == "1" else unit, "number"))
    history = tuple({c.key: _number(v) for c, v in zip(columns, row)} for row in rows[1:])
    for c_index, c in enumerate(columns):
        values = [h.get(c.key) for h in history]
        if values and not all(v is None or isinstance(v, (int, float)) for v in values):
            columns[c_index] = Column(c.key, c.key, "", "text")
    return tuple(columns), history


def _request(si, n_x, n_y, N, supports, entered=()):
    return RunRequest(tuple(entered), float(si["a"]), float(si["b"]), float(si["t"]), float(si["E"]), float(si["nu"]),
                      float(si["q"]), int(n_x), int(n_y), int(N), supports=supports)


def load(folder) -> RunOutcome:
    """The RunOutcome of a saved calculation folder (GUI package or CLI folder); SavedRunError if not readable."""
    folder = Path(folder)
    manifests = sorted(folder.glob("manifest*.json"))
    if manifests:
        return _load_package(folder, manifests[0])
    if (folder / "result.json").is_file():
        return _load_cli(folder)
    raise SavedRunError(f"в папке {folder} нет manifest.json (пакет расчёта) или result.json (папка CLI)")


def _load_package(folder: Path, path: Path) -> RunOutcome:
    try:
        info = export.read_manifest(path)
    except ValueError as exc:
        raise SavedRunError(str(exc)) from exc
    doc = json.loads(path.read_text(encoding="utf-8"))
    if not info["field_file"]:
        raise SavedRunError("в пакете нет файла поля: анализировать нечего")
    inp = doc["input"]
    si = {k: _value(v) for k, v in inp["si"].items()}
    entered = tuple((e["key"], e["text"], e["unit"]) for e in inp.get("entered", []))
    request = _request(si, inp["grid"]["n_x"], inp["grid"]["n_y"], inp["N"], inp["supports"], entered)
    x, y, W, B = read_field(folder / info["field_file"], request.n_x, request.n_y)
    history_name = doc.get("files", {}).get("history")
    columns, history = read_history(folder / history_name if history_name else None)
    st, res = doc["statuses"], doc["results"]
    role = info["field_role"]
    if role not in FIELD_ROLES:
        raise SavedRunError(f"роль поля {role!r} не распознана")
    return RunOutcome(
        origin=doc.get("data_origin") or ORIGIN_SOLVER, request=request, solver=doc["solver"]["stage"],
        call=doc["solver"]["call"], status=st["solver_status"], reason=st.get("solver_reason") or "", error=st.get("error"),
        stop_kind=st["stop"]["kind"], stop_text=st["stop"]["text"], science_kind=st["scientific"]["kind"],
        science_text=st["scientific"]["text"], mark=st.get("mark"), D=_value(res["D"]), x=frozen_copy(x), y=frozen_copy(y),
        w=frozen_copy(W), w_bar=frozen_copy(B if B is not None else W * 0.0), s=int(res["s"]), columns=columns,
        history=history, parameters=tuple(doc["solver"].get("parameters", {}).items()),
        diagnostics=tuple(doc.get("diagnostics", {}).items()),
        notes=tuple(doc.get("notes", ())) + (f"Открыт сохранённый расчёт: {folder} (значения прочитаны, не пересчитаны)",),
        versions=doc["program"].get("versions", {}), route=doc["model"].get("route", request.route or ""),
        field_role=role, requested_completed=bool(res.get("requested_run_completed")),
        s_requested=res.get("half_steps", {}).get("requested"), decisions=tuple(doc["model"].get("decisions", ())),
        limitations=tuple(doc.get("limitations", ())))


def _load_cli(folder: Path) -> RunOutcome:
    doc = json.loads((folder / "result.json").read_text(encoding="utf-8"))
    field = doc.get("field") or {}
    if not isinstance(field, dict) or not field.get("file"):
        raise SavedRunError(f"у расчёта нет поля прогиба (статус {doc.get('status')}): анализировать нечего")
    model = doc.get("model") or {}
    supports = model.get("supports") or model.get("boundary") or doc.get("boundary_conditions")
    if supports not in ("SSSS", "CCCC"):
        raise SavedRunError(f"закрепление {supports!r} не поддерживается")
    inp = doc["input"]
    si = {k: _value(inp[k]) for k in ("a", "b", "t", "E", "nu", "q")}
    N = doc.get("N") or model.get("N") or 1
    request = _request(si, _value(inp["n_x"]), _value(inp["n_y"]), N, supports)
    x, y, W, B = read_field(folder / field["file"], request.n_x, request.n_y)
    columns, history = read_history(folder / "history.csv")
    role = ROLE_OF_CLI.get(field.get("role"), "diagnostic")
    status = doc.get("status", "")
    mark = doc.get("application_status") or (None if role == "accepted" else "RESEARCH_ONLY / PROVISIONAL / NOT_QUALIFIED")
    s = doc.get("half_steps") or doc.get("half_steps_completed") or field.get("half_step") or 0
    if isinstance(s, dict):
        s = s.get("completed", 0)
    return RunOutcome(
        origin=ORIGIN_SOLVER, request=request, solver=str(doc.get("schema", "kkvim CLI")), call="kkvim CLI (saved)",
        status=status, reason=doc.get("reason") or "", error=doc.get("error"),
        stop_kind="error" if status.startswith("ERROR") else ("met" if status == "CONVERGED" else "none"),
        stop_text=doc.get("status_note") or status, science_kind="success" if role == "accepted" else "warning",
        science_text=(model.get("acceptance") or mark or ""), mark=None if role == "accepted" else mark,
        D=_value(doc["D"]), x=frozen_copy(x), y=frozen_copy(y), w=frozen_copy(W),
        w_bar=frozen_copy(B if B is not None else W * 0.0), s=int(s), columns=columns, history=history,
        notes=(f"Открыта папка CLI: {folder} (значения прочитаны, не пересчитаны)",), route=request.route or "",
        field_role=role, requested_completed=role != "diagnostic")


def saved_record(outcome: RunOutcome, folder, run_id: int = 0):
    """A finished RunRecord around a loaded outcome (for the analysis window; nothing runs)."""
    from kkvim_gui.model.runs import RunRecord

    folder = Path(folder)
    try:
        started = datetime.fromtimestamp(folder.stat().st_mtime)
    except OSError:
        started = datetime.now()
    return RunRecord(run_id=run_id, started=started, method="saved", snapshot=None, project=folder.name,
                     status="finished", stop_criterion=outcome.stop_kind, s=outcome.s, data_origin=outcome.origin,
                     request=outcome.request, outcome=outcome, _elapsed_final=0.0)


def summary_rows(outcome: RunOutcome):
    """Input rows of the analysis window for a saved run (SI values as saved)."""
    r = outcome.request
    return [("Геометрия", f"a {r.a!r} m · b {r.b!r} m · h {r.t!r} m"), ("Материал", f"E {r.E!r} Pa · ν {r.nu!r}"),
            ("Нагрузка", f"равномерная, q {r.q!r} Pa"), ("Закрепление", r.supports),
            ("Метод", f"КК-ВИМ, N = {r.N} (сохранённый расчёт)"), ("Сетка", f"n_x {r.n_x} · n_y {r.n_y}")]
