"""«Собрать отчёт…»: the report folder of one calculation (GUI_DESIGN.md 18, GUI-D18; P7). Qt-free.

The interface builds a ``ReportPlan``: the chosen components, each with the function that writes its file(s); the
figures are drawn by ``figures.publication`` and the combined PDF by ``figures.report_pdf`` (Qt), the tables by the
existing writers of the calculation package (``backend.export``, ``backend.report``, ``backend.vim_cycles``) and
``backend.chart_export``. ``build`` then:

* refuses an existing non-empty folder (18.3: nothing is overwritten) and writes everything into one new folder:
  ``report[_TEST-DATA].pdf``, ``figures/``, ``tables/``, ``data/`` (the existing raw JSON + CSV export of the run) and
  ``manifest.json``;
* writes only components that are available; the unavailable ones and why are listed in the manifest (18.3 R3);
* records in ``manifest.json``: the run identifier, the input parameters in SI, program and dependency versions, git
  commit when known, the three statuses (execution, stopping, scientific) and the marks, ``data_origin``, the figure
  language, the build date and every file with its component, format, resolution and SHA256 (R2);
* proves that the run data were only read: a digest of the outcome's arrays and history before and after (R5);
* XLSX is not written: it needs an optional dependency the owner has not approved (PROPOSED_CHANGES.md 1.4); the
  manifest states it, every other part of the report is built.

Nothing is computed here; demonstration data carry TEST DATA in every file name and in the manifest (R4).
"""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from kkvim_gui import program_version
from kkvim_gui.backend import export

SCHEMA = "kkvim-gui-report/1"
XLSX_STATUS = ("NOT_BUILT: XLSX needs an optional GUI dependency that is not approved (PROPOSED_CHANGES.md 1.4); "
               "the tables are written as CSV")
XLSX_RECOMMENDATION = ("openpyxl (MIT, pure Python, maintained, wheels for CPython 3.13/3.14) or XlsxWriter (BSD-2, write "
                       "only) — owner decision required before it is added to the optional group 'gui'")


@dataclass
class Item:
    """One component of the report: ``write(folder) -> [(path, format, dpi)]`` writes its files below ``folder``."""

    id: str
    kind: str  # "figure" | "table" | "data" | "pdf"
    title: str
    write: object = None
    available: bool = True
    reason: str = ""


@dataclass
class ReportPlan:
    outcome: object
    language: str
    items: list = field(default_factory=list)
    run_meta: dict = field(default_factory=dict)
    log: tuple = ()
    options: dict = field(default_factory=dict)
    title: str = "KK-VIM — отчёт расчёта"


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_digest(outcome) -> str:
    """SHA256 of the stored data of the run (arrays and history): equal before and after the build (R5)."""
    h = hashlib.sha256()
    for name in ("x", "y", "w", "w_bar"):
        value = getattr(outcome, name, None)
        h.update(name.encode())
        if value is not None:
            h.update(value.tobytes())
    h.update(json.dumps([dict(r) for r in outcome.history], default=repr, sort_keys=True).encode())
    h.update(json.dumps([list(p) for p in outcome.parameters], default=repr).encode())
    return h.hexdigest()


def folder_name(base, lang, test_data):
    return f"{base}_report_{lang}{'_TEST-DATA' if test_data else ''}"


def free_folder(parent, name):
    """A folder name under ``parent`` that does not exist yet (``name``, ``name_2``, …): never an existing folder."""
    parent = Path(parent)
    candidate, k = parent / name, 2
    while candidate.exists():
        candidate = parent / f"{name}_{k}"
        k += 1
    return candidate


def build(plan: ReportPlan, folder) -> dict:
    """Write the report of ``plan`` into the new folder ``folder``; returns {"folder", "manifest", "files"}."""
    outcome = plan.outcome
    before = source_digest(outcome)
    folder = export.new_folder(folder)  # FileExistsError for an existing non-empty folder: nothing is overwritten
    for sub in ("figures", "tables"):
        (folder / sub).mkdir(exist_ok=True)
    files = []
    chosen = [i for i in plan.items if i.available and i.write is not None]
    for item in chosen:
        for path, fmt, dpi in item.write(folder):
            path = Path(path)
            files.append({"path": path.relative_to(folder).as_posix(), "component": item.id, "kind": item.kind,
                          "title": item.title, "format": fmt, "dpi": dpi,
                          "language": plan.language if item.kind in ("figure", "pdf") else None,
                          "bytes": path.stat().st_size, "sha256": sha256(path)})
    after = source_digest(outcome)
    base = export.manifest(outcome, plan.run_meta, plan.log, {})
    data = {
        "schema": SCHEMA,
        "kind": "calculation_report",
        "run_id": plan.run_meta.get("run_id"),
        "data_origin": outcome.origin,
        "test_data": outcome.is_demo,
        "marks": base["marks"],
        "report": {
            "title": plan.title, "created": datetime.now().isoformat(timespec="seconds"),
            "builder": f"KK-VIM GUI {program_version()}", "figure_language": plan.language,
            "options": plan.options,
            "components": [{"id": i.id, "kind": i.kind, "title": i.title} for i in chosen],
            "unavailable": [{"id": i.id, "kind": i.kind, "title": i.title, "reason": i.reason}
                            for i in plan.items if not i.available],
            "xlsx": {"requested": bool(plan.options.get("xlsx")), "status": XLSX_STATUS,
                     "recommendation": XLSX_RECOMMENDATION},
            "figures": "PublicationRenderer (Qt QPainter): white background independent of the theme, visible curves "
                       "only, no interaction overlay; TEST DATA watermark for demonstration data",
        },
        "input": base["input"],
        "model": base["model"],
        "solver": base["solver"],
        "statuses": base["statuses"],
        "results": base["results"],
        "program": {**base["program"], "git_commit": outcome.versions.get("git_commit") if outcome.versions else None},
        "provenance": {"run": base["run"], "notes": base["notes"], "limitations": base["limitations"]},
        "files": files,
        "source_data": {"sha256_before": before, "sha256_after": after, "unchanged": before == after,
                        "definition": "SHA256 of x, y, w, w_bar, the half-step history and the parameters of the run"},
    }
    manifest = folder / "manifest.json"
    manifest.write_text(json.dumps(export._plain(data), ensure_ascii=False, indent=2), encoding="utf-8")
    if before != after:  # never expected: the report only reads the run
        raise RuntimeError("the data of the run changed while the report was built")
    return {"folder": folder, "manifest": manifest, "files": files}
