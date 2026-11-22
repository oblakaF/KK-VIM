"""Figures of a solver result for the saved calculation package (PNG), drawn from the same RunOutcome.

``renderer`` returns the ``render(folder)`` callback of ``backend.report.write_package``: it builds the colour
field with isolines and the 3D view off screen (the same figure classes as the analysis window), grabs them and
saves PNG files. ``view_record`` describes the drawing for the manifest and the report. With ``cycles`` (a
``vim_cycles.CycleTable`` that has drawn states) it also draws the VIM-1 … VIM-3 graph, and with ``series`` (the finished
load series of the run) the load–deflection graph with every curve shown (GUI-VIM-CYCLES-123-01).
"""

from pathlib import Path

from kkvim_gui.analysis.results import FieldFigure
from kkvim_gui.analysis.surface import SurfaceFigure, auto_factor, exaggeration_text

SIZE = (1100, 820)


def view_record(outcome, factor=None) -> dict:
    factor = auto_factor(outcome.x, outcome.y, outcome.w) if factor is None else float(factor)
    return {"field_png": "colour field of w in mm on the result nodes with 10 isolines (no exaggeration)",
            "surface_png": ("axonometric colour-mapped surface of the stored field (viridis; colour = true w in mm, "
                            "independent of the factor) with a thin node mesh; drawn height = -factor * w"),
            "exaggeration_factor": factor, "exaggeration_text": exaggeration_text(factor),
            "values_scaled": False}


def renderer(session, outcome, factor=None, cycles=None, series=None):
    def render(folder):
        folder = Path(folder)
        suffix = "_TEST-DATA" if outcome.is_demo else ""
        out = {}
        figures = [("field_png", FieldFigure(session, outcome, isolines=True)),
                   ("surface_png", SurfaceFigure(session, outcome, factor=factor))]
        if cycles is not None and cycles.drawn:
            from kkvim_gui.analysis.cycles_tab import CycleFigure

            figures.append(("vim_cycles_png", CycleFigure(session, cycles)))
        if series is not None:
            from kkvim_gui.analysis.series import SeriesFigure

            figures.append(("load_series_png", SeriesFigure(session, series)))
        for key, figure in figures:
            figure.resize(*SIZE)
            figure.ensurePolished()
            figure.layout().activate()
            if isinstance(figure, SurfaceFigure):
                figure.set_factor(figure.factor)  # range after the final size
            path = folder / f"{key.removesuffix('_png')}{suffix}.png"
            # the load series figure is saved without its Snap / projection controls and overlay (GUI_DESIGN.md 16.7)
            image = figure.export_pixmap() if hasattr(figure, "export_pixmap") else figure.grab()
            if not image.save(str(path), "PNG"):
                raise OSError(f"figure not saved: {path}")
            figure.deleteLater()
            out[key] = path
        return out

    return render
