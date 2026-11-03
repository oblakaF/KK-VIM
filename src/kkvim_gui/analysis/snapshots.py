"""Figures of a solver result for the saved calculation package (PNG), drawn from the same RunOutcome.

``renderer`` returns the ``render(folder)`` callback of ``backend.report.write_package``: it builds the colour
field with isolines and the 3D view off screen (the same figure classes as the analysis window), grabs them and
saves PNG files. ``view_record`` describes the drawing for the manifest and the report.
"""

from pathlib import Path

from kkvim_gui.analysis.results import FieldFigure
from kkvim_gui.analysis.surface import SurfaceFigure, auto_factor, exaggeration_text

SIZE = (1100, 820)


def view_record(outcome, factor=None) -> dict:
    factor = auto_factor(outcome.x, outcome.y, outcome.w) if factor is None else float(factor)
    return {"field_png": "colour field of w in mm on the result nodes with 10 isolines (no exaggeration)",
            "surface_png": "axonometric wireframe of the stored field; drawn height = -factor * w",
            "exaggeration_factor": factor, "exaggeration_text": exaggeration_text(factor),
            "values_scaled": False}


def renderer(session, outcome, factor=None):
    def render(folder):
        folder = Path(folder)
        suffix = "_TEST-DATA" if outcome.is_demo else ""
        out = {}
        for key, figure in (("field_png", FieldFigure(session, outcome, isolines=True)),
                            ("surface_png", SurfaceFigure(session, outcome, factor=factor))):
            figure.resize(*SIZE)
            figure.ensurePolished()
            figure.layout().activate()
            if isinstance(figure, SurfaceFigure):
                figure.set_factor(figure.factor)  # range after the final size
            path = folder / f"{key.removesuffix('_png')}{suffix}.png"
            if not figure.grab().save(str(path), "PNG"):
                raise OSError(f"figure not saved: {path}")
            figure.deleteLater()
            out[key] = path
        return out

    return render
