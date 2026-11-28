"""«Данные CSV…» of an X–Y graph (GUI_DESIGN.md 8.4, 16.3–16.4; P5). Qt-free.

The file holds the stored values of the curves shown on the graph — read from the result records and arrays (solver
values in SI units at full precision), never the plotted coordinates, never a Snap position or any other interpolation and
never a display scaling (mm, kPa). Hidden curves are not written. Format independent of the language: decimal point,
separator ``,``, a header with the units in brackets (``[1]`` dimensionless); the first line of demonstration data is
``# TEST DATA``; the marks of the result (research, provisional, diagnostic) follow as ``#`` lines.
"""

import csv
import math
from pathlib import Path

NOT_INCLUDED = ("# not included: Snap and projection positions, interpolated values between nodes, display scaling, "
                "hidden curves")


def _cell(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(value) if math.isfinite(value) else ""
    if hasattr(value, "item"):  # numpy scalar
        return _cell(value.item())
    return str(value)


def header_cell(name, unit):
    return f"{name} [{unit}]" if unit else name


def write_curves_csv(path, columns, curves, comments=(), test_data=False) -> Path:
    """Write ``curves`` [(curve name, rows)] with ``columns`` [(name, unit)]: one row per stored value, curve first.

    ``comments`` are written as ``#`` lines after ``# TEST DATA`` (demonstration data) and before the header.
    """
    path = Path(path)
    lines = (["# TEST DATA"] if test_data else []) + [c if c.startswith("#") else f"# {c}" for c in comments]
    lines.append(NOT_INCLUDED)
    with path.open("w", encoding="utf-8", newline="") as fh:
        for line in lines:
            fh.write(line + "\n")
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(["curve", *(header_cell(n, u) for n, u in columns)])
        for name, rows in curves:
            for row in rows:
                writer.writerow([name, *(_cell(v) for v in row)])
    return path
