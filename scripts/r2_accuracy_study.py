"""R2, group B: accuracy study of KK-VIM (N = 1, square SSSS, uniform load) against the Navier reference.

QA research script (D-016, D-025, D-033, D-034; docs/NUMERICAL_SPEC.md 3.4, 11.4, 12, 16-18).
It only runs the qualified main package ``kkvim`` and the qualified references
``kkvim_reference.navier`` and ``kkvim_reference.parseval_recursion`` and compares them at the same
nodes; it changes neither of them. Group B has no PASS/FAIL thresholds: the differences are measured.

Reproduction (from the repository root):

    uv run python scripts/r2_accuracy_study.py

writes compact CSV tables, generated Markdown tables and SVG figures to docs/results/r2/ and regenerates
the AUTO blocks of docs/R2_ACCURACY_STUDY.md from those CSV files (the report's numerical tables have no
other source). Option --grids selects the main grids (default 20 40 80 160 320); --diagnostic-grid the
additional grid (default 640, 0 to skip); --out the output directory; --report the report ('' to skip).

The semi-analytic sequence (Parseval recursion) is used for every studied state s = 1, 2, 3, 4 and the
actual stopping half-step of the controller; it is an independent computation without the spatial
finite-difference approximation, but it has its own truncation and error accumulation (budgets
reported); it is not the limit of the sequence for s -> infinity.

Dimensionless case a = D = q = 1, so w = w_bar = w D / (q a^4) and Pi is in q^2 a^6 / D.
"""

import argparse
import csv
import hashlib
import json
import math
import platform
import re
import sys
from pathlib import Path

import numpy as np

from kkvim.controller import run_until_stopped
from kkvim_reference.navier import DELTA_W, M_R2, NavierPlate, deflection_tail_bound
from kkvim_reference.parseval_recursion import ParsevalRecursion, parseval_tails

A = D = Q = 1.0
NU = 0.3
MAIN_GRIDS = (20, 40, 80, 160, 320)
DIAGNOSTIC_GRID = 640
P_WINDOW = (1.9, 2.1)  # NUMERICAL_SPEC.md 12: Richardson only if the observed p is in this window
QUARTERS = ((1, 4, 1, 4), (1, 4, 1, 2), (1, 2, 1, 4))  # (x numerator, denominator, y numerator, denominator)


# ---------------------------------------------------------------------------------------------------
# Metrics (pure functions; checked on small independent arrays by tests/cross/test_r2_accuracy_study.py)
# ---------------------------------------------------------------------------------------------------


def trapezoid_weights(n_x, n_y, a=1.0, b=1.0):
    """Two-dimensional trapezoid weights on the (n_x + 1) x (n_y + 1) tensor nodes."""
    wx = np.full(n_x + 1, a / n_x)
    wx[[0, -1]] *= 0.5
    wy = np.full(n_y + 1, b / n_y)
    wy[[0, -1]] *= 0.5
    return np.outer(wx, wy)


def centre_index(n):
    if n % 2:
        raise ValueError("the centre is a node only for even n")
    return n // 2


def delta_centre(w, w_ref):
    """(w(centre) - w_ref(centre)) / w_ref(centre), signed (NUMERICAL_SPEC.md 17)."""
    c = centre_index(w.shape[0] - 1)
    return (w[c, c] - w_ref[c, c]) / w_ref[c, c]


def error_inf(w, w_ref):
    """max|w - w_ref| / max|w_ref| over the same nodes."""
    return float(np.max(np.abs(w - w_ref)) / np.max(np.abs(w_ref)))


def error_l2(w, w_ref, weights):
    """sqrt(sum weights (w - w_ref)^2 / sum weights w_ref^2)."""
    return float(math.sqrt(np.sum(weights * (w - w_ref) ** 2) / np.sum(weights * w_ref**2)))


def max_abs_difference(w, w_ref):
    """Largest |w - w_ref| and its node coordinates (x/a, y/a)."""
    d = np.abs(w - w_ref)
    i, j = np.unravel_index(int(np.argmax(d)), d.shape)
    return float(d[i, j]), i / (w.shape[0] - 1), j / (w.shape[1] - 1)


def restrict(w_fine):
    """Values of the grid-2n field at the nodes of grid n (common nodes, no interpolation)."""
    return w_fine[::2, ::2]


def grid_difference(w_n, w_2n):
    """E_disc of the pair (n, 2n): max over the common nodes |w_n - w_2n| / max|w_n|."""
    return float(np.max(np.abs(w_n - restrict(w_2n))) / np.max(np.abs(w_n)))


def self_convergence_order(w_n, w_2n, w_4n):
    """p_n = log2(max|w_n - w_2n| / max|w_2n - w_4n|) on the nodes of grid n (NUMERICAL_SPEC.md 12)."""
    w2 = restrict(w_2n)
    w4 = restrict(restrict(w_4n))
    return math.log2(np.max(np.abs(w_n - w2)) / np.max(np.abs(w2 - w4)))


def observed_order(e_n, e_2n):
    """p = log2(E(n) / E(2n))."""
    return math.log2(e_n / e_2n)


def richardson(w_n, w_2n):
    """w_RE = (4 w_2n - w_n) / 3 at the nodes of grid n (NUMERICAL_SPEC.md 12)."""
    return (4.0 * restrict(w_2n) - w_n) / 3.0


def asymmetry(w):
    """A_sym = max|w - w^T| / max|w| (square grids)."""
    return float(np.max(np.abs(w - w.T)) / np.max(np.abs(w)))


def energy_error(pi, pi_ref):
    """(Pi - Pi_ref) / |Pi_ref|."""
    return (pi - pi_ref) / abs(pi_ref)


def in_window(p):
    return p is not None and math.isfinite(p) and P_WINDOW[0] <= p <= P_WINDOW[1]


# ---------------------------------------------------------------------------------------------------
# Minimal SVG figures (no plotting dependency)
# ---------------------------------------------------------------------------------------------------

COLOURS = ("#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b", "#17becf")


def _fmt_tick(v, log):
    if log:
        e = math.floor(math.log10(v) + 1e-9)
        k = round(v / 10.0**e)
        return f"{k}e{e}"
    return f"{v:.4g}"


def svg_line_chart(path, series, *, title, xlabel, ylabel, xlog=False, ylog=False, hlines=(), width=720, height=440):
    """series: list of (label, xs, ys, dashed). hlines: list of (label, y)."""
    left, right, top, bottom = 90, 190, 40, 60
    pw, ph = width - left - right, height - top - bottom
    xs_all = [x for _, xs, _, _ in series for x in xs]
    ys_all = [y for _, _, ys, _ in series for y in ys] + [y for _, y in hlines]
    tx = (lambda v: math.log10(v)) if xlog else (lambda v: v)
    ty = (lambda v: math.log10(v)) if ylog else (lambda v: v)
    x0, x1 = min(map(tx, xs_all)), max(map(tx, xs_all))
    y0, y1 = min(map(ty, ys_all)), max(map(ty, ys_all))
    if x1 == x0:
        x1 = x0 + 1
    pad = 0.06 * (y1 - y0 if y1 > y0 else abs(y0) + 1)
    y0, y1 = y0 - pad, y1 + pad

    def px(v):
        return left + (tx(v) - x0) / (x1 - x0) * pw

    def py(v):
        return top + ph - (ty(v) - y0) / (y1 - y0) * ph

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" font-family="sans-serif" font-size="12">',
           '<rect width="100%" height="100%" fill="white"/>',
           f'<text x="{left + pw / 2}" y="22" text-anchor="middle" font-size="14">{title}</text>',
           f'<rect x="{left}" y="{top}" width="{pw}" height="{ph}" fill="none" stroke="#444"/>']
    if xlog:
        ticks_x = [k * 10.0**e for e in range(math.floor(x0), math.ceil(x1) + 1) for k in (1, 2, 5)
                   if x0 <= math.log10(k * 10.0**e) <= x1]
    else:
        ticks_x = sorted(set(xs_all))
    for t in ticks_x:
        out.append(f'<line x1="{px(t):.1f}" y1="{top + ph}" x2="{px(t):.1f}" y2="{top + ph + 5}" stroke="#444"/>')
        out.append(f'<text x="{px(t):.1f}" y="{top + ph + 18}" text-anchor="middle">{_fmt_tick(t, xlog)}</text>')
    if ylog:
        ticks_y = [k * 10.0**e for e in range(math.floor(y0), math.ceil(y1) + 1) for k in (1, 2, 5)
                   if y0 <= math.log10(k * 10.0**e) <= y1]
    else:
        ticks_y = [y0 + (y1 - y0) * k / 5 for k in range(6)]
    for t in ticks_y:
        out.append(f'<line x1="{left - 5}" y1="{py(t):.1f}" x2="{left + pw}" y2="{py(t):.1f}" stroke="#ddd"/>')
        out.append(f'<text x="{left - 8}" y="{py(t) + 4:.1f}" text-anchor="end">{_fmt_tick(t, ylog)}</text>')
    out.append(f'<text x="{left + pw / 2}" y="{height - 15}" text-anchor="middle">{xlabel}</text>')
    out.append(f'<text x="18" y="{top + ph / 2}" text-anchor="middle" transform="rotate(-90 18 {top + ph / 2})">{ylabel}</text>')
    legend_y = top + 10
    for k, (label, y) in enumerate(hlines):
        colour = "#555"
        out.append(f'<line x1="{left}" y1="{py(y):.1f}" x2="{left + pw}" y2="{py(y):.1f}" stroke="{colour}" stroke-dasharray="6,4"/>')
        out.append(f'<line x1="{left + pw + 12}" y1="{legend_y}" x2="{left + pw + 36}" y2="{legend_y}" stroke="{colour}" stroke-dasharray="6,4"/>')
        out.append(f'<text x="{left + pw + 42}" y="{legend_y + 4}">{label}</text>')
        legend_y += 18
    for k, (label, xs, ys, dashed) in enumerate(series):
        colour = COLOURS[k % len(COLOURS)]
        pts = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in zip(xs, ys))
        dash = ' stroke-dasharray="4,3"' if dashed else ""
        out.append(f'<polyline points="{pts}" fill="none" stroke="{colour}" stroke-width="1.8"{dash}/>')
        for x, y in zip(xs, ys):
            out.append(f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="3" fill="{colour}"/>')
        out.append(f'<line x1="{left + pw + 12}" y1="{legend_y}" x2="{left + pw + 36}" y2="{legend_y}" stroke="{colour}" stroke-width="1.8"{dash}/>')
        out.append(f'<text x="{left + pw + 42}" y="{legend_y + 4}">{label}</text>')
        legend_y += 18
    out.append("</svg>")
    Path(path).write_text("\n".join(out) + "\n", encoding="utf-8")


def svg_heatmap(path, field, *, title, stride, width=560, height=520):
    """Diverging map of a signed nodal field, every ``stride``-th node (exact node values, no interpolation)."""
    sub = field[::stride, ::stride]
    nx, ny = sub.shape
    left, top, size = 60, 40, 400
    cell = size / max(nx, ny)
    vmax = float(np.max(np.abs(sub))) or 1.0

    def colour(v):
        t = max(-1.0, min(1.0, v / vmax))
        if t >= 0:
            r, g, b = 255, int(255 * (1 - t)), int(255 * (1 - t))
        else:
            r, g, b = int(255 * (1 + t)), int(255 * (1 + t)), 255
        return f"#{r:02x}{g:02x}{b:02x}"

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" font-family="sans-serif" font-size="12">',
           '<rect width="100%" height="100%" fill="white"/>',
           f'<text x="{left + size / 2}" y="22" text-anchor="middle" font-size="14">{title}</text>']
    for i in range(nx):
        for j in range(ny):
            # x to the right, y upwards
            out.append(f'<rect x="{left + i * cell:.2f}" y="{top + (ny - 1 - j) * cell:.2f}" width="{cell + 0.05:.2f}" height="{cell + 0.05:.2f}" fill="{colour(sub[i, j])}"/>')
    out.append(f'<rect x="{left}" y="{top}" width="{nx * cell:.2f}" height="{ny * cell:.2f}" fill="none" stroke="#444"/>')
    out.append(f'<text x="{left + size / 2}" y="{top + size + 22}" text-anchor="middle">x/a (0 ... 1)</text>')
    out.append(f'<text x="20" y="{top + size / 2}" text-anchor="middle" transform="rotate(-90 20 {top + size / 2})">y/a (0 ... 1)</text>')
    bar_x = left + size + 25
    for k in range(101):
        v = vmax * (1 - 2 * k / 100)
        out.append(f'<rect x="{bar_x}" y="{top + k * size / 101:.2f}" width="16" height="{size / 101 + 0.3:.2f}" fill="{colour(v)}"/>')
    out.append(f'<text x="{bar_x + 20}" y="{top + 8}">{vmax:+.2e}</text>')
    out.append(f'<text x="{bar_x + 20}" y="{top + size / 2 + 4}">0</text>')
    out.append(f'<text x="{bar_x + 20}" y="{top + size}">{-vmax:+.2e}</text>')
    out.append("</svg>")
    Path(path).write_text("\n".join(out) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------------------------------

STATES = ("1", "2", "3", "4", "stop")  # half-step states of the study; "stop" = actual s of the controller
S_EXTENDED = 10  # half-steps of the extended recursion shown in the qualification table


def source_digest(root):
    """SHA-256 of the sources used (kkvim, kkvim_reference, this script): identifies the code version."""
    h = hashlib.sha256()
    files = sorted((root / "src").rglob("*.py")) + [Path(__file__).resolve()]
    for f in files:
        h.update(str(f.relative_to(root)).replace("\\", "/").encode())
        h.update(f.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()


def s_of(entry, state):
    """Actual half-step number of a state on a grid (the controller's s for "stop")."""
    return entry["run"].s if state == "stop" else int(state)


def compute(grids, diagnostic_grid):
    all_grids = tuple(grids) + ((diagnostic_grid,) if diagnostic_grid else ())
    navier = NavierPlate(a=A, q=Q, D=D, nu=NU)
    runs = {n: run_until_stopped(A, D, Q, n, n) for n in all_grids}
    s_max = max([4] + [run.s for run in runs.values()])
    recursion = ParsevalRecursion(a=A, q=Q, D=D, s_max=max(s_max, S_EXTENDED))
    data = {}
    for n in all_grids:
        run = runs[n]
        reference = navier.at_nodes(n, n, M=M_R2, diagnostics=False)
        data[n] = {
            "run": run,
            "fields": {s: run.history.field(s) for s in range(1, run.s + 1)},
            "w_N": reference.w_coef,
            "w_N_rounding": reference.rounding_w,
            "w_SA": {s: recursion.field_on_grid(s, n, n) for s in range(1, max(4, run.s) + 1)},
            "weights": trapezoid_weights(n, n),
        }
    return data, navier, recursion


def field_of(entry, state):
    return entry["fields"][s_of(entry, state)]


def sa_of(entry, state):
    return entry["w_SA"][s_of(entry, state)]


M_OWN = 4001  # truncation of the QA cross-check recursion (independent of the rule И-12 of the Reference)


def own_recursion(s_max, M=M_OWN):
    """QA cross-check: the semi-analytic sequence written here from NUMERICAL_SPEC.md 15.1-15.3 (ChV-17 .. ChV-19).

    c_m = 4F / (D m pi (A lam^4 + 2B lam^2 + C)), odd m; next A, B, C, F by Parseval; integral 2 sum c_m/lam_m.
    Returns per s: invariants of G_s, centre value of w_s and Pi_s = -q/2 I_X I_Y. Different organisation
    (NumPy arrays, fixed M) from the Reference; no Reference values are used.
    """
    m = np.arange(1, M + 1, 2, dtype=float)
    lam = m * math.pi / A
    sign = np.where(np.arange(m.size) % 2 == 0, 1.0, -1.0)  # sin(m pi / 2)
    G = (A / 2, math.pi**2 / (2 * A), math.pi**4 / (2 * A**3), 2 * Q * A / math.pi)
    centre = {0: 1.0}
    integral = {0: 2 * A / math.pi}
    out = {}
    for s in range(1, s_max + 1):
        a_, b_, c_, f_ = G
        coef = 4 * f_ / (D * m * math.pi * (a_ * lam**4 + 2 * b_ * lam**2 + c_))
        centre[s] = math.fsum(coef * sign)
        integral[s] = 2 * math.fsum(coef / lam)
        ix, iy = (s if s % 2 == 0 else s - 1), (s if s % 2 == 1 else s - 1)
        out[s] = {"invariants": (b_ / a_, c_ / a_, f_ * f_ / a_), "centre": centre[ix] * centre[iy],
                  "energy": -0.5 * Q * integral[ix] * integral[iy]}
        G = (A / 2 * math.fsum(coef**2), A / 2 * math.fsum(lam**2 * coef**2), A / 2 * math.fsum(lam**4 * coef**2), 2 * Q * math.fsum(coef / lam))
    return out


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        for row in rows:
            w.writerow([_cell(v) for v in row])


def _cell(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        if not math.isfinite(v):
            return "nan"
        return f"{v:.12e}"
    if v is None:
        return ""
    return str(v)


def read_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _escape_cell(text):
    """A literal '|' inside a Markdown table cell must be escaped."""
    return str(text).replace("|", "\\|")


def md_table(header, rows):
    lines = ["| " + " | ".join(_escape_cell(h) for h in header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(_escape_cell(_md(v)) for v in row) + " |")
    return "\n".join(lines)


def _md(v):
    if isinstance(v, float):
        if not math.isfinite(v):
            return "—"
        if v == 0.0:
            return "0"
        return f"{v:.4e}" if (abs(v) < 1e-3 or abs(v) >= 1e4) else f"{v:.6f}"
    if v is None:
        return "—"
    return str(v)


_SUP = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def ru(v, digits=4):
    """Number for the Russian report: decimal comma, Unicode minus, a·10ⁿ for small/large values."""
    if v is None or v == "":
        return "—"
    v = float(v)
    if not math.isfinite(v):
        return "—"
    if v == 0.0:
        return "0"
    sign = "−" if v < 0 else ""
    a = abs(v)
    e = math.floor(math.log10(a))
    if -3 <= e <= 0:
        text = f"{a:.{max(digits - 1 - e, 0)}f}"
        return sign + text.replace(".", ",")
    mant = a / 10.0**e
    text = f"{mant:.{digits - 1}f}"
    if text.startswith("10"):
        e += 1
        text = f"{a / 10.0**e:.{digits - 1}f}"
    return f"{sign}{text.replace('.', ',')}·10{str(e).translate(_SUP)}"


def signed(v, digits=4):
    s = ru(v, digits)
    return s if s.startswith("−") or s in ("0", "—") else "+" + s


# ---------------------------------------------------------------------------------------------------
# Study
# ---------------------------------------------------------------------------------------------------


def run_study(out_dir, grids=MAIN_GRIDS, diagnostic_grid=DIAGNOSTIC_GRID, report=None):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    data, navier, recursion = compute(grids, diagnostic_grid)
    all_grids = tuple(grids) + ((diagnostic_grid,) if diagnostic_grid else ())
    finest = max(grids)
    md = []

    # --- Navier reference accuracy -------------------------------------------------------------
    integral = navier.deflection_integral(M=M_R2)
    pi_N, pi_N_bound = integral.Pi_M, integral.Pi_tail_bound
    diffs = [(float(np.max(np.abs(field_of(data[n], st) - data[n]["w_N"]))), n, st) for n in all_grids for st in STATES]
    smallest = min(diffs)
    finest_stop = float(np.max(np.abs(field_of(data[finest], "stop") - data[finest]["w_N"])))
    navier_rows = [
        ("M", M_R2),
        ("w_N_centre", float(data[finest]["w_N"][finest // 2, finest // 2])),
        ("strict_truncation_bound_8_2_uniform", deflection_tail_bound(M_R2)),
        ("delta_w", DELTA_W),
        ("first_order_rounding_bound_w_N_nodes", data[finest]["w_N_rounding"]),
        ("smallest_max_abs_diff_all_grids_states", smallest[0]),
        ("smallest_at_grid", smallest[1]),
        ("smallest_at_state", smallest[2]),
        ("max_abs_diff_stop_finest_main_grid", finest_stop),
        ("finest_main_grid", finest),
        ("ratio_bound_to_smallest", deflection_tail_bound(M_R2) / smallest[0]),
        ("Pi_N", pi_N),
        ("Pi_N_truncation_bound", pi_N_bound),
        ("J_M", integral.J_M),
    ]
    write_csv(out / "navier_reference.csv", ["quantity", "value"], navier_rows)
    md.append("### Эталон Навье\n\n" + md_table(["Величина", "Значение"], navier_rows))

    # --- deflection errors -----------------------------------------------------------------------
    rows, quarter_rows = [], []
    for n in all_grids:
        e = data[n]
        for st in STATES:
            w, wN = field_of(e, st), e["w_N"]
            c = n // 2
            value, xm, ym = max_abs_difference(w, wN)
            s = s_of(e, st)
            rows.append((n, st, s, s // 2, float(w[c, c]), float(wN[c, c]), float(delta_centre(w, wN)),
                         error_inf(w, wN), error_l2(w, wN, e["weights"]), value, xm, ym, "diagnostic" if n == diagnostic_grid else "main"))
            for xn, xd, yn, yd in QUARTERS:
                i, j = n * xn // xd, n * yn // yd
                quarter_rows.append((n, st, f"({xn}/{xd}, {yn}/{yd})", float(w[i, j]), float(wN[i, j]), float((w[i, j] - wN[i, j]) / wN[i, j])))
    write_csv(out / "deflection_errors.csv",
              ["n", "s", "s_actual", "k", "w_h_centre", "w_N_centre", "delta_c", "E_inf", "E_2", "max_abs_diff", "x_at_max", "y_at_max", "grid"], rows)
    write_csv(out / "quarters.csv", ["n", "s", "point", "w_h", "w_N", "relative_deviation"], quarter_rows)
    md.append("### Отличие прогиба от Навье\n\n" + md_table(
        ["n", "s", "s факт.", "k", "w_h(центр)", "δ_c", "E_∞", "E_2", "max|w_h−w_N|", "(x, y) max"],
        [(r[0], r[1], r[2], r[3], r[4], r[6], r[7], r[8], r[9], f"({r[10]:.4f}, {r[11]:.4f})") for r in rows]))

    # --- semi-analytic states (no grid; values at the nodes of the finest main grid) ---------------
    sa_rows = []
    e_f = data[finest]
    own = own_recursion(max(S_EXTENDED, recursion.s_max))
    for st in STATES:
        s = s_of(e_f, st)
        wsa, wN = sa_of(e_f, st), e_f["w_N"]
        c = finest // 2
        en = recursion.energy(s)
        budget = recursion.halfstep(s).budget
        e_sa = energy_error(en.energy, pi_N)
        e_sa_unc = (abs(en.energy) * en.error + pi_N_bound) / abs(pi_N)
        sa_rows.append((st, s, s // 2, float(wsa[c, c]), float(delta_centre(wsa, wN)), error_inf(wsa, wN),
                        error_l2(wsa, wN, e_f["weights"]), en.energy, e_sa, e_sa_unc, budget.field_error,
                        abs(own[s]["energy"] - en.energy) / abs(en.energy), budget.M))
    write_csv(out / "sa_states.csv",
              ["s", "s_actual", "k", "w_SA_centre", "delta_c_SA", "E_inf_SA", "E_2_SA", "Pi_SA", "e_SA", "e_SA_uncertainty",
               "SA_field_error_budget", "own_recursion_energy_rel", "M"], sa_rows)

    # --- extended recursion: budgets, energy and change identity (В-24) ------------------------------
    ext_rows = []
    for s in range(1, S_EXTENDED + 1):
        h, b, en = recursion.halfstep(s), recursion.halfstep(s).budget, recursion.energy(s)
        resolved = None if s == 1 else bool(en.change_rel > en.change_rel_error)
        inv_ref = recursion.coefficients(s).invariants()
        d_inv = max(abs(x - y) / abs(y) for x, y in zip(inv_ref, own[s]["invariants"]))
        d_centre = abs(recursion.centre_value(s) - own[s]["centre"]) / abs(own[s]["centre"])
        d_energy = abs(en.energy - own[s]["energy"]) / abs(own[s]["energy"])
        within = bool(d_inv <= b.invariant_error and d_centre <= b.field_error and d_energy <= en.error)
        ext_rows.append((s, h.k, h.variable, b.M, b.M_rule, b.input_error, b.coefficient_error, b.field_error, b.invariant_error,
                         b.integral_error, en.error, b.valid, en.energy, (en.energy_full - en.energy) / abs(en.energy),
                         None if s == 1 else -en.delta, None if s == 1 else 0.5 * en.change_norm2,
                         en.change_rel, en.change_rel_error, resolved, d_inv, d_centre, d_energy, within))
    write_csv(out / "parseval_extended.csv",
              ["s", "k", "variable", "M", "M_rule", "rho", "r_c", "field_error", "invariant_error", "integral_error", "energy_error",
               "budget_valid", "Pi_s", "Pi_full_minus_Pi_rel", "Pi_prev_minus_Pi_by_subtraction", "half_change_norm2",
               "change_rel", "change_rel_error_bound", "change_resolved", "own_invariants_rel", "own_centre_rel", "own_energy_rel",
               "own_within_budgets"], ext_rows)

    # --- behaviour over half-steps (n = 160, 320) --------------------------------------------------
    step_rows = []
    for n in [g for g in (160, 320) if g in data]:
        e = data[n]
        run = e["run"]
        changes = {ch.s: ch for ch in run.changes}
        for s in range(1, run.s + 1):
            w, wN = e["fields"][s], e["w_N"]
            ch, chk = changes.get(s), run.energy_checks[s - 1]
            prev = e["fields"][s - 1][n // 2, n // 2] if s > 1 else None
            wsa = e["w_SA"][s]
            step_rows.append((n, s, s // 2, float(w[n // 2, n // 2]), None if prev is None else float(w[n // 2, n // 2] - prev),
                              float(delta_centre(w, wN)), float(delta_centre(wsa, wN)), error_inf(w, wN), error_l2(w, wN, e["weights"]),
                              None if ch is None else ch.eps_inf, None if ch is None else ch.eps_E, None if ch is None else ch.theta,
                              chk.P_now, chk.status))
    write_csv(out / "halfsteps_n160_n320.csv",
              ["n", "s", "k", "w_h_centre", "centre_change", "delta_c", "delta_c_SA", "E_inf", "E_2", "eps_inf", "eps_E", "theta",
               "Pi_h", "energy_status"], step_rows)
    status_rows = [(n, data[n]["run"].s, data[n]["run"].s // 2, data[n]["run"].status) for n in all_grids]
    write_csv(out / "stopping.csv", ["n", "s_stop", "k_stop", "status"], status_rows)

    # --- grid convergence and Richardson --------------------------------------------------------------
    conv_rows, re_rows = [], []
    for st in STATES:
        for idx, n in enumerate(all_grids[:-1]):
            n2 = all_grids[idx + 1]
            if n2 != 2 * n:
                continue
            w_n, w_2n = field_of(data[n], st), field_of(data[n2], st)
            e_disc = grid_difference(w_n, w_2n)
            p_self = self_convergence_order(w_n, w_2n, field_of(data[4 * n], st)) if 4 * n in data else None
            p_ref = observed_order(error_inf(w_n, sa_of(data[n], st)), error_inf(w_2n, sa_of(data[n2], st)))
            applicable = in_window(p_ref)
            conv_rows.append((st, n, n2, e_disc, p_self, p_ref, applicable))
            if applicable:
                w_re = richardson(w_n, w_2n)
                wN, wsa = data[n]["w_N"], sa_of(data[n], st)
                c = n // 2
                re_rows.append((st, n, n2, float(w_2n[n2 // 2, n2 // 2]), float(w_re[c, c]), float(wsa[c, c]), float(wN[c, c]),
                                float(delta_centre(w_2n[::2, ::2], wN)), float(delta_centre(w_re, wN)), float(delta_centre(wsa, wN)),
                                float((w_re[c, c] - wsa[c, c]) / wsa[c, c]), error_inf(w_re, wsa), error_inf(w_2n[::2, ::2], wN), error_inf(w_re, wN)))
    write_csv(out / "grid_convergence.csv",
              ["s", "n", "2n", "E_disc_max_over_max_w_n", "p_self", "p_against_SA", "richardson_applicable"], conv_rows)
    write_csv(out / "richardson.csv",
              ["s", "n", "2n", "w_2n_centre", "w_RE_centre", "w_SA_centre", "w_N_centre", "delta_c_2n", "delta_c_RE", "delta_c_SA",
               "RE_minus_SA_centre_rel", "E_inf_RE_vs_SA", "E_inf_2n_on_nodes_n", "E_inf_RE"], re_rows)

    # --- energy -------------------------------------------------------------------------------------
    energy_rows = []
    for n in all_grids:
        run = data[n]["run"]
        for st in STATES:
            s = s_of(data[n], st)
            pi_h = run.energy_checks[s - 1].P_now
            energy_rows.append((n, st, s, pi_h, energy_error(pi_h, pi_N)))
    write_csv(out / "energy_discrete.csv", ["n", "s", "s_actual", "Pi_h", "e_h"], energy_rows)
    energy_re_rows = []
    pi_by = {(r[0], r[1]): r[3] for r in energy_rows}
    e_sa_by = {r[0]: r[8] for r in sa_rows}
    for st in STATES:
        for n in all_grids:
            if (2 * n, st) in pi_by and (4 * n, st) in pi_by:
                d1 = pi_by[(n, st)] - pi_by[(2 * n, st)]
                d2 = pi_by[(2 * n, st)] - pi_by[(4 * n, st)]
                p = math.log2(d1 / d2) if d2 != 0 and d1 / d2 > 0 else float("nan")
                ok = in_window(p)
                pi_re = (4 * pi_by[(2 * n, st)] - pi_by[(n, st)]) / 3 if ok else None
                e_re = energy_error(pi_re, pi_N) if ok else None
                energy_re_rows.append((st, n, 2 * n, p, ok, pi_re, e_re, None if e_re is None else (e_re - e_sa_by[st]) / e_sa_by[st]))
    write_csv(out / "energy_extrapolation.csv", ["s", "n", "2n", "p_energy", "applicable", "Pi_RE", "e_RE", "e_RE_minus_e_SA_rel"], energy_re_rows)

    # --- error decomposition (D-016) ------------------------------------------------------------------
    dec_rows = []
    for n in all_grids:
        e = data[n]
        wN = e["w_N"]
        scale = float(np.max(np.abs(wN)))
        c = n // 2
        for st in STATES:
            w, wsa = field_of(e, st), sa_of(e, st)
            disc, cont = w - wsa, wsa - wN
            dec_rows.append((n, st, s_of(e, st), "exact (SA)", float(np.max(np.abs(disc)) / scale), float(np.max(np.abs(cont)) / scale),
                             float(disc[c, c] / wN[c, c]), float(cont[c, c] / wN[c, c]), error_l2(w, wsa, e["weights"]), error_l2(wsa, wN, e["weights"]),
                             float(np.max(np.abs(cont)) / np.max(np.abs(disc)))))
    for st in ("4", "stop"):
        for n in all_grids:
            if 2 * n not in data or not any(r[0] == st and r[1] == n and r[6] for r in conv_rows):
                continue
            w_n = field_of(data[n], st)
            w_re = richardson(w_n, field_of(data[2 * n], st))
            wN = data[n]["w_N"]
            scale = float(np.max(np.abs(wN)))
            c = n // 2
            dec_rows.append((n, st, s_of(data[n], st), "diagnostic (Richardson)", float(np.max(np.abs(w_n - w_re)) / scale),
                             float(np.max(np.abs(w_re - wN)) / scale), float((w_n - w_re)[c, c] / wN[c, c]), float((w_re - wN)[c, c] / wN[c, c]),
                             error_l2(w_n, w_re, data[n]["weights"]), error_l2(w_re, wN, data[n]["weights"]),
                             float(np.max(np.abs(w_re - wN)) / np.max(np.abs(w_n - w_re)))))
    write_csv(out / "error_decomposition.csv",
              ["n", "s", "s_actual", "basis", "disc_inf_over_max_wN", "cont_inf_over_max_wN", "disc_centre_rel", "cont_centre_rel",
               "disc_L2_rel", "cont_L2_rel", "ratio_cont_to_disc"], dec_rows)

    # --- asymmetry ------------------------------------------------------------------------------------
    asym_rows = [(n, st, s_of(data[n], st), asymmetry(field_of(data[n], st)), asymmetry(sa_of(data[n], st))) for n in all_grids for st in STATES]
    write_csv(out / "asymmetry.csv", ["n", "s", "s_actual", "A_sym", "A_sym_SA"], asym_rows)

    # --- figures ----------------------------------------------------------------------------------------
    fig_c, fig_d = [], []
    for n in ([g for g in (80, 160, 320) if g in data] or list(grids)):
        run = data[n]["run"]
        ss = list(range(1, run.s + 1))
        fig_c.append((f"KK-VIM n = {n}", ss, [float(data[n]["fields"][s][n // 2, n // 2]) for s in ss], False))
        fig_d.append((f"n = {n}", ss, [float(delta_centre(data[n]["fields"][s], data[n]["w_N"])) for s in ss], False))
    ss = list(range(1, data[finest]["run"].s + 1))
    fig_c.append(("semi-analytic (no grid)", ss, [float(data[finest]["w_SA"][s][finest // 2, finest // 2]) for s in ss], True))
    fig_d.append(("semi-analytic (no grid)", ss, [float(delta_centre(data[finest]["w_SA"][s], data[finest]["w_N"])) for s in ss], True))
    wNc = float(data[finest]["w_N"][finest // 2, finest // 2])
    svg_line_chart(out / "centre_vs_s.svg", fig_c, title="Centre deflection w_bar(1/2, 1/2) over half-steps",
                   xlabel="half-step s", ylabel="w D/(q a^4)", hlines=[("Navier, M = 1279", wNc)])
    svg_line_chart(out / "delta_c_vs_s.svg", fig_d, title="Relative centre difference delta_c from Navier",
                   xlabel="half-step s", ylabel="delta_c", hlines=[("0", 0.0)])
    hs = [1.0 / n for n in grids]
    err_series = [(f"E_inf, s = {st}", hs, [next(r[7] for r in rows if r[0] == n and r[1] == st) for n in grids], False) for st in STATES]
    err_series += [(f"E_2, s = {st}", hs, [next(r[8] for r in rows if r[0] == n and r[1] == st) for n in grids], True) for st in ("1", "stop")]
    svg_line_chart(out / "errors_vs_h.svg", err_series, title="Differences from Navier against the grid step",
                   xlabel="h / a", ylabel="relative difference", xlog=True, ylog=True)
    en_series = [(f"e_h, s = {st}", hs, [next(r[4] for r in energy_rows if r[0] == n and r[1] == st) for n in grids], False) for st in STATES]
    svg_line_chart(out / "energy_vs_h.svg", en_series, title="Discrete energy measure e_h = (Pi_h - Pi_N)/|Pi_N|",
                   xlabel="h / a", ylabel="e_h", xlog=True, hlines=[(f"e_SA, s = {r[0]}", r[8]) for r in sa_rows if r[0] in ("1", "stop")])
    diff = field_of(data[finest], "stop") - data[finest]["w_N"]
    stride = max(1, finest // 40)
    svg_heatmap(out / "difference_field_stop.svg", diff, stride=stride,
                title=f"w_h - w_N after stopping, n = {finest} (every {stride}th node)")

    # --- metadata, full generated tables, report blocks ---------------------------------------------
    meta = {
        "case": {"a": A, "D": D, "q": Q, "nu_navier": NU, "M_navier": M_R2},
        "grids": list(grids), "diagnostic_grid": diagnostic_grid,
        "recursion_s_max": recursion.s_max,
        "python": platform.python_version(), "numpy": np.__version__,
        "source_sha256": source_digest(root),
        "stop_status": {str(n): [data[n]["run"].status, data[n]["run"].s] for n in all_grids},
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    full = ["# R2, группа B — сгенерированные таблицы\n", "Сформировано `scripts/r2_accuracy_study.py` по CSV этого каталога; не редактировать вручную.\n"]
    for name, block in report_blocks(out).items():
        full.append(f"## {name}\n\n{block}\n")
    (out / "tables.md").write_text("\n".join(full), encoding="utf-8")
    if report is not None:
        inject_report(report, report_blocks(out))
    return data


# ---------------------------------------------------------------------------------------------------
# Report blocks generated from the CSV files (the only source of the report's numbers)
# ---------------------------------------------------------------------------------------------------


def report_blocks(out_dir):
    """Markdown blocks of the report, built only from the CSV files in out_dir."""
    d = Path(out_dir)
    nav = {r["quantity"]: r["value"] for r in read_csv(d / "navier_reference.csv")}
    de = read_csv(d / "deflection_errors.csv")
    qu = read_csv(d / "quarters.csv")
    hs = read_csv(d / "halfsteps_n160_n320.csv")
    sa = read_csv(d / "sa_states.csv")
    ext = read_csv(d / "parseval_extended.csv")
    conv = read_csv(d / "grid_convergence.csv")
    rich = read_csv(d / "richardson.csv")
    edis = read_csv(d / "energy_discrete.csv")
    ere = read_csv(d / "energy_extrapolation.csv")
    dec = read_csv(d / "error_decomposition.csv")
    asy = read_csv(d / "asymmetry.csv")
    stop = read_csv(d / "stopping.csv")
    finest = nav["finest_main_grid"]
    grids = [r["n"] for r in stop]
    blocks = {}

    blocks["stopping"] = md_table(["n", "s остановки", "k", "статус"], [(r["n"], r["s_stop"], r["k_stop"], r["status"]) for r in stop])

    blocks["navier"] = md_table(["Величина", "Значение"], [
        ("w_N(центр), M = 1279", ru(nav["w_N_centre"], 11)),
        ("строгая равномерная граница усечения прогиба (NAVIER_REFERENCE_SPEC, 8.2)", ru(nav["strict_truncation_bound_8_2_uniform"]) + " ≤ δ_w = " + ru(nav["delta_w"], 3)),
        ("модель округления w_N в узлах (первого порядка, не строгая)", ru(nav["first_order_rounding_bound_w_N_nodes"], 3)),
        ("наименьшее max|w_h − w_N| среди всех сеток и состояний", f"{ru(nav['smallest_max_abs_diff_all_grids_states'])} (n = {nav['smallest_at_grid']}, s = {nav['smallest_at_state']})"),
        (f"max|w_h − w_N| после остановки на n = {finest}", ru(nav["max_abs_diff_stop_finest_main_grid"])),
        ("отношение границы эталона к наименьшему измеренному отличию", ru(nav["ratio_bound_to_smallest"], 2)),
        ("Π_N = Π_M (M = 1279)", ru(nav["Pi_N"], 13) + "; граница усечения " + ru(nav["Pi_N_truncation_bound"], 4)),
    ])

    def de_row(r):
        return (r["n"], r["s"], r["s_actual"], r["k"], ru(r["w_h_centre"], 9), signed(r["delta_c"], 5), ru(r["E_inf"], 5), ru(r["E_2"], 5),
                ru(r["max_abs_diff"]), f"({ru(r['x_at_max'], 3)}; {ru(r['y_at_max'], 3)})")

    head = ["n", "s", "s факт.", "k", "w_h(центр)", "δ_c", "E_∞", "E_2", "max|w_h − w_N|", "где (x/a; y/a)"]
    blocks["deflection_finest"] = md_table(head, [de_row(r) for r in de if r["n"] == finest])
    blocks["deflection_stop_all"] = md_table(head, [de_row(r) for r in de if r["s"] == "stop"])
    blocks["quarters"] = md_table(["s", "точка", "w_h", "w_N", "относительное отклонение"],
                                  [(r["s"], r["point"], ru(r["w_h"], 7), ru(r["w_N"], 7), signed(r["relative_deviation"], 4))
                                   for r in qu if r["n"] == finest and r["s"] in ("1", "stop")])

    blocks["halfsteps"] = md_table(
        ["s", "k", "w_h(центр)", "Δ центра", "δ_c (МКР)", "δ_c (SA)", "E_∞", "ε_∞", "ε_E", "θ", "Π_h", "энерг. статус"],
        [(r["s"], r["k"], ru(r["w_h_centre"], 11), signed(r["centre_change"], 3), signed(r["delta_c"], 5), signed(r["delta_c_SA"], 5),
          ru(r["E_inf"], 5), ru(r["eps_inf"], 3), ru(r["eps_E"], 3), ru(r["theta"], 3), ru(r["Pi_h"], 7), r["energy_status"])
         for r in hs if r["n"] == finest])

    blocks["sa_states"] = md_table(
        ["состояние", "s", "k", "w_SA(центр)", "δ_c^SA", "E_∞^SA", "E_2^SA", "бюджет поля SA"],
        [(r["s"], r["s_actual"], r["k"], ru(r["w_SA_centre"], 11), signed(r["delta_c_SA"], 5), ru(r["E_inf_SA"], 5), ru(r["E_2_SA"], 5),
          ru(r["SA_field_error_budget"], 2)) for r in sa])

    blocks["parseval_extended"] = md_table(
        ["s", "k", "полушаг", "M", "ρ_s", "бюджет поля", "бюджет инвариантов", "бюджет энергии", "Π_{s−1} − Π_s (вычитанием)",
         "½‖w_s − w_{s−1}‖² (без вычитания)", "‖Δw‖/‖w‖", "граница ошибки ‖Δw‖/‖w‖", "изменение установлено"],
        [(r["s"], r["k"], r["variable"], r["M"], ru(r["rho"], 2), ru(r["field_error"], 2), ru(r["invariant_error"], 2), ru(r["energy_error"], 2),
          signed(r["Pi_prev_minus_Pi_by_subtraction"], 3), ru(r["half_change_norm2"], 3), ru(r["change_rel"], 3), ru(r["change_rel_error_bound"], 2),
          {"true": "да", "false": "нет", "": "—"}[r["change_resolved"]]) for r in ext])
    blocks["parseval_own_check"] = md_table(
        ["s", "инварианты: |Reference − QA| / QA", "центр", "Π_s", "в пределах бюджетов Reference"],
        [(r["s"], ru(r["own_invariants_rel"], 2), ru(r["own_centre_rel"], 2), ru(r["own_energy_rel"], 2),
          {"true": "да", "false": "нет"}[r["own_within_budgets"]]) for r in ext])

    states = ["1", "2", "3", "4", "stop"]
    pairs = sorted({(int(r["n"]), int(r["2n"])) for r in conv})
    blocks["convergence"] = md_table(
        ["s"] + [f"E_disc ({a}, {b})" for a, b in pairs] + [f"p по SA ({a}, {b})" for a, b in pairs[-3:]],
        [[st] + [ru(next(r["E_disc_max_over_max_w_n"] for r in conv if r["s"] == st and int(r["n"]) == a), 4) for a, _ in pairs]
         + [ru(next(r["p_against_SA"] for r in conv if r["s"] == st and int(r["n"]) == a), 5) for a, _ in pairs[-3:]] for st in states])

    blocks["richardson"] = md_table(
        ["s", "пара", "w_2n(центр)", "w_RE(центр)", "w_SA(центр)", "(w_RE − w_SA)/w_SA в центре", "max|w_RE − w_SA|/max|w_SA|", "δ_c(RE)", "δ_c(SA)"],
        [(r["s"], f"({r['n']}, {r['2n']})", ru(r["w_2n_centre"], 11), ru(r["w_RE_centre"], 11), ru(r["w_SA_centre"], 11),
          signed(r["RE_minus_SA_centre_rel"], 2), ru(r["E_inf_RE_vs_SA"], 2), signed(r["delta_c_RE"], 5), signed(r["delta_c_SA"], 5))
         for r in rich if (r["n"], r["2n"]) == (str(int(finest) // 2), finest)])

    blocks["energy_discrete"] = md_table(["n"] + [f"s = {st}" for st in states],
                                         [[n] + [ru(next(r["e_h"] for r in edis if r["n"] == n and r["s"] == st), 4) for st in states] for n in grids])
    blocks["energy_sa"] = md_table(
        ["состояние", "s", "Π(w_SA)", "e_SA", "неопределённость e_SA", "Π: Reference против рекурсии QA (относит.)", "e_RE (160, 320)", "(e_RE − e_SA)/e_SA"],
        [(r["s"], r["s_actual"], ru(r["Pi_SA"], 13), ru(r["e_SA"], 6), ru(r["e_SA_uncertainty"], 2), ru(r["own_recursion_energy_rel"], 2),
          ru(next((x["e_RE"] for x in ere if x["s"] == r["s"] and x["n"] == str(int(finest) // 2)), None), 6),
          signed(next((x["e_RE_minus_e_SA_rel"] for x in ere if x["s"] == r["s"] and x["n"] == str(int(finest) // 2)), None), 2)) for r in sa])

    dec_grids = [g for g in ("20", "80", finest) if g in grids]
    blocks["decomposition"] = md_table(
        ["n", "s", "основание", "max|w_h − w_SA| / max|w_N|", "max|w_SA − w_N| / max|w_N|", "дискр. в центре", "непрер. в центре", "непрер./дискр."],
        [(r["n"], r["s"], "точное (SA)" if r["basis"] == "exact (SA)" else "диагностика (Ричардсон)", ru(r["disc_inf_over_max_wN"], 4), ru(r["cont_inf_over_max_wN"], 4),
          signed(r["disc_centre_rel"], 4), signed(r["cont_centre_rel"], 5), ru(r["ratio_cont_to_disc"], 3))
         for r in dec if r["n"] in dec_grids and (r["basis"] == "exact (SA)" or r["n"] == str(int(finest) // 2))])

    blocks["asymmetry"] = md_table(
        ["n"] + [f"s = {st}" for st in states] + ["SA, остановка"],
        [[n] + [ru(next(r["A_sym"] for r in asy if r["n"] == n and r["s"] == st), 3) for st in states]
         + [ru(next(r["A_sym_SA"] for r in asy if r["n"] == n and r["s"] == "stop"), 2)] for n in grids])

    fs = next(r for r in de if r["n"] == finest and r["s"] == "stop")
    s1 = next(r for r in de if r["n"] == finest and r["s"] == "1")
    sas = next(r for r in sa if r["s"] == "stop")
    re_stop = next(r for r in rich if r["s"] == "stop" and r["2n"] == finest)
    dstop = next(r for r in dec if r["n"] == finest and r["s"] == "stop" and r["basis"] == "exact (SA)")
    blocks["key_results"] = md_table(["Величина", "Значение"], [
        (f"s = 1 (метод Канторовича–Власова), n = {finest}: δ_c; E_∞", f"{signed(s1['delta_c'], 4)}; {ru(s1['E_inf'], 4)}"),
        (f"остановка (s = {fs['s_actual']}), n = {finest}: δ_c; E_∞; E_2", f"{signed(fs['delta_c'], 5)}; {ru(fs['E_inf'], 5)}; {ru(fs['E_2'], 5)}"),
        (f"полуаналитическое состояние s = {sas['s_actual']} (без сетки): δ_c^SA; E_∞^SA; E_2^SA",
         f"{signed(sas['delta_c_SA'], 5)}; {ru(sas['E_inf_SA'], 5)}; {ru(sas['E_2_SA'], 5)}"),
        ("энергетическая мера e_SA после остановки (непрерывная)", f"{ru(sas['e_SA'], 6)} ± {ru(sas['e_SA_uncertainty'], 2)}"),
        (f"дискретизация после остановки, n = {finest}: max|w_h − w_SA| / max|w_N|", ru(dstop["disc_inf_over_max_wN"], 4)),
        (f"отношение непрерывного отличия к дискретизации, n = {finest}", ru(dstop["ratio_cont_to_disc"], 3)),
        (f"Ричардсон ({int(finest) // 2}, {finest}) против SA после остановки, центр", signed(re_stop["RE_minus_SA_centre_rel"], 2)),
    ])
    return blocks


def inject_report(path, blocks):
    """Replace every <!-- AUTO:name --> ... <!-- /AUTO:name --> block of the report by the generated table."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    for name, block in blocks.items():
        start, end = f"<!-- AUTO:{name} -->", f"<!-- /AUTO:{name} -->"
        i, j = text.find(start), text.find(end)
        if i < 0 or j < i:
            raise KeyError(f"report block {name!r} is missing")
        text = text[: i + len(start)] + "\n" + block + "\n" + text[j:]
    path.write_text(text, encoding="utf-8")


_BLOCK = re.compile(r"<!-- AUTO:(\w+) -->\n(.*?)\n<!-- /AUTO:\1 -->", re.S)


def report_block_contents(path):
    """Blocks currently in the report (for the consistency test): only complete <!-- AUTO:name --> pairs."""
    return {name: body for name, body in _BLOCK.findall(Path(path).read_text(encoding="utf-8"))}


def main(argv=None):
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(root / "docs" / "results" / "r2"))
    parser.add_argument("--grids", type=int, nargs="+", default=list(MAIN_GRIDS))
    parser.add_argument("--diagnostic-grid", type=int, default=DIAGNOSTIC_GRID)
    parser.add_argument("--report", default=str(root / "docs" / "R2_ACCURACY_STUDY.md"),
                        help="report whose AUTO blocks are regenerated; '' to skip")
    args = parser.parse_args(argv)
    run_study(args.out, tuple(args.grids), args.diagnostic_grid or None, report=args.report or None)
    print(f"written to {args.out}")


if __name__ == "__main__":
    sys.exit(main())
