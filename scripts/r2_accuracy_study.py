"""R2, group B: accuracy study of KK-VIM (N = 1, square SSSS, uniform load) against the Navier reference.

QA research script (D-016, D-025, D-033, D-034; docs/NUMERICAL_SPEC.md 3.4, 11.4, 12, 16-18).
It only runs the qualified main package ``kkvim`` and the qualified references
``kkvim_reference.navier`` and ``kkvim_reference.parseval_recursion`` and compares them at the same
nodes; it changes neither of them. Group B has no PASS/FAIL thresholds: the differences are measured.

Reproduction (from the repository root):

    uv run python scripts/r2_accuracy_study.py

writes compact CSV tables, generated Markdown tables and SVG figures to docs/results/r2/.
Option --grids selects the main grids (default 20 40 80 160 320); --diagnostic-grid the additional
grid (default 640, 0 to skip); --out the output directory.

Dimensionless case a = D = q = 1, so w = w_bar = w D / (q a^4) and Pi is in q^2 a^6 / D.
"""

import argparse
import csv
import hashlib
import json
import math
import platform
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
S_FIXED = (1, 2, 3, 4)
S_REFERENCE = (1, 2, 3)  # semi-analytic recursion is implemented for s <= 3
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


def source_digest(root):
    """SHA-256 of the sources used (kkvim, kkvim_reference, this script): identifies the code version."""
    h = hashlib.sha256()
    files = sorted((root / "src").rglob("*.py")) + [Path(__file__).resolve()]
    for f in files:
        h.update(str(f.relative_to(root)).replace("\\", "/").encode())
        h.update(f.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()


def compute(grids, diagnostic_grid):
    all_grids = tuple(grids) + ((diagnostic_grid,) if diagnostic_grid else ())
    navier = NavierPlate(a=A, q=Q, D=D, nu=NU)
    recursion = ParsevalRecursion(a=A, q=Q, D=D)
    data = {}
    for n in all_grids:
        run = run_until_stopped(A, D, Q, n, n)
        fields = {s: run.history.field(s) for s in range(1, run.s + 1)}
        reference = navier.at_nodes(n, n, M=M_R2, diagnostics=False)
        data[n] = {
            "run": run,
            "fields": fields,
            "w_N": reference.w_coef,
            "w_N_rounding": reference.rounding_w,
            "w_SA": {s: recursion.field_on_grid(s, n, n) for s in S_REFERENCE},
            "weights": trapezoid_weights(n, n),
        }
    return data, navier, recursion


def label(s, n_stop):
    return "stop" if s == "stop" else str(s)


def field_of(entry, s):
    return entry["fields"][entry["run"].s] if s == "stop" else entry["fields"][s]


def continuous_energy(recursion):
    """Pi(w_SA(s)) = -q/2 (int X)(int Y) from the sine series (MATH_SPEC.md 7.4, В-23), with truncation bounds."""
    two_over_pi = 2.0 / math.pi
    integrals, bounds = {0: two_over_pi * recursion.x0_scale}, {0: 0.0}
    for s in S_REFERENCE:
        record = recursion.halfstep(s)
        step = record.step
        terms = [c * 2.0 / (m * math.pi) for m, c in zip(range(1, step.M + 1, 2), step.coefficients())]
        integrals[s] = math.fsum(terms)
        bounds[s] = parseval_tails(record.coefficients, D=D, ell=A, q=Q, M=step.M)[4] / abs(Q)
    out = {}
    for s in S_REFERENCE:
        ix, iy = (s if s % 2 == 0 else s - 1), (s if s % 2 == 1 else s - 1)
        IX, IY = integrals[ix], integrals[iy]
        product = IX * IY
        bound = abs(IX) * bounds[iy] + abs(IY) * bounds[ix] + bounds[ix] * bounds[iy]
        out[s] = {"pi": -0.5 * Q * product, "pi_bound": 0.5 * abs(Q) * bound,
                  "field_error_budget": recursion.halfstep(s).budget.field_error, "M": recursion.halfstep(s).budget.M}
    return out


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        for row in rows:
            w.writerow([_cell(v) for v in row])


def _cell(v):
    if isinstance(v, float):
        if not math.isfinite(v):
            return "nan"
        return f"{v:.12e}"
    if v is None:
        return ""
    return str(v)


def md_table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(_md(v) for v in row) + " |")
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


def run_study(out_dir, grids=MAIN_GRIDS, diagnostic_grid=DIAGNOSTIC_GRID):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    data, navier, recursion = compute(grids, diagnostic_grid)
    all_grids = tuple(grids) + ((diagnostic_grid,) if diagnostic_grid else ())
    labels = list(S_FIXED) + ["stop"]
    md = []

    # --- Navier reference accuracy -------------------------------------------------------------
    integral = navier.deflection_integral(M=M_R2)
    pi_N, pi_N_bound = integral.Pi_M, integral.Pi_tail_bound
    finest = max(grids)
    stop_diff_abs = float(np.max(np.abs(field_of(data[finest], "stop") - data[finest]["w_N"])))
    navier_rows = [
        ("M", M_R2), ("w_N centre (n = %d)" % finest, float(data[finest]["w_N"][finest // 2, finest // 2])),
        ("strict truncation bound 8.2, uniform", deflection_tail_bound(M_R2)), ("delta_w", DELTA_W),
        ("first-order rounding bound of w_N at nodes", data[finest]["w_N_rounding"]),
        ("smallest measured max|w_h - w_N| (stop, n = %d)" % finest, stop_diff_abs),
        ("ratio bound / smallest measured difference", deflection_tail_bound(M_R2) / stop_diff_abs),
        ("Pi_N = Pi_M (M = 1279)", pi_N), ("Pi_N truncation bound", pi_N_bound), ("J_M", integral.J_M),
    ]
    write_csv(out / "navier_reference.csv", ["quantity", "value"], navier_rows)
    md.append("### Эталон Навье\n\n" + md_table(["Величина", "Значение"], navier_rows))

    # --- deflection errors -----------------------------------------------------------------------
    rows, quarter_rows = [], []
    for n in all_grids:
        e = data[n]
        for s in labels:
            w, wN = field_of(e, s), e["w_N"]
            c = n // 2
            value, xm, ym = max_abs_difference(w, wN)
            s_actual = e["run"].s if s == "stop" else s
            rows.append((n, label(s, None), s_actual, s_actual // 2, float(w[c, c]), float(wN[c, c]), float(delta_centre(w, wN)),
                         error_inf(w, wN), error_l2(w, wN, e["weights"]), value, xm, ym, "diagnostic" if n == diagnostic_grid else "main"))
            for xn, xd, yn, yd in QUARTERS:
                i, j = n * xn // xd, n * yn // yd
                quarter_rows.append((n, label(s, None), f"({xn}/{xd}, {yn}/{yd})", float(w[i, j]), float(wN[i, j]),
                                     float((w[i, j] - wN[i, j]) / wN[i, j])))
    header = ["n", "s", "s_actual", "k", "w_h_centre", "w_N_centre", "delta_c", "E_inf", "E_2", "max_abs_diff", "x_at_max", "y_at_max", "grid"]
    write_csv(out / "deflection_errors.csv", header, rows)
    write_csv(out / "quarters.csv", ["n", "s", "point", "w_h", "w_N", "relative_deviation"], quarter_rows)
    md.append("### Отличие прогиба от Навье (центр, E_inf, E_2)\n\n" + md_table(
        ["n", "s", "s факт.", "k", "w_h(центр)", "δ_c", "E_∞", "E_2", "max|w_h−w_N|", "(x, y) max"],
        [(r[0], r[1], r[2], r[3], r[4], r[6], r[7], r[8], r[9], f"({r[10]:.4f}, {r[11]:.4f})") for r in rows]))
    md.append("### Четверти (n = %d)\n\n" % finest + md_table(
        ["s", "точка", "w_h", "w_N", "относит. откл."], [(r[1], r[2], r[3], r[4], r[5]) for r in quarter_rows if r[0] == finest]))

    # --- behaviour over half-steps (n = 160, 320) --------------------------------------------------
    step_rows = []
    for n in [g for g in (160, 320) if g in data]:
        e = data[n]
        run = e["run"]
        changes = {ch.s: ch for ch in run.changes}
        for s in range(1, run.s + 1):
            w, wN = e["fields"][s], e["w_N"]
            ch = changes.get(s)
            chk = run.energy_checks[s - 1]
            prev_centre = e["fields"][s - 1][n // 2, n // 2] if s > 1 else None
            step_rows.append((n, s, s // 2, float(w[n // 2, n // 2]),
                              None if prev_centre is None else float(w[n // 2, n // 2] - prev_centre),
                              float(delta_centre(w, wN)), error_inf(w, wN), error_l2(w, wN, e["weights"]),
                              None if ch is None else ch.eps_inf, None if ch is None else ch.eps_E,
                              None if ch is None else ch.theta, chk.P_now, chk.status))
    write_csv(out / "halfsteps_n160_n320.csv",
              ["n", "s", "k", "w_h_centre", "centre_change", "delta_c", "E_inf", "E_2", "eps_inf", "eps_E", "theta", "Pi_h", "energy_status"], step_rows)
    md.append("### По полушагам (n = 160, 320; остановка — последняя строка каждой сетки)\n\n" + md_table(
        ["n", "s", "k", "w_h(центр)", "Δ центра", "δ_c", "E_∞", "E_2", "ε_∞", "ε_E", "θ", "Π_h", "энерг. статус"], step_rows))
    status_rows = [(n, data[n]["run"].s, data[n]["run"].s // 2, data[n]["run"].status) for n in all_grids]
    write_csv(out / "stopping.csv", ["n", "s_stop", "k_stop", "status"], status_rows)

    # --- grid convergence and Richardson --------------------------------------------------------------
    conv_rows, re_rows = [], []
    for s in labels:
        for idx, n in enumerate(all_grids[:-1]):
            n2 = all_grids[idx + 1]
            if n2 != 2 * n:
                continue
            w_n, w_2n = field_of(data[n], s), field_of(data[n2], s)
            e_disc = grid_difference(w_n, w_2n)
            n4 = 4 * n
            p_self = self_convergence_order(w_n, w_2n, field_of(data[n4], s)) if n4 in data else None
            p_ref = None
            if s in S_REFERENCE:
                e_n = error_inf(w_n, data[n]["w_SA"][s])
                e_2n = error_inf(w_2n, data[n2]["w_SA"][s])
                p_ref = observed_order(e_n, e_2n)
            p_used = p_ref if s in S_REFERENCE else p_self
            applicable = in_window(p_used)
            conv_rows.append((label(s, None), n, n2, e_disc, p_self, p_ref, "reference" if s in S_REFERENCE else "self-convergence",
                              p_used, applicable))
            if applicable:
                w_re = richardson(w_n, w_2n)
                wN = data[n]["w_N"]
                c = n // 2
                re_rows.append((label(s, None), n, n2, float(w_2n[n2 // 2, n2 // 2]), float(w_re[c, c]), float(wN[c, c]),
                                float(delta_centre(w_2n[::2, ::2], wN)), float(delta_centre(w_re, wN)),
                                error_inf(w_2n[::2, ::2], wN), error_inf(w_re, wN)))
    write_csv(out / "grid_convergence.csv",
              ["s", "n", "2n", "E_disc_max_over_max_w_n", "p_self", "p_reference_s<=3", "p_basis", "p_used", "richardson_applicable"], conv_rows)
    write_csv(out / "richardson.csv",
              ["s", "n", "2n", "w_2n_centre", "w_RE_centre", "w_N_centre", "delta_c_2n", "delta_c_RE", "E_inf_2n_on_nodes_n", "E_inf_RE"], re_rows)
    md.append("### Сеточная сходимость (E_disc = max по общим узлам |w_n − w_2n| / max|w_n|)\n\n" + md_table(
        ["s", "n", "2n", "E_disc", "p (самосх.)", "p (по SA, s ≤ 3)", "основание p", "Ричардсон"],
        [(r[0], r[1], r[2], r[3], r[4], r[5], r[6], "да" if r[8] else "нет") for r in conv_rows]))
    md.append("### Ричардсон при фиксированном s (только при p ∈ [1,9; 2,1])\n\n" + md_table(
        ["s", "n", "2n", "w_2n(центр)", "w_RE(центр)", "w_N(центр)", "δ_c(2n)", "δ_c(RE)", "E_∞(2n)", "E_∞(RE)"], re_rows))

    # --- energy -------------------------------------------------------------------------------------
    cont = continuous_energy(recursion)
    energy_rows = []
    for n in all_grids:
        run = data[n]["run"]
        for s in labels:
            s_actual = run.s if s == "stop" else s
            pi_h = run.energy_checks[s_actual - 1].P_now
            energy_rows.append((n, label(s, None), s_actual, pi_h, energy_error(pi_h, pi_N)))
    write_csv(out / "energy_discrete.csv", ["n", "s", "s_actual", "Pi_h", "e_h"], energy_rows)
    cont_rows = [(s, cont[s]["pi"], cont[s]["pi_bound"], energy_error(cont[s]["pi"], pi_N),
                  (cont[s]["pi_bound"] + pi_N_bound) / abs(pi_N), cont[s]["field_error_budget"], cont[s]["M"]) for s in S_REFERENCE]
    write_csv(out / "energy_continuous_sa.csv",
              ["s", "Pi_SA", "Pi_SA_truncation_bound", "e_SA", "e_SA_truncation_bound", "SA_field_error_budget", "M"], cont_rows)
    # grid extrapolation of the discrete energy at fixed s (estimate only)
    energy_re_rows = []
    pi_by = {(r[0], r[1]): r[3] for r in energy_rows}
    for s in labels:
        ls = label(s, None)
        for n in all_grids:
            if (n * 2, ls) in pi_by and (n * 4, ls) in pi_by:
                d1 = pi_by[(n, ls)] - pi_by[(2 * n, ls)]
                d2 = pi_by[(2 * n, ls)] - pi_by[(4 * n, ls)]
                p = math.log2(abs(d1 / d2)) if d2 != 0 and d1 / d2 > 0 else float("nan")
                pi_re = (4 * pi_by[(2 * n, ls)] - pi_by[(n, ls)]) / 3
                energy_re_rows.append((ls, n, 2 * n, p, in_window(p), pi_re if in_window(p) else None,
                                       energy_error(pi_re, pi_N) if in_window(p) else None))
    write_csv(out / "energy_extrapolation.csv", ["s", "n", "2n", "p_energy", "applicable", "Pi_RE", "e_RE"], energy_re_rows)
    md.append(f"### Энергия\n\nΠ_N = {pi_N:.16e} (M = 1279), граница усечения Π_N: {pi_N_bound:.3e}.\n\n"
              "Дискретная мера e_h = (Π_h(w_h) − Π_N)/|Π_N|:\n\n" + md_table(
                  ["n"] + [f"s = {s}" for s in labels], [
                      [n] + [next(r[4] for r in energy_rows if r[0] == n and r[1] == label(s, None)) for s in labels] for n in all_grids]))
    md.append("Непрерывная мера по полуаналитической рекурсии, e_SA = (Π(w_SA) − Π_N)/|Π_N|:\n\n" + md_table(
        ["s", "Π(w_SA)", "граница усечения Π(w_SA)", "e_SA", "граница усечения e_SA", "бюджет поля SA", "M"], cont_rows))
    md.append("Экстраполяция дискретной энергии при фиксированном s (оценка, не непрерывный результат):\n\n" + md_table(
        ["s", "n", "2n", "p", "применима", "Π_RE", "e_RE"], energy_re_rows))

    # --- error decomposition (D-016) ------------------------------------------------------------------
    dec_rows = []
    for n in all_grids:
        e = data[n]
        wN = e["w_N"]
        scale = float(np.max(np.abs(wN)))
        c = n // 2
        for s in S_REFERENCE:
            w, wsa = e["fields"][s], e["w_SA"][s]
            disc, cont_part = w - wsa, wsa - wN
            dec_rows.append((n, str(s), "exact (SA)", float(np.max(np.abs(disc)) / scale), float(np.max(np.abs(cont_part)) / scale),
                             float(disc[c, c] / wN[c, c]), float(cont_part[c, c] / wN[c, c]), error_l2(w, wsa, e["weights"]),
                             error_l2(wsa, wN, e["weights"]), float(np.max(np.abs((w - wN) - disc - cont_part)))))
    for s in ["4", "stop"]:
        sv = 4 if s == "4" else "stop"
        for idx, n in enumerate(all_grids[:-1]):
            if 2 * n not in data:
                continue
            ok = any(r[0] == s and r[1] == n and r[8] for r in conv_rows)
            if not ok:
                continue
            w_n, w_re = field_of(data[n], sv), richardson(field_of(data[n], sv), field_of(data[2 * n], sv))
            wN = data[n]["w_N"]
            scale = float(np.max(np.abs(wN)))
            c = n // 2
            dec_rows.append((n, s, "estimate (Richardson)", float(np.max(np.abs(w_n - w_re)) / scale), float(np.max(np.abs(w_re - wN)) / scale),
                             float((w_n - w_re)[c, c] / wN[c, c]), float((w_re - wN)[c, c] / wN[c, c]), error_l2(w_n, w_re, data[n]["weights"]),
                             error_l2(w_re, wN, data[n]["weights"]), None))
    write_csv(out / "error_decomposition.csv",
              ["n", "s", "basis", "disc_inf_over_max_wN", "cont_inf_over_max_wN", "disc_centre_rel", "cont_centre_rel",
               "disc_L2_rel_to_SA_or_RE", "cont_L2_rel_to_wN", "bookkeeping_residual"], dec_rows)
    md.append("### Разделение источников (D-016): w_h − w_N = (w_h − w_SA) + (w_SA − w_N)\n\n" + md_table(
        ["n", "s", "основание", "дискр. max/max|w_N|", "непрер. max/max|w_N|", "дискр. центр", "непрер. центр"],
        [r[:7] for r in dec_rows]))

    # --- asymmetry ------------------------------------------------------------------------------------
    asym_rows = [(n, label(s, None), asymmetry(field_of(data[n], s))) for n in all_grids for s in labels]
    write_csv(out / "asymmetry.csv", ["n", "s", "A_sym"], asym_rows)
    md.append("### Асимметрия x ↔ y, A_sym = max|w − wᵀ| / max|w|\n\n" + md_table(
        ["n"] + [f"s = {s}" for s in labels], [[n] + [next(r[2] for r in asym_rows if r[0] == n and r[1] == label(s, None)) for s in labels] for n in all_grids]))

    # --- figures ----------------------------------------------------------------------------------------
    fig_series_c, fig_series_d = [], []
    for n in ([g for g in (80, 160, 320) if g in data] or list(grids)):
        run = data[n]["run"]
        ss = list(range(1, run.s + 1))
        centres = [float(data[n]["fields"][s][n // 2, n // 2]) for s in ss]
        fig_series_c.append((f"KK-VIM n = {n}", ss, centres, False))
        fig_series_d.append((f"n = {n}", ss, [float(delta_centre(data[n]["fields"][s], data[n]["w_N"])) for s in ss], False))
    wNc = float(data[finest]["w_N"][finest // 2, finest // 2])
    svg_line_chart(out / "centre_vs_s.svg", fig_series_c, title="Centre deflection w_bar(1/2, 1/2) over half-steps",
                   xlabel="half-step s", ylabel="w D/(q a^4)", hlines=[("Navier, M = 1279", wNc)])
    svg_line_chart(out / "delta_c_vs_s.svg", fig_series_d, title="Relative centre difference delta_c from Navier",
                   xlabel="half-step s", ylabel="delta_c", hlines=[("0", 0.0)])
    err_series = []
    for s in labels:
        hs = [1.0 / n for n in grids]
        err_series.append((f"E_inf, s = {label(s, None)}", hs, [next(r[7] for r in rows if r[0] == n and r[1] == label(s, None)) for n in grids], False))
    for s in (1, "stop"):
        hs = [1.0 / n for n in grids]
        err_series.append((f"E_2, s = {label(s, None)}", hs, [next(r[8] for r in rows if r[0] == n and r[1] == label(s, None)) for n in grids], True))
    svg_line_chart(out / "errors_vs_h.svg", err_series, title="Differences from Navier against the grid step",
                   xlabel="h / a", ylabel="relative difference", xlog=True, ylog=True)
    en_series = []
    for s in labels:
        hs = [1.0 / n for n in grids]
        en_series.append((f"e_h, s = {label(s, None)}", hs, [next(r[4] for r in energy_rows if r[0] == n and r[1] == label(s, None)) for n in grids], False))
    svg_line_chart(out / "energy_vs_h.svg", en_series, title="Discrete energy measure e_h = (Pi_h - Pi_N)/|Pi_N|",
                   xlabel="h / a", ylabel="e_h", xlog=True,
                   hlines=[(f"e_SA, s = {r[0]}", r[3]) for r in cont_rows])
    diff = field_of(data[finest], "stop") - data[finest]["w_N"]
    svg_heatmap(out / "difference_field_stop.svg", diff, stride=max(1, finest // 40),
                title=f"w_h - w_N after stopping, n = {finest} (every {max(1, finest // 40)}th node)")

    # --- metadata and Markdown tables ------------------------------------------------------------------
    meta = {
        "case": {"a": A, "D": D, "q": Q, "nu_navier": NU, "M_navier": M_R2},
        "grids": list(grids), "diagnostic_grid": diagnostic_grid,
        "python": platform.python_version(), "numpy": np.__version__,
        "source_sha256": source_digest(root),
        "stop_status": {str(n): [data[n]["run"].status, data[n]["run"].s] for n in all_grids},
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "tables.md").write_text("# R2, группа B — сгенерированные таблицы\n\n"
                                   "Сформировано `scripts/r2_accuracy_study.py`; не редактировать вручную.\n\n"
                                   + "\n\n".join(md) + "\n", encoding="utf-8")
    return data


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "docs" / "results" / "r2"))
    parser.add_argument("--grids", type=int, nargs="+", default=list(MAIN_GRIDS))
    parser.add_argument("--diagnostic-grid", type=int, default=DIAGNOSTIC_GRID)
    args = parser.parse_args(argv)
    run_study(args.out, tuple(args.grids), args.diagnostic_grid or None)
    print(f"written to {args.out}")


if __name__ == "__main__":
    sys.exit(main())
