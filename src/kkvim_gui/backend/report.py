"""Saved calculation package: export files, figures and a readable report (functional preview). Qt-free.

``write_package`` writes into one folder:

- ``manifest.json``, ``history.csv`` and the field file of its role (``field.csv``, ``field_research.csv`` or
  ``field_diagnostic.csv``) — the export of ``backend.export``;
- the figures rendered by the interface (PNG; the caller passes a function that writes them);
- ``report.html`` — a readable report built from the same ``RunOutcome`` as the export files;
- ``comparison.csv`` — only when the run has a comparison with independent references (``backend.benchmarks``: the
  saved references or, for SSSS, the Navier reference computed for the run's own inputs): the same ``Comparison`` object
  as the "Сравнение с эталоном" tab, the same table texts in the report section "Сравнение с независимыми эталонами",
  the same binary64 values in the CSV. Without a comparison (other inputs,
  a diagnostic field, demonstration data, damaged reference data) the report states the reason and no CSV is written;
  the reference data never prevent saving the calculation.

D2 of a CCCC run (``d2``: the record of ``backend.d2_job``, computed for the run's own inputs in a separate process):
``d2_field.csv`` (the D2 field on all nodes, metres, full precision) and ``d2_result.json`` (the record without the
field: method, input echo, status, dimensional and dimensionless values, refinement, timings with their definitions,
runtime) are written next to the KK-VIM files; the manifest gets ``independent_d2`` (the record and the SHA256 of both
files) and ``timing`` (every timer of both methods with its definition); report.html gets the section «МКР — двумерный
(D2)» and «Время вычислений». A D2 that did not run is stated with its reason and writes no file. D2 is a calculation
executed for this run, never stored benchmark data; comparison.csv names the source of every row.

VIM-1, VIM-2, VIM-3 (GUI-VIM-CYCLES-123-01; ``cycles``: ``backend.vim_cycles.CycleTable``): ``vim_cycles.csv`` and, when
drawn, ``vim_cycles.png`` (centre deflection after the completed full cycles k = 1, 2, 3 of this run, from its stored
half-step history, next to its reference); for a run of a finished load series (``series``) also
``load_series_vim_cycles.csv`` and ``load_series.png``; report.html sections 6в and 6г; the manifest gets ``vim_cycles``
(the records and the SHA256 of these files).

Every number of the report is taken from the outcome; the report shows it in the interface form and, next to it,
in Python's round-trip form (``repr``), so the report can be checked against ``manifest.json`` and ``field.csv``
exactly. Nothing is computed here except unit scaling for display (m -> mm) and lookups of stored values. The
visual exaggeration of the 3D view applies to the figure only and is recorded as such.
"""

import hashlib
import html
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from kkvim_gui import program_version
from kkvim_gui.backend import benchmarks, d2_job, export, vim_cycles
from kkvim_gui.backend.outcome import RunOutcome

PREVIEW_TITLE = "KK-VIM — отчёт расчёта"
PREVIEW_NOTICE = (
    "Функциональная предварительная версия интерфейса. Не официальный выпуск R7 и не научная приёмка "
    "отдельного расчёта.")
ROLE_TITLE = {
    "accepted": "результат принятого решателя R3 (CONVERGED)",
    "research": "исследовательское поле (запрошенный расчёт выполнен); не квалифицированный результат",
    "diagnostic": "только диагностика: поле последнего выполненного полушага; не результат",
}
LIMITATIONS = (  # common to every route; the limits of the route itself come from the outcome (``RunOutcome.limitations``)
    "Теории Тимошенко и Редди в этой версии недоступны.",
    "Результат — значения в узлах сетки n_x × n_y; максимальный |w| найден по узлам, а не между ними.",
    "Моменты и напряжения в этой версии не выводятся.",
    "Сравнение с эталонами (раздел 6): SSSS — эталон Навье, рассчитанный для тех же входных данных; CCCC — сохранённые "
    "эталоны только при точном совпадении входных данных. Порога согласия нет, научный статус расчёта оно не меняет.",
    "Масштаб прогиба на 3D-виде увеличен только для изображения; все числа отчёта и файлов — без увеличения.",
)
IMAGE_TITLES = {"field_png": "Поле прогиба w(x, y) с изолиниями", "surface_png": "Деформированная форма (3D-вид)"}
COMPARISON_TITLE = "Сравнение с независимыми эталонами"
COMPARISON_FILE = "comparison.csv"
D2_FIELD_FILE = "d2_field.csv"
D2_RESULT_FILE = "d2_result.json"
D2_FIELD_COLUMNS = (("i", ""), ("j", ""), ("x", "m"), ("y", "m"), ("w_d2", "m"))
COMPARISON_COLUMNS = (  # (name, unit): "1" dimensionless, "" text or count
    ("reference_kind", ""), ("reference_method", ""), ("reference_source_provenance", ""),
    ("beta_reference", "1"), ("w_centre_reference", "m"), ("beta_kkvim", "1"), ("w_centre_kkvim", "m"),
    ("delta_beta", "1"), ("delta_beta_relative", "%"), ("delta_w_centre", "m"),
    ("supports", ""), ("N", ""), ("a", "m"), ("b", "m"), ("h", "m"), ("E", "Pa"), ("nu", "1"), ("q", "Pa"),
    ("n_x", ""), ("n_y", ""), ("centre_i", ""), ("centre_j", ""), ("x_centre", "m"), ("y_centre", "m"),
    ("D", "N·m"), ("w_bar_solver_centre", "1"), ("half_steps_completed", ""), ("half_steps_requested", ""),
    ("solver_status", ""), ("requested_run_completed", ""), ("field_role", ""), ("scientific_status", ""),
    ("mark", ""), ("reference_data", ""), ("reference_data_payload_sha256", ""), ("reference_level_M", ""),
    ("reference_origin", ""),
)
COMPARISON_DEFINITIONS = (
    "# comparison with independent references: beta = w(a/2, b/2) * D / (q * a^4); "
    "delta_beta = beta_kkvim - beta_reference; delta_beta_relative = delta_beta / beta_reference * 100 %; "
    "delta_w_centre = w_centre_kkvim - w_centre_reference",
    "# reference_origin: executed = computed for this run in a separate process (navier_live, D2_live); stored = saved "
    "benchmark data (not recomputed); D2_live: beta_reference = w_centre_d2 * D_reference / (q * a^4), D2 is the same "
    "discrete problem as KK-VIM CCCC (D-049.1): the difference is not an error of the continuous plate model",
    "# signed differences only: no agreement threshold is approved; the comparison is not a scientific acceptance "
    "and does not change the status of the run",
)


def _e(text) -> str:
    return html.escape(str(text))


def _num(value, digits=6) -> str:
    """Interface form: decimal comma, ·10ⁿ."""
    if value is None:
        return "—"
    v = float(value)
    text = f"{v:.{digits}g}"
    if "e" in text:
        mantissa, exponent = text.split("e")
        sup = str.maketrans("0123456789-+", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺")
        text = f"{mantissa}·10{str(int(exponent)).translate(sup)}"
    return text.replace(".", ",")


def _exact(value) -> str:
    return "—" if value is None else repr(float(value))


def centre_node(outcome: RunOutcome) -> dict | None:
    """The node at (a/2, b/2) if the grid has one (even n_x and n_y), else None. A lookup, not an interpolation."""
    r = outcome.request
    if outcome.w is None or r.n_x % 2 or r.n_y % 2:
        return None
    i, j = r.n_x // 2, r.n_y // 2
    return {"i": i, "j": j, "x": float(outcome.x[i]), "y": float(outcome.y[j]), "w": float(outcome.w[i, j]),
            "w_bar": float(outcome.w_bar[i, j])}


def summary(outcome: RunOutcome) -> dict:
    """Key values of the report (plain floats of the outcome), also written to the manifest."""
    e = outcome.extremum()
    c = centre_node(outcome)
    return {
        "w_max_abs_node": None if e is None else e,
        "w_centre_node": c,
        "w_bar_max_abs": None if outcome.w_bar is None else float(np.max(np.abs(outcome.w_bar))),
    }


def _rows(rows) -> str:
    return "\n".join(f"<tr><th>{_e(k)}</th><td>{v}</td></tr>" for k, v in rows)


def comparison_of(outcome: RunOutcome, navier=None) -> benchmarks.Comparison:
    """``benchmarks.compare`` of the outcome; any failure of the reference data becomes DATA_ERROR, never a lost package."""
    try:
        return benchmarks.compare(outcome, navier=navier)
    except Exception as exc:  # noqa: BLE001 - the calculation package must stay savable without the reference data
        supports = getattr(getattr(outcome, "request", None), "supports", "")
        return benchmarks.Comparison(benchmarks.DATA_ERROR, f"Сравнение недоступно: данные эталонов не прочитаны ({exc}).",
                                     supports)


def comparison_rows(outcome: RunOutcome, c: benchmarks.Comparison) -> list[tuple]:
    """Rows of comparison.csv in the order of COMPARISON_COLUMNS: the values of ``c`` and of the outcome, not recomputed."""
    r = outcome.request
    run = (r.supports, r.N, r.a, r.b, r.t, r.E, r.nu, r.q, r.n_x, r.n_y, c.centre["i"], c.centre["j"], c.centre["x"],
           c.centre["y"], c.D, c.w_bar_solver, outcome.s, outcome.s_requested, outcome.status, outcome.requested_completed,
           outcome.field_role, export.scientific_status(outcome), outcome.mark or "")

    def source(x):
        executed = x.reference.kind in benchmarks.EXECUTED_KINDS
        return (benchmarks.row_source(c, x) if executed else f"{benchmarks.DATA_FILE} ({benchmarks.SCHEMA})",
                "" if executed else benchmarks.PAYLOAD_SHA256,
                c.navier["M"] if x.reference.kind == "navier_live" else "", "executed" if executed else "stored")

    return [(x.reference.kind, x.reference.title, x.reference.detail, x.reference.beta, x.reference.w_centre_m,
             x.beta_kkvim, c.w_centre_m, x.delta, x.rel_pct, x.delta_w_m, *run, *source(x)) for x in c.rows]


def write_comparison_csv(outcome: RunOutcome, c: benchmarks.Comparison, path) -> Path:
    if not c.has_numbers:
        raise ValueError("no comparison with numbers: comparison.csv is not written")
    path = Path(path)
    header = [f"{name} [{unit}]" if unit else name for name, unit in COMPARISON_COLUMNS]
    export._write_csv(path, [*export._header_lines(outcome), *COMPARISON_DEFINITIONS], header, comparison_rows(outcome, c))
    return path


def _comparison_section(outcome: RunOutcome, c: benchmarks.Comparison, csv_name) -> str:
    """Section 6: the same Comparison as the "Сравнение с эталоном" tab, its table texts (benchmarks.cells) and statuses."""
    reason = c.reason
    if outcome.is_demo:
        reason = "Демонстрационные данные (TEST DATA): решатель не вызывался, сравнение не выполняется."
    if c.executed:
        data = benchmarks.source_text(c)
    elif c.state == benchmarks.DATA_ERROR:
        data = f"не прошли проверку; ожидались {benchmarks.dataset_text()}"
    elif c.data is not None:
        data = f"{benchmarks.dataset_text()} — проверено при чтении"
    else:
        data = f"не читались (сравнение не выполняется); версия программы: {benchmarks.dataset_text()}"
    rows = [("Состояние сравнения", f"<b>{_e(benchmarks.state_title(c))}</b>")]
    if reason:
        rows.append(("Причина", _e(reason)))
    rows += [("Источник эталона", _e(data)),
             ("Статус решателя", _e(outcome.status)),
             ("Научный статус", _e(outcome.science_text)),  # the text starts with the mark of a non-accepted route
             ("Роль поля", _e(ROLE_TITLE.get(outcome.field_role, "поле не получено")))]
    if not c.has_numbers:
        rows.append(("Файл comparison.csv", "не создан: сравнения с числами для этого расчёта нет"))
        return f"<h2>6. {_e(COMPARISON_TITLE)}</h2><table>{_rows(rows)}</table>"
    ce = c.centre
    rows += [
        ("Узел центра", _e(f"i = {ce['i']}, j = {ce['j']}: x = {ce['x']!r} m, y = {ce['y']!r} m (точно a/2, b/2)")),
        ("Прогиб в центре w_c (КК-ВИМ)", f"{_e(benchmarks.full(c.w_centre_m))} m"),
        ("Жёсткость D (kkvim)", f"{_e(benchmarks.full(c.D))} N·m"),
        ("β = w_c·D/(q·a⁴) (КК-ВИМ)", f"<b>{_e(benchmarks.full(c.beta))}</b>"),
        ("w̄ решателя в этом узле", f"{_e(benchmarks.full(c.w_bar_solver))} (MATH_SPEC 2.5, контроль)"),
        ("Файл comparison.csv", f"<code>{_e(csv_name)}</code> — те же значения"),
    ]
    head = "".join(f"<th>{_e(h)}</th>" for h in benchmarks.HEADERS)
    body = "\n".join("<tr>" + "".join(f"<td>{_e(t)}</td>" for t in benchmarks.cells(x)) + "</tr>" for x in c.rows)
    missing = "".join(f"<li>{_e(benchmarks.KIND_TITLE[k])}: {_e(why)}.</li>" for k, why in c.missing)
    notes = "".join(f"<li>{_e(n)}</li>" for n in c.notes)
    return (f"<h2>6. {_e(COMPARISON_TITLE)}</h2><table>{_rows(rows)}</table>\n"
            f"<table class=\"h c\"><tr>{head}</tr>\n{body}</table>\n"
            + (f"<p>Нет данных:</p><ul>{missing}</ul>" if missing else "")
            + f"<p>Как читать сравнение:</p><ul>{notes}</ul>")


def _sec(value) -> str:
    return "—" if value is None else f"{float(value):.4f} с ({float(value)!r})"


def timing_table(timing=None, d2=None) -> list[dict]:
    """Every timer of the run: KK-VIM (``RunRecord.timing``) and D2 (``d2["timing"]``), with its definition."""
    timing = timing or {}
    t = (d2 or {}).get("timing") or {}
    rows = [{"timer": key, "seconds": timing.get(key), "definition": d2_job.TIMER_TEXT[key]}
            for key in ("kkvim_solver_s", "kkvim_import_s", "kkvim_wall_s")]
    if d2 is not None:
        rows += [{"timer": f"d2_{key}", "seconds": t.get(key), "definition": d2_job.TIMER_TEXT[f"d2_{key}"]}
                 for key in ("solve_s", "import_s", "post_s", "worker_s", "wall_s", "parse_s")]
    return rows


def _d2_section(outcome: RunOutcome, d2, files) -> str:
    """Section 6a: D2 computed for this run (CCCC), or the reason it is absent."""
    title = f"<h2>6а. {_e(d2_job.TITLE)} — рассчитан для этих входных данных</h2>"
    if d2 is None:
        return ""
    if d2.get("state") != d2_job.OK:
        rows = [("Состояние", f"<b>{_e(d2_job.STATE_TEXT.get(d2.get('state'), d2.get('state')))}</b>"),
                ("Причина", _e(d2.get("reason", ""))), ("Файлы D2", "не созданы: значения D2 нет и не подставляются")]
        return f"{title}<table>{_rows(rows)}</table>"
    inp, p = d2["inputs"], d2.get("problem") or {}
    rows = [("Состояние", f"<b>{_e(d2_job.STATE_TEXT[d2_job.OK])}</b> "
                          "(расчёт выполнен для этого запуска, не сохранённые данные)"),
            ("Метод", f"<code>{_e(d2['method'])}</code>"), ("Источник", _e(d2["source"])),
            ("Вход (СИ)", _e(f"a = {inp['a']!r} m, b = {inp['b']!r} m, h = {inp['t']!r} m, E = {inp['E']!r} Pa, "
                             f"ν = {inp['nu']!r}, q = {inp['q']!r} Pa, {inp['supports']}")),
            ("Сетка", _e(f"n_x = {inp['n_x']}, n_y = {inp['n_y']}; {p.get('unknowns')} неизвестных; упорядочение "
                         f"{p.get('ordering')}; ненулевых в LU {d2['factor_nnz']}")),
            ("Жёсткость D (эталон)", f"{_e(benchmarks.full(d2['D']))} N·m")]
    if d2["w_centre_m"] is None:
        rows.append(("w в центре", _e(d2.get("centre_reason", ""))))
    else:
        rows += [("w в центре (узел a/2, b/2)", f"<b>{_num(d2['w_centre_m'] * 1000.0, 10)} mm</b> "
                                                 f"<span class=x>({_exact(d2['w_centre_m'])} m)</span>"),
                 ("β_a = w_c·D/(q·a⁴) (длина a)", f"{_e(benchmarks.full(d2['beta_a']))}"),
                 (f"α = w_c·D/(q·l⁴), l = min(a, b) = {d2['short_side_m']!r} m", _e(benchmarks.full(d2["alpha_short_side"])))]
    diff = d2_job.field_difference(d2, outcome)
    rows += [("max |w_КК-ВИМ − w_D2| по узлам", "не вычисляется: поле КК-ВИМ не является результатом" if diff is None else
              _e(f"{diff['delta_w_m']!r} m (знаковое) в узле i = {diff['i']}, j = {diff['j']}")),
             ("Собственная оценка ошибки e_D2", f"{_e(benchmarks.full(d2['e_D2_m']))} m (последняя поправка уточнения)"),
             ("Уточнение", _e(f"{d2['refinement_steps']} шагов; поправки {d2['corrections_m']!r} m")),
             ("Файлы D2", f"<code>{_e(files.get('d2_field', '—'))}</code>, <code>{_e(files.get('d2_result', '—'))}</code>"),
             ("Ограничения", _e("D2 — независимая реализация той же дискретной задачи (D-049.1, D-049.8): разность КК-ВИМ − D2 "
                                "не является ошибкой континуальной модели пластины и не проверяет сеточную погрешность; "
                                "наблюдаемый второй порядок по сетке не доказан как глобальная оценка; порога согласия нет; "
                                "научный статус КК-ВИМ не меняется."))]
    return f"{title}<table>{_rows(rows)}</table>"


def _timing_section(timing, d2) -> str:
    if not timing and d2 is None:
        return ""
    rows = [(r["timer"], f"{_e(_sec(r['seconds']))} — {_e(r['definition'])}") for r in timing_table(timing, d2)]
    rt = (d2 or {}).get("runtime") or {}
    if rt:
        rows.append(("Компьютер и среда", _e(f"{rt.get('platform')}; {rt.get('processor') or rt.get('machine')}; "
                                             f"логических процессоров {rt.get('cpu_count')}; Python {rt.get('python')}")))
    rows.append(("Роль измерений", _e(d2_job.TIMING_ROLE)))
    return f"<h2>6б. Время вычислений</h2><table>{_rows(rows)}</table>"


def _cycle_cells(c) -> list:
    def mm(value):
        return "—" if value is None else f"{_num(value * 1000.0)} mm"

    return [_e(c.label), str(c.k), str(c.s),
            "—" if c.w_m is None else f"{mm(c.w_m)} <span class=x>({_exact(c.w_m)} m)</span>",
            mm(c.ref_w_m), mm(c.delta_w_m), "—" if c.delta_pct is None else f"{_num(c.delta_pct)} %",
            "да" if c.on_graph else "нет", _e(c.reason or ("неполный цикл: только Y¹; не VIM-1" if c.k == 0 else ""))]


def _cycles_section(cycles, files) -> str:
    """6в: VIM-1 … VIM-3 of this run (GUI-VIM-CYCLES-123-01): the table and the graph of the same records."""
    if cycles is None:
        return ""
    title = "<h2>6в. Полные циклы одного расчёта КК-ВИМ: VIM-1, VIM-2, VIM-3</h2>"
    ref = vim_cycles.REFERENCE_TITLE.get(cycles.reference, "—")
    info = [("Обозначения", _e(vim_cycles.OWNER_NOTE)), ("Статус решателя", f"<b>{_e(cycles.solver_status)}</b>"),
            ("Выполнено", _e(cycles.termination)), ("Остановка", _e(cycles.stop_text)),
            ("Утверждённый критерий остановки выполнен", "да" if cycles.criterion_met else "нет"),
            ("Научный статус", _e(cycles.scientific_status)), ("Эталон", _e(ref) + (
                "" if cycles.ref_w_m is not None else f" — недоступен: {_e(cycles.reference_reason)}")),
            ("Происхождение значений", _e(cycles.provenance or cycles.route_reason)),
            ("Файл", f"<code>{_e(files.get('vim_cycles', '—'))}</code>")]
    head = "".join(f"<th>{h}</th>" for h in ("Состояние", "k", "s", "w в центре КК-ВИМ", "w эталона", "Δw = w_КК-ВИМ − w_эт.",
                                             "Δw / w_эт.", "На графике", "Примечание"))
    body = "\n".join("<tr>" + "".join(f"<td>{cell}</td>" for cell in _cycle_cells(c)) + "</tr>"
                     for c in (cycles.partial, *cycles.cycles))
    image = (f"<figure><img src=\"{_e(files['vim_cycles_png'])}\" alt=\"VIM-1–3\"><figcaption>Прогиб в центре после "
             "завершённых полных циклов k (s = 2k) и эталон; те же записи, что в таблице</figcaption></figure>"
             if "vim_cycles_png" in files else "<p>График не построен: нет ни одного состояния VIM-k, показываемого как "
                                               "результат (причины — в таблице).</p>")
    return (f"{title}<table>{_rows(info)}</table>\n<table class=\"h\"><tr>{head}</tr>\n{body}</table>\n{image}\n"
            "<p class=x>Наличие VIM-3 — не сходимость; разности знаковые, порога согласия нет; более поздний цикл не "
            "объявляется более точным.</p>")


def _series_section(series, files) -> str:
    """6г: the finished load series of this run, VIM-1 … VIM-3 at every q (each from its own run) and the reference."""
    if series is None:
        return ""
    head = "".join(f"<th>{h}</th>" for h in ("q", "Расчёт", "Статус решателя", "w итог КК-ВИМ", *(
        f"w {vim_cycles.label(k)} (s = {2 * k})" for k in range(1, vim_cycles.K_MAX + 1)), "w эталона", "Примечание"))
    lines = []
    for p in series.points:
        cells = [f"{_num(p.q)} Pa", f"№ {p.run_id}" if p.run_id else "—", _e(p.solver_status or p.run_status),
                 "—" if p.w_centre_m is None else f"{_num(p.w_centre_m * 1000.0)} mm"]
        notes = []
        for k in range(1, vim_cycles.K_MAX + 1):
            c = p.cycle(k)
            cells.append(f"{_num(c.w_m * 1000.0)} mm" if c is not None and c.on_graph else "—")
            if c is not None and not c.on_graph and c.reason:
                notes.append(f"{c.label}: {c.reason}")
        ref = p.navier_w_m if series.navier is not None else p.d2_w_m
        cells += ["—" if ref is None else f"{_num(ref * 1000.0)} mm", _e("; ".join(notes) or p.reason or "")]
        lines.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    image = (f"<figure><img src=\"{_e(files['load_series_png'])}\" alt=\"load-deflection\"><figcaption>Нагрузка — прогиб: "
             "КК-ВИМ (итог), VIM-1, VIM-2, VIM-3 и эталон при каждом q</figcaption></figure>"
             if "load_series_png" in files else "")
    return (f"<h2>6г. Нагрузка — прогиб (серия {series.series_id}): VIM-1–3 и эталон</h2>"
            f"<p>Каждое q — отдельный расчёт; VIM-k при каждом q — из истории полушагов расчёта этой точки, не пересчёт "
            f"по нагрузке. Файл: <code>{_e(files.get('load_series_vim_cycles', '—'))}</code></p>"
            f"<table class=\"h\"><tr>{head}</tr>\n{''.join(lines)}</table>\n{image}")


def report_html(outcome: RunOutcome, files: dict, run_meta=None, view=None, comparison=None, d2=None, timing=None,
                cycles=None, series=None) -> str:
    r = outcome.request
    view = view or {}
    comparison = comparison if comparison is not None else comparison_of(outcome)
    s = summary(outcome)
    e, c = s["w_max_abs_node"], s["w_centre_node"]
    created = datetime.now().isoformat(timespec="seconds")
    banner = [PREVIEW_NOTICE]
    if outcome.diagnostic_only:
        banner.insert(0, f"ДИАГНОСТИКА: расчёт не выполнен как запрошено ({outcome.status}); поле последнего выполненного "
                         "полушага — только диагностика, не результат.")
    elif outcome.is_error:
        banner.insert(0, f"ДИАГНОСТИКА: решатель завершился ошибкой {outcome.status}; поле не получено.")
    if outcome.mark:
        banner.insert(0, outcome.mark)
    if outcome.is_demo:
        banner.insert(0, "TEST DATA: демонстрационные данные, решатель не вызывался.")
    entered = [(k, f"{_e(text)} {_e(unit)}".strip()) for k, text, unit in r.entered]
    si = [("a", f"{_exact(r.a)} m"), ("b", f"{_exact(r.b)} m"), ("h (t)", f"{_exact(r.t)} m"),
          ("E", f"{_exact(r.E)} Pa"), ("ν", _exact(r.nu)), ("q", f"{_exact(r.q)} Pa"),
          ("b/a", _num(r.b / r.a, 6)), ("D = Et³/(12(1−ν²)) (kkvim)", f"{_num(outcome.D)} N·m ({_exact(outcome.D)})")]
    model = [("Теория", "Кирхгоф–Ляв (линейная)"), ("Закрепление", _e(r.supports)), ("Нагрузка", "равномерная q"),
             ("Маршрут", _e(outcome.route)),
             ("Метод", f"КК-ВИМ, N = {r.N}"), ("Решатель", f"{_e(outcome.solver)}: <code>{_e(outcome.call)}</code>"),
             ("Сетка", f"n_x = {r.n_x}, n_y = {r.n_y} ({len(outcome.x) if outcome.x is not None else '—'} × "
                       f"{len(outcome.y) if outcome.y is not None else '—'} узлов)")]
    model += [(f"Параметр {k}", _e(v)) for k, v in outcome.parameters]
    results = []
    if e is not None:
        results += [
            ("Максимальный |w| (по узлам)", f"<b>{_num(e['w'] * 1000.0)} mm</b> = {_num(e['w'])} m "
                                            f"<span class=x>({_exact(e['w'])} m)</span>"),
            ("Координаты узла", f"x = {_num(e['x'])} m, y = {_num(e['y'])} m (i = {e['i']}, j = {e['j']})"),
            ("w̄ = wD/(qa⁴) в этом узле", f"<b>{_num(e['w_bar'])}</b> <span class=x>({_exact(e['w_bar'])})</span>"),
        ]
    if c is not None:
        results += [("w в центре (узел a/2, b/2)", f"{_num(c['w'] * 1000.0)} mm <span class=x>({_exact(c['w'])} m)"
                                                    "</span>"),
                    ("w̄ в центре", f"{_num(c['w_bar'])} <span class=x>({_exact(c['w_bar'])})</span>")]
    elif outcome.w is not None:
        results.append(("w в центре", "в центре нет узла сетки (n_x или n_y нечётно); значение не интерполируется"))
    if outcome.w is None:
        results.append(("Поле прогиба", "не получено"))
    statuses = [
        ("Статус решателя", f"<b>{_e(outcome.status)}</b>"),
        ("Запрошенный расчёт выполнен", "да" if outcome.requested_completed else "нет"),
        ("Остановка", _e(outcome.stop_text)),
        ("Научный статус", _e(outcome.science_text)),
        ("Полушаги, циклы", f"s = {outcome.s}" + (f" из запрошенных {outcome.s_requested}" if outcome.s_requested else "")
                            + f", k = {outcome.k} (D-030)"),
        ("Роль поля", _e(ROLE_TITLE.get(outcome.field_role, "поле не получено"))),
        ("Файл поля", f"<code>{_e(files.get('field', '—'))}</code>"),
        ("Решения владельца", _e(", ".join(outcome.decisions) or "—")),
    ]
    if outcome.reason:
        statuses.append(("Причина (решатель)", _e(outcome.reason)))
    if outcome.error:
        statuses.append(("Ошибка", _e(outcome.error)))
    statuses += [("Примечание", _e(n)) for n in outcome.notes]
    images = "".join(
        f"<figure><img src=\"{_e(files[key])}\" alt=\"{_e(title)}\"><figcaption>{_e(title)}"
        + (f". {_e(view['exaggeration_text'])}" if key == "surface_png" and view.get("exaggeration_text") else "")
        + "</figcaption></figure>"
        for key, title in IMAGE_TITLES.items() if key in files)
    history_cols = [c for c in outcome.columns if c.key in ("s", "k", "direction", "status", "eps_inf", "eps_E", "P", "P_hat",
                                                            "energy_status", "w_bar_centre", "rn5_verdict", "rt7_passed")]
    head = "".join(f"<th>{_e(col.title)}{', ' + _e(col.unit) if col.unit else ''}</th>" for col in history_cols)
    body = "\n".join("<tr>" + "".join(
        f"<td>{_num(row.get(col.key)) if col.kind == 'number' else _e(row.get(col.key) or '—')}</td>"
        for col in history_cols) + "</tr>" for row in outcome.history)
    file_list = "".join(f"<li><code>{_e(name)}</code></li>" for name in sorted(set(files.values())))
    versions = ", ".join(f"{k} {v}" for k, v in outcome.versions.items())
    meta = run_meta or {}
    return f"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><title>Отчёт KK-VIM</title>
<style>
body{{font-family:"Segoe UI",Arial,sans-serif;max-width:980px;margin:24px auto;padding:0 16px;color:#1B2B3A;
background:#fff;line-height:1.45}}
h1{{font-size:22px}} h2{{font-size:17px;margin-top:28px;border-bottom:1px solid #DCE3EA;padding-bottom:4px}}
table{{border-collapse:collapse;width:100%;font-size:14px}} th,td{{border:1px solid #DCE3EA;padding:4px 8px;
text-align:left;vertical-align:top}} th{{background:#F3F6F9;font-weight:600;width:32%}}
table.h th{{width:auto}} table.c td{{font-family:Consolas,monospace;font-size:12px;overflow-wrap:anywhere}}
table.c td:first-child,table.c td:last-child{{font-family:inherit;font-size:13px}}
.banner{{background:#FFF4E0;border:1px solid #E3AE52;padding:8px 12px}}
.x{{color:#5F6E7C;font-family:Consolas,monospace;font-size:12px}} figure{{margin:12px 0}}
img{{max-width:100%;border:1px solid #DCE3EA}} figcaption{{font-size:13px;color:#5F6E7C}}
</style></head><body>
<h1>{_e(PREVIEW_TITLE)} ({_e(r.supports)}, N = {r.N})</h1>
<div class="banner">{"<br>".join(_e(b) for b in banner)}</div>
<p>Отчёт создан {_e(created)} · запуск {_e(meta.get('run_id', '—'))} · kkvim-gui {_e(program_version())} ·
{_e(versions)}</p>
<h2>1. Исходные данные (как введены)</h2><table>{_rows(entered)}</table>
<h2>2. Значения для решателя (СИ)</h2><table>{_rows(si)}</table>
<h2>3. Расчётная модель и метод</h2><table>{_rows(model)}</table>
<h2>4. Результаты</h2><table>{_rows(results)}</table>
<p class="x">В скобках — точные значения массивов решателя (repr), те же, что в manifest.json и field.csv.</p>
{images}
<h2>5. Статус расчёта</h2><table>{_rows(statuses)}</table>
{_comparison_section(outcome, comparison, files.get("comparison"))}
{_d2_section(outcome, d2, files)}
{_timing_section(timing, d2)}
{_cycles_section(cycles, files)}
{_series_section(series, files)}
<h2>7. История полушагов</h2><table class="h"><tr>{head}</tr>
{body}</table>
<h2>8. Ограничения применимости</h2><ul>{"".join(f"<li>{_e(t)}</li>" for t in (*outcome.limitations, *LIMITATIONS))}</ul>
<h2>9. Файлы расчёта</h2><ul>{file_list}</ul>
</body></html>
"""


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_d2_field_csv(outcome: RunOutcome, d2, path) -> Path:
    """The D2 field on all nodes (i h_x, j h_y), metres, full precision."""
    r = outcome.request
    field = d2["field"]
    h_x, h_y = r.a / r.n_x, r.b / r.n_y
    header = [f"{name} [{unit}]" if unit else name for name, unit in D2_FIELD_COLUMNS]
    comments = [f"# {d2_job.TITLE}: computed for this run in a separate process ({d2['method']})",
                "# QA reference D2: the same discrete problem as KK-VIM CCCC (D-049.1, D-049.8); not a qualified result "
                "of the plate model; nodes x = i h_x, y = j h_y; w = 0 on the boundary"]
    rows = ((i, j, i * h_x, j * h_y, float(field[i, j])) for i in range(field.shape[0]) for j in range(field.shape[1]))
    export._write_csv(Path(path), comments, header, rows)
    return Path(path)


def write_package(outcome: RunOutcome, folder, run_meta=None, log=(), render=None, view=None, comparison=None,
                  d2=None, timing=None, cycles=None, series=None) -> dict:
    """Write the calculation package into ``folder`` and return {name: path}.

    ``render(folder) -> {"field_png": path, "surface_png": path}`` draws the figures (Qt, by the caller); it is
    called only for a result with a field. ``view`` records how the figures were drawn (e.g. the exaggeration).
    ``comparison``: the ``benchmarks.Comparison`` shown in the "Сравнение" tab of this outcome (computed here if None).
    ``d2``: the D2 record of the run (``backend.d2_job``), None without D2; ``timing``: ``RunRecord.timing`` of KK-VIM.
    ``cycles``: the ``vim_cycles.CycleTable`` of the run (vim_cycles.csv, section 6в); ``series``: the finished load series
    of the run (``model.series.SeriesData``; load_series_vim_cycles.csv, section 6г). The SHA256 of their files are in the
    manifest (``vim_cycles``).
    """
    folder = Path(folder)
    paths = export.write_result(outcome, folder, run_meta, log)  # refuses an existing non-empty folder
    if render is not None and outcome.has_field:
        paths.update({k: Path(v) for k, v in render(folder).items()})
    comparison = comparison if comparison is not None else comparison_of(outcome)
    if comparison.has_numbers:
        paths["comparison"] = write_comparison_csv(outcome, comparison, folder / COMPARISON_FILE)
    d2_ok = d2 is not None and d2.get("state") == d2_job.OK
    if d2_ok:
        paths["d2_field"] = write_d2_field_csv(outcome, d2, folder / D2_FIELD_FILE)
        result = {**export._plain(d2_job.plain(d2)), "field_file": D2_FIELD_FILE,
                  "timer_definitions": {k: v for k, v in d2_job.TIMER_TEXT.items() if k.startswith("d2_")},
                  "role": "executed for this run in a separate process; the same discrete problem as KK-VIM CCCC "
                          "(D-049.1, D-049.8); not stored benchmark data and not a qualified result"}
        paths["d2_result"] = folder / D2_RESULT_FILE
        paths["d2_result"].write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    run_id = (run_meta or {}).get("run_id")
    if cycles is not None:
        paths["vim_cycles"] = folder / vim_cycles.CSV_FILE
        vim_cycles.write_csv(paths["vim_cycles"], [(run_id, cycles)], export._header_lines(outcome))
    if series is not None:
        paths["load_series_vim_cycles"] = folder / vim_cycles.SERIES_CSV_FILE
        vim_cycles.write_csv(paths["load_series_vim_cycles"],
                             [(p.run_id, p.cycles) for p in series.points if p.cycles is not None],
                             [f"# load series {series.series_id}: {series.supports}, N = {series.N}, grid {series.grid[0]} x "
                              f"{series.grid[1]}; every row group is the run of one q"])
    suffix = "_TEST-DATA" if outcome.is_demo else ""
    paths["report"] = folder / f"report{suffix}.html"
    files = {name: path.name for name, path in paths.items() if name != "manifest"}
    paths["report"].write_text(report_html(outcome, files, run_meta, view, comparison, d2, timing, cycles, series),
                               encoding="utf-8")
    data = export.manifest(outcome, run_meta, log, files)
    data["report_summary"] = export._plain(summary(outcome))
    data["visualization"] = export._plain(view or {})
    if d2 is not None:
        data["independent_d2"] = {
            "state": d2.get("state"), "reason": d2.get("reason", ""), "title": d2_job.TITLE,
            "record": export._plain(d2_job.plain(d2)) if d2_ok else None,
            "files_sha256": {paths[k].name: _sha256(paths[k]) for k in ("d2_field", "d2_result") if k in paths},
        }
    data["timing"] = {"timers": export._plain(timing_table(timing, d2)), "role": d2_job.TIMING_ROLE}
    if cycles is not None:
        names = ("vim_cycles", "vim_cycles_png", "load_series_vim_cycles", "load_series_png")
        data["vim_cycles"] = {**vim_cycles.plain(cycles, run_id),
                              "series_id": None if series is None else series.series_id,
                              "files_sha256": {paths[k].name: _sha256(paths[k]) for k in names if k in paths}}
    paths["manifest"].write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return paths
