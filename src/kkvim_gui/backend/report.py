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

Every number of the report is taken from the outcome; the report shows it in the interface form and, next to it,
in Python's round-trip form (``repr``), so the report can be checked against ``manifest.json`` and ``field.csv``
exactly. Nothing is computed here except unit scaling for display (m -> mm) and lookups of stored values. The
visual exaggeration of the 3D view applies to the figure only and is recorded as such.
"""

import html
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from kkvim_gui import program_version
from kkvim_gui.backend import benchmarks, export
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
COMPARISON_COLUMNS = (  # (name, unit): "1" dimensionless, "" text or count
    ("reference_kind", ""), ("reference_method", ""), ("reference_source_provenance", ""),
    ("beta_reference", "1"), ("w_centre_reference", "m"), ("beta_kkvim", "1"), ("w_centre_kkvim", "m"),
    ("delta_beta", "1"), ("delta_beta_relative", "%"), ("delta_w_centre", "m"),
    ("supports", ""), ("N", ""), ("a", "m"), ("b", "m"), ("h", "m"), ("E", "Pa"), ("nu", "1"), ("q", "Pa"),
    ("n_x", ""), ("n_y", ""), ("centre_i", ""), ("centre_j", ""), ("x_centre", "m"), ("y_centre", "m"),
    ("D", "N·m"), ("w_bar_solver_centre", "1"), ("half_steps_completed", ""), ("half_steps_requested", ""),
    ("solver_status", ""), ("requested_run_completed", ""), ("field_role", ""), ("scientific_status", ""),
    ("mark", ""), ("reference_data", ""), ("reference_data_payload_sha256", ""), ("reference_level_M", ""),
)
COMPARISON_DEFINITIONS = (
    "# comparison with independent references: beta = w(a/2, b/2) * D / (q * a^4); "
    "delta_beta = beta_kkvim - beta_reference; delta_beta_relative = delta_beta / beta_reference * 100 %; "
    "delta_w_centre = w_centre_kkvim - w_centre_reference",
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
           outcome.field_role, export.scientific_status(outcome), outcome.mark or "",
           benchmarks.source_text(c) if c.live else f"{benchmarks.DATA_FILE} ({benchmarks.SCHEMA})",
           "" if c.live else benchmarks.PAYLOAD_SHA256, c.navier["M"] if c.live else "")
    return [(x.reference.kind, x.reference.title, x.reference.detail, x.reference.beta, x.reference.w_centre_m,
             x.beta_kkvim, c.w_centre_m, x.delta, x.rel_pct, x.delta_w_m, *run) for x in c.rows]


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
    if c.live:
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


def report_html(outcome: RunOutcome, files: dict, run_meta=None, view=None, comparison=None) -> str:
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
<h2>7. История полушагов</h2><table class="h"><tr>{head}</tr>
{body}</table>
<h2>8. Ограничения применимости</h2><ul>{"".join(f"<li>{_e(t)}</li>" for t in (*outcome.limitations, *LIMITATIONS))}</ul>
<h2>9. Файлы расчёта</h2><ul>{file_list}</ul>
</body></html>
"""


def write_package(outcome: RunOutcome, folder, run_meta=None, log=(), render=None, view=None, comparison=None) -> dict:
    """Write the calculation package into ``folder`` and return {name: path}.

    ``render(folder) -> {"field_png": path, "surface_png": path}`` draws the figures (Qt, by the caller); it is
    called only for a result with a field. ``view`` records how the figures were drawn (e.g. the exaggeration).
    ``comparison``: the ``benchmarks.Comparison`` shown in the "Сравнение" tab of this outcome (computed here if None).
    """
    folder = Path(folder)
    paths = export.write_result(outcome, folder, run_meta, log)  # refuses an existing non-empty folder
    if render is not None and outcome.has_field:
        paths.update({k: Path(v) for k, v in render(folder).items()})
    comparison = comparison if comparison is not None else comparison_of(outcome)
    if comparison.has_numbers:
        paths["comparison"] = write_comparison_csv(outcome, comparison, folder / COMPARISON_FILE)
    suffix = "_TEST-DATA" if outcome.is_demo else ""
    paths["report"] = folder / f"report{suffix}.html"
    files = {name: path.name for name, path in paths.items() if name != "manifest"}
    paths["report"].write_text(report_html(outcome, files, run_meta, view, comparison), encoding="utf-8")
    data = export.manifest(outcome, run_meta, log, files)
    data["report_summary"] = export._plain(summary(outcome))
    data["visualization"] = export._plain(view or {})
    paths["manifest"].write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return paths
