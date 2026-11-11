"""Saved calculation package: export files, figures and a readable report (functional preview). Qt-free.

``write_package`` writes into one folder:

- ``manifest.json``, ``history.csv`` and the field file of its role (``field.csv``, ``field_research.csv`` or
  ``field_diagnostic.csv``) — the export of ``backend.export``;
- the figures rendered by the interface (PNG; the caller passes a function that writes them);
- ``report.html`` — a readable report built from the same ``RunOutcome`` as the export files.

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
from kkvim_gui.backend import export
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
    "Моменты, напряжения и сравнение с эталонами в этой версии не выводятся.",
    "Масштаб прогиба на 3D-виде увеличен только для изображения; все числа отчёта и файлов — без увеличения.",
)
IMAGE_TITLES = {"field_png": "Поле прогиба w(x, y) с изолиниями", "surface_png": "Деформированная форма (3D-вид)"}


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


def report_html(outcome: RunOutcome, files: dict, run_meta=None, view=None) -> str:
    r = outcome.request
    view = view or {}
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
table.h th{{width:auto}} .banner{{background:#FFF4E0;border:1px solid #E3AE52;padding:8px 12px}}
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
<h2>6. История полушагов</h2><table class="h"><tr>{head}</tr>
{body}</table>
<h2>7. Ограничения применимости</h2><ul>{"".join(f"<li>{_e(t)}</li>" for t in (*outcome.limitations, *LIMITATIONS))}</ul>
<h2>8. Файлы расчёта</h2><ul>{file_list}</ul>
</body></html>
"""


def write_package(outcome: RunOutcome, folder, run_meta=None, log=(), render=None, view=None) -> dict:
    """Write the calculation package into ``folder`` and return {name: path}.

    ``render(folder) -> {"field_png": path, "surface_png": path}`` draws the figures (Qt, by the caller); it is
    called only for a result with a field. ``view`` records how the figures were drawn (e.g. the exaggeration).
    """
    folder = Path(folder)
    paths = export.write_result(outcome, folder, run_meta, log)
    if render is not None and outcome.has_field:
        paths.update({k: Path(v) for k, v in render(folder).items()})
    suffix = "_TEST-DATA" if outcome.is_demo else ""
    paths["report"] = folder / f"report{suffix}.html"
    files = {name: path.name for name, path in paths.items() if name != "manifest"}
    paths["report"].write_text(report_html(outcome, files, run_meta, view), encoding="utf-8")
    data = export.manifest(outcome, run_meta, log, files)
    data["report_summary"] = export._plain(summary(outcome))
    data["visualization"] = export._plain(view or {})
    paths["manifest"].write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return paths
