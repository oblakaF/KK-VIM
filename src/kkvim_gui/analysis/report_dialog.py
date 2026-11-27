"""«Собрать отчёт…» of the analysis window (GUI_DESIGN.md 18, GUI-D18; P7).

``components(window)`` lists what the run of the window can give — the figures of its tabs (as shown: visible curves,
chosen styles, the selected node lines), the numerical tables of the calculation package, the raw JSON + CSV export and
the combined PDF — each available or not with the reason. ``plan(window, …)`` turns a choice into a
``backend.report_builder.ReportPlan`` and ``build_report`` writes the folder. The dialog only collects the choice; every
file is written from the stored run (only read; nothing recomputed, no solver call).
"""

from pathlib import Path

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from kkvim_gui.backend import export, report_builder
from kkvim_gui.figures import publication as pub

FIGURE_TITLES = {
    "field": "Поле прогиба w(x, y) с изолиниями",
    "section_x": "Сечение w(x) при выбранном y_j",
    "section_y": "Сечение w(y) при выбранном x_i",
    "history": "Изменение поля по полушагам (ε^∞, ε^E)",
    "energy": "Энергия Π_h по полушагам",
    "vim_cycles": "Циклы VIM-1–3: прогиб в центре и эталон",
    "d2_section_x": "КК-ВИМ и МКР D2: сечение при y_j",
    "d2_section_y": "КК-ВИМ и МКР D2: сечение при x_i",
    "load_series": "Нагрузка — прогиб",
    "moments_Mx": "R6: изгибающий момент M_x(x, y) (RESEARCH / NOT_QUALIFIED)",
    "moments_My": "R6: изгибающий момент M_y(x, y) (RESEARCH / NOT_QUALIFIED)",
    "moments_Mxy": "R6: крутящий момент M_xy(x, y), соглашение L-107 (RESEARCH / NOT_QUALIFIED)",
    "moments_sigma_x": "R6: напряжение σ_x на нижней поверхности z = +t/2 (RESEARCH / NOT_QUALIFIED)",
    "moments_section_x": "R6: сечение выбранной величины при y_j (вкладка «Моменты и напряжения»)",
    "moments_section_y": "R6: сечение выбранной величины при x_i (вкладка «Моменты и напряжения»)",
}
TABLE_TITLES = {
    "field_table": "Значения во всех узлах (поле решателя)",
    "history_table": "История полушагов",
    "comparison_table": "Сравнение с независимыми эталонами",
    "vim_cycles_table": "Циклы VIM-1–3 (таблица)",
    "d2_field_table": "Поле МКР D2 во всех узлах",
    "chart_data": "Данные графиков (исходные значения видимых кривых)",
    "moments_table": "R6: моменты и напряжения во всех узлах и сводка (moments_*.csv, moments.json)",
    "moments_comparison_table": "R6: сравнение моментов с независимым эталоном (наблюдения, без порогов)",
}


class Component:
    def __init__(self, id, kind, title, source=None, available=True, reason=""):
        self.id, self.kind, self.title, self.source = id, kind, title, source
        self.available, self.reason = available, reason


def _has_points(tools):
    return any(tools.has_points(c) and tools.is_shown(c.key) for c in tools.curves())


def components(window):
    """The components of the report of ``window`` (an AnalysisWindow of a solver run), available or not."""
    from kkvim_gui.analysis.cycles_tab import availability_text

    o = window.outcome
    out = [Component("field", "figure", FIGURE_TITLES["field"], window.fields.publisher, o.has_field,
                     "" if o.has_field else "поле не получено")]
    for key, tools in (("section_x", window.sections.tools_x), ("section_y", window.sections.tools_y),
                       ("history", window.history.tools), ("energy", window.energy.tools)):
        ok = _has_points(tools)
        out.append(Component(key, "figure", FIGURE_TITLES[key], tools, ok,
                             "" if ok else "на графике нет видимых значений (маршрут их не вычисляет или кривые скрыты)"))
    cycles = window.cycles.table
    tools = getattr(window.cycles.figure, "tools", None)
    ok = bool(cycles.drawn) and tools is not None and _has_points(tools)
    out.append(Component("vim_cycles", "figure", FIGURE_TITLES["vim_cycles"], tools, ok,
                         "" if ok else availability_text(cycles)))
    d2 = getattr(window, "_d2", None)
    d2_tab = getattr(window, "d2_tab", None)
    for key, attr in (("d2_section_x", "figure_x"), ("d2_section_y", "figure_y")):
        fig = getattr(d2_tab, attr, None) if d2_tab is not None else None
        ok = fig is not None and getattr(fig, "tools", None) is not None
        reason = ("МКР D2 не выбран для этого расчёта" if d2 is None else
                  f"D2 не рассчитан: {d2.get('reason') or d2.get('state')}")
        out.append(Component(key, "figure", FIGURE_TITLES[key], fig.tools if ok else None, ok, "" if ok else reason))
    series = window.series if window.series is not None and window.series.complete else None
    ok = series is not None and window.series_tab is not None
    out.append(Component("load_series", "figure", FIGURE_TITLES["load_series"],
                         window.series_tab.figure.tools if ok else None, ok,
                         "" if ok else "расчёт не входит в завершённую серию нагрузок"))
    mt = getattr(window, "moments", None)
    m_ok = mt is not None and not mt.unavailable_reason
    m_reason = "" if m_ok else (mt.unavailable_reason if mt is not None else "вкладки моментов нет")
    for key, surface in (("Mx", "bottom"), ("My", "bottom"), ("Mxy", "bottom"), ("sigma_x", "bottom")):
        out.append(Component(f"moments_{key}", "figure", FIGURE_TITLES[f"moments_{key}"],
                             mt.spec_source(key, surface) if m_ok else None, m_ok, m_reason))
    for key, attr in (("moments_section_x", "tools_x"), ("moments_section_y", "tools_y")):
        tools = getattr(mt, attr, None) if m_ok else None
        ok = tools is not None and _has_points(tools)
        out.append(Component(key, "figure", FIGURE_TITLES[key], tools, ok, "" if ok else (m_reason or "нет значений")))
    comparison = window.comparison.comparison
    refusal = mt.reference_refusal if m_ok else None  # R6-1: a diagnostic field has no moment comparison
    m_ref = m_ok and refusal is None and bool(mt.refs)
    if m_ref:
        ref_reason = ""
    elif not m_ok:
        ref_reason = m_reason
    elif refusal is not None:
        ref_reason = refusal
    elif mt.record:
        ref_reason = f"эталон моментов недоступен: {mt.record.get('reason') or mt.record.get('state')}"
    else:
        ref_reason = "эталон моментов не запрошен (кнопка «Сравнить с эталоном моментов»)"
    out += [
        Component("moments_table", "table", TABLE_TITLES["moments_table"], mt if m_ok else None, m_ok, m_reason),
        Component("moments_comparison_table", "table", TABLE_TITLES["moments_comparison_table"], mt if m_ref else None,
                  m_ref, ref_reason),
    ]
    out += [
        Component("field_table", "table", TABLE_TITLES["field_table"], None, o.has_field, "поле не получено"),
        Component("history_table", "table", TABLE_TITLES["history_table"], None, bool(o.history), "истории нет"),
        Component("comparison_table", "table", TABLE_TITLES["comparison_table"], comparison, comparison.has_numbers,
                  "" if comparison.has_numbers else (comparison.reason or "сравнения с числами нет")),
        Component("vim_cycles_table", "table", TABLE_TITLES["vim_cycles_table"], cycles, cycles.supported,
                  "" if cycles.supported else cycles.route_reason),
        Component("d2_field_table", "table", TABLE_TITLES["d2_field_table"], d2,
                  d2 is not None and d2.get("state") == "ok",
                  "МКР D2 не выбран" if d2 is None else f"D2 не рассчитан: {d2.get('reason') or d2.get('state')}"),
        Component("chart_data", "table", TABLE_TITLES["chart_data"]),
    ]
    return out


def _figure_writer(component, lang, formats, dpi, width_mm, test_data):
    def write(folder):
        files = []
        spec = component.source.spec(lang)
        for fmt in formats:
            path = Path(folder) / "figures" / f"{component.id}_{lang}{'_TEST-DATA' if test_data else ''}.{fmt}"  # 18.2
            pub.save(spec, path, fmt, dpi, width_mm)
            files.append((path, fmt, dpi if fmt == "png" else None))
        return files
    return write


def plan(window, chosen, lang, formats=("png",), dpi=300, pdf=True, raw=True, width_mm=pub.WIDTHS_MM["two_columns"],
         xlsx=False):
    """The ReportPlan of the chosen component ids (unavailable ones stay listed with their reason)."""
    from kkvim_gui.backend import report as package
    from kkvim_gui.backend import vim_cycles

    o, run = window.outcome, window.run
    test_data = o.is_demo
    suffix = "_TEST-DATA" if test_data else ""
    meta = run.meta(run._t0 + (run._elapsed_final or 0.0))
    comps = components(window)
    by_id = {c.id: c for c in comps}
    items = []
    figures = [c for c in comps if c.kind == "figure" and (c.id in chosen or not c.available)]  # 18.3: R3
    for c in figures:
        items.append(report_builder.Item(c.id, "figure", c.title,
                                         _figure_writer(c, lang, formats, dpi, width_mm, test_data) if c.available else None,
                                         c.available, c.reason))

    def table(cid, writer):
        c = by_id[cid]
        if cid in chosen or not c.available:  # an unavailable table is listed with its reason, never written
            items.append(report_builder.Item(cid, "table", c.title, writer if c.available else None, c.available,
                                             c.reason))

    table("field_table", lambda f: [(export.write_field_csv(o, Path(f) / "tables" / export.field_file_name(o)), "csv", None)])
    table("history_table", lambda f: [(export.write_history_csv(o, Path(f) / "tables" / f"history{suffix}.csv"), "csv",
                                       None)])
    table("comparison_table", lambda f: [(package.write_comparison_csv(o, by_id["comparison_table"].source,
                                                                       Path(f) / "tables" / f"comparison{suffix}.csv"),
                                          "csv", None)])

    def cycles_csv(f):
        path = Path(f) / "tables" / f"vim_cycles{suffix}.csv"
        vim_cycles.write_csv(path, [(meta.get("run_id"), by_id["vim_cycles_table"].source)], export._header_lines(o))
        return [(path, "csv", None)]

    table("vim_cycles_table", cycles_csv)
    table("d2_field_table", lambda f: [(package.write_d2_field_csv(o, by_id["d2_field_table"].source,
                                                                   Path(f) / "tables" / f"d2_field{suffix}.csv"), "csv",
                                        None)])
    def moments_files(f):
        import json

        from kkvim_gui.backend import kkvim_adapter

        files = kkvim_adapter.moment_files(o)
        stem = files["csv_name"][:-4]
        path = Path(f) / "tables" / f"{stem}{suffix}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(("# TEST DATA" + chr(10) if test_data else "") + files["csv"], encoding="utf-8", newline="")
        doc = Path(f) / "tables" / f"moments{suffix}.json"
        doc.write_text(json.dumps(dict(files["json"], file=path.name, data_origin=o.origin), ensure_ascii=False,
                                  indent=2, allow_nan=False), encoding="utf-8")
        return [(path, "csv", None), (doc, "json", None)]

    table("moments_table", moments_files)

    def moments_comparison(f):
        from kkvim_gui.backend import moments_view as mv

        tab = by_id["moments_comparison_table"].source
        if tab.reference_refusal is not None:  # R6-1: never written for a diagnostic field (export boundary)
            raise mv.ReferenceRefused(tab.reference_refusal)
        path = Path(f) / "tables" / f"moments_comparison{suffix}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        note = (f"# R6 moments: {tab.data.texts['status']}; reference {tab.ref_choice.currentText()}; "
                "differences are observations (no threshold, no PASS/FAIL)")
        export._write_csv(path, export._header_lines(o) + [note],
                          ["quantity", "place", "kkvim [N]", "reference [N]", "difference [N]", "relative [1]"],
                          tab.comparison_rows())
        return [(path, "csv", None)]

    table("moments_comparison_table", moments_comparison)
    if "chart_data" in chosen:
        def chart_data(f):
            out = []
            for c in figures:
                if c.available and hasattr(c.source, "write_csv"):
                    path = Path(f) / "tables" / f"{c.id}_data{suffix}.csv"
                    out.append((c.source.write_csv(path), "csv", None))
            return out
        items.append(report_builder.Item("chart_data", "table", TABLE_TITLES["chart_data"], chart_data))
    if raw:
        items.append(report_builder.Item("raw_results", "data", "Исходные результаты расчёта: manifest.json, history.csv, "
                                         "файл поля (существующий экспорт)",
                                         lambda f: [(p, Path(p).suffix.lstrip("."), None) for p in
                                                    export.write_result(o, Path(f) / "data", meta, run.detailed).values()]))
    if pdf:
        items.append(report_builder.Item("report_pdf", "pdf", "Общий PDF-отчёт",
                                         lambda f: [(write_pdf(window, Path(f) / f"report{suffix}.pdf", lang, figures,
                                                               items, meta), "pdf", None)]))
    options = {"formats": list(formats), "png_dpi": dpi, "figure_width_mm": width_mm, "pdf": pdf, "raw_results": raw,
               "xlsx": bool(xlsx), "chosen": sorted(chosen)}
    return report_builder.ReportPlan(o, lang, items, meta, tuple(run.detailed), options)


def build_report(window, parent, chosen, lang, **kw):
    """Build the report into a new folder under ``parent`` (never an existing one); returns the build result."""
    p = plan(window, chosen, lang, **kw)
    name = report_builder.folder_name(window.export_base(), lang, window.outcome.is_demo)
    return report_builder.build(p, report_builder.free_folder(parent, name))


# ------------------------------------------------------------------------------------------------ PDF


def write_pdf(window, path, lang, figures, items, meta):
    from kkvim_gui.analysis.cycles_tab import table_row
    from kkvim_gui.analysis.results import MM, fmt, status_details, statuses
    from kkvim_gui.backend import report as package
    from kkvim_gui.figures.report_pdf import PdfReport

    o, run = window.outcome, window.run
    r = o.request
    marks = export.marks(o)
    header = f"KK-VIM · {window.export_base()} · {r.supports}, N = {r.N}, {r.n_x} × {r.n_y}"
    doc = PdfReport(path, "KK-VIM — отчёт расчёта", header, marks, o.is_demo,
                    meta={"run_id": meta.get("run_id"), "data_origin": o.origin, "figure_language": lang,
                          "solver": o.solver, "marks": marks})
    doc.heading("KK-VIM — отчёт расчёта", 0)
    doc.paragraph(package.PREVIEW_NOTICE, 7.5, "#555555")
    if o.is_demo:
        doc.paragraph("TEST DATA — демонстрационные данные, не получены вычислительным алгоритмом.", 9, "#8A1C1C")
    doc.heading("1. Расчёт", 1)
    doc.key_values([("Расчёт", f"№ {run.run_id} · {run.started:%d.%m.%Y %H:%M:%S}"), ("Происхождение данных", o.origin),
                    ("Решатель", f"{o.solver}: {o.call}"), ("Маршрут и решения", f"{o.route} · {', '.join(o.decisions)}"),
                    ("Версия программы", pub.program_version()),
                    ("Версии", ", ".join(f"{k} {v}" for k, v in o.versions.items()) or "—")])
    doc.heading("2. Три статуса (раздельно)", 1)
    (k1, t1), (k2, t2), (k3, t3) = statuses(o, run)
    d1, d2, d3 = status_details(o)
    doc.key_values([("Выполнение расчёта", f"{t1}. {d1}"), ("Критерий остановки", d2),
                    ("Научная приёмка", d3), ("Отметка", o.mark or "—"),
                    ("Роль поля", export.ROLE_TEXT.get(o.field_role, "поле не получено"))])
    doc.heading("3. Входные данные (СИ)", 1)
    doc.key_values([("Пластина", f"a = {r.a!r} m · b = {r.b!r} m · h = {r.t!r} m"),
                    ("Материал", f"E = {r.E!r} Pa · ν = {r.nu!r}"), ("Нагрузка", f"равномерная, q = {r.q!r} Pa"),
                    ("Закрепление", r.supports), ("Метод", f"КК-ВИМ, N = {r.N}, теория Кирхгофа–Лява"),
                    ("Сетка", f"n_x = {r.n_x}, n_y = {r.n_y}")])
    doc.heading("4. Ключевые значения (значения решателя в узлах)", 1)
    e = o.extremum()
    s = package.summary(o)
    rows = [("Жёсткость D", f"{o.D!r} N·m"), ("Полушаги", f"s = {o.s}, k = {o.k}" +
                                                (f", запрошено {o.s_requested}" if o.s_requested else ""))]
    if e is not None:
        rows.append(("max |w| по узлам", f"{fmt(e['w'] * MM, 8)} mm = {e['w']!r} m в узле i = {e['i']}, j = {e['j']}"))
    if s["w_centre_node"]:
        c = s["w_centre_node"]
        rows.append(("w в центре (узел)", f"{c['w']!r} m в узле i = {c['i']}, j = {c['j']}"))
    doc.key_values(rows)
    comparison = window.comparison.comparison
    doc.heading("5. Сравнение с независимыми эталонами", 1)
    if comparison.has_numbers:
        doc.table(("Эталон", "β эталона", "β КК-ВИМ", "Δβ", "Δβ/β, %", "Источник"),
                  [(x.reference.title, repr(x.reference.beta), repr(x.beta_kkvim), repr(x.delta), fmt(x.rel_pct, 6),
                    "executed" if x.reference.kind in ("navier_live", "D2_live") else "stored") for x in comparison.rows])
        doc.paragraph("Знаковые разности без порога согласия; сравнение не является научной приёмкой и не меняет статус.",
                      7.5, "#555555")
    else:
        doc.paragraph(comparison.reason or "Сравнения с числами нет.")
    table = window.cycles.table
    doc.heading("6. Циклы VIM-1–3", 1)
    if table.supported:
        states = (table.partial, *table.cycles)
        doc.table(("Состояние", "k", "s", "w, mm", "Δw/w_эт, %", "На графике", "Примечание"),
                  [(lambda t: (t[0], t[1], t[2], fmt(t[4], 8), fmt(t[9], 6), "да" if t[10] else "нет", t[11]))(
                      table_row(c, table)) for c in states])
        doc.paragraph("VIM-1 = s = 2, VIM-2 = s = 4, VIM-3 = s = 6 — обозначения интерфейса, не отдельные методы и не N; "
                      "отсутствующие циклы не подставляются.", 7.5, "#555555")
    else:
        doc.paragraph(table.route_reason)
    doc.heading("7. Рисунки", 1)
    n = 0
    for c in figures:
        if not c.available:
            doc.paragraph(f"{c.title}: недоступно — {c.reason}", 8, "#8A1C1C")
            continue
        n += 1
        doc.figure(c.source.spec(lang), f"Рис. {n}. {c.title}. Язык рисунка: {lang.upper()}; только видимые кривые; "
                                        "значения — сохранённые значения расчёта, без интерполяции.")
    _moments_section(doc, window, o)
    doc.heading("9. История полушагов", 1)
    keys = [col for col in o.columns if col.kind == "number"][:7]
    doc.table([f"{col.key}" + (f", {col.unit}" if col.unit else "") for col in keys],
              [[fmt(row.get(col.key), 6) for col in keys] for row in o.history])
    doc.heading("10. Ограничения и примечания", 1)
    for text in (*o.limitations, *o.notes, *package.LIMITATIONS):
        doc.paragraph(f"• {text}", 7.5)
    doc.heading("11. Состав папки отчёта", 1)
    doc.paragraph("manifest.json — входные данные, версии, статусы, происхождение и SHA256 каждого файла. "
                  + "; ".join(i.title for i in items if i.available and i.kind != "pdf"), 7.5)
    doc.close()
    return path


def _num(value):
    return "—" if value is None else repr(value)


def _moments_section(doc, window, o):
    """Section 8 of the PDF: the R6 moments and stresses (status, conventions, summary, comparison, limitations)."""
    mt = getattr(window, "moments", None)
    doc.heading("8. Моменты и напряжения (R6, исследование)", 1)
    if mt is None or mt.unavailable_reason:
        doc.paragraph(f"Недоступно: {mt.unavailable_reason if mt is not None else 'вкладки нет'}. Ничего не подставляется.")
        return
    from kkvim_gui.analysis.moments_tab import summary_rows
    from kkvim_gui.backend import moments_view as mv

    texts = mt.data.texts
    text = f"Статус: {texts['status']}. Статус прогиба ({o.status}) на моменты и напряжения не переносится."
    if o.diagnostic_only:
        text += " ДИАГНОСТИКА: моменты поля последнего выполненного полушага, не результат."
    doc.paragraph(text, 8, "#8A1C1C")
    doc.key_values([("Соглашения", texts["conventions"]), ("Граничная кривизна CCCC", texts["cc_edge_rule"]),
                    ("Модель округления", f"{mt.data.moments.rounding_model!r} N (модель, не граница)"),
                    ("Не вычисляется", "; ".join(mv.NOT_COMPUTED))])
    doc.table(("Величина", "Ед. СИ", "Центр", "Max", "Узел max", "Min", "Узел min"),
              [(r[0], r[1], _num(r[2]), _num(r[3]), r[4], _num(r[5]), r[6]) for r in summary_rows(mt)])
    if mt.reference_refusal is not None:  # R6-1: no comparison section of a diagnostic field
        doc.paragraph(f"Сравнение с эталоном моментов: {mt.reference_refusal}", 7.5, "#8A1C1C")
    elif mt.refs:
        rows = [r for r in mt.comparison_rows() if r[0] in ("Mx", "My", "Mxy")]
        refs = " ".join(f"{ref['title']}: {ref['status']}." for ref in mt.refs.values())
        doc.paragraph(f"Сравнение с эталоном «{mt.ref_choice.currentText()}»: разности — наблюдения, без порога и без "
                      f"PASS/FAIL. {refs}", 7.5, "#555555")
        doc.table(("Величина", "Место", "КК-ВИМ, N", "Эталон, N", "Разность, N", "Отн."),
                  [(r[0], r[1], _num(r[2]), _num(r[3]), _num(r[4]), _num(r[5])) for r in rows])
    else:
        doc.paragraph("Эталон моментов для этого отчёта не запрашивался.", 7.5, "#555555")
    for line in mv.LIMITATIONS_RU:
        doc.paragraph(f"• {line}", 7.5)


# ------------------------------------------------------------------------------------------------ dialog


class ReportDialog(QDialog):
    """The constructor (18.1): components, formats, PNG dpi, XLSX (not available), figure language, folder."""

    def __init__(self, window, parent=None):
        super().__init__(parent or window)
        self.window_ = window
        self.setWindowTitle("Собрать отчёт")
        self.comps = components(window)
        lay = QVBoxLayout(self)
        body = QHBoxLayout()
        left = QVBoxLayout()
        self.boxes = {}
        for kind, title in (("figure", "Рисунки"), ("table", "Таблицы и данные")):
            group = QGroupBox(title)
            g = QVBoxLayout(group)
            for c in (c for c in self.comps if c.kind == kind):
                box = QCheckBox(c.title)
                box.setChecked(c.available)
                box.setEnabled(c.available)
                if not c.available:
                    box.setText(f"{c.title} — недоступно")
                    box.setToolTip(c.reason)
                g.addWidget(box)
                self.boxes[c.id] = box
            self.raw = None
            if kind == "table":
                self.raw = QCheckBox("Исходные результаты: manifest.json, history.csv, файл поля (JSON + CSV)")
                self.raw.setChecked(True)
                g.addWidget(self.raw)
            left.addWidget(group)
        left.addStretch(1)
        holder = QWidget()
        holder.setLayout(left)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(holder)
        body.addWidget(scroll, 3)
        right = QGridLayout()
        self.pdf = QCheckBox("Общий PDF-отчёт")
        self.pdf.setChecked(True)
        right.addWidget(self.pdf, 0, 0, 1, 2)
        right.addWidget(QLabel("Рисунки"), 1, 0)
        fmts = QHBoxLayout()
        self.formats = {f: QCheckBox(f.upper()) for f in ("png", "svg", "pdf")}
        self.formats["png"].setChecked(True)
        for b in self.formats.values():
            fmts.addWidget(b)
        right.addLayout(fmts, 1, 1)
        right.addWidget(QLabel("Качество PNG"), 2, 0)
        self.dpi = QComboBox()
        for d in pub.DPI_CHOICES:
            self.dpi.addItem(f"{d} dpi", d)
        right.addWidget(self.dpi, 2, 1)
        self.xlsx = QCheckBox("Таблицы XLSX")
        self.xlsx.setEnabled(False)
        self.xlsx.setToolTip(f"{report_builder.XLSX_STATUS}. Рекомендация: {report_builder.XLSX_RECOMMENDATION}")
        right.addWidget(self.xlsx, 3, 0, 1, 2)
        right.addWidget(QLabel("XLSX недоступен: нужна необязательная зависимость, не утверждённая владельцем; таблицы — CSV."),
                        4, 0, 1, 2)
        right.addWidget(QLabel("Язык рисунков"), 5, 0)
        langs = QHBoxLayout()
        self.ru, self.en = QRadioButton("RU"), QRadioButton("EN")
        (self.ru if window.session.figure_language == "ru" else self.en).setChecked(True)
        langs.addWidget(self.ru)
        langs.addWidget(self.en)
        right.addLayout(langs, 5, 1)
        right.addWidget(QLabel("Папка"), 6, 0)
        folder = QHBoxLayout()
        self.folder = QLineEdit(str(Path.home() / "Documents"))
        pick = QPushButton("Выбрать…")
        pick.clicked.connect(self._pick)
        folder.addWidget(self.folder, 1)
        folder.addWidget(pick)
        right.addLayout(folder, 6, 1)
        self.target = QLabel()
        self.target.setWordWrap(True)
        right.addWidget(self.target, 7, 0, 1, 2)
        right.setRowStretch(8, 1)
        for w in (self.folder.textChanged, self.ru.toggled):
            w.connect(self._show_target)
        side = QWidget()
        side.setLayout(right)
        body.addWidget(side, 2)
        lay.addLayout(body, 1)
        buttons = QDialogButtonBox()
        self.build_button = buttons.addButton("Собрать", QDialogButtonBox.ButtonRole.AcceptRole)
        buttons.addButton("Отмена", QDialogButtonBox.ButtonRole.RejectRole)
        self.build_button.clicked.connect(self._build)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)
        self._show_target()
        self.resize(900, 560)

    def lang(self):
        return "ru" if self.ru.isChecked() else "en"

    def chosen(self):
        return {cid for cid, box in self.boxes.items() if box.isChecked() and box.isEnabled()}

    def _pick(self):
        path = QFileDialog.getExistingDirectory(self, "Папка для отчёта", self.folder.text())
        if path:
            self.folder.setText(path)

    def _show_target(self, *_):
        name = report_builder.folder_name(self.window_.export_base(), self.lang(), self.window_.outcome.is_demo)
        target = report_builder.free_folder(self.folder.text() or ".", name)
        self.target.setText(f"Отчёт будет записан в новую папку: {target}. Существующие папки не перезаписываются.")

    def build(self):
        formats = tuple(f for f, b in self.formats.items() if b.isChecked()) or ("png",)
        return build_report(self.window_, self.folder.text(), self.chosen(), self.lang(), formats=formats,
                            dpi=self.dpi.currentData(), pdf=self.pdf.isChecked(), raw=self.raw.isChecked())

    def _build(self):
        try:
            result = self.build()
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Отчёт не собран", str(exc))
            return
        self.result = result
        QMessageBox.information(self, "Отчёт собран", f"Папка отчёта: {result['folder']}\nФайлов: {len(result['files'])} "
                                                      "и manifest.json")
        self.accept()

