"""Saved independent benchmark results as data, and their comparison with one solver run (GUI_DESIGN.md, 3.1, 6). Qt-free.

The GUI does not import ``kkvim_reference`` and computes no reference: ``benchmarks_v1.json`` (next to this module,
installed with the package) holds values copied at full precision from completed, SHA256-checked experiments:
- CCCC: published Taylor-Govindjee coefficients (L-120), D2 (the QA implementation of the SAME discrete problem as
  KK-VIM, D-049.1, D-049.8) and FEM (scikit-fem Morley: a DIFFERENT discretization of the same plate model), b/a = 1.0,
  1.2, 1.4, 2.0; D2 and FEM only on the stored grids n = 16, 32, 64;
- SSSS: the qualified Navier reference of the square (D-037, D-043).

SSSS runs also get the Navier reference computed for exactly their own inputs (``backend.navier_job``: a separate
process running ``kkvim_reference.navier`` at the plate centre, M = 2559; the GUI imports neither). ``compare`` takes
that point record (``navier=``) and lays it out with the same rows, cells and states as the saved references; it never
derives a reference value from the KK-VIM field.

A comparison is shown only when the inputs passed to the solver equal the reference inputs exactly (no interpolation,
extrapolation or rescaling), the run ended as requested (never a diagnostic field) and the plate centre is a node of the
result grid. It gives signed differences, never a verdict: no agreement threshold is approved, and the comparison never
changes the scientific status of the run.
"""

import hashlib
import json
from dataclasses import dataclass, field
from importlib import resources

import numpy as np

DATA_FILE = "benchmarks_v1.json"
SCHEMA = "kkvim-gui-benchmarks/1"
# SHA256 of the canonical JSON of the payload (independent of line endings of the checkout)
PAYLOAD_SHA256 = "3543073f2b0f642a9e3bb0a571c400e9d2dc3eba1bda436755d5ae9bbb27b568"

AVAILABLE = "available"  # at least one reference matches: numbers are shown
UNAVAILABLE = "unavailable"  # no reference for these inputs or no exact centre node
DIAGNOSTIC = "diagnostic"  # the run did not end as requested: no comparison of a diagnostic field
DATA_ERROR = "data_error"  # the packaged data are missing or corrupted
NO_RUN = "no_run"  # no solver result (demonstration or no field)
PENDING = "pending"  # the Navier reference of these inputs is still being computed in its own process

KIND_TITLE = {"published": "Опубликованное решение (Taylor, Govindjee 2002)", "D2": "D2 (та же дискретная задача)",
              "FEM": "МКЭ, элемент Морли (другая дискретизация)", "navier": "Аналитическое решение Навье",
              "navier_live": "Навье — рассчитан для этих входных данных"}
STATE_TITLE = {
    AVAILABLE: "Сравнение с сохранёнными эталонами: знаковые разности, без порога согласия",
    UNAVAILABLE: "Эталонные данные для этого расчёта недоступны",
    DIAGNOSTIC: "Сравнение не выполняется: поле диагностическое",
    DATA_ERROR: "Данные эталонов повреждены или отсутствуют",
    NO_RUN: "Нет результата решателя",
    PENDING: "Эталон Навье вычисляется в отдельном процессе",
}
# one layout of the reference table for the "Сравнение" tab and report.html (comparison.csv carries the same values)
HEADERS = ("Эталон", "β эталона", "w эталона в центре, m", "Δβ = β_КК-ВИМ − β_эталона", "Δβ / β_эталона, %",
           "Δw = w_КК-ВИМ − w_эталона, m", "Метод, сетка, происхождение")
DETAIL_COLUMN = HEADERS.index("Метод, сетка, происхождение")


class BenchmarkDataError(Exception):
    """The packaged benchmark data cannot be used (missing, unreadable, wrong schema or checksum)."""


def canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def load(text: str | None = None) -> dict:
    """The verified payload of the packaged data (or of ``text``); raises BenchmarkDataError otherwise."""
    try:
        if text is None:
            text = resources.files("kkvim_gui.backend").joinpath(DATA_FILE).read_text(encoding="utf-8")
        doc = json.loads(text)
        payload, digest = doc["payload"], doc["payload_sha256"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise BenchmarkDataError(f"данные эталонов не прочитаны: {exc}") from exc
    actual = hashlib.sha256(canonical(payload)).hexdigest()
    if actual != digest or digest != PAYLOAD_SHA256:
        raise BenchmarkDataError("контрольная сумма данных эталонов не совпадает: данные изменены или повреждены")
    if payload.get("schema") != SCHEMA:
        raise BenchmarkDataError(f"неизвестная схема данных эталонов {payload.get('schema')!r}")
    return payload


@dataclass(frozen=True)
class Reference:
    kind: str  # "published" | "D2" | "FEM" | "navier"
    title: str
    beta: float
    w_centre_m: float | None  # stored centre deflection of the reference, m (None for the published table)
    detail: str  # method, mesh and provenance in one line


@dataclass(frozen=True)
class Row:
    reference: Reference
    beta_kkvim: float
    delta: float  # beta_KK-VIM - beta_ref (dimensionless)
    rel_pct: float  # delta / beta_ref * 100 (denominator: the reference)
    delta_w_m: float | None = None  # w_KK-VIM - w_ref at the centre, m (None without a stored reference deflection)


@dataclass(frozen=True)
class Comparison:
    state: str
    reason: str = ""
    supports: str = ""
    centre: dict | None = None  # i, j, x, y of the centre node
    w_centre_m: float | None = None  # KK-VIM deflection at the centre node, m
    beta: float | None = None  # w_centre * D / (q a^4) of the run
    w_bar_solver: float | None = None  # w_bar of the solver at the same node (MATH_SPEC 2.5), shown as a cross-check
    D: float | None = None
    rows: tuple = ()
    missing: tuple = ()  # (kind, reason) of references without data for these inputs
    notes: tuple = ()
    data: dict | None = field(default=None, repr=False)
    navier: dict | None = field(default=None, repr=False)  # point record of the Navier reference computed for this run

    @property
    def live(self) -> bool:
        """True if the reference was computed for this run (Navier worker), not read from the saved data."""
        return self.navier is not None and self.navier.get("state") == "ok" and bool(self.rows)

    @property
    def has_numbers(self) -> bool:
        return self.state == AVAILABLE and bool(self.rows)


INPUT_KEYS = (("a", "a, m"), ("t", "h, m"), ("E", "E, Pa"), ("nu", "ν"), ("q", "q, Pa"))


def _input_mismatches(request, inputs, b_values):
    """Names of the inputs that differ from the reference inputs (exact comparison of the SI values)."""
    wrong = [f"{name} = {getattr(request, key)!r} (эталон {inputs[key]!r})" for key, name in INPUT_KEYS
             if getattr(request, key) != inputs[key]]
    if request.b not in b_values:
        wrong.append(f"b, m = {request.b!r} (эталоны: {', '.join(repr(b) for b in b_values)})")
    return wrong


def centre_node(outcome):
    """Indices and coordinates of the node exactly at (a/2, b/2), or None."""
    r = outcome.request
    i = np.flatnonzero(outcome.x == r.a / 2)
    j = np.flatnonzero(outcome.y == r.b / 2)
    if len(i) != 1 or len(j) != 1:
        return None
    return {"i": int(i[0]), "j": int(j[0]), "x": float(outcome.x[i[0]]), "y": float(outcome.y[j[0]])}


def _notes(supports):
    common = ("Утверждённого порога согласия нет: показаны знаковые разности, без оценки «точно / неточно».",
              "Сравнение не меняет научного статуса расчёта КК-ВИМ и не делает его принятым (ACCEPTED).")
    if supports == "SSSS":
        return ("Навье: двойной ряд синусов, M = 2559 членов, оценка остатка в β указана в строке (D-037); квалифицированный "
                "эталон R3 (D-043), рассчитан вне GUI и прочитан как данные. Это континуальное решение: разность "
                "включает сеточную погрешность КК-ВИМ.",) + common
    return ("D2 — независимая реализация той же дискретной задачи, что и КК-ВИМ CCCC (D-049.1, D-049.8): совпадение с D2 "
            "не проверяет сеточную погрешность.",
            "МКЭ (scikit-fem 12.0.2, элемент Морли): другая дискретизация той же физической модели Кирхгофа–Лява; "
            "n × n ячеек МКЭ (2n² треугольников) не равны сетке МКР n × n.",
            "Опубликованное решение — континуальное (Taylor, Govindjee 2002, табл. I–II), 10 значащих цифр.",
            "Три уровня сетки (16, 32, 64) не дают ни строгой оценки погрешности, ни порядка сходимости. Знак разности "
            "с опубликованным решением не всегда положителен (например, N = 1, s = 8, b/a = 2, сетка 64: КК-ВИМ ниже).") \
        + common


CENTRE_MISSING = ("Центр пластины (a/2, b/2) не является узлом сетки результата (нужны чётные n_x, n_y): значение в центре "
                  "не интерполируется.")


def _live_notes(outcome, nav):
    route_note = (("Маршрут этого расчёта — PROVISIONAL / NOT_QUALIFIED или только исследование: Навье здесь — справочные "
                   "данные; сравнение не повышает его статус.",) if outcome.mark else ())
    return ((f"Эталон Навье рассчитан для точно тех же входных данных (a, b, h, E, ν, q) в отдельном процессе: двойной ряд "
             f"синусов, M = {nav['M']}, в точке (a/2, b/2); он не использует КК-ВИМ и его поле.",
             "Навье — квалифицированный эталон прогиба SSSS при равномерной нагрузке (D-037: проверка на десяти контрольных "
             "b/a; область 0,2 ≤ b/a ≤ 5 по D-036, D-038). Моменты не сравниваются.",
             "Это континуальное решение: разность включает сеточную погрешность КК-ВИМ и не является оценкой точности метода.")
            + route_note
            + ("Утверждённого порога согласия нет: показаны знаковые разности, без оценки «точно / неточно».",
               "Сравнение не меняет научного статуса расчёта КК-ВИМ и не делает его принятым (ACCEPTED)."))


def _compare_live(outcome, nav) -> Comparison:
    """The run next to the Navier reference computed for its own inputs (``backend.navier_job`` point record)."""
    r, supports = outcome.request, outcome.request.supports
    state = nav.get("state")
    if state == "pending":
        return Comparison(PENDING, "Эталон Навье для этих входных данных ещё вычисляется в отдельном процессе.", supports,
                          navier=nav)
    if state != "ok":
        return Comparison(UNAVAILABLE, f"Эталон Навье недоступен — {nav.get('reason') or state}. Значение не подставляется.",
                          supports, navier=nav)
    if any(nav[key] != getattr(r, key) for key in ("a", "b", "t", "E", "nu", "q")):
        return Comparison(UNAVAILABLE, "Эталон Навье рассчитан для других входных данных: не сравнивается.", supports,
                          navier=nav)
    centre = centre_node(outcome)
    if centre is None:
        return Comparison(UNAVAILABLE, CENTRE_MISSING, supports, navier=nav)
    w_c = float(outcome.w[centre["i"], centre["j"]])
    beta = w_c * outcome.D / (r.q * r.a ** 4)
    w_bar = None if outcome.w_bar is None else float(outcome.w_bar[centre["i"], centre["j"]])
    detail = (f"двойной ряд синусов, M = {nav['M']}, в точке x = {nav['x']!r} m, y = {nav['y']!r} m; оценка остатка "
              f"и округления в β {nav['error_bound_coef']!r}; D эталона = {nav['D']!r} N·m; рассчитан в отдельном процессе "
              f"({nav['method']})")
    ref = Reference("navier_live", KIND_TITLE["navier_live"], nav["w_coef"], nav["w_m"], detail)
    row = Row(ref, beta, beta - ref.beta, (beta - ref.beta) / ref.beta * 100.0, w_c - ref.w_centre_m)
    return Comparison(AVAILABLE, "", supports, centre, w_c, beta, w_bar, outcome.D, (row,), (), _live_notes(outcome, nav),
                      navier=nav)


def compare(outcome, text: str | None = None, navier: dict | None = None) -> Comparison:
    """Comparison of a solver run with the references.

    ``navier``: the point record of the Navier reference computed for this SSSS run (``backend.navier_job``); without it
    the saved references are used. ``text``: data instead of the packaged file (tests).
    """
    if outcome is None or outcome.is_demo or not outcome.has_field:
        return Comparison(NO_RUN, "Нет поля решателя КК-ВИМ: сравнивать нечего.")
    supports = outcome.request.supports
    if outcome.diagnostic_only or outcome.is_error or not outcome.requested_completed:
        return Comparison(DIAGNOSTIC, f"Расчёт не завершён так, как запрошено ({outcome.status}): поле — диагностическое, "
                                      "с эталонами не сравнивается.", supports)
    if supports == "SSSS" and navier is not None:
        return _compare_live(outcome, navier)
    try:
        data = load(text)
    except BenchmarkDataError as exc:
        return Comparison(DATA_ERROR, f"Сравнение недоступно: {exc}.", supports)
    r = outcome.request
    key = "cccc" if supports == "CCCC" else "ssss" if supports == "SSSS" else None
    cases = data.get(key, []) if key else []
    wrong = _input_mismatches(r, data["inputs"], [c["b"] for c in cases])
    if not cases or wrong:
        return Comparison(UNAVAILABLE, "Эталонных данных для этих входных данных нет: " + "; ".join(wrong or
                          [f"закрепление {supports}"]) + ". Значения не подставляются и не пересчитываются.",
                          supports, data=data)
    centre = centre_node(outcome)
    if centre is None:
        return Comparison(UNAVAILABLE, CENTRE_MISSING, supports, data=data)
    case = next(c for c in cases if c["b"] == r.b)
    w_c = float(outcome.w[centre["i"], centre["j"]])
    beta = w_c * outcome.D / (r.q * r.a ** 4)
    w_bar = None if outcome.w_bar is None else float(outcome.w_bar[centre["i"], centre["j"]])
    refs, missing = [], []
    if supports == "CCCC":
        p = case["published"]
        refs.append(Reference("published", KIND_TITLE["published"], p["beta"], None,
                              f"{p['table']}; напечатано {p['printed']}; континуальное решение"))
        grid = next((g for g in case["grids"] if (g["n_x"], g["n_y"]) == (r.n_x, r.n_y)), None)
        if grid is None:
            have = ", ".join(f"{g['n_x']}×{g['n_y']}" for g in case["grids"])
            for kind in ("D2", "FEM"):
                missing.append((kind, f"нет сохранённых данных для сетки {r.n_x}×{r.n_y} (есть {have})"))
        else:
            d2, fem = grid["D2"], grid["FEM"]
            refs.append(Reference("D2", KIND_TITLE["D2"], d2["beta"], d2["w_centre_m"],
                                  f"МКР, 13 точек, сетка {r.n_x}×{r.n_y}; e_D2 = {d2['e_D2_m']!r} m; {d2['source']}"))
            refs.append(Reference("FEM", KIND_TITLE["FEM"], fem["beta"], fem["w_centre_m"],
                                  f"scikit-fem 12.0.2, Морли, {r.n_x}×{r.n_y} ячеек, {fem['triangles']} треугольников, "
                                  f"{fem['dofs']} степеней свободы; {fem['solver_status']}"))
    else:
        nav = case["navier"]
        refs.append(Reference("navier", KIND_TITLE["navier"], nav["beta"], nav["w_centre_m"],
                              f"двойной ряд синусов, M = {nav['M']}, оценка остатка в β {nav['error_bound_beta']!r}"))
    rows = tuple(Row(ref, beta, beta - ref.beta, (beta - ref.beta) / ref.beta * 100.0,
                     None if ref.w_centre_m is None else w_c - ref.w_centre_m) for ref in refs)
    return Comparison(AVAILABLE, "", supports, centre, w_c, beta, w_bar, outcome.D, rows, tuple(missing),
                      _notes(supports), data)


def full(value) -> str:
    """Full-precision text of a number (shortest repr that round-trips the binary64 value); "—" for None."""
    return "—" if value is None else repr(float(value))


def cells(row: Row) -> tuple:
    """Texts of one reference row in the order of HEADERS (the same in the tab and in report.html)."""
    ref = row.reference
    return (ref.title, full(ref.beta), full(ref.w_centre_m), full(row.delta), full(row.rel_pct), full(row.delta_w_m),
            ref.detail)


LIVE_TITLE = "Сравнение с эталоном Навье, рассчитанным для этих входных данных: знаковые разности, без порога согласия"


def state_title(c: Comparison) -> str:
    """The title of the state of ``c`` (the live Navier reference is not a saved one)."""
    return LIVE_TITLE if c.live else STATE_TITLE[c.state]


def source_text(c: Comparison) -> str:
    """Where the reference values of ``c`` come from (the report and comparison.csv)."""
    if c.live:
        return f"рассчитан для этого расчёта в отдельном процессе: {c.navier['method']}; {c.navier['source']}"
    return dataset_text()


def dataset_text() -> str:
    """Version and integrity identifier of the reference data of this program."""
    return f"{DATA_FILE}, схема {SCHEMA}, SHA256 содержимого {PAYLOAD_SHA256}"


def as_record(c: Comparison) -> dict:
    """JSON-ready record of a comparison (full-precision floats)."""
    return {"state": c.state, "reason": c.reason, "supports": c.supports, "centre": c.centre, "w_centre_m": c.w_centre_m,
            "beta": c.beta, "w_bar_solver": c.w_bar_solver, "D": c.D,
            "rows": [{"kind": x.reference.kind, "beta_ref": x.reference.beta, "w_ref_m": x.reference.w_centre_m,
                      "beta_kkvim": x.beta_kkvim, "delta": x.delta, "rel_pct": x.rel_pct, "delta_w_m": x.delta_w_m,
                      "detail": x.reference.detail}
                     for x in c.rows],
            "live_reference": c.live,
            "missing": [list(m) for m in c.missing]}
