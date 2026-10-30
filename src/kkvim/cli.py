"""Command line run of the accepted R3 solver (D-043): rectangle a x b, SSSS, q = const, N = 1.

    python -m kkvim run --a 1 --b 1 --t 0.01 --E 2.1e11 --nu 0.3 --q 1000 --nx 24 --ny 24 --out scratch/demo

The command calls ``rectangle_run.run_rectangle_physical`` and ``rectangle_run.halfstep_report`` unchanged and
writes what they return. It computes no quantity of the method itself:

- the input checks, D, the stopping rule (NUMERICAL_SPEC.md 11.3) and w_bar = w D / (q a^4) are those of kkvim;
- the status of the solver (CONVERGED, STAGNATED, MAX_HALF_STEPS, ERROR_*) is written unchanged;
- ``field.csv`` is written only for CONVERGED. Any other run with a completed half-step gets
  ``field_diagnostic.csv`` (marked: the field of the last half-step, not a computed solution);
- existing result files are never overwritten without ``--overwrite``; every file is written through a new
  temporary file, so no result file is ever half-written and no file of another name is touched. The set of files
  is not replaced atomically: a failure in the middle may leave files of two runs, and the command then says
  exactly which files were changed (``write_results``);
- an error of creating the folder or writing the files is exit code 4, an unexpected internal error is exit code 5;
  neither is reported as a status of the solver;
- ``result.json`` lists the necessary conditions of a qualification grid (R3_NUMERICAL_SPEC.md 6.1, D-039) as
  plain facts about n_x and n_y; the command makes no qualification verdict about a run.

Values are written with ``repr`` and read back bit for bit; non-finite values are empty (CSV) or null (JSON).
Units are SI: lengths in m, E and q in Pa, D in N·m, w in m.
"""

import argparse
import csv
import io
import json
import math
import os
import sys
import tempfile
import traceback
from dataclasses import asdict
from pathlib import Path

from kkvim.controller import CONVERGED, MAX_HALF_STEPS, S_MAX, S_MIN, STAGNATED, TOL_E, TOL_INF
from kkvim.rectangle_run import CENTRE, halfstep_report, run_rectangle_physical

SCHEMA = "kkvim.cli.r3/1"
RESULT, HISTORY, FIELD, FIELD_DIAGNOSTIC = "result.json", "history.csv", "field.csv", "field_diagnostic.csv"
OUTPUTS = (RESULT, HISTORY, FIELD, FIELD_DIAGNOSTIC)
CALLS = ("kkvim.rectangle_run.run_rectangle_physical", "kkvim.rectangle_run.halfstep_report")

# exit codes
EXIT_CONVERGED = 0
EXIT_NOT_CONVERGED = 1  # STAGNATED or MAX_HALF_STEPS: the run ended without the stopping criterion
EXIT_USAGE = 2  # command line errors (argparse) and refused overwriting; nothing is computed
EXIT_SOLVER_ERROR = 3  # ERROR_* of the solver, ERROR_INPUT included
EXIT_OUTPUT_ERROR = 4  # the results could not be saved (folder or file I/O); not a status of the solver
EXIT_INTERNAL_ERROR = 5  # unexpected internal error of the program; not a status of the solver

STATUS_NOTES = {
    CONVERGED: "Критерий остановки NUMERICAL_SPEC.md 11.3, п. 2 выполнен. Это не научная приёмка отдельного расчёта "
               "и не доказательство сходимости последовательности при s → ∞ (D-043, ограничение 4).",
    STAGNATED: "Не сошлось: застой по NUMERICAL_SPEC.md 11.3, п. 3. Поле последнего полушага — только диагностика.",
    MAX_HALF_STEPS: "Не сошлось: достигнуто s_max без выполнения критерия остановки (NUMERICAL_SPEC.md 11.3, п. 4). "
                    "Поле последнего полушага — только диагностика.",
}
ERROR_NOTE = ("Ошибочный выход решателя (NUMERICAL_SPEC.md 11.3, п. 5). Решения нет; поле последнего выполненного "
              "полушага, если есть, — только диагностика.")
APPLICABILITY = (
    "Решатель R3 ACCEPTED (D-043): линейная прямоугольная пластина Кирхгофа–Лява, однородный изотропный материал, "
    "SSSS, равномерная нагрузка q = const, N = 1.",
    "Решатель проверяет 0,2 ≤ b/a ≤ 5 (D-038); вне диапазона — ERROR_INPUT.",
    "Численная квалификация относится только к десяти утверждённым отношениям сторон 1/5, 1/3, 1/2, 2/3, 1, 7/5, 3/2, "
    "2, 3, 5 и к установленным семействам сеток I (n_x = n_y) и II (h_x = h_y) квалификационного набора "
    "R3_NUMERICAL_SPEC.md 6.2 (D-039; D-043, ограничение 5). Вне этой области расчёт выполняется как диагностический "
    "и не квалифицирован.",
    "Не доказаны сходимость при s → ∞, единственность предела и независимость от старта (D-043, ограничение 4; Q-025).",
    "Моменты и напряжения R3 не квалифицированы (D-034, Q-027) и не вычисляются.",
    "CCCC, N ≥ 2 и нелинейные постановки вне R3. Успешный запуск, сходимость итераций и научная приёмка — разные "
    "статусы (AGENTS.md, разд. 4).",
)
GRID_CONDITIONS_NOTE = (
    "Необходимые условия квалификационной сетки R3_NUMERICAL_SPEC.md 6.1 (D-039): min(n_x, n_y) ≥ 20, "
    "max(n_x, n_y) ≤ 640, n_x и n_y кратны 4. Здесь записано только, выполнено ли каждое из них для n_x, n_y этого "
    "расчёта. Их выполнение не означает, что сетка входит в квалификационный набор 6.2 (десять утверждённых "
    "отношений сторон, семейства I и II): отношение сторон и семейство командой не проверяются. Принадлежность "
    "сетки к квалификационному набору сама по себе не означает научной приёмки конкретного расчёта. Заключения "
    "QUALIFIED или PASS команда не делает. Сетки, не удовлетворяющие условиям, — только диагностика."
)
INPUT_UNITS = (("a", "m"), ("b", "m"), ("t", "m"), ("E", "Pa"), ("nu", "1"), ("q", "Pa"))
W_BAR = "w_bar = w*D/(q*a^4), a — сторона вдоль x (R3_NUMERICAL_SPEC.md 1.2; RectangleResult.dimensionless_field)"

HISTORY_COLUMNS = (
    # (name, unit or "", source); quantities of NUMERICAL_SPEC.md 11.5 from halfstep_report
    ("s", "", "half-step"), ("k", "", "cycle s // 2"),
    ("direction", "", "computed factor: Y (odd s) or X (even s)"), ("solve_axis", "", "axis of the 1D problem"),
    ("solve_n", "", "intervals of the 1D problem"), ("c", "", "normalization factor of the computed factor"),
    ("beta", "", "B/A"), ("gamma", "", "C/A"), ("chi", "", "F^2/A"),
    ("residual_norm", "", "|rho|_inf of the 1D system"), ("eta", "", "residual eta"),
    ("refined", "", "refinement step done"), ("imag_ratio", "", "|Im Y|/|Y|"),
    ("P", "N·m", "Pi_h(w(s))"), ("norm_a", "(N·m)^0.5", "|w(s)|_a"),
    ("eps_inf", "", "11.1"), ("eps_E", "", "11.1"), ("eps_L2", "", "11.1, diagnostic"), ("theta", "", "11.1"),
    ("satisfied", "", "eps_inf <= tol_inf and eps_E <= tol_E"),
    ("tail_estimate", "", "T_s (11.3, item 6): assumes contraction, not a proven error bound"),
    ("energy_status", "", "energy test: NOT_APPLICABLE, UNRESOLVED, PASS, FAIL"),
    ("delta_P", "N·m", "10.2"), ("Q", "N·m", "10.2"), ("gamma_s", "N·m", "10.2"), ("tau", "N·m", "10.4"),
    ("w_bar_centre", "", "w_bar at (a/2, b/2), or the message of a point that is not a node"),
)


class OutputExistsError(Exception):
    """Result files are present in the output folder and --overwrite was not given."""


class OutputWriteError(OSError):
    """Saving failed; records what had already been changed in the folder when it failed."""

    def __init__(self, action: str, name: str, cause: OSError, created, replaced, removed, pending, leftovers):
        super().__init__(f"{action} {name}: {cause}")
        self.action, self.name, self.cause = action, name, cause
        self.created = tuple(created)  # result files written where none existed
        self.replaced = tuple(replaced)  # result files of the earlier run replaced by new ones
        self.removed = tuple(removed)  # stale result files of the earlier run removed
        self.pending = tuple(pending)  # operations not done: (action, name)
        self.leftovers = tuple(leftovers)  # own temporary files that could not be removed

    @property
    def changed(self) -> bool:
        """True if any result file in the folder was changed before the failure."""
        return bool(self.created or self.replaced or self.removed)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m kkvim",
        description="KK-VIM: расчёт принятым решателем R3 (D-043) — прямоугольная пластина a × b, SSSS, q = const, N = 1.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser(
        "run", help="выполнить расчёт R3 и сохранить result.json, history.csv и поле",
        description="Расчёт R3: kkvim.rectangle_run.run_rectangle_physical и halfstep_report. Единицы СИ.",
    )
    for name, unit, text in (("a", "m", "сторона вдоль x"), ("b", "m", "сторона вдоль y"), ("t", "m", "толщина"),
                             ("E", "Pa", "модуль упругости"), ("nu", "", "коэффициент Пуассона"),
                             ("q", "Pa", "равномерная нагрузка")):
        run.add_argument(f"--{name}", type=float, required=True, metavar="VALUE",
                         help=f"{text}" + (f", {unit}" if unit else ""))
    run.add_argument("--nx", type=int, required=True, metavar="N", help="число интервалов сетки по x")
    run.add_argument("--ny", type=int, required=True, metavar="N", help="число интервалов сетки по y")
    run.add_argument("--out", type=Path, required=True, metavar="DIR", help="папка результатов (создаётся)")
    run.add_argument("--overwrite", action="store_true",
                     help="разрешить замену файлов результатов, уже имеющихся в папке")
    stopping = run.add_argument_group(
        "параметры остановки (NUMERICAL_SPEC.md 11.2)",
        "По умолчанию — значения kkvim, с которыми R3 принят (D-043); передаются решателю без изменений.")
    stopping.add_argument("--tol-inf", type=float, metavar="VALUE", help=f"tol_∞ (по умолчанию {TOL_INF:g})")
    stopping.add_argument("--tol-E", type=float, metavar="VALUE", help=f"tol_E (по умолчанию {TOL_E:g})")
    stopping.add_argument("--s-min", type=int, metavar="N", help=f"s_min (по умолчанию {S_MIN})")
    stopping.add_argument("--s-max", type=int, metavar="N", help=f"s_max (по умолчанию {S_MAX})")
    return parser


def check_output_folder(out: Path, overwrite: bool) -> None:
    """Refuse a folder that is a file, or one with result files unless overwriting is allowed."""
    if out.exists() and not out.is_dir():
        raise OutputExistsError(f"{out} существует и не является папкой")
    present = [name for name in OUTPUTS if (out / name).exists()]
    if present and not overwrite:
        raise OutputExistsError(f"в {out} уже есть файлы результатов: {', '.join(present)}. "
                                "Укажите другую папку или --overwrite")


def solve(args: argparse.Namespace):
    """(RectangleResult, RunReport) of the solver for the parsed arguments."""
    options = {key: value for key, value in (("tol_inf", args.tol_inf), ("tol_E", args.tol_E),
                                              ("s_min", args.s_min), ("s_max", args.s_max)) if value is not None}
    result = run_rectangle_physical(args.a, args.b, args.t, args.E, args.nu, args.q, args.nx, args.ny, **options)
    return result, halfstep_report(result)


def field_file(status: str, s: int) -> str | None:
    """field.csv only for CONVERGED; the field of another run is diagnostic; no file without a half-step."""
    if s == 0:
        return None
    return FIELD if status == CONVERGED else FIELD_DIAGNOSTIC


def exit_code(status: str) -> int:
    if status == CONVERGED:
        return EXIT_CONVERGED
    if status in (STAGNATED, MAX_HALF_STEPS):
        return EXIT_NOT_CONVERGED
    return EXIT_SOLVER_ERROR


def grid_conditions(n_x: int, n_y: int) -> dict:
    """Each necessary condition of a qualification grid (R3_NUMERICAL_SPEC.md 6.1) as a fact; no verdict."""
    return {
        "min_n_at_least_20": min(n_x, n_y) >= 20,
        "max_n_at_most_640": max(n_x, n_y) <= 640,
        "n_x_and_n_y_multiples_of_4": n_x % 4 == 0 and n_y % 4 == 0,
        "note": GRID_CONDITIONS_NOTE,
    }


def _number(value):
    """JSON value: finite floats as they are (json writes repr), non-finite as null."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _cell(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):  # numpy floats included: written as the same Python float
        return repr(float(value)) if math.isfinite(value) else ""
    return str(value)


def history_rows(report):
    for h in report.halfsteps:
        ch, en = h.changes, h.energy
        centre = dict(h.control_values).get(CENTRE)
        yield {
            "s": h.s, "k": h.k, "direction": h.direction, "solve_axis": h.solve_axis, "solve_n": h.solve_grid[1],
            "c": h.c, "beta": h.beta, "gamma": h.gamma, "chi": h.chi, "residual_norm": h.residual_norm,
            "eta": h.eta, "refined": h.refined, "imag_ratio": h.imag_ratio, "P": h.P, "norm_a": h.norm_a,
            "eps_inf": None if ch is None else ch.eps_inf, "eps_E": None if ch is None else ch.eps_E,
            "eps_L2": None if ch is None else ch.eps_L2, "theta": None if ch is None else ch.theta,
            "satisfied": None if ch is None else ch.satisfied,
            "tail_estimate": None if ch is None else ch.tail_estimate,
            "energy_status": None if en is None else en.status, "delta_P": None if en is None else en.delta_P,
            "Q": None if en is None else en.Q, "gamma_s": None if en is None else en.gamma,
            "tau": None if en is None else en.tau, "w_bar_centre": centre,
        }


def _write_temporary(folder: Path, text: str) -> Path:
    """A new file with a unique name in ``folder`` (never an existing one) holding ``text``."""
    handle, name = tempfile.mkstemp(dir=folder, prefix=".kkvim-", suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="") as f:
            f.write(text)
    except BaseException:
        Path(name).unlink(missing_ok=True)
        raise
    return Path(name)


def _csv_text(header, rows, comments=()) -> str:
    lines = [f"# {c}\n" for c in comments]
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)
    for row in rows:
        writer.writerow([_cell(v) for v in row])
    return "".join(lines) + buffer.getvalue()


def history_csv(report) -> str:
    header = [name for name, _, _ in HISTORY_COLUMNS]
    return _csv_text(header, ([row[name] for name in header] for row in history_rows(report)))


def field_csv(result, diagnostic: bool) -> str:
    """Nodes i, j, x, y with w = run.last_field() and w_bar = result.dimensionless_field() of the solver."""
    run = result.run
    last = run.history.records[-1]
    x, y = last.grid_x.nodes, last.grid_y.nodes
    w, w_bar = run.last_field(), result.dimensionless_field()
    comments = ()
    if diagnostic:
        comments = (f"DIAGNOSTIC: статус решателя {result.status}; поле последнего выполненного полушага s = {run.s}. "
                    "Это не рассчитанное решение.",)
    rows = ((i, j, x[i], y[j], float(w[i, j]), float(w_bar[i, j]))
            for i in range(w.shape[0]) for j in range(w.shape[1]))
    return _csv_text(("i", "j", "x [m]", "y [m]", "w [m]", "w_bar [1]"), rows, comments)


def result_document(args, argv, result, report, field_name) -> dict:
    run_ = result.run
    s = 0 if run_ is None else run_.s
    status = result.status
    centre = None
    if report.halfsteps:
        centre = dict(report.halfsteps[-1].control_values).get(CENTRE)
    field = None
    if field_name is not None:
        w = run_.last_field()
        field = {
            "file": field_name,
            "role": "converged" if field_name == FIELD else "diagnostic",
            "note": (STATUS_NOTES[CONVERGED] if field_name == FIELD else
                     f"Поле последнего выполненного полушага s = {s} при статусе {status}; не рассчитанное решение."),
            "half_step": s,
            "shape": list(w.shape),
            "axes": "axis 0 = x (i = 0..n_x), axis 1 = y (j = 0..n_y), boundary nodes included",
            "units": {"x": "m", "y": "m", "w": "m", "w_bar": "1"},
            "w_bar_definition": W_BAR,
            "w_bar_centre": _number(centre),
        }
    parameters = report.parameters
    return {
        "schema": SCHEMA,
        "program": {"command": "python -m kkvim " + " ".join(argv), "calls": list(CALLS), "versions": report.versions},
        "model": {"solver": "R3", "acceptance": "ACCEPTED (D-043)", "theory": "Kirchhoff-Love, linear",
                  "supports": "SSSS", "load": "uniform q = const", "N": 1},
        "input": {
            **{name: {"value": _number(getattr(args, name)), "unit": unit} for name, unit in INPUT_UNITS},
            "n_x": args.nx, "n_y": args.ny,
        },
        "stopping_parameters": None if parameters is None else {k: _number(v) for k, v in asdict(parameters).items()},
        "D": None if result.D is None else {"value": _number(result.D), "unit": "N·m"},
        "status": status,
        "status_note": STATUS_NOTES.get(status, ERROR_NOTE),
        "converged": status == CONVERGED,
        "error": report.error,
        "half_steps": s,
        "cycles_k": s // 2,
        "field": field,
        "history": {
            "file": HISTORY, "rows": len(report.halfsteps),
            "columns": [{"name": n, "unit": u, "meaning": m} for n, u, m in HISTORY_COLUMNS],
            "tail_estimate_note": None if run_ is None else run_.tail_estimate_note,
        },
        "applicability": {
            "limits": list(APPLICABILITY),
            "necessary_grid_conditions_6_1": grid_conditions(args.nx, args.ny),
        },
    }


def result_texts(args, argv, result, report) -> dict:
    """{file name: text} of one run (no file is touched)."""
    s = 0 if result.run is None else result.run.s
    field_name = field_file(result.status, s)
    texts = {}
    if field_name is not None:
        texts[field_name] = field_csv(result, diagnostic=field_name == FIELD_DIAGNOSTIC)
    texts[HISTORY] = history_csv(report)
    document = result_document(args, argv, result, report, field_name)
    texts[RESULT] = json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    return texts


def write_results(out: Path, texts: dict, overwrite: bool) -> dict:
    """Write the files of one run into the existing folder ``out``; on failure raise ``OutputWriteError``.

    1. Every file is written completely to a new temporary file with a unique name. A failure here changes no
       result file: the folder keeps its earlier results.
    2. The result files are moved into place one by one (``os.replace``).
    3. With --overwrite, result files of an earlier run that this run does not produce are removed (a converged
       field.csv must not stay next to a later failed run).

    The set is not replaced atomically. A failure in step 2 or 3 (for example a file locked by another program)
    leaves a partial update: some files are of the new run, others of the earlier one. ``OutputWriteError`` then
    lists the files created, replaced and removed, the file that failed and the operations not done; the folder
    must not be used as a completed run, and the run has to be repeated after the cause is removed. Only own
    temporary files are deleted; files of other names are never touched.
    """
    temporary, created, replaced, removed = {}, [], [], []
    stale = [name for name in OUTPUTS if overwrite and name not in texts]
    step = ("запись", "")
    try:
        for name, text in texts.items():
            step = ("запись временного файла для", name)
            temporary[name] = _write_temporary(out, text)
        for name, path in list(temporary.items()):
            step = ("замена", name)
            existed = (out / name).exists()
            os.replace(path, out / name)
            del temporary[name]
            (replaced if existed else created).append(name)
        for name in stale:  # only after every new file is in place
            step = ("удаление устаревшего файла", name)
            if (out / name).exists():
                os.remove(out / name)
                removed.append(name)
    except OSError as exc:
        action, name = step
        done = set(created) | set(replaced)
        pending = [("замена", n) for n in texts if n not in done and n != name]
        pending += [("удаление устаревшего файла", n) for n in stale
                    if n not in removed and n != name and (out / n).exists()]
        raise OutputWriteError(action, name, exc, created, replaced, removed, pending,
                               _remove_temporary(temporary.values())) from exc
    finally:
        _remove_temporary(temporary.values())
    return {name: out / name for name in texts}


def _remove_temporary(paths) -> list:
    """Remove own temporary files (never a result or another file); names of those that remain."""
    left = []
    for path in list(paths):
        try:
            path.unlink(missing_ok=True)
        except OSError:
            left.append(path.name)
    return left


def output_error_lines(out: Path, exc: OutputWriteError, status: str) -> list:
    """Messages of a failed save: what failed, what had changed, and whether the folder may be used."""
    lines = [f"Сохранение результатов завершилось ошибкой: {exc.action} {exc.name} в {out}: {exc.cause}"]
    if exc.changed:
        if exc.replaced:
            lines.append("Уже заменены новыми: " + ", ".join(exc.replaced))
        if exc.created:
            lines.append("Уже записаны (новые): " + ", ".join(exc.created))
        if exc.removed:
            lines.append("Уже удалены устаревшие: " + ", ".join(exc.removed))
        not_done = [f"{name} ({action})" for action, name in exc.pending]
        lines.append(f"Не выполнено: {exc.name} ({exc.action})" + "".join(f", {item}" for item in not_done))
        lines.append("ВНИМАНИЕ: каталог может содержать результаты разных расчётов. Использовать его как завершённый "
                     "расчёт нельзя. Устраните причину ошибки (например, файл открыт другой программой) и повторите "
                     "расчёт с --overwrite.")
    else:
        lines.append(f"Прежние файлы результатов в {out} не изменены.")
    if exc.leftovers:
        lines.append("Не удалось удалить собственные временные файлы: " + ", ".join(exc.leftovers))
    lines.append(f"Расчёт завершился со статусом решателя {status}; полный набор результатов этого расчёта не сохранён.")
    return lines


def _utf8_streams() -> None:
    """Messages are Russian: a pipe or file with a legacy code page (cp1252 on Windows) gets UTF-8 instead of '?'.

    A Windows console writes Unicode itself (PEP 528) and is left as it is.
    """
    for stream in (sys.stdout, sys.stderr):
        encoding = (getattr(stream, "encoding", None) or "").lower().replace("-", "")
        if encoding != "utf8" and hasattr(stream, "reconfigure") and not stream.isatty():
            stream.reconfigure(encoding="utf-8", errors="replace")


def _say(stream, text: str) -> None:
    try:
        stream.write(text + "\n")
    except UnicodeEncodeError:  # a console that cannot show the text: keep it readable, never fail on it
        encoding = stream.encoding or "ascii"
        stream.write(text.encode(encoding, "replace").decode(encoding) + "\n")
    stream.flush()


def _internal_error(stage: str) -> int:
    _say(sys.stderr, f"Внутренняя ошибка программы ({stage}). Это не статус решателя; результаты не сохранены "
                     "или сохранены не полностью.")
    _say(sys.stderr, traceback.format_exc().rstrip())
    return EXIT_INTERNAL_ERROR


def main(argv=None) -> int:
    _utf8_streams()
    argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv)
    try:
        check_output_folder(args.out, args.overwrite)
    except OutputExistsError as exc:
        _say(sys.stderr, f"Расчёт не выполнен: {exc}")
        return EXIT_USAGE
    try:
        args.out.mkdir(parents=True, exist_ok=True)  # before the computation: an unusable path costs no run
    except OSError as exc:
        _say(sys.stderr, f"Расчёт не выполнен: не удалось создать папку результатов {args.out}: {exc}")
        return EXIT_OUTPUT_ERROR
    try:
        result, report = solve(args)
        texts = result_texts(args, argv, result, report)
    except Exception:  # noqa: BLE001 - reported as an internal error, never as a status of the solver
        return _internal_error("расчёт или подготовка результатов")
    try:
        paths = write_results(args.out, texts, args.overwrite)
    except OutputWriteError as exc:
        for line in output_error_lines(args.out, exc, result.status):
            _say(sys.stderr, line)
        return EXIT_OUTPUT_ERROR
    except Exception:  # noqa: BLE001
        return _internal_error("запись результатов")
    s = 0 if result.run is None else result.run.s
    _say(sys.stdout, f"Статус решателя R3: {result.status} (полушагов s = {s}, циклов k = {s // 2})")
    _say(sys.stdout, STATUS_NOTES.get(result.status, ERROR_NOTE))
    if result.error is not None:
        _say(sys.stdout, f"Сообщение решателя: {result.error}")
    if result.D is not None:
        _say(sys.stdout, f"D = {result.D!r} N·m")
    _say(sys.stdout, f"Результаты: {args.out.resolve()}")
    for name, path in paths.items():
        role = " (ДИАГНОСТИКА, не решение)" if name == FIELD_DIAGNOSTIC else ""
        _say(sys.stdout, f"  {name}{role}")
    conditions = grid_conditions(args.nx, args.ny)
    if not all(value for key, value in conditions.items() if key != "note"):
        _say(sys.stdout, f"Сетка {args.nx}×{args.ny} не удовлетворяет необходимым условиям квалификационной сетки "
                         "R3_NUMERICAL_SPEC.md 6.1 (min ≥ 20, max ≤ 640, кратность 4): расчёт диагностический.")
    return exit_code(result.status)
