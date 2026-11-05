"""``python -m kkvim run-cccc-n1``: CCCC, N = 1, exactly --half-steps half-steps; the limited research mode of D-061.

    python -m kkvim run-cccc-n1 --a 1 --b 1 --t 0.01 --E 2.1e11 --nu 0.3 --q 1000 --nx 32 --ny 32 --half-steps 8 --out scratch/cccc

The command calls ``kkvim.cccc_n1.run_cccc_n1_physical`` and writes what it returns; it computes no quantity of the method. Every result is
RESEARCH_ONLY / PROVISIONAL / NOT_QUALIFIED: "completed" means only that the given half-steps were done and no guard fired.

Files: result.json; history.csv; factors.csv (all factors of every half-step, for the QA checks of Policy 10.2); field_research.csv after the
requested half-steps, field_diagnostic.csv (the field of the last completed half-step) after a stop; no field without a half-step.

Saving is ``kkvim.cli.write_results`` (Policy 11), so a failure or a killed process while the files are moved in place may leave files of two
runs. A refused input, an ERROR_* run or a run without a half-step does not touch earlier results: the status and the reason go to stderr. If
the folder has no earlier results, such a run writes its result.json. factors.csv counts as a result file of this command.

Exit codes (Policy 6): 0 - the requested half-steps were done (not convergence); 2 - outside the domain or a refused command line, nothing
computed; 3 - ERROR_*; 4, 5 - as in ``kkvim.cli``; 6 - a guard or a protective stop; 130 - interrupted by the user.
"""

import json
import sys
from pathlib import Path

import numpy as np

import kkvim.cccc_n1 as cn
import kkvim.cli as cli
from kkvim.cli_ssss import EXIT_INTERRUPTED, software_provenance

SCHEMA = "kkvim.cli.cccc_n1/1"
FACTORS = "factors.csv"
EXIT_PROTECTIVE_STOP = 6
EXIT_CODES = {
    cn.COMPLETED_FIXED_HALF_STEPS: cli.EXIT_CONVERGED, cn.NOT_EXECUTED_OUTSIDE_DOMAIN: cli.EXIT_USAGE, cn.ERROR_INPUT: cli.EXIT_SOLVER_ERROR,
    cn.ERROR_NONFINITE: cli.EXIT_SOLVER_ERROR, cn.UNRESOLVED_C3PLUS_PREMISE: EXIT_PROTECTIVE_STOP,
    cn.PROTECTIVE_STOP_FACTORIZATION: EXIT_PROTECTIVE_STOP, cn.STOPPED_ETA_GUARD: EXIT_PROTECTIVE_STOP, cn.STOP_ZERO_FIELD: EXIT_PROTECTIVE_STOP,
}
KEEPS_EARLIER = (cn.ERROR_INPUT, cn.ERROR_NONFINITE, cn.NOT_EXECUTED_OUTSIDE_DOMAIN)  # these, and any run without a half-step, replace no earlier file

STATUS_NOTES = {
    cn.COMPLETED_FIXED_HALF_STEPS: "Выполнено заданное число полушагов, защиты не сработали. Это не сходимость и не квалификация.",
    cn.NOT_EXECUTED_OUTSIDE_DOMAIN: "Расчёт не выполнялся: ввод вне области D-061.2. Значений решения нет.",
    cn.ERROR_INPUT: "Расчёт не выполнялся: недопустимый ввод. Значений решения нет.",
    cn.ERROR_NONFINITE: "Неконечное значение в расчёте. Решения нет; поле последнего завершённого полушага, если есть, — только диагностика.",
    cn.UNRESOLVED_C3PLUS_PREMISE: "Предпосылка сборки C-3⁺ не установлена. Поле последнего завершённого полушага, если есть, — только диагностика.",
    cn.PROTECTIVE_STOP_FACTORIZATION: "Защитная остановка решения полушага; при g ≠ 0 это противоречит теории, то есть признак дефекта, а не вывод о "
                                      "методе. Поле последнего завершённого полушага, если есть, — только диагностика.",
    cn.STOPPED_ETA_GUARD: "Невязка выше защитного порога η_max^C = 100u (D-061.2, вариант А). Без уточнения и смены решателя. Поле последнего "
                          "завершённого полушага, если есть, — только диагностика.",
    cn.STOP_ZERO_FIELD: "Решение полушага тождественно равно нулю. Поле последнего завершённого полушага, если есть, — только диагностика.",
}
LIMITS = (
    "Линейная изотропная однородная пластина Кирхгофа–Лява, CCCC, q = const ≠ 0, N = 1, 0,2 ≤ b/a ≤ 5, 7 ≤ n_x, n_y ≤ 1280, s = 1…8 (D-061.2).",
    "Режим не утверждает сходимости при s → ∞, близости к решению непрерывной или дискретной задачи, квалификации B-C на рабочих сетках и "
    "приёмки R5, Q4 или Q5 (D-061.3; ПОЛИТИКА, разд. 8).",
    "Фиксированный N = 1 не сходится к решению дискретной задачи, если её ранг больше 1: разность — мера ранга (NS 9, О-8). SCRATCH: α при s = 8 "
    "на 1 × 1 — 1,27413·10⁻³ (32 × 32), 1,26824·10⁻³ (40 × 64); опубликованное L-120 — 1,265319·10⁻³. Разница включает сетку и приближение N = 1 "
    "и допуском не является.",
    "При больших s изменения энергии уходят ниже сертифицируемого уровня. SCRATCH: на пяти малых сетках внешняя классификация даёт PASS при "
    "s = 1…5 и UNRESOLVED_RESOLVABILITY при s = 6…8. Это не ошибка, но содержательность больших s энергетически не подтверждена.",
    "Предпосылка П-4 (классическое скалярное произведение в BLAS) принята как допущение и не проверяется.",
)
EVIDENCE_SCOPE = ("Данные D0 получены при a = 1, D = 1, q = 1. Другие значения — та же дискретная задача: w̄ = wD/(qa⁴) зависит только от x/a, y/b, b/a "
                  "(ПОЛИТИКА, разд. 1). База разд. 7.1 — парные полушаги на входах Q3, а не траектории на рабочих сетках.")
W_BAR = "w_bar = w D / (q a^4)"
ALPHA = "alpha_centre = w(a/2, b/2) D / (q l^4), l = min(a, b); only with even n_x and n_y"
NOT_A_NODE = "центр не узел"

HISTORY_COLUMNS = (
    ("s", "", "номер полушага"), ("k", "", "⌊s/2⌋"), ("direction", "", "решаемый множитель: Y при нечётном s, X при чётном"),
    ("status", "", "HALF_STEP_DONE или, в последней строке, статус расчёта"), ("n_s", "", "интервалов по решаемому направлению"),
    ("h_s", "m", "шаг по решаемому направлению"), ("A", "*", "коэффициент заданного множителя"), ("B", "*", "коэффициент заданного множителя"),
    ("C", "*", "коэффициент заданного множителя, сборка C-3⁺"), ("F", "*", "правая часть"), ("B_over_A", "*", "инвариант нормировки"),
    ("C_over_A", "*", "инвариант нормировки"), ("F2_over_A", "*", "инвариант нормировки"), ("C_L", "*", "часть C"), ("E", "*", "часть C"),
    ("premises_status", "", "П-1…П-5; П-4 — допущение"), ("cond_S", "", "только запись"), ("min_DM", "*", "min DM"),
    ("eta", "", "невязка исходной системы"), ("c", "*", "множитель нормировки И-4"), ("sigma", "", "знак нормировки"),
    ("P_hat", "N·m", "энергия при w = X⊗Y; наблюдение"), ("delta_P_hat", "N·m", "P̂ предыдущего минус P̂; наблюдение, отрицательное значение не ошибка"),
    ("field_change_inf", "", "max|w(s) − w(s−1)| / max|w(s)|; при s = 1 w(0) = 0"), ("alpha_centre", "", ALPHA), ("alpha_note", "", NOT_A_NODE),
    ("reason", "", "причина остановки"),
)


class OptionError(Exception):
    """A command line this command does not offer; nothing is computed or written."""


def add_parser(commands) -> None:
    run = commands.add_parser(
        "run-cccc-n1", help="расчёт CCCC, N = 1, с заданным числом полушагов; только исследование (D-061)",
        description="CCCC, q = const, N = 1, режим FIXED_HALF_STEPS_RESEARCH (D-061). Результат всегда RESEARCH_ONLY / PROVISIONAL / NOT_QUALIFIED: "
                    "ни сходимость, ни квалификация не утверждаются. Единицы СИ.",
    )
    for name, unit, text in (("a", "m", "сторона вдоль x"), ("b", "m", "сторона вдоль y"), ("t", "m", "толщина"), ("E", "Pa", "модуль упругости"),
                             ("nu", "", "коэффициент Пуассона"), ("q", "Pa", "равномерная нагрузка, не ноль")):
        run.add_argument(f"--{name}", type=float, required=True, metavar="VALUE", help=f"{text}" + (f", {unit}" if unit else ""))
    run.add_argument("--nx", type=int, required=True, metavar="N", help=f"интервалов сетки по x, {cn.N_MIN}…{cn.N_MAX}")
    run.add_argument("--ny", type=int, required=True, metavar="N", help=f"интервалов сетки по y, {cn.N_MIN}…{cn.N_MAX}")
    run.add_argument("--half-steps", type=int, required=True, metavar="S", help=f"ровно столько полушагов, {cn.S_MIN}…{cn.S_MAX}; критерия остановки нет")
    run.add_argument("--out", type=Path, required=True, metavar="DIR", help="папка результатов (создаётся)")
    run.add_argument("--overwrite", action="store_true", help="разрешить замену файлов результатов, уже имеющихся в папке; отказ и ошибка их не меняют")


# --- the files ---------------------------------------------------------------------------------------------------------------------

def _history_csv(result: cn.CCCCN1Result) -> str:
    header = [name for name, _, _ in HISTORY_COLUMNS]
    centre_is_node = result.n_x % 2 == 0 and result.n_y % 2 == 0
    rows = []
    for step in result.steps:
        values = {**{name: getattr(step, name, None) for name in header}}
        if step.A is not None and step.A != 0.0:
            values.update(B_over_A=step.B / step.A, C_over_A=step.C / step.A, F2_over_A=step.F * step.F / step.A)
        values["alpha_note"] = None if centre_is_node or not step.done else NOT_A_NODE
        rows.append([values[name] for name in header])
    return cli._csv_text(header, rows, (f"{cn.APPLICATION_STATUS}; режим {cn.MODE}; единицы столбцов — в result.json.",))


def _factors_csv(result: cn.CCCCN1Result) -> str:
    """Every factor of every half-step with repr: enough to restore the half-steps bit for bit."""
    rows = []
    for step in result.steps:
        for role, values in (("given", step.given), ("raw", step.raw), ("next_given", step.next_given), ("next_conjugate", step.next_conjugate)):
            if values is not None:
                rows.extend((step.s, step.direction, role, index + 1, float(value)) for index, value in enumerate(values))
    return cli._csv_text(("s", "direction", "role", "index", "value"), rows, (f"{cn.APPLICATION_STATUS}.",))


def _field_csv(result: cn.CCCCN1Result, diagnostic: bool) -> str:
    w, w_bar = result.field(), result.dimensionless_field()
    h_x, h_y = result.a / result.n_x, result.b / result.n_y
    comment = f"{cn.APPLICATION_STATUS}; поле полушага s = {result.half_steps_completed}."
    if diagnostic:
        comment = f"DIAGNOSTIC: статус {result.status}; поле последнего завершённого полушага s = {result.half_steps_completed}. Не результат расчёта. {comment}"
    rows = ((i, j, i * h_x, j * h_y, float(w[i, j]), float(w_bar[i, j])) for i in range(w.shape[0]) for j in range(w.shape[1]))
    return cli._csv_text(("i", "j", "x [m]", "y [m]", "w [m]", "w_bar [1]"), rows, (comment,))


def _field_record(result: cn.CCCCN1Result, name: str | None) -> dict | None:
    if name is None:
        return None
    w = result.field()
    i, j = np.unravel_index(int(np.argmax(np.abs(w))), w.shape)
    h_x, h_y = result.a / result.n_x, result.b / result.n_y
    last = result.completed[-1]
    return {
        "file": name, "role": "research" if name == cli.FIELD_RESEARCH else "diagnostic", "half_step": result.half_steps_completed,
        "shape": list(w.shape), "axes": "axis 0 = x (i = 0..n_x), axis 1 = y (j = 0..n_y), boundary nodes included",
        "units": {"x": "m", "y": "m", "w": "m", "w_bar": "1"}, "w_bar_definition": W_BAR,
        "w_max": {"value": float(abs(w[i, j])), "i": int(i), "j": int(j), "x": float(i * h_x), "y": float(j * h_y)},
        "alpha_centre": cli._number(last.alpha_centre), "alpha_definition": ALPHA,
        "alpha_note": None if last.alpha_centre is not None else NOT_A_NODE,
    }


def _premises_record(result: cn.CCCCN1Result) -> dict:
    statuses = sorted({step.premises_status for step in result.steps if step.premises_status})
    reasons = [f"s = {step.s}: {reason}" for step in result.steps for reason in step.premises_reasons]
    return {"status_by_half_steps": statuses, "reasons": reasons, "P-4": cn.P4_ASSUMED}


def result_texts(args, argv, result: cn.CCCCN1Result) -> dict:
    """{file name: text} of one run, result.json last (no file is touched)."""
    texts = {}
    s = result.half_steps_completed
    field_name = None
    if s > 0:
        field_name = cli.FIELD_RESEARCH if result.status == cn.COMPLETED_FIXED_HALF_STEPS else cli.FIELD_DIAGNOSTIC
        texts[field_name] = _field_csv(result, field_name == cli.FIELD_DIAGNOSTIC)
    if result.steps:
        texts[FACTORS] = _factors_csv(result)
        texts[cli.HISTORY] = _history_csv(result)
    document = {
        "schema": SCHEMA,
        "program": {"command": "python -m kkvim " + " ".join(argv), "argv": list(argv), "calls": ["kkvim.cccc_n1.run_cccc_n1_physical"]},
        "provenance": {**software_provenance(), "module": "kkvim.cli_cccc"},
        "model": {"boundary": "CCCC", "N": 1, "theory": "Kirchhoff-Love, linear; homogeneous isotropic plate; uniform q = const",
                  "start": "X0 = sin^2(pi x / a); the first half-step solves Y", "assembly": "C-3+ (D-054.3)", "half_step_solver": "B-C (D-049.2)",
                  "discretization": "проект, уровень Б (D-049.1); авторам КК-МВИ не приписывается", "decision": "D-061"},
        "input": {**{name: {"value": cli._number(getattr(args, name)), "unit": unit} for name, unit in cli.INPUT_UNITS},
                  "n_x": args.nx, "n_y": args.ny, "half_steps": args.half_steps},
        "D": None if result.D is None else {"value": cli._number(result.D), "unit": "N·m"},
        "mode": result.mode,
        "application_status": result.application_status,
        "status": result.status,
        "status_note": STATUS_NOTES[result.status],
        "reason": result.reason,
        "half_steps_requested": args.half_steps,
        "half_steps_completed": s,
        "cycles_k": s // 2,
        "numerical_policy": {
            "eta_max_C": {"value": cn.ETA_MAX_C, "expression": "100u, u = 2^-53", "role": "защитный порог только этого режима, не квалифицированное значение (D-061.2)"},
            "cond_S": "только запись",
            "qa_criteria": "TS-1 (c_f = 10) и |η − η_QA| ≤ 16u — критерии приёмки первой версии только на случаях ПОЛИТИКИ, разд. 9; решатель их не применяет",
        },
        "premises_c3plus": _premises_record(result),
        "evidence_scope": EVIDENCE_SCOPE,
        "limits": list(LIMITS),
        "field": _field_record(result, field_name),
        "history": None if not result.steps else {
            "file": cli.HISTORY, "rows": len(result.steps),
            "columns": [{"name": name, "unit": unit, "meaning": meaning} for name, unit, meaning in HISTORY_COLUMNS],
            "units_note": "* — единицы зависят от нормировки множителей (И-4); значения пишутся без пересчёта.",
        },
        "factors": None if not result.steps else {"file": FACTORS, "note": "все множители каждого полушага; значения через repr"},
    }
    texts[cli.RESULT] = json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    return texts


# --- the command -------------------------------------------------------------------------------------------------------------------

def _kept(out: Path) -> list:
    return [name for name in (*cli.OUTPUTS, FACTORS) if (out / name).exists()]


def _check_folder(out: Path, overwrite: bool) -> None:
    cli.check_output_folder(out, overwrite)
    if (out / FACTORS).exists() and not overwrite:
        raise cli.OutputExistsError(f"в {out} уже есть файл результатов {FACTORS}. Укажите другую папку или --overwrite")


def _say_kept(out: Path) -> None:
    kept = _kept(out)
    if kept:
        cli._say(sys.stderr, f"Прежние файлы результатов в {out} не изменены: {', '.join(kept)}.")


def main(args, argv) -> int:
    try:
        _check_folder(args.out, args.overwrite)
    except cli.OutputExistsError as exc:
        cli._say(sys.stderr, f"Расчёт не выполнен: {exc}")
        return cli.EXIT_USAGE
    try:
        args.out.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        cli._say(sys.stderr, f"Расчёт не выполнен: не удалось создать папку результатов {args.out}: {exc}")
        return cli.EXIT_OUTPUT_ERROR
    try:
        result = cn.run_cccc_n1_physical(args.a, args.b, args.t, args.E, args.nu, args.q, args.nx, args.ny, args.half_steps)
        texts = result_texts(args, argv, result)
    except KeyboardInterrupt:
        cli._say(sys.stderr, "Расчёт прерван пользователем; результаты этого расчёта не сохранены.")
        _say_kept(args.out)
        return EXIT_INTERRUPTED
    except Exception:  # noqa: BLE001 - reported as an internal error, never as a status of the solver
        code = cli._internal_error("расчёт или подготовка результатов")
        _say_kept(args.out)
        return code
    code = EXIT_CODES[result.status]
    if (result.status in KEEPS_EARLIER or result.half_steps_completed == 0) and _kept(args.out):
        cli._say(sys.stderr, f"Результат не заменён: статус {result.status}. {STATUS_NOTES[result.status]}")
        if result.reason:
            cli._say(sys.stderr, f"Причина: {result.reason}")
        _say_kept(args.out)
        return code
    try:
        _check_folder(args.out, args.overwrite)  # another run may have written here during this one
        paths = cli.write_results(args.out, texts, args.overwrite)
    except cli.OutputExistsError as exc:
        cli._say(sys.stderr, f"Результаты не сохранены: {exc}")
        return cli.EXIT_USAGE
    except cli.OutputWriteError as exc:
        for line in cli.output_error_lines(args.out, exc, result.status):
            cli._say(sys.stderr, line)
        return cli.EXIT_OUTPUT_ERROR
    except Exception:  # noqa: BLE001
        return cli._internal_error("запись результатов")
    _summary(args, result, texts, paths)
    return code


def _summary(args, result: cn.CCCCN1Result, texts: dict, paths: dict) -> None:
    document = json.loads(texts[cli.RESULT])
    stream = sys.stderr if result.status != cn.COMPLETED_FIXED_HALF_STEPS else sys.stdout
    cli._say(stream, f"CCCC, N = 1, {result.mode}: {result.application_status}")
    cli._say(stream, f"Статус: {result.status}; полушагов s = {result.half_steps_completed} из {args.half_steps}")
    cli._say(stream, document["status_note"])
    if result.reason:
        cli._say(stream, f"Причина: {result.reason}")
    field = document["field"]
    if field:
        w_max = field["w_max"]
        cli._say(stream, f"max|w| = {w_max['value']!r} м в узле ({w_max['i']}, {w_max['j']}), x = {w_max['x']!r}, y = {w_max['y']!r}; поле: {field['role']}")
        cli._say(stream, f"alpha_centre = {field['alpha_centre']!r}" if field["alpha_centre"] is not None else f"alpha_centre: {NOT_A_NODE}")
    diagnostic = document["provenance"]["git_diagnostic"]
    if diagnostic:
        cli._say(stream, f"Версия: {diagnostic}")
    cli._say(stream, f"Результаты: {args.out.resolve()}")
    for name in paths:
        cli._say(stream, f"  {name}")
