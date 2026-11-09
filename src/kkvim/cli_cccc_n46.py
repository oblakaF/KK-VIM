"""``python -m kkvim run-cccc-n46``: CCCC, N = 4, 5 or 6, the first half-step s = 1 only (D-067); a limited research mode.

    python -m kkvim run-cccc-n46 --N 5 --a 1 --b 1 --t 0.01 --E 2.1e11 --nu 0.3 --q 1000 --nx 32 --ny 32 --half-steps 1 --out scratch/cccc46

The command calls ``kkvim.cccc_n46.run_cccc_n46_physical`` and writes what it returns; it computes no quantity of the method. Every result is
RESEARCH_ONLY / PROVISIONAL / NOT_QUALIFIED. --half-steps 1: s = 1. --half-steps 2...8: after s = 1 a stop before any QR of the solved set
(STOPPED_RANK_GUARD_UNCERTIFIED, D-057.1 extended by D-067), the field of s = 1 diagnostic. No X half-step is executed.

Files: result.json; history.csv (one row per half-step reached); factors.csv (T, R, G, A, B, C, C_L, E, F, V with repr);
field_research.csv when the half-step asked for is done, field_diagnostic.csv after a stop; no field without the half-step s = 1.
eta, cond(S), the singular values, the exact rank of the stored factors and the representation diagnostics are OBSERVATION only: the
thresholds of N = 2, 3 (eta_max_C = 100u, |eta - eta_QA| <= 16u, c_f = 10) are not transferred and no eta guard exists (D-067).

Saving is ``kkvim.cli_ssss.replace_results`` with the check of an unfinished replacement and exit code 130 on Ctrl+C (D-062.1), with the
guarantees and limits of that mechanism; two runs saving into one folder at the same moment are not guarded against each other. A refused
input, an ERROR_* run or a run without the half-step s = 1 does not touch earlier results: the status and the reason go to stderr. If the folder
has no earlier results, such a run writes its result.json.

Exit codes: 0 - the half-step s = 1 asked for is done (not convergence); 2 - outside the domain, nothing computed; 3 - ERROR_*; 4, 5 - as in
``kkvim.cli``; 6 - a guard or a protective stop, the rank guard of s = 2 included; 130 - interrupted by the user.
"""

import json
import sys
from pathlib import Path

import numpy as np

import kkvim.cccc_n46 as cn
from kkvim import cli
from kkvim.cli_cccc_n23 import (
    FACTOR_ROLES,
    MATRIX_UNITS,
    NOT_A_NODE,
    _factors_csv,
    _field_csv,
    _field_record,
    _joined,
    _norm,
    _premises_record,
)
from kkvim.cli_ssss import (
    _interrupted,
    _kept,
    _say_kept,
    check_no_interrupted_save,
    replace_results,
    software_provenance,
)

SCHEMA = "kkvim.cli.cccc_n46/1"
EXIT_PROTECTIVE_STOP = 6
EXIT_CODES = {
    cn.COMPLETED_FIXED_HALF_STEPS: cli.EXIT_CONVERGED, cn.STOPPED_RANK_GUARD_UNCERTIFIED: EXIT_PROTECTIVE_STOP, cn.STOP_BEFORE_QR_S1: EXIT_PROTECTIVE_STOP,
    cn.NOT_EXECUTED_OUTSIDE_DOMAIN: cli.EXIT_USAGE, cn.ERROR_INPUT: cli.EXIT_SOLVER_ERROR, cn.ERROR_NONFINITE: cli.EXIT_SOLVER_ERROR,
    cn.UNRESOLVED_C3PLUS_PREMISE: EXIT_PROTECTIVE_STOP, cn.PROTECTIVE_STOP_FACTORIZATION: EXIT_PROTECTIVE_STOP, cn.STOP_ZERO_FIELD: EXIT_PROTECTIVE_STOP,
}
KEEPS_EARLIER = (cn.ERROR_INPUT, cn.ERROR_NONFINITE, cn.NOT_EXECUTED_OUTSIDE_DOMAIN)  # these, and any run without s = 1, replace no earlier file

STATUS_NOTES = {
    cn.COMPLETED_FIXED_HALF_STEPS: "Выполнен запрошенный фиксированный исследовательский полушаг s = 1, защиты не сработали. Это не сходимость, "
                                   "не точность поля, не полный ранг и не квалификация.",
    cn.STOPPED_RANK_GUARD_UNCERTIFIED: "Защитная остановка до QR решённого набора: при N = 4–6 исполняется только s = 1 (D-057.1, D-067). Поле s = 1 — "
                                       "только диагностика. Остановка не доказывает потери ранга.",
    cn.STOP_BEFORE_QR_S1: "Доказанная зависимость набора (точный ранг меньше N) до QR (D-049.5, п. 2). Поле s = 1, если есть, — только диагностика.",
    cn.NOT_EXECUTED_OUTSIDE_DOMAIN: "Расчёт не выполнялся: ввод вне области D-067. Значений решения нет.",
    cn.ERROR_INPUT: "Расчёт не выполнялся: недопустимый ввод. Значений решения нет.",
    cn.ERROR_NONFINITE: "Неконечное значение в расчёте полушага. Решения этого полушага нет.",
    cn.UNRESOLVED_C3PLUS_PREMISE: "Предпосылка сборки C-3⁺ не установлена. Решения этого полушага нет.",
    cn.PROTECTIVE_STOP_FACTORIZATION: "Защитная остановка QR или решения полушага; при независимом наборе это противоречит теории, то есть признак "
                                      "дефекта, а не вывод о методе. Решения этого полушага нет.",
    cn.STOP_ZERO_FIELD: "Решение полушага тождественно равно нулю. Решения этого полушага нет.",
}
LIMITS = (
    ("Линейная изотропная однородная пластина Кирхгофа–Лява, CCCC, q = const ≠ 0, N = 4, 5 или 6 фиксировано, 0,2 ≤ b/a ≤ 5, "
     "2N + 1 ≤ n_x, n_y ≤ 1280; исполняется только полушаг s = 1 (D-067); s ≥ 2 — остановка до QR решённого набора."),
    "Граница сетки 2N + 1 — инженерный предел применимости, а не сертификат точности; нижняя граница по y не устанавливает полного ранга поля.",
    ("Полный ранг точной дискретной последовательности при N ≥ 4 — UNRESOLVED (MATH_SPEC C.1: N ≥ 4 не рассматривается). Точный ранг сохранённых "
     "binary64-множителей — не сертификат для неокруглённого дискретного решения."),
    ("η, TS-1, cond(S), σ_i, ρ̂_N, κ_rep и точный ранг V — только наблюдения. Пороги N = 2, 3 (η_max^C = 100u, |η − η_QA| ≤ 16u, c_f = 10) не "
     "переносятся (D-067); защиты по η нет."),
    "Один полушаг не приближает решение дискретной задачи ω_h^C и непрерывной ω₀. Режим не утверждает сходимости, квалификации и приёмки R5, Q4, Q5.",
    "Поле s = 1 относится к span(G), стартовому набору 𝒯_N после округления QR.",
    "Предпосылка П-4 (классическое скалярное произведение в BLAS) принята как допущение и не проверяется.",
    "B-C не квалифицирован на рабочих сетках. Независимая QA пути N = 4–6 ещё не выполнена; тесты Builder её не заменяют.",
)
W_BAR = "w_bar = w D / (q a^4)"
ALPHA = "alpha_centre = w(a/2, b/2) D / (q l^4), l = min(a, b); only with even n_x and n_y"
SIGMA_COLUMNS = tuple(f"sigma_{i}" for i in range(1, max(cn.N_VALUES) + 1))

HISTORY_COLUMNS = (
    ("s", "", "номер полушага"), ("k", "", "⌊s/2⌋"), ("direction", "", "решаемые множители: Y при s = 1, X при s = 2 (только защита)"),
    ("status", "", "HALF_STEP_DONE или, в последней строке, статус расчёта"), ("n_s", "", "интервалов по решаемому направлению"),
    ("h_s", "m", "шаг по решаемому направлению"), ("N", "", "число членов, фиксировано"), ("orientation", "", cn.ORIENTATION),
    ("conditions", "", "s = 1: N ≤ ⌊n_x/2⌋, 2N < n_x, N ≤ ⌊n_y/2⌋; s = 2: конфигурационный S-1 (РН-3), только запись"),
    ("exact_rank_T", "", "точный ранг стартового набора"), ("exact_rank_G", "", "точный ранг G после QR"),
    ("qr_executed", "", "s = 1: QR стартового набора; s = 2: QR не выполняется"), ("R_diag_min_abs", "*", "min |diag R|"),
    ("cond_A_before_QR", "", "cond(R)^2; только запись"), ("cond_A_after_QR", "", "по собственным числам A(G); только запись"),
    ("A_diag", "*", "диагональ A через ';'"), ("B_diag", "*", "диагональ B"), ("C_diag", "*", "диагональ C (C-3⁺)"), ("F", "*", "правая часть"),
    ("A_norm_inf", "*", "‖A‖∞"), ("B_norm_inf", "*", "‖B‖∞"), ("C_norm_inf", "*", "‖C‖∞"), ("C_L_norm_inf", "*", "часть C"), ("E_norm_inf", "*", "часть C"),
    ("premises_status", "", "П-1…П-5; П-4 — допущение"), ("cond_S", "", "только запись"), ("min_eig_DM", "*", "min собственного числа DM/D"),
    ("eta", "", "невязка исходной системы; OBSERVATION, порога нет (D-067)"), ("V_exact_rank", "", "точный ранг сохранённого V; диагностика"),
    *((name, "", f"σ_{name[6:]} матрицы R_G R_Vᵀ (до N)") for name in SIGMA_COLUMNS),
    ("rho_N", "", "σ_N/σ₁; только запись"), ("kappa_rep", "", "‖R_G‖₂‖R_V‖₂/σ₁; только запись"),
    ("rank_decision", "", cn.RANK_DECISION), ("P_hat", "N·m", "Π_h^C поля полушага; наблюдение"), ("alpha_centre", "", ALPHA),
    ("alpha_note", "", NOT_A_NODE), ("reason", "", "причина остановки"),
)


def add_parser(commands) -> None:
    run = commands.add_parser(
        "run-cccc-n46", help="расчёт CCCC, N = 4, 5 или 6, только полушаг s = 1; только исследование (D-067)",
        description="CCCC, q = const, N = 4, 5 или 6, режим FIXED_HALF_STEPS_RESEARCH (D-067). Исполняется только полушаг s = 1; при s ≥ 2 — "
                    "защитная остановка до QR решённого набора. Результат всегда RESEARCH_ONLY / PROVISIONAL / NOT_QUALIFIED. Единицы СИ.",
    )
    run.add_argument("--N", type=int, required=True, metavar="{4,5,6}", help="число членов, 4, 5 или 6; N = 1 — run-cccc-n1, N = 2, 3 — run-cccc-n23")
    for name, unit, text in (("a", "m", "сторона вдоль x"), ("b", "m", "сторона вдоль y"), ("t", "m", "толщина"), ("E", "Pa", "модуль упругости"),
                             ("nu", "", "коэффициент Пуассона"), ("q", "Pa", "равномерная нагрузка, не ноль")):
        run.add_argument(f"--{name}", type=float, required=True, metavar="VALUE", help=f"{text}" + (f", {unit}" if unit else ""))
    run.add_argument("--nx", type=int, required=True, metavar="N", help=f"интервалов сетки по x, 2N + 1…{cn.N_MAX}")
    run.add_argument("--ny", type=int, required=True, metavar="N", help=f"интервалов сетки по y, 2N + 1…{cn.N_MAX}")
    run.add_argument("--half-steps", type=int, required=True, metavar="S",
                     help=f"{cn.S_MIN}…{cn.S_MAX}; исполняется только s = 1, при S ≥ 2 — остановка до QR решённого набора")
    run.add_argument("--out", type=Path, required=True, metavar="DIR", help="папка результатов (создаётся)")
    run.add_argument("--overwrite", action="store_true", help="разрешить замену файлов результатов, уже имеющихся в папке; отказ и ошибка их не меняют")


# --- the files ---------------------------------------------------------------------------------------------------------------------

def _history_csv(result: cn.CCCCN46Result) -> str:
    header = [name for name, _, _ in HISTORY_COLUMNS]
    centre_is_node = result.n_x % 2 == 0 and result.n_y % 2 == 0
    rows = []
    for step in result.steps:
        sigma = list(step.sigma or ()) + [None] * len(SIGMA_COLUMNS)
        values = {name: getattr(step, name, None) for name in header}
        values.update(
            orientation=cn.ORIENTATION if step.s == 1 else None,
            conditions=None if step.conditions is None else "; ".join(f"{name}: {held}" for name, held in step.conditions.items()),
            A_diag=None if step.A is None else _joined(np.diag(step.A)), B_diag=None if step.B is None else _joined(np.diag(step.B)),
            C_diag=None if step.C is None else _joined(np.diag(step.C)), F=_joined(step.F), A_norm_inf=_norm(step.A), B_norm_inf=_norm(step.B),
            C_norm_inf=_norm(step.C), C_L_norm_inf=_norm(step.C_L), E_norm_inf=_norm(step.E),
            alpha_note=None if centre_is_node or not step.done else NOT_A_NODE,
            **dict(zip(SIGMA_COLUMNS, sigma, strict=False)),
        )
        rows.append([values[name] for name in header])
    return cli._csv_text(header, rows, (f"{cn.APPLICATION_STATUS}; режим {cn.MODE}; {cn.DECISION}; единицы столбцов — в result.json.",))


def _rank_record(result: cn.CCCCN46Result) -> dict | None:
    if not result.steps:
        return None
    first = result.steps[0]
    return {
        "rule": result.rank_guard_rule, "decision": first.rank_decision, "exact_rank_T": first.exact_rank_T, "exact_rank_G": first.exact_rank_G,
        "V_exact_rank": first.V_exact_rank, "sigma": None if first.sigma is None else list(first.sigma), "rho_N": cli._number(first.rho_N),
        "kappa_rep": cli._number(first.kappa_rep), "cond_S": cli._number(first.cond_S),
        "role": "OBSERVATION: значения статус расчёта не меняют; точный ранг сохранённых binary64-множителей — не сертификат ранга неокруглённого "
                "дискретного решения (D-067)",
        "full_rank_of_exact_discrete_sequence": "UNRESOLVED",
        "half_step_2": None if len(result.steps) < 2 else {"status": result.steps[1].status, "qr_executed": False,
                                                            "conditions": result.steps[1].conditions, "reason": result.steps[1].reason},
    }


def result_texts(args, argv, result: cn.CCCCN46Result) -> dict:
    """{file name: text} of one run, result.json last (no file is touched)."""
    texts = {}
    field_name = None
    if result.half_steps_completed:
        field_name = cli.FIELD_RESEARCH if result.status == cn.COMPLETED_FIXED_HALF_STEPS else cli.FIELD_DIAGNOSTIC
        texts[field_name] = _field_csv(result, field_name == cli.FIELD_DIAGNOSTIC)
    if result.steps:
        texts[cli.FACTORS] = _factors_csv(result)
        texts[cli.HISTORY] = _history_csv(result)
    first = result.steps[0] if result.steps else None
    document = {
        "schema": SCHEMA,
        "program": {"command": "python -m kkvim " + " ".join(argv), "argv": list(argv), "calls": ["kkvim.cccc_n46.run_cccc_n46_physical"]},
        "provenance": {**software_provenance(), "module": "kkvim.cli_cccc_n46"},
        "model": {"boundary": "CCCC", "N": args.N, "theory": "Kirchhoff-Love, linear; homogeneous isotropic plate; uniform q = const",
                  "representation": "w = sum_i X_i(x) Y_i(y), N fixed, no rank reduction", "orientation": cn.ORIENTATION,
                  "qr": "QR of the start set only (D-057.1 item 2, D-067); no QR of the solved set",
                  "assembly": "C-3+ (D-054.3)", "half_step_solver": "B-C, rank-2N correction (D-049.2)",
                  "discretization": "проект, уровень Б (D-049.1, D-050; операции полушага s = 1 ПОЛИТИКИ-23, разд. 2–3, распространены на N = 4–6 "
                                    "решением D-067); авторам КК-МВИ не приписывается",
                  "decision": cn.DECISION},
        "input": {**{name: {"value": cli._number(getattr(args, name)), "unit": unit} for name, unit in cli.INPUT_UNITS},
                  "N": args.N, "n_x": args.nx, "n_y": args.ny, "half_steps": args.half_steps},
        "D": None if result.D is None else {"value": cli._number(result.D), "unit": "N·m"},
        "mode": result.mode,
        "application_status": result.application_status,
        "scientific_status": "RESEARCH_ONLY / PROVISIONAL / NOT_QUALIFIED; полный ранг точной дискретной последовательности — UNRESOLVED; "
                             "Q4 — IN_PROGRESS, R5 — IN_PROGRESS; D1_START = BLOCKED",
        "rank_guard_rule": result.rank_guard_rule,
        "status": result.status,
        "status_note": STATUS_NOTES[result.status],
        "reason": result.reason,
        "half_steps_requested": args.half_steps,
        "half_steps_completed": result.half_steps_completed,
        "half_steps_executable": "только s = 1 (D-067); s ≥ 2 — остановка до QR решённого набора",
        "fixed_half_steps_only": "выполнен только фиксированный исследовательский полушаг; это не сходимость",
        "numerical_policy": {
            "eta": {"value": None if first is None else cli._number(first.eta), "role": "OBSERVATION; защитного порога η нет (D-067)"},
            "not_transferred": "η_max^C = 100u, |η − η_QA| ≤ 16u, c_f = 10 — критерии N = 2, 3 (D-063.3); для N = 4–6 не применяются",
            "recorded_only": "η, TS-1, cond(S), cond(A) до и после QR, σ_i, ρ̂_N, κ_rep, точный ранг сохранённых множителей",
            "safeguards": "неконечные значения; доказанная зависимость стартового набора; предпосылки C-3⁺; факторизация (D-067)",
        },
        "rank": _rank_record(result),
        "premises_c3plus": _premises_record(result),
        "limits": list(LIMITS),
        "field": _field_record(result, field_name),
        "history": None if not result.steps else {
            "file": cli.HISTORY, "rows": len(result.steps),
            "columns": [{"name": name, "unit": unit, "meaning": meaning} for name, unit, meaning in HISTORY_COLUMNS], "units_note": MATRIX_UNITS,
        },
        "factors": None if not result.steps else {"file": cli.FACTORS, "roles": list(FACTOR_ROLES),
                                                  "note": "матрицы полушага s = 1 поэлементно; значения через repr"},
    }
    texts[cli.RESULT] = json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    return texts


# --- the command -------------------------------------------------------------------------------------------------------------------

def main(args, argv) -> int:
    try:
        check_no_interrupted_save(args.out)
        cli.check_output_folder(args.out, args.overwrite)
    except cli.OutputExistsError as exc:
        cli._say(sys.stderr, f"Расчёт не выполнен: {exc}")
        return cli.EXIT_USAGE
    try:
        args.out.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        cli._say(sys.stderr, f"Расчёт не выполнен: не удалось создать папку результатов {args.out}: {exc}")
        return cli.EXIT_OUTPUT_ERROR
    try:
        result = cn.run_cccc_n46_physical(args.a, args.b, args.t, args.E, args.nu, args.q, args.nx, args.ny, args.N, args.half_steps)
        texts = result_texts(args, argv, result)
    except KeyboardInterrupt:
        return _interrupted(args.out)
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
        if not args.overwrite:  # another run may have written here during this one
            cli.check_output_folder(args.out, False)
        check_no_interrupted_save(args.out)  # a save may have been killed here during this run
        paths = replace_results(args.out, texts)
    except KeyboardInterrupt:
        return _interrupted(args.out)
    except cli.OutputExistsError as exc:
        cli._say(sys.stderr, f"Результаты не сохранены: {exc}")
        return cli.EXIT_USAGE
    except cli.OutputWriteError as exc:
        for line in cli.output_error_lines(args.out, exc, result.status):
            cli._say(sys.stderr, line)
        return cli.EXIT_OUTPUT_ERROR
    except Exception:  # noqa: BLE001
        code = cli._internal_error("запись результатов")
        _say_kept(args.out)
        return code
    _summary(args, result, texts, paths)
    return code


def _summary(args, result: cn.CCCCN46Result, texts: dict, paths: dict) -> None:
    document = json.loads(texts[cli.RESULT])
    stream = sys.stderr if result.status != cn.COMPLETED_FIXED_HALF_STEPS else sys.stdout
    cli._say(stream, f"CCCC, N = {args.N}, {result.mode}: {result.application_status}")
    cli._say(stream, f"Статус: {result.status}; выполнено полушагов {result.half_steps_completed} (исполняется только s = 1), запрошено {args.half_steps}")
    cli._say(stream, document["status_note"])
    if result.reason:
        cli._say(stream, f"Причина: {result.reason}")
    field = document["field"]
    if field:
        w_max = field["w_max"]
        cli._say(stream, f"max|w| = {w_max['value']!r} м в узле ({w_max['i']}, {w_max['j']}), x = {w_max['x']!r}, y = {w_max['y']!r}; поле: {field['role']}")
        cli._say(stream, f"alpha_centre = {field['alpha_centre']!r}" if field["alpha_centre"] is not None else f"alpha_centre: {NOT_A_NODE}")
        step = result.completed[-1]
        cli._say(stream, f"s = 1: η = {step.eta!r} (наблюдение, порога нет); P_hat = {step.P_hat!r}; точный ранг V = {step.V_exact_rank}; "
                         f"ρ̂_N = {step.rho_N!r}; cond(S) = {step.cond_S!r} (только запись)")
    diagnostic = document["provenance"]["git_diagnostic"]
    if diagnostic:
        cli._say(stream, f"Версия: {diagnostic}")
    cli._say(stream, f"Результаты: {args.out.resolve()}")
    for name in paths:
        cli._say(stream, f"  {name}")
