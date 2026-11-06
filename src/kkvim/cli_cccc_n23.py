"""``python -m kkvim run-cccc-n23``: CCCC, N = 2 or 3, the half-step s = 1; the limited research mode of D-063.

    python -m kkvim run-cccc-n23 --N 3 --a 1 --b 1 --t 0.01 --E 2.1e11 --nu 0.3 --q 1000 --nx 32 --ny 32 --half-steps 1 --out scratch/cccc23

The command calls ``kkvim.cccc_n23.run_cccc_n23_physical`` and writes what it returns; it computes no quantity of the method. Every result is
RESEARCH_ONLY / PROVISIONAL / NOT_QUALIFIED. Only s = 1 is executed: with --half-steps 2...8 the half-step s = 2 stops before the QR of the
solved set (STOPPED_RANK_GUARD_UNCERTIFIED, D-057.1; D-063.2), and the field of s = 1 is saved as diagnostic.

Files: result.json; history.csv (s = 1 and, if asked for, s = 2); factors.csv (T, R, G, A, B, C, C_L, E, F, V with repr, for the QA checks of
Policy 9.2); field_research.csv after --half-steps 1, field_diagnostic.csv after a stop at s = 2; no field without the half-step s = 1.

Saving is ``kkvim.cli_ssss.replace_results`` with the check of an unfinished replacement and exit code 130 on Ctrl+C (D-062.1; Policy 10),
with the guarantees and limits of that mechanism; two runs saving into one folder at the same moment are not guarded against each other.
A refused input, an ERROR_* run or a run without the half-step s = 1 does not touch earlier results: the status and the reason go to stderr.
If the folder has no earlier results, such a run writes its result.json.

Exit codes (Policy 5): 0 - --half-steps 1, s = 1 done (not convergence); 2 - outside the domain, nothing computed; 3 - ERROR_*; 4, 5 - as in
``kkvim.cli``; 6 - a guard or a protective stop, the rank guard of s = 2 included; 130 - interrupted by the user.
"""

import json
import sys
from pathlib import Path

import numpy as np

import kkvim.cccc_n23 as cn
import kkvim.cli as cli
from kkvim.cli_ssss import _interrupted, _kept, _say_kept, check_no_interrupted_save, replace_results, software_provenance

SCHEMA = "kkvim.cli.cccc_n23/1"
EXIT_PROTECTIVE_STOP = 6
EXIT_CODES = {
    cn.COMPLETED_FIXED_HALF_STEPS: cli.EXIT_CONVERGED, cn.STOPPED_RANK_GUARD_UNCERTIFIED: EXIT_PROTECTIVE_STOP, cn.STOP_BEFORE_QR_S1: EXIT_PROTECTIVE_STOP,
    cn.NOT_EXECUTED_OUTSIDE_DOMAIN: cli.EXIT_USAGE, cn.ERROR_INPUT: cli.EXIT_SOLVER_ERROR, cn.ERROR_NONFINITE: cli.EXIT_SOLVER_ERROR,
    cn.UNRESOLVED_C3PLUS_PREMISE: EXIT_PROTECTIVE_STOP, cn.PROTECTIVE_STOP_FACTORIZATION: EXIT_PROTECTIVE_STOP, cn.STOPPED_ETA_GUARD: EXIT_PROTECTIVE_STOP,
    cn.STOP_ZERO_FIELD: EXIT_PROTECTIVE_STOP,
}
KEEPS_EARLIER = (cn.ERROR_INPUT, cn.ERROR_NONFINITE, cn.NOT_EXECUTED_OUTSIDE_DOMAIN)  # these, and any run without s = 1, replace no earlier file

STATUS_NOTES = {
    cn.COMPLETED_FIXED_HALF_STEPS: "Полушаг s = 1 выполнен, защиты не сработали. Это не сходимость, не полный ранг поля и не квалификация.",
    cn.STOPPED_RANK_GUARD_UNCERTIFIED: "Полушаг s = 1 выполнен; полушаг s = 2 остановлен до QR решённого набора: утверждённой ранговой процедуры "
                                       "нет (D-057.1; D-063.2). Поле s = 1 — только диагностика. Защитная остановка — не вывод о методе.",
    cn.STOP_BEFORE_QR_S1: "Доказанная зависимость набора (точный ранг меньше N) до QR (D-049.5, п. 2). Поле s = 1, если есть, — только диагностика.",
    cn.NOT_EXECUTED_OUTSIDE_DOMAIN: "Расчёт не выполнялся: ввод вне области D-063.2. Значений решения нет.",
    cn.ERROR_INPUT: "Расчёт не выполнялся: недопустимый ввод. Значений решения нет.",
    cn.ERROR_NONFINITE: "Неконечное значение в расчёте полушага s = 1. Решения нет.",
    cn.UNRESOLVED_C3PLUS_PREMISE: "Предпосылка сборки C-3⁺ не установлена. Решения нет.",
    cn.PROTECTIVE_STOP_FACTORIZATION: "Защитная остановка QR или решения полушага; при независимом наборе это противоречит теории, то есть признак "
                                      "дефекта, а не вывод о методе. Решения нет.",
    cn.STOPPED_ETA_GUARD: "Невязка выше защитного порога η_max^C = 100u (D-063.3, вариант А). Без уточнения и смены решателя. Решения нет.",
    cn.STOP_ZERO_FIELD: "Решение полушага тождественно равно нулю. Решения нет.",
}
LIMITS = (
    "Линейная изотропная однородная пластина Кирхгофа–Лява, CCCC, q = const ≠ 0, N = 2 или 3, 0,2 ≤ b/a ≤ 5, 7 ≤ n_x, n_y ≤ 1280; исполняется "
    "только полушаг s = 1 (D-063.2).",
    "Полушаги s ≥ 2 при N ≥ 2 не разрешены до отдельного решения о ранговой процедуре: сертификата или численного критерия ранга нет (O4 — "
    "UNRESOLVED; ПОЛИТИКА-23, разд. 4).",
    "Один полушаг не приближает решение дискретной задачи ω_h^C и непрерывной ω₀ (NS 9, О-8). Режим не утверждает сходимости, полного ранга "
    "w⁽¹⁾ (N = 3 непрерывно и любое N ≥ 2 дискретно — UNRESOLVED, ВБC-20), квалификации и приёмки R5, Q4, Q5 (D-063.4).",
    "Поле относится к span(G), стартовому набору 𝒯_N после округления QR, а не к точной оболочке 𝒯_N.",
    "Предпосылка П-4 (классическое скалярное произведение в BLAS) принята как допущение и не проверяется.",
    "B-C не квалифицирован на рабочих сетках; К-1…К-5 ПОЛИТИКИ-23 — проверки QA на малых случаях разд. 8.",
)
EVIDENCE_SCOPE = ("Данные ПОЛИТИКИ-23, разд. 6.1, получены при a = 1, D = q = 1 на том же пути s = 1 (стартовый набор, QR прототипа, нулевой "
                  "сопряжённый множитель). Другие a, D, q — та же дискретная задача: w̄ = wD/(qa⁴) зависит только от x/a, y/b, b/a.")
W_BAR = "w_bar = w D / (q a^4)"
ALPHA = "alpha_centre = w(a/2, b/2) D / (q l^4), l = min(a, b); only with even n_x and n_y"
NOT_A_NODE = "центр не узел"
MATRIX_UNITS = "* — единицы зависят от базиса G после QR; значения пишутся без пересчёта."

HISTORY_COLUMNS = (
    ("s", "", "номер полушага"), ("k", "", "⌊s/2⌋"), ("direction", "", "решаемые множители: Y при s = 1, X при s = 2"),
    ("status", "", "HALF_STEP_DONE или, в последней строке, статус расчёта"), ("n_s", "", "интервалов по решаемому направлению"),
    ("h_s", "m", "шаг по решаемому направлению"), ("N", "", "число членов"), ("orientation", "", cn.ORIENTATION),
    ("conditions", "", "s = 1: N ≤ ⌊n_x/2⌋, 2N < n_x, N ≤ ⌊n_y/2⌋; s = 2: конфигурационный S-1 (РН-3), только запись"),
    ("exact_rank_T", "", "точный ранг стартового набора"), ("exact_rank_G", "", "точный ранг G после QR"),
    ("qr_executed", "", "s = 1: QR стартового набора; s = 2: QR не выполняется"), ("R_diag_min_abs", "*", "min |diag R|"),
    ("cond_A_before_QR", "", "cond(R)^2; только запись"), ("cond_A_after_QR", "", "по собственным числам A(G); только запись"),
    ("A_diag", "*", "диагональ A через ';'"), ("B_diag", "*", "диагональ B"), ("C_diag", "*", "диагональ C (C-3⁺)"), ("F", "*", "правая часть"),
    ("A_norm_inf", "*", "‖A‖∞"), ("B_norm_inf", "*", "‖B‖∞"), ("C_norm_inf", "*", "‖C‖∞"), ("C_L_norm_inf", "*", "часть C"), ("E_norm_inf", "*", "часть C"),
    ("premises_status", "", "П-1…П-5; П-4 — допущение"), ("cond_S", "", "только запись"), ("min_eig_DM", "*", "min собственного числа DM/D"),
    ("eta", "", "невязка исходной системы"), ("V_exact_rank", "", "точный ранг V"), ("sigma_1", "", "σ₁ матрицы R_G R_Vᵀ"),
    ("sigma_2", "", "σ₂"), ("sigma_3", "", "σ₃ (N = 3)"), ("rho_N", "", "σ_N/σ₁; только запись"), ("kappa_rep", "", "‖R_G‖₂‖R_V‖₂/σ₁; только запись"),
    ("rank_decision", "", cn.RANK_DECISION), ("P_hat", "N·m", "Π_h^C при w = G Vᵀ; наблюдение"), ("alpha_centre", "", ALPHA),
    ("alpha_note", "", NOT_A_NODE), ("reason", "", "причина остановки"),
)
FACTOR_ROLES = ("T", "R", "G", "A", "B", "C", "C_L", "E", "F", "V")


def add_parser(commands) -> None:
    run = commands.add_parser(
        "run-cccc-n23", help="расчёт CCCC, N = 2 или 3, полушаг s = 1; только исследование (D-063)",
        description="CCCC, q = const, N = 2 или 3, режим FIXED_HALF_STEPS_RESEARCH (D-063). Исполняется только полушаг s = 1; при --half-steps ≥ 2 "
                    "полушаг s = 2 останавливается до QR (D-057.1). Результат всегда RESEARCH_ONLY / PROVISIONAL / NOT_QUALIFIED. Единицы СИ.",
    )
    run.add_argument("--N", type=int, required=True, metavar="{2,3}", help="число членов, 2 или 3; N = 1 — команда run-cccc-n1")
    for name, unit, text in (("a", "m", "сторона вдоль x"), ("b", "m", "сторона вдоль y"), ("t", "m", "толщина"), ("E", "Pa", "модуль упругости"),
                             ("nu", "", "коэффициент Пуассона"), ("q", "Pa", "равномерная нагрузка, не ноль")):
        run.add_argument(f"--{name}", type=float, required=True, metavar="VALUE", help=f"{text}" + (f", {unit}" if unit else ""))
    run.add_argument("--nx", type=int, required=True, metavar="N", help=f"интервалов сетки по x, {cn.N_MIN}…{cn.N_MAX}")
    run.add_argument("--ny", type=int, required=True, metavar="N", help=f"интервалов сетки по y, {cn.N_MIN}…{cn.N_MAX}")
    run.add_argument("--half-steps", type=int, required=True, metavar="S",
                     help=f"{cn.S_MIN}…{cn.S_MAX}; исполняется только s = 1, при S ≥ 2 — защитная остановка полушага s = 2 до QR")
    run.add_argument("--out", type=Path, required=True, metavar="DIR", help="папка результатов (создаётся)")
    run.add_argument("--overwrite", action="store_true", help="разрешить замену файлов результатов, уже имеющихся в папке; отказ и ошибка их не меняют")


# --- the files ---------------------------------------------------------------------------------------------------------------------

def _joined(values) -> str | None:
    return None if values is None else ";".join(repr(float(value)) for value in values)


def _norm(M) -> float | None:
    return None if M is None else float(np.max(np.sum(np.abs(M), axis=1)))


def _history_csv(result: cn.CCCCN23Result) -> str:
    header = [name for name, _, _ in HISTORY_COLUMNS]
    centre_is_node = result.n_x % 2 == 0 and result.n_y % 2 == 0
    rows = []
    for step in result.steps:
        sigma = list(step.sigma or ()) + [None] * 3
        values = {name: getattr(step, name, None) for name in header}
        values.update(
            orientation=cn.ORIENTATION if step.s == 1 else None,
            conditions=None if step.conditions is None else "; ".join(f"{name}: {held}" for name, held in step.conditions.items()),
            A_diag=None if step.A is None else _joined(np.diag(step.A)), B_diag=None if step.B is None else _joined(np.diag(step.B)),
            C_diag=None if step.C is None else _joined(np.diag(step.C)), F=_joined(step.F), A_norm_inf=_norm(step.A), B_norm_inf=_norm(step.B),
            C_norm_inf=_norm(step.C), C_L_norm_inf=_norm(step.C_L), E_norm_inf=_norm(step.E), sigma_1=sigma[0], sigma_2=sigma[1], sigma_3=sigma[2],
            alpha_note=None if centre_is_node or not step.done else NOT_A_NODE,
        )
        rows.append([values[name] for name in header])
    return cli._csv_text(header, rows, (f"{cn.APPLICATION_STATUS}; режим {cn.MODE}; единицы столбцов — в result.json.",))


def _factors_csv(result: cn.CCCCN23Result) -> str:
    """Every matrix of the half-step s = 1 with repr, element by element (i, j from 1): enough to restore it bit for bit."""
    rows = []
    for step in result.steps:
        if step.s != 1:
            continue
        for role in FACTOR_ROLES:
            values = getattr(step, role)
            if values is None:
                continue
            matrix = np.asarray(values, dtype=float).reshape(len(values), -1)
            rows.extend((step.s, step.direction, role, i + 1, j + 1, float(matrix[i, j])) for i in range(matrix.shape[0]) for j in range(matrix.shape[1]))
    return cli._csv_text(("s", "direction", "role", "i", "j", "value"), rows, (f"{cn.APPLICATION_STATUS}; F — вектор: j = 1.",))


def _field_csv(result: cn.CCCCN23Result, diagnostic: bool) -> str:
    w, w_bar = result.field(), result.dimensionless_field()
    h_x, h_y = result.a / result.n_x, result.b / result.n_y
    comment = f"{cn.APPLICATION_STATUS}; поле полушага s = 1, N = {result.N}."
    if diagnostic:
        comment = f"DIAGNOSTIC: статус {result.status}; поле полушага s = 1. Не результат расчёта. {comment}"
    rows = ((i, j, i * h_x, j * h_y, float(w[i, j]), float(w_bar[i, j])) for i in range(w.shape[0]) for j in range(w.shape[1]))
    return cli._csv_text(("i", "j", "x [m]", "y [m]", "w [m]", "w_bar [1]"), rows, (comment,))


def _field_record(result: cn.CCCCN23Result, name: str | None) -> dict | None:
    if name is None:
        return None
    w = result.field()
    i, j = np.unravel_index(int(np.argmax(np.abs(w))), w.shape)
    h_x, h_y = result.a / result.n_x, result.b / result.n_y
    first = result.completed[-1]
    return {
        "file": name, "role": "research" if name == cli.FIELD_RESEARCH else "diagnostic", "half_step": 1,
        "shape": list(w.shape), "axes": "axis 0 = x (i = 0..n_x), axis 1 = y (j = 0..n_y), boundary nodes included", "definition": "w(1) = G V^T",
        "units": {"x": "m", "y": "m", "w": "m", "w_bar": "1"}, "w_bar_definition": W_BAR,
        "w_max": {"value": float(abs(w[i, j])), "i": int(i), "j": int(j), "x": float(i * h_x), "y": float(j * h_y)},
        "alpha_centre": cli._number(first.alpha_centre), "alpha_definition": ALPHA,
        "alpha_note": None if first.alpha_centre is not None else NOT_A_NODE,
    }


def _rank_record(result: cn.CCCCN23Result) -> dict | None:
    if not result.steps:
        return None
    first = result.steps[0]
    return {
        "rule": result.rank_guard_rule, "decision": first.rank_decision, "exact_rank_T": first.exact_rank_T, "exact_rank_G": first.exact_rank_G,
        "V_exact_rank": first.V_exact_rank, "sigma": None if first.sigma is None else list(first.sigma), "rho_N": cli._number(first.rho_N),
        "kappa_rep": cli._number(first.kappa_rep), "role": "OBSERVATION: значения статус расчёта не меняют (ПОЛИТИКА-23, разд. 3, шаг 7)",
        "half_step_2": None if len(result.steps) < 2 else {"status": result.steps[1].status, "qr_executed": False, "conditions": result.steps[1].conditions},
    }


def _premises_record(result: cn.CCCCN23Result) -> dict:
    statuses = sorted({step.premises_status for step in result.steps if step.premises_status})
    reasons = [f"s = {step.s}: {reason}" for step in result.steps for reason in step.premises_reasons]
    return {"status_by_half_steps": statuses, "reasons": reasons, "P-4": cn.P4_ASSUMED}


def result_texts(args, argv, result: cn.CCCCN23Result) -> dict:
    """{file name: text} of one run, result.json last (no file is touched)."""
    texts = {}
    field_name = None
    if result.half_steps_completed:
        field_name = cli.FIELD_RESEARCH if result.status == cn.COMPLETED_FIXED_HALF_STEPS else cli.FIELD_DIAGNOSTIC
        texts[field_name] = _field_csv(result, field_name == cli.FIELD_DIAGNOSTIC)
    if result.steps:
        texts[cli.FACTORS] = _factors_csv(result)
        texts[cli.HISTORY] = _history_csv(result)
    document = {
        "schema": SCHEMA,
        "program": {"command": "python -m kkvim " + " ".join(argv), "argv": list(argv), "calls": ["kkvim.cccc_n23.run_cccc_n23_physical"]},
        "provenance": {**software_provenance(), "module": "kkvim.cli_cccc_n23"},
        "model": {"boundary": "CCCC", "N": args.N, "theory": "Kirchhoff-Love, linear; homogeneous isotropic plate; uniform q = const",
                  "representation": "w = sum_i X_i(x) Y_i(y), N fixed", "orientation": cn.ORIENTATION, "qr": "QR of the start set only (D-057.1 item 2)",
                  "assembly": "C-3+ (D-054.3)", "half_step_solver": "B-C, rank-2N correction (D-049.2)",
                  "discretization": "проект, уровень Б (D-049.1, D-050); авторам КК-МВИ не приписывается", "decision": "D-063"},
        "input": {**{name: {"value": cli._number(getattr(args, name)), "unit": unit} for name, unit in cli.INPUT_UNITS},
                  "N": args.N, "n_x": args.nx, "n_y": args.ny, "half_steps": args.half_steps},
        "D": None if result.D is None else {"value": cli._number(result.D), "unit": "N·m"},
        "mode": result.mode,
        "application_status": result.application_status,
        "rank_guard_rule": result.rank_guard_rule,
        "status": result.status,
        "status_note": STATUS_NOTES[result.status],
        "reason": result.reason,
        "half_steps_requested": args.half_steps,
        "half_steps_completed": result.half_steps_completed,
        "half_steps_executable": "только s = 1 (D-063.2)",
        "numerical_policy": {
            "eta_max_C": {"value": cn.ETA_MAX_C, "expression": "100u, u = 2^-53", "role": "защитный порог только этого режима, не квалифицированное значение (D-063.3)"},
            "recorded_only": "cond(S), cond(A) до и после QR, σ_i, ρ̂_N, κ_rep",
            "qa_criteria": "|η − η_QA| ≤ 16u и TS-1 (c_f = 10) — критерии приёмки первой версии только на случаях ПОЛИТИКИ-23, разд. 8; решатель их не применяет",
        },
        "rank": _rank_record(result),
        "premises_c3plus": _premises_record(result),
        "evidence_scope": EVIDENCE_SCOPE,
        "limits": list(LIMITS),
        "field": _field_record(result, field_name),
        "history": None if not result.steps else {
            "file": cli.HISTORY, "rows": len(result.steps),
            "columns": [{"name": name, "unit": unit, "meaning": meaning} for name, unit, meaning in HISTORY_COLUMNS], "units_note": MATRIX_UNITS,
        },
        "factors": None if not result.steps else {"file": cli.FACTORS, "roles": list(FACTOR_ROLES), "note": "матрицы полушага s = 1 поэлементно; значения через repr"},
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
        result = cn.run_cccc_n23_physical(args.a, args.b, args.t, args.E, args.nu, args.q, args.nx, args.ny, args.N, args.half_steps)
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


def _summary(args, result: cn.CCCCN23Result, texts: dict, paths: dict) -> None:
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
        first = result.steps[0]
        cli._say(stream, f"η = {first.eta!r}; точный ранг V = {first.V_exact_rank}; ρ̂_N = {first.rho_N!r} (только запись; {cn.RANK_DECISION})")
    diagnostic = document["provenance"]["git_diagnostic"]
    if diagnostic:
        cli._say(stream, f"Версия: {diagnostic}")
    cli._say(stream, f"Результаты: {args.out.resolve()}")
    for name in paths:
        cli._say(stream, f"  {name}")
