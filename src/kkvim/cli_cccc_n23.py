"""``python -m kkvim run-cccc-n23``: CCCC, N = 2 or 3, the half-steps s = 1 (D-063) and s = 2 (D-065); limited research modes.

    python -m kkvim run-cccc-n23 --N 3 --a 1 --b 1 --t 0.01 --E 2.1e11 --nu 0.3 --q 1000 --nx 32 --ny 32 --half-steps 2 --out scratch/cccc23

The command calls ``kkvim.cccc_n23.run_cccc_n23_physical`` and writes what it returns; it computes no quantity of the method. Every result is
RESEARCH_ONLY / PROVISIONAL / NOT_QUALIFIED. --half-steps 1: s = 1. --half-steps 2: s = 2 only for n_x, n_y <= 32 and only after the RN-5
certificate before the QR of the solved set and RT-7 after it (D-065); otherwise a stop before s = 2 (STOPPED_RANK_GUARD_UNCERTIFIED) with
the field of s = 1 as diagnostic. --half-steps 3...8: after s = 2 a stop before the QR that would open s = 3, the field of s = 2 diagnostic.

Files: result.json (with the RN-5 and RT-7 records); history.csv (one row per half-step reached); factors.csv (T, R, G, A, B, C, C_L, E, F, V
and, at s = 2, the companion, with repr, for the QA checks); field_research.csv when all half-steps asked for are done, field_diagnostic.csv
after a stop; no field without the half-step s = 1.

Saving is ``kkvim.cli_ssss.replace_results`` with the check of an unfinished replacement and exit code 130 on Ctrl+C (D-062.1; Policy 10),
with the guarantees and limits of that mechanism; two runs saving into one folder at the same moment are not guarded against each other.
A refused input, an ERROR_* run or a run without the half-step s = 1 does not touch earlier results: the status and the reason go to stderr.
If the folder has no earlier results, such a run writes its result.json.

Exit codes (Policy 5): 0 - the half-steps asked for (1 or 2) done (not convergence); 2 - outside the domain, nothing computed; 3 - ERROR_*;
4, 5 - as in ``kkvim.cli``; 6 - a guard or a protective stop, the rank guards of s = 2 and s = 3 included; 130 - interrupted by the user.
"""

import json
import sys
from pathlib import Path

import numpy as np

import kkvim.cccc_n23 as cn
import kkvim.cli as cli
from kkvim.cli_ssss import _interrupted, _kept, _say_kept, check_no_interrupted_save, replace_results, software_provenance

SCHEMA = "kkvim.cli.cccc_n23/2"
EXIT_PROTECTIVE_STOP = 6
EXIT_CODES = {
    cn.COMPLETED_FIXED_HALF_STEPS: cli.EXIT_CONVERGED, cn.STOPPED_RANK_GUARD_UNCERTIFIED: EXIT_PROTECTIVE_STOP, cn.STOP_BEFORE_QR_S1: EXIT_PROTECTIVE_STOP,
    cn.NOT_EXECUTED_OUTSIDE_DOMAIN: cli.EXIT_USAGE, cn.ERROR_INPUT: cli.EXIT_SOLVER_ERROR, cn.ERROR_NONFINITE: cli.EXIT_SOLVER_ERROR,
    cn.UNRESOLVED_C3PLUS_PREMISE: EXIT_PROTECTIVE_STOP, cn.PROTECTIVE_STOP_FACTORIZATION: EXIT_PROTECTIVE_STOP, cn.STOPPED_ETA_GUARD: EXIT_PROTECTIVE_STOP,
    cn.STOP_ZERO_FIELD: EXIT_PROTECTIVE_STOP,
}
KEEPS_EARLIER = (cn.ERROR_INPUT, cn.ERROR_NONFINITE, cn.NOT_EXECUTED_OUTSIDE_DOMAIN)  # these, and any run without s = 1, replace no earlier file

STATUS_NOTES = {
    cn.COMPLETED_FIXED_HALF_STEPS: "Выполнены запрошенные фиксированные исследовательские полушаги (s = 1 или s = 1, 2), защиты не сработали. "
                                   "Это не сходимость, не точность поля, не полный ранг и не квалификация.",
    cn.STOPPED_RANK_GUARD_UNCERTIFIED: "Защитная остановка до QR, открывающего следующий полушаг (s = 2: сетка вне D-065.5 или RN-5 не выполнен; "
                                       "после QR — RT-7 не выполнен; s = 3 не разрешён). Поле последнего выполненного полушага — только "
                                       "диагностика. Невыполненный достаточный сертификат не доказывает потери ранга.",
    cn.STOP_BEFORE_QR_S1: "Доказанная зависимость набора (точный ранг меньше N) до QR (D-049.5, п. 2). Поле s = 1, если есть, — только диагностика.",
    cn.NOT_EXECUTED_OUTSIDE_DOMAIN: "Расчёт не выполнялся: ввод вне области D-063.2. Значений решения нет.",
    cn.ERROR_INPUT: "Расчёт не выполнялся: недопустимый ввод. Значений решения нет.",
    cn.ERROR_NONFINITE: "Неконечное значение в расчёте полушага. Решения этого полушага нет.",
    cn.UNRESOLVED_C3PLUS_PREMISE: "Предпосылка сборки C-3⁺ не установлена. Решения этого полушага нет.",
    cn.PROTECTIVE_STOP_FACTORIZATION: "Защитная остановка QR или решения полушага; при независимом наборе это противоречит теории, то есть признак "
                                      "дефекта, а не вывод о методе. Решения этого полушага нет.",
    cn.STOPPED_ETA_GUARD: "Невязка выше защитного порога η_max^C = 100u (D-063.3, вариант А). Без уточнения и смены решателя. Решения нет.",
    cn.STOP_ZERO_FIELD: "Решение полушага тождественно равно нулю. Решения этого полушага нет.",
}
LIMITS = (
    "Линейная изотропная однородная пластина Кирхгофа–Лява, CCCC, q = const ≠ 0, N = 2 или 3, 0,2 ≤ b/a ≤ 5, 7 ≤ n_x, n_y ≤ 1280; полушаг s = 1 "
    "(D-063.2); полушаг s = 2 только при n_x, n_y ≤ 32 (D-065.5) после сертификатов RN-5 и RT-7 этого расчёта; s ≥ 3 не исполняется.",
    "RN-5 доказывает только rank w_h⁽¹⁾ = N точной дискретной последовательности для этого расчёта; RT-7 — только трансверсальность span G2 "
    "к точному подпространству по y (sin θ_max ≤ √τ + t < 1). Ни точности поля s = 2, ни приемлемой обусловленности, ни ранга w⁽²⁾, ни "
    "сходимости, ни свойств s ≥ 3 они не устанавливают (D-065). RN-5 и RT-7 — выводы проекта (уровень Б), не авторов КК-МВИ.",
    "Фиксированные полушаги не приближают решение дискретной задачи ω_h^C и непрерывной ω₀ (NS 9, О-8). Режим не утверждает сходимости, полного "
    "ранга (ВБC-20 — UNRESOLVED), квалификации и приёмки R5, Q4, Q5 (D-063.4, D-065). Пороги s = 1 (100u, 16u, c_f = 10) на s = 2 не переносятся: "
    "η, cond(S), σ_i и энергия s = 2 — только наблюдения.",
    "Поле s = 1 относится к span(G), стартовому набору 𝒯_N после округления QR; поле s = 2 — к span(G2) после округления QR решённого набора.",
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
    ("s", "", "номер полушага"), ("k", "", "⌊s/2⌋"), ("direction", "", "решаемые множители: Y при s = 1, X при s = 2, Y при s = 3"),
    ("status", "", "HALF_STEP_DONE или, в последней строке, статус расчёта"), ("n_s", "", "интервалов по решаемому направлению"),
    ("h_s", "m", "шаг по решаемому направлению"), ("N", "", "число членов"), ("orientation", "", cn.ORIENTATION),
    ("conditions", "", "s = 1: N ≤ ⌊n_x/2⌋, 2N < n_x, N ≤ ⌊n_y/2⌋; s ≥ 2: конфигурационный S-1 (РН-3) и сетка D-065.5"),
    ("exact_rank_T", "", "точный ранг стартового набора"), ("exact_rank_G", "", "точный ранг G после QR"),
    ("qr_executed", "", "s = 1: QR стартового набора; s = 2: QR решённого набора V после RN-5; s = 3: QR не выполняется"), ("R_diag_min_abs", "*", "min |diag R|"),
    ("cond_A_before_QR", "", "cond(R)^2; только запись"), ("cond_A_after_QR", "", "по собственным числам A(G); только запись"),
    ("A_diag", "*", "диагональ A через ';'"), ("B_diag", "*", "диагональ B"), ("C_diag", "*", "диагональ C (C-3⁺)"), ("F", "*", "правая часть"),
    ("A_norm_inf", "*", "‖A‖∞"), ("B_norm_inf", "*", "‖B‖∞"), ("C_norm_inf", "*", "‖C‖∞"), ("C_L_norm_inf", "*", "часть C"), ("E_norm_inf", "*", "часть C"),
    ("premises_status", "", "П-1…П-5; П-4 — допущение"), ("cond_S", "", "только запись"), ("min_eig_DM", "*", "min собственного числа DM/D"),
    ("eta", "", "невязка исходной системы"), ("V_exact_rank", "", "точный ранг V"), ("sigma_1", "", "σ₁ матрицы R_G R_Vᵀ"),
    ("sigma_2", "", "σ₂"), ("sigma_3", "", "σ₃ (N = 3)"), ("rho_N", "", "σ_N/σ₁; только запись"), ("kappa_rep", "", "‖R_G‖₂‖R_V‖₂/σ₁; только запись"),
    ("rank_decision", "", cn.RANK_DECISION), ("P_hat", "N·m", "Π_h^C поля полушага; наблюдение"), ("alpha_centre", "", ALPHA),
    ("alpha_note", "", NOT_A_NODE),
    ("rn5_verdict", "", "s = 2: результат RN-5 до QR"), ("rn5_eps", "", "s = 2: ε ≥ ‖(ŵ₁ − w_h⁽¹⁾)_w‖₂, округлено для записи"),
    ("rn5_sigma_low", "", "s = 2: σ_N(ŵ₁,w) ≥ это значение, округлено для записи"), ("rn5_margin", "", "s = 2: σ_low/ε, строго > 1 при RN5_CERTIFIED"),
    ("rt7_passed", "", "s = 2: R7-1, R7-2 и R7-3 выполнены"), ("rt7_t", "", "s = 2: t = ε/σ_low"), ("rt7_tau", "", "s = 2: τ, точно; округлено для записи"),
    ("rt7_sin_theta_upper", "", "s = 2: верхняя граница √τ + t"), ("P_hat_previous_QR", "N·m", "s = 2: Π_h^C поля s = 1 по преобразованным массивам"),
    ("field_preservation", "", "s = 2: max|G Rᵀ G2ᵀ − G Vᵀ| / max|G Vᵀ|; наблюдение"), ("reason", "", "причина остановки"),
)
FACTOR_ROLES = ("T", "R", "G", "A", "B", "C", "C_L", "E", "F", "V", "companion")


def add_parser(commands) -> None:
    run = commands.add_parser(
        "run-cccc-n23", help="расчёт CCCC, N = 2 или 3, полушаги s = 1 и s = 2; только исследование (D-063, D-065)",
        description="CCCC, q = const, N = 2 или 3, режим FIXED_HALF_STEPS_RESEARCH (D-063, D-065). Полушаг s = 1; s = 2 только при n_x, n_y ≤ 32 "
                    "после сертификатов RN-5 и RT-7; s ≥ 3 — защитная остановка до QR. Результат всегда RESEARCH_ONLY / PROVISIONAL / "
                    "NOT_QUALIFIED. Единицы СИ.",
    )
    run.add_argument("--N", type=int, required=True, metavar="{2,3}", help="число членов, 2 или 3; N = 1 — команда run-cccc-n1")
    for name, unit, text in (("a", "m", "сторона вдоль x"), ("b", "m", "сторона вдоль y"), ("t", "m", "толщина"), ("E", "Pa", "модуль упругости"),
                             ("nu", "", "коэффициент Пуассона"), ("q", "Pa", "равномерная нагрузка, не ноль")):
        run.add_argument(f"--{name}", type=float, required=True, metavar="VALUE", help=f"{text}" + (f", {unit}" if unit else ""))
    run.add_argument("--nx", type=int, required=True, metavar="N", help=f"интервалов сетки по x, {cn.N_MIN}…{cn.N_MAX}")
    run.add_argument("--ny", type=int, required=True, metavar="N", help=f"интервалов сетки по y, {cn.N_MIN}…{cn.N_MAX}")
    run.add_argument("--half-steps", type=int, required=True, metavar="S",
                     help=f"{cn.S_MIN}…{cn.S_MAX}; s = 2 только при n_x, n_y ≤ {cn.S2_GRID_MAX} после RN-5 и RT-7, при S ≥ 3 — остановка до QR s = 3")
    run.add_argument("--out", type=Path, required=True, metavar="DIR", help="папка результатов (создаётся)")
    run.add_argument("--overwrite", action="store_true", help="разрешить замену файлов результатов, уже имеющихся в папке; отказ и ошибка их не меняют")


# --- the files ---------------------------------------------------------------------------------------------------------------------

def _joined(values) -> str | None:
    return None if values is None else ";".join(repr(float(value)) for value in values)


def _norm(M) -> float | None:
    return None if M is None else float(np.max(np.sum(np.abs(M), axis=1)))


def _float(value) -> float | None:
    return None if value is None else float(value)


def _certificate_columns(step) -> dict:
    rn5, rt7 = step.rn5 or {}, step.rt7 or {}
    values = rn5.get("values", {})
    return {"rn5_verdict": rn5.get("verdict"), "rn5_eps": _float(values.get("eps")), "rn5_sigma_low": _float(values.get("sigma_low")),
            "rn5_margin": _float(values.get("margin_sigma_over_eps")), "rt7_passed": rt7.get("passed"), "rt7_t": _float(rt7.get("t")),
            "rt7_tau": _float(rt7.get("tau")), "rt7_sin_theta_upper": _float(rt7.get("sin_theta_upper"))}


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
            alpha_note=None if centre_is_node or not step.done else NOT_A_NODE, **_certificate_columns(step),
        )
        rows.append([values[name] for name in header])
    return cli._csv_text(header, rows, (f"{cn.APPLICATION_STATUS}; режим {cn.MODE}; единицы столбцов — в result.json.",))


def _factors_csv(result: cn.CCCCN23Result) -> str:
    """Every matrix of the half-steps s = 1 and s = 2 with repr, element by element (i, j from 1): enough to restore it bit for bit."""
    rows = []
    for step in result.steps:
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
    s = result.completed[-1].s
    comment = f"{cn.APPLICATION_STATUS}; поле полушага s = {s}, N = {result.N}."
    if diagnostic:
        comment = f"DIAGNOSTIC: статус {result.status}; поле полушага s = {s}. Не результат расчёта. {comment}"
    rows = ((i, j, i * h_x, j * h_y, float(w[i, j]), float(w_bar[i, j])) for i in range(w.shape[0]) for j in range(w.shape[1]))
    return cli._csv_text(("i", "j", "x [m]", "y [m]", "w [m]", "w_bar [1]"), rows, (comment,))


def _field_record(result: cn.CCCCN23Result, name: str | None) -> dict | None:
    if name is None:
        return None
    w = result.field()
    i, j = np.unravel_index(int(np.argmax(np.abs(w))), w.shape)
    h_x, h_y = result.a / result.n_x, result.b / result.n_y
    first = result.completed[-1]
    definition = "w(1) = G V^T" if first.s == 1 else "w(2) = X G2^T (X solved at s = 2, G2 = QR of V of s = 1)"
    return {
        "file": name, "role": "research" if name == cli.FIELD_RESEARCH else "diagnostic", "half_step": first.s,
        "shape": list(w.shape), "axes": "axis 0 = x (i = 0..n_x), axis 1 = y (j = 0..n_y), boundary nodes included", "definition": definition,
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
        "half_step_2": None if len(result.steps) < 2 else _second_record(result.steps[1]),
        "half_step_3": None if len(result.steps) < 3 else {"status": result.steps[2].status, "qr_executed": False, "conditions": result.steps[2].conditions},
    }


def _exact(value) -> float | None:
    """A rational of the certificate as the double nearest to it, for reading only; every decision was taken on the exact value."""
    return None if value is None else float(value)


def _second_record(step) -> dict:
    rn5, rt7 = step.rn5, step.rt7
    record = {"status": step.status, "done": step.done, "qr_executed": bool(step.qr_executed), "conditions": step.conditions,
              "reason": step.reason, "rn5": None, "rt7": None}
    if rn5 is not None:
        values = rn5["values"]
        record["rn5"] = {
            "verdict": rn5["verdict"], "certified": rn5["certified"], "reason": rn5["reason"], "hypotheses": rn5["hypotheses"],
            "values": {name: _exact(value) for name, value in values.items()},
            "strict_inequality": "σ_N(ŵ₁,w)² > ε² (Вейль), проверено точно",
            "rounding_note": "значения записаны как ближайшие double; решения приняты на точных рациональных значениях",
            "inputs": "сохранённые G и V полушага s = 1 этого расчёта как точные рациональные числа",
            "source": "РН-5: docs/R5_CCCC_Q4_PRE_D1_RANK_SCIENTIFIC_STUDY.md, разд. 4.6 (PR #49 @ 9377e18); вывод проекта, уровень Б",
        }
    if rt7 is not None:
        record["rt7"] = {
            "passed": rt7["passed"], "R7-1": rt7["R7-1"], "R7-2": rt7["R7-2"], "R7-3": rt7["R7-3"], "reason": rt7["reason"],
            "exact_ranks": rt7["ranks"], "rank_procedures": ["исключение в Fraction", f"исключение над GF({cn.certificate.MODULUS})"],
            "t": _exact(rt7["t"]), "tau": _exact(rt7["tau"]), "sin_theta_upper": _exact(rt7["sin_theta_upper"]),
            "inequalities": "R7-2: 0 ≤ t < 1; R7-3: τ < (1 − t)², точно; вывод: sin θ_max(span G2, U2*) ≤ √τ + t < 1",
            "scope": "только евклидова трансверсальность; не точность w(2), не обусловленность, не ранг w(2), не s ≥ 3 (D-065, 6)",
        }
    if step.done:
        record.update(exact_rank_X=step.V_exact_rank, sigma=list(step.sigma), rho_N=cli._number(step.rho_N), kappa_rep=cli._number(step.kappa_rep),
                      eta=cli._number(step.eta), cond_S=cli._number(step.cond_S), P_hat=cli._number(step.P_hat),
                      P_hat_previous_QR=cli._number(step.P_hat_previous_QR), field_preservation=cli._number(step.field_preservation),
                      role="OBSERVATION: величины s = 2 статус расчёта не меняют; пороги s = 1 не переносятся (D-065)")
    return record


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
                  "representation": "w = sum_i X_i(x) Y_i(y), N fixed", "orientation": cn.ORIENTATION,
                  "qr": "QR of the start set (D-057.1 item 2); QR of the solved V before s = 2 only after RN-5, then RT-7 (D-065)",
                  "half_step_2": "X-half-step on G2, roles of x and y exchanged (ВБC-13 p. 1); companion G R^T", "assembly": "C-3+ (D-054.3)",
                  "half_step_solver": "B-C, rank-2N correction (D-049.2)",
                  "discretization": "проект, уровень Б (D-049.1, D-050); авторам КК-МВИ не приписывается", "decision": "D-063, D-065"},
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
        "half_steps_executable": f"s = 1 (D-063.2); s = 2 при n_x, n_y ≤ {cn.S2_GRID_MAX} после RN-5 и RT-7 (D-065); s ≥ 3 — нет",
        "fixed_half_steps_only": "выполнены только фиксированные исследовательские полушаги; это не сходимость",
        "numerical_policy": {
            "eta_max_C": {"value": cn.ETA_MAX_C, "expression": "100u, u = 2^-53",
                          "role": "защитный порог только полушага s = 1 этого режима, не квалифицированное значение (D-063.3); на s = 2 не применяется (D-065)"},
            "recorded_only": "cond(S), cond(A) до и после QR, σ_i, ρ̂_N, κ_rep; на s = 2 также η и энергия",
            "certificates_s2": "RN-5 и RT-7 — строгие неравенства в точной арифметике без допусков (D-065, 5, 6)",
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
        "factors": None if not result.steps else {"file": cli.FACTORS, "roles": list(FACTOR_ROLES),
                                                  "note": "матрицы полушагов s = 1 и s = 2 поэлементно; значения через repr; при s = 2 G — это G2, V — это X"},
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
    cli._say(stream, f"Статус: {result.status}; выполнено полушагов {result.half_steps_completed} (исполняются не более s = 2), запрошено {args.half_steps}")
    cli._say(stream, document["status_note"])
    if result.reason:
        cli._say(stream, f"Причина: {result.reason}")
    field = document["field"]
    if field:
        w_max = field["w_max"]
        cli._say(stream, f"max|w| = {w_max['value']!r} м в узле ({w_max['i']}, {w_max['j']}), x = {w_max['x']!r}, y = {w_max['y']!r}; поле: {field['role']}")
        cli._say(stream, f"alpha_centre = {field['alpha_centre']!r}" if field["alpha_centre"] is not None else f"alpha_centre: {NOT_A_NODE}")
        for step in result.completed:
            cli._say(stream, f"s = {step.s}: η = {step.eta!r}; точный ранг решённого набора = {step.V_exact_rank}; ρ̂_N = {step.rho_N!r} (только запись)")
        if len(result.steps) > 1 and result.steps[1].rn5 is not None:
            second = result.steps[1]
            cli._say(stream, f"RN-5: {second.rn5['verdict']}" + (f"; RT-7: {'выполнен' if second.rt7['passed'] else 'не выполнен'}" if second.rt7 else ""))
    diagnostic = document["provenance"]["git_diagnostic"]
    if diagnostic:
        cli._say(stream, f"Версия: {diagnostic}")
    cli._say(stream, f"Результаты: {args.out.resolve()}")
    for name in paths:
        cli._say(stream, f"  {name}")
