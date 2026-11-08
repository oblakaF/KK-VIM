"""``python -m kkvim run-ssss``: SSSS, q = const, explicitly selected N = 1..6 (D-060). Internal routing, no method here.

    python -m kkvim run-ssss --N 1 --a 1 --b 1 --t 0.01 --E 2.1e11 --nu 0.3 --q 1000 --nx 40 --ny 40 --out scratch/n1
    python -m kkvim run-ssss --N 3 ... --half-steps 8 --out scratch/n3
    python -m kkvim run-ssss --N 6 --research-only ... --half-steps 8 --out scratch/n6

Routes, each unchanged and each with the status of its own decision:
- N = 1: the accepted R3 solver (D-043) through ``kkvim.cli``: the same computation, history.csv and field file as
  ``python -m kkvim run``, and its result.json with the common keys added;
- N = 2-3: ``kkvim.coupled_run.run_coupled`` in the research mode of D-048 (ACCEPTED_LIMITED_ENGINEERING, D-049.7; every result
  PROVISIONAL / NOT_QUALIFIED). Outside the area of D-048 the run is done, as by run_coupled itself, and is marked as outside;
- N = 4-6: ``kkvim.research_ssss``, only with --research-only and only inside the domain of the study (D-056.7; RESEARCH_ONLY).

N is never reduced and CONVERGED is never given to N >= 2 (D-048.4). Fields: field.csv only for CONVERGED of R3 (``kkvim.cli``);
field_research.csv for a research run that ended as requested; field_diagnostic.csv for any other run with a formed field.
Units are SI as in ``kkvim.cli``, except the energies of N = 4-6: they stay in the unit q^2 a^6 / D of the studied cell. history.csv
says so in a comment line, result.json per column; the column names are the same for every N >= 2.

Saving (``replace_results``) is all or nothing, also with --overwrite. A refused input (invalid, outside the domain, S-1), an ERROR_*
run and an interrupted run replace nothing: earlier result files stay as they are and the status and the reason go to stderr. If the
folder has no earlier results, such a run writes its result.json as usual. A run stopped by a rule of its route (MAX_HALF_STEPS,
STAGNATED, a guard, E1) is a valid replacement. result.json is written last and is absent while the data are replaced, also after a
killed process; such a folder is refused until it is dealt with. ``kkvim.cli.write_results`` of ``run`` is not used.

Exit codes: those of ``kkvim.cli``, and for N >= 2: 0 - the requested research run ended (PROVISIONAL_FIELD_STABILIZED,
RESEARCH_LIMIT_REACHED; not convergence); 1 - MAX_HALF_STEPS; 2 - also an input outside the domain, nothing computed, the reason
in result.json or, with earlier results kept, in stderr; 3 - ERROR_*; 6 - a guard, energy or proven rank stop; for any N: 130 - interrupted.
"""

import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

import kkvim.cli as cli
import kkvim.coupled_run as cr
from kkvim.coefficients import ERROR_INPUT
from kkvim.controller import CONVERGED
from kkvim.coupled import N_SOLVE
from kkvim.physical import PhysicalInputError
from kkvim.rectangle import validated_rectangle_input
from kkvim.rectangle_run import _versions
from kkvim.research_ssss import FAMILY_I, FAMILY_II, N_RESEARCH, S_STUDY, control_ratio, domain_refusal, run_research

SCHEMA = "kkvim.cli.ssss/1"
N_ALL = (1, *N_SOLVE, *N_RESEARCH)
BOUNDARY = "SSSS"
THEORY = "Kirchhoff-Love, linear; homogeneous isotropic plate; uniform q = const"
METHOD = ("КК-МВИ: w = Σ_j X_j(x) Y_j(y), N фиксировано; полушаги чередуются Y → X → Y; при N ≥ 2 все N функций полушага "
          "находятся совместно из одной связанной системы (D-004, D-044, D-045)")

MARK = cr.MARK
NOT_EXECUTED = "NOT_EXECUTED_OUTSIDE_DOMAIN"
EXIT_GUARD_STOP = 6
EXIT_INTERRUPTED = 130  # interrupted by the user (Ctrl+C); earlier results are kept
STAGE_PREFIX = ".kkvim-run-"  # the folder of one save inside --out
REPLACING_MARK = "replacing"  # written and synced in that folder before the first result file changes
REPLACING = ("COMPLETED", "STOPPED")  # executions whose files may replace earlier results
ENERGY = "energy"  # placeholder of COUPLED_COLUMNS: the unit depends on the route
ENERGY_PHYSICAL = "N·m"
ENERGY_NORMALIZED = "q^2 a^6 / D"

REQUESTED_END = (cr.PROVISIONAL_FIELD_STABILIZED, cr.RESEARCH_LIMIT_REACHED)
PROTECTIVE_STOPS = (cr.PROVISIONAL_RANK_GUARD_STOP, cr.PROVISIONAL_SET_GUARD_STOP, cr.PROVISIONAL_ENERGY_E1_STOP,
                    cr.STOPPED_RANK_LOSS_PROVEN)
STATUS_NOTES = {
    cr.PROVISIONAL_FIELD_STABILIZED: "Выполнен критерий стабилизации поля конечной дискретной задачи D-048.3. Это не сходимость к "
                                     "точному решению и не сертификат ранга N (D-048.4).",
    cr.RESEARCH_LIMIT_REACHED: "Выполнено заданное число полушагов. Это не сходимость.",
    cr.MAX_HALF_STEPS: "Достигнуто s_max = 100 (D-048.3). Поле последнего полушага — только диагностика.",
    cr.PROVISIONAL_RANK_GUARD_STOP: "Защита ранга: остановка до QR, N не уменьшено. Это не доказанная потеря ранга. Поле "
                                    "последнего полушага — только диагностика.",
    cr.PROVISIONAL_SET_GUARD_STOP: "Защита набора: остановка до QR, N не уменьшено. Это не доказанная потеря ранга. Поле "
                                   "последнего полушага — только диагностика.",
    cr.PROVISIONAL_ENERGY_E1_STOP: "Разрешённый энергетический тест нарушает E1 (исследовательская модель). Поле — только "
                                   "диагностика.",
    cr.STOPPED_RANK_LOSS_PROVEN: "S-1: доказанная зависимость набора. N не уменьшено; решения нет.",
    NOT_EXECUTED: "Расчёт не выполнялся: ввод вне разрешённой области. Значений решения нет.",
}
ERROR_NOTE = "Ошибочный выход решателя. Решения нет; поле последнего выполненного полушага, если есть, — только диагностика."

COMMON_LIMITS = (
    "Линейная прямоугольная пластина Кирхгофа–Лява, однородный изотропный материал, SSSS, q = const, 0,2 ≤ b/a ≤ 5.",
    "CCCC, другие закрепления и нагрузки, моменты и напряжения здесь не считаются.",
    "Успешный запуск, стабилизация итераций и научная приёмка — разные статусы (AGENTS.md, разд. 4).",
)
LIMITS_N23 = (
    "Реализация D-048 — ACCEPTED_LIMITED_ENGINEERING (D-049.7) в области D-047, D-048; статус сходимости не присваивается "
    "(D-048.4, D-056.8). Каждый результат — PROVISIONAL / NOT_QUALIFIED.",
    "NOT_QUALIFIED: c_d, c_rank, τ_set, c_DST, накопленная δ_f, константа QR, нормативный S-2, K_E; K_RESEARCH — временная "
    "защита (D-048.5, D-056.8).",
    "Сходимость при s → ∞, единственность предела и независимость от старта не утверждаются (D-056.8; Q-020, Q-022, Q-025).",
)
LIMITS_N46 = (
    "N = 4–6 — только исследование (RESEARCH_ONLY, D-056.7): не рабочий и не квалифицированный решатель. Каждый результат — "
    "PROVISIONAL / NOT_QUALIFIED.",
    "D-048 и K_RESEARCH на N = 4–6 не распространяются (D-056.7, п. 3): исследовательский контроллер применён без изменений как "
    "выбор исследования, числа полушагов фиксированы, стабилизация только регистрируется.",
    "Остановка защитой (например, N = 6, r = 1/5, 1/3) — не потеря ранга; продолжать за защитой нельзя.",
    "Считается ячейка исследования a = 1, b = r, q = D = 1; физические w получены точным множителем qa⁴/D "
    "(вывод проекта, kkvim.research_ssss). Энергии в history.csv — в единицах q²a⁶/D.",
    "Сходимость при фиксированном N не доказана; при s ≥ 2 утверждения — не выше PROVISIONAL (Q-022).",
)

COUPLED_COLUMNS = (
    ("s", "", "half-step"), ("k", "", "cycle s // 2"), ("direction", "", "computed set: Y (odd s) or X (even s)"),
    ("eta_initial", "", "eta_N before the refinement decision"), ("eta_N", "", "eta_N of the accepted half-step"),
    ("refined", "", "one refinement step done (D-048.2)"),
    ("rho_N", "", "sigma_N / sigma_1 of the field"), ("kappa_rep", "", "|U_w|_2 |V_w|_2 / |W_w|_2"),
    ("cond_U", "", "cond_2(U_w)"), ("cond_V", "", "cond_2(V_w)"), ("set_sigma_min", "", "sigma_min(R_V D_c)"),
    ("cond_A_given", "", "cond_2(A) of the given set"), ("cond_M_max", "", "max_k cond_2(M_k)"),
    ("w_max", "m", "max|w_s|"), ("w_bar_max", "", "max|w_s| D / (|q| a^4)"),
    ("eps_inf", "", "max|w_s - w_{s-1}| / max|w_s|"), ("eps_E", "", "|w_s - w_{s-1}|_a / |w_s|_a"),
    ("stabilization_ok", "", "eps_inf <= tol_inf and eps_E <= tol_E at this s"),
    ("energy_status", "", "energy test: NOT_APPLICABLE, UNRESOLVED, E1_E2_HOLD_PROVISIONAL, ..."),
    ("P", ENERGY, "Pi_h(w_s)"), ("delta_P", ENERGY, "Delta P_s"), ("Q", ENERGY, "Q_s"),
    ("gamma_s", ENERGY, "gamma_s"), ("tau", ENERGY, "tau_s^(N)"), ("m_s", "", "Q_s / (10 tau_s)"),
    ("guard_applicable", "", "K u kappa_rep < 1"), ("guard_rank_ok", "", "rho_N > K u kappa_rep"),
    ("guard_set_ok", "", "sigma_min(R_V D_c) > K u"), ("rank_threshold", "", "K u kappa_rep"), ("set_threshold", "", "K u"),
    ("qr_done", "", "the QR was done after the guards"), ("qr_field_error", "", "field change of the QR"),
    ("P_after_qr", ENERGY, "Pi_h from the transformed sets"), ("stop_reason", "", "status taken at this half-step"),
    ("mark", "", "mark of the record"),
)


class OptionError(Exception):
    """A command line that selects something this command does not offer; nothing is computed or written."""


def add_parser(commands) -> None:
    run = commands.add_parser(
        "run-ssss", help="выполнить расчёт SSSS при явно выбранном N = 1…6 и сохранить результат и его научный статус",
        description="Расчёт SSSS, q = const. N = 1 — принятый R3 (D-043); N = 2–3 — исследовательский режим D-048, "
                    "PROVISIONAL / NOT_QUALIFIED; N = 4–6 — только исследование (D-056.7), нужен --research-only. Единицы СИ.",
    )
    run.add_argument("--N", type=int, required=True, choices=N_ALL, metavar="{1..6}", help="число членов разложения; не уменьшается")
    run.add_argument("--bc", default=BOUNDARY, metavar="BC", help=f"закрепление; доступно только {BOUNDARY} (по умолчанию)")
    for name, unit, text in (("a", "m", "сторона вдоль x"), ("b", "m", "сторона вдоль y"), ("t", "m", "толщина"),
                             ("E", "Pa", "модуль упругости"), ("nu", "", "коэффициент Пуассона"),
                             ("q", "Pa", "равномерная нагрузка")):
        run.add_argument(f"--{name}", type=float, required=True, metavar="VALUE",
                         help=f"{text}" + (f", {unit}" if unit else ""))
    run.add_argument("--nx", type=int, required=True, metavar="N", help="число интервалов сетки по x")
    run.add_argument("--ny", type=int, required=True, metavar="N", help="число интервалов сетки по y")
    run.add_argument("--out", type=Path, required=True, metavar="DIR", help="папка результатов (создаётся)")
    run.add_argument("--overwrite", action="store_true",
                     help="разрешить замену файлов результатов, уже имеющихся в папке; отказ, ошибка расчёта или прерывание их не меняют")
    run.add_argument("--half-steps", type=int, metavar="S",
                     help=f"N = 2–3: не более S полушагов, S ≤ {cr.S_MAX} (по умолчанию {cr.S_MAX}, раньше — при стабилизации); "
                          f"N = 4–6: обязательно, ровно S ≤ {S_STUDY}; N = 1: не применяется")
    run.add_argument("--fixed-half-steps", action="store_true",
                     help="N = 2–3: стабилизация регистрируется и не останавливает (дополнение к D-048); нужен --half-steps")
    run.add_argument("--research-only", action="store_true",
                     help="обязательное подтверждение для N = 4–6: только исследование, не приёмка и не квалификация (D-056.7)")
    stopping = run.add_argument_group("параметры остановки R3, только N = 1 (NUMERICAL_SPEC.md 11.2)")
    stopping.add_argument("--tol-inf", type=float, metavar="VALUE", help="tol_∞")
    stopping.add_argument("--tol-E", type=float, metavar="VALUE", help="tol_E")
    stopping.add_argument("--s-min", type=int, metavar="N", help="s_min")
    stopping.add_argument("--s-max", type=int, metavar="N", help="s_max")


def check_options(args) -> None:
    """Refuse what this command does not offer, before anything is computed or created."""
    if args.bc != BOUNDARY:
        raise OptionError(f"закрепление {args.bc!r} недоступно: команда считает только {BOUNDARY}. CCCC доступен только как "
                          "RESEARCH_ONLY / PROVISIONAL / NOT_QUALIFIED: при N = 1 — команда run-cccc-n1 (D-061), при N = 2, 3 — команда "
                          "run-cccc-n23, полушаг s = 1 и при n_x, n_y ≤ 32 после сертификатов s = 2 (D-063, D-065); CCCC при N ≥ 4 недоступен. Другие закрепления вне принятых областей.")
    tolerances = [name for name, value in (("--tol-inf", args.tol_inf), ("--tol-E", args.tol_E), ("--s-min", args.s_min),
                                           ("--s-max", args.s_max)) if value is not None]
    if args.N == 1:
        applies = [name for name, given in (("--half-steps", args.half_steps is not None),
                                            ("--fixed-half-steps", args.fixed_half_steps),
                                            ("--research-only", args.research_only)) if given]
        if applies:
            raise OptionError(f"{', '.join(applies)}: не применяется при N = 1. Принятый R3 останавливается по своему правилу "
                              "(NUMERICAL_SPEC.md 11.3); предел числа полушагов задаётся --s-max.")
        return
    if tolerances:
        raise OptionError(f"{', '.join(tolerances)}: только для N = 1. При N ≥ 2 параметры остановки фиксированы D-048.3 и не "
                          "меняются.")
    if args.half_steps is not None and args.half_steps < 1:
        raise OptionError(f"--half-steps должно быть положительным, получено {args.half_steps}.")
    if args.N in N_SOLVE:
        if args.research_only:
            raise OptionError("--research-only относится к N = 4–6. При N = 2–3 всегда действует исследовательский режим D-048.")
        if args.fixed_half_steps and args.half_steps is None:
            raise OptionError("--fixed-half-steps требует явного --half-steps.")
        if args.half_steps is not None and args.half_steps > cr.S_MAX:
            raise OptionError(f"--half-steps = {args.half_steps} > s_max = {cr.S_MAX} (D-048.3).")
        return
    if not args.research_only:
        raise OptionError(f"N = {args.N} — только исследование, не приёмка и не квалификация (D-056.7). Подтвердите явно: "
                          "--research-only.")
    if args.half_steps is None:
        raise OptionError(f"при N = {args.N} обязательно --half-steps (фиксированное число полушагов, не более {S_STUDY}): "
                          "правила остановки для N = 4–6 не приняты.")


def _git(root: Path, *command: str) -> bytes:
    return subprocess.run(["git", *command], cwd=root, capture_output=True, timeout=30, check=True).stdout


def _git_failure(what: str, exc: Exception) -> str:
    return f"{what}: {type(exc).__name__}: " + str(exc).encode("utf-8", "replace").decode("utf-8")


def software_provenance() -> dict:
    """Versions, and the commit if the package runs from a source checkout (an installed copy has no commit of its own).

    Git output stays bytes (file names in it can have any encoding). No failure of git costs a calculation: what is not known is
    null and ``git_diagnostic`` says why.
    """
    source = Path(__file__).resolve()
    root = source.parents[2]
    commit = dirty = diagnostic = None
    if source.parents[1].name != "src" or not (root / "pyproject.toml").is_file():
        diagnostic = "коммит неизвестен: пакет запущен не из рабочей копии исходного кода"
    else:
        try:
            commit = _git(root, "rev-parse", "HEAD").decode("ascii").strip()
            if not re.fullmatch(r"[0-9a-f]{40}(?:[0-9a-f]{24})?", commit):
                raise ValueError("вывод git rev-parse не является SHA")
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            commit, diagnostic = None, _git_failure("версия Git недоступна", exc)
        else:
            try:
                dirty = bool(_git(root, "status", "--porcelain").strip())
            except (OSError, subprocess.SubprocessError) as exc:
                diagnostic = _git_failure("состояние рабочей копии неизвестно", exc)
    return {"versions": _versions(), "git_commit": commit, "git_worktree_modified": dirty, "git_diagnostic": diagnostic,
            "module": "kkvim.cli_ssss"}


def _route(N: int) -> dict:
    if N == 1:
        return {"mode": "R3_ACCEPTED_SOLVER", "route": "kkvim.rectangle_run.run_rectangle_physical",
                "decisions": ["D-043"], "research": False}
    if N in N_SOLVE:
        return {"mode": "R4_LIMITED_RESEARCH_MODE", "route": "kkvim.coupled_run.run_coupled(research_mode=True, transform='fft')",
                "decisions": ["D-047", "D-048", "D-049.7", "D-056.8", "D-060"], "research": True}
    return {"mode": "RESEARCH_ONLY_N4_N6", "route": "kkvim.research_ssss.run_research", "decisions": ["D-056.7", "D-060"],
            "research": True}


def _common(args, status: str, note: str, execution: str, half_steps: int, reason=None) -> dict:
    return {
        "schema": SCHEMA,
        "N": args.N,
        "boundary_conditions": BOUNDARY,
        "theory": THEORY,
        "method": METHOD,
        "solver_path": _route(args.N),
        "provenance": software_provenance(),
        "execution": execution,
        "status": status,
        "status_note": note,
        "reason": reason,
        "half_steps_completed": half_steps,
    }


# --- N = 1: the R3 path of kkvim.cli ---------------------------------------------------------------------------------------

def execute_r3(args, argv) -> tuple[str, dict, int]:
    result, report = cli.solve(args)
    texts = cli.result_texts(args, argv, result, report)
    document = json.loads(texts[cli.RESULT])
    s = 0 if result.run is None else result.run.s
    execution = "COMPLETED" if result.status == CONVERGED else ("ERROR" if result.status.startswith("ERROR") else "STOPPED")
    merged = {**document, **_common(args, result.status, document["status_note"], execution, s, report.error)}
    merged["program"]["argv"] = list(argv)
    merged["scientific_applicability"] = {
        "accepted_solver": True, "accepted_by": "D-043", "research_only": False, "mark": None,
        "domain": "applicability.limits; отдельный расчёт — не научная приёмка", "limits": list(cli.APPLICABILITY),
    }
    texts[cli.RESULT] = json.dumps(merged, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    return result.status, texts, cli.exit_code(result.status)


# --- N = 2-6: research routes ------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class _Run:
    """A completed call of a research route; ``w_scale`` turns the values of a run in cell units into metres."""

    run: cr.CoupledRun
    D: float
    w_scale: float  # g = q a^4 / D
    cell_units: bool  # the run is the cell a = 1, b = r, q = D = 1 of the study


def d048_area(a: float, b: float, n_x: int, n_y: int) -> dict:
    """Facts about the area of D-048 for N = 2-3 (D-048.1, D-048.3, D-048.9); no verdict about the run."""
    r = control_ratio(a, b)
    family_i = n_x == n_y and n_x in cr.MANDATORY_S2_FAMILY_I + (40,)
    family_ii = r is not None and FAMILY_II.get(r) == (n_x, n_y)
    inside = r is not None and (family_i or family_ii)
    return {
        "control_r": None if r is None else str(r),
        "grid_level_of_d048": bool(family_i or family_ii),
        "inside_area": inside,
        "mandatory_s2_grid": cr.is_mandatory_s2_grid(a, b, n_x, n_y),
        "note": "Область D-048: десять контрольных r и проверенные уровни сеток; она не распространяется на непрерывный "
                "интервал r и произвольные сетки (D-048.9). Вне области расчёт run_coupled выполняется как диагностический.",
    }


def study_domain(a: float, b: float, n_x: int, n_y: int) -> dict:
    r = control_ratio(a, b)
    return {
        "control_r": None if r is None else str(r),
        "study_matrix": f"десять r; n_x = n_y из {FAMILY_I}; шесть сеток семейства II; s ≤ {S_STUDY}",
        "note": "Вне этой матрицы команда отказывает: исследованная область не расширяется молча.",
    }


def _applicability(args, rectangle) -> dict:
    if args.N in N_SOLVE:
        return {"accepted_solver": False, "implementation": "ACCEPTED_LIMITED_ENGINEERING (D-049.7) в области D-048",
                "qualified": False, "research_only": False, "mark": MARK,
                "domain": d048_area(rectangle.a, rectangle.b, rectangle.n_x, rectangle.n_y),
                "limits": [*LIMITS_N23, *COMMON_LIMITS]}
    return {"accepted_solver": False, "qualified": False, "research_only": True, "status": "RESEARCH_ONLY (D-056.7)", "mark": MARK,
            "domain": study_domain(rectangle.a, rectangle.b, rectangle.n_x, rectangle.n_y),
            "limits": [*LIMITS_N46, *COMMON_LIMITS]}


def _input_record(args, half_steps, fixed) -> dict:
    return {
        **{name: {"value": cli._number(getattr(args, name)), "unit": unit} for name, unit in cli.INPUT_UNITS},
        "n_x": args.nx, "n_y": args.ny, "half_steps_requested": half_steps, "fixed_half_steps": fixed,
    }


def history_rows(call: _Run):
    for log in call.run.steps:
        en, rk, gd = log.energy, log.rank, log.guards
        if log.w_max is None:
            w_max = w_bar_max = None
        elif call.cell_units:
            w_max, w_bar_max = log.w_max * abs(call.w_scale), log.w_max
        else:
            w_max, w_bar_max = log.w_max, log.w_max / abs(call.w_scale)
        yield {
            "s": log.s, "k": log.k, "direction": log.direction, "eta_initial": log.eta_initial, "eta_N": log.eta_N,
            "refined": log.refined, "rho_N": rk.rho_N, "kappa_rep": rk.kappa_rep, "cond_U": rk.cond_U, "cond_V": rk.cond_V,
            "set_sigma_min": rk.set_sigma_min, "cond_A_given": log.cond_A_given, "cond_M_max": log.cond_M_max,
            "w_max": w_max, "w_bar_max": w_bar_max, "eps_inf": log.eps_inf, "eps_E": log.eps_E,
            "stabilization_ok": log.stabilization_ok, "energy_status": None if en is None else en.status,
            "P": None if en is None else en.P, "delta_P": None if en is None else en.delta_P, "Q": None if en is None else en.Q,
            "gamma_s": None if en is None else en.gamma, "tau": None if en is None else en.tau, "m_s": log.m_s,
            "guard_applicable": None if gd is None else gd.applicable, "guard_rank_ok": None if gd is None else gd.rank_ok,
            "guard_set_ok": None if gd is None else gd.set_ok, "rank_threshold": None if gd is None else gd.rank_threshold,
            "set_threshold": None if gd is None else gd.set_threshold, "qr_done": log.qr_done,
            "qr_field_error": log.qr_field_error, "P_after_qr": log.P_after_qr, "stop_reason": log.stop_reason, "mark": log.mark,
        }


def history_csv(call: _Run | None, comments=()) -> str:
    header = [name for name, _, _ in COUPLED_COLUMNS]
    rows = () if call is None else ([row[name] for name in header] for row in history_rows(call))
    return cli._csv_text(header, rows, comments)


def _field_arrays(call: _Run, a: float):
    """(x, y, w, w_bar) of the field of the run in metres and in w_bar = w D / (q a^4)."""
    run = call.run
    field = run.field()
    if call.cell_units:
        return a * run.grid_x.nodes, a * run.grid_y.nodes, field * call.w_scale, field
    return run.grid_x.nodes, run.grid_y.nodes, field, field / call.w_scale


def _research_field_csv(x, y, w, w_bar, comments) -> str:
    rows = ((i, j, x[i], y[j], float(w[i, j]), float(w_bar[i, j])) for i in range(w.shape[0]) for j in range(w.shape[1]))
    return cli._csv_text(("i", "j", "x [m]", "y [m]", "w [m]", "w_bar [1]"), rows, comments)


def _research_field(call: _Run, a: float, name: str) -> tuple[str, dict]:
    x, y, w, w_bar = _field_arrays(call, a)
    run = call.run
    if name == cli.FIELD_DIAGNOSTIC:
        comments = (f"DIAGNOSTIC: статус {run.status}; поле последнего выполненного полушага s = {run.s}. Это не рассчитанное "
                    f"решение. {MARK}",)
        role, note = "diagnostic", f"Поле последнего выполненного полушага s = {run.s} при статусе {run.status}; не решение."
    else:
        comments = (f"RESEARCH: {MARK}; N = {run.parameters.N}, статус {run.status}, s = {run.s}. Не принятое и не "
                    "квалифицированное решение.",)
        role, note = "research", f"Исследовательское поле полушага s = {run.s}: {MARK}."
    i, j = (int(v) for v in np.unravel_index(int(np.argmax(np.abs(w))), w.shape))
    n_x, n_y = w.shape[0] - 1, w.shape[1] - 1
    centre = float(w_bar[n_x // 2, n_y // 2]) if n_x % 2 == 0 and n_y % 2 == 0 else None
    record = {
        "file": name, "role": role, "note": note, "half_step": run.s, "shape": list(w.shape),
        "axes": "axis 0 = x (i = 0..n_x), axis 1 = y (j = 0..n_y), boundary nodes included",
        "units": {"x": "m", "y": "m", "w": "m", "w_bar": "1"}, "w_bar_definition": cli.W_BAR,
        "w_max": {"value": cli._number(float(abs(w[i, j]))), "i": i, "j": j, "x": float(x[i]), "y": float(y[j]), "unit": "m"},
        "w_bar_centre": cli._number(centre),
    }
    return _research_field_csv(x, y, w, w_bar, comments), record


def _energy_comment(cell_units: bool) -> str:
    names = ", ".join(n for n, u, _ in COUPLED_COLUMNS if u == ENERGY)
    if cell_units:
        return (f"Энергии {names}: НОРМИРОВАННЫЕ, в единицах {ENERGY_NORMALIZED} (ячейка a = 1, b = r, q = D = 1); в N·m — умножить на "
                "energy_scale_q2_a6_over_D из result.json.")
    return f"Энергии {names}: ФИЗИЧЕСКИЕ, в {ENERGY_PHYSICAL}."


def _history_record(rows: int, cell_units: bool, comments) -> dict:
    unit = ENERGY_NORMALIZED if cell_units else ENERGY_PHYSICAL
    return {"file": cli.HISTORY, "rows": rows, "comments": list(comments),
            "energy_kind": "normalized" if cell_units else "physical", "energy_unit": unit,
            "energy_note": _energy_comment(cell_units),
            "columns": [{"name": n, "unit": unit if u == ENERGY else u, "meaning": m} for n, u, m in COUPLED_COLUMNS]}


def _exit_code(status: str) -> int:
    if status in REQUESTED_END:
        return cli.EXIT_CONVERGED
    if status == cr.MAX_HALF_STEPS:
        return cli.EXIT_NOT_CONVERGED
    if status in PROTECTIVE_STOPS:
        return EXIT_GUARD_STOP
    if status == NOT_EXECUTED:
        return cli.EXIT_USAGE
    return cli.EXIT_SOLVER_ERROR


def _not_computed(args, argv, status, reason, D, half_steps, fixed, applicability) -> tuple[str, dict, int]:
    """result.json with the reason and history.csv without rows: no value of a solution exists, none is written."""
    comments = (f"Полушагов нет: {status}.", _energy_comment(args.N in N_RESEARCH))
    document = {
        **_common(args, status, STATUS_NOTES.get(status, ERROR_NOTE), "NOT_EXECUTED", 0, reason),
        "program": _program(argv), "input": _input_record(args, half_steps, fixed),
        "D": None if D is None else {"value": cli._number(D), "unit": "N·m"},
        "scientific_applicability": applicability, "field": None,
        "history": _history_record(0, args.N in N_RESEARCH, comments), "numerical_parameters": None,
    }
    texts = {cli.HISTORY: history_csv(None, comments=comments),
             cli.RESULT: json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n"}
    return status, texts, _exit_code(status) if status != ERROR_INPUT else cli.EXIT_SOLVER_ERROR


def _program(argv) -> dict:
    return {"command": "python -m kkvim " + " ".join(argv), "argv": list(argv)}


def _limits_before_validation(args) -> dict:
    mark = {"mark": MARK, "accepted_solver": False, "qualified": False, "research_only": args.N in N_RESEARCH}
    return {**mark, "domain": None, "limits": [*(LIMITS_N46 if args.N in N_RESEARCH else LIMITS_N23), *COMMON_LIMITS]}


def execute_research(args, argv) -> tuple[str, dict, int]:
    fixed = args.N in N_RESEARCH or args.fixed_half_steps
    half_steps = cr.S_MAX if args.half_steps is None else args.half_steps
    try:
        rectangle, D = validated_rectangle_input(args.a, args.b, args.t, args.E, args.nu, args.q, args.nx, args.ny)
        try:
            w_scale = rectangle.q * rectangle.a**4 / D
        except OverflowError:
            w_scale = math.inf
        if not math.isfinite(w_scale) or w_scale == 0.0:
            raise PhysicalInputError(f"the scale q a^4 / D = {w_scale!r} is not finite and nonzero")
    except PhysicalInputError as exc:
        return _not_computed(args, argv, ERROR_INPUT, str(exc), None, half_steps, fixed, _limits_before_validation(args))
    applicability = _applicability(args, rectangle)
    if args.N in N_RESEARCH:
        refusal = domain_refusal(rectangle.a, rectangle.b, rectangle.n_x, rectangle.n_y, args.N, half_steps)
        if refusal is not None:
            return _not_computed(args, argv, NOT_EXECUTED, refusal, D, half_steps, fixed, applicability)
        control = control_ratio(rectangle.a, rectangle.b)
        r = float(control) if control is not None else rectangle.b / rectangle.a  # None only for the S-1 exit
        run = run_research(r, rectangle.n_x, rectangle.n_y, args.N, half_steps=half_steps)
        call = _Run(run, D, w_scale, True)
    else:
        run = cr.run_coupled(rectangle.a, rectangle.b, D, rectangle.q, rectangle.n_x, rectangle.n_y, args.N, transform="fft",
                             half_steps=half_steps, research_mode=True, stop_on_stabilization=not fixed)
        call = _Run(run, D, w_scale, False)
    status = run.status
    if run.parameters is None or (status == ERROR_INPUT and not run.steps):
        return _not_computed(args, argv, status, run.reason, D, half_steps, fixed, applicability)
    comments = (f"{MARK}; N = {args.N}; {_route(args.N)['mode']}; статус {status}.", _energy_comment(call.cell_units))
    texts, field = {}, None
    if run.x is not None:
        name = cli.FIELD_RESEARCH if status in REQUESTED_END else cli.FIELD_DIAGNOSTIC
        texts[name], field = _research_field(call, rectangle.a, name)
    texts[cli.HISTORY] = history_csv(call, comments)
    if status in REQUESTED_END:
        execution = "COMPLETED"
    elif status == cr.MAX_HALF_STEPS or status in PROTECTIVE_STOPS:
        execution = "STOPPED"
    else:
        execution = "ERROR"
    parameters = asdict(run.parameters)
    if call.cell_units:
        parameters["cell"] = "a, b, D, q — значения ячейки исследования a = 1, b = r, q = D = 1"
    document = {
        **_common(args, status, STATUS_NOTES.get(status, ERROR_NOTE), execution, run.s, run.reason),
        "program": _program(argv), "input": _input_record(args, half_steps, fixed),
        "D": {"value": cli._number(D), "unit": "N·m"},
        "w_scale_q_a4_over_D": {"value": cli._number(w_scale), "unit": "m",
                                "use": "w = w_bar * value" + ("; поле считалось в ячейке" if call.cell_units else "")},
        **({"energy_scale_q2_a6_over_D": {"value": cli._number(w_scale * rectangle.q * rectangle.a * rectangle.a),
                                          "unit": ENERGY_PHYSICAL, "use": "energy [N·m] = energy of history.csv * value"}}
           if call.cell_units else {}),
        "cycles_k": run.s // 2,
        "converged": False,
        "first_stabilization_s": run.first_stabilization_s,
        "energy_s2": {"checked": run.energy_s2_checked, "mandatory_d048": run.energy_s2_mandatory,
                      "status": run.energy_s2_status, "m_2": cli._number(run.energy_s2_m),
                      "mandatory_unresolved": run.energy_s2_mandatory_unresolved, "reason": run.energy_s2_reason},
        "numerical_parameters": {k: cli._number(v) for k, v in parameters.items()},
        "scientific_applicability": applicability,
        "field": field,
        "history": _history_record(len(run.steps), call.cell_units, comments),
    }
    texts[cli.RESULT] = json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    return status, texts, _exit_code(status)


# --- saving --------------------------------------------------------------------------------------------------------------------

def check_no_interrupted_save(out: Path) -> None:
    """Refuse a folder with an unfinished replacement (a killed process, a failed restoration); it is never recovered silently.

    A save folder without the mark changed no result file (for example a file in it was held by another program): it is removed.
    """
    stages = sorted(out.glob(STAGE_PREFIX + "*")) if out.is_dir() else []
    unfinished = [stage.name for stage in stages if (stage / REPLACING_MARK).exists()]
    if unfinished:
        raise cli.OutputExistsError(f"в {out} осталась незавершённая замена результатов: {', '.join(unfinished)}. Файлы результатов в этой "
                                    "папке нельзя считать результатом завершённого расчёта. Чтобы вернуть прежний результат, удалите из "
                                    f"папки {', '.join(cli.OUTPUTS)}, перенесите в неё файлы из old этого каталога и удалите каталог. "
                                    "Чтобы начать заново, удалите те же файлы и каталог.")
    for stage in stages:
        shutil.rmtree(stage, ignore_errors=True)


def _write_file(path: Path, text: str) -> None:
    with open(path, "x", encoding="utf-8", newline="") as file:
        file.write(text)
        file.flush()
        os.fsync(file.fileno())


def _sync_folder(path: Path) -> None:
    """Make the changes of names in a folder durable before the next step, where a folder can be synced (not on Windows)."""
    if os.name == "nt":
        return
    try:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError:  # not every file system syncs a folder; the order of a killed process does not depend on it
        pass


def _keep(source: Path, copy: Path) -> None:
    """Keep the original under another name; the result file itself stays in place until it is replaced."""
    try:
        os.link(source, copy)
    except OSError:
        shutil.copyfile(source, copy)
        with open(copy, "r+b") as file:
            os.fsync(file.fileno())
        shutil.copystat(source, copy)


def _change(changed: list, name: str, action, *paths) -> None:
    """Do one change of a result file, its name recorded once and before it: an interrupt right after the change must not lose it."""
    recorded = name in changed
    if not recorded:
        changed.append(name)
    try:
        action(*paths)
    except OSError:  # this action changed nothing
        if not recorded:
            changed.remove(name)
        raise


def _undo(out: Path, old: Path, name: str) -> bool:
    """Put one name back through a second name of the original: ``old`` stays complete until the end, also for a killed restoration."""
    try:
        if (old / name).exists():
            spare = old.parent / name
            _keep(old / name, spare)
            os.replace(spare, out / name)
        else:
            (out / name).unlink(missing_ok=True)
    except (OSError, KeyboardInterrupt):
        return False
    return True


def _restore(out: Path, old: Path, changed) -> list:
    """Undo the changed names; the names that could not be undone.

    result.json goes first and comes back last, and only when all the data are back, so that a killed or failed restoration leaves no
    result.json beside data of another run. If it cannot be removed, the data stay as they are: nothing is undone.
    """
    if cli.RESULT in changed:
        try:
            (out / cli.RESULT).unlink(missing_ok=True)
        except (OSError, KeyboardInterrupt):
            return list(changed)
        _sync_folder(out)
    failed = [name for name in reversed(changed) if name != cli.RESULT and not _undo(out, old, name)]
    if cli.RESULT in changed and (old / cli.RESULT).exists():
        _sync_folder(out)
        if failed or not _undo(out, old, cli.RESULT):
            failed.append(cli.RESULT)
    return failed


def _finish_stage(stage: Path) -> bool:
    """The result files are consistent again: drop the mark and the folder; False if something of the folder remains."""
    try:
        (stage / REPLACING_MARK).unlink(missing_ok=True)
        shutil.rmtree(stage, ignore_errors=True)
    except (OSError, KeyboardInterrupt):  # the outcome is already decided; the next run looks at what is left
        pass
    return not stage.exists()


def replace_results(out: Path, texts: dict) -> dict:
    """All files of one run in place of the earlier ones, or no change at all; on a failure raise ``OutputWriteError``.

    1. The new files are written completely (and synced) in a new folder ``out/.kkvim-run-*/new``.
    2. Every earlier result file gets a second name in ``old`` (a hard link, or a copy); it stays in place meanwhile.
    3. The mark ``replacing`` is written and synced: from here on the folder is an unfinished transaction.
    4. The earlier result.json is removed, so that it does not vouch for what follows.
    5. The other new files are moved into place, stale earlier files are removed.
    6. The new result.json is moved into place, last: only now does the folder hold a result, and it is complete.
    7. The mark and the folder of the save are removed.

    result.json next to the data without the mark therefore always means one complete run. Any failure or an interrupt in 1-6 restores
    the names changed so far (``_restore``). If that fails too, the error names the files, and the mark and ``old`` are kept. A process
    killed in 4-7 leaves the mark and the originals, and the next run refuses until it is dealt with (``check_no_interrupted_save``).
    Two runs saving into one folder at the same moment are not guarded against each other.
    """
    current = [name for name in cli.OUTPUTS if (out / name).exists()]
    stage, changed, step = None, [], ("создание временного каталога в", str(out))
    try:
        stage = Path(tempfile.mkdtemp(dir=out, prefix=STAGE_PREFIX))
        new, old = stage / "new", stage / "old"
        new.mkdir()
        old.mkdir()
        for name, text in texts.items():
            step = ("запись временного файла для", name)
            _write_file(new / name, text)
        for name in current:
            step = ("сохранение копии прежнего файла", name)
            _keep(out / name, old / name)
        step = ("начало замены в", str(out))
        _sync_folder(new)
        _sync_folder(old)
        _write_file(stage / REPLACING_MARK, f"{', '.join(texts)}\n")
        _sync_folder(stage)
        _sync_folder(out)
        if cli.RESULT in current:
            step = ("снятие прежнего файла", cli.RESULT)
            _change(changed, cli.RESULT, os.remove, out / cli.RESULT)
            _sync_folder(out)
        for name in texts:
            if name != cli.RESULT:
                step = ("замена", name)
                _change(changed, name, os.replace, new / name, out / name)
        for name in current:
            if name not in texts:
                step = ("удаление устаревшего файла", name)
                _change(changed, name, os.remove, out / name)
        _sync_folder(out)
        step = ("замена", cli.RESULT)
        _change(changed, cli.RESULT, os.replace, new / cli.RESULT, out / cli.RESULT)
        _sync_folder(out)
    except BaseException as exc:
        action, name = step
        failed = [] if stage is None else _restore(out, stage / "old", changed)
        if failed:
            raise cli.OutputWriteError(action, name, exc if isinstance(exc, OSError) else OSError(repr(exc)), (),
                                       [item for item in failed if (out / item).exists()],
                                       [item for item in failed if not (out / item).exists()], (),
                                       [f"{stage.name}/old (здесь прежние файлы результатов)"]) from exc
        left = [] if stage is None or _finish_stage(stage) else [stage.name]
        if isinstance(exc, OSError):
            raise cli.OutputWriteError(action, name, exc, (), (), (), (), left) from exc
        raise
    if not _finish_stage(stage):
        cli._say(sys.stderr, f"Результаты записаны, но не удалось удалить временный каталог {out / stage.name}; следующий запуск удалит его "
                             "или сообщит, что делать.")
    return {name: out / name for name in texts}


def _kept(out: Path) -> list:
    return [name for name in cli.OUTPUTS if (out / name).exists()]


def _say_kept(out: Path) -> None:
    kept = _kept(out)
    if kept:
        cli._say(sys.stderr, f"Прежние файлы результатов в {out} не изменены: {', '.join(kept)}.")


def _interrupted(out: Path) -> int:
    cli._say(sys.stderr, "Расчёт прерван пользователем; результаты этого расчёта не сохранены.")
    _say_kept(out)
    return EXIT_INTERRUPTED


# --- the command ---------------------------------------------------------------------------------------------------------------

def main(args, argv) -> int:
    try:
        check_options(args)
    except OptionError as exc:
        cli._say(sys.stderr, f"Расчёт не выполнен: {exc}")
        return cli.EXIT_USAGE
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
        status, texts, code = execute_r3(args, argv) if args.N == 1 else execute_research(args, argv)
        document = json.loads(texts[cli.RESULT])
    except KeyboardInterrupt:
        return _interrupted(args.out)
    except Exception:  # noqa: BLE001 - reported as an internal error, never as a status of the solver
        code = cli._internal_error("расчёт или подготовка результатов")
        _say_kept(args.out)
        return code
    if (document["execution"] not in REPLACING or document["half_steps_completed"] == 0) and _kept(args.out):
        cli._say(sys.stderr, f"Результат не заменён: статус {status}. {document['status_note']}")
        if document["reason"]:
            cli._say(sys.stderr, f"Причина: {document['reason']}")
        _say_kept(args.out)
        cli._say(sys.stderr, "Запись этого расчёта не сохранена; чтобы получить её, укажите новую папку.")
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
        for line in cli.output_error_lines(args.out, exc, status):
            cli._say(sys.stderr, line)
        return cli.EXIT_OUTPUT_ERROR
    except Exception:  # noqa: BLE001
        code = cli._internal_error("запись результатов")
        _say_kept(args.out)
        return code
    summary(args, status, texts, paths)
    return code


def summary(args, status, texts, paths) -> None:
    document = json.loads(texts[cli.RESULT])
    route = document["solver_path"]
    stream = sys.stdout
    cli._say(stream, f"SSSS, N = {args.N}: {route['mode']} ({route['route']})")
    cli._say(stream, f"Статус: {status}; полушагов s = {document['half_steps_completed']}")
    cli._say(stream, document["status_note"])
    if document["reason"]:
        cli._say(stream, f"Причина: {document['reason']}")
    mark = document["scientific_applicability"]["mark"]
    if mark:
        cli._say(stream, f"Научный статус: {mark}")
    if args.N in N_RESEARCH:
        cli._say(stream, "N = 4–6 — только исследование (RESEARCH_ONLY, D-056.7).")
    field = document.get("field")
    if field and "w_max" in field:
        w_max = field["w_max"]
        cli._say(stream, f"max|w| = {w_max['value']!r} м в узле ({w_max['i']}, {w_max['j']}), x = {w_max['x']!r}, "
                         f"y = {w_max['y']!r}; поле: {field['role']}")
    area = document["scientific_applicability"]["domain"]
    if isinstance(area, dict) and area.get("inside_area") is False:
        cli._say(stream, "Вне области D-048 (D-048.9): расчёт диагностический.")
    energy_s2 = document.get("energy_s2")
    if energy_s2 and energy_s2["mandatory_unresolved"]:
        cli._say(stream, f"ВНИМАНИЕ: обязательная проверка энергии при s = 2 не разрешена (UNRESOLVED, D-048.1): "
                         f"{energy_s2['reason']}. Статус расчёта от этого не меняется.")
    if document["history"].get("energy_note"):
        cli._say(stream, document["history"]["energy_note"])
    diagnostic = document["provenance"]["git_diagnostic"]
    if diagnostic:
        cli._say(stream, f"Версия: {diagnostic}")
    cli._say(stream, f"Результаты: {args.out.resolve()}")
    for name in paths:
        cli._say(stream, f"  {name}")
    if args.N == 1:  # the line of the run command; the status stays that of the solver
        conditions = document["applicability"]["necessary_grid_conditions_6_1"]
        if not all(value for key, value in conditions.items() if key != "note"):
            cli._say(stream, f"Сетка {args.nx}×{args.ny} не удовлетворяет необходимым условиям квалификационной сетки "
                             "R3_NUMERICAL_SPEC.md 6.1 (min ≥ 20, max ≤ 640, кратность 4): расчёт диагностический.")
