"""Adapter between the GUI and the scientific package kkvim (GUI_DESIGN.md, 3.4). Qt-free.

This is the only module of ``kkvim_gui`` that imports ``kkvim`` (tests/test_gui_independence.py).
It calls the existing solvers unchanged and copies their results; no formula, solver, numerical
policy, stopping rule, guard or certificate is implemented or changed here. Routes (``request.route_of``):

- SSSS, N = 1: ``kkvim.rectangle_run.run_rectangle_physical`` (R3, ACCEPTED, D-043) and its
  ``halfstep_report`` (NUMERICAL_SPEC.md 11.5);
- SSSS, N = 2-3: ``kkvim.rectangle.validated_rectangle_input`` (the accepted input checks and
  D = E t^3 / (12 (1 - nu^2)) of ``kkvim.physical``), then ``kkvim.coupled_run.run_coupled``
  (R4; D-047, D-048 and its addendum). Every R4 result carries PROVISIONAL / NOT_QUALIFIED;
- SSSS, N = 4-6: only with the explicit research choice and only inside the studied domain
  (``kkvim.research_ssss.domain_refusal``): ``kkvim.research_ssss.run_research`` on the cell of the study,
  the output in metres by the existing route of ``python -m kkvim run-ssss`` (``kkvim.cli_ssss._field_arrays``,
  w = g w_bar, g = q a^4 / D; D-056.7, D-060). RESEARCH_ONLY / PROVISIONAL / NOT_QUALIFIED;
- CCCC, N = 1: ``kkvim.cccc_n1.run_cccc_n1_physical``, fixed half-steps s = 1...8 (D-061);
- CCCC, N = 2-3: ``kkvim.cccc_n23.run_cccc_n23_physical``, s = 1 and s = 2 under RN-5 and RT-7 (D-063, D-065);
- CCCC, N = 4-6: ``kkvim.cccc_n46.run_cccc_n46_physical``, s = 1 and s = 2 under RN-5 and RT-7 (D-067, D-068).
  For CCCC the GUI offers s = 1, 2 only (N >= 2); s = 2 with a grid above 32 is passed to the solver, whose own
  guard stops before RN-5 and QR. Every CCCC result is RESEARCH_ONLY / PROVISIONAL / NOT_QUALIFIED.

``field_role`` (FIELD_ROLE_RULE) is the one mapping from the solver status to the role of the field: the requested
end of the route (REQUESTED_END) gives "accepted" for R3 CONVERGED and "research" otherwise; any other status with a
field gives "diagnostic"; no field gives None. Domain checks before a run are the solvers' own functions.

The input arrives in the base SI units m and Pa (``units``); D is in N·m and w in m.
w_bar = w D / (q a^4) with the reference side a (MATH_SPEC 2.5) is taken from
``RectangleResult.dimensionless_field`` for N = 1, formed by the same expression for SSSS N = 2-3,
taken from the scaled research route for SSSS N = 4-6 and from ``dimensionless_field`` of the CCCC results.
"""

import math
import platform
from importlib import metadata

from kkvim import (
    cccc_n1,
    cccc_n23,
    cccc_n46,
    cli_cccc,
    cli_cccc_n23,
    cli_cccc_n46,
    cli_ssss,
    controller,
    coupled_run,
    research_ssss,
)
from kkvim.coefficients import ERROR_INPUT
from kkvim.grid import N_MIN
from kkvim.physical import PhysicalInputError
from kkvim.r3_input import R_MAX
from kkvim.rectangle import validated_rectangle_input
from kkvim.rectangle_run import halfstep_report, run_rectangle_physical
from kkvim_gui.backend.outcome import (
    ORIGIN_SOLVER,
    ROLE_ACCEPTED,
    ROLE_DIAGNOSTIC,
    ROLE_RESEARCH,
    Column,
    RunOutcome,
    frozen_copy,
)
from kkvim_gui.backend.request import (
    CCCC_HALF_STEPS,
    CCCC_ROUTES,
    N_ALL,
    R4_MODES,
    R4_NORMATIVE,
    R4_RESEARCH_FIXED,
    ROUTE_CCCC_N1,
    ROUTE_CCCC_N23,
    ROUTE_CCCC_N46,
    ROUTE_R3,
    ROUTE_R4,
    ROUTE_SSSS_N46,
    SUPPORTS,
    RunRequest,
)

MARK = coupled_run.MARK  # "PROVISIONAL / NOT_QUALIFIED"
RESEARCH_MARK = cccc_n1.APPLICATION_STATUS  # "RESEARCH_ONLY / PROVISIONAL / NOT_QUALIFIED"
N_AVAILABLE = N_ALL  # every N = 1-6 has a route; what it computes is the route's own domain
DENSE_N_MAX = 40  # D-048.6: dense is admissible only up to max(n_x, n_y) = 40; fft is the main path
CCCC_MODULES = {ROUTE_CCCC_N1: cccc_n1, ROUTE_CCCC_N23: cccc_n23, ROUTE_CCCC_N46: cccc_n46}
CCCC_CLI = {ROUTE_CCCC_N1: cli_cccc, ROUTE_CCCC_N23: cli_cccc_n23, ROUTE_CCCC_N46: cli_cccc_n46}

# the existing entry, its stage name and the decisions of every route
ROUTES = {
    ROUTE_R3: {"solver": "R3", "call": "kkvim.rectangle_run.run_rectangle_physical", "decisions": ("D-043",),
               "title": "SSSS, N = 1 — R3, принятый решатель"},
    ROUTE_R4: {"solver": "R4", "call": "kkvim.coupled_run.run_coupled", "decisions": ("D-047", "D-048", "D-049.7"),
               "title": "SSSS, N = 2–3 — R4, PROVISIONAL / NOT_QUALIFIED"},
    ROUTE_SSSS_N46: {"solver": "SSSS-RESEARCH", "call": "kkvim.research_ssss.run_research", "decisions": ("D-056.7", "D-060"),
                     "title": "SSSS, N = 4–6 — только исследование (RESEARCH_ONLY)"},
    ROUTE_CCCC_N1: {"solver": "CCCC-N1", "call": "kkvim.cccc_n1.run_cccc_n1_physical", "decisions": ("D-061",),
                    "title": "CCCC, N = 1 — исследование, фиксированные полушаги s = 1…8"},
    ROUTE_CCCC_N23: {"solver": "CCCC-N23", "call": "kkvim.cccc_n23.run_cccc_n23_physical", "decisions": ("D-063", "D-065"),
                     "title": "CCCC, N = 2–3 — исследование, s = 1; s = 2 только после RN-5 и RT-7"},
    ROUTE_CCCC_N46: {"solver": "CCCC-N46", "call": "kkvim.cccc_n46.run_cccc_n46_physical", "decisions": ("D-067", "D-068"),
                     "title": "CCCC, N = 4–6 — исследование, s = 1; s = 2 только после RN-5 и RT-7"},
}
# FIELD_ROLE_RULE: the statuses with which a route ended as requested; every other status keeps only a diagnostic field
REQUESTED_END = {
    ROUTE_R3: (controller.CONVERGED,),
    ROUTE_R4: (coupled_run.PROVISIONAL_FIELD_STABILIZED, coupled_run.RESEARCH_LIMIT_REACHED, coupled_run.NOT_QUALIFIED,
               coupled_run.HALF_STEP_LIMIT_REACHED),  # the last two: the end of the normative mode D-047
    ROUTE_SSSS_N46: tuple(cli_ssss.REQUESTED_END),
    **{route: (module.COMPLETED_FIXED_HALF_STEPS,) for route, module in CCCC_MODULES.items()},
}
FIELD_ROLE_RULE = ("accepted: R3 CONVERGED (field.csv); research: the requested end of another route (field_research.csv); "
                   "diagnostic: any other status with the field of the last completed half-step (field_diagnostic.csv); "
                   "no field: no field file")
R3_LIMITATIONS = (
    ("Модель: линейная теория пластин Кирхгофа–Лява; статика; постоянная толщина; однородный изотропный упругий "
     "материал; равномерная нагрузка q; SSSS."),
    "Метод: КК-ВИМ, N = 1 — принятый решатель R3 (D-043); область решателя 0,2 ≤ b/a ≤ 5.",
    ("Статус CONVERGED означает выполнение критерия остановки решателя (NUMERICAL_SPEC.md 11.2), а не оценку "
     "погрешности дискретизации или метода для данного расчёта."),
)
# The limits of R4 are those of ``kkvim.cli_ssss.LIMITS_N23``; the GUI words its two statements about convergence with
# the negation first (the rule of the GUI texts: every mention of convergence of R4 carries a preceding negation).
R4_LIMITATIONS = (
    ("Реализация D-048 — ACCEPTED_LIMITED_ENGINEERING (D-049.7) в области D-047, D-048; результат — не сходимость "
     "(D-048.4, D-056.8). Каждый результат — PROVISIONAL / NOT_QUALIFIED."),
    cli_ssss.LIMITS_N23[1],
    ("При s → ∞ — не доказанная сходимость; единственность предела и независимость от старта не утверждаются "
     "(D-056.8; Q-020, Q-022, Q-025)."),
)
LIMITATIONS = {
    ROUTE_R3: R3_LIMITATIONS,
    ROUTE_R4: (*R4_LIMITATIONS, *cli_ssss.COMMON_LIMITS),
    ROUTE_SSSS_N46: (*cli_ssss.LIMITS_N46, *cli_ssss.COMMON_LIMITS),
    **{route: tuple(module.LIMITS) for route, module in CCCC_CLI.items()},
}


def field_role(route: str, status: str, has_field: bool) -> str | None:
    """FIELD_ROLE_RULE: the role of the field of a run of ``route`` that ended with ``status``."""
    if not has_field:
        return None
    if status in REQUESTED_END[route]:
        return ROLE_ACCEPTED if route == ROUTE_R3 else ROLE_RESEARCH
    return ROLE_DIAGNOSTIC

# Defaults of the solvers, read from kkvim (single source)
DEFAULTS = {
    "tol_inf": controller.TOL_INF,
    "tol_E": controller.TOL_E,
    "s_min": controller.S_MIN,
    "s_max": controller.S_MAX,
    "half_steps": coupled_run.S_MAX,
    "transform": "fft",
    "r4_mode": R4_NORMATIVE,  # research_mode=False unless chosen explicitly (D-048.3, D-048.5)
}
R4_FIXED = {
    "tol_inf": coupled_run.TOL_INF, "tol_E": coupled_run.TOL_E, "s_min": coupled_run.S_MIN,
    "s_max": coupled_run.S_MAX, "eta_max": coupled_run.ETA_MAX, "K_research": coupled_run.K_RESEARCH,
}
CAPABILITIES = {
    "theory": "Кирхгоф–Ляв",
    "supports": SUPPORTS,  # SSSS: R3, R4, research N = 4-6; CCCC: research N = 1-6 (D-061, D-063, D-065, D-067, D-068)
    "load": "uniform",
    "N": N_AVAILABLE,
    "n_min": N_MIN,
    "r_range": (1 / R_MAX, R_MAX),  # 0.2 <= b/a <= 5, checked exactly by kkvim (D-038)
    "dense_n_max": DENSE_N_MAX,
    "cccc_half_steps": CCCC_HALF_STEPS,
    "cccc_s2_grid_max": cccc_n46.S2_GRID_MAX,  # informative: the solver's own guard decides (D-065.5, D-068)
    "ssss_research_s_max": research_ssss.S_STUDY,
}

R3_STOP = {
    controller.CONVERGED: ("met", "Критерий остановки выполнен: CONVERGED"),
    controller.STAGNATED: ("not_met", "Остановка по застою: STAGNATED — критерий остановки не выполнен"),
    controller.MAX_HALF_STEPS: ("not_met", "Достигнуто s_max: MAX_HALF_STEPS — критерий остановки не выполнен"),
}
R4_STOP = {
    coupled_run.PROVISIONAL_FIELD_STABILIZED: (
        "met", "Стабилизация поля конечной задачи: PROVISIONAL_FIELD_STABILIZED (не сходимость к точному решению)"),
    coupled_run.MAX_HALF_STEPS: ("not_met", "Достигнуто s_max = 100 без стабилизации: MAX_HALF_STEPS"),
    coupled_run.RESEARCH_LIMIT_REACHED: ("not_met", "Выполнено заданное число полушагов: RESEARCH_LIMIT_REACHED"),
    coupled_run.NOT_QUALIFIED: ("none", "Нормативный режим D-047: после s = 1 — NOT_QUALIFIED, продолжение не квалифицировано"),
    coupled_run.HALF_STEP_LIMIT_REACHED: ("none", "Нормативный режим D-047: выполнено заданное число полушагов"),
    coupled_run.PROVISIONAL_RANK_GUARD_STOP: ("not_met", "Остановка защитой ранга до QR: PROVISIONAL_RANK_GUARD_STOP"),
    coupled_run.PROVISIONAL_SET_GUARD_STOP: ("not_met", "Остановка защитой набора до QR: PROVISIONAL_SET_GUARD_STOP"),
    coupled_run.PROVISIONAL_ENERGY_E1_STOP: ("not_met", "Остановка по нарушению E1: PROVISIONAL_ENERGY_E1_STOP"),
    coupled_run.STOPPED_RANK_LOSS_PROVEN: ("not_met", "Доказанная потеря ранга (S-1): STOPPED_RANK_LOSS_PROVEN"),
}
R3_SCIENCE = ("Решатель R3 — ACCEPTED (D-043): SSSS, q = const, N = 1, 0,2 ≤ b/a ≤ 5. "
              "Отдельный расчёт научной приёмкой не является")
R4_SCIENCE = (f"{MARK}: исследовательский решатель R4, N = 2–3 (D-047, D-048). Не научная приёмка, "
              "не сходимость к точному решению, не сертификация ранга N")

R3_COLUMNS = (
    Column("s", "s"), Column("k", "k"), Column("direction", "Множитель", kind="text"),
    Column("eps_inf", "ε^∞"), Column("eps_E", "ε^E"), Column("eps_L2", "ε^L2"), Column("theta", "θ"),
    Column("satisfied", "ε ≤ tol", kind="flag"), Column("tail", "Оценка хвоста T_s"),
    Column("P", "Π_h", "N·m"), Column("energy_status", "Энергетический тест", kind="text"),
    Column("delta_P", "ΔP", "N·m"), Column("Q", "Q", "N·m"), Column("tau", "τ", "N·m"),
    Column("eta", "η (невязка)"), Column("refined", "Уточнение", kind="flag"), Column("w_bar_centre", "w̄ в центре"),
)
R4_COLUMNS = (
    Column("s", "s"), Column("k", "k"), Column("direction", "Множитель", kind="text"),
    Column("eps_inf", "ε^∞"), Column("eps_E", "ε^E"), Column("stabilization_ok", "ε ≤ tol", kind="flag"),
    Column("w_max", "max|w|", "m"), Column("P", "Π_h", "N·m"), Column("energy_status", "Энергетический тест", kind="text"),
    Column("delta_P", "ΔP", "N·m"), Column("Q", "Q", "N·m"), Column("tau", "τ", "N·m"), Column("m_s", "m_s"),
    Column("eta", "η_N (невязка)"), Column("eta_initial", "η_N до уточнения"), Column("refined", "Уточнение", kind="flag"),
    Column("rho_N", "ρ_N"), Column("kappa_rep", "κ_rep"), Column("guards_passed", "Защиты D-048.5", kind="flag"),
    Column("qr_done", "QR", kind="flag"), Column("qr_field_error", "Ошибка поля QR"), Column("stop_reason", "Причина остановки", kind="text"),
    Column("mark", "Статус", kind="text"),
)


class AdapterInputError(ValueError):
    """Input rejected before the computation (by kkvim's own checks or a GUI scope rule)."""


def dense_problem(transform, n_x, n_y) -> str | None:
    """Message if ``transform`` leaves the scope of D-048.6, else None.

    dense is admissible only while both sizes are within the audited range: max(n_x, n_y) <= 40.
    The single rule of the GUI: the form, ``check_request``, the dispatcher and the computing process use it.
    """
    if transform != "dense" or max(n_x, n_y) <= DENSE_N_MAX:
        return None
    return (f"Преобразование dense допустимо только при max(n_x, n_y) ≤ {DENSE_N_MAX} (D-048.6); "
            f"задано n_x = {n_x}, n_y = {n_y}. Выберите fft — основной путь DST")


def check_request(request: RunRequest) -> list[str]:
    """Problems of the input found before a computation; [] if the request may be started.

    Physical input, sides and grid sizes are checked by ``validated_rectangle_input`` of kkvim (the
    same function the solvers call); the domains by the solvers' own ``domain_refusal``. The GUI adds only
    the scope rules it must not let pass: a known route, the explicit research choice of SSSS N = 4-6
    (D-056.7), dense only while max(n_x, n_y) <= 40 (D-048.6) and, for CCCC N >= 2, no s >= 3 (D-065, D-068).
    """
    problems = []
    route = request.route
    if request.supports not in SUPPORTS:
        problems.append(f"Закрепление {request.supports}: неизвестно; доступны {', '.join(SUPPORTS)}")
    elif route is None:
        problems.append(f"N = {request.N}: нет решателя; доступны N = 1–6")
    plate = None
    try:
        plate, _ = validated_rectangle_input(request.a, request.b, request.t, request.E, request.nu, request.q,
                                             request.n_x, request.n_y)
    except PhysicalInputError as exc:
        problems.append(f"Ввод отклонён решателем (ERROR_INPUT): {exc}")
    if route == ROUTE_R4:
        if request.r4_mode not in R4_MODES:
            problems.append(f"Неизвестный режим R4: {request.r4_mode!r}")
        dense = dense_problem(request.transform, request.n_x, request.n_y)
        if dense:
            problems.append(dense)
    elif route == ROUTE_SSSS_N46:
        if not request.ssss_research:
            problems.append(f"SSSS, N = {request.N}: только исследование (RESEARCH_ONLY, D-056.7); расчёт выполняется лишь при явном "
                            "выборе исследовательского режима N = 4–6")
        if not _whole(request.half_steps):
            problems.append("SSSS, N = 4–6: число полушагов должно быть целым не меньше 1")
        elif plate is not None:
            refusal = research_ssss.domain_refusal(plate.a, plate.b, plate.n_x, plate.n_y, request.N, request.half_steps)
            if refusal:
                problems.append(f"Вне исследованной области N = 4–6 (NOT_EXECUTED_OUTSIDE_DOMAIN): {refusal}")
    elif route in CCCC_ROUTES:
        allowed = CCCC_HALF_STEPS[route]
        if request.half_steps not in allowed:
            problems.append(f"CCCC, N = {request.N}: число полушагов {request.half_steps!r} недоступно; допустимо "
                            f"{allowed[0]}…{allowed[-1]}" + ("; s ≥ 3 не исполняется (D-065, D-068)" if route != ROUTE_CCCC_N1 else ""))
        else:
            refusal = _cccc_domain_refusal(route, request)
            if refusal:
                problems.append(f"Вне области решателя CCCC (NOT_EXECUTED_OUTSIDE_DOMAIN): {refusal}")
    return problems


def _whole(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def _cccc_domain_refusal(route, request):
    module = CCCC_MODULES[route]
    if route == ROUTE_CCCC_N1:
        return module.domain_refusal(request.a, request.b, request.n_x, request.n_y, request.half_steps)
    return module.domain_refusal(request.a, request.b, request.n_x, request.n_y, request.N, request.half_steps)


def stopping_parameters(request: RunRequest) -> tuple[float, float]:
    """(tol_inf, tol_E) passed to the R3 controller: as entered, or those of the strict mode (11.2, 11.4)."""
    if request.strict:
        return controller.strict_parameters(request.n_x, request.n_y, request.tol_inf, request.tol_E)
    return request.tol_inf, request.tol_E


def execute(request: RunRequest) -> RunOutcome:
    """Run the solver for the request and return a copy of its result."""
    problems = check_request(request)
    if problems:
        raise AdapterInputError("; ".join(problems))
    route = request.route
    if route == ROUTE_SSSS_N46:
        return execute_ssss_research(request)
    if route in CCCC_ROUTES:
        return execute_cccc(request)
    if route == ROUTE_R3:
        tol_inf, tol_E = stopping_parameters(request)
        result = run_rectangle_physical(request.a, request.b, request.t, request.E, request.nu, request.q,
                                        request.n_x, request.n_y, tol_inf=tol_inf, tol_E=tol_E,
                                        s_min=request.s_min, s_max=request.s_max)
        return outcome_from_r3(request, result)
    _, D = validated_rectangle_input(request.a, request.b, request.t, request.E, request.nu, request.q,
                                     request.n_x, request.n_y)
    run = run_coupled_for(request, D)
    return outcome_from_r4(request, D, run)


def run_coupled_for(request: RunRequest, D: float):
    """``run_coupled`` with the mode of the request; arguments only, nothing else.

    ``research_mode`` is True only for an explicitly chosen research mode (D-048.5); the normative mode passes
    the library default (False) and returns NOT_QUALIFIED after s = 1 (D-047, D-048.3).
    """
    return coupled_run.run_coupled(request.a, request.b, D, request.q, request.n_x, request.n_y, request.N,
                                   transform=request.transform, half_steps=request.half_steps,
                                   research_mode=request.research_mode,
                                   stop_on_stabilization=request.stop_on_stabilization)


# ---- R3 ------------------------------------------------------------------------------------------

def outcome_from_r3(request: RunRequest, result) -> RunOutcome:
    report = halfstep_report(result)
    run = result.run
    w = w_bar = x = y = None
    if run is not None and run.s:
        last = run.history.records[-1]
        x, y = last.grid_x.nodes, last.grid_y.nodes
        w, w_bar = run.last_field(), result.dimensionless_field()
    stop_kind, stop_text = _r3_stop(result.status)
    notes = []
    tol_inf, tol_E = stopping_parameters(request)
    if (tol_inf, tol_E, request.s_min, request.s_max) != (
            DEFAULTS["tol_inf"], DEFAULTS["tol_E"], DEFAULTS["s_min"], DEFAULTS["s_max"]):
        notes.append("Параметры остановки отличаются от принятых по умолчанию (NUMERICAL_SPEC.md 11.2); "
                     "приёмка D-043 выполнена с параметрами по умолчанию")
    parameters = ()
    if run is not None and run.parameters is not None:
        p = run.parameters
        parameters = (("tol_inf", p.tol_inf), ("tol_E", p.tol_E), ("s_min", p.s_min), ("s_max", p.s_max),
                      ("eps_floor", p.eps_floor), ("strict", request.strict))
    diagnostics = (("status", result.status), ("tail_estimate_note", run.tail_estimate_note if run else None))
    return RunOutcome(
        origin=ORIGIN_SOLVER, request=request, solver="R3", call="kkvim.rectangle_run.run_rectangle_physical",
        status=result.status, reason="", error=None if result.error is None else str(result.error),
        stop_kind=stop_kind, stop_text=stop_text, science_kind="success" if stop_kind == "met" else "neutral",
        science_text=R3_SCIENCE, mark=None, D=result.D, x=frozen_copy(x), y=frozen_copy(y), w=frozen_copy(w),
        w_bar=frozen_copy(w_bar), s=0 if run is None else run.s, columns=R3_COLUMNS,
        history=tuple(_r3_row(h) for h in report.halfsteps), parameters=parameters, diagnostics=diagnostics,
        notes=tuple(notes), versions=dict(report.versions), **_route_fields(request, result.status, w is not None),
    )


def _route_fields(request: RunRequest, status: str, has_field: bool, s_requested=None) -> dict:
    """route, field_role, requested_completed, s_requested, decisions and limitations of an outcome (FIELD_ROLE_RULE)."""
    route = request.route
    return {"route": route, "field_role": field_role(route, status, has_field),
            "requested_completed": status in REQUESTED_END[route], "s_requested": s_requested,
            "decisions": ROUTES[route]["decisions"], "limitations": LIMITATIONS[route]}


def _r3_stop(status):
    if status in R3_STOP:
        return R3_STOP[status]
    return "error", f"Ошибка решателя: {status}"


def _r3_row(h) -> dict:
    ch, en = h.changes, h.energy
    centre = h.control_values[0][1] if h.control_values else None
    return {
        "s": h.s, "k": h.k, "direction": h.direction,
        "eps_inf": None if ch is None else ch.eps_inf, "eps_E": None if ch is None else ch.eps_E,
        "eps_L2": None if ch is None else ch.eps_L2, "theta": None if ch is None else ch.theta,
        "satisfied": None if ch is None else ch.satisfied, "tail": None if ch is None else ch.tail_estimate,
        "P": h.P, "energy_status": None if en is None else en.status,
        "delta_P": None if en is None else en.delta_P, "Q": None if en is None else en.Q,
        "tau": None if en is None else en.tau, "eta": h.eta, "refined": h.refined,
        "w_bar_centre": centre if isinstance(centre, float) else None,
    }


# ---- R4 ------------------------------------------------------------------------------------------

def outcome_from_r4(request: RunRequest, D: float, run) -> RunOutcome:
    w = run.field()
    x = y = w_bar = None
    if run.grid_x is not None:
        x, y = run.grid_x.nodes, run.grid_y.nodes
    if w is not None:
        w_bar = w * D / (request.q * request.a**4)  # MATH_SPEC 2.5, reference side a (as for N = 1)
    stop_kind, stop_text = R4_STOP.get(run.status, ("error", f"Ошибка решателя: {run.status}"))
    p = run.parameters
    parameters = () if p is None else tuple(
        (name, getattr(p, name)) for name in (
            "N", "transform", "half_steps", "research_mode", "stop_on_stabilization", "K_research", "eps_floor",
            "tol_inf", "tol_E", "s_min", "s_max", "eta_max"))
    diagnostics = (
        ("status", run.status), ("reason", run.reason), ("mark", run.mark),
        ("energy_s2_checked", run.energy_s2_checked), ("energy_s2_mandatory", run.energy_s2_mandatory),
        ("energy_s2_status", run.energy_s2_status), ("energy_s2_m", run.energy_s2_m),
        ("energy_s2_mandatory_unresolved", run.energy_s2_mandatory_unresolved),
        ("energy_s2_reason", run.energy_s2_reason), ("first_stabilization_s", run.first_stabilization_s),
    )
    notes = [f"Все величины R4 — {MARK}"]
    if request.r4_mode == R4_NORMATIVE:
        notes.append("Нормативный режим D-047: продолжение после s = 1 через QR не квалифицировано")
    else:
        notes.append("Исследовательское продолжение через QR включено явно (D-048.5). PROVISIONAL_FIELD_STABILIZED — "
                     "стабилизация поля конечной задачи, а не доказанная сходимость (D-048.4)")
    if request.r4_mode == R4_RESEARCH_FIXED:
        notes.append("Фиксированное число полушагов (дополнение к D-048 от 2026-10-18): "
                     "стабилизация регистрируется, но не останавливает расчёт")
    if run.energy_s2_mandatory_unresolved:
        notes.append(f"Обязательный энергетический тест s = 2 — UNRESOLVED: {run.energy_s2_reason}")
    return RunOutcome(
        origin=ORIGIN_SOLVER, request=request, solver="R4", call="kkvim.coupled_run.run_coupled",
        status=run.status, reason=run.reason, error=None if run.error is None else str(run.error),
        stop_kind=stop_kind, stop_text=stop_text, science_kind="warning", science_text=R4_SCIENCE, mark=run.mark,
        D=D, x=frozen_copy(x), y=frozen_copy(y), w=frozen_copy(w), w_bar=frozen_copy(w_bar), s=run.s,
        columns=R4_COLUMNS, history=tuple(_r4_row(step) for step in run.steps), parameters=parameters,
        diagnostics=diagnostics, notes=tuple(notes), versions=versions(),
        **_route_fields(request, run.status, w is not None, request.half_steps),
    )


def _r4_row(step) -> dict:
    e, rank, g = step.energy, step.rank, step.guards
    return {
        "s": step.s, "k": step.k, "direction": step.direction, "eps_inf": step.eps_inf, "eps_E": step.eps_E,
        "stabilization_ok": step.stabilization_ok, "w_max": step.w_max, "P": e.P, "energy_status": e.status,
        "delta_P": e.delta_P, "Q": e.Q, "tau": e.tau, "m_s": step.m_s, "eta": step.eta_N,
        "eta_initial": step.eta_initial, "refined": step.refined, "rho_N": rank.rho_N, "kappa_rep": rank.kappa_rep,
        "guards_passed": None if g is None else g.passed, "qr_done": step.qr_done,
        "qr_field_error": step.qr_field_error, "stop_reason": step.stop_reason, "mark": step.mark,
    }


def versions() -> dict:
    """Python, kkvim, numpy and scipy versions (the same record as the R3 report of NUMERICAL_SPEC.md 11.5)."""
    record = {"python": platform.python_version()}
    for package in ("kkvim", "numpy", "scipy"):
        try:
            record[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            record[package] = "unknown"
    return record


__all__ = ["CAPABILITIES", "DEFAULTS", "ERROR_INPUT", "FIELD_ROLE_RULE", "MARK", "N_AVAILABLE", "R4_FIXED", "RESEARCH_MARK",
           "ROUTES", "AdapterInputError", "check_request", "execute", "field_role", "outcome_from_cccc", "outcome_from_r3",
           "outcome_from_r4", "outcome_from_ssss_research", "run_coupled_for", "stopping_parameters"]


# ---- SSSS, N = 4-6 (research only) ----------------------------------------------------------------

R46_COLUMNS = tuple(
    Column(c.key, c.title, "q²a⁶/D", c.kind) if c.unit == "N·m" else
    Column("w_max", "max|w̄| (ячейка)", "", "number") if c.key == "w_max" else c
    for c in R4_COLUMNS)
R46_STOP = {  # the fixed number of half-steps is the requested end of this route (stabilization is only recorded)
    coupled_run.RESEARCH_LIMIT_REACHED: ("none", ("Выполнено заданное число полушагов: RESEARCH_LIMIT_REACHED (стабилизация "
                                                  "только регистрируется; не сходимость)")),
}
SSSS_RESEARCH_SCIENCE = (f"{RESEARCH_MARK}: исследовательский путь SSSS, N = 4–6 (D-056.7, D-060). Не рабочий и не "
                         "квалифицированный решатель; не сходимость и не научная приёмка")


def execute_ssss_research(request: RunRequest) -> RunOutcome:
    """The route of ``run-ssss --research-only`` for N = 4-6: the studied cell, the output scaled by g = q a^4 / D."""
    plate, D = validated_rectangle_input(request.a, request.b, request.t, request.E, request.nu, request.q,
                                         request.n_x, request.n_y)
    try:
        w_scale = plate.q * plate.a**4 / D
    except OverflowError:
        w_scale = float("inf")
    if not math.isfinite(w_scale) or w_scale == 0.0:  # the check of run-ssss (kkvim.cli_ssss.execute_research)
        raise AdapterInputError(f"Ввод отклонён (ERROR_INPUT): масштаб q a⁴ / D = {w_scale!r} не конечен или равен нулю")
    control = research_ssss.control_ratio(plate.a, plate.b)
    r = float(control) if control is not None else plate.b / plate.a  # None only before the S-1 exit (as run-ssss)
    run = research_ssss.run_research(r, plate.n_x, plate.n_y, request.N, half_steps=request.half_steps)
    return outcome_from_ssss_research(request, D, w_scale, plate.a, run)


def outcome_from_ssss_research(request: RunRequest, D: float, w_scale: float, a: float, run) -> RunOutcome:
    x = y = w = w_bar = None
    if run.x is not None:
        x, y, w, w_bar = cli_ssss._field_arrays(cli_ssss._Run(run, D, w_scale, True), a)
    stop_kind, stop_text = R46_STOP.get(run.status) or R4_STOP.get(run.status, ("error", f"Ошибка решателя: {run.status}"))
    p = run.parameters
    parameters = () if p is None else tuple(
        (name, getattr(p, name)) for name in ("N", "transform", "half_steps", "research_mode", "stop_on_stabilization",
                                              "K_research", "eps_floor", "tol_inf", "tol_E", "s_min", "s_max", "eta_max"))
    parameters += (("cell", "a = 1, b = r, q = D = 1 (ячейка исследования); w = w̄ · qa⁴/D"), ("w_scale_q_a4_over_D", w_scale))
    diagnostics = (("status", run.status), ("reason", run.reason), ("mark", RESEARCH_MARK),
                   ("first_stabilization_s", run.first_stabilization_s), ("energy_unit", "q²a⁶/D (ячейка)"))
    notes = [f"SSSS, N = {request.N} — только исследование: {RESEARCH_MARK} (D-056.7)",
             "Энергии в истории — в единицах q²a⁶/D ячейки исследования; поле w — в метрах (w = w̄ · qa⁴/D)",
             cli_ssss.STATUS_NOTES.get(run.status, cli_ssss.ERROR_NOTE)]
    return RunOutcome(
        origin=ORIGIN_SOLVER, request=request, solver=ROUTES[ROUTE_SSSS_N46]["solver"], call=ROUTES[ROUTE_SSSS_N46]["call"],
        status=run.status, reason=run.reason, error=None if run.error is None else str(run.error),
        stop_kind=stop_kind, stop_text=stop_text, science_kind="warning", science_text=SSSS_RESEARCH_SCIENCE,
        mark=RESEARCH_MARK, D=D, x=frozen_copy(x), y=frozen_copy(y), w=frozen_copy(w), w_bar=frozen_copy(w_bar), s=run.s,
        columns=R46_COLUMNS, history=tuple(_r4_row(step) for step in run.steps), parameters=parameters,
        diagnostics=diagnostics, notes=tuple(notes), versions=versions(),
        **_route_fields(request, run.status, w is not None, request.half_steps),
    )


# ---- CCCC, N = 1-6 (research only) ----------------------------------------------------------------

CCCC_PROTECTIVE = (cccc_n23.STOPPED_RANK_GUARD_UNCERTIFIED, cccc_n23.STOP_BEFORE_QR_S1, cccc_n1.UNRESOLVED_C3PLUS_PREMISE,
                   cccc_n1.PROTECTIVE_STOP_FACTORIZATION, cccc_n1.STOPPED_ETA_GUARD, cccc_n1.STOP_ZERO_FIELD)
CCCC_N1_COLUMNS = (
    Column("s", "s"), Column("k", "k"), Column("direction", "Множитель", kind="text"), Column("status", "Статус полушага", kind="text"),
    Column("premises_status", "Предпосылки C-3⁺", kind="text"), Column("eta", "η (невязка)"), Column("cond_S", "cond(S)"),
    Column("P_hat", "Π̂_h", "N·m"), Column("delta_P_hat", "ΔΠ̂", "N·m"), Column("field_change_inf", "max|Δw|/max|w|"),
    Column("alpha_centre", "α в центре"), Column("sigma", "Знак нормировки"), Column("reason", "Причина", kind="text"),
)
CCCC_N26_COLUMNS = (
    Column("s", "s"), Column("k", "k"), Column("direction", "Множитель", kind="text"), Column("status", "Статус полушага", kind="text"),
    Column("qr_executed", "QR", kind="flag"), Column("premises_status", "Предпосылки C-3⁺", kind="text"), Column("eta", "η (наблюдение)"),
    Column("cond_S", "cond(S)"), Column("V_exact_rank", "Точный ранг V"), Column("rho_N", "ρ_N"), Column("kappa_rep", "κ_rep"),
    Column("P_hat", "Π̂_h", "N·m"), Column("alpha_centre", "α в центре"),
    Column("rn5_verdict", "RN-5", kind="text"), Column("rn5_margin", "σ_low/ε (RN-5)"), Column("rt7_passed", "RT-7", kind="flag"),
    Column("rt7_t", "t (RT-7)"), Column("rt7_tau", "τ (RT-7)"), Column("rt7_sin_theta_upper", "sin θ ≤ (RT-7)"),
    Column("P_hat_previous_QR", "Π̂_h до QR (s = 1)", "N·m"), Column("field_preservation", "Сохранение поля QR"),
    Column("reason", "Причина", kind="text"),
)


def execute_cccc(request: RunRequest) -> RunOutcome:
    """The CCCC entry of the route with the physical input; nothing but its arguments."""
    route = request.route
    module = CCCC_MODULES[route]
    args = (request.a, request.b, request.t, request.E, request.nu, request.q, request.n_x, request.n_y)
    if route == ROUTE_CCCC_N1:
        result = module.run_cccc_n1_physical(*args, request.half_steps)
    elif route == ROUTE_CCCC_N23:
        result = module.run_cccc_n23_physical(*args, request.N, request.half_steps)
    else:
        result = module.run_cccc_n46_physical(*args, request.N, request.half_steps)
    return outcome_from_cccc(request, result)


def _cccc_stop(status):
    if status == cccc_n1.COMPLETED_FIXED_HALF_STEPS:
        return "none", ("Выполнено заданное число полушагов: COMPLETED_FIXED_HALF_STEPS (фиксированные полушаги, критерия "
                        "остановки нет; не сходимость)")
    if status in CCCC_PROTECTIVE:
        return "not_met", f"Защитная остановка решателя: {status}; запрошенный расчёт не выполнен"
    return "error", f"Ошибка решателя: {status}"


def outcome_from_cccc(request: RunRequest, result) -> RunOutcome:
    route = request.route
    w = result.field()
    w_bar = result.dimensionless_field()
    x = y = None
    if w is not None:
        h_x, h_y = result.a / result.n_x, result.b / result.n_y
        x = [i * h_x for i in range(result.n_x + 1)]  # the node coordinates of the CCCC commands (i h_x, j h_y)
        y = [j * h_y for j in range(result.n_y + 1)]
    stop_kind, stop_text = _cccc_stop(result.status)
    decisions = ", ".join(ROUTES[route]["decisions"])
    science = (f"{RESEARCH_MARK}: CCCC, N = {request.N}, режим {result.mode} ({decisions}). Не сходимость, не точность поля, "
               "не полный ранг и не научная приёмка")
    n1_route = route == ROUTE_CCCC_N1
    columns = CCCC_N1_COLUMNS if n1_route else CCCC_N26_COLUMNS
    history = tuple(_cccc_row(step, columns, n1_route) for step in result.steps)
    parameters = (("mode", result.mode), ("half_steps_requested", result.half_steps_requested),
                  ("application_status", result.application_status))
    if not n1_route:
        parameters += (("rank_guard_rule", result.rank_guard_rule), ("s2_grid_max", CCCC_MODULES[route].S2_GRID_MAX))
    last = result.steps[-1] if result.steps else None
    diagnostics = (("status", result.status), ("reason", result.reason), ("mark", RESEARCH_MARK),
                   ("half_steps_completed", result.half_steps_completed), ("P-4", cccc_n1.P4_ASSUMED),
                   ("last_premises_status", None if last is None else last.premises_status))
    notes = [f"CCCC — только исследование: {RESEARCH_MARK} ({decisions})"]
    note = CCCC_CLI[route].STATUS_NOTES.get(result.status)
    if note:
        notes.append(note)
    if not n1_route and request.half_steps == 2 and max(request.n_x, request.n_y) > CCCC_MODULES[route].S2_GRID_MAX:
        notes.append(f"s = 2 разрешён только при n_x, n_y ≤ {CCCC_MODULES[route].S2_GRID_MAX}: решатель остановился до RN-5 и QR")
    return RunOutcome(
        origin=ORIGIN_SOLVER, request=request, solver=ROUTES[route]["solver"], call=ROUTES[route]["call"],
        status=result.status, reason=result.reason or "", error=None, stop_kind=stop_kind, stop_text=stop_text,
        science_kind="warning", science_text=science, mark=RESEARCH_MARK, D=result.D, x=frozen_copy(x), y=frozen_copy(y),
        w=frozen_copy(w), w_bar=frozen_copy(w_bar), s=result.half_steps_completed, columns=columns, history=history,
        parameters=parameters, diagnostics=diagnostics, notes=tuple(notes), versions=versions(),
        **_route_fields(request, result.status, w is not None, request.half_steps),
    )


def _cccc_row(step, columns, n1_route) -> dict:
    row = {c.key: getattr(step, c.key, None) for c in columns}
    if not n1_route:
        row.update(cli_cccc_n23._certificate_columns(step))
        row = {c.key: row.get(c.key) for c in columns}
    for key, value in row.items():
        if hasattr(value, "item") and not isinstance(value, (str, bytes)):
            row[key] = value.item()  # numpy scalars as Python values (read-only copies of the solver values)
    return row
