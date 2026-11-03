"""Adapter between the GUI and the scientific package kkvim (GUI_DESIGN.md, 3.4). Qt-free.

This is the only module of ``kkvim_gui`` that imports ``kkvim`` (tests/test_gui_independence.py).
It calls the existing solvers unchanged and copies their results; no formula, solver, numerical
policy or stopping rule is implemented or changed here:

- N = 1: ``kkvim.rectangle_run.run_rectangle_physical`` (R3, ACCEPTED, D-043) and its
  ``halfstep_report`` (NUMERICAL_SPEC.md 11.5);
- N = 2-3: ``kkvim.rectangle.validated_rectangle_input`` (the accepted input checks and
  D = E t^3 / (12 (1 - nu^2)) of ``kkvim.physical``), then ``kkvim.coupled_run.run_coupled``
  (R4; D-047, D-048 and its addendum). Every R4 result carries PROVISIONAL / NOT_QUALIFIED.

The input arrives in the base SI units m and Pa (``units``); D is in N·m and w in m.
w_bar = w D / (q a^4) with the reference side a (MATH_SPEC 2.5) is taken from
``RectangleResult.dimensionless_field`` for N = 1 and formed by the same expression for N = 2-3.
"""

import platform
from importlib import metadata

from kkvim import controller, coupled_run
from kkvim.coefficients import ERROR_INPUT
from kkvim.grid import N_MIN
from kkvim.physical import PhysicalInputError
from kkvim.r3_input import R_MAX
from kkvim.rectangle import validated_rectangle_input
from kkvim.rectangle_run import halfstep_report, run_rectangle_physical

from kkvim_gui.backend.outcome import ORIGIN_SOLVER, Column, RunOutcome, frozen_copy
from kkvim_gui.backend.request import R4_MODES, R4_NORMATIVE, R4_RESEARCH_FIXED, RunRequest

MARK = coupled_run.MARK  # "PROVISIONAL / NOT_QUALIFIED"
N_AVAILABLE = (1, *coupled_run.N_SOLVE)  # 1: R3; 2-3: R4 (D-047)
DENSE_N_MAX = 40  # D-048.6: dense is admissible only up to max(n_x, n_y) = 40; fft is the main path

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
    "supports": ("SSSS",),  # CCCC: not implemented (no owner decision)
    "load": "uniform",
    "N": N_AVAILABLE,
    "n_min": N_MIN,
    "r_range": (1 / R_MAX, R_MAX),  # 0.2 <= b/a <= 5, checked exactly by kkvim (D-038)
    "dense_n_max": DENSE_N_MAX,
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
    same function the solvers call); the GUI adds only the scope rules it must not let pass: SSSS
    (CCCC is not implemented), N in 1-3 (D-047), dense only while max(n_x, n_y) <= 40 (D-048.6).
    """
    problems = []
    if request.supports not in CAPABILITIES["supports"]:
        problems.append(f"Закрепление {request.supports}: не реализовано (нет решения владельца); доступно только SSSS")
    if request.N not in N_AVAILABLE:
        problems.append(f"N = {request.N}: не реализовано; доступны N = 1 (R3) и N = 2–3 (R4, D-047)")
    try:
        validated_rectangle_input(request.a, request.b, request.t, request.E, request.nu, request.q,
                                  request.n_x, request.n_y)
    except PhysicalInputError as exc:
        problems.append(f"Ввод отклонён решателем (ERROR_INPUT): {exc}")
    if request.N in coupled_run.N_SOLVE:
        if request.r4_mode not in R4_MODES:
            problems.append(f"Неизвестный режим R4: {request.r4_mode!r}")
        dense = dense_problem(request.transform, request.n_x, request.n_y)
        if dense:
            problems.append(dense)
    return problems


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
    if request.N == 1:
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
        notes=tuple(notes), versions=dict(report.versions),
    )


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


__all__ = ["CAPABILITIES", "DEFAULTS", "ERROR_INPUT", "MARK", "N_AVAILABLE", "R4_FIXED", "AdapterInputError",
           "check_request", "execute", "outcome_from_r3", "outcome_from_r4", "run_coupled_for", "stopping_parameters"]
