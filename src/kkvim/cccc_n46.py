"""CCCC, N = 4, 5 or 6: the half-steps s = 1 (D-067) and s = 2 (D-068) of the coupled KK-MVI, limited research modes. Never a converged or
qualified solution.

    run_cccc_n46(a, b, D, q, n_x, n_y, N, half_steps)
    run_cccc_n46_physical(a, b, t, E, nu, q, n_x, n_y, N, half_steps)

D-067 extends to N = 4...6 only the operations of the half-step s = 1 that D-063 accepted for N = 2, 3 ("Policy" is
docs/R5_CCCC_N23_LIMITED_POLICY.proposal.md, sections 2-3; the discrete formulas of MATH_SPEC part C, ВБC-12...ВБC-14 and the start of C.7,
ВБC-13 p. 3-4, hold for any N >= 1):

    1  the start set T_N, U_i = sin^2(i pi l / n_x), i = 1..N; its exact rank (STOP_BEFORE_QR_S1 if < N, D-049.5 p. 2)
    2  QR of the start set with the normalisation sqrt(h_x) (G); exact rank of G
    4  C-3+ assembly of A, B, C, F with the premises P-1...P-5 (P-4 assumed)
    5  B-C solve (DST-I + rank-2N Sherman-Morrison-Woodbury) of D [A ⊗ D4 - 2 B ⊗ L + C ⊗ I] vec(V) = F ⊗ 1
    6  the residual eta - an OBSERVATION only: no guard (D-067; 100u, 16u and c_f = 10 of N = 2, 3 are not transferred)
    7  exact rank of the stored V, singular values, rho_N, kappa_rep - OBSERVATION only; P_hat, alpha_centre

The operations are those of ``kkvim.cccc_n23`` (``_solve_given`` with ``eta_guard=False``), which is not changed. With half_steps >= 2 the
half-step s = 2 is ``cccc_n23._half_step_2`` unchanged (D-068: the sequence of D-065, 7 C-F extended to N = 4...6):

    C  exact rank of V (STOP_BEFORE_QR_S1 if < N); n_x or n_y > 32: stop before RN-5 and before QR; the RN-5 certificate on the saved G, V of
       this run; not certified -> STOPPED_RANK_GUARD_UNCERTIFIED before QR
    D  QR of V only after RN-5 (G2), the companion G R^T
    E  RT-7 (R7-1...R7-3) on the saved V, the actual G2 and the RN-5 of this run; failed -> stop, G2 is not used for assembly or solve
    F  the X-half-step on G2 (x and y exchanged): C-3+ with its premises, B-C, the field X G2^T; no eta guard
    G  half_steps >= 3: ``cccc_n23._guard_3``, a stop before the QR that would open s = 3

Every stop before s = 2 keeps the field of s = 1 as diagnostic; it never proves a loss of rank. The reasons written by ``cccc_n23`` name D-065;
for N = 4...6 the authorization is D-068, which this module appends to each reason. D-066 is not extended.

| Component                                      | Specification           | Level        | Decision           | Check            |
| w = sum X_i(x) Y_i(y), the half-step over y    | MATH_SPEC C.3-C.4       | A (authors)  | D-024, D-050       | tests            |
| energy Pi_h^C, system (ЧФ-C1), A, B, C, F       | Policy 2; ВБC-12, -13   | B (project)  | D-049.1, D-067     | tests            |
| start T_N (sin^2 basis), orientation           | C.7; Policy 3 step 1    | A/B          | D-049.3, D-067     | tests            |
| QR of the start set                            | Policy 3 step 2         | V (numerics) | D-057.1, D-067     | tests            |
| assembly of C (C-3+), premises P-1...P-5       | Policy 3 step 4         | V (numerics) | D-054.3, D-067     | tests            |
| half-step solve B-C                            | Policy 3 step 5         | V (numerics) | D-049.2, D-067     | tests (dense)    |
| eta, rank diagnostic, cond(S)                  | Policy 3 steps 6-7      | V (numerics) | D-067 (OBSERVATION)| tests            |
| RN-5 before the QR of V, RT-7 after it         | study 4.6; D-065, 6     | B (project)  | D-068              | tests; QA (D-068)|
| QR of V, X-half-step s = 2 (C-3+, B-C over x)  | ВБC-13 p. 1; Policy 3   | B / V        | D-068              | tests; QA (D-068)|
| guard of s >= 3 (before the QR of X)           | D-065, 7 G              | V (numerics) | D-068              | tests            |

Limits: P-4 is assumed; B-C is not qualified on working grids; the field belongs to span(G), the start set after the rounding of QR; one
half-step does not approximate the discrete or the continuous solution; RN-5 certifies rank w_h(1) = N of one run and RT-7 only the Euclidean
transversality of span G2 to the exact y-subspace, neither the accuracy, the conditioning nor the rank of w(2), nor anything of s >= 3; the full rank of w(1) for N >= 4 is not considered by MATH_SPEC C.1 and
is UNRESOLVED; the exact rank of the stored binary64 factors is not a certificate for the unrounded discrete solution. The grid bound
2N + 1 <= n_x, n_y <= 1280 is an engineering applicability limit, not an accuracy certificate; the lower bound over y does not establish the
full rank of the field. s = 2 needs n_x, n_y <= 32 (D-068). Independent QA of the s = 2 path is still to come; the tests of the module are not
that QA.

Statuses never raise: a refused input is a result with its status and reason, without a field.
"""

import math
from dataclasses import dataclass, field

import numpy as np

import kkvim.cccc_n1 as n1
import kkvim.cccc_n23 as n23
from kkvim.physical import PhysicalInputError
from kkvim.rectangle import validated_rectangle_input

MODE = n23.MODE
APPLICATION_STATUS = n23.APPLICATION_STATUS
DECISION = "D-067, D-068"
RANK_GUARD_RULE = ("D-057.1 (вариант В) с исключением D-068: N = 4–6 — s = 1 (D-067); s = 2 только при n_x, n_y ≤ 32 после RN-5 до QR и RT-7 "
                   "после QR; s ≥ 3 — остановка до QR")
RANK_DECISION = n23.RANK_DECISION
ORIENTATION = n23.ORIENTATION

COMPLETED_FIXED_HALF_STEPS = n23.COMPLETED_FIXED_HALF_STEPS
STOPPED_RANK_GUARD_UNCERTIFIED = n23.STOPPED_RANK_GUARD_UNCERTIFIED
STOP_BEFORE_QR_S1 = n23.STOP_BEFORE_QR_S1
NOT_EXECUTED_OUTSIDE_DOMAIN = n23.NOT_EXECUTED_OUTSIDE_DOMAIN
ERROR_INPUT = n23.ERROR_INPUT
ERROR_NONFINITE = n23.ERROR_NONFINITE
UNRESOLVED_C3PLUS_PREMISE = n23.UNRESOLVED_C3PLUS_PREMISE
PROTECTIVE_STOP_FACTORIZATION = n23.PROTECTIVE_STOP_FACTORIZATION
STOP_ZERO_FIELD = n23.STOP_ZERO_FIELD
HALF_STEP_DONE = n23.HALF_STEP_DONE
STATUSES = (COMPLETED_FIXED_HALF_STEPS, STOPPED_RANK_GUARD_UNCERTIFIED, STOP_BEFORE_QR_S1, NOT_EXECUTED_OUTSIDE_DOMAIN, ERROR_INPUT,
            ERROR_NONFINITE, UNRESOLVED_C3PLUS_PREMISE, PROTECTIVE_STOP_FACTORIZATION, STOP_ZERO_FIELD)  # no STOPPED_ETA_GUARD (D-067)

PREMISES_ESTABLISHED = n23.PREMISES_ESTABLISHED
PREMISES_UNRESOLVED = n23.PREMISES_UNRESOLVED
P4_ASSUMED = n23.P4_ASSUMED

N_VALUES = (4, 5, 6)  # D-067
N_MAX = n1.N_MAX  # D-067: n_x, n_y <= 1280, a bound of the domain, not a tolerance
S_MIN, S_MAX = n1.S_MIN, n1.S_MAX  # the number of half-steps asked for, 1..8 as the other CCCC commands; at most s = 2 is executed
S2_GRID_MAX = 32  # D-068: s = 2 only for n_x, n_y <= 32; a bound of the authorization, not a certification of these grids
D068_NOTE = "N = 4–6: исполнение s ≥ 2 разрешено D-068 (последовательность D-065, 7 C–G)"
GRID_KEY_N23 = f"n_x, n_y <= {n23.S2_GRID_MAX} (D-065.5)"
GRID_KEY = f"n_x, n_y <= {S2_GRID_MAX} (D-068)"

HalfStep = n23.HalfStep


def grid_min(N: int) -> int:
    """The smallest number of intervals of each direction, 2N + 1 (D-067); an applicability limit, not an accuracy certificate."""
    return 2 * N + 1


@dataclass(frozen=True)
class CCCCN46Result(n23.CCCCN23Result):
    """The run. Every result is RESEARCH_ONLY / PROVISIONAL / NOT_QUALIFIED; ``field()`` and ``factors()`` as for N = 2, 3."""

    rank_guard_rule: str = field(default=RANK_GUARD_RULE)


def _half_step_1(step: HalfStep, a: float, b: float, D: float, q: float, n_x: int, n_y: int) -> None:
    """Policy 3, steps 1-8 for N = 4...6, operation by operation as ``cccc_n23._half_step_1``, without the guard of eta (D-067)."""
    N = step.N
    step.conditions = n23.start_conditions(N, n_x, n_y)
    step.T = n23.start_set(n_x, N)
    step.exact_rank_T = n23.exact_rank(step.T)
    if step.exact_rank_T < N:
        raise n23._Stop(STOP_BEFORE_QR_S1, f"точный ранг стартового набора {step.exact_rank_T} < N = {N} (D-049.5, п. 2)")
    hf = a / n_x
    step.G, step.R = n23.orthonormalise(step.T, hf)
    step.qr_executed = True
    step.R_diag_min_abs = float(np.min(np.abs(np.diag(step.R))))
    singular = np.linalg.svd(step.R, compute_uv=False)
    step.cond_A_before_QR = float((singular[0] / singular[-1]) ** 2) if singular[-1] > 0 else math.inf
    step.exact_rank_G = n23.exact_rank(step.G)
    if not np.all(np.diag(step.R) != 0.0) or step.exact_rank_G < N:
        raise n23._Stop(PROTECTIVE_STOP_FACTORIZATION, f"после QR: min|diag R| = {step.R_diag_min_abs!r}, точный ранг G = {step.exact_rank_G}")
    n23._solve_given(step, a, n_x, b, n_y, D, q, eta_guard=False)


def _half_step_2(first: HalfStep, step: HalfStep, a: float, b: float, D: float, q: float, n_x: int, n_y: int) -> None:
    """D-068: ``cccc_n23._half_step_2`` unchanged (rank of V, grid, RN-5, QR, RT-7, X-half-step); a stop raises ``_Stop`` naming D-068."""
    if n23.S2_GRID_MAX != S2_GRID_MAX:  # the grid guard is that of cccc_n23; D-068 fixes 32
        raise n23._Stop(STOPPED_RANK_GUARD_UNCERTIFIED, f"граница сетки cccc_n23 {n23.S2_GRID_MAX} ≠ {S2_GRID_MAX} (D-068); QR не выполняется")
    try:
        n23._half_step_2(first, step, a, b, D, q, n_x, n_y)
    except n23._Stop as stop:
        raise n23._Stop(stop.status, f"{stop.reason} [{D068_NOTE}]") from None
    finally:
        if step.conditions is not None and GRID_KEY_N23 in step.conditions:
            step.conditions = {(GRID_KEY if name == GRID_KEY_N23 else name): held for name, held in step.conditions.items()}


def _guard_3(second: HalfStep, b: float, n_y: int) -> HalfStep:
    """``cccc_n23._guard_3`` unchanged: a stop before the QR of X that would open s = 3; nothing of s = 3 is computed."""
    step = n23._guard_3(second, b, n_y)
    step.reason = f"{step.reason} [{D068_NOTE}]"
    return step


def _execute(a: float, b: float, D: float, q: float, n_x: int, n_y: int, N: int, half_steps: int, physical=None) -> CCCCN46Result:
    first = HalfStep(s=1, k=0, direction="Y", n_s=n_y, h_s=b / n_y, N=N)
    try:
        _half_step_1(first, a, b, D, q, n_x, n_y)
    except n23._Stop as stop:
        first.status, first.reason = stop.status, stop.reason
        return CCCCN46Result(stop.status, stop.reason, a, b, D, q, n_x, n_y, N, half_steps, (first,), physical)
    if half_steps == 1:
        first.status = COMPLETED_FIXED_HALF_STEPS
        return CCCCN46Result(COMPLETED_FIXED_HALF_STEPS, None, a, b, D, q, n_x, n_y, N, half_steps, (first,), physical)
    second = HalfStep(s=2, k=1, direction="X", n_s=n_x, h_s=a / n_x, N=N, qr_executed=False)
    try:
        _half_step_2(first, second, a, b, D, q, n_x, n_y)
    except n23._Stop as stop:
        second.status, second.reason = stop.status, stop.reason
        return CCCCN46Result(stop.status, stop.reason, a, b, D, q, n_x, n_y, N, half_steps, (first, second), physical)
    if half_steps == 2:
        second.status = COMPLETED_FIXED_HALF_STEPS
        return CCCCN46Result(COMPLETED_FIXED_HALF_STEPS, None, a, b, D, q, n_x, n_y, N, half_steps, (first, second), physical)
    third = _guard_3(second, b, n_y)
    return CCCCN46Result(third.status, third.reason, a, b, D, q, n_x, n_y, N, half_steps, (first, second, third), physical)


def domain_refusal(a, b, n_x, n_y, N, half_steps) -> str | None:
    """Why the input is outside the domain of D-067, or None. Values that are not valid numbers are left to the input checks."""
    if not n1._is_integer(N) or N not in N_VALUES:
        return f"N должно быть 4, 5 или 6, получено {N!r}; N = 1 — команда run-cccc-n1, N = 2, 3 — run-cccc-n23"
    if not n1._is_integer(half_steps) or not S_MIN <= half_steps <= S_MAX:
        return f"число полушагов должно быть целым от {S_MIN} до {S_MAX}, получено {half_steps!r}"
    low = grid_min(int(N))
    for name, value in (("n_x", n_x), ("n_y", n_y)):
        if n1._is_integer(value) and not low <= value <= N_MAX:
            return f"{name} должно быть от {low} (2N + 1 при N = {N}) до {N_MAX}, получено {value!r}"
    return n1.domain_refusal(a, b, n_x, n_y, half_steps)  # 0,2 <= b/a <= 5; its grid bound 7 is below 2N + 1


def _not_run(status, reason, a, b, D, q, n_x, n_y, N, half_steps, physical=None) -> CCCCN46Result:
    return CCCCN46Result(status, reason, a, b, D, q, n_x, n_y, N, half_steps, (), physical)


def run_cccc_n46(a, b, D, q, n_x, n_y, N, half_steps) -> CCCCN46Result:
    """Plate a x b, stiffness D, load q != 0, clamped on all sides, N = 4, 5 or 6: s = 1; s = 2 under RN-5 and RT-7 (D-068); s >= 3 guarded."""
    reason = domain_refusal(a, b, n_x, n_y, N, half_steps)
    if reason is not None:
        return _not_run(NOT_EXECUTED_OUTSIDE_DOMAIN, reason, a, b, D, q, n_x, n_y, N, half_steps)
    for name, value in (("a", a), ("b", b), ("D", D)):
        if not n1._is_positive_finite(value):
            return _not_run(ERROR_INPUT, f"{name} должно быть конечным и положительным, получено {value!r}", a, b, D, q, n_x, n_y, N, half_steps)
    if isinstance(q, bool) or not isinstance(q, (int, float, np.integer, np.floating)) or not math.isfinite(q) or q == 0:
        return _not_run(ERROR_INPUT, f"q должно быть конечным и ненулевым, получено {q!r}", a, b, D, q, n_x, n_y, N, half_steps)
    for name, value in (("n_x", n_x), ("n_y", n_y)):
        if not n1._is_integer(value):
            return _not_run(ERROR_INPUT, f"{name} должно быть целым, получено {value!r}", a, b, D, q, n_x, n_y, N, half_steps)
    return _execute(float(a), float(b), float(D), float(q), int(n_x), int(n_y), int(N), int(half_steps))


def run_cccc_n46_physical(a, b, t, E, nu, q, n_x, n_y, N, half_steps) -> CCCCN46Result:
    """The same with the physical input; D = E t^3 / (12 (1 - nu^2)) and the checks of ``validated_rectangle_input``."""
    physical = {"t": t, "E": E, "nu": nu}
    reason = domain_refusal(a, b, n_x, n_y, N, half_steps)
    if reason is not None:
        return _not_run(NOT_EXECUTED_OUTSIDE_DOMAIN, reason, a, b, None, q, n_x, n_y, N, half_steps, physical)
    try:
        plate, D = validated_rectangle_input(a, b, t, E, nu, q, n_x, n_y)
    except PhysicalInputError as exc:
        return _not_run(ERROR_INPUT, str(exc), a, b, None, q, n_x, n_y, N, half_steps, physical)
    return _execute(plate.a, plate.b, D, plate.q, plate.n_x, plate.n_y, int(N), int(half_steps), {"t": plate.t, "E": plate.E, "nu": plate.nu})
