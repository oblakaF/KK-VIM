"""CCCC, N = 2 or 3: the half-steps s = 1 and s = 2 of the coupled KK-MVI, the limited research modes of D-063 and D-065. Never a converged or
qualified solution.

    run_cccc_n23(a, b, D, q, n_x, n_y, N, half_steps)
    run_cccc_n23_physical(a, b, t, E, nu, q, n_x, n_y, N, half_steps)

"Policy" is docs/R5_CCCC_N23_LIMITED_POLICY.proposal.md, whose sections 2-5, 7 and 10 are the specification by D-063.1; "Policy-1" is
docs/R5_CCCC_N1_LIMITED_POLICY.proposal.md. The half-step s = 1 is that of D-063 and is not changed. With half_steps >= 2 the half-step s = 2
follows D-065, 7 (C-G), only for n_x, n_y <= 32 (D-065.5):

    C  exact rank of V (STOP_BEFORE_QR_S1 if < N); the RN-5 certificate on the saved G, V of this run; not certified -> stop before QR
    D  QR of V with the normalisation of Policy 3, step 2 over y (G2), the companion G R^T, so that w(1) = (G R^T) G2^T exactly
    E  RT-7 (R7-1...R7-3) on the saved V and G2; failed -> stop after QR, G2 not used
    F  the X-half-step on G2 (roles of x and y exchanged, ВБC-13 p. 1 "symmetrically"): C-3+ with its premises, B-C, the field X G2^T
    G  half_steps >= 3: stop before the QR that would open s = 3

Every stop before s = 2 is STOPPED_RANK_GUARD_UNCERTIFIED (exit 6) with the field of s = 1 as diagnostic, except a proven dependence
(STOP_BEFORE_QR_S1); it never proves a loss of rank. At s = 2 eta, cond(S), the singular values and the energy are only written: the thresholds
of s = 1 (100u, 16u, c_f = 10) are not transferred (D-065).

| Component                                      | Specification          | Level        | Decision           | Check        |
| w = sum X_i(x) Y_i(y), half-steps              | MATH_SPEC C.3-C.4      | A (authors)  | D-024, D-050       | QA K-3       |
| energy Pi_h^C, system (ЧФ-C1), A, B, C, F       | Policy 2; ВБC-12, -13  | B (project)  | D-049.1, D-050     | B-1, B-4     |
| start T_N (sin^2 basis), orientation           | Policy 1, 3 step 1     | A/B          | D-049.3            | B-3          |
| QR of the start set                            | Policy 3 step 2        | V (numerics) | D-057.1 item 2     | QA K-1       |
| assembly of C (C-3+), premises P-1...P-5       | Policy 3 step 4        | V (numerics) | D-054.3, D-055.1   | B-2          |
| half-step solve B-C (DST-I + rank-2N SMW)      | Policy 3 step 5        | V (numerics) | D-049.2            | QA K-1, K-3  |
| eta and its guard 100u                         | Policy 3 step 6        | V (numerics) | D-063.3 (variant A)| B-5          |
| rank diagnostic, the guard of s >= 2           | Policy 3 step 7, 4     | V (numerics) | D-049.5, D-057.1   | B-5          |
| RN-5 before the QR of V, RT-7 after it         | study 4.6; D-065, 6    | B (project)  | D-065.2, D-065.3   | QA (D-065)   |
| QR of V, companion G R^T                       | R4 NS 5.4, 5.4.1       | V (numerics) | D-065.4            | QA (D-065)   |
| X-half-step s = 2 (C-3+, B-C over x)           | ВБC-13 p. 1; Policy 3  | B / V        | D-065.1            | QA (D-065)   |
| guard of s >= 3                                | D-065, 7 G             | V (numerics) | D-065              | tests        |
| P_hat (Pi_h^C at w = G V^T)                    | Policy 5               | B (project)  | D-049.1            | B-1          |
| mode, statuses                                 | Policy 5, 7            | V (numerics) | D-063.2, D-063.4   | B-5          |

Limits: P-4 is assumed (classical dot products in BLAS); B-C is not qualified on working grids; the field belongs to span(G), the start set
after the rounding of QR; one half-step does not approximate the discrete or the continuous solution (NS 9, O-8); the full rank of w(1) is
UNRESOLVED (N = 3 continuously, any N >= 2 discretely); RN-5 certifies rank w_h(1) = N of one run and RT-7 only the transversality of span G2
to the exact y-subspace, neither the accuracy, the conditioning nor the rank of w(2), nor anything of s >= 3. The guards have no fallback:
no refinement, no other assembly, no A-C, no lower N, no
regularization, no rank tolerance. cond(A), cond(S), the singular values and their ratios are only written. |eta - eta_QA| <= 16u and TS-1
(c_f = 10) are criteria of the QA acceptance of the first version (D-063.3); the solver does not apply them. No kkvim module of the SSSS
scheme is used; the stencils, the rounding probes and the input checks are those of ``kkvim.cccc_n1``, which is not changed.

Statuses never raise: a refused input is a result with its status and reason, without a field.
"""

import math
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np
import scipy.fft
import scipy.linalg

import kkvim.cccc_n1 as n1
import kkvim.cccc_n23_certificate as certificate
from kkvim.physical import PhysicalInputError
from kkvim.rectangle import validated_rectangle_input

MODE = n1.MODE
APPLICATION_STATUS = n1.APPLICATION_STATUS
RANK_GUARD_RULE = ("D-057.1 (вариант В) с исключением D-065: N = 2, 3 — s = 1; s = 2 только при n_x, n_y ≤ 32 после RN-5 до QR и RT-7 после QR; "
                   "s ≥ 3 — остановка до QR")
RANK_DECISION = "UNRESOLVED (no approved numerical rank criterion; D-049.5, D-049.6)"
ORIENTATION = "start T_N in x: U_i = sin^2(i pi x / a), i = 1..N (basis A, D-049.3), companion 0 (w(0) = 0); the first half-step solves Y"

COMPLETED_FIXED_HALF_STEPS = n1.COMPLETED_FIXED_HALF_STEPS
STOPPED_RANK_GUARD_UNCERTIFIED = "STOPPED_RANK_GUARD_UNCERTIFIED"
STOP_BEFORE_QR_S1 = "STOP_BEFORE_QR_S1"
NOT_EXECUTED_OUTSIDE_DOMAIN = n1.NOT_EXECUTED_OUTSIDE_DOMAIN
ERROR_INPUT = n1.ERROR_INPUT
ERROR_NONFINITE = n1.ERROR_NONFINITE
UNRESOLVED_C3PLUS_PREMISE = n1.UNRESOLVED_C3PLUS_PREMISE
PROTECTIVE_STOP_FACTORIZATION = n1.PROTECTIVE_STOP_FACTORIZATION
STOPPED_ETA_GUARD = n1.STOPPED_ETA_GUARD
STOP_ZERO_FIELD = n1.STOP_ZERO_FIELD
HALF_STEP_DONE = n1.HALF_STEP_DONE  # the row s = 1 when a row s = 2 follows; never the status of a run

PREMISES_ESTABLISHED = n1.PREMISES_ESTABLISHED
PREMISES_UNRESOLVED = n1.PREMISES_UNRESOLVED
P4_ASSUMED = n1.P4_ASSUMED

U = n1.U
ETA_MAX_C = 100.0 * U  # D-063.3, variant A: the guard of this mode only, not a qualified value
N_VALUES = (2, 3)  # D-063.2
N_MIN, N_MAX = n1.N_MIN, n1.N_MAX  # D-063.2: 7 <= n_x, n_y <= 1280, bounds of the domain, not tolerances
S_MIN, S_MAX = n1.S_MIN, n1.S_MAX  # D-063.2: the number of half-steps asked for; at most s = 2 is executed (D-065)
S2_GRID_MAX = 32  # D-065.5: s = 2 only for n_x, n_y <= 32; a bound of the development, not a certification of these grids
K_EXTRA = n1.K_EXTRA  # k = m + 13 of the bound of C-3+
SMALLEST_NORMAL = n1.SMALLEST_NORMAL
DBL_MAX = n1.DBL_MAX


@dataclass
class HalfStep:
    """The row s = 1, s = 2 (D-065) or s = 3 (the guard only); what it did not reach stays None.

    At s = 1 G is the start set after QR (x) and V the solved Y factors; at s = 2 G is G2, the solved V of s = 1 after QR (y), and V the
    solved X factors. The field of a done row is ``field_factors()``: (G, V) at s = 1, (V, G) at s = 2 (axis 0 = x).
    """

    s: int
    k: int
    direction: str  # the solved factors: Y at s = 1, X at s = 2, Y at s = 3
    n_s: int
    h_s: float
    N: int
    status: str = HALF_STEP_DONE
    reason: str | None = None
    done: bool = False  # s = 1 solved and its guards passed
    conditions: dict | None = None  # Policy 1: N <= floor(n_x/2), 2N < n_x, N <= floor(n_y/2); at s >= 2 S-1 of RN-3 and the grid of D-065.5
    T: np.ndarray | None = None  # s = 1: the start set
    exact_rank_T: int | None = None
    R: np.ndarray | None = None
    G: np.ndarray | None = None  # the given set after QR
    exact_rank_G: int | None = None
    R_diag_min_abs: float | None = None
    cond_A_before_QR: float | None = None
    cond_A_after_QR: float | None = None
    A: np.ndarray | None = None
    B: np.ndarray | None = None
    C: np.ndarray | None = None
    F: np.ndarray | None = None
    C_L: np.ndarray | None = None
    E: np.ndarray | None = None
    premises_status: str | None = None
    premises_reasons: tuple = ()
    cond_S: float | None = None
    min_eig_DM: float | None = None
    eta: float | None = None
    V: np.ndarray | None = None  # the solved factors: Y (n_y - 1, N) at s = 1, X (n_x - 1, N) at s = 2
    V_exact_rank: int | None = None
    sigma: tuple | None = None
    rho_N: float | None = None
    kappa_rep: float | None = None
    rank_decision: str | None = None
    qr_executed: bool | None = None
    P_hat: float | None = None
    alpha_centre: float | None = None
    given: np.ndarray | None = None  # s >= 2: the solved set of the previous half-step before QR
    companion: np.ndarray | None = None  # s = 2: G(s = 1) R^T, the x-factor of w(1) in the basis G2
    rn5: dict | None = None  # s = 2: ``cccc_n23_certificate.rn5_certificate``
    rt7: dict | None = None  # s = 2: ``cccc_n23_certificate.rt7_conditions``
    P_hat_previous_QR: float | None = None  # s = 2: Pi_h^C of w(1) from the transformed arrays (companion, G2); OBSERVATION
    field_preservation: float | None = None  # s = 2: max|companion G2^T - G V^T| / max|G V^T| of s = 1; OBSERVATION

    def field_factors(self):
        """(x factor, y factor) of the field of this row."""
        return (self.G, self.V) if self.s == 1 else (self.V, self.G)


@dataclass(frozen=True)
class CCCCN23Result:
    """The run; ``status`` of Policy 5. Every result is RESEARCH_ONLY / PROVISIONAL / NOT_QUALIFIED."""

    status: str
    reason: str | None
    a: object
    b: object
    D: float | None
    q: object
    n_x: object
    n_y: object
    N: object
    half_steps_requested: object
    steps: tuple = ()
    physical: dict | None = None  # t, E, nu of run_cccc_n23_physical
    mode: str = field(default=MODE)
    application_status: str = field(default=APPLICATION_STATUS)
    rank_guard_rule: str = field(default=RANK_GUARD_RULE)

    @property
    def completed(self) -> tuple:
        return tuple(step for step in self.steps if step.done)

    @property
    def half_steps_completed(self) -> int:
        return len(self.completed)

    def factors(self):
        """(x factor, y factor) of the last completed half-step on the inner nodes, or None: (G, V) at s = 1, (X, G2) at s = 2."""
        if not self.completed:
            return None
        return self.completed[-1].field_factors()

    def field(self):
        """w of the last completed half-step with the boundary nodes (axis 0 = x, axis 1 = y), or None."""
        factors = self.factors()
        return None if factors is None else _field(*factors)

    def dimensionless_field(self):
        """w_bar = w D / (q a^4), or None."""
        w = self.field()
        return None if w is None else w * self.D / (self.q * self.a**4)


class _Stop(Exception):
    """A row of the table of Policy 5: the status and the reason."""

    def __init__(self, status: str, reason: str):
        super().__init__(reason)
        self.status, self.reason = status, reason


# --- exact ranks ---------------------------------------------------------------------------------------------------------------

def _exact_determinant(rows: list) -> Fraction:
    rows = [list(row) for row in rows]
    n, det = len(rows), Fraction(1)
    for col in range(n):
        pivot = next((r for r in range(col, n) if rows[r][col] != 0), None)
        if pivot is None:
            return Fraction(0)
        if pivot != col:
            rows[col], rows[pivot] = rows[pivot], rows[col]
            det = -det
        det *= rows[col][col]
        for r in range(col + 1, n):
            if rows[r][col] != 0:
                factor = rows[r][col] / rows[col][col]
                rows[r] = [x - factor * y for x, y in zip(rows[r], rows[col], strict=True)]
    return det


def _eliminate(M: np.ndarray) -> int:
    rows = [[Fraction(float(value)) for value in column] for column in M.T]
    rank = 0
    for col in range(M.shape[0]):
        pivot = next((r for r in range(rank, len(rows)) if rows[r][col] != 0), None)
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        for r in range(rank + 1, len(rows)):
            if rows[r][col] != 0:
                factor = rows[r][col] / rows[rank][col]
                rows[r] = [x - factor * y for x, y in zip(rows[r], rows[rank], strict=True)]
        rank += 1
        if rank == len(rows):
            break
    return rank


def exact_rank(M) -> int:
    """The rank of the stored doubles in exact rational arithmetic, without a threshold (Policy 3, step 1).

    A nonzero N x N minor in Fraction proves the full column rank; pivoted QR only chooses which rows to try. Otherwise the elimination
    over all rows decides.
    """
    M = np.asarray(M, dtype=float)
    if M.ndim != 2 or M.shape[1] == 0 or not np.any(M):
        return 0
    m, N = M.shape
    if m >= N and np.all(np.isfinite(M)):
        _, _, order = scipy.linalg.qr(M.T, mode="economic", pivoting=True)
        rows = sorted(order[:N].tolist())
        if _exact_determinant([[Fraction(float(M[r, j])) for j in range(N)] for r in rows]) != 0:
            return N
    return _eliminate(M)


# --- the start, QR and the operators of Policy 2 ---------------------------------------------------------------------------------

def start_set(n_x: int, N: int) -> np.ndarray:
    """T: U_i = sin^2(i pi l / n_x), l = 1..n_x - 1, i = 1..N (basis A, D-049.3; Policy 3, step 1). Its bits depend on the platform."""
    l = np.arange(1, n_x)  # noqa: E741 - the name of the policy
    return np.column_stack([np.sin(i * math.pi * l / n_x) ** 2 for i in range(1, N + 1)])


def orthonormalise(T: np.ndarray, h: float):
    """(G, R) with sqrt(h) T = Q R (Householder, LAPACK) and G = Q / sqrt(h): another basis of the same span (Policy 3, step 2)."""
    Q, R = np.linalg.qr(math.sqrt(h) * T)
    return Q / math.sqrt(h), R


def second_difference(V: np.ndarray, h: float) -> np.ndarray:
    """L V by columns on the inner nodes, the boundary values being zero."""
    P = np.zeros((V.shape[0] + 2, V.shape[1]))
    P[1:-1] = V
    return (P[:-2] - 2.0 * P[1:-1] + P[2:]) / h**2


def d4_apply(V: np.ndarray, h: float) -> np.ndarray:
    """D4 V by columns: inner rows (1, -4, 6, -4, 1) / h^4, the first and the last row (7, -4, 1) / h^4 (as ``cccc_n1.d4_apply``)."""
    P = np.zeros((V.shape[0] + 4, V.shape[1]))
    P[2:-2] = V
    out = P[:-4] - 4.0 * P[1:-3] + 6.0 * P[2:-2] - 4.0 * P[3:-1] + P[4:]
    out[0] = 7.0 * V[0] - 4.0 * V[1] + V[2]
    out[-1] = 7.0 * V[-1] - 4.0 * V[-2] + V[-3]
    return out / h**4


def assemble(G: np.ndarray, l_f: float, n_f: int, q: float) -> dict:
    """A, B, C (by C-3+), F of the set G (columns) and the intermediates of C for the premises (Policy 3, step 4)."""
    with np.errstate(all="ignore"):
        hf = l_f / n_f
        A = hf * (G.T @ G)
        P = np.zeros((G.shape[0] + 2, G.shape[1]))
        P[1:-1] = G
        LG = (P[:-2] - 2.0 * P[1:-1] + P[2:]) / hf**2
        B = -hf * (G.T @ LG)
        F = q * hf * G.sum(axis=0)
        Y3 = np.column_stack([n1._third_order_sums(G[:, j]) for j in range(G.shape[1])])
        h2 = hf * hf
        h3 = h2 * hf
        s = 2.0 / h3
        X = Y3 / h2
        C_L = hf * (X.T @ X)
        E = s * (np.outer(G[0], G[0]) + np.outer(G[-1], G[-1]))
        C = C_L + E
    return {"A": A, "B": B, "C": C, "F": F, "C_L": C_L, "E": E, "Y3": Y3, "X": X, "g1": G[0].copy(), "gm": G[-1].copy(),
            "h": hf, "h2": h2, "h3": h3, "s": s}


def premise_failures(parts: dict, n_f: int) -> list:
    """P-2, P-3 and P-5 of Policy 3, step 4, element by element. P-1 is ``cccc_n1.rounding_probe_failures``; P-4 cannot be checked."""
    failed = []
    for name in ("Y3", "X", "g1", "gm", "C", "C_L", "E", "h", "h2", "h3", "s"):
        value = np.atleast_1d(parts[name])
        if not np.all(np.isfinite(value)):
            failed.append(f"P-2: {name} не конечно")
            continue
        nonzero = np.abs(value[value != 0.0])
        if nonzero.size and float(nonzero.min()) < SMALLEST_NORMAL:
            failed.append(f"P-3: {name} субнормально")
    if failed:
        return failed
    for name in ("h", "h2", "h3"):
        if parts[name] == 0.0:
            failed.append(f"P-3: {name} = 0")
    m = n_f - 1
    x = np.abs(parts["X"])
    g1, gm = np.abs(parts["g1"]), np.abs(parts["gm"])
    with np.errstate(all="ignore"):
        if not m * float(x.max()) ** 2 < DBL_MAX:
            failed.append("P-2: m max|X|^2 не меньше DBL_MAX")
        if not 2.0 * max(float(g1.max()) ** 2, float(gm.max()) ** 2) * parts["s"] < DBL_MAX:
            failed.append("P-2: 2 max(‖G[0]‖∞^2, ‖G[-1]‖∞^2) s не меньше DBL_MAX")
        for name, value in (("X", x), ("G[0]", g1), ("G[-1]", gm)):
            nonzero = value[value != 0.0]
            if not nonzero.size:
                continue
            square = float(nonzero.min()) ** 2
            if square < SMALLEST_NORMAL:
                failed.append(f"P-3: квадрат наименьшего ненулевого |{name}| ниже 2^-1022")
            elif name != "X" and square * parts["s"] < SMALLEST_NORMAL:
                failed.append(f"P-3: произведение квадрата наименьшего ненулевого |{name}| и 2/h^3 ниже 2^-1022")
        if not np.array_equal(parts["C"], parts["C_L"] + parts["E"]):
            failed.append("P-2, P-3: C побитово не равно fl(C_L + E)")
    if not (m + K_EXTRA) * U < 1.0:
        failed.append("P-5: k u не меньше 1")
    return failed


# --- the half-step: B-C with the correction of rank 2N (Policy 3, step 5) -----------------------------------------------------------

def _dst(v: np.ndarray, axis: int = -1) -> np.ndarray:
    return scipy.fft.dst(v, type=1, norm="ortho", axis=axis)


def solve_bc(step: HalfStep, A: np.ndarray, B: np.ndarray, C: np.ndarray, F: np.ndarray, D: float, l_s: float, n_s: int) -> np.ndarray:
    """V of D [A ⊗ D4 - 2 B ⊗ L + C ⊗ I] vec(V) = F ⊗ 1, vec(V) = [V_1; ...; V_N]: DST-I and Sherman-Morrison-Woodbury of rank 2N."""
    N = A.shape[0]
    h = l_s / n_s
    m = n_s - 1
    k = np.arange(1, n_s)
    with np.errstate(all="ignore"):
        mu = 4.0 / h**2 * np.sin(k * math.pi / (2 * n_s)) ** 2
        DM = D * (mu[:, None, None] ** 2 * A + 2.0 * mu[:, None, None] * B + C)  # (m, N, N)
        if not np.all(np.isfinite(DM)):
            raise _Stop(ERROR_NONFINITE, "DM не конечно")
        try:
            np.linalg.cholesky(DM)
            np.linalg.cholesky(A)
        except np.linalg.LinAlgError as exc:
            raise _Stop(PROTECTIVE_STOP_FACTORIZATION, f"Холецкий DM или A: {exc}") from exc
        step.min_eig_DM = float(np.linalg.eigvalsh(DM / D).min())
        rhs = F[None, :] * _dst(np.ones(m))[:, None]
        x0 = _dst(np.linalg.solve(DM, rhs[:, :, None])[:, :, 0].T, axis=1).reshape(-1)
        rows = [i * m + p for i in range(N) for p in (0, m - 1)]
        Minv = np.linalg.solve(DM, np.broadcast_to(np.eye(N), DM.shape))
        z = [_dst(np.eye(1, m, p)[0]) for p in (0, m - 1)]
        Y0 = np.empty((N * m, 2 * N))
        for i in range(N):
            for side in (0, 1):
                Y0[:, 2 * i + side] = _dst(Minv[:, :, i].T * z[side][None, :], axis=1).reshape(-1)
        Ginv = h**4 / (2.0 * D) * np.kron(np.linalg.inv(A), np.eye(2))
        psi1 = math.sqrt(2.0 / n_s) * np.sin(k * math.pi / n_s)
        zk = np.stack([psi1, np.where(k % 2 == 1, 1.0, -1.0) * psi1], axis=1)
        S = Ginv + np.einsum("kij,ks,kt->isjt", Minv, zk, zk).reshape(2 * N, 2 * N)
        if not np.all(np.isfinite(S)):
            raise _Stop(ERROR_NONFINITE, "S не конечно")
        try:
            c = np.linalg.solve(S, x0[rows])
        except np.linalg.LinAlgError as exc:
            raise _Stop(PROTECTIVE_STOP_FACTORIZATION, f"S: {exc}") from exc
        step.cond_S = float(np.linalg.cond(S))
        return (x0 - Y0 @ c).reshape(N, m).T


def _norm_inf(M: np.ndarray) -> float:
    """The largest sum of the absolute values in a row."""
    return float(np.max(np.sum(np.abs(M), axis=1)))


def residual_eta(V: np.ndarray, A: np.ndarray, B: np.ndarray, C: np.ndarray, F: np.ndarray, D: float, h: float) -> float:
    """The relative residual of the original system, without the modal transformation (Policy 3, step 6)."""
    with np.errstate(all="ignore"):
        r = F[None, :] - D * (d4_apply(V, h) @ A.T - 2.0 * second_difference(V, h) @ B.T + V @ C.T)
        scale = D * (16 * _norm_inf(A) / h**4 + 8 * _norm_inf(B) / h**2 + _norm_inf(C)) * np.max(np.abs(V)) + np.max(np.abs(F))
        return float(np.max(np.abs(r)) / scale)


def rank_diagnostic(G: np.ndarray, h_g: float, V: np.ndarray, h_v: float):
    """(sigma_1 >= ... >= sigma_N of R_G R_V^T, rho_N = sigma_N / sigma_1, kappa_rep = |R_G|_2 |R_V|_2 / sigma_1); OBSERVATION only."""
    with np.errstate(all="ignore"):
        R_G = np.linalg.qr(math.sqrt(h_g) * G, mode="r")
        R_V = np.linalg.qr(math.sqrt(h_v) * V, mode="r")
        sigma = np.linalg.svd(R_G @ R_V.T, compute_uv=False)
        return (tuple(float(value) for value in sigma), float(sigma[-1] / sigma[0]),
                float(np.linalg.norm(R_G, 2) * np.linalg.norm(R_V, 2) / sigma[0]))


def discrete_energy(G: np.ndarray, V: np.ndarray, a: float, b: float, D: float, q: float, n_x: int, n_y: int) -> float:
    """P_hat = Pi_h^C at w = G V^T: 1/2 D sum_ij (C^x A^y + 2 B^x B^y + A^x C^y)_ij - q sum_i (h_x 1^T G_i)(h_y 1^T V_i).

    The coefficients of x are of G, those of y of V, both by ``assemble`` (Policy 5; derivation of the project, level B). At N = 1 it is the
    formula of Policy-1, section 6.
    """
    x, y = assemble(G, a, n_x, 1.0), assemble(V, b, n_y, 1.0)
    bending = 0.5 * D * float(np.sum(x["C"] * y["A"] + 2.0 * x["B"] * y["B"] + x["A"] * y["C"]))
    return bending - q * float(np.sum((x["h"] * G.sum(axis=0)) * (y["h"] * V.sum(axis=0))))


def _field(G: np.ndarray, V: np.ndarray) -> np.ndarray:
    w = np.zeros((G.shape[0] + 2, V.shape[0] + 2))
    w[1:-1, 1:-1] = G @ V.T
    return w


def start_conditions(N: int, n_x: int, n_y: int) -> dict:
    """The conditions of Policy 1; in the domain they all hold (a property tested for every n)."""
    return {"N <= floor(n_x/2)": N <= n_x // 2, "2N < n_x": 2 * N < n_x, "N <= floor(n_y/2)": N <= n_y // 2}


# --- s = 1 and the guard of s = 2 -------------------------------------------------------------------------------------------------

def _half_step_1(step: HalfStep, a: float, b: float, D: float, q: float, n_x: int, n_y: int) -> None:
    """Policy 3, steps 1-8; a row of the table of Policy 5 raises ``_Stop``."""
    N = step.N
    step.conditions = start_conditions(N, n_x, n_y)
    step.T = start_set(n_x, N)
    step.exact_rank_T = exact_rank(step.T)
    if step.exact_rank_T < N:
        raise _Stop(STOP_BEFORE_QR_S1, f"точный ранг стартового набора {step.exact_rank_T} < N = {N} (D-049.5, п. 2)")
    hf = a / n_x
    step.G, step.R = orthonormalise(step.T, hf)
    step.qr_executed = True
    step.R_diag_min_abs = float(np.min(np.abs(np.diag(step.R))))
    singular = np.linalg.svd(step.R, compute_uv=False)
    step.cond_A_before_QR = float((singular[0] / singular[-1]) ** 2) if singular[-1] > 0 else math.inf
    step.exact_rank_G = exact_rank(step.G)
    if not np.all(np.diag(step.R) != 0.0) or step.exact_rank_G < N:
        raise _Stop(PROTECTIVE_STOP_FACTORIZATION, f"после QR: min|diag R| = {step.R_diag_min_abs!r}, точный ранг G = {step.exact_rank_G}")
    _solve_given(step, a, n_x, b, n_y, D, q, eta_guard=True)


def _solve_given(step: HalfStep, l_g: float, n_g: int, l_s: float, n_s: int, D: float, q: float, eta_guard: bool) -> None:
    """Policy 3, steps 4-8 on the given set step.G (direction of l_g, n_g), solving the other direction (l_s, n_s).

    At s = 1 (x given, y solved) this is the half-step of D-063 operation by operation; at s = 2 the roles of x and y are exchanged
    (ВБC-13 p. 1) and the guard of eta is not applied (D-065: the thresholds of s = 1 are not transferred).
    """
    hf = l_g / n_g
    failures = n1.rounding_probe_failures()
    parts = assemble(step.G, l_g, n_g, q)
    step.A, step.B, step.C, step.F, step.C_L, step.E = (parts[name] for name in ("A", "B", "C", "F", "C_L", "E"))
    if not all(np.all(np.isfinite(value)) for value in (step.A, step.B, step.F)):
        raise _Stop(ERROR_NONFINITE, "A, B или F не конечно")
    eigenvalues = np.linalg.eigvalsh(step.A)
    step.cond_A_after_QR = float(eigenvalues[-1] / eigenvalues[0]) if eigenvalues[0] > 0 else math.inf
    failures += premise_failures(parts, n_g)
    step.premises_status = PREMISES_UNRESOLVED if failures else PREMISES_ESTABLISHED
    step.premises_reasons = tuple(failures)
    if failures:
        raise _Stop(UNRESOLVED_C3PLUS_PREMISE, "; ".join(failures))
    V = solve_bc(step, step.A, step.B, step.C, step.F, D, l_s, n_s)
    failures = n1.rounding_probe_failures()
    if failures:
        step.premises_status, step.premises_reasons = PREMISES_UNRESOLVED, tuple(failures)
        raise _Stop(UNRESOLVED_C3PLUS_PREMISE, "; ".join(failures))
    if not np.all(np.isfinite(V)):
        raise _Stop(ERROR_NONFINITE, "V не конечно")
    step.V = V
    if not np.any(V):  # eta is not defined at V = 0
        raise _Stop(STOP_ZERO_FIELD, "V = 0 точно")
    step.eta = residual_eta(V, step.A, step.B, step.C, step.F, D, step.h_s)
    if not math.isfinite(step.eta):
        raise _Stop(ERROR_NONFINITE, f"η = {step.eta!r}")
    if eta_guard and not step.eta <= ETA_MAX_C:
        raise _Stop(STOPPED_ETA_GUARD, f"η = {step.eta!r} > η_max^C = 100u (D-063.3)")
    step.V_exact_rank = exact_rank(V)
    step.sigma, step.rho_N, step.kappa_rep = rank_diagnostic(step.G, hf, V, l_s / n_s)
    step.rank_decision = RANK_DECISION
    x, y = step.field_factors()
    a, b, n_x, n_y = (l_g, l_s, n_g, n_s) if step.s == 1 else (l_s, l_g, n_s, n_g)
    step.P_hat = discrete_energy(x, y, a, b, D, q, n_x, n_y)
    if n_x % 2 == 0 and n_y % 2 == 0:
        step.alpha_centre = float(_field(x, y)[n_x // 2, n_y // 2]) * D / (q * min(a, b) ** 4)
    step.done = True


def _half_step_2(first: HalfStep, step: HalfStep, a: float, b: float, D: float, q: float, n_x: int, n_y: int) -> None:
    """D-065, 7 C-F after a done s = 1; a stop raises ``_Stop``. Nothing of s = 1 is changed."""
    N = step.N
    step.given = first.V
    step.conditions = {"S-1 (RN-3): N > floor(n_x/2)": N > n_x // 2, f"n_x, n_y <= {S2_GRID_MAX} (D-065.5)": max(n_x, n_y) <= S2_GRID_MAX}
    step.rank_decision = RANK_DECISION
    if first.V_exact_rank < N:
        raise _Stop(STOP_BEFORE_QR_S1, f"точный ранг решённого набора V {first.V_exact_rank} < N = {N} (D-049.5, п. 2)")
    if max(n_x, n_y) > S2_GRID_MAX:
        raise _Stop(STOPPED_RANK_GUARD_UNCERTIFIED, f"N = {N}, s = 2: сетка {n_x} × {n_y} вне области D-065.5 (n_x, n_y ≤ {S2_GRID_MAX}); "
                    "сертификат не строится, QR не выполняется (D-057.1, п. 2)")
    step.rn5 = certificate.rn5_certificate(a, b, n_x, n_y, N, first.G, first.V, D, q)
    if not step.rn5["certified"]:
        raise _Stop(STOPPED_RANK_GUARD_UNCERTIFIED, f"RN-5 не выполнен ({step.rn5['verdict']}: {step.rn5['reason']}); QR решённого набора не "
                    "выполняется. N не понижается, регуляризации нет (D-065, 7 C)")
    step.G, step.R = orthonormalise(first.V, b / n_y)
    step.qr_executed = True
    step.R_diag_min_abs = float(np.min(np.abs(np.diag(step.R))))
    singular = np.linalg.svd(step.R, compute_uv=False)
    step.cond_A_before_QR = float((singular[0] / singular[-1]) ** 2) if singular[-1] > 0 else math.inf
    step.companion = first.G @ step.R.T
    step.rt7 = certificate.rt7_conditions(first.V, step.G, step.rn5)
    if step.rt7["ranks"] is not None:
        step.exact_rank_G = min(step.rt7["ranks"]["G2"])
    if not step.rt7["passed"]:
        raise _Stop(STOPPED_RANK_GUARD_UNCERTIFIED, f"RT-7 не выполнен после QR ({step.rt7['reason']}); G2 не используется (D-065, 7 E)")
    with np.errstate(all="ignore"):
        w1 = first.G @ first.V.T
        step.field_preservation = float(np.max(np.abs(step.companion @ step.G.T - w1)) / np.max(np.abs(w1)))
    step.P_hat_previous_QR = discrete_energy(step.companion, step.G, a, b, D, q, n_x, n_y)
    _solve_given(step, b, n_y, a, n_x, D, q, eta_guard=False)


def _guard_3(second: HalfStep, b: float, n_y: int) -> HalfStep:
    """The half-step s = 3 up to its guard (D-065, 7 G): no QR of X, no assembly, no solve."""
    N = second.N
    step = HalfStep(s=3, k=1, direction="Y", n_s=n_y, h_s=b / n_y, N=N, given=second.V, qr_executed=False)
    step.conditions = {"S-1 (RN-3): N > floor(n_y/2)": N > n_y // 2}  # impossible in the domain; written only
    if second.V_exact_rank < N:
        step.status, step.reason = STOP_BEFORE_QR_S1, f"точный ранг решённого набора X {second.V_exact_rank} < N = {N} (D-049.5, п. 2)"
    else:
        step.status = STOPPED_RANK_GUARD_UNCERTIFIED
        step.reason = (f"N = {N}, s = 3: защитная остановка до QR решённого набора X; полушаги s ≥ 3 не разрешены (D-065). N не понижается, "
                       "регуляризации и рангового допуска нет.")
    step.rank_decision = RANK_DECISION
    return step


def _execute(a: float, b: float, D: float, q: float, n_x: int, n_y: int, N: int, half_steps: int, physical=None) -> CCCCN23Result:
    first = HalfStep(s=1, k=0, direction="Y", n_s=n_y, h_s=b / n_y, N=N)
    steps = [first]
    try:
        _half_step_1(first, a, b, D, q, n_x, n_y)
    except _Stop as stop:
        first.status, first.reason = stop.status, stop.reason
        return CCCCN23Result(stop.status, stop.reason, a, b, D, q, n_x, n_y, N, half_steps, tuple(steps), physical)
    if half_steps == 1:
        first.status = COMPLETED_FIXED_HALF_STEPS
        return CCCCN23Result(COMPLETED_FIXED_HALF_STEPS, None, a, b, D, q, n_x, n_y, N, half_steps, tuple(steps), physical)
    second = HalfStep(s=2, k=1, direction="X", n_s=n_x, h_s=a / n_x, N=N, qr_executed=False)
    steps.append(second)
    try:
        _half_step_2(first, second, a, b, D, q, n_x, n_y)
    except _Stop as stop:
        second.status, second.reason = stop.status, stop.reason
        return CCCCN23Result(stop.status, stop.reason, a, b, D, q, n_x, n_y, N, half_steps, tuple(steps), physical)
    if half_steps == 2:
        second.status = COMPLETED_FIXED_HALF_STEPS
        return CCCCN23Result(COMPLETED_FIXED_HALF_STEPS, None, a, b, D, q, n_x, n_y, N, half_steps, tuple(steps), physical)
    third = _guard_3(second, b, n_y)
    steps.append(third)
    return CCCCN23Result(third.status, third.reason, a, b, D, q, n_x, n_y, N, half_steps, tuple(steps), physical)


# --- refusals and the two entries ------------------------------------------------------------------------------------------------

def domain_refusal(a, b, n_x, n_y, N, half_steps) -> str | None:
    """Why the input is outside the domain of D-063.2, or None. Values that are not valid numbers are left to the input checks."""
    if not n1._is_integer(N) or N not in N_VALUES:
        return (f"N должно быть 2 или 3, получено {N!r}: эта команда принимает только N = 2, 3; N = 1 — команда run-cccc-n1 (D-061); "
                "N = 4, 5, 6 — отдельная исследовательская команда run-cccc-n46, только полушаг s = 1 (D-067)")
    return n1.domain_refusal(a, b, n_x, n_y, half_steps)


def _not_run(status, reason, a, b, D, q, n_x, n_y, N, half_steps, physical=None) -> CCCCN23Result:
    return CCCCN23Result(status, reason, a, b, D, q, n_x, n_y, N, half_steps, (), physical)


def run_cccc_n23(a, b, D, q, n_x, n_y, N, half_steps) -> CCCCN23Result:
    """Plate a x b, stiffness D, load q != 0, clamped on all sides, N = 2 or 3: s = 1; s = 2 under D-065; s >= 3 only up to its guard."""
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


def run_cccc_n23_physical(a, b, t, E, nu, q, n_x, n_y, N, half_steps) -> CCCCN23Result:
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
