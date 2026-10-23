"""QA research prototype of the B-C half-step path for CCCC (R5-V; preparation of the Q3 measurements E-4, E-6).

A QA tool, not the working solver: no import of ``kkvim`` (D-028), nothing of Builder or of the R4 solver is used, no dispatcher and
no user API. Basis: D-049.2 (B-C main path, order NS 5.1), docs/R5_NUMERICAL_SPEC_DRAFT.md (NS) 4.2-4.3, 4.7, 5, 5.1;
docs/R4_NUMERICAL_SPEC.md 3.2 (ChF-3, modal form) and 5.4 (QR with the grid weight); plan PL 2 (BC-3...BC-9).

Half-step system (NS 4.3, ChF-C1), term-first ordering vec(S) = [S_:1; ...; S_:N], m = n_s - 1, h = h_s (solved direction):

    K^C = D [A (x) D4 - 2 B (x) L + C (x) I] = K0 + Z G Z^T,
    K0  = D [A (x) L^2 - 2 B (x) L + C (x) I],   Z = I_N (x) [e_1, e_m],   G = D A (x) (2/h^4) I_2           (NS 5, B-C)

because D4 = L^2 + (2/h^4)(e_1 e_1^T + e_m e_m^T) (NS 3.1). With L = -Psi Lambda Psi^T (DST-I, psi_k(i) = sqrt(2/n) sin(k pi i/n),
mu_k = (4/h^2) sin^2(k pi/(2n)) > 0; R4 3.2) K0 = D sum_k M_k (x) psi_k psi_k^T, M_k = mu_k^2 A + 2 mu_k B + C (the sign "+2 mu_k B"
comes from -2B (-mu_k)). B-C (Sherman-Morrison-Woodbury):

    x0 = K0^-1 b,  Y0 = K0^-1 Z (2N modal solutions),  S = G^-1 + Z^T Y0 (2N x 2N),  x = x0 - Y0 S^-1 Z^T x0.

Algebraic equivalence with ChF-C1 - NS 5 (review 4b18761, 4.1). Level of this module: B (own discretisation, as specified) and C
(engineering: scipy.fft.dst type 1 with norm="ortho" for Psi, numpy LAPACK for the N x N and 2N x 2N systems and for the QR).

Order of a half-step (NS 5.1; D-049.2, item 3), ``run_halfstep``:
1. rank: exact rank of the given set (no threshold); an exactly dependent set stops before QR and before the solve;
2. QR (N >= 2, on request): sqrt(h_f) G = Q R (Householder, numpy.linalg.qr), U~ = Q/sqrt(h_f), A(U~) = I in exact arithmetic
   (R4 5.4); N = 1: no QR (NS 5.1);
3. diagnostics, values only: cond(A) before QR = cond(R)^2, cond(A) after QR, the exact orthonormality defect max|h_f U~^T U~ - I|,
   the angle between span(U~) and span(G), cond(S) of both capacitance paths and their difference, max cond(M_k), and the factors
   of the corrected model VOK-13 (fe5d60f, f62cd79): rho_0 = ||x0||/||x||, kappa_bar_c = ||Y0||_2 ||c||_2/||x||, ||M||_2 (VOK-12,
   item 4); the former kappa_c = ||Y0 c||/||x|| is logged as ratio_Y0c_x (review 9f0d5dc, PB-2, PB-3). Every QR run is RESEARCH_ONLY
   until a numerical rank criterion is qualified (PB-4). q = 0: x = 0, undefined ratios are UNRESOLVED (PB-1);
4. B-C solve;
5. residual of the original operator with D4: eta_QA and the exact residual (R-HP, kkvim_reference.r5_cccc_halfstep).

Comparisons (``compare``): R-AC (with refinement) and, on small grids, R-H on the SAME system (the given set actually solved). The
accepted strict certificates of r5_cccc_halfstep (reviews 57d3194, 4f92c82) are applied only in their proven domain: exact system
from the input doubles of that given set, n_f, n_s >= 3, certified lambda_low; otherwise UNRESOLVED. They bound the error of the SET
S of that system, not of the field W = G S^T and not of the system of the original set before QR (the computed QR is not exactly
U R^-1). No thresholds are introduced; numbers are OBSERVATION.
"""

import math
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np
import scipy.fft
import scipy.linalg

from kkvim_reference import r5_cccc_halfstep as hs

UNRESOLVED = "UNRESOLVED"


# --- coefficients (own path: stencils applied to the given vectors, not the dense operator matrices of R-AC) -------------

def _apply_L(G: np.ndarray, h: float) -> np.ndarray:
    """L G column by column: (g_{i-1} - 2 g_i + g_{i+1})/h^2, Dirichlet zeros outside."""
    P = np.zeros((G.shape[0] + 2, G.shape[1]))
    P[1:-1] = G
    return (P[:-2] - 2.0 * P[1:-1] + P[2:]) / h**2


def _apply_D4(G: np.ndarray, h: float) -> np.ndarray:
    """D4 G by the stencil (1, -4, 6, -4, 1)/h^4, first and last rows (7, -4, 1)/h^4 (NS 3.1; n >= 3)."""
    P = np.zeros((G.shape[0] + 4, G.shape[1]))
    P[2:-2] = G
    diag = np.full((G.shape[0], 1), 6.0)
    diag[0] = diag[-1] = 7.0
    return (P[:-4] - 4.0 * P[1:-3] + diag * P[2:-2] - 4.0 * P[3:-1] + P[4:]) / h**4


def bc_coefficients(p: hs.HalfStep):
    """A, B, C^C, F of NS 4.2 in floating point (stencil path)."""
    h = p.l_f / p.n_f
    G = p.given
    A = h * (G.T @ G)
    B = -h * (G.T @ _apply_L(G, h))
    C = h * (G.T @ _apply_D4(G, h))
    F = p.q * h * G.sum(axis=0)
    return A, B, C, F


# --- modal part K0 ------------------------------------------------------------------------------------------------------

def dst(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Psi^T x = Psi x, psi_k(i) = sqrt(2/n) sin(k pi i/n), i, k = 1..n-1 (scipy DST-I, norm="ortho": symmetric, orthogonal)."""
    return scipy.fft.dst(x, type=1, norm="ortho", axis=axis)


def modal_eigenvalues(n: int, h: float) -> np.ndarray:
    """mu_k = (4/h^2) sin^2(k pi/(2n)) > 0, k = 1..n-1: L psi_k = -mu_k psi_k (R4 3.2)."""
    k = np.arange(1, n)
    return 4.0 / h**2 * np.sin(k * math.pi / (2 * n)) ** 2


def boundary_rows(N: int, m: int) -> list[int]:
    """Rows of Z = I_N (x) [e_1, e_m] in the term-first ordering; column 2 i + side of Z has its 1 in row boundary_rows[2 i + side]."""
    return [i * m + q for i in range(N) for q in (0, m - 1)]


CAPACITANCE_PATHS = ("modal", "rows")  # (m) modal sum, (g) boundary rows of the computed Y0 (fe5d60f, VOK-13)


@dataclass
class BCSolution:
    status: str
    s: np.ndarray | None = None  # term-first
    diagnostics: dict = field(default_factory=dict)
    S_modal: np.ndarray | None = None  # capacitance by the path (m)
    S_rows: np.ndarray | None = None  # capacitance by the path (g)


def _ratio(num: float, den: float):
    """num/den as an OBSERVATION; UNRESOLVED when den = 0 (e.g. q = 0: x = 0, the ratio is undefined; PB-1)."""
    return float(num / den) if den != 0 else UNRESOLVED


def m_norm(Y0: np.ndarray, S: np.ndarray, rows: list[int]) -> tuple[float, str]:
    """||M||_2, M = I - W Z^T, W = Y0 S^-1 (VOK-12, item 4, f62cd79), without the dense M on large grids.

    n = N m. If n <= 4N (m <= 4, n_s <= 5): M is formed densely (at most 12 x 12), no branch is chosen. Otherwise Q is an
    orthonormal basis of the columns of [Z, W] (n x 4N, Householder QR without a rank decision): span Q = S' contains
    S := span[Z, W] for ANY rank of [Z, W], and dim S' <= 4N < n. With C := S^perp the operator is M = M|_S (+) I_C (VOK-12, item 4,
    proof) and S' = S (+) (S' minus S) with S' minus S inside C, so ||Q^T M Q||_2 = max(||M|_S||_2, 1 or nothing) and, since C != {0},
    ||M||_2 = max(1, ||Q^T (I - W Z^T) Q||_2) exactly in exact arithmetic (own derivation, level B). The branch therefore does not
    depend on a numerical rank decision; the computed value is a floating-point value (OBSERVATION), not a guarantee."""
    n, k = Y0.shape
    W = np.linalg.solve(S.T, Y0.T).T  # Y0 S^-1
    if n <= 2 * k:  # n <= 4N
        M = np.eye(n)
        M[:, rows] -= W
        return float(np.linalg.norm(M, 2)), "dense"
    Z = np.zeros((n, k))
    Z[rows, np.arange(k)] = 1.0
    Q, _ = np.linalg.qr(np.hstack([Z, W]))
    MQ = np.eye(Q.shape[1]) - (Q.T @ W) @ Q[rows, :]  # Q^T (I - W Z^T) Q, Z^T Q = Q[rows, :]
    return max(1.0, float(np.linalg.norm(MQ, 2))), "subspace (dim <= 4N < N m)"


def solve_bc(p: hs.HalfStep, capacitance: str = "modal") -> BCSolution:
    """B-C for the half-step system of ``p`` (given set used as is). ``capacitance``: "modal" - path (m), Z^T Y0 =
    sum_k (D M_k)^-1 (x) z_k z_k^T, z_k = (psi_k(1), psi_k(m)) analytically (NS 5; fe5d60f, VOK-13, step 4 of 3.1) - the default of
    this research prototype (PB-2); "rows" - path (g), the boundary rows of the computed Y0, kept for comparisons. Both S are
    computed and logged with their difference and the chosen path. Statuses: EXACTLY_DEPENDENT (no solve), FACTORIZATION_FAILED
    (a modal block, A or S not factorizable in floating point), INDEPENDENT."""
    if capacitance not in CAPACITANCE_PATHS:
        raise ValueError(f"capacitance must be one of {CAPACITANCE_PATHS}")
    if hs.given_set_status(p) == hs.EXACTLY_DEPENDENT:
        return BCSolution(hs.EXACTLY_DEPENDENT)
    A, B, C, F = bc_coefficients(p)
    N, m, D = p.N, p.n_s - 1, p.D
    h = p.l_s / p.n_s
    mu = modal_eigenvalues(p.n_s, h)
    DM = D * (mu[:, None, None] ** 2 * A + 2.0 * mu[:, None, None] * B + C)  # (m, N, N), D M_k
    try:
        np.linalg.cholesky(DM)  # M_k > 0 (positivity of the modal blocks; NS 5)
        np.linalg.cholesky(A)
    except np.linalg.LinAlgError:
        return BCSolution(hs.FACTORIZATION_FAILED)
    # x0 = K0^-1 (F (x) 1): modal right-hand side F_i (Psi^T 1)_k
    rhs = F[None, :] * dst(np.ones(m))[:, None]  # (m, N)
    xh = np.linalg.solve(DM, rhs[:, :, None])[:, :, 0]  # (m, N)
    x0 = dst(xh.T, axis=1).reshape(-1)
    # Y0 = K0^-1 Z, columns (term i, side) in the order 2 i + side; Psi^T e_pos
    pos = (0, m - 1)
    rows = boundary_rows(N, m)
    Minv = np.linalg.solve(DM, np.broadcast_to(np.eye(N), DM.shape))  # (m, N, N)
    z = [dst(np.eye(m)[q]) for q in pos]
    Y0 = np.empty((N * m, 2 * N))
    for i in range(N):
        for side in (0, 1):
            Vh = Minv[:, :, i].T * z[side][None, :]  # (N, m): modal coefficients of all terms
            Y0[:, 2 * i + side] = dst(Vh, axis=1).reshape(-1)
    Ginv = h**4 / (2.0 * D) * np.kron(np.linalg.inv(A), np.eye(2))
    S_rows = Ginv + Y0[rows, :]  # path (g)
    k = np.arange(1, p.n_s)
    psi1 = math.sqrt(2.0 / p.n_s) * np.sin(k * math.pi / p.n_s)  # psi_k(1)
    zk = np.stack([psi1, np.where(k % 2 == 1, 1.0, -1.0) * psi1], axis=1)  # (psi_k(1), psi_k(m) = (-1)^(k+1) psi_k(1))
    S_modal = Ginv + np.einsum("kij,ks,kt->isjt", Minv, zk, zk).reshape(2 * N, 2 * N)  # path (m)
    S = S_modal if capacitance == "modal" else S_rows
    try:
        c = np.linalg.solve(S, x0[rows])
    except np.linalg.LinAlgError:
        return BCSolution(hs.FACTORIZATION_FAILED)
    Y0c = Y0 @ c
    x = x0 - Y0c
    norm_x = float(np.linalg.norm(x))
    eM = np.linalg.eigvalsh(DM / D)
    eA = np.linalg.eigvalsh(A)
    Mn, Mn_method = m_norm(Y0, S, rows)
    diag = {"capacitance_path": capacitance, "size_S": S.shape[0], "cond_S": float(np.linalg.cond(S)),
            "cond_S_modal": float(np.linalg.cond(S_modal)), "cond_S_rows": float(np.linalg.cond(S_rows)),
            "S_rows_minus_modal_rel": float(np.linalg.norm(S_rows - S_modal, 2) / np.linalg.norm(S_modal, 2)),
            "min_eig_sym_S": float(np.linalg.eigvalsh((S + S.T) / 2)[0]),
            "asym_S": float(np.max(np.abs(S - S.T)) / np.max(np.abs(S))),
            "cond_A": float(eA[-1] / eA[0]), "max_cond_Mk": float(np.max(eM[:, -1] / eM[:, 0])),
            "min_eig_Mk": float(np.min(eM[:, 0])),
            "rho_0": _ratio(float(np.linalg.norm(x0)), norm_x),  # ||x0||_2/||x||_2 (VOK-13)
            "kappa_bar_c": _ratio(float(np.linalg.norm(Y0, 2)) * float(np.linalg.norm(c)), norm_x),  # ||Y0||_2 ||c||_2/||x||_2
            "ratio_Y0c_x": _ratio(float(np.linalg.norm(Y0c)), norm_x),  # former kappa_c (rejected by Z-1, f4a4cb9)
            "M_norm_2": Mn, "M_norm_method": Mn_method, "mu_1": float(mu[0])}
    return BCSolution(hs.INDEPENDENT, x, diag, S_modal, S_rows)


# --- NS 5.1 order: rank -> QR -> diagnostics -> B-C -> residual ---------------------------------------------------------

def orthonormalise(p: hs.HalfStep):
    """sqrt(h_f) G = Q R (Householder), U~ = Q/sqrt(h_f) (R4 5.4). Returns (U~, R)."""
    h = p.l_f / p.n_f
    Q, R = np.linalg.qr(math.sqrt(h) * p.given)
    return Q / math.sqrt(h), R


def orthonormality_defect(p: hs.HalfStep) -> float:
    """max |h_f G^T G - I| in exact arithmetic (h_f = l_f/n_f exactly), converted upwards: 0 for an exactly orthonormal set."""
    h = Fraction(p.l_f) / p.n_f
    cols = [[Fraction(float(v)) for v in p.given[:, k]] for k in range(p.N)]
    worst = max(abs(h * sum(a * b for a, b in zip(cols[i], cols[j])) - (1 if i == j else 0))
                for i in range(p.N) for j in range(p.N))
    return hs._float_up(worst)


@dataclass
class HalfStepRecord:
    status: str
    solved: hs.HalfStep | None  # the half-step actually solved (after QR if applied)
    R: np.ndarray | None  # QR factor: S of the original basis = S~ R^-T (exact arithmetic)
    bc: BCSolution | None
    log: dict = field(default_factory=dict)


def span_change_sin(original: np.ndarray, new: np.ndarray, h: float) -> float:
    """sin of the largest principal angle between span(new) and span(original) (new orthonormal in the weight h): the residual of
    the projection of each column of ``new`` onto span(original) is computed EXACTLY (rational Gram system), then
    ||(I - P_orig) sqrt(h) new||_2 in double. OBSERVATION (PB-4)."""
    N = original.shape[1]
    G = [[Fraction(float(v)) for v in original[:, k]] for k in range(N)]
    gram = [[sum(a * b for a, b in zip(G[i], G[j])) for j in range(N)] for i in range(N)]
    res = np.empty_like(new, dtype=float)
    for c in range(new.shape[1]):
        u = [Fraction(float(v)) for v in new[:, c]]
        rhs = [sum(a * b for a, b in zip(G[i], u)) for i in range(N)]
        coef = _solve_exact(gram, rhs)
        res[:, c] = [float(u[r] - sum(coef[i] * G[i][r] for i in range(N))) for r in range(len(u))]
    return float(np.linalg.norm(math.sqrt(h) * res, 2))


def _solve_exact(K, f):
    n = len(K)
    M = [list(K[i]) + [f[i]] for i in range(n)]
    for c in range(n):
        piv = next(r for r in range(c, n) if M[r][c] != 0)
        M[c], M[piv] = M[piv], M[c]
        for r in range(c + 1, n):
            if M[r][c]:
                g = M[r][c] / M[c][c]
                M[r] = [a - g * b for a, b in zip(M[r], M[c])]
    x = [Fraction(0)] * n
    for i in reversed(range(n)):
        x[i] = (M[i][n] - sum(M[i][j] * x[j] for j in range(i + 1, n))) / M[i][i]
    return x


def _eta(p: hs.HalfStep, s, coeffs=None):
    """eta_QA (R-HP, NS 5); UNRESOLVED when its denominator is 0 (s = 0 and F = 0, i.e. q = 0: 0/0; PB-1)."""
    if not np.any(s) and p.q == 0:
        return UNRESOLVED
    return hs.eta_qa(p, s, coeffs)


RESEARCH_QR = ("RESEARCH_ONLY: QR without a qualified numerical rank decision (NS 5.1 step 1; D-049.5, D-049.6); the result refers "
               "to span(U~), not to span(G); not for normative Q3 distributions")
RESEARCH_NO_QR = "RESEARCH_ONLY: N >= 2 without QR (BC-6 analogue), not the NS 5.1 order"


def run_halfstep(p: hs.HalfStep, qr: bool = True, capacitance: str = "modal") -> HalfStepRecord:
    """One half-step in the order NS 5.1 (D-049.2, item 3). The numerical rank decision of NS 5.1, step 1, has no qualified
    criterion yet (D-049.5, D-049.6): only the exact rank stops here; cond(A) before QR is recorded for that decision. Every QR run
    is therefore marked ``qr_without_qualified_rank_decision`` and RESEARCH_ONLY (PB-4), with the angle between span(U~) and
    span(G) as an OBSERVATION."""
    log = {"N": p.N, "direction": p.direction, "n_f": p.n_f, "n_s": p.n_s, "qr": bool(qr and p.N >= 2),
           "qr_without_qualified_rank_decision": bool(qr and p.N >= 2),
           "run_class": RESEARCH_QR if (qr and p.N >= 2) else (RESEARCH_NO_QR if p.N >= 2 else "N = 1 (no QR, NS 5.1)")}
    if hs.given_set_status(p) == hs.EXACTLY_DEPENDENT:
        log["set_status"] = hs.EXACTLY_DEPENDENT
        return HalfStepRecord(hs.EXACTLY_DEPENDENT, None, None, None, log)
    log["set_status"] = hs.INDEPENDENT
    A0 = bc_coefficients(p)[0]
    e0 = np.linalg.eigvalsh(A0)
    log["cond_A_set"] = float(e0[-1] / e0[0]) if e0[0] > 0 else math.inf
    R = None
    solved = p
    if p.N == 1:
        log["cond_A_before_QR"] = 1.0  # A scalar (NS 5.1)
    elif qr:
        U, R = orthonormalise(p)
        sv = np.linalg.svd(R, compute_uv=False)
        log["cond_A_before_QR"] = float((sv[0] / sv[-1]) ** 2)  # = cond(R)^2
        solved = hs.HalfStep(p.lx, p.nx, p.ly, p.ny, p.direction, U, D=p.D, q=p.q)
        eA = np.linalg.eigvalsh(bc_coefficients(solved)[0])
        log["cond_A_after_QR"] = float(eA[-1] / eA[0])
        log["orthonormality_defect_up"] = orthonormality_defect(solved)
        log["sin_angle_span_G_Utilde"] = span_change_sin(p.given, U, p.l_f / p.n_f)
    else:
        log["cond_A_before_QR"] = log["cond_A_set"]
    bc = solve_bc(solved, capacitance)
    log["bc_status"] = bc.status
    log.update(bc.diagnostics)
    if bc.s is not None:
        log["eta_bc"] = _eta(solved, bc.s)
    return HalfStepRecord(bc.status, solved, R, bc, log)


def _tau(p: hs.HalfStep) -> float | None:
    """Trial value tau = lambda_min(K)/2 from the floating-point K (eigvalsh); its validity is proved by the certificate only."""
    lam = float(scipy.linalg.eigvalsh(hs.coefficient_system(p)[0], subset_by_index=[0, 0])[0])
    return lam / 2 if lam > 0 else None


def _certified_pair_consistent(s1, b1, s2, b2) -> bool:
    """Exact check ||s1 - s2||_2 <= b1 + b2 (both bounds of the same exact solution: must hold)."""
    d2 = sum(((Fraction(float(a)) - Fraction(float(b))) ** 2 for a, b in zip(s1, s2)), Fraction(0))
    return d2 <= (Fraction(b1) + Fraction(b2)) ** 2


def compare(rec: HalfStepRecord, with_rh: bool = True) -> dict:
    """B-C against R-AC (refined) and, for n_f, n_s <= 32, R-H, on the system actually solved. Certified quantities are floats or
    UNRESOLVED; differences are OBSERVATION."""
    out = {"status": rec.status}
    if rec.bc is None or rec.bc.s is None:
        return out
    p, s_bc = rec.solved, rec.bc.s
    coeffs = hs.exact_coefficients(p)
    tau = _tau(p)
    low = None if tau is None else hs.certify_lambda_min(p, tau, coeffs)
    out["lambda_low"] = UNRESOLVED if low is None else low

    def bounds(s):
        if low is None:
            return None, hs.RelativeBounds(None, None)
        b = hs.error_bound(p, s, low, coeffs)
        return b, hs.relative_bounds(b, s)

    b_bc, rel_bc = bounds(s_bc)
    out["bound_abs_bc"] = UNRESOLVED if b_bc is None else b_bc
    out["rel_to_computed_bc"] = rel_bc.to_computed if rel_bc.to_computed is not None else UNRESOLVED
    out["rel_to_exact_bc"] = rel_bc.to_exact if rel_bc.to_exact is not None else UNRESOLVED
    out["eta_bc"] = _eta(p, s_bc, coeffs)
    ac = hs.solve_ac(p)
    out["ac_status"] = ac.status
    if ac.s is not None:
        b_ac, rel_ac = bounds(ac.s)
        out["bound_abs_ac"] = UNRESOLVED if b_ac is None else b_ac
        out["rel_to_exact_ac"] = rel_ac.to_exact if rel_ac.to_exact is not None else UNRESOLVED
        out["bc_minus_ac_rel"] = _ratio(float(np.linalg.norm(s_bc - ac.s)), float(np.linalg.norm(ac.s)))  # OBSERVATION
        out["bc_ac_certified_consistent"] = UNRESOLVED if b_ac is None else \
            _certified_pair_consistent(s_bc, b_bc, ac.s, b_ac)
    if with_rh and p.n_f <= 32 and p.n_s <= 32:
        status_h, s_h = hs.solve_hessian(p)
        out["rh_status"] = status_h
        if s_h is not None:
            b_h, _ = bounds(s_h)
            out["bc_minus_rh_rel"] = _ratio(float(np.linalg.norm(s_bc - s_h)), float(np.linalg.norm(s_h)))  # OBSERVATION
            out["bc_rh_certified_consistent"] = UNRESOLVED if b_h is None else \
                _certified_pair_consistent(s_bc, b_bc, s_h, b_h)
    return out


def field_of(rec: HalfStepRecord) -> np.ndarray | None:
    """Field W = U~ S~^T on the grid of the record (fixed direction rows, solved direction columns); OBSERVATION only."""
    if rec.bc is None or rec.bc.s is None:
        return None
    return rec.solved.given @ rec.bc.s.reshape(rec.solved.N, -1)
