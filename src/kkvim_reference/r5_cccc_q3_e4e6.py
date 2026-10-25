"""Pre-Q3 instrument for the measurements E-4 (error of the research B-C) and E-6 (log of the model factors) of R5-V CCCC.

Preparation only: no Q3 campaign, no thresholds, no K_E, c_R, c_DST, eps_floor^C; no import of ``kkvim`` (D-028). Uses only the
accepted QA components kkvim_reference.r5_cccc_bc (research prototype B-C), kkvim_reference.r5_cccc_halfstep (R-H, R-AC, R-HP,
certificates of Q1b). Sources: docs/R5_CCCC_ROUNDOFF_PREQUAL_DRAFT.md (RO) 4.1-4.3, table 4.3a, VOK-10...VOK-14; PL 2, 4, 7;
D-049.2, D-049.5, D-049.6.

E-4 (``e4_case``), for one half-step of the system actually solved (after QR if applied):
- REFERENCE. Exact tier (N m <= EXACT_LIMIT): x* = (K^C)^-1 (F (x) 1) in rational arithmetic from the exact system of the input
  doubles (delta_ref = 0). Otherwise the certified tier: R-AC with refinement and the Q1b certificate (lambda_low, error_bound):
  b_ref >= ||x_ref - x*||_2, b_ref_K = ||r_ref||_2/lambda_low^(1/2) >= ||x_ref - x*||_K.
- SET error of the coefficients ||x^ - x*||_2 (exact tier: exact interval; certified tier: [d - b_ref, d + b_ref], d exact).
- R-HP residual of x^ (exact), its 2-norm interval and eta_QA.
- VOK-14 (b) certificate r^/(D mu_1^2 lambda) and r^/(D^(1/2) mu_1 lambda^(1/2)) with r^ >= ||r||_2 (exact residual, root up),
  mu_1 >= 4/l_s^2 (sin x >= 2x/pi; exact rational) and a certified lambda <= lambda_min(A) (exact LDL^T of A - lambda I); otherwise
  UNRESOLVED. Consistency with the reference (must hold): exact check.
- Budget 1/100 (D-049.6, item 4): reference error <= measured error/100 in the same norm, checked exactly; otherwise the comparison
  is UNRESOLVED.
- FIELD error separately: ||U~(V^ - V*)^T||_a = h_s^(1/2) ||x^ - x*||_K (NS 4.3, P-S1) - an exact identity of the energy norm of
  the field of the SOLVED set; ||x^ - x*||_K^2 = -(x^ - x*)^T r(x^) exactly. This is not a transfer of the 2-norm set certificate to
  the field; the field in the max- or 2-norm is not certified (it needs ||G||_2 and the product rounding); the formation error of the
  field (E-2, component (i)) is not part of it.
- Components eps_0, eps_Y, eps_S, eps_k and T_1...T_4 (VOK-13): exact references x0*, Y0*, S*, c*, M* are computed (exact tier),
  but the prototype B-C does not expose x0^, Y0^, c^, G^-1^ (r5_cccc_bc is not changed here): these components are UNRESOLVED with
  the missing instrument named. Measurable now: the delta S part of eps_S for both capacitance paths (S^ is exposed).
E-6 (``e6_case``): factors of table 4.3a and the checks of VOK-10...VOK-12 evaluated in double (OBSERVATION of a proven inequality),
exact-reference values of rho_0, kappa_bar_c, cond(S), ||M||_2 on the exact tier, preconditions and statuses; before and after QR.
N >= 2 runs are RESEARCH_ONLY (r5_cccc_bc) and PROVISIONAL / NOT_QUALIFIED (D-049.5); certificates refer to the solved set (span U~
after QR), never to the field of the original set.
"""

import math
from fractions import Fraction

import numpy as np
import scipy.linalg

from kkvim_reference import r5_cccc_bc as bc
from kkvim_reference import r5_cccc_halfstep as hs

UNRESOLVED = "UNRESOLVED"
EXACT_LIMIT = 36  # N (n_s - 1) up to which the exact rational reference is computed (cost)
MISSING_COMPONENTS = ("UNRESOLVED: the research B-C (r5_cccc_bc) does not expose x0^, Y0^, c^, G^-1^ (API not changed in this "
                      "task); exact references x0*, Y0*, S*, c*, M* are prepared")
MISSING_RAISED = "UNRESOLVED: no raised-precision B-C reference for N m > EXACT_LIMIT (OK-7: DST-I with certified sines)"
MISSING_EPS_K = "UNRESOLVED: eps_k needs the exact z_k = (psi_k(1), psi_k(m)) (irrational sines; certified sine reference, OK-7)"


# --- exact helpers ------------------------------------------------------------------------------------------------------

def _sqrt_bounds(x: Fraction) -> tuple[Fraction, Fraction]:
    if x == 0:
        return Fraction(0), Fraction(0)
    P, Q = x.numerator, x.denominator
    r = math.isqrt(P * Q * 4**64)
    return Fraction(r, Q * 2**64), Fraction(r + 1, Q * 2**64)


def _up(x: Fraction) -> float:
    return hs._float_up(x)


DBL_MAX = 1.7976931348623157e308


def _down(x: Fraction) -> float:
    """Largest double <= x; above DBL_MAX the valid lower bound DBL_MAX."""
    return DBL_MAX if x > Fraction(DBL_MAX) else hs._float_down(x)


def _norm2_interval(v) -> tuple[float, float]:
    lo, hi = _sqrt_bounds(sum((Fraction(x) ** 2 for x in v), Fraction(0)))
    return _down(lo), _up(hi)


def _solve_many(K, cols):
    """Exact solution of K X = [cols] (rational Gaussian elimination, row pivoting on nonzero entries)."""
    n, k = len(K), len(cols)
    M = [list(K[i]) + [c[i] for c in cols] for i in range(n)]
    for c in range(n):
        piv = next(r for r in range(c, n) if M[r][c] != 0)
        M[c], M[piv] = M[piv], M[c]
        for r in range(c + 1, n):
            if M[r][c]:
                g = M[r][c] / M[c][c]
                M[r] = [a - g * b for a, b in zip(M[r], M[c])]
    X = [[Fraction(0)] * n for _ in range(k)]
    for j in range(k):
        for i in reversed(range(n)):
            X[j][i] = (M[i][n + j] - sum(M[i][q] * X[j][q] for q in range(i + 1, n))) / M[i][i]
    return X  # list of solution columns


def _spd_exact(M) -> bool:
    A = [list(r) for r in M]
    for k in range(len(A)):
        if A[k][k] <= 0:
            return False
        for i in range(k + 1, len(A)):
            f = A[i][k] / A[k][k]
            if f:
                for j in range(k, len(A)):
                    A[i][j] -= f * A[k][j]
    return True


def exact_system(p: hs.HalfStep, coeffs=None):
    """Dense exact K^C (term-first) from the accepted exact band of Q1b, and b = F (x) 1."""
    coeffs = coeffs or hs.exact_coefficients(p)
    band = hs._exact_band_matrix(p, coeffs)
    m, N = p.n_s - 1, p.N
    perm = hs._node_term(m, N)
    K = [[Fraction(0)] * p.size for _ in range(p.size)]
    for (r, c), v in band.items():
        K[perm[r]][perm[c]] = v
        K[perm[c]][perm[r]] = v
    b = [coeffs[3][i] for i in range(N) for _ in range(m)]
    return K, b


def certified_lambda_min_A(A_exact) -> float | None:
    """A double lambda with 0 < lambda <= lambda_min(A) proved by the exact LDL^T of A - lambda I; None if not found."""
    lam = float(np.linalg.eigvalsh(np.array([[float(v) for v in row] for row in A_exact]))[0])
    if not (lam > 0):
        return None
    for shrink in (1 - 2.0**-20, 0.5, 0.25):
        t = hs._float_down(Fraction(lam) * Fraction(shrink))
        if t > 0 and _spd_exact([[A_exact[i][j] - (Fraction(t) if i == j else 0) for j in range(len(A_exact))]
                                 for i in range(len(A_exact))]):
            return t
    return None


# --- E-4 ------------------------------------------------------------------------------------------------------------------

def _budget(ref_err_sq_upper: Fraction, measured_sq_lower: Fraction) -> str:
    """D-049.6, item 4: reference error <= measured/100 in the same norm (exact comparison of squares)."""
    if measured_sq_lower == 0:
        return "UNRESOLVED (measured error is 0 or not separated from 0: the relative rule is undefined)"
    return "SATISFIED" if 10**4 * ref_err_sq_upper <= measured_sq_lower else "NOT_MET: comparison UNRESOLVED"


def e4_case(p: hs.HalfStep, qr: bool = True, capacitance: str = "modal") -> dict:
    """E-4 for one half-step (module docstring). Returns a flat dict; certified values are floats, otherwise status strings."""
    rec = bc.run_halfstep(p, qr=qr, capacitance=capacitance)
    out = {"N": p.N, "direction": p.direction, "n_f": p.n_f, "n_s": p.n_s, "qr": rec.log["qr"], "capacitance": capacitance,
           "status": rec.status, "run_class": rec.log["run_class"],
           "marking": "PROVISIONAL / NOT_QUALIFIED (D-049.5)" if p.N >= 2 else "N = 1",
           "certificates_refer_to": "solved set (span U~ after QR)" if rec.log["qr"] else "given set"}
    if rec.bc is None or rec.bc.s is None:
        return out
    ps, s = rec.solved, rec.bc.s
    coeffs = hs.exact_coefficients(ps)
    D = Fraction(ps.D)
    h_s = Fraction(ps.l_s) / ps.n_s
    r = hs.residual_exact(ps, s, coeffs)
    r2 = sum((x * x for x in r), Fraction(0))
    out["rhp_residual_2_down"], out["rhp_residual_2_up"] = _norm2_interval(r)
    out["eta_qa"] = bc._eta(ps, s, coeffs)
    # VOK-14 (b)
    lam_A = certified_lambda_min_A(coeffs[0])
    mu1_low = Fraction(4) / Fraction(ps.l_s) ** 2  # mu_1 >= 4/b^2 (sin x >= 2x/pi), exact
    if lam_A is None:
        out["vok14_status"] = "UNRESOLVED (no certified lower bound of lambda_min(A))"
        b14_2 = b14_K = None
    else:
        r_up = _sqrt_bounds(r2)[1]
        b14_2 = r_up / (D * mu1_low**2 * Fraction(lam_A))
        b14_K = r_up / (_sqrt_bounds(D)[0] * mu1_low * _sqrt_bounds(Fraction(lam_A))[0])
        out["vok14_status"] = "CERTIFICATE (conditions 1-3 of VOK-14 (b) met)"
        out["vok14_bound_2_up"], out["vok14_bound_K_up"] = _up(b14_2), _up(b14_K)
        out["vok14_field_a_bound_up"] = _up(_sqrt_bounds(h_s)[1] * b14_K)
        out["lambda_min_A_certified"], out["mu1_lower"] = lam_A, _down(mu1_low)
    s_frac = [Fraction(float(x)) for x in s]
    if ps.size <= EXACT_LIMIT:
        out["reference"] = "EXACT (rational, delta_ref = 0)"
        K, b = exact_system(ps, coeffs)
        x_star = _solve_many(K, [b])[0]
        delta = [a - c for a, c in zip(s_frac, x_star)]
        d2 = sum((x * x for x in delta), Fraction(0))
        dK2 = -sum((a * c for a, c in zip(delta, r)), Fraction(0))  # K delta = -r(x^)
        out["set_error_2_down"], out["set_error_2_up"] = _norm2_interval(delta)
        xs2 = sum((x * x for x in x_star), Fraction(0))
        out["set_error_rel_up"] = _up(_sqrt_bounds(d2)[1] / _sqrt_bounds(xs2)[0]) if xs2 else UNRESOLVED
        fa_lo, fa_hi = _sqrt_bounds(h_s * dK2)
        out["field_error_a_down"], out["field_error_a_up"] = _down(fa_lo), _up(fa_hi)
        wK2 = sum((a * c for a, c in zip(x_star, b)), Fraction(0))  # x*^T K x* = x*^T b
        out["field_error_a_rel_up"] = _up(fa_hi / _sqrt_bounds(h_s * wK2)[0]) if wK2 else UNRESOLVED
        out["budget_set"] = out["budget_field"] = "SATISFIED (exact reference)" if d2 else \
            "UNRESOLVED (error exactly 0: relative rule undefined)"
        if b14_2 is not None:
            out["vok14_consistent"] = d2 <= b14_2**2 and dK2 <= b14_K**2  # theorem (a): must hold
        out.update(_components(ps, rec, coeffs, K, b, x_star))
    else:
        out["reference"] = "CERTIFIED R-AC (Q1b certificate)"
        tau = bc._tau(ps)
        low = None if tau is None else hs.certify_lambda_min(ps, tau, coeffs)
        ac = hs.solve_ac(ps)
        if low is None or ac.s is None:
            out["budget_set"] = out["budget_field"] = "UNRESOLVED (no certificate of the reference)"
        else:
            b_ref = Fraction(hs.error_bound(ps, ac.s, low, coeffs))
            r_ac = hs.residual_exact(ps, ac.s, coeffs)
            b_refK = _sqrt_bounds(sum((x * x for x in r_ac), Fraction(0)))[1] / _sqrt_bounds(Fraction(low))[0]
            delta = [a - Fraction(float(c)) for a, c in zip(s_frac, ac.s)]
            d2 = sum((x * x for x in delta), Fraction(0))
            dK2 = sum((a * (c - e) for a, c, e in zip(delta, r_ac, r)), Fraction(0))  # K(x^ - x_ac) = r_ac - r(x^)
            dl, dh = _sqrt_bounds(d2)
            out["set_error_2_down"] = _down(max(Fraction(0), dl - b_ref))
            out["set_error_2_up"] = _up(dh + b_ref)
            kl, kh = _sqrt_bounds(h_s * dK2)
            hl, hh = _sqrt_bounds(h_s)
            out["field_error_a_down"] = _down(max(Fraction(0), kl - hh * b_refK))
            out["field_error_a_up"] = _up(kh + hh * b_refK)
            out["reference_bound_2_up"], out["reference_bound_K_up"] = _up(b_ref), _up(b_refK)
            out["budget_set"] = _budget(b_ref**2, max(Fraction(0), dl) ** 2)
            out["budget_field"] = _budget(b_refK**2, max(Fraction(0), _sqrt_bounds(dK2)[0]) ** 2)
            if b14_2 is not None:
                out["vok14_consistent"] = max(Fraction(0), dl - b_ref) <= b14_2  # true error >= d - b_ref
        for key in ("eps_0", "eps_Y", "eps_S", "T1_T4", "eps_dS_modal", "eps_dS_rows"):
            out[key] = MISSING_RAISED
        out["eps_k"] = MISSING_EPS_K
    return out


def _components(ps, rec, coeffs, K, b, x_star) -> dict:
    """Exact references of the B-C components (exact tier) and what can be measured with the exposed S^."""
    A, B, C, F = coeffs
    N, m = ps.N, ps.n_s - 1
    D = Fraction(ps.D)
    h = Fraction(ps.l_s) / ps.n_s
    rows = bc.boundary_rows(N, m)
    n = N * m
    g = D * 2 / h**4
    K0 = [row[:] for row in K]
    for a, ra in enumerate(rows):  # K0 = K^C - Z G Z^T, G = D A (x) (2/h^4) I_2 (NS 5; exact identity BC-3)
        for c, rc in enumerate(rows):
            if a % 2 == c % 2:
                K0[ra][rc] -= g * A[a // 2][c // 2]
    cols = [b] + [[Fraction(int(i == rc)) for i in range(n)] for rc in rows]
    sol = _solve_many(K0, cols)
    x0, Y0 = sol[0], sol[1:]  # Y0 as columns
    Ainv = _solve_many(A, [[Fraction(int(i == j)) for i in range(N)] for j in range(N)])  # columns of A^-1
    Ginv = [[(h**4 / (2 * D)) * Ainv[c // 2][a // 2] * (a % 2 == c % 2) for c in range(2 * N)] for a in range(2 * N)]
    S = [[Ginv[a][c] + Y0[c][rows[a]] for c in range(2 * N)] for a in range(2 * N)]
    c_star = _solve_many(S, [[x0[ra] for ra in rows]])[0]
    x_check = [x0[i] - sum(Y0[c][i] * c_star[c] for c in range(2 * N)) for i in range(n)]
    out = {"exact_woodbury_equals_solution": x_check == x_star}
    Sf = np.array([[float(v) for v in row] for row in S])
    Snorm = np.linalg.norm(Sf, 2)
    for path, Shat in (("modal", rec.bc.S_modal), ("rows", rec.bc.S_rows)):
        dS = np.array([[float(Fraction(float(Shat[a, c])) - S[a][c]) for c in range(2 * N)] for a in range(2 * N)])
        out[f"eps_dS_{path}"] = float(np.linalg.norm(dS, 2) / Snorm)  # delta S part only (E of the 2N solve not exposed)
    for key in ("eps_0", "eps_Y", "eps_S", "T1_T4"):
        out[key] = MISSING_COMPONENTS
    out["eps_k"] = MISSING_EPS_K
    # exact-reference values of the E-6 factors (exact matrices, final norms in double)
    xs = np.array([float(v) for v in x_star])
    nx = np.linalg.norm(xs)
    Y0f = np.array([[float(Y0[c][i]) for c in range(2 * N)] for i in range(n)])
    cf = np.array([float(v) for v in c_star])
    W = _solve_many([[S[c][a] for c in range(2 * N)] for a in range(2 * N)],
                    [[Y0[a][i] for a in range(2 * N)] for i in range(n)])  # rows of W = Y0 S^-1 (solve S^T w_i = y_i)
    M = np.eye(n)
    for i in range(n):
        for a, ra in enumerate(rows):
            M[i, ra] -= float(W[i][a])
    out.update({"rho_0_exactref": float(np.linalg.norm([float(v) for v in x0]) / nx) if nx else UNRESOLVED,
                "kappa_bar_c_exactref": float(np.linalg.norm(Y0f, 2) * np.linalg.norm(cf) / nx) if nx else UNRESOLVED,
                "cond_S_exactref": float(np.linalg.cond(Sf)), "M_norm_2_exactref": float(np.linalg.norm(M, 2))})
    return out


# --- E-6 ------------------------------------------------------------------------------------------------------------------

def pencil_factors(p: hs.HalfStep) -> dict:
    """beta_U = lambda_max(A^-1 B), chi_U = lambda_max(A^-1 C), theta = beta + chi^(1/2), varrho = mu_1/(mu_1 + theta) (RO 0.2), in
    double (generalized eigenvalues N x N; OBSERVATION)."""
    A, B, C, _ = bc.bc_coefficients(p)
    mu1 = float(bc.modal_eigenvalues(p.n_s, p.l_s / p.n_s)[0])
    beta = float(scipy.linalg.eigh(B, A, eigvals_only=True)[-1])
    chi = float(scipy.linalg.eigh(C, A, eigvals_only=True)[-1])
    theta = beta + math.sqrt(max(chi, 0.0))
    eA = np.linalg.eigvalsh(A)
    return {"beta_U": beta, "chi_U": chi, "theta_U": theta, "varrho": mu1 / (mu1 + theta), "mu_1": mu1,
            "cond_A_solved": float(eA[-1] / eA[0]), "lambda_min_A": float(eA[0]), "A": A, "B": B, "C": C}


def numeric_check(value: float, bound: float, sense: str) -> tuple[str, float]:
    """Unified rule (IR-1, review 6e50297) for comparing DOUBLE evaluations of a PROVEN inequality (VOK-10...VOK-12).

    The inequality itself holds under its verified preconditions (status ``vok1x_theorem``). Its double evaluation is compared:
    - consistent in double -> "HOLDS (double evaluation)";
    - contradictory -> "UNRESOLVED_ROUNDING", with the relative gap |value - bound|/|bound| recorded: no rigorous bound of the
      evaluation error of these double quantities (eigenvalues, SVD, condition numbers) is available, so a contradiction is a
      diagnostic finding (rounding or a code defect), never reported as a violation of the theorem and never as HOLDS. Large gaps
      stay visible in ``*_gap_rel``; no tolerance is applied. Returns (status, gap)."""
    holds = value <= bound if sense == "<=" else value >= bound
    if holds:
        return "HOLDS (double evaluation)", 0.0
    gap = abs(value - bound) / abs(bound) if bound else math.inf
    return (f"UNRESOLVED_ROUNDING (double values contradict the proven inequality by a relative {gap:.3e}; no rigorous bound of "
            "the evaluation error; diagnostic, not a violation of the theorem)"), gap


LOWER_BOUND_WITNESS = ("lower bound 1 for n_s >= 4 by the exact witness: for M = I - W Z^T (any W) there is a row k outside the 2N "
                       "boundary rows, Z^T e_k = 0 and M e_k = e_k exactly")


def lower_bound_one(m_norm: float, n_s: int) -> tuple[str, float]:
    """CHECK of the double value of ||M||_2 against the lower bound 1 of VOK-12, item 3 (IR-1a, review 485708f). The bound itself is a
    theorem for n_s >= 4 (``LOWER_BOUND_WITNESS``; status in ``vok12_theorem``). The computed value is compared by the same rule as
    every other bound (``numeric_check``): >= 1 -> HOLDS (double evaluation); < 1 -> UNRESOLVED_ROUNDING with the gap 1 - value. The
    cause of a value below 1 is NOT established (rounding of the SVD or a defect of the implementation: no rigorous bound of the SVD
    error is available), so none is claimed. n_s <= 3: NOT_APPLICABLE. Returns (status, gap)."""
    if n_s <= 3:
        return "NOT_APPLICABLE (n_s <= 3: span Z may be the whole space)", 0.0
    return numeric_check(m_norm, 1.0, ">=")


def e6_case(p: hs.HalfStep, qr: bool = True, capacitance: str = "modal") -> dict:
    """E-6 factors (table 4.3a) and the double evaluations of the proven inequalities VOK-10...VOK-12 for one half-step."""
    rec = bc.run_halfstep(p, qr=qr, capacitance=capacitance)
    log = rec.log
    out = {"N": p.N, "direction": p.direction, "n_f": p.n_f, "n_s": p.n_s, "qr": log["qr"], "capacitance": capacitance,
           "status": rec.status, "run_class": log["run_class"],
           "marking": "PROVISIONAL / NOT_QUALIFIED (D-049.5)" if p.N >= 2 else "N = 1",
           "cond_A_before_QR": log.get("cond_A_before_QR"), "cond_A_after_QR": log.get("cond_A_after_QR", "no QR")}
    if rec.bc is None or rec.bc.s is None:
        out["vok10_13_preconditions"] = "NOT_APPLICABLE (" + rec.status + ")"
        return out
    for key in ("rho_0", "kappa_bar_c", "ratio_Y0c_x", "cond_S", "cond_S_modal", "cond_S_rows", "S_rows_minus_modal_rel",
                "M_norm_2", "M_norm_method", "max_cond_Mk", "min_eig_Mk"):
        out[key] = log[key]
    pf = pencil_factors(rec.solved)
    out.update({k: pf[k] for k in ("beta_U", "chi_U", "theta_U", "varrho", "mu_1", "cond_A_solved")})
    n, N = p.n_s, p.N
    out["vok10_13_preconditions"] = "MET: exact rank N, A > 0, B, C >= 0 (Gram matrices; NS 4.2)"
    # IR-1 (review 6e50297): the theorem (proven under verified preconditions) and the check of double values are separate
    proven = "PROVEN ({}; preconditions verified: exact rank N => A > 0, B, C >= 0)"
    # VOK-10: cond(S) <= cond(A) C_S(n, varrho)
    CS = n / (1 + pf["varrho"] ** 2 * (n - 1) * (n - 2) / (3 * n))
    out["vok10_bound"] = pf["cond_A_solved"] * CS
    out["vok10_theorem"] = proven.format("VOK-10")
    out["vok10_check"], out["vok10_gap_rel"] = numeric_check(log["cond_S"], out["vok10_bound"], "<=")
    # VOK-11: cond(M_k) <= cond(A)/varrho^2
    out["vok11_bound"] = pf["cond_A_solved"] / pf["varrho"] ** 2
    out["vok11_theorem"] = proven.format("VOK-11, item 1")
    out["vok11_check"], out["vok11_gap_rel"] = numeric_check(log["max_cond_Mk"], out["vok11_bound"], "<=")
    # VOK-12 item 3: ||M||_2 <= (lambda_max(K0)/(D mu_1^2 lambda_min(A)))^(1/2); 1 <= ||M||_2 for n_s >= 4
    mu = bc.modal_eigenvalues(n, p.l_s / n)
    lam_max_K0 = p.D * max(float(np.linalg.eigvalsh(mk ** 2 * pf["A"] + 2 * mk * pf["B"] + pf["C"])[-1]) for mk in mu)
    out["vok12_upper"] = math.sqrt(lam_max_K0 / (p.D * pf["mu_1"] ** 2 * pf["lambda_min_A"]))
    out["vok12_theorem"] = proven.format("VOK-12, item 3: upper bound; " + (LOWER_BOUND_WITNESS if n >= 4 else
                                                                             "lower bound 1 not applicable for n_s <= 3"))
    upper, out["vok12_upper_gap_rel"] = numeric_check(log["M_norm_2"], out["vok12_upper"], "<=")
    out["vok12_lower"], out["vok12_lower_gap_rel"] = lower_bound_one(log["M_norm_2"], n)
    tail = "; lower bound 1: " + out["vok12_lower"] if n >= 4 else "; lower bound 1 not applicable (n_s <= 3)"
    resolved = upper.startswith("HOLDS") and (n <= 3 or out["vok12_lower"].startswith("HOLDS"))
    # IR-1a: the combined status starts with HOLDS only if every numerical check of VOK-12 holds
    out["vok12_check"] = upper + tail if resolved else f"UNRESOLVED_ROUNDING (upper bound: {upper}{tail})"
    out["vok13_status"] = "MODEL (first order; no check; factors logged)"
    out["factor_provenance"] = ("rho_0, kappa_bar_c, cond(S), ||M||_2, max cond(M_k), varrho, beta_U, chi_U: computed in double; "
                                "VOK-10, VOK-11, VOK-12 bounds: proven inequalities evaluated in double; exact-reference "
                                "values: e4_case on the exact tier; eps_0, eps_Y, eps_S, eps_k, T_1...T_4: UNRESOLVED")
    return out
