"""Research QA controller of CCCC half-step sequences s = 1..8 (R5-V; audit ce59753, G-1). QA reference, not the working solver:
no import of ``kkvim``; owner task (D-049.6, item 5; D-049.7). Results are OBSERVATION; N >= 2 runs are RESEARCH_ONLY and
PROVISIONAL / NOT_QUALIFIED (D-049.5). No thresholds: no PASS, no stop criterion; the sequence runs a fixed number of half-steps
unless a status stops it.

Basis: D-049.1 (Pi_h^C, D4 rows (7, -4, 1)); D-049.3 (start T_N = span{sin^2(i pi x/a)}, orientation x: the first half-step is by y
with given functions of x); NS 4.3 (ChF-C1/C2), 4.4 (normalisation I-4 for N = 1), 5.1 and R4 5.2, 5.4, 5.4.1 (rank diagnostic
rho_N, QR with the grid weight, the companion set); RO VOK-2 (energy identity with an inexact half-step), VOK-6, VOK-8.

Every half-step s (direction alternates Y, X, Y, ... from ``first``):
1. given set G (the factor of the fixed direction) and the companion C (the other factor of the previous field, w^(s-1) = G C^T in
   the orientation x-first; s = 1: C = 0, w^(0) = 0);
2. status of G: exact rank (r5_cccc_halfstep.given_set_status); an exactly dependent set stops the sequence;
3. solution by R-AC with refinement against the R-HP residual (r5_cccc_halfstep.solve_ac); the Q1b certificate lambda_low of the
   system (certify_lambda_min, trial tau = lambda_min(K in double)/2); set bound error_bound; energy-norm bound
   ||x^ - x*||_K <= ||r||_2/lambda_low^(1/2) and the field bound ||w~ - w*||_a <= h_s^(1/2) ||x^ - x*||_K (NS 4.3, P-S1); without a
   certificate: CERT_UNRESOLVED (the sequence continues, diagnostic status kept);
4. energy (exact, r5_cccc_energy): Pi_h^C of the exact products w~^(s-1) = G C^T, w~^(s) = G V^T; VOK-2 identity
   Pi(w~^(s-1)) - Pi(w~^(s)) = ||d~||_a^2/2 - gamma_s, gamma_s = h_s sum_i rho_i^T Delta_i (rho exact R-HP residual, Delta = C - V)
   checked EXACTLY; the computed check in double (Pi, Q, gamma^) and its exact decomposition into solution (-gamma_s), formation
   (f(w^) - f(w~)) and assembly ((dP^ - Q^) - f(w^)) parts; model terms R^C, R_Q^C, R_f^C (r5_cccc_energy; c_R not assigned);
5. transformation for the next half-step:
   - N = 1: I-4 (NS 4.4): c = sigma ||f||_inf, sigma = sign <f, 1>; f <- f/c, g <- g c; f = 0 stops (ZERO_FIELD);
   - N >= 2, qr: rank diagnostic of the field (R4 5.2: sqrt(h) G = Q_G R_G, sqrt(h_s) V = Q_V R_V, sigma_k(R_G R_V^T), rho^_N,
     kappa_rep, sigma_N(R_V D_c)); exact rank of V < N stops (S-1: proven dependence of the computed set, D-049.5, item 2); otherwise
     the numerical rank decision is UNRESOLVED (no approved criterion; no library rank heuristic is used) and QR is applied as a
     RESEARCH_ONLY step: V <- Q_V/sqrt(h_s), companion G <- G R_V^T; cond(A) before QR = cond(R_V)^2, after QR from the new set;
   - N >= 2 without qr: no transformation (BC-6 analogue, RESEARCH_ONLY).
"""

import math
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np
import scipy.linalg

from kkvim_reference import r5_cccc_energy as en
from kkvim_reference import r5_cccc_halfstep as hs

UNRESOLVED = "UNRESOLVED"


@dataclass
class SequenceConfig:
    lx: float
    nx: int
    ly: float
    ny: int
    N: int
    D: float = 1.0
    q: float = 1.0
    s_max: int = 8
    first: str = "Y"  # D-049.3: the first half-step is by y with given x-functions
    qr: bool = True
    start: np.ndarray | None = None  # start set of the fixed direction of the first half-step; default T_N (sin^2 basis)

    def __post_init__(self):
        if self.first not in hs.DIRECTIONS:
            raise ValueError("first must be 'Y' or 'X'")
        if not (isinstance(self.s_max, int) and 1 <= self.s_max <= 8):
            raise ValueError("s_max must be an integer 1..8")
        if not (isinstance(self.N, int) and 1 <= self.N <= 3):
            raise ValueError("N must be 1, 2 or 3")


@dataclass
class SequenceResult:
    config: SequenceConfig
    steps: list[dict] = field(default_factory=list)
    status: str = "RUNNING"
    X: np.ndarray | None = None  # final x-factor
    Y: np.ndarray | None = None  # final y-factor


def _orient(direction: str, given: np.ndarray, other: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(x-factor, y-factor) of a field given = fixed-direction factor, other = solved-direction factor."""
    return (given, other) if direction == "Y" else (other, given)


def _exact_product(X: np.ndarray, Y: np.ndarray) -> list[list[Fraction]]:
    Xf = [[Fraction(float(v)) for v in row] for row in X]
    Yf = [[Fraction(float(v)) for v in row] for row in Y]
    return [[sum((a * b for a, b in zip(x, y)), Fraction(0)) for y in Yf] for x in Xf]


def _rank_diagnostic(G: np.ndarray, hg: float, V: np.ndarray, hv: float) -> dict:
    """R4 5.2 / 5.4.1, OBSERVATION: rho^_N = sigma_N/sigma_1 of R_G R_V^T, kappa_rep, sigma_N(R_V D_c), cond(R_V)^2."""
    _, RG = np.linalg.qr(math.sqrt(hg) * G)
    QV, RV = np.linalg.qr(math.sqrt(hv) * V)
    sig = np.linalg.svd(RG @ RV.T, compute_uv=False)
    sv = np.linalg.svd(RV, compute_uv=False)
    Dc = np.diag(1.0 / np.linalg.norm(RV, axis=0))
    return {"sigma": [float(x) for x in sig], "rho_N": float(sig[-1] / sig[0]) if sig[0] > 0 else 0.0,
            "kappa_rep": float(np.linalg.norm(RG, 2) * np.linalg.norm(RV, 2) / sig[0]) if sig[0] > 0 else math.inf,
            "sigma_N_RV_Dc": float(np.linalg.svd(RV @ Dc, compute_uv=False)[-1]),
            "cond_A_before_QR": float((sv[0] / sv[-1]) ** 2) if sv[-1] > 0 else math.inf}, QV, RV


def _certificate(p: hs.HalfStep, s: np.ndarray, coeffs, r_exact) -> dict:
    out = {}
    lam = float(scipy.linalg.eigvalsh(hs.coefficient_system(p)[0], subset_by_index=[0, 0])[0])
    low = hs.certify_lambda_min(p, lam / 2, coeffs) if lam > 0 else None
    if low is None:
        out["certificate"] = "CERT_UNRESOLVED (lambda_min(K) not certified)"
        out["set_bound_2"] = out["field_bound_a"] = UNRESOLVED
        return out
    out["certificate"] = "CERTIFIED (Q1b: exact defect of the decimal Cholesky; exact residual)"
    out["lambda_low"] = low
    out["set_bound_2"] = hs.error_bound(p, s, low, coeffs)
    r_up = en.sqrt_interval(sum((x * x for x in r_exact), Fraction(0)))[1]
    bK = r_up / en.sqrt_interval(Fraction(low))[0]
    h_s = Fraction(p.l_s) / p.n_s
    out["field_bound_a_exact"] = en.sqrt_interval(h_s)[1] * bK
    out["field_bound_a"] = en.up(out["field_bound_a_exact"])
    return out


def run_sequence(cfg: SequenceConfig, energy: bool = True) -> SequenceResult:
    """Run the research sequence (module docstring). ``energy`` = False skips the exact energy analysis (faster)."""
    res = SequenceResult(cfg)
    g = en.Grid(cfg.lx, cfg.nx, cfg.ly, cfg.ny, cfg.D, cfg.q)
    n_fix = cfg.nx if cfg.first == "Y" else cfg.ny
    n_other = cfg.ny if cfg.first == "Y" else cfg.nx
    given = np.array(hs.start_set(n_fix, cfg.N) if cfg.start is None else cfg.start, dtype=float)
    companion = np.zeros((n_other - 1, cfg.N))
    prev_eps = None  # VOK-5 bound of the previous field (formed from the same arrays)
    for s in range(1, cfg.s_max + 1):
        direction = cfg.first if s % 2 else ("X" if cfg.first == "Y" else "Y")
        p = hs.HalfStep(cfg.lx, cfg.nx, cfg.ly, cfg.ny, direction, given, D=cfg.D, q=cfg.q)
        step = {"s": s, "direction": direction, "N": cfg.N, "n_f": p.n_f, "n_s": p.n_s,
                "run_class": "N = 1 (I-4 normalisation)" if cfg.N == 1 else
                ("RESEARCH_ONLY: QR without an approved rank criterion" if cfg.qr else "RESEARCH_ONLY: no QR"),
                "marking": "PROVISIONAL / NOT_QUALIFIED (D-049.5)" if cfg.N >= 2 else "OBSERVATION"}
        if hs.given_set_status(p) == hs.EXACTLY_DEPENDENT:
            step["status"] = "STOP: given set exactly dependent (exact rank < N)"
            res.steps.append(step)
            res.status = step["status"]
            break
        ac = hs.solve_ac(p)
        if ac.s is None:
            step["status"] = f"STOP: {ac.status}"
            res.steps.append(step)
            res.status = step["status"]
            break
        m = p.n_s - 1
        V = ac.s.reshape(cfg.N, m).T  # solved-direction factor ((n_s - 1) x N)
        coeffs = hs.exact_coefficients(p)
        r_exact = hs.residual_exact(p, ac.s, coeffs)
        step.update(_certificate(p, ac.s, coeffs, r_exact))
        step["eta_qa"] = hs.eta_qa(p, ac.s, coeffs) if (np.any(ac.s) or cfg.q != 0) else UNRESOLVED
        step["e_AC_estimate"] = ac.e_AC
        if energy:
            step.update(_energy(g, p, direction, given, companion, V, r_exact, prev_eps, step))
        # transformation for the next half-step
        h_g = (cfg.lx / cfg.nx) if direction == "Y" else (cfg.ly / cfg.ny)
        h_s = (cfg.ly / cfg.ny) if direction == "Y" else (cfg.lx / cfg.nx)
        if cfg.N == 1:
            f = V[:, 0]
            if not np.any(f):
                step["status"] = "STOP: ZERO_FIELD (computed factor is 0, e.g. q = 0)"
                res.steps.append(step)
                res.status = step["status"]
                break
            sigma = 1.0 if np.sum(f) >= 0 else -1.0
            c = sigma * float(np.max(np.abs(f)))
            V = (f / c)[:, None]
            given = given * c
            step["normalisation_c"] = c
        elif cfg.qr:
            if hs.exact_rank(V) < cfg.N:
                step["status"] = "STOP: S-1, computed set exactly dependent (D-049.5, item 2)"
                res.steps.append(step)
                res.status = step["status"]
                break
            diag, QV, RV = _rank_diagnostic(given, h_g, V, h_s)
            step.update(diag)
            step["rank_decision"] = "UNRESOLVED (no approved numerical rank criterion; D-049.5, D-049.6)"
            V = QV / math.sqrt(h_s)
            given = given @ RV.T
            eA = np.linalg.eigvalsh(h_s * (V.T @ V))
            step["cond_A_after_QR"] = float(eA[-1] / eA[0])
        else:  # N >= 2 without QR: diagnostic only (OBSERVATION), no transformation
            step.update(_rank_diagnostic(given, h_g, V, h_s)[0])
            step["rank_decision"] = "UNRESOLVED (no approved numerical rank criterion; no QR in this run)"
        if energy:
            Xn, Yn = _orient(direction, given, V)
            Xo, Yo = _orient(direction, step.pop("_given_before"), step.pop("_V_before"))
            P_before = step["P_exact_s"]
            P_after = en.pi_exact(g, _exact_product(Xn, Yn))
            step["P_after_transform_minus_before"] = float(P_after - P_before)  # exact difference, then rounded (OBSERVATION)
            step["P_exact_s_after_transform"] = float(P_after)
            prev_eps = en.eps_a_C(g, Xn, Yn)
        step.setdefault("status", "OK")
        res.steps.append(step)
        # next half-step: the solved factor becomes the given set; the old given set becomes the companion
        given, companion = V, given
    else:
        res.status = f"COMPLETED s = 1..{cfg.s_max} (no stop criterion applied)"
    if res.status == "RUNNING":
        res.status = "COMPLETED"
    last = res.steps[-1] if res.steps else None
    if last is not None and last.get("status") == "OK":
        res.X, res.Y = _orient(last["direction"], companion, given)
    return res


def _energy(g, p, direction, given, companion, V, r_exact, prev_eps, step) -> dict:
    """Exact energy analysis of half-step s (module docstring, item 4)."""
    out = {}
    Xp, Yp = _orient(direction, given, companion)
    Xs, Ys = _orient(direction, given, V)
    w_prev, w_s = _exact_product(Xp, Yp), _exact_product(Xs, Ys)
    d = [[a - b for a, b in zip(ra, rb)] for ra, rb in zip(w_prev, w_s)]
    P_prev, P_s = en.pi_exact(g, w_prev), en.pi_exact(g, w_s)
    Q = en.norm_a_sq_exact(g, d) / 2
    h_s = Fraction(p.l_s) / p.n_s
    N, m = p.N, p.n_s - 1
    Delta = [Fraction(float(companion[j, i])) - Fraction(float(V[j, i])) for i in range(N) for j in range(m)]  # term-first
    gamma = h_s * sum((a * b for a, b in zip(r_exact, Delta)), Fraction(0))
    out["P_exact_prev"], out["P_exact_s"] = P_prev, P_s
    out["Q_exact"] = float(Q)
    out["gamma_exact"] = float(gamma)
    out["vok2_identity_exact"] = (P_prev - P_s) == Q - gamma
    ws_a2 = en.norm_a_sq_exact(g, w_s)
    if ws_a2 > 0:
        lo, hi = en.sqrt_interval(en.norm_a_sq_exact(g, d))
        wl, wh = en.sqrt_interval(ws_a2)
        out["eps_E_exact_down"], out["eps_E_exact_up"] = en.down(lo / wh), en.up(hi / wl)
    else:
        out["eps_E_exact_down"] = out["eps_E_exact_up"] = UNRESOLVED
    # computed check in double and its exact decomposition (VOK-6, VOK-8)
    wh_prev, wh_s = Xp @ Yp.T, Xs @ Ys.T
    dh = wh_prev - wh_s
    Pp_d, Ps_d, Q_d = en.pi_double(g, wh_prev), en.pi_double(g, wh_s), en.q_double(g, dh)
    K, rhs = hs.coefficient_system(p)
    s_vec = V.T.reshape(-1)
    rho_d = rhs - K @ s_vec
    Delta_d = (companion - V).T.reshape(-1)
    gamma_d = float(p.l_s / p.n_s) * float(rho_d @ Delta_d)
    computed = (Fraction(Pp_d) - Fraction(Ps_d)) - Fraction(Q_d)  # exact value of the computed numbers' combination
    diff_hat = [[Fraction(float(a)) - Fraction(float(b)) for a, b in zip(ra, rb)] for ra, rb in zip(wh_prev, wh_s)]
    f_hat = en.pi_exact(g, wh_prev) - en.pi_exact(g, wh_s) - en.norm_a_sq_exact(g, diff_hat) / 2  # f(w^(s-1), w^(s)), exact
    out["P_double_prev"], out["P_double_s"], out["Q_double"], out["gamma_double"] = Pp_d, Ps_d, Q_d, gamma_d
    out["defect_computed"] = float(computed)  # dP^ - Q^ (should be -gamma^ up to rounding)
    out["part_solution"] = float(-gamma)  # f(w~) = -gamma_s (VOK-2)
    out["part_formation"] = float(f_hat - (-gamma))  # f(w^) - f(w~), exact
    out["part_assembly"] = float(computed - f_hat)  # (dP^ - Q^) - f(w^), exact
    out["gamma_computation_error"] = float(Fraction(gamma_d) - gamma)
    out["m_s_numerator"] = float(abs(computed + Fraction(gamma_d)))
    # model terms (MODEL; c_R not assigned)
    Rp, Rs = en.model_R(g, wh_prev), en.model_R(g, wh_s)
    RQ = en.model_RQ(g, dh)
    eps_s = en.eps_a_C(g, Xs, Ys)
    eps_prev = prev_eps if prev_eps is not None else (Fraction(0) if not np.any(companion) else en.eps_a_C(g, Xp, Yp))
    cert = step.get("field_bound_a_exact")
    Bw, how = en.b_omega(g, wh_s, eps_s, cert)
    d_a = en.sqrt_interval(en.norm_a_sq_exact(g, [[Fraction(float(x)) for x in row] for row in dh]))[1]
    Rf = eps_s * d_a + Bw * (eps_prev + eps_s)
    out.update({"R_base_up": en.up(Fraction(Rp["R_base_up"]) + Fraction(Rs["R_base_up"]) + Fraction(RQ["RQ_base_up"])),
                "R_per_cR_up": en.up(Fraction(Rp["R_per_cR_up"]) + Fraction(Rs["R_per_cR_up"]) + Fraction(RQ["RQ_per_cR_up"])),
                "R_f_up": en.up(Rf), "B_omega_up": en.up(Bw), "B_omega_route": how,
                "eps_a_C_s_up": en.up(eps_s), "model_status": "MODEL (VOK-3, VOK-8); c_R, K_E not assigned"})
    out["_given_before"], out["_V_before"] = given.copy(), V.copy()
    return out


def journal(res: SequenceResult) -> list[dict]:
    """Flat journal rows (Fractions converted to directed intervals)."""
    rows = []
    for st in res.steps:
        row = {}
        for k, v in st.items():
            if k.startswith("_"):
                continue
            if isinstance(v, Fraction):
                row[f"{k}_down"], row[f"{k}_up"] = en.interval(v)
            elif isinstance(v, list):
                row[k] = ";".join(f"{x:.6e}" for x in v)
            else:
                row[k] = v
        rows.append(row)
    return rows
