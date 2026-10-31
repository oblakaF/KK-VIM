"""R4, group B (B-PREQUALIFICATION, N = 2-3): accuracy of the original N-term KK-VIM against the exact SSSS plate.

QA research script (D-016, D-025, D-046; R4_NUMERICAL_SPEC 9.2, 9.4, 9.5; MATH_SPEC part B, Б.12-Б.13). No PASS/FAIL
on the accuracy of KK-VIM: the errors are measured, every comparison carries the budget of its references and gets
UNRESOLVED where the budget exceeds 1/100 of the measured quantity in the same relative norm (D-046).

Three fields, never mixed:
    w_h^(N,s)     discrete N-term sequence      QA discrete control (kkvim_reference.r4_spectral_control; the same
                                                 algorithm as run_control, spectral data only - checked against it)
    w_cont^(N,s)  continuous N-term sequence    matrix Parseval recursion (kkvim_reference.r4_continuous_sequence)
    omega_0       exact plate                   Levy series (kkvim_reference.levy) and the energy reference
                                                 (kkvim_reference.r4_energy_reference)
    omega_h       full 2D discrete solution     closed spectral form (SpectralGrid.w_star)
and E_disc = w_h - w_cont, E_method = w_cont - omega_0, E_total = w_h - omega_0 = E_disc + E_method (signed).

Energy error (MATH_SPEC Б.13, В-23): err_abs^2 = |w_cont - omega_0|_{H_T}^2 = 2 [Pi(w_cont) - Pi(omega_0)],
e_rel = err_abs^2 / |omega_0|^2 = [Pi(w_cont) - Pi(omega_0)] / [-Pi(omega_0)]. Two evaluations:
    route E  - difference of the two energies (budgets of Pi(w_cont) and Pi(omega_0) separately);
    route D  - own evaluation (level Б, Parseval): |v|_{H_T}^2 = D (ab/4) pi^4 sum_{m,n} lam_mn v_mn^2 with
               lam_mn = (m^2/a^2 + n^2/b^2)^2 for v = sum v_mn sin(m pi x/a) sin(n pi y/b) (В-9: |v|_{H_T} = D^1/2 |Delta v|),
               applied to v = w_cont - omega_0: a sum of non-negative terms, no cancellation between the two energies.
               Outside the computed block (m or n > M) w_cont has no coefficients; the omega_0 part there is bounded
               strictly by the energy tail (r4_energy_reference.S_TAIL).

M denotes the LARGEST KEPT ODD INDEX; the number of odd modes per axis is (M + 1)/2 (M = 4001: 2001 modes).
The continuous recursion truncated at M is itself an exact KK-VIM sequence of the problem restricted to the sines
with odd indices <= M in both directions (every half-step decouples over the modes of the computed function); its field is a legitimate
element of H_T, so Pi and |.|_{H_T} of it are exact functionals of that field. Its distance from the untruncated
w_cont is the truncation of the reference: diagnosed by doubling M (not a proof; transfer over s - Q-022).

Two statuses are kept apart (review b0c208d, И-1): status_meas_* - the measurement of the FINITE spectral problem
(budget = strict tails + rounding models with stated premises; the rounding of the recursion arrays themselves is not
in the budget - checked by high precision: QA at M <= 81, Reviewer at M = 4001); status_cont_* - the statement on the
UNTRUNCATED continuous KK-VIM: s = 1 RESOLVED/UNRESOLVED with the strict Navier-m tail, s >= 2 at best PROVISIONAL
(Q-022). Budgets are written in three separate parts: *_strict_rel, *_model_rel, *_diag_rel (diagnostics: doubling
of M, least squares). Routes D and E share the coefficient arrays of the recursion.

Reproduction (repository root):
    uv run python scripts/r4_group_b_study.py                       # all tables except high precision
    uv run --with mpmath python scripts/r4_group_b_study.py --hp    # + temporary high-precision checks (QA only)
Outputs: CSV files and tables.md in docs/results/r4_group_b/ (--out). Options: --quick (small set for tests),
--max-n (largest grid of the square chains), --no-grid (continuous part only).
"""

import argparse
import csv
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np

from kkvim_reference.levy import levy_centre_140, levy_centre_bounds, levy_centre_j, levy_field_bounds, levy_series_w_grid
from kkvim_reference.r4_continuous_sequence import continuous_sequence, field_at_nodes, omega0_coefficients
from kkvim_reference.r4_energy_reference import ROUNDING_REL, S_TAIL, omega0_energy
from kkvim_reference.r4_levy_partial import navier_m_rounding_bound, navier_m_tail_bound, navier_m_truncated_at_nodes
from kkvim_reference.r4_spectral_control import SpectralGrid, _weighted_lstsq, odd_sine_set, run_control

U = 2.0**-53
S_LIST = (1, 2, 3, 4, 10, 20, 40)  # fixed before the comparison (task B-1)
S_MAX = max(S_LIST)
NS = (2, 3)
R_INITIAL = {"1/5": 0.2, "1/3": 1 / 3, "1/2": 0.5, "1": 1.0, "2": 2.0, "3": 3.0, "5": 5.0}
R_EXTRA = {"2/3": 2 / 3, "7/5": 1.4, "3/2": 1.5}  # the ten control r of R3_NUMERICAL_SPEC (line 78)
R_ALL = {**R_INITIAL, **R_EXTRA}
M_CONT = 4001  # largest kept odd index of the continuous recursion, both axes (2001 odd modes per axis)
M_CHECK = 8001  # doubling diagnostic (largest odd index; 4001 odd modes per axis)
M_ENERGY = 4001  # Pi(omega_0), odd m, n <= 4001: strict tail (a^4+b^4) pi^2/(80 M^5), rounding 24u (count)
SAMPLE = 64  # nodes per side of the sampling grid of E_inf for the continuous part
SQUARE_CHAIN = (40, 80, 160, 320, 640, 1280)
P_WINDOW = (1.9, 2.1)  # asymptotic window accepted for R2 (NUMERICAL_SPEC 12) - used here as a research criterion only
RULE = 0.01  # D-046: budget of a reference <= 1/100 of the measured error
REFERENCE_VALUES = {  # preliminary values of the task (B-2), signed relative centre errors, r = 1, continuous method
    (3, 1): 1.5e-4, (3, 2): -8.4e-7, (3, 3): -2.7e-6, (3, 10): -3.63e-6,
    (2, 1): -8.8e-4, (2, 2): -1.9e-5, (2, 3): -3.5e-5, (2, 10): -3.75e-5,
}
REFERENCE_FULL_GRID_640 = -2.76e-7  # (omega_h - omega_0)(centre) / omega_0(centre), r = 1, n = 640 (task B-4)


def status(budget, measured):
    """Status of a MEASUREMENT: RESOLVED if the budget of the references (strict parts + rounding models, with their
    stated premises) is <= 1/100 of the measured quantity in the same relative norm (D-046), else UNRESOLVED."""
    if not (math.isfinite(budget) and math.isfinite(measured)):
        return "UNRESOLVED"
    return "RESOLVED" if budget <= RULE * abs(measured) else "UNRESOLVED"


def continuous_status(s, meas_status, budget_s1, diagnostic, measured):
    """Status of the SCIENTIFIC statement on the untruncated continuous KK-VIM (report 2.3; review b0c208d, И-1).

    s = 1: the difference from the untruncated half-step is strictly bounded (Navier-m tail), so RESOLVED is possible
    at the level of the measurement - budget_s1 includes that strict tail (not a normative qualification of the s = 1
    reference, РВ4-8). s >= 2: no strict bound of the transfer of the truncation over the half-steps (Q-022): at best
    PROVISIONAL - the measurement is RESOLVED and the diagnostics (doubling of M, least squares) are <= 1/100;
    otherwise UNRESOLVED."""
    if s == 1:
        return status(budget_s1, measured)
    if meas_status == "RESOLVED" and status(diagnostic, measured) == "RESOLVED":
        return "PROVISIONAL"
    return "UNRESOLVED"


# ------------------------------------------------------------------------------------------------------------------
# Continuous quantities (pure functions of the coefficient arrays; checked by tests/cross/test_r4_group_b_study.py)
# ------------------------------------------------------------------------------------------------------------------


def odd_modes(M):
    """The odd indices 1, 3, ..., M (M = largest kept odd index)."""
    return np.arange(1, M + 1, 2, dtype=float)


def mode_count(M):
    """Number of odd modes per axis for the largest kept odd index M: (M + 1)/2 (M = 81 -> 41, 4001 -> 2001)."""
    if isinstance(M, bool) or not isinstance(M, int) or M < 1 or M % 2 == 0:
        raise ValueError(f"M must be a positive odd integer (largest kept odd index), got {M!r}")
    return (M + 1) // 2


_LAM = {}


def lam_weights(a, b, M):
    key = (a, b, M)
    if key not in _LAM:
        m = odd_modes(M)
        _LAM.clear()
        _LAM[key] = (m[:, None] ** 2 / a**2 + m[None, :] ** 2 / b**2) ** 2
    return _LAM[key]


def psum(T):
    """Sum of a 2D array of non-negative terms: pairwise row sums (numpy) and math.fsum of the rows."""
    return math.fsum(np.sum(T, axis=1).tolist())


def ht_norm2(V, a, b, D, M):
    """|v|_{H_T}^2 = D (ab/4) pi^4 sum lam_mn v_mn^2 (Parseval, В-9), v_mn the odd-mode sine coefficients."""
    return D * a * b / 4.0 * math.pi**4 * psum(lam_weights(a, b, M) * V * V)


def l2_norm2(V, a, b):
    return a * b / 4.0 * psum(V * V)


def direct_rounding(x, y, w0, err2, a, b, D, M):
    """First-order model of the error of ht_norm2(C - w0): entries C_mn = x_m . y_n ((N + 1)u of |x_m||y_n|) and w0_mn
    (12u), delta_mn <= (N + 12)u (|x_m||y_n| + |w0_mn|); |err2~ - err2| <= 2 err delta_H + delta_H^2 (Cauchy-Schwarz),
    plus the summation (log2 of the number of terms, u each, of err2)."""
    N = x.shape[1]
    d = (N + 12) * U * (np.abs(x) @ np.abs(y).T + np.abs(w0))
    dh = math.sqrt(ht_norm2(d, a, b, D, M))
    return 2.0 * math.sqrt(err2) * dh + dh * dh + 2.0 * math.log2(d.size) * U * err2


def omega0_ht_outside(a, b, q, D, M):
    """Strict bound of the omega_0 part of |.|_{H_T}^2 outside the block m, n <= M: (64 q^2 ab/(pi^8 D)) (S - S_M)."""
    return 64.0 * q * q * a * b / (math.pi**8 * D) * S_TAIL(a, b, M)


def omega0_l2_outside(a, b, q, D, M):
    """Strict bound of the omega_0 part of |.|_{L2}^2 outside the block: (ab/4) sum w_mn^2 over m or n > M.

    w_mn^2 = 256 q^2/(pi^12 D^2 m^2 n^2 lam_mn^2), lam_mn^2 >= (m/a)^8 resp. (n/b)^8, sum_{odd} k^-2 = pi^2/8,
    sum_{odd k > M} k^-10 <= 1/(18 M^9) (integral of x^-10 / 2 from M)."""
    c = 256.0 * q * q / (math.pi**12 * D * D)
    return a * b / 4.0 * c * (math.pi**2 / 8.0) * (a**8 + b**8) / (18.0 * M**9)


def centre_signs(K):
    """sin(m pi/2) for odd m = 1, 3, ... (K values): +1, -1, +1, ..."""
    return np.where(np.arange(K) % 2 == 0, 1.0, -1.0)


def centre_value(x, y):
    """w(a/2, b/2) = sum_j X_j(a/2) Y_j(b/2) of a step of the continuous sequence."""
    sx, sy = centre_signs(x.shape[0]), centre_signs(y.shape[0])
    return float(np.dot(sx @ x, sy @ y))


def centre_rounding(x, y):
    """First-order rounding model of centre_value: gamma_K for the two sums, gamma_N for the product sum."""
    K, N = x.shape
    g = (K + N + 1) * U
    return g * float(np.dot(np.abs(x).sum(axis=0), np.abs(y).sum(axis=0)))


def energy_rounding_model(x, y, a, b, q, D, modes):
    """First-order rounding model of Pi(w) by r4_continuous_sequence._energy (MODEL, not a strict bound).

    Gram entries (l/2) fsum(k^p c_i c_j): k = m pi/l (2.35u), k^4 by pow (4 x 2.35u + 2u), two products, fsum, x l/2:
    <= 15.4u of the absolute Gram G_abs; norm2 = D fsum(gx ay + 2 bx by + ax gy): two Gram errors + product + fsum + x D;
    integral: x 2l/(m pi) (4u), fsum, products: 7u. Total <= 36u of the sum of absolute values S_abs (+ O(u^2))."""
    def abs_grams(c, length):
        k = modes * math.pi / length
        ca = np.abs(c)
        return [0.5 * length * (ca.T * k**p) @ ca for p in (0, 2, 4)]
    ax, bx, gx = abs_grams(x, a)
    ay, by, gy = abs_grams(y, b)
    norm_abs = D * float(np.sum(gx * ay + 2.0 * bx * by + ax * gy))
    ix = (np.abs(x) * (2.0 * a / (modes * math.pi))[:, None]).sum(axis=0)
    iy = (np.abs(y) * (2.0 * b / (modes * math.pi))[:, None]).sum(axis=0)
    return 36.0 * U * (0.5 * norm_abs + abs(q) * float(np.dot(ix, iy)))


def continuous_case(a, b, q, D, N, M=M_CONT, M_check=M_CHECK, s_max=S_MAX):
    """Rows of the continuous study (B-1, B-2, B-5) for one (N, a, b): every s = 1..s_max."""
    steps = continuous_sequence(a, b, N, s_max, M, q=q, D=D)
    check = continuous_sequence(a, b, N, s_max, M_check, q=q, D=D)
    ls = continuous_sequence(a, b, N, min(s_max, 10), M, q=q, D=D, method="ls")
    modes = odd_modes(M)
    K = len(modes)
    w0 = omega0_coefficients(a, b, q, D, M)
    w0_check = omega0_coefficients(a, b, q, D, M_check)
    pi0, pi0_budget = omega0_energy(a, b, q, D, M=M_ENERGY)
    norm0 = -2.0 * pi0  # |omega_0|_{H_T}^2 = q J
    out_ht = omega0_ht_outside(a, b, q, D, M)
    l2_0 = l2_norm2(w0, a, b)
    out_l2 = omega0_l2_outside(a, b, q, D, M)
    # omega_0 at the centre (Levy (140)) and on the sampling grid (Levy (139))
    c0 = levy_centre_140(a=a, b=b, q=q, D=D)
    cb = levy_centre_bounds(a=a, b=b, q=q, D=D)
    c0_budget = cb.truncation + cb.rounding_model
    c0_j = levy_centre_j(a=a, b=b, q=q, D=D)
    xs, ys = np.linspace(0.0, a, SAMPLE + 1), np.linspace(0.0, b, SAMPLE + 1)
    W0 = levy_series_w_grid(xs, ys, a=a, b=b, q=q, D=D)
    lb = levy_field_bounds(a=a, b=b, q=q, D=D)
    W0_max = float(np.max(np.abs(W0)))
    pi0_tail = pi0_budget - ROUNDING_REL * abs(pi0)  # omega0_energy returns tail + 24u |Pi|; split back
    pi0_round = ROUNDING_REL * abs(pi0)
    # s = 1: the untruncated first half-step is the m-truncated Navier sum with ALL n (ВБ-23); the computed one keeps
    # n <= M. The difference is strictly bounded: in H_T^2 by the omega_0 part outside the block (out_ht), pointwise by
    # the Navier-m tail (R4_NUMERICAL_SPEC 9.1 (б)). For s >= 2 no strict bound of the transfer exists (Q-022).
    modes_L = list(range(1, 2 * N, 2))
    s1_point_tail = navier_m_tail_bound(a, b, q, D, modes_L, M)
    rows, prev = [], None
    for st, ck in zip(steps, check):
        s = st.s
        C = st.x @ st.y.T
        Cc = ck.x @ ck.y.T
        dv = C - w0
        err2_block = ht_norm2(dv, a, b, D, M)
        err2_check = ht_norm2(Cc - w0_check, a, b, D, M_check)
        err2_direct = err2_block + 0.5 * out_ht  # midpoint of [block, block + out_ht]
        # route E: difference of the energies. strict: tail of Pi(omega_0); model: 24u count of Pi(omega_0) and the
        # rounding model of Pi(w); diagnostic: doubling of M.
        dpi = st.energy - pi0
        pi_w_round = energy_rounding_model(st.x, st.y, a, b, q, D, modes)
        pi_w_trunc = abs(st.energy - ck.energy)
        strict_E, model_E, diag_E = pi0_tail, pi0_round + pi_w_round, pi_w_trunc
        # route D: strict - omega_0 outside the block (half width of [block, block + out_ht]); model - rounding of the
        # positive sum of the coefficients' differences; diagnostic - doubling of M. Routes D and E use the SAME
        # coefficient arrays of the recursion: their agreement checks the evaluation of the norm, not the recursion.
        strict_D, model_D = 0.5 * out_ht, direct_rounding(st.x, st.y, w0, err2_block, a, b, D, M)
        diag_D = abs(err2_check - err2_block)
        # point values: strict - Levy truncation; model - Levy rounding model and the rounding of the centre sum;
        # diagnostic - doubling of M and the independent least squares (centre only, s <= 10)
        wc = centre_value(st.x, st.y)
        wc_check = centre_value(ck.x, ck.y)
        wc_ls = centre_value(ls[s - 1].x, ls[s - 1].y) if s <= len(ls) else math.nan
        strict_c, model_c = cb.truncation, cb.rounding_model + centre_rounding(st.x, st.y)
        diag_c = abs(wc - wc_check) + (abs(wc - wc_ls) if s <= len(ls) else 0.0)
        dc = (wc - c0) / c0
        Wc = field_at_nodes(st, SAMPLE, SAMPLE)
        Wc_check = field_at_nodes(ck, SAMPLE, SAMPLE)
        e_inf = float(np.max(np.abs(Wc - W0))) / W0_max
        strict_inf, model_inf = lb.truncation, lb.rounding_model + K * U * float(np.max(np.abs(Wc)))
        diag_inf = float(np.max(np.abs(Wc - Wc_check)))
        l2 = l2_norm2(dv, a, b) + 0.5 * out_l2
        e_rel = err2_direct / norm0
        row = {
            "N": N, "s": s, "r": b / a, "a": a, "b": b, "q": q, "D": D, "M": M, "modes_per_axis": mode_count(M),
            "Pi_w": st.energy, "Pi_0": pi0, "Pi_0_tail_strict": pi0_tail, "Pi_0_rounding_count": pi0_round,
            "Pi_w_rounding_model": pi_w_round, "Pi_w_truncation_diag": pi_w_trunc, "dPi_energy_route": dpi,
            "e_rel_energy_route": dpi / (-pi0),
            "err2_direct": err2_direct, "err_abs": math.sqrt(err2_direct), "e_rel": e_rel, "sqrt_e_rel": math.sqrt(e_rel),
            # route D (main value e_rel): budget parts relative to err^2
            "D_strict_rel": strict_D / err2_direct, "D_model_rel": model_D / err2_direct, "D_diag_rel": diag_D / err2_direct,
            # route E: budget parts relative to Pi(w) - Pi(omega_0)
            "E_strict_rel": strict_E / dpi, "E_model_rel": model_E / dpi, "E_diag_rel": diag_E / dpi,
            "route_E_minus_D_over_budget_E": (2.0 * dpi - err2_direct) / (2.0 * (strict_E + model_E + diag_E)),
            "w_cont_centre": wc, "omega0_centre": c0, "omega0_centre_budget": c0_budget,
            "omega0_centre_140_minus_j": (c0 - c0_j) / c0,
            "delta_c": dc, "c_strict_rel": strict_c / abs(c0 * dc), "c_model_rel": model_c / abs(c0 * dc),
            "c_diag_rel": diag_c / abs(c0 * dc),
            "w_cont_centre_ls_minus_rec": (wc_ls - wc) / c0 if s <= len(ls) else math.nan,
            "w_cont_centre_M_doubling": (wc_check - wc) / c0,
            "E_inf_sampled": e_inf, "inf_strict_rel": strict_inf / (W0_max * e_inf),
            "inf_model_rel": model_inf / (W0_max * e_inf), "inf_diag_rel": diag_inf / (W0_max * e_inf),
            "L2_rel": math.sqrt(l2 / l2_0),
            "given_cond": st.given_cond,
        }
        # two statuses (report, section 2.3): measurement of the finite spectral problem (odd modes <= M) and the
        # scientific statement on the untruncated continuous KK-VIM
        row["status_meas_e_rel"] = status(strict_D + model_D, err2_direct)
        row["status_meas_e_rel_E"] = status(strict_E + model_E, dpi) if dpi > 0 else "UNRESOLVED"
        row["status_meas_delta_c"] = status(strict_c + model_c, c0 * dc)
        row["status_meas_E_inf"] = status(strict_inf + model_inf, W0_max * e_inf)
        row["status_cont_e_rel"] = continuous_status(s, row["status_meas_e_rel"], strict_D + model_D + out_ht, diag_D,
                                                     err2_direct)
        row["status_cont_e_rel_E"] = continuous_status(s, row["status_meas_e_rel_E"], strict_E + model_E + 0.5 * out_ht,
                                                       diag_E, dpi)
        row["status_cont_delta_c"] = continuous_status(s, row["status_meas_delta_c"], strict_c + model_c + s1_point_tail,
                                                       diag_c, c0 * dc)
        row["status_cont_E_inf"] = continuous_status(s, row["status_meas_E_inf"], strict_inf + model_inf + s1_point_tail,
                                                     diag_inf, W0_max * e_inf)
        if prev is not None:
            pst, perr2, pC = prev
            inc2 = ht_norm2(C - pC, a, b, D, M)  # |w_s - w_{s-1}|_{H_T}^2 (exact identity: = 2 (Pi_{s-1} - Pi_s))
            row.update({
                "rel_energy_change_energy_route": (pst.energy - st.energy) / abs(st.energy),
                "rel_energy_change_direct": 0.5 * inc2 / abs(st.energy),
                "err2_decrease": perr2 - err2_block,
                "pythagoras_defect_rel": (perr2 - err2_block - inc2) / perr2,
                "increment_rel": math.sqrt(inc2 / ht_norm2(C, a, b, D, M)),
            })
        rows.append(row)
        prev = (st, err2_block, C)
    return rows


def n23_transfer_case(a, b, q, D, N, M=M_CONT, s_max=S_MAX):
    """QA diagnostic of docs/R4_N23_QUALIFICATION.plan.md, run ONLY with --n23-transfer; changes no other output.

    On the arrays of continuous_case (the same call of continuous_sequence) evaluates in double the transfer bound of the
    truncation E'_s (Corollary N23-6: theta'_s = min(1, E'_{s-1} / (D^1/2 lambda11 sigma_N^L2(w'_{s-1}))), no stop), the
    monotone upper bound U_s of the untruncated energy error (Corollary N23-7) and adds E'_s to the measurement budget of
    group B (strict + model) in the same relative norm of every measure (plan, sec. 4). The rounding of the inputs of E'_s
    is bounded separately (scripts/r4_n23_qualification.py); the statuses status_cont_* are not changed (D-048.10)."""
    steps = continuous_sequence(a, b, N, s_max, M, q=q, D=D)
    modes = odd_modes(M)
    K = len(modes)
    w0 = omega0_coefficients(a, b, q, D, M)
    out_ht = omega0_ht_outside(a, b, q, D, M)
    T = math.sqrt(out_ht)  # strict bound of |omega_0 - omega_M|_{H_T} (Q22, sec. 2)
    lam11 = math.pi**2 * (1.0 / a**2 + 1.0 / b**2)
    pi0, pi0_budget = omega0_energy(a, b, q, D, M=M_ENERGY)
    beta_pi = pi0_budget / abs(pi0)
    c0 = levy_centre_140(a=a, b=b, q=q, D=D)
    cb = levy_centre_bounds(a=a, b=b, q=q, D=D)
    xs, ys = np.linspace(0.0, a, SAMPLE + 1), np.linspace(0.0, b, SAMPLE + 1)
    W0 = levy_series_w_grid(xs, ys, a=a, b=b, q=q, D=D)
    lb = levy_field_bounds(a=a, b=b, q=q, D=D)
    # Riesz constant of the centre (Q22-6, item 1); for odd-mode fields also of the sup norm (Q22, З-3)
    gamma2 = psum(1.0 / (D * a * b / 4.0 * math.pi**4 * lam_weights(a, b, M)))
    gamma2 += (a * b) ** 2 / (4.0 * math.pi**4) * 2.0 * (1.0 / (2.0 * M)) * (math.pi**2 / 8.0) / (D * a * b / 4.0)
    gc = math.sqrt(gamma2)
    rows, prev, E, upper, sig_prev = [], None, None, math.inf, None
    for st in steps:
        s = st.s
        C = st.x @ st.y.T
        err2_block = ht_norm2(C - w0, a, b, D, M)
        Rs = math.sqrt(T * T + err2_block)
        sig = math.sqrt(a * b) / 2.0 * float(np.linalg.svd(np.linalg.qr(st.x)[1] @ np.linalg.qr(st.y)[1].T,
                                                            compute_uv=False)[N - 1])
        if prev is None:
            d, theta, E = math.nan, math.nan, T
        else:
            d = math.sqrt(ht_norm2(C - prev, a, b, D, M))
            theta = min(1.0, E / (math.sqrt(D) * lam11 * sig_prev))
            E = math.sqrt((T + theta * Rs) ** 2 + (E + theta * d) ** 2)
        upper = min(upper, math.sqrt(err2_block + T * T) + E)
        err2 = err2_block + 0.5 * out_ht  # the measured value of continuous_case (err2_direct)
        err = math.sqrt(err2)
        beta_D = (0.5 * out_ht + direct_rounding(st.x, st.y, w0, err2_block, a, b, D, M)) / err2
        beta_err = 1.0 - math.sqrt(1.0 - beta_D)
        qs = E / (err * (1.0 - beta_err))
        wc = centre_value(st.x, st.y)
        dcabs = abs(wc - c0)
        beta_c = (cb.truncation + cb.rounding_model + centre_rounding(st.x, st.y)) / dcabs
        Wc = field_at_nodes(st, SAMPLE, SAMPLE)
        einf = float(np.max(np.abs(Wc - W0)))
        beta_inf = (lb.truncation + lb.rounding_model + K * U * float(np.max(np.abs(Wc)))) / einf
        row = {"N": N, "s": s, "r": b / a, "M": M, "T": T, "lambda11": lam11, "Gamma_c": gc,
               "sigma_N_L2": sig, "d_norm": d, "err_M": math.sqrt(err2_block), "R_s": Rs, "theta": theta,
               "E_transfer": E, "U_upper": upper, "err2_direct": err2, "err_abs": err, "e_rel": err2 / (-2.0 * pi0),
               "w_cont_centre": wc, "delta_c": (wc - c0) / c0, "E_inf_abs": einf,
               "beta_D": beta_D, "beta_err": beta_err, "beta_Pi0": beta_pi, "beta_c": beta_c, "beta_inf": beta_inf,
               "R_err_abs": beta_err + qs, "R_e_rel": beta_D + beta_pi + 2.0 * qs + qs * qs,
               "R_delta_c": beta_c + gc * E / (dcabs * (1.0 - beta_c)),
               "R_E_inf": beta_inf + gc * E / (einf * (1.0 - beta_inf)),
               "RU_err_abs": (upper - err) / err + beta_err, "RU_e_rel": (upper * upper - err2) / err2 + beta_D + beta_pi}
        for key in ("err_abs", "e_rel", "delta_c", "E_inf"):
            row[f"rule_{key}"] = bool(row[f"R_{key}"] <= RULE)
        for key in ("err_abs", "e_rel"):
            row[f"rule_upper_{key}"] = bool(row[f"RU_{key}"] <= RULE)
        rows.append(row)
        prev, sig_prev = C, sig
    return rows


def scale_invariance(N, r, cases=((2.9, 3.7, 0.013), (0.37, -1.0, 5.0))):
    """Relative quantities for (a, b = r a, q, D) scaled against a = q = D = 1 (task B-2)."""
    base = {row["s"]: row for row in continuous_case(1.0, r, 1.0, 1.0, N, M_check=2001)}
    out = []
    for a, q, D in cases:
        for row in continuous_case(a, r * a, q, D, N, M_check=2001):
            b0 = base[row["s"]]
            if row["s"] in S_LIST:
                out.append({"N": N, "r": r, "a": a, "q": q, "D": D, "s": row["s"],
                            "d_e_rel": (row["e_rel"] - b0["e_rel"]) / b0["e_rel"],
                            "delta_c": row["delta_c"], "d_delta_c_abs": row["delta_c"] - b0["delta_c"],
                            "d_E_inf_abs": row["E_inf_sampled"] - b0["E_inf_sampled"]})
    return out


def centre_decomposition(N, r, M=M_CONT):
    """Signed contributions to delta_c of the mode groups (own analysis): x-modes m <= 2N-1 (L) or > 2N-1 (H), the same
    for the y-modes n; delta_c = sum over the groups LL, LH, HL, HH of (C_mn - w_mn) sin(m pi/2) sin(n pi/2) / omega_0(c)
    plus the omega_0 tail outside the block (|.| <= sum_{m or n > M} |w_mn|, reported as a bound)."""
    a, b = 1.0, r
    steps = continuous_sequence(a, b, N, S_MAX, M)
    w0 = omega0_coefficients(a, b, 1.0, 1.0, M)
    K = w0.shape[0]
    sg = centre_signs(K)
    S = np.outer(sg, sg)
    c0 = levy_centre_140(a=a, b=b, q=1.0, D=1.0)
    low = np.arange(K) < N  # odd m <= 2N - 1
    out = []
    for st in steps:
        if st.s not in S_LIST:
            continue
        dv = st.x @ st.y.T - w0
        E = dv * S / c0
        H = lam_weights(a, b, M) * dv * dv
        h_total = psum(H)
        row = {"N": N, "r": r, "s": st.s, "delta_c": float(E.sum()),
               "e_rel": ht_norm2(dv, a, b, 1.0, M) / (-2.0 * omega0_energy(a, b)[0])}
        for gx, mx in (("L", low), ("H", ~low)):
            for gy, my in (("L", low), ("H", ~low)):
                row[f"part_{gx}{gy}"] = math.fsum(E[np.ix_(mx, my)].ravel().tolist())
                row[f"energy_share_{gx}{gy}"] = psum(H[np.ix_(mx, my)]) / h_total
        row["parts_abs_sum"] = sum(abs(row[f"part_{g}"]) for g in ("LL", "LH", "HL", "HH"))
        row["cancellation"] = row["parts_abs_sum"] / abs(row["delta_c"])
        out.append(row)
    return out


def s1_against_navier(N, r):
    """s = 1 of the recursion against the m-truncated Navier sum (R4_NUMERICAL_SPEC 9.1 (б), strict tail and rounding)."""
    st = continuous_sequence(1.0, r, N, 1, M_CONT)[0]
    modes = list(range(1, 2 * N, 2))
    Mn = 40001
    Wn = navier_m_truncated_at_nodes(1.0, r, 1.0, 1.0, modes, SAMPLE, SAMPLE, Mn)
    Wc = field_at_nodes(st, SAMPLE, SAMPLE)
    budget = navier_m_tail_bound(1.0, r, 1.0, 1.0, modes, Mn) + navier_m_rounding_bound(1.0, r, 1.0, 1.0, modes, Mn)
    m = float(np.max(np.abs(Wn)))
    return {"N": N, "r": r, "max_diff_rel": float(np.max(np.abs(Wc - Wn))) / m, "navier_budget_rel": budget / m,
            "recursion_tail_bound_rel": st.tail_bound / m}


# ------------------------------------------------------------------------------------------------------------------
# Discrete quantities
# ------------------------------------------------------------------------------------------------------------------


def discrete_sequence(g, N, s_keep):
    """The QA control sequence (run_control's algorithm: QR of the given set, weighted least squares by Householder QR)
    in spectral data only; returns {s: (W_hat, U_hat, V_hat, ls_cond_max)} for s in s_keep."""
    sq = np.sqrt(g.lam)
    U_hat, V_hat = g.psi_x @ odd_sine_set(g.nx, N), None
    out = {}
    for s in range(1, max(s_keep) + 1):
        if s % 2:
            U_hat, _ = np.linalg.qr(U_hat)
            V_hat, cond = _weighted_lstsq(sq.T, U_hat, g.w_star.T)
        else:
            V_hat, _ = np.linalg.qr(V_hat)
            U_hat, cond = _weighted_lstsq(sq, V_hat, g.w_star)
        if s in s_keep:
            out[s] = (U_hat @ V_hat.T, U_hat.copy(), V_hat.copy(), float(np.max(cond)))
    return out


def discrete_err2(g, W_hat):
    """|w - omega_h|_h^2 = D h_x h_y sum lam (W^ - W*)^2 (module docstring, item 2): a non-negative sum."""
    d = W_hat - g.w_star
    return g.D * g.hx * g.hy * math.fsum((g.lam * d * d).ravel().tolist())


POINTS = {"c": (1, 2, 1, 2), "q14": (1, 4, 1, 4), "q12": (1, 4, 1, 2), "q21": (1, 2, 1, 4)}  # (x num, den, y num, den)


def node_index(n, num, den):
    if (n * num) % den:
        return None
    return n * num // den  # node index including the edge node 0


def grid_case(N, lab, r, nx, ny, cont_steps, lev_cache, base, cont_e_rel):
    """Rows of the grid study (B-3, B-4) for one grid; cont_steps: continuous steps (M_CONT) of the same N, r."""
    a, b, q, D = 1.0, r, 1.0, 1.0
    g = SpectralGrid(a, b, nx, ny, q, D)
    seq = discrete_sequence(g, N, S_LIST)
    key = (r, nx, ny)
    if key not in lev_cache:
        xs, ys = np.arange(nx + 1) * (a / nx), np.arange(ny + 1) * (b / ny)
        lev_cache.clear()
        lev_cache[key] = levy_series_w_grid(xs, ys, a=a, b=b, q=q, D=D)
    W0 = lev_cache[key][1:-1, 1:-1]
    W0max = float(np.max(np.abs(W0)))
    Wh_full = g.to_nodal(g.w_star)
    pi_h_omega = g.omega_energy()
    norm_h = -2.0 * pi_h_omega
    pi0, _ = omega0_energy(a, b, q, D, M=M_ENERGY)
    rows = []
    for s in S_LIST:
        W_hat, U_hat, V_hat, cond = seq[s]
        Wh = g.to_nodal(W_hat)
        Wc = field_at_nodes(cont_steps[s - 1], nx, ny)[1:-1, 1:-1]
        err2_h = discrete_err2(g, W_hat)
        row = {"N": N, "r_label": lab, "r": r, "a": a, "b": b, "nx": nx, "ny": ny, "s": s,
               "E_inf_disc": float(np.max(np.abs(Wh - Wc))) / W0max,
               "E_inf_method": float(np.max(np.abs(Wc - W0))) / W0max,
               "E_inf_total": float(np.max(np.abs(Wh - W0))) / W0max,
               "E_inf_full_grid": float(np.max(np.abs(Wh_full - W0))) / W0max,
               "e_rel_h": err2_h / norm_h, "e_rel_cont": cont_e_rel[s],
               "e_rel_h_minus_cont_rel": (err2_h / norm_h - cont_e_rel[s]) / cont_e_rel[s], "ls_cond_max": cond,
               "Pi_h_w": g.spectral_energy(W_hat), "Pi_h_omega_h": pi_h_omega,
               "Pi_h_omega_h_minus_Pi_0_rel": (pi_h_omega - pi0) / abs(pi0)}
        for name, (xn, xd, yn, yd) in POINTS.items():
            i, j = node_index(nx, xn, xd), node_index(ny, yn, yd)
            if i is None or j is None:
                continue
            w0 = W0[i - 1, j - 1]
            row[f"d_disc_{name}"] = (Wh[i - 1, j - 1] - Wc[i - 1, j - 1]) / w0
            row[f"d_method_{name}"] = (Wc[i - 1, j - 1] - w0) / w0
            row[f"d_total_{name}"] = (Wh[i - 1, j - 1] - w0) / w0
            row[f"d_full_grid_{name}"] = (Wh_full[i - 1, j - 1] - w0) / w0
        fx, fy = nx // base[0], ny // base[1]
        sub = (slice(fx - 1, None, fx), slice(fy - 1, None, fy))  # nodes of the coarsest grid of the chain
        row["_fields"] = (Wh[sub], Wc[sub], W0[sub])  # kept in memory for the Richardson table, not written
        rows.append(row)
    return rows


def check_light_path(N, r, nx, ny):
    """The spectral-only path equals run_control (same operations) - field difference relative to max|w|."""
    g = SpectralGrid(1.0, r, nx, ny)
    ctrl = run_control(g, N, 10)
    light = discrete_sequence(g, N, (1, 2, 3, 4, 10))
    return max(float(np.max(np.abs(g.to_nodal(light[s][0]) - ctrl[s - 1].W))) / float(np.max(np.abs(ctrl[s - 1].W)))
               for s in (1, 2, 3, 4, 10))


def richardson_rows(chain_rows):
    """Observed orders and Richardson extrapolation on a chain of nested grids (fixed N, s, r); coarse-grid nodes."""
    out = []
    by_s = {}
    for row in chain_rows:
        by_s.setdefault(row["s"], []).append(row)
    for s, rows in by_s.items():
        rows = sorted(rows, key=lambda t: t["nx"])
        W0max = float(np.max(np.abs(rows[0]["_fields"][2])))
        E = [t["_fields"][:2] for t in rows]
        for k in range(len(rows) - 1):
            (h1, c1), (h2, c2) = E[k], E[k + 1]
            d1, d2 = h1 - c1, h2 - c2
            e1, e2 = float(np.max(np.abs(d1))), float(np.max(np.abs(d2)))
            rec = {"N": rows[k]["N"], "r_label": rows[k]["r_label"], "s": s, "nx": rows[k]["nx"], "ny": rows[k]["ny"],
                   "nx2": rows[k + 1]["nx"], "ny2": rows[k + 1]["ny"],
                   "E_disc_coarse_n": e1 / W0max, "E_disc_coarse_2n": e2 / W0max,
                   "p_max": math.log2(e1 / e2) if e2 > 0 else math.nan,
                   "dc_n": rows[k].get("d_disc_c", math.nan), "dc_2n": rows[k + 1].get("d_disc_c", math.nan)}
            dc1, dc2 = rec["dc_n"], rec["dc_2n"]
            rec["p_centre"] = math.log2(dc1 / dc2) if dc1 * dc2 > 0 else math.nan
            WR = (4.0 * h2 - h1) / 3.0
            rec["richardson_residual"] = float(np.max(np.abs(WR - c1))) / W0max
            if k + 2 < len(rows):
                h3, c3 = E[k + 2]
                pred = d2 / 4.0  # E_disc(4n) predicted from h^2 on the pair (n, 2n)
                rec["prediction_error_4n"] = float(np.max(np.abs((h3 - c3) - pred))) / W0max
                rec["E_disc_coarse_4n"] = float(np.max(np.abs(h3 - c3))) / W0max
            out.append(rec)
    return out


# ------------------------------------------------------------------------------------------------------------------
# High precision (temporary mpmath; QA only, not a dependency)
# ------------------------------------------------------------------------------------------------------------------


def hp_checks():
    import mpmath as mp
    mp.mp.dps = 40
    rows = []
    for N in NS:
        for lab in ("1/5", "1", "5"):
            r = R_INITIAL[lab]
            for M in (41, 81):
                a, b = mp.mpf(1), mp.mpf(r)
                modes = [mp.mpf(k) for k in range(1, M + 1, 2)]
                K = len(modes)

                def halfstep(coef, lg, ln):
                    # orthonormalise (modified Gram-Schmidt in mp), then the recursion of R4_NUMERICAL_SPEC 9.2
                    Q = [list(col) for col in zip(*coef)]
                    for i in range(N):
                        for j in range(i):
                            d = mp.fsum(Q[i][k] * Q[j][k] for k in range(K))
                            Q[i] = [Q[i][k] - d * Q[j][k] for k in range(K)]
                        nrm = mp.sqrt(mp.fsum(v * v for v in Q[i]))
                        Q[i] = [v / nrm for v in Q[i]]
                    kk = [m * mp.pi / lg for m in modes]
                    G = [mp.matrix(N, N) for _ in range(3)]
                    for p, Gp in zip((0, 2, 4), G):
                        for i in range(N):
                            for j in range(N):
                                Gp[i, j] = lg / 2 * mp.fsum(kk[k] ** p * Q[i][k] * Q[j][k] for k in range(K))
                    F = mp.matrix([mp.fsum(Q[j][k] * 2 * lg / (modes[k] * mp.pi) for k in range(K)) for j in range(N)])
                    new = []
                    for k in range(K):
                        nu = modes[k] * mp.pi / ln
                        A = nu**4 * G[0] + 2 * nu**2 * G[1] + G[2]
                        v = mp.lu_solve(A, F * (4 / (modes[k] * mp.pi)))
                        new.append([v[i] for i in range(N)])
                    return [[Q[i][k] for i in range(N)] for k in range(K)], new

                x = [[mp.mpf(1) if k == j else mp.mpf(0) for j in range(N)] for k in range(K)]
                y = None
                dbl = continuous_sequence(1.0, r, N, 10, M)
                sgn = [1 if k % 2 == 0 else -1 for k in range(K)]
                for s in range(1, 11):
                    if s % 2:
                        x, y = halfstep(x, a, b)
                    else:
                        y, x = halfstep(y, b, a)
                    C = [[mp.fsum(x[m][j] * y[n][j] for j in range(N)) for n in range(K)] for m in range(K)]
                    lam = [[(modes[m] ** 2 / a**2 + modes[n] ** 2 / b**2) ** 2 for n in range(K)] for m in range(K)]
                    st = dbl[s - 1]
                    Cd = st.x @ st.y.T
                    diff = mp.fsum(lam[m][n] * (mp.mpf(Cd[m, n]) - C[m][n]) ** 2 for m in range(K) for n in range(K))
                    nrm = mp.fsum(lam[m][n] * C[m][n] ** 2 for m in range(K) for n in range(K))
                    wc = mp.fsum(sgn[m] * sgn[n] * C[m][n] for m in range(K) for n in range(K))
                    # energy of the stored double arrays, exactly
                    xs = [[mp.mpf(v) for v in row] for row in st.x]
                    ys = [[mp.mpf(v) for v in row] for row in st.y]
                    Ce = [[mp.fsum(xs[m][j] * ys[n][j] for j in range(N)) for n in range(K)] for m in range(K)]
                    norm2 = (a * b / 4) * mp.pi**4 * mp.fsum(lam[m][n] * Ce[m][n] ** 2 for m in range(K) for n in range(K))
                    integral = mp.fsum(Ce[m][n] * 4 * a * b / (mp.pi**2 * modes[m] * modes[n]) for m in range(K) for n in range(K))
                    pi_exact = norm2 / 2 - integral
                    rows.append({"N": N, "r": r, "M": M, "s": s,
                                 "field_ht_rel_double_vs_hp": float(mp.sqrt(diff / nrm)),
                                 "centre_rel_double_vs_hp": float((centre_value(st.x, st.y) - wc) / wc),
                                 "Pi_double_minus_exact_rel": float((st.energy - pi_exact) / abs(pi_exact)),
                                 "Pi_rounding_model_rel": energy_rounding_model(st.x, st.y, 1.0, r, 1.0, 1.0,
                                                                                 odd_modes(M)) / float(abs(pi_exact))})
    # omega_0 at the centre: Levy (140) in double against 40 digits
    for lab, r in R_ALL.items():
        rL = max(r, 1 / r)
        aL4 = min(1.0, r) ** 4
        c = mp.pi * mp.mpf(rL) / 2
        terms = []
        for m in range(1, 401, 2):
            al = m * c
            p = (al * mp.tanh(al) + 2) / (2 * mp.cosh(al))
            terms.append((-1) ** ((m - 1) // 2) * p / mp.mpf(m) ** 5)
        wc = (mp.mpf(5) / 384 - 4 / mp.pi**5 * mp.fsum(terms)) * aL4
        dbl = levy_centre_140(a=1.0, b=r, q=1.0, D=1.0)
        cb = levy_centre_bounds(a=1.0, b=r, q=1.0, D=1.0)
        rows.append({"omega0_centre_r": r, "levy140_double_minus_hp_rel": float((dbl - wc) / wc),
                     "levy140_budget_rel": (cb.truncation + cb.rounding_model) / float(wc)})
    return rows


# ------------------------------------------------------------------------------------------------------------------
# Output
# ------------------------------------------------------------------------------------------------------------------


def write_csv(path, rows, spec=".6e"):
    keys = []
    for row in rows:
        for k in row:
            if not k.startswith("_") and k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
        w.writeheader()
        for row in rows:
            w.writerow({k: (format(v, spec) if isinstance(v, float) else v) for k, v in row.items() if not k.startswith("_")})


def read_csv(path):
    def conv(v):
        try:
            return int(v)
        except ValueError:
            try:
                return float(v)
            except ValueError:
                return v
    with open(path, encoding="utf-8") as f:
        return [{k: (v if k in ("r_label", "family") else conv(v)) for k, v in row.items()} for row in csv.DictReader(f)]


def fmt(v, spec=".3e"):
    if isinstance(v, str):
        return v
    if v is None or v == "" or (isinstance(v, float) and math.isnan(v)):
        return "—"
    if isinstance(v, (int, np.integer)) and not isinstance(v, bool):
        return str(v)
    return format(v, spec)


def md_table(header, rows):
    lines = ["| " + " | ".join(h.replace("|", r"\|") for h in header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(fmt(v) if not isinstance(v, str) else v for v in row) + " |" for row in rows]
    return "\n".join(lines)


def make_tables(out):
    """master.csv (the columns required by the task) and tables.md (the tables of the report) from the CSV files."""
    cont = read_csv(out / "continuous.csv")
    cidx = {(t["N"], t["r_label"], t["s"]): t for t in cont}
    grid = read_csv(out / "grid.csv") if (out / "grid.csv").exists() else []
    master = []
    for g in grid:
        c = cidx[(g["N"], g["r_label"], g["s"])]
        master.append({
            "N": g["N"], "s": g["s"], "r": g["r_label"], "a": g["a"], "b": g["b"], "n_x": g["nx"], "n_y": g["ny"],
            "family": g["family"], "Pi_w_cont": c["Pi_w"], "Pi_omega0": c["Pi_0"], "err_abs": c["err_abs"], "e_rel": c["e_rel"],
            "sqrt_e_rel": c["sqrt_e_rel"],
            "delta_c_method": c["delta_c"], "E_inf_method_sampled": c["E_inf_sampled"],
            "delta_c_disc": g.get("d_disc_c", math.nan), "delta_c_total": g.get("d_total_c", math.nan),
            "delta_c_full_grid": g.get("d_full_grid_c", math.nan),
            "E_inf_disc": g["E_inf_disc"], "E_inf_method_nodes": g["E_inf_method"], "E_inf_total": g["E_inf_total"],
            "E_inf_full_grid": g["E_inf_full_grid"], "e_rel_h": g["e_rel_h"],
            "budget_meas_e_rel_rel": c["D_strict_rel"] + c["D_model_rel"], "diag_e_rel_rel": c["D_diag_rel"],
            "budget_meas_delta_c_rel": c["c_strict_rel"] + c["c_model_rel"], "diag_delta_c_rel": c["c_diag_rel"],
            "status_meas_e_rel": c["status_meas_e_rel"], "status_cont_e_rel": c["status_cont_e_rel"],
            "status_meas_delta_c": c["status_meas_delta_c"], "status_cont_delta_c": c["status_cont_delta_c"],
            "status_disc_vs_method_centre": ("DISTINGUISHABLE" if abs(g.get("d_disc_c", math.inf)) <= RULE * abs(c["delta_c"])
                                             else "NOT_DISTINGUISHABLE"),
        })
    if master:
        write_csv(out / "master.csv", master)
    parts = []
    # T1: continuous, r = 1
    for N in NS:
        rows = [(t["s"], t["Pi_w"], t["Pi_0"], t["err_abs"], t["e_rel"], t["sqrt_e_rel"], t["D_strict_rel"],
                 t["D_model_rel"], t["D_diag_rel"], t["e_rel_energy_route"], t["E_strict_rel"], t["E_model_rel"],
                 t["E_diag_rel"], t.get("rel_energy_change_direct", math.nan), t["delta_c"], t["c_strict_rel"],
                 t["c_model_rel"], t["c_diag_rel"], t["E_inf_sampled"], t["L2_rel"],
                 f'{t["status_meas_e_rel"]} / {t["status_cont_e_rel"]}', f'{t["status_meas_delta_c"]} / {t["status_cont_delta_c"]}')
                for t in cont if t["N"] == N and t["r_label"] == "1" and t["s"] in S_LIST]
        parts.append(f"### T1-N{N}: finite spectral problem (odd modes <= {M_CONT}, {mode_count(M_CONT)} per axis) and the "
                     "untruncated continuous method; r = 1, a = b = 1, q = D = 1\n\n" + md_table(
            ["s", "Π(w_cont)", "Π(ω₀)", "err_abs", "e_rel (D)", "√e_rel", "D: строгое/err²", "D: модель/err²",
             "D: диагн./err²", "e_rel (E)", "E: строгое/ΔΠ", "E: модель/ΔΠ", "E: диагн./ΔΠ", "ΔΠ_s/|Π_s|", "δ_c",
             "δ_c: строгое", "δ_c: модель", "δ_c: диагн.", "E_∞ (65²)", "L₂",
             "статус e_rel: измерение / непрерывный метод", "статус δ_c: измерение / непрерывный метод"], rows))
    # T2: all r
    rows = []
    for N in NS:
        for lab in R_ALL:
            ts = {t["s"]: t for t in cont if t["N"] == N and t["r_label"] == lab}
            if not ts:
                continue
            rows.append((N, lab, *[v for k in (1, 2, 3, 10, 40) for v in (ts[k]["e_rel"], ts[k]["sqrt_e_rel"])],
                         ts[1]["status_cont_e_rel"], ts[2]["status_cont_e_rel"]))
    parts.append("### T2a: energy error, all r (b = r, a = 1): e_rel and sqrt(e_rel)\n\n" + md_table(
        ["N", "r", "e_rel s=1", "√e_rel s=1", "e_rel s=2", "√ s=2", "e_rel s=3", "√ s=3", "e_rel s=10", "√ s=10",
         "e_rel s=40", "√ s=40", "непр. метод s=1", "непр. метод s≥2"], rows))
    rows = []
    for N in NS:
        for lab in R_ALL:
            ts = {t["s"]: t for t in cont if t["N"] == N and t["r_label"] == lab}
            if not ts:
                continue
            meas_D = max(t["D_strict_rel"] + t["D_model_rel"] for t in ts.values())
            meas_E = max(t["E_strict_rel"] + t["E_model_rel"] for t in ts.values())
            meas_c = max(t["c_strict_rel"] + t["c_model_rel"] for t in ts.values())
            diag = max(max(t["D_diag_rel"], t["E_diag_rel"], t["c_diag_rel"]) for t in ts.values())
            meas_all = {t[k] for t in ts.values() for k in ("status_meas_e_rel", "status_meas_e_rel_E", "status_meas_delta_c",
                                                            "status_meas_E_inf")}
            rows.append((N, lab, ts[1]["delta_c"], ts[2]["delta_c"], ts[3]["delta_c"], ts[10]["delta_c"], ts[40]["delta_c"],
                         ts[10]["E_inf_sampled"], meas_D, meas_E, meas_c, diag, "/".join(sorted(meas_all)),
                         max(t["given_cond"] for t in ts.values())))
    parts.append("### T2b: centre and budgets, all r: measurement budget = strict + model (relative), diagnostics separately\n\n"
                 + md_table(["N", "r", "δ_c s=1", "s=2", "s=3", "s=10", "s=40", "E_∞ s=10", "max изм. бюджет/err² (D)",
                             "max изм. бюджет/ΔΠ (E)", "max изм. бюджет/|δ_c|", "max диагностика (D, E, δ_c)",
                             "статусы измерения", "max cond заданного набора"], rows))
    # T3: monotonicity. Noise level of the computed fields: eps_rec = largest relative H_T difference of the double
    # recursion from the 40-digit one (T10); without it 1e-13 (prequalification report 6.3: <= 1e-13).
    hp = read_csv(out / "high_precision.csv") if (out / "high_precision.csv").exists() else []
    eps_rec = max([t["field_ht_rel_double_vs_hp"] for t in hp if isinstance(t.get("field_ht_rel_double_vs_hp"), float)]
                  or [1e-13])
    rows = []
    for N in NS:
        for lab in R_ALL:
            ts = [t for t in cont if t["N"] == N and t["r_label"] == lab and t["s"] >= 2]
            if not ts:
                continue
            # err2 of a field with relative noise eps in H_T: relative error ~ 2 eps |w|/err = 2 eps / sqrt(e_rel)
            tol = [2.0 * eps_rec / math.sqrt(t["e_rel"]) for t in ts]
            viol = sum(1 for t, e in zip(ts, tol) if t["err2_decrease"] / t["err2_direct"] < -e)
            inc = [t for t in ts if t["increment_rel"] > 100.0 * eps_rec]
            neg_E = [t for t in ts if t["rel_energy_change_energy_route"] < 0]
            neg_E_resolved = sum(1 for t in neg_E
                                 if -t["rel_energy_change_energy_route"] * abs(t["Pi_w"]) > 2 * t["Pi_w_rounding_model"])
            rows.append((N, lab, max(abs(t["pythagoras_defect_rel"]) for t in ts),
                         min(t["err2_decrease"] / t["err2_direct"] for t in ts), viol,
                         max(t["s"] for t in inc) if inc else 1, len(neg_E), neg_E_resolved))
    parts.append(f"### T3: energy monotonicity of the computed continuous sequence (s = 2..40; eps_rec = {eps_rec:.1e})\n\n"
                 + md_table(["N", "r", "max |дефект Пифагора|/err²", "min (err²(s−1) − err²(s))/err²",
                             "нарушений сверх шума", "последнее s с |Δw|/|w| > 100·eps_rec", "ΔΠ<0 (путь E)",
                             "из них сверх модели округления"], rows))
    # T4: grid, r = 1
    for N in NS:
        rows = [(g["nx"], g["s"], g.get("d_disc_c"), g.get("d_method_c"), g.get("d_total_c"), g.get("d_full_grid_c"),
                 g["E_inf_disc"], g["E_inf_method"], g["E_inf_total"], g["E_inf_full_grid"], g["e_rel_h"], g["e_rel_cont"],
                 g["e_rel_h_minus_cont_rel"])
                for g in grid if g["N"] == N and g["r_label"] == "1" and g["family"] == "I" and g["s"] in (1, 2, 3, 10, 40)]
        parts.append(f"### T4-N{N}: grid, r = 1, n_x = n_y = n (signed centre errors relative to ω₀(c))\n\n" + md_table(
            ["n", "s", "δ_c disc", "δ_c method", "δ_c total", "δ_c full grid", "E_∞ disc", "E_∞ method", "E_∞ total",
             "E_∞ full grid", "e_rel,h", "e_rel", "(e_rel,h − e_rel)/e_rel"], rows))
    # T5: Richardson summary
    rich = read_csv(out / "richardson.csv") if (out / "richardson.csv").exists() else []
    rows = []
    for fam in ("I", "II"):
        for N in NS:
            for lab in R_ALL:
                rs = [t for t in rich if t["family"] == fam and t["N"] == N and t["r_label"] == lab]
                if not rs:
                    continue
                finest = max(t["nx"] for t in rs)
                top = [t for t in rs if t["nx"] == finest]
                pm = [t["p_max"] for t in rs]
                pc = [t["p_centre"] for t in rs if not (isinstance(t["p_centre"], float) and math.isnan(t["p_centre"]))]
                pred = [t["prediction_error_4n"] / t["E_disc_coarse_4n"] for t in rs
                        if isinstance(t.get("prediction_error_4n"), float) and t["E_disc_coarse_4n"] > 0]
                rows.append((fam, N, lab, f"{min(t['nx'] for t in rs)}…{finest}", min(pm), max(pm),
                             min(pc) if pc else math.nan, max(pc) if pc else math.nan,
                             min(t["p_max"] for t in top), max(t["p_max"] for t in top),
                             max(t["richardson_residual"] for t in top), max(t["E_disc_coarse_2n"] for t in top),
                             max(pred) if pred else math.nan))
    parts.append("### T5: observed order p (coarse-grid nodes, all s) and Richardson\n\n" + md_table(
        ["сем.", "N", "r", "n_x (пары)", "min p_∞", "max p_∞", "min p_c", "max p_c", "p_∞ (посл. пара) min",
         "max", "остаток Ричардсона (посл. пара)", "E_disc(2n) (посл. пара)",
         "max |E(4n) − E(2n)/4| / E(4n)"], rows))
    # T6: distinguishability (B-4)
    rows = []
    for fam, N, lab in [(f, N, lab) for f in ("I", "II") for N in NS for lab in R_ALL]:
        gs = [g for g in grid if g["N"] == N and g["r_label"] == lab and g["family"] == fam]
        if gs:
            nf = max(g["nx"] for g in gs)
            ny_f = max(g["ny"] for g in gs)
            for s in (1, 2, 10):
                g = next(t for t in gs if t["nx"] == nf and t["s"] == s)
                c = cidx[(N, lab, s)]
                need_c = nf * math.sqrt(abs(g["d_disc_c"]) / (RULE * abs(c["delta_c"])))
                need_inf = nf * math.sqrt(g["E_inf_disc"] / (RULE * g["E_inf_method"]))
                need_e = nf * math.sqrt(abs(g["e_rel_h_minus_cont_rel"]) / RULE)
                rows.append((fam, N, lab, s, f"({nf}, {ny_f})", g["d_disc_c"], c["delta_c"],
                             abs(g["d_disc_c"] / c["delta_c"]), g["E_inf_disc"] / g["E_inf_method"],
                             g["e_rel_h_minus_cont_rel"], f"{need_c:.0f}", f"{need_inf:.0f}", f"{need_e:.0f}",
                             g["ls_cond_max"]))
    parts.append("### T6: B-4 — distinguishability of the method error on the grid (finest grid of each chain)\n\n"
                 + md_table(["сем.", "N", "r", "s", "(n_x, n_y)", "δ_c disc", "δ_c method", "|disc/method| (центр)",
                             "E_∞ disc/method", "(e_rel,h−e_rel)/e_rel", "n_x для 1/100 (центр)*",
                             "n_x для 1/100 (E_∞)*", "n_x: согласие e_rel,h с e_rel (1 %)**", "cond МНК (макс.)"], rows))
    if hp:
        rows = []
        for N, r, M in sorted({(t["N"], t["r"], t["M"]) for t in hp if t.get("M") not in ("", None)}):
            ts = [t for t in hp if (t["N"], t["r"], t["M"]) == (N, r, M)]
            rows.append((N, r, M, max(abs(t["field_ht_rel_double_vs_hp"]) for t in ts),
                         max(abs(t["centre_rel_double_vs_hp"]) for t in ts),
                         max(abs(t["Pi_double_minus_exact_rel"]) for t in ts), max(t["Pi_rounding_model_rel"] for t in ts)))
        parts.append("### T10: high precision (40 digits, temporary mpmath; s = 1..10, maxima)\n\n" + md_table(
            ["N", "r", "M (наиб. нечётный индекс; мод на ось = (M+1)/2)", "поле (H_T), double vs 40 зн.", "центр",
             "Π(w) double vs точно (те же массивы)", "модель округления Π"], rows))
        rows = [(t["omega0_centre_r"], t["levy140_double_minus_hp_rel"], t["levy140_budget_rel"])
                for t in hp if t.get("omega0_centre_r") not in ("", None)]
        parts.append("### T10b: omega_0(centre), Levy (140) in double against 40 digits\n\n" + md_table(
            ["r", "(double − 40 зн.)/ω₀(c)", "бюджет Леви (хвост + модель)/ω₀(c)"], rows))
    for name, title in (("full_grid_r1.csv", "T7: omega_h - omega_0, r = 1 (centre, full 2D discretisation)"),
                        ("s1_navier.csv", "T8: s = 1, recursion against the m-truncated Navier sum"),
                        ("scale_invariance.csv", "T9: scale invariance of the relative quantities (differences)"),
                        ("centre_decomposition.csv", "T11: centre error by mode groups (L: m <= 2N-1)")):
        if (out / name).exists():
            rs = read_csv(out / name)
            keys = [k for k in rs[0]]
            parts.append(f"### {title}\n\n" + md_table(keys, [[t.get(k) for k in keys] for t in rs]))
    ref = []
    for (N, s), v in REFERENCE_VALUES.items():
        t = cidx[(N, "1", s)]
        ref.append((N, s, v, t["delta_c"], (t["delta_c"] - v) / abs(v)))
    parts.append("### T12: B-2 — preliminary values of the task against this study (r = 1)\n\n" + md_table(
        ["N", "s", "ориентир", "δ_c (это исследование)", "(δ_c − ориентир)/|ориентир|"], ref))
    (out / "tables.md").write_text("# R4 group B — generated tables (scripts/r4_group_b_study.py)\n\n"
                                   f"M = largest kept odd index; odd modes per axis = (M + 1)/2 (M = {M_CONT}: "
                                   f"{mode_count(M_CONT)}; M = {M_CHECK}: {mode_count(M_CHECK)}).\n\n"
                                   "Statuses: status_meas_* - measurement of the finite spectral problem (strict tails + "
                                   "rounding models; the rounding of the recursion arrays is not in the budget); "
                                   "status_cont_* - untruncated continuous KK-VIM: s = 1 with the strict Navier-m tail, "
                                   "s >= 2 at best PROVISIONAL (Q-022). Not a normative qualification (D-046, РВ4-8).\n\n"
                                   "* n for 1/100: estimate n_f (|E_disc(n_f)| / (0.01 |E_method|))^(1/2) from the "
                                   "observed h^2 law on the finest grid - a research estimate, not a requirement.\n"
                                   "** agreement of the two squared energy errors e_rel,h (discrete, against omega_h) and "
                                   "e_rel (continuous): NOT the energy norm |w_h - w_cont|_{H_T} of E_disc (not computed) "
                                   "and not a ratio |E_disc| / |E_method|.\n\n"
                                   + "\n\n".join(parts) + "\n", encoding="utf-8")


def chains(quick, max_n):
    """Grid chains (fixed N, r): family I n_x = n_y (all ten r); family II h_x = h_y (seven initial r)."""
    out = []
    sq = tuple(n for n in SQUARE_CHAIN if n <= max_n) if not quick else (20, 40, 80)
    for lab, r in (R_ALL.items() if not quick else (("1", 1.0), ("5", 5.0))):
        out.append(("I", lab, r, [(n, n) for n in sq]))
    if not quick:
        for lab, r in R_INITIAL.items():
            if r == 1.0:
                continue
            if r < 1:
                base = (round(20 / r), 20)
            else:
                base = (20, round(20 * r))
            out.append(("II", lab, r, [(base[0] * 2**k, base[1] * 2**k) for k in range(5)]))
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", default="docs/results/r4_group_b")
    p.add_argument("--hp", action="store_true", help="temporary high-precision checks (needs mpmath)")
    p.add_argument("--quick", action="store_true", help="small set (tests)")
    p.add_argument("--no-grid", action="store_true")
    p.add_argument("--max-n", type=int, default=1280)
    p.add_argument("--tables-only", action="store_true", help="regenerate master.csv and tables.md from the CSV files")
    p.add_argument("--n23-transfer", metavar="CSV", help="QA diagnostic only (docs/R4_N23_QUALIFICATION.plan.md): write the "
                   "N23-6/N23-7 transfer bounds on the continuous arrays to CSV (17 digits); no other output is written")
    args = p.parse_args(argv)
    if args.n23_transfer:
        rset = R_ALL if not args.quick else {"1": 1.0, "5": 5.0}
        rows = []
        for N in NS:
            for lab, r in rset.items():
                for row in n23_transfer_case(1.0, r, 1.0, 1.0, N):
                    rows.append({"r_label": lab, **row})
                print("n23-transfer", N, lab, flush=True)
        Path(args.n23_transfer).parent.mkdir(parents=True, exist_ok=True)
        write_csv(Path(args.n23_transfer), rows, spec=".17g")
        return
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.tables_only:
        make_tables(out)
        return
    t0 = time.time()
    rset = R_ALL if not args.quick else {"1": 1.0, "5": 5.0}
    cont, cont_steps = [], {}
    for N in NS:
        for lab, r in rset.items():
            rows = continuous_case(1.0, r, 1.0, 1.0, N)
            for row in rows:
                row["r_label"] = lab
            cont += rows
            cont_steps[(N, lab)] = continuous_sequence(1.0, r, N, S_MAX, M_CONT)
    write_csv(out / "continuous.csv", cont)
    print("continuous", round(time.time() - t0), flush=True)
    s1 = [s1_against_navier(N, r) for N in NS for r in rset.values()]
    write_csv(out / "s1_navier.csv", s1)
    write_csv(out / "centre_decomposition.csv", [row for N in NS for r in rset.values() for row in centre_decomposition(N, r)])
    if not args.quick:
        write_csv(out / "scale_invariance.csv", scale_invariance(3, 1.0) + scale_invariance(2, 2.0))
    # a partial run (--no-grid, no --hp) keeps the records of the parts it does not recompute
    old = json.loads((out / "meta.json").read_text(encoding="utf-8")) if (out / "meta.json").exists() else {}
    meta = {**old, "python": sys.version.split()[0], "numpy": np.__version__, "platform": platform.platform(),
            "M_CONT": M_CONT, "M_CHECK": M_CHECK, "M_ENERGY": M_ENERGY,
            "M_meaning": "largest kept odd index; odd modes per axis = (M + 1)/2",
            "modes_per_axis": {"M_CONT": mode_count(M_CONT), "M_CHECK": mode_count(M_CHECK), "M_ENERGY": mode_count(M_ENERGY)},
            "S_LIST": S_LIST, "RULE": RULE,
            "last_run_parts": ["continuous"] + ([] if args.no_grid else ["grid"]) + (["hp"] if args.hp else [])}
    meta.pop("seconds", None)
    if not args.no_grid:
        light = [check_light_path(N, r, nx, ny) for N, r, nx, ny in ((2, 1.0, 40, 40), (3, 0.2, 100, 20), (3, 5.0, 20, 100))]
        meta["light_path_vs_run_control_max"] = max(light)
        grid_rows, rich = [], []
        lev_cache = {}
        for fam, lab, r, chain in chains(args.quick, args.max_n):
            for N in NS:
                chain_rows = []
                cont_e = {row["s"]: row["e_rel"] for row in cont if row["N"] == N and row["r_label"] == lab}
                for nx, ny in chain:
                    rows = grid_case(N, lab, r, nx, ny, cont_steps[(N, lab)], lev_cache, chain[0], cont_e)
                    for row in rows:
                        row["family"] = fam
                    chain_rows += rows
                    print(fam, N, lab, nx, ny, round(time.time() - t0), flush=True)
                rich += [{**t, "family": fam} for t in richardson_rows(chain_rows)]
                for row in chain_rows:
                    row.pop("_fields")
                grid_rows += chain_rows
        write_csv(out / "grid.csv", grid_rows)
        write_csv(out / "richardson.csv", rich)
        if not args.quick:
            # omega_h at the centre on larger square grids (r = 1): the full 2D discretisation error alone
            full = []
            c0 = levy_centre_140(a=1.0, b=1.0, q=1.0, D=1.0)
            for n in (40, 80, 160, 320, 640, 1280, 2560):
                g = SpectralGrid(1.0, 1.0, n, n)
                row_c = g.psi_x[n // 2 - 1]
                wh = float(row_c @ g.w_star @ row_c)
                full.append({"n": n, "delta_c_full_grid": (wh - c0) / c0,
                             "e_Pi": (g.omega_energy() - omega0_energy(1.0, 1.0)[0]) / abs(omega0_energy(1.0, 1.0)[0])})
            write_csv(out / "full_grid_r1.csv", full)
    if args.hp:
        write_csv(out / "high_precision.csv", hp_checks())
    meta["seconds"] = round(time.time() - t0)
    (out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    make_tables(out)
    print("done", meta["seconds"])


if __name__ == "__main__":
    main()
