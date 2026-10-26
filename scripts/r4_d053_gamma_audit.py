"""QA audit of gamma_s in the R4 energy test (D-053.3): exact gamma against the binary64 value of the implementation.

Task: QA R4, D-053.3 (goals G3.3/G4, link to G3.4). Not Q4. The working solver is not changed; this tool only
records the arrays that ``kkvim.coupled_run.run_coupled`` passes to ``kkvim.coupled_energy.pair_energy`` at s = 2
and recomputes gamma_s and the energy test from them.

Levels (AGENTS.md, section 2): the energy identity is the project's discretization (R4_NUMERICAL_SPEC 6.2, ChV4-14,
item 3); tau_s^(N), E1, E2 and the resolution rule Q_s >= 10 tau_s are the accepted form D-048.1 (a first-order model,
6.3); everything here is QA engineering.

Reference (independent of ``pair_energy``, ``discrete_energy`` and ``coupled_residual``):

- the binary64 arrays of the half-step (given set U, solved set V, previous companion V_prev) and the binary64 grid
  steps h = length / n of the implementation are carried over EXACTLY as dyadic rationals (ints * 2**E);
- gamma_HP = h_s sum_i rho_i^T (V_prev,i - V_i) with rho = F (x) 1 - K vec(V) in exact rational arithmetic, where
  A, B, C, F are assembled exactly from U by their definitions (6.2; 2.2) and K is applied with the integer stencils
  and the exact h (no rounding at all; the reference error is zero up to the correctness of the program);
- self-check on every case: the exact energies of the exact products w~ = V U^T (separable form of a_h, l_h:
  a different path, it does not use A, B, C, F or K) satisfy Delta P~ - Q~ + gamma_HP = 0 exactly (6.2, item 3);
- self-check on small grids: full two-dimensional exact assembly of Pi_h (second independent path) and a
  Decimal(80 digits) evaluation of gamma (different arithmetic); a deliberate perturbation of V shows that the
  binary64 formula is accurate when gamma is far above its evaluation noise.

The variant "HP gamma" replaces ONLY gamma in tau_s^(N) = |gamma| + 10 [R(w_{s-1}) + R(w_s) + R_f^(N)]; Delta P, Q,
R, R_f stay the binary64 values of the implementation. No new threshold is introduced; m_2 >= 4 is the accepted
D-048 criterion (K-3, D-048.1). math.fsum on the rounded residual is NOT used as a reference.

Usage (from the repository root):
    uv run python scripts/r4_d053_gamma_audit.py run --profile full      # 72 mandatory s = 2 cases + self-checks
    uv run python scripts/r4_d053_gamma_audit.py run --profile ci --out scratch/r4_d053_ci
    uv run python scripts/r4_d053_gamma_audit.py selfcheck
"""

import argparse
import csv
import importlib.util
import json
import math
import platform
import subprocess
import sys
import time
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path

import numpy as np
import scipy

import kkvim.coupled_run as cr
from kkvim.coupled import coupled_residual
from kkvim.coupled_energy import E1_E2_HOLD, E1_VIOLATED, E2_VIOLATED, FACTOR_10, UNRESOLVED

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "results" / "r4_d053_gamma_audit"
B5_CSV = ROOT / "docs" / "results" / "r4_numerical_prequal" / "b5_energy.csv"
K3_CSV = ROOT / "docs" / "results" / "r4_d048_qualification" / "k3_energy.csv"
QA_TEST = ROOT / "tests" / "cross" / "test_r4_controller_comparison.py"

# B5 / K-3 sample (scripts/r4_numerical_prequal.py b5_energy; scripts/r4_d048_qualification.py k3_energy), not changed
R10 = {"1/5": 0.2, "1/3": 1 / 3, "1/2": 0.5, "2/3": 2 / 3, "1": 1.0, "7/5": 1.4, "3/2": 1.5, "2": 2.0, "3": 3.0, "5": 5.0}
FAMILY_I_N = (160, 640, 1280)
FAMILY_II = {"1/5": (1600, 320), "1/3": (960, 320), "1/2": (640, 320), "2": (320, 640), "3": (320, 960), "5": (320, 1600)}
M2_D048 = 4.0  # accepted D-048 criterion m_2 >= 4 (K-3, criterion 1; B5 (ii)); not a new threshold
SIDE_A, D_PLATE, Q_LOAD = 1.0, 1.0, 1.0  # a = 1, b = r, D = 1, q = 1 as in B5 / K-3


def mandatory_cases():
    grids = [("I", lab, r, n, n) for lab, r in R10.items() for n in FAMILY_I_N]
    grids += [("II", lab, R10[lab], nx, ny) for lab, (nx, ny) in FAMILY_II.items()]
    return [(fam, lab, r, nx, ny, N) for fam, lab, r, nx, ny in grids for N in (2, 3)]


# ---------------------------------------------------------------------------------------------------------------
# exact dyadic sets: binary64 values carried over exactly as ints * 2**E (one exponent per group of arrays)
# ---------------------------------------------------------------------------------------------------------------

def dyadic(*arrays):
    """Columns of each (m, N) array as Python int lists with ONE common exponent E: value = int * 2**E exactly."""
    flat = [np.asarray(a, dtype=np.float64) for a in arrays]
    for a in flat:
        if not np.all(np.isfinite(a)):
            raise ValueError("non-finite input")
    ratios = [[x.as_integer_ratio() for x in a.ravel(order="F").tolist()] for a in flat]
    K = max((q.bit_length() - 1 for rs in ratios for _, q in rs), default=0)
    out = []
    for a, rs in zip(flat, ratios):
        ints = [p << (K - (q.bit_length() - 1)) for p, q in rs]
        m = a.shape[0]
        cols = [ints[j * m:(j + 1) * m] for j in range(a.shape[1])]
        for j in range(a.shape[1]):  # exact round trip (no rounding in the transfer)
            for i in (0, m // 2, m - 1):
                assert Fraction(cols[j][i]) * Fraction(2) ** -K == Fraction(float(a[i, j]))
        out.append(cols)
    return out, -K


def p2(e):
    return Fraction(2) ** e


def d2(c):
    """Integer second difference with zero ends (h not included)."""
    p = [0] + c + [0]
    return [p[k] - 2 * p[k + 1] + p[k + 2] for k in range(len(c))]


def fwd(c):
    """Integer forward differences on all n intervals, ends included (h not included)."""
    p = [0] + c + [0]
    return [p[k + 1] - p[k] for k in range(len(c) + 1)]


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def exact_coefficients(U, E, h, q):
    """A, B, C, F of the given set U (int columns, exponent E) on a grid of exact step h (R4_NUMERICAL_SPEC 2.2)."""
    N = len(U)
    dU = [fwd(c) for c in U]
    lU = [d2(c) for c in U]
    s = p2(2 * E)
    A = [[h * dot(U[i], U[j]) * s for j in range(N)] for i in range(N)]
    Bm = [[dot(dU[i], dU[j]) * s / h for j in range(N)] for i in range(N)]
    C = [[dot(lU[i], lU[j]) * s / h**3 for j in range(N)] for i in range(N)]
    F = [q * h * sum(U[i]) * p2(E) for i in range(N)]
    return A, Bm, C, F


def exact_gamma(coef, V, Vprev, E, h, Dm):
    """gamma = h sum_i rho_i^T (Vprev_i - V_i), rho = F (x) 1 - K vec(V), exact; also max|rho| and rho as floats."""
    A, Bm, C, F = coef
    N = len(V)
    I2 = [d2(c) for c in V]
    I4 = [d2(c) for c in I2]
    dlt = [[a - b for a, b in zip(Vprev[i], V[i])] for i in range(N)]
    gamma = Fraction(0)
    rho_float = []
    rho_max = Fraction(0)
    for i in range(N):
        c4 = [Dm * A[i][j] * p2(E) / h**4 for j in range(N)]
        c2 = [-2 * Dm * Bm[i][j] * p2(E) / h**2 for j in range(N)]
        c0 = [Dm * C[i][j] * p2(E) for j in range(N)]
        den = 1
        for c in (F[i], *c4, *c2, *c0):
            den = math.lcm(den, Fraction(c).denominator)
        f = int(F[i] * den)
        n4, n2, n0 = ([int(c * den) for c in cs] for cs in (c4, c2, c0))
        num = [f - sum(n4[j] * I4[j][k] + n2[j] * I2[j][k] + n0[j] * V[j][k] for j in range(N)) for k in range(len(V[i]))]
        gamma += h * Fraction(dot(num, dlt[i])) * p2(E) / den
        rho_max = max(rho_max, Fraction(max(abs(x) for x in num), den))
        rho_float.append([x / den for x in num])  # Python int / int: correctly rounded
    return gamma, rho_max, np.array(rho_float).T


def exact_energy(V, EV, hs, U, EU, hg, Dm, q):
    """Pi_h(V U^T) of the EXACT product by the separable form of a_h and l_h (does not use A, B, C, F or K)."""
    N = len(V)
    lV, lU = [d2(c) for c in V], [d2(c) for c in U]
    sv, su = p2(2 * EV), p2(2 * EU)
    a = Fraction(0)
    for i in range(N):
        for j in range(N):
            vv2 = dot(lV[i], lV[j]) * sv / hs**3          # h_s <L V_i, L V_j>
            vv1 = dot(lV[i], V[j]) * sv / hs                # h_s <L V_i, V_j>
            vv0 = hs * dot(V[i], V[j]) * sv                 # h_s <V_i, V_j>
            uu0 = hg * dot(U[i], U[j]) * su                 # h_g <U_i, U_j>
            uu1 = dot(U[i], lU[j]) * su / hg                # h_g <U_i, L U_j>
            uu2 = dot(lU[i], lU[j]) * su / hg**3            # h_g <L U_i, L U_j>
            a += vv2 * uu0 + 2 * vv1 * uu1 + vv0 * uu2
    a *= Dm
    l = q * sum(hs * sum(V[i]) * p2(EV) * hg * sum(U[i]) * p2(EU) for i in range(N))
    return a / 2 - l, a


def exact_energy_2d(V, EV, hs, U, EU, hg, Dm, q):
    """Pi_h of the exact product by full two-dimensional assembly (independent path; small grids only)."""
    ms, mg, N = len(V[0]), len(U[0]), len(V)
    W = [[sum(V[i][k] * U[i][l] for i in range(N)) for l in range(mg)] for k in range(ms)]
    P = [[0] * (mg + 2)] + [[0] + row + [0] for row in W] + [[0] * (mg + 2)]
    sx = sxy = sy = tot = 0
    for k in range(1, ms + 1):
        for l in range(1, mg + 1):
            lx = P[k - 1][l] - 2 * P[k][l] + P[k + 1][l]
            ly = P[k][l - 1] - 2 * P[k][l] + P[k][l + 1]
            sx += lx * lx
            sxy += lx * ly
            sy += ly * ly
            tot += P[k][l]
    s2 = p2(2 * (EV + EU))
    a = Dm * hs * hg * (sx / hs**4 + 2 * Fraction(sxy) / (hs**2 * hg**2) + sy / hg**4) * s2
    l = q * hs * hg * tot * p2(EV + EU)
    return a / 2 - l, a


def decimal_gamma(Uf, Vf, Vpf, hg, hs, Dm, q, digits=80):
    """gamma by the defining formulas in Decimal arithmetic with ``digits`` digits (different arithmetic)."""
    with localcontext() as ctx:
        ctx.prec = digits
        Dc = lambda x: Decimal(float(x))  # noqa: E731 - exact conversion of a binary64 value
        hg_, hs_, D_, q_ = Dc(hg), Dc(hs), Dc(Dm), Dc(q)
        N = Uf.shape[1]
        U = [[Dc(x) for x in Uf[:, j]] for j in range(N)]
        V = [[Dc(x) for x in Vf[:, j]] for j in range(N)]
        Vp = [[Dc(x) for x in Vpf[:, j]] for j in range(N)]

        def sd(c):
            p = [Decimal(0)] + c + [Decimal(0)]
            return [p[k] - 2 * p[k + 1] + p[k + 2] for k in range(len(c))]

        def fd(c):
            p = [Decimal(0)] + c + [Decimal(0)]
            return [p[k + 1] - p[k] for k in range(len(c) + 1)]

        dt = lambda a, b: sum((x * y for x, y in zip(a, b)), Decimal(0))  # noqa: E731
        A_ = [[hg_ * dt(U[i], U[j]) for j in range(N)] for i in range(N)]
        B_ = [[dt(fd(U[i]), fd(U[j])) / hg_ for j in range(N)] for i in range(N)]
        C_ = [[dt(sd(U[i]), sd(U[j])) / hg_**3 for j in range(N)] for i in range(N)]
        F_ = [q_ * hg_ * sum(U[i], Decimal(0)) for i in range(N)]
        L2 = [sd(c) for c in V]
        L4 = [sd(c) for c in L2]
        g = Decimal(0)
        for i in range(N):
            for k in range(len(V[i])):
                rho = F_[i] - D_ * sum((A_[i][j] * L4[j][k] / hs_**4 - 2 * B_[i][j] * L2[j][k] / hs_**2 + C_[i][j] * V[j][k]
                                        for j in range(N)), Decimal(0))
                g += rho * (Vp[i][k] - V[i][k])
        return g * hs_


# ---------------------------------------------------------------------------------------------------------------
# recording the inputs of pair_energy at s = 2 inside run_coupled (no change of the solver)
# ---------------------------------------------------------------------------------------------------------------

def record_s2(r, nx, ny, N, half_steps=2):
    """run_coupled as in B5 / K-3 (fft, research mode, fixed half-steps); returns the run and the s = 2 pair_energy call."""
    calls = []
    original = cr.pair_energy

    def recorder(grid_x, grid_y, prev_X, prev_Y, X, Y, D_, q_, *, solve_grid, coefficients, received, companion_prev):
        out = original(grid_x, grid_y, prev_X, prev_Y, X, Y, D_, q_, solve_grid=solve_grid, coefficients=coefficients,
                       received=received, companion_prev=companion_prev)
        calls.append(dict(grid_x=grid_x, grid_y=grid_y, prev_X=np.array(prev_X), prev_Y=np.array(prev_Y), X=np.array(X),
                          Y=np.array(Y), D=D_, q=q_, solve_grid=solve_grid, coefficients=coefficients,
                          received=np.array(received), companion_prev=np.array(companion_prev), energy=out))
        return out

    cr.pair_energy = recorder
    try:
        run = cr.run_coupled(SIDE_A, r, D_PLATE, Q_LOAD, nx, ny, N, transform="fft", half_steps=half_steps, research_mode=True,
                             stop_on_stabilization=False)
    finally:
        cr.pair_energy = original
    if not calls:
        raise RuntimeError(f"no s = 2 energy call (run status {run.status}: {run.reason})")
    return run, calls[0]


def load_qa_energy_module():
    spec = importlib.util.spec_from_file_location("r4_controller_comparison_qa", QA_TEST)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def classify(dP, Q, tau, ten):
    e1 = dP >= -tau
    e2 = abs(dP - Q) <= tau
    if Q < ten * tau:
        status = UNRESOLVED
    elif not e1:
        status = E1_VIOLATED
    elif not e2:
        status = E2_VIOLATED
    else:
        status = E1_E2_HOLD
    return bool(e1), bool(e2), status


def audit_case(fam, lab, r, nx, ny, N, qa_mod=None, small=False):
    t0 = time.perf_counter()
    run, c = record_s2(r, nx, ny, N)
    e = c["energy"]
    st = run.steps[1]
    gx, gy, gs = c["grid_x"], c["grid_y"], c["solve_grid"]
    # s = 2 is an X-half-step: the given set is Y (grid_y), the solved set X (grid_x); w_{s-1} = prev_X Y^T, w_s = X Y^T
    assert st.s == 2 and gs is gx
    assert np.array_equal(c["prev_Y"], c["Y"]), "w_{s-1} and w_s must share the given set"
    assert np.array_equal(c["received"], c["X"]) and np.array_equal(c["companion_prev"], c["prev_X"])
    Uf, Vf, Vpf = c["Y"], c["received"], c["companion_prev"]
    hg, hs = Fraction(gy.h), Fraction(gx.h)
    Dm, q = Fraction(c["D"]), Fraction(c["q"])

    # recorder check: the logged gamma is reproduced bitwise from the recorded arrays by the implementation formula
    rho_hat, _ = coupled_residual(gs, c["coefficients"], c["D"], Vf)
    gamma_rep = gs.h * float(np.sum(rho_hat * (Vpf - Vf)))
    recorder_ok = gamma_rep == e.gamma

    (Ui,), EU = dyadic(Uf)
    (Vi, Vpi), EV = dyadic(Vf, Vpf)
    coef = exact_coefficients(Ui, EU, hg, q)
    g_hp, rho_max_ex, rho_ex = exact_gamma(coef, Vi, Vpi, EV, hs, Dm)
    cd = c["coefficients"]  # binary64 coefficients of the solver, taken exactly
    coef_dbl = ([[Fraction(float(cd.A[i, j])) for j in range(N)] for i in range(N)],
                [[Fraction(float(cd.B[i, j])) for j in range(N)] for i in range(N)],
                [[Fraction(float(cd.C[i, j])) for j in range(N)] for i in range(N)],
                [Fraction(float(cd.F[i])) for i in range(N)])
    g_dc, _, _ = exact_gamma(coef_dbl, Vi, Vpi, EV, hs, Dm)

    # self-check: exact energies of the exact products (separable path) satisfy the identity of 6.2, item 3
    P_prev, _ = exact_energy(Vpi, EV, hs, Ui, EU, hg, Dm, q)
    P_now, _ = exact_energy(Vi, EV, hs, Ui, EU, hg, Dm, q)
    dl = [[a - b for a, b in zip(Vpi[i], Vi[i])] for i in range(N)]
    _, a_d = exact_energy(dl, EV, hs, Ui, EU, hg, Dm, q)
    dP_ex, Q_ex = P_prev - P_now, a_d / 2
    identity = dP_ex - Q_ex + g_hp
    selfcheck_2d = None
    if small:
        P2p, _ = exact_energy_2d(Vpi, EV, hs, Ui, EU, hg, Dm, q)
        P2n, _ = exact_energy_2d(Vi, EV, hs, Ui, EU, hg, Dm, q)
        _, a2d = exact_energy_2d(dl, EV, hs, Ui, EU, hg, Dm, q)
        selfcheck_2d = (P2p - P2n == dP_ex) and (a2d / 2 == Q_ex) and (a2d / 2 - (P2p - P2n) == g_hp)

    # qa_energy path (independent double implementation with fsum on the ROUNDED residual; not a reference)
    g_qa = None
    if qa_mod is not None:
        Aq, Bq, Cq, Fq = qa_mod.qa_coefficients(np.asarray(Uf), gy.h)
        rho_q = Fq[None, :] - qa_mod.qa_apply_K(np.asarray(Vf), gx.h, Aq, Bq, Cq)
        g_qa = gx.h * math.fsum((rho_q * (np.asarray(Vpf) - np.asarray(Vf))).ravel().tolist())

    # first-order noise scale of the binary64 gamma (MODEL, no proven constant; a diagnostic, not a threshold):
    # u h_s sum_i sum_k S_ik |d_ik|, S = |F| (x) 1 + D (|L|^2 |V| |A|^T + 2 |L| |V| |B|^T + |V| |C|^T), |L| = (1, 2, 1)/h^2
    def abs_l(v):
        pad = np.zeros((v.shape[0] + 2, v.shape[1]))
        pad[1:-1] = v
        return (pad[:-2] + 2.0 * pad[1:-1] + pad[2:]) / gx.h**2

    aV = np.abs(Vf)
    S_abs = np.abs(cd.F)[None, :] + c["D"] * (abs_l(abs_l(aV)) @ np.abs(cd.A).T + 2.0 * abs_l(aV) @ np.abs(cd.B).T
                                              + aV @ np.abs(cd.C).T)
    noise_scale = 2.0**-53 * gx.h * float(np.sum(S_abs * np.abs(Vpf - Vf)))

    # energy test: implementation (binary64) and the variant with ONLY gamma replaced by gamma_HP
    rest = FACTOR_10 * (e.R_prev + e.R + e.R_f)
    tau_ok = abs(e.gamma) + rest == e.tau
    g_hp_f = float(g_hp)
    tau_hp = abs(g_hp_f) + rest
    m_dbl = e.Q / (FACTOR_10 * e.tau)
    m_hp = e.Q / (FACTOR_10 * tau_hp)
    e1_d, e2_d, status_d = classify(e.delta_P, e.Q, e.tau, FACTOR_10)
    e1_h, e2_h, status_h = classify(e.delta_P, e.Q, tau_hp, FACTOR_10)
    tau_hp_x = abs(g_hp) + Fraction(rest)  # exact comparisons with the computed Delta P, Q
    e1_x, e2_x, status_x = classify(Fraction(e.delta_P), Fraction(e.Q), tau_hp_x, 10)
    e1_t, e2_t, status_t = classify(dP_ex, Q_ex, tau_hp_x, 10)  # exact energies of w~ (informative)
    err = Fraction(e.gamma) - g_hp
    defect = Fraction(e.delta_P) - Fraction(e.Q) + g_hp  # rounding part of Delta P - Q (exact, from the binary64 values)
    rmodel = Fraction(e.R_prev) + Fraction(e.R) + Fraction(e.R_f)
    rel = lambda x, y: float(abs(x) / abs(y)) if y != 0 else math.inf  # noqa: E731
    row = {
        "family": fam, "r": lab, "nx": nx, "ny": ny, "N": N, "s": 2, "mandatory_flag": run.energy_s2_mandatory,
        "refined": st.refined, "eta_N": st.eta_N,
        "gamma_double": e.gamma, "gamma_hp": g_hp_f, "gamma_hp_double_coef": float(g_dc), "gamma_qa_fsum": g_qa,
        "gamma_err_abs": float(abs(err)), "gamma_err_rel": rel(err, g_hp), "gamma_err_over_tau": rel(err, Fraction(e.tau)),
        "gamma_err_over_Q": rel(err, Fraction(e.Q)), "gamma_hp_over_tau": rel(g_hp, Fraction(e.tau)),
        "gamma_double_over_tau": rel(Fraction(e.gamma), Fraction(e.tau)),
        "gamma_coef_part": float(g_dc - g_hp), "gamma_eval_part": float(Fraction(e.gamma) - g_dc),
        "gamma_qa_err_rel": rel(Fraction(g_qa) - g_hp, g_hp) if g_qa is not None else None,
        "gamma_double_minus_qa_rel": rel(Fraction(e.gamma) - Fraction(g_qa), g_hp) if g_qa is not None else None,
        "gamma_double_minus_qa_over_tau": rel(Fraction(e.gamma) - Fraction(g_qa), Fraction(e.tau)) if g_qa is not None else None,
        "gamma_double_minus_qa_over_err": rel(Fraction(e.gamma) - Fraction(g_qa), err) if g_qa is not None else None,
        "gamma_noise_scale_model": noise_scale, "gamma_err_over_noise_scale": rel(err, Fraction(noise_scale)),
        "noise_scale_over_tau": noise_scale / e.tau,
        "rho_max_exact": float(rho_max_ex), "rho_max_double": float(np.max(np.abs(rho_hat))),
        "rho_diff_max_over_rho_max": float(np.max(np.abs(rho_hat - rho_ex))) / float(rho_max_ex),
        "Q_double": e.Q, "Q_exact_product": float(Q_ex), "dP_double": e.delta_P, "dP_exact_product": float(dP_ex),
        "R_prev": e.R_prev, "R_now": e.R, "R_f": e.R_f, "tau_double": e.tau, "tau_hp_gamma": tau_hp,
        "tau_rel_change": float((tau_hp_x - Fraction(e.tau)) / Fraction(e.tau)),
        "m2_double": m_dbl, "m2_hp_gamma": m_hp, "m2_rel_change": (m_hp - m_dbl) / m_dbl,
        "m2_margin_double": m_dbl / M2_D048, "m2_margin_hp_gamma": m_hp / M2_D048,
        "e1_double": e1_d, "e2_double": e2_d, "status_double": status_d,
        "e1_hp_gamma": e1_h, "e2_hp_gamma": e2_h, "status_hp_gamma": status_h,
        "status_hp_gamma_exact_cmp": status_x, "status_exact_product_energies": status_t,
        "classification_changed": (status_d != status_h or status_d != status_x
                                   or (e1_d, e2_d) != (e1_h, e2_h) or (e1_d, e2_d) != (e1_x, e2_x)),
        "defect_over_Rsum": rel(defect, rmodel), "defect_over_tau_hp": rel(defect, tau_hp_x),
        "e2_slack_hp_over_tau": float((tau_hp_x - abs(Fraction(e.delta_P) - Fraction(e.Q))) / tau_hp_x),
        "identity_exact_zero": identity == 0, "selfcheck_2d": selfcheck_2d,
        "recorder_bitwise": recorder_ok, "tau_sum_bitwise": tau_ok, "log_status": e.status,
        "log_status_consistent": e.status == status_d and (e.e1, e.e2) == (e1_d, e2_d) and st.m_s == m_dbl,
        "exact_product_e1_e2": f"{e1_t},{e2_t}",
    }
    print(f"  ({time.perf_counter() - t0:.1f} s)", end="", flush=True)
    return row, dict(Uf=Uf, Vf=Vf, Vpf=Vpf, hg=gy.h, hs=gx.h, g_hp=g_hp, coef=c["coefficients"], gs=gs, EV=EV)


# ---------------------------------------------------------------------------------------------------------------
# comparison with the stored measurements (B5, K-3): source measurements, NOT a gamma reference
# ---------------------------------------------------------------------------------------------------------------

def compare_stored(rows):
    def key(t):
        return (t["family"], t["r"], int(t["nx"]), int(t["ny"]), int(t["N"]))

    def read(path):
        with open(path, encoding="utf-8") as f:
            return {key(t): t for t in csv.DictReader(f) if t["s"] == "2" and t["family"] in ("I", "II")}

    b5, k3 = read(B5_CSV), read(K3_CSV)
    f6 = lambda v: f"{v:.6e}"  # noqa: E731 - the format of both CSV files
    for row in rows:
        k = (row["family"], row["r"], row["nx"], row["ny"], row["N"])
        tb, tk = b5.get(k), k3.get(k)
        bm = {}
        if tb is not None:
            bm = {"Q": tb["Q"] == f6(row["Q_double"]), "tau": tb["tau"] == f6(row["tau_double"]),
                  "gamma": tb["gamma"] == f6(row["gamma_double"]), "m": tb["m"] == f6(row["m2_double"]),
                  "e1": tb["e1"] == str(row["e1_double"]), "e2": tb["e2"] == str(row["e2_double"]),
                  "status": tb["energy_status"] == row["status_double"], "eta_N": tb["eta_N"] == f6(row["eta_N"])}
        km = {}
        if tk is not None:
            km = {"m": tk["m"] == f6(row["m2_double"]), "e1": tk["e1"] == str(row["e1_double"]),
                  "e2": tk["e2"] == str(row["e2_double"]), "status": tk["status"] == row["status_double"],
                  "mandatory": tk["mandatory_flag"] == str(row["mandatory_flag"]), "eta_N": tk["eta_N"] == f6(row["eta_N"])}
        row["b5_present"], row["k3_present"] = tb is not None, tk is not None
        row["b5_mismatch"] = ";".join(n for n, ok in bm.items() if not ok)
        row["k3_mismatch"] = ";".join(n for n, ok in km.items() if not ok)
        row["b5_gamma_stored"] = tb["gamma"] if tb is not None else None
        row["b5_m_stored"] = tb["m"] if tb is not None else None
        row["k3_m_stored"] = tk["m"] if tk is not None else None
        row["k3_independent_energy_agrees"] = tk["independent_energy_agrees"] if tk is not None else None
    return rows


# ---------------------------------------------------------------------------------------------------------------
# self-checks of the QA reference
# ---------------------------------------------------------------------------------------------------------------

SMALL = [("I", "1/5", 0.2, 10, 10, 2), ("I", "1", 1.0, 8, 8, 3), ("I", "5", 5.0, 12, 12, 2), ("I", "3/2", 1.5, 9, 13, 3),
         ("I", "1/3", 1 / 3, 16, 11, 2), ("I", "7/5", 1.4, 14, 14, 3)]


def selfcheck(qa_mod):
    out = {"small_cases": []}
    ok = True
    for fam, lab, r, nx, ny, N in SMALL:
        row, data = audit_case(fam, lab, r, nx, ny, N, qa_mod, small=True)
        gd = decimal_gamma(data["Uf"], data["Vf"], data["Vpf"], data["hg"], data["hs"], 1.0, 1.0)
        g = data["g_hp"]
        dec_rel = float(abs(Fraction(gd) - g) / abs(g)) if g != 0 else float(abs(Fraction(gd)))
        good = bool(row["identity_exact_zero"] and row["selfcheck_2d"] and dec_rel <= 1e-60 and row["recorder_bitwise"])
        ok &= good
        out["small_cases"].append({"r": lab, "nx": nx, "ny": ny, "N": N, "identity_exact_zero": row["identity_exact_zero"],
                                   "two_d_exact_equal": row["selfcheck_2d"], "decimal80_rel_diff": dec_rel,
                                   "gamma_hp": row["gamma_hp"], "gamma_double": row["gamma_double"],
                                   "gamma_err_rel": row["gamma_err_rel"], "ok": good})
    # perturbation: gamma far above its evaluation noise -> the binary64 formula becomes accurate; the identity stays exact
    out["perturbation"] = []
    for lab, r, n, N in (("1", 1.0, 160, 2), ("5", 5.0, 160, 3), ("1", 1.0, 1280, 2)):
        _, c = record_s2(r, n, n, N)
        gs, gy = c["solve_grid"], c["grid_y"]
        Vf = c["received"]
        k = np.arange(1, Vf.shape[0] + 1)[:, None] * np.arange(1, N + 1)[None, :]
        shape = np.sin(math.pi * k / (Vf.shape[0] + 1))
        (Ui,), EU = dyadic(c["Y"])
        hg, hs = Fraction(gy.h), Fraction(gs.h)
        coef = exact_coefficients(Ui, EU, hg, Fraction(1))
        for eps in (0.0, 1e-8, 1e-6, 1e-4, 1e-2):
            Vp = Vf + eps * float(np.max(np.abs(Vf))) * shape  # binary64, then taken exactly
            rho_hat, _ = coupled_residual(gs, c["coefficients"], c["D"], Vp)
            g_dbl = gs.h * float(np.sum(rho_hat * (c["companion_prev"] - Vp)))
            (Vi, Vpi), EV = dyadic(Vp, c["companion_prev"])
            g_ex, _, _ = exact_gamma(coef, Vi, Vpi, EV, hs, Fraction(1))
            Pp, _ = exact_energy(Vpi, EV, hs, Ui, EU, hg, Fraction(1), Fraction(1))
            Pn, _ = exact_energy(Vi, EV, hs, Ui, EU, hg, Fraction(1), Fraction(1))
            dl = [[a - b for a, b in zip(Vpi[i], Vi[i])] for i in range(N)]
            _, ad = exact_energy(dl, EV, hs, Ui, EU, hg, Fraction(1), Fraction(1))
            ident = (Pp - Pn) - ad / 2 + g_ex == 0
            ok &= bool(ident)
            out["perturbation"].append({"r": lab, "n": n, "N": N, "eps": eps, "gamma_exact": float(g_ex),
                                        "gamma_double": g_dbl, "abs_diff": float(abs(Fraction(g_dbl) - g_ex)),
                                        "rel_diff": float(abs(Fraction(g_dbl) - g_ex) / abs(g_ex)),
                                        "identity_exact_zero": ident})
    out["all_ok"] = bool(ok)
    return out


# ---------------------------------------------------------------------------------------------------------------
# campaign
# ---------------------------------------------------------------------------------------------------------------

def fmt(v):
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, (float, np.floating)):
        return repr(float(v))
    return "" if v is None else v


def write_csv(path, rows):
    keys = []
    for row in rows:
        keys += [k for k in row if k not in keys]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
        w.writeheader()
        for row in rows:
            w.writerow({k: fmt(v) for k, v in row.items()})


def summary(rows):
    def mx(k):
        return max(r[k] for r in rows)

    def mn(k):
        return min(r[k] for r in rows)

    by_n = {}
    for N in (2, 3):
        sub = [r for r in rows if r["N"] == N]
        by_n[N] = {"cases": len(sub), "min_m2_double": min(r["m2_double"] for r in sub),
                   "min_m2_hp_gamma": min(r["m2_hp_gamma"] for r in sub)}
    return {
        "cases": len(rows),
        "identity_exact_zero": sum(r["identity_exact_zero"] for r in rows),
        "recorder_bitwise": sum(r["recorder_bitwise"] for r in rows),
        "tau_sum_bitwise": sum(r["tau_sum_bitwise"] for r in rows),
        "log_status_consistent": sum(r["log_status_consistent"] for r in rows),
        "classification_changed": sum(r["classification_changed"] for r in rows),
        "status_double": sorted({r["status_double"] for r in rows}),
        "status_hp_gamma": sorted({r["status_hp_gamma"] for r in rows}),
        "status_hp_gamma_exact_cmp": sorted({r["status_hp_gamma_exact_cmp"] for r in rows}),
        "mandatory_flag": sum(bool(r["mandatory_flag"]) for r in rows),
        "gamma_err_rel": [mn("gamma_err_rel"), mx("gamma_err_rel")],
        "max_gamma_err_over_tau": mx("gamma_err_over_tau"), "max_gamma_err_over_Q": mx("gamma_err_over_Q"),
        "max_gamma_hp_over_tau": mx("gamma_hp_over_tau"), "max_gamma_double_over_tau": mx("gamma_double_over_tau"),
        "max_abs_tau_rel_change": max(abs(r["tau_rel_change"]) for r in rows),
        "max_abs_m2_rel_change": max(abs(r["m2_rel_change"]) for r in rows),
        "min_m2_double": mn("m2_double"), "min_m2_hp_gamma": mn("m2_hp_gamma"), "by_N": by_n,
        "max_defect_over_Rsum": mx("defect_over_Rsum"), "min_e2_slack_hp_over_tau": mn("e2_slack_hp_over_tau"),
        "max_rho_diff_over_rho_max": mx("rho_diff_max_over_rho_max"),
        "gamma_qa_err_rel": [mn("gamma_qa_err_rel"), mx("gamma_qa_err_rel")],
        "max_gamma_double_minus_qa_over_tau": mx("gamma_double_minus_qa_over_tau"),
        "gamma_double_minus_qa_over_err": [mn("gamma_double_minus_qa_over_err"), mx("gamma_double_minus_qa_over_err")],
        "gamma_err_rel_gt_1": sum(r["gamma_err_rel"] > 1 for r in rows),
        "gamma_sign_differs": sum((r["gamma_double"] > 0) != (r["gamma_hp"] > 0) for r in rows),
        "max_gamma_err_over_noise_scale": mx("gamma_err_over_noise_scale"), "max_noise_scale_over_tau": mx("noise_scale_over_tau"),
        "b5_mismatch_cases": sum(bool(r["b5_mismatch"]) for r in rows),
        "k3_mismatch_cases": sum(bool(r["k3_mismatch"]) for r in rows),
        "b5_mismatch_fields": sorted({f for r in rows for f in r["b5_mismatch"].split(";") if f}),
        "k3_mismatch_fields": sorted({f for r in rows for f in r["k3_mismatch"].split(";") if f}),
    }


def _json_default(o):
    if isinstance(o, (np.bool_, np.integer, np.floating)):
        return o.item()
    raise TypeError(f"not JSON serializable: {type(o).__name__}")


def git_state():
    def g(*args):
        return subprocess.run(["git", *args], check=False, cwd=ROOT, capture_output=True, text=True).stdout.strip()

    return {"commit": g("rev-parse", "HEAD"), "branch": g("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty_paths": g("status", "--porcelain").splitlines()}


def run_campaign(profile, out):
    t0 = time.perf_counter()
    qa_mod = load_qa_energy_module()
    cases = mandatory_cases() if profile == "full" else [c for c in mandatory_cases() if c[3] == c[4] == 160][:4]
    rows = []
    for fam, lab, r, nx, ny, N in cases:
        row, _ = audit_case(fam, lab, r, nx, ny, N, qa_mod)
        rows.append(row)
        print(f"{fam} r={lab} {nx}x{ny} N={N}: gamma_double={row['gamma_double']:.3e} gamma_hp={row['gamma_hp']:.3e} "
              f"err/tau={row['gamma_err_over_tau']:.2e} m2={row['m2_double']:.4g}->{row['m2_hp_gamma']:.4g} "
              f"{row['status_double']}->{row['status_hp_gamma']} id={row['identity_exact_zero']}",
              flush=True)
    compare_stored(rows)
    checks = selfcheck(qa_mod)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "s2.csv", rows)
    meta = {
        "task": "QA R4 D-053.3 gamma audit (G3.3/G4, link G3.4); not Q4",
        "profile": profile, "git": git_state(), "python": sys.version.split()[0], "numpy": np.__version__,
        "scipy": scipy.__version__, "platform": platform.platform(), "machine": platform.machine(),
        "processor": platform.processor(), "transform": "fft", "half_steps": 2,
        "run_coupled": "research_mode=True, stop_on_stabilization=False, a=1, b=r, D=1, q=1 (B5 / K-3)",
        "reference": "exact rational (fractions, Python int); binary64 arrays and grid steps carried over exactly",
        "criterion_m2": M2_D048, "summary": summary(rows), "selfcheck": checks,
        "seconds": time.perf_counter() - t0,
    }
    with open(out / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1, ensure_ascii=False, default=_json_default)
        f.write("\n")
    print(json.dumps(meta["summary"], indent=1, default=_json_default))
    print("selfcheck all_ok:", checks["all_ok"])
    return meta


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("run")
    pr.add_argument("--profile", choices=("full", "ci"), default="full")
    pr.add_argument("--out", type=Path, default=None)
    sub.add_parser("selfcheck")
    a = p.parse_args(argv)
    if a.cmd == "selfcheck":
        res = selfcheck(load_qa_energy_module())
        print(json.dumps(res, indent=1, default=_json_default))
        return 0 if res["all_ok"] else 1
    out = a.out if a.out is not None else (OUT if a.profile == "full" else ROOT / "scratch" / "r4_d053_ci")
    meta = run_campaign(a.profile, out)
    s = meta["summary"]
    ok = (meta["selfcheck"]["all_ok"] and s["identity_exact_zero"] == s["cases"] and s["recorder_bitwise"] == s["cases"]
          and s["tau_sum_bitwise"] == s["cases"])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
