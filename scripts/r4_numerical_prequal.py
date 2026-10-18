"""R4 numerical prequalification N = 2-3: checks B1-B9 on the merged solver (QA; basis for the owner's D-048).

Criteria, samples and a-priori models were fixed BEFORE these computations in docs/R4_NUMERICAL_PREQUALIFICATION.plan.md
(commit 7748877). Nothing here chooses a threshold from the results; observed maxima are reported as observations.
The finite discrete problem is never identified with the untruncated continuous method (Q-022, Q-025 stay open).

Reproduction (repository root):
    uv run --with mpmath python scripts/r4_numerical_prequal.py            # everything (high precision needs mpmath)
    uv run python scripts/r4_numerical_prequal.py --no-hp                   # B5-B9 only (no mpmath)
    uv run python scripts/r4_numerical_prequal.py --tables-only             # tables.md from the CSV files
Outputs: docs/results/r4_numerical_prequal/ (--out).
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

import kkvim.coupled_run as cr
from kkvim.coupled import coupled_coefficients, odd_sine_set, sine_eigenvalues, sine_matrix, sine_transform, solve_coupled_halfstep
from kkvim.coupled_rank import orthogonalize_received, rank_diagnostics
from kkvim.grid import Grid1D
from kkvim_reference.levy import levy_series_w_grid
from kkvim_reference.r4_rank_diagnostics import cancellation_set, dense_transform, exact_rank

U = 2.0**-53
R_TEN = {"1/5": 0.2, "1/3": 1 / 3, "1/2": 0.5, "2/3": 2 / 3, "1": 1.0, "7/5": 1.4, "3/2": 1.5, "2": 2.0, "3": 3.0, "5": 5.0}
R_THREE = {"1/5": 0.2, "1": 1.0, "5": 5.0}
FAMILY_II_TOP = {"1/5": (1600, 320), "1/3": (960, 320), "1/2": (640, 320), "2": (320, 640), "3": (320, 960), "5": (320, 1600)}
TOL_INF, TOL_E, S_MAX_R2, S_MIN_R2, ETA_MAX_R2 = 1e-10, 1e-8, 100, 3, 1e-12  # R2 candidates (NUMERICAL_SPEC 11.2, 3.3)
ZETA_Q, ZETA_A = 0.039420, 0.157681  # zeta_A5^{R2}(80), zeta_A5^{R2}(20) (R3_NUMERICAL_SPEC 6.3; D-039)
RESERVE_D039 = 4.0  # D-039: engineering reserve "about one doubling" for a mandatory area
K_RESEARCH = cr.K_RESEARCH


def write_csv(path, rows):
    keys = []
    for row in rows:
        keys += [k for k in row if k not in keys]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
        w.writeheader()
        for row in rows:
            w.writerow({k: (f"{v:.6e}" if isinstance(v, float) else v) for k, v in row.items()})


def read_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------------------------------------------
# Harness: Builder's own functions in the order of run_coupled (research mode), keeping the arrays of each half-step
# ---------------------------------------------------------------------------------------------------------------


def harness(r, N, nx, ny, s_max, transform="fft"):
    gx, gy = Grid1D(1.0, nx), Grid1D(r, ny)
    X, Y = odd_sine_set(gx, N), None
    out = []
    for s in range(1, s_max + 1):
        direction = "Y" if s % 2 else "X"
        given_grid, solve_grid = (gx, gy) if direction == "Y" else (gy, gx)
        given = X if direction == "Y" else Y
        step = solve_coupled_halfstep(given_grid, given, solve_grid, 1.0, 1.0, direction=direction, transform=transform)
        new = (X, step.solved) if direction == "Y" else (step.solved, Y)
        out.append({"s": s, "given": given, "given_grid": given_grid, "solve_grid": solve_grid, "step": step, "new": new,
                    "gx": gx, "gy": gy})
        if s < s_max:
            o = orthogonalize_received(solve_grid, step.solved, given)
            X, Y = (o.companion, o.received) if direction == "Y" else (o.received, o.companion)
        else:
            X, Y = new
    return out


def harness_matches_run_coupled(r, N, nx, ny, s_max):
    run = cr.run_coupled(1.0, r, 1.0, 1.0, nx, ny, N, transform="fft", half_steps=s_max, research_mode=True)
    rec = harness(r, N, nx, ny, s_max)
    return bool(np.array_equal(run.x, rec[-1]["new"][0]) and np.array_equal(run.y, rec[-1]["new"][1]))


# ---------------------------------------------------------------------------------------------------------------
# B1, B2: exact discrete problem in high precision (spectral form of Pi_h; r4_spectral_control, items 2-3)
# ---------------------------------------------------------------------------------------------------------------


class ExactDiscrete:
    """The same discrete problem in mpmath: Psi, mu, lambda_kl = (mu^x_k + mu^y_l)^2, W*_kl = q c^x_k c^y_l/(D lambda_kl)."""

    def __init__(self, mp, r, nx, ny):
        self.mp = mp
        self.nx, self.ny = nx, ny
        a, b = mp.mpf(1), mp.mpf(r)

        def psi(n):
            tab = [mp.sin(mp.pi * j / n) for j in range(2 * n)]
            return [[mp.sqrt(mp.mpf(2) / n) * tab[(i * k) % (2 * n)] for k in range(1, n)] for i in range(1, n)]

        def mu(n, length):
            return [4 * (n / length) ** 2 * mp.sin(k * mp.pi / (2 * n)) ** 2 for k in range(1, n)]

        self.Px, self.Py = psi(nx), psi(ny)
        mx, my = mu(nx, a), mu(ny, b)
        cx = [mp.fsum(row[k] for row in self.Px) for k in range(nx - 1)]
        cy = [mp.fsum(row[k] for row in self.Py) for k in range(ny - 1)]
        self.lam = [[(mx[k] + my[l]) ** 2 for l in range(ny - 1)] for k in range(nx - 1)]
        self.ws = [[cx[k] * cy[l] / self.lam[k][l] for l in range(ny - 1)] for k in range(nx - 1)]

    def spectral(self, P, G):
        """Psi G for a double array G (n-1, N) -> list of rows (mp)."""
        mp = self.mp
        Gm = [[mp.mpf(float(v)) for v in row] for row in np.asarray(G)]
        n1, N = len(P), len(Gm[0])
        return [[mp.fsum(P[i][t] * Gm[t][j] for t in range(n1)) for j in range(N)] for i in range(n1)]

    def halfstep(self, Ghat, direction):
        """Exact half-step from the spectral given set (rows of mp): orthonormalise (MGS), normal equations per mode."""
        mp = self.mp
        N = len(Ghat[0])
        cols = [[row[j] for row in Ghat] for j in range(N)]
        for i in range(N):
            for j in range(i):
                d = mp.fsum(x * y for x, y in zip(cols[i], cols[j]))
                cols[i] = [x - d * y for x, y in zip(cols[i], cols[j])]
            nrm = mp.sqrt(mp.fsum(x * x for x in cols[i]))
            cols[i] = [x / nrm for x in cols[i]]
        Q = [[cols[j][t] for j in range(N)] for t in range(len(cols[0]))]
        if direction == "Y":
            K, M = self.nx - 1, self.ny - 1
            w, tg = (lambda m, t: self.lam[t][m]), (lambda m, t: self.ws[t][m])
        else:
            K, M = self.ny - 1, self.nx - 1
            w, tg = (lambda m, t: self.lam[m][t]), (lambda m, t: self.ws[m][t])
        V = []
        for m in range(M):
            A = mp.matrix(N, N)
            rhs = mp.matrix(N, 1)
            for t in range(K):
                wt = w(m, t)
                for i in range(N):
                    rhs[i] += wt * tg(m, t) * Q[t][i]
                    for j in range(N):
                        A[i, j] += wt * Q[t][i] * Q[t][j]
            v = mp.lu_solve(A, rhs)
            V.append([v[i] for i in range(N)])
        return Q, V  # field hat = Q V^T (Y-direction: x-modes x y-modes; X-direction: y-modes x x-modes)


def field_diff_norm(mp, Xhat_b, Yhat_b, What_exact):
    """|| Xhat_b Yhat_b^T - What_exact ||_2 / ||What_exact||_2 (spectral; Psi orthogonal => same as nodal)."""
    n1, n2, N = len(Xhat_b), len(Yhat_b), len(Xhat_b[0])
    D = np.empty((n1, n2))
    E = np.empty((n1, n2))
    for i in range(n1):
        for j in range(n2):
            ex = What_exact[i][j]
            D[i, j] = float(mp.fsum(Xhat_b[i][k] * Yhat_b[j][k] for k in range(N)) - ex)
            E[i, j] = float(ex)
    return float(np.linalg.norm(D, 2) / np.linalg.norm(E, 2))


def b1_b2_high_precision(sample=((2, "1/5"), (2, "1"), (2, "5"), (3, "1/5"), (3, "1"), (3, "5")), grids=(16, 40), s_max=20):
    import mpmath as mp
    mp.mp.dps = 40
    rows_b1, rows_b2 = [], []
    for n in grids:
        for N, lab in sample:
            r = R_THREE[lab]
            ex = ExactDiscrete(mp, r, n, n)
            rec = harness(r, N, n, n, s_max)
            # exact sequence from the odd-sine start
            X0 = odd_sine_set(Grid1D(1.0, n), N)
            Ghat = ex.spectral(ex.Px, X0)
            exact_fields = []
            for s in range(1, s_max + 1):
                direction = "Y" if s % 2 else "X"
                Q, V = ex.halfstep(Ghat, direction)
                if direction == "Y":
                    What = [[mp.fsum(Q[i][k] * V[j][k] for k in range(N)) for j in range(n - 1)] for i in range(n - 1)]
                else:
                    What = [[mp.fsum(V[i][k] * Q[j][k] for k in range(N)) for j in range(n - 1)] for i in range(n - 1)]
                exact_fields.append(What)
                Ghat = V
            acc = 0.0
            for h in rec:
                s, direction, step = h["s"], h["step"].direction, h["step"]
                Xb, Yb = h["new"]
                Xhat_b, Yhat_b = ex.spectral(ex.Px, Xb), ex.spectral(ex.Py, Yb)
                d_f = field_diff_norm(mp, Xhat_b, Yhat_b, exact_fields[s - 1])
                # one exact half-step from Builder's stored given set
                P_given = ex.Px if direction == "Y" else ex.Py
                Q, V = ex.halfstep(ex.spectral(P_given, h["given"]), direction)
                if direction == "Y":
                    What = [[mp.fsum(Q[i][k] * V[j][k] for k in range(N)) for j in range(n - 1)] for i in range(n - 1)]
                else:
                    What = [[mp.fsum(V[i][k] * Q[j][k] for k in range(N)) for j in range(n - 1)] for i in range(n - 1)]
                d_step = field_diff_norm(mp, Xhat_b, Yhat_b, What)
                acc += d_step
                rows_b1.append({"N": N, "r": lab, "n": n, "s": s, "delta_f": d_f, "delta_step": d_step, "sum_steps": acc,
                                "linear_model_holds": d_f <= acc * (1 + 1e-12)})
                rows_b2 += b2_modes(mp, step, N, lab, n, s)
            print("B1/B2", N, lab, n, flush=True)
    # B2 on n = 640, r = 1, s = 1..4
    for N in (2, 3):
        for h in harness(1.0, N, 640, 640, 4):
            rows_b2 += b2_modes(mp, h["step"], N, "1", 640, h["s"])
        print("B2 640", N, flush=True)
    return rows_b1, rows_b2


def b2_modes(mp, step, N, lab, n, s):
    """Strict bound |v_k - v^_k| <= |r_k|/(D lambda_min(M_k)) on the Builder data of every mode (40 digits)."""
    co = step.coefficients
    grid = step.solve_grid
    mu = sine_eigenvalues(grid)
    load = sine_transform(grid, np.ones(grid.interior_size), step.transform)
    Am, Bm, Cm = (mp.matrix([[mp.mpf(float(x)) for x in row] for row in M]) for M in (co.A, co.B, co.C))
    Fm = [mp.mpf(float(x)) for x in co.F]
    viol, worst_ratio, worst_rel, min_ratio = 0, 0.0, 0.0, math.inf
    for k in range(mu.size):
        mk = mp.mpf(float(mu[k]))
        M = mk**2 * Am + 2 * mk * Bm + Cm
        b = mp.matrix([f * mp.mpf(float(load[k])) for f in Fm])
        vh = mp.matrix([mp.mpf(float(x)) for x in step.modal[k]])
        rk = b - M * vh  # D = 1
        lam_min = min(mp.eigsy(M)[0])
        bound = mp.norm(rk) / lam_min
        v = mp.lu_solve(M, b)
        actual = mp.norm(v - vh)
        if actual > bound:
            viol += 1
        if actual > 0:
            min_ratio = min(min_ratio, float(bound / actual))
        nv = mp.norm(v)
        if nv > 0:
            worst_rel = max(worst_rel, float(actual / nv))
            worst_ratio = max(worst_ratio, float(bound / nv))
    return [{"N": N, "r": lab, "n": n, "s": s, "modes": int(mu.size), "violations": viol, "min_bound_over_actual": min_ratio,
             "max_actual_rel": worst_rel, "max_bound_rel": worst_ratio}]


# ---------------------------------------------------------------------------------------------------------------
# B3: DST of the actual kernel
# ---------------------------------------------------------------------------------------------------------------


def b3_dst():
    import mpmath as mp
    mp.mp.dps = 30
    rows_fast, rows_dense = [], []
    n = 1280
    grid = Grid1D(1.0, n)
    tab = [mp.sin(mp.pi * j / n) for j in range(2 * n)]
    sc = mp.sqrt(mp.mpf(2) / n)
    rng = np.random.default_rng(3)
    rec = harness(1.0, 3, n, n, 2)
    inputs = {"e_1": np.eye(n - 1)[:, 0], "e_mid": np.eye(n - 1)[:, n // 2 - 1], "e_last": np.eye(n - 1)[:, -1],
              "harm_1": np.sin(np.pi * np.arange(1, n) / n), "harm_mid": np.sin(np.pi * np.arange(1, n) * (n // 2) / n),
              "harm_last": np.sin(np.pi * np.arange(1, n) * (n - 1) / n), "odd_sine_5": odd_sine_set(grid, 3)[:, 2],
              "modal_real_s2": rec[1]["step"].modal[:, 0], "random": rng.standard_normal(n - 1)}
    for name, x in inputs.items():
        xm = [mp.mpf(float(v)) for v in x]
        nz = [i for i in range(n - 1) if x[i] != 0.0]
        exact = [sc * mp.fsum(tab[((k + 1) * (i + 1)) % (2 * n)] * xm[i] for i in nz) for k in range(n - 1)]
        for method in ("fft", "dense"):
            y = sine_transform(grid, x, method)
            err = math.sqrt(float(mp.fsum((mp.mpf(float(y[k])) - exact[k]) ** 2 for k in range(n - 1))))
            nx_ = float(np.linalg.norm(x))
            rows_fast.append({"n": n, "input": name, "method": method, "err2_over_u_log2n_x2": err / (U * math.log2(n) * nx_),
                              "err2_over_u_x2": err / (U * nx_)})
        print("B3", name, flush=True)
    # dense Psi of the Builder (argument without integer reduction): element model
    for n in (40, 160, 640, 1280):
        g = Grid1D(1.0, n)
        P = sine_matrix(g)
        tab = [mp.sin(mp.pi * j / n) for j in range(2 * n)]
        sc = mp.sqrt(mp.mpf(2) / n)
        scf = math.sqrt(2.0 / n)
        viol, worst = 0, 0.0
        worst_abs = 0.0
        for i in range(1, n):
            for k in range(1, n):
                ex = sc * tab[(i * k) % (2 * n)]
                err = abs(float(mp.mpf(float(P[i - 1, k - 1])) - ex))
                theta = i * k * math.pi / n
                bound = scf * (2.35 * U * theta + 2 * U) + 1.5 * U * abs(float(ex))
                if err > bound:
                    viol += 1
                worst = max(worst, err / bound)
                worst_abs = max(worst_abs, err / scf)
        rows_dense.append({"n": n, "elements": (n - 1) ** 2, "model_violations": viol, "max_err_over_model": worst,
                           "max_err_over_sqrt2n": worst_abs, "max_err_over_7u_sqrt2n_QA_reduced_bound": worst_abs / (7 * U)})
        print("B3 dense", n, flush=True)
    return rows_fast, rows_dense


# ---------------------------------------------------------------------------------------------------------------
# B4: rank diagnostics of the actual kernel on constructed factors
# ---------------------------------------------------------------------------------------------------------------


def b4_rank():
    rows = []
    for N in (2, 3):
        for K in (1e0, 1e2, 1e4, 1e6, 1e8, 1e10, 1e12):
            worst, guard_ok, kap = 0.0, True, 0.0
            for seed in range(20):
                Uf, Vf = cancellation_set(12, 10, K, seed, N=N)
                assert exact_rank(Uf, Vf) == N - 1
                gu, gv = Grid1D(1.0, Uf.shape[0] + 1), Grid1D(1.0, Vf.shape[0] + 1)
                d = rank_diagnostics(gu, Uf, gv, Vf)
                g = cr._guards(d)
                ratio = d.rho_N / (U * d.kappa_rep)
                worst, kap = max(worst, ratio), max(kap, d.kappa_rep)
                if g.applicable and g.rank_ok:  # criterion (i): an exact rank-(N-1) field must not pass the rank guard
                    guard_ok = False
            rows.append({"construction": "cancellation (exact integer factors)", "N": N, "K_or_cond": K, "kappa_rep_max": kap,
                         "rho_over_u_kappa_max": worst, "guard_K_u_kappa_lt_1": K_RESEARCH * U * kap < 1,
                         "criterion_i_rank_guard_fails": guard_ok})
    rng_n = 40
    g = Grid1D(1.0, rng_n)
    base = odd_sine_set(g, 3)
    for cond in (4e2, 2.8e5, 9e8):
        worst, guard_ok, kap = 0.0, True, 0.0
        for seed in range(20):
            T = dense_transform(3, cond, seed)
            Vr = np.column_stack([base[:, 0], base[:, 1], np.zeros(rng_n - 1)])  # field of exact rank 2 in 3 terms
            Uf = base @ T
            Vf = Vr @ np.linalg.inv(T).T
            d = rank_diagnostics(g, Uf, g, Vf)
            gg = cr._guards(d)
            worst, kap = max(worst, d.rho_N / (U * d.kappa_rep)), max(kap, d.kappa_rep)
            if gg.applicable and gg.rank_ok:
                guard_ok = False
        rows.append({"construction": "rank 2 in 3 terms, dense T (computed factors, Reviewer type)", "N": 3, "K_or_cond": cond,
                     "kappa_rep_max": kap, "rho_over_u_kappa_max": worst, "guard_K_u_kappa_lt_1": K_RESEARCH * U * kap < 1,
                     "criterion_i_rank_guard_fails": guard_ok})
    return rows


# ---------------------------------------------------------------------------------------------------------------
# B5-B9 on run_coupled
# ---------------------------------------------------------------------------------------------------------------


def b5_energy():
    rows = []
    configs = [("I", lab, r, n, n) for lab, r in R_TEN.items() for n in (160, 640, 1280)]
    configs += [("II", lab, R_TEN[lab], nx, ny) for lab, (nx, ny) in FAMILY_II_TOP.items()]
    for fam, lab, r, nx, ny in configs:
        for N in (2, 3):
            run = cr.run_coupled(1.0, r, 1.0, 1.0, nx, ny, N, transform="fft", half_steps=4, research_mode=True)
            for lg in run.steps:
                e = lg.energy
                row = {"family": fam, "r": lab, "nx": nx, "ny": ny, "N": N, "s": lg.s, "status_run": run.status,
                       "eta_N": lg.eta_N, "rho_N": lg.rank.rho_N, "kappa_rep": lg.rank.kappa_rep,
                       "set_sigma_min": lg.rank.set_sigma_min, "qr_field_error": lg.qr_field_error if lg.qr_done else math.nan}
                if lg.s >= 2:
                    row.update({"Q": e.Q, "tau": e.tau, "gamma": e.gamma, "m": e.Q / (10 * e.tau), "e1": e.e1, "e2": e.e2,
                                "energy_status": e.status, "gamma_over_tau": abs(e.gamma) / e.tau})
                rows.append(row)
        print("B5", fam, lab, nx, ny, flush=True)
    return rows


def b6_stopping():
    rows, summary = [], []
    for lab, r in R_THREE.items():
        for n in (40, 160, 640):
            for N in (2, 3):
                run = cr.run_coupled(1.0, r, 1.0, 1.0, n, n, N, transform="fft", half_steps=40, research_mode=True)
                floor = 10 * U * (n / math.pi) ** 2 * math.sqrt(N) / (1 - N * U * math.sqrt(N))  # kappa_W <= sqrt N/(1 - gamma_N sqrt N)
                s_stop = None
                prev_ok = False
                for lg in run.steps:
                    ok = (lg.eps_inf is not None and lg.eps_inf <= TOL_INF and lg.eps_E <= TOL_E)
                    rows.append({"r": lab, "n": n, "N": N, "s": lg.s, "eps_inf": lg.eps_inf if lg.eps_inf is not None else math.nan,
                                 "eps_E": lg.eps_E if lg.eps_E is not None else math.nan, "eta_N": lg.eta_N, "P": lg.energy.P,
                                 "rho_N": lg.rank.rho_N, "kappa_rep": lg.rank.kappa_rep, "set_sigma_min": lg.rank.set_sigma_min})
                    if s_stop is None and ok and prev_ok and lg.s >= S_MIN_R2:
                        s_stop = lg.s
                    prev_ok = ok
                # drift after the stop: centre value and energy against s = 40
                W0c = levy_series_w_grid(np.array([0.5]), np.array([0.5 * r]), a=1.0, b=r, q=1.0, D=1.0)[0, 0]
                c = (n // 2 - 1, n // 2 - 1)
                w40 = (run.x @ run.y.T)[c]
                drift_c = drift_P = math.nan
                if s_stop is not None:
                    rs = cr.run_coupled(1.0, r, 1.0, 1.0, n, n, N, transform="fft", half_steps=s_stop, research_mode=True)
                    ws = (rs.x @ rs.y.T)[c]
                    drift_c = abs(w40 - ws) / abs(w40 - W0c)
                    drift_P = abs(run.steps[-1].energy.P - run.steps[s_stop - 1].energy.P) / abs(run.steps[-1].energy.P)
                eps_E_tail = [lg.eps_E for lg in run.steps[20:]]
                summary.append({"r": lab, "n": n, "N": N, "eps_floor_N": floor, "tolE_over_10floor": TOL_E / (10 * floor),
                                "s_stop_R2_rule": s_stop if s_stop is not None else "none<=40",
                                "drift_delta_c_after_stop": drift_c, "drift_P_after_stop": drift_P,
                                "eps_E_min_s21_40": min(eps_E_tail), "eps_inf_min_s21_40": min(lg.eps_inf for lg in run.steps[20:]),
                                "max_eta_N": max(lg.eta_N for lg in run.steps)})
            print("B6", lab, flush=True)
    return rows, summary


def zeta_of(r, N, nx, ny, s_max=3):
    rec = harness(r, N, nx, ny, s_max)
    z = 0.0
    for h in rec:
        co = coupled_coefficients(h["given_grid"], h["given"], 1.0)
        lam = np.linalg.eigvals(np.linalg.solve(co.A, co.C)).real
        z = max(z, float(np.max(lam)) ** 0.25 * h["given_grid"].h)
    return z


def b8_zeta(group_b_dir):
    rows = []
    for lab, r in R_THREE.items():
        for N in (2, 3):
            for n in (20, 40, 80, 160, 320, 640, 1280):
                z = zeta_of(r, N, n, n)
                rows.append({"family": "I", "r": lab, "N": N, "nx": n, "ny": n, "zeta_N": z,
                             "class": "Q" if z <= ZETA_Q else ("A" if z <= ZETA_A else "G")})
            if lab != "1":
                base = (100, 20) if r < 1 else (20, 100)
                for k in range(5):
                    nx, ny = base[0] * 2**k, base[1] * 2**k
                    z = zeta_of(r, N, nx, ny)
                    rows.append({"family": "II", "r": lab, "N": N, "nx": nx, "ny": ny, "zeta_N": z,
                                 "class": "Q" if z <= ZETA_Q else ("A" if z <= ZETA_A else "G")})
    # observed order p of group B on the same grids (richardson.csv: pair (nx, 2 nx); centre p and max p over s)
    rich = read_csv(group_b_dir / "richardson.csv")
    for row in rows:
        ps = [float(t["p_max"]) for t in rich if t["family"] == row["family"] and int(t["N"]) == row["N"]
              and t["r_label"] == row["r"] and int(t["nx"]) == row["nx"] and int(t["ny"]) == row["ny"]]
        row["p_group_B_min"] = min(ps) if ps else math.nan
        row["p_group_B_max"] = max(ps) if ps else math.nan
        row["p_in_window"] = (bool(ps) and 1.9 <= min(ps) and max(ps) <= 2.1) if ps else "no pair"
    return rows


# ---------------------------------------------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------------------------------------------


def fmt(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return str(v)
    if math.isnan(x):
        return "—"
    if x == int(x) and abs(x) < 1e6 and "." not in str(v) and "e" not in str(v):
        return str(int(x))
    return f"{x:.3e}"


def md(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(fmt(v) for v in r) + " |" for r in rows]
    return "\n".join(out)


def make_tables(out):
    parts = []
    if (out / "b1_delta_f.csv").exists():
        b1 = read_csv(out / "b1_delta_f.csv")
        rows = []
        for key in sorted({(t["N"], t["r"], t["n"]) for t in b1}):
            ts = [t for t in b1 if (t["N"], t["r"], t["n"]) == key]
            df = [float(t["delta_f"]) for t in ts]
            ds = [float(t["delta_step"]) for t in ts]
            last = ts[-1]
            rows.append((*key, max(df), float(ts[0]["delta_f"]), float(last["delta_f"]), max(ds),
                         float(last["delta_f"]) / float(last["sum_steps"]), all(t["linear_model_holds"] == "True" for t in ts)))
        parts.append("### B1 (QA-delta1): field error of run_coupled against the exact discrete sequence (40 digits), s = 1..20\n\n"
                     + md(["N", "r", "n", "max delta_f", "delta_f(1)", "delta_f(20)", "max delta_step", "delta_f(20)/sum steps",
                           "linear model holds"], rows))
    if (out / "b2_modal.csv").exists():
        b2 = read_csv(out / "b2_modal.csv")
        rows = []
        for key in sorted({(t["N"], t["n"]) for t in b2}):
            ts = [t for t in b2 if (t["N"], t["n"]) == key]
            rows.append((*key, sum(int(t["modes"]) for t in ts), sum(int(t["violations"]) for t in ts),
                         min(float(t["min_bound_over_actual"]) for t in ts), max(float(t["max_actual_rel"]) for t in ts),
                         max(float(t["max_bound_rel"]) for t in ts)))
        parts.append("### B2 (QA-delta4): strict modal bound on real M_k (40 digits)\n\n"
                     + md(["N", "n", "modes checked", "violations", "min bound/actual", "max actual rel", "max bound rel"], rows))
    for name, title in (("b3_dst_fast.csv", "B3 (QA-delta2): DST of the kernel, n = 1280, against 30 digits"),
                        ("b3_dense_elements.csv", "B3: elements of the Builder dense Psi (no argument reduction) against the a-priori model"),
                        ("b4_rank.csv", "B4 (QA-delta3): rank diagnostics of the kernel on constructed factors")):
        if (out / name).exists():
            rs = read_csv(out / name)
            keys = list(rs[0])
            parts.append(f"### {title}\n\n" + md(keys, [[t[k] for k in keys] for t in rs]))
    if (out / "b5_energy.csv").exists():
        b5 = [t for t in read_csv(out / "b5_energy.csv") if t.get("m")]
        rows = []
        for fam, level in sorted({(t["family"], (int(t["nx"]), int(t["ny"])) if t["family"] == "II" else int(t["nx"])) for t in b5},
                                 key=str):
            for N in ("2", "3"):
                for s in ("2", "3", "4"):
                    ts = [t for t in b5 if t["family"] == fam and t["N"] == N and t["s"] == s and
                          ((int(t["nx"]), int(t["ny"])) if fam == "II" else int(t["nx"])) == level]
                    if not ts:
                        continue
                    ms = [float(t["m"]) for t in ts]
                    resolved = [t for t in ts if float(t["m"]) >= 1]
                    viol = sum(1 for t in resolved if t["e1"] != "True" or t["e2"] != "True")
                    rows.append((fam, str(level), N, s, len(ts), min(ms), max(ms), len(resolved), viol,
                                 "mandatory" if min(ms) >= RESERVE_D039 else "not mandatory"))
        parts.append("### B5: energy test of run_coupled, m_s = Q_s/(10 tau_s) (min/max over the r of the level)\n\n"
                     + md(["family", "level", "N", "s", "cases", "min m", "max m", "resolved", "E1/E2 violations when resolved",
                           "area by D-039 reserve (m >= 4 for all r)"], rows))
        eta = max(float(t["eta_N"]) for t in read_csv(out / "b5_energy.csv"))
        parts.append(f"B7 (B5 runs): max eta_N = {eta:.3e}; R2 candidate eta_max = 1e-12; ratio = {eta / ETA_MAX_R2:.3e}")
    if (out / "b6_summary.csv").exists():
        rs = read_csv(out / "b6_summary.csv")
        keys = list(rs[0])
        parts.append("### B6: stopping (R2 rule with tol_inf = 1e-10, tol_E = 1e-8, s_min = 3) on run_coupled\n\n"
                     + md(keys, [[t[k] for k in keys] for t in rs]))
    if (out / "b8_zeta.csv").exists():
        rs = read_csv(out / "b8_zeta.csv")
        keys = list(rs[0])
        parts.append("### B8: zeta^(N) on the sets of run_coupled and the observed order of group B\n\n"
                     + md(keys, [[t[k] for k in keys] for t in rs]))
    if (out / "b9_rank_margins.csv").exists():
        rs = read_csv(out / "b9_rank_margins.csv")
        keys = list(rs[0])
        parts.append("### B9: rank margins on the fields of run_coupled\n\n" + md(keys, [[t[k] for k in keys] for t in rs]))
    (out / "tables.md").write_text("# R4 numerical prequalification - generated tables (scripts/r4_numerical_prequal.py)\n\n"
                                   + "\n\n".join(parts) + "\n", encoding="utf-8")


def b9_margins(out):
    """Rank margins on all fields of B5/B6 against eps_obs = c_d,obs u kappa_rep + delta_f,obs (B1, B4)."""
    b1 = read_csv(out / "b1_delta_f.csv") if (out / "b1_delta_f.csv").exists() else []
    b4 = read_csv(out / "b4_rank.csv")
    delta_f_obs = max((float(t["delta_f"]) for t in b1), default=math.nan)
    c_d_obs = max(float(t["rho_over_u_kappa_max"]) for t in b4 if t["construction"].startswith("cancellation"))
    fields = read_csv(out / "b5_energy.csv") + [dict(t, family="I", nx=t["n"], ny=t["n"]) for t in read_csv(out / "b6_stopping.csv")]
    rows = []
    for N in ("2", "3"):
        ts = [t for t in fields if t["N"] == N]
        eps = [c_d_obs * U * float(t["kappa_rep"]) + delta_f_obs for t in ts]
        ratio = [float(t["rho_N"]) / (2 * e) for t, e in zip(ts, eps)]
        rows.append({"N": N, "fields": len(ts), "min_rho_N": min(float(t["rho_N"]) for t in ts),
                     "max_kappa_rep": max(float(t["kappa_rep"]) for t in ts), "c_d_obs": c_d_obs, "delta_f_obs": delta_f_obs,
                     "min_rho_over_2eps_obs": min(ratio), "criterion_ratio_ge_100": min(ratio) >= 100,
                     "min_rho_over_K_u_kappa": min(float(t["rho_N"]) / (K_RESEARCH * U * float(t["kappa_rep"])) for t in ts),
                     "min_set_sigma_over_K_u": min(float(t["set_sigma_min"]) / (K_RESEARCH * U) for t in ts)})
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", default="docs/results/r4_numerical_prequal")
    p.add_argument("--no-hp", action="store_true")
    p.add_argument("--tables-only", action="store_true")
    args = p.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.tables_only:
        make_tables(out)
        return
    t0 = time.time()
    meta = {"base": "main ebd29154", "plan_commit": "7748877", "python": sys.version.split()[0], "numpy": np.__version__,
            "platform": platform.platform(), "K_RESEARCH": K_RESEARCH,
            "harness_matches_run_coupled": all(harness_matches_run_coupled(r, N, n, n, 8)
                                               for r, N, n in ((1.0, 2, 40), (0.2, 3, 40), (5.0, 3, 160)))}
    write_csv(out / "b4_rank.csv", b4_rank())
    write_csv(out / "b5_energy.csv", b5_energy())
    rows, summary = b6_stopping()
    write_csv(out / "b6_stopping.csv", rows)
    write_csv(out / "b6_summary.csv", summary)
    write_csv(out / "b8_zeta.csv", b8_zeta(Path("docs/results/r4_group_b")))
    if not args.no_hp:
        b1, b2 = b1_b2_high_precision()
        write_csv(out / "b1_delta_f.csv", b1)
        write_csv(out / "b2_modal.csv", b2)
        fast, dense = b3_dst()
        write_csv(out / "b3_dst_fast.csv", fast)
        write_csv(out / "b3_dense_elements.csv", dense)
    write_csv(out / "b9_rank_margins.csv", b9_margins(out))
    meta["seconds"] = round(time.time() - t0)
    (out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    make_tables(out)
    print("done", meta["seconds"])


if __name__ == "__main__":
    main()
