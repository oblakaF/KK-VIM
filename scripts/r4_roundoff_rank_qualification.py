"""R4, N = 2-3: QA qualification of round-off, QR/SVD, DST and the numerical rank (docs/R4_ROUNDOFF_RANK_QUALIFICATION.plan.md).

Plan fixed before the measurements: df37652. The working solver (``kkvim.coupled_run.run_coupled``) is the object under test
and is not changed; its arrays are recorded by transparent wrappers around ``solve_coupled_halfstep``, ``refine_once`` and
``orthogonalize_received`` in the namespace of ``kkvim.coupled_run`` (the numerical path is the same, checked bitwise).

Exact reference (plan, sec. 3, 5): the finite discrete problem with the binary64 inputs a = 1, b = float(r), D = q = 1 and exact
arithmetic (h = a/n exact). In DST-I coordinates (N23-1 (c)) the weight is (mu^x_m + mu^y_n)^2 and the target
Omega_mn = q c^x_m c^y_n / (D (mu^x_m + mu^y_n)^2); a Y-half-step with the given x-coefficients X solves for every y-mode k

    D (mu_k^2 A' + 2 mu_k B' + C') y_k = q c^y_k F',  A' = X^T X, B' = X^T diag(mu^x) X, C' = X^T diag(mu^x^2) X, F' = X^T c^x,

which is ChF-3 of the solver divided by h. Everything is ball arithmetic (python-flint arb): the exact sequence W*(s), the
exact transforms Psi x of the stored binary64 arrays, the exact one-half-step solutions with the stored given set, Gram
matrices of low-rank fields, energies, and singular values (Rump enclosures, arb). Strictness rests on arb (level V).

Shared operators (plan, sec. 5): the spectral formulation is the one of ``kkvim_reference.r4_spectral_control``; the only path
independent of the modal splitting is the nodal block system of ``nodal_sequence`` (small grids). ``_d_lower`` is the QA code
of scripts/r4_n23_qualification.py (cfa9888).

Usage (repository root):
  uv run --with python-flint==0.9.0 --with mpmath==1.4.1 python scripts/r4_roundoff_rank_qualification.py validate
  ... campaign [--transform fft|dense] [--cells SPEC] [--S 100]
  ... boundary | gl | ranktests | repro | meta
"""

import argparse
import csv
import importlib.util
import json
import math
import platform
import sys
import time
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "results" / "r4_roundoff_rank_qualification"
U = 2.0**-53
R_LABELS = ("1/5", "1/3", "1/2", "2/3", "1", "7/5", "3/2", "2", "3", "5")
FAMILY_I = (40, 80, 160, 320, 640, 1280)
FAMILY_II = ((1600, 320), (960, 320), (640, 320), (320, 640), (320, 960), (320, 1600))
S_END = 100
PREC = 256  # transforms of stored arrays, one-step exact solutions, Grams, energies
SEQ_PREC0, SEQ_PREC_MAX, SEQ_TARGET = 256, 16384, 2.0**-120  # adaptive precision of W*(s) (plan, sec. 5, item 2)
DECOMP_S = tuple(range(1, 13)) + tuple(range(20, 101, 10))  # plan, sec. 6
FORM_S = ("1", "2", "stop", "40", "end")

_flint = None


def flint():
    global _flint
    if _flint is None:
        import flint as f

        _flint = f
    return _flint


def gamma(k):
    return k * U / (1.0 - k * U)


# ======================================================================================== cells
def r_of(label):
    return Fraction(label)


def all_cells():
    cells = []
    for N in (2, 3):
        for n in FAMILY_I:
            for lab in R_LABELS:
                cells.append({"family": "I", "N": N, "r": lab, "nx": n, "ny": n})
        for nx, ny in FAMILY_II:
            lab = str(Fraction(ny, nx))
            cells.append({"family": "II", "N": N, "r": lab, "nx": nx, "ny": ny})
    return cells


def cell_key(c):
    return f"{c['family']}:N{c['N']}:r{c['r']}:{c['nx']}x{c['ny']}"


def select_cells(spec):
    """spec: 'all' or comma list of N:r:nx:ny (r as fraction)."""
    if spec in (None, "", "all"):
        return all_cells()
    out = []
    for item in spec.split(","):
        N, lab, nx, ny = item.split(":")
        fam = "I" if nx == ny else "II"
        out.append({"family": fam, "N": int(N), "r": lab, "nx": int(nx), "ny": int(ny)})
    return out


# ======================================================================================== recording the solver
class Recorder:
    """Transparent wrappers in kkvim.coupled_run: every call goes to the original function; copies are kept."""

    def __enter__(self):
        import kkvim.coupled_run as cr

        self.cr = cr
        self.calls = []
        self.orig = (cr.solve_coupled_halfstep, cr.refine_once, cr.orthogonalize_received)
        solve0, refine0, ortho0 = self.orig

        def solve(fixed_grid, fixed, solve_grid, q, D, *, direction, transform):
            step = solve0(fixed_grid, fixed, solve_grid, q, D, direction=direction, transform=transform)
            self.calls.append(("solve", step, np.array(fixed, dtype=float, copy=True), step.solved.copy(), step.modal.copy()))
            return step

        def refine(step):
            out = refine0(step)
            self.calls.append(("refine", out, out.solved.copy(), out.modal.copy()))
            return out

        def ortho(grid_received, received, companion):
            out = ortho0(grid_received, received, companion)
            self.calls.append(("qr", np.array(received, copy=True), np.array(companion, copy=True), out.received.copy(),
                               out.companion.copy(), out.field_error))
            return out

        cr.solve_coupled_halfstep, cr.refine_once, cr.orthogonalize_received = solve, refine, ortho
        return self

    def __exit__(self, *exc):
        self.cr.solve_coupled_halfstep, self.cr.refine_once, self.cr.orthogonalize_received = self.orig
        return False

    def records(self):
        recs = []
        for c in self.calls:
            if c[0] == "solve":
                step = c[1]
                recs.append({"step": step, "given": c[2], "solved": c[3], "modal": c[4], "refined": False, "qr": None})
            elif c[0] == "refine":
                recs[-1].update(refined=True, solved=c[2], modal=c[3], step=c[1])
            else:
                recs[-1]["qr"] = {"received_in": c[1], "companion_in": c[2], "received": c[3], "companion": c[4],
                                  "field_error": c[5]}
        return recs


def run_solver(cell, transform, mode, record=False):
    from kkvim.coupled_run import run_coupled

    kw = dict(transform=transform, half_steps=S_END)
    if mode == "M0":
        kw.update(research_mode=False)
    elif mode == "M1":
        kw.update(research_mode=True, stop_on_stabilization=True)
    else:
        kw.update(research_mode=True, stop_on_stabilization=False)
    b = float(r_of(cell["r"]))
    if record:
        with Recorder() as rec:
            run = run_coupled(1.0, b, 1.0, 1.0, cell["nx"], cell["ny"], cell["N"], **kw)
        return run, rec.records()
    return run_coupled(1.0, b, 1.0, 1.0, cell["nx"], cell["ny"], cell["N"], **kw), None


def log_signature(steps):
    """repr of every number of the step logs (equal strings <=> bitwise equal doubles)."""
    out = []
    for st in steps:
        rk = st.rank
        e = st.energy
        vals = [st.s, st.direction, st.eta_N, *rk.sigma.tolist(), rk.rho_N, rk.kappa_rep, *rk.set_sigma.tolist(),
                st.cond_A_given, st.cond_M_max, st.w_max, st.eps_inf, st.eps_E, st.qr_done, st.qr_field_error,
                st.P_after_qr, st.stop_reason, st.refined, st.m_s]
        if e is not None:
            vals += [e.P, e.norm_a, e.R, e.delta_P, e.Q, e.gamma, e.tau, e.status]
        out.append(repr(vals))
    return out


def num_signature(st):
    """Numerical part of one step log (without stop reason, guards and QR fields): repr, i.e. bitwise."""
    rk, e = st.rank, st.energy
    vals = [st.s, st.direction, st.eta_N, *rk.sigma.tolist(), rk.rho_N, rk.kappa_rep, *rk.set_sigma.tolist(),
            st.cond_A_given, st.cond_M_max, st.w_max, st.eps_inf, st.eps_E]
    if e is not None:
        vals += [e.P, e.norm_a, e.R, e.delta_P, e.Q, e.gamma, e.tau, e.status]
    return repr(vals)


def check_records(run, recs):
    """The recorded arrays are the arrays of the run (plan, sec. 5, item 1)."""
    ok = len(recs) == len(run.steps)
    for i, (rec, st) in enumerate(zip(recs, run.steps)):
        ok &= rec["refined"] == st.refined
        ok &= (rec["qr"] is not None) == st.qr_done
        if i + 1 < len(recs) and rec["qr"] is not None:
            ok &= np.array_equal(recs[i + 1]["given"], rec["qr"]["received"])
            ok &= np.array_equal(rec["qr"]["received_in"], rec["solved"])
    last = recs[-1]
    if last["qr"] is None:
        X, Y = (last["given"], last["solved"]) if last["step"].direction == "Y" else (last["solved"], last["given"])
        ok &= np.array_equal(X, run.x) and np.array_equal(Y, run.y)
    return bool(ok)


# ======================================================================================== ball arithmetic: axes and transforms
class Axis:
    """Exact data of G(length, n): mu_m, c_m (m = 1..n-1), the DST-I matrix Psi (lazy), at the current precision."""

    _psi_cache = {}

    def __init__(self, length: Fraction, n: int):
        f = flint()
        arb = f.arb
        self.n, self.length = n, length
        self.h = arb(length.numerator) / (length.denominator * n)
        L2 = arb(length.numerator) ** 2 / arb(length.denominator) ** 2
        self.mu, self.c = [], []
        scale = (arb(2) / n).sqrt()
        for m in range(1, n):
            t = arb(m) / (2 * n)
            self.mu.append(4 * arb(n) ** 2 / L2 * t.sin_pi() ** 2)
            self.c.append(scale * t.cot_pi() if m % 2 else arb(0))
        self.odd = list(range(0, n - 1, 2))  # 0-based indices of the odd modes m = 1, 3, ...
        self.mu_odd = [self.mu[i] for i in self.odd]
        self.c_odd = [self.c[i] for i in self.odd]

    def psi(self):
        key = (self.n, flint().ctx.prec)
        if key not in Axis._psi_cache:
            if len(Axis._psi_cache) >= 2:
                Axis._psi_cache.pop(next(iter(Axis._psi_cache)))
            f = flint()
            arb, arb_mat = f.arb, f.arb_mat
            n = self.n
            scale = (arb(2) / n).sqrt()
            tab = [scale * (arb(j) / n).sin_pi() for j in range(2 * n)]
            for j in range(0, 2 * n, n):
                tab[j] = arb(0)
            P = arb_mat(n - 1, n - 1)
            for i in range(1, n):
                for k in range(i, n):
                    v = tab[(i * k) % (2 * n)]
                    P[i - 1, k - 1] = v
                    P[k - 1, i - 1] = v
            Axis._psi_cache[key] = P
        return Axis._psi_cache[key]

    def abs_psi(self):
        key = ("abs", self.n, flint().ctx.prec)
        if key not in Axis._psi_cache:
            f = flint()
            Axis._psi_cache[key] = f.arb_mat([[abs(x) for x in row] for row in self.psi().tolist()])
        return Axis._psi_cache[key]

    def transform(self, X):
        """Psi X for a stored binary64 array X ((n - 1) x N): exact input, ball output."""
        f = flint()
        return self.psi() * f.arb_mat(np.asarray(X, dtype=float).tolist())

    def embed(self, Xodd):
        """Odd-mode coefficients (n//2 x N) -> all modes (n - 1 x N) with exact zeros at even modes."""
        f = flint()
        N = Xodd.ncols()
        M = f.arb_mat(self.n - 1, N)
        for t, i in enumerate(self.odd):
            for j in range(N):
                M[i, j] = Xodd[t, j]
        return M


def to_arb_mat(A):
    f = flint()
    A = np.asarray(A, dtype=float)
    if A.ndim == 1:
        A = A[:, None]
    return f.arb_mat(A.tolist())


def scale_rows(X, w):
    f = flint()
    K, N = X.nrows(), X.ncols()
    out = f.arb_mat(K, N)
    for i in range(K):
        wi = w[i]
        for j in range(N):
            out[i, j] = X[i, j] * wi
    return out


def hsum(A, B):
    f = flint()
    s = f.arb(0)
    for i in range(A.nrows()):
        for j in range(A.ncols()):
            s += A[i, j] * B[i, j]
    return s


def grams(X, mu):
    X1 = scale_rows(X, mu)
    X2 = scale_rows(X1, mu)
    Xt = X.transpose()
    return Xt * X, Xt * X1, Xt * X2


def col(values):
    f = flint()
    return f.arb_mat([[v] for v in values])


def halfstep_exact(G, given_mu, given_c, new_mu, new_c, modes, q=1, D=1):
    """Exact half-step: given coefficients G (rows = modes of given_mu), solution rows for the indices ``modes`` of the new
    axis (others zero is the caller's business). Returns the list of N-row solutions."""
    G0, G1, G2 = grams(G, given_mu)
    fvec = G.transpose() * col(given_c)
    out = {}
    for k in modes:
        m = new_mu[k]
        M = G0 * (m * m) + G1 * (2 * m) + G2
        out[k] = M.solve(fvec) * (q * new_c[k] / D)
    return out


def rows_to_mat(rows, K, N, index=None):
    f = flint()
    M = f.arb_mat(K, N)
    for t, (k, v) in enumerate(rows.items()):
        i = k if index is None else index[t]
        for j in range(N):
            M[i, j] = v[j, 0]
    return M


def rel_radius(M):
    big, rad = 0.0, 0.0
    for i in range(M.nrows()):
        for j in range(M.ncols()):
            x = M[i, j]
            big = max(big, abs(float(x.mid())))
            rad = max(rad, float(x.rad()))
    return rad / big if big > 0 else math.inf


def rebase(Mt):
    """Multiply by an exactly representable invertible matrix (approximate R^-1 of the QR): span unchanged (VB-16)."""
    f = flint()
    K, N = Mt.nrows(), Mt.ncols()
    Fm = np.array([[float(Mt[i, j].mid()) for j in range(N)] for i in range(K)])
    Ri = np.linalg.inv(np.linalg.qr(Fm)[1])
    T = f.arb_mat(Ri.tolist())
    if T.det().contains(0):
        raise RuntimeError("basis change not certified invertible")
    return Mt * T


# ======================================================================================== exact sequence W*(s)
class Problem:
    def __init__(self, cell):
        self.cell = cell
        self.N = cell["N"]
        self.a, self.b = Fraction(1.0), Fraction(float(r_of(cell["r"])))
        self.nx, self.ny = cell["nx"], cell["ny"]
        self.build()

    def build(self):
        self.ax, self.ay = Axis(self.a, self.nx), Axis(self.b, self.ny)
        self.prec = flint().ctx.prec


def exact_sequence_at(P, S):
    """W*(s), s = 1..S at the current precision: list of (X_odd, Y_odd) and the worst relative radius."""
    f = flint()
    ax, ay, N = P.ax, P.ay, P.N
    G = f.arb_mat(len(ax.odd), N)
    for i in range(N):
        G[i, i] = f.arb(1)  # odd modes m_i = 2i - 1: coefficient sqrt(n_x/2) e_{m_i}, scale irrelevant (field invariant)
    seq, worst = [], 0.0
    for s in range(1, S + 1):
        if s % 2:
            rows = halfstep_exact(G, ax.mu_odd, ax.c_odd, ay.mu_odd, ay.c_odd, range(len(ay.odd)))
            R = rows_to_mat(rows, len(ay.odd), N)
            pair = (G, R)
        else:
            rows = halfstep_exact(G, ay.mu_odd, ay.c_odd, ax.mu_odd, ax.c_odd, range(len(ax.odd)))
            R = rows_to_mat(rows, len(ax.odd), N)
            pair = (R, G)
        worst = max(worst, rel_radius(pair[0]), rel_radius(pair[1]))
        seq.append(pair)
        G = rebase(R)
    return seq, worst


def exact_sequence(cell, S):
    """Adaptive precision (plan, sec. 5, item 2): returns (Problem at PREC for comparisons, seq, prec used, worst radius)."""
    f = flint()
    prec = SEQ_PREC0
    while True:
        f.ctx.prec = prec
        P = Problem(cell)
        try:
            seq, worst = exact_sequence_at(P, S)
        except (ZeroDivisionError, RuntimeError):  # balls too wide to certify a solve or a basis change
            seq, worst = None, math.inf
        if worst <= SEQ_TARGET:
            break
        prec *= 2
        if prec > SEQ_PREC_MAX:
            return None, None, prec, worst
    f.ctx.prec = PREC
    Pc = Problem(cell)
    return Pc, seq, prec, worst


# ======================================================================================== certified field quantities
def chol(G):
    f = flint()
    n = G.nrows()
    L = [[f.arb(0)] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = G[i, j] - sum((L[i][k] * L[j][k] for k in range(j)), f.arb(0))
            if i == j:
                if not s > 0:
                    return None
                L[i][j] = s.sqrt()
            else:
                L[i][j] = s / L[j][j]
    return f.arb_mat(L)


def eig_desc(S):
    """Rump enclosures of the (real) eigenvalues, descending by midpoint; None if not certified."""
    f = flint()
    try:
        ev = f.acb_mat(S).eig(algorithm="rump")
    except (ValueError, ArithmeticError):
        return None
    vals = [e.real for e in ev]
    if any(not e.imag.contains(0) and not math.isfinite(float(e.imag.mid())) for e in ev):
        return None
    vals.sort(key=lambda x: float(x.mid()), reverse=True)
    return vals


def sigmas(X, Y):
    """Enclosures of the singular values of X Y^T (rank <= N), descending; None if not certified."""
    GX, GY = X.transpose() * X, Y.transpose() * Y
    L = chol(GX)
    if L is not None:
        S = L.transpose() * GY * L
    else:
        S = GX * GY
    ev = eig_desc(S)
    if ev is None:
        return None
    return [e.nonnegative_part().sqrt() if hasattr(e, "nonnegative_part") else e.sqrt() for e in ev]


def _inertia_bound(G, which):
    """Fallback for (nearly) multiple eigenvalues: Sylvester inertia of LDL^T (QA code of scripts/r4_n23_qualification.py)."""
    mod = n23_module()
    B = mod._Bare(flint().ctx.prec)
    return mod.eig_bounds(B, G, which)


def lam_max(G):
    """Ball containing an upper bound of lambda_max (Rump; else inertia: at most 0 eigenvalues above t_hi)."""
    ev = eig_desc(G)
    if ev is not None:
        return ev[0]
    _, t_hi, _ = _inertia_bound(G, 1)
    return t_hi


def lam_min(G):
    """Ball whose lower end bounds lambda_min from below (Rump; else inertia: n eigenvalues above t_lo)."""
    ev = eig_desc(G)
    if ev is not None:
        return ev[-1]
    t_lo, _, _ = _inertia_bound(G, G.nrows())
    return t_lo


def frob_diff(Xa, Ya, Xb, Yb):
    """|Xa Ya^T - Xb Yb^T|_F (balls)."""
    t = hsum(Xa.transpose() * Xa, Ya.transpose() * Ya) - 2 * hsum(Xa.transpose() * Xb, Ya.transpose() * Yb) \
        + hsum(Xb.transpose() * Xb, Yb.transpose() * Yb)
    return t.nonnegative_part().sqrt()


def frob_lr(X, Y):
    return hsum(X.transpose() * X, Y.transpose() * Y).nonnegative_part().sqrt()


class Energy:
    """a_h, l_h of low-rank fields in DST coordinates (c0 = D h_x h_y, D = q = 1)."""

    def __init__(self, P):
        self.P = P
        self.c0 = P.ax.h * P.ay.h
        self.cx, self.cy = col(P.ax.c), col(P.ay.c)
        self._omega2 = None

    def a_pairs(self, terms):
        """a_h(sum_t s_t X_t Y_t^T) for terms [(sign, X, Y)] (all-mode coefficients)."""
        ax, ay = self.P.ax, self.P.ay
        pre = [(s, X, scale_rows(X, ax.mu), Y, scale_rows(Y, ay.mu)) for s, X, Y in terms]
        pre = [(s, X, X1, scale_rows(X1, ax.mu), Y, Y1, scale_rows(Y1, ay.mu)) for s, X, X1, Y, Y1 in pre]
        tot = flint().arb(0)
        for i, ti in enumerate(pre):
            for j, tj in enumerate(pre):
                if j < i:
                    continue
                si, Xi, Xi1, Xi2, Yi, Yi1, Yi2 = ti
                sj, Xj, Xj1, Xj2, Yj, Yj1, Yj2 = tj
                v = hsum(Xi2.transpose() * Xj, Yi.transpose() * Yj) + 2 * hsum(Xi1.transpose() * Xj, Yi1.transpose() * Yj) \
                    + hsum(Xi.transpose() * Xj, Yi2.transpose() * Yj)
                tot += (1 if i == j else 2) * si * sj * v
        return self.c0 * tot

    def l(self, X, Y):
        return self.c0 * hsum(X.transpose() * self.cx, Y.transpose() * self.cy)

    def pi(self, X, Y):
        return self.a_pairs([(1, X, Y)]) / 2 - self.l(X, Y)

    def omega2(self):
        """|omega_h|^2_a = l_h(omega_h) = h_x h_y sum (c^x_m c^y_n)^2 / (mu^x_m + mu^y_n)^2."""
        if self._omega2 is None:
            f = flint()
            ax, ay = self.P.ax, self.P.ay
            cy2 = [ay.c_odd[j] ** 2 for j in range(len(ay.odd))]
            tot = f.arb(0)
            for i in range(len(ax.odd)):
                cx2, mx = ax.c_odd[i] ** 2, ax.mu_odd[i]
                row = f.arb(0)
                for j in range(len(ay.odd)):
                    d = mx + ay.mu_odd[j]
                    row += cy2[j] / (d * d)
                tot += cx2 * row
            self._omega2 = self.c0 * tot
        return self._omega2


_N23 = None


def n23_module():
    global _N23
    if _N23 is None:
        spec = importlib.util.spec_from_file_location("r4_n23_qualification", ROOT / "scripts" / "r4_n23_qualification.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _N23 = mod
    return _N23


def d_low(P, k, ksub=12):
    """N23-3 (QA implementation `_d_lower`, cfa9888) for omega_h of this problem: lower bound of d_k^2 in a_h."""
    mod = n23_module()
    B = mod._Bare(flint().ctx.prec)
    ax, ay = P.ax, P.ay
    ks = min(ksub, len(ax.odd), len(ay.odd))
    Om = [[ax.c_odd[i] * ay.c_odd[j] / (ax.mu_odd[i] + ay.mu_odd[j]) ** 2 for j in range(ks)] for i in range(ks)]
    v = mod._d_lower(B, Om, ax.mu_odd[:ks], ay.mu_odd[:ks], k)
    return None if v is None else (ax.h * ay.h) * v


# ======================================================================================== nodal path (small grids)
def nodal_L(n, h):
    f = flint()
    L = f.arb_mat(n - 1, n - 1)
    inv = 1 / (h * h)
    for i in range(n - 1):
        L[i, i] = -2 * inv
        if i > 0:
            L[i, i - 1] = inv
        if i < n - 2:
            L[i, i + 1] = inv
    return L


def nodal_halfstep(U, hg, Lg, ns, hs, Ls, sign_B=1, q=1, D=1):
    """Variant A (R4_NUMERICAL_SPEC 3.1): nodal sums A, B, C, F and the block system, solved without DST."""
    f = flint()
    N = U.ncols()
    LU = Lg * U
    A = U.transpose() * U * hg
    B = (U.transpose() * LU) * (-hg)
    C = LU.transpose() * LU * hg
    ones = f.arb_mat([[1] for _ in range(U.nrows())])
    F = U.transpose() * ones * (q * hg)
    m = ns - 1
    L2 = Ls * Ls
    K = f.arb_mat(N * m, N * m)
    for i in range(N):
        for j in range(N):
            for p in range(m):
                for t in range(max(0, p - 2), min(m, p + 3)):
                    v = A[i, j] * L2[p, t] - 2 * sign_B * B[i, j] * Ls[p, t] + (C[i, j] if p == t else 0)
                    K[i * m + p, j * m + t] = D * v
    rhs = f.arb_mat([[F[i, 0]] for i in range(N) for _ in range(m)])
    v = K.solve(rhs)
    V = f.arb_mat(m, N)
    for i in range(N):
        for p in range(m):
            V[p, i] = v[i * m + p, 0]
    return V


def nodal_sequence(cell, S, sign_B=1, swap=False):
    """Exact sequence by the nodal path (fields as nodal (X, Y) pairs)."""
    f = flint()
    N = cell["N"]
    a, b = Fraction(1.0), Fraction(float(r_of(cell["r"])))
    nx, ny = cell["nx"], cell["ny"]
    hx = f.arb(a.numerator) / (a.denominator * nx)
    hy = f.arb(b.numerator) / (b.denominator * ny)
    Lx, Ly = nodal_L(nx, hx), nodal_L(ny, hy)
    X0 = f.arb_mat(nx - 1, N)
    for l in range(1, nx):
        for i in range(N):
            X0[l - 1, i] = (f.arb((2 * i + 1) * l) / nx).sin_pi()
    G = X0
    seq = []
    for s in range(1, S + 1):
        if (s % 2) ^ swap:
            R = nodal_halfstep(G, hx, Lx, ny, hy, Ly, sign_B)
            seq.append((G, R))
        else:
            R = nodal_halfstep(G, hy, Ly, nx, hx, Lx, sign_B)
            seq.append((R, G))
        G = rebase(R)
    return seq, (hx, hy, Lx, Ly)


def nodal_sequence_adaptive(cell, S, prec0=512):
    """nodal_sequence with the precision doubled until every factor has relative radius <= SEQ_TARGET (as path 2)."""
    f = flint()
    prec = prec0
    while prec <= SEQ_PREC_MAX:
        f.ctx.prec = prec
        try:
            seq, ops = nodal_sequence(cell, S)
            worst = max(max(rel_radius(X), rel_radius(Y)) for X, Y in seq)
            if worst <= SEQ_TARGET:
                return seq, ops, prec
        except ZeroDivisionError:
            pass
        prec *= 2
    raise RuntimeError("nodal path: precision limit")


def projection_residual(W, given, hx, hy, Lx, Ly, axis, q=1, D=1):
    """2-D nodal stationarity residual of Pi_h on M_h(given) (plan V-2): max |R| and its scale."""
    f = flint()
    LW = Lx * W + W * Ly  # Delta_h W (interior, zero boundary)
    worst, scale = 0.0, 0.0
    if axis == "x":  # given x-set U_i, test functions U_i e_l^T
        for i in range(given.ncols()):
            Ui = f.arb_mat([[given[p, i]] for p in range(given.nrows())])
            t1 = (Lx * Ui).transpose() * LW
            t2 = (Ui.transpose() * LW) * Ly
            su = sum((given[p, i] for p in range(given.nrows())), f.arb(0))
            for l in range(W.ncols()):
                r = D * hx * hy * (t1[0, l] + t2[0, l]) - q * hx * hy * su
                worst = max(worst, float(abs(r).upper()))
                scale = max(scale, float(abs(q * hx * hy * su).upper()))
    else:
        for i in range(given.ncols()):
            Vi = f.arb_mat([[given[p, i]] for p in range(given.nrows())])
            t1 = LW * (Ly * Vi)
            t2 = Lx * (LW * Vi)
            sv = sum((given[p, i] for p in range(given.nrows())), f.arb(0))
            for l in range(W.nrows()):
                r = D * hx * hy * (t1[l, 0] + t2[l, 0]) - q * hx * hy * sv
                worst = max(worst, float(abs(r).upper()))
                scale = max(scale, float(abs(q * hx * hy * sv).upper()))
    return worst, scale


# ======================================================================================== numpy: rigorous nodal bounds (TwoProduct/TwoSum)
_SPLIT = 134217729.0  # 2^27 + 1


def two_prod(a, b):
    p = a * b
    ca = _SPLIT * a
    ah = ca - (ca - a)
    al = a - ah
    cb = _SPLIT * b
    bh = cb - (cb - b)
    bl = b - bh
    e = ((ah * bh - p) + ah * bl + al * bh) + al * bl
    return p, e


def two_sum(a, b):
    s = a + b
    bb = s - a
    e = (a - (s - bb)) + (b - bb)
    return s, e


def exact_terms(X, Y, sign=None):
    """Terms t (exact doubles) with sum_t t_ij = sum_k sign_k X_ik Y_jk exactly (TwoProduct; no overflow/underflow)."""
    terms = []
    for k in range(X.shape[1]):
        p, e = two_prod(X[:, k][:, None], Y[:, k][None, :])
        sg = 1.0 if sign is None else sign[k]
        terms += [sg * p, sg * e]
    return terms


def enclose_sum(terms):
    """Rigorous [lo, hi] of the exact sum of the terms: cascaded TwoSum (exact) + bound of the double sum of the errors."""
    s = terms[0]
    errs = []
    for t in terms[1:]:
        s, e = two_sum(s, t)
        errs.append(e)
    E = np.zeros_like(s)
    A = np.zeros_like(s)
    for e in errs:
        E = E + e
        A = A + np.abs(e)
    k = len(errs)
    slack = gamma(k + 1) * A * 1.0000001  # |fl(sum e) - sum e| <= gamma_{k-1} sum|e|, safety
    lo = (s + (E - slack))
    hi = (s + (E + slack))
    # the final additions s + E +- slack are themselves rounded: widen by one ulp-scale relative term
    widen = 2 * U * (np.abs(s) + np.abs(E) + slack)
    return lo - widen, hi + widen


# ======================================================================================== per-cell campaign
def solver_grids(cell):
    from kkvim.grid import Grid1D

    return Grid1D(1.0, cell["nx"]), Grid1D(float(r_of(cell["r"])), cell["ny"])


def up(x):
    return float(x.upper())


def lo(x):
    return float(x.lower())


def fmt(x):
    return "" if x is None else (repr(float(x)) if not isinstance(x, (bool, str)) else str(x))


def decomposition(P, rec, Gc, given_axis, solve_axis, transform, sig1_lo, GUmax_up):
    """Sources (1), (1'), (3'), (2a), (2b), (3) of e_step (plan, sec. 3) and the strict-bound checks (plan, sec. 4)."""
    from kkvim.coupled import sine_eigenvalues, sine_transform

    f = flint()
    arb = f.arb
    step = rec["step"]
    co = step.coefficients
    N = co.N
    ns = solve_axis.n
    Afl, Bfl, Cfl, Ffl = (to_arb_mat(M) for M in (co.A, co.B, co.C, co.F))
    mu_fl = sine_eigenvalues(step.solve_grid)
    load_fl = sine_transform(step.solve_grid, np.ones(ns - 1), transform)
    modal_fl = rec["modal"]
    # exact data of the given set
    G0, G1, G2 = grams(Gc, given_axis.mu)
    fvec = Gc.transpose() * col(given_axis.c)
    odd = solve_axis.odd
    allk = range(ns - 1)

    def solve_set(make, ks):
        V = f.arb_mat(ns - 1, N)
        for k in ks:
            M, rhs = make(k)
            v = M.solve(rhs)
            for j in range(N):
                V[k, j] = v[j, 0]
        return V

    V0 = solve_set(lambda k: (G0 * solve_axis.mu[k] ** 2 + G1 * (2 * solve_axis.mu[k]) + G2, fvec * solve_axis.c[k]), odd)
    V1 = solve_set(lambda k: (Afl * solve_axis.mu[k] ** 2 + Bfl * (2 * solve_axis.mu[k]) + Cfl, Ffl * solve_axis.c[k]), odd)
    mfl = [arb(float(m)) for m in mu_fl]
    lfl = [arb(float(x)) for x in load_fl]
    V2 = solve_set(lambda k: (Afl * mfl[k] ** 2 + Bfl * (2 * mfl[k]) + Cfl, Ffl * solve_axis.c[k]), odd)
    V3 = solve_set(lambda k: (Afl * mfl[k] ** 2 + Bfl * (2 * mfl[k]) + Cfl, Ffl * lfl[k]), allk)
    A, B, C, F = co.A, co.B, co.C, co.F
    Mfl = [mu_fl[k] ** 2 * A + 2.0 * mu_fl[k] * B + C for k in allk]  # the expression of kkvim.coupled._solve_modes
    bfl = [F * load_fl[k] for k in allk]
    V4 = solve_set(lambda k: (to_arb_mat(Mfl[k]), to_arb_mat(bfl[k])), allk)  # D = 1
    V5 = to_arb_mat(modal_fl)
    V6 = solve_axis.transform(rec["solved"])
    names = ("assembly", "eigenvalues", "load_dst", "mode_forming", "cholesky", "dst")
    seq = (V0, V1, V2, V3, V4, V5, V6)
    GU = Gc.transpose() * Gc
    out = {}
    for name, Va, Vb in zip(names, seq[:-1], seq[1:]):
        Dl = Vb - Va
        out["src_" + name] = up(hsum(GU, Dl.transpose() * Dl).nonnegative_part().sqrt()) / sig1_lo
    Dl = V6 - V0
    out["src_total"] = up(hsum(GU, Dl.transpose() * Dl).nonnegative_part().sqrt()) / sig1_lo
    # ---- (1) strict elementwise assembly bounds and their influence bound
    g = given_axis
    Un = to_arb_mat(rec["given"])
    n = g.n
    h = g.h
    Ab = np.abs(rec["given"])
    full = np.vstack([np.zeros((1, N)), rec["given"], np.zeros((1, N))])
    # exact differences
    dU = f.arb_mat(n, N)
    lU = f.arb_mat(n - 1, N)
    beta = f.arb_mat(n - 1, N)
    fa = [[arb(float(full[i, j])) for j in range(N)] for i in range(n + 1)]
    for j in range(N):
        for i in range(n):
            dU[i, j] = (fa[i + 1][j] - fa[i][j]) / h
        for i in range(1, n):
            e = fa[i - 1][j] - 2 * fa[i][j] + fa[i + 1][j]
            lU[i - 1, j] = e / (h * h)
            beta[i - 1, j] = (abs(e) * gamma(6) + abs(fa[i - 1][j] - 2 * fa[i][j]) * U * (1 + gamma(6))) / (h * h)
    A_ex = Un.transpose() * Un * h
    B_ex = dU.transpose() * dU * h
    C_ex = lU.transpose() * lU * h
    ones = f.arb_mat([[1] for _ in range(n - 1)])
    F_ex = Un.transpose() * ones * h
    absU = to_arb_mat(Ab)
    absdU = f.arb_mat([[abs(dU[i, j]) for j in range(N)] for i in range(n)])
    abslU = f.arb_mat([[abs(lU[i, j]) for j in range(N)] for i in range(n - 1)])
    gA, gB, gC = gamma(n + 1), gamma(n + 8), gamma(n + 2)
    bA = absU.transpose() * absU * (h * gA)
    bB = absdU.transpose() * absdU * (h * gB)
    lb = abslU + beta
    bC = (lb.transpose() * lb * (1 + gC) - abslU.transpose() * abslU) * h
    bF = absU.transpose() * ones * (h * gamma(n + 1))
    worst = 0.0
    viol = 0
    for (Mfl_, Mex, Mb) in ((Afl, A_ex, bA), (Bfl, B_ex, bB), (Cfl, C_ex, bC), (Ffl, F_ex, bF)):
        for i in range(Mex.nrows()):
            for j in range(Mex.ncols()):
                err = abs(Mfl_[i, j] - Mex[i, j])
                bnd = Mb[i, j]
                if up(err) > lo(bnd):
                    viol += 1
                if lo(bnd) > 0:
                    worst = max(worst, up(err) / lo(bnd))
    out["assembly_bound_ratio_max"] = worst
    out["assembly_bound_status"] = "PASS" if viol == 0 else "UNRESOLVED_OR_FAIL"
    # influence bound of (1): |v1 - v0| <= (|db| + |dM| |v0|) / lambda_low(M1) per mode
    fro = lambda M: hsum(M, M).nonnegative_part().sqrt()  # noqa: E731
    nA, nB, nC, nF = fro(bA), fro(bB), fro(bC), fro(bF)
    lAmin, lBmin, lCmin = lam_min(Afl), lam_min(Bfl), lam_min(Cfl)
    tot = arb(0)
    ok = lAmin is not None and lBmin is not None and lCmin is not None
    if ok:
        for k in odd:
            m = solve_axis.mu[k]
            lamlow = m * m * lAmin + 2 * m * lBmin + lCmin
            if not lamlow > 0:
                ok = False
                break
            v0 = f.arb_mat([[V0[k, j]] for j in range(N)])
            nv0 = fro(v0)
            bnd = (abs(solve_axis.c[k]) * nF + (m * m * nA + 2 * m * nB + nC) * nv0) / lamlow
            tot += bnd * bnd
    if ok:
        fb = (tot.sqrt() * GUmax_up.sqrt())
        out["assembly_influence_bound"] = up(fb) / sig1_lo
        out["assembly_influence_status"] = "PASS" if out["src_assembly"] <= lo(fb) / sig1_lo * (1 - 1e-12) else "UNRESOLVED_OR_FAIL"
    else:
        out["assembly_influence_bound"] = None
        out["assembly_influence_status"] = "UNRESOLVED"
    # ---- (2) strict a posteriori modal bound (exact residual) and Higham model ratio
    import scipy.linalg

    viol, worst_model = 0, 0.0
    for k in allk:
        Mk = to_arb_mat(Mfl[k])
        vk = f.arb_mat([[V5[k, j]] for j in range(N)])
        rk = to_arb_mat(bfl[k]) - Mk * vk
        lmin = lam_min(Mk)
        dv = f.arb_mat([[V5[k, j] - V4[k, j]] for j in range(N)])
        err = fro(dv)
        if lmin is None or not lmin > 0:
            viol += 1
            continue
        bnd = fro(rk) / lmin
        if up(err) > lo(bnd) * (1 + 1e-12) and up(err) > 0:
            viol += 1
        Lc = scipy.linalg.cho_factor(Mfl[k], lower=True, check_finite=False)[0]
        Lc = np.tril(Lc)
        model = gamma(3 * N + 1) * np.linalg.norm(np.abs(Lc) @ np.abs(Lc).T, 2) * np.linalg.norm(modal_fl[k]) / float(lmin.mid())
        if model > 0:
            worst_model = max(worst_model, up(err) / model)
    out["modal_bound_status"] = "PASS" if viol == 0 else "UNRESOLVED_OR_FAIL"
    out["cholesky_model_ratio_max"] = worst_model
    # ---- (3) DST of the result and of the load: norms, model c u log2 n, dense componentwise bound
    T1 = solve_axis.psi() * V5  # exact Psi modal (nodal)
    Sn = to_arb_mat(rec["solved"])
    E3 = Sn - T1
    e3 = fro(E3)
    nm = fro(V5)
    out["dst_err_over_ulog2n"] = up(e3) / (U * math.log2(ns) * lo(nm))
    cexact = col(solve_axis.c)
    El = to_arb_mat(load_fl) - cexact
    out["load_err_over_ulog2n"] = up(fro(El)) / (U * math.log2(ns) * lo(fro(cexact)))
    if transform == "dense":
        absP = solve_axis.abs_psi()
        bnd_c = 7 * U + gamma(ns - 1) * (1 + 7 * U)
        AV = absP * f.arb_mat([[abs(V5[i, j]) for j in range(N)] for i in range(ns - 1)])
        viol, worst = 0, 0.0
        for i in range(ns - 1):
            for j in range(N):
                b_ = AV[i, j] * bnd_c
                if up(abs(E3[i, j])) > lo(b_):
                    viol += 1
                if lo(b_) > 0:
                    worst = max(worst, up(abs(E3[i, j])) / lo(b_))
        out["dense_bound_status"] = "PASS" if viol == 0 else "UNRESOLVED_OR_FAIL"
        out["dense_bound_ratio_max"] = worst
    return out


def formation_check(X, Y, gx, gy):
    """Source (5): every interior node of fl(X Y^T) (field_from_sets) against the exact product; bound gamma_N |X||Y|^T."""
    from kkvim.coupled_energy import field_from_sets

    N = X.shape[1]
    w = field_from_sets(gx, gy, X, Y)[1:-1, 1:-1]
    terms = exact_terms(X, Y)
    terms.append(-w)
    lo_, hi_ = enclose_sum(terms)  # enclosure of (exact - computed)
    err_up = np.maximum(np.abs(lo_), np.abs(hi_))
    S = np.abs(X) @ np.abs(Y).T
    S_lo = S * (1 - gamma(N + 2))
    bnd = gamma(N) * S_lo
    viol = int(np.sum(err_up > bnd))
    pos = bnd > 0
    ratio = float(np.max(err_up[pos] / bnd[pos])) if np.any(pos) else 0.0
    return {"form_status": "PASS" if viol == 0 else "UNRESOLVED_OR_FAIL", "form_ratio_max": ratio, "form_violations": viol}


def eps_inf_exact(Xa, Ya, Xb, Yb):
    """Enclosure of max|Xa Ya^T - Xb Yb^T| / max|Xa Ya^T| over the interior nodes."""
    N = Xa.shape[1]
    t_d = exact_terms(np.hstack([Xa, Xb]), np.hstack([Ya, Yb]), sign=[1.0] * N + [-1.0] * N)
    dlo, dhi = enclose_sum(t_d)
    wlo, whi = enclose_sum(exact_terms(Xa, Ya))
    dmax_hi = float(np.max(np.maximum(np.abs(dlo), np.abs(dhi))))
    dmax_lo = float(np.max(np.where(dlo > 0, dlo, np.where(dhi < 0, -dhi, 0.0))))
    wmax_hi = float(np.max(np.maximum(np.abs(wlo), np.abs(whi))))
    wmax_lo = float(np.max(np.where(wlo > 0, wlo, np.where(whi < 0, -whi, 0.0))))
    return dmax_lo / wmax_hi * (1 - 4 * U), dmax_hi / wmax_lo * (1 + 4 * U)


def classify_exact(dP, Q, tau):
    """E1/E2 and the resolution rule with exact Delta P, Q (balls) and the solver's tau; AMBIGUOUS if a ball straddles."""
    tests = [(dP, -tau), (Q, 10 * tau)]
    if any(up(x) >= t > lo(x) for x, t in tests):
        return "AMBIGUOUS"
    g = abs(dP - Q)
    if up(g) > tau >= lo(g):
        return "AMBIGUOUS"
    if not lo(Q) >= 10 * tau:
        return "UNRESOLVED"
    if not lo(dP) >= -tau:
        return "E1_VIOLATED_PROVISIONAL"
    if not up(g) <= tau:
        return "E2_VIOLATED_PROVISIONAL"
    return "E1_E2_HOLD_PROVISIONAL"


def campaign_cell(cell, transform="fft", S=S_END, decomp=True, verbose=False, mode="M2"):
    """mode M2: the fixed-half-step run is recorded (M1, M0 checked as prefixes); mode M1: the stopping run is recorded."""
    from kkvim.coupled_run import TOL_E, TOL_INF

    f = flint()
    t0 = time.time()
    run2, recs = run_solver(cell, transform, mode, record=True)
    meta = {"cell": cell_key(cell), "transform": transform, "recorded_mode": mode, "status_recorded": run2.status,
            "s_end": len(run2.steps)}
    meta["records_ok"] = check_records(run2, recs)
    run1 = run2 if mode == "M1" else run_solver(cell, transform, "M1")[0]
    s_stop = len(run1.steps)
    meta.update(status_M1=run1.status, s_stop=s_stop, first_stab=run1.first_stabilization_s)
    # the last M1 step differs only in stop_reason / qr: compare the numerical part of every step
    num = num_signature
    meta["M1_prefix"] = all(num(a) == num(b) for a, b in zip(run1.steps, run2.steps[:s_stop]))
    last1 = recs[s_stop - 1]
    if run1.status == "PROVISIONAL_FIELD_STABILIZED":
        X1, Y1 = (last1["given"], last1["solved"]) if last1["step"].direction == "Y" else (last1["solved"], last1["given"])
        meta["M1_final_equal"] = bool(np.array_equal(run1.x, X1) and np.array_equal(run1.y, Y1))
    run0, _ = run_solver(cell, transform, "M0")
    meta.update(status_M0=run0.status, M0_prefix=num(run0.steps[0]) == num(run2.steps[0]),
                M0_final_equal=bool(np.array_equal(run0.y, recs[0]["solved"])))
    t_solver = time.time() - t0
    Send = min(S, len(recs))
    Pc, seq, prec_used, worst = exact_sequence(cell, Send)
    meta.update(seq_prec=prec_used, seq_worst_rel_radius=worst)
    if seq is None:
        meta["result"] = "UNRESOLVED"
        return meta, []
    f.ctx.prec = PREC
    ax, ay = Pc.ax, Pc.ay
    En = Energy(Pc)
    om2 = En.omega2()
    dl = d_low(Pc, cell["N"] - 1)
    meta["D_low_N1_lo"] = lo(dl) if dl is not None else None
    gx, gy = solver_grids(cell)
    hxy = (ax.h * ay.h).sqrt()
    rows = []
    prev = None  # (Xc, Yc, X, Y) of W~(s-1) before QR
    prevQR = None  # (Xc, Yc) of W~^QR(s-1)
    prev_df_qr = None
    for s in range(1, Send + 1):
        rec, st = recs[s - 1], run2.steps[s - 1]
        direction = rec["step"].direction
        g_axis, s_axis = (ax, ay) if direction == "Y" else (ay, ax)
        Gc = g_axis.transform(rec["given"])
        Vc = s_axis.transform(rec["solved"])
        Xs, Ys = seq[s - 1]
        Xe, Ye = ax.embed(Xs), ay.embed(Ys)
        if direction == "Y":
            Xc, Yc, Xn, Yn = Gc, Vc, rec["given"], rec["solved"]
        else:
            Xc, Yc, Xn, Yn = Vc, Gc, rec["solved"], rec["given"]
        row = {"cell": cell_key(cell), "family": cell["family"], "N": cell["N"], "r": cell["r"], "nx": cell["nx"],
               "ny": cell["ny"], "transform": transform, "s": s, "direction": direction, "in_M1": s <= s_stop,
               "refined": rec["refined"], "eta_N": st.eta_N, "status": "OK"}
        sig_star = sigmas(Xe, Ye)
        sig_tld = sigmas(Xc, Yc)
        if sig_star is None or sig_tld is None:
            row["status"] = "UNRESOLVED_SIGMA"
            rows.append(row)
            continue
        s1lo = lo(sig_star[0])
        # field errors
        dfu = up(frob_diff(Xc, Yc, Xe, Ye)) / s1lo
        # one exact half-step from the stored given set
        rowsn = halfstep_exact(Gc, g_axis.mu, g_axis.c, s_axis.mu, s_axis.c, s_axis.odd)
        Vn = rows_to_mat(rowsn, s_axis.n - 1, cell["N"])
        if direction == "Y":
            Xn_, Yn_ = Gc, Vn
        else:
            Xn_, Yn_ = Vn, Gc
        estep = up(frob_diff(Xc, Yc, Xn_, Yn_)) / s1lo
        trans = up(frob_diff(Xn_, Yn_, Xe, Ye)) / s1lo
        row.update(delta_f=dfu, e_step=estep, transfer=trans, triangle_ok=dfu <= (estep + trans) * (1 + 1e-9) + 1e-300)
        if prev_df_qr is not None and prev_df_qr > 0:
            row["L_s"] = trans / prev_df_qr
        # rank diagnostics: certified vs solver
        rk = st.rank
        sig_hat = rk.sigma
        sw_t = [x * hxy for x in sig_tld]
        sw_s = [x * hxy for x in sig_star]
        N = cell["N"]
        GX, GY = Xc.transpose() * Xc, Yc.transpose() * Yc
        lx, ly = lam_max(GX), lam_max(GY)
        kap = (lx * ly).sqrt() / sig_tld[0]
        row.update(rho_hat=rk.rho_N, kappa_hat=rk.kappa_rep, kappa_cert_up=up(kap),
                   rho_tilde_lo=lo(sig_tld[-1] / sig_tld[0]), rho_tilde_up=up(sig_tld[-1] / sig_tld[0]),
                   rho_star_lo=lo(sig_star[-1] / sig_star[0]), rho_star_up=up(sig_star[-1] / sig_star[0]))
        cd = max(up(abs(f.arb(float(sig_hat[k])) - sw_t[k])) for k in range(N)) / (U * lo(kap) * lo(sw_t[0]))
        eps = max(up(abs(f.arb(float(sig_hat[k])) - sw_s[k])) for k in range(N)) / lo(sw_s[0])
        weyl = all(up(abs(sig_tld[k] - sig_star[k])) <= up(frob_diff(Xc, Yc, Xe, Ye)) * (1 + 1e-12) for k in range(N))
        row.update(c_d_obs=cd, eps_cert=eps, weyl_ok=weyl,
                   rank_distinguishable=bool(eps < 1 and rk.rho_N > eps / (1 - eps)),
                   rho_err_over_ukappa=max(up(abs(f.arb(rk.rho_N) - sig_tld[-1] / sig_tld[0])), 0.0) / (U * lo(kap)))
        # set check sigma(V_w D_c) of the received set
        Rc = Vc
        GR = Rc.transpose() * Rc
        Dc = [1 / GR[j, j].sqrt() for j in range(N)]
        GRn = f.arb_mat([[GR[i, j] * Dc[i] * Dc[j] for j in range(N)] for i in range(N)])
        evs = eig_desc(GRn)
        if evs is not None:
            ss = evs[-1].nonnegative_part().sqrt()
            row.update(set_sigma_hat=rk.set_sigma_min, set_sigma_cert_lo=lo(ss), set_sigma_cert_up=up(ss),
                       set_err_over_u=up(abs(f.arb(rk.set_sigma_min) - ss)) / U)
        # energy route
        aW = En.a_pairs([(1, Xc, Yc)])
        lW = En.l(Xc, Yc)
        e2 = (aW - 2 * lW + om2).nonnegative_part()
        e = e2.sqrt()
        row.update(energy_err_up=up(e), energy_err_rel_up=up(e) / lo(om2.sqrt()))
        if dl is not None:
            sq = dl.sqrt()
            row.update(dlow_sqrt_lo=lo(sq), energy_rank_certified=bool(up(e) < lo(sq)),
                       energy_margin=(lo(sq) - up(e)) / lo(sq))
            Lam = (4 / ax.h ** 2 + 4 / ay.h ** 2) / (ax.mu[0] + ay.mu[0])
            rb = (sq - e) / (Lam * aW.sqrt())
            row["rho_lower_from_energy"] = lo(rb) if lo(rb) > 0 else 0.0
        # energies vs solver
        Pt = aW / 2 - lW
        en = st.energy
        row.update(P_rel_err=up(abs(f.arb(en.P) - Pt)) / lo(abs(Pt)), P_err_over_R=up(abs(f.arb(en.P) - Pt)) / en.R)
        if prevQR is not None and en.delta_P is not None:
            Pp = En.pi(prevQR[0], prevQR[1])
            dP = Pp - Pt
            Qt = En.a_pairs([(1, prevQR[0], prevQR[1]), (-1, Xc, Yc)]) / 2
            tau = en.tau
            row.update(dP_err_over_tau=up(abs(f.arb(en.delta_P) - dP)) / tau, Q_err_over_tau=up(abs(f.arb(en.Q) - Qt)) / tau,
                       gamma_exact_over_tau=up(abs(dP - Qt)) / tau, status_hat=en.status,
                       status_exact=classify_exact(dP, Qt, tau))
        if prev is not None:
            dEa = En.a_pairs([(1, Xc, Yc), (-1, prev[0], prev[1])]).nonnegative_part().sqrt() / aW.sqrt()
            row.update(epsE_hat=st.eps_E, epsE_exact_lo=lo(dEa), epsE_exact_up=up(dEa),
                       epsE_err_over_tol=up(abs(f.arb(st.eps_E) - dEa)) / TOL_E)
            ilo, ihi = eps_inf_exact(Xn, Yn, prev[2], prev[3])
            row.update(epsinf_hat=st.eps_inf, epsinf_exact_lo=ilo, epsinf_exact_up=ihi,
                       epsinf_err_over_tol=max(abs(st.eps_inf - ilo), abs(st.eps_inf - ihi)) / TOL_INF)
            sure = ihi <= TOL_INF and up(dEa) <= TOL_E
            maybe = ilo <= TOL_INF and lo(dEa) <= TOL_E
            row["stab_exact"] = "OK" if sure else ("AMBIGUOUS" if maybe else "NOT_OK")
            row["stab_hat"] = st.stabilization_ok
            # sequence change of the exact field (eps_E of W*)
        # QR: field preservation
        if rec["qr"] is not None:
            q = rec["qr"]
            Qc = s_axis.transform(q["received"])
            Cc = g_axis.transform(q["companion"])
            if direction == "Y":
                Xq, Yq = Cc, Qc
            else:
                Xq, Yq = Qc, Cc
            dqr = up(frob_diff(Xq, Yq, Xc, Yc)) / s1lo
            dfqr = up(frob_diff(Xq, Yq, Xe, Ye)) / s1lo
            row.update(delta_QR=dqr, delta_QR_over_ukappa=dqr / (U * lo(kap)), qr_field_error_hat=q["field_error"],
                       delta_f_after_QR=dfqr)
            Pq = En.pi(Xq, Yq)
            if st.P_after_qr is not None:
                row["PQR_rel_err"] = up(abs(f.arb(st.P_after_qr) - Pq)) / lo(abs(Pq))
            prevQR = (Xq, Yq)
            prev_df_qr = dfqr
        else:
            prevQR = None
            prev_df_qr = None
        prev = (Xc, Yc, Xn, Yn)
        # decomposition and strict bounds
        if decomp and s in DECOMP_S and not rec["refined"]:
            row.update(decomposition(Pc, rec, Gc, g_axis, s_axis, transform, s1lo, lam_max(Gc.transpose() * Gc)))
        marks = {"1": 1, "2": 2, "stop": s_stop, "40": 40, "end": Send}
        if s in marks.values():
            row.update(formation_check(Xn, Yn, gx, gy))
        rows.append(row)
        if verbose:
            print(s, f"df={dfu:.2e} es={estep:.2e} t={trans:.2e} cd={cd:.2f} eps={eps:.2e}", flush=True)
    meta["t_solver"] = t_solver
    meta["t_total"] = time.time() - t0
    meta["result"] = "DONE"
    return meta, rows


# ======================================================================================== aggregation of a cell
STEP_KEEP = set(DECOMP_S)


def aggregate(meta, rows):
    """Cell-level maxima/minima over ALL half-steps (plan, sec. 6: delta_f, e_step, t, delta_QR, L_s, sigma at all s)."""
    agg = dict(meta)
    ok = [r for r in rows if r.get("status") == "OK"]
    agg["n_steps_certified"] = len(ok)
    agg["n_steps_unresolved"] = len(rows) - len(ok)

    def pick(key, subset, fn):
        vals = [(r[key], r["s"]) for r in subset if r.get(key) is not None]
        if not vals:
            return None, None
        return fn(vals)

    m1 = [r for r in ok if r["in_M1"]]
    for key in ("delta_f", "e_step", "transfer", "delta_QR", "delta_QR_over_ukappa", "delta_f_after_QR", "L_s", "eps_cert",
                "c_d_obs", "rho_err_over_ukappa", "set_err_over_u", "P_err_over_R", "dP_err_over_tau", "Q_err_over_tau",
                "gamma_exact_over_tau", "epsE_err_over_tol", "epsinf_err_over_tol", "eta_N", "kappa_cert_up",
                "assembly_bound_ratio_max", "cholesky_model_ratio_max", "dst_err_over_ulog2n", "load_err_over_ulog2n",
                "form_ratio_max", "dense_bound_ratio_max", "src_assembly", "src_eigenvalues", "src_load_dst",
                "src_mode_forming", "src_cholesky", "src_dst"):
        agg["max_" + key], agg["argmax_" + key] = pick(key, ok, max)
    for key in ("delta_f", "eps_cert", "c_d_obs"):
        agg["M1_max_" + key] = pick(key, m1, max)[0]
    agg["min_rho_hat"], agg["argmin_rho_hat"] = pick("rho_hat", ok, min)
    agg["min_rho_star_lo"] = pick("rho_star_lo", ok, min)[0]
    agg["min_set_sigma_cert_lo"] = pick("set_sigma_cert_lo", ok, min)[0]
    ratios = [r["rho_hat"] * (1 - r["eps_cert"]) / r["eps_cert"] for r in ok if r.get("eps_cert") and r["eps_cert"] < 1]
    agg["min_rho_hat_over_eps_ratio"] = min(ratios) if ratios else None
    agg["all_rank_distinguishable"] = bool(ok) and all(r.get("rank_distinguishable") for r in ok) and len(ok) == len(rows)
    agg["all_triangle_ok"] = all(r.get("triangle_ok") for r in ok)
    agg["all_weyl_ok"] = all(r.get("weyl_ok") for r in ok)
    for key in ("assembly_bound_status", "assembly_influence_status", "modal_bound_status", "form_status", "dense_bound_status"):
        st = [r[key] for r in ok if r.get(key) is not None]
        agg[key] = ("NOT_RUN" if not st else ("PASS" if all(x == "PASS" for x in st) else
                    ("UNRESOLVED" if all(x in ("PASS", "UNRESOLVED") for x in st) else "UNRESOLVED_OR_FAIL")))
        agg[key + "_n"] = len(st)
        agg[key + "_n_pass"] = sum(1 for x in st if x == "PASS")
        agg[key + "_n_unresolved"] = sum(1 for x in st if x == "UNRESOLVED")
    er = [r for r in ok if r.get("energy_rank_certified") is not None]
    agg["energy_rank_certified_steps"] = sum(1 for r in er if r["energy_rank_certified"])
    agg["energy_rank_not_certified_s_ge2"] = sum(1 for r in er if r["s"] >= 2 and not r["energy_rank_certified"])
    agg["min_energy_margin_s_ge2"] = pick("energy_margin", [r for r in er if r["s"] >= 2], min)[0]
    agg["min_rho_lower_from_energy_s_ge2"] = pick("rho_lower_from_energy", [r for r in er if r["s"] >= 2], min)[0]
    flips = [r for r in ok if r.get("status_exact") is not None and r["status_exact"] != r["status_hat"]]
    agg["energy_tests"] = sum(1 for r in ok if r.get("status_exact") is not None)
    agg["energy_status_flips"] = len(flips)
    agg["energy_status_ambiguous"] = sum(1 for r in ok if r.get("status_exact") == "AMBIGUOUS")
    agg["false_hold"] = sum(1 for r in flips if r["status_hat"] == "E1_E2_HOLD_PROVISIONAL")
    sf = [r for r in ok if r.get("stab_exact") is not None]
    agg["stab_checks"] = len(sf)
    agg["stab_flips"] = sum(1 for r in sf if r["stab_exact"] != "AMBIGUOUS" and (r["stab_exact"] == "OK") != bool(r["stab_hat"]))
    agg["stab_ambiguous"] = sum(1 for r in sf if r["stab_exact"] == "AMBIGUOUS")
    agg["refined_steps"] = sum(1 for r in rows if r.get("refined"))
    for name, (s0, s1) in S_BUCKETS.items():
        sub = [r for r in ok if s0 <= r["s"] <= s1]
        for key in ("delta_f", "e_step", "transfer", "L_s", "eps_cert", "c_d_obs", "delta_QR"):
            agg[f"max_{key}_{name}"] = pick(key, sub, max)[0] if sub else None
    return agg


S_BUCKETS = {"s1": (1, 1), "s2": (2, 2), "s3_12": (3, 12), "s13_40": (13, 40), "s41_100": (41, 100)}


def keep_row(r, s_stop):
    return r["s"] in STEP_KEEP or r["s"] in (s_stop, s_stop + 1)


# ======================================================================================== output
_UP = ("delta_f", "e_step", "transfer", "delta_QR", "eps_cert", "c_d_obs", "energy_err", "assembly_influence_bound",
       "src_", "_up", "_err_over", "P_rel_err", "PQR_rel_err", "dP_err", "Q_err", "gamma_exact", "max_", "_ratio_max",
       "L_s", "kappa_cert", "halfstep_e", "measured", "baseline", "eps")
_LO = ("_lo", "min_", "energy_margin", "rho_lower")


def _direction(key):
    if any(t in key for t in _LO):
        return "down"
    if any(t in key for t in _UP):
        return "up"
    return "nearest"


def fmt_value(key, x, digits=9):
    """Floats with 9 significant digits, rounded outward for bounds (_UP keys up, _LO keys down), nearest otherwise."""
    from decimal import ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_EVEN, Context, Decimal

    if x is None:
        return ""
    if isinstance(x, (bool, np.bool_)):
        return str(bool(x))
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, str):
        return x
    x = float(x)
    if not math.isfinite(x):
        return repr(x)
    if x == 0.0:
        return "0"
    mode = {"up": ROUND_CEILING, "down": ROUND_FLOOR}.get(_direction(key), ROUND_HALF_EVEN)
    d = Context(prec=digits, rounding=mode).plus(Decimal(x))
    return format(d, "E") if (abs(x) < 1e-4 or abs(x) >= 1e6) else format(d, "f")


def write_csv(path, rows):
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: fmt_value(k, r.get(k)) for k in keys})


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# ======================================================================================== campaign driver (parallel)
def _jsonable(o):
    return o.item() if hasattr(o, "item") else str(o)


def _work(args):
    cell, transform, S, work, mode = args
    key = cell_key(cell).replace(":", "_").replace("/", "-")
    done = Path(work) / f"{transform}_{mode}_{key}.json"
    if done.exists():
        return json.loads(done.read_text(encoding="utf-8"))
    try:
        meta, rows = campaign_cell(cell, transform, S, mode=mode)
        agg = aggregate(meta, rows)
        keep = [r for r in rows if keep_row(r, meta.get("s_stop", 0))]
    except Exception as exc:  # recorded, never silent (plan, sec. 2)
        agg = {"cell": cell_key(cell), "transform": transform, "result": "ERROR", "error": repr(exc)}
        keep = []
    payload = json.loads(json.dumps({"agg": agg, "rows": keep}, default=_jsonable))
    done.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def _init_worker():
    import os

    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # bitwise the same arrays with 1 and 24 threads (checked, meta)


def run_campaign(cells, transform, S, work, workers, tag, out, mode="M2"):
    from multiprocessing import Pool

    Path(work).mkdir(parents=True, exist_ok=True)
    jobs = [(c, transform, S, work, mode) for c in cells]
    order = {cell_key(c): i for i, c in enumerate(cells)}
    results = []
    t0 = time.time()
    if workers <= 1:
        for j in jobs:
            results.append(_work(j))
            print(results[-1]["agg"].get("cell"), results[-1]["agg"].get("result"), round(time.time() - t0), flush=True)
    else:
        with Pool(workers, initializer=_init_worker) as pool:
            for res in pool.imap_unordered(_work, sorted(jobs, key=lambda j: -j[0]["nx"] * j[0]["ny"]), chunksize=1):
                results.append(res)
                print(res["agg"].get("cell"), res["agg"].get("result"), round(time.time() - t0), flush=True)
    results.sort(key=lambda r: order.get(r["agg"]["cell"], 10**9))
    write_csv(Path(out) / f"cells_{tag}.csv", [r["agg"] for r in results])
    write_csv(Path(out) / f"steps_{tag}.csv", [row for r in results for row in r["rows"]])
    return results


# ======================================================================================== validation V-1 ... V-5
V_GRIDS = ((8, 8), (12, 12), (16, 16), (24, 24), (12, 20), (20, 12))


def _nodal_from_coeff(P, Xodd, Yodd):
    Xn = P.ax.psi() * P.ax.embed(Xodd)
    Yn = P.ay.psi() * P.ay.embed(Yodd)
    return Xn * Yn.transpose()


def _overlap_width(A, B):
    ov, wid, big = True, 0.0, 0.0
    for i in range(A.nrows()):
        for j in range(A.ncols()):
            ov &= bool(A[i, j].overlaps(B[i, j]))
            wid = max(wid, float(A[i, j].rad()), float(B[i, j].rad()))
            big = max(big, abs(float(A[i, j].mid())))
    return ov, wid / big


def validate(out, S=20):
    f = flint()
    rows = []
    t0 = time.time()
    for (nx, ny) in V_GRIDS:
        for N in (2, 3):
            for lab in ("1/5", "1", "5"):
                cell = {"family": "V", "N": N, "r": lab, "nx": nx, "ny": ny}
                nseq, (hx, hy, Lx, Ly), prec = nodal_sequence_adaptive(cell, S)
                f.ctx.prec = prec  # path 2 at the same precision, adaptive as in the campaign
                P = Problem(cell)
                seq, worst = exact_sequence_at(P, S)
                while worst > SEQ_TARGET:
                    f.ctx.prec *= 2
                    P = Problem(cell)
                    seq, worst = exact_sequence_at(P, S)
                    nseq, (hx, hy, Lx, Ly) = nodal_sequence(cell, S)
                neg = {}
                for name, kw in (("sign", {"sign_B": -1}), ("swap", {"swap": True})):
                    try:
                        neg[name] = nodal_sequence(cell, 2, **kw)[0]
                    except (ZeroDivisionError, ValueError) as exc:  # wrong formula: singular system or shape mismatch
                        neg[name] = "failed:" + type(exc).__name__
                for s in range(1, S + 1):
                    W2 = _nodal_from_coeff(P, *seq[s - 1])
                    W6 = nseq[s - 1][0] * nseq[s - 1][1].transpose()
                    ov, wid = _overlap_width(W2, W6)
                    Xodd, Yodd = seq[s - 1]
                    given = P.ax.psi() * P.ax.embed(Xodd) if s % 2 else P.ay.psi() * P.ay.embed(Yodd)
                    res, scale = projection_residual(W2, given, hx, hy, Lx, Ly, "x" if s % 2 else "y")
                    row = {"nx": nx, "ny": ny, "N": N, "r": lab, "s": s, "V1_overlap": ov, "V1_rel_width": wid,
                           "V1_pass": bool(ov and wid <= 1e-30), "V2_residual_over_scale": res / scale,
                           "V2_pass": bool(res <= 1e-30 * scale)}
                    if s <= 2:
                        for name, col_, use in (("sign", "V5b_sign_detected", s == 2),
                                                ("swap", "V5c_swap_detected", nx != ny or lab != "1")):
                            if not use:
                                row[col_] = None
                            elif isinstance(neg[name], str):
                                row[col_] = True
                                row[col_ + "_how"] = neg[name]
                            else:
                                Wn = neg[name][s - 1][0] * neg[name][s - 1][1].transpose()
                                row[col_] = not _overlap_width(W2, Wn)[0]
                                row[col_ + "_how"] = "mismatch" if row[col_] else "no_difference"
                    rows.append(row)
        print("V-1/V-2", nx, ny, round(time.time() - t0), flush=True)
    write_csv(Path(out) / "validate_v1_v2.csv", rows)
    from kkvim_reference.r4_spectral_control import SpectralGrid, run_control

    rows3 = []
    for n in (40, 160):
        for N in (2, 3):
            for lab in ("1/5", "1", "5"):
                cell = {"family": "I", "N": N, "r": lab, "nx": n, "ny": n}
                f.ctx.prec = 256
                P = Problem(cell)
                seq, _ = exact_sequence_at(P, 20)
                ctrl = run_control(SpectralGrid(1.0, float(r_of(lab)), n, n), N, 20)
                for s in range(1, 21):
                    Xn = P.ax.psi() * P.ax.embed(seq[s - 1][0])
                    Yn = P.ay.psi() * P.ay.embed(seq[s - 1][1])
                    Xm = np.array([[float(Xn[i, j].mid()) for j in range(N)] for i in range(n - 1)])
                    Ym = np.array([[float(Yn[i, j].mid()) for j in range(N)] for i in range(n - 1)])
                    Wm = Xm @ Ym.T
                    d = float(np.max(np.abs(ctrl[s - 1].W - Wm)) / np.max(np.abs(Wm)))
                    rows3.append({"n": n, "N": N, "r": lab, "s": s, "V3_rel_diff": d, "V3_pass": d <= 1e-12})
    write_csv(Path(out) / "validate_v3.csv", rows3)
    print("V-3", round(time.time() - t0), flush=True)
    rows4 = []
    for n in (40, 160):
        for N in (2, 3):
            for lab in ("1/5", "1", "5"):
                cell = {"family": "I", "N": N, "r": lab, "nx": n, "ny": n}
                runr, recs = run_solver(cell, "fft", "M2", record=True)
                run, _ = run_solver(cell, "fft", "M2")
                rows4.append({"n": n, "N": N, "r": lab, "s_end": len(run.steps), "records_ok": check_records(runr, recs),
                              "log_equal": log_signature(runr.steps) == log_signature(run.steps),
                              "final_equal": bool(np.array_equal(runr.x, run.x) and np.array_equal(runr.y, run.y))})
    write_csv(Path(out) / "validate_v4.csv", rows4)
    print("V-4", round(time.time() - t0), flush=True)
    rows5 = validate_v5()
    write_csv(Path(out) / "validate_v5.csv", rows5)
    summary = {
        "V1": all(r["V1_pass"] for r in rows), "V2": all(r["V2_pass"] for r in rows),
        "V3": all(r["V3_pass"] for r in rows3),
        "V4": all(r["records_ok"] and r["log_equal"] and r["final_equal"] for r in rows4),
        "V5": all(r["detected"] for r in rows5) and all(r.get("V5b_sign_detected") in (True, None) for r in rows)
        and all(r.get("V5c_swap_detected") in (True, None) for r in rows),
        "V1_cases": len(rows), "V3_cases": len(rows3), "V4_cells": len(rows4), "V5_controls": len(rows5),
        "V1_max_rel_width": max(r["V1_rel_width"] for r in rows),
        "V2_max_residual_over_scale": max(r["V2_residual_over_scale"] for r in rows),
        "V3_max_rel_diff": max(r["V3_rel_diff"] for r in rows3), "seconds": round(time.time() - t0, 1)}
    (Path(out) / "validate_summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return summary


def validate_v5():
    """Negative controls (a), (d), (e) of V-5; (b), (c) are in validate()."""
    from kkvim.coupled_rank import rank_diagnostics
    from kkvim.grid import Grid1D
    from kkvim_reference.r4_rank_diagnostics import cancellation_set, exact_rank

    f = flint()
    rows = []
    cell = {"family": "I", "N": 3, "r": "1", "nx": 40, "ny": 40}
    run, recs = run_solver(cell, "fft", "M2", record=True)
    f.ctx.prec = 256
    P = Problem(cell)
    seq, _ = exact_sequence_at(P, 6)
    ax, ay = P.ax, P.ay
    s = 5
    rec = recs[s - 1]
    Gc = ax.transform(rec["given"])
    Xe, Ye = ax.embed(seq[s - 1][0]), ay.embed(seq[s - 1][1])
    s1 = lo(sigmas(Xe, Ye)[0])
    base = up(frob_diff(Gc, ay.transform(rec["solved"]), Xe, Ye)) / s1
    i = np.arange(1, 40)
    pert = np.zeros_like(rec["solved"])
    pert[:, 0] = 1e-12 * np.max(np.abs(rec["solved"])) * np.sin(np.pi * i / 40)
    Vp = rec["solved"] + pert
    meas = up(frob_diff(Gc, ay.transform(Vp), Xe, Ye)) / s1
    dV = to_arb_mat(Vp) - to_arb_mat(rec["solved"])
    Un = to_arb_mat(rec["given"])
    expct = up(hsum(Un.transpose() * Un, dV.transpose() * dV).sqrt()) / s1
    rows.append({"control": "a_injection_1e-12", "baseline": base, "measured": meas, "expected": expct,
                 "detected": bool(0.5 * expct <= meas <= 2 * expct)})
    Pw = Problem(cell)
    Pw.b = Pw.b * Fraction(1 + 10**-10)
    Pw.build()
    seqw, _ = exact_sequence_at(Pw, 6)
    Xw, Yw = ax.embed(seqw[s - 1][0]), ay.embed(seqw[s - 1][1])
    measw = up(frob_diff(Gc, ay.transform(rec["solved"]), Xw, Yw)) / s1
    rows.append({"control": "d_wrong_h_1e-10", "baseline": base, "measured": measw, "expected": 1e-10,
                 "detected": bool(measw > 1e3 * base)})
    g = Grid1D(1.0, 40)
    En = Energy(P)
    dl = d_low(P, 2)
    hxy = (ax.h * ay.h).sqrt()
    for K in (1.0, 1e3, 1e6):
        Uc, Vc = cancellation_set(39, 39, K, seed=1, N=3)
        rk = rank_diagnostics(g, Uc, g, Vc)
        Xc_, Yc_ = ax.transform(Uc), ay.transform(Vc)
        sg = sigmas(Xc_, Yc_)
        eps = math.inf if sg is None else max(up(abs(f.arb(float(rk.sigma[k])) - sg[k] * hxy)) for k in range(3)) / lo(sg[0] * hxy)
        cert_rank = bool(eps < 1 and rk.rho_N > eps / (1 - eps))
        e = (En.a_pairs([(1, Xc_, Yc_)]) - 2 * En.l(Xc_, Yc_) + En.omega2()).nonnegative_part().sqrt()
        cert_energy = bool(up(e) < lo(dl.sqrt()))
        rows.append({"control": f"e_exact_rank2_K{K:g}", "exact_rank": exact_rank(Uc, Vc), "rho_hat": rk.rho_N,
                     "eps_cert": eps, "certified_by_rho": cert_rank, "certified_by_energy": cert_energy,
                     "detected": bool(not cert_rank and not cert_energy)})
    # (e') the rank-2 truncation of the stored field W~(5), written with 3 terms: energy route must not certify
    Xr, Yr = Gc, ay.transform(rec["solved"])
    Ym = np.array([[float(Yr[i_, j].mid()) for j in range(3)] for i_ in range(39)])
    Xm = np.array([[float(Xr[i_, j].mid()) for j in range(3)] for i_ in range(39)])
    Uu, Ss, Vt = np.linalg.svd(Xm @ Ym.T)
    X2 = to_arb_mat(np.column_stack([Uu[:, 0] * Ss[0], Uu[:, 1] * Ss[1], Uu[:, 0] * Ss[0] + Uu[:, 1] * Ss[1]]))
    Y2 = to_arb_mat(np.column_stack([Vt[0], Vt[1], 0 * Vt[0]]))
    e = (En.a_pairs([(1, X2, Y2)]) - 2 * En.l(X2, Y2) + En.omega2()).nonnegative_part().sqrt()
    rows.append({"control": "e_rank2_truncation_of_W5", "certified_by_energy": bool(up(e) < lo(dl.sqrt())),
                 "energy_err_up": up(e), "dlow_sqrt_lo": lo(dl.sqrt()), "detected": bool(not up(e) < lo(dl.sqrt()))})
    return rows


# ======================================================================================== G-1 boundary grids
def boundary(out):
    from kkvim.coupled_run import STOPPED_RANK_LOSS_PROVEN

    f = flint()
    rows = []
    for N in (2, 3):
        for lab in ("1/5", "1", "5"):
            for nx, ny in ((2 * N, 2 * N - 1), (2 * N - 1, 2 * N)):
                cell = {"family": "B", "N": N, "r": lab, "nx": nx, "ny": ny}
                run, _ = run_solver(cell, "fft", "M1")
                rows.append({"cell": cell_key(cell), "status": run.status, "steps": len(run.steps),
                             "S1_ok": run.status == STOPPED_RANK_LOSS_PROVEN and len(run.steps) == 0})
            for n in (2 * N, 2 * N + 1):
                cell = {"family": "B", "N": N, "r": lab, "nx": n, "ny": n}
                run, recs = run_solver(cell, "fft", "M1", record=True)
                f.ctx.prec = 256
                P = Problem(cell)
                seq, _ = exact_sequence_at(P, len(recs))
                for s in range(1, len(recs) + 1):
                    rec, st = recs[s - 1], run.steps[s - 1]
                    g_axis, s_axis = (P.ax, P.ay) if rec["step"].direction == "Y" else (P.ay, P.ax)
                    Gc, Vc = g_axis.transform(rec["given"]), s_axis.transform(rec["solved"])
                    Xc, Yc = (Gc, Vc) if rec["step"].direction == "Y" else (Vc, Gc)
                    Xe, Ye = P.ax.embed(seq[s - 1][0]), P.ay.embed(seq[s - 1][1])
                    ss, st_ = sigmas(Xe, Ye), sigmas(Xc, Yc)
                    rows.append({"cell": cell_key(cell), "status": run.status, "s": s, "rho_hat": st.rank.rho_N,
                                 "rho_star_lo": lo(ss[-1] / ss[0]) if ss else None,
                                 "sigmaN_star_positive": bool(ss and lo(ss[-1]) > 0),
                                 "rho_tilde_lo": lo(st_[-1] / st_[0]) if st_ else None,
                                 "delta_f": up(frob_diff(Xc, Yc, Xe, Ye)) / lo(ss[0]) if ss else None,
                                 "guards_passed": bool(st.guards.passed) if st.guards else None})
    write_csv(Path(out) / "boundary_g1.csv", rows)
    return rows


# ======================================================================================== G-2 GL(N)
GL_CELLS = [(3, lab, n) for lab in ("1/5", "1", "5") for n in (40, 640, 1280)] + [(2, "1", 1280)]
GL_J = (0, 2, 4, 6, 8, 10, 12)


def _shear(N, cond):
    """T = I + k (strictly upper ones) with cond_2(T) = cond (bisection on k); I for cond = 1."""
    if cond <= 1:
        return np.eye(N)
    E = np.triu(np.ones((N, N)), 1)
    lo_, hi_ = 0.0, 1e13
    for _ in range(300):
        k = 0.5 * (lo_ + hi_)
        if np.linalg.cond(np.eye(N) + k * E) < cond:
            lo_ = k
        else:
            hi_ = k
    return np.eye(N) + hi_ * E


def gl_tests(out, cells=None):
    from kkvim.coupled import CoupledHalfStepError, coupled_residual, solve_coupled_halfstep
    from kkvim.coupled_rank import rank_diagnostics
    from kkvim_reference.r4_rank_diagnostics import dense_transform

    f = flint()
    rows = []
    for N, lab, n in (cells or GL_CELLS):
        cell = {"family": "I", "N": N, "r": lab, "nx": n, "ny": n}
        run, recs = run_solver(cell, "fft", "M2", record=True)
        run1, _ = run_solver(cell, "fft", "M1")
        Send = len(recs)
        Pc, seq, _, _ = exact_sequence(cell, Send)
        f.ctx.prec = PREC
        ax, ay = Pc.ax, Pc.ay
        hxy = (ax.h * ay.h).sqrt()
        for s in sorted({1, 2, len(run1.steps), 40, Send}):
            rec = recs[s - 1]
            direction = rec["step"].direction
            g_axis, s_axis = (ax, ay) if direction == "Y" else (ay, ax)
            Xe, Ye = ax.embed(seq[s - 1][0]), ay.embed(seq[s - 1][1])
            sstar = sigmas(Xe, Ye)
            s1 = lo(sstar[0])
            for kind in ("dense", "shear"):
                for j in GL_J:
                    cond = 10.0**j
                    T = dense_transform(N, cond, seed=j + 7) if kind == "dense" else _shear(N, cond)
                    UT = rec["given"] @ T
                    VT = rec["solved"] @ np.linalg.inv(T).T
                    rk = rank_diagnostics(rec["step"].fixed_grid, UT, rec["step"].solve_grid, VT)
                    Gc, Vc = g_axis.transform(UT), s_axis.transform(VT)
                    Xc, Yc = (Gc, Vc) if direction == "Y" else (Vc, Gc)
                    st_ = sigmas(Xc, Yc)
                    row = {"cell": cell_key(cell), "s": s, "T": kind, "cond_T": float(np.linalg.cond(T)),
                           "kappa_hat": rk.kappa_rep, "rho_hat": rk.rho_N, "rho_star_lo": lo(sstar[-1] / sstar[0])}
                    if st_ is not None:
                        GX, GY = Xc.transpose() * Xc, Yc.transpose() * Yc
                        kap = (lam_max(GX) * lam_max(GY)).sqrt() / st_[0]
                        dft = up(frob_diff(Xc, Yc, Xe, Ye)) / s1
                        cd = max(up(abs(f.arb(float(rk.sigma[k])) - st_[k] * hxy)) for k in range(N)) \
                            / (U * lo(kap) * lo(st_[0] * hxy))
                        eps = max(up(abs(f.arb(float(rk.sigma[k])) - sstar[k] * hxy)) for k in range(N)) / lo(sstar[0] * hxy)
                        row.update(kappa_cert=up(kap), delta_f=dft, delta_f_over_ukappa=dft / (U * lo(kap)), c_d_obs=cd,
                                   eps_cert=eps, distinguishable=bool(eps < 1 and rk.rho_N > eps / (1 - eps)),
                                   rho_tilde_lo=lo(st_[-1] / st_[0]))
                    stp = None
                    if rec["qr"] is not None:
                        Gq = rec["qr"]["received"] @ T
                        try:
                            stp = solve_coupled_halfstep(rec["step"].solve_grid, Gq, rec["step"].fixed_grid, 1.0, 1.0,
                                                         direction="X" if direction == "Y" else "Y", transform="fft")
                            row["halfstep_status"] = "SOLVED"
                        except CoupledHalfStepError as exc:  # explicit error of the solver, no fallback (recorded)
                            row["halfstep_status"] = exc.code
                    if stp is not None:
                        _, eta = coupled_residual(stp.solve_grid, stp.coefficients, 1.0, stp.solved)
                        Gqc = s_axis.transform(Gq)
                        rowsn = halfstep_exact(Gqc, s_axis.mu, s_axis.c, g_axis.mu, g_axis.c, g_axis.odd)
                        Vn = rows_to_mat(rowsn, g_axis.n - 1, N)
                        Vh = g_axis.transform(stp.solved)
                        if direction == "Y":
                            es, nrm = up(frob_diff(Vh, Gqc, Vn, Gqc)), lo(frob_lr(Vn, Gqc))
                        else:
                            es, nrm = up(frob_diff(Gqc, Vh, Gqc, Vn)), lo(frob_lr(Gqc, Vn))
                        ct = float(np.linalg.cond(Gq))
                        row.update(halfstep_cond_given=ct, halfstep_e_step=es / nrm, halfstep_eta=eta,
                                   halfstep_e_over_u_cond2=es / nrm / (U * ct * ct))
                    rows.append(row)
        print("G-2", cell_key(cell), flush=True)
    write_csv(Path(out) / "gl_g2.csv", rows)
    return rows


# ======================================================================================== G-3 rank loss / indistinguishability
def rank_tests(out):
    from kkvim.coupled_rank import rank_diagnostics
    from kkvim.coupled_run import _guards
    from kkvim.grid import Grid1D
    from kkvim_reference.r4_rank_diagnostics import cancellation_set, exact_rank, tiny_rank_set

    f = flint()
    f.ctx.prec = 512
    rows = []
    for n in (40, 160):
        cell = {"family": "I", "N": 3, "r": "1", "nx": n, "ny": n}
        P = Problem(cell)
        g = Grid1D(1.0, n)
        hxy = (P.ax.h * P.ay.h).sqrt()
        cases = []
        for K in (1.0, 1e2, 1e4, 1e6, 1e8):
            for N in (2, 3):
                cases.append((f"a_rank{N - 1}_K{K:g}_N{N}", *cancellation_set(n - 1, n - 1, K, seed=3, N=N)))
        rng = np.random.default_rng(5)
        uu = rng.integers(-7, 8, size=(n - 1, 2)).astype(float)
        vv = rng.integers(-7, 8, size=(n - 1, 3)).astype(float)
        cases.append(("a_rank2_u1_u2_u1+u2", np.column_stack([uu[:, 0], uu[:, 1], uu[:, 0] + uu[:, 1]]), vv))
        for e in (30, 45, 55, 70):
            cases.append((f"b_full_rank_2^-{e}", *tiny_rank_set(n - 1, n - 1, e, seed=4)))
        for name, Uc, Vc in cases:
            N = Uc.shape[1]
            rk = rank_diagnostics(g, Uc, g, Vc)
            gd = _guards(rk)
            Xc, Yc = P.ax.transform(Uc), P.ay.transform(Vc)
            sg = sigmas(Xc, Yc)
            row = {"n": n, "case": name, "N": N, "exact_rank": exact_rank(Uc, Vc), "rho_hat": rk.rho_N,
                   "kappa_hat": rk.kappa_rep, "guards_passed": gd.passed}
            if sg is not None:
                eps = max(up(abs(f.arb(float(rk.sigma[k])) - sg[k] * hxy)) for k in range(N)) / lo(sg[0] * hxy)
                row.update(sigmaN_rel_lo=lo(sg[-1] / sg[0]), sigmaN_rel_up=up(sg[-1] / sg[0]),
                           sigmaN_positive_certified=bool(lo(sg[-1]) > 0), eps_cert=eps,
                           distinguishable=bool(eps < 1 and rk.rho_N > eps / (1 - eps)))
            else:
                row["sigma_certified"] = False
            rows.append(row)
    write_csv(Path(out) / "ranktests_g3.csv", rows)
    return rows


# ======================================================================================== meta
THREAD_CELLS = ("3:1:320:320", "2:5:1600:320")


def _run_hash(spec):
    import hashlib

    run, recs = run_solver(select_cells(spec)[0], "fft", "M2", record=True)
    h = hashlib.sha256()
    for line in log_signature(run.steps):
        h.update(line.encode())
    for r in recs:
        h.update(r["solved"].tobytes())
        h.update(r["given"].tobytes())
    return h.hexdigest()


def meta(out):
    import os
    import subprocess

    import scipy

    def git(*a):
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()

    threads = {}
    for t in ("1", "24"):
        env = dict(os.environ, OPENBLAS_NUM_THREADS=t)
        code = ("import sys; sys.path.insert(0, %r); import r4_roundoff_rank_qualification as q; "
                "print(' '.join(q._run_hash(c) for c in q.THREAD_CELLS))") % str(ROOT / "scripts")
        threads[t] = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True).stdout.split()
    info = {
        "plan_commit": "df376524ffa638bca9de2cd9f31cff5e5c1cb81d", "head": git("rev-parse", "HEAD"),
        "base": "a2307cd8030db812ff1769bc775c0e5485b8f9e2", "untracked_or_modified": git("status", "--porcelain").splitlines(),
        "python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__, "python_flint": flint().__version__,
        "mpmath": __import__("mpmath").__version__, "platform": platform.platform(), "processor": platform.processor(),
        "openblas": np.show_config(mode="dicts")["Build Dependencies"]["blas"].get("openblas configuration", ""),
        "blas_threads_bitwise": {"cells": THREAD_CELLS, "hash_threads_1": threads["1"], "hash_threads_24": threads["24"],
                                 "equal": threads["1"] == threads["24"] and len(threads["1"]) == len(THREAD_CELLS)},
        "precision": {"PREC": PREC, "SEQ_PREC0": SEQ_PREC0, "SEQ_PREC_MAX": SEQ_PREC_MAX, "SEQ_TARGET": SEQ_TARGET},
        "u": U}
    (Path(out) / "meta.json").write_text(json.dumps(info, indent=1) + chr(10), encoding="utf-8")
    print(json.dumps(info, indent=1))
    return info


# ======================================================================================== summary tables
def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _mx(rows, key):
    v = [_f(r.get(key)) for r in rows]
    v = [x for x in v if x is not None]
    return max(v) if v else None


def _mn(rows, key):
    v = [_f(r.get(key)) for r in rows]
    v = [x for x in v if x is not None]
    return min(v) if v else None


def _cnt(rows, key, val):
    return sum(1 for r in rows if r.get(key) == val)


def summarize_cells(cells):
    out = {"cells": len(cells), "done": _cnt(cells, "result", "DONE")}
    for key in ("max_delta_f", "M1_max_delta_f", "max_e_step", "max_transfer", "max_L_s", "max_eps_cert", "M1_max_eps_cert",
                "max_c_d_obs", "M1_max_c_d_obs", "max_rho_err_over_ukappa", "max_set_err_over_u", "max_delta_QR",
                "max_delta_QR_over_ukappa", "max_delta_f_after_QR", "max_P_err_over_R", "max_dP_err_over_tau",
                "max_Q_err_over_tau", "max_gamma_exact_over_tau", "max_epsE_err_over_tol", "max_epsinf_err_over_tol",
                "max_eta_N", "max_kappa_cert_up", "max_assembly_bound_ratio_max", "max_cholesky_model_ratio_max",
                "max_dst_err_over_ulog2n", "max_load_err_over_ulog2n", "max_form_ratio_max", "max_dense_bound_ratio_max",
                "max_src_assembly", "max_src_eigenvalues", "max_src_load_dst", "max_src_mode_forming", "max_src_cholesky",
                "max_src_dst", "seq_prec", "s_end", "s_stop"):
        out[key] = _mx(cells, key)
    for key in ("min_rho_hat", "min_rho_star_lo", "min_set_sigma_cert_lo", "min_rho_hat_over_eps_ratio",
                "min_energy_margin_s_ge2", "min_rho_lower_from_energy_s_ge2", "s_stop"):
        out[key] = _mn(cells, key)
    for name in S_BUCKETS:
        for key in ("delta_f", "e_step", "transfer", "L_s", "eps_cert", "c_d_obs", "delta_QR"):
            out[f"max_{key}_{name}"] = _mx(cells, f"max_{key}_{name}")
    for key in ("all_rank_distinguishable", "all_triangle_ok", "all_weyl_ok", "records_ok", "M1_prefix", "M0_prefix"):
        out[key + "_true"] = _cnt(cells, key, "True")
    for key in ("assembly_bound_status", "assembly_influence_status", "modal_bound_status", "form_status", "dense_bound_status"):
        out[key + "_PASS"] = _cnt(cells, key, "PASS")
        out[key + "_NOT_RUN"] = _cnt(cells, key, "NOT_RUN")
    for key in ("energy_rank_not_certified_s_ge2", "energy_status_flips", "energy_status_ambiguous", "false_hold", "stab_flips",
                "stab_ambiguous", "refined_steps", "energy_tests", "stab_checks", "n_steps_certified", "n_steps_unresolved"):
        out[key + "_sum"] = sum(int(float(r[key])) for r in cells if r.get(key) not in (None, ""))
    st = {}
    for r in cells:
        st[r.get("status_recorded", "")] = st.get(r.get("status_recorded", ""), 0) + 1
    out["status_recorded"] = st
    st1 = {}
    for r in cells:
        st1[r.get("status_M1", "")] = st1.get(r.get("status_M1", ""), 0) + 1
    out["status_M1"] = st1
    return out


def summary(out):
    out = Path(out)
    res = {}
    groups = {}
    for tag in ("fft_M2", "dense_M1", "dense_M2"):
        path = out / f"cells_{tag}.csv"
        if not path.exists():
            continue
        cells = read_csv(path)
        res[tag] = {"all": summarize_cells(cells)}
        for N in ("2", "3"):
            for fam in ("I", "II"):
                sub = [c for c in cells if c["cell"].startswith(f"{fam}:N{N}:")]
                if sub:
                    res[tag][f"{fam}_N{N}"] = summarize_cells(sub)
            for n in FAMILY_I:
                sub = [c for c in cells if c["cell"].startswith(f"I:N{N}:") and c["cell"].endswith(f":{n}x{n}")]
                if sub:
                    groups.setdefault(tag, {})[f"I_N{N}_n{n}"] = summarize_cells(sub)
    res["by_grid"] = groups
    for name in ("gl_g2", "ranktests_g3", "boundary_g1"):
        path = out / f"{name}.csv"
        if path.exists():
            res[name] = {"rows": len(read_csv(path))}
    (out / "summary.json").write_text(json.dumps(res, indent=1, default=str) + chr(10), encoding="utf-8")
    print(json.dumps(res, indent=1, default=str)[:4000])
    return res


# ======================================================================================== recheck of non-evaluated strict bounds
RECHECK_KEYS = ("assembly_influence_status", "modal_bound_status", "assembly_bound_status")


def recheck(out, tag="fft_M2"):
    """Rows of steps_<tag>.csv whose strict-bound checks were not PASS: the run is recorded again (bitwise the same, plan
    sec. 5) and only the decomposition of these half-steps is recomputed (inertia fallback for multiple eigenvalues)."""
    f = flint()
    rows = read_csv(Path(out) / f"steps_{tag}.csv")
    todo = {}
    for r in rows:
        if any(r.get(k) not in ("", "PASS", None) for k in RECHECK_KEYS):
            todo.setdefault(r["cell"], []).append(int(r["s"]))
    transform = tag.split("_")[0]
    mode = tag.split("_")[1]
    res = []
    for key, ss in todo.items():
        fam, Nn, rr, grid = key.split(":")
        nx, ny = (int(x) for x in grid.split("x"))
        cell = {"family": fam, "N": int(Nn[1:]), "r": rr[1:], "nx": nx, "ny": ny}
        run, recs = run_solver(cell, transform, mode, record=True)
        f.ctx.prec = PREC
        P = Problem(cell)
        for s_ in ss:
            rec = recs[s_ - 1]
            g_axis, s_axis = (P.ax, P.ay) if rec["step"].direction == "Y" else (P.ay, P.ax)
            Gc = g_axis.transform(rec["given"])
            Vc = s_axis.transform(rec["solved"])
            Xc, Yc = (Gc, Vc) if rec["step"].direction == "Y" else (Vc, Gc)
            s1 = lo(sigmas(Xc, Yc)[0])  # sigma_1 of the stored field (the ratio test does not depend on the scale)
            d = decomposition(P, rec, Gc, g_axis, s_axis, transform, s1, lam_max(Gc.transpose() * Gc))
            res.append({"cell": key, "s": s_, **{k: d.get(k) for k in RECHECK_KEYS},
                        "src_assembly": d.get("src_assembly"), "assembly_influence_bound": d.get("assembly_influence_bound"),
                        "assembly_bound_ratio_max": d.get("assembly_bound_ratio_max")})
        print("recheck", key, len(ss), flush=True)
    write_csv(Path(out) / f"recheck_{tag}.csv", res)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["validate", "campaign", "boundary", "gl", "ranktests", "one", "meta", "summary", "recheck"])
    ap.add_argument("--transform", default="fft", choices=["fft", "dense"])
    ap.add_argument("--cells", default="all")
    ap.add_argument("--S", type=int, default=S_END)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--work", default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--tag", default="")
    ap.add_argument("--mode-run", default="M2", choices=["M1", "M2"])
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.mode == "one":
        for c in select_cells(args.cells):
            meta, rows = campaign_cell(c, args.transform, args.S, verbose=True)
            print(json.dumps(aggregate(meta, rows), default=str, indent=1))
    elif args.mode == "campaign":
        run_campaign(select_cells(args.cells), args.transform, args.S, args.work or str(out / "_work"), args.workers,
                     args.tag or f"{args.transform}_{args.mode_run}", out, mode=args.mode_run)
    elif args.mode == "validate":
        validate(out)
    elif args.mode == "boundary":
        boundary(out)
    elif args.mode == "gl":
        gl_tests(out)
    elif args.mode == "ranktests":
        rank_tests(out)
    elif args.mode == "meta":
        meta(out)
    elif args.mode == "summary":
        summary(out)
    elif args.mode == "recheck":
        recheck(out, args.tag or "fft_M2")
    return 0


if __name__ == "__main__":
    sys.exit(main())
