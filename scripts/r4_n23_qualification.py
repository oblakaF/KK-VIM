"""R4, N = 2-3: QA qualification of the N23 error bounds (docs/R4_N23_QUALIFICATION.plan.md, docs/R4_N23_QUALIFICATION.qa-report.md).

Independent QA code: the packages kkvim and kkvim_reference are NOT imported; the working solver is never called. Objects
(plan, sec. 1): the exact truncated KK-VIM sequence w'_s (odd modes <= M = 4001, target omega_M, Q22-1), its inputs of the
transfer bound E'_s (Corollary N23-6), the monotone upper bound U_s (Corollary N23-7) and the rank certificates (Theorem N23-4).

Modes (units a = 1, b = r, q = D = 1; ten control r of D-048; N = 2, 3):
  arb      ball arithmetic (python-flint, default 1024 bits): enclosures of sigma_N^L2(w'_s), |d'_s|, err_M(s) = |w'_s - omega_M|,
           w'_s(centre), T, Gamma(c), and rigorous upper bounds of E'_s and U_s. Fields are handled by Gram matrices of their
           factors; before every half-step the given set is multiplied by an exactly representable invertible matrix (the fields
           do not change, MATH_SPEC VB-16). Result: hp_arb.csv. Strictness rests on the correctness of arb (level V).
  mpmath   the same quantities by a separate implementation in mpmath (default 60 digits; Euclidean Gram-Schmidt of the given set,
           round to nearest): hp_mpmath.csv. Diagnostic.
  rank     addressed checks of N23-4 / N23-8 (plan, sec. 8): own arb certificates of mu and d_N^low/err2^up for selected
           continuous, truncated (M = 21, 23, 25, 4001) and discrete cells; directions of the printed bounds of
           scripts/r4_n23/expected/ (needs mpmath for the author's script): rank_checks.csv, printed_bounds.csv.
  combine  the matrix of the plan, sec. 4 and 9, from group_b_transfer.csv (the --n23-transfer diagnostic of
           scripts/r4_group_b_study.py on the arrays of group B), hp_arb.csv, hp_mpmath.csv and the group B results:
           matrix.csv, smax.csv, summary.json.
No status of the project is changed: PASS in matrix.csv is a candidate for the owner's decision (D-048.10).

Usage (from the repository root):
  uv run --with python-flint python scripts/r4_n23_qualification.py arb
  uv run --with mpmath python scripts/r4_n23_qualification.py mpmath
  uv run --with python-flint --with mpmath python scripts/r4_n23_qualification.py rank
  uv run python scripts/r4_n23_qualification.py combine
Options: --out DIR, --M M, --S S, --cells 1/5:3,5:2 (subset), --prec BITS, --dps DIGITS.
"""

import argparse
import csv
import importlib.util
import json
import math
import platform
import sys
import time
from decimal import ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_EVEN, Context, Decimal
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "results" / "r4_n23_qualification"
R_LABELS = ("1/5", "1/3", "1/2", "2/3", "1", "7/5", "3/2", "2", "3", "5")
NS = (2, 3)
M_MAIN, S_MAIN = 4001, 40
RULE = 0.01  # D-046, D-048.8: budget <= 1/100 of the measured quantity in the same relative norm (not changed here)
DIGITS = 45  # printed significant digits of the high-precision control values (task: >= 40)
MEASURES = ("err_abs", "e_rel", "delta_c", "E_inf")

_BIG = Context(prec=12000)


# ------------------------------------------------------------------------------------------------ decimal output, directed
def _dec(man, exp):
    man, exp = int(man), int(exp)
    if exp >= 0:
        return Decimal(man * 2**exp)
    return _BIG.divide(Decimal(man), Decimal(2**-exp))


def _fmt(d, digits, rounding):
    return format(Context(prec=digits, rounding=rounding).plus(d), "e")


def up(x, digits=DIGITS):
    """Decimal string >= the upper endpoint of the ball x (outward rounding up)."""
    man, exp = x.upper().man_exp()
    return _fmt(_dec(man, exp), digits, ROUND_CEILING)


def lo(x, digits=DIGITS):
    """Decimal string <= the lower endpoint of the ball x (outward rounding down)."""
    man, exp = x.lower().man_exp()
    return _fmt(_dec(man, exp), digits, ROUND_FLOOR)


def mid(x, digits=DIGITS):
    man, exp = x.mid().man_exp()
    return _fmt(_dec(man, exp), digits, ROUND_HALF_EVEN)


def relrad(x):
    m = abs(float(x.mid()))
    return float(x.rad()) / m if m > 0 else math.inf


def cells_of(spec):
    if not spec:
        return [(lab, N) for N in NS for lab in R_LABELS]
    out = []
    for item in spec.split(","):
        lab, N = item.split(":")
        out.append((lab, int(N)))
    return out


# ------------------------------------------------------------------------------------------------ arb: truncated sequence
class ArbProblem:
    """Truncated SSSS problem (odd modes <= M, both axes) in ball arithmetic; one instance per r (sums are shared by N)."""

    def __init__(self, rlab, M, prec):
        from flint import arb, arb_mat, ctx

        self.arb, self.arb_mat, self.ctx = arb, arb_mat, ctx
        ctx.prec = prec
        self.prec = prec
        r = Fraction(rlab)
        a, b = arb(1), arb(r.numerator) / r.denominator
        pi = arb.pi()
        self.M, self.K = M, (M + 1) // 2
        self.modes = list(range(1, M + 1, 2))
        self.kap2 = [(pi * m / a) ** 2 for m in self.modes]
        self.nu2 = [(pi * m / b) ** 2 for m in self.modes]
        self.inv_m = arb_mat([[arb(1) / m] for m in self.modes])
        self.sgn = arb_mat([[1 if i % 2 == 0 else -1] for i in range(self.K)])
        self.c0 = a * b / 4  # D (ab/4), D = 1
        self.coef = 16 / pi**2  # lambda_mn^2 omega_mn = 16 / (pi^2 m n)
        self.lam11 = pi**2 * (1 / a**2 + 1 / b**2)
        self.T = (64 * a * b / pi**8 * (a**4 + b**4) * pi**2 / (80 * arb(M) ** 5)).sqrt()  # Q22, sec. 2 (S_TAIL)
        # |omega_M|^2 = c0 sum (16/(pi^2 m n))^2 / lambda^2 and Gamma(c)^2 = sum 1/(c0 lambda^2) + strict tail (Q22-6, item 1)
        ctx.prec = 320
        w2, g2 = arb(0), arb(0)
        nsq = [arb(n * n) for n in self.modes]
        for i, m in enumerate(self.modes):
            km, rw, rg = self.kap2[i], arb(0), arb(0)
            for j in range(self.K):
                lam = km + self.nu2[j]
                t = 1 / (lam * lam)
                rg += t
                rw += t / nsq[j]
            g2 += rg
            w2 += rw / (m * m)
        ctx.prec = prec
        self.normW2 = self.c0 * self.coef**2 * w2
        tail = (a * b) ** 2 / (4 * pi**4) * 2 * (1 / (2 * arb(M))) * (pi**2 / 8) / self.c0
        self.Gc = (g2 / self.c0 + tail).sqrt()

    def grams(self, F, w2):
        arb_mat = self.arb_mat
        n = F.ncols()
        F2 = arb_mat([[F[i, j] * w2[i] for j in range(n)] for i in range(self.K)])
        F4 = arb_mat([[F2[i, j] * w2[i] for j in range(n)] for i in range(self.K)])
        Ft = F.transpose()
        return Ft * F, Ft * F2, Ft * F4

    def norm2(self, GX, GY):
        """|X Y^T|^2_{H_T} = c0 sum_ij (Gx4 Gy0 + 2 Gx2 Gy2 + Gx0 Gy4)_ij (weights (kappa^2 + nu^2)^2)."""
        n = GX[0].nrows()
        s = self.arb(0)
        for i in range(n):
            for j in range(n):
                s += GX[2][i, j] * GY[0][i, j] + 2 * GX[1][i, j] * GY[1][i, j] + GX[0][i, j] * GY[2][i, j]
        return self.c0 * s

    def rebase(self, F):
        """Multiply the set by an exactly representable invertible matrix (approximate R^-1 of its QR): span unchanged."""
        n = F.ncols()
        Fm = np.array([[float(F[i, j].mid()) for j in range(n)] for i in range(self.K)])
        Ri = np.linalg.inv(np.linalg.qr(Fm)[1])
        Rm = self.arb_mat([[self.arb(float(v)) for v in row] for row in Ri])
        if Rm.det().contains(0):
            raise RuntimeError("basis change not certified invertible")
        return F * Rm

    def halfstep(self, G, w2given, w2new):
        N = G.ncols()
        G0, G2, G4 = self.grams(G, w2given)
        f = G.transpose() * self.inv_m
        rows = []
        for i, m in enumerate(self.modes):
            v = w2new[i]
            y = (G0 * (v * v) + G2 * (2 * v) + G4).solve(f) * (self.coef / m)
            rows.append([y[j, 0] for j in range(N)])
        return self.arb_mat(rows)


def _n_positive(A, t):
    """Number of positive eigenvalues of the symmetric ball matrix A - t I (Sylvester inertia of LDL^T without pivoting),
    or None if a pivot ball contains 0."""
    k = A.nrows()
    W = [[A[i, j] - (t if i == j else 0) for j in range(k)] for i in range(k)]
    pos = 0
    for p in range(k):
        d = W[p][p]
        if d.contains(0):
            return None
        if d > 0:
            pos += 1
        for i in range(p + 1, k):
            f = W[i][p] / d
            for j in range(p + 1, k):
                W[i][j] = W[i][j] - f * W[p][j]
    return pos


def _eig_guess(A, prec):
    import mpmath

    k = A.nrows()
    mp = mpmath.mp.clone() if hasattr(mpmath.mp, "clone") else mpmath.mp
    mp.prec = prec + 64
    B = mp.matrix(k, k)
    for i in range(k):
        for j in range(k):
            man, ex = A[i, j].mid().man_exp()
            B[i, j] = mp.mpf((int(man), int(ex)))
    B = (B + B.T) / 2
    return sorted((mp.mpf(e) for e in mp.eigsy(B, eigvals_only=True)), reverse=True), mp


def _to_arb(arb, v):
    man, exp = v.man_exp
    return arb(int(man)) * arb(2) ** int(exp) if exp >= 0 else arb(int(man)) / arb(2) ** int(-exp)


def eig_bounds(P, A, which):
    """Certified enclosure [t_lo, t_hi] of an eigenvalue of the symmetric ball matrix A (k x k): which = j (1-based, j-th
    largest). t_lo: at least j eigenvalues > t_lo; t_hi: at most j - 1 eigenvalues > t_hi (inertia)."""
    arb = P.arb
    k = A.nrows()
    ev, mp = _eig_guess(A, P.prec)
    lam = ev[which - 1]
    eps = mp.mpf(2) ** (-(P.prec // 2))
    t_lo = _to_arb(arb, mp.mpf(lam) * (1 - eps)) if lam > 0 else arb(0)
    t_hi = _to_arb(arb, mp.mpf(lam) * (1 + eps) + mp.mpf(2) ** (-P.prec) * abs(ev[0]))
    n_lo, n_hi = _n_positive(A, t_lo), _n_positive(A, t_hi)
    ok_lo = n_lo is not None and n_lo >= which
    ok_hi = n_hi is not None and n_hi <= which - 1
    return (t_lo if ok_lo else None), (t_hi if ok_hi else None), k


def _chol(P, A):
    n = A.nrows()
    L = [[P.arb(0)] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = A[i, j] - sum((L[i][k] * L[j][k] for k in range(j)), P.arb(0))
            L[i][j] = s.sqrt() if i == j else s / L[j][j]
    return P.arb_mat(L)


def sigma_N(P, GX0, GY0, N):
    """Enclosure of sigma_N^L2(X Y^T) = sqrt(c0 lambda_min(L^T Gy0 L)), Gx0 = L L^T."""
    L = _chol(P, GX0)
    S = L.transpose() * GY0 * L
    t_lo, t_hi, _ = eig_bounds(P, S, N)
    if t_lo is None or t_hi is None:
        raise RuntimeError("sigma_N not certified")
    return (P.c0 * t_lo).sqrt(), (P.c0 * t_hi).sqrt()


def arb_case(P, N, S):
    """Rows s = 1..S of the truncated sequence with enclosures and rigorous upper bounds of E'_s and U_s."""
    arb, arb_mat = P.arb, P.arb_mat
    K = P.K
    X = arb_mat([[1 if i == j else 0 for j in range(N)] for i in range(K)])  # odd sines 1, 3, ..., 2N - 1
    Y = None
    T_up = P.T.upper()
    lam11_lo = P.lam11.lower()
    rows, prev, E_up, E_mid, U_up, sig_lo_prev, sig_mid_prev = [], None, None, None, None, None, None
    for s in range(1, S + 1):
        if s % 2:  # Y-half-step: given x-set
            X = P.rebase(X)
            Y = P.halfstep(X, P.kap2, P.nu2)
        else:  # X-half-step: given y-set
            Y = P.rebase(Y)
            X = P.halfstep(Y, P.nu2, P.kap2)
        GX, GY = P.grams(X, P.kap2), P.grams(Y, P.nu2)
        nrm2 = P.norm2(GX, GY)
        fx, fy = X.transpose() * P.inv_m, Y.transpose() * P.inv_m
        cross = P.c0 * P.coef * sum((fx[j, 0] * fy[j, 0] for j in range(N)), arb(0))
        errM2 = P.normW2 - 2 * cross + nrm2
        errM = errM2.nonnegative_part().sqrt()
        sx, sy = X.transpose() * P.sgn, Y.transpose() * P.sgn
        wc = sum((sx[j, 0] * sy[j, 0] for j in range(N)), arb(0))
        sig_lo, sig_hi = sigma_N(P, GX[0], GY[0], N)
        R_up = (T_up**2 + errM2.upper()).sqrt().upper()
        if prev is None:
            d = arb(0)
            th_up = arb(0)
            E_up = T_up
            E_mid = P.T
        else:
            Xp, Yp = prev
            Z = arb_mat([[X[i, j] for j in range(N)] + [Xp[i, j] for j in range(N)] for i in range(K)])
            W = arb_mat([[Y[i, j] for j in range(N)] + [-Yp[i, j] for j in range(N)] for i in range(K)])
            d = P.norm2(P.grams(Z, P.kap2), P.grams(W, P.nu2)).nonnegative_part().sqrt()
            th = (E_up / (lam11_lo * sig_lo_prev)).upper()
            th_up = th if th < 1 else arb(1)
            E_up = ((T_up + th_up * R_up) ** 2 + (E_up + th_up * d.upper()) ** 2).sqrt().upper()
            thm = E_mid / (P.lam11 * sig_mid_prev)
            thm = thm if thm < 1 else arb(1)
            E_mid = ((P.T + thm * (P.T**2 + errM2).sqrt()) ** 2 + (E_mid + thm * d) ** 2).sqrt()
        cand = ((errM2.upper() + T_up**2).sqrt() + E_up).upper()
        U_up = cand if U_up is None or cand < U_up else U_up
        sig_mid = (sig_lo + sig_hi) / 2
        rows.append({"N": N, "s": s, "prec": P.prec, "T_up": up(P.T), "lambda11_lo": lo(P.lam11), "Gamma_c_up": up(P.Gc),
                     "normW_mid": mid(P.normW2.sqrt()),
                     "sigma_lo": lo(sig_lo), "sigma_hi": up(sig_hi), "d_lo": lo(d), "d_hi": up(d), "d_mid": mid(d),
                     "d_relrad": relrad(d) if s > 1 else 0.0,
                     "errM_lo": lo(errM), "errM_hi": up(errM), "errM_mid": mid(errM), "errM_relrad": relrad(errM),
                     "wc_mid": mid(wc), "wc_lo": lo(wc), "wc_hi": up(wc), "wc_relrad": relrad(wc),
                     "theta_up": up(th_up), "E_up": up(E_up), "E_mid": mid(E_mid), "U_up": up(U_up),
                     "sigma_relwidth": float((sig_hi - sig_lo).mid()) / float(sig_lo.mid())})
        prev, sig_lo_prev, sig_mid_prev = (X, Y), sig_lo.lower(), sig_mid
    return rows


def run_arb(args):
    cells = cells_of(args.cells)
    out_rows = []
    t0 = time.time()
    for lab in dict.fromkeys(lab for lab, _ in cells):
        P = ArbProblem(lab, args.M, args.prec)
        for N in sorted({N for l2, N in cells if l2 == lab}):
            for row in arb_case(P, N, args.S):
                out_rows.append({"r_label": lab, **row})
            print("arb", lab, N, round(time.time() - t0), flush=True)
    write_csv(Path(args.out) / ("hp_arb.csv" if args.M == M_MAIN else f"hp_arb_M{args.M}.csv"), out_rows)


# ------------------------------------------------------------------------------------------------ mpmath: separate path
def mp_case(rlab, N, M, S, dps, cache):
    import mpmath
    from mpmath import mp, mpf

    mp.dps = dps
    r = Fraction(rlab)
    a, b = mpf(1), mpf(r.numerator) / r.denominator
    pi = mp.pi
    modes = list(range(1, M + 1, 2))
    K = len(modes)
    kap2 = [(pi * m / a) ** 2 for m in modes]
    nu2 = [(pi * m / b) ** 2 for m in modes]
    inv_m = [mpf(1) / m for m in modes]
    sgn = [mpf(1) if i % 2 == 0 else mpf(-1) for i in range(K)]
    c0 = a * b / 4
    coef = 16 / pi**2
    lam11 = pi**2 * (1 / a**2 + 1 / b**2)
    T = mp.sqrt(64 * a * b / pi**8 * (a**4 + b**4) * pi**2 / (80 * mpf(M) ** 5))
    if rlab not in cache:
        w2 = []
        g2 = []
        for i, m in enumerate(modes):
            ts = [1 / (kap2[i] + nu2[j]) ** 2 for j in range(K)]
            g2.append(mp.fsum(ts))
            w2.append(mp.fsum(t / (modes[j] * modes[j]) for j, t in enumerate(ts)) / (m * m))
        normW2 = c0 * coef**2 * mp.fsum(w2)
        tail = (a * b) ** 2 / (4 * pi**4) * 2 * (1 / (2 * mpf(M))) * (pi**2 / 8) / c0
        cache[rlab] = (normW2, mp.sqrt(mp.fsum(g2) / c0 + tail))
    normW2, Gc = cache[rlab]

    def mgs(cols):
        Q, R = [], mp.zeros(len(cols), len(cols))
        for j, v in enumerate(cols):
            v = list(v)
            for i, q in enumerate(Q):
                c = mp.fdot(q, v)
                R[i, j] = c
                v = [vv - c * qq for vv, qq in zip(v, q)]
            nv = mp.sqrt(mp.fdot(v, v))
            R[j, j] = nv
            Q.append([vv / nv for vv in v])
        return Q, R

    def grams(cols, w2l):
        n = len(cols)
        c2 = [[c * w for c, w in zip(col, w2l)] for col in cols]
        c4 = [[c * w for c, w in zip(col, w2l)] for col in c2]
        return [[[mp.fdot(cols[i], src[j]) for j in range(n)] for i in range(n)] for src in (cols, c2, c4)]

    def norm2(GX, GY):
        n = len(GX[0])
        return c0 * mp.fsum(GX[2][i][j] * GY[0][i][j] + 2 * GX[1][i][j] * GY[1][i][j] + GX[0][i][j] * GY[2][i][j]
                            for i in range(n) for j in range(n))

    def solve(A, rhs):
        n = len(rhs)
        A = [list(row) + [rhs[i]] for i, row in enumerate(A)]
        for p in range(n):
            for i in range(p + 1, n):
                f = A[i][p] / A[p][p]
                A[i] = [x - f * y for x, y in zip(A[i], A[p])]
        x = [mpf(0)] * n
        for i in reversed(range(n)):
            x[i] = (A[i][n] - mp.fsum(A[i][k] * x[k] for k in range(i + 1, n))) / A[i][i]
        return x

    def halfstep(cols, w2g, w2n):
        G0, G2, G4 = grams(cols, w2g)
        f = [mp.fdot(c, inv_m) for c in cols]
        new = [[mpf(0)] * K for _ in range(N)]
        for i, m in enumerate(modes):
            v = w2n[i]
            A = [[v * v * G0[p][q] + 2 * v * G2[p][q] + G4[p][q] for q in range(N)] for p in range(N)]
            y = solve(A, f)
            for j in range(N):
                new[j][i] = y[j] * coef / m
        return new

    def recombine(cols, R):  # w = F G^T = Q (G R^T)^T for F = Q R: columns of G R^T
        return [[mp.fsum(R[i, j] * cols[j][k] for j in range(N)) for k in range(K)] for i in range(N)]

    X = [[mpf(1) if i == j else mpf(0) for i in range(K)] for j in range(N)]  # columns
    Y = None
    rows, prev, E, U, sig_prev = [], None, None, None, None
    for s in range(1, S + 1):
        # the increment d'_s = w'_s - w'_{s-1} in the common basis of the given set (coefficient differences, no cancellation
        # of norms): w'_{s-1} = Q (old R^T)^T
        if s % 2:
            X, Rg = mgs(X)
            old = recombine(Y, Rg) if Y is not None else None
            Y = halfstep(X, kap2, nu2)
            inc = (X, [[a - b for a, b in zip(cy, co)] for cy, co in zip(Y, old)]) if old is not None else None
        else:
            Y, Rg = mgs(Y)
            old = recombine(X, Rg)
            X = halfstep(Y, nu2, kap2)
            inc = ([[a - b for a, b in zip(cx, co)] for cx, co in zip(X, old)], Y)
        GX, GY = grams(X, kap2), grams(Y, nu2)
        cross = c0 * coef * mp.fsum(mp.fdot(X[j], inv_m) * mp.fdot(Y[j], inv_m) for j in range(N))
        errM2 = normW2 - 2 * cross + norm2(GX, GY)
        wc = mp.fsum(mp.fdot(X[j], sgn) * mp.fdot(Y[j], sgn) for j in range(N))
        Rx, Ry = mgs(X)[1], mgs(Y)[1]
        sig = mp.sqrt(c0) * min(mp.svd_r(Rx * Ry.T, compute_uv=False))
        Rs = mp.sqrt(T**2 + errM2)
        if prev is None:
            d, E = mpf(0), T
        else:
            d = mp.sqrt(norm2(grams(inc[0], kap2), grams(inc[1], nu2)))
            th = min(mpf(1), E / (lam11 * sig_prev))
            E = mp.sqrt((T + th * Rs) ** 2 + (E + th * d) ** 2)
        cand = mp.sqrt(errM2 + T**2) + E
        U = cand if U is None or cand < U else U
        rows.append({"N": N, "s": s, "dps": dps, "sigma": mpmath.nstr(sig, DIGITS), "d": mpmath.nstr(d, DIGITS),
                     "errM": mpmath.nstr(mp.sqrt(errM2), DIGITS), "wc": mpmath.nstr(wc, DIGITS),
                     "E": mpmath.nstr(E, DIGITS), "U": mpmath.nstr(U, DIGITS), "Gamma_c": mpmath.nstr(Gc, DIGITS),
                     "T": mpmath.nstr(T, DIGITS)})
        prev, sig_prev = (X, Y), sig
    return rows


def run_mpmath(args):
    cells = cells_of(args.cells)
    cache, out_rows = {}, []
    t0 = time.time()
    for lab, N in sorted(cells, key=lambda c: (R_LABELS.index(c[0]), c[1])):
        for row in mp_case(lab, N, args.M, args.S, args.dps, cache):
            out_rows.append({"r_label": lab, **row})
        print("mpmath", lab, N, round(time.time() - t0), flush=True)
    write_csv(Path(args.out) / ("hp_mpmath.csv" if args.M == M_MAIN else f"hp_mpmath_M{args.M}.csv"), out_rows)


# ------------------------------------------------------------------------------------------------ rank certificates (N23-4)
def _tail_lower(P, A, k):
    """Certified lower bound of sum_{j > k} sigma_j(A)^2 = tr(A^T A) - sum_{j <= k} lambda_j (upper bounds by inertia)."""
    Sm = A.transpose() * A
    tot = Sm.trace()
    for j in range(1, k + 1):
        _, t_hi, _ = eig_bounds(P, Sm, j)
        if t_hi is None:
            return None
        tot -= t_hi
    return tot


def _d_lower(P, Om, alpha, beta, k):
    """Lemma N23-3 on the submatrix: c0-free lower bound of d_k^2."""
    arb_mat = P.arb_mat
    n = len(alpha)
    sa, sb = [x.sqrt() for x in alpha], [x.sqrt() for x in beta]
    A1 = arb_mat([[alpha[i] * Om[i][j] for j in range(n)] for i in range(n)])
    A2 = arb_mat([[sa[i] * Om[i][j] * sb[j] for j in range(n)] for i in range(n)])
    A3 = arb_mat([[Om[i][j] * beta[j] for j in range(n)] for i in range(n)])
    parts = [_tail_lower(P, A, k) for A in (A1, A2, A3)]
    if any(p is None for p in parts):
        return None
    return parts[0] + 2 * parts[1] + parts[2]


class _Bare:
    """Minimal ball context for the rank checks (no truncated-sequence sums)."""

    def __init__(self, prec):
        from flint import arb, arb_mat, ctx

        ctx.prec = prec
        self.arb, self.arb_mat, self.prec = arb, arb_mat, prec


def rank_continuous(rlab, N, M, prec=256, untruncated=False, ksub=12):
    """mu and d_N^low/err2^up for the truncated problem (б) at M (finite sums, no tails) or, with untruncated=True, for (а):
    the same partial sums plus the strict row and column tails of N23-2."""
    P = _Bare(prec)
    arb = P.arb
    r = Fraction(rlab)
    a, b = arb(1), arb(r.numerator) / r.denominator
    pi = arb.pi()
    modes = list(range(1, M + 1, 2))
    kap2 = [(pi * m / a) ** 2 for m in modes]
    nu2 = [(pi * m / b) ** 2 for m in modes]
    coef = 16 / pi**2

    def om(i, j):
        lam = kap2[i] + nu2[j]
        return coef / (modes[i] * modes[j] * lam * lam)

    ks = min(ksub, len(modes))
    Om = [[om(i, j) for j in range(ks)] for i in range(ks)]
    dN1 = _d_lower(P, Om, kap2[:ks], nu2[:ks], N - 1)
    dN = _d_lower(P, Om, kap2[:ks], nu2[:ks], N)
    # E2: rows m not in L, weighted LS trial vector in double, residual in balls
    rf = float(r)
    mf = np.array(modes, dtype=float)
    lamf = np.pi**2 * (mf[:, None] ** 2 + mf[None, :] ** 2 / rf**2)
    omf = 16.0 / (np.pi**2 * mf[:, None] * mf[None, :] * lamf**2)
    OmL = [[om(j, n) for n in range(len(modes))] for j in range(N)]
    E2 = arb(0)
    for i in range(N, len(modes)):
        A = (omf[:N, :] * lamf[i, :]).T
        t = np.linalg.lstsq(A, omf[i, :] * lamf[i, :], rcond=None)[0]
        tv = [arb(float(x)) for x in t]
        acc = arb(0)
        for n in range(len(modes)):
            lam = kap2[i] + nu2[n]
            dd = coef / (modes[i] * modes[n] * lam * lam) - sum((tv[j] * OmL[j][n] for j in range(N)), arb(0))
            acc += (lam * dd) ** 2
        if untruncated:  # strict column tail n > M of N23-2
            rho = 1 + arb(modes[i]) ** 2 * b**2 / (a**2 * arb(M) ** 2)
            ws = 1 / arb(modes[i]) + sum((abs(tv[j]) / (2 * j + 1) for j in range(N)), arb(0))
            acc += 256 * rho**2 * b**4 / pi**8 * ws**2 / (10 * arb(M) ** 5)
        E2 += acc
    if untruncated:  # strict row tail m > M
        E2 += 32 * a**4 / pi**6 / (10 * arb(M) ** 5)
    return _rank_result(dN1, dN, E2)


def rank_discrete(rlab, nx, ny, N, prec=256, ksub=12):
    P = _Bare(prec)
    arb = P.arb
    r = Fraction(rlab)
    a, b = arb(1), arb(r.numerator) / r.denominator
    pi = arb.pi()

    def axis(n, h):
        out = []
        for m in range(1, n, 2):
            th = arb(m) * pi / (2 * n)
            out.append((4 / h**2 * th.sin() ** 2, (arb(2) / n).sqrt() * th.cot()))
        return out

    ax, ay = axis(nx, a / nx), axis(ny, b / ny)
    mux, cx = [p[0] for p in ax], [p[1] for p in ax]
    muy, cy = [p[0] for p in ay], [p[1] for p in ay]

    def om(i, j):
        return cx[i] * cy[j] / (mux[i] + muy[j]) ** 2

    ks = min(ksub, len(mux), len(muy))
    Om = [[om(i, j) for j in range(ks)] for i in range(ks)]
    dN1 = _d_lower(P, Om, mux[:ks], muy[:ks], N - 1)
    dN = _d_lower(P, Om, mux[:ks], muy[:ks], N)
    muxf = np.array([float(x.mid()) for x in mux])
    muyf = np.array([float(x.mid()) for x in muy])
    cxf = np.array([float(x.mid()) for x in cx])
    cyf = np.array([float(x.mid()) for x in cy])
    Wf = muxf[:, None] + muyf[None, :]
    Omf = cxf[:, None] * cyf[None, :] / Wf**2
    OmL = [[om(j, n) for n in range(len(muy))] for j in range(N)]
    E2 = arb(0)
    for i in range(N, len(mux)):
        A = (Omf[:N, :] * Wf[i, :]).T
        t = np.linalg.lstsq(A, Omf[i, :] * Wf[i, :], rcond=None)[0]
        tv = [arb(float(x)) for x in t]
        acc = arb(0)
        for n in range(len(muy)):
            dd = om(i, n) - sum((tv[j] * OmL[j][n] for j in range(N)), arb(0))
            acc += ((mux[i] + muy[n]) * dd) ** 2
        E2 += acc
    return _rank_result(dN1, dN, E2)


def _rank_result(dN1, dN, E2):
    if dN1 is None or dN is None or not dN1 > 0:
        return {"certified": False, "mu_up": "", "enc_lo": ""}
    mu = (E2.upper() / dN1.lower()).sqrt()
    enc = (dN.lower() / E2.upper()).nonnegative_part().sqrt()
    return {"certified": bool(mu < 1), "mu_up": up(mu, 8), "enc_lo": lo(enc, 8), "E2_up": up(E2, 12),
            "dN1_lo": lo(dN1, 12), "dN_lo": lo(dN, 12)}


def _author_module():
    path = ROOT / "scripts" / "r4_n23" / "n23_rank_certificate.py"
    spec = importlib.util.spec_from_file_location("n23_rank_certificate", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _printed(path):
    import re

    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.search(r"\| mu<=([0-9.]+) .*d\(N\)_low/err2_up>=([0-9.]+)", line)
        if m:
            out[line.split("|")[0].strip()] = (m.group(1), m.group(2))
    return out


def run_rank(args):
    rows = []
    t0 = time.time()
    plan = [("continuous (a)", lab, N, None) for lab, N in (("1/5", 3), ("5", 3), ("1", 2))]
    for kind, lab, N, _ in plan:
        res = rank_continuous(lab, N, args.M, untruncated=True)
        rows.append({"problem": kind, "r": lab, "N": N, "grid_or_M": f"M_r = M_c = {args.M} + strict tails", **res})
        print("rank", kind, lab, N, round(time.time() - t0), flush=True)
        for Mt in (21, 23, 25, args.M):
            res = rank_continuous(lab, N, Mt)
            rows.append({"problem": "truncated (b)", "r": lab, "N": N, "grid_or_M": f"M = {Mt}", **res})
            print("rank (b)", lab, N, Mt, round(time.time() - t0), flush=True)
    for lab, nx, ny in (("1/5", 1280, 1280), ("1/5", 1600, 320), ("5", 320, 1600)):
        for N in NS:
            res = rank_discrete(lab, nx, ny, N)
            rows.append({"problem": "discrete (c)", "r": lab, "N": N, "grid_or_M": f"{nx} x {ny}", **res})
            print("rank (c)", lab, nx, ny, N, round(time.time() - t0), flush=True)
    write_csv(Path(args.out) / "rank_checks.csv", rows)
    # directions of the printed bounds of the author's outputs (round-to-nearest printing, review N-1)
    mod = _author_module()
    from mpmath import mp

    exp = ROOT / "scripts" / "r4_n23" / "expected"
    printed = {**_printed(exp / "certificate_continuous.txt"), **_printed(exp / "certificate_discrete.txt")}
    prow = []
    cases = [("r=%-4s N=%d" % (r, N), lambda r=r, N=N: mod.continuous_case(r, N)) for r in mod.R_CONTROL for N in NS]
    grids = [(n, n, r) for n in (40, 80, 160, 320, 640, 1280) for r in mod.R_CONTROL]
    grids += [(1600, 320, Fraction(1, 5)), (960, 320, Fraction(1, 3)), (640, 320, Fraction(1, 2)),
              (320, 640, Fraction(2)), (320, 960, Fraction(3)), (320, 1600, Fraction(5))]
    cases += [("nx=%-4d ny=%-4d r=%-4s N=%d" % (nx, ny, r, N), lambda nx=nx, ny=ny, r=r, N=N: mod.discrete_case(nx, ny, r, N))
              for nx, ny, r in grids for N in NS]
    for label, fn in cases:
        res = fn()
        mu_v = mp.sqrt(mod.hi(res["E2"]) / mod.lo(res["dN1"]))
        enc_v = mp.sqrt(mod.lo(res["dN"]) / mod.hi(res["E2"]))
        p_mu, p_enc = printed[label]
        prow.append({"label": label, "mu_author": mp.nstr(mu_v, 12), "mu_printed": p_mu,
                     "mu_printed_is_upper_bound": bool(mp.mpf(p_mu) >= mu_v),
                     "enc_author": mp.nstr(enc_v, 12), "enc_printed": p_enc,
                     "enc_printed_is_lower_bound": bool(mp.mpf(p_enc) <= enc_v)})
    write_csv(Path(args.out) / "printed_bounds.csv", prow)
    print("printed bounds checked", len(prow), round(time.time() - t0), flush=True)


# ------------------------------------------------------------------------------------------------ combine
def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = []
    for row in rows:
        for k in row:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
        w.writeheader()
        for row in rows:
            w.writerow({k: (repr(v) if isinstance(v, float) else v) for k, v in row.items()})


def fup(x):
    """float >= the decimal x (x a decimal string of an upper bound)."""
    v = float(x)
    return v if Decimal(v) >= Decimal(str(x)) else math.nextafter(v, math.inf)


def fdn(x):
    v = float(x)
    return v if Decimal(v) <= Decimal(str(x)) else math.nextafter(v, -math.inf)


def _key(row):
    return row["r_label"], int(row["N"]), int(row["s"])


def run_combine(args):
    import decimal

    decimal.getcontext().prec = 120  # comparisons of the 45-digit control values (the default 28 digits would round them)
    out = Path(args.out)
    gb = {_key(r): r for r in read_csv(out / "group_b_transfer.csv")}
    ab = {_key(r): r for r in read_csv(out / "hp_arb.csv")}
    mpv = {_key(r): r for r in read_csv(out / "hp_mpmath.csv")}
    cont = {(r["r_label"], int(r["N"]), int(r["s"])): r for r in read_csv(ROOT / "docs" / "results" / "r4_group_b" / "continuous.csv")}
    rows, agree = [], []
    for key in sorted(gb, key=lambda k: (k[1], R_LABELS.index(k[0]), k[2])):
        g, A, m, c = gb[key], ab[key], mpv[key], cont[key]
        lab, N, s = key
        f = {k: float(v) for k, v in g.items() if k not in ("r_label",) and not k.startswith("rule")}
        # regression: the same data as group B (continuous.csv, 7 significant digits)
        same = all(format(f[k], ".6e") == c[k] for k in ("err2_direct", "err_abs", "e_rel", "delta_c", "w_cont_centre"))
        # high precision: arb enclosure vs mpmath (>= 40 digits), double vs arb (budget)
        errM_lo, errM_hi = Decimal(A["errM_lo"]), Decimal(A["errM_hi"])
        dig = {}
        for q_arb, q_mp, q_rel in (("errM_mid", "errM", "errM_relrad"), ("wc_mid", "wc", "wc_relrad"), ("d_mid", "d", "d_relrad")):
            va, vm = Decimal(A[q_arb]), Decimal(m[q_mp])
            dig[q_arb] = float(abs(va - vm) / abs(va)) if va != 0 else 0.0
            dig[q_rel] = float(A[q_rel])
        sig_lo, sig_hi = Decimal(A["sigma_lo"]), Decimal(A["sigma_hi"])
        dig["sigma_mp_in_ball"] = bool(sig_lo * (1 - Decimal("1e-40")) <= Decimal(m["sigma"]) <= sig_hi * (1 + Decimal("1e-40")))
        dig["E_mid_vs_mp"] = float(abs(Decimal(A["E_mid"]) - Decimal(m["E"])) / Decimal(m["E"]))
        dig["sigma_relwidth"] = float(A["sigma_relwidth"])
        E_dbl, E_up = f["E_transfer"], fup(A["E_up"])
        B_tr = max(E_dbl, E_up)
        rho = E_up - E_dbl
        # recursion rounding of the measured quantities: double (group B arrays) against the arb enclosure
        b4 = fup(max(abs(f["err_M"] - fdn(errM_lo)), abs(fup(errM_hi) - f["err_M"])))
        b4c = fup(max(abs(f["w_cont_centre"] - fdn(A["wc_lo"])), abs(fup(A["wc_hi"]) - f["w_cont_centre"])))
        gc_up = fup(A["Gamma_c_up"])
        err = f["err_abs"]
        be, bD, bP, bc, bi = f["beta_err"], f["beta_D"], f["beta_Pi0"], f["beta_c"], f["beta_inf"]
        qq = (B_tr + b4) / (err * (1 - be))
        dcabs = abs(f["delta_c"]) * abs(f["w_cont_centre"] / (1 + f["delta_c"]))  # |w'(c) - omega_0(c)|
        R = {"err_abs": be + qq, "e_rel": bD + bP + 2 * qq + qq * qq,
             "delta_c": bc + (gc_up * B_tr + b4c) / (dcabs * (1 - bc)),
             "E_inf": bi + gc_up * B_tr / (f["E_inf_abs"] * (1 - bi))}
        U_up = max(fup(A["U_up"]), f["U_upper"])
        RU = {"err_abs": (U_up - err) / err + be, "e_rel": (U_up * U_up - err * err) / (err * err) + bD + bP}
        row = {"r_label": lab, "N": N, "s": s, "same_data_as_group_B": same,
               "err_abs": err, "e_rel": f["e_rel"], "delta_c": f["delta_c"], "E_inf_abs": f["E_inf_abs"],
               "err_M_arb_lo": A["errM_lo"], "err_M_arb_hi": A["errM_hi"],
               "sigma_N_L2_arb_lo": A["sigma_lo"], "sigma_N_L2_dbl": f["sigma_N_L2"], "d_dbl": f["d_norm"],
               "E_dbl": E_dbl, "E_arb_up": A["E_up"], "rounding_of_inputs": rho, "B_tr": B_tr,
               "recursion_rounding_err": b4, "recursion_rounding_centre": b4c, "Gamma_c_up": A["Gamma_c_up"],
               "U_dbl": f["U_upper"], "U_arb_up": A["U_up"],
               "beta_err": be, "beta_D": bD, "beta_Pi0": bP, "beta_c": bc, "beta_inf": bi}
        for k in MEASURES:
            row[f"R_{k}"] = R[k]
            row[f"rule_{k}"] = "PASS" if R[k] <= RULE else "UNRESOLVED"
            row[f"reason_{k}"] = "" if R[k] <= RULE else ("R-MEAS" if {"err_abs": be, "e_rel": bD + bP, "delta_c": bc,
                                                                            "E_inf": bi}[k] > RULE else "R-TR")
            row[f"dbl_agrees_{k}"] = (g[f"rule_{k}"] == "True") == (R[k] <= RULE)
        row["reason_E_inf"] = row["reason_E_inf"] + ("; recursion rounding of the field - group B premise"
                                                     if row["rule_E_inf"] == "PASS" else "")
        for k in ("err_abs", "e_rel"):
            row[f"RU_{k}"] = RU[k]
            row[f"upper_{k}"] = "PASS" if RU[k] <= RULE else "UNRESOLVED"
        row["L2"] = "UNRESOLVED (R-BUD: no measurement budget of L2 in group B)"
        row.update({f"hp_{k}": v for k, v in dig.items()})
        rows.append(row)
        agree.append(dig)
    write_csv(out / "matrix.csv", rows)
    # s_max (prefix rule of Q22 table 7.3) and isolated passes
    sm = []
    for N in NS:
        for lab in R_LABELS:
            cell = [r for r in rows if r["N"] == N and r["r_label"] == lab]
            entry = {"r_label": lab, "N": N}
            for k in MEASURES:
                last, isolated, alive = 0, [], True
                for r in cell:
                    ok = r[f"rule_{k}"] == "PASS"
                    if alive and ok:
                        last = r["s"]
                    elif ok:
                        isolated.append(r["s"])
                    else:
                        alive = False
                entry[f"smax_{k}"] = last
                entry[f"isolated_{k}"] = " ".join(map(str, isolated))
                bound = [r for r in cell if r["s"] in (last, last + 1)]
                entry[f"R_{k}_at_smax"] = bound[0][f"R_{k}"] if bound else ""
                entry[f"R_{k}_first_fail"] = bound[1][f"R_{k}"] if len(bound) > 1 else ""
            for k in ("err_abs", "e_rel"):
                ok = [r for r in cell if r["s"] >= 2 and r[f"upper_{k}"] == "PASS"]
                entry[f"upper_{k}_s2_40"] = "all" if len(ok) == len(cell) - 1 else " ".join(str(r["s"]) for r in ok)
                entry[f"max_RU_{k}"] = max(r[f"RU_{k}"] for r in cell if r["s"] >= 2)
            sm.append(entry)
    write_csv(out / "smax.csv", sm)
    # boundary cells (plan, sec. 2): last confirmed and first unconfirmed half-step of every (r, N, measure), with the
    # high-precision inputs of E'_s (sigma_N of w'_{s-1}, |d'_s|, err_M(s)) from both paths
    bnd = []
    for e in sm:
        lab, N = e["r_label"], e["N"]
        for k in MEASURES:
            for role, s in (("last_confirmed", e[f"smax_{k}"]), ("first_unconfirmed", e[f"smax_{k}"] + 1)):
                if not 1 <= s <= S_MAIN:
                    continue
                A, m = ab[(lab, N, s)], mpv[(lab, N, s)]
                Ap, mpp = (ab[(lab, N, s - 1)], mpv[(lab, N, s - 1)]) if s > 1 else ({}, {})
                rr = [r for r in rows if r["r_label"] == lab and r["N"] == N and r["s"] == s][0]
                bnd.append({"r_label": lab, "N": N, "measure": k, "role": role, "s": s, "R": rr[f"R_{k}"],
                            "rule": rr[f"rule_{k}"], "E_arb_up": A["E_up"], "E_mpmath": m["E"], "E_dbl": rr["E_dbl"],
                            "sigma_prev_arb_lo": Ap.get("sigma_lo", ""), "sigma_prev_mpmath": mpp.get("sigma", ""),
                            "d_arb_mid": A["d_mid"], "d_mpmath": m["d"], "errM_arb_mid": A["errM_mid"],
                            "errM_mpmath": m["errM"], "T_up": A["T_up"], "Gamma_c_up": A["Gamma_c_up"]})
    write_csv(out / "boundary_cells.csv", bnd)
    summary = {
        "cells": len(rows), "same_data_as_group_B": sum(r["same_data_as_group_B"] for r in rows),
        "max_relrad_errM": max(a["errM_relrad"] for a in agree), "max_relrad_wc": max(a["wc_relrad"] for a in agree),
        "max_relrad_d": max(a["d_relrad"] for a in agree), "max_sigma_relwidth": max(a["sigma_relwidth"] for a in agree),
        "max_mp_vs_arb_errM": max(a["errM_mid"] for a in agree), "max_mp_vs_arb_wc": max(a["wc_mid"] for a in agree),
        "max_mp_vs_arb_d": max(a["d_mid"] for a in agree), "sigma_mp_in_ball": all(a["sigma_mp_in_ball"] for a in agree),
        "max_E_mid_vs_mp": max(a["E_mid_vs_mp"] for a in agree),
        "max_rounding_of_inputs_rel": max(abs(r["rounding_of_inputs"]) / r["E_dbl"] for r in rows),
        "pass_counts": {k: sum(r[f"rule_{k}"] == "PASS" for r in rows) for k in MEASURES},
        "upper_pass_counts": {k: sum(r[f"upper_{k}"] == "PASS" for r in rows if r["s"] >= 2) for k in ("err_abs", "e_rel")},
        "dbl_vs_rigorous_disagreements": {k: [f"{r['r_label']},N={r['N']},s={r['s']}" for r in rows
                                              if not r[f"dbl_agrees_{k}"]] for k in MEASURES},
        "closest_to_rule": {k: sorted(((abs(r[f"R_{k}"] - RULE) / RULE, r["r_label"], r["N"], r["s"]) for r in rows))[:3]
                            for k in MEASURES},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1, ensure_ascii=False))


# ------------------------------------------------------------------------------------------------ main
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("mode", choices=("arb", "mpmath", "rank", "combine", "meta"))
    p.add_argument("--out", default=str(OUT))
    p.add_argument("--M", type=int, default=M_MAIN)
    p.add_argument("--S", type=int, default=S_MAIN)
    p.add_argument("--cells", default="")
    p.add_argument("--prec", type=int, default=1024)
    p.add_argument("--dps", type=int, default=80)
    args = p.parse_args(argv)
    t0 = time.time()
    {"arb": run_arb, "mpmath": run_mpmath, "rank": run_rank, "combine": run_combine, "meta": run_meta}[args.mode](args)
    print("done", args.mode, round(time.time() - t0), "s", flush=True)


def run_meta(args):
    import subprocess

    versions = {"python": sys.version.split()[0], "numpy": np.__version__, "platform": platform.platform()}
    for mod in ("flint", "mpmath"):
        try:
            versions[mod] = __import__(mod).__version__
        except ImportError:
            versions[mod] = None
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    meta = {"head_at_meta": head, "base_main": "f0c0d989defd37cbfe3b0929bc1c348a0cc2fd10",
            "literature": "206c1351fc1424951b2fd65df756b40a651e7308", "review": "cee195134d30ec90e50f91c6e9a0a5a7dca64aeb",
            "versions": versions, "M": M_MAIN, "modes_per_axis": (M_MAIN + 1) // 2, "S": S_MAIN, "r": R_LABELS, "N": NS,
            "units": "a = 1, b = r, q = D = 1", "start": "odd sines in x (RV4-1)", "rule": RULE,
            "arb_prec_bits": args.prec, "mpmath_dps": args.dps, "printed_digits": DIGITS,
            "data": {"group_b_transfer.csv": "scripts/r4_group_b_study.py --n23-transfer (double, arrays of group B)",
                     "hp_arb.csv": "this script, mode arb", "hp_mpmath.csv": "this script, mode mpmath",
                     "rank_checks.csv, printed_bounds.csv": "this script, mode rank",
                     "matrix.csv, smax.csv, summary.json": "this script, mode combine",
                     "group B measurement budgets": "docs/results/r4_group_b/continuous.csv (unchanged) and the same formulas"},
            "rank_check_cells": {"continuous": ["1/5:3", "5:3", "1:2"], "truncated_M": [21, 23, 25, M_MAIN],
                                 "discrete": ["1/5: 1280 x 1280", "1/5: 1600 x 320", "5: 320 x 1600"]}}
    (Path(args.out) / "meta.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
