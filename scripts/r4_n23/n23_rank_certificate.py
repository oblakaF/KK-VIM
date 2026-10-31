"""R4, N = 2-3: interval certificate of the sufficient rank condition VB-20 (MATH_SPEC B.14) at s0 = 2.

Research script of docs/R4_N23_SCIENTIFIC_CLOSURE.md (Literature). Not part of the solver and not a reference of the project:
the packages kkvim and kkvim_reference are NOT imported. Everything is computed from closed-form target coefficients.

Problem (sec. 2 of the document): weighted low-rank alternating projection in coefficient space,
    ||V||^2 = c0 * sum_mn (alpha_m + beta_n)^2 V_mn^2,   target Omega_mn,
  continuous SSSS (sine basis):  alpha = kappa_m^2 = (m pi/a)^2, beta = nu_n^2 = (n pi/b)^2, c0 = D ab/4,
                                 Omega_mn = 16 q / (pi^2 D m n lambda_mn^2),  lambda_mn = alpha_m + beta_n  (V-27);
  discrete SSSS (DST-I):         alpha = mu^x_m = (4/h_x^2) sin^2(m pi/(2 n_x)), beta = mu^y_n, c0 = D h_x h_y,
                                 Omega_mn = q c^x_m c^y_n / (D (alpha_m + beta_n)^2),  c_m = sqrt(2/n) cot(m pi/(2n)) (odd m).
Odd-sine start X^0 = (s_1, s_3, ..., s_{2N-1}).

Certified quantities (interval arithmetic, mpmath.iv; the rounding of every operation is enclosed):
  E2_up >= err(N, 2)^2 / c0   (Lemma N23-2: the second half-step is a row-wise weighted LS onto the rows m in L; any trial vector
                                gives an upper bound; strict analytic tails for the continuous problem),
  D_low(k) <= d_k^2 / c0      (Lemma N23-3: (alpha+beta)^2 = alpha^2 + 2 alpha beta + beta^2, Mirsky tails of three product-weighted
                                matrices, submatrix interlacing, eigenvalue lower bounds by Sylvester inertia of an interval LDL^T).
Certificate of VB-20 at s0 = 2:  mu := sqrt(E2_up / D_low(N-1)) < 1  =>  rank w^(s) = N for all s >= 1 (Theorem N23-4).
Units a = 1, b = r, q = D = 1 (all printed ratios are scale-free).

Usage:  python n23_rank_certificate.py continuous      (ten control r, N = 2, 3)
        python n23_rank_certificate.py discrete        (D-048 grids and family-I n = 80, 320; N = 2, 3)
        python n23_rank_certificate.py dense           (201 log-uniform r in [1/5, 5]: point certificates; NOT a proof on the
                                                        interval, see sec. 3.5 of the document)
"""
import sys
from fractions import Fraction

import numpy as np
from mpmath import iv, mp

iv.prec = 256
mp.prec = 256

R_CONTROL = [Fraction(1, 5), Fraction(1, 3), Fraction(1, 2), Fraction(2, 3), Fraction(1), Fraction(7, 5), Fraction(3, 2),
             Fraction(2), Fraction(3), Fraction(5)]
K_SUB = 12          # odd modes per axis in the submatrix of the lower bound (Lemma N23-3)
M_ROWS = 161        # continuous: rows m <= M_ROWS evaluated with a trial vector; the rest by the strict row tail
M_COLS = 161        # continuous: columns n <= M_COLS in interval arithmetic; the rest by the strict column tail
M_ROWS_DISCRETE = 161


def ivq(fr):
    return iv.mpf(fr.numerator) / fr.denominator


def lo(x):
    return mp.mpf(x.a)


def hi(x):
    return mp.mpf(x.b)


# ---------------------------------------------------------------- certified eigenvalue lower bounds (Sylvester inertia)
def n_positive(S, t):
    """Number of positive eigenvalues of the symmetric interval matrix S - t I, or None if a pivot interval contains 0.

    Gaussian elimination without pivoting: for every symmetric matrix in the box the pivots lie in the computed intervals,
    and by Sylvester's law of inertia the signs of the pivots are the signs of the eigenvalues."""
    k = len(S)
    A = [[S[i][j] - (t if i == j else 0) for j in range(k)] for i in range(k)]
    pos = 0
    for p in range(k):
        d = A[p][p]
        if d.a <= 0 <= d.b:
            return None
        if d.a > 0:
            pos += 1
        for i in range(p + 1, k):
            f = A[i][p] / d
            for j in range(p + 1, i + 1):
                A[i][j] = A[i][j] - f * A[j][p]
                A[j][i] = A[i][j]
    return pos


def tail_lower(Amat, k):
    """Certified lower bound of sum_{j > k} sigma_j(A)^2 for the interval matrix A (list of rows); j is 1-based."""
    K = len(Amat)
    S = [[sum((Amat[l][i] * Amat[l][j] for l in range(K)), iv.mpf(0)) for j in range(K)] for i in range(K)]
    mid = mp.matrix(K, K)
    for i in range(K):
        for j in range(K):
            mid[i, j] = (lo(S[i][j]) + hi(S[i][j])) / 2
    ev = sorted([mp.mpf(e) for e in mp.eigsy(mid, eigvals_only=True)], reverse=True)
    total = iv.mpf(0)
    certified = 0
    for j in range(k + 1, K + 1):           # eigenvalue lambda_j (j-th largest)
        lam = ev[j - 1]
        if lam <= 0:
            break
        t = iv.mpf(lam * (1 - mp.mpf(2) ** -40))
        cnt = n_positive(S, t)
        if cnt is None or cnt < j:
            break                           # stop at the first eigenvalue that is not certified; the bound stays valid
        total += t
        certified = j
    return total, certified


def d_lower(Om, alpha, beta, k):
    """Certified lower bound of inf_{rank W <= k} sum (alpha_m + beta_n)^2 (Om - W)_mn^2 over the submatrix (without c0)."""
    K = len(Om)
    sa = [iv.sqrt(x) for x in alpha]
    sb = [iv.sqrt(x) for x in beta]
    A1 = [[alpha[i] * Om[i][j] for j in range(K)] for i in range(K)]
    A2 = [[sa[i] * Om[i][j] * sb[j] for j in range(K)] for i in range(K)]
    A3 = [[Om[i][j] * beta[j] for j in range(K)] for i in range(K)]
    t1, c1 = tail_lower(A1, k)
    t2, c2 = tail_lower(A2, k)
    t3, c3 = tail_lower(A3, k)
    return t1 + 2 * t2 + t3, (c1, c2, c3)


# ---------------------------------------------------------------- continuous (untruncated) SSSS problem
def continuous_case(r, N):
    a, b = iv.mpf(1), ivq(r)
    pi = iv.pi
    L = [2 * j - 1 for j in range(1, N + 1)]

    def lam(m, n):
        return pi ** 2 * (iv.mpf(m) ** 2 / a ** 2 + iv.mpf(n) ** 2 / b ** 2)

    def coef(m, n):
        return 16 / (pi ** 2 * m * n * lam(m, n) ** 2)

    modes = [2 * i - 1 for i in range(1, K_SUB + 1)]
    Om = [[coef(m, n) for n in modes] for m in modes]
    alpha = [pi ** 2 * iv.mpf(m) ** 2 / a ** 2 for m in modes]
    beta = [pi ** 2 * iv.mpf(n) ** 2 / b ** 2 for n in modes]
    dN1, cert1 = d_lower(Om, alpha, beta, N - 1)
    dN, cert2 = d_lower(Om, alpha, beta, N)
    rf = float(r)
    nn = np.arange(1, M_COLS + 1, 2, dtype=float)

    def coef_f(m):
        lamf = np.pi ** 2 * (m ** 2 + nn ** 2 / rf ** 2)
        return 16.0 / (np.pi ** 2 * m * nn * lamf ** 2), lamf

    Yf = np.array([coef_f(mj)[0] for mj in L])
    ncols = [iv.mpf(int(n)) for n in nn]
    Ycols = [[coef(mj, n) for n in ncols] for mj in L]
    E2 = iv.mpf(0)
    E1 = iv.mpf(0)
    for m in range(2 * N + 1, M_ROWS + 1, 2):
        cm, lamf = coef_f(m)
        A = (Yf * lamf).T
        t = np.linalg.lstsq(A, cm * lamf, rcond=None)[0]
        tiv = [iv.mpf(float(x)) for x in t]
        s = iv.mpf(0)
        s1 = iv.mpf(0)
        for idx, n in enumerate(ncols):
            l = lam(m, n)
            c = coef(m, n)
            d = c - sum((tiv[j] * Ycols[j][idx] for j in range(N)), iv.mpf(0))
            s += (l * d) ** 2
            s1 += (l * c) ** 2
        # strict column tail n > M_COLS (sec. 3.2 of the document)
        rho = 1 + iv.mpf(m) ** 2 * b ** 2 / (a ** 2 * iv.mpf(M_COLS) ** 2)
        wsum = 1 / iv.mpf(m) + sum((abs(tiv[j]) / L[j] for j in range(N)), iv.mpf(0))
        tail_c = 256 * rho ** 2 * b ** 4 / pi ** 8 * wsum ** 2 / (10 * iv.mpf(M_COLS) ** 5)
        tail_c1 = 256 * rho ** 2 * b ** 4 / pi ** 8 * (1 / iv.mpf(m)) ** 2 / (10 * iv.mpf(M_COLS) ** 5)
        E2 += s + tail_c
        E1 += s1 + tail_c1
    tail_rows = 32 * a ** 4 / pi ** 6 / (10 * iv.mpf(M_ROWS) ** 5)        # rows m > M_ROWS (row-energy bound, sec. 3.2)
    E2 += tail_rows
    E1 += tail_rows
    W0 = iv.mpf(0)                                                       # ||omega_0||^2 / c0 from below (display only)
    for m in range(1, M_COLS + 1, 2):
        for n in ncols:
            W0 += (lam(m, n) * coef(m, n)) ** 2
    return dict(c0=a * b / 4, dN1=dN1, dN=dN, E2=E2, E1=E1, W0=W0, cert=(cert1, cert2))


# ---------------------------------------------------------------- discrete SSSS problem (D-048 grids)
def discrete_case(nx, ny, r, N):
    a, b = iv.mpf(1), ivq(r)
    hx, hy = a / nx, b / ny
    pi = iv.pi

    def mu_c(m, n, h):
        th = iv.mpf(m) * pi / (2 * n)
        return 4 / h ** 2 * iv.sin(th) ** 2, iv.sqrt(iv.mpf(2) / n) * iv.cot(th)

    mx = [mu_c(m, nx, hx) for m in range(1, nx, 2)]           # odd m < nx
    my = [mu_c(n, ny, hy) for n in range(1, ny, 2)]
    mux, cx = [p[0] for p in mx], [p[1] for p in mx]
    muy, cy = [p[0] for p in my], [p[1] for p in my]

    def om(i, j):
        return cx[i] * cy[j] / (mux[i] + muy[j]) ** 2

    K = min(K_SUB, len(mux), len(muy))
    Om = [[om(i, j) for j in range(K)] for i in range(K)]
    dN1, cert1 = d_lower(Om, mux[:K], muy[:K], N - 1)
    dN, cert2 = d_lower(Om, mux[:K], muy[:K], N)
    mux_f = np.array([float(lo(x)) for x in mux])
    muy_f = np.array([float(lo(x)) for x in muy])
    cx_f = np.array([float(lo(x)) for x in cx])
    cy_f = np.array([float(lo(x)) for x in cy])
    Omf = cx_f[:, None] * cy_f[None, :] / (mux_f[:, None] + muy_f[None, :]) ** 2
    Wf = mux_f[:, None] + muy_f[None, :]
    Lidx = list(range(N))                                      # odd mode 2j-1 has index j-1
    Yrows = [[om(i, j) for j in range(len(muy))] for i in Lidx]
    E2 = iv.mpf(0)
    nrows_full = min(len(mux), (M_ROWS_DISCRETE + 1) // 2)
    for i in range(N, nrows_full):
        A = (Omf[Lidx, :] * Wf[i, :]).T
        t = np.linalg.lstsq(A, Omf[i, :] * Wf[i, :], rcond=None)[0]
        tiv = [iv.mpf(float(x)) for x in t]
        s = iv.mpf(0)
        for j in range(len(muy)):
            d = om(i, j) - sum((tiv[k] * Yrows[k][j] for k in range(N)), iv.mpf(0))
            s += ((mux[i] + muy[j]) * d) ** 2
        E2 += s
    if nrows_full < len(mux):                                  # rows beyond: rho_m^2 <= c_m^2 / mu_m^2 * sum_n c_n^2
        scn = sum((c ** 2 for c in cy), iv.mpf(0))
        E2 += scn * sum((cx[i] ** 2 / mux[i] ** 2 for i in range(nrows_full, len(mux))), iv.mpf(0))
    W0 = sum((cx[i] ** 2 * cy[j] ** 2 / (mux[i] + muy[j]) ** 2 for i in range(min(len(mux), 40))
              for j in range(min(len(muy), 40))), iv.mpf(0))
    return dict(c0=hx * hy, dN1=dN1, dN=dN, E2=E2, E1=None, W0=W0, cert=(cert1, cert2))


def outward(x, fmt, upward):
    """Printed bound: x rounded OUTWARD to the digits of fmt ("%.4f": 4 decimals, "%.4g": 4 significant digits), up for an
    upper bound and down for a lower bound (review N-1, QA-1). Printing only; the certificate and mu < 1 are not changed.

    x comes from interval endpoints through round-to-nearest mp operations (relative error ~2^-250); the factor 1 +- 2^-200
    covers it before the directed rounding."""
    x = x * (1 + mp.mpf(2) ** -200) if upward else x * (1 - mp.mpf(2) ** -200)
    places = 4 if fmt == "%.4f" else 3 - int(mp.floor(mp.log10(x)))
    k = (mp.ceil if upward else mp.floor)(x * mp.mpf(10) ** places)
    text = fmt % float(k / mp.mpf(10) ** places)
    assert mp.nint(mp.mpf(text) * mp.mpf(10) ** places) == k, (x, text)  # the printed digits are exactly the rounded value
    return text


def report(label, res):
    c0 = res['c0']
    mu = mp.sqrt(hi(res['E2']) / lo(res['dN1']))
    norm = mp.sqrt(lo(c0 * res['W0']))
    e2 = mp.sqrt(hi(c0 * res['E2'])) / norm
    e1 = outward(mp.sqrt(hi(c0 * res['E1'])) / norm, "%.4g", True) if res['E1'] is not None else "-"
    dn1 = mp.sqrt(lo(c0 * res['dN1'])) / norm
    dn = mp.sqrt(lo(c0 * res['dN'])) / norm
    enc = mp.sqrt(lo(res['dN']) / hi(res['E2']))
    ok = mu < 1
    print("%s | err1/|w0|<=%s err2/|w0|<=%s d(N-1)/|w0|>=%.4g d(N)/|w0|>=%.4g | mu<=%s %s | d(N)_low/err2_up>=%s | j_cert %s"
          % (label, e1, outward(e2, "%.4g", True), float(dn1), float(dn), outward(mu, "%.4f", True),
             "CERTIFIED" if ok else "NOT_CERTIFIED", outward(enc, "%.4f", False), res['cert']), flush=True)
    return ok


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "continuous"
    allok = True
    if mode == "continuous":
        print("continuous untruncated SSSS, odd-sine start; K_SUB=%d M_ROWS=%d M_COLS=%d iv.prec=%d" % (K_SUB, M_ROWS, M_COLS, iv.prec))
        for r in R_CONTROL:
            for N in (2, 3):
                allok &= report("r=%-4s N=%d" % (r, N), continuous_case(r, N))
    elif mode == "discrete":
        grids = [(n, n, r) for n in (40, 80, 160, 320, 640, 1280) for r in R_CONTROL]
        grids += [(1600, 320, Fraction(1, 5)), (960, 320, Fraction(1, 3)), (640, 320, Fraction(1, 2)),
                  (320, 640, Fraction(2)), (320, 960, Fraction(3)), (320, 1600, Fraction(5))]
        print("discrete SSSS (DST-I), odd-sine start; K_SUB=%d rows<=%d by trial vectors iv.prec=%d" % (K_SUB, M_ROWS_DISCRETE, iv.prec))
        for nx, ny, r in grids:
            for N in (2, 3):
                allok &= report("nx=%-4d ny=%-4d r=%-4s N=%d" % (nx, ny, r, N), discrete_case(nx, ny, r, N))
    elif mode == "dense":
        print("dense sample of r (point certificates at each sampled r only); K_SUB=%d M_ROWS=%d M_COLS=%d" % (K_SUB, M_ROWS, M_COLS))
        for i in range(201):
            r = Fraction(0.2 * 25.0 ** (i / 200)).limit_denominator(10 ** 6)
            for N in (2, 3):
                allok &= report("i=%3d r=%.6f N=%d" % (i, float(r), N), continuous_case(r, N))
    print("ALL CERTIFIED" if allok else "SOME NOT CERTIFIED")
