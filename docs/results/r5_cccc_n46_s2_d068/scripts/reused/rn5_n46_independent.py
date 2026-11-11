# R5-CCCC-N46-RN5-INDEPENDENT-QA (QA scratch, not in git). Independent exact verification of RN-5 (study PR #49 @ 9377e18, 4.6)
# on Builder's ORIGINAL saved s = 1 factors (read-only). No import of kkvim, kkvim_reference or the Builder script; numpy only reads the
# npz files and proposes nothing that enters a proof except as a search start. No QR of V, no RT-7, no X-half-step, no s = 2.
#
# Independent route (differs from kkvim.cccc_n23_certificate by construction):
#   pi       : BBP series with a geometric tail bound (not Machin);
#   sin^2    : Taylor series of sin with the alternating remainder on [theta_lo, theta_hi] (not cos 2x);
#   lambda   : lambda_min(K(U~)) >= D[a0*l0^2 + 2*b0*l0 + c0] (Kronecker sum of PSD terms), a0, b0, c0 lower eigenvalue bounds of the
#              N x N coefficient matrices proved by the exact characteristic polynomial + Descartes on the shifted polynomial (real roots),
#              l0 <= lambda_min(-L_y) from the closed form, re-proved by the exact continuant (Sylvester) recursion; no LDL of K;
#   x*       : never solved; ||x_hat - x*|| <= ||K x_hat - f||_2 / lambda_low with the EXACT residual (the BR bound of 4.6);
#   sigma_N  : sigma_N(w_hat_w)^2 = lambda_min(A_U GV) by characteristic polynomial + Descartes + exact bisection;
#   norms    : ||dA||_2, ||dB_c||_2, ||dC_c||_2 by lambda_max(X^T X) (char. polynomial), intervals by Frobenius; R rounded to 2^-128.
#
#   python -I rn5_n46_independent.py selftest
#   python -I rn5_n46_independent.py run C4            (minimum grid first)
#   python -I rn5_n46_independent.py run all
import csv
import hashlib
import json
import math
import platform
import sys
import time
from fractions import Fraction as Q
from pathlib import Path

import numpy as np

sys.set_int_max_str_digits(0)
HERE = Path(__file__).resolve().parent
QA_WT = HERE.parents[1]
SRC = Path(r"<BUILDER_WORKTREE>\scratch\d067\rn5_feasibility")
PREV = QA_WT / "scratch" / "d067" / "cmp_runs"
PROVRUN = HERE / "prov_runs"
P = 256  # dyadic grid of the sin^2 centres
IMPL_SHA = "a97cdbeb583ac0d4389fe738c85009492feb8240"
PHYS = {"a": 1.0, "t": 0.01, "E": 2.1e11, "nu": 0.3, "q": 1000.0}
CASES = [("C1", "N4_32x32", 4, 32, 32, 1.0), ("C2", "N5_32x32", 5, 32, 32, 1.0), ("C3", "N6_32x32", 6, 32, 32, 1.0),
         ("C4", "N4_9x9", 4, 9, 9, 1.0), ("C5", "N5_11x11", 5, 11, 11, 1.0), ("C6", "N6_13x13", 6, 13, 13, 1.0),
         ("C7", "N5_aniso_b0.6_20x13", 5, 20, 13, 0.6)]
PREV_DIR = {"C1": PREV / "n4_32_s1", "C2": PREV / "n5_32_s1", "C3": PREV / "n6_32_s1", "C4": PREV / "n4_9_s1",
            "C5": PREV / "n5_11_s1", "C6": PREV / "n6_13_s1", "C7": PROVRUN / "n5_20x13_b06_s1"}


# ---------------------------------------------------------------- exact helpers
def pow2(e):
    return Q(1 << e) if e >= 0 else Q(1, 1 << -e)


def down(x, bits):
    return Q(math.floor(x * (1 << bits)), 1 << bits)


def up(x, bits):
    return Q(math.ceil(x * (1 << bits)), 1 << bits)


def isqrt_bracket(x, bits):
    """r, e with r/2^e <= sqrt(x) < (r+1)/2^e for rational x > 0 (integer isqrt)."""
    p, q = x.numerator, x.denominator
    e = bits - (p.bit_length() - q.bit_length()) // 2
    y = (p << (2 * e)) // q if e >= 0 else p // (q << (-2 * e))
    return math.isqrt(y), e


def sqrt_up(x, bits=100):
    x = Q(x)
    assert x >= 0
    if x == 0:
        return Q(0)
    r, e = isqrt_bracket(x, bits)
    return Q(r + 1) / pow2(e)


def sqrt_dn(x, bits=100):
    x = Q(x)
    assert x >= 0
    if x == 0:
        return Q(0)
    r, e = isqrt_bracket(x, bits)
    return Q(r) / pow2(e)


def mm(X, Y):
    Yt = list(zip(*Y))
    return [[sum(a * b for a, b in zip(row, col)) for col in Yt] for row in X]


def tp(X):
    return [list(r) for r in zip(*X)]


def eye(n):
    return [[Q(int(i == j)) for j in range(n)] for i in range(n)]


def madd(X, Y, s=1):
    return [[a + s * b for a, b in zip(rx, ry)] for rx, ry in zip(X, Y)]


def mscale(X, s):
    return [[s * a for a in r] for r in X]


def solve(M, B):
    """Exact Gauss elimination with first-nonzero pivot; returns (X, det)."""
    n = len(M)
    A = [list(M[i]) + list(B[i]) for i in range(n)]
    det = Q(1)
    for k in range(n):
        p = next((i for i in range(k, n) if A[i][k] != 0), None)
        if p is None:
            return None, Q(0)
        if p != k:
            A[k], A[p] = A[p], A[k]
            det = -det
        det *= A[k][k]
        piv = A[k][k]
        A[k] = [v / piv for v in A[k]]
        for i in range(n):
            if i != k and A[i][k] != 0:
                f = A[i][k]
                A[i] = [a - f * b for a, b in zip(A[i], A[k])]
    return [row[n:] for row in A], det


def charpoly(M):
    """det(xI - M), coefficients high -> low, Faddeev-LeVerrier in exact arithmetic."""
    n = len(M)
    c = [Q(1)]
    Mk = [[Q(0)] * n for _ in range(n)]
    for k in range(1, n + 1):
        Mk = madd(mm(M, Mk), mscale(eye(n), c[-1]))
        AM = mm(M, Mk)
        c.append(-sum(AM[i][i] for i in range(n)) / k)
    return c


def taylor_shift(p, s):
    q = list(p)
    n = len(q) - 1
    for i in range(n):
        for j in range(1, n - i + 1):
            q[j] += s * q[j - 1]
    return q


def roots_above(p, s):
    """Number of roots > s of a REAL-ROOTED polynomial p (Descartes is exact then); None if p(s) = 0."""
    q = taylor_shift(p, s)
    if q[-1] == 0:
        return None
    sg = [c > 0 for c in q if c != 0]
    return sum(1 for a, b in zip(sg, sg[1:]) if a != b)


def lam_min_bracket(p, iters=72):
    """(lo, hi): every root > lo (proved), lambda_min <= hi; None if not all roots > 0 (not PD)."""
    n = len(p) - 1

    def ok(s):
        return roots_above(p, s) == n
    if not ok(Q(0)):
        return None
    e = 0
    if ok(Q(1)):
        while ok(pow2(e + 1)):
            e += 1
            if e > 4000:
                return None
        lo, hi = pow2(e), pow2(e + 1)
    else:
        while not ok(pow2(e - 1)):
            e -= 1
            if e < -6000:
                return None
        lo, hi = pow2(e - 1), pow2(e)
    for _ in range(iters):
        mid = (lo + hi) / 2
        if ok(mid):
            lo = mid
        else:
            hi = mid
    return lo, hi


def lam_max_upper(p, iters=72):
    """t with every root < t... precisely: no root > t and p(t) != 0 (proved)."""
    def none_above(t):
        return roots_above(p, t) == 0
    e = 0
    if none_above(Q(1)):
        while none_above(pow2(e - 1)) and e > -6000:
            e -= 1
        lo, hi = pow2(e - 1), pow2(e)
        if not none_above(hi):
            raise ArithmeticError("bracket")
    else:
        while not none_above(pow2(e)):
            e += 1
            if e > 4000:
                raise ArithmeticError("no upper bound")
        lo, hi = pow2(e - 1), pow2(e)
    for _ in range(iters):
        mid = (lo + hi) / 2
        if none_above(mid):
            hi = mid
        else:
            lo = mid
    return hi


def spec_norm_up(X):
    """||X||_2 <= returned value, via lambda_max(X^T X) exactly bracketed."""
    G = mm(tp(X), X)
    if all(v == 0 for r in G for v in r):
        return Q(0)
    return sqrt_up(lam_max_upper(charpoly(G)))


def frob_up(entries):
    return sqrt_up(sum(v * v for v in entries))


# ---------------------------------------------------------------- pi and sin^2 enclosures (independent route)
def pi_enclosure(K=110):
    s = Q(0)
    for k in range(K):
        s += Q(1, 16**k) * (Q(4, 8 * k + 1) - Q(2, 8 * k + 4) - Q(1, 8 * k + 5) - Q(1, 8 * k + 6))
    tail = Q(4 * 16, 15 * 16**K)  # every bracket is in (0, 4]
    return down(s, 420), up(s + tail, 420)


PI = pi_enclosure()


def sin_enclosure(x, bits):
    """lo <= sin x <= hi for rational 0 <= x <= 2 (alternating Taylor series, decreasing terms)."""
    assert 0 <= x <= 2
    if x == 0:
        return Q(0), Q(0)
    s, term, k = Q(0), x, 0
    while True:
        s += term
        nxt = -term * x * x / ((2 * k + 2) * (2 * k + 3))
        k += 1
        if abs(nxt) < pow2(-(bits + 24)):
            return min(s, s + nxt), max(s, s + nxt)
        term = nxt


def sin2_enclosure(k, n, bits=P):
    """centre (dyadic 2^-bits) and radius of sin^2(k pi / n), 0 <= k <= n/2."""
    if k == 0:
        return Q(0), Q(0)
    if 2 * k == n:
        return Q(1), Q(0)
    lo = sin_enclosure(PI[0] * k / n, bits)[0]
    hi = sin_enclosure(PI[1] * k / n, bits)[1]
    lo, hi = max(lo, Q(0)), min(hi, Q(1))  # sin increasing on [0, pi/2]
    lo2, hi2 = lo * lo, hi * hi
    c = Q(round((lo2 + hi2) / 2 * (1 << bits)), 1 << bits)
    return c, max(hi2 - c, c - lo2)


def tstar(n, N):
    """T* (m x N) centres and one radius rho >= |T* - Tc| entrywise."""
    cache, rho = {}, Q(0)
    T = [[None] * N for _ in range(n - 1)]
    for i in range(1, N + 1):
        for l in range(1, n):  # noqa: E741
            k = (i * l) % n
            k = min(k, n - k)
            if k not in cache:
                cache[k] = sin2_enclosure(k, n)
            T[l - 1][i - 1] = cache[k][0]
            rho = max(rho, cache[k][1])
    return T, rho


# ---------------------------------------------------------------- operators (from ВБC-12: D4 = L^2 + E2)
def ops(n, h):
    m = n - 1
    L = [[(Q(-2) if i == j else (Q(1) if abs(i - j) == 1 else Q(0))) / h**2 for j in range(m)] for i in range(m)]
    D4 = mm(L, L)
    D4[0][0] += Q(2) / h**4
    D4[m - 1][m - 1] += Q(2) / h**4
    assert D4[0][:3] == [Q(7) / h**4, Q(-4) / h**4, Q(1) / h**4]  # boundary row (7, -4, 1)/h^4
    return L, D4


def gersh(M):
    return max(sum(abs(v) for v in r) for r in M)


def continuant_pd(h, shift, m):
    """-L - shift I > 0 (tridiagonal, diag 2/h^2 - shift, off -1/h^2) by positive leading minors (Sylvester)."""
    d, o2 = Q(2) / h**2 - shift, Q(1) / h**4
    p0, p1 = Q(1), d
    if p1 <= 0:
        return False
    for _ in range(2, m + 1):
        p0, p1 = p1, d * p1 - o2 * p0
        if p1 <= 0:
            return False
    return True


def coeffs(U, h, q, L, D4):
    Ut = tp(U)
    A = mscale(mm(Ut, U), h)
    B = mscale(mm(Ut, mm(L, U)), -h)
    C = mscale(mm(Ut, mm(D4, U)), h)
    F = [q * h * sum(r[i] for r in U) for i in range(len(U[0]))]
    return A, B, C, F


# ---------------------------------------------------------------- the independent RN-5 verification
def rn5_independent(a, b, nx, ny, N, G, V, D, q):
    t0 = time.perf_counter()
    H, val = {}, {}
    out = {"hypotheses": H, "values": val, "certified": False, "first_failed": None}

    def stop(name):
        out["first_failed"] = name
        out["seconds"] = time.perf_counter() - t0
        return out
    H["H1 2N < n_x (ВБC-13 p.3)"] = 2 * N < nx
    H["H2 N <= floor(n_x/2) (ВБC-13 p.4, T_N independent)"] = N <= nx // 2
    H["H3 N <= floor(n_y/2) (RN-3, rank N admissible)"] = N <= ny // 2
    for k in list(H):
        if not H[k]:
            return stop(k)
    la, lb, D, q = Q(a), Q(b), Q(D), Q(q)
    hx, hy, mx, my = la / nx, lb / ny, nx - 1, ny - 1
    Lx, D4x = ops(nx, hx)
    Ly, D4y = ops(ny, hy)
    Gq = [[Q(float(v)) for v in r] for r in G]
    Vq = [[Q(float(v)) for v in r] for r in V]
    # --- sin^2 enclosures and the closed forms of ВБC-13 p.3 (coefficient identities)
    Tc, rho = tstar(nx, N)
    val["rho_T"] = rho
    mu_c = [4 / hx**2 * Tc[0][i] for i in range(N)]
    mu_r = 4 / hx**2 * rho
    A_T = [[la * (Q(1, 8) * (i == j) + Q(1, 4)) for j in range(N)] for i in range(N)]
    F_T = [q * la / 2] * N
    Ac, Bc, Cc, Fc = coeffs(Tc, hx, q, Lx, D4x)
    ok = True
    worst = Q(0)
    for name, X, M, rs in (("A", Ac, None, Q(1)), ("B", Bc, Lx, gersh(Lx)), ("C", Cc, D4x, gersh(D4x))):
        MT = Tc if M is None else mm(M, Tc)
        cs = [sum(abs(MT[l][i]) for l in range(mx)) for i in range(N)]
        for i in range(N):
            for j in range(N):
                pert = hx * (rho * (cs[i] + cs[j]) + rho * rho * mx * rs)
                if name == "A":
                    closed, crad = A_T[i][j], Q(0)
                elif name == "B":
                    closed, crad = (la / 8 * mu_c[i], la / 8 * mu_r) if i == j else (Q(0), Q(0))
                else:
                    closed, crad = (la / 8 * mu_c[i] ** 2, la / 8 * (2 * mu_c[i] * mu_r + mu_r**2)) if i == j else (Q(0), Q(0))
                dev = abs(X[i][j] - closed)
                if dev > pert + crad:
                    ok = False
                worst = max(worst, dev)
    for i in range(N):
        if abs(Fc[i] - F_T[i]) > q * hx * mx * rho:
            ok = False
    H["H4 closed forms ВБC-13 p.3 inside the sin^2 enclosure"] = ok
    val["closed_form_max_abs_dev_at_centres"] = worst
    if not ok:
        return stop("H4 closed forms ВБC-13 p.3 inside the sin^2 enclosure")
    # --- R, U' = T* R (in T_N), E = U~ - U'
    TtT = mm(tp(Tc), Tc)
    Rf, _ = solve(TtT, mm(tp(Tc), Gq))
    R = [[Q(round(v * (1 << 128)), 1 << 128) for v in r] for r in Rf]  # any rational R is admissible
    _, detR = solve(R, eye(N))
    H["H5 R nonsingular (span U' = T_N)"] = detR != 0
    if detR == 0:
        return stop("H5 R nonsingular (span U' = T_N)")
    TR = mm(Tc, R)
    Er = [[Gq[l][i] - TR[l][i] for i in range(N)] for l in range(mx)]
    Erad = [rho * sum(abs(R[k][i]) for k in range(N)) for i in range(N)]
    EF = frob_up([abs(Er[l][i]) + Erad[i] for l in range(mx) for i in range(N)])
    # --- coefficient perturbations
    AU, BU, CU, FU = coeffs(Gq, hx, q, Lx, D4x)
    Rt = tp(R)
    dA = madd(AU, mm(mm(Rt, A_T), R), -1)
    dBc = madd(BU, mm(mm(Rt, [[la / 8 * mu_c[i] if i == j else Q(0) for j in range(N)] for i in range(N)]), R), -1)
    dCc = madd(CU, mm(mm(Rt, [[la / 8 * mu_c[i] ** 2 if i == j else Q(0) for j in range(N)] for i in range(N)]), R), -1)
    dBr = [[la / 8 * mu_r * sum(abs(R[k][i] * R[k][j]) for k in range(N)) for j in range(N)] for i in range(N)]
    dCr = [[la / 8 * sum(abs(R[k][i] * R[k][j]) * (2 * mu_c[k] * mu_r + mu_r**2) for k in range(N)) for j in range(N)] for i in range(N)]
    nA = spec_norm_up(dA)
    nB = spec_norm_up(dBc) + frob_up([v for r in dBr for v in r])
    nC = spec_norm_up(dCc) + frob_up([v for r in dCr for v in r])
    dF = [FU[i] - sum(R[k][i] * F_T[k] for k in range(N)) for i in range(N)]
    nF = frob_up(dF) * sqrt_up(my)
    nD4y, nLy = gersh(D4y), gersh(Ly)
    nK = D * (nA * nD4y + 2 * nB * nLy + nC)
    val.update(norm_dA=nA, norm_dB=nB, norm_dC=nC, norm_dF_x1=nF, norm_D4y_bound=nD4y, norm_Ly_bound=nLy, norm_dK=nK, norm_E_F=EF)
    # --- lambda_low(K(U~)) by the Kronecker PSD bound
    brA, brB, brC = (lam_min_bracket(charpoly(X)) for X in (AU, BU, CU))
    if brA is None or brB is None or brC is None:
        H["H6a A_U, B_U, C_U positive definite (char. poly.)"] = False
        return stop("H6a A_U, B_U, C_U positive definite (char. poly.)")
    H["H6a A_U, B_U, C_U positive definite (char. poly.)"] = True
    a0, b0, c0 = brA[0], brB[0], brC[0]
    s_lo = sin_enclosure(PI[0] / (2 * ny), 160)[0]
    l0 = down(4 / hy**2 * s_lo * s_lo, 200)
    H["H6b -L_y - l0 I > 0 (continuant, exact)"] = l0 > 0 and continuant_pd(hy, l0, my)
    if not H["H6b -L_y - l0 I > 0 (continuant, exact)"]:
        return stop("H6b -L_y - l0 I > 0 (continuant, exact)")
    lam = D * (a0 * l0 * l0 + 2 * b0 * l0 + c0)
    val.update(lamA_lo=a0, lamB_lo=b0, lamC_lo=c0, l0=l0, lambda_low=lam)
    H["H6 lambda_low(K(U~)) > ||dK||_2"] = lam > nK
    if not lam > nK:
        return stop("H6 lambda_low(K(U~)) > ||dK||_2")
    # --- exact residual of the saved V on the exact system of U~ (BR bound; x* is never computed)
    T1 = mm(mm(D4y, Vq), AU)
    T2 = mm(mm(Ly, Vq), BU)
    T3 = mm(Vq, CU)
    res = [D * (T1[p][i] - 2 * T2[p][i] + T3[p][i]) - FU[i] for p in range(my) for i in range(N)]
    nres = frob_up(res)
    xh = frob_up([v for r in Vq for v in r])
    dxh = nres / lam
    xs = xh + dxh
    dprime = (nF + nK * xs) / (lam - nK)
    sx, sy = sqrt_up(hx), sqrt_up(hy)
    UW = sqrt_up(lam_max_upper(charpoly(AU)))
    eps1 = sx * EF * sy * (xs + dprime) + UW * sy * dprime
    eps2 = UW * sy * dxh
    eps = eps1 + eps2
    GV = mscale(mm(tp(Vq), Vq), hy)
    br = lam_min_bracket(charpoly(mm(AU, GV)))
    val.update(norm_residual=nres, norm_x_hat=xh, dx_hat_bound=dxh, norm_x_star_bound=xs, dx_prime=dprime, norm_Uw=UW,
               eps1=eps1, eps2=eps2, eps=eps)
    if br is None:
        H["H7 sigma_N(w_hat_w)^2 > eps^2 (Weyl)"] = False
        return stop("H7 sigma_N(w_hat_w)^2 > eps^2 (Weyl)")
    s2lo, s2hi = br
    val.update(sigma2_low=s2lo, sigma2_high=s2hi, sigma_low=sqrt_dn(s2lo), sigma_high=sqrt_up(s2hi))
    H["H7 sigma_N(w_hat_w)^2 > eps^2 (Weyl)"] = s2lo > eps * eps
    if not H["H7 sigma_N(w_hat_w)^2 > eps^2 (Weyl)"]:
        return stop("H7 sigma_N(w_hat_w)^2 > eps^2 (Weyl)")
    val["margin_sigma_over_eps_low"] = sqrt_dn(s2lo) / eps
    val["margin_lambda_over_dK"] = lam / nK if nK else None
    out["certified"] = True
    out["aux"] = {"AU": AU, "BU": BU, "CU": CU, "Ly": Ly, "D4y": D4y, "D": D}
    out["seconds"] = time.perf_counter() - t0
    return out


def rayleigh_upper(aux, N, my):
    """An upper bound of lambda_min(K(U~)): exact Rayleigh quotient of a float-proposed vector (consistency check only)."""
    AU, BU, CU, Ly, D4y, D = (aux[k] for k in ("AU", "BU", "CU", "Ly", "D4y", "D"))
    f = lambda M: np.array([[float(v) for v in r] for r in M])  # noqa: E731
    K = float(D) * (np.kron(f(D4y), f(AU)) - 2 * np.kron(f(Ly), f(BU)) + np.kron(np.eye(my), f(CU)))
    w, Vec = np.linalg.eigh(K)
    X = [[Q(float(Vec[p * N + i, 0])) for i in range(N)] for p in range(my)]
    T1, T2, T3 = mm(mm(D4y, X), AU), mm(mm(Ly, X), BU), mm(X, CU)
    num = D * sum(X[p][i] * (T1[p][i] - 2 * T2[p][i] + T3[p][i]) for p in range(my) for i in range(N))
    return num / sum(X[p][i] ** 2 for p in range(my) for i in range(N)), float(w[0])


# ---------------------------------------------------------------- provenance and integrity
def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def integrity():
    sums = {}
    for line in (SRC / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        if line.strip():
            h, name = line.split(" *", 1)
            sums[name.strip()] = h.strip()
    checked = {n: sha_file(SRC / n) == h for n, h in sums.items()}
    required = [f"factors_{c[1]}.npz" for c in CASES] + [f"case_{c[1]}.json" for c in CASES] + ["provenance.json", "rn5_feasibility_results.json"]
    return {"listed": len(sums), "all_match": all(checked.values()), "mismatch": [n for n, v in checked.items() if not v],
            "required_missing": [r for r in required if r not in sums or not (SRC / r).exists()],
            "SHA256SUMS_sha256": sha_file(SRC / "SHA256SUMS.txt"), "files": sums}


def read_prev_factors(d, nx, ny, N):
    G = np.full((nx - 1, N), np.nan)
    V = np.full((ny - 1, N), np.nan)
    with (d / "factors.csv").open(encoding="utf-8") as fh:
        rows = [r for r in csv.reader(fh) if r and not r[0].startswith("#")]
    for r in rows[1:]:
        if r[0] == "1" and r[2] in ("G", "V"):
            (G if r[2] == "G" else V)[int(r[3]) - 1, int(r[4]) - 1] = float(r[5])
    return G, V


def find_key(obj, key):
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            r = find_key(v, key)
            if r is not None:
                return r
    return None


def load_case(cid, label, N, nx, ny, b):
    rec = {"case": cid, "label": label, "N": N, "nx": nx, "ny": ny, "b": b}
    case = json.loads((SRC / f"case_{label}.json").read_text(encoding="utf-8"))
    with np.load(SRC / f"factors_{label}.npz", allow_pickle=False) as z:
        keys = sorted(z.files)
        G, V = np.array(z["G"]), np.array(z["V"])
    chk = {
        "npz_keys_exactly_G_V": keys == ["G", "V"],
        "dtype_float64": G.dtype == np.float64 and V.dtype == np.float64,
        "G_shape": list(G.shape) == [nx - 1, N], "V_shape": list(V.shape) == [ny - 1, N],
        "finite": bool(np.all(np.isfinite(G)) and np.all(np.isfinite(V))),
        "G_sha256_eq_case_json": hashlib.sha256(np.ascontiguousarray(G).tobytes()).hexdigest() == case.get("G_sha256"),
        "V_sha256_eq_case_json": hashlib.sha256(np.ascontiguousarray(V).tobytes()).hexdigest() == case.get("V_sha256"),
        "case_params": (case["N"], case["n_x"], case["n_y"], case["b"], case["a"], case["t"], case["E"], case["nu"], case["q"], case["half_steps"])
        == (N, nx, ny, b, PHYS["a"], PHYS["t"], PHYS["E"], PHYS["nu"], PHYS["q"], 1),
        "s1_status_completed": case.get("s1_status") == "COMPLETED_FIXED_HALF_STEPS",
        "provenance_commit": case.get("provenance_commit") == IMPL_SHA,
        "D_hex_consistent": float.fromhex(case["D_hex"]) == case["D"],
    }
    Gp, Vp = read_prev_factors(PREV_DIR[cid], nx, ny, N)
    prev = json.loads((PREV_DIR[cid] / "result.json").read_text(encoding="utf-8"))
    chk["bitwise_eq_QA_rerun_G"] = bool(np.array_equal(Gp, G))
    chk["bitwise_eq_QA_rerun_V"] = bool(np.array_equal(Vp, V))
    chk["QA_rerun_commit"] = find_key(prev, "git_commit") == IMPL_SHA and find_key(prev, "git_worktree_modified") is False
    chk["D_eq_QA_rerun"] = prev["D"]["value"] == case["D"]
    rec["checks"] = chk
    rec["provenance_ok"] = all(chk.values())
    rec["D"], rec["D_hex"] = case["D"], case["D_hex"]
    rec["builder_rn5"] = case.get("rn5", {})
    return rec, G, V, case


def fx(v):
    if isinstance(v, Q):
        return float(v)
    return v


def exact_str(v):
    return f"{v.numerator}/{v.denominator}" if isinstance(v, Q) else v


def bval(case, k):
    v = (case.get("rn5", {}).get("values", {}) or {}).get(k)
    return v.get("float") if isinstance(v, dict) else v


def run(ids):
    sel = [c for c in CASES if ids == ["all"] or c[0] in ids]
    integ = integrity()
    print("integrity:", integ["all_match"], integ["mismatch"], integ["required_missing"], flush=True)
    results = []
    for cid, label, N, nx, ny, b in sel:
        rec, G, V, case = load_case(cid, label, N, nx, ny, b)
        r = rn5_independent(PHYS["a"], b, nx, ny, N, G, V, rec["D"], PHYS["q"])
        aux = r.pop("aux", None)
        rec["independent"] = {"certified": r["certified"], "first_failed": r["first_failed"], "hypotheses": r["hypotheses"],
                              "values_float": {k: fx(v) for k, v in r["values"].items()},
                              "decisive_exact": {k: exact_str(r["values"][k]) for k in ("lambda_low", "norm_dK", "eps", "sigma2_low")
                                                 if k in r["values"] and len(exact_str(r["values"][k])) < 20000},
                              "seconds": r["seconds"]}
        cmp = {}
        if r["certified"]:
            vals = r["values"]
            ub, wf = rayleigh_upper(aux, N, ny - 1)
            bl, bs, bd = bval(case, "lambda_low"), bval(case, "sigma_low"), bval(case, "dx_hat")
            cmp = {
                "rayleigh_upper_lambda_min_K": float(ub), "float_eig_min_K": wf,
                "QA_lambda_low_below_rayleigh": vals["lambda_low"] < ub,
                "Builder_lambda_low_below_rayleigh": bl is not None and Q(bl) < ub,
                "Builder_sigma_low_le_QA_sigma_high": bs is not None and Q(bs) <= vals["sigma_high"],
                "Builder_dx_hat_le_QA_bound": bd is not None and Q(bd) <= vals["dx_hat_bound"],
                "Builder": {k: bval(case, k) for k in ("norm_dA", "norm_dB", "norm_dC", "norm_dF", "norm_dK", "lambda_low", "dx_hat",
                                                       "norm_x_star", "eps", "sigma_low", "margin_sigma_over_eps")},
                "Builder_verdict": case.get("rn5", {}).get("verdict"),
            }
        rec["comparison"] = cmp
        rec.pop("builder_rn5", None)
        results.append(rec)
        iv = rec["independent"]["values_float"]
        print(cid, label, "prov", rec["provenance_ok"], "cert", r["certified"], r["first_failed"],
              "lam", f"{iv.get('lambda_low', 0):.4g}", "dK", f"{iv.get('norm_dK', 0):.3g}", "eps", f"{iv.get('eps', 0):.3g}",
              "sig", f"{iv.get('sigma_low', 0):.4g}", "margin", f"{iv.get('margin_sigma_over_eps_low', 0):.4g}",
              f"{r['seconds']:.1f}s", flush=True)
    out = {"task": "R5-CCCC-N46-RN5-INDEPENDENT-QA", "label": "RESEARCH_ONLY / PROVISIONAL / NOT_QUALIFIED",
           "script_sha256": sha_file(__file__), "python": platform.python_version(), "numpy": np.__version__,
           "platform": platform.platform(), "integrity": integ, "cases": results}
    name = "rn5_n46_results_all.json" if ids == ["all"] else f"rn5_n46_results_{'_'.join(ids)}.json"
    (HERE / name).write_text(json.dumps(out, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    print("written", name, flush=True)


# ---------------------------------------------------------------- self-tests of the primitives and negative controls
def selftest():
    res = {}
    pl = Q(314159265358979323846264338327950288419716939937510, 10**50)  # 50 decimals of pi, truncated: pl < pi < pl + 1e-50
    res["pi_enclosure"] = pl < PI[0] < PI[1] < pl + Q(1, 10**50) and float(PI[1] - PI[0]) < 1e-120
    errs = []
    for n in (9, 13, 20, 32):
        for k in range(1, n // 2 + 1):
            c, r = sin2_enclosure(k, n)
            errs.append(abs(float(c) - math.sin(k * math.pi / n) ** 2))
            assert r < pow2(-P + 4)
    res["sin2_vs_libm_max_abs"] = max(errs)
    p = charpoly([[Q(2), Q(1)], [Q(1), Q(2)]])
    res["charpoly_2x2"] = p == [1, -4, 3]
    res["descartes"] = (roots_above(p, Q(0)), roots_above(p, Q(2)), roots_above(p, Q(4)), roots_above(p, Q(1))) == (2, 1, 0, None)
    lo, hi = lam_min_bracket(p)
    res["lam_min_bracket_1"] = lo < 1 <= hi and float(hi - lo) < 1e-20
    rng = np.random.default_rng(7)
    M = rng.standard_normal((6, 6))
    S = M @ M.T + 1e-3 * np.eye(6)
    Sq = [[Q(float(v)) for v in r] for r in S]
    lo, hi = lam_min_bracket(charpoly(Sq))
    res["lam_min_vs_numpy_rel"] = abs(float(lo) - np.linalg.eigvalsh(S)[0]) / np.linalg.eigvalsh(S)[0]
    res["lam_max_vs_numpy_rel"] = abs(float(lam_max_upper(charpoly(Sq))) - np.linalg.eigvalsh(S)[-1]) / np.linalg.eigvalsh(S)[-1]
    n, h = 13, Q(6, 10) / 13
    lmin = 4 / h**2 * math.sin(math.pi / (2 * n)) ** 2
    res["continuant_pos_below"] = continuant_pd(h, Q(lmin * (1 - 1e-12)), n - 1)
    res["continuant_neg_above"] = not continuant_pd(h, Q(lmin * (1 + 1e-12)), n - 1)
    # negative controls on a real case (C4): (i) V with two equal columns (rank N - 1): must NOT certify;
    # (ii) G replaced by a set not close to T_N: must NOT certify.
    rec, G, V, case = load_case(*CASES[3])
    V1 = V.copy()
    V1[:, -1] = V1[:, 0]
    r1 = rn5_independent(PHYS["a"], 1.0, 9, 9, 4, G, V1, rec["D"], PHYS["q"])
    res["neg_rank_deficient_V_not_certified"] = (r1["certified"], r1["first_failed"])
    G2 = G.copy()
    G2[:, 0] = G2[:, 0] + 1e-3 * np.arange(1, 9)
    r2 = rn5_independent(PHYS["a"], 1.0, 9, 9, 4, G2, V, rec["D"], PHYS["q"])
    res["neg_G_off_T_N_not_certified"] = (r2["certified"], r2["first_failed"])
    for k, v in res.items():
        print(k, v)
    (HERE / "rn5_n46_selftest.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    if sys.argv[1] == "selftest":
        selftest()
    elif sys.argv[1] == "run":
        run(sys.argv[2:])
