"""CCCC, N = 2 or 3: the RN-5 certificate of the rank of the first half-step and the conditions RT-7 of the QR before s = 2 (D-065).

    rn5_certificate(a, b, n_x, n_y, N, G, V, D, q) -> dict
    rt7_conditions(V, G2, rn5) -> dict

RN-5 is the derivation of the project in docs/R5_CCCC_Q4_PRE_D1_RANK_SCIENTIFIC_STUDY.md, 4.6 (PR #49 @ 9377e18; level B, not of the authors);
the construction follows the QA certificate of R5-CCCC-N23-RN5-S2 (RN5_CERTIFIED_LIMITED, scratch), moved here without a change of the mathematics.
RT-7 (R7-1...R7-3) is the rule of D-065, 6: sin theta_max(span G2, U2*) <= sqrt(tau) + t < 1, Euclidean transversality only.

| Component                                               | Specification          | Level        | Decision     |
| exact subspace T_N, enclosures of sin^2 and pi          | ВБC-13 p. 3; study 4.6 | B (project)  | D-065.2      |
| coefficient perturbation, lambda_low, field error, Weyl | study 4.6 (RN-5)       | B (project)  | D-065.2      |
| exact ranks of V and G2 by two procedures, tau, R7-3     | D-065, 6 (RT-7)        | B / V        | D-065.3      |
| outward rational square roots                            | integer isqrt          | V (numerics) | D-065.6      |

Rigour: the stored doubles are exact rationals; every bound is exact rational arithmetic (``fractions.Fraction``) or an outward rational bound.
NumPy floats only propose trial values (shifts), which are then proved or rejected exactly. No tolerance is chosen. A hypothesis that is not
established, a strict inequality that does not hold, an input that is not finite or a computation that cannot be completed gives
``certified = False`` with the reason; it never proves a loss of rank.
"""

import math
from fractions import Fraction

import numpy as np

BITS = 320  # dyadic grid of the centres of the enclosures of sin^2 (as the QA certificate)
MODULUS = 2**61 - 1  # the prime of the second exact rank procedure
_COMPUTATION_ERRORS = (ArithmeticError, ValueError, OverflowError, np.linalg.LinAlgError)


class CertificateInputError(ValueError):
    """An input of the certificate is not a finite real matrix of the expected shape."""


# --- outward rational square roots -------------------------------------------------------------------------------------------------

def _root_bracket(x, precision: int):
    """(r, e) with r 2^-e <= sqrt(x) <= (r + 1) 2^-e and r >= 2^precision, for a rational x > 0, by the integer isqrt only.

    With y = floor(x 4^e): y <= x 4^e < y + 1, so isqrt(y) <= sqrt(x 4^e) < isqrt(y) + 1. The dyadic result keeps the later exact arithmetic
    short (the denominator of x is not carried), and no float is involved, so a tiny x cannot make it loop.
    """
    p, q = x.numerator, x.denominator
    e = precision + 2 - (p.bit_length() - q.bit_length()) // 2
    y = (p << 2 * e) // q if e >= 0 else p // (q << -2 * e)
    return math.isqrt(y), e


def _checked(x) -> Fraction:
    if isinstance(x, bool) or not isinstance(x, (int, Fraction, float)):
        raise TypeError(f"квадратный корень определён только для рационального числа, получено {x!r}")
    x = Fraction(x)  # ValueError / OverflowError for nan, inf
    if x < 0:
        raise ValueError(f"квадратный корень отрицательного числа {x!r}")
    return x


def _dyadic(r: int, e: int) -> Fraction:
    return Fraction(r, 1 << e) if e >= 0 else Fraction(r << -e)


def sqrt_up(x, precision: int = 80) -> Fraction:
    """A dyadic s >= sqrt(x) for a rational x >= 0, relative excess at most about 2^-precision."""
    x = _checked(x)
    if x == 0:
        return Fraction(0)
    r, e = _root_bracket(x, precision)
    return _dyadic(r + 1, e)


def sqrt_dn(x, precision: int = 80) -> Fraction:
    """A dyadic s <= sqrt(x) for a rational x >= 0."""
    x = _checked(x)
    if x == 0:
        return Fraction(0)
    r, e = _root_bracket(x, precision)
    return _dyadic(r, e)


# --- exact matrices ------------------------------------------------------------------------------------------------------------------

def exact_matrix(X, rows: int | None = None, cols: int | None = None) -> list:
    """The stored doubles as exact rationals, row by row; CertificateInputError for anything else."""
    try:
        M = np.asarray(X, dtype=float)
    except (TypeError, ValueError) as exc:
        raise CertificateInputError(f"не вещественная матрица: {exc}") from exc
    if M.ndim != 2 or (rows is not None and M.shape[0] != rows) or (cols is not None and M.shape[1] != cols):
        raise CertificateInputError(f"форма {M.shape}, ожидалась ({rows}, {cols})")
    if not np.all(np.isfinite(M)):
        raise CertificateInputError("неконечный элемент")
    return [[Fraction(float(value)) for value in row] for row in M]


def matmul(X: list, Y: list) -> list:
    return [[sum(X[i][k] * Y[k][j] for k in range(len(Y))) for j in range(len(Y[0]))] for i in range(len(X))]


def transpose(X: list) -> list:
    return [list(row) for row in zip(*X, strict=True)]


def inverse(M: list) -> list:
    """Gauss-Jordan in Fraction; ZeroDivisionError for a singular matrix."""
    n = len(M)
    A = [list(row) + [Fraction(int(i == j)) for j in range(n)] for i, row in enumerate(M)]
    for k in range(n):
        p = next((i for i in range(k, n) if A[i][k] != 0), None)
        if p is None:
            raise ZeroDivisionError("вырожденная матрица")
        A[k], A[p] = A[p], A[k]
        pivot = A[k][k]
        A[k] = [value / pivot for value in A[k]]
        for i in range(n):
            if i != k and A[i][k] != 0:
                f = A[i][k]
                A[i] = [x - f * y for x, y in zip(A[i], A[k], strict=True)]
    return [row[n:] for row in A]


# --- exact rank: two different procedures (R7-1) --------------------------------------------------------------------------------------

def rank_fraction(M: list) -> int:
    """The rank over Q by elimination in Fraction, without a threshold."""
    A = [list(row) for row in M]
    r = 0
    for c in range(len(A[0]) if A else 0):
        p = next((i for i in range(r, len(A)) if A[i][c] != 0), None)
        if p is None:
            continue
        A[r], A[p] = A[p], A[r]
        for i in range(r + 1, len(A)):
            if A[i][c] != 0:
                f = A[i][c] / A[r][c]
                A[i] = [x - f * y for x, y in zip(A[i], A[r], strict=True)]
        r += 1
    return r


def rank_modp(M: list, p: int = MODULUS) -> int:
    """The rank over GF(p) of the integer matrix 2^k M (stored doubles are dyadic); rank_p <= rank_Q, so rank_p = N proves rank N."""
    L = max((x.denominator for row in M for x in row), default=1)
    if L & (L - 1):
        raise CertificateInputError("элементы не двоично-рациональны")
    A = [[(x.numerator * (L // x.denominator)) % p for x in row] for row in M]
    r = 0
    for c in range(len(A[0]) if A else 0):
        piv = next((i for i in range(r, len(A)) if A[i][c]), None)
        if piv is None:
            continue
        A[r], A[piv] = A[piv], A[r]
        iv = pow(A[r][c], p - 2, p)
        for i in range(r + 1, len(A)):
            if A[i][c]:
                f = A[i][c] * iv % p
                A[i] = [(x - f * y) % p for x, y in zip(A[i], A[r], strict=True)]
        r += 1
    return r


# --- enclosures of pi, cos and sin^2 (Machin; Taylor with the Lagrange remainder) -------------------------------------------------

def _atan_inv(x: int, terms: int):
    s, parts = Fraction(0), []
    for k in range(terms):
        s += Fraction((-1) ** k, (2 * k + 1) * x ** (2 * k + 1))
        parts.append(s)
    return min(parts[-2:]), max(parts[-2:])  # alternating decreasing terms: the limit lies between consecutive partial sums


def _pi_enclosure():
    a, b = _atan_inv(5, 160), _atan_inv(239, 40)
    lo, hi = 16 * a[0] - 4 * b[1], 16 * a[1] - 4 * b[0]
    K = 2 ** (BITS + 40)
    return Fraction(math.floor(lo * K), K), Fraction(math.ceil(hi * K), K)


PI = _pi_enclosure()


def _cos2pi(num: int, den: int):
    """cos(2 pi num / den) = centre +- radius, 0 <= num <= den / 2."""
    if num == 0:
        return Fraction(1), Fraction(0)
    if 2 * num == den:
        return Fraction(-1), Fraction(0)
    tlo, thi = 2 * PI[0] * num / den, 2 * PI[1] * num / den
    t, rt = (tlo + thi) / 2, (thi - tlo) / 2
    s, term, j = Fraction(0), Fraction(1), 0
    while True:
        s += term
        nxt = -term * t * t / ((2 * j + 1) * (2 * j + 2))
        j += 1
        if abs(nxt) < Fraction(1, 2 ** (BITS + 60)):
            return s, abs(nxt) + rt  # |R| <= t^(2J) / (2J)!; |cos t - cos t'| <= |t - t'|
        term = nxt


def _round_dyadic(c: Fraction, r: Fraction):
    K = 2**BITS
    return Fraction(round(c * K), K), r + Fraction(1, 2 * K)


def tstar(n: int, N: int):
    """Exact T*_{l,i} = sin^2(i pi l / n) = (1 - cos(2 pi i l / n)) / 2 as centres (columns) and one radius for all entries."""
    cache, cols, rad = {}, [], Fraction(0)
    for i in range(1, N + 1):
        col = []
        for l in range(1, n):  # noqa: E741 - the node index of the policy
            k = (i * l) % n
            k = min(k, n - k)
            if k not in cache:
                c, r = _cos2pi(k, n)
                cache[k] = _round_dyadic((1 - c) / 2, r / 2) if k else (Fraction(0), Fraction(0))
            col.append(cache[k][0])
            rad = max(rad, cache[k][1])
        cols.append(col)
    return cols, rad


# --- the exact discrete half-step (Policy-23 2; ВБC-13 p. 1) -------------------------------------------------------------------------

def coefficients(cols: list, length: Fraction, n: int, q: Fraction):
    """A, B, C (= h U^T D4 U), F of the columns ``cols`` in exact arithmetic, and L U, D4 U by columns."""
    h, m, N = length / n, n - 1, len(cols)
    z = [[Fraction(0), *c, Fraction(0)] for c in cols]

    def lap(v, i):
        return (v[i - 1] - 2 * v[i] + v[i + 1]) / h**2

    def d4(v, i):
        if i == 1:
            return (7 * v[1] - 4 * v[2] + v[3]) / h**4
        if i == m:
            return (7 * v[m] - 4 * v[m - 1] + v[m - 2]) / h**4
        return (v[i - 2] - 4 * v[i - 1] + 6 * v[i] - 4 * v[i + 1] + v[i + 2]) / h**4

    Lz = [[lap(c, i) for i in range(1, n)] for c in z]
    Dz = [[d4(c, i) for i in range(1, n)] for c in z]
    A = [[h * sum(x * y for x, y in zip(cols[i], cols[j], strict=True)) for j in range(N)] for i in range(N)]
    B = [[-h * sum(x * y for x, y in zip(cols[i], Lz[j], strict=True)) for j in range(N)] for i in range(N)]
    C = [[h * sum(x * y for x, y in zip(cols[i], Dz[j], strict=True)) for j in range(N)] for i in range(N)]
    F = [q * h * sum(cols[i]) for i in range(N)]
    return A, B, C, F, Lz, Dz


def system(A: list, B: list, C: list, F: list, length: Fraction, n: int, D: Fraction):
    """K = D [A (x) D4 - 2 B (x) L + C (x) I], rhs = F (x) 1; node-first index p N + i; half bandwidth 3N - 1."""
    h, m, N = length / n, n - 1, len(A)

    def d4(p, r):
        d = abs(p - r)
        if d == 0:
            return Fraction(7 if p in (0, m - 1) else 6) / h**4
        return Fraction(-4) / h**4 if d == 1 else (Fraction(1) / h**4 if d == 2 else Fraction(0))

    def lap(p, r):
        d = abs(p - r)
        return Fraction(-2) / h**2 if d == 0 else (Fraction(1) / h**2 if d == 1 else Fraction(0))

    K = {}
    for p in range(m):
        for r in range(max(0, p - 2), min(m, p + 3)):
            for i in range(N):
                for j in range(N):
                    v = D * (A[i][j] * d4(p, r) - 2 * B[i][j] * lap(p, r) + (C[i][j] if p == r else 0))
                    if v:
                        K[(p * N + i, r * N + j)] = v
    return K, [F[i] for p in range(m) for i in range(N)], m * N, 3 * N - 1


def band_ldl(K: dict, n: int, w: int, shift: Fraction = Fraction(0), rhs=None):
    """Exact symmetric elimination without pivoting in the band of K - shift I: (all pivots > 0, i.e. K - shift I > 0; solution or None)."""
    rows = [{j: K[(i, j)] for j in range(max(0, i - w), min(n, i + w + 1)) if (i, j) in K} for i in range(n)]
    for i in range(n):
        rows[i][i] = rows[i].get(i, Fraction(0)) - shift
    b = list(rhs) if rhs is not None else None
    for k in range(n):
        piv = rows[k][k]
        if piv <= 0:
            return False, None
        for i in range(k + 1, min(n, k + w + 1)):
            f = rows[i].get(k)
            if not f:
                continue
            f = f / piv
            for j, v in rows[k].items():
                if j >= k:
                    rows[i][j] = rows[i].get(j, Fraction(0)) - f * v
            if b is not None:
                b[i] -= f * b[k]
    if b is None:
        return True, None
    x = [Fraction(0)] * n
    for i in reversed(range(n)):
        x[i] = (b[i] - sum(v * x[j] for j, v in rows[i].items() if j > i)) / rows[i][i]
    return True, x


def _dense_pd(M: list, shift: Fraction = Fraction(0)) -> bool:
    n = len(M)
    return band_ldl({(i, j): M[i][j] for i in range(n) for j in range(n)}, n, n, shift)[0]


_TRIALS = (1 - 1e-9, 1 - 1e-6, 1 - 1e-3, 0.9, 0.5, 0.1)


def _lambda_low(K: dict, n: int, w: int, estimate: float):
    """The largest tried rational s > 0 with K - s I > 0 proved exactly; 0 if only K > 0 is proved; None if K > 0 is not proved."""
    if math.isfinite(estimate):
        for f in _TRIALS:
            s = Fraction(estimate * f)
            if s > 0 and band_ldl(K, n, w, s)[0]:
                return s
    return Fraction(0) if band_ldl(K, n, w)[0] else None


def _float_matrix(K: dict, n: int) -> np.ndarray:
    M = np.zeros((n, n))
    for (i, j), v in K.items():
        M[i, j] = float(v)
    return M


def _floats(M: list) -> np.ndarray:
    return np.array([[float(v) for v in row] for row in M])


def _sigma_low_squared(GU: list, GV: list, estimate: float):
    """sigma_N^2 = lambda_min(GU GV) >= s proved by GV - s GU^-1 > 0 (congruence with GU^1/2), GU > 0 proved first; None otherwise."""
    if not math.isfinite(estimate) or estimate <= 0 or not _dense_pd(GU):
        return None
    GUi = inverse(GU)
    for f in _TRIALS:
        s = Fraction(estimate * f)
        if s > 0 and _dense_pd([[GV[i][j] - s * GUi[i][j] for j in range(len(GV))] for i in range(len(GV))]):
            return s
    return None


# --- RN-5 -------------------------------------------------------------------------------------------------------------------------------

def _closed_forms_consistent(Tc, rho, mu_c, mu_r, la, hx, m, N, q, A_T, F_T, Bc, Br, Cc, Cr, coeffs_of_centres):
    """ВБC-13 p. 3 closed forms against the coefficients of the enclosure centres, within the propagated radius (an independent check)."""
    A_c, B_c, C_c, F_c, LT, DT = coeffs_of_centres
    rowsum = {"A": Fraction(1), "B": 4 / hx**2, "C": 16 / hx**4}
    vectors = {"A": Tc, "B": [[-v for v in c] for c in LT], "C": DT}
    for name, computed, closed, radius in (("A", A_c, A_T, lambda i, j: 0), ("B", B_c, Bc, lambda i, j: Br if i == j else 0),
                                           ("C", C_c, Cc, lambda i, j: Cr[i] if i == j else 0)):
        Mv = vectors[name]
        for i in range(N):
            for j in range(N):
                pert = hx * (rho * sum(abs(v) for v in Mv[j]) + rho * sum(abs(v) for v in Mv[i]) + rho * rho * m * rowsum[name])
                if abs(computed[i][j] - closed[i][j]) > pert + radius(i, j):
                    return False
    return all(abs(F_c[i] - F_T[i]) <= q * hx * m * rho for i in range(N))


def rn5_certificate(a: float, b: float, n_x: int, n_y: int, N: int, G, V, D: float, q: float) -> dict:
    """RN-5 on the saved s = 1 factors G (the given set after QR, (n_x - 1, N)) and V (solved, (n_y - 1, N)) of the actual run.

    Returns {"certified": bool, "verdict", "reason", "hypotheses", "values"} and, for the use of RT-7, the exact "eps" (a bound of
    ||(w_hat - w_h(1))_w||_2) and "sigma_low" (a lower bound of sigma_N(w_hat_w)); both None when not certified.
    """
    out = {"certified": False, "verdict": "RN5_NOT_COMPLETED", "reason": None, "hypotheses": {}, "values": {}, "eps": None, "sigma_low": None}
    hyp = out["hypotheses"]
    hyp["2N < n_x (ВБC-13 p. 3)"] = 2 * N < n_x
    hyp["N <= floor(n_x/2) (ВБC-13 p. 4)"] = N <= n_x // 2
    hyp["N <= floor(n_y/2) (RN-3)"] = N <= n_y // 2
    if not all(hyp.values()):
        out["verdict"], out["reason"] = "RN5_HYPOTHESIS_NOT_ESTABLISHED", "условие области сертификата не выполнено"
        return out
    try:
        Gf = exact_matrix(G, n_x - 1, N)
        Vf = exact_matrix(V, n_y - 1, N)
        _certify(out, Fraction(a), Fraction(b), n_x, n_y, N, Gf, Vf, Fraction(D), Fraction(q))
    except CertificateInputError as exc:
        out["verdict"], out["reason"] = "RN5_INVALID_INPUT", str(exc)
    except _COMPUTATION_ERRORS as exc:
        out["verdict"], out["reason"] = "RN5_NOT_COMPLETED", f"{type(exc).__name__}: {exc}"
    if not out["certified"]:
        out["eps"] = out["sigma_low"] = None
    return out


def _certify(out: dict, la: Fraction, lb: Fraction, nx: int, ny: int, N: int, Gf: list, Vf: list, D: Fraction, q: Fraction) -> None:
    hyp, val = out["hypotheses"], out["values"]
    hx, hy, m = la / nx, lb / ny, nx - 1
    Tc, rho = tstar(nx, N)
    mu_c = [4 / hx**2 * Tc[i][0] for i in range(N)]
    mu_r = 4 / hx**2 * rho
    # closed forms of ВБC-13 p. 3 at the exact T*
    A_T = [[la * (Fraction(1, 8) * (i == j) + Fraction(1, 4)) for j in range(N)] for i in range(N)]
    F_T = [q * la / 2] * N
    Bc = [[la / 8 * mu_c[i] if i == j else Fraction(0) for j in range(N)] for i in range(N)]
    Br = la / 8 * mu_r
    Cc = [[la / 8 * mu_c[i] ** 2 if i == j else Fraction(0) for j in range(N)] for i in range(N)]
    Cr = [la / 8 * (2 * mu_c[i] * mu_r + mu_r**2) for i in range(N)]
    hyp["closed forms consistent with the enclosure"] = _closed_forms_consistent(
        Tc, rho, mu_c, mu_r, la, hx, m, N, q, A_T, F_T, Bc, Br, Cc, Cr, coefficients(Tc, la, nx, q))
    if not hyp["closed forms consistent with the enclosure"]:
        out["verdict"], out["reason"] = "RN5_HYPOTHESIS_NOT_ESTABLISHED", "точные формулы ВБC-13, п. 3 не согласуются с оболочкой"
        return
    Gc = transpose(Gf)  # columns
    # R = (T*^T T*)^-1 T*^T U~ at the centres (a rational point of the interval R); U' = T* R lies in span T*; E = U~ - U'
    TtT = [[sum(x * y for x, y in zip(Tc[i], Tc[j], strict=True)) for j in range(N)] for i in range(N)]
    TtG = [[sum(x * y for x, y in zip(Tc[i], Gc[j], strict=True)) for j in range(N)] for i in range(N)]
    R = matmul(inverse(TtT), TtG)
    Er = [[Gc[i][l] - sum(Tc[k][l] * R[k][i] for k in range(N)) for l in range(m)] for i in range(N)]  # noqa: E741
    Erad = [rho * sum(abs(R[k][i]) for k in range(N)) for i in range(N)]
    EF2 = sum((abs(v) + Erad[i]) ** 2 for i in range(N) for v in Er[i])
    A_U, B_U, C_U, F_U, _, _ = coefficients(Gc, la, nx, q)
    Rt = transpose(R)
    A1 = matmul(matmul(Rt, A_T), R)
    F1 = [sum(R[k][i] * F_T[k] for k in range(N)) for i in range(N)]
    B1 = [[la / 8 * sum(R[k][i] * mu_c[k] * R[k][j] for k in range(N)) for j in range(N)] for i in range(N)]
    B1r = [[la / 8 * sum(abs(R[k][i] * R[k][j]) for k in range(N)) * mu_r for j in range(N)] for i in range(N)]
    C1 = [[la / 8 * sum(R[k][i] * mu_c[k] ** 2 * R[k][j] for k in range(N)) for j in range(N)] for i in range(N)]
    C1r = [[la / 8 * sum(abs(R[k][i] * R[k][j]) * (2 * mu_c[k] * mu_r + mu_r**2) for k in range(N)) for j in range(N)] for i in range(N)]
    nA = sqrt_up(sum((A_U[i][j] - A1[i][j]) ** 2 for i in range(N) for j in range(N)))
    nB = sqrt_up(sum((abs(B_U[i][j] - B1[i][j]) + B1r[i][j]) ** 2 for i in range(N) for j in range(N)))
    nC = sqrt_up(sum((abs(C_U[i][j] - C1[i][j]) + C1r[i][j]) ** 2 for i in range(N) for j in range(N)))
    dFn = sqrt_up(sum((F_U[i] - F1[i]) ** 2 for i in range(N)) * (ny - 1))
    nK = D * (nA * 16 / hy**4 + 2 * nB * 4 / hy**2 + nC)  # ||dK||_2, ||D4|| <= 16/h^4, ||L|| <= 4/h^2
    K, rhs, n, w = system(A_U, B_U, C_U, F_U, lb, ny, D)
    lam = _lambda_low(K, n, w, float(np.linalg.eigvalsh(_float_matrix(K, n))[0]))
    val.update(norm_dA=nA, norm_dB=nB, norm_dC=nC, norm_dF=dFn, norm_dK=nK, lambda_low=lam, norm_E_F=sqrt_up(EF2), rho_T=rho)
    hyp["lambda_low(K(U~)) > ||dK||_2"] = lam is not None and lam > nK
    if not hyp["lambda_low(K(U~)) > ||dK||_2"]:
        out["verdict"], out["reason"] = "RN5_UNRESOLVED", "λ_low(𝒦(Ũ)) не больше ‖δ𝒦‖₂"
        return
    _, xs = band_ldl(K, n, w, Fraction(0), rhs)
    xs_n = sqrt_up(sum(v * v for v in xs))
    vhat = [Vf[p][i] for p in range(ny - 1) for i in range(N)]
    dxh = sqrt_up(sum((x - y) ** 2 for x, y in zip(vhat, xs, strict=True)))
    dprime = (dFn + nK * xs_n) / (lam - nK)
    sx, sy = sqrt_up(hx), sqrt_up(hy)
    UW = sqrt_up(max(sum(abs(v) for v in row) for row in A_U))  # ||U~_w||_2^2 = lambda_max(A_U) <= Gershgorin
    eps1 = sx * sqrt_up(EF2) * sy * (xs_n + dprime) + UW * sy * dprime  # ||(w_h(1) - W*)_w||_2
    eps2 = UW * sy * dxh  # ||(w_hat - W*)_w||_2
    eps = eps1 + eps2
    GV = [[hy * sum(Vf[p][i] * Vf[p][j] for p in range(ny - 1)) for j in range(N)] for i in range(N)]
    with np.errstate(all="ignore"):
        estimate = float(np.min(np.linalg.eigvals(_floats(matmul(A_U, GV))).real))
    s2 = _sigma_low_squared(A_U, GV, estimate)
    sigma_low = sqrt_dn(s2) if s2 is not None else Fraction(0)
    val.update(norm_x_star=xs_n, dx_prime=dprime, dx_hat=dxh, norm_Uw=UW, eps1=eps1, eps2=eps2, eps=eps, sigma_low=sigma_low)
    hyp["sigma_N(w_hat_w)^2 > eps^2 (Weyl)"] = s2 is not None and s2 > eps * eps
    if not hyp["sigma_N(w_hat_w)^2 > eps^2 (Weyl)"]:
        out["verdict"], out["reason"] = "RN5_UNRESOLVED", "σ_N(ŵ_w) не больше ε: неравенство Вейля не доказано"
        return
    val["margin_sigma_over_eps"] = sigma_low / eps
    out.update(certified=True, verdict="RN5_CERTIFIED", reason="rank w_h^(1) = N", eps=eps, sigma_low=sigma_low)


# --- RT-7 ------------------------------------------------------------------------------------------------------------------------------

def rt7_conditions(V, G2, rn5: dict) -> dict:
    """R7-1...R7-3 of D-065, 6 on the actual saved V (before QR) and the actual G2 (after QR).

    tau = tr(I - GV^-1 V^T G2 GG^-1 G2^T V), GV = V^T V, GG = G2^T G2, exact; t = eps / sigma_low of RN-5. Passed means
    sin theta_max(span G2, U2*) <= sqrt(tau) + t < 1 (Euclidean transversality only).
    """
    out = {"passed": False, "R7-1": False, "R7-2": False, "R7-3": False, "reason": None, "ranks": None, "t": None, "tau": None,
           "sin_theta_upper": None}
    try:
        Vf, Gf = exact_matrix(V), exact_matrix(G2)
        if len(Vf) != len(Gf) or len(Vf[0]) != len(Gf[0]):
            raise CertificateInputError("формы V и G2 различны")
        N = len(Vf[0])
        out["ranks"] = {"V": [rank_fraction(Vf), rank_modp(Vf)], "G2": [rank_fraction(Gf), rank_modp(Gf)]}
        out["R7-1"] = all(r == N for ranks in out["ranks"].values() for r in ranks)
        eps, sigma_low = rn5.get("eps"), rn5.get("sigma_low")
        if rn5.get("certified") and isinstance(eps, Fraction) and isinstance(sigma_low, Fraction) and sigma_low > 0:
            out["t"] = eps / sigma_low
            out["R7-2"] = 0 <= out["t"] < 1
        if not out["R7-1"]:
            out["reason"] = "R7-1: точный ранг V или G2 меньше N или процедуры расходятся"
            return out
        Vt, Gt = transpose(Vf), transpose(Gf)
        M = matmul(matmul(inverse(matmul(Vt, Vf)), matmul(Vt, Gf)), matmul(inverse(matmul(Gt, Gf)), matmul(Gt, Vf)))
        out["tau"] = sum(1 - M[i][i] for i in range(N))
        if not out["R7-2"]:
            out["reason"] = "R7-2: RN-5 не выполнен на этом расчёте или не 0 ≤ t < 1"
            return out
        out["R7-3"] = out["tau"] < (1 - out["t"]) ** 2
        out["sin_theta_upper"] = sqrt_up(out["tau"]) + out["t"]
        if not out["R7-3"]:
            out["reason"] = "R7-3: τ не меньше (1 − t)²"
            return out
    except CertificateInputError as exc:
        out["reason"] = f"недопустимый вход: {exc}"
        return out
    except _COMPUTATION_ERRORS as exc:
        out["reason"] = f"не завершено: {type(exc).__name__}: {exc}"
        return out
    out["passed"] = True
    return out
