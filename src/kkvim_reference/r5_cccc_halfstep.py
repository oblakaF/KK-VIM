"""QA controls of the CCCC half-step system (R5-V, stage Q1b): R-H, R-AC, R-HP.

Independent of the main solver: no import of ``kkvim`` (D-028); nothing of Builder is used. Basis: D-049.1, D-049.2;
docs/R5_NUMERICAL_SPEC_DRAFT.md (NS) 4.2-4.3, 5; plan docs/R5_CCCC_PREQUALIFICATION.plan.md (PL) 2.1, 7.

Half-step system (NS 4.3, ChF-C1/C2). Given set G (n_f - 1) x N on the fixed grid G(l_f, n_f), unknown set S (n_s - 1) x N on the
solve grid G(l_s, n_s), h_f = l_f/n_f, h_s = l_s/n_s:

    A = h_f G^T G,  B = -h_f G^T L_f G,  C = h_f G^T D4_f G,  F = q h_f G^T 1,
    K = D [A (x) D4_s - 2 B (x) L_s + C (x) I_s],   K vec(S) = F (x) 1,      (term index first: vec(S) = [S_:1; S_:2; ...])

L = tridiag(1, -2, 1)/h^2 (Dirichlet), D4 = L^2 + (2/h^4)(e_1 e_1^T + e_{n-1} e_{n-1}^T), rows (7, -4, 1)/h^4 at both ends.
Y-half-step: fixed = x (l_x, n_x), solve = y (l_y, n_y); X-half-step: fixed = y, solve = x.

Three independent computational paths (PL 2.1):
- **R-H** (``hessian_system``): restriction of the Hessian of the full discrete functional Pi_h^C to the half-step subspace,
  K_H = (D h_x h_y / h_s) P^T A2 P, rhs_H = (q h_x h_y / h_s) P^T 1, with A2 = Hess Pi_h^C/(D h_x h_y) assembled from the
  trapezoid weights and the even-ghost second difference (kkvim_reference.r5_cccc_d2.energy_hessian, Proposition D) and P the
  embedding of the half-step unknowns into the nodal field (G (x) I for Y, [I (x) G_:k]_k for X);
- **R-AC** (``solve_ac``): coefficients A, B, C, F in floating point, the block-pentadiagonal SPD matrix in the node-term ordering
  (index j N + i, half bandwidth 3N - 1; U-R1), banded Cholesky and iterative refinement with the R-HP residual of the ORIGINAL
  operator; the last correction is an ESTIMATE of the remaining error, not a bound;
- **R-HP** (``residual_exact``): F (x) 1 - K vec(S) in exact rational arithmetic (fractions), coefficients from the given floats,
  D4 applied node by node by its stencil with the (7, -4, 1) boundary rows, no modal transformation (NS 5, "Control").

A posteriori bound (QA, level B; review 57d3194, I-1...I-4). Preconditions: K is the matrix of the EXACT system built from the
input doubles (exact coefficients, D4 = L^2 + E_2 with n_f, n_s >= 3), s* its exact solution, s^ any double vector.
1. ``certify_lambda_min``: a Cholesky factor R^ of K - t I (t the exact double tau) computed in decimal arithmetic with any rounding;
   the defect E = R^T R^ - (K - t I) is then computed EXACTLY in rational arithmetic. R^ upper triangular with a positive diagonal
   gives R^T R^ > 0, hence lambda_min(K - t I) > -||E||_2 >= -||E||_inf and lambda_min(K) > t - ||E||_inf (exact rational number),
   converted to double downwards. No a priori rounding model is used; a failed factorization, a non-positive or subnormal bound
   gives None (no certificate).
2. ``error_bound``: ||s^ - s*||_2 = ||K^-1 r||_2 <= ||r||_2 / lambda_low, r the exact residual (R-HP); ||r||_2 = m sqrt(rho) with
   m = max|r_i| and rho = sum (r_i/m)^2 exact, sqrt(rho) bounded above by an integer square root, the result converted to double
   upwards (math.inf on overflow).
3. ``relative_bounds`` (R-4, review 4f92c82): L = ``norm_lower(s^)`` <= ||s^||_2 from the exact sum of squares (integer square
   root downwards); b/L >= ||s^ - s*||_2/||s^||_2 if L > 0, and b/(L - b) >= ||s^ - s*||_2/||s*||_2 if L > b (since
   ||s*||_2 >= ||s^||_2 - b >= L - b > 0); exact quotients converted upwards, otherwise None (UNRESOLVED).
4. Printing: ``decimal_up`` / ``decimal_down`` give decimal strings rounded only upwards / downwards from the exact double.
The bound is a bound for the SET S of the half-step (vector of unknowns), not for the field W = G S^T: for the field add the factor
||G||_2 and the rounding of the product (QA report, section 4). It needs an independent given set (exact rank N), otherwise the
system is singular and no bound exists (status, not a regularization).
"""

import math
from dataclasses import dataclass, field
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal, getcontext, localcontext
from fractions import Fraction

import numpy as np
import scipy.linalg

from kkvim_reference import r5_cccc_d2 as d2

U = 2.0**-53
DIRECTIONS = ("Y", "X")
EXACTLY_DEPENDENT = "EXACTLY_DEPENDENT_GIVEN_SET"
INDEPENDENT = "INDEPENDENT_GIVEN_SET"
FACTORIZATION_FAILED = "NUMERICAL_FACTORIZATION_FAILED"  # independent set, but the floating-point factorization failed


@dataclass(frozen=True)
class HalfStep:
    """Half-step on the plate l_x x l_y with grids n_x, n_y; ``given`` is the fixed set (n_f - 1) x N."""

    lx: float
    nx: int
    ly: float
    ny: int
    direction: str
    given: np.ndarray = field(compare=False)
    D: float = 1.0
    q: float = 1.0

    def __post_init__(self):
        if self.direction not in DIRECTIONS:
            raise ValueError(f"direction must be one of {DIRECTIONS}")
        g = np.asarray(self.given, dtype=float)
        if g.ndim == 1:
            g = g[:, None]
        if g.shape[0] != self.n_f - 1 or not 1 <= g.shape[1] <= 3:
            raise ValueError(f"given set must be (n_f - 1) x N, N = 1..3, got {g.shape}")
        if not np.all(np.isfinite(g)):
            raise ValueError("given set must be finite")
        for name in ("nx", "ny"):  # I-4: with one interior node both corner terms of D4 fall on one row (8/h^4, not 7/h^4)
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v < 3:
                raise ValueError(f"{name} must be an integer >= 3 (at least two interior nodes), got {v!r}")
        object.__setattr__(self, "given", g)

    @property
    def N(self) -> int:
        return self.given.shape[1]

    @property
    def n_f(self) -> int:
        return self.nx if self.direction == "Y" else self.ny

    @property
    def n_s(self) -> int:
        return self.ny if self.direction == "Y" else self.nx

    @property
    def l_f(self) -> float:
        return self.lx if self.direction == "Y" else self.ly

    @property
    def l_s(self) -> float:
        return self.ly if self.direction == "Y" else self.lx

    @property
    def size(self) -> int:
        return self.N * (self.n_s - 1)


# --- floating-point operators (coefficient path) -----------------------------------------------------------------

def _L(n: int, h: float) -> np.ndarray:
    m = n - 1
    return (np.diag(-2.0 * np.ones(m)) + np.diag(np.ones(m - 1), 1) + np.diag(np.ones(m - 1), -1)) / h**2


def _D4(n: int, h: float) -> np.ndarray:
    L = _L(n, h)
    D4 = L @ L
    D4[0, 0] += 2.0 / h**4
    D4[-1, -1] += 2.0 / h**4
    return D4


def coefficients(hs: HalfStep):
    """A, B, C, F in floating point (NS 4.2)."""
    h = hs.l_f / hs.n_f
    G = hs.given
    A = h * (G.T @ G)
    B = -h * (G.T @ (_L(hs.n_f, h) @ G))
    C = h * (G.T @ (_D4(hs.n_f, h) @ G))
    F = hs.q * h * G.sum(axis=0)
    return A, B, C, F


def coefficient_system(hs: HalfStep):
    """K and rhs of ChF-C1/C2 from the coefficients (term-first ordering), floating point."""
    A, B, C, F = coefficients(hs)
    hsol = hs.l_s / hs.n_s
    m = hs.n_s - 1
    K = hs.D * (np.kron(A, _D4(hs.n_s, hsol)) - 2.0 * np.kron(B, _L(hs.n_s, hsol)) + np.kron(C, np.eye(m)))
    return K, np.repeat(F, m)


# --- R-H: restriction of the Hessian of Pi_h^C ---------------------------------------------------------------------

def embedding(hs: HalfStep) -> np.ndarray:
    """P: half-step unknowns (term-first) -> nodal field in the D2 ordering k = (i - 1)(n_y - 1) + (j - 1)."""
    G = hs.given
    if hs.direction == "Y":
        return np.kron(G, np.eye(hs.ny - 1))
    return np.hstack([np.kron(np.eye(hs.nx - 1), G[:, [k]]) for k in range(hs.N)])


def hessian_system(hs: HalfStep):
    """R-H: K_H = (D h_x h_y / h_s) P^T A2 P and rhs_H = (q h_x h_y / h_s) P^T 1 (PL 2.1; NS 4.3)."""
    A2 = d2.energy_hessian(d2.D2Problem(hs.lx, hs.ly, hs.nx, hs.ny)).toarray()
    P = embedding(hs)
    hx, hy = hs.lx / hs.nx, hs.ly / hs.ny
    hsol = hs.l_s / hs.n_s
    K = (hs.D * hx * hy / hsol) * (P.T @ (A2 @ P))
    rhs = (hs.q * hx * hy / hsol) * (P.T @ np.ones(P.shape[0]))
    return K, rhs


# --- exact rational path (R-HP) -----------------------------------------------------------------------------------

def _apply_L_exact(f, h2):
    n1 = len(f)
    out = []
    for i in range(n1):
        left = f[i - 1] if i > 0 else 0
        right = f[i + 1] if i + 1 < n1 else 0
        out.append((left - 2 * f[i] + right) / h2)
    return out


def _apply_D4_exact(f, h4):
    """D4 f by its stencil: interior (1, -4, 6, -4, 1), boundary rows (7, -4, 1) and (1, -4, 7) (even ghost node)."""
    n1 = len(f)
    get = lambda k: f[k] if 0 <= k < n1 else 0  # noqa: E731  (boundary nodes w = 0)
    out = []
    for i in range(n1):
        diag = 7 if i in (0, n1 - 1) else 6
        out.append((get(i - 2) - 4 * get(i - 1) + diag * f[i] - 4 * get(i + 1) + get(i + 2)) / h4)
    return out


def exact_coefficients(hs: HalfStep):
    """A, B, C, F of the given floats in exact rational arithmetic."""
    h = Fraction(hs.l_f) / hs.n_f
    G = [[Fraction(float(v)) for v in hs.given[:, k]] for k in range(hs.N)]
    LG = [_apply_L_exact(g, h * h) for g in G]
    D4G = [_apply_D4_exact(g, h**4) for g in G]
    dot = lambda a, b: sum(x * y for x, y in zip(a, b))  # noqa: E731
    A = [[h * dot(G[i], G[k]) for k in range(hs.N)] for i in range(hs.N)]
    B = [[-h * dot(G[i], LG[k]) for k in range(hs.N)] for i in range(hs.N)]
    C = [[h * dot(G[i], D4G[k]) for k in range(hs.N)] for i in range(hs.N)]
    F = [Fraction(hs.q) * h * sum(G[i]) for i in range(hs.N)]
    return A, B, C, F


def residual_exact(hs: HalfStep, s, coeffs=None) -> list[Fraction]:
    """R-HP: F (x) 1 - K vec(S) exactly (term-first ordering of ``s``)."""
    A, B, C, F = coeffs or exact_coefficients(hs)
    m = hs.n_s - 1
    hsol = Fraction(hs.l_s) / hs.n_s
    D = Fraction(hs.D)
    cols = [[Fraction(float(v)) for v in s[k * m:(k + 1) * m]] for k in range(hs.N)]
    Ls = [_apply_L_exact(c, hsol * hsol) for c in cols]
    D4s = [_apply_D4_exact(c, hsol**4) for c in cols]
    r = []
    for i in range(hs.N):
        for j in range(m):
            acc = sum(A[i][k] * D4s[k][j] - 2 * B[i][k] * Ls[k][j] + C[i][k] * cols[k][j] for k in range(hs.N))
            r.append(F[i] - D * acc)
    return r


def eta_qa(hs: HalfStep, s, coeffs=None) -> float:
    """NS 5 normalisation with h of the solved direction: ||r||_inf / (D(16||A||/h^4 + 8||B||/h^2 + ||C||)||S||_inf + ||F||_inf)."""
    coeffs = coeffs or exact_coefficients(hs)
    A, B, C, F = coeffs
    r = residual_exact(hs, s, coeffs)
    h = Fraction(hs.l_s) / hs.n_s
    norm = lambda M: max(sum(abs(x) for x in row) for row in M)  # noqa: E731
    scale = Fraction(hs.D) * (16 * norm(A) / h**4 + 8 * norm(B) / h**2 + norm(C)) * Fraction(float(np.max(np.abs(s)))) \
        + max(abs(f) for f in F)
    return float(max(abs(x) for x in r) / scale)


def exact_rank(M: np.ndarray) -> int:
    """Rank of the float matrix in exact rational arithmetic (no threshold)."""
    rows = [[Fraction(float(v)) for v in row] for row in np.asarray(M)]
    rank, ncol = 0, len(rows[0])
    for c in range(ncol):
        pivot = next((r for r in range(rank, len(rows)) if rows[r][c] != 0), None)
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        for r in range(len(rows)):
            if r != rank and rows[r][c] != 0:
                f = rows[r][c] / rows[rank][c]
                rows[r] = [a - f * b for a, b in zip(rows[r], rows[rank])]
        rank += 1
    return rank


# --- certified lambda_min and the a posteriori bound -------------------------------------------------------------

def _exact_band_matrix(hs: HalfStep, coeffs):
    """Exact K in the node-term ordering as {(row, col): Fraction} with |row - col| <= 3N - 1 (upper part incl. diagonal)."""
    A, B, C, _ = coeffs
    N, m = hs.N, hs.n_s - 1
    h = Fraction(hs.l_s) / hs.n_s
    D = Fraction(hs.D)
    d4 = {0: Fraction(6), 1: Fraction(-4), 2: Fraction(1)}
    entries = {}
    for j in range(m):
        for dj in (0, 1, 2):
            jj = j + dj
            if jj >= m:
                continue
            v4 = (Fraction(7) if (dj == 0 and j in (0, m - 1)) else d4[dj]) / h**4
            vl = {0: Fraction(-2), 1: Fraction(1), 2: Fraction(0)}[dj] / h**2
            vi = Fraction(1) if dj == 0 else Fraction(0)
            for i in range(N):
                for k in range(N):
                    row, col = j * N + i, jj * N + k
                    if col < row:
                        continue
                    val = D * (A[i][k] * v4 - 2 * B[i][k] * vl + C[i][k] * vi)
                    if val != 0:
                        entries[(row, col)] = val
    return entries


def _float_down(x: Fraction) -> float:
    """Largest double <= x (x finite, representable range)."""
    f = float(x)
    while Fraction(f) > x:
        f = math.nextafter(f, -math.inf)
    return f


def _float_up(x: Fraction) -> float:
    """Smallest double >= x; math.inf if x exceeds the largest double."""
    try:
        f = float(x)
    except OverflowError:
        return math.inf
    while math.isfinite(f) and Fraction(f) < x:
        f = math.nextafter(f, math.inf)
    return f


def certify_lambda_min(hs: HalfStep, tau: float, coeffs=None, digits: int = 50) -> float | None:
    """Certified lower bound of lambda_min(K) of the exact system (I-2, review 57d3194).

    t = Fraction(tau) exactly; M = K - t I exactly (node-term band, half bandwidth p = 3N - 1); a banded Cholesky factor R^ of M is
    computed in decimal arithmetic (``digits`` digits, any rounding); the defect E = R^T R^ - M is computed exactly (fractions).
    R^ upper triangular with a positive diagonal => R^T R^ > 0 => lambda_min(K) > t - ||E||_inf. Returns the largest double not above
    t - ||E||_inf, or None if tau is not finite and positive, the factorization fails, or the bound is not a positive normal double."""
    if not (isinstance(tau, (int, float)) and math.isfinite(tau) and tau > 0):
        return None
    coeffs = coeffs or exact_coefficients(hs)
    entries = _exact_band_matrix(hs, coeffs)
    n = hs.size
    p = 3 * hs.N - 1
    t = Fraction(float(tau))
    M = dict(entries)
    for i in range(n):
        M[(i, i)] = M.get((i, i), Fraction(0)) - t
    R = {}
    with localcontext() as ctx:
        ctx.prec = digits
        a = {key: Decimal(v.numerator) / Decimal(v.denominator) for key, v in M.items()}
        for i in range(n):
            s = a.get((i, i), Decimal(0)) - sum(R[(k, i)] * R[(k, i)] for k in range(max(0, i - p), i) if (k, i) in R)
            if s <= 0:
                return None
            rii = s.sqrt()
            if rii <= 0:
                return None
            R[(i, i)] = rii
            for j in range(i + 1, min(n, i + p + 1)):
                s = a.get((i, j), Decimal(0)) - sum(R[(k, i)] * R[(k, j)] for k in range(max(0, j - p), i)
                                                    if (k, i) in R and (k, j) in R)
                if s != 0:
                    R[(i, j)] = s / rii
    Rf = {key: Fraction(v) for key, v in R.items()}  # exact values of the decimal entries
    row_sum = [Fraction(0)] * n
    for i in range(n):
        for j in range(i, min(n, i + p + 1)):
            prod = sum((Rf[(k, i)] * Rf[(k, j)] for k in range(max(0, j - p), i + 1) if (k, i) in Rf and (k, j) in Rf),
                       Fraction(0))
            e = abs(prod - M.get((i, j), Fraction(0)))
            row_sum[i] += e
            if j != i:
                row_sum[j] += e
    low = t - max(row_sum)  # exact: lambda_min(K) > low
    if low <= 0:
        return None
    f = _float_down(low)
    if not (math.isfinite(f) and f >= 2.2250738585072014e-308):  # zero or subnormal: no certificate
        return None
    return f


def error_bound(hs: HalfStep, s, lambda_low: float, coeffs=None) -> float:
    """Upper bound of ||s - s*||_2 = ||K^-1 r||_2 <= ||r||_2 / lambda_low (I-1, review 57d3194).

    r is the exact residual (R-HP). ||r||_2 = m sqrt(rho), m = max|r_i|, rho = sum (r_i/m)^2 in [1, len(r)] exactly;
    sqrt(rho) <= (isqrt(P Q 4^k) + 1)/(Q 2^k) for rho = P/Q; the exact rational bound is converted to double upwards
    (math.inf on overflow). Zero residual gives 0.0. ``lambda_low`` must be a finite positive float (M-1, review 4f92c82: an int
    is refused, so that no int above 2^53 is rounded upwards on conversion)."""
    if not isinstance(lambda_low, float) or not math.isfinite(lambda_low) or lambda_low <= 0:
        raise ValueError(f"lambda_low must be a finite positive float, got {lambda_low!r}")
    r = residual_exact(hs, s, coeffs)
    m = max((abs(x) for x in r), default=Fraction(0))
    if m == 0:
        return 0.0
    rho = sum(((x / m) ** 2 for x in r), Fraction(0))
    P, Q = rho.numerator, rho.denominator
    k = 64
    sqrt_up = Fraction(math.isqrt(P * Q * 4**k) + 1, Q * 2**k)  # >= sqrt(P/Q): isqrt(z) + 1 > sqrt(z)
    return _float_up(m * sqrt_up / Fraction(lambda_low))


@dataclass(frozen=True)
class RelativeBounds:
    """Upper bounds of the relative error of the SET S (R-4, review 4f92c82); None means UNRESOLVED (condition not met)."""

    to_computed: float | None  # >= ||s^ - s*||_2 / ||s^||_2, requires L > 0
    to_exact: float | None  # >= ||s^ - s*||_2 / ||s*||_2, requires L > b


def norm_lower(s) -> Fraction:
    """Exact rational L <= ||s||_2 of a finite double vector: exact sum of squares P/Q, L = isqrt(P Q 4^k)/(Q 2^k) (0 for s = 0)."""
    v = np.asarray(s, dtype=float).ravel()
    if not np.all(np.isfinite(v)):
        raise ValueError("s must be finite")
    ss = sum((Fraction(float(x)) ** 2 for x in v), Fraction(0))
    if ss == 0:
        return Fraction(0)
    P, Q = ss.numerator, ss.denominator
    k = 64
    return Fraction(math.isqrt(P * Q * 4**k), Q * 2**k)  # isqrt(z) <= sqrt(z), sqrt(P/Q) = sqrt(z)/(Q 2^k)


def relative_bounds(b: float, s) -> RelativeBounds:
    """Relative bounds of the set S from a certified absolute bound b >= ||s - s*||_2 (``error_bound``) and the computed set s.

    L = norm_lower(s): to_computed = b/L if L > 0; to_exact = b/(L - b) if L > b, since ||s*||_2 >= ||s||_2 - b >= L - b > 0.
    The exact quotients are converted to double upwards. b = inf (no finite bound) or a failed condition gives None (UNRESOLVED);
    no number is substituted."""
    if not isinstance(b, float) or math.isnan(b) or b < 0:
        raise ValueError(f"b must be a non-negative float, got {b!r}")
    L = norm_lower(s)
    if math.isinf(b) or L == 0:
        return RelativeBounds(None, None)
    B = Fraction(b)
    return RelativeBounds(_float_up(B / L), _float_up(B / (L - B)) if L > B else None)


def _decimal_directed(x: float, digits: int, rounding: str) -> str:
    if not isinstance(x, float):
        raise ValueError(f"x must be a float, got {x!r}")
    if not math.isfinite(x):
        return repr(x)  # 'inf', '-inf', 'nan'
    if x == 0:
        return f"{0:.{digits - 1}e}"  # exact zero (Decimal would keep an exponent of the context)
    with localcontext() as ctx:
        ctx.prec = digits
        ctx.rounding = rounding
        d = ctx.plus(Decimal(x))  # Decimal(x) is the exact value of the double; one directed rounding to ``digits`` digits
    return f"{d:.{digits - 1}e}"  # d has at most ``digits`` significant digits: formatting is exact


def decimal_up(x: float, digits: int = 7) -> str:
    """Decimal string with ``digits`` significant digits whose value is >= x (rounding towards +inf only)."""
    return _decimal_directed(x, digits, ROUND_CEILING)


def decimal_down(x: float, digits: int = 7) -> str:
    """Decimal string with ``digits`` significant digits whose value is <= x (rounding towards -inf only)."""
    return _decimal_directed(x, digits, ROUND_FLOOR)


# --- solvers --------------------------------------------------------------------------------------------------------

def given_set_status(hs: HalfStep) -> str:
    return INDEPENDENT if exact_rank(hs.given) == hs.N else EXACTLY_DEPENDENT


def solve_hessian(hs: HalfStep):
    """Dense Cholesky solve of the R-H system (small grids). Returns (status, s): (EXACTLY_DEPENDENT, None) for an exactly
    dependent given set (K singular), (FACTORIZATION_FAILED, None) if the floating-point factorization fails, otherwise
    (INDEPENDENT, s). No regularization."""
    if given_set_status(hs) == EXACTLY_DEPENDENT:
        return EXACTLY_DEPENDENT, None
    K, rhs = hessian_system(hs)
    try:
        return INDEPENDENT, scipy.linalg.solve(K, rhs, assume_a="pos")
    except np.linalg.LinAlgError:
        return FACTORIZATION_FAILED, None


@dataclass
class ACSolution:
    status: str
    s: np.ndarray | None  # term-first
    s_unrefined: np.ndarray | None
    corrections: list[float] = field(default_factory=list)  # max |ds| per refinement step (estimate, not a bound)

    @property
    def e_AC(self) -> float:
        return self.corrections[-1] if self.corrections else math.inf


def _node_term(m: int, N: int) -> np.ndarray:
    """perm[p] = term-first index of node-term position p = j N + i."""
    return np.array([i * m + j for j in range(m) for i in range(N)])


def solve_ac(hs: HalfStep, max_refine: int = 5) -> ACSolution:
    """R-AC: banded Cholesky of the block-pentadiagonal SPD matrix (node-term ordering, half bandwidth 3N - 1) and iterative
    refinement with the exact residual of the original system (R-HP)."""
    if given_set_status(hs) == EXACTLY_DEPENDENT:
        return ACSolution(EXACTLY_DEPENDENT, None, None)
    K, rhs = coefficient_system(hs)
    m, N = hs.n_s - 1, hs.N
    perm = _node_term(m, N)
    Kp = K[np.ix_(perm, perm)]
    p = 3 * N - 1
    n = Kp.shape[0]
    ab = np.zeros((p + 1, n))
    for d in range(p + 1):
        ab[p - d, d:] = np.diagonal(Kp, d)
    try:
        cb = scipy.linalg.cholesky_banded(ab, lower=False)
    except np.linalg.LinAlgError:
        return ACSolution(FACTORIZATION_FAILED, None, None)
    solve = lambda b: scipy.linalg.cho_solve_banded((cb, False), b)  # noqa: E731
    x = np.empty(n)
    x[:] = solve(rhs[perm])
    s = np.empty(n)
    s[perm] = x
    out = ACSolution(INDEPENDENT, s.copy(), s.copy())
    coeffs = exact_coefficients(hs)
    previous = math.inf
    for _ in range(max_refine):
        r = np.array([float(v) for v in residual_exact(hs, s, coeffs)])
        ds = np.empty(n)
        ds[perm] = solve(r[perm])
        size = float(np.max(np.abs(ds)))
        s = s + ds
        out.corrections.append(size)
        if size >= previous or size == 0.0:
            break
        previous = size
    out.s = s
    return out


# --- start sets and diagnostics -------------------------------------------------------------------------------------

def start_set(n: int, N: int, basis: str = "sin2") -> np.ndarray:
    """T_N on the interior nodes l = 1..n-1: 'sin2' -> sin^2(i pi l/n); 'product' -> sin(pi l/n) sin((2i - 1) pi l/n) (CM VBC-15)."""
    l = np.arange(1, n)
    if basis == "sin2":
        return np.column_stack([np.sin(i * math.pi * l / n) ** 2 for i in range(1, N + 1)])
    if basis == "product":
        return np.column_stack([np.sin(math.pi * l / n) * np.sin((2 * i - 1) * math.pi * l / n) for i in range(1, N + 1)])
    raise ValueError("basis must be 'sin2' or 'product'")


def diagnostics(hs: HalfStep) -> dict:
    """Values only (no verdicts): exact rank, cond_2(A), cond_2(K) of the coefficient path, discrete start condition 2N < n_f."""
    A, _, _, F = coefficients(hs)
    eA = np.linalg.eigvalsh(A)
    K, _ = coefficient_system(hs)
    eK = np.linalg.eigvalsh(K)
    return {"N": hs.N, "direction": hs.direction, "status": given_set_status(hs), "exact_rank": exact_rank(hs.given),
            "two_N_less_than_n_f": 2 * hs.N < hs.n_f, "cond_A": float(eA[-1] / eA[0]) if eA[0] > 0 else math.inf,
            "cond_K": float(eK[-1] / eK[0]) if eK[0] > 0 else math.inf, "lambda_min_K_float": float(eK[0]),
            "F_nonzero": bool(np.any(F != 0))}


def gamma(m: int) -> float:
    """gamma_m = m u / (1 - m u) (standard bound for a floating-point dot product of length m)."""
    return m * U / (1 - m * U)
