"""Independent Navier reference: double sine series for the square SSSS plate (R2).

Analytic solution of the same 2D Kirchhoff-Love problem that KK-VIM
approximates in R2: square plate a = b, all edges simply supported (SSSS),
uniform load q, flexural rigidity D (D-015, D-024, D-025, D-033). Used by QA
for the accuracy study of KK-VIM (NUMERICAL_SPEC.md, section 16, A-10a); it is
not part of the KK-VIM solver. No grid operators, no half-steps, no energy
assembly of KK-VIM, no imports of ``kkvim`` (D-028).

Formulas (docs/NAVIER_REFERENCE_SPEC.md, sections 3-5), source
[L-107, eq. (131), p. 129; eq. (101), p. 98; eq. (3), p. 16]
references/Plate_Benchmarks_Linear_BC/1966_Timoshenko_Woinowsky-Krieger_Plates_and_Shells_Nauka.djvu.
Origin at a corner, 0 <= x, y <= a, z and q downwards. With s_m(x) =
sin(m pi x / a) and odd m, n only (even terms are exactly zero, p. 129):

    w   = (16 q a^4 / (pi^6 D)) sum_{m,n odd} s_m(x) s_n(y) / (m n (m^2 + n^2)^2),     (131)
    M_x = (16 q a^2 / pi^4) sum_{m,n odd} (m^2 + nu n^2) s_m(x) s_n(y) / (m n (m^2 + n^2)^2),
    M_y = (16 q a^2 / pi^4) sum_{m,n odd} (nu m^2 + n^2) s_m(x) s_n(y) / (m n (m^2 + n^2)^2).

(131) is printed; the moment series are the project derivation of
NAVIER_REFERENCE_SPEC.md, section 5 (term-by-term (101) applied to (131)).
D = E t^3 / (12 (1 - nu^2)) with the full thickness t [L-107, eq. (3), p. 16].

Truncation (NAVIER_REFERENCE_SPEC.md, 8.1): square truncation m, n <= M, M odd,
the same M for w, M_x, M_y; levels NAVIER_LEVELS; R2 base level M = 1279
(NUMERICAL_SPEC.md, section 16). Values are summed directly at the requested
points, never interpolated. Two summation orders (8.5, П-11):

- ``"separable"`` - first over m, then over n: w = S_x C S_y^T at grid nodes
  (NUMERICAL_SPEC.md, section 16), row by row at arbitrary points;
- ``"diagonal"`` - by increasing m + n (cost proportional to points * K^2).

Sines. At grid nodes x_i = i a / n_x the argument is reduced exactly in
integers, r = (m i) mod 2 n_x (И-11); at arbitrary points through
t = fmod(m x / a, 2) and the reflections sin(pi t) = -sin(pi (t - 1)),
sin(pi t) = sin(pi (1 - t)). On the edges x = 0, a and y = 0, a every term of
all three series vanishes identically (sin(k pi) = 0 for integer k), so the
values there are set to exactly 0.0 by that rule; float sin(m pi) is not
expected to be zero.

Error estimates, all dimensionless (w in q a^4 / D, moments in q a^2), truncation
only unless stated:

- deflection, conservative strict bound (8.2), the basis of the choice of M:
  |w - w_M| <= (16 / pi^6) lambda_3 / (8 M^2), lambda_3 = 7 zeta(3) / 8;
- deflection, refined bound (8.2, proposal; not a basis of acceptance without
  a decision of the owner): (16 / pi^6) [(ln M + 1.25) / (4 M^4) + 1 / (16 M^4)];
- moments, rough strict bound (8.3, 0 <= nu <= 1): 1 / (pi^2 M); it does not
  reach delta_M at M = 1279;
- moments, refined bound (8.3, proposal, 0 <= nu <= 1):
  (16 / pi^4) (ln M + 2) / (2 M^2);
- rounding: first-order model (constants below, [СТАНД.]), kept separately;
- practical control of convergence (8.4): absolute increments
  Delta_j = |Q(M_j) - Q(M_{j-1})| over the levels, see NavierConvergence. Small
  increments do not prove a small remainder (8.4, limitations).

Integral of the deflection and energy (MATH_SPEC.md, В-27, item 6;
NUMERICAL_SPEC.md, section 16), square a = b, truncated at the same odd M:

    J_M  = (64 / pi^8) sum_{m,n odd <= M} 1 / (m^2 n^2 (m^2 + n^2)^2),
    I_M  = (q a^6 / D) J_M        (approximation of the integral of w over the plate),
    Pi_M = -q I_M / 2             (approximation of Pi(omega_0) = -(q, omega_0) / 2),

with the conservative remainder bound of section 16 [ЧВ-22]: terms
<= 1 / (4 m^4 n^4), sum_{m > M odd} m^-4 <= 1 / (6 M^3), lambda_4 = pi^4 / 96, so

    0 <= J - J_M <= (64 / pi^8) (lambda_4 / 4) 2 / (6 M^3) = 1 / (18 pi^4 M^3).

J_M, I_M, Pi_M are values of the truncated series, not the exact limits.

Only the R2 setting (square, SSSS, q = const, linear plate) is implemented:
no rectangle b != a (R3), no M_xy, no shear forces, no P-12 (Levy cross-check).
"""

import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "DELTA_M",
    "DELTA_W",
    "M_R2",
    "NAVIER_LEVELS",
    "SUMMATIONS",
    "NavierConvergence",
    "NavierIntegral",
    "NavierPlate",
    "NavierResult",
    "deflection_integral_coef",
    "deflection_integral_tail_bound",
    "deflection_tail_bound",
    "deflection_tail_bound_refined",
    "diagnostic_levels",
    "flexural_rigidity",
    "moment_tail_bound",
    "moment_tail_bound_refined",
    "practical_control",
    "select_level",
]

U = 2.0**-53
# Levels of the truncation index M (NAVIER_REFERENCE_SPEC.md, 8.1).
NAVIER_LEVELS = (1, 3, 9, 19, 39, 79, 159, 319, 639, 1279)
# Base level of R2 by the conservative bound 8.2 (NUMERICAL_SPEC.md, section 16).
M_R2 = 1279
# Accuracy of the reference (NAVIER_REFERENCE_SPEC.md, 8.6): w in q a^4 / D,
# moments in q a^2.
DELTA_W = 4.06e-9
DELTA_M = 4.79e-6
# Apery's constant zeta(3) [СТАНД.]; lambda_3 = sum over odd n of n^-3 = 7 zeta(3) / 8.
ZETA3 = 1.2020569031595942
LAMBDA3 = 7.0 * ZETA3 / 8.0
# First-order rounding model per term c_mn s_m s_n: coefficient (m n, m^2, n^2,
# sums, square, products, quotient, constant: about 9 roundings), two sines
# (argument and library sin: about 3 each), two products and the physical
# scale: about 20 u in all. Summation of K^2 terms: 2K roundings at most for
# the separable order (two sums of K terms), 3K for the diagonal one (a sum of
# at most K terms per diagonal, 2K - 1 diagonals). At arbitrary points the
# rounded argument m x / a adds 2 pi (m + n) u |c_mn|. Margin 2.
TERM_ROUNDINGS = 20
ROUNDING_MARGIN = 2.0
SUMMATIONS = ("separable", "diagonal")
# Points evaluated together (memory of the sine tables: chunk * K floats).
_POINT_CHUNK = 2048


def _check_finite(name, value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {value!r}")


def _check_positive(name, value):
    _check_finite(name, value)
    if not value > 0:
        raise ValueError(f"{name} must be positive, got {value!r}")


def _check_nu(nu):
    _check_finite("nu", nu)
    if not -1.0 < nu < 1.0:
        raise ValueError(f"nu must lie in (-1, 1), got {nu!r}")


def _check_odd_index(M):
    if isinstance(M, bool) or not isinstance(M, int) or M < 1 or M % 2 == 0:
        raise ValueError(f"M must be a positive odd integer, got {M!r}")


def flexural_rigidity(E, thickness, nu):
    """D = E t^3 / (12 (1 - nu^2)), t - full thickness [L-107, eq. (3), p. 16]."""
    _check_positive("E", E)
    _check_positive("thickness", thickness)
    _check_nu(nu)
    return E * thickness**3 / (12.0 * (1.0 - nu * nu))


def deflection_tail_bound(M):
    """Conservative strict bound of |w - w_M| in q a^4 / D, uniform in (x, y) (8.2).

    (16 / pi^6) lambda_3 / (8 M^2); the basis of the choice of M (8.6).
    """
    _check_odd_index(M)
    return 16.0 / math.pi**6 * LAMBDA3 / (8.0 * M * M)


def deflection_tail_bound_refined(M):
    """Refined bound of |w - w_M| in q a^4 / D (8.2): a proposal, informational only.

    Not a basis of acceptance without a decision of the owner (8.2, 8.6).
    """
    _check_odd_index(M)
    M4 = float(M) ** 4
    return 16.0 / math.pi**6 * ((math.log(M) + 1.25) / (4.0 * M4) + 1.0 / (16.0 * M4))


def moment_tail_bound(M):
    """Rough strict bound of |M_x - M_x,M|, |M_y - M_y,M| in q a^2 (8.3, 0 <= nu <= 1).

    1 / (pi^2 M); at M = 1279 it is about 7.9e-5 > DELTA_M, so alone it does
    not guarantee DELTA_M.
    """
    _check_odd_index(M)
    return 1.0 / (math.pi**2 * M)


def moment_tail_bound_refined(M):
    """Refined bound of the moment remainder in q a^2 (8.3, 0 <= nu <= 1): a proposal.

    (16 / pi^4) (ln M + 2) / (2 M^2). Its steps were rechecked
    (NAVIER_REFERENCE_SPEC.recheck.md, 3.2) by the author of the proposal;
    the status in the specification is still a proposal (owner's decision).
    """
    _check_odd_index(M)
    return 16.0 / math.pi**4 * (math.log(M) + 2.0) / (2.0 * M * M)


def select_level(w_tol):
    """Smallest level of NAVIER_LEVELS whose conservative bound 8.2 is <= w_tol.

    w_tol is absolute, in q a^4 / D. For w_tol = DELTA_W the result is 1279.
    """
    _check_positive("w_tol", w_tol)
    for M in NAVIER_LEVELS:
        if deflection_tail_bound(M) <= w_tol:
            return M
    raise ValueError(
        f"w_tol = {w_tol!r} is below the bound {deflection_tail_bound(NAVIER_LEVELS[-1]):.3e} "
        f"of the last level M = {NAVIER_LEVELS[-1]}; give M explicitly"
    )


def deflection_integral_coef(M):
    """J_M = (64 / pi^8) sum_{m,n odd <= M} 1 / (m^2 n^2 (m^2 + n^2)^2) (В-27, item 6).

    Dimensionless integral of the truncated series (131) over the square, in
    q a^6 / D. Every term 1 / den is formed from an exact integer den and
    rounded once; the sum is taken by math.fsum.
    """
    _check_odd_index(M)
    odd = range(1, M + 1, 2)
    terms = [1.0 / (m * m * n * n * (m * m + n * n) ** 2) for m in odd for n in odd]
    return 64.0 / math.pi**8 * math.fsum(terms)


def deflection_integral_tail_bound(M):
    """Conservative bound 0 <= J - J_M <= 1 / (18 pi^4 M^3), in q a^6 / D (section 16, ЧВ-22)."""
    _check_odd_index(M)
    return 1.0 / (18.0 * math.pi**4 * float(M) ** 3)


# Rounding of J_M relative to J_M: one rounding per term (exact integer
# denominator), fsum, the constant 64 / pi^8 (about 4 roundings): 6 u, margin 2.
ROUND_INTEGRAL = 12 * U


def diagnostic_levels(M):
    """Levels of the practical control for the final M: levels of 8.1 below M, then M."""
    _check_odd_index(M)
    return tuple(level for level in NAVIER_LEVELS if level < M) + (M,)


def practical_control(deltas, delta, strict_bound=None):
    """Acceptance of a level by NAVIER_REFERENCE_SPEC.md, 8.4, items 3-4, per point.

    deltas -- array (number of increments, ...) of Delta_j, absolute, in the
    units of delta; delta -- accuracy of the reference delta_Q (8.6);
    strict_bound -- for w, the strict bound 8.2 at the final level (item 4).

    Accepted where the last three increments do not increase,
    Delta_{j-2} >= Delta_{j-1} >= Delta_j (monotone decrease of item 3; equal
    values occur only as exact zeros on the edges), and Delta_j <= delta / 10;
    for w also strict_bound <= delta. With fewer than three increments
    nothing is accepted.
    """
    deltas = np.asarray(deltas, dtype=float)
    if deltas.shape[0] < 3:
        return np.zeros(deltas.shape[1:], dtype=bool)
    d3, d2, d1 = deltas[-3], deltas[-2], deltas[-1]
    accepted = (d3 >= d2) & (d2 >= d1) & (d1 <= delta / 10.0)
    if strict_bound is not None:
        accepted &= strict_bound <= delta
    return accepted


def _odd(M):
    return np.arange(1, M + 1, 2, dtype=np.int64)


def _coefficients(M, nu):
    """Dimensionless coefficient matrices (w, M_x, M_y) for odd m (rows), n (columns)."""
    k = _odd(M).astype(float)
    m = k[:, None]
    n = k[None, :]
    m2 = m * m
    n2 = n * n
    den = m * n * (m2 + n2) ** 2
    cw = (16.0 / math.pi**6) / den
    cx = (16.0 / math.pi**4) * (m2 + nu * n2) / den
    cy = (16.0 / math.pi**4) * (nu * m2 + n2) / den
    return cw, cx, cy


def _node_sines(n, M):
    """S[i, k] = sin(m_k pi i / n), i = 0..n, by exact integer reduction (И-11)."""
    r = np.arange(2 * n, dtype=np.int64)
    sign = np.where(r >= n, -1.0, 1.0)
    r = np.where(r >= n, r - n, r)
    r = np.minimum(r, n - r)
    table = sign * np.sin(math.pi * r / n)
    table[r == 0] = 0.0
    i = np.arange(n + 1, dtype=np.int64)
    return table[np.outer(i, _odd(M)) % (2 * n)]


def _point_sines(theta, M):
    """S[p, k] = sin(m_k pi theta_p) for 0 <= theta <= 1 by reduction to [0, 1/2]."""
    t = np.fmod(np.outer(theta, _odd(M).astype(float)), 2.0)
    upper = t >= 1.0
    sign = np.where(upper, -1.0, 1.0)
    t = np.where(upper, t - 1.0, t)
    t = np.minimum(t, 1.0 - t)
    return sign * np.sin(math.pi * t)


def _sum_separable(Sx, Sy, c, tensor):
    """Sum over m first, then over n."""
    if tensor:
        return (Sx @ c) @ Sy.T
    return np.sum((Sx @ c) * Sy, axis=1)


def _sum_diagonal(Sx, Sy, c):
    """Sum by increasing m + n for paired rows of Sx and Sy."""
    K = c.shape[0]
    out = np.zeros(Sx.shape[0])
    for d in range(2 * K - 1):
        i = np.arange(max(0, d - K + 1), min(d, K - 1) + 1)
        j = d - i
        out += (Sx[:, i] * Sy[:, j]) @ c[i, j]
    return out


def _rounding_bound(c, M, summation, at_nodes):
    """First-order rounding bound of one series, dimensionless, uniform in (x, y)."""
    K = (M + 1) // 2
    summing = (2 if summation == "separable" else 3) * K
    abs_c = np.abs(c)
    bound = (summing + TERM_ROUNDINGS) * U * float(abs_c.sum())
    if not at_nodes:
        k = _odd(M).astype(float)
        bound += 2.0 * math.pi * U * float((abs_c * (k[:, None] + k[None, :])).sum())
    return ROUNDING_MARGIN * bound


@dataclass(frozen=True)
class NavierConvergence:
    """Values on the levels of the practical control (8.4), dimensionless.

    levels -- the levels M_0 < M_1 < ... (diagnostic_levels of the final M);
    w, mx, my -- arrays (len(levels), ...) of w D/(q a^4), M_x/(q a^2), M_y/(q a^2).
    """

    levels: tuple
    w: np.ndarray
    mx: np.ndarray
    my: np.ndarray

    @property
    def delta_w(self):
        """Delta_j = |w(M_j) - w(M_{j-1})|, j = 1.., in q a^4 / D."""
        return np.abs(np.diff(self.w, axis=0))

    @property
    def delta_mx(self):
        """Delta_j for M_x, in q a^2."""
        return np.abs(np.diff(self.mx, axis=0))

    @property
    def delta_my(self):
        """Delta_j for M_y, in q a^2."""
        return np.abs(np.diff(self.my, axis=0))

    @staticmethod
    def ratios(deltas):
        """Delta_{j-1} / Delta_j (8.4, item 5; a guide, not a criterion); nan where 0/0."""
        deltas = np.asarray(deltas, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            return deltas[:-1] / deltas[1:]

    def accepted_w(self):
        """8.4, items 3-4 for w, per point (strict bound 8.2 of the final level)."""
        return practical_control(self.delta_w, DELTA_W, deflection_tail_bound(self.levels[-1]))

    def accepted_mx(self):
        """8.4, item 3 for M_x, per point."""
        return practical_control(self.delta_mx, DELTA_M)

    def accepted_my(self):
        """8.4, item 3 for M_y, per point."""
        return practical_control(self.delta_my, DELTA_M)


@dataclass(frozen=True)
class NavierResult:
    """Navier reference at points or grid nodes.

    x, y -- coordinates (at nodes: the node vectors x_i, y_j; fields are
    indexed [i, j]); M -- largest kept odd index; summation -- order used;
    at_nodes -- grid evaluation with exact integer reduction;
    w_coef, mx_coef, my_coef -- w D/(q a^4), M_x/(q a^2), M_y/(q a^2);
    rounding_w, rounding_mx, rounding_my -- first-order rounding bounds of the
    dimensionless series; convergence -- values on the levels of 8.4 or None.
    """

    plate: "NavierPlate"
    x: np.ndarray
    y: np.ndarray
    M: int
    summation: str
    at_nodes: bool
    w_coef: np.ndarray
    mx_coef: np.ndarray
    my_coef: np.ndarray
    rounding_w: float
    rounding_mx: float
    rounding_my: float
    convergence: NavierConvergence | None

    @property
    def w_scale(self):
        """q a^4 / D."""
        return self.plate.q * self.plate.a**4 / self.plate.D

    @property
    def moment_scale(self):
        """q a^2."""
        return self.plate.q * self.plate.a**2

    @property
    def w(self):
        """Deflection w (units of the input, D-009)."""
        return self.w_scale * self.w_coef

    @property
    def Mx(self):
        """Bending moment M_x."""
        return self.moment_scale * self.mx_coef

    @property
    def My(self):
        """Bending moment M_y."""
        return self.moment_scale * self.my_coef

    @property
    def w_tail_bound_coef(self):
        """Conservative strict truncation bound of w (8.2), in q a^4 / D."""
        return deflection_tail_bound(self.M)

    @property
    def w_tail_bound(self):
        """Conservative strict truncation bound of w (8.2), physical: |q| a^4 / D times 8.2."""
        return abs(self.w_scale) * self.w_tail_bound_coef

    @property
    def w_error_bound_coef(self):
        """Truncation bound 8.2 plus the rounding bound of w, in q a^4 / D (8.6: delta_w)."""
        return self.w_tail_bound_coef + self.rounding_w

    @property
    def w_tail_bound_refined_coef(self):
        """Refined bound 8.2 (proposal, informational only), in q a^4 / D."""
        return deflection_tail_bound_refined(self.M)

    @property
    def moment_tail_bound_coef(self):
        """Rough strict bound 8.3 in q a^2, or None outside 0 <= nu <= 1 (scope of 8.3)."""
        return moment_tail_bound(self.M) if 0.0 <= self.plate.nu <= 1.0 else None

    @property
    def moment_tail_bound_refined_coef(self):
        """Refined bound 8.3 (proposal) in q a^2, or None outside 0 <= nu <= 1."""
        return moment_tail_bound_refined(self.M) if 0.0 <= self.plate.nu <= 1.0 else None


def _resolve_level(M, w_tol):
    if M is not None and w_tol is not None:
        raise ValueError("give either M or w_tol, not both")
    if w_tol is not None:
        return select_level(w_tol)
    if M is None:
        return M_R2
    _check_odd_index(M)
    return M


def _check_summation(summation):
    if summation not in SUMMATIONS:
        raise ValueError(f"summation must be one of {SUMMATIONS}, got {summation!r}")


def _check_count(name, value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")


@dataclass(frozen=True)
class NavierIntegral:
    """Integral of the truncated deflection series and the energy at level M.

    J_M -- dimensionless integral (q a^6 / D); I_M = (q a^6 / D) J_M;
    Pi_M = -q I_M / 2. All three are values of the series truncated at M,
    approximations of the limits J, I, Pi(omega_0); their remainder bounds are
    the separate fields J_tail_bound (section 16), I_tail_bound =
    |q| a^6 / D J_tail_bound, Pi_tail_bound = q^2 a^6 / (2 D) J_tail_bound.
    rounding_J -- first-order rounding bound of J_M (absolute, dimensionless).
    """

    M: int
    J_M: float
    I_M: float
    Pi_M: float
    J_tail_bound: float
    I_tail_bound: float
    Pi_tail_bound: float
    rounding_J: float


@dataclass(frozen=True)
class NavierPlate:
    """Square SSSS plate under uniform load: a = b > 0, q, D > 0, nu in (-1, 1).

    b may be given for the contract of NAVIER_REFERENCE_SPEC.md, section 9,
    but must equal a: the rectangle (R3) is not implemented. nu enters only
    the moments. Units must be consistent (D-009).
    """

    a: float
    q: float
    D: float
    nu: float
    b: float | None = None

    def __post_init__(self):
        _check_positive("a", self.a)
        _check_finite("q", self.q)
        _check_positive("D", self.D)
        _check_nu(self.nu)
        if self.b is not None:
            _check_positive("b", self.b)
            if self.b != self.a:
                raise ValueError(f"only the square b = a is implemented (R2), got a={self.a!r}, b={self.b!r}")

    @classmethod
    def from_material(cls, *, a, q, E, thickness, nu, b=None):
        """Plate with D = E t^3 / (12 (1 - nu^2)) [L-107, eq. (3), p. 16]."""
        return cls(a=a, q=q, D=flexural_rigidity(E, thickness, nu), nu=nu, b=b)

    def deflection_integral(self, *, M=None):
        """J_M, I_M, Pi_M and their remainder bounds at level M (default M_R2 = 1279)."""
        M = M_R2 if M is None else M
        J = deflection_integral_coef(M)
        bound = deflection_integral_tail_bound(M)
        scale = self.q * self.a**6 / self.D
        I = scale * J
        return NavierIntegral(
            M=M,
            J_M=J,
            I_M=I,
            Pi_M=-0.5 * self.q * I,
            J_tail_bound=bound,
            I_tail_bound=abs(scale) * bound,
            Pi_tail_bound=0.5 * abs(self.q * scale) * bound,
            rounding_J=ROUND_INTEGRAL * J,
        )

    def at_points(self, x, y, *, M=None, w_tol=None, summation="separable", diagnostics=True):
        """w, M_x, M_y at the points (x, y) (arrays broadcast together).

        M -- largest kept odd index; or w_tol -- absolute tolerance in q a^4 / D,
        M by select_level; neither - M = M_R2 = 1279. Points must lie in the
        closed square [0, a]^2.
        """
        M = _resolve_level(M, w_tol)
        _check_summation(summation)
        x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        for name, values in (("x", x), ("y", y)):
            if not np.all(np.isfinite(values)) or not np.all((values >= 0.0) & (values <= self.a)):
                raise ValueError(f"{name} must lie in the closed interval [0, a] = [0, {self.a!r}]")
        shape = x.shape
        xf = x.ravel()
        yf = y.ravel()
        edge = (xf == 0.0) | (xf == self.a) | (yf == 0.0) | (yf == self.a)
        levels = diagnostic_levels(M) if diagnostics else (M,)
        coeffs = _coefficients(M, self.nu)
        values = np.zeros((3, len(levels), xf.size))
        for start in range(0, xf.size, _POINT_CHUNK):
            part = slice(start, start + _POINT_CHUNK)
            Sx = _point_sines(xf[part] / self.a, M)
            Sy = _point_sines(yf[part] / self.a, M)
            for li, level in enumerate(levels):
                K = (level + 1) // 2
                for qi, c in enumerate(coeffs):
                    if summation == "separable":
                        v = _sum_separable(Sx[:, :K], Sy[:, :K], c[:K, :K], tensor=False)
                    else:
                        v = _sum_diagonal(Sx[:, :K], Sy[:, :K], c[:K, :K])
                    values[qi, li, part] = v
        values[:, :, edge] = 0.0
        values = values.reshape((3, len(levels)) + shape)
        return self._result(x, y, M, summation, False, coeffs, levels, values, diagnostics)

    def at_nodes(self, nx, ny, *, M=None, w_tol=None, summation="separable", diagnostics=True):
        """w, M_x, M_y at all nodes x_i = i a / nx, y_j = j a / ny of the tensor grid.

        Fields have shape (nx + 1, ny + 1) and index [i, j]. Sines by exact
        integer reduction; separable summation S_x C S_y^T (NUMERICAL_SPEC.md,
        section 16). The diagonal order is meant for small grids.
        """
        _check_count("nx", nx)
        _check_count("ny", ny)
        M = _resolve_level(M, w_tol)
        _check_summation(summation)
        levels = diagnostic_levels(M) if diagnostics else (M,)
        coeffs = _coefficients(M, self.nu)
        Sx = _node_sines(nx, M)
        Sy = _node_sines(ny, M)
        if summation == "diagonal":
            ii, jj = np.meshgrid(np.arange(nx + 1), np.arange(ny + 1), indexing="ij")
            Sx_pairs = Sx[ii.ravel()]
            Sy_pairs = Sy[jj.ravel()]
        values = np.zeros((3, len(levels), nx + 1, ny + 1))
        for li, level in enumerate(levels):
            K = (level + 1) // 2
            for qi, c in enumerate(coeffs):
                if summation == "separable":
                    v = _sum_separable(Sx[:, :K], Sy[:, :K], c[:K, :K], tensor=True)
                else:
                    v = _sum_diagonal(Sx_pairs[:, :K], Sy_pairs[:, :K], c[:K, :K]).reshape(nx + 1, ny + 1)
                values[qi, li] = v
        values[:, :, [0, nx], :] = 0.0
        values[:, :, :, [0, ny]] = 0.0
        x = self.a * np.arange(nx + 1) / nx
        y = self.a * np.arange(ny + 1) / ny
        return self._result(x, y, M, summation, True, coeffs, levels, values, diagnostics)

    def _result(self, x, y, M, summation, at_nodes, coeffs, levels, values, diagnostics):
        rounding = [_rounding_bound(c, M, summation, at_nodes) for c in coeffs]
        convergence = (
            NavierConvergence(levels=levels, w=values[0], mx=values[1], my=values[2]) if diagnostics else None
        )
        return NavierResult(
            plate=self,
            x=x,
            y=y,
            M=M,
            summation=summation,
            at_nodes=at_nodes,
            w_coef=values[0, -1],
            mx_coef=values[1, -1],
            my_coef=values[2, -1],
            rounding_w=rounding[0],
            rounding_mx=rounding[1],
            rounding_my=rounding[2],
            convergence=convergence,
        )
