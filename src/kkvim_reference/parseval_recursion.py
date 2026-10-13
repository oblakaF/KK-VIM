"""Semi-analytic KK-VIM sequence s = 1, ..., s_max by sine series and Parseval (R2).

Independent of the finite-difference solver: no grid operators, no grid
quadratures, no matrices, no imports of ``kkvim`` (D-028, D-033). Square SSSS
plate of side a, uniform load q != 0, N = 1, counters by D-030.

Half-steps (MATH_SPEC.md, sections 4.3-4.5; NUMERICAL_SPEC.md, 15.3):

    G1 = (a/2, pi^2/(2a), pi^4/(2a^3), 2 q a/pi)       from X0 = sin(pi x/a),
    s = 1:  G1 -> Y1 (sine series on (0, a) in y),      w1 = X0 * Y1,
    s = 2:  Y1 -> G2 (Parseval) -> X2 (series in x),    w2 = X2 * Y1,
    s = 3:  X2 -> G3 (Parseval) -> Y3 (series in y),    w3 = X2 * Y3,
    ...     odd s: Y-half-step, w_s = X^{s-1} Y^s; even s: X-half-step, w_s = X^s Y^{s-1}.

Each half-step is solved by ``SineHalfStep`` (ЧВ-17). The coefficients of the
next half-step follow from the kept coefficients c_m, odd m, by Parseval
(NUMERICAL_SPEC.md, 15.2, ЧВ-18), lambda_m = m pi / ell:

    A = (ell/2) sum c_m^2,  B = (ell/2) sum lambda_m^2 c_m^2,
    C = (ell/2) sum lambda_m^4 c_m^2,  F = 2 q sum c_m / lambda_m,

all sums by math.fsum. No normalisation is applied: the multipliers are
stored unnormalised, so the field is the plain product of the two multipliers
(MATH_SPEC.md, В-13). An optional scale of X0 (``x0_scale``) multiplies G1 by
(c^2, c^2, c^2, c); the fields do not depend on it. The number of half-steps
s_max is any positive integer (default 3); there is no upper limit.

Choice of M (NUMERICAL_SPEC.md, 15.4, И-12). M is the largest kept odd index,
chosen for every half-step from the analytic tail bounds alone, before any
comparison: the smallest odd M with

    K / (8 M^4) <= eps * L_f,   tail(A) <= eps A_M,  tail(B) <= eps B_M,
    tail(C) <= eps C_M,  tail(F) <= eps |F_M|,      eps = 1e-14,

with K = 4 |F| ell^4 / (D A pi^5) and the tails of section 15.4, item 3.
L_f = |c_1| - |c_3| is a lower bound of max |f| and of max |f_M| (the
centre value is an alternating sum with decreasing |c_m|); it replaces
|f_M|_max, which makes the rule slightly stricter and the criterion monotone
in M. The relative and the absolute budget (1e-14 * max |f_M|) coincide.
Diagnostic mode (practical control, 15.5): ``m_scale`` = k > 1 keeps
M = k (M_rule + 1) - 1 >= k M_rule terms on every half-step, where M_rule is
the M of the rule above for the G of that run; the rule itself is unchanged.

Error accumulation (NUMERICAL_SPEC.md, 15.5, ЧВ-21), kept separately for every
half-step: (a) series tails at exact input G; (b)+(c) the error rho_s of the
input G_s and its transfer; (d) rounding. The transfer is used in closed form,
not only to first order: if G_s has relative error <= rho, every c_m has
relative error <= r_c = (1 + rho)/(1 - rho) - 1, the next A, B, C have
<= (1 + r_c)^2 - 1, the next F <= r_c, so

    rho_{s+1} = (1 + r_c)^2 (1 + tau_tail) (1 + ROUND_PARSEVAL) - 1,

which is not smaller than the first-order rule rho_{s+1} <= 4 rho_s + tau.
Rounding uses a first-order model (constants below, [СТАНД.]); the accuracy of
the library sin is taken as about u (NUMERICAL_SPEC.md, 15.5, [ИСТОЧНИК НЕ
ПОДТВЕРЖДЁН]). The bound grows like 4^s and is pessimistic for large s
(15.5). Where it stops being a bound (rho >= 1 or a non-positive
denominator), the budget is marked invalid and its errors are inf; the
values are still computed. Non-finite or degenerate coefficients raise
FloatingPointError. Doubling M is a practical control, not a proof of
accuracy.

Continuous energy (MATH_SPEC.md, В-23, В-27; NUMERICAL_SPEC.md, 17): for
w_s = X(x) Y(y), with A, B, C, I = integral of the multiplier, all from its
sine coefficients (no quadrature; I = 2 sum c_m / lambda_m, X0: 2a/pi),

    I_s = I_X I_Y,   Pi_s = -q I_s / 2   (energy of the projection),
    Pi_full = D (C_X A_Y + 2 B_X B_Y + A_X C_Y) / 2 - q I_X I_Y.

The computed multiplier is the exact Galerkin solution on the kept sines for
the computed G, so the two forms differ only by rounding (truncation does
not enter their difference). Pi_s is the energy of the truncated sequence;
its distance to the exact sequence is bounded by the transferred errors.

Rectangle a x b (R3; D-036, D-030; R3_NUMERICAL_SPEC_DRAFT.md, 2, 3.1, 6.4).
With b given (0.2 <= r = b/a <= 5), X0 = sin(pi x/a) on (0, a) as before and the
half-steps alternate in length:

    ell_s = b for odd s (Y-half-step, multiplier Y^s on (0, b)),
    ell_s = a for even s (X-half-step, multiplier X^s on (0, a)).

Half-step s is solved on ell_s with G_s; G_{s+1} are the Parseval sums of the
multiplier of half-step s on its own physical length ell_s (G1 from X0 on a,
G2 from Y1 on b, G3 from X2 on a, ...; MATH_SPEC.md, 4.2-4.4). The tails of
15.4 and the rule И-12 are applied with ell_s; M is chosen automatically and
is unrelated to the Navier level 2559. Fields at x_i = i a/n_x, y_j = j b/n_y;
the energy uses A, B, C, I of X on (0, a) and of Y on (0, b). b = None (the
default) or b = a is the square of R2 with bitwise the same operations.
The ratio is checked exactly on the given floats, without a tolerance; the
representation tolerance 4u of D-037 is decided for the Navier reference only
and is not applied here. The second start (by y) is the plate b x a started
by x (ЧВ-R1, swapped()).

Area of validity: square SSSS (R2) and rectangle SSSS with 0.2 <= b/a <= 5
(R3, not yet independently qualified), q = const, N = 1. The accuracy target
1e-12 (15.7) is guaranteed by the budget for s <= 3 of the square only; for
other s and r the budget is reported as computed. Not a solution of the 2D
plate problem, and a stabilisation of w_s in s is not a proof of a limit.
"""

import math
from dataclasses import dataclass

import numpy as np

from kkvim_reference._r3_range import check_r3_ratio
from kkvim_reference.sine_halfstep import SineHalfStep, _sin_pi_ratio

__all__ = [
    "EPS_TAIL",
    "R3_RATIO_MAX",
    "R3_RATIO_MIN",
    "S_DEFAULT",
    "Coefficients",
    "HalfStepBudget",
    "HalfStepEnergy",
    "HalfStepRecord",
    "ParsevalRecursion",
    "first_coefficients",
    "parseval_coefficients",
    "parseval_tails",
    "select_odd_index",
]

U = 2.0**-53
# Relative budget of every analytic tail (И-12).
EPS_TAIL = 1e-14
# First-order rounding models, counted for the formulas of this module and of
# SineHalfStep.coefficient (operations on positive numbers, no subtraction):
# G1 from pi, a, q (about 8 roundings, margin 2);
ROUND_G1 = 16 * U
# c_m from given G costs about 24 U (lambda_m, its powers, P_m, the
# denominator, the quotient). A, B, C, F from the c_m: c_m^2 doubles 24 U,
# lambda_m^4 adds 15 U, products, fsum and the factor ell/2: about 68 U for C;
# margin 1.4;
ROUND_PARSEVAL = 96 * U
# value of a sum sum c_m sin(pi r/n), relative to sum |c_m|: c_m 24 U, sin of
# a reduced argument about 5 U, product and fsum 2 U; margin 1.5.
ROUND_VALUES = 48 * U
# sin(pi r/n) of X0 = x0_scale * sin(pi x/a), relative to |x0_scale|.
ROUND_X0 = 8 * U
# Pi_s against Pi_full, relative to |Pi_s|: D(...) by Parseval sums of the
# computed multiplier (<= ROUND_PARSEVAL, the rounding of c_m included) and
# products, I_X I_Y (<= ROUND_PARSEVAL each), then D(...)/2 - q I: about
# 3.1 ROUND_PARSEVAL for Pi_full and 1.1 for Pi_s; margin 1.2.
ROUND_ENERGY = 5 * ROUND_PARSEVAL

# Default number of half-steps (the earlier contract of the module).
S_DEFAULT = 3


def _check_finite(name, value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {value!r}")


def _check_positive(name, value):
    _check_finite(name, value)
    if not value > 0:
        raise ValueError(f"{name} must be positive, got {value!r}")


def _check_count(name, value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")


# Approved range of r = b / a for R3 (D-036, Р-1), compared exactly.
R3_RATIO_MIN = 0.2
R3_RATIO_MAX = 5.0


def _check_r3_ratio(a, b):
    """Shared D-038 aspect-ratio check on the original float lengths."""
    check_r3_ratio(a, b)


@dataclass(frozen=True)
class Coefficients:
    """G_s = (A, B, C, F): the coefficients by which half-step s is solved."""

    A: float
    B: float
    C: float
    F: float

    def invariants(self):
        """(B/A, C/A, F^2/A): independent of the scale of the multiplier (15.3)."""
        return (self.B / self.A, self.C / self.A, self.F * self.F / self.A)

    def is_finite(self):
        return all(math.isfinite(v) for v in (self.A, self.B, self.C, self.F))


def first_coefficients(*, a, q, x0_scale=1.0):
    """G1 of X0 = x0_scale * sin(pi x/a) (MATH_SPEC.md, 5.1, scaled by В-13)."""
    _check_positive("a", a)
    _check_finite("q", q)
    _check_finite("x0_scale", x0_scale)
    c2 = x0_scale * x0_scale
    return Coefficients(
        A=c2 * (a / 2.0),
        B=c2 * (math.pi**2 / (2.0 * a)),
        C=c2 * (math.pi**4 / (2.0 * a**3)),
        F=x0_scale * (2.0 * q * a / math.pi),
    )


def _odd_lambdas(n_terms, ell):
    return [(2 * j + 1) * math.pi / ell for j in range(n_terms)]


def parseval_coefficients(coefficients, *, ell, q):
    """A, B, C, F of f = sum c_m sin(lambda_m xi) on (0, ell) by Parseval (ЧВ-18).

    coefficients -- c_1, c_3, c_5, ... (odd indices m = 2j + 1 only);
    no spatial quadrature is used.
    """
    _check_positive("ell", ell)
    _check_finite("q", q)
    coefficients = [float(c) for c in coefficients]
    if not coefficients:
        raise ValueError("at least one coefficient is required")
    lambdas = _odd_lambdas(len(coefficients), ell)
    squares = [c * c for c in coefficients]
    lam2 = [lam * lam for lam in lambdas]
    half = ell / 2.0
    return Coefficients(
        A=half * math.fsum(squares),
        B=half * math.fsum(l2 * c2 for l2, c2 in zip(lam2, squares)),
        C=half * math.fsum(l2 * l2 * c2 for l2, c2 in zip(lam2, squares)),
        F=2.0 * q * math.fsum(c / lam for c, lam in zip(coefficients, lambdas)),
    )


def _tail_constant(G, D, ell):
    return 4.0 * abs(G.F) * ell**4 / (D * G.A * math.pi**5)


def parseval_tails(G, *, D, ell, q, M):
    """Analytic tails (15.4, item 3) of a half-step solved with G, kept up to odd M.

    Returns (function, A, B, C, F): bounds of max|f - f_M| and of the
    remainders of the Parseval sums, all for the exact series with input G.
    """
    k = _tail_constant(G, D, ell)
    r = math.pi / ell
    return (
        k / (8.0 * M**4),
        (ell / 2.0) * k * k / (18.0 * M**9),
        (ell / 2.0) * k * k * r**2 / (14.0 * M**7),
        (ell / 2.0) * k * k * r**4 / (10.0 * M**5),
        abs(q) * (2.0 * ell / math.pi) * k / (10.0 * M**5),
    )


def _max_lower_bound(c1, c3):
    """|c_1| - |c_3| <= max|f|: centre value of a same-sign series with decreasing |c_m|."""
    return abs(c1) - abs(c3)


def _relative_tails(G, terms, *, D, ell, q, M, f_lower):
    tails = parseval_tails(G, D=D, ell=ell, q=q, M=M)
    kept = parseval_coefficients(terms[: (M + 1) // 2], ell=ell, q=q)
    return (
        tails[0] / f_lower,
        tails[1] / kept.A,
        tails[2] / kept.B,
        tails[3] / kept.C,
        tails[4] / abs(kept.F),
    )


def select_odd_index(G, *, D, ell, q, eps=EPS_TAIL):
    """Smallest odd M whose analytic relative tails are all <= eps (И-12).

    Depends only on G, D, ell, q and eps, never on a comparison with another
    solution. Every relative tail decreases in M (its denominator, a partial
    sum of positive terms or the fixed L_f, does not), so the smallest M is
    found by doubling and bisection.
    """
    _check_positive("eps", eps)
    if q == 0:
        raise ValueError("q must be nonzero (MATH_SPEC.md, 4.6, condition 3)")

    def ok(M, terms):
        return max(_relative_tails(G, terms, D=D, ell=ell, q=q, M=M, f_lower=f_lower)) <= eps

    probe = SineHalfStep(A=G.A, B=G.B, C=G.C, F=G.F, D=D, ell=ell, M=3)
    f_lower = _max_lower_bound(probe.coefficient(1), probe.coefficient(3))
    hi = 63
    while True:
        terms = SineHalfStep(A=G.A, B=G.B, C=G.C, F=G.F, D=D, ell=ell, M=hi).coefficients()
        if ok(hi, terms):
            break
        hi = 2 * hi + 1
    if ok(1, terms):
        return 1
    lo = 1
    # invariant: lo, hi odd, ok(hi) and not ok(lo)
    while hi - lo > 2:
        mid = lo + 2 * ((hi - lo) // 4)
        if ok(mid, terms):
            hi = mid
        else:
            lo = mid
    return hi


@dataclass(frozen=True)
class HalfStepBudget:
    """Separate error accounting of one half-step (NUMERICAL_SPEC.md, 15.4-15.5).

    Relative quantities refer to max|f| of the multiplier computed on this
    half-step (bounded below by L_f), to the kept Parseval sums A_M, B_M, C_M,
    |F_M|, or to max|w_s|.

    M, n_terms -- largest kept odd index and the number of kept terms;
    tail_constant -- K, |c_m| <= K / m^5;
    tail_function -- K / (8 M^4), absolute tail of f at exact input G_s;
    tail_function_rel, tail_A_rel, tail_B_rel, tail_C_rel, tail_F_rel --
        relative tails at the kept M (all <= EPS_TAIL);
    input_error -- rho_s, relative error of every component of G_s;
    coefficient_error -- r_c, relative error of every c_m caused by rho_s;
    rounding_coefficients -- rounding of the Parseval sums (relative);
    rounding_values -- rounding of one series value (absolute);
    multiplier_error -- total relative error of the multiplier values;
    field_error -- total relative error of w_s values (to max|w_s|);
    invariant_error -- relative error of B/A, C/A, F^2/A of G_s;
    first_order_input_error -- sum_j 4^(s-j) tau_j of the first-order rule;
    M_rule -- M of the rule И-12 for this G_s (M = M_rule unless m_scale > 1);
    integral_error -- relative error of the integral of the computed multiplier;
    valid -- False where the transfer no longer bounds the error (rho_s >= 1
        or a non-positive denominator); the error fields are then inf.
    """

    M: int
    n_terms: int
    tail_constant: float
    tail_function: float
    tail_function_rel: float
    tail_A_rel: float
    tail_B_rel: float
    tail_C_rel: float
    tail_F_rel: float
    input_error: float
    coefficient_error: float
    rounding_coefficients: float
    rounding_values: float
    multiplier_error: float
    field_error: float
    invariant_error: float
    first_order_input_error: float
    M_rule: int
    integral_error: float
    valid: bool


@dataclass(frozen=True)
class HalfStepRecord:
    """Half-step s: counters, input G_s, its sine series and error budget.

    variable -- "Y" for odd s (series in y on (0, a)), "X" for even s;
    k -- k = s // 2 (D-030); step -- SineHalfStep of the computed multiplier.
    """

    s: int
    k: int
    variable: str
    coefficients: Coefficients
    step: SineHalfStep
    budget: HalfStepBudget


@dataclass(frozen=True)
class HalfStepEnergy:
    """Continuous energy of w_s = X Y from the sine coefficients (no grid, no Pi_h).

    integral -- I_s = I_X I_Y, integral of w_s over the plate;
    energy -- Pi_s = -q I_s / 2 (В-23, energy of the projection);
    energy_full -- D (C_X A_Y + 2 B_X B_Y + A_X C_Y) / 2 - q I_s by Parseval;
    energy_norm2 -- a(w_s, w_s) = D (C_X A_Y + 2 B_X B_Y + A_X C_Y);
    error -- relative bound of |Pi_s - Pi(w_s exact sequence)| / |Pi_s| from
        the transferred errors, tails and rounding (inf if the budget is invalid);
    rounding -- relative bound of |Pi_s - energy_full| / |Pi_s| (ROUND_ENERGY);
    delta -- Pi_s - Pi_{s-1} by subtraction (None for s = 1);
    change_norm2 -- ||w_s - w_{s-1}||^2_{H_T} by Parseval of the difference of
        the changed multiplier, without subtraction of energies (None for s = 1);
        equal to 2 (Pi_{s-1} - Pi_s) for exact projections (В-24);
    change_rel -- ||w_s - w_{s-1}||_{H_T} / ||w_s||_{H_T} (None for s = 1);
    change_rel_error -- bound of the error of change_rel against the exact
        sequence (transferred errors of both versions of the changed multiplier).
    """

    s: int
    k: int
    integral: float
    energy: float
    energy_full: float
    energy_norm2: float
    error: float
    rounding: float
    delta: float | None
    change_norm2: float | None
    change_rel: float | None
    change_rel_error: float | None


class ParsevalRecursion:
    """KK-VIM sequence s = 1..s_max for the square or rectangular SSSS plate.

    a -- side along x (X0 = sin(pi x/a), D-030), b -- side along y (default
    None: b = a, the square of R2; otherwise 0.2 <= b/a <= 5, R3), D --
    flexural rigidity, q -- uniform load (q != 0), s_max -- number of
    half-steps, any positive integer (default 3), x0_scale -- scale of
    X0 = x0_scale * sin(pi x/a); the fields do not depend on it (MATH_SPEC.md,
    В-13); m_scale -- diagnostic multiplier of the truncation level (practical
    control 15.5), 1 for the rule И-12.
    """

    def __init__(self, *, a, q, D, s_max=S_DEFAULT, x0_scale=1.0, eps=EPS_TAIL, m_scale=1, b=None):
        _check_positive("a", a)
        if b is not None:
            _check_positive("b", b)
            _check_r3_ratio(a, b)
        _check_positive("D", D)
        _check_finite("q", q)
        _check_finite("x0_scale", x0_scale)
        if q == 0 or x0_scale == 0:
            raise ValueError("q and x0_scale must be nonzero (MATH_SPEC.md, 4.6)")
        _check_count("s_max", s_max)
        _check_count("m_scale", m_scale)
        _check_positive("eps", eps)
        self.a = float(a)
        self.b = self.a if b is None else float(b)
        self.q = float(q)
        self.D = float(D)
        self.x0_scale = float(x0_scale)
        self.s_max = s_max
        self.eps = eps
        self.m_scale = m_scale

        records = []
        G = first_coefficients(a=self.a, q=self.q, x0_scale=self.x0_scale)
        # parseval[j] -- A, B, C, F of the multiplier of half-step j (0 is X0),
        # i.e. G_{j+1}; integrals[j] and their relative errors likewise.
        parseval = [G]
        integrals = [self.x0_scale * 2.0 * self.a / math.pi]
        integral_errors = [ROUND_X0]
        rho = ROUND_G1
        rho_first_order = ROUND_G1
        multiplier_errors = {0: ROUND_X0}
        for s in range(1, s_max + 1):
            record = self._halfstep(s, G, rho, rho_first_order, multiplier_errors)
            records.append(record)
            b = record.budget
            # the multiplier of half-step s lives on its own length ell_s
            G = parseval_coefficients(record.step.coefficients(), ell=self.length(s), q=self.q)
            if not (G.is_finite() and G.A > 0):
                raise FloatingPointError(f"half-step {s}: non-finite or degenerate next coefficients {G!r}")
            parseval.append(G)
            integrals.append(_integral(record.step))
            integral_errors.append(b.integral_error)
            tail = max(b.tail_A_rel, b.tail_B_rel, b.tail_C_rel, b.tail_F_rel)
            # The tails were bounded with the computed G_s; for the exact G_s
            # the constant K and the partial sums move by at most r_c.
            r_c = b.coefficient_error
            if b.valid:
                tail_exact = tail * (1.0 + r_c) ** 2 / (1.0 - r_c) ** 2
                rho = _compose(r_c, r_c, tail_exact, ROUND_PARSEVAL)
            else:
                rho = math.inf
            rho_first_order = 4.0 * rho_first_order + tail + ROUND_PARSEVAL
        self._records = tuple(records)
        self._parseval = tuple(parseval)
        self._integrals = tuple(integrals)
        self._integral_errors = tuple(integral_errors)

    def length(self, s):
        """ell_s: b for odd s (Y-half-step), a for even s (X-half-step); s >= 1."""
        return self.b if s % 2 else self.a

    @property
    def r(self):
        """r = b / a."""
        return self.b / self.a

    def swapped(self):
        """The plate b x a started by x with the same data (ЧВ-R1).

        Equivalent to the start by y on a x b: w_{a x b, start y}^(s)(x, y) =
        w_{b x a, start x}^(s)(y, x) for every s; its fields at nodes are the
        transposed fields of the grid (n_y, n_x).
        """
        return ParsevalRecursion(
            a=self.b, b=self.a, q=self.q, D=self.D, s_max=self.s_max, x0_scale=self.x0_scale,
            eps=self.eps, m_scale=self.m_scale,
        )

    def _halfstep(self, s, G, rho, rho_first_order, multiplier_errors):
        ell = self.length(s)
        if not (G.is_finite() and G.A > 0 and G.B >= 0 and G.C >= 0 and G.F != 0):
            raise FloatingPointError(f"half-step {s}: non-finite or degenerate coefficients {G!r}")
        try:
            M_rule = select_odd_index(G, D=self.D, ell=ell, q=self.q, eps=self.eps)
            M = self.m_scale * (M_rule + 1) - 1
            step = SineHalfStep(A=G.A, B=G.B, C=G.C, F=G.F, D=self.D, ell=ell, M=M)
            terms = step.coefficients()
            f_lower = _max_lower_bound(step.coefficient(1), step.coefficient(3))
            rel = _relative_tails(G, terms, D=self.D, ell=ell, q=self.q, M=M, f_lower=f_lower)
        except (ZeroDivisionError, OverflowError, ValueError) as error:
            # underflow or overflow of the series (e.g. extreme q, a, D): no reliable result
            raise FloatingPointError(f"half-step {s}: the sine series cannot be evaluated reliably ({error})") from error

        abs_sum = step.abs_coefficient_sum()
        rounding_values = ROUND_VALUES * abs_sum
        valid = rho < 1.0
        if valid:
            r_c = 2.0 * rho / (1.0 - rho)  # (1 + rho)/(1 - rho) - 1
            error_abs = r_c * abs_sum + step.tail_bound * (1.0 + r_c) + rounding_values
            denominator = f_lower * (1.0 - r_c) - error_abs
            valid = r_c < 1.0 and denominator > 0
        if valid:
            multiplier_error = error_abs / denominator
        else:
            r_c = multiplier_error = math.inf
        multiplier_errors[s] = multiplier_error
        e_x = multiplier_errors[_x_index(s)]
        e_y = multiplier_errors[_y_index(s)]
        if valid and math.isfinite(e_x) and math.isfinite(e_y):
            field_error = _compose(e_x, e_y, U)
            # (1 + rho)^2 / (1 - rho) (1 + 2u) - 1: F^2/A is the worst of the three
            invariant_error = _compose(rho, rho, rho / (1.0 - rho), 2 * U)
            integral_error = _compose(r_c, rel[4], ROUND_PARSEVAL)
        else:
            valid = False
            field_error = invariant_error = integral_error = math.inf

        budget = HalfStepBudget(
            M=M,
            n_terms=step.n_terms,
            tail_constant=step.tail_constant,
            tail_function=step.tail_bound,
            tail_function_rel=rel[0],
            tail_A_rel=rel[1],
            tail_B_rel=rel[2],
            tail_C_rel=rel[3],
            tail_F_rel=rel[4],
            input_error=rho,
            coefficient_error=r_c,
            rounding_coefficients=ROUND_PARSEVAL,
            rounding_values=rounding_values,
            multiplier_error=multiplier_error,
            field_error=field_error,
            invariant_error=invariant_error,
            first_order_input_error=rho_first_order,
            M_rule=M_rule,
            integral_error=integral_error,
            valid=valid,
        )
        return HalfStepRecord(
            s=s,
            k=s // 2,
            variable="Y" if s % 2 else "X",
            coefficients=G,
            step=step,
            budget=budget,
        )

    def _check_s(self, s):
        if isinstance(s, bool) or not isinstance(s, int) or not 1 <= s <= self.s_max:
            raise ValueError(f"s must be an integer in 1..{self.s_max}, got {s!r}")

    def halfstep(self, s):
        """Record of half-step s: counters, G_s, sine series, budget."""
        self._check_s(s)
        return self._records[s - 1]

    def coefficients(self, s):
        """G_s, the coefficients by which half-step s is solved."""
        return self.halfstep(s).coefficients

    def _multiplier_nodal(self, index, n):
        if index == 0:
            return [self.x0_scale * _sin_pi_ratio(i, n) for i in range(n + 1)]
        return self._records[index - 1].step.nodal_values(n)

    def field_on_grid(self, s, nx, ny, *, i=None, j=None):
        """w_s at the nodes x_i = i a/nx, y_j = j b/ny of a tensor grid (b = a for the square).

        Returns an array W with W[p, r] = w_s(x_{i[p]}, y_{j[r]}); by default
        all nodes i = 0..nx, j = 0..ny. Arguments of sin are reduced in
        integers (И-11); boundary nodes are exact zeros.
        """
        self._check_s(s)
        for name, n in (("nx", nx), ("ny", ny)):
            if isinstance(n, bool) or not isinstance(n, int) or n < 1:
                raise ValueError(f"{name} must be a positive integer, got {n!r}")
        i = range(nx + 1) if i is None else _check_indices("i", i, nx)
        j = range(ny + 1) if j is None else _check_indices("j", j, ny)
        x_values = self._multiplier_nodal(_x_index(s), nx)
        y_values = self._multiplier_nodal(_y_index(s), ny)
        return np.outer([x_values[p] for p in i], [y_values[r] for r in j])

    def field_at_points(self, s, x, y):
        """w_s at arbitrary points 0 <= x <= a, 0 <= y <= b (arrays broadcast together).

        Pointwise sums f(xi) = sum c_m sin(m pi xi / ell) of the multipliers
        (SineHalfStep.value) and X0 = x0_scale sin(pi x/a); every term vanishes
        on the edges x = 0, a and y = 0, b, so the values there are exact
        zeros. Away from the nodes the argument m (xi / ell) is rounded: the
        error of a multiplier value is about pi u sum m |c_m| on top of the
        nodal rounding (sine_halfstep, value()); exact integer reduction exists
        only at grid nodes (field_on_grid).
        """
        self._check_s(s)
        x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        for name, values, length, label in (("x", x, self.a, "a"), ("y", y, self.b, "b")):
            if not np.all(np.isfinite(values)) or not np.all((values >= 0.0) & (values <= length)):
                raise ValueError(f"{name} must lie in the closed interval [0, {label}] = [0, {length!r}]")
        xs, ys = x.ravel(), y.ravel()
        fx = self._multiplier_at(_x_index(s), xs, self.a)
        fy = self._multiplier_at(_y_index(s), ys, self.b)
        w = fx * fy
        w[(xs == 0.0) | (xs == self.a) | (ys == 0.0) | (ys == self.b)] = 0.0
        return w.reshape(x.shape)

    def _multiplier_at(self, index, xi, length):
        if index == 0:
            t = np.fmod(xi / length, 2.0)
            return self.x0_scale * np.sin(math.pi * np.minimum(t, 1.0 - t))
        step = self._records[index - 1].step
        return np.array([step.value(float(v)) for v in xi])

    def centre_value(self, s):
        """w_s(a/2, b/2): the node i = j = 1 of the grid nx = ny = 2."""
        return float(self.field_on_grid(s, 2, 2, i=[1], j=[1])[0, 0])

    def field_change(self, s, nx, ny):
        """max |w_s - w_{s-1}| / max |w_s| over all nodes of the grid (s >= 2).

        Uncertainty against the exact sequence: about field_error of both
        half-steps (relative to max |w|).
        """
        self._check_s(s)
        if s < 2:
            raise ValueError("the change is defined for s >= 2 (X0 is not an approximation)")
        w = self.field_on_grid(s, nx, ny)
        return float(np.max(np.abs(w - self.field_on_grid(s - 1, nx, ny))) / np.max(np.abs(w)))

    def _multiplier_coefficients(self, index):
        if index == 0:
            return [self.x0_scale]
        return self._records[index - 1].step.coefficients()

    def energy(self, s):
        """Continuous energy of w_s and its diagnostics (HalfStepEnergy)."""
        self._check_s(s)
        x, y = _x_index(s), _y_index(s)
        Px, Py = self._parseval[x], self._parseval[y]
        integral = self._integrals[x] * self._integrals[y]
        norm2 = self.D * math.fsum((Px.C * Py.A, 2.0 * Px.B * Py.B, Px.A * Py.C))
        energy = -0.5 * self.q * integral
        energy_full = 0.5 * norm2 - self.q * integral
        error = _compose(self._integral_errors[x], self._integral_errors[y], 2 * U)
        if s == 1:
            delta = change_norm2 = change_rel = change_rel_error = None
        else:
            # the multiplier changed on half-step s replaced that of s - 2;
            # the other one (index s - 1) is common to w_s and w_{s-1}
            new = self._multiplier_coefficients(s)
            old = self._multiplier_coefficients(s - 2)
            size = max(len(new), len(old))
            new = new + [0.0] * (size - len(new))
            old = old + [0.0] * (size - len(old))
            # both versions of the changed multiplier live on ell_s
            d = parseval_coefficients([p - r for p, r in zip(new, old)], ell=self.length(s), q=self.q)
            P = self._parseval[s - 1]
            change_norm2 = self.D * math.fsum((P.C * d.A, 2.0 * P.B * d.B, P.A * d.C))
            change_rel = math.sqrt(change_norm2 / norm2)
            e_new = self._records[s - 1].budget.coefficient_error
            e_old = ROUND_X0 if s == 2 else self._records[s - 3].budget.coefficient_error
            change_rel_error = e_new + e_old + 2 * EPS_TAIL + 2 * ROUND_PARSEVAL
            previous = self._integrals[_x_index(s - 1)] * self._integrals[_y_index(s - 1)]
            delta = energy - (-0.5 * self.q * previous)
        return HalfStepEnergy(
            s=s,
            k=s // 2,
            integral=integral,
            energy=energy,
            energy_full=energy_full,
            energy_norm2=norm2,
            error=error,
            rounding=ROUND_ENERGY,
            delta=delta,
            change_norm2=change_norm2,
            change_rel=change_rel,
            change_rel_error=change_rel_error,
        )


def _integral(step):
    """Integral of f = sum c_m sin(lambda_m xi) over (0, ell): 2 sum c_m / lambda_m (ЧВ-18)."""
    lambdas = _odd_lambdas(step.n_terms, step.ell)
    return 2.0 * math.fsum(c / lam for c, lam in zip(step.coefficients(), lambdas))


def _compose(*relative):
    """prod(1 + r) - 1 without cancellation (log1p, expm1)."""
    return math.expm1(math.fsum(math.log1p(r) for r in relative))


def _x_index(s):
    """Half-step whose multiplier is the x-factor of w_s (0 stands for X0)."""
    return s if s % 2 == 0 else s - 1


def _y_index(s):
    """Half-step whose multiplier is the y-factor of w_s."""
    return s if s % 2 == 1 else s - 1


def _check_indices(name, indices, n):
    indices = list(indices)
    for p in indices:
        if isinstance(p, bool) or not isinstance(p, (int, np.integer)) or not 0 <= p <= n:
            raise ValueError(f"{name} indices must be integers in 0..{n}, got {p!r}")
    return [int(p) for p in indices]
