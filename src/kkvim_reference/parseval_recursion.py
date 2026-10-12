"""Semi-analytic KK-VIM sequence s = 1, 2, 3 by sine series and Parseval (R2).

Independent of the finite-difference solver: no grid operators, no grid
quadratures, no matrices, no imports of ``kkvim`` (D-028, D-033). Square SSSS
plate of side a, uniform load q != 0, N = 1, counters by D-030.

Half-steps (MATH_SPEC.md, sections 4.3-4.5; NUMERICAL_SPEC.md, 15.3):

    G1 = (a/2, pi^2/(2a), pi^4/(2a^3), 2 q a/pi)       from X0 = sin(pi x/a),
    s = 1:  G1 -> Y1 (sine series on (0, a) in y),      w1 = X0 * Y1,
    s = 2:  Y1 -> G2 (Parseval) -> X2 (series in x),    w2 = X2 * Y1,
    s = 3:  X2 -> G3 (Parseval) -> Y3 (series in y),    w3 = X2 * Y3.

Each half-step is solved by ``SineHalfStep`` (ЧВ-17). The coefficients of the
next half-step follow from the kept coefficients c_m, odd m, by Parseval
(NUMERICAL_SPEC.md, 15.2, ЧВ-18), lambda_m = m pi / ell:

    A = (ell/2) sum c_m^2,  B = (ell/2) sum lambda_m^2 c_m^2,
    C = (ell/2) sum lambda_m^4 c_m^2,  F = 2 q sum c_m / lambda_m,

all sums by math.fsum. No normalisation is applied: the multipliers are
stored unnormalised, so the field is the plain product of the two multipliers
(MATH_SPEC.md, В-13). An optional scale of X0 (``x0_scale``) multiplies G1 by
(c^2, c^2, c^2, c); the fields do not depend on it.

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
ПОДТВЕРЖДЁН]). Doubling M is not used as a proof of accuracy.

Area of validity: only the R2 setting (square SSSS, q = const, N = 1) and
s <= 3 (NUMERICAL_SPEC.md, 15.1, 15.7). Not a solution of the 2D plate problem.
"""

import math
from dataclasses import dataclass

import numpy as np

from kkvim_reference.sine_halfstep import SineHalfStep, _sin_pi_ratio

__all__ = [
    "EPS_TAIL",
    "Coefficients",
    "HalfStepBudget",
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

S_MAX = 3


def _check_finite(name, value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {value!r}")


def _check_positive(name, value):
    _check_finite(name, value)
    if not value > 0:
        raise ValueError(f"{name} must be positive, got {value!r}")


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
        relative tails used in the choice of M (all <= EPS_TAIL);
    input_error -- rho_s, relative error of every component of G_s;
    coefficient_error -- r_c, relative error of every c_m caused by rho_s;
    rounding_coefficients -- rounding of the Parseval sums (relative);
    rounding_values -- rounding of one series value (absolute);
    multiplier_error -- total relative error of the multiplier values;
    field_error -- total relative error of w_s values (to max|w_s|);
    invariant_error -- relative error of B/A, C/A, F^2/A of G_s;
    first_order_input_error -- sum_j 4^(s-j) tau_j of the first-order rule.
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


class ParsevalRecursion:
    """KK-VIM sequence s = 1..s_max (s_max <= 3) for the square SSSS plate.

    a -- side of the square, D -- flexural rigidity, q -- uniform load (q != 0),
    x0_scale -- scale of X0 = x0_scale * sin(pi x/a); the fields do not
    depend on it (MATH_SPEC.md, В-13).
    """

    def __init__(self, *, a, q, D, s_max=S_MAX, x0_scale=1.0, eps=EPS_TAIL):
        _check_positive("a", a)
        _check_positive("D", D)
        _check_finite("q", q)
        _check_finite("x0_scale", x0_scale)
        if q == 0 or x0_scale == 0:
            raise ValueError("q and x0_scale must be nonzero (MATH_SPEC.md, 4.6)")
        if isinstance(s_max, bool) or not isinstance(s_max, int) or not 1 <= s_max <= S_MAX:
            raise ValueError(f"s_max must be 1, 2 or 3, got {s_max!r}")
        self.a = float(a)
        self.q = float(q)
        self.D = float(D)
        self.x0_scale = float(x0_scale)
        self.s_max = s_max
        self.eps = eps

        records = []
        G = first_coefficients(a=self.a, q=self.q, x0_scale=self.x0_scale)
        rho = ROUND_G1
        rho_first_order = ROUND_G1
        multiplier_errors = {0: ROUND_X0}
        for s in range(1, s_max + 1):
            record = self._halfstep(s, G, rho, rho_first_order, multiplier_errors)
            records.append(record)
            b = record.budget
            G = parseval_coefficients(record.step.coefficients(), ell=self.a, q=self.q)
            tail = max(b.tail_A_rel, b.tail_B_rel, b.tail_C_rel, b.tail_F_rel)
            # The tails were bounded with the computed G_s; for the exact G_s
            # the constant K and the partial sums move by at most r_c.
            r_c = b.coefficient_error
            tail_exact = tail * (1.0 + r_c) ** 2 / (1.0 - r_c) ** 2
            rho = _compose(r_c, r_c, tail_exact, ROUND_PARSEVAL)
            rho_first_order = 4.0 * rho_first_order + tail + ROUND_PARSEVAL
        self._records = tuple(records)

    def _halfstep(self, s, G, rho, rho_first_order, multiplier_errors):
        ell = self.a
        M = select_odd_index(G, D=self.D, ell=ell, q=self.q, eps=self.eps)
        step = SineHalfStep(A=G.A, B=G.B, C=G.C, F=G.F, D=self.D, ell=ell, M=M)
        terms = step.coefficients()
        f_lower = _max_lower_bound(step.coefficient(1), step.coefficient(3))
        rel = _relative_tails(G, terms, D=self.D, ell=ell, q=self.q, M=M, f_lower=f_lower)

        r_c = 2.0 * rho / (1.0 - rho)  # (1 + rho)/(1 - rho) - 1
        abs_sum = step.abs_coefficient_sum()
        rounding_values = ROUND_VALUES * abs_sum
        error_abs = r_c * abs_sum + step.tail_bound * (1.0 + r_c) + rounding_values
        multiplier_error = error_abs / (f_lower * (1.0 - r_c) - error_abs)
        multiplier_errors[s] = multiplier_error

        e_x = multiplier_errors[_x_index(s)]
        e_y = multiplier_errors[_y_index(s)]
        field_error = _compose(e_x, e_y, U)
        # (1 + rho)^2 / (1 - rho) (1 + 2u) - 1: F^2/A is the worst of the three
        invariant_error = _compose(rho, rho, rho / (1.0 - rho), 2 * U)

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
        """w_s at the nodes x_i = i a/nx, y_j = j a/ny of a tensor grid.

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
