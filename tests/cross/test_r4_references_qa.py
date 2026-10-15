"""QA tests of the R4 reference candidates (R4_NUMERICAL_SPEC 9.1, 9.2, 9.5; РВ4-8; all NOT_QUALIFIED as normative
references until the owner's decision): continuous N-term sequence (matrix Parseval recursion), x-oriented s = 1 partial
sums (Levy stable form and m-truncated Navier), the energy reference Pi(omega_0), and the full omega_0 by Levy on R4
grids. Accepted R2/R3 references are only called, never modified. Thresholds check the QA tools (not R4 tolerances).
High-precision checks run only where mpmath is present (temporary QA environment: uv run --with mpmath).
"""

import math

import numpy as np
import pytest

from kkvim_reference.levy import levy_field_bounds, levy_series_w_grid
from kkvim_reference.navier import NavierRectangle
from kkvim_reference.parseval_recursion import ParsevalRecursion
from kkvim_reference.r4_continuous_sequence import continuous_sequence, field_at_nodes, sine_values
from kkvim_reference.r4_energy_reference import S_TAIL, closed_form_sum, energy_sum, omega0_energy
from kkvim_reference.r4_levy_partial import (
    bracket,
    levy_x_partial_at_nodes,
    levy_x_partial_rounding_model,
    navier_m_tail_bound,
    navier_m_truncated_at_nodes,
)

RATIOS = (0.2, 1 / 3, 1.0, 2.0, 5.0)


# --- continuous sequence -----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("r", RATIOS)
def test_continuous_n1_reproduces_the_qualified_r3_recursion(r):
    steps = continuous_sequence(1.0, r, 1, 4, 2001)
    P = ParsevalRecursion(a=1.0, b=r, q=1.0, D=1.0, s_max=4)
    for s in range(1, 5):
        ref = P.field_on_grid(s, 40, 40)
        assert np.max(np.abs(field_at_nodes(steps[s - 1], 40, 40) - ref)) <= 1e-12 * np.max(np.abs(ref))
        assert abs(steps[s - 1].energy - P.energy(s).energy) <= 1e-14 * abs(P.energy(s).energy)


@pytest.mark.parametrize("N", (2, 3))
@pytest.mark.parametrize("r", RATIOS)
def test_recursion_equals_the_independent_least_squares_form(N, r):
    a = continuous_sequence(1.0, r, N, 4, 801)
    b = continuous_sequence(1.0, r, N, 4, 801, method="ls")
    for x, y in zip(a, b):
        wa, wb = field_at_nodes(x, 48, 48), field_at_nodes(y, 48, 48)
        assert np.max(np.abs(wa - wb)) <= 1e-13 * np.max(np.abs(wa))
        assert abs(x.energy - y.energy) <= 1e-14 * abs(x.energy)


@pytest.mark.parametrize("N", (2, 3))
@pytest.mark.parametrize("r", (0.2, 1.0, 5.0))
def test_mode_doubling_and_energy_monotonicity(N, r):
    a = continuous_sequence(1.0, r, N, 4, 1001)
    b = continuous_sequence(1.0, r, N, 4, 2001)
    for x, y in zip(a, b):
        wa, wb = field_at_nodes(x, 64, 64), field_at_nodes(y, 64, 64)
        # the change from M = 1001 to 2001 stays within the strict one-half-step tails (diagnostic, not a proof)
        assert np.max(np.abs(wa - wb)) <= 4 * max(s.tail_bound for s in a) + 1e-14 * np.max(np.abs(wb))
    e = [x.energy for x in b]
    assert all(e2 <= e1 + 1e-14 * abs(e1) for e1, e2 in zip(e, e[1:]))
    pi0, bound = omega0_energy(1.0, r)
    assert all(x >= pi0 - bound for x in e)


# --- s = 1 partial sums ------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("N", (1, 2, 3, 6))
@pytest.mark.parametrize("r", RATIOS)
def test_s1_partial_sums_agree(N, r):
    modes = list(range(1, 2 * N, 2))
    L = levy_x_partial_at_nodes(1.0, r, 1.0, 1.0, modes, 40, 80)
    Nv = navier_m_truncated_at_nodes(1.0, r, 1.0, 1.0, modes, 40, 80, 40001)
    C = field_at_nodes(continuous_sequence(1.0, r, N, 1, 2001)[0], 40, 80)
    m = np.max(np.abs(Nv))
    assert navier_m_tail_bound(1.0, r, 1.0, 1.0, modes, 40001) <= 1e-16 * m
    assert np.max(np.abs(C - Nv)) <= 1e-12 * m
    if r >= 1:  # alpha_m >= pi/2: the stable form is accurate, the rounding model applies
        model = levy_x_partial_rounding_model(1.0, r, 1.0, 1.0, modes, 40, 80)
        assert np.max(np.abs(L - Nv) - 4 * model) <= 4e-16 * m  # model of both sums; the Navier sum is not modelled
        ls = levy_series_w_grid(np.arange(41) / 40, r * np.arange(81) / 80, a=1.0, b=r, q=1.0, D=1.0, M=2 * N - 1)
        assert np.max(np.abs(L - ls)) <= 1e-14 * m  # accepted Levy series along the short side = x for r >= 1
    else:
        with pytest.raises(ValueError, match="alpha_m < pi/2"):
            levy_x_partial_rounding_model(1.0, r, 1.0, 1.0, modes, 40, 80)
        assert np.max(np.abs(L - Nv)) <= 1e-13 * m  # accuracy loss of the stable form for small alpha (section 6.1)


def test_bracket_edges_and_symmetry():
    for alpha in (0.157, 0.5, math.pi / 2, 12.0, 800.0):
        assert bracket(alpha, 0.0) == 0.0  # w = 0 on the edges y = 0, b exactly
        assert np.all(np.isfinite(bracket(alpha, np.linspace(0, 1, 101))))


# --- energy Pi(omega_0) --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("r", (0.2, 1 / 3, 0.5, 2 / 3, 1.0, 1.4, 1.5, 2.0, 3.0, 5.0))
def test_energy_reference(r):
    S1, S2 = energy_sum(1.0, r, 2001), energy_sum(1.0, r, 4001)
    assert 0.0 <= S2 - S1 <= S_TAIL(1.0, r, 2001) + 8 * 2.0**-53 * S2  # truncation within the strict tail
    assert abs(closed_form_sum(1.0, r, 4001) - S2) <= 1e-12 * S2  # independent semi-analytic n-sums
    pi0, bound = omega0_energy(1.0, r)
    nav = NavierRectangle(a=1.0, b=r, q=1.0, D=1.0, nu=0.3).deflection_integral()
    assert abs(nav.Pi_M - pi0) <= nav.Pi_tail_bound + bound + 32 * 2.0**-53 * abs(pi0)
    assert bound <= 5e-15 * abs(pi0)


# --- full omega_0 by Levy on R4 grids: P-12 type cross-check against Navier (D-038 rule) ---------------------------

@pytest.mark.parametrize("r, grid", [(0.2, (40, 40)), (0.2, (160, 32)), (1 / 3, (144, 48)), (1.0, (80, 80)),
                                     (2.0, (80, 160)), (5.0, (64, 320)), (5.0, (160, 160))])
def test_levy_omega0_against_navier_on_r4_grids(r, grid):
    nx, ny = grid
    xs, ys = np.arange(nx + 1) / nx, r * np.arange(ny + 1) / ny
    wl = levy_series_w_grid(xs, ys, a=1.0, b=r, q=1.0, D=1.0)
    nav = NavierRectangle(a=1.0, b=r, q=1.0, D=1.0, nu=0.3).at_nodes(nx, ny, diagnostics=False)
    bounds = levy_field_bounds(a=1.0, b=r, q=1.0, D=1.0)
    B_N = 0.0021881 * r**2 / 2559**2
    allowed = B_N + bounds.truncation + nav.rounding_w + bounds.rounding_model  # D-038, item 2 (q = D = a = 1)
    assert np.max(np.abs(wl - nav.w_coef)) <= allowed


# --- high precision (temporary QA environment only; skipped without mpmath) -------------------------------------------

def _mp():
    mp = pytest.importorskip("mpmath")
    mp.mp.dps = 50
    return mp


@pytest.mark.parametrize("r", (0.2, 1.0, 5.0))
@pytest.mark.parametrize("N", (2, 3))
def test_high_precision_recursion(r, N):
    """The double recursion against the same truncated recursion (M = 41) in 50 digits: rounding of the algorithm.

    The raw double path (orthonormalize=False) loses about u cond(given set) (cond up to ~1e5 at N = 3, r = 0.2);
    the default orthonormalised path is checked here."""
    mp = _mp()
    M, a, b = 41, mp.mpf(1), mp.mpf(r)
    modes = list(range(1, M + 1, 2))

    def halfstep(coef, lg, ln):
        k = [m * mp.pi / lg for m in modes]
        G = {p: mp.matrix(N, N) for p in (0, 2, 4)}
        for p in (0, 2, 4):
            for i in range(N):
                for j in range(N):
                    G[p][i, j] = lg / 2 * mp.fsum(k[t] ** p * coef[t][i] * coef[t][j] for t in range(len(modes)))
        F = mp.matrix([mp.fsum(coef[t][i] * 2 * lg / (modes[t] * mp.pi) for t in range(len(modes))) for i in range(N)])
        new = []
        for n in modes:
            nu = n * mp.pi / ln
            Mt = nu**4 * G[0] + 2 * nu**2 * G[2] + G[4]
            sol = mp.lu_solve(Mt, F * (4 / (n * mp.pi)))
            new.append([sol[i] for i in range(N)])
        return new

    x = [[mp.mpf(1) if t == i else mp.mpf(0) for i in range(N)] for t in range(len(modes))]
    # default double path (orthonormal given sets) against the raw 50-digit path: equal fields in exact arithmetic
    steps = continuous_sequence(1.0, r, N, 4, M)
    y = None
    for s in range(1, 5):
        if s % 2:
            y = halfstep(x, a, b)
        else:
            x = halfstep(y, b, a)
        # compare fields at nodes (representation of the double path is the raw one, as here)
        xs = np.array([[float(v) for v in row] for row in x])
        ys = np.array([[float(v) for v in row] for row in y])
        w_hp = sine_values(xs, 16) @ sine_values(ys, 16).T
        w = field_at_nodes(steps[s - 1], 16, 16)
        assert np.max(np.abs(w - w_hp)) <= 1e-13 * np.max(np.abs(w_hp)), s


@pytest.mark.parametrize("r", (0.2, 1.0, 5.0))
def test_high_precision_energy_sum(r):
    mp = _mp()
    a, b, M = mp.mpf(1), mp.mpf(r), 401
    S_hp = mp.fsum(1 / (m**2 * n**2 * (mp.mpf(m) ** 2 / a**2 + mp.mpf(n) ** 2 / b**2) ** 2)
                   for m in range(1, M + 1, 2) for n in range(1, M + 1, 2))
    S = energy_sum(1.0, r, M)
    assert abs(S - float(S_hp)) <= 9 * 2.0**-53 * float(S_hp)  # rounding of the sum within its stated count


def test_high_precision_bracket_accuracy_domain():
    """ЧВ-Л1 stable form: <= 8u relative for alpha >= pi/2; loses accuracy for small alpha (QA finding, section 6.1)."""
    mp = _mp()

    def exact(alpha, s):
        al, t = mp.mpf(alpha), 1 - mp.mpf(s)
        P = (al * mp.tanh(al) + 2) / (2 * mp.cosh(al))
        return 1 - P * mp.cosh(al * t) + al / (2 * mp.cosh(al)) * t * mp.sinh(al * t)

    ss = np.linspace(0.01, 1.0, 100)
    for alpha, lo, hi in ((math.pi / 2, 0, 8), (7.854, 0, 8), (0.314, 50, 1e4), (0.157, 300, 1e5)):
        T = bracket(alpha, ss)
        rel = max(abs(float(mp.mpf(T[i]) - exact(alpha, ss[i])) / exact(alpha, ss[i])) for i in range(len(ss)))
        assert lo <= rel / 2.0**-53 <= hi, (alpha, rel / 2.0**-53)
