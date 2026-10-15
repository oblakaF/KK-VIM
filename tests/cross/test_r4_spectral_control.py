"""QA tests of the independent two-dimensional spectral control of discrete KK-VIM half-steps (R4 prequalification).

kkvim_reference.r4_spectral_control minimises the accepted discrete Pi_h in the two-dimensional sine coefficients of
the field (weighted least squares per mode, no A/B/C/F, no ЧФ-3). Checked: (1) N = 1 against the accepted R3 solver
(the only accepted discrete implementation; R2/R3 path, D-043) on five ratios and square and rectangular grids;
(2) internal identities that hold for every N: closed form of s = 1, spectral energy against nodal 2D assembly,
stationarity on M_h by nodal assembly, the energy identity of ЧВ4-14, reflections; (3) representation invariance
(GL(N), МATH_SPEC ВБ-16, ВБ-24); (4) s = 1 against the continuous partial sum (second order, ЧВ4-7 proven for s = 1).
The thresholds below check the QA tool itself; they are NOT tolerances of R4 (A4-1..A4-14 remain NOT_QUALIFIED).
"""

import math

import numpy as np
import pytest

from kkvim.rectangle import rectangle_halfsteps
from kkvim_reference.r4_levy_partial import navier_m_tail_bound, navier_m_truncated_at_nodes
from kkvim_reference.r4_spectral_control import (
    SpectralGrid,
    dst_matrix,
    eigenvalues,
    first_halfstep_closed_form,
    odd_sine_set,
    run_control,
)

RATIOS = (0.2, 1 / 3, 1.0, 2.0, 5.0)
GRIDS = ((20, 20), (40, 60), (24, 120))


def test_dst_matrix_and_spectrum_of_the_second_difference():
    for n, length in ((4, 1.0), (13, 2.5), (40, 0.2)):
        P = dst_matrix(n)
        assert np.allclose(P, P.T, atol=0) and np.max(np.abs(P @ P - np.eye(n - 1))) < 64 * n * 2.0**-53
        L = (np.diag(np.full(n - 2, 1.0), -1) - 2 * np.eye(n - 1) + np.diag(np.full(n - 2, 1.0), 1)) / (length / n) ** 2
        mu = eigenvalues(n, length)
        assert np.max(np.abs(-L - P @ np.diag(mu) @ P)) <= 1e-12 * np.max(mu)


@pytest.mark.parametrize("r", RATIOS)
@pytest.mark.parametrize("grid", GRIDS)
def test_n1_control_reproduces_the_accepted_r3_solver(r, grid):
    nx, ny = grid
    steps = run_control(SpectralGrid(1.0, r, nx, ny), 1, 4)
    hist = rectangle_halfsteps(1.0, r, 1.0, 1.0, nx, ny, 4)
    for s in range(1, 5):
        w = hist.field(s)[1:-1, 1:-1]
        assert np.max(np.abs(steps[s - 1].W - w)) <= 1e-11 * np.max(np.abs(w)), (r, grid, s)


@pytest.mark.parametrize("N", (2, 3, 6))
@pytest.mark.parametrize("r", RATIOS)
def test_internal_identities(N, r):
    g = SpectralGrid(1.0, r, 4 * N + 8, 8 * N)
    steps = run_control(g, N, 5)
    cf = first_halfstep_closed_form(g, N)
    assert np.max(np.abs(steps[0].W - cf)) <= 1e-13 * np.max(np.abs(cf))
    energies = [x.energy_spectral for x in steps]
    for x in steps:
        assert abs(x.energy_nodal - x.energy_spectral) <= 1e-13 * abs(x.energy_spectral)
        assert x.projection_residual <= 1e-13  # stationarity on M_h(given set), nodal assembly, relative to rounding scale
        if x.energy_identity_defect is not None:
            assert x.energy_identity_defect <= 1e-12
        assert np.max(np.abs(x.W - x.W[::-1, :])) <= 1e-12 * np.max(np.abs(x.W))  # reflections (ВБ-22, item 6)
        assert np.max(np.abs(x.W - x.W[:, ::-1])) <= 1e-12 * np.max(np.abs(x.W))
    # Pi_h does not increase (ЧВ4-14); late changes are below rounding (|dPi|/|Pi| ~ 1e-16, unresolved), so only an
    # increase above the rounding level of the energy sum would be a defect of the control
    assert all(e2 <= e1 + 1e-14 * abs(e1) for e1, e2 in zip(energies, energies[1:]))
    assert all(e >= g.omega_energy() * (1 + 1e-14) for e in energies)  # not below the discrete minimum (rounding)


def test_nested_subspaces_at_s1_and_rank_loss_guard():
    g = SpectralGrid(1.0, 2.0, 40, 80)
    e = [run_control(g, N, 1)[0].energy_spectral for N in (1, 2, 3)]
    assert e[0] >= e[1] >= e[2]  # M_1 subset of M_2 subset of M_3 at s = 1 (odd sines)
    with pytest.raises(ValueError, match="rank loss"):
        run_control(SpectralGrid(1.0, 1.0, 5, 40), 3, 1)  # n_x < 2N (ЧВ4-8)
    with pytest.raises(ValueError, match="rank loss"):
        run_control(SpectralGrid(1.0, 1.0, 40, 5), 3, 1)  # n_y < 2N (ЧВ4-9)


@pytest.mark.parametrize("cond", (1.0, 1e3, 1e6))
def test_representation_invariance(cond):
    """X0 T for T in GL(N) and the raw (non-orthonormalised) path give the same fields (span invariance, ВБ-16)."""
    g = SpectralGrid(1.0, 1 / 3, 36, 24)
    rng = np.random.default_rng(1)
    Q1, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    Q2, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    T = Q1 @ np.diag(np.geomspace(1, 1 / cond, 3)) @ Q2
    base = run_control(g, 3, 4)
    for st in (run_control(g, 3, 4, start=odd_sine_set(36, 3) @ T),
               run_control(g, 3, 4, start=odd_sine_set(36, 3) @ T, orthonormalize=False)):
        for a, b in zip(base, st):
            assert np.max(np.abs(a.W - b.W)) <= 1e-13 * cond * np.max(np.abs(a.W))
            assert abs(a.rho_N - b.rho_N) <= 1e-12 * cond


def test_cross_terms_vb24():
    """U = (s1, s3) and U' = (s1, s1 + s3) with non-diagonal coefficients, without orthonormalisation: same fields."""
    g = SpectralGrid(1.0, 2.0, 24, 48)
    X = odd_sine_set(24, 2)
    a = run_control(g, 2, 3, orthonormalize=False)
    b = run_control(g, 2, 3, start=np.column_stack([X[:, 0], X[:, 0] + X[:, 1]]), orthonormalize=False)
    for x, y in zip(a, b):
        assert np.max(np.abs(x.W - y.W)) <= 1e-12 * np.max(np.abs(x.W))
        assert abs(x.energy_spectral - y.energy_spectral) <= 1e-14 * abs(x.energy_spectral)


@pytest.mark.parametrize("N", (2, 3))
@pytest.mark.parametrize("r", (0.2, 1.0, 5.0))
def test_s1_second_order_against_the_continuous_partial_sum(N, r):
    """Discrete s = 1 against the m-truncated Navier sum (continuous s = 1, ВБ-23); p = log2(E(n)/E(2n)) near 2."""
    modes = list(range(1, 2 * N, 2))
    errs = []
    for n in (80, 160, 320):
        g = SpectralGrid(1.0, r, n, n)
        W = run_control(g, N, 1)[0].W
        ref = navier_m_truncated_at_nodes(1.0, r, 1.0, 1.0, modes, n, n, 20001)[1:-1, 1:-1]
        assert navier_m_tail_bound(1.0, r, 1.0, 1.0, modes, 20001) < 1e-6 * np.max(np.abs(W - ref))
        errs.append(np.max(np.abs(W - ref)) / np.max(np.abs(ref)))
    for e1, e2 in zip(errs, errs[1:]):
        assert 1.9 <= math.log2(e1 / e2) <= 2.1, errs


@pytest.mark.parametrize("N", (2, 3))
@pytest.mark.parametrize("r, grid", [(0.2, (8, 12)), (1.0, (12, 8)), (5.0, (12, 12))])
def test_high_precision_control_on_small_grids(N, r, grid):
    """Double control against the same discrete minimisation in 50 digits (normal equations of every mode; raw
    representation, no orthonormalisation). Temporary QA environment only (mpmath); skipped otherwise."""
    mp = pytest.importorskip("mpmath")
    mp.mp.dps = 50
    nx, ny = grid
    a, b = mp.mpf(1), mp.mpf(r)

    def psi(n):
        return [[mp.sqrt(mp.mpf(2) / n) * mp.sin(mp.pi * i * k / n) for k in range(1, n)] for i in range(1, n)]

    Px, Py = psi(nx), psi(ny)
    mux = [4 * (nx / a) ** 2 * mp.sin(k * mp.pi / (2 * nx)) ** 2 for k in range(1, nx)]
    muy = [4 * (ny / b) ** 2 * mp.sin(k * mp.pi / (2 * ny)) ** 2 for k in range(1, ny)]
    cx = [mp.fsum(row) for row in Px]
    cy = [mp.fsum(row) for row in Py]
    lam = [[(mux[k] + muy[l]) ** 2 for l in range(ny - 1)] for k in range(nx - 1)]
    ws = [[cx[k] * cy[l] / lam[k][l] for l in range(ny - 1)] for k in range(nx - 1)]
    X0 = [[mp.sin((2 * j + 1) * mp.pi * i / nx) for j in range(N)] for i in range(1, nx)]
    Uh = [[mp.fsum(Px[k][i] * X0[i][j] for i in range(nx - 1)) for j in range(N)] for k in range(nx - 1)]

    def solve(basis, weights, target):  # rows m: min sum_t weights[m][t] (basis[t] . v - target[m][t])^2
        out = []
        for m in range(len(weights)):
            G = mp.matrix(N, N)
            rhs = mp.matrix(N, 1)
            for t, row in enumerate(basis):
                for i in range(N):
                    rhs[i] += weights[m][t] * target[m][t] * row[i]
                    for j in range(N):
                        G[i, j] += weights[m][t] * row[i] * row[j]
            v = mp.lu_solve(G, rhs)
            out.append([v[i] for i in range(N)])
        return out

    steps = run_control(SpectralGrid(1.0, r, nx, ny), N, 4)
    Vh = None
    lamT = [list(col) for col in zip(*lam)]
    wsT = [list(col) for col in zip(*ws)]
    for s in range(1, 5):
        if s % 2:
            Vh = solve(Uh, lamT, wsT)
        else:
            Uh = solve(Vh, lam, ws)
        What = [[mp.fsum(Uh[k][j] * Vh[l][j] for j in range(N)) for l in range(ny - 1)] for k in range(nx - 1)]
        tmp = [[mp.fsum(Px[i][k] * What[k][l] for k in range(nx - 1)) for l in range(ny - 1)] for i in range(nx - 1)]
        W = np.array([[float(mp.fsum(tmp[i][l] * Py[l][j] for l in range(ny - 1))) for j in range(ny - 1)]
                      for i in range(nx - 1)])
        assert np.max(np.abs(steps[s - 1].W - W)) <= 1e-13 * np.max(np.abs(W)), s


# Control values for the future comparison with the solver (R4 prequalification report, section 8): field-level and
# representation-independent quantities; plate 1 x r, q = D = 1, so w_bar = w (q a^4/D) and Pi_h in q^2 a^6 / D;
# start: odd sines (R4_NUMERICAL_SPEC 4.1); counters s, k = s // 2 (D-030). Regression of the QA tool (rel. 1e-12),
# not an R4 tolerance.
CONTROL_VALUES = [
    (2, 0.2, (80, 80), 1, 1.975259151461439e-05, 1.477279374045174e-05, -1.063946835816003e-06),
    (2, 0.2, (80, 80), 2, 2.075515957403344e-05, 1.397841449956289e-05, -1.078505798998348e-06),
    (2, 0.2, (80, 80), 3, 2.075499923714251e-05, 1.397846540896155e-05, -1.078505912875087e-06),
    (2, 0.2, (160, 32), 1, 1.976625125593796e-05, 1.478362550061057e-05, -1.064138942239706e-06),
    (2, 0.2, (160, 32), 2, 2.076911298014957e-05, 1.398915219156418e-05, -1.078727233772858e-06),
    (2, 0.2, (160, 32), 3, 2.076894869245423e-05, 1.398920471732274e-05, -1.078727351288498e-06),
    (2, 1.0, (80, 80), 1, 4.058717834454575e-03, 2.135360722796501e-03, -8.507779745317945e-04),
    (2, 1.0, (80, 80), 2, 4.062202732015154e-03, 2.132112269010738e-03, -8.510334370924200e-04),
    (2, 1.0, (80, 80), 3, 4.062138336730282e-03, 2.132076072541487e-03, -8.510334974179383e-04),
    (2, 5.0, (80, 80), 1, 1.296841221638537e-02, 8.739868870808720e-03, -1.685015235673101e-02),
    (2, 5.0, (80, 80), 2, 1.297192609068029e-02, 8.736539778009794e-03, -1.685165469196432e-02),
    (2, 5.0, (80, 80), 3, 1.297185494656919e-02, 8.736540321704905e-03, -1.685165490253060e-02),
    (2, 5.0, (32, 160), 1, 1.297708052302451e-02, 8.746638277510113e-03, -1.685361301055739e-02),
    (2, 5.0, (32, 160), 2, 1.298064533226571e-02, 8.743251628283042e-03, -1.685511465973479e-02),
    (2, 5.0, (32, 160), 3, 1.298057283237923e-02, 8.743252410274563e-03, -1.685511487859101e-02),
    (3, 0.2, (80, 80), 1, 2.106759983407292e-05, 1.409912291165883e-05, -1.074753325500381e-06),
    (3, 0.2, (80, 80), 2, 2.075518602425508e-05, 1.397845521626923e-05, -1.078506220915824e-06),
    (3, 0.2, (80, 80), 3, 2.075517797398253e-05, 1.397845492000257e-05, -1.078506226887028e-06),
    (3, 0.2, (160, 32), 1, 2.108117057150533e-05, 1.410990562963621e-05, -1.074964413941212e-06),
    (3, 0.2, (160, 32), 2, 2.076913971555187e-05, 1.398919335752363e-05, -1.078727678783688e-06),
    (3, 0.2, (160, 32), 3, 2.076913088083758e-05, 1.398919307266843e-05, -1.078727685791106e-06),
    (3, 1.0, (80, 80), 1, 4.062897633444688e-03, 2.132568438676727e-03, -8.509932337637237e-04),
    (3, 1.0, (80, 80), 2, 4.062277409413115e-03, 2.132209822111895e-03, -8.510343493116317e-04),
    (3, 1.0, (80, 80), 3, 4.062269950228353e-03, 2.132204039698308e-03, -8.510343567339212e-04),
    (3, 5.0, (80, 80), 1, 1.297260839186445e-02, 8.736901726853749e-03, -1.685143068343200e-02),
    (3, 5.0, (80, 80), 2, 1.297198905306741e-02, 8.736534914975017e-03, -1.685165978053129e-02),
    (3, 5.0, (80, 80), 3, 1.297198382993594e-02, 8.736533697529103e-03, -1.685165979779352e-02),
    (3, 5.0, (32, 160), 1, 1.298134721805426e-02, 8.743621268658542e-03, -1.685489190374620e-02),
    (3, 5.0, (32, 160), 2, 1.298070984379181e-02, 8.743246252147266e-03, -1.685512007244446e-02),
    (3, 5.0, (32, 160), 3, 1.298070410341966e-02, 8.743245009792429e-03, -1.685512009412408e-02),
]


@pytest.mark.parametrize("N, r, grid, s, centre, quarter, energy", CONTROL_VALUES)
def test_control_values_for_the_solver_comparison(N, r, grid, s, centre, quarter, energy):
    nx, ny = grid
    x = run_control(SpectralGrid(1.0, r, nx, ny), N, s)[s - 1]
    assert x.s == s and x.k == s // 2
    W = np.zeros((nx + 1, ny + 1))
    W[1:-1, 1:-1] = x.W
    assert abs(W[nx // 2, ny // 2] - centre) <= 1e-12 * abs(centre)
    assert abs(W[nx // 4, ny // 4] - quarter) <= 1e-12 * abs(quarter)
    assert abs(x.energy_spectral - energy) <= 1e-12 * abs(energy)
