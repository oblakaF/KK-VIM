"""Energy diagnostics of coupled half-steps, N >= 2 (R4, step 16; D-046 RV4-6, D-047). Internal.

Levels (AGENTS.md, section 2): the energy decrease of an exact half-step is a property of
the method (VB-17); the discrete identity, the rounding model and the candidate tau_s^(N)
are the project's discretization (R4_NUMERICAL_SPEC 6.2, 6.3; ChV4-14, ChV4-15); the
array evaluation is engineering.

- Pi_h, a_h, l_h and the rounding model R(w) are the two-dimensional R2 forms
  (``kkvim.energy.discrete_energy`` and ``rounding_threshold``, NUMERICAL_SPEC 4.1, 10.3),
  evaluated on the formed fields, independently of the half-step matrices.
- 6.2, item 3: Delta P_s = P_{s-1} - P_s, Q_s = a_h(d, d) / 2 for d = w_{s-1} - w_s, and
  gamma_s = h sum_i rho_i^T d_i with the nodal residual rho of the solved system and
  d_i = V_prev,i - V_i on the solve grid. P_{s-1} is taken from the arrays used by
  half-step s (after a QR the transformed ones).
- 6.3: eps_a(j) = D^{1/2} (4/h_x^2 + 4/h_y^2) gamma_N |S_j|_h, S = |U| |V|^T;
  B_omega by the two-dimensional residual, |D Delta_h^2 w - q 1|_h / (D^{1/2} (mu_1^x + mu_1^y)),
  (6.3, item 6(b)); the a priori bound (a) is recorded as well;
  R_f^(N) = eps_a(s) |d|_{a_h} + B_omega (eps_a(s-1) + eps_a(s));
  tau_s^(N) = |gamma_s| + 10 [R(w_{s-1}) + R(w_s) + R_f^(N)]; E1: Delta P_s >= -tau;
  E2: |Delta P_s - Q_s| <= tau; resolved if Q_s >= 10 tau.

tau_s^(N) is a first-order model, not a proven bound; the factor 10 and the resolution rule
are NOT_QUALIFIED for N >= 2. The statuses here are research diagnostics, never PASS.
"""

import math
from dataclasses import dataclass

import numpy as np

from kkvim.coupled import CoupledCoefficients, coupled_residual, sine_eigenvalues
from kkvim.energy import discrete_energy, rounding_threshold
from kkvim.grid import Grid1D

UNIT_ROUNDOFF = 2.0**-53
FACTOR_10 = 10.0  # R2 form (I-9); NOT_QUALIFIED for N >= 2

NOT_APPLICABLE = "NOT_APPLICABLE"
UNRESOLVED = "UNRESOLVED"
E1_E2_HOLD = "E1_E2_HOLD_PROVISIONAL"
E1_VIOLATED = "E1_VIOLATED_PROVISIONAL"
E2_VIOLATED = "E2_VIOLATED_PROVISIONAL"


@dataclass(frozen=True)
class CoupledEnergy:
    """Energy quantities of half-step s (research diagnostics, PROVISIONAL / NOT_QUALIFIED)."""

    P: float  # Pi_h(w_s)
    norm_a: float  # |w_s|_{a_h}
    eps_a: float  # eps_a(s)
    B_omega: float  # bound of |w_s - omega_h|_{a_h} by the 2-D residual (6.3, 6(b))
    B_omega_a_priori: float  # |q| sqrt(ab) / (D^{1/2} (mu_1^x + mu_1^y)) (6.3, 6(a))
    R: float  # R(w_s), R2 rounding model
    P_prev: float | None = None  # P_{s-1} from the arrays used by half-step s
    delta_P: float | None = None
    Q: float | None = None
    gamma: float | None = None
    R_prev: float | None = None
    eps_a_prev: float | None = None
    R_f: float | None = None
    tau: float | None = None
    e1: bool | None = None
    e2: bool | None = None
    status: str = NOT_APPLICABLE


def field_from_sets(grid_x: Grid1D, grid_y: Grid1D, X, Y) -> np.ndarray:
    """w = X Y^T on all tensor nodes with the exact zero boundary."""
    X, Y = np.asarray(X, dtype=float), np.asarray(Y, dtype=float)
    Xf = np.zeros((grid_x.n + 1, X.shape[1]))
    Yf = np.zeros((grid_y.n + 1, Y.shape[1]))
    Xf[1:-1], Yf[1:-1] = X, Y
    return Xf @ Yf.T


def eps_a(grid_x: Grid1D, grid_y: Grid1D, X, Y, D: float) -> float:
    """D^{1/2} (4/h_x^2 + 4/h_y^2) gamma_N |S|_h with S = |X| |Y|^T (6.3, items 3-4)."""
    N = np.asarray(X).shape[1]
    gamma_N = N * UNIT_ROUNDOFF / (1.0 - N * UNIT_ROUNDOFF)
    S = np.abs(np.asarray(X, dtype=float)) @ np.abs(np.asarray(Y, dtype=float)).T
    norm_S = math.sqrt(grid_x.h * grid_y.h * float(np.sum(S * S)))
    return math.sqrt(D) * (4.0 / grid_x.h**2 + 4.0 / grid_y.h**2) * gamma_N * norm_S


def omega_bounds(grid_x: Grid1D, grid_y: Grid1D, w, D: float, q: float) -> tuple[float, float]:
    """(B_omega by the 2-D residual, a priori bound) for |w - omega_h|_{a_h} (6.3, item 6)."""
    mu = sine_eigenvalues(grid_x)[0] + sine_eigenvalues(grid_y)[0]
    inner = np.asarray(w, dtype=float)[1:-1, 1:-1]
    lap2 = _laplacian(grid_x, grid_y, _laplacian(grid_x, grid_y, inner))
    r = D * lap2 - q
    norm_r = math.sqrt(grid_x.h * grid_y.h * float(np.sum(r * r)))
    root = math.sqrt(D) * mu
    return norm_r / root, abs(q) * math.sqrt(grid_x.length * grid_y.length) / root


def first_energy(grid_x: Grid1D, grid_y: Grid1D, X, Y, D: float, q: float) -> CoupledEnergy:
    """s = 1: P_1 and the quantities of w_1; the test is NOT_APPLICABLE (no previous field)."""
    w = field_from_sets(grid_x, grid_y, X, Y)
    e = discrete_energy(grid_x, grid_y, w, D, q)
    bw, bw_a = omega_bounds(grid_x, grid_y, w, D, q)
    return CoupledEnergy(P=e.pi, norm_a=e.norm_a, eps_a=eps_a(grid_x, grid_y, X, Y, D), B_omega=bw,
                         B_omega_a_priori=bw_a, R=rounding_threshold(grid_x, grid_y, w, D, q))


def pair_energy(grid_x: Grid1D, grid_y: Grid1D, prev_X, prev_Y, X, Y, D: float, q: float, *,
                solve_grid: Grid1D, coefficients: CoupledCoefficients, received, companion_prev) -> CoupledEnergy:
    """Energy test of half-step s >= 2 from w_{s-1} = prev_X prev_Y^T (arrays of half-step s) and w_s = X Y^T."""
    w_prev = field_from_sets(grid_x, grid_y, prev_X, prev_Y)
    w_now = field_from_sets(grid_x, grid_y, X, Y)
    e_prev = discrete_energy(grid_x, grid_y, w_prev, D, q)
    e_now = discrete_energy(grid_x, grid_y, w_now, D, q)
    d = w_prev - w_now
    e_d = discrete_energy(grid_x, grid_y, d, D, q)
    Q = 0.5 * e_d.a
    delta_P = e_prev.pi - e_now.pi
    rho, _ = coupled_residual(solve_grid, coefficients, D, received)
    gamma = solve_grid.h * float(np.sum(rho * (np.asarray(companion_prev) - np.asarray(received))))
    R_prev = rounding_threshold(grid_x, grid_y, w_prev, D, q)
    R_now = rounding_threshold(grid_x, grid_y, w_now, D, q)
    ea_prev = eps_a(grid_x, grid_y, prev_X, prev_Y, D)
    ea_now = eps_a(grid_x, grid_y, X, Y, D)
    bw, bw_a = omega_bounds(grid_x, grid_y, w_now, D, q)
    R_f = ea_now * e_d.norm_a + bw * (ea_prev + ea_now)
    tau = abs(gamma) + FACTOR_10 * (R_prev + R_now + R_f)
    e1 = bool(delta_P >= -tau)
    e2 = bool(abs(delta_P - Q) <= tau)
    if Q < FACTOR_10 * tau:
        status = UNRESOLVED
    elif not e1:
        status = E1_VIOLATED
    elif not e2:
        status = E2_VIOLATED
    else:
        status = E1_E2_HOLD
    return CoupledEnergy(P=e_now.pi, norm_a=e_now.norm_a, eps_a=ea_now, B_omega=bw, B_omega_a_priori=bw_a, R=R_now,
                         P_prev=e_prev.pi, delta_P=delta_P, Q=Q, gamma=gamma, R_prev=R_prev, eps_a_prev=ea_prev,
                         R_f=R_f, tau=tau, e1=e1, e2=e2, status=status)


def _laplacian(grid_x: Grid1D, grid_y: Grid1D, inner: np.ndarray) -> np.ndarray:
    """Delta_h on the interior nodes with zero values on the boundary (L_x and L_y with zero ends)."""
    full = np.zeros((inner.shape[0] + 2, inner.shape[1] + 2))
    full[1:-1, 1:-1] = inner
    lx = (full[:-2, 1:-1] - 2.0 * inner + full[2:, 1:-1]) / grid_x.h**2
    ly = (full[1:-1, :-2] - 2.0 * inner + full[1:-1, 2:]) / grid_y.h**2
    return lx + ly
