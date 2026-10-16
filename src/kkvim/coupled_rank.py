"""Rank diagnostics and QR orthogonalization of coupled sets, N >= 2 (R4, step 16; D-046, D-047). Internal.

Levels (AGENTS.md, section 2): the field w = U V^T and its half-step spaces belong to the
authors' method; rho_N, kappa_rep and the transformation V_prev <- V_prev R^T are the
project's discretization (R4_NUMERICAL_SPEC 5.2, 5.4, 5.4.1; ChV4-12, ChV4-13); the
Householder QR, the SVD of an N x N matrix and the array handling are engineering.

- 5.2: weighted factors U_w = sqrt(h_u) U, V_w = sqrt(h_v) V; thin QR U_w = Q_U R_U,
  V_w = Q_V R_V; sigma_k(W_w) = sigma_k(R_U R_V^T); rho_N = sigma_N / sigma_1;
  kappa_rep = |R_U|_2 |R_V|_2 / sigma_1. The diagnostic QR never changes the stored sets.
- 5.4.1 (ii): singular values of R_V D_c, D_c = diag(1 / |v_{w,j}|_2), for the received set.
- 5.4: working orthogonalization of the received set, sqrt(h) V = Q R, V <- Q / sqrt(h),
  companion U <- U R^T (multiplication, no inverse of R); the field U V^T is kept up to
  rounding of order u kappa_rep. It is applied only after the field and set diagnostics
  and the guards of the caller have passed.

Nothing here certifies the rank: c_d, c_rank, tau_set and delta_f are NOT_QUALIFIED.
"""

import math
from dataclasses import dataclass

import numpy as np

from kkvim.coupled import CoupledCoefficients, sine_eigenvalues
from kkvim.grid import Grid1D


@dataclass(frozen=True)
class RankDiagnostics:
    """Field and set diagnostics of w = U V^T (5.2, 5.4.1, 5.7); values only, no verdict."""

    sigma: np.ndarray  # singular values of W_w, descending
    rho_N: float  # sigma_N / sigma_1
    kappa_rep: float  # |U_w|_2 |V_w|_2 / |W_w|_2
    cond_U: float  # cond_2(U_w) = cond_2(R_U)
    cond_V: float  # cond_2(V_w) = cond_2(R_V)
    set_sigma: np.ndarray  # singular values of R_V D_c (received set, normalized columns)

    @property
    def set_sigma_min(self) -> float:
        return float(self.set_sigma[-1])

    @property
    def finite(self) -> bool:
        values = [self.rho_N, self.kappa_rep, self.cond_U, self.cond_V, *self.sigma, *self.set_sigma]
        return all(math.isfinite(v) for v in values)


@dataclass(frozen=True)
class Orthogonalized:
    """Received set after QR and the transformed companion set; the field is unchanged in exact arithmetic."""

    received: np.ndarray  # Q / sqrt(h_received)
    companion: np.ndarray  # companion @ R^T
    R: np.ndarray
    field_error: float  # max |U' V'^T - U V^T| / max |U V^T| over the interior nodes


def rank_diagnostics(grid_u: Grid1D, U, grid_v: Grid1D, V) -> RankDiagnostics:
    """rho_N, kappa_rep, singular values and conditions for w = U V^T; V is the received set."""
    U, V = np.asarray(U, dtype=float), np.asarray(V, dtype=float)
    Uw, Vw = math.sqrt(grid_u.h) * U, math.sqrt(grid_v.h) * V
    with np.errstate(all="ignore"):
        R_U = np.linalg.qr(Uw, mode="r")
        R_V = np.linalg.qr(Vw, mode="r")
        sigma = np.linalg.svd(R_U @ R_V.T, compute_uv=False)
        s_U = np.linalg.svd(R_U, compute_uv=False)
        s_V = np.linalg.svd(R_V, compute_uv=False)
        col = np.linalg.norm(Vw, axis=0)
        set_sigma = np.linalg.svd(R_V / col[np.newaxis, :], compute_uv=False)
        rho = sigma[-1] / sigma[0]
        kappa = s_U[0] * s_V[0] / sigma[0]
        cond_U, cond_V = s_U[0] / s_U[-1], s_V[0] / s_V[-1]
    return RankDiagnostics(sigma=sigma, rho_N=float(rho), kappa_rep=float(kappa), cond_U=float(cond_U),
                           cond_V=float(cond_V), set_sigma=set_sigma)


def mode_conditions(coefficients: CoupledCoefficients, solve_grid: Grid1D) -> tuple[float, float]:
    """(max_k cond_2(M_k), cond_2(A)) of a half-step: system and given-set representation (5.7)."""
    mu = sine_eigenvalues(solve_grid)
    A, B, C = coefficients.A, coefficients.B, coefficients.C
    worst = 0.0
    for m in mu:
        lam = np.linalg.eigvalsh(m * m * A + 2.0 * m * B + C)
        worst = max(worst, lam[-1] / lam[0] if lam[0] > 0 else math.inf)
    lam_A = np.linalg.eigvalsh(A)
    return float(worst), float(lam_A[-1] / lam_A[0]) if lam_A[0] > 0 else math.inf


def orthogonalize_received(grid_received: Grid1D, received, companion) -> Orthogonalized:
    """sqrt(h) V = Q R; V <- Q / sqrt(h); companion <- companion R^T (R4_NUMERICAL_SPEC 5.4; no explicit R^-1)."""
    V, U = np.asarray(received, dtype=float), np.asarray(companion, dtype=float)
    root = math.sqrt(grid_received.h)
    Q, R = np.linalg.qr(root * V, mode="reduced")
    V_new = Q / root
    U_new = U @ R.T
    before = U @ V.T
    after = U_new @ V_new.T
    error = float(np.max(np.abs(after - before)) / np.max(np.abs(before)))
    return Orthogonalized(received=V_new, companion=U_new, R=R, field_error=error)
