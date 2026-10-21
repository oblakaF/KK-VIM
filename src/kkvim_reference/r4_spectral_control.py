"""QA control of the discrete KK-VIM half-steps for N >= 1 in the full two-dimensional sine representation (R4; D-046).

Independent of the main solver: no import of ``kkvim`` (D-028), no one-dimensional coefficient matrices A, B, C, F, no
modal systems ЧФ-3; the control minimises the accepted discrete functional Pi_h directly in the space of the
two-dimensional spectral coefficients of the field. Own derivation (docs/R4_REFERENCE_PREQUALIFICATION.qa-report.md,
section 2):

1. Grid G(l, n): h = l/n, interior nodes i = 1..n-1, (L f)_i = (f_{i-1} - 2 f_i + f_{i+1})/h^2 with f_0 = f_n = 0
   (NUMERICAL_SPEC 3.1). Psi_{ik} = sqrt(2/n) sin(pi i k / n) is symmetric and orthogonal and -L = Psi diag(mu) Psi,
   mu_k = (4/h^2) sin^2(k pi/(2n)) (standard spectrum of the Dirichlet second difference).
2. A field on the interior nodes W (n_x-1 by n_y-1) has the coefficients W^ = Psi_x W Psi_y, and
   Delta_h W = L_x W + W L_y = -Psi_x (diag(mu^x) W^ + W^ diag(mu^y)) Psi_y. Psi is orthogonal, hence
       a_h(w, w) = D h_x h_y sum_{k,l} lambda_kl W^_kl^2,  lambda_kl = (mu^x_k + mu^y_l)^2,
       l_h(w) = q h_x h_y c^xT W^ c^y,  c = Psi 1,
       Pi_h(w) = 1/2 D h_x h_y sum lambda_kl (W^_kl - W*_kl)^2 - 1/2 D h_x h_y sum lambda_kl W*_kl^2,
   with W*_kl = q c^x_k c^y_l / (D lambda_kl), the coefficients of the unconstrained discrete minimiser omega_h.
3. A Y-half-step with the given set U (nodal, n_x-1 by N) minimises Pi_h over W = U V^T. With U^ = Psi_x U and
   V^ = Psi_y V the field coefficients are W^ = U^ V^^T, and the functional splits over the y-modes l:
       min over v_l in R^N of  sum_k lambda_kl (U^_k . v_l - W*_kl)^2,   v_l = row l of V^,
   a weighted linear least-squares problem with the matrix diag(sqrt(lambda_.l)) U^; it is solved by Householder QR
   (no normal equations). The X-half-step is the same with the roles of x and y exchanged. The half-step field is the
   a_h-orthogonal projection of omega_h on M_h(U); nothing else is assumed.
4. Every relation above is an identity of the same discrete problem (no new approximation); the representation of the
   given set does not change the field (span invariance), so an orthonormal basis of the given set may be used.

Nodal quantities (2D assembly of Delta_h by second differences, the path of R2) are computed separately and serve as
an independent check of the spectral ones.

``projection_residual`` checks the variational equation (stationarity of Pi_h on M_h of the given set) by nodal assembly;
it detects errors of the FORMULA (assembly, sign of 2B, cross terms, directions), but it is NOT an estimate of the error
of the field: it is scaled by the rounding scale |Delta_h|^2 |W|, which grows like n^4, so its sensitivity to a field
error falls roughly like n^-4 (Reviewer 3aaca8e, 1.4: a smooth field error 1e-6 passes the 1e-13 check at n = 320).
The quantitative comparisons of QA are the field, the energy and the independent spectral quantities.
"""

import math
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "SpectralGrid",
    "ControlStep",
    "dst_matrix",
    "eigenvalues",
    "odd_sine_set",
    "nodal_energy",
    "nodal_laplacian",
    "field_singular_values",
    "run_control",
    "first_halfstep_closed_form",
]

U_ROUND = 2.0**-53


def _sin_pi_ratio(r, n):
    """sin(pi r / n) for integers r (array), argument reduced in integers to [0, pi/2]."""
    r = np.mod(r, 2 * n)
    sign = np.where(r < n, 1.0, -1.0)
    r1 = np.mod(r, n)
    r2 = np.minimum(r1, n - r1)
    return sign * np.sin(np.pi * r2 / n)


def dst_matrix(n):
    """Psi_{ik} = sqrt(2/n) sin(pi i k / n), i, k = 1..n-1 (symmetric, orthogonal; DST-I, orthonormal)."""
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 2:
        raise ValueError(f"n must be an integer >= 2, got {n!r}")
    k = np.arange(1, n)
    return math.sqrt(2.0 / n) * _sin_pi_ratio(np.outer(k, k), n)


def eigenvalues(n, length):
    """mu_k = (4/h^2) sin^2(k pi/(2n)), k = 1..n-1, h = length/n: eigenvalues of -L on G(length, n)."""
    h = length / n
    return 4.0 / h**2 * np.sin(np.pi * np.arange(1, n) / (2 * n)) ** 2


def odd_sine_set(n, N):
    """Start set X0_j = sin((2j - 1) pi x / a) at the interior nodes of G(a, n), j = 1..N (R4_NUMERICAL_SPEC 4.1)."""
    i = np.arange(1, n)
    return np.stack([_sin_pi_ratio(i * (2 * j - 1), n) for j in range(1, N + 1)], axis=1)


@dataclass(frozen=True)
class SpectralGrid:
    """Plate a x b, q, D on G(a, n_x) x G(b, n_y) with the spectral data of the discrete energy."""

    a: float
    b: float
    nx: int
    ny: int
    q: float = 1.0
    D: float = 1.0
    psi_x: np.ndarray = field(init=False, repr=False)
    psi_y: np.ndarray = field(init=False, repr=False)
    lam: np.ndarray = field(init=False, repr=False)
    w_star: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        for name in ("a", "b", "q", "D"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                raise ValueError(f"{name} must be a finite number, got {v!r}")
        if self.a <= 0 or self.b <= 0 or self.D <= 0 or self.q == 0:
            raise ValueError("a, b, D must be positive and q nonzero")
        object.__setattr__(self, "psi_x", dst_matrix(self.nx))
        object.__setattr__(self, "psi_y", dst_matrix(self.ny))
        mu_x, mu_y = eigenvalues(self.nx, self.a), eigenvalues(self.ny, self.b)
        lam = (mu_x[:, None] + mu_y[None, :]) ** 2
        cx, cy = self.psi_x.sum(axis=1), self.psi_y.sum(axis=1)  # c = Psi 1
        object.__setattr__(self, "lam", lam)
        object.__setattr__(self, "w_star", self.q * np.outer(cx, cy) / (self.D * lam))

    @property
    def hx(self):
        return self.a / self.nx

    @property
    def hy(self):
        return self.b / self.ny

    @property
    def cx(self):
        return self.psi_x.sum(axis=1)

    @property
    def cy(self):
        return self.psi_y.sum(axis=1)

    def to_spectral(self, W):
        return self.psi_x @ W @ self.psi_y

    def to_nodal(self, W_hat):
        return self.psi_x @ W_hat @ self.psi_y

    def spectral_energy(self, W_hat):
        """Pi_h by the spectral formula of item 2."""
        a_term = 0.5 * self.D * self.hx * self.hy * math.fsum((self.lam * W_hat * W_hat).ravel().tolist())
        l_term = self.q * self.hx * self.hy * float(self.cx @ W_hat @ self.cy)
        return a_term - l_term

    def spectral_norm2(self, W_hat):
        """a_h(w, w) by the spectral formula."""
        return self.D * self.hx * self.hy * math.fsum((self.lam * W_hat * W_hat).ravel().tolist())

    def omega_energy(self):
        """Pi_h(omega_h) = -1/2 D h_x h_y sum lambda W*^2 (exact discrete minimum)."""
        return -0.5 * self.D * self.hx * self.hy * math.fsum((self.lam * self.w_star**2).ravel().tolist())


def _weighted_lstsq(weights_sqrt, basis, target):
    """Batched min ||diag(w_m) (basis v_m - target_m)||: weights_sqrt (M, K), basis (K, N), target (M, K) -> v (M, N).

    Householder QR of every weighted matrix; returns the solutions and the 2-norm condition numbers of the matrices.
    """
    A = weights_sqrt[:, :, None] * basis[None, :, :]
    rhs = weights_sqrt * target
    Q, R = np.linalg.qr(A, mode="reduced")
    y = np.einsum("mkn,mk->mn", Q, rhs)
    v = np.linalg.solve(R, y[:, :, None])[:, :, 0]
    s = np.linalg.svd(R, compute_uv=False)
    cond = s[:, 0] / s[:, -1]
    return v, cond


def field_singular_values(U_hat, V_hat):
    """Singular values of W^ = U^ V^^T (equal to those of the nodal field) via thin QR of both factors."""
    _, RU = np.linalg.qr(U_hat)
    _, RV = np.linalg.qr(V_hat)
    return np.linalg.svd(RU @ RV.T, compute_uv=False)


def nodal_laplacian(W, hx, hy):
    """Delta_h W at the interior nodes by second differences with zero boundary values (2D assembly, R2 path)."""
    P = np.zeros((W.shape[0] + 2, W.shape[1] + 2))
    P[1:-1, 1:-1] = W
    return (P[:-2, 1:-1] - 2.0 * W + P[2:, 1:-1]) / hx**2 + (P[1:-1, :-2] - 2.0 * W + P[1:-1, 2:]) / hy**2


def _abs_laplacian(W, hx, hy):
    """|Delta_h| |W|: the second-difference stencil with absolute weights (1, 2, 1)/h^2 in each direction."""
    P = np.zeros((W.shape[0] + 2, W.shape[1] + 2))
    P[1:-1, 1:-1] = np.abs(W)
    return (P[:-2, 1:-1] + 2.0 * P[1:-1, 1:-1] + P[2:, 1:-1]) / hx**2 + (P[1:-1, :-2] + 2.0 * P[1:-1, 1:-1] + P[1:-1, 2:]) / hy**2


def nodal_energy(W, hx, hy, q=1.0, D=1.0):
    """Pi_h(w) = 1/2 D h_x h_y sum (Delta_h w)^2 - q h_x h_y sum w by nodal assembly."""
    lap = nodal_laplacian(W, hx, hy)
    return 0.5 * D * hx * hy * math.fsum((lap * lap).ravel().tolist()) - q * hx * hy * math.fsum(W.ravel().tolist())


@dataclass(frozen=True)
class ControlStep:
    """Half-step s of the control sequence. Representation-dependent data are marked as such."""

    s: int
    k: int
    direction: str  # computed set: "Y" (odd s) or "X" (even s)
    W: np.ndarray  # nodal field on the interior nodes (representation independent)
    U: np.ndarray  # nodal x-set of the stored representation
    V: np.ndarray  # nodal y-set of the stored representation
    energy_spectral: float
    energy_nodal: float
    sigma: np.ndarray  # singular values of W (descending)
    rho_N: float
    kappa_rep: float  # representation dependent
    given_cond: float  # sigma_1/sigma_N of the given set with unit columns (before its orthonormalisation)
    ls_cond_max: float  # largest condition number of the weighted least-squares matrices
    projection_residual: float  # nodal check of the variational equation (formula check, not a field-error estimate)
    energy_identity_defect: float | None  # |Pi(s-1) - Pi(s) - 1/2|w(s) - w(s-1)|_a^2| / |Pi(s)| (s >= 2)


def _unit_columns_cond(X):
    norms = np.linalg.norm(X, axis=0)
    s = np.linalg.svd(X / norms, compute_uv=False)
    return float(s[0] / s[-1])


def _projection_residual(grid, W, given, axis):
    """max over basis directions of |a_h(w, g_i x e) - l_h(g_i x e)| / scale, by nodal assembly only.

    A check of the variational equation, not of the accuracy of the field (sensitivity ~ n^-4, module docstring).

    axis "x": given set on G(a, n_x) (Y-half-step); "y": given set on G(b, n_y) (X-half-step).
    """
    hx, hy, D, q = grid.hx, grid.hy, grid.D, grid.q
    Z = nodal_laplacian(nodal_laplacian(W, hx, hy), hx, hy)  # Delta_h^T Delta_h W (symmetric operator)
    absW = _abs_laplacian(_abs_laplacian(np.abs(W), hx, hy), hx, hy)  # |Delta_h|^2 |W|: scale of the rounding only
    if axis == "x":
        grad = D * hx * hy * (given.T @ Z) - q * hx * hy * given.sum(axis=0)[:, None]
        scale = D * hx * hy * (np.abs(given).T @ absW) + abs(q) * hx * hy * np.abs(given).sum(axis=0)[:, None]
    else:
        grad = D * hx * hy * (Z @ given) - q * hx * hy * given.sum(axis=0)[None, :]
        scale = D * hx * hy * (absW @ np.abs(given)) + abs(q) * hx * hy * np.abs(given).sum(axis=0)[None, :]
    return float(np.max(np.abs(grad)) / np.max(scale))


def run_control(grid, N, s_max, *, start=None, orthonormalize=True):
    """Control sequence s = 1..s_max for N terms; start set: odd sines (default) or a given nodal x-set (n_x-1 by N).

    orthonormalize -- replace the given set of every half-step by an orthonormal basis of its span (field unchanged
    in exact arithmetic); with False the raw computed set is used as given set. No regularisation is applied: a
    numerically dependent given set shows as a large given_cond / ls_cond_max and is reported, not repaired.
    """
    if isinstance(N, bool) or not isinstance(N, int) or N < 1:
        raise ValueError(f"N must be a positive integer, got {N!r}")
    if grid.nx < 2 * N or grid.ny < 2 * N:
        raise ValueError("rank loss proven: n_x >= 2N and n_y >= 2N are necessary (R4_NUMERICAL_SPEC 4.2, 4.3)")
    X = odd_sine_set(grid.nx, N) if start is None else np.asarray(start, dtype=float)
    if X.shape != (grid.nx - 1, N):
        raise ValueError(f"start set must have shape {(grid.nx - 1, N)}")
    sq = np.sqrt(grid.lam)
    U_hat, V_hat = grid.psi_x @ X, None
    steps = []
    prev_energy, prev_W_hat = None, None
    for s in range(1, s_max + 1):
        if s % 2:  # Y-half-step: given x-set U_hat
            given_cond = _unit_columns_cond(U_hat)
            if orthonormalize:
                U_hat, _ = np.linalg.qr(U_hat)
            given_nodal = grid.psi_x @ U_hat
            V_hat, cond = _weighted_lstsq(sq.T, U_hat, grid.w_star.T)
        else:  # X-half-step: given y-set V_hat
            given_cond = _unit_columns_cond(V_hat)
            if orthonormalize:
                V_hat, _ = np.linalg.qr(V_hat)
            given_nodal = grid.psi_y @ V_hat
            U_hat, cond = _weighted_lstsq(sq, V_hat, grid.w_star)
        W_hat = U_hat @ V_hat.T
        W = grid.to_nodal(W_hat)
        U, V = grid.psi_x @ U_hat, grid.psi_y @ V_hat
        sigma = field_singular_values(U_hat, V_hat)
        e_spec, e_nod = grid.spectral_energy(W_hat), nodal_energy(W, grid.hx, grid.hy, grid.q, grid.D)
        identity = None
        if prev_energy is not None:
            d = grid.spectral_norm2(W_hat - prev_W_hat)
            identity = abs(prev_energy - e_spec - 0.5 * d) / abs(e_spec)
        steps.append(ControlStep(
            s=s, k=s // 2, direction="Y" if s % 2 else "X", W=W, U=U, V=V, energy_spectral=e_spec,
            energy_nodal=e_nod, sigma=sigma, rho_N=float(sigma[-1] / sigma[0]),
            kappa_rep=float(np.linalg.norm(U_hat, 2) * np.linalg.norm(V_hat, 2) / sigma[0]),
            given_cond=given_cond, ls_cond_max=float(np.max(cond)),
            projection_residual=_projection_residual(grid, W, given_nodal, "x" if s % 2 else "y"),
            energy_identity_defect=identity))
        prev_energy, prev_W_hat = e_spec, W_hat
    return steps


def first_halfstep_closed_form(grid, N):
    """s = 1 with odd sines: W1 = Psi_x P W* Psi_y, P keeping the x-modes m = 1, 3, ..., 2N - 1.

    Columns of the start set are the eigenvectors sqrt(n_x/2) psi_m of L_x, so the weighted problem of every y-mode is
    diagonal and the half-step keeps exactly the x-modes m_j of omega_h (closed form; no least squares).
    """
    if grid.nx < 2 * N or grid.ny < 2 * N:
        raise ValueError("rank loss proven: n_x >= 2N and n_y >= 2N are necessary")
    P = np.zeros_like(grid.w_star)
    rows = [2 * j - 2 for j in range(1, N + 1)]  # mode m = 2j - 1 is row m - 1
    P[rows, :] = grid.w_star[rows, :]
    return grid.to_nodal(P)
