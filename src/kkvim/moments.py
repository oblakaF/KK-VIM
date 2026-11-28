"""R6 research postprocessing: bending moments and in-plane bending stresses of a computed plate field.

Status: RESEARCH / NOT_QUALIFIED (owner's restricted R6 authorization, OD-7 of the R6 review; docs/R6_MOMENTS_IMPLEMENTATION.md).
Contract: section 8 of R6_MOMENTS_STRESSES_SPEC.review.md (Reviewer, REVIEW_PASS_R6_READY_FOR_OWNER_IMPLEMENTATION_DECISION,
SHA-256 e9b9c1e8...). Nothing here is part of the KK-VIM method: the solvers, their operators and their boundary
discretization are not used or changed. Input: arrays and scalars only (no solver is imported).

Plate theory (L-107 = Timoshenko, Woinowsky-Krieger, Plates and Shells, Nauka 1966; transcription S-14, reviewed):

    M_x  = -D (w_xx + nu w_yy)          (101), p. 97
    M_y  = -D (w_yy + nu w_xx)          (101)
    M_xy =  D (1 - nu) w_xy             (102), p. 98 - the VECTOR convention of p. 53; the symmetric tensor component
                                        is M_xy^tens = M_yx = -M_xy (review 3.1). The sign is never changed silently.

Stresses (project derivation, review 3.2): sigma_x = 12 M_x z / t^3, sigma_y = 12 M_y z / t^3, tau_xy = -12 M_xy z / t^3,
-t/2 <= z <= t/2, z downward (load side up); bottom surface z = +t/2, top surface z = -t/2 (review R-5: a project reading).
Principal moments (review 3.4, U-1): M_1,2 = (M_x + M_y)/2 +- rho, rho = sqrt(((M_x - M_y)/2)^2 + M_xy^2), M_1 >= M_2;
alpha_1 = atan2(-2 M_xy, M_x - M_y) / 2 in (-pi/2, pi/2], the angle of the normal n = (cos a, sin a) from x towards y with
M_n(alpha_1) = M_1; where rho = 0 exactly the direction is undefined: NaN (an empty cell), never a fabricated angle.

Discrete operators (review 8.2, the only variant; level B of the project):

    quantity   interior nodes 1..n-1            SS edge                                   CC edge
    f''        (f[i-1] - 2 f[i] + f[i+1])/h^2   0 (exact)                                 2 f[1]/h^2, 2 f[n-1]/h^2  (A)
    f'         (f[i+1] - f[i-1])/(2h)           (4 f[1] - f[2])/(2h), -(4 f[n-1] - f[n-2])/(2h)   0 (exact)

w_xx = f'' along axis 0, w_yy = f'' along axis 1, w_xy = f' along axis 0 then f' along axis 1. Corners need no rule.
The CC edge rule A (2 f1 / h^2) is the curvature of the energy operator of the CCCC solver (reflection f[-1] = f[1]) and
is of second order on the computed field (review 6.1-6.2; observed for CCCC N = 1, not proved for every N, grid or field).
The rule B (8 f1 - f2)/(2 h^2) is NOT a postprocessing option (OD-3); it appears only in the operator tests on exact data.

Units: lengths m, w m, D N m, moments N (N m / m), stresses Pa. Dimensionless M_bar = M / (q a^2), a = side along x.
The rounding model of 8.6 is a model, not a bound (BL-6).
"""

import math
from dataclasses import dataclass, field

import numpy as np

U = 2.0**-53
STATUS = "RESEARCH / NOT_QUALIFIED"
CONVENTIONS = ("L-107 (101), (102); M_xy by (102), vector convention p. 53 (M_xy^tensor = -M_xy); z downward; "
               "bottom z = +t/2, top z = -t/2")
CC_EDGE_RULE = "A (2 f1 / h^2), energy-consistent"
OPERATORS = {
    "f2_interior": "(f[i-1] - 2 f[i] + f[i+1]) / h^2",
    "f2_SS_edge": "0 (exact)",
    "f2_CC_edge": "2 f[1] / h^2, 2 f[n-1] / h^2 (rule A)",
    "f1_interior": "(f[i+1] - f[i-1]) / (2 h)",
    "f1_SS_edge": "(4 f[1] - f[2]) / (2 h), -(4 f[n-1] - f[n-2]) / (2 h)",
    "f1_CC_edge": "0 (exact)",
    "w_xy": "f' along x, then f' along y",
    "cc_edge_rule": CC_EDGE_RULE,
}
BC_OF_SUPPORTS = {"SSSS": "SS", "CCCC": "CC"}
MIN_INTERVALS = 4


class MomentsInputError(ValueError):
    """ERROR_INPUT of the postprocessing (the message names the reason)."""

    def __init__(self, reason):
        super().__init__(f"ERROR_INPUT: {reason}")
        self.reason = reason


def _check_bc(bc):
    if bc not in ("SS", "CC"):
        raise MomentsInputError(f"bc must be 'SS' or 'CC', got {bc!r}")


def _positive(name, value):
    if isinstance(value, bool) or not isinstance(value, (int, float, np.floating, np.integer)):
        raise MomentsInputError(f"{name} must be a number, got {value!r}")
    value = float(value)
    if not (math.isfinite(value) and value > 0.0):
        raise MomentsInputError(f"{name} must be finite and > 0, got {value!r}")
    return value


def _moved(f, axis):
    f = np.asarray(f, dtype=float)
    if f.ndim < 1 or f.shape[axis] < MIN_INTERVALS + 1:
        raise MomentsInputError(f"at least {MIN_INTERVALS} intervals are needed along axis {axis}")
    return np.moveaxis(f, axis, 0)


def second_derivative_nodes(f, h, bc, axis=0):
    """f'' at the nodes 0..n along ``axis`` (review 8.2): central inside, 0 on an SS edge, rule A on a CC edge."""
    _check_bc(bc)
    h = _positive("h", h)
    g = _moved(f, axis)
    out = np.empty_like(g)
    out[1:-1] = (g[:-2] - 2.0 * g[1:-1] + g[2:]) / (h * h)
    if bc == "SS":
        out[0] = 0.0
        out[-1] = 0.0
    else:
        out[0] = 2.0 * g[1] / (h * h)
        out[-1] = 2.0 * g[-2] / (h * h)
    return np.moveaxis(out, 0, axis)


def first_derivative_nodes(f, h, bc, axis=0):
    """f' at the nodes 0..n along ``axis`` (review 8.2): central inside, one-sided on an SS edge, 0 on a CC edge."""
    _check_bc(bc)
    h = _positive("h", h)
    g = _moved(f, axis)
    out = np.empty_like(g)
    out[1:-1] = (g[2:] - g[:-2]) / (2.0 * h)
    if bc == "SS":
        out[0] = (4.0 * g[1] - g[2]) / (2.0 * h)
        out[-1] = -(4.0 * g[-2] - g[-3]) / (2.0 * h)
    else:
        out[0] = 0.0
        out[-1] = 0.0
    return np.moveaxis(out, 0, axis)


def principal(Mx, My, Mxy):
    """M_1 >= M_2 and alpha_1 = atan2(-2 M_xy, M_x - M_y) / 2 (review 3.4, U-1); alpha_1 = NaN where rho = 0 exactly."""
    Mx, My, Mxy = (np.asarray(v, dtype=float) for v in (Mx, My, Mxy))
    mean = 0.5 * (Mx + My)
    rho = np.hypot(0.5 * (Mx - My), Mxy)
    alpha = 0.5 * np.arctan2(-2.0 * Mxy, Mx - My)
    alpha = np.where(alpha <= -0.5 * math.pi, alpha + math.pi, alpha)  # (-pi/2, pi/2]
    alpha = np.where(rho == 0.0, np.nan, alpha)
    return mean + rho, mean - rho, alpha


def moment_on_normal(Mx, My, Mxy, alpha):
    """M_n and M_nt on the normal n = (cos alpha, sin alpha) in the L-107 convention (review 3.3)."""
    c2, s2 = math.cos(2.0 * alpha), math.sin(2.0 * alpha)
    ca, sa = math.cos(alpha), math.sin(alpha)
    Mn = Mx * ca * ca + My * sa * sa - Mxy * s2
    Mnt = 0.5 * (Mx - My) * s2 + Mxy * c2
    return Mn, Mnt


@dataclass(frozen=True)
class MomentsResult:
    """Moments of one field (arrays of the shape of W, axis 0 = x) with the data of their computation."""

    Mx: np.ndarray
    My: np.ndarray
    Mxy: np.ndarray
    M1: np.ndarray
    M2: np.ndarray
    alpha1: np.ndarray
    rounding_model: float  # N; review 8.6: a model, not a bound
    a: float
    b: float
    h_x: float
    h_y: float
    D: float
    nu: float
    bc: str
    w_xx: np.ndarray = field(repr=False, default=None)
    w_yy: np.ndarray = field(repr=False, default=None)
    w_xy: np.ndarray = field(repr=False, default=None)

    def bar(self, name, q):
        """M / (q a^2) of one of Mx, My, Mxy, M1, M2; None for q = 0 (review 8.3)."""
        if q == 0:
            return None
        return getattr(self, name) / (q * self.a * self.a)


def _readonly(*arrays):
    for array in arrays:
        array.setflags(write=False)
    return arrays


def check_field(W, a, b, D, nu, bc):
    """Input checks of review 8.1 -> MomentsInputError with the reason."""
    _check_bc(bc)
    a, b, D = _positive("a", a), _positive("b", b), _positive("D", D)
    if isinstance(nu, bool) or not isinstance(nu, (int, float, np.floating)) or not math.isfinite(float(nu)):
        raise MomentsInputError(f"nu must be a finite number, got {nu!r}")
    if not abs(float(nu)) < 1.0:
        raise MomentsInputError(f"|nu| < 1 is required, got {nu!r}")
    W = np.asarray(W)
    if W.ndim != 2:
        raise MomentsInputError(f"the field must be a 2-D array (n_x + 1, n_y + 1), got shape {W.shape}")
    if W.shape[0] < MIN_INTERVALS + 1 or W.shape[1] < MIN_INTERVALS + 1:
        raise MomentsInputError(f"n_x, n_y >= {MIN_INTERVALS} are required, got {W.shape[0] - 1} x {W.shape[1] - 1}")
    if not np.issubdtype(W.dtype, np.number) or not np.all(np.isfinite(W)):
        raise MomentsInputError("the field has non-finite values")
    edges = np.concatenate([W[0], W[-1], W[:, 0], W[:, -1]])
    if np.any(edges != 0.0):
        raise MomentsInputError("the boundary nodes of the field are not 0: not an SSSS or CCCC field")
    return np.array(W, dtype=float), a, b, D, float(nu)


def plate_moments(W, a, b, D, nu, bc):
    """M_x, M_y, M_xy, M_1, M_2, alpha_1 of the nodal field W (metres; boundary nodes included; axis 0 = x).

    ``bc`` - "SS" (SSSS) or "CC" (CCCC) on all four edges. The input array is not changed (a copy is used).
    """
    W, a, b, D, nu = check_field(W, a, b, D, nu, bc)
    nx, ny = W.shape[0] - 1, W.shape[1] - 1
    hx, hy = a / nx, b / ny
    wxx = second_derivative_nodes(W, hx, bc, axis=0)
    wyy = second_derivative_nodes(W, hy, bc, axis=1)
    wxy = first_derivative_nodes(first_derivative_nodes(W, hx, bc, axis=0), hy, bc, axis=1)
    Mx = -D * (wxx + nu * wyy)
    My = -D * (wyy + nu * wxx)
    Mxy = D * (1.0 - nu) * wxy
    M1, M2, alpha1 = principal(Mx, My, Mxy)
    rounding = 4.0 * U * float(np.max(np.abs(W))) / min(hx, hy) ** 2 * D * (1.0 + abs(nu))
    _readonly(Mx, My, Mxy, M1, M2, alpha1, wxx, wyy, wxy)
    return MomentsResult(Mx, My, Mxy, M1, M2, alpha1, rounding, a, b, hx, hy, D, nu, bc, wxx, wyy, wxy)


def _check_z(t, z):
    t = _positive("t", t)
    z = float(z)
    if not (math.isfinite(z) and -0.5 * t <= z <= 0.5 * t):
        raise MomentsInputError(f"z must lie in [-t/2, t/2] = [{-0.5 * t!r}, {0.5 * t!r}], got {z!r}")
    return t, z


def stresses_at(M: MomentsResult, t, z):
    """(sigma_x, sigma_y, tau_xy) in Pa at the thickness coordinate z (m, downward; review 3.2)."""
    t, z = _check_z(t, z)
    k = 12.0 * z / t**3
    return k * M.Mx, k * M.My, -k * M.Mxy


def surface_stresses(M: MomentsResult, t):
    """{"bottom": stresses at z = +t/2, "top": stresses at z = -t/2}."""
    t = _positive("t", t)
    return {"bottom": stresses_at(M, t, 0.5 * t), "top": stresses_at(M, t, -0.5 * t)}


def moments_from_factors(X, Y, a, b, D, nu, bc):
    """QA path PM-2 (review 2.2): moments of w = sum_i X_i Y_i^T from the factors (X: (n_x+1, N), Y: (n_y+1, N))."""
    _check_bc(bc)
    X, Y = np.asarray(X, dtype=float), np.asarray(Y, dtype=float)
    X = X[:, None] if X.ndim == 1 else X
    Y = Y[:, None] if Y.ndim == 1 else Y
    if X.ndim != 2 or Y.ndim != 2 or X.shape[1] != Y.shape[1]:
        raise MomentsInputError(f"factors must be (n_x+1, N) and (n_y+1, N), got {X.shape} and {Y.shape}")
    a, b, D = _positive("a", a), _positive("b", b), _positive("D", D)
    hx, hy = a / (X.shape[0] - 1), b / (Y.shape[0] - 1)
    X2 = second_derivative_nodes(X, hx, bc, axis=0)
    Y2 = second_derivative_nodes(Y, hy, bc, axis=0)
    X1 = first_derivative_nodes(X, hx, bc, axis=0)
    Y1 = first_derivative_nodes(Y, hy, bc, axis=0)
    wxx, wyy, wxy = X2 @ Y.T, X @ Y2.T, X1 @ Y1.T
    return -D * (wxx + nu * wyy), -D * (wyy + nu * wxx), D * (1.0 - nu) * wxy


# ------------------------------------------------------------------------------------------------ summaries

QUANTITIES = ("Mx", "My", "Mxy", "M1", "M2")
STRESS_QUANTITIES = ("sigma_x", "sigma_y", "tau_xy")


def centre_index(shape):
    """(i, j) of the plate centre if it is a node (even n_x and n_y), else None (nothing is interpolated)."""
    nx, ny = shape[0] - 1, shape[1] - 1
    return (nx // 2, ny // 2) if nx % 2 == 0 and ny % 2 == 0 else None


def sampled_extrema(values, x, y):
    """The largest and the smallest value among the nodes, with their nodes and coordinates (first in i, j order)."""
    values = np.asarray(values, dtype=float)
    finite = np.where(np.isfinite(values), values, np.nan)
    out = {}
    for name, k in (("max", int(np.nanargmax(finite))), ("min", int(np.nanargmin(finite)))):
        i, j = np.unravel_index(k, values.shape)
        out[name] = {"value": float(values[i, j]), "i": int(i), "j": int(j), "x": float(x[i]), "y": float(y[j])}
    return out


def field_summary(values, x, y):
    """Centre value (if the centre is a node) and the sampled extrema of one field."""
    c = centre_index(np.shape(values))
    centre = None if c is None else {"value": float(values[c]), "i": c[0], "j": c[1], "x": float(x[c[0]]),
                                     "y": float(y[c[1]])}
    return {"centre": centre, **sampled_extrema(values, x, y)}
