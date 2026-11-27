"""R6 moment references (role Reference; review R6-MOMENTS-STRESSES-COMPLETE-SCIENTIFIC-REVIEW-01, sections 4, 5, 7, 8.8).

No import of ``kkvim`` (D-028). The existing modules are reused unchanged: ``navier`` (coefficients, node sines, the
D-038 ratio gate), ``levy`` (the Levy frame and the stable bracket T), ``r5_cccc_ritz`` (the Ritz solution), and
``r5_cccc_d2`` (the D2 field). Status: every output is a RESEARCH reference of R6; the qualification of the SSSS moment
reference (OD-4) and of any CCCC moment reference is NOT decided here (Q-027 moment part and Q-030 stay open until the
owner decides; CCCC moments: NOT_QUALIFIED, BL-3).

SSSS (plate a x b, simply supported, q = const; L-107 (101), (102), the series (131) differentiated term by term):

    Navier, r = b/a, in q a^2:
      M_x  = 16/pi^4 sum_{m,n odd} (m^2 + nu n^2/r^2) / (m n (m^2 + n^2/r^2)^2) sin(m pi x/a) sin(n pi y/b)
      M_y  = 16/pi^4 sum (nu m^2 + n^2/r^2) / (m n (m^2 + n^2/r^2)^2) sin sin
      M_xy = (1 - nu) 16/(pi^4 r) sum cos(m pi x/a) cos(n pi y/b) / (m^2 + n^2/r^2)^2              (review 5.2)
    square truncation m, n <= M (M odd). Strict tail bounds with exact terms (review 5.1, 5.2; proposal 4.1, 4.2):
      N1 = 16/pi^4 [r^2/2 I(r) + 1/2 I(1/r)],  I(c) = (5 + 2 ln(c M)) / (8 c^2 M^2) + 1/(16 c^4 M^4)   (|nu| <= 1)
      N2 = |1 - nu| 16/(pi^4 r) [pi (r + r^3)/(32 M^2) + (1 + r^4)/(6 M^3)]
    Levy (proposal 3.2, review 4.2-4.3; frame of ``levy``: a_L = min(a, b) along x_L, eta from the centre line), in q a_L^2:
      M_x^L  = 4/pi^3 sum (T + nu K_uu)/m^3 sin(m pi x_L/a_L),  M_y^L = 4/pi^3 sum (nu T + K_uu)/m^3 sin,
      M_xy^L = -(1 - nu) 4/pi^3 sum K_u/m^3 cos(m pi x_L/a_L),  K_u odd in u = m pi eta / a_L,
    with the stable forms of K_uu and K_u (all exponents <= 0); z and p are formed directly from the coordinates.
    Back to the original axes (review 2.5): swapped (b < a): M_x(x, y) = M_y^L(y, x), M_y = M_x^L, M_xy = M_xy^L.
    Bounds (review 5.3, 5.4): L2 uniform (1 + C_K|nu|, |nu| + C_K, |1 - nu|) / (pi^3 M^2); L3 at interior points.
    Rounding: models only (BL-6), never added to a bound silently.

CCCC (review 7; OD-5): no qualified moment reference exists.
    Ritz (P7 basis phi_m = 1 - cos 2 m pi xi): moments by differentiating the basis term by term -
      an independent CONTINUOUS approximation, slow (~1/M) at the edges: OBSERVATION only.
    D2 (the same discrete problem as KK-VIM, D-049.1): the moments of rule A (review 8.2) applied to the D2 field by
      an implementation of this module (not ``kkvim.moments``) - a same-discrete-equation CONSISTENCY check, not accuracy.
"""

import math

import numpy as np

from kkvim_reference import levy, navier

U = 2.0**-53
STATUS_SSSS = "RESEARCH reference (R6); qualification of the moment reference by owner decision OD-4 (not taken here)"
STATUS_CCCC = "NOT_QUALIFIED (no moment reference with a strict bound, BL-3); OBSERVATION / CONSISTENCY only"
C_K = 1.4484 / math.e  # review 5.3: L <= 1.4484 (an upper bound of 1.447743)
L_SUP = 1.4484
NAVIER_ROUNDINGS = 24  # rounding model: operations per term of the separable sum (model, not a bound)


# ------------------------------------------------------------------------------------------------ Navier


def _odd(M):
    return np.arange(1, M + 1, 2, dtype=np.int64)


def _check_level(M):
    if isinstance(M, bool) or not isinstance(M, (int, np.integer)) or M < 1 or M % 2 == 0:
        raise ValueError(f"ERROR_INPUT: M must be an odd integer >= 1, got {M!r}")
    return int(M)


def navier_coefficients(M, nu, r):
    """Coefficient matrices (cx, cy, cxy) in q a^2 for odd m (rows), n (columns); r = b/a."""
    k = _odd(M).astype(float)
    m, n = k[:, None], k[None, :]
    m2, n2 = m * m, n * n / (r * r)
    den = (m2 + n2) ** 2
    cx = (16.0 / math.pi**4) * (m2 + nu * n2) / (m * n * den)
    cy = (16.0 / math.pi**4) * (nu * m2 + n2) / (m * n * den)
    cxy = (1.0 - nu) * (16.0 / (math.pi**4 * r)) / den
    return cx, cy, cxy


def _node_cosines(n, M):
    """C[i, k] = cos(m_k pi i / n), i = 0..n, by exact integer reduction of m i modulo 2n."""
    r = np.arange(2 * n, dtype=np.int64)
    table = np.cos(math.pi * r / n)
    table[r == 0] = 1.0
    table[r == n] = -1.0
    if n % 2 == 0:
        table[(r == n // 2) | (r == 3 * n // 2)] = 0.0
    i = np.arange(n + 1, dtype=np.int64)
    return table[np.outer(i, _odd(M)) % (2 * n)]


def _point_trig(theta, M):
    """sin and cos of m pi theta for points theta in [0, 1]."""
    t = np.outer(np.asarray(theta, dtype=float), _odd(M).astype(float)) * math.pi
    return np.sin(t), np.cos(t)


def navier_tail_bounds(M, r, nu):
    """(N1, N2) in q a^2: strict bounds of the square-truncation tails of M_x, M_y (|nu| <= 1) and of M_xy (review 5.1, 5.2)."""
    M = _check_level(M)
    if abs(nu) > 1:
        raise ValueError("ERROR_INPUT: N1 holds for |nu| <= 1")

    def integral(c):  # I(c) of review 5.1
        return (5.0 + 2.0 * math.log(c * M)) / (8.0 * c * c * M * M) + 1.0 / (16.0 * c**4 * M**4)

    n1 = 16.0 / math.pi**4 * (0.5 * r * r * integral(r) + 0.5 * integral(1.0 / r))
    n2 = abs(1.0 - nu) * 16.0 / (math.pi**4 * r) * (math.pi * (r + r**3) / (32.0 * M * M) + (1.0 + r**4) / (6.0 * M**3))
    return n1, n2


def navier_moments_nodes(a, b, q, nu, n_x, n_y, M=navier.M_R3):
    """Navier M_x, M_y, M_xy (N) at the nodes x_i = i a/n_x, y_j = j b/n_y; dict with the bounds and the models."""
    M = _check_level(M)
    r = navier._ratio(a, b)
    cx, cy, cxy = navier_coefficients(M, nu, r)
    Sx, Sy = navier._node_sines(n_x, M), navier._node_sines(n_y, M)
    Cx, Cy = _node_cosines(n_x, M), _node_cosines(n_y, M)
    scale = q * a * a
    out = {"Mx": scale * ((Sx @ cx) @ Sy.T), "My": scale * ((Sx @ cy) @ Sy.T), "Mxy": scale * ((Cx @ cxy) @ Cy.T)}
    return _navier_record(out, M, r, nu, a, q, cx, cy, cxy)


def navier_moments_points(a, b, q, nu, x, y, M=navier.M_R3):
    """Navier moments at points (x_p, y_p) (paired arrays)."""
    M = _check_level(M)
    r = navier._ratio(a, b)
    cx, cy, cxy = navier_coefficients(M, nu, r)
    x, y = np.atleast_1d(np.asarray(x, float)), np.atleast_1d(np.asarray(y, float))
    sx, cxs = _point_trig(x / a, M)
    sy, cys = _point_trig(y / b, M)
    scale = q * a * a
    out = {"Mx": scale * np.sum((sx @ cx) * sy, axis=1), "My": scale * np.sum((sx @ cy) * sy, axis=1),
           "Mxy": scale * np.sum((cxs @ cxy) * cys, axis=1)}
    return _navier_record(out, M, r, nu, a, q, cx, cy, cxy)


def _navier_record(out, M, r, nu, a, q, cx, cy, cxy):
    n1, n2 = navier_tail_bounds(M, r, nu)
    scale = abs(q) * a * a
    out.update(method="Navier double series (131) differentiated by (101), (102); square truncation m, n <= M",
               M=M, r=r, nu=nu, scale_qa2=q * a * a,
               tail_bound={"Mx": n1 * scale, "My": n1 * scale, "Mxy": n2 * scale},
               rounding_model={"Mx": NAVIER_ROUNDINGS * U * float(np.abs(cx).sum()) * scale,
                               "My": NAVIER_ROUNDINGS * U * float(np.abs(cy).sum()) * scale,
                               "Mxy": NAVIER_ROUNDINGS * U * float(np.abs(cxy).sum()) * scale},
               status=STATUS_SSSS)
    return out


def delta_m(r, nu=0.3, M=None):
    """Review 5.5: delta_M(r) = 1e-4 min(|M_x(centre)|, |M_y(centre)|) in q a^2 (the normalization proposed in OD-4)."""
    M = navier.M_R3 if M is None else M
    c = navier_moments_points(1.0, float(r), 1.0, nu, [0.5], [0.5 * r], M=M)
    return 1e-4 * min(abs(float(c["Mx"][0])), abs(float(c["My"][0])))


# ------------------------------------------------------------------------------------------------ Levy


def _k_uu(alpha, z, p):
    """K_uu (even in u) in the stable form (proposal 3.2): both parts >= 0, exactly 0 at the edge z = 0."""
    with np.errstate(under="ignore"):
        g = np.exp(-2.0 * alpha)
        E = np.exp(-2.0 * alpha + z)
        return (-z * np.exp(-z) * np.expm1(-2.0 * p) - 2.0 * alpha * E * np.expm1(-2.0 * z) / (1.0 + g)) / (2.0 * (1.0 + g))


def _k_u(alpha, z, p):
    """K_u for u >= 0 (proposal 3.2, two stable branches); exactly 0 on the centre line p = 0."""
    with np.errstate(under="ignore", over="ignore", invalid="ignore"):
        g = np.exp(-2.0 * alpha)
        th = -np.expm1(-2.0 * alpha) / (1.0 + g)
        E = np.exp(-2.0 * alpha + z)
        near = np.exp(p - alpha) * (-(alpha * th + 1.0) * np.expm1(-2.0 * p) - p * (1.0 + np.exp(-2.0 * p)))
        far = (1.0 + z) * np.exp(-z) - (1.0 - z) * E - 2.0 * alpha * E * (1.0 + np.exp(-2.0 * z)) / (1.0 + g)
        return np.where(p <= 0.5 * alpha, near, far) / (2.0 * (1.0 + g))


def levy_k_direct(alpha, u):
    """K, K_u, K_uu by the direct formulas (for checks at moderate alpha only)."""
    ch = math.cosh(alpha)
    th = math.tanh(alpha)
    K = ((alpha * th + 2.0) * math.cosh(u) - u * math.sinh(u)) / (2.0 * ch)
    Ku = ((alpha * th + 1.0) * math.sinh(u) - u * math.cosh(u)) / (2.0 * ch)
    Kuu = (alpha * th * math.cosh(u) - u * math.sinh(u)) / (2.0 * ch)
    return K, Ku, Kuu


def _levy_terms(xs, ys, a, b, M):
    """Levy frame data for the original points (xs, ys) (paired arrays)."""
    orient = levy.levy_orientation(a, b)
    xl, yl = (ys, xs) if orient.swapped else (xs, ys)
    a_L, b_L = orient.a_L, orient.b_L
    m = _odd(M).astype(float)[:, None]
    alpha = m * (math.pi * orient.r_L / 2.0)
    d = np.minimum(yl, b_L - yl)[None, :]  # distance to the nearest edge
    eta = (yl - 0.5 * b_L)[None, :]  # from the centre line, signed
    z = m * math.pi * d / a_L
    p = m * math.pi * np.abs(eta) / a_L
    s = 2.0 * d / b_L
    return orient, xl, m, alpha, z, p, s, np.sign(eta)


def levy_moments_points(a, b, q, nu, x, y, M=6117):
    """Levy M_x, M_y, M_xy (N) at the points (x_p, y_p) of the plate a x b (original axes; paired arrays)."""
    M = _check_level(M)
    x, y = np.broadcast_arrays(np.atleast_1d(np.asarray(x, float)), np.atleast_1d(np.asarray(y, float)))
    x, y = x.ravel(), y.ravel()
    orient = levy.levy_orientation(a, b)
    c = 4.0 / math.pi**3 * q * orient.a_L**2
    MxL, MyL, MxyL, S = (np.empty(x.size) for _ in range(4))
    chunk = max(1, (1 << 20) // max(1, (M + 1) // 2))
    for lo in range(0, x.size, chunk):
        sl = slice(lo, lo + chunk)
        _, xl, m, alpha, z, p, s, sign = _levy_terms(x[sl], y[sl], a, b, M)
        T = levy._bracket(alpha, s)
        Kuu = _k_uu(alpha, z, p)
        Ku = sign * _k_u(alpha, z, p)
        inv3 = 1.0 / m**3
        ang = m * math.pi * xl[None, :] / orient.a_L
        sin, cos = np.sin(ang), np.cos(ang)
        MxL[sl] = c * np.sum(inv3 * (T + nu * Kuu) * sin, axis=0)
        MyL[sl] = c * np.sum(inv3 * (nu * T + Kuu) * sin, axis=0)
        MxyL[sl] = -(1.0 - nu) * c * np.sum(inv3 * Ku * cos, axis=0)
        S[sl] = s[0]
    Mx, My = (MyL, MxL) if orient.swapped else (MxL, MyL)
    bounds = levy_uniform_bounds(M, nu)
    scale = abs(q) * orient.a_L**2
    return {"Mx": Mx, "My": My, "Mxy": MxyL, "M": M, "nu": nu, "a_L": orient.a_L, "swapped": orient.swapped,
            "method": "Levy series (139) differentiated by (101), (102) in the stable forms of the R6 proposal 3.2",
            "tail_bound": {k: v * scale for k, v in bounds.items()},
            "interior_bound": levy_interior_bounds(M, nu, orient.r_L, S, scale),
            "status": STATUS_SSSS}


def levy_uniform_bounds(M, nu):
    """L2 (review 5.3): uniform tail bounds over the closed plate, in q a_L^2."""
    M = _check_level(M)
    den = math.pi**3 * M * M
    return {"Mx": (1.0 + C_K * abs(nu)) / den, "My": (abs(nu) + C_K) / den, "Mxy": abs(1.0 - nu) / den}


def levy_interior_bounds(M, nu, r_L, s, scale=1.0):
    """L3 (review 5.4): tail bounds at points with s = 2d/b_L > 0 (None at the edges s = 0)."""
    M = _check_level(M)
    s = np.atleast_1d(np.asarray(s, dtype=float))
    out = {"Mx": [], "My": [], "Mxy": []}
    m = M + 2
    alpha = m * math.pi * r_L / 2.0
    for sv in s:
        if sv <= 0.0:
            for key in out:
                out[key].append(None)
            continue
        e = math.exp(-alpha * sv)
        denom = 1.0 - math.exp(-math.pi * r_L * sv)
        base = 4.0 / math.pi**3 / m**3 * e / denom
        out["Mx"].append(scale * base * ((alpha + 2.0) + abs(nu) * L_SUP * alpha * sv))
        out["My"].append(scale * base * (abs(nu) * (alpha + 2.0) + L_SUP * alpha * sv))
        out["Mxy"].append(scale * base * abs(1.0 - nu) * (1.0 + alpha * sv))
    return out


# ------------------------------------------------------------------------------------------------ CCCC: Ritz and D2


def ritz_moments(solution, x, y, nu):
    """Ritz M_x, M_y, M_xy (N) on the tensor grid x (len I) by y (len J): the basis differentiated term by term."""
    p = solution.problem
    c = solution.coefficients()  # M x N
    x, y = np.asarray(x, float), np.asarray(y, float)
    km, kn = np.arange(1, p.M + 1, dtype=float), np.arange(1, p.N + 1, dtype=float)
    ax, ay = 2.0 * math.pi * km / p.a, 2.0 * math.pi * kn / p.b
    px, py = np.outer(x, ax), np.outer(y, ay)
    phi_x, phi_y = 1.0 - np.cos(px), 1.0 - np.cos(py)
    d1x, d1y = np.sin(px) * ax, np.sin(py) * ay
    d2x, d2y = np.cos(px) * ax**2, np.cos(py) * ay**2
    wxx = d2x @ c @ phi_y.T
    wyy = phi_x @ c @ d2y.T
    wxy = d1x @ c @ d1y.T
    D = p.D
    return {"Mx": -D * (wxx + nu * wyy), "My": -D * (wyy + nu * wxx), "Mxy": D * (1.0 - nu) * wxy,
            "M": p.M, "N": p.N, "method": "Ritz (P7 double cosine basis), basis differentiated term by term",
            "status": STATUS_CCCC}


def d2_moments(field, a, b, D, nu):
    """Rule-A moments (review 8.2, CC edges) of the D2 field, by this module's own operators (consistency check)."""
    W = np.asarray(field, dtype=float)
    nx, ny = W.shape[0] - 1, W.shape[1] - 1
    hx, hy = a / nx, b / ny

    def d2(f, h):  # along axis 0
        out = np.empty_like(f)
        out[1:-1] = (f[:-2] - 2.0 * f[1:-1] + f[2:]) / (h * h)
        out[0] = 2.0 * f[1] / (h * h)
        out[-1] = 2.0 * f[-2] / (h * h)
        return out

    def d1(f, h):
        out = np.empty_like(f)
        out[1:-1] = (f[2:] - f[:-2]) / (2.0 * h)
        out[0] = 0.0
        out[-1] = 0.0
        return out

    wxx = d2(W, hx)
    wyy = d2(W.T, hy).T
    wxy = d1(d1(W, hx).T, hy).T
    return {"Mx": -D * (wxx + nu * wyy), "My": -D * (wyy + nu * wxx), "Mxy": D * (1.0 - nu) * wxy,
            "method": "D2 field (13-point scheme, the same discrete problem as KK-VIM) with the rule-A operators",
            "status": STATUS_CCCC}
