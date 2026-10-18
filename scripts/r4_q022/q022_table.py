"""Q-022 (SCRATCH): largest s for which the rigorous transfer bound E_s meets the 1/100 rule (err_abs, e_rel, centre)."""
import sys, numpy as np
PI = np.pi
M, S = int(sys.argv[1]), int(sys.argv[2])
R = [1/5, 1/3, 1/2, 2/3, 1.0, 1.4, 1.5, 2.0, 3.0, 5.0]
print("M=%d, criterion: budget <= 1/100 of measured quantity; s_max = last s with all s'<=s satisfied" % M)
for r in R:
    a, b = 1.0, r
    m = np.arange(1, M + 1, 2, dtype=float); K = len(m)
    k2 = (PI * m / a) ** 2; n2 = (PI * m / b) ** 2
    lam = (k2[:, None] + n2[None, :]) ** 2
    om = 16.0 / (PI ** 2 * m[:, None] * m[None, :] * lam)
    en = lambda C: np.sqrt(a * b / 4 * np.sum(lam * C * C))
    T = np.sqrt(64 * a * b / PI ** 8 * (a ** 4 + b ** 4) * PI ** 2 / (80 * M ** 5))
    lam11 = PI ** 2 * (1 / a ** 2 + 1 / b ** 2)
    sgn = np.outer(np.sin(m * PI / 2), np.sin(m * PI / 2)); w0c = np.sum(om * sgn)
    tail = (a * b) ** 2 / (4 * PI ** 4) * 2 * (1 / (2 * M)) * (PI ** 2 / 8) / (a * b / 4)
    Gc = np.sqrt(np.sum(1 / (a * b / 4 * lam)) + tail)
    res = []
    for N in (1, 2, 3):
        X = np.zeros((K, N)); X[np.arange(N), np.arange(N)] = 1.0
        given, side, prev, E, sigprev = X, 'x', None, None, None
        smax = {'err': 0, 'erel': 0, 'cen': 0}; alive = dict.fromkeys(smax, True); void_at = None; gs = []
        for s in range(1, S + 1):
            Q, _ = np.linalg.qr(given)
            if side == 'x':
                beta, gam, f = Q.T @ (k2[:, None] * Q), Q.T @ ((k2 ** 2)[:, None] * Q), Q.T @ (1.0 / m)
                G = n2[:, None, None] ** 2 * np.eye(N) + 2 * n2[:, None, None] * beta + gam
                B = np.linalg.solve(G, np.broadcast_to(f, (K, N))[..., None])[..., 0] * (16 / PI ** 2 / m)[:, None]
                C = Q @ B.T; given, side = B, 'y'; nu2 = (PI / b) ** 2
            else:
                beta, gam, f = Q.T @ (n2[:, None] * Q), Q.T @ ((n2 ** 2)[:, None] * Q), Q.T @ (1.0 / m)
                G = k2[:, None, None] ** 2 * np.eye(N) + 2 * k2[:, None, None] * beta + gam
                A = np.linalg.solve(G, np.broadcast_to(f, (K, N))[..., None])[..., 0] * (16 / PI ** 2 / m)[:, None]
                C = A @ Q.T; given, side = A, 'x'; nu2 = (PI / a) ** 2
            err = en(om - C); Rr = np.sqrt(T ** 2 + err ** 2)
            if s == 1:
                E = T
            elif np.isfinite(E) and E / lam11 < sigprev:
                sn = min(1.0, E / (nu2 * sigprev)); d = en(C - prev)
                gs.append((Rr + d) / (nu2 * sigprev))
                E = np.sqrt((T + sn * Rr) ** 2 + (E + sn * d) ** 2)
            else:
                E = np.inf; void_at = void_at or s
            sv = np.linalg.svd(given, compute_uv=False); sigprev = np.sqrt(a * b) / 2 * sv[N - 1]
            q = E / err
            ok = {'err': q <= 1e-2, 'erel': 2 * q + q * q <= 1e-2, 'cen': Gc * E <= 1e-2 * abs(np.sum(C * sgn) - w0c)}
            for k in smax:
                if alive[k] and ok[k]: smax[k] = s
                else: alive[k] = False
            prev = C
        gmed = np.median(gs[-5:]) if gs else 0.0
        res.append("N=%d: s_max err_abs=%2d e_rel=%2d centre=%2d (void from %s; g_s tail~%.2g)" % (N, smax['err'], smax['erel'], smax['cen'], void_at, gmed))
    print("r=%-5.3g " % r + " | ".join(res))
