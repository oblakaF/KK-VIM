"""Q-022 (SCRATCH): evaluate the rigorous transfer recurrence at production truncation (streaming, low memory)."""
import sys, numpy as np
PI = np.pi
M, S = int(sys.argv[1]), int(sys.argv[2])
rs = [float(x) for x in sys.argv[3].split(',')]
for r in rs:
    a, b = 1.0, r
    m = np.arange(1, M + 1, 2, dtype=float); K = len(m)
    k2 = (PI * m / a) ** 2; n2 = (PI * m / b) ** 2
    lam = (k2[:, None] + n2[None, :]) ** 2
    om = 16.0 / (PI ** 2 * m[:, None] * m[None, :] * lam)
    en = lambda C: np.sqrt(a * b / 4 * np.sum(lam * C * C))
    norm0 = en(om)                                                   # ||omega_M||_a <= ||omega_0||_a
    T = np.sqrt(64 * a * b / PI ** 8 * (a ** 4 + b ** 4) * PI ** 2 / (80 * M ** 5))
    lam11 = PI ** 2 * (1 / a ** 2 + 1 / b ** 2)
    sgn = np.outer(np.sin(m * PI / 2), np.sin(m * PI / 2)); w0c = np.sum(om * sgn)
    # Riesz constant of centre evaluation: sum over box of 1/((ab/4) lam) + strict tail (1/lam <= a^2 b^2/(4 pi^4 m^2 n^2))
    tail = (a * b) ** 2 / (4 * PI ** 4) * 2 * (1 / (2 * M)) * (PI ** 2 / 8) / (a * b / 4)
    Gc = np.sqrt(np.sum(1 / (a * b / 4 * lam)) + tail)
    for N in (1, 2, 3):
        X = np.zeros((K, N)); X[np.arange(N), np.arange(N)] = 1.0
        given, side, prev, E, Eprev, sigprev = X, 'x', None, None, None, None
        out = []; void_at = None
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
            err = en(om - C); R = np.sqrt(T ** 2 + err ** 2)
            if s == 1:
                E = T; g = 0.0; sn = 0.0
            else:
                if E is not None and np.isfinite(E) and E / lam11 < sigprev:
                    sn = min(1.0, E / (nu2 * sigprev)); d = en(C - prev)
                    E = np.sqrt((T + sn * R) ** 2 + (E + sn * d) ** 2)
                    g = (R + d) / (nu2 * sigprev)
                else:
                    E = np.inf; void_at = void_at or s
            sv = np.linalg.svd(given, compute_uv=False); sigprev = np.sqrt(a * b) / 2 * sv[N - 1]
            wc = np.sum(C * sgn)
            out.append((s, E / norm0, E / err, Gc * E / w0c, abs(wc - w0c) / w0c, g, sigprev / (np.sqrt(a * b) / 2 * sv[0])))
            prev = C
        print("r=%g N=%d M=%d: T/|w|a=%.2e  void from s=%s" % (r, N, M, T / norm0, void_at))
        for (s, Er, Ee, Ec, ce, g, rho) in out:
            if s in (1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 30, 40):
                print("  s=%2d  E/|w0|a=%.2e  E/err_abs=%.2e  centre: bound/w0c=%.2e err/w0c=%.2e  g_s=%.2f  rho_N(w_s)=%.2e" % (s, Er, Ee, Ec, ce, g, rho))
