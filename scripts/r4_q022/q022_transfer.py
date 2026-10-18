"""Q-022 study (SCRATCH, not part of the project): continuous N-term KK-MVI (matrix Parseval recursion, sine basis,
odd modes <= M) and a rigorous a-posteriori recurrence for the truncation-error transfer.  Units a = 1, b = r, q = D = 1.
Bound (derived in the document):  E_1 <= T,  E_s <= sqrt((T + sin_s*R_s)^2 + (E_{s-1} + sin_s*|d'_s|)^2),
sin_s <= min(1, E_{s-1}/(nu1^2 * sigma_N^{L2}(w'_{s-1}))), valid while E_{s-1}/lambda11 < sigma_N^{L2}(w'_{s-1})."""
import sys, numpy as np
PI = np.pi

def setup(M, r):
    a, b = 1.0, r
    m = np.arange(1, M + 1, 2, dtype=float)
    k2 = (PI * m / a) ** 2; n2 = (PI * m / b) ** 2            # same index sets for x and y
    lam = (k2[:, None] + n2[None, :]) ** 2
    om = 16.0 / (PI ** 2 * m[:, None] * m[None, :] * lam)      # Navier coefficients (B-27)
    return dict(a=a, b=b, m=m, k2=k2, n2=n2, lam=lam, om=om)

def enorm(P, C):  # ||w||_a for coefficient matrix C (rows x-modes, cols y-modes)
    return np.sqrt(P['a'] * P['b'] / 4 * np.sum(P['lam'] * C * C))

def run(P, N, S):
    m, k2, n2 = P['m'], P['k2'], P['n2']
    X = np.zeros((len(m), N)); X[np.arange(N), np.arange(N)] = 1.0        # odd sines 1,3,..,2N-1
    fields, sig = [], []
    given, side = X, 'x'
    for s in range(1, S + 1):
        Q, _ = np.linalg.qr(given)
        if side == 'x':     # Y-half-step: given x-set, per y-mode n solve (nu^4 I + 2 nu^2 beta + gamma) b = (16/(pi^2 n)) f
            beta, gam, f = Q.T @ (k2[:, None] * Q), Q.T @ ((k2 ** 2)[:, None] * Q), Q.T @ (1.0 / m)
            G = n2[:, None, None] ** 2 * np.eye(N) + 2 * n2[:, None, None] * beta + gam
            B = np.linalg.solve(G, np.broadcast_to(f, (len(m), N))[..., None])[..., 0] * (16 / PI ** 2 / m)[:, None]
            C = Q @ B.T; given, side = B, 'y'
        else:               # X-half-step: given y-set, per x-mode
            beta, gam, f = Q.T @ (n2[:, None] * Q), Q.T @ ((n2 ** 2)[:, None] * Q), Q.T @ (1.0 / m)
            G = k2[:, None, None] ** 2 * np.eye(N) + 2 * k2[:, None, None] * beta + gam
            A = np.linalg.solve(G, np.broadcast_to(f, (len(m), N))[..., None])[..., 0] * (16 / PI ** 2 / m)[:, None]
            C = A @ Q.T; given, side = A, 'x'
        fields.append(C)
        sv = np.linalg.svd(given, compute_uv=False)                 # sigma(C) = sigma(B) since Q orthonormal
        sig.append(np.sqrt(P['a'] * P['b']) / 2 * sv[N - 1])       # sigma_N in the L2 kernel sense
    return fields, sig

def bound(P, fields, sig, T, omega, N):
    """omega: target coefficient matrix on the same box (omega_M); T: bound of ||tau||_a (tau = omega_true - omega_M)."""
    a, b = P['a'], P['b']; lam11 = PI ** 2 * (1 / a ** 2 + 1 / b ** 2)
    E = [T]; info = []
    for s in range(2, len(fields) + 1):
        nu2 = (PI / b) ** 2 if s % 2 == 1 else (PI / a) ** 2       # Y-step (odd s): nu_1^2 = (pi/b)^2; X-step: kappa_1^2
        Ep, sN = E[-1], sig[s - 2]
        valid = Ep / lam11 < sN
        sn = min(1.0, Ep / (nu2 * sN))
        R = np.sqrt(T ** 2 + enorm(P, omega - fields[s - 1]) ** 2)
        d = enorm(P, fields[s - 1] - fields[s - 2])
        Es = np.sqrt((T + sn * R) ** 2 + (Ep + sn * d) ** 2) if valid else np.inf
        info.append((s, sn, R, d, sN)); E.append(Es)
    return E, info

if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'validate':   # "true" target = omega_{M2} (finite), truncation to the M1 box; exact E_s computable
        for r in (1.0, 0.2, 5.0):
            for N in (1, 2, 3):
                M1, M2, S = 101, 1601, 12
                P2 = setup(M2, r); K1 = (M1 + 1) // 2
                f2, _ = run(P2, N, S)
                P1 = setup(M1, r); f1, s1 = run(P1, N, S)
                om2 = P2['om']; tau = om2.copy(); tau[:K1, :K1] = 0
                T = enorm(P2, tau)
                E, info = bound(P1, f1, s1, T, P1['om'], N)
                worst = 0
                for s in range(1, S + 1):
                    Z = np.zeros_like(om2); Z[:K1, :K1] = f1[s - 1]
                    obs = enorm(P2, f2[s - 1] - Z)
                    worst = max(worst, obs / E[s - 1])
                print("validate r=%.1f N=%d: T=%.2e  max(observed/bound) over s=1..%d = %.3f  bound/obs at s=%d: %.1f" %
                      (r, N, T, S, worst, S, E[S - 1] / obs))
    elif mode == 'production':
        M, S = int(sys.argv[2]), int(sys.argv[3])
        for r in (1.0, 0.2, 5.0):
            P = setup(M, r); a, b = P['a'], P['b']
            Ssum = np.sum(1 / (P['m'][:, None] ** 2 * P['m'][None, :] ** 2 * (P['m'][:, None] ** 2 / a ** 2 + P['m'][None, :] ** 2 / b ** 2) ** 2))
            norm0 = np.sqrt(64 * a * b / PI ** 8 * Ssum)                     # ||omega_M||_a (lower bound of ||omega_0||_a)
            T = np.sqrt(64 * a * b / PI ** 8 * (a ** 4 + b ** 4) * PI ** 2 / (80 * M ** 5))   # strict tail bound (QA, verified 3aaca8e)
            # Riesz constant of the centre evaluation: |v(c)| <= Gc ||v||_a ; series over the box + strict tail bound
            lamv = P['lam']; Gc2 = np.sum(1 / (a * b / 4 * lamv)) + (a * b) ** 2 / (4 * PI ** 4) * (1 / (2 * M)) * (PI ** 2 / 8) * 2 * 4 / (a * b)
            Gc = np.sqrt(Gc2)
            sgn = np.outer(np.sin(P['m'] * PI / 2), np.sin(P['m'] * PI / 2)); w0c = np.sum(P['om'] * sgn)
            for N in (1, 2, 3):
                f, sg = run(P, N, S)
                E, info = bound(P, f, sg, T, P['om'], N)
                err = [enorm(P, P['om'] - C) for C in f]
                wc = [np.sum(C * sgn) for C in f]
                rows = []
                for s in (1, 2, 3, 4, 6, 8, 10, 20, 40):
                    if s > S: continue
                    Es = E[s - 1]
                    rows.append("s=%2d E/|w0|a=%.1e E/err=%.1e centre-bound/|w0(c)|=%.1e centre-err=%.1e g=%s" % (
                        s, Es / norm0, Es / err[s - 1], Gc * Es / w0c, abs(wc[s - 1] - w0c) / w0c,
                        ("%.2f" % ((info[s - 2][2] + info[s - 2][3]) / ((PI / (b if s % 2 else a)) ** 2 * info[s - 2][4]))) if s >= 2 else "-"))
                print("r=%.1f N=%d M=%d  T/|w0|a=%.2e  rho-ish sigma_N(w1)/|w0|_L2" % (r, N, M, T / norm0))
                for line in rows: print("   " + line)
