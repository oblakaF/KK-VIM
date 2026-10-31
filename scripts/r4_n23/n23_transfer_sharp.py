"""R4, N = 2-3: Q-022 transfer bound with the sharpened angle estimate (Lemma N23-6) and the monotone upper bound (Corollary N23-7).

Research script of docs/R4_N23_SCIENTIFIC_CLOSURE.md (Literature). Not part of the solver and not a reference of the project:
kkvim and kkvim_reference are NOT imported. The truncated matrix Parseval recursion is the one of scripts/r4_q022/q022_table.py
(same algorithm, same truncation: odd indices <= M on both axes, odd-sine start, QR of the given set); only the bound changes.

  old (Q22-5):  theta_s = min(1, E_{s-1} / (nu^2 sigma_N)),   stop at the first s with E_{s-1} >= lambda11 sigma_N;
  new (N23-6):  theta_s = min(1, E_{s-1} / (lambda11 sigma_N)), no stop (rank of the exact field certified by N23-4 for the
                ten control r, N = 2, 3; outside the certified cells the new column is not claimed).
Monotone upper bound (N23-7): err(N, s) <= U_s := min_{s1 <= s} ( sqrt(err_M(s1)^2 + T^2) + E_{s1} )  for the untruncated method.
All numbers: double evaluation of rigorous formulas (as in Q22, sec. 7); rounding of the recursion is outside the budget.
Units a = 1, b = r, q = D = 1.  Usage: python n23_transfer_sharp.py M S   (e.g. 4001 40)
"""
import sys

import numpy as np

PI = np.pi
R = [1 / 5, 1 / 3, 1 / 2, 2 / 3, 1.0, 1.4, 1.5, 2.0, 3.0, 5.0]


def study(M, S, r, N):
    a, b = 1.0, r
    m = np.arange(1, M + 1, 2, dtype=float)
    K = len(m)
    k2 = (PI * m / a) ** 2
    n2 = (PI * m / b) ** 2
    lam = (k2[:, None] + n2[None, :]) ** 2
    om = 16.0 / (PI ** 2 * m[:, None] * m[None, :] * lam)
    en = lambda C: np.sqrt(a * b / 4 * np.sum(lam * C * C))
    T = np.sqrt(64 * a * b / PI ** 8 * (a ** 4 + b ** 4) * PI ** 2 / (80 * M ** 5))
    lam11 = PI ** 2 * (1 / a ** 2 + 1 / b ** 2)
    sgn = np.outer(np.sin(m * PI / 2), np.sin(m * PI / 2))
    w0c = np.sum(om * sgn)
    tail = (a * b) ** 2 / (4 * PI ** 4) * 2 * (1 / (2 * M)) * (PI ** 2 / 8) / (a * b / 4)
    Gc = np.sqrt(np.sum(1 / (a * b / 4 * lam)) + tail)
    X = np.zeros((K, N))
    X[np.arange(N), np.arange(N)] = 1.0
    given, side, prev = X, 'x', None
    Eo = En = None
    sig = None
    rows = []
    for s in range(1, S + 1):
        Q, _ = np.linalg.qr(given)
        if side == 'x':
            beta, gam, f = Q.T @ (k2[:, None] * Q), Q.T @ ((k2 ** 2)[:, None] * Q), Q.T @ (1.0 / m)
            G = n2[:, None, None] ** 2 * np.eye(N) + 2 * n2[:, None, None] * beta + gam
            B = np.linalg.solve(G, np.broadcast_to(f, (K, N))[..., None])[..., 0] * (16 / PI ** 2 / m)[:, None]
            C = Q @ B.T
            given, side, nu2 = B, 'y', (PI / b) ** 2
        else:
            beta, gam, f = Q.T @ (n2[:, None] * Q), Q.T @ ((n2 ** 2)[:, None] * Q), Q.T @ (1.0 / m)
            G = k2[:, None, None] ** 2 * np.eye(N) + 2 * k2[:, None, None] * beta + gam
            A = np.linalg.solve(G, np.broadcast_to(f, (K, N))[..., None])[..., 0] * (16 / PI ** 2 / m)[:, None]
            C = A @ Q.T
            given, side, nu2 = A, 'x', (PI / a) ** 2
        err = en(om - C)
        Rr = np.sqrt(T ** 2 + err ** 2)
        if s == 1:
            Eo = En = T
            go = gn = 0.0
        else:
            d = en(C - prev)
            if np.isfinite(Eo) and Eo / lam11 < sig:
                so = min(1.0, Eo / (nu2 * sig))
                Eo = np.sqrt((T + so * Rr) ** 2 + (Eo + so * d) ** 2)
            else:
                Eo = np.inf
            sn = min(1.0, En / (lam11 * sig))
            En = np.sqrt((T + sn * Rr) ** 2 + (En + sn * d) ** 2)
            go = (Rr + d) / (nu2 * sig)
            gn = (Rr + d) / (lam11 * sig)
        sv = np.linalg.svd(given, compute_uv=False)
        sig = np.sqrt(a * b) / 2 * sv[N - 1]
        cen = abs(np.sum(C * sgn) - w0c)
        rows.append(dict(s=s, err=err, Rr=Rr, Eo=Eo, En=En, go=go, gn=gn, cen=cen))
        prev = C
    return dict(T=T, Gc=Gc, rows=rows)


def smax(rows, key, Gc):
    out = {}
    for meas in ('err', 'erel', 'cen'):
        last = 0
        for row in rows:
            E = row[key]
            q = E / row['err']
            ok = {'err': q <= 1e-2, 'erel': 2 * q + q * q <= 1e-2, 'cen': Gc * E <= 1e-2 * row['cen']}[meas]
            if ok:
                last = row['s']
            else:
                break
        out[meas] = last
    return out


if __name__ == "__main__":
    M, S = int(sys.argv[1]), int(sys.argv[2])
    print("M=%d S=%d; s_max = last s with the 1/100 rule satisfied for all s' <= s (transfer only, as Q22 table 7.3)" % (M, S))
    print("U-excess = max_{2<=s<=S} (U_s - err_M(s)) / err_M(s): upper side of the untruncated energy error (Corollary N23-7)")
    for r in R:
        cells = []
        for N in (2, 3):
            st = study(M, S, r, N)
            rows = st['rows']
            so = smax(rows, 'Eo', st['Gc'])
            sn = smax(rows, 'En', st['Gc'])
            U = np.inf
            exc = 0.0
            exc2 = 0.0
            s1best = []
            for row in rows:
                cand = np.sqrt(row['err'] ** 2 + st['T'] ** 2) + row['En']
                if cand < U:
                    U = cand
                    s1 = row['s']
                if row['s'] >= 2:
                    exc = max(exc, (U - row['err']) / row['err'])
                    exc2 = max(exc2, (U ** 2 - row['err'] ** 2) / row['err'] ** 2)
                if row['s'] in (2, 3, 4, 10, 20, 40):
                    s1best.append("%d:%d" % (row['s'], s1))
            g_old = np.median([rw['go'] for rw in rows[-5:]])
            g_new = np.median([rw['gn'] for rw in rows[-5:]])
            cells.append("N=%d old(err/erel/cen)=%d/%d/%d new=%d/%d/%d g_tail old %.2g new %.2g | U-excess err %.2e erel %.2e "
                         "(s:s1 %s)" % (N, so['err'], so['erel'], so['cen'], sn['err'], sn['erel'], sn['cen'], g_old, g_new,
                                         exc, exc2, ",".join(s1best)))
        print("r=%-5.3g " % r + " || ".join(cells), flush=True)
