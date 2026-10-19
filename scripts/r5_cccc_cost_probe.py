"""R5 CCCC QA plan, cost probe only (not a reference): sparse LU (SuperLU) of the 13-point CCCC matrix
A = D4x(x)I + 2Lx(x)Ly + I(x)D4y on (r*n) x n intervals, h = 1/n (short side 1), random right-hand side. Measures time, factor nnz,
peak memory (with psutil) and the forward-error estimate from one refinement step in double precision.
No load vector and no alpha values are computed: docs/R5_CCCC_PREQUALIFICATION.plan.md is fixed before any D2 result.

    uv run --with psutil python scripts/r5_cccc_cost_probe.py MMD_AT_PLUS_A 5,256 2,512   # ordering, then r,n pairs
"""
import sys
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def ops(n, h):
    m = n - 1
    L = sp.diags([np.ones(m - 1), -2 * np.ones(m), np.ones(m - 1)], [-1, 0, 1], format="csr") / h**2
    D4 = (L @ L).tolil()
    D4[0, 0] += 2 / h**4
    D4[m - 1, m - 1] += 2 / h**4
    return L, D4.tocsr()


def matrix(nx, ny, hx, hy):
    Lx, D4x = ops(nx, hx)
    Ly, D4y = ops(ny, hy)
    Ix, Iy = sp.identity(nx - 1, format="csr"), sp.identity(ny - 1, format="csr")
    return (sp.kron(D4x, Iy) + 2 * sp.kron(Lx, Ly) + sp.kron(Ix, D4y)).tocsc()


def probe(r, n, spec):
    nx, ny = r * n, n
    A = matrix(nx, ny, 1.0 / n, 1.0 / n)
    t0 = time.perf_counter()
    lu = spla.splu(A, permc_spec=spec)
    t1 = time.perf_counter()
    rng = np.random.default_rng(1)
    b = rng.standard_normal(A.shape[0])
    x = lu.solve(b)
    t2 = time.perf_counter()
    dx = lu.solve(b - A @ x)
    fill = lu.L.nnz + lu.U.nnz
    return {"r": r, "n": n, "unknowns": A.shape[0], "nnz_A": A.nnz, "spec": spec, "factor_s": round(t1 - t0, 2),
            "solve_s": round(t2 - t1, 3), "factor_nnz": fill, "factor_MB_est": round(fill * 12 / 2**20),
            "fwd_err_est": float(np.max(np.abs(dx)) / np.max(np.abs(x))), "peak_MB": _peak()}


def _peak():
    try:
        import psutil
        info = psutil.Process().memory_info()
        return round(getattr(info, "peak_wset", getattr(info, "rss", 0)) / 2**20)
    except ImportError:
        return None


if __name__ == "__main__":
    cases = [(int(a), int(b)) for a, b in (s.split(",") for s in sys.argv[2:])]
    for r, n in cases:
        print(probe(r, n, sys.argv[1]), flush=True)
