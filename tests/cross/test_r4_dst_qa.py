"""QA-δ2 (R4_NUMERICAL_SPEC 5.2.1 (4); РВ4-5): DST-I by the dense orthonormal Psi and by scipy.fft.dst(type=1,
norm="ortho"), n <= 640, N = 2-3 columns, low and high harmonics, random and constructed inputs.

Without mpmath: identities with exact references (Psi psi_k = e_k is not exact in floating point, so only agreement
with the dense product, orthogonality, the involution and the closed form of Psi 1 are checked).
With mpmath (temporary QA environment, `uv run --with mpmath`; not a project dependency, skipped in CI): the strict
componentwise bound of the dense product (kkvim_reference.r4_dst_check) is checked against a 40-digit reference; the
fast transform is only measured (empirical, normwise) - no bound for it is claimed (c_DST NOT_VERIFIED).
"""

import math

import numpy as np
import pytest

from kkvim_reference.r4_dst_check import U_ROUND, dense_bound, dense_dst, fast_dst, ones_transform
from kkvim_reference.r4_spectral_control import dst_matrix

SIZES = (4, 7, 20, 64, 128, 320, 640)


def inputs(n, rng):
    m = n - 1
    i = np.arange(1, n)
    out = {
        "random N=3": rng.standard_normal((m, 3)),
        "graded": rng.standard_normal((m, 3)) * np.array([1.0, 1e-8, 1e8]),
        "dynamic range": rng.standard_normal((m, 2)) * 10.0 ** rng.uniform(-8, 8, (m, 2)),
        "modes 1, mid, n-1": dst_matrix(n)[:, [0, (m - 1) // 2, m - 1]],
        "spikes": np.eye(m)[:, sorted({0, (m - 1) // 2, m - 1})],
        "ones, alternating": np.column_stack([np.ones(m), (-1.0) ** i]),
    }
    if n >= 6:
        out["odd sines N=3"] = np.column_stack([np.sin((2 * j - 1) * np.pi * i / n) for j in (1, 2, 3)])
    return out


@pytest.mark.parametrize("n", SIZES)
def test_scipy_ortho_dst_is_the_dense_psi(n):
    rng = np.random.default_rng(n)
    for name, x in inputs(n, rng).items():
        d, f = dense_dst(x), fast_dst(x)
        scale = np.abs(dst_matrix(n)) @ np.abs(x)
        # both are the same linear map; the difference is rounding of two algorithms (observed <= ~40 u |Psi||x|)
        assert np.max(np.abs(d - f) / np.where(scale > 0, scale, 1.0)) <= 200 * U_ROUND, name
        assert np.max(np.linalg.norm(fast_dst(f) - x, axis=0) / np.linalg.norm(x, axis=0)) <= 16 * U_ROUND, name


@pytest.mark.parametrize("n", SIZES)
def test_orthogonality_involution_and_closed_form_of_psi_1(n):
    P = dst_matrix(n)
    assert np.max(np.abs(P @ P - np.eye(n - 1))) <= 4 * n * U_ROUND
    c = ones_transform(n)
    for y in (dense_dst(np.ones(n - 1)), fast_dst(np.ones(n - 1))):
        assert np.max(np.abs(y - c)) <= 16 * U_ROUND * np.max(np.abs(c))
        assert np.max(np.abs(y[1::2])) <= 16 * U_ROUND * np.max(np.abs(c))  # even modes vanish (R4_NUMERICAL_SPEC 3.2)


@pytest.mark.parametrize("n", (20, 64, 320, 640))
def test_high_precision_dense_bound_and_fast_errors(n):
    mp = pytest.importorskip("mpmath")
    mp.mp.dps = 40
    s = [mp.mpf(0) if r % n == 0 else mp.sin(mp.pi * r / n) for r in range(2 * n)]
    c = mp.sqrt(mp.mpf(2) / n)
    P = [[c * s[(i * k) % (2 * n)] for k in range(1, n)] for i in range(1, n)]
    rng = np.random.default_rng(1000 + n)
    worst_fast = 0.0
    for name, x in inputs(n, rng).items():
        x = x[:, :2]
        exact = [[mp.fsum(P[i][j] * mp.mpf(float(x[j, col])) for j in range(n - 1)) for col in range(x.shape[1])]
                 for i in range(n - 1)]
        for y, kind in ((dense_dst(x), "dense"), (fast_dst(x), "fast")):
            err = np.array([[abs(float(mp.mpf(float(y[i, col])) - exact[i][col])) for col in range(x.shape[1])]
                            for i in range(n - 1)])
            if kind == "dense":
                assert np.all(err <= dense_bound(x)), name  # strict bound (assumptions in r4_dst_check)
            else:
                worst_fast = max(worst_fast, float(np.max(np.linalg.norm(err, axis=0) / np.linalg.norm(x, axis=0))))
    # observation only: normwise error of the fast transform in units of u log2(n) (no bound is claimed)
    assert worst_fast <= 10 * U_ROUND * math.log2(n)
