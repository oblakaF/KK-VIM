"""QA-δ2 (R4_NUMERICAL_SPEC 5.2.1, item (4); РВ4-5): the sine transform DST-I by a dense orthonormal Psi and by
``scipy.fft.dst(type=1, norm="ortho")``. No import of ``kkvim`` (D-028).

Definition: Psi_{ik} = sqrt(2/n) sin(pi i k / n), i, k = 1..n-1, symmetric and orthogonal (Psi^2 = I). For a vector of
length n - 1, scipy's DST-I with norm="ortho" is y_k = sqrt(2/n) sum_i x_i sin(pi i k / n) = (Psi x)_k (checked here).

Strict bound for the dense product (own; IEEE model, round to nearest, no underflow; ASSUMPTIONS: the library sin is
accurate within one ulp for arguments in [0, pi/2], and the matrix product uses classical dot products, any order):
every computed entry Psi^_{ik} = sqrt(2/n) sin(pi r/n), r reduced in integers to [0, n/2], satisfies
|Psi^ - Psi| <= 7u |Psi| (2/n, sqrt, pi r / n: 4 roundings; sin within one ulp plus the argument error |x cot x| <= 1:
2u; product: 1u), and |fl(Psi^ x) - Psi^ x| <= gamma_{n-1} |Psi^| |x|, so
    |fl(Psi^ x) - Psi x| <= (7u + gamma_{n-1} (1 + 7u)) |Psi| |x|   (componentwise).
For the fast transform no bound is derived: the c u log2(n) model of R4_NUMERICAL_SPEC 5.2.1 requires a Cooley-Tukey
algorithm with accurate twiddle factors, which is not verified for the library (NOT_VERIFIED); only empirical errors
against a high-precision reference are measured (report, section 4).
"""

import math

import numpy as np
import scipy.fft

from kkvim_reference.r4_spectral_control import dst_matrix

__all__ = ["dense_dst", "fast_dst", "dense_bound", "ones_transform", "U_ROUND"]

U_ROUND = 2.0**-53


def dense_dst(x):
    """Psi x along axis 0 (x of shape (n-1,) or (n-1, N)) with the dense orthonormal matrix."""
    x = np.asarray(x, dtype=float)
    return dst_matrix(x.shape[0] + 1) @ x


def fast_dst(x):
    """scipy.fft.dst(x, type=1, norm="ortho") along axis 0."""
    return scipy.fft.dst(np.asarray(x, dtype=float), type=1, norm="ortho", axis=0)


def dense_bound(x):
    """Componentwise strict bound of |fl(Psi^ x) - Psi x| for the dense product (assumptions in the docstring)."""
    x = np.asarray(x, dtype=float)
    n = x.shape[0] + 1
    gamma = (n - 1) * U_ROUND / (1 - (n - 1) * U_ROUND)
    return (7.0 * U_ROUND + gamma * (1.0 + 7.0 * U_ROUND)) * (np.abs(dst_matrix(n)) @ np.abs(x))


def ones_transform(n):
    """Exact (Psi 1)_k = sqrt(2/n) cot(k pi/(2n)) for odd k and 0 for even k (closed form, R4_NUMERICAL_SPEC 3.2)."""
    k = np.arange(1, n)
    return np.where(k % 2 == 1, math.sqrt(2.0 / n) / np.tan(k * np.pi / (2 * n)), 0.0)
