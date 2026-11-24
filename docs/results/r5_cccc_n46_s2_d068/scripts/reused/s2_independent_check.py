"""QA R5-CCCC-N46-S2-D068-INDEPENDENT: independent check of the saved s = 2 X-half-step, CCCC N = 4-6 (QA scratch, not in git).

No import of ``kkvim`` (checked). The expected discrete problem is assembled from ``kkvim_reference.r5_cccc_d2.nodewise_parts``
(13-point node-by-node stencil with even ghost nodes, D-049.1: D1 = D2) and projected on the ACTUAL saved G2:

    K x = f,  K = P^T A P,  f = (q/D) P^T 1,  P = I_(n_x-1) (x) G2,  vec(w) = P vec(X), w = X G2^T  (x index slow)

Criteria, fixed BEFORE any comparison (see CRITERIA below; no universal tolerance):
  X1  rigorous forward bound  ||X_hat - X*||_2 <= ||rho||_2 / lambda_low(K),  rho = K x_hat - f computed EXACTLY (fractions);
      lambda_low(K) >= lambda_low(A) * lambda_low(G2^T G2): lambda_low(A) >= mu_x^2 + mu_y^2 (A = D4x(x)I + 2Lx(x)Ly + I(x)D4y,
      verified exactly on the integer parts; Lx(x)Ly >= 0, D4 >= L^2), mu = (4/h^2) sin^2(pi/2n) with sin t >= t - t^3/6 and a
      lower decimal of pi; lambda(G2^T G2) by exact Gershgorin.
      PASS iff  B_rel = bound / ||x_hat|| <= 2 kappa_up n u,  n = N (n_x - 1), kappa_up = lambda_up(K)/lambda_low(K) with
      lambda_up by exact Gershgorin: the classical normwise forward bound of a solve with backward error <= n u ||K|| (factor 2 for
      the 1/(1 - kappa eps) term and ||x|| vs ||x_hat||). A wrong operator, orientation, basis or scale gives B_rel ~ 1.
      If B_rel exceeds it: UNRESOLVED (not PASS).
  F1  field w = X G2^T: |w_csv - (X G2^T)_exact| <= gamma_N sum_k |X_ik||G2_jk| at every interior node (rigorous for a dot product
      of length N); boundary nodes exactly 0; x = i h_x on axis 0.
  E1  P_hat (Builder, binary64, C-3+ form) vs Pi_exact (nodewise operator, Proposition D: Hess = D h_x h_y A, load q h_x h_y sum w):
      |P_hat - Pi_exact| <= gamma_m S_abs, m = 8 max(n_x, n_y) + 4 N^2 + 16, S_abs = 1/2 c |w|^T|A||w| + q h_x h_y sum|w|
      (a stated first-order rounding estimate, not a proof for C-3+).
  E2  monotonicity at the proven precision: Pi(w2_hat) <= Pi(companion G2^T) + 1/2 c ||rho||^2 / lambda_low(K) (Galerkin
      optimality of w2* in the space that contains companion G2^T); Pi(w2) - Pi(w1) reported (observation; not convergence).
  R7  RT-7 re-derived independently: exact ranks of V and G2 by own fraction elimination; tau = N - tr(P_V P_G2) exactly;
      t from the INDEPENDENT RN-5 QA bounds (eps, sigma2_low of rn5_n46_results_all.json, same saved G, V);
      R7-2: t < 1; R7-3: tau < (1 - t)^2, strict, exact.
"""

import csv
import hashlib
import io
import json
import math
import sys
import time
from fractions import Fraction as Q
from math import isqrt
from pathlib import Path

import numpy as np

QA = Path(r"<QA_WORKTREE>")
sys.path.insert(0, str(QA / "src"))
from kkvim_reference import r5_cccc_d2 as d2  # noqa: E402

assert not any(m == "kkvim" or m.startswith("kkvim.") for m in sys.modules), "kkvim must not be imported"

BUILDER = Path(r"<BUILDER_WORKTREE>\scratch\d068")
RN5SRC = Path(r"<BUILDER_WORKTREE>\scratch\d067\rn5_feasibility")
RN5QA = QA / "scratch" / "rn5_n46" / "rn5_n46_results_all.json"
RUNS = QA / "scratch" / "d068_s2" / "runs"
OUT = QA / "scratch" / "d068_s2"
COMMIT = "754dcfe1141ca3bb04812839ce0e7db69f67f057"
CASES = [("C1", "N4_32x32", 4, 32, 32, 1.0, "C1_N4_32x32"), ("C2", "N5_32x32", 5, 32, 32, 1.0, "C2_N5_32x32"),
         ("C3", "N6_32x32", 6, 32, 32, 1.0, "C3_N6_32x32"), ("C4", "N4_9x9", 4, 9, 9, 1.0, "C4_N4_9x9"),
         ("C5", "N5_11x11", 5, 11, 11, 1.0, "C5_N5_11x11"), ("C6", "N6_13x13", 6, 13, 13, 1.0, "C6_N6_13x13"),
         ("C7", "N5_aniso_b0.6_20x13", 5, 20, 13, 0.6, "C7_N5_20x13")]
A_SIDE, T, E, NU, QLOAD = 1.0, 0.01, 2.1e11, 0.3, 1000.0
D_HEX = "0x1.2c7b13b13b13cp+14"
U = Q(1, 2**53)
PI_LO = Q(314159265358979323846, 10**20)  # truncated decimal of pi, below pi
CRITERIA = __doc__[__doc__.index("Criteria"):]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gamma(n: int) -> Q:
    return n * U / (1 - n * U)


def sqrt_up(x: Q, bits: int = 120) -> Q:
    if x <= 0:
        return Q(0)
    s = 1 << bits
    r = isqrt((x.numerator * s * s) // x.denominator)
    return Q(r + 1, s)


def sqrt_dn(x: Q, bits: int = 120) -> Q:
    if x <= 0:
        return Q(0)
    s = 1 << bits
    return Q(isqrt((x.numerator * s * s) // x.denominator), s)


def rank_exact(M) -> int:
    """Rank of a matrix of fractions by own Gaussian elimination."""
    rows = [list(r) for r in M]
    rank, ncol = 0, len(rows[0])
    for c in range(ncol):
        p = next((r for r in range(rank, len(rows)) if rows[r][c] != 0), None)
        if p is None:
            continue
        rows[rank], rows[p] = rows[p], rows[rank]
        piv = rows[rank][c]
        for r in range(rank + 1, len(rows)):
            if rows[r][c] != 0:
                f = rows[r][c] / piv
                rows[r] = [a - f * b for a, b in zip(rows[r], rows[rank])]
        rank += 1
    return rank


def solve_exact(M, B):
    """M^-1 B for a nonsingular fraction matrix by Gauss-Jordan (small N only)."""
    n = len(M)
    aug = [list(M[i]) + list(B[i]) for i in range(n)]
    for c in range(n):
        p = next(r for r in range(c, n) if aug[r][c] != 0)
        aug[c], aug[p] = aug[p], aug[c]
        piv = aug[c][c]
        aug[c] = [v / piv for v in aug[c]]
        for r in range(n):
            if r != c and aug[r][c] != 0:
                f = aug[r][c]
                aug[r] = [a - f * b for a, b in zip(aug[r], aug[c])]
    return [row[n:] for row in aug]


def gram(A, B):
    return [[sum(A[p][i] * B[p][j] for p in range(len(A))) for j in range(len(B[0]))] for i in range(len(A[0]))]


def gersh(M):
    lo = min(M[i][i] - sum(abs(M[i][j]) for j in range(len(M)) if j != i) for i in range(len(M)))
    hi = max(M[i][i] + sum(abs(M[i][j]) for j in range(len(M)) if j != i) for i in range(len(M)))
    return lo, hi


# ------------------------------------------------------------------------------------------------------------------ reading
def read_csv_rows(path: Path):
    text = path.read_text(encoding="utf-8")
    lines = [ln for ln in text.splitlines() if not ln.startswith("#")]
    return list(csv.DictReader(io.StringIO("\n".join(lines))))


def read_factors(path: Path) -> dict:
    out = {}
    for r in read_csv_rows(path):
        out.setdefault((int(r["s"]), r["role"]), []).append((int(r["i"]), int(r["j"]), float(r["value"])))
    mats = {}
    for key, items in out.items():
        m = max(i for i, _, _ in items)
        n = max(j for _, j, _ in items)
        A = np.full((m, n), np.nan)
        for i, j, v in items:
            A[i - 1, j - 1] = v
        assert not np.isnan(A).any(), key
        mats[key] = A
    return mats


def fr(A: np.ndarray):
    return [[Q(float(v)) for v in row] for row in np.atleast_2d(A)]


def walk_strings(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield str(k)
            yield from walk_strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from walk_strings(v)
    elif isinstance(obj, str):
        yield obj


# ------------------------------------------------------------------------------------------------------------------ operator
class Operator:
    """The exact nodewise operator A of D2 (A vec(w) = q/D 1) for one grid; integer parts and exact scales of the reference."""

    def __init__(self, a: float, b: float, nx: int, ny: int):
        self.problem = d2.D2Problem(a, b, nx, ny)
        self.parts = [z.tocsr() for z in d2.nodewise_parts(nx, ny)]
        self.scales = self.problem.scales()
        self.nx, self.ny = nx, ny
        self.hx, self.hy = Q(a) / nx, Q(b) / ny

    def kron_structure_exact(self) -> bool:
        """Z_x = D4i_x (x) I, Z_xy = 2 Li_x (x) Li_y, Z_y = I (x) D4i_y as integer matrices (D4i = Li^2 + 2(e1 e1^T + em em^T))."""
        import scipy.sparse as sp

        def li(n):
            m = n - 1
            return sp.diags([np.ones(m - 1, np.int64), -2 * np.ones(m, np.int64), np.ones(m - 1, np.int64)], [-1, 0, 1], dtype=np.int64)

        def d4i(n):
            L = li(n).tocsr()
            c = np.zeros(n - 1, np.int64)
            c[0] = c[-1] = 2
            return (L @ L + sp.diags([c], [0], dtype=np.int64)).tocsr()

        Ix, Iy = sp.identity(self.nx - 1, dtype=np.int64), sp.identity(self.ny - 1, dtype=np.int64)
        ref = [sp.kron(d4i(self.nx), Iy), 2 * sp.kron(li(self.nx), li(self.ny)), sp.kron(Ix, d4i(self.ny))]
        ok = all((z.astype(np.int64) - r.astype(np.int64)).count_nonzero() == 0 for z, r in zip(self.parts, ref))
        hs = (1 / self.hx**4, 1 / (self.hx**2 * self.hy**2), 1 / self.hy**4)
        return ok and all(s == h for s, h in zip(self.scales, hs))

    def apply_exact(self, w: list) -> list:
        total = [Q(0)] * len(w)
        for z, s in zip(self.parts, self.scales):
            ip, ix, dv = z.indptr, z.indices, z.data
            for r in range(len(w)):
                acc = 0
                for t in range(ip[r], ip[r + 1]):
                    acc += int(dv[t]) * w[ix[t]]
                total[r] += s * acc
        return total

    def abs_apply_float(self, w: np.ndarray) -> np.ndarray:
        return sum(float(s) * (abs(z).astype(float) @ w) for z, s in zip(self.parts, self.scales))

    def gersh_up(self) -> Q:
        best = Q(0)
        for r in range(self.problem.size):
            v = sum(s * int(np.abs(z.data[z.indptr[r]:z.indptr[r + 1]]).sum()) for z, s in zip(self.parts, self.scales))
            best = max(best, v)
        return best

    def lambda_low(self) -> Q:
        def mu(n, h):
            t = PI_LO / (2 * n)
            s = t - t**3 / 6  # sin t >= t - t^3/6 for t > 0
            return 4 / h**2 * s * s
        return mu(self.nx, self.hx) ** 2 + mu(self.ny, self.hy) ** 2


def pi_exact(op: Operator, w: list, D: Q, q: Q):
    """(Pi_h^C(w), A w) exactly: Pi = 1/2 D h_x h_y w^T A w - q h_x h_y sum w (Proposition D)."""
    Aw = op.apply_exact(w)
    c = D * op.hx * op.hy
    return c / 2 * sum(a * b for a, b in zip(w, Aw)) - q * op.hx * op.hy * sum(w), Aw


def field_exact(Xf, Yf):
    """vec of Xf Yf^T, x index slow (ordering k = (i - 1)(n_y - 1) + (j - 1) of the reference)."""
    N = len(Xf[0])
    return [sum(Xf[i][k] * Yf[j][k] for k in range(N)) for i in range(len(Xf)) for j in range(len(Yf))]


# ------------------------------------------------------------------------------------------------------------------ one case
def check_case(case, rn5qa: dict, eps_x: float | None = None) -> dict:
    cid, label, N, nx, ny, b, bdir = case
    t0 = time.time()
    src = BUILDER / bdir
    res = {"case": cid, "N": N, "nx": nx, "ny": ny, "b": b, "builder_dir": str(src), "checks": {}, "values": {}}
    ck, val = res["checks"], res["values"]
    files = {name: src / name for name in ("result.json", "history.csv", "factors.csv", "field_research.csv")}
    res["builder_sha256"] = {k: sha(p) for k, p in files.items()}
    doc = json.loads(files["result.json"].read_text(encoding="utf-8"))
    hist = read_csv_rows(files["history.csv"])
    fac = read_factors(files["factors.csv"])
    field = read_csv_rows(files["field_research.csv"])

    # --- 1. run, provenance, statuses, s1 = RN-5 inputs of the same run
    D = float(doc["D"]["value"]) if isinstance(doc["D"], dict) else float(doc["D"])
    ck["D_eq_hex"] = D == float.fromhex(D_HEX)
    ck["provenance_commit"] = doc["provenance"]["git_commit"] == COMMIT and doc["provenance"]["git_worktree_modified"] is False
    ck["schema"] = doc["schema"] == "kkvim.cli.cccc_n46/2"
    inp = doc["input"]
    ck["input"] = (inp["N"], inp["n_x"], inp["n_y"], inp["half_steps"]) == (N, nx, ny, 2) and inp["b"]["value"] == b and \
        inp["a"]["value"] == A_SIDE and inp["t"]["value"] == T and inp["E"]["value"] == E and inp["nu"]["value"] == NU and inp["q"]["value"] == QLOAD
    ck["status_completed_2"] = doc["status"] == "COMPLETED_FIXED_HALF_STEPS" and doc["half_steps_completed"] == 2 and doc["half_steps_requested"] == 2 \
        and doc["reason"] is None
    ck["not_qualified_label"] = "NOT_QUALIFIED" in json.dumps(doc["application_status"], ensure_ascii=False)
    forbidden = {"CONVERGED", "ACCEPTED", "QUALIFIED", "Q5_ACCEPTED"}
    ck["no_convergence_or_qualification_status"] = not any(s in forbidden for s in walk_strings(doc)) and \
        all(r["status"] in ("HALF_STEP_DONE", "COMPLETED_FIXED_HALF_STEPS") for r in hist)
    with np.load(RN5SRC / f"factors_{label}.npz", allow_pickle=False) as z:
        G1n, V1n = z["G"], z["V"]
    G1, V1 = fac[(1, "G")], fac[(1, "V")]
    ck["s1_G_V_bitwise_eq_rn5_feasibility_npz"] = np.array_equal(G1, G1n) and np.array_equal(V1, V1n) and G1.tobytes() == G1n.tobytes() \
        and V1.tobytes() == V1n.tobytes()
    rnq = next(c for c in rn5qa["cases"] if c["case"] == cid)
    ck["independent_rn5_qa_certified_same_factors"] = rnq["independent"]["certified"] is True and rnq["label"] == label
    h2 = doc["rank"]["half_step_2"]
    feas = json.loads((RN5SRC / "rn5_feasibility_results.json").read_text(encoding="utf-8"))
    ck["rn5_in_run_certified"] = h2["rn5"]["certified"] is True and h2["rn5"]["verdict"] == "RN5_CERTIFIED" and all(h2["rn5"]["hypotheses"].values())
    fv = next((c for c in (feas["cases"] if isinstance(feas, dict) and "cases" in feas else feas) if label in json.dumps(c)), None)
    val["rn5_run"] = {k: h2["rn5"]["values"].get(k) for k in ("lambda_low", "norm_dK", "eps", "sigma_low", "margin_sigma_over_eps")}
    val["rn5_feasibility_match"] = None if fv is None else {k: (json.dumps(fv).find(repr(v)) >= 0) for k, v in val["rn5_run"].items() if v is not None}
    ck["qr_after_rn5_rt7_passed_record"] = h2["qr_executed"] is True and h2["rt7"]["passed"] is True and all(h2["rt7"][k] for k in ("R7-1", "R7-2", "R7-3"))

    # --- 2. QR of V, companion, R
    G2, R2, X, comp = fac[(2, "G")], fac[(2, "R")], fac[(2, "V")], fac[(2, "companion")]
    ck["shapes_s2"] = G2.shape == (ny - 1, N) and X.shape == (nx - 1, N) and R2.shape == (N, N) and comp.shape == (nx - 1, N)
    ck["R_upper_triangular"] = bool(np.all(np.tril(R2, -1) == 0.0)) and bool(np.all(np.diag(R2) != 0.0))
    hy_f = b / ny
    val["qr_relation_max_rel"] = float(np.max(np.abs(G2 @ R2 - V1)) / np.max(np.abs(V1)))
    val["G2_orthonormality_dev"] = float(np.max(np.abs(hy_f * G2.T @ G2 - np.eye(N))))
    G1f, R2f, compf = fr(G1), fr(R2), fr(comp)
    ok = True
    gN = gamma(N)
    for i in range(nx - 1):
        for k in range(N):
            ex = sum(G1f[i][m] * R2f[k][m] for m in range(N))
            bound = gN * sum(abs(G1f[i][m] * R2f[k][m]) for m in range(N))
            ok &= abs(compf[i][k] - ex) <= bound
    ck["companion_eq_G1_RT_within_gamma_N"] = bool(ok)

    # --- 3. RT-7 independently (exact)
    V1f, G2f = fr(V1), fr(G2)
    rV, rG = rank_exact(V1f), rank_exact(G2f)
    GV, GG, VG = gram(V1f, V1f), gram(G2f, G2f), gram(V1f, G2f)
    M1 = solve_exact(GV, VG)                       # GV^-1 V^T G2
    M2 = solve_exact(GG, [list(r) for r in zip(*VG)])  # GG^-1 G2^T V
    tau = N - sum(sum(M1[i][m] * M2[m][i] for m in range(N)) for i in range(N))
    dec = rnq["independent"]["decisive_exact"]
    eps_q, s2_q = Q(dec["eps"]), Q(dec["sigma2_low"])
    t_up = eps_q / sqrt_dn(s2_q)
    ck["RT7_R7_1_independent_ranks"] = rV == N and rG == N
    ck["RT7_R7_2_independent"] = 0 <= t_up < 1
    ck["RT7_R7_3_independent"] = tau < (1 - t_up) ** 2
    val["rt7"] = {"tau_exact_float": float(tau), "tau_builder": h2["rt7"]["tau"], "t_qa_upper": float(t_up), "t_builder": h2["rt7"]["t"],
                  "sin_theta_upper_qa": float(sqrt_up(tau) + t_up), "rank_V": rV, "rank_G2": rG}

    # --- 4. the X system on the actual G2, independent operator
    op = Operator(A_SIDE, b, nx, ny)
    ck["operator_kron_structure_exact"] = op.kron_structure_exact()
    Dq, qq = Q(D), Q(QLOAD)
    Xf = fr(X)
    w2 = field_exact(Xf, G2f)
    Pi2, Aw2 = pi_exact(op, w2, Dq, qq)
    load = qq / Dq
    r = [v - load for v in Aw2]
    m_y = ny - 1
    rho = [[sum(G2f[j][k] * r[i * m_y + j] for j in range(m_y)) for k in range(N)] for i in range(nx - 1)]
    rho2 = sum(v * v for row in rho for v in row)
    x2 = sum(v * v for row in Xf for v in row)
    gl, gh = gersh(GG)
    lamA = op.lambda_low()
    lam_low = lamA * gl
    lam_up = op.gersh_up() * gh
    ck["lambda_low_K_positive"] = gl > 0 and lam_low > 0
    n_dim = N * (nx - 1)
    bound = sqrt_up(rho2) / lam_low
    B_rel = bound / sqrt_dn(x2)
    kappa = lam_up / lam_low
    crit = 2 * kappa * n_dim * U
    ck["X1_forward_bound_within_criterion"] = B_rel <= crit
    val["X"] = {"rho_norm2": float(sqrt_up(rho2)), "lambda_low_K": float(lam_low), "lambda_up_K": float(lam_up), "kappa_up": float(kappa),
                "abs_bound_X": float(bound), "B_rel": float(B_rel), "criterion_2kappa_n_u": float(crit), "n": n_dim,
                "lambda_low_A": float(lamA), "gram_G2_gersh": [float(gl), float(gh)]}
    # float reference solve of the same projected system (observation only)
    import scipy.sparse as sp
    Af = d2.assemble(op.problem)
    P = sp.kron(sp.identity(nx - 1), sp.csr_matrix(G2)).tocsc()
    Kf = (P.T @ (Af @ P)).toarray()
    ff = float(load) * (P.T @ np.ones(op.problem.size))
    xref = np.linalg.solve(Kf, ff).reshape(nx - 1, N)
    val["X"]["float_reference_rel_diff"] = float(np.linalg.norm(xref - X) / np.linalg.norm(X))
    val["X"]["normwise_backward_error_obs"] = float(sqrt_up(rho2) / (lam_up * sqrt_dn(x2) + sqrt_up(sum(v * v for v in [load * sum(G2f[j][k] for j in range(m_y)) for k in range(N)]) * (nx - 1))))
    if eps_x is not None:  # negative control: perturbed X must fail X1
        Xp = X * (1 + eps_x)
        Xpf = fr(Xp)
        w2p = field_exact(Xpf, G2f)
        rp = [v - load for v in op.apply_exact(w2p)]
        rhop = sum(sum(G2f[j][k] * rp[i * m_y + j] for j in range(m_y)) ** 2 for k in range(N) for i in range(nx - 1))
        Bp = sqrt_up(rhop) / lam_low / sqrt_dn(sum(v * v for row in Xpf for v in row))
        val["negative_control_perturbed_X"] = {"eps": eps_x, "B_rel": float(Bp), "fails_X1": Bp > crit}
        # wrong orientation control: X interpreted with the s = 1 basis G1 in place of G2 is not the system of G2 (only square grids)
        if nx == ny:
            rw = [v - load for v in op.apply_exact(field_exact(Xf, G1f))]
            rhow = sum(sum(G1f[j][k] * rw[i * m_y + j] for j in range(m_y)) ** 2 for k in range(N) for i in range(nx - 1))
            Bw = sqrt_up(rhow) / lam_low / sqrt_dn(x2)
            val["negative_control_wrong_basis_G1"] = {"B_rel": float(Bw), "fails_X1": Bw > crit}

    # --- 5. field orientation w2 = X G2^T
    W = {(int(f["i"]), int(f["j"])): f for f in field}
    ck["field_grid_complete"] = len(W) == (nx + 1) * (ny + 1)
    okb = okx = okw = okbar = True
    hx_f = A_SIDE / nx
    for (i, j), f in W.items():
        w = float(f["w [m]"])
        if i in (0, nx) or j in (0, ny):
            okb &= w == 0.0
            continue
        ex = w2[(i - 1) * m_y + (j - 1)]
        bnd = gN * sum(abs(Xf[i - 1][k] * G2f[j - 1][k]) for k in range(N))
        okw &= abs(Q(w) - ex) <= bnd
        okx &= abs(float(f["x [m]"]) - i * hx_f) <= 4e-16 * A_SIDE and abs(float(f["y [m]"]) - j * hy_f) <= 4e-16 * b
        okbar &= abs(float(f["w_bar [1]"]) - w * D / (QLOAD * A_SIDE**4)) <= 4 * 2.0**-53 * abs(w * D / (QLOAD * A_SIDE**4))
    ck["F1_field_eq_X_G2T_within_gamma_N"] = bool(okw)
    ck["field_boundary_zero"] = bool(okb)
    ck["field_axis0_is_x"] = bool(okx)
    ck["field_w_bar"] = bool(okbar)
    wmax = doc["field"]["w_max"]
    absw = {k: abs(float(f["w [m]"])) for k, f in W.items()}
    kmax = max(absw, key=absw.get)
    ck["field_record_w_max"] = abs(wmax["value"]) == absw[kmax] and (wmax["i"], wmax["j"]) == kmax

    # --- 6. energy
    w1 = field_exact(G1f, V1f)
    Pi1, _ = pi_exact(op, w1, Dq, qq)
    wc = field_exact(compf, G2f)
    Pic, _ = pi_exact(op, wc, Dq, qq)
    c = Dq * op.hx * op.hy
    galerkin_defect = c / 2 * sum(Xf[i][k] * rho[i][k] for i in range(nx - 1) for k in range(N))
    ck["energy_identity_exact_consistency"] = Pi2 == galerkin_defect - qq * op.hx * op.hy * sum(w2) / 2
    p_hist = {int(r["s"]): float(r["P_hat"]) for r in hist}
    P1b, P2b, Pcb = p_hist[1], p_hist[2], h2["P_hat_previous_QR"]
    ck["P_hat_history_eq_result"] = h2["P_hat"] == P2b
    m = 8 * max(nx, ny) + 4 * N * N + 16
    est = {}
    for name, wl, pb, pe in (("s1", w1, P1b, Pi1), ("s2", w2, P2b, Pi2), ("companion_G2", wc, Pcb, Pic)):
        wf = np.array([float(v) for v in wl])
        S_abs = 0.5 * float(c) * float(np.abs(wf) @ op.abs_apply_float(np.abs(wf))) + QLOAD * float(op.hx * op.hy) * float(np.abs(wf).sum())
        bnd = float(gamma(m)) * S_abs
        est[name] = {"P_hat_builder": pb, "Pi_exact": float(pe), "abs_diff": float(abs(Q(pb) - pe)), "estimate_gamma_m_S_abs": bnd,
                     "within": float(abs(Q(pb) - pe)) <= bnd}
    ck["E1_P_hat_vs_exact_within_estimate"] = all(e["within"] for e in est.values())
    solve_gap = c / 2 * rho2 / lam_low  # 1/2 c ||rho||^2 / lambda_low >= Pi(w2_hat) - Pi(w2*)
    ck["E2_monotone_within_proven_precision"] = Pi2 <= Pic + solve_gap
    val["energy"] = {**est, "Pi2_minus_Pi1": float(Pi2 - Pi1), "Pi2_minus_Pic": float(Pi2 - Pic), "Pic_minus_Pi1": float(Pic - Pi1),
                     "solve_gap_bound": float(solve_gap), "galerkin_defect_rel": float(abs(galerkin_defect) / abs(Pi2)),
                     "Pi2_le_Pi1_exact": Pi2 <= Pi1}

    # --- 7. history / result consistency
    ck["history_two_rows"] = len(hist) == 2 and hist[0]["status"] == "HALF_STEP_DONE" and hist[1]["status"] == "COMPLETED_FIXED_HALF_STEPS" \
        and hist[1]["direction"] == "X" and int(hist[1]["n_s"]) == nx and hist[1]["qr_executed"] == "true" \
        and hist[1]["rn5_verdict"] == "RN5_CERTIFIED" and hist[1]["rt7_passed"] == "true"
    eta = doc["numerical_policy"]["eta"]
    ck["eta_record"] = eta["half_step"] == 2 and float(eta["by_half_step"]["2"]) == float(hist[1]["eta"]) and \
        float(eta["by_half_step"]["1"]) == float(hist[0]["eta"]) and eta["value"] == eta["by_half_step"]["2"]
    ck["factors_s2_F_eq_qhG2T1"] = bool(np.max(np.abs(fac[(2, "F")].ravel() - QLOAD * hy_f * G2.sum(axis=0))) <= 1e-12 * np.max(np.abs(fac[(2, "F")])))

    # --- 8. QA rerun at the same SHA
    qa = RUNS / cid
    if (qa / "result.json").exists():
        qdoc = json.loads((qa / "result.json").read_text(encoding="utf-8"))
        same = {n: sha(qa / n) == res["builder_sha256"][n] for n in ("history.csv", "factors.csv", "field_research.csv")}
        a_, b_ = dict(doc), dict(qdoc)
        for d_ in (a_, b_):
            d_.pop("program", None)
        res["qa_rerun"] = {"files_bitwise_equal": same, "result_json_equal_except_program": a_ == b_,
                           "commit": qdoc["provenance"]["git_commit"], "modified": qdoc["provenance"]["git_worktree_modified"]}
        ck["qa_rerun_bitwise"] = all(same.values()) and a_ == b_ and qdoc["provenance"]["git_commit"] == COMMIT
    else:
        res["qa_rerun"] = None
        ck["qa_rerun_bitwise"] = False
    res["seconds"] = time.time() - t0
    res["all_checks"] = all(ck.values())
    return res


def main(argv):
    which = argv[1] if len(argv) > 1 else "all"
    rn5qa = json.loads(RN5QA.read_text(encoding="utf-8"))
    script_sha = sha(Path(__file__))
    out = {"task": "R5-CCCC-N46-S2-D068-INDEPENDENT", "script_sha256": script_sha, "rn5_qa_sha256": sha(RN5QA), "criteria": CRITERIA,
           "python": sys.version, "numpy": np.__version__, "kkvim_imported": any(m == "kkvim" or m.startswith("kkvim.") for m in sys.modules),
           "cases": []}
    for case in CASES:
        if which != "all" and case[0] != which:
            continue
        res = check_case(case, rn5qa, eps_x=1e-9 if case[0] == "C4" else None)
        out["cases"].append(res)
        bad = [k for k, v in res["checks"].items() if not v]
        print(case[0], "ALL_OK" if not bad else f"FAILED {bad}", f"B_rel={res['values']['X']['B_rel']:.3e} crit={res['values']['X']['criterion_2kappa_n_u']:.3e}",
              f"{res['seconds']:.1f}s", flush=True)
    out["kkvim_imported_end"] = any(m == "kkvim" or m.startswith("kkvim.") for m in sys.modules)
    (OUT / f"s2_check_{which}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv)
