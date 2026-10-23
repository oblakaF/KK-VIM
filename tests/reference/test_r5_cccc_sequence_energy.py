"""Tests of the research sequence controller (G-1), the energy instrument (G-2, G-4) and D2-9 (polynomial field) of R5-V CCCC.

Criteria, no tolerance taken from observations: exact rational identities (definition of Pi_h^C = D2 node-by-node route; VOK-2 energy
identity on every half-step; closed form of the D2 operator on the polynomial; pairwise summation of integers), bitwise repeatability
on the same platform, statuses for stops and refused certificates. Numbers are OBSERVATION.
"""

import inspect
import subprocess
import sys
from fractions import Fraction

import numpy as np
import pytest

from kkvim_reference import r5_cccc_d2 as d2
from kkvim_reference import r5_cccc_d2_poly as dp
from kkvim_reference import r5_cccc_energy as en
from kkvim_reference import r5_cccc_halfstep as hs
from kkvim_reference import r5_cccc_sequence as sq


def test_modules_do_not_import_the_main_solver():
    code = ("import sys, kkvim_reference.r5_cccc_sequence, kkvim_reference.r5_cccc_energy, kkvim_reference.r5_cccc_d2_poly; "
            "print(any(k == 'kkvim' or k.startswith('kkvim.') for k in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    assert out == "False"


# --- G-2: energy ------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("lx, nx, ly, ny, D, q", [(1.0, 3, 1.0, 3, 1.0, 1.0), (1.0, 6, 1.7, 5, 2.5, 0.7), (0.37, 7, 1.13, 9, 0.3, -2.0),
                                                  (1.3, 10, 0.7, 4, 1.0, 0.0)])
def test_energy_definition_equals_the_d2_nodewise_route(lx, nx, ly, ny, D, q):
    """a_h(v, w) by the definition of Pi_h^C (boundary rows 2 f_1/h^2, trapezoid weights) equals D h_x h_y v^T A w with A from the
    node-by-node D2 parts with explicit ghost nodes EXACTLY; symmetry; Pi = a/2 - l; the polynomial field included."""
    g = en.Grid(lx, nx, ly, ny, D, q)
    rng = np.random.default_rng(nx * 10 + ny)
    poly = np.array([float(x) for x in dp.field_values(lx, nx, ly, ny)]).reshape(nx - 1, ny - 1)
    fields = [rng.standard_normal((nx - 1, ny - 1)), np.ones((nx - 1, ny - 1)), poly, np.zeros((nx - 1, ny - 1))]
    for v in fields:
        for w in fields[:2]:
            assert en.bilinear_exact(g, v, w) == en.bilinear_d2_exact(g, v, w) == en.bilinear_exact(g, w, v)
        assert en.pi_exact(g, v) == en.bilinear_exact(g, v, v) / 2 - en.load_exact(g, v)
    assert en.pi_exact(g, fields[3]) == 0 and en.norm_a_sq_exact(g, fields[3]) == 0


def test_energy_gradient_is_the_d2_residual():
    """The exact gradient of Pi_h^C at x in direction v: a_h(x, v) - l_h(v) = -D h_x h_y r^T v with r = q/D - A x the exact D2
    residual (Proposition D). Links the energy instrument to D2 EXACTLY (here at the refined D2 solution)."""
    p = d2.D2Problem(1.0, 1.7, 6, 5, D=2.5, q=0.7)
    g = en.Grid(1.0, 6, 1.7, 5, 2.5, 0.7)
    x = d2.solve(p).x
    r = d2.residual_exact(p, x)
    v = np.random.default_rng(0).standard_normal((5, 4))
    lhs = en.bilinear_exact(g, x.reshape(5, 4), v) - en.load_exact(g, v)
    rhs = -Fraction(2.5) * g.hx * g.hy * sum(a * Fraction(float(b)) for a, b in zip(r, v.reshape(-1)))
    assert lhs == rhs


def test_pairwise_sum_depth_and_double_paths():
    """pairwise_sum adds integers exactly; the double Pi and Q paths exist (their errors are OBSERVATION); model R^C split."""
    for M in (1, 2, 3, 7, 8, 9, 1000):
        assert en.pairwise_sum(np.ones(M)) == M
    g = en.Grid(1.0, 8, 1.3, 7, 1.0, 1.0)
    w = np.random.default_rng(2).standard_normal((7, 6))
    assert isinstance(en.pi_double(g, w), float) and isinstance(en.q_double(g, w), float)
    R = en.model_R(g, w)
    assert R["ceil_log2_M"] == int(np.ceil(np.log2(g.M))) and R["R_base_up"] > 0 and R["R_per_cR_up"] > 0
    assert en.C_R_COUNT == 22  # first-order count of this path (module docstring), not an owner's value


def test_grid_input_checks():
    for bad in ({"lx": 0.0}, {"D": -1.0}, {"nx": 2}, {"ny": 4.0}, {"q": float("nan")}, {"lx": True}):
        args = {"lx": 1.0, "nx": 4, "ly": 1.0, "ny": 4, "D": 1.0, "q": 1.0}
        args.update(bad)
        with pytest.raises(ValueError):
            en.Grid(**args)
    with pytest.raises(ValueError):
        en.pi_exact(en.Grid(1.0, 4, 1.0, 4), np.ones((2, 3)))


# --- G-1: sequences -----------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("N", [1, 2, 3])
@pytest.mark.parametrize("first", ["Y", "X"])
def test_sequence_eight_half_steps_with_exact_energy_identity(N, first):
    """s = 1..8, both orientations: every half-step certified (Q1b), the VOK-2 identity holds EXACTLY on the exact product fields;
    statuses OK; N >= 2 RESEARCH_ONLY with rank decision UNRESOLVED; N = 1 normalised (I-4)."""
    r = sq.run_sequence(sq.SequenceConfig(1.0, 10, 1.3, 9, N, s_max=8, first=first))
    assert r.status.startswith("COMPLETED") and len(r.steps) == 8
    dirs = [st["direction"] for st in r.steps]
    assert dirs[0] == first and all(a != b for a, b in zip(dirs, dirs[1:]))
    for st in r.steps:
        assert st["status"] == "OK" and st["certificate"].startswith("CERTIFIED")
        assert st["vok2_identity_exact"] is True
        if N == 1:
            assert "normalisation_c" in st and st["run_class"].startswith("N = 1")
        else:
            assert st["run_class"].startswith("RESEARCH_ONLY") and st["rank_decision"].startswith("UNRESOLVED")
            assert st["marking"].startswith("PROVISIONAL") and 0 < st["rho_N"] <= 1
    assert r.X.shape == (9, N) and r.Y.shape == (8, N)


@pytest.mark.parametrize("s_max", [1, 2, 3])
def test_one_two_and_three_half_steps(s_max):
    r = sq.run_sequence(sq.SequenceConfig(1.0, 8, 1.0, 7, 2, s_max=s_max))
    assert len(r.steps) == s_max and all(st["vok2_identity_exact"] for st in r.steps)


def test_zero_load_stops_with_a_status():
    """q = 0: the first half-step gives 0 (energy 0, identity exact); N = 1 stops (ZERO_FIELD), N = 2 stops at S-1."""
    one = sq.run_sequence(sq.SequenceConfig(1.0, 8, 1.0, 7, 1, q=0.0))
    assert one.status.startswith("STOP: ZERO_FIELD") and one.steps[0]["vok2_identity_exact"] is True
    assert one.steps[0]["P_exact_s"] == 0 and one.steps[0]["eta_qa"] == sq.UNRESOLVED
    two = sq.run_sequence(sq.SequenceConfig(1.0, 8, 1.0, 7, 2, q=0.0))
    assert two.status.startswith("STOP: S-1") and len(two.steps) == 1


def test_orthonormal_and_near_dependent_start_sets():
    """Orthonormal start and near-dependent start (G_2 = G_1 + 1e-6 G_2): both run, RESEARCH_ONLY, identity exact."""
    nx, h = 10, 1.0 / 10
    T = hs.start_set(nx, 2)
    Q, _ = np.linalg.qr(np.sqrt(h) * T)
    near = T.copy()
    near[:, 1] = T[:, 0] + 1e-6 * T[:, 1]
    for start in (Q / np.sqrt(h), near):
        r = sq.run_sequence(sq.SequenceConfig(1.0, nx, 1.0, 9, 2, s_max=4, start=start))
        assert r.status.startswith("COMPLETED") and all(st["vok2_identity_exact"] for st in r.steps)
        assert all(st["run_class"].startswith("RESEARCH_ONLY") for st in r.steps)


def test_without_qr_is_research_only_with_diagnostics():
    r = sq.run_sequence(sq.SequenceConfig(1.0, 10, 1.3, 9, 3, s_max=4, qr=False))
    for st in r.steps:
        assert st["run_class"] == "RESEARCH_ONLY: no QR" and "rho_N" in st and "cond_A_after_QR" not in st
        assert st["vok2_identity_exact"] is True


def test_repeatability_bitwise():
    a = sq.journal(sq.run_sequence(sq.SequenceConfig(1.0, 9, 1.2, 8, 2, s_max=6)))
    b = sq.journal(sq.run_sequence(sq.SequenceConfig(1.0, 9, 1.2, 8, 2, s_max=6)))
    assert a == b


def test_refused_certificate_keeps_a_diagnostic_status(monkeypatch):
    """If the Q1b certificate cannot be given, the half-step is CERT_UNRESOLVED, its bounds UNRESOLVED, B_omega uses route (b) and
    the sequence continues; no PASS is produced."""
    monkeypatch.setattr(hs, "certify_lambda_min", lambda *a, **k: None)
    r = sq.run_sequence(sq.SequenceConfig(1.0, 8, 1.0, 7, 1, s_max=3))
    assert r.status.startswith("COMPLETED")
    for st in r.steps:
        assert st["certificate"].startswith("CERT_UNRESOLVED") and st["set_bound_2"] == sq.UNRESOLVED
        assert st["B_omega_route"].startswith("VOK-7 (b)") and st["vok2_identity_exact"] is True
    assert not any("PASS" in str(v) for st in r.steps for v in st.values())


def test_energy_decomposition_and_model_terms_are_reported():
    """The computed check splits into solution (-gamma), formation and assembly parts; model terms are present and labelled MODEL;
    no c_R, K_E are applied."""
    st = sq.run_sequence(sq.SequenceConfig(1.0, 8, 1.3, 7, 2, s_max=2)).steps[1]
    for key in ("defect_computed", "part_solution", "part_formation", "part_assembly", "gamma_double", "m_s_numerator",
                "R_base_up", "R_per_cR_up", "R_f_up", "B_omega_up", "eps_a_C_s_up"):
        assert isinstance(st[key], float)
    assert st["model_status"].startswith("MODEL")


def test_config_checks():
    for bad in ({"first": "Z"}, {"s_max": 9}, {"s_max": 0}, {"N": 4}):
        args = {"lx": 1.0, "nx": 8, "ly": 1.0, "ny": 7, "N": 1}
        args.update(bad)
        with pytest.raises(ValueError):
            sq.SequenceConfig(**args)


# --- D2-9 -------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("lx, nx, ly, ny", [(1.0, 8, 1.0, 8), (0.37, 6, 1.13, 9), (1.0, 3, 2.0, 5), (1.3, 12, 0.7, 7)])
def test_d2_9_polynomial_operator_closed_form_equals_the_nodewise_d2(lx, nx, ly, ny):
    """D2-9: w = x^2 (a - x)^2 y^2 (b - y)^2; the node-by-node D2 operator (integer parts with ghost nodes, exact scales) applied to
    the exact nodal values equals the independent closed form (delta4 = 24 inside, 24 - 4a/h at the boundary rows; delta2 = X'' +
    2h^2) EXACTLY; the right-hand side is exact rational."""
    w = dp.field_values(lx, nx, ly, ny)
    prob = d2.D2Problem(lx, ly, nx, ny)
    Aw = [Fraction(0)] * len(w)
    for z, s in zip(d2.nodewise_parts(nx, ny), prob.scales()):
        z = z.tocsr()
        for r in range(len(w)):
            Aw[r] += s * sum((int(z.data[k]) * w[z.indices[k]] for k in range(z.indptr[r], z.indptr[r + 1])), Fraction(0))
    assert Aw == dp.operator_closed(lx, nx, ly, ny)


def test_d2_9_closed_form_is_not_built_from_d2():
    """Non-tautology: the closed-form functions do not call r5_cccc_d2."""
    for fn in (dp.nodal_values, dp.delta2_closed, dp.delta4_closed, dp.operator_closed, dp.field_values):
        assert "d2." not in inspect.getsource(fn)


def test_d2_9_solution_with_the_polynomial_load_is_recorded():
    """Second part of D2-9: D2 LU solve of A x = A w refined with the exact residual; the exact error is recorded (OBSERVATION,
    plan 1.4 has no accepted threshold)."""
    errs = dp.solve_with_polynomial_load(0.37, 6, 1.13, 9)["relative_max_error"]
    assert len(errs) == 5 and all(isinstance(e, float) for e in errs)
