"""Tests of the energy control on B-C (C-3+) solutions (scripts/r5_cccc_q4_energy_bc_audit.py): exact energies of double fields, the
K_E interval of the test form and its exact enclosure, the paired B-C input (bitwise reproduction of the Q3 state after the
prototype's QR), only C replaced by C-3+, exact gamma of the B-C solution by two paths, B_omega by VOK-7 (b), reproduction of the Q3
R-AC denominators, statuses of failed inputs, no change of the used modules and data, no import of the working kkvim, deterministic
CI profile. QA research tool."""

import ast
import csv
import importlib.util
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from kkvim_reference import r5_cccc_bc as bc
from kkvim_reference import r5_cccc_energy as en
from kkvim_reference import r5_cccc_halfstep as hs

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "r5_cccc_q4_energy_bc_audit.py"
spec = importlib.util.spec_from_file_location("r5_cccc_q4_energy_bc_audit", SCRIPT)
eb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eb)


@pytest.fixture(scope="module")
def ci(tmp_path_factory):
    """Small Q3-format sequences (Q3 seq_job, N = 1-3, both first directions) and their stored arrays."""
    rows, arrays, _ = eb.ci_specs_rows(tmp_path_factory.mktemp("ci"))
    ga = eb._ga()
    groups = {}
    for r in rows:
        groups.setdefault(ga.seq_key(r), []).append(r)
    seqs = []
    for rs in groups.values():
        sp = ga._ci_spec(rs[0])
        z = np.load(arrays / ga._q3().seq_file(sp))
        seqs.append((sp, rs, ga._steps_from_npz(z, sp)))
    return seqs


# --- exact helpers ----------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("nx, ny, ly", [(5, 4, 1.3), (7, 9, 0.6), (4, 6, 1.0)])
def test_exact_energies_of_double_fields(nx, ny, ly):
    rng = np.random.default_rng(nx * ny)
    g = en.Grid(1.0, nx, ly, ny)
    Wp = rng.standard_normal((nx - 1, ny - 1)) * 2.0 ** rng.integers(-8, 8, size=(nx - 1, ny - 1))
    Ws = rng.standard_normal((nx - 1, ny - 1))
    Pp, Ps, Qd = eb.energies_fields(1.0, nx, ly, ny, 1.0, 1.0, Wp, Ws)
    assert Pp == en.pi_exact(g, Wp) and Ps == en.pi_exact(g, Ws)
    diff = [[Fraction(float(a)) - Fraction(float(b)) for a, b in zip(ra, rb)] for ra, rb in zip(Wp, Ws)]
    assert Qd == en.norm_a_sq_exact(g, diff) / 2


def _f(Q, a, R, K):
    return Q - a * K - R * K * K


@pytest.mark.parametrize("dP, Q, a, R", [(Fraction(3, 2), Fraction(3, 2), Fraction(1, 10**20), Fraction(1, 10**16)),
                                          (Fraction(1, 10**6), Fraction(2, 10**6), Fraction(1, 10**9), Fraction(1, 10**8)),
                                          (Fraction(-5, 10**12), Fraction(1, 10**11), Fraction(0), Fraction(1, 10**10)),
                                          (Fraction(7, 3), Fraction(7, 3) + Fraction(1, 10**14), Fraction(1, 10**15),
                                           Fraction(1, 10**17))])
def test_k_interval_enclosure_and_class(dP, Q, a, R):
    k = eb.k_interval(dP, Q, a, R)
    lo, hi = Fraction(k["K_max_down"]), Fraction(k["K_max_up"])
    assert k["K_max_bracket_ok"] and _f(Q, a, R, lo) >= 0 >= _f(Q, a, R, hi)
    assert k["K_low"] == max(max(Fraction(0), abs(dP - Q) - a), max(Fraction(0), -dP - a)) / R
    if k["K_class"] == "INTERVAL":  # at K_E = K_low the test is resolved and E1/E2 hold, exactly
        K = k["K_low"]
        tau = a + K * R
        assert abs(dP - Q) <= tau and dP >= -tau and Q >= K * tau
    if k["K_class"] == "EMPTY":  # at K_E = K_max (down): resolved, but E2 or E1 violated
        K = lo
        tau = a + K * R
        assert Q >= K * tau and (abs(dP - Q) > tau or dP < -tau)


def test_k_interval_empty_and_not_applicable():
    k = eb.k_interval(Fraction(-1, 10**3), Fraction(1, 10**9), Fraction(0), Fraction(1, 10**4))
    assert k["K_class"] == "EMPTY"
    assert eb.k_interval(Fraction(1), Fraction(1), Fraction(0), Fraction(0))["K_class"] == "NOT_APPLICABLE"
    assert eb.k_interval(Fraction(0), Fraction(0), Fraction(0), Fraction(1))["K_class"] == "NOT_APPLICABLE"


def test_directed_ratio_is_verified():
    lo, hi = eb.directed_ratio(Fraction(1), Fraction(3))
    assert Fraction(lo) < Fraction(1, 3) < Fraction(hi)
    assert eb.directed_ratio(Fraction(1), Fraction(4)) == (0.25, 0.25)


# --- paired B-C input ----------------------------------------------------------------------------------------------------------

def test_prototype_qr_reproduces_the_q3_state_bitwise(ci):
    seen = 0
    for sp, _, steps in ci:
        for i, st in enumerate(steps):
            p_in, comp, kind = eb.bc_input(sp, steps, i)
            rec = bc.run_halfstep(p_in, qr=st["given"].shape[1] >= 2)
            Cp = comp @ rec.R.T if rec.R is not None else comp
            if st["given"].shape[1] >= 2 and st["s"] >= 2:
                assert np.array_equal(rec.solved.given, st["given"]) and np.array_equal(Cp, st["companion"]), kind
                seen += 1
            elif st["s"] == 1:
                assert not np.any(Cp) and not np.any(st["companion"])  # previous field = 0 exactly
            else:
                assert np.array_equal(rec.solved.given, st["given"]) and np.array_equal(Cp, st["companion"])
    assert seen >= 6


def test_only_C_is_replaced_by_c3plus(ci):
    sp, _, steps = next(c for c in ci if c[0]["N"] == 3)
    p_in, _, _ = eb.bc_input(sp, steps, 2)
    solved = bc.run_halfstep(p_in, qr=True).solved
    coeffs = hs.exact_coefficients(solved)
    res = eb.solve_c3plus(solved, coeffs)
    assert res["only_C_replaced"] and res["c3plus_theorem"] == "PASS" and res["c3plus_C_spd_exact"]
    C = eb._ca().c3plus(solved)["C"]
    with eb._ca().replaced_c(C):
        ref = bc.solve_bc(solved).s
    assert np.array_equal(res["_sol"].s, ref)
    assert bc.bc_coefficients.__name__ == "bc_coefficients"  # restored


def test_gamma_of_the_bc_solution_two_paths_and_vok2(ci):
    ga, gi = eb._ga(), eb._gi()
    sp, _, steps = next(c for c in ci if c[0]["N"] == 2)
    for i in (0, 1, 3):
        st = steps[i]
        p_in, comp, _ = eb.bc_input(sp, steps, i)
        rec = bc.run_halfstep(p_in, qr=True)
        sol = eb.solve_c3plus(rec.solved, hs.exact_coefficients(rec.solved))["_sol"]
        Cp = comp @ rec.R.T
        V = sol.s.reshape(rec.solved.N, -1).T
        g = gi.gamma_b1(sp["lx"], sp["nx"], sp["ly"], sp["ny"], st["direction"], rec.solved.given, Cp, V)
        Xp, Yp = eb.orient(st["direction"], rec.solved.given, Cp)
        Xs, Ys = eb.orient(st["direction"], rec.solved.given, V)
        Pp, Ps, Q = ga.energies_path_b(sp["lx"], sp["nx"], sp["ly"], sp["ny"], 1.0, 1.0, Xp, Yp, Xs, Ys)
        assert Pp - Ps == Q - g  # VOK-2 for the B-C arrays, exactly
        # B-C without refinement: |gamma| differs from the refined R-AC one (not a copy of the Q3 value)
        g_ac = gi.gamma_b1(sp["lx"], sp["nx"], sp["ly"], sp["ny"], st["direction"], st["given"], st["companion"], st["V"])
        assert g != g_ac


def test_b_omega_route_b_for_the_bc_field_and_q3_rule_reproduced(ci):
    sp, rows, steps = next(c for c in ci if c[0]["N"] == 1)
    g = en.Grid(sp["lx"], sp["nx"], sp["ly"], sp["ny"])
    st = steps[0]
    mt = eb.model_terms(g, *st["prev"], *st["now"])
    assert mt["B_omega_route"] == "VOK-7 (b) + formation"
    row = {int(r["j_s"]): r for r in rows}[1]
    p = hs.HalfStep(sp["lx"], sp["nx"], sp["ly"], sp["ny"], st["direction"], st["given"])
    r = hs.residual_exact(p, st["V"].T.reshape(-1))
    low = float(row["j_lambda_low"])
    cert = en.sqrt_interval(Fraction(p.l_s) / p.n_s)[1] * en.sqrt_interval(sum((x * x for x in r), Fraction(0)))[1] \
        / en.sqrt_interval(Fraction(low))[0]
    mq = eb.model_terms(g, *st["prev"], *st["now"], field_cert=cert)
    assert mq["B_omega_route"] == row["j_B_omega_route"]
    for k in ("R_base_up", "R_per_cR_up", "R_f_up", "B_omega_up"):
        assert hs.decimal_up(mq[k], 17) == row[f"j_{k}"]
    assert eb.C_R == en.C_R_COUNT == 22


def test_failed_input_mapping_is_unresolved_not_replaced(ci):
    sp, rows, steps = next(c for c in ci if c[0]["N"] == 1)
    steps = [dict(st) for st in steps]
    steps[1]["given_consistent"] = False
    row = {int(r["j_s"]): r for r in rows}[2]
    out = eb.analyse_step(sp, steps, 1, row, None, False, False)
    assert out["status"].startswith("UNRESOLVED") and "given_consistent" in out["status"]


def test_numerator_uses_the_exact_gamma(ci):
    sp, rows, steps = next(c for c in ci if c[0]["N"] == 1)
    row = {int(r["j_s"]): r for r in rows}[1]
    out = eb.analyse_step(sp, steps, 0, row, None, True, True)
    assert out["status"] == "COMPLETE"
    st = steps[0]
    p_in, comp, _ = eb.bc_input(sp, steps, 0)
    rec = bc.run_halfstep(p_in, qr=False)
    V = eb.solve_c3plus(rec.solved, hs.exact_coefficients(rec.solved))["_sol"].s.reshape(1, -1).T
    g = eb._gi().gamma_b1(sp["lx"], sp["nx"], sp["ly"], sp["ny"], st["direction"], rec.solved.given, comp, V)
    num = abs(Fraction(out["P_hat_prev"]) - Fraction(out["P_hat_s"]) - Fraction(out["Q_hat"]) + g)
    assert float(num) == out["numerator"]
    assert Fraction(out["gamma_down"]) <= g <= Fraction(out["gamma_up"])


# --- protection, provenance, reproducibility ------------------------------------------------------------------------------------

def test_used_modules_and_data_unchanged():
    for rel in ("src/kkvim_reference", "src/kkvim", "docs/results/r5_cccc_q3_prequalification", "docs/results/r5_cccc_q4_gamma_audit",
                "docs/results/r5_cccc_q4_gamma_intervals", "docs/results/r5_cccc_q4_c_assembly_e4",
                "scripts/r5_cccc_q4_gamma_audit.py", "scripts/r5_cccc_q4_gamma_interval_audit.py",
                "scripts/r5_cccc_q4_c_assembly_e4_audit.py", "scripts/r5_cccc_q3_prequalification.py"):
        out = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", rel], cwd=ROOT)
        assert out.returncode == 0, rel


def test_no_import_of_the_working_kkvim():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    names = [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
    names += [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    assert not [m for m in names if m == "kkvim" or m.startswith("kkvim.")]


def test_ci_profile_is_deterministic(tmp_path):
    outs = []
    for k in range(2):
        out = tmp_path / f"run{k}"
        res = subprocess.run([sys.executable, str(SCRIPT), "run", "--profile", "ci", "--out", str(out)], cwd=ROOT,
                             capture_output=True, text=True, timeout=900)
        assert res.returncode == 0, res.stderr[-2000:]
        outs.append((out / "rows.csv").read_bytes())
    assert outs[0] == outs[1]
    rows = list(csv.DictReader(outs[0].decode("utf-8").splitlines()))
    assert len(rows) == 16 and all(r["status"] == "COMPLETE" for r in rows)
    for r in rows:
        assert r["gamma_inclusion_exact"] == r["gamma_equals_path_A"] == r["vok2_identity_energy_path"] == "True"
        assert r["B_omega_route"] == "VOK-7 (b) + formation" and r["control_q3_R_reproduced"] == "True"
        assert r["c3plus_theorem"] == "PASS" and r["only_C_replaced"] == "True"
