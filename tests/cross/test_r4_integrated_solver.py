"""QA integration check of the merged coupled solver N = 2-3 (main 36f8fbb: PR #19, #20) - R4, G2, G3.3, G4.

The real ``kkvim.coupled_run.run_coupled`` (research mode, D-047 addendum 1) is compared on a representative sample with
- the independent QA control ``kkvim_reference.r4_spectral_control`` (same discrete problem, another algorithm), and
- the stored group-B results ``docs/results/r4_group_b/grid.csv`` and ``master.csv`` (B-PREQUALIFICATION).
Only DISCRETE quantities of the same grid are compared (w_h, Pi_h, signed centre error of w_h against omega_0, E_inf of
w_h - omega_0); the discrete field is never identified with the untruncated continuous solution.

Tolerances are the existing ones, fixed before this check:
- field and singular values: the first-order model of tests/cross/test_r4_controller_comparison.py (``_model``, Weyl);
- stored CSV values: the written precision (7 significant digits, half a unit of the last digit: 5e-7 relative) plus that
  model; no new threshold is introduced. Every Builder result stays PROVISIONAL / NOT_QUALIFIED.
"""

import csv
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

import kkvim.coupled_run as cr
from kkvim_reference.levy import levy_series_w_grid
from kkvim_reference.r4_spectral_control import SpectralGrid, run_control

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "docs" / "results" / "r4_group_b"
_spec = importlib.util.spec_from_file_location("r4cmp", Path(__file__).with_name("test_r4_controller_comparison.py"))
r4cmp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(r4cmp)

U = 2.0**-53
CSV_REL = 5e-7  # written precision of the group-B CSV files (format .6e)
RATIOS = {"1/5": 0.2, "1": 1.0, "5": 5.0}
S_CHECK = (1, 2, 3, 10)
CASES = [(N, lab, (n, n), "I") for N in (2, 3) for lab in RATIOS for n in (40, 80, 160)] + \
        [(N, "5", (40, 200), "II") for N in (2, 3)]


def _rows(name):
    with open(RESULTS / name, encoding="utf-8") as f:
        return list(csv.DictReader(f))


GRID = {(int(t["N"]), t["r_label"], int(t["nx"]), int(t["ny"]), t["family"], int(t["s"])): t for t in _rows("grid.csv")}
MASTER = {(int(t["N"]), t["r"], int(t["n_x"]), int(t["n_y"]), t["family"], int(t["s"])): t for t in _rows("master.csv")}


def builder_runs(N, r, nx, ny, transform="fft"):
    """run_coupled with half_steps = s for every checked s (the sequence is deterministic: same prefix)."""
    return {s: cr.run_coupled(1.0, r, 1.0, 1.0, nx, ny, N, transform=transform, half_steps=s, research_mode=True, stop_on_stabilization=False)
            for s in S_CHECK}


@pytest.mark.parametrize("N, lab, grid, family", CASES, ids=lambda v: str(v))
def test_merged_solver_against_the_qa_control_and_the_group_b_tables(N, lab, grid, family):
    r = RATIOS[lab]
    nx, ny = grid
    runs = builder_runs(N, r, nx, ny)
    qa = run_control(SpectralGrid(1.0, r, nx, ny), N, max(S_CHECK))
    hx, hy = 1.0 / nx, r / ny
    W0 = levy_series_w_grid(np.arange(nx + 1) * hx, np.arange(ny + 1) * hy, a=1.0, b=r, q=1.0, D=1.0)[1:-1, 1:-1]
    w0c = W0[nx // 2 - 1, ny // 2 - 1]
    for s, run in runs.items():
        # statuses, sequence, N kept, marks
        assert run.status == cr.RESEARCH_LIMIT_REACHED and not run.converged and run.mark == cr.MARK
        assert len(run.steps) == s and run.x.shape[1] == run.y.shape[1] == N
        for lg in run.steps:
            assert lg.mark == cr.MARK and lg.direction == ("Y" if lg.s % 2 else "X") and lg.k == lg.s // 2
            assert lg.qr_done == (lg.s < s) and lg.guards is not None and lg.guards.passed
        # field against the QA control (existing first-order model)
        W, Wq = np.asarray(run.x) @ np.asarray(run.y).T, qa[s - 1].W  # interior nodes (run.field() includes the edges)
        # run.field() forms the same products on zero-padded arrays (another BLAS shape): not bitwise (AGENTS.md, 4);
        # two evaluations of a dot product of N terms differ by <= 2 gamma_N |X| |Y|^T elementwise
        gamma = 2 * N * U / (1 - N * U)
        assert np.all(np.abs(run.field()[1:-1, 1:-1] - W) <= 2 * gamma * (np.abs(run.x) @ np.abs(run.y).T))
        eps = r4cmp._model(run.steps, qa, s, nx, ny)
        assert np.max(np.abs(W - Wq)) <= eps * np.max(np.abs(Wq)), (s, eps)
        # singular values / rho_N (Weyl) and QR field preservation (existing bounds)
        lg = run.steps[-1]
        dW2 = np.linalg.norm(W - Wq, 2)
        sb, sq = lg.rank.sigma / math.sqrt(hx * hy), qa[s - 1].sigma
        assert np.all(np.abs(sb - sq) <= dW2 + 8 * U * sq[0] * max(1.0, lg.rank.kappa_rep))
        for prev in run.steps[:-1]:
            assert prev.qr_field_error <= 8 * U * math.sqrt(N) * prev.rank.kappa_rep * N
        # stored group-B results of the same grid (QA discrete control, B-PREQUALIFICATION)
        g = GRID[(N, lab, nx, ny, family, s)]
        m = MASTER[(N, lab, nx, ny, family, s)]
        model_c = eps * np.max(np.abs(Wq)) / abs(w0c)
        d_total = (W[nx // 2 - 1, ny // 2 - 1] - w0c) / w0c
        for stored in (float(g["d_total_c"]), float(m["delta_c_total"])):
            assert abs(d_total - stored) <= CSV_REL * abs(stored) + model_c, (s, d_total, stored)
        e_inf_total = float(np.max(np.abs(W - W0))) / float(np.max(np.abs(W0)))
        assert abs(e_inf_total - float(g["E_inf_total"])) <= CSV_REL * float(g["E_inf_total"]) + eps
        assert abs(lg.energy.P - float(g["Pi_h_w"])) <= CSV_REL * abs(float(g["Pi_h_w"])) + 2 * lg.energy.R


def test_dense_and_fft_on_a_common_configuration():
    """Both modal transforms of the merged solver against each other and against the QA control (N = 3, r = 1, n = 80)."""
    qa = run_control(SpectralGrid(1.0, 1.0, 80, 80), 3, 10)
    dense, fft = builder_runs(3, 1.0, 80, 80, "dense"), builder_runs(3, 1.0, 80, 80, "fft")
    for s in S_CHECK:
        eps = r4cmp._model(dense[s].steps, qa, s, 80, 80)
        Wd, Wf = dense[s].field()[1:-1, 1:-1], fft[s].field()[1:-1, 1:-1]
        assert np.max(np.abs(Wd - Wf)) <= 2 * eps * np.max(np.abs(Wd))
        assert dense[s].status == fft[s].status == cr.RESEARCH_LIMIT_REACHED


def test_statuses_after_integration():
    """C-1 fixed (guards before the half-step limit, also on the last half-step); the normative mode never continues
    through a QR and never declares convergence; N = 1 is not taken over (R2/R3 path)."""
    import pytest as _pt

    mp = _pt.MonkeyPatch()
    try:
        mp.setattr(cr, "K_RESEARCH", 10**13)  # test only: makes rho_N <= K u kappa_rep on the last half-step
        run = cr.run_coupled(1.0, 1.0, 1.0, 1.0, 40, 40, 3, transform="fft", half_steps=1, research_mode=True)
        assert run.status == cr.PROVISIONAL_RANK_GUARD_STOP and not run.steps[-1].qr_done
    finally:
        mp.undo()
    run = cr.run_coupled(1.0, 1.0, 1.0, 1.0, 40, 40, 2, transform="fft", half_steps=5)
    assert run.status == cr.NOT_QUALIFIED and run.s == 1 and not run.converged and not run.steps[0].qr_done
    run = cr.run_coupled(1.0, 1.0, 1.0, 1.0, 40, 40, 2, transform="fft", half_steps=1)
    assert run.status == cr.HALF_STEP_LIMIT_REACHED and not run.converged
    assert cr.run_coupled(1.0, 1.0, 1.0, 1.0, 40, 40, 1, transform="fft", half_steps=2).status == "ERROR_INPUT"
    assert not hasattr(cr, "CONVERGED")
