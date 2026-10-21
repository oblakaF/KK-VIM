"""R4 D-048: independent QA qualification K-1...K-9 of the Builder implementation (plan e4e1109, fixed before measurements).

Object under test: kkvim (Builder) - only called, never used to build reference values. References: kkvim_reference,
own QA code (tests/cross/test_r4_controller_comparison.py: harness, qa_energy, qa_coefficients, qa_apply_K, _model),
decimal / mpmath (high precision; mpmath only in a temporary environment, outside the mandatory CI).

Reproduction (repository root):
    uv run --with mpmath python scripts/r4_d048_qualification.py run            # K-2...K-8 tables (+ K-6 high precision)
    PYTHONPATH=<src of version> uv run python scripts/r4_d048_qualification.py k1-dump OUT.npz   # K-1, per version
    uv run python scripts/r4_d048_qualification.py k1-compare OLD.npz NEW.npz OUT.csv
    uv run python scripts/r4_d048_qualification.py k1-reference     # stored n = 40 fields for the CI part of K-1
Outputs: docs/results/r4_d048_qualification/ (--out).
"""

import argparse
import csv
import importlib.util
import json
import math
import platform
import subprocess
import sys
import time
from decimal import Decimal, getcontext
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
U = 2.0**-53
R3 = {"1/5": 0.2, "1": 1.0, "5": 5.0}
R10 = {"1/5": 0.2, "1/3": 1 / 3, "1/2": 0.5, "2/3": 2 / 3, "1": 1.0, "7/5": 1.4, "3/2": 1.5, "2": 2.0, "3": 3.0, "5": 5.0}
FAMILY_II = {"1/5": (1600, 320), "1/3": (960, 320), "1/2": (640, 320), "2": (320, 640), "3": (320, 960), "5": (320, 1600)}
TOL_INF, TOL_E, S_MIN, S_MAX, ETA_MAX = 1e-10, 1e-8, 3, 100, 1e-12  # D-048 values, re-entered independently


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def write_csv(path, rows):
    keys = []
    for row in rows:
        keys += [k for k in row if k not in keys]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
        w.writeheader()
        for row in rows:
            w.writerow({k: (f"{v:.6e}" if isinstance(v, float) else v) for k, v in row.items()})


# ------------------------------------------------------------------------------------------------------------------
# high precision without mpmath (decimal, 50+ digits) - usable in CI
# ------------------------------------------------------------------------------------------------------------------

def dec_pi():
    """pi to the current decimal precision (recipe of the Python decimal documentation)."""
    getcontext().prec += 2
    three = Decimal(3)
    lasts, t, s, n, na, d, da = 0, three, 3, 1, 0, 0, 24
    while s != lasts:
        lasts = s
        n, na = n + na, na + 8
        d, da = d + da, da + 32
        t = (t * n) / d
        s += t
    getcontext().prec -= 2
    return +s


def dec_sin(x):
    """sin of a Decimal x in [0, pi/2] by the Taylor series to the current precision."""
    getcontext().prec += 2
    i, lasts, s, fact, num, sign = 1, 0, x, 1, x, 1
    while s != lasts:
        lasts = s
        i += 2
        fact *= i * (i - 1)
        num *= x * x
        sign *= -1
        s += num / fact * sign
    getcontext().prec -= 2
    return +s


def builder_sine_arguments(n):
    """The arguments the Builder dense path passes to np.sin, rebuilt with the same expressions as kkvim.coupled.sine_matrix
    (3da624a): m = i k mod 2n, m >= n -> m - n (negative), 2m > n -> n - m; x = m * (pi / n). Returns m, negative, x."""
    i = np.arange(1, n, dtype=np.int64)
    m = np.outer(i, i) % (2 * n)
    negative = m >= n
    m = np.where(negative, m - n, m)
    m = np.where(2 * m > n, n - m, m)
    return m, negative, m * (math.pi / n)


def sin_premise(n_list=(40, 160, 640, 1280), digits=50, use_mpmath=False):
    """K-6.0: np.sin(x) against sin of the same double x (decimal in CI, mpmath locally), in ulp; formation error of x
    against pi m / n."""
    from kkvim.coupled import sine_matrix
    from kkvim.grid import Grid1D
    getcontext().prec = digits
    pi = dec_pi()
    if use_mpmath:
        import mpmath as mp
        mp.mp.dps = digits

        def ref_sin(x):  # sin of the double x itself, converted to Decimal with the same number of digits
            return Decimal(mp.nstr(mp.sin(mp.mpf(x)), digits, min_fixed=-mp.inf, max_fixed=mp.inf))
    else:
        def ref_sin(x):
            return dec_sin(Decimal(x))
    rows = []
    for n in n_list:
        m, negative, x2d = builder_sine_arguments(n)
        v2d = np.sin(x2d)
        rebuilt = math.sqrt(2.0 / n) * np.where(negative, -v2d, v2d)
        rebuilt[m == 0] = 0.0
        same = bool(np.array_equal(rebuilt, sine_matrix(Grid1D(1.0, n))))
        ms = np.unique(m)
        x1d = ms * (math.pi / n)
        v1d = np.sin(x1d)
        # 2D values per m (first occurrence)
        flat_m, flat_v = m.ravel(), v2d.ravel()
        first = {}
        for mm, vv in zip(flat_m.tolist(), flat_v.tolist()):
            first.setdefault(mm, vv)
        same_1d_2d = all(first[int(mm)] == float(vv) for mm, vv in zip(ms, v1d))
        worst_ulp_1d = worst_ulp_2d = 0.0
        worst_form_ulp, worst_form_abs = 0.0, 0.0
        for mm, x, y in zip(ms.tolist(), x1d.tolist(), v1d.tolist()):
            if mm == 0:
                continue
            exact = ref_sin(x)
            ulp = Decimal(math.ulp(y))
            worst_ulp_1d = max(worst_ulp_1d, float(abs(Decimal(y) - exact) / ulp))
            y2 = first[mm]
            worst_ulp_2d = max(worst_ulp_2d, float(abs(Decimal(y2) - exact) / Decimal(math.ulp(y2))))
            form = abs(Decimal(x) - pi * mm / n)
            worst_form_abs = max(worst_form_abs, float(form))
            worst_form_ulp = max(worst_form_ulp, float(form / Decimal(math.ulp(x))))
        rows.append({"n": n, "reference": "mpmath" if use_mpmath else "decimal", "digits": digits,
                     "distinct_arguments": int(ms.size - 1), "arguments_rebuilt_bitwise": same,
                     "same_values_1d_2d": same_1d_2d, "max_ulp_np_sin_1d": worst_ulp_1d, "max_ulp_np_sin_2d": worst_ulp_2d,
                     "premise_le_1ulp": max(worst_ulp_1d, worst_ulp_2d) <= 1.0,
                     "max_formation_err_ulp_of_x": worst_form_ulp, "max_formation_err_abs": worst_form_abs})
    return rows


def platform_record():
    rec = {"python": sys.version.split()[0], "numpy": np.__version__, "os": platform.platform(),
           "machine": platform.machine(), "processor": platform.processor()}
    try:
        from numpy._core._multiarray_umath import __cpu_features__ as feats
        rec["cpu_features_enabled"] = ",".join(k for k, v in feats.items() if v)
    except Exception as exc:  # pragma: no cover - depends on the NumPy build
        rec["cpu_features_enabled"] = f"unavailable: {exc}"
    try:
        from numpy._core._multiarray_umath import __cpu_baseline__, __cpu_dispatch__
        rec["cpu_baseline"] = ",".join(__cpu_baseline__)
        rec["cpu_dispatch"] = ",".join(__cpu_dispatch__)
    except Exception as exc:  # pragma: no cover
        rec["cpu_baseline"] = f"unavailable: {exc}"
    return rec


# ------------------------------------------------------------------------------------------------------------------
# K-1: dumps of a version (run under PYTHONPATH of that version) and comparison
# ------------------------------------------------------------------------------------------------------------------

K1_FIXED = [(N, lab, (n, n)) for N in (2, 3) for lab in R3 for n in (40, 160, 640)] + [(N, "5", (40, 200)) for N in (2, 3)]
K1_S = 14  # covers s_stop = 9..12 of the new version


def k1_dump(out):
    import kkvim
    import kkvim.coupled_run as cr
    from kkvim.controller import run_until_stopped
    from kkvim.rectangle_run import run_rectangle_until_stopped
    arrays, meta = {}, {"kkvim": kkvim.__file__}
    for N, lab, (nx, ny) in K1_FIXED:
        r = R3[lab]
        for s in range(1, K1_S + 1):
            run = cr.run_coupled(1.0, r, 1.0, 1.0, nx, ny, N, transform="fft", half_steps=s, research_mode=True)
            key = f"c_{N}_{lab}_{nx}_{ny}_{s}"
            arrays[key + "_x"], arrays[key + "_y"] = run.x, run.y
            lg = np.array([[st.energy.P, st.rank.rho_N, st.rank.kappa_rep, st.eta_N,
                            st.eps_inf if st.eps_inf is not None else np.nan, st.eps_E if st.eps_E is not None else np.nan]
                           for st in run.steps])
            arrays[key + "_log"] = lg
            meta[key] = [run.status, len(run.steps)]
    for n in (20, 40, 80, 160, 320):
        res = run_until_stopped(1.0, 1.0, 1.0, n, n)
        # half-step log: repr of the float diagnostics (shortest round-trip repr: equal strings <=> equal doubles)
        meta[f"r2_{n}"] = [res.status, len(res.history.records), repr(res.changes), repr(res.energy_checks)]
        for rec in res.history.records:
            arrays[f"r2_{n}_{rec.s}_x"], arrays[f"r2_{n}_{rec.s}_y"] = rec.x, rec.y
    for lab, r in R3.items():
        for nx, ny in ((40, 40), (160, 160), (640, 640)) + (((40, 200),) if lab == "5" else ((200, 40),) if lab == "1/5" else ()):
            res = run_rectangle_until_stopped(1.0, r, 1.0, 1.0, nx, ny)
            meta[f"r3_{lab}_{nx}_{ny}"] = [res.status, len(res.history.records), repr(res.changes), repr(res.energy_checks)]
            for rec in res.history.records:
                arrays[f"r3_{lab}_{nx}_{ny}_{rec.s}_x"], arrays[f"r3_{lab}_{nx}_{ny}_{rec.s}_y"] = rec.x, rec.y
    np.savez(out, **arrays)
    Path(str(out) + ".json").write_text(json.dumps(meta, indent=0), encoding="utf-8")


K1_REFERENCE_KEYS = [(N, lab, s) for N in (2, 3) for lab in R3 for s in (1, 2, 3, 10)]  # order of the stored stack


def k1_reference(out):
    """Stored fields for the cross-platform part of K-1 (CI test): fixed mode, n = 40, W = x y^T at s = 1, 2, 3, 10,
    stacked in the order K1_REFERENCE_KEYS into one uncompressed .npy (no archive formats in git, D-021 hook)."""
    import kkvim.coupled_run as cr
    fields = []
    for N, lab, s in K1_REFERENCE_KEYS:
        run = cr.run_coupled(1.0, R3[lab], 1.0, 1.0, 40, 40, N, transform="fft", half_steps=s, research_mode=True,
                             stop_on_stabilization=False)
        assert len(run.steps) == s
        fields.append(run.x @ run.y.T)
    np.save(out, np.stack(fields))


def k1_compare(old, new, out_csv):
    A, B = np.load(old), np.load(new)
    mo = json.loads(Path(str(old) + ".json").read_text(encoding="utf-8"))
    mn = json.loads(Path(str(new) + ".json").read_text(encoding="utf-8"))
    rows = []
    for N, lab, (nx, ny) in K1_FIXED:
        stop = next((s for s in range(1, K1_S + 1) if mn[f"c_{N}_{lab}_{nx}_{ny}_{s}"][0] == "PROVISIONAL_FIELD_STABILIZED"), None)
        s_star = stop if stop is not None else K1_S
        diff, first = 0, None
        for s in range(1, s_star + 1):
            key = f"c_{N}_{lab}_{nx}_{ny}_{s}"
            for part in ("_x", "_y"):
                if not np.array_equal(A[key + part], B[key + part]):
                    diff += 1
                    first = first or f"s={s}{part}"
            la, lb = A[key + "_log"], B[key + "_log"]
            if not (la.shape == lb.shape and np.array_equal(la, lb, equal_nan=True)):
                diff += 1
                first = first or f"s={s}_log"
        rows.append({"path": "coupled fft research", "N": N, "r": lab, "nx": nx, "ny": ny, "s_star": s_star,
                     "new_status_at_s_star": mn[f"c_{N}_{lab}_{nx}_{ny}_{s_star}"][0],
                     "old_status_at_s_star": mo[f"c_{N}_{lab}_{nx}_{ny}_{s_star}"][0], "differing_items": diff,
                     "first_difference": first or "", "verdict": "PASS" if diff == 0 else "FAIL"})
    for key in sorted(k for k in mo if k.startswith(("r2_", "r3_"))):
        prefix = key + "_"
        ka = sorted(k for k in A.files if k.startswith(prefix))
        kb = sorted(k for k in B.files if k.startswith(prefix))
        diff = int(ka != kb) + sum(0 if np.array_equal(A[k], B[k]) else 1 for k in set(ka) & set(kb)) + int(mo[key] != mn[key])
        rows.append({"path": "N = 1 " + key.split("_")[0].upper(), "N": 1, "r": key, "nx": "", "ny": "", "s_star": mo[key][1],
                     "new_status_at_s_star": mn[key][0], "old_status_at_s_star": mo[key][0], "differing_items": diff,
                     "first_difference": "", "verdict": "PASS" if diff == 0 else "FAIL"})
    write_csv(out_csv, rows)
    return rows


# ------------------------------------------------------------------------------------------------------------------
# K-2 ... K-8
# ------------------------------------------------------------------------------------------------------------------

def qa_eps(W_prev, W, hx, hy):
    """QA's own eps_inf and eps_E (definitions NUMERICAL_SPEC 11.1): max norm and a_h norm by 2D second differences."""
    from kkvim_reference.r4_spectral_control import nodal_laplacian
    d = W - W_prev
    e_inf = float(np.max(np.abs(d)) / np.max(np.abs(W)))
    na = lambda V: math.sqrt(hx * hy * math.fsum((nodal_laplacian(V, hx, hy) ** 2).ravel().tolist()))  # noqa: E731
    return e_inf, na(d) / na(W)


def eps_floor_formula(nx, ny, N):
    g = N * U / (1 - N * U)
    return 10 * U * (max(nx, ny) / math.pi) ** 2 * math.sqrt(N) / (1 - g * math.sqrt(N))


def k2_stopping(cmp_mod):
    import kkvim.coupled_run as cr
    rows, modes = [], []
    groups = [("area", n) for n in (40, 160, 640)] + [("independent n=320", 320), ("research n=1280", 1280)]
    for group, n in groups:
        for N in (2, 3):
            for lab, r in R3.items():
                hx, hy = 1.0 / n, r / n
                auto = cr.run_coupled(1.0, r, 1.0, 1.0, n, n, N, transform="fft", half_steps=S_MAX, research_mode=True)
                fixed = cr.run_coupled(1.0, r, 1.0, 1.0, n, n, N, transform="fft", half_steps=40, research_mode=True,
                                       stop_on_stabilization=False)
                rec = cmp_mod.harness(r, N, n, n, "fft", 40)
                harness_ok = bool(np.array_equal(rec[-1]["new"][0], fixed.x) and np.array_equal(rec[-1]["new"][1], fixed.y))
                floor = eps_floor_formula(n, n, N)
                s_qa, border, prev_ok = None, False, False
                for s in range(2, 41):
                    W_prev = cmp_mod.field(*rec[s - 2]["new"])
                    W = cmp_mod.field(*rec[s - 1]["new"])
                    e_inf, e_E = qa_eps(W_prev, W, hx, hy)
                    ok = e_inf <= TOL_INF and e_E <= TOL_E
                    if s_qa is None and (abs(e_E - TOL_E) <= floor or abs(e_inf - TOL_INF) <= 100 * U):
                        border = True
                    if s_qa is None and ok and prev_ok and s >= S_MIN:
                        s_qa = s
                    prev_ok = ok
                s_stop = len(auto.steps) if auto.status == cr.PROVISIONAL_FIELD_STABILIZED else None
                guards_all = all(st.guards is not None and st.guards.passed for st in auto.steps)
                verdict = ("UNRESOLVED" if border else
                           "PASS" if (s_stop is not None and s_stop == s_qa and s_stop < S_MAX and guards_all) else "FAIL")
                if group.startswith("research"):  # plan K-2.9: n = 1280 is an observation only, no PASS / FAIL
                    verdict = "observation (" + ("agrees" if verdict == "PASS" else verdict.lower()) + ")"
                every = list(auto.steps) + list(fixed.steps)  # K-4.1 and K-5.4 on all half-steps of both runs
                K_u = cr.K_RESEARCH * U
                eps_E_plateau = min(st.eps_E for st in fixed.steps[20:])
                rows.append({"group": group, "N": N, "r": lab, "n": n, "status_auto": auto.status, "s_stop": s_stop,
                             "s_QA": s_qa, "boundary_band": border, "guards_all_passed": guards_all,
                             "eps_floor_impl": auto.parameters.eps_floor, "eps_floor_formula": floor,
                             "floor_rel_diff_over_u": abs(auto.parameters.eps_floor - floor) / (floor * U),
                             "tolE_over_10floor": TOL_E / (10 * floor), "eps_E_plateau_s21_40": eps_E_plateau,
                             "harness_equals_fixed_run": harness_ok,
                             "max_eta_N_all_steps": max(st.eta_N for st in every),
                             "refinements": sum(st.refined for st in every),
                             "min_rho_over_K_u_kappa": min(st.rank.rho_N / (K_u * st.rank.kappa_rep) for st in every),
                             "min_set_sigma_over_K_u": min(st.rank.set_sigma_min / K_u for st in every),
                             "max_kappa_rep": max(st.rank.kappa_rep for st in every),
                             "fixed_status": fixed.status, "verdict": verdict})
                # both research modes on the same configuration (assignment, section 3)
                upto = cr.run_coupled(1.0, r, 1.0, 1.0, n, n, N, transform="fft", half_steps=s_stop or 1, research_mode=True,
                                      stop_on_stabilization=False) if s_stop else None
                same_prefix = bool(s_stop and all(
                    a.energy.P == b.energy.P and a.rank.rho_N == b.rank.rho_N and a.eta_N == b.eta_N
                    and (a.eps_E == b.eps_E) for a, b in zip(auto.steps, fixed.steps[:s_stop])))
                same_field = bool(upto is not None and np.array_equal(upto.x, auto.x) and np.array_equal(upto.y, auto.y))
                after_guards = all(st.guards is not None and st.guards.passed for st in fixed.steps[(s_stop or 0):])
                modes.append({"group": group, "N": N, "r": lab, "n": n, "s_stop_auto": s_stop,
                              "first_stabilization_s_fixed": fixed.first_stabilization_s,
                              "fixed_status": fixed.status, "fixed_len": len(fixed.steps),
                              "logs_bitwise_equal_up_to_s_stop": same_prefix, "field_at_s_stop_bitwise_equal": same_field,
                              "guards_passed_after_s_stop": after_guards,
                              "fixed_never_stabilized_status": fixed.status != cr.PROVISIONAL_FIELD_STABILIZED,
                              "verdict": "PASS" if (s_stop == fixed.first_stabilization_s and fixed.status == cr.RESEARCH_LIMIT_REACHED
                                                    and len(fixed.steps) == 40 and same_prefix and same_field and after_guards)
                              else "FAIL"})
                print("K-2", group, N, lab, flush=True)
    return rows, modes


def k3_energy(cmp_mod):
    import kkvim.coupled_run as cr
    rows = []
    configs = [("I", lab, r, n, n) for lab, r in R10.items() for n in (160, 640, 1280)]
    configs += [("II", lab, R10[lab], nx, ny) for lab, (nx, ny) in FAMILY_II.items()]
    configs += [("I research n=320", lab, r, 320, 320) for lab, r in R10.items()]
    for fam, lab, r, nx, ny in configs:
        for N in (2, 3):
            run = cr.run_coupled(1.0, r, 1.0, 1.0, nx, ny, N, transform="fft", half_steps=4, research_mode=True,
                                 stop_on_stabilization=False)
            rec = cmp_mod.harness(r, N, nx, ny, "fft", 4)
            assert len(run.steps) == len(rec) == 4
            eta_all = max(st.eta_N for st in run.steps)  # K-4.1: s = 1 included
            for st, h in zip(run.steps[1:], rec[1:]):
                e = st.energy
                q = cmp_mod.qa_energy(h, 1.0, r, nx, ny)
                agree = (abs(e.delta_P - q["delta_P"]) <= q["tau"] and abs(e.Q - q["Q"]) <= q["tau"]
                         and abs(e.gamma - q["gamma"]) <= q["tau"] and abs(e.tau - q["tau"]) <= 0.01 * q["tau"]
                         and (e.e1, e.e2) == (q["e1"], q["e2"]))
                rows.append({"family": fam, "r": lab, "nx": nx, "ny": ny, "N": N, "s": st.s, "m": st.m_s,
                             "e1": e.e1, "e2": e.e2, "status": e.status, "independent_energy_agrees": agree,
                             "mandatory_flag": run.energy_s2_mandatory, "s2_unresolved_flag": run.energy_s2_mandatory_unresolved,
                             "eta_N": st.eta_N, "max_eta_N_s1_4": eta_all, "refined": st.refined, "rho_N": st.rank.rho_N, "kappa_rep": st.rank.kappa_rep,
                             "set_sigma_min": st.rank.set_sigma_min, "status_run": run.status})
        print("K-3", fam, lab, nx, ny, flush=True)
    return rows


def k4_eta(cmp_mod):
    """eta_N of the implementation against eta_QA (QA's own operator) on the K-2 area runs, s = 1..12 (>= s_stop)."""
    rows = []
    for n in (40, 160, 640):
        for N in (2, 3):
            for lab, r in R3.items():
                rec = cmp_mod.harness(r, N, n, n, "fft", 12)
                import kkvim.coupled_run as cr
                run = cr.run_coupled(1.0, r, 1.0, 1.0, n, n, N, transform="fft", half_steps=12, research_mode=True,
                                     stop_on_stabilization=False)
                assert len(run.steps) == len(rec) == 12
                worst = 0.0
                for st, h in zip(run.steps, rec):
                    A, B, C, F = cmp_mod.qa_coefficients(np.asarray(h["given"]), h["given_grid"].h)
                    hs = h["solve_grid"].h
                    V = np.asarray(h["step"].solved)
                    rho = F[None, :] - cmp_mod.qa_apply_K(V, hs, A, B, C)
                    norm = lambda M: float(np.max(np.sum(np.abs(M), axis=1)))  # noqa: E731
                    scale = (16 * norm(A) / hs**4 + 8 * norm(B) / hs**2 + norm(C)) * float(np.max(np.abs(V))) + float(np.max(np.abs(F)))
                    worst = max(worst, abs(float(np.max(np.abs(rho))) / scale - st.eta_N))
                rows.append({"n": n, "N": N, "r": lab, "steps": len(run.steps), "max_eta_N": max(st.eta_N for st in run.steps),
                             "max_abs_eta_minus_eta_QA_over_u": worst / U, "refinements": sum(st.refined for st in run.steps)})
    return rows


def k6_high_precision():
    """K-6.2 elements (>= 50 digits, mpmath) and K-6.3 impulses for n = 40, 160, 640, 1280."""
    import mpmath as mp
    from kkvim.coupled import sine_matrix, sine_transform
    from kkvim.grid import Grid1D
    mp.mp.dps = 50
    el, imp = [], []
    for n in (40, 160, 640, 1280):
        g = Grid1D(1.0, n)
        P = sine_matrix(g)
        # exact zeros for j = 0 mod n (mpmath returns ~1e-50 instead of 0 for sin(pi j/n) there)
        tab = [mp.mpf(0) if j % n == 0 else mp.sin(mp.pi * j / n) for j in range(2 * n)]
        sc = mp.sqrt(mp.mpf(2) / n)
        exact = np.empty((n - 1, n - 1))
        worst, viol = 0.0, 0
        for i in range(1, n):
            for k in range(1, n):
                ex = sc * tab[(i * k) % (2 * n)]
                exf = float(ex)
                exact[i - 1, k - 1] = exf
                if exf == 0.0:
                    continue
                err = abs(float(mp.mpf(float(P[i - 1, k - 1])) - ex))
                ratio = err / (U * abs(exf))
                worst = max(worst, ratio)
                if ratio > 7.0:
                    viol += 1
        el.append({"n": n, "nonzero_elements": int(np.count_nonzero(exact)), "max_err_over_u_abs_psi": worst,
                   "violations_7u": viol})
        gam = (n - 1) * U / (1 - (n - 1) * U)
        for name, j in (("e_1", 0), ("e_mid", n // 2 - 1), ("e_last", n - 2)):
            x = np.zeros(n - 1)
            x[j] = 1.0
            ref = [sc * tab[((k + 1) * (j + 1)) % (2 * n)] for k in range(n - 1)]
            yd, yf = sine_transform(g, x, "dense"), sine_transform(g, x, "fft")
            bound = [(7 * U + gam * (1 + 7 * U)) * abs(float(v)) for v in ref]
            ed = [abs(float(mp.mpf(float(yd[k])) - ref[k])) for k in range(n - 1)]
            ef = [abs(float(mp.mpf(float(yf[k])) - ref[k])) for k in range(n - 1)]
            imp.append({"n": n, "impulse": name, "dense_violations": sum(1 for e, b in zip(ed, bound) if e > b),
                        "dense_err2_over_u": math.sqrt(sum(e * e for e in ed)) / U,
                        "fft_err2_over_u": math.sqrt(sum(e * e for e in ef)) / U,
                        "dense_minus_fft_max_over_u": float(np.max(np.abs(yd - yf))) / U})
        print("K-6 HP", n, flush=True)
    return el, imp


def k6_exact(n_list=(40, 160, 640, 1280)):
    """K-6.1: exact zeros, symmetry, sign antisymmetry psi_{i,n-k} = (-1)^(i+1) psi_{i,k}, own reduction (bitwise)."""
    from kkvim.coupled import sine_matrix
    from kkvim.grid import Grid1D
    rows = []
    for n in n_list:
        P = sine_matrix(Grid1D(1.0, n))
        i = np.arange(1, n)
        ik = np.outer(i, i)
        zeros_ok = bool(np.all(P[(ik % n) == 0] == 0.0))
        sym_ok = bool(np.array_equal(P, P.T))
        sign = np.where(i % 2 == 1, 1.0, -1.0)[:, None]  # (-1)^(i+1)
        anti_ok = bool(np.array_equal(P[:, ::-1], sign * P))  # column n-k of row i
        # own reduction (written independently): r = ik mod 2n in [0, 2n); sin(pi r/n) = s * sin(pi t/n), t in [0, n/2]
        r = ik % (2 * n)
        s = np.ones_like(r)
        t = r.copy()
        hi = t >= n
        t[hi] -= n
        s[hi] = -1
        big = 2 * t > n
        t[big] = n - t[big]
        m, negative, _ = builder_sine_arguments(n)
        own_ok = bool(np.array_equal(t, m) and np.array_equal(s < 0, negative))
        rows.append({"n": n, "exact_zeros": zeros_ok, "symmetry": sym_ok, "sign_antisymmetry": anti_ok,
                     "own_reduction_equal": own_ok,
                     "verdict": "PASS" if (zeros_ok and sym_ok and anti_ok and own_ok) else "FAIL"})
    return rows


def k7_a43(n_list=(40, 160, 320, 640, 1280), digits=50):
    """A4-3 per coefficient at s = 1 against closed forms (ChV4-8) evaluated in decimal."""
    from kkvim.coupled import coupled_coefficients, odd_sine_set
    from kkvim.grid import Grid1D
    getcontext().prec = digits
    pi = dec_pi()
    rows = []
    for n in n_list:
        for N in (2, 3):
            g = Grid1D(1.0, n)
            co = coupled_coefficients(g, odd_sine_set(g, N), 1.0)
            h = Decimal(1) / n
            worst = 0.0
            mdiag = {}
            for j in range(N):
                mj = 2 * j + 1
                sn = dec_sin(pi * mj / (2 * n))
                cs = dec_sin(pi / 2 - pi * mj / (2 * n))
                mu = 4 * Decimal(n) ** 2 * sn * sn
                exact = {"A": Decimal("0.5"), "B": Decimal("0.5") * mu, "C": Decimal("0.5") * mu * mu, "F": h * cs / sn}
                for name, ex in exact.items():
                    val = (getattr(co, name)[j] if name == "F" else getattr(co, name)[j, j])
                    worst = max(worst, float(abs(Decimal(float(val)) - ex) / ex))
                    if name != "F":
                        mdiag[name] = max(mdiag.get(name, 0.0), float(ex))
            for name in ("A", "B", "C"):
                M = getattr(co, name)
                off = M - np.diag(np.diag(M))
                worst = max(worst, float(np.max(np.abs(off))) / mdiag[name])
            rows.append({"n": n, "N": N, "max_rel_error_per_coefficient": worst,
                         "area": "D-048.7 (n <= 640)" if n <= 640 else "observation only (n = 1280)",
                         "verdict": ("PASS" if worst <= 1e-13 else "FAIL") if n <= 640 else "observation",
                         "margin": 1e-13 / worst if worst > 0 else math.inf})
    return rows


def k7_a412():
    import kkvim.coupled_run as cr
    rows = []
    for N in (2, 3):
        for lab, r in R3.items():
            for n in (40, 160):
                run = cr.run_coupled(1.0, r, 1.0, 1.0, n, n, N, transform="fft", half_steps=6, research_mode=True,
                                     stop_on_stabilization=False)
                W = run.field()
                m = float(np.max(np.abs(W)))
                refl = max(float(np.max(np.abs(W - W[::-1, :]))), float(np.max(np.abs(W - W[:, ::-1])))) / m
                worst_scale = 0.0
                for a, q, D in ((2.9, 1.0, 1.0), (1.0, 3.7, 1.0), (1.0, 1.0, 0.013), (2.9, -3.7, 0.013)):
                    o = cr.run_coupled(a, r * a, D, q, n, n, N, transform="fft", half_steps=6, research_mode=True,
                                       stop_on_stabilization=False)
                    worst_scale = max(worst_scale, float(np.max(np.abs(o.field() * D / (q * a**4) - W))) / m)
                rows.append({"N": N, "r": lab, "n": n, "s": len(run.steps), "reflection_rel": refl, "scaling_rel": worst_scale,
                             "verdict": "PASS" if (refl <= 1e-12 and worst_scale <= 1e-11 and len(run.steps) == 6) else "FAIL",
                             "margin_reflection": 1e-12 / refl if refl > 0 else math.inf,
                             "margin_scaling": 1e-11 / worst_scale if worst_scale > 0 else math.inf})
    return rows


def run_all(out, hp):
    cmp_mod = _load("r4cmp", ROOT / "tests" / "cross" / "test_r4_controller_comparison.py")
    pq = _load("r4pq", ROOT / "scripts" / "r4_numerical_prequal.py")
    t0 = time.time()
    meta = {"object": "builder/r4-d048-implementation 3da624a", "plan": "e4e1109", **platform_record()}
    k2, modes = k2_stopping(cmp_mod)
    write_csv(out / "k2_stopping.csv", k2)
    write_csv(out / "k2_two_modes.csv", modes)
    k3 = k3_energy(cmp_mod)
    write_csv(out / "k3_energy.csv", k3)
    write_csv(out / "k4_eta.csv", k4_eta(cmp_mod))
    write_csv(out / "k5_rank_constructions.csv", pq.b4_rank())
    write_csv(out / "k6_premise_local.csv", sin_premise(use_mpmath=hp))
    write_csv(out / "k6_exact.csv", k6_exact())
    write_csv(out / "k7_a4_3.csv", k7_a43())
    write_csv(out / "k7_a4_12.csv", k7_a412())
    if hp:
        el, imp = k6_high_precision()
        write_csv(out / "k6_elements.csv", el)
        write_csv(out / "k6_impulses.csv", imp)
    meta["seconds"] = round(time.time() - t0)
    (out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    print("done", meta["seconds"])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", default="docs/results/r4_d048_qualification")
    r.add_argument("--no-hp", action="store_true")
    d = sub.add_parser("k1-dump")
    d.add_argument("out")
    f = sub.add_parser("k1-reference")
    f.add_argument("out", nargs="?", default="docs/results/r4_d048_qualification/k1_reference_fields_n40.npy")
    c = sub.add_parser("k1-compare")
    c.add_argument("old")
    c.add_argument("new")
    c.add_argument("out_csv")
    args = p.parse_args(argv)
    if args.cmd == "run":
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        run_all(out, not args.no_hp)
    elif args.cmd == "k1-reference":
        k1_reference(args.out)
    elif args.cmd == "k1-dump":
        k1_dump(args.out)
    else:
        for row in k1_compare(args.old, args.new, args.out_csv):
            print(row["path"], row["r"], row["N"], row["verdict"], row["first_difference"])


if __name__ == "__main__":
    main()
