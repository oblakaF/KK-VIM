"""D2 worker of the GUI: one isolated process computes the QA reference D2 of a clamped (CCCC) plate.

The GUI (``kkvim_gui``) never imports this package or ``kkvim_reference``: it starts ``python -I -m kkvim_d2_worker`` as a
separate process (``kkvim_gui.backend.d2_job``) and exchanges JSON only. This package imports ``kkvim_reference`` and the
standard library; it never imports ``kkvim`` (the KK-VIM solver) or ``kkvim_gui`` (tests/test_gui_independence.py). D2
therefore never sees a KK-VIM field: it receives only the physical input of the plate and the grid.

Reference (unchanged, used through its public API; D-049.1, D-049.8, D-066.1, п. 4):

- ``kkvim_reference.navier.flexural_rigidity(E, t, nu)`` — D = E t^3 / (12 (1 - nu^2)) [L-107, eq. (3), p. 16];
- ``kkvim_reference.r5_cccc_d2.solve(D2Problem(a, b, n_x, n_y, D, q))`` — the 13-point finite-difference scheme of the
  clamped plate with even ghost nodes, sparse LU and iterative refinement with the double-double residual of the exact
  operator; ``D2Solution.field``, ``D2Solution.centre``, ``D2Solution.alpha`` and ``D2Solution.e_D2``.

D2 is an independent implementation of the SAME discrete problem as the KK-VIM CCCC route (D1 = D2, D-049.1): agreement
with D2 does not check the grid error and says nothing about the continuous plate (D-049.1, п. 2; D-066.2). Nothing is
scaled, interpolated or fitted here: every load q is its own ``solve`` call (one request = one q).

Normalization. ``D2Solution.alpha`` uses the shorter full side l = min(a, b); the GUI compares beta_a = w_c D / (q a^4)
(reference length a). Both are reported; beta_a is formed from the dimensional centre deflection w_c (m) and the D of
this worker, never from alpha. Without a centre node (odd n_x or n_y) there is no centre value: nothing is interpolated.

Engineering limit of the GUI (not a scientific limit): at most MAX_UNKNOWNS = (1280 - 1)(256 - 1) interior unknowns —
the largest D2 grid executed in the QA plan (docs/R5_CCCC_PREQUALIFICATION.plan.md, 1.5: r = 5, n = 256). A larger grid
is refused ("out_of_scope") without a computation.

Timing (``timing``, seconds, ``time.perf_counter`` of this process; one execution, an observation, never a ranking):
``import_s`` — import of NumPy, SciPy and the reference modules; ``solve_s`` — the call ``r5_cccc_d2.solve``: node-wise
assembly, sparse LU (SuperLU) factorization, first solve and the iterative refinement (the reference does not expose
these phases separately, so they are not timed separately); ``post_s`` — field, centre and alpha from the solution
and their conversion to lists; ``worker_s`` — from the entry of ``main`` (after the interpreter and this package
started) to the end of the computation, before the answer is serialized. Interpreter start, serialization and transfer
are outside every timer of this process (the GUI measures the whole wall time).

Contract ``kkvim-d2-worker/1`` (one request per process):

    stdin  {"contract", "supports", "a", "b", "t", "E", "nu", "q", "n_x", "n_y"}          SI: m, Pa
    stdout {"contract", "state": "ok" | "out_of_scope" | "rejected" | "error", "reason", "inputs" (echo), "method",
            "source", "problem", "D", "result", "timing", "runtime", "versions"}

Floats travel as JSON numbers (Python's shortest round-trip repr), so the GUI receives the binary64 values exactly.
"""

import time

CONTRACT = "kkvim-d2-worker/1"
INPUT_KEYS = ("a", "b", "t", "E", "nu", "q")
GRID_KEYS = ("n_x", "n_y")
SUPPORTS = "CCCC"
MAX_UNKNOWNS = (1280 - 1) * (256 - 1)  # 326 145: the largest D2 grid of the QA plan (r = 5, n = 256)
ORDERING = "MMD_AT_PLUS_A"  # the default ordering of r5_cccc_d2.solve
METHOD = ("kkvim_reference.r5_cccc_d2.solve(D2Problem(a, b, n_x, n_y, D, q)) with "
          "D = kkvim_reference.navier.flexural_rigidity(E, t, nu)")
SOURCE = ("QA reference D2: 13-point finite-difference scheme of the clamped plate with even ghost nodes, sparse LU and "
          "iterative refinement (docs/R5_CCCC_REFERENCE_MATH.pre-review.md, 9.2-9.5; docs/R5_CCCC_PREQUALIFICATION.plan.md, "
          "section 1; D-049.1, D-049.8): the same discrete problem as KK-VIM CCCC, not a check of the grid error")


def _float(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, got {value!r}")
    return float(value)


def _int(value, name):
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, got {value!r}")
    return value


def runtime() -> dict:
    """The hardware and runtime context of this process (informative; the role of every timing is an observation)."""
    import os
    import platform

    return {"python": platform.python_version(), "implementation": platform.python_implementation(),
            "platform": platform.platform(), "machine": platform.machine(), "processor": platform.processor(),
            "cpu_count": os.cpu_count()}


def compute(request: dict, t_entry: float | None = None) -> dict:
    """Run D2 for one request of the contract; never raises (the state names the outcome)."""
    t_entry = time.perf_counter() if t_entry is None else t_entry
    out = {"contract": CONTRACT, "state": "error", "reason": "", "inputs": request, "method": METHOD, "source": SOURCE,
           "problem": None, "D": None, "result": None, "timing": {}, "runtime": runtime(), "versions": {}}
    try:
        if not isinstance(request, dict) or request.get("contract") != CONTRACT:
            raise TypeError(f"unknown contract {request.get('contract') if isinstance(request, dict) else request!r}")
        supports = request.get("supports")
        a, b, t, E, nu, q = (_float(request[key], key) for key in INPUT_KEYS)
        n_x, n_y = (_int(request[key], key) for key in GRID_KEYS)
    except (KeyError, TypeError) as exc:
        out.update(state="rejected", reason=f"invalid request: {exc}")
        return out
    if supports != SUPPORTS:
        out.update(state="rejected", reason=f"D2 is the clamped-plate scheme (CCCC: w = dw/dn = 0 on the boundary); "
                                            f"supports {supports!r} are not computed by it")
        return out
    if n_x >= 3 and n_y >= 3 and (n_x - 1) * (n_y - 1) > MAX_UNKNOWNS:
        out.update(state="out_of_scope", reason=f"grid {n_x} x {n_y}: {(n_x - 1) * (n_y - 1)} unknowns exceed the engineering "
                                                f"limit {MAX_UNKNOWNS} of the GUI (largest D2 grid executed in the QA plan, "
                                                "r = 5, n = 256); not computed")
        return out
    t0 = time.perf_counter()
    try:
        from kkvim_reference import navier, r5_cccc_d2
    except Exception as exc:  # noqa: BLE001 - reported to the GUI as a failed reference, never as a value
        out.update(state="error", reason=f"reference not imported: {type(exc).__name__}: {exc}")
        return out
    t1 = time.perf_counter()
    out["timing"]["import_s"] = t1 - t0
    try:
        D = navier.flexural_rigidity(E, t, nu)
        problem = r5_cccc_d2.D2Problem(a, b, n_x, n_y, D, q)
    except ValueError as exc:
        out.update(state="rejected", reason=f"input not accepted by the reference: {exc}")
        return out
    out["D"] = D
    out["problem"] = {"lx": problem.lx, "ly": problem.ly, "nx": problem.nx, "ny": problem.ny, "D": problem.D,
                      "q": problem.q, "unknowns": problem.size, "ordering": ORDERING}
    try:
        t2 = time.perf_counter()
        solution = r5_cccc_d2.solve(problem, ordering=ORDERING)
        t3 = time.perf_counter()
        field = solution.field()
        centre = solution.centre()
        alpha = solution.alpha()
        k = int(abs(field).argmax())
        i_max, j_max = divmod(k, n_y + 1)
        result = {
            "w_centre_m": centre,
            "centre": None if centre is None else {"i": n_x // 2, "j": n_y // 2},
            "centre_reason": "" if centre is not None else
            "the plate centre (a/2, b/2) is not a node of the grid (odd n_x or n_y): no centre value, nothing interpolated",
            "beta_a": None if centre is None else centre * D / (q * a ** 4),
            "alpha_short_side": alpha,
            "short_side_m": min(a, b),
            "w_max_abs_node": {"i": i_max, "j": j_max, "w_m": float(field[i_max, j_max])},
            "e_D2_m": solution.e_D2,
            "refinement_steps": len(solution.corrections),
            "corrections_m": list(solution.corrections),
            "residual_max": list(solution.residual_max),
            "factor_nnz": solution.factor_nnz,
            "field_m": field.tolist(),
        }
        t4 = time.perf_counter()
    except Exception as exc:  # noqa: BLE001 - reported to the GUI as a failed reference, never as a value
        out.update(state="error", reason=f"{type(exc).__name__}: {exc}")
        return out
    out["timing"].update(solve_s=t3 - t2, post_s=t4 - t3, worker_s=t4 - t_entry)
    out.update(state="ok", result=result)
    try:
        from importlib import metadata

        out["versions"] = {name: metadata.version(name) for name in ("kkvim", "numpy", "scipy")}
    except Exception:  # noqa: BLE001 - versions are informative only
        pass
    return out


def main() -> int:
    import json
    import sys

    t_entry = time.perf_counter()
    try:
        request = json.loads(sys.stdin.read())
    except ValueError as exc:
        request = {"contract": None, "error": str(exc)}
    from kkvim_d2_worker import moments  # R6: the moment contract of this worker (kkvim_reference only)

    r6 = isinstance(request, dict) and request.get("contract") == moments.CONTRACT
    answer = moments.compute(request) if r6 else compute(request, t_entry)
    sys.stdout.write(json.dumps(answer, ensure_ascii=True))  # ASCII: independent of the code page
    sys.stdout.flush()
    return 0
