"""R6 moment comparisons of the CCCC plate for the GUI (contract ``kkvim-d2-moments/1``); imports kkvim_reference only.

Request: {"contract", "supports": "CCCC", "a", "b", "t", "E", "nu", "q", "n_x", "n_y"} (SI). Answer: {"contract",
"state", "reason", "inputs" (echo), "D", "d2": {"Mx", "My", "Mxy" (node arrays, N), "e_D2_m", "method", "status"},
"ritz": {"Mx", "My", "Mxy" (node arrays), "M", "N", "method", "status"}, "timing", "versions"}.

- D2: the same discrete problem as KK-VIM CCCC (D-049.1) with the rule-A moments of ``r6_moments.d2_moments`` -
  a same-discrete-equation CONSISTENCY check, not accuracy;
- Ritz (P7 basis, M = N = 160): an independent continuous approximation, slow (~1/M) at the edges - OBSERVATION.
Neither establishes continuous moment accuracy: CCCC moments stay NOT_QUALIFIED (review 7, BL-3).
"""

import time

CONTRACT = "kkvim-d2-moments/1"
RITZ_TERMS = 160  # review 7: M = N = 160


def compute(request: dict) -> dict:
    from kkvim_d2_worker import GRID_KEYS, INPUT_KEYS, MAX_UNKNOWNS, ORDERING, SUPPORTS, _float, _int

    out = {"contract": CONTRACT, "state": "error", "reason": "", "inputs": request, "timing": {}, "versions": {}}
    try:
        a, b, t, E, nu, q = (_float(request[key], key) for key in INPUT_KEYS)
        n_x, n_y = (_int(request[key], key) for key in GRID_KEYS)
        supports = request.get("supports")
    except (KeyError, TypeError) as exc:
        out.update(state="rejected", reason=f"invalid request: {exc}")
        return out
    if supports != SUPPORTS:
        out.update(state="rejected", reason="D2 and Ritz are references of the clamped plate (CCCC) only")
        return out
    if n_x >= 3 and n_y >= 3 and (n_x - 1) * (n_y - 1) > MAX_UNKNOWNS:
        out.update(state="out_of_scope", reason=f"grid {n_x} x {n_y} exceeds the engineering limit of the GUI")
        return out
    t0 = time.perf_counter()
    try:
        import numpy as np

        from kkvim_reference import navier, r5_cccc_d2, r5_cccc_ritz, r6_moments

        D = navier.flexural_rigidity(E, t, nu)
        solution = r5_cccc_d2.solve(r5_cccc_d2.D2Problem(a, b, n_x, n_y, D, q), ordering=ORDERING)
        t1 = time.perf_counter()
        d2 = r6_moments.d2_moments(solution.field(), a, b, D, nu)
        ritz = r5_cccc_ritz.solve_smw(r5_cccc_ritz.RitzProblem(a, b, RITZ_TERMS, RITZ_TERMS, D, q))
        rm = r6_moments.ritz_moments(ritz, np.linspace(0.0, a, n_x + 1), np.linspace(0.0, b, n_y + 1), nu)
        t2 = time.perf_counter()
    except ValueError as exc:
        out.update(state="rejected", reason=f"input not accepted by the reference: {exc}")
        return out
    except Exception as exc:  # noqa: BLE001 - reported to the GUI as a failed reference, never as a value
        out.update(state="error", reason=f"{type(exc).__name__}: {exc}")
        return out
    out.update(state="ok", D=D,
               d2={key: d2[key].tolist() for key in ("Mx", "My", "Mxy")} | {"e_D2_m": solution.e_D2,
                                                                          "method": d2["method"], "status": d2["status"]},
               ritz={key: rm[key].tolist() for key in ("Mx", "My", "Mxy")} | {"M": rm["M"], "N": rm["N"],
                                                                             "method": rm["method"], "status": rm["status"]},
               timing={"d2_s": t1 - t0, "ritz_s": t2 - t1})
    try:
        from importlib import metadata

        out["versions"] = {name: metadata.version(name) for name in ("kkvim", "numpy", "scipy")}
    except Exception:  # noqa: BLE001 - versions are informative only
        pass
    return out
