"""Navier reference worker of the GUI: one isolated process evaluates the qualified SSSS Navier reference.

The GUI (``kkvim_gui``) never imports this package or ``kkvim_reference``: it starts ``python -m kkvim_navier_worker``
as a separate process (``kkvim_gui.backend.navier_job``) and exchanges JSON only. This package imports
``kkvim_reference.navier`` and the standard library; it never imports ``kkvim`` (the KK-VIM solver) or ``kkvim_gui``
(tests/test_gui_independence.py). The reference therefore never sees a KK-VIM field: it receives only the physical
input of the plate.

Reference (unchanged, used through its public API): ``kkvim_reference.navier.NavierRectangle.from_material`` and
``NavierRectangle.at_points`` at the plate centre (a/2, b/2), level M = M_R3 = 2559 (D-036, D-037), SSSS rectangle
under uniform load, Kirchhoff–Love; the exact aspect-ratio gate 0.2 <= b/a <= 5 with the representation tolerance 4u
of D-038 is the reference's own check (ValueError ERROR_INPUT outside). Every value returned is a field of the
reference result: ``w`` (m), ``w_coef`` = w D / (q a^4) and ``w_error_bound_coef`` (truncation bound ЧВ-R2 plus the
rounding model of the reference, in q a^4 / D). Nothing is scaled, interpolated or fitted here; every q of a load
series is evaluated by its own ``at_points`` call.

Contract ``kkvim-navier-worker/1`` (one request per process):

    stdin  {"contract", "a", "b", "t", "E", "nu", "q": [q_1, ...]}            SI: m, Pa
    stdout {"contract", "state": "ok" | "out_of_scope" | "rejected" | "error", "reason", "inputs" (echo),
            "method", "source", "M", "x", "y", "D", "points": [{"q", "w_m", "w_coef", "error_bound_coef"}],
            "versions"}

Floats travel as JSON numbers (Python's shortest round-trip repr), so the GUI receives the binary64 values exactly.
"""

CONTRACT = "kkvim-navier-worker/1"
INPUT_KEYS = ("a", "b", "t", "E", "nu")
MAX_POINTS = 50
METHOD = "kkvim_reference.navier.NavierRectangle.from_material(...).at_points(a/2, b/2, M=M_R3, diagnostics=False)"
SOURCE = ("Navier double sine series for the SSSS rectangle under uniform load [L-107, eq. (131), p. 129]; "
          "docs/NAVIER_REFERENCE_SPEC.md; qualified deflection reference of R3 (D-036, D-037, D-038, D-043)")


def _float(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, got {value!r}")
    return float(value)


def compute(request: dict) -> dict:
    """Evaluate the reference for one request of the contract; never raises (the state names the outcome)."""
    import platform

    from kkvim_reference import navier

    out = {"contract": CONTRACT, "state": "error", "reason": "", "inputs": request, "method": METHOD, "source": SOURCE,
           "M": navier.M_R3, "points": [], "versions": {"python": platform.python_version()}}
    try:
        if not isinstance(request, dict) or request.get("contract") != CONTRACT:
            raise TypeError(f"unknown contract {request.get('contract') if isinstance(request, dict) else request!r}")
        a, b, t, E, nu = (_float(request[key], key) for key in INPUT_KEYS)
        loads = request["q"]
        if not isinstance(loads, list) or not 1 <= len(loads) <= MAX_POINTS:
            raise TypeError(f"q must be a list of 1...{MAX_POINTS} numbers")
        loads = [_float(q, "q") for q in loads]
    except (KeyError, TypeError, ValueError) as exc:
        out.update(state="rejected", reason=f"invalid request: {exc}")
        return out
    x, y = a / 2.0, b / 2.0
    out.update(x=x, y=y)
    try:
        points = []
        D = None
        for q in loads:
            plate = navier.NavierRectangle.from_material(a=a, b=b, q=q, E=E, thickness=t, nu=nu)
            result = plate.at_points([x], [y], M=navier.M_R3, diagnostics=False)
            D = plate.D
            points.append({"q": q, "w_m": float(result.w[0]), "w_coef": float(result.w_coef[0]),
                           "error_bound_coef": float(result.w_error_bound_coef)})
    except ValueError as exc:
        text = str(exc)
        out.update(state="out_of_scope" if text.startswith("ERROR_INPUT") else "rejected", reason=text)
        return out
    except Exception as exc:  # noqa: BLE001 - reported to the GUI as a failed reference, never as a value
        out.update(state="error", reason=f"{type(exc).__name__}: {exc}")
        return out
    out.update(state="ok", D=D, points=points)
    try:
        from importlib import metadata

        out["versions"].update({name: metadata.version(name) for name in ("kkvim", "numpy")})
    except Exception:  # noqa: BLE001 - versions are informative only
        pass
    return out


def main() -> int:
    import json
    import sys

    try:
        request = json.loads(sys.stdin.read())
    except ValueError as exc:
        request = {"contract": None, "error": str(exc)}
    sys.stdout.write(json.dumps(compute(request), ensure_ascii=True))  # ASCII: independent of the console code page
    sys.stdout.flush()
    return 0
