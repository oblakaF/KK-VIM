"""R6 moment reference of the SSSS plate for the GUI (contract ``kkvim-navier-moments/1``); imports kkvim_reference only.

Request (stdin): {"contract", "a", "b", "t", "E", "nu", "q", "n_x", "n_y"} (SI). Answer (stdout):
{"contract", "state", "reason", "inputs" (echo), "D", "navier": {"Mx", "My", "Mxy" (node arrays, N), "M", "tail_bound",
"rounding_model", "method", "status"}, "points": [{"name", "x", "y", "navier": {..}, "levy": {..}}], "levy": {"M",
"tail_bound", "method", "status"}, "versions"}. Navier at the result nodes (M = M_R3 = 2559) and Navier and Levy at named
points (M_L = 6117): ``kkvim_reference.r6_moments``. The worker never sees a KK-VIM field. Status: R6 research reference;
the qualification of the moment reference is the owner's decision OD-4 (not taken).
"""

CONTRACT = "kkvim-navier-moments/1"
INPUT_KEYS = ("a", "b", "t", "E", "nu", "q")
MAX_NODES = 1281


def named_points(a, b):
    """The points of the comparison (review 8.9): centre, edge midpoints, corners, quarter points."""
    return [("centre", a / 2, b / 2), ("edge x=0", 0.0, b / 2), ("edge x=a", a, b / 2), ("edge y=0", a / 2, 0.0),
            ("edge y=b", a / 2, b), ("corner (0,0)", 0.0, 0.0), ("corner (a,b)", a, b), ("corner (a,0)", a, 0.0),
            ("quarter (a/4,b/4)", a / 4, b / 4), ("quarter (3a/4,b/4)", 3 * a / 4, b / 4)]


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, got {value!r}")
    return float(value)


def _integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or not 2 <= value < MAX_NODES:
        raise TypeError(f"{name} must be an integer 2...{MAX_NODES - 1}, got {value!r}")
    return value


def compute(request: dict) -> dict:
    import platform

    out = {"contract": CONTRACT, "state": "error", "reason": "", "inputs": request,
           "versions": {"python": platform.python_version()}}
    try:
        a, b, t, E, nu, q = (_number(request[k], k) for k in INPUT_KEYS)
        n_x, n_y = _integer(request["n_x"], "n_x"), _integer(request["n_y"], "n_y")
    except (KeyError, TypeError) as exc:
        out.update(state="rejected", reason=f"invalid request: {exc}")
        return out
    try:
        from kkvim_reference import navier, r6_moments

        D = navier.flexural_rigidity(E, t, nu)
        nodes = r6_moments.navier_moments_nodes(a, b, q, nu, n_x, n_y)
        pts = named_points(a, b)
        xs, ys = [p[1] for p in pts], [p[2] for p in pts]
        nav = r6_moments.navier_moments_points(a, b, q, nu, xs, ys)
        lev = r6_moments.levy_moments_points(a, b, q, nu, xs, ys)
    except ValueError as exc:
        text = str(exc)
        out.update(state="out_of_scope" if text.startswith("ERROR_INPUT") else "rejected", reason=text)
        return out
    except Exception as exc:  # noqa: BLE001 - reported to the GUI as a failed reference, never as a value
        out.update(state="error", reason=f"{type(exc).__name__}: {exc}")
        return out
    points = []
    for k, (name, x, y) in enumerate(pts):
        points.append({"name": name, "x": x, "y": y,
                       "navier": {key: float(nav[key][k]) for key in ("Mx", "My", "Mxy")},
                       "levy": {key: float(lev[key][k]) for key in ("Mx", "My", "Mxy")},
                       "levy_interior_bound": {key: lev["interior_bound"][key][k] for key in ("Mx", "My", "Mxy")}})
    out.update(state="ok", D=D,
               navier={"Mx": nodes["Mx"].tolist(), "My": nodes["My"].tolist(), "Mxy": nodes["Mxy"].tolist(),
                       "M": nodes["M"], "tail_bound": nodes["tail_bound"], "rounding_model": nodes["rounding_model"],
                       "method": nodes["method"], "status": nodes["status"]},
               levy={"M": lev["M"], "tail_bound": lev["tail_bound"], "method": lev["method"], "status": lev["status"]},
               points=points)
    try:
        from importlib import metadata

        out["versions"].update({name: metadata.version(name) for name in ("kkvim", "numpy")})
    except Exception:  # noqa: BLE001 - versions are informative only
        pass
    return out
