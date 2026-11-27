"""R6 quantities of the analysis window: the registry of field quantities, the moments of an outcome and their
independent references (GUI_DESIGN.md 8; R6 review 8.5). Qt-free; imports no ``kkvim`` and no ``kkvim_reference``.

- Registry (``QUANTITIES``) instead of the fixed {"w", "w_bar"}: key, SI unit, display unit and scale, figure-text keys.
- ``MomentsData``: the moments of one outcome, computed once by ``kkvim_adapter.outcome_moments`` (kkvim.moments) and
  kept read-only; stresses at a surface or at a thickness coordinate by ``kkvim_adapter.moment_stresses``. The GUI
  contains no moment formula: values are taken from kkvim, displayed with a unit scale only, never interpolated.
- ``MomentsRefJob``: the independent moment references in their own process (``python -I -m kkvim_navier_worker`` with
  the contract ``kkvim-navier-moments/1`` for SSSS; ``kkvim_d2_worker`` with ``kkvim-d2-moments/1`` for CCCC).
- ``compare``: differences of stored node values (KK-VIM minus reference) and their maxima over all nodes, over the
  interior nodes at least 2h from the edges, at the centre, the edge midpoints and the corners (review 8.9). These are
  OBSERVATIONS: no threshold, no PASS/FAIL (none is approved; D-025).

Every quantity of this module is RESEARCH / NOT_QUALIFIED (R6 not accepted); the moments of a diagnostic field are
diagnostic. The status of the displacement result is never transferred to its moments.
"""

import json
import subprocess
import sys
import threading
import time
import weakref
from dataclasses import dataclass

import numpy as np

STATUS = "RESEARCH / NOT_QUALIFIED"
NAVIER_CONTRACT = "kkvim-navier-moments/1"
D2_CONTRACT = "kkvim-d2-moments/1"
TIMEOUT = 300.0


@dataclass(frozen=True)
class Quantity:
    key: str
    kind: str  # "deflection" | "moment" | "stress"
    unit: str  # SI unit of the stored values
    display_unit: str
    scale: float  # display value = SI value * scale
    name_ru: str
    name_en: str
    symbol: str  # for files and the figure text


QUANTITIES = {
    "w": Quantity("w", "deflection", "m", "mm", 1e3, "прогиб w", "deflection w", "w"),
    "Mx": Quantity("Mx", "moment", "N", "N·m/m", 1.0, "изгибающий момент M_x", "bending moment M_x", "M_x"),
    "My": Quantity("My", "moment", "N", "N·m/m", 1.0, "изгибающий момент M_y", "bending moment M_y", "M_y"),
    "Mxy": Quantity("Mxy", "moment", "N", "N·m/m", 1.0, "крутящий момент M_xy (соглашение L-107)",
                    "twisting moment M_xy (L-107 convention)", "M_xy"),
    "M1": Quantity("M1", "moment", "N", "N·m/m", 1.0, "главный момент M_1", "principal moment M_1", "M_1"),
    "M2": Quantity("M2", "moment", "N", "N·m/m", 1.0, "главный момент M_2", "principal moment M_2", "M_2"),
    "sigma_x": Quantity("sigma_x", "stress", "Pa", "MPa", 1e-6, "напряжение σ_x", "stress σ_x", "σ_x"),
    "sigma_y": Quantity("sigma_y", "stress", "Pa", "MPa", 1e-6, "напряжение σ_y", "stress σ_y", "σ_y"),
    "tau_xy": Quantity("tau_xy", "stress", "Pa", "MPa", 1e-6, "касательное напряжение τ_xy", "shear stress τ_xy", "τ_xy"),
}
MOMENT_KEYS = ("Mx", "My", "Mxy")
SURFACES = ("bottom", "top", "z")
SURFACE_TEXT = {"bottom": "нижняя поверхность, z = +t/2", "top": "верхняя поверхность, z = −t/2", "z": "заданное z"}
LIMITATIONS_RU = (  # the limitations of kkvim.cli_moments.LIMITATIONS, in Russian for the interface
    "RESEARCH / NOT_QUALIFIED: точность моментов и напряжений не квалифицирована (R6 и R5 не приняты).",
    "Теория Кирхгофа–Лява: нет τ_xz, τ_yz, Q_x, Q_y и σ_z (БЛ-5); трёхмерной точности нет (БЛ-4).",
    "Граничная кривизна CCCC — правило А (2f₁/h²); его второй порядок на вычисленном поле наблюдался для CCCC N = 1 "
    "(рецензия 6.2) и не доказан для всех N, сеток и полей.",
    "Согласие прогибов не переносится на моменты, а моменты — на напряжения (рецензия 8.9).",
    "Модель округления — модель, не граница (БЛ-6).",
)
NOT_COMPUTED = ("τ_xz, τ_yz, Q_x, Q_y (не специфицированы, БЛ-5)", "эквивалентные напряжения (не специфицированы, OD-2)",
                "σ_z (теория Кирхгофа–Лява)")


class MomentsData:
    """The moments of one outcome (computed once, read-only), or the reason why there are none."""

    _cache: dict = {}

    def __init__(self, outcome):
        from kkvim_gui.backend import kkvim_adapter

        self.outcome = outcome
        self.moments, self.reason = None, ""
        try:
            self.moments = kkvim_adapter.outcome_moments(outcome)
        except kkvim_adapter.MomentsUnavailable as exc:
            self.reason = str(exc)
        self.texts = kkvim_adapter.moment_texts()
        self._stress = {}

    @classmethod
    def of(cls, outcome):
        """The data of ``outcome`` (one computation per outcome object while it lives)."""
        key = id(outcome)
        entry = cls._cache.get(key)
        if entry is not None and entry[0]() is outcome:
            return entry[1]
        data = cls(outcome)
        cls._cache[key] = (weakref.ref(outcome, lambda _, k=key: cls._cache.pop(k, None)), data)
        return data

    @property
    def available(self) -> bool:
        return self.moments is not None

    @property
    def diagnostic(self) -> bool:
        return self.outcome.diagnostic_only

    def z_of(self, surface, z=None):
        t = self.outcome.request.t
        return 0.5 * t if surface == "bottom" else -0.5 * t if surface == "top" else float(z)

    def stresses(self, surface="bottom", z=None):
        from kkvim_gui.backend import kkvim_adapter

        zz = self.z_of(surface, z)
        if zz not in self._stress:
            self._stress[zz] = kkvim_adapter.moment_stresses(self.moments, self.outcome.request.t, zz)
        return self._stress[zz]

    def values(self, key, surface="bottom", z=None):
        """SI values of a quantity on all nodes (read-only arrays of kkvim; w: the stored field)."""
        if key == "w":
            return self.outcome.w
        if not self.available:
            raise ValueError(self.reason)
        if QUANTITIES[key].kind == "moment":
            return getattr(self.moments, key)
        return self.stresses(surface, z)[("sigma_x", "sigma_y", "tau_xy").index(key)]

    def display(self, key, surface="bottom", z=None):
        return np.asarray(self.values(key, surface, z)) * QUANTITIES[key].scale

    def summary(self, key, surface="bottom", z=None):
        from kkvim_gui.backend import kkvim_adapter

        return kkvim_adapter.moment_summary(self.values(key, surface, z), self.outcome.x, self.outcome.y)


# ------------------------------------------------------------------------------------------------ reference job


def reference_kind(outcome) -> str | None:
    return {"SSSS": "navier", "CCCC": "d2"}.get(outcome.request.supports)


def reference_request(outcome) -> dict:
    r = outcome.request
    base = {"a": r.a, "b": r.b, "t": r.t, "E": r.E, "nu": r.nu, "q": r.q, "n_x": r.n_x, "n_y": r.n_y}
    if r.supports == "SSSS":
        return {"contract": NAVIER_CONTRACT, **base}
    return {"contract": D2_CONTRACT, "supports": "CCCC", **base}


def reference_command(outcome) -> list:
    module = "kkvim_navier_worker" if outcome.request.supports == "SSSS" else "kkvim_d2_worker"
    return [sys.executable, "-I", "-m", module]


class MomentsRefJob:
    """One moment reference in its own process; ``poll`` never blocks; the answer is checked against the request."""

    def __init__(self, outcome, command=None, clock=time.monotonic, timeout=TIMEOUT):
        self.inputs = reference_request(outcome)
        self.clock, self.timeout, self.t0 = clock, timeout, clock()
        self.record = None
        self._answer = None
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self._process = subprocess.Popen(command or reference_command(outcome), stdin=subprocess.PIPE,
                                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=flags)
        payload = json.dumps(self.inputs).encode("utf-8")
        self._reader = threading.Thread(target=self._communicate, args=(payload,), daemon=True)
        self._reader.start()

    def _communicate(self, payload):
        try:
            self._answer = self._process.communicate(payload)
        except (OSError, ValueError) as exc:
            self._answer = (b"", str(exc).encode("utf-8", "replace"))

    def poll(self):
        if self.record is not None:
            return self.record
        if self._reader.is_alive():
            if self.clock() - self.t0 > self.timeout:
                self.terminate()
                self.record = {"state": "failed", "reason": f"процесс эталона моментов не ответил за {self.timeout:g} с"}
            return self.record
        out, err = self._answer or (b"", b"")
        if self._process.returncode != 0 or not out:
            tail = err.decode("utf-8", "replace").strip().splitlines()[-1:] or [f"код {self._process.returncode}"]
            self.record = {"state": "failed", "reason": f"процесс эталона моментов завершился без ответа: {tail[0]}"}
            return self.record
        try:
            data = json.loads(out.decode("utf-8"))
        except ValueError as exc:
            self.record = {"state": "failed", "reason": f"ответ эталона моментов не прочитан: {exc}"}
            return self.record
        if not isinstance(data, dict) or data.get("contract") != self.inputs["contract"] or data.get("inputs") != self.inputs:
            self.record = {"state": "failed", "reason": "ответ эталона моментов не соответствует запросу"}
            return self.record
        self.record = data
        return self.record

    def terminate(self, timeout=5.0):
        if self._process.poll() is None:
            self._process.kill()
            try:
                self._process.wait(timeout)
            except subprocess.TimeoutExpired:
                return False
        return True


# ------------------------------------------------------------------------------------------------ comparison


REFERENCE_TITLES = {
    "navier": ("Навье (двойной ряд (131), M = {M}; эталон R6, квалификация — решение владельца OD-4)", "Navier"),
    "levy": ("Леви (устойчивые формы, M = {M})", "Levy"),
    "d2": ("МКР D2 — та же дискретная задача (согласованность, не точность)", "FDM D2"),
    "ritz": ("Ритц P7, M = N = {M} (непрерывное приближение; у края медленно, наблюдение)", "Ritz"),
}


def reference_fields(record) -> dict:
    """{name: {"Mx", "My", "Mxy" (arrays N), "title", ...}} of a reference answer with node fields."""
    out = {}
    if not record or record.get("state") != "ok":
        return out
    for name in ("navier", "d2", "ritz"):
        part = record.get(name)
        if isinstance(part, dict) and "Mx" in part:
            arrays = {k: np.array(part[k], dtype=float) for k in MOMENT_KEYS}
            for a in arrays.values():
                a.setflags(write=False)
            title = REFERENCE_TITLES[name][0].format(M=part.get("M", ""))
            out[name] = {**arrays, "title": title, "short": REFERENCE_TITLES[name][1],
                         "tail_bound": part.get("tail_bound"), "rounding_model": part.get("rounding_model"),
                         "status": part.get("status", "")}
    return out


def named_nodes(outcome) -> list:
    """(name, i, j) of the centre, edge midpoints and corners that are nodes of the result grid (nothing interpolated)."""
    nx, ny = len(outcome.x) - 1, len(outcome.y) - 1
    out = []
    if nx % 2 == 0 and ny % 2 == 0:
        out.append(("центр", nx // 2, ny // 2))
    if ny % 2 == 0:
        out += [("середина края x = 0", 0, ny // 2), ("середина края x = a", nx, ny // 2)]
    if nx % 2 == 0:
        out += [("середина края y = 0", nx // 2, 0), ("середина края y = b", nx // 2, ny)]
    out += [("угол (0, 0)", 0, 0), ("угол (a, b)", nx, ny)]
    return out


def compare(data: MomentsData, ref: dict) -> list:
    """Rows (quantity, place, KK-VIM, reference, difference) and norms; differences only - observations."""
    rows = []
    o = data.outcome
    nx, ny = len(o.x) - 1, len(o.y) - 1
    k = 2
    for key in MOMENT_KEYS:
        mine, theirs = np.asarray(getattr(data.moments, key)), ref[key]
        if theirs.shape != mine.shape:
            continue
        diff = mine - theirs
        scale = float(np.max(np.abs(theirs))) or 1.0
        interior = diff[k:nx - k + 1, k:ny - k + 1] if nx > 2 * k and ny > 2 * k else diff[1:-1, 1:-1]
        for name, i, j in named_nodes(o):
            rows.append({"quantity": key, "place": name, "i": i, "j": j, "kkvim": float(mine[i, j]),
                         "reference": float(theirs[i, j]), "difference": float(diff[i, j])})
        rows.append({"quantity": key, "place": "max |Δ| по всем узлам", "kkvim": None, "reference": scale,
                     "difference": float(np.max(np.abs(diff))), "relative": float(np.max(np.abs(diff))) / scale})
        rows.append({"quantity": key, "place": "max |Δ| по внутренним узлам (≥ 2h от края)", "kkvim": None,
                     "reference": scale, "difference": float(np.max(np.abs(interior))) if interior.size else None,
                     "relative": float(np.max(np.abs(interior))) / scale if interior.size else None})
    return rows


def levy_rows(record) -> list:
    """Navier and Levy at the named points (the reference's own points; the KK-VIM field is not involved)."""
    if not record or record.get("state") != "ok" or "points" not in record:
        return []
    out = []
    for p in record["points"]:
        for key in MOMENT_KEYS:
            n, lv = p["navier"][key], p["levy"][key]
            out.append({"place": p["name"], "x": p["x"], "y": p["y"], "quantity": key, "navier": n, "levy": lv,
                        "difference": n - lv})
    return out
