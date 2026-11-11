"""RunRequest: the input of one computation as the solver receives it (GUI_DESIGN.md, 3.3). Qt-free.

Every dimensional value keeps the entered text and unit (``entered``) next to its value in the base
SI unit that is passed to the solver (``units.to_base``). The request is immutable and picklable: it
is sent to the computing process unchanged and is written to the run protocol.

``route`` names the existing kkvim entry of (supports, N); the GUI implements no method of its own.
"""

from dataclasses import asdict, dataclass

# Modes of N = 2-3 (kkvim.coupled_run; D-047, D-048 and its dated addendum of 2026-10-18)
R4_RESEARCH_STOP = "research_stop"  # research_mode=True, stop_on_stabilization=True (D-048.3-4)
R4_RESEARCH_FIXED = "research_fixed"  # research_mode=True, stop_on_stabilization=False (addendum to D-048)
R4_NORMATIVE = "normative"  # research_mode=False (D-047): NOT_QUALIFIED after s = 1
R4_MODES = (R4_NORMATIVE, R4_RESEARCH_STOP, R4_RESEARCH_FIXED)  # the first is the default
R4_RESEARCH_MODES = (R4_RESEARCH_STOP, R4_RESEARCH_FIXED)  # continuation through a QR only on explicit choice (D-048.5)
TRANSFORMS = ("fft", "dense")  # D-048.6: fft is the main path

# Routes: (supports, N) -> the existing kkvim entry that computes it (backend.kkvim_adapter; no method in the GUI)
SUPPORTS = ("SSSS", "CCCC")
ROUTE_R3 = "ssss_r3"  # SSSS, N = 1: kkvim.rectangle_run.run_rectangle_physical (R3, ACCEPTED, D-043)
ROUTE_R4 = "ssss_r4"  # SSSS, N = 2-3: kkvim.coupled_run.run_coupled (R4, D-047, D-048)
ROUTE_SSSS_N46 = "ssss_n46"  # SSSS, N = 4-6: kkvim.research_ssss.run_research, explicit research only (D-056.7, D-060)
ROUTE_CCCC_N1 = "cccc_n1"  # CCCC, N = 1: kkvim.cccc_n1, fixed half-steps s = 1...8 (D-061)
ROUTE_CCCC_N23 = "cccc_n23"  # CCCC, N = 2-3: kkvim.cccc_n23, s = 1 and guarded s = 2 (D-063, D-065)
ROUTE_CCCC_N46 = "cccc_n46"  # CCCC, N = 4-6: kkvim.cccc_n46, s = 1 and guarded s = 2 (D-067, D-068)
CCCC_ROUTES = (ROUTE_CCCC_N1, ROUTE_CCCC_N23, ROUTE_CCCC_N46)
N_ALL = (1, 2, 3, 4, 5, 6)
CCCC_HALF_STEPS = {ROUTE_CCCC_N1: tuple(range(1, 9)), ROUTE_CCCC_N23: (1, 2), ROUTE_CCCC_N46: (1, 2)}  # s >= 3 is not offered


def route_of(supports: str, N: int) -> str | None:
    """The route of (supports, N), or None if no solver computes it."""
    if N not in N_ALL:
        return None
    if supports == "SSSS":
        return ROUTE_R3 if N == 1 else ROUTE_R4 if N <= 3 else ROUTE_SSSS_N46
    if supports == "CCCC":
        return ROUTE_CCCC_N1 if N == 1 else ROUTE_CCCC_N23 if N <= 3 else ROUTE_CCCC_N46
    return None


@dataclass(frozen=True)
class RunRequest:
    entered: tuple  # ((key, text, unit), ...) exactly as typed
    a: float  # m
    b: float  # m
    t: float  # m
    E: float  # Pa
    nu: float
    q: float  # Pa
    n_x: int
    n_y: int
    N: int
    supports: str = "SSSS"
    # N = 1 (R2/R3 stopping, NUMERICAL_SPEC.md 11.2)
    tol_inf: float | None = None
    tol_E: float | None = None
    s_min: int | None = None
    s_max: int | None = None
    strict: bool = False
    # N = 2-3 (R4); research continuation is never the default (D-048.3, D-048.5)
    r4_mode: str = R4_NORMATIVE
    transform: str | None = None
    half_steps: int | None = None  # R4: limit or fixed number; SSSS N = 4-6 and CCCC: the exact number of half-steps
    # SSSS, N = 4-6: computed only when the research route is selected explicitly (D-056.7, D-060)
    ssss_research: bool = False

    @property
    def route(self) -> str | None:
        return route_of(self.supports, self.N)

    @property
    def research_mode(self) -> bool:
        """``research_mode`` of ``run_coupled``: True only for an explicitly chosen research mode."""
        return self.r4_mode in R4_RESEARCH_MODES

    @property
    def stop_on_stabilization(self) -> bool:
        """``stop_on_stabilization`` of ``run_coupled``: False only for the fixed number of half-steps."""
        return self.r4_mode != R4_RESEARCH_FIXED

    def as_dict(self) -> dict:
        d = asdict(self)
        d["entered"] = [{"key": k, "text": text, "unit": unit} for k, text, unit in self.entered]
        return d
