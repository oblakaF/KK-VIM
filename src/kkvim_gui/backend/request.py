"""RunRequest: the input of one computation as the solver receives it (GUI_DESIGN.md, 3.3). Qt-free.

Every dimensional value keeps the entered text and unit (``entered``) next to its value in the base
SI unit that is passed to the solver (``units.to_base``). The request is immutable and picklable: it
is sent to the computing process unchanged and is written to the run protocol.
"""

from dataclasses import asdict, dataclass

# Modes of N = 2-3 (kkvim.coupled_run; D-047, D-048 and its dated addendum of 2026-10-18)
R4_RESEARCH_STOP = "research_stop"  # research_mode=True, stop_on_stabilization=True (D-048.3-4)
R4_RESEARCH_FIXED = "research_fixed"  # research_mode=True, stop_on_stabilization=False (addendum to D-048)
R4_NORMATIVE = "normative"  # research_mode=False (D-047): NOT_QUALIFIED after s = 1
R4_MODES = (R4_NORMATIVE, R4_RESEARCH_STOP, R4_RESEARCH_FIXED)  # the first is the default
R4_RESEARCH_MODES = (R4_RESEARCH_STOP, R4_RESEARCH_FIXED)  # continuation through a QR only on explicit choice (D-048.5)
TRANSFORMS = ("fft", "dense")  # D-048.6: fft is the main path


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
    half_steps: int | None = None

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
