"""GUI-CORE-N16-P1-QA: real schema-1 manifests written by the BASE export (306e85a, non-Git snapshot base306/src).

usage: python -I gen_schema1.py <out_dir>
Runs the base adapter in-process (R3 CONVERGED, R3 MAX_HALF_STEPS, R4 normative, R4 research) and writes each package with the
base ``export.write_result`` (schema 1). Asserts every kkvim / kkvim_gui module comes from the base snapshot.
"""
import json
import sys
from pathlib import Path

BASE = (Path(__file__).resolve().parent / "base306" / "src").resolve()
sys.path.insert(0, str(BASE))
from kkvim_gui.backend import export, kkvim_adapter  # noqa: E402
from kkvim_gui.backend.request import R4_NORMATIVE, R4_RESEARCH_STOP, RunRequest  # noqa: E402

out = Path(sys.argv[1])
P = dict(entered=(), a=1.0, b=1.0, t=0.01, E=2.1e11, nu=0.3, q=1000.0)
D = kkvim_adapter.DEFAULTS
cases = {
    "r3_converged": RunRequest(**P, n_x=16, n_y=16, N=1, tol_inf=D["tol_inf"], tol_E=D["tol_E"], s_min=D["s_min"], s_max=D["s_max"]),
    "r3_max_half_steps": RunRequest(**P, n_x=16, n_y=16, N=1, tol_inf=D["tol_inf"], tol_E=D["tol_E"], s_min=1, s_max=2),
    "r4_normative": RunRequest(**P, n_x=16, n_y=16, N=2, r4_mode=R4_NORMATIVE, transform="fft", half_steps=D["half_steps"]),
    "r4_research": RunRequest(**P, n_x=16, n_y=16, N=2, r4_mode=R4_RESEARCH_STOP, transform="fft", half_steps=D["half_steps"]),
}
summary = {}
for tag, request in cases.items():
    outcome = kkvim_adapter.execute(request)
    paths = export.write_result(outcome, out / tag)
    summary[tag] = {"status": outcome.status, "is_error": outcome.is_error, "files": sorted(p.name for p in paths.values())}
mods = {k: getattr(v, "__file__", None) for k, v in sys.modules.items() if k.split(".")[0] in ("kkvim", "kkvim_gui")}
assert all(Path(p).resolve().is_relative_to(BASE) for p in mods.values() if p), mods
summary["_modules_from_base"] = True
(out / "schema1_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
print(json.dumps(summary, indent=1))
