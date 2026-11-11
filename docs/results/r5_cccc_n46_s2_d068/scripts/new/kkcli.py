"""Run `python -m kkvim` from the non-Git snapshot (754dcfe + RN-5 certificate of d78962a). usage: python -I kkcli.py <args...>
Asserts that every kkvim module is imported from the snapshot and records the module paths in the env QA_MODULES file if set."""
import json
import os
import sys
from pathlib import Path

SNAP = (Path(__file__).resolve().parent / "snapshot" / "src").resolve()
sys.path.insert(0, str(SNAP))
from kkvim.cli import main  # noqa: E402

code = main(sys.argv[1:])
mods = {k: getattr(v, "__file__", None) for k, v in sys.modules.items() if k == "kkvim" or k.startswith("kkvim.")}
assert all(Path(p).resolve().is_relative_to(SNAP) for p in mods.values() if p), mods
if os.environ.get("QA_MODULES"):
    Path(os.environ["QA_MODULES"]).write_text(json.dumps(mods, indent=1), encoding="utf-8")
sys.exit(code)
