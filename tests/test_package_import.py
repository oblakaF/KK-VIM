"""Infrastructure check: kkvim is installed in the environment and importable.

The test suite runs without PYTHONPATH=src, so the import below succeeds only
when the package has been installed (for example, by `uv sync`).
"""

import importlib
import importlib.metadata
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"


def test_package_is_installed():
    assert importlib.metadata.distribution("kkvim").metadata["Name"] == "kkvim"


def test_package_imports_from_this_checkout():
    module = importlib.import_module("kkvim")
    assert Path(module.__file__).resolve().parent == SRC_DIR / "kkvim"
