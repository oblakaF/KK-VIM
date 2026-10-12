"""Infrastructure check: kkvim and kkvim_reference never import each other (D-028).

The independent reference code must not call the main KK-VIM solver, and the
solver must not depend on the references (AGENTS.md, section 4). Imports are
found by static analysis of the source files with ast: import statements and
calls to importlib.import_module / __import__ with a literal module name.
"""

import ast
from pathlib import Path

import pytest

SRC_DIR = Path(__file__).resolve().parent.parent / "src"

# package -> top-level package it must not import
FORBIDDEN = {
    "kkvim": "kkvim_reference",
    "kkvim_reference": "kkvim",
}


def imported_modules(tree):
    """Yield (line, module name) for every absolute import found in the tree."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom):
            # Relative imports (level > 0) cannot leave their own top-level package.
            if node.level == 0 and node.module:
                yield node.lineno, node.module
        elif isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            if name in ("import_module", "__import__") and node.args:
                arg = node.args[0]
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    yield node.lineno, arg.value


def find_violations(package_dir, forbidden):
    """Return "file:line: module" entries where package_dir imports `forbidden`."""
    violations = []
    for path in sorted(package_dir.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for line, module in imported_modules(tree):
            if module.split(".")[0] == forbidden:
                violations.append(f"{path}:{line}: {module}")
    return violations


@pytest.mark.parametrize(("package", "forbidden"), sorted(FORBIDDEN.items()))
def test_packages_do_not_import_each_other(package, forbidden):
    package_dir = SRC_DIR / package
    if not package_dir.is_dir():
        pytest.skip(f"src/{package}/ does not exist yet")
    assert find_violations(package_dir, forbidden) == []


@pytest.mark.parametrize(
    "source",
    [
        "import kkvim\n",
        "import kkvim.core as core\n",
        "from kkvim import solver\n",
        "from kkvim.sub import x\n",
        "def f():\n    import kkvim\n",
        "import importlib\nimportlib.import_module('kkvim.core')\n",
        "__import__('kkvim')\n",
    ],
)
def test_checker_detects_forbidden_import(tmp_path, source):
    (tmp_path / "module.py").write_text(source, encoding="utf-8")
    assert len(find_violations(tmp_path, "kkvim")) == 1


@pytest.mark.parametrize(
    "source",
    [
        "import kkvim_reference\n",
        "import numpy\n",
        "from . import kkvim\n",
        "from .kkvim import x\n",
        "name = 'import kkvim'\n",
    ],
)
def test_checker_accepts_allowed_import(tmp_path, source):
    (tmp_path / "module.py").write_text(source, encoding="utf-8")
    assert find_violations(tmp_path, "kkvim") == []
