"""Infrastructure check: the GUI stays outside the scientific packages (docs/gui/GUI_DESIGN.md, 3.1).

- kkvim and kkvim_reference never import kkvim_gui;
- kkvim_gui never imports kkvim_reference.

Static analysis with ast, like tests/test_package_independence.py; no Qt needed.
"""

import ast
from pathlib import Path

import pytest

SRC_DIR = Path(__file__).resolve().parent.parent / "src"

FORBIDDEN = (
    ("kkvim", "kkvim_gui"),
    ("kkvim_reference", "kkvim_gui"),
    ("kkvim_gui", "kkvim_reference"),
)


def imported_top_level(path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            yield node.lineno, node.module.split(".")[0]
        elif isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            if name in ("import_module", "__import__") and node.args:
                arg = node.args[0]
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    yield node.lineno, arg.value.split(".")[0]


@pytest.mark.parametrize(("package", "forbidden"), FORBIDDEN)
def test_gui_and_scientific_packages_are_separated(package, forbidden):
    package_dir = SRC_DIR / package
    if not package_dir.is_dir():
        pytest.skip(f"src/{package}/ does not exist yet")
    violations = [
        f"{path}:{line}: {module}"
        for path in sorted(package_dir.rglob("*.py"))
        for line, module in imported_top_level(path)
        if module == forbidden
    ]
    assert violations == []


def test_only_the_adapter_imports_kkvim():
    """GUI_DESIGN.md, 3.1: kkvim_gui imports kkvim only in the backend adapter module."""
    package_dir = SRC_DIR / "kkvim_gui"
    importers = sorted(
        path.relative_to(package_dir).as_posix()
        for path in package_dir.rglob("*.py")
        if any(module == "kkvim" for _, module in imported_top_level(path))
    )
    assert importers == ["backend/kkvim_adapter.py"]


def test_the_backend_is_qt_free():
    """The computing process imports only the Qt-free backend (GUI_DESIGN.md, 3.6)."""
    for path in sorted((SRC_DIR / "kkvim_gui" / "backend").rglob("*.py")):
        assert not any(module in ("PySide6", "pyqtgraph") for _, module in imported_top_level(path)), path
