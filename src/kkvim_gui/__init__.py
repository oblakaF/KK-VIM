"""KK-VIM graphical interface (docs/gui/GUI_DESIGN.md).

Interface only: no mathematical model, formulas or solver code live here. The
scientific packages kkvim and kkvim_reference never import this package.

Importing kkvim_gui itself does not import Qt, so Qt-free parts (for example the
figure text catalog) can be used and tested without the optional "gui" extra.
"""

from importlib.metadata import PackageNotFoundError, version


def program_version():
    """Installed version of the kkvim distribution, or "dev" when not installed."""
    try:
        return version("kkvim")
    except PackageNotFoundError:
        return "dev"
