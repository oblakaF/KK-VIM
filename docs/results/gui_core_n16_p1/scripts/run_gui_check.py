"""Launcher of gui_check.py with the __main__ guard needed by the spawn child processes of kkvim_gui.backend.process."""
import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parent / "gui_check.py"), run_name="gui_check_main")
