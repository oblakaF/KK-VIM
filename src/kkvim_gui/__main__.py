"""Run the GUI with `python -m kkvim_gui`."""

import sys

from kkvim_gui.app import main

# The guard is required: the computing process (multiprocessing, "spawn") imports the main module again.
if __name__ == "__main__":
    sys.exit(main())
