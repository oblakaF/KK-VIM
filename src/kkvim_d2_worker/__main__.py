"""``python -m kkvim_d2_worker``: one JSON request on stdin, one JSON answer on stdout (see the package text)."""

import sys

from kkvim_d2_worker import main

sys.exit(main())
