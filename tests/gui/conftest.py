"""GUI tests run without a visible display (QT_QPA_PLATFORM=offscreen) unless set otherwise.

Qt window tests need the "gui" extra and the "gui-test" group:
    uv sync --extra gui --group gui-test
Without them those modules are skipped (pytest.importorskip).
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
