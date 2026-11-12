"""GUI-CORE-N16-P1-QA: visibility of the analysis badge and the run-view limits note for an R3 and a CCCC run (offscreen)."""
import json
import sys
import time
from pathlib import Path

if __name__ == "__main__":
    from PySide6.QtWidgets import QApplication

    from kkvim_gui.app import GuiApplication, create_qapplication
    from kkvim_gui.model.setup_state import N_VALUES
    from kkvim_gui.session import Session

    out = Path(sys.argv[1])
    app = create_qapplication([])
    gui = GuiApplication(Session(settings=None), protocol_dir=out / "protocols_vis")
    main = gui.main_window
    main.show()
    res = {}
    for case, supports, N, s in (("R3", "SSSS", 1, None), ("CCCC_N1_s2", "CCCC", 1, 2)):
        v = main.setup_view
        main.set_state("setup")
        v.supports.set_current(supports)
        v.N.setCurrentIndex(N_VALUES.index(N))
        v.n_x.edit.setText("16")
        v.n_y.edit.setText("16")
        if s:
            v.cccc_half_steps.setCurrentIndex(v.cccc_half_steps.findData(s))
        v.try_run()
        while main.runs.active is not None:
            QApplication.processEvents()
            time.sleep(0.05)
        for _ in range(10):
            QApplication.processEvents()
        rv = main.run_view
        rv.analysis_button.click()
        QApplication.processEvents()
        w = gui.analysis_windows[-1]
        res[case] = {"status": main.runs.runs[-1].outcome.status,
                     "badge_text": w.header.badge.text(), "badge_visible": w.header.badge.isVisibleTo(w),
                     "limits_note_visible": rv.limits_note.isVisibleTo(rv), "run_tag": rv.details_tag.text()}
        w.close()
    main.runs.shutdown()
    (out / "qt_visibility.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))
