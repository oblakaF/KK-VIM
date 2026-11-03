"""P2.2: plate scheme drawing (GUI_DESIGN.md, 5.2)."""

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

from PySide6.QtCore import QRectF  # noqa: E402

from kkvim_gui.main_window.plate_scheme import AXES_NOTE, PlateScheme  # noqa: E402
from kkvim_gui.model.setup_model import SetupModel  # noqa: E402
from kkvim_gui.session import Session  # noqa: E402


@pytest.fixture
def scheme(qtbot, qapp):
    session = Session(settings=None)
    model = SetupModel()
    w = PlateScheme(session, model)
    qtbot.addWidget(w)
    w.resize(480, 520)
    w.show()
    qtbot.waitExposed(w)
    return w


def render(w):
    image = w.grab()
    assert not image.isNull()
    return w.layout_info


@pytest.mark.parametrize(("a", "b"), [("1", "1"), ("2", "1"), ("1", "3"), ("50", "1"), ("1", "50"), ("1e6", "1")])
def test_plan_keeps_true_proportion_inside_the_widget(scheme, a, b):
    scheme.model.set_text("a", a)
    scheme.model.set_text("b", b)
    lay = render(scheme)
    plan = lay.plan
    assert plan.width() / plan.height() == pytest.approx(float(a) / float(b), rel=1e-6)
    assert QRectF(scheme.rect()).contains(plan)
    assert lay.side is not None and QRectF(scheme.rect()).contains(lay.side)
    assert lay.side.top() > plan.bottom()  # side view under the plan, no overlap


@pytest.mark.parametrize("text", ["", "-", "0,", "1e-", "abc", "0", "1e400"])
def test_unfinished_or_invalid_input_never_crashes(scheme, text):
    for key in ("a", "b", "h", "q"):
        scheme.model.set_text(key, text)
        render(scheme)
    assert not scheme.data().valid or text == "0,"


def test_waiting_state_has_no_geometry(scheme):
    scheme.model.set_text("h", "abc")
    lay = render(scheme)
    assert lay.side is None
    assert "Толщина h" in scheme.accessibleDescription()


def test_scheme_follows_the_model(scheme):
    scheme.model.set_text("a", "3")
    scheme.model.set_choice(supports="CCCC")
    assert scheme.data().a_label == "a = 3 m"
    assert scheme.data().supports == "CCCC"
    assert "CCCC" in scheme.accessibleDescription()


def test_thin_plate_side_view_is_marked_not_to_scale(scheme):
    lay = render(scheme)  # h = 10 mm, a = 2 m: thickness must be exaggerated
    assert lay.side_to_scale is False


def test_both_themes_and_small_sizes(scheme):
    for theme in ("dark", "light"):
        scheme.session.set_theme(theme, persist=False)
        for w, h in ((260, 260), (480, 520), (900, 700)):
            scheme.resize(w, h)
            render(scheme)


def test_axes_and_load_sign_are_not_interpreted():
    assert "Оси и направление нагрузки не показаны" in AXES_NOTE


def test_setup_scheme_shows_current_input_not_the_run_snapshot(qtbot, qapp):
    from kkvim_gui.app import GuiApplication

    g = GuiApplication(Session(settings=None), demo=True)
    qtbot.addWidget(g.main_window)
    w = g.main_window
    w.set_state("setup")
    w.setup_model.set_text("a", "3")
    w.setup_view.try_run()
    snapshot_rows = w.run_rows()
    w.set_state("setup")
    w.setup_model.set_text("a", "7")
    assert w.setup_view.plate.data().a_label == "a = 7 m"   # scheme: current input
    assert w.run_rows() == snapshot_rows                     # run and analysis: snapshot
    assert "a 3 m" in dict(snapshot_rows)["Геометрия"]
    g.help.close()
