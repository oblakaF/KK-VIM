"""Project file .kkv, Qt-free (GUI_DESIGN.md, 3.3; P2.3; GUI-PROJECT-KKV-P23-01).

The saved setup opens as the same SetupState, hence the same solver request and series requests; entered texts, units,
research and display choices are kept; every defective file is refused before anything is changed; a failed write
leaves the previous file and no temporary file. The window workflow: test_project_window.py.
"""

import json
import os
import subprocess
import sys
from dataclasses import fields, replace

import pytest

from kkvim_gui.backend.request import R4_RESEARCH_FIXED, R4_RESEARCH_STOP
from kkvim_gui.model import project_file
from kkvim_gui.model.project_file import ProjectError
from kkvim_gui.model.setup_state import (SetupState, to_config, to_request, to_series_requests,
                                         validate)


def plate(state, **texts):
    base = {"a": "2", "b": "1", "h": "10", "E": "210", "nu": "0,3", "q": "5", "n_x": "16", "n_y": "8"}
    for key, text in {**base, **texts}.items():
        state = state.with_text(key, text)
    return state


def configurations():
    s = SetupState()
    return {
        "ssss_n1_units": plate(s, a="2000", b="1000", h="0,01", E="210000", q="5000", nu="0.3", tol_inf="1e-9")
        .with_unit("a", "mm").with_unit("b", "mm").with_unit("h", "m").with_unit("E", "MPa").with_unit("q", "Pa")
        .with_choice(strict=True, display={"kkvim", 1, 3, "navier"}),
        "r4_n2_normative": plate(s).with_choice(N=2),
        "r4_n3_research_stop": plate(s).with_choice(N=3).with_choice(r4_mode=R4_RESEARCH_STOP),
        "r4_n3_research_fixed_dense": plate(s, half_steps="7").with_choice(N=3)
        .with_choice(r4_mode=R4_RESEARCH_FIXED, transform="dense"),
        "ssss_n5_research": plate(s, half_steps="4").with_choice(N=5).with_choice(ssss_research=True),
        "ssss_n6_research_not_chosen": plate(s).with_choice(N=6),  # saved as it is: the run stays refused (no research)
        "cccc_n1_s5_no_d2": plate(s).with_choice(supports="CCCC").with_choice(cccc_half_steps=5, with_d2=False),
        "cccc_n3_s2_guarded": plate(s).with_choice(supports="CCCC", N=3).with_choice(cccc_half_steps=2,
                                                                                     display={"kkvim", 1, "d2"}),
        "series_ssss_n1": plate(s, q_series="1; 2,5; 5").with_choice(load_mode="series", display=set()),
        "series_cccc_n2": plate(s, q_series="2 4 8").with_unit("q", "Pa").with_choice(supports="CCCC", N=2)
        .with_choice(load_mode="series"),
        "invalid_form": plate(s, a="", nu="abc"),
    }


CONFIGS = configurations()


def save_and_open(tmp_path, state, name="p"):
    path = tmp_path / f"{name}.kkv"
    project_file.write(path, state)
    return path, project_file.read(path)


def test_schema_covers_every_setup_field():
    assert project_file.SETUP_KEYS == tuple(f.name for f in fields(SetupState))


@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_round_trip_gives_the_same_setup_and_the_same_requests(tmp_path, name):
    state = CONFIGS[name]
    path, project = save_and_open(tmp_path, state, name)
    assert project.state == state and project.name == name
    assert to_request(project.state) == to_request(state)
    assert to_series_requests(project.state) == to_series_requests(state)
    assert to_config(project.state) == to_config(state) and validate(project.state) == validate(state)
    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["format"] == "kkvim-project" and document["schema_version"] == 1
    assert set(document) == {"format", "schema_version", "project", "setup", "run_config"}  # no results embedded
    assert document["setup"]["texts"] == state.texts and document["setup"]["units"] == state.units
    project_file.write(tmp_path / "again.kkv", project.state)  # saved again after opening: still the same setup
    assert project_file.read(tmp_path / "again.kkv").state == state


def test_requests_are_really_built_for_the_runs_checked():
    for name in ("ssss_n1_units", "r4_n2_normative", "r4_n3_research_fixed_dense", "ssss_n5_research",
                 "cccc_n1_s5_no_d2", "cccc_n3_s2_guarded"):
        assert to_request(CONFIGS[name]) is not None, name
    assert len(to_series_requests(CONFIGS["series_ssss_n1"])) == 3
    assert [r.q for r in to_series_requests(CONFIGS["series_cccc_n2"])] == [2.0, 4.0, 8.0]
    assert to_request(CONFIGS["ssss_n6_research_not_chosen"]) is None  # research is never switched on by saving


def test_entered_text_units_research_and_display_are_kept_verbatim(tmp_path):
    state = CONFIGS["ssss_n1_units"]
    _, project = save_and_open(tmp_path, state)
    st = project.state
    assert (st.texts["h"], st.units["h"], st.texts["nu"], st.texts["E"], st.units["E"]) == ("0,01", "m", "0.3", "210000",
                                                                                          "MPa")
    assert st.display == frozenset({"kkvim", 1, 3, "navier"}) and st.strict
    assert to_request(st).a == 2.0 and to_request(st).E == 210e9  # converted for the solver only, as before
    _, research = save_and_open(tmp_path, CONFIGS["r4_n3_research_fixed_dense"], "r")
    assert (research.state.r4_mode, research.state.transform, research.state.texts["half_steps"]) == (
        R4_RESEARCH_FIXED, "dense", "7")
    _, cccc = save_and_open(tmp_path, CONFIGS["cccc_n3_s2_guarded"], "c")
    assert (cccc.state.cccc_half_steps, cccc.state.with_d2, cccc.state.display) == (2, True, {"kkvim", 1, "d2"})
    _, empty = save_and_open(tmp_path, CONFIGS["series_ssss_n1"], "s")
    assert empty.state.display == frozenset() and empty.state.texts["q_series"] == "1; 2,5; 5"


def test_an_unfinished_form_is_saved_as_entered_without_a_run_config(tmp_path):
    path, project = save_and_open(tmp_path, CONFIGS["invalid_form"])
    assert json.loads(path.read_text(encoding="utf-8"))["run_config"] is None
    assert project.state.texts["a"] == "" and project.state.texts["nu"] == "abc"


# --- defective files -----------------------------------------------------------------------------------------------

def document(state=None):
    return json.loads(project_file.dumps(state or CONFIGS["ssss_n1_units"], "p"))


def edited(change):
    doc = document()
    change(doc)
    return json.dumps(doc)


def _set(path, value):
    def change(doc):
        target = doc
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
    return change


def _drop(path):
    def change(doc):
        target = doc
        for key in path[:-1]:
            target = target[key]
        del target[path[-1]]
    return change


DEFECTS = {
    "not_json": ("{", "не является JSON"),
    "empty": ("", "не является JSON"),
    "list": ("[]", "не является проектом"),
    "other_format": (edited(_set(["format"], "other")), "не является проектом"),
    "no_schema": (edited(_drop(["schema_version"])), "schema_version"),
    "schema_2": (edited(_set(["schema_version"], 2)), "Версия схемы проекта 2 не поддерживается"),
    "schema_text": (edited(_set(["schema_version"], "1")), "schema_version"),
    "schema_bool": (edited(_set(["schema_version"], True)), "schema_version"),
    "duplicate_key": ('{"format": "kkvim-project", "format": "kkvim-project"}', "Ключ повторяется"),
    "nan": (edited(_set(["setup", "N"], 1)).replace('"N": 1', '"N": NaN'), "NaN"),
    "unknown_top_key": (edited(_set(["results"], {"w": [1.0]})), "неизвестные ключи: results"),
    "unknown_setup_key": (edited(_set(["setup", "q_unit"], "kPa")), "неизвестные ключи: q_unit"),
    "missing_text": (edited(_drop(["setup", "texts", "nu"])), "нет обязательных ключей: nu"),
    "missing_display": (edited(_drop(["setup", "display"])), "нет обязательных ключей: display"),
    "text_not_string": (edited(_set(["setup", "texts", "a"], 2.0)), "setup.texts.a"),
    "unit_not_allowed": (edited(_set(["setup", "units", "a"], "km")), "setup.units.a"),
    "supports": (edited(_set(["setup", "supports"], "CSCS")), "setup.supports"),
    "theory": (edited(_set(["setup", "theory"], "Mindlin")), "setup.theory"),
    "N_7": (edited(_set(["setup", "N"], 7)), "setup.N"),
    "N_bool": (edited(_set(["setup", "N"], True)), "setup.N"),
    "N_text": (edited(_set(["setup", "N"], "1")), "setup.N"),
    "N_float": (edited(_set(["setup", "N"], 1.0)), "setup.N"),
    "mode": (edited(_set(["setup", "r4_mode"], "fast")), "setup.r4_mode"),
    "strict_text": (edited(_set(["setup", "strict"], "yes")), "setup.strict"),
    "d2_number": (edited(_set(["setup", "with_d2"], 1)), "setup.with_d2"),
    "load_mode": (edited(_set(["setup", "load_mode"], "pressure")), "setup.load_mode"),
    "display_flag": (edited(_set(["setup", "display"], [True])), "setup.display"),
    "display_unknown": (edited(_set(["setup", "display"], ["fem"])), "setup.display"),
    "display_twice": (edited(_set(["setup", "display"], [1, 1])), "указана дважды"),
    "display_not_list": (edited(_set(["setup", "display"], "kkvim")), "setup.display"),
    "research_r4_for_n1": (edited(_set(["setup", "r4_mode"], R4_RESEARCH_STOP)), "исследовательский режим R4"),
    "research_n46_for_n1": (edited(_set(["setup", "ssss_research"], True)), "исследование SSSS N = 4–6"),
    "cccc_steps_for_ssss": (edited(_set(["setup", "cccc_half_steps"], 2)), "число полушагов CCCC"),
    "run_config_changed": (edited(_set(["run_config", "material", "nu"], 0.25)), "run_config"),
    "run_config_missing": (edited(_drop(["run_config"])), "нет обязательных ключей: run_config"),
    "texts_changed_only": (edited(_set(["setup", "texts", "q"], "5000,5")), "run_config"),
    "bad_date": (edited(_set(["project", "created"], "вчера")), "project.created"),
    "deep": ("[" * 100000 + "]" * 100000, ""),
}


@pytest.mark.parametrize("name", sorted(DEFECTS))
def test_every_defective_document_is_refused_with_its_reason(name):
    text, reason = DEFECTS[name]
    with pytest.raises(ProjectError, match=reason or None):  # "deep": any reason, never a crash
        project_file.loads(text)


def test_cccc_half_steps_beyond_the_route_are_refused():
    doc = document(CONFIGS["cccc_n3_s2_guarded"])
    doc["setup"]["cccc_half_steps"] = 3  # s >= 3 is not offered for N = 2-6
    doc["run_config"]["method"]["half_steps"] = 3
    with pytest.raises(ProjectError, match="число полушагов 3 не предусмотрено"):
        project_file.loads(json.dumps(doc))


def test_research_cannot_be_moved_to_another_route_in_the_file():
    doc = document(CONFIGS["ssss_n5_research"])
    doc["setup"]["N"] = 2  # an N = 4-6 research choice is never carried to R4
    with pytest.raises(ProjectError, match="исследование SSSS N = 4–6"):
        project_file.loads(json.dumps(doc))


def test_files_that_are_not_projects(tmp_path):
    with pytest.raises(ProjectError, match="не прочитан"):
        project_file.read(tmp_path / "missing.kkv")
    with pytest.raises(ProjectError, match="расширение .kkv"):
        project_file.read(tmp_path / "manifest.json")
    big = tmp_path / "big.kkv"
    big.write_bytes(b" " * (project_file.MAX_BYTES + 1))
    with pytest.raises(ProjectError, match="больше"):
        project_file.read(big)
    latin = tmp_path / "latin.kkv"
    latin.write_bytes("{\"format\": \"\xe9\"}".encode("latin-1"))
    with pytest.raises(ProjectError, match="UTF-8"):
        project_file.read(latin)
    bom = tmp_path / "bom.kkv"  # saved again by a text editor with a byte order mark: still the same project
    bom.write_bytes(b"\xef\xbb\xbf" + project_file.dumps(CONFIGS["r4_n2_normative"], "bom").encode("utf-8"))
    assert project_file.read(bom).state == CONFIGS["r4_n2_normative"]


# --- safe writing --------------------------------------------------------------------------------------------------

def test_a_failed_write_keeps_the_previous_file_and_leaves_no_temporary_file(tmp_path, monkeypatch):
    path = tmp_path / "plate.kkv"
    project_file.write(path, CONFIGS["r4_n2_normative"])
    before = path.read_bytes()
    for target in ("replace", "fsync"):
        with monkeypatch.context() as m:
            m.setattr(project_file.os, target, lambda *a, **k: (_ for _ in ()).throw(OSError(28, "No space left")))
            with pytest.raises(OSError):
                project_file.write(path, CONFIGS["cccc_n3_s2_guarded"])
        assert path.read_bytes() == before, target
        assert sorted(p.name for p in tmp_path.iterdir()) == ["plate.kkv"], target  # no temporary file left
        assert project_file.read(path).state == CONFIGS["r4_n2_normative"]


def test_writing_refuses_other_extensions_and_missing_folders(tmp_path):
    with pytest.raises(ProjectError, match="расширение"):
        project_file.write(tmp_path / "manifest.json", CONFIGS["r4_n2_normative"])
    with pytest.raises(OSError):
        project_file.write(tmp_path / "no" / "such" / "p.kkv", CONFIGS["r4_n2_normative"])
    assert list(tmp_path.iterdir()) == []


def test_a_setup_that_would_not_open_again_is_never_written(tmp_path, monkeypatch):
    real = project_file.loads
    monkeypatch.setattr(project_file, "loads", lambda text: replace(real(text), state=SetupState()))
    with pytest.raises(ProjectError, match="не воспроизводит"):
        project_file.write(tmp_path / "p.kkv", CONFIGS["r4_n2_normative"])
    assert list(tmp_path.iterdir()) == []


def test_project_file_imports_no_qt():
    code = "import sys, kkvim_gui.model.project_file; print(any(m.startswith('PySide6') for m in sys.modules))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                         env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    assert out.stdout.strip() == "False"
