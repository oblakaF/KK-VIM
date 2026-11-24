"""Project file .kkv: the setup of a calculation saved and opened again (GUI_DESIGN.md, 3.3; P2.3). Qt-free.

A project is a UTF-8 JSON document::

    {"format": "kkvim-project", "schema_version": 1,
     "project": {"name", "created", "modified", "program_version"},
     "setup": {the SetupState exactly as entered: texts and units of every field, supports, N, load mode,
               method and research choices, D2 choice, display choice},
     "run_config": to_config(setup) — the parsed RunConfig (numbers with their SI units as entered), or null while
                   the form has errors}

- The entered texts and units are kept unchanged (GUI_DESIGN.md, 11.1-11.2): opening a project gives exactly the
  SetupState that was saved, hence the same RunRequest (``to_request``) and series requests (``to_series_requests``).
- No results: a project never holds fields, histories or reference values (those are the run packages of
  ``backend.export``). The extension .kkv is required, so a project is never written over a result file.
- Reading checks the complete document before anything is returned: the JSON itself (no duplicate keys, no NaN or
  Infinity), the format and schema version, the exact set of keys at every level, the type and the allowed values of
  every entry, the rules of the form (a research choice exists only for its own route, the CCCC half-steps offered for
  N) and that "run_config" is the one the setup gives. Any defect raises ProjectError; the caller's state is untouched.
- Writing is atomic: the document is checked by reading it back, written to a temporary file in the same folder,
  flushed to disk and only then moved over the target (os.replace). A failed write leaves the previous file as it was
  and removes the temporary file.

No scientific default is set here: every value comes from the saved setup; a value that is not there is an error.
"""

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from kkvim_gui import program_version
from kkvim_gui.backend.kkvim_adapter import N_AVAILABLE
from kkvim_gui.backend.request import R4_MODES, R4_NORMATIVE, SUPPORTS, TRANSFORMS
from kkvim_gui.model.setup_state import (DISPLAY_KEYS, FIELDS, LOAD_MODES, THEORY, SetupState, cccc_half_steps_of,
                                         to_config)

FORMAT = "kkvim-project"
SCHEMA_VERSION = 1
SUFFIX = ".kkv"
MAX_BYTES = 1_000_000  # a project is a few kilobytes; a larger file is not a project
MAX_TEXT = 4000  # characters of one entered value
TOP_KEYS = ("format", "schema_version", "project", "setup", "run_config")
PROJECT_KEYS = ("name", "created", "modified", "program_version")
SETUP_KEYS = ("texts", "units", "supports", "theory", "N", "r4_mode", "transform", "strict", "ssss_research",
              "cccc_half_steps", "load_mode", "with_d2", "display")
UNIT_KEYS = tuple(key for key, spec in FIELDS.items() if spec.units)


class ProjectError(Exception):
    """The file is not a valid KK-VIM project (the message is for the user, in Russian)."""


@dataclass(frozen=True)
class Project:
    state: SetupState
    name: str
    created: str
    modified: str
    program_version: str


def setup_document(state):
    """The "setup" entry: every SetupState field as plain JSON values (display in the order of DISPLAY_KEYS)."""
    return {
        "texts": {key: state.texts[key] for key in FIELDS},
        "units": {key: state.units[key] for key in UNIT_KEYS},
        "supports": state.supports, "theory": state.theory, "N": state.N, "r4_mode": state.r4_mode,
        "transform": state.transform, "strict": state.strict, "ssss_research": state.ssss_research,
        "cccc_half_steps": state.cccc_half_steps, "load_mode": state.load_mode, "with_d2": state.with_d2,
        "display": [key for key in DISPLAY_KEYS if key in state.display],
    }


def _plain(value):
    """The JSON form of a value (tuples as lists), as it is read back."""
    return json.loads(json.dumps(value, allow_nan=False))


def dumps(state, name, created=None, now=None):
    """The project document of ``state`` as JSON text."""
    stamp = (now or datetime.now(timezone.utc).astimezone()).isoformat(timespec="seconds")
    document = {
        "format": FORMAT,
        "schema_version": SCHEMA_VERSION,
        "project": {"name": name, "created": created or stamp, "modified": stamp,
                    "program_version": program_version()},
        "setup": setup_document(state),
        "run_config": to_config(state),
    }
    return json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


# --- reading ----------------------------------------------------------------------------------------------------------

def _no_duplicates(pairs):
    keys = [key for key, _ in pairs]
    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    if duplicates:
        raise ProjectError(f"Ключ повторяется: {', '.join(duplicates)}.")
    return dict(pairs)


def _no_constant(name):
    raise ProjectError(f"Недопустимое число {name} (в файле проекта допускаются только конечные числа).")


def _object(value, where, keys):
    if not isinstance(value, dict):
        raise ProjectError(f"«{where}»: ожидается объект JSON.")
    missing = [key for key in keys if key not in value]
    extra = sorted(key for key in value if key not in keys)
    if missing:
        raise ProjectError(f"«{where}»: нет обязательных ключей: {', '.join(missing)}.")
    if extra:
        raise ProjectError(f"«{where}»: неизвестные ключи: {', '.join(extra)}.")
    return value


def _string(value, where, limit=MAX_TEXT):
    if not isinstance(value, str):
        raise ProjectError(f"«{where}»: ожидается строка.")
    if len(value) > limit:
        raise ProjectError(f"«{where}»: строка длиннее {limit} символов.")
    return value


def _flag(value, where):
    if not isinstance(value, bool):
        raise ProjectError(f"«{where}»: ожидается true или false.")
    return value


def _integer(value, where):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProjectError(f"«{where}»: ожидается целое число.")
    return value


def _one_of(value, where, allowed):
    if isinstance(value, bool) or value not in allowed:
        shown = ", ".join(map(str, allowed))
        raise ProjectError(f"«{where}»: недопустимое значение {value!r}; допускается: {shown}.")
    return value


def _stamp(value, where):
    _string(value, where, 64)
    try:
        datetime.fromisoformat(value)
    except ValueError:
        raise ProjectError(f"«{where}»: ожидается дата ISO 8601.") from None
    return value


def _display(value):
    if not isinstance(value, list):
        raise ProjectError("«setup.display»: ожидается список кривых.")
    keys = []
    for item in value:
        if isinstance(item, bool) or item not in DISPLAY_KEYS:  # True == 1 in Python: a flag is not the cycle k = 1
            raise ProjectError(f"«setup.display»: неизвестная кривая {item!r}; допускается: "
                               f"{', '.join(map(str, DISPLAY_KEYS))}.")
        if item in keys:
            raise ProjectError(f"«setup.display»: кривая {item!r} указана дважды.")
        keys.append(item)
    return frozenset(keys)


def _state(setup):
    """SetupState of the "setup" entry, with every value and every rule of the form checked."""
    _object(setup, "setup", SETUP_KEYS)
    texts = _object(setup["texts"], "setup.texts", tuple(FIELDS))
    for key in FIELDS:
        _string(texts[key], f"setup.texts.{key}")
    units = _object(setup["units"], "setup.units", UNIT_KEYS)
    for key in UNIT_KEYS:
        _one_of(units[key], f"setup.units.{key}", FIELDS[key].units)
    state = SetupState(
        texts={key: texts[key] for key in FIELDS},
        units={key: units[key] for key in UNIT_KEYS},
        supports=_one_of(setup["supports"], "setup.supports", SUPPORTS),
        theory=_one_of(setup["theory"], "setup.theory", (THEORY,)),
        N=_one_of(_integer(setup["N"], "setup.N"), "setup.N", tuple(N_AVAILABLE)),
        r4_mode=_one_of(setup["r4_mode"], "setup.r4_mode", R4_MODES),
        transform=_one_of(setup["transform"], "setup.transform", TRANSFORMS),
        strict=_flag(setup["strict"], "setup.strict"),
        ssss_research=_flag(setup["ssss_research"], "setup.ssss_research"),
        cccc_half_steps=_integer(setup["cccc_half_steps"], "setup.cccc_half_steps"),
        load_mode=_one_of(setup["load_mode"], "setup.load_mode", LOAD_MODES),
        with_d2=_flag(setup["with_d2"], "setup.with_d2"),
        display=_display(setup["display"]),
    )
    # The rules of the form: a research choice belongs to its own route only and is reset with N and the supports
    # (D-048.5, D-056.7), so a file with a research choice for another route was not saved by the program.
    scope = state.scope
    if state.r4_mode != R4_NORMATIVE and scope != "r4":
        raise ProjectError(f"Несогласованный файл: исследовательский режим R4 ({state.r4_mode}) указан для "
                           f"{state.supports}, N = {state.N}; он допустим только для SSSS, N = 2–3.")
    if state.ssss_research and scope != "r46":
        raise ProjectError(f"Несогласованный файл: исследование SSSS N = 4–6 указано для {state.supports}, N = {state.N}.")
    if state.supports == "CCCC":
        offered = cccc_half_steps_of(state.N)
        if state.cccc_half_steps not in offered:
            raise ProjectError(f"Несогласованный файл: CCCC, N = {state.N} — число полушагов {state.cccc_half_steps} не "
                               f"предусмотрено (допускается: {', '.join(map(str, offered))}).")
    elif state.cccc_half_steps != 1:
        raise ProjectError(f"Несогласованный файл: число полушагов CCCC = {state.cccc_half_steps} указано для "
                           f"{state.supports}.")
    return state


def loads(text):
    """Project of a JSON text; ProjectError if anything in it is not valid."""
    try:
        document = json.loads(text, object_pairs_hook=_no_duplicates, parse_constant=_no_constant)
    except json.JSONDecodeError as exc:
        raise ProjectError(f"Файл не является JSON: {exc.msg} (строка {exc.lineno}, столбец {exc.colno}).") from None
    except RecursionError:
        raise ProjectError("Файл не является проектом КК-ВИМ: слишком глубокая вложенность.") from None
    if not isinstance(document, dict) or document.get("format") != FORMAT:
        raise ProjectError("Файл не является проектом КК-ВИМ (нет признака формата «kkvim-project»).")
    version = document.get("schema_version")
    if isinstance(version, bool) or not isinstance(version, int):
        raise ProjectError("Нет номера версии схемы проекта (schema_version).")
    if version != SCHEMA_VERSION:
        raise ProjectError(f"Версия схемы проекта {version} не поддерживается; эта программа читает версию "
                           f"{SCHEMA_VERSION}.")
    _object(document, "файл", TOP_KEYS)
    meta = _object(document["project"], "project", PROJECT_KEYS)
    state = _state(document["setup"])
    if _plain(to_config(state)) != document["run_config"]:
        raise ProjectError("Несогласованный файл: «run_config» не соответствует введённым значениям «setup».")
    return Project(state=state, name=_string(meta["name"], "project.name", 255),
                   created=_stamp(meta["created"], "project.created"),
                   modified=_stamp(meta["modified"], "project.modified"),
                   program_version=_string(meta["program_version"], "project.program_version", 64))


def _check_suffix(path):
    if path.suffix.lower() != SUFFIX:
        raise ProjectError(f"Файл проекта должен иметь расширение {SUFFIX}: {path.name}")


def read(path):
    """Project of a .kkv file; ProjectError for a missing, unreadable, too large or invalid file."""
    path = Path(path)
    _check_suffix(path)
    try:
        with open(path, "rb") as f:
            data = f.read(MAX_BYTES + 1)
    except OSError as exc:
        raise ProjectError(f"Файл не прочитан: {exc.strerror or exc}.") from None
    if len(data) > MAX_BYTES:
        raise ProjectError(f"Файл больше {MAX_BYTES // 1000} кБ: это не файл проекта.")
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise ProjectError("Файл не в кодировке UTF-8.") from None
    return loads(text)


def write(path, state, name=None, created=None):
    """Write ``state`` to a .kkv file atomically; returns the Project as written.

    The text is read back and must give the same state before anything is written. The previous file (if any) is
    replaced only by a complete, flushed temporary file in the same folder; on any failure it is left as it was."""
    path = Path(path)
    _check_suffix(path)
    text = dumps(state, name or path.stem, created)
    project = loads(text)
    if project.state != state:  # never write a project that would not open as this setup
        raise ProjectError("Внутренняя ошибка: проект не воспроизводит текущие параметры; файл не записан.")
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(text.encode("utf-8"))
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    return project
