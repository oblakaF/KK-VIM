"""The Windows launcher KK-VIM-GUI.cmd and its build script scripts\\windows\\build_gui.ps1 (GUI-WIN-RUNNABLE-P1).

Fast checks (always on Windows): the file format of the launcher, that it checks the installation before every start and
starts nothing after a failed or impossible build, one build path, the full GUI as the entry point, the fingerprint, the
-Check states and the refusal without uv. The real builds (first build, an unchanged second start, a rebuild after a
source change, a failed build, a locked installation, user files kept) run only with KKVIM_WINDOWS_BUILD_TESTS=1: they
build a copy of the sources whose path contains spaces under scratch\\ (ignored by git) and take a few minutes.
"""

import inspect
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "KK-VIM-GUI.cmd"
SCRIPT = ROOT / "scripts" / "windows" / "build_gui.ps1"
windows = pytest.mark.skipif(sys.platform != "win32", reason="Windows launcher")
STAMP = "kkvim-gui-build.json"
EXE = Path("Scripts") / "kkvim-gui.exe"


def powershell(*args, env=None, script=SCRIPT):
    return subprocess.run(["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script), *map(str, args)],
                          capture_output=True, text=True, env=env, timeout=1800, check=False)


def launch(launcher, *args, env=None):
    """cmd.exe /c ""<launcher>" args": the quoting that keeps a path with spaces whole."""
    command = f'cmd.exe /c ""{launcher}" {" ".join(args)}"'
    return subprocess.run(command, capture_output=True, text=True, env=env, timeout=1800, check=False)


def no_pause(**extra):
    return {**os.environ, "KKVIM_GUI_NO_PAUSE": "1", **extra}


# --- the files ---------------------------------------------------------------------------------------------------------------

def test_the_launcher_is_ascii_with_crlf_line_ends():
    data = LAUNCHER.read_bytes()
    assert all(b < 128 for b in data) and data.count(b"\n") == data.count(b"\r\n") > 0
    assert "*.cmd text eol=crlf" in (ROOT / ".gitattributes").read_text(encoding="utf-8")
    assert all(b < 128 for b in SCRIPT.read_bytes())  # Windows PowerShell 5.1 reads a BOM-less script as ANSI


def test_the_launcher_checks_the_installation_before_every_start_and_never_starts_after_a_failure():
    text = LAUNCHER.read_text(encoding="ascii")
    check = text.index(r'scripts\windows\build_gui.ps1" -IfStale -Target "%TARGET%"')
    guard = text.index('if not "%CODE%"=="0" goto failed')
    start = text.index('start "" "%EXE%" %*')
    assert check < guard < start
    assert text.index(":failed") > start and "exit /b %CODE%" in text
    commands = " ".join(line for line in text.splitlines() if not line.lower().startswith("rem "))
    assert "--demo" not in commands and "preview" not in commands.lower()  # the normal GUI, never the demo or the preview
    assert r'set "TARGET=%ROOT%dist\kkvim-gui"' in text


def test_there_is_one_build_path():
    assert sorted(p.name for p in SCRIPT.parent.glob("*.ps1")) == ["build_gui.ps1"]
    script = SCRIPT.read_text(encoding="ascii")
    assert "uv build --wheel" in script and "uv export --locked" in script and "--relocatable" in script


def test_the_entry_point_is_the_full_gui():
    from kkvim_gui import app

    assert 'kkvim-gui = "kkvim_gui.app:main"' in (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    source = inspect.getsource(app.main)
    assert "GuiApplication(" in source and "preview" not in source


# --- fingerprint and -Check ------------------------------------------------------------------------------------------------

@windows
def test_the_fingerprint_is_stable():
    first, second = powershell("-Fingerprint"), powershell("-Fingerprint")
    assert first.returncode == second.returncode == 0
    value = first.stdout.strip()
    assert value == second.stdout.strip() and len(value) == 64 and all(c in "0123456789ABCDEF" for c in value)


def fake_installation(folder, fingerprint=None):
    (folder / EXE).parent.mkdir(parents=True, exist_ok=True)
    (folder / EXE).write_bytes(b"not a program")
    if fingerprint is not None:
        (folder / STAMP).write_text(json.dumps({"fingerprint": fingerprint, "built": "test"}), encoding="utf-8")


@windows
def test_check_tells_missing_stale_and_current(tmp_path):
    target = tmp_path / "install with spaces"
    assert powershell("-Check", "-Target", target).returncode == 3
    fake_installation(target)
    result = powershell("-Check", "-Target", target)
    assert result.returncode == 3 and "without a build stamp" in result.stdout  # e.g. the old N = 1 preview
    fake_installation(target, "0" * 64)
    result = powershell("-Check", "-Target", target)
    assert result.returncode == 3 and "other sources" in result.stdout
    fake_installation(target, powershell("-Fingerprint").stdout.strip())
    assert powershell("-Check", "-Target", target).returncode == 0


@windows
def test_without_uv_a_stale_installation_is_kept_and_not_started(tmp_path):
    target = tmp_path / "install with spaces"
    fake_installation(target, "0" * 64)
    before = {p.relative_to(target): p.read_bytes() for p in target.rglob("*") if p.is_file()}
    system = Path(os.environ.get("SystemRoot", r"C:\Windows"))
    env = no_pause(KKVIM_GUI_TARGET=str(target),
                   PATH=f"{system / 'System32'};{system / 'System32' / 'WindowsPowerShell' / 'v1.0'}")
    result = launch(LAUNCHER, "--help", env=env)
    assert result.returncode == 2
    assert "uv was not found" in result.stdout and "was not started" in result.stdout and "not started" in result.stdout
    assert {p.relative_to(target): p.read_bytes() for p in target.rglob("*") if p.is_file()} == before
    assert not Path(f"{target}.staging").exists()


# --- real builds (opt-in) --------------------------------------------------------------------------------------------------

COPY_ITEMS = ("KK-VIM-GUI.cmd", "pyproject.toml", "uv.lock", "README.md", "scripts", "src")


def source_copy(base):
    """The files the build reads, copied to a path with spaces (no git clone, no worktree)."""
    copy = base / "copy with spaces" / "KK VIM"
    if copy.exists():
        shutil.rmtree(copy)
    copy.mkdir(parents=True)
    for name in COPY_ITEMS:
        source = ROOT / name
        if source.is_dir():
            shutil.copytree(source, copy / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        else:
            shutil.copy2(source, copy / name)
    return copy


def snapshot(folder):
    return {p.relative_to(folder): p.read_bytes() for p in folder.rglob("*") if p.is_file()}


@windows
@pytest.mark.skipif(os.environ.get("KKVIM_WINDOWS_BUILD_TESTS") != "1", reason="real builds: set KKVIM_WINDOWS_BUILD_TESTS=1")
def test_real_builds_first_start_unchanged_start_rebuild_failure_and_lock():
    copy = source_copy(ROOT / "scratch" / "launcher-tests")
    launcher, script, target = copy / "KK-VIM-GUI.cmd", copy / "scripts" / "windows" / "build_gui.ps1", copy / "dist" / "kkvim-gui"
    user = {copy / "my results" / "keep.txt": b"user result", copy / "dist" / "my results" / "keep.txt": b"user file in dist"}
    for path, data in user.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    env = no_pause()

    first = launch(launcher, "--help", env=env)  # first start: nothing installed
    assert first.returncode == 0, first.stdout + first.stderr
    assert "not installed" in first.stdout and (target / EXE).exists() and (target / STAMP).exists()
    built = json.loads((target / STAMP).read_text(encoding="utf-8-sig"))
    assert built["fingerprint"] == powershell("-Fingerprint", script=script).stdout.strip()
    mtime = (target / EXE).stat().st_mtime_ns

    second = launch(launcher, "--help", env=env)  # unchanged sources: the installation is used as it is
    assert second.returncode == 0 and "is current" in second.stdout and (target / EXE).stat().st_mtime_ns == mtime

    init = copy / "src" / "kkvim_gui" / "__init__.py"  # a controlled source change
    init.write_text(init.read_text(encoding="utf-8") + "\n# launcher test: source change\n", encoding="utf-8")
    assert powershell("-Check", script=script).returncode == 3
    third = launch(launcher, "--help", env=env)
    assert third.returncode == 0 and "other sources" in third.stdout, third.stdout + third.stderr
    rebuilt = json.loads((target / STAMP).read_text(encoding="utf-8-sig"))
    assert rebuilt["fingerprint"] != built["fingerprint"] and powershell("-Check", script=script).returncode == 0
    probe = subprocess.run([str(target / "Scripts" / "python.exe"), "-c", "import kkvim_gui, inspect; print(inspect.getsource(kkvim_gui))"],
                           capture_output=True, text=True, check=False)
    assert "launcher test: source change" in probe.stdout  # the installed program is the changed source

    installed = snapshot(target)
    pyproject = copy / "pyproject.toml"
    good = pyproject.read_bytes()
    pyproject.write_bytes(good + b"\n[broken\n")  # a failed build
    failed = launch(launcher, "--help", env=env)
    assert failed.returncode == 1 and "build failed" in failed.stdout and "unchanged" in failed.stdout
    assert snapshot(target) == installed and not Path(f"{target}.staging").exists() and not Path(f"{target}.previous").exists()
    pyproject.write_bytes(good)

    init.write_text(init.read_text(encoding="utf-8") + "# second change\n", encoding="utf-8")
    with open(target / "Scripts" / "python.exe", "rb"):  # the installation in use: its folder cannot be replaced
        locked = launch(launcher, "--help", env=env)
    assert locked.returncode == 1 and "in use or locked" in locked.stdout
    assert snapshot(target) == installed and not Path(f"{target}.staging").exists()

    for path, data in user.items():
        assert path.read_bytes() == data
    shutil.rmtree(copy.parent)
