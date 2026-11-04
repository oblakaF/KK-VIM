"""D-059: the D1 campaign execution gate (C1, C2), the runner (C3) and the re-freeze addendum (C4).

No D1 row is executed here: the gate is exercised on a throw-away git repository (positive case and every tamper case), the real
repository must refuse (PENDING_REFREEZE), and the runner is run only as a dry run on non-D1 grids (launch, watchdog, interruption,
restart)."""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = ROOT / "scripts" / "r5_cccc_q4_d1_campaign.py"
READY = ROOT / "scripts" / "r5_cccc_q4_d1_ready.py"
CONTROLS = ROOT / "scripts" / "r5_cccc_q4_d1_ready_controls.py"
BASE_COMMIT = "cef3d3c6de5fb61c552d7ae421f3407fe38977c3"  # main of D-059: every frozen file at its D-058 blob
CANDIDATE_BLOB = "5a794e6e4c3e0031e0bbd496635b27d31bb02e44"  # the PRELAUNCH candidate before the D-058 freeze act (F1)
CSV_BLOB = "76c4443b750c82893c35b50dcabe4d44573c5a5b"
D1_CFG = {"lx": 1.0, "nx": 50, "ly": 1.0, "ny": 50, "N": 1, "D": 1.0, "q": 1.0, "s_max": 1}


def _load(path: Path, name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


c = _load(CAMPAIGN, "_d1_campaign_test")
rd = c.rd()
pl = c.pl()
pf = c.pf()
ct = _load(CONTROLS, "_ct_campaign_test")


@pytest.fixture(autouse=True)
def _fresh_gate(monkeypatch):
    monkeypatch.delenv(pl.CAMPAIGN_AUTH_ENV, raising=False)
    pl._GATE_CACHE.clear()
    yield
    pl._GATE_CACHE.clear()


def _sensor(monkeypatch):
    """Any computation of a row or of a trajectory fails the test."""
    def boom(*a, **k):
        raise AssertionError("computed")
    monkeypatch.setattr(pl, "_compute_row_core", boom)
    monkeypatch.setattr(pl.op8(), "run_trajectory", boom)
    monkeypatch.setattr(pl.op8(), "run_step", boom)


# --- the gate on the real repository: refused ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("signal", [None, "", "yes", "D-058:" + "0" * 40, "D-059:" + "0" * 39, "D-059:" + "0" * 40])
def test_d1_refused_before_computation_without_valid_authorisation(signal, monkeypatch, tmp_path):
    if signal is not None:
        monkeypatch.setenv(pl.CAMPAIGN_AUTH_ENV, signal)
    _sensor(monkeypatch)
    g = pl.campaign_gate()
    assert g["authorized"] is False
    assert g["state"] in ("NOT_REQUESTED", "MALFORMED_SIGNAL", pl.PENDING_REFREEZE)
    if signal == "D-059:" + "0" * 40:
        assert g["state"] == pl.PENDING_REFREEZE  # no owner re-freeze act in docs/DECISIONS.md
    TC = pl.op8().TrajectoryConfig
    with pytest.raises(pl.D1Refused, match="D1_START = BLOCKED"):
        pl.guard_config(TC(**D1_CFG))
    with pytest.raises(pl.D1Refused):
        pl.compute_row(TC(**D1_CFG), 1, np.zeros((49, 1)), np.zeros((49, 1)), np.zeros((49, 1)))
    with pytest.raises(pl.D1Refused):
        rd.bcd().write_paired(TC(**{**D1_CFG, "nx": 80, "ny": 320}), tmp_path / "x")
    with pytest.raises(rd.D1Blocked, match="D1_START = BLOCKED"):
        rd.campaign(ROOT / "scratch" / "never")
    with pytest.raises(c.CampaignRefused, match="D1_START = BLOCKED"):
        c.run_campaign(ROOT / "scratch" / "never")


def test_command_lines_refuse(tmp_path):
    env = {**os.environ, pl.CAMPAIGN_AUTH_ENV: "D-059:" + "0" * 40}
    for cmd in ([str(READY), "campaign", "--out", "scratch/never"], [str(CAMPAIGN), "run", "--out", "scratch/never"]):
        for e in (None, env):
            p = subprocess.run([sys.executable, *cmd], cwd=ROOT, capture_output=True, text=True, env=e)
            assert p.returncode == 2 and "D1_START = BLOCKED" in p.stderr, (cmd, p.stderr[-500:])
    assert not (ROOT / "scratch" / "never").exists()


def test_replay_process_validates_the_gate_itself(monkeypatch):
    """Even with the parent process authorised (patched here), the _row child validates the gate in its own process and refuses."""
    monkeypatch.setattr(pl, "campaign_gate", lambda *a, **k: {"authorized": True, "state": "AUTHORIZED", "reasons": []})
    monkeypatch.setattr(pl, "runtime_defects", lambda *a, **k: [])
    pl.guard_config(pl.op8().TrajectoryConfig(**D1_CFG))  # the parent passes
    spec = {"kind": "arrays", "arrays_npz": "absent.npz", "arrays_sha256": {}, "config": D1_CFG, "s": 1}
    for signal in (None, "D-059:" + "0" * 40):
        if signal:
            monkeypatch.setenv(pl.CAMPAIGN_AUTH_ENV, signal)
        with pytest.raises(pl.D1Refused, match="D1_START = BLOCKED"):
            pl.run_child(spec)


def test_permanent_guards_hold_under_an_authorised_gate(monkeypatch, tmp_path):
    monkeypatch.setattr(pl, "campaign_gate", lambda *a, **k: {"authorized": True, "state": "AUTHORIZED", "reasons": []})
    monkeypatch.setattr(pl, "runtime_defects", lambda *a, **k: [])
    _sensor(monkeypatch)
    TC = pl.op8().TrajectoryConfig
    pl.guard_config(TC(**D1_CFG))
    for N in (2, 3):
        for s in range(2, 9):
            with pytest.raises(pl.RankGuardRefused):
                pl.guard_row(TC(**{**D1_CFG, "N": N, "s_max": 8}), s)
            with pytest.raises(pl.RankGuardRefused):
                pl.compute_row(TC(**{**D1_CFG, "N": N, "s_max": 8}), s, np.zeros((49, N)), np.zeros((49, N)), np.zeros((49, N)))
    with pytest.raises(pl.D1Refused, match="N must be 1, 2 or 3"):
        pl.guard_config(SimpleNamespace(nx=50, ny=50, N=4))  # (OP-8's own config refuses N = 4 earlier)
    with pytest.raises(pl.D1Refused, match="docs/"):
        pl.guard_out(ROOT / "docs" / "results" / "x")
    with pytest.raises(c.CampaignRefused):
        c.run_campaign(ROOT.parent / "outside_scratch_never", plan=[], dry_run=True)  # explicit: tmp_path may lie under scratch/


# --- the gate on a throw-away repository: authorised only with everything in place ---------------------------------------------------

def _git(repo: Path, *args: str) -> str:
    p = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True)
    return p.stdout.strip()


def _base_bytes(path: str) -> bytes:
    p = subprocess.run(["git", "cat-file", "blob", f"{BASE_COMMIT}:{path}"], cwd=ROOT, capture_output=True)
    if p.returncode:
        pytest.skip(f"the base commit {BASE_COMMIT} is not available: gate check NOT performed")
    return p.stdout


def _d058_map(frozen: dict) -> dict:
    return {**{x["path"]: x["blob"] for x in frozen["sha_inventory"]["components"] if x.get("path") and x.get("blob")},
            **{x["path"]: x["blob"] for x in frozen["sha_inventory"]["other_used_files"]}, **frozen["d057"]["tool_blobs"]}


def _commit(repo: Path, msg: str, integrate: bool = True) -> None:
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", msg)
    if integrate:
        _git(repo, "update-ref", pl.INTEGRATION_REF, "HEAD")


ACT = ("D1_CAMPAIGN_REFREEZE_ACT: decision={d}; authorizes=D-059; addendum=docs/results/r5_cccc_q4_d1_campaign/campaign_manifest.json; "
       "addendum_blob={b}; state=ACCEPTED")


def _child_env(repo: Path, **extra) -> dict:
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH",)}
    return {**env, "PYTHONPATH": str(repo / "src"), **extra}


_CHILD_ENV: dict = {}


def _child_environment() -> dict:
    """The environment a child really has (one BLAS thread by the launcher), as the owner's addendum pins it."""
    if not _CHILD_ENV:
        code = (f"import importlib.util, json, sys; s = importlib.util.spec_from_file_location('p', {str(ROOT / pl.SELF)!r}); "
                "p = importlib.util.module_from_spec(s); sys.modules['p'] = p; s.loader.exec_module(p); print(json.dumps(p.environment_record()))")
        p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=c.child_env(), check=True)
        _CHILD_ENV.update(json.loads(p.stdout.strip().splitlines()[-1]))
    return _CHILD_ENV


@pytest.fixture
def gate_repo(tmp_path):
    if shutil.which("git") is None:
        pytest.skip("git not available: gate check NOT performed")
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    for k, v in (("user.email", "qa@example.invalid"), ("user.name", "qa"), ("core.autocrlf", "false")):
        _git(repo, "config", k, v)
    frozen_bytes = _base_bytes(pl.FROZEN_MANIFEST)
    frozen = json.loads(frozen_bytes)
    d058 = _d058_map(frozen)
    changed = ("scripts/r5_cccc_q4_d1_row_pipeline_prelaunch.py", "scripts/r5_cccc_q4_d1_ready.py")
    tools = {"changed": {}, "new": {}, "unchanged": {}}
    for path, blob in d058.items():
        data = (ROOT / path).read_bytes().replace(b"\r\n", b"\n") if path in changed else _base_bytes(path)
        (repo / path).parent.mkdir(parents=True, exist_ok=True)
        (repo / path).write_bytes(data)
        if path in changed:
            tools["changed"][path] = {"d058_blob": blob, "blob": pl.git_blob_id(repo / path)}
        else:
            tools["unchanged"][path] = {"blob": blob}
    runner = "scripts/r5_cccc_q4_d1_campaign.py"
    (repo / runner).write_bytes(CAMPAIGN.read_bytes().replace(b"\r\n", b"\n"))
    tools["new"][runner] = {"blob": pl.git_blob_id(repo / runner)}
    adapter = "scripts/r5_cccc_q4_d1_bc_reference_adapter.py"
    (repo / adapter).write_bytes(_base_bytes(adapter))
    tools["unchanged"][adapter] = {"blob": pl.git_blob_id(repo / adapter)}
    (repo / pl.FROZEN_MANIFEST).parent.mkdir(parents=True, exist_ok=True)
    (repo / pl.FROZEN_MANIFEST).write_bytes(frozen_bytes)
    tools["reference_package"] = {}
    for src in sorted((ROOT / pl.REFERENCE_PACKAGE).rglob("*.py")):
        path = src.relative_to(ROOT).as_posix()
        (repo / path).parent.mkdir(parents=True, exist_ok=True)
        (repo / path).write_bytes(src.read_bytes().replace(b"\r\n", b"\n"))
        tools["reference_package"][path] = {"blob": pl.git_blob_id(repo / path)}
    add = {"decision": "D-059", "state": "PENDING_REFREEZE", "environment": _child_environment(),
           "frozen_baseline": {"d058_frozen_manifest": {"path": pl.FROZEN_MANIFEST, "blob": pl.FROZEN_MANIFEST_BLOB}}, "tools": tools}
    (repo / pl.CAMPAIGN_ADDENDUM).parent.mkdir(parents=True, exist_ok=True)
    (repo / pl.CAMPAIGN_ADDENDUM).write_text(json.dumps(add, indent=1) + "\n", encoding="utf-8", newline="\n")
    dec = "# DECISIONS\n\n### D-059. campaign permission\n\n### D-060. re-freeze act\n\n"
    (repo / pl.DECISIONS_DOC).parent.mkdir(parents=True, exist_ok=True)
    (repo / pl.DECISIONS_DOC).write_text(dec, encoding="utf-8", newline="\n")
    _commit(repo, "base", integrate=False)
    blob = _git(repo, "rev-parse", f"HEAD:{pl.CAMPAIGN_ADDENDUM}")
    (repo / pl.DECISIONS_DOC).write_text(dec + ACT.format(d="D-060", b=blob) + "\n", encoding="utf-8", newline="\n")
    _commit(repo, "act")
    return repo, blob


def _gate(repo: Path, blob: str, signal: str | None = "default") -> dict:
    pl._GATE_CACHE.clear()
    env = {} if signal is None else {pl.CAMPAIGN_AUTH_ENV: f"D-059:{blob}" if signal == "default" else signal}
    return pl.campaign_gate(root=repo, environ=env)


def test_gate_authorises_only_the_complete_record(gate_repo):
    repo, blob = gate_repo
    g = _gate(repo, blob)
    assert g["authorized"] is True and g["state"] == "AUTHORIZED" and g["reasons"] == [], g
    assert g["refreeze_act"] == {"decision": "D-060", "addendum_blob": blob}
    assert _gate(repo, blob, None)["authorized"] is False  # the act alone does not launch
    assert _gate(repo, blob, "D-059:" + "1" * 40)["authorized"] is False  # a signal naming another addendum


def _decisions(repo: Path, text: str) -> None:
    (repo / pl.DECISIONS_DOC).write_text(text, encoding="utf-8", newline="\n")


TAMPER = {
    "dirty tool": (lambda repo, blob: (repo / "scripts/r5_cccc_q4_d1_ready.py").write_bytes(b"# edited\n"), False, "not clean"),
    "committed tool": (lambda repo, blob: (repo / "scripts/r5_cccc_q4_d1_campaign.py").write_bytes(b"# edited\n"), True,
                       "blob at HEAD differs"),
    "unchanged frozen tool": (lambda repo, blob: (repo / "scripts/r5_cccc_q4_d1_c3plus_trajectories.py").write_bytes(b"# x\n"), True,
                              "blob at HEAD differs"),
    "edited addendum": (lambda repo, blob: (repo / pl.CAMPAIGN_ADDENDUM).write_text(
        (repo / pl.CAMPAIGN_ADDENDUM).read_text(encoding="utf-8").replace("PENDING_REFREEZE", "AUTHORIZED"), encoding="utf-8",
        newline="\n"), True, "not the approved"),
    "frozen manifest": (lambda repo, blob: (repo / pl.FROZEN_MANIFEST).write_bytes((repo / pl.FROZEN_MANIFEST).read_bytes() + b" "),
                        True, "D-058 frozen manifest"),
    "no act": (lambda repo, blob: _decisions(repo, "# DECISIONS\n\n### D-059. x\n\n### D-060. y\n"), True, "PENDING_REFREEZE"),
    "act not accepted": (lambda repo, blob: _decisions(repo, "### D-059. x\n### D-060. y\n" +
                                                       ACT.format(d="D-060", b=blob).replace("ACCEPTED", "PROPOSED") + "\n"),
                         True, "PENDING_REFREEZE"),
    "act by D-059 itself": (lambda repo, blob: _decisions(repo, "### D-059. x\n" + ACT.format(d="D-059", b=blob) + "\n"), True,
                            "separate recorded owner decision"),
    "act without its decision": (lambda repo, blob: _decisions(repo, "### D-059. x\n" + ACT.format(d="D-061", b=blob) + "\n"), True,
                                 "separate recorded owner decision"),
    "two acts": (lambda repo, blob: _decisions(repo, "### D-059. x\n### D-060. y\n" + 2 * (ACT.format(d="D-060", b=blob) + "\n")),
                 True, "exactly one"),
    "no D-059": (lambda repo, blob: _decisions(repo, "### D-060. y\n" + ACT.format(d="D-060", b=blob) + "\n"), True,
                 "no decision D-059"),
}


@pytest.mark.parametrize("name", sorted(TAMPER))
def test_gate_refuses_every_tampered_record(gate_repo, name):
    repo, blob = gate_repo
    edit, commit, reason = TAMPER[name]
    edit(repo, blob)
    if commit:
        _commit(repo, name)
    g = _gate(repo, blob)
    assert g["authorized"] is False and any(reason in r for r in g["reasons"]), g


def _replay_child(repo: Path, blob: str, tmp_path: Path, env: dict | None = None) -> subprocess.CompletedProcess:
    """The _row entry of the pipeline copy in the throw-away repository, in its own process, on a D1 grid. The arrays do not exist:
    an authorised child fails while loading them, before any computation."""
    sp = tmp_path / "spec.json"
    sp.write_text(json.dumps({"kind": "arrays", "arrays_npz": str(tmp_path / "absent.npz"), "arrays_sha256": {}, "config": D1_CFG, "s": 1}),
                  encoding="utf-8")
    env = _child_env(repo, **{pl.CAMPAIGN_AUTH_ENV: f"D-059:{blob}"}, **(env or {}))
    return subprocess.run([sys.executable, str(repo / pl.SELF), "_row", str(sp)], cwd=repo, capture_output=True, text=True, env=env)


def test_replay_child_validates_the_tampered_repository_itself(gate_repo, tmp_path):
    repo, blob = gate_repo
    p = _replay_child(repo, blob, tmp_path)
    assert "D1_START = BLOCKED" not in p.stderr and "absent.npz" in p.stderr, p.stderr[-800:]  # passed the gate; no array, no row
    op8 = repo / "scripts/r5_cccc_q4_d1_c3plus_trajectories.py"
    op8.write_bytes(op8.read_bytes() + b"\n# tampered\n")
    _commit(repo, "tampered tool")
    p = _replay_child(repo, blob, tmp_path)
    assert p.returncode == 2 and "D1_START = BLOCKED" in p.stderr and "blob at HEAD differs" in p.stderr, p.stderr[-800:]


# --- B-2: the code that actually runs ----------------------------------------------------------------------------------------------------

HS = "src/kkvim_reference/r5_cccc_halfstep.py"


def _gate_in_child(repo: Path, blob: str) -> subprocess.CompletedProcess:
    """The gate of the pipeline copy, in a child with the real thread settings: the environment of the addendum is that of a child."""
    code = (f"import importlib.util, json, sys; s = importlib.util.spec_from_file_location('p', {str(repo / pl.SELF)!r}); "
            "p = importlib.util.module_from_spec(s); sys.modules['p'] = p; s.loader.exec_module(p); g = p.campaign_gate(); "
            "print(json.dumps({'gate': g['authorized'], 'reasons': g['reasons'], 'runtime': p.runtime_defects(g)}))")
    p = subprocess.run([sys.executable, "-c", code], cwd=repo, capture_output=True, text=True, env=_child_env(repo, **{pl.CAMPAIGN_AUTH_ENV: f"D-059:{blob}"}))
    assert p.returncode == 0, p.stderr[-800:]
    return json.loads(p.stdout.strip().splitlines()[-1])


def test_the_approved_copy_passes_gate_and_runtime_checks_in_a_child(gate_repo):
    repo, blob = gate_repo
    r = _gate_in_child(repo, blob)
    assert r == {"gate": True, "reasons": [], "runtime": []}, r


def _evil_package(tmp_path: Path) -> Path:
    evil = tmp_path / "evil"
    shutil.copytree(ROOT / pl.REFERENCE_PACKAGE, evil / "kkvim_reference", ignore=shutil.ignore_patterns("__pycache__"))
    return evil


def test_pythonpath_injection_is_refused_in_the_child(gate_repo, tmp_path):
    repo, blob = gate_repo
    p = _replay_child(repo, blob, tmp_path, {"PYTHONPATH": str(_evil_package(tmp_path))})
    assert p.returncode == 2 and "D1_START = BLOCKED" in p.stderr, p.stderr[-800:]
    assert "PYTHONPATH" in p.stderr and "not under the intended source root" in p.stderr


def test_pythonpath_with_the_intended_root_first_and_more_is_refused(gate_repo, tmp_path):
    repo, blob = gate_repo
    p = _replay_child(repo, blob, tmp_path, {"PYTHONPATH": os.pathsep.join((str(repo / "src"), str(_evil_package(tmp_path))))})
    assert p.returncode == 2 and "PYTHONPATH" in p.stderr, p.stderr[-800:]


def test_wrong_package_path_and_foreign_numpy_are_named(monkeypatch, tmp_path):
    fake = SimpleNamespace(__file__=str(tmp_path / "elsewhere" / "kkvim_reference" / "x.py"))
    monkeypatch.setitem(sys.modules, "kkvim_reference.planted", fake)
    monkeypatch.setitem(sys.modules, "numpy.planted", SimpleNamespace(__file__=str(tmp_path / "numpy" / "x.py")))
    rec = pl.loaded_code_record()
    assert any("kkvim_reference.planted" in x and "intended source root" in x for x in rec["outside"])
    assert any(x.startswith("numpy:") and "outside the environment" in x for x in rec["outside"])
    assert any("intended source root" in x for x in pl.loaded_code_defects(rec, {"x": "y"}))


def test_module_under_the_root_that_is_not_approved_is_named(monkeypatch):
    monkeypatch.setitem(sys.modules, "planted", SimpleNamespace(__file__=str(ROOT / "scripts" / "planted.py")))
    rec = pl.loaded_code_record()
    assert rec["loaded"]["scripts/planted.py"] is None
    assert any("scripts/planted.py is not recorded" in x for x in pl.loaded_code_defects(rec, {"x": "y"}))
    monkeypatch.setitem(sys.modules, "sourceless", SimpleNamespace(__file__=str(ROOT / "src" / "kkvim_reference" / "x.pyc")))
    assert pl.loaded_code_record()["loaded"]["src/kkvim_reference/x.pyc"] == "NOT_SOURCE:.pyc"


@pytest.mark.parametrize("flag", ["--assume-unchanged", "--skip-worktree"])
@pytest.mark.parametrize("path", [HS, "scripts/r5_cccc_q4_d1_c3plus_trajectories.py"])
def test_on_disk_mutation_hidden_from_git_is_refused(gate_repo, tmp_path, flag, path):
    repo, blob = gate_repo
    _git(repo, "update-index", flag, path)
    f = repo / path
    f.write_bytes(f.read_bytes() + b"\n# changed under a flag\n")
    assert _git(repo, "status", "--porcelain", "--untracked-files=no") == ""  # git sees nothing
    g = _gate(repo, blob)
    assert g["authorized"] is False and any("on-disk bytes" in r and path in r for r in g["reasons"]), g
    p = _replay_child(repo, blob, tmp_path)
    assert p.returncode == 2 and "D1_START = BLOCKED" in p.stderr and "on-disk bytes" in p.stderr, p.stderr[-800:]


@pytest.mark.parametrize("name", ["evil.py", "r5_cccc_new.pyc"])
def test_untracked_substitution_in_the_package_is_refused(gate_repo, name):
    repo, blob = gate_repo
    (repo / pl.REFERENCE_PACKAGE / name).write_bytes(b"x = 1\n")
    g = _gate(repo, blob)
    assert g["authorized"] is False and any(name in r and "not recorded" in r for r in g["reasons"]), g


def test_dependency_replaced_after_the_checks_is_refused_in_the_same_process(gate_repo, tmp_path):
    repo, blob = gate_repo
    code = f"""
import importlib.util, json, sys
s = importlib.util.spec_from_file_location("p", {str(repo / pl.SELF)!r})
p = importlib.util.module_from_spec(s); sys.modules["p"] = p; s.loader.exec_module(p)
cfg = p.op8().TrajectoryConfig(1.0, 50, 1.0, 50, 1, 1.0, 1.0, 1)
p.guard_config(cfg)
print("first pass", flush=True)
f = p.ROOT / {HS!r}
f.write_bytes(f.read_bytes() + b"\\n# replaced\\n")
try:
    p.guard_config(cfg)
except p.D1Refused as e:
    print("refused:", e)
"""
    r = subprocess.run([sys.executable, "-c", code], cwd=repo, capture_output=True, text=True, env=_child_env(repo, **{pl.CAMPAIGN_AUTH_ENV: f"D-059:{blob}"}))
    assert "first pass" in r.stdout and "refused:" in r.stdout and "r5_cccc_halfstep" in r.stdout, r.stdout + r.stderr[-800:]
    pl._GATE_CACHE.clear()
    assert _gate(repo, blob)["authorized"] is False


def test_replay_child_import_mismatch_is_named_by_the_parent():
    good = {"loaded": {"a.py": "1"}, "outside": []}
    env = _child_environment()
    ctx = {"real": True, "environment": env, "code": {"a.py": "1"}}
    assert c._process_defects(env, good, ctx, "replay") == []
    assert any("bytes are blob 2" in x for x in c._process_defects(env, {"loaded": {"a.py": "2"}, "outside": []}, ctx, "replay"))
    assert any("elsewhere" in x for x in c._process_defects(env, {"loaded": {}, "outside": ["elsewhere"]}, ctx, "replay"))
    assert any("not recorded" in x for x in c._process_defects(env, None, ctx, "replay"))
    assert any("unavailable" in x or "NOT VERIFIED" in x for x in c._process_defects(None, good, ctx, "replay"))


def _advance_main(repo: Path, edit) -> None:
    """origin/main moves on; this HEAD stays where it was (an obsolete ancestor)."""
    head = _git(repo, "rev-parse", "HEAD")
    edit()
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "main moves on")
    _git(repo, "update-ref", pl.INTEGRATION_REF, "HEAD")
    _git(repo, "reset", "-q", "--hard", head)


def test_withdrawn_or_replaced_act_on_main_refuses_an_older_head(gate_repo):
    repo, blob = gate_repo
    head_dec = (repo / pl.DECISIONS_DOC).read_text(encoding="utf-8")
    for name, new in (("removed", head_dec.replace(ACT.format(d="D-060", b=blob), "")),
                      ("withdrawn", head_dec.replace("state=ACCEPTED", "state=WITHDRAWN")),
                      ("replaced", head_dec.replace(f"addendum_blob={blob}", "addendum_blob=" + "1" * 40))):
        _advance_main(repo, lambda: _decisions(repo, new))  # noqa: B023
        g = _gate(repo, blob)
        assert _git(repo, "merge-base", "--is-ancestor", "HEAD", pl.INTEGRATION_REF) == ""  # the head is contained: the old check would pass
        assert g["authorized"] is False and any("withdrawn, replaced or not integrated" in r for r in g["reasons"]), (name, g)


def test_tools_changed_on_main_after_the_act_refuse_an_older_head(gate_repo):
    repo, blob = gate_repo
    op8 = repo / "scripts/r5_cccc_q4_d1_c3plus_trajectories.py"
    _advance_main(repo, lambda: op8.write_bytes(op8.read_bytes() + b"\n# later\n"))
    g = _gate(repo, blob)
    assert g["authorized"] is False and any("the source changed after the act" in r for r in g["reasons"]), g


def test_unrelated_commit_on_main_keeps_the_authorisation(gate_repo):
    repo, blob = gate_repo
    _advance_main(repo, lambda: (repo / "note.txt").write_text("x", encoding="utf-8"))
    assert _gate(repo, blob)["authorized"] is True


def test_stale_local_main_and_an_unreachable_remote_are_defects(gate_repo, tmp_path):
    repo, blob = gate_repo
    assert any("NOT VERIFIED" in x for x in pl.remote_main_defects(repo))  # no remote: unknown is a defect
    bare = tmp_path / "remote.git"
    _git(tmp_path, "init", "-q", "--bare", str(bare))
    _git(repo, "remote", "add", "origin", str(bare))
    _git(repo, "push", "-q", str(bare), "HEAD:refs/heads/main")
    assert pl.remote_main_defects(repo) == []
    (repo / "note.txt").write_text("x", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "later")
    _git(repo, "push", "-q", str(bare), "HEAD:refs/heads/main")  # by its path: the local ref of origin stays behind
    assert any("stale" in x for x in pl.remote_main_defects(repo))  # the local ref was not updated


def test_gate_requires_the_campaign_tools_in_the_addendum(gate_repo):
    repo, blob = gate_repo
    add = json.loads((repo / pl.CAMPAIGN_ADDENDUM).read_text(encoding="utf-8"))
    del add["tools"]["new"]["scripts/r5_cccc_q4_d1_campaign.py"]
    bad = pl.addendum_defects(add, json.loads((repo / pl.FROZEN_MANIFEST).read_bytes()), lambda p: pl._git_blob(repo, p))
    assert any("campaign tool" in x for x in bad)


def test_gate_refuses_a_head_not_integrated(gate_repo):
    repo, blob = gate_repo
    (repo / "note.txt").write_text("x", encoding="utf-8")
    _commit(repo, "local", integrate=False)
    g = _gate(repo, blob)
    assert g["authorized"] is False and any("not integrated" in r for r in g["reasons"]), g


# --- plan and accounting (F2) ---------------------------------------------------------------------------------------------------------

def test_plan_accounting_and_rank_guard_rows():
    plan = rd.row_plan()
    acc = c.plan_accounting(plan)
    assert acc["classified_rows"] == 824 and acc["mandatory_keys"] == 102
    assert acc["n23_s2plus"] == {"classified_rows": 399, "rac_half_steps": 98, "total": 497}
    assert acc["rac_half_steps_not_campaign_rows"] == 168
    assert c.plan_defects(plan, c.frozen_manifest()) == []
    assert [r["key"] for r in plan if r["kind"] == "row" and r["mandatory"]] == [r["key"] for r in c.frozen_manifest()["mandatory_rows"]]
    us = c.units(plan)
    assert sum(len(u["rows"]) for u in us) == 824 and sum(len(u["rac"]) for u in us) == 168
    for u in us:
        cfg = c._cfg(u)
        for r in u["rows"] + u["rac"]:
            if r["N"] >= 2 and r["s"] >= 2:
                assert not r["executable"]
                reason = c._planned_reason(r, cfg)
                assert "STOPPED_RANK_GUARD_UNCERTIFIED" in reason
                if r["kind"] == "row":
                    rec = c._not_executed_record(r, reason)
                    assert rec["status"] == "UNRESOLVED" and rec["executed"] is False
    broken = [dict(r) for r in plan]
    k = next(i for i, r in enumerate(broken) if r["kind"] == "row" and r["N"] == 2 and r["s"] == 2)
    broken[k]["executable"] = True
    assert any("D-057.1" in x for x in c.plan_defects(broken, c.frozen_manifest()))


# --- exact classifier input (C1) ------------------------------------------------------------------------------------------------------

def test_classifier_input_round_trip(tmp_path):
    TC = pl.op8().TrajectoryConfig
    jd = pl.run_trajectory_journal(TC(1.0, 10, 1.0, 12, 1, s_max=3), tmp_path / "j")
    seen = []
    for s in (1, 2, 3):
        res = pl.analyse({"kind": "journal", "journal": str(jd), "s": s}, reproduce=(s == 2))
        seen.append(res)
    seen.append(pl.not_executed(None, 4, reason="not executed: test"))
    for res in seen:
        inp = res["classification"]["input"]
        direct = pf.classify_row(inp)
        assert direct["status"] == res["status"]
        enc = c.encode_input(inp)
        assert c.encode_input(c.decode_input(enc)) == enc
        assert pf.classify_row(c.decode_input(enc)) == direct
    assert seen[1]["premises"]["P-a"]["state"] == pl.ESTABLISHED and seen[1]["status"] == "PASS"


@pytest.mark.parametrize("threads", [{"numpy": 2, "scipy": 1}, {"numpy": 1, "scipy": 2}, {"numpy": 1, "scipy": None}, None, "absent"])
def test_replay_blas_threads_rule(threads, monkeypatch, tmp_path):
    """D-058.3: a replay process whose BLAS threads (NumPy and SciPy libraries) are not 1 / 1 or not verified: П-а not established."""
    TC = pl.op8().TrajectoryConfig
    jd = pl.run_trajectory_journal(TC(1.0, 10, 1.0, 12, 1, s_max=1), tmp_path / "j")
    real = pl.run_child

    def child(spec, *a, **k):
        job = real(spec, *a, **k)
        assert job["blas_threads"] == {"numpy": 1, "scipy": 1}  # the real child: one thread in both libraries
        if threads == "absent":
            job.pop("blas_threads")
        else:
            job["blas_threads"] = threads
        return job

    monkeypatch.setattr(pl, "run_child", child)
    res = pl.analyse({"kind": "journal", "journal": str(jd), "s": 1}, reproduce=True)
    assert res["premises"]["P-a"]["state"] == pl.NOT_ESTABLISHED and "OD-6 (D-058.3)" in res["premises"]["P-a"]["reason"]
    assert res["status"] == "UNRESOLVED_CERTIFICATE"
    assert pl.replay_blas_defects({"blas_threads": {"numpy": 1, "scipy": 1}}, "x") == []


# --- blocking semantics (F4) ----------------------------------------------------------------------------------------------------------

def _rows(spec: dict):
    mand = pf.mandatory_index(c.frozen_manifest()["mandatory_rows"])
    out = []
    for k, (blk, N, s) in mand.items():
        st, z4 = spec.get(k, ("PASS", None))
        out.append({"key": k, "block": blk, "N": N, "s": s, "mutated": False, "input": ct._syn(st), "mandatory": True, "z4": z4,
                    "tool_defects": []})
    return out, mand


def test_campaign_blocking_semantics():
    rows, mand = _rows({})
    base = rd.verdict(rows, mand)
    assert base == rd.verdict(rows, mand, input_hash_mismatch=(), control_v_fail=False)  # defaults: the verdict of QA-D1-READY
    assert base["verdict"] == "PASS_CANDIDATE"
    v = rd.verdict(rows, mand, input_hash_mismatch=(rows[0]["key"],))
    assert v["verdict"] == "BLOCKED_CAMPAIGN" and any("input hash mismatch" in x for x in v["classifier"]["block_reasons"])
    v = rd.verdict(rows, mand, control_v_fail=True)
    assert v["verdict"] == "BLOCKED_CAMPAIGN" and any("control V" in x for x in v["classifier"]["block_reasons"])
    b_fail = {"key": "D1-B|r=1|50x50|N2|s1", "block": "B", "N": 2, "s": 1, "mutated": False, "input": ct._syn("FAIL_COMPUTATION"),
              "mandatory": False, "z4": {"outcome": "CONFIRMED"}, "tool_defects": []}
    assert rd.verdict(rows + [b_fail], mand)["verdict"] == "BLOCKED_CAMPAIGN"  # confirmed computation defect
    recs = [c._not_executed_record({"key": r["key"], "block": r["block"], "N": r["N"], "s": r["s"], "mandatory": True}, "blocked")
            for r in rows]
    assert c.campaign_verdict(recs, input_hash_mismatch=["k"])["verdict"] == "BLOCKED_CAMPAIGN"
    assert c.campaign_verdict(recs, control_v_fail=True)["verdict"] == "BLOCKED_CAMPAIGN"


# --- dry runs on non-D1 grids: launch, watchdog, interruption, restart ---------------------------------------------------------------

def _dry_plan():
    plan = [rd._row("A", "r=1", "1", 1, 10, 12, 1.0, 1.0, 1, s) for s in (1, 2)]
    plan += [rd._row("B", "r=1", "1", 1, 12, 10, 1.0, 1.0, 2, s) for s in (1, 2)]
    return plan


@pytest.fixture
def dry_out():
    out = ROOT / "scratch" / f"test_d1_campaign_{os.getpid()}"
    shutil.rmtree(out, ignore_errors=True)
    yield out
    shutil.rmtree(out, ignore_errors=True)


def test_dry_run_refuses_d1_grids_and_real_overrides(dry_out):
    with pytest.raises(c.CampaignRefused, match="never touches the D1 grids"):
        c.run_campaign(dry_out, plan=[rd._row("A", "r=1", "1", 1, 50, 50, 1.0, 1.0, 1, 1)], dry_run=True)
    with pytest.raises(c.CampaignRefused, match="frozen plan"):
        c.run_campaign(dry_out, plan=_dry_plan())
    with pytest.raises(c.CampaignRefused, match="frozen plan"):
        c.run_campaign(dry_out, memory_limit_bytes=1)


def _core(doc_rows, out):
    recs = {}
    for f in (out / "units").glob("*.json"):
        for r in json.loads(f.read_text(encoding="utf-8"))["rows"]:
            i = r["input"]
            recs[r["key"]] = (r["status"], {k: i.get(k) for k in ("executed", "dP", "tau", "Q", "P_p", "P_s", "Gamma_up", "B_E")})
    assert set(recs) == {r["key"] for r in doc_rows}
    return recs


def test_dry_run_launch_interrupt_restart(dry_out):
    plan = _dry_plan()
    with pytest.raises(c.CampaignInterrupted):
        c.run_campaign(dry_out / "a", plan=plan, dry_run=True, stop_after_jobs=2)
    assert len(list((dry_out / "a" / "units").glob("*.json"))) == 1 and not (dry_out / "a" / "campaign_result.json").exists()
    doc_a = c.run_campaign(dry_out / "a", plan=plan, dry_run=True)
    doc_b = c.run_campaign(dry_out / "b", plan=plan, dry_run=True)
    assert doc_a["state"] == doc_b["state"] == "COMPLETED" and doc_a["data_status"] == c.DATA_STATUS_DRY
    assert doc_a["accounting"]["classified_rows"] == 4 and doc_a["accounting"]["n23_s2plus"]["classified_rows"] == 1
    assert _core(doc_a["rows"], dry_out / "a") == _core(doc_b["rows"], dry_out / "b")
    st = {r["key"]: r["status"] for r in doc_a["rows"]}
    assert st["D1-A|r=1|10x12|N1|s1"] == st["D1-A|r=1|10x12|N1|s2"] == st["D1-B|r=1|12x10|N2|s1"] == "PASS"
    assert st["D1-B|r=1|12x10|N2|s2"] == "UNRESOLVED"
    for f in (dry_out / "a" / "units").glob("*.json"):
        u = json.loads(f.read_text(encoding="utf-8"))
        assert u["prep"]["process"]["blas_threads"] == {"numpy": 1, "scipy": 1}
        for r in u["rows"]:
            if r["executed"]:
                assert r["process"]["blas_threads"] == {"numpy": 1, "scipy": 1} and r["process"]["launch_signal_present"] is False
    s = json.loads((dry_out / "a" / "state.json").read_text(encoding="utf-8"))
    s["provenance"]["head"] = "0" * 40
    (dry_out / "a" / "state.json").write_text(json.dumps(s), encoding="utf-8")
    with pytest.raises(c.CampaignRefused, match="restart refused"):
        c.run_campaign(dry_out / "a", plan=plan, dry_run=True)


def test_watchdog_makes_rows_unresolved_resources(dry_out):
    doc = c.run_campaign(dry_out, plan=_dry_plan(), dry_run=True, memory_limit_bytes=1)
    st = {r["key"]: r["status"] for r in doc["rows"]}
    assert st["D1-A|r=1|10x12|N1|s1"] == st["D1-A|r=1|10x12|N1|s2"] == st["D1-B|r=1|12x10|N2|s1"] == "UNRESOLVED_RESOURCES"
    assert st["D1-B|r=1|12x10|N2|s2"] == "UNRESOLVED"  # never computed, whatever the resources
    for f in (dry_out / "units").glob("*.json"):
        for r in json.loads(f.read_text(encoding="utf-8"))["rows"]:
            if r["status"] == "UNRESOLVED_RESOURCES":
                m = r["input"]["not_executed"]["measured"]
                assert m["aborted"] == "memory" and m["working_set_bytes"] > m["limit_bytes"] == 1


# --- frozen data ----------------------------------------------------------------------------------------------------------------------

def test_frozen_manifest_and_inputs_unchanged():
    assert pl.git_blob_id(ROOT / pl.FROZEN_MANIFEST) == pl.FROZEN_MANIFEST_BLOB != CANDIDATE_BLOB
    assert pl.git_blob_id(ROOT / "docs/results/r5_cccc_q4_d1_prelaunch/mandatory_rows.csv") == CSV_BLOB
    frozen = c.frozen_manifest()
    assert frozen["freeze_act"]["state"] == "FROZEN" and frozen["d1_start"] == "BLOCKED"
    for path, blob in frozen["d057"]["tool_blobs"].items():
        if path not in ("scripts/r5_cccc_q4_d1_row_pipeline_prelaunch.py", "scripts/r5_cccc_q4_d1_ready.py"):
            assert pl.git_blob_id(ROOT / path) == blob, path


# --- the re-freeze addendum (C4) ------------------------------------------------------------------------------------------------------

def test_addendum_is_pending_and_consistent(monkeypatch):
    add = json.loads((ROOT / pl.CAMPAIGN_ADDENDUM).read_text(encoding="utf-8"))
    assert add["decision"] == "D-059" and add["state"] == pl.PENDING_REFREEZE and add["d1_start"] == "BLOCKED"
    fb = add["frozen_baseline"]
    assert fb["main_commit"] == BASE_COMMIT
    assert fb["d058_frozen_manifest"]["blob"] == pl.FROZEN_MANIFEST_BLOB and fb["prelaunch_candidate_manifest"]["blob"] == CANDIDATE_BLOB
    assert fb["mandatory_rows_csv"]["blob"] == CSV_BLOB
    frozen = c.frozen_manifest()
    assert pl.addendum_defects(add, frozen, lambda p: pl.git_blob_id(ROOT / p)) == []
    t = add["tools"]
    assert set(t["changed"]) == set(c.CHANGED) and set(t["new"]) == {"scripts/r5_cccc_q4_d1_campaign.py"}
    assert t["changed"]["scripts/r5_cccc_q4_d1_row_pipeline_prelaunch.py"]["d058_blob"].startswith("ee9a36d")
    assert t["changed"]["scripts/r5_cccc_q4_d1_ready.py"]["d058_blob"].startswith("3dcdce5")
    assert add["plan"]["accounting"]["classified_rows"] == 824 and add["plan"]["accounting"]["mandatory_keys"] == 102
    assert add["plan"]["accounting"]["n23_s2plus"] == {"classified_rows": 399, "rac_half_steps": 98, "total": 497}
    assert add["controls"]["all_match"] is True and add["controls"]["expected_sha256"] == ct.EXPECTED_SHA256
    assert add["platform"]["sys_platform"] == "win32" and add["platform"]["python"] == "3.13.15"
    assert add["platform"]["blas_threads_both_libraries"] == {"numpy": 1, "scipy": 1}
    env = add["environment"]
    assert sorted(env) == sorted(pl.ENVIRONMENT_KEYS) and all(env[k] is not None for k in pl.ENVIRONMENT_KEYS)
    assert env["python"] == "3.13.15" and env["blas_threads"] == {"numpy": 1, "scipy": 1} and env["numpy"] == "2.5.4" and env["scipy"] == "1.18.1"
    disk = {p.relative_to(ROOT).as_posix() for p in (ROOT / pl.REFERENCE_PACKAGE).rglob("*.py")}
    assert disk <= set(t["reference_package"])
    assert "trust_assumptions" in add and pl.code_identity_defects(ROOT, add) == []
    # the addendum alone authorises nothing: no owner re-freeze act exists
    monkeypatch.setenv(pl.CAMPAIGN_AUTH_ENV, "D-059:" + pl.git_blob_id(ROOT / pl.CAMPAIGN_ADDENDUM))
    g = pl.campaign_gate()
    assert g["authorized"] is False and g["state"] == pl.PENDING_REFREEZE
    with pytest.raises(pl.D1Refused, match="D1_START = BLOCKED"):
        pl.guard_config(pl.op8().TrajectoryConfig(**D1_CFG))


def test_every_module_a_job_really_loads_is_recorded_by_the_addendum(dry_out):
    """All kinds of job on non-D1 grids, every selected path: what the children loaded is approved code (else a real run refuses)."""
    add = json.loads((ROOT / pl.CAMPAIGN_ADDENDUM).read_text(encoding="utf-8"))
    code, seen = pl.recorded_code(add), set()
    plan = [rd._row("A", "r=1", "1", 1, 10, 12, 1.0, 1.0, 1, 1), rd._row("C", "r=1", "1", 2, 10, 12, 1.0, 1.0, 1, 1), rd._row("C", "r=1", "1", 2, 10, 12, 1.0, 1.0, 1, 1, "rac")]
    doc = c.run_campaign(dry_out, plan=plan, dry_run=True, subsets={plan[0]["key"]: ["SP-RH", "SP-LARGE", "SP-C", "SP-BR", "SP-OP", "SP-BE", "SP-GAMMA"]})
    assert doc["state"] == "COMPLETED"

    def walk(o):
        if isinstance(o, dict):
            if isinstance(o.get("loaded"), dict) and "outside" in o:
                seen.update(o["loaded"])
                assert o["outside"] == []
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    for f in (dry_out / "units").glob("*.json"):
        walk(json.loads(f.read_text(encoding="utf-8")))
    assert seen and seen <= set(code), sorted(seen - set(code))
    assert set(c._code_paths()) <= set(code), sorted(set(c._code_paths()) - set(code))


# --- the real-run path before the first row: no D1 row is computed (a started unit fails the test) ---------------------------------------

def _authorised_until_prerun(monkeypatch, prerun):
    monkeypatch.setattr(pl, "campaign_gate", lambda *a, **k: {"authorized": True, "state": "AUTHORIZED", "reasons": []})
    monkeypatch.setattr(c, "platform_defects", lambda: [])
    monkeypatch.setattr(c, "prerun_checks", lambda *a, **k: prerun)

    def no_unit(*a, **k):
        raise AssertionError("a unit of the campaign was started")
    monkeypatch.setattr(c, "run_unit", no_unit)


@pytest.mark.parametrize("mismatch, v_fail", [(["D1-A|r=1|50x50|N1|s1"], False), ([], True)])
def test_blocked_before_the_run_gives_blocked_campaign(mismatch, v_fail, monkeypatch, dry_out):
    _authorised_until_prerun(monkeypatch, {"input_hash_mismatch": mismatch, "pr48_controls_all_match": True, "controls_state": c.CONTROLS_VALID,
                                           "tool_defects": [], "ready_controls_all_match": not v_fail, "control_v_fail": v_fail})
    doc = c.run_campaign(dry_out)
    assert doc["state"] == "BLOCKED_BEFORE_THE_RUN" and doc["verdict"]["verdict"] == "BLOCKED_CAMPAIGN"
    assert doc["accounting"]["classified_rows"] == 824 and not (dry_out / "units").exists()


def test_controls_mismatch_does_not_start_the_campaign(monkeypatch, dry_out):
    _authorised_until_prerun(monkeypatch, {"input_hash_mismatch": [], "pr48_controls_all_match": True, "controls_state": c.CONTROLS_VALID,
                                           "tool_defects": [], "ready_controls_all_match": False, "control_v_fail": False})
    doc = c.run_campaign(dry_out)
    assert doc["state"].startswith("NOT_STARTED_CONTROLS_MISMATCH") and "verdict" not in doc and not (dry_out / "units").exists()


def test_prerun_checks_real_controls_and_hash_mismatch(dry_out):
    """The real checks: frozen hashes and archive, PR #48 harness, QA-D1-READY controls in a child without the launch signal."""
    plan, frozen = rd.row_plan(), c.frozen_manifest()
    res = c.prerun_checks(plan, frozen, dry_out)
    assert res["pr48_controls_all_match"] is True and res["ready_controls_all_match"] is True and res["control_v_fail"] is False, res
    if pf._same_hash_platform(frozen["hash_platform"]):  # start sets depend on numpy.sin (OD-12): strict on the recorded platform
        assert res["input_hash_mismatch"] == []
        bad = json.loads(json.dumps(frozen))
        bad["d057"]["input_hashes_s1"][0]["given_in_sha"] = "0" * 64
        k = bad["d057"]["input_hashes_s1"][0]["key"]
        assert k in c.prerun_checks(plan, bad, dry_out / "b")["input_hash_mismatch"]


def test_dry_run_block_c_keeps_rac_apart(dry_out):
    plan = [rd._row("C", "r=1", "1", 2, 10, 12, 1.0, 1.0, 1, s, kind) for s in (1, 2) for kind in ("row", "rac")]
    plan += [rd._row("C", "r=1", "1", 2, 10, 12, 1.0, 1.0, 2, s, kind) for s in (1, 2) for kind in ("row", "rac")]
    doc = c.run_campaign(dry_out, plan=plan, dry_run=True)
    assert doc["accounting"]["classified_rows"] == 4 and doc["accounting"]["rac_half_steps_not_campaign_rows"] == 4
    assert doc["accounting"]["n23_s2plus"] == {"classified_rows": 1, "rac_half_steps": 1, "total": 2}
    st = {r["key"]: r["status"] for r in doc["rows"]}
    assert not any("RAC" in k for k in st) and st["D1-C|r=1|10x12|N1|s1"] == st["D1-C|r=1|10x12|N1|s2"] == "PASS"
    assert st["D1-C|r=1|10x12|N2|s1"] == "PASS" and st["D1-C|r=1|10x12|N2|s2"] == "UNRESOLVED"
    rac = {x["key"]: x for x in doc["rac"]}
    assert rac["D1-C-RAC|r=1|10x12|N1|s2"]["chain_record_status"] == "OK" and rac["D1-C-RAC|r=1|10x12|N1|s2"]["chain_certificate"]
    assert "STOPPED_RANK_GUARD_UNCERTIFIED" in rac["D1-C-RAC|r=1|10x12|N2|s2"]["trajectory_reason"]
