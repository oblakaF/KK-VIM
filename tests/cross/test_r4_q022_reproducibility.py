"""Reproducibility of the Q-022 study (docs/R4_Q022_ERROR_TRANSFER.md) - scripts/r4_q022/.

Four kinds of quantities are kept apart (scripts/r4_q022/README.md):
  1. analytic bounds (Q22-5, Q22-6 of the document) - proved there, not by these tests;
  2. evaluations of those bounds in double precision (production / table outputs);
  3. the actual difference of two finite sequences (validate, observed) - exact up to double rounding;
  4. hypothesis H-1 (non-expansiveness in the target) - NOT asserted anywhere here.

Fast tests (CI): the implication of the theorem "actual difference <= bound" on small exactly computable cases, and
regression of the short outputs (validate, observed, short table) against scripts/r4_q022/expected/ with a tolerance of one unit in the last printed
digit (no bitwise equality, AGENTS.md, section 4). Heavy runs at M = 4001 only with KKVIM_Q022_HEAVY=1.
"""

import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts" / "r4_q022"
EXPECTED = SCRIPTS / "expected"

_spec = importlib.util.spec_from_file_location("q022_transfer", SCRIPTS / "q022_transfer.py")
tr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tr)

NUM = re.compile(r"[-+]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[eE][-+]?\d+)?|inf|None")


def _tokens(text):
    return NUM.findall(text)


def _unit(tok):
    """One unit in the last printed digit of a numeric token."""
    mant, _, exp = tok.lower().partition("e")
    decimals = len(mant.split(".")[1]) if "." in mant else 0
    return 10.0 ** (-decimals + (int(exp) if exp else 0))


def _same_output(got, want):
    """Compare two outputs token by token: integers and inf/None exactly, floats to one unit of the last digit."""
    g, w = _tokens(got), _tokens(want)
    assert len(g) == len(w), (len(g), len(w))
    for a, b in zip(g, w):
        if b in ("inf", "None") or a in ("inf", "None") or ("." not in b and "e" not in b.lower()):
            assert a == b, (a, b)
        else:
            assert abs(float(a) - float(b)) <= 1.000001 * max(_unit(a), _unit(b)), (a, b)


def _run(*args):
    out = subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, capture_output=True, text=True, check=True)
    return out.stdout


@pytest.mark.parametrize("r", [1.0, 0.2, 5.0])
@pytest.mark.parametrize("N", [1, 2, 3])
def test_actual_difference_never_exceeds_the_bound(r, N):
    """Category 3 against category 2: a finite 'true' target omega_{M2}, truncation to the M1 box (as in 7.1)."""
    M1, M2, S = 21, 201, 8
    P2, P1 = tr.setup(M2, r), tr.setup(M1, r)
    K1 = (M1 + 1) // 2
    f2, _ = tr.run(P2, N, S)
    f1, s1 = tr.run(P1, N, S)
    tau = P2["om"].copy()
    tau[:K1, :K1] = 0
    T = tr.enorm(P2, tau)
    E, _ = tr.bound(P1, f1, s1, T, P1["om"], N)
    checked = 0
    for s in range(1, S + 1):
        Z = np.zeros_like(P2["om"])
        Z[:K1, :K1] = f1[s - 1]
        obs = tr.enorm(P2, f2[s - 1] - Z)
        if np.isfinite(E[s - 1]):
            assert obs <= E[s - 1] * (1 + 1e-12), (s, obs, E[s - 1])
            checked += 1
    assert checked >= 1
    assert E[0] == T  # s = 1: same start, the bound is the target perturbation itself


def test_validate_output_reproduces_expected():
    """Section 7.1 of the document: M2 = 1601, M1 = 101, s <= 12; ratios observed/bound."""
    got = _run(SCRIPTS / "q022_transfer.py", "validate")
    _same_output(got, (EXPECTED / "validate.txt").read_text(encoding="utf-8"))


def test_observed_transfer_reproduces_expected():
    """Section 7.4: E_obs(s)/|tau| for the finite target omega_{1601}, truncation M = 101, N = 2, 3, s <= 40 (observation)."""
    got = _run(SCRIPTS / "q022_observed.py")
    _same_output(got, (EXPECTED / "observed.txt").read_text(encoding="utf-8"))


def test_short_table_reproduces_expected():
    """Short mode of table 7.3 (M = 101, s <= 8; the production table is M = 4001, s <= 40)."""
    got = _run(SCRIPTS / "q022_table.py", 101, 8)
    _same_output(got, (EXPECTED / "table_M101_S8.txt").read_text(encoding="utf-8"))


HEAVY = os.environ.get("KKVIM_Q022_HEAVY") == "1"


@pytest.mark.skipif(not HEAVY, reason="heavy run (M = 4001); set KKVIM_Q022_HEAVY=1")
@pytest.mark.parametrize(("args", "expected"), [
    (("q022_production.py", 4001, 40, "1,0.2,5"), "production_M4001.txt"),
    (("q022_table.py", 4001, 40), "table_M4001.txt"),
])
def test_heavy_outputs_reproduce_expected(args, expected):
    got = _run(SCRIPTS / args[0], *args[1:])
    _same_output(got, (EXPECTED / expected).read_text(encoding="utf-8"))
