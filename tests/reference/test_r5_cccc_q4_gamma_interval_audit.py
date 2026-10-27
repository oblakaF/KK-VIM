"""Tests of QA-gamma3 (scripts/r5_cccc_q4_gamma_interval_audit.py): directed binary64 rounding of exact rationals (positive, negative,
zero, subnormal, overflow, ties, representable values), the exact inclusion check and its refusal of wrong intervals, Gamma_up,
path B1 against the existing exact paths, non-finite inputs, provenance and reproducibility. QA research tool."""

import ast
import csv
import importlib.util
import math
import random
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from kkvim_reference import r5_cccc_halfstep as hs

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "r5_cccc_q4_gamma_interval_audit.py"
spec = importlib.util.spec_from_file_location("r5_cccc_q4_gamma_interval_audit", SCRIPT)
gi = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gi)

TINY = 5e-324  # smallest positive subnormal double
DBL_MAX = gi.DBL_MAX


def largest_double_le(x: Fraction) -> float:
    """Independent reference for the tests: bisection over the ordered binary64 bit patterns (no float() rounding)."""
    import struct

    def val(k):  # k-th double of the ordered set of non-negative doubles
        return struct.unpack("<d", struct.pack("<q", k))[0]

    if x < 0:
        return -smallest_double_ge(-x)
    lo, hi = 0, 0x7FEFFFFFFFFFFFFF  # 0.0 .. DBL_MAX
    if Fraction(val(hi)) <= x:
        return val(hi)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if Fraction(val(mid)) <= x:
            lo = mid
        else:
            hi = mid
    return val(lo)


def smallest_double_ge(x: Fraction) -> float:
    if x <= 0:
        return -largest_double_le(-x)
    d = largest_double_le(x)
    return d if Fraction(d) == x else math.nextafter(d, math.inf)


CASES = [Fraction(1, 3), Fraction(-1, 3), Fraction(0), Fraction(1), Fraction(-1), Fraction(1, 10), Fraction(-7, 10**30),
         Fraction(3, 2**1075), Fraction(-3, 2**1075), Fraction(1, 2**1080), Fraction(-1, 2**1080), Fraction(TINY),
         Fraction(gi.MIN_NORMAL) * Fraction(3, 2), Fraction(gi.MIN_NORMAL) - Fraction(1, 2**1100),
         Fraction(1) + Fraction(1, 2**53),  # a tie between 1 and 1 + 2^-52 (round half to even)
         -(Fraction(1) + Fraction(3, 2**54)), Fraction(DBL_MAX) - Fraction(1, 7), Fraction(2**1023) * 3, -Fraction(2**1023) * 3]


@pytest.mark.parametrize("x", CASES, ids=[str(i) for i in range(len(CASES))])
def test_directed_rounding_is_the_tightest_enclosure(x):
    down, up = gi.directed(x)
    assert gi.verify(x, down, up) is True
    if abs(x) <= Fraction(DBL_MAX):
        assert down == largest_double_le(x) and up == smallest_double_ge(x)
    elif x > 0:
        assert (down, up) == (DBL_MAX, math.inf)
    else:
        assert (down, up) == (-math.inf, -DBL_MAX)
    G = gi.gamma_up(down, up)
    assert gi.gamma_up_holds(x, G)


def test_directed_rounding_random_both_signs():
    rnd = random.Random(20261027)
    for _ in range(3000):
        x = Fraction(rnd.randint(1, 10**40), rnd.randint(1, 10**40)) * Fraction(2) ** rnd.randint(-1100, 1000)
        x = x if rnd.random() < 0.5 else -x
        down, up = gi.directed(x)
        assert gi.verify(x, down, up) is True
        assert down == largest_double_le(x) and up == smallest_double_ge(x)
        assert down <= up and (down < 0 or up >= 0 or x == 0)
        G = gi.gamma_up(down, up)
        assert Fraction(G) >= abs(x)
        assert (x > 0 and down >= 0) or (x < 0 and up <= 0) or x == 0  # signs of the endpoints follow the sign of x


def test_zero_and_signed_zero():
    assert gi.directed(Fraction(0)) == (0.0, 0.0)
    down, up = gi.directed(-Fraction(1, 2**1100))  # below the smallest subnormal: [-5e-324, -0.0]
    assert down == -TINY and up == 0.0 and math.copysign(1, up) == -1
    assert gi.gamma_up(down, up) == TINY


def test_verify_refuses_wrong_intervals():
    x = Fraction(1, 3)
    d, u = gi.directed(x)
    assert gi.verify(x, u, d) is not True  # swapped
    assert gi.verify(x, math.nextafter(d, -math.inf), u) is not True  # not adjacent (not tight)
    assert gi.verify(x, d, d) is not True  # x not enclosed
    assert gi.verify(Fraction(1), 1.0, math.nextafter(1.0, 2)) is not True  # a double must give a degenerate interval
    assert gi.verify(x, math.nan, u) is not True


def test_non_fraction_and_non_finite_inputs_are_refused():
    with pytest.raises(TypeError):
        gi.directed(0.5)
    G = hs.start_set(9, 2)
    V = np.ones((7, 2))
    C = np.zeros((7, 2))
    with pytest.raises(ValueError):
        gi.gamma_b1(1.0, 9, 1.0, 8, "Y", G, C, np.where(V > 0, np.nan, V))
    with pytest.raises(ValueError):
        gi.gamma_b1(1.0, 9, 1.0, 8, "Y", G, np.full((7, 2), np.inf), V)


@pytest.mark.parametrize("lx, nx, ly, ny, N, direction", [(1.0, 9, 1.3, 8, 1, "Y"), (1.0, 8, 0.6, 10, 2, "X"),
                                                          (1.0 / 3, 10, 1.0, 9, 3, "Y")])
def test_b1_equals_the_existing_exact_paths(lx, nx, ly, ny, N, direction):
    """B1 (R-HP) = path A of the Q4 gamma audit = energy path (gamma = Q - dP), exactly, on random half-step arrays."""
    ga = gi._ga()
    rng = np.random.default_rng(nx + N)
    n_f, n_s = (nx, ny) if direction == "Y" else (ny, nx)
    G, C, V = (rng.standard_normal((n_f - 1, N)), rng.standard_normal((n_s - 1, N)), rng.standard_normal((n_s - 1, N)))
    g = gi.gamma_b1(lx, nx, ly, ny, direction, G, C, V)
    lf, ls = (lx, ly) if direction == "Y" else (ly, lx)
    (Gi,), EG = ga.dyadic(G)
    (Vi, Ci), EV = ga.dyadic(V, C)
    gA = ga.gamma_from_coefficients(ga.coefficients_exact(Gi, EG, Fraction(lf) / n_f, Fraction(1)), Vi, Ci, EV,
                                    Fraction(ls) / n_s, Fraction(1))
    orient = (lambda a, b: (a, b)) if direction == "Y" else (lambda a, b: (b, a))
    Pp, Ps, Q = ga.energies_path_b(lx, nx, ly, ny, 1.0, 1.0, *orient(G, C), *orient(G, V))
    assert g == gA == Q - (Pp - Ps)


def test_no_import_of_the_working_kkvim():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    names = [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
    names += [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    assert not [m for m in names if m == "kkvim" or m.startswith("kkvim.")]


def test_q3_data_unchanged_since_the_campaign_commit():
    ga = gi._ga()
    out = subprocess.run(["git", "cat-file", "-t", ga.Q3_DATA_COMMIT], cwd=ROOT, capture_output=True, text=True)
    if out.stdout.strip() != "commit":
        pytest.skip("Q3 campaign commit not in this clone")
    assert ga.provenance(ga.Q3_CSV)["unchanged_since_q3_campaign"]


def test_ci_profile_is_deterministic_and_passes(tmp_path):
    outs = []
    for k in range(2):
        out = tmp_path / f"run{k}"
        res = subprocess.run([sys.executable, str(SCRIPT), "run", "--profile", "ci", "--out", str(out)], cwd=ROOT,
                             capture_output=True, text=True, timeout=900)
        assert res.returncode == 0, res.stderr[-2000:]
        outs.append((out / "gamma_intervals.csv").read_bytes())
    assert outs[0] == outs[1]
    rows = list(csv.DictReader(outs[0].decode("utf-8").splitlines()))
    assert len(rows) == 16 and all(r["status"] == "PASS" for r in rows)
    assert {r["sign"] for r in rows} >= {"positive", "negative"}
    for r in rows:
        num, den = (int(v) for v in r["gamma_exact"].split("/"))
        x = Fraction(num, den)
        assert Fraction(float(r["gamma_down"])) <= x <= Fraction(float(r["gamma_up"]))
        assert Fraction(float(r["Gamma_up"])) >= abs(x)
