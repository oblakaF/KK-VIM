"""R5-V Q4: mutation research of the B_E computations (QA research tool; not D1, no tolerance, no SP-BE).

Each mutation is ONE textual defect applied to the source text of a B_E implementation, executed as a separate module object
(the files are never changed):
- target "primary": the QA-EB path scripts/r5_cccc_q4_energy_bound_premises_audit.py (grid_constants, GridQ, field_sums, b_e);
  checked by the SECOND path (scripts/r5_cccc_q4_be_second_path_research.py);
- target "second": the second path itself; checked by the unmutated primary, by its own certifications (squared roots, directed
  rounding, the exact identity of E-1, the bounds of the parts) and by analytic references of this module.

For every mutation the expectation is fixed in MUTATIONS before the run (effect, channel, verdict per control input), with the
control inputs of CONTROL_INPUTS. The verdict is computed from values only:
- INAPPLICABLE: the mutated implementation gives on that input the SAME components as the unmutated one (the defect has no effect
  there; nothing to detect);
- DETECTED: the mutation is effective and at least one channel fires;
- UNDETECTED: effective and no channel fires (a gap, reported, never PASS).
A channel never reads the name of the mutation: the code under test is changed, the checks compare values.

Channels: "comparison" (first divergent component between the two paths, with its conclusion), "inequality" (|E_s| > B_E, E_s exact
from the exact product fields), "parts" (|E^(k)| > B_k for a part of E-1), "self_check" (the second path refuses: directed root or
rounding not certified, identity of E-1 broken, a part above its bound), "reference" (analytic values of the reference constructions
differ). G-5 analogue (P-01): a dropped term of B_E must be found by the comparison with the independently computed full bound even
when |E_s| <= the reduced bound still holds.

Usage: uv run python scripts/r5_cccc_q4_be_mutation_research.py run --out scratch/be2_mutations
"""

import argparse
import importlib.util
import json
import sys
import time
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SECOND_TOOL = "scripts/r5_cccc_q4_be_second_path_research.py"


def _load(rel: str, name: str, source: str | None = None):
    if source is None and name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    if source is None:
        spec.loader.exec_module(mod)
    else:
        exec(compile(source, str(ROOT / rel), "exec"), mod.__dict__)
    return mod


def sp():
    return _load(SECOND_TOOL, "_be2_second")


# control inputs (synthetic, fixed construction; outside D1): name -> synthetic_row arguments
CONTROL_INPUTS = {
    "rect": {"lx": 1.0, "nx": 8, "ly": 1.0, "ny": 10, "N": 2, "seed": 11},  # h_x = 1/8 != h_y = 1/10
    "rect_D2": {"lx": 1.0, "nx": 8, "ly": 1.0, "ny": 10, "N": 2, "seed": 11, "D": 2.0},
    "square": {"lx": 1.0, "nx": 9, "ly": 1.0, "ny": 9, "N": 1, "seed": 12},
    "zero_prev": {"lx": 1.0, "nx": 10, "ly": 0.75, "ny": 8, "N": 1, "seed": 13, "zero_prev": True},
}

QAEB = "scripts/r5_cccc_q4_energy_bound_premises_audit.py"
_BE_SUM = '"BE": bnd["B1"] + bnd["B2"] + bnd["B3"] + bnd["B4"]'
_BP_QAEB = 'return kc["kappa_P"] * f["Sq"] + kc["kappa_L"] * f["Sl"] + (1 + kc["gamma_L"]) * kc["nu"] * f["S2"]'

# id, target, category, old, new, expected {input: verdict}, expected effect and channel, reason
MUTATIONS = [
    ("P-01", "primary", "term of B_E omitted (G-5 analogue: B1 dropped)", _BE_SUM, '"BE": bnd["B2"] + bnd["B3"] + bnd["B4"]',
     {"rect": "DETECTED", "zero_prev": "DETECTED"}, "B_E decreases by B1; comparison at BE (B1 itself unchanged); "
     "the inequality |E_s| <= B_E usually still holds", "the full bound is computed independently: the sum differs"),
    ("P-02", "primary", "wrong sign", '"B3": beta * dh_a + beta**2 / 2', '"B3": beta * dh_a - beta**2 / 2',
     {"rect": "DETECTED"}, "B3 decreases by beta^2; comparison at B3 (unsafe)", "second-order term subtracted"),
    ("P-03", "primary", "upper bound replaced by the lower one (denominator gets the upper root)",
     "denom = sD_lo * gq.mu_low", "denom = sD_hi * gq.mu_low", {"rect": "INAPPLICABLE", "rect_D2": "DETECTED"},
     "B_omega decreases when D is not a square; comparison at B_omega (unsafe)",
     "D = 1: the root of D is exact (lo = hi), the defect has no effect; D = 2: effective"),
    ("P-04", "primary", "root rounded down instead of up", 'dh_a = sqrt_bounds(sums["d"]["a2"])[1]',
     'dh_a = sqrt_bounds(sums["d"]["a2"])[0]', {"rect": "DETECTED"}, "||d^||_a below the exact root; comparison at d_a (unsafe)",
     "upper root replaced by the lower one"),
    ("P-05", "primary", "wrong grid-step coefficient", "self.Lam = 4 * rx + 4 * ry", "self.Lam = 4 * rx + 4 * rx",
     {"square": "INAPPLICABLE", "rect": "DETECTED"}, "Lambda wrong unless h_x = h_y; comparison at eps_s",
     "on a square mesh 1/h_x^2 = 1/h_y^2 and the defect has no effect"),
    ("P-06", "primary", "wrong normalisation", "base = gq.hx * gq.hy / 4 * p2(2 * E) / gq.Z**2",
     "base = gq.hx * gq.hy / 2 * p2(2 * E) / gq.Z**2", {"rect": "DETECTED"}, "S_q, S_2, ||.||_a^2 doubled; comparison at p.Sq",
     "weight normalisation 4 w_I w_J/(h_x h_y) misapplied"),
    ("P-07", "primary", "previous and current fields swapped", '"B1": eps_s * (dh_a + beta)', '"B1": eps_p * (dh_a + beta)',
     {"rect": "DETECTED", "zero_prev": "DETECTED"}, "B1 uses eps_p for eps_s; comparison at B1 (unsafe on zero_prev: eps_p = 0)",
     "formation bound of the wrong field"),
    ("P-08", "primary", "input array substituted (residual of the previous field)",
     "AI = apply_A(gq, ws[0])\n    res_h2 = residual_h2(gq, ws[0], ws[1], AI)",
     "AI = apply_A(gq, wp[0])\n    res_h2 = residual_h2(gq, wp[0], wp[1], AI)", {"rect": "DETECTED"},
     "D2 residual of w^_p instead of w^_s; comparison at res_h2", "B_omega must bound ||w~_s - omega||_a"),
    ("P-09", "primary", "rational sum rounded prematurely", _BE_SUM,
     '"BE": Fraction(float(bnd["B1"] + bnd["B2"] + bnd["B3"] + bnd["B4"]))', {"rect": "DETECTED"},
     "B_E replaced by its nearest binary64; comparison at BE (unsafe when rounded down)", "exact sum lost before the comparison"),
    ("P-10", "primary", "wrong number of terms (load terms not counted in M)",
     "M = (nx + 1) * (ny + 1) + (nx - 1) * (ny - 1)", "M = (nx + 1) * (ny + 1)", {"rect": "DETECTED"},
     "L = ceil(log2 M) too small; comparison at grid.L", "the pairwise depth of pi_double counts all M terms"),
    ("P-11", "primary", "second-order term lost (nu S_2 in B_P)", _BP_QAEB,
     'return kc["kappa_P"] * f["Sq"] + kc["kappa_L"] * f["Sl"]', {"rect": "DETECTED"},
     "B2 decreases by (1 + gamma_L) nu (S_2(w^_p) + S_2(w^_s)); comparison at B2", "L-5 remainder dropped"),
    # second path
    ("S-01", "second", "term of B_E omitted", "    BE = B1 + B2 + B3 + B4\n", "    BE = B2 + B3 + B4\n",
     {"rect": "DETECTED"}, "comparison with the primary at BE", "B1 dropped"),
    ("S-02", "second", "wrong sign", "    B3 = beta * dh_a + beta**2 / 2\n", "    B3 = beta * dh_a - beta**2 / 2\n",
     {"rect": "DETECTED"}, "comparison at B3; part bound |E^(3)| <= B3 may fail", "second-order term subtracted"),
    ("S-03", "second", "upper bound replaced by the lower one (denominator gets the upper root)",
     'denom = r(rad["D"], numerator=False) * kc["mu_low"]', 'denom = r(rad["D"]) * kc["mu_low"]',
     {"rect": "INAPPLICABLE", "rect_D2": "DETECTED"}, "comparison at B_omega (D = 2)", "D = 1: exact root"),
    ("S-04", "second", "root rounded down instead of up (upper root = floor/floor)",
     "lo, hi = Fraction(math.isqrt(a), _isqrt_up(b)), Fraction(_isqrt_up(a), math.isqrt(b))",
     "lo, hi = Fraction(math.isqrt(a), _isqrt_up(b)), Fraction(math.isqrt(a), _isqrt_up(b))", {"rect": "DETECTED"},
     "self-certification by squaring refuses the root (UNRESOLVED)", "hi below the root"),
    ("S-05", "second", "wrong grid-step coefficient", "self.Lam = 4 * self.cx + 4 * self.cy", "self.Lam = 4 * self.cx + 4 * self.cx",
     {"square": "INAPPLICABLE", "rect": "DETECTED"}, "comparison at grid.Lambda", "square mesh: no effect"),
    ("S-06", "second", "wrong normalisation", '"Sq": m.D / 2 * s_tA', '"Sq": m.D * s_tA', {"rect": "DETECTED"},
     "comparison at p.Sq; reference (unit perturbation S_q = ||w||_a^2/2)", "factor D/2 lost"),
    ("S-07", "second", "previous and current fields swapped", "    B1 = eps_s * (dh_a + beta) + Bw * (eps_p + eps_s)\n",
     "    B1 = eps_p * (dh_a + beta) + Bw * (eps_p + eps_s)\n", {"rect": "DETECTED", "zero_prev": "DETECTED"},
     "comparison at B1", "formation bound of the wrong field"),
    ("S-08", "second", "input array substituted (residual of the previous field)", 'rad["res_h2"] = residual_h2(m, fs["computed"])',
     'rad["res_h2"] = residual_h2(m, fp["computed"])', {"rect": "DETECTED"}, "comparison at res_h2", "wrong field"),
    ("S-09", "second", "rational sum rounded prematurely", "    BE = B1 + B2 + B3 + B4\n", "    BE = Fraction(float(B1 + B2 + B3 + B4))\n",
     {"rect": "DETECTED"}, "comparison at BE", "exact sum lost"),
    ("S-10", "second", "wrong number of terms", "    M = m.n_nodes + m.n_int\n", "    M = m.n_nodes\n", {"rect": "DETECTED"},
     "comparison at grid.L", "load terms not counted"),
    ("S-11", "second", "second-order term lost (nu S_2 in B4)",
     '    B4 = kc["kappa_Q"] * sd["Sq"] + (1 + kc["gamma_LQ"]) * kc["nu"] * sd["S2"]\n', '    B4 = kc["kappa_Q"] * sd["Sq"]\n',
     {"rect": "DETECTED"}, "comparison at B4", "L-5 remainder of Q^ dropped"),
    # null mutation: the same value written differently; must be INAPPLICABLE (no channel reads the mutation flag)
    ("N-01", "primary", "null mutation (equivalent rewrite)", '"B3": beta * dh_a + beta**2 / 2',
     '"B3": beta * dh_a + beta * beta * Fraction(1, 2)', {"rect": "INAPPLICABLE"}, "no effect", "control of the method"),
]


def references(mod) -> dict:
    """Analytic values of reference constructions, against the (possibly mutated) second path. Closed forms derived by hand in the
    report (sect. 5); computed here without the field functions of the second path."""
    out = {}
    lx, nx, ly, ny = 1.0, 8, 1.0, 10  # h_x != h_y
    m = mod.Mesh(lx, nx, ly, ny)
    hx, hy, D, q = Fraction(lx) / nx, Fraction(ly) / ny, Fraction(1), Fraction(1)
    # unit perturbation at an interior node away from the boundary: t has 5 nonzero values, A = |t|
    w = [Fraction(0)] * m.n_int
    w[m.col(4, 5)] = Fraction(1)
    fq = mod.field_quantities(m, w)
    a2 = D * hx * hy * ((2 / hx**2 + 2 / hy**2) ** 2 + 2 / hx**4 + 2 / hy**4)
    out["unit_a2"] = fq["a2"] == a2
    out["unit_Sq_S2"] = fq["Sq"] == a2 / 2 and fq["S2"] == a2 / 2
    out["unit_Sl_h2"] = fq["Sl"] == abs(q) * hx * hy and fq["h2"] == hx * hy
    # constant field c: a_h = D [6 c^2 (n_y - 1) h_y/h_x^3 + 6 c^2 (n_x - 1) h_x/h_y^3 + 8 c^2/(h_x h_y)]
    c = Fraction(3, 4)
    fc = mod.field_quantities(m, [c] * m.n_int)
    out["constant_a2"] = fc["a2"] == D * (6 * c**2 * (ny - 1) * hy / hx**3 + 6 * c**2 * (nx - 1) * hx / hy**3 + 8 * c**2 / (hx * hy))
    out["constant_Sl_h2"] = fc["Sl"] == abs(q) * hx * hy * (nx - 1) * (ny - 1) * c and fc["h2"] == hx * hy * (nx - 1) * (ny - 1) * c**2
    return out


def _row(name: str):
    a = dict(CONTROL_INPUTS[name])
    return sp().synthetic_row(a.pop("lx"), a.pop("nx"), a.pop("ly"), a.pop("ny"), **a)


def _primary(mod, row) -> dict:
    return sp().primary_qaeb(mod, row["mesh"], row["Xp"], row["Yp"], row["Xs"], row["Ys"], row["dh"], row["wh_p"], row["wh_s"])


def _channels_primary(row, base_sp: dict, mutated: dict) -> dict:
    S = sp()
    cmp = S.compare(base_sp, mutated)
    fired = {}
    first = cmp["first_divergence"]
    fired["comparison"] = None if first is None else f"{first['component']}: {first['status']}"
    BE = mutated.get("BE")
    fired["inequality"] = (None if BE is None or abs(base_sp["E_s"]) <= Fraction(BE)
                           else "|E_s| > B_E of the primary")
    parts = base_sp["parts"]
    bad = [k for k, b in (("E1", "B1"), ("E2", "B2"), ("E3", "B3"), ("E4", "B4"))
           if mutated.get(b) is not None and abs(parts[k]) > Fraction(mutated[b])]
    fired["parts"] = ("|E^(k)| > B_k for " + ", ".join(bad)) if bad else None
    return {"fired": fired, "comparison": cmp, "inequality_holds": BE is not None and abs(base_sp["E_s"]) <= Fraction(BE)}


def _channels_second(row, res: dict, primary: dict, refs: dict) -> dict:
    S = sp()
    fired = {}
    if res.get("status") != S.CERTIFIED or "components" not in res:
        fired["self_check"] = "refused: " + "; ".join(f"{r['component']}: {r['reason']}" for r in res.get("reasons", []))[:300]
    elif res.get("contradictions"):
        fired["self_check"] = "; ".join(res["contradictions"])
    else:
        fired["self_check"] = None
    if "components" in res:
        cmp = S.compare(res, primary)
        first = cmp["first_divergence"]
        # the labels of ``compare`` speak about the primary; here the second path is the mutated side: only "the paths disagree"
        fired["comparison"] = None if first is None else f"the two paths disagree first at {first['component']}"
        rel = cmp["BE_own_vs_primary"] or {}
        if fired["comparison"] is None and rel.get("own_rule_is_upper_bound") not in (None, "ABOVE_OR_EQUAL"):
            fired["comparison"] = f"own-rule B_E not an upper bound ({rel['own_rule_is_upper_bound']})"
    else:
        fired["comparison"] = None
    bad = [k for k, v in refs.items() if not v]
    fired["reference"] = ("analytic reference differs: " + ", ".join(bad)) if bad else None
    return {"fired": fired}


def _signature(res_or_primary: dict, second: bool) -> dict:
    """Comparable output of an implementation on one input (to decide whether the mutation is effective)."""
    if second:
        if "components" not in res_or_primary:
            return {"status": res_or_primary.get("status"), "reasons": str(res_or_primary.get("reasons"))}
        c = dict(res_or_primary["components"])
        c["own_BE"] = res_or_primary["assembly"]["BE"]
        c["status"] = res_or_primary["status"]
        return c
    return {k: v for k, v in res_or_primary.items() if not k.startswith("_")}


def run_mutation(m: tuple) -> dict:
    mid, target, category, old, new, expected, effect, reason = m
    S = sp()
    rel = QAEB if target == "primary" else SECOND_TOOL
    src = (ROOT / rel).read_text(encoding="utf-8")
    if src.count(old) != 1:
        raise RuntimeError(f"{mid}: the text to mutate occurs {src.count(old)} times")
    mod = _load(rel, f"_be2_mut_{mid.replace('-', '_')}", src.replace(old, new))
    per = {}
    for inp in expected:
        row = _row(inp)
        base = S.run_row(row, parts=True)
        base_primary = _primary(S.load_qaeb(), row)
        if target == "primary":
            mutated = _primary(mod, row)
            effective = _signature(mutated, False) != _signature(base_primary, False)
            ch = _channels_primary(row, base, mutated)
            fired = ch["fired"]
            extra = {"inequality_still_holds": ch["inequality_holds"],
                     "first_divergence": ch["comparison"]["first_divergence"]}
        else:
            res = mod.run_row(row, parts=True)
            effective = _signature(res, True) != _signature(base, True)
            ch = _channels_second(row, res, base_primary, references(mod))
            fired = ch["fired"]
            extra = {"status_of_mutated_second_path": res.get("status")}
        detected = any(v for v in fired.values())
        verdict = "INAPPLICABLE" if not effective else ("DETECTED" if detected else "UNDETECTED")
        if not effective and detected:
            verdict = "INCONSISTENT (a channel fired without an effect)"
        per[inp] = {"expected": expected[inp], "verdict": verdict, "as_expected": verdict == expected[inp], "effective": effective,
                    "channels_fired": {k: v for k, v in fired.items() if v}, **extra}
    return {"id": mid, "target": target, "category": category, "expected_effect_and_channel": effect, "reason": reason,
            "mutation": {"file": rel, "old": old, "new": new}, "inputs": per,
            "as_expected": all(x["as_expected"] for x in per.values())}


def run_all() -> dict:
    t0 = time.perf_counter()
    S = sp()
    base_refs = references(S)
    out = [run_mutation(m) for m in MUTATIONS]
    counts = {}
    for r in out:
        for x in r["inputs"].values():
            counts[x["verdict"]] = counts.get(x["verdict"], 0) + 1
    return {"document": "R5-V Q4: mutation research of the B_E computations", "data_status": S.DATA_STATUS,
            "provenance": S.provenance(), "references_unmutated": base_refs, "results": out, "counts_per_input": counts,
            "all_as_expected": all(r["as_expected"] for r in out),
            "undetected": [r["id"] for r in out for x in r["inputs"].values() if x["verdict"] == "UNDETECTED"],
            "seconds": round(time.perf_counter() - t0, 1)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    sp().guard_out(a.out)
    a.out.mkdir(parents=True, exist_ok=True)
    res = run_all()
    with open(a.out / "mutations.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, ensure_ascii=False, default=sp()._json)
    for r in res["results"]:
        print(r["id"], {k: v["verdict"] for k, v in r["inputs"].items()}, "as expected" if r["as_expected"] else "NOT AS EXPECTED",
              flush=True)
    print(json.dumps({"counts": res["counts_per_input"], "all_as_expected": res["all_as_expected"], "seconds": res["seconds"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
