"""QA tests: independent qualification of ParsevalRecursion for the rectangle a x b (R3; D-036, D-030).

Written by QA. Reference values HP come from QA's own high-precision chain (mpmath 1.4.1, 50 digits), which does
not use the sine series of the module: half-step 1 by the closed form В-29 (MATH_SPEC.md, 9.1); half-steps
s >= 2 by the closed-form solution of D(A f'''' - 2B f'' + C f) = F, f = f'' = 0 at both ends of ell_s, through
the complex roots k^2 = (B + i sqrt(AC - B^2))/A; A, B, C, F of the next half-step by quadrature of f^2, f'^2,
f''^2 and q f over the multiplier's own length (MATH_SPEC.md, 4.2-4.4); Pi_s = -q I_X I_Y / 2. The generator
is described in docs/R3_PARSEVAL_REFERENCE.qa-review.md (section 2).

Tolerances are the module's own error budgets (input_error for G_s, field_error for fields, energy.error for
Pi_s), fixed before the comparison; the high-precision values carry 22 digits. Own rule И-12 check by the tail
formulas of NUMERICAL_SPEC.md, 15.4. Diagnostics (r = 2, two starts) are not asserted here.
"""

import math

import numpy as np
import pytest

from kkvim_reference.parseval_recursion import ParsevalRecursion

U = 2.0**-53
EPS = 1e-14
S_HP = 5
# fmt: off
HP = {
    (1.0, 0.2): {
        1: ("5.0e-1", "4.934802200544679309417", "4.870454551700121861822e+1", "6.366197723675813430755e-1", "2.451783921770771661975e-5", "1.605237205135224337602e-5", "9.529777106715961939227e-6", "-9.993272277962160713846e-7"),
        2: ("6.061171988518868654117e-11", "1.495771723340825618806e-8", "3.696151274970647944105e-6", "3.139479077376846233028e-6", "2.074611707897702694366e-5", "1.815830372102129941019e-5", "8.063755938561404319515e-6", "-1.078852603309586712137e-6"),
        3: ("5.206691453525814764175e-1", "7.086739260145462561925", "3.502688474484685594945e+2", "6.872812824801552458124e-1", "2.074224588054883594675e-5", "1.815623225166361747164e-5", "8.066414567414953077154e-6", "-1.078854398060087645217e-6"),
        4: ("6.060574189939565471569e-11", "1.495640252140955629556e-8", "3.69619797918903048216e-6", "3.139484300130751174131e-6", "2.074205472306710693011e-5", "1.815622019150771564667e-5", "8.066340228526861152863e-6", "-1.078854398231651610112e-6"),
        5: ("5.206673568763901985376e-1", "7.086892921523766054691", "3.503020862522026924464e+2", "6.872812825894496066015e-1", "2.074205434355829709397e-5", "1.8156219987311477868e-5", "8.066340485521460299691e-6", "-1.078854398231668551804e-6"),
    },
    (3.0, 1.0): {
        1: ("1.5", "1.644934066848226436472", "1.803872056185230319193", "1.909859317102744029227", "1.341816610474809359703e-2", "8.787536130552944862955e-3", "5.225516726574352090594e-3", "-8.209455009732860706374e-3"),
        2: ("9.086408907290791709986e-5", "8.96972492184887359678e-4", "8.869387739623350368182e-3", "8.596921182850893248006e-3", "1.222500106530610617191e-2", "9.517241375260431756879e-3", "4.760855324822767274168e-3", "-8.526682275730660773918e-3"),
        3: ("1.516392429529696155003", "1.816114471872805526354", "3.806593828259881068926", "1.983659520512914603047", "1.222306537443985551873e-2", "9.516318722808999838941e-3", "4.762180948838975692895e-3", "-8.526691391828941966213e-3"),
        4: ("9.085645197977935403325e-5", "8.969062253532080162303e-4", "8.869577663950694029937e-3", "8.596930374043420920443e-3", "1.222297590275975759548e-2", "9.516335195606985101677e-3", "4.762146090125766652943e-3", "-8.526691392878090341452e-3"),
        5: ("1.516387712699402352282", "1.816126304986271173012", "3.806820691671349058944", "1.983659520756989715812", "1.222297567571574082582e-2", "9.516335086920689998477e-3", "4.762146243985920379179e-3", "-8.526691392878214095311e-3"),
    },
    (1.0, 1.4): {
        1: ("5.0e-1", "4.934802200544679309417", "4.870454551700121861822e+1", "6.366197723675813430755e-1", "7.134507949241997024934e-3", "4.716022777065781603564e-3", "2.973096436364581349441e-3", "-2.08814590980048911068e-3"),
        2: ("3.729548473269188884123e-5", "1.88718131731614701601e-4", "9.945177647105125860798e-4", "6.560103849852791624442e-3", "7.078379751975468666608e-3", "4.752208721626207473813e-3", "2.949706660298747234283e-3", "-2.094388559885947154944e-3"),
        3: ("5.000361134910280113173e-1", "4.938053662589132921676", "4.900544427346261365551e+1", "6.385229892154664962945e-1", "7.077811450114837055013e-3", "4.751970261279620298554e-3", "2.950106090427686186403e-3", "-2.094388679378953753211e-3"),
        4: ("3.729394231143448135703e-5", "1.887150562911015260741e-4", "9.947301481818346258273e-4", "6.560104224132210271793e-3", "7.077808506547350015477e-3", "4.751971092144461085708e-3", "2.950104863517866349459e-3", "-2.094388679388100935726e-3"),
        5: ("5.000360019901856808031e-1", "4.938052782953404720945", "4.900545615025890422052e+1", "6.385229892182552271445e-1", "7.077808462957830661244e-3", "4.751971073846035296514e-3", "2.950104894120925659628e-3", "-2.094388679388101637752e-3"),
    },
    (1.0, 2.0): {
        1: ("5.0e-1", "4.934802200544679309417", "4.870454551700121861822e+1", "6.366197723675813430755e-1", "1.017883471193769323017e-2", "6.805653104045985693582e-3", "4.621423414328601859735e-3", "-4.39394244757683831533e-3"),
        2: ("1.137292593475090362092e-4", "2.86324173696340378922e-4", "8.456933635928229787797e-4", "1.380397731360375045019e-2", "1.012031383219139203063e-2", "6.843857869663453650515e-3", "4.594853598476352829448e-3", "-4.403499331139597977273e-3"),
        3: ("5.000191821064732204775e-1", "4.936527669977943462147", "4.886377857478984767658e+1", "6.380044288829671516418e-1", "1.011920630682522071809e-2", "6.843379536557825936547e-3", "4.595667513072648048105e-3", "-4.403499564912846716824e-3"),
        4: ("1.137232392388114036837e-4", "2.863288787459445239868e-4", "8.461897037306758354292e-4", "1.380397804643016405e-2", "1.011920238724085939331e-2", "6.843380662886627276486e-3", "4.595665732981802194081e-3", "-4.403499564935714862109e-3"),
        5: ("5.00019077709205174242e-1", "4.936526789429483193574", "4.88637837860249496309e+1", "6.380044288862804204993e-1", "1.011920227875783792788e-2", "6.843380616019022999312e-3", "4.595665812646551314507e-3", "-4.403499564935717103029e-3"),
    },
    (2.0, 1.0): {
        1: ("1.0", "2.467401100272339654709", "6.088068189625152327278", "1.273239544735162686151", "1.05908655332411724523e-2", "6.939798115704893941527e-3", "4.140918623129231299326e-3", "-4.326048395980506696595e-3"),
        2: ("5.672671642306441626692e-5", "5.600397010063962209795e-4", "5.543055525285119892876e-3", "6.795340929943134331226e-3", "1.012035071344745246878e-2", "7.235546804094620843923e-3", "3.956952206633162119378e-3", "-4.403497691802874113029e-3"),
        3: ("1.002476905349409142323", "2.524000702440748098782", "7.459078607910349962833", "1.296034367429368711176", "1.011923321902552222442e-2", "7.235056381464533822369e-3", "3.957715374157027834061e-3", "-4.40349956475472487041e-3"),
        4: ("5.672337471463760623727e-5", "5.600115497836227363801e-4", "5.543225882707560427792e-3", "6.795343820224284008305e-3", "1.011920239054828855424e-2", "7.23506397334011788274e-3", "3.957703316885921546035e-3", "-4.403499564935699367282e-3"),
        5: ("1.002475426167744438306", "2.524001853650662965582", "7.459200434553698857232", "1.296034367482632976137", "1.011920228139513978524e-2", "7.235063925319423231606e-3", "3.957703390984040862802e-3", "-4.40349956493571710151e-3"),
    },
    (1.0, 5.0): {
        1: ("5.0e-1", "4.934802200544679309417", "4.870454551700121861822e+1", "6.366197723675813430755e-1", "1.302105326920610847339e-2", "9.138808324444889352312e-3", "9.434947688737555607512e-3", "-1.683011277146276216182e-2"),
        2: ("6.163375615029382069686e-4", "3.354302414298531831882e-4", "6.624418755871746405199e-4", "5.287335864191516838585e-2", "1.296567112653855788165e-2", "9.176794223099459379046e-3", "9.394818245430977502785e-3", "-1.685709930629079598197e-2"),
        3: ("5.000104539385640383214e-1", "4.935741903436380798409", "4.879108208125614959088e+1", "6.376405713302801242321e-1", "1.296378787626062230787e-2", "9.175711781436817597332e-3", "9.396873249914711437097e-3", "-1.685709997230409722078e-2"),
        4: ("6.163042279834563331677e-4", "3.355449004626275021466e-4", "6.634335663057396484257e-4", "5.287336073090803108699e-2", "1.296378413917114176902e-2", "9.175712912769082750265e-3", "9.396870541064758984578e-3", "-1.685709997236981020024e-2"),
        5: ("5.000103759160639576874e-1", "4.935741215682983586217", "4.879108291434870120338e+1", "6.376405713327657986267e-1", "1.29637839532887085874e-2", "9.175712805905639132534e-3", "9.396870743850423890057e-3", "-1.685709997236981668935e-2"),
    },
}
# fmt: on
CONTROL = ((1.0, 0.2), (3.0, 1.0), (2.0, 1.0), (3.0, 2.0), (1.0, 1.0), (1.0, 1.4), (2.0, 3.0), (1.0, 2.0),
           (1.0, 3.0), (1.0, 5.0))  # r = 0.2, 1/3, 1/2, 2/3, 1, 1.4, 1.5, 2, 3, 5 (b/a exact for the given floats)


def recursion(a, b, s_max=S_HP):
    return ParsevalRecursion(a=a, b=b, q=1.0, D=1.0, s_max=s_max)


def rel(x, ref):
    return abs(x - ref) / abs(ref)


@pytest.mark.parametrize("plate", list(HP))
def test_against_independent_high_precision_chain(plate):
    """G_s, centre, two off-centre nodes and Pi_s for s = 1..5 within the module's budgets."""
    a, b = plate
    rec = recursion(a, b)
    for s in range(1, S_HP + 1):
        A, B, C, F, wc, w1, w2, Pi = (float(v) for v in HP[plate][s])
        G, budget = rec.coefficients(s), rec.halfstep(s).budget
        for got, ref in ((G.A, A), (G.B, B), (G.C, C), (G.F, F)):
            assert rel(got, ref) <= budget.input_error + 2 * U, (s, got, ref)
        wmax = float(np.max(np.abs(rec.field_on_grid(s, 8, 8))))
        W = rec.field_on_grid(s, 8, 8)
        for got, ref in ((W[4, 4], wc), (W[2, 3], w1), (W[4, 1], w2)):
            assert abs(got - ref) <= budget.field_error * wmax, (s, got, ref)
        en = rec.energy(s)
        assert abs(en.energy - Pi) <= en.error * abs(Pi) + 4 * U * abs(Pi)


@pytest.mark.parametrize(("a", "b"), CONTROL)
def test_lengths_counters_and_start(a, b):
    """ell_s = b (odd s, Y), a (even s, X); k = s // 2; G_1 from X0 = sin(pi x / a) on (0, a)."""
    rec = recursion(a, b, s_max=10)
    for s in range(1, 11):
        h = rec.halfstep(s)
        ell = b if s % 2 else a
        assert rec.length(s) == ell and h.step.ell == ell
        assert (h.s, h.k, h.variable) == (s, s // 2, "Y" if s % 2 else "X")
    G = rec.coefficients(1)
    for got, ref in ((G.A, a / 2), (G.B, math.pi**2 / (2 * a)), (G.C, math.pi**4 / (2 * a**3)), (G.F, 2 * a / math.pi)):
        assert rel(got, ref) <= 16 * U


def own_relative_tails(G, ell, coeffs, M):
    """Relative remainders of NUMERICAL_SPEC.md 15.4, item 3 (D = q = 1), against sums truncated at M (QA code)."""
    K = 4 * abs(G.F) * ell**4 / (G.A * math.pi**5)
    c = coeffs[: (M + 1) // 2]
    lam = [(2 * j + 1) * math.pi / ell for j in range(len(c))]
    A = ell / 2 * math.fsum(x * x for x in c)
    B = ell / 2 * math.fsum((p * x) ** 2 for p, x in zip(lam, c))
    C = ell / 2 * math.fsum((p * p * x) ** 2 for p, x in zip(lam, c))
    F = 2 * math.fsum(x / p for p, x in zip(lam, c))
    f_max = abs(math.fsum(x if j % 2 == 0 else -x for j, x in enumerate(c)))  # value at the centre
    k = math.pi / ell
    return (K / (8 * M**4) / f_max, ell / 2 * K * K / (18 * M**9) / A, ell / 2 * K * K * k**2 / (14 * M**7) / B,
            ell / 2 * K * K * k**4 / (10 * M**5) / C, (2 * ell / math.pi) * K / (10 * M**5) / abs(F))


@pytest.mark.parametrize(("a", "b"), CONTROL)
def test_rule_i12_and_tail_constant(a, b):
    """The chosen M satisfies the tail criterion of 15.4 with ell_s (QA's own formulas); s = 1: K/|c_1| = (1 + r^2)^2."""
    rec = recursion(a, b, s_max=4)
    for s in range(1, 5):
        h = rec.halfstep(s)
        assert max(own_relative_tails(h.coefficients, rec.length(s), h.step.coefficients(), h.budget.M)) <= EPS
    step = rec.halfstep(1).step
    r = b / a
    assert rel(step.tail_constant / abs(step.coefficient(1)), (1 + r * r) ** 2) <= 64 * U


@pytest.mark.parametrize(("a", "b"), [(1.0, 0.2), (1.0, 2.0), (1.0, 5.0)])
def test_points_agree_with_nodes(a, b):
    """field_at_points at node coordinates equals field_on_grid within field_error plus the argument rounding."""
    rec = recursion(a, b, s_max=4)
    nx, ny = 6, 10
    X, Y = np.meshgrid(a * np.arange(nx + 1) / nx, b * np.arange(ny + 1) / ny, indexing="ij")
    for s in range(1, 5):
        W = rec.field_on_grid(s, nx, ny)
        P = rec.field_at_points(s, X, Y)
        arg = 0.0
        for index in {s, s - 1}:
            if index >= 1:
                c = rec.halfstep(index).step.coefficients()
                arg += math.pi * U * math.fsum((2 * j + 1) * abs(x) for j, x in enumerate(c)) / math.fsum(abs(x) for x in c)
        wmax = float(np.max(np.abs(W)))
        assert np.max(np.abs(P - W)) <= (2 * rec.halfstep(s).budget.field_error + 4 * arg) * wmax


@pytest.mark.parametrize(("a", "b"), [(1.0, 2.0), (1.0, 0.2), (1.0, 5.0)])
def test_second_start_is_the_turned_plate(a, b):
    """swapped() is the plate b x a started by x; its node fields are those of ParsevalRecursion(a=b, b=a) bitwise."""
    rec = recursion(a, b)
    sw, direct = rec.swapped(), recursion(b, a)
    for s in range(1, S_HP + 1):
        assert np.array_equal(sw.field_on_grid(s, 8, 4), direct.field_on_grid(s, 8, 4))


@pytest.mark.parametrize(("a", "b"), [(1.0, 0.2), (1.0, 5.0)])
def test_budget_grows_and_loses_validity(a, b):
    """Budgets grow with s and become invalid (inf) at some s while the values stay finite (Q-022)."""
    rec = recursion(a, b, s_max=30)
    budgets = [rec.halfstep(s).budget for s in range(1, 31)]
    valid = [bud.valid for bud in budgets]
    first_invalid = valid.index(False)
    assert all(valid[:first_invalid]) and not any(valid[first_invalid:])
    errors = [bud.field_error for bud in budgets[:first_invalid]]
    assert all(e2 > e1 for e1, e2 in zip(errors, errors[1:]))
    assert all(np.all(np.isfinite(rec.field_on_grid(s, 4, 4))) for s in range(1, 31))
