"""Published CCCC values (reference C of R5-V): data, not tolerances (D-049.8, item 1; NS 6; PL 4.7).

Every value is converted to alpha = w(centre) D / (q l^4), l = the shorter FULL side, r = longer/shorter side. Transcriptions:
docs/R5_CCCC_REFERENCE_LITERATURE.review.md (LR) 4.2-4.5, 4.8; pages compared by QA in docs/R5_CCCC_REFERENCES.q0-readiness.md 3
(P7 p. 6 of the report, L-058 p. 13). ``step`` is the unit of the last printed digit after the conversion (the printed rounding
is half of it). Roles (D-049.8): L-120 and L-058 control A, B and the final numbers; at r = 3, 5 L-058 is the only published
source; L-114, L-119, L-121 confirm the order of magnitude only; L-121 is not an independent reference.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Published:
    source: str  # L-number and short reference
    place: str  # table, page
    r: float
    alpha: float  # converted to alpha
    step: float  # unit of the last printed digit of alpha
    method: str
    role: str
    note: str = ""


L120 = "L-120 Taylor & Govindjee, UCB/SEMM-2002/09"
L058 = "L-058 Batista, arXiv:1001.3016"
L114 = "L-114 Li, Zhong, Tian, Liu, AML 22 (2009)"
L119 = "L-119 Morley, QJMAM XVI (1963)"
L121 = "L-121 An, Gu, Su, JBSMSE 38 (2016)"
CONTROL = "control of A, B and final numbers"
MAGNITUDE = "order of magnitude only"

ALPHA = (
    # L-120, table II (w D/(q a^4) 1e3, a the shorter side; 'Double cosine' and 'Hencky' agree in all printed digits)
    Published(L120, "table I, report p. 6", 1.0, 1.265319087e-3, 1e-12, "Ritz (M = 2000) and Hencky", CONTROL),
    Published(L120, "table II, report p. 6", 1.2, 1.724870503e-3, 1e-12, "Ritz 2000 x 2000 and Hencky", CONTROL),
    Published(L120, "table II, report p. 6", 1.4, 2.068143209e-3, 1e-12, "Ritz 2000 x 2000 and Hencky", CONTROL),
    Published(L120, "table II, report p. 6", 1.6, 2.299966977e-3, 1e-12, "Ritz 2000 x 2000 and Hencky", CONTROL),
    Published(L120, "table II, report p. 6", 1.8, 2.446162656e-3, 1e-12, "Ritz 2000 x 2000 and Hencky", CONTROL),
    Published(L120, "table II, report p. 6", 2.0, 2.532955769e-3, 1e-12, "Ritz 2000 x 2000 and Hencky", CONTROL),
    Published(L120, "table II, report p. 6", 20.0, 2.604166667e-3, 1e-12, "Hencky (Double cosine 2.60416666)", CONTROL),
    # L-058, table 5a (w = alpha q a'^4/D, a' = 2a the full side; 8 decimals: the author's stopping heuristic)
    Published(L058, "table 5a, p. 13", 1.0, 0.00126532, 1e-8, "Hencky (quadruple precision)", CONTROL),
    Published(L058, "table 5a, p. 13", 1.5, 0.00219652, 1e-8, "Hencky", CONTROL),
    Published(L058, "table 5a, p. 13", 2.0, 0.00253296, 1e-8, "Hencky", CONTROL),
    Published(L058, "table 5a, p. 13", 3.0, 0.00261723, 1e-8, "Hencky", CONTROL, "the only published source at r = 3"),
    Published(L058, "table 5a, p. 13", 4.0, 0.00260659, 1e-8, "Hencky", CONTROL),
    Published(L058, "table 5a, p. 13", 5.0, 0.00260423, 1e-8, "Hencky", CONTROL, "the only published source at r = 5"),
    Published(L058, "table 5a, p. 13", 10.0, 0.00260417, 1e-8, "Hencky", CONTROL),
    # L-114, table 1 (q a^4/D, a the shorter full side)
    Published(L114, "table 1, p. 1825", 1.0, 0.00127, 1e-5, "finite integral transform, 100 terms", MAGNITUDE),
    Published(L114, "table 1, p. 1825", 1.5, 0.00220, 1e-5, "finite integral transform", MAGNITUDE),
    Published(L114, "table 1, p. 1825", 2.0, 0.00253, 1e-5, "finite integral transform", MAGNITUDE),
    # L-119, table 4, second approximation (q0 b^4/D, b the half shorter side: alpha = value / 16)
    Published(L119, "table 4, p. 113", 1.0, 0.0204 / 16, 1e-4 / 16, "Papkovich-Fadle, 2nd approximation", MAGNITUDE),
    Published(L119, "table 4, p. 113", 2.0, 0.0405 / 16, 1e-4 / 16, "Papkovich-Fadle, 2nd approximation", MAGNITUDE),
    # L-121, table 1, N = 128 (w/(a^4 q/D), a the full side, c = b/a >= 1); non-monotone in N; not independent
    Published(L121, "table 1, p. 604", 1.0, 0.0012654348, 1e-10, "GITT, N = 128", MAGNITUDE, "not an independent reference"),
    Published(L121, "table 1, p. 604", 1.2, 0.0017272653, 1e-10, "GITT, N = 128", MAGNITUDE, "not an independent reference"),
    Published(L121, "table 1, p. 604", 1.4, 0.0020681748, 1e-10, "GITT, N = 128", MAGNITUDE, "not an independent reference"),
    Published(L121, "table 1, p. 604", 2.0, 0.0025331099, 1e-10, "GITT, N = 128", MAGNITUDE, "not an independent reference"),
)

# L-120, table I: the Ritz sequence of the square by M = N (w D/(q a^4) 1e3 and E Work D/(q^2 a^4) 1e4 as printed; the
# E column is int int w D/(q l^6) 1e4, Reviewer 62db9f2 R-17), and the printed condition number of the scaled capacity matrix
L120_TABLE_I = {  # M: (w 1e3, E 1e4, kappa or None)
    200: (1.265319036, 3.891200386, 3.9208),
    400: (1.265319081, 3.891200726, 4.0035),
    600: (1.265319086, 3.891200761, 4.0442),
    800: (1.265319087, 3.891200769, 4.0702),
    1000: (1.265319087, 3.891200772, 4.0890),
    1200: (1.265319087, 3.891200773, None),
    1400: (1.265319087, 3.891200774, None),
    1600: (1.265319087, 3.891200774, None),
    1800: (1.265319087, 3.891200774, None),
    2000: (1.265319087, 3.891200775, None),
}
L120_HENCKY_E_TYPO = 3.981200775  # printed in the 'Hencky' row; transposed digits of 3.891200775 (OV-6, R-17)

# L-120, table II: E Work D/(q^2 a^4) 1e4 (both methods, a the shorter side)
L120_TABLE_II_WORK = {1.2: 6.41537043, 1.4: 9.14890620, 1.6: 11.94175880, 1.8: 14.73958338, 2.0: 17.53009520}

STRIP_LIMIT = 1.0 / 384.0  # alpha of the clamped strip (r -> infinity)


def alpha_values(r: float, role: str | None = None):
    return [p for p in ALPHA if p.r == r and (role is None or p.role == role)]
