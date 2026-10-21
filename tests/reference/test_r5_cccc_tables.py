"""Checks of the transcribed published data C (kkvim_reference.r5_cccc_tables): conversions and internal consistency.
Data, not tolerances (D-049.8)."""

from kkvim_reference import r5_cccc_tables as tb


def test_roles_and_sources_follow_d049_8():
    roles = {p.source: p.role for p in tb.ALPHA}
    assert roles[tb.L120] == roles[tb.L058] == tb.CONTROL
    assert roles[tb.L114] == roles[tb.L119] == roles[tb.L121] == tb.MAGNITUDE
    assert all("not an independent reference" in p.note for p in tb.ALPHA if p.source == tb.L121)
    for r in (3.0, 5.0):  # one published source at r = 3, 5 (L-058)
        assert {p.source for p in tb.alpha_values(r, tb.CONTROL)} == {tb.L058}


def test_conversions_and_printed_rounding_are_consistent_between_sources():
    """P7 (10 digits) rounded to the printed step of L-058 gives the L-058 value at r = 1, 2 (both Hencky)."""
    p7 = {p.r: p.alpha for p in tb.ALPHA if p.source == tb.L120}
    bat = {p.r: p.alpha for p in tb.ALPHA if p.source == tb.L058}
    for r in (1.0, 2.0):
        assert round(p7[r], 8) == bat[r]
    morley = [p for p in tb.ALPHA if p.source == tb.L119]
    assert morley[0].alpha == 0.0204 / 16  # half shorter side b: alpha = value (b/l)^4 = value / 16
    assert tb.L120_TABLE_I[2000][0] == 1.265319087 and tb.L120_HENCKY_E_TYPO == 3.981200775


def test_magnitude_sources_agree_with_control_in_their_printed_digits_only():
    """L-114 and L-119 at r = 1, 2 agree with L-120 to their printed step (order of magnitude); no tolerance is derived."""
    p7 = {p.r: p.alpha for p in tb.ALPHA if p.source == tb.L120}
    for p in tb.ALPHA:
        if p.source in (tb.L114, tb.L119) and p.r in p7:
            assert abs(p.alpha - p7[p.r]) <= 2 * p.step, p
