# Rules fixed BEFORE any R3 margin computation (written after the r = 1 calibration only)

Date: 2026-10-14. Inputs seen so far: accepted R2 margins (STATUS.md, step 7) and the r = 1 calibration
(scratch/r7/calibration_r1.csv). No R3 (r != 1) margin has been computed when this file is written.

1. Quantities per case (r, family, nx, ny, s): Q_sur, R_prev, R_now, R_f from the surrogate; tau_R = 10 (R_prev + R_now + R_f);
   m_up = Q_sur / (10 tau_R) (gamma = 0: upper bound of the margin for these Q, R); gamma allowance g_allow = Q/10 - tau_R.
2. Surrogate uncertainty model [engineering choice, preliminary]:
   e_sur = 2 * C_s * ((hx/l_min)^2 + (hy/l_min)^2) / 2,  l_min = min(a, b),
   C_s = max over the calibration grids n >= 20 of |dm_rel| * n^2 at r = 1 (s = 2, 3; for s = 4 the s = 3 value).
   At r = 1 and nx = ny = n this is 2 * C_s / n^2 (factor 2 over the calibration law).
3. Categories (s = 2, 3; s = 4 diagnostic only):
   - MANDATORY_CANDIDATE: m_up * (1 - e_sur) >= 4  and  min(nx, ny) >= 20;
   - RESOLVABLE_NO_HEADROOM: 1 <= m_up * (1 - e_sur) < 4 (or >= 4 with min(nx, ny) < 20);
   - BORDERLINE: m_up * (1 - e_sur) < 1 <= m_up * (1 + e_sur);
   - EXPECTED_UNRESOLVED: m_up * (1 + e_sur) < 1.
   Factor 4 = headroom of one grid doubling (tau ~ n^2 at fixed Q); it reproduces the accepted R2 rule at r = 1
   (s = 3: 5.75 at n = 160 mandatory, 1.35 at n = 320 not). It is not derived from R3 data.
4. Grids with min(nx, ny) < 20 are outside the calibration domain (n >= 20): reported, flagged DIAGNOSTIC, never MANDATORY_CANDIDATE.
5. All MANDATORY_CANDIDATE statements remain conditional on the future R3 solver: |gamma_s| <= g_allow (measured), Q, R of the real
   discrete fields within e_sur of the surrogate. Nothing is declared proven by the surrogate.
