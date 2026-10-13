"""KK-VIM independent reference solutions (D-015, D-028, D-033).

This package holds reference solutions computed independently of the main
KK-VIM solver, used by QA to verify it. It must not import ``kkvim``, and
``kkvim`` must not import it (D-028).

Contents:
- ``kkvim_reference.levy`` - independent analytic reference for the first
  KK-VIM half-step s = 1 by Levy's formula (square R2, В-29, В-30), plus
  the full L-107 series (139) for R3 rectangular SSSS plates, with independently
  verified truncation bounds (ЧВ-Л1..Л4); P-12 itself still pending;
- ``kkvim_reference.sine_halfstep`` - sine-series solution of one 1D
  half-step D(A f'''' - 2B f'' + C f) = F with f = f'' = 0 at both ends and
  its analytic truncation bound (NUMERICAL_SPEC.md, 15.1, 15.4);
- ``kkvim_reference.parseval_recursion`` - semi-analytic KK-VIM sequence
  s = 1, ..., s_max (square R2 and rectangular R3 SSSS, uniform load, N = 1; default s_max = 3):
  coefficients G_s by Parseval, sine series of every half-step, fields w_s at
  grid nodes, choice of M from analytic tails, error accumulation, continuous
  energy Pi_s and a diagnostic mode with enlarged M (NUMERICAL_SPEC.md,
  15.2-15.5, 15.7, 17);
- ``kkvim_reference.navier`` - independent Navier double sine series for the
  square SSSS plate under uniform load: w by [L-107, eq. (131), p. 129],
  M_x, M_y, truncation bounds and convergence levels at points and at all
  nodes of the tensor grid; truncated integral of the deflection J_M, I_M
  and energy Pi_M with the remainder bound of section 16 (MATH_SPEC.md,
  В-27; NAVIER_REFERENCE_SPEC.md; NUMERICAL_SPEC.md, section 16, A-10a);
  for R3 also the rectangle a x b, 0.2 <= b/a <= 5, M = 2559 (NavierRectangle;
  D-036, D-037; bounds ЧВ-R2, ЧВ-R2a; independently qualified for deflection,
  docs/R3_NAVIER_REFERENCE.md). All three R3 references share D-038's exact
  symmetric 4u range validation (kkvim_reference._r3_range).

The Levy cross-check P-12 is not implemented yet.
"""
