---
id: param.stop.s_max
title: Максимум полушагов s_max
category: iterations
summary: Предельный бюджет итераций (по умолчанию 100). Достижение s_max — статус MAX_HALF_STEPS: критерий остановки не выполнен; это не гарантия сходимости.
keywords: s_max, максимум полушагов, бюджет итераций, MAX_HALF_STEPS, STAGNATED
related: param.stop.s_min, status.overview
sources: docs/NUMERICAL_SPEC.md, разд. 11.1–11.4; docs/DECISIONS.md, D-032, D-040, D-043
status: draft
---
Если критерий остановки не выполнен к s = s_max, расчёт останавливается со статусом MAX_HALF_STEPS. Расчёт также
останавливается со статусом STAGNATED, если 10 полушагов подряд отношение последовательных изменений θ ≥ 0,9 при
ε^E > tol_E. В обоих случаях поле последнего полушага показывается только как диагностика, а не как результат.
