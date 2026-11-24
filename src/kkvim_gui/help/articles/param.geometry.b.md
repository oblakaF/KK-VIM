---
id: param.geometry.b
title: Ширина пластины b
category: model
summary: Размер пластины вдоль оси y. Вводится в m, cm или mm; смена единицы в поле не пересчитывает число.
keywords: ширина, b, размер, геометрия, пластина, ось y, отношение сторон
related: param.geometry.a, param.geometry.h, param.grid.ny
sources: docs/DECISIONS.md, D-005 и D-009; docs/gui/GUI_DESIGN.md, GUI-D14; docs/MATH_SPEC.md, 2.5
status: draft
---
Ось y направлена вдоль стороны b; пластина занимает 0 ≤ y ≤ b. Отношение сторон b/a решатель проверяет перед
расчётом: допустимо 0,2 ≤ b/a ≤ 5. Решение Навье, с которым сравнивается расчёт SSSS, применимо в той же области.
