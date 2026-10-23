# R5-V CCCC — интеграция Literature (модели округления) и исследовательского B-C перед Q3

**Роль:** Integrator. **Цели:** G2, G3.4, G4. **Дата:** 2026-10-23.
**Ветка:** `integration/r5-cccc-q3-preparation` от `integration/r5-cccc-halfstep-controls` @ `0772bb888069831e94a07de198d5af06cd7af0bf`
(PR #28, Q1b).
**Вердикт:** **INTEGRATION_PASS** — техническая интеграция. Научных решений, порогов и разрешений не вводит.

## 1. Основа

- Задание называет основой `0772bb8`. К моменту интеграции PR #28 уже слит: `origin/main` = `8e269e159c8caec1bb4ff61f46e107d8bc38c506`.
  Это merge `d315e24` + `0772bb8`, его дерево **совпадает** с деревом `0772bb8`.
- Ветка создана от `0772bb8`, как указано в задании. Её diff относительно обновлённого `main` совпадает с diff относительно `0772bb8`:
  17 файлов, только добавления. Merge в `main` проходит без конфликтов.

## 2. История объединения

Обычные `git merge --no-ff`, без rebase, squash и cherry-pick. **Конфликтов нет.** Промежуточные ветки повторно не вливались.

| № | Merge-коммит | Первый родитель | Второй родитель |
|---|---|---|---|
| 1 | `a27d276de40145e6da43a9ba338facea6899511d` | `0772bb8` | Literature `96914c95152eb962aba0171576c3e067e7090d6f` |
| 2 | `50b5a775bbb484f7b28d9253d5cc2661861ab962` | `a27d276` | B-C `5bcc09d0c3cf1e4ec278e104c83c8c9e398f24ad` |

Затем — коммит Integrator с этим отчётом.

**Происхождение.**

Literature (`d315e24..96914c9`):

| Коммит | Содержание |
|---|---|
| `8b7570b` | модели ВОК-1…ВОК-14 (ОВ-S4, BC-14) |
| `f4a4cb9` | рецензия Reviewer — CHANGES_REQUIRED, З-1…З-4 |
| `fe5d60f` | исправления З-1…З-4 |
| `f62cd79` | крайний случай ВОК-12 и формулировка правила бюджета |
| `abeb4dc` | заключительная проверка — ADDRESS_PASS |
| `96914c9` | редакционные уточнения Н-1 (методика Е-2) и Н-2 (ε_fS) |

B-C (`9d2e487..5bcc09d`, основан на QA-коммите Q1b `9d2e487`, который входит в `0772bb8`):

| Коммит | Содержание |
|---|---|
| `1d33fe1` | исследовательский QA-прототип B-C |
| `9f0d5dc` | рецензия — CHANGES_REQUIRED |
| `f02e6a8` | исправления ПБ-1…ПБ-5, Р-1, Р-2 |
| `5bcc09d` | заключительная проверка — ADDRESS_PASS |

## 3. Состав (17 новых файлов; изменённых и удалённых нет)

| Группа | Файлы | Источник |
|---|---|---|
| Literature | `R5_CCCC_ROUNDOFF_PREQUAL_DRAFT.md`, `R5_CCCC_ROUNDOFF_PREQUAL.independent-review.md`, `R5_CCCC_ROUNDOFF.address-fix.md`, `R5_CCCC_ROUNDOFF.final-address.md` | `96914c9` |
| B-C, код | `src/kkvim_reference/r5_cccc_bc.py`, `tests/reference/test_r5_cccc_bc.py` (99), `scripts/r5_cccc_bc_prototype_observations.py` | `5bcc09d` |
| B-C, данные | `docs/results/r5_cccc_bc/{chain.csv, ill_conditioned.csv, meta.json}` (редакция `f02e6a8`); `history_1d33fe1/` (три файла) | `5bcc09d` |
| B-C, отчёты | `R5_CCCC_BC_QA_PROTOTYPE.{report, independent-review, address-fix, final-address}.md` | `5bcc09d` |

Все 17 файлов **побайтно** совпадают с источниками.

## 4. Проверки

| № | Проверка | Результат |
|---|---|---|
| 1 | Непредусмотренные изменения | нет. `src/kkvim`, `tests/kkvim`, `pyproject.toml`, `uv.lock`, `.github`, `.githooks`, `MATH_SPEC.md`, `DECISIONS.md`, `STATUS.md`, `ROADMAP.md`, `OPEN_QUESTIONS.md`, численные спецификации, план QA — 0 строк diff |
| 2 | Совместимость с Q1b | Файлы Q1b (модуль, тесты, данные, скрипт) не изменены. B-C использует `kkvim_reference.r5_cccc_halfstep` в редакции `9d2e487`, она же в HEAD. Literature — только документы |
| 3 | Тесты Q1b | **91 passed** |
| 4 | Тесты B-C | **99 passed** |
| 5 | Полный pytest (Windows 10, CPython 3.14.8) | **3319 passed, 31 skipped**, 0 failed — равен ориентиру (3220 + 99) |
| 6 | `uv lock --check`, `git diff --check` | без замечаний |
| 7 | Независимость от рабочего `kkvim` | Импортов `kkvim` в модуле B-C, скрипте и тестах нет. Используются `math`, `dataclasses`, `fractions`, `numpy`, `scipy` и `kkvim_reference.r5_cccc_halfstep` |
| 8 | Воспроизводимость CSV | Перезапуск `r5_cccc_bc_prototype_observations.py --out <scratch>` (8,6 с): `chain.csv` (42 строки) и `ill_conditioned.csv` (8 строк) совпадают во всех столбцах, кроме времени `seconds`. Перезапуск `r5_cccc_halfstep_observations.py`: `ill_conditioned.csv` — побайтно; `chain.csv` (36 строк) — все столбцы, кроме `seconds`. В `meta.json` различаются только `python`, `seconds`, `git_commit` |
| 9 | Исторические данные | `docs/results/r5_cccc_bc/history_1d33fe1/*` побайтно равны файлам `1d33fe1`; текущие — файлам `f02e6a8`. Исторические редакции Q1b (`history_b72515b/`, `history_aed5ba4/`) не изменены |
| 10 | Новые численные пороги и научные решения | Нет. Literature не назначает численных значений tol_solve, η_max^C, ε_floor^C, K, cond(S) и множителя энергетического теста; c_R = 16 и другие — кандидаты. B-C: «собственные пороги не вводились; все числа — OBSERVATION»; запуски с QR — RESEARCH_ONLY. Решений D-051 и далее нет |
| 11 | CI | Integrator не наблюдает: авторизованного `gh` нет (AGENTS.md, разд. 6). Подтверждение — владелец |

## 5. Статусы (сохранены)

- **Q1b** — независимые QA-контроли R-H, R-AC, R-HP, принятые аудитом (`aa568aa`, AUDIT_PASS_FOR_INTEGRATION).
- **Literature** — проверенная математическая основа подготовки Q3 (ADDRESS_PASS `abeb4dc`, уточнения Н-1, Н-2). Численные пороги
  не утверждены.
- **B-C** — исследовательский QA-инструмент в `kkvim_reference`, не рабочий решатель (ADDRESS_PASS `5bcc09d`).
- **N ≥ 2** — RESEARCH_ONLY до решения о численном ранге.
- **Энергетическая модель округления и универсальные численные допуски** — не квалифицированы.
- **Программирование CCCC в `kkvim`** не разрешено.
- **R4 и R5** — IN_PROGRESS.

## 6. Наблюдения (не блокируют)

- `R5_CCCC_ROUNDOFF_PREQUAL_DRAFT.md`, строка 5: «Статус документа: DRAFT — для независимой проверки Reviewer». Проверка уже выполнена
  (`f4a4cb9`, `abeb4dc` — ADDRESS_PASS). Шапку стоит обновить при следующей правке Literature; Integrator её не менял.
- Отметка в `STATUS.md` не добавлялась: задание её не предусматривает.

## 7. PR

PR #28 уже слит. По заданию PR в `main` открывается после слияния #28 и повторной проверки diff относительно обновлённого `main`.
Проверка выполнена (разд. 1): diff совпадает, конфликтов нет. Ссылка — в сообщении о сдаче. Merge в `main` не выполнялся.
