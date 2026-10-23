# R5-V CCCC — интеграция независимого инструмента Е-2 (ошибка образования поля W = U Vᵀ)

**Роль:** Integrator. **Цели:** G3.4, G4. **Дата:** 2026-10-24.
**Ветка:** `integration/r5-cccc-e2-instrument` от `main` @ `7645fa7bec2d3c5a01df77806a2f5c2387b45c20` (PR #29).
**Вердикт:** **INTEGRATION_PASS** — техническая интеграция исследовательского инструмента.

Это не квалификация Q3. Нет новых порогов CCCC, нормативных допусков, решений D-051 и далее, разрешения рабочего решателя.
Q3 не запускался.

## 1. Условие интеграции

| Объект | SHA | Статус |
|---|---|---|
| Источник QA `qa/r5-cccc-e2-ie3a-address` | `37bdf89a998e1d95291507ec0dc0bb01b8ad8140` | исправления ИЕ-3а, Н-а…Н-в |
| Финальная ветка Reviewer `review/r5-cccc-e2-ie3a-final` | `b1278fc489a1560ff337d55ec9ed6abfe6749683` | **ADDRESS_PASS** (`docs/R5_CCCC_E2_INSTRUMENT.ie3a-final.md`) |
| CI на `b1278fc` | — | SUCCESS — по сообщению владельца; Integrator CI не наблюдает (разд. 3, п. 12) |

`b1278fc` — прямой потомок `37bdf89`. Сверх QA он добавляет только документ рецензии `R5_CCCC_E2_INSTRUMENT.ie3a-final.md`.

## 2. История и объединение

| Коммит | Роль | Содержание |
|---|---|---|
| `3aaa80e` | QA | инструмент Е-2 (`qa/r5-cccc-e2-field-roundoff-prep`); основа — B-C `5bcc09d`, входит в `main` |
| `907de22` | Reviewer | независимая рецензия — ИЕ-1…ИЕ-3 (`review/r5-cccc-e2-instrument`) |
| `c7b8170` | QA | исправления ИЕ-1…ИЕ-3, Н-1, Н-2, Н-4 (`qa/r5-cccc-e2-instrument-address`) |
| `f5f85e1` | Reviewer | адресная проверка — CHANGES_REQUIRED по ИЕ-3а (`review/r5-cccc-e2-final-address`) |
| `37bdf89` | QA | исправления ИЕ-3а, Н-а…Н-в (`qa/r5-cccc-e2-ie3a-address`) |
| `b1278fc` | Reviewer | заключительная сверка — **ADDRESS_PASS** (`review/r5-cccc-e2-ie3a-final`) |

- **Merge `--no-ff`:** `7474ffcd7f3d7075a15c9dba0f3aa8bca7c3a5cb`, родители `7645fa7` и `b1278fc`.
- **Конфликтов нет:** `git ls-files -u` пуст.
- **Промежуточные ветки повторно не вливались.** Все пять — предки `b1278fc`.
- Rebase, squash и cherry-pick не применялись.

## 3. Проверки

| № | Проверка | Результат |
|---|---|---|
| 1 | Состав | 7 новых файлов; изменённых и удалённых нет (разд. 4) |
| 2 | Побайтная сохранность | Все 7 файлов равны `b1278fc`. Шесть из них равны и `37bdf89` (QA); седьмой — документ Reviewer |
| 3 | Независимость от рабочего `kkvim` | Импортов `kkvim` в инструменте и тестах нет. Используются стандартная библиотека, `numpy` и `kkvim_reference.r5_cccc_{d2, halfstep}`; тесты дополнительно используют `r5_cccc_bc`. `threadpoolctl` необязателен и нужен только для записи происхождения. Тест `test_instrument_does_not_import_the_main_solver` проверяет `sys.modules` в отдельном процессе. `tests/test_package_independence.py` — 14 passed |
| 4 | Неизменность Q1b, B-C, решений и спецификаций | `git diff 7645fa7 HEAD` даёт 0 строк по файлам: `src/kkvim`, `tests/kkvim`, `scripts`, `docs/results`; модули и тесты `r5_cccc_halfstep`, `r5_cccc_bc`, `r5_cccc_d2`; `MATH_SPEC.md`; `DECISIONS.md` (D-049, D-050); `ROADMAP.md`; `OPEN_QUESTIONS.md`; `*NUMERICAL_SPEC*`; `*PREQUAL*`; `pyproject.toml`; `uv.lock`; `.github`; `.githooks`. `STATUS.md` изменён только технической отметкой Integrator (разд. 5) |
| 5 | Совместимость с Q1b и B-C | Е-2 построен на `5bcc09d`. Модули и тесты Q1b, B-C и `r5_cccc_d2` в HEAD побайтно равны этой редакции. Методика — `R5_CCCC_ROUNDOFF_PREQUAL_DRAFT.md` @ `96914c9`; в `main` файл побайтно равен этой редакции. Все пути, упомянутые в файлах Е-2, существуют в HEAD |
| 6 | Тесты Е-2 | **100 passed, 4 skipped** — как в отчёте Reviewer `b1278fc`. Пропуски — только случаи с расширенным `long double`; на этой платформе `numpy.longdouble` совпадает с binary64 (nmant = 52) |
| 7 | Тесты Q1b | `test_r5_cccc_halfstep.py` — **91 passed** |
| 8 | Тесты B-C | `test_r5_cccc_bc.py` — **99 passed** |
| 9 | Полный pytest (Windows 10, CPython 3.14.8) | **3419 passed, 35 skipped**, 0 failed (107 с); равен ориентиру Reviewer и сумме 3319 + 100 passed, 31 + 4 skipped. Новые пропуски — только 4 случая Е-2 с `long double`. Прежние 31: `mpmath` не установлен, тяжёлые прогоны выключены флагом, `long double` в R3 |
| 10 | Воспроизводимость тестов | Два независимых прогона `test_r5_cccc_e2.py` дали одинаковые исходы для всех 104 случаев (`-rA`, без кэша pytest). Е-2 не пишет файлов данных, поэтому сверять CSV не нужно. CSV Q1b и B-C с историческими редакциями `history_b72515b/`, `history_aed5ba4/`, `history_1d33fe1/` не изменены |
| 11 | `uv lock --check`, `git diff --check` | без замечаний |
| 12 | CI | Integrator не наблюдает: авторизованного `gh` нет (AGENTS.md, разд. 6). CI `b1278fc` — SUCCESS по сообщению владельца. CI интеграционной ветки подтверждает владелец |

## 4. Состав

| Группа | Файлы | Источник |
|---|---|---|
| Инструмент | `src/kkvim_reference/r5_cccc_e2.py` | QA `37bdf89` |
| Тесты | `tests/reference/test_r5_cccc_e2.py` (104 случая) | QA `37bdf89` |
| Отчёты QA | `R5_CCCC_E2_INSTRUMENT_PREP.report.md`, `R5_CCCC_E2_INSTRUMENT.address-fix.md` | QA `37bdf89` |
| Рецензии | `R5_CCCC_E2_INSTRUMENT.independent-review.md` (`907de22`), `.final-address.md` (`f5f85e1`), `.ie3a-final.md` (`b1278fc`) | Reviewer |

От Integrator: этот отчёт и техническая отметка в `docs/STATUS.md`.

## 5. Статусы (сохранены)

- **Е-2** — исследовательский QA-инструмент в `kkvim_reference`, не рабочий решатель и не квалификация Q3.
- **Область Е-2.** Точная арифметика (`fractions`) измеряет только ошибку образования поля ŵ = U Vᵀ при общих U, V. По отчёту QA,
  разд. «Рамки», она не смешивается с ошибкой решения полушага (ii) и смены базиса (iii). Полное поле W с ошибкой решения полушага
  инструмент **не сертифицирует**.
- **Н-3** (штатный путь вычисления ‖·‖_a) и **c_R** открыты для Q3. Строгая граница ошибки вычисления ‖·‖_a в double — UNRESOLVED.
- **Численный пол ε_floor^C, K_E и другие пороги CCCC** не назначены и не утверждены; нормативного допуска нет.
- Числа на тестовых данных — OBSERVATION.
- Эталон повышенной точности для больших сеток (ОК-7) не реализован — UNRESOLVED.
- **Q1b, B-C, Literature (ВОК-1…ВОК-14)** — статусы прежние.
- **N ≥ 2** — RESEARCH_ONLY.
- Q3 не запускался.
- Программирование CCCC в `kkvim` не разрешено.
- **R4 и R5** — IN_PROGRESS.
- Решений D-051 и далее нет; `OPEN_QUESTIONS.md` не менялся.

## 6. PR

PR и merge в `main` выполняются отдельно, по разрешению владельца. Diff ветки относительно `main`: 8 новых файлов (7 из источника
и этот отчёт) и отметка в `STATUS.md`.
