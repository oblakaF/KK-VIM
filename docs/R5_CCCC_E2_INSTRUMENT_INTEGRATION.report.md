# R5-V CCCC — интеграция исследовательского инструмента Е-2 (ошибка формирования поля W = U Vᵀ)

**Роль:** Integrator. **Цели:** G3.4, G4. **Дата:** 2026-10-24.
**Ветка:** `integration/r5-cccc-e2` от `main` @ `7645fa7bec2d3c5a01df77806a2f5c2387b45c20` (PR #29).
**Вердикт:** **INTEGRATION_PASS**.

Это техническая интеграция исследовательского QA-инструмента, а не квалификация Q3. Новых порогов CCCC, решений D-051 и далее,
разрешения рабочего решателя и запуска Q3 нет.

## 1. Источник и объединение

| Объект | SHA |
|---|---|
| `main` | `7645fa7bec2d3c5a01df77806a2f5c2387b45c20` |
| Источник `review/r5-cccc-e2-ie3a-final` | `b1278fc489a1560ff337d55ec9ed6abfe6749683`, основан на B-C `5bcc09d` (входит в `main`) |
| Merge `--no-ff` | `df88881434d2abe203a843e985d497ebc81f97c6`, родители `7645fa7` и `b1278fc`; конфликтов нет |

История источника (`5bcc09d..b1278fc`):

| Коммит | Роль | Содержание |
|---|---|---|
| `3aaa80e` | QA | инструмент Е-2 (`qa/r5-cccc-e2-field-roundoff-prep`) |
| `907de22` | Reviewer | независимая рецензия — ИЕ-1…ИЕ-3 (`review/r5-cccc-e2-instrument`) |
| `c7b8170` | QA | исправления ИЕ-1…ИЕ-3 (`qa/r5-cccc-e2-instrument-address`) |
| `f5f85e1` | Reviewer | адресная проверка — CHANGES_REQUIRED по ИЕ-3а (`review/r5-cccc-e2-final-address`) |
| `37bdf89` | QA | исправления ИЕ-3а, Н-а…Н-в (`qa/r5-cccc-e2-ie3a-address`) |
| `b1278fc` | Reviewer | заключительная сверка — **ADDRESS_PASS** (`review/r5-cccc-e2-ie3a-final`) |

Влита только финальная ветка Reviewer. Промежуточные ветки QA и Reviewer отдельно не вливались: все пять — предки `b1278fc`.
Rebase, squash и cherry-pick не применялись.

## 2. Состав (7 новых файлов; изменённых и удалённых нет)

| Группа | Файлы |
|---|---|
| Инструмент | `src/kkvim_reference/r5_cccc_e2.py` |
| Тесты | `tests/reference/test_r5_cccc_e2.py` (104 случая: 100 passed, 4 skipped) |
| Отчёты | `R5_CCCC_E2_INSTRUMENT_PREP.report.md`, `R5_CCCC_E2_INSTRUMENT.{independent-review, address-fix, final-address, ie3a-final}.md` |

Все 7 файлов **побайтно** совпадают с `b1278fc`. От Integrator — только этот отчёт. Скриптов и файлов данных в `docs/results/` источник
не содержит.

## 3. Проверки

| № | Проверка | Результат |
|---|---|---|
| 1 | Непредусмотренные изменения | нет. `git diff 7645fa7 HEAD` по `src/kkvim`, `tests/kkvim`, `scripts`, `docs/results`, `MATH_SPEC.md`, `DECISIONS.md`, `STATUS.md`, `ROADMAP.md`, `OPEN_QUESTIONS.md`, `*NUMERICAL_SPEC*`, `pyproject.toml`, `uv.lock`, `.github`, `.githooks` — 0 строк |
| 2 | Совместимость с Q1b и B-C | Инструмент импортирует `kkvim_reference.r5_cccc_halfstep` и `kkvim_reference.r5_cccc_d2`, тесты — ещё `r5_cccc_bc`. Эти модули и тесты Q1b и B-C в HEAD побайтно равны редакции `5bcc09d`, на которой построен Е-2. Файлы Q1b и B-C не изменены |
| 3 | Методика | Е-2 ссылается на `R5_CCCC_ROUNDOFF_PREQUAL_DRAFT.md` @ `96914c9` (разд. 0.4, 4.3, Н-1/Н-2). В `main` этот файл побайтно равен `96914c9`. Все пути, упомянутые в файлах Е-2, существуют в HEAD |
| 4 | Тесты Е-2 | **100 passed, 4 skipped**. Пропущены только случаи с расширенным `long double`; на Windows `np.finfo(np.longdouble).nmant = 52`. Совпадает с отчётом Reviewer `b1278fc` |
| 5 | Тесты Q1b, B-C, независимость пакетов | `test_r5_cccc_halfstep.py` (91), `test_r5_cccc_bc.py` (99), `test_package_independence.py` (14) — **204 passed** |
| 6 | Полный pytest (Windows 10, CPython 3.14.8) | **3419 passed, 35 skipped**, 0 failed (110 с) — равен ориентиру Reviewer `b1278fc` и сумме 3319 + 100 passed, 31 + 4 skipped. Новые пропуски — только 4 случая Е-2 с `long double`; остальные 31 — прежние (`mpmath` не установлен, тяжёлые прогоны по флагу, `long double` в R3) |
| 7 | `uv lock --check`, `git diff --check` | без замечаний |
| 8 | Независимость от рабочего `kkvim` | Импортов `kkvim` в инструменте и тестах нет. Используются стандартная библиотека, `numpy`, `kkvim_reference.r5_cccc_{d2, halfstep, bc}`; `threadpoolctl` — необязательно, только для записи происхождения. Тест Е-2 отдельно проверяет отсутствие `kkvim` в `sys.modules` |
| 9 | Исследовательские данные и история | Е-2 не добавляет и не перезаписывает CSV. Данные Q1b (`docs/results/r5_cccc_halfstep/`, вместе с `history_b72515b/` и `history_aed5ba4/`) и B-C (`docs/results/r5_cccc_bc/`, вместе с `history_1d33fe1/`) не изменены — 0 строк diff |
| 10 | Новые пороги и научные решения | Нет. Отчёт QA: «ε_floor^C, K_E и другие пороги не назначены», выборка Q3 не проводилась. Числа — OBSERVATION; строгая граница вычисления ‖·‖_a в double — UNRESOLVED. Решений D-051 и далее нет |
| 11 | CI | Integrator не наблюдает: авторизованного `gh` нет (AGENTS.md, разд. 6). CI `b1278fc` — SUCCESS (по сообщению владельца); CI интеграционной ветки подтверждает владелец |

## 4. Статусы (сохранены)

- **Е-2** — исследовательский QA-инструмент в `kkvim_reference`, не рабочий решатель и не квалификация Q3 (ADDRESS_PASS `b1278fc`).
  Измерение Е-2 на выборке — только в Q3, по отдельному заданию.
- **Н-3** (штатный путь вычисления ‖·‖_a и c_R) — открыт для Q3.
- Строгая граница ошибки вычисления ‖·‖_a в double (компонент (iv)) — **UNRESOLVED** (ВОК-3 — модель, c_R не подсчитан).
- Эталон повышенной точности для больших сеток (ОК-7) не реализован — **UNRESOLVED**.
- **Q1b** — независимые QA-контроли, принятые аудитом; **B-C** — исследовательский QA-инструмент; **Literature** (ВОК-1…ВОК-14) —
  проверенная основа подготовки Q3. Численные пороги не утверждены.
- **N ≥ 2** — RESEARCH_ONLY до решения о численном ранге.
- Численные пороги CCCC (ε_floor^C, K_E и др.) не утверждены; Q3 не запускался.
- Программирование CCCC в `kkvim` не разрешено.
- **R4 и R5** — IN_PROGRESS.
- `OPEN_QUESTIONS.md` и `STATUS.md` не менялись: задание отметки не предусматривает.

## 5. PR

Задание PR не предусматривает. PR и merge в `main` не выполнялись. Diff ветки относительно `main` — 8 добавленных файлов
(7 из источника и этот отчёт).
