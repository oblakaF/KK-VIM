# R5-V Q4: устранение Н-1 и Н-2 второго пути B_E — отчёт QA

- **Роль:** QA. **Цели:** G3.4, G4. **Этап:** R5-V Q4, подготовка D1.
- **База:** `main` = `563547e9a075369ad54a461bca89f9b7cea8dff1` (PR #51 слит). Ветка `qa/r5-q4-be-hardening`, отдельный worktree
  на D:; локальный коммит, без push.
- **Задание:** исправить только два подтверждённых дефекта рецензии `docs/R5_CCCC_Q4_BE_SECOND_PATH.independent-review.md`
  (далее RV, разд. 7): **Н-1** (OverflowError при нарушенной У-1) и **Н-2** (проба FPU только до расчёта).
- **Статус:** исследовательский инструмент QA (D-049, D-053, D-054, D-055). Это не кампания D1, не квалификация и не SP-BE (OD-11).
  Формулы B_E, правило корней, теорема Э-2, пороги и протокол D1 не изменены. **D1_START = BLOCKED. R5 = IN_PROGRESS.**
- **Вердикт:** разд. 7.

Обозначения: BE2 — `scripts/r5_cccc_q4_be_second_path_research.py`; ST — `docs/R5_CCCC_Q4_ENERGY_ERROR_BOUND.study.md`;
QA-отчёт PR #51 — `docs/R5_CCCC_Q4_BE_SECOND_PATH_RESEARCH.qa-report.md`.

## 1. Состав изменений

| Файл | Изменение |
|---|---|
| `scripts/r5_cccc_q4_be_second_path_research.py` | Н-1: `log2_ceil`, `ratio_up`; `formation()` возвращает `max_ratio` направленно и `max_ratio_log2_up`. Н-2: `_probe_text`, `_probe_premises`; `second_path()` повторяет `probes()` после расчёта; `out["probes"] = {"before", "after", "same_thread"}` |
| `tests/reference/test_r5_cccc_q4_be_second_path_research.py` | три новых теста (разд. 4); `import json` |
| этот отчёт | — |

Другие файлы не изменены. Инструмент мутаций `scripts/r5_cccc_q4_be_mutation_research.py` и его тесты не тронуты: ни один
мутируемый фрагмент текста не пересекается с изменёнными строками (проверено `grep` по каталогу `MUTATIONS`).

## 2. Н-1 — OverflowError при нарушенной У-1

**Дефект (RV, Н-1).** В `formation()` диагностика `"max_ratio": float(worst)` для точного рационального отношения
max |ŵ − XYᵀ| / (γ_N |X||Y|ᵀ) вызывала необработанный `OverflowError`, когда отношение превышало DBL_MAX. Контрпример Reviewer:
N = 1, X = 2⁻⁶⁰⁰·𝟏 (5 × 1), Y = 2⁻⁶⁰⁰·𝟏 (4 × 1), ŵ_s = 𝟏, сетка 6 × 5. Воспроизведено на базе `563547e`: `formation` и
`second_path` завершались `OverflowError: integer division result too large for a float`.

**Исправление.** Сама проверка У-1 не изменена: сравнение `err > gN * s` остаётся точным в `Fraction`, счётчик нарушений и
`holds` вычисляются как прежде. Изменено только представление диагностического значения:

- `max_ratio = ratio_up(worst)` — собственное сертифицированное направленное округление вверх (`round_up`, уже имевшееся в BE2:
  ближайший binary64 сверху, проверенный по соседнему double). Если отношение превышает DBL_MAX, `round_up` даёт +∞, а `ratio_up`
  возвращает `None`, чтобы JSON оставался строгим (без `Infinity`; `json.dumps(..., allow_nan=False)` проходит);
- `max_ratio_log2_up = log2_ceil(worst)` — точное целое ⌈log₂ worst⌉ по длинам битов числителя и знаменателя (None для 0).
  Целое не переполняется ни в binary64, ни в JSON, поэтому величина нарушения сохраняется и при `max_ratio = None`.

Для контрпримера: `holds = False`, `violations = 20`, `max_ratio = None`, `max_ratio_log2_up = 1253`
(точно: (1 − 2⁻¹²⁰⁰)(2⁵³ − 1)·2¹²⁰⁰ ∈ (2¹²⁵², 2¹²⁵³)); `second_path` → `UNRESOLVED` с причиной `premise U-1`, без исключения.

**Отличие от прежнего значения.** `float(worst)` округляло к ближайшему; `ratio_up` — вверх. На сертифицированных строках это
меняет только диагностическое поле `max_ratio` не более чем на один ulp вверх (проверено в разд. 5); `holds`, `violations`,
B_E, E_s, статусы и противоречия не затронуты — поле `max_ratio` ни в один статус и ни в одну формулу не входит.

## 3. Н-2 — проба FPU после расчёта

**Дефект (RV, Н-2).** ПР 4.1, П-в требует проб режима FPU «в том же потоке до и после строки»; BE2 выполняла `probes()` один раз
до расчёта.

**Исправление.** В `second_path()`:

1. до расчёта — `pb = probes()` как прежде; предпосылки P-1 и P-3'(a) ставятся предварительно (`_probe_premises(premises, pb,
   None)`): при отказе первой пробы — `NOT_ESTABLISHED`;
2. после всех вычислений (повторение G-4, точные радиканды, сборка, корни, E_s и части Э-1) — в том же потоке `pa = probes()`;
   `_probe_premises(premises, pb, pa)` задаёт окончательное состояние P-1 и P-3'(a): `ASSUMED`/`ESTABLISHED` только если обе
   пробы OK **и** `pb["thread"] == pa["thread"]`; иначе `NOT_ESTABLISHED` с причиной, называющей отказавшую пробу
   («before the computation OK; after the computation FAILED: …», либо «probes in different threads»);
3. `out["probes"] = {"before": pb, "after": pa, "same_thread": …}` — результаты обеих проб сохраняются. При отказе
   `SecondPathError` до завершения расчёта (область D, нефинитный вход, корень, сборка) строка и так `UNRESOLVED`;
   записывается `{"before": pb, "after": None}` — вторая проба после прерванного расчёта не выполняется, что отмечено
   в причине «after the computation NOT PERFORMED» (состояние `NOT_ESTABLISHED`).

Сертификат `CERTIFIED` недостижим при отказе любой из проб: статус выводится из предпосылок после второй пробы.
Нарушения P-1 и P-3'(a) не скрываются — обе предпосылки получают `NOT_ESTABLISHED` и попадают в `reasons`.

Проба проверяет режим потока в моменты до и после; она не доказывает, что режим не менялся между ними (ограничение самой схемы
проб ПР 4.1, не устранимое пробами). Побитовая привязка И-1 повторения G-4 остаётся вторым каналом.

## 4. Тесты

Новые тесты в `tests/reference/test_r5_cccc_q4_be_second_path_research.py` (ожидания зафиксированы до запуска):

| Тест | Что проверяет |
|---|---|
| `test_n1_violated_u1_beyond_binary64_is_refused_not_raised` | контрпример RV: `formation` без исключения, `max_ratio = None`, `max_ratio_log2_up = 1253` (точное значение выведено вручную), `second_path` → UNRESOLVED с причиной U-1, `json.dumps(allow_nan=False)` проходит |
| `test_formation_ratio_is_a_certified_upper_bound_with_its_exact_magnitude` | конечное отношение: `max_ratio` — наименьший binary64 ≥ точного отношения, пересчитанного независимо для сдвинутого элемента; 2^(k−1) < ratio ≤ 2^k; У-1 выполнена → 0 < `max_ratio` ≤ 1; `log2_ceil` и `ratio_up` на точных значениях, включая DBL_MAX и соседние |
| `test_n2_probes_before_and_after_the_computation` | нормальная строка: CERTIFIED, обе пробы OK, один поток; имитация отказа второй пробы (monkeypatch `probes`, вторая проба возвращает `ok = False`) → UNRESOLVED, `reasons` = {P-1, P-3'(a)} с текстом «after the computation FAILED: …», результат расчёта сохранён, но не сертифицирован; разные потоки → UNRESOLVED; отказ первой пробы → UNRESOLVED, вторая проба всё же выполнена; отказ до расчёта → `{"before": None, "after": None}` |

Все прежние 51 тест двух файлов не изменены.

## 5. Проверки

Платформа: Windows 10 (10.0.19045), x86-64; Python 3.13.15 и 3.14.8 (uv); NumPy 2.5.4. Адресные тесты:

| Проверка | Python 3.13 | Python 3.14 |
|---|---|---|
| `tests/reference/test_r5_cccc_q4_be_second_path_research.py` + `…_be_mutation_research.py` | 54 passed | 54 passed |

1. Контрпример Н-1 → UNRESOLVED без исключения — тест и прямой прогон (разд. 2). **PASS.**
2. Искусственный отказ повторной пробы → UNRESOLVED — тест. **PASS.**
3. Ложного CERTIFIED нет: во всех случаях отказа проб и У-1 статус UNRESOLVED; CERTIFIED требует обеих проб OK в одном потоке. **PASS.**
4. **Результаты B_E не изменились.** Скрипт `scratch/be_hardening/compare_base_fixed.py` (вне git) загружает BE2 из `563547e`
   и исправленный BE2 и прогоняет одни и те же 10 синтетических входов (4 входа фикстуры тестов, 4 контрольных входа инструмента
   мутаций, почти сокращающаяся разность, вход вблизи нормального диапазона): `status`, `assembly`, `assembly_qaeb80`,
   `assembly_lower`, `BE`, `BE_up`, `E_s`, `E_s_le_B_E` (Э-2), `B_E_le_Kstar_R` (Э-3), `components`, `contradictions`, `kc`,
   `radicands`, `subtraction`, `op_checks`, `double_path`, `parts`, `reasons` — **идентичны** на всех 10 входах, на обоих Python;
   `max_ratio` отличается не более чем на один ulp вверх (ожидаемое следствие направленного округления). 9 строк CERTIFIED, 1
   (масштаб 2⁻⁵⁰⁰) UNRESOLVED по P-3'(c), P-2 — как прежде.
5. Существующие адресные тесты — 51 прежний + 3 новых, все PASS.
6. **Способность обнаруживать аналог Г-5 сохранена:** тест `test_every_mutation_reaches_its_declared_verdict` (все 23 мутации на
   фактическом исправленном BE2) и `test_unsafe_primary_mutations_are_classified_unsafe` — PASS; прямой прогон мутаций P-01 и S-01
   (пропуск члена B1) — DETECTED на всех объявленных входах, нулевая мутация N-01 — INAPPLICABLE
   (`scratch/be_hardening/g5_analogue.json`, вне git).
7. `git diff --check` — чисто; `uv lock --check` — PASS; самотест hook — ALL PASSED.

Не запускались (по заданию): полный pytest, опубликованная Q3-кампания, профили `resources`/`q3`, большие пробы.

## 6. Что не входит и остаётся

- **Н-3** (обязательный P̂ для конвейера D1) — не изменялось: задача интеграции конвейера после D-057.
- **Н-4…Н-6** — не изменялись; остаются замечаниями RV.
- Правила SP-BE (OD-11), теорема Э-2, формулы B_E, правило корней, пороги, протокол D1 — не тронуты.
- Ограничения независимости: исправление, тесты и этот отчёт выполнены одним исполнителем QA; рецензия исправления не
  проводилась. Исполнители QA и Reviewer одного типа (ограничение D-053.2 сохраняется).
- Платформа: проверено только на Windows x86-64 (Python 3.13/3.14, NumPy 2.5.4); Linux локально не исполнялся; CI для этой ветки
  не запускался (push запрещён).

## 7. Вердикт

**QA_PASS_BE_HARDENING_N1_N2_LOCAL_READY.**

Н-1 и Н-2 устранены адресно; результаты B_E и проверки Э-2 на синтетических строках не изменились; ложного CERTIFIED нет;
обнаружение аналога Г-5 сохранено. Следующий шаг — рецензия исправления и решение владельца о публикации ветки; D1 не запускалась,
рабочий CCCC не программировался, merge/rebase/push не выполнялись.
