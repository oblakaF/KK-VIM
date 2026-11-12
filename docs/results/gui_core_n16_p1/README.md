# Независимая QA GUI P1: маршруты SSSS/CCCC N = 1–6

Цели G3.3, G3.4, G4, G5; этап R5-V / Q4; инженерный пакет GUI P1. Это пакет доказательств независимой QA интеграции GUI с решателями SSSS и CCCC при N = 1–6 (коммит Builder `d48c10e`). **Это приёмка GUI, а не научная.** Научные статусы решателей не меняются:
- R3 при CONVERGED — принятый решатель (D-043);
- R4 — PROVISIONAL / NOT_QUALIFIED;
- SSSS N = 4–6 и CCCC N = 1–6 — RESEARCH_ONLY / PROVISIONAL / NOT_QUALIFIED;
- П-4 — ASSUMED; R5 и Q4 — IN_PROGRESS; `D1_START = BLOCKED`.

Пакет P1 не совпадает с формальными этапами прототипа из `docs/gui/GUI_DESIGN.md` и не закрывает их.

## Вердикт и происхождение

- **Задача QA:** `GUI-CORE-N16-P1-QA`. **Вердикт:** **`QA_PASS_GUI_CORE_N16_P1_LIMITED`**, блокирующих замечаний нет. Полный отчёт — [QA_REPORT.md](QA_REPORT.md).
- **Проверенная реализация:** коммит `d48c10e713d4ec4700331782bd583bd63b6f75da` (ветка `builder/gui-core-n16-p1`), единственный родитель — `306e85ace3008deac869663245591b36880185e7` (`main`, merge PR #71).
- **Окружение QA** — [ENVIRONMENT.txt](ENVIRONMENT.txt): Windows 10, Python 3.14.8, NumPy 2.5.4, SciPy 1.18.1, PySide6-Essentials 6.12.0, pyqtgraph 0.14.0, Qt offscreen.
- **Исторические контрольные данные GUI** — `1d6af27e3a4e3434080e074d0d21c848f2261362:docs/results/gui_backend`, 50 файлов. Их blob-id — в [results/controls_blobs.txt](results/controls_blobs.txt); Integrator сверил их с `git ls-tree`, все совпали.
  - Коммит `1d6af27` — голова удалённой ветки `qa/gui-backend-acceptance`. В `main` эти данные не входят.
  - Для повторения 29 контрольных тестов эта ветка должна быть доступна.
- **Перенос в репозиторий:** задача Integrator `GUI-CORE-N16-P1-INTEGRATION-PREP`. Вердикт и вычисленные значения не менялись.

## Почему в `gui_check.json` записано `verdict_checks = FAIL`

`results/check/gui_check.json` — машинный итог независимых функциональных проверок QA (`scripts/gui_check.py`). Он содержит **`"verdict_checks": "FAIL"`**. Значение оставлено как есть. Это не противоречит ограниченному PASS QA: из 386 проверок по случаям не выполнены ровно три, и каждая объяснена наблюдением QA (QA_REPORT, разд. 6).

| Проверка с FAIL | Наблюдение | Что именно расходится |
|---|---|---|
| `S3_R4_N2_research: field_csv_data_bitwise_vs_cli` | Н1 | `w [m]`, координаты и индексы побитово равны CLI. `w_bar [1]` расходится не более чем на 1 ulp в 62 из 289 узлов |
| `S4_R4_N3_fixed4: field_csv_data_bitwise_vs_cli` | Н1 | то же, 54 из 289 узлов |
| `S6_SSSS_N4_research: history_common_columns_text_equal` | Н2 | столбец `w_max` истории: в GUI безразмерный, в CLI в метрах |

**Как это понимать:**
- **Н1.** Для R4 **`w_bar` GUI и CLI не равны побитово**, и этот пакет так их и не называет.
  - GUI считает `w·D/(q·a⁴)`, CLI делит на `w_scale` — отсюда разница не более 1 ulp.
  - Формула GUI существует ещё в базе `306e85a` и в diff P1 не входит.
  - Поле прогиба `w` в метрах совпадает побитово.
- **Н2.** В GUI столбец `w_max` истории SSSS N = 4–6 называется «max|w̄| (ячейка)», имеет единицу `[1]` и в манифесте `unit = "1"`. Его значения равны столбцу CLI `w_bar_max`.
  - Умноженные на `w_scale = q a⁴/D` = 0,052 м, они побитово равны столбцу CLI `w_max` в метрах.
  - Подписи и единицы указаны верно; различается только представление при одинаковом имени ключа `w_max`.
  - Числовых расхождений поля нет.

Других невыполненных проверок нет. Отдельный прогон Qt (`results/qt/qt_smoke.json`) дал `PASS` без замечаний.

**Н3 (в `gui_check.json` не FAIL — это наблюдение по тексту).** Нормативный режим R4 (D-047) всегда завершается при s = 1 статусом NOT_QUALIFIED; GUI относит этот конец к роли `research` (`docs/gui/KKVIM_BACKEND.md`). Однако:
- манифест записывает `half_steps = {completed: 1, requested: 100}`;
- отчёт пишет «s = 1 из запрошенных 100».

Здесь 100 — бюджет полушагов, а не задание выполнить ровно 100 полушагов. Наблюдение не блокирует; это кандидат на будущую адресную правку текста и манифеста.

Все утверждения этого раздела воспроизводятся по файлам пакета скриптом Integrator `scripts/integrator/verify_observations.py`; результат — `results/integrator/verify_observations.json`. Его `PASS` означает целостность пакета и полное совпадение FAIL QA с наблюдениями Н1 и Н2. Он не превращает `verdict_checks` в PASS.

## Содержимое

| Путь | Что это |
|---|---|
| `QA_REPORT.md` | итоговый отчёт QA: ревизии, diff, тесты, функциональные проверки, Qt, наблюдения Н1–Н6, ограничения |
| `controls.json` | компактный машинный итог QA |
| `COMMANDS.txt` | точные команды воспроизведения QA |
| `ENVIRONMENT.txt` | версии ПО и состояние worktree QA |
| `scripts/gui_check.py`, `run_gui_check.py` | независимые функциональные проверки: путь GUI в дочернем процессе против CLI и прямого вызова библиотеки |
| `scripts/qt_smoke.py`, `qt_visibility.py` | настоящий Qt (offscreen): форма → процесс → анализ → экспорт |
| `scripts/gen_schema1.py` | манифесты схемы 1 из исходников базы `306e85a`, для проверки обратного чтения |
| `scripts/integrator/verify_observations.py` | проверка этого пакета Integrator |
| `results/check/gui_check.json`, `results/gui_check.log.txt` | итог функциональных проверок, `verdict_checks = FAIL` (см. выше) |
| `results/check/cli/…`, `results/check/export/…` | файлы GUI и CLI для объяснения Н1–Н3: поля S3 и S4, история, `result.json` и манифест S6, манифест и отчёт S5 |
| `results/qt/qt_smoke.json`, `qt_visibility.json`, `results/qt_smoke.log.txt` | итоги Qt: `verdict = PASS` |
| `results/smoke/smoke.json` | штатный `--smoke-test`: CONVERGED, 80×40, код 0 |
| `results/schema1/schema1_summary.json` | обратное чтение манифестов схемы 1 |
| `results/controls_blobs.txt` | blob-id исторических контролей `1d6af27` |
| `results/pytest_*.log.txt` | журналы pytest: 34 passed; 506 passed, 12 skipped; 535 passed, 11 skipped; 193 passed |
| `results/static_checks.log.txt` | ruff, `git diff --check`, `uv lock --check`, самотест hook |
| `results/integrator/verify_observations.json` | результат проверки Integrator |
| `MANIFEST.qa_original.sha256` | исходный манифест QA: 200 файлов, хэши исходных байтов |
| `REPACKAGING.csv` | соответствие файлов QA файлам здесь: исходный и итоговый SHA256, преобразования |
| `MANIFEST.sha256` | SHA256 файлов этого каталога в том виде, в каком они записаны в Git |

## Воспроизведение

**Проверка пакета.** Только по его файлам, без GUI и решателя, из корня клона:

```sh
python -I -X utf8 docs/results/gui_core_n16_p1/scripts/integrator/verify_observations.py
cd docs/results/gui_core_n16_p1 && sha256sum -c MANIFEST.sha256
```

`sha256sum -c` нужно запускать на рабочей копии с LF: Linux или `core.autocrlf=false`. Можно также проверять по объектам Git: `git show <коммит>:docs/results/gui_core_n16_p1/<файл> | sha256sum`. Скрипт Integrator от окончаний строк не зависит, если файлы читаются как текст. Его вывод детерминирован и при повторном запуске совпадает побитово.

**Полное повторение QA** — по [COMMANDS.txt](COMMANDS.txt):
- нужен клон с `d48c10e`, `306e85a` и `1d6af27`;
- скрипты QA копируются в `scratch/gui_n16/` клона: их пути отсчитываются от собственного каталога (`ROOT = HERE.parents[1]`), абсолютных путей в них нет;
- побитовое сравнение с прогонами QA D-068 использует `docs/results/r5_cccc_n46_s2_d068` и, если есть, файлы прогонов `scratch/d068_final/runs`; без них эти пункты записываются как примечания, а не как PASS;
- 11 пропусков побитового сравнения истории оправданы: контроли `1d6af27` созданы на Python 3.13.15.

## Перенос в репозиторий: что изменено относительно файлов QA

Для каждого файла в `REPACKAGING.csv` указан один или несколько из трёх видов преобразований:
1. `eol` — окончания строк CRLF заменены на LF.
2. `path` — абсолютный путь worktree QA заменён меткой `<QA_WORKTREE>`. Это `QA_REPORT.md` (строка 16), строка о пропуске в `pytest_gui_nocontrols.log.txt` и команда и выходной каталог в CLI `result.json` случая S6.
3. `name` — `*.log` переименованы в `*.log.txt` (`*.log` исключены `.gitignore`); `MANIFEST.sha256` QA сохранён как `MANIFEST.qa_original.sha256`.

Других изменений нет. Исходные SHA256 относятся к байтам файлов QA до переноса. Отсюда:
- у `QA_REPORT.md` здесь другой SHA256, чем `7c7c9ce1…` у QA: изменена только строка 16;
- у `gui_check.json` и `qt_smoke.json` хэши отличаются только из-за `eol`;
- `controls.json` и `scripts/gui_check.py` перенесены без изменений.

Integrator добавил `README.md`, `REPACKAGING.csv`, `MANIFEST.sha256` и `scripts/integrator/` с результатом `results/integrator/`.

## Что не включено

168 из 200 файлов QA остались локально в `<QA_WORKTREE>/scratch/gui_n16/package/`. Их SHA256 записаны в `MANIFEST.qa_original.sha256`. Это:
- 13 снимков экрана: 10 Qt offscreen и 3 `--smoke-test` (на них квадраты вместо текста);
- 36 файлов пакетов экспорта Qt и 6 файлов пакета `--smoke-test`;
- 101 файл полных выходов CLI и экспорта для 13 случаев и демо-режима, кроме перечисленных выше файлов Н1–Н3;
- 12 файлов манифестов и CSV схемы 1.

Их итоги есть в `gui_check.json`, `qt_smoke.json` и `smoke.json`. Повторное создание — по `COMMANDS.txt`.

## Ограничения и неблокирующие наблюдения

- **Н1–Н3** — см. выше. Н3 — кандидат на будущую адресную правку текста и манифеста.
- **Н4. Консервативный отказ RN-5** на прямоугольнике N = 6, a = 5, b = 1, 32×13 (уже описан в D-068 и `docs/results/r5_cccc_n46_s2_d068`). GUI лишь передаёт решение решателя: диагностическое поле s = 1, `field_diagnostic.csv`.
- **Н5. Ruff:** новых нарушений нет. В базе было 2 F401, после P1 — 1; оно было и в базе.
- **Н6. Прерванный рендеринг не проверялся.** При сбое рисования рисунков может остаться неполный пакет в новой папке экспорта.
- **Только offscreen.** Qt проверен на Windows offscreen, без межплатформенной визуальной кампании.
- **Ограничения независимости:** тот же тип исполнителя, одна платформа. Эталоны — CLI и библиотека, то есть тот же код решателей. Поэтому проверена передача результатов через GUI, а не правильность решателей и не континуальная модель.
- **Повторное использование приватных helpers CLI** (`cli_ssss._field_arrays`, `_Run`, `cli_cccc_n23._certificate_columns`) — ограничение сопровождаемости, а не корректности.
