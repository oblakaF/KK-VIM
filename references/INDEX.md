# Научная библиотека KK-VIM — каталог

Каталог составлен 2026-10-11 **непосредственно по файлам** каталога `references/`. Источники сведений —
только сами файлы: путь, размер, сигнатура формата (первые байты файла) и SHA256. Индексы, обзоры
и протоколы прежних проектов не использовались ([D-026](../docs/DECISIONS.md)).

Библиотека содержит **106 оригинальных научных публикаций**: 91 PDF, 7 DjVu, 8 DOC.
Расширение каждого файла совпадает с его сигнатурой. Дубликатов по SHA256 нет. Назначены
идентификаторы L-001–L-107; L-099 выведен из каталога (раздел 4.1).

## 1. Правила каталога

- **Иерархия источников** — [`AGENTS.md`](../AGENTS.md), раздел 2: `primary/` — первоисточники КК-ВИМ;
  `additional/` — поздние работы, развитие метода, не замена определения; остальное — справочная
  библиотека. Наличие файла в библиотеке не даёт разрешения менять метод.
- **Имена файлов и папок** — физическое размещение на диске, а не проверенное библиографическое описание
  и не научная классификация. Исключение — папки `primary/` и `additional/`, задающие иерархию;
  соответствие каждого файла `primary/` роли первоисточника алгоритма проверяется в R1 (Q-001).
- **Идентификаторы** `L-001`… постоянны. Новые файлы получают следующий свободный номер (сейчас —
  L-108), номера не переиспользуются, в том числе номера выведенных файлов (раздел 4.1). Метки первоисточников (P80, P81, P81-S, P82) — дополнительные короткие имена.
- **Публикации не изменяются и не переименовываются.** Целостность контролируется по SHA256 (раздел 5).

### Статусы библиографических сведений

| Статус | Значение |
|---|---|
| **НЕ ПРОВЕРЕНО** | Авторы, название, год и выходные данные не сверены с самой публикацией. Имя файла не считается библиографическим описанием |
| **ЧАСТИЧНО** | Сверена только указанная часть; где записан результат — в скобках |
| **ПРОВЕРЕНО** | Выходные данные сверены с титульной страницей публикации; указано, кем, когда и где записано. Пока таких записей нет |

## 2. Сводка по папкам

| Папка (`references/…`) | PDF | DjVu | DOC | Всего |
|---|---:|---:|---:|---:|
| `primary/` | 4 |  |  | 4 |
| `additional/` | 2 |  |  | 2 |
| `Boundary_Conditions/` | 1 |  |  | 1 |
| `FDM/` | 1 |  |  | 1 |
| `Geometric_Nonlinearity_Theory/` | 4 | 1 |  | 5 |
| `Low_Rank_Tensor_Methods/` | 3 |  |  | 3 |
| `Mori_Tanaka_FGM/` | 4 |  |  | 4 |
| `Physical_Nonlinearity_Birger_Ilyushin/` | 12 | 2 |  | 14 |
| `Plate_Benchmarks_3D_Thick/` | 11 |  |  | 11 |
| `Plate_Benchmarks_Linear_BC/` | 5 |  |  | 5 |
| `Rectangular_Plates/` | 11 |  |  | 11 |
| `Refined_Plate_Theories/` | 12 | 1 |  | 13 |
| `Strain_Localization/` | 5 |  |  | 5 |
| `vim/02_Krysko_Group_Later/` | 7 |  | 4 | 11 |
| `vim/03_Dissertations_and_Chapters/` |  |  | 4 | 4 |
| `vim/04_Kantorovich_EKM_Background/` | 3 |  |  | 3 |
| `vim/05_Reviews/` | 1 |  |  | 1 |
| `Оболочки/` | 2 | 2 |  | 4 |
| `Поиск статей/` | 3 | 1 |  | 4 |
| **Итого** | **91** | **7** | **8** | **106** |

Папка `Поиск статей/` — входящие файлы; их отнесение к тематическим разделам не выполнялось.

## 3. Метки первоисточников

Метки P80 и P82 определены в [`docs/DECISIONS.md`](../docs/DECISIONS.md); P81 и P81-S используются в
[`docs/ROADMAP.md`](../docs/ROADMAP.md) и [`docs/OPEN_QUESTIONS.md`](../docs/OPEN_QUESTIONS.md).
Метка — только короткое имя файла. Год в метке взят из имени файла и не проверен.

| Метка | ID | Файл (`references/primary/`) | Библиографические сведения |
|---|---|---|---|
| **P80** | L-001 | `1980_Kirichenko_Krysko_Kantorovich_Vlasov_Nonlinear_BVP_DiffUravn.pdf` | ЧАСТИЧНО: печатные с. 2186–2189 (D-022–D-024) |
| **P81-S** | L-002 | `1981_Kirichenko_Krysko_Reduction_of_Multidimensional_Mechanics_Problems_Saratov.pdf` | НЕ ПРОВЕРЕНО |
| **P81** | L-003 | `1981_Kirichenko_Krysko_Substantiation_VIM_Theory_of_Plates_SovApplMech.pdf` | НЕ ПРОВЕРЕНО |
| **P82** | L-004 | `19XX_Kirichenko_Krysko_Physically_Nonlinear_Plates_Shells_MVI_IzvVUZ.pdf` | ЧАСТИЧНО: печатные с. 78–80 (D-022–D-024) |

Для P80 и P82 по изображениям страниц PDF сверены печатная нумерация и содержание, записанное
в D-022–D-024 и OPEN_QUESTIONS.md. Название журнала, год и том для всех четырёх файлов не проверены.

## 4. Каталог публикаций

Пути — от корня репозитория. Размер — в байтах.

| ID | Метка | Путь | Формат | Размер | SHA256 | Библиографические сведения |
|---|---|---|---|---:|---|---|
| L-001 | P80 | `references/primary/1980_Kirichenko_Krysko_Kantorovich_Vlasov_Nonlinear_BVP_DiffUravn.pdf` | PDF | 427751 | `b6f2773157f97164a77ea8e1ddec76ccad4af97b5495af7f6f289f47dbcd4d91` | ЧАСТИЧНО: печатные с. 2186–2189 (D-022–D-024) |
| L-002 | P81-S | `references/primary/1981_Kirichenko_Krysko_Reduction_of_Multidimensional_Mechanics_Problems_Saratov.pdf` | PDF | 6736424 | `bf7905c431e11f95afb08da2bece1fba138ce07c7d8ad85a0d7f28da9132a913` | НЕ ПРОВЕРЕНО |
| L-003 | P81 | `references/primary/1981_Kirichenko_Krysko_Substantiation_VIM_Theory_of_Plates_SovApplMech.pdf` | PDF | 393134 | `a0cb2351364e01ac02c9d4ad40e34980f97d6b0675507d9d566572f65ebcbc26` | НЕ ПРОВЕРЕНО |
| L-004 | P82 | `references/primary/19XX_Kirichenko_Krysko_Physically_Nonlinear_Plates_Shells_MVI_IzvVUZ.pdf` | PDF | 235709 | `2ef9013483b9ff411216a7686557bdff6289d18ae1a6210bd26f4432ddcb59fa` | ЧАСТИЧНО: печатные с. 78–80 (D-022–D-024) |
| L-005 |  | `references/additional/2021_Awrejcewicz_Krysko_PDE_to_ODE_Review.pdf` | PDF | 3897636 | `9e7dce1a7fecc551747f31c0e7f94e3224f333b189a7e4a2d50391005ae50222` | НЕ ПРОВЕРЕНО |
| L-006 |  | `references/additional/2023_Tebyakin_VIM_Physically_Geometrically_Nonlinear_Kirchhoff_Nanoplates.pdf` | PDF | 15437161 | `3b5919ff5e9e455fc0aeecd7e03f9187184bf78385784ee697ec9b2c95c20253` | НЕ ПРОВЕРЕНО |
| L-007 |  | `references/Boundary_Conditions/2008_Thomas_Bilbao_Nonlinear_Plate_Vibrations_Inplane_BC.pdf` | PDF | 442136 | `1b08bb88a2ae9bdc8fec16ee7b7ce711367808168241e0d6eb2075dfad0b6e0f` | НЕ ПРОВЕРЕНО |
| L-008 |  | `references/FDM/2021_Nguyen_Stable_Accurate_Numerical_Methods_Generalized_Kirchhoff_Love_Plates.pdf` | PDF | 7305447 | `b480d779993997ef9c0060bd2de00b8ee9ad37852d81309d784d2a200bd458a0` | НЕ ПРОВЕРЕНО |
| L-009 |  | `references/Geometric_Nonlinearity_Theory/1972_Volmir_Nonlinear_Dynamics_Plates_Shells_Nauka.djvu` | DjVu | 32232883 | `a946dce5dd3c3616add476d19478b461961824f4c63cf8a02166a4e37cc75d07` | НЕ ПРОВЕРЕНО |
| L-010 |  | `references/Geometric_Nonlinearity_Theory/2011_Bathe_MIT_2.094_Lecture12_Total_Lagrangian_Formulation.pdf` | PDF | 302957 | `80bc3263f86deb73292a24db86ec0a9b1fd68fbb4ecd9d7eca55a8a8a6dec6e1` | НЕ ПРОВЕРЕНО |
| L-011 |  | `references/Geometric_Nonlinearity_Theory/2019_Wu_Pagani_Filippi_Chen_Carrera_Large_Deflection_Postbuckling_Plates_CUF_IJNLM.pdf` | PDF | 1754956 | `d4508f6755a95458840c65672b9ad5fd09d85d1ae07915fce49f3f8c6117fba0` | НЕ ПРОВЕРЕНО |
| L-012 |  | `references/Geometric_Nonlinearity_Theory/2020_Pagani_Daneshkhah_Xu_Carrera_Geometrically_Nonlinear_Terms_Plates_IJNLM.pdf` | PDF | 1494518 | `2107a9d0a714ca0ea1bb39221c2705572c5d30e9926b4a60e40336315784c2c6` | НЕ ПРОВЕРЕНО |
| L-013 |  | `references/Geometric_Nonlinearity_Theory/2023_Ciarlet_Mardare_Intrinsic_von_Karman_Equations_CMAA.pdf` | PDF | 175698 | `d79f07b38739e697bfa80287a2529053d3ba475a96becebebe3755e113d834e6` | НЕ ПРОВЕРЕНО |
| L-014 |  | `references/Low_Rank_Tensor_Methods/2011_Falco_Nouy_Progressive_PGD_Elliptic_Problems_Functional_Eckart_Young_JMAA.pdf` | PDF | 229074 | `c8c289b6a0a10fad9d1b6e41ecfe7ee64d839043f98ecb15dabec0beacd71e16` | НЕ ПРОВЕРЕНО |
| L-015 |  | `references/Low_Rank_Tensor_Methods/2015_Giraldi_Liu_Matthies_Nouy_Nonintrusive_Low_Rank_PGD_Alternating_Minimization_SISC.pdf` | PDF | 515113 | `1cc80a026a84802e482a288640f9c3831a284c5c2d6fe4d18f447e3d7381a8dc` | НЕ ПРОВЕРЕНО |
| L-016 |  | `references/Low_Rank_Tensor_Methods/2023_Minster_Viviano_Liu_Ballard_CP_ALS_with_QR_and_SVD_NLAA.pdf` | PDF | 1702151 | `57746bffd932c8297666ac918a4fccee8c8fd81536eab5240f50269b715064b3` | НЕ ПРОВЕРЕНО |
| L-017 |  | `references/Mori_Tanaka_FGM/2018_Jrad_Nonlinear_Elastoplastic_FG_Shells_MoriTanaka.pdf` | PDF | 4248826 | `9d64092becf33bda8deb70a8fa20e528a36d1cf9489f39db580a482cd18bcffa` | НЕ ПРОВЕРЕНО |
| L-018 |  | `references/Mori_Tanaka_FGM/2019_Zafarmand_Nonlinear_Material_Geometric_Thick_FG_Plates_TTO.pdf` | PDF | 6938354 | `35e19b2893c3521c9ed1e61532c5eb4cce4879ecf40c517438998694f1a1a92b` | НЕ ПРОВЕРЕНО |
| L-019 |  | `references/Mori_Tanaka_FGM/2020_Arslan_Nonlinear_Elastoplastic_FG_Plates_Loading_Unloading.pdf` | PDF | 3634407 | `e83d0899e613508e0e03c8d3c4af8a4159fceb843f0e5a648c960e703797d5a1` | НЕ ПРОВЕРЕНО |
| L-020 |  | `references/Mori_Tanaka_FGM/2022_Krysko_FG_Porous_Euler_Bernoulli_Nanobeams_Nonlinearities.pdf` | PDF | 1140518 | `1eb52df034a9e27217f8319480a6551df7c07d75ea775acb46672b359c07515b` | НЕ ПРОВЕРЕНО |
| L-021 |  | `references/Physical_Nonlinearity_Birger_Ilyushin/1948_Ilyushin_Plasticity_Part1_Elastoplastic_Deformations_Gostekhizdat.djvu` | DjVu | 3403414 | `b0b4047b2e6323102053636cf9df4795b0842c4268a170ae49e8408ace42f461` | НЕ ПРОВЕРЕНО |
| L-022 |  | `references/Physical_Nonlinearity_Birger_Ilyushin/1948_Voce_Stress_Strain_Homogeneous_Deformation_JInstMetals.pdf` | PDF | 6702780 | `d5fee3906aea166f82db00bca23342f9deab5521737cf36bebe5a54c9af7dfb3` | НЕ ПРОВЕРЕНО |
| L-023 |  | `references/Physical_Nonlinearity_Birger_Ilyushin/1951_Birger_General_Methods_Plasticity_Problems_PMM15.pdf` | PDF | 865971 | `758e5b5f0cc9618cb619365b655105f7e39d05ef7dd457b5060ce58d6490a19e` | НЕ ПРОВЕРЕНО |
| L-024 |  | `references/Physical_Nonlinearity_Birger_Ilyushin/1966_Ohashi_Murakami_Elastoplastic_Bending_Clamped_Circular_Plate.pdf` | PDF | 1330463 | `f1087d1c48b6c60664752c17fd93c109ce85202ab2512270c986a4c545770a3f` | НЕ ПРОВЕРЕНО |
| L-025 |  | `references/Physical_Nonlinearity_Birger_Ilyushin/1966_Ohashi_Murakami_Large_Deflection_Elastoplastic_SS_Circular_Plate_JAM.pdf` | PDF | 2303177 | `4bb0ec5d59bfd4110f9948069af137b590c9438f43fbe3e95d0ff8d72a183695` | НЕ ПРОВЕРЕНО |
| L-026 |  | `references/Physical_Nonlinearity_Birger_Ilyushin/1972_Sarkisyan_Method_of_Elastic_Solutions_Theory_of_Plates_IzvANArmSSR.pdf` | PDF | 392318 | `7e7fa448c17639164811951edf74c55998a332f719b5fed5ef221dfdd71b49d4` | НЕ ПРОВЕРЕНО |
| L-027 |  | `references/Physical_Nonlinearity_Birger_Ilyushin/1986_Birger_Mavlyutov_Strength_of_Materials_Nauka.djvu` | DjVu | 25799642 | `581a5c7013eb86c7852bc8d0279fdf2f115d0296a3a456d4cb7e63c58cf61003` | НЕ ПРОВЕРЕНО |
| L-028 |  | `references/Physical_Nonlinearity_Birger_Ilyushin/2010_AbuHalawa_D16T_Nonlinear_Deformation_Approximation_JCompSci.pdf` | PDF | 123356 | `3f1e426773820cfff54ac07a41fb01dfb143651e737b820dcdc79af91ef976e4` | НЕ ПРОВЕРЕНО |
| L-029 |  | `references/Physical_Nonlinearity_Birger_Ilyushin/2017_Awrejcewicz_Krysko_Physically_Geometrically_Nonlinear_Micro-Shells_Thermal.pdf` | PDF | 2920824 | `40cc3cb75d4ec7aeaa26c35a563048485bc02955dcf985f2f25c026e55257a2d` | НЕ ПРОВЕРЕНО |
| L-030 |  | `references/Physical_Nonlinearity_Birger_Ilyushin/2020_Saltykova_Krylova_Krysko_Birger_Variable_Parameters_Flexible_Plates.pdf` | PDF | 827124 | `b7a10f7782c2257d2244f4c4d9509e05fc59a536a68e853f84cba93772332f65` | НЕ ПРОВЕРЕНО |
| L-031 |  | `references/Physical_Nonlinearity_Birger_Ilyushin/2020_Starov_Kalashnikov_Ilyushin_Final_Relation_Plates_Shells_Part2.pdf` | PDF | 2236335 | `44ad3254fc64e923b7520241434c24709ef21726d4ab983f259c598cbf30c944` | НЕ ПРОВЕРЕНО |
| L-032 |  | `references/Physical_Nonlinearity_Birger_Ilyushin/2021_Awrejcewicz_Krysko_Heterogeneous_Shells_Part1_Statics_Dynamics_Birger.pdf` | PDF | 1514293 | `709385e9a99a923f8fa6fd0f76db50a785df872d62dd78c15ed6ea1325690edb` | НЕ ПРОВЕРЕНО |
| L-033 |  | `references/Physical_Nonlinearity_Birger_Ilyushin/2021_Krysko_Awrejcewicz_Physically_Nonlinear_3D_Beams_Plates_Multimodulus_FEM.pdf` | PDF | 7593905 | `235c9f34df2dabd2d41291907777ad8d66a840bea94508534ef5f64ec854af3c` | НЕ ПРОВЕРЕНО |
| L-034 |  | `references/Physical_Nonlinearity_Birger_Ilyushin/2021_Krysko_Awrejcewicz_Physically_Nonlinear_Plates_Rectangular_Cuts_3D_FEM.pdf` | PDF | 1720837 | `c1358784bcd50625092f96e93674de67c54f166542be4657fc2c0e063d572173` | НЕ ПРОВЕРЕНО |
| L-035 |  | `references/Plate_Benchmarks_3D_Thick/1969_Srinivas_Rao_JogaRao_Flexure_SS_Thick_Plates_3D_ZAMM.pdf` | PDF | 701778 | `f1d365411e3879a16069d5b5583b2b90276d03271f3a754081b33e13b7cdc4ba` | НЕ ПРОВЕРЕНО |
| L-036 |  | `references/Plate_Benchmarks_3D_Thick/1970_Pagano_Exact_Solutions_Bidirectional_Composite_Sandwich_Plates_JCM.pdf` | PDF | 546017 | `1bafe92a9da1314b5205ce98d7d19b7de7a2abb4b3def0efe1133d11af9e0123` | НЕ ПРОВЕРЕНО |
| L-037 |  | `references/Plate_Benchmarks_3D_Thick/1970_Srinivas_JogaRao_Rao_Vibration_SS_Thick_Plates_3D_JSV.pdf` | PDF | 905061 | `4a2f0cb468426f77a07d91e5b2589a537759e4ee6395d3d8677461d65ee82bce` | НЕ ПРОВЕРЕНО |
| L-038 |  | `references/Plate_Benchmarks_3D_Thick/1970_Srinivas_Rao_SS_Thick_Orthotropic_Plates_Laminates_3D_IJSS.pdf` | PDF | 1480636 | `36e8d7fca7071bcfffaea16573469604acb7c0538d89de5a3f607b9c0c82ae0c` | НЕ ПРОВЕРЕНО |
| L-039 |  | `references/Plate_Benchmarks_3D_Thick/1972_Pagano_Hatfield_Elastic_Behavior_Multilayered_Bidirectional_Composites_AIAAJ.pdf` | PDF | 296604 | `5c40f3d9bf5a48f591c02b345f68ae3cc7ff968616dfa6c4f122dc8be6c5315a` | НЕ ПРОВЕРЕНО |
| L-040 |  | `references/Plate_Benchmarks_3D_Thick/1974_Iyengar_Chandrashekhara_Sebastian_Thick_Rectangular_Plates_MIF_IngArch.pdf` | PDF | 765853 | `1100a92f985d0361e75168ef5d2ae39a4870bf9a80e19d718134e3ce300bbed3` | НЕ ПРОВЕРЕНО |
| L-041 |  | `references/Plate_Benchmarks_3D_Thick/1985_Levinson_SS_Rectangular_Plate_Exact_3D_Elasticity_JElast.pdf` | PDF | 411598 | `a5b926da7cb30ad5ba14ea0d30d0d36c77c434b282eacb5e5e6f5e5363064762` | НЕ ПРОВЕРЕНО |
| L-042 |  | `references/Plate_Benchmarks_3D_Thick/1998_Liu_Liew_Differential_Cubature_Thick_Plates_IJSS.pdf` | PDF | 1173460 | `cf649ab9fdf5ce45a598456cad17707c8bbfe152a8a56a3049432db27299dd28` | НЕ ПРОВЕРЕНО |
| L-043 |  | `references/Plate_Benchmarks_3D_Thick/1998_Okumura_Oguma_Completely_Clamped_Thick_Plate_3D_Series_ArchApplMech.pdf` | PDF | 350117 | `cd2e543e4c79438eb7003c46fe12acb429840216b17b5a4e6c4079a6f3a82764` | НЕ ПРОВЕРЕНО |
| L-044 |  | `references/Plate_Benchmarks_3D_Thick/2000_Piltner_Comment_on_Okumura_Oguma_ArchApplMech.pdf` | PDF | 32317 | `6e7d40826fe85adf16f85e946ac9435f15582cf7ef4cfe11307809a32feaec3b` | НЕ ПРОВЕРЕНО |
| L-045 |  | `references/Plate_Benchmarks_3D_Thick/2015_Li_Ni_Cheng_Symplectic_Superposition_Thick_Plate_Benchmarks_JEM.pdf` | PDF | 364770 | `5799e0be55e4ca7b16dc793f7295da246c5ceb59c79b03f06ab4cc5ee4498a20` | НЕ ПРОВЕРЕНО |
| L-046 |  | `references/Plate_Benchmarks_Linear_BC/1997_Meleshko_Bending_Clamped_Rectangular_Plate_Exact_vs_Engineering_JElast.pdf` | PDF | 385226 | `6b58623790b454576a2513a0f9df6ee3282f34268a27ae03b1bf7454e5ac236c` | НЕ ПРОВЕРЕНО |
| L-047 |  | `references/Plate_Benchmarks_Linear_BC/2004_Bhaskar_Kaushik_Exact_Series_Orthotropic_Plates_Clamped_SS_Edges_CompStruct.pdf` | PDF | 334928 | `cb386120667acbaab6ce20f6adb602c3201697e3479bad3fdbdf9b5cb554c9b8` | НЕ ПРОВЕРЕНО |
| L-048 |  | `references/Plate_Benchmarks_Linear_BC/2018_Li_Liu_Pang_Du_Li_FG_Mindlin_Plates_General_BC_ShockVib.pdf` | PDF | 3958539 | `5fbd88dae3d3e12f6159763636f2e3ad5ceb3af6bad53150368f1e03328e502c` | НЕ ПРОВЕРЕНО |
| L-049 |  | `references/Plate_Benchmarks_Linear_BC/2019_He_An_Su_Orthotropic_Plates_Two_Opposite_Edges_Clamped_GITT_ProcIMechE_C.pdf` | PDF | 341993 | `e742ce4cab48fc1f9fb1458e3bc4b63f76abe3e5df5535f4cb17cb181ddc172a` | НЕ ПРОВЕРЕНО |
| L-050 |  | `references/Plate_Benchmarks_Linear_BC/2021_Ike_Onyia_RowlandLato_GITM_CSCS_Bending_Buckling_ASTESJ.pdf` | PDF | 958541 | `844bbc2c8b2c8bc915198437f5bc27d98dbdfc538af4442adf2f2df1a851c5c1` | НЕ ПРОВЕРЕНО |
| L-051 |  | `references/Rectangular_Plates/1942_Levy_Bending_Rectangular_Plates_Large_Deflections_NACA_TN846.pdf` | PDF | 1921843 | `89fb97727ffd0f114ffcbc5c8f4495603eab071fcb4687ef03ac09f2723e1835` | НЕ ПРОВЕРЕНО |
| L-052 |  | `references/Rectangular_Plates/1942_Levy_Square_Plate_Clamped_Edges_Large_Deflections_NACA_TN847.pdf` | PDF | 1114365 | `4bc0709785ca86c24bf4f2360c57c144da483ef9300a16656c3359e52ebb32e9` | НЕ ПРОВЕРЕНО |
| L-053 |  | `references/Rectangular_Plates/1948_Wang_Bending_Rectangular_Plates_Large_Deflections_NACA_TN1462.pdf` | PDF | 1023011 | `f5d4dfdc050886c2476b930fc26cf61df18a548ce918daa9d1f7611cf8f17592` | НЕ ПРОВЕРЕНО |
| L-054 |  | `references/Rectangular_Plates/1966_Iyengar_Naqvi_Large_Deflections_Rectangular_Plates.pdf` | PDF | 659726 | `a0f5f2e770ceed74613acb0553ea55542d36903a00f5665d45af54490fbf3302` | НЕ ПРОВЕРЕНО |
| L-055 |  | `references/Rectangular_Plates/1969_Hooke_Large_Deflection_Clamped_Rectangular_Plates_JMES.pdf` | PDF | 822007 | `f0e652cecc00dc8393d3f2aa40fa515aeab07beb7f64902102889d57c0123215` | НЕ ПРОВЕРЕНО |
| L-056 |  | `references/Rectangular_Plates/1990_Turvey_Osman_Large_Deflection_Rectangular_Mindlin_Plates_IJMS.pdf` | PDF | 640049 | `7585c342c00f77ba9aca6a34a2bf04a3722e692fab8f65848bb615938cdc8cec` | НЕ ПРОВЕРЕНО |
| L-057 |  | `references/Rectangular_Plates/2005_Wang_El-Sheikh_Large_Deflection_Rectangular_Plates_JEM.pdf` | PDF | 578603 | `71ea4736d0ca5d274bfc9912661f75450fd2173e59d15a5a68fea6970799915d` | НЕ ПРОВЕРЕНО |
| L-058 |  | `references/Rectangular_Plates/2010_Batista_Uniformly_Loaded_Rectangular_Thin_Plates_Symmetrical_BC_arXiv.pdf` | PDF | 221889 | `53cb334209eb672136cb0b92049a19152522c184844ed5b1f4b53463c6b0262d` | НЕ ПРОВЕРЕНО |
| L-059 |  | `references/Rectangular_Plates/2022_Hakim_Abramovich_Large_Deflections_Thin-Walled_Plates_InPlane_Stresses.pdf` | PDF | 1360107 | `b8f0e64df7a17293b5cb70cd9dd7b906b258b6d83c2f29676c85eaee3391040b` | НЕ ПРОВЕРЕНО |
| L-060 |  | `references/Rectangular_Plates/2023_Liu_Zhong_Liao_Thin_Rectangular_Plate_Large_Uniform_Loading_HAM.pdf` | PDF | 2536256 | `925d03ee09d960181aab10666f3933fb02b415e59a3192223fe55bffe9767015` | НЕ ПРОВЕРЕНО |
| L-061 |  | `references/Rectangular_Plates/2024_Zhang_Nonlinear_Bending_Benchmarks_Rectangular_Plates_SSSS.pdf` | PDF | 8948443 | `f7172ddaac4981b86095ed0e63a0f7c59dfd437a3aabad1e557b691c8b5528a2` | НЕ ПРОВЕРЕНО |
| L-062 |  | `references/Refined_Plate_Theories/1945_Reissner_Transverse_Shear_Deformation_Bending_Elastic_Plates_JAM.pdf` | PDF | 3562012 | `7cbe02324e06d256a3d09a12480af7552a20ddf2d14b5b818bc3fe78a28a9661` | НЕ ПРОВЕРЕНО |
| L-063 |  | `references/Refined_Plate_Theories/1951_Mindlin_Rotatory_Inertia_Shear_Flexural_Motions_Plates_JAM.pdf` | PDF | 2524795 | `e515f38e210f8afc2f5bb9ecc46327655bac806a3340ca57dbfbde067042551a` | НЕ ПРОВЕРЕНО |
| L-064 |  | `references/Refined_Plate_Theories/1964_Sheremetev_Pelekh_Refined_Plate_Theory_InzhZhurnal.pdf` | PDF | 262153 | `e10ad162821a1f9da4e805eafa190c40df869ee9e0ef37b3ddcd71b1350df415` | НЕ ПРОВЕРЕНО |
| L-065 |  | `references/Refined_Plate_Theories/1973_Grigolyuk_Selezov_Nonclassical_Vibration_Theories_Rods_Plates_Shells_VINITI.djvu` | DjVu | 3617980 | `12faf73262970223df90e76ecd00d0bb0e528b541825dd25ca81f71483b360b6` | НЕ ПРОВЕРЕНО |
| L-066 |  | `references/Refined_Plate_Theories/1980_Levinson_Accurate_Simple_Theory_Elastic_Plates_MRC.pdf` | PDF | 398296 | `cfcf8ff978e71dfa1d1d87cad7af0c0e42d7ef6289984a662d099086ed649a53` | НЕ ПРОВЕРЕНО |
| L-067 |  | `references/Refined_Plate_Theories/1981_Murthy_Improved_Transverse_Shear_Theory_Laminated_Plates_NASA_TP1903.pdf` | PDF | 1324016 | `c2068cde9d6b360e730bc31749a95f44d67abb13379e9efe00cba57fd740f0b1` | НЕ ПРОВЕРЕНО |
| L-068 |  | `references/Refined_Plate_Theories/1984_Reddy_Refined_Nonlinear_Theory_Plates_Transverse_Shear_IJSS.pdf` | PDF | 1040651 | `09154caf24a310b0e7d3c7477272a195de4321e5d30ee17ee13b1d78608d9e9a` | НЕ ПРОВЕРЕНО |
| L-069 |  | `references/Refined_Plate_Theories/1984_Reddy_Simple_Higher-Order_Theory_Laminated_Plates_JAM.pdf` | PDF | 624740 | `420549252bc4db80a27976776cd94692e22adf86e10b781c7e459ee4e258348e` | НЕ ПРОВЕРЕНО |
| L-070 |  | `references/Refined_Plate_Theories/1990_Reddy_General_Nonlinear_Third-Order_Theory_Plates_IJNLM.pdf` | PDF | 776586 | `ac08f6425ebbf26b65943d42ccac8333edf0ee939ec717c85c5859aee0cc4556` | НЕ ПРОВЕРЕНО |
| L-071 |  | `references/Refined_Plate_Theories/2007_Izv_NAN_Armenii_Mekhanika_60-4_Issue_Gevorkyan_Kirakosyan_Vasilyan.pdf` | PDF | 2214833 | `b6162c32219ecc8ff1e890c3984fb9c4d685cbc799208e834b45361138d7ef26` | НЕ ПРОВЕРЕНО |
| L-072 |  | `references/Refined_Plate_Theories/2007_Shi_Simple_Third-Order_Shear_Deformation_Theory_Plates_IJSS.pdf` | PDF | 398775 | `68cc3d05955fc3030f2fd1f025460426d20661ea3de918fb28498e339b3087a9` | НЕ ПРОВЕРЕНО |
| L-073 |  | `references/Refined_Plate_Theories/2015_Martinez-Valle_Refined_Theory_Moderately_Thick_Plates_arXiv.pdf` | PDF | 161741 | `2a707379f7e6cb313272e7119516fe6660158430f2397d36ecf060af05fab077` | НЕ ПРОВЕРЕНО |
| L-074 |  | `references/Refined_Plate_Theories/2017_Nguyen_Ngo_Nguyen-Xuan_Three-Variable_Shear_Plate_IGA_CMAME.pdf` | PDF | 1054237 | `674d2379a49a53f70e9c0227cf450eb4b75ce2d61f828482e01afad9ff9a0682` | НЕ ПРОВЕРЕНО |
| L-075 |  | `references/Strain_Localization/1992_deBorst_Muhlhaus_Gradient_Dependent_Plasticity.pdf` | PDF | 945273 | `7b483d4d60354b5e559f863fc61a5fa25d4fa29c678b185571b190c39082fe18` | НЕ ПРОВЕРЕНО |
| L-076 |  | `references/Strain_Localization/2002_Jirasek_Objective_Modeling_Strain_Localization.pdf` | PDF | 651906 | `f7a3adfacd0f2913693fe56d52a33ed7b26bda982ee72a8e33ed79ed2f87d687` | НЕ ПРОВЕРЕНО |
| L-077 |  | `references/Strain_Localization/2019_Sabet_deBorst_Structural_Softening_Mesh_Dependence_NonAssociated_Flow.pdf` | PDF | 1816247 | `2c35370ea21c29b2808d67ccaa4770f4f73fa27d269cecb814bc89ed4737a6f0` | НЕ ПРОВЕРЕНО |
| L-078 |  | `references/Strain_Localization/2022_Cervera_Wu_Kim_Chiumenti_Strain_Localization_Plasticity_PREPRINT.pdf` | PDF | 4726337 | `ca20fb12384e14976f40c585e2e74940e3d60b52fb95fd81768c35c1d8cf742b` | НЕ ПРОВЕРЕНО |
| L-079 |  | `references/Strain_Localization/2024_Hofer_Neuner_Strain_Localization_Analysis_Elastoplastic_Geomechanics.pdf` | PDF | 4295958 | `96e9296bbb1a53feae9263f399aec44deb6c7ef0426815c242c9338bce09c632` | НЕ ПРОВЕРЕНО |
| L-080 |  | `references/vim/02_Krysko_Group_Later/1982_Bochkarev_Krysko_Optimal_Design_Physically_Nonlinear_Plates_Shells_PM18-7.doc` | DOC | 621056 | `b8d80d9d9356e1e8214ee1b55c2411bdceac81d7d09cab4c78e258fd4bb816fc` | НЕ ПРОВЕРЕНО |
| L-081 |  | `references/vim/02_Krysko_Group_Later/1982_Krysko_Pavlov_Optimal_Control_Eigenfrequency_Shells_PM18-4.doc` | DOC | 544256 | `eba8e9669e1ffde1a6a6744652d921d53552b0fcd567a9802278ea86891fe952` | НЕ ПРОВЕРЕНО |
| L-082 |  | `references/vim/02_Krysko_Group_Later/1984_Bochkarev_Krysko_Variable_Thickness_Plate_Mashinostroenie.doc` | DOC | 283136 | `5999733ec9d2fa5897661e1fcbc00bdd4d5bc9cbdb48fb4def7cbebac8534c6f` | НЕ ПРОВЕРЕНО |
| L-083 |  | `references/vim/02_Krysko_Group_Later/1986_Krysko_Bochkareva_Optimal_Design_Flexible_Plates_Nonlinear_Foundation_PM22-4.doc` | DOC | 529920 | `b56440ecbd99005030a155a441dc051f65649569a120be6d5454d6273db81ac2` | НЕ ПРОВЕРЕНО |
| L-084 |  | `references/vim/02_Krysko_Group_Later/2018_Awrejcewicz_Krysko_Contact_Two_Plates_Physical_Nonlinearity_VIM.pdf` | PDF | 3580278 | `ae8fccbb1c13affb05dae1ee1ceda00218831c6325afc83fd428514dc509daec` | НЕ ПРОВЕРЕНО |
| L-085 |  | `references/vim/02_Krysko_Group_Later/2020_Awrejcewicz_Krysko_Elastic_Plastic_Plates_Shells_Thermal_Corrosion.pdf` | PDF | 1997395 | `6227216f68ec905276fae326265c6bb0b0616785e9d5b95f190754a15a59a20a` | НЕ ПРОВЕРЕНО |
| L-086 |  | `references/vim/02_Krysko_Group_Later/2022_Tebyakin_Elastic_Plastic_Nanoplates_VIM_EKM.pdf` | PDF | 590033 | `3ca10c03dce6fe681d508612c6e12b951aaf3c863590453772f0dd39ad18d226` | НЕ ПРОВЕРЕНО |
| L-087 |  | `references/vim/02_Krysko_Group_Later/2023_Krysko-jr_Physical_Nonlinearity_Porous_FG_Kirchhoff_Nanoplates.pdf` | PDF | 12258269 | `ea01c1a6c47e695437b728e5bde1e6effc38a54b1fb766cd39ba4d851e3ede82` | НЕ ПРОВЕРЕНО |
| L-088 |  | `references/vim/02_Krysko_Group_Later/2023_Krysko-jr_Physically_Nonlinear_Kirchhoff_Plates_Iterative_Methods.pdf` | PDF | 11900533 | `f86746f056ce56096e22c763e37b2d3ad65c09a19385fb754b499109920f55a5` | НЕ ПРОВЕРЕНО |
| L-089 |  | `references/vim/02_Krysko_Group_Later/2024_Krysko_Nonlinear_Porous_FG_Plates_Temperature_VIM.pdf` | PDF | 1388496 | `94cf0fd966e30876723ddbe9594a30dc3eeb29d6bfed3c07846ce27dbb901dd3` | НЕ ПРОВЕРЕНО |
| L-090 |  | `references/vim/02_Krysko_Group_Later/2025_Kalutsky_Krysko_VIM_Porous_FG_Oblique_Plates.pdf` | PDF | 1133699 | `427c418bd807009e2c5efa7805b93571c1d7ac9fc0ff37dd9c8a1c01dc6fb483` | НЕ ПРОВЕРЕНО |
| L-091 |  | `references/vim/03_Dissertations_and_Chapters/19XX_Kirichenko_Krysko_Chapter_VIM_Theory_of_Plates_Substantiation.doc` | DOC | 367616 | `38963b6721782b0db2eeb0644cf1e4890b125f065615994caad79056adfece9b` | НЕ ПРОВЕРЕНО |
| L-092 |  | `references/vim/03_Dissertations_and_Chapters/19XX_UNKNOWN_Chapter_Vlasov_Kantorovich_Physically_Nonlinear_Shallow_Shells.doc` | DOC | 1013760 | `185d38c62d5d3a98ba2b707314d6ccd335270eeebb18958d020632d53e32aea5` | НЕ ПРОВЕРЕНО |
| L-093 |  | `references/vim/03_Dissertations_and_Chapters/19XX_UNKNOWN_Monograph_Sec4-8_Optimal_Design_Physically_Nonlinear_Plates_Shells.doc` | DOC | 1395712 | `df19356bf6fe62bfddea2a569cb7ddf4b4f8f3b2cf163b47766c76b7d4245359` | НЕ ПРОВЕРЕНО |
| L-094 |  | `references/vim/03_Dissertations_and_Chapters/19XX_UNKNOWN_Monograph_Sec4-9_Optimal_Design_Flexible_Plates_Nonlinear_Foundation.doc` | DOC | 1217536 | `20eda62d1c9c792913d669c2a0aec7288cb0c38182bee5afc9f7ba1321d02694` | НЕ ПРОВЕРЕНО |
| L-095 |  | `references/vim/04_Kantorovich_EKM_Background/1968_Kerr_Alexander_EKM_Clamped_Rectangular_Plate.pdf` | PDF | 647172 | `111e02c281138295d211ffe43aefba2ebe4b7590ab02862d916512ffaf17410e` | НЕ ПРОВЕРЕНО |
| L-096 |  | `references/vim/04_Kantorovich_EKM_Background/1968_Kerr_Extension_of_Kantorovich_Method.pdf` | PDF | 835609 | `79e673d27f75caaa59776ac67a7e01980b5897ab59ffc775c506d43972e41033` | НЕ ПРОВЕРЕНО |
| L-097 |  | `references/vim/04_Kantorovich_EKM_Background/1969_Kerr_Extended_Kantorovich_Method_Eigenvalue_Problems.pdf` | PDF | 718894 | `21a5fef2e809702bb0fb56da7bcfe7499478ceeecb9ebc41ca729364fdff708f` | НЕ ПРОВЕРЕНО |
| L-098 |  | `references/vim/05_Reviews/2020_Zhigalov_Kalutsky_Variational_Iteration_Method_Plate_Theory.pdf` | PDF | 1146600 | `c4d7d9fdd9e8bae8326c924791c461452d65bfe9ba048b3f4fcc52fd2aafb694` | НЕ ПРОВЕРЕНО |
| L-100 |  | `references/Оболочки/1973_Pelekh_Theory_of_Shells_Finite_Shear_Stiffness_NaukovaDumka.djvu` | DjVu | 2348084 | `208958fcf4f8362d651ddab990925c9f37a45f29af8210d70c45c462a9e23f2d` | НЕ ПРОВЕРЕНО |
| L-101 |  | `references/Оболочки/1974_Ambartsumyan_General_Theory_Anisotropic_Shells_Nauka.djvu` | DjVu | 4489712 | `16ed380bce256442d8aea28e890898cc59b3cf56c1bb14644d4cd90b3566ad56` | НЕ ПРОВЕРЕНО |
| L-102 |  | `references/Оболочки/19XX_Manukyan_Equations_Orthotropic_Shallow_Shells_Kazan_Collection.pdf` | PDF | 750093 | `3dbd58bc22ea0fccb3403e6a339d20ca1a8695cf2625ece4e57d7d4cddda3fc8` | НЕ ПРОВЕРЕНО |
| L-103 |  | `references/Оболочки/2012_Kirakosyan_Refined_Thermoelasticity_Orthotropic_Shells_Variable_Thickness_DoklNANArmenii.pdf` | PDF | 182786 | `dbcb5c0b4a02bc7b0cfa4ba9ae1fd64db3113faa200e83cf8411618d593e4368` | НЕ ПРОВЕРЕНО |
| L-104 |  | `references/Поиск статей/The Scientific World Journal - 2014 - Krysko - On the Iterative Methods of Linearization  Decrease of Order and Dimension.pdf` | PDF | 397545 | `63e170351e4b56473bd94ca8f570296447aff22456a1ea2db59286392567e79c` | НЕ ПРОВЕРЕНО |
| L-105 |  | `references/Поиск статей/timoshenko1921.pdf` | PDF | 101684 | `dacbfbc5ec70f49645b4fc7d65330843ea63340f58c4485c6a45c6d8ebc63d48` | НЕ ПРОВЕРЕНО |
| L-106 |  | `references/Поиск статей/timoshenko1922.pdf` | PDF | 243681 | `043df6557a3e434ad191608281568381b058434741fd3e55fe4e77e30b231bd9` | НЕ ПРОВЕРЕНО |
| L-107 |  | `references/Поиск статей/Пластинки и оболочки. Тимошенко С.П., Войновский-Кригер С. 1966.djvu` | DjVu | 7139913 | `1f3eb5cf7b805b686a7149d1133e5cd1fe3b34a172f0caf035b55246479b1461` | НЕ ПРОВЕРЕНО |

### 4.1. Выведенные идентификаторы

Идентификатор выведенного файла не переиспользуется. Хеш сохранён, чтобы файл не был повторно
внесён в библиотеку без нового решения.

| ID | Прежний путь | SHA256 | Выведен | Основание |
|---|---|---|---|---|
| L-099 | `references/vim/06_Internal_Notes/2026_Internal_Report_Kirichenko_Krysko_MVI_Summary.docx` | `c7a2ad55036c9cf544e99df5f30e1f3dfbc268d919ae821576569261c69d2019` | 2026-10-11; файл удалён с диска, содержимое не читалось | Решение владельца по Q-109: внутренний отчёт, не являющийся подтверждённой оригинальной научной публикацией; D-026 |

## 5. Проверка целостности

SHA256 вычислены по содержимому файлов 2026-10-11. Проверка локальной копии: вычислить SHA256 каждого
файла (например, `sha256sum` или `Get-FileHash -Algorithm SHA256`) и сравнить со столбцом SHA256
раздела 4. Файл с несовпадающим хешем не считается публикацией каталога, пока расхождение
не объяснено.

## 6. Хранение публикаций и сведения о лицензиях

### 6.1. Политика проекта (D-021)

Оригинальные научные публикации (PDF, DjVu, DOC, DOCX) **не загружаются в GitHub, независимо
от лицензии**. Они хранятся только локально в `references/`. В GitHub хранятся индексы,
библиографические данные (`*.md`, `*.csv`, `*.txt`) и самостоятельно написанные литературные обзоры.
Решение: [`docs/DECISIONS.md`](../docs/DECISIONS.md), D-021.

Правила [`.gitignore`](../.gitignore) исключают эти форматы (а также архивы) во всём `references/`
без исключений. В клоне репозитория публикаций нет: их нужно получить самостоятельно и сверить
с SHA256 из раздела 4.

Это **добровольная политика проекта**, а не юридическая оценка.

### 6.2. Сведения о лицензиях

**Не каталогизированы; статус — НЕ ПРОВЕРЕНО.** Для D-021 они не нужны: все публикации исключены
из git независимо от лицензии. Если сведения понадобятся, их собирают заново по тексту самих
публикаций и записывают сюда как справочные; основанием для исключений из D-021 они не являются.

## 7. Происхождение

Публикации сохранены в KK-VIM по решению владельца проекта (D-026) как единственный материал,
допущенный из прежней работы. Источник получения отдельных файлов в каталоге не учитывается;
статус происхождения — **НЕ ПРОВЕРЕНО**. Идентичность файла устанавливается по SHA256 (раздел 5).
