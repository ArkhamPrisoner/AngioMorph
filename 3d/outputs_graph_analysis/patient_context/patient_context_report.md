# Пациентский анализ коронарного графа

Этот отчёт смотрит не один кадр, а последовательности кадров одного пациента и разные серии как разные проекции или разные проходы контраста.

## Что добавлено

- Временная аналитика: площадь маски, число бифуркаций и средний сосудистый сигнал по номеру кадра.
- Межпроекционное сравнение: серии одного пациента сравниваются по сложности графа, числу бифуркаций, наложений и радиусам.
- Анатомическая проверка: результат сверяется с упрощённым шаблоном LM/LAD/LCx/RCA, но только как подсказка, а не как жёсткое правило.

## Важное ограничение

Без геометрии C-arm и синхронизации двух проекций это не настоящая 3D-реконструкция. Это согласование графов и временных признаков, которое помогает найти подозрительные места.

## Сводка по сериям

### p0001 / серия 00000001

- Анатомическая зона: `left_coronary`
- Ожидаемая схема: LM -> LAD + LCx
- Ожидаемые ветви: LAD diagonal/septal branches; LCx obtuse marginal branches; optional ramus intermedius
- Разметчиков/последовательностей: 2/2
- Медианный пик бифуркаций: 13.5
- Всего crossing/overlap: 63
- Средний анатомический балл: 0.696

### p0001 / серия 00000002

- Анатомическая зона: `left_coronary`
- Ожидаемая схема: LM -> LAD + LCx
- Ожидаемые ветви: LAD diagonal/septal branches; LCx obtuse marginal branches; optional ramus intermedius
- Разметчиков/последовательностей: 1/1
- Медианный пик бифуркаций: 10.0
- Всего crossing/overlap: 36
- Средний анатомический балл: 0.705

### p0001 / серия 00000003

- Анатомическая зона: `left_coronary`
- Ожидаемая схема: LM -> LAD + LCx
- Ожидаемые ветви: LAD diagonal/septal branches; LCx obtuse marginal branches; optional ramus intermedius
- Разметчиков/последовательностей: 1/1
- Медианный пик бифуркаций: 12.0
- Всего crossing/overlap: 17
- Средний анатомический балл: 0.675

### p0001 / серия 00000004

- Анатомическая зона: `right_coronary`
- Ожидаемая схема: RCA -> acute marginal branches -> PDA/PL near crux, depending on dominance
- Ожидаемые ветви: acute marginal, PDA, posterolateral branches; dominance can change distal topology
- Разметчиков/последовательностей: 1/1
- Медианный пик бифуркаций: 4.0
- Всего crossing/overlap: 4
- Средний анатомический балл: 0.731

### p0003 / серия 00000001

- Анатомическая зона: `unknown`
- Ожидаемая схема: unknown projection/territory
- Ожидаемые ветви: use generic coronary tree checks only
- Разметчиков/последовательностей: 1/1
- Медианный пик бифуркаций: 3.0
- Всего crossing/overlap: 4
- Средний анатомический балл: 0.724

### p0004 / серия 00000003

- Анатомическая зона: `unknown`
- Ожидаемая схема: unknown projection/territory
- Ожидаемые ветви: use generic coronary tree checks only
- Разметчиков/последовательностей: 2/2
- Медианный пик бифуркаций: 7.0
- Всего crossing/overlap: 14
- Средний анатомический балл: 0.758

### p0004 / серия 00000004

- Анатомическая зона: `unknown`
- Ожидаемая схема: unknown projection/territory
- Ожидаемые ветви: use generic coronary tree checks only
- Разметчиков/последовательностей: 1/1
- Медианный пик бифуркаций: 4.0
- Всего crossing/overlap: 1
- Средний анатомический балл: 0.777

### p0005 / серия 00000001

- Анатомическая зона: `unknown`
- Ожидаемая схема: unknown projection/territory
- Ожидаемые ветви: use generic coronary tree checks only
- Разметчиков/последовательностей: 1/1
- Медианный пик бифуркаций: 10.0
- Всего crossing/overlap: 27
- Средний анатомический балл: 0.713

### p0006 / серия 00000001

- Анатомическая зона: `unknown`
- Ожидаемая схема: unknown projection/territory
- Ожидаемые ветви: use generic coronary tree checks only
- Разметчиков/последовательностей: 2/2
- Медианный пик бифуркаций: 9.0
- Всего crossing/overlap: 98
- Средний анатомический балл: 0.731

### p0006 / серия 00000002

- Анатомическая зона: `unknown`
- Ожидаемая схема: unknown projection/territory
- Ожидаемые ветви: use generic coronary tree checks only
- Разметчиков/последовательностей: 1/1
- Медианный пик бифуркаций: 9.0
- Всего crossing/overlap: 88
- Средний анатомический балл: 0.698

## Графики

- `p0001_temporal_curves.png`
- `p0003_temporal_curves.png`
- `p0004_temporal_curves.png`
- `p0005_temporal_curves.png`
- `p0006_temporal_curves.png`
- `projection_anatomy_score.png`
- `projection_bifurcation_crossing.png`
- `canonical_coronary_topology.png`

## Таблицы

- `patient_series_temporal_summary.csv`: одна строка на пациента/серию/разметчика.
- `patient_projection_summary.csv`: агрегат по пациенту и серии.