# Handoff for Reconstruction Work

## Что в проекте уже считается рабочим

- Подбор DICOM-backed пар проекций: `prepare_dicom_projection_pairs.py`
- Экспорт reconstruction-ready кейсов: `export_reconstruction_cases.py`
- Calibrated voxel baseline: `run_calibrated_voxel_baseline.py`, `run_baseline_experiments.py`
- Подготовка и оценка NeCA-кейсов: `prepare_neca_case.py`, `evaluate_neca_case.py`, `run_neca_sweep.py`
- Просмотр 3D-артефактов в браузере: `artifact_viewer_server.py`
- Ручная разметка кадров: `frame_annotator_server.py`

## Главная структура

- Исходный проект: `.`
- Подобранные пары и DICOM-мета: `outputs_dicom_projection_pairs/`
- Кейсы для реконструкции: `reconstruction_cases/`
- Локально стянутые remote-артефакты: `remote_artifacts/20260422/`
- Реальные phase-pair реконструкции: `outputs_p0001_realmask_phasepair/`
- Tube scaffold: `outputs_p0001_calibrated_tube_graph/`

## Быстрый рабочий маршрут

### 1. Обновить пары проекций

```bash
cd "/home/nosignalx2k/Рабочий стол/work/БСМП/3d"
./.venv-dicom/bin/python prepare_dicom_projection_pairs.py
```

Результаты:
- `outputs_dicom_projection_pairs/summary.json`
- `outputs_dicom_projection_pairs/projection_pairs_angle_ge25.json`
- `outputs_dicom_projection_pairs/projection_pairs_angle_ge25.csv`

Что делает:
- матчится к DICOM
- вытягивает геометрию съёмки
- собирает пары с углом `>= 25°`
- прикладывает phase-pair примеры

### 2. Экспортировать кейсы для реконструкции

```bash
./.venv-dicom/bin/python export_reconstruction_cases.py
```

Результат: `reconstruction_cases/<case_id>/`

Внутри каждого кейса:
- `case_manifest.json`
- `phase_pairs/`
- `images.npy`
- `masks.npy`
- PNG-маски и кадры для выбранных пар фаз

### 3. Прогнать baseline

Один кейс:

```bash
python3 run_calibrated_voxel_baseline.py \
  --case-manifest reconstruction_cases/p0001__00000001__00000002/case_manifest.json \
  --output-dir tmp_baseline_run
```

Пакетно по кейсам:

```bash
python3 run_baseline_experiments.py
```

Основной результат уже есть в:
- `remote_artifacts/20260422/baseline_summary.json`
- `remote_artifacts/20260422/baseline_summary.csv`

Лучший baseline-кейс:
- `remote_artifacts/20260422/baseline_best_p0001/`

### 4. Подготовить кейс под NeCA

Пример:

```bash
python3 prepare_neca_case.py \
  --case-id p0001__00000001__00000002 \
  --phase-index 2 \
  --input-mode edt_mask \
  --bundle-size 3 \
  --detector-size 256 \
  --volume-size 128 \
  --volume-extent-mm 160 \
  --epochs 120
```

Что важно:
- `input-mode`: `mask`, `image`, `masked_image`, `edt_mask`, `blurred_mask`
- лучший текущий режим: `edt_mask`
- `bundle-size=3` лучше, чем `1`

### 5. Оценить NeCA-кейс

```bash
python3 evaluate_neca_case.py \
  --repo-dir /path/to/NeCA \
  --case-dir neca_cases/p0001__00000001__00000002/phase_02_b3_edt_mask_256
```

Что пишет:
- `evaluation/summary.json`
- `evaluation/overlay_a_*.png`
- `evaluation/overlay_b_*.png`

### 6. Прогнать sweep по NeCA

```bash
python3 run_neca_sweep.py \
  --repo-dir /path/to/NeCA \
  --top-k 4 \
  --input-modes mask,edt_mask,blurred_mask \
  --bundle-sizes 1,3 \
  --detector-sizes 256 \
  --volume-sizes 128 \
  --volume-extents 160 \
  --bounds 0.25 \
  --epochs 120
```

Смотреть сюда:
- `remote_artifacts/20260422/neca_sweep_best/neca_sweep_summary.json`
- `remote_artifacts/20260422/neca_sweep_best/remote_artifacts/20260422/neca_sweep_summary.csv`

## Что сейчас считается лучшим результатом

### Practical best

Calibrated voxel baseline, не NeCA.

Подтверждение:
- `REMOTE_EXPERIMENT_RESULTS_2026-04-22.md`
- лучший baseline Dice примерно `0.595`
- лучший NeCA Dice примерно `0.139`

### Почему так

- baseline стабильно отработал на всех кейсах
- NeCA технически поднят, но пока даёт слишком слабый volumetric signal
- raw real-mask matching без stabilizer тоже слабый

## Что нельзя перепутать

### 1. Не путать real masks и temporal/fused masks

- Для честной phase-pair реконструкции нужен `build_p0001_realmask_phasepair_recon.py`
- Он работает только по реальным маскам выбранной пары фаз
- Результат сейчас хуже, чем на temporal-assisted matching, но он честнее

Смотреть сюда:
- `outputs_p0001_realmask_phasepair/`

### 2. Не считать, что NeCA volume уже хороший 3D

Лучший volume NeCA почти константный по амплитуде.
Поэтому `obj/ply`, сделанные напрямую из него, визуально мусорные.

Для проверки есть:
- `convert_volume_artifact.py`
- `remote_artifacts/20260422/neca_sweep_best/converted/`

## Визуализация и ручная работа

### Browser viewer

Запуск:

```bash
python3 artifact_viewer_server.py --host 127.0.0.1 --port 8090
```

Открыть:
- `http://127.0.0.1:8090`

Документация:
- `ARTIFACT_VIEWER.md`

Что умеет:
- смотреть `ply`, `npy`, `npz`
- загружать свои файлы через браузер

### Разметчик кадров

Запуск:

```bash
python3 frame_annotator_server.py
```

Открыть:
- `http://127.0.0.1:8765`

Документация:
- `ANNOTATOR.md`

Что умеет:
- точки, отрезки, области
- типы: стеноз, бифуркация, пересечение, спорный участок
- `Ctrl+Z` / `Ctrl+Shift+Z`

## Что читать в первую очередь

1. `REMOTE_EXPERIMENT_RESULTS_2026-04-22.md`
2. `NECA_SWEEP_RESULTS_2026-04-22.md`
3. `CANDIDATE_AUDIT_2026-04-22.md`
4. `outputs_dicom_projection_pairs/summary.json`
5. `reconstruction_cases/<case_id>/case_manifest.json`

## Что делать дальше, если продолжать работу

### Если нужен лучший practical result

Идти от calibrated graph / tube / centerline pipeline, а не от NeCA-меша.

Ключевые файлы:
- `build_p0001_calibrated_tube_graph.py`
- `build_p0001_realmask_phasepair_recon.py`

### Если нужен именно SOTA-like neural direction

Нужно не "запустить ещё раз NeCA", а делать адаптацию следующего уровня:
- лучшее описание geometry
- другая representation/loss
- жёсткий topology prior
- возможно адаптация `SDF-CAR`

Перед этим обязательно прочитать:
- `CANDIDATE_AUDIT_2026-04-22.md`

## Честный статус кандидатов

### NeCA
- поднят
- адаптирован
- запускается
- пока проигрывает baseline

### 3DGR-CAR
- клонирован на remote
- не адаптирован
- не benchmarked

### SDF-CAR
- клонирован и просмотрен
- в текущем виде не drop-in
- требует адаптации, потому что ожидает GT volume

## Минимальный handoff-комплект протеже

Если передавать только главное, передать эти файлы:
- `prepare_dicom_projection_pairs.py`
- `export_reconstruction_cases.py`
- `run_calibrated_voxel_baseline.py`
- `run_baseline_experiments.py`
- `prepare_neca_case.py`
- `evaluate_neca_case.py`
- `run_neca_sweep.py`
- `artifact_viewer_server.py`
- `frame_annotator_server.py`
- `REMOTE_EXPERIMENT_RESULTS_2026-04-22.md`
- `NECA_SWEEP_RESULTS_2026-04-22.md`
- `CANDIDATE_AUDIT_2026-04-22.md`
