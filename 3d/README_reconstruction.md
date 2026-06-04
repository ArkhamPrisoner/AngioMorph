# Coronary Pseudo-3D Prototype

Этот прототип строит приближённое 3D-представление коронарных сосудов из одиночной бинарной маски.

Что делает скрипт:

- выбирает самый полный кадр по площади маски для левого и правого дерева;
- дополнительно пытается выровнять крупные фазы сдвигом и собрать `temporal-fused` маску;
- очищает маску, скелетизирует её и строит граф ветвей;
- оценивает локальный радиус сосуда по distance transform;
- назначает эвристическую глубину `z` для разделения ветвей в 3D;
- экспортирует превью, centerline OBJ и tube-mesh OBJ.

Запуск:

```bash
python3 reconstruct_coronary_tree.py
```

Выходные файлы лежат в `outputs/`:

- `left_tree.obj`, `right_tree.obj`, `combined.obj`
- `left_tree_fused.obj`, `right_tree_fused.obj`, `combined_fused.obj`
- `left_tree_centerlines.obj`, `right_tree_centerlines.obj`, `combined_centerlines.obj`
- `left_tree_fused_centerlines.obj`, `right_tree_fused_centerlines.obj`, `combined_fused_centerlines.obj`
- `left_tree_preview.png`, `right_tree_preview.png`, `combined_preview.png`
- `left_tree_fused_preview.png`, `right_tree_fused_preview.png`, `combined_fused_preview.png`
- `left_tree_fused_mask.png`, `right_tree_fused_mask.png`
- `left_tree_fused_confidence.png`, `right_tree_fused_confidence.png`
- `left_tree_fused_alignment.json`, `right_tree_fused_alignment.json`
- `left_tree.json`, `right_tree.json`, `left_tree_fused.json`, `right_tree_fused.json`, `summary.json`

## 2D граф сосудов и бифуркации

Для попиксельного анализа `image + mask`, построения skeleton-графа и эвристической разметки бифуркаций/пересечений:

```bash
python3 analyze_coronary_graph.py \
  --image 'все/our_data_with_dublicates_297img/images/p0001_00000001_f0018_1.png' \
  --mask 'все/our_data_with_dublicates_297img/masks/p0001_00000001_f0018_1.png' \
  --output-dir outputs_graph_analysis \
  --name p0001_00000001_f0018_1
```

Пачка кадров с совпадающими именами файлов:

```bash
python3 analyze_coronary_graph.py \
  --image-dir 'все/our_data_with_dublicates_297img/images' \
  --mask-dir 'все/our_data_with_dublicates_297img/masks' \
  --glob '*.png' \
  --output-dir outputs_graph_analysis/manual_all \
  --no-json \
  --max-overlays 12
```

Скрипт сохраняет:

- `*_graph_analysis.json` — узлы, рёбра, радиусы, интенсивности, локальные градиенты и классификацию узлов;
- `*_graph_overlay.png` — исходное изображение с контуром маски, скелетом, отмеченными узлами, метаинформацией и легендой;
- `batch_graph_analysis_report.md` — пояснение и сводка по batch-запуску;
- `plot_*.png` — статистические графики;
- `batch_*_table.csv` — таблицы по изображениям, рёбрам и узлам.

Ограничение данных:

- В текущем наборе есть левое и правое коронарные русла, но нет двух калиброванных проекций одного и того же дерева.
- Поэтому результат не является анатомически точной 3D-реконструкцией.
- Для настоящей 3D-реконструкции нужны минимум две синхронные проекции одного сосудистого дерева и геометрия C-arm:
  - angulation/rotation;
  - source-to-isocenter и source-to-detector distance;
  - pixel spacing;
  - по возможности ECG-синхронизация.
