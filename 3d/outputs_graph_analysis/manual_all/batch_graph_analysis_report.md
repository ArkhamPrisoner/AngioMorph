# Batch graph analysis

Сводка по статистическому анализу пар `image + mask` через skeleton-граф.

## Пояснение

- `bifurcation` ставится в основном по degree-3 узлу skeleton-графа, углам ветвей, локальным радиусам, Murray-like балансу радиусов, согласованности серого сигнала и локальному контрасту.
- `crossing_or_overlap` ставится в основном по degree-4 узлу, где есть две пары почти противоположных продолжений с похожими радиусами и сигналом.
- `uncertain_*` нужно рассматривать вручную: одна 2D-проекция не всегда отделяет реальную анатомическую бифуркацию от проекционного наложения.

## Dataset summary

- Images analyzed: 297
- Edges analyzed: 35493
- Junctions analyzed: 5478
- Bifurcations: 961
- Crossings/overlaps: 352
- Uncertain junctions: 4165
- Mean edge radius, px: 5.162
- Median edge length, px: 8.243
- Mean junction confidence: 0.674

## Generated plots

- `plot_junction_counts.png`
- `plot_edge_distributions.png`
- `plot_length_vs_radius.png`
- `plot_junction_confidence.png`
- `plot_junctions_by_annotator.png`

## Output tables

- `batch_graph_analysis_table.csv`: per-image graph and signal statistics.
- `batch_edges_table.csv`: per-edge length/radius/signal statistics.
- `batch_junctions_table.csv`: per-junction class and confidence.
