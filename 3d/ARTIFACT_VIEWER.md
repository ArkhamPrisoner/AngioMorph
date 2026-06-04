# Artifact Viewer

Запуск:

```bash
cd "/home/nosignalx2k/Рабочий стол/work/БСМП/3d"
python3 artifact_viewer_server.py
```

Открыть в браузере:

```text
http://127.0.0.1:8090
```

По умолчанию viewer смотрит в:

```text
remote_artifacts/20260422
```

Можно указать другую папку:

```bash
python3 artifact_viewer_server.py --artifact-root /path/to/artifacts --port 8091
```

Что умеет:
- `PLY` открывает как point cloud
- `NPZ` с `occupancy` открывает как sampled volume points
- `NPY` объёмы открывает по настраиваемому порогу
- можно загрузить свой `ply / npz / npy` через кнопку или drag-and-drop
- загруженные файлы сохраняются в подпапку `_uploads` внутри текущего `artifact-root`
