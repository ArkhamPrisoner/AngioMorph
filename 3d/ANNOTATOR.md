# Frame Annotator

Локальный веб-разметчик кадров для ручных точечных меток:

- `stenosis`
- `bifurcation`
- `intersection`
- `uncertain`

С поддержкой нескольких форм:

- `point`
- `segment`
- `box`

## Запуск

```bash
cd "/home/nosignalx2k/Рабочий стол/work/БСМП/3d"
python3 frame_annotator_server.py
```

По умолчанию откроется датасет `p0001_unique` на `http://127.0.0.1:8765`.

## Другой набор кадров

```bash
python3 frame_annotator_server.py --dataset-dir "все/your_dataset"
```

Ожидается структура:

```text
dataset/
  images/
  masks/
```

## Что умеет

- переключение по сериям и кадрам
- наложение маски на исходник
- постановка точек, отрезков и прямоугольных областей
- комментарий к каждой точке
- удаление точки
- `Ctrl+S` для сохранения
- `Ctrl+Z` и `Ctrl+Shift+Z` для истории действий

## Горячие клавиши

- `1`: `stenosis`
- `2`: `bifurcation`
- `3`: `intersection`
- `4`: `uncertain`
- `Q`: `point`
- `W`: `segment`
- `E`: `box`
- `←/→`: предыдущий/следующий кадр
- `Delete`: удалить выбранную метку
- `Esc`: снять выделение или отменить незавершённую фигуру
- `F`: вписать изображение в окно

## Сохранение

По умолчанию JSON пишется в:

```text
<dataset-dir>/manual_annotations.json
```

Можно переопределить:

```bash
python3 frame_annotator_server.py --annotations outputs/manual_annotations.json
```
