import os
import sys
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# Добавляем папку '3d' в пути поиска Python, 
# чтобы обойти запрет на импорт из папок, начинающихся с цифры
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '3d'))

# Импортируем алгоритм из AngioMorph
from analyze_coronary_graph import analyze_graph 

def create_image_pipeline(input_dir, yolo_weights_path, output_video_path, temp_dir="temp_pipeline", fps=10):
    print(f"🚀 Запуск пайплайна для папки: {input_dir}")
    
    # 1. Подготовка папок
    temp_dir = Path(temp_dir)
    img_dir = temp_dir / "images"
    mask_dir = temp_dir / "masks"
    overlay_dir = temp_dir / "overlays"
    
    for d in [img_dir, mask_dir, overlay_dir]:
        d.mkdir(parents=True, exist_ok=True)
        
    # 2. Ищем все изображения в исходной папке и сортируем их
    valid_extensions = ('.png', '.jpg', '.jpeg')
    image_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(valid_extensions)])
    
    if not image_files:
        print(f"❌ В папке {input_dir} не найдено изображений!")
        return
        
    print(f"Найдено изображений для обработки: {len(image_files)}")
    
    # 3. Загрузка модели YOLO
    print("Загрузка модели YOLO...")
    model = YOLO(yolo_weights_path)

    print("Шаг 1 и 2: Генерация масок и анализ графов...")
    for frame_count, img_name in enumerate(image_files):
        img_path_src = os.path.join(input_dir, img_name)
        frame = cv2.imread(img_path_src)
        
        if frame is None:
            print(f"⚠️ Не удалось прочитать изображение: {img_name}")
            continue
            
        height, width = frame.shape[:2]
        
        # Временные пути для AngioMorph
        img_path = img_dir / img_name
        mask_path = mask_dir / img_name
        
        # Сохраняем копию кадра
        cv2.imwrite(str(img_path), frame)
        
        # --- Инференс YOLO ---
        results = model(frame, verbose=False)
        binary_mask = np.zeros((height, width), dtype=np.uint8)
        
        if results[0].masks is not None:
            mask_tensor = results[0].masks.data[0].cpu().numpy()
            mask_resized = cv2.resize(mask_tensor, (width, height), interpolation=cv2.INTER_NEAREST)
            binary_mask = (mask_resized * 255).astype(np.uint8)
            
        # Сохраняем маску для AngioMorph
        cv2.imwrite(str(mask_path), binary_mask)
        
        # --- Запуск AngioMorph ---
        try:
            analyze_graph(
                image_path=img_path,       # Убрали str(), передаем объект Path
                mask_path=mask_path,       # Убрали str(), передаем объект Path
                output_dir=overlay_dir,    # Убрали str(), передаем объект Path
                name=img_path.stem         # <-- ДОБАВЛЕН НЕДОСТАЮЩИЙ АРГУМЕНТ (имя без расширения)
            )
        except Exception as e:
            print(f"Ошибка AngioMorph на кадре {img_name}: {e}")
            
        # Небольшой лог прогресса (каждые 10 кадров)
        if (frame_count + 1) % 10 == 0:
            print(f"Обработано {frame_count + 1} / {len(image_files)} кадров...")
            
# 4. Сборка итогового видео из картинок AngioMorph
    print("Шаг 3: Сборка итогового видеоряда...")
    
    # Ищем ТОЛЬКО сгенерированные PNG картинки
    overlay_files = sorted(list(overlay_dir.glob("*.png")))
    
    if not overlay_files:
        print("❌ AngioMorph не сгенерировал ни одного overlay-кадра. Видео не собрано.")
        return
        
    # Ищем первый нормально читаемый кадр, чтобы узнать размер
    first_overlay = None
    for ov_path in overlay_files:
        img = cv2.imread(str(ov_path))
        if img is not None and img.size > 0:
            first_overlay = img
            break
            
    if first_overlay is None:
        print(f"❌ Ни один из {len(overlay_files)} сгенерированных кадров не удалось прочитать библиотекой OpenCV.")
        return
        
    height, width = first_overlay.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    valid_count = 0
    for ov_path in overlay_files:
        img = cv2.imread(str(ov_path))
        if img is not None:
            img = cv2.resize(img, (width, height))
            out.write(img)
            valid_count += 1
        else:
            print(f"⚠️ Пропущен битый кадр: {ov_path.name}")
            
    out.release()
    print(f"🎉 Готово! Итоговое видео ({valid_count} кадров из {len(overlay_files)}) сохранено в: {output_video_path}")

# --- Запуск ---
if __name__ == "__main__":
    create_image_pipeline(
        input_dir="/home/yaroslav-demkin/patient/22_crop",           # Ваша папка со снимками
        yolo_weights_path="/home/yaroslav-demkin/ddmp/best.pt",  # Замените на путь к вашим весам
        output_video_path="final_angiogram_analysis.mp4", # Куда сохранить видео
        fps=10  # Скорость видео (кадров в секунду)
    )