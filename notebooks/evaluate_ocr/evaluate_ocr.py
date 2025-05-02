import os
import csv
import pandas as pd
import cv2
import numpy as np
from Levenshtein import ratio
from PIL import Image
from pathlib import Path
from datetime import datetime
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import easyocr
import torch
from ultralytics import YOLO


# Конфигурация
IMAGES_DIR = Path("/home/lastinm/PROJECTS/credit_cards_detection/dataset/coco/valid/images")
RESULTS_CSV = "results.csv"
COMPARE_TXT = "ocr_compare.txt"
YOLO_MODEL_PATH = f'/home/lastinm/PROJECTS/credit_cards_detection/train/YOLOv12/runs/detect/train3/weights/best.pt'
VAL_DATA_PATH = r"/home/lastinm/PROJECTS/credit_cards_detection/notebooks/true OCR data/image_data.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Инициализация моделей
print("Инициализация моделей...")
yolo_model = YOLO(YOLO_MODEL_PATH)
easyocr_reader = easyocr.Reader(['en'])
trocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed").to(DEVICE)

def normalize_text(text):
    """Нормализация текста для сравнения"""
    return ''.join(str(text).upper().split())

def calculate_metrics(pred_text, true_text):
    """Безопасный расчет метрик"""
    try:
        pred_norm = normalize_text(str(pred_text))
        true_norm = normalize_text(str(true_text))
        
        if not true_norm:  # Если нет эталонного текста
            return {'similarity': 0.0, 'exact_match': False}
        
        return {
            'similarity': min(1.0, max(0.0, ratio(pred_norm, true_norm))),  # Ограничиваем 0-1
            'exact_match': pred_norm == true_norm
        }
    except:
        return {'similarity': 0.0, 'exact_match': False}

def yolo_detect(image_path):
    """Детекция объектов с помощью YOLOv8"""
    try:
        # Чтение изображения через OpenCV для сохранения цветового канала
        img_cv = cv2.imread(str(image_path))
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(img_rgb)
        
        # Детекция объектов
        results = yolo_model(image)
        
        detections = []
        for result in results:
            for box in result.boxes:
                detections.append({
                    'xmin': box.xyxy[0][0].item(),
                    'ymin': box.xyxy[0][1].item(),
                    'xmax': box.xyxy[0][2].item(),
                    'ymax': box.xyxy[0][3].item(),
                    'confidence': box.conf.item(),
                    'class': box.cls.item(),
                    'class_name': yolo_model.names[int(box.cls)]
                })
        return image, detections  # Возвращаем и изображение и детекции
    
    except Exception as e:
        print(f"YOLO detection error for {image_path}: {str(e)}")
        return None, []

def recognize_with_easyocr(image, coords, field_type):
    """Распознавание с жёсткой фильтрацией символов по типу поля"""
    try:
        img_np = np.array(image)
        x1, y1, x2, y2 = map(int, coords)
        cropped = img_np[y1:y2, x1:x2]
        
        # Определяем allowlist для каждого типа поля
        allowlists = {
            'CardNumber': '0123456789 ',
            'DateExpired': '0123456789/',
            'CardHolder': None  # Без ограничений
        }
        
        # Конфигурация для цифровых полей
        if field_type in ['CardNumber', 'DateExpired']:
            config = {
                'allowlist': allowlists[field_type],
                'text_threshold': 0.7,  # Повышаем порог для цифр
                'width_ths': 0.5,
                'link_threshold': 0.4
            }
            # Улучшаем контраст для цифр
            gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
            _, enhanced = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            cropped = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
        else:
            config = {}
        
        result = easyocr_reader.readtext(cropped, **config, detail=0)
        raw_text = ' '.join(result) if result else ''
        
        # # Пост-обработка
        # if field_type == 'CardNumber':
        #     digits = ''.join(filter(str.isdigit, raw_text))
        #     if digits:
        #         return ' '.join([digits[i:i+4] for i in range(0, len(digits), 4)])
        #     return ''
        
        # elif field_type == 'DateExpired':
        #     cleaned = ''.join(c for c in raw_text if c in '0123456789/')
        #     if len(cleaned) == 4 and '/' not in cleaned:
        #         return f"{cleaned[:2]}/{cleaned[2:4]}"
        #     return cleaned
        
        return raw_text
        
    except Exception as e:
        print(f"EasyOCR error ({field_type}): {str(e)}")
        return ""

def recognize_with_trocr(image, coords):
    """Распознавание текста с помощью TrOCR с обработкой ошибок"""
    try:
        # Обрезаем область интереса
        x1, y1, x2, y2 = map(int, coords)
        cropped = image.crop((x1, y1, x2, y2))
        
        # Подготовка изображения для TrOCR
        pixel_values = trocr_processor(cropped, return_tensors="pt").pixel_values.to(DEVICE)
        
        # Генерация текста
        generated_ids = trocr_model.generate(pixel_values)
        return trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    except Exception as e:
        print(f"TrOCR error: {str(e)}")
        return ""

def process_image(image_path, true_data):
    """Обработка одного изображения с полной обработкой ошибок"""
    try:
        # Получаем изображение и детекции
        image, detections = yolo_detect(image_path)
        if image is None:
            return []
            
        results = []
        for det in detections:
            if det['confidence'] < 0.3:  # Фильтр по уверенности
                continue
                
            coords = (det['xmin'], det['ymin'], det['xmax'], det['ymax'])
            field_type = det['class_name']
            true_text = str(true_data.get(field_type, "")).strip()
            
            try:
                # Распознавание текста
                easyocr_text = recognize_with_easyocr(image, coords, field_type)
                trocr_text = recognize_with_trocr(image, coords)
                
                # Расчет метрик
                metrics = {
                    'easyocr': calculate_metrics(easyocr_text, true_text) if true_text else {'similarity': 0, 'exact_match': False},
                    'trocr': calculate_metrics(trocr_text, true_text) if true_text else {'similarity': 0, 'exact_match': False}
                }
                
                results.append({
                    'image': os.path.basename(image_path),
                    'field_type': field_type,
                    'true_text': true_text,
                    'easyocr_text': easyocr_text,
                    'easyocr_similarity': round(metrics['easyocr']['similarity'], 4),
                    'easyocr_exact_match': metrics['easyocr']['exact_match'],
                    'trocr_text': trocr_text,
                    'trocr_similarity': round(metrics['trocr']['similarity'], 4),
                    'trocr_exact_match': metrics['trocr']['exact_match'],
                    'confidence': round(det['confidence'], 4),
                    'processing_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'bbox': f"{det['xmin']},{det['ymin']},{det['xmax']},{det['ymax']}"
                })
                
            except Exception as e:
                print(f"Processing error for {field_type} in {image_path}: {str(e)}")
                continue
                
        return results
        
    except Exception as e:
        print(f"Critical error processing {image_path}: {str(e)}")
        return []

def main():
    """Основная функция выполнения с полной обработкой ошибок"""
    try:
        # Загрузка эталонных данных с проверкой
        try:
            true_data_df = pd.read_csv(VAL_DATA_PATH)
            required_columns = ['image', 'CardNumber']
            if not all(col in true_data_df.columns for col in required_columns):
                raise ValueError(f"CSV должен содержать колонки: {required_columns}")
        except Exception as e:
            print(f"Ошибка загрузки эталонных данных: {str(e)}")
            return

        # Определение структуры результатов
        fieldnames = [
            'image', 'field_type', 'true_text', 
            'easyocr_text', 'easyocr_similarity', 'easyocr_exact_match',
            'trocr_text', 'trocr_similarity', 'trocr_exact_match',
            'processing_time', 'confidence', 'bbox'
        ]

        # 3. Инициализация файла результатов с ПЕРЕЗАПИСЬЮ
        try:
            with open(RESULTS_CSV, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()  # Всегда записываем заголовок
        except Exception as e:
            print(f"Ошибка создания файла результатов: {str(e)}")
            return

        # Обработка изображений
        processed_count = 0
        all_results = []  # Собираем все результаты в памяти

        for _, row in true_data_df.iterrows():
            image_path = IMAGES_DIR / row['image']
            if not image_path.exists():
                print(f"Предупреждение: изображение не найдено - {image_path}")
                continue

            try:
                # Подготовка эталонных данных
                true_data = {
                    'CardHolder': str(row.get('CardHolder', '')).strip(),
                    'CardNumber': str(row.get('CardNumber', '')).strip(),
                    'DateExpired': str(row.get('DateExpired', '')).strip()
                }

                # Обработка изображения
                results = process_image(image_path, true_data)
                if not results:
                    print(f"Предупреждение: нет результатов для {image_path}")
                    continue

                all_results.extend(results)
                processed_count += 1
                print(f"Успешно обработано: {image_path}")

            except Exception as e:
                print(f"Ошибка обработки изображения {image_path}: {str(e)}")
                continue

        # 5. Запись ВСЕХ результатов одним блоком
        if all_results:
            try:
                with open(RESULTS_CSV, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    for result in all_results:
                        # Гарантируем наличие всех полей
                        row_data = {field: result.get(field, '') for field in fieldnames}
                        writer.writerow(row_data)
            except Exception as e:
                print(f"Ошибка записи результатов: {str(e)}")
                return

        # Генерация статистики
        if processed_count > 0:
            try:
                generate_summary_stats()
                print(f"\nОбработка завершена. Обработано изображений: {processed_count}")
                print(f"Результаты сохранены в: {RESULTS_CSV}")
                print(f"Статистика сохранена в: {COMPARE_TXT}")
            except Exception as e:
                print(f"Ошибка генерации статистики: {str(e)}")
        else:
            print("Нет изображений для обработки")

        if processed_count > 0:
            try:
                # Простая визуализация
                import matplotlib.pyplot as plt
                df = pd.read_csv(RESULTS_CSV)
                
                plt.figure(figsize=(10, 5))
                df['easyocr_similarity'].hist(alpha=0.5, label='EasyOCR')
                df['trocr_similarity'].hist(alpha=0.5, label='TrOCR')
                plt.title("Сравнение точности OCR")
                plt.legend()
                plt.savefig("ocr_comparison.png")
                print("График сохранён в ocr_comparison.png")
            except Exception as e:
                print(f"Ошибка визуализации: {str(e)}")

    except Exception as e:
        print(f"Критическая ошибка в main(): {str(e)}")

def generate_summary_stats():
    """Генерация статистики с полной обработкой ошибок"""
    try:
        # Чтение с проверкой типов данных
        df = pd.read_csv(RESULTS_CSV)
        
        # Проверка и преобразование числовых полей
        required_columns = {
            'easyocr_similarity': float,
            'trocr_similarity': float,
            'confidence': float
        }
        
        for col, dtype in required_columns.items():
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(dtype)
            else:
                print(f"Предупреждение: отсутствует столбец {col}")
                df[col] = 0.0
        
        # Расчет статистики
        stats = {
            'total_images': df['image'].nunique(),
            'total_fields': len(df),
            'easyocr': {
                'avg_similarity': df['easyocr_similarity'].mean(),
                'exact_match_rate': df['easyocr_exact_match'].mean(),
                'top_errors': df[df['easyocr_similarity'] < 0.5]['easyocr_text'].value_counts().head(5)
            },
            'trocr': {
                'avg_similarity': df['trocr_similarity'].mean(),
                'exact_match_rate': df['trocr_exact_match'].mean(),
                'top_errors': df[df['trocr_similarity'] < 0.5]['trocr_text'].value_counts().head(5)
            }
        }
        
        # Сохранение статистики
        with open(COMPARE_TXT, 'w', encoding='utf-8') as f:
            f.write("=== Итоговая статистика ===\n\n")
            f.write(f"Всего обработано изображений: {stats['total_images']}\n")
            f.write(f"Всего распознанных полей: {stats['total_fields']}\n\n")
            
            f.write("EasyOCR:\n")
            f.write(f"  Средняя схожесть: {stats['easyocr']['avg_similarity']:.2%}\n")
            f.write(f"  Точное совпадение: {stats['easyocr']['exact_match_rate']:.2%}\n")
            f.write("  Топ ошибок:\n")
            for text, count in stats['easyocr']['top_errors'].items():
                f.write(f"    - {text}: {count} случаев\n")
            
            f.write("\nTrOCR:\n")
            f.write(f"  Средняя схожесть: {stats['trocr']['avg_similarity']:.2%}\n")
            f.write(f"  Точное совпадение: {stats['trocr']['exact_match_rate']:.2%}\n")
            f.write("  Топ ошибок:\n")
            for text, count in stats['trocr']['top_errors'].items():
                f.write(f"    - {text}: {count} случаев\n")
        
        print("Статистика успешно сгенерирована")
        
    except Exception as e:
        print(f"Критическая ошибка при генерации статистики: {str(e)}")

if __name__ == "__main__":
    main()