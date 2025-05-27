# 1. Детектируем поля реквизитов Yolo
# 2. Далее распознаем текст в областях EasyOCR, PaddleOCR и TrOCR
# 3. Сравниваем с эталонными значениями
# 4. Результаты пишем в файл results.csv
# === Описание версии
# Для распознавания номеров карт в TrOCR используем в наборе только цифры

import os
import tempfile
#import re
import csv
import pandas as pd
import cv2
import numpy as np
from Levenshtein import ratio
from PIL import Image
from pathlib import Path
from datetime import datetime
from transformers import TrOCRProcessor, TrOCRForCausalLM, VisionEncoderDecoderModel, AutoTokenizer
import easyocr
from paddleocr import TextRecognition
import torch
from ultralytics import YOLO


# Конфигурация
ROOT_DIR = '/home/lastinm/PROJECTS/credit_cards_detection'
IMAGES_DIR = Path(rf"{ROOT_DIR}/dataset/ocr val")
RESULTS_CSV = fr"{ROOT_DIR}/auxiliary/Calculating statistics OCR/results.csv"
YOLO_MODEL_PATH = rf"{ROOT_DIR}/train/YOLOv12/runs/detect/train4/weights/best.pt"
VAL_DATA_PATH = rf"{ROOT_DIR}/auxiliary/Reference values for OCR/image_data.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Инициализация моделей
print("Инициализация моделей...")
yolo_model = YOLO(YOLO_MODEL_PATH)

easyocr_reader = easyocr.Reader(['en'])
paddleocr_reader = TextRecognition()

# Инициализация компонентов TrOCR
print("Инициализация TrOCR...")
trocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed").to(DEVICE)
trocr_tokenizer = AutoTokenizer.from_pretrained("microsoft/trocr-base-printed")

# Создаем словарь цифровых токенов
DIGIT_TOKEN_IDS = {
    token_id: token for token, token_id in trocr_tokenizer.get_vocab().items()
    if token.isdigit() or token in ['', '</s>', '<s>', '<pad>']
}

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
            'similarity': min(1.0, max(0.0, ratio(pred_norm, true_norm))),  # Ограничиваем 0-1. Используем расстояние Ливенштейна
            'exact_match': pred_norm == true_norm
        }
    except:
        return {'similarity': 0.0, 'exact_match': False}

def generate_digits_only(pixel_values, max_new_tokens=20):
    """Генерация только цифровых последовательностей с правильными параметрами"""
    generated_ids = trocr_model.generate(
        pixel_values,
        max_new_tokens=max_new_tokens,
        bad_words_ids=[[tid] for tid in range(len(trocr_tokenizer)) if tid not in DIGIT_TOKEN_IDS],
        num_beams=3,
        early_stopping=True
    )
    return trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

def yolo_detect(image_path):
    """Детекция объектов с помощью YOLOv8"""
    try:
        img_cv = cv2.imread(str(image_path))
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(img_rgb)
        
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
        return image, detections
    except Exception as e:
        print(f"YOLO detection error for {image_path}: {str(e)}")
        return None, []

def recognize_with_easyocr(image, coords, field_type):
    """Распознавание с жёсткой фильтрацией символов по типу поля"""
    try:
        img_np = np.array(image)
        x1, y1, x2, y2 = map(int, coords)
        cropped = img_np[y1:y2, x1:x2]
        
        allowlists = {
            'CardNumber': '0123456789 ',
            'DateExpired': '0123456789/-',
            'CardHolder': None
        }
        
        if field_type in ['CardNumber', 'DateExpired']:
            config = {
                'allowlist': allowlists[field_type],
                'text_threshold': 0.7,
                'width_ths': 0.5,
                'link_threshold': 0.4
            }
        else:
            config = {}
            cropped = cv2.resize(cropped, None, fx=2, fy=1, interpolation=cv2.INTER_CUBIC)
        
        result = easyocr_reader.readtext(cropped, **config, detail=0)
        return ' '.join(result) if result else ''
    except Exception as e:
        print(f"EasyOCR error ({field_type}): {str(e)}")
        return ""

def recognize_cardnumber_with_trocr(image, coords):
    """Распознавание только номера карты (только цифры)"""
    try:
        x1, y1, x2, y2 = map(int, coords)
        cropped = image.crop((x1, y1, x2, y2))
        
        # Минимальная обработка изображения
        pixel_values = trocr_processor(cropped, return_tensors="pt").pixel_values.to(DEVICE)
        
        # Генерация только цифр
        generated_ids = trocr_model.generate(
            pixel_values,
            max_new_tokens=19,  # 16 цифр + 3 пробела
            bad_words_ids=[[tid] for tid in range(len(trocr_tokenizer)) if tid not in DIGIT_TOKEN_IDS],
            num_beams=3,
            early_stopping=True
        )
        
        raw_text = trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Постобработка: оставляем только цифры и форматируем
        digits = ''.join(filter(str.isdigit, raw_text))[:16]  # Берем первые 16 цифр
        return ' '.join([digits[i:i+4] for i in range(0, len(digits), 4)])  # Формат XXXX XXXX XXXX XXXX
        
    except Exception as e:
        print(f"CardNumber recognition error: {str(e)}")
        return ""
    
# В функции recognize_with_trocr заменяем обработку CardNumber:
def recognize_with_trocr(image, coords, field_type):
    if field_type == "CardNumber":
        return recognize_cardnumber_with_trocr(image, coords)
    # Остальная обработка других полей...
    try:
        x1, y1, x2, y2 = map(int, coords)
        cropped = image.crop((x1, y1, x2, y2))
        pixel_values = trocr_processor(cropped, return_tensors="pt").pixel_values.to(DEVICE)
        generated_ids = trocr_model.generate(pixel_values)
        return trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    except Exception as e:
        print(f"TrOCR error: {str(e)}")
        return ""
    
def recognize_with_paddleocr(image, coords):
    try:
        x1, y1, x2, y2 = map(int, coords)
        cropped = image.crop((x1, y1, x2, y2))
        # Создаем временный файл
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            temp_path = tmp_file.name        
        # Сохраняем вырезанную область во временный файл
        cropped.save(temp_path, quality=95)

        results = paddleocr_reader.predict(input=temp_path)
        
        # Удаляем временный файл
        os.unlink(temp_path)

        return results[0]['rec_text']
    except Exception as e:
        print(f"PaddleOCR error: {str(e)}")
        return ""

def recognize_with_ensemble(image, coords, field_type):
    if field_type == "CardNumber":
        return recognize_with_paddleocr(image, coords)
    else:
        return recognize_with_trocr(image, coords, field_type)

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
                easyocr_text    = recognize_with_easyocr(image, coords, field_type)
                trocr_text      = recognize_with_trocr(image, coords, field_type)
                paddleocr_text  = recognize_with_paddleocr(image, coords)
                ensemble_text   = recognize_with_ensemble(image, coords, field_type)
                
                # Расчет метрик
                metrics = {
                    'easyocr':  calculate_metrics(easyocr_text, true_text) if true_text else {'similarity': 0, 'exact_match': False},
                    'trocr':    calculate_metrics(trocr_text, true_text) if true_text else {'similarity': 0, 'exact_match': False},
                    'paddleocr': calculate_metrics(paddleocr_text, true_text) if true_text else {'similarity': 0, 'exact_match': False},
                    'ensemble': calculate_metrics(ensemble_text, true_text) if true_text else {'similarity': 0, 'exact_match': False}
                }
                
                results.append({
                    'image': os.path.basename(image_path),
                    'field_type': field_type,
                    'true_text': true_text,
                    'paddleocr_text': paddleocr_text,
                    'paddleocr_similarity': round(metrics['paddleocr']['similarity'], 4),
                    'paddleocr_exact_match': metrics['paddleocr']['exact_match'],
                    'easyocr_text': easyocr_text,
                    'easyocr_similarity': round(metrics['easyocr']['similarity'], 4),
                    'easyocr_exact_match': metrics['easyocr']['exact_match'],
                    'trocr_text': trocr_text,
                    'trocr_similarity': round(metrics['trocr']['similarity'], 4),
                    'trocr_exact_match': metrics['trocr']['exact_match'],
                    'ensemble_text': ensemble_text,
                    'ensemble_similarity': round(metrics['ensemble']['similarity'], 4),
                    'ensemble_exact_match': metrics['ensemble']['exact_match'],
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
            'paddleocr_text', 'paddleocr_similarity', 'paddleocr_exact_match',
            'easyocr_text', 'easyocr_similarity', 'easyocr_exact_match',
            'trocr_text', 'trocr_similarity', 'trocr_exact_match',
            'ensemble_text', 'ensemble_similarity', 'ensemble_exact_match',
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

        #true_data_df = true_data_df[:20]   # для тестирования можно ограничить выборку

        for _, row in true_data_df.iterrows():
            image_path = IMAGES_DIR / row['image']
            if not image_path.exists():
                print(f"Предупреждение: изображение не найдено - {image_path}")
                continue

            try:
                # start -------------
                # # Подготовка эталонных данных
                # true_data = {
                #     'CardHolder': str(row.get('CardHolder', '')).strip(),
                #     'CardNumber': str(row.get('CardNumber', '')).strip(),
                #     'DateExpired': str(row.get('DateExpired', '')).strip()
                # }
                # +++++++++++++
                # Собираем только существующие классы
                true_data = {}
                for field in ['CardHolder', 'CardNumber', 'DateExpired']:
                    if field in row and pd.notna(row[field]) and str(row[field]).strip():
                        true_data[field] = str(row[field]).strip()
                    else:
                        pass
                # Пропускаем изображения без аннотированных классов
                if not true_data:
                    print(f"Предупреждение: нет аннотированных классов для {image_path}")
                    continue
                # end ++++++++++++++++

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

    except Exception as e:
        print(f"Критическая ошибка в main(): {str(e)}")

if __name__ == "__main__":
    main()
