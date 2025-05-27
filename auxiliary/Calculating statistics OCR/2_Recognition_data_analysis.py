# Анализируем данные о распознании из файла results.csv
# 1. Высчитываем общую статистику по ошибкам распознавания
# 2. Топ-10 ошибок для каждого класса. Пишем в файл top_fails/top10_errors.txt
# 3. Анализ условно точных совпадений. Пишем в файл top_fails/problem_accuracy.txt
# 4. Сохранение изображений для Топ-10 ошибочных распознаваний по каждому фреймворку и классу. Путь сохранения top_fails/

import os
import pandas as pd
import shutil
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
# Добавляем импорт для построения графиков
import matplotlib.pyplot as plt


# Конфигурация
ROOT_DIR = rf'/home/lastinm/PROJECTS/credit_cards_detection'
INPUT_CSV = fr"{ROOT_DIR}/auxiliary/Calculating statistics OCR/results.csv"
IMAGES_DIR = Path(rf"{ROOT_DIR}/dataset/ocr val")
OUTPUT_DIR = Path('top_fails')
OUTPUT_DIR.mkdir(exist_ok=True)
FRAMEWORKS = ['easyocr', 'trocr', 'paddleocr', 'ensemble']

# Параметры отрисовки
BOX_COLOR = (255, 0, 0)  # Красный
TEXT_COLOR = (255, 255, 255)  # Белый
BACKGROUND_COLOR = (0, 0, 0)  # Черный
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"  # Путь к шрифту
FONT_SIZE = 14

# Загрузка данных
df = pd.read_csv(INPUT_CSV)

def draw_annotation(image_path, bbox, true_text, pred_text, similarity, output_path):
    """Отрисовывает bounding box и текст на изображении"""
    try:
        # Загрузка изображения
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Не удалось загрузить изображение: {image_path}")
            return
        
        # Конвертация в PIL для работы с текстом
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        
        try:
            font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
        except:
            font = ImageFont.load_default()
        
        # Парсинг координат bbox
        xmin, ymin, xmax, ymax = map(float, bbox.split(','))
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
        
        # Отрисовка bounding box
        draw.rectangle([xmin, ymin, xmax, ymax], outline=BOX_COLOR, width=2)
        
        # Подготовка текста
        text_lines = [
            f"Эталон: {true_text}",
            f"Распознано: {pred_text}",
            f"Схожесть: {similarity:.2f}"
        ]
        
        # Расчет положения текста (слева внизу от bbox)
        text_x = xmin
        text_y = ymax + 5
        
        # Отрисовка текста с фоном
        for i, line in enumerate(text_lines):
            # Получаем размеры текста (новый способ для Pillow >= 9.0.0)
            left, top, right, bottom = draw.textbbox((0, 0), line, font=font)
            text_width = right - left
            text_height = bottom - top
            
            # Отрисовка фона
            draw.rectangle(
                [text_x, text_y + i*(text_height+5), 
                 text_x + text_width, text_y + (i+1)*(text_height+5)],
                fill=BACKGROUND_COLOR
            )
            
            # Отрисовка текста
            draw.text(
                (text_x, text_y + i*(text_height+5)),
                line,
                fill=TEXT_COLOR,
                font=font
            )
        
        # Сохранение изображения
        img_pil.save(output_path)
        
    except Exception as e:
        print(f"Ошибка при обработке изображения {image_path}: {str(e)}")

def save_worst_images():
    """Сохраняет топ-10 худших изображений с аннотациями"""
    
    for framework in FRAMEWORKS:
        framework_dir = OUTPUT_DIR / framework
        framework_dir.mkdir(exist_ok=True)
        
        for field_type in df['field_type'].unique():
            class_dir = framework_dir / field_type
            class_dir.mkdir(exist_ok=True)
            
            # Фильтруем ошибки для данного фреймворка и класса
            errors = df[(df['field_type'] == field_type) & 
                      (df[f'{framework}_exact_match'] == False)]
            print(f"\n{framework} - {field_type}: Всего ошибок - {len(errors)}")

            # Сортируем по худшей схожести
            worst_errors = errors.sort_values(by=f'{framework}_similarity').head(10)
            
            # Обрабатываем каждое изображение
            for _, row in worst_errors.iterrows():
                src = IMAGES_DIR / row['image']
                dst = class_dir / row['image']
                
                if src.exists():
                    draw_annotation(
                        image_path=src,
                        bbox=row['bbox'],
                        true_text=row['true_text'],
                        pred_text=row[f'{framework}_text'],
                        similarity=row[f'{framework}_similarity'],
                        output_path=dst
                    )
                else:
                    print(f"Файл не найден: {src}")

def save_error_statistics():
    """Сохраняет общую статистику по ошибкам в файл"""
    total_records = len(df)
    
    with open(OUTPUT_DIR / 'error_statistics.txt', 'w') as f:
        f.write("=== Общая статистика по ошибкам распознавания ===\n\n")
        f.write(f"Всего записей в данных: {total_records}\n\n")
        
        for framework in FRAMEWORKS:
            errors = len(df[df[f'{framework}_exact_match'] == False])
            error_percent = errors / total_records * 100
            
            f.write(f"{framework.upper()}:\n")
            f.write(f"  Всего ошибочных распознаваний: {errors}\n")
            f.write(f"  Процент ошибок: {error_percent:.2f}%\n")
            f.write(f"  Точность распознавания: {100 - error_percent:.2f}%\n\n")

def save_top_errors():
    """Сохраняет топ-10 ошибок для каждого класса"""
    top_errors = {}
    
    for field_type in df['field_type'].unique():
        errors_easyocr = df[(df['field_type'] == field_type) & 
                          (df['easyocr_exact_match'] == False)][['true_text', 'easyocr_text']].drop_duplicates()
        
        errors_trocr = df[(df['field_type'] == field_type) & 
                        (df['trocr_exact_match'] == False)][['true_text', 'trocr_text']].drop_duplicates()
        
        errors_paddleocr = df[(df['field_type'] == field_type) & 
                        (df['paddleocr_exact_match'] == False)][['true_text', 'paddleocr_text']].drop_duplicates()
        
        errors_ensemble = df[(df['field_type'] == field_type) & 
                (df['ensemble_exact_match'] == False)][['true_text', 'ensemble_text']].drop_duplicates()
        
        top_errors[field_type] = {
            'EasyOCR': errors_easyocr.head(10),
            'TrOCR': errors_trocr.head(10),
            'PaddleOCR': errors_paddleocr.head(10),
            'Ensemble': errors_ensemble.head(10)

        }
    
    with open(OUTPUT_DIR / 'top10_errors.txt', 'w') as f:
        for field_type, errors in top_errors.items():
            f.write(f"\n=== Типичные ошибки для {field_type} ===\n")
            for framework, error_df in errors.items():
                f.write(f"\nФреймворк: {framework}\n")
                for _, row in error_df.iterrows():
                    true = row['true_text']
                    if framework == 'EasyOCR':
                        pred = row['easyocr_text']
                    elif framework == 'TrOCR':
                        pred = row['trocr_text']
                    elif framework == 'PaddleOCR':
                        pred = row['paddleocr_text']
                    else:
                        pred = row['ensemble_text']
                      
                    f.write(f"Эталон: '{true}'\nРаспознано: '{pred}'\n{'='*50}\n")

def analyze_conditional_accuracy():
    """Анализирует условно точные совпадения"""
    df['true_norm'] = df['true_text'].str.replace(' ', '').str.upper()
    df['easyocr_norm'] = df['easyocr_text'].str.replace(' ', '').str.upper()
    df['trocr_norm'] = df['trocr_text'].str.replace(' ', '').str.upper()
    df['paddleocr_norm'] = df['paddleocr_text'].str.replace(' ', '').str.upper()
    df['ensemble_norm'] = df['ensemble_text'].str.replace(' ', '').str.upper()      
    cond_accurate_easyocr = df[(df['easyocr_exact_match'] == True) & 
                             (df['true_text'] != df['easyocr_text'])]
    cond_accurate_trocr = df[(df['trocr_exact_match'] == True) & 
                           (df['true_text'] != df['trocr_text'])]
    cond_accurate_paddleocr = df[(df['paddleocr_exact_match'] == True) & 
                           (df['true_text'] != df['paddleocr_text'])]
    cond_accurate_ensemble = df[(df['ensemble_exact_match'] == True) & 
                           (df['true_text'] != df['ensemble_text'])]    
    
    total_easyocr = len(df[df['easyocr_exact_match'] == True])
    percent_easyocr = (len(cond_accurate_easyocr) / total_easyocr) * 100 if total_easyocr > 0 else 0
    
    total_trocr = len(df[df['trocr_exact_match'] == True])
    percent_trocr = (len(cond_accurate_trocr) / total_trocr) * 100 if total_trocr > 0 else 0

    total_paddleocr = len(df[df['paddleocr_exact_match'] == True])
    percent_paddleocr = (len(cond_accurate_paddleocr) / total_paddleocr) * 100 if total_paddleocr > 0 else 0

    total_ensemble = len(df[df['ensemble_exact_match'] == True])
    percent_ensemble = (len(cond_accurate_ensemble) / total_ensemble) * 100 if total_ensemble > 0 else 0   
    with open(OUTPUT_DIR / 'problem_accuracy.txt', 'w') as f:
        f.write("=== EasyOCR ===\n")
        f.write(f"Процент условно точных совпадений: {percent_easyocr:.2f}%\n")
        f.write(f"Всего точных совпадений: {total_easyocr}\n")
        f.write(f"Из них условно точных: {len(cond_accurate_easyocr)}\n\n")
        
        f.write("=== TrOCR ===\n")
        f.write(f"Процент условно точных совпадений: {percent_trocr:.2f}%\n")
        f.write(f"Всего точных совпадений: {total_trocr}\n")
        f.write(f"Из них условно точных: {len(cond_accurate_trocr)}\n\n")

        f.write("=== PaddleOCR ===\n")
        f.write(f"Процент условно точных совпадений: {percent_paddleocr:.2f}%\n")
        f.write(f"Всего точных совпадений: {total_paddleocr}\n")
        f.write(f"Из них условно точных: {len(cond_accurate_paddleocr)}\n\n")

        f.write("=== Ensemble ===\n")
        f.write(f"Процент условно точных совпадений: {percent_ensemble:.2f}%\n")
        f.write(f"Всего точных совпадений: {total_ensemble}\n")
        f.write(f"Из них условно точных: {len(cond_accurate_ensemble)}\n\n")        
        f.write("Примеры (EasyOCR):\n")

        for _, row in cond_accurate_easyocr.head(5).iterrows():
            f.write(f"\nИзображение: {row['image']}\nТип: {row['field_type']}\n")
            f.write(f"Эталон: '{row['true_text']}'\nРаспознано: '{row['easyocr_text']}'\n")
        
        f.write("\nПримеры (TrOCR):\n")
        for _, row in cond_accurate_trocr.head(5).iterrows():
            f.write(f"\nИзображение: {row['image']}\nТип: {row['field_type']}\n")
            f.write(f"Эталон: '{row['true_text']}'\nРаспознано: '{row['trocr_text']}'\n")

        f.write("\nПримеры (PaddleOCR):\n")
        for _, row in cond_accurate_paddleocr.head(5).iterrows():
            f.write(f"\nИзображение: {row['image']}\nТип: {row['field_type']}\n")
            f.write(f"Эталон: '{row['true_text']}'\nРаспознано: '{row['paddleocr_text']}'\n")

        f.write("\nПримеры (Ensemble):\n")
        for _, row in cond_accurate_ensemble.head(5).iterrows():
            f.write(f"\nИзображение: {row['image']}\nТип: {row['field_type']}\n")
            f.write(f"Эталон: '{row['true_text']}'\nРаспознано: '{row['ensemble_text']}'\n")

def plot_similarity_histogram_заслоняют():
    """Строит гистограмму распределения схожести для трех фреймворков"""
    plt.figure(figsize=(12, 6))
    
    # Данные для гистограммы
    easyocr_sim = df['easyocr_similarity']
    trocr_sim = df['trocr_similarity']
    paddleocr_sim = df['paddleocr_similarity']
    ensemble_sim = df['ensemble_similarity']
          
    # Настройки гистограммы
    bins = np.linspace(0, 1, 21)  # 20 корзин от 0 до 1
    alpha = 0.7
    
    # Построение гистограммы для EasyOCR
    plt.hist(easyocr_sim, bins=bins, alpha=alpha, color='#1f77b4', label='EasyOCR')
    
    # Построение гистограммы для TrOCR
    plt.hist(trocr_sim, bins=bins, alpha=alpha, color="#ffab0e", label='TrOCR')

    # Построение гистограммы для PadleOCR
    plt.hist(paddleocr_sim, bins=bins, alpha=alpha, color="#3eff0e", label='PaddleOCR')

    # Построение гистограммы для Ensemble
    plt.hist(ensemble_sim_sim, bins=bins, alpha=alpha, color="#ff0e0e", label='PaddleOCR')

    # Настройка графика
    plt.title('Распределение схожести распознавания текста', fontsize=14)
    plt.xlabel('Схожесть (через растояние Левенштейна)', fontsize=12)
    plt.ylabel('Количество распознаваний', fontsize=12)
    plt.legend(loc='upper left', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Сохранение графика
    output_path = OUTPUT_DIR / 'similarity_distribution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Гистограмма сохранена в {output_path}")

def plot_similarity_histogram():
    """Строит гистограмму распределения схожести для трех фреймворков"""
    plt.figure(figsize=(12, 6))
    
    # Данные для гистограммы
    easyocr_sim = df['easyocr_similarity']
    trocr_sim = df['trocr_similarity']
    paddleocr_sim = df['paddleocr_similarity']
    ensemble_sim = df['ensemble_similarity']
   
    # Настройки гистограммы
    bins = np.linspace(0, 1, 21)  # 20 корзин от 0 до 1
    alpha = 0.9
    bar_width = 0.33  # Ширина столбца
    
    # Создаем массив позиций для столбцов
    x = np.arange(len(bins)-1)
    
    # Построение гистограмм со смещением
    plt.bar(x - bar_width, np.histogram(easyocr_sim, bins=bins)[0], 
            width=bar_width, alpha=alpha, color='#1f77b4', label='EasyOCR')
    
    plt.bar(x, np.histogram(trocr_sim, bins=bins)[0], 
            width=bar_width, alpha=alpha, color='#ffab0e', label='TrOCR')
    
    plt.bar(x + bar_width, np.histogram(paddleocr_sim, bins=bins)[0], 
            width=bar_width, alpha=alpha, color='#3eff0e', label='PaddleOCR')

    plt.bar(x + bar_width, np.histogram(ensemble_sim, bins=bins)[0], 
            width=bar_width, alpha=alpha, color='#ff0e0e', label='Ensemble')
        
    # Настройка осей и подписей
    plt.xticks(x, [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(len(bins)-1)], rotation=45)
    
    # Настройка графика
    plt.title('Распределение схожести распознавания текста', fontsize=14)
    plt.xlabel('Диапазон схожести (через расстояние Левенштейна)', fontsize=12)
    plt.ylabel('Количество распознаваний', fontsize=12)
    plt.legend(loc='upper left', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # Улучшаем читаемость
    plt.tight_layout()
    
    # Сохранение графика
    output_path = OUTPUT_DIR / 'similarity_distribution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

# Выполняем анализ
print("Анализ данных...")
save_error_statistics()
save_top_errors()
analyze_conditional_accuracy()
save_worst_images()  # Сохраняем изображения с аннотациями
plot_similarity_histogram()     # вызов функции построения гистограммы
print(f"Анализ завершен. Результаты сохранены в папке {OUTPUT_DIR}")


