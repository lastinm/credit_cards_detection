import os
import pandas as pd
import shutil
from pathlib import Path

# Конфигурация
INPUT_CSV = 'results.csv'
IMAGES_DIR = Path('/home/lastinm/PROJECTS/credit_cards_detection/dataset/coco/valid/images')
OUTPUT_DIR = Path('top_false')
OUTPUT_DIR.mkdir(exist_ok=True)

# Загрузка данных
df = pd.read_csv(INPUT_CSV)

def save_error_statistics():
    """Сохраняет общую статистику по ошибкам в файл"""
    total_records = len(df)
    
    with open(OUTPUT_DIR / 'error_statistics.txt', 'w') as f:
        f.write("=== Общая статистика по ошибкам распознавания ===\n\n")
        f.write(f"Всего записей в данных: {total_records}\n\n")
        
        for framework in ['easyocr', 'trocr']:
            errors = len(df[df[f'{framework}_exact_match'] == False])
            error_percent = errors / total_records * 100
            
            f.write(f"{framework.upper()}:\n")
            f.write(f"  Всего ошибочных распознаваний: {errors}\n")
            f.write(f"  Процент ошибок: {error_percent:.2f}%\n")
            f.write(f"  Точность распознавания: {100 - error_percent:.2f}%\n\n")

# 1. Топ ошибок для каждого класса
def save_top_errors():
    top_errors = {}
    
    for field_type in df['field_type'].unique():
        errors_easyocr = df[(df['field_type'] == field_type) & 
                          (df['easyocr_exact_match'] == False)][['true_text', 'easyocr_text']].drop_duplicates()
        
        errors_trocr = df[(df['field_type'] == field_type) & 
                        (df['trocr_exact_match'] == False)][['true_text', 'trocr_text']].drop_duplicates()
        
        top_errors[field_type] = {
            'EasyOCR': errors_easyocr.head(10),
            'TrOCR': errors_trocr.head(10)
        }
    
    with open(OUTPUT_DIR / 'top10_errors.txt', 'w') as f:
        for field_type, errors in top_errors.items():
            f.write(f"\n=== Типичные ошибки для {field_type} ===\n")
            for framework, error_df in errors.items():
                f.write(f"\nФреймворк: {framework}\n")
                for _, row in error_df.iterrows():
                    true = row['true_text']
                    pred = row['easyocr_text'] if framework == 'EasyOCR' else row['trocr_text']
                    f.write(f"Эталон: '{true}'\nРаспознано: '{pred}'\n{'='*50}\n")

# 2. Анализ условно точных совпадений
def analyze_conditional_accuracy():
    df['true_norm'] = df['true_text'].str.replace(' ', '').str.upper()
    df['easyocr_norm'] = df['easyocr_text'].str.replace(' ', '').str.upper()
    df['trocr_norm'] = df['trocr_text'].str.replace(' ', '').str.upper()
    
    cond_accurate_easyocr = df[(df['easyocr_exact_match'] == True) & 
                             (df['true_text'] != df['easyocr_text'])]
    cond_accurate_trocr = df[(df['trocr_exact_match'] == True) & 
                           (df['true_text'] != df['trocr_text'])]
    
    total_easyocr = len(df[df['easyocr_exact_match'] == True])
    percent_easyocr = (len(cond_accurate_easyocr) / total_easyocr) * 100 if total_easyocr > 0 else 0
    
    total_trocr = len(df[df['trocr_exact_match'] == True])
    percent_trocr = (len(cond_accurate_trocr) / total_trocr) * 100 if total_trocr > 0 else 0
    
    with open(OUTPUT_DIR / 'problem_accuracy.txt', 'w') as f:
        f.write("=== EasyOCR ===\n")
        f.write(f"Процент условно точных совпадений: {percent_easyocr:.2f}%\n")
        f.write(f"Всего точных совпадений: {total_easyocr}\n")
        f.write(f"Из них условно точных: {len(cond_accurate_easyocr)}\n\n")
        
        f.write("=== TrOCR ===\n")
        f.write(f"Процент условно точных совпадений: {percent_trocr:.2f}%\n")
        f.write(f"Всего точных совпадений: {total_trocr}\n")
        f.write(f"Из них условно точных: {len(cond_accurate_trocr)}\n\n")
        
        f.write("Примеры (EasyOCR):\n")
        for _, row in cond_accurate_easyocr.head(5).iterrows():
            f.write(f"\nИзображение: {row['image']}\nТип: {row['field_type']}\n")
            f.write(f"Эталон: '{row['true_text']}'\nРаспознано: '{row['easyocr_text']}'\n")
        
        f.write("\nПримеры (TrOCR):\n")
        for _, row in cond_accurate_trocr.head(5).iterrows():
            f.write(f"\nИзображение: {row['image']}\nТип: {row['field_type']}\n")
            f.write(f"Эталон: '{row['true_text']}'\nРаспознано: '{row['trocr_text']}'\n")

# Выполняем анализ
print("Анализ данных...")
save_error_statistics()  # Сохраняем статистику ошибок
save_top_errors()
analyze_conditional_accuracy()
print(f"Анализ завершен. Результаты сохранены в папке {OUTPUT_DIR}")
