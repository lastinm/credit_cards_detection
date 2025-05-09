# Топ-10 ошибочных символов для каждого фреймворка
# Результат в файл top_false/top10_false_simbols.txt

import pandas as pd
from collections import Counter
from pathlib import Path

def analyze_false_symbols():
    # Создаем папку для результатов
    output_dir = Path('top_false')
    output_dir.mkdir(exist_ok=True)
    
    # Загрузка данных
    df = pd.read_csv('results.csv')
    
    # Функция для сравнения символов
    def get_symbol_errors(true_text, pred_text):
        errors = []
        true_text = str(true_text)
        pred_text = str(pred_text)
        min_len = min(len(true_text), len(pred_text))
        
        for i in range(min_len):
            if true_text[i] != pred_text[i]:
                errors.append((true_text[i], pred_text[i]))
        
        # Обрабатываем остаток более длинной строки
        if len(true_text) > min_len:
            for c in true_text[min_len:]:
                errors.append((c, None))
        elif len(pred_text) > min_len:
            for c in pred_text[min_len:]:
                errors.append((None, c))
                
        return errors
    
    # Анализ для каждого фреймворка
    frameworks = ['easyocr', 'trocr']
    results = {}
    
    for framework in frameworks:
        # Фильтруем ошибочные распознавания
        false_matches = df[df[f'{framework}_exact_match'] == False]
        all_errors = []
        
        # Собираем все ошибки символов
        for _, row in false_matches.iterrows():
            errors = get_symbol_errors(row['true_text'], row[f'{framework}_text'])
            all_errors.extend(errors)
        
        # Считаем частоту ошибок
        error_counter = Counter(all_errors)
        top_errors = error_counter.most_common(10)
        
        # Сохраняем результаты
        results[framework] = top_errors
    
    # Записываем результаты в файл
    with open(output_dir / 'top10_false_simbols.txt', 'w') as f:
        f.write("=== Топ-10 ошибочных символов ===\n\n")
        
        for framework, top_errors in results.items():
            f.write(f"Фреймворк: {framework.upper()}\n")
            f.write("Рейтинг | Эталонный | Распознанный | Количество\n")
            f.write("----------------------------------------------\n")
            
            for rank, ((true_char, pred_char), count) in enumerate(top_errors, 1):
                true_disp = 'None' if true_char is None else repr(true_char)
                pred_disp = 'None' if pred_char is None else repr(pred_char)
                f.write(f"{rank:6} | {true_disp:^9} | {pred_disp:^11} | {count}\n")
            
            f.write("\n")

if __name__ == "__main__":
    print("Анализ ошибочных символов...")
    analyze_false_symbols()
    print("Анализ завершен. Результаты сохранены в top_false/top10_false_simbols.txt")
    