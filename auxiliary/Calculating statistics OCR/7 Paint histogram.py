import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Конфигурация
INPUT_CSV = 'results.csv'
OUTPUT_FILE = 'ocr_frameworks_comparison.png'

def plot_frameworks_comparison(df):
    """Сравнение фреймворков через гистограмму и boxplot"""
    # Фильтруем только номера карт
    df = df[df['field_type'] == 'CardNumber'].copy()
    
    # Преобразуем булевы значения в числовые (1=True, 0=False)
    df['easyocr'] = df['easyocr_exact_match'].astype(int)
    df['trocr'] = df['trocr_exact_match'].astype(int)
    
    # Создаем фигуру с двумя субплoтами
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), 
                                  gridspec_kw={'height_ratios': [2, 1]})
    
    # Гистограмма (верхний график)
    bins = np.linspace(0, 1, 11)  # 10 bins от 0 до 1
    ax1.hist([df['easyocr'], df['trocr']], 
             bins=bins,
             label=['EasyOCR', 'TrOCR'],
             color=['#1f77b4', '#ff7f0e'],
             alpha=0.7,
             edgecolor='black')
    
    ax1.set_title('Распределение точности распознавания номера карты', fontsize=14)
    ax1.set_ylabel('Количество наблюдений', fontsize=12)
    ax1.set_xlabel('Точность (1 = верно, 0 = неверно)', fontsize=12)
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.3)
    
    # Ящики с усами (нижний график)
    boxprops = {'facecolor': 'lightgray', 'alpha': 0.7}
    ax2.boxplot([df['easyocr'], df['trocr']],
                labels=['EasyOCR', 'TrOCR'],
                patch_artist=True,
                widths=0.6,
                showmeans=True,
                meanprops={'marker': 'o', 'markerfacecolor': 'white', 'markeredgecolor': 'black'})
    
    ax2.set_ylabel('Точность', fontsize=12)
    ax2.set_ylim(-0.1, 1.1)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['0 (False)', '1 (True)'])
    ax2.grid(True, linestyle='--', alpha=0.3)
    
    # Общие настройки
    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"График сравнения сохранен в: {OUTPUT_FILE}")

if __name__ == "__main__":
    try:
        # Загрузка данных
        df = pd.read_csv(INPUT_CSV)
        
        # Проверка данных
        if 'field_type' not in df.columns:
            raise ValueError("Отсутствует колонка 'field_type'")
        if not all(col in df.columns for col in ['easyocr_exact_match', 'trocr_exact_match']):
            raise ValueError("Отсутствуют колонки с метриками фреймворков")
        
        # Генерация графиков
        plot_frameworks_comparison(df)
        
    except Exception as e:
        print(f"Ошибка: {str(e)}")
        