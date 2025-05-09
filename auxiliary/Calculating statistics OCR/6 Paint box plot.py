import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Конфигурация
INPUT_CSV = 'results.csv'
BOXPLOT_FILE = 'cardnumber_comparison_boxplot.png'

def generate_comparison_boxplot(df):
    """Генерирует сравнение двух фреймворков для номера карты"""
    # Фильтруем только записи для CardNumber
    cardnumber_data = df[df['field_type'] == 'CardNumber'].copy()
    
    if len(cardnumber_data) == 0:
        raise ValueError("Нет данных для field_type = 'CardNumber'")
    
    # Подготовка данных
    frameworks = {
        'EasyOCR': pd.to_numeric(cardnumber_data['easyocr_similarity'], errors='coerce'),
        'TrOCR': pd.to_numeric(cardnumber_data['trocr_similarity'], errors='coerce')
    }
    
    # Создание фигуры
    plt.figure(figsize=(12, 7))
    
    # Создание boxplot
    bp = plt.boxplot(
        frameworks.values(),
        labels=frameworks.keys(),
        patch_artist=True,
        widths=0.6,
        showmeans=True,
        meanprops={'marker':'D', 'markerfacecolor':'white', 'markeredgecolor':'black', 'markersize':'8'}
    )
    
    # Настройка цветов
    colors = ['#4C72B0', '#DD8452']
    for patch, color in zip(bp['boxes'], colors):
        patch.set(facecolor=color, alpha=0.8)
    
    # Добавление точек данных
    for i, (name, values) in enumerate(frameworks.items(), 1):
        x = np.random.normal(i, 0.08, size=len(values))
        plt.plot(x, values, 'k.', alpha=0.4)
    
    # Настройки графика
    plt.title('Сравнение качества распознавания номера карты', fontsize=16, pad=20)
    plt.ylabel('Метрика схожести (0-1)', fontsize=12)
    plt.xlabel('Фреймворк', fontsize=12)
    plt.ylim(0, 1.05)
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.grid(True, linestyle='--', alpha=0.3, axis='y')
    
    # Добавление статистики
    stats_text = ""
    for name, values in frameworks.items():
        stats_text += (
            f"{name}:\n"
            f"  Медиана: {values.median():.2f}\n"
            f"  Среднее: {values.mean():.2f}\n"
            f"  Q1-Q3: {values.quantile(0.25):.2f}-{values.quantile(0.75):.2f}\n"
            f"  n = {len(values)}\n\n"
        )
    
    plt.text(2.7, 0.3, stats_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    
    # Добавление легенды
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=colors[0], lw=8, label='EasyOCR'),
        Line2D([0], [0], color=colors[1], lw=8, label='TrOCR'),
        Line2D([0], [0], marker='D', color='w', label='Среднее',
               markerfacecolor='black', markersize=8)
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(BOXPLOT_FILE, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Сравнительный график сохранён в: {BOXPLOT_FILE}")
    print("Статистика по фреймворкам:")
    print(stats_text)

if __name__ == "__main__":
    try:
        # Загрузка данных
        df = pd.read_csv(INPUT_CSV)
        
        # Генерация графика
        generate_comparison_boxplot(df)
        
    except Exception as e:
        print(f"Ошибка: {str(e)}")
        