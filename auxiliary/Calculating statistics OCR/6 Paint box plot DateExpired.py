import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from scipy.stats import mannwhitneyu, ttest_ind

# Конфигурация
INPUT_CSV = 'results.csv'
BOXPLOT_FILE = 'DateExpired_comparison_boxplot.png'

def generate_comparison_boxplot(df):
    """Генерирует сравнение двух фреймворков для номера карты"""
    # Фильтруем только записи для DateExpired
    cardnumber_data = df[df['field_type'] == 'DateExpired'].copy()
    
    if len(cardnumber_data) == 0:
        raise ValueError("Нет данных для field_type = 'DateExpired'")
    
    # Подготовка данных
    frameworks = {
        'EasyOCR': pd.to_numeric(cardnumber_data['easyocr_similarity']), 
        'TrOCR': pd.to_numeric(cardnumber_data['trocr_similarity'])
    }
    
    # Удаление NaN значений
    frameworks = {k: v.dropna() for k, v in frameworks.items()}
    
    # Статистические тесты
    t_stat, t_p = ttest_ind(frameworks['EasyOCR'], frameworks['TrOCR'])
    u_stat, u_p = mannwhitneyu(frameworks['EasyOCR'], frameworks['TrOCR'])
    
    # Создание фигуры
    plt.figure(figsize=(14, 8))
    
    # Создание boxplot с улучшенными настройками
    bp = plt.boxplot(
        frameworks.values(),
        labels=frameworks.keys(),
        patch_artist=True,
        widths=0.6,
        showmeans=True,
        meanprops={'marker':'D', 'markerfacecolor':'white', 
                  'markeredgecolor':'black', 'markersize':'10'},
        whiskerprops={'linestyle':'--', 'alpha':0.7},
        flierprops={'marker':'o', 'markersize':6, 'alpha':0.5}
    )
    
    # Настройка цветов
    colors = ['#4C72B0', '#DD8452']
    for patch, color in zip(bp['boxes'], colors):
        patch.set(facecolor=color, alpha=0.8, linewidth=2)
    
    # Добавление точек данных с джиттером
    for i, (name, values) in enumerate(frameworks.items(), 1):
        x = np.random.normal(i, 0.1, size=len(values))
        plt.plot(x, values, 'o', color=colors[i-1], alpha=0.4, markersize=6)
    
    # Настройки графика
    plt.title('Сравнение качества распознавания номера карты\n', 
             fontsize=16, pad=20)
    plt.ylabel('Метрика схожести (0-1)', fontsize=12)
    plt.xlabel('Фреймворк', fontsize=12)
    plt.ylim(0, 1.05)
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.grid(True, linestyle='--', alpha=0.3, axis='y')
    
    # Добавление расширенной статистики
    stats_text = ""
    for name, values in frameworks.items():
        ci = stats.t.interval(0.95, len(values)-1, 
                            loc=values.mean(), 
                            scale=stats.sem(values))
        stats_text += (
            f"{name}:\n"
            f"  Медиана: {values.median():.3f}\n"
            f"  Среднее: {values.mean():.3f}\n"
            f"    95% ДИ: [{ci[0]:.3f}, {ci[1]:.3f}]\n"
            f"  Станд.откл.: {values.std():.3f}\n"
            f"  IQR: [{values.quantile(0.25):.3f}, {values.quantile(0.75):.3f}]\n"
            f"  n = {len(values)}\n\n"
        )
    
    # Добавление результатов тестов
    stats_text += (
        f"Статистические тесты:\n"
        f"  t-тест: p = {t_p:.4f}\n"
        f"  U-тест: p = {u_p:.4f}\n"
        f"  Разница средних: {frameworks['TrOCR'].mean()-frameworks['EasyOCR'].mean():.3f}"
    )
    
    # Позиционирование текста по центру справа от графиков
    plt.text(2.7, 0.5,  # x=2.7 (правее boxplot), y=0.5 (по центру)
            stats_text, 
            fontsize=10, 
            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'),
            verticalalignment='center')  # Выравнивание по вертикали по центру
    
    # Добавление звездочек значимости (обновленные координаты)
    if u_p < 0.05:
        y_pos = max(frameworks['EasyOCR'].max(), frameworks['TrOCR'].max()) + 0.07
        plt.plot([1, 2], [y_pos, y_pos], 'k-', lw=1.5)
        plt.text(1.5, y_pos+0.02, 
                '***' if u_p < 0.001 else '**' if u_p < 0.01 else '*',
                ha='center', fontsize=16)
    
    # Добавление легенды
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=colors[0], lw=8, label='EasyOCR'),
        Line2D([0], [0], color=colors[1], lw=8, label='TrOCR'),
        Line2D([0], [0], marker='D', color='w', label='Среднее',
               markerfacecolor='black', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Выбросы',
               markerfacecolor='gray', markersize=8)
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(BOXPLOT_FILE, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Сравнительный график сохранён в: {BOXPLOT_FILE}")
    print("\nСтатистический анализ:")
    print(stats_text)

if __name__ == "__main__":
    try:
        # Загрузка данных
        df = pd.read_csv(INPUT_CSV)
        
        # Генерация графика
        generate_comparison_boxplot(df)
        
    except Exception as e:
        print(f"Ошибка: {str(e)}")
