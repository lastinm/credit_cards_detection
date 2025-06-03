import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from scipy.stats import mannwhitneyu, ttest_ind
from matplotlib.lines import Line2D

# Конфигурация
INPUT_CSV = 'results.csv'
BOXPLOT_FILE = 'CardHolder_comparison_boxplot'

def generate_comparison_boxplot(df):
    """Генерирует сравнение двух фреймворков для номера карты"""
    # Фильтруем только записи для CardHolder
    cardholder_data = df[df['field_type'] == 'CardHolder'].copy()
    
    if len(cardholder_data) == 0:
        raise ValueError("Нет данных для field_type = 'CardHolder'")
    
    # Подготовка данных
    frameworks = {
        'TrOCR': pd.to_numeric(cardholder_data['trocr_similarity']),
        'PaddleOCR': pd.to_numeric(cardholder_data['paddleocr_similarity']) 
    }
    
    # Удаление NaN значений
    frameworks = {k: v.dropna() for k, v in frameworks.items()}
    
    # Статистические тесты
    t_stat, t_p = ttest_ind(frameworks['PaddleOCR'], frameworks['TrOCR'])
    u_stat, u_p = mannwhitneyu(frameworks['PaddleOCR'], frameworks['TrOCR'])
    
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
    plt.title('Сравнение качества распознавания держателя карты\n', 
             fontsize=18, pad=20)
    plt.ylabel('Метрика схожести (0-1)', fontsize=16)
    plt.xlabel('Фреймворк', fontsize=16)
    plt.ylim(0, 1.05)
    plt.yticks(np.arange(0, 1.1, 0.1))

    # Увеличение шрифта делений на осях
    plt.rc('xtick', labelsize=20)  # Размер шрифта для делений оси X
    plt.rc('ytick', labelsize=20)  # Размер шрифта для делений оси Y

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
        f"  Разница средних: {frameworks['PaddleOCR'].mean()-frameworks['TrOCR'].mean():.2f}"
    )
    
    # Позиционирование текста по центру справа от графиков
    plt.text(2.7, 0.5,  # x=2.7 (правее boxplot), y=0.5 (по центру)
            stats_text, 
            fontsize=14, 
            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'),
            verticalalignment='center')  # Выравнивание по вертикали по центру
    
    # Добавление звездочек значимости (обновленные координаты)
    if u_p < 0.05:
        y_pos = max(frameworks['PaddleOCR'].max(), frameworks['TrOCR'].max()) + 0.07
        plt.plot([1, 2], [y_pos, y_pos], 'k-', lw=1.5)
        plt.text(1.5, y_pos+0.02, 
                '***' if u_p < 0.001 else '**' if u_p < 0.01 else '*',
                ha='center', fontsize=16)
    
    # Добавление легенды
    legend_elements = [
        Line2D([0], [0], color=colors[0], lw=12, label='TrOCR'), 
        Line2D([0], [0], color=colors[1], lw=12, label='PaddleOCR'),
        Line2D([0], [0], marker='D', color='w', label='Среднее',
               markerfacecolor='black', markersize=12),
        Line2D([0], [0], marker='o', color='w', label='Выбросы',
               markerfacecolor='gray', markersize=12)
    ]
    plt.legend(handles=legend_elements, loc='lower right', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(BOXPLOT_FILE, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Сравнительный график сохранён в: {BOXPLOT_FILE}")
    print("\nСтатистический анализ:")
    print(stats_text)


def generate_comparison_boxplot_ver2(df):
    """Генерирует сравнение двух фреймворков для номера карты"""
    # Фильтруем только записи для CardHolder
    cardholder_data = df[df['field_type'] == 'CardHolder'].copy()
    
    if len(cardholder_data) == 0:
        raise ValueError("Нет данных для field_type = 'CardHolder'")
    
    # Подготовка данных
    frameworks = {
        'TrOCR': pd.to_numeric(cardholder_data['trocr_similarity']),
        'PaddleOCR': pd.to_numeric(cardholder_data['paddleocr_similarity']) 
    }
    
    # Удаление NaN значений
    frameworks = {k: v.dropna() for k, v in frameworks.items()}
    
    # Статистические тесты
    t_stat, t_p = ttest_ind(frameworks['PaddleOCR'], frameworks['TrOCR'])
    u_stat, u_p = mannwhitneyu(frameworks['PaddleOCR'], frameworks['TrOCR'])
    
    # Создание фигуры с явным указанием осей
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Создание boxplot
    bp = ax.boxplot(
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
    
    # Добавление точек данных
    for i, (name, values) in enumerate(frameworks.items(), 1):
        x = np.random.normal(i, 0.1, size=len(values))
        ax.plot(x, values, 'o', color=colors[i-1], alpha=0.4, markersize=6)
    
    # НАСТРОЙКА ШРИФТА ДЕЛЕНИЙ ОСЕЙ (главное изменение)
    ax.tick_params(axis='both', which='major', labelsize=20, pad=8)
    
    # Настройки графика
    ax.set_title('Сравнение схожести распознавания держателя карты\n', 
                fontsize=24, pad=20)
    ax.set_ylabel('Метрика схожести (0-1)', fontsize=22)
    ax.set_xlabel('Фреймворк', fontsize=22)
    ax.set_ylim(0, 1.05)
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.grid(True, linestyle='--', alpha=0.3, axis='y')
    
    # Добавление легенды
    legend_elements = [
        Line2D([0], [0], color=colors[0], lw=12, label='TrOCR'), 
        Line2D([0], [0], color=colors[1], lw=12, label='PaddleOCR'),
        Line2D([0], [0], marker='D', color='w', label='Среднее',
               markerfacecolor='black', markersize=12),
        Line2D([0], [0], marker='o', color='w', label='Выбросы',
               markerfacecolor='gray', markersize=12)
    ]
    plt.legend(handles=legend_elements, loc='lower right', fontsize=18)
    
    plt.tight_layout()
    plt.savefig(f"{BOXPLOT_FILE}_ver2.png", dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    try:
        # Загрузка данных
        df = pd.read_csv(INPUT_CSV)
        
        # Генерация графика
        generate_comparison_boxplot(df)
        generate_comparison_boxplot_ver2(df)

    except Exception as e:
        print(f"Ошибка: {str(e)}")
