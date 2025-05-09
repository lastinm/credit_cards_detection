# Расчет интервальных оценок точности распознавания с доверительными интервалами (уровень значимости 0.05)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

# Конфигурация
INPUT_CSV = 'results.csv'
OUTPUT_FILE = 'ocr_ci_metrics.txt'
ALPHA = 0.05


def calculate_ci(df, metric_col, group_cols=None):
    """Вычисляет доверительные интервалы"""
    results = []
    
    if not group_cols:
        group_df = df
        group_name = 'all'
        
        n = len(group_df)
        successes = group_df[metric_col].sum()
        p = successes / n
        
        z = stats.norm.ppf(1 - ALPHA/2)
        denominator = 1 + z**2 / n
        centre = (p + z**2 / (2 * n)) / denominator
        margin = z * np.sqrt(p*(1-p)/n + z**2/(4*n**2)) / denominator
        
        results.append({
            'group': group_name,
            'n': n,
            'accuracy': p,
            'ci_lower': max(0, centre - margin),
            'ci_upper': min(1, centre + margin)
        })
    else:
        for group, group_df in df.groupby(group_cols):
            n = len(group_df)
            successes = group_df[metric_col].sum()
            p = successes / n
            
            z = stats.norm.ppf(1 - ALPHA/2)
            denominator = 1 + z**2 / n
            centre = (p + z**2 / (2 * n)) / denominator
            margin = z * np.sqrt(p*(1-p)/n + z**2/(4*n**2)) / denominator
            
            results.append({
                'group': str(group),
                'n': n,
                'accuracy': p,
                'ci_lower': max(0, centre - margin),
                'ci_upper': min(1, centre + margin)
            })
    
    return pd.DataFrame(results)

def save_ci_metrics():
    """Сохраняет метрики и генерирует графики"""
    df = pd.read_csv(INPUT_CSV)
    
    with open(OUTPUT_FILE, 'w') as f:
        f.write("=== Доверительные интервалы точности (95% CI) ===\n\n")
               
        for framework in ['easyocr', 'trocr']:
            f.write(f"Фреймворк: {framework.upper()}\n")
            
            framework_df = calculate_ci(df, f'{framework}_exact_match')
            f.write(f"Общая точность: {framework_df['accuracy'].iloc[0]:.3f} ")
            f.write(f"[{framework_df['ci_lower'].iloc[0]:.3f}, {framework_df['ci_upper'].iloc[0]:.3f}]\n")
            f.write(f"Объём выборки: {framework_df['n'].iloc[0]}\n\n")
            
            class_ci = calculate_ci(df, f'{framework}_exact_match', ['field_type'])
            for _, row in class_ci.iterrows():
                f.write(f"  Класс {row['group']}:\n")
                f.write(f"    Точность: {row['accuracy']:.3f} ")
                f.write(f"[{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]\n")
                f.write(f"    Объём выборки: {row['n']}\n")
            
            f.write("\n" + "="*50 + "\n\n")

if __name__ == "__main__":
    print("Расчёт метрик...")
    save_ci_metrics()
    
