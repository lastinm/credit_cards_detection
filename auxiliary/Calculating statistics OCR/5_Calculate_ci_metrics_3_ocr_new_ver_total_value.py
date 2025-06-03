# Подсчитывем точечные оценки
# Word Accuracy (полное совпадение строк), Precision, Recall, F1-Score
# Сохраняем график ocr_metrics_comparison.png 

import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from statsmodels.stats.proportion import proportion_confint
from sklearn.utils import resample
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from pathlib import Path


ROOT_DIR = rf'/home/lastinm/PROJECTS/credit_cards_detection'
OUTPUT_DIR = fr"{ROOT_DIR}/auxiliary/Calculating statistics OCR/top_fails"
RESULTS_CSV = fr"{ROOT_DIR}/auxiliary/Calculating statistics OCR/results.csv"

def normalize_text_word_accuracy(text, ignore_case=True, remove_spaces=False):
    """Нормализация текста: приведение к нижнему регистру и удаление пробелов."""
    return ''.join(str(text).upper().split())

def normalize_text(text, ignore_case=True, remove_spaces=False):
    """Нормализация текста: приведение к нижнему регистру и удаление пробелов."""
    text = str(text)
    if ignore_case:
        text = text.lower()
    if remove_spaces:
        text = text.replace(" ", "")
    return text.strip()

def calculate_metrics(true_texts, pred_texts, is_char_level=False):
    """Вычисление метрик с гибридным сравнением."""
    # Word Accuracy (полное совпадение строк)
    word_accuracy = sum(
        1 for true, pred in zip(true_texts, pred_texts)
        if normalize_text_word_accuracy(true) == normalize_text_word_accuracy(pred)
    ) / len(true_texts)

    # Инициализация счетчиков
    tp = fp = fn = 0

    for true, pred in zip(true_texts, pred_texts):
        true_normalized = normalize_text(true, remove_spaces=is_char_level)
        pred_normalized = normalize_text(pred, remove_spaces=is_char_level)

        if is_char_level:
            # Символьное сравнение для CardNumber и DateExpired
            true_items = list(true_normalized)
            pred_items = list(pred_normalized)
        else:
            # Словесное сравнение для CardHolder
            true_items = true_normalized.split()
            pred_items = pred_normalized.split()

        # Считаем совпадения
        for i in range(max(len(true_items), len(pred_items))):
            true_item = true_items[i] if i < len(true_items) else None
            pred_item = pred_items[i] if i < len(pred_items) else None

            if true_item and pred_item:
                if true_item == pred_item:
                    tp += 1  # True Positive
                else:
                    fp += 1  # False Positive (лишний символ/слово)
                    fn += 1  # False Negative (пропущенный символ/слово)
            elif true_item:
                fn += 1  # Пропущенный символ/слово
            elif pred_item:
                fp += 1  # Лишний символ/слово

    # Расчёт метрик
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "Word Accuracy": round(word_accuracy, 2),
        "Precision": round(precision,2),
        "Recall": round(recall,2),
        "F1-Score": round(f1,2)
    }

def calculate_metrics_with_ci(true_texts, pred_texts, is_char_level=False, confidence=0.95, n_bootstrap=1000):
    """Расчет метрик с корректными доверительными интервалами"""
    n = len(true_texts)
    
    # Word Accuracy (биномиальный интервал)
    correct = sum(1 for true, pred in zip(true_texts, pred_texts) 
                if normalize_text_word_accuracy(true) == normalize_text_word_accuracy(pred))
    word_acc = correct / n   
    word_acc_ci = stats.binomtest(correct, n).proportion_ci(confidence_level=confidence)
    
    # Расчет TP, FP, FN
    def calculate_tp_fp_fn(true, pred):
        tp = fp = fn = 0
        true_norm = normalize_text(true, remove_spaces=is_char_level)
        pred_norm = normalize_text(pred, remove_spaces=is_char_level)
        
        if is_char_level:
            true_items = list(true_norm)
            pred_items = list(pred_norm)
        else:
            true_items = true_norm.split()
            pred_items = pred_norm.split()
        
        for i in range(max(len(true_items), len(pred_items))):
            true_item = true_items[i] if i < len(true_items) else None
            pred_item = pred_items[i] if i < len(pred_items) else None
            
            if true_item and pred_item:
                if true_item == pred_item:
                    tp += 1
                else:
                    fp += 1
                    fn += 1
            elif true_item:
                fn += 1
            elif pred_item:
                fp += 1
        return tp, fp, fn
    
    # Расчет базовых метрик
    tp, fp, fn = 0, 0, 0
    for t, p in zip(true_texts, pred_texts):
        tpfpfn = calculate_tp_fp_fn(t, p)
        tp += tpfpfn[0]
        fp += tpfpfn[1]
        fn += tpfpfn[2]
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Бутстреп для F1
    f1_scores = []
    for _ in range(n_bootstrap):
        # Генерация бутстреп-выборки
        sample_true, sample_pred = resample(true_texts, pred_texts)
        
        # Расчет метрик для выборки
        bt_tp, bt_fp, bt_fn = 0, 0, 0
        for t, p in zip(sample_true, sample_pred):
            tpfpfn = calculate_tp_fp_fn(t, p)
            bt_tp += tpfpfn[0]
            bt_fp += tpfpfn[1]
            bt_fn += tpfpfn[2]
        
        bt_precision = bt_tp / (bt_tp + bt_fp) if (bt_tp + bt_fp) > 0 else 0
        bt_recall = bt_tp / (bt_tp + bt_fn) if (bt_tp + bt_fn) > 0 else 0
        bt_f1 = 2 * (bt_precision * bt_recall) / (bt_precision + bt_recall) if (bt_precision + bt_recall) > 0 else 0
        f1_scores.append(bt_f1)
    
    f1_ci = np.percentile(f1_scores, [2.5, 97.5])
    
    # Доверительные интервалы для Precision и Recall (Wilson)
    precision_ci = proportion_confint(tp, tp+fp, alpha=1-confidence, method='wilson') if (tp+fp) > 0 else (0, 1)
    recall_ci = proportion_confint(tp, tp+fn, alpha=1-confidence, method='wilson') if (tp+fn) > 0 else (0, 1)
    
    return {
        "Word Accuracy": (round(word_acc,2), word_acc_ci),
        "Precision": (round(precision,2), precision_ci),
        "Recall": (round(recall,2), recall_ci),
        "F1-Score": (round(f1,2), f1_ci)
    }
# Анализ с доверительными интервалами
def analyze_with_scipy_ci(df, field_types, confidence=0.95):
    results = []
    
    for field, params in field_types.items():
        subset = df[df["field_type"] == field]
        true_texts = subset["true_text"].tolist()

        # KerasOCR
        kerasocr_metrics = calculate_metrics_with_ci(
            true_texts,
            subset["kerasocr_text"].tolist(),
            params["is_char_level"],
            confidence
        )

        # EasyOCR
        easyocr_metrics = calculate_metrics_with_ci(
            true_texts,
            subset["easyocr_text"].tolist(),
            params["is_char_level"],
            confidence
        )
        
        # TrOCR
        trocr_metrics = calculate_metrics_with_ci(
            true_texts,
            subset["trocr_text"].tolist(),
            params["is_char_level"],
            confidence
        )
        
        # PaddleOCR
        paddleocr_metrics = calculate_metrics_with_ci(
            true_texts,
            subset["paddleocr_text"].tolist(),
            params["is_char_level"],
            confidence
        )

        # Ensemble
        ensemble_metrics = calculate_metrics_with_ci(
            true_texts,
            subset["ensemble_text"].tolist(),
            params["is_char_level"],
            confidence
        )

        # Сохранение результатов
        for model, metrics in [("KerasOCR", kerasocr_metrics),
                                ("EasyOCR", easyocr_metrics), 
                               ("TrOCR", trocr_metrics),
                                ("PaddleOCR", paddleocr_metrics),
                                ("Ensemble", ensemble_metrics)]:
            for metric, (value, (ci_lower, ci_upper)) in metrics.items():
                results.append({
                    "Field": field,
                    "Model": model,
                    "Metric": metric,
                    "Value": value,
                    "CI Lower": ci_lower,
                    "CI Upper": ci_upper
                })
    
    return pd.DataFrame(results)

def print_results_with_totals(results_df, ci_results):
    """Выводит результаты с итоговыми значениями по каждой библиотеке"""
    
    # Вывод детализированных результатов по классам
    print("\nДетализированные результаты по классам:")
    print(results_df.to_markdown(index=False))
    
    # Расчет итоговых значений (среднее взвешенное по количеству примеров)
    metrics = ["Word Accuracy", "Precision", "Recall", "F1-Score"]
    libraries = results_df['Model'].unique()
    
    # Подготовка данных для итогов
    totals = []
    for lib in libraries:
        lib_data = results_df[results_df['Model'] == lib]
        total_count = len(lib_data)  # Или используйте реальное количество примеров, если доступно
        
        # Рассчитываем среднее по всем классам
        lib_totals = {
            'Model': f"{lib} (Итог)",
            'Field': 'Все поля'
        }
        
        for metric in metrics:
            lib_totals[metric] = round(lib_data[metric].mean(), 2)
            
        totals.append(lib_totals)
    
    # Создаем DataFrame с итогами
    totals_df = pd.DataFrame(totals)
    
    # Выводим итоговые значения
    print("\nИтоговые значения по библиотекам:")
    print(totals_df.to_markdown(index=False)) 

    # Для результатов с доверительными интервалами
    if ci_results is not None:
        print("\nРезультаты с доверительными интервалами (95%):")
        print(ci_results.to_markdown())
        
        # Итоги для доверительных интервалов
        ci_totals = []
        for lib in libraries:
            lib_data = ci_results[ci_results['Model'] == lib]
            
            # Формируем строку с итогами
            ci_totals.append({
                'Model': f"{lib} (Итог)",
                'Field': 'Все поля',
                'Word Accuracy': format_metric_with_ci(lib_data, 'Word Accuracy'),
                'Precision': format_metric_with_ci(lib_data, 'Precision'),
                'Recall': format_metric_with_ci(lib_data, 'Recall'),
                'F1-Score': format_metric_with_ci(lib_data, 'F1-Score')
            })
        
        ci_totals_df = pd.DataFrame(ci_totals)
        print("\nИтоговые значения с доверительными интервалами:")
        print(ci_totals_df.to_markdown(index=False))
        
        # Записываем результаты в файл
        output_dir = Path(OUTPUT_DIR)
        with open(output_dir / 'ci_metrics_ocr_total_ci.txt', 'w') as f:
            f.write("Результаты с доверительными интервалами (95%):\n\n")
            f.write(ci_totals_df.to_markdown()) 

def format_metric_with_ci(df, metric):
    """Форматирует метрику с доверительным интервалом"""
    values = df[df['Metric'] == metric]
    if len(values) == 0:
        return "N/A"
    
    mean_val = values['Value'].mean()
    ci_lower = values['CI Lower'].mean()
    ci_upper = values['CI Upper'].mean()
    return f"{mean_val:.2f} [{ci_lower:.2f}, {ci_upper:.2f}]"

def visualise_four_metrics(ci_results, field_types):            
    # Визуализация с интервалами (исправленная версия)
    #metrics = ["Word Accuracy", "F1-Score"]
    metrics = ["Word Accuracy", "Precision", "Recall", "F1-Score"]
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))    # на четыре метрики
    #fig, axes = plt.subplots(1, 2, figsize=(16, 12))    # для двух метрик

    for i, metric in enumerate(metrics):
        ax = axes[i//2, i%2]
        
        # Подготовка данных
        plot_data = ci_results[ci_results["Metric"] == metric]
        x = np.arange(len(field_types))
        width = 0.35
        
        # Построение для каждой модели
        for j, model in enumerate(["TrOCR", "PaddleOCR", "Ensemble"]):
            model_data = plot_data[plot_data["Model"] == model]
            values = model_data["Value"].values
            ci_lower = model_data["CI Lower"].values
            ci_upper = model_data["CI Upper"].values
            
            # Корректировка интервалов
            yerr_lower = values - np.maximum(ci_lower, 0)
            yerr_upper = np.minimum(ci_upper, 1) - values
            
            # Проверка на отрицательные значения
            yerr_lower = np.clip(yerr_lower, 0, None)
            yerr_upper = np.clip(yerr_upper, 0, None)
            
            # Проверка на NaN значения
            if np.isnan(yerr_lower).any() or np.isnan(yerr_upper).any():
                print(f"Предупреждение: NaN значения обнаружены для {metric} ({model})")
                yerr_lower = np.nan_to_num(yerr_lower)
                yerr_upper = np.nan_to_num(yerr_upper)
            
            # Построение графика
            bars = ax.bar(x + j*width, values, width, label=model)
            ax.errorbar(
                x + j*width, 
                values,
                yerr=[yerr_lower, yerr_upper],
                fmt='none', 
                color='black', 
                capsize=5,
                elinewidth=1.5
            )
            
        ax.set_title(f"{metric} с 95% ДИ", fontsize=12)
        ax.set_xticks(x + width/2)
        ax.set_xticklabels(field_types.keys(), fontsize=10)
        ax.set_ylim(0, 1.15)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig("ocr_metrics_with_ci.png", dpi=300, bbox_inches='tight')


def visualise_two_metrics(ci_results, field_types):
    """Визуализация Word Accuracy и F1-Score с горизонтальными подписями"""
    metrics = ["Word Accuracy", "F1-Score"]
    
    # Создаем фигуру с двумя графиками
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Цветовая схема для моделей
    model_colors = {
        "TrOCR": "#1f77b4",
        "PaddleOCR": "#ff7f0e", 
        "Ensemble": "#2ca02c"
    }
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # Фильтруем данные по метрике
        plot_data = ci_results[ci_results["Metric"] == metric]
        x = np.arange(len(field_types))
        width = 0.25  # Ширина столбцов
        
        # Строим для каждой модели
        for j, model in enumerate(["TrOCR", "PaddleOCR", "Ensemble"]):
            model_data = plot_data[plot_data["Model"] == model]
            
            if model_data.empty:
                continue
                
            values = model_data["Value"].values
            ci_lower = model_data["CI Lower"].values
            ci_upper = model_data["CI Upper"].values
            
            # Рассчет ошибок
            yerr = [
                values - np.maximum(ci_lower, 0),
                np.minimum(ci_upper, 1) - values
            ]
            
            # Столбцы с цветами из схемы
            bars = ax.bar(
                x + j*width,
                values,
                width,
                label=model,
                color=model_colors[model],
                alpha=0.8,
                edgecolor='white',
                linewidth=0.5
            )
            
            # Доверительные интервалы
            ax.errorbar(
                x + j*width,
                values,
                yerr=yerr,
                fmt='none',
                color='black',
                capsize=3,
                elinewidth=1
            )
            
            # Подписи значений
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width()/2,
                    height + 0.01,
                    f'{height:.2f}',
                    ha='center',
                    va='bottom',
                    fontsize=9
                )
        
        # Оформление графиков
        ax.set_title(metric, fontsize=14, pad=12)
        ax.set_xticks(x + width)
        ax.set_xticklabels(
            field_types.keys(),
            fontsize=11,
            rotation=0  # Горизонтальные подписи
        )
        ax.set_ylim(0, 1.15)
        ax.grid(axis='y', linestyle=':', alpha=0.4)
        ax.legend(
            loc='upper left',
            fontsize=10,
            framealpha=0.9
        )
        
        # Подписи осей
        ax.set_ylabel('Значение метрики', fontsize=11)
        ax.set_xlabel('Тип поля', fontsize=11)

    # Общее оформление
    plt.tight_layout(pad=2.0)
    plt.savefig(
        "ocr_metrics_comparison.png",
        dpi=300,
        bbox_inches='tight',
        facecolor='white'
    )
    plt.close()


def visualize_four_metrics_aggregated(ci_results):
    """
    Визуализация 4 метрик в ряд с общими значениями для трёх моделей (без разбивки по полям)
    
    Параметры:
    ci_results - DataFrame с результатами (должен содержать колонки: Metric, Model, Value, CI Lower, CI Upper)
    """
    metrics = ["Word Accuracy", "Precision", "Recall", "F1-Score"]
    
    # Настройка стиля (используем доступный стиль)
    plt.style.use('ggplot')  # Альтернатива: 'seaborn-v0_8', 'seaborn', 'seaborn-poster'
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 7))
    
    # Цвета для моделей
    model_colors = {
        "TrOCR": "#4C72B0", 
        "PaddleOCR": "#DD8452",
        "Ensemble": "#55A868"
    }
    
    # Позиции для столбцов (только три модели)
    x = np.arange(3)
    bar_width = 0.8  # Увеличиваем ширину столбцов
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        metric_data = ci_results[ci_results["Metric"] == metric]
        
        # Агрегируем данные по моделям (среднее по всем полям)
        aggregated = metric_data.groupby('Model').agg({
            'Value': 'mean',
            'CI Lower': 'mean',
            'CI Upper': 'mean'
        }).reindex(["TrOCR", "PaddleOCR", "Ensemble"])  # Гарантируем порядок
        
        values = aggregated['Value'].values
        ci_lower = aggregated['CI Lower'].values
        ci_upper = aggregated['CI Upper'].values
        
        # Строим столбцы
        bars = ax.bar(
            x,
            values,
            width=bar_width,  # Явно указываем ширину
            color=[model_colors[model] for model in aggregated.index],
            alpha=0.8,
            edgecolor='white',
            linewidth=1
        )
        
        # Добавляем доверительные интервалы
        ax.errorbar(
            x,
            values,
            yerr=[values - ci_lower, ci_upper - values],
            fmt='none',
            color='black',
            capsize=8,
            elinewidth=1.5
        )
        
        # Подписи значений на столбцах
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2,
                height + 0.1,
                f'{height:.2f}',
                ha='center',
                va='bottom',
                fontsize=10,
                fontweight='bold'
            )
        
        # Настройка осей и оформления
        ax.set_title(metric, fontsize=14, pad=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(
            aggregated.index,
            fontsize=12,
            rotation=0,
            color='black'  # Черные подписи по оси X
        )

        # Делаем черными подписи значений по оси Y
        ax.tick_params(axis='y', colors='black')

        ax.set_ylim(0, 1.15)
        ax.grid(axis='y', linestyle=':', alpha=0.4)
        
        # Убираем рамку
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Делаем черными оставшиеся рамки (левую и нижнюю)
        # for spine in ['left', 'bottom']:
        #     ax.spines[spine].set_color('black')
        
        # Убираем промежутки между столбцами
        ax.set_xlim(-0.5, 2.5)  # Устанавливаем границы, чтобы столбцы занимали всё пространство
    
    # Общее оформление
    plt.suptitle("Сравнение моделей OCR по ключевым метрикам", fontsize=16, y=1.05)
    plt.tight_layout(pad=2.0)
    plt.savefig(
        "ocr_models_comparison.png",
        dpi=300,
        bbox_inches='tight',
        facecolor='white'
    )
    plt.close()


def main():
    # Загрузка данных
    df = pd.read_csv(RESULTS_CSV)
    field_types = {
        "CardHolder": {"is_char_level": False},
        "CardNumber": {"is_char_level": True},
        "DateExpired": {"is_char_level": True}
    }
    results = []

    for field, params in field_types.items():
        subset = df[df["field_type"] == field]
        
        # KerasOCR
        kerasocr_metrics = calculate_metrics(
            subset["true_text"].tolist(),
            subset["kerasocr_text"].tolist(),
            **params
        )
        kerasocr_metrics.update({"Model": "KerasOCR", "Field": field})

        # EasyOCR
        easyocr_metrics = calculate_metrics(
            subset["true_text"].tolist(),
            subset["easyocr_text"].tolist(),
            **params
        )
        easyocr_metrics.update({"Model": "EasyOCR", "Field": field})
        
        # TrOCR
        trocr_metrics = calculate_metrics(
            subset["true_text"].tolist(),
            subset["trocr_text"].tolist(),
            **params
        )
        trocr_metrics.update({"Model": "TrOCR", "Field": field})

        # PaddleOCR
        paddleocr_metrics = calculate_metrics(
            subset["true_text"].tolist(),
            subset["paddleocr_text"].tolist(),
            **params
        )
        paddleocr_metrics.update({"Model": "PaddleOCR", "Field": field})

        # Ensemble
        ensemble_metrics = calculate_metrics(
            subset["true_text"].tolist(),
            subset["ensemble_text"].tolist(),
            **params
        )
        ensemble_metrics.update({"Model": "Ensemble", "Field": field})
        
        # Модифицированный вывод результатов
        
        results.extend([kerasocr_metrics, easyocr_metrics, trocr_metrics, paddleocr_metrics, ensemble_metrics])
    
    results_df = pd.DataFrame(results)
    new_order = ['Field', 'Model', 'Word Accuracy', 'Precision', 'Recall', 'F1-Score']
    results_df = results_df[new_order]

    # Запуск анализа с доверительными интервалами
    ci_results = analyze_with_scipy_ci(df, field_types)

    # Красивое отображение результатов
    def format_ci(row):
        return f"{row['Value']:.2f} [{row['CI Lower']:.2f}, {row['CI Upper']:.2f}]"

    ci_results["Value (95% CI)"] = ci_results.apply(format_ci, axis=1)
    pivot_results = ci_results.pivot_table(
        index=["Field", "Model"],
        columns="Metric",
        values="Value (95% CI)",
        aggfunc="first"
    )

    # Выводим результаты с итогами
    print_results_with_totals(results_df, ci_results)

    print(pivot_results.to_markdown()) 
    # Записываем результаты в файл
    output_dir = Path(OUTPUT_DIR)
    with open(output_dir / 'ci_metrics_3_ocr.txt', 'w') as f:
        f.write("Результаты с доверительными интервалами (95%):\n\n")
        f.write(pivot_results.to_markdown()) 

    #visualise_two_metrics(ci_results, field_types) 
    visualize_four_metrics_aggregated(ci_results) 

if __name__== "__main__":
    main()