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
        if normalize_text(true) == normalize_text(pred)
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
        "Word Accuracy": word_accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1
    }

# Загрузка данных
df = pd.read_csv("results.csv")
field_types = {
    "CardHolder": {"is_char_level": False},
    "CardNumber": {"is_char_level": True},
    "DateExpired": {"is_char_level": True}
}
results = []

for field, params in field_types.items():
    subset = df[df["field_type"] == field]
    
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
    
    results.extend([easyocr_metrics, trocr_metrics])

def calculate_metrics_with_ci(true_texts, pred_texts, is_char_level=False, confidence=0.95, n_bootstrap=1000):
    """Расчет метрик с корректными доверительными интервалами"""
    n = len(true_texts)
    
    # Word Accuracy (биномиальный интервал)
    correct = sum(1 for true, pred in zip(true_texts, pred_texts) 
                if normalize_text(true) == normalize_text(pred))
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
        "Word Accuracy": (word_acc, word_acc_ci),
        "Precision": (precision, precision_ci),
        "Recall": (recall, recall_ci),
        "F1-Score": (f1, f1_ci)
    }

# Анализ с доверительными интервалами
def analyze_with_scipy_ci(df, field_types, confidence=0.95):
    results = []
    
    for field, params in field_types.items():
        subset = df[df["field_type"] == field]
        true_texts = subset["true_text"].tolist()
        
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
        
        # Сохранение результатов
        for model, metrics in [("EasyOCR", easyocr_metrics), ("TrOCR", trocr_metrics)]:
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

# Результаты
results_df = pd.DataFrame(results)
new_order = ['Field', 'Model', 'Word Accuracy', 'Precision', 'Recall', 'F1-Score']
results_df = results_df[new_order]
print(results_df.to_markdown(index=False))

metrics = ["Word Accuracy", "Precision", "Recall", "F1-Score"]
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for i, metric in enumerate(metrics):
    ax = axes[i//2, i%2]
    results_df.pivot(index="Field", columns="Model", values=metric).plot(kind="bar", ax=ax)
    ax.set_title(metric)
    ax.set_ylim(0, 1.1)
plt.tight_layout()
plt.savefig("ocr_metrics_comparison.png")  # Сохраняем график

# Запуск анализа
ci_results = analyze_with_scipy_ci(df, field_types)

# Красивое отображение результатов
def format_ci(row):
    return f"{row['Value']:.3f} [{row['CI Lower']:.3f}, {row['CI Upper']:.3f}]"

ci_results["Value (95% CI)"] = ci_results.apply(format_ci, axis=1)
pivot_results = ci_results.pivot_table(
    index=["Field", "Model"],
    columns="Metric",
    values="Value (95% CI)",
    aggfunc="first"
)

# Применяем новый порядок (с проверкой наличия столбцов)
pivot_results = pivot_results.reindex(
    columns=[col for col in new_order if col in pivot_results.columns]
)

print("\nРезультаты с доверительными интервалами (95%):")
print(pivot_results.to_markdown())

# Визуализация с интервалами (исправленная версия)
metrics = ["Word Accuracy", "Precision", "Recall", "F1-Score"]
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

for i, metric in enumerate(metrics):
    ax = axes[i//2, i%2]
    
    # Подготовка данных
    plot_data = ci_results[ci_results["Metric"] == metric]
    x = np.arange(len(field_types))
    width = 0.35
    
    # Построение для каждой модели
    for j, model in enumerate(["EasyOCR", "TrOCR"]):
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