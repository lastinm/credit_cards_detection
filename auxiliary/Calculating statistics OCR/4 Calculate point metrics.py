# Подсчитывем точечные оценки
# Word Accuracy (полное совпадение строк), Precision, Recall, F1-Score
# Сохраняем график ocr_metrics_comparison.png 

import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt


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

# Результаты
results_df = pd.DataFrame(results)
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

