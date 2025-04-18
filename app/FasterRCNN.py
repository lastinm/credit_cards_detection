import sys, os
import torch
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
import cv2
import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches

from constants import ARTEFACTS_DIR
import common_utils as utils


def load_model(checkpoint_path, num_classes=4):
    # 1. Инициализация модели с предобученными весами backbone
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=weights)
    
    # 2. Замена box_predictor для своего числа классов
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # 3. Загрузка checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 4. Извлекаем только веса модели (игнорируем optimizer и epoch)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # 5. Аккуратная загрузка весов
    model.load_state_dict(state_dict, strict=False)  # strict=False пропустит отсутствующие ключи
    
    # 6. Перевод в режим оценки
    model.eval()
    
    return model


def detect_objects(model, image_path, confidence_threshold=0.7):
    """
    Детекция объектов на изображении с помощью загруженной модели
    Args:
        model: Загруженная модель Faster R-CNN
        image_path: Путь к изображению
        confidence_threshold: Порог уверенности для отображения (0.0-1.0)
    Returns:
        Словарь с результатами детекции
    """
    # 1. Загрузка и преобразование изображения
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # OpenCV -> RGB
    image_tensor = F.to_tensor(image)  # Конвертация в тензор [0-1]

    # 2. Добавление batch dimension и перемещение на устройство (GPU/CPU)
    image_tensor = image_tensor.unsqueeze(0).to(next(model.parameters()).device)
    
    # 3. Запуск модели (без вычисления градиентов)
    with torch.no_grad():
        predictions = model(image_tensor)
    
    # 4. Фильтрация результатов по порогу уверенности
    pred_boxes = predictions[0]['boxes'].cpu().numpy()
    pred_scores = predictions[0]['scores'].cpu().numpy()
    pred_classes = predictions[0]['labels'].cpu().numpy()
    
    keep = pred_scores >= confidence_threshold
    results = {
        'boxes': pred_boxes[keep],
        'scores': pred_scores[keep],
        'classes': pred_classes[keep]
    }
    
    return results

def visualize_results_matplotlib(image_path, results, class_names):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)
    
    colors = ['red', 'green', 'blue', 'yellow', 'purple']
    
    for box, score, cls in zip(results['boxes'], results['scores'], results['classes']):
        x1, y1, x2, y2 = map(int, box)
        rect = patches.Rectangle(
            (x1, y1), x2-x1, y2-y1,
            linewidth=2, edgecolor=colors[cls % len(colors)], facecolor='none'
        )
        ax.add_patch(rect)
        
        plt.text(
            x1, y1-10, f"{class_names[cls]}: {score:.2f}",
            color=colors[cls % len(colors)], fontsize=10,
            bbox=dict(facecolor='white', alpha=0.8)
        )
    
    plt.axis('off')
    plt.show()


def main(image_path):
    # 1. Загрузка модели
    model = load_model(
        checkpoint_path='models/faster_rcnn', #/home/lastinm/PROJECTS/FINAL/
        num_classes=4  # Укажите реальное число ваших классов (3 класса + фон)
        )

    # 2. Детекция на изображении
    results = detect_objects(
        model=model,
        image_path=image_path,
        confidence_threshold=0.5  # Можно регулировать
        )

    if __name__ == "__main__":
    # 3. Вывод результатов в консоль
        class_names = ["Background", "Cardholder", "CardNumber", "DateExpired"]  # Пример
        print(f"Найдено объектов: {len(results['boxes'])}")
        for i, (box, score, cls) in enumerate(zip(results['boxes'], results['scores'], results['classes'])):
            print(f"Объект {i+1}: {class_names[cls]} (уверенность: {score:.2f})")
            print(f"Координаты: x1={box[0]:.1f}, y1={box[1]:.1f}, x2={box[2]:.1f}, y2={box[3]:.1f}")

    # 4. Сохраняем изображения продетекировнных областей
    utils.save_detect_result(image_path, results, "FASTER-RCNN")

    # 5. Визуализация
    #visualize_results_matplotlib(image_path, results, class_names)


if __name__ == "__main__":
    image_path = sys.argv[1]  # Первый аргумент — путь к исходному изображению
    main(sys.argv[1])

