import cv2
from ultralytics import YOLO
import supervision as sv
import sys, os

from constants import ARTEFACTS_DIR
import common_utils as utils


def main(image_path):
    # 1. Загрузка модели
    image = cv2.imread(image_path)
    model = YOLO('/home/lastinm/PROJECTS/credit_cards_detection/train/YOLOv12/runs/detect/train4/weights/best.pt')

    # 2. Детекция на изображении
    results = model(image, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results).with_nms()

    # 3. Аннотирование изображения
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    annotated_image = image.copy()
    annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections)
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)

    # 4. Если вызов из консоли, то вывод результатов в консоль
    if __name__ == "__main__":
        class_names = ["Cardholder", "CardNumber", "DateExpired"]
        for i, (xyxy, conf, cls_id) in enumerate(zip(detections.xyxy, detections.confidence, detections.class_id)):
            print(f"Объект {i+1}: {class_names[cls_id]}({cls_id}). Уверенность: {conf:.2%}")

        # 5. Отображение результата
        cv2.imshow("Detection Results", annotated_image)
        print("Нажмите любую клавишу для закрытия окна...")
        cv2.waitKey(0)  # Ожидание нажатия любой клавиши
        cv2.destroyAllWindows()

    # 4. Сохраняем изображения продетекировнных областей
    utils.save_detect_result(image_path, detections, 'YOLO')

    # 5. Визуализация
    #visualize_results_matplotlib(image_path, results, class_names)


if __name__ == "__main__":
    # image_path = sys.argv[1]  # Первый аргумент — путь к исходному изображению
    # main(sys.argv[1])
    #img_path = '/home/lastinm/PROJECTS/DATA/DATA YOLOv12/test/images/card26_jpg.rf.9defe96ac5b853d3f84650575dc86d39.jpg'   # Золотая
    img_path = '/home/lastinm/PROJECTS/FINAL/test_img/1742460932138_resized_jpg.rf.0ee03bf49164c59d1ff4b929d05132cf.jpg'    # Сбер
    main(img_path)