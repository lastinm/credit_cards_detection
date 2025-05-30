import keras_ocr
from tensorflow.keras import layers, models
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tempfile 

from constants import ARTEFACTS_DIR
import common_utils as utils


def get_tmp_image_file(posix_img_path, predictions):
    # Создаем временный файл
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
        temp_path = tmp_file.name        

    # Визуализация с указанием класса
    img_path = str(posix_img_path.absolute())
    img = keras_ocr.tools.read(img_path)
    keras_ocr.tools.drawAnnotations(img, predictions[0])
    plt.imshow(img)
    plt.axis('off')  # Отключаем оси
    plt.savefig(temp_path, bbox_inches='tight', pad_inches=0)
    plt.close()  # Закрываем фигуру чтобы освободить память
    print(f"Временный файл: {temp_path}.")

    return temp_path


def get_sorted_predictions(predictions):
    """Сортировка предсказаний по координатам (слева направо, сверху вниз)"""
    # Вычисляем средние Y-координаты для строк
    line_heights = []
    for text, box in predictions:
        y_coords = [point[1] for point in box]
        line_heights.append(np.mean(y_coords))
    
    # Группируем по строкам (с учетом возможного наклона)
    lines = {}
    for i, (text, box) in enumerate(predictions):
        y_mean = line_heights[i]
        found_line = False
        for line_y in lines.keys():
            if abs(y_mean - line_y) < 20:  # Пороговое значение для строк
                lines[line_y].append((box[0][0], text))  # (X-координата, текст)
                found_line = True
                break
        if not found_line:
            lines[y_mean] = [(box[0][0], text)]
    
    # Сортируем строки сверху вниз
    sorted_lines = sorted(lines.items(), key=lambda x: x[0])
    
    # Сортируем слова в каждой строке слева направо
    full_text = []
    for line_y, words in sorted_lines:
        words_sorted = sorted(words, key=lambda x: x[0])
        full_text.extend([word[1] for word in words_sorted])
    
    return ' '.join(full_text)

def recognize_with_confidence(image_path):
    """Основная функция распознавания с правильным handling confidence"""
    pipeline = keras_ocr.pipeline.Pipeline()
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    predictions = pipeline.recognize([img])
 
    correct_order_text = get_sorted_predictions(predictions[0])
    # result_text = []
    # for pred in predictions:
    #     result_text.append(pred[0])
          
    # full_text = ' '.join(result_text)
           
    return correct_order_text, predictions


def main():

    image_files = utils.get_list_of_images()

    if not image_files:
        print(f"Нет изображений с корректным форматом имени.")
        return

    for image_file, class_name, confidence in image_files:
        full_text, predictions = recognize_with_confidence(image_file)
    
        print("\nОбъединенный текст:")
        print(full_text)


# Пример использования
if __name__ == "__main__":
    main()