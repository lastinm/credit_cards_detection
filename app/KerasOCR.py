import keras_ocr
import numpy as np
import cv2
import matplotlib.pyplot as plt

from constants import ARTEFACTS_DIR
import common_utils as utils


def recognize_with_confidence(image_path):
    """Основная функция распознавания с правильным handling confidence"""
    pipeline = keras_ocr.pipeline.Pipeline()
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    predictions = pipeline.recognize([img])[0]
 
    result_text = []
    for pred in predictions:
        result_text.append(pred[0])
          
    full_text = ' '.join(result_text)
           
    return full_text


def main():

    image_files = utils.get_list_of_images()

    if not image_files:
        print(f"Нет изображений с корректным форматом имени.")
        return

    for image_file, class_name, confidence in image_files:
        full_text = recognize_with_confidence(image_file)
    
        print("\nОбъединенный текст:")
        print(full_text)


# Пример использования
if __name__ == "__main__":
    main()