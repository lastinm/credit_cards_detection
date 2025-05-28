from paddleocr import TextRecognition
from PIL import Image
import cv2
#from torchvision.transforms import functional as F
import numpy as np
import statistics
import os
import tempfile


from constants import ARTEFACTS_DIR
import common_utils as utils


def recognize_images_in_directory(posix_img_path):
    """
    Распознает текст на изображениях с учетом класса из имени файла
    Args:
        img_path: posix путь к изображению
        processor: 
        model: 
    """
    # Инициализация PaddleOCR
    paddleocr_reader = TextRecognition()

    try:
        img_path = str(posix_img_path.absolute())
        # Извлекаем класс из имени файла (первый символ перед '_')
        class_id = posix_img_path.name.split('_')[0]
              
        # Загрузка и предобработка изображения
        cv_image = cv2.imread(img_path)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

        # Увеличим маленький текст:
        cv_image = cv2.resize(cv_image, None, fx=1.5, fy=1, interpolation=cv2.INTER_CUBIC)
      
        # Создаем временный файл
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            temp_path = tmp_file.name        
        # Сохраняем изображение во временный файл
        cv2.imwrite(temp_path, cv_image)

        results = paddleocr_reader.predict(input=img_path)
        
        # Удаляем временный файл
        os.unlink(temp_path)

        return results[0]['rec_text'], results[0]['rec_score']
    
    except Exception as e:
        print(f"Ошибка при обработке файла {posix_img_path}: {str(e)}")


def main():

    image_files = utils.get_list_of_images()

    if not image_files:
        print(f"Нет изображений с корректным форматом имени.")
        return

    for image_file, class_name, confidence in image_files:
        print(f"Передаем в PaddleOCR файл: {image_file.name}")

        # Результат
        full_text, confidences = recognize_images_in_directory(image_file)

        # Проверка
        print(f"Распознанный текст: '{full_text}'")
        print(f"'{full_text}', (уверенность: {confidences:.3f})")

        #visualize_enhanced_results(img_path, results, class_id, processed_img)
        

if __name__ == "__main__":
    main()