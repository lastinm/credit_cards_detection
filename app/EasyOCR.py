import sys, os, io, logging
import cv2
import easyocr
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import statistics

from constants import ARTEFACTS_DIR
import common_utils as utils


def recognize_images_in_directory(posix_img_path, languages=['en', 'ru'], gpu=False):
    """
    Распознает текст на изображениях с учетом класса из имени файла
    
    Args:
        img_path: posix путь к изображению
        languages: список языков для распознавания
        gpu: использовать ли GPU
    """
    # Инициализация EasyOCR
    reader = easyocr.Reader(languages, gpu=gpu)
        
    try:
        img_path = str(posix_img_path.absolute())
        # Извлекаем класс из имени файла (первый символ перед '_')
        class_id = posix_img_path.name.split('_')[0]
        
        # Настройки распознавания в зависимости от класса
        allowlist = None
        if class_id == '2':  # Только цифры
            allowlist = '0123456789'
            print("Режим распознавания: только цифры")
        elif class_id == '3':  # Цифры и символ '/'
            allowlist = '0123456789/'
            print("Режим распознавания: цифры и символ /")
        
        # Загрузка и предобработка изображения
        img = cv2.imread(img_path)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Увеличьте маленький текст:
        img = cv2.resize(img, None, fx=2, fy=1, interpolation=cv2.INTER_CUBIC)

        # # Улучшение контраста (CLAHE)
        #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(6,6))
        #img = clahe.apply(img)
        
        # # Бинаризация (адаптивный метод)
        #img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        #                           cv2.THRESH_BINARY_INV, 11, 2)

        #img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]  # Бинаризация
        
        # Увеличение резкости (для мелкого текста)
        #kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        #img = cv2.filter2D(img, -1, kernel)
        
        # Распознавание с улучшенными параметрами
        results = reader.readtext(img, 
                                allowlist=allowlist,
                                text_threshold=0.3,
                                link_threshold=0.3,
                                width_ths=0.3,
                                slope_ths=0.2,
                                ycenter_ths=0.3)
        
        #print(f"recognize_images_in_directory возвращает {img_path}, {results}, {class_id}, {img}")
        return img_path, results, class_id, img
    
    except Exception as e:
        print(f"Ошибка при обработке файла {posix_img_path}: {str(e)}")


def visualize_enhanced_results(orig_path, results, class_id, processed_img):
    """
    Визуализация с сравнением оригинального и обработанного изображения
    
    Args:
        orig_path: Путь к изображению (str или Path)
        results: Список результатов [(bbox, text, prob), ...]
        class_id: ID класса
        processed_img: Обработанное изображение (numpy array)
    """
    # Загрузка изображения
    orig_img = cv2.cvtColor(cv2.imread(str(orig_path)), cv2.COLOR_BGR2RGB)
    
    # Собираем результаты
    result_text = []
    confidences = []
    for i, (_, text, prob) in enumerate(results):
        #print(f"{i+1}. {text} (точность: {prob:.2f})")
        result_text.append(text)
        confidences.append(prob)

    # Собираем полный текст
    full_text = ''.join(result_text)

    # Теперь высчитаем среднее значение
    if confidences:  # Проверяем, что массив не пустой
        average_confidence = statistics.mean(confidences)
    
    # Вывод результатов
    print("Распознанный текст:")
    print(f"'{full_text}' (уверенность: {average_confidence:.3f})")

    # Создаем фигуру только в интерактивном режиме
    cv2.imshow("Original", cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR))
    cv2.imshow("Processed", processed_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    image_files = utils.get_list_of_images()

    if not image_files:
        print(f"Нет изображений с корректным форматом имени.")
        return

    for image_file, class_name, confidence in image_files:
        print(f"Передаем в EasyOCR файл: {image_file.name}")

        img_path, results, class_id, processed_img = recognize_images_in_directory(image_file, languages=['en'], gpu=False)

        visualize_enhanced_results(img_path, results, class_id, processed_img)


if __name__ == "__main__":
    main()

