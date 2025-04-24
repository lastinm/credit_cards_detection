from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch
from PIL import Image
import cv2
import numpy as np


from constants import ARTEFACTS_DIR
import common_utils as utils


# Извлечение уверенности
def get_text_with_confidence(outputs, processor):
        
    # Получаем все токены (игнорируя <s> и </s>)
    tokens = outputs.sequences[0][1:-1]
    
    # Собираем результаты
    result_text = []
    confidences = []
    
    # Обрабатываем каждый токен
    for i, token in enumerate(tokens):
        # Получаем текстовое представление токена
        token_text = processor.decode([token], skip_special_tokens=True)
        
        # Получаем уверенность для этого токена
        confidence = torch.softmax(outputs.scores[i], dim=-1)[0, token].item()
        
        # Для каждого символа в декодированном токене добавляем уверенность
        for char in token_text:
            result_text.append(char)
            confidences.append(confidence)
    
    # Собираем полный текст
    full_text = ''.join(result_text)
        
    return full_text, confidences


def recognize_images_in_directory(posix_img_path):
    """
    Распознает текст на изображениях с учетом класса из имени файла
    Args:
        img_path: posix путь к изображению
        processor: 
        model: 
    """
    # Загрузка модели и процессора
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed") 

    try:
        img_path = str(posix_img_path.absolute())
        # Извлекаем класс из имени файла (первый символ перед '_')
        class_id = posix_img_path.name.split('_')[0]
              
        # Загрузка и предобработка изображения
        cv_image = cv2.imread(img_path)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

        # Увеличьте маленький текст:
        cv_image = cv2.resize(cv_image, None, fx=1.25, fy=1, interpolation=cv2.INTER_CUBIC)

        pil_image = Image.fromarray(cv_image)
        
        # Предобработка и распознавание
        pixel_values = processor(pil_image, return_tensors="pt").pixel_values

        # Генерация текста
        outputs = model.generate(
            pixel_values,
            output_scores=True,
            return_dict_in_generate=True,
            max_length=20)

        return outputs, processor
    
    except Exception as e:
        print(f"Ошибка при обработке файла {posix_img_path}: {str(e)}")


def main():

    image_files = utils.get_list_of_images()

    if not image_files:
        print(f"Нет изображений с корректным форматом имени.")
        return

    for image_file, class_name, confidence in image_files:
        print(f"Передаем в TrOCR файл: {image_file.name}")

        # Результат
        outputs, processor = recognize_images_in_directory(image_file)
        full_text, confidences = get_text_with_confidence(outputs, processor)

        # Проверка
        print(f"Распознанный текст: '{full_text}'")
        print(f"Длина текста: {len(full_text)}, Уверенностей: {len(confidences)}")
        print("Соответствие символов и уверенностей:")
        for char, conf in zip(full_text, confidences):
            print(f"'{char}': {conf:.1%}")

        #visualize_enhanced_results(img_path, results, class_id, processed_img)


if __name__ == "__main__":
    main()