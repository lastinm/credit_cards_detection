# app/common_utils.py
import os, shutil, io, imghdr, re, tempfile
import logging
import cv2
from PIL import Image


from constants import ARTEFACTS_DIR, CLASS_NAMES

def clean_artefact_dir():
    if os.path.exists('artefacts'):
        shutil.rmtree('artefacts')
    os.makedirs('artefacts')


def delete_old_detections():
    artefacts_dir = 'artefacts'
    
    # Проверяем существование директории
    if not os.path.exists(artefacts_dir):
        os.makedirs(artefacts_dir)
        return
    
    # Шаблон для поиска файлов: начинаются с цифры и _
    pattern = re.compile(r'^\d+_.*')
    
    # Удаляем только соответствующие файлы
    for filename in os.listdir(artefacts_dir):
        if pattern.match(filename):
            file_path = os.path.join(artefacts_dir, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Ошибка при удалении {file_path}: {e}")


def get_image_from_artefacts():  
    # Проверяем существование каталога
    if not os.path.exists('artefacts'):
        raise FileNotFoundError(f"Каталог {'artefacts'} не существует")
    
    # Получаем список файлов в каталоге
    files = os.listdir('artefacts')
    
    # Получаем список файлов с полными путями
    image_files = []
    for item in ARTEFACTS_DIR.iterdir():
        if item.is_file() and imghdr.what(item):
            image_files.append(item)
    
    # Проверяем количество найденных изображений
    if len(image_files) == 0:
        raise ValueError("В каталоге нет файлов изображений")
    elif len(image_files) > 1:
        raise ValueError(f"В каталоге более одного изображения: {image_files}")
    
    # Возвращаем абсолютный путь к файлу
    return str(image_files[0].absolute())


def get_list_of_images():
    # Регулярное выражение для проверки формата имени файла
    pattern = re.compile(r'^(\d+)_(\d+\.\d+)_')

    # Получаем и фильтруем изображения
    image_files = []
    for file in ARTEFACTS_DIR.glob('*.*'):
        logging.info(f"Имя файла: {file.name}")
        if file.is_file() and file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            logging.info(f"Файл существует и явл. изображением: {file.name}")
            match = pattern.match(file.stem)
            if match:
                #class_id = match.group(1)
                class_name = CLASS_NAMES[int(match.group(1))]
                logging.info(f"Класс изображения: {class_name}")
                confidence = match.group(2) #.replace('_', '.')
                #image_files.append((file, class_id, confidence))
                image_files.append((file, class_name, confidence))
    
    logging.info(f"Возвращаем коллекцию изображений: {image_files}")
    return image_files


def save_detect_result(image_path, results, method):
    # Загружаем изображение
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Не удалось загрузить изображение: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

   # Получаем базовое имя файла
    original_name = os.path.splitext(os.path.basename(image_path))[0]
    ext = os.path.splitext(image_path)[1].lower()  # Всегда в нижнем регистре

    delete_old_detections()
    
    if method == 'FASTER-RCNN':
        save_result_faster_rcnn(original_name, ext, image, results)
    elif method == 'YOLO':
        save_result_yolo(original_name, ext, image, results)
    else:
        print("Не указан метод детектирования!")
        return


def save_result_faster_rcnn(original_name, ext, image, results):
    # Сохраняем каждую обнаруженную область
    for i, (box, score, cls) in enumerate(zip(results['boxes'], results['scores'], results['classes'])):
        save_image()


def save_result_yolo(original_name, ext, image, detections):
    # Сохраняем каждую обнаруженную область
    for i, (box, score, cls) in enumerate(zip(detections.xyxy, detections.confidence, detections.class_id)):
        save_image(original_name, ext, image, box, score, cls)


def save_image(original_name, ext, image, box, score, cls):
    x1, y1, x2, y2 = map(int, box)
    cropped = image[y1:y2, x1:x2]
    
    if cropped.size > 0:
        # Формируем имя файла по шаблону: class_score_originalname.ext
        # Округляем score до 2 знаков и заменяем точку на подчеркивание
        score_str = f"{score:.2f}"  #.replace('.', '_')
        filename = f"{cls}_{score_str}_{original_name}{ext}"
        crop_path = os.path.join(str(ARTEFACTS_DIR), filename)
        
        # Проверяем, не существует ли уже файл
        counter = 1
        while os.path.exists(crop_path):
            filename = f"{cls}_{score_str}_{original_name}_{counter}{ext}"
            crop_path = os.path.join(str(ARTEFACTS_DIR), filename)
            counter += 1
        
        # Сохраняем изображение
        Image.fromarray(cropped).save(crop_path)

def prepare_enhanced_results(orig_path, results, class_id, processed_img):
    """
    Подготавливает изображения с результатами для отправки в Telegram бота
    
    Args:
        orig_path (str): Путь к оригинальному изображению
        results (list): Результаты распознавания в формате [(bbox, text, prob), ...]
        class_id (int): ID класса объекта
        processed_img (numpy.ndarray): Обработанное изображение
        
    Returns:
        tuple: (original_image_path, processed_image_path, recognized_texts)
    """
    print("Загружаем оригинальное изображение")
    orig_img = cv2.cvtColor(cv2.imread(orig_path), cv2.COLOR_BGR2RGB)
    
    # Рисуем bounding boxes на оригинальном изображении
    for (bbox, text, prob) in results:
        tl = (int(bbox[0][0]), int(bbox[0][1]))
        br = (int(bbox[2][0]), int(bbox[2][1]))
        
        # Рисуем прямоугольник
        cv2.rectangle(processed_img, tl, br, (255, 0, 0), 2)
        
        # Добавляем текст
        cv2.putText(processed_img, f"{text} ({prob:.2f})", 
                   (tl[0], tl[1]-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    # Создаем временные файлы для изображений
    def save_temp_image(img):
        temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        temp_path = temp_file.name
        cv2.imwrite(temp_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        temp_file.close()
        return temp_path

    orig_temp_path = save_temp_image(orig_img)
    processed_temp_path = save_temp_image(processed_img)
    
    # Формируем текстовые результаты
    recognized_texts = [f"{i+1}. {text} (точность: {prob:.2f})" 
                       for i, (_, text, prob) in enumerate(results)]
    
    print(f"{recognized_texts}")    
    return orig_temp_path, processed_temp_path, recognized_texts